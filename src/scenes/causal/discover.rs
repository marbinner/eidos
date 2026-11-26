//! Discovering a graph: the causal order of several variables found from data alone,
//! one root at a time (DirectLiNGAM), with the true graph beside the one being found.

use super::*;
use crate::math::causal::{regress, residuals, variance, Noise};

pub const DEF: SceneDef = SceneDef {
    id: "discover",
    title: "Discovering a graph",
    group: GROUP,
    blurb: "Find the variable that nothing else causes, take its effect away, and repeat: a causal order from residuals.",
    make: |_| Box::new(Discover::create()),
};

// ----- colours: one meaning each, in the picture and in the key ------------------------

/// A root: a variable placed in the order because nothing left causes it.
const ROOT: Color = CAUSE;
/// A residual: what is left of a variable once a line through the candidate is taken away.
const RESIDUAL: Color = EFFECT;
/// The common cause that is not measured.
const HIDDEN: Color = THIRD;
/// An arrow found, before it is compared with the truth.
const KEPT: Color = pal::TEXT;
/// A found arrow that points the wrong way.
const REVERSED: Color = pal::YELLOW;
/// A variable, as drawn in the graphs before it is placed.
const NODE: Color = Color::hex(0xB4BED3);
/// The bars of the candidates not chosen.
const BAR: Color = Color::hex(0x5A6C93);
/// What random pairing gives: the line the bars are measured against; and the threshold.
const CHANCE: Color = pal::WHITE;

// ----- settings ---------------------------------------------------------------------------

const N_MIN: usize = 20;
const N_MAX: usize = 500;
/// How many new samples the tally runs the whole search on.
const RUNS: usize = 20;
/// The strength below which an arrow is dropped, to begin with.
const THRESHOLD: f32 = 0.2;
const THRESHOLD_MAX: f32 = 1.0;

/// Seconds a step takes to play forwards and backwards, and the pause after each one
/// when they are played through.
const STEP_TIME: f32 = 1.6;
const BACK_TIME: f32 = 0.5;
const HOLD: f32 = 0.9;
/// How quickly what is drawn follows what is so (seconds, for each of two eases).
const EASE: f32 = 0.07;

fn smoothstep(t: f32) -> f32 {
    let t = t.max(0.0).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

// ----- the true graphs ----------------------------------------------------------------------

/// The letters variables are called by: the measured ones A to E, and H for one that
/// is not measured.
const LETTERS: [&str; 6] = ["A", "B", "C", "D", "E", "H"];
/// The same as mathematics, and with a tilde: what is left of the variable once the
/// effects of those already placed are taken away.
const MATH: [&str; 6] = ["$A$", "$B$", "$C$", "$D$", "$E$", "$H$"];
const MATH_LEFT: [&str; 6] = [r"$\tilde{A}$", r"$\tilde{B}$", r"$\tilde{C}$", r"$\tilde{D}$", r"$\tilde{E}$", r"$\tilde{H}$"];

/// A true graph: its variables (the letter, as an index into `LETTERS`, a place in the
/// unit square, and whether it is hidden), the measured ones in alphabetical order; and
/// its arrows (from, to, strength), as indices into the variables.
#[derive(Clone, Copy)]
struct Truth {
    name: &'static str,
    vars: &'static [(usize, [f32; 2], bool)],
    arrows: &'static [(usize, usize, f64)],
}

const TRUTHS: [Truth; 5] = [
    // C → A → D → B, round the square
    Truth { name: "a chain", vars: &[(0, [1.0, 0.0], false), (1, [0.0, 1.0], false), (2, [0.0, 0.0], false), (3, [1.0, 1.0], false)], arrows: &[(2, 0, 0.8), (0, 3, -0.7), (3, 1, 0.8)] },
    // B → A and B → E (a fork at B); E → D ← C (a collider at D)
    Truth {
        name: "a fork and a collider",
        vars: &[(0, [0.0, 0.42], false), (1, [0.5, 0.0], false), (2, [0.18, 1.0], false), (3, [0.82, 1.0], false), (4, [1.0, 0.42], false)],
        arrows: &[(1, 0, 0.8), (1, 4, 0.7), (4, 3, -0.8), (2, 3, 0.7)],
    },
    // D → A → B and D → C → B: two routes from D to B, whose effects add up (0.8 × 0.7
    // and −0.7 × −0.8; with a sign turned they would cancel, and D and B would look unrelated)
    Truth { name: "a diamond", vars: &[(0, [1.0, 0.0], false), (1, [1.0, 1.0], false), (2, [0.0, 1.0], false), (3, [0.0, 0.0], false)], arrows: &[(3, 0, 0.8), (3, 2, -0.7), (0, 1, 0.7), (2, 1, -0.8)] },
    // B → A ← H → D → C, with H not measured
    Truth {
        name: "a hidden common cause",
        vars: &[(0, [1.0, 0.0], false), (1, [0.0, 0.0], false), (2, [0.0, 1.0], false), (3, [1.0, 1.0], false), (5, [0.6, 0.5], true)],
        arrows: &[(1, 0, 0.8), (4, 0, 0.9), (4, 3, 0.9), (3, 2, 0.8)],
    },
    Truth { name: "no arrows", vars: &[(0, [0.0, 0.0], false), (1, [1.0, 0.0], false), (2, [1.0, 1.0], false), (3, [0.0, 1.0], false)], arrows: &[] },
];

impl Truth {
    /// The variables in an order in which every arrow goes forwards.
    fn causal_order(&self) -> Vec<usize> {
        let n = self.vars.len();
        let mut placed = vec![false; n];
        let mut order = Vec::with_capacity(n);
        while order.len() < n {
            let before = order.len();
            for v in 0..n {
                if !placed[v] && self.arrows.iter().all(|&(from, to, _)| to != v || placed.get(from).copied().unwrap_or(true)) {
                    placed[v] = true;
                    order.push(v);
                }
            }
            if order.len() == before {
                // (a cycle: none of the presets has one, but it must not hang)
                order.extend((0..n).filter(|v| !placed[*v]));
                break;
            }
        }
        order
    }

    /// The measured variables: indices into `vars`.
    fn seen(&self) -> Vec<usize> {
        (0..self.vars.len()).filter(|&v| !self.vars[v].2).collect()
    }

    /// How many variables are measured.
    fn count(&self) -> usize {
        self.vars.iter().filter(|v| !v.2).count()
    }

    /// The letter of measured variable `k` (an index among the measured ones).
    fn letter(&self, k: usize) -> usize {
        self.seen().get(k).and_then(|&v| self.vars.get(v)).map_or(0, |v| v.0)
    }

    /// The arrows between measured variables, as indices among the measured ones.
    fn seen_arrows(&self) -> Vec<(usize, usize, f64)> {
        let seen = self.seen();
        let at = |v: usize| seen.iter().position(|&s| s == v);
        self.arrows.iter().filter_map(|&(a, b, s)| Some((at(a)?, at(b)?, s))).collect()
    }

    /// `n` samples of each measured variable: each is its causes added up by the
    /// strengths, plus noise of its own of the given shape (mean 0, variance 1).
    fn sample(&self, noise: Noise, n: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Rng::new(seed);
        let mut all = vec![vec![0.0; n]; self.vars.len()];
        for v in self.causal_order() {
            let mut column: Vec<f64> = (0..n).map(|_| noise.sample(&mut rng)).collect();
            for &(from, to, b) in self.arrows {
                if let (true, Some(cause)) = (to == v, all.get(from)) {
                    for (x, p) in column.iter_mut().zip(cause) {
                        *x += b * p;
                    }
                }
            }
            all[v] = column;
        }
        self.seen().into_iter().map(|v| std::mem::take(&mut all[v])).collect()
    }

    /// Does every arrow between measured variables point forwards in `order` (of the
    /// measured ones)?
    fn agrees(&self, order: &[usize]) -> bool {
        let place = |v: usize| order.iter().position(|&o| o == v);
        self.seen_arrows().iter().all(|&(a, b, _)| matches!((place(a), place(b)), (Some(x), Some(y)) if x < y))
    }

    /// How many orders of the measured variables agree with the arrows between them
    /// (out of all the orders there are).
    fn right_orders(&self) -> (usize, usize) {
        let p = self.count();
        let mut order: Vec<usize> = (0..p).collect();
        let (mut right, mut all) = (0, 0);
        // every order, by Heap's algorithm
        let mut c = vec![0usize; p];
        let mut visit = |o: &[usize]| {
            all += 1;
            right += self.agrees(o) as usize;
        };
        visit(&order);
        let mut i = 1;
        while i < p {
            if c[i] < i {
                order.swap(if i % 2 == 0 { 0 } else { c[i] }, i);
                visit(&order);
                c[i] += 1;
                i = 1;
            } else {
                c[i] = 0;
                i += 1;
            }
        }
        (right, all)
    }
}

// ----- the dependence measure, kept for one candidate ---------------------------------------

/// The width of the bells HSIC compares samples with: the median distance between two
/// of them, among up to 200 evenly picked (as `crate::math::causal::Hsic` has it).
fn typical(x: &[f64]) -> f64 {
    let step = x.len().div_ceil(200).max(1);
    let picked: Vec<f64> = x.iter().step_by(step).copied().collect();
    let mut d: Vec<f64> = Vec::with_capacity(picked.len() * picked.len() / 2);
    for i in 0..picked.len() {
        for j in 0..i {
            d.push((picked[i] - picked[j]).abs());
        }
    }
    d.retain(|v| v.is_finite() && *v > 0.0);
    if d.is_empty() {
        return 1.0;
    }
    let mid = d.len() / 2;
    *d.select_nth_unstable_by(mid, f64::total_cmp).1
}

/// One candidate's table of likenesses for HSIC (the first of `Hsic`'s two tables),
/// centred and kept, so that each of the others' residuals can be measured against it
/// without building it again; with what a random re-pairing gives on average, exactly.
/// The same numbers as `Hsic::new(x, y).value()` (a test says so), for a third of the work.
struct Likeness {
    n: usize,
    k: Vec<f64>,
    trace: f64,
}

impl Likeness {
    fn of(x: &[f64]) -> Likeness {
        let n = x.len();
        let s = typical(x);
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            k[i * n + i] = 1.0;
            for j in 0..i {
                let v = (-0.5 * ((x[i] - x[j]) / s).powi(2)).exp();
                k[i * n + j] = v;
                k[j * n + i] = v;
            }
        }
        // centre: take each row's mean and each column's away, and add the mean of all back
        let mut trace = 0.0;
        if n > 0 {
            let rows: Vec<f64> = (0..n).map(|i| k[i * n..(i + 1) * n].iter().sum::<f64>() / n as f64).collect();
            let all = rows.iter().sum::<f64>() / n as f64;
            for i in 0..n {
                for j in 0..n {
                    k[i * n + j] += all - rows[i] - rows[j];
                }
                trace += k[i * n + i];
            }
        }
        Likeness { n, k, trace }
    }

    /// HSIC between this variable and `y`, and its average over every way of dealing
    /// `y`'s samples out at random among this variable's (what independence gives).
    fn against(&self, y: &[f64]) -> (f64, f64) {
        let n = self.n;
        if n < 2 || y.len() != n {
            return (0.0, 0.0);
        }
        let s = typical(y);
        let (mut value, mut sum) = (self.trace, n as f64);
        for i in 0..n {
            for (j, kij) in self.k[i * n..i * n + i].iter().enumerate() {
                let l = (-0.5 * ((y[i] - y[j]) / s).powi(2)).exp();
                value += 2.0 * kij * l;
                sum += 2.0 * l;
            }
        }
        // a random re-pairing keeps the diagonal (each bell is 1 at its own sample) and
        // spreads the rest evenly; and the centred table's entries add up to nothing
        let off = (sum - n as f64) / (n * (n - 1)) as f64;
        let nn = (n * n) as f64;
        (value / nn, self.trace * (1.0 - off) / nn)
    }
}

/// What is left of `y` once the best line through `x` is taken away; nothing at all
/// (zeros) where all that is left is rounding, as for a variable that is an exact copy
/// of the other.
fn left_over(x: &[f64], y: &[f64]) -> Vec<f64> {
    let r = residuals(&[x], y);
    if variance(&r) <= 1e-20 * variance(y) {
        vec![0.0; r.len()]
    } else {
        r
    }
}

// ----- the search -------------------------------------------------------------------------

/// One round of the search: the variables not yet placed (alphabetically), what is
/// left of each, and for each candidate and each other variable how much the other's
/// residual depends on the candidate (HSIC, and what chance pairing gives).
#[derive(Clone)]
struct Round {
    left: Vec<usize>,
    now: Vec<Vec<f64>>,
    pairs: Vec<Vec<Option<(f64, f64)>>>,
    winner: Option<usize>,
}

impl Round {
    fn new(left: Vec<usize>, now: Vec<Vec<f64>>) -> Round {
        let p = now.len();
        Round { left, now, pairs: vec![vec![None; p]; p], winner: None }
    }

    /// The next pair to measure (candidate, other), candidate by candidate.
    fn next(&self) -> Option<(usize, usize)> {
        self.left.iter().flat_map(|&j| self.left.iter().map(move |&i| (j, i))).find(|&(j, i)| i != j && self.pairs[j][i].is_none())
    }

    fn pair(&self, j: usize, i: usize) -> Option<(f64, f64)> {
        self.pairs.get(j).and_then(|row| row.get(i)).copied().flatten()
    }

    /// A candidate's score, the dependence of all the others' residuals on it added up,
    /// and what chance gives (None until every pair is measured).
    fn score(&self, j: usize) -> Option<(f64, f64)> {
        let mut total = (0.0, 0.0);
        for &i in self.left.iter().filter(|&&i| i != j) {
            let (h, c) = self.pair(j, i)?;
            total = (total.0 + h, total.1 + c);
        }
        Some(total)
    }

    /// The unit the scores are written in: what chance gives a candidate, on average
    /// over the candidates measured so far (1 if none is).
    fn unit(&self) -> f64 {
        let chances: Vec<f64> = self.left.iter().filter_map(|&j| self.score(j)).map(|s| s.1).collect();
        let mean = chances.iter().sum::<f64>() / chances.len().max(1) as f64;
        if mean > 0.0 && mean.is_finite() {
            mean
        } else {
            1.0
        }
    }

    /// The same for one pair.
    fn pair_unit(&self) -> f64 {
        (self.unit() / self.left.len().saturating_sub(1).max(1) as f64).max(1e-300)
    }

    /// The pair of candidate `j` whose residual depends on it most.
    fn worst_of(&self, j: usize) -> Option<usize> {
        self.left.iter().copied().filter(|&i| i != j).filter_map(|i| Some((i, self.pair(j, i)?.0))).max_by(|a, b| a.1.total_cmp(&b.1)).map(|p| p.0)
    }

    /// The candidates in order of their scores, smallest first (only once all are measured).
    fn ranked(&self) -> Vec<(usize, f64)> {
        let mut all: Vec<(usize, f64)> = self.left.iter().filter_map(|&j| Some((j, self.score(j)?.0))).collect();
        all.sort_by(|a, b| a.1.total_cmp(&b.1));
        all
    }
}

/// Fit `data[v]` from the variables `before` it by least squares: their strengths, in
/// order. A variable that is a combination of those already in the fit (an exact copy,
/// say) adds nothing and gets 0.
fn fit_row(data: &[Vec<f64>], before: &[usize], v: usize) -> Vec<f64> {
    let mut used: Vec<usize> = Vec::new();
    for (m, &u) in before.iter().enumerate() {
        let cols: Vec<&[f64]> = used.iter().map(|&k| data[before[k]].as_slice()).collect();
        let new = residuals(&cols, &data[u]);
        if variance(&new) > 1e-12 * variance(&data[u]) {
            used.push(m);
        }
    }
    let cols: Vec<&[f64]> = used.iter().map(|&k| data[before[k]].as_slice()).collect();
    let mut out = vec![0.0; before.len()];
    if let Some(w) = regress(&cols, &data[v]) {
        for (k, &m) in used.iter().enumerate() {
            out[m] = w[k];
        }
    }
    out
}

/// The search (DirectLiNGAM), done a piece at a time so that it can share the frame:
/// round by round, the root found and its effect taken away; then each variable fitted
/// from those before it.
struct Search {
    data: Vec<Vec<f64>>,
    rounds: Vec<Round>,
    /// The candidate whose table is in hand, and the table.
    kept: Option<(usize, Likeness)>,
    order: Vec<usize>,
    /// Every arrow from earlier to later in the order, with its fitted strength (once the
    /// order is complete).
    fit: Option<Vec<(usize, usize, f64)>>,
}

impl Search {
    fn new(data: Vec<Vec<f64>>) -> Search {
        let p = data.len();
        let first = Round::new((0..p).collect(), data.clone());
        Search { data, rounds: vec![first], kept: None, order: Vec::new(), fit: None }
    }

    fn done(&self) -> bool {
        self.fit.is_some()
    }

    /// One piece of work: a candidate's table, one pair, or the end of a round. False
    /// when there was nothing left to do.
    fn unit(&mut self) -> bool {
        if self.done() {
            return false;
        }
        let Some(round) = self.rounds.last_mut() else { return false };
        if let Some((j, i)) = round.next() {
            if self.kept.as_ref().is_none_or(|(k, _)| *k != j) {
                self.kept = Some((j, Likeness::of(&round.now[j])));
                return true;
            }
            let r = left_over(&round.now[j], &round.now[i]);
            round.pairs[j][i] = self.kept.as_ref().map(|(_, table)| table.against(&r));
            return true;
        }
        // every pair is measured: the candidate with the smallest score is the root (on
        // a tie, the first in the alphabet)
        self.kept = None;
        let Some(&(winner, _)) = round.ranked().first() else {
            self.order.extend(round.left.iter().copied());
            self.finish();
            return true;
        };
        round.winner = Some(winner);
        self.order.push(winner);
        let left: Vec<usize> = round.left.iter().copied().filter(|&v| v != winner).collect();
        if left.len() <= 1 {
            self.order.extend(left);
            self.finish();
            return true;
        }
        // its effect taken away: every other replaced by what is left of it
        let mut now = vec![Vec::new(); round.now.len()];
        for &i in &left {
            now[i] = left_over(&round.now[winner], &round.now[i]);
        }
        self.rounds.push(Round::new(left, now));
        true
    }

    /// The strengths, once the order is complete.
    fn finish(&mut self) {
        let mut fit = Vec::new();
        for (k, &v) in self.order.iter().enumerate() {
            let before = &self.order[..k];
            for (m, b) in fit_row(&self.data, before, v).into_iter().enumerate() {
                fit.push((before[m], v, b));
            }
        }
        self.fit = Some(fit);
    }

    /// As much of the work as the budget allows (at least one piece). True if any was left.
    fn work(&mut self, budget: &crate::clock::Budget) -> bool {
        let mut any = false;
        while self.unit() {
            any = true;
            if budget.spent() {
                break;
            }
        }
        any
    }

    /// The whole search at once.
    #[cfg(test)]
    fn run(&mut self) {
        while self.unit() {}
    }
}

/// How an arrow of the found graph, or of the true one, comes out when the two are compared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// In both, the same way round.
    Found,
    /// In both, the other way round.
    Reversed,
    /// Found, and in the true graph neither way round.
    Extra,
    /// In the true graph, and found neither way round.
    Missed,
}

/// The two graphs compared arrow by arrow: each true arrow once (as it was found, if it
/// was), and each found arrow (strength at least `threshold`) that is in the truth
/// neither way round.
fn compare(truth: &[(usize, usize, f64)], fit: &[(usize, usize, f64)], threshold: f64) -> Vec<(usize, usize, Verdict)> {
    let kept: Vec<(usize, usize)> = fit.iter().filter(|a| a.2.abs() >= threshold).map(|a| (a.0, a.1)).collect();
    let mut out = Vec::new();
    for &(a, b, _) in truth {
        out.push(if kept.contains(&(a, b)) {
            (a, b, Verdict::Found)
        } else if kept.contains(&(b, a)) {
            (b, a, Verdict::Reversed)
        } else {
            (a, b, Verdict::Missed)
        });
    }
    for &(a, b) in &kept {
        if !truth.iter().any(|t| (t.0, t.1) == (a, b) || (t.0, t.1) == (b, a)) {
            out.push((a, b, Verdict::Extra));
        }
    }
    out
}

/// Every arrow from earlier to later in an order, with its fitted strength.
type Fit = Vec<(usize, usize, f64)>;

/// The same search on new samples, several times over: how often it comes out right.
struct Tally {
    /// For each run done: whether its order agrees with the truth, and its fitted strengths.
    runs: Vec<(bool, Fit)>,
    current: Option<Search>,
}

// ----- where everything goes ----------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    min: Vec2,
    max: Vec2,
}

impl Rect {
    fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
        // (never upside down: a tiny window squeezes, it does not fold)
        Rect { min: Vec2::new(x0.min(x1), y0.min(y1)), max: Vec2::new(x0.max(x1), y0.max(y1)) }
    }

    fn w(&self) -> f32 {
        self.max.x - self.min.x
    }

    fn h(&self) -> f32 {
        self.max.y - self.min.y
    }

    fn contains(&self, p: Vec2) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }
}

/// Where everything goes on a canvas of `view` logical pixels, `inset` of whose top lies
/// under the application's buttons: the two graphs side by side at the top, the step
/// with its buttons and words under them, then the candidates' scores (or the strengths)
/// and the two scatters, beside each other on a wide canvas and one under the other on
/// a narrow one, and the tally at the bottom. What there is no room for is left out.
#[derive(Clone, Copy, Debug)]
struct Layout {
    narrow: bool,
    /// Text sizes.
    big: f32,
    small: f32,
    truth: Rect,
    found: Rect,
    /// The step: its title and buttons, and its words.
    head: Rect,
    buttons: Option<[Rect; 3]>,
    words: Option<Rect>,
    details: Option<Rect>,
    scatters: Option<[Rect; 2]>,
    tally: Option<Rect>,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let (w, h) = (view.x.max(1.0), view.y.max(1.0));
        if w >= 720.0 && h >= 440.0 {
            let m = 16.0;
            let top = 12.0f32.max(inset + 8.0);
            // the details' row: the table on the left, two square scatters (under two lines
            // of title and over one of numbers) on the right
            let split = m + (w - 2.0 * m) * 0.42;
            let side = ((w - m - split - 26.0) * 0.5).min(340.0);
            let (head_h, tally_h, caption) = (78.0, 30.0, 26.0);
            let wanted = side + 50.0;
            // the graphs get what is left, within reason
            let graph = (h - top - 12.0 - head_h - 12.0 - wanted - 14.0 - tally_h - caption).clamp(140.0, 400.0);
            let truth = Rect::new(m, top, w * 0.5 - 8.0, top + graph);
            let found = Rect::new(w * 0.5 + 8.0, top, w - m, top + graph);
            let y = truth.max.y + 12.0;
            let head = Rect::new(m, y, w - m, y + 26.0);
            let words = Rect::new(m, head.max.y + 4.0, w - m, head.max.y + 4.0 + 3.0 * 16.0);
            let x = w - m - 236.0;
            let buttons = [Rect::new(x, y, x + 70.0, y + 26.0), Rect::new(x + 76.0, y, x + 146.0, y + 26.0), Rect::new(x + 152.0, y, x + 236.0, y + 26.0)];
            let y = words.max.y + 12.0;
            let details_h = wanted.min(h - 8.0 - y);
            let details = (details_h >= 110.0).then(|| Rect::new(m, y, split - 10.0, y + details_h));
            let scatters = (details.is_some() && side >= 120.0).then(|| {
                let x = split + 10.0;
                [Rect::new(x, y, x + side, y + details_h), Rect::new(x + side + 16.0, y, x + 2.0 * side + 16.0, y + details_h)]
            });
            let ty = y + details_h + 14.0;
            let tally = (details.is_some() && ty + tally_h <= h - 4.0).then(|| Rect::new(m, ty, w - m, ty + tally_h));
            Layout { narrow: false, big: 13.5, small: 11.5, truth, found, head, buttons: Some(buttons), words: Some(words), details, scatters, tally }
        } else if h >= 520.0 && w >= 280.0 {
            let m = 10.0;
            let top = 10.0f32.max(inset + 6.0);
            let graph = (w * 0.46).clamp(120.0, 230.0).min(h * 0.26);
            let truth = Rect::new(m, top, w * 0.5 - 4.0, top + graph);
            let found = Rect::new(w * 0.5 + 4.0, top, w - m, top + graph);
            let y = truth.max.y + 10.0;
            let head = Rect::new(m, y, w - m, y + 20.0);
            let by = head.max.y + 4.0;
            let bw = ((w - 2.0 * m - 12.0) / 3.0).min(96.0);
            let buttons = [0.0, 1.0, 2.0].map(|k| Rect::new(m + k * (bw + 6.0), by, m + k * (bw + 6.0) + bw, by + 26.0));
            let words = Rect::new(m, by + 32.0, w - m, by + 32.0 + 5.0 * 14.5);
            let tally_h = 28.0;
            let y = words.max.y + 10.0;
            let rest = h - 8.0 - y - tally_h - 10.0;
            let details_h = (rest * 0.42).clamp(110.0, 200.0).min(rest);
            let details = (details_h >= 90.0).then(|| Rect::new(m, y, w - m, y + details_h));
            // (the two scatters side by side, as a pair in the middle)
            let sy = y + details_h + 10.0;
            let side = ((w - 2.0 * m - 10.0) * 0.5).min(h - 8.0 - tally_h - 10.0 - sy);
            let x0 = (w - 2.0 * side - 10.0) * 0.5;
            let scatters = (details.is_some() && side >= 100.0).then(|| [Rect::new(x0, sy, x0 + side, sy + side), Rect::new(x0 + side + 10.0, sy, x0 + 2.0 * side + 10.0, sy + side)]);
            let ty = if scatters.is_some() { sy + side + 10.0 } else { sy };
            let tally = (details.is_some() && ty + tally_h <= h - 4.0).then(|| Rect::new(m, ty, w - m, ty + tally_h));
            Layout { narrow: true, big: 12.5, small: 10.5, truth, found, head, buttons: Some(buttons), words: Some(words), details, scatters, tally }
        } else if w >= 520.0 && h >= 300.0 {
            // wide and short (a phone on its side): the graphs and the step on the left,
            // the scores on the right
            let m = 10.0;
            let top = 8.0f32.max(inset + 6.0);
            let half = w * 0.5;
            let graph = ((h - top) * 0.48).clamp(100.0, 260.0);
            let truth = Rect::new(m, top, half * 0.5 - 3.0, top + graph);
            let found = Rect::new(half * 0.5 + 3.0, top, half - 6.0, top + graph);
            let y = truth.max.y + 8.0;
            let head = Rect::new(m, y, half - 6.0, y + 18.0);
            let by = head.max.y + 4.0;
            let bw = ((half - m - 18.0) / 3.0).min(90.0);
            let buttons = [0.0, 1.0, 2.0].map(|k| Rect::new(m + k * (bw + 6.0), by, m + k * (bw + 6.0) + bw, by + 24.0));
            let wy = by + 30.0;
            let words = (h - 6.0 - wy >= 26.0).then(|| Rect::new(m, wy, half - 6.0, h - 6.0));
            let details = Some(Rect::new(half + 6.0, top + 4.0, w - m, h - 8.0));
            Layout { narrow: true, big: 11.5, small: 10.0, truth, found, head, buttons: Some(buttons), words, details, scatters: None, tally: None }
        } else {
            // a small canvas: the two graphs and the step
            let m = 8.0;
            let top = 6.0f32.max(inset + 4.0);
            let graph = (h * 0.5).clamp(40.0, 220.0);
            let truth = Rect::new(m, top, w * 0.5 - 4.0, top + graph);
            let found = Rect::new(w * 0.5 + 4.0, top, w - m, top + graph);
            let y = truth.max.y + 6.0;
            let head = Rect::new(m, y, w - m, y + 18.0);
            let fits = w >= 220.0 && y + 50.0 <= h;
            let bw = ((w - 2.0 * m - 12.0) / 3.0).min(80.0);
            let by = head.max.y + 3.0;
            let buttons = fits.then(|| [0.0, 1.0, 2.0].map(|k| Rect::new(m + k * (bw + 6.0), by, m + k * (bw + 6.0) + bw, by + 24.0)));
            let y = by + 30.0;
            let details = (fits && h - y >= 120.0).then(|| Rect::new(m, y, (w - m).min(m + 520.0), h - 6.0));
            Layout { narrow: true, big: 11.5, small: 10.0, truth, found, head, buttons, words: None, details, scatters: None, tally: None }
        }
    }
}

// ----- the scene ----------------------------------------------------------------------------

/// An experiment: (button, what to look for, the true graph, the noise, samples, seed,
/// threshold, the stage it starts from and the one it plays to, whether the tally runs).
type Trial = (&'static str, &'static str, usize, Noise, usize, u64, f32, f32, f32, bool);

const TRIALS: [Trial; 5] = [
    (
        "a chain, link by link",
        r"The chain $C \to A \to D \to B$, step by step. At each step the root scores below 1 (what random pairing gives) and its residuals form an even band (left scatter); every other candidate scores above 2, its residuals changing shape along it (right). $C$, $A$ and $D$ are placed in turn, and $B$ is left last. In the true graph what is placed fades: what is left is again a chain. The fit keeps exactly the three true arrows.",
        0,
        Noise::Uniform,
        300,
        1,
        THRESHOLD,
        0.0,
        99.0,
        false,
    ),
    (
        "a diamond",
        r"Two routes from $D$ to $B$. $D$ is placed first. Then $\tilde A$ and $\tilde C$ both score below 1: either could be a root, and both are, since no arrow joins $A$ and $C$. $B$, which both point into, comes last. The fit finds all four arrows and adds none.",
        2,
        Noise::Uniform,
        300,
        1,
        THRESHOLD,
        0.0,
        99.0,
        false,
    ),
    (
        "bell-shaped noise",
        r"The chain again, with normal noise. At every step every candidate scores below 1: when every noise is bell-shaped, a residual uncorrelated with its input is independent of it whichever way the arrow points, so nothing tells a root from the rest. The order comes out $D, A, C, B$, against two true arrows. In 20 new samples (the strip) the true order came out once; a random order is right 1 time in 24.",
        0,
        Noise::Normal,
        300,
        1,
        THRESHOLD,
        0.0,
        99.0,
        true,
    ),
    (
        "a hidden common cause",
        r"$H$ causes both $A$ and $D$, and is not measured. The order agrees with every true arrow, yet the fit adds an arrow $A \to D$ that is not there, and with it $B \to D$: to the method, a cause it never sees looks like an arrow between its effects. In 20 new samples (the strip) the order was always right and the graph never.",
        3,
        Noise::Uniform,
        300,
        1,
        THRESHOLD,
        4.0,
        5.0,
        true,
    ),
    (
        "thirty samples",
        r"The chain with only 30 samples. The scores are rough: at the second step no candidate scores below 1, and $\tilde D$ is placed before $\tilde A$, against the true arrow $A \to D$. In 20 new samples (the strip) the true order came out 5 times (a random order: 1 time in 24), and the whole graph once.",
        0,
        Noise::Uniform,
        30,
        1,
        THRESHOLD,
        0.0,
        99.0,
        true,
    ),
];

pub struct Discover {
    truth: usize,
    noise: Noise,
    n: usize,
    seed: u64,
    threshold: f32,
    hide: bool,
    search: Search,
    /// The tally, when asked for (for the settings as they are: a change clears it).
    tally: Option<Tally>,
    /// How far through the steps the picture is, and where it is going; playing through
    /// with a pause after each step, and what is left of the pause.
    stage: f32,
    goal: f32,
    through: bool,
    hold: f32,
    /// The bars as drawn: each pair's score (in the unit of its round), following the
    /// search through two eases, keyed by round, candidate and other.
    shown: Vec<f32>,
    between: Vec<f32>,
    /// The pair picked for the second scatter: round, candidate, other.
    pick: Option<(usize, usize, usize)>,
    /// The places of the variables in the graphs as drawn (unit square), following the
    /// true graph's when it changes.
    places: Vec<Vec2>,
    places_between: Vec<Vec2>,
    /// A setting changed: the scatters fade out and in again over this many seconds.
    fade: f32,
}

impl Discover {
    fn create() -> Discover {
        let truth = 0;
        let mut s = Discover {
            truth,
            noise: Noise::Uniform,
            n: 300,
            seed: 1,
            threshold: THRESHOLD,
            hide: false,
            search: Search::new(Vec::new()),
            tally: None,
            stage: 1.0,
            goal: 1.0,
            through: false,
            hold: 0.0,
            shown: Vec::new(),
            between: Vec::new(),
            pick: None,
            places: TRUTHS[truth].vars.iter().map(|v| Vec2::from(v.1)).collect(),
            places_between: TRUTHS[truth].vars.iter().map(|v| Vec2::from(v.1)).collect(),
            fade: 0.0,
        };
        s.resample();
        s
    }

    fn the_truth(&self) -> &'static Truth {
        &TRUTHS[self.truth.min(TRUTHS.len() - 1)]
    }

    /// How many variables are measured.
    fn p(&self) -> usize {
        self.the_truth().count()
    }

    /// The steps: one for each place in the order but the last, then the fit, then the
    /// comparison.
    fn steps(&self) -> usize {
        self.p().saturating_sub(1) + 2
    }

    /// Draw new samples with the settings as they are, and start the search on them.
    fn resample(&mut self) {
        let data = self.the_truth().sample(self.noise, self.n, self.seed);
        self.search = Search::new(data);
        self.pick = None;
        self.tally = None;
        self.fade = 0.3;
        let steps = self.steps() as f32;
        (self.stage, self.goal) = (self.stage.min(steps), self.goal.min(steps));
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let Some(&(_, _, truth, noise, n, seed, threshold, from, to, tally)) = TRIALS.get(k) else { return };
        (self.truth, self.noise, self.n, self.seed, self.threshold, self.hide) = (truth, noise, n, seed, threshold, false);
        self.resample();
        let steps = self.steps() as f32;
        (self.stage, self.goal) = (from.min(steps), to.min(steps));
        self.through = self.goal > self.stage + 1.0;
        self.hold = if self.through { HOLD * 0.5 } else { 0.0 };
        if tally {
            self.start_tally();
        }
    }

    fn start_tally(&mut self) {
        self.tally = Some(Tally { runs: Vec::new(), current: None });
    }

    /// The seed of run `k` of the tally: new samples, not the one on screen.
    fn tally_seed(&self, k: usize) -> u64 {
        self.seed.wrapping_mul(1_000_003).wrapping_add(7919 * (k as u64 + 1))
    }

    /// Do the work owed: the search on screen first, then the tally, within the budget.
    fn work(&mut self, budget: &crate::clock::Budget) {
        if !self.search.done() {
            self.search.work(budget);
            if budget.spent() {
                return;
            }
        }
        let truth = *self.the_truth();
        let (noise, n) = (self.noise, self.n);
        let seeds: Vec<u64> = (0..RUNS).map(|k| self.tally_seed(k)).collect();
        let Some(tally) = self.tally.as_mut() else { return };
        while tally.runs.len() < RUNS {
            let search = tally.current.get_or_insert_with(|| Search::new(truth.sample(noise, n, seeds[tally.runs.len().min(RUNS - 1)])));
            search.work(budget);
            if search.done() {
                let fit = search.fit.clone().unwrap_or_default();
                tally.runs.push((truth.agrees(&search.order), fit));
                tally.current = None;
            }
            if budget.spent() {
                break;
            }
        }
    }

    /// Everything at once (for the tests).
    #[cfg(test)]
    fn finish_all(&mut self) {
        let budget = crate::clock::Budget::new(f64::INFINITY);
        self.work(&budget);
        while !self.search.done() || self.tally.as_ref().is_some_and(|t| t.runs.len() < RUNS) {
            self.work(&budget);
        }
    }

    /// Whether a found graph is exactly the true one (with the threshold as it is).
    fn exact(&self, fit: &[(usize, usize, f64)]) -> bool {
        compare(&self.the_truth().seen_arrows(), fit, self.threshold as f64).iter().all(|v| v.2 == Verdict::Found)
    }

    /// The tally so far: runs done, orders right, graphs exactly right.
    fn counts(&self) -> Option<(usize, usize, usize)> {
        let t = self.tally.as_ref()?;
        Some((t.runs.len(), t.runs.iter().filter(|r| r.0).count(), t.runs.iter().filter(|r| r.0 && self.exact(&r.1)).count()))
    }

    // ----- stepping (as in `linalg/gram.rs`) -----

    fn next(&mut self) {
        let from = if self.through { self.stage } else { self.goal.max(self.stage) };
        self.goal = (from.floor() + 1.0).min(self.steps() as f32);
        (self.through, self.hold) = (false, 0.0);
    }

    fn can_go_on(&self) -> bool {
        (if self.through { self.stage } else { self.goal.max(self.stage) }) < self.steps() as f32
    }

    fn back(&mut self) {
        self.goal = (self.goal.min(self.stage).ceil() - 1.0).max(0.0);
        (self.through, self.hold) = (false, 0.0);
    }

    fn replay(&mut self) {
        (self.stage, self.goal, self.through, self.hold) = (0.0, self.steps() as f32, true, HOLD * 0.6);
    }

    fn moving(&self) -> bool {
        self.stage != self.goal || self.hold > 0.0
    }

    /// Move the steps on by `dt` seconds: towards the goal a step at a time, pausing
    /// after each when playing through.
    fn advance(&mut self, dt: f32) {
        let mut dt = dt.max(0.0);
        while dt > 0.0 && self.moving() {
            if self.hold > 0.0 {
                let wait = self.hold.min(dt);
                self.hold -= wait;
                dt -= wait;
                if self.hold <= 0.0 {
                    self.hold = 0.0;
                }
                continue;
            }
            let forward = self.goal > self.stage;
            let (next, time) = if forward { ((self.stage.floor() + 1.0).min(self.goal), STEP_TIME) } else { ((self.stage.ceil() - 1.0).max(self.goal), BACK_TIME) };
            let need = (next - self.stage).abs() * time;
            if dt >= need {
                self.stage = next;
                dt -= need;
                if self.through && forward && self.stage < self.goal {
                    self.hold = HOLD;
                }
            } else {
                self.stage += if forward { dt / time } else { -dt / time };
                dt = 0.0;
            }
        }
        if !self.moving() {
            self.through = false;
        }
    }

    /// The step on screen (0 to `steps`): the one playing or the last one finished; and
    /// how far through it the picture is (1 when it is done).
    fn step(&self) -> (usize, f32) {
        let k = (self.stage.ceil().max(0.0) as usize).min(self.steps());
        let u = if k == 0 { 1.0 } else { (self.stage - (k as f32 - 1.0)).max(0.0).min(1.0) };
        (k, u)
    }

    // ----- what is drawn, following what is so -----

    /// The pair's score as the bars have it, in the unit of its round: (round, candidate, other).
    fn key(&self, r: usize, j: usize, i: usize) -> usize {
        let p = self.p().max(1);
        (r * p + j) * p + i
    }

    fn targets(&self) -> Vec<f32> {
        let p = self.p().max(1);
        let mut out = vec![0.0; p * p * p];
        for (r, round) in self.search.rounds.iter().enumerate().take(p) {
            let unit = round.pair_unit();
            for &j in &round.left {
                for &i in &round.left {
                    if let Some((h, _)) = round.pair(j, i) {
                        let at = self.key(r, j, i);
                        if let Some(v) = out.get_mut(at) {
                            *v = (h / unit) as f32;
                        }
                    }
                }
            }
        }
        out
    }

    /// Bring the drawn bars, places and fades `dt` seconds nearer to what is so. True
    /// while anything is still on its way.
    fn follow(&mut self, dt: f32) -> bool {
        let to = self.targets();
        if self.shown.len() != to.len() {
            self.shown = vec![0.0; to.len()];
            self.between = vec![0.0; to.len()];
        }
        let k = 1.0 - (-dt.max(0.0) / EASE).exp();
        let mut moving = false;
        for ((s, b), t) in self.shown.iter_mut().zip(self.between.iter_mut()).zip(&to) {
            *b += (t - *b) * k;
            *s += (*b - *s) * k;
            if (*s - t).abs() < 1e-3 && (*b - t).abs() < 1e-3 || !s.is_finite() || !b.is_finite() {
                (*s, *b) = (*t, *t);
            } else {
                moving = true;
            }
        }
        let want: Vec<Vec2> = self.the_truth().vars.iter().map(|v| Vec2::from(v.1)).collect();
        if self.places.len() != want.len() {
            // (a variable that was not there before starts where it is to be)
            let start = self.places.len().min(want.len());
            self.places.truncate(want.len());
            self.places.extend_from_slice(&want[start..]);
            self.places_between.truncate(want.len());
            self.places_between.extend_from_slice(&want[self.places_between.len().min(want.len())..]);
        }
        if self.places_between.len() != want.len() {
            self.places_between = self.places.clone();
        }
        let k2 = 1.0 - (-dt.max(0.0) / (1.5 * EASE)).exp();
        for ((p, b), w) in self.places.iter_mut().zip(self.places_between.iter_mut()).zip(&want) {
            *b += (*w - *b) * k2;
            *p += (*b - *p) * k2;
            // (exactly there in the end, from a twentieth of a pixel off)
            if (*p - *w).length() < 5e-5 && (*b - *w).length() < 5e-5 || !p.is_finite() || !b.is_finite() {
                (*p, *b) = (*w, *w);
            } else {
                moving = true;
            }
        }
        if self.fade > 0.0 {
            self.fade = (self.fade - dt.max(0.0)).max(0.0);
            moving = true;
        }
        moving
    }

    fn settled(&self) -> bool {
        let to = self.targets();
        self.shown == to && self.between == to && self.fade <= 0.0 && self.places.len() == self.the_truth().vars.len() && self.places.iter().chain(&self.places_between).zip(self.the_truth().vars.iter().chain(self.the_truth().vars)).all(|(p, v)| *p == Vec2::from(v.1))
    }
}

// ----- words ------------------------------------------------------------------------------

/// A list of names as one formula: `$A, B, D$`, with tildes if `left` (what is left of
/// each once the variables already placed are taken away).
fn names(truth: &Truth, vars: &[usize], left: bool) -> String {
    let each: Vec<String> = vars.iter().map(|&v| if left { format!(r"\tilde{{{}}}", LETTERS[truth.letter(v)]) } else { LETTERS[truth.letter(v)].to_string() }).collect();
    format!("${}$", each.join(", "))
}

/// One variable's name as a formula.
fn name(truth: &Truth, v: usize, left: bool) -> &'static str {
    let l = truth.letter(v);
    if left {
        MATH_LEFT[l]
    } else {
        MATH[l]
    }
}

fn ordinal(k: usize) -> &'static str {
    ["first", "second", "third", "fourth", "fifth"].get(k).copied().unwrap_or("next")
}

impl Discover {
    /// The step's title.
    fn title(&self, k: usize) -> String {
        let truth = self.the_truth();
        let p = self.p();
        if k < p {
            let r = k.saturating_sub(1);
            match self.search.rounds.get(r) {
                Some(round) => format!("Which of {} has nothing causing it?", names(truth, &round.left, r > 0)),
                None => "Which has nothing causing it?".into(),
            }
        } else if k == p {
            "Fit each from those before it; drop weak arrows".into()
        } else {
            "Compare with the true graph".into()
        }
    }

    /// The words while step `k` is looking for its root.
    fn testing(&self, k: usize) -> String {
        let truth = self.the_truth();
        match self.search.rounds.get(k.saturating_sub(1)) {
            Some(round) => format!(
                "Testing each of {} as the root: every other one is fitted from it by a straight line, and its residuals are measured against it.",
                names(truth, &round.left, k > 1)
            ),
            None => "Measuring…".into(),
        }
    }

    /// The step's words: what was found and why.
    fn reasoning(&self, k: usize) -> String {
        let truth = self.the_truth();
        let p = self.p();
        if k == 0 {
            return "Try each variable in turn as the root: fit every other one from it by a straight line, and measure how much what is left over (the residuals) depends on it. A root's residuals are its effects' own noise, independent of it; anything else's are not. next ▶ runs the tests.".into();
        }
        if k < p {
            let Some(round) = self.search.rounds.get(k - 1) else { return "Measuring…".into() };
            let Some(w) = round.winner else { return "Measuring…".into() };
            let unit = round.unit();
            let ranked = round.ranked();
            let best = ranked.first().map_or(0.0, |r| r.1 / unit);
            let second = ranked.get(1).map_or(0.0, |r| r.1 / unit);
            let runner = ranked.get(1).map_or(w, |r| r.0);
            let tilde = k > 1;
            let (wn, rn) = (name(truth, w, tilde), name(truth, runner, tilde));
            let placed = format!("{} is placed {}.", name(truth, w, false), ordinal(k - 1));
            let below = ranked.iter().filter(|r| r.1 / unit <= 1.0).count();
            let verdict = if below == ranked.len() && ranked.len() > 2 {
                format!("Every candidate leaves residuals scoring no more than random pairing gives (1): any could be a root, and the scores hardly tell them apart. {wn} scores lowest ({}). {placed}", num(best, 1))
            } else if below >= 2 {
                format!("{wn} and {rn} both leave residuals scoring no more than random pairing gives ({} and {}): either could be a root. {wn} scores lowest. {placed}", num(best, 1), num(second, 1))
            } else if below == 1 {
                format!("Fitted from {wn}, the others leave residuals scoring {}, no more than random pairing gives (1); every other candidate scores more ({} at least). {placed}", num(best, 1), num(second, 1))
            } else {
                format!("No candidate leaves residuals scoring as low as random pairing gives (1): the lowest is {wn}, with {}, then {rn}, with {}. {placed} It may be a wrong turn.", num(best, 1), num(second, 1))
            };
            let rest: Vec<usize> = round.left.iter().copied().filter(|&v| v != w).collect();
            let then = if rest.len() == 1 {
                format!(" {} is the last one left, and comes last.", name(truth, rest[0], false))
            } else {
                format!(" Its effect is taken away: {} are replaced by their residuals fitted from it. If it is a root, these follow the same arrows as before, among themselves: a smaller model of the same kind.", names(truth, &rest, false))
            };
            return verdict + &then;
        }
        let Some(fit) = self.search.fit.as_ref() else { return "Measuring…".into() };
        let t = self.threshold as f64;
        if k == p {
            let kept = fit.iter().filter(|a| a.2.abs() >= t).count();
            return format!(
                "In the order {}, each variable is fitted by least squares from all those before it: one strength for every arrow that could be. Strengths below {} count as no arrow (drag the dashed line). {} kept.",
                names(truth, &self.search.order, false),
                num(t, 2),
                match kept {
                    0 => "No arrow is".to_string(),
                    1 => "1 arrow is".to_string(),
                    n => format!("{n} arrows are"),
                }
            );
        }
        let verdicts = compare(&truth.seen_arrows(), fit, t);
        let count = |v: Verdict| verdicts.iter().filter(|x| x.2 == v).count();
        let real = truth.seen_arrows().len();
        let arrow = |&(a, b, _): &(usize, usize, Verdict)| format!("${} \\to {}$", LETTERS[truth.letter(a)], LETTERS[truth.letter(b)]);
        let list = |v: Verdict| verdicts.iter().filter(|x| x.2 == v).map(arrow).collect::<Vec<_>>().join(", ");
        let mut out = if real == 0 { "The true graph has no arrows.".to_string() } else { format!("Found {} of the {} true arrows.", count(Verdict::Found), real) };
        for (v, what) in [(Verdict::Reversed, "the wrong way round"), (Verdict::Missed, "missed"), (Verdict::Extra, "not in the true graph")] {
            if count(v) > 0 {
                out += &format!(" {}: {}.", what[..1].to_uppercase() + &what[1..], list(v));
            }
        }
        // the effects of a cause that is not measured, among the measured variables
        let seen = truth.seen();
        let hidden: Vec<usize> = (0..truth.vars.len()).filter(|&v| truth.vars[v].2).collect();
        let shared: Vec<usize> = (0..seen.len()).filter(|&k| truth.arrows.iter().any(|a| hidden.contains(&a.0) && a.1 == seen[k])).collect();
        let extra: Vec<(usize, usize)> = verdicts.iter().filter(|x| x.2 == Verdict::Extra).map(|x| (x.0, x.1)).collect();
        let between = |&(a, b): &(usize, usize)| shared.contains(&a) && shared.contains(&b);
        if extra.iter().any(between) {
            let cause = hidden.first().map_or("$H$", |&h| MATH[truth.vars[h].0]);
            let each: Vec<&str> = shared.iter().map(|&k| name(truth, k, false)).collect();
            let both = match each.split_last() {
                Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
                _ => each.join(""),
            };
            out += &format!(" {both} share a cause, {cause}, that is not measured: to the method, it looks like an arrow between them.");
            if !extra.iter().all(between) {
                out += " Other extra arrows can come with it.";
            }
        } else if verdicts.iter().all(|v| v.2 == Verdict::Found) {
            out += if real == 0 { " None was found." } else { " Nothing missed, nothing extra, nothing reversed." };
        }
        out
    }
}

// ----- drawing helpers ----------------------------------------------------------------------

/// A button over the picture (logical pixels): true when it is clicked. One that cannot
/// be used now is drawn dim and ignores clicks.
fn hud_button(f: &mut Frame, r: Rect, text: &str, usable: bool, pointer: Option<Vec2>, size: f32) -> bool {
    let over = pointer.is_some_and(|p| r.contains(p));
    let fill = match (usable, over) {
        (false, _) => Color::hex(0x141925),
        (true, true) => Color::hex(0x33405E),
        (true, false) => Color::hex(0x222B40),
    };
    hud_rect(f, r.min, r.max, fill);
    f.text_px((r.min + r.max) * 0.5, text, size, if usable { pal::TEXT } else { pal::FAINT }, Align2::CENTER_CENTER);
    usable && over && f.ptr.clicked
}

/// A dashed arrow in screen space, from rim to rim of two discs of radius `radius`.
fn dashed_arrow(f: &mut Frame, a: Vec2, b: Vec2, radius: f32, width: f32, color: Color) {
    let along = (b - a).normalize_or_zero();
    if along == Vec2::ZERO {
        return;
    }
    let (p, q) = (a + along * (radius + 3.0), b - along * (radius + 4.0));
    let length = (q - p).dot(along);
    if length <= 0.0 {
        return;
    }
    let dash = 6.0;
    let count = (length / (2.0 * dash)).ceil().max(1.0) as usize;
    for k in 0..count {
        let (s, e) = (k as f32 * 2.0 * dash, (k as f32 * 2.0 * dash + dash).min(length));
        hud_line(f, p + along * s, p + along * e, width, color);
    }
    let (head, side) = (4.0 + 2.5 * width, Vec2::new(-along.y, along.x));
    hud_line(f, q, q - along * head + side * head * 0.5, width, color);
    hud_line(f, q, q - along * head - side * head * 0.5, width, color);
}

/// A dashed ring in screen space.
fn dashed_ring(f: &mut Frame, centre: Vec2, radius: f32, width: f32, color: Color) {
    let dashes = 12;
    for k in 0..dashes {
        let a0 = k as f32 / dashes as f32 * TAU;
        let a1 = a0 + 0.55 * TAU / dashes as f32;
        hud_path(f, (0..=6).map(|s| centre + Vec2::from_angle(a0 + (a1 - a0) * s as f32 / 6.0) * radius), width, color);
    }
}

/// Words broken into lines no wider than `width` logical pixels (formulas kept whole).
fn wrap(f: &Frame, text: &str, size: f32, width: f32) -> Vec<String> {
    let measure = |t: &str| f.text_size(t, size).x;
    let space = f.text_size("a b", size).x - f.text_size("ab", size).x;
    let mut lines: Vec<(String, f32)> = vec![(String::new(), 0.0)];
    for word in words_of(text) {
        let w = measure(word);
        let Some(last) = lines.last_mut() else { break };
        if last.0.is_empty() {
            *last = (word.to_string(), w);
        } else if last.1 + space + w > width {
            lines.push((word.to_string(), w));
        } else {
            last.0.push(' ');
            last.0.push_str(word);
            last.1 += space + w;
        }
    }
    lines.into_iter().map(|l| l.0).collect()
}

/// The words of a text, each formula whole.
fn words_of(text: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let (mut start, mut inside) = (0, false);
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 1,
            b'$' => inside = !inside,
            b' ' if !inside => {
                if i > start {
                    words.push(&text[start..i]);
                }
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < text.len() {
        words.push(&text[start..]);
    }
    words
}

/// The first of `options` no wider than `room` (the last if none is).
fn fitting<'a>(f: &Frame, options: &[&'a str], size: f32, room: f32) -> &'a str {
    options.iter().copied().find(|t| f.text_size(t, size).x <= room).unwrap_or(options.last().copied().unwrap_or(""))
}

impl Discover {
    /// Where variable `v` (an index into the true graph's variables) is drawn in a graph's box.
    fn node_at(&self, r: Rect, v: usize, radius: f32, title: f32, foot: f32) -> Vec2 {
        let inner = Rect::new(r.min.x + radius + 12.0, r.min.y + title + radius + 6.0, r.max.x - radius - 12.0, r.max.y - foot - radius - 6.0);
        // (not stretched wider than it is tall by more than half again)
        let w = inner.w().min(inner.h() * 1.6);
        let x0 = (inner.min.x + inner.max.x - w) * 0.5;
        let p = self.places.get(v).copied().unwrap_or(Vec2::splat(0.5));
        Vec2::new(x0 + p.x * w, inner.min.y + p.y * inner.h())
    }

    fn radius(r: Rect) -> f32 {
        (r.w().min(r.h()) * 0.075).clamp(10.0, 21.0)
    }

    /// Is a graph's box large enough to draw the graph in (rings that do not overlap)?
    fn roomy(r: Rect, size: f32) -> bool {
        let radius = Self::radius(r);
        r.h() - 2.0 * (size + 10.0) - 2.0 * radius - 12.0 >= 2.0 * radius + 8.0 && r.w() >= 6.0 * radius + 24.0
    }

    /// A strength written beside the middle of an arrow: on the side away from the
    /// middle of the graph if it fits in the box there, else on the other side; gliding
    /// there when the side changes (`name` and `number` tell the label from the others).
    #[allow(clippy::too_many_arguments)]
    fn strength_label(f: &mut Frame, name: &'static str, number: u32, (a, b): (Vec2, Vec2), centre: Vec2, bounds: Rect, value: f64, size: f32, color: Color) {
        let text = num(value, 2);
        let half = f.text_size(&text, size) * 0.5 + Vec2::new(3.0, 1.0);
        let mid = (a + b) * 0.5;
        let along = (b - a).normalize_or_zero();
        let mut side = Vec2::new(-along.y, along.x);
        if (mid - centre).dot(side) < 0.0 || ((mid - centre).length() < 4.0 && side.y > 0.0) {
            side = -side;
        }
        // (the box's centre as far off the line as its own half-width across it, and a gap more)
        let place = |side: Vec2| side * (6.0 + half.x * side.x.abs() + half.y * side.y.abs());
        let inside = |c: Vec2| c.x - half.x >= bounds.min.x + 2.0 && c.x + half.x <= bounds.max.x - 2.0 && c.y - half.y >= bounds.min.y && c.y + half.y <= bounds.max.y;
        let off = if inside(mid + place(side)) { place(side) } else { place(-side) };
        // (an unseen label is written as nothing, still in its place among the others)
        f.text_placed(name, number, mid, off, if color.0[3] == 0 { String::new() } else { text }, size, color);
    }

    /// The true graph: as it is, with the variables already placed (and their arrows)
    /// faded, so that what is left can be seen to be a graph of the same kind; or, if it
    /// is hidden, the variables alone until the comparison.
    fn draw_truth(&self, f: &mut Frame, r: Rect, lay: &Layout, placed: &[usize], newest: Option<usize>, verdict_in: f32) {
        hud_panel(f, r.min, r.max);
        let truth = self.the_truth();
        let size = lay.small;
        let title = size + 10.0;
        let hidden = self.hide && verdict_in <= 0.0;
        // (hidden until the comparison, and then fading in)
        let alpha = if self.hide { verdict_in } else { 1.0 };
        let heading = if hidden { fitting(f, &["true graph: hidden until the end (guess first)", "true graph: hidden", "hidden"], size, r.w() - 16.0) } else { "true graph" };
        f.text_px(r.min + Vec2::new(8.0, 6.0), heading, size, pal::DIM, Align2::LEFT_TOP);
        if !Self::roomy(r, size) {
            return;
        }
        let radius = Self::radius(r);
        let seen = truth.seen();
        let measured = |v: usize| seen.iter().position(|&s| s == v);
        // (the same places as in the graph found, which has the order written under it)
        let foot = size + 10.0;
        let at: Vec<Vec2> = (0..truth.vars.len()).map(|v| self.node_at(r, v, radius, title, foot)).collect();
        f.text_px(Vec2::new(r.min.x + 8.0, r.max.y - 6.0), truth.name, size, pal::DIM, Align2::LEFT_BOTTOM);
        let centre = (r.min + r.max) * 0.5 + Vec2::new(0.0, title * 0.5);
        let is_placed = |v: usize| measured(v).is_some_and(|k| placed.contains(&k));
        // the arrows: from a hidden variable dashed; from one already placed faded; into
        // the newest root from a variable still left, pink (it was not a root)
        let mut links = Vec::new();
        for &(a, b, s) in truth.arrows {
            let (Some(&pa), Some(&pb)) = (at.get(a), at.get(b)) else { continue };
            let wrong = newest.is_some_and(|w| measured(b) == Some(w)) && measured(a).is_some_and(|k| !placed.contains(&k));
            let faded = is_placed(a);
            if alpha <= 0.0 {
                continue;
            }
            let color = if wrong { NAIVE.a(alpha) } else { TRUTH.a(alpha * if faded { 0.22 } else { 1.0 }) };
            if truth.vars[a].2 {
                dashed_arrow(f, pa, pb, radius, 1.8, color);
            } else {
                links.push((a, b, Link::Arrow, color, 2.0));
            }
            if r.w() >= 150.0 && alpha > 0.0 {
                Self::strength_label(f, "true strength", (a * 8 + b) as u32, (pa, pb), centre, r, s, size - 1.0, if faded { pal::FAINT } else { pal::DIM }.a(alpha));
            }
        }
        let nodes: Vec<Node> = (0..truth.vars.len())
            .map(|v| {
                let (letter, _, hid) = truth.vars[v];
                let color = if hid {
                    Color::hex(0).a(0.0)
                } else if newest.is_some_and(|w| measured(v) == Some(w)) {
                    ROOT
                } else if is_placed(v) {
                    NODE.a(0.35)
                } else {
                    NODE
                };
                // (the hidden one is drawn below, dashed: here it is only a place for its arrows)
                Node { at: at[v], name: if hid { "" } else { MATH[letter] }, color }
            })
            .collect();
        hud_graph(f, &nodes, &links, radius);
        // the hidden variable: a dashed ring, and what it is
        for (v, &(letter, _, hid)) in truth.vars.iter().enumerate() {
            if hid && alpha > 0.0 {
                let color = HIDDEN.a(alpha);
                dashed_ring(f, at[v], radius, 1.6, color);
                f.text_px(at[v], MATH[letter], (radius * 0.95).max(11.0), color, Align2::CENTER_CENTER);
                // (to its left, clear of its arrows, if the box has room there)
                let room = at[v].x - radius - 6.0 - r.min.x;
                if room >= text_width("not measured", size - 1.0) + 4.0 {
                    f.text_px(at[v] - Vec2::new(radius + 6.0, 0.0), "not measured", size - 1.0, color, Align2::RIGHT_CENTER);
                } else {
                    f.text_px(at[v] + Vec2::new(0.0, radius + 3.0), "not measured", size - 1.0, color, Align2::CENTER_TOP);
                }
            }
        }
    }

    /// The graph found: the variables in the same places, those placed numbered in the
    /// order found (the newest lit), and once the order is complete the arrows fitted:
    /// kept ones with their strengths, and at the end each coloured by how it compares
    /// with the truth.
    #[allow(clippy::too_many_arguments)]
    fn draw_found(&self, f: &mut Frame, r: Rect, lay: &Layout, placed: &[usize], newest: Option<usize>, arrows_in: f32, verdict_in: f32) {
        hud_panel(f, r.min, r.max);
        let truth = self.the_truth();
        let size = lay.small;
        let title = size + 10.0;
        let foot = size + 10.0;
        f.text_px(r.min + Vec2::new(8.0, 6.0), "graph found", size, pal::DIM, Align2::LEFT_TOP);
        let order = if placed.is_empty() { "order: none placed yet".to_string() } else { format!("order: {}", names(truth, placed, false)) };
        if !Self::roomy(r, size) {
            return;
        }
        f.text_px(Vec2::new(r.min.x + 8.0, r.max.y - 6.0), order, size, if placed.is_empty() { pal::DIM } else { ROOT }, Align2::LEFT_BOTTOM);
        let radius = Self::radius(r);
        let seen = truth.seen();
        let at: Vec<Vec2> = seen.iter().map(|&v| self.node_at(r, v, radius, title, foot)).collect();
        let centre = (r.min + r.max) * 0.5;
        let mut links = Vec::new();
        if let (Some(fit), true) = (self.search.fit.as_ref(), arrows_in > 0.0) {
            let t = self.threshold as f64;
            let verdicts = compare(&truth.seen_arrows(), fit, t);
            for &(a, b, s) in fit {
                let kept = s.abs() >= t;
                let v = verdicts.iter().find(|x| (x.0, x.1) == (a, b)).map(|x| x.2).unwrap_or(Verdict::Found);
                let color = KEPT.mix(verdict_color(v), verdict_in).a(arrows_in);
                if kept {
                    links.push((a, b, Link::Arrow, color, 2.2));
                }
                // (the label of an arrow dropped is kept, unseen, so that the labels keep
                // their order while the threshold is dragged across)
                if let (Some(&pa), Some(&pb)) = (at.get(a), at.get(b)) {
                    if r.w() >= 150.0 {
                        Self::strength_label(f, "found strength", (a * 8 + b) as u32, (pa, pb), centre, r, s, size - 1.0, if kept { color } else { color.a(0.0) });
                    }
                }
            }
            // the true arrows not found either way round, dashed
            if verdict_in > 0.0 {
                for &(a, b, v) in &verdicts {
                    if let (Verdict::Missed, Some(&pa), Some(&pb)) = (v, at.get(a), at.get(b)) {
                        dashed_arrow(f, pa, pb, radius, 1.6, TRUTH.a(0.6 * verdict_in));
                    }
                }
            }
        }
        let nodes: Vec<Node> = (0..seen.len())
            .map(|k| {
                let color = if newest == Some(k) {
                    ROOT
                } else if placed.contains(&k) {
                    ROOT.mix(NODE, 0.45)
                } else {
                    NODE
                };
                Node { at: at[k], name: MATH[truth.letter(k)], color }
            })
            .collect();
        hud_graph(f, &nodes, &links, radius);
        // the order, as a number beside each variable placed
        for (m, &k) in placed.iter().enumerate() {
            if let Some(&p) = at.get(k) {
                let badge = p + Vec2::new(radius + 7.0, -radius * 0.75);
                f.text_px(badge, (m + 1).to_string(), size, ROOT, Align2::CENTER_CENTER);
            }
        }
    }
}

fn verdict_color(v: Verdict) -> Color {
    match v {
        Verdict::Found => TRUTH,
        Verdict::Reversed => REVERSED,
        Verdict::Extra => NAIVE,
        Verdict::Missed => TRUTH.a(0.6),
    }
}

impl Discover {
    /// The step: its number and title, the buttons, and its words.
    fn draw_head(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let (k, u) = self.step();
        let steps = self.steps();
        let r = lay.head;
        let room = match lay.buttons {
            Some(b) if !lay.narrow => b[0].min.x - r.min.x - 12.0,
            _ => r.w(),
        };
        let label = format!("Step {k} of {steps}");
        let title = self.title(k);
        let lead = text_width(&label, lay.small) + 10.0;
        f.text_px(Vec2::new(r.min.x, r.min.y + r.h() * 0.5), label.as_str(), lay.small, pal::DIM, Align2::LEFT_CENTER);
        let fits = f.text_size(&title, lay.big).x <= room - lead;
        let p = self.p();
        let short = if k < p {
            "Which has nothing causing it?"
        } else if k == p {
            "Fit, and drop weak arrows"
        } else {
            "Compare with the truth"
        };
        let shown = if fits { title } else { fitting(f, &[title.as_str(), short], lay.big, room - lead).to_string() };
        f.text_px(Vec2::new(r.min.x + lead, r.min.y + r.h() * 0.5), shown, lay.big, pal::TEXT, Align2::LEFT_CENTER);
        if let Some(b) = lay.buttons {
            let at_start = self.goal.min(self.stage) <= 0.0;
            if hud_button(f, b[0], "◀ back", !at_start, pointer, lay.small + 0.5) {
                self.back();
            }
            let on = self.can_go_on();
            if hud_button(f, b[1], "next ▶", on, pointer, lay.small + 0.5) {
                self.next();
            }
            if hud_button(f, b[2], "↺ replay", true, pointer, lay.small + 0.5) {
                self.replay();
            }
        }
        if let Some(w) = lay.words {
            // (while a root is being looked for, what is being done; once it is lit, why)
            let text = if k >= 1 && k < self.p() && u < 0.75 { self.testing(k) } else { self.reasoning(k) };
            let line = lay.small + 4.0;
            let lines = wrap(f, &text, lay.small, w.w());
            let most = ((w.h() + 2.0) / line).floor().max(1.0) as usize;
            for (i, l) in lines.iter().take(most).enumerate() {
                let last = i + 1 == most && lines.len() > most;
                let text = if last { format!("{l} …") } else { l.clone() };
                f.text_px(Vec2::new(w.min.x, w.min.y + i as f32 * line), text, lay.small, pal::TEXT.mix(pal::DIM, 0.35), Align2::LEFT_TOP);
            }
        }
    }

    /// The candidates of round `r`: for each, the dependence of the others' residuals on
    /// it as a bar of stacked pieces, one per other variable, the shortest the root;
    /// drawn grown by `grow` (0 to 1), with the winner lit by `lit`. `slide` is the round
    /// whose places the rows are sliding up from (as its root leaves), and how far.
    #[allow(clippy::too_many_arguments)]
    fn draw_table(&mut self, f: &mut Frame, rect: Rect, lay: &Layout, r: usize, grow: f32, lit: f32, slide: Option<(usize, f32)>, alpha: f32, pointer: Option<Vec2>) {
        let truth = *self.the_truth();
        let Some(round) = self.search.rounds.get(r) else { return };
        let left = round.left.clone();
        let winner = round.winner;
        let size = lay.small;
        hud_panel(f, rect.min - Vec2::splat(4.0), rect.max + Vec2::splat(4.0));
        let head = fitting(
            f,
            &["How much the others' residuals depend on each candidate", "How much the residuals depend on each", "Dependence on each candidate"],
            size,
            rect.w() - 8.0,
        );
        f.text_px(rect.min + Vec2::new(4.0, 2.0), head, size, pal::TEXT.a(alpha), Align2::LEFT_TOP);
        let sub = fitting(f, &["(one piece for each other variable's residual, fitted from it by a line; shortest bar wins)", "(the others fitted from it by a line; shortest bar wins)", "(shortest bar wins)", ""], size - 1.0, rect.w() - 8.0);
        f.text_px(rect.min + Vec2::new(4.0, 4.0 + size + 3.0), sub, size - 1.0, pal::DIM.a(alpha), Align2::LEFT_TOP);
        let top = rect.min.y + 2.0 * size + 16.0;
        let axis = rect.max.y - size - 8.0;
        let rows = left.len().max(1) as f32;
        let row_h = ((axis - top) / rows).min(64.0).max(8.0);
        let bar_h = (row_h * 0.55).min(24.0);
        let name_w = 30.0;
        let value_w = 40.0;
        let x0 = rect.min.x + name_w;
        let x1 = rect.max.x - value_w;
        let tilde = r > 0;
        // the scale: the longest bar across, but never so short that random pairing (1) fills it
        let totals: Vec<f32> = left.iter().map(|&j| left.iter().filter(|&&i| i != j).map(|&i| self.shown.get(self.key(r, j, i)).copied().unwrap_or(0.0)).sum::<f32>()).collect();
        let others = (left.len().saturating_sub(1)).max(1) as f32;
        let longest = totals.iter().fold(0.0f32, |m, v| m.max(*v)) / others;
        let scale = (x1 - x0) / (longest.max(4.0) * 1.08);
        // where each row is: its place in this round, or sliding up from its place in the
        // round before as that round's root leaves
        let slot = |j: usize| left.iter().position(|&v| v == j).unwrap_or(0) as f32;
        let row_y = |j: usize| -> f32 {
            let now = slot(j);
            let from = slide.and_then(|(prev, _)| self.search.rounds.get(prev)).and_then(|pr| pr.left.iter().position(|&v| v == j)).map(|s| s as f32).unwrap_or(now);
            let t = slide.map_or(1.0, |s| smoothstep(s.1));
            top + (from + (now - from) * t) * row_h + row_h * 0.5
        };
        // the line random pairing gives
        let chance_x = x0 + scale * 1.0;
        let mut picked_box = None;
        let pick = self.pick.filter(|p| p.0 == r);
        let mut clicked = None;
        for (row, &j) in left.iter().enumerate() {
            let y = row_y(j);
            let is_winner = winner == Some(j);
            let glow = if is_winner { lit } else { 0.0 };
            if glow > 0.0 {
                hud_rect(f, Vec2::new(rect.min.x, y - row_h * 0.5), Vec2::new(rect.max.x, y + row_h * 0.5), ROOT.a(0.035 * glow * alpha));
            }
            let name_color = NODE.mix(ROOT, glow).a(alpha);
            f.text_px(Vec2::new(rect.min.x + 6.0, y), name(&truth, j, tilde), size + 1.5, name_color, Align2::LEFT_CENTER);
            let mut x = x0;
            let fill = BAR.mix(ROOT, glow);
            for (piece, &i) in left.iter().filter(|&&i| i != j).enumerate() {
                let v = self.shown.get(self.key(r, j, i)).copied().unwrap_or(0.0) / others * grow;
                let w = (v * scale).max(0.0).min(x1 - x0);
                let shade = if piece % 2 == 0 { fill } else { fill.mix(pal::WHITE, 0.18) };
                let (a, b) = (Vec2::new(x, y - bar_h * 0.5), Vec2::new((x + w).min(x1), y + bar_h * 0.5));
                if w > 0.5 {
                    hud_rect(f, a, b, shade.a(alpha));
                }
                if w >= 18.0 && bar_h >= 12.0 {
                    f.text_px((a + b) * 0.5, name(&truth, i, tilde), size - 1.0, pal::WHITE.a(0.85 * alpha), Align2::CENTER_CENTER);
                }
                if pick == Some((r, j, i)) {
                    picked_box = Some((a, b));
                }
                if pointer.is_some_and(|p| p.y >= y - row_h * 0.5 && p.y <= y + row_h * 0.5 && p.x >= a.x && p.x <= b.x.max(a.x + 4.0)) {
                    clicked = Some((j, Some(i)));
                }
                x = (x + w).min(x1);
            }
            if pointer.is_some_and(|p| p.y >= y - row_h * 0.5 && p.y <= y + row_h * 0.5 && p.x >= rect.min.x && p.x <= rect.max.x) && clicked.is_none() {
                clicked = Some((j, None));
            }
            let total = totals.get(row).copied().unwrap_or(0.0) / others;
            if grow > 0.0 && round.score(j).is_some() {
                f.text_px(Vec2::new(rect.max.x - 4.0, y), num((total * grow) as f64, 1), size, NODE.mix(ROOT, glow).a(alpha), Align2::RIGHT_CENTER);
            }
        }
        if let Some((a, b)) = picked_box {
            hud_outline(f, a - Vec2::splat(1.5), b + Vec2::splat(1.5), 1.4, pal::WHITE.a(alpha));
        }
        // the axis, and chance
        hud_line(f, Vec2::new(x0, axis), Vec2::new(x1, axis), 1.0, pal::DIM.a(0.7 * alpha));
        f.text_px(Vec2::new(x0, axis + 3.0), "0", size - 1.0, pal::DIM.a(alpha), Align2::CENTER_TOP);
        let mut y = top - 2.0;
        while y < axis {
            hud_line(f, Vec2::new(chance_x, y), Vec2::new(chance_x, (y + 4.0).min(axis)), 1.2, CHANCE.a(0.7 * alpha));
            y += 8.0;
        }
        f.text_px(Vec2::new(chance_x, axis + 3.0), "1: random pairing", size - 1.0, CHANCE.a(0.85 * alpha), Align2::CENTER_TOP);
        // a click on a piece of a bar (or a row) shows its residuals in the second scatter
        if f.ptr.clicked && alpha > 0.5 {
            if let Some((j, i)) = clicked {
                let i = i.or_else(|| self.search.rounds.get(r).and_then(|round| round.worst_of(j)));
                if let Some(i) = i {
                    self.pick = Some((r, j, i));
                }
            }
        }
    }

    /// The pairs shown in the two scatters for round `r`: the root and the residual that
    /// depends on it most (or the one picked); and a candidate not chosen, the one picked
    /// or else the one scoring highest, with its most dependent residual.
    fn scatter_pairs(&self, r: usize) -> [Option<(usize, usize)>; 2] {
        let Some(round) = self.search.rounds.get(r) else { return [None, None] };
        let Some(w) = round.winner else { return [None, None] };
        let first = match self.pick {
            Some((pr, j, i)) if pr == r && j == w => Some((w, i)),
            _ => round.worst_of(w).map(|i| (w, i)),
        };
        let second = match self.pick {
            Some((pr, j, i)) if pr == r && j != w => Some((j, i)),
            _ => round.ranked().last().and_then(|&(j, _)| if j == w { None } else { round.worst_of(j).map(|i| (j, i)) }),
        };
        [first, second]
    }

    /// One scatter: what is left of variable `i` once a line through candidate `j` is
    /// taken away, against `j`, in round `r`; with the spread of each of five vertical
    /// slices marked, which stays the same along an independent band.
    #[allow(clippy::too_many_arguments)]
    fn draw_scatter(&self, f: &mut Frame, rect: Rect, lay: &Layout, r: usize, pair: Option<(usize, usize)>, chosen: bool, alpha: f32) {
        hud_panel(f, rect.min - Vec2::splat(4.0), rect.max + Vec2::splat(4.0));
        let truth = self.the_truth();
        let size = lay.small;
        let Some((j, i)) = pair else {
            let words = if chosen { "the root's residuals, once it is found" } else { "a candidate not chosen" };
            let t = fitting(f, &[words, "(none yet)"], size, rect.w() - 8.0);
            f.text_px((rect.min + rect.max) * 0.5, t, size, pal::FAINT, Align2::CENTER_CENTER);
            return;
        };
        let Some(round) = self.search.rounds.get(r) else { return };
        let tilde = r > 0;
        let (x, y) = (&round.now[j], left_over(&round.now[j], &round.now[i]));
        // the title: which candidate, and whose residual
        let who = if chosen { format!("{} (placed)", name(truth, j, tilde)) } else { format!("{} (not placed)", name(truth, j, tilde)) };
        f.text_px(rect.min + Vec2::new(4.0, 2.0), who, size + 1.0, if chosen { ROOT } else { NODE }.a(alpha), Align2::LEFT_TOP);
        let what = format!("residual of {} fitted from it", name(truth, i, tilde));
        let what = fitting(f, &[what.as_str(), "its residual"], size - 0.5, rect.w() - 8.0).to_string();
        f.text_px(rect.min + Vec2::new(4.0, size + 7.0), what, size - 0.5, RESIDUAL.a(alpha), Align2::LEFT_TOP);
        let score = round.pair(j, i).map(|(h, _)| h / round.pair_unit()).unwrap_or(0.0);
        let foot = format!("score {}  (random pairing: 1)", num(score, 1));
        let plot = Rect::new(rect.min.x + 6.0, rect.min.y + 2.0 * size + 16.0, rect.max.x - 6.0, rect.max.y - size - 10.0);
        f.text_px(Vec2::new(rect.min.x + 4.0, rect.max.y - 2.0), foot, size - 0.5, pal::DIM.a(alpha), Align2::LEFT_BOTTOM);
        if plot.w() < 20.0 || plot.h() < 20.0 || x.is_empty() {
            return;
        }
        // each axis from its smallest to its largest value (within four spreads of the middle)
        let range = |v: &[f64]| {
            let sd = variance(v).sqrt();
            let m = v.iter().sum::<f64>() / v.len().max(1) as f64;
            let (lo, hi) = v.iter().fold((f64::MAX, f64::MIN), |(a, b), &q| (a.min(q), b.max(q)));
            let (lo, hi) = (lo.max(m - 4.0 * sd), hi.min(m + 4.0 * sd));
            if hi - lo > 1e-12 {
                (lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
            } else {
                (m - 1.0, m + 1.0)
            }
        };
        let ((xl, xh), (yl, yh)) = (range(x), range(&y));
        let to = |a: f64, b: f64| Vec2::new(plot.min.x + ((a - xl) / (xh - xl)) as f32 * plot.w(), plot.max.y - ((b - yl) / (yh - yl)) as f32 * plot.h());
        // the axes: the candidate across, the residual up, through 0
        let zero = to(0.0, 0.0);
        let zy = zero.y.max(plot.min.y).min(plot.max.y);
        hud_line(f, Vec2::new(plot.min.x, zy), Vec2::new(plot.max.x, zy), 1.0, ROOT.a(0.5 * alpha));
        hud_line(f, Vec2::new(plot.min.x, plot.min.y), Vec2::new(plot.min.x, plot.max.y), 1.0, RESIDUAL.a(0.5 * alpha));
        f.text_px(Vec2::new(plot.max.x, zy + 2.0), name(truth, j, tilde), size, ROOT.a(alpha), Align2::RIGHT_TOP);
        let dot = if x.len() < 60 { 5.0 } else if x.len() < 200 { 4.0 } else { 3.4 };
        for (a, b) in x.iter().zip(&y) {
            let p = to(*a, *b);
            if p.is_finite() && plot.contains(p) {
                f.dot(f.hud_at(2, p), dot, pal::TEXT.a(0.7 * alpha));
            }
        }
        // the spread of each slice: from a tenth of the way up its residuals to nine tenths
        let slices = 5;
        for s in 0..slices {
            let (a, b) = (xl + (xh - xl) * s as f64 / slices as f64, xl + (xh - xl) * (s + 1) as f64 / slices as f64);
            let mut inside: Vec<f64> = x.iter().zip(&y).filter(|(q, _)| **q >= a && **q < b).map(|(_, v)| *v).collect();
            if inside.len() < 6 {
                continue;
            }
            inside.sort_by(f64::total_cmp);
            let at = |share: f64| inside[((inside.len() - 1) as f64 * share).round() as usize];
            let mid = (a + b) * 0.5;
            let (lo, hi) = (to(mid, at(0.1)), to(mid, at(0.9)));
            hud_line(f, lo, hi, 2.0, RESIDUAL.a(0.8 * alpha));
            hud_line(f, lo - Vec2::X * 6.0, lo + Vec2::X * 6.0, 2.0, RESIDUAL.a(0.9 * alpha));
            hud_line(f, hi - Vec2::X * 6.0, hi + Vec2::X * 6.0, 2.0, RESIDUAL.a(0.9 * alpha));
        }
    }

    /// The strengths fitted, one bar for each arrow that could be (earlier to later in
    /// the order), with the threshold as a dashed line that can be dragged; at the end
    /// each bar is coloured by how its arrow compares with the truth.
    fn draw_strengths(&mut self, f: &mut Frame, rect: Rect, lay: &Layout, alpha: f32, verdict_in: f32) {
        let truth = *self.the_truth();
        let Some(fit) = self.search.fit.clone() else { return };
        let size = lay.small;
        hud_panel(f, rect.min - Vec2::splat(4.0), rect.max + Vec2::splat(4.0));
        let head = fitting(f, &["Strength of each arrow that could be, from earlier to later", "Strength of each possible arrow", "Strengths"], size, rect.w() - 8.0);
        f.text_px(rect.min + Vec2::new(4.0, 2.0), head, size, pal::TEXT.a(alpha), Align2::LEFT_TOP);
        let sub = fitting(f, &["(dashed: the threshold; drag its dot. Weaker arrows are dropped)", "(dashed: the threshold; drag its dot)", ""], size - 1.0, rect.w() - 8.0);
        f.text_px(rect.min + Vec2::new(4.0, size + 7.0), sub, size - 1.0, pal::DIM.a(alpha), Align2::LEFT_TOP);
        let top = rect.min.y + 2.0 * size + 22.0;
        let axis = rect.max.y - size - 8.0;
        let rows = fit.len().max(1) as f32;
        let row_h = ((axis - top) / rows).min(40.0).max(6.0);
        let label_w = 62.0;
        let value_w = 46.0;
        let (x0, x1) = (rect.min.x + label_w, rect.max.x - value_w);
        let largest = fit.iter().fold(0.0f64, |m, a| m.max(a.2.abs())).max(1.0) as f32;
        let scale = (x1 - x0) / (largest * 1.05);
        let t = self.threshold as f64;
        let verdicts = compare(&truth.seen_arrows(), &fit, t);
        let verdict_of = |a: usize, b: usize| verdicts.iter().find(|v| (v.0, v.1) == (a, b) || (v.2 == Verdict::Missed && (v.0, v.1) == (b, a))).map(|v| v.2);
        for (k, &(a, b, s)) in fit.iter().enumerate() {
            let y = top + (k as f32 + 0.5) * row_h;
            let kept = s.abs() >= t;
            let v = verdict_of(a, b);
            let color = match (kept, v) {
                (true, Some(v)) => KEPT.mix(verdict_color(v), verdict_in),
                (true, None) => KEPT,
                (false, _) => pal::FAINT,
            };
            let label = format!("${} \\to {}$", LETTERS[truth.letter(a)], LETTERS[truth.letter(b)]);
            let missed = !kept && v == Some(Verdict::Missed);
            if row_h >= 10.0 {
                let named = if missed { color.mix(TRUTH, 0.7 * verdict_in) } else { color };
                f.text_px(Vec2::new(rect.min.x + 6.0, y), label, size, named.a(alpha), Align2::LEFT_CENTER);
            }
            let w = (s.abs() as f32 * scale).min(x1 - x0);
            let half = (row_h * 0.3).min(10.0);
            hud_rect(f, Vec2::new(x0, y - half), Vec2::new(x0 + w, y + half), color.a(alpha * if kept { 0.9 } else { 0.6 }));
            if row_h >= 10.0 {
                f.text_px(Vec2::new(rect.max.x - 4.0, y), num(s, 2), size, color.a(alpha), Align2::RIGHT_CENTER);
            }
            if row_h >= 10.0 {
                // (unseen unless the arrow is a true one dropped, but always there, so
                // that the labels keep their order while the threshold is dragged)
                let shown = if missed && verdict_in > 0.0 { "missed" } else { "" };
                f.text_px(Vec2::new(x0 + w + 6.0, y), shown, size - 1.0, TRUTH.a(0.85 * alpha * verdict_in), Align2::LEFT_CENTER);
            }
        }
        hud_line(f, Vec2::new(x0, axis), Vec2::new(x1, axis), 1.0, pal::DIM.a(0.7 * alpha));
        f.text_px(Vec2::new(x0, axis + 3.0), "0", size - 1.0, pal::DIM.a(alpha), Align2::CENTER_TOP);
        // the threshold: a dashed line, and a handle on it to drag
        let tx = x0 + self.threshold * scale;
        let mut y = top - 6.0;
        while y < axis {
            hud_line(f, Vec2::new(tx, y), Vec2::new(tx, (y + 5.0).min(axis)), 1.4, CHANCE.a(0.85 * alpha));
            y += 9.0;
        }
        f.text_px(Vec2::new(tx + 4.0, axis + 3.0), num(t, 2), size - 1.0, CHANCE.a(alpha), Align2::LEFT_TOP);
        if alpha > 0.5 {
            let handle = Vec2::new(tx, top - 8.0);
            if let Some(mut world) = f.cam.pick_ground(handle * f.px).map(|p| p.truncate()) {
                if f.drag2(1, &mut world, CHANCE) {
                    if let Some(px) = on_screen(f, world.extend(0.0)) {
                        self.threshold = ((px.x - x0) / scale).max(0.0).min(THRESHOLD_MAX.max(largest));
                    }
                }
            }
        }
    }

    /// Each variable's fit from those before it in the order, as an equation, the terms
    /// dropped by the threshold faint; and at the end the comparison, arrow by arrow.
    fn draw_fits(&self, f: &mut Frame, rect: Rect, lay: &Layout, alpha: f32, verdict_in: f32) {
        let truth = self.the_truth();
        let Some(fit) = self.search.fit.as_ref() else { return };
        let size = lay.small;
        hud_panel(f, rect.min - Vec2::splat(4.0), rect.max + Vec2::splat(4.0));
        let t = self.threshold as f64;
        let verdicts = compare(&truth.seen_arrows(), fit, t);
        // the comparison's rows: each kind of verdict there is, and "found" always (the
        // kinds there are none of too, where there is room)
        let kinds = [(Verdict::Found, "found"), (Verdict::Reversed, "the wrong way round"), (Verdict::Missed, "missed"), (Verdict::Extra, "extra: not in the true graph")];
        let present = kinds.iter().filter(|(v, _)| *v == Verdict::Found || verdicts.iter().any(|x| x.2 == *v)).count();
        let line = (size + 10.0).min((rect.h() * 0.5) / self.search.order.len().max(1) as f32).max(size + 3.0);
        let equations_h = size + 14.0 + line * self.search.order.len() as f32 + size + 4.0;
        let comparison_h = |rows: usize| 2.0 * size + 28.0 + rows as f32 * (size + 8.0);
        let all_kinds = equations_h + comparison_h(4) <= rect.h() || (verdict_in > 0.0 && comparison_h(4) <= rect.h() - 4.0 && equations_h + comparison_h(present) > rect.h());
        // (on a small canvas, the comparison takes the place of the equations once it comes)
        let equations = verdict_in <= 0.0 || equations_h + comparison_h(present) <= rect.h();
        let mut y = rect.min.y + 2.0;
        if equations {
            let head = fitting(f, &["Each variable fitted from those before it in the order", "Each fitted from those before it", "The fits"], size, rect.w() - 8.0);
            f.text_px(Vec2::new(rect.min.x + 4.0, y), head, size, pal::TEXT.a(alpha), Align2::LEFT_TOP);
            y += size + 12.0;
            for (k, &v) in self.search.order.iter().enumerate() {
                let me = LETTERS[truth.letter(v)];
                let mut terms = String::new();
                for &(a, _, s) in fit.iter().filter(|a| a.1 == v) {
                    let term = format!(r"{}\,{}", num(s.abs(), 2), LETTERS[truth.letter(a)]);
                    let negative = num(s, 2).starts_with('−');
                    let sign = match (terms.is_empty(), negative) {
                        (true, true) => "-",
                        (true, false) => "",
                        (false, true) => " - ",
                        (false, false) => " + ",
                    };
                    // (a term weaker than the threshold, faint: it is dropped)
                    terms += &if s.abs() >= t { format!("{sign}{term}") } else { tint(pal::FAINT, &format!("{sign}{term}")) };
                }
                let eq = if k == 0 { format!("${me} = e_{{{me}}}$") } else { format!("${me} = {terms} + e_{{{me}}}$") };
                f.text_px(Vec2::new(rect.min.x + 8.0, y), eq, size + 1.0, pal::TEXT.a(alpha), Align2::LEFT_TOP);
                y += line;
            }
            let note = fitting(f, &[r"faint: weaker than the threshold, so dropped;  $e$: each variable's own noise", "faint: dropped by the threshold", ""], size - 1.0, rect.w() - 8.0);
            f.text_px(Vec2::new(rect.min.x + 8.0, y + 2.0), note, size - 1.0, pal::DIM.a(alpha), Align2::LEFT_TOP);
            y += size + 18.0;
            hud_line(f, Vec2::new(rect.min.x + 4.0, y - 8.0), Vec2::new(rect.max.x - 4.0, y - 8.0), 1.0, pal::FAINT.a(verdict_in));
        }
        if verdict_in <= 0.0 {
            return;
        }
        let head = fitting(f, &["Compared with the true graph, arrow by arrow", "Compared with the true graph"], size, rect.w() - 8.0);
        f.text_px(Vec2::new(rect.min.x + 4.0, y), head, size, pal::TEXT.a(verdict_in), Align2::LEFT_TOP);
        y += size + 10.0;
        // (the list starts after the longest name of a kind, the short name of extra
        // arrows on a narrow panel)
        let short = rect.w() < 420.0;
        let indent = kinds.iter().map(|(v, what)| f.text_size(&format!("{}:", if short && *v == Verdict::Extra { "extra" } else { what }), size).x).fold(0.0, f32::max) + 14.0;
        for (v, what) in kinds {
            let those: Vec<String> = verdicts.iter().filter(|x| x.2 == v).map(|&(a, b, _)| format!(r"{} \to {}", LETTERS[truth.letter(a)], LETTERS[truth.letter(b)])).collect();
            if those.is_empty() && v != Verdict::Found && !all_kinds {
                continue;
            }
            let list = if those.is_empty() { "none".to_string() } else { format!("${}$", those.join(r",\; ")) };
            let what = if short && v == Verdict::Extra { "extra" } else { what };
            f.text_px(Vec2::new(rect.min.x + 8.0, y), format!("{what}:"), size, verdict_color(v).a(verdict_in), Align2::LEFT_TOP);
            f.text_px(Vec2::new(rect.min.x + 8.0 + indent, y), list, size, pal::TEXT.a(verdict_in), Align2::LEFT_TOP);
            y += size + 8.0;
            if y > rect.max.y - size {
                break;
            }
        }
    }

    /// The tally: one square for each run on new samples (green: the order and the
    /// graph right; green outline: the order right, the graph not; pink: the order
    /// wrong), and the counts.
    fn draw_tally(&mut self, f: &mut Frame, rect: Rect, lay: &Layout, pointer: Option<Vec2>) {
        let size = lay.small;
        let Some((done, orders, graphs)) = self.counts() else {
            let w = text_width("▶ the same on 20 new samples", size + 0.5) + 24.0;
            let b = Rect::new(rect.min.x, rect.min.y + 2.0, rect.min.x + w.min(rect.w()), rect.max.y - 2.0);
            if hud_button(f, b, "▶ the same on 20 new samples", true, pointer, size + 0.5) {
                self.start_tally();
            }
            return;
        };
        let cell = ((rect.w() * 0.55) / RUNS as f32).min(rect.h() - 8.0).max(4.0);
        let label = "20 new samples:";
        let lw = text_width(label, size) + 10.0;
        f.text_px(Vec2::new(rect.min.x, rect.min.y + rect.h() * 0.5), label, size, pal::DIM, Align2::LEFT_CENTER);
        let x0 = rect.min.x + lw;
        let y = rect.min.y + (rect.h() - cell) * 0.5;
        let runs: Vec<(bool, bool)> = self.tally.as_ref().map_or(Vec::new(), |t| t.runs.iter().map(|r| (r.0, r.0 && self.exact(&r.1))).collect());
        for k in 0..RUNS {
            let (a, b) = (Vec2::new(x0 + k as f32 * cell + 1.0, y), Vec2::new(x0 + (k + 1) as f32 * cell - 1.0, y + cell));
            match runs.get(k) {
                Some((true, true)) => hud_rect(f, a, b, TRUTH),
                Some((true, false)) => hud_outline(f, a + Vec2::splat(0.7), b - Vec2::splat(0.7), 1.4, TRUTH),
                Some((false, _)) => hud_rect(f, a, b, NAIVE),
                None => hud_outline(f, a + Vec2::splat(0.7), b - Vec2::splat(0.7), 1.0, pal::FAINT),
            }
        }
        let text = if done < RUNS {
            format!("running… {done} of {RUNS}")
        } else {
            format!("true order: {orders} of {RUNS};  whole graph: {graphs} of {RUNS}")
        };
        let tx = x0 + RUNS as f32 * cell + 12.0;
        let short = format!("order {orders}, graph {graphs}");
        let t = fitting(f, &[text.as_str(), short.as_str()], size, rect.max.x - tx);
        f.text_px(Vec2::new(tx, rect.min.y + rect.h() * 0.5), t, size, pal::TEXT, Align2::LEFT_CENTER);
    }
}

impl Scene for Discover {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 3.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.moving() || !self.search.done() || self.tally.as_ref().is_some_and(|t| t.runs.len() < RUNS) || !self.settled()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("buttons on the picture: step through    a bar: see its residuals    white dot: drag the threshold")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "Several variables, each a sum of straight-line effects of others plus noise of its own, with no loop of cause and effect. If the noises are independent of each other and none is bell-shaped, and nothing unmeasured causes two of the variables, then the order of cause and effect is fixed by the data, and enough samples reveal it (DirectLiNGAM: Shimizu and others, 2011). A root, a variable nothing else causes, is one from which every other, fitted by a line, leaves residuals independent of it. Find it, take its effect away from the rest, and repeat; then fit each variable from those before it.");
        ui::equation(ui, r"x_i = \sum_{j} b_{ij}\, x_j + e_i");
        ui::note(ui, r"$b_{ij}$ is the strength of the arrow from $x_j$ to $x_i$ (0 if there is none), and $e_i$ is $x_i$'s own noise.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Steps");
        ui.horizontal_wrapped(|ui| {
            if ui.button("◀ back").clicked() {
                self.back();
            }
            if ui.button("next ▶").clicked() {
                self.next();
            }
            if ui.button("↺ replay all").clicked() {
                self.replay();
            }
            if ui.button("the end").clicked() {
                let end = self.steps() as f32;
                (self.stage, self.goal, self.through, self.hold) = (end, end, false, 0.0);
            }
        });
        ui::note(ui, "The same buttons are on the picture. Replay plays every step, pausing after each.");

        ui::section(ui, "The data");
        let mut truth = self.truth;
        let graphs: Vec<&str> = TRUTHS.iter().map(|t| t.name).collect();
        if ui::choice(ui, &graphs, &mut truth) {
            self.truth = truth.min(TRUTHS.len() - 1);
            self.resample();
        }
        ui.checkbox(&mut self.hide, "hide the true graph until the end (guess first)");
        let mut noise = Noise::ALL.iter().position(|&n| n == self.noise).unwrap_or(0);
        let noises: Vec<&str> = Noise::ALL.iter().map(|n| n.name()).collect();
        if ui::choice(ui, &noises, &mut noise) {
            self.noise = Noise::ALL[noise.min(Noise::ALL.len() - 1)];
            self.resample();
        }
        ui::note(ui, "The shape of every variable's own noise, each with mean 0 and variance 1. With 300 samples the chain's order came out right in 200 of 200 new samples with flat noise, and with bell-shaped noise in 8 of 200, about as often as a random order (1 in 24). With flat noise and 30 samples: 53 of 200.");
        let mut n = self.n;
        if ui::slider_int_log(ui, "samples", &mut n, N_MIN..=N_MAX) {
            self.n = n.max(N_MIN).min(N_MAX);
            self.resample();
        }
        if ui.button("new sample").clicked() {
            self.seed = self.seed.wrapping_add(1);
            self.resample();
        }
        ui::slider(ui, "threshold", &mut self.threshold, 0.0..=THRESHOLD_MAX);
        ui::note(ui, "A fitted strength smaller than this, either sign, counts as no arrow. With fewer samples, strengths that should be 0 come out larger.");
        if ui.button("the same on 20 new samples").clicked() {
            self.start_tally();
        }
        ui::note(ui, "Runs the whole search again on 20 new samples with these settings, and counts how often it comes out right (the strip under the picture).");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (ROOT, "cyan: a root, placed in the order (its number beside it), with its bar and scatter"),
                (BAR, "grey-blue bars: a candidate's score, one piece for each other variable's residual"),
                (CHANCE, "white dashed line: what random pairing gives (1); later, the threshold"),
                (RESIDUAL, "orange: residuals; each bar spans the middle 80 % of a slice of them"),
                (TRUTH, "green: the true arrows, and arrows found that are in the true graph"),
                (NAIVE, "pink: an arrow found that is not there; an arrow into a variable placed too soon; a wrong order"),
                (REVERSED, "yellow: an arrow found the wrong way round"),
                (HIDDEN, "purple: a cause that is not measured"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Root:", "a variable that nothing among those still left causes.");
        ui::term(ui, "Residual:", "what is left of a variable once the best straight line through another (least squares) is taken away. It is uncorrelated with the other, but not always independent of it.");
        ui::term(ui, "Independent:", "knowing one says nothing about the other: the residuals form the same band at every value of the candidate.");
        ui::term(ui, "Score:", "how much the others' residuals depend on a candidate, added up over them. Each is measured by HSIC, which sees any kind of dependence, not only a straight-line trend. (DirectLiNGAM as published uses a kernel estimate of mutual information, a measure of the same kind.)");
        ui::term(ui, "Random pairing:", "the score the same residuals would get, on average, with their samples dealt out at random among the candidate's. Scores are written in units of it. A fitted line takes away even the slight straight-line link that random pairing leaves, so independent residuals usually score a little below 1.");
        ui::term(ui, "Causal order:", "an order of the variables in which every arrow points from earlier to later. A graph can have several.");
        ui::term(ui, "Strength:", "how much a variable moves when one before it moves by 1 and the others before it stay put: a coefficient of the least-squares fit.");
        ui::term(ui, "Hidden common cause:", "a variable that is not measured and causes two that are.");

        ui::section(ui, "Numbers");
        let truth = self.the_truth();
        let order = if self.search.done() { names(truth, &self.search.order, false) } else { "measuring…".to_string() };
        ui::readout(ui, "order found", order);
        if let (true, Some(fit)) = (self.search.done(), self.search.fit.as_ref()) {
            ui::readout(ui, "agrees with every true arrow", if truth.agrees(&self.search.order) { "yes" } else { "no" });
            let v = compare(&truth.seen_arrows(), fit, self.threshold as f64);
            let count = |x: Verdict| v.iter().filter(|a| a.2 == x).count().to_string();
            ui::readout_colored(ui, "arrows found", count(Verdict::Found), TRUTH);
            ui::readout_colored(ui, "the wrong way round", count(Verdict::Reversed), REVERSED);
            ui::readout_colored(ui, "missed", count(Verdict::Missed), TRUTH.a(0.6));
            ui::readout_colored(ui, "extra", count(Verdict::Extra), NAIVE);
        }
        let (right, all) = truth.right_orders();
        ui::readout(ui, "orders that agree with the true graph", format!("{right} of {all}"));
        if let Some((done, orders, graphs)) = self.counts() {
            ui::readout(ui, "new samples run", format!("{done} of {RUNS}"));
            ui::readout_colored(ui, "true order found", orders.to_string(), TRUTH);
            ui::readout_colored(ui, "whole graph found", graphs.to_string(), TRUTH);
        }
        ui::see_also(ui, &[("direction", "two variables: which way the arrow points"), ("lingam", "why it works for straight lines"), ("pc", "the other road: independence tests"), ("gram", "taking away a projection, one at a time")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the work owed: within the frame's allowance
        let budget = crate::clock::frame_budget(f.real_dt, 0.004);
        self.work(&budget);
        self.advance(f.dt);
        self.follow(f.real_dt);
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        if view.x < 120.0 || view.y < 100.0 {
            return;
        }
        let lay = Layout::of(view, f.top_inset);
        let pointer = f.ptr.pos.map(|p| p / f.px);
        // the buttons and bars are for clicking, not for moving the view
        let over = |r: Option<Rect>| r.zip(pointer).is_some_and(|(r, p)| r.contains(p));
        if over(Some(lay.head)) || over(lay.details) || over(lay.tally) || lay.buttons.is_some_and(|b| b.iter().any(|r| pointer.is_some_and(|p| r.contains(p)))) {
            f.handles.capture();
            f.handles.claim_clicks();
        }

        let p = self.p();
        let (k, u) = self.step();
        // the order as far as it is shown: root m is placed three quarters of the way
        // through step m (and the last one with the last root)
        let lit_at = |m: usize| self.stage >= m as f32 - 0.25;
        let mut placed = Vec::new();
        for (m, &v) in self.search.order.iter().enumerate() {
            let step = (m + 1).min(p.saturating_sub(1)).max(1);
            if lit_at(step) {
                placed.push(v);
            }
        }
        let ordering = k < p;
        let newest = if ordering && k >= 1 && placed.len() >= k { self.search.order.get(k - 1).copied() } else { None };
        let arrows_in = if k > p { 1.0 } else if k == p { smoothstep((u - 0.3) / 0.5) } else { 0.0 };
        let verdict_in = if k == p + 1 { smoothstep(u / 0.6) } else { 0.0 };
        // (the true graph shows what is left only while the order is being found)
        self.draw_truth(f, lay.truth, &lay, if ordering { &placed } else { &[] }, newest, verdict_in);
        self.draw_found(f, lay.found, &lay, &placed, newest, arrows_in, verdict_in);
        self.draw_head(f, &lay, pointer);

        // the details: the round on screen, or the strengths
        if let Some(rect) = lay.details {
            let fade_in = 1.0 - smoothstep(self.fade / 0.3);
            if ordering {
                let (r, grow, lit, slide) = if k == 0 {
                    (0, 0.0, 0.0, None)
                } else if k >= 2 && u < 0.3 {
                    (k - 2, 1.0 - smoothstep(u / 0.3), 1.0, None)
                } else if k >= 2 {
                    (k - 1, smoothstep((u - 0.3) / 0.45), smoothstep((u - 0.75) / 0.25), Some((k - 2, (u - 0.3) / 0.2)))
                } else {
                    (0, smoothstep(u / 0.7), smoothstep((u - 0.75) / 0.25), None)
                };
                self.draw_table(f, rect, &lay, r, grow, lit, slide, 1.0, pointer);
                if let Some(sc) = lay.scatters {
                    let pairs = self.scatter_pairs(r);
                    let a = (lit * fade_in).max(0.0);
                    self.draw_scatter(f, sc[0], &lay, r, pairs[0].filter(|_| a > 0.0), true, a);
                    self.draw_scatter(f, sc[1], &lay, r, pairs[1].filter(|_| a > 0.0), false, a);
                }
            } else {
                let alpha = if k == p { smoothstep(u / 0.4) } else { 1.0 };
                self.draw_strengths(f, rect, &lay, alpha, verdict_in);
                if let Some(sc) = lay.scatters {
                    self.draw_fits(f, Rect::new(sc[0].min.x, sc[0].min.y, sc[1].max.x, sc[1].max.y), &lay, alpha, verdict_in);
                }
            }
        }
        if let Some(rect) = lay.tally {
            self.draw_tally(f, rect, &lay, pointer);
        }
        // what has no name of its own in the picture, named along its bottom
        let below = lay.tally.map_or(view.y, |t| t.max.y);
        if lay.scatters.is_some() && view.y - below >= lay.small + 14.0 {
            let parts: &[&str] = if ordering {
                &["orange bars: the middle 80 % of each slice of residuals, alike all along when they are independent", "orange bars: the middle 80 % of each slice"]
            } else {
                &["green: in the true graph   yellow: the wrong way round   pink: not in the true graph   dashed: missed", "green: right   yellow: reversed   pink: extra"]
            };
            let text = fitting(f, parts, lay.small, view.x - 32.0);
            if k == p + 1 || ordering {
                f.caption_px(Vec2::new(16.0, view.y - 8.0), text, lay.small, pal::DIM, Align2::LEFT_BOTTOM);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kept table gives exactly what `Hsic` gives, and its level for random
    /// pairing is the average of many shuffles.
    #[test]
    fn the_kept_table_is_hsic() {
        use crate::math::causal::Hsic;
        let mut rng = Rng::new(4);
        for n in [2usize, 3, 17, 150, 333] {
            let x: Vec<f64> = (0..n).map(|_| Noise::Uniform.sample(&mut rng)).collect();
            let y: Vec<f64> = x.iter().map(|v| v * v + 0.3 * Noise::Laplace.sample(&mut rng)).collect();
            let (value, chance) = Likeness::of(&x).against(&y);
            let h = Hsic::new(&x, &y);
            assert!((value - h.value()).abs() <= 1e-12 * h.value().abs().max(1e-12), "n {n}: {value} against {}", h.value());
            if n >= 17 {
                let shuffles = 4000;
                let mean = (0..shuffles).map(|_| h.shuffled(&mut rng)).sum::<f64>() / shuffles as f64;
                assert!((chance - mean).abs() < 0.03 * mean, "n {n}: {chance} against {mean}");
            }
        }
    }

    /// An experiment as its button sets it up, with all the work done and the steps
    /// played to rest.
    fn run(k: usize) -> Discover {
        let mut s = Discover::create();
        s.set_up(k);
        s.finish_all();
        for _ in 0..60 * 60 {
            s.advance(1.0 / 60.0);
        }
        assert!(!s.moving());
        s
    }

    /// The scores of a round, in its unit (1: what random pairing gives), by candidate.
    fn scores(s: &Discover, r: usize) -> Vec<(usize, f64)> {
        let round = &s.search.rounds[r];
        round.left.iter().map(|&j| (j, round.score(j).unwrap().0 / round.unit())).collect()
    }

    fn letters(s: &Discover, vars: &[usize]) -> String {
        vars.iter().map(|&v| LETTERS[s.the_truth().letter(v)]).collect()
    }

    fn verdicts(s: &Discover) -> Vec<(String, Verdict)> {
        let fit = s.search.fit.as_ref().unwrap();
        compare(&s.the_truth().seen_arrows(), fit, s.threshold as f64).into_iter().map(|(a, b, v)| (letters(s, &[a, b]), v)).collect()
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // a chain: at each step the root scores below 1 and every other candidate above 2;
        // C, A, D placed in turn and B left last; exactly the three true arrows kept
        let s = run(0);
        assert_eq!(letters(&s, &s.search.order), "CADB");
        assert_eq!(s.search.rounds.len(), 3);
        for r in 0..3 {
            let winner = s.search.rounds[r].winner.unwrap();
            for (j, v) in scores(&s, r) {
                assert!(if j == winner { v < 1.0 } else { v > 2.0 }, "round {r}: {} scores {v}", letters(&s, &[j]));
            }
        }
        assert!(verdicts(&s).iter().all(|v| v.1 == Verdict::Found) && verdicts(&s).len() == 3);
        assert_eq!(s.search.fit.as_ref().unwrap().iter().filter(|a| a.2.abs() >= s.threshold as f64).count(), 3);
        assert!(s.stage == 5.0 && s.tally.is_none());

        // a diamond: D first; then A and C both below 1 (no arrow joins them); B last;
        // all four arrows found, none added
        let s = run(1);
        let order = letters(&s, &s.search.order);
        assert!(order == "DACB" || order == "DCAB", "{order}");
        let second: Vec<(String, f64)> = scores(&s, 1).into_iter().map(|(j, v)| (letters(&s, &[j]), v)).collect();
        assert!(second.iter().all(|(n, v)| if n == "B" { *v > 1.0 } else { *v < 1.0 }), "{second:?}");
        assert!(verdicts(&s).len() == 4 && verdicts(&s).iter().all(|v| v.1 == Verdict::Found));

        // bell-shaped noise: at every step every candidate below 1; the order D, A, C, B,
        // against two true arrows; the true order once in 20 new samples; 1 in 24 by chance
        let s = run(2);
        for r in 0..s.search.rounds.len() {
            assert!(scores(&s, r).iter().all(|v| v.1 < 1.0), "round {r}: {:?}", scores(&s, r));
        }
        assert_eq!(letters(&s, &s.search.order), "DACB");
        assert_eq!(verdicts(&s).iter().filter(|v| v.1 == Verdict::Reversed).count(), 2);
        assert_eq!(s.counts().map(|c| (c.0, c.1)), Some((20, 1)));
        assert_eq!(s.the_truth().right_orders(), (1, 24));

        // a hidden common cause: the order agrees with every true arrow; the fit adds
        // A -> D and B -> D; in 20 new samples the order always right, the graph never
        let s = run(3);
        assert!(s.the_truth().agrees(&s.search.order));
        let v = verdicts(&s);
        let extra: Vec<&str> = v.iter().filter(|x| x.1 == Verdict::Extra).map(|x| x.0.as_str()).collect();
        assert!(extra.len() == 2 && extra.contains(&"AD") && extra.contains(&"BD"), "{v:?}");
        assert_eq!(v.iter().filter(|x| x.1 == Verdict::Found).count(), 2);
        assert_eq!(s.counts(), Some((20, 20, 0)));
        assert!(s.stage == 5.0);
        assert!(s.reasoning(5).contains("$A$ and $D$ share a cause, $H$, that is not measured") && s.reasoning(5).contains("Other extra arrows"), "{}", s.reasoning(5));

        // thirty samples: at the second step no candidate below 1, and D placed before A;
        // in 20 new samples the true order 5 times and the whole graph once
        let s = run(4);
        assert!(scores(&s, 1).iter().all(|v| v.1 > 1.0), "{:?}", scores(&s, 1));
        assert_eq!(letters(&s, &s.search.order), "CDAB");
        assert_eq!(s.counts(), Some((20, 5, 1)));
    }

    /// Why taking a root's effect away is right: what is left of the others follows the
    /// same arrows among themselves, with the same strengths (as the steps say).
    #[test]
    fn what_is_left_is_a_model_of_the_same_kind() {
        let s = run(0);
        let round = &s.search.rounds[1];
        let [a, b, d] = [0usize, 1, 3];
        let line = |x: usize, y: usize| crate::math::causal::line(&round.now[x], &round.now[y]).0;
        // the chain A -> D -> B among what is left, with strengths -0.7 and 0.8
        assert!((line(a, d) + 0.7).abs() < 0.08, "{}", line(a, d));
        let fit = regress(&[&round.now[d], &round.now[a]], &round.now[b]).unwrap();
        assert!((fit[0] - 0.8).abs() < 0.08 && fit[1].abs() < 0.08, "{fit:?}");
        // and C, taken away, is not in it at all
        assert!(round.now[2].is_empty() && !round.left.contains(&2));
    }

    /// The words about the scores: independent residuals score below 1 (random
    /// pairing) more often than not, as a fitted line takes away even the chance
    /// straight-line link; a candidate with a cause still left scores above 2 more often
    /// than not.
    #[test]
    fn roots_score_below_random_pairing() {
        let truth = TRUTHS[0];
        let arrows = truth.seen_arrows();
        let (mut roots, mut others) = (Vec::new(), Vec::new());
        for seed in 0..12 {
            let mut search = Search::new(truth.sample(Noise::Uniform, 200, 300 + seed));
            search.run();
            for round in &search.rounds {
                for &j in &round.left {
                    let v = round.score(j).unwrap().0 / round.unit();
                    if arrows.iter().any(|a| a.1 == j && round.left.contains(&a.0)) {
                        others.push(v);
                    } else {
                        roots.push(v);
                    }
                }
            }
        }
        let median = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (r, o) = (median(&mut roots), median(&mut others));
        assert!(r < 1.0 && r > 0.3 && o > 2.0, "roots {r}, others {o}");
    }

    /// The steps: next, back and replay go where they say, at any frame rate, and the
    /// scene says it is still exactly when nothing moves.
    #[test]
    fn the_steps_play_and_come_to_rest() {
        let mut s = Discover::create();
        s.finish_all();
        assert!(s.stage == 1.0 && !s.moving() && s.step() == (1, 1.0) && s.steps() == 5);
        // a replay takes the same time at every frame rate
        let whole = HOLD * 0.6 + 5.0 * STEP_TIME + 4.0 * HOLD;
        for hz in [30.0f32, 60.0, 144.0, 240.0] {
            let mut s = Discover::create();
            s.finish_all();
            s.replay();
            let (mut t, dt) = (0.0, 1.0 / hz);
            while s.moving() && t < 60.0 {
                s.advance(dt);
                t += dt;
            }
            assert!((t - whole).abs() <= dt * 1.01 && s.stage == 5.0, "{hz} Hz: {t} s, not {whole}");
        }
        // one step on, one back; the end is the end
        s.next();
        s.advance(STEP_TIME + 0.01);
        assert!(s.stage == 2.0 && !s.moving());
        s.back();
        s.advance(BACK_TIME + 0.01);
        assert!(s.stage == 1.0);
        for _ in 0..9 {
            s.next();
        }
        s.advance(60.0);
        assert!(s.stage == 5.0 && !s.can_go_on());
        // a preset with five variables has six steps, and the stage is kept within them
        s.truth = 1;
        s.resample();
        assert_eq!(s.steps(), 6);
        s.truth = 0;
        s.resample();
        assert!(s.stage <= 5.0 && s.goal <= 5.0);
        // what is drawn comes to rest, exactly
        let mut s = Discover::create();
        s.finish_all();
        for _ in 0..600 {
            s.follow(1.0 / 60.0);
        }
        assert!(s.settled() && !s.follow(1.0 / 60.0));
    }

    /// Every preset: the number of orders that agree with it, and a search that ends.
    #[test]
    fn every_true_graph_is_searched() {
        let wanted = [(1, 24), (11, 120), (2, 24), (6, 24), (24, 24)];
        for (truth, want) in TRUTHS.iter().zip(wanted) {
            assert_eq!(truth.right_orders(), want, "{}", truth.name);
            let mut search = Search::new(truth.sample(Noise::Uniform, 60, 3));
            search.run();
            let p = truth.count();
            assert_eq!(search.order.len(), p);
            assert_eq!(search.fit.as_ref().unwrap().len(), p * (p - 1) / 2);
            assert!(search.fit.as_ref().unwrap().iter().all(|a| a.2.is_finite()));
        }
    }

    /// Degenerate data: copies, a constant, no samples, one sample, every arrow kept
    /// or dropped. Nothing panics, every number is finite, and ties go alphabetically.
    #[test]
    fn degenerate_data_leaves_everything_finite() {
        let base = TRUTHS[0].sample(Noise::Uniform, 50, 2);
        let mut copies = base.clone();
        copies[1] = copies[0].clone();
        let mut constant = base.clone();
        constant[2] = vec![1.5; 50];
        let independent = TRUTHS[4].sample(Noise::Uniform, 50, 2);
        for (name, data) in [("copies", copies), ("constant", constant), ("none", vec![Vec::new(); 4]), ("one", vec![vec![0.3]; 4]), ("two", vec![vec![0.3, -0.1]; 4]), ("independent", independent)] {
            let mut search = Search::new(data);
            search.run();
            assert_eq!(search.order.len(), 4, "{name}");
            for round in &search.rounds {
                for &j in &round.left {
                    let (h, c) = round.score(j).unwrap();
                    assert!(h.is_finite() && c.is_finite() && round.unit().is_finite() && round.unit() > 0.0, "{name}");
                }
            }
            let fit = search.fit.unwrap();
            assert!(fit.iter().all(|a| a.2.is_finite()), "{name}: {fit:?}");
            for t in [0.0, THRESHOLD as f64, 100.0] {
                let v = compare(&TRUTHS[0].seen_arrows(), &fit, t);
                assert!(v.len() >= 3, "{name}");
                if t > 10.0 {
                    assert!(v.iter().all(|x| x.2 == Verdict::Missed), "{name}");
                }
            }
        }
        // two exact copies score nothing against each other: a tie, broken alphabetically
        let x: Vec<f64> = (0..40).map(|i| ((i * 37) % 17) as f64 - 8.0).collect();
        let mut search = Search::new(vec![x.clone(), x.clone()]);
        search.run();
        assert_eq!(search.order, [0, 1]);
        assert!(search.rounds[0].score(0).unwrap().0.abs() < 1e-12 && search.rounds[0].score(1).unwrap().0.abs() < 1e-12);
        // and the scene, set to the extremes, says what happened in finite numbers
        let mut s = Discover::create();
        (s.n, s.threshold) = (N_MIN, THRESHOLD_MAX);
        s.resample();
        s.finish_all();
        for k in 0..=s.steps() {
            assert!(!s.title(k).contains("NaN") && !s.reasoning(k).contains("NaN") && !s.reasoning(k).contains('?'), "{}", s.reasoning(k));
        }
        assert!(s.reasoning(5).contains("Missed: "), "{}", s.reasoning(5));
    }

    /// The layout keeps every part on the canvas, below the application's buttons, and
    /// out of the others' way.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (x, y) in [(1078.0f32, 1000.0f32), (578.0, 700.0), (1100.0, 700.0), (390.0, 800.0), (390.0, 754.0), (360.0, 640.0), (844.0, 390.0), (640.0, 360.0), (2000.0, 1100.0), (900.0, 520.0), (300.0, 240.0), (240.0, 180.0)] {
            for inset in [0.0, 46.0] {
                let lay = Layout::of(Vec2::new(x, y), inset);
                let mut parts = vec![lay.truth, lay.found, lay.head];
                parts.extend(lay.buttons.into_iter().flatten());
                parts.extend(lay.words);
                parts.extend(lay.details);
                parts.extend(lay.scatters.into_iter().flatten());
                parts.extend(lay.tally);
                for r in &parts {
                    assert!(r.min.x >= 0.0 && r.max.x <= x && r.max.y <= y + 0.5 && r.min.y >= inset.min(y) - 0.5 && r.w() >= 0.0 && r.h() >= 0.0, "{x}x{y}, inset {inset}: {r:?} in {lay:?}");
                }
                let apart = |a: &Rect, b: &Rect| a.max.x <= b.min.x + 0.5 || b.max.x <= a.min.x + 0.5 || a.max.y <= b.min.y + 0.5 || b.max.y <= a.min.y + 0.5;
                let mut blocks = vec![lay.truth, lay.found];
                blocks.extend(lay.words);
                blocks.extend(lay.details);
                blocks.extend(lay.scatters.into_iter().flatten());
                blocks.extend(lay.tally);
                for i in 0..blocks.len() {
                    for j in 0..i {
                        assert!(apart(&blocks[i], &blocks[j]), "{x}x{y}: {:?} and {:?} overlap", blocks[i], blocks[j]);
                    }
                }
                // a desktop and a phone have everything
                if (x, y) == (1078.0, 1000.0) || (x, y) == (390.0, 800.0) {
                    assert!(lay.details.is_some() && lay.scatters.is_some() && lay.tally.is_some() && lay.words.is_some(), "{x}x{y}: {lay:?}");
                }
            }
        }
    }

    /// The numbers behind the panel's note on the noise (run with --ignored): over 200
    /// new samples of the chain with 300 samples each, the order is right every time
    /// with flat noise, and with bell-shaped noise about as often as a random order.
    #[test]
    #[ignore]
    fn how_often_the_order_is_right() {
        let rate = |noise: Noise, n: usize| -> usize {
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..200u64)
                    .map(|seed| {
                        scope.spawn(move || {
                            let mut search = Search::new(TRUTHS[0].sample(noise, n, 1000 + seed));
                            search.run();
                            TRUTHS[0].agrees(&search.order) as usize
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap_or(0)).sum()
            })
        };
        let (flat, bell, thirty) = (rate(Noise::Uniform, 300), rate(Noise::Normal, 300), rate(Noise::Uniform, 30));
        println!("flat {flat}, bell {bell}, thirty {thirty} of 200");
        assert_eq!(flat, 200);
        assert!((3..=15).contains(&bell), "{bell}");
        assert!((40..=70).contains(&thirty), "{thirty}");
    }

    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            let truth = s.the_truth();
            println!("{}: order {} agrees {} counts {:?}", TRIALS[k].0, letters(&s, &s.search.order), truth.agrees(&s.search.order), s.counts());
            for r in 0..s.search.rounds.len() {
                println!("   round {r}: {:?}", scores(&s, r).iter().map(|(j, v)| format!("{} {:.2}", letters(&s, &[*j]), v)).collect::<Vec<_>>());
            }
            for step in 0..=s.steps() {
                println!("   {step}: {} | {}", s.title(step), s.reasoning(step));
            }
        }
    }
}

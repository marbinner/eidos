//! Discovery from independence: the PC algorithm of Spirtes and Glymour, step by step.

use super::*;
use crate::math::causal::{residuals, variance};
use crate::math::linalg::solve;

pub const DEF: SceneDef = SceneDef {
    id: "pc",
    title: "Discovery from independence",
    group: GROUP,
    blurb: "How far independences alone can orient a graph, and where they must stop.",
    make: |_| Box::new(Pc::create()),
};

/// The most variables a graph here has.
const MAX: usize = 5;
const NAMES: [&str; MAX] = ["A", "B", "C", "D", "E"];
/// The same as they are written in the discs (variables, so typeset).
const DISC_NAMES: [&str; MAX] = ["$A$", "$B$", "$C$", "$D$", "$E$"];

/// A set of variables, one bit each (bit k for variable k).
type Set = u8;

fn members(set: Set) -> impl Iterator<Item = usize> {
    (0..MAX).filter(move |k| set & (1 << k) != 0)
}

/// The verb "explains" or "explain", as the set has one variable or more.
fn explain_verb(set: Set) -> &'static str {
    if set.count_ones() > 1 {
        "explain"
    } else {
        "explains"
    }
}

/// The set as words: "nothing", "B", "B and D", "A, B and D".
fn set_words(set: Set) -> String {
    let names: Vec<&str> = members(set).map(|k| NAMES[k]).collect();
    match names.len() {
        0 => "nothing".to_string(),
        1 => format!("${}$", names[0]),
        n => format!("${}$ and ${}$", names[..n - 1].join(", "), names[n - 1]),
    }
}

/// A true graph to draw samples from.
struct Preset {
    /// Its name on the button.
    name: &'static str,
    /// How many variables it has (the first of A … E).
    count: usize,
    /// Where each variable sits in its picture: from (0, 0) at the top left to (1, 1).
    at: [(f32, f32); MAX],
    /// The arrows (from, to) and how strongly each acts at first.
    arrows: &'static [(usize, usize, f64)],
}

/// Three variables, the middle one on top, so that the chain, the fork and the collider
/// are drawn alike and can be compared.
const TRIANGLE: [(f32, f32); MAX] = [(0.0, 1.0), (0.5, 0.0), (1.0, 1.0), (0.5, 0.5), (0.5, 0.5)];
/// Four variables at the corners of a kite, so that no link of the six passes through a
/// variable it does not join: the third (B) on top, as in the triangle.
const KITE_TOP: [(f32, f32); MAX] = [(0.0, 0.5), (0.5, 0.0), (1.0, 0.5), (0.5, 1.0), (0.5, 0.5)];
/// The same with the first (A) on top.
const KITE: [(f32, f32); MAX] = [(0.5, 0.0), (0.0, 0.5), (1.0, 0.5), (0.5, 1.0), (0.5, 0.5)];
/// Five at the corners of a regular pentagon (as wide as it is tall, near enough).
const PENTAGON: [(f32, f32); MAX] = [(0.5, 0.0), (0.0, 0.363), (1.0, 0.363), (0.191, 0.951), (0.809, 0.951)];

const PRESETS: [Preset; 7] = [
    Preset { name: "chain", count: 3, at: TRIANGLE, arrows: &[(0, 1, 0.8), (1, 2, 0.8)] },
    Preset { name: "fork", count: 3, at: TRIANGLE, arrows: &[(1, 0, 0.8), (1, 2, 0.8)] },
    Preset { name: "collider", count: 3, at: TRIANGLE, arrows: &[(0, 1, 0.8), (2, 1, 0.8)] },
    Preset { name: "collider with a tail", count: 4, at: KITE_TOP, arrows: &[(0, 1, 0.8), (2, 1, 0.8), (1, 3, 0.8)] },
    Preset { name: "diamond", count: 4, at: KITE, arrows: &[(0, 1, 0.8), (0, 2, 0.8), (1, 3, 0.8), (2, 3, 0.8)] },
    Preset { name: "diamond with a tail", count: 5, at: PENTAGON, arrows: &[(0, 1, 0.8), (0, 2, 0.8), (1, 3, 0.8), (2, 3, 0.8), (3, 4, 0.8)] },
    Preset { name: "two paths that cancel", count: 3, at: TRIANGLE, arrows: &[(0, 1, 0.8), (1, 2, 0.8), (0, 2, -0.64)] },
];

/// The preset the scene opens with: the one that shows every kind of step.
const OPENING: usize = 5;

// ----- the data ------------------------------------------------------------------------

/// `n` samples of the variables of a straight-line model: each variable is the sum of
/// its causes, each times its arrow's strength, plus bell-shaped noise of its own with
/// variance 1. One vector of samples per variable.
fn sample(count: usize, arrows: &[(usize, usize, f64)], n: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut rng = Rng::new(seed);
    let mut data = vec![vec![0.0; n]; count];
    // (each variable once all its causes are made: the graphs here have no cycles)
    let mut done = [false; MAX];
    for _ in 0..count {
        for v in 0..count {
            let ready = arrows.iter().all(|&(from, to, _)| to != v || from >= count || done[from]);
            if done[v] || !ready {
                continue;
            }
            for i in 0..n {
                let pushed: f64 = arrows.iter().filter(|a| a.1 == v && a.0 < count).map(|&(from, _, b)| b * data[from][i]).sum();
                data[v][i] = pushed + rng.normal() as f64;
            }
            done[v] = true;
        }
    }
    data
}

/// The covariances of the variables, from which every partial correlation the tests
/// ask for is worked out without going over the samples again.
#[derive(Clone, Debug)]
struct Cov {
    count: usize,
    c: [[f64; MAX]; MAX],
}

impl Cov {
    fn of(data: &[Vec<f64>]) -> Cov {
        let count = data.len().min(MAX);
        let mut c = [[0.0; MAX]; MAX];
        for i in 0..count {
            for j in 0..=i {
                let v = crate::math::causal::covariance(&data[i], &data[j]);
                (c[i][j], c[j][i]) = (v, v);
            }
        }
        Cov { count, c }
    }

    /// The correlation of `i` and `j` once what the variables of `given` explain of
    /// each (by a straight-line fit) is taken away: the same number as
    /// `math::causal::partial_correlation` gives from the samples, but from the
    /// covariances alone. (With residuals e_i = x_i − w_iᵀ x_S, where Σ_SS w_i = Σ_Si,
    /// cov(e_i, e_j) = Σ_ij − w_iᵀ Σ_Sj.) As there, the plain correlation if the
    /// variables held settle nothing, and 0 if either has nothing left.
    fn partial(&self, i: usize, j: usize, given: Set) -> f64 {
        let s: Vec<usize> = members(given).filter(|&k| k < self.count && k != i && k != j).collect();
        let c = &self.c;
        let plain = |a: f64, aa: f64, bb: f64| {
            let spread = (aa * bb).sqrt();
            if spread > 0.0 && spread.is_finite() {
                (a / spread).max(-1.0).min(1.0)
            } else {
                0.0
            }
        };
        if s.is_empty() {
            return plain(c[i][j], c[i][i], c[j][j]);
        }
        let k = s.len();
        let fit = |target: usize| -> Option<Vec<f64>> {
            let mut a: Vec<f64> = s.iter().flat_map(|&p| s.iter().map(move |&q| c[p][q])).collect();
            let mut b: Vec<f64> = s.iter().map(|&p| c[p][target]).collect();
            solve(&mut a, &mut b, k)
        };
        let (Some(wi), Some(wj)) = (fit(i), fit(j)) else { return plain(c[i][j], c[i][i], c[j][j]) };
        let less = |w: &[f64], to: usize| s.iter().zip(w).map(|(&p, w)| w * c[p][to]).sum::<f64>();
        let (ij, ii, jj) = (c[i][j] - less(&wi, j), c[i][i] - less(&wi, i), c[j][j] - less(&wj, j));
        // (what is left of a variable that its held ones settle entirely is rounding)
        if ii <= 1e-12 * c[i][i] || jj <= 1e-12 * c[j][j] {
            return 0.0;
        }
        plain(ij, ii, jj)
    }
}

/// How large a partial correlation two independent variables give by chance alone, 95
/// times in 100, from `n` samples with `held` variables held: Fisher's rule that
/// √(n − held − 3) · atanh r is close to a standard normal number.
fn chance(n: usize, held: usize) -> f64 {
    let room = n as f64 - held as f64 - 3.0;
    if room > 0.0 {
        (1.96 / room.sqrt()).tanh()
    } else {
        1.0
    }
}

// ----- the algorithm ---------------------------------------------------------------------

/// What the tests have found so far: which pairs are linked, and the arrowheads.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
struct Pattern {
    count: usize,
    link: [[bool; MAX]; MAX],
    /// `head[i][j]`: the link between i and j has an arrowhead at j.
    head: [[bool; MAX]; MAX],
}

impl Pattern {
    /// Every pair joined, with no arrowheads.
    fn complete(count: usize) -> Pattern {
        let mut p = Pattern { count, ..Default::default() };
        for i in 0..count {
            for j in 0..count {
                p.link[i][j] = i != j;
            }
        }
        p
    }

    /// i → j and not the other way as well.
    fn arrow(&self, i: usize, j: usize) -> bool {
        self.link[i][j] && self.head[i][j] && !self.head[j][i]
    }

    /// A plain line: linked, no arrowhead at either end.
    fn line(&self, i: usize, j: usize) -> bool {
        self.link[i][j] && !self.head[i][j] && !self.head[j][i]
    }

    fn neighbours(&self, i: usize) -> Set {
        (0..self.count).filter(|&j| self.link[i][j]).fold(0, |s, j| s | 1 << j)
    }

    fn links(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.count).flat_map(move |i| (i + 1..self.count).map(move |j| (i, j))).filter(|&(i, j)| self.link[i][j])
    }
}

/// The rule that orients a link once the colliders are known (Meek's rules).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Rule {
    /// a → b — c, with a and c not linked: b → c, or a → b ← c would be a collider the
    /// tests did not find.
    NoNewCollider,
    /// a → c → b and a — b: a → b, or there would be a cycle.
    NoCycle,
    /// a — c → b, a — d → b, a — b, with c and d not linked: a → b, as either way
    /// round for a — c and a — d makes one of the others.
    TwoWays,
}

/// One step of the algorithm, as the scene shows it.
#[derive(Clone, Debug, PartialEq)]
enum Step {
    /// Every pair joined.
    Start,
    /// Are `x` and `y` independent given the set `given`? `r` is their partial
    /// correlation; `parted` if it is nearer 0 than the threshold, and the link goes.
    Test { x: usize, y: usize, given: Set, r: f64, parted: bool },
    /// `x` — `z` — `y` with `x` and `y` not linked: they were parted given `given` (by
    /// the test that is step `by`), and `z` is a collider if it is not in that set.
    Triple { x: usize, z: usize, y: usize, given: Set, by: usize, collider: bool },
    /// The link is oriented `from` → `to` by `rule`; `via` are the variables the rule
    /// looked at (a, then c and d), and `by` the step whose test the rule leans on (for
    /// the rule that no new collider may appear: the test that parted a and c).
    Orient { from: usize, to: usize, rule: Rule, via: [usize; 2], by: Option<usize> },
    /// Nothing more can be oriented.
    End,
}

/// A whole run of the algorithm: its steps, and what was found after each.
#[derive(Clone, Debug)]
struct Run {
    steps: Vec<Step>,
    after: Vec<Pattern>,
}

impl Run {
    fn last(&self) -> Pattern {
        self.after.last().copied().unwrap_or_default()
    }

    /// The colliders judged (x, z, y) with x < y.
    fn colliders(&self) -> Vec<(usize, usize, usize)> {
        self.steps.iter().filter_map(|s| if let Step::Triple { x, z, y, collider: true, .. } = *s { Some((x, z, y)) } else { None }).collect()
    }

    fn tests(&self) -> usize {
        self.steps.iter().filter(|s| matches!(s, Step::Test { .. })).count()
    }
}

/// The PC algorithm on `count` variables whose covariances are `cov`: a link is removed
/// when the partial correlation of its ends, given some set of the others, is nearer 0
/// than `threshold`.
///
/// The sets are tried smallest first (nothing, then one variable, then two …), drawn
/// from the variables linked to either end as the links stood when that size began
/// (the "stable" form of the algorithm, whose links do not depend on the order the pairs
/// are visited in).
fn discover(cov: &Cov, threshold: f64) -> Run {
    discover_by(cov.count, threshold, |x, y, given| cov.partial(x, y, given))
}

/// The same with the partial correlations given by `r` (x, y, the set held).
fn discover_by(count: usize, threshold: f64, mut r: impl FnMut(usize, usize, Set) -> f64) -> Run {
    let mut p = Pattern::complete(count);
    let mut steps = vec![Step::Start];
    let mut after = vec![p];
    let mut sep: [[Option<(Set, usize)>; MAX]; MAX] = [[None; MAX]; MAX];
    for size in 0..count.max(1) {
        let before = p;
        let mut any = false;
        for x in 0..count {
            for y in x + 1..count {
                if !p.link[x][y] {
                    continue;
                }
                let mut tried: Vec<Set> = Vec::new();
                'sides: for (a, b) in [(x, y), (y, x)] {
                    let others = before.neighbours(a) & !(1 << b);
                    if (others.count_ones() as usize) < size {
                        continue;
                    }
                    any = true;
                    for given in 0..=Set::MAX {
                        if given & !others != 0 || given.count_ones() as usize != size || tried.contains(&given) {
                            continue;
                        }
                        tried.push(given);
                        let r = r(x, y, given);
                        let parted = r.abs() < threshold;
                        steps.push(Step::Test { x, y, given, r, parted });
                        if parted {
                            (p.link[x][y], p.link[y][x]) = (false, false);
                            sep[x][y] = Some((given, steps.len() - 1));
                        }
                        after.push(p);
                        if parted {
                            break 'sides;
                        }
                    }
                }
            }
        }
        if !any {
            break;
        }
    }

    // the colliders: x — z — y with x and y not linked, z not in the set that parted them
    for z in 0..count {
        for x in 0..count {
            for y in x + 1..count {
                if x == z || y == z || !p.link[x][z] || !p.link[y][z] || p.link[x][y] {
                    continue;
                }
                let Some((given, by)) = sep[x][y] else { continue };
                let collider = given & (1 << z) == 0;
                if collider {
                    (p.head[x][z], p.head[y][z]) = (true, true);
                }
                steps.push(Step::Triple { x, z, y, given, by, collider });
                after.push(p);
            }
        }
    }

    // what follows: orient a plain line where the other way would make a new collider
    // or a cycle, until nothing changes
    'more: loop {
        for a in 0..count {
            for b in 0..count {
                if a == b || !p.line(a, b) {
                    continue;
                }
                let found = (0..count).find(|&c| c != b && p.arrow(c, a) && !p.link[c][b]).map(|c| (Rule::NoNewCollider, [c, c]));
                let found = found.or_else(|| (0..count).find(|&c| p.arrow(a, c) && p.arrow(c, b)).map(|c| (Rule::NoCycle, [c, c])));
                let found = found.or_else(|| {
                    (0..count).flat_map(|c| (c + 1..count).map(move |d| (c, d))).find(|&(c, d)| {
                        c != a && d != a && p.line(a, c) && p.line(a, d) && p.arrow(c, b) && p.arrow(d, b) && !p.link[c][d]
                    })
                    .map(|(c, d)| (Rule::TwoWays, [c, d]))
                });
                if let Some((rule, via)) = found {
                    p.head[a][b] = true;
                    let by = match rule {
                        Rule::NoNewCollider => sep[via[0].min(b)][via[0].max(b)].map(|s| s.1),
                        Rule::TwoWays => sep[via[0]][via[1]].map(|s| s.1),
                        Rule::NoCycle => None,
                    };
                    steps.push(Step::Orient { from: a, to: b, rule, via, by });
                    after.push(p);
                    continue 'more;
                }
            }
        }
        break;
    }
    steps.push(Step::End);
    after.push(p);
    Run { steps, after }
}

/// A graph with every link directed: for each variable, the set of its causes.
type Dag = [Set; MAX];

/// Is the graph free of cycles?
fn acyclic(count: usize, dag: &Dag) -> bool {
    let mut left: Set = (1 << count) - 1;
    while left != 0 {
        // a variable none of whose causes is left
        let Some(v) = members(left).find(|&v| dag[v] & left == 0) else { return false };
        left &= !(1 << v);
    }
    true
}

/// Its colliders x → z ← y with x and y not linked (x < y).
fn colliders_of(count: usize, dag: &Dag) -> Vec<(usize, usize, usize)> {
    let linked = |a: usize, b: usize| dag[a] & (1 << b) != 0 || dag[b] & (1 << a) != 0;
    let mut out = Vec::new();
    for z in 0..count {
        for x in 0..count {
            for y in x + 1..count {
                if dag[z] & (1 << x) != 0 && dag[z] & (1 << y) != 0 && !linked(x, y) {
                    out.push((x, z, y));
                }
            }
        }
    }
    out.sort_by_key(|&(x, z, y)| (z, x, y));
    out
}

/// Every graph on the links found whose colliders are exactly the ones the tests found:
/// the graphs that give exactly these independences (Verma and Pearl: two graphs do so
/// if and only if they have the same links and the same colliders).
fn class(count: usize, found: &Pattern, colliders: &[(usize, usize, usize)]) -> Vec<Dag> {
    let links: Vec<(usize, usize)> = found.links().collect();
    let mut want = colliders.to_vec();
    want.sort_by_key(|&(x, z, y)| (z, x, y));
    let mut out = Vec::new();
    for turn in 0u32..1 << links.len() {
        let mut dag: Dag = [0; MAX];
        for (k, &(i, j)) in links.iter().enumerate() {
            if turn & (1 << k) == 0 {
                dag[j] |= 1 << i;
            } else {
                dag[i] |= 1 << j;
            }
        }
        if acyclic(count, &dag) && colliders_of(count, &dag) == want {
            out.push(dag);
        }
    }
    out
}

/// The true graph as a `Dag`: its arrows that act at all.
fn truth_dag(arrows: &[(usize, usize, f64)]) -> Dag {
    let mut dag: Dag = [0; MAX];
    for &(from, to, b) in arrows {
        if b != 0.0 && from < MAX && to < MAX {
            dag[to] |= 1 << from;
        }
    }
    dag
}

/// Is every path between `x` and `y` blocked by the set `given` in the graph `dag`
/// (d-separation)? Then the graph makes them independent given that set, whatever the
/// strengths of its arrows. Worked out the usual way: keep only the variables asked
/// about and their ancestors, join every two causes of a common effect, forget the
/// directions, and see whether `x` can still reach `y` without passing through `given`.
fn separated(count: usize, dag: &Dag, x: usize, y: usize, given: Set) -> bool {
    let mut keep: Set = given | 1 << x | 1 << y;
    loop {
        let more = members(keep).fold(keep, |k, v| k | dag[v]);
        if more == keep {
            break;
        }
        keep = more;
    }
    let mut near = [0 as Set; MAX];
    for v in members(keep).filter(|&v| v < count) {
        let causes = dag[v] & keep;
        for c in members(causes) {
            near[v] |= 1 << c;
            near[c] |= 1 << v;
            // (the causes of one effect, married)
            near[c] |= causes & !(1 << c);
        }
    }
    let mut reached: Set = 1 << x;
    loop {
        let more = members(reached).filter(|&v| v == x || given & (1 << v) == 0).fold(reached, |r, v| r | near[v]);
        if more == reached {
            break;
        }
        reached = more;
    }
    reached & (1 << y) == 0
}

/// The pattern a graph's independences give if every test is right and no
/// independence is a coincidence: its links, the arrowheads of its colliders, and what
/// follows from them. (The algorithm with each test answered by the graph itself.)
fn true_pattern(count: usize, dag: &Dag) -> Run {
    discover_by(count, 0.5, |x, y, given| if separated(count, dag, x, y, given) { 0.0 } else { 1.0 })
}

/// The covariances the model gives exactly (as if it were sampled for ever).
fn model_cov(count: usize, arrows: &[(usize, usize, f64)]) -> Cov {
    // x = B x + e, so x = (I − B)⁻¹ e and Σ = (I − B)⁻¹ (I − B)⁻ᵀ; worked out by
    // putting each variable after its causes, as the sampling does
    let mut order = Vec::new();
    let mut done = [false; MAX];
    for _ in 0..count {
        for v in 0..count {
            if !done[v] && arrows.iter().all(|&(from, to, _)| to != v || from >= count || done[from]) {
                order.push(v);
                done[v] = true;
            }
        }
    }
    // each variable as a sum of the noises: m[v][k] is how much of noise k it carries
    let mut m = [[0.0f64; MAX]; MAX];
    for &v in &order {
        m[v][v] = 1.0;
        for &(from, to, b) in arrows {
            if to == v && from < count {
                for k in 0..count {
                    m[v][k] += b * m[from][k];
                }
            }
        }
    }
    let mut c = [[0.0; MAX]; MAX];
    for i in 0..count {
        for j in 0..count {
            c[i][j] = (0..count).map(|k| m[i][k] * m[j][k]).sum();
        }
    }
    Cov { count, c }
}

// ----- the scene -------------------------------------------------------------------------

// What the colours mean here, beside the section's own (`TRUTH` for the graph the samples
// came from, and for an arrow found as it is there; `THIRD` for the variables held).
/// The pair being tested now, its link and its partial correlation.
const NOW: Color = pal::YELLOW;
/// What the tests have found, before it is compared with the truth.
const FOUND: Color = pal::TEXT;
/// A mistake, at the end: a link that is not there, an arrow the wrong way, a link lost.
const WRONG: Color = pal::RED;
/// The variables held fixed in a test.
const HELD: Color = THIRD;
/// A variable that is not being looked at.
const PLAIN: Color = Color::hex(0xB8C2D6);

/// How long each kind of step is shown when the steps are played (seconds of scene time).
fn hold(step: &Step) -> f32 {
    match step {
        Step::Start => 1.8,
        Step::Test { parted: false, .. } => 0.9,
        Step::Test { parted: true, .. } => 2.2,
        Step::Triple { .. } => 2.6,
        Step::Orient { .. } => 3.2,
        Step::End => 0.0,
    }
}

/// How quickly the picture follows a change: the time in which it goes two thirds of
/// the way (seconds).
const EASE: f32 = 0.09;

/// At most this many of the samples are drawn in the scatter (evenly picked).
const SHOWN: usize = 400;

/// The scatter's half-width, in standard deviations. (The rare sample further out is
/// drawn at the edge.)
const REACH: f32 = 3.5;

/// What an experiment sets: the true graph (a preset with its strengths), the samples
/// and the threshold. The steps are then played from the start.
#[derive(Clone, Copy)]
struct Setup {
    preset: usize,
    n: usize,
    seed: u64,
    threshold: f32,
}

/// The opening: the graph with every kind of step, plenty of samples, the usual threshold.
const OPENING_SETUP: Setup = Setup { preset: OPENING, n: 1000, seed: 1, threshold: 0.1 };

/// Experiments: (button, what to look for, setup).
const TRIALS: [(&str, &str, Setup); 5] = [
    (
        "a chain",
        r"$A \to B \to C$. The tests keep $A$ – $B$ and $B$ – $C$, and part $A$ from $C$ only once $B$ is held. So $B$ is no collider, and no arrow can be drawn: three graphs give exactly these independences, the chain each way round and the fork $A \leftarrow B \to C$.",
        Setup { preset: 0, n: 1000, seed: 1, threshold: 0.1 },
    ),
    (
        "a collider",
        r"$A \to B \leftarrow C$. Now $A$ and $C$ are independent with nothing held, and $B$ is not in that empty set, so it is a collider: both arrows are found. This is the only graph with these independences.",
        Setup { preset: 2, n: 1000, seed: 1, threshold: 0.1 },
    ),
    (
        "a collider with a tail",
        r"The collider $A \to B \leftarrow C$ is found as before, and then one more arrow follows: $B \to D$. The other way, $D \to B$, would make $A \to B \leftarrow D$ a second collider, but the tests parted $A$ and $D$ given $B$.",
        Setup { preset: 3, n: 1000, seed: 1, threshold: 0.1 },
    ),
    (
        "thirty samples",
        r"The collider again, from 30 samples. $A$ and $C$ are independent, but from so few samples chance alone gives correlations up to $\pm 0.36$ (95 times in 100: the grey band); here $r = 0.26$, past the threshold, so the link $A$ – $C$ stays. With all three linked no collider can be found, and no arrow is: six graphs fit, the truth not among them. Fresh samples of 30 find the collider about two times in five.",
        Setup { preset: 2, n: 30, seed: 5, threshold: 0.1 },
    ),
    (
        "two paths that cancel",
        r"$A$ acts on $C$ directly ($-0.64$) and through $B$ ($0.8 \times 0.8 = 0.64$), and the two cancel: $A$ and $C$ are uncorrelated although $A$ causes $C$. The tests remove the link and call $B$ a collider, with $C \to B$ the wrong way. The method assumes that no independence comes from effects that cancel.",
        Setup { preset: 6, n: 1000, seed: 1, threshold: 0.1 },
    ),
];

/// A rectangle on the screen (logical pixels).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    min: Vec2,
    max: Vec2,
}

impl Rect {
    fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
        Rect { min: Vec2::new(x0, y0), max: Vec2::new(x1.max(x0), y1.max(y0)) }
    }

    fn size(&self) -> Vec2 {
        self.max - self.min
    }

    fn contains(&self, p: Vec2) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }
}

/// Where everything goes on a canvas: the two graphs side by side at the top, the
/// step's words and buttons under them, and under those the present test (its scatter
/// and where its partial correlation falls) beside the list of tests. On a phone held
/// upright the list goes under the scatter. On a canvas too short for all of it, the
/// scatter and the list are left out, and then the words and the buttons.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    /// A phone held upright: the list under the scatter, its words beside it.
    narrow: bool,
    /// Short words on the buttons and over the graphs.
    compact: bool,
    /// The size of the words.
    size: f32,
    /// The radius of a variable's disc.
    radius: f32,
    truth: Rect,
    found: Rect,
    words: Option<Rect>,
    buttons: Option<[Rect; 5]>,
    /// The scatter's square, the line under it where r is marked, and the room beside
    /// it (on a phone) for what the scatter is.
    scatter: Option<Rect>,
    bar: Option<Rect>,
    beside: Option<Rect>,
    list: Option<Rect>,
}

/// The height of a row of the list of tests.
fn row_height(size: f32) -> f32 {
    size + 6.0
}

/// The radius of the variables' discs in a graph drawn `w` by `h`.
fn radius_for(w: f32, h: f32) -> f32 {
    (w.min(h) * 0.075).clamp(8.0, 17.0)
}

/// Five buttons side by side from `x` along the row at `y`: equal widths across `width`,
/// or the widths given if there is room for them.
fn button_row(x: f32, y: f32, width: f32, height: f32, wide: Option<[f32; 5]>) -> [Rect; 5] {
    let gap = 6.0;
    let widths = wide.filter(|w| w.iter().sum::<f32>() + 4.0 * gap <= width).unwrap_or([(width - 4.0 * gap) / 5.0; 5]);
    let mut at = x;
    widths.map(|w| {
        let r = Rect::new(at, y, at + w, y + height);
        at += w + gap;
        r
    })
}

impl Layout {
    /// None for a canvas too small to draw anything on (one the controls cover).
    fn of(view: Vec2, inset: f32) -> Option<Layout> {
        if view.x < 160.0 || view.y < 120.0 {
            return None;
        }
        let narrow = view.x < 620.0 && view.y >= 560.0;
        if narrow {
            let (pad, size) = (10.0, 12.0);
            let top = inset + 22.0;
            let gw = (view.x - 3.0 * pad) * 0.5;
            let gh = (view.y * 0.2).clamp(120.0, 190.0);
            let truth = Rect::new(pad, top, pad + gw, top + gh);
            let found = Rect::new(2.0 * pad + gw, top, view.x - pad, top + gh);
            let words = Rect::new(pad, found.max.y + 10.0, view.x - pad, found.max.y + 92.0);
            let by = words.max.y + 4.0;
            let buttons = button_row(pad, by, view.x - 2.0 * pad, 32.0, None);
            let low = by + 32.0 + 16.0;
            let side = (view.x * 0.46).min(190.0).min(view.y - low - 150.0).max(60.0);
            let scatter = Rect::new(pad + 18.0, low, pad + 18.0 + side, low + side);
            let bar = Rect::new(scatter.min.x, scatter.max.y + 30.0, scatter.max.x, scatter.max.y + 30.0);
            let beside = Rect::new(scatter.max.x + 14.0, low, view.x - pad, bar.max.y + 14.0);
            let list = Rect::new(pad, bar.max.y + 34.0, view.x - pad, view.y - 10.0);
            let list = (list.size().y >= 3.0 * row_height(size)).then_some(list);
            return Some(Layout {
                narrow,
                compact: true,
                size,
                radius: radius_for(gw, gh).min(12.0),
                truth,
                found,
                words: Some(words),
                buttons: Some(buttons),
                scatter: Some(scatter),
                bar: Some(bar),
                beside: Some(beside),
                list,
            });
        }
        let pad = if view.x < 600.0 { 10.0 } else { 22.0 };
        let compact = view.x < 600.0;
        let size = if compact { 12.0 } else { 13.0 };
        let top = inset.max(0.0) + 32.0;
        let avail = view.y - top - 12.0;
        // under the graphs: the words and the buttons, and under those the scatter and the list
        let (words_h, buttons_h) = (78.0, 30.0);
        let below = 14.0 + words_h + 6.0 + buttons_h;
        let lower = 26.0 + 176.0;
        let gh = if avail >= 160.0 + below + lower {
            (avail * 0.3).clamp(160.0, 300.0)
        } else if avail >= 110.0 + below {
            avail - below
        } else {
            avail
        };
        let gw = (view.x - 3.0 * pad) * 0.5;
        let truth = Rect::new(pad, top, pad + gw, top + gh);
        let found = Rect::new(2.0 * pad + gw, top, view.x - pad, top + gh);
        let mut lay =
            Layout { narrow, compact, size, radius: radius_for(gw, gh), truth, found, words: None, buttons: None, scatter: None, bar: None, beside: None, list: None };
        if avail < 110.0 + below {
            return Some(lay);
        }
        let words = Rect::new(pad, found.max.y + 14.0, view.x - pad, found.max.y + 14.0 + words_h);
        let by = words.max.y + 6.0;
        lay.words = Some(words);
        lay.buttons = Some(button_row(pad, by, view.x - 2.0 * pad, buttons_h, (!compact).then_some([86.0, 86.0, 86.0, 132.0, 102.0])));
        let low = by + buttons_h + 26.0;
        let room = view.y - 16.0 - low;
        if room >= 176.0 {
            let side = (room - 100.0).min((view.x - 3.0 * pad) * 0.36).min(320.0);
            let scatter = Rect::new(pad + 22.0, low, pad + 22.0 + side, low + side);
            lay.scatter = Some(scatter);
            lay.bar = Some(Rect::new(scatter.min.x, scatter.max.y + 36.0, scatter.max.x, scatter.max.y + 36.0));
            lay.list = Some(Rect::new(scatter.max.x + 44.0, low - 4.0, view.x - pad, view.y - 14.0));
        }
        Some(lay)
    }
}

/// A button drawn over the picture; one that cannot be used now is drawn dim. (Its
/// click is seen before anything is drawn: see `Pc::frame`.)
fn hud_button(f: &mut Frame, r: Rect, text: &str, usable: bool, pointer: Option<Vec2>) {
    let over = pointer.is_some_and(|p| r.contains(p));
    let fill = match (usable, over) {
        (false, _) => Color::hex(0x141925),
        (true, true) => Color::hex(0x33405E),
        (true, false) => Color::hex(0x222B40),
    };
    hud_rect(f, r.min, r.max, fill);
    f.text_px((r.min + r.max) * 0.5, text, 12.5, if usable { pal::TEXT } else { pal::FAINT }, Align2::CENTER_CENTER);
}

/// A link between two discs, from rim to rim, with an arrowhead at either end grown
/// to `head_a` and `head_b` of its size (0 to 1); dashed if asked.
#[allow(clippy::too_many_arguments)]
fn hud_link(f: &mut Frame, a: Vec2, b: Vec2, radius: f32, width: f32, color: Color, head_a: f32, head_b: f32, dashed: bool) {
    // (two discs on top of each other, as one graph glides into another, leave no room
    // for a link: it is drawn all the same, unseen, so that what is drawn keeps its
    // parts from frame to frame and can be followed)
    let along = (b - a).normalize_or_zero();
    let (p, q) = (a + along * (radius + 3.0), b - along * (radius + 4.0));
    let length = (q - p).dot(along);
    let (p, q, length, color) = if along != Vec2::ZERO && length > 1.0 { (p, q, length, color) } else { ((a + b) * 0.5, (a + b) * 0.5, 0.0, color.a(0.0)) };
    if dashed {
        let dash = 5.0;
        let count = (length / (2.0 * dash)).ceil().max(1.0) as usize;
        for k in 0..count {
            let (s, e) = (k as f32 * 2.0 * dash, (k as f32 * 2.0 * dash + dash).min(length));
            hud_line(f, p + along * s, p + along * e, width, color);
        }
    } else {
        hud_line(f, p, q, width, color);
    }
    let side = Vec2::new(-along.y, along.x);
    let full = (4.0 + 2.5 * width).min(length * 0.45);
    // (a head not grown yet is drawn too, unseen, for the same reason)
    for (tip, toward, grown) in [(q, along, head_b), (p, -along, head_a)] {
        let grown = grown.clamp(0.0, 1.0);
        let (h, c) = (full * grown, color.a(color.0[3] as f32 / 255.0 * (grown * 4.0).min(1.0)));
        hud_line(f, tip, tip - toward * h + side * h * 0.5, width, c);
        hud_line(f, tip, tip - toward * h - side * h * 0.5, width, c);
    }
}

/// The variables' discs, as `hud_graph` draws them (a ring with the name inside), each
/// as far there as `shown` says. All five rings are always drawn, those not in the
/// graph unseen, so that the picture keeps the same parts as one graph glides into
/// another; a name is written only while its disc is more there than not (a label's
/// shadow does not fade with it).
fn discs(f: &mut Frame, at: &[Vec2; MAX], colors: &[Color; MAX], shown: &[f32; MAX], radius: f32) {
    for k in 0..MAX {
        hud_circle(f, at[k], radius, 1.8, colors[k].a(shown[k]));
        if shown[k] > 0.5 {
            f.text_px(at[k], DISC_NAMES[k], (radius * 0.95).max(11.0), colors[k].a(shown[k]), Align2::CENTER_CENTER);
        }
    }
}

/// Words broken into lines no wider than `width` (logical pixels), formulas kept whole.
fn wrap(f: &Frame, text: &str, width: f32, size: f32) -> Vec<String> {
    let mut words: Vec<&str> = Vec::new();
    let (mut start, mut inside) = (0, false);
    for (i, c) in text.char_indices() {
        match c {
            '$' => inside = !inside,
            ' ' if !inside => {
                words.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    words.push(&text[start.min(text.len())..]);
    let mut lines: Vec<String> = Vec::new();
    for word in words.into_iter().filter(|w| !w.is_empty()) {
        match lines.last_mut() {
            Some(last) if f.text_size(&format!("{last} {word}"), size).x <= width => {
                last.push(' ');
                last.push_str(word);
            }
            _ => lines.push(word.to_string()),
        }
    }
    lines
}

/// Where the variables of a picture go in the rectangle `r`, from their places in the
/// unit square `at` (scaled alike both ways, and centred), with room for discs of `radius`.
fn spots(r: Rect, at: &[Vec2; MAX], radius: f32) -> [Vec2; MAX] {
    // (room at the sides for the strengths written beside the arrows, where there is room)
    let side = if r.size().x > 220.0 { 16.0 } else { 4.0 };
    let inner = Rect::new(r.min.x + radius + side, r.min.y + radius + 4.0, r.max.x - radius - side, r.max.y - radius - 4.0);
    let (lo, hi) = (Vec2::ZERO, Vec2::new(1.0, 1.0));
    let span = (hi - lo).max(Vec2::splat(1e-3));
    let scale = (inner.size().x / span.x).min(inner.size().y / span.y).max(0.0);
    let corner = (inner.min + inner.max) * 0.5 - span * scale * 0.5;
    at.map(|p| corner + (p - lo) * scale)
}

/// The words of a partial correlation's verdict.
fn verdict(parted: bool) -> &'static str {
    if parted {
        "independent"
    } else {
        "dependent"
    }
}

/// What is drawn, following what is so.
#[derive(Clone, Debug)]
struct Shown {
    /// Each variable's place in its unit square, and how far it is there (0 to 1).
    at: [Vec2; MAX],
    node: [f32; MAX],
    /// The true arrows, i → j.
    truth: [[f32; MAX]; MAX],
    /// The links found (i < j), and the arrowheads (at j on the link from i).
    link: [[f32; MAX]; MAX],
    head: [[f32; MAX]; MAX],
    /// How far each variable is ringed as held.
    held: [f32; MAX],
    /// Where the present test's r is marked on the line under the scatter (stretched,
    /// as it is drawn: `stretch(r)`).
    r: f32,
}

/// Move `v` a share `k` of the way to `to`, exactly there once close; true if it moved.
fn ease(v: &mut f32, to: f32, k: f32, close: f32) -> bool {
    if !v.is_finite() || (*v - to).abs() <= close {
        let moved = *v != to;
        *v = to;
        return moved;
    }
    *v += (to - *v) * k;
    true
}

pub struct Pc {
    preset: usize,
    /// The strength of each arrow of the preset, in its order.
    strength: Vec<f64>,
    n: usize,
    seed: u64,
    threshold: f32,
    data: Vec<Vec<f64>>,
    cov: Cov,
    run: Run,
    /// What the tests would find if every one were right, and no independence a
    /// coincidence of strengths.
    truth: Run,
    /// What the tests would find from endless samples of the model, with every
    /// independence it has, whether its graph's shape gives it or its strengths cancel.
    unfaithful: Run,
    /// Every graph with the independences found, the true graph first if it is one.
    class: Vec<Dag>,
    truth_in_class: bool,
    /// The step shown; whether the steps are being played, and how long the present
    /// one has left.
    step: usize,
    playing: bool,
    wait: f32,
    shown: Shown,
    /// The test whose samples the scatter shows (a step), where they are going, and
    /// where they are drawn.
    focus: Option<usize>,
    target: Vec<Vec2>,
    points: Vec<Vec2>,
    /// Whether anything drawn is still on its way, and whether the threshold's handle
    /// is held.
    moving: bool,
    dragging: bool,
    /// The model's own covariances (as if sampled for ever), and how many rows of tests
    /// the list had room for when last drawn.
    model: Cov,
    room_rows: usize,
    /// Every test's r as the line under the scatter shows it at the end (stretched,
    /// smallest first, each with whether it parted its pair), and where each is drawn:
    /// when the threshold is dragged across one and the tests done change, the marks
    /// glide to their new places.
    ticks: Vec<(f32, bool)>,
    tick_at: Vec<f32>,
}

impl Pc {
    fn create() -> Pc {
        let mut s = Pc {
            preset: OPENING,
            strength: Vec::new(),
            n: 1000,
            seed: 1,
            threshold: 0.1,
            data: Vec::new(),
            cov: Cov { count: 0, c: [[0.0; MAX]; MAX] },
            run: Run { steps: Vec::new(), after: Vec::new() },
            truth: Run { steps: Vec::new(), after: Vec::new() },
            unfaithful: Run { steps: Vec::new(), after: Vec::new() },
            class: Vec::new(),
            truth_in_class: false,
            step: 0,
            playing: false,
            wait: 0.0,
            shown: Shown { at: [Vec2::splat(0.5); MAX], node: [0.0; MAX], truth: [[0.0; MAX]; MAX], link: [[0.0; MAX]; MAX], head: [[0.0; MAX]; MAX], held: [0.0; MAX], r: 0.0 },
            focus: None,
            target: Vec::new(),
            points: Vec::new(),
            moving: false,
            dragging: false,
            model: Cov { count: 0, c: [[0.0; MAX]; MAX] },
            room_rows: 12,
            ticks: Vec::new(),
            tick_at: Vec::new(),
        };
        s.apply(OPENING_SETUP);
        // (it opens on the finished run, the picture already where it is going)
        s.go_to(s.last());
        s.settle();
        s
    }

    fn count(&self) -> usize {
        PRESETS[self.preset].count
    }

    /// The true graph's arrows with the strengths set now.
    fn arrows(&self) -> Vec<(usize, usize, f64)> {
        PRESETS[self.preset].arrows.iter().zip(self.strength.iter().chain(std::iter::repeat(&0.0))).map(|(&(from, to, _), &b)| (from, to, b)).collect()
    }

    /// Set up one of the experiments, or the opening, and play its steps from the start.
    fn apply(&mut self, setup: Setup) {
        self.preset = setup.preset.min(PRESETS.len() - 1);
        self.strength = PRESETS[self.preset].arrows.iter().map(|a| a.2).collect();
        (self.n, self.seed, self.threshold) = (setup.n, setup.seed, setup.threshold);
        self.resample();
        self.rerun(false);
        (self.step, self.playing) = (0, true);
        self.wait = self.run.steps.first().map_or(0.0, hold);
    }

    fn set_up(&mut self, k: usize) {
        if let Some((_, _, setup)) = TRIALS.get(k) {
            self.apply(*setup);
        }
    }

    /// Draw new samples from the true graph as it is set now.
    fn resample(&mut self) {
        let arrows = self.arrows();
        self.data = sample(self.count(), &arrows, self.n.max(4), self.seed);
        self.cov = Cov::of(&self.data);
    }

    /// Run the tests again (after new samples, or a new threshold), and work out what
    /// the found graph allows. `keep` keeps the step shown (as near as the new run
    /// has one); otherwise the run is shown finished.
    fn rerun(&mut self, keep: bool) {
        let arrows = self.arrows();
        let count = self.count();
        self.run = discover(&self.cov, self.threshold as f64);
        self.model = model_cov(count, &arrows);
        let truth = truth_dag(&arrows);
        self.truth = true_pattern(count, &truth);
        // (what endless samples would show, independences by coincidence and all)
        self.unfaithful = discover(&self.model, 1e-9);
        self.class = class(count, &self.run.last(), &self.run.colliders());
        self.truth_in_class = match self.class.iter().position(|d| *d == truth) {
            Some(k) => {
                let first = self.class.remove(k);
                self.class.insert(0, first);
                true
            }
            None => false,
        };
        let last = self.run.steps.len().saturating_sub(1);
        if keep {
            self.step = self.step.min(last);
        } else {
            (self.step, self.playing) = (last, false);
        }
        // (the scatter's samples are new, or the test it shows is another)
        self.focus = None;
        self.ticks = self.run.steps.iter().filter_map(|s| if let Step::Test { r, parted, .. } = *s { Some((stretch(r as f32), parted)) } else { None }).collect();
        self.ticks.sort_by(|a, b| a.0.total_cmp(&b.0));
        if self.tick_at.len() != self.ticks.len() {
            self.tick_at = self.ticks.iter().map(|t| t.0).collect();
        }
    }

    /// A new sample of the same size (and everything after it).
    fn new_sample(&mut self) {
        self.seed = self.seed.wrapping_add(1);
        self.resample();
        self.rerun(false);
    }

    fn last(&self) -> usize {
        self.run.steps.len().saturating_sub(1)
    }

    fn go_to(&mut self, step: usize) {
        self.step = step.min(self.last());
        self.playing = false;
    }

    fn next(&mut self) {
        self.go_to(self.step + 1);
    }

    fn back(&mut self) {
        self.go_to(self.step.saturating_sub(1));
    }

    fn play(&mut self) {
        if self.playing {
            self.playing = false;
            return;
        }
        if self.step >= self.last() {
            self.step = 0;
        }
        self.playing = true;
        self.wait = self.run.steps.get(self.step).map_or(0.0, hold);
    }

    fn replay(&mut self) {
        self.step = 0;
        self.playing = true;
        self.wait = self.run.steps.first().map_or(0.0, hold);
    }

    /// Move the steps on by `dt` seconds of scene time while they are played.
    fn advance(&mut self, dt: f32) {
        if !self.playing {
            return;
        }
        let mut dt = dt.max(0.0);
        while self.playing && dt > 0.0 {
            if dt < self.wait {
                self.wait -= dt;
                return;
            }
            dt -= self.wait;
            if self.step >= self.last() {
                self.playing = false;
                return;
            }
            self.step += 1;
            self.wait = self.run.steps.get(self.step).map_or(0.0, hold);
            if self.step >= self.last() {
                self.playing = false;
            }
        }
    }

    /// The step whose test the scatter shows: the test being done, or the one that
    /// parted the pair a later step leans on.
    fn focus_of(&self, step: usize) -> Option<usize> {
        match self.run.steps.get(step)? {
            Step::Test { .. } => Some(step),
            Step::Triple { by, .. } => Some(*by),
            Step::Orient { by, .. } => *by,
            Step::Start | Step::End => None,
        }
    }

    /// The samples of a test as the scatter shows them: each variable less what the
    /// held ones explain of it (a straight-line fit), in standard deviations.
    fn residual_points(&self, step: usize) -> Vec<Vec2> {
        let Some(&Step::Test { x, y, given, .. }) = self.run.steps.get(step) else { return Vec::new() };
        let (Some(dx), Some(dy)) = (self.data.get(x), self.data.get(y)) else { return Vec::new() };
        let held: Vec<&[f64]> = members(given).filter_map(|v| self.data.get(v).map(|d| d.as_slice())).collect();
        let (ex, ey) = (residuals(&held, dx), residuals(&held, dy));
        let unit = |e: &[f64]| {
            let sd = variance(e).sqrt();
            if sd > 0.0 && sd.is_finite() {
                1.0 / sd
            } else {
                0.0
            }
        };
        let (ux, uy) = (unit(&ex), unit(&ey));
        let n = ex.len().min(ey.len());
        let m = n.min(SHOWN);
        (0..m).map(|k| k * n / m.max(1)).map(|i| Vec2::new((ex[i] * ux) as f32, (ey[i] * uy) as f32)).collect()
    }

    /// The targets of everything that glides, for the step shown.
    fn targets(&self) -> Shown {
        let preset = &PRESETS[self.preset];
        let count = preset.count;
        let mut t = self.shown.clone();
        for k in 0..MAX {
            t.at[k] = Vec2::from(preset.at[k]);
            t.node[k] = if k < count { 1.0 } else { 0.0 };
        }
        t.truth = [[0.0; MAX]; MAX];
        for &(from, to, b) in &self.arrows() {
            if from < MAX && to < MAX {
                t.truth[from][to] = if b != 0.0 { 1.0 } else { 0.35 };
            }
        }
        let now = self.run.after.get(self.step).copied().unwrap_or_default();
        t.link = [[0.0; MAX]; MAX];
        t.head = [[0.0; MAX]; MAX];
        for i in 0..count {
            for j in 0..count {
                t.link[i][j] = if now.link[i][j] { 1.0 } else { 0.0 };
                t.head[i][j] = if now.link[i][j] && now.head[i][j] { 1.0 } else { 0.0 };
            }
        }
        // a link the present test removes is still drawn (dashed) while that step is shown
        if let Some(&Step::Test { x, y, parted: true, .. }) = self.run.steps.get(self.step) {
            (t.link[x][y], t.link[y][x]) = (1.0, 1.0);
        }
        if let Some(focus) = self.focus_of(self.step) {
            if let Some(Step::Test { r, .. }) = self.run.steps.get(focus) {
                t.r = stretch(*r as f32);
            }
        }
        let held = match self.run.steps.get(self.step) {
            Some(&Step::Test { given, .. }) | Some(&Step::Triple { given, .. }) => given,
            _ => 0,
        };
        t.held = std::array::from_fn(|k| if held & (1 << k) != 0 && k < count { 1.0 } else { 0.0 });
        t
    }

    /// How many tests the list shows now (those done so far), the row lit (if any), and
    /// the step of each row.
    fn rows(&self) -> (Vec<usize>, Option<usize>) {
        let done = match self.run.steps.get(self.step) {
            Some(Step::Test { .. }) | Some(Step::Start) => self.step,
            _ => self.run.steps.len(),
        };
        let rows: Vec<usize> = (0..=done.min(self.last())).filter(|&k| matches!(self.run.steps.get(k), Some(Step::Test { .. }))).collect();
        let lit = self.focus_of(self.step).and_then(|f| rows.iter().position(|&k| k == f));
        (rows, lit)
    }

    /// How many rows of tests the list has room for (the last layout's).
    fn room(&self) -> usize {
        self.room_rows.max(1)
    }

    /// The first row the list shows: as many as it has room for, with the lit one in
    /// the middle, or the last ones. (The rows step rather than glide: each stays in
    /// its slot and only what it says changes.)
    fn first_row(&self) -> usize {
        let (rows, lit) = self.rows();
        let room = self.room();
        let most = rows.len().saturating_sub(room);
        lit.map_or(most, |k| k.saturating_sub(room / 2)).min(most)
    }

    /// Follow what is so, by `dt` seconds of real time.
    fn follow(&mut self, dt: f32) {
        let focus = self.focus_of(self.step);
        if focus != self.focus {
            self.focus = focus;
            self.target = focus.map(|k| self.residual_points(k)).unwrap_or_default();
            if self.points.len() != self.target.len() {
                self.points = self.target.clone();
            }
        }
        let t = self.targets();
        let k = 1.0 - (-dt.max(0.0) / EASE).exp();
        let mut moving = false;
        let s = &mut self.shown;
        for v in 0..MAX {
            let mut at = s.at[v];
            moving |= ease(&mut at.x, t.at[v].x, k, 1e-4);
            moving |= ease(&mut at.y, t.at[v].y, k, 1e-4);
            s.at[v] = at;
            moving |= ease(&mut s.node[v], t.node[v], k, 2e-3);
            for w in 0..MAX {
                moving |= ease(&mut s.truth[v][w], t.truth[v][w], k, 2e-3);
                moving |= ease(&mut s.link[v][w], t.link[v][w], k, 2e-3);
                moving |= ease(&mut s.head[v][w], t.head[v][w], k, 2e-3);
            }
            moving |= ease(&mut s.held[v], t.held[v], k, 2e-3);
        }
        moving |= ease(&mut s.r, t.r, k, 1e-4);
        for (p, q) in self.tick_at.iter_mut().zip(&self.ticks) {
            moving |= ease(p, q.0, k, 1e-4);
        }
        for (p, q) in self.points.iter_mut().zip(&self.target) {
            let (mut x, mut y) = (p.x, p.y);
            moving |= ease(&mut x, q.x, k, 1e-3);
            moving |= ease(&mut y, q.y, k, 1e-3);
            *p = Vec2::new(x, y);
        }
        self.moving = moving;
    }

    /// Everything where it is going, at once.
    fn settle(&mut self) {
        self.follow(0.0);
        self.shown = self.targets();
        self.points = self.target.clone();
        self.tick_at = self.ticks.iter().map(|t| t.0).collect();
        self.moving = false;
    }
}

/// Where a correlation goes on the line under the scatter, from −1 to 1: stretched
/// near 0 (the square root of its size), where the threshold and chance lie.
fn stretch(r: f32) -> f32 {
    let r = r.clamp(-1.0, 1.0);
    r.signum() * r.abs().sqrt()
}

/// A variable's name as a formula.
fn v(k: usize) -> String {
    format!("${}$", NAMES.get(k).copied().unwrap_or("?"))
}

impl Pc {
    /// The heading of a step and what it does, in words.
    fn explain(&self, step: usize) -> (String, String) {
        let total = self.run.steps.len();
        let threshold = self.threshold as f64;
        let head = |phase: &str| format!("Step {} of {total}  ·  {phase}", step + 1);
        match self.run.steps.get(step) {
            None | Some(Step::Start) => (
                head("every pair joined"),
                "To begin, every pair of variables is joined: nothing is known yet. A link stays until some test finds its two ends independent, alone or given some of the others.".to_string(),
            ),
            Some(&Step::Test { x, y, given, r, parted }) => {
                let held = given.count_ones();
                let phase = match held {
                    0 => "removing links, holding nothing".to_string(),
                    1 => "removing links, holding one other variable".to_string(),
                    k => format!("removing links, holding {} others", ["", "", "two", "three"][(k as usize).min(3)]),
                };
                // (to three places where two would make r and the threshold look alike)
                let places = if num(r.abs(), 2) == num(threshold, 2) { 3 } else { 2 };
                let (r, t) = (num(r, places), num(threshold, places));
                let (a, b) = (v(x), v(y));
                let text = match (held, parted) {
                    (0, false) => format!("Are {a} and {b} independent? Their correlation is $r = {r}$, further from 0 than the threshold, ${t}$: dependent. The link stays."),
                    (0, true) => format!("Are {a} and {b} independent? Their correlation is $r = {r}$, nearer 0 than the threshold, ${t}$: independent. The link goes, and nothing had to be held to part them."),
                    (_, false) => format!(
                        "Are {a} and {b} independent given {s}? Take from each what {s} {e} (a straight-line fit): what is left has $r = {r}$, further from 0 than ${t}$. The link stays.",
                        s = set_words(given),
                        e = explain_verb(given)
                    ),
                    (_, true) => format!(
                        "Are {a} and {b} independent given {s}? Take from each what {s} {e} (a straight-line fit): what is left has $r = {r}$, nearer 0 than ${t}$. Independent: the link goes, and the set that parted them, {s}, is remembered.",
                        s = set_words(given),
                        e = explain_verb(given)
                    ),
                };
                (head(&phase), text)
            }
            Some(&Step::Triple { x, z, y, given, collider, .. }) => {
                let (a, m, b) = (v(x), v(z), v(y));
                let text = if collider {
                    format!(
                        "{a} – {m} – {b}, and no link between {a} and {b}: they were parted given {s}, without {m}. Were {m} the middle of a chain or a fork, this path would carry dependence unless {m} were held. So {m} is a collider: ${} \\to {} \\leftarrow {}$.",
                        NAMES[x],
                        NAMES[z],
                        NAMES[y],
                        s = set_words(given)
                    )
                } else {
                    format!("{a} – {m} – {b}, and no link between {a} and {b}: they were parted given {s}, with {m} held. Holding a collider joins its two causes rather than parting them, so {m} is no collider: no arrowheads here.", s = set_words(given))
                };
                (head("finding the colliders"), text)
            }
            Some(&Step::Orient { from, to, rule, via, .. }) => {
                let (a, b, c, d) = (NAMES[from], NAMES[to], NAMES[via[0]], NAMES[via[1]]);
                let text = match rule {
                    Rule::NoNewCollider => format!(
                        "${c} \\to {a}$ – ${b}$, and no link between ${c}$ and ${b}$. Were it ${b} \\to {a}$, then ${c} \\to {a} \\leftarrow {b}$ would be a collider; but ${c}$ and ${b}$ were parted with ${a}$ held, so ${a}$ is no collider between them. So ${a} \\to {b}$."
                    ),
                    Rule::NoCycle => format!(
                        "${a} \\to {c} \\to {b}$, and ${a}$ – ${b}$. Were it ${b} \\to {a}$, the arrows would go round in a loop, ${a} \\to {c} \\to {b} \\to {a}$, and a variable would be among its own causes. So ${a} \\to {b}$."
                    ),
                    Rule::TwoWays => format!(
                        "${a}$ – ${c} \\to {b}$ and ${a}$ – ${d} \\to {b}$, with no link between ${c}$ and ${d}$. Were it ${b} \\to {a}$, then ${c} \\to {a}$ and ${d} \\to {a}$ (the other way, each would close a loop through ${b}$), and ${c} \\to {a} \\leftarrow {d}$ would be a collider the tests did not find. So ${a} \\to {b}$."
                    ),
                };
                (head("orienting what follows"), text)
            }
            Some(Step::End) => {
                let found = self.run.last();
                let links = found.links().count();
                let arrows = found.links().filter(|&(i, j)| found.head[i][j] || found.head[j][i]).count();
                let text = match self.class.len() {
                    0 => "Done. The tests contradict each other: no graph gives exactly these independences.".to_string(),
                    1 if links == 0 => "Done. Every pair was parted: as far as these tests can tell, no variable has anything to do with any other.".to_string(),
                    1 => "Done. Every link is oriented: this is the only graph that gives exactly these independences.".to_string(),
                    k if arrows == 0 => format!(
                        "Done: no link can be oriented. Each plain line points one way in some of the graphs that give exactly these independences and the other way in others. There are {k} such graphs."
                    ),
                    k => format!(
                        "Done: nothing more can be oriented. The arrows are the same in every graph that gives exactly these independences; each plain line points one way in some of them and the other way in others. There are {k} such graphs."
                    ),
                };
                (head("done"), text)
            }
        }
    }

    /// How a found link is coloured once the steps are done: as the truth has it
    /// (green, for an arrow), not oriented (white), or wrong (red).
    fn verdict_color(&self, i: usize, j: usize, found: &Pattern) -> Color {
        let truth = truth_dag(&self.arrows());
        let (ij, ji) = (truth[j] & (1 << i) != 0, truth[i] & (1 << j) != 0);
        if !ij && !ji {
            return WRONG;
        }
        match (found.head[i][j], found.head[j][i]) {
            (false, false) => FOUND,
            (true, false) if ij => TRUTH,
            (false, true) if ji => TRUTH,
            _ => WRONG,
        }
    }

    /// Why the true graph is not among those the tests allow: a test was wrong about
    /// the samples, or every test was right and the model itself has an independence
    /// that its graph's shape does not give (paths that cancel).
    fn why_not(&self) -> &'static str {
        if self.run.last() == self.unfaithful.last() && self.unfaithful.last() != self.truth.last() {
            "The truth is not among them, though the tests found just what endless samples would show: in this model an independence comes from paths that cancel, not from the shape of the graph."
        } else {
            "The truth is not among them: some test was wrong, calling dependent variables independent or the other way round."
        }
    }

    /// The true graph.
    fn draw_truth(&self, f: &mut Frame, lay: &Layout) {
        let r = lay.truth;
        let title = if lay.compact || r.size().x < 300.0 { "the truth" } else { "the truth: the graph the samples came from" };
        f.text_px(Vec2::new(r.min.x, r.min.y - 6.0), title, lay.size, TRUTH, Align2::LEFT_BOTTOM);
        let at = spots(r, &self.shown.at, lay.radius);
        let count = self.count();
        // (which side of its arrow a strength goes is decided on the layout the picture
        // is going to, not on the one gliding there, so that it never flips on the way)
        let layout = PRESETS[self.preset].at.map(Vec2::from);
        let middle = layout.iter().take(count).fold(Vec2::ZERO, |m, p| m + *p) / count.max(1) as f32;
        for &(from, to, b) in &self.arrows() {
            let (Some(&a), Some(&c)) = (at.get(from), at.get(to)) else { continue };
            let shown = self.shown.truth[from][to].min(self.shown.node[from]).min(self.shown.node[to]);
            let zero = b == 0.0;
            let color = if zero { TRUTH.a(0.35 * shown) } else { TRUTH.a(shown) };
            hud_link(f, a, c, lay.radius, if zero { 1.2 } else { 2.2 }, color, 0.0, 1.0, zero);
            // the strength, beside the arrow's middle, on the side away from the graph's middle
            let room = r.size().min_element();
            if shown > 0.5 && room >= 140.0 && (count <= 4 || room >= 200.0) {
                let along = (c - a).normalize_or_zero();
                let mid = (a + c) * 0.5;
                let side = Vec2::new(along.y, -along.x);
                let (ta, tc) = (layout[from.min(MAX - 1)], layout[to.min(MAX - 1)]);
                let target = Vec2::new(tc.y - ta.y, ta.x - tc.x);
                let away = target.dot((ta + tc) * 0.5 - middle);
                // away from the middle of the graph; for an arrow through the middle, to the right
                let outward = if away.abs() > 1e-3 { away > 0.0 } else { target.x > 0.0 || (target.x == 0.0 && target.y < 0.0) };
                let side = if outward == (side.dot(target) > 0.0) { side } else { -side };
                let text = num(b, 2);
                let half = f.text_size(&text, lay.size - 2.0) * 0.5;
                let off = 5.0 + side.x.abs() * half.x + side.y.abs() * half.y;
                f.text_px(mid + side * off, text, lay.size - 2.0, TRUTH.a(0.8 * shown), Align2::CENTER_CENTER);
            }
        }
        discs(f, &at, &[PLAIN; MAX], &self.shown.node, lay.radius);
    }

    /// The graph found so far, with the present step's variables lit.
    fn draw_found(&self, f: &mut Frame, lay: &Layout) {
        let r = lay.found;
        let step = self.run.steps.get(self.step);
        let done = matches!(step, Some(Step::End));
        let title = if lay.compact || r.size().x < 300.0 { "found by the tests" } else { "found by the tests, step by step" };
        f.text_px(Vec2::new(r.min.x, r.min.y - 6.0), title, lay.size, FOUND, Align2::LEFT_BOTTOM);
        let at = spots(r, &self.shown.at, lay.radius);
        let count = self.count();
        let found = self.run.after.get(self.step).copied().unwrap_or_default();
        // what this step looks at: a pair of variables and its link, and some held
        let (mut lit, mut lit_links): (Set, Vec<(usize, usize)>) = (0, Vec::new());
        match step {
            Some(&Step::Test { x, y, .. }) => {
                lit = 1 << x | 1 << y;
                lit_links.push((x, y));
            }
            Some(&Step::Triple { x, z, y, .. }) => {
                lit = 1 << x | 1 << z | 1 << y;
                lit_links.extend([(x, z), (z, y)]);
            }
            Some(&Step::Orient { from, to, via, rule, .. }) => {
                lit = 1 << from | 1 << to | 1 << via[0] | 1 << via[1];
                lit_links.push((from, to));
                lit_links.extend(match rule {
                    Rule::NoNewCollider => vec![(via[0], from)],
                    Rule::NoCycle => vec![(from, via[0]), (via[0], to)],
                    Rule::TwoWays => vec![(from, via[0]), (from, via[1]), (via[0], to), (via[1], to)],
                });
            }
            _ => {}
        }
        let removing = matches!(step, Some(Step::Test { parted: true, .. }));
        for i in 0..count {
            for j in i + 1..count {
                let shown = self.shown.link[i][j].min(self.shown.node[i]).min(self.shown.node[j]);
                let now = lit_links.iter().any(|&(a, b)| (a, b) == (i, j) || (a, b) == (j, i));
                if done && !found.link[i][j] {
                    // a link of the truth that the tests lost
                    let truth = truth_dag(&self.arrows());
                    if truth[j] & (1 << i) != 0 || truth[i] & (1 << j) != 0 {
                        hud_link(f, at[i], at[j], lay.radius, 1.4, WRONG.a(0.8), 0.0, 0.0, true);
                    }
                }
                let color = if now {
                    NOW
                } else if done {
                    self.verdict_color(i, j, &found)
                } else {
                    FOUND.a(0.8)
                };
                let dashed = now && removing;
                hud_link(f, at[i], at[j], lay.radius, if now { 2.6 } else { 2.0 }, color.a(shown), self.shown.head[j][i], self.shown.head[i][j], dashed);
            }
        }
        let colors: [Color; MAX] = std::array::from_fn(|k| if lit & (1 << k) != 0 { NOW } else { PLAIN });
        discs(f, &at, &colors, &self.shown.node, lay.radius);
        // the held variables, ringed (a ring fades from one to the next rather than jumping)
        for k in 0..MAX {
            hud_circle(f, at[k], lay.radius + 4.5, 2.2, HELD.a(self.shown.held[k] * self.shown.node[k]));
        }
    }

    /// The step's heading and words.
    fn draw_words(&self, f: &mut Frame, lay: &Layout, r: Rect) {
        let (heading, text) = self.explain(self.step);
        f.text_px(r.min, heading, lay.size - 1.0, pal::DIM, Align2::LEFT_TOP);
        // (a little smaller, where the words would not fit at their size: none is left out
        // but on a canvas far too small for any of them)
        let fit = [0.0, 1.0, 2.0].into_iter().map(|less| lay.size - less).find_map(|size| {
            let line = size + 3.5;
            let lines = wrap(f, &text, r.size().x, size);
            (lines.len() as f32 * line <= r.size().y - line).then_some((size, line, lines))
        });
        let (size, line, lines) = fit.unwrap_or_else(|| (lay.size - 2.0, lay.size + 1.5, wrap(f, &text, r.size().x, lay.size - 2.0)));
        for (k, l) in lines.iter().enumerate() {
            f.text_px(r.min + Vec2::new(0.0, line * (k as f32 + 1.0) + 2.0), l.as_str(), size, pal::TEXT, Align2::LEFT_TOP);
        }
    }

    /// The buttons that step through the algorithm: what each says, and whether it can
    /// be pressed now.
    fn buttons(&self, compact: bool) -> [(&'static str, bool); 5] {
        let at_end = self.step >= self.last();
        let play = match (self.playing, compact) {
            (true, _) => "❚❚ pause",
            (false, true) => "play",
            (false, false) => "▶ play",
        };
        [
            (if compact { "◀" } else { "◀ back" }, self.step > 0),
            (if compact { "▶" } else { "next ▶" }, !at_end),
            (play, true),
            (if compact { "↺ start" } else { "↺ from the start" }, true),
            (if compact { "end" } else { "to the end" }, !at_end),
        ]
    }

    /// Press button `k`.
    fn press(&mut self, k: usize) {
        match k {
            0 => self.back(),
            1 => self.next(),
            2 => self.play(),
            3 => self.replay(),
            _ => self.go_to(self.last()),
        }
    }
}

impl Pc {
    /// The present test: its samples, and where its partial correlation falls against
    /// the threshold and against chance.
    fn draw_scatter(&mut self, f: &mut Frame, lay: &Layout, sq: Rect, bar: Rect) {
        let Some(focus) = self.focus else { return };
        let Some(&Step::Test { x, y, given, .. }) = self.run.steps.get(focus) else { return };
        let size = lay.size - 1.0;
        let centre = (sq.min + sq.max) * 0.5;
        let half = sq.size().x * 0.5;
        hud_outline(f, sq.min, sq.max, 1.0, pal::FAINT);
        hud_line(f, Vec2::new(sq.min.x, centre.y), Vec2::new(sq.max.x, centre.y), 1.0, pal::FAINT.a(0.7));
        hud_line(f, Vec2::new(centre.x, sq.min.y), Vec2::new(centre.x, sq.max.y), 1.0, pal::FAINT.a(0.7));
        let scale = half / REACH;
        let dot = if lay.narrow { 3.0 } else { 3.6 };
        for p in &self.points {
            let p = p.clamp(Vec2::splat(-REACH), Vec2::splat(REACH));
            f.dl.overlay_discs.push(f.hud_at(1, centre + Vec2::new(p.x, -p.y) * scale), dot, PLAIN.a(0.8));
        }
        // the least-squares line through the cloud: in standard units its slope is r
        let slope = self.shown.r * self.shown.r.abs();
        let reach = REACH.min(REACH / slope.abs().max(1e-6));
        hud_line(f, centre + Vec2::new(-reach, reach * slope) * scale, centre + Vec2::new(reach, -reach * slope) * scale, 2.2, NOW);
        f.text_px(Vec2::new(sq.max.x, sq.max.y + 3.0), v(x), size, PLAIN, Align2::RIGHT_TOP);
        f.text_px(Vec2::new(sq.min.x - 5.0, sq.min.y), v(y), size, PLAIN, Align2::RIGHT_TOP);

        // what all this is
        let given_words = if given == 0 { String::new() } else { format!(", each less what {} {}", set_words(given), explain_verb(given)) };
        let title = format!("{} and {}{given_words}", v(x), v(y));
        let exact = self.model.partial(x, y, given);
        let c = chance(self.n, given.count_ones() as usize);
        let legend = [
            (format!("white marks: the threshold, $\\pm {}$ (drag them)", num(self.threshold.min(1.0) as f64, 2)), pal::WHITE),
            (format!("grey band: chance alone, 95 times in 100, $\\pm {}$", num(c, 2)), pal::DIM),
            (format!("in the model itself (endless samples): $r = {}$", num(exact, 2)), TRUTH),
        ];
        match lay.beside {
            Some(b) => {
                let mut yy = b.min.y;
                for line in wrap(f, &title, b.size().x, size) {
                    f.text_px(Vec2::new(b.min.x, yy), line, size, pal::TEXT, Align2::LEFT_TOP);
                    yy += size + 4.0;
                }
                yy += 6.0;
                for (text, color) in legend {
                    for line in wrap(f, &text, b.size().x, size - 0.5) {
                        f.text_px(Vec2::new(b.min.x, yy), line, size - 0.5, color, Align2::LEFT_TOP);
                        yy += size + 3.0;
                    }
                    yy += 4.0;
                }
            }
            None => {
                let lines = wrap(f, &title, sq.size().x + 20.0, size);
                for (k, line) in lines.iter().enumerate() {
                    let up = (lines.len() - 1 - k) as f32 * (size + 4.0);
                    f.text_px(Vec2::new(sq.min.x - 18.0, sq.min.y - 7.0 - up), line.as_str(), size, pal::TEXT, Align2::LEFT_BOTTOM);
                }
                for (k, (text, color)) in legend.into_iter().enumerate() {
                    f.text_px(Vec2::new(sq.min.x - 18.0, bar.min.y + 24.0 + k as f32 * (size + 5.0)), text, size - 0.5, color, Align2::LEFT_TOP);
                }
            }
        }
    }

    /// The line under the scatter, from −1 to 1 and stretched near 0 where the
    /// decisions are: the threshold (its two white marks can be dragged), what chance
    /// alone gives, and the present test's r; at the start and the end, every test's r.
    fn draw_bar(&mut self, f: &mut Frame, lay: &Layout, bar: Rect, pointer: Option<Vec2>) {
        let size = lay.size - 1.0;
        let (cx, hw, yb) = ((bar.min.x + bar.max.x) * 0.5, bar.size().x * 0.5, bar.min.y);
        let test = self.focus.and_then(|k| match self.run.steps.get(k) {
            Some(&Step::Test { given, r, .. }) => Some((given, r)),
            _ => None,
        });
        let held = test.map_or(0, |t| t.0.count_ones() as usize);
        let c = chance(self.n, held);
        let band = stretch(c as f32);
        hud_rect(f, Vec2::new(cx - band * hw, yb - 5.0), Vec2::new(cx + band * hw, yb + 5.0), pal::DIM.a(0.4));
        hud_line(f, Vec2::new(bar.min.x, yb), Vec2::new(bar.max.x, yb), 1.2, pal::DIM);
        for (t, label) in [(-1.0f32, "−1"), (-0.5, "−0.5"), (-0.1, "−0.1"), (0.0, "0"), (0.1, "0.1"), (0.5, "0.5"), (1.0, "1")] {
            let x = cx + stretch(t) * hw;
            hud_line(f, Vec2::new(x, yb - 3.0), Vec2::new(x, yb + 3.0), 1.2, pal::DIM);
            f.text_px(Vec2::new(x, yb + 7.0), label, size - 2.5, pal::DIM, Align2::CENTER_TOP);
        }
        // at the end, every test: where its r fell, bright for those that parted a pair
        let end = matches!(self.run.steps.get(self.step), Some(Step::End));
        if end {
            for (&at, &(_, parted)) in self.tick_at.iter().zip(&self.ticks) {
                let x = cx + at.clamp(-1.0, 1.0) * hw;
                hud_line(f, Vec2::new(x, yb - 7.0), Vec2::new(x, yb + 7.0), 1.4, if parted { pal::WHITE.a(0.9) } else { NOW.a(0.55) });
            }
        }
        let t = self.threshold.min(1.0);
        let marks = [cx - stretch(t) * hw, cx + stretch(t) * hw];
        let near = pointer.is_some_and(|p| marks.iter().any(|&m| (p - Vec2::new(m, yb)).length() < 14.0));
        for m in marks {
            hud_line(f, Vec2::new(m, yb - 9.0), Vec2::new(m, yb + 9.0), 1.6, pal::WHITE);
            let s = if self.dragging { 13.0 } else if near { 11.0 } else { 9.0 };
            if self.dragging || near {
                f.dl.glows.push(f.hud_at(2, Vec2::new(m, yb)), s * 3.0, pal::WHITE.a(0.4));
            }
            f.dl.overlay_discs.push(f.hud_at(2, Vec2::new(m, yb)), s, pal::WHITE);
        }
        if let Some((_, r)) = test {
            let at = cx + self.shown.r.clamp(-1.0, 1.0) * hw;
            hud_line(f, Vec2::new(at, yb - 13.0), Vec2::new(at, yb + 13.0), 2.4, NOW);
            f.text_px(Vec2::new(at, yb - 15.0), format!("$r = {}$", num(r, 2)), size, NOW, Align2::CENTER_BOTTOM);
            return;
        }
        // what the line is, when there is no scatter to say it
        let says = if end { "every test's $r$: yellow, dependent; white, independent" } else { "the line each test's $r$ will be marked on" };
        let threshold = format!("white marks: the threshold, $\\pm {}$ (drag them)", num(t as f64, 2));
        if lay.narrow {
            f.text_px(Vec2::new(bar.min.x - 18.0, yb - 14.0), if end { "every test's $r$, and the threshold (drag)" } else { "the threshold (drag)" }, size - 0.5, pal::DIM, Align2::LEFT_BOTTOM);
        } else {
            for (k, (text, color)) in [(says.to_string(), pal::DIM), (threshold, pal::WHITE), (format!("grey band: chance alone, 95 times in 100, $\\pm {}$", num(c, 2)), pal::DIM)].into_iter().enumerate() {
                f.text_px(Vec2::new(bar.min.x - 18.0, yb + 24.0 + k as f32 * (size + 5.0)), text, size - 0.5, color, Align2::LEFT_TOP);
            }
        }
    }

    /// Before the first test: what the scatter will show.
    fn draw_promise(&self, f: &mut Frame, lay: &Layout, area: Rect) {
        let text = "Each test is drawn here as it is done: the samples of the two variables tested, each with what the held variables explain of it taken away. Independent, they make a cloud with no tilt.";
        let mut y = area.min.y;
        for line in wrap(f, text, area.size().x, lay.size) {
            f.text_px(Vec2::new(area.min.x, y), line, lay.size, pal::DIM, Align2::LEFT_TOP);
            y += lay.size + 4.0;
        }
    }

    /// At the end: every graph with exactly the independences found, the truth ringed.
    fn draw_class(&self, f: &mut Frame, lay: &Layout, area: Rect) {
        let size = lay.size - 0.5;
        let count = self.count();
        let k = self.class.len();
        let title = match k {
            0 => "No graph has exactly these independences: the tests contradict each other.".to_string(),
            1 => "The one graph with exactly these independences:".to_string(),
            2 => "Both graphs with exactly these independences:".to_string(),
            k => format!("All {k} graphs with exactly these independences:"),
        };
        let mut y = area.min.y;
        for line in wrap(f, &title, area.size().x, size) {
            f.text_px(Vec2::new(area.min.x, y), line, size, pal::TEXT, Align2::LEFT_TOP);
            y += size + 4.0;
        }
        y += 8.0;
        // the pictures as large as lets them all fit (up to a size), or as many as fit
        let room = Vec2::new(area.size().x, area.max.y - y - size - 8.0);
        let largest = if lay.narrow { 150.0 } else { 175.0 };
        let fits = |cols: usize| {
            let rows = k.div_ceil(cols).max(1);
            let tall = (room.y / rows as f32 - 24.0) / 0.8;
            Some(((room.x - (cols as f32 - 1.0) * 10.0) / cols as f32).min(largest).min(tall))
        };
        let best = (1..=k.max(1)).filter_map(fits).fold(0.0f32, f32::max);
        let tw = if best >= 70.0 { best } else { 70.0f32.min(room.x) };
        let th = tw * 0.8;
        let cols = ((room.x + 10.0) / (tw + 10.0)).floor().max(1.0) as usize;
        let rows = ((room.y + 2.0) / (th + 24.0)).floor().max(1.0) as usize;
        let fit = cols * rows;
        let shown = if k > fit { fit.saturating_sub(1).max(1) } else { k };
        let radius = (tw * 0.075).clamp(8.0, 12.0);
        let at: [Vec2; MAX] = PRESETS[self.preset].at.map(Vec2::from);
        for (n, dag) in self.class.iter().take(shown).enumerate() {
            let (c, rr) = (n % cols, n / cols);
            let min = Vec2::new(area.min.x + c as f32 * (tw + 10.0), y + rr as f32 * (th + 24.0));
            let r = Rect::new(min.x, min.y, min.x + tw, min.y + th);
            let truth = n == 0 && self.truth_in_class;
            hud_rect(f, r.min, r.max, Color::hex(0x10141F));
            if truth {
                hud_outline(f, r.min, r.max, 1.6, TRUTH);
                f.text_px(Vec2::new(r.min.x + 4.0, r.max.y + 3.0), "the truth", size - 1.5, TRUTH, Align2::LEFT_TOP);
            }
            let spot = spots(r, &at, radius);
            let mut links = Vec::new();
            for to in 0..count {
                for from in members(dag[to]).filter(|&from| from < count) {
                    links.push((from, to, Link::Arrow, if truth { TRUTH } else { FOUND.a(0.85) }, 1.5));
                }
            }
            let nodes: Vec<Node> = (0..count).map(|q| Node { at: spot[q], name: DISC_NAMES[q], color: PLAIN }).collect();
            hud_graph(f, &nodes, &links, radius);
        }
        let below = y + rows.min(shown.div_ceil(cols)) as f32 * (th + 24.0);
        if shown < k {
            f.text_px(Vec2::new(area.min.x, below), format!("… and {} more", k - shown), size, pal::DIM, Align2::LEFT_TOP);
        }
        if !self.truth_in_class {
            let below = if shown < k { below + size + 8.0 } else { below };
            for (j, line) in wrap(f, self.why_not(), area.size().x, size).into_iter().enumerate() {
                f.text_px(Vec2::new(area.min.x, below + j as f32 * (size + 4.0)), line, size, WRONG, Align2::LEFT_TOP);
            }
        }
    }

    /// The tests done so far, the present one lit. Returns the step of a row clicked.
    fn draw_list(&self, f: &mut Frame, lay: &Layout, r: Rect, pointer: Option<Vec2>) -> Option<usize> {
        let size = lay.size - 0.5;
        let rh = row_height(lay.size);
        let (rows, lit) = self.rows();
        let total = self.run.tests();
        f.text_px(r.min, format!("the tests, in the order done ({} of {total})", rows.len()), size, pal::DIM, Align2::LEFT_TOP);
        let cols: [f32; 4] = if lay.narrow { [0.0, 58.0, 126.0, 186.0] } else { [0.0, 74.0, 160.0, 228.0] };
        let head = r.min.y + rh * 1.2;
        for (c, text) in cols.iter().zip(["pair", "held", "$r$", ""]) {
            f.text_px(Vec2::new(r.min.x + c, head), text, size - 1.0, pal::FAINT.mix(pal::DIM, 0.5), Align2::LEFT_TOP);
        }
        let top = head + rh;
        let room = self.room();
        let mut clicked = None;
        let first = self.first_row();
        for k in first..(first + room).min(rows.len()) {
            let y = top + (k - first) as f32 * rh;
            // (the rows cut off by the window, faded at its edges)
            let edge = if (k == first && first > 0) || (k + 1 == first + room && k + 1 < rows.len()) { 0.45 } else { 1.0 };
            let Some(&Step::Test { x, y: b, given, r: rr, parted }) = rows.get(k).and_then(|&s| self.run.steps.get(s)) else { continue };
            let row = Rect::new(r.min.x - 4.0, y - 2.0, r.max.x, y + rh - 2.0);
            if lit == Some(k) {
                hud_rect(f, row.min, row.max, NOW.a(0.16 * edge));
            }
            if edge > 0.5 && pointer.is_some_and(|p| row.contains(p)) {
                hud_outline(f, row.min, row.max, 1.0, pal::DIM.a(0.5 * edge));
                if f.ptr.clicked {
                    clicked = rows.get(k).copied();
                }
            }
            let color = if lit == Some(k) {
                pal::WHITE
            } else if parted {
                pal::TEXT
            } else {
                pal::DIM
            };
            let color = color.a(edge);
            let held = if given == 0 { "–".to_string() } else { format!("${}$", members(given).map(|m| NAMES[m]).collect::<Vec<_>>().join(", ")) };
            let cells = [format!("${}$ – ${}$", NAMES[x], NAMES[b]), held, num(rr, 2), verdict(parted).to_string()];
            for (c, text) in cols.iter().zip(cells) {
                f.text_px(Vec2::new(r.min.x + c, y), text, size, color, Align2::LEFT_TOP);
            }
        }
        clicked
    }
}

impl Scene for Pc {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 3.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    /// While the steps play, something glides, or the threshold is dragged.
    fn animated(&self) -> bool {
        self.playing || self.moving || self.dragging
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("buttons over the picture: step through the tests    a row of the list: go to that test    white dots under the scatter: drag to move the threshold")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            "The older road to causal discovery. Assume nothing about the shape of the effects or of the noise; assume only that the causes form a graph without loops, that every common cause is measured, and that every independence in the data comes from the shape of the graph. Then, if its tests are right, the PC algorithm (Spirtes and Glymour, 1991) finds as much of the graph as independences can say:",
        );
        ui::note(ui, "1. Join every pair of variables.");
        ui::note(ui, "2. Remove a link when its two ends are independent, alone or given some of the others; remember the set that parted them.");
        ui::note(ui, r"3. For every $X$ – $Z$ – $Y$ with no link between $X$ and $Y$: if $Z$ was not in the set that parted them, $Z$ is a collider, $X \to Z \leftarrow Y$.");
        ui::note(ui, "4. Orient the links that the other way round would make a new collider or a loop.");
        ui::note(
            ui,
            "Here the effects are straight lines and the noise is bell-shaped (normal), so independence given a set is a partial correlation of 0, tested against a threshold. It is the classic case in which the residual methods of the other scenes cannot tell cause from effect, since a straight line with normal noise looks alike both ways round. These tests work in it, but only so far.",
        );

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The true graph");
        let names: Vec<&str> = PRESETS.iter().map(|p| p.name).collect();
        let mut preset = self.preset;
        if ui::choice(ui, &names, &mut preset) {
            self.preset = preset.min(PRESETS.len() - 1);
            self.strength = PRESETS[self.preset].arrows.iter().map(|a| a.2).collect();
            self.resample();
            self.rerun(false);
        }
        let mut changed = false;
        for (k, &(from, to, _)) in PRESETS[self.preset].arrows.iter().enumerate() {
            if let Some(b) = self.strength.get_mut(k) {
                // (kept in double precision, so that the paths that cancel cancel exactly)
                let mut v = *b as f32;
                if ui::slider(ui, &format!(r"${} \to {}$", NAMES[from], NAMES[to]), &mut v, -1.5..=1.5) {
                    *b = v as f64;
                    changed = true;
                }
            }
        }
        if changed {
            self.resample();
            self.rerun(false);
        }
        ui::note(ui, "How strongly each arrow acts: the effect is this number times the cause, plus bell-shaped noise of its own. At 0 there is no arrow.");

        ui::section(ui, "Samples and tests");
        if ui::slider_int_log(ui, "samples", &mut self.n, 20..=5000) {
            self.n = self.n.clamp(20, 5000);
            self.resample();
            self.rerun(false);
        }
        if ui.button("new sample").clicked() {
            self.new_sample();
        }
        if ui::slider_log_or_zero(ui, "threshold", &mut self.threshold, 0.005, 1.0) {
            self.threshold = self.threshold.max(0.0).min(1.0);
            self.rerun(true);
        }
        ui::note(ui, "A link goes when a partial correlation is nearer 0 than this. Samples never give exactly 0: the threshold should stand clear of what chance alone gives (the grey band under the scatter), and below what real links give.");

        ui::section(ui, "Steps");
        ui.horizontal_wrapped(|ui| {
            if ui.button("◀ back").clicked() {
                self.back();
            }
            if ui.button("next ▶").clicked() {
                self.next();
            }
            if ui.button(if self.playing { "❚❚ pause" } else { "▶ play" }).clicked() {
                self.play();
            }
            if ui.button("↺ from the start").clicked() {
                self.replay();
            }
            if ui.button("to the end").clicked() {
                self.go_to(self.last());
            }
        });

        ui::note(ui, "The pairs are taken in turn, the smallest sets held first: nothing, then one other variable, then two, and so on. The sets are drawn from the variables linked to either end as the links stood when that size began: the \"stable\" form of Colombo and Maathuis (2014), whose links do not depend on the order the pairs are taken in.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (TRUTH, "green, left: the graph the samples came from, each arrow with its strength"),
                (FOUND, "white, right: what the tests have found so far; a plain line where they cannot say which way"),
                (NOW, "yellow: the test being done: its pair, its link (dashed when it goes) and its partial correlation; at the end, the partial correlation of every test that kept its link"),
                (HELD, "purple rings: the variables held in the test"),
                (TRUTH, "green, right (at the end): an arrow found as it is in the truth"),
                (WRONG, "red (at the end): a link that is not there, an arrow the wrong way, or, dashed, a link lost"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, r"Independent given $B$:", r"once $B$ is known, one variable says nothing more about the other.");
        ui::term(
            ui,
            r"Partial correlation $r$:",
            r"the correlation of what is left of two variables once what the held ones explain (by a straight-line fit) is taken from each. With straight-line effects and normal noise it is 0 exactly when they are independent given the held ones.",
        );
        ui::term(ui, "Collider:", r"a variable two arrows point into, from variables not linked to each other: $A \to B \leftarrow C$.");
        ui::term(ui, "Markov equivalent:", "giving exactly the same independences. Two graphs are, exactly when they have the same links and the same colliders (Verma and Pearl, 1990).");
        ui::term(ui, "Faithful:", "every independence in the data comes from the shape of the graph, none from strengths that happen to cancel. The method assumes it.");

        ui::section(ui, "Numbers");
        let found = self.run.last();
        let links = found.links().count();
        let arrows = found.links().filter(|&(i, j)| found.head[i][j] != found.head[j][i]).count();
        ui::readout(ui, "tests done", format!("{}", self.run.tests()));
        ui::readout(ui, "links found", format!("{links}"));
        ui::readout(ui, "of them oriented", format!("{arrows}"));
        ui::readout(ui, "graphs with these independences", format!("{}", self.class.len()));
        ui::readout(ui, "right, as far as independences can say", if self.run.last() == self.truth.last() { "yes" } else { "no" });
        ui::see_also(
            ui,
            &[
                ("junctions", "chains, forks and colliders, and what holding the middle does"),
                ("direction", "the residual road: cause or effect from two variables"),
                ("discover", "the residual road with many variables"),
                ("adjust", "what to control for, once the graph is known"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt);
        let view = f.view_size();
        let Some(lay) = Layout::of(view, f.top_inset) else {
            self.follow(f.real_dt);
            return;
        };
        if let Some(list) = lay.list {
            let rh = row_height(lay.size);
            self.room_rows = ((list.size().y - rh * 2.2) / rh).floor().max(1.0) as usize;
        }
        let pointer = f.ptr.pos.map(|p| p / f.px);

        // what the pointer can do here: press the buttons, pick a test, move the threshold
        let mut busy: Vec<Rect> = Vec::new();
        busy.extend(lay.buttons.into_iter().flatten());
        busy.extend(lay.list);
        if let Some(bar) = lay.bar {
            busy.push(Rect::new(bar.min.x - 14.0, bar.min.y - 14.0, bar.max.x + 14.0, bar.max.y + 14.0));
        }
        if self.dragging || pointer.is_some_and(|p| busy.iter().any(|r| r.contains(p))) {
            f.handles.capture();
            f.handles.claim_clicks();
        }
        if let Some(bar) = lay.bar {
            let (cx, hw) = ((bar.min.x + bar.max.x) * 0.5, (bar.size().x * 0.5).max(1.0));
            let t = stretch(self.threshold.min(1.0));
            let near = pointer.is_some_and(|p| [cx - t * hw, cx + t * hw].iter().any(|&m| (p - Vec2::new(m, bar.min.y)).length() < 14.0));
            if f.ptr.pressed && near {
                self.dragging = true;
            }
            if self.dragging {
                if let Some(p) = pointer {
                    let t = ((p.x - cx).abs() / hw).clamp(0.0, 1.0).powi(2);
                    if t != self.threshold {
                        self.threshold = t;
                        self.rerun(true);
                    }
                }
            }
        }
        if !f.ptr.down {
            self.dragging = false;
        }
        if let (Some(b), Some(p), true) = (lay.buttons, pointer, f.ptr.clicked) {
            let usable = self.buttons(lay.compact).map(|(_, usable)| usable);
            if let Some(k) = (0..5).find(|&k| usable[k] && b[k].contains(p)) {
                self.press(k);
            }
        }

        self.follow(f.real_dt);
        // (what keeps its parts from frame to frame first, what can come and go after
        // it: a moving picture is followed part by part, in the order drawn)
        self.draw_truth(f, &lay);
        self.draw_found(f, &lay);
        if let Some(bar) = lay.bar {
            self.draw_bar(f, &lay, bar, pointer);
        }
        if let Some(b) = lay.buttons {
            for (r, (text, usable)) in b.into_iter().zip(self.buttons(lay.compact)) {
                hud_button(f, r, text, usable, pointer);
            }
        }
        if let Some(words) = lay.words {
            self.draw_words(f, &lay, words);
        }
        let clicked = lay.list.and_then(|list| self.draw_list(f, &lay, list, pointer));
        if let (Some(sq), Some(bar)) = (lay.scatter, lay.bar) {
            let area = Rect::new(sq.min.x - 18.0, sq.min.y - 4.0, lay.beside.map_or(sq.max.x + 20.0, |b| b.max.x), bar.min.y - if lay.narrow { 34.0 } else { 26.0 });
            match self.run.steps.get(self.step) {
                Some(Step::End) => self.draw_class(f, &lay, area),
                Some(Step::Start) => self.draw_promise(f, &lay, area),
                _ => self.draw_scatter(f, &lay, sq, bar),
            }
        }
        if let Some(step) = clicked {
            // (seen in the next frame)
            self.go_to(step);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::causal::partial_correlation;

    /// A scene set up as an experiment (or opened), with its steps played to the end.
    fn run(setup: Setup) -> Pc {
        let mut scene = Pc::create();
        scene.apply(setup);
        for _ in 0..10_000 {
            scene.advance(1.0 / 60.0);
            if !scene.playing {
                break;
            }
        }
        assert!(!scene.playing && scene.step == scene.last(), "the steps did not play to the end");
        scene.settle();
        scene
    }

    fn trial(k: usize) -> Pc {
        run(TRIALS[k].2)
    }

    /// A graph from its arrows.
    fn dag(arrows: &[(usize, usize)]) -> Dag {
        let mut d: Dag = [0; MAX];
        for &(from, to) in arrows {
            d[to] |= 1 << from;
        }
        d
    }

    /// The test of `x` and `y` given `given` in the run, if it was done.
    fn test_of(run: &Run, x: usize, y: usize, given: Set) -> Option<(f64, bool)> {
        run.steps.iter().find_map(|s| match *s {
            Step::Test { x: a, y: b, given: g, r, parted } if (a, b, g) == (x, y, given) => Some((r, parted)),
            _ => None,
        })
    }

    /// The partial correlations worked out from the covariances are the ones
    /// `partial_correlation` gives from the samples.
    #[test]
    fn partial_correlations_from_covariances_match_the_samples() {
        for (k, preset) in PRESETS.iter().enumerate() {
            for n in [25, 300] {
                let data = sample(preset.count, preset.arrows, n, 40 + k as u64);
                let cov = Cov::of(&data);
                for x in 0..preset.count {
                    for y in x + 1..preset.count {
                        for given in 0..(1u8 << preset.count) {
                            if given & (1 << x | 1 << y) != 0 {
                                continue;
                            }
                            let held: Vec<&[f64]> = members(given).map(|v| data[v].as_slice()).collect();
                            let (a, b) = (cov.partial(x, y, given), partial_correlation(&data[x], &data[y], &held));
                            assert!((a - b).abs() < 1e-9, "{}: {x}, {y} given {given:b}: {a} against {b}", preset.name);
                        }
                    }
                }
            }
        }
        // and the model's own covariances are what samples tend to
        let preset = &PRESETS[OPENING];
        let (sampled, exact) = (Cov::of(&sample(preset.count, preset.arrows, 200_000, 9)), model_cov(preset.count, preset.arrows));
        for i in 0..preset.count {
            for j in 0..preset.count {
                assert!((sampled.c[i][j] - exact.c[i][j]).abs() < 0.03 * exact.c[i][i].max(exact.c[j][j]), "{i}, {j}");
            }
        }
    }

    /// With every arrow acting, the model's partial correlations are 0 exactly where
    /// the graph separates the pair (it is faithful), but for the paths that cancel:
    /// there $A$ and $C$ are uncorrelated although nothing separates them.
    #[test]
    fn independences_come_from_the_shape_of_the_graph_unless_paths_cancel() {
        for preset in &PRESETS {
            let (cov, truth) = (model_cov(preset.count, preset.arrows), truth_dag(preset.arrows));
            for x in 0..preset.count {
                for y in x + 1..preset.count {
                    for given in (0..(1u8 << preset.count)).filter(|g| g & (1 << x | 1 << y) == 0) {
                        let zero = cov.partial(x, y, given).abs() < 1e-12;
                        let apart = separated(preset.count, &truth, x, y, given);
                        let cancel = preset.name == "two paths that cancel" && (x, y, given) == (0, 2, 0);
                        assert_eq!(zero, apart || cancel, "{}: {x}, {y} given {given:b}", preset.name);
                        assert!(!(cancel && apart));
                    }
                }
            }
        }
    }

    /// What each true graph's independences give when every test is right: the links,
    /// the colliders, what follows, and how many graphs share them.
    #[test]
    fn the_true_patterns_are_the_textbook_ones() {
        let pattern = |k: usize| {
            let p = &PRESETS[k];
            let run = true_pattern(p.count, &truth_dag(p.arrows));
            let last = run.last();
            let members = class(p.count, &last, &run.colliders());
            (last, members, truth_dag(p.arrows))
        };
        let lines = |p: &Pattern| p.links().filter(|&(i, j)| p.line(i, j)).collect::<Vec<_>>();
        let arrows = |p: &Pattern| (0..p.count).flat_map(|i| (0..p.count).map(move |j| (i, j))).filter(|&(i, j)| p.arrow(i, j)).collect::<Vec<_>>();
        // the chain and the fork: the same plain lines, and three graphs
        for k in [0, 1] {
            let (found, members, truth) = pattern(k);
            assert_eq!((lines(&found), arrows(&found)), (vec![(0, 1), (1, 2)], vec![]), "{}", PRESETS[k].name);
            assert_eq!(members.len(), 3);
            for d in [dag(&[(0, 1), (1, 2)]), dag(&[(2, 1), (1, 0)]), dag(&[(1, 0), (1, 2)])] {
                assert!(members.contains(&d));
            }
            assert!(members.contains(&truth));
        }
        assert_eq!(pattern(0).0, pattern(1).0, "a chain and a fork give different answers");
        // the collider: both arrows, alone in its class
        let (found, members, truth) = pattern(2);
        assert_eq!((lines(&found), arrows(&found), members), (vec![], vec![(0, 1), (2, 1)], vec![truth]));
        // with a tail: everything oriented
        let (found, members, truth) = pattern(3);
        assert_eq!((lines(&found), arrows(&found), members), (vec![], vec![(0, 1), (1, 3), (2, 1)], vec![truth]));
        // the diamond, with and without its tail: the top stays plain, three graphs
        let (found, members, _) = pattern(4);
        assert_eq!((lines(&found), arrows(&found), members.len()), (vec![(0, 1), (0, 2)], vec![(1, 3), (2, 3)], 3));
        let (found, members, truth) = pattern(5);
        assert_eq!((lines(&found), arrows(&found), members.len()), (vec![(0, 1), (0, 2)], vec![(1, 3), (2, 3), (3, 4)], 3));
        assert!(members.contains(&truth));
        // two paths that cancel, read off the graph: a triangle, nothing oriented, six graphs
        let (found, members, _) = pattern(6);
        assert_eq!((lines(&found).len(), members.len()), (3, 6));
    }

    /// What the last step says, on every graph of up to five variables there is (with
    /// every test right): the arrows found are in every graph with the same
    /// independences, and each plain line points one way in some of them and the
    /// other way in others. (Meek's rules leave nothing out and add nothing wrong.)
    #[test]
    fn the_arrows_found_are_all_that_independences_can_say() {
        let mut checked = 0;
        for count in 2..=5 {
            let pairs: Vec<(usize, usize)> = (0..count).flat_map(|i| (i + 1..count).map(move |j| (i, j))).collect();
            // every graph whose arrows go from earlier to later variables, with the
            // variables renamed so that every order turns up somewhere
            for links in 0u32..1 << pairs.len() {
                let order: Vec<usize> = (0..count).map(|k| (k * 3 + links as usize) % count).collect();
                let mut seen = [false; MAX];
                if order.iter().any(|&v| std::mem::replace(&mut seen[v], true)) {
                    continue;
                }
                let mut truth: Dag = [0; MAX];
                for (k, &(i, j)) in pairs.iter().enumerate() {
                    if links & (1 << k) != 0 {
                        truth[order[j]] |= 1 << order[i];
                    }
                }
                let run = true_pattern(count, &truth);
                let found = run.last();
                let members = class(count, &found, &run.colliders());
                assert!(members.contains(&truth), "{truth:?}");
                for (i, j) in found.links() {
                    let ways = |a: usize, b: usize| members.iter().filter(|d| d[b] & (1 << a) != 0).count();
                    let (ij, ji) = (ways(i, j), ways(j, i));
                    if found.arrow(i, j) {
                        assert_eq!((ij, ji), (members.len(), 0), "{truth:?}: {i} -> {j}");
                    } else if found.arrow(j, i) {
                        assert_eq!((ji, ij), (members.len(), 0), "{truth:?}: {j} -> {i}");
                    } else {
                        assert!(ij > 0 && ji > 0, "{truth:?}: {i} - {j} goes one way only");
                    }
                }
                checked += 1;
            }
        }
        assert!(checked > 500, "{checked}");
    }

    /// The grey band: of the partial correlations of independent variables, 95 in 100
    /// fall within it, with few samples and many, and with variables held.
    #[test]
    fn chance_stays_inside_the_band_95_times_in_100() {
        for (n, held) in [(30usize, 0usize), (30, 2), (1000, 0), (100, 3)] {
            let trials = 4000;
            let mut rng = Rng::new(n as u64 + held as u64);
            let mut inside = 0;
            for _ in 0..trials {
                let data: Vec<Vec<f64>> = (0..2 + held).map(|_| (0..n).map(|_| rng.normal() as f64).collect()).collect();
                let cov = Cov::of(&data);
                let given = (0..held).fold(0, |s, k| s | 1 << (k + 2));
                inside += (cov.partial(0, 1, given).abs() < chance(n, held)) as usize;
            }
            let share = inside as f64 / trials as f64;
            assert!((share - 0.95).abs() < 0.012, "{n} samples, {held} held: {share}");
        }
        assert_eq!(num(chance(30, 0), 2), "0.36");
        assert_eq!(num(chance(1000, 1), 2), "0.06");
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let chain = trial(0);
        let (r, parted) = test_of(&chain.run, 0, 2, 0).unwrap();
        assert!(!parted && r > 0.3, "A and C alone: {r}");
        assert!(test_of(&chain.run, 0, 2, 0b010).is_some_and(|(r, parted)| parted && r.abs() < 0.05));
        assert!(test_of(&chain.run, 0, 1, 0).is_some_and(|t| !t.1) && test_of(&chain.run, 1, 2, 0).is_some_and(|t| !t.1));
        assert!(chain.run.steps.contains(&Step::Triple { x: 0, z: 1, y: 2, given: 0b010, by: 5, collider: false }), "{:?}", chain.run.steps);
        let found = chain.run.last();
        assert!(found.line(0, 1) && found.line(1, 2) && !found.link[0][2]);
        assert_eq!(chain.class.len(), 3);
        assert!(chain.truth_in_class && chain.class.contains(&dag(&[(2, 1), (1, 0)])) && chain.class.contains(&dag(&[(1, 0), (1, 2)])));

        let collider = trial(1);
        assert!(test_of(&collider.run, 0, 2, 0).is_some_and(|(r, parted)| parted && r.abs() < 0.03));
        let found = collider.run.last();
        assert!(found.arrow(0, 1) && found.arrow(2, 1) && !found.link[0][2]);
        assert!(collider.class.len() == 1 && collider.truth_in_class);

        let tail = trial(2);
        let found = tail.run.last();
        assert!(found.arrow(0, 1) && found.arrow(2, 1) && found.arrow(1, 3) && found.links().count() == 3);
        let orient = tail.run.steps.iter().find_map(|s| if let Step::Orient { from, to, rule, via, by } = *s { Some((from, to, rule, via, by)) } else { None });
        let (from, to, rule, via, by) = orient.expect("an arrow follows");
        assert_eq!((from, to, rule, via[0]), (1, 3, Rule::NoNewCollider, 0));
        assert!(matches!(tail.run.steps.get(by.unwrap_or(0)), Some(Step::Test { x: 0, y: 3, given: 0b0010, parted: true, .. })), "A and D were not parted given B alone");
        assert!(tail.class.len() == 1 && tail.truth_in_class);

        let few = trial(3);
        assert_eq!(few.n, 30);
        let (r, parted) = test_of(&few.run, 0, 2, 0).unwrap();
        assert!(!parted && num(r, 2) == "0.26" && r.abs() < chance(30, 0), "{r}");
        let found = few.run.last();
        assert!(found.link[0][1] && found.link[1][2] && found.link[0][2] && (0..3).all(|i| (0..3).all(|j| !found.head[i][j])));
        assert!(few.class.len() == 6 && !few.truth_in_class);
        assert!(few.why_not().contains("some test was wrong") && few.explain(few.last()).1.contains("no link can be oriented"));
        // and fresh samples of 30 find the collider about two times in five (783 of 2000)
        let truth = few.truth.last();
        let right = (0..2000u64).filter(|&seed| discover(&Cov::of(&sample(3, PRESETS[2].arrows, 30, seed)), 0.1).last() == truth).count();
        assert!((720..=880).contains(&right), "{right} of 2000");

        let cancel = trial(4);
        assert!(cancel.model.partial(0, 2, 0).abs() < 1e-12);
        assert!(test_of(&cancel.run, 0, 2, 0).is_some_and(|(r, parted)| parted && r.abs() < 0.04));
        let found = cancel.run.last();
        assert!(found.arrow(0, 1) && found.arrow(2, 1) && !found.link[0][2]);
        assert_eq!((cancel.verdict_color(0, 1, &found), cancel.verdict_color(1, 2, &found)), (TRUTH, WRONG));
        assert!(cancel.class.len() == 1 && !cancel.truth_in_class);
        assert!(cancel.why_not().contains("just what endless samples would show"));
        // the strengths the caption quotes
        assert_eq!(PRESETS[6].arrows, &[(0, 1, 0.8), (1, 2, 0.8), (0, 2, -0.64)]);
    }

    /// The scene opens on the finished run of the five variables, found as the truth
    /// has it: three graphs, the truth one of them.
    #[test]
    fn the_opening_finds_what_independences_can_say() {
        let scene = Pc::create();
        assert!(!scene.playing && scene.step == scene.last() && !scene.moving);
        assert_eq!(scene.run.last(), scene.truth.last());
        assert!(scene.class.len() == 3 && scene.truth_in_class);
        // and that is no accident of this sample: at this size and threshold it is
        // found on almost every sample (measured: 198 of 200)
        let p = &PRESETS[OPENING];
        let right = (100..200u64).filter(|&seed| discover(&Cov::of(&sample(p.count, p.arrows, 1000, seed)), 0.1).last() == scene.truth.last()).count();
        assert!(right >= 97, "{right} of 100");
    }

    /// Every step's words, on every preset, parse; the steps play through and stop.
    #[test]
    fn every_step_can_be_told() {
        for k in 0..PRESETS.len() {
            let mut scene = run(Setup { preset: k, ..OPENING_SETUP });
            for step in 0..scene.run.steps.len() {
                let (head, text) = scene.explain(step);
                for t in [&head, &text] {
                    assert!(crate::typeset::inspect(t).error.is_none(), "{}: {t}", PRESETS[k].name);
                }
                assert!(text.len() > 40);
            }
            // back, next, play and replay stay within the steps
            scene.back();
            assert_eq!(scene.step, scene.last() - 1);
            scene.next();
            scene.next();
            assert_eq!(scene.step, scene.last());
            scene.play();
            assert!(scene.playing && scene.step == 0);
            scene.play();
            assert!(!scene.playing);
        }
    }

    /// The degenerate settings: nothing removed (threshold 0), everything removed
    /// (threshold 1), no arrow acting, the fewest samples. Everything stays finite,
    /// and the words say what happened.
    #[test]
    fn degenerate_settings_are_told_honestly() {
        let mut scene = run(OPENING_SETUP);
        scene.threshold = 0.0;
        scene.rerun(false);
        assert_eq!(scene.run.last().links().count(), 10);
        assert_eq!(scene.class.len(), 120, "every order of five variables");
        assert!(!scene.truth_in_class);
        scene.threshold = 1.0;
        scene.rerun(false);
        assert_eq!(scene.run.last().links().count(), 0);
        assert!(scene.class.len() == 1 && !scene.truth_in_class);
        assert!(scene.explain(scene.last()).1.contains("Every pair was parted"));
        // no arrow acting at all: every pair parted with nothing held, as is true
        let mut scene = run(OPENING_SETUP);
        scene.strength = vec![0.0; scene.strength.len()];
        scene.resample();
        scene.rerun(false);
        assert_eq!(scene.run.last().links().count(), 0);
        assert!(scene.truth_in_class && scene.run.last() == scene.truth.last());
        // the fewest samples, and a variable that is an exact copy of another
        for n in [20, 4] {
            let mut scene = run(Setup { n, ..OPENING_SETUP });
            scene.resample();
            scene.rerun(false);
            for step in 0..scene.run.steps.len() {
                scene.step = step;
                scene.follow(1.0 / 60.0);
                assert!(scene.points.iter().all(|p| p.is_finite()) && scene.shown.r.is_finite());
            }
        }
        let mut data = sample(3, PRESETS[0].arrows, 200, 3);
        data[1] = data[0].clone();
        let cov = Cov::of(&data);
        for (x, y, given) in [(0, 1, 0), (0, 2, 0b010), (1, 2, 0b001), (0, 2, 0)] {
            assert!(cov.partial(x, y, given).is_finite());
        }
        let run = discover(&cov, 0.1);
        assert!(run.steps.iter().all(|s| !matches!(s, Step::Test { r, .. } if !r.is_finite())));
    }

    /// Whatever the tests find, the last step's words hold: if some graph has exactly
    /// the independences found, its arrows are in all such graphs and its plain lines
    /// go both ways among them (on samples small and large, at many thresholds).
    #[test]
    fn the_last_step_holds_for_whatever_the_tests_find() {
        let mut rng = Rng::new(77);
        let (mut empty, mut some, mut oriented) = (0, 0, [0; 3]);
        for round in 0..600u64 {
            let p = &PRESETS[round as usize % PRESETS.len()];
            let arrows: Vec<(usize, usize, f64)> = p.arrows.iter().map(|&(a, b, _)| (a, b, rng.range(-1.2, 1.2) as f64)).collect();
            let n = [12, 20, 40, 200][round as usize % 4];
            let run = discover(&Cov::of(&sample(p.count, &arrows, n, round)), rng.range(0.02, 0.5) as f64);
            let found = run.last();
            // what the steps that orient say of the tests they lean on
            for step in &run.steps {
                if let Step::Orient { from, to, rule, via, by } = *step {
                    oriented[rule as usize] += 1;
                    match rule {
                        Rule::NoNewCollider => assert!(matches!(run.steps.get(by.unwrap_or(0)), Some(&Step::Test { x, y, given, parted: true, .. }) if (x, y) == (via[0].min(to), via[0].max(to)) && given & (1 << from) != 0), "round {round}: {step:?}"),
                        Rule::TwoWays => assert!(matches!(run.steps.get(by.unwrap_or(0)), Some(&Step::Test { x, y, given, parted: true, .. }) if (x, y) == (via[0], via[1]) && given & (1 << from) != 0), "round {round}: {step:?}"),
                        Rule::NoCycle => assert!(run.after.iter().any(|p| p.arrow(from, via[0]) && p.arrow(via[0], to))),
                    }
                }
            }
            let members = class(p.count, &found, &run.colliders());
            if members.is_empty() {
                empty += 1;
                continue;
            }
            some += 1;
            for (i, j) in found.links() {
                let ways = |a: usize, b: usize| members.iter().filter(|d| d[b] & (1 << a) != 0).count();
                let (ij, ji) = (ways(i, j), ways(j, i));
                match (found.head[i][j], found.head[j][i]) {
                    (true, false) => assert_eq!((ij, ji), (members.len(), 0)),
                    (false, true) => assert_eq!((ji, ij), (members.len(), 0)),
                    (false, false) => assert!(ij > 0 && ji > 0, "round {round}: {i} - {j}"),
                    (true, true) => panic!("round {round}: a link with two heads, yet a graph fits"),
                }
            }
        }
        assert!(some > 400, "{some} with graphs, {empty} without");
        assert!(oriented[0] >= 15 && oriented[1] >= 10 && oriented[2] >= 5, "orientations checked, by rule: {oriented:?}");
    }

    /// The links found do not depend on the order the pairs are taken in: the same
    /// samples with the variables renamed (so visited in another order) lose the same
    /// links, on samples small and large and at many thresholds.
    #[test]
    fn the_links_found_do_not_depend_on_the_order() {
        let mut rng = Rng::new(5);
        for round in 0..300u64 {
            let p = &PRESETS[round as usize % PRESETS.len()];
            let data = sample(p.count, p.arrows, [15, 30, 100][round as usize % 3], round);
            let t = rng.range(0.05, 0.4) as f64;
            let found = discover(&Cov::of(&data), t).last();
            // renamed: variable k is called order[k]
            let mut order: Vec<usize> = (0..p.count).collect();
            rng.shuffle(&mut order);
            let mut renamed = vec![Vec::new(); p.count];
            for k in 0..p.count {
                renamed[order[k]] = data[k].clone();
            }
            let again = discover(&Cov::of(&renamed), t).last();
            for i in 0..p.count {
                for j in 0..p.count {
                    assert_eq!(found.link[i][j], again.link[order[i]][order[j]], "round {round}: {i} - {j}");
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn explore() {
        for preset in &PRESETS {
            let truth = true_pattern(preset.count, &truth_dag(preset.arrows));
            let members = class(preset.count, &truth.last(), &truth.colliders());
            println!("{:<24} {} steps, {} tests, class of {}", preset.name, truth.steps.len(), truth.tests(), members.len());
        }
        for (k, t) in TRIALS.iter().enumerate() {
            let scene = trial(k);
            println!("{}: {} steps, class {}, truth in it {}", t.0, scene.run.steps.len(), scene.class.len(), scene.truth_in_class);
            for s in &scene.run.steps {
                println!("    {s:?}");
            }
        }
        // the collider from 30 samples: how often each thing happens
        let p = &PRESETS[2];
        let truth = true_pattern(3, &truth_dag(p.arrows)).last();
        let (mut right, mut kept) = (0, 0);
        for seed in 0..2000u64 {
            let run = discover(&Cov::of(&sample(3, p.arrows, 30, seed)), 0.1);
            right += (run.last() == truth) as usize;
            kept += run.last().link[0][2] as usize;
        }
        println!("collider, 30 samples: right {right} of 2000, A - C kept {kept}");
    }
}

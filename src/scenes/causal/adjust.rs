//! What to control for: which variables to hold when an effect is measured from data
//! that were only observed, and which ones make the answer worse.

use super::*;
use crate::math::linalg::solve;

pub const DEF: SceneDef = SceneDef {
    id: "adjust",
    title: "What to control for",
    group: GROUP,
    blurb: "Which variables to hold fixed to measure an effect, and which ones make the answer worse.",
    make: |_| Box::new(Adjust::new()),
};

// ----- the graphs ------------------------------------------------------------------------

/// The most variables a graph here has.
const MOST: usize = 7;
/// Units in a sample.
const UNITS: usize = 2000;
/// Other samples of as many units, for how far the estimate wanders from sample to sample.
const OTHERS: usize = 40;
/// Strengths closer to 0 than this are 0 (see `set_strength`).
const SNAP: f64 = 0.04;
/// The seed of the sample the scene opens with (and every experiment shows).
const SEED: u64 = 13;

/// What part a variable plays (X and Y are always variables 0 and 1).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Role {
    Cause,
    Effect,
    /// Measured: it can be held.
    Measured,
    /// Not measured: it is there, but no fit can use it.
    Hidden,
}

struct VarDef {
    /// Its letter as markup without the dollars ("Z", "U_1"), and with them.
    letter: &'static str,
    name: &'static str,
    /// What it is, written beside it, below it or above it.
    kind: Kind,
    below: bool,
    /// Where it is drawn, in a box 1 wide and 1 high (stretched to the graph's shape).
    at: (f32, f32),
    role: Role,
}

struct ArrowDef {
    from: usize,
    to: usize,
    /// Its strength to begin with.
    strength: f64,
    /// Which side of it its strength is written on: 1 to the left as it points, −1 to the right.
    side: f32,
}

struct GraphDef {
    /// Its name on its button.
    name: &'static str,
    vars: &'static [VarDef],
    arrows: &'static [ArrowDef],
    /// The height of its box, for a width of 1.
    tall: f32,
}

/// What is written beside a variable.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// This, whatever the arrows ("" for nothing).
    Said(&'static str),
    /// What it is to $X$ and $Y$ through the arrows that are there (`kind_of`); this
    /// when every arrow is there.
    ByArrows(&'static str),
}

impl Kind {
    /// What it says with every arrow there (which the layout leaves room for).
    fn first(self) -> &'static str {
        match self {
            Kind::Said(t) | Kind::ByArrows(t) => t,
        }
    }
}

const fn var(letter: &'static str, name: &'static str, kind: Kind, below: bool, at: (f32, f32), role: Role) -> VarDef {
    VarDef { letter, name, kind, below, at, role }
}

const fn arrow(from: usize, to: usize, strength: f64, side: f32) -> ArrowDef {
    ArrowDef { from, to, strength, side }
}

/// The plain graph: one variable of each kind.
const PLAIN: GraphDef = GraphDef {
    name: "one of each kind",
    vars: &[
        var("X", "$X$", Kind::Said(""), true, (0.16, 0.42), Role::Cause),
        var("Y", "$Y$", Kind::Said(""), true, (0.84, 0.42), Role::Effect),
        var("Z", "$Z$", Kind::ByArrows("confounder"), false, (0.5, 0.0), Role::Measured),
        var("M", "$M$", Kind::ByArrows("mediator"), true, (0.5, 0.68), Role::Measured),
        var("C", "$C$", Kind::ByArrows("collider"), true, (0.5, 1.0), Role::Measured),
        var("I", "$I$", Kind::ByArrows("cause of $X$ only"), false, (0.0, 0.0), Role::Measured),
        var("P", "$P$", Kind::ByArrows("cause of $Y$ only"), false, (1.0, 0.0), Role::Measured),
    ],
    arrows: &[
        arrow(2, 0, 1.0, -1.0),
        arrow(2, 1, 1.0, 1.0),
        arrow(0, 1, 0.5, 1.0),
        arrow(0, 3, 0.8, 1.0),
        arrow(3, 1, 0.5, 1.0),
        arrow(0, 4, 0.6, -1.0),
        arrow(1, 4, 0.6, 1.0),
        arrow(5, 0, 1.5, -1.0),
        arrow(6, 1, 1.5, 1.0),
    ],
    tall: 0.62,
};

/// The M shape: a variable measured before X, linked to both, that is no confounder.
const M_SHAPE: GraphDef = GraphDef {
    name: "the ‘M’ shape",
    vars: &[
        var("X", "$X$", Kind::Said(""), true, (0.0, 0.62), Role::Cause),
        var("Y", "$Y$", Kind::Said(""), true, (1.0, 0.62), Role::Effect),
        var("Z", "$Z$", Kind::ByArrows("confounder"), true, (0.5, 1.0), Role::Measured),
        var("B", "$B$", Kind::Said("measured before $X$"), true, (0.5, 0.33), Role::Measured),
        var("U_1", "$U_1$", Kind::Said("not measured"), false, (0.2, 0.0), Role::Hidden),
        var("U_2", "$U_2$", Kind::Said("not measured"), false, (0.8, 0.0), Role::Hidden),
    ],
    arrows: &[
        arrow(2, 0, 1.0, 1.0),
        arrow(2, 1, 1.0, -1.0),
        arrow(0, 1, 0.5, -1.0),
        arrow(4, 0, 1.0, -1.0),
        arrow(4, 3, 1.0, 1.0),
        arrow(5, 3, 1.0, -1.0),
        arrow(5, 1, 1.0, 1.0),
    ],
    tall: 0.62,
};

/// The butterfly: B is a confounder and a collider at once.
const BUTTERFLY: GraphDef = GraphDef {
    name: "the butterfly",
    vars: &[
        var("X", "$X$", Kind::Said(""), true, (0.0, 1.0), Role::Cause),
        var("Y", "$Y$", Kind::Said(""), true, (1.0, 1.0), Role::Effect),
        var("B", "$B$", Kind::Said(""), true, (0.5, 0.45), Role::Measured),
        var("A", "$A$", Kind::Said(""), false, (0.0, 0.0), Role::Measured),
        var("D", "$D$", Kind::Said(""), false, (1.0, 0.0), Role::Measured),
    ],
    arrows: &[
        arrow(0, 1, 0.5, -1.0),
        arrow(2, 0, 0.8, -1.0),
        arrow(2, 1, 0.8, 1.0),
        arrow(3, 0, 1.0, 1.0),
        arrow(3, 2, 1.0, 1.0),
        arrow(4, 2, 1.0, -1.0),
        arrow(4, 1, 1.0, -1.0),
    ],
    tall: 0.62,
};

const GRAPHS: [&GraphDef; 3] = [&PLAIN, &M_SHAPE, &BUTTERFLY];

/// Variables of a graph as a set: bit `i` for variable `i`.
type Set = u32;

fn has(set: Set, i: usize) -> bool {
    i < 32 && set >> i & 1 == 1
}

/// The names of a set, in order ("$Z$, $I$"), or `none` if it is empty.
fn names(g: &GraphDef, set: Set, none: &str) -> String {
    let names: Vec<&str> = (0..g.vars.len()).filter(|&i| has(set, i)).map(|i| g.vars[i].name).collect();
    match names.as_slice() {
        [] => none.to_string(),
        [one] => one.to_string(),
        [first @ .., last] => format!("{} and {last}", first.join(", ")),
    }
}

// ----- the model: straight-line effects, each variable with noise of its own ------------

/// A graph with the strengths its arrows have now. Every variable is the sum of its
/// causes, each times the strength of its arrow, and a noise of its own (normal, mean 0,
/// variance 1): `v = B^T v + e`, so `v = T e` with `T = (I - B^T)^{-1}`.
struct Model {
    p: usize,
    /// `t[i * p + j]`: how much of variable j's own noise there is in variable i.
    t: Vec<f64>,
}

impl Model {
    fn new(g: &GraphDef, strengths: &[f64]) -> Model {
        let p = g.vars.len();
        // in an order where every cause comes before its effects, each variable is
        // its own noise plus its causes' shares
        let mut t = vec![0.0; p * p];
        for &i in &order(g) {
            t[i * p + i] = 1.0;
            for (a, s) in g.arrows.iter().zip(strengths).filter(|(a, _)| a.to == i) {
                for j in 0..p {
                    t[i * p + j] += s * t[a.from * p + j];
                }
            }
        }
        Model { p, t }
    }

    /// The covariances of the variables, given those of the noises (`MOST` × `MOST`,
    /// of which the first `p` rows and columns are used): `T S T^T`.
    fn covariance(&self, noise: &[f64; MOST * MOST]) -> Vec<f64> {
        let p = self.p;
        let mut ts = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..p {
                ts[i * p + j] = (0..p).map(|k| self.t[i * p + k] * noise[k * MOST + j]).sum();
            }
        }
        let mut c = vec![0.0; p * p];
        for i in 0..p {
            for j in 0..p {
                c[i * p + j] = (0..p).map(|k| ts[i * p + k] * self.t[j * p + k]).sum();
            }
        }
        c
    }

    /// The true total effect of X on Y: how much Y moves when X is set one higher
    /// and everything else is left to follow (Y's share of X's own noise).
    fn truth(&self) -> f64 {
        self.t[self.p]
    }

    /// The values of every variable for the units whose noises are `noise` (`MOST` per
    /// unit, of which the first `p` are used), `p` per unit.
    fn values(&self, noise: &[f64]) -> Vec<f64> {
        let p = self.p;
        noise.chunks_exact(MOST).flat_map(|e| (0..p).map(move |i| (0..p).map(|j| self.t[i * p + j] * e[j]).sum::<f64>())).collect()
    }
}

/// The variables in an order where every arrow goes forwards.
fn order(g: &GraphDef) -> Vec<usize> {
    let p = g.vars.len();
    let mut done: Vec<usize> = Vec::with_capacity(p);
    while done.len() < p {
        match (0..p).find(|&i| !done.contains(&i) && g.arrows.iter().all(|a| a.to != i || done.contains(&a.from))) {
            Some(i) => done.push(i),
            // (a cycle: no graph here has one)
            None => done.extend((0..p).filter(|i| !done.contains(i)).collect::<Vec<_>>()),
        }
    }
    done
}

/// The coefficient of X when Y is fitted by least squares on X and the variables `held`
/// (and a constant), from the variables' covariances `c` (p × p). None if X is a
/// combination of the variables held, when nothing settles it.
fn coefficient(c: &[f64], p: usize, held: Set) -> Option<f64> {
    let on: Vec<usize> = std::iter::once(0).chain((2..p).filter(|&i| has(held, i))).collect();
    let k = on.len();
    let mut a: Vec<f64> = on.iter().flat_map(|&i| on.iter().map(move |&j| c[i * p + j])).collect();
    let mut rhs: Vec<f64> = on.iter().map(|&i| c[i * p + 1]).collect();
    solve(&mut a, &mut rhs, k).and_then(|w| w.first().copied()).filter(|v| v.is_finite())
}

/// The noises of one sample: each unit's own noise for every variable a graph can
/// have (`MOST` per unit), and their covariances (dividing by the number of units, as
/// a least-squares fit with a constant does). The variables of any graph, with any
/// strengths, follow from these: their covariances are `T S T^T`.
struct Sample {
    noise: Vec<f64>,
    mean: [f64; MOST],
    cov: [f64; MOST * MOST],
}

impl Sample {
    fn draw(rng: &mut Rng, units: usize) -> Sample {
        let noise: Vec<f64> = (0..units * MOST).map(|_| rng.normal() as f64).collect();
        let mean = noise_mean(&noise);
        let cov = noise_covariance(&noise, &mean);
        Sample { noise, mean, cov }
    }
}

fn noise_mean(noise: &[f64]) -> [f64; MOST] {
    let n = (noise.len() / MOST).max(1) as f64;
    let mut mean = [0.0; MOST];
    for row in noise.chunks_exact(MOST) {
        for (m, v) in mean.iter_mut().zip(row) {
            *m += v / n;
        }
    }
    mean
}

fn noise_covariance(noise: &[f64], mean: &[f64; MOST]) -> [f64; MOST * MOST] {
    let n = (noise.len() / MOST).max(1) as f64;
    let mut cov = [0.0; MOST * MOST];
    for row in noise.chunks_exact(MOST) {
        let d: [f64; MOST] = std::array::from_fn(|i| row[i] - mean[i]);
        for i in 0..MOST {
            for j in i..MOST {
                cov[i * MOST + j] += d[i] * d[j];
            }
        }
    }
    for i in 0..MOST {
        for j in i..MOST {
            cov[i * MOST + j] /= n;
            cov[j * MOST + i] = cov[i * MOST + j];
        }
    }
    cov
}

/// The covariances of noises with no sampling in them: what endless data would give.
fn endless() -> [f64; MOST * MOST] {
    std::array::from_fn(|k| if k / MOST == k % MOST { 1.0 } else { 0.0 })
}

// ----- paths between X and Y, and what holding variables does to them --------------------

/// A path between X and Y along the arrows, whichever way they point.
#[derive(Clone, Debug, PartialEq)]
struct Path {
    /// From X to Y.
    nodes: Vec<usize>,
    /// The arrow of each step (an index into the graph's arrows).
    arrows: Vec<usize>,
    /// Whether each step's arrow points along the path, towards Y.
    along: Vec<bool>,
}

impl Path {
    /// All arrows point from X towards Y: a path the effect travels along.
    fn causal(&self) -> bool {
        self.along.iter().all(|a| *a)
    }

    /// Starts with an arrow into X.
    fn back_door(&self) -> bool {
        self.along.first() == Some(&false)
    }

    /// Is the `k`-th variable on it (counting X as 0) one where two arrows meet?
    fn collider(&self, k: usize) -> bool {
        k > 0 && k + 1 < self.nodes.len() && self.along[k - 1] && !self.along[k]
    }

    /// The path written out: `X \gets Z \to Y`.
    fn markup(&self, g: &GraphDef) -> String {
        let mut s = g.vars[self.nodes[0]].letter.to_string();
        for (k, along) in self.along.iter().enumerate() {
            s += if *along { r" \to " } else { r" \gets " };
            s += g.vars[self.nodes[k + 1]].letter;
        }
        s
    }
}

/// Every path from X to Y that visits no variable twice: causal ones first, then back
/// doors, then the rest, shorter before longer.
fn paths(g: &GraphDef) -> Vec<Path> {
    let mut out = Vec::new();
    let mut stack = vec![Path { nodes: vec![0], arrows: Vec::new(), along: Vec::new() }];
    while let Some(path) = stack.pop() {
        let Some(&at) = path.nodes.last() else { continue };
        if at == 1 {
            out.push(path);
            continue;
        }
        for (k, a) in g.arrows.iter().enumerate().rev() {
            let (next, along) = if a.from == at {
                (a.to, true)
            } else if a.to == at {
                (a.from, false)
            } else {
                continue;
            };
            if path.nodes.contains(&next) {
                continue;
            }
            let mut longer = path.clone();
            longer.nodes.push(next);
            longer.arrows.push(k);
            longer.along.push(along);
            stack.push(longer);
        }
    }
    let rank = |p: &Path| (if p.causal() { 0 } else if p.back_door() { 1 } else { 2 }, p.nodes.len());
    out.sort_by_key(rank);
    out
}

/// What a path does once some variables are held.
#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    /// Nothing on it is held and no two arrows meet on it: it carries association
    /// from end to end (the effect, if it is causal).
    Open,
    /// Two arrows meet at `collider`, and `held` is held (the collider itself, or
    /// something it causes): that opens it.
    Opened { collider: usize, held: usize },
    /// Blocked at a held variable where the arrows do not meet.
    Blocked(usize),
    /// Closed at a variable where two arrows meet and that is not held (nor is
    /// anything it causes).
    Closed(usize),
    /// One of its arrows has strength 0: it is no path at all.
    Absent(usize),
}

impl State {
    fn open(self) -> bool {
        matches!(self, State::Open | State::Opened { .. })
    }
}

/// The arrows as they are: a strength of 0 is no arrow.
fn present(strength: f64) -> bool {
    strength != 0.0
}

/// Variable `i` and everything it causes, through arrows that are there.
fn descendants(g: &GraphDef, strengths: &[f64], i: usize) -> Set {
    let mut set: Set = 1 << i;
    loop {
        let more = g.arrows.iter().zip(strengths).filter(|(a, s)| present(**s) && has(set, a.from)).fold(set, |s, (a, _)| s | 1 << a.to);
        if more == set {
            return set;
        }
        set = more;
    }
}

/// Everything `i` causes through arrows that are there without passing through `avoid`
/// (`i` itself included).
fn reach_avoiding(g: &GraphDef, strengths: &[f64], i: usize, avoid: usize) -> Set {
    let mut set: Set = 1 << i;
    loop {
        let more = g
            .arrows
            .iter()
            .zip(strengths)
            .filter(|(a, s)| present(**s) && has(set, a.from) && a.to != avoid)
            .fold(set, |s, (a, _)| s | 1 << a.to);
        if more == set {
            return set;
        }
        set = more;
    }
}

/// What variable `v` is to X and Y through the arrows that are there: a mediator (on
/// the way from X to Y), a confounder (a cause of X, and of Y not through X), a cause
/// of one of them only, a collider (where two arrows of a path between them meet), an
/// effect of one, or nothing.
fn kind_of(g: &GraphDef, strengths: &[f64], v: usize) -> &'static str {
    let (from_x, from_y) = (has(descendants(g, strengths, 0), v), has(descendants(g, strengths, 1), v));
    let there = |p: &Path| p.arrows.iter().all(|&k| strengths.get(k).is_some_and(|s| present(*s)));
    let collider = paths(g).iter().any(|p| there(p) && (1..p.nodes.len().saturating_sub(1)).any(|k| p.nodes[k] == v && p.collider(k)));
    let to_x = has(descendants(g, strengths, v), 0);
    let to_y = has(reach_avoiding(g, strengths, v, 0), 1);
    if from_x && to_y {
        "mediator"
    } else if to_x && to_y {
        "confounder"
    } else if to_x {
        "cause of $X$ only"
    } else if to_y {
        "cause of $Y$ only"
    } else if collider {
        "collider"
    } else if from_y {
        "effect of $Y$"
    } else if from_x {
        "effect of $X$"
    } else {
        ""
    }
}

fn state(g: &GraphDef, strengths: &[f64], path: &Path, held: Set) -> State {
    if let Some(&k) = path.arrows.iter().find(|&&k| !strengths.get(k).is_some_and(|s| present(*s))) {
        return State::Absent(k);
    }
    let mut opened = None;
    for k in 1..path.nodes.len().saturating_sub(1) {
        let v = path.nodes[k];
        if path.collider(k) {
            let below = descendants(g, strengths, v) & held;
            if below == 0 {
                return State::Closed(v);
            }
            if opened.is_none() {
                // (the collider itself if it is held, else the first thing it causes that is)
                let held = if has(held, v) { v } else { below.trailing_zeros() as usize };
                opened = Some(State::Opened { collider: v, held });
            }
        } else if has(held, v) {
            return State::Blocked(v);
        }
    }
    opened.unwrap_or(State::Open)
}

/// What is wrong with a set of variables to hold, from the arrows alone.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Errors {
    /// A back door is open: something that causes both X and Y is not held.
    confounding: bool,
    /// A causal path is blocked: a mediator is held.
    part_removed: bool,
    /// A path was opened by holding a collider (or something it causes).
    collider: bool,
    /// Something Y causes (or a mediator causes) is held, which none of the above explains.
    effect_held: bool,
}

impl Errors {
    fn none(&self) -> bool {
        *self == Errors::default()
    }
}

/// The paths with their states, and the errors they add up to. A set is right (the
/// coefficient of X is then the total effect, given data enough) exactly when it
/// blocks every path that does not carry the effect, blocks none that does, and holds
/// nothing that is caused by a variable on a causal path other than X: the adjustment
/// criterion of Shpitser, VanderWeele and Robins (2010).
fn judge(g: &GraphDef, strengths: &[f64], all: &[Path], held: Set) -> (Vec<State>, Errors) {
    let states: Vec<State> = all.iter().map(|p| state(g, strengths, p, held)).collect();
    let mut e = Errors::default();
    // what lies on causal paths, and what held variable is accounted for by a path
    let (mut on_causal, mut explained): (Set, Set) = (0, 0);
    for (p, s) in all.iter().zip(&states) {
        if matches!(s, State::Absent(_)) {
            continue;
        }
        if p.causal() {
            on_causal |= p.nodes[1..].iter().fold(0, |m, &v| m | 1 << v);
            if let State::Blocked(v) = s {
                e.part_removed = true;
                explained |= 1 << v;
            }
        } else if let State::Opened { held, .. } = s {
            e.collider = true;
            explained |= 1 << held;
        } else if s.open() {
            e.confounding = true;
        }
    }
    // what the variables on causal paths (other than X) cause may not be held
    let forbidden = (1..g.vars.len()).filter(|&v| has(on_causal, v)).fold(0, |m, v| m | descendants(g, strengths, v));
    e.effect_held = forbidden & held & !explained != 0;
    (states, e)
}

/// The rule (the back-door criterion): the set blocks every path that starts with an
/// arrow into X, and holds nothing X causes. Enough for a set to be right, whatever
/// the strengths; not always needed.
fn rule(g: &GraphDef, strengths: &[f64], all: &[Path], states: &[State], held: Set) -> (bool, bool) {
    let doors = all.iter().zip(states).all(|(p, s)| !p.back_door() || !s.open());
    let caused = descendants(g, strengths, 0) & !1;
    (doors, held & caused == 0)
}

// ----- what the settings give -------------------------------------------------------------

/// Every set of measured variables of a graph, the empty one first.
fn every_set(g: &GraphDef) -> Vec<Set> {
    let measured: Vec<usize> = (0..g.vars.len()).filter(|&i| g.vars[i].role == Role::Measured).collect();
    (0..1u32 << measured.len()).map(|m| measured.iter().enumerate().filter(|(b, _)| m >> b & 1 == 1).fold(0, |s, (_, &i)| s | 1 << i)).collect()
}

/// What the present settings give, worked out whenever they change.
struct Reading {
    paths: Vec<Path>,
    states: Vec<State>,
    errors: Errors,
    /// The rule's two parts: every back door blocked; nothing X causes held.
    rule: (bool, bool),
    truth: f64,
    /// The strengths along each causal path that is there (their products add up to the truth).
    terms: Vec<Vec<f64>>,
    /// The coefficient of X from this sample, and from endless data.
    estimate: Option<f64>,
    endless: Option<f64>,
    /// From each of the other samples drawn so far.
    others: Vec<Option<f64>>,
    /// Every set that could be held: the set, its estimate from this sample, and whether it is right.
    sets: Vec<(Set, Option<f64>, bool)>,
    /// The stretch of the axis that shows all of them.
    range: (f64, f64),
}

impl Reading {
    fn of(g: &GraphDef, strengths: &[f64], held: Set, sample: &Sample, others: &[[f64; MOST * MOST]]) -> Reading {
        let p = g.vars.len();
        let model = Model::new(g, strengths);
        let paths = paths(g);
        let (states, errors) = judge(g, strengths, &paths, held);
        let rule = rule(g, strengths, &paths, &states, held);
        let (here, ever) = (model.covariance(&sample.cov), model.covariance(&endless()));
        let terms: Vec<Vec<f64>> = paths
            .iter()
            .filter(|path| path.causal() && path.arrows.iter().all(|&k| strengths.get(k).is_some_and(|s| present(*s))))
            .map(|path| path.arrows.iter().map(|&k| strengths[k]).collect())
            .collect();
        let sets: Vec<(Set, Option<f64>, bool)> = every_set(g).into_iter().map(|s| (s, coefficient(&here, p, s), judge(g, strengths, &paths, s).1.none())).collect();
        let truth = model.truth();
        // the axis: every set's estimate, from this sample and from endless data, the truth and 0
        let values = sets.iter().filter_map(|s| s.1).chain(every_set(g).into_iter().filter_map(|s| coefficient(&ever, p, s))).chain([truth, 0.0]);
        let (lo, hi) = values.filter(|v| v.is_finite()).fold((f64::MAX, f64::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
        let pad = 0.08 * (hi - lo).max(0.5) + 0.06;
        Reading {
            estimate: coefficient(&here, p, held),
            endless: coefficient(&ever, p, held),
            others: others.iter().map(|c| coefficient(&model.covariance(c), p, held)).collect(),
            paths,
            states,
            errors,
            rule,
            truth,
            terms,
            sets,
            range: (lo - pad, hi + pad),
        }
    }

    /// What the present set gets wrong, in words ("" if nothing).
    fn errors_named(&self) -> String {
        let e = self.errors;
        let names: Vec<&str> = [(e.confounding, "confounding"), (e.part_removed, "part of the effect removed"), (e.collider, "a collider opened"), (e.effect_held, "an effect of $Y$ held")]
            .into_iter()
            .filter(|(on, _)| *on)
            .map(|(_, name)| name)
            .collect();
        match names.as_slice() {
            [] => String::new(),
            [one] => one.to_string(),
            [first @ .., last] => format!("{} and {last}", first.join(", ")),
        }
    }

    /// The spread of the other samples' estimates (their standard deviation), once
    /// there are a few.
    fn spread(&self) -> Option<f64> {
        let v: Vec<f64> = self.others.iter().flatten().copied().collect();
        (v.len() >= 5).then(|| {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
        })
    }
}

/// The truth as a sum, one term for each causal path, the product of its strengths:
/// `0.50 + 0.80 \cdot 0.50 = 0.90` (markup, without the dollars).
fn truth_sum(terms: &[Vec<f64>], truth: f64) -> String {
    if terms.is_empty() {
        return "0".to_string();
    }
    if terms.len() == 1 && terms[0].len() == 1 {
        return num(truth, 2);
    }
    // (a factor below 0 is put in brackets, but for the very first)
    let factor = |v: f64, first: bool| if !first && num(v, 2).starts_with('−') { format!("({})", num(v, 2)) } else { num(v, 2) };
    let sum: Vec<String> = terms.iter().enumerate().map(|(i, t)| t.iter().enumerate().map(|(j, v)| factor(*v, i == 0 && j == 0)).collect::<Vec<_>>().join(r" \cdot ")).collect();
    format!("{} = {}", sum.join(" + "), num(truth, 2))
}

/// A difference with its sign, to two places: +0.24, −0.40, 0.00.
fn diff(v: f64) -> String {
    let n = num(v, 2);
    if v > 0.0 && n != "0.00" {
        format!("+{n}")
    } else {
        n
    }
}

/// A path that carries nothing into the fit.
const BLOCKED: Color = Color::hex(0x5A6478);

/// The colour a path is drawn in: what it carries into the fit (the effect, bias, or
/// nothing), `open` from 0 to 1.
fn path_color(causal: bool, open: f32) -> Color {
    let base = if causal { TRUTH } else { NAIVE };
    BLOCKED.mix(base, open.clamp(0.0, 1.0))
}

/// Why a path is as it is, in a few words (markup), and whether that is as it should
/// be; `short` for a narrow canvas.
fn reason(g: &GraphDef, path: &Path, state: State, short: bool) -> (String, bool) {
    let name = |i: usize| g.vars.get(i).map_or("?", |v| v.name);
    let middle: Vec<&str> = path.nodes.iter().skip(1).take(path.nodes.len().saturating_sub(2)).map(|&v| name(v)).collect();
    let (long, brief, fine) = match state {
        State::Absent(k) => {
            let arrow = g.arrows.get(k).map_or(String::new(), |a| format!(r"${} \to {}$", g.vars[a.from].letter, g.vars[a.to].letter));
            (format!("no path: {arrow} has strength 0"), format!("none: {arrow} is 0"), true)
        }
        State::Open if path.causal() && middle.is_empty() => ("open: the direct effect".to_string(), "open: the direct effect".to_string(), true),
        State::Open if path.causal() => (format!("open: the effect through {}", middle.join(" and ")), format!("open: the effect via {}", middle.join(", ")), true),
        State::Open => match middle.as_slice() {
            [one] => (format!("open back door: {one} is not held"), format!("open: {one} not held"), false),
            _ => ("open back door: nothing on it is held".to_string(), "open: nothing held".to_string(), false),
        },
        State::Opened { collider, held } if collider == held => {
            (format!("opened by holding {}, where two arrows meet", name(collider)), format!("opened: collider {} held", name(collider)), false)
        }
        State::Opened { collider, held } => (format!("opened by holding {}, which {} causes", name(held), name(collider)), format!("opened: {} held", name(held)), false),
        State::Blocked(v) if path.causal() => {
            (format!("blocked at {}, which is held: part of the effect is lost", name(v)), format!("blocked at {}: effect lost", name(v)), false)
        }
        State::Blocked(v) => (format!("blocked at {}, which is held", name(v)), format!("blocked: {} held", name(v)), true),
        State::Closed(v) => (format!("blocked at {}, where two arrows meet", name(v)), format!("blocked at collider {}", name(v)), true),
    };
    (if short { brief } else { long }, fine)
}

// ----- where things go --------------------------------------------------------------------

type Rect = (Vec2, Vec2);

fn inside(p: Vec2, (lo, hi): Rect) -> bool {
    p.cmpge(lo).all() && p.cmple(hi).all()
}

/// Where the parts of the picture go on a canvas (logical pixels from its top-left corner).
#[derive(Clone, Copy, Debug)]
struct Lay {
    /// One part above another (a phone held upright), not side by side.
    narrow: bool,
    text: f32,
    small: f32,
    /// The buttons that choose the graph.
    tabs: Option<Rect>,
    graph: Rect,
    /// The line under the graph that says what a click there does.
    status: Option<Vec2>,
    /// The list of paths, and how its rows are written.
    list: Option<Rect>,
    rows: Rows,
    /// What the fit sees (beside the graph, under the list, on a wide canvas only).
    scatter: Option<Rect>,
    /// The comparison of the estimate with the truth.
    gauge: Option<Rect>,
    /// The radius of a variable's disc.
    radius: f32,
}

/// How the rows of the list of paths are written: the path and why on one line or two,
/// with the reasons in full or short, and how wide the widest path is.
#[derive(Clone, Copy, Debug, Default)]
struct Rows {
    one_line: bool,
    short: bool,
    formula: f32,
    height: f32,
}

/// A canvas laid out side by side, not one part above another.
fn is_wide(view: Vec2) -> bool {
    view.x >= 700.0 && view.y >= 560.0
}

/// The sizes of text, and the left and right of the list of paths.
fn list_metrics(view: Vec2) -> (f32, f32, f32, f32) {
    let side = 14.0;
    if is_wide(view) {
        (14.5, 12.5, side + (view.x - 2.0 * side) * 0.5 + 34.0, view.x - side)
    } else {
        (13.0, 11.5, side, view.x - side)
    }
}

impl Lay {
    fn of(view: Vec2, inset: f32, count: usize, rows: Rows) -> Lay {
        let view = view.max(Vec2::ONE);
        let top = inset.max(0.0) + 10.0;
        let side = 14.0;
        let (text, small, left, right) = list_metrics(view);
        let list_h = text + small + 24.0 + count as f32 * rows.height + small + 12.0;
        if is_wide(view) {
            // the graph on the left, the list and what the fit sees on the right, the
            // gauge across the foot
            let split = left - 34.0;
            let gauge_h = (view.y * 0.3).clamp(232.0, 290.0);
            let bottom = view.y - gauge_h - 10.0;
            let list = (Vec2::new(left, top + 4.0), Vec2::new(right, top + 4.0 + list_h));
            let scatter = (list.1.y + 30.0, bottom - 6.0);
            Lay {
                narrow: false,
                text,
                small,
                tabs: Some((Vec2::new(side, top), Vec2::new(split, top + 28.0))),
                graph: (Vec2::new(side, top + 46.0), Vec2::new(split, bottom - 30.0)),
                status: Some(Vec2::new(side + 4.0, bottom - 12.0)),
                list: Some(list),
                rows,
                scatter: (scatter.1 - scatter.0 >= 150.0).then(|| (Vec2::new(left, scatter.0), Vec2::new(right, scatter.1))),
                gauge: Some((Vec2::new(side, bottom + 10.0), Vec2::new(view.x - side, view.y - 10.0))),
                radius: 18.0,
            }
        } else {
            // one above another: tabs, graph, list, gauge; on a canvas too short for
            // all of them the list is left out first, then the gauge
            let gauge_h = 192.0;
            let tabs_h = if view.y >= 300.0 && view.x >= 300.0 { 36.0 } else { 0.0 };
            let room = view.y - top - tabs_h;
            let graph_min = 150.0;
            let with_list = room >= graph_min + 24.0 + list_h + 26.0 + gauge_h;
            let with_gauge = room >= graph_min + 24.0 + gauge_h;
            let used = if with_list {
                list_h + 26.0 + gauge_h
            } else if with_gauge {
                gauge_h
            } else {
                0.0
            };
            let graph_h = (room - used - 24.0).min(view.x * 0.8).max(10.0);
            let g0 = top + tabs_h;
            let g1 = g0 + graph_h;
            let list = with_list.then(|| (Vec2::new(side, g1 + 30.0), Vec2::new(view.x - side, g1 + 30.0 + list_h)));
            let gauge_top = list.map_or(g1 + 30.0, |l| l.1.y + 4.0);
            let gauge = with_gauge.then(|| (Vec2::new(side, gauge_top), Vec2::new(view.x - side, (gauge_top + gauge_h).min(view.y - 4.0))));
            Lay {
                narrow: true,
                text,
                small,
                tabs: (tabs_h > 0.0).then(|| (Vec2::new(side, top), Vec2::new(view.x - side, top + 26.0))),
                graph: (Vec2::new(side, g0), Vec2::new(view.x - side, g1.max(g0 + 10.0))),
                status: with_gauge.then(|| Vec2::new(side + 2.0, g1 + 12.0)),
                list,
                rows,
                scatter: None,
                gauge,
                radius: if view.x < 360.0 { 13.0 } else { 15.0 },
            }
        }
    }
}

/// Where a graph's variables go in the rectangle `area`, leaving room round them for
/// their names and labels (`margin` either side, `above` and `below`).
fn places(g: &GraphDef, (lo, hi): Rect, margin: f32, above: f32, below: f32) -> Vec<Vec2> {
    let avail = Vec2::new((hi.x - lo.x - 2.0 * margin).max(1.0), (hi.y - lo.y - above - below).max(1.0));
    // the graph's own shape, stretched by at most a third either way to fill the room
    let w = avail.x.min(avail.y / g.tall * 1.35);
    let h = avail.y.min(w * g.tall * 1.35);
    let origin = Vec2::new(lo.x + margin + (avail.x - w) * 0.5, lo.y + above + (avail.y - h) * 0.5);
    g.vars.iter().map(|v| origin + Vec2::new(v.at.0 * w, v.at.1 * h)).collect()
}

/// A button over the picture: true when it is clicked. The one chosen is drawn lit.
fn hud_button(f: &mut Frame, (min, max): Rect, text: &str, size: f32, chosen: bool, pointer: Option<Vec2>) -> bool {
    let over = pointer.is_some_and(|p| inside(p, (min, max)));
    let fill = match (chosen, over) {
        (true, _) => Color::hex(0x3A4C78),
        (false, true) => Color::hex(0x33405E),
        (false, false) => Color::hex(0x222B40),
    };
    hud_rect(f, min, max, fill);
    f.text_px((min + max) * 0.5, text, size, if chosen { pal::WHITE } else { pal::TEXT }, Align2::CENTER_CENTER);
    over && f.ptr.clicked
}

/// A filled disc in screen space, under the lines and labels.
fn hud_disc(f: &mut Frame, centre: Vec2, radius: f32, color: Color) {
    let n = 32;
    let c = f.hud_at(1, centre);
    let at = |f: &Frame, i: usize| f.hud_at(1, centre + Vec2::from_angle(i as f32 / n as f32 * TAU) * radius);
    for i in 0..n {
        let (a, b) = (at(f, i), at(f, i + 1));
        f.dl.tris_alpha.tri(c, a, b, color);
    }
}

/// Levels for the marks of the sets, so that marks closer than `gap` pixels go one
/// under another: each mark's level, in the order given.
fn levels(xs: &[f32], gap: f32) -> Vec<usize> {
    let mut order: Vec<usize> = (0..xs.len()).collect();
    order.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
    let mut ends: Vec<f32> = Vec::new();
    let mut out = vec![0; xs.len()];
    for i in order {
        let level = ends.iter().position(|&e| xs[i] - e >= gap).unwrap_or(ends.len());
        if level == ends.len() {
            ends.push(xs[i]);
        } else {
            ends[level] = xs[i];
        }
        out[i] = level;
    }
    out
}

/// The axis of the gauge: from value to pixel.
#[derive(Clone, Copy)]
struct Axis {
    range: (f64, f64),
    x0: f32,
    x1: f32,
}

impl Axis {
    fn x(&self, v: f64) -> f32 {
        let t = (v - self.range.0) / (self.range.1 - self.range.0).max(1e-9);
        let x = self.x0 + (self.x1 - self.x0) * t.clamp(-0.02, 1.02) as f32;
        if x.is_finite() {
            x
        } else {
            self.x0
        }
    }
}

// ----- the scene ---------------------------------------------------------------------------

/// How long a change takes to glide into place: the time in which it covers about two
/// thirds of the way (seconds).
const GLIDE: f32 = 0.09;

/// The experiments: (button, what to look for, graph, the letters held). Each starts
/// from the graph's own strengths and the sample the scene opens with.
const TRIALS: [(&str, &str, usize, &[&str]); 5] = [
    (
        "hold nothing",
        r"Fitted on $X$ alone, $Y$ rises by 1.14 for each unit of $X$, but the true effect is 0.90. The back door $X \gets Z \to Y$ is open: units with more $X$ tend to have more $Z$, which raises $Y$ too. That is confounding, and more data would not cure it: the 40 other samples make the same error.",
        0,
        &[],
    ),
    (
        "hold the confounder",
        r"Holding $Z$ blocks the back door and leaves the paths along the arrows open. The estimate, 0.91, is the true effect but for the luck of this sample: the 40 other samples centre on 0.90.",
        0,
        &["Z"],
    ),
    (
        "hold everything",
        r"More is not better. Holding $M$ blocks $X \to M \to Y$, the part of the effect that goes through $M$; holding $C$, where two arrows meet, opens $X \to C \gets Y$. The estimate falls to 0.10. Let $C$ go and it is 0.48, near the direct effect alone (0.50, the arrow $X \to Y$); hold $C$ again and let $M$ go, and it is 0.31.",
        0,
        &["Z", "M", "C", "I", "P"],
    ),
    (
        "a cause of $X$ only",
        r"$Z$ and $I$ held: right on average, but the faint dots spread about twice as wide as with $Z$ alone. Holding $I$ takes away much of the spread of $X$, so there is less of it to learn from. Hold $P$ instead and the dots draw together, as $P$ takes away part of the spread of $Y$. Hold $I$ without $Z$ and the error doubles, from $+0.24$ to $+0.48$.",
        0,
        &["Z", "I"],
    ),
    (
        "the ‘M’ shape",
        r"$B$ is measured before $X$ and goes with both $X$ and $Y$, so it looks like a confounder. But two causes nobody measured meet at it: holding $B$ as well as $Z$ opens $X \gets U_1 \to B \gets U_2 \to Y$, and the estimate falls from 0.52 to 0.31 (the true effect is 0.50). Let $B$ go.",
        1,
        &["Z", "B"],
    ),
];

/// How many units of the sample are drawn in the picture of what the fit sees.
const CLOUD: usize = 600;

/// The units of a sample as the fit sees them: X and Y, each less its least-squares
/// fit (with a constant) on the held variables, for the first `count` units. The
/// slope of Y's left-overs against X's is the coefficient of X in the fit of Y on X
/// and the held variables (the theorem of Frisch, Waugh and Lovell).
fn left_over(model: &Model, sample: &Sample, held: Set, count: usize) -> Vec<(f64, f64)> {
    let p = model.p;
    let cov = model.covariance(&sample.cov);
    let c = &cov;
    let on: Vec<usize> = (2..p).filter(|&i| has(held, i)).collect();
    let k = on.len();
    // the fits of X and of Y on the held variables
    let fit = |target: usize| -> Vec<f64> {
        if k == 0 {
            return Vec::new();
        }
        let mut a: Vec<f64> = on.iter().flat_map(|&i| on.iter().map(move |&j| c[i * p + j])).collect();
        let mut b: Vec<f64> = on.iter().map(|&i| c[i * p + target]).collect();
        solve(&mut a, &mut b, k).unwrap_or_else(|| vec![0.0; k])
    };
    let (bx, by) = (fit(0), fit(1));
    let mean: Vec<f64> = (0..p).map(|i| (0..p).map(|j| model.t[i * p + j] * sample.mean[j]).sum()).collect();
    let units = (sample.noise.len() / MOST).min(count);
    let values = model.values(&sample.noise[..units * MOST]);
    values
        .chunks_exact(p)
        .map(|v| {
            let d = |i: usize| v[i] - mean[i];
            let less = |b: &[f64]| on.iter().zip(b).map(|(&i, b)| b * d(i)).sum::<f64>();
            (d(0) - less(&bx), d(1) - less(&by))
        })
        .collect()
}

/// What the panel says about each graph.
const GRAPH_NOTES: [&str; 3] = [
    r"Built with one variable of each kind: a confounder $Z$, a mediator $M$, a collider $C$, a cause of $X$ only, $I$, and a cause of $Y$ only, $P$. (An arrow set to 0 can change what a variable is: the labels follow.)",
    r"$B$ is measured before $X$, and goes with both $X$ and $Y$, so it looks like a confounder. It is not: two causes nobody measured, $U_1$ and $U_2$, meet at it.",
    r"$B$ is a confounder ($X \gets B \to Y$) and a collider ($A \to B \gets D$) at once. Holding it closes three back doors and opens a fourth, which holding $A$ or $D$ as well closes again.",
];

/// What is drawn, gliding towards what the settings give.
struct Shown {
    estimate: f64,
    others: Vec<f64>,
    sets: Vec<f64>,
    range: (f64, f64),
    truth: f64,
    /// How open each path is drawn, from 0 (blocked) to 1.
    open: Vec<f32>,
    /// How held each variable is drawn, from 0 to 1.
    held: [f32; MOST],
    /// The units as the fit sees them, and how far the picture of them reaches either way.
    cloud: Vec<(f64, f64)>,
    reach: (f64, f64),
}

struct Adjust {
    graph: usize,
    /// Each graph's strengths and held set, kept while another graph is shown.
    strengths: [Vec<f64>; 3],
    held: [Set; 3],
    seed: u64,
    sample: Sample,
    /// The covariances of the other samples' noises, drawn a few at a time.
    others: Vec<[f64; MOST * MOST]>,
    rng: Rng,
    now: Reading,
    /// The units as the fit sees them, and how far their picture reaches either way.
    cloud: Vec<(f64, f64)>,
    reach: (f64, f64),
    shown: Shown,
    /// The settings changed: work them out again.
    dirty: bool,
    /// The graph changed: draw the new one as it is, without gliding.
    snap: bool,
    moving: bool,
    /// The path the pointer is on, and the one clicked (shown alone on the graph).
    hover: Option<usize>,
    picked: Option<usize>,
    /// The variable, and the mark of a set, the pointer is on.
    hover_var: Option<usize>,
    hover_set: Option<usize>,
    /// How the list's rows are written, for the graph and canvas width it was decided for.
    rows_kept: Option<((usize, i32, bool), Rows)>,
    /// The size of the canvas the picture is laid out for.
    canvas: Vec2,
}

impl Adjust {
    fn new() -> Adjust {
        let seed = SEED;
        let mut rng = Rng::new(seed);
        let sample = Sample::draw(&mut rng, UNITS);
        let strengths = GRAPHS.map(|g| g.arrows.iter().map(|a| a.strength).collect::<Vec<f64>>());
        let now = Reading::of(GRAPHS[0], &strengths[0], 0, &sample, &[]);
        let shown = Shown {
            estimate: 0.0,
            others: Vec::new(),
            sets: Vec::new(),
            range: now.range,
            truth: now.truth,
            open: Vec::new(),
            held: [0.0; MOST],
            cloud: Vec::new(),
            reach: (1.0, 1.0),
        };
        Adjust {
            graph: 0,
            strengths,
            held: [0; 3],
            seed,
            sample,
            others: Vec::new(),
            rng,
            now,
            cloud: Vec::new(),
            reach: (1.0, 1.0),
            shown,
            dirty: true,
            snap: true,
            moving: false,
            hover: None,
            picked: None,
            hover_var: None,
            hover_set: None,
            rows_kept: None,
            canvas: Vec2::ZERO,
        }
    }

    fn g(&self) -> &'static GraphDef {
        GRAPHS[self.graph]
    }

    fn held(&self) -> Set {
        self.held[self.graph]
    }

    /// The variables with these letters, as a set.
    fn set_of(g: &GraphDef, letters: &[&str]) -> Set {
        letters.iter().filter_map(|n| g.vars.iter().position(|v| v.letter == *n)).fold(0, |s, i| s | 1 << i)
    }

    fn choose_graph(&mut self, k: usize) {
        if k < GRAPHS.len() && k != self.graph {
            self.graph = k;
            (self.hover, self.picked, self.hover_var, self.hover_set) = (None, None, None, None);
            self.snap = true;
            self.dirty = true;
        }
    }

    /// Set up one of the experiments in `TRIALS`: its graph with the strengths it
    /// starts with, its variables held, and the first sample.
    fn set_up(&mut self, k: usize) {
        let Some(&(_, _, graph, letters)) = TRIALS.get(k) else { return };
        self.choose_graph(graph);
        let g = self.g();
        self.strengths[graph] = g.arrows.iter().map(|a| a.strength).collect();
        self.held[graph] = Self::set_of(g, letters);
        self.picked = None;
        if self.seed != SEED {
            self.new_sample(SEED);
        }
        self.dirty = true;
    }

    /// Hold the variable `i`, or let it go.
    fn toggle(&mut self, i: usize) {
        if self.g().vars.get(i).is_some_and(|v| v.role == Role::Measured) {
            self.held[self.graph] ^= 1 << i;
            self.dirty = true;
        }
    }

    /// Draw a new sample (and the others again), from `seed`.
    fn new_sample(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = Rng::new(seed);
        self.sample = Sample::draw(&mut self.rng, UNITS);
        self.others.clear();
        self.shown.others.clear();
        self.dirty = true;
    }

    /// Draw the other samples, as many as there is time for in this frame (at least one).
    fn draw_others(&mut self, budget: &crate::clock::Budget) {
        while self.others.len() < OTHERS {
            let s = Sample::draw(&mut self.rng, UNITS);
            self.others.push(s.cov);
            self.dirty = true;
            if budget.spent() {
                break;
            }
        }
    }

    /// Draw every other sample at once (for the tests).
    #[cfg(test)]
    fn settle(&mut self) {
        while self.others.len() < OTHERS {
            let s = Sample::draw(&mut self.rng, UNITS);
            self.others.push(s.cov);
        }
        self.refresh();
        for _ in 0..400 {
            self.glide(1.0 / 60.0);
        }
    }

    /// Work out what the settings give.
    fn refresh(&mut self) {
        let (g, strengths, held) = (self.g(), &self.strengths[self.graph], self.held[self.graph]);
        self.now = Reading::of(g, strengths, held, &self.sample, &self.others);
        let model = Model::new(g, strengths);
        // (the picture keeps its scale whatever is held: what holding takes away shows;
        // the units are kept in shares of how far it reaches either way)
        let c = model.covariance(&self.sample.cov);
        let sd = |i: usize| c.get(i * model.p + i).map_or(1.0, |v| v.max(1e-12).sqrt());
        self.reach = (3.0 * sd(0), 3.0 * sd(1));
        let (rx, ry) = self.reach;
        self.cloud = left_over(&model, &self.sample, held, CLOUD).into_iter().map(|(x, y)| (x / rx, y / ry)).collect();
        self.dirty = false;
        let (now, s) = (&self.now, &mut self.shown);
        let estimate = now.estimate.unwrap_or(s.estimate);
        if self.snap || s.sets.len() != now.sets.len() || s.open.len() != now.states.len() || s.cloud.len() != self.cloud.len() {
            s.estimate = estimate;
            s.others.clear();
            s.sets = now.sets.iter().map(|x| x.1.unwrap_or(0.0)).collect();
            s.range = now.range;
            s.truth = now.truth;
            s.open = now.states.iter().map(|x| if x.open() { 1.0 } else { 0.0 }).collect();
            s.held = std::array::from_fn(|i| if has(held, i) { 1.0 } else { 0.0 });
            s.cloud = self.cloud.clone();
            s.reach = self.reach;
            self.snap = false;
        }
        // samples just drawn are drawn where they belong at once
        let have = s.others.len();
        s.others.extend(now.others.iter().skip(have).map(|o| o.unwrap_or(estimate)));
        s.others.truncate(now.others.len());
    }

    /// Bring what is drawn `dt` seconds nearer to what the settings give. True while
    /// anything is still on its way.
    fn glide(&mut self, dt: f32) -> bool {
        let k = 1.0 - (-dt.max(0.0) / GLIDE).exp() as f64;
        let (now, s, held) = (&self.now, &mut self.shown, self.held[self.graph]);
        // Values that move together arrive together: each group comes nearer its
        // targets by the same share, and is put exactly there (not nearly: the picture
        // then stops changing) once all of it is within `close`. Snapped one by one,
        // the last few would trickle in for a second after the rest had arrived.
        let mut moving = false;
        let mut group = |pairs: Vec<(&mut f64, Option<f64>)>, close: f64| {
            let mut far = 0.0f64;
            let mut pairs: Vec<(&mut f64, f64)> = pairs.into_iter().filter_map(|(x, to)| to.filter(|v| v.is_finite()).map(|to| (x, to))).collect();
            for (x, to) in pairs.iter_mut() {
                **x += (*to - **x) * k;
                far = far.max(if x.is_finite() { (*to - **x).abs() } else { f64::INFINITY });
            }
            if far < close {
                for (x, to) in pairs {
                    *x = to;
                }
            } else {
                moving = true;
            }
        };
        let close = (now.range.1 - now.range.0).abs().max(1e-6) * 1e-4;
        group(vec![(&mut s.estimate, now.estimate), (&mut s.truth, Some(now.truth)), (&mut s.range.0, Some(now.range.0)), (&mut s.range.1, Some(now.range.1))], close);
        group(s.others.iter_mut().zip(&now.others).map(|(x, to)| (x, *to)).collect(), close);
        group(s.sets.iter_mut().zip(&now.sets).map(|(x, to)| (x, to.1)).collect(), close);
        let mut cloud: Vec<(&mut f64, Option<f64>)> = Vec::with_capacity(2 * s.cloud.len());
        for (x, to) in s.cloud.iter_mut().zip(&self.cloud) {
            cloud.push((&mut x.0, Some(to.0)));
            cloud.push((&mut x.1, Some(to.1)));
        }
        group(cloud, 3e-4);
        let (rx, ry) = self.reach;
        group(vec![(&mut s.reach.0, Some(rx)), (&mut s.reach.1, Some(ry))], rx.min(ry).max(1e-12) * 3e-4);
        let mut ease32 = |x: &mut f32, to: f32| {
            *x += (to - *x) * k as f32;
            if (to - *x).abs() < 0.004 || !x.is_finite() {
                *x = to;
            } else {
                moving = true;
            }
        };
        for (x, state) in s.open.iter_mut().zip(&now.states) {
            ease32(x, if state.open() { 1.0 } else { 0.0 });
        }
        for (i, x) in s.held.iter_mut().enumerate() {
            ease32(x, if has(held, i) { 1.0 } else { 0.0 });
        }
        moving
    }

    /// Set a strength from a slider. Near 0 it is 0, the arrow gone: a pixel of the
    /// slider is worth more than a hundredth, so a hand could hardly find 0 otherwise.
    fn set_strength(&mut self, k: usize, v: f32) {
        if let Some(s) = self.strengths[self.graph].get_mut(k) {
            let v = v as f64;
            *s = if v.abs() < SNAP { 0.0 } else { v };
            self.dirty = true;
        }
    }

    /// How the rows of the list of paths are written on a canvas of size `view`: on one
    /// line in full where they fit, else (on a narrow canvas) on one line in short, else
    /// on two lines. Decided by every reason the graph's paths could give, whatever is
    /// held, so that the picture keeps its layout as variables are held and let go
    /// (and kept until the graph or the canvas changes).
    fn rows(&mut self, f: &Frame, view: Vec2) -> Rows {
        let key = (self.graph, view.x.round() as i32, is_wide(view));
        if let Some((k, rows)) = self.rows_kept {
            if k == key {
                return rows;
            }
        }
        let g = self.g();
        let (text, small, left, right) = list_metrics(view);
        let width = right - left - 24.0;
        let all = paths(g);
        let formula = all.iter().map(|p| f.text_size(&format!("${}$", p.markup(g)), text).x).fold(0.0, f32::max);
        let why = |short: bool| all.iter().flat_map(|p| possible_states(g, p)).map(|(p, st)| f.text_size(&reason(g, p, st, short).0, small).x).fold(0.0, f32::max);
        let (one_line, short) = if formula + 26.0 + why(false) <= width {
            (true, false)
        } else if !is_wide(view) && formula + 20.0 + why(true) <= width {
            (true, true)
        } else {
            (false, !is_wide(view))
        };
        let rows = Rows { one_line, short, formula, height: if one_line { text + 10.0 } else { text + small + 13.0 } };
        self.rows_kept = Some((key, rows));
        rows
    }
}

/// Every state a path could be in, whatever is held and whichever arrows are there.
fn possible_states<'a>(g: &GraphDef, path: &'a Path) -> Vec<(&'a Path, State)> {
    let mut out = vec![(path, State::Open)];
    out.extend(path.arrows.iter().map(|&k| (path, State::Absent(k))));
    for k in 1..path.nodes.len().saturating_sub(1) {
        let v = path.nodes[k];
        if path.collider(k) {
            out.push((path, State::Closed(v)));
            out.extend((0..g.vars.len()).filter(|&d| d != v && g.vars[d].role == Role::Measured).map(|d| (path, State::Opened { collider: v, held: d })));
            out.push((path, State::Opened { collider: v, held: v }));
        } else {
            out.push((path, State::Blocked(v)));
        }
    }
    out
}

impl Scene for Adjust {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 3.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    /// Still, unless something glides or the other samples are still being drawn.
    fn animated(&self) -> bool {
        self.moving || self.dirty || self.others.len() < OTHERS
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("click a variable: hold it, or let it go    point at a path: see it alone on the graph    click a dot of \"every set\": hold that set")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "How much does $X$ change $Y$? With data that were only observed, not from an experiment, units with more $X$ may differ in other ways too. The usual remedy is to compare units that are alike in some other variables: to control, or adjust, for them. Here that means putting them in the least-squares fit of $Y$ beside $X$ and reading off the coefficient of $X$. Which variables matters, and more is not better.");
        ui::note(ui, "Click a variable in the graph to hold it, or to let it go. Every variable is its causes, each times the strength of its arrow, plus a noise of its own; a sample is 2000 units.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Graph");
        let mut graph = self.graph;
        let names: Vec<&str> = GRAPHS.iter().map(|g| g.name).collect();
        if ui::choice(ui, &names, &mut graph) {
            self.choose_graph(graph);
        }
        ui::note(ui, GRAPH_NOTES[self.graph]);
        ui::section(ui, "Strengths of the arrows");
        let g = self.g();
        for (k, a) in g.arrows.iter().enumerate() {
            let mut v = self.strengths[self.graph].get(k).copied().unwrap_or(0.0) as f32;
            if ui::slider(ui, &format!(r"${} \to {}$", g.vars[a.from].letter, g.vars[a.to].letter), &mut v, -2.0..=2.0) {
                self.set_strength(k, v);
            }
        }
        ui::note(ui, "How much the arrow's tail adds to its head, per unit. Near the middle a slider snaps to 0: the arrow is gone, and so is every path through it.");
        if ui.button("draw a new sample").clicked() {
            self.new_sample(self.seed + 1);
        }
        ui::note(ui, "Another 2000 units, and 40 other samples, from the same arrows: what changes is the luck of the draw. (The scene opens on a sample whose estimates are typical.)");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (CAUSE, "$X$: the cause whose effect on $Y$ is measured"),
                (EFFECT, "$Y$: the effect"),
                (THIRD, "the other variables; one in a box is held (it is in the fit beside $X$)"),
                (pal::FAINT, "a variable nobody measured: no fit can hold it"),
                (TRUTH, "green: a path that carries the effect into the fit; the true effect; a set that is right"),
                (NAIVE, "pink: a path that carries bias into the fit; the error; a set that is wrong"),
                (BLOCKED, "grey: a path that carries nothing, being blocked"),
                (pal::WHITE, "white: the estimate from this sample, and faint, from 40 other samples; in what the fit sees, the units and the line whose slope is the estimate"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Unit:", "one of the things measured (a person, say).");
        ui::term(ui, "Holding a variable (controlling for it, adjusting for it):", "comparing units that are alike in it. Here: putting it in the fit of $Y$ beside $X$. The coefficient of $X$ is then how $Y$ goes with $X$ among units alike in what is held.");
        ui::term(ui, "Path:", "a way from $X$ to $Y$ along arrows, whichever way they point. It carries association between them unless it is blocked.");
        ui::term(ui, "Collider:", r"a variable where two arrows of a path meet, as at $C$ in $X \to C \gets Y$. Holding it, or anything it causes, opens the path.");
        ui::term(ui, "Blocked:", "a path is blocked at a held variable where its arrows do not meet, and at a collider when neither it nor anything it causes is held.");
        ui::term(ui, "Back door:", "a path that starts with an arrow into $X$.");
        ui::term(ui, "Confounder:", "a common cause of $X$ and $Y$; it opens a back door.");
        ui::term(ui, "Mediator:", "a variable on the way from $X$ to $Y$; part of the effect goes through it.");
        ui::term(ui, "Total effect:", "how much $Y$ changes when $X$ is set one higher and everything it causes follows. With straight-line effects: the sum, over the paths along the arrows from $X$ to $Y$, of the product of each path's strengths.");
        ui::term(ui, "The rule (the back-door criterion):", r"a set gives the total effect if it blocks every back door and holds nothing $X$ causes. It is enough whatever the strengths, and not always needed: set $Y \to C$ to 0, and holding $C$ leaves the estimate right.");
        ui::term(ui, "An effect of $Y$ held:", r"holding something $Y$ causes picks out units by their $Y$, which bends how $Y$ goes with $X$ even where it opens no path (set $X \to C$ to 0, and hold $C$).");
        ui::term(ui, "Precision:", "how little the estimate wanders from sample to sample (the spread of the faint dots). Once the back doors are blocked, holding a cause of $Y$ only improves it, and holding a cause of $X$ only (an instrument) worsens it; neither moves the estimate on average.");

        ui::section(ui, "Numbers");
        let now = &self.now;
        ui::readout_colored(ui, "true effect", num(now.truth, 3), TRUTH);
        match now.estimate {
            Some(e) => {
                ui::readout_colored(ui, "estimate from this sample", num(e, 3), pal::WHITE);
                ui::readout_colored(ui, "error", diff(e - now.truth), if now.errors.none() { TRUTH } else { NAIVE });
            }
            None => ui::readout(ui, "estimate from this sample", "none"),
        }
        if let Some(e) = now.endless {
            ui::readout(ui, "with endless data", num(e, 3));
        }
        if let Some(sd) = now.spread() {
            ui::readout(ui, &format!("spread over {} other samples", now.others.len()), num(sd, 3));
        }
        ui::note(ui, "With endless data the estimate is exactly the true effect when the set is right, and still wrong when it is not: more data shrinks the spread, not the error.");
        ui::see_also(
            ui,
            &[
                ("confound", "correlation is not causation: one confounder, slice by slice"),
                ("junctions", "chains, forks and colliders: what holding does to three variables"),
                ("dosee", "seeing and doing: what an experiment does instead"),
                ("regress", "least-squares fitting"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        // the pointer belongs to the picture: there is nothing to pan or zoom
        f.handles.capture();
        f.handles.claim_clicks();
        if self.others.len() < OTHERS {
            let budget = crate::clock::frame_budget(f.real_dt, 0.003);
            self.draw_others(&budget);
        }
        if self.dirty {
            self.refresh();
        }
        self.moving = self.glide(f.real_dt);
        // (the canvas is laid out again only when it changes by a pixel or more either
        // way: the application's panels can leave it a pixel wider for a single frame,
        // which would nudge the whole picture and back)
        let seen = f.view_size();
        if (seen - self.canvas).abs().max_element() >= 1.5 {
            self.canvas = seen;
        }
        let view = self.canvas;
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        if view.x < 150.0 || view.y < 120.0 {
            return;
        }
        let rows = self.rows(f, view);
        let lay = Lay::of(view, f.top_inset, self.now.paths.len(), rows);
        let pointer = f.ptr.pos.map(|p| p / f.px);
        self.draw_tabs(f, &lay, pointer);
        // (the list first says which path the pointer is on, so that the graph can show it)
        let rects = self.row_rects(&lay);
        self.hover = pointer.and_then(|p| rects.iter().position(|r| inside(p, *r)));
        if let (Some(k), true) = (self.hover, f.ptr.clicked) {
            self.picked = if self.picked == Some(k) { None } else { Some(k) };
        }
        self.draw_graph(f, &lay, pointer);
        self.draw_list(f, &lay, &rects);
        self.draw_scatter(f, &lay);
        self.draw_gauge(f, &lay, pointer);
    }
}

impl Adjust {
    fn draw_tabs(&mut self, f: &mut Frame, lay: &Lay, pointer: Option<Vec2>) {
        let Some((lo, hi)) = lay.tabs else { return };
        let size = lay.small;
        let widths: Vec<f32> = GRAPHS.iter().map(|g| f.text_size(g.name, size).x + 22.0).collect();
        let total: f32 = widths.iter().sum::<f32>() + 8.0 * (GRAPHS.len() - 1) as f32;
        // (on a narrow canvas the three share the width)
        let stretch = if lay.narrow { (hi.x - lo.x) / total.max(1.0) } else { 1.0 };
        let mut x = lo.x;
        for (k, g) in GRAPHS.iter().enumerate() {
            let w = widths[k] * stretch;
            if hud_button(f, (Vec2::new(x, lo.y), Vec2::new(x + w, hi.y)), g.name, size, k == self.graph, pointer) {
                self.choose_graph(k);
            }
            x += w + 8.0 * stretch;
        }
    }

    fn draw_graph(&mut self, f: &mut Frame, lay: &Lay, pointer: Option<Vec2>) {
        let g = self.g();
        let strengths = self.strengths[self.graph].clone();
        let small = lay.small - 0.5;
        // (in a window too small for them, the variables are smaller and nothing is
        // written beside them or on the arrows)
        let size = lay.graph.1 - lay.graph.0;
        let roomy = size.x >= 260.0 && size.y >= 230.0;
        let r = if roomy { lay.radius } else { 11.0 };
        // room round the variables for what is written beside them
        let said = |v: &VarDef| roomy && !v.kind.first().is_empty();
        let widest = g.vars.iter().filter(|v| said(v) && (v.at.0 <= 0.0 || v.at.0 >= 1.0)).map(|v| f.text_size(v.kind.first(), small).x * 0.5).fold(r + 4.0, f32::max);
        let above = r + if g.vars.iter().any(|v| said(v) && !v.below) { small + 10.0 } else { 6.0 };
        let below = r + if g.vars.iter().any(|v| said(v) && v.below && v.at.1 >= 1.0) { small + 10.0 } else { 6.0 };
        let at = places(g, lay.graph, widest + 2.0, above, below);
        // the path shown alone, if any
        let focus = self.hover.or(self.picked).filter(|&k| k < self.now.paths.len());

        // each arrow in the colour of what it carries into the fit: that of the most
        // open path along it
        let mut links = Vec::with_capacity(g.arrows.len());
        for (k, a) in g.arrows.iter().enumerate() {
            let (mut color, mut best, mut on_focus) = (BLOCKED, -1.0f32, false);
            for (p, path) in self.now.paths.iter().enumerate().filter(|(_, path)| path.arrows.contains(&k)) {
                let open = self.shown.open.get(p).copied().unwrap_or(0.0);
                on_focus |= focus == Some(p);
                if open > best {
                    (best, color) = (open, path_color(path.causal(), open));
                }
            }
            let mut width = 2.2;
            if !strengths.get(k).is_some_and(|s| present(*s)) {
                (color, width) = (pal::FAINT.a(0.6), 1.2);
            }
            if focus.is_some() {
                if on_focus {
                    width += 1.6;
                } else {
                    color = color.a(0.25);
                }
            }
            links.push((a.from, a.to, Link::Arrow, color, width));
        }
        let nodes: Vec<Node> = g
            .vars
            .iter()
            .zip(&at)
            .map(|(v, p)| Node {
                at: *p,
                name: v.name,
                color: match v.role {
                    Role::Cause => CAUSE,
                    Role::Effect => EFFECT,
                    Role::Measured => THIRD,
                    Role::Hidden => pal::FAINT,
                },
            })
            .collect();
        // held variables: filled, and boxed
        for (i, p) in at.iter().enumerate() {
            let h = self.shown.held.get(i).copied().unwrap_or(0.0);
            if h > 0.0 {
                hud_disc(f, *p, r - 1.0, THIRD.a(0.45 * h));
                let half = Vec2::splat(r + 5.0);
                hud_outline(f, *p - half, *p + half, 1.6, pal::WHITE.a(0.9 * h));
            }
        }
        // the pointer on a variable
        self.hover_var = pointer.and_then(|q| at.iter().position(|p| p.distance(q) <= r + 7.0));
        if let Some(i) = self.hover_var {
            if g.vars.get(i).is_some_and(|v| v.role == Role::Measured) {
                hud_circle(f, at[i], r + 3.0, 1.2, pal::WHITE.a(0.6));
            }
            if f.ptr.clicked {
                self.toggle(i);
            }
        }
        hud_graph(f, &nodes, &links, r);
        // the strengths, beside the middle of each arrow, on the side asked for unless
        // the other side is clearer of the other arrows and the variables
        for (k, a) in g.arrows.iter().enumerate().filter(|_| roomy) {
            let (Some(p), Some(q)) = (at.get(a.from), at.get(a.to)) else { continue };
            let d = (*q - *p).normalize_or_zero();
            let s = strengths.get(k).copied().unwrap_or(0.0);
            let text = num(s, 2);
            let half = f.text_size(&text, small) * 0.5 + Vec2::splat(1.0);
            let place = |along: f32, side: f32| {
                let left = Vec2::new(d.y, -d.x) * side;
                p.lerp(*q, along) + left * (6.0 + half.x * left.x.abs() + half.y * left.y.abs())
            };
            let clear = |c: Vec2| {
                let lines = g.arrows.iter().enumerate().filter(|(j, _)| *j != k).map(|(_, b)| seg_box_dist(at[b.from], at[b.to], c - half, c + half));
                let discs = at.iter().map(|n| seg_box_dist(*n, *n, c - half, c + half) - r);
                lines.chain(discs).fold(f32::MAX, f32::min)
            };
            // the middle on the side asked for, unless it is crowded: then the clearest
            // of the other side and a little either way along the arrow
            let tries = [(0.5, a.side), (0.5, -a.side), (0.38, a.side), (0.62, a.side), (0.38, -a.side), (0.62, -a.side)].map(|(t, side)| place(t, side));
            let centre = tries.iter().copied().find(|c| clear(*c) >= 5.0).unwrap_or_else(|| tries.iter().copied().max_by(|x, y| clear(*x).total_cmp(&clear(*y))).unwrap_or(tries[0]));
            let faded = focus.is_some_and(|p| !self.now.paths[p].arrows.contains(&k));
            let color = if !present(s) {
                pal::FAINT
            } else if faded {
                pal::DIM.a(0.4)
            } else {
                pal::DIM
            };
            f.text_px(centre, text, small, color, Align2::CENTER_CENTER);
        }
        // what each variable is
        for (i, (v, p)) in g.vars.iter().zip(&at).enumerate().filter(|_| roomy) {
            let kind = match v.kind {
                Kind::Said(t) => t,
                Kind::ByArrows(_) => kind_of(g, &strengths, i),
            };
            if kind.is_empty() {
                continue;
            }
            let (y, align) = if v.below { (p.y + r + 5.0, Align2::CENTER_TOP) } else { (p.y - r - 5.0, Align2::CENTER_BOTTOM) };
            f.text_px(Vec2::new(p.x, y), kind, small, pal::DIM, align);
        }
        // what a click does here
        if let Some(at) = lay.status {
            let text = match self.hover_var.and_then(|i| g.vars.get(i).map(|v| (i, v))) {
                Some((i, v)) => match v.role {
                    Role::Measured if has(self.held(), i) => format!("{} is held: click to let it go", v.name),
                    Role::Measured => format!("click to hold {}: it goes into the fit beside $X$", v.name),
                    Role::Hidden => format!("{} was not measured: no fit can hold it", v.name),
                    Role::Cause => "$X$, the cause: its effect on $Y$ is measured".to_string(),
                    Role::Effect => "$Y$, the effect".to_string(),
                },
                None => "click a variable to hold it (it is boxed) or to let it go".to_string(),
            };
            f.text_px(at, text, lay.small, pal::DIM, Align2::LEFT_CENTER);
        }
    }

    /// The rectangles of the rows of the list of paths (none if there is no list).
    fn row_rects(&self, lay: &Lay) -> Vec<Rect> {
        let Some((lo, hi)) = lay.list else { return Vec::new() };
        let top = lo.y + lay.text + lay.small + 24.0;
        let h = lay.rows.height;
        (0..self.now.paths.len()).map(|k| (Vec2::new(lo.x - 6.0, top + k as f32 * h - 4.0), Vec2::new(hi.x, top + (k + 1) as f32 * h - 4.0))).collect()
    }

    fn draw_list(&self, f: &mut Frame, lay: &Lay, rects: &[Rect]) {
        let Some((lo, hi)) = lay.list else { return };
        let g = self.g();
        let (text, small, rows) = (lay.text, lay.small, lay.rows);
        f.text_px(lo, "every path between $X$ and $Y$", text, pal::TEXT, Align2::LEFT_TOP);
        // what the colours say
        let mut x = lo.x;
        let y = lo.y + text + 12.0;
        for (color, words) in [(TRUTH, "carries the effect"), (NAIVE, "carries bias"), (BLOCKED, "carries nothing")] {
            hud_rect(f, Vec2::new(x, y - 1.5), Vec2::new(x + 16.0, y + 1.5), color);
            f.text_px(Vec2::new(x + 22.0, y), words, small, pal::DIM, Align2::LEFT_CENTER);
            x += 22.0 + f.text_size(words, small).x + 16.0;
        }
        for (k, ((path, state), (top, bottom))) in self.now.paths.iter().zip(&self.now.states).zip(rects).enumerate() {
            if self.hover == Some(k) || self.picked == Some(k) {
                hud_rect_at(f, 0, *top, *bottom, Color::hex(0x1B2334));
            }
            let absent = matches!(state, State::Absent(_));
            let color = if absent { pal::FAINT } else { path_color(path.causal(), self.shown.open.get(k).copied().unwrap_or(0.0)) };
            let y = top.y + 4.0;
            hud_rect(f, Vec2::new(lo.x, y + 1.0), Vec2::new(lo.x + 4.0, bottom.y - 5.0), color);
            f.text_px(Vec2::new(lo.x + 12.0, y), format!("${}$", path.markup(g)), text, if absent { pal::FAINT } else { pal::TEXT }, Align2::LEFT_TOP);
            let (why, fine) = reason(g, path, *state, rows.short);
            let tone = if absent {
                pal::FAINT
            } else if !fine {
                NAIVE
            } else if path.causal() {
                TRUTH
            } else {
                pal::DIM
            };
            let gap = if rows.short { 20.0 } else { 26.0 };
            let at = if rows.one_line { Vec2::new(lo.x + 12.0 + rows.formula + gap, y + (text - small) * 0.5 + 1.0) } else { Vec2::new(lo.x + 12.0, y + text + 5.0) };
            f.text_px(at, why, small, tone, Align2::LEFT_TOP);
        }
        // the rule
        let y = rects.last().map_or(lo.y + 60.0, |r| r.1.y) + 6.0;
        let (doors, caused) = self.now.rule;
        let yes = |b: bool| if b { "yes" } else { "no" };
        let parts = [(format!("every back door blocked: {}", yes(doors)), doors), (format!("nothing $X$ causes held: {}", yes(caused)), caused)];
        let mut x = lo.x;
        f.text_px(Vec2::new(x, y), "the rule:", small, pal::DIM, Align2::LEFT_TOP);
        x += f.text_size("the rule:", small).x + 10.0;
        for (words, ok) in parts {
            let w = f.text_size(&words, small).x;
            if x + w > hi.x + 4.0 {
                break;
            }
            f.text_px(Vec2::new(x, y), words.as_str(), small, if ok { TRUTH } else { NAIVE }, Align2::LEFT_TOP);
            x += w + 14.0;
        }
    }

    /// What the fit sees: each unit's Y against its X, each less its fit on the held
    /// variables, with the line whose slope is the estimate and the line whose slope is
    /// the true effect.
    fn draw_scatter(&self, f: &mut Frame, lay: &Lay) {
        let Some((lo, hi)) = lay.scatter else { return };
        let g = self.g();
        let (text, small) = (lay.text, lay.small);
        let s = &self.shown;
        let held = self.held();
        f.text_px(lo, "what the fit sees", text, pal::TEXT, Align2::LEFT_TOP);
        let what = if held == 0 {
            "each dot a unit: $Y$ against $X$ (less their averages)".to_string()
        } else {
            format!("each dot a unit: $Y$ against $X$, each less its fit on {}", names(g, held, ""))
        };
        f.text_px(lo + Vec2::new(0.0, text + 6.0), what, small, pal::DIM, Align2::LEFT_TOP);
        // the plot, as wide as the room less its lines' names, and no wider than it is
        // tall by half again
        let names = [(format!("slope ${}$: the estimate", num(s.estimate, 2)), pal::WHITE), (format!("slope ${}$: the true effect", num(s.truth, 2)), TRUTH)];
        // (measured on a pattern, not the numbers: the plot must not change size as they glide)
        let named = f.text_size("slope $−0.00$: the true effect", small).x + 14.0;
        // (room above the plot and below it for the names of its axes, outside it,
        // where no line can run through them)
        let top = lo.y + text + small + 22.0 + small + 6.0;
        let h = (hi.y - top - small - 8.0).max(10.0);
        let w = (hi.x - lo.x - named - 8.0).min(h * 1.5).max(10.0);
        let mid = Vec2::new(lo.x + 4.0 + w * 0.5, top + h * 0.5);
        let half = Vec2::new(w * 0.5, h * 0.5);
        // (the units are in shares of the reach either way; a slope is turned into those shares)
        let ratio = s.reach.0 / s.reach.1.max(1e-12);
        let to = |x: f64, y: f64| mid + Vec2::new(x as f32 * half.x, -y as f32 * half.y);
        hud_outline(f, mid - half, mid + half, 1.0, pal::FAINT.a(0.6));
        hud_line(f, Vec2::new(mid.x - half.x, mid.y), Vec2::new(mid.x + half.x, mid.y), 1.0, pal::FAINT);
        hud_line(f, Vec2::new(mid.x, mid.y - half.y), Vec2::new(mid.x, mid.y + half.y), 1.0, pal::FAINT);
        f.text_px(Vec2::new(mid.x + half.x, mid.y + half.y + 4.0), "$X$ left over (across)", small, pal::DIM, Align2::RIGHT_TOP);
        f.text_px(Vec2::new(mid.x, mid.y - half.y - 4.0), "$Y$ left over (up)", small, pal::DIM, Align2::CENTER_BOTTOM);
        for (x, y) in &s.cloud {
            if x.abs() <= 1.0 && y.abs() <= 1.0 {
                f.dl.discs.push(f.hud_at(1, to(*x, *y)), 3.6, pal::WHITE.a(0.5));
            }
        }
        // the two lines through the middle: as far as the plot reaches
        let line = |slope: f64| {
            let m = if (slope * ratio).is_finite() { slope * ratio } else { 0.0 };
            let x = if m.abs() > 1.0 { 1.0 / m.abs() } else { 1.0 };
            (to(-x, -m * x), to(x, m * x))
        };
        let estimate = s.estimate;
        let (t0, t1) = line(s.truth);
        let (e0, e1) = line(estimate);
        hud_line(f, t0, t1, 1.8, TRUTH.a(0.85));
        hud_line(f, e0, e1, 2.2, pal::WHITE);
        // their names, at their right-hand ends, the higher one above the other
        let ends = [e1, t1];
        let gap = (e1.y - t1.y).abs();
        for (k, (words, color)) in names.into_iter().enumerate() {
            let (end, other) = (ends[k], ends[1 - k]);
            let higher = end.y < other.y || (end.y == other.y && k == 0);
            let nudge = if gap < small + 4.0 { (small + 4.0 - gap) * 0.5 * if higher { -1.0 } else { 1.0 } } else { 0.0 };
            f.text_px(Vec2::new(end.x + 8.0, end.y + nudge), words, small, color, Align2::LEFT_CENTER);
        }
    }

    fn draw_gauge(&mut self, f: &mut Frame, lay: &Lay, pointer: Option<Vec2>) {
        self.hover_set = None;
        let Some((lo, hi)) = lay.gauge else { return };
        let g = self.g();
        let (text, small, narrow) = (lay.text, lay.small, lay.narrow);
        let width = hi.x - lo.x;
        let line = text + 8.0;
        let now = &self.now;
        let s = &self.shown;
        let held = self.held();

        // the truth, as a sum over the causal paths
        let sum = truth_sum(&now.terms, now.truth);
        let main = format!("true effect of $X$ on $Y$: ${sum}$");
        let w = f.text_size(&main, text).x;
        f.text_px(lo, main.as_str(), text, TRUTH, Align2::LEFT_TOP);
        let causal: Vec<String> = now.paths.iter().zip(&now.states).filter(|(p, s)| p.causal() && !matches!(s, State::Absent(_))).map(|(p, _)| format!("${}$", p.markup(g))).collect();
        let along = match causal.as_slice() {
            [] => "no path along the arrows leads from $X$ to $Y$".to_string(),
            [one] => format!("the one path along the arrows, {one}"),
            many => format!("one term for each path along the arrows ({}): the product of its strengths", many.join(", ")),
        };
        for note in [along.as_str(), "one term for each path along the arrows"] {
            if w + 16.0 + f.text_size(note, small).x <= width {
                f.text_px(Vec2::new(lo.x + w + 16.0, lo.y + (text - small) * 0.5 + 1.0), note, small, pal::DIM, Align2::LEFT_TOP);
                break;
            }
        }
        // the estimate
        let on = if held == 0 { "on $X$ alone".to_string() } else { format!("on $X$ and {}", names(g, held, "")) };
        let estimate = match now.estimate {
            Some(e) => [
                format!("estimate: ${}$, the coefficient of $X$ when $Y$ is fitted {on}", num(e, 2)),
                format!("estimate ${}$: fit $Y$ {on}", num(e, 2)),
                format!("estimate ${}$", num(e, 2)),
            ]
            .into_iter()
            .find(|t| f.text_size(t, text).x <= width)
            .unwrap_or_default(),
            None => format!("no estimate: $X$ is fixed by {}", names(g, held, "")),
        };
        f.text_px(lo + Vec2::new(0.0, line), estimate, text, pal::WHITE, Align2::LEFT_TOP);
        // the verdict
        let (verdict, tone) = match now.estimate {
            Some(e) if now.errors.none() => {
                let long = format!("right: no error but the luck of this sample (${}$)", diff(e - now.truth));
                (if f.text_size(&long, text).x <= width { long } else { format!("right, but for luck (${}$)", diff(e - now.truth)) }, TRUTH)
            }
            Some(e) => (format!("wrong by ${}$: {}", diff(e - now.truth), now.errors_named()), NAIVE),
            None => (format!("nothing to compare: {}", now.errors_named()), NAIVE),
        };
        f.text_px(lo + Vec2::new(0.0, 2.0 * line), verdict, text, tone, Align2::LEFT_TOP);

        // the axis, and on it this sample, the others, and every set
        let labels = if narrow { 0.0 } else { 150.0 };
        let axis = Axis { range: s.range, x0: lo.x + labels + 8.0, x1: hi.x - 12.0 };
        let axis_y = lo.y + 3.0 * line + 20.0;
        let (row_e, row_o, row_s) = if narrow { (axis_y + 48.0, axis_y + 48.0, axis_y + 88.0) } else { (axis_y + 36.0, axis_y + 62.0, axis_y + 90.0) };
        hud_line(f, Vec2::new(axis.x0, axis_y), Vec2::new(axis.x1, axis_y), 1.2, pal::DIM);
        let span = (s.range.1 - s.range.0).max(1e-6);
        let step = nice_step((span * 80.0 / (axis.x1 - axis.x0).max(40.0) as f64) as f32) as f64;
        // (a step of a tenth may come out a hair under it)
        let decimals = if step >= 0.999 {
            0
        } else if step >= 0.0999 {
            1
        } else {
            2
        };
        let first = (s.range.0 / step).ceil() as i64;
        let tx = axis.x(s.truth);
        for i in first..=first + 60 {
            let v = i as f64 * step;
            if v > s.range.1 {
                break;
            }
            let x = axis.x(v);
            hud_line(f, Vec2::new(x, axis_y), Vec2::new(x, axis_y + 5.0), 1.0, pal::DIM);
            // (not a number the truth's line would run through)
            let label = num(v, decimals);
            if (x - tx).abs() > f.text_size(&label, small - 1.0).x * 0.5 + 3.0 {
                f.text_px(Vec2::new(x, axis_y + 7.0), label, small - 1.0, pal::DIM, Align2::CENTER_TOP);
            }
        }
        // the truth: a line down through the rows
        hud_line(f, Vec2::new(tx, axis_y - 3.0), Vec2::new(tx, row_s + 20.0), 2.0, TRUTH);
        let tw = f.text_size("true effect", small).x;
        f.text_px(Vec2::new(tx.max(lo.x + tw * 0.5).min(hi.x - tw * 0.5), axis_y - 5.0), "true effect", small, TRUTH, Align2::CENTER_BOTTOM);
        // the rows' names
        let count = self.others.len();
        let sample_name = if narrow { format!("this sample (white), {count} other samples (faint)") } else { "this sample".to_string() };
        let others_name = format!("{count} other samples");
        let mut rows = vec![(row_e, sample_name.as_str()), (row_s, "every set")];
        if !narrow {
            rows.push((row_o, others_name.as_str()));
        }
        for (y, name) in rows {
            if narrow {
                f.text_px(Vec2::new(lo.x, y - 18.0), name, small, pal::DIM, Align2::LEFT_CENTER);
            } else {
                f.text_px(Vec2::new(lo.x + labels - 6.0, y), name, small, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
        // what the colours of the sets say
        let key_at = if narrow { Vec2::new(lo.x + f.text_size("every set", small).x + 16.0, row_s - 18.0) } else { Vec2::new(lo.x + labels - 6.0, row_s + 16.0) };
        let parts = [("right", TRUTH), ("wrong", NAIVE)];
        let mut x = if narrow { key_at.x } else { key_at.x - parts.iter().map(|(w, _)| f.text_size(w, small - 1.0).x + 26.0).sum::<f32>() + 14.0 };
        for (words, color) in parts {
            f.dl.overlay_discs.push(f.hud(Vec2::new(x + 4.0, key_at.y)), 7.0, color);
            f.text_px(Vec2::new(x + 12.0, key_at.y), words, small - 1.0, pal::DIM, Align2::LEFT_CENTER);
            x += f.text_size(words, small - 1.0).x + 26.0;
        }

        // the other samples
        for (k, v) in s.others.iter().enumerate() {
            let jitter = ((k * 7) % 9) as f32 - 4.0;
            f.dl.discs.push(f.hud_at(1, Vec2::new(axis.x(*v), row_o + jitter * 1.7)), 5.0, pal::WHITE.a(if narrow { 0.35 } else { 0.55 }));
        }
        // this sample's estimate, and the error from the truth to it
        let ex = axis.x(s.estimate);
        let bad = !now.errors.none();
        hud_line(f, Vec2::new(tx, row_e), Vec2::new(ex, row_e), 4.0, if bad { NAIVE } else { TRUTH.a(0.5) });
        f.dl.overlay_discs.push(f.hud(Vec2::new(ex, row_e)), 11.0, pal::WHITE);
        if let Some(e) = now.estimate {
            let label = num(e, 2);
            let w = f.text_size(&label, text).x;
            let right = s.estimate >= s.truth;
            let roomy = if right { ex + w + 14.0 < hi.x } else { ex - w - 14.0 > axis.x0 };
            let side = if right == roomy { 1.0 } else { -1.0 };
            let lift = if narrow { -13.0 } else { 0.0 };
            f.text_placed("estimate", 0, Vec2::new(ex, row_e), Vec2::new(side * (w * 0.5 + 11.0), lift), label, text, pal::WHITE);
        }
        // every set, at its estimate from this sample: green if right, pink if wrong
        let targets: Vec<f32> = now.sets.iter().map(|x| axis.x(x.1.unwrap_or(0.0))).collect();
        let level = levels(&targets, 8.0);
        let mut marks = Vec::with_capacity(s.sets.len());
        for (k, (v, (set, est, right))) in s.sets.iter().zip(&now.sets).enumerate() {
            if est.is_none() {
                continue;
            }
            let p = Vec2::new(axis.x(*v), row_s + level.get(k).copied().unwrap_or(0).min(3) as f32 * 7.0 - 4.0);
            f.dl.overlay_discs.push(f.hud(p), 7.0, if *right { TRUTH } else { NAIVE });
            if *set == held {
                hud_circle(f, p, 6.5, 1.4, pal::WHITE);
            }
            marks.push((k, p));
        }
        self.hover_set = pointer.and_then(|q| marks.iter().filter(|(_, p)| p.distance(q) < 7.0).min_by(|a, b| a.1.distance(q).total_cmp(&b.1.distance(q))).map(|m| m.0));
        if let Some(k) = self.hover_set {
            if let (Some((set, est, _)), Some((_, p))) = (now.sets.get(k), marks.iter().find(|m| m.0 == k)) {
                hud_circle(f, *p, 7.5, 1.2, pal::WHITE.a(0.7));
                let label = format!("{}: ${}$", names(g, *set, "nothing held"), num(est.unwrap_or(0.0), 2));
                let w = f.text_size(&label, small).x;
                let x = p.x.max(lo.x + w * 0.5).min(hi.x - w * 0.5);
                f.caption_px(Vec2::new(x, p.y - 12.0), label, small, pal::TEXT, Align2::CENTER_BOTTOM);
                if f.ptr.clicked {
                    self.held[self.graph] = *set;
                    self.dirty = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::causal;

    /// The scene set up as experiment `k`, with every other sample drawn and everything
    /// glided into place.
    fn run(k: usize) -> Adjust {
        let mut s = Adjust::new();
        s.set_up(k);
        s.settle();
        s
    }

    /// The scene with the variables of these letters held instead.
    fn holding(s: &mut Adjust, letters: &[&str]) {
        s.held[s.graph] = Adjust::set_of(s.g(), letters);
        s.refresh();
    }

    fn estimate(s: &Adjust) -> f64 {
        s.now.estimate.expect("an estimate")
    }

    fn mean_of_others(s: &Adjust) -> f64 {
        let v: Vec<f64> = s.now.others.iter().flatten().copied().collect();
        v.iter().sum::<f64>() / v.len() as f64
    }

    fn path(s: &Adjust, markup: &str) -> usize {
        s.now.paths.iter().position(|p| p.markup(s.g()) == markup).unwrap_or_else(|| panic!("no path {markup}"))
    }

    fn errors(confounding: bool, part_removed: bool, collider: bool) -> Errors {
        Errors { confounding, part_removed, collider, effect_held: false }
    }

    /// "Fitted on X alone, Y rises by 1.14 for each unit of X, but the true effect is
    /// 0.90. The back door X ← Z → Y is open … That is confounding, and more data would
    /// not cure it: the 40 other samples make the same error."
    #[test]
    fn holding_nothing_leaves_the_back_door_open() {
        let s = run(0);
        assert_eq!((num(estimate(&s), 2).as_str(), num(s.now.truth, 2).as_str()), ("1.14", "0.90"));
        assert_eq!(s.now.errors, errors(true, false, false));
        assert_eq!(s.now.states[path(&s, r"X \gets Z \to Y")], State::Open);
        // every other sample is as far off, and so is endless data
        assert_eq!(s.now.others.len(), OTHERS);
        assert!(s.now.others.iter().all(|o| o.is_some_and(|v| v - s.now.truth > 0.18)), "{:?}", s.now.others);
        let off = s.now.endless.unwrap_or(0.0) - s.now.truth;
        assert!(off > 0.2, "{off}");
        // ten times the units: the spread shrinks, the error does not
        let model = Model::new(s.g(), &s.strengths[0]);
        let mut rng = Rng::new(5);
        for _ in 0..4 {
            let big = Sample::draw(&mut rng, 10 * UNITS);
            let v = coefficient(&model.covariance(&big.cov), 7, 0).unwrap_or(0.0) - s.now.truth;
            assert!((v - off).abs() < 0.02, "{v} against {off}");
        }
    }

    /// "Holding Z blocks the back door and leaves the paths along the arrows open. The
    /// estimate, 0.91, is the true effect but for the luck of this sample: the 40 other
    /// samples centre on 0.90."
    #[test]
    fn holding_the_confounder_gives_the_effect() {
        let s = run(1);
        assert_eq!(num(estimate(&s), 2), "0.91");
        assert!(s.now.errors.none());
        assert!(matches!(s.now.states[path(&s, r"X \gets Z \to Y")], State::Blocked(2)));
        assert!(s.now.paths.iter().zip(&s.now.states).all(|(p, st)| !p.causal() || *st == State::Open));
        assert_eq!(num(mean_of_others(&s), 2), "0.90");
        assert!((s.now.endless.unwrap_or(0.0) - s.now.truth).abs() < 1e-12);
    }

    /// "Holding M blocks X → M → Y …; holding C … opens X → C ← Y. The estimate falls
    /// to 0.10. Let C go and it is 0.48, near the direct effect alone (0.50, the arrow
    /// X → Y); hold C again and let M go, and it is 0.31."
    #[test]
    fn holding_everything_is_wrong_twice() {
        let mut s = run(2);
        assert_eq!(num(estimate(&s), 2), "0.10");
        assert_eq!(s.now.errors, errors(false, true, true));
        assert_eq!(s.now.states[path(&s, r"X \to M \to Y")], State::Blocked(3));
        assert_eq!(s.now.states[path(&s, r"X \to C \gets Y")], State::Opened { collider: 4, held: 4 });
        holding(&mut s, &["Z", "M", "I", "P"]);
        assert_eq!(num(estimate(&s), 2), "0.48");
        assert_eq!(s.now.errors, errors(false, true, false));
        // (the direct effect: the strength of X → Y, with data enough)
        assert!((s.now.endless.unwrap_or(0.0) - s.strengths[0][2]).abs() < 1e-12 && num(s.strengths[0][2], 2) == "0.50");
        holding(&mut s, &["Z", "C", "I", "P"]);
        assert_eq!(num(estimate(&s), 2), "0.31");
        assert_eq!(s.now.errors, errors(false, false, true));
    }

    /// "Z and I held: right on average, but the faint dots spread about twice as wide as
    /// with Z alone. Holding I takes away much of the spread of X … Hold P instead and the
    /// dots draw together, as P takes away part of the spread of Y. Hold I without Z
    /// and the error doubles, from +0.24 to +0.48."
    #[test]
    fn a_cause_of_x_only_costs_precision() {
        let mut s = run(3);
        assert!(s.now.errors.none());
        assert!((mean_of_others(&s) - s.now.truth).abs() < 0.01 && (s.now.endless.unwrap_or(0.0) - s.now.truth).abs() < 1e-12);
        let spread = |s: &Adjust| s.now.spread().unwrap_or(0.0);
        let cloud = |s: &Adjust, k: usize| {
            let v: Vec<f64> = s.cloud.iter().map(|c| if k == 0 { c.0 } else { c.1 }).collect();
            causal::variance(&v).sqrt()
        };
        let (with_i, x_with_i) = (spread(&s), cloud(&s, 0));
        holding(&mut s, &["Z"]);
        let (alone, x_alone, y_alone) = (spread(&s), cloud(&s, 0), cloud(&s, 1));
        holding(&mut s, &["Z", "P"]);
        let (with_p, y_with_p) = (spread(&s), cloud(&s, 1));
        assert!(s.now.errors.none());
        assert!((1.8..2.4).contains(&(with_i / alone)), "{with_i} against {alone}");
        assert!(with_p < 0.75 * alone, "{with_p} against {alone}");
        // and so they should, by the spread least squares itself expects: the variance
        // of Y left over by the fit, against the spread of X left over by what is held
        let model = Model::new(s.g(), &s.strengths[0]);
        let cov = model.covariance(&endless());
        let c = &cov;
        let expected = |letters: &[&str]| {
            let set = Adjust::set_of(s.g(), letters);
            let on: Vec<usize> = std::iter::once(0).chain((2..7).filter(|&i| has(set, i))).collect();
            let k = on.len();
            let table = || -> Vec<f64> { on.iter().flat_map(|&i| on.iter().map(move |&j| c[i * 7 + j])).collect() };
            let beta = solve(&mut table(), &mut on.iter().map(|&i| c[i * 7 + 1]).collect::<Vec<_>>(), k).unwrap_or_default();
            let left = c[8] - on.iter().zip(&beta).map(|(&i, b)| b * c[i * 7 + 1]).sum::<f64>();
            let mut unit = vec![0.0; k];
            unit[0] = 1.0;
            let first = solve(&mut table(), &mut unit, k).unwrap_or_default()[0];
            (left * first).sqrt()
        };
        let (z, zi, zp) = (expected(&["Z"]), expected(&["Z", "I"]), expected(&["Z", "P"]));
        assert!((1.7..2.3).contains(&(zi / z)) && zp < 0.75 * z, "expected: {zi} and {zp} against {z}");
        assert!(x_with_i < 0.6 * x_alone && y_with_p < 0.85 * y_alone, "X: {x_with_i} against {x_alone}; Y: {y_with_p} against {y_alone}");
        // holding I without Z makes the confounding worse
        holding(&mut s, &[]);
        let (before, endless_before) = (diff(estimate(&s) - s.now.truth), s.now.endless.unwrap_or(0.0) - s.now.truth);
        holding(&mut s, &["I"]);
        assert_eq!(s.now.errors, errors(true, false, false));
        let (after, endless_after) = (diff(estimate(&s) - s.now.truth), s.now.endless.unwrap_or(0.0) - s.now.truth);
        assert_eq!((before.as_str(), after.as_str()), ("+0.24", "+0.48"));
        assert!((1.9..2.3).contains(&(endless_after / endless_before)), "{endless_after} against {endless_before}");
    }

    /// "B is measured before X and goes with both X and Y … holding B as well as Z
    /// opens X ← U1 → B ← U2 → Y, and the estimate falls from 0.52 to 0.31 (the true
    /// effect is 0.50)."
    #[test]
    fn the_m_shape_punishes_holding_b() {
        let mut s = run(4);
        let model = Model::new(s.g(), &s.strengths[1]);
        let c = model.covariance(&s.sample.cov);
        let corr = |i: usize, j: usize| c[i * 6 + j] / (c[i * 6 + i] * c[j * 6 + j]).sqrt();
        assert!(corr(3, 0) > 0.25 && corr(3, 1) > 0.25, "{} {}", corr(3, 0), corr(3, 1));
        let u = path(&s, r"X \gets U_1 \to B \gets U_2 \to Y");
        assert_eq!(num(estimate(&s), 2), "0.31");
        assert_eq!(s.now.errors, errors(false, false, true));
        assert_eq!(s.now.states[u], State::Opened { collider: 3, held: 3 });
        holding(&mut s, &["Z"]);
        assert_eq!((num(estimate(&s), 2).as_str(), num(s.now.truth, 2).as_str()), ("0.52", "0.50"));
        assert!(s.now.errors.none());
        assert_eq!(s.now.states[u], State::Closed(3));
    }

    /// The butterfly's note: holding B closes three back doors and opens a fourth,
    /// which holding A or D as well closes again.
    #[test]
    fn the_butterfly_needs_b_and_one_more() {
        let mut s = Adjust::new();
        s.choose_graph(2);
        s.settle();
        let doors = |s: &Adjust| s.now.paths.iter().zip(&s.now.states).filter(|(p, st)| p.back_door() && st.open()).count();
        assert_eq!((doors(&s), s.now.paths.iter().filter(|p| p.back_door()).count()), (3, 4));
        holding(&mut s, &["B"]);
        let blocked = s.now.states.iter().filter(|st| **st == State::Blocked(2)).count();
        assert_eq!((blocked, doors(&s)), (3, 1));
        assert_eq!(s.now.states[path(&s, r"X \gets A \to B \gets D \to Y")], State::Opened { collider: 2, held: 2 });
        for other in ["A", "D"] {
            holding(&mut s, &["B", other]);
            assert!(s.now.errors.none() && doors(&s) == 0, "B and {other}");
        }
    }

    /// The truth is the sum, over the paths along the arrows, of the products of their
    /// strengths; and it is what setting X one higher does to Y, whatever else is
    /// cut away from X.
    #[test]
    fn the_true_effect_is_the_sum_over_the_paths_along_the_arrows() {
        let mut rng = Rng::new(3);
        for g in GRAPHS {
            for trial in 0..60 {
                let strengths: Vec<f64> = g.arrows.iter().map(|a| if trial == 0 { a.strength } else { rng.range(-2.0, 2.0) as f64 }).collect();
                let model = Model::new(g, &strengths);
                let sum: f64 = paths(g).iter().filter(|p| p.causal()).map(|p| p.arrows.iter().map(|&k| strengths[k]).product::<f64>()).sum();
                assert!((model.truth() - sum).abs() < 1e-12, "{}: {} against {sum}", g.name, model.truth());
                // the arrows into X cut: X is set from outside, and Y follows by as much
                let cut: Vec<f64> = g.arrows.iter().zip(&strengths).map(|(a, s)| if a.to == 0 { 0.0 } else { *s }).collect();
                assert!((Model::new(g, &cut).truth() - sum).abs() < 1e-12);
            }
        }
        let s = run(0);
        assert_eq!(truth_sum(&s.now.terms, s.now.truth), r"0.50 + 0.80 \cdot 0.50 = 0.90");
        assert_eq!(truth_sum(&[vec![-0.5], vec![0.8, -0.5]], -0.9), r"−0.50 + 0.80 \cdot (−0.50) = −0.90");
        assert_eq!(truth_sum(&[], 0.0), "0");
    }

    /// A set is right (with endless data the coefficient of X is the true effect)
    /// exactly when the paths say so; when it is not, it is off. And a set that keeps
    /// the rule is right. Over every set of every graph, for the graphs' own strengths
    /// and for random ones with some arrows taken away.
    #[test]
    fn a_set_is_right_exactly_when_its_paths_say_so() {
        let mut rng = Rng::new(9);
        let (mut right, mut wrong, mut kinds) = (0, 0, [0; 4]);
        for g in GRAPHS {
            let all = paths(g);
            for trial in 0..300 {
                let strengths: Vec<f64> = g
                    .arrows
                    .iter()
                    .map(|a| {
                        if trial == 0 {
                            a.strength
                        } else if rng.bool(0.2) {
                            0.0
                        } else {
                            rng.range(0.3, 2.0) as f64 * if rng.bool(0.5) { 1.0 } else { -1.0 }
                        }
                    })
                    .collect();
                let model = Model::new(g, &strengths);
                let c = model.covariance(&endless());
                for set in every_set(g) {
                    let (states, e) = judge(g, &strengths, &all, set);
                    let off = (coefficient(&c, g.vars.len(), set).unwrap_or(f64::NAN) - model.truth()).abs();
                    if e.none() {
                        right += 1;
                        assert!(off < 1e-9, "{}, {strengths:?}, holding {}: off by {off}", g.name, names(g, set, "nothing"));
                    } else {
                        wrong += 1;
                        assert!(off > 1e-7, "{}, {strengths:?}, holding {}: {e:?} but off by {off}", g.name, names(g, set, "nothing"));
                        for (k, on) in [e.confounding, e.part_removed, e.collider, e.effect_held].into_iter().enumerate() {
                            kinds[k] += on as usize;
                        }
                    }
                    let (doors, caused) = rule(g, &strengths, &all, &states, set);
                    assert!(!(doors && caused) || e.none(), "{}: the rule holds for {} but {e:?}", g.name, names(g, set, "nothing"));
                }
            }
        }
        assert!(right > 1000 && wrong > 1000 && kinds.iter().all(|k| *k > 20), "{right} right, {wrong} wrong, {kinds:?}");
    }

    /// The rule is enough, not needed: with Y → C at 0, C is caused by X alone, and
    /// holding it breaks the rule and leaves the estimate right. With the graphs' own
    /// strengths, though, a set is right exactly when it keeps the rule. And with X → C
    /// at 0 instead, C is an effect of Y alone: holding it opens no path, and is wrong.
    #[test]
    fn the_rule_is_enough_but_not_always_needed() {
        let mut s = run(1);
        s.set_strength(6, 0.0);
        holding(&mut s, &["Z", "C"]);
        assert!(s.now.errors.none() && s.now.rule == (true, false));
        assert!((s.now.endless.unwrap_or(0.0) - s.now.truth).abs() < 1e-12);
        s.set_strength(6, 0.6);
        s.set_strength(5, 0.0);
        s.refresh();
        assert_eq!(s.now.errors, Errors { effect_held: true, ..Errors::default() });
        assert!(!s.now.states.iter().any(|st| matches!(st, State::Opened { .. })));
        assert!((s.now.endless.unwrap_or(0.0) - s.now.truth).abs() > 0.1, "{:?} against {}", s.now.endless, s.now.truth);
        for g in GRAPHS {
            let strengths: Vec<f64> = g.arrows.iter().map(|a| a.strength).collect();
            let all = paths(g);
            for set in every_set(g) {
                let (states, e) = judge(g, &strengths, &all, set);
                let (doors, caused) = rule(g, &strengths, &all, &states, set);
                assert_eq!(e.none(), doors && caused, "{}: {}", g.name, names(g, set, "nothing"));
            }
        }
    }

    /// The fit worked out from covariances is the least-squares fit on the units, and
    /// the slope of what the fit sees is the estimate.
    #[test]
    fn the_fit_from_covariances_is_the_fit_on_the_units() {
        let s = Adjust::new();
        for g in GRAPHS {
            let strengths: Vec<f64> = g.arrows.iter().map(|a| a.strength).collect();
            let model = Model::new(g, &strengths);
            let p = g.vars.len();
            let values = model.values(&s.sample.noise);
            let column = |i: usize| -> Vec<f64> { values.chunks_exact(p).map(|v| v[i]).collect() };
            let c = model.covariance(&s.sample.cov);
            for set in every_set(g) {
                let held: Vec<Vec<f64>> = std::iter::once(0).chain((2..p).filter(|&i| has(set, i))).map(column).collect();
                let refs: Vec<&[f64]> = held.iter().map(|v| v.as_slice()).collect();
                let units = causal::regress(&refs, &column(1)).expect("a fit")[0];
                let ours = coefficient(&c, p, set).expect("a fit");
                assert!((units - ours).abs() < 1e-9, "{}: {units} against {ours}", g.name);
                let cloud = left_over(&model, &s.sample, set, UNITS);
                let (xs, ys): (Vec<f64>, Vec<f64>) = cloud.into_iter().unzip();
                assert!((causal::line(&xs, &ys).0 - ours).abs() < 1e-9);
            }
        }
    }

    /// Every path is found, written out and judged as the list says.
    #[test]
    fn the_paths_are_found_and_judged() {
        let written = |g: &GraphDef| paths(g).iter().map(|p| p.markup(g)).collect::<Vec<_>>();
        assert_eq!(written(&PLAIN), [r"X \to Y", r"X \to M \to Y", r"X \gets Z \to Y", r"X \to C \gets Y"]);
        assert_eq!(written(&M_SHAPE), [r"X \to Y", r"X \gets Z \to Y", r"X \gets U_1 \to B \gets U_2 \to Y"]);
        assert_eq!(written(&BUTTERFLY).len(), 5);
        let s = run(0);
        let why: Vec<String> = s.now.paths.iter().zip(&s.now.states).map(|(p, st)| reason(&PLAIN, p, *st, false).0).collect();
        assert_eq!(why, ["open: the direct effect", "open: the effect through $M$", "open back door: $Z$ is not held", "blocked at $C$, where two arrows meet"]);
        // an arrow of strength 0 takes its paths away, and with them the confounding
        let mut s = run(0);
        s.set_strength(1, 0.003);
        s.refresh();
        assert_eq!(s.strengths[0][1], 0.0);
        assert_eq!(s.now.states[path(&s, r"X \gets Z \to Y")], State::Absent(1));
        assert!(s.now.errors.none() && (s.now.endless.unwrap_or(0.0) - s.now.truth).abs() < 1e-12);
    }

    /// The labels beside the variables say what they are through the arrows there:
    /// with every arrow, what the graph was built with; with one taken away, what is left.
    #[test]
    fn the_labels_follow_the_arrows() {
        for g in GRAPHS {
            let strengths: Vec<f64> = g.arrows.iter().map(|a| a.strength).collect();
            for (i, v) in g.vars.iter().enumerate() {
                if let Kind::ByArrows(first) = v.kind {
                    assert_eq!(kind_of(g, &strengths, i), first, "{}: {}", g.name, v.letter);
                }
            }
        }
        let without = |k: usize| -> Vec<f64> { PLAIN.arrows.iter().enumerate().map(|(j, a)| if j == k { 0.0 } else { a.strength }).collect() };
        // (arrows: 1 is Z → Y, 3 is X → M, 4 is M → Y, 5 is X → C, 6 is Y → C)
        assert_eq!(kind_of(&PLAIN, &without(1), 2), "cause of $X$ only");
        assert_eq!(kind_of(&PLAIN, &without(3), 3), "cause of $Y$ only");
        assert_eq!(kind_of(&PLAIN, &without(4), 3), "effect of $X$");
        assert_eq!(kind_of(&PLAIN, &without(5), 4), "effect of $Y$");
        assert_eq!(kind_of(&PLAIN, &without(6), 4), "effect of $X$");
        assert_eq!(kind_of(&PLAIN, &[0.0; 9], 2), "");
    }

    /// An experiment shows the same thing whatever was done before it.
    #[test]
    fn an_experiment_sets_everything_it_depends_on() {
        for k in 0..TRIALS.len() {
            let fresh = run(k);
            let mut s = Adjust::new();
            s.choose_graph(2);
            s.set_strength(0, -1.5);
            s.toggle(2);
            s.new_sample(77);
            s.settle();
            s.choose_graph(TRIALS[k].2);
            s.set_strength(3, 0.0);
            s.set_up(k);
            s.settle();
            assert_eq!((s.graph, s.held(), s.seed), (fresh.graph, fresh.held(), fresh.seed));
            assert_eq!(s.strengths[s.graph], fresh.strengths[fresh.graph]);
            assert_eq!((s.now.estimate, s.now.others.clone()), (fresh.now.estimate, fresh.now.others.clone()));
            // and once it has glided into place, it is still
            assert!(!s.glide(1.0 / 60.0), "experiment {k} still moving");
        }
    }

    /// No arrows at all, every arrow at either end of its slider, nothing held or
    /// everything: every number drawn is finite, and the picture says what happened.
    #[test]
    fn degenerate_settings_leave_everything_finite() {
        for graph in 0..3 {
            for fill in [0.0, 2.0, -2.0, 1e-3] {
                let mut s = Adjust::new();
                s.choose_graph(graph);
                for k in 0..s.g().arrows.len() {
                    s.set_strength(k, if k % 2 == 0 { fill } else { -fill } as f32);
                }
                for set in every_set(s.g()) {
                    s.held[graph] = set;
                    s.settle();
                    let now = &s.now;
                    let finite = |v: &Option<f64>| v.is_some_and(f64::is_finite);
                    assert!(finite(&now.estimate) && finite(&now.endless) && now.truth.is_finite(), "graph {graph}, {fill}");
                    assert!(now.others.iter().all(finite) && now.sets.iter().all(|x| finite(&x.1)));
                    assert!(now.range.0.is_finite() && now.range.1.is_finite() && now.range.1 > now.range.0);
                    assert!(s.cloud.iter().all(|c| c.0.is_finite() && c.1.is_finite()) && s.reach.0 > 0.0 && s.reach.1 > 0.0);
                    if fill == 0.0 {
                        // no arrows: no effect, nothing to confound it, every set right
                        assert!(now.truth == 0.0 && now.terms.is_empty() && now.errors.none());
                    }
                }
            }
        }
    }

    /// On every canvas the parts of the picture lie inside it and apart.
    #[test]
    fn the_parts_of_the_picture_keep_apart() {
        for view in [Vec2::new(1080.0, 1000.0), Vec2::new(580.0, 700.0), Vec2::new(390.0, 800.0), Vec2::new(390.0, 330.0), Vec2::new(240.0, 180.0), Vec2::new(700.0, 560.0)] {
            for (count, one_line) in [(4, true), (5, true), (5, false)] {
                let rows = Rows { one_line, short: false, formula: 100.0, height: if one_line { 24.0 } else { 38.0 } };
                let lay = Lay::of(view, if view.x < 700.0 { 46.0 } else { 0.0 }, count, rows);
                let parts: Vec<Rect> = [Some(lay.graph), lay.list, lay.scatter, lay.gauge].into_iter().flatten().collect();
                for (i, a) in parts.iter().enumerate() {
                    assert!(a.0.cmpge(Vec2::ZERO).all() && a.1.cmple(view + 0.5).all() && a.0.cmple(a.1).all(), "{view}: {a:?}");
                    for b in &parts[..i] {
                        assert!(a.1.x <= b.0.x || b.1.x <= a.0.x || a.1.y <= b.0.y || b.1.y <= a.0.y, "{view}: {a:?} and {b:?}");
                    }
                }
                if view == Vec2::new(390.0, 800.0) && one_line {
                    assert!(lay.list.is_some() && lay.gauge.is_some() && lay.graph.1.y - lay.graph.0.y > 220.0, "{lay:?}");
                }
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        let mut scene = Adjust::new();
        scene.settle();
        for (gi, g) in GRAPHS.iter().enumerate() {
            let strengths = &scene.strengths[gi];
            let model = Model::new(g, strengths);
            let all = paths(g);
            println!("== {} : truth {:.4}", g.name, model.truth());
            let pop = model.covariance(&endless());
            let shown = model.covariance(&scene.sample.cov);
            for set in every_set(g) {
                let (_, e) = judge(g, strengths, &all, set);
                let others: Vec<f64> = scene.others.iter().filter_map(|c| coefficient(&model.covariance(c), g.vars.len(), set)).collect();
                let m = others.iter().sum::<f64>() / others.len() as f64;
                let sd = (others.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (others.len() - 1) as f64).sqrt();
                println!(
                    "   {:<24} endless {:+.4} sample {:+.4}  others mean {:+.4} sd {:.4}  {}",
                    names(g, set, "nothing"),
                    coefficient(&pop, g.vars.len(), set).unwrap_or(f64::NAN),
                    coefficient(&shown, g.vars.len(), set).unwrap_or(f64::NAN),
                    m,
                    sd,
                    if e.none() { "right".to_string() } else { format!("{e:?}") }
                );
            }
        }
    }

    /// A seed whose sample is typical where the experiments look (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn seeds() {
        let looks: [(usize, &[&str]); 7] = [(0, &[]), (0, &["Z"]), (0, &["Z", "M", "C", "I", "P"]), (0, &["Z", "I"]), (0, &["Z", "P"]), (1, &["Z", "B"]), (1, &["Z"])];
        for seed in 1..40u64 {
            let mut rng = Rng::new(seed);
            let sample = Sample::draw(&mut rng, UNITS);
            let others: Vec<[f64; MOST * MOST]> = (0..OTHERS).map(|_| Sample::draw(&mut rng, UNITS).cov).collect();
            let mut worst = 0.0f64;
            let mut line = String::new();
            for (gi, letters) in looks {
                let g = GRAPHS[gi];
                let strengths: Vec<f64> = g.arrows.iter().map(|a| a.strength).collect();
                let model = Model::new(g, &strengths);
                let set = Adjust::set_of(g, letters);
                let here = coefficient(&model.covariance(&sample.cov), g.vars.len(), set).unwrap();
                let ever = coefficient(&model.covariance(&endless()), g.vars.len(), set).unwrap();
                let v: Vec<f64> = others.iter().filter_map(|c| coefficient(&model.covariance(c), g.vars.len(), set)).collect();
                let m = v.iter().sum::<f64>() / v.len() as f64;
                let sd = (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt();
                let z = (here - ever) / sd;
                worst = worst.max(z.abs());
                line += &format!(" {z:+.2}");
            }
            println!("seed {seed:>2}: worst {worst:.2} |{line}");
        }
    }
}

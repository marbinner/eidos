//! Derivatives by machine: a formula drawn as a graph of small steps. One sweep forwards
//! gives its value; one sweep backwards gives the slope of the output with respect to
//! every box at once. Forward mode needs a sweep for each input; and a difference of two
//! values, the other way to a slope, is right to about ten digits at the best step.
//!
//! The graphs are written out here, a box at a time, and worked in f64. (The library's
//! formula compiler, `math/expr.rs`, keeps its steps to itself, so a typed formula cannot
//! be drawn as a graph; it is used here for what it can do: its dual numbers check the
//! slopes, and its Taylor series give the bend that the differences are wrong by.)

use super::*;
use crate::scenes::stats::plot::{self, Plot, Rect};

pub const DEF: SceneDef = SceneDef {
    id: "autodiff",
    title: "Derivatives by machine: the computation graph",
    group: GROUP,
    blurb: "A formula as a graph of small steps, each with a slope of its own: one sweep forwards for the value, one sweep backwards for every derivative at once.",
    make: |_| Box::new(Autodiff::new()),
};

/// The colour of a tangent, in forward mode: how fast a box changes as one input moves.
const TANGENT: Color = pal::BLUE;
/// The colour of the numbers on the arrows, each arrow's own slope.
const LOCAL: Color = pal::TEXT;
/// The colour of the digits of a difference that are wrong.
const WRONG: Color = pal::RED;
/// The colour of the chart's measured error.
const MEASURED: Color = pal::WHITE;

// ----- the formulas, as graphs ---------------------------------------------------------

/// What a box does to the boxes its arrows come from.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    /// Nothing: it is one of the formula's inputs, a number the visitor sets.
    Input,
    Add,
    Mul,
    Sin,
    Tanh,
    /// max(0, a).
    Relu,
}

/// How many arrows come into a box that does `op`.
fn arity(op: Op) -> usize {
    match op {
        Op::Input => 0,
        Op::Add | Op::Mul => 2,
        Op::Sin | Op::Tanh | Op::Relu => 1,
    }
}

/// One box of a graph.
#[derive(Clone, Copy, Debug)]
struct Node {
    op: Op,
    /// The boxes its arrows come from (the second only for a sum or a product).
    args: [usize; 2],
    /// Its name, and what it is made of, as markup: "u" and "x\,y".
    name: &'static str,
    does: &'static str,
    /// Its place: the column (the inputs to the left) and the row, in cells.
    at: [f32; 2],
}

const fn input(name: &'static str, at: [f32; 2]) -> Node {
    Node { op: Op::Input, args: [0, 0], name, does: "", at }
}

const fn step(op: Op, args: [usize; 2], name: &'static str, does: &'static str, at: [f32; 2]) -> Node {
    Node { op, args, name, does, at }
}

/// The most boxes a formula has here.
const MOST: usize = 10;

/// A formula, box by box: its inputs first, then each step after those it takes its
/// arguments from, the last of them the output.
struct Formula {
    /// On its button, on a wide screen and on a narrow one.
    label: &'static str,
    short: &'static str,
    /// And on the narrowest of screens.
    tiny: &'static str,
    /// Written out whole, as a heading.
    title: &'static str,
    /// As the library's formula compiler reads it, with its inputs' names there.
    text: &'static str,
    vars: &'static [&'static str],
    nodes: &'static [Node],
    /// The inputs' values it opens with.
    start: &'static [f64],
    /// What to notice about it.
    note: &'static str,
}

impl Formula {
    fn inputs(&self) -> usize {
        self.nodes.iter().take_while(|n| n.op == Op::Input).count()
    }

    fn last(&self) -> usize {
        self.nodes.len() - 1
    }
}

const TWO_ROADS: [Node; 5] = [
    input("x", [0.0, 0.0]),
    input("y", [0.0, 1.0]),
    step(Op::Mul, [0, 1], "u", r"x\,y", [1.0, 1.0]),
    step(Op::Sin, [0, 0], "v", r"\sin x", [1.0, 0.0]),
    step(Op::Add, [2, 3], "f", "u + v", [2.0, 0.5]),
];

const PRODUCT: [Node; 3] = [input("x", [0.0, 1.0]), step(Op::Sin, [0, 0], "s", r"\sin x", [1.0, 0.0]), step(Op::Mul, [0, 1], "f", r"x\,s", [2.0, 1.0])];

const NEURON: [Node; 10] = [
    input("w_1", [0.0, 0.0]),
    input("x_1", [0.0, 1.0]),
    input("w_2", [0.0, 2.0]),
    input("x_2", [0.0, 3.0]),
    input("b", [2.0, 3.2]),
    step(Op::Mul, [0, 1], "p_1", "w_1 x_1", [1.0, 0.5]),
    step(Op::Mul, [2, 3], "p_2", "w_2 x_2", [1.0, 2.5]),
    step(Op::Add, [5, 6], "s", "p_1 + p_2", [2.0, 1.5]),
    step(Op::Add, [7, 4], "z", "s + b", [3.0, 2.35]),
    step(Op::Tanh, [8, 8], "f", r"\tanh z", [4.0, 2.35]),
];

const KINK: [Node; 6] = [
    input("w", [0.0, 0.0]),
    input("x", [0.0, 1.0]),
    input("b", [1.0, 1.7]),
    step(Op::Mul, [0, 1], "p", r"w\,x", [1.0, 0.5]),
    step(Op::Add, [3, 2], "z", "p + b", [2.0, 1.1]),
    step(Op::Relu, [4, 4], "f", r"\max(0, z)", [3.0, 1.1]),
];

const FORMULAS: [Formula; 4] = [
    Formula {
        label: r"$x\,y + \sin x$",
        short: r"$xy + \sin x$",
        tiny: "1",
        title: r"$f = x\,y + \sin x$",
        text: "x*y + sin(x)",
        vars: &["x", "y"],
        nodes: &TWO_ROADS,
        start: &[2.0, 1.5],
        note: r"A sum, a product and a sine. $x$ is used twice, by the product and by the sine: two arrows leave its box, and its slope is a sum of two shares.",
    },
    Formula {
        label: r"$x \sin x$",
        short: r"$x \sin x$",
        tiny: "2",
        title: r"$f = x \sin x$",
        text: "x*sin(x)",
        vars: &["x"],
        nodes: &PRODUCT,
        start: &[0.8],
        note: r"One input, used twice: once in the product and once through the sine. The two shares of its slope are the two terms of the product rule.",
    },
    Formula {
        label: r"$\tanh(w_1 x_1 + w_2 x_2 + b)$",
        short: "a neuron",
        tiny: "3",
        title: r"$f = \tanh(w_1 x_1 + w_2 x_2 + b)$",
        text: "tanh(w1*x1 + w2*x2 + b)",
        vars: &["w1", "x1", "w2", "x2", "b"],
        nodes: &NEURON,
        start: &[0.8, 1.5, -0.6, 0.5, 0.2],
        note: r"One neuron: two weights, two inputs and a bias, five numbers in, one out. The backward sweep gives all five slopes at once; forward mode needs a sweep for each.",
    },
    Formula {
        label: r"$\max(0, w\,x + b)$",
        short: "a kink",
        tiny: "4",
        title: r"$f = \max(0, w\,x + b)$",
        text: "max(0, w*x + b)",
        vars: &["w", "x", "b"],
        nodes: &KINK,
        start: &[1.0, 0.5, -0.5],
        note: r"A ReLU neuron. $\max(0, z)$ has a corner at $z = 0$, where it has no slope: there the sweeps report the slope a rule gives them.",
    },
];

// ----- the arithmetic, in f64 ----------------------------------------------------------

/// Every box's value, for these inputs.
fn values(fm: &Formula, inputs: &[f64]) -> [f64; MOST] {
    let mut v = [0.0; MOST];
    for (i, n) in fm.nodes.iter().enumerate().take(MOST) {
        let (a, b) = (v[n.args[0].min(MOST - 1)], v[n.args[1].min(MOST - 1)]);
        v[i] = match n.op {
            // (the inputs are the first boxes)
            Op::Input => inputs.get(i).copied().unwrap_or(0.0),
            Op::Add => a + b,
            Op::Mul => a * b,
            Op::Sin => a.sin(),
            Op::Tanh => a.tanh(),
            Op::Relu => a.max(0.0),
        };
    }
    v
}

/// The slope of the arrow into box `i` from its argument `slot`: how much `i` changes
/// per unit change of that argument, the other held still. At the corner of max(0, z)
/// there is none, and 0 is taken (the usual rule, and the one the library's dual
/// numbers follow).
fn local(fm: &Formula, v: &[f64; MOST], i: usize, slot: usize) -> f64 {
    let Some(n) = fm.nodes.get(i) else { return 0.0 };
    let arg = |k: usize| v[n.args[k].min(MOST - 1)];
    match n.op {
        Op::Input => 0.0,
        Op::Add => 1.0,
        Op::Mul => arg(1 - slot.min(1)),
        Op::Sin => arg(0).cos(),
        Op::Tanh => 1.0 - v[i] * v[i],
        Op::Relu => {
            if arg(0) > 0.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// The arrows leaving box `i`: the box each enters, and which of its arguments it is.
fn leaving(fm: &Formula, i: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (j, n) in fm.nodes.iter().enumerate() {
        for slot in 0..arity(n.op) {
            if n.args[slot] == i {
                out.push((j, slot));
            }
        }
    }
    out
}

/// The backward sweep: the slope of the output with respect to every box. The output's
/// own is 1; each other box's is the sum, over the arrows leaving it, of the slope at
/// the arrow's far end times the arrow's slope (the far ends are all done by then).
fn slopes(fm: &Formula, v: &[f64; MOST]) -> [f64; MOST] {
    let mut s = [0.0; MOST];
    let last = fm.last().min(MOST - 1);
    s[last] = 1.0;
    for i in (0..last).rev() {
        s[i] = leaving(fm, i).iter().map(|&(j, slot)| s[j] * local(fm, v, j, slot)).sum();
    }
    s
}

/// A forward-mode sweep for input `seed`: how fast every box changes as that input
/// moves and the others stay. The input's own is 1, the others' 0; each box's is the
/// sum, over the arrows coming in, of the arrow's slope times the tangent at its start.
fn tangents(fm: &Formula, v: &[f64; MOST], seed: usize) -> [f64; MOST] {
    let mut t = [0.0; MOST];
    for (i, n) in fm.nodes.iter().enumerate().take(MOST) {
        t[i] = match n.op {
            Op::Input => (i == seed) as u8 as f64,
            op => (0..arity(op)).map(|slot| local(fm, v, i, slot) * t[n.args[slot].min(MOST - 1)]).sum(),
        };
    }
    t
}

/// The central difference for input `i`: (f(.. + h ..) − f(.. − h ..)) / 2h.
fn difference(fm: &Formula, inputs: &[f64], i: usize, h: f64) -> f64 {
    let n = fm.inputs().min(MOST);
    let (mut up, mut down) = ([0.0; MOST], [0.0; MOST]);
    let m = n.min(inputs.len());
    up[..m].copy_from_slice(&inputs[..m]);
    down[..m].copy_from_slice(&inputs[..m]);
    let i = i.min(n.saturating_sub(1));
    up[i] += h;
    down[i] -= h;
    let last = fm.last();
    (values(fm, &up)[last] - values(fm, &down)[last]) / (2.0 * h)
}

// ----- the chart of the differences' error ----------------------------------------------

/// The steps h the chart is drawn at: a tenth of a decade apart, from 10⁻¹⁵ to 1.
const H_POINTS: usize = 151;
const H_FROM: f64 = -15.0;
/// The errors it shows (powers of ten): an error of exactly 0 is drawn on the floor.
const ERR_FLOOR: f64 = -17.0;
const ERR_TOP: f64 = 1.0;

/// log₁₀ h at the chart's point `k`.
fn log_h(k: usize) -> f64 {
    H_FROM + k.min(H_POINTS - 1) as f64 / 10.0
}

/// The error at each of the chart's steps, as a power of ten (kept on the chart).
fn valley(fm: &Formula, inputs: &[f64], i: usize, slope: f64) -> [f64; H_POINTS] {
    let mut out = [0.0; H_POINTS];
    for (k, e) in out.iter_mut().enumerate() {
        let err = (difference(fm, inputs, i, 10f64.powf(log_h(k))) - slope).abs();
        *e = if err > 0.0 { err.log10().max(ERR_FLOOR).min(ERR_TOP) } else { ERR_FLOOR };
        if !e.is_finite() {
            *e = ERR_TOP;
        }
    }
    out
}

// ----- the film ------------------------------------------------------------------------

/// Backward mode: one sweep forwards for the values, one back for every slope. Forward
/// mode: one sweep for each input, carrying the values and how fast they change.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Backward,
    Forward,
}

/// What one step of the film works out.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Event {
    /// The inputs, as given.
    Start,
    /// A box's value, from the values coming in.
    Value(usize),
    /// The backward sweep begins: the output's slope with respect to itself, 1.
    Seed,
    /// A box's slope, from the slopes of the boxes it feeds.
    Slope(usize),
    /// A forward-mode sweep begins, for this input.
    Sweep(usize),
    /// A box's value and tangent, in the sweep for an input: (input, box).
    Tangent(usize, usize),
}

fn film(fm: &Formula, mode: Mode) -> Vec<Event> {
    let (n, last) = (fm.inputs(), fm.last());
    let mut e = vec![Event::Start];
    match mode {
        Mode::Backward => {
            e.extend((n..=last).map(Event::Value));
            e.push(Event::Seed);
            e.extend((0..last).rev().map(Event::Slope));
        }
        Mode::Forward => {
            for k in 0..n {
                e.push(Event::Sweep(k));
                e.extend((n..=last).map(|i| Event::Tangent(k, i)));
            }
        }
    }
    e
}

/// What the film shows of each box at a place in it (a step, or between two).
struct Seen {
    /// How far each box's value, slope (backward mode) and tangent (forward mode, in
    /// the sweep under way) have come in: 0, not yet; 1, there.
    value: [f32; MOST],
    slope: [f32; MOST],
    tangent: [f32; MOST],
    /// The sweep under way, in forward mode.
    sweep: usize,
    /// How lit each box is, and in what colour; how lit the arrows into it and out of it.
    lit: [f32; MOST],
    lit_color: [Color; MOST],
    into: [f32; MOST],
    out_of: [f32; MOST],
    /// Forward mode: how far each input's slope (found at the end of its sweep) has come in.
    found: [f32; MOST],
    /// The sweeps finished, and the inputs whose slope is known.
    sweeps: usize,
    known: usize,
}

fn seen(fm: &Formula, events: &[Event], at: f64) -> Seen {
    let n = fm.inputs();
    let last = fm.last();
    let mut s = Seen {
        value: [0.0; MOST],
        slope: [0.0; MOST],
        tangent: [0.0; MOST],
        sweep: 0,
        lit: [0.0; MOST],
        lit_color: [FORWARD; MOST],
        into: [0.0; MOST],
        out_of: [0.0; MOST],
        found: [0.0; MOST],
        sweeps: 0,
        known: 0,
    };
    let at = if at.is_finite() { at } else { 0.0 };
    // (each step's numbers come in as the film goes from the step before to it)
    let come = |e: usize| ((at - e as f64 + 1.0) as f32).max(0.0).min(1.0);
    let near = |e: usize| (1.0 - (at - e as f64).abs() as f32).max(0.0);
    let reached = |e: usize| at >= e as f64 - 0.5;
    let named = (at.round().max(0.0) as usize).min(events.len().saturating_sub(1));
    // the sweep under way: that of the step named
    s.sweep = events
        .iter()
        .take(named + 1)
        .rev()
        .find_map(|e| match e {
            Event::Sweep(k) | Event::Tangent(k, _) => Some(*k),
            _ => None,
        })
        .unwrap_or(0);
    for i in 0..n.min(MOST) {
        s.value[i] = 1.0;
    }
    let light = |s: &mut Seen, i: usize, amount: f32, color: Color| {
        if i < MOST && amount > s.lit[i] {
            s.lit[i] = amount;
            s.lit_color[i] = color;
        }
    };
    for (e, event) in events.iter().enumerate() {
        match *event {
            Event::Start => {
                for i in 0..n.min(MOST) {
                    light(&mut s, i, near(e), FORWARD);
                }
            }
            Event::Value(i) if i < MOST => {
                s.value[i] = come(e);
                light(&mut s, i, near(e), FORWARD);
                s.into[i] = s.into[i].max(near(e));
                if i == last && reached(e) {
                    s.sweeps += 1;
                }
            }
            Event::Seed => {
                s.slope[last.min(MOST - 1)] = come(e);
                light(&mut s, last, near(e), BACKWARD);
            }
            Event::Slope(i) if i < MOST => {
                s.slope[i] = come(e);
                light(&mut s, i, near(e), BACKWARD);
                s.out_of[i] = s.out_of[i].max(near(e));
                if i < n && reached(e) {
                    s.known += 1;
                }
                if i == 0 && reached(e) {
                    s.sweeps += 1;
                }
            }
            Event::Sweep(k) => {
                light(&mut s, k, near(e), TANGENT);
                if k == s.sweep {
                    for i in 0..n.min(MOST) {
                        s.tangent[i] = come(e);
                    }
                }
            }
            Event::Tangent(k, i) if i < MOST => {
                if k == 0 {
                    s.value[i] = come(e);
                }
                if k == s.sweep {
                    s.tangent[i] = come(e);
                }
                light(&mut s, i, near(e), TANGENT);
                s.into[i] = s.into[i].max(near(e));
                if i == last && k < MOST {
                    s.found[k] = come(e);
                    if reached(e) {
                        s.sweeps += 1;
                        s.known += 1;
                    }
                }
            }
            _ => {}
        }
    }
    s
}

// ----- writing numbers -------------------------------------------------------------------

/// A number of the graph to `decimals` places, as markup; a number too small for those
/// places that is not 0 is written with its power of ten (a box just off the kink's
/// corner says so, rather than "0.000").
fn number(v: f64, decimals: usize) -> String {
    if !v.is_finite() {
        return "?".into();
    }
    if v != 0.0 && v.abs() < 0.5 * 10f64.powi(-(decimals as i32)) {
        return format!("{v:.1e}");
    }
    num(v, decimals)
}

/// The places a box's value is written to: an input's, set in hundredths, to two.
fn places(fm: &Formula, i: usize) -> usize {
    if i < fm.inputs() {
        2
    } else {
        3
    }
}

/// A number as `number` writes it, rounded as it is written (what a reader adds up).
fn as_written(v: f64, decimals: usize) -> f64 {
    let scale = 10f64.powi(decimals as i32);
    (v * scale).round() / scale
}

/// A factor in a product, bracketed if it is negative.
fn factor(text: String) -> String {
    if text.starts_with('−') || text.starts_with('-') {
        format!("({text})")
    } else {
        text
    }
}

/// "=" if the numbers as written work out to the result as written, "≈" if their
/// rounding makes them miss it in the last place.
fn equals(from_written: f64, result: f64, decimals: usize) -> &'static str {
    if num(from_written, decimals) == num(result, decimals) {
        "="
    } else {
        r"\approx"
    }
}

/// The slope of box `of` with respect to box `by`, as markup: ∂u/∂x, a fraction or not.
fn partial(of: &str, by: &str, display: bool) -> String {
    if display {
        format!(r"\dfrac{{\partial {of}}}{{\partial {by}}}")
    } else {
        format!(r"\partial {of}/\partial {by}")
    }
}

/// What an arrow's slope is, in symbols: for the arrow into box `i` from its argument `slot`.
fn local_symbol(fm: &Formula, i: usize, slot: usize, v: &[f64; MOST]) -> String {
    let n = &fm.nodes[i];
    let arg = |k: usize| fm.nodes[n.args[k]].name;
    match n.op {
        Op::Input => String::new(),
        Op::Add => "1".into(),
        Op::Mul => arg(1 - slot.min(1)).into(),
        Op::Sin => format!(r"\cos {}", arg(0)),
        Op::Tanh => format!("1 - {}^2", n.name),
        Op::Relu => {
            if v[n.args[0]] > 0.0 {
                "1".into()
            } else {
                "0".into()
            }
        }
    }
}

/// What the rule for an arrow's slope is, in words, for the box it enters.
fn rule(op: Op) -> &'static str {
    match op {
        Op::Input => "",
        Op::Add => "for a sum, 1",
        Op::Mul => "for a product, the other factor",
        Op::Sin => "for a sine, the cosine",
        Op::Tanh => r"for $\tanh$, $1 - \tanh^2$",
        Op::Relu => r"for $\max(0, z)$, 1 where $z > 0$ and 0 where $z < 0$",
    }
}

/// What the film's step works out, written out: a heading, the rule in symbols, the
/// same with the numbers, and notes (markup; empty where there is nothing to say).
struct Told {
    head: String,
    rule: String,
    numbers: String,
    notes: Vec<String>,
}

fn told(fm: &Formula, mode: Mode, event: Event, inputs: &[f64]) -> Told {
    let v = values(fm, inputs);
    let s = slopes(fm, &v);
    let last = fm.last();
    let n = fm.inputs();
    let name = |i: usize| fm.nodes[i].name;
    let written = |i: usize| number(v[i], places(fm, i));
    // (each quantity in its colour: values cyan, slopes of the output pink, tangents blue)
    let bar = |i: usize| tint(BACKWARD, &format!(r"\bar {}", name(i)));
    let dot = |i: usize| tint(TANGENT, &format!(r"\dot {}", name(i)));
    let mut t = Told { head: String::new(), rule: String::new(), numbers: String::new(), notes: Vec::new() };
    match event {
        Event::Start => {
            let given: Vec<String> = (0..n).map(|i| format!("${} = {}$", name(i), tint(FORWARD, &written(i)))).collect();
            t.head = match mode {
                Mode::Backward => "the inputs, as given".into(),
                Mode::Forward => "forward mode: the inputs, as given".into(),
            };
            t.rule = given.join(", ");
            t.notes.push(match mode {
                Mode::Backward => "Everything else is worked out from them, a box at a time: forwards for the values, then backwards for the slopes.".into(),
                Mode::Forward => "Forward mode works out every value and, with it, how fast it changes as one input moves: a sweep for each input.".into(),
            });
        }
        Event::Value(i) => {
            let node = &fm.nodes[i];
            t.head = format!("forwards, box ${}$: its value from the values coming in", name(i));
            t.rule = format!("${} = {}$", name(i), node.does);
            let (a, b) = (node.args[0], node.args[1]);
            let (wa, wb) = (as_written(v[a], places(fm, a)), as_written(v[b], places(fm, b)));
            let (text, from_written) = match node.op {
                Op::Add => (format!("{} + {}", written(a), factor(written(b))), wa + wb),
                Op::Mul => (format!(r"{} \times {}", written(a), factor(written(b))), wa * wb),
                Op::Sin => (format!(r"\sin {}", factor(written(a))), wa.sin()),
                Op::Tanh => (format!(r"\tanh {}", factor(written(a))), wa.tanh()),
                Op::Relu => (format!(r"\max(0, {})", written(a)), wa.max(0.0)),
                Op::Input => (written(i), v[i]),
            };
            t.numbers = format!("$= {text} {} {}$", equals(from_written, v[i], 3), tint(FORWARD, &written(i)));
            let mut into: Vec<String> = (0..arity(node.op))
                .map(|slot| {
                    let (symbol, value) = (local_symbol(fm, i, slot, &v), number(local(fm, &v, i, slot), 3));
                    // (a slope that is a plain number, 1 or 0, is written once)
                    let said = if symbol.parse::<f64>().is_ok() { symbol } else { format!("{symbol} = {value}") };
                    format!("${} = {said}$", partial(name(i), name(node.args[slot]), false))
                })
                .collect();
            into.dedup();
            t.notes.push(format!("And the slopes of the arrows into it, kept for the way back: {}.", into.join(", ")));
        }
        Event::Seed => {
            t.head = format!("backwards from the output, box ${}$", name(last));
            t.rule = format!("${} = 1$", bar(last));
            t.notes.push(format!(r"$\bar u$ is the slope of ${0}$ with respect to the box $u$. For ${0}$ itself it is 1: a change in ${0}$ changes ${0}$ by as much.", name(last)));
        }
        Event::Slope(i) => {
            let out = leaving(fm, i);
            t.head = format!("backwards, box ${}$: its slope from the slopes of the boxes it feeds", name(i));
            let terms: Vec<String> = out.iter().map(|&(j, _)| format!(r"{}\,{}", bar(j), partial(name(j), name(i), true))).collect();
            t.rule = format!("${} = {}$", bar(i), terms.join(" + "));
            let numbers: Vec<String> = out.iter().map(|&(j, slot)| format!(r"{} \times {}", tint(BACKWARD, &number(s[j], 3)), factor(number(local(fm, &v, j, slot), 3)))).collect();
            let from_written: f64 = out.iter().map(|&(j, slot)| as_written(s[j], 3) * as_written(local(fm, &v, j, slot), 3)).sum();
            t.numbers = format!("$= {} {} {}$", numbers.join(" + "), equals(from_written, s[i], 3), tint(BACKWARD, &number(s[i], 3)));
            let mut rules: Vec<String> = out
                .iter()
                .map(|&(j, slot)| format!("${} = {}$, {}", partial(name(j), name(i), false), local_symbol(fm, j, slot, &v), rule(fm.nodes[j].op)))
                .collect();
            rules.dedup();
            t.notes.push(format!("The arrows' slopes: {}.", rules.join("; ")));
            if out.len() > 1 {
                t.notes.push(format!("Two arrows leave ${}$: two roads to ${}$, and their shares add.", name(i), name(last)));
            }
            if out.len() > 1 && out.iter().any(|&(j, _)| j == last) {
                let other = fm.nodes[last].args.iter().copied().find(|&a| a != i).map_or("", |a| name(a));
                t.notes.push(format!(r"(${}$ is the arrow's own slope, ${other}$ held still; ${}$ counts both roads.)", partial(name(last), name(i), false), bar(i)));
            }
            if out.iter().any(|&(j, _)| fm.nodes[j].op == Op::Relu && v[fm.nodes[j].args[0]] == 0.0) {
                t.notes.push(r"$z$ is exactly 0, the corner, where $\max(0, z)$ has no slope: the rule puts 0 on the arrow.".into());
            }
        }
        Event::Sweep(k) => {
            t.head = format!("forward mode, sweep {} of {n}: for the input ${}$", k + 1, name(k));
            let others: Vec<String> = (0..n).filter(|&i| i != k).map(dot).collect();
            t.rule = if others.is_empty() { format!("${} = 1$", dot(k)) } else { format!("${} = 1$, ${} = 0$", dot(k), others.join(" = ")) };
            t.notes.push(format!(r"$\dot u$ is how fast $u$ changes as ${}$ moves, the other inputs held still.", name(k)));
        }
        Event::Tangent(k, i) => {
            let node = &fm.nodes[i];
            let tg = tangents(fm, &v, k);
            t.head = format!("forward mode, sweep {} of {n}, box ${}$: how fast it changes as ${}$ moves", k + 1, name(i), name(k));
            let args: Vec<usize> = (0..arity(node.op)).collect();
            let terms: Vec<String> = args.iter().map(|&slot| format!(r"{}\,{}", partial(name(i), name(node.args[slot]), true), dot(node.args[slot]))).collect();
            t.rule = format!("${} = {}$", dot(i), terms.join(" + "));
            let tangent_text = |a: usize| tint(TANGENT, &if a < n { format!("{}", tg[a] as i64) } else { factor(number(tg[a], 3)) });
            let numbers: Vec<String> = args.iter().map(|&slot| format!(r"{} \times {}", number(local(fm, &v, i, slot), 3), tangent_text(node.args[slot]))).collect();
            let from_written: f64 = args.iter().map(|&slot| as_written(local(fm, &v, i, slot), 3) * as_written(tg[node.args[slot]], 3)).sum();
            t.numbers = format!("$= {} {} {}$", numbers.join(" + "), equals(from_written, tg[i], 3), tint(TANGENT, &number(tg[i], 3)));
            t.notes.push(format!("With its value, ${} = {}$.", name(i), tint(FORWARD, &written(i))));
            if i == last {
                t.notes.push(format!("At the output: ${} = {} = {}$, the one slope this sweep finds.", partial(name(last), name(k), false), dot(last), number(tg[i], 3)));
            }
        }
    }
    t
}

// ----- where things go -------------------------------------------------------------------

/// The buttons' words, the longest each can say (for their widths).
const FILM_WORDS: [&str; 4] = ["◀", "▶ step", "❚❚ pause", "↺ again"];
const MODE_WORDS: [&str; 2] = ["backward", "forward"];

#[derive(Clone, Debug)]
struct Layout {
    narrow: bool,
    /// The size of the words on the buttons, and of the small print.
    text: f32,
    small: f32,
    /// Back, step, play, again; backward, forward; the formulas.
    film: Vec<Rect>,
    modes: Vec<Rect>,
    mode_lead: Option<Vec2>,
    formulas: Vec<Rect>,
    formula_lead: Option<Vec2>,
    /// The graph's heading (where, and how wide), and the room for the graph.
    heading: Option<(Vec2, f32)>,
    graph: Option<Rect>,
    /// What the step works out, and the count of sweeps: top-left corners and widths.
    sums: Option<(Vec2, f32)>,
    tally: Option<(Vec2, f32)>,
    /// The chart, and its numbers under it.
    chart: Option<Rect>,
    readouts: Option<(Vec2, f32)>,
}

impl Layout {
    /// The room the chart's title takes above it, and its numbers below it.
    fn above(&self) -> f32 {
        if self.narrow {
            28.0
        } else {
            46.0
        }
    }

    fn below(&self) -> f32 {
        if self.narrow {
            30.0
        } else {
            34.0
        }
    }

    fn of(f: &Frame, view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::ONE);
        let narrow = view.x < 760.0;
        let (text, small) = if narrow { (11.0, 10.5) } else { (12.5, 12.0) };
        let side = if narrow { 8.0 } else { 16.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let (h, gap) = if narrow { (26.0, 5.0) } else { (28.0, 8.0) };
        let mut lay = Layout {
            narrow,
            text,
            small,
            film: Vec::new(),
            modes: Vec::new(),
            mode_lead: None,
            formulas: Vec::new(),
            formula_lead: None,
            heading: None,
            graph: None,
            sums: None,
            tally: None,
            chart: None,
            readouts: None,
        };
        if view.x < 240.0 || view.y < 180.0 {
            return lay;
        }
        let width = |w: &str| f.text_size(w, text).x + if narrow { 14.0 } else { 20.0 };
        // the film's buttons and the modes, in a row; then the formulas
        let words: Vec<f32> = FILM_WORDS.iter().chain(MODE_WORDS.iter()).map(|w| width(w)).collect();
        let lead = if narrow { 0.0 } else { f.text_size("mode:", text).x + 10.0 };
        let wanted: f32 = words.iter().sum::<f32>() + gap * 5.0 + lead + 18.0;
        let row = if wanted <= view.x - 2.0 * side {
            let mut x = side;
            let mut rects = Vec::new();
            for (k, w) in words.iter().enumerate() {
                if k == 4 {
                    x += 18.0;
                    if !narrow {
                        lay.mode_lead = Some(Vec2::new(x, top + h * 0.5));
                    }
                    x += lead;
                }
                rects.push((Vec2::new(x, top), Vec2::new(x + w, top + h)));
                x += w + gap;
            }
            rects
        } else {
            plot::even_row(6, side, view.x - side, top, h, 3.0)
        };
        lay.film = row[..4].to_vec();
        lay.modes = row[4..].to_vec();
        let row2 = top + h + gap;
        if narrow {
            lay.formulas = plot::even_row(FORMULAS.len(), side, view.x - side, row2, h, 4.0);
        } else {
            let lead = f.text_size("formula:", text).x + 10.0;
            let labels: Vec<&str> = FORMULAS.iter().map(|fm| fm.label).collect();
            lay.formulas = plot::button_row(f, &labels, side + lead, row2, h, text, gap);
            lay.formula_lead = Some(Vec2::new(side, row2 + h * 0.5));
        }
        let body = row2 + h + if narrow { 10.0 } else { 16.0 };
        if narrow {
            // one under another: the graph, what the step works out, the count of
            // sweeps, the chart and its numbers (the formula is named on its button)
            let sums_h = 120.0;
            let tally_h = 52.0;
            let read_h = 3.0 * 17.5 + 4.0;
            let chart_h = (view.y * 0.15).max(80.0).min(170.0);
            let below = sums_h + tally_h + lay.above() + chart_h + lay.below() + read_h + 6.0;
            let mut graph_bottom = view.y - below;
            let with_chart = graph_bottom - body >= 200.0;
            if !with_chart {
                graph_bottom = view.y - sums_h - tally_h - 6.0;
            }
            if graph_bottom - body < 90.0 {
                return lay;
            }
            lay.graph = Some((Vec2::new(side, body), Vec2::new(view.x - side, graph_bottom)));
            lay.sums = Some((Vec2::new(side, graph_bottom + 6.0), view.x - 2.0 * side));
            lay.tally = Some((Vec2::new(side, graph_bottom + 6.0 + sums_h), view.x - 2.0 * side));
            if with_chart {
                let chart_top = graph_bottom + 6.0 + sums_h + tally_h + lay.above();
                lay.chart = Some((Vec2::new(side + 34.0, chart_top), Vec2::new(view.x - side - 6.0, chart_top + chart_h)));
                lay.readouts = Some((Vec2::new(side, chart_top + chart_h + lay.below()), view.x - 2.0 * side));
            }
            return lay;
        }
        // wide: the graph across the top; under it, what the step works out and the
        // count of sweeps on the left, and the chart on the right
        let bottom_h = ((view.y - body) * 0.46).max(250.0).min(440.0);
        let graph_bottom = view.y - 12.0 - bottom_h - 14.0;
        if graph_bottom - body - 22.0 < 120.0 {
            return lay;
        }
        lay.heading = Some((Vec2::new(side, body), view.x - 2.0 * side));
        lay.graph = Some((Vec2::new(side, body + 24.0), Vec2::new(view.x - side, graph_bottom)));
        let left = ((view.x - 2.0 * side) * 0.52).floor();
        let bottom_top = graph_bottom + 14.0;
        lay.sums = Some((Vec2::new(side, bottom_top), left));
        lay.tally = Some((Vec2::new(side, bottom_top + 196.0), left));
        let x = side + left + 40.0;
        let read_h = 3.0 * 19.0;
        let chart_bottom = view.y - 12.0 - read_h - lay.below();
        if chart_bottom - bottom_top - lay.above() >= 70.0 && view.x - side - x >= 200.0 {
            lay.chart = Some((Vec2::new(x + 34.0, bottom_top + lay.above()), Vec2::new(view.x - side - 6.0, chart_bottom)));
            lay.readouts = Some((Vec2::new(x, chart_bottom + lay.below()), view.x - side - x));
        }
        lay
    }
}

/// Where the boxes of a formula go in the room for the graph.
struct Placed {
    rects: [Rect; MOST],
    /// The text in the boxes and on the arrows, and the height of a line in a box.
    text: f32,
    label: f32,
    line: f32,
    /// Three lines to a box, saying what their numbers are (∂f/∂u = …), or two, with
    /// the numbers alone side by side.
    full: bool,
    /// The room for an input's knob, at the foot of its box.
    knob: f32,
    /// The inputs on the left (or at the top).
    across: bool,
}

fn place(fm: &Formula, (min, max): Rect, narrow: bool) -> Option<Placed> {
    let room = max - min;
    if !(room.x > 40.0 && room.y > 40.0) {
        return None;
    }
    let full = !narrow;
    // a box at scale 1: its width, its text, the knob's room in an input's; the gaps
    // between boxes along the arrows (across the screen, and down it) and beside them
    let (w, text, knob, along, along_down, beside) = if full { (140.0f32, 13.0f32, 16.0f32, 64.0f32, 40.0f32, 30.0f32) } else { (80.0, 11.0, 12.0, 26.0, 16.0, 10.0) };
    let line = text + 5.0;
    let lines = if full { 3.0 } else { 2.0 };
    let h = lines * line + 6.0 + knob;
    let (cols, rows) = fm.nodes.iter().fold((0.0f32, 0.0f32), |(c, r), n| (c.max(n.at[0]), r.max(n.at[1])));
    // inputs on the left, or along the top: whichever lets the boxes be larger
    let across = ((room.x / (cols * (w + along) + w)).min(room.y / (rows * (h + beside) + h)), true);
    let down = ((room.x / (rows * (w + beside) + w)).min(room.y / (cols * (h + along_down) + h)), false);
    let (scale, across) = if across.0 >= 0.95 * down.0 { across } else { down };
    let scale = scale.min(if full { 1.25 } else { 1.15 });
    if scale < 0.6 {
        return None;
    }
    let (bw, bh) = (w * scale, h * scale);
    let (gx, gy) = if across { (cols, rows) } else { (rows, cols) };
    let pitch = |space: f32, cells: f32, size: f32, most: f32| if cells > 0.0 { ((space - size) / cells).min(size * most).max(0.0) } else { 0.0 };
    let (px, py) = (pitch(room.x, gx, bw, 2.0), pitch(room.y, gy, bh, 2.4));
    let used = Vec2::new(gx * px + bw, gy * py + bh);
    let origin = min + (room - used) * 0.5;
    let mut rects = [(Vec2::ZERO, Vec2::ZERO); MOST];
    for (i, n) in fm.nodes.iter().enumerate().take(MOST) {
        let (cx, cy) = if across { (n.at[0], n.at[1]) } else { (n.at[1], n.at[0]) };
        let at = origin + Vec2::new(cx * px, cy * py);
        let tall = if n.op == Op::Input { bh } else { bh - knob * scale };
        rects[i] = (at, at + Vec2::new(bw, tall));
    }
    let size = (text * scale).max(9.0).min(15.0);
    Some(Placed { rects, text: size, label: (size - 1.5).max(8.5), line: line * scale, full, knob: knob * scale, across })
}

/// Where the arrow from one box to another starts and ends: on the edges of the two.
fn arrow_ends((a0, a1): Rect, (b0, b1): Rect) -> (Vec2, Vec2) {
    let (ca, cb) = ((a0 + a1) * 0.5, (b0 + b1) * 0.5);
    let d = cb - ca;
    // how far along d the line leaves a box of half-size `half` round its centre
    let out = |half: Vec2| {
        let tx = if d.x.abs() > 1e-6 { half.x / d.x.abs() } else { f32::MAX };
        let ty = if d.y.abs() > 1e-6 { half.y / d.y.abs() } else { f32::MAX };
        tx.min(ty)
    };
    let gap = 4.0 / d.length().max(1.0);
    let (ta, tb) = (out((a1 - a0) * 0.5) + gap, 1.0 - out((b1 - b0) * 0.5) - gap);
    if ta >= tb {
        return (ca, ca);
    }
    (ca + d * ta, ca + d * tb)
}

/// Where an input's knob is on its track, for the value `v`.
fn knob_at(p: &Placed, r: &Rect, v: f64) -> Vec2 {
    let (min, max) = *r;
    let y = max.y - p.knob * 0.5 - 2.0;
    let (x0, x1) = (min.x + 10.0, max.x - 10.0);
    let t = ((v + KNOB) / (2.0 * KNOB)).max(0.0).min(1.0) as f32;
    Vec2::new(x0 + (x1 - x0) * t, y)
}

/// The values a knob sets run from −KNOB to KNOB, in hundredths.
const KNOB: f64 = 3.0;

fn hundredths(v: f64) -> f64 {
    ((v * 100.0).round() / 100.0).max(-KNOB).min(KNOB)
}

// ----- lines by kind ---------------------------------------------------------------------

/// The chart's curve, its dashed guides and the line at the step h, each in a batch of
/// its own (see `plot::apart`).
const CURVE: u8 = plot::OWN;
const GUIDES: u8 = plot::OWN + 1;
const MARK: u8 = plot::OWN + 2;

/// Every kind of line in the order its batch is made in every frame.
fn in_order(f: &mut Frame) {
    plot::apart(f, plot::AXES, false);
    plot::apart(f, plot::BUTTONS, false);
    plot::apart(f, CURVE, false);
    plot::apart(f, GUIDES, true);
    plot::apart(f, MARK, true);
}

// ----- numbers that glide ----------------------------------------------------------------

/// How fast what is drawn follows a change of setting: each of two eases one after the
/// other covers two thirds of what is left in this many seconds.
const FOLLOW: f64 = 0.08;

/// Numbers as set and as drawn: the drawn ones follow through two eases one after the
/// other, so that they set off gently and arrive exactly. (In f64: what arrives is the
/// number set, to the last bit, which the kink's corner needs.)
#[derive(Clone, Debug, Default)]
struct Follow {
    to: Vec<f64>,
    between: Vec<f64>,
    shown: Vec<f64>,
}

impl Follow {
    fn at(v: &[f64]) -> Follow {
        Follow { to: v.to_vec(), between: v.to_vec(), shown: v.to_vec() }
    }

    /// Aim at `v` (a list of another length starts there).
    fn aim(&mut self, v: &[f64]) {
        if v.len() != self.shown.len() {
            *self = Follow::at(v);
        } else {
            self.to = v.to_vec();
        }
    }

    /// Be at `v` at once: what a hand moves.
    fn now(&mut self, v: &[f64]) {
        *self = Follow::at(v);
    }

    fn arrived(&self) -> bool {
        self.shown == self.to && self.between == self.to
    }

    fn step(&mut self, dt: f64) {
        if self.arrived() {
            return;
        }
        let k = 1.0 - (-dt.max(0.0) / FOLLOW).exp();
        let mut far = false;
        for ((b, s), t) in self.between.iter_mut().zip(self.shown.iter_mut()).zip(&self.to) {
            *b += (t - *b) * k;
            *s += (*b - *s) * k;
            far |= !((t - *b).abs() < 1e-4 && (t - *s).abs() < 1e-4);
        }
        // (exactly there in the end: the picture then stops changing)
        if !far || self.shown.iter().chain(&self.between).any(|v| !v.is_finite()) {
            *self = Follow::at(&self.to.clone());
        }
    }
}

// ----- the experiments -------------------------------------------------------------------

/// Where an experiment's film starts.
#[derive(Clone, Copy, Debug, PartialEq)]
enum From {
    /// At the inputs, and plays.
    Beginning,
    /// At the start of the backward sweep, and plays.
    Backwards,
    /// At its end, everything worked out.
    End,
}

struct Trial {
    label: &'static str,
    look_for: &'static str,
    formula: usize,
    values: &'static [f64],
    mode: Mode,
    from: From,
    /// Steps a second, if it plays.
    rate: f32,
    /// The input the chart is about, and the step h, as its place on the chart.
    chosen: usize,
    h: usize,
    /// Whether h walks down from 0.1 to there, a decade at a time.
    walk: bool,
}

/// The chart's places of some steps: 10⁻¹², 10⁻⁵, 10⁻⁴ and 0.1.
const H_E12: usize = 30;
const H_E5: usize = 100;
const H_E4: usize = 110;
const H_TENTH: usize = 140;

const TRIALS: [Trial; 5] = [
    Trial {
        label: "forwards for the value, backwards for every slope",
        look_for: r"$f = x\,y + \sin x$ at $x = 2$, $y = 1.5$, a box at a time. Forwards, each box's value from the values coming in (cyan), to $f = 3.909$. Then backwards from $f$, whose slope with respect to itself is $\bar f = 1$: each box's slope $\bar u$ (pink) is the sum, over the arrows leaving it, of the slope at the arrow's far end times the number on the arrow. The last two are the inputs: $\bar y = \partial f/\partial y = 2.000$ and $\bar x = \partial f/\partial x = 1.084$, both from the one sweep back.",
        formula: 0,
        values: &[2.0, 1.5],
        mode: Mode::Backward,
        from: From::Beginning,
        rate: 1.2,
        chosen: 0,
        h: H_E5,
        walk: false,
    },
    Trial {
        label: "a variable used twice: its roads add",
        look_for: r"$f = x \sin x$ at $x = 0.8$. Two arrows leave $x$: one into the sine, with slope $\cos x = 0.697$, which goes on into the product with slope $x$, so $\bar s = 0.800$; and one straight into the product, with slope the other factor, $\sin x = 0.717$, where $\bar f = 1$. Each road brings back the slope at its far end times its own, and they add: $\bar x = 0.800 \times 0.697 + 1 \times 0.717 = 1.275$. That is the product rule, $(x \sin x)' = x \cos x + \sin x$, from the graph alone.",
        formula: 1,
        values: &[0.8],
        mode: Mode::Backward,
        from: From::Backwards,
        rate: 1.0,
        chosen: 0,
        h: H_E5,
        walk: false,
    },
    Trial {
        label: "one sweep back, or one forward for each input",
        look_for: r"A neuron, five inputs. Forward mode carries, from the inputs to $f$, how fast every box changes as one input moves (blue): a sweep ends with one slope, so the five take five sweeps. Switch back to backward: there all five came out of one sweep forwards and one back, the same numbers, $0.539$, $0.287$, $0.180$, $-0.216$ and $0.359$. The loss of a network is one output of millions of inputs, its weights: backwards that is still two sweeps; forwards, millions.",
        formula: 2,
        values: &[0.8, 1.5, -0.6, 0.5, 0.2],
        mode: Mode::Forward,
        from: From::Beginning,
        rate: 3.5,
        chosen: 0,
        h: H_E5,
        walk: false,
    },
    Trial {
        label: "differences: a step too large, and too small",
        look_for: r"Back to $f = x\,y + \sin x$, and $\partial f/\partial x$ from two values, $(f(x + h) - f(x - h))/2h$. The dot walks $h$ down from 0.1, a tenth at a time. At $h = 0.1$ the difference is off by $7 \times 10^{-4}$: $f$ bends within the step, an error that shrinks a hundredfold with each tenth of $h$. At $h = 10^{-5}$, near the floor of the valley, it is off by $3 \times 10^{-11}$: ten digits right. Below that, the values of $f$, each rounded to about 16 digits, are divided by $2h$, and by $h = 10^{-12}$ the error is back up to about $10^{-4}$. The sweeps give the slope to the last digit, for about the work of a few values of $f$.",
        formula: 0,
        values: &[2.0, 1.5],
        mode: Mode::Backward,
        from: From::End,
        rate: 1.2,
        chosen: 0,
        h: H_E12,
        walk: true,
    },
    Trial {
        label: r"a kink: the slope at the corner of $\max(0, z)$",
        look_for: r"$f = \max(0, w\,x + b)$ at $w = 1$, $x = 0.5$, $b = -0.5$, where $z$ is exactly 0: the corner, with slope 0 to the left and 1 to the right, and none at all at the corner. The backward sweep must still put a number on the arrow out of $z$, and by the usual rule it puts 0: $\bar z = 0$, and every slope behind it is 0. The difference straddles the corner and gives the average of the two sides: $\partial f/\partial b$ comes out about 0.5 for every $h$, a flat line on the chart, no valley. Drag $b$ off the corner and the two agree again, for every $h$ smaller than the distance to it.",
        formula: 3,
        values: &[1.0, 0.5, -0.5],
        mode: Mode::Backward,
        from: From::Backwards,
        rate: 1.0,
        chosen: 2,
        h: H_E4,
        walk: false,
    },
];

/// Seconds between one step of the walk of h and the next.
const WALK_EVERY: f32 = 0.6;
/// How long a change of formula takes to fade through (seconds).
const FADE: f32 = 0.4;

// ----- the scene -----------------------------------------------------------------------

/// The chart as measured: for which inputs and which of them chosen, the errors (as
/// measured, and as drawn), and the time since. It follows the inputs at most every
/// `REFRESH` seconds, gliding to each new measurement: the rounding part of the curve
/// jumps about with every hundredth an input moves, and redrawn as often as that it
/// would flicker while a knob is dragged.
struct Measured {
    at: Vec<f64>,
    chosen: usize,
    curve: Follow,
    /// The power of ten of the bend there (see `bend`), as measured and as drawn.
    bend: Follow,
    since: f32,
}

/// The bend's power of ten where f has none along the input (a line far below the chart).
const BEND_NONE: f64 = -40.0;

/// The least time between two measurements of the chart while the inputs change (seconds).
const REFRESH: f32 = 0.25;

/// A formula as it was, fading out after a change of formula.
struct Old {
    formula: usize,
    values: Vec<f64>,
    mode: Mode,
    at: f64,
}

pub struct Autodiff {
    formula: usize,
    mode: Mode,
    /// The inputs as set, and as drawn (gliding to a new setting).
    values: Follow,
    /// The input the chart is about.
    chosen: usize,
    /// The film's steps, the step reached and the step drawn (following it).
    events: Vec<Event>,
    step: usize,
    at: Follow,
    playing: bool,
    rate: f32,
    owed: f32,
    /// The step h, as its place on the chart, and the chart's dot as drawn: (log₁₀ h,
    /// log₁₀ of the error there).
    hk: usize,
    dot: Follow,
    measured: Measured,
    /// Walking h down a decade at a time: seconds to the next step, and where it stops.
    walk: Option<f32>,
    walk_to: usize,
    /// After a change of formula, the old one fading out, and how far the change has gone.
    old: Option<Old>,
    fade: f32,
    /// Each formula as the library's compiler reads it (for its Taylor series).
    compiled: Vec<Expr>,
}

impl Autodiff {
    fn new() -> Autodiff {
        let fm = &FORMULAS[0];
        let events = film(fm, Mode::Backward);
        let end = events.len() - 1;
        let compiled = FORMULAS.iter().map(|fm| Expr::parse(fm.text, fm.vars).unwrap_or_else(|_| Expr::parse("0", fm.vars).expect("0 parses"))).collect();
        let mut scene = Autodiff {
            formula: 0,
            mode: Mode::Backward,
            values: Follow::at(fm.start),
            chosen: 0,
            events,
            step: end,
            at: Follow::at(&[end as f64]),
            playing: false,
            rate: 1.5,
            owed: 0.0,
            hk: H_E5,
            dot: Follow::default(),
            measured: Measured { at: Vec::new(), chosen: 0, curve: Follow::default(), bend: Follow::default(), since: 0.0 },
            walk: None,
            walk_to: H_E5,
            old: None,
            fade: 1.0,
            compiled,
        };
        scene.measure(true);
        scene.dot = Follow::at(&scene.dot_wanted());
        scene
    }

    fn fm(&self) -> &'static Formula {
        &FORMULAS[self.formula.min(FORMULAS.len() - 1)]
    }

    fn last_step(&self) -> usize {
        self.events.len().saturating_sub(1)
    }

    /// The inputs as drawn.
    fn drawn(&self) -> &[f64] {
        &self.values.shown
    }

    /// Another formula: the old one fades out and the new one in, worked out to the end.
    fn choose(&mut self, k: usize) {
        let k = k.min(FORMULAS.len() - 1);
        if k == self.formula {
            return;
        }
        self.old = Some(Old { formula: self.formula, values: self.values.shown.clone(), mode: self.mode, at: self.at.shown.first().copied().unwrap_or(0.0) });
        self.fade = 0.0;
        self.formula = k;
        self.values = Follow::at(self.fm().start);
        self.chosen = 0;
        self.events = film(self.fm(), self.mode);
        self.step = self.last_step();
        self.at = Follow::at(&[self.step as f64]);
        (self.playing, self.owed, self.walk) = (false, 0.0, None);
        self.measure(true);
        self.dot = Follow::at(&self.dot_wanted());
    }

    /// Backward or forward: the new mode's film from its start, playing.
    fn set_mode(&mut self, mode: Mode) {
        if mode == self.mode {
            return;
        }
        self.mode = mode;
        self.events = film(self.fm(), mode);
        self.step = 0;
        self.at = Follow::at(&[0.0]);
        (self.playing, self.owed) = (true, 0.0);
    }

    /// Set the inputs (in hundredths): at once if a hand moved them, gliding if not.
    fn set_values(&mut self, v: &[f64], by_hand: bool) {
        let n = self.fm().inputs();
        let v: Vec<f64> = (0..n).map(|i| hundredths(v.get(i).copied().unwrap_or(0.0))).collect();
        if by_hand {
            self.values.now(&v);
        } else {
            self.values.aim(&v);
        }
    }

    /// Set experiment `k` up.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k.min(TRIALS.len() - 1)];
        self.choose(t.formula);
        self.set_values(t.values, false);
        self.mode = t.mode;
        self.events = film(self.fm(), t.mode);
        let last = self.last_step();
        self.step = match t.from {
            From::Beginning => 0,
            From::Backwards => self.events.iter().position(|e| *e == Event::Seed).unwrap_or(0),
            From::End => last,
        };
        // (the film is drawn from where it now is: no running through every box between)
        self.at = Follow::at(&[self.step as f64]);
        self.playing = t.from != From::End;
        (self.rate, self.owed) = (t.rate, 0.0);
        self.chosen = t.chosen.min(self.fm().inputs() - 1);
        if t.walk {
            (self.hk, self.walk, self.walk_to) = (H_TENTH, Some(WALK_EVERY), t.h);
        } else {
            (self.hk, self.walk) = (t.h, None);
        }
    }

    fn step_on(&mut self) {
        self.playing = false;
        self.step = (self.step + 1).min(self.last_step());
    }

    fn step_back(&mut self) {
        self.playing = false;
        self.step = self.step.saturating_sub(1);
    }

    fn toggle_play(&mut self) {
        if self.playing {
            self.playing = false;
        } else {
            if self.step >= self.last_step() {
                self.step = 0;
            }
            (self.playing, self.owed) = (true, 0.0);
        }
    }

    fn again(&mut self) {
        self.step = 0;
        (self.playing, self.owed) = (true, 0.0);
    }

    /// The step the film is drawn at.
    fn film_at(&self) -> f64 {
        self.at.shown.first().copied().unwrap_or(0.0)
    }

    /// The step named: the one whose arithmetic is written out.
    fn named(&self) -> usize {
        (self.film_at().round().max(0.0) as usize).min(self.last_step())
    }

    /// Measure the chart again, for the inputs as drawn: the curve glides to the new
    /// measurement, or is there at once (`snap`).
    fn measure(&mut self, snap: bool) {
        let fm = self.fm();
        let at = self.drawn().to_vec();
        let i = self.chosen.min(fm.inputs().saturating_sub(1));
        let s = slopes(fm, &values(fm, &at));
        let curve = valley(fm, &at, i, s[i]);
        let c3 = bend(&self.compiled[self.formula], &at, i).abs();
        let bend = [if c3 > 1e-9 { c3.log10() } else { BEND_NONE }];
        if snap {
            self.measured.curve = Follow::at(&curve);
            self.measured.bend = Follow::at(&bend);
        } else {
            self.measured.curve.aim(&curve);
            self.measured.bend.aim(&bend);
        }
        (self.measured.at, self.measured.chosen, self.measured.since) = (at, i, 0.0);
    }

    /// Whether the chart is for other inputs than those drawn, or another input.
    fn stale(&self) -> bool {
        self.measured.at != self.values.shown || self.measured.chosen != self.chosen
    }

    /// The chart's dot, where it should be: at h, on the error measured there.
    fn dot_wanted(&self) -> Vec<f64> {
        let y = self.measured.curve.to.get(self.hk).copied().unwrap_or(ERR_TOP);
        vec![log_h(self.hk), y]
    }

    /// Move everything on: `dt` seconds of the scene's time (the film, the walk) and
    /// `real_dt` of the clock's (the glides, so that a step asked for while the
    /// scene's time stands still is still seen through).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        if self.playing {
            self.owed = (self.owed + self.rate.max(0.0) * dt.max(0.0)).min(3.0);
            while self.owed >= 1.0 && self.step < self.last_step() {
                self.step += 1;
                self.owed -= 1.0;
            }
            if self.step >= self.last_step() {
                (self.playing, self.owed) = (false, 0.0);
            }
        }
        if let Some(left) = self.walk {
            let left = left - dt.max(0.0);
            if left <= 0.0 {
                self.hk = self.hk.saturating_sub(10).max(self.walk_to);
                self.walk = (self.hk > self.walk_to).then_some(left + WALK_EVERY);
            } else {
                self.walk = Some(left);
            }
        }
        let real = real_dt.max(0.0) as f64;
        self.at.aim(&[self.step as f64]);
        self.at.step(real);
        self.values.step(real);
        // the chart follows the inputs, but no more often than every REFRESH seconds
        // (another input chosen, at once)
        self.measured.since += real_dt.max(0.0);
        if self.stale() && (self.measured.since >= REFRESH || self.measured.chosen != self.chosen) {
            self.measure(false);
        }
        self.measured.curve.step(real);
        self.measured.bend.step(real);
        self.dot.aim(&self.dot_wanted());
        self.dot.step(real);
        if self.old.is_some() {
            self.fade += real_dt.max(0.0) / FADE;
            if self.fade >= 1.0 || !self.fade.is_finite() {
                (self.fade, self.old) = (1.0, None);
            }
        }
    }

    fn moving(&self) -> bool {
        self.playing || self.at.to.first() != Some(&(self.step as f64)) || !self.at.arrived() || !self.values.arrived() || !self.dot.arrived() || self.stale() || !self.measured.curve.arrived() || !self.measured.bend.arrived() || self.walk.is_some() || self.old.is_some()
    }
}

/// A sixth of the third derivative of a formula along input `i` at `at`: its Taylor
/// coefficient of the third order, from the library's series (in f32, plenty for a
/// line drawn through the chart). 0 where it is not a number.
fn bend(e: &Expr, at: &[f64], i: usize) -> f64 {
    let vars: Vec<f32> = at.iter().map(|v| *v as f32).collect();
    let c3 = e.taylor(&vars, i).c[3] as f64;
    if c3.is_finite() {
        c3
    } else {
        0.0
    }
}

/// Text broken into lines no wider than `width` pixels, its top-left corner at `at`
/// (pixels): drawn if it is no taller than `room`. Returns how tall it is (0 if it was
/// not drawn).
fn wrap_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color, room: f32) -> f32 {
    let ppu = pixels_per_unit(f).max(1e-6);
    let Some(world) = f.cam.pick_ground(at * f.px) else { return 0.0 };
    let units = width / ppu;
    let h = wrapped_height(f, units, text, size);
    if h > room {
        return 0.0;
    }
    wrapped(f, world.truncate(), units, text, size, color);
    h
}

fn smoothstep(t: f32) -> f32 {
    let t = t.max(0.0).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Scene for Autodiff {
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
        Some("the dots in the input boxes: drag    buttons and input boxes: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A computer finds a formula's derivatives the way it finds its value: a small step at a time. Drawn as a graph, each box is one step (a sum, a product, a sine) and each arrow carries a value forward. Each arrow also has a slope of its own, the number on it: how much the box it enters changes per unit change of the box it leaves. One sweep forwards gives the value; one sweep backwards gives the slope of the output $f$ with respect to every box at once. That is all backpropagation is: a network is one large formula, its loss the output and its weights among the inputs.");

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.label, t.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The formula");
        let labels: Vec<&str> = FORMULAS.iter().map(|fm| fm.label).collect();
        let mut k = self.formula;
        if ui::choice(ui, &labels, &mut k) {
            self.choose(k);
        }
        ui::note(ui, self.fm().note);
        let fm = self.fm();
        let mut v: Vec<f64> = self.values.to.clone();
        let mut changed = false;
        for (i, x) in v.iter_mut().enumerate().take(fm.inputs()) {
            let mut x32 = *x as f32;
            if ui::slider(ui, &format!("${}$", fm.nodes[i].name), &mut x32, -KNOB as f32..=KNOB as f32) {
                *x = hundredths(x32 as f64);
                changed = true;
            }
        }
        if changed {
            self.set_values(&v, true);
        }
        ui::note(ui, "The inputs, in hundredths (or drag the dots in their boxes). Each box is worked out once, however many arrows leave it.");

        ui::section(ui, "The sweeps");
        let mut m = (self.mode == Mode::Forward) as usize;
        if ui::choice(ui, &["backward mode", "forward mode"], &mut m) {
            self.set_mode(if m == 1 { Mode::Forward } else { Mode::Backward });
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button("◀ back").clicked() {
                self.step_back();
            }
            if ui.button("▶ step").clicked() {
                self.step_on();
            }
            if ui.button(if self.playing { "❚❚ pause" } else { "▶▶ play" }).clicked() {
                self.toggle_play();
            }
            if ui.button("↺ again").clicked() {
                self.again();
            }
        });
        ui::slider_log(ui, "steps a second", &mut self.rate, 0.5..=8.0);
        ui::note(ui, "Backward mode: one sweep forwards for the values, then one back for the slopes of all the boxes. Forward mode: one sweep for each input, carrying with each value how fast it changes as that input moves.");

        ui::section(ui, "Differences");
        let names: Vec<String> = (0..fm.inputs()).map(|i| format!("${}$", fm.nodes[i].name)).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut c = self.chosen;
        if ui::choice(ui, &names, &mut c) {
            self.chosen = c;
            self.walk = None;
        }
        let mut h = 10f32.powf(log_h(self.hk) as f32);
        if ui::slider_log(ui, "the step $h$", &mut h, 1e-15..=1.0) {
            self.hk = (((h as f64).log10() - H_FROM) * 10.0).round().max(0.0).min((H_POINTS - 1) as f64) as usize;
            self.walk = None;
        }
        ui::note(ui, r"The chart: for the input chosen here (or click its box), how far the difference $(f(\ldots + h) - f(\ldots - h))/2h$ is from the slope the sweeps found, for every step $h$ from $10^{-15}$ to 1. The dot is the $h$ set here; drag it along the chart.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (FORWARD, "cyan: each box's value, worked out forwards from the inputs; the dot in an input's box sets it"),
                (BACKWARD, r"pink: $\bar u$, the slope of the output $f$ with respect to the box $u$, how much $f$ changes per unit change of $u$ by every road from $u$ to $f$; worked out backwards from $\bar f = 1$. For an input it is $\partial f/\partial x$"),
                (TANGENT, r"blue, in forward mode: how fast each box changes as one input moves, $\dot u$; that input's own is 1 and the others' 0"),
                (LOCAL, "numbers on the arrows: the arrow's own slope, how much the box it enters changes per unit change of the box it leaves"),
                (pal::WHITE, "the box lit up is the one being worked out; its arithmetic is written under the graph. A white dot marks the input the chart is about"),
                (MEASURED, r"the chart: how far the difference is from the slope, for each step $h$, both scales powers of ten; dashed, the two causes: the bend of $f$ within the step, and rounding"),
                (WRONG, "red: the digits of the difference that are wrong"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Computation graph:", "a formula drawn as its steps: a box for each sum, product or function, and an arrow for each value one box hands to another.");
        ui::term(
            ui,
            "Slope of an arrow:",
            r"the derivative of the box it enters with respect to the box it leaves, any other arrow into that box held still: for $u + v$ it is 1; for $x\,y$ the other factor; for $\sin x$, $\cos x$; for $\tanh z$, $1 - \tanh^2 z$; for $\max(0, z)$, 1 or 0.",
        );
        ui::term(
            ui,
            "Backward sweep (reverse mode):",
            "from the output back to the inputs. Each box's slope is the sum, over the arrows leaving it, of the slope at the arrow's far end times the arrow's slope: the chain rule, once for each road from the box to the output.",
        );
        ui::term(
            ui,
            "Forward mode:",
            r"from the inputs to the output, for one input at a time: every box carries how fast it changes as that input moves, $\dot u$. The dual numbers of this site's formula boxes work so: each number carries its rate of change along.",
        );
        ui::term(ui, "Central difference:", r"$(f(x + h) - f(x - h))/2h$, the slope of the line through two points of $f$ a step $h$ either side. Too large an $h$ and $f$ bends within it; too small, and rounding swamps it.");
        ui::term(ui, "Rounding:", r"a computer keeps about 16 significant digits of each number (here, in double precision), so each value of $f$ is off by up to about $10^{-16}$ times its size; dividing the difference of two of them by $2h$ makes that an error of roughly $10^{-16}/h$ for values of $f$ near 1.");
        ui::term(ui, "Backpropagation:", "the backward sweep, run on the formula that takes a network's weights to its loss: it gives the slope of the loss with respect to every weight at once.");

        ui::section(ui, "Numbers");
        let fm = self.fm();
        let v = values(fm, &self.values.to);
        let s = slopes(fm, &v);
        ui::readout(ui, "$f$", format!("${}$", num(v[fm.last()], 6)));
        for i in 0..fm.inputs() {
            ui::readout_colored(ui, &format!("${}$", partial("f", fm.nodes[i].name, false)), format!("${}$", num(s[i], 6)), BACKWARD);
        }
        let n = fm.inputs();
        ui::readout(ui, "sweeps for every slope", format!("backward 2, forward {n}"));
        ui::see_also(
            ui,
            &[
                ("chain", "the chain rule: Jacobians multiply, forwards or backwards"),
                ("calc1", "what a derivative is"),
                ("backprop1", "the backward sweep in the smallest network"),
                ("backpropmat", "the same, a layer at a time, in matrices"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let covered = view.x < 160.0;
        let lay = Layout::of(f, view, f.top_inset);
        if !covered {
            // the picture is for its buttons and handles: no moving the view, and two
            // quick clicks are two steps
            f.handles.capture();
            f.handles.claim_clicks();
            in_order(f);
            self.buttons(f, &lay);
            let alpha = if self.old.is_some() { (2.0 * smoothstep(self.fade) - 1.0).max(0.0) } else { 1.0 };
            self.input(f, &lay, alpha);
        }
        self.advance(f.dt, f.real_dt);
        if covered {
            return;
        }
        self.heading(f, &lay);
        let fade = smoothstep(self.fade);
        if let Some(old) = &self.old {
            // the old formula fades out in the first half, the new one in in the second
            let alpha = (1.0 - 2.0 * fade).max(0.0);
            let events = film(&FORMULAS[old.formula], old.mode);
            self.graph(f, &lay, &FORMULAS[old.formula], &old.values, old.mode, &events, old.at, alpha, false);
        }
        let alpha = if self.old.is_some() { (2.0 * fade - 1.0).max(0.0) } else { 1.0 };
        let (fm, mode) = (self.fm(), self.mode);
        let drawn = self.drawn().to_vec();
        self.graph(f, &lay, fm, &drawn, mode, &self.events, self.film_at(), alpha, true);
        // (the words under the graph change at once: their colours are in their markup)
        self.sums(f, &lay);
        self.tally(f, &lay);
        self.chart(f, &lay, alpha);
    }
}

impl Autodiff {
    /// The buttons over the picture: the film, the modes and the formulas.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let size = lay.text;
        // (each button says the longest of its words that fits it)
        let fit = |f: &Frame, r: Rect, words: &[&'static str]| -> &'static str {
            let room = r.1.x - r.0.x - 6.0;
            words.iter().copied().find(|w| f.text_size(w, size).x <= room).or(words.last().copied()).unwrap_or("")
        };
        if let [back, step, play, again] = lay.film[..] {
            let words = [
                fit(f, back, &["◀"]),
                fit(f, step, &["▶ step", "step", "▶"]),
                if self.playing { fit(f, play, &["❚❚ pause", "pause", "❚❚"]) } else { fit(f, play, &["▶▶ play", "play", "▶▶"]) },
                fit(f, again, &["↺ again", "again", "↺"]),
            ];
            if plot::button(f, back, words[0], size, false) {
                self.step_back();
            }
            if plot::button(f, step, words[1], size, false) {
                self.step_on();
            }
            if plot::button(f, play, words[2], size, self.playing) {
                self.toggle_play();
            }
            if plot::button(f, again, words[3], size, false) {
                self.again();
            }
        }
        if let Some(at) = lay.mode_lead {
            f.text_px(at, "mode:", size, pal::DIM, Align2::LEFT_CENTER);
        }
        for (k, r) in lay.modes.iter().enumerate() {
            let mode = if k == 0 { Mode::Backward } else { Mode::Forward };
            let word = if k == 0 { fit(f, *r, &["backward", "back", "◀◀"]) } else { fit(f, *r, &["forward", "fwd", "▶▶"]) };
            if plot::button(f, *r, word, size, self.mode == mode) {
                self.set_mode(mode);
            }
        }
        if let Some(at) = lay.formula_lead {
            f.text_px(at, "formula:", size, pal::DIM, Align2::LEFT_CENTER);
        }
        for (k, r) in lay.formulas.iter().enumerate() {
            let fm = &FORMULAS[k];
            let word = if lay.narrow { fit(f, *r, &[fm.short, fm.tiny]) } else { fm.label };
            if plot::button(f, *r, word, size, k == self.formula) {
                self.choose(k);
            }
        }
    }

    /// The knobs in the input boxes, and a click on an input's box (it chooses the
    /// input the chart is about).
    fn input(&mut self, f: &mut Frame, lay: &Layout, alpha: f32) {
        let Some(area) = lay.graph else { return };
        let fm = self.fm();
        let Some(p) = place(fm, area, lay.narrow) else { return };
        let n = fm.inputs();
        let drawn = self.drawn().to_vec();
        for i in 0..n {
            let r = p.rects[i];
            let at = knob_at(&p, &r, drawn.get(i).copied().unwrap_or(0.0));
            // the track
            let (x0, x1) = (r.0.x + 10.0, r.1.x - 10.0);
            hud_line(f, Vec2::new(x0, at.y), Vec2::new(x1, at.y), 1.2, pal::DIM.a(0.7 * alpha));
            // (while a new formula fades in, its knobs fade in with it, and cannot be taken yet)
            if self.old.is_some() {
                plot::disc(f, at, 12.0, FORWARD.a(alpha));
                continue;
            }
            if let Some(to) = plot::screen_handle(f, 10 + i as u32, at, FORWARD) {
                let t = ((to.x - x0) / (x1 - x0).max(1.0)).max(0.0).min(1.0) as f64;
                let mut v = self.values.to.clone();
                if let Some(x) = v.get_mut(i) {
                    *x = hundredths(-KNOB + 2.0 * KNOB * t);
                }
                self.set_values(&v, true);
                self.chosen = i;
                self.walk = None;
            }
        }
        if f.clicked_canvas() && self.old.is_none() {
            if let Some(at) = plot::pointer(f) {
                if let Some(i) = (0..n).find(|&i| plot::within(at, p.rects[i], 0.0)) {
                    self.chosen = i;
                    self.walk = None;
                }
            }
        }
    }

    /// What the graph is, written above it.
    fn heading(&self, f: &mut Frame, lay: &Layout) {
        let Some((at, width)) = lay.heading else { return };
        let fm = self.fm();
        let across = lay.graph.and_then(|area| place(fm, area, lay.narrow)).is_none_or(|p| p.across);
        let long = format!("{}, a box for each step: the inputs {}, the output $f$ {}", fm.title, if across { "on the left" } else { "at the top" }, if across { "on the right" } else { "at the foot" });
        let text = if f.text_size(&long, lay.small).x <= width { long } else { format!("{}, a box for each step", fm.title) };
        f.text_px(at, text, lay.small + 0.5, pal::TEXT, Align2::LEFT_TOP);
    }

    /// A formula's graph: its boxes with their numbers, the arrows with theirs, as the
    /// film has it at `at`, at the strength `alpha`.
    #[allow(clippy::too_many_arguments)]
    fn graph(&self, f: &mut Frame, lay: &Layout, fm: &Formula, inputs: &[f64], mode: Mode, events: &[Event], at: f64, alpha: f32, live: bool) {
        let Some(area) = lay.graph else { return };
        let Some(p) = place(fm, area, lay.narrow) else { return };
        let v = values(fm, inputs);
        let s = slopes(fm, &v);
        let seen = seen(fm, events, at);
        let tg = tangents(fm, &v, seen.sweep);
        let n = fm.inputs();
        let named = (at.round().max(0.0) as usize).min(events.len().saturating_sub(1));
        let sweep_color = match events.get(named) {
            Some(Event::Seed | Event::Slope(_)) => BACKWARD,
            Some(Event::Sweep(_) | Event::Tangent(..)) => TANGENT,
            _ => FORWARD,
        };
        // the boxes' backing, and the light on the one being worked out
        for (i, r) in p.rects.iter().enumerate().take(fm.nodes.len()) {
            hud_rect_at(f, 0, r.0, r.1, Color::hex(0x111829).a(0.96 * alpha));
            hud_rect_at(f, 1, r.0, r.1, seen.lit_color[i].a(0.12 * seen.lit[i] * alpha));
        }
        // the arrows, each with its slope written on it once the box it enters is worked out
        for (j, node) in fm.nodes.iter().enumerate() {
            for slot in 0..arity(node.op) {
                let i = node.args[slot];
                let (a, b) = arrow_ends(p.rects[i], p.rects[j]);
                hud_arrow(f, a, b, 1.5, pal::DIM.a(0.85 * alpha));
                // (lit while its box is worked out: going forwards, the arrows in; backwards, those out)
                let lit = if mode == Mode::Backward && matches!(events.get(named), Some(Event::Slope(_))) { seen.out_of[i] } else { seen.into[j] };
                hud_line(f, a, b, 3.0, sweep_color.a(0.8 * lit * alpha));
                let known = seen.value[j];
                if known > 0.0 && (b - a).length() > 24.0 {
                    let mid = a + (b - a) * 0.5;
                    f.caption_px(mid, format!("${}$", number(local(fm, &v, j, slot), 3)), p.label, LOCAL.a(known * alpha), Align2::CENTER_CENTER);
                }
            }
        }
        // the boxes' edges, lit or not
        for (i, r) in p.rects.iter().enumerate().take(fm.nodes.len()) {
            hud_outline(f, r.0, r.1, 1.0, pal::DIM.a(0.55 * alpha));
            hud_outline(f, r.0 - Vec2::splat(2.0), r.1 + Vec2::splat(2.0), 2.2, seen.lit_color[i].a(seen.lit[i] * alpha));
        }
        // what each box says
        for (i, node) in fm.nodes.iter().enumerate().take(MOST) {
            let (min, max) = p.rects[i];
            let cx = (min.x + max.x) * 0.5;
            let line = |k: f32| min.y + 4.0 + p.line * (k + 0.5);
            let header = if node.op == Op::Input { format!("${}$", node.name) } else { format!("${} = {}$", node.name, node.does) };
            f.text_px(Vec2::new(cx, line(0.0)), header, p.text, pal::TEXT.a(alpha), Align2::CENTER_CENTER);
            // the value, and the slope (backward mode) or the tangent (forward mode)
            let value = number(v[i], places(fm, i));
            let (amount, color, said) = match mode {
                Mode::Backward => (seen.slope[i], BACKWARD, (number(s[i], 3), format!(r"\bar {}", node.name))),
                Mode::Forward => {
                    let t = if i < n { format!("{}", tg[i] as i64) } else { number(tg[i], 3) };
                    (seen.tangent[i], TANGENT, (t, format!(r"\dot {}", node.name)))
                }
            };
            if p.full {
                if seen.value[i] > 0.0 {
                    f.text_px(Vec2::new(cx, line(1.0)), format!("${value}$"), p.text, FORWARD.a(seen.value[i] * alpha), Align2::CENTER_CENTER);
                } else {
                    f.text_px(Vec2::new(cx, line(1.0)), "…", p.text, pal::FAINT.a(alpha), Align2::CENTER_CENTER);
                }
                if amount > 0.0 {
                    let full = format!("${} = {}$", said.1, said.0);
                    let text = if f.text_size(&full, p.text).x <= max.x - min.x - 6.0 { full } else { format!("${}$", said.0) };
                    f.text_px(Vec2::new(cx, line(2.0)), text, p.text, color.a(amount * alpha), Align2::CENTER_CENTER);
                } else {
                    f.text_px(Vec2::new(cx, line(2.0)), "…", p.text, pal::FAINT.a(alpha), Align2::CENTER_CENTER);
                }
            } else {
                // (the two numbers side by side, each in its colour; one not yet worked
                // out is a faint dot, so that the other keeps its place)
                let quarter = (max.x - min.x) * 0.25;
                for (k, (text, c, amount)) in [(value, FORWARD, seen.value[i]), (said.0, color, amount)].into_iter().enumerate() {
                    let at = Vec2::new(cx + if k == 0 { -quarter } else { quarter }, line(1.0));
                    if amount > 0.0 {
                        f.text_px(at, format!("${text}$"), p.text, c.a(amount * alpha), Align2::CENTER_CENTER);
                    } else {
                        f.text_px(at, "·", p.text, pal::FAINT.a(alpha), Align2::CENTER_CENTER);
                    }
                }
            }
            // the input the chart is about
            if live && i == self.chosen {
                plot::disc(f, Vec2::new(max.x - 7.0, min.y + 7.0), 6.0, pal::WHITE.a(alpha));
            }
        }
    }

    /// What the step named works out, written out under the graph.
    fn sums(&self, f: &mut Frame, lay: &Layout) {
        let Some((at, width)) = lay.sums else { return };
        let fm = self.fm();
        let event = self.events.get(self.named()).copied().unwrap_or(Event::Start);
        let t = told(fm, self.mode, event, self.drawn());
        let (head, big, small) = if lay.narrow { (10.5, 12.5, 10.5) } else { (12.0, 15.0, 12.0) };
        let mut y = at.y;
        // (a line too long for the room is drawn smaller, to fit)
        let fit = |f: &Frame, text: &str, size: f32| {
            let w = f.text_size(text, size).x;
            if w > width && w > 0.0 {
                (size * width / w).max(8.0)
            } else {
                size
            }
        };
        let s = fit(f, &t.head, head);
        f.text_px(Vec2::new(at.x, y), t.head.as_str(), s, pal::DIM, Align2::LEFT_TOP);
        y += s + 8.0;
        for line in [&t.rule, &t.numbers] {
            if line.is_empty() {
                continue;
            }
            let s = fit(f, line, big);
            let h = f.text_size(line, s).y;
            f.text_px(Vec2::new(at.x, y), line.as_str(), s, pal::TEXT, Align2::LEFT_TOP);
            y += h + 4.0;
        }
        // the notes, broken into lines, as many as there is room for above the count of sweeps
        let limit = lay.tally.map_or(f32::MAX, |(t, _)| t.y - 4.0);
        y += 2.0;
        for note in &t.notes {
            let h = wrap_px(f, Vec2::new(at.x, y), width, note, small, pal::DIM, limit - y);
            if h <= 0.0 {
                break;
            }
            y += h + 2.0;
        }
    }

    /// The count of sweeps, and the slopes of the inputs found so far.
    fn tally(&self, f: &mut Frame, lay: &Layout) {
        let Some((at, width)) = lay.tally else { return };
        let fm = self.fm();
        let n = fm.inputs();
        let seen = seen(fm, &self.events, self.film_at());
        let v = values(fm, self.drawn());
        let s = slopes(fm, &v);
        let size = if lay.narrow { 10.5 } else { 12.0 };
        let (head, other) = match self.mode {
            Mode::Backward => (
                format!("backward mode: all {n} slopes from 2 sweeps, 1 forward and 1 back"),
                if n == 1 { "forward mode would take 1: with one input it is the cheaper way".to_string() } else { format!("forward mode would take {n}, one for each input") },
            ),
            Mode::Forward => (
                format!("forward mode: 1 sweep for each input, {n} in all"),
                if n == 1 { "backward mode would take 2: with one input, forward is the cheaper way".to_string() } else { format!("backward mode would take 2 for all {n}") },
            ),
        };
        let total = if self.mode == Mode::Backward { 2 } else { n };
        let progress = format!("sweeps done: {} of {total}      slopes found: {} of {n}", seen.sweeps.min(total), seen.known.min(n));
        let mut y = at.y;
        let lines: Vec<(String, Color)> = if lay.narrow {
            let short = match self.mode {
                Mode::Backward => format!("backward: all {n} slopes from 2 sweeps (done: {}); forward mode would take {n}", seen.sweeps.min(total)),
                Mode::Forward => format!("forward: 1 sweep for each input (done: {} of {n}); backward mode would take 2", seen.sweeps.min(total)),
            };
            vec![(short, pal::TEXT)]
        } else {
            vec![(head, pal::TEXT), (progress, pal::DIM), (other, pal::DIM)]
        };
        for (text, color) in &lines {
            let w = f.text_size(text, size).x;
            let s = if w > width && w > 0.0 { (size * width / w).max(8.0) } else { size };
            f.text_px(Vec2::new(at.x, y), text.as_str(), s, *color, Align2::LEFT_TOP);
            y += s + 7.0;
        }
        // the slopes found so far, side by side
        let mut x = at.x;
        for i in 0..n {
            let amount = match self.mode {
                Mode::Backward => seen.slope[i],
                Mode::Forward => seen.found[i],
            };
            let text = format!("${} = {}$", partial("f", fm.nodes[i].name, false), number(s[i], 3));
            let w = f.text_size(&text, size).x;
            if x + w > at.x + width {
                x = at.x;
                y += size + 9.0;
            }
            if amount > 0.0 {
                f.text_px(Vec2::new(x, y), text, size, BACKWARD.a(amount), Align2::LEFT_TOP);
            }
            x += w + 16.0;
        }
    }

    /// The chart of the difference's error against the step h, and its numbers.
    fn chart(&mut self, f: &mut Frame, lay: &Layout, alpha: f32) {
        let Some(rect) = lay.chart else { return };
        let p = Plot::new(rect, [H_FROM, 0.0], [ERR_FLOOR, ERR_TOP]);
        if p.tiny() {
            return;
        }
        let fm = self.fm();
        // (everything here is for the inputs the chart was last measured for)
        let drawn = self.measured.at.clone();
        let v = values(fm, &drawn);
        let s = slopes(fm, &v);
        let i = self.measured.chosen.min(fm.inputs().saturating_sub(1));
        let name = fm.nodes[i].name;
        let curve = &self.measured.curve.shown;
        let small = lay.small;
        let (min, max) = rect;
        p.backing(f, 40.0, lay.below(), lay.above(), 10.0);
        let title = if lay.narrow {
            format!(r"error of the difference for $\partial f/\partial {name}$, against $h$")
        } else {
            format!(r"how far $(f({name} + h) - f({name} - h))/2h$ is from $\partial f/\partial {name}$, for each step $h$")
        };
        f.text_px(Vec2::new(min.x - 34.0, min.y - lay.above() + 4.0), title, small + 0.5, pal::TEXT, Align2::LEFT_TOP);
        // the powers of ten along both sides
        for e in (ERR_FLOOR as i32 + 1)..=0 {
            let y = p.px_y(e as f64);
            if e % 4 == 0 {
                plot::line_of(f, plot::AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::FAINT.a(0.5));
                let text = if e == 0 { "$1$".to_string() } else { format!("$10^{{{e}}}$") };
                f.text_px(Vec2::new(min.x - 5.0, y), text, small - 1.5, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
        for e in (H_FROM as i32)..=0 {
            let x = p.px_x(e as f64);
            plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM);
            if e % 3 == 0 {
                let text = if e == 0 { "$1$".to_string() } else { format!("$10^{{{e}}}$") };
                f.text_px(Vec2::new(x, max.y + 6.0), text, small - 1.5, pal::DIM, Align2::CENTER_TOP);
            }
        }
        plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        f.text_px(Vec2::new(max.x, max.y + 8.0 + small), "step $h$", small - 1.0, pal::DIM, Align2::RIGHT_TOP);
        // the two causes, dashed: rounding, about 10⁻¹⁶/h, and the bend, h² |f'''| / 6
        let rounding = [(H_FROM, -16.0 - H_FROM), (0.0, -16.0)];
        let pts: Vec<Vec2> = rounding.iter().map(|&(x, y)| p.at(x, y)).collect();
        for w in plot::clip(&pts, rect).windows(2) {
            plot::dashed_of(f, GUIDES, w[0], w[1], 1.2, pal::DIM.a(0.8));
        }
        // (where f has no bend along this input, the line is far below the chart, and cut away)
        let bend = self.measured.bend.shown.first().copied().unwrap_or(BEND_NONE);
        let line = |x: f64| bend + 2.0 * x;
        let pts = [p.at(H_FROM, line(H_FROM)), p.at(0.0, line(0.0))];
        for w in plot::clip(&pts, rect).windows(2) {
            plot::dashed_of(f, GUIDES, w[0], w[1], 1.2, pal::DIM.a(0.8));
        }
        if !lay.narrow {
            let key = r"dashed, the two causes: the bend of $f$ within the step, $h^2 |f'''| / 6$, and rounding, about $10^{-16}/h$";
            f.text_px(Vec2::new(min.x - 34.0, min.y - lay.above() + 23.0), key, small - 1.5, pal::DIM, Align2::LEFT_TOP);
        }
        // the measured error
        p.path(f, CURVE, (0..H_POINTS.min(curve.len())).map(|k| (log_h(k), curve[k])), 1.8, MEASURED.a(0.9 * alpha));
        // the step h: a line up the chart and the dot on the curve, which can be dragged
        let (dx, dy) = (self.dot.shown.first().copied().unwrap_or(-5.0), self.dot.shown.get(1).copied().unwrap_or(-10.0));
        p.vline(f, MARK, dx, 1.0, pal::DIM.a(0.6), true);
        if let Some((x, _)) = p.handle(f, 40, dx, dy, pal::WHITE.a(alpha)) {
            self.hk = ((x - H_FROM) * 10.0).round().max(0.0).min((H_POINTS - 1) as f64) as usize;
            self.walk = None;
        }
        // the numbers at h
        let Some((at, width)) = lay.readouts else { return };
        let h = 10f64.powf(log_h(self.hk));
        let d = difference(fm, &drawn, i, h);
        let err = (d - s[i]).abs();
        let h_text = if self.hk.is_multiple_of(10) { if self.hk == H_POINTS - 1 { "1".to_string() } else { format!("10^{{{}}}", log_h(self.hk).round() as i32) } } else { format!("{h:.1e}") };
        let (good, bad, slope) = digits(d, s[i]);
        let size = if lay.narrow { 10.5 } else { 12.0 };
        let lines = [
            format!("with $h = {h_text}$, the difference: ${}{}$", tint(pal::TEXT, &good), tint(WRONG, &bad)),
            format!("the sweeps: ${}$", tint(BACKWARD, &slope)),
            if err > 0.0 { format!("off by ${}$", sci(err)) } else { "off by nothing: exactly right".to_string() },
        ];
        let mut y = at.y;
        for line in &lines {
            let w = f.text_size(line, size).x;
            let s = if w > width && w > 0.0 { (size * width / w).max(8.0) } else { size };
            f.text_px(Vec2::new(at.x, y), line.as_str(), s, pal::DIM, Align2::LEFT_TOP);
            y += s + 7.0;
        }
    }
}

/// A number written with one figure and its power of ten: 3 × 10⁻¹¹ (markup).
fn sci(v: f64) -> String {
    if !(v.is_finite() && v > 0.0) {
        return num(v, 0);
    }
    let e = v.log10().floor() as i32;
    let m = (v / 10f64.powi(e)).round() as i32;
    let (m, e) = if m >= 10 { (1, e + 1) } else { (m, e) };
    if (-2..=2).contains(&e) {
        // (two figures, without the zeros at the end: 0.25, 0.5, 1.3)
        let text = num(v, (1 - e).max(0) as usize);
        if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            text
        }
    } else {
        format!(r"{m} \times 10^{{{e}}}")
    }
}

/// A difference and the slope it is after, to fifteen significant figures: the digits
/// of the difference that agree with the slope's, the rest of it, and the slope.
fn digits(d: f64, slope: f64) -> (String, String, String) {
    let big = d.abs().max(slope.abs()).max(1e-300);
    let decimals = (14 - big.log10().floor().max(0.0) as i32).max(0) as usize;
    let (a, b) = (num(d, decimals), num(slope, decimals));
    let same = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
    let split = a.char_indices().nth(same).map_or(a.len(), |(k, _)| k);
    (a[..split].to_string(), a[split..].to_string(), b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every graph is a formula: its arguments come before it, its inputs first.
    #[test]
    fn the_graphs_are_well_formed() {
        for fm in &FORMULAS {
            assert!(fm.nodes.len() <= MOST);
            assert_eq!(fm.start.len(), fm.inputs());
            assert_eq!(fm.vars.len(), fm.inputs());
            for (i, n) in fm.nodes.iter().enumerate() {
                assert_eq!(n.op == Op::Input, i < fm.inputs(), "{}: box {i}", fm.text);
                for slot in 0..arity(n.op) {
                    assert!(n.args[slot] < i, "{}: box {i}", fm.text);
                }
            }
            // and every box but the output feeds another
            for i in 0..fm.last() {
                assert!(!leaving(fm, i).is_empty(), "{}: box {i}", fm.text);
            }
        }
    }

    /// Random inputs for a formula, in the knobs' range, kept off the kink's corner by
    /// `clear` (the slope is not defined there).
    fn somewhere(fm: &Formula, rng: &mut Rng, clear: f64) -> Vec<f64> {
        loop {
            let x: Vec<f64> = (0..fm.inputs()).map(|_| rng.range(-KNOB as f32, KNOB as f32) as f64).collect();
            let v = values(fm, &x);
            let kinked = fm.nodes.iter().any(|n| n.op == Op::Relu && v[n.args[0]].abs() < clear);
            if !kinked {
                return x;
            }
        }
    }

    /// Each graph is the formula it is named for: its value is the library compiler's.
    #[test]
    fn the_graphs_compute_their_formulas() {
        let mut rng = Rng::new(7);
        for fm in &FORMULAS {
            let e = Expr::parse(fm.text, fm.vars).expect(fm.text);
            for _ in 0..200 {
                let x = somewhere(fm, &mut rng, 0.0);
                let x32: Vec<f32> = x.iter().map(|v| *v as f32).collect();
                let (mine, theirs) = (values(fm, &x)[fm.last()], e.eval(&x32) as f64);
                assert!((mine - theirs).abs() < 1e-5 * (1.0 + mine.abs()), "{} at {x:?}: {mine} against {theirs}", fm.text);
            }
        }
    }

    /// Every slope the backward sweep finds is the dual numbers' (forward mode in the
    /// library, in f32) and the central difference's, at random points.
    #[test]
    fn every_slope_is_the_dual_numbers_and_the_differences() {
        let mut rng = Rng::new(11);
        for fm in &FORMULAS {
            let e = Expr::parse(fm.text, fm.vars).expect(fm.text);
            let n = fm.inputs();
            for _ in 0..200 {
                let x = somewhere(fm, &mut rng, 1e-3);
                let x32: Vec<f32> = x.iter().map(|v| *v as f32).collect();
                let s = slopes(fm, &values(fm, &x));
                for i in 0..n {
                    let (_, dual, _) = e.grad2(&x32, i, (i + 1) % n);
                    assert!((s[i] - dual as f64).abs() < 2e-5 * (1.0 + s[i].abs()), "{}, input {i} at {x:?}: {} against the dual numbers' {dual}", fm.text, s[i]);
                    let d = difference(fm, &x, i, 1e-5);
                    assert!((s[i] - d).abs() < 1e-8 * (1.0 + s[i].abs()), "{}, input {i} at {x:?}: {} against the difference {d}", fm.text, s[i]);
                }
            }
        }
    }

    /// Every box's slope, not only the inputs': nudge one box's value and carry the
    /// change on to the output, the boxes it feeds worked out again from it.
    #[test]
    fn every_box_slope_is_a_difference_of_the_output() {
        let mut rng = Rng::new(5);
        for fm in &FORMULAS {
            for _ in 0..50 {
                let x = somewhere(fm, &mut rng, 1e-3);
                let v = values(fm, &x);
                let s = slopes(fm, &v);
                for i in 0..fm.nodes.len() {
                    let out = |h: f64| {
                        let mut w = v;
                        w[i] += h;
                        for (j, n) in fm.nodes.iter().enumerate().skip(i + 1) {
                            let (a, b) = (w[n.args[0]], w[n.args[1]]);
                            w[j] = match n.op {
                                Op::Input => w[j],
                                Op::Add => a + b,
                                Op::Mul => a * b,
                                Op::Sin => a.sin(),
                                Op::Tanh => a.tanh(),
                                Op::Relu => a.max(0.0),
                            };
                        }
                        w[fm.last()]
                    };
                    let d = (out(1e-6) - out(-1e-6)) / 2e-6;
                    assert!((s[i] - d).abs() < 1e-6 * (1.0 + s[i].abs()), "{}, box {i} at {x:?}: {} against {d}", fm.text, s[i]);
                }
                // and each arrow's own slope is a difference of the box it enters
                for (j, n) in fm.nodes.iter().enumerate() {
                    for slot in 0..arity(n.op) {
                        let at = |h: f64| {
                            let mut w = v;
                            w[n.args[slot]] += h;
                            values_of_box(n, &w)
                        };
                        let d = (at(1e-6) - at(-1e-6)) / 2e-6;
                        let l = local(fm, &v, j, slot);
                        assert!((l - d).abs() < 1e-6 * (1.0 + l.abs()), "{}, arrow into box {j}: {l} against {d}", fm.text);
                    }
                }
            }
        }
    }

    /// What a box makes of the values `w` (for the test above).
    fn values_of_box(n: &Node, w: &[f64; MOST]) -> f64 {
        let (a, b) = (w[n.args[0]], w[n.args[1]]);
        match n.op {
            Op::Input => a,
            Op::Add => a + b,
            Op::Mul => a * b,
            Op::Sin => a.sin(),
            Op::Tanh => a.tanh(),
            Op::Relu => a.max(0.0),
        }
    }

    /// Forward mode, one sweep for each input, ends with the same slopes as the one
    /// sweep back, to rounding.
    #[test]
    fn forward_mode_finds_the_same_slopes() {
        let mut rng = Rng::new(3);
        for fm in &FORMULAS {
            for _ in 0..100 {
                let x = somewhere(fm, &mut rng, 0.0);
                let v = values(fm, &x);
                let s = slopes(fm, &v);
                for i in 0..fm.inputs() {
                    let t = tangents(fm, &v, i)[fm.last()];
                    assert!((t - s[i]).abs() <= 1e-14 * (1.0 + s[i].abs()), "{}: {t} and {}", fm.text, s[i]);
                }
            }
        }
    }

    /// The films: backward mode is the values, the output's 1, and every box's slope,
    /// two sweeps; forward mode is a sweep for each input.
    #[test]
    fn the_sweeps_are_counted_right() {
        for fm in &FORMULAS {
            let (n, m) = (fm.inputs(), fm.nodes.len() - fm.inputs());
            let back = film(fm, Mode::Backward);
            assert_eq!(back.len(), 1 + m + 1 + fm.last());
            let end = seen(fm, &back, (back.len() - 1) as f64);
            assert!(end.sweeps == 2 && end.known == n, "{}: {} sweeps, {} known", fm.text, end.sweeps, end.known);
            // the forward sweep alone: one sweep, no slopes
            let seed = back.iter().position(|e| *e == Event::Seed).unwrap();
            let half = seen(fm, &back, (seed - 1) as f64);
            assert!(half.sweeps == 1 && half.known == 0);
            let fwd = film(fm, Mode::Forward);
            assert_eq!(fwd.iter().filter(|e| matches!(e, Event::Sweep(_))).count(), n);
            assert_eq!(fwd.len(), 1 + n * (m + 1));
            let end = seen(fm, &fwd, (fwd.len() - 1) as f64);
            assert!(end.sweeps == n && end.known == n, "{}: {} sweeps, {} known", fm.text, end.sweeps, end.known);
            // after the first sweep, one slope
            let first = fwd.iter().position(|e| *e == Event::Tangent(0, fm.last())).unwrap();
            let one = seen(fm, &fwd, first as f64);
            assert!(one.sweeps == 1 && one.known == 1 && one.found[0] == 1.0 && one.found[1..n].iter().all(|f| *f == 0.0));
        }
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// it comes to rest.
    fn trial(k: usize) -> Autodiff {
        let mut scene = Autodiff::new();
        // (from somewhere else, so that nothing is as the experiment wants it by chance)
        scene.set_up((k + 2) % TRIALS.len());
        for _ in 0..30 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        scene.set_up(k);
        for _ in 0..60 * 60 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if !scene.moving() {
                break;
            }
        }
        assert!(!scene.moving() && !scene.playing, "experiment {k} does not come to rest");
        scene
    }

    /// The numbers of the experiment's formula, as the scene has them.
    fn worked(scene: &Autodiff) -> ([f64; MOST], [f64; MOST]) {
        let v = values(scene.fm(), scene.drawn());
        let s = slopes(scene.fm(), &v);
        (v, s)
    }

    #[test]
    fn forwards_then_backwards_does_what_its_caption_says() {
        let scene = trial(0);
        assert_eq!((scene.formula, scene.mode, scene.drawn()), (0, Mode::Backward, &[2.0, 1.5][..]));
        assert_eq!(scene.step, scene.last_step());
        let (v, s) = worked(&scene);
        assert_eq!(number(v[4], 3), "3.909");
        assert_eq!((number(s[1], 3), number(s[0], 3)), ("2.000".to_string(), "1.084".to_string()));
        // the last two steps are the inputs, y and then x
        let n = scene.events.len();
        assert_eq!(&scene.events[n - 2..], &[Event::Slope(1), Event::Slope(0)]);
        // and the opening picture is that last step: x's two roads
        let open = Autodiff::new();
        assert!(open.formula == 0 && open.named() == open.last_step() && !open.moving());
        let t = told(open.fm(), open.mode, open.events[open.named()], open.drawn());
        assert!(t.numbers.contains("1.500") && t.numbers.contains("(−0.416)") && t.numbers.contains("1.084") && !t.numbers.contains("approx"), "{}", t.numbers);
    }

    #[test]
    fn a_variable_used_twice_does_what_its_caption_says() {
        let scene = trial(1);
        assert_eq!((scene.formula, scene.drawn()), (1, &[0.8][..]));
        assert_eq!(scene.events[scene.named()], Event::Slope(0));
        let (v, s) = worked(&scene);
        let fm = scene.fm();
        assert_eq!((number(v[1], 3), number(local(fm, &v, 1, 0), 3), number(s[1], 3)), ("0.717".into(), "0.697".into(), "0.800".into()));
        assert_eq!(number(s[0], 3), "1.275");
        // the two roads, written as the caption writes them, add up as written
        assert_eq!(leaving(fm, 0), vec![(1, 0), (2, 0)]);
        assert_eq!(equals(0.800 * 0.697 + 0.717, s[0], 3), "=");
        let t = told(fm, scene.mode, Event::Slope(0), scene.drawn());
        assert!(t.numbers.contains("0.800") && t.numbers.contains("0.697") && t.numbers.contains("0.717") && !t.numbers.contains("approx"), "{}", t.numbers);
        // the product rule
        let x = 0.8f64;
        assert!((s[0] - (x * x.cos() + x.sin())).abs() < 1e-15);
    }

    #[test]
    fn one_sweep_back_does_what_its_caption_says() {
        let scene = trial(2);
        assert_eq!((scene.formula, scene.mode), (2, Mode::Forward));
        assert_eq!(scene.step, scene.last_step());
        let fm = scene.fm();
        let (v, s) = worked(&scene);
        let said: Vec<String> = (0..5).map(|i| number(s[i], 3)).collect();
        assert_eq!(said, ["0.539", "0.287", "0.180", "−0.216", "0.359"]);
        // five sweeps forward, each with one slope, the same as backward's
        let seen = seen(fm, &scene.events, scene.film_at());
        assert!(seen.sweeps == 5 && seen.known == 5);
        for i in 0..5 {
            assert_eq!(number(tangents(fm, &v, i)[fm.last()], 3), said[i]);
        }
    }

    /// The error of the difference for ∂f/∂x of the first formula at (2, 1.5).
    fn error_at(h: f64) -> f64 {
        let fm = &FORMULAS[0];
        let x = [2.0, 1.5];
        let s = slopes(fm, &values(fm, &x));
        (difference(fm, &x, 0, h) - s[0]).abs()
    }

    #[test]
    fn differences_do_what_their_caption_says() {
        let scene = trial(3);
        assert_eq!((scene.formula, scene.chosen, scene.hk, scene.drawn()), (0, 0, H_E12, &[2.0, 1.5][..]));
        assert!((log_h(H_TENTH) + 1.0).abs() < 1e-12 && (log_h(H_E5) + 5.0).abs() < 1e-12 && (log_h(H_E12) + 12.0).abs() < 1e-12);
        // too large: off by 7 × 10⁻⁴, a hundredfold less for each tenth of h
        assert_eq!(sci(error_at(0.1)), r"7 \times 10^{-4}");
        for h in [0.1, 0.01, 0.001] {
            let ratio = error_at(h / 10.0) / error_at(h);
            assert!((0.0099..0.0101).contains(&ratio), "h = {h}: {ratio}");
        }
        // and as the bend says: h² |f'''| / 6
        let c3 = bend(&scene.compiled[0], &[2.0, 1.5], 0).abs();
        assert!((error_at(0.01) / (c3 * 1e-4) - 1.0).abs() < 1e-3, "{} against {}", error_at(0.01), c3 * 1e-4);
        // near the floor: 3 × 10⁻¹¹ at 10⁻⁵, ten digits right; the floor of the chart about 10⁻¹¹
        let s = slopes(&FORMULAS[0], &values(&FORMULAS[0], &[2.0, 1.5]))[0];
        assert_eq!(sci(error_at(1e-5)), r"3 \times 10^{-11}");
        let digits = -(error_at(1e-5) / s).log10();
        assert!((10.0..11.0).contains(&digits), "{digits}");
        let curve = valley(&FORMULAS[0], &[2.0, 1.5], 0, s);
        let floor = curve.iter().fold(f64::MAX, |a, b| a.min(*b));
        assert!((-12.0..-10.5).contains(&floor), "the floor of the valley: 10^{floor}");
        // too small: about 10⁻⁴ by 10⁻¹²
        let small = error_at(1e-12);
        assert!((2e-5..5e-4).contains(&small), "{small}");
        // the walk passed through every decade and stopped at 10⁻¹²
        assert!(scene.walk.is_none());
    }

    #[test]
    fn a_kink_does_what_its_caption_says() {
        let scene = trial(4);
        let fm = scene.fm();
        assert_eq!((scene.formula, scene.chosen, scene.drawn()), (3, 2, &[1.0, 0.5, -0.5][..]));
        let (v, s) = worked(&scene);
        // z exactly 0, the corner; every slope behind it 0
        assert_eq!(v[4], 0.0);
        assert!(s[..5].iter().all(|s| *s == 0.0), "{s:?}");
        // as the library's dual numbers have it too
        let e = Expr::parse(fm.text, fm.vars).unwrap();
        for i in 0..3 {
            assert_eq!(e.grad2(&[1.0, 0.5, -0.5], i, i).1, 0.0);
        }
        // the difference straddles the corner: about 0.5 for b at every h on the chart
        let x = [1.0, 0.5, -0.5];
        for k in 0..H_POINTS {
            let h = 10f64.powf(log_h(k));
            let d = difference(fm, &x, 2, h);
            let near = if h >= 1e-12 { 1e-3 } else { 0.02 };
            assert!((d - 0.5).abs() < near, "h = {h}: {d}");
        }
        // the film ends on the inputs, having shown the corner on the way
        assert!(scene.events.contains(&Event::Slope(4)));
        let t = told(fm, Mode::Backward, Event::Slope(4), &x);
        assert!(t.notes.iter().any(|n| n.contains("exactly 0, the corner")), "{:?}", t.notes);
        // off the corner, the two agree for every h smaller than the distance to it
        let off = [1.0, 0.5, -0.49];
        let s = slopes(fm, &values(fm, &off));
        assert_eq!(s[2], 1.0);
        for k in 0..H_POINTS {
            let h = 10f64.powf(log_h(k));
            let d = difference(fm, &off, 2, h);
            if h < 0.009 {
                let near = if h >= 1e-12 { 1e-3 } else { 0.05 };
                assert!((d - 1.0).abs() < near, "h = {h}: {d}");
            } else if h > 0.011 {
                assert!((d - 1.0).abs() > 0.05, "h = {h}: {d}");
            }
        }
        // and on the other side, where the slope is 0
        let below = [1.0, 0.5, -0.51];
        assert_eq!(slopes(fm, &values(fm, &below))[2], 0.0);
        for k in 0..H_POINTS {
            let h = 10f64.powf(log_h(k));
            let d = difference(fm, &below, 2, h);
            if h < 0.009 {
                assert_eq!(d, 0.0, "h = {h}");
            } else if h > 0.011 {
                assert!(d > 0.05, "h = {h}: {d}");
            }
        }
    }

    /// The scene opens at rest; a step asked for glides there and comes exactly to rest,
    /// at any display's rate; playing goes on to the end and stops.
    #[test]
    fn it_opens_at_rest_and_a_step_glides() {
        for hz in [30.0f32, 60.0, 144.0, 240.0] {
            let mut scene = Autodiff::new();
            assert!(!scene.moving());
            scene.step_back();
            let (mut frames, mut last) = (0, scene.film_at());
            while scene.moving() {
                scene.advance(1.0 / hz, 1.0 / hz);
                assert!(scene.film_at() <= last);
                last = scene.film_at();
                frames += 1;
                assert!((frames as f32) / hz < 1.0, "still moving at {hz} Hz");
            }
            assert_eq!(scene.film_at(), (scene.last_step() - 1) as f64);
            scene.again();
            let mut frames = 0;
            while scene.moving() {
                scene.advance(1.0 / hz, 1.0 / hz);
                frames += 1;
                assert!(frames < 60 * 60);
            }
            assert_eq!(scene.film_at(), scene.last_step() as f64);
        }
    }

    /// Whatever is pressed, in whatever order, with frames of any length, everything
    /// drawn from stays finite and in range.
    #[test]
    fn any_input_leaves_everything_finite() {
        let mut rng = Rng::new(9);
        let mut scene = Autodiff::new();
        for k in 0..4000 {
            match rng.below(12) {
                0 => scene.step_on(),
                1 => scene.step_back(),
                2 => scene.toggle_play(),
                3 => scene.again(),
                4 => scene.choose(rng.below(FORMULAS.len() + 2)),
                5 => scene.set_mode(if rng.bool(0.5) { Mode::Forward } else { Mode::Backward }),
                6 => scene.set_up(rng.below(TRIALS.len())),
                7 => {
                    let v: Vec<f64> = (0..6).map(|_| rng.range(-5.0, 5.0) as f64).collect();
                    scene.set_values(&v, rng.bool(0.5));
                }
                8 => scene.hk = rng.below(H_POINTS),
                9 => scene.chosen = rng.below(scene.fm().inputs()),
                _ => {}
            }
            let dt = [0.0, 1.0 / 240.0, 1.0 / 60.0, 0.5, 5.0][rng.below(5)];
            scene.advance(dt, dt);
            assert!(scene.step <= scene.last_step() && scene.film_at().is_finite(), "{k}");
            assert!(scene.drawn().iter().chain(&scene.dot.shown).all(|v| v.is_finite()), "{k}");
            assert!(scene.drawn().iter().all(|v| v.abs() <= KNOB), "{k}");
            let fm = scene.fm();
            let (v, s) = worked(&scene);
            assert!(v.iter().chain(&s).all(|x| x.is_finite()), "{k}");
            assert!(valley(fm, scene.drawn(), scene.chosen, s[scene.chosen]).iter().all(|e| e.is_finite() && *e >= ERR_FLOOR && *e <= ERR_TOP));
            let seen = seen(fm, &scene.events, scene.film_at());
            assert!(seen.value.iter().chain(&seen.slope).chain(&seen.tangent).chain(&seen.lit).all(|a| (0.0..=1.0).contains(a)), "{k}");
            let t = told(fm, scene.mode, scene.events[scene.named()], scene.drawn());
            assert!(!t.head.is_empty() && !t.rule.is_empty());
            assert!(bend(&scene.compiled[scene.formula], scene.drawn(), scene.chosen).is_finite());
        }
    }

    /// Every experiment comes to rest, and the same way whatever was done before.
    #[test]
    fn the_experiments_come_to_rest_the_same_from_anywhere() {
        for k in 0..TRIALS.len() {
            let a = trial(k);
            let mut b = Autodiff::new();
            b.choose(2);
            b.set_mode(Mode::Forward);
            b.set_values(&[1.0, -1.0, 2.0, 0.3, 0.1], true);
            b.set_up(k);
            for _ in 0..60 * 60 {
                b.advance(1.0 / 60.0, 1.0 / 60.0);
            }
            assert_eq!((a.formula, a.mode, a.step, a.chosen, a.hk), (b.formula, b.mode, b.step, b.chosen, b.hk), "experiment {k}");
            assert_eq!(a.drawn(), b.drawn(), "experiment {k}");
        }
    }

    /// The numbers as the scene writes them.
    #[test]
    fn numbers_are_written_plainly() {
        assert_eq!(number(1.0838531634, 3), "1.084");
        assert_eq!(number(-0.41614683, 3), "−0.416");
        assert_eq!(number(0.0, 3), "0.000");
        // a number just off the corner is not written as 0
        assert_eq!(number(3.5e-18, 3), "3.5e-18");
        assert_eq!(number(-2.0e-5, 3), "-2.0e-5");
        assert_eq!(number(f64::NAN, 3), "?");
        assert_eq!(factor("−0.416".into()), "(−0.416)");
        assert_eq!(sci(6.93e-4), r"7 \times 10^{-4}");
        assert_eq!(sci(3.3e-11), r"3 \times 10^{-11}");
        assert_eq!(sci(9.7e-5), r"1 \times 10^{-4}");
        assert_eq!(sci(0.5), "0.5");
        assert_eq!(sci(0.25), "0.25");
        assert_eq!(sci(0.0123), "0.012");
        assert_eq!(sci(1.0), "1");
        // rounding that makes the written numbers miss the written result says so
        assert_eq!(equals(1.5 - 0.416, 1.0838531634, 3), "=");
        assert_eq!(equals(0.359 * 0.6, 0.21552078969, 3), r"\approx");
        // the digits of a difference that agree with the slope
        let (good, bad, slope) = digits(1.0838531634860, 1.0838531634529);
        assert_eq!((good.as_str(), bad.as_str(), slope.as_str()), ("1.0838531634", "8600", "1.08385316345290"));
        let (good, bad, _) = digits(0.49999999999994, 0.0);
        assert!(good == "0." && bad.starts_with('4'));
        assert_eq!(hundredths(0.287), 0.29);
        assert_eq!(hundredths(9.0), KNOB);
    }
}

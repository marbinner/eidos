//! Backpropagation in the smallest network: two inputs, two tanh units, one output and
//! nine weights, every number on the screen. A film steps forward a layer at a time and
//! then back, writing each step's arithmetic with its numbers; a nudge checks a slope
//! against the loss itself; a step of descent uses them all; and four examples train it.

use super::*;
use crate::clock::Pace;
use crate::scenes::stats::plot::{self, Plot, Rect};

pub const DEF: SceneDef = SceneDef {
    id: "backprop1",
    title: "Backpropagation in the smallest network",
    group: GROUP,
    blurb: "Two layers, a handful of weights and every number shown: the forward pass, the loss, and the gradient handed back step by step.",
    make: |_| Box::new(Backprop1::create()),
};

// ----- the network ---------------------------------------------------------------------

/// The nine numbers to learn, in this order: unit 1's two weights and bias, unit 2's,
/// then the output's two weights and its bias.
type Weights = [f64; 9];

const W11: usize = 0;
const W12: usize = 1;
const B1: usize = 2;
const W21: usize = 3;
const W22: usize = 4;
const B2: usize = 5;
const V1: usize = 6;
const V2: usize = 7;
const C: usize = 8;

/// Their names, as the picture and the arithmetic write them.
const NAMES: [&str; 9] = [r"w_{11}", r"w_{12}", r"b_1", r"w_{21}", r"w_{22}", r"b_2", r"v_1", r"v_2", r"c"];

/// The weights the scene starts from, and the one example it starts with.
const START: Weights = [0.5, -0.3, 0.1, -0.4, 0.8, -0.2, 0.7, -0.6, 0.1];
const X0: [f64; 2] = [1.0, 0.5];
const Y0: f64 = 1.0;

/// "One or the other, not both": the four corners, the target +1 where exactly one
/// input is +1 and −1 where both or neither are.
const FOUR: [([f64; 2], f64); 4] = [([-1.0, -1.0], -1.0), ([-1.0, 1.0], 1.0), ([1.0, -1.0], 1.0), ([1.0, 1.0], -1.0)];

/// How far a weight may be dragged or typed, and how far descent may take one (beyond
/// this a step that runs away is held, rather than carried to infinity).
const DRAG_MOST: f64 = 4.0;
const HELD_AT: f64 = 50.0;
/// The range of the inputs and of the target.
const INPUT_MOST: f64 = 2.0;

/// One example's way through the network, forward and back.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Pass {
    x: [f64; 2],
    y: f64,
    /// Forward: each unit's sum and its tanh, the output, the loss ½(ŷ − y)².
    z: [f64; 2],
    a: [f64; 2],
    out: f64,
    loss: f64,
    /// Back: the loss's slope with respect to the output (δ), to each unit's tanh (what
    /// reaches it along its wire), the tanh's own slope 1 − a², and to each unit's sum (δⱼ).
    d: f64,
    da: [f64; 2],
    gate: [f64; 2],
    dz: [f64; 2],
    /// And with respect to every weight: what came in on its wire times what came back.
    grad: Weights,
}

fn pass(w: &Weights, x: [f64; 2], y: f64) -> Pass {
    let mut p = Pass { x, y, ..Default::default() };
    for j in 0..2 {
        let k = 3 * j;
        p.z[j] = w[k] * x[0] + w[k + 1] * x[1] + w[k + 2];
        p.a[j] = p.z[j].tanh();
    }
    p.out = w[V1] * p.a[0] + w[V2] * p.a[1] + w[C];
    p.loss = 0.5 * (p.out - y) * (p.out - y);
    p.d = p.out - y;
    p.grad[V1] = p.a[0] * p.d;
    p.grad[V2] = p.a[1] * p.d;
    p.grad[C] = p.d;
    for j in 0..2 {
        let k = 3 * j;
        p.da[j] = w[V1 + j] * p.d;
        p.gate[j] = 1.0 - p.a[j] * p.a[j];
        p.dz[j] = p.gate[j] * p.da[j];
        p.grad[k] = x[0] * p.dz[j];
        p.grad[k + 1] = x[1] * p.dz[j];
        p.grad[k + 2] = p.dz[j];
    }
    p
}

/// The four examples' passes, their mean loss and the mean of their slopes.
fn passes_of_four(w: &Weights) -> ([Pass; 4], f64, Weights) {
    let passes = FOUR.map(|(x, y)| pass(w, x, y));
    let loss = passes.iter().map(|p| p.loss).sum::<f64>() / 4.0;
    let mut grad = [0.0; 9];
    for (i, g) in grad.iter_mut().enumerate() {
        *g = passes.iter().map(|p| p.grad[i]).sum::<f64>() / 4.0;
    }
    (passes, loss, grad)
}

/// What is trained on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    One,
    Four,
}

/// The loss and its slopes for what is trained on.
fn loss_and_grad(mode: Mode, w: &Weights, x: [f64; 2], y: f64) -> (f64, Weights) {
    match mode {
        Mode::One => {
            let p = pass(w, x, y);
            (p.loss, p.grad)
        }
        Mode::Four => {
            let (_, loss, grad) = passes_of_four(w);
            (loss, grad)
        }
    }
}

fn loss_of(mode: Mode, w: &Weights, x: [f64; 2], y: f64) -> f64 {
    match mode {
        Mode::One => pass(w, x, y).loss,
        Mode::Four => FOUR.iter().map(|(x, y)| pass(w, *x, *y).loss).sum::<f64>() / 4.0,
    }
}

/// One step downhill: every weight minus the rate times its slope (held within
/// ±`HELD_AT`; a step that is not a number is not taken).
fn descend(w: &Weights, grad: &Weights, rate: f64) -> Weights {
    let mut next = *w;
    for (n, g) in next.iter_mut().zip(grad) {
        let v = *n - rate * g;
        if v.is_finite() {
            *n = v.max(-HELD_AT).min(HELD_AT);
        }
    }
    next
}

/// The two hidden units made alike: unit 2's weights are unit 1's.
fn alike(w: &Weights) -> Weights {
    let mut same = *w;
    same[W21] = w[W11];
    same[W22] = w[W12];
    same[B2] = w[B1];
    same[V2] = w[V1];
    same
}

// ----- numbers as the scene writes them ------------------------------------------------

/// A number to three significant figures, with a true minus sign; one smaller than a
/// thousandth as a power of ten (which a formula sets as 1.39 × 10⁻³).
fn sig(v: f64) -> String {
    figures(v, 3)
}

/// The same to `n` significant figures.
fn figures(v: f64, n: i32) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    let a = v.abs();
    if a == 0.0 {
        return "0".to_string();
    }
    let n = n.max(1);
    if a < 1e-3 * (1.0 - 0.5 * 10f64.powi(-n)) {
        let text = format!("{a:.*e}", (n - 1) as usize);
        return if v < 0.0 { format!("−{text}") } else { text };
    }
    // (decided by the number as it rounds: 0.9996 is 1.00, not 1.000)
    let mut decimals = (n - 1 - a.log10().floor() as i32).max(0) as usize;
    let rounded: f64 = format!("{a:.decimals$}").parse().unwrap_or(a);
    if decimals > 0 && rounded >= 10f64.powi(n - decimals as i32) {
        decimals -= 1;
    }
    num(v, decimals)
}

/// The number a reader gets back from what is written.
fn shown(v: f64) -> f64 {
    read_back(&sig(v))
}

fn read_back(text: &str) -> f64 {
    text.replace('−', "-").parse().unwrap_or(f64::NAN)
}

/// Numbers added up, as a reader would write the sum: "0.12 − 0.03 + 0.20".
fn signed_sum(values: &[f64]) -> String {
    let mut out = String::new();
    for (i, v) in values.iter().enumerate() {
        let s = sig(*v);
        match (i, s.strip_prefix('−')) {
            (0, _) => out.push_str(&s),
            (_, Some(rest)) => out.push_str(&format!(" - {rest}")),
            (_, None) => out.push_str(&format!(" + {s}")),
        }
    }
    out
}

/// A factor in a product: in brackets if it is negative.
fn factor(v: f64) -> String {
    let s = sig(v);
    if s.starts_with('−') {
        format!("({s})")
    } else {
        s
    }
}

/// "=" if what a reader works out from the written numbers is written the same as the
/// true value, "≈" if rounding has made them differ.
fn rel(reader: f64, truth: f64) -> &'static str {
    if sig(reader) == sig(truth) {
        "="
    } else {
        r"\approx"
    }
}

/// A line of the arithmetic: formulas written one after another, which the strip
/// breaks between (never inside one).
type Line = Vec<String>;

/// What the strip under the network says: a heading and lines of arithmetic.
#[derive(Clone, Debug, Default, PartialEq)]
struct Said {
    title: String,
    lines: Vec<Line>,
}

/// The steps of the film: what each does.
const STEPS: usize = 9;
const STEP_TITLES: [&str; STEPS] = [
    "forward: each hidden unit adds up what comes in on its wires",
    r"forward: each sum goes through the $\tanh$",
    "forward: the output adds up what comes in on its wires",
    "forward: the loss",
    "back: the loss's slope with respect to the output",
    "back: each last-layer weight's slope is what came in on its wire times what came back",
    "back: what came back is handed along the same wires to the hidden units",
    r"back: through the $\tanh$, times its slope $1 - a_j^2$",
    "back: each first-layer weight's slope is what came in on its wire times what came back",
];

/// The arithmetic of step `step` (1 to 9) of the film, with the numbers of `p`.
fn film_lines(step: usize, w: &Weights, p: &Pass) -> Vec<Line> {
    let (x, z, a) = (p.x, p.z, p.a);
    match step {
        1 => (0..2)
            .map(|j| {
                let k = 3 * j;
                let reader = shown(w[k]) * shown(x[0]) + shown(w[k + 1]) * shown(x[1]) + shown(w[k + 2]);
                vec![
                    format!(r"$z_{u} = w_{{{u}1}} x_1 + w_{{{u}2}} x_2 + b_{u}$", u = j + 1),
                    format!(r"$= {} \cdot {} + {} \cdot {} + {}$", factor(w[k]), factor(x[0]), factor(w[k + 1]), factor(x[1]), factor(w[k + 2])),
                    format!("${} {}$", rel(reader, z[j]), sig(z[j])),
                ]
            })
            .collect(),
        2 => (0..2)
            .map(|j| vec![format!(r"$a_{u} = \tanh z_{u} = \tanh({})$", sig(z[j]), u = j + 1), format!("${} {}$", rel(shown(z[j]).tanh(), a[j]), sig(a[j]))])
            .collect(),
        3 => {
            let reader = shown(w[V1]) * shown(a[0]) + shown(w[V2]) * shown(a[1]) + shown(w[C]);
            vec![vec![
                r"$\hat y = v_1 a_1 + v_2 a_2 + c$".to_string(),
                format!(r"$= {} \cdot {} + {} \cdot {} + {}$", factor(w[V1]), factor(a[0]), factor(w[V2]), factor(a[1]), factor(w[C])),
                format!("${} {}$", rel(reader, p.out), sig(p.out)),
            ]]
        }
        4 => {
            let reader = 0.5 * (shown(p.out) - shown(p.y)).powi(2);
            vec![vec![
                r"$L = \tfrac12 (\hat y - y)^2$".to_string(),
                format!(r"$= \tfrac12 ({} - {})^2$", sig(p.out), factor(p.y)),
                format!("${} {}$", rel(reader, p.loss), sig(p.loss)),
            ]]
        }
        5 => {
            let reader = shown(p.out) - shown(p.y);
            vec![vec![
                r"$\delta = \partial L/\partial \hat y = \hat y - y$".to_string(),
                format!(r"$= {} - {}$", sig(p.out), factor(p.y)),
                format!("${} {}$", rel(reader, p.d), sig(p.d)),
            ]]
        }
        6 => {
            let mut lines: Vec<Line> = (0..2)
                .map(|j| {
                    let reader = shown(a[j]) * shown(p.d);
                    vec![
                        format!(r"$\partial L/\partial v_{u} = a_{u} \delta$", u = j + 1),
                        format!(r"$= {} \cdot {}$", factor(a[j]), factor(p.d)),
                        format!("${} {}$", rel(reader, p.grad[V1 + j]), sig(p.grad[V1 + j])),
                    ]
                })
                .collect();
            lines.push(vec![r"$\partial L/\partial c = 1 \cdot \delta$".to_string(), format!("$= {}$", sig(p.grad[C]))]);
            lines
        }
        7 => (0..2)
            .map(|j| {
                let reader = shown(w[V1 + j]) * shown(p.d);
                vec![
                    format!(r"$\partial L/\partial a_{u} = v_{u} \delta$", u = j + 1),
                    format!(r"$= {} \cdot {}$", factor(w[V1 + j]), factor(p.d)),
                    format!("${} {}$", rel(reader, p.da[j]), sig(p.da[j])),
                ]
            })
            .collect(),
        8 => (0..2)
            .map(|j| {
                let gate_reader = 1.0 - shown(a[j]).powi(2);
                let reader = shown(p.gate[j]) * shown(p.da[j]);
                vec![
                    format!(r"$\delta_{u} = (1 - a_{u}^2)\, \partial L/\partial a_{u}$", u = j + 1),
                    format!(r"$= (1 - {}^2) \cdot {}$", factor(a[j]), factor(p.da[j])),
                    format!(r"${} {} \cdot {}$", rel(gate_reader, p.gate[j]), sig(p.gate[j]), factor(p.da[j])),
                    format!("${} {}$", rel(reader, p.dz[j]), sig(p.dz[j])),
                ]
            })
            .collect(),
        9 => {
            let mut lines = Vec::new();
            for j in 0..2 {
                let k = 3 * j;
                for i in 0..2 {
                    let reader = shown(x[i]) * shown(p.dz[j]);
                    lines.push(vec![
                        format!(r"$\partial L/\partial w_{{{u}{i}}} = x_{i} \delta_{u}$", u = j + 1, i = i + 1),
                        format!(r"$= {} \cdot {}$", factor(x[i]), factor(p.dz[j])),
                        format!("${} {}$", rel(reader, p.grad[k + i]), sig(p.grad[k + i])),
                    ]);
                }
                lines.push(vec![format!(r"$\partial L/\partial b_{u} = 1 \cdot \delta_{u}$", u = j + 1), format!("$= {}$", sig(p.grad[k + 2]))]);
            }
            lines
        }
        _ => Vec::new(),
    }
}

/// What came in on weight `k`'s wire going forward (its name and value), and what came
/// back to the unit it feeds (its name and value).
fn wire_ends(k: usize, p: &Pass) -> ((String, f64), (String, f64)) {
    match k {
        V1 | V2 => ((format!("a_{}", k - V1 + 1), p.a[k - V1]), (r"\delta".to_string(), p.d)),
        C => (("1".to_string(), 1.0), (r"\delta".to_string(), p.d)),
        _ => {
            let (j, i) = (k / 3, k % 3);
            let back = (format!(r"\delta_{}", j + 1), p.dz[j]);
            if i == 2 {
                (("1".to_string(), 1.0), back)
            } else {
                ((format!("x_{}", i + 1), p.x[i]), back)
            }
        }
    }
}

/// The rule for one weight, with its numbers: its slope is what came in on its wire
/// times what came back to the unit it feeds.
fn rule_line(k: usize, p: &Pass) -> Line {
    let ((inn, vin), (back, vback)) = wire_ends(k, p);
    let reader = shown(vin) * shown(vback);
    vec![
        format!(r"$\partial L/\partial {} = {} {}$", NAMES[k], if inn == "1" { r"1 \cdot".to_string() } else { inn }, back),
        format!(r"$= {} \cdot {}$", factor(vin), factor(vback)),
        format!("${} {}$", rel(reader, p.grad[k]), sig(p.grad[k])),
    ]
}

// ----- experiments ---------------------------------------------------------------------

struct Trial {
    button: &'static str,
    caption: &'static str,
}

const TRIALS: [Trial; 5] = [
    Trial {
        button: "forwards, then back",
        caption: r"The film runs a step at a time, and the line under the network writes out each step's arithmetic. Forward, the cyan numbers fill in from the inputs to the loss, $L = 0.118$. Back, the pink slopes fill in from the loss to the inputs, along the same wires: $\delta = \hat y - y = -0.486$ at the output, then $\partial L/\partial a_1 = v_1 \delta = -0.340$, then through the $\tanh$, $\delta_1 = 0.822 \times (-0.340) \approx -0.280$ (the line writes $\approx$ where the rounded numbers it shows do not give the last figure exactly). At the end every wire has its slope, and each is what came in on that wire going forward times what came back to the unit it feeds: $\partial L/\partial w_{12} = x_2 \delta_1 = 0.5 \times (-0.280) = -0.140$.",
    },
    Trial {
        button: "nudge one weight",
        caption: r"$w_{12}$ is raised by 0.1, from $-0.3$ to $-0.2$. Its slope, $-0.140$, foretold that the loss would change by about $-0.140 \times 0.1 = -0.0140$; it changed by $-0.0133$, from 0.1182 to 0.1049. In the chart the slope is the straight dashed line, and close to the starting point the loss's curve hardly leaves it. Drag the yellow dot along the curve: the further from the start, the more the curve bends away and the worse the foretelling.",
    },
    Trial {
        button: "a saturated unit",
        caption: r"Unit 1's weights are now large and negative: $z_1 = -4.00$, far out on the flat part of $\tanh$ (the dot in its little graph), where the slope $1 - a_1^2$ is only 0.00134. The loss wants $a_1$ higher, $\partial L/\partial a_1 = -1.04$, but only that small share of it gets through: $\delta_1 = -0.00139$, and $w_{11}$, $w_{12}$ and $b_1$ have slopes as small. The loss is large, 1.10, and unit 1 is the cause; yet it is the unit that hardly learns, while unit 2's slopes are 614 times as large. The chart: the loss along $w_{11}$ is almost flat.",
    },
    Trial {
        button: "one step downhill, and one too long",
        caption: r"One step with the rate $\eta = 0.1$: every weight moved by $-0.1$ times its slope, and the loss fell from 0.118 to 0.0628. The chart is the loss along that same direction for steps of every length: at first it falls as the slopes foretell (the dashed line), it is lowest near $\eta = 0.37$, where it nearly reaches 0, and from $\eta = 0.76$ on a step lands higher than it started. With $\eta = 1$ the loss would have risen to 0.303. Try the rate slider, then step again.",
    },
    Trial {
        button: "train on the four points",
        caption: r"The four corners of the square, the target $+1$ where exactly one input is $+1$ (one or the other, not both) and $-1$ where both or neither are. Each step averages the four examples' slopes, with $\eta = 0.3$. The loss falls from 0.80 to 0.50 within twenty steps and stays near 0.50 while the output is about 0 everywhere, then drops: every output is within 0.1 of its target after 106 steps, and the plane shows the pattern. Start the two units alike (in the panel) and they stay alike for ever: the loss then never gets below $1/3$.",
    },
];

// ----- the layout ----------------------------------------------------------------------

/// Where everything goes in a window, in logical pixels.
#[derive(Clone, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    /// A narrow canvas (a phone held upright): smaller text, the hidden units' halves one
    /// above the other, one chart.
    narrow: bool,
    /// Too small for more than the network and its buttons.
    compact: bool,
    text: f32,
    small: f32,
    net: Geo,
    /// The film's buttons, descent's, and the choice of what is trained on (none on a
    /// narrow canvas: it is in the panel).
    film: [Rect; 3],
    descent: [Rect; 2],
    modes: Option<[Rect; 2]>,
    strip: Rect,
    /// The plane of inputs (its plot's rectangle) and the output's scale beside it.
    plane: Option<Rect>,
    scale: Option<Rect>,
    /// The chart of the loss along a line, and of the loss over the steps taken.
    line_chart: Option<Rect>,
    steps_chart: Option<Rect>,
}

/// The network's boxes, wires and the chips on the wires.
#[derive(Clone, Debug, PartialEq)]
struct Geo {
    room: Rect,
    /// The size of its text, and the sizes of its boxes.
    s: f32,
    z: Sizes,
    /// The units' halves stacked (sum above, tanh below) rather than side by side; and
    /// the loss under the output rather than beside it.
    stacked: bool,
    x: [Rect; 2],
    /// Each hidden unit: its sum's half, its little tanh graph, and its tanh's half.
    zs: [Rect; 2],
    glyphs: [Rect; 2],
    acts: [Rect; 2],
    out: Rect,
    loss: Rect,
    target: Rect,
    /// The sources of the biases' wires, the constant 1: for unit 1, unit 2 and the output.
    ones: [Vec2; 3],
    /// Each weight's wire, from its start to the unit it feeds, and where its numbers sit.
    wires: [(Vec2, Vec2); 9],
    chips: [Rect; 9],
    /// The wires that carry no weight: the output to the loss, the target to the loss.
    to_loss: (Vec2, Vec2),
    from_target: (Vec2, Vec2),
    /// Everything is inside the room and no box is on another.
    fits: bool,
}

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
    (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)))
}

fn centred(c: Vec2, size: Vec2) -> Rect {
    (c - size * 0.5, c + size * 0.5)
}

fn mid(r: Rect) -> Vec2 {
    (r.0 + r.1) * 0.5
}

fn left_mid(r: Rect) -> Vec2 {
    Vec2::new(r.0.x, (r.0.y + r.1.y) * 0.5)
}

fn right_mid(r: Rect) -> Vec2 {
    Vec2::new(r.1.x, (r.0.y + r.1.y) * 0.5)
}

fn top_mid(r: Rect) -> Vec2 {
    Vec2::new((r.0.x + r.1.x) * 0.5, r.0.y)
}

fn bottom_mid(r: Rect) -> Vec2 {
    Vec2::new((r.0.x + r.1.x) * 0.5, r.1.y)
}

fn overlap(a: Rect, b: Rect) -> bool {
    a.0.x < b.1.x && b.0.x < a.1.x && a.0.y < b.1.y && b.0.y < a.1.y
}

/// The sizes of the network's boxes for text of size `s`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sizes {
    /// A line of text in a box, and the room either side of what is written.
    line: f32,
    pad: f32,
    /// The widths of an input's box, a sum's half, a tanh's half, the output's, the
    /// loss's, and a chip's (the chips carry the weights' names where there is room).
    input: f32,
    sum: f32,
    act: f32,
    out: f32,
    loss: f32,
    chip: f32,
    named: bool,
}

/// Room for a value: a number as `sig` writes it, at its widest in ordinary use (on a
/// phone a little less, and a wider one is drawn smaller).
const WIDE_VALUE: &str = "$−0.0000$";
const NARROW_VALUE: &str = "$−0.000$";

/// The radius of the circles the biases' wires start from.
const ONE_R: f32 = 8.0;

fn sizes(s: f32, narrow: bool) -> Sizes {
    let pad = if narrow { 5.0 } else { 8.0 };
    let value = text_width(if narrow { NARROW_VALUE } else { WIDE_VALUE }, s);
    let boxed = |name: &str| pad + text_width(name, s) + pad + value + pad;
    Sizes {
        line: s + if narrow { 5.0 } else { 6.0 },
        pad,
        input: boxed("$x_1$"),
        sum: boxed(r"$\delta_1$"),
        act: boxed(r"$\partial L/\partial a_1$"),
        out: boxed(r"$\hat y$"),
        loss: boxed(r"$\ell$"),
        chip: if narrow { pad + value + pad } else { boxed("$w_{11}$").max(pad + text_width("slope", s - 1.5) + pad + value + pad) },
        named: !narrow,
    }
}

impl Geo {
    /// The network laid out in `room`, with text of size `s`. The flow is left to right.
    /// Where there is width (`!stacked`), a hidden unit's two halves are side by side,
    /// unit 2's bias comes from below and the loss is beside the output; where there is
    /// not, the halves are one above the other, every bias comes from above and the loss
    /// is under the output. (`fits` says whether that worked.)
    fn of(room: Rect, s: f32, stacked: bool) -> Geo {
        let z = sizes(s, stacked);
        let (w, h) = (room.1.x - room.0.x, room.1.y - room.0.y);
        let half_h = 2.0 * z.line + if stacked { 4.0 } else { 6.0 };
        let chip_h = 2.0 * z.line + if stacked { 2.0 } else { 4.0 };
        let glyph = if stacked { Vec2::new(z.act, 2.3 * s) } else { Vec2::new(5.0 * s, half_h) };
        let unit_w = if stacked { z.act } else { z.sum + glyph.x + z.act };
        let unit_h = if stacked { 2.0 * half_h + glyph.y } else { half_h };
        let input_h = z.line + 6.0;
        // from a bias's 1 to its box: room for the chip on the wire between
        let reach = chip_h + if stacked { 14.0 } else { 18.0 };
        let above = reach + ONE_R + 3.0;
        // across: the input boxes, the first layer's wires, the units, the second
        // layer's, the output (and the loss beside it)
        let fixed = z.input + unit_w + z.out + if stacked { 0.0 } else { z.loss };
        let least = if stacked { [1.5 * z.chip + 4.0, z.chip + 8.0, 0.0] } else { [1.6 * z.chip, 1.3 * z.chip, 34.0] };
        let spare = (w - 8.0 - fixed - least.iter().sum::<f32>()).max(0.0);
        let share = if stacked { [0.55, 0.45, 0.0] } else { [0.42, 0.36, 0.22] };
        let [g1, g2, g3] = [0, 1, 2].map(|i| least[i] + spare * share[i]);
        let x_left = room.0.x + 4.0;
        let unit_left = x_left + z.input + g1;
        let out_left = unit_left + unit_w + g2;
        // down: the units as far apart as their chips need and the biases allow
        let middle = room.0.y + h * 0.5;
        let tops = if stacked {
            let needed = 2.0 * above + 2.0 * unit_h + 4.0;
            let first = room.0.y + ((h - needed) * 0.5).max(0.0) + above;
            [first, first + unit_h + above + 4.0]
        } else {
            let least = (chip_h + 6.0) / 0.32 * 0.5;
            let most = h * 0.5 - above - unit_h * 0.5;
            let spread = (h * 0.27).min(most).max(least);
            [middle - spread - unit_h * 0.5, middle + spread - unit_h * 0.5]
        };
        let mut zs = [rect(0.0, 0.0, 0.0, 0.0); 2];
        let mut glyphs = zs;
        let mut acts = zs;
        for j in 0..2 {
            let (l, top) = (unit_left, tops[j]);
            if stacked {
                zs[j] = rect(l, top, l + unit_w, top + half_h);
                glyphs[j] = rect(l, top + half_h, l + unit_w, top + half_h + glyph.y);
                acts[j] = rect(l, top + half_h + glyph.y, l + unit_w, top + unit_h);
            } else {
                zs[j] = rect(l, top, l + z.sum, top + half_h);
                glyphs[j] = rect(l + z.sum, top, l + z.sum + glyph.x, top + half_h);
                acts[j] = rect(l + z.sum + glyph.x, top, l + unit_w, top + half_h);
            }
        }
        let ins = [left_mid(zs[0]), left_mid(zs[1])];
        let outs = [right_mid(acts[0]), right_mid(acts[1])];
        let x_mid = x_left + z.input * 0.5;
        let x = [centred(Vec2::new(x_mid, ins[0].y), Vec2::new(z.input, input_h)), centred(Vec2::new(x_mid, ins[1].y), Vec2::new(z.input, input_h))];
        // the output between the two units' outputs; the loss beside it, or under it
        let out_mid = out_left + z.out * 0.5;
        let (out, loss, target);
        if stacked {
            // (room above it for its bias, and below it for the loss and the target)
            let below = 0.5 * half_h + 16.0 + half_h + 14.0 + input_h + 2.0;
            let y = ((outs[0].y + outs[1].y) * 0.5).min(room.1.y - below).max(room.0.y + above + half_h * 0.5);
            out = centred(Vec2::new(out_mid, y), Vec2::new(z.out, half_h));
            loss = centred(Vec2::new(out_mid, out.1.y + 16.0 + half_h * 0.5), Vec2::new(z.out, half_h));
            target = centred(Vec2::new(out_mid, loss.1.y + 14.0 + input_h * 0.5), Vec2::new(z.out, input_h));
        } else {
            let y = (outs[0].y + outs[1].y) * 0.5;
            out = centred(Vec2::new(out_mid, y), Vec2::new(z.out, half_h));
            let loss_mid = out.1.x + g3 + z.loss * 0.5;
            loss = centred(Vec2::new(loss_mid, y), Vec2::new(z.loss, half_h));
            target = centred(Vec2::new(loss_mid, loss.1.y + 26.0 + input_h * 0.5), Vec2::new(z.loss, input_h));
        }
        // The first layer's chips in a column 0.66 of the way along to the units: the
        // crossing wires' chips are then a third of the way between the straight ones',
        // in the order w11, w12, w21, w22. The second layer's half way.
        let along = |from: Vec2, to: Vec2, at_x: f32| {
            let t = ((at_x - from.x) / (to.x - from.x).max(1e-3)).clamp(0.0, 1.0);
            from + (to - from) * t
        };
        let col1 = x[0].1.x + (ins[0].x - x[0].1.x) * 0.66;
        let col2 = outs[0].x + (out.0.x - outs[0].x) * 0.5;
        let chip = |c: Vec2| centred(c, Vec2::new(z.chip, chip_h));
        let mut wires = [(Vec2::ZERO, Vec2::ZERO); 9];
        let mut chips = [rect(0.0, 0.0, 0.0, 0.0); 9];
        for j in 0..2 {
            for i in 0..2 {
                let (from, to) = (right_mid(x[i]), ins[j]);
                wires[3 * j + i] = (from, to);
                chips[3 * j + i] = chip(along(from, to, col1));
            }
        }
        // the biases: from a 1 just outside a box, into it
        let into = [top_mid(zs[0]), if stacked { top_mid(zs[1]) } else { bottom_mid(zs[1]) }, if stacked { top_mid(out) } else { bottom_mid(out) }];
        let dir = [-1.0, if stacked { -1.0 } else { 1.0 }, if stacked { -1.0 } else { 1.0 }];
        let ones = [0, 1, 2].map(|i| into[i] + Vec2::new(0.0, dir[i] * reach));
        for (k, i) in [(B1, 0), (B2, 1), (C, 2)] {
            wires[k] = (ones[i], into[i]);
            // (the chip half way between the circle's edge and the box)
            chips[k] = chip(into[i] + Vec2::new(0.0, dir[i] * (reach + ONE_R) * 0.5));
        }
        for j in 0..2 {
            let (from, to) = (outs[j], left_mid(out));
            wires[V1 + j] = (from, to);
            chips[V1 + j] = chip(along(from, to, col2));
        }
        let (to_loss, from_target) =
            if stacked { ((bottom_mid(out), top_mid(loss)), (top_mid(target), bottom_mid(loss))) } else { ((right_mid(out), left_mid(loss)), (top_mid(target), bottom_mid(loss))) };
        let mut geo = Geo { room, s, z, stacked, x, zs, glyphs, acts, out, loss, target, ones, wires, chips, to_loss, from_target, fits: false };
        let boxes = geo.boxes();
        let (lo, hi) = geo.extent();
        let apart = boxes.iter().enumerate().all(|(i, a)| boxes.iter().skip(i + 1).all(|b| !overlap(*a, *b)));
        geo.fits = apart && lo.x >= room.0.x - 0.5 && lo.y >= room.0.y - 0.5 && hi.x <= room.1.x + 0.5 && hi.y <= room.1.y + 0.5 && w > 0.0 && h > 0.0;
        geo
    }

    /// Every box of the network (to keep apart, and inside the room).
    fn boxes(&self) -> Vec<Rect> {
        let mut all = vec![self.x[0], self.x[1], self.out, self.loss, self.target];
        for j in 0..2 {
            all.push(self.zs[j]);
            all.push(self.glyphs[j]);
            all.push(self.acts[j]);
        }
        all.extend(self.chips);
        all
    }

    /// The rectangle everything is drawn in.
    fn extent(&self) -> Rect {
        let mut lo = Vec2::splat(f32::MAX);
        let mut hi = Vec2::splat(f32::MIN);
        for b in self.boxes() {
            lo = lo.min(b.0);
            hi = hi.max(b.1);
        }
        for one in self.ones {
            lo = lo.min(one - Vec2::splat(ONE_R + 1.0));
            hi = hi.max(one + Vec2::splat(ONE_R + 1.0));
        }
        (lo, hi)
    }

    /// The network in `room` with the largest text, from `most` down, at which it fits
    /// (at the smallest, if none does).
    fn fitted(room: Rect, most: f32, stacked: bool) -> Geo {
        let mut s = most;
        loop {
            let geo = Geo::of(room, s, stacked);
            if geo.fits || s <= 8.5 {
                return geo;
            }
            s -= 0.5;
        }
    }
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 700.0;
        let (text, small) = if narrow { (11.0, 10.0) } else { (13.0, 11.5) };
        let side = if narrow { 8.0 } else { 18.0 };
        let top = inset.max(0.0) + if narrow { 4.0 } else { 10.0 };
        let button_h = if narrow { 26.0 } else { 28.0 };
        // the network with the strip and the charts under it, or (where it would not fit
        // so) with the whole height to itself
        let mut compact = view.y - inset < 440.0 || view.x < 300.0;
        let mut net = None;
        if !compact {
            let net_h = if narrow { ((view.y - inset) * 0.44).clamp(250.0, 360.0) } else { ((view.y - inset) * 0.46).clamp(250.0, 460.0) };
            let geo = Geo::fitted(rect(side, top, view.x - side, top + net_h), text, narrow);
            compact = !geo.fits;
            net = Some(geo);
        }
        let net = match net {
            Some(geo) if !compact => geo,
            _ => Geo::fitted(rect(side, top, view.x - side, (view.y - button_h - 14.0).max(top + 40.0)), text, narrow),
        };
        let room = net.room;
        let y = room.1.y + 6.0;
        let (film, descent, modes);
        if narrow {
            let row = plot::even_row(5, side, view.x - side, y, button_h, 5.0);
            film = [row[0], row[1], row[2]];
            descent = [row[3], row[4]];
            modes = None;
        } else {
            let width = |w: &str| text_width(w, small) + 22.0;
            let mut x = side;
            let mut next = |w: &str, gap: f32| {
                let r = rect(x, y, x + width(w), y + button_h);
                x += width(w) + gap;
                r
            };
            film = [next("◀ back", 6.0), next("step ▶", 6.0), next("❚❚ pause", 24.0)];
            descent = [next("one step downhill", 6.0), next("stop training", 6.0)];
            let w2 = [text_width("one example", small) + 22.0, text_width("four examples", small) + 22.0];
            let x1 = view.x - side;
            let left = x1 - w2[1] - 6.0 - w2[0];
            modes = (left > x + 12.0).then(|| [rect(left, y, x1 - w2[1] - 6.0, y + button_h), rect(x1 - w2[1], y, x1, y + button_h)]);
        }
        let strip_top = y + button_h + 8.0;
        let strip_lines = if narrow { 5.0 } else { 3.0 };
        let strip = rect(side, strip_top, view.x - side, strip_top + small + 8.0 + strip_lines * (text + 9.0));
        let mut lay = Layout { view, narrow, compact, text, small, net, film, descent, modes, strip, plane: None, scale: None, line_chart: None, steps_chart: None };
        if compact {
            lay.strip = rect(side, y + button_h + 2.0, view.x - side, y + button_h + 3.0);
            return lay;
        }
        // the row of charts: the plane and its scale, then the charts (titles above,
        // tick numbers to the left and below)
        let row_top = strip.1.y + 10.0 + 44.0;
        let row_bottom = view.y - if narrow { 30.0 } else { 40.0 };
        let height = row_bottom - row_top;
        if height < 80.0 {
            return lay;
        }
        let x0 = side + 30.0;
        let x1 = view.x - side;
        // the scale: a bar, the names of the output and the target to its left and its
        // numbers to its right
        let (scale_gap, scale_w, scale_after) = if narrow { (26.0, 10.0, 20.0) } else { (34.0, 12.0, 26.0) };
        if narrow {
            let plane_side = height.min((x1 - x0 - scale_gap - scale_w - scale_after - 40.0) * 0.5).max(60.0);
            let p1 = x0 + plane_side;
            lay.plane = Some(rect(x0, row_top, p1, row_top + plane_side));
            lay.scale = Some(rect(p1 + scale_gap, row_top, p1 + scale_gap + scale_w, row_top + plane_side));
            let c0 = p1 + scale_gap + scale_w + scale_after + 34.0;
            if x1 - c0 > 90.0 {
                lay.line_chart = Some(rect(c0, row_top, x1, row_top + plane_side));
            }
        } else {
            let plane_side = height.min(330.0).min((x1 - x0) * 0.28);
            let p1 = x0 + plane_side;
            lay.plane = Some(rect(x0, row_top, p1, row_top + plane_side));
            lay.scale = Some(rect(p1 + scale_gap, row_top, p1 + scale_gap + scale_w, row_top + plane_side));
            let c0 = p1 + scale_gap + scale_w + scale_after + 52.0;
            let each = (x1 - c0 - 52.0) * 0.5;
            let chart_h = height.min(plane_side.max(180.0));
            if each > 140.0 {
                lay.line_chart = Some(rect(c0, row_top, c0 + each, row_top + chart_h));
                lay.steps_chart = Some(rect(c0 + each + 52.0, row_top, x1, row_top + chart_h));
            } else if x1 - c0 > 120.0 {
                lay.line_chart = Some(rect(c0, row_top, x1, row_top + chart_h));
            }
        }
        lay
    }
}

// ----- the scene -----------------------------------------------------------------------

/// A weight moved by hand from where it was, to check its slope against the loss.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Check {
    k: usize,
    /// Everything as it was before (the other weights must still be so).
    base: Weights,
    loss: f64,
    slope: f64,
}

/// The last step downhill: where it started, its slopes and its rate (the chart of the
/// loss along it is drawn while the weights are still where it left them).
#[derive(Clone, Copy, Debug, PartialEq)]
struct LastStep {
    before: Weights,
    grad: Weights,
    rate: f64,
    after: Weights,
    loss: (f64, f64),
}

/// What the strip is about.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Topic {
    Film,
    Rule,
    Check,
    Step,
    Train,
}

/// Seconds between the film's steps as it plays, and what one step's animation takes.
const PLAY_EVERY: f32 = 1.7;
const REVEAL: f32 = 1.0;
/// The share of a step's animation over which its pulses travel, and over which its
/// numbers then fade in.
const TRAVEL: f32 = 0.6;

/// How a change of weights not made by hand is glided to (each of two eases, seconds).
const EASE: f64 = 0.08;
/// The least height of a chart of the loss.
const LEAST_TOP: f64 = 0.02;
/// The least length of the steps' axis.
const STEPS_SHOWN: f64 = 10.0;

/// `value` (between its two eases, and as drawn) `dt` seconds nearer to `goal`, with
/// each ease taking `tau`; exactly there once it is near. Says whether it moved.
fn ease_to(value: &mut (f64, f64), goal: f64, dt: f64, tau: f64) -> bool {
    if *value == (goal, goal) {
        return false;
    }
    if !(goal.is_finite() && value.0.is_finite() && value.1.is_finite()) {
        *value = (goal, goal);
        return true;
    }
    let k = 1.0 - (-dt.max(0.0) / tau).exp();
    value.0 += (goal - value.0) * k;
    value.1 += (value.0 - value.1) * k;
    let near = 1e-4 * goal.abs().max(1.0);
    if (goal - value.0).abs() < near && (goal - value.1).abs() < near {
        *value = (goal, goal);
    }
    true
}

pub struct Backprop1 {
    mode: Mode,
    w: Weights,
    x: [f64; 2],
    y: f64,
    /// Which of the four examples the units show.
    picked: usize,
    /// The film: steps done, what is drawn (gliding to it), playing, and the time to
    /// the next step.
    film: usize,
    reveal: f32,
    playing: bool,
    play_wait: f32,
    topic: Topic,
    selected: Option<usize>,
    check: Option<Check>,
    rate: f64,
    last: Option<LastStep>,
    training: bool,
    speed: f32,
    pace: Pace,
    /// What the pace owes, kept beside it: the share of a step the picture has reached.
    owed: f32,
    steps: u64,
    hist: History,
    /// The weights as drawn, and between the two eases; the weights before the last
    /// step of training (the picture is drawn between those and the weights now).
    drawn: Weights,
    between: Weights,
    before: Weights,
    /// The middle of the chart of the loss along one weight, and the weight it is for;
    /// the length of the steps' axis, as drawn (between its two eases, and drawn).
    centre: (f64, Option<usize>),
    span: (f64, f64),
    hover: Option<usize>,
    /// Was a handle held in the last frame (then the picture answers at once).
    holding: bool,
    /// The layout, for the window and the inset it was worked out for.
    layout: Option<(Vec2, f32, Layout)>,
}

impl Backprop1 {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Backprop1 {
        let mut s = Backprop1 {
            mode: Mode::One,
            w: START,
            x: X0,
            y: Y0,
            picked: 1,
            film: STEPS,
            reveal: STEPS as f32,
            playing: false,
            play_wait: 0.0,
            topic: Topic::Rule,
            selected: None,
            check: None,
            rate: 0.3,
            last: None,
            training: false,
            speed: 20.0,
            pace: Pace::default(),
            owed: 0.0,
            steps: 0,
            hist: History::new(400),
            drawn: START,
            between: START,
            before: START,
            centre: (0.0, None),
            span: (STEPS_SHOWN, STEPS_SHOWN),
            hover: None,
            holding: false,
            layout: None,
        };
        s.set_up(0);
        // (opening, the film is at its end: the whole picture, nothing to glide from)
        s.film = STEPS;
        s.reveal = STEPS as f32;
        s.playing = false;
        s.topic = Topic::Rule;
        s.snap();
        s
    }

    /// Everything back to the scene's start, in `mode`, with the weights `w`.
    fn restart(&mut self, mode: Mode, w: Weights) {
        self.mode = mode;
        self.w = w;
        self.x = X0;
        self.y = Y0;
        self.picked = 1;
        self.selected = None;
        self.check = None;
        self.last = None;
        self.training = false;
        self.playing = false;
        self.steps = 0;
        self.hist.clear();
        self.pace.reset();
        self.owed = 0.0;
        self.before = w;
        self.show_all();
        self.topic = if mode == Mode::Four { Topic::Train } else { Topic::Rule };
    }

    /// Set up experiment `k` of `TRIALS`.
    fn set_up(&mut self, k: usize) {
        self.restart(Mode::One, START);
        self.rate = 0.3;
        self.speed = 20.0;
        match k {
            0 => {
                (self.film, self.reveal, self.playing, self.play_wait) = (0, 0.0, true, 0.6);
                self.topic = Topic::Film;
                self.selected = Some(W12);
            }
            1 => {
                self.selected = Some(W12);
                self.start_check(W12);
                self.w[W12] += 0.1;
                self.topic = Topic::Check;
            }
            2 => {
                (self.w[W11], self.w[W12], self.w[B1]) = (-2.5, -2.0, -0.5);
                self.selected = Some(W11);
                self.topic = Topic::Rule;
            }
            3 => {
                self.rate = 0.1;
                self.step_downhill();
            }
            _ => {
                self.restart(Mode::Four, START);
                self.training = true;
            }
        }
    }

    /// The film at its end: every number shown.
    fn show_all(&mut self) {
        (self.film, self.reveal, self.playing) = (STEPS, STEPS as f32, false);
    }

    /// The drawn weights are the weights, at once.
    fn snap(&mut self) {
        (self.drawn, self.between, self.before) = (self.w, self.w, self.w);
    }

    /// The pass the units show: the one example's, or the picked one of the four.
    fn shown_pass(&self, w: &Weights) -> Pass {
        match self.mode {
            Mode::One => pass(w, self.x, self.y),
            Mode::Four => {
                let (x, y) = FOUR[self.picked.min(3)];
                pass(w, x, y)
            }
        }
    }

    fn loss_and_grad(&self, w: &Weights) -> (f64, Weights) {
        loss_and_grad(self.mode, w, self.x, self.y)
    }

    fn loss_at(&self, w: &Weights) -> f64 {
        loss_of(self.mode, w, self.x, self.y)
    }

    /// Start checking weight `k`'s slope: remember where everything is.
    fn start_check(&mut self, k: usize) {
        let (loss, grad) = self.loss_and_grad(&self.w);
        self.check = Some(Check { k, base: self.w, loss, slope: grad[k] });
    }

    /// The check, if it still holds: only its own weight has moved since it began.
    fn live_check(&self) -> Option<Check> {
        self.check.filter(|c| (0..9).all(|i| i == c.k || c.base[i] == self.w[i]))
    }

    /// The last step, if the weights are still where it left them.
    fn last_step(&self) -> Option<LastStep> {
        self.last.filter(|l| l.after == self.w)
    }

    /// One step downhill at the rate set, from the weights as they are.
    fn step_downhill(&mut self) {
        let (l0, grad) = self.loss_and_grad(&self.w);
        let after = descend(&self.w, &grad, self.rate);
        let l1 = self.loss_at(&after);
        if self.hist.is_empty() {
            self.hist.push(l0 as f32);
        }
        self.hist.push(l1 as f32);
        self.steps += 1;
        self.last = Some(LastStep { before: self.w, grad, rate: self.rate, after, loss: (l0, l1) });
        self.w = after;
        self.check = None;
        self.show_all();
        self.topic = Topic::Step;
    }

    /// One step of training (the same step; the picture is drawn between the weights
    /// before it and after it).
    fn train_step(&mut self) {
        self.before = self.w;
        self.step_downhill();
        self.topic = Topic::Train;
    }

    /// A change made by hand: what depended on the weights as they were is let go.
    fn changed_by_hand(&mut self) {
        self.last = None;
        self.hist.clear();
        self.steps = 0;
        self.show_all_unless_film();
    }

    fn show_all_unless_film(&mut self) {
        if !self.playing && self.film == STEPS {
            self.reveal = STEPS as f32;
        }
    }

    fn set_mode(&mut self, mode: Mode) {
        if self.mode != mode {
            self.mode = mode;
            self.check = None;
            self.last = None;
            self.training = false;
            self.hist.clear();
            self.steps = 0;
            self.show_all();
            self.topic = if mode == Mode::Four { Topic::Train } else { Topic::Rule };
        }
    }

    /// The film one step on (back to the start from its end), or back.
    fn film_step(&mut self) {
        self.training = false;
        self.playing = false;
        if self.film >= STEPS {
            (self.film, self.reveal) = (0, 0.0);
        } else {
            self.film += 1;
        }
        self.topic = Topic::Film;
    }

    fn film_back(&mut self) {
        self.training = false;
        self.playing = false;
        self.film = self.film.saturating_sub(1);
        self.reveal = self.film as f32;
        self.topic = Topic::Film;
    }

    fn film_play(&mut self) {
        self.training = false;
        if self.playing {
            self.playing = false;
            return;
        }
        if self.film >= STEPS {
            (self.film, self.reveal) = (0, 0.0);
        }
        (self.playing, self.play_wait) = (true, 0.4);
        self.topic = Topic::Film;
    }

    fn toggle_training(&mut self) {
        self.training = !self.training;
        self.playing = false;
        if self.training {
            self.show_all();
            self.check = None;
            self.pace.reset();
            self.owed = 0.0;
            self.before = self.w;
            self.topic = Topic::Train;
        }
    }

    /// Move everything on: the film by the scene's time, training by the pace, and
    /// the drawn weights towards the weights (by the clock: they answer buttons).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        if self.playing {
            self.play_wait -= dt;
            if self.play_wait <= 0.0 {
                if self.film < STEPS {
                    self.film += 1;
                    self.play_wait += PLAY_EVERY;
                }
                if self.film >= STEPS {
                    self.playing = false;
                }
            }
        }
        // the film's animation runs at an even pace to the step it has reached
        let to = self.film as f32;
        if self.reveal < to {
            self.reveal = (self.reveal + real_dt.max(0.0) / REVEAL).min(to);
        } else {
            self.reveal = to;
        }
        if self.training && dt > 0.0 {
            let rate = self.speed.max(0.1);
            let mut turn = self.pace.begin(rate, dt, real_dt, 0.004);
            self.owed = (self.owed + rate * dt).min((rate * 0.1).max(2.0));
            while turn.next() {
                self.owed -= 1.0;
                self.train_step();
            }
            self.pace.end(turn);
        }
        if self.training {
            // drawn between the weights before the last step and the weights now
            let t = self.owed.clamp(0.0, 1.0) as f64;
            for i in 0..9 {
                self.drawn[i] = self.before[i] + (self.w[i] - self.before[i]) * t;
            }
            self.between = self.drawn;
        } else if self.holding {
            (self.drawn, self.between) = (self.w, self.w);
        } else {
            let k = 1.0 - (-(real_dt.max(0.0) as f64) / EASE).exp();
            let mut apart = 0.0f64;
            for i in 0..9 {
                self.between[i] += (self.w[i] - self.between[i]) * k;
                self.drawn[i] += (self.between[i] - self.drawn[i]) * k;
                apart = apart.max((self.w[i] - self.between[i]).abs()).max((self.w[i] - self.drawn[i]).abs());
            }
            if apart < 1e-4 || !apart.is_finite() {
                (self.drawn, self.between) = (self.w, self.w);
            }
        }
        // The chart of the loss along one weight is centred where its check began, or on
        // the weight; it is moved, at once, only when the weight is a new one or has gone
        // well off its middle (a chart that slid after every step would never be still).
        if let Some(k) = self.selected {
            let goal = self.centre_goal(k);
            if self.centre.1 != Some(k) || (goal - self.centre.0).abs() > 0.6 || !self.centre.0.is_finite() {
                self.centre = (goal, Some(k));
            }
        }
        // The steps' axis grows with the steps taken: while training with the drawn
        // weights, which move evenly; after a single step, gliding.
        let span = self.span_goal();
        if self.training {
            self.span = (span, span);
        } else {
            ease_to(&mut self.span, span, real_dt as f64, EASE);
        }
    }

    fn centre_goal(&self, k: usize) -> f64 {
        self.live_check().map_or(self.w[k], |c| c.base[k])
    }

    /// How far along the steps the picture has got: the steps taken, less what the
    /// drawn weights still lag behind while training.
    fn taken_drawn(&self) -> f64 {
        let n = self.hist.points().len();
        let lag = if self.training { 1.0 - self.owed.clamp(0.0, 1.0) as f64 } else { 0.0 };
        ((n.max(1) - 1) as f64 - lag).max(0.0) * self.hist.every() as f64
    }

    /// The steps' axis: 10 steps long, and then a quarter again as long as the steps
    /// taken (so that it grows as evenly as they are drawn).
    fn span_goal(&self) -> f64 {
        (self.taken_drawn() / 0.8).max(STEPS_SHOWN)
    }

    fn moving(&self) -> bool {
        let span = self.span_goal();
        self.playing
            || self.reveal < self.film as f32
            || self.training
            || self.drawn != self.w
            || self.between != self.w
            || self.span != (span, span)
    }

    /// What the strip says now.
    fn said(&self) -> Said {
        let p = self.shown_pass(&self.w);
        let w = &self.w;
        match self.topic {
            Topic::Film if self.mode == Mode::One => {
                if self.film == 0 {
                    return Said {
                        title: "Step 0 of 9: the inputs, the target and the weights are set; the rest is to be worked out. Press “step ▶”.".to_string(),
                        lines: Vec::new(),
                    };
                }
                let s = self.film.min(STEPS);
                Said { title: format!("Step {s} of 9, {}", STEP_TITLES[s - 1]), lines: film_lines(s, w, &p) }
            }
            Topic::Check => match self.live_check() {
                Some(c) => {
                    let nudge = w[c.k] - c.base[c.k];
                    let now = self.loss_at(w);
                    let foretold = c.slope * nudge;
                    Said {
                        title: format!(r"${}$ moved by hand from ${}$ to ${}$: its slope foretold the change in the loss, the slope times the nudge", NAMES[c.k], sig(c.base[c.k]), sig(w[c.k])),
                        lines: vec![
                            vec![
                                format!(r"$\text{{foretold}} = {} \times {}$", factor(c.slope), factor(nudge)),
                                format!("${} {}$", rel(shown(c.slope) * shown(nudge), foretold), sig(foretold)),
                            ],
                            {
                                let (a, b) = (figures(now, 4), figures(c.loss, 4));
                                let reader = read_back(&a) - read_back(&b);
                                vec![format!(r"$\text{{measured}} = {a} - {b}$"), format!("${} {}$", rel(reader, now - c.loss), sig(now - c.loss))]
                            },
                        ],
                    }
                }
                None => self.said_rule(&p),
            },
            Topic::Step | Topic::Train if self.mode == Mode::One || self.topic == Topic::Step => match self.last_step() {
                Some(l) => {
                    let k = self.selected.unwrap_or(W12);
                    let reader = shown(l.before[k]) - shown(l.rate) * shown(l.grad[k]);
                    Said {
                        title: format!(r"One step downhill with the rate $\eta = {}$: every weight minus $\eta$ times its slope. The loss went from ${}$ to ${}$.", sig(l.rate), sig(l.loss.0), sig(l.loss.1)),
                        lines: vec![vec![
                            format!(r"${} \leftarrow {} - \eta\, \partial L/\partial {}$", NAMES[k], NAMES[k], NAMES[k]),
                            format!(r"$= {} - {} \cdot {}$", sig(l.before[k]), sig(l.rate), factor(l.grad[k])),
                            format!("${} {}$", rel(reader, l.after[k]), sig(l.after[k])),
                        ]],
                    }
                }
                None => self.said_rule(&p),
            },
            _ if self.mode == Mode::Four => {
                let k = self.selected.unwrap_or(W11);
                let (passes, loss, grad) = passes_of_four(w);
                let terms: Vec<f64> = passes.iter().map(|q| q.grad[k]).collect();
                let reader = terms.iter().map(|v| shown(*v)).sum::<f64>() / 4.0;
                let n = NAMES[k];
                Said {
                    title: format!(r"Four examples, mean loss $L = {}$ after {} steps. $\ell_i$ is example $i$'s own loss, $\tfrac12 (\hat y - y)^2$; each step uses the mean of the four examples' slopes, for every weight:", sig(loss), self.steps),
                    lines: vec![vec![
                        format!(r"$\partial L/\partial {n} = \tfrac14 (\partial \ell_1/\partial {n} + \cdots + \partial \ell_4/\partial {n})$"),
                        format!(r"$= \tfrac14 ({})$", signed_sum(&terms)),
                        format!("${} {}$", rel(reader, grad[k]), sig(grad[k])),
                    ]],
                }
            }
            _ => self.said_rule(&p),
        }
    }

    fn said_rule(&self, p: &Pass) -> Said {
        let k = self.selected.unwrap_or(W12);
        Said { title: "Every weight's slope is what came in on its wire going forward, times what came back to the unit it feeds:".to_string(), lines: vec![rule_line(k, p)] }
    }
}

// ----- drawing -------------------------------------------------------------------------

/// Batches of lines by kind (see `plot::apart`): each is drawn in every frame, with as
/// many points, so that what moves can be followed from frame to frame.
const K_WIRES: u8 = plot::OWN;
const K_DASHED: u8 = plot::OWN + 1;
const K_LIT: u8 = plot::OWN + 2;
const K_BOXES: u8 = plot::OWN + 3;
const K_GLYPH: u8 = plot::OWN + 4;
/// The chart along one weight, and the chart along the step: each its own axes, curve
/// and marks (so that one taking the other's place is a change, not a movement).
const K_A_AXES: u8 = plot::OWN + 5;
const K_A_CURVE: u8 = plot::OWN + 6;
const K_A_MARKS: u8 = plot::OWN + 7;
const K_B_AXES: u8 = plot::OWN + 8;
const K_B_CURVE: u8 = plot::OWN + 9;
const K_B_MARKS: u8 = plot::OWN + 10;
const K_HISTORY: u8 = plot::OWN + 11;
const K_STEPS_AXES: u8 = plot::OWN + 12;

/// Batches of discs of their own (told apart by a hair of their size): the pulses, and
/// the dots of each of the two charts.
const PULSE_DOTS: u32 = 1;
const A_DOTS: u32 = 2;
const B_DOTS: u32 = 3;
const NOW_DOT: u32 = 4;

fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [plot::AXES, plot::CURVES, plot::MARKS, plot::BUTTONS, K_BOXES, K_GLYPH, K_A_AXES, K_A_CURVE, K_A_MARKS, K_B_AXES, K_B_CURVE, K_B_MARKS, K_HISTORY, K_STEPS_AXES] {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [K_A_MARKS, K_B_MARKS] {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for (kind, dashed) in [(K_WIRES, false), (K_DASHED, true), (K_LIT, false)] {
        wire(f, kind, Vec2::ZERO, Vec2::ZERO, 1.0, Color::CLEAR, dashed);
    }
    plot::disc(f, Vec2::ZERO, 1.0, Color::CLEAR);
    for batch in [PULSE_DOTS, A_DOTS, B_DOTS, NOW_DOT] {
        dot_in(f, batch, Vec2::ZERO, 1.0, Color::CLEAR);
    }
}

/// A disc on top of everything, in a batch of its own.
fn dot_in(f: &mut Frame, batch: u32, at: Vec2, size: f32, color: Color) {
    if at.is_finite() {
        let p = f.hud(at);
        f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, size_scale: 1.0 + 1e-4 * batch as f32, ..Default::default() }).push(p, size, color);
    }
}

/// A chart's axes, as `Plot::x_axis` and `Plot::y_axis` draw them, the two lines in a
/// batch of their own and the ticks as thin filled marks: a chart whose range slides
/// or grows has ticks that come and go at its ends, and a tick drawn as a line would
/// then, from one frame to the next, be in the place another one was.
fn axes_in(f: &mut Frame, pl: &Plot, kind: u8, x_name: &str, y_name: &str, size: f32) {
    if pl.tiny() {
        return;
    }
    let (min, max) = pl.rect;
    let c = pal::DIM.a(0.8);
    plot::line_of(f, kind, Vec2::new(min.x, max.y), max, 1.0, c);
    plot::line_of(f, kind, min, Vec2::new(min.x, max.y), 1.0, c);
    let ticks = plot::Ticks::of(pl.x[0], pl.x[1], pl.width());
    for v in ticks.values() {
        let x = pl.px_x(v);
        plot::fill(f, (Vec2::new(x - 0.5, max.y), Vec2::new(x + 0.5, max.y + 4.0)), c);
        f.text_px(Vec2::new(x, max.y + 6.0), ticks.name(v), size, pal::DIM, Align2::CENTER_TOP);
    }
    f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 8.0 + size), x_name, size, pal::DIM, Align2::CENTER_TOP);
    let ticks = plot::Ticks::of(pl.y[0], pl.y[1], pl.height() * 1.6);
    for v in ticks.values() {
        let y = pl.px_y(v);
        plot::fill(f, (Vec2::new(min.x - 4.0, y - 0.5), Vec2::new(min.x, y + 0.5)), c);
        f.text_px(Vec2::new(min.x - 6.0, y), ticks.name(v), size, pal::DIM, Align2::RIGHT_CENTER);
    }
    f.text_px(Vec2::new(min.x - 30.0, min.y - 6.0), y_name, size, pal::DIM, Align2::LEFT_BOTTOM);
}

/// The ends of the straight line of slope `slope` through (x, y) of a chart, from
/// `left` before x to `right` after it, but shortened at both ends alike so that it
/// stays inside the chart (shortened, not cut, and alike: its ends move evenly as the
/// line turns, also through level).
fn tangent_ends(pl: &Plot, x: f64, y: f64, slope: f64, left: f64, right: f64) -> (Vec2, Vec2) {
    let (bottom, top) = (pl.y[0], pl.y[1]);
    let y = y.max(bottom).min(top);
    // how far along it may go either way before it leaves through the top or the bottom
    let room = (top - y).min(y - bottom).max(0.0);
    let most = if slope == 0.0 || !slope.is_finite() { f64::INFINITY } else { room / slope.abs() };
    let (l, r) = (left.min(most).max(0.0), right.min(most).max(0.0));
    let (l, r) = if l.is_finite() && r.is_finite() { (l, r) } else { (0.0, 0.0) };
    (pl.at(x - l, y - slope * l), pl.at(x + r, y + slope * r))
}

/// A curve of a chart that stops where it leaves the chart through its top or its
/// bottom: the points beyond are all put where it crossed the edge (rather than cut
/// off, so that it keeps its number of points and can be followed from frame to frame).
fn clamped_path(f: &mut Frame, pl: &Plot, kind: u8, pts: &[(f64, f64)], width: f32, color: Color) {
    let (lo, hi) = pl.rect;
    let mut at: Vec<Vec2> = pts.iter().map(|&(x, y)| pl.at(x, if y.is_finite() { y } else { pl.y[1] * 10.0 })).collect();
    let inside = |p: Vec2| p.y >= lo.y - 0.5 && p.y <= hi.y + 0.5;
    // where the segment from a (inside) to b (outside) meets the edge b is beyond
    let crossing = |a: Vec2, b: Vec2| {
        let edge = if b.y < lo.y { lo.y } else { hi.y };
        let t = ((edge - a.y) / (b.y - a.y)).clamp(0.0, 1.0);
        a + (b - a) * if t.is_finite() { t } else { 0.0 }
    };
    let n = at.len();
    let mut i = 0;
    while i < n {
        if inside(at[i]) {
            i += 1;
            continue;
        }
        let start = i;
        while i < n && !inside(at[i]) {
            i += 1;
        }
        // the run of points beyond the edge: the first half to where the curve left,
        // the rest to where it came back (or all to the one there is)
        let left = (start > 0).then(|| crossing(at[start - 1], at[start]));
        let back = (i < n).then(|| crossing(at[i], at[i - 1]));
        let half = start + (i - start).div_ceil(2);
        for k in start..i {
            let to = match (left, back) {
                (Some(l), Some(b)) => if k < half { l } else { b },
                (Some(l), None) => l,
                (None, Some(b)) => b,
                (None, None) => Vec2::new(at[k].x, at[k].y.max(lo.y).min(hi.y)),
            };
            at[k] = to;
        }
    }
    for p in &mut at {
        *p = p.max(lo).min(hi);
    }
    plot::path_of(f, kind, &at, width, color);
}

/// A dot at (x, y) of a chart, kept to its edges, in a batch of its own.
#[allow(clippy::too_many_arguments)]
fn chart_dot(f: &mut Frame, pl: &Plot, batch: u32, x: f64, y: f64, size: f32, color: Color) {
    let at = pl.at(x, y).max(pl.rect.0).min(pl.rect.1);
    dot_in(f, batch, at, size, color);
}

/// The colour of an output value on the plane: blue below 0, red above, dark at 0.
fn value_color(v: f64) -> Color {
    let t = (v / 1.5).clamp(-1.0, 1.0) as f32;
    let dark = Color::hex(0x161C29);
    if !t.is_finite() {
        return dark;
    }
    // (a ramp that is steep near 0 and gentle near the ends: the sign shows at once)
    let k = 0.66 * t.abs().sqrt();
    if t < 0.0 {
        dark.mix(Color::hex(0x4A84F0), k)
    } else {
        dark.mix(Color::hex(0xE85A50), k)
    }
}

/// A wire's width for its weight.
fn wire_width(v: f64) -> f32 {
    (1.0 + 1.5 * v.abs() as f32).min(7.0)
}

/// How much of step `s` of the film is drawn, 0 to 1, at the film's place `reveal`;
/// and how far its numbers have faded in.
fn progress(reveal: f32, s: usize) -> f32 {
    (reveal - (s as f32 - 1.0)).clamp(0.0, 1.0)
}

fn fade(reveal: f32, s: usize) -> f32 {
    ((progress(reveal, s) - TRAVEL * 0.8) / (1.0 - TRAVEL * 0.8)).clamp(0.0, 1.0)
}

/// Text in a box: a name at its left and a value at its right, on one line (or the
/// value alone, in the middle, if there is no name).
#[allow(clippy::too_many_arguments)]
fn name_value(f: &mut Frame, r: Rect, y: f32, pad: f32, name: &str, value: &str, size: f32, name_color: Color, value_color: Color) {
    let room = (r.1.x - r.0.x) - 2.0 * pad - if name.is_empty() { 0.0 } else { f.text_size(name, size).x + pad };
    // (a number wider than its room, a large power of ten, is drawn smaller)
    let wide = f.text_size(value, size).x;
    let fitted = if wide > room && wide > 0.0 { (size * room / wide).max(size * 0.6) } else { size };
    if name.is_empty() {
        f.text_px(Vec2::new((r.0.x + r.1.x) * 0.5, y), value, fitted, value_color, Align2::CENTER_CENTER);
        return;
    }
    f.text_px(Vec2::new(r.0.x + pad, y), name, size, name_color, Align2::LEFT_CENTER);
    f.text_px(Vec2::new(r.1.x - pad, y), value, fitted, value_color, Align2::RIGHT_CENTER);
}

/// A box's dark backing (in front of the wires) and its edge.
fn boxed(f: &mut Frame, r: Rect, edge: Color) {
    hud_rect(f, r.0, r.1, Color::hex(0x0D1220));
    let c = [r.0, Vec2::new(r.1.x, r.0.y), r.1, Vec2::new(r.0.x, r.1.y), r.0];
    plot::path_of(f, K_BOXES, &c, 1.0, edge);
}

/// A wire: a line behind the boxes (depth-tested, on the screen's farthest sheet), in
/// a batch of its own kind, dashed or not.
#[allow(clippy::too_many_arguments)]
fn wire(f: &mut Frame, kind: u8, a: Vec2, b: Vec2, width: f32, color: Color, dashed: bool) {
    if !(a.is_finite() && b.is_finite()) {
        return;
    }
    let (a, b) = (f.hud_at(0, a), f.hud_at(0, b));
    let fade_power = 60.0 + kind.min(31) as f32;
    let style = if dashed { LineStyle { dash: 7.0, duty: 0.55, depth_bias: 0.0, fade_power, ..Default::default() } } else { LineStyle { depth_bias: 0.0, fade_power, ..Default::default() } };
    f.dl.lines_with(LineKind::Blend, style).segment(a, b, width, color);
}

fn math(v: f64) -> String {
    format!("${}$", sig(v))
}

impl Backprop1 {
    fn draw_network(&mut self, f: &mut Frame, lay: &Layout) {
        let g = &lay.net;
        let (s, pad) = (g.s, g.z.pad);
        let w = self.w;
        let dw = self.drawn;
        let p = self.shown_pass(&w);
        let pd = self.shown_pass(&dw);
        let four = self.mode == Mode::Four;
        let mean = if four { Some(passes_of_four(&w)) } else { None };
        let r = self.reveal;
        // what is known at this point of the film, and what the step being shown works on
        let known = |step: usize| fade(r, step);
        let lit = |step: usize| self.mode == Mode::One && r > step as f32 - 1.0 && (r < step as f32 || self.film == step && self.topic == Topic::Film);
        let pointer = plot::pointer(f);

        // the wires: width by the size of the weight, dashed where it is negative
        for k in 0..9 {
            let (a, b) = g.wires[k];
            let width = wire_width(dw[k]);
            let selected = self.selected == Some(k) || self.hover == Some(k);
            let base = WEIGHT.a(if selected { 0.95 } else { 0.45 });
            let negative = dw[k] < 0.0;
            wire(f, K_WIRES, a, b, width, if negative { Color::CLEAR } else { base }, false);
            wire(f, K_DASHED, a, b, width, if negative { base } else { Color::CLEAR }, true);
        }
        wire(f, K_WIRES, g.to_loss.0, g.to_loss.1, 1.6, LOSS.a(0.55), false);
        wire(f, K_WIRES, g.from_target.0, g.from_target.1, 1.6, LOSS.a(0.55), false);
        // the wires of the step being taken, lit in its direction's colour
        let mut lit_wires = [Color::CLEAR; 9];
        for (step, ks) in STEP_WIRES.iter().enumerate() {
            if lit(step + 1) {
                for &k in *ks {
                    lit_wires[k] = if step < 4 { FORWARD } else { BACKWARD };
                }
            }
        }
        for k in 0..9 {
            let (a, b) = g.wires[k];
            wire(f, K_LIT, a, b, 1.8, lit_wires[k].a(0.85), false);
        }
        // the ones the biases come from
        for one in g.ones {
            hud_rect(f, one - Vec2::splat(ONE_R - 1.0), one + Vec2::splat(ONE_R - 1.0), Color::hex(0x0D1220));
            hud_circle(f, one, ONE_R, 1.2, FORWARD.a(0.6));
            f.text_px(one, "$1$", s - 1.0, FORWARD.a(0.9), Align2::CENTER_CENTER);
        }

        // the boxes
        let edge = |on: bool| if on { pal::TEXT.a(0.85) } else { pal::FAINT };
        let fwd = |a: f32| FORWARD.a(a);
        let back = |a: f32| BACKWARD.a(a);
        let line = |r: Rect, k: f32| r.0.y + 2.0 + g.z.line * (k + 0.5);
        for i in 0..2 {
            boxed(f, g.x[i], edge(lit(1) || lit(9)));
            name_value(f, g.x[i], mid(g.x[i]).y, pad, &format!("$x_{}$", i + 1), &math(p.x[i]), s, pal::DIM, fwd(1.0));
        }
        for j in 0..2 {
            let u = j + 1;
            boxed(f, g.zs[j], edge(lit(1) || lit(2) || lit(8) || lit(9)));
            name_value(f, g.zs[j], line(g.zs[j], 0.0), pad, &format!("$z_{u}$"), &math(p.z[j]), s, pal::DIM, fwd(known(1)));
            if known(8) > 0.0 {
                name_value(f, g.zs[j], line(g.zs[j], 1.0), pad, &format!(r"$\delta_{u}$"), &math(p.dz[j]), s, back(0.8 * known(8)), back(known(8)));
            }
            boxed(f, g.acts[j], edge(lit(2) || lit(3) || lit(6) || lit(7) || lit(8)));
            name_value(f, g.acts[j], line(g.acts[j], 0.0), pad, &format!("$a_{u}$"), &math(p.a[j]), s, pal::DIM, fwd(known(2)));
            if known(7) > 0.0 {
                name_value(f, g.acts[j], line(g.acts[j], 1.0), pad, &format!(r"$\partial L/\partial a_{u}$"), &math(p.da[j]), s, back(0.8 * known(7)), back(known(7)));
            }
            self.draw_glyph(f, g.glyphs[j], pd.z[j], known(1), known(8), lit(2) || lit(8));
        }
        boxed(f, g.out, edge(lit(3) || lit(4) || lit(5) || lit(6) || lit(7)));
        name_value(f, g.out, line(g.out, 0.0), pad, r"$\hat y$", &math(p.out), s, pal::DIM, fwd(known(3)));
        if known(5) > 0.0 {
            name_value(f, g.out, line(g.out, 1.0), pad, r"$\delta$", &math(p.d), s, back(0.8 * known(5)), back(known(5)));
        }
        boxed(f, g.loss, edge(lit(4) || lit(5)));
        match &mean {
            Some((_, l, _)) => {
                name_value(f, g.loss, line(g.loss, 0.0), pad, "$L$", &math(*l), s, pal::DIM, LOSS);
                name_value(f, g.loss, line(g.loss, 1.0), pad, r"$\ell$", &math(p.loss), s, pal::DIM, LOSS.a(0.8));
            }
            None => name_value(f, g.loss, line(g.loss, 0.0), pad, "$L$", &math(p.loss), s, pal::DIM, LOSS.a(known(4))),
        }
        boxed(f, g.target, edge(lit(4)));
        name_value(f, g.target, mid(g.target).y, pad, "$y$", &math(p.y), s, pal::DIM, LOSS);

        // the chips: each weight, and its slope once it is known
        let slope_known = |k: usize| if k >= V1 { known(6) } else { known(9) };
        for k in 0..9 {
            let c = g.chips[k];
            let over = pointer.is_some_and(|q| plot::within(q, c, 2.0));
            let selected = self.selected == Some(k);
            let edge = if selected {
                WEIGHT
            } else if over {
                WEIGHT.a(0.7)
            } else {
                pal::FAINT
            };
            boxed(f, c, edge);
            let name = if g.z.named { format!("${}$", NAMES[k]) } else { String::new() };
            name_value(f, c, line(c, 0.0), pad, &name, &math(w[k]), s, WEIGHT.a(0.8), WEIGHT);
            let a = slope_known(k);
            if a > 0.0 {
                let slope = mean.as_ref().map_or(p.grad[k], |m| m.2[k]);
                let name = match (g.z.named, four) {
                    (false, _) => "",
                    (true, true) => "mean",
                    (true, false) => "slope",
                };
                name_value(f, c, line(c, 1.0), pad, name, &math(slope), s - if g.z.named { 1.0 } else { 0.0 }, back(0.7 * a), back(a));
            }
        }

        // the pulses of the step being taken: forward from a wire's start, back from
        // its end, and for a weight's slope from both ends to the chip
        let mut pulses: Vec<(Vec2, Color)> = Vec::new();
        let step = self.reveal.floor() as usize + 1;
        if self.mode == Mode::One && step <= STEPS && self.reveal < self.film as f32 {
            let t = (progress(self.reveal, step) / TRAVEL).clamp(0.0, 1.0);
            // (eased in and out along the wire, then faded)
            let t = t * t * (3.0 - 2.0 * t);
            let alpha = 1.0 - ((progress(self.reveal, step) - TRAVEL) / (1.0 - TRAVEL)).clamp(0.0, 1.0);
            let lerp = |a: Vec2, b: Vec2, t: f32| a + (b - a) * t;
            let ks: &[usize] = STEP_WIRES[step - 1];
            match step {
                1 | 3 => ks.iter().for_each(|&k| pulses.push((lerp(g.wires[k].0, g.wires[k].1, t), FORWARD.a(alpha)))),
                7 => ks.iter().for_each(|&k| pulses.push((lerp(g.wires[k].1, g.wires[k].0, t), BACKWARD.a(alpha)))),
                6 | 9 => ks.iter().for_each(|&k| {
                    let c = mid(g.chips[k]);
                    pulses.push((lerp(g.wires[k].0, c, t), FORWARD.a(alpha)));
                    pulses.push((lerp(g.wires[k].1, c, t), BACKWARD.a(alpha)));
                }),
                2 | 8 => (0..2).for_each(|j| {
                    let (a, b) = (mid(g.zs[j]), mid(g.acts[j]));
                    pulses.push(if step == 2 { (lerp(a, b, t), FORWARD.a(alpha)) } else { (lerp(b, a, t), BACKWARD.a(alpha)) });
                }),
                4 => {
                    pulses.push((lerp(g.to_loss.0, g.to_loss.1, t), FORWARD.a(alpha)));
                    pulses.push((lerp(g.from_target.0, g.from_target.1, t), LOSS.a(alpha)));
                }
                _ => pulses.push((lerp(g.to_loss.1, g.to_loss.0, t), BACKWARD.a(alpha))),
            }
        }
        for (at, c) in pulses {
            dot_in(f, PULSE_DOTS, at, 11.0, c);
        }
    }

    /// A unit's little graph of tanh: where its sum is on the curve (once it is known),
    /// and (once the slope has come back) the curve's slope there, the gate the slope
    /// goes through.
    fn draw_glyph(&self, f: &mut Frame, r: Rect, z: f64, shown: f32, back_shown: f32, on: bool) {
        hud_rect(f, r.0, r.1, Color::hex(0x0A0E18));
        let inner = (r.0 + Vec2::new(5.0, 4.0), r.1 - Vec2::new(5.0, 4.0));
        let plot = Plot::new(inner, [-4.5, 4.5], [-1.15, 1.15]);
        plot::line_of(f, K_GLYPH, plot.at(-4.5, 0.0), plot.at(4.5, 0.0), 1.0, pal::FAINT);
        plot::line_of(f, K_GLYPH, plot.at(0.0, -1.1), plot.at(0.0, 1.1), 1.0, pal::FAINT);
        let pts: Vec<Vec2> = (0..=32).map(|i| -4.5 + 9.0 * i as f64 / 32.0).map(|v| plot.at(v, v.tanh())).collect();
        plot::path_of(f, K_GLYPH, &pts, 1.4, if on { pal::TEXT.a(0.9) } else { pal::DIM.a(0.8) });
        // the sum, kept to the graph's edge when it is beyond it
        let zc = z.max(-4.5).min(4.5);
        let at = plot.at(zc, z.tanh());
        let slope = 1.0 - z.tanh().powi(2);
        // the tangent: a short piece of the line of slope 1 − a² through the point
        let half = 1.7;
        let (a, b) = (plot.at(zc - half, z.tanh() - half * slope), plot.at(zc + half, z.tanh() + half * slope));
        plot::line_of(f, K_GLYPH, a, b, 1.6, BACKWARD.a(0.9 * back_shown));
        plot::disc(f, at, 7.0, FORWARD.a(shown.max(0.0)));
    }
}

/// The weights whose wires each step of the film works along.
const STEP_WIRES: [&[usize]; STEPS] = [&[W11, W12, B1, W21, W22, B2], &[], &[V1, V2, C], &[], &[], &[V1, V2, C], &[V1, V2], &[], &[W11, W12, B1, W21, W22, B2]];

impl Scene for Backprop1 {
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
        Some("numbers on a wire: click to choose that weight    dots on the plane and the charts: drag    buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A network small enough to show every number: two inputs, two hidden units and one output, with nine weights. Going forward, each unit adds up what comes in on its wires, each times the wire's weight, plus a bias; the hidden units pass that sum through $\\tanh$, and the loss says how far the output is from the target:");
        ui::equation(ui, r"z_j = w_{j1} x_1 + w_{j2} x_2 + b_j, \quad a_j = \tanh z_j, \\ \hat y = v_1 a_1 + v_2 a_2 + c, \quad L = \tfrac12 (\hat y - y)^2");
        ui::note(ui, "Going back, the chain rule hands the loss's slope back along the same wires, a layer at a time, and each weight's slope comes out as what came in on its wire going forward times what came back to the unit it feeds. That is backpropagation; a step downhill then moves every weight against its slope.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The film");
        if self.mode == Mode::One {
            ui.horizontal(|ui| {
                if ui.button("◀ back").clicked() {
                    self.film_back();
                }
                if ui.button("step ▶").clicked() {
                    self.film_step();
                }
                if ui.button(if self.playing { "❚❚ pause" } else { "▶ play" }).clicked() {
                    self.film_play();
                }
            });
            ui::note(ui, "Forward a layer at a time, then back. Each step lights the wires it works along and writes its arithmetic under the network.");
        } else {
            ui::note(ui, "The film goes through one example: choose “one example” below. With four, the units show the pass of the one ringed on the plane (click another to see its own).");
        }

        ui::section(ui, "What is trained on");
        let mut four = (self.mode == Mode::Four) as usize;
        if ui::choice(ui, &["one example", "four examples"], &mut four) {
            self.set_mode(if four == 1 { Mode::Four } else { Mode::One });
        }
        if self.mode == Mode::One {
            let mut changed = false;
            for i in 0..2 {
                let mut v = self.x[i] as f32;
                if ui::slider(ui, &format!("input $x_{}$", i + 1), &mut v, -INPUT_MOST as f32..=INPUT_MOST as f32) {
                    self.x[i] = v as f64;
                    changed = true;
                }
            }
            let mut v = self.y as f32;
            if ui::slider(ui, "target $y$", &mut v, -INPUT_MOST as f32..=INPUT_MOST as f32) {
                self.y = v as f64;
                changed = true;
            }
            if changed {
                self.check = None;
                self.changed_by_hand();
            }
            ui::note(ui, "Or drag the example's dot on the plane, and the target's mark on the scale beside it.");
        } else {
            let names = ["1", "2", "3", "4"];
            ui.horizontal(|ui| {
                ui.label(ui::label_text(ui, "the units show example", 12.5, Some(pal::DIM), false));
                for (i, n) in names.iter().enumerate() {
                    if ui.selectable_label(self.picked == i, *n).clicked() {
                        self.picked = i;
                    }
                }
            });
        }

        ui::section(ui, "The weights");
        let names: Vec<String> = NAMES.iter().map(|n| format!("${n}$")).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut k = self.selected.unwrap_or(W12);
        if ui::choice(ui, &names, &mut k) {
            self.selected = Some(k);
        }
        let mut v = self.w[k] as f32;
        if ui::slider(ui, &format!("${}$", NAMES[k]), &mut v, -DRAG_MOST as f32..=DRAG_MOST as f32) {
            if self.live_check().is_none_or(|c| c.k != k) {
                self.start_check(k);
            }
            self.selected = Some(k);
            self.w[k] = v as f64;
            (self.drawn, self.between) = (self.w, self.w);
            self.training = false;
            self.last = None;
            self.hist.clear();
            self.steps = 0;
            self.topic = Topic::Check;
        }
        ui::note(ui, "Or click a weight's numbers on its wire and drag the dot along the chart of the loss.");
        ui.horizontal_wrapped(|ui| {
            if ui.button("start again").clicked() {
                let mode = self.mode;
                self.restart(mode, START);
            }
            if ui.button("the two units alike").clicked() {
                let mode = self.mode;
                self.restart(mode, alike(&START));
            }
            if ui.button("every weight 0").clicked() {
                let mode = self.mode;
                self.restart(mode, [0.0; 9]);
            }
        });
        ui::note(ui, "Two units that start alike get the same slopes, so they stay alike: the network is then one unit. With every weight 0, every unit gives 0 and hands nothing back.");

        ui::section(ui, "Descent");
        let mut rate = self.rate as f32;
        if ui::slider_log(ui, r"rate $\eta$", &mut rate, 0.01..=3.0) {
            self.rate = rate as f64;
        }
        ui.horizontal(|ui| {
            if ui.button("one step downhill").clicked() {
                self.training = false;
                self.step_downhill();
            }
            if ui.button(if self.training { "stop training" } else { "train" }).clicked() {
                self.toggle_training();
            }
        });
        let mut speed = self.speed as usize;
        if ui::slider_int_log(ui, "steps per second", &mut speed, 1..=200) {
            self.speed = speed as f32;
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (FORWARD, r"cyan: what goes forward: the inputs $x$, each unit's sum $z$ and its $\tanh$, $a$, and the output $\hat y$"),
                (BACKWARD, r"pink: what comes back: the loss's slope with respect to the number beside it ($\delta$ is the slope with respect to $\hat y$, $\delta_j$ to $z_j$); on a wire, with respect to its weight"),
                (WEIGHT, "yellow: the weights, on their wires; a wire is as wide as its weight is large, and dashed where it is negative"),
                (LOSS, r"orange: the target $y$ and the loss $L$"),
                (pal::DIM, r"in each hidden unit: the $\tanh$ curve, the unit's sum on it (the dot) and its slope there (the pink line)"),
                (value_color(1.0), r"the plane: the output $\hat y$ for every input, blue where it is below 0 and red above; a dot is an example, in its target's colour"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Weight, bias:", "the numbers a network learns: a weight multiplies what comes in on a wire, a bias is added to the sum (a weight on a wire that always carries 1).");
        ui::term(ui, r"$\tanh$:", r"a smooth step from $-1$ to $1$; its slope is $1 - \tanh^2$: 1 at the middle, nearly 0 far out on either side.");
        ui::term(ui, "Loss:", r"one number for how wrong the output is: here $L = \tfrac12 (\hat y - y)^2$ on one example, or the mean of that over the four.");
        ui::term(ui, "Slope:", r"the partial derivative $\partial L/\partial w$: how fast the loss changes as one number changes while the others stay put. The nine slopes of the weights together are the gradient.");
        ui::term(ui, "Backpropagation:", "working out every slope in one sweep from the loss back to the inputs, with the chain rule: what reaches a unit from above is multiplied by its own local slope and handed on.");
        ui::term(ui, r"Rate $\eta$:", "how long a step downhill is: each weight moves by minus the rate times its slope.");
        ui::term(ui, "Saturated:", r"a unit whose sum is far from 0, where $\tanh$ is flat: its slope $1 - a^2$ is nearly 0, so almost nothing is handed back through it.");

        ui::section(ui, "Numbers");
        let (loss, grad) = self.loss_and_grad(&self.w);
        // (as formulas, every one alike: a small number is written as a power of ten)
        ui::readout(ui, if self.mode == Mode::Four { "mean loss $L$" } else { "loss $L$" }, math(loss));
        ui::readout(ui, "steps downhill", format!("${}$", self.steps));
        let size = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
        ui::readout(ui, "size of the gradient", math(size));
        if let Some(k) = self.selected {
            ui::readout(ui, &format!("${}$ and its slope", NAMES[k]), format!("${}, {}$", sig(self.w[k]), sig(grad[k])));
        }
        ui::see_also(
            ui,
            &[
                ("autodiff", "the same sweep on any formula"),
                ("chain", "the chain rule, which the backward pass applies"),
                ("backpropmat", "the same, a layer at a time in matrices"),
                ("mlp", "a larger network learning in the plane"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        let (view, inset) = (f.view_size(), f.top_inset);
        let lay = match &self.layout {
            Some((v, i, lay)) if *v == view && *i == inset => lay.clone(),
            _ => {
                let lay = Layout::of(view, inset);
                self.layout = Some((view, inset, lay.clone()));
                lay
            }
        };
        let tiny = lay.view.x < 160.0 || lay.view.y < 160.0;
        self.holding = false;
        if !tiny {
            self.buttons(f, &lay);
            self.handles(f, &lay);
        }
        self.advance(f.dt, f.real_dt);
        if tiny {
            return;
        }
        self.draw_network(f, &lay);
        if !lay.compact {
            self.draw_strip(f, &lay);
            self.draw_plane(f, &lay);
            self.draw_line_chart(f, &lay);
            self.draw_steps_chart(f, &lay);
        }
    }
}

impl Backprop1 {
    /// The buttons over the picture, and clicks on the chips.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let size = lay.small;
        let one = self.mode == Mode::One;
        let film_words = if lay.narrow { ["◀", "step", if self.playing { "pause" } else { "play" }] } else { ["◀ back", "step ▶", if self.playing { "❚❚ pause" } else { "▶ play" }] };
        if one {
            if plot::button(f, lay.film[0], film_words[0], size, false) {
                self.film_back();
            }
            if plot::button(f, lay.film[1], film_words[1], size, false) {
                self.film_step();
            }
            if plot::button(f, lay.film[2], film_words[2], size, self.playing) {
                self.film_play();
            }
        }
        let downhill = if lay.narrow { "downhill" } else { "one step downhill" };
        if plot::button(f, lay.descent[0], downhill, size, false) {
            self.training = false;
            self.step_downhill();
        }
        let train = match (self.training, lay.narrow) {
            (true, true) => "stop",
            (true, false) => "stop training",
            (false, _) => "train",
        };
        if plot::button(f, lay.descent[1], train, size, self.training) {
            self.toggle_training();
        }
        if let Some(modes) = lay.modes {
            if plot::button(f, modes[0], "one example", size, self.mode == Mode::One) {
                self.set_mode(Mode::One);
            }
            if plot::button(f, modes[1], "four examples", size, self.mode == Mode::Four) {
                self.set_mode(Mode::Four);
            }
        }
        // a chip clicked: its weight is chosen (clicked again: no longer)
        let pointer = plot::pointer(f);
        self.hover = pointer.and_then(|p| (0..9).find(|&k| plot::within(p, lay.net.chips[k], 2.0)));
        if self.hover.is_some() {
            f.handles.claim_clicks();
        }
        if f.clicked_canvas() {
            if let Some(k) = self.hover {
                self.selected = if self.selected == Some(k) { None } else { Some(k) };
                if self.topic == Topic::Check && self.live_check().is_some_and(|c| Some(c.k) != self.selected) {
                    self.topic = Topic::Rule;
                }
            }
        }
    }

    /// The handles: the example on the plane, the target on the scale, the selected
    /// weight on its chart; and a click on one of the four examples.
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        if let Some(plane) = lay.plane {
            let pl = Plot::new(plane, [-INPUT_MOST, INPUT_MOST], [-INPUT_MOST, INPUT_MOST]);
            match self.mode {
                Mode::One => {
                    if let Some((a, b)) = pl.handle(f, 1, self.x[0], self.x[1], pal::WHITE) {
                        if a.is_finite() && b.is_finite() {
                            self.x = [a.max(-INPUT_MOST).min(INPUT_MOST), b.max(-INPUT_MOST).min(INPUT_MOST)];
                            self.holding = true;
                            self.check = None;
                            self.changed_by_hand();
                        }
                    }
                }
                Mode::Four => {
                    if f.clicked_canvas() {
                        if let Some(p) = plot::pointer(f) {
                            if let Some(i) = (0..4).find(|&i| (pl.at(FOUR[i].0[0], FOUR[i].0[1]) - p).length() < 14.0) {
                                self.picked = i;
                            }
                        }
                    }
                }
            }
        }
        if let (Some(scale), Mode::One) = (lay.scale, self.mode) {
            // (the bar is narrower than a chart's least width: the handle's chart is wider)
            let sc = Plot::new((scale.0 - Vec2::new(10.0, 0.0), scale.1 + Vec2::new(10.0, 0.0)), [0.0, 1.0], [-INPUT_MOST, INPUT_MOST]);
            if let Some((_, y)) = sc.handle(f, 2, 0.5, self.y, LOSS) {
                if y.is_finite() {
                    self.y = y.max(-INPUT_MOST).min(INPUT_MOST);
                    self.holding = true;
                    self.check = None;
                    self.changed_by_hand();
                }
            }
        }
        if let (Some(chart), Some(k)) = (lay.line_chart, self.selected) {
            let (pl, _) = self.along_weight(chart, k);
            let [lo, hi] = pl.x;
            let wk = self.drawn[k];
            if (lo..=hi).contains(&wk) {
                let mut w = self.drawn;
                w[k] = wk;
                if let Some((v, _)) = pl.handle(f, 3, wk, self.loss_at(&w), WEIGHT) {
                    if v.is_finite() {
                        if self.live_check().is_none_or(|c| c.k != k) {
                            self.start_check(k);
                        }
                        self.w[k] = v.max(-DRAG_MOST).min(DRAG_MOST);
                        self.holding = true;
                        self.training = false;
                        self.last = None;
                        self.hist.clear();
                        self.steps = 0;
                        self.topic = Topic::Check;
                    }
                }
            }
        }
    }

    fn draw_strip(&mut self, f: &mut Frame, lay: &Layout) {
        let said = self.said();
        let (min, max) = lay.strip;
        let s = lay.text;
        let width = max.x - min.x;
        let mut y = min.y + 2.0;
        for line in wrap_words(f, &said.title, lay.small, width) {
            f.text_px(Vec2::new(min.x, y), line, lay.small, pal::DIM, Align2::LEFT_TOP);
            y += lay.small + 5.0;
        }
        y += 4.0;
        // the arithmetic in full where it fits; else each line as its formula and its
        // result only; else as much of that as fits
        let room = max.y - y;
        let full = place_lines(f, &said.lines, s, width);
        let placed = if full.1 <= room + 1.0 {
            full
        } else {
            let short: Vec<Line> = said.lines.iter().map(|l| if l.len() > 2 { vec![l[0].clone(), l[l.len() - 1].clone()] } else { l.clone() }).collect();
            place_lines(f, &short, s, width)
        };
        for (at, piece) in placed.0 {
            if at.y + s <= room + 1.0 {
                f.text_px(Vec2::new(min.x, y) + at, piece, s, pal::TEXT, Align2::LEFT_TOP);
            }
        }
    }

    fn draw_plane(&mut self, f: &mut Frame, lay: &Layout) {
        let (Some(plane), Some(scale)) = (lay.plane, lay.scale) else { return };
        let size = lay.small;
        let pl = Plot::new(plane, [-INPUT_MOST, INPUT_MOST], [-INPUT_MOST, INPUT_MOST]);
        pl.backing(f, 34.0, 30.0, 44.0, scale.1.x - plane.1.x + 24.0);
        let n = if lay.narrow { 24 } else { 32 };
        let dw = self.drawn;
        let cell = Vec2::new(pl.width() / n as f32, pl.height() / n as f32);
        for i in 0..n {
            for j in 0..n {
                let a = -INPUT_MOST + 2.0 * INPUT_MOST * (i as f64 + 0.5) / n as f64;
                let b = INPUT_MOST - 2.0 * INPUT_MOST * (j as f64 + 0.5) / n as f64;
                let out = pass(&dw, [a, b], 0.0).out;
                let at = pl.rect.0 + Vec2::new(i as f32 * cell.x, j as f32 * cell.y);
                plot::fill(f, (at, at + cell), value_color(out));
            }
        }
        pl.x_axis(f, "$x_1$", size);
        pl.y_axis(f, "$x_2$", size);
        let title = match (self.mode, lay.narrow) {
            (Mode::One, false) => r"the output $\hat y$ at every input",
            (Mode::One, true) => r"$\hat y$ at every input",
            (Mode::Four, _) => "the four examples",
        };
        pl.title(f, title, size + 0.5);
        match self.mode {
            Mode::One => {
                // (the example's dot is the handle, drawn by `handles`: its target's colour in it)
                let at = pl.at(self.x[0], self.x[1]);
                plot::disc(f, at, 7.0, value_color(self.y));
            }
            Mode::Four => {
                for (i, (x, y)) in FOUR.iter().enumerate() {
                    let at = pl.at(x[0], x[1]);
                    if i == self.picked {
                        hud_circle(f, at, 11.0, 1.6, pal::WHITE);
                    }
                    plot::disc(f, at, 13.0, pal::WHITE.a(0.9));
                    plot::disc(f, at, 9.0, value_color(*y));
                    let below = if x[1] > 0.0 { -13.0 } else { 13.0 };
                    let align = if x[1] > 0.0 { Align2::CENTER_BOTTOM } else { Align2::CENTER_TOP };
                    f.text_px(at + Vec2::new(0.0, below), format!("{}", i + 1), size, pal::TEXT, align);
                }
            }
        }
        // the output's scale: its colours, numbers to the right; the output at the
        // example (cyan) and the target (orange, a handle with one example) to the left
        let sc = Plot::new(scale, [0.0, 1.0], [-INPUT_MOST, INPUT_MOST]);
        let bands = 40;
        for k in 0..bands {
            let v0 = -INPUT_MOST + 2.0 * INPUT_MOST * k as f64 / bands as f64;
            let v1 = -INPUT_MOST + 2.0 * INPUT_MOST * (k + 1) as f64 / bands as f64;
            plot::fill(f, (Vec2::new(scale.0.x, sc.px_y(v1)), Vec2::new(scale.1.x, sc.px_y(v0))), value_color((v0 + v1) * 0.5));
        }
        for v in [-2.0, -1.0, 0.0, 1.0, 2.0] {
            f.text_px(Vec2::new(scale.1.x + 4.0, sc.px_y(v)), num(v, 0), size - 1.0, pal::DIM, Align2::LEFT_CENTER);
        }
        let p = self.shown_pass(&dw);
        let (yo, yt) = (sc.px_y(p.out.max(-INPUT_MOST).min(INPUT_MOST)), sc.px_y(p.y));
        plot::line_of(f, plot::MARKS, Vec2::new(scale.0.x - 7.0, yo), Vec2::new(scale.1.x, yo), 2.2, FORWARD);
        plot::line_of(f, plot::MARKS, Vec2::new(scale.0.x - 7.0, yt), Vec2::new(scale.1.x, yt), 2.2, LOSS);
        // (the two names kept apart when the marks are close)
        let apart = (size + 2.0 - (yo - yt).abs()).max(0.0) * 0.5;
        let (no, nt) = if yo >= yt { (yo + apart, yt - apart) } else { (yo - apart, yt + apart) };
        f.text_px(Vec2::new(scale.0.x - 10.0, no), r"$\hat y$", size, FORWARD, Align2::RIGHT_CENTER);
        f.text_px(Vec2::new(scale.0.x - 10.0, nt), "$y$", size, LOSS, Align2::RIGHT_CENTER);
    }

    /// The chart of the loss along weight `k`, the other weights held as drawn: its
    /// plot, and the loss at its samples.
    fn along_weight(&self, chart: Rect, k: usize) -> (Plot, Vec<(f64, f64)>) {
        let (lo, hi) = (self.centre.0 - 1.0, self.centre.0 + 1.0);
        let n = 61;
        let dw = self.drawn;
        let pts: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let v = lo + (hi - lo) * i as f64 / (n - 1) as f64;
                let mut w = dw;
                w[k] = v;
                (v, self.loss_at(&w))
            })
            .collect();
        // (its height follows what it shows, from frame to frame, as evenly as that moves;
        // but a loss near 0 is not magnified without end)
        let top = (pts.iter().map(|p| p.1).filter(|v| v.is_finite()).fold(0.0f64, f64::max) * 1.15).max(LEAST_TOP);
        (Plot::new(chart, [lo, hi], [0.0, top]), pts)
    }

    /// Where the steps' chart goes: its own place, or on a narrow canvas the one chart's
    /// place, with four examples (where the loss along a line is the less telling).
    fn steps_place(&self, lay: &Layout) -> Option<Rect> {
        lay.steps_chart.or(lay.line_chart.filter(|_| lay.narrow && self.mode == Mode::Four && self.selected.is_none()))
    }

    fn draw_line_chart(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(chart) = lay.line_chart else { return };
        if lay.steps_chart.is_none() && self.steps_place(lay).is_some() {
            return;
        }
        let dw = self.drawn;
        let size = lay.small;
        match self.selected {
            Some(k) => {
                // the loss along weight k, the others held where they are
                let (pl, pts) = self.along_weight(chart, k);
                let [lo, hi] = pl.x;
                let along = |v: f64| {
                    let mut w = dw;
                    w[k] = v;
                    self.loss_at(&w)
                };
                pl.backing(f, 40.0, 30.0, 44.0, 10.0);
                let title = if lay.narrow { format!("the loss along ${}$", NAMES[k]) } else { format!("the loss along ${}$, the other weights held", NAMES[k]) };
                pl.title(f, &title, size + 0.5);
                axes_in(f, &pl, K_A_AXES, &format!("${}$", NAMES[k]), "$L$", size);
                clamped_path(f, &pl, K_A_CURVE, &pts, 2.0, LOSS);
                let check = self.live_check().filter(|c| c.k == k);
                let (w0, l0, s0) = match check {
                    Some(c) => (c.base[k], c.loss, c.slope),
                    None => {
                        let (l, g) = self.loss_and_grad(&dw);
                        (dw[k], l, g[k])
                    }
                };
                // the slope's straight line through where the check began, across the
                // chart; or a short piece of it through where the weight is (it turns as
                // the weight moves); as far as it stays inside the chart, and not at all
                // while the weight is beyond the chart
                if (lo..=hi).contains(&w0) {
                    let reach = if check.is_some() { (w0 - lo, hi - w0) } else { (0.5, 0.5) };
                    let (a, b) = tangent_ends(&pl, w0, l0, s0, reach.0.min(w0 - lo), reach.1.min(hi - w0));
                    plot::dashed_of(f, K_A_MARKS, a, b, 1.6, BACKWARD.a(0.9));
                }
                let wk = dw[k];
                if check.is_some() {
                    chart_dot(f, &pl, A_DOTS, w0, l0, 7.0, pal::DIM);
                    // what the slope foretold, and the gap to what happened
                    let foretold = l0 + s0 * (wk - w0);
                    chart_dot(f, &pl, A_DOTS, wk, foretold, 7.0, BACKWARD);
                    plot::line_of(f, K_A_MARKS, pl.at(wk, foretold), pl.at(wk, along(wk)), 1.2, pal::DIM);
                }
                // (the weight's own dot is the handle, drawn on the curve: here only when
                // it is beyond the chart and cannot be one)
                if !(lo..=hi).contains(&wk) {
                    chart_dot(f, &pl, A_DOTS, wk, along(wk), 9.0, WEIGHT);
                }
            }
            None => {
                // the loss along the step downhill: the last step's, while the weights
                // are where it left them, or the next one's
                let (from, grad, rate, last) = match self.last_step().filter(|_| !self.training) {
                    Some(l) => (l.before, l.grad, l.rate, true),
                    None => {
                        let (_, g) = self.loss_and_grad(&dw);
                        (dw, g, self.rate, false)
                    }
                };
                let most = (rate * 1.6).max(1.2);
                let n = 81;
                let at_rate = |e: f64| self.loss_at(&descend(&from, &grad, e));
                let pts: Vec<(f64, f64)> = (0..n).map(|i| most * i as f64 / (n - 1) as f64).map(|e| (e, at_rate(e))).collect();
                let l0 = pts[0].1;
                let landed = at_rate(rate);
                let top = (l0.max(landed) * 1.25).max(LEAST_TOP);
                let pl = Plot::new(chart, [0.0, most], [0.0, top]);
                pl.backing(f, 40.0, 30.0, 44.0, 10.0);
                let title = match (last, lay.narrow) {
                    (true, false) => "the loss along the last step, for steps of every length",
                    (false, false) => "the loss along the next step, for steps of every length",
                    (true, true) => "the loss along the last step",
                    (false, true) => "the loss along the next step",
                };
                pl.title(f, title, size + 0.5);
                axes_in(f, &pl, K_B_AXES, r"rate $\eta$", "$L$", size);
                clamped_path(f, &pl, K_B_CURVE, &pts, 2.0, LOSS);
                // the slopes foretell the first part: the loss falls at the rate |g|²
                let g2: f64 = grad.iter().map(|g| g * g).sum();
                let (a, b) = tangent_ends(&pl, 0.0, l0, -g2, 0.0, most);
                plot::dashed_of(f, K_B_MARKS, a, b, 1.6, BACKWARD.a(0.9));
                plot::dashed_of(f, K_B_MARKS, pl.at(0.0, l0), pl.at(most, l0), 1.0, pal::DIM.a(0.6));
                plot::dashed_of(f, K_B_MARKS, pl.at(rate, 0.0).max(pl.rect.0).min(pl.rect.1), pl.at(rate, top).max(pl.rect.0).min(pl.rect.1), 1.0, WEIGHT.a(0.6));
                chart_dot(f, &pl, B_DOTS, 0.0, l0, 7.0, pal::DIM);
                chart_dot(f, &pl, B_DOTS, rate, landed, 9.0, WEIGHT);
                if !pl.tiny() {
                    // (above the line, just after the start, where the curve has gone
                    // below it; the rate at the top of its line, on the side away from
                    // the curve's rise)
                    let name = if lay.narrow { "start" } else { "the loss before the step" };
                    f.text_px(pl.at(0.0, l0) + Vec2::new(10.0, -3.0), name, size - 1.0, pal::DIM, Align2::LEFT_BOTTOM);
                    let x = pl.px_x(rate).min(pl.rect.1.x);
                    let left = x - pl.rect.0.x > 0.3 * pl.width();
                    let (dx, align) = if left { (-5.0, Align2::RIGHT_TOP) } else { (5.0, Align2::LEFT_TOP) };
                    f.text_px(Vec2::new(x + dx, pl.rect.0.y + 2.0), format!(r"$\eta = {}$", sig(rate)), size, WEIGHT, align);
                }
            }
        }
    }

    fn draw_steps_chart(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(chart) = self.steps_place(lay) else { return };
        let pts = self.hist.points();
        let every = self.hist.every() as f64;
        let top = if pts.len() < 2 { 1.0 } else { pts.iter().cloned().filter(|v| v.is_finite()).fold(0.0f32, f32::max).max(1e-3) as f64 * 1.1 };
        // (the steps' axis glides out as they are taken, rather than stepping)
        let pl = Plot::new(chart, [0.0, self.span.1], [0.0, top]);
        pl.backing(f, 40.0, 30.0, 44.0, 10.0);
        pl.title(f, if self.mode == Mode::Four { "the mean loss over the steps of descent" } else { "the loss over the steps of descent" }, lay.small + 0.5);
        axes_in(f, &pl, K_STEPS_AXES, "steps", "$L$", lay.small);
        // The loss as far as the drawn weights have got, at a fixed number of places
        // along it (so that the curve can be followed from frame to frame as it grows
        // between the steps), and a dot where it has got to.
        if pts.len() >= 2 {
            let end = self.taken_drawn();
            let at = |x: f64| {
                let i = (x / every).max(0.0);
                let k = (i.floor() as usize).min(pts.len() - 2);
                let t = (i - k as f64).clamp(0.0, 1.0);
                pts[k] as f64 + (pts[k + 1] as f64 - pts[k] as f64) * t
            };
            let n = 160;
            let curve: Vec<(f64, f64)> = (0..n).map(|j| end * j as f64 / (n - 1) as f64).map(|x| (x, at(x))).collect();
            clamped_path(f, &pl, K_HISTORY, &curve, 2.0, LOSS);
            chart_dot(f, &pl, NOW_DOT, end, at(end), 7.0, LOSS);
        }
        if pts.len() < 2 {
            f.text_px(mid(pl.rect), "no steps taken yet", lay.small, pal::DIM, Align2::CENTER_CENTER);
        }
    }
}

/// Where the pieces of the lines of arithmetic go, `width` pixels wide: each line's
/// pieces one after another, broken between pieces (the rest indented), and the lines
/// side by side, a little apart, while they fit. Returns each piece's place from the
/// top-left corner, and the height taken.
fn place_lines(f: &Frame, lines: &[Line], s: f32, width: f32) -> (Vec<(Vec2, String)>, f32) {
    let (gap, apart, pitch) = (6.0, 26.0, s + 9.0);
    let mut placed = Vec::new();
    let (mut x, mut y) = (0.0f32, 0.0f32);
    for (n, line) in lines.iter().enumerate() {
        let widths: Vec<f32> = line.iter().map(|p| f.text_size(p, s).x).collect();
        let whole = widths.iter().sum::<f32>() + gap * widths.len().saturating_sub(1) as f32;
        // a line starts beside the last if it fits there whole, else on a line of its own
        if n > 0 {
            if x > 0.0 && x + apart + whole <= width {
                x += apart;
            } else {
                (x, y) = (0.0, y + pitch);
            }
        }
        for (k, (piece, w)) in line.iter().zip(&widths).enumerate() {
            if k > 0 && x + gap + w > width {
                (x, y) = (18.0, y + pitch);
            } else if k > 0 {
                x += gap;
            }
            placed.push((Vec2::new(x, y), piece.clone()));
            x += w;
        }
    }
    let height = if lines.is_empty() { 0.0 } else { y + pitch };
    (placed, height)
}

/// Words broken into lines no wider than `width` pixels (formulas kept whole).
fn wrap_words(f: &Frame, text: &str, size: f32, width: f32) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut words = Vec::new();
    let (mut start, mut inside) = (0, false);
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
    for word in words {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The loss of one example as a plain function of the nine weights, the inputs and
    /// the target, written out afresh (not through `pass`).
    fn loss_plain(w: &Weights, x: [f64; 2], y: f64) -> f64 {
        let a1 = (w[0] * x[0] + w[1] * x[1] + w[2]).tanh();
        let a2 = (w[3] * x[0] + w[4] * x[1] + w[5]).tanh();
        let out = w[6] * a1 + w[7] * a2 + w[8];
        0.5 * (out - y) * (out - y)
    }

    fn random_weights(rng: &mut Rng, scale: f64) -> Weights {
        let mut w = [0.0; 9];
        for v in &mut w {
            *v = rng.range(-1.0, 1.0) as f64 * scale;
        }
        w
    }

    /// Every slope the scene shows, against a central difference of the same loss:
    /// the nine weights' and the in-between ones (δ, ∂L/∂aⱼ, δⱼ), for one example and
    /// for the mean of the four, at many random weights, inputs and targets.
    #[test]
    fn every_slope_agrees_with_a_difference_of_the_loss() {
        let mut rng = Rng::new(7);
        let h = 1e-5;
        for trial in 0..300 {
            let w = random_weights(&mut rng, if trial % 3 == 0 { 3.0 } else { 1.0 });
            let x = [rng.range(-2.0, 2.0) as f64, rng.range(-2.0, 2.0) as f64];
            let y = rng.range(-2.0, 2.0) as f64;
            let p = pass(&w, x, y);
            assert!((p.loss - loss_plain(&w, x, y)).abs() < 1e-12);
            for k in 0..9 {
                let (mut up, mut down) = (w, w);
                up[k] += h;
                down[k] -= h;
                let diff = (loss_plain(&up, x, y) - loss_plain(&down, x, y)) / (2.0 * h);
                assert!((diff - p.grad[k]).abs() < 1e-7 * (1.0 + diff.abs()), "trial {trial}, weight {k}: {} against {diff}", p.grad[k]);
            }
            // the in-between slopes: the loss as a function of the output, of a tanh, of a sum
            let out_loss = |o: f64| 0.5 * (o - y) * (o - y);
            let diff = (out_loss(p.out + h) - out_loss(p.out - h)) / (2.0 * h);
            assert!((diff - p.d).abs() < 1e-7 * (1.0 + diff.abs()));
            for j in 0..2 {
                let a_loss = |aj: f64| {
                    let mut a = p.a;
                    a[j] = aj;
                    out_loss(w[V1] * a[0] + w[V2] * a[1] + w[C])
                };
                let diff = (a_loss(p.a[j] + h) - a_loss(p.a[j] - h)) / (2.0 * h);
                assert!((diff - p.da[j]).abs() < 1e-7 * (1.0 + diff.abs()), "trial {trial}: ∂L/∂a{j}");
                let z_loss = |zj: f64| a_loss(zj.tanh());
                let diff = (z_loss(p.z[j] + h) - z_loss(p.z[j] - h)) / (2.0 * h);
                assert!((diff - p.dz[j]).abs() < 1e-7 * (1.0 + diff.abs()), "trial {trial}: δ{j}");
            }
            // the mean over the four
            let (_, mean, grad) = passes_of_four(&w);
            let four = |w: &Weights| FOUR.iter().map(|(x, y)| loss_plain(w, *x, *y)).sum::<f64>() / 4.0;
            assert!((mean - four(&w)).abs() < 1e-12);
            for k in 0..9 {
                let (mut up, mut down) = (w, w);
                up[k] += h;
                down[k] -= h;
                let diff = (four(&up) - four(&down)) / (2.0 * h);
                assert!((diff - grad[k]).abs() < 1e-7 * (1.0 + diff.abs()), "trial {trial}, four, weight {k}");
            }
        }
    }

    /// Every weight's slope is what came in on its wire times what came back.
    #[test]
    fn each_slope_is_what_came_in_times_what_came_back() {
        let mut rng = Rng::new(3);
        for _ in 0..50 {
            let w = random_weights(&mut rng, 2.0);
            let p = pass(&w, [rng.range(-2.0, 2.0) as f64, rng.range(-2.0, 2.0) as f64], rng.range(-2.0, 2.0) as f64);
            for k in 0..9 {
                let ((_, vin), (_, vback)) = wire_ends(k, &p);
                assert_eq!(vin * vback, p.grad[k], "weight {k}");
            }
        }
    }

    #[test]
    fn numbers_are_written_to_three_figures() {
        assert_eq!(sig(0.5), "0.500");
        assert_eq!(sig(-0.3), "−0.300");
        assert_eq!(sig(1.0), "1.00");
        assert_eq!(sig(0.42189), "0.422");
        assert_eq!(sig(-0.00139), "−0.00139");
        assert_eq!(sig(0.9996), "1.00");
        assert_eq!(sig(9.9996), "10.0");
        assert_eq!(sig(123.4), "123");
        assert_eq!(sig(0.000695), "6.95e-4");
        assert_eq!(sig(-0.000695), "−6.95e-4");
        assert_eq!(sig(0.0), "0");
        assert_eq!(sig(f64::NAN), "?");
        assert_eq!(shown(-0.000695), -0.000695);
        assert_eq!(shown(0.42189), 0.422);
        assert_eq!(factor(-0.3), "(−0.300)");
        assert_eq!(rel(0.5, 0.5001), "=");
        assert_eq!(rel(0.5, 0.51), r"\approx");
    }

    /// The numbers written in the film's lines are the numbers in the network, to the
    /// digits shown, and "=" stands only where the written numbers give the written result.
    #[test]
    fn the_films_lines_say_the_networks_numbers() {
        let mut rng = Rng::new(11);
        let mut checks = 0;
        for trial in 0..200 {
            let w = if trial == 0 { START } else { random_weights(&mut rng, 2.0) };
            let (x, y) = if trial == 0 { (X0, Y0) } else { ([rng.range(-2.0, 2.0) as f64, rng.range(-2.0, 2.0) as f64], rng.range(-2.0, 2.0) as f64) };
            let p = pass(&w, x, y);
            let has = |lines: &[Line], v: f64| lines.iter().flatten().any(|piece| piece.contains(&sig(v)));
            let results: [(usize, Vec<f64>); 9] = [
                (1, vec![p.z[0], p.z[1]]),
                (2, vec![p.a[0], p.a[1]]),
                (3, vec![p.out]),
                (4, vec![p.loss]),
                (5, vec![p.d]),
                (6, vec![p.grad[V1], p.grad[V2], p.grad[C]]),
                (7, vec![p.da[0], p.da[1]]),
                (8, vec![p.gate[0], p.gate[1], p.dz[0], p.dz[1]]),
                (9, vec![p.grad[W11], p.grad[W12], p.grad[B1], p.grad[W21], p.grad[W22], p.grad[B2]]),
            ];
            for (step, values) in results {
                let lines = film_lines(step, &w, &p);
                for v in values {
                    assert!(has(&lines, v), "trial {trial}, step {step}: {} not in {lines:?}", sig(v));
                    checks += 1;
                }
                // every line ends with the true value, after "=" or "≈"
                for line in &lines {
                    let last = line.last().expect("a line");
                    assert!(last.contains('=') || last.contains(r"\approx") || last.contains(&sig(p.grad[C])), "{last}");
                }
            }
        }
        assert!(checks > 1000);
        // at the scene's start, "≈" only where the caption of the film says so, and
        // where the figures shown give the figure after them exactly, "="
        let p = pass(&START, X0, Y0);
        let about: Vec<String> = (1..=9).flat_map(|step| film_lines(step, &START, &p)).flatten().filter(|piece| piece.contains(r"\approx")).collect();
        assert!(about.iter().any(|a| a.contains("−0.280")), "{about:?}");
        assert!(about.len() <= 4, "{about:?}");
        assert_eq!(film_lines(8, &START, &p)[0].last().map(String::as_str), Some(r"$\approx −0.280$"));
    }

    /// The first experiment's caption: the numbers it quotes.
    #[test]
    fn the_film_says_what_its_caption_says() {
        let mut s = Backprop1::create();
        s.set_up(0);
        assert!(s.playing && s.film == 0 && s.mode == Mode::One);
        let p = s.shown_pass(&s.w);
        assert_eq!(sig(p.loss), "0.118");
        assert_eq!(sig(p.d), "−0.486");
        assert_eq!(sig(p.da[0]), "−0.340");
        assert_eq!(sig(p.gate[0]), "0.822");
        assert_eq!(sig(p.dz[0]), "−0.280");
        assert_eq!(sig(p.grad[W12]), "−0.140");
        assert_eq!(p.grad[W12], p.x[1] * p.dz[0]);
        assert_eq!(p.x[1], 0.5);
        // it plays to its end, a step at a time, and stops there
        let mut seen = vec![0];
        for _ in 0..(60.0 * 20.0) as usize {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
            if seen.last() != Some(&s.film) {
                seen.push(s.film);
            }
        }
        assert_eq!(seen, (0..=STEPS).collect::<Vec<_>>());
        assert!(!s.playing && s.reveal == STEPS as f32 && !s.moving());
        assert!(TRIALS[0].caption.contains("0.118") && TRIALS[0].caption.contains("-0.486") && TRIALS[0].caption.contains("-0.340"));
    }

    /// The nudge: what the slope foretold, and what happened.
    #[test]
    fn the_nudge_does_what_its_caption_says() {
        let mut s = Backprop1::create();
        s.set_up(1);
        let c = s.live_check().expect("a check");
        assert_eq!(c.k, W12);
        assert!((s.w[W12] - c.base[W12] - 0.1).abs() < 1e-12);
        assert_eq!(sig(c.slope), "−0.140");
        let foretold = c.slope * 0.1;
        let measured = s.loss_at(&s.w) - c.loss;
        assert_eq!((sig(foretold), sig(measured)), ("−0.0140".to_string(), "−0.0133".to_string()));
        assert_eq!(format!("{:.4}", c.loss), "0.1182");
        assert_eq!(format!("{:.4}", s.loss_at(&s.w)), "0.1049");
        // close in, better; further out, worse
        let at = |nudge: f64| {
            let mut w = c.base;
            w[W12] += nudge;
            ((loss_of(Mode::One, &w, X0, Y0) - c.loss) / (c.slope * nudge) - 1.0).abs()
        };
        assert!(at(0.01) < at(0.1) && at(0.1) < at(0.5) && at(0.5) < at(1.0), "{} {} {} {}", at(0.01), at(0.1), at(0.5), at(1.0));
        assert!(at(0.001) < 1e-3);
        // the strip says the same numbers
        let said = s.said();
        let text: String = said.lines.iter().flatten().cloned().collect();
        assert!(text.contains("−0.0140") && text.contains("−0.0133"), "{text}");
    }

    /// The saturated unit: its numbers, and that it hardly learns while unit 2 does.
    #[test]
    fn the_saturated_unit_does_what_its_caption_says() {
        let mut s = Backprop1::create();
        s.set_up(2);
        let p = s.shown_pass(&s.w);
        assert_eq!(sig(p.z[0]), "−4.00");
        assert_eq!(sig(p.gate[0]), "0.00134");
        assert_eq!(sig(p.da[0]), "−1.04");
        assert_eq!(sig(p.dz[0]), "−0.00139");
        assert_eq!(sig(p.loss), "1.10");
        for k in [W11, W12, B1] {
            assert!(p.grad[k].abs() < 0.0014, "{k}: {}", p.grad[k]);
        }
        assert_eq!(format!("{:.0}", (p.dz[1] / p.dz[0]).abs()), "614");
        assert_eq!(s.selected, Some(W11));
        // a step moves unit 1's weights hardly at all, unit 2's plainly
        let before = s.w;
        s.step_downhill();
        assert!((s.w[W11] - before[W11]).abs() < 5e-4 && (s.w[W21] - before[W21]).abs() > 0.2);
        // the loss along w11 is nearly flat: over the chart's width it changes by little
        let along = |v: f64| {
            let mut w = before;
            w[W11] = v;
            loss_of(Mode::One, &w, X0, Y0)
        };
        assert!((along(-1.5) - along(-3.5)).abs() < 0.05 * along(-2.5), "{} {}", along(-1.5), along(-3.5));
    }

    /// The step downhill: where the rate 0.1 lands, and the longer steps.
    #[test]
    fn the_step_does_what_its_caption_says() {
        let mut s = Backprop1::create();
        s.set_up(3);
        let l = s.last_step().expect("a step");
        assert_eq!(l.rate, 0.1);
        assert_eq!((sig(l.loss.0), sig(l.loss.1)), ("0.118".to_string(), "0.0628".to_string()));
        let along = |e: f64| loss_of(Mode::One, &descend(&l.before, &l.grad, e), X0, Y0);
        assert_eq!(sig(along(1.0)), "0.303");
        // lowest near 0.3
        let best = (1..=200).map(|i| i as f64 * 0.01).min_by(|a, b| along(*a).total_cmp(&along(*b))).unwrap_or(0.0);
        assert!((best - 0.37).abs() < 0.006 && along(best) < 1e-4, "lowest at {best}");
        // higher than the start from 0.76 on (and below it before)
        let crossing = (1..=2000).map(|i| i as f64 * 0.001).find(|e| along(*e) > l.loss.0).unwrap_or(0.0);
        assert!((crossing - 0.76).abs() < 0.005, "worse from {crossing}");
        assert!((1..=20).all(|i| along(crossing + i as f64 * 0.05) > l.loss.0));
    }

    /// Training on the four points: the plateau, the drop, and when it has them.
    #[test]
    fn training_does_what_its_caption_says() {
        let mut s = Backprop1::create();
        s.set_up(4);
        assert!(s.training && s.mode == Mode::Four && s.rate == 0.3);
        let (_, l0, _) = passes_of_four(&s.w);
        assert_eq!(sig(l0), "0.803");
        let mut losses = vec![l0];
        let mut has_it = None;
        for step in 1..=400 {
            s.train_step();
            let (passes, loss, _) = passes_of_four(&s.w);
            losses.push(loss);
            if has_it.is_none() && passes.iter().all(|p| (p.out - p.y).abs() < 0.1) {
                has_it = Some(step);
            }
        }
        assert_eq!(has_it, Some(106));
        // 0.80 to 0.50 within twenty steps, then near 0.50 (within 0.03) for a while
        assert!(losses[20] < 0.51 && losses[20] > 0.49, "{}", losses[20]);
        assert!(losses[10..45].iter().all(|l| (l - 0.5).abs() < 0.03), "{:?}", &losses[10..45]);
        // the output is about 0 everywhere on the plateau
        let mut t = Backprop1::create();
        t.set_up(4);
        for _ in 0..30 {
            t.train_step();
        }
        let (passes, _, _) = passes_of_four(&t.w);
        assert!(passes.iter().all(|p| p.out.abs() < 0.15), "{:?}", passes.map(|p| p.out));
        assert!(losses[400] < 1e-6);
    }

    /// Two units that start alike stay alike, and the loss never gets below a third;
    /// with every weight 0 nothing moves at all.
    #[test]
    fn the_starts_that_do_not_learn_do_not() {
        let mut s = Backprop1::create();
        s.set_up(4);
        s.restart(Mode::Four, alike(&START));
        let mut lowest = f64::MAX;
        for _ in 0..6000 {
            s.train_step();
            let w = s.w;
            assert!(w[W11] == w[W21] && w[W12] == w[W22] && w[B1] == w[B2] && w[V1] == w[V2]);
            lowest = lowest.min(passes_of_four(&w).1);
        }
        assert!(lowest > 1.0 / 3.0 && lowest < 1.0 / 3.0 + 2e-3, "{lowest}");
        s.restart(Mode::Four, [0.0; 9]);
        let (_, loss, grad) = passes_of_four(&s.w);
        assert!(grad.iter().all(|g| *g == 0.0) && loss == 0.5);
    }

    /// The caption of the last experiment, word for word where it quotes numbers.
    #[test]
    fn the_captions_quote_the_numbers_the_tests_hold() {
        assert!(TRIALS[1].caption.contains("-0.0140") && TRIALS[1].caption.contains("-0.0133") && TRIALS[1].caption.contains("0.1182 to 0.1049"));
        assert!(TRIALS[2].caption.contains("0.00134") && TRIALS[2].caption.contains("-0.00139") && TRIALS[2].caption.contains("614"));
        assert!(TRIALS[3].caption.contains("0.0628") && TRIALS[3].caption.contains("0.303") && TRIALS[3].caption.contains("0.76") && TRIALS[3].caption.contains("0.37"));
        assert!(TRIALS[4].caption.contains("106 steps"));
    }

    /// Nothing that a slider, a drag or a long run can do leaves a number that is not
    /// finite: weights at their ends, a rate far too large, inputs and target at theirs.
    #[test]
    fn degenerate_settings_stay_finite() {
        let mut s = Backprop1::create();
        for mode in [Mode::One, Mode::Four] {
            s.restart(mode, [DRAG_MOST; 9]);
            s.x = [INPUT_MOST, -INPUT_MOST];
            s.y = -INPUT_MOST;
            s.rate = 3.0;
            for _ in 0..500 {
                s.train_step();
                assert!(s.w.iter().all(|v| v.is_finite() && v.abs() <= HELD_AT));
            }
            let said = s.said();
            assert!(!said.title.is_empty());
            for step in 1..=9 {
                let p = s.shown_pass(&s.w);
                for line in film_lines(step, &s.w, &p) {
                    assert!(line.iter().all(|piece| !piece.contains('?')), "{line:?}");
                }
            }
        }
        // a run that diverges is held, not carried to infinity
        let w = descend(&[1.0; 9], &[f64::INFINITY; 9], 1.0);
        assert!(w.iter().all(|v| v.is_finite()));
    }

    /// The layout: everything inside the window, and no box of the network on another,
    /// on every canvas from a phone's to a large desktop's.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1075.0, 1000.0), (1600.0, 1000.0), (868.0, 700.0), (580.0, 700.0), (390.0, 752.0), (374.0, 640.0), (700.0, 480.0), (300.0, 470.0), (240.0, 180.0), (1300.0, 900.0), (1075.0, 1400.0), (600.0, 400.0), (800.0, 600.0), (2400.0, 1300.0)] {
            let lay = Layout::of(Vec2::new(w, h), 0.0);
            let inside = |r: Rect| r.0.x >= -0.5 && r.0.y >= -0.5 && r.1.x <= w + 0.5 && r.1.y <= h + 0.5;
            let boxes = lay.net.boxes();
            if lay.compact {
                // (with the whole height to itself, a network in a window not far too
                // small fits)
                assert!(lay.net.fits || w < 480.0 || h < 300.0, "{w}x{h}: the network does not fit");
                continue;
            }
            for (i, a) in boxes.iter().enumerate() {
                assert!(inside(*a), "{w}x{h}: box {i} {a:?} outside");
                for (j, b) in boxes.iter().enumerate().skip(i + 1) {
                    assert!(!overlap(*a, *b), "{w}x{h}: boxes {i} and {j} overlap: {a:?} {b:?}");
                }
            }
            let ext = lay.net.extent();
            assert!(lay.net.fits && ext.1.y <= lay.film[0].0.y && ext.0.y >= lay.net.room.0.y - 0.5, "{w}x{h}: the network {ext:?} runs into the buttons");
            for r in [lay.plane, lay.scale, lay.line_chart, lay.steps_chart].into_iter().flatten() {
                assert!(inside(r) && r.0.y >= lay.strip.1.y, "{w}x{h}: {r:?}");
            }
        }
    }
}

//! Backpropagation through attention: one head on a row of six tokens, a gradient
//! arriving at one position's output, and the two roads it goes back by. Through the
//! values it is shared out in proportion to the attention (what to move); through the
//! softmax it tells each score whether to rise or fall (where to look), and that road
//! closes as the head becomes sure.

use super::*;
use crate::math::autograd::Tape;
use crate::scenes::stats::plot::{self, Plot};

pub const DEF: SceneDef = SceneDef {
    id: "attngrad",
    title: "Backpropagation through attention",
    group: GROUP,
    blurb: "One attention head has two roads for the gradient, through what is moved and through where it looks, and the second closes when attention is sure.",
    make: |_| Box::new(AttnGrad::new()),
};

// ----- the head -------------------------------------------------------------------------

/// Tokens in the row.
const T: usize = 6;
/// Numbers in a token (the width of the stream the head reads).
const D: usize = 4;
/// Numbers in a query, a key and a value (the head's width): two, so that the values
/// can be drawn in a plane. (The scores are divided by its square root, as the
/// library's heads divide theirs.)
const DH: usize = 2;

/// What the head reads at each position. (Chosen, with the weights below, so that the
/// values lie round a circle and the last position spreads its attention unevenly:
/// see the test `the_head_is_as_its_comments_say`.)
const TOKENS: [[f64; D]; T] = [
    [-0.82, 1.14, 1.17, -1.03],
    [1.27, -1.52, -0.20, 0.78],
    [-1.04, 0.44, -0.11, 0.44],
    [0.58, -0.49, -1.38, 0.60],
    [-0.44, 0.84, -0.92, -0.06],
    [0.06, 0.10, 0.74, 0.14],
];

/// The query, key and value weights: one row per number of the query (key, value),
/// as the library stores a layer's weights. The query weights are multiplied by the
/// sharpness β before use.
const W_Q: [[f64; D]; DH] = [[0.80, -0.60, 0.54, 0.51], [-0.40, 0.53, 1.34, 0.45]];
const W_K: [[f64; D]; DH] = [[0.8, 1.2, 0.0, 1.3], [0.6, -0.4, 0.5, 0.2]];
const W_V: [[f64; D]; DH] = [[-0.3, -0.9, -0.8, -0.7], [-1.1, -1.1, 0.5, 0.8]];

/// The two roads back, both in the course's colour for gradients: the value road in
/// it, the pattern road in a lighter tint of it, so that the two can be told apart on
/// the chart and in what reaches each token.
const VALUE_ROAD: Color = BACKWARD;
const PATTERN_ROAD: Color = Color::hex(0xFF9CC2);

/// The sharpness knob's range: β multiplies the query weights, and so every score.
const BETA_MAX: f64 = 10.0;
/// The gradient arriving at the output: longest allowed.
const G_MOST: f64 = 1.5;

/// The roles in which a token is read, in the order the picture lists them.
const VALUE: usize = 0;
const KEY: usize = 1;
const QUERY: usize = 2;

/// Everything one forward and backward pass of the head gives, read off the tape.
/// The loss is L = g · o_dest: the gradient arriving at the output of position `dest`
/// is g, and nothing arrives at the other positions. (In a transformer the head's
/// output is then multiplied by its output weights and added to the stream; what comes
/// back from the stream reaches the mix through those weights, transposed, and arrives
/// here as g. Scene tfgrad follows it there.)
#[derive(Clone, Debug, Default)]
struct Pass {
    dest: usize,
    beta: f64,
    g: [f64; DH],
    q: [[f64; DH]; T],
    k: [[f64; DH]; T],
    v: [[f64; DH]; T],
    /// The scores q_i · k_j / √d (above the diagonal worked out, but masked).
    scores: [[f64; T]; T],
    /// The attention: row i the causal softmax of row i of the scores.
    a: [[f64; T]; T],
    /// The mix o_i = Σ_j a_ij v_j (the head's output before its output weights).
    o: [[f64; DH]; T],
    /// The gradients of L, each in the shape of what it is the gradient of.
    dv: [[f64; DH]; T],
    da: [[f64; T]; T],
    ds: [[f64; T]; T],
    dq: [[f64; DH]; T],
    dk: [[f64; DH]; T],
    /// What reaches each token in each of its roles (value, key, query): the gradient
    /// at the copy of the tokens that role reads.
    roles: [[[f64; D]; T]; 3],
    /// The gradients of the value, key and query weights (in that order).
    dw: [[[f64; D]; DH]; 3],
    /// The softmax's Jacobian for the row of `dest`: entry (j, l) is ∂a_dj / ∂s_dl,
    /// from the tape (one backward sweep for each j); zero beyond `dest`.
    jac: [[f64; T]; T],
}

/// The length of a vector.
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Pass {
    /// The size of the gradient that enters the value road: |∂L/∂V|, over every
    /// position's value.
    fn value_road(&self) -> f64 {
        norm(self.dv.as_flattened())
    }

    /// The size of the gradient that enters the pattern road, past the softmax:
    /// |∂L/∂s| over the scores of the destination's row (the only row that gets any).
    fn pattern_road(&self) -> f64 {
        norm(&self.ds[self.dest])
    }

    /// The whole gradient reaching token j: the three roles added up.
    fn total(&self, j: usize) -> [f64; D] {
        std::array::from_fn(|c| self.roles[VALUE][j][c] + self.roles[KEY][j][c] + self.roles[QUERY][j][c])
    }

    /// g · o_dest, the loss's value (how far along g the mix lies).
    fn g_dot_mix(&self) -> f64 {
        dot(&self.g, &self.o[self.dest])
    }
}

/// One pass of the head: the forward pass recorded on `tape` as the library's
/// transformer records each of its heads (Q = x W_qᵀ, K = x W_kᵀ, V = x W_vᵀ, the
/// scores Q Kᵀ / √d, their causal softmax, the mix A V), then the gradient g seeded at
/// the mix of position `dest` and swept back. The tokens are given to the head as three
/// copies, one for each role, so that the gradient at each copy is what that role
/// hands back (their sum is the gradient a single copy would get: see the tests). With
/// `jacobian`, also the softmax's Jacobian for that row, by one more sweep per source.
fn run(tape: &mut Tape, dest: usize, beta: f64, g: [f64; DH], jacobian: bool) -> Pass {
    let dest = dest.min(T - 1);
    let flat = |m: &[[f64; D]], scale: f64| -> [f64; DH * D] { std::array::from_fn(|e| scale * m[e / D][e % D]) };
    tape.clear();
    let x = TOKENS.as_flattened();
    let xv = tape.constant(T, D, x);
    let xk = tape.constant(T, D, x);
    let xq = tape.constant(T, D, x);
    let wv = tape.param(DH, D, &flat(&W_V, 1.0));
    let wk = tape.param(DH, D, &flat(&W_K, 1.0));
    let wq = tape.param(DH, D, &flat(&W_Q, beta));
    let q = tape.matmul_t(xq, wq);
    let k = tape.matmul_t(xk, wk);
    let v = tape.matmul_t(xv, wv);
    let s = tape.matmul_t(q, k);
    let s = tape.scale(s, 1.0 / (DH as f64).sqrt());
    let a = tape.softmax_rows_causal(s);
    let o = tape.matmul(a, v);
    let mut seed = [0.0; T * DH];
    seed[dest * DH..(dest + 1) * DH].copy_from_slice(&g);
    tape.backward_seeded(o, &seed);

    let mut p = Pass { dest, beta, g, ..Default::default() };
    let rows = |dst: &mut [[f64; DH]; T], src: &[f64]| {
        for (row, chunk) in dst.iter_mut().zip(src.chunks_exact(DH)) {
            row.copy_from_slice(chunk);
        }
    };
    let square = |dst: &mut [[f64; T]; T], src: &[f64]| {
        for (row, chunk) in dst.iter_mut().zip(src.chunks_exact(T)) {
            row.copy_from_slice(chunk);
        }
    };
    let wide = |dst: &mut [[f64; D]; T], src: &[f64]| {
        for (row, chunk) in dst.iter_mut().zip(src.chunks_exact(D)) {
            row.copy_from_slice(chunk);
        }
    };
    let weights = |dst: &mut [[f64; D]; DH], src: &[f64]| {
        for (row, chunk) in dst.iter_mut().zip(src.chunks_exact(D)) {
            row.copy_from_slice(chunk);
        }
    };
    rows(&mut p.q, tape.value(q));
    rows(&mut p.k, tape.value(k));
    rows(&mut p.v, tape.value(v));
    square(&mut p.scores, tape.value(s));
    square(&mut p.a, tape.value(a));
    rows(&mut p.o, tape.value(o));
    rows(&mut p.dv, tape.grad(v));
    square(&mut p.da, tape.grad(a));
    square(&mut p.ds, tape.grad(s));
    rows(&mut p.dq, tape.grad(q));
    rows(&mut p.dk, tape.grad(k));
    wide(&mut p.roles[VALUE], tape.grad(xv));
    wide(&mut p.roles[KEY], tape.grad(xk));
    wide(&mut p.roles[QUERY], tape.grad(xq));
    weights(&mut p.dw[VALUE], tape.grad(wv));
    weights(&mut p.dw[KEY], tape.grad(wk));
    weights(&mut p.dw[QUERY], tape.grad(wq));
    if jacobian {
        for j in 0..=dest {
            let mut seed = [0.0; T * T];
            seed[dest * T + j] = 1.0;
            tape.backward_seeded(a, &seed);
            p.jac[j].copy_from_slice(&tape.grad(s)[dest * T..(dest + 1) * T]);
        }
    }
    p
}

/// Points of the chart: β from 0 to `BETA_MAX`.
const SAMPLES: usize = 41;

fn beta_at(k: usize) -> f64 {
    BETA_MAX * k as f64 / (SAMPLES - 1) as f64
}

/// The size of the gradient entering each road (value, pattern) at every sample of β,
/// for a destination and a gradient: what the chart draws.
fn curve(tape: &mut Tape, dest: usize, g: [f64; DH]) -> Vec<[f64; 2]> {
    (0..SAMPLES)
        .map(|k| {
            let p = run(tape, dest, beta_at(k), g, false);
            [p.value_road(), p.pattern_road()]
        })
        .collect()
}

// ----- what the picture shows -----------------------------------------------------------

/// Where each of the picture's numbers is kept in the one list that glides
/// (`plot::Glide`): the destination's place, β, g, the destination's mix, its row of
/// attention, the steps a descent would take on its scores, the gradients of the
/// values, the sizes of what reaches each token in each role, the softmax's Jacobian,
/// the whole pattern, the studied token's three roles and their sum, the chart's two
/// curves, and which token is studied. (The chart's top glides on its own: see
/// `AttnGrad::top`.)
mod at {
    use super::{D, SAMPLES, T};
    pub const DEST: usize = 0;
    pub const BETA: usize = 1;
    pub const G: usize = 2;
    pub const MIX: usize = 4;
    pub const ATTN: usize = 6;
    pub const STEP: usize = ATTN + T;
    pub const DV: usize = STEP + T;
    pub const ROLE: usize = DV + 2 * T;
    pub const JAC: usize = ROLE + 3 * T;
    pub const PATTERN: usize = JAC + T * T;
    pub const SPLIT: usize = PATTERN + T * T;
    pub const CURVE: usize = SPLIT + 4 * D;
    pub const STUDY: usize = CURVE + 2 * SAMPLES;
    pub const LEN: usize = STUDY + 1;
}

/// The picture's numbers as drawn (gliding to the pass's), read by name.
struct Shown<'a>(&'a [f32]);

impl Shown<'_> {
    fn get(&self, i: usize) -> f32 {
        self.0.get(i).copied().filter(|v| v.is_finite()).unwrap_or(0.0)
    }
    fn dest(&self) -> f32 {
        self.get(at::DEST)
    }
    fn beta(&self) -> f32 {
        self.get(at::BETA)
    }
    fn g(&self) -> Vec2 {
        Vec2::new(self.get(at::G), self.get(at::G + 1))
    }
    fn mix(&self) -> Vec2 {
        Vec2::new(self.get(at::MIX), self.get(at::MIX + 1))
    }
    fn attn(&self, j: usize) -> f32 {
        self.get(at::ATTN + j).max(0.0).min(1.0)
    }
    fn step(&self, j: usize) -> f32 {
        self.get(at::STEP + j)
    }
    fn dv(&self, j: usize) -> Vec2 {
        Vec2::new(self.get(at::DV + 2 * j), self.get(at::DV + 2 * j + 1))
    }
    fn role(&self, j: usize, r: usize) -> f32 {
        self.get(at::ROLE + 3 * j + r)
    }
    fn jac(&self, j: usize, l: usize) -> f32 {
        self.get(at::JAC + j * T + l)
    }
    fn pattern(&self, i: usize, j: usize) -> f32 {
        self.get(at::PATTERN + i * T + j).max(0.0).min(1.0)
    }
    fn curve(&self, k: usize, road: usize) -> f32 {
        self.get(at::CURVE + 2 * k + road)
    }
    fn study(&self) -> f32 {
        self.get(at::STUDY)
    }
}

/// The chart's top: a round number a little above both curves.
fn chart_top(curve: &[[f64; 2]]) -> f64 {
    let most = curve.iter().flatten().filter(|v| v.is_finite()).fold(0.0f64, |a, &b| a.max(b));
    let want = (most * 1.08).max(0.05);
    let power = 10f64.powf(want.log10().floor());
    let lead = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0].into_iter().find(|l| l * power >= want * (1.0 - 1e-9)).unwrap_or(10.0);
    lead * power
}

// ----- experiments ----------------------------------------------------------------------

/// The parts of the picture besides the diagram. On a wide canvas the plane of values
/// is beside the diagram and the other four are in a row under them (or, if the row
/// does not fit, one at a time, chosen by tabs); on a narrow one, all five are taken
/// one at a time under the diagram. An experiment chooses the one it is about.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Card {
    Values,
    Pattern,
    Jacobian,
    Chart,
    Roles,
}

const CARDS: [Card; 5] = [Card::Values, Card::Pattern, Card::Jacobian, Card::Chart, Card::Roles];

impl Card {
    fn index(self) -> usize {
        CARDS.iter().position(|c| *c == self).unwrap_or(0)
    }

    fn tab(self) -> &'static str {
        match self {
            Card::Values => "values",
            Card::Pattern => "attention",
            Card::Jacobian => "Jacobian",
            Card::Chart => "chart",
            Card::Roles => "roles",
        }
    }
}

struct Trial {
    label: &'static str,
    look_for: &'static str,
    /// The destination, from 0; the sharpness (where a film starts, if there is one);
    /// the gradient; the token studied; the card it is about.
    dest: usize,
    beta: f64,
    g: [f64; DH],
    study: usize,
    card: Card,
    /// Whether β then plays up to `BETA_MAX`.
    film: bool,
}

/// The gradient the scene opens with, and the one most experiments use.
const G0: [f64; DH] = [0.0, 0.8];
/// The gradient of the experiments on the pattern road.
const G1: [f64; DH] = [0.65, 0.4];

const TRIALS: [Trial; 5] = [
    Trial {
        label: "the value road: the gradient shared out by attention",
        look_for: r#"$g = (0, 0.8)$ arrives at the output of position 6, the mix $o_6 = \sum_j a_{6j} v_j$. Each value gets its share, $\partial L/\partial v_j = a_{6j}\,g$: the pink arcs under the tokens are as wide as the cyan ones over them, and in the plane each value carries a copy of $g$ scaled by its attention. $x_2$, given 42 % of the attention, gets 42 % of $g$; $x_1$ and $x_5$ get 7 % each; together the shares make up $g$. From each value the gradient goes on through $W_V^\top$ to its token (the row "as a value") and to the value weights, $|\partial L/\partial W_V| = 0.662$. This road teaches the head what to move: a step on $W_V$ changes the values, never where the head looks."#,
        dest: 5,
        beta: 1.0,
        g: G0,
        study: 1,
        card: Card::Values,
        film: false,
    },
    Trial {
        label: "the pattern road: look more at what would help",
        look_for: r#"Would more of $x_j$ help? The slope of the loss for its attention is $\partial L/\partial a_{6j} = g \cdot v_j$: $-0.88$ for $x_1$, whose value lies furthest against $g$, and $+0.71$ for $x_2$. The softmax turns these into slopes for the scores, $a_{6j}(g \cdot v_j - g \cdot o_6)$, measured from the mix's own $g \cdot o_6 = 0.23$: a step of descent raises the scores of the values beyond the dashed line, better than the mix (bars up), and lowers the rest. $x_2$, the head's favourite, lies on the wrong side: its bar goes furthest down, to $-0.201$. $x_1$ is the best, yet $x_6$'s bar is the taller (0.126 against 0.078): each slope is multiplied by the attention already paid, 0.20 against 0.07. The bars add up to 0: raising every score alike would change nothing."#,
        dest: 5,
        beta: 1.0,
        g: G1,
        study: 0,
        card: Card::Values,
        film: false,
    },
    Trial {
        label: "a head that is sure stops learning where to look",
        look_for: r#"$\beta$ plays from 0 to 10, and the attention goes from even ($1/6$ each) to 0.9993 on $x_2$. The softmax's Jacobian, $\operatorname{diag}(a) - a a^\top$, shrinks to nothing (its largest entry from 0.14 to 0.0006), and the pattern road closes with it: the scores' gradient falls from 0.230 to 0.0004, all the way down the chart. The value road widens instead, from 0.327 to 0.799, nearly $|g| = 0.8$: $v_2$ now gets almost all of $g$. The head goes on changing what it moves, but has stopped learning where to look, though $x_5$ would be better: moving attention from $x_2$ to $x_5$ would lower the loss at a rate of 1.40 per unit."#,
        dest: 5,
        beta: 0.0,
        g: G0,
        study: 5,
        card: Card::Chart,
        film: true,
    },
    Trial {
        label: "nothing flows to the future",
        look_for: r#"The gradient now arrives at position 3. A causal head looks only at the positions up to its own: the scores above the diagonal (dark in the attention grid) are left out of the softmax, so they get no gradient, and neither do $x_4$, $x_5$ and $x_6$, exactly 0 in every role. Only $x_1$, $x_2$ and $x_3$ are reached. Turned round: what reaches a token comes from its own output and from the outputs after it, never from one before it."#,
        dest: 2,
        beta: 1.0,
        g: G1,
        study: 4,
        card: Card::Pattern,
        film: false,
    },
    Trial {
        label: "one token, three roles",
        look_for: r#"$x_6$ is read three times: as a value and as a key, like every source of position 6 (it attends to itself, $a_{66} = 0.20$), and as the query of position 6, which meets all six keys. So its gradient comes back three ways, through $W_V^\top$, $W_K^\top$ and $W_Q^\top$, of sizes 0.251, 0.126 and 0.365. They are vectors and add up, entry by entry, to the whole gradient at $x_6$, $(-0.279, -0.117, 0.212, 0.022)$, of size only 0.370: in part they cancel. (The tape gives the same whole when one copy of $x_6$ feeds all three roles.) Every other source, $x_2$ say, gets only the first two: the query's share is the destination's alone."#,
        dest: 5,
        beta: 1.0,
        g: G1,
        study: 5,
        card: Card::Roles,
        film: false,
    },
];

/// How long the sharpness takes to play from 0 to `BETA_MAX` (seconds of the scene's
/// time), and where it is after `t` of them (eased in and out).
const FILM: f32 = 4.0;

fn film_beta(t: f32) -> f64 {
    let s = (t / FILM).max(0.0).min(1.0) as f64;
    BETA_MAX * s * s * (3.0 - 2.0 * s)
}

// ----- the scene ------------------------------------------------------------------------

pub struct AttnGrad {
    tape: Tape,
    /// What is asked for: the destination (from 0), the sharpness, the gradient, the
    /// token whose three roles are shown.
    dest: usize,
    beta: f64,
    g: [f64; DH],
    study: usize,
    /// The sharpness playing up to `BETA_MAX`: the seconds it has played.
    film: Option<f32>,
    /// The pass at the settings, and the chart's curves for the destination and g.
    pass: Pass,
    pass_for: Option<(usize, u64, [u64; DH])>,
    curve: Vec<[f64; 2]>,
    curve_for: Option<(usize, [u64; DH])>,
    /// The picture as drawn, gliding to the pass's numbers.
    shown: plot::Glide,
    /// The hand moved something this frame: the picture follows it exactly.
    snap: bool,
    moving: bool,
    card: Card,
    /// The column of the diagram under the pointer.
    hover: Option<usize>,
    /// Which cards were shown in the last frame.
    showing: [bool; 5],
    /// The chart's top as drawn, gliding to the round number above its curves.
    top: plot::Glide,
}

impl AttnGrad {
    fn new() -> AttnGrad {
        let t = &TRIALS[0];
        let mut scene = AttnGrad {
            tape: Tape::new(),
            dest: t.dest,
            beta: t.beta,
            g: t.g,
            study: t.study,
            film: None,
            pass: Pass::default(),
            pass_for: None,
            curve: Vec::new(),
            curve_for: None,
            shown: plot::Glide::default(),
            snap: false,
            moving: false,
            card: t.card,
            hover: None,
            showing: [false; 5],
            top: plot::Glide::default(),
        };
        scene.refresh();
        scene.shown.snap(&scene.target());
        scene.top.snap(&[chart_top(&scene.curve) as f32]);
        scene
    }

    /// Set up experiment `k` of `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k.min(TRIALS.len() - 1)];
        self.set_dest(t.dest);
        self.set_beta(t.beta);
        self.set_g(t.g);
        self.set_study(t.study);
        self.card = t.card;
        self.film = t.film.then_some(0.0);
    }

    fn set_dest(&mut self, d: usize) {
        self.dest = d.min(T - 1);
    }

    fn set_beta(&mut self, b: f64) {
        self.beta = if b.is_finite() { b.max(0.0).min(BETA_MAX) } else { 1.0 };
        self.film = None;
    }

    /// A new gradient, no longer than `G_MOST` (and 0 for one that is not a number).
    fn set_g(&mut self, g: [f64; DH]) {
        let g = if g.iter().all(|v| v.is_finite()) { g } else { [0.0; DH] };
        let size = norm(&g);
        self.g = if size > G_MOST { g.map(|v| v * G_MOST / size) } else { g };
    }

    fn set_study(&mut self, j: usize) {
        self.study = j.min(T - 1);
    }

    /// Work out the pass, and the chart's curves, again if what they are for changed.
    fn refresh(&mut self) {
        let key = (self.dest, self.beta.to_bits(), self.g.map(f64::to_bits));
        if self.pass_for != Some(key) {
            self.pass = run(&mut self.tape, self.dest, self.beta, self.g, true);
            self.pass_for = Some(key);
        }
        let key = (self.dest, self.g.map(f64::to_bits));
        if self.curve_for != Some(key) {
            self.curve = curve(&mut self.tape, self.dest, self.g);
            self.curve_for = Some(key);
        }
    }

    /// The picture's numbers for the settings (see `at`).
    fn target(&self) -> Vec<f32> {
        let p = &self.pass;
        let d = p.dest;
        let mut out = vec![0.0f32; at::LEN];
        out[at::DEST] = d as f32;
        out[at::STUDY] = self.study as f32;
        out[at::BETA] = p.beta as f32;
        out[at::G] = p.g[0] as f32;
        out[at::G + 1] = p.g[1] as f32;
        out[at::MIX] = p.o[d][0] as f32;
        out[at::MIX + 1] = p.o[d][1] as f32;
        for j in 0..T {
            out[at::ATTN + j] = p.a[d][j] as f32;
            out[at::STEP + j] = -p.ds[d][j] as f32;
            out[at::DV + 2 * j] = p.dv[j][0] as f32;
            out[at::DV + 2 * j + 1] = p.dv[j][1] as f32;
            for r in 0..3 {
                out[at::ROLE + 3 * j + r] = norm(&p.roles[r][j]) as f32;
            }
            for l in 0..T {
                out[at::JAC + j * T + l] = p.jac[j][l] as f32;
                out[at::PATTERN + j * T + l] = p.a[j][l] as f32;
            }
        }
        let total = p.total(self.study);
        for c in 0..D {
            for r in 0..3 {
                out[at::SPLIT + r * D + c] = p.roles[r][self.study][c] as f32;
            }
            out[at::SPLIT + 3 * D + c] = total[c] as f32;
        }
        for (k, [v, s]) in self.curve.iter().enumerate().take(SAMPLES) {
            out[at::CURVE + 2 * k] = *v as f32;
            out[at::CURVE + 2 * k + 1] = *s as f32;
        }
        out
    }

    /// Move everything on: the film by `dt` seconds of the scene's time, the picture's
    /// glide by `real_dt` of the clock's (a change of setting glides even while the
    /// scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        if let Some(t) = self.film {
            let t = t + if dt.is_finite() { dt.max(0.0) } else { 0.0 };
            self.beta = film_beta(t);
            self.film = (t < FILM).then_some(t);
        }
        self.refresh();
        let target = self.target();
        let real_dt = if real_dt.is_finite() { real_dt.max(0.0) } else { 0.0 };
        if self.snap {
            self.shown.snap(&target);
            self.moving = false;
        } else {
            self.moving = self.shown.follow(&target, real_dt, false);
        }
        // (the chart's top glides even while the hand moves g: it goes up and down in
        // steps, and the whole chart would leap with it)
        self.moving |= self.top.follow(&[chart_top(&self.curve) as f32], real_dt, false);
        self.snap = false;
    }

    fn at_rest(&self) -> bool {
        !self.moving && self.film.is_none()
    }
}

// ----- where things go ------------------------------------------------------------------

type Rect = plot::Rect;

/// The value plane shows [−R, R] each way.
const R: f32 = 1.75;
/// The pattern road's bars reach the edge of their band at this slope.
const BAR_FULL: f32 = 0.35;
/// The roles' cells are fully tinted at this size.
const ROLE_FULL: f32 = 0.6;

#[derive(Clone, Debug)]
struct Lay {
    /// A narrow canvas: the band names above the bands, and every card one at a time.
    narrow: bool,
    text: f32,
    small: f32,
    pad: f32,
    /// The heading: top-left corner and width, and whether it is the long one.
    heading: (Vec2, f32),
    long_heading: bool,
    /// The diagram's columns: the left edge of the first and their width; where the
    /// bands' names end (on a wide canvas, beside the bands).
    x0: f32,
    col: f32,
    names: Option<f32>,
    /// The bands of the diagram, top and bottom each, and where the attention's
    /// numbers and the bars' numbers are written.
    arcs: (f32, f32),
    tokens: (f32, f32),
    attn: f32,
    road: (f32, f32),
    bars: (f32, f32),
    bar_numbers: f32,
    roles: Option<(f32, f32)>,
    /// Lines above the bands for their names (narrow only).
    above: [f32; 4],
    /// The cards' rectangles, their words included (see `Card`), and the tabs that
    /// choose one where they are shown one at a time.
    cards: [Option<Rect>; 5],
    tabs: Vec<(Card, Rect)>,
}

/// The room under the plane for the note on what its marks mean.
const UNDER_PLANE: f32 = 60.0;
/// The room over a card for its title.
const OVER_PLANE: f32 = 20.0;

impl Lay {
    fn of(view: Vec2, inset: f32, card: Card) -> Lay {
        let view = view.max(Vec2::ONE);
        let narrow = view.x < 760.0;
        let (text, small) = if narrow { (11.5, 10.5) } else { (12.5, 11.5) };
        let pad = if narrow { 10.0 } else { 16.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 10.0 };
        let mut lay = Lay {
            narrow,
            text,
            small,
            pad,
            heading: (Vec2::new(pad, top), view.x - 2.0 * pad),
            long_heading: false,
            x0: pad,
            col: 1.0,
            names: None,
            arcs: (0.0, 0.0),
            tokens: (0.0, 0.0),
            attn: 0.0,
            road: (0.0, 0.0),
            bars: (0.0, 0.0),
            bar_numbers: 0.0,
            roles: None,
            above: [0.0; 4],
            cards: [None; 5],
            tabs: Vec::new(),
        };
        if view.x < 200.0 || view.y < 150.0 {
            return lay;
        }
        if narrow {
            lay.narrow_layout(view, top, card);
        } else {
            lay.wide_layout(view, top, card);
        }
        lay
    }

    /// Beside each other: the diagram and the plane, and the other cards in a row under
    /// them.
    fn wide_layout(&mut self, view: Vec2, top: f32, card: Card) {
        let pad = self.pad;
        // the long heading if it takes one line, else the short one
        self.long_heading = heading_lines(true, self.heading.1, self.text) <= 1.0;
        let y = top + heading_lines(self.long_heading, self.heading.1, self.text) * (self.text + 4.0) + 10.0;
        let left_w = ((view.x - 2.0 * pad) * 0.57).floor();
        let names_w = (left_w * 0.24).max(116.0).min(150.0);
        self.x0 = pad + names_w;
        self.col = ((left_w - names_w) / T as f32).floor();
        self.names = Some(self.x0 - 10.0);
        // the bands, at a scale k that leaves the cards under them some 230 pixels
        let fixed = 2.0 + 30.0 + 18.0 + 12.0 + 15.0 + 12.0 + 18.0 + 3.0 * 18.0;
        let k = ((view.y - 8.0 - y - 22.0 - 250.0 - fixed) / 216.0).max(0.5).min(1.5);
        self.arcs = (y, y + 76.0 * k);
        self.tokens = (self.arcs.1 + 2.0, self.arcs.1 + 32.0);
        self.attn = self.tokens.1 + 9.0;
        self.road = (self.tokens.1 + 18.0, self.tokens.1 + 18.0 + 48.0 * k);
        self.bars = (self.road.1 + 12.0, self.road.1 + 12.0 + 92.0 * k);
        self.bar_numbers = self.bars.1 + 7.0;
        let roles_top = self.bars.1 + 15.0 + 12.0 + 18.0;
        self.roles = Some((roles_top, roles_top + 3.0 * 18.0));
        let diagram_bottom = roles_top + 3.0 * 18.0;
        // the plane, square, right of the diagram, its words above and below it
        let right = (pad + left_w + 30.0, view.x - pad);
        let side = (right.1 - right.0).min(diagram_bottom - y - OVER_PLANE - UNDER_PLANE + 20.0).floor();
        let mut bottom_of_top = diagram_bottom;
        if side >= 120.0 {
            let x = right.0 + ((right.1 - right.0 - side) * 0.5).floor();
            self.cards[Card::Values.index()] = Some((Vec2::new(x, y), Vec2::new(x + side, y + OVER_PLANE + side + UNDER_PLANE)));
            bottom_of_top = bottom_of_top.max(y + OVER_PLANE + side + UNDER_PLANE);
        }
        // the other cards: all four if they fit, else the one chosen and tabs
        let cards_top = bottom_of_top + 18.0;
        let height = (view.y - 8.0 - cards_top).min(360.0);
        if height < 150.0 {
            return;
        }
        let width = view.x - 2.0 * pad;
        let grid = (height - 2.0 * self.small - 50.0).min(160.0);
        let widths = [grid + 30.0, grid * 1.15 + 10.0];
        let rest = width - widths[0] - widths[1] - 3.0 * 26.0;
        if rest >= 520.0 {
            let chart = (rest * 0.53).floor();
            let mut x = pad;
            for (k, w) in [widths[0], widths[1], chart, rest - chart].into_iter().enumerate() {
                self.cards[k + 1] = Some((Vec2::new(x, cards_top), Vec2::new(x + w, cards_top + height)));
                x += w + 26.0;
            }
        } else {
            let shown = if card == Card::Values { Card::Chart } else { card };
            let rows = plot::even_row(4, pad, pad + (width * 0.6).min(420.0), cards_top, 24.0, 5.0);
            self.tabs = CARDS[1..].iter().copied().zip(rows).collect();
            let body = (Vec2::new(pad, cards_top + 34.0), Vec2::new(view.x - pad, cards_top + height));
            if body.1.y - body.0.y >= 110.0 {
                self.cards[shown.index()] = Some(body);
            }
        }
    }

    /// One under another: the diagram, then one card chosen by the tabs.
    fn narrow_layout(&mut self, view: Vec2, top: f32, card: Card) {
        let pad = self.pad;
        let line = self.small + 5.0;
        let y = top + heading_lines(false, self.heading.1, self.text) * (self.text + 4.0) + 6.0;
        self.x0 = pad;
        self.col = ((view.x - 2.0 * pad) / T as f32).floor();
        self.names = None;
        // the bands, at a scale that leaves the cards some 330 pixels, and taller where
        // the columns are wide (or the arcs would lie flat)
        let tall = (self.col / 60.0).max(1.0).min(1.6);
        let fixed = 26.0 + 15.0 + 3.0 * line + 14.0 + 10.0;
        let k = ((view.y - 8.0 - y - fixed - 330.0) / (150.0 * tall)).max(0.45).min(1.0) * tall;
        // (a canvas too short for even the smallest diagram shows the heading alone)
        if y + fixed + 150.0 * k > view.y - 4.0 {
            self.col = 0.0;
            return;
        }
        self.above[0] = y;
        self.arcs = (y + line, y + line + 50.0 * k);
        self.tokens = (self.arcs.1 + 2.0, self.arcs.1 + 28.0);
        self.attn = self.tokens.1 + 8.0;
        self.above[1] = self.tokens.1 + 17.0;
        self.road = (self.above[1] + line + 6.0, self.above[1] + line + 6.0 + 36.0 * k);
        self.above[2] = self.road.1 + 6.0;
        self.bars = (self.above[2] + line, self.above[2] + line + 64.0 * k);
        self.bar_numbers = self.bars.1 + 7.0;
        let y = self.bars.1 + 22.0;
        if view.y - 8.0 - (y + 30.0) >= 110.0 {
            let rows = plot::even_row(5, pad, view.x - pad, y, 24.0, 4.0);
            self.tabs = CARDS.iter().copied().zip(rows).collect();
            self.cards[card.index()] = Some((Vec2::new(pad, y + 34.0), Vec2::new(view.x - pad, view.y - 8.0)));
        }
    }

    fn centre(&self, j: f32) -> f32 {
        self.x0 + (j + 0.5) * self.col
    }

    /// The column of the diagram's upper part (arcs to bars) under a point.
    fn column(&self, p: Vec2) -> Option<usize> {
        let j = ((p.x - self.x0) / self.col).floor();
        (p.y >= self.arcs.0 && p.y <= self.bars.1 + 10.0 && j >= 0.0 && j < T as f32).then_some(j as usize)
    }

    /// The column of the roles' rows under a point.
    fn role_column(&self, p: Vec2) -> Option<usize> {
        let (a, b) = self.roles?;
        let j = ((p.x - self.x0) / self.col).floor();
        (p.y >= a && p.y <= b && j >= 0.0 && j < T as f32).then_some(j as usize)
    }
}

// ----- drawing --------------------------------------------------------------------------

/// Line kinds with a batch each (see `plot::apart`): the forward arcs, the value road's
/// arcs and heads, the arrows in the plane, the outlines that move (the destination's
/// ring), the dashed line in the plane and the dashed curve of the chart.
const ARCS: u8 = plot::OWN;
const ROAD: u8 = plot::OWN + 1;
const ARROWS: u8 = plot::OWN + 2;
const RINGS: u8 = plot::OWN + 3;
const DASHED: u8 = plot::OWN + 4;
const PATTERN_CURVE: u8 = plot::OWN + 5;

const KINDS: [(u8, bool); 10] = [
    (plot::AXES, false),
    (plot::CURVES, false),
    (plot::MARKS, false),
    (plot::BUTTONS, false),
    (ARCS, false),
    (ROAD, false),
    (ARROWS, false),
    (RINGS, false),
    (DASHED, true),
    (PATTERN_CURVE, false),
];

/// Every batch of lines, made in the same order in every frame.
fn in_order(f: &mut Frame) {
    for (kind, dashed) in KINDS {
        plot::apart(f, kind, dashed);
    }
}

/// A batch with nothing in it this frame is given a point that cannot be seen: one left
/// empty is dropped, and every batch after it would be compared, point for point, with
/// the one before it in the last frame.
fn keep_batches(f: &mut Frame) {
    for (kind, dashed) in KINDS {
        let batch = plot::apart(f, kind, dashed);
        if batch.pts.len() <= 1 {
            batch.segment(Vec3::ZERO, Vec3::ZERO, 0.0, Color::CLEAR);
        }
    }
}

/// Points along an arc from `from` to `to` (both on the line y = `base`), rising
/// `height` above it (or falling below it, with `down`). Where the two ends meet, the
/// arc is a loop `loop_w` wide, so that a position's attention to itself shows.
const ARC_PTS: usize = 25;

fn arc(from: f32, to: f32, base: f32, height: f32, loop_w: f32, down: bool) -> [Vec2; ARC_PTS] {
    let dir = if down { 1.0 } else { -1.0 };
    let near = (1.0 - (to - from).abs() / loop_w.max(1e-3)).max(0.0);
    let c = loop_w * 0.6 * near;
    let p0 = Vec2::new(from, base);
    let p3 = Vec2::new(to, base);
    let p1 = Vec2::new(from - c, base + dir * height * 1.33);
    let p2 = Vec2::new(to + c, base + dir * height * 1.33);
    std::array::from_fn(|i| {
        let t = i as f32 / (ARC_PTS - 1) as f32;
        let u = 1.0 - t;
        p0 * (u * u * u) + p1 * (3.0 * u * u * t) + p2 * (3.0 * u * t * t) + p3 * (t * t * t)
    })
}

/// How wide an arc of attention `a` is drawn over the tokens.
fn arc_width(a: f32) -> f32 {
    0.8 + 6.0 * a
}

/// How wide the value road's arc to a source is drawn: the size of its share of g,
/// a |g|, on the scale on which it is as wide as the attention's arc when |g| = 0.8
/// (and no wider than at |g| = 1.5).
fn road_width(a: f32, g_size: f32) -> f32 {
    0.8 + 6.0 * a * (g_size / 0.8).min(G_MOST as f32 / 0.8)
}

/// An arrow's head at `tip`, pointing along `dir`, in the batch of its kind.
fn head(f: &mut Frame, kind: u8, tip: Vec2, dir: Vec2, size: f32, width: f32, color: Color) {
    let d = dir.normalize_or_zero();
    let d = if d == Vec2::ZERO { Vec2::Y } else { d };
    let n = Vec2::new(-d.y, d.x);
    plot::line_of(f, kind, tip, tip - d * size + n * size * 0.5, width, color);
    plot::line_of(f, kind, tip, tip - d * size - n * size * 0.5, width, color);
}

/// An arrow from `a` to `b` in the batch of its kind: always a shaft and a head (so
/// that the batch keeps its number of points), the head no longer than half the shaft.
fn arrow(f: &mut Frame, kind: u8, a: Vec2, b: Vec2, width: f32, color: Color) {
    plot::line_of(f, kind, a, b, width, color);
    let size = (5.0 + 2.0 * width).min((b - a).length() * 0.5);
    head(f, kind, b, b - a, size, width, color);
}

/// A colour for the course's backward colour at a strength from 0 to 1, on the dark
/// cells.
const CELL: Color = Color::hex(0x161C2B);

fn plane_to_px((min, max): Rect, p: Vec2) -> Vec2 {
    Vec2::new(min.x + (p.x + R) / (2.0 * R) * (max.x - min.x), min.y + (R - p.y) / (2.0 * R) * (max.y - min.y))
}

fn px_to_plane((min, max): Rect, q: Vec2) -> Vec2 {
    Vec2::new((q.x - min.x) / (max.x - min.x).max(1.0) * 2.0 * R - R, R - (q.y - min.y) / (max.y - min.y).max(1.0) * 2.0 * R)
}

/// "position 6" and the like count from 1 on the screen.
fn nth(j: usize) -> usize {
    j + 1
}

impl AttnGrad {
    /// The buttons, handles and clicks of the picture, before anything is drawn.
    fn input(&mut self, f: &mut Frame, lay: &Lay) {
        for (card, r) in &lay.tabs {
            if plot::button(f, *r, card.tab(), lay.small, lay.cards[card.index()].is_some()) {
                self.card = *card;
            }
        }
        // the studied token, on the roles card
        if let Some(card) = lay.cards[Card::Roles.index()] {
            for (j, r) in self.study_buttons(lay, card).iter().enumerate() {
                if plot::button(f, *r, &format!("$x_{}$", nth(j)), lay.small, self.study == j) {
                    self.set_study(j);
                }
            }
        }
        let s = self.shown.shown.clone();
        let s = Shown(&s);
        // (in the frame in which the cards shown change, neither handle is offered: the
        // one dot would otherwise leap from the plane to the chart, or back)
        let showing = lay.cards.map(|c| c.is_some());
        let steady = showing == self.showing;
        self.showing = showing;
        // the tip of g, in the plane
        if let Some(plane) = lay.cards[Card::Values.index()].and_then(plane_square).filter(|_| steady) {
            let tip = plane_to_px(plane, s.mix() + s.g());
            if let Some(to) = plot::screen_handle(f, 1, tip, pal::PINK) {
                let to = to.max(plane.0).min(plane.1);
                let g = px_to_plane(plane, to) - s.mix();
                self.set_g([g.x as f64, g.y as f64]);
                self.snap = true;
            }
        }
        // the sharpness, on the chart's axis
        if let Some(card) = lay.cards[Card::Chart.index()].filter(|_| steady) {
            let chart = chart_plot(card, self.top.get(0).max(1e-3) as f64, lay);
            if let Some((b, _)) = chart.handle(f, 2, s.beta() as f64, 0.0, pal::WHITE) {
                self.set_beta(b);
                self.snap = true;
            }
        }
        // clicks and the pointer: a column of the diagram is the destination, a column of
        // the roles' rows the token studied, a row of the attention grid the destination
        let at = plot::pointer(f);
        self.hover = at.and_then(|p| lay.column(p));
        if f.clicked_canvas() {
            if let Some(p) = at {
                if let Some(j) = lay.column(p) {
                    self.set_dest(j);
                } else if let Some(j) = lay.role_column(p) {
                    self.set_study(j);
                } else if let Some(card) = lay.cards[Card::Pattern.index()] {
                    let (min, side) = pattern_grid(card, lay);
                    let i = ((p.y - min.y) / (side / T as f32)).floor();
                    if p.x >= min.x && p.x <= min.x + side && i >= 0.0 && i < T as f32 {
                        self.set_dest(i as usize);
                    }
                }
            }
        }
    }

    /// The buttons that choose the token studied, under the roles card's title.
    fn study_buttons(&self, lay: &Lay, (min, max): Rect) -> Vec<Rect> {
        let h = if lay.narrow { 22.0 } else { 24.0 };
        let w = ((max.x - min.x - 5.0 * 4.0) / T as f32).min(44.0);
        plot::even_row(T, min.x, min.x + T as f32 * w + 5.0 * 4.0, min.y + lay.small + 8.0, h, 4.0)
    }

    fn draw(&self, f: &mut Frame, lay: &Lay) {
        let s = Shown(&self.shown.shown);
        let p = &self.pass;
        let d = nth(p.dest);
        // the heading
        wrapped_px(f, lay.heading.0, lay.heading.1, &heading(lay.long_heading, d), lay.text, pal::TEXT);
        if lay.col < 8.0 {
            return;
        }
        self.diagram(f, lay, &s);
        for (k, card) in lay.cards.iter().enumerate() {
            let Some(card) = *card else { continue };
            match CARDS[k] {
                Card::Values => {
                    if let Some(plane) = plane_square(card) {
                        self.plane(f, lay, plane, &s);
                    }
                }
                Card::Pattern => self.pattern_card(f, lay, card, &s),
                Card::Jacobian => self.jacobian_card(f, lay, card, &s),
                Card::Chart => self.chart_card(f, lay, card, &s),
                Card::Roles => self.roles_card(f, lay, card, &s),
            }
        }
    }

    /// The tokens in a row, the attention from the destination above them, the value
    /// road below, the pattern road's bars, and what reaches each token.
    fn diagram(&self, f: &mut Frame, lay: &Lay, s: &Shown) {
        let p = &self.pass;
        let d = nth(p.dest);
        let (text, small) = (lay.text, lay.small);
        let dest = s.dest();
        let from = lay.centre(dest);
        let span = lay.col * (T - 1) as f32;
        let g_size = s.g().length();
        // names of the bands
        let (pink, light) = (VALUE_ROAD, PATTERN_ROAD);
        if let Some(x) = lay.names {
            let mid = |a: f32, b: f32| (a + b) * 0.5;
            let lines = |f: &mut Frame, y: f32, lines: &[(String, Color)], size: f32| {
                let n = lines.len() as f32;
                for (k, (line, c)) in lines.iter().enumerate() {
                    f.text_px(Vec2::new(x, y + (k as f32 - (n - 1.0) * 0.5) * (size + 4.0)), line.as_str(), size, *c, Align2::RIGHT_CENTER);
                }
            };
            lines(f, mid(lay.arcs.0, lay.arcs.1) + 6.0, &[("forward: where".into(), pal::DIM), (format!("position {d} looks"), pal::DIM)], small);
            f.text_px(Vec2::new(x, mid(lay.tokens.0, lay.tokens.1)), "tokens", small, pal::DIM, Align2::RIGHT_CENTER);
            f.text_px(Vec2::new(x, lay.attn), format!("attention $a_{{{d}j}}$"), small, FORWARD, Align2::RIGHT_CENTER);
            lines(f, mid(lay.road.0, lay.road.1), &[("value road:".into(), pink), (format!(r"$v_j$ gets $a_{{{d}j}}\,g$"), pink)], small);
            lines(
                f,
                mid(lay.bars.0, lay.bars.1),
                &[("pattern road:".into(), light), (format!(r"$-\partial L/\partial s_{{{d}j}}$"), light), ("up: look here".into(), pal::DIM), ("more".into(), pal::DIM)],
                small,
            );
            if let Some((a, _)) = lay.roles {
                f.text_px(Vec2::new(x, a - 11.0), "what each token gets:", small, pal::DIM, Align2::RIGHT_CENTER);
                for (r, name) in ["as a value", "as a key", "as the query"].iter().enumerate() {
                    f.text_px(Vec2::new(x, a + (r as f32 + 0.5) * 18.0), *name, small, if r == VALUE { pink } else { light }, Align2::RIGHT_CENTER);
                }
            }
        } else {
            let x = lay.x0;
            f.text_px(Vec2::new(x, lay.above[0]), format!("forward: where position {d} looks, attention $a_{{{d}j}}$"), small, FORWARD, Align2::LEFT_TOP);
            f.text_px(Vec2::new(x, lay.above[1]), format!(r"value road: $v_j$ gets $a_{{{d}j}}\,g$"), small, pink, Align2::LEFT_TOP);
            f.text_px(Vec2::new(x, lay.above[2]), format!(r"pattern road: $-\partial L/\partial s_{{{d}j}}$ (up: look here more)"), small, light, Align2::LEFT_TOP);
            if let Some((a, _)) = lay.roles {
                f.text_px(Vec2::new(x, a - small - 5.0), "what reaches each token: as a value, as a key, as the query", small, pal::DIM, Align2::LEFT_TOP);
            }
        }

        // the forward arcs, from the destination up and over to each source
        let room = (lay.arcs.1 - lay.arcs.0 - 4.0).max(4.0);
        for j in 0..T {
            let a = s.attn(j);
            let to = lay.centre(j as f32);
            let reach = ((to - from).abs() / span.max(1.0)).sqrt();
            let pts = arc(from, to, lay.arcs.1, room * (0.3 + 0.62 * reach), lay.col * 0.5, false);
            let shown = (a * 25.0).min(1.0);
            plot::path_of(f, ARCS, &pts, arc_width(a), FORWARD.a(shown * (0.35 + 0.65 * a.sqrt())));
        }
        // the tokens, and the ring round the destination
        for j in 0..T {
            let future = j > p.dest;
            let (a, b) = (lay.x0 + j as f32 * lay.col + 3.0, lay.x0 + (j + 1) as f32 * lay.col - 3.0);
            hud_rect_at(f, 1, Vec2::new(a, lay.tokens.0), Vec2::new(b, lay.tokens.1), if future { Color::hex(0x10141F) } else { Color::hex(0x1E2740) });
            let ink = if future { pal::FAINT } else { pal::TEXT };
            f.text_px(Vec2::new(lay.centre(j as f32), (lay.tokens.0 + lay.tokens.1) * 0.5), format!("$x_{}$", nth(j)), text + 1.0, ink, Align2::CENTER_CENTER);
            let value = p.a[p.dest][j];
            f.text_px(Vec2::new(lay.centre(j as f32), lay.attn), num(value, 2), small, if future { pal::FAINT } else { FORWARD }, Align2::CENTER_CENTER);
        }
        let ring = |f: &mut Frame, at: f32, width: f32, color: Color| {
            let (a, b) = (lay.x0 + at * lay.col + 1.0, lay.x0 + (at + 1.0) * lay.col - 1.0);
            let (t, u) = (lay.tokens.0 - 2.0, lay.tokens.1 + 2.0);
            plot::path_of(f, RINGS, &[Vec2::new(a, t), Vec2::new(b, t), Vec2::new(b, u), Vec2::new(a, u), Vec2::new(a, t)], width, color);
        };
        ring(f, dest, 2.2, pal::WHITE);
        let hover = self.hover.filter(|&j| j != self.dest).map_or(dest, |j| j as f32);
        ring(f, hover, 1.2, pal::WHITE.a(if self.hover.is_some_and(|j| j != self.dest) { 0.55 } else { 0.0 }));

        // the value road: back from the destination to each source, as wide as the arc
        let room = (lay.road.1 - lay.road.0 - 3.0).max(4.0);
        let base = lay.road.0;
        for j in 0..T {
            let a = s.attn(j);
            let to = lay.centre(j as f32);
            let reach = ((to - from).abs() / span.max(1.0)).sqrt();
            let pts = arc(from, to, base, room * (0.32 + 0.6 * reach), lay.col * 0.5, true);
            let shown = (a * 25.0).min(1.0) * (g_size * 6.0).min(1.0);
            let width = road_width(a, g_size);
            let color = pink.a(shown * (0.4 + 0.6 * a.sqrt()));
            plot::path_of(f, ROAD, &pts, width, color);
            head(f, ROAD, pts[ARC_PTS - 1], pts[ARC_PTS - 1] - pts[ARC_PTS - 3], 5.0 + 2.0 * width.min(4.0), width.min(2.5), color);
        }

        // the pattern road: a bar for each source, up where a step of descent raises
        // its score; and what it means when there is none
        let mid = (lay.bars.0 + lay.bars.1) * 0.5;
        let quiet = if p.dest == 0 {
            Some(r"position 1 can look only at itself: $a_{11} = 1$ whatever its score, so the pattern road is shut".to_string())
        } else if norm(&p.g) == 0.0 {
            Some(r"$g = 0$: no gradient arrives, so neither road carries anything".to_string())
        } else {
            None
        };
        if let Some(words) = quiet {
            // (right of the one bar there is, or across them all when every one is 0)
            let x = lay.x0 + if p.dest == 0 { lay.col } else { 0.0 } + 6.0;
            let width = lay.x0 + lay.col * T as f32 - x;
            wrapped_px(f, Vec2::new(x, lay.bars.0 + 2.0), width, &words, small - 0.5, pal::DIM);
        }
        let half = (lay.bars.1 - lay.bars.0) * 0.5 - 2.0;
        plot::line_of(f, plot::AXES, Vec2::new(lay.x0, mid), Vec2::new(lay.x0 + lay.col * T as f32, mid), 1.0, pal::DIM.a(0.6));
        for j in 0..T {
            let v = s.step(j) / BAR_FULL;
            let h = v.max(-1.0).min(1.0) * half;
            let x = lay.centre(j as f32);
            let w = (lay.col * 0.2).min(14.0);
            let (t, b) = if h >= 0.0 { (mid - h, mid) } else { (mid, mid - h) };
            hud_rect_at(f, 2, Vec2::new(x - w, t), Vec2::new(x + w, b.max(t + 0.5)), light.a(if j as f32 <= dest + 0.5 { 0.9 } else { 0.0 }));
            // past the end of the band: a chevron says so. (It is kept at the bar's end,
            // and turns over smoothly as the bar passes 0, so that it never leaps from one
            // end of the band to the other while it cannot be seen.)
            let over = v.abs() > 1.0;
            let tip = mid - v.max(-1.0).min(1.0) * (half + 1.0);
            let dir = -(v * 8.0).max(-1.0).min(1.0);
            let c = pal::WHITE.a(if over { 0.9 } else { 0.0 });
            plot::line_of(f, plot::MARKS, Vec2::new(x - w, tip - dir * 5.0), Vec2::new(x, tip), 1.4, c);
            plot::line_of(f, plot::MARKS, Vec2::new(x, tip), Vec2::new(x + w, tip - dir * 5.0), 1.4, c);
            let value = -p.ds[p.dest][j];
            let future = j > p.dest;
            f.text_px(Vec2::new(x, lay.bar_numbers), signed(value as f32, 3), small - 0.5, if future { pal::FAINT } else { light }, Align2::CENTER_CENTER);
        }

        // what reaches each token, in each role
        if let Some((top, _)) = lay.roles {
            let row = if lay.narrow { 14.0 } else { 18.0 };
            for j in 0..T {
                for r in 0..3 {
                    let size = s.role(j, r);
                    let (a, b) = (Vec2::new(lay.x0 + j as f32 * lay.col + 3.0, top + r as f32 * row + 1.0), Vec2::new(lay.x0 + (j + 1) as f32 * lay.col - 3.0, top + (r + 1) as f32 * row - 1.0));
                    hud_rect_at(f, 1, a, b, CELL);
                    hud_rect_at(f, 2, a, b, (if r == VALUE { pink } else { light }).a((size / ROLE_FULL).max(0.0).min(1.0) * 0.75));
                    let value = norm(&p.roles[r][j]);
                    let ink = if value == 0.0 { pal::FAINT } else { pal::WHITE.a(0.9) };
                    f.text_px((a + b) * 0.5, num(value, 2), small - 1.0, ink, Align2::CENTER_CENTER);
                }
            }
            let x = lay.x0 + s.study() * lay.col + 1.0;
            let (t, u) = (top - 1.0, top + 3.0 * row + 1.0);
            plot::path_of(f, RINGS, &[Vec2::new(x, t), Vec2::new(x + lay.col - 2.0, t), Vec2::new(x + lay.col - 2.0, u), Vec2::new(x, u), Vec2::new(x, t)], 1.4, pal::WHITE.a(0.8));
        }
    }

    /// The head's values in their plane, the mix, g, the value road's arrows and the line
    /// of sources as good as the mix.
    fn plane(&self, f: &mut Frame, lay: &Lay, rect: Rect, s: &Shown) {
        let p = &self.pass;
        let d = nth(p.dest);
        let (min, max) = rect;
        let small = lay.small;
        hud_panel(f, min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        f.text_px(Vec2::new(min.x, min.y - 6.0), format!(r"the values $v_j$, their mix $o_{d}$, and $g$ (drag its tip)"), small, pal::TEXT, Align2::LEFT_BOTTOM);
        let at = |q: Vec2| plane_to_px(rect, q);
        // axes through 0, and the frame
        let frame = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
        plot::path_of(f, plot::AXES, &frame, 1.0, pal::DIM.a(0.45));
        plot::line_of(f, plot::AXES, at(Vec2::new(-R, 0.0)), at(Vec2::new(R, 0.0)), 1.0, pal::DIM.a(0.25));
        plot::line_of(f, plot::AXES, at(Vec2::new(0.0, -R)), at(Vec2::new(0.0, R)), 1.0, pal::DIM.a(0.25));
        for v in [-1.0f32, 1.0] {
            f.text_px(at(Vec2::new(v, 0.0)) + Vec2::new(0.0, 4.0), num(v as f64, 0), small - 1.5, pal::FAINT, Align2::CENTER_TOP);
            f.text_px(at(Vec2::new(0.0, v)) + Vec2::new(-4.0, 0.0), num(v as f64, 0), small - 1.5, pal::FAINT, Align2::RIGHT_CENTER);
        }
        let mix = s.mix();
        let g = s.g();
        let unit = if g.length() > 1e-4 { g / g.length() } else { Vec2::new(0.6, 0.8) };
        // the sources as good as the mix: the line through it at right angles to g
        let across = Vec2::new(-unit.y, unit.x);
        let line = plot::clip(&[at(mix - across * 4.0 * R), at(mix + across * 4.0 * R)], rect);
        let on = g.length() > 1e-4;
        if let [a, b, ..] = line[..] {
            plot::dashed_of(f, DASHED, a, b, 1.2, pal::WHITE.a(if on { 0.5 } else { 0.0 }));
        } else {
            plot::dashed_of(f, DASHED, mix, mix, 0.0, Color::CLEAR);
        }
        // the mix as the attention's weighted average: a thread to each source
        let mix_px = at(mix);
        for j in 0..T {
            let a = s.attn(j);
            plot::line_of(f, plot::CURVES, mix_px, at(Vec2::new(p.v[j][0] as f32, p.v[j][1] as f32)), 1.0, FORWARD.a((a * 25.0).min(1.0) * (0.06 + 0.3 * a)));
        }
        // the values, the gradient each receives, and their names (on the side away from g)
        let screen_unit = Vec2::new(unit.x, -unit.y);
        for j in 0..T {
            let v = Vec2::new(p.v[j][0] as f32, p.v[j][1] as f32);
            let a = s.attn(j);
            let future = j > p.dest;
            let dv = s.dv(j);
            let width = 1.2 + 2.6 * a.sqrt();
            arrow(f, ARROWS, at(v), at(v + dv), width, VALUE_ROAD.a(((a * 25.0).min(1.0) * (g.length() * 8.0).min(1.0)) * 0.95));
            plot::disc(f, at(v), 9.0, if future { pal::FAINT } else { FORWARD });
            f.caption_px(at(v) - screen_unit * 17.0, format!("$v_{}$", nth(j)), small, if future { pal::FAINT } else { FORWARD }, Align2::CENTER_CENTER);
        }
        // the mix, and g from it
        hud_circle(f, mix_px, 6.5, 1.8, pal::WHITE);
        f.caption_px(mix_px - screen_unit * 19.0, format!("$o_{d}$"), small + 0.5, pal::WHITE, Align2::CENTER_CENTER);
        arrow(f, ARROWS, mix_px, at(mix + g), 2.6, BACKWARD);
        let tip = at(mix + g);
        let side = Vec2::new(-screen_unit.y, screen_unit.x);
        f.text_px(tip + side * 14.0 + screen_unit * 4.0, "$g$", small + 1.5, BACKWARD, Align2::CENTER_CENTER);
        // what the marks mean, under the plane
        let note = format!(r"Pink on each value: its gradient, $a_{{{d}j}}\,g$. Dashed: as good as the mix. Values beyond it, on the side away from $g$, are better than the mix: a step of descent raises their scores.");
        wrapped_px(f, Vec2::new(min.x, max.y + 8.0), max.x - min.x, &note, small - 0.5, pal::DIM);
    }

    /// The whole attention pattern, the destination's row ringed.
    fn pattern_card(&self, f: &mut Frame, lay: &Lay, card: Rect, s: &Shown) {
        let small = lay.small;
        let (min, side) = pattern_grid(card, lay);
        f.text_px(card.0, r"attention $a_{ij}$: row $i$ looks at $j$", small, pal::TEXT, Align2::LEFT_TOP);
        let cell = side / T as f32;
        for i in 0..T {
            for j in 0..T {
                let a = Vec2::new(min.x + j as f32 * cell, min.y + i as f32 * cell);
                let b = a + Vec2::splat(cell - 1.0);
                if j > i {
                    hud_rect_at(f, 1, a, b, Color::hex(0x0B0E16));
                } else {
                    hud_rect_at(f, 1, a, b, CELL.mix(FORWARD, s.pattern(i, j).sqrt()));
                }
            }
            f.text_px(Vec2::new(min.x - 5.0, min.y + (i as f32 + 0.5) * cell), format!("{}", nth(i)), small - 1.5, pal::DIM, Align2::RIGHT_CENTER);
            f.text_px(Vec2::new(min.x + (i as f32 + 0.5) * cell, min.y + side + 3.0), format!("{}", nth(i)), small - 1.5, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(min.x + side - 2.0, min.y + 2.0), "masked", small - 1.5, pal::FAINT.mix(pal::DIM, 0.4), Align2::RIGHT_TOP);
        let y = min.y + s.dest() * cell;
        plot::path_of(f, RINGS, &[Vec2::new(min.x - 1.0, y - 1.0), Vec2::new(min.x + side, y - 1.0), Vec2::new(min.x + side, y + cell), Vec2::new(min.x - 1.0, y + cell), Vec2::new(min.x - 1.0, y - 1.0)], 1.6, pal::WHITE);
        f.text_px(Vec2::new(card.0.x, min.y + side + 18.0), "click a row: the gradient arrives there", small - 1.0, pal::DIM, Align2::LEFT_TOP);
    }

    /// The softmax's Jacobian for the destination's row.
    fn jacobian_card(&self, f: &mut Frame, lay: &Lay, card: Rect, s: &Shown) {
        let p = &self.pass;
        let d = nth(p.dest);
        let small = lay.small;
        f.text_px(card.0, format!(r"the softmax's Jacobian, row {d}"), small, pal::TEXT, Align2::LEFT_TOP);
        f.text_px(card.0 + Vec2::new(0.0, small + 6.0), format!(r"$\partial a_{{{d}j}}/\partial s_{{{d}l}} = a_{{{d}j}}(\delta_{{jl}} - a_{{{d}l}})$"), small, pal::DIM, Align2::LEFT_TOP);
        let top = card.0.y + 2.0 * small + 18.0;
        let side = (card.1.x - card.0.x - 4.0).min(card.1.y - top - small - 10.0).max(0.0);
        let cell = side / T as f32;
        let fire = Cmap::IceFire.table();
        for j in 0..T {
            for l in 0..T {
                let a = Vec2::new(card.0.x + l as f32 * cell, top + j as f32 * cell);
                let b = a + Vec2::splat(cell - 1.0);
                let inside = j <= p.dest && l <= p.dest;
                hud_rect_at(f, 1, a, b, if inside { Color::hex(0x10141F) } else { Color::hex(0x0B0E16) });
                // (no entry is larger than 1/4 in size; the map's pale ends are left out,
                // so that white numbers stay legible on every cell)
                let v = s.jac(j, l) / 0.25;
                hud_rect_at(f, 2, a, b, fire[table_index(0.5 + 0.42 * v.max(-1.0).min(1.0))].a((v.abs() * 3.0).min(1.0)));
                if inside && cell >= 30.0 {
                    f.text_px((a + b) * 0.5, num(p.jac[j][l], 2), small - 2.0, pal::WHITE.a(0.9), Align2::CENTER_CENTER);
                }
            }
        }
        let biggest = p.jac.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
        f.text_px(Vec2::new(card.0.x, top + side + 6.0), format!("largest entry {}", num(biggest, 4)), small - 1.0, pal::DIM, Align2::LEFT_TOP);
    }

    /// The size of what enters each road, against the sharpness.
    fn chart_card(&self, f: &mut Frame, lay: &Lay, card: Rect, s: &Shown) {
        let p = &self.pass;
        let small = lay.small;
        let d = nth(p.dest);
        let chart = chart_plot(card, self.top.get(0).max(1e-3) as f64, lay);
        f.text_px(card.0, "the size of the gradient entering each road", small, pal::TEXT, Align2::LEFT_TOP);
        chart.backing(f, 36.0, 34.0, 6.0, 8.0);
        chart.y_grid(f);
        chart.x_axis(f, r"sharpness $\beta$", small - 1.0);
        chart.y_axis(f, "", small - 1.0);
        let pts = |road: usize| (0..SAMPLES).map(move |k| (beta_at(k), s.curve(k, road) as f64));
        chart.path(f, plot::CURVES, pts(0), 2.0, VALUE_ROAD);
        let dashed: Vec<Vec3> = plot::clip(&pts(1).map(|(x, y)| chart.at(x, y)).collect::<Vec<_>>(), chart.rect).iter().filter(|q| q.is_finite()).map(|q| f.hud(*q)).collect();
        plot::apart(f, PATTERN_CURVE, false).polyline(dashed, 2.0, PATTERN_ROAD);
        // where the sharpness is now
        let b = s.beta() as f64;
        chart.vline(f, plot::MARKS, b, 1.0, pal::WHITE.a(0.5), false);
        let (value, pattern) = (p.value_road(), p.pattern_road());
        chart.dot(f, b, value, 7.0, VALUE_ROAD);
        chart.dot(f, b, pattern, 7.0, PATTERN_ROAD);
        // which curve is which, under the title
        let y = card.0.y + small + 12.0;
        let mut x = card.0.x;
        let pattern_ink = PATTERN_ROAD;
        plot::line_of(f, plot::MARKS, Vec2::new(x, y), Vec2::new(x + 18.0, y), 2.0, VALUE_ROAD);
        let value_name = r"value road, $|\partial L/\partial V|$";
        f.text_px(Vec2::new(x + 23.0, y), value_name, small - 1.0, VALUE_ROAD, Align2::LEFT_CENTER);
        x += 23.0 + text_width(value_name, small - 1.0) + 16.0;
        plot::line_of(f, plot::MARKS, Vec2::new(x, y), Vec2::new(x + 18.0, y), 2.0, pattern_ink);
        f.text_px(Vec2::new(x + 23.0, y), format!(r"pattern road, $|\partial L/\partial s_{{{d}\cdot}}|$"), small - 1.0, pattern_ink, Align2::LEFT_CENTER);
        f.text_px(
            Vec2::new(card.0.x, card.1.y - small),
            format!(r"$\beta = {}$: value road {}, pattern road {}", num(p.beta, 2), num(value, 3), num(pattern, 4)),
            small - 1.0,
            pal::DIM,
            Align2::LEFT_CENTER,
        );
    }

    /// The studied token's gradient, split into its three roles.
    fn roles_card(&self, f: &mut Frame, lay: &Lay, card: Rect, s: &Shown) {
        let p = &self.pass;
        let small = lay.small;
        let j = nth(self.study);
        f.text_px(card.0, format!(r"what reaches $x_{j}$, in its three roles"), small, pal::TEXT, Align2::LEFT_TOP);
        let names = ["as a value", "$+$ as a key", "$+$ as the query", "$=$ in all"];
        let top = card.0.y + small + 8.0 + 24.0 + small + 18.0;
        let w = ((card.1.x - card.0.x) / 4.0).floor();
        let row = (small + 6.0).min((card.1.y - top - 60.0) / D as f32).max(small);
        for (r, name) in names.iter().enumerate() {
            let x = card.0.x + r as f32 * w + w * 0.5;
            f.text_px(Vec2::new(x, top - 8.0), *name, small - 1.0, [VALUE_ROAD, PATTERN_ROAD, PATTERN_ROAD, pal::TEXT][r], Align2::CENTER_BOTTOM);
            for c in 0..D {
                let v = s.get(at::SPLIT + r * D + c);
                f.text_px(Vec2::new(x, top + (c as f32 + 0.5) * row), num(v as f64, 3), small - 0.5, if v == 0.0 { pal::FAINT } else { pal::TEXT }, Align2::CENTER_CENTER);
            }
        }
        let sizes = |w: &[[f64; D]; DH]| norm(w.as_flattened());
        let y = top + D as f32 * row + 10.0;
        let weights = format!(
            r"and the weights: $|\partial L/\partial W_V| = {}$ by the value road, $|\partial L/\partial W_K| = {}$ and $|\partial L/\partial W_Q| = {}$ by the pattern road",
            num(sizes(&p.dw[VALUE]), 3),
            num(sizes(&p.dw[KEY]), 3),
            num(sizes(&p.dw[QUERY]), 3)
        );
        wrapped_px(f, Vec2::new(card.0.x, y), card.1.x - card.0.x, &weights, small - 1.0, pal::DIM);
    }
}

/// The plane's square in its card: as large as the card leaves room for, under the
/// card's title and over the note on its marks, centred.
fn plane_square((min, max): Rect) -> Option<Rect> {
    let side = (max.x - min.x).min(max.y - min.y - OVER_PLANE - UNDER_PLANE).floor();
    if side < 100.0 {
        return None;
    }
    let x = min.x + ((max.x - min.x - side) * 0.5).floor();
    Some((Vec2::new(x, min.y + OVER_PLANE), Vec2::new(x + side, min.y + OVER_PLANE + side)))
}

/// The line over the picture, long or short, for the destination `d` (from 1).
fn heading(long: bool, d: usize) -> String {
    if long {
        format!(r"One attention head reading six tokens. A gradient $g$ arrives at the output of position {d} and goes back by two roads: through the values, and through the attention.")
    } else {
        format!(r"One attention head on six tokens. A gradient $g$ arrives at position {d}'s output and goes back by two roads.")
    }
}

/// How many lines the heading takes at `width` (near enough: a few pixels' room is
/// kept, since the estimate may be short by a little).
fn heading_lines(long: bool, width: f32, size: f32) -> f32 {
    (text_width(&heading(long, 6), size) / (width - 12.0).max(40.0)).ceil().max(1.0)
}

/// Text broken into lines no wider than `width`, from `at` down, in logical pixels of
/// the screen (`kit::wrapped` takes the plane's own units). Returns the height taken.
fn wrapped_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> f32 {
    let ppu = pixels_per_unit(f).max(1e-6);
    let Some(world) = f.cam.pick_ground(at * f.px) else { return 0.0 };
    let w = width / ppu;
    wrapped(f, world.truncate(), w, text, size, color);
    wrapped_height(f, w, text, size)
}

/// Where the attention grid goes in its card: its top-left corner and its side.
fn pattern_grid(card: Rect, lay: &Lay) -> (Vec2, f32) {
    let top = card.0.y + lay.small + 10.0;
    let side = (card.1.x - card.0.x - 22.0).min(card.1.y - top - 2.0 * lay.small - 26.0).max(0.0).floor();
    (Vec2::new(card.0.x + 18.0, top), side)
}

/// The chart in its card.
fn chart_plot(card: Rect, top: f64, lay: &Lay) -> Plot {
    let rect = (card.0 + Vec2::new(38.0, 2.0 * lay.small + 32.0), card.1 - Vec2::new(8.0, 2.0 * lay.small + 26.0));
    Plot::new(rect, [0.0, BETA_MAX], [0.0, top])
}

impl Scene for AttnGrad {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        !self.at_rest()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("click a token: the gradient arrives there    drag the pink arrow's tip: change $g$    drag the dot under the chart: sharpness")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"An attention head lets each position $i$ gather from the positions up to it. It scores how well its query matches each key, turns the scores into attention $a_{ij}$ with a softmax, and outputs the mix of the values weighted by the attention. Here one head reads six tokens, and a gradient $g$ arrives at the output of one position, the destination. On the way back it splits into two roads: through the values (what the head moves) and through the attention (where it looks).",
        );
        ui::equation(ui, r"o_i = \sum_{j \le i} a_{ij}\,v_j, \quad a_{ij} = \frac{e^{s_{ij}}}{\sum_{l \le i} e^{s_{il}}}, \quad s_{ij} = \frac{q_i \cdot k_j}{\sqrt{2}}");
        ui::note(ui, r"and, with $g = \partial L/\partial o_i$,");
        ui::equation(ui, r"\frac{\partial L}{\partial v_j} = a_{ij}\,g, \qquad \frac{\partial L}{\partial s_{ij}} = a_{ij}\,(g \cdot v_j - g \cdot o_i)");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The head");
        let mut dest = self.dest + 1;
        if ui::slider_int(ui, "destination $i$, where $g$ arrives", &mut dest, 1..=T) {
            self.set_dest(dest - 1);
        }
        let mut beta = self.beta as f32;
        if ui::slider(ui, r"sharpness $\beta$", &mut beta, 0.0..=BETA_MAX as f32) {
            self.set_beta(beta as f64);
        }
        ui::note(
            ui,
            r"$\beta$ multiplies the query weights $W_Q$, and with them every score. At 0 the attention is even; as $\beta$ grows it piles onto the source with the largest score. (At 0 every query is 0, so nothing reaches a token as a key or as the query, though the scores and $W_Q$ still get their gradients.)",
        );
        let mut study = self.study + 1;
        if ui::slider_int(ui, "token studied", &mut study, 1..=T) {
            self.set_study(study - 1);
        }
        ui::note(ui, "The token whose gradient the roles card splits into its three roles (a click on a column of the rows under the bars chooses it too).");
        ui::note(ui, r"$g$ is set by dragging the tip of the pink arrow in the plane of the values.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (FORWARD, r"cyan, what goes forward: the attention from the destination (arcs over the tokens, as wide as $a_{ij}$; and the grid), the values $v_j$ and their mix $o_i$"),
                (VALUE_ROAD, "pink, gradients: $g$, and the value road (arcs under the tokens, as wide as each value's share; the arrows on the values; what reaches a token as a value)"),
                (PATTERN_ROAD, "light pink, the pattern road: the bars (up where a step of descent raises the score), and what reaches a token as a key or as the query"),
                (pal::WHITE, "white: the ring round the destination; in the plane, dashed, the values exactly as good as the mix"),
                (Cmap::IceFire.color(0.9), "the Jacobian's cells: orange positive (its diagonal), blue negative"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Query, key, value:", r"three vectors made from each token by the weights $W_Q$, $W_K$ and $W_V$. The query says what a position looks for, a key what a position offers, the value what it hands on if it is looked at. Here each has 2 numbers, so the values can be drawn in a plane; a token has 4.");
        ui::term(ui, r"Score $s_{ij}$:", r"how well the query of position $i$ matches the key of position $j$, $q_i \cdot k_j/\sqrt{2}$: divided by the square root of the head's width, as in most transformers.");
        ui::term(ui, r"Attention $a_{ij}$:", r"the softmax of row $i$ of the scores: positive weights adding up to 1. The causal mask leaves out the scores of $j > i$: a position cannot look ahead.");
        ui::term(ui, r"Mix $o_i$:", r"the values averaged with the attention as weights, $\sum_j a_{ij} v_j$: the head's output at $i$. (In a transformer it is then multiplied by the head's output weights $W_O$ and added to the stream.)");
        ui::term(ui, r"$g$:", r"the gradient of the loss with respect to $o_i$, as it comes back from the layers above. Here it is set by hand: the loss is then $L = g \cdot o_i$, plus whatever does not depend on this head.");
        ui::term(ui, "Value road:", r"the gradient through the values, $\partial L/\partial v_j = a_{ij}\,g$, and on through $W_V$. It changes what the head moves.");
        ui::term(ui, "Pattern road:", r"the gradient through the attention and the softmax to the scores, $\partial L/\partial s_{ij}$, and from them to the query of $i$ and the keys of the sources. It changes where the head looks.");
        ui::term(ui, "A step of descent:", r"a small move of every number against its gradient, which, if small enough, lowers the loss. The bars show $-\partial L/\partial s_{ij}$, the way such a step moves each score.");
        ui::term(ui, "The softmax's Jacobian:", r"how each attention weight answers a nudge of each score, $\partial a_{ij}/\partial s_{il} = a_{ij}(\delta_{jl} - a_{il})$, with $\delta_{jl}$ 1 if $j = l$ and 0 if not: the matrix $\operatorname{diag}(a) - a a^\top$. It is near 0 when one weight is near 1.");

        ui::section(ui, "Numbers");
        let p = &self.pass;
        let d = p.dest;
        ui::readout(ui, "destination $i$", format!("{}", nth(d)));
        ui::readout(ui, r"sharpness $\beta$", num(p.beta, 2));
        ui::readout_colored(ui, "$g$", format!("${}$", tuple(&p.g, 2)), BACKWARD);
        ui::readout(ui, r"$L = g \cdot o_i$", num(p.g_dot_mix(), 4));
        ui::readout_colored(ui, r"value road, $|\partial L/\partial V|$", num(p.value_road(), 4), BACKWARD);
        ui::readout_colored(ui, r"pattern road, $|\partial L/\partial s_{i\cdot}|$", num(p.pattern_road(), 4), PATTERN_ROAD);
        ui::readout(ui, r"$\sum_j \partial L/\partial s_{ij}$", num(p.ds[d].iter().sum::<f64>(), 4));
        let biggest = p.jac.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
        ui::readout(ui, "the Jacobian's largest entry", num(biggest, 4));
        let size = |w: &[[f64; D]; DH]| num(norm(w.as_flattened()), 4);
        ui::readout(ui, r"$|\partial L/\partial W_V|$, value road", size(&p.dw[VALUE]));
        ui::readout(ui, r"$|\partial L/\partial W_K|$, pattern road", size(&p.dw[KEY]));
        ui::readout(ui, r"$|\partial L/\partial W_Q|$, pattern road", size(&p.dw[QUERY]));
        ui::see_also(
            ui,
            &[
                ("lossgrad", "where the gradient starts: a softmax and its Jacobian"),
                ("backpropmat", "backpropagation a layer at a time, in matrices"),
                ("tfgrad", "the gradient through a whole transformer"),
                ("transformer", "a transformer, opened up: what heads learn"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let covered = view.x < 160.0;
        let lay = Lay::of(view, f.top_inset, self.card);
        if !covered {
            // the picture is for its buttons, handles and clicks: no moving the view
            f.handles.capture();
            f.handles.claim_clicks();
            self.input(f, &lay);
        }
        self.advance(f.dt, f.real_dt);
        if covered {
            return;
        }
        in_order(f);
        self.draw(f, &lay);
        keep_batches(f);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROLES: [&str; 3] = ["value", "key", "query"];

    /// The head computed again, plainly, in f64 and without the tape: the mix of
    /// position `dest`, for tokens `x` and query, key and value weights.
    fn mix_by_hand(x: &[[f64; D]; T], wq: &[[f64; D]; DH], wk: &[[f64; D]; DH], wv: &[[f64; D]; DH], dest: usize) -> [f64; DH] {
        let apply = |w: &[[f64; D]; DH], x: &[f64; D]| -> [f64; DH] { std::array::from_fn(|r| dot(&w[r], x)) };
        let q = apply(wq, &x[dest]);
        let scores: Vec<f64> = (0..=dest).map(|j| dot(&q, &apply(wk, &x[j])) / (DH as f64).sqrt()).collect();
        let top = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let e: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
        let sum: f64 = e.iter().sum();
        let mut o = [0.0; DH];
        for j in 0..=dest {
            let v = apply(wv, &x[j]);
            for c in 0..DH {
                o[c] += e[j] / sum * v[c];
            }
        }
        o
    }

    fn scaled(w: &[[f64; D]; DH], s: f64) -> [[f64; D]; DH] {
        w.map(|row| row.map(|x| x * s))
    }

    /// L = g · o_dest worked out by hand.
    fn loss(x: &[[f64; D]; T], wq: &[[f64; D]; DH], wk: &[[f64; D]; DH], wv: &[[f64; D]; DH], dest: usize, g: [f64; DH]) -> f64 {
        dot(&g, &mix_by_hand(x, wq, wk, wv, dest))
    }

    /// Every gradient the picture shows that is a gradient of the loss with respect
    /// to something the loss is a function of (the tokens in each role, the weights)
    /// is the tape's, and agrees with a central difference of the loss worked out by
    /// hand; for every destination, at several sharpnesses and gradients.
    #[test]
    fn every_gradient_is_that_of_its_finite_differences() {
        let mut tape = Tape::new();
        for dest in 0..T {
            for beta in [0.0, 0.7, 1.0, 3.0, BETA_MAX] {
                for g in [[0.5, 0.6], [-1.2, 0.3], [0.0, -0.9]] {
                    let p = run(&mut tape, dest, beta, g, false);
                    let wq = scaled(&W_Q, beta);
                    let close = |got: f64, want: f64, what: &str| assert!((got - want).abs() <= 1e-7 * (1.0 + want.abs()), "dest {dest} β {beta} g {g:?}: {what}: tape {got}, differences {want}");
                    for j in 0..T {
                        for c in 0..D {
                            let h = 1e-6;
                            let (mut up, mut down) = (TOKENS, TOKENS);
                            up[j][c] += h;
                            down[j][c] -= h;
                            // each role alone: the token moved where that role reads it
                            let fd = |role: usize| {
                                let pick = |r: usize, moved: &[[f64; D]; T]| if r == role { *moved } else { TOKENS };
                                let plus = mix_role(&pick(QUERY, &up), &pick(KEY, &up), &pick(VALUE, &up), &wq, dest, g);
                                let minus = mix_role(&pick(QUERY, &down), &pick(KEY, &down), &pick(VALUE, &down), &wq, dest, g);
                                (plus - minus) / (2.0 * h)
                            };
                            for role in [VALUE, KEY, QUERY] {
                                close(p.roles[role][j][c], fd(role), &format!("token {j} entry {c} as {}", ROLES[role]));
                            }
                            // and all at once
                            let whole = (loss(&up, &wq, &W_K, &W_V, dest, g) - loss(&down, &wq, &W_K, &W_V, dest, g)) / (2.0 * h);
                            close(p.total(j)[c], whole, &format!("token {j} entry {c} in all"));
                        }
                    }
                    for r in 0..DH {
                        for c in 0..D {
                            let h = 1e-6;
                            let moved = |w: &[[f64; D]; DH], by: f64| {
                                let mut w = *w;
                                w[r][c] += by;
                                w
                            };
                            let fd_v = (loss(&TOKENS, &wq, &W_K, &moved(&W_V, h), dest, g) - loss(&TOKENS, &wq, &W_K, &moved(&W_V, -h), dest, g)) / (2.0 * h);
                            let fd_k = (loss(&TOKENS, &wq, &moved(&W_K, h), &W_V, dest, g) - loss(&TOKENS, &wq, &moved(&W_K, -h), &W_V, dest, g)) / (2.0 * h);
                            let fd_q = (loss(&TOKENS, &moved(&wq, h), &W_K, &W_V, dest, g) - loss(&TOKENS, &moved(&wq, -h), &W_K, &W_V, dest, g)) / (2.0 * h);
                            close(p.dw[VALUE][r][c], fd_v, "value weights");
                            close(p.dw[KEY][r][c], fd_k, "key weights");
                            close(p.dw[QUERY][r][c], fd_q, "query weights");
                        }
                    }
                    // and the loss itself
                    close(p.g_dot_mix(), loss(&TOKENS, &wq, &W_K, &W_V, dest, g), "the loss");
                }
            }
        }
    }

    /// The loss with the tokens read as queries, keys and values from three separate
    /// copies (what the tape is given).
    fn mix_role(xq: &[[f64; D]; T], xk: &[[f64; D]; T], xv: &[[f64; D]; T], wq: &[[f64; D]; DH], dest: usize, g: [f64; DH]) -> f64 {
        let apply = |w: &[[f64; D]; DH], x: &[f64; D]| -> [f64; DH] { std::array::from_fn(|r| dot(&w[r], x)) };
        let q = apply(wq, &xq[dest]);
        let scores: Vec<f64> = (0..=dest).map(|j| dot(&q, &apply(&W_K, &xk[j])) / (DH as f64).sqrt()).collect();
        let top = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let e: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
        let sum: f64 = e.iter().sum();
        (0..=dest).map(|j| e[j] / sum * dot(&g, &apply(&W_V, &xv[j]))).sum()
    }

    /// The closed forms the scene writes are what the tape computes: the value road
    /// (each value gets a_dj g), the slope for the attention (g · v_j), the softmax's
    /// Jacobian (diag a − a aᵀ, from the tape's own sweeps), the slope for the scores
    /// (a_dj (g · v_j − g · o_d)), and from there the query of the destination and the
    /// keys of the sources. Nothing reaches a row other than the destination's, a
    /// masked score, or a position after the destination; and the slopes of a row's
    /// scores add up to 0.
    #[test]
    fn the_closed_forms_are_the_tapes() {
        let mut tape = Tape::new();
        let root = (DH as f64).sqrt();
        for dest in 0..T {
            for beta in [0.0, 1.0, 4.0, BETA_MAX] {
                for g in [[0.5, 0.6], [-1.2, 0.3], [0.0, 0.0]] {
                    let p = run(&mut tape, dest, beta, g, true);
                    let near = |a: f64, b: f64| (a - b).abs() <= 1e-12 * (1.0 + b.abs());
                    let a = &p.a[dest];
                    let gv: Vec<f64> = (0..T).map(|j| dot(&g, &p.v[j])).collect();
                    let mix: f64 = (0..=dest).map(|j| a[j] * gv[j]).sum();
                    assert!(near(mix, p.g_dot_mix()));
                    for i in 0..T {
                        for j in 0..T {
                            let causal = j <= i;
                            if i != dest {
                                assert!(p.da[i][j] == 0.0 && p.ds[i][j] == 0.0, "dest {dest}: ({i}, {j}) got some");
                            }
                            if !causal {
                                // (the product A V hands a slope to a masked entry of the
                                // pattern too, but the mask passes none of it to its score)
                                assert!(p.a[i][j] == 0.0 && p.ds[i][j] == 0.0);
                            }
                            if i != dest || !causal {
                                continue;
                            }
                            assert!(near(p.da[i][j], gv[j]), "∂L/∂a");
                            assert!(near(p.ds[i][j], a[j] * (gv[j] - mix)), "∂L/∂s");
                        }
                    }
                    for j in 0..T {
                        let dv: [f64; DH] = std::array::from_fn(|c| if j <= dest { a[j] * g[c] } else { 0.0 });
                        assert!(near(p.dv[j][0], dv[0]) && near(p.dv[j][1], dv[1]), "∂L/∂v_{j}");
                        let dk: [f64; DH] = std::array::from_fn(|c| p.ds[dest][j] * p.q[dest][c] / root);
                        assert!(near(p.dk[j][0], dk[0]) && near(p.dk[j][1], dk[1]), "∂L/∂k_{j}");
                        let dq: [f64; DH] = std::array::from_fn(|c| if j == dest { (0..T).map(|l| p.ds[dest][l] * p.k[l][c]).sum::<f64>() / root } else { 0.0 });
                        assert!(near(p.dq[j][0], dq[0]) && near(p.dq[j][1], dq[1]), "∂L/∂q_{j}");
                        // on to the tokens: through W_vᵀ, W_kᵀ and β W_qᵀ
                        for c in 0..D {
                            let through = |w: &[[f64; D]; DH], d: &[f64; DH], s: f64| s * (0..DH).map(|r| w[r][c] * d[r]).sum::<f64>();
                            assert!(near(p.roles[VALUE][j][c], through(&W_V, &p.dv[j], 1.0)));
                            assert!(near(p.roles[KEY][j][c], through(&W_K, &p.dk[j], 1.0)));
                            assert!(near(p.roles[QUERY][j][c], through(&W_Q, &p.dq[j], beta)));
                            if j > dest {
                                assert!(p.roles.iter().all(|r| r[j][c] == 0.0), "token {j} after {dest} got some");
                            }
                        }
                        // the Jacobian
                        for l in 0..T {
                            let want = if j <= dest && l <= dest { a[j] * (if j == l { 1.0 } else { 0.0 } - a[l]) } else { 0.0 };
                            assert!(near(p.jac[j][l], want), "Jacobian ({j}, {l}): {} against {want}", p.jac[j][l]);
                        }
                    }
                    let sum: f64 = p.ds[dest].iter().sum();
                    assert!(sum.abs() < 1e-15, "the slopes add up to {sum}");
                    // the gradient of the scores is the Jacobian times g · v
                    for l in 0..T {
                        let via: f64 = (0..T).map(|j| p.jac[j][l] * p.da[dest][j]).sum();
                        assert!(near(via, p.ds[dest][l]));
                    }
                }
            }
        }
    }

    /// The three roles add up to what a single copy of the tokens receives when it
    /// feeds all three at once (the head as the library builds it).
    #[test]
    fn the_three_roles_add_up_to_the_whole() {
        let mut tape = Tape::new();
        for dest in 0..T {
            let p = run(&mut tape, dest, 1.3, [0.5, 0.6], false);
            let mut t = Tape::new();
            let x = t.constant(T, D, TOKENS.as_flattened());
            let wq = t.param(DH, D, scaled(&W_Q, 1.3).as_flattened());
            let wk = t.param(DH, D, W_K.as_flattened());
            let wv = t.param(DH, D, W_V.as_flattened());
            let (q, k, v) = (t.matmul_t(x, wq), t.matmul_t(x, wk), t.matmul_t(x, wv));
            let s = t.matmul_t(q, k);
            let s = t.scale(s, 1.0 / (DH as f64).sqrt());
            let a = t.softmax_rows_causal(s);
            let o = t.matmul(a, v);
            let mut seed = [0.0; T * DH];
            seed[dest * DH..dest * DH + DH].copy_from_slice(&[0.5, 0.6]);
            t.backward_seeded(o, &seed);
            for j in 0..T {
                for c in 0..D {
                    let whole = t.grad(x)[j * D + c];
                    assert!((p.total(j)[c] - whole).abs() < 1e-14, "dest {dest} token {j}");
                }
            }
        }
    }

    /// The gradients the picture shows at the head's inner values (the scores, the
    /// attention, the query, the keys, the values) are those of central differences of
    /// the same loss, each with everything that made it held fixed: what the tape's
    /// gradient at a node in the middle of a computation means.
    #[test]
    fn the_inner_gradients_are_those_of_finite_differences() {
        let mut tape = Tape::new();
        let root = (DH as f64).sqrt();
        for dest in 0..T {
            for beta in [0.0, 1.0, 4.0] {
                for g in [G0, G1, [-1.1, 0.4]] {
                    let p = run(&mut tape, dest, beta, g, false);
                    let close = |got: f64, want: f64, what: &str| assert!((got - want).abs() <= 1e-7 * (1.0 + want.abs()), "dest {dest} β {beta}: {what}: tape {got}, differences {want}");
                    // the loss from the scores of the destination's row, and the values
                    let from_scores = |s: &[f64], v: &[[f64; DH]; T]| {
                        let top = s[..=dest].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let e: Vec<f64> = s[..=dest].iter().map(|x| (x - top).exp()).collect();
                        let sum: f64 = e.iter().sum();
                        (0..=dest).map(|j| e[j] / sum * dot(&g, &v[j])).sum::<f64>()
                    };
                    let scores_of = |q: &[f64; DH], k: &[[f64; DH]; T]| -> Vec<f64> { (0..T).map(|j| dot(q, &k[j]) / root).collect() };
                    let h = 1e-6;
                    for j in 0..=dest {
                        let (mut up, mut down) = (p.scores[dest].to_vec(), p.scores[dest].to_vec());
                        up[j] += h;
                        down[j] -= h;
                        close(p.ds[dest][j], (from_scores(&up, &p.v) - from_scores(&down, &p.v)) / (2.0 * h), &format!("∂L/∂s_{j}"));
                        // the attention as a free number (the product A V alone)
                        let mix = |a: f64| (0..=dest).map(|l| if l == j { a } else { p.a[dest][l] } * dot(&g, &p.v[l])).sum::<f64>();
                        close(p.da[dest][j], (mix(p.a[dest][j] + h) - mix(p.a[dest][j] - h)) / (2.0 * h), &format!("∂L/∂a_{j}"));
                        for c in 0..DH {
                            let (mut vu, mut vd) = (p.v, p.v);
                            vu[j][c] += h;
                            vd[j][c] -= h;
                            close(p.dv[j][c], (from_scores(&p.scores[dest], &vu) - from_scores(&p.scores[dest], &vd)) / (2.0 * h), &format!("∂L/∂v_{j}"));
                            let (mut ku, mut kd) = (p.k, p.k);
                            ku[j][c] += h;
                            kd[j][c] -= h;
                            let fd = (from_scores(&scores_of(&p.q[dest], &ku), &p.v) - from_scores(&scores_of(&p.q[dest], &kd), &p.v)) / (2.0 * h);
                            close(p.dk[j][c], fd, &format!("∂L/∂k_{j}"));
                        }
                    }
                    for c in 0..DH {
                        let (mut qu, mut qd) = (p.q[dest], p.q[dest]);
                        qu[c] += h;
                        qd[c] -= h;
                        let fd = (from_scores(&scores_of(&qu, &p.k), &p.v) - from_scores(&scores_of(&qd, &p.k), &p.v)) / (2.0 * h);
                        close(p.dq[dest][c], fd, "∂L/∂q");
                    }
                }
            }
        }
    }

    /// The head is as the comments on its numbers say: the values lie round a circle
    /// (lengths from 0.85 to 1.16), and at β = 1 the last position spreads its attention
    /// unevenly over all six (from 0.07 to 0.42, x_2 first and x_6 second), while no
    /// other position but the first is sure of one source (none above 0.61).
    #[test]
    fn the_head_is_as_its_comments_say() {
        let p = run(&mut Tape::new(), 5, 1.0, G0, false);
        let lengths: Vec<f64> = (0..T).map(|j| norm(&p.v[j])).collect();
        assert!(lengths.iter().all(|&l| (0.85..1.16).contains(&l)), "{lengths:?}");
        let a = p.a[5];
        let mut order: Vec<usize> = (0..T).collect();
        order.sort_by(|&x, &y| a[y].total_cmp(&a[x]));
        assert!(order[0] == 1 && order[1] == 5 && a.iter().all(|&x| (0.069..0.43).contains(&x)), "{a:?}");
        for i in 1..T {
            assert!(p.a[i][..=i].iter().all(|&x| x < 0.61), "row {}: {:?}", i + 1, p.a[i]);
        }
    }

    /// The scene as experiment k's button leaves it, run at 60 frames a second until it
    /// has come to rest, from where another experiment left it.
    fn after(k: usize) -> AttnGrad {
        let mut scene = AttnGrad::new();
        scene.set_up((k + 2) % TRIALS.len());
        for _ in 0..30 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        scene.set_up(k);
        for _ in 0..60 * 10 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.at_rest() {
                break;
            }
        }
        assert!(scene.at_rest(), "experiment {k} has not come to rest");
        scene
    }

    /// The pass as the scene shows it now (at its settings).
    fn shown_pass(scene: &mut AttnGrad) -> Pass {
        run(&mut Tape::new(), scene.dest, scene.beta, scene.g, true)
    }

    fn percent(x: f64) -> String {
        num(100.0 * x, 0)
    }

    /// The caption of experiment k says each of these, as written.
    fn says(k: usize, parts: &[&str]) {
        for part in parts {
            assert!(TRIALS[k].look_for.contains(part), "the caption of \"{}\" does not say {part}", TRIALS[k].label);
        }
    }

    #[test]
    fn the_scene_opens_on_the_first_experiment() {
        let (opened, first) = (AttnGrad::new(), after(0));
        assert!(opened.dest == first.dest && opened.beta == first.beta && opened.g == first.g && opened.study == first.study && opened.card == first.card);
        // and what it draws is what the tape says for it, exactly
        assert_eq!(opened.shown.shown, opened.target());
    }

    #[test]
    fn the_value_road_does_what_its_caption_says() {
        let mut scene = after(0);
        let p = shown_pass(&mut scene);
        assert!(p.dest == 5 && p.beta == 1.0 && p.g == [0.0, 0.8]);
        let a = p.a[5];
        assert_eq!([percent(a[1]), percent(a[0]), percent(a[4])], ["42", "7", "7"]);
        // x_2 has the most attention, and the most of g
        assert!((0..T).all(|j| a[1] >= a[j]));
        // each value's share is a_6j g, and the shares make up g
        let mut sum = [0.0; DH];
        let mut lengths = 0.0;
        for j in 0..T {
            for c in 0..DH {
                assert!((p.dv[j][c] - a[j] * p.g[c]).abs() < 1e-15);
                sum[c] += p.dv[j][c];
            }
            lengths += norm(&p.dv[j]);
        }
        assert!((sum[0] - p.g[0]).abs() < 1e-14 && (sum[1] - p.g[1]).abs() < 1e-14 && (lengths - norm(&p.g)).abs() < 1e-14);
        // the arcs under the tokens are as wide as those over them (|g| = 0.8)
        for j in 0..T {
            assert!((road_width(a[j] as f32, norm(&p.g) as f32) - arc_width(a[j] as f32)).abs() < 1e-5);
        }
        assert_eq!(num(norm(p.dw[VALUE].as_flattened()), 3), "0.662");
        says(0, &["$g = (0, 0.8)$", "position 6", "42 %", "7 %", "= 0.662$"]);
    }

    #[test]
    fn the_pattern_road_does_what_its_caption_says() {
        let mut scene = after(1);
        let p = shown_pass(&mut scene);
        assert!(p.dest == 5 && p.beta == 1.0 && p.g == G1);
        let (a, gv): ([f64; T], Vec<f64>) = (p.a[5], (0..T).map(|j| dot(&p.g, &p.v[j])).collect());
        assert_eq!([num(gv[0], 2), num(gv[1], 2), num(p.g_dot_mix(), 2)], ["−0.88", "0.71", "0.23"]);
        // x_1's value lies furthest against g, and x_2 is the head's favourite
        assert!((0..T).all(|j| gv[0] <= gv[j] && a[1] >= a[j]));
        // a step of descent raises exactly the scores of the values better than the mix
        for j in 0..T {
            assert_eq!(-p.ds[5][j] > 0.0, gv[j] < p.g_dot_mix(), "x_{}", j + 1);
        }
        // x_2's score falls most; x_6 rises more than x_1, with five-tenths as much... the
        // attention of each as written
        let step: Vec<f64> = (0..T).map(|j| -p.ds[5][j]).collect();
        assert!((0..T).all(|j| step[1] <= step[j]));
        assert_eq!([num(-step[1], 3), num(step[5], 3), num(step[0], 3), num(a[5], 2), num(a[0], 2)], ["0.201", "0.126", "0.078", "0.20", "0.07"]);
        assert!(step[5] > step[0] && step.iter().cloned().fold(f64::NEG_INFINITY, f64::max) == step[5]);
        assert!(step.iter().sum::<f64>().abs() < 1e-15);
        says(1, &["$-0.88$", "$+0.71$", "= 0.23$", "$-0.201$", "0.126 against 0.078", "0.20 against 0.07"]);
    }

    #[test]
    fn a_sure_head_does_what_its_caption_says() {
        let mut scene = AttnGrad::new();
        scene.set_up(2);
        let start = shown_pass(&mut scene);
        assert!(scene.film.is_some() && start.beta == 0.0 && start.g == G0 && start.dest == 5);
        let biggest = |p: &Pass| p.jac.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
        assert!(start.a[5].iter().all(|a| (a - 1.0 / 6.0).abs() < 1e-15));
        // the film plays in the scene's time: not while it is paused
        scene.advance(0.0, 1.0 / 60.0);
        assert_eq!(scene.beta, 0.0);
        let mut frames = 0;
        while !scene.at_rest() {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            frames += 1;
            assert!(frames < 60 * 10);
        }
        let end = shown_pass(&mut scene);
        assert_eq!(end.beta, BETA_MAX);
        assert_eq!(num(end.a[5][1], 4), "0.9993");
        assert_eq!([num(biggest(&start), 2), num(biggest(&end), 4)], ["0.14", "0.0006"]);
        assert_eq!([num(start.pattern_road(), 3), num(end.pattern_road(), 4)], ["0.230", "0.0004"]);
        assert_eq!([num(start.value_road(), 3), num(end.value_road(), 3), num(norm(&end.g), 1)], ["0.327", "0.799", "0.8"]);
        // x_5 is the best source, and moving attention from x_2 to it lowers the loss
        let gv: Vec<f64> = (0..T).map(|j| dot(&end.g, &end.v[j])).collect();
        assert!((0..T).all(|j| gv[4] <= gv[j]));
        assert_eq!(num(gv[1] - gv[4], 2), "1.40");
        // on the chart, measured over the knob's whole range: the pattern road is widest
        // at β = 0 and falls all the way to under a five-hundredth of that; the value road
        // widens all the way, to within a thousandth of |g|
        let c = curve(&mut Tape::new(), 5, G0);
        assert!(c.windows(2).all(|w| w[1][1] < w[0][1] && w[1][0] > w[0][0]));
        assert!(c[SAMPLES - 1][1] < c[0][1] / 500.0 && (c[SAMPLES - 1][0] - 0.8).abs() < 1e-3);
        // and the gradient of the query and key weights (where the head learns where to
        // look) goes with it, while the value weights' grows
        let size = |w: &[[f64; D]; DH]| norm(w.as_flattened());
        assert!(size(&end.dw[QUERY]) < 1e-3 && size(&end.dw[KEY]) < 0.01 && size(&end.dw[VALUE]) > 5.0 * size(&start.dw[VALUE]));
        says(2, &["from 0 to 10", "0.9993", "from 0.14 to 0.0006", "from 0.230 to 0.0004", "from 0.327 to 0.799", "$|g| = 0.8$", "$x_5$ would be better", "1.40 per unit"]);
    }

    /// Whatever g is, the pattern road falls over the knob's range to a small share of
    /// its largest, for every destination: the chart's claim is no accident of the
    /// experiments' g.
    #[test]
    fn the_pattern_road_closes_for_every_g() {
        let mut tape = Tape::new();
        for dest in 1..T {
            for k in 0..24 {
                let angle = k as f64 * std::f64::consts::TAU / 24.0;
                let g = [angle.cos(), angle.sin()];
                let c = curve(&mut tape, dest, g);
                let most = c.iter().map(|v| v[1]).fold(0.0f64, f64::max);
                // (positions 5 and 6 are the surest by β = 10; the others' top scores are
                // closer together, and position 2 is only 0.987 sure there: the worst g
                // leaves 3.9 to 6.1 % of the road open at positions 2 to 4, under 0.4 % at
                // 5 and 6)
                let share = [0.0, 0.055, 0.065, 0.04, 0.003, 0.004][dest];
                assert!(c[SAMPLES - 1][1] < share * most, "dest {dest} g {g:?}: {} of {most}", c[SAMPLES - 1][1]);
                // (it need not be widest at β = 0: for some g it is widest a little later)
            }
        }
    }

    #[test]
    fn nothing_flows_to_the_future_as_its_caption_says() {
        let mut scene = after(3);
        let p = shown_pass(&mut scene);
        assert_eq!(p.dest, 2);
        for j in 3..T {
            assert!(p.roles.iter().all(|r| r[j].iter().all(|&x| x == 0.0)), "x_{} got some", j + 1);
            assert!(p.dv[j] == [0.0; DH] && p.dk[j] == [0.0; DH] && p.dq[j] == [0.0; DH]);
            assert!(p.a[2][j] == 0.0 && p.ds[2][j] == 0.0);
        }
        for j in 0..=2 {
            assert!(norm(&p.total(j)) > 0.01, "x_{} got nothing", j + 1);
        }
        // and for every destination: nothing reaches a token after it
        let mut tape = Tape::new();
        for dest in 0..T {
            let p = run(&mut tape, dest, 1.0, G1, false);
            for j in dest + 1..T {
                assert!(p.total(j).iter().all(|&x| x == 0.0));
            }
        }
        // the studied token, x_5, is among those that get nothing
        assert_eq!(scene.study, 4);
        says(3, &["position 3", "$x_4$, $x_5$ and $x_6$", "$x_1$, $x_2$ and $x_3$"]);
    }

    #[test]
    fn one_token_three_roles_as_its_caption_says() {
        let mut scene = after(4);
        let p = shown_pass(&mut scene);
        assert!(p.dest == 5 && scene.study == 5 && p.g == G1 && p.beta == 1.0);
        assert_eq!(num(p.a[5][5], 2), "0.20");
        let sizes = [VALUE, KEY, QUERY].map(|r| num(norm(&p.roles[r][5]), 3));
        assert_eq!(sizes, ["0.251", "0.126", "0.365"]);
        assert_eq!(tuple(&p.total(5), 3), "(−0.279, −0.117, 0.212, 0.022)");
        assert_eq!(num(norm(&p.total(5)), 3), "0.370");
        // a source gets the first two, and no query's share
        assert!(norm(&p.roles[VALUE][1]) > 0.1 && norm(&p.roles[KEY][1]) > 0.1 && p.roles[QUERY][1] == [0.0; D]);
        // only the destination gets the query's share
        assert!((0..T).all(|j| (j == 5) == (norm(&p.roles[QUERY][j]) > 0.0)));
        says(4, &["$a_{66} = 0.20$", "0.251, 0.126 and 0.365", "$(-0.279, -0.117, 0.212, 0.022)$", "0.370"]);
    }

    /// Whatever is asked, everything worked out and drawn stays finite, and the scene
    /// comes to rest: the first position (whose softmax has one score), a gradient of
    /// 0, of more than the most, and not a number; every sharpness and more.
    #[test]
    fn degenerate_settings_leave_everything_finite() {
        let mut scene = AttnGrad::new();
        for dest in [0, 5, 1000] {
            for beta in [0.0, BETA_MAX, -3.0, 1e9, f64::NAN, f64::INFINITY] {
                for g in [[0.0, 0.0], [1e9, -1e9], [f64::NAN, 0.3], [0.0, -0.01]] {
                    scene.set_dest(dest);
                    scene.set_beta(beta);
                    scene.set_g(g);
                    scene.set_study(dest);
                    for _ in 0..60 * 3 {
                        scene.advance(1.0 / 60.0, 1.0 / 60.0);
                        if scene.at_rest() {
                            break;
                        }
                    }
                    assert!(scene.at_rest());
                    assert!(scene.dest < T && scene.study < T && (0.0..=BETA_MAX).contains(&scene.beta) && norm(&scene.g) <= G_MOST + 1e-12);
                    assert!(scene.target().iter().all(|v| v.is_finite()), "dest {dest} β {beta} g {g:?}");
                    let p = &scene.pass;
                    assert!(p.value_road().is_finite() && p.pattern_road().is_finite() && p.g_dot_mix().is_finite());
                    for text in [num(p.g_dot_mix(), 4), num(p.pattern_road(), 4), tuple(&p.g, 2)] {
                        assert!(!text.contains('?') && !text.contains("NaN") && !text.contains("inf"), "{text}");
                    }
                }
            }
        }
        // the first position looks only at itself: its pattern road is shut whatever g is
        let p = run(&mut Tape::new(), 0, 3.0, G1, true);
        assert!(p.a[0][0] == 1.0 && p.ds[0][0] == 0.0 && p.jac[0][0] == 0.0 && p.pattern_road() == 0.0);
        // at β = 0 every query is 0: nothing reaches a token as a key or the query, but
        // the scores and the query weights still get their gradients (the panel's note)
        let p = run(&mut Tape::new(), 5, 0.0, G1, true);
        assert!((0..T).all(|j| p.roles[KEY][j] == [0.0; D] && p.roles[QUERY][j] == [0.0; D]));
        assert!(p.pattern_road() > 0.1 && norm(p.dw[QUERY].as_flattened()) > 0.1);
        // and with g = 0 nothing arrives anywhere
        let p = run(&mut Tape::new(), 5, 1.0, [0.0, 0.0], true);
        assert!(p.value_road() == 0.0 && p.pattern_road() == 0.0 && (0..T).all(|j| p.total(j) == [0.0; D]));
    }

    /// A change of setting glides: the picture moves a little in each frame and comes
    /// exactly to what the tape says.
    #[test]
    fn changes_glide_and_come_to_rest() {
        let mut scene = AttnGrad::new();
        scene.set_dest(2);
        let mut last = scene.shown.shown.clone();
        let mut frames = 0;
        loop {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            let now = scene.shown.shown.clone();
            let moved = now.iter().zip(&last).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(moved < 0.6, "frame {frames}: a jump of {moved}");
            last = now;
            frames += 1;
            if scene.at_rest() {
                break;
            }
            assert!(frames < 600);
        }
        assert!(frames > 5);
        assert_eq!(scene.shown.shown, scene.target());
    }

    /// The picture's parts fit the canvas, from a phone to a desktop, and keep off one
    /// another.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1078.0, 1000.0), (1600.0, 1000.0), (390.0, 800.0), (390.0, 560.0), (800.0, 390.0), (760.0, 600.0), (240.0, 180.0), (620.0, 480.0), (1300.0, 700.0), (2400.0, 1400.0), (900.0, 1200.0)] {
            for inset in [0.0, 46.0] {
                for card in CARDS {
                    let view = Vec2::new(w, h);
                    let lay = Lay::of(view, inset, card);
                    let within = |r: Rect| r.0.cmpge(Vec2::ZERO).all() && r.1.cmple(view + Vec2::splat(0.5)).all() && r.0.y >= inset;
                    let rects: Vec<Rect> = lay.cards.iter().flatten().copied().chain(lay.tabs.iter().map(|t| t.1)).collect();
                    for r in &rects {
                        assert!(within(*r), "{w}x{h}: {r:?}");
                    }
                    // no two of them overlap
                    for (i, a) in rects.iter().enumerate() {
                        for b in &rects[i + 1..] {
                            assert!(a.1.x <= b.0.x || b.1.x <= a.0.x || a.1.y <= b.0.y || b.1.y <= a.0.y, "{w}x{h}: {a:?} and {b:?}");
                        }
                    }
                    if lay.col < 8.0 {
                        continue;
                    }
                    // the diagram is inside the canvas, under the heading, and above or
                    // beside every card
                    let right = lay.x0 + lay.col * T as f32;
                    let bottom = lay.roles.map_or(lay.bar_numbers + 8.0, |r| r.1);
                    assert!(lay.x0 >= 0.0 && right <= w + 0.5 && bottom <= h, "{w}x{h}: the diagram");
                    for r in &rects {
                        assert!(r.0.y >= bottom || r.0.x >= right, "{w}x{h}: {r:?} runs into the diagram");
                    }
                    // a card chosen on a narrow canvas is the one shown
                    if lay.narrow && !lay.tabs.is_empty() {
                        assert!(lay.cards[card.index()].is_some());
                    }
                }
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        let mut tape = Tape::new();
        let show = |p: &Pass| {
            let d = p.dest;
            println!("dest {} β {} g {:?}", d + 1, p.beta, p.g);
            println!("  a      {:?}", p.a[d].map(|x| (x * 1e4).round() / 1e4));
            println!("  g·v    {:?}", (0..T).map(|j| (dot(&p.g, &p.v[j]) * 1e4).round() / 1e4).collect::<Vec<_>>());
            println!("  g·o    {:.5}", p.g_dot_mix());
            println!("  -ds    {:?}", p.ds[d].map(|x| (-x * 1e5).round() / 1e5));
            println!("  dv     {:?}", p.dv.map(|r| r.map(|x| (x * 1e4).round() / 1e4)));
            println!("  |dV| {:.5}  |ds| {:.6}  |g| {:.5}", p.value_road(), p.pattern_road(), norm(&p.g));
            let biggest = p.jac.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
            println!("  largest Jacobian entry {biggest:.6}");
            println!("  |dWv| {:.4} |dWk| {:.4} |dWq| {:.4}", norm(p.dw[VALUE].as_flattened()), norm(p.dw[KEY].as_flattened()), norm(p.dw[QUERY].as_flattened()));
            for j in 0..T {
                println!("  x{}: value {:?} key {:?} query {:?} total {:?}", j + 1, p.roles[VALUE][j].map(|x| (x * 1e3).round() / 1e3), p.roles[KEY][j].map(|x| (x * 1e3).round() / 1e3), p.roles[QUERY][j].map(|x| (x * 1e3).round() / 1e3), p.total(j).map(|x| (x * 1e3).round() / 1e3));
                println!("       sizes {:.3} {:.3} {:.3} total {:.3}", norm(&p.roles[VALUE][j]), norm(&p.roles[KEY][j]), norm(&p.roles[QUERY][j]), norm(&p.total(j)));
            }
        };
        println!("v {:?}", run(&mut tape, 5, 1.0, G0, true).v);
        for t in &TRIALS {
            println!("===== {}", t.label);
            show(&run(&mut tape, t.dest, t.beta, t.g, true));
            if t.film {
                show(&run(&mut tape, t.dest, BETA_MAX, t.g, true));
            }
        }
        for g in [G0, G1] {
            let c = curve(&mut tape, 5, g);
            let (k, top) = c.iter().enumerate().fold((0, 0.0), |b, (k, v)| if v[1] > b.1 { (k, v[1]) } else { b });
            println!("g {g:?}: pattern road largest {top:.5} at β {:.3}; at 0 {:.5}; at 10 {:.6}; value at 0 {:.5} at 10 {:.5}", beta_at(k), c[0][1], c[SAMPLES - 1][1], c[0][0], c[SAMPLES - 1][0]);
        }
    }
}

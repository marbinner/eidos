//! Skip connections: a block that adds its result to its input hands the gradient back
//! unchanged plus a part through itself, so a stack of them is a sum over paths, and one
//! of the paths goes round every block.

use super::*;
use crate::clock::Pace;
use crate::math::arch::{self, Init, Loss, MlpSpec, Model, Norm, Place, Stage, Trace};
use crate::math::autograd::{Act, Tape, Var};
use crate::scenes::stats::plot::{self, Glide, Plot, Rect};

pub const DEF: SceneDef = SceneDef {
    id: "residual",
    title: "Skip connections",
    group: GROUP,
    blurb: "Adding each layer's output to its input gives the gradient a straight road back: paths of every length, and why depth stops hurting.",
    make: |_| Box::new(Residual::new()),
};

// ----- the networks ----------------------------------------------------------------------

/// Units in every hidden layer.
const WIDTH: usize = 16;
/// Points of the spirals: the whole training set, and the batch of every pass.
const POINTS: usize = 24;
/// The seeds of the data and of the starting weights.
const DATA_SEED: u64 = 7;
const WEIGHT_SEED: u64 = 11;
/// The deepest network the depth control offers.
const MAX_DEPTH: usize = 50;
/// The blocks of the network drawn as paths, and of the one whose paths are counted by
/// length (2^10 paths: few enough to work out exactly).
const PATH_BLOCKS: usize = 3;
const LENGTH_BLOCKS: usize = 10;
/// The training run: steps, and Adam's rate.
const STEPS: usize = 100;
const RATE: f64 = 0.01;

/// The factor s on each block's result in x + s f(x).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scale {
    One,
    RootDepth,
}

impl Scale {
    /// s for a network of `depth` blocks.
    fn of(self, depth: usize) -> f64 {
        match self {
            Scale::One => 1.0,
            Scale::RootDepth => 1.0 / (depth.max(1) as f64).sqrt(),
        }
    }
}

/// The normalisations offered (batch normalisation is the scene normgrad's).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NormKind {
    None,
    Layer,
    Rms,
}

impl NormKind {
    const ALL: [NormKind; 3] = [NormKind::None, NormKind::Layer, NormKind::Rms];

    fn norm(self) -> Norm {
        match self {
            NormKind::None => Norm::None,
            NormKind::Layer => Norm::Layer,
            NormKind::Rms => Norm::Rms,
        }
    }

    fn name(self) -> &'static str {
        match self {
            NormKind::None => "none",
            NormKind::Layer => "layer",
            NormKind::Rms => "RMS",
        }
    }
}

/// What every block is: its scale, its normalisation and where that goes. (Without a
/// normalisation "pre" and "post" are the same network, and the design says "pre".)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Design {
    scale: Scale,
    norm: NormKind,
    place: Place,
}

impl Design {
    fn new(scale: Scale, norm: NormKind, place: Place) -> Design {
        Design { scale, norm, place: if norm == NormKind::None { Place::Pre } else { place } }
    }

    /// The normalisation after the sum, if that is where it is.
    fn post(self) -> Option<NormKind> {
        (self.norm != NormKind::None && self.place == Place::Post).then_some(self.norm)
    }
}

/// The specification of a network of `depth` blocks, plain or with skips. Every
/// network here is ReLU, 16 wide, its weights normal with variance 1/(number of
/// inputs), on points in the plane in three classes.
fn spec(depth: usize, residual: bool, d: Design) -> MlpSpec {
    MlpSpec {
        inputs: 2,
        width: WIDTH,
        depth: depth.max(1),
        outputs: 3,
        act: Act::Relu,
        norm: d.norm.norm(),
        place: d.place,
        residual,
        residual_scale: if residual { d.scale.of(depth) } else { 1.0 },
        init: Init::Lecun,
        gain: 1.0,
        bias: true,
    }
}

/// The spirals, each coordinate scaled to a root mean square of 1 (inputs are
/// standardised before a network sees them, so that the first block receives a stream
/// of about the size the later ones do).
fn data() -> (Vec<f64>, Vec<f64>) {
    let (mut x, y) = arch::spirals(POINTS, 3, &mut Rng::new(DATA_SEED));
    for c in 0..2 {
        let size = (x.iter().skip(c).step_by(2).map(|v| v * v).sum::<f64>() / POINTS as f64).sqrt();
        x.iter_mut().skip(c).step_by(2).for_each(|v| *v /= size.max(1e-12));
    }
    (x, y)
}

/// The network of `depth` blocks as it starts (the same seed for every design).
fn model(depth: usize, residual: bool, d: Design) -> Model {
    Model::mlp(&spec(depth, residual, d), &mut Rng::new(WEIGHT_SEED))
}

/// The root mean square of a slice (0 for nothing).
fn rms(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The stream x_l: what the first block receives (l = 0: in a network with skips the
/// input map's result, in a plain one the data), or what block l hands on.
fn stream(tr: &Trace, l: usize) -> Option<Var> {
    if l == 0 {
        tr.get(Stage::Input, 1)
    } else {
        tr.get(Stage::Output, l)
    }
}

/// The root mean square of the stream x_l and of the loss's gradient g_l there, at
/// every layer l from 0 to the depth, after one forward and backward pass.
#[derive(Clone, Debug, Default)]
struct Profile {
    act: Vec<f64>,
    grad: Vec<f64>,
}

fn profile(model: &Model, x: &[f64], y: &[f64], tape: &mut Tape) -> Profile {
    let tr = model.loss_mlp(tape, x, y, Loss::CrossEntropy);
    tape.backward(tr.loss);
    let (mut act, mut grad) = (Vec::new(), Vec::new());
    for l in 0..=tr.layers {
        if let Some(v) = stream(&tr, l) {
            act.push(rms(tape.value(v)));
            grad.push(rms(tape.grad(v)));
        }
    }
    Profile { act, grad }
}

/// The gradient `v` at what a post block hands on, handed back through its
/// normalisation to the sum it normalised (`sum`, a row per example; `gain`, one number
/// per unit): row by row, with h = v times the gain and y the sum standardised,
/// (h − the mean of h − y × the mean of h y) / the sum's spread; RMS normalisation
/// takes no means. What the tape's backward sweep does there (a test holds the two
/// together), without the sweep.
fn norm_back(norm: NormKind, sum: &[f64], gain: &[f64], v: &[f64]) -> Vec<f64> {
    let n = gain.len();
    if norm == NormKind::None || n == 0 || sum.len() != v.len() {
        return v.to_vec();
    }
    let centre = norm == NormKind::Layer;
    let mut out = vec![0.0; v.len()];
    let mut h = vec![0.0; n];
    let mut y = vec![0.0; n];
    for ((o, x), g) in out.chunks_exact_mut(n).zip(sum.chunks_exact(n)).zip(v.chunks_exact(n)) {
        let mean = if centre { x.iter().sum::<f64>() / n as f64 } else { 0.0 };
        let spread = x.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n as f64;
        let inv = 1.0 / (spread + arch::EPS).sqrt();
        for j in 0..n {
            h[j] = g[j] * gain[j];
            y[j] = (x[j] - mean) * inv;
        }
        let mh = if centre { h.iter().sum::<f64>() / n as f64 } else { 0.0 };
        let mhy = dot(&h, &y) / n as f64;
        for j in 0..n {
            o[j] = inv * (h[j] - mh - y[j] * mhy);
        }
    }
    out
}

/// The two roads back through block l of a network with skips, for a gradient v that
/// has reached what the block hands on: along the stream (`road`: in the post
/// arrangement back through the normalisation to the sum, otherwise unchanged), and
/// through the block (`through`: the transposed Jacobian of s f_l, back to what the
/// block received).
struct Roads<'a> {
    tape: &'a mut Tape,
    tr: &'a Trace,
    model: &'a Model,
    post: Option<NormKind>,
    s: f64,
}

impl Roads<'_> {
    fn road(&mut self, l: usize, v: &[f64]) -> Vec<f64> {
        let Some(kind) = self.post else { return v.to_vec() };
        match (self.tr.get(Stage::Sum, l), self.model.group(&format!("layer {l} norm gain"))) {
            (Some(sum), Some(gain)) => norm_back(kind, self.tape.value(sum), &gain.value, v),
            _ => v.to_vec(),
        }
    }

    fn through(&mut self, l: usize, v: &[f64]) -> Vec<f64> {
        let (Some(a), Some(input)) = (self.tr.get(Stage::PostAct, l), self.tr.get(Stage::Input, l)) else { return vec![0.0; v.len()] };
        if self.tape.value(a).len() != v.len() {
            return vec![0.0; v.len()];
        }
        let seed: Vec<f64> = v.iter().map(|g| self.s * g).collect();
        self.tape.backward_seeded(a, &seed);
        self.tape.grad(input).to_vec()
    }
}

/// A pass of a network with skips and its gradient: the gradient g_0 at x_0 (the whole)
/// and the gradient g_L at the top, with the trace to hand it back through the blocks.
fn top_and_whole(model: &Model, x: &[f64], y: &[f64], tape: &mut Tape) -> Option<(Trace, Vec<f64>, Vec<f64>)> {
    let tr = model.loss_mlp(tape, x, y, Loss::CrossEntropy);
    tape.backward(tr.loss);
    let (bottom, top) = (stream(&tr, 0)?, stream(&tr, tr.layers)?);
    let (whole, g_top) = (tape.grad(bottom).to_vec(), tape.grad(top).to_vec());
    Some((tr, whole, g_top))
}

/// The normalisation after the sum of a model with skips, if it has one there.
fn post_of(model: &Model) -> Option<NormKind> {
    let s = model.mlp_spec()?;
    let kind = match s.norm {
        Norm::Layer => NormKind::Layer,
        Norm::Rms => NormKind::Rms,
        _ => return None,
    };
    (s.residual && s.place == Place::Post).then_some(kind)
}

/// The gradient g_0 at x_0, and its parts by path: `terms[S]` for every set S of
/// blocks passed through (bit l − 1 of S for block l), the gradient g_L at the top
/// handed down the stream round the other blocks. They add up to g_0.
fn paths(model: &Model, x: &[f64], y: &[f64], tape: &mut Tape) -> (Vec<f64>, Vec<Vec<f64>>) {
    let s = model.mlp_spec().map_or(1.0, |sp| sp.residual_scale);
    let post = post_of(model);
    let Some((tr, whole, g_top)) = top_and_whole(model, x, y, tape) else { return (Vec::new(), Vec::new()) };
    let depth = tr.layers;
    let mut roads = Roads { tape, tr: &tr, model, post, s };
    let terms = (0..1usize << depth)
        .map(|set| {
            let mut v = g_top.clone();
            for l in (1..=depth).rev() {
                v = roads.road(l, &v);
                if set & (1 << (l - 1)) != 0 {
                    v = roads.through(l, &v);
                }
            }
            v
        })
        .collect();
    (whole, terms)
}

/// The gradient g_0 at x_0, and its parts gathered by the number of blocks the paths
/// pass through: `parts[k]` is the sum of the terms of every path through exactly k of
/// them. Worked out from the top down without listing the paths: below block l, the
/// paths through k blocks are those through k blocks above it that went round it, and
/// those through k − 1 above it that went through it.
fn by_length(model: &Model, x: &[f64], y: &[f64], tape: &mut Tape) -> (Vec<f64>, Vec<Vec<f64>>) {
    let s = model.mlp_spec().map_or(1.0, |sp| sp.residual_scale);
    let post = post_of(model);
    let Some((tr, whole, g_top)) = top_and_whole(model, x, y, tape) else { return (Vec::new(), Vec::new()) };
    let depth = tr.layers;
    let mut parts = vec![g_top];
    let mut roads = Roads { tape, tr: &tr, model, post, s };
    for l in (1..=depth).rev() {
        for v in parts.iter_mut() {
            *v = roads.road(l, v);
        }
        let through: Vec<Vec<f64>> = parts.iter().map(|v| roads.through(l, v)).collect();
        parts.push(vec![0.0; whole.len()]);
        for (k, t) in through.iter().enumerate() {
            for (p, &q) in parts[k + 1].iter_mut().zip(t) {
                *p += q;
            }
        }
    }
    (whole, parts)
}

/// The straight road's part of the gradient at every layer, 0 to the depth (root mean
/// squares): the gradient at the top handed down the stream alone, through no block.
fn road_profile(model: &Model, x: &[f64], y: &[f64], tape: &mut Tape) -> Vec<f64> {
    let post = post_of(model);
    let Some((tr, _, g_top)) = top_and_whole(model, x, y, tape) else { return Vec::new() };
    let depth = tr.layers;
    let mut v = g_top;
    let mut out = vec![rms(&v)];
    let mut roads = Roads { tape, tr: &tr, model, post, s: 1.0 };
    for l in (1..=depth).rev() {
        v = roads.road(l, &v);
        out.push(rms(&v));
    }
    out.reverse();
    out
}

/// A term's share of the whole (its part along the whole, over the whole's length: the
/// shares of terms that add up to the whole add up to 1) and its length over the
/// whole's.
fn share_and_length(term: &[f64], whole: &[f64]) -> (f64, f64) {
    let w2 = dot(whole, whole);
    if !(w2 > 1e-300 && w2.is_finite()) {
        return (0.0, 0.0);
    }
    (dot(term, whole) / w2, (dot(term, term) / w2).sqrt())
}

/// The order the eight paths of three blocks are listed in: by how many blocks they go
/// through, then by which.
const PATH_ORDER: [usize; 8] = [0b000, 0b001, 0b010, 0b100, 0b011, 0b101, 0b110, 0b111];

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    (0..k).fold(1usize, |c, i| c * (n - i) / (i + 1))
}

/// What the two pictures of paths show for a design: worked out when the design is
/// first chosen, and kept.
#[derive(Clone, Debug)]
struct Analysis {
    design: Design,
    /// Three blocks: each path's share of g_0 and its length over |g_0|, in
    /// `PATH_ORDER`; and s.
    path_share: [f64; 8],
    path_len: [f64; 8],
    s3: f64,
    /// Ten blocks: the same for the paths through k blocks, k from 0 to 10; and s.
    len_share: Vec<f64>,
    len_len: Vec<f64>,
    s10: f64,
}

impl Analysis {
    fn of(d: Design, x: &[f64], y: &[f64], tape: &mut Tape) -> Analysis {
        let (whole, terms) = paths(&model(PATH_BLOCKS, true, d), x, y, tape);
        let mut path_share = [0.0; 8];
        let mut path_len = [0.0; 8];
        for (i, &set) in PATH_ORDER.iter().enumerate() {
            if let Some(t) = terms.get(set) {
                (path_share[i], path_len[i]) = share_and_length(t, &whole);
            }
        }
        let (whole, parts) = by_length(&model(LENGTH_BLOCKS, true, d), x, y, tape);
        let (mut len_share, mut len_len): (Vec<f64>, Vec<f64>) = parts.iter().map(|t| share_and_length(t, &whole)).unzip();
        len_share.resize(LENGTH_BLOCKS + 1, 0.0);
        len_len.resize(LENGTH_BLOCKS + 1, 0.0);
        Analysis { design: d, path_share, path_len, s3: d.scale.of(PATH_BLOCKS), len_share, len_len, s10: d.scale.of(LENGTH_BLOCKS) }
    }

    /// The fewest blocks m such that the paths through at most m of the ten bring at
    /// least 95% of g_0, with the share they bring and how many paths they are.
    fn short_paths(&self) -> (usize, f64, usize) {
        let mut total = 0.0;
        for (m, s) in self.len_share.iter().enumerate() {
            total += s;
            if total >= 0.95 {
                return (m, total, (0..=m).map(|k| binomial(LENGTH_BLOCKS, k)).sum());
            }
        }
        (LENGTH_BLOCKS, total, 1 << LENGTH_BLOCKS)
    }
}

/// What the chart of every layer shows: the plain network and the one with skips at
/// the chosen depth and design.
#[derive(Clone, Debug)]
struct Deep {
    depth: usize,
    design: Design,
    plain: Profile,
    skips: Profile,
    road: Vec<f64>,
}

impl Deep {
    fn of(depth: usize, d: Design, x: &[f64], y: &[f64], tape: &mut Tape) -> Deep {
        let skips_model = model(depth, true, d);
        Deep {
            depth,
            design: d,
            plain: profile(&model(depth, false, d), x, y, tape),
            skips: profile(&skips_model, x, y, tape),
            road: road_profile(&skips_model, x, y, tape),
        }
    }

    /// The gradient at x_0 over the gradient at the top (root mean squares), plain and
    /// with skips.
    fn grad_ratio(&self) -> [f64; 2] {
        [&self.plain, &self.skips].map(|p| ratio(p.grad.first(), p.grad.last()))
    }

    /// The stream at the top over the stream at x_0.
    fn act_ratio(&self) -> [f64; 2] {
        [&self.plain, &self.skips].map(|p| ratio(p.act.last(), p.act.first()))
    }

    /// The straight road's part at x_0 over the gradient at the top.
    fn road_ratio(&self) -> f64 {
        ratio(self.road.first(), self.road.last())
    }
}

fn ratio(a: Option<&f64>, b: Option<&f64>) -> f64 {
    match (a, b) {
        (Some(a), Some(b)) if *b != 0.0 => a / b,
        _ => f64::NAN,
    }
}

/// A short training run of both networks, plain and with skips, on the same points.
struct Run {
    depth: usize,
    design: Design,
    nets: [Model; 2],
    /// The loss before each step, and after the last (plain, then with skips).
    losses: [Vec<f64>; 2],
    tape: Tape,
}

impl Run {
    /// A run of both networks, its first step taken (so that the losses they start at
    /// are known at once).
    fn new(depth: usize, d: Design, x: &[f64], y: &[f64]) -> Run {
        let mut run = Run { depth, design: d, nets: [model(depth, false, d), model(depth, true, d)], losses: [Vec::with_capacity(STEPS + 1), Vec::with_capacity(STEPS + 1)], tape: Tape::new() };
        run.step(x, y);
        run
    }

    fn finished(&self) -> bool {
        self.losses[0].len() > STEPS
    }

    /// One step of Adam for each network (or, after the last, the loss it ends at).
    fn step(&mut self, x: &[f64], y: &[f64]) {
        if self.finished() {
            return;
        }
        for (net, losses) in self.nets.iter_mut().zip(self.losses.iter_mut()) {
            let tr = net.loss_mlp(&mut self.tape, x, y, Loss::CrossEntropy);
            let loss = self.tape.value(tr.loss).first().copied().unwrap_or(f64::NAN);
            losses.push(loss);
            if losses.len() > STEPS {
                continue;
            }
            self.tape.backward(tr.loss);
            net.zero_grads();
            net.accumulate(&self.tape, &tr);
            net.step_adam(RATE);
        }
    }

    /// Steps a second of the scene's time: fewer for a deeper network, so that the
    /// frame's share of the work stays about the same.
    fn rate(&self) -> f32 {
        (1000.0 / self.depth.max(1) as f32).max(20.0).min(60.0)
    }

    /// The last loss of each, as far as the run has got.
    fn latest(&self) -> [f64; 2] {
        [0, 1].map(|k| self.losses[k].last().copied().unwrap_or(f64::NAN))
    }
}

// ----- numbers as the picture writes them --------------------------------------------------

/// A number in a few digits: 0.0123, 1.099, 26.5, 1.6 × 10⁷ (markup, without dollars).
fn short(v: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    let a = v.abs();
    if a == 0.0 {
        "0".to_string()
    } else if !(1e-3..1e4).contains(&a) {
        let mut e = a.log10().floor() as i32;
        let mut m = v / 10f64.powi(e);
        if (m.abs() * 10.0).round() >= 100.0 {
            m /= 10.0;
            e += 1;
        }
        format!(r"{} \times 10^{{{}}}", num(m, 1), e)
    } else if a >= 100.0 {
        num(v, 0)
    } else if a >= 10.0 {
        num(v, 1)
    } else if a >= 0.1 {
        num(v, 3)
    } else {
        num(v, 4)
    }
}

/// A number for a read-out: in a formula, or a dash where there is none.
fn readout(v: f64) -> String {
    if v.is_finite() {
        format!("${}$", short(v))
    } else {
        "—".to_string()
    }
}

/// A share as a percentage, no decimals.
fn percent(v: f64) -> String {
    format!("{}%", num(100.0 * v, 0))
}

fn log10_of(v: f64) -> f32 {
    if v > 0.0 && v.is_finite() {
        v.log10() as f32
    } else {
        -300.0
    }
}

fn smooth(t: f32) -> f32 {
    let t = if t.is_finite() { t.max(0.0).min(1.0) } else { 1.0 };
    t * t * (3.0 - 2.0 * t)
}

// ----- experiments -------------------------------------------------------------------------

struct Trial {
    label: &'static str,
    look_for: &'static str,
    depth: usize,
    scale: Scale,
    norm: NormKind,
    place: Place,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "fifty plain layers, and fifty with skips",
        look_for: r"Fifty blocks, with skips ($s = 1/\sqrt{50}$, solid) and without (dashed). Plain, each layer shrinks the stream and the gradient to about two thirds on average (scene gradflow says why): the gradient that reaches the data is $1 \times 10^{-9}$ of what arrives at the top, and in 100 steps the loss stays at 1.099, which is $\ln 3$, a guess of a third for each class. With skips, the straight road (green) hands $g_{50}$ down unchanged and the blocks add to it: $g_0$ is 4.2 times $g_{50}$, and the loss falls from 6.5 to 0.0004.",
        depth: 50,
        scale: Scale::RootDepth,
        norm: NormKind::None,
        place: Place::Pre,
    },
    Trial {
        label: "three blocks are eight paths",
        look_for: r"Three blocks, $s = 1$ (the chart of every layer shows this same network). Multiplied out, $(I + J_1)^\top (I + J_2)^\top (I + J_3)^\top g_3$ is eight terms, one for each path, adding up to $g_0$ exactly. The straight road's term is $g_3$ itself, handed back unchanged: it brings 35% of $g_0$, the paths through one block 57%, through two 11%, through all three $-2$%. A term's length (its outline) can be well over its share: the path through $f_3$ alone is half as long as $g_0$ but brings 10%; the rest of it points across $g_0$.",
        depth: 3,
        scale: Scale::One,
        norm: NormKind::None,
        place: Place::Pre,
    },
    Trial {
        label: "the short paths carry it",
        look_for: r"Ten blocks, $s = 1/\sqrt{10}$ (the chart of every layer shows this same network). Of its 1024 paths, the 56 through at most two blocks bring 99.97% of $g_0$: the straight road 62%, the ten through one block 28%, the 45 through two 10%. A path through $k$ blocks picks up $k$ Jacobians, and at this $s$ each block hands back through itself at most 0.28 of the gradient that reaches it (in root mean square): the 120 paths through three come to 11% of $|g_0|$ in length and bring nothing along it. This is a fact about this network at this scale, not a law: at $s = 1$ the paths through 2 to 5 blocks bring 82%, the straight road $-0.5$%.",
        depth: 10,
        scale: Scale::RootDepth,
        norm: NormKind::None,
        place: Place::Pre,
    },
    Trial {
        label: "blocks not scaled down: the signal grows",
        look_for: r"The fifty blocks with $s = 1$. Each now adds about as much as the stream holds: its mean square about doubles at every block (1.42 a block in root mean square), to $4 \times 10^7$ times its size at the bottom, and on the way back the gradient grows about as fast, to $2 \times 10^7$ times what arrives at the top. The loss starts at $1.6 \times 10^7$ and is still 26 after 100 steps. With $s = 1/\sqrt{L}$ (the first thing to try) the stream grows 6.8 times; a layer normalisation inside each block (pre) holds it to 24 times.",
        depth: 50,
        scale: Scale::One,
        norm: NormKind::None,
        place: Place::Pre,
    },
    Trial {
        label: "normalised after the sum: the road is not clear",
        look_for: r"Layer normalisation after each sum, $x_l = N(x_{l-1} + f_l(x_{l-1}))$, as in the original transformer; fifty blocks, $s = 1$. Now every step of the straight road passes through a normalisation's Jacobian, which takes out the gradient's parts along two directions (the all-ones one, and the sum's own once its mean is taken off) and divides the rest by the sum's spread. The green line falls: the road alone hands $x_0$ $1/126$ of $g_{50}$, while $g_0$ is 105 times $g_{50}$, all but a sliver of it through blocks. Training wanders and ends at $\ln 3$. Click pre above the picture: the green line is level, and the loss is 0.001 by step 75.",
        depth: 50,
        scale: Scale::One,
        norm: NormKind::Layer,
        place: Place::Post,
    },
];

// ----- where things go ---------------------------------------------------------------------

/// What a button on the picture sets.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Choice {
    Scale(Scale),
    Norm(NormKind),
    Place(Place),
    Depth(usize),
}

/// The depths offered on the picture: the network of three blocks, of ten, and deeper.
const DEPTHS: [usize; 4] = [3, 10, 20, 50];

#[derive(Clone, Debug)]
struct Layout {
    narrow: bool,
    /// The size of the words on the buttons and in the titles, and of the small print.
    text: f32,
    small: f32,
    buttons: Vec<(Rect, Choice)>,
    leads: Vec<(Vec2, &'static str)>,
    /// The four pictures: the paths of three blocks, every layer of the deep network,
    /// the paths of ten blocks by length, and training.
    paths: Option<Rect>,
    layers: Option<Rect>,
    lengths: Option<Rect>,
    training: Option<Rect>,
}

/// What a button says; with `long`, also what it sets (on a phone, where nothing is
/// written before the buttons).
fn words_of(c: Choice, long: bool) -> String {
    match (c, long) {
        (Choice::Scale(Scale::One), false) => "$1$".into(),
        (Choice::Scale(Scale::RootDepth), false) => r"$1/\sqrt{L}$".into(),
        (Choice::Scale(Scale::One), true) => "$s = 1$".into(),
        (Choice::Scale(Scale::RootDepth), true) => r"$s = 1/\sqrt{L}$".into(),
        (Choice::Norm(NormKind::None), true) => "no norm".into(),
        (Choice::Norm(n), _) => n.name().into(),
        (Choice::Place(p), _) => p.name().into(),
        (Choice::Depth(d), false) => format!("{d}"),
        (Choice::Depth(d), true) => format!("$L = {d}$"),
    }
}

/// The buttons, in groups with what is written before each group.
fn groups() -> [(&'static str, Vec<Choice>); 4] {
    [
        ("$s$:", vec![Choice::Scale(Scale::One), Choice::Scale(Scale::RootDepth)]),
        ("norm:", NormKind::ALL.iter().map(|&n| Choice::Norm(n)).collect()),
        ("where:", vec![Choice::Place(Place::Pre), Choice::Place(Place::Post)]),
        ("depth $L$:", DEPTHS.iter().map(|&d| Choice::Depth(d)).collect()),
    ]
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::ONE);
        let narrow = view.x < 560.0;
        let (text, small) = if narrow { (11.5, 10.5) } else { (12.5, 11.5) };
        let side = if narrow { 10.0 } else { 16.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let h = if narrow { 26.0 } else { 28.0 };
        let mut lay = Layout { narrow, text, small, buttons: Vec::new(), leads: Vec::new(), paths: None, layers: None, lengths: None, training: None };
        if view.x < 200.0 || view.y < 160.0 {
            return lay;
        }
        let gap = if narrow { 5.0 } else { 6.0 };
        let body;
        if narrow {
            // two rows filling the width: the scale and the normalisation, then where it
            // goes and the depth
            let [scale, norm, place, depth] = groups();
            let row1: Vec<Choice> = scale.1.iter().chain(&norm.1).copied().collect();
            let row2: Vec<Choice> = place.1.iter().chain(&depth.1).copied().collect();
            for (k, row) in [row1, row2].iter().enumerate() {
                let y = top + k as f32 * (h + gap);
                for (r, c) in plot::even_row(row.len(), side, view.x - side, y, h, gap).into_iter().zip(row) {
                    lay.buttons.push((r, *c));
                }
            }
            body = top + 2.0 * h + gap + 12.0;
        } else {
            // groups side by side, wrapped to a second row if they do not fit
            let (mut x, mut y) = (side, top);
            for (lead, choices) in groups() {
                let lead_w = text_width(lead, text) + 8.0;
                let widths: Vec<f32> = choices.iter().map(|c| text_width(&words_of(*c, false), text) + 20.0).collect();
                let group_w = lead_w + widths.iter().sum::<f32>() + gap * (widths.len() as f32 - 1.0);
                if x > side && x + group_w > view.x - side {
                    (x, y) = (side, y + h + gap);
                }
                lay.leads.push((Vec2::new(x, y + h * 0.5), lead));
                x += lead_w;
                for (c, w) in choices.iter().zip(widths) {
                    lay.buttons.push(((Vec2::new(x, y), Vec2::new(x + w, y + h)), *c));
                    x += w + gap;
                }
                x += 22.0;
            }
            body = y + h + 16.0;
        }
        let bottom = view.y - 8.0;
        let room = bottom - body;
        if narrow {
            if room < 300.0 {
                return lay;
            }
            // one under another; on a short screen the last two side by side
            let tall = room >= 600.0;
            let rows: &[f32] = if tall { &[0.33, 0.25, 0.19, 0.23] } else { &[0.38, 0.30, 0.32] };
            let gaps = 12.0 * (rows.len() - 1) as f32;
            let mut y = body;
            let mut rects = Vec::new();
            for share in rows {
                let hh = ((room - gaps) * share).floor();
                rects.push((Vec2::new(side, y), Vec2::new(view.x - side, y + hh)));
                y += hh + 12.0;
            }
            lay.paths = Some(rects[0]);
            lay.layers = Some(rects[1]);
            if tall {
                (lay.lengths, lay.training) = (Some(rects[2]), Some(rects[3]));
            } else {
                let (min, max) = rects[2];
                let half = ((max.x - min.x - 14.0) * 0.5).floor();
                lay.lengths = Some((min, Vec2::new(min.x + half, max.y)));
                lay.training = Some((Vec2::new(max.x - half, min.y), max));
            }
            return lay;
        }
        if room < 260.0 {
            return lay;
        }
        // two columns: the paths over the paths by length, every layer over training
        let col_gap = 34.0;
        let left_w = ((view.x - 2.0 * side - col_gap) * 0.47).floor();
        let (lx, rx) = (side, side + left_w + col_gap);
        let a_h = (room * 0.62).min(PATHS_H).floor();
        lay.paths = Some((Vec2::new(lx, body), Vec2::new(lx + left_w, body + a_h)));
        lay.lengths = Some((Vec2::new(lx, body + a_h + 16.0), Vec2::new(lx + left_w, bottom)));
        let b_h = (room * 0.55).floor();
        lay.layers = Some((Vec2::new(rx, body), Vec2::new(view.x - side, body + b_h)));
        lay.training = Some((Vec2::new(rx, body + b_h + 16.0), Vec2::new(view.x - side, bottom)));
        lay
    }
}

/// The height the paths of three blocks need at most: title, diagram, headings, nine
/// rows and the key.
const PATHS_H: f32 = 420.0;

// ----- lines by kind -----------------------------------------------------------------------

/// Kinds of line with a batch each (see `plot::apart`): the diagram, the paths' glyphs,
/// the bars' outlines, the chart of every layer (solid, and dashed), training (solid,
/// dashed; the run fading out, solid and dashed), the dashed lines of the keys, and the
/// charts' ticks and powers of ten.
const DIAGRAM: u8 = plot::OWN;
const GLYPHS: u8 = plot::OWN + 1;
const OUTLINES: u8 = plot::OWN + 2;
const LAYERS: u8 = plot::OWN + 3;
const LAYERS_DASHED: u8 = plot::OWN + 4;
const LOSS_SKIPS: u8 = plot::OWN + 5;
const LOSS_PLAIN: u8 = plot::OWN + 6;
const OLD_SKIPS: u8 = plot::OWN + 7;
const OLD_PLAIN: u8 = plot::OWN + 8;
const REFS: u8 = plot::OWN + 9;
const TICKS: u8 = plot::OWN + 10;

const KINDS: [u8; 15] = [plot::AXES, plot::CURVES, plot::MARKS, plot::BUTTONS, DIAGRAM, GLYPHS, OUTLINES, LAYERS, LAYERS_DASHED, LOSS_SKIPS, LOSS_PLAIN, OLD_SKIPS, OLD_PLAIN, REFS, TICKS];

/// Every batch of lines, made in the same order in every frame.
fn in_order(f: &mut Frame) {
    for kind in KINDS {
        plot::apart(f, kind, false);
    }
}

/// A batch with nothing in it this frame is given a point that cannot be seen: one left
/// empty would be dropped, and every batch after it compared, point for point, with the
/// one before it in the last frame.
fn keep_batches(f: &mut Frame) {
    for kind in KINDS {
        let batch = plot::apart(f, kind, false);
        if batch.pts.len() <= 1 {
            batch.segment(Vec3::ZERO, Vec3::ZERO, 0.0, Color::CLEAR);
        }
    }
}

/// The first of `texts` that fits in `width` pixels at `size` (the last if none does):
/// a title or a line with shorter ways of saying it for a narrower picture.
fn fitting(texts: &[String], size: f32, width: f32) -> String {
    // (with a tenth to spare: the width of words is an estimate)
    texts.iter().find(|t| text_width(t, size) <= 0.9 * width).or(texts.last()).cloned().unwrap_or_default()
}

/// The powers of ten up the side of a chart with a logarithmic y axis, as many named as
/// there is room for (at least `apart` pixels apart), with a faint line across at
/// each. A line is drawn for every power of ten a chart here can show, the ones out of
/// range on the edge and unseen, so that each keeps its place in the batch while the
/// range glides.
fn powers_of_ten(f: &mut Frame, p: &Plot, apart: f32, small: f32) {
    let (lo, hi) = (p.y[0], p.y[1]);
    let span = (hi - lo).max(1.0);
    let every = [1, 2, 4, 5, 10].into_iter().find(|e| span / *e as f64 <= (p.height() / apart) as f64).unwrap_or(20);
    let (min, max) = p.rect;
    for e in -40i32..=12 {
        let shown = e as f64 >= lo - 1e-6 && e as f64 <= hi + 1e-6;
        let named = shown && e.rem_euclid(every) == 0;
        let y = p.px_y(e as f64).max(min.y).min(max.y);
        plot::line_of(f, TICKS, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::FAINT.a(if named { 0.35 } else { 0.0 }));
        if named {
            let label = if e == 0 { "$1$".to_string() } else { format!("$10^{{{e}}}$") };
            f.text_px(Vec2::new(min.x - 5.0, y), label, small - 1.0, pal::DIM, Align2::RIGHT_CENTER);
        }
    }
}

/// A dashed polyline in screen space, kept inside `rect` (a point outside it drawn on
/// its edge): every segment between two of its points cut into `per` dashes, or with
/// `per` 0 two segments of every three drawn (for points close together). The dashes go
/// by the points, not by the length along the line, so that while the line moves (a
/// chart's range gliding) each dash stays on its stretch of it and the batch keeps its
/// number of points.
#[allow(clippy::too_many_arguments)]
fn dashed_path(f: &mut Frame, kind: u8, pts: &[Vec2], rect: Rect, per: usize, width: f32, color: Color) {
    let pts: Vec<Vec2> = pts.iter().map(|p| inside(*p, rect)).collect();
    for (i, w) in pts.windows(2).enumerate() {
        let (a, b) = (w[0], w[1]);
        if !(a.is_finite() && b.is_finite()) {
            continue;
        }
        if per == 0 {
            if i % 3 != 2 {
                plot::line_of(f, kind, a, b, width, color);
            }
            continue;
        }
        for j in 0..per {
            let (t0, t1) = (j as f32 / per as f32, (j as f32 + 0.55) / per as f32);
            plot::line_of(f, kind, a.lerp(b, t0), a.lerp(b, t1), width, color);
        }
    }
}

/// A polyline in screen space in the batch of its kind, kept inside `rect`: a point
/// outside it is drawn on its edge. (Not cut there: a line cut at the edge has more or
/// fewer points as it moves across it, and its points no longer follow each other from
/// one frame to the next; only while a chart's range glides is anything outside.)
fn path_in(f: &mut Frame, kind: u8, pts: &[Vec2], rect: Rect, width: f32, color: Color) {
    let kept: Vec<Vec2> = pts.iter().map(|p| inside(*p, rect)).collect();
    plot::path_of(f, kind, &kept, width, color);
}

/// The point, moved onto the rectangle's edge if it is outside (a point that is not a
/// number stays one: it breaks a line).
fn inside(p: Vec2, (min, max): Rect) -> Vec2 {
    if p.is_finite() {
        p.max(min).min(max)
    } else {
        p
    }
}

// ----- the scene ---------------------------------------------------------------------------

/// How long a change of setting takes to fade the old training run out (seconds), and
/// how fast the drawn step follows the steps taken (each of two eases).
const FADE: f32 = 0.4;
const FOLLOW: f64 = 0.08;

pub struct Residual {
    depth: usize,
    scale: Scale,
    norm: NormKind,
    place: Place,
    x: Vec<f64>,
    y: Vec<f64>,
    tape: Tape,
    /// The pictures of paths for every design asked for so far.
    analyses: Vec<Analysis>,
    deep: Deep,
    run: Run,
    pace: Pace,
    /// The step of the run drawn, following the losses known through two eases.
    between: f64,
    shown: f64,
    /// The run before the last change of setting, fading out (its curves as they were
    /// drawn, the step they were drawn to), and how far the fade has gone.
    old: Option<([Vec<f32>; 2], f64)>,
    fade: f32,
    /// The numbers as drawn, gliding to the numbers worked out: the three blocks'
    /// shares and lengths (and the share they add up to), the ten blocks', every
    /// layer's five series (log₁₀), the chart of layers' range (depth, lowest and
    /// highest power of ten) and the training chart's top.
    three: Glide,
    ten: Glide,
    series: [Glide; 5],
    layer_range: Glide,
    loss_top: Glide,
    /// The two loss curves as drawn (the log of the loss at each step), gliding to
    /// `curve_target`: a loss that zigzags from step to step would otherwise turn the
    /// heads round abruptly.
    curves: [Glide; 2],
    moving: bool,
}

impl Residual {
    fn new() -> Residual {
        let t = &TRIALS[0];
        let (x, y) = data();
        let mut tape = Tape::new();
        let d = Design::new(t.scale, t.norm, t.place);
        let deep = Deep::of(t.depth, d, &x, &y, &mut tape);
        let analysis = Analysis::of(d, &x, &y, &mut tape);
        let run = Run::new(t.depth, d, &x, &y);
        let mut scene = Residual {
            depth: t.depth,
            scale: t.scale,
            norm: t.norm,
            place: t.place,
            x,
            y,
            tape,
            analyses: vec![analysis],
            deep,
            run,
            pace: Pace::default(),
            between: 0.0,
            shown: 0.0,
            old: None,
            fade: 1.0,
            three: Glide::default(),
            ten: Glide::default(),
            series: Default::default(),
            layer_range: Glide::default(),
            loss_top: Glide::default(),
            curves: Default::default(),
            moving: false,
        };
        scene.three.snap(&scene.three_target());
        scene.ten.snap(&scene.ten_target());
        let targets = scene.deep_targets();
        for (g, t) in scene.series.iter_mut().zip(targets) {
            g.snap(&t);
        }
        scene.layer_range.snap(&scene.layer_range_target());
        scene.loss_top.snap(&[scene.loss_top_target()]);
        scene
    }

    fn design(&self) -> Design {
        Design::new(self.scale, self.norm, self.place)
    }

    /// Set up experiment `k` of `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k.min(TRIALS.len() - 1)];
        (self.depth, self.scale, self.norm, self.place) = (t.depth, t.scale, t.norm, t.place);
        self.refresh();
    }

    fn choose(&mut self, c: Choice) {
        match c {
            Choice::Scale(s) => self.scale = s,
            Choice::Norm(n) => self.norm = n,
            Choice::Place(p) => self.place = p,
            Choice::Depth(d) => self.depth = d.max(1).min(MAX_DEPTH),
        }
        self.refresh();
    }

    fn is_on(&self, c: Choice) -> bool {
        match c {
            Choice::Scale(s) => self.scale == s,
            Choice::Norm(n) => self.norm == n,
            Choice::Place(p) => self.norm != NormKind::None && self.place == p,
            Choice::Depth(d) => self.depth == d,
        }
    }

    /// Work out what the setting asks for, where it has changed: the pictures of paths
    /// for a new design, the chart of every layer, and a new training run (the old one
    /// fades out).
    fn refresh(&mut self) {
        self.depth = self.depth.max(1).min(MAX_DEPTH);
        let d = self.design();
        if !self.analyses.iter().any(|a| a.design == d) {
            let a = Analysis::of(d, &self.x, &self.y, &mut self.tape);
            self.analyses.push(a);
        }
        if self.deep.depth != self.depth || self.deep.design != d {
            self.deep = Deep::of(self.depth, d, &self.x, &self.y, &mut self.tape);
        }
        if self.run.depth != self.depth || self.run.design != d {
            self.run = Run::new(self.depth, d, &self.x, &self.y);
            self.old = Some((self.curves.clone().map(|g| g.shown), self.shown));
            self.fade = 0.0;
            (self.between, self.shown) = (0.0, 0.0);
            self.pace.reset();
        }
    }

    fn analysis(&self) -> &Analysis {
        let d = self.design();
        self.analyses.iter().find(|a| a.design == d).unwrap_or(&self.analyses[0])
    }

    fn three_target(&self) -> Vec<f32> {
        let a = self.analysis();
        a.path_share.iter().chain(&a.path_len).map(|&v| v as f32).chain([a.path_share.iter().sum::<f64>() as f32]).collect()
    }

    fn ten_target(&self) -> Vec<f32> {
        let a = self.analysis();
        a.len_share.iter().chain(&a.len_len).map(|&v| v as f32).collect()
    }

    /// The five series of the chart of every layer, as powers of ten: the stream and its
    /// gradient plain, the same with skips, and the straight road's part.
    fn deep_targets(&self) -> [Vec<f32>; 5] {
        let d = &self.deep;
        [&d.plain.act, &d.plain.grad, &d.skips.act, &d.skips.grad, &d.road].map(|s| s.iter().map(|&v| log10_of(v)).collect())
    }

    /// The chart of every layer's range: the depth, and whole powers of ten round
    /// everything on it.
    fn layer_range_target(&self) -> Vec<f32> {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for s in self.deep_targets() {
            for v in s.into_iter().filter(|v| *v > -299.0) {
                (lo, hi) = (lo.min(v), hi.max(v));
            }
        }
        if lo > hi {
            (lo, hi) = (-3.0, 0.0);
        }
        let (mut lo, mut hi) = ((lo - 0.15).floor(), (hi + 0.15).ceil());
        if hi - lo < 2.0 {
            lo = hi - 2.0;
        }
        lo = lo.max(hi - 40.0);
        hi = hi.max(lo + 2.0);
        vec![self.depth as f32, lo, hi]
    }

    /// The training chart's top: a whole power of ten above every loss of the run (the
    /// bottom is fixed at 10⁻⁴).
    fn loss_top_target(&self) -> f32 {
        let most = self.run.losses.iter().flatten().filter(|v| v.is_finite()).fold(1.2f64, |a, &b| a.max(b));
        (most.log10() + 0.05).ceil().max(1.0) as f32
    }

    /// Move everything on: training by `dt` seconds of the scene's time, the glides by
    /// `real_dt` of the clock's (a change of setting is seen through even while the
    /// scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        let real = if real_dt.is_finite() { real_dt.max(0.0) } else { 0.0 };
        if !self.run.finished() {
            let mut turn = self.pace.begin(self.run.rate(), dt, real_dt, 0.004);
            while turn.next() {
                self.run.step(&self.x, &self.y);
                if self.run.finished() {
                    break;
                }
            }
            self.pace.end(turn);
        }
        let mut moving = !self.run.finished();
        let to = self.run.losses[0].len().saturating_sub(1) as f64;
        let e = 1.0 - (-(real as f64) / FOLLOW).exp();
        self.between += (to - self.between) * e;
        self.shown += (self.between - self.shown) * e;
        if ((to - self.between).abs() < 1e-3 && (to - self.shown).abs() < 1e-3) || !self.shown.is_finite() || !self.between.is_finite() {
            (self.between, self.shown) = (to, to);
        } else {
            moving = true;
        }
        if self.old.is_some() {
            self.fade += real / FADE;
            if self.fade >= 1.0 || !self.fade.is_finite() {
                (self.fade, self.old) = (1.0, None);
            }
            moving = true;
        }
        let three = self.three_target();
        moving |= self.three.follow(&three, real, false);
        let ten = self.ten_target();
        moving |= self.ten.follow(&ten, real, false);
        let targets = self.deep_targets();
        for (g, t) in self.series.iter_mut().zip(targets) {
            moving |= g.follow(&t, real, true);
        }
        let range = self.layer_range_target();
        moving |= self.layer_range.follow(&range, real, false);
        let top = self.loss_top_target();
        moving |= self.loss_top.follow(&[top], real, false);
        for (g, losses) in self.curves.iter_mut().zip(&self.run.losses) {
            let target = curve_target(losses, self.shown);
            if losses.len() <= 1 {
                // (a run just begun starts where it is)
                g.snap(&target);
            } else {
                moving |= g.follow(&target, real, false);
            }
        }
        self.moving = moving;
    }
}

impl Scene for Residual {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.moving || !self.run.finished()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A skip connection adds a block's input to what the block makes of it: $x_l = x_{l-1} + s\,f_l(x_{l-1})$. Going back, the gradient $g_l$ of the loss at $x_l$ reaches $x_{l-1}$ by two roads at once, unchanged along the skip and through the block, $J_l$ being the Jacobian of $s\,f_l$:");
        ui::equation(ui, r"g_{l-1} = g_l + J_l^\top g_l = (I + J_l)^\top g_l");
        ui::note(ui, r"Through $L$ blocks these factors multiply, and multiplied out they are $2^L$ terms, one for each way of going through some blocks and round the others. One term goes round every block: the gradient at the top reaches the bottom unchanged, by a straight road, whatever the blocks do. (A normalisation placed after the sum puts something on that road: the last thing to try.)");

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.label, t.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The blocks");
        let mut depth = self.depth;
        if ui::slider_int(ui, "depth $L$", &mut depth, 1..=MAX_DEPTH) {
            self.choose(Choice::Depth(depth));
        }
        ui::note(ui, "For the chart of every layer and for training. The paths are drawn for a network of three blocks and counted for one of ten.");
        let mut s = if self.scale == Scale::One { 0 } else { 1 };
        if ui::choice(ui, &["$s = 1$", r"$s = 1/\sqrt{L}$"], &mut s) {
            self.choose(Choice::Scale(if s == 0 { Scale::One } else { Scale::RootDepth }));
        }
        ui::note(ui, r"What each block's result is multiplied by before it is added. With $1/\sqrt{L}$ each network takes its own $L$: 3, 10 or the depth.");
        let mut n = NormKind::ALL.iter().position(|k| *k == self.norm).unwrap_or(0);
        if ui::choice(ui, &["no normalisation", "layer norm", "RMS norm"], &mut n) {
            self.choose(Choice::Norm(NormKind::ALL[n.min(2)]));
        }
        let mut p = if self.place == Place::Pre { 0 } else { 1 };
        if ui::choice(ui, &["pre", "post"], &mut p) {
            self.choose(Choice::Place(if p == 0 { Place::Pre } else { Place::Post }));
        }
        ui::note(ui, "Every block, with $N$ the normalisation:");
        ui::equation(ui, self.formula());

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (FORWARD, r"cyan: the root mean square of the stream $x_l$ at each layer"),
                (BACKWARD, r"pink: the gradient $g_l$ there; on the paths, the stretches through a block"),
                (SKIP, r"green: the skips, and the straight road; on the chart, the part of $g_l$ that came by the straight road alone"),
                (NORMED, "purple: the normalisations"),
                (LOSS, "orange: the loss during training"),
                (pal::DIM, "solid: the network with skips; dashed: the same blocks without them"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Skip connection:", r"also called a residual connection: the block's input is added to its output, so the block only has to make the change, $x_l - x_{l-1}$.");
        ui::term(ui, "Block:", r"what a skip goes round. Here one layer: weights $W_l$, bias $b_l$ and a ReLU, $f_l(x) = \operatorname{ReLU}(W_l x + b_l)$, with a normalisation if one is chosen.");
        ui::term(ui, "Stream:", r"the $x_l$ the blocks add to, also called the residual stream. With skips, $x_0$ is the data mapped to 16 numbers by a layer without a ReLU; plain, $x_0$ is the data.");
        ui::term(ui, r"Jacobian $J_l$:", r"the matrix of the derivatives of block $l$'s result $s\,f_l$ with respect to its input $x_{l-1}$ (scene jacobian). Going back, the gradient is multiplied by its transpose.");
        ui::term(ui, "Path:", r"one of the $2^L$ terms of the product: a choice, block by block, of through ($J_l^\top$) or round ($I$).");
        ui::term(ui, "Share of $g_0$:", r"a term $t$ brings $t \cdot g_0 / |g_0|^2$ of $g_0$: its part along $g_0$, over $g_0$'s length. The terms add up to $g_0$, so their shares add up to 100%. A term can also point partly across $g_0$, so its length can be more than its share.");
        ui::term(ui, "Root mean square:", "the square root of the mean of the squares of the numbers, over every unit and every point: the typical size of an entry.");
        ui::term(ui, "Layer and RMS normalisation:", "each point's 16 numbers are divided by their spread (layer norm, after taking their mean off) or by their root mean square (RMS norm); scene normgrad.");
        ui::term(ui, "Pre and post:", r"the normalisation inside the block, before its weights (pre: the skip carries $x$ untouched), or after the sum (post: every layer of the stream passes through it).");
        ui::term(ui, "The data and the training:", r"24 points of three interleaved spirals, each coordinate scaled to a root mean square of 1, and the cross-entropy loss. Weights start normal with variance $1/n$, $n$ their number of inputs; training is Adam at rate 0.01, on all 24 points at every step.");

        ui::section(ui, "Numbers");
        let d = &self.deep;
        let [gp, gs] = d.grad_ratio();
        let [ap, as_] = d.act_ratio();
        ui::readout(ui, r"$|g_0| / |g_L|$, plain", readout(gp));
        ui::readout(ui, r"$|g_0| / |g_L|$, with skips", readout(gs));
        ui::readout(ui, r"straight road at $x_0$, over $|g_L|$", readout(d.road_ratio()));
        ui::readout(ui, r"$|x_L| / |x_0|$, plain", readout(ap));
        ui::readout(ui, r"$|x_L| / |x_0|$, with skips", readout(as_));
        let a = self.analysis();
        ui::readout(ui, "three blocks: the straight road's share", percent(a.path_share[0]));
        let (m, share, count) = a.short_paths();
        ui::readout(ui, &format!("ten blocks: paths through at most {m}"), format!("{}, {count} paths", percent(share)));
        let [lp, ls] = self.run.latest();
        ui::readout(ui, &format!("loss after {} steps, plain", self.run.losses[0].len().saturating_sub(1)), readout(lp));
        ui::readout(ui, "the same, with skips", readout(ls));
        ui::see_also(
            ui,
            &[
                ("gradflow", "the gradient through a deep plain network"),
                ("normgrad", "what a normalisation does to the gradient"),
                ("expsum", r"a product of things near $I$"),
                ("tfgrad", "the gradient in a transformer, whose blocks have skips"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let covered = view.x < 160.0;
        let lay = Layout::of(view, f.top_inset);
        if !covered {
            // the picture is for its buttons: no moving the view
            f.handles.capture();
            f.handles.claim_clicks();
            self.buttons(f, &lay);
        }
        self.advance(f.dt, f.real_dt);
        if covered {
            return;
        }
        in_order(f);
        if let Some(r) = lay.paths {
            self.draw_paths(f, &lay, r);
        }
        if let Some(r) = lay.layers {
            self.draw_layers(f, &lay, r);
        }
        if let Some(r) = lay.lengths {
            self.draw_lengths(f, &lay, r);
        }
        if let Some(r) = lay.training {
            self.draw_training(f, &lay, r);
        }
        keep_batches(f);
    }
}

impl Residual {
    /// Every block's formula, plain and with skips, for the design.
    fn formula(&self) -> &'static str {
        match self.design().post() {
            None if self.norm == NormKind::None => r"\text{plain: } x_l = \operatorname{ReLU}(W_l x_{l-1} + b_l), \\ \text{skips: } x_l = x_{l-1} + s \operatorname{ReLU}(W_l x_{l-1} + b_l)",
            None => r"\text{plain: } x_l = \operatorname{ReLU}(W_l N(x_{l-1}) + b_l), \\ \text{skips: } x_l = x_{l-1} + s \operatorname{ReLU}(W_l N(x_{l-1}) + b_l)",
            Some(_) => r"\text{plain: } x_l = \operatorname{ReLU}(N(W_l x_{l-1} + b_l)), \\ \text{skips: } x_l = N(x_{l-1} + s \operatorname{ReLU}(W_l x_{l-1} + b_l))",
        }
    }

    /// The buttons over the picture.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        for (at, lead) in &lay.leads {
            f.text_px(*at, *lead, lay.text, pal::DIM, Align2::LEFT_CENTER);
        }
        // (on a phone the words say what each button sets, where they fit on every button
        // of the kind)
        let kind = |c: &Choice| std::mem::discriminant(c);
        let fits = |c: &Choice| lay.buttons.iter().filter(|(_, d)| kind(d) == kind(c)).all(|(r, d)| text_width(&words_of(*d, true), lay.text) <= r.1.x - r.0.x - 8.0);
        let mut chosen = None;
        for (r, c) in &lay.buttons {
            let words = words_of(*c, lay.narrow && fits(c));
            if matches!(c, Choice::Place(_)) && self.norm == NormKind::None {
                // (without a normalisation there is nothing to place: drawn, not offered)
                hud_rect(f, r.0, r.1, Color::hex(0x10141F));
                f.text_px((r.0 + r.1) * 0.5, words, lay.text, pal::FAINT, Align2::CENTER_CENTER);
                continue;
            }
            if plot::button(f, *r, &words, lay.text, self.is_on(*c)) {
                chosen = Some(*c);
            }
        }
        if let Some(c) = chosen {
            self.choose(c);
        }
    }

    /// Three blocks as a diagram, and multiplied out into their eight paths, each with
    /// its share of the gradient at x_0.
    fn draw_paths(&self, f: &mut Frame, lay: &Layout, (min, max): Rect) {
        let a = self.analysis();
        let (w, h) = (max.x - min.x, max.y - min.y);
        if w < 120.0 || h < 120.0 {
            return;
        }
        let s3 = num(a.s3, 2);
        let titles = [
            format!(r"three blocks ($s = {s3}$), multiplied out: eight paths from $x_3$ back to $x_0$"),
            format!("three blocks ($s = {s3}$): eight paths back"),
            "three blocks: eight paths".to_string(),
        ];
        f.text_px(min, fitting(&titles, lay.text, w), lay.text, pal::TEXT, Align2::LEFT_TOP);
        let mut y = min.y + lay.text + 10.0;
        let post = self.design().post().is_some();
        let normed = self.norm != NormKind::None;
        // the diagram, where there is room
        let diagram_h = if h >= 380.0 {
            84.0
        } else if h >= 330.0 {
            74.0
        } else if h >= 250.0 {
            56.0
        } else {
            0.0
        };
        if diagram_h > 0.0 {
            let strip = Strip { left: min.x + 34.0, right: max.x - 40.0, y: y + 14.0, dip: diagram_h - 34.0 };
            self.diagram(f, lay, &strip, normed, post);
            y += diagram_h + 4.0;
        }
        // the rows: glyph, term, bar, share
        let key_h = if h - (y - min.y) >= 9.0 * 17.0 + 34.0 { 2.0 * (lay.small + 4.0) + 4.0 } else { 0.0 };
        let row_h = ((max.y - key_h - y - lay.small - 6.0) / 9.0).min(26.0).floor();
        if row_h < 11.0 {
            return;
        }
        let glyph_w = (w * 0.25).max(64.0).min(118.0);
        let with_terms = w >= 400.0;
        let term_w = if with_terms { (w * 0.30).min(150.0) } else { 0.0 };
        let value_w = 40.0;
        let gx = min.x;
        let tx = gx + glyph_w + 12.0;
        let bx0 = tx + term_w;
        let bx1 = max.x - value_w;
        let zero = bx0 + 0.13 * (bx1 - bx0);
        let unit = 0.78 * (bx1 - bx0);
        let small = lay.small;
        // headings
        f.text_px(Vec2::new(gx, y), "path", small, pal::DIM, Align2::LEFT_TOP);
        if with_terms {
            f.text_px(Vec2::new(tx, y), "its term", small, pal::DIM, Align2::LEFT_TOP);
        }
        f.text_px(Vec2::new(zero, y), "its share of $g_0$", small, pal::DIM, Align2::LEFT_TOP);
        y += small + 6.0;
        let at = |i: usize| y + (i as f32 + 0.5) * row_h;
        for (i, &set) in PATH_ORDER.iter().enumerate() {
            let cy = at(i);
            let route = [set & 1 != 0, set & 2 != 0, set & 4 != 0];
            let strip = Strip { left: gx + 4.0, right: gx + glyph_w - 4.0, y: cy - row_h * 0.18, dip: row_h * 0.42 };
            glyph(f, &strip, route, normed, post);
            if with_terms {
                f.text_px(Vec2::new(tx, cy), term_of(set, post), small, pal::TEXT, Align2::LEFT_CENTER);
            }
            let color = if set == 0 { SKIP } else { BACKWARD };
            let (share, len) = (self.three.get(i) as f64, self.three.get(8 + i) as f64);
            bar(f, zero, unit, (bx0, bx1), cy, row_h * 0.5, share, Some(len), color);
            f.text_px(Vec2::new(max.x, cy), percent(share), small, color.mix(pal::WHITE, 0.25), Align2::RIGHT_CENTER);
            // the groups by how many blocks they pass, set apart
            if matches!(i, 0 | 3 | 6) {
                plot::line_of(f, plot::AXES, Vec2::new(gx, cy + row_h * 0.5), Vec2::new(max.x, cy + row_h * 0.5), 1.0, pal::FAINT.a(0.45));
            }
        }
        // the whole
        let cy = at(8);
        f.text_px(Vec2::new(gx + 4.0, cy), "all eight", small, pal::DIM, Align2::LEFT_CENTER);
        if with_terms {
            f.text_px(Vec2::new(tx, cy), "$g_0$", small, pal::TEXT, Align2::LEFT_CENTER);
        }
        let total = self.three.get(16) as f64;
        bar(f, zero, unit, (bx0, bx1), cy, row_h * 0.5, total, None, pal::TEXT.a(0.75));
        f.text_px(Vec2::new(max.x, cy), percent(total), small, pal::TEXT, Align2::RIGHT_CENTER);
        // the zero line
        plot::line_of(f, plot::AXES, Vec2::new(zero, y - 2.0), Vec2::new(zero, at(8) + row_h * 0.5), 1.0, pal::DIM.a(0.6));
        if key_h > 0.0 {
            let ky = at(8) + row_h * 0.5 + 6.0;
            let lines = [
                [r"bars: what each term brings, measured along $g_0$; the eight add up to 100%", r"bars: each term's share of $g_0$ (they add up to 100%)", r"bars: shares of $g_0$"],
                [r"outlines: each term's length, over $|g_0|$ (a term can also point across $g_0$)", r"outlines: each term's length, over $|g_0|$", r"outlines: lengths"],
            ];
            for (k, line) in lines.iter().enumerate() {
                let line = fitting(&line.map(String::from), small - 0.5, w);
                f.text_px(Vec2::new(gx, ky + k as f32 * (small + 4.0)), line, small - 0.5, pal::DIM, Align2::LEFT_TOP);
            }
        }
    }

    /// The three blocks as a diagram: the stream and its skips along the top, each
    /// block hanging below, the normalisations where they are, and the gradient at
    /// each end.
    fn diagram(&self, f: &mut Frame, lay: &Layout, strip: &Strip, normed: bool, post: bool) {
        let small = lay.small;
        let y = strip.y;
        plot::line_of(f, DIAGRAM, Vec2::new(strip.left, y), Vec2::new(strip.right, y), 2.4, SKIP);
        for l in 1..=3 {
            let p = strip.parts(l, normed, post);
            let low = y + strip.dip;
            for (a, b) in [(Vec2::new(p.out, y), Vec2::new(p.out, low)), (Vec2::new(p.out, low), Vec2::new(p.plus, low)), (Vec2::new(p.plus, low), Vec2::new(p.plus, y + 5.0))] {
                plot::line_of(f, DIAGRAM, a, b, 1.5, pal::DIM);
            }
            // the block
            let half = Vec2::new(p.box_half, 10.0);
            hud_rect(f, Vec2::new(p.block, low) - half, Vec2::new(p.block, low) + half, Color::hex(0x1A2233));
            outline(f, DIAGRAM, Vec2::new(p.block, low) - half, Vec2::new(p.block, low) + half, 1.2, pal::TEXT.a(0.7));
            f.text_px(Vec2::new(p.block, low), format!("$f_{l}$"), small, pal::TEXT, Align2::CENTER_CENTER);
            f.text_px(Vec2::new(p.plus + 4.0, (y + low) * 0.5 + 2.0), "$s$", small - 1.0, pal::DIM, Align2::LEFT_CENTER);
            // the sum
            hud_rect(f, Vec2::new(p.plus - 5.0, y - 5.0), Vec2::new(p.plus + 5.0, y + 5.0), Color::hex(0x0B0F19));
            outline(f, DIAGRAM, Vec2::new(p.plus - 5.0, y - 5.0), Vec2::new(p.plus + 5.0, y + 5.0), 1.2, pal::TEXT.a(0.8));
            plot::line_of(f, DIAGRAM, Vec2::new(p.plus - 3.0, y), Vec2::new(p.plus + 3.0, y), 1.2, pal::TEXT);
            plot::line_of(f, DIAGRAM, Vec2::new(p.plus, y - 3.0), Vec2::new(p.plus, y + 3.0), 1.2, pal::TEXT);
            // the normalisation, inside the block or on the stream after the sum
            if let Some(nx) = p.norm {
                let ny = if post { y } else { low };
                let half = Vec2::new(p.norm_half, 8.0);
                hud_rect(f, Vec2::new(nx, ny) - half, Vec2::new(nx, ny) + half, Color::hex(0x1F1838));
                outline(f, DIAGRAM, Vec2::new(nx, ny) - half, Vec2::new(nx, ny) + half, 1.2, NORMED);
                f.text_px(Vec2::new(nx, ny), "$N$", small - 1.0, NORMED.mix(pal::WHITE, 0.3), Align2::CENTER_CENTER);
            }
            f.text_px(Vec2::new(p.right, y - 7.0), format!("$x_{l}$"), small, pal::DIM, Align2::CENTER_BOTTOM);
        }
        f.text_px(Vec2::new(strip.left, y - 7.0), "$x_0$", small, pal::DIM, Align2::CENTER_BOTTOM);
        // the gradient comes in at the top and leaves at the bottom
        let (r, l) = (strip.right, strip.left);
        arrow(f, Vec2::new(r + 34.0, y + 16.0), Vec2::new(r + 6.0, y + 16.0), BACKWARD);
        f.text_px(Vec2::new(r + 20.0, y + 20.0), "$g_3$", small, BACKWARD, Align2::CENTER_TOP);
        arrow(f, Vec2::new(l - 6.0, y + 16.0), Vec2::new(l - 32.0, y + 16.0), BACKWARD);
        f.text_px(Vec2::new(l - 20.0, y + 20.0), "$g_0$", small, BACKWARD, Align2::CENTER_TOP);
    }

    /// The chart of every layer: the root mean square of the stream and of its gradient,
    /// plain and with skips, and the straight road's part.
    fn draw_layers(&self, f: &mut Frame, lay: &Layout, (min, max): Rect) {
        let small = lay.small;
        let w = max.x - min.x;
        let l = self.depth;
        let titles = [format!("every layer of the network, $L = {l}$: the root mean squares"), format!("every layer, $L = {l}$ (root mean squares)"), format!("every layer, $L = {l}$")];
        f.text_px(min, fitting(&titles, lay.text, w), lay.text, pal::TEXT, Align2::LEFT_TOP);
        // the key, as much of it as fits on a line
        let ky = min.y + lay.text + 10.0;
        let mut x = min.x;
        let long = [r"the stream $x_l$", r"its gradient $g_l$", r"the straight road's part of $g_l$"];
        let long_w: f32 = long.iter().map(|t| text_width(t, small) + 32.0).sum();
        let words = if long_w <= w { long } else { ["$x_l$", "$g_l$", "straight road"] };
        let entries = [(FORWARD, words[0]), (BACKWARD, words[1]), (SKIP, words[2])];
        for (c, t) in entries {
            plot::line_of(f, plot::MARKS, Vec2::new(x, ky), Vec2::new(x + 14.0, ky), 2.2, c);
            f.text_px(Vec2::new(x + 18.0, ky), t, small, pal::DIM, Align2::LEFT_CENTER);
            x += 18.0 + f.text_size(t, small).x + 14.0;
        }
        let ky2 = ky + small + 6.0;
        plot::line_of(f, plot::MARKS, Vec2::new(min.x, ky2), Vec2::new(min.x + 14.0, ky2), 2.0, pal::DIM);
        f.text_px(Vec2::new(min.x + 18.0, ky2), "with skips", small, pal::DIM, Align2::LEFT_CENTER);
        let x2 = min.x + 18.0 + f.text_size("with skips", small).x + 14.0;
        dashed_path(f, REFS, &[Vec2::new(x2, ky2), Vec2::new(x2 + 16.0, ky2)], (min, max), 3, 2.0, pal::DIM);
        f.text_px(Vec2::new(x2 + 20.0, ky2), "plain", small, pal::DIM, Align2::LEFT_CENTER);
        let rect = (Vec2::new(min.x + 40.0, ky2 + small + 8.0), Vec2::new(max.x - 6.0, max.y - 30.0));
        let depth = self.layer_range.get(0).max(1.0) as f64;
        let (lo, hi) = (self.layer_range.get(1) as f64, self.layer_range.get(2) as f64);
        let p = Plot::new(rect, [0.0, depth], [lo, hi]);
        if p.tiny() {
            return;
        }
        powers_of_ten(f, &p, 26.0, small);
        // whole layers along the bottom: a tick for every layer the depth control
        // offers, those not named (and those past the depth) unseen, so that each
        // keeps its place in the batch while the depth glides
        plot::line_of(f, plot::AXES, Vec2::new(rect.0.x, rect.1.y), rect.1, 1.0, pal::DIM.a(0.8));
        let step = plot::round_step(depth / (p.width() as f64 / 60.0).max(1.0)).max(1.0) as usize;
        for l in 0..=MAX_DEPTH {
            let x = p.px_x(l as f64).min(rect.1.x);
            let named = l % step == 0 && l as f64 <= depth + 1e-6;
            plot::line_of(f, TICKS, Vec2::new(x, rect.1.y), Vec2::new(x, rect.1.y + 4.0), 1.0, pal::DIM.a(if named { 0.8 } else { 0.0 }));
            if named {
                f.text_px(Vec2::new(x, rect.1.y + 6.0), format!("{l}"), small - 1.0, pal::DIM, Align2::CENTER_TOP);
            }
        }
        f.text_px(Vec2::new(rect.1.x, rect.1.y + 8.0 + small), "layer $l$", small - 1.0, pal::DIM, Align2::RIGHT_TOP);
        // the series: plain dashed, with skips solid, the road thinner
        let pts = |g: &Glide| -> Vec<Vec2> { g.shown.iter().enumerate().map(|(l, &v)| if v > -299.0 { p.at(l as f64, v as f64) } else { Vec2::NAN }).collect() };
        let dots = self.depth <= 12;
        // (dashes about nine pixels apart, so many to a layer as the depth asked for allows)
        let per = ((p.width() / self.depth.max(1) as f32 / 9.0).round() as usize).max(1).min(40);
        for (k, color) in [(0, FORWARD), (1, BACKWARD)] {
            dashed_path(f, LAYERS_DASHED, &pts(&self.series[k]), rect, per, 1.8, color.a(0.85));
        }
        for (k, color, width) in [(4, SKIP, 1.6), (2, FORWARD, 2.2), (3, BACKWARD, 2.2)] {
            let line = pts(&self.series[k]);
            path_in(f, LAYERS, &line, rect, width, color);
            if dots && k != 4 {
                for q in &line {
                    plot::disc(f, inside(*q, rect), 5.0, color);
                }
            }
        }
    }

    /// The ten blocks' gradient at x_0, by how many blocks its paths go through.
    fn draw_lengths(&self, f: &mut Frame, lay: &Layout, (min, max): Rect) {
        let a = self.analysis();
        let small = lay.small;
        let w = max.x - min.x;
        let s10 = num(a.s10, 2);
        let titles = [
            format!("ten blocks ($s = {s10}$): $g_0$ by how many blocks its paths go through"),
            format!("ten blocks ($s = {s10}$): $g_0$ by blocks passed"),
            "ten blocks: by blocks passed".to_string(),
            "ten blocks".to_string(),
        ];
        f.text_px(min, fitting(&titles, lay.text, w), lay.text, pal::TEXT, Align2::LEFT_TOP);
        let (m, share, count) = a.short_paths();
        let (share, blocks) = (percent(share), if m == 1 { "block" } else { "blocks" });
        let lines = if m == 0 {
            vec![format!("the straight road alone brings {share} of it")]
        } else {
            vec![format!("the {count} paths through at most {m} of the 10 blocks bring {share} of it"), format!("paths through at most {m} {blocks}: {share}"), format!("at most {m} {blocks}: {share}")]
        };
        f.text_px(Vec2::new(min.x, min.y + lay.text + 8.0), fitting(&lines, small, w), small, pal::DIM, Align2::LEFT_TOP);
        let rect = (Vec2::new(min.x + 40.0, min.y + lay.text + small + 22.0), Vec2::new(max.x - 4.0, max.y - 2.0 * (small + 4.0) - 6.0));
        let p = Plot::new(rect, [-0.5, LENGTH_BLOCKS as f64 + 0.5], [-0.12, 1.12]);
        if p.tiny() {
            return;
        }
        for (v, name) in [(0.0, "0%"), (0.5, "50%"), (1.0, "100%")] {
            let y = p.px_y(v);
            plot::line_of(f, plot::AXES, Vec2::new(rect.0.x, y), Vec2::new(rect.1.x, y), 1.0, pal::FAINT.a(if v == 0.0 { 0.9 } else { 0.35 }));
            f.text_px(Vec2::new(rect.0.x - 5.0, y), name, small - 1.0, pal::DIM, Align2::RIGHT_CENTER);
        }
        let cell = p.width() / (LENGTH_BLOCKS + 1) as f32;
        for k in 0..=LENGTH_BLOCKS {
            let (share, len) = (self.ten.get(k) as f64, self.ten.get(LENGTH_BLOCKS + 1 + k) as f64);
            let color = if k == 0 { SKIP } else { BACKWARD };
            let gap = (cell * 0.18).max(1.0);
            p.bar(f, k as f64 - 0.5, k as f64 + 0.5, 0.0, share, gap, color.a(0.85));
            let (x0, x1) = (p.px_x(k as f64 - 0.5) + gap, p.px_x(k as f64 + 0.5) - gap);
            let (y0, y1) = (p.px_y(0.0), p.px_y(len.min(1.12)));
            outline(f, OUTLINES, Vec2::new(x0, y1), Vec2::new(x1, y0), 1.0, color.a(0.6));
            let x = p.px_x(k as f64);
            f.text_px(Vec2::new(x, rect.1.y + 4.0), format!("{k}"), small - 0.5, pal::TEXT, Align2::CENTER_TOP);
            if cell >= 22.0 {
                f.text_px(Vec2::new(x, rect.1.y + small + 7.0), format!("{}", binomial(LENGTH_BLOCKS, k)), small - 1.5, pal::DIM, Align2::CENTER_TOP);
            }
        }
        if !lay.narrow {
            f.text_px(Vec2::new(rect.0.x - 6.0, rect.1.y + 4.0), "blocks", small - 1.5, pal::DIM, Align2::RIGHT_TOP);
            f.text_px(Vec2::new(rect.0.x - 6.0, rect.1.y + small + 7.0), "paths", small - 1.5, pal::DIM, Align2::RIGHT_TOP);
        }
    }

    /// Training both networks: the loss step by step, plain dashed and with skips solid.
    fn draw_training(&self, f: &mut Frame, lay: &Layout, (min, max): Rect) {
        let small = lay.small;
        let w = max.x - min.x;
        let l = self.depth;
        let titles = [format!("training both, $L = {l}$: the loss over {STEPS} steps (Adam, rate 0.01)"), format!("training both, $L = {l}$: the loss"), format!("training, $L = {l}$")];
        f.text_px(min, fitting(&titles, lay.text, w), lay.text, pal::TEXT, Align2::LEFT_TOP);
        let ky = min.y + lay.text + 10.0;
        plot::line_of(f, plot::MARKS, Vec2::new(min.x, ky), Vec2::new(min.x + 14.0, ky), 2.2, LOSS);
        f.text_px(Vec2::new(min.x + 18.0, ky), "with skips", small, pal::DIM, Align2::LEFT_CENTER);
        let x2 = min.x + 18.0 + f.text_size("with skips", small).x + 14.0;
        dashed_path(f, REFS, &[Vec2::new(x2, ky), Vec2::new(x2 + 16.0, ky)], (min, max), 3, 2.0, LOSS);
        f.text_px(Vec2::new(x2 + 20.0, ky), "plain", small, pal::DIM, Align2::LEFT_CENTER);
        let x3 = x2 + 20.0 + f.text_size("plain", small).x + 14.0;
        let guess = fitting(&[r"$\ln 3$: a guess of $1/3$ for each class".to_string(), r"$\ln 3$: a guess".to_string(), r"$\ln 3$".to_string()], small, max.x - x3 - 18.0);
        // (drawn in every frame, unseen where there is no room for it, so that the batch
        // of marks keeps its number of points)
        let room = text_width(&guess, small) <= max.x - x3 - 18.0;
        plot::line_of(f, plot::MARKS, Vec2::new(x3, ky).min(max), Vec2::new(x3 + 14.0, ky).min(max), 1.0, pal::DIM.a(if room { 0.6 } else { 0.0 }));
        if room {
            f.text_px(Vec2::new(x3 + 18.0, ky), guess, small, pal::DIM, Align2::LEFT_CENTER);
        }
        let rect = (Vec2::new(min.x + 40.0, ky + small), Vec2::new(max.x - 44.0, max.y - 30.0));
        let top = self.loss_top.get(0).max(1.0) as f64;
        let p = Plot::new(rect, [0.0, STEPS as f64], [-4.0, top]);
        if p.tiny() {
            return;
        }
        powers_of_ten(f, &p, 24.0, small);
        p.x_axis(f, "", small - 1.0);
        f.text_px(Vec2::new(rect.1.x, rect.1.y + 8.0 + small), "step", small - 1.0, pal::DIM, Align2::RIGHT_TOP);
        // a guess of a third for each class
        let ln3 = 3f64.ln().log10();
        plot::line_of(f, plot::AXES, p.at(0.0, ln3), p.at(STEPS as f64, ln3), 1.0, pal::DIM.a(0.45));
        // the run fading out, then the run
        if let Some((ys, shown)) = &self.old {
            let alpha = 1.0 - smooth(self.fade);
            for (k, kind) in [(1, OLD_SKIPS), (0, OLD_PLAIN)] {
                curve(f, &p, &ys[k], *shown, kind, k == 0, LOSS.a(0.9 * alpha));
            }
        }
        let alpha = if self.old.is_some() { smooth(self.fade) } else { 1.0 };
        let losses = &self.run.losses;
        let heads = [0, 1].map(|k| curve(f, &p, &self.curves[k].shown, self.shown, if k == 0 { LOSS_PLAIN } else { LOSS_SKIPS }, k == 0, LOSS.a(alpha)));
        let values = [0, 1].map(|k| head_at(&losses[k], self.shown).map_or(f64::NAN, |h| h.1));
        // the losses at the heads beside them, kept apart
        if let ([Some(h0), Some(h1)], [v0, v1]) = (heads, values) {
            let (mut y0, mut y1) = (h0.y, h1.y);
            let apart = small + 3.0;
            if (y0 - y1).abs() < apart {
                let mid = (y0 + y1) * 0.5;
                let sign = if y0 <= y1 { -1.0 } else { 1.0 };
                (y0, y1) = (mid + sign * apart * 0.5, mid - sign * apart * 0.5);
            }
            let color = LOSS.mix(pal::WHITE, 0.3).a(alpha);
            f.text_px(Vec2::new(rect.1.x + 4.0, y0), format!("${}$", short(v0)), small - 1.0, color, Align2::LEFT_CENTER);
            f.text_px(Vec2::new(rect.1.x + 4.0, y1), format!("${}$", short(v1)), small - 1.0, color, Align2::LEFT_CENTER);
        }
    }
}

/// Where the head of a loss curve is at the step `now` (a fraction between two steps):
/// the log of the loss there, straight between the two, and the loss.
fn head_at(losses: &[f64], now: f64) -> Option<(f64, f64)> {
    let last = losses.len().checked_sub(1)?;
    let now = now.max(0.0).min(last as f64);
    let k0 = (now.floor() as usize).min(last);
    let k1 = (k0 + 1).min(last);
    let u = now - k0 as f64;
    let y = |k: usize| losses[k].max(1e-300).log10();
    Some((y(k0) + (y(k1) - y(k0)) * u, losses[k0] + (losses[k1] - losses[k0]) * u))
}

/// The loss curve as it is to be drawn up to the step `now`: the log of each step's
/// loss up to there, and beyond it the head's (between the two steps either side of
/// `now`), so that the line has as many points in every frame. Empty before the first
/// loss.
fn curve_target(losses: &[f64], now: f64) -> Vec<f32> {
    let Some((head, _)) = head_at(losses, now) else { return Vec::new() };
    (0..=STEPS).map(|k| if k as f64 <= now && k < losses.len() { losses[k].max(1e-300).log10() as f32 } else { head as f32 }).collect()
}

/// A loss curve as drawn: `ys` (see `curve_target`, gliding to it) at the steps up to
/// `now`, the rest gathered at the head, where a dot is. Returns where the head is.
#[allow(clippy::too_many_arguments)]
fn curve(f: &mut Frame, p: &Plot, ys: &[f32], now: f64, kind: u8, dashed: bool, color: Color) -> Option<Vec2> {
    if ys.len() != STEPS + 1 {
        return None;
    }
    let now = now.max(0.0).min(STEPS as f64);
    let pts: Vec<Vec2> = ys.iter().enumerate().map(|(k, &y)| p.at((k as f64).min(now), y as f64)).collect();
    if dashed {
        dashed_path(f, kind, &pts, p.rect, 0, 1.8, color);
    } else {
        path_in(f, kind, &pts, p.rect, 2.2, color);
    }
    let at = inside(pts[(now.ceil() as usize).min(STEPS)], p.rect);
    plot::disc(f, at, 5.5, color);
    Some(at)
}

/// The term of a path through three blocks: the factors it picks up, from x_0 up.
fn term_of(set: usize, post: bool) -> String {
    let mut t = String::from("$");
    for l in 1..=3 {
        if set & (1 << (l - 1)) != 0 {
            t += &format!(r"J_{l}^\top ");
        }
        if post {
            t += &format!(r"N_{l}^\top ");
        }
    }
    t + "g_3$"
}

/// Where the parts of a drawing of three blocks go: a stretch of the road (the stream)
/// at height `y`, the blocks hanging `dip` below it.
struct Strip {
    left: f32,
    right: f32,
    y: f32,
    dip: f32,
}

/// The places along one block's stretch of road.
struct Parts {
    right: f32,
    /// Where the branch leaves the road, where it comes back (the sum), and the block's
    /// centre and half-width.
    out: f32,
    plus: f32,
    block: f32,
    box_half: f32,
    /// The normalisation's centre and half-width (inside the branch, or on the road
    /// after the sum).
    norm: Option<f32>,
    norm_half: f32,
}

impl Strip {
    fn parts(&self, l: usize, normed: bool, post: bool) -> Parts {
        let w = (self.right - self.left) / 3.0;
        let a = self.left + (l - 1) as f32 * w;
        // (after the sum, the normalisation needs room on the road before the next block)
        let (block, plus, norm) = match (normed, post) {
            (true, false) => (a + 0.55 * w, a + 0.78 * w, Some(a + 0.27 * w)),
            (true, true) => (a + 0.38 * w, a + 0.63 * w, Some(a + 0.83 * w)),
            _ => (a + 0.45 * w, a + 0.78 * w, None),
        };
        let box_half = if post { 0.13 * w } else { 0.15 * w };
        Parts { right: a + w, out: a + 0.12 * w, plus, block, box_half, norm, norm_half: if post { 0.08 * w } else { 0.09 * w } }
    }
}

/// One path as a glyph: the road and the blocks faint, and the path's route bright, green
/// where it goes round a block and pink where it goes through.
fn glyph(f: &mut Frame, strip: &Strip, route: [bool; 3], normed: bool, post: bool) {
    let y = strip.y;
    let low = y + strip.dip;
    for l in 1..=3 {
        let p = strip.parts(l, normed, post);
        let a = p.right - (strip.right - strip.left) / 3.0;
        let through = route[l - 1];
        let faint = pal::FAINT.a(0.8);
        // the block, small
        let half = Vec2::new(p.box_half, strip.dip * 0.28);
        outline(f, GLYPHS, Vec2::new(p.block, low) - half, Vec2::new(p.block, low) + half, 1.0, if through { pal::DIM } else { faint });
        if through {
            plot::path_of(f, GLYPHS, &[Vec2::new(a, y), Vec2::new(p.out, y), Vec2::new(p.out, low), Vec2::new(p.plus, low), Vec2::new(p.plus, y), Vec2::new(p.right, y)], 1.8, BACKWARD);
        } else {
            plot::path_of(f, GLYPHS, &[Vec2::new(p.out, y), Vec2::new(p.out, low), Vec2::new(p.plus, low), Vec2::new(p.plus, y)], 1.0, faint);
            plot::line_of(f, GLYPHS, Vec2::new(a, y), Vec2::new(p.right, y), 1.8, SKIP);
        }
        if let Some(nx) = p.norm {
            let ny = if post { y } else { low };
            plot::disc(f, Vec2::new(nx, ny), 5.0, NORMED);
        }
    }
}

/// A rectangle's outline in the batch of its kind.
fn outline(f: &mut Frame, kind: u8, min: Vec2, max: Vec2, width: f32, color: Color) {
    plot::path_of(f, kind, &[min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min], width, color);
}

/// An arrow on the screen, its head at `to`.
fn arrow(f: &mut Frame, from: Vec2, to: Vec2, color: Color) {
    let d = (to - from).normalize_or_zero();
    let n = Vec2::new(-d.y, d.x);
    plot::line_of(f, DIAGRAM, from, to, 2.0, color);
    plot::line_of(f, DIAGRAM, to, to - d * 7.0 + n * 4.0, 2.0, color);
    plot::line_of(f, DIAGRAM, to, to - d * 7.0 - n * 4.0, 2.0, color);
}

/// A bar of a row of the paths: filled from the zero line to `share`, its outline to
/// `len`, `height` tall about `cy`; kept inside the bars' stretch `(x0, x1)`.
#[allow(clippy::too_many_arguments)]
fn bar(f: &mut Frame, zero: f32, unit: f32, (x0, x1): (f32, f32), cy: f32, height: f32, share: f64, len: Option<f64>, color: Color) {
    let at = |v: f64| (zero + unit * v as f32).max(x0).min(x1);
    let (a, b) = (at(share.min(0.0)), at(share.max(0.0)));
    if b - a >= 0.5 {
        hud_rect(f, Vec2::new(a, cy - height * 0.5), Vec2::new(b, cy + height * 0.5), color.a(0.8));
    }
    if let Some(len) = len {
        let h = height * 0.5 + 2.0;
        outline(f, OUTLINES, Vec2::new(zero, cy - h), Vec2::new(at(len), cy + h), 1.0, color.a(0.55));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DESIGNS: [(Scale, NormKind, Place); 5] = [
        (Scale::RootDepth, NormKind::None, Place::Pre),
        (Scale::One, NormKind::Layer, Place::Pre),
        (Scale::One, NormKind::Layer, Place::Post),
        (Scale::RootDepth, NormKind::Rms, Place::Pre),
        (Scale::One, NormKind::Rms, Place::Post),
    ];

    fn designs() -> impl Iterator<Item = Design> {
        DESIGNS.into_iter().map(|(s, n, p)| Design::new(s, n, p))
    }

    // ----- the networks by hand, from the library's description of them ------------------

    fn group(model: &Model, name: &str) -> Option<Vec<f64>> {
        model.group(name).map(|g| g.value.clone())
    }

    /// x Wᵀ + b, for rows of `cols` numbers.
    fn linear(x: &[f64], cols: usize, w: &[f64], b: &[f64]) -> Vec<f64> {
        let out = b.len();
        x.chunks(cols).flat_map(|x| (0..out).map(move |j| (0..cols).map(|k| x[k] * w[j * cols + k]).sum::<f64>() + b[j])).collect()
    }

    fn relu(x: Vec<f64>) -> Vec<f64> {
        x.into_iter().map(|v| v.max(0.0)).collect()
    }

    /// Block l's normalisation of rows of `WIDTH` numbers (none if the block has none).
    fn normalise(model: &Model, l: usize, x: &[f64]) -> Vec<f64> {
        let Some(gain) = group(model, &format!("layer {l} norm gain")) else { return x.to_vec() };
        let shift = group(model, &format!("layer {l} norm bias"));
        let centre = model.mlp_spec().unwrap().norm == Norm::Layer;
        x.chunks(WIDTH)
            .flat_map(|r| {
                let mean = if centre { r.iter().sum::<f64>() / WIDTH as f64 } else { 0.0 };
                let spread = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / WIDTH as f64;
                let inv = 1.0 / (spread + arch::EPS).sqrt();
                let (gain, shift) = (gain.clone(), shift.clone());
                (0..WIDTH).map(move |j| (r[j] - mean) * inv * gain[j] + shift.as_ref().map_or(0.0, |s| s[j]))
            })
            .collect()
    }

    /// What block l of a network with skips adds, s f_l(x) (pre: of the normalised x),
    /// and what it hands on.
    fn branch(model: &Model, l: usize, x: &[f64]) -> Vec<f64> {
        let s = model.mlp_spec().unwrap();
        let (w, b) = (group(model, &format!("layer {l} weights")).unwrap(), group(model, &format!("layer {l} bias")).unwrap());
        let input = if s.place == Place::Pre { normalise(model, l, x) } else { x.to_vec() };
        relu(linear(&input, WIDTH, &w, &b)).into_iter().map(|v| s.residual_scale * v).collect()
    }

    fn block(model: &Model, l: usize, x: &[f64]) -> Vec<f64> {
        let s = model.mlp_spec().unwrap();
        let cols = x.len() / POINTS;
        let (w, b) = (group(model, &format!("layer {l} weights")).unwrap(), group(model, &format!("layer {l} bias")).unwrap());
        match (s.residual, s.place) {
            (false, Place::Post) => relu(normalise(model, l, &linear(x, cols, &w, &b))),
            (false, Place::Pre) => relu(linear(&normalise(model, l, x), cols, &w, &b)),
            (true, Place::Pre) => x.iter().zip(branch(model, l, x)).map(|(a, b)| a + b).collect(),
            (true, Place::Post) => normalise(model, l, &x.iter().zip(branch(model, l, x)).map(|(a, b)| a + b).collect::<Vec<_>>()),
        }
    }

    /// The loss as a function of the stream at layer `from`: the blocks above it and
    /// the read-out, by hand.
    fn top_loss(model: &Model, from: usize, xl: &[f64], y: &[f64]) -> f64 {
        let s = model.mlp_spec().unwrap();
        let mut x = xl.to_vec();
        for l in from + 1..=s.depth {
            x = block(model, l, &x);
        }
        let logits = linear(&x, WIDTH, &group(model, "read-out weights").unwrap(), &group(model, "read-out bias").unwrap());
        let total: f64 = logits
            .chunks(3)
            .zip(y)
            .map(|(z, &t)| {
                let top = z.iter().fold(f64::MIN, |a, &b| a.max(b));
                top + z.iter().map(|v| (v - top).exp()).sum::<f64>().ln() - z[t as usize]
            })
            .sum();
        total / y.len() as f64
    }

    // ----- every gradient shown, against finite differences ------------------------------

    /// The gradient of the loss at the stream, at the bottom, the middle and the top, in
    /// plain networks and with skips: the tape's against central differences of the
    /// loss worked out by hand from the stream up (which also holds the formulas the
    /// panel writes to what the library computes).
    #[test]
    fn the_gradients_at_every_layer_are_those_of_finite_differences() {
        let (x, y) = data();
        let mut tape = Tape::new();
        let mut rng = Rng::new(5);
        for d in designs() {
            for residual in [false, true] {
                let depth = 4;
                let m = model(depth, residual, d);
                let tr = m.loss_mlp(&mut tape, &x, &y, Loss::CrossEntropy);
                tape.backward(tr.loss);
                let loss = tape.value(tr.loss)[0];
                let x0 = tape.value(stream(&tr, 0).unwrap()).to_vec();
                assert!((top_loss(&m, 0, &x0, &y) - loss).abs() < 1e-12, "{d:?} {residual}: the network by hand is not the library's");
                for l in [0, 2, depth] {
                    let v = stream(&tr, l).unwrap();
                    let (value, grad) = (tape.value(v).to_vec(), tape.grad(v).to_vec());
                    let largest = grad.iter().fold(0.0f64, |a, b| a.max(b.abs()));
                    for _ in 0..6 {
                        let k = rng.below(value.len());
                        let h = 1e-6 * (1.0 + value[k].abs());
                        let mut moved = value.clone();
                        moved[k] += h;
                        let up = top_loss(&m, l, &moved, &y);
                        moved[k] -= 2.0 * h;
                        let down = top_loss(&m, l, &moved, &y);
                        let fd = (up - down) / (2.0 * h);
                        assert!((fd - grad[k]).abs() <= 1e-6 * grad[k].abs().max(fd.abs()) + 1e-7 * largest, "{d:?} {residual} layer {l} entry {k}: tape {}, differences {fd}", grad[k]);
                    }
                }
            }
        }
    }

    /// Each road back through a block hands the gradient back by the transposed
    /// Jacobian of what it goes through: v · (J u) by finite differences of the block by
    /// hand equals (Jᵀ v) · u from the tape, for random v and u.
    #[test]
    fn each_road_hands_back_its_jacobian() {
        let (x, y) = data();
        let mut tape = Tape::new();
        let mut rng = Rng::new(6);
        let mut random = |n: usize| -> Vec<f64> { (0..n).map(|_| rng.normal() as f64).collect() };
        for d in designs() {
            let m = model(3, true, d);
            let post = post_of(&m);
            let tr = m.loss_mlp(&mut tape, &x, &y, Loss::CrossEntropy);
            tape.backward(tr.loss);
            for l in 1..=3 {
                let input = tape.value(tr.get(Stage::Input, l).unwrap()).to_vec();
                let sum: Vec<f64> = input.iter().zip(branch(&m, l, &input)).map(|(a, b)| a + b).collect();
                let (v, u) = (random(input.len()), random(input.len()));
                let mut roads = Roads { tape: &mut tape, tr: &tr, model: &m, post, s: m.mlp_spec().unwrap().residual_scale };
                let back = roads.through(l, &v);
                let road = roads.road(l, &v);
                let h = 1e-6;
                let along = |g: &dyn Fn(&[f64]) -> Vec<f64>, at: &[f64]| {
                    let (a, b): (Vec<f64>, Vec<f64>) = at.iter().zip(&u).map(|(p, q)| (p + h * q, p - h * q)).unzip();
                    dot(&v, &g(&a).iter().zip(g(&b)).map(|(p, q)| (p - q) / (2.0 * h)).collect::<Vec<_>>())
                };
                let fd = along(&|z: &[f64]| branch(&m, l, z), &input);
                assert!((fd - dot(&back, &u)).abs() < 1e-6 * fd.abs().max(1.0), "{d:?} block {l} through: {} against {fd}", dot(&back, &u));
                // along the stream: unchanged, or back through the normalisation after the sum
                let fd = if post.is_some() { along(&|z: &[f64]| normalise(&m, l, z), &sum) } else { dot(&v, &u) };
                assert!((fd - dot(&road, &u)).abs() < 1e-6 * fd.abs().max(1.0), "{d:?} block {l} road: {} against {fd}", dot(&road, &u));
            }
        }
    }

    /// The normalisation's road back, worked out here, is the tape's sweep.
    #[test]
    fn the_road_back_through_a_normalisation_is_the_tapes() {
        let mut rng = Rng::new(7);
        let mut random = |n: usize, scale: f64| -> Vec<f64> { (0..n).map(|_| scale * rng.normal() as f64 + 0.3).collect() };
        for kind in [NormKind::Layer, NormKind::Rms] {
            let (sum, gain, v) = (random(5 * WIDTH, 2.0), random(WIDTH, 1.0), random(5 * WIDTH, 1.0));
            let mut tape = Tape::new();
            let c = tape.constant(5, WIDTH, &sum);
            let n = if kind == NormKind::Layer { tape.layer_norm(c, arch::EPS) } else { tape.rms_norm(c, arch::EPS) };
            let g = tape.param(1, WIDTH, &gain);
            let out = tape.mul_row(n, g);
            tape.backward_seeded(out, &v);
            let ours = norm_back(kind, &sum, &gain, &v);
            for (a, b) in ours.iter().zip(tape.grad(c)) {
                assert!((a - b).abs() < 1e-12 * (1.0 + b.abs()), "{kind:?}: {a} against the tape's {b}");
            }
        }
        // no normalisation: unchanged
        assert_eq!(norm_back(NormKind::None, &[1.0, 2.0], &[1.0, 1.0], &[3.0, 4.0]), vec![3.0, 4.0]);
    }

    // ----- the paths --------------------------------------------------------------------

    fn sum_of(terms: &[Vec<f64>]) -> Vec<f64> {
        let n = terms.first().map_or(0, |t| t.len());
        (0..n).map(|i| terms.iter().map(|t| t[i]).sum()).collect()
    }

    fn close(a: &[f64], b: &[f64], tol: f64) -> bool {
        let size = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        a.len() == b.len() && a.iter().zip(b).all(|(p, q)| (p - q).abs() <= tol * size.max(1e-300))
    }

    /// The paths' terms add up to the gradient at x_0 exactly: the eight of three blocks,
    /// the paths of ten gathered by length, and the paths of six both ways (listed one by
    /// one and gathered by length); and before a normalisation after the sum, the
    /// straight road's term is the gradient at the top itself.
    #[test]
    fn the_paths_add_up_to_the_gradient() {
        let (x, y) = data();
        let mut tape = Tape::new();
        for d in designs() {
            let m = model(3, true, d);
            let (whole, terms) = paths(&m, &x, &y, &mut tape);
            assert_eq!(terms.len(), 8);
            assert!(close(&sum_of(&terms), &whole, 1e-12), "{d:?}: the eight terms");
            let (_, top) = (0, top_and_whole(&m, &x, &y, &mut tape).unwrap().2);
            assert_eq!(terms[0] == top, d.post().is_none(), "{d:?}: the straight road");
            let (whole, parts) = by_length(&model(10, true, d), &x, &y, &mut tape);
            assert_eq!(parts.len(), 11);
            assert!(close(&sum_of(&parts), &whole, 1e-12), "{d:?}: ten blocks by length");
            let six = model(6, true, d);
            let (_, terms) = paths(&six, &x, &y, &mut tape);
            let (_, parts) = by_length(&six, &x, &y, &mut tape);
            for (k, part) in parts.iter().enumerate() {
                let gathered: Vec<Vec<f64>> = terms.iter().enumerate().filter(|(s, _)| s.count_ones() as usize == k).map(|(_, t)| t.clone()).collect();
                assert!(close(&sum_of(&gathered), part, 1e-11), "{d:?}: six blocks, the paths through {k}");
            }
            // the shares add up to 1
            let a = Analysis::of(d, &x, &y, &mut tape);
            assert!((a.path_share.iter().sum::<f64>() - 1.0).abs() < 1e-12 && (a.len_share.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{d:?}: the shares");
        }
    }

    /// The straight road's part on the chart of every layer: level before a
    /// normalisation after the sum (it is the gradient at the top), falling after one.
    #[test]
    fn the_straight_road_is_level_unless_normalised_after_the_sum() {
        let (x, y) = data();
        let mut tape = Tape::new();
        for d in designs() {
            let deep = Deep::of(20, d, &x, &y, &mut tape);
            let level = deep.road.iter().all(|r| *r == deep.road[20]);
            assert_eq!(level, d.post().is_none(), "{d:?}: {:?}", deep.road);
            assert_eq!(deep.road[20], deep.skips.grad[20]);
        }
    }

    // ----- what the captions say ----------------------------------------------------------

    fn scene(k: usize) -> Residual {
        let mut s = Residual::new();
        s.set_up(k);
        s
    }

    fn trained(mut s: Residual) -> Residual {
        let (x, y) = (s.x.clone(), s.y.clone());
        while !s.run.finished() {
            s.run.step(&x, &y);
        }
        s
    }

    fn within(v: f64, lo: f64, hi: f64) -> bool {
        v >= lo && v < hi
    }

    /// The factor per layer, on average over the layers: (last / first)^(1 / layers).
    fn per_layer(v: &[f64]) -> f64 {
        (v[v.len() - 1] / v[0]).powf(1.0 / (v.len() - 1) as f64)
    }

    #[test]
    fn the_scene_opens_on_the_first_thing_to_try() {
        let s = Residual::new();
        let t = scene(0);
        assert_eq!((s.depth, s.design()), (t.depth, t.design()));
    }

    /// The numbers the captions give, as the scene works them out after each button.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // fifty plain layers: about two thirds a layer, both ways; with skips, 4.2 and 6.8
        let s = scene(0);
        let d = &s.deep;
        let [plain, skips] = d.grad_ratio();
        assert!(within(plain, 0.5e-9, 1.5e-9) && within(skips, 4.15, 4.25), "{plain} {skips}");
        let (down, back) = (per_layer(&d.plain.act), 1.0 / per_layer(&d.plain.grad));
        assert!(within(down, 0.62, 0.70) && within(back, 0.62, 0.70), "{down} {back}");
        assert!(d.road_ratio() == 1.0 && within(d.act_ratio()[1], 6.75, 6.85), "{:?}", d.act_ratio());
        // (the primer:) most of that growth is a shared offset, the ReLUs' outputs never
        // being negative: at the top the stream's mean is most of its size, and about
        // each point's own mean it has grown under twice
        let (x, y) = data();
        let mut tape = Tape::new();
        let tr = model(50, true, s.design()).loss_mlp(&mut tape, &x, &y, Loss::CrossEntropy);
        let spread = |v: &[f64]| (v.chunks(WIDTH).map(|r| r.iter().map(|x| (x - r.iter().sum::<f64>() / WIDTH as f64).powi(2)).sum::<f64>()).sum::<f64>() / v.len() as f64).sqrt();
        let (bottom, top) = (tape.value(stream(&tr, 0).unwrap()).to_vec(), tape.value(stream(&tr, 50).unwrap()).to_vec());
        let mean = top.iter().sum::<f64>() / top.len() as f64;
        assert!(mean * mean > 0.7 * rms(&top).powi(2) && spread(&top) < 2.0 * spread(&bottom), "{mean} {} {} {}", rms(&top), spread(&top), spread(&bottom));
        // three blocks, s = 1: 35%, 57%, 11%, −2%; the path through f₃ half as long as g₀, bringing 10%
        let a = scene(1).analysis().clone();
        let p = a.path_share;
        assert!(within(p[0], 0.345, 0.355) && within(p[1] + p[2] + p[3], 0.565, 0.575) && within(p[4] + p[5] + p[6], 0.105, 0.115) && within(p[7], -0.025, -0.015), "{p:?}");
        assert!(within(a.path_len[3], 0.45, 0.55) && within(p[3], 0.095, 0.105), "{:?}", a.path_len);
        // ten blocks, s = 1/√10: the 56 short paths bring 99.97%: 62%, 28%, 10%;
        // those through three come to 11% in length and bring nothing
        let a = scene(2).analysis().clone();
        let l = &a.len_share;
        let rest: f64 = l[3..].iter().sum();
        assert!(within(rest, 0.00025, 0.00035) && within(l[0], 0.615, 0.625) && within(l[1], 0.275, 0.285) && within(l[2], 0.095, 0.105), "{l:?}");
        assert!(within(a.len_len[3], 0.105, 0.115) && l[3].abs() < 0.005, "{:?}", a.len_len);
        assert_eq!(a.short_paths().0, 2);
        assert_eq!(a.short_paths().2, 56);
        // each block hands back through itself at most 0.28 of the gradient reaching it
        let (x, y) = data();
        let mut tape = Tape::new();
        let m = model(10, true, scene(2).design());
        let tr = m.loss_mlp(&mut tape, &x, &y, Loss::CrossEntropy);
        tape.backward(tr.loss);
        let grads: Vec<Vec<f64>> = (0..=10).map(|l| tape.grad(stream(&tr, l).unwrap()).to_vec()).collect();
        let mut roads = Roads { tape: &mut tape, tr: &tr, model: &m, post: None, s: m.mlp_spec().unwrap().residual_scale };
        let most = (1..=10).map(|l| rms(&roads.through(l, &grads[l])) / rms(&grads[l])).fold(0.0, f64::max);
        assert!(within(most, 0.27, 0.28), "{most}");
        // and with s = 1: 82% by paths through 2 to 5 blocks, the straight road −0.5%
        let a = scene(1).analysis().clone();
        let l = &a.len_share;
        assert!(within(l[2..=5].iter().sum(), 0.815, 0.825) && within(l[0], -0.0055, -0.0045), "{l:?}");
        // not scaled down: 1.42 a block, 4 × 10⁷ up and 2 × 10⁷ back
        let s = scene(3);
        let d = &s.deep;
        let (up, back) = (d.act_ratio()[1], d.grad_ratio()[1]);
        assert!(within(up, 3.5e7, 4.5e7) && within(back, 1.5e7, 2.5e7) && within(per_layer(&d.skips.act), 1.415, 1.425), "{up} {back}");
        assert!(within(per_layer(&d.skips.act).powi(2), 1.95, 2.05) && within(1.0 / per_layer(&d.skips.grad), 1.35, 1.45));
        // and a layer normalisation inside each block holds the stream's growth to 24 times
        let mut s = scene(3);
        s.choose(Choice::Norm(NormKind::Layer));
        assert!(s.design().post().is_none() && within(s.deep.act_ratio()[1], 23.5, 24.5), "{:?}", s.deep.act_ratio());
        // normalised after the sum: the road hands down 1/126, g₀ is 105 times g₅₀,
        // all but a sliver by paths through blocks
        let s = scene(4);
        let d = &s.deep;
        assert!(within(1.0 / d.road_ratio(), 125.5, 126.5) && within(d.grad_ratio()[1], 104.5, 105.5), "{} {:?}", d.road_ratio(), d.grad_ratio());
        assert!(d.road[0] / d.skips.grad[0] < 1e-3);
    }

    /// The training runs the captions describe, step by step as the scene takes them.
    #[test]
    fn the_training_runs_do_what_their_captions_say() {
        let ln3 = 3f64.ln();
        // fifty plain layers stay at ln 3 = 1.099; with skips, from 6.5 to 0.0004
        let s = trained(scene(0));
        let [plain, skips] = &s.run.losses;
        assert!(plain.iter().all(|l| (l - ln3).abs() < 5e-4), "{plain:?}");
        assert!(within(skips[0], 6.45, 6.55) && within(skips[STEPS], 0.00035, 0.00045), "{} {}", skips[0], skips[STEPS]);
        // not scaled down: from 1.6 × 10⁷ to 26
        let s = trained(scene(3));
        let skips = &s.run.losses[1];
        assert!(within(skips[0], 1.55e7, 1.65e7) && within(skips[STEPS], 25.5, 26.5), "{} {}", skips[0], skips[STEPS]);
        // post: wanders and ends at ln 3; pre: 0.001 by step 75
        let s = trained(scene(4));
        let skips = &s.run.losses[1];
        let (low, high) = skips[10..].iter().fold((f64::MAX, f64::MIN), |(a, b), &v| (a.min(v), b.max(v)));
        assert!(high - low > 0.2 && (skips[STEPS] - ln3).abs() < 5e-4, "{skips:?}");
        let mut pre = scene(4);
        pre.choose(Choice::Place(Place::Pre));
        let pre = trained(pre);
        assert!(pre.deep.road_ratio() == 1.0 && pre.run.losses[1][75] < 0.0015, "{}", pre.run.losses[1][75]);
    }

    // ----- the scene as it runs -------------------------------------------------------------

    /// Every depth and design gives finite numbers everywhere, the shares add up to 1, and
    /// the scene comes to rest once its run is over.
    #[test]
    fn every_setting_is_drawable_and_comes_to_rest() {
        let mut s = Residual::new();
        for depth in [1, 2, 50] {
            for (scale, norm, place) in DESIGNS {
                s.choose(Choice::Depth(depth));
                (s.scale, s.norm, s.place) = (scale, norm, place);
                s.refresh();
                let d = &s.deep;
                for v in [&d.plain.act, &d.plain.grad, &d.skips.act, &d.skips.grad, &d.road] {
                    assert!(v.len() == depth + 1 && v.iter().all(|x| x.is_finite() && *x >= 0.0), "{depth} {:?}: {v:?}", s.design());
                }
                let a = s.analysis();
                assert!(a.path_share.iter().chain(&a.len_share).all(|v| v.is_finite()));
            }
        }
        let (x, y) = (s.x.clone(), s.y.clone());
        let mut frames = 0;
        loop {
            if !s.run.finished() {
                s.run.step(&x, &y);
            }
            s.advance(0.0, 1.0 / 60.0);
            frames += 1;
            if !s.animated() {
                break;
            }
            assert!(frames < 2000, "the scene never comes to rest");
        }
        assert!(s.run.losses.iter().all(|l| l.len() == STEPS + 1 && l.iter().all(|v| v.is_finite())));
        assert_eq!(s.shown, STEPS as f64);
    }

    /// A change of setting starts a new run and fades the old one out; the same setting
    /// chosen again changes nothing.
    #[test]
    fn a_new_setting_starts_a_new_run() {
        let mut s = Residual::new();
        let (x, y) = (s.x.clone(), s.y.clone());
        for _ in 0..10 {
            s.run.step(&x, &y);
        }
        s.advance(0.0, 1.0 / 60.0);
        s.choose(Choice::Depth(50));
        assert!(s.old.is_none() && s.run.losses[0].len() == 11);
        s.choose(Choice::Depth(10));
        assert!(s.old.as_ref().is_some_and(|(c, _)| c[0].len() == STEPS + 1) && s.run.losses[0].len() == 1 && s.run.depth == 10);
        // with no normalisation, where it goes changes nothing
        s.old = None;
        s.choose(Choice::Place(Place::Post));
        assert!(s.old.is_none() && s.design().place == Place::Pre);
    }

    #[test]
    #[ignore]
    fn explore_drift() {
        let (x, y) = data();
        let mut tape = Tape::new();
        for d in [Design::new(Scale::RootDepth, NormKind::None, Place::Pre), Design::new(Scale::One, NormKind::None, Place::Pre), Design::new(Scale::One, NormKind::Layer, Place::Pre)] {
            let m = model(50, true, d);
            let tr = m.loss_mlp(&mut tape, &x, &y, Loss::CrossEntropy);
            for l in [0, 1, 10, 25, 50] {
                let v = tape.value(stream(&tr, l).unwrap());
                let mean = v.iter().sum::<f64>() / v.len() as f64;
                // the spread about each point's own mean over its units
                let spread = (v.chunks(WIDTH).map(|r| { let m = r.iter().sum::<f64>() / WIDTH as f64; r.iter().map(|x| (x - m) * (x - m)).sum::<f64>() }).sum::<f64>() / v.len() as f64).sqrt();
                println!("{d:?} layer {l}: rms {:.3e} mean {:.3e} spread about each point's mean {:.3e}", rms(v), mean, spread);
            }
        }
        // how much each block's own road shrinks the gradient that reaches it
        for d in [Design::new(Scale::RootDepth, NormKind::None, Place::Pre), Design::new(Scale::One, NormKind::None, Place::Pre)] {
            let m = model(10, true, d);
            let tr = m.loss_mlp(&mut tape, &x, &y, Loss::CrossEntropy);
            tape.backward(tr.loss);
            let grads: Vec<Vec<f64>> = (0..=10).map(|l| tape.grad(stream(&tr, l).unwrap()).to_vec()).collect();
            let mut roads = Roads { tape: &mut tape, tr: &tr, model: &m, post: None, s: m.mlp_spec().unwrap().residual_scale };
            let ratios: Vec<String> = (1..=10).map(|l| format!("{:.3}", rms(&roads.through(l, &grads[l])) / rms(&grads[l]))).collect();
            println!("{d:?}: |J_l^T g_l| / |g_l| = {}", ratios.join(" "));
        }
        // a one-dimensional check of the binomial example
        let j: f64 = 0.5;
        println!("(1 + j)^3 = {}, terms {:?}", (1.0 + j).powi(3), (0..=3).map(|k| binomial(3, k) as f64 * j.powi(k as i32)).collect::<Vec<_>>());
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let s = scene(k);
            let d = &s.deep;
            let a = s.analysis().clone();
            println!("== {}", TRIALS[k].label);
            println!("  grad ratio {:?} act ratio {:?} road {:.3e}", d.grad_ratio(), d.act_ratio(), d.road_ratio());
            println!("  per layer: plain act {:.3} grad {:.3}; skips act {:.3} grad {:.3}", per_layer(&d.plain.act), per_layer(&d.plain.grad), per_layer(&d.skips.act), per_layer(&d.skips.grad));
            println!("  three: {:?}", a.path_share.iter().zip(&a.path_len).map(|(s, l)| format!("{s:+.4}/{l:.3}")).collect::<Vec<_>>());
            println!("  ten: {:?}", a.len_share.iter().map(|s| format!("{s:+.4}")).collect::<Vec<_>>());
            println!("  ten lengths: {:?}", a.len_len.iter().map(|s| format!("{s:.3}")).collect::<Vec<_>>());
            println!("  short paths {:?}", a.short_paths());
            let s = trained(s);
            for (name, l) in ["plain", "skips"].iter().zip(&s.run.losses) {
                println!("  {name}: {}", [0, 10, 25, 50, 75, 100].iter().map(|&k| format!("{}:{:.5}", k, l[k])).collect::<Vec<_>>().join(" "));
            }
        }
    }
}

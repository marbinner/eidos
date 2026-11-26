//! Gradients in a transformer: the loss at one position of a small decoder-only
//! transformer differentiated, and its gradient followed through the network, place
//! by place of the residual stream and position by position; with the gradient that
//! reaches each part's weights, and what each design choice (pre-norm or post-norm,
//! skip connections, the activation, the starting weights, the depth) does to it all.
//!
//! Everything is drawn in screen space (logical pixels), laid out afresh for the size
//! of the view, as in `interp/transformer.rs`: one column per position of the
//! sequence, one row per place of the stream, the tokens under the grid and the next
//! tokens over it. Every number comes from the library's tape and trace
//! (`math::arch`): one forward and backward pass of the sequence on show, done again
//! only when something it depends on has changed.

use super::*;
use crate::clock;
use crate::math::arch::{repeats, Group, Kind, Model, Norm, Part, Place, Stage, TransformerSpec};
use crate::math::autograd::{Act, Tape, Var, NO_TARGET};
use crate::scenes::stats::plot::{self, Glide, Plot, Rect};

pub const DEF: SceneDef = SceneDef {
    id: "tfgrad",
    title: "Gradients in a transformer",
    group: GROUP,
    blurb: "A small modern transformer, layer by layer and token by token: where the gradient of the loss goes, through which parts, and what pre-norm and skip connections change.",
    make: |_| Box::new(TfGrad::new()),
};

// ----- the network and its task ----------------------------------------------------------

/// Different tokens, and the length of a sequence: a run of six tokens and the same
/// run again (`arch::repeats`), so that the second copy can be predicted from the first.
const VOCAB: usize = 10;
const CONTEXT: usize = 12;
/// The width of the stream, heads per attention layer, hidden units of each MLP.
const WIDTH: usize = 32;
const HEADS: usize = 2;
const HIDDEN: usize = 64;
/// The starting weights, the training sequences and the sequence on show each come
/// from a seed of their own, so that every design is compared at the same ones.
const WEIGHTS_SEED: u64 = 7;
const DATA_SEED: u64 = 11;
const SHOWN_SEED: u64 = 1;
/// Training: Adam at this rate, this many sequences a step, the gradient's length
/// clipped to this, and this many steps a press of "train".
const RATE: f64 = 4e-3;
const BATCH: usize = 8;
const CLIP: f64 = 1.0;
const ROUND: usize = 300;
/// Sequences run forward and back each second while training (as many as the frame's
/// allowance has room for, which is fewer in a browser).
const PASSES_PER_SECOND: f32 = 2400.0;
/// The position whose loss is studied to begin with: in the second copy.
const OPENING: usize = 8;
const LETTERS: [&str; VOCAB] = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];
const DEPTHS: [usize; 3] = [2, 4, 8];
const GAINS: [f64; 3] = [0.5, 1.0, 2.0];
/// The most powers of ten the colours of the grid span, and those of the bars.
const DECADES: f32 = 5.0;
const BAR_DECADES: f32 = 4.0;
/// The colour level a cell whose gradient is exactly 0 glides to (below the scale).
const NOTHING: f32 = -0.25;

/// How the normalisations are placed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Arrange {
    /// Layer normalisation before each attention and MLP, inside the skip connection,
    /// and once more before the read-out.
    Pre,
    /// Layer normalisation after each sum of the stream and what a part adds.
    Post,
    /// None at all.
    Plain,
}

impl Arrange {
    fn name(self) -> &'static str {
        match self {
            Arrange::Pre => "pre-norm",
            Arrange::Post => "post-norm",
            Arrange::Plain => "no normalisation",
        }
    }
}

/// The design switches, as the buttons set them.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Design {
    arrange: Arrange,
    skips: bool,
    relu: bool,
    /// Into `GAINS` and `DEPTHS`.
    gain: usize,
    depth: usize,
}

const DEFAULT: Design = Design { arrange: Arrange::Pre, skips: true, relu: false, gain: 1, depth: 1 };

impl Design {
    fn blocks(&self) -> usize {
        DEPTHS[self.depth.min(DEPTHS.len() - 1)]
    }

    fn gain(&self) -> f64 {
        GAINS[self.gain.min(GAINS.len() - 1)]
    }

    fn act(&self) -> Act {
        if self.relu {
            Act::Relu
        } else {
            Act::Gelu
        }
    }

    /// The places of the stream: after the embeddings, and after each attention and
    /// each MLP.
    fn rows(&self) -> usize {
        2 * self.blocks() + 1
    }

    fn spec(&self) -> TransformerSpec {
        TransformerSpec {
            vocab: VOCAB,
            context: CONTEXT,
            d_model: WIDTH,
            heads: HEADS,
            layers: self.blocks(),
            d_mlp: HIDDEN,
            act: self.act(),
            norm: if self.arrange == Arrange::Plain { Norm::None } else { Norm::Layer },
            place: if self.arrange == Arrange::Post { Place::Post } else { Place::Pre },
            residual: self.skips,
            init_gain: self.gain(),
            tied_unembed: false,
        }
    }

    /// "pre-norm, GELU, weights ×1, 4 blocks" (and "no skips" when they are off).
    fn describe(&self) -> String {
        let gain = ["0.5", "1", "2"][self.gain.min(2)];
        let skips = if self.skips { "" } else { ", no skips" };
        format!(r"{}{skips}, {}, weights $\times {gain}$, {} blocks", self.arrange.name(), self.act().name(), self.blocks())
    }

    /// The same, short, for the chart's legend on a phone.
    fn short(&self) -> String {
        let norm = match self.arrange {
            Arrange::Pre => "pre",
            Arrange::Post => "post",
            Arrange::Plain => "no norm",
        };
        let skips = if self.skips { "" } else { ", no skips" };
        format!("{norm}{skips}, {} blocks", self.blocks())
    }
}

/// A design's network at its starting weights. The library starts the embeddings at a
/// length of about the gain; here they are scaled up so that the stream starts with
/// entries of about the gain, the size of what each block adds to it, as in
/// transformers whose embeddings are multiplied by the root of their width. (With the
/// library's own, the stream's first place is several times smaller than the rest, and
/// the normalisations hand back a gradient several times larger there: see
/// `normgrad`.)
fn build(d: &Design) -> Model {
    let mut model = Model::transformer(&d.spec(), &mut Rng::new(WEIGHTS_SEED));
    let k = (WIDTH as f64 / 2.0).sqrt();
    for g in model.groups_mut().iter_mut().filter(|g| matches!(g.kind, Kind::Embedding | Kind::Position)) {
        g.value.iter_mut().for_each(|v| *v *= k);
    }
    model
}

/// The sequence on show and, at each position, the token that comes next (the last
/// position has none).
fn shown() -> (Vec<usize>, Vec<usize>) {
    repeats(CONTEXT, VOCAB, &mut Rng::new(SHOWN_SEED))
}

/// Whose loss is differentiated: one position's, or the mean over every position that
/// has a next token.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Focus {
    One(usize),
    All,
}

impl Focus {
    fn targets(self, next: &[usize]) -> Vec<usize> {
        match self {
            Focus::All => next.to_vec(),
            Focus::One(t) => (0..next.len()).map(|s| if s == t { next[s] } else { NO_TARGET }).collect(),
        }
    }
}

// ----- what is measured ------------------------------------------------------------------

/// The gradient reaching one part's weights in this pass, and the weights' own size.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct PartSize {
    grad: f64,
    value: f64,
}

impl PartSize {
    /// What a plain gradient-descent step at the training rate would change the
    /// weights by, as a share of their size.
    fn relative_step(&self) -> f64 {
        if self.value > 0.0 {
            RATE * self.grad / self.value
        } else {
            0.0
        }
    }
}

/// The parts with weights, in the order of the bars: the embeddings, then for each
/// block its heads and its MLP, then the read-out. `None` for what is in no bar (the
/// normalisations' gains and biases).
fn part_of(g: &Group, blocks: usize) -> Option<usize> {
    let block = g.layer.max(1) - 1;
    match (g.part, g.kind) {
        (Part::Embedding, _) => Some(0),
        (Part::Attention, _) if (1..=HEADS).contains(&g.head) => Some(1 + 3 * block + g.head - 1),
        (Part::Mlp, Kind::Weight | Kind::Bias) => Some(1 + 3 * block + 2),
        (Part::Top, Kind::Unembedding) => Some(1 + 3 * blocks),
        _ => None,
    }
}

/// One forward and backward pass of the sequence on show, and what the picture draws
/// from it.
#[derive(Clone, Debug)]
struct Measure {
    blocks: usize,
    loss: f64,
    /// The gradient at each place of the stream, row by row (0: after the embeddings;
    /// 2b − 1: after attention b; 2b: after the MLP of block b), `CONTEXT * WIDTH`
    /// numbers each, position by position.
    grads: Vec<Vec<f64>>,
    /// For each place below the top, the skip road's share of its gradient: what the
    /// place above hands straight back (pre-norm, or no normalisation), or through the
    /// normalisation after the sum (post-norm); zeros without skip connections. The
    /// rest of the gradient there came through the part in between.
    skips: Vec<Vec<f64>>,
    /// The lengths of the gradients, row k and position s at `k * CONTEXT + s`.
    sizes: Vec<f64>,
    /// The embeddings, each block's two heads and MLP, the read-out.
    parts: Vec<PartSize>,
    /// The chance the network gives the right next token at each position (NaN for the
    /// last).
    chance: Vec<f64>,
    /// The gradient's length at each row: in the column of the position whose loss it
    /// is, or over the whole row for the loss of every position.
    profile: Vec<f64>,
}

fn length(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

impl Measure {
    fn rows(&self) -> usize {
        2 * self.blocks + 1
    }

    /// The colour scales of the cells and of the bars.
    fn cell_scale(&self) -> Scale {
        Scale::of(self.sizes.iter().copied(), DECADES)
    }

    fn bar_scale(&self) -> Scale {
        Scale::of(self.parts.iter().map(|p| p.grad), BAR_DECADES)
    }

    #[cfg(test)]
    fn size(&self, k: usize, s: usize) -> f64 {
        self.sizes.get(k * CONTEXT + s).copied().unwrap_or(0.0)
    }

    /// The gradient at row k, position s (its `WIDTH` numbers).
    fn grad(&self, k: usize, s: usize) -> &[f64] {
        self.grads.get(k).and_then(|g| g.get(s * WIDTH..(s + 1) * WIDTH)).unwrap_or(&[])
    }

    fn skip(&self, k: usize, s: usize) -> &[f64] {
        self.skips.get(k).and_then(|g| g.get(s * WIDTH..(s + 1) * WIDTH)).unwrap_or(&[])
    }
}

/// The stage of the trace that is row k of the stream.
fn place(k: usize) -> (Stage, usize) {
    if k == 0 {
        (Stage::Stream, 0)
    } else if k % 2 == 1 {
        (Stage::Mid, k.div_ceil(2))
    } else {
        (Stage::Stream, k / 2)
    }
}

fn measure(model: &Model, tape: &mut Tape, tokens: &[usize], next: &[usize], focus: Focus) -> Measure {
    let spec = model.transformer_spec().copied().unwrap_or_else(|| DEFAULT.spec());
    let blocks = spec.layers;
    let rows = 2 * blocks + 1;
    let post = spec.place == Place::Post && spec.norm != Norm::None;
    let targets = focus.targets(next);
    let tr = model.loss_transformer(tape, tokens, &targets);
    tape.backward(tr.loss);
    let copy = |v: Option<Var>| {
        let mut out = vec![0.0; CONTEXT * WIDTH];
        if let Some(v) = v {
            for (o, g) in out.iter_mut().zip(tape.grad(v)) {
                *o = *g;
            }
        }
        out
    };
    let grads: Vec<Vec<f64>> = (0..rows).map(|k| copy(tr.get(place(k).0, place(k).1))).collect();
    let skips: Vec<Vec<f64>> = (0..rows - 1)
        .map(|k| {
            if !spec.residual {
                vec![0.0; CONTEXT * WIDTH]
            } else if post {
                // (the sum before the normalisation above this place)
                let stage = if k % 2 == 0 { Stage::AttnSum } else { Stage::MlpSum };
                copy(tr.get(stage, k / 2 + 1))
            } else {
                grads[k + 1].clone()
            }
        })
        .collect();
    let sizes: Vec<f64> = (0..rows).flat_map(|k| (0..CONTEXT).map(move |s| (k, s))).map(|(k, s)| length(&grads[k][s * WIDTH..(s + 1) * WIDTH])).collect();
    let mut sums = vec![(0.0, 0.0); 3 * blocks + 2];
    for (g, &v) in model.groups().iter().zip(&tr.params) {
        if let Some(sum) = part_of(g, blocks).and_then(|i| sums.get_mut(i)) {
            sum.0 += tape.grad(v).iter().map(|x| x * x).sum::<f64>();
            sum.1 += g.value.iter().map(|x| x * x).sum::<f64>();
        }
    }
    let parts = sums.iter().map(|&(g, v): &(f64, f64)| PartSize { grad: g.sqrt(), value: v.sqrt() }).collect();
    let logits = tape.value(tr.output);
    let chance = (0..CONTEXT)
        .map(|s| match (next.get(s), logits.get(s * VOCAB..(s + 1) * VOCAB)) {
            (Some(&want), Some(row)) if want < VOCAB => {
                let top = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let total: f64 = row.iter().map(|x| (x - top).exp()).sum();
                (row[want] - top).exp() / total
            }
            _ => f64::NAN,
        })
        .collect();
    let profile = (0..rows)
        .map(|k| match focus {
            Focus::One(t) => sizes[k * CONTEXT + t.min(CONTEXT - 1)],
            Focus::All => (0..CONTEXT).map(|s| sizes[k * CONTEXT + s].powi(2)).sum::<f64>().sqrt(),
        })
        .collect();
    Measure { blocks, loss: tape.value(tr.loss).first().copied().unwrap_or(f64::NAN), grads, skips, sizes, parts, chance, profile }
}

/// A logarithmic scale for sizes: the powers of ten at its two ends.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Scale {
    lo: f32,
    hi: f32,
}

impl Scale {
    /// From the power of ten at or above the largest size down to the one at or below
    /// the smallest above 0, but at least one power and at most `most` of them. (No
    /// size above 0: 0.01 to 1.)
    fn of(sizes: impl Iterator<Item = f64> + Clone, most: f32) -> Scale {
        let positive = sizes.filter(|v| v.is_finite() && *v > 0.0);
        let largest = positive.clone().fold(0.0f64, f64::max);
        let smallest = positive.fold(f64::INFINITY, f64::min);
        if largest <= 0.0 {
            return Scale { lo: -2.0, hi: 0.0 };
        }
        let hi = (largest.log10() - 1e-9).ceil() as f32;
        let lo = ((smallest.log10() + 1e-9).floor() as f32).max(hi - most).min(hi - 1.0);
        Scale { lo, hi }
    }

    /// Where a size is on the scale: 0 at the bottom, 1 at the top (and kept within);
    /// `NOTHING` for exactly 0 (or no number at all).
    fn level(&self, v: f64) -> f32 {
        if v == 0.0 || v.is_nan() {
            return NOTHING;
        }
        if v.is_infinite() {
            return 1.0;
        }
        ((v.log10() as f32 - self.lo) / (self.hi - self.lo).max(1e-6)).max(0.0).min(1.0)
    }
}

// ----- the scene ---------------------------------------------------------------------------

/// A design kept for comparison, with its network as it was when kept.
struct Kept {
    design: Design,
    model: Model,
    steps: usize,
    now: Option<Measure>,
}

/// The things to try: what each sets up (see `TfGrad::set_up`).
struct Trial {
    label: &'static str,
    look_for: &'static str,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "where one loss's gradient goes",
        look_for: "The loss at position 9, whose next token is $\\mathrm{E}$. At the top only position 9 has a gradient. Going down, each time it crosses an attention layer (from a row \"after attention\" to the row under it) part of it goes to every earlier position; crossing an MLP it stays in its columns. Positions 10 to 12 stay empty, exactly 0, in every row: attention only looks back, so a loss cannot reach a later position. Down position 9's own column, with the skip road, the gradient's length stays between 0.50 and 0.67 from the top to the embeddings (the chart); \"take the skips away\" shows it without.",
    },
    Trial {
        label: "only attention carries it sideways",
        look_for: "The ringed cell is the stream after MLP 3 at position 4. Above it, after attention 4, every position but 9 is exactly 0: the MLP of block 4 works on each position by itself, and the skip road stays in its column. So this cell's skip road brings nothing (green), and all of its gradient, of length 0.016, came through attention 4 (blue). Point at the cell under it: there the skip road brings a part of length 0.016, and MLP 3 adds one of 0.004 in the same column.",
    },
    Trial {
        label: "pre-norm against post-norm, eight blocks deep",
        look_for: "Eight blocks from the same starting weights: post-norm now, and pre-norm kept for comparison (dashed). In pre-norm the skip road hands the gradient down unchanged and the parts add to it: at position 9 its length stays between 0.34 and 0.56 from the top to the embeddings. In post-norm each step down the road passes through a normalisation, which divides it by the spread of the sum it normalised (1.0 to 1.5 here, at position 9): at position 9 it falls from 1.11 at the top to 0.21 at the embeddings, where the first position's, 2.9, is the largest of the row.",
    },
    Trial {
        label: "take the skips away",
        look_for: "Four pre-norm blocks from the same starting weights, without skip connections now, and with them kept for comparison (dashed). With no skip road everything passes through the parts, and each attention layer shares the gradient out among the earlier positions: at position 9 its length falls from 1.13 at the top to 0.0056 at the embeddings, 200 times smaller, and there positions 1 and 2 get the most (0.26 and 0.20).",
    },
    Trial {
        label: "after training: what still gets gradient",
        look_for: "300 steps of Adam, each on 8 random sequences of the task (this takes some seconds; the untrained network is kept for comparison). It then gives the right next token a chance of 0.96 or more at positions 6 to 11, and the loss of the whole sequence falls from 3.12 to 1.08. That loss is differentiated here, and its gradient now comes almost all from positions 1 to 5, whose next tokens are the rest of the first copy, new random ones, which cannot be predicted: 0.012 to 0.014 at the top there, against 0.00004 to 0.0008 at positions 6 to 11. On the chart it is 5 to 8 times smaller all the way down than before training.",
    },
];

/// The cell the second thing to try takes apart: after block 3 (row 6), at position 4.
const SIDEWAYS_CELL: (usize, usize) = (6, 3);

pub struct TfGrad {
    design: Design,
    model: Model,
    /// Training steps taken, the step training stops at, whether it is running, the
    /// sequences it draws, and how many of the present step's sequences are done.
    steps: usize,
    goal: usize,
    training: bool,
    data: Rng,
    in_batch: usize,
    pace: clock::Pace,
    kept: Option<Kept>,
    focus: Focus,
    /// The cell whose gradient is taken apart (row, position), and the one under the
    /// pointer, which is shown instead while there is one.
    picked: Option<(usize, usize)>,
    hover: Option<(usize, usize)>,
    tokens: Vec<usize>,
    next: Vec<usize>,
    tape: Tape,
    now: Option<Measure>,
    /// What is shown must be measured again (the network or the loss has changed).
    stale: bool,
    rest: clock::Rest,
    /// The colour levels of the cells and the bars, the heights of the chart's two
    /// curves and its map from powers of ten to heights, as drawn: they glide to what
    /// was measured (see `wanted`).
    cells: Glide,
    bars: Glide,
    curve: Glide,
    curve_b: Glide,
    range: Glide,
}

impl TfGrad {
    fn new() -> TfGrad {
        let (tokens, next) = shown();
        let mut scene = TfGrad {
            design: DEFAULT,
            model: build(&DEFAULT),
            steps: 0,
            goal: 0,
            training: false,
            data: Rng::new(DATA_SEED),
            in_batch: 0,
            pace: clock::Pace::default(),
            kept: None,
            focus: Focus::One(OPENING),
            picked: None,
            hover: None,
            tokens,
            next,
            tape: Tape::new(),
            now: None,
            stale: true,
            rest: clock::Rest::default(),
            cells: Glide::default(),
            bars: Glide::default(),
            curve: Glide::default(),
            curve_b: Glide::default(),
            range: Glide::default(),
        };
        scene.remeasure();
        scene.snap();
        scene
    }

    /// A design from its starting weights, untrained.
    fn start(&mut self, design: Design) {
        self.design = design;
        self.again();
    }

    /// Back to the starting weights of the present design.
    fn again(&mut self) {
        self.model = build(&self.design);
        self.steps = 0;
        self.goal = 0;
        self.training = false;
        self.data = Rng::new(DATA_SEED);
        self.in_batch = 0;
        self.pace.reset();
        if self.picked.is_some_and(|(k, _)| k >= self.design.rows()) {
            self.picked = None;
        }
        self.stale = true;
    }

    /// Change one design switch: a network of the new design from its start.
    fn choose(&mut self, c: Choice) {
        let mut d = self.design;
        match c {
            Choice::Arrange(a) => d.arrange = a,
            Choice::Skips(on) => d.skips = on,
            Choice::Relu(on) => d.relu = on,
            Choice::Gain(i) => d.gain = i,
            Choice::Depth(i) => d.depth = i,
        }
        if d != self.design {
            self.start(d);
        }
    }

    /// The present network becomes B, for comparison.
    fn keep(&mut self) {
        self.kept = Some(Kept { design: self.design, model: self.model.clone(), steps: self.steps, now: None });
        self.stale = true;
    }

    fn set_focus(&mut self, focus: Focus) {
        if focus != self.focus {
            self.focus = focus;
            if let Some(b) = &mut self.kept {
                b.now = None;
            }
            self.stale = true;
        }
    }

    /// Train for another round of steps, or pause.
    fn toggle_training(&mut self) {
        if self.training {
            self.training = false;
        } else {
            self.goal = self.steps + ROUND;
            self.training = true;
            self.pace.reset();
        }
    }

    /// One sequence forward and back, its gradient added to the step's; after the
    /// batch's last, the step itself.
    fn pass(&mut self) {
        let (tokens, targets) = repeats(CONTEXT, VOCAB, &mut self.data);
        let tr = self.model.loss_transformer(&mut self.tape, &tokens, &targets);
        self.tape.backward(tr.loss);
        self.model.accumulate(&self.tape, &tr);
        self.in_batch += 1;
        if self.in_batch >= BATCH {
            self.model.clip(CLIP);
            self.model.step_adam(RATE);
            self.model.zero_grads();
            self.in_batch = 0;
            self.steps += 1;
            self.stale = true;
            if self.steps >= self.goal {
                self.training = false;
            }
        }
    }

    /// Train to the goal at once (for the tests).
    #[cfg(test)]
    fn finish_training(&mut self) {
        while self.training {
            self.pass();
        }
        self.remeasure();
    }

    fn remeasure(&mut self) {
        self.now = Some(measure(&self.model, &mut self.tape, &self.tokens, &self.next, self.focus));
        if let Some(b) = &mut self.kept {
            if b.now.is_none() {
                b.now = Some(measure(&b.model, &mut self.tape, &self.tokens, &self.next, self.focus));
            }
        }
        self.stale = false;
    }

    /// What the glides are heading for: the cells' and bars' colour levels; the two
    /// curves, as heights in the chart (0 at its foot, 1 at its top); and the map from
    /// powers of ten to heights, (a, c) for a k + c. (Heights, not values, glide, so that
    /// every point of the chart moves along a straight line at the same pace, the curves
    /// and the ticks alike, while the range changes under them.)
    fn wanted(&self) -> [Vec<f32>; 5] {
        let Some(m) = &self.now else { return Default::default() };
        let scale = m.cell_scale();
        let cells = m.sizes.iter().map(|&v| scale.level(v)).collect();
        let scale = m.bar_scale();
        let bars = m.parts.iter().map(|p| scale.level(p.grad)).collect();
        let log = |v: &f64| if *v > 0.0 && v.is_finite() { v.log10() as f32 } else { f32::NAN };
        let curve: Vec<f32> = m.profile.iter().map(log).collect();
        let curve_b: Vec<f32> = self.kept.as_ref().and_then(|b| b.now.as_ref()).map_or(Vec::new(), |b| b.profile.iter().map(log).collect());
        let finite = curve.iter().chain(&curve_b).copied().filter(|v| v.is_finite());
        let (lo, hi) = finite.fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
        let (lo, hi) = if lo <= hi { ((lo - 0.15).floor(), (hi + 0.15).ceil().max((lo - 0.15).floor() + 1.0)) } else { (-2.0, 1.0) };
        let (a, c) = (1.0 / (hi - lo), -lo / (hi - lo));
        // (a curve's missing points sit at the foot)
        let height = |c2: Vec<f32>| c2.into_iter().map(|v| if v.is_finite() { (a * v + c).max(0.0).min(1.0) } else { 0.0 }).collect();
        [cells, bars, height(curve), height(curve_b), vec![a, c]]
    }

    fn snap(&mut self) {
        let [cells, bars, curve, curve_b, range] = self.wanted();
        self.cells.snap(&cells);
        self.bars.snap(&bars);
        self.curve.snap(&curve);
        self.curve_b.snap(&curve_b);
        self.range.snap(&range);
    }

    /// Everything that moves: training (scene time), measuring, the glides.
    fn advance(&mut self, dt: f32, real_dt: f32) {
        if self.training && dt > 0.0 {
            let mut turn = self.pace.begin(PASSES_PER_SECOND, dt, real_dt, 0.008);
            while self.training && turn.next() {
                self.pass();
            }
            self.pace.end(turn);
        }
        if self.stale && (!self.training || self.rest.over()) {
            let start = clock::now();
            self.remeasure();
            self.rest = clock::Rest::after(clock::now() - start);
        }
        let [cells, bars, curve, curve_b, range] = self.wanted();
        // (while training, what is measured changes in every step, and not evenly: the
        // picture follows it more slowly, which smooths the steps out)
        let dt = real_dt.max(0.0) * if self.training { 0.4 } else { 1.0 };
        self.cells.follow(&cells, dt, false);
        self.bars.follow(&bars, dt, false);
        self.curve.follow(&curve, dt, false);
        self.curve_b.follow(&curve_b, dt, false);
        self.range.follow(&range, dt, false);
    }

    fn moving(&self) -> bool {
        let [cells, bars, curve, curve_b, range] = self.wanted();
        self.training || self.stale || self.cells.shown != cells || self.bars.shown != bars || self.curve.shown != curve || self.curve_b.shown != curve_b || self.range.shown != range
    }

    /// Set the `i`th thing to try up.
    fn set_up(&mut self, i: usize) {
        self.kept = None;
        self.picked = None;
        self.hover = None;
        self.focus = Focus::One(OPENING);
        match i {
            1 => {
                self.start(DEFAULT);
                self.picked = Some(SIDEWAYS_CELL);
            }
            2 => {
                self.start(Design { depth: 2, ..DEFAULT });
                self.keep();
                self.start(Design { arrange: Arrange::Post, depth: 2, ..DEFAULT });
            }
            3 => {
                self.start(DEFAULT);
                self.keep();
                self.start(Design { skips: false, ..DEFAULT });
            }
            4 => {
                self.start(DEFAULT);
                self.keep();
                self.focus = Focus::All;
                self.toggle_training();
            }
            _ => self.start(DEFAULT),
        }
        self.stale = true;
    }

    /// The cell whose gradient the panel on the picture takes apart.
    fn studied(&self) -> Option<(usize, usize)> {
        self.hover.or(self.picked).filter(|&(k, s)| k < self.design.rows() && s < CONTEXT)
    }

    fn status(&self) -> String {
        if self.training {
            format!("training: step {} of {}", self.steps, self.goal)
        } else if self.steps == 0 {
            "untrained".to_string()
        } else {
            format!("trained for {} steps", self.steps)
        }
    }
}

// ----- where things go ---------------------------------------------------------------------

/// A design switch on the picture.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Choice {
    Arrange(Arrange),
    Skips(bool),
    Relu(bool),
    Gain(usize),
    Depth(usize),
}

/// The switches in groups, each with the words before it (wide, and on a phone).
type Group3 = (&'static str, &'static str, &'static [(&'static str, Choice)]);
const GROUPS: [Group3; 5] = [
    ("normalisation:", "norm:", &[("pre", Choice::Arrange(Arrange::Pre)), ("post", Choice::Arrange(Arrange::Post)), ("none", Choice::Arrange(Arrange::Plain))]),
    ("skips:", "skips:", &[("on", Choice::Skips(true)), ("off", Choice::Skips(false))]),
    ("activation:", "act:", &[("GELU", Choice::Relu(false)), ("ReLU", Choice::Relu(true))]),
    ("starting weights:", "weights:", &[(r"$\times 0.5$", Choice::Gain(0)), (r"$\times 1$", Choice::Gain(1)), (r"$\times 2$", Choice::Gain(2))]),
    ("blocks:", "blocks:", &[("2", Choice::Depth(0)), ("4", Choice::Depth(1)), ("8", Choice::Depth(2))]),
];

impl Choice {
    fn on(self, d: &Design) -> bool {
        match self {
            Choice::Arrange(a) => d.arrange == a,
            Choice::Skips(on) => d.skips == on,
            Choice::Relu(on) => d.relu == on,
            Choice::Gain(i) => d.gain == i,
            Choice::Depth(i) => d.depth == i,
        }
    }
}

const CONTROL_WORDS: [&str; 4] = ["▶ train 300 steps", "↺ start again", "keep for comparison", "all positions"];

/// Kinds of line with a batch each (see `plot::apart`).
const OUTLINES: u8 = plot::OWN;
const CURVE_B: u8 = plot::OWN + 1;
const PICKED: u8 = plot::OWN + 2;
/// The chart's powers of ten, whose number changes as its range glides.
const DECADE_TICKS: u8 = plot::OWN + 3;

fn in_order(f: &mut Frame) {
    for kind in [plot::AXES, plot::CURVES, plot::MARKS, plot::BUTTONS, OUTLINES, PICKED, DECADE_TICKS] {
        plot::apart(f, kind, false);
    }
    plot::apart(f, CURVE_B, true);
}

#[derive(Clone, Debug)]
struct Layout {
    narrow: bool,
    text: f32,
    small: f32,
    controls: Vec<Rect>,
    /// Where the words saying how far training has gone go, if there is room.
    status: Option<Vec2>,
    switches: Vec<(Rect, &'static str, Choice)>,
    leads: Vec<(Vec2, &'static str)>,
    heading: Vec2,
    /// The right end of the row labels, the left edge of the columns, their width.
    labels_x: f32,
    x0: f32,
    col: f32,
    /// The row of next tokens over the grid.
    next_y: f32,
    next_h: f32,
    /// The grid: its top, the height of a row, the room between blocks, the rows.
    top: f32,
    row: f32,
    gap: f32,
    rows: usize,
    tokens_y: f32,
    tile: f32,
    key_y: f32,
    /// The bars of the parts beside the grid: from, to.
    parts: Option<(f32, f32)>,
    chart: Option<Rect>,
    split: Option<Rect>,
}

/// The room a chart needs round its rectangle: left for the tick numbers, above for the
/// title and the legend, below for the axis.
const CHART_LEFT: f32 = 38.0;
const CHART_ABOVE: f32 = 58.0;
const CHART_BELOW: f32 = 34.0;

impl Layout {
    fn of(f: &Frame, view: Vec2, inset: f32, blocks: usize) -> Layout {
        let view = view.max(Vec2::ONE);
        let narrow = view.x < 620.0;
        let (text, small) = if narrow { (11.0, 10.0) } else { (12.5, 11.5) };
        let pad = if narrow { 8.0 } else { 16.0 };
        let (h, gap) = if narrow { (24.0, 4.0) } else { (26.0, 6.0) };
        let mut y = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        // the controls, then the switches, flowing onto further rows as they need
        let width = |w: &str| f.text_size(w, text).x + if narrow { 12.0 } else { 18.0 };
        let mut x = pad;
        let mut controls = Vec::new();
        for (i, w) in CONTROL_WORDS.iter().enumerate() {
            let w = if i == 0 { width(w).max(width("❚❚ pause")) } else { width(w) };
            if x + w > view.x - pad && x > pad {
                x = pad;
                y += h + gap;
            }
            controls.push((Vec2::new(x, y), Vec2::new(x + w, y + h)));
            x += w + gap;
        }
        let status = (view.x - pad - x >= f.text_size("training: step 300 of 300", small).x + 8.0).then(|| Vec2::new(x + 6.0, y + h * 0.5));
        x = pad;
        y += h + gap;
        let (mut switches, mut leads) = (Vec::new(), Vec::new());
        for (wide_lead, short_lead, options) in GROUPS {
            let lead = if narrow { short_lead } else { wide_lead };
            let lead_w = f.text_size(lead, small).x + 5.0;
            let group_w = lead_w + options.iter().map(|(w, _)| width(w) + 3.0).sum::<f32>();
            if x + group_w > view.x - pad && x > pad {
                x = pad;
                y += h + gap;
            }
            leads.push((Vec2::new(x, y + h * 0.5), lead));
            x += lead_w;
            for &(w, c) in options {
                let bw = width(w);
                switches.push(((Vec2::new(x, y), Vec2::new(x + bw, y + h)), w, c));
                x += bw + 3.0;
            }
            x += if narrow { 8.0 } else { 16.0 };
        }
        y += h + if narrow { 8.0 } else { 12.0 };
        let heading = Vec2::new(pad, y);
        let rows = 2 * blocks + 1;
        let next_y = y + small + 10.0;
        let next_h = if narrow { 17.0 } else { 22.0 };
        let top = next_y + next_h + 4.0;
        let tile = if narrow { 17.0 } else { 22.0 };
        let block_gap = if narrow { 2.0 } else { 3.0 };
        // under the grid: the tokens, their numbers, the colour key
        let under = 4.0 + tile + 14.0 + 22.0;
        let labels_w = if narrow { 44.0 } else { 112.0 };
        let parts_w = if narrow { 62.0 } else { 190.0 };
        let right_w = 360.0;
        let x0 = pad + labels_w;
        let side = view.x - (x0 + 12.0 * 30.0 + 14.0 + parts_w + 28.0 + pad) >= right_w;
        let col = ((view.x - x0 - 14.0 - parts_w - pad - if side { right_w + 28.0 } else { 0.0 }) / CONTEXT as f32).min(50.0).floor();
        // below the grid, if not beside it: the chart and the split, side by side or one
        // under the other
        let pair = !side && view.x >= 640.0;
        let chart_h = if narrow { 138.0 } else { 170.0 };
        let split_h = if narrow { 126.0 } else { 140.0 };
        let below = |chart: bool, split: bool| -> f32 {
            if side {
                return 0.0;
            }
            let c = if chart { chart_h + CHART_ABOVE + CHART_BELOW } else { 0.0 };
            let s = if split { split_h + 10.0 } else { 0.0 };
            if pair {
                c.max(s)
            } else {
                c + s
            }
        };
        let room = |chart: bool, split: bool| (view.y - top - under - pad - below(chart, split) - block_gap * blocks as f32) / rows as f32;
        let (mut chart_on, mut split_on) = (true, true);
        if room(true, true) < 9.0 {
            split_on = false;
            if room(true, false) < 9.0 {
                chart_on = false;
            }
        }
        let row = room(chart_on, split_on).min(if narrow { 22.0 } else { 30.0 }).floor();
        // (what the grid leaves, the chart may have)
        let spare = (room(chart_on, split_on) - row).max(0.0) * rows as f32;
        let chart_h = if pair { (chart_h + spare).min(300.0) } else { chart_h + (spare * 0.6).min(120.0) };
        let split_h = if pair { split_h.max(chart_h + CHART_ABOVE + CHART_BELOW - 10.0).min(220.0) } else { split_h };
        let none = Layout {
            narrow,
            text,
            small,
            controls,
            status,
            switches,
            leads,
            heading,
            labels_x: x0 - 6.0,
            x0,
            col: 0.0,
            next_y,
            next_h,
            top,
            row: 0.0,
            gap: block_gap,
            rows,
            tokens_y: top,
            tile,
            key_y: top,
            parts: None,
            chart: None,
            split: None,
        };
        if col < 8.0 || row < 5.0 {
            return none;
        }
        let grid_h = row * rows as f32 + block_gap * blocks as f32;
        let tokens_y = top + grid_h + 4.0;
        let key_y = tokens_y + tile + 16.0;
        let grid_right = x0 + col * CONTEXT as f32;
        let parts = Some((grid_right + 14.0, grid_right + 14.0 + parts_w));
        let (mut chart, mut split) = (None, None);
        if side {
            let x = grid_right + 14.0 + parts_w + 28.0;
            let chart_top = heading.y + CHART_ABOVE;
            let bottom = (key_y + 20.0).max(chart_top + 330.0).min(view.y - pad);
            let ch = ((bottom - chart_top) * 0.5).clamp(120.0, 260.0);
            chart = Some((Vec2::new(x + CHART_LEFT, chart_top), Vec2::new(view.x - pad, chart_top + ch)));
            let split_top = chart_top + ch + CHART_BELOW + 12.0;
            if view.y - pad - split_top >= 110.0 {
                split = Some((Vec2::new(x, split_top), Vec2::new(view.x - pad, (split_top + split_h + 20.0).min(view.y - pad))));
            }
        } else {
            let y = key_y + 26.0;
            if pair {
                let half = ((view.x - 2.0 * pad - 20.0) * 0.5).floor();
                if chart_on {
                    chart = Some((Vec2::new(pad + CHART_LEFT, y + CHART_ABOVE), Vec2::new(pad + half, y + CHART_ABOVE + chart_h)));
                }
                if split_on {
                    split = Some((Vec2::new(pad + half + 20.0, y), Vec2::new(view.x - pad, y + split_h + 10.0)));
                }
            } else {
                let mut y = y;
                if chart_on {
                    chart = Some((Vec2::new(pad + CHART_LEFT, y + CHART_ABOVE), Vec2::new(view.x - pad, y + CHART_ABOVE + chart_h)));
                    y += CHART_ABOVE + chart_h + CHART_BELOW;
                }
                if split_on {
                    split = Some((Vec2::new(pad, y), Vec2::new(view.x - pad, y + split_h)));
                }
            }
        }
        Layout { col, row, tokens_y, key_y, parts, chart, split, ..none }
    }

    fn centre(&self, s: usize) -> f32 {
        self.x0 + (s as f32 + 0.5) * self.col
    }

    /// The rectangle of the cell at row k (0 at the bottom), position s.
    fn cell(&self, k: usize, s: usize) -> Rect {
        let blocks_below = k.div_ceil(2) as f32;
        let bottom = self.top + self.row * self.rows as f32 + self.gap * ((self.rows - 1) / 2) as f32;
        let y1 = bottom - self.row * k as f32 - self.gap * blocks_below;
        let x = self.x0 + s as f32 * self.col;
        (Vec2::new(x + 1.0, y1 - self.row + 1.0), Vec2::new(x + self.col - 1.0, y1))
    }

    /// The cell under a point, if any.
    fn cell_at(&self, p: Vec2) -> Option<(usize, usize)> {
        if self.col <= 0.0 || self.row <= 0.0 {
            return None;
        }
        let s = ((p.x - self.x0) / self.col).floor();
        if !(0.0..CONTEXT as f32).contains(&s) {
            return None;
        }
        (0..self.rows)
            .find(|&k| {
                let (min, max) = self.cell(k, 0);
                p.y >= min.y - 1.0 && p.y <= max.y + 1.0
            })
            .map(|k| (k, s as usize))
    }

    /// The token (or next-token) column under a point, if it is on either row.
    fn tile_at(&self, p: Vec2) -> Option<usize> {
        let on_row = (p.y >= self.tokens_y && p.y <= self.tokens_y + self.tile) || (p.y >= self.next_y && p.y <= self.next_y + self.next_h);
        let s = ((p.x - self.x0) / self.col.max(1e-3)).floor();
        (on_row && (0.0..CONTEXT as f32).contains(&s)).then_some(s as usize)
    }
}

// ----- colours -----------------------------------------------------------------------------

/// A cell with no gradient at all.
const EMPTY: Color = Color::hex(0x121725);

/// The colour of a cell at colour level `u`: from a dark wine at the bottom of the
/// scale through the course's pink to a pale pink at its top, the brightness rising
/// all the way; below the scale (a gradient of exactly 0), the empty cell.
fn heat(u: f32) -> Color {
    const STOPS: [(f32, Color); 4] = [(0.0, Color::hex(0x2B0F26)), (0.4, Color::hex(0x8A1E55)), (0.75, BACKWARD), (1.0, Color::hex(0xFFD6E6))];
    if u < 0.0 {
        return EMPTY.mix(STOPS[0].1, ((u - NOTHING) / -NOTHING).max(0.0).min(1.0));
    }
    let u = u.min(1.0);
    for pair in STOPS.windows(2) {
        let ((a, ca), (b, cb)) = (pair[0], pair[1]);
        if u <= b {
            return ca.mix(cb, (u - a) / (b - a));
        }
    }
    STOPS[3].1
}

/// A token's name, as mathematics: a letter, upright (a name, not a variable).
fn token(t: usize) -> String {
    format!(r"$\mathrm{{{}}}$", LETTERS[t.min(VOCAB - 1)])
}

/// A token's tile: a colour of its own, quiet, so that repeats are easy to spot.
fn token_color(token: usize) -> Color {
    Cmap::Turbo.color(0.08 + 0.84 * token as f32 / (VOCAB - 1) as f32).mix(Color::hex(0x0B0E16), 0.55)
}

/// What came through a part (attention or an MLP), in a cell's split.
const THROUGH: Color = pal::BLUE;

/// A power of ten as a label: $10^{-3}$, and 1 and 10 as numbers.
fn decade(k: i32) -> String {
    match k {
        0 => "1".to_string(),
        1 => "10".to_string(),
        _ => format!("$10^{{{k}}}$"),
    }
}

/// A length as the picture writes it: as mathematics, since a small one is written
/// as 1.5e-4.
fn short(v: f64) -> String {
    if !v.is_finite() {
        return "–".to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    if v.abs() >= 0.01 && v.abs() < 1000.0 {
        let decimals = if v.abs() >= 10.0 {
            1
        } else if v.abs() >= 1.0 {
            2
        } else {
            3
        };
        return format!("${}$", num(v, decimals));
    }
    format!("${:.1e}$", v)
}

/// What row k of the stream is, for its label.
fn row_name(k: usize, narrow: bool) -> String {
    match (k, narrow) {
        (0, false) => "embeddings".to_string(),
        (0, true) => "emb.".to_string(),
        (k, false) if k % 2 == 1 => format!("after attention {}", k.div_ceil(2)),
        (k, true) if k % 2 == 1 => format!("att {}", k.div_ceil(2)),
        (k, false) => format!("after MLP {}", k / 2),
        (k, true) => format!("MLP {}", k / 2),
    }
}

// ----- the scene's frame -------------------------------------------------------------------

impl Scene for TfGrad {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.15);
    }

    fn animated(&self) -> bool {
        self.moving()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("click a token: its loss    point at a cell: how its gradient is made up    buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A small transformer of the modern kind reads 12 tokens: a run of six and the same run again, so that the second half can be predicted from the first. Its loss at one position (orange) is differentiated, and the grid shows where the gradient goes: one column for each position, one row for each place in the residual stream, from the embeddings at the bottom to the top. Beside it, the gradient reaching each part's weights; under it, the gradient's length up the stream, and how one cell's gradient is made up.");

        ui::section(ui, "Things to try");
        for (i, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.label, t.look_for) {
                self.set_up(i);
            }
        }

        ui::section(ui, "The buttons over the picture");
        ui::note(ui, "Train 300 steps: Adam at rate 0.004, on batches of 8 random sequences of the task, with the gradient's length clipped to 1 (press again to pause). Start again: the same design from its starting weights.");
        ui::note(ui, "Keep for comparison: the network as it is now is kept, dashed on the chart. Every design starts from the same seed, and all are measured on the same sequence.");
        ui::note(ui, "The switches: layer normalisation before each part (pre), after each sum (post), or none; the skip connections; the MLPs' activation; a factor on every starting weight; the number of blocks. Each change starts the network again, untrained.");
        ui::note(ui, "Click a token (or its next token) to differentiate the loss at its position; \"all positions\" takes the mean loss of the 11 that have a next token.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (BACKWARD, "cells: the length of the gradient of the loss at each place of the stream (log scale, under the grid); an empty cell: exactly 0"),
                (LOSS, "the position whose loss is differentiated, and its next token"),
                (pal::TEXT, "the thin bar under a next token: the chance the network gives it"),
                (WEIGHT, "bars: the length of the gradient reaching each part's weights (log scale); h1, h2: the two heads of an attention layer; the number: its relative step"),
                (BACKWARD.mix(pal::WHITE, 0.2), "chart: the gradient's length up the stream at the loss's position (all positions: over the whole row); dashed: the network kept for comparison"),
                (SKIP, "how a cell's gradient is made up: green, what the skip road hands down from the cell above"),
                (THROUGH, "and blue, what came through the part in between. Each strip is the 32 numbers of a gradient, and the top one is the sum of the two under it"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Residual stream:", "the 32 numbers at each position that every part reads from and adds to: the token's and the position's embeddings (scaled here so that its numbers start at about 1, about the size of what each block adds), plus what each attention layer and each MLP writes. The read-out turns its top into a guess at the next token.");
        ui::term(ui, "Gradient at a cell:", "the derivative of the loss with respect to the stream's 32 numbers there: how the loss would change if they alone were nudged. The cells show its length.");
        ui::term(ui, "Skip connection:", r"a part's input added to its output, $x + f(x)$. On the way back the gradient arriving at $x + f(x)$ reaches $x$ unchanged (the skip road), plus what comes back through $f$.");
        ui::term(ui, "Attention:", "the only part that moves information between positions: each position takes a weighted mix of what earlier positions offer. It looks only back (causal), so a loss reaches only its own and earlier positions.");
        ui::term(ui, "MLP:", "two layers with an activation between them, applied to each position by itself.");
        ui::term(ui, "Layer normalisation:", "rescales each position's 32 numbers to mean 0 and spread 1, then multiplies them by a learnt gain and adds a learnt bias. Pre-norm puts it before each part, inside the skip connection, as most transformers built today do; post-norm after each sum, as the first transformer did.");
        ui::term(ui, "Loss:", "minus the log of the chance the network gives the right next token (cross-entropy).");
        ui::term(ui, "Relative step:", "rate times the gradient's length over the weights' length, at the training rate 0.004: the share of its size one plain gradient-descent step on this loss would change a part's weights by. Practitioners watch it in training. (Adam sizes its steps otherwise.)");

        ui::section(ui, "Numbers");
        if let Some(m) = &self.now {
            let loss = match self.focus {
                Focus::One(t) => format!("loss at position {}", t + 1),
                Focus::All => "mean loss of every position".to_string(),
            };
            ui::readout(ui, &loss, short(m.loss));
            if let Focus::One(t) = self.focus {
                ui::readout(ui, "chance it gives the right next token", short(m.chance.get(t).copied().unwrap_or(f64::NAN)));
            }
            let ends = |m: &Measure| format!("{}, {}", short(m.profile.last().copied().unwrap_or(f64::NAN)), short(m.profile.first().copied().unwrap_or(f64::NAN)));
            ui::readout(ui, "gradient's length at the top, at the embeddings", ends(m));
            if let Some(b) = self.kept.as_ref().and_then(|b| b.now.as_ref()) {
                ui::readout(ui, "the same, kept for comparison", ends(b));
            }
        }
        ui::readout(ui, "training steps", count(self.steps));
        ui::readout(ui, "weights", count(self.model.n_params()));
        ui::see_also(ui, &[("attngrad", "backpropagation through attention"), ("residual", "skip connections"), ("normgrad", "normalisation and the gradient"), ("transformer", "a transformer, opened up")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let covered = view.x < 160.0;
        let lay = Layout::of(f, view, f.top_inset, self.design.blocks());
        if !covered {
            f.handles.capture();
            f.handles.claim_clicks();
            self.buttons(f, &lay);
            self.pointer(f, &lay);
        }
        self.advance(f.dt, f.real_dt);
        if covered {
            return;
        }
        in_order(f);
        self.grid(f, &lay);
        self.parts(f, &lay);
        self.chart(f, &lay);
        self.split(f, &lay);
    }
}

impl TfGrad {
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let size = lay.text;
        if let [train, again, keep, all] = lay.controls[..] {
            let words = if self.training { "❚❚ pause" } else { CONTROL_WORDS[0] };
            if plot::button(f, train, words, size, self.training) {
                self.toggle_training();
            }
            if plot::button(f, again, CONTROL_WORDS[1], size, false) {
                self.again();
            }
            if plot::button(f, keep, CONTROL_WORDS[2], size, false) {
                self.keep();
            }
            if plot::button(f, all, CONTROL_WORDS[3], size, self.focus == Focus::All) {
                self.set_focus(if self.focus == Focus::All { Focus::One(OPENING) } else { Focus::All });
            }
        }
        if let Some(at) = lay.status {
            f.text_px(at, self.status(), lay.small, if self.training { pal::TEXT } else { pal::DIM }, Align2::LEFT_CENTER);
        }
        for &(at, lead) in &lay.leads {
            f.text_px(at, lead, lay.small, pal::DIM, Align2::LEFT_CENTER);
        }
        for &(r, words, c) in &lay.switches {
            if plot::button(f, r, words, size, c.on(&self.design)) {
                self.choose(c);
            }
        }
    }

    /// The pointer over the picture: a cell under it is taken apart, a click on a cell
    /// keeps it so, a click on a token (or the next token over it) chooses its loss.
    fn pointer(&mut self, f: &mut Frame, lay: &Layout) {
        let at = plot::pointer(f);
        self.hover = at.and_then(|p| lay.cell_at(p));
        if !f.ptr.clicked {
            return;
        }
        let Some(p) = at else { return };
        if let Some(c) = lay.cell_at(p) {
            self.picked = if self.picked == Some(c) { None } else { Some(c) };
        } else if let Some(s) = lay.tile_at(p) {
            if s + 1 < CONTEXT {
                self.set_focus(Focus::One(s));
            }
        }
    }

    fn grid(&self, f: &mut Frame, lay: &Layout) {
        let Some(m) = &self.now else { return };
        if lay.col <= 0.0 || m.rows() != lay.rows {
            return;
        }
        let small = lay.small;
        let heading = match (self.focus, lay.narrow) {
            (Focus::One(t), false) => format!("the gradient of the loss at position {} (its next token: {}), at every place of the stream", t + 1, token(self.next[t])),
            (Focus::One(t), true) => format!("gradient of the loss at position {}", t + 1),
            (Focus::All, false) => "the gradient of the loss of every position, at every place of the stream".to_string(),
            (Focus::All, true) => "gradient of the loss of every position".to_string(),
        };
        f.text_px(lay.heading, heading, small + 0.5, pal::TEXT, Align2::LEFT_TOP);
        // the next tokens and the chance given to each; the differentiated ones in the loss's colour
        let scored = |s: usize| s + 1 < CONTEXT && (self.focus == Focus::All || self.focus == Focus::One(s));
        f.text_px(Vec2::new(lay.labels_x, lay.next_y + lay.next_h * 0.5), if lay.narrow { "next" } else { "next token" }, small, pal::DIM, Align2::RIGHT_CENTER);
        for s in 0..CONTEXT {
            let (a, b) = (lay.x0 + s as f32 * lay.col + 1.0, lay.x0 + (s + 1) as f32 * lay.col - 1.0);
            hud_rect(f, Vec2::new(a, lay.next_y), Vec2::new(b, lay.next_y + lay.next_h), EMPTY);
            if s + 1 >= CONTEXT {
                continue;
            }
            let p = m.chance.get(s).copied().unwrap_or(f64::NAN);
            if p.is_finite() {
                let h = 3.0;
                hud_rect_at(f, 2, Vec2::new(a, lay.next_y + lay.next_h - h), Vec2::new(a + (b - a) * p as f32, lay.next_y + lay.next_h), pal::TEXT.a(0.55));
            }
            let ink = if scored(s) { LOSS } else { pal::DIM };
            f.text_px(Vec2::new(lay.centre(s), lay.next_y + lay.next_h * 0.45), token(self.next[s]), (lay.next_h * 0.6).min(13.0), ink, Align2::CENTER_CENTER);
        }
        // the cells
        for k in 0..lay.rows {
            let size = if lay.narrow { 9.5 } else { small - 0.5 };
            if lay.row >= size + 3.0 {
                let (min, max) = lay.cell(k, 0);
                f.text_px(Vec2::new(lay.labels_x, (min.y + max.y) * 0.5), row_name(k, lay.narrow), size, pal::DIM, Align2::RIGHT_CENTER);
            } else if k % 2 == 0 && lay.row >= 5.0 {
                // (rows too low for a label each: one for each block, across its two rows,
                // and the embeddings')
                let (min, max) = lay.cell(k, 0);
                let (y, name) = if k == 0 { ((min.y + max.y) * 0.5, row_name(0, lay.narrow)) } else { (max.y + 0.5, format!("block {}", k / 2)) };
                f.text_px(Vec2::new(lay.labels_x, y), name, size, pal::DIM, Align2::RIGHT_CENTER);
            }
            for s in 0..CONTEXT {
                let (min, max) = lay.cell(k, s);
                hud_rect(f, min, max, heat(self.cells.get(k * CONTEXT + s)));
            }
        }
        // the tokens, and their numbers
        f.text_px(Vec2::new(lay.labels_x, lay.tokens_y + lay.tile * 0.5), "token", small, pal::DIM, Align2::RIGHT_CENTER);
        for s in 0..CONTEXT {
            let (a, b) = (lay.x0 + s as f32 * lay.col + 1.0, lay.x0 + (s + 1) as f32 * lay.col - 1.0);
            let (t, token_name) = (self.tokens[s].min(VOCAB - 1), token(self.tokens[s]));
            hud_rect(f, Vec2::new(a, lay.tokens_y), Vec2::new(b, lay.tokens_y + lay.tile), token_color(t));
            f.text_px(Vec2::new(lay.centre(s), lay.tokens_y + lay.tile * 0.5), token_name, (lay.tile * 0.6).min(13.0), pal::WHITE, Align2::CENTER_CENTER);
            if lay.col >= 14.0 || s % 2 == 0 {
                f.text_px(Vec2::new(lay.centre(s), lay.tokens_y + lay.tile + 2.0), format!("{}", s + 1), 9.5, pal::FAINT.mix(pal::DIM, 0.5), Align2::CENTER_TOP);
            }
        }
        // the loss's column, outlined from the next token down to the token (with every
        // position's loss, only the next tokens)
        for s in (0..CONTEXT).filter(|&s| scored(s)) {
            let x = lay.x0 + s as f32 * lay.col;
            let y0 = lay.next_y - 1.0;
            let y1 = if self.focus == Focus::All { lay.next_y + lay.next_h + 1.0 } else { lay.tokens_y + lay.tile + 1.0 };
            let corners = [Vec2::new(x, y0), Vec2::new(x + lay.col, y0), Vec2::new(x + lay.col, y1), Vec2::new(x, y1), Vec2::new(x, y0)];
            plot::path_of(f, OUTLINES, &corners, 1.2, LOSS.a(0.75));
        }
        // the cell taken apart
        if let Some((k, s)) = self.studied() {
            let (min, max) = lay.cell(k, s);
            let corners = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
            plot::path_of(f, PICKED, &corners, 2.0, pal::WHITE);
        }
        // the colour scale
        let scale = m.cell_scale();
        let (x, y) = (lay.x0, lay.key_y);
        let w = (lay.col * CONTEXT as f32 * 0.45).clamp(60.0, 220.0);
        f.text_px(Vec2::new(lay.labels_x, y + 4.0), if lay.narrow { "length" } else { "its length" }, small - 0.5, pal::DIM, Align2::RIGHT_CENTER);
        let strips = 24;
        for i in 0..strips {
            let (a, b) = (x + w * i as f32 / strips as f32, x + w * (i + 1) as f32 / strips as f32);
            hud_rect(f, Vec2::new(a, y), Vec2::new(b + 0.5, y + 8.0), heat((i as f32 + 0.5) / strips as f32));
        }
        f.text_px(Vec2::new(x, y + 10.0), decade(scale.lo as i32), 9.5, pal::DIM, Align2::LEFT_TOP);
        f.text_px(Vec2::new(x + w, y + 10.0), decade(scale.hi as i32), 9.5, pal::DIM, Align2::RIGHT_TOP);
        hud_rect(f, Vec2::new(x + w + 14.0, y), Vec2::new(x + w + 22.0, y + 8.0), EMPTY);
        f.text_px(Vec2::new(x + w + 26.0, y + 4.0), "exactly 0", 9.5, pal::DIM, Align2::LEFT_CENTER);
    }

    /// The gradient reaching each part's weights, a bar beside the row the part writes.
    fn parts(&self, f: &mut Frame, lay: &Layout) {
        let (Some(m), Some((x0, x1))) = (&self.now, lay.parts) else { return };
        if m.rows() != lay.rows || lay.row <= 0.0 {
            return;
        }
        let tiny = if lay.narrow { 8.5 } else { 9.5 };
        let names = if lay.narrow { 0.0 } else { 26.0 };
        let numbers = if lay.narrow { 0.0 } else { 60.0 };
        let (a, b) = (x0 + names, x1 - numbers);
        let heading = if lay.narrow { "weights" } else { "gradient reaching the weights" };
        f.text_px(Vec2::new(x0, lay.heading.y), heading, lay.small, pal::TEXT, Align2::LEFT_TOP);

        let bar = |f: &mut Frame, i: usize, y0: f32, y1: f32, name: &str| {
            let u = self.bars.get(i);
            hud_rect(f, Vec2::new(a, y0), Vec2::new(b, y1), EMPTY);
            if u > 0.0 {
                hud_rect_at(f, 2, Vec2::new(a, y0), Vec2::new(a + (b - a) * u, y1), WEIGHT.mix(EMPTY, 0.15));
            }
            if names > 0.0 && !name.is_empty() {
                f.text_px(Vec2::new(x0, (y0 + y1) * 0.5), name, tiny, pal::DIM, Align2::LEFT_CENTER);
            }
            if numbers > 0.0 {
                let step = m.parts.get(i).map_or(0.0, |p| p.relative_step());
                f.text_px(Vec2::new(x1, (y0 + y1) * 0.5), short(step), tiny, pal::DIM, Align2::RIGHT_CENTER);
            }
        };
        // the read-out, over the grid
        bar(f, 3 * m.blocks + 1, lay.next_y + 3.0, lay.next_y + lay.next_h - 3.0, "out");
        for k in 0..lay.rows {
            let (min, max) = lay.cell(k, 0);
            let (y0, y1) = (min.y + 1.0, max.y - 1.0);
            if k == 0 {
                bar(f, 0, y0, y1, "emb");
            } else if k % 2 == 1 {
                let i = 1 + 3 * (k / 2);
                let mid = (y0 + y1) * 0.5;
                bar(f, i, y0, mid - 0.5, "h1");
                bar(f, i + 1, mid + 0.5, y1, "h2");
            } else {
                bar(f, 1 + 3 * (k / 2 - 1) + 2, y0, y1, "MLP");
            }
        }
        // the bars' scale
        let scale = m.bar_scale();
        let y = lay.tokens_y + 2.0;
        f.text_px(Vec2::new(a, y), decade(scale.lo as i32), 9.5, pal::DIM, Align2::LEFT_TOP);
        f.text_px(Vec2::new(b, y), decade(scale.hi as i32), 9.5, pal::DIM, Align2::RIGHT_TOP);
        if !lay.narrow {
            f.text_px(Vec2::new((a + b) * 0.5, y), "log scale", 9.5, pal::FAINT.mix(pal::DIM, 0.5), Align2::CENTER_TOP);
            f.text_px(Vec2::new(x1, y + 14.0), "numbers: relative step", 9.5, pal::DIM, Align2::RIGHT_TOP);
        }
    }

    /// The gradient's length up the stream, for this design and the one kept.
    fn chart(&self, f: &mut Frame, lay: &Layout) {
        let (Some(rect), Some(m)) = (lay.chart, &self.now) else { return };
        let blocks_b = self.kept.as_ref().map_or(0, |b| b.design.blocks());
        let most = m.blocks.max(blocks_b) as f64;
        let (a, c) = (self.range.get(0), self.range.get(1));
        let p = Plot::new(rect, [0.0, most], [0.0, 1.0]);
        p.backing(f, CHART_LEFT, CHART_BELOW - 4.0, CHART_ABOVE - 4.0, 8.0);
        if p.tiny() {
            return;
        }
        let small = lay.small;
        let what = match self.focus {
            Focus::One(t) => format!("length of the gradient at position {}", t + 1),
            Focus::All => "length of the gradient over every position".to_string(),
        };
        let left = rect.0.x - CHART_LEFT + 4.0;
        f.text_px(Vec2::new(left, rect.0.y - CHART_ABOVE + 6.0), what, small, pal::TEXT, Align2::LEFT_TOP);
        // the legend
        let y = rect.0.y - CHART_ABOVE + 10.0 + small;
        let name = |d: &Design| if lay.narrow || rect.1.x - rect.0.x < 300.0 { d.short() } else { d.describe() };
        let trained = |steps: usize| if steps == 0 { String::new() } else { format!(", trained {steps} steps") };
        plot::line_of(f, plot::AXES, Vec2::new(left, y + 6.0), Vec2::new(left + 14.0, y + 6.0), 2.0, BACKWARD);
        f.text_px(Vec2::new(left + 18.0, y + 6.0), format!("now: {}{}", name(&self.design), trained(self.steps)), small - 1.0, pal::TEXT, Align2::LEFT_CENTER);
        if let Some(b) = &self.kept {
            plot::dashed_of(f, CURVE_B, Vec2::new(left, y + 20.0), Vec2::new(left + 14.0, y + 20.0), 2.0, pal::DIM);
            f.text_px(Vec2::new(left + 18.0, y + 20.0), format!("kept: {}{}", name(&b.design), trained(b.steps)), small - 1.0, pal::DIM, Align2::LEFT_CENTER);
        } else {
            f.text_px(Vec2::new(left, y + 20.0), "(\"keep for comparison\" keeps this one, to compare)", small - 1.5, pal::FAINT.mix(pal::DIM, 0.5), Align2::LEFT_CENTER);
        }
        // axes: blocks across, powers of ten up
        p.x_axis(f, "", small - 1.5);
        f.text_px(Vec2::new(rect.1.x, rect.1.y + 20.0), "block (0: the embeddings)", small - 1.5, pal::DIM, Align2::RIGHT_TOP);
        let (min, max) = rect;
        plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        // (as many tick lines in every frame, those off the chart unseen at its edge, so
        // that the batch can be followed from frame to frame while the range glides)
        for k in -12..=12 {
            let height = a * k as f32 + c;
            let inside = (-1e-4..=1.0 + 1e-4).contains(&height);
            let y = p.px_y(height.max(0.0).min(1.0) as f64);
            let ink = if inside { 1.0 } else { 0.0 };
            plot::line_of(f, DECADE_TICKS, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8 * ink));
            plot::line_of(f, DECADE_TICKS, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.12 * ink));
            if inside {
                f.text_px(Vec2::new(min.x - 6.0, y), decade(k), 9.5, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
        let at = |k: usize, height: f32| p.at(k as f64 * 0.5, height as f64);
        if let Some(b) = self.kept.as_ref().and_then(|b| b.now.as_ref()) {
            for k in 1..b.rows() {
                plot::dashed_of(f, CURVE_B, at(k - 1, self.curve_b.get(k - 1)), at(k, self.curve_b.get(k)), 2.0, pal::DIM);
            }
        }
        let pts: Vec<Vec2> = (0..m.rows()).map(|k| at(k, self.curve.get(k))).collect();
        plot::path_of(f, plot::CURVES, &pts, 2.2, BACKWARD);
        for q in pts {
            plot::disc(f, q, 5.0, BACKWARD);
        }
    }

    /// How the gradient at one cell is made up: what the skip road hands down from the
    /// place above, and what came through the part in between, each drawn as its 32
    /// numbers (bars up and down from a line), so that the sum can be seen entry by
    /// entry.
    fn split(&self, f: &mut Frame, lay: &Layout) {
        let (Some((min, max)), Some(m)) = (lay.split, &self.now) else { return };
        hud_panel(f, min, max);
        let small = lay.small;
        let x = min.x + 10.0;
        let mut y = min.y + 8.0;
        let Some((k, s)) = self.studied() else {
            f.text_px(Vec2::new(x, y), "how a cell's gradient is made up", small, pal::TEXT, Align2::LEFT_TOP);
            f.text_px(Vec2::new(x, y + small + 6.0), "point at a cell, or click one to keep it here", small - 1.0, pal::DIM, Align2::LEFT_TOP);
            return;
        };
        if k >= m.rows() {
            return;
        }
        let g = m.grad(k, s);
        let top = k + 1 >= m.rows();
        let skip = if top { &[][..] } else { m.skip(k, s) };
        let road: Vec<f64> = g.iter().zip(skip).map(|(a, b)| a - b).collect();
        f.text_px(Vec2::new(x, y), format!("{}, position {}", row_name(k, false), s + 1), small, pal::TEXT, Align2::LEFT_TOP);
        y += small + 8.0;
        let strip_x = x + if lay.narrow { 100.0 } else { 128.0 };
        let cell = ((max.x - 10.0 - strip_x - 56.0) / WIDTH as f32).min(10.0).max(1.5);
        let most = g.iter().chain(skip).chain(&road).map(|v| v.abs()).fold(0.0f64, f64::max);
        let h = if lay.narrow { 20.0 } else { 28.0 };
        let strip = |f: &mut Frame, y: f32, name: &str, color: Color, v: &[f64]| {
            let mid = y + h * 0.5;
            f.text_px(Vec2::new(x, mid), name, small - 1.0, color, Align2::LEFT_CENTER);
            hud_rect(f, Vec2::new(strip_x, mid - 0.5), Vec2::new(strip_x + WIDTH as f32 * cell, mid + 0.5), pal::FAINT);
            for (j, &e) in v.iter().enumerate().take(WIDTH) {
                let a = strip_x + j as f32 * cell;
                let reach = if most > 0.0 { (e / most) as f32 * (h * 0.5 - 1.0) } else { 0.0 };
                if reach.abs() >= 0.3 {
                    hud_rect_at(f, 2, Vec2::new(a, mid.min(mid - reach)), Vec2::new(a + (cell - 0.6).max(0.8), mid.max(mid - reach)), color);
                }
            }
            f.text_px(Vec2::new(strip_x + WIDTH as f32 * cell + 8.0, mid), short(length(v)), small - 1.0, pal::DIM, Align2::LEFT_CENTER);
        };
        strip(f, y, "gradient here", BACKWARD, g);
        f.text_px(Vec2::new(strip_x - 8.0, y + h + 1.0), "$=$", small, pal::DIM, Align2::RIGHT_CENTER);
        y += h + 2.0;
        if top {
            f.text_px(Vec2::new(x, y + 4.0), "the top of the stream: its gradient comes", small - 1.0, pal::DIM, Align2::LEFT_TOP);
            f.text_px(Vec2::new(x, y + small + 6.0), "from the read-out at this position alone", small - 1.0, pal::DIM, Align2::LEFT_TOP);
            return;
        }
        let part = if k % 2 == 0 { format!("attention {}", k / 2 + 1) } else { format!("MLP {}", k.div_ceil(2)) };
        strip(f, y, "skip road", SKIP, skip);
        f.text_px(Vec2::new(strip_x - 8.0, y + h + 1.0), "$+$", small, pal::DIM, Align2::RIGHT_CENTER);
        y += h + 2.0;
        strip(f, y, &format!("through {part}"), THROUGH, &road);
        y += h + 6.0;
        let note = if !self.design.skips {
            "no skip connections: all of it came through the part"
        } else if self.design.arrange == Arrange::Post {
            "skip road: the gradient above, through its normalisation"
        } else {
            "skip road: the gradient of the cell above, unchanged"
        };
        if y + small <= max.y {
            f.text_px(Vec2::new(x, y), note, small - 1.5, pal::DIM, Align2::LEFT_TOP);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::arch::EPS;

    /// Where a finite difference nudges the forward pass: the stream at row k, or (post-norm)
    /// the sum before the normalisation that makes the row above row k.
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum At {
        Row(usize),
        Sum(usize),
    }

    /// The forward pass as `TransformerSpec`'s documentation describes it, written out again
    /// in plain loops, independently of the tape, with `nudge` added to one number of the
    /// stream (or of a sum before a normalisation) on the way: (where, position, entry,
    /// amount). Returns the loss of `targets`.
    fn loss_by_hand(model: &Model, tokens: &[usize], targets: &[usize], nudge: Option<(At, usize, usize, f64)>) -> f64 {
        let spec = model.transformer_spec().copied().unwrap();
        let (n, d, dh) = (tokens.len(), spec.d_model, spec.d_head());
        let g = |name: &str| model.group(name).map(|g| g.value.as_slice()).unwrap_or_else(|| panic!("no group {name}"));
        let normed = spec.norm != Norm::None;
        let (pre, post) = (normed && spec.place == Place::Pre, normed && spec.place == Place::Post);
        let push = |at: At, v: &mut [f64]| {
            if let Some((w, s, j, by)) = nudge {
                if w == at {
                    v[s * d + j] += by;
                }
            }
        };
        let norm = |v: &[f64], prefix: &str| -> Vec<f64> {
            let (gain, bias) = (g(&format!("{prefix}norm gain")), g(&format!("{prefix}norm bias")));
            let mut out = v.to_vec();
            for row in out.chunks_mut(d) {
                let mean = row.iter().sum::<f64>() / d as f64;
                let var = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / d as f64;
                for (j, x) in row.iter_mut().enumerate() {
                    *x = (*x - mean) / (var + EPS).sqrt() * gain[j] + bias[j];
                }
            }
            out
        };
        // x W^T (+ b) for x of n rows, W of `rows` rows
        let linear = |x: &[f64], w: &[f64], b: Option<&[f64]>, rows: usize, cols: usize| -> Vec<f64> {
            let mut out = vec![0.0; n * rows];
            for s in 0..n {
                for r in 0..rows {
                    out[s * rows + r] = (0..cols).map(|c| x[s * cols + c] * w[r * cols + c]).sum::<f64>() + b.map_or(0.0, |b| b[r]);
                }
            }
            out
        };
        let mut x: Vec<f64> = (0..n * d).map(|e| g("token embedding")[tokens[e / d] * d + e % d] + g("position embedding")[e]).collect();
        push(At::Row(0), &mut x);
        for b in 1..=spec.layers {
            let h = if pre { norm(&x, &format!("block {b} attention ")) } else { x.clone() };
            let mut attn = vec![0.0; n * d];
            for head in 1..=spec.heads {
                let m = |what: &str| g(&format!("block {b} head {head} {what}"));
                let (q, k, v) = (linear(&h, m("query"), None, dh, d), linear(&h, m("key"), None, dh, d), linear(&h, m("value"), None, dh, d));
                let mut mix = vec![0.0; n * dh];
                for i in 0..n {
                    let scores: Vec<f64> = (0..=i).map(|j| (0..dh).map(|c| q[i * dh + c] * k[j * dh + c]).sum::<f64>() / (dh as f64).sqrt()).collect();
                    let top = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let total: f64 = scores.iter().map(|s| (s - top).exp()).sum();
                    for (j, sc) in scores.iter().enumerate() {
                        let a = (sc - top).exp() / total;
                        for c in 0..dh {
                            mix[i * dh + c] += a * v[j * dh + c];
                        }
                    }
                }
                for (o, y) in attn.iter_mut().zip(linear(&mix, m("output"), None, d, dh)) {
                    *o += y;
                }
            }
            let mut sum: Vec<f64> = if spec.residual { x.iter().zip(&attn).map(|(a, b)| a + b).collect() } else { attn };
            if post {
                push(At::Sum(2 * b - 2), &mut sum);
                x = norm(&sum, &format!("block {b} attention "));
            } else {
                x = sum;
            }
            push(At::Row(2 * b - 1), &mut x);
            let h = if pre { norm(&x, &format!("block {b} MLP ")) } else { x.clone() };
            let w = |what: &str| g(&format!("block {b} MLP {what}"));
            let hidden: Vec<f64> = linear(&h, w("in weights"), Some(w("in bias")), spec.d_mlp, d).into_iter().map(|z| spec.act.f(z)).collect();
            let y = linear(&hidden, w("out weights"), Some(w("out bias")), d, spec.d_mlp);
            let mut sum: Vec<f64> = if spec.residual { x.iter().zip(&y).map(|(a, b)| a + b).collect() } else { y };
            if post {
                push(At::Sum(2 * b - 1), &mut sum);
                x = norm(&sum, &format!("block {b} MLP "));
            } else {
                x = sum;
            }
            push(At::Row(2 * b), &mut x);
        }
        if pre {
            x = norm(&x, "final ");
        }
        let logits = linear(&x, g("unembedding"), None, spec.vocab, d);
        let (mut total, mut scored) = (0.0, 0);
        for (s, &t) in targets.iter().enumerate().take(n) {
            if t < spec.vocab {
                let row = &logits[s * spec.vocab..(s + 1) * spec.vocab];
                let top = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                total += row.iter().map(|z| (z - top).exp()).sum::<f64>().ln() + top - row[t];
                scored += 1;
            }
        }
        total / scored.max(1) as f64
    }

    /// The designs the gradients are checked on: every arrangement with and without skips,
    /// two blocks deep (and the default four), GELU, and ReLU once.
    fn designs() -> Vec<Design> {
        let mut out = Vec::new();
        for arrange in [Arrange::Pre, Arrange::Post, Arrange::Plain] {
            for skips in [true, false] {
                out.push(Design { arrange, skips, depth: 0, ..DEFAULT });
            }
        }
        out.push(DEFAULT);
        out.push(Design { relu: true, gain: 2, ..DEFAULT });
        out
    }

    fn close(a: f64, b: f64, rel: f64, abs: f64) -> bool {
        (a - b).abs() <= rel * a.abs().max(b.abs()) + abs
    }

    /// The hand-written forward pass gives the library's loss, so that its finite
    /// differences are differences of the same loss.
    #[test]
    fn the_forward_pass_written_out_gives_the_librarys_loss() {
        let (tokens, next) = shown();
        let mut tape = Tape::new();
        for d in designs().into_iter().chain([Design { depth: 2, ..DEFAULT }, Design { arrange: Arrange::Post, depth: 2, gain: 0, ..DEFAULT }]) {
            let model = build(&d);
            for focus in [Focus::One(OPENING), Focus::One(0), Focus::All] {
                let m = measure(&model, &mut tape, &tokens, &next, focus);
                let by_hand = loss_by_hand(&model, &tokens, &focus.targets(&next), None);
                assert!(close(m.loss, by_hand, 1e-12, 0.0), "{}: {focus:?}: {} against {by_hand}", d.describe(), m.loss);
            }
        }
    }

    /// Every cell's gradient, entry by entry, against central differences of the loss
    /// with that number of the stream nudged; and so is the skip road's share where it
    /// is the gradient at the sum before a normalisation (post-norm).
    #[test]
    fn every_cell_is_the_finite_difference_of_the_loss() {
        let (tokens, next) = shown();
        let mut tape = Tape::new();
        let mut rng = Rng::new(5);
        for d in designs() {
            let model = build(&d);
            for focus in [Focus::One(OPENING), Focus::All] {
                let targets = focus.targets(&next);
                let m = measure(&model, &mut tape, &tokens, &next, focus);
                let fd = |at: At, s: usize, j: usize| {
                    let h = 1e-5;
                    (loss_by_hand(&model, &tokens, &targets, Some((at, s, j, h))) - loss_by_hand(&model, &tokens, &targets, Some((at, s, j, -h)))) / (2.0 * h)
                };
                let post = d.arrange == Arrange::Post && d.skips;
                for k in 0..m.rows() {
                    for s in [0, 3, OPENING - 1, OPENING, OPENING + 1, CONTEXT - 1] {
                        for j in [rng.below(WIDTH), rng.below(WIDTH)] {
                            let (tape_value, differences) = (m.grad(k, s)[j], fd(At::Row(k), s, j));
                            assert!(close(tape_value, differences, 1e-6, 1e-9), "{}, {focus:?}, row {k}, position {s}, entry {j}: tape {tape_value}, differences {differences}", d.describe());
                            if post && k + 1 < m.rows() {
                                let (tape_value, differences) = (m.skip(k, s)[j], fd(At::Sum(k), s, j));
                                assert!(close(tape_value, differences, 1e-6, 1e-9), "{}: skip road at row {k}, position {s}: tape {tape_value}, differences {differences}", d.describe());
                            }
                        }
                    }
                }
            }
        }
    }

    /// Every bar is the length of the gradient of the loss with respect to a part's
    /// weights: each of its groups checked, a few entries each, against central
    /// differences of the loss with that weight nudged; and the bar's length is the
    /// length of their gradients together.
    #[test]
    fn every_bar_is_the_finite_difference_of_the_loss() {
        let (tokens, next) = shown();
        let mut tape = Tape::new();
        let mut rng = Rng::new(6);
        for d in [DEFAULT, Design { arrange: Arrange::Post, depth: 0, ..DEFAULT }, Design { skips: false, depth: 0, relu: true, ..DEFAULT }] {
            let model = build(&d);
            for focus in [Focus::One(OPENING), Focus::All] {
                let targets = focus.targets(&next);
                let m = measure(&model, &mut tape, &tokens, &next, focus);
                let tr = model.loss_transformer(&mut tape, &tokens, &targets);
                tape.backward(tr.loss);
                let grads: Vec<Vec<f64>> = tr.params.iter().map(|&v| tape.grad(v).to_vec()).collect();
                let loss_of = |m: &Model, tape: &mut Tape| {
                    let tr = m.loss_transformer(tape, &tokens, &targets);
                    tape.value(tr.loss)[0]
                };
                let mut sums = vec![0.0; m.parts.len()];
                for (gi, (group, grad)) in model.groups().iter().zip(&grads).enumerate() {
                    if let Some(i) = part_of(group, m.blocks) {
                        sums[i] += grad.iter().map(|x| x * x).sum::<f64>();
                    }
                    let n = grad.len();
                    for e in [0, n - 1, rng.below(n)] {
                        let mut moved = model.clone();
                        let x = moved.groups()[gi].value[e];
                        let h = 1e-5 * (1.0 + x.abs());
                        moved.groups_mut()[gi].value[e] = x + h;
                        let up = loss_of(&moved, &mut tape);
                        moved.groups_mut()[gi].value[e] = x - h;
                        let down = loss_of(&moved, &mut tape);
                        let differences = (up - down) / (2.0 * h);
                        assert!(close(grad[e], differences, 1e-5, 1e-9), "{}: {} entry {e}: tape {}, differences {differences}", d.describe(), group.name, grad[e]);
                    }
                }
                for (bar, sum) in m.parts.iter().zip(&sums) {
                    assert!(close(bar.grad, sum.sqrt(), 1e-12, 0.0));
                }
                // every part has a bar: the embeddings, two heads and an MLP a block, the read-out
                assert_eq!(m.parts.len(), 3 * d.blocks() + 2);
                assert!(m.parts.iter().all(|p| p.grad > 0.0 && p.value > 0.0), "{}", d.describe());
            }
        }
    }

    /// A cell's split adds up: the skip road's share and the part's, the latter worked
    /// out on its own by sending the gradient above back through the part alone
    /// (attention or the MLP, with its normalisation), sum to the cell's gradient. In
    /// pre-norm the skip road's share is the gradient of the cell above itself; without
    /// skips it is 0.
    #[test]
    fn a_cells_split_adds_up_to_its_gradient() {
        let (tokens, next) = shown();
        let mut tape = Tape::new();
        for d in designs() {
            let model = build(&d);
            let focus = Focus::One(OPENING);
            let m = measure(&model, &mut tape, &tokens, &next, focus);
            let tr = model.loss_transformer(&mut tape, &tokens, &focus.targets(&next));
            for k in 0..m.rows() - 1 {
                let b = k / 2 + 1;
                // the part between row k and row k + 1, and what its output receives
                let (out, input) = if k % 2 == 0 { (tr.get(Stage::Attn, b).unwrap(), place(k)) } else { (tr.get(Stage::Mlp, b).unwrap(), place(k)) };
                let arriving: Vec<f64> = if d.skips { m.skips[k].clone() } else { m.grads[k + 1].clone() };
                // (without skips in post-norm, the part's output is the sum itself: what
                // arrives there is the gradient through the normalisation, which the trace
                // records as the sum's)
                let arriving = if !d.skips && d.arrange == Arrange::Post {
                    let stage = if k % 2 == 0 { Stage::AttnSum } else { Stage::MlpSum };
                    tape.backward(tr.loss);
                    tape.grad(tr.get(stage, b).unwrap()).to_vec()
                } else {
                    arriving
                };
                tape.backward_seeded(out, &arriving);
                let road = tape.grad(tr.get(input.0, input.1).unwrap()).to_vec();
                for s in 0..CONTEXT {
                    for j in 0..WIDTH {
                        let (whole, skip, through) = (m.grad(k, s)[j], m.skip(k, s)[j], road[s * WIDTH + j]);
                        assert!(close(whole, skip + through, 1e-10, 1e-14), "{}: row {k}, position {s}, entry {j}: {whole} against {skip} + {through}", d.describe());
                        if !d.skips {
                            assert_eq!(skip, 0.0);
                        } else if d.arrange != Arrange::Post {
                            assert_eq!(skip, m.grad(k + 1, s)[j]);
                        }
                    }
                }
            }
        }
    }

    /// With the loss at one position, every cell right of it is exactly 0 in every
    /// design, and at the top only its own; an MLP hands a gradient arriving at one
    /// position back to that position alone, while attention hands it to that position
    /// and earlier ones, never later ones.
    #[test]
    fn nothing_reaches_later_positions_and_only_attention_goes_sideways() {
        let (tokens, next) = shown();
        let mut tape = Tape::new();
        for d in designs() {
            let model = build(&d);
            for t in [0, 4, OPENING, CONTEXT - 2] {
                let m = measure(&model, &mut tape, &tokens, &next, Focus::One(t));
                for k in 0..m.rows() {
                    for s in t + 1..CONTEXT {
                        assert!(m.grad(k, s).iter().all(|&g| g == 0.0), "{}: row {k}, position {s}, loss at {t}", d.describe());
                    }
                }
                let top = m.rows() - 1;
                assert!((0..CONTEXT).all(|s| (m.size(top, s) > 0.0) == (s == t)), "{}: the top row", d.describe());
            }
            // each part on its own: a gradient sent into one position of its output
            let tr = model.loss_transformer(&mut tape, &tokens, &next);
            for b in 1..=d.blocks() {
                for u in [0, 5, CONTEXT - 1] {
                    // (not the same number in every entry: a layer normalisation sends that back as 0)
                    let mut seed = vec![0.0; CONTEXT * WIDTH];
                    for (j, e) in seed[u * WIDTH..(u + 1) * WIDTH].iter_mut().enumerate() {
                        *e = (j as f64 * 0.7).sin();
                    }
                    let rows_reached = |tape: &Tape, stage: Stage, layer: usize| -> Vec<bool> {
                        let g = tape.grad(tr.get(stage, layer).unwrap());
                        (0..CONTEXT).map(|s| g[s * WIDTH..(s + 1) * WIDTH].iter().any(|&x| x != 0.0)).collect()
                    };
                    // the MLP of block b: from after it back to after attention b
                    tape.backward_seeded(tr.get(Stage::Stream, b).unwrap(), &seed);
                    let reached = rows_reached(&tape, Stage::Mid, b);
                    assert!((0..CONTEXT).all(|s| reached[s] == (s == u)), "{}: MLP {b} from position {u}: {reached:?}", d.describe());
                    // attention b: from after it back to the row under it
                    tape.backward_seeded(tr.get(Stage::Mid, b).unwrap(), &seed);
                    let reached = rows_reached(&tape, Stage::Stream, b - 1);
                    assert!((0..CONTEXT).all(|s| !reached[s] || s <= u) && (0..=u).all(|s| reached[s]), "{}: attention {b} from position {u}: {reached:?}", d.describe());
                }
            }
        }
    }

    /// Every setting the buttons can reach, with either loss, gives finite numbers to
    /// draw, and the colour levels and the chart's range are finite too.
    #[test]
    fn every_design_gives_numbers_to_draw() {
        let (tokens, next) = shown();
        let mut tape = Tape::new();
        for arrange in [Arrange::Pre, Arrange::Post, Arrange::Plain] {
            for skips in [true, false] {
                for relu in [false, true] {
                    for gain in 0..3 {
                        for depth in 0..3 {
                            let d = Design { arrange, skips, relu, gain, depth };
                            let model = build(&d);
                            for focus in [Focus::One(0), Focus::One(CONTEXT - 2), Focus::All] {
                                let m = measure(&model, &mut tape, &tokens, &next, focus);
                                assert_eq!(m.rows(), d.rows());
                                let numbers = m.sizes.iter().chain(&m.profile).chain(m.parts.iter().flat_map(|p| [&p.grad, &p.value])).chain([&m.loss]);
                                assert!(numbers.clone().all(|v| v.is_finite()), "{}: {focus:?}", d.describe());
                                assert!(m.chance.iter().take(CONTEXT - 1).all(|p| (0.0..=1.0).contains(p)));
                                let levels = m.sizes.iter().map(|&v| m.cell_scale().level(v)).chain(m.parts.iter().map(|p| m.bar_scale().level(p.grad)));
                                assert!(levels.clone().all(|u| u.is_finite() && (NOTHING..=1.0).contains(&u)));
                            }
                        }
                    }
                }
            }
        }
        // the scale with nothing, or nothing but zeros and NaN, on it
        let empty = Scale::of([0.0, f64::NAN].into_iter(), DECADES);
        assert_eq!(empty, Scale { lo: -2.0, hi: 0.0 });
        assert_eq!(empty.level(0.0), NOTHING);
        assert_eq!(empty.level(f64::NAN), NOTHING);
        assert_eq!(empty.level(f64::INFINITY), 1.0);
        // one size alone spans one power of ten; many at most `DECADES`
        assert_eq!(Scale::of([0.3].into_iter(), DECADES), Scale { lo: -1.0, hi: 0.0 });
        assert_eq!(Scale::of([3e-9, 2.0].into_iter(), DECADES), Scale { lo: -4.0, hi: 1.0 });
    }

    /// The stream starts with numbers of about 1 (their root mean square), and each
    /// block of the opening's network adds about as much (what the panel says of the
    /// embeddings' scale).
    #[test]
    fn the_stream_starts_at_the_size_each_block_adds() {
        let (tokens, _) = shown();
        let mut tape = Tape::new();
        let model = build(&DEFAULT);
        let tr = model.forward_transformer(&mut tape, &tokens);
        let rms = |v: &[f64]| (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt();
        let stream = |b: usize| tape.value(tr.get(Stage::Stream, b).unwrap()).to_vec();
        assert!((0.9..1.1).contains(&rms(&stream(0))), "{}", rms(&stream(0)));
        for b in 1..=DEFAULT.blocks() {
            let added: Vec<f64> = stream(b).iter().zip(stream(b - 1)).map(|(x, y)| x - y).collect();
            assert!((0.5..1.5).contains(&rms(&added)), "block {b} adds {}", rms(&added));
        }
    }

    /// The scene measured as a thing to try leaves it.
    fn tried(i: usize) -> TfGrad {
        let mut s = TfGrad::new();
        s.set_up(i);
        if s.training {
            s.finish_training();
        }
        s.remeasure();
        s
    }

    /// A number as a caption gives it, to so many decimals.
    fn says(v: f64, decimals: i32, shown: f64) -> bool {
        let k = 10f64.powi(decimals);
        ((v * k).round() - shown * k).abs() < 1e-9
    }

    #[test]
    fn where_one_losss_gradient_goes() {
        let s = tried(0);
        let m = s.now.as_ref().unwrap();
        let t = OPENING;
        assert_eq!((t + 1, LETTERS[s.next[t]]), (9, "E"));
        assert!(TRIALS[0].look_for.contains(&format!("next token is {}", token(s.next[t]))));
        assert_eq!(s.design, DEFAULT);
        assert!(s.kept.is_none() && s.focus == Focus::One(t));
        let top = m.rows() - 1;
        assert!((0..CONTEXT).all(|c| (m.size(top, c) > 0.0) == (c == t)));
        for b in 1..=m.blocks {
            // crossing attention b, down to the row under it: every earlier position gets some
            assert!((0..=t).all(|c| m.size(2 * b - 2, c) > 0.0), "under attention {b}");
            // crossing MLP b, down to after attention b: each column keeps its zeros
            assert!((0..CONTEXT).all(|c| m.size(2 * b, c) != 0.0 || m.size(2 * b - 1, c) == 0.0), "under MLP {b}");
        }
        assert!((0..m.rows()).all(|k| (t + 1..CONTEXT).all(|c| m.size(k, c) == 0.0)));
        let column: Vec<f64> = (0..m.rows()).map(|k| m.size(k, t)).collect();
        let (lo, hi) = (column.iter().fold(f64::INFINITY, |a, &b| a.min(b)), column.iter().fold(0.0f64, |a, &b| a.max(b)));
        assert!(says(lo, 2, 0.50) && says(hi, 2, 0.67), "{column:?}");
        assert_eq!(m.profile, column);
    }

    #[test]
    fn only_attention_carries_it_sideways() {
        let s = tried(1);
        let m = s.now.as_ref().unwrap();
        let (k, c) = s.picked.unwrap();
        assert_eq!((row_name(k, false).as_str(), c + 1), ("after MLP 3", 4));
        assert_eq!(row_name(k + 1, false), "after attention 4");
        assert!((0..CONTEXT).all(|u| (m.size(k + 1, u) == 0.0) == (u != OPENING)));
        let road = |k: usize, c: usize| -> Vec<f64> { m.grad(k, c).iter().zip(m.skip(k, c)).map(|(a, b)| a - b).collect() };
        assert_eq!(length(m.skip(k, c)), 0.0);
        assert!(says(length(&road(k, c)), 3, 0.016));
        // the cell under it: after attention 3, the skip road and MLP 3
        assert_eq!(row_name(k - 1, false), "after attention 3");
        assert!(says(length(m.skip(k - 1, c)), 3, 0.016) && says(length(&road(k - 1, c)), 3, 0.004));
    }

    #[test]
    fn pre_norm_against_post_norm() {
        let s = tried(2);
        let (a, b) = (s.now.as_ref().unwrap(), s.kept.as_ref().and_then(|b| b.now.as_ref()).unwrap());
        let kept = s.kept.as_ref().unwrap();
        assert_eq!((s.design.arrange, s.design.blocks(), kept.design.arrange, kept.design.blocks()), (Arrange::Post, 8, Arrange::Pre, 8));
        assert!(s.design.skips && kept.design.skips && s.design.gain == 1 && kept.design.gain == 1 && kept.steps == 0 && s.steps == 0);
        let t = OPENING;
        let pre: Vec<f64> = (0..b.rows()).map(|k| b.size(k, t)).collect();
        let (lo, hi) = (pre.iter().fold(f64::INFINITY, |x, &y| x.min(y)), pre.iter().fold(0.0f64, |x, &y| x.max(y)));
        assert!(says(lo, 2, 0.34) && says(hi, 2, 0.56), "{pre:?}");
        let top = a.rows() - 1;
        assert!(says(a.size(top, t), 2, 1.11) && says(a.size(0, t), 2, 0.21));
        assert!(says(a.size(0, 0), 1, 2.9) && (1..CONTEXT).all(|c| a.size(0, c) < a.size(0, 0)));
        // the spread of each sum a normalisation on position 9's road divides by
        let mut tape = Tape::new();
        let tr = s.model.loss_transformer(&mut tape, &s.tokens, &Focus::One(t).targets(&s.next));
        let spreads: Vec<f64> = (1..=8)
            .flat_map(|b| [Stage::AttnSum, Stage::MlpSum].map(|stage| (stage, b)))
            .map(|(stage, b)| {
                let row = &tape.value(tr.get(stage, b).unwrap())[t * WIDTH..(t + 1) * WIDTH];
                let mean = row.iter().sum::<f64>() / WIDTH as f64;
                (row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / WIDTH as f64 + EPS).sqrt()
            })
            .collect();
        let (lo, hi) = (spreads.iter().fold(f64::INFINITY, |x, &y| x.min(y)), spreads.iter().fold(0.0f64, |x, &y| x.max(y)));
        assert!(says(lo, 1, 1.0) && says(hi, 1, 1.5), "{spreads:?}");
    }

    #[test]
    fn take_the_skips_away() {
        let s = tried(3);
        let (a, b) = (s.now.as_ref().unwrap(), s.kept.as_ref().and_then(|b| b.now.as_ref()).unwrap());
        assert_eq!(s.design, Design { skips: false, ..DEFAULT });
        assert_eq!(s.kept.as_ref().unwrap().design, DEFAULT);
        let t = OPENING;
        let top = a.rows() - 1;
        let (high, low) = (a.size(top, t), a.size(0, t));
        assert!(says(high, 2, 1.13) && says(low, 4, 0.0056));
        assert!(says(high / low, -1, 200.0), "{}", high / low);
        let mut order: Vec<usize> = (0..CONTEXT).collect();
        order.sort_by(|&x, &y| a.size(0, y).total_cmp(&a.size(0, x)));
        assert_eq!(&order[..2], &[0, 1]);
        assert!(says(a.size(0, 0), 2, 0.26) && says(a.size(0, 1), 2, 0.20));
        // and B is the opening's network
        assert_eq!(b.profile, tried(0).now.unwrap().profile);
    }

    /// (300 training steps: about a second and a half.)
    #[test]
    fn after_training() {
        let s = tried(4);
        let (a, b) = (s.now.as_ref().unwrap(), s.kept.as_ref().and_then(|b| b.now.as_ref()).unwrap());
        assert_eq!((s.steps, s.kept.as_ref().unwrap().steps, s.design, s.focus), (300, 0, DEFAULT, Focus::All));
        assert!((5..=10).all(|c| a.chance[c] >= 0.96), "{:?}", a.chance);
        assert!(says(a.loss, 2, 1.08) && says(b.loss, 2, 3.12), "{} {}", a.loss, b.loss);
        let top = a.rows() - 1;
        let first: Vec<f64> = (0..5).map(|c| a.size(top, c)).collect();
        let second: Vec<f64> = (5..11).map(|c| a.size(top, c)).collect();
        let lo = |v: &[f64]| v.iter().fold(f64::INFINITY, |x, &y| x.min(y));
        let hi = |v: &[f64]| v.iter().fold(0.0f64, |x, &y| x.max(y));
        assert!(says(lo(&first), 3, 0.012) && says(hi(&first), 3, 0.014), "{first:?}");
        assert!(says(lo(&second), 5, 0.00004) && says(hi(&second), 4, 0.0008), "{second:?}");
        let ratios: Vec<f64> = b.profile.iter().zip(&a.profile).map(|(b, a)| b / a).collect();
        assert!(says(lo(&ratios), 0, 5.0) && says(hi(&ratios), 0, 8.0), "{ratios:?}");
    }

    /// Training takes the steps asked for and stops; pressing again pauses it; a change
    /// of design starts the network again; the glides come to rest.
    #[test]
    fn training_stops_where_it_should_and_the_picture_comes_to_rest() {
        let mut s = TfGrad::new();
        s.toggle_training();
        assert!(s.training && s.goal == ROUND);
        for _ in 0..3 * BATCH {
            s.pass();
        }
        assert_eq!(s.steps, 3);
        s.toggle_training();
        assert!(!s.training);
        s.goal = 4;
        s.training = true;
        while s.training {
            s.pass();
        }
        assert_eq!((s.steps, s.in_batch), (4, 0));
        s.choose(Choice::Depth(0));
        assert_eq!((s.steps, s.design.blocks()), (0, 2));
        // (a new shape is drawn at once; a new gain glides there)
        s.advance(1.0 / 60.0, 1.0 / 60.0);
        s.choose(Choice::Gain(2));
        let mut frames = 0;
        while s.moving() {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
            frames += 1;
            assert!(frames < 600, "never comes to rest");
        }
        assert!(frames > 1);
    }
}

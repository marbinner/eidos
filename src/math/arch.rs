//! Networks built from a specification and made to be inspected: a multilayer
//! perceptron and a small decoder-only transformer, recorded on an autograd [`Tape`]
//! so that after a pass every intermediate value and every gradient can be read, and
//! measured the same way by every scene ([`Probe`]).
//!
//! How a scene uses it, each frame or each training step:
//!
//! ```text
//!     let mut model = Model::mlp(&spec, &mut Rng::new(seed));
//!     let (x, y) = spirals(200, 3, &mut Rng::new(seed));
//!     let mut tape = Tape::new();                       // kept from frame to frame
//!
//!     let trace = model.loss_mlp(&mut tape, &x, &y, Loss::CrossEntropy);
//!     tape.backward(trace.loss);
//!     let probe = Probe::new(&tape, &trace);            // what to draw
//!     tape.grad(trace.get(Stage::PreAct, 3).unwrap())   // or any value or gradient
//!     model.zero_grads();
//!     model.accumulate(&tape, &trace);
//!     model.clip(1.0);
//!     model.step_adam(0.01);
//! ```
//!
//! Conventions (those of `autograd`): a batch has one row per example, or in a
//! transformer one row per position of the one sequence a pass reads; every weight
//! matrix is stored with one row per output unit, so that a layer is `W x` for one
//! example as a column and `X Wᵀ` for a batch.
//!
//! Numbering, the same in the names of the parameters and stages and in their
//! `layer` and `head` fields: layer 0 is what comes before the first hidden layer or
//! block (the data and the input map, or the embeddings); layers 1 to `depth` (or
//! `layers`) are the hidden layers or blocks; the one after the last is the read-out
//! (for a transformer: the final normalisation and the unembedding). Heads count from
//! 1, and head 0 means "not a head's".
//!
//! Sizes in a specification are taken as at least 1, and nothing here panics on any
//! number a slider can produce: a gain of 0 gives zero weights, a NaN gives NaN, and
//! both are carried through.
//!
//! What a pass costs, to budget a frame by: one forward pass, its backward sweep and
//! accumulating its gradient, measured natively in a release build (three runs on a
//! shared and busy server, so a quiet machine may do better; the browser has not been
//! measured): a 20-layer MLP 32 wide on a batch of 16 about 0.3 ms with ReLU and no
//! normalisation, 0.65 to 0.75 ms with tanh and layer normalisation or GELU and batch
//! normalisation; a transformer of 4 blocks 32 wide (4 heads, MLPs of 128) on 12 tokens
//! about 0.7 ms with ReLU and 0.8 to 0.95 ms with GELU, and with 2 heads and MLPs of 64
//! about 0.55 ms. The smooth activations cost a library call each (an exponential, a
//! tanh), which is most of the difference. The ignored test `the_cost_of_a_pass`
//! prints these figures.

// (Nothing calls this yet: the backpropagation scenes are its users.)
#![allow(dead_code)]

use super::autograd::{Act, Tape, Var, NO_TARGET};
use super::linalg;
use super::rng::Rng;

/// The guard every normalisation in these networks uses (inside the square root, with
/// the variance or mean square).
pub const EPS: f64 = 1e-5;

/// Which normalisation, if any. Each has a learnt gain per unit (starting at 1); layer
/// and batch normalisation also a learnt bias per unit (starting at 0), RMS
/// normalisation none (as in the transformers that use it). With gain g and bias β,
/// `norm(x)` is g times the standardised x, plus β, unit by unit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Norm {
    /// No normalisation: norm(x) is x, and there are no gains.
    None,
    /// Each unit standardised over the examples of the batch (in a transformer: over
    /// the positions of the sequence, so that every position sees statistics of later
    /// ones and the model is no longer strictly causal; it is offered so that the
    /// choice can be seen to be a poor one).
    Batch,
    /// Each example standardised over its units: mean taken off, divided by the
    /// standard deviation.
    Layer,
    /// Each example divided by the root mean square of its units; nothing taken off.
    Rms,
}

impl Norm {
    /// All four, in the order a control would list them.
    pub const ALL: [Norm; 4] = [Norm::None, Norm::Batch, Norm::Layer, Norm::Rms];

    /// A name for a label.
    pub fn name(self) -> &'static str {
        match self {
            Norm::None => "none",
            Norm::Batch => "batch",
            Norm::Layer => "layer",
            Norm::Rms => "RMS",
        }
    }
}

/// Where a normalisation goes: before the layer, inside the skip connection ("pre":
/// the straight road of a skip connection then has nothing on it), or after the layer
/// and the sum ("post": every step along the road passes through a normalisation).
/// See [`MlpSpec`] and [`TransformerSpec`] for the exact formulas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Place {
    /// Before the layer, inside the skip connection.
    Pre,
    /// After the layer (and the sum, with a skip connection).
    Post,
}

impl Place {
    /// Both.
    pub const ALL: [Place; 2] = [Place::Pre, Place::Post];

    /// A name for a label.
    pub fn name(self) -> &'static str {
        match self {
            Place::Pre => "pre",
            Place::Post => "post",
        }
    }
}

/// How the starting weights of an MLP are drawn, each times the specification's
/// `gain`. A matrix with `fan_in` inputs and `fan_out` outputs gets independent
/// normal entries of mean 0 and variance gain² times 1 / fan_in (LeCun), 2 / (fan_in +
/// fan_out) (Xavier, also called Glorot), or 2 / fan_in (He, also called Kaiming:
/// what keeps the signal's size through ReLU layers); or (orthogonal) gain times a
/// random orthogonal matrix, drawn evenly over all of them: for a square layer Wᵀ W =
/// gain² I exactly; a tall one has orthonormal columns and a wide one orthonormal rows,
/// times gain. Biases start at 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Init {
    /// Variance gain² / fan_in.
    Lecun,
    /// Variance gain² × 2 / (fan_in + fan_out).
    Xavier,
    /// Variance gain² × 2 / fan_in.
    He,
    /// gain times a random orthogonal matrix.
    Orthogonal,
}

impl Init {
    /// All four.
    pub const ALL: [Init; 4] = [Init::Lecun, Init::Xavier, Init::He, Init::Orthogonal];

    /// A name for a label.
    pub fn name(self) -> &'static str {
        match self {
            Init::Lecun => "LeCun",
            Init::Xavier => "Xavier",
            Init::He => "He",
            Init::Orthogonal => "orthogonal",
        }
    }

    /// The variance of one entry of a starting weight matrix at gain 1: the formulas
    /// above, and for an orthogonal matrix 1 / the larger of fan_in and fan_out (each
    /// unit row or column spread over that many entries).
    pub fn variance(self, fan_in: usize, fan_out: usize) -> f64 {
        let (i, o) = (fan_in.max(1) as f64, fan_out.max(1) as f64);
        match self {
            Init::Lecun => 1.0 / i,
            Init::Xavier => 2.0 / (i + o),
            Init::He => 2.0 / i,
            Init::Orthogonal => 1.0 / i.max(o),
        }
    }
}

/// A multilayer perceptron: `depth` hidden layers of `width` units and a linear
/// read-out of `outputs` units. With x the input of a hidden layer, W its weights
/// (`width` rows), b its bias, s = `residual_scale`, and norm the chosen normalisation
/// (the identity for `Norm::None`), the layer computes x' =
///
/// - plain, post: `act(norm(W x + b))`
/// - plain, pre: `act(W norm(x) + b)`
/// - residual, pre: `x + s * act(W norm(x) + b)`
/// - residual, post: `norm(x + s * act(W x + b))`
///
/// A residual network first maps its input to `width` units by a linear layer
/// (`W₀ x + b₀`, no activation), so that x and x' have the same size. In the plain pre
/// arrangement the first hidden layer reads the data as it is, without a
/// normalisation: the data are not the network's to normalise (a layer normalisation
/// of the two coordinates of a point in the plane would leave only which of them is
/// larger). The read-out is `W x + b` on the last hidden layer's output, with no
/// activation or normalisation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlpSpec {
    /// Numbers per example.
    pub inputs: usize,
    /// Units in every hidden layer.
    pub width: usize,
    /// Hidden layers.
    pub depth: usize,
    /// Units of the read-out: classes for cross-entropy, numbers per target for squared
    /// error.
    pub outputs: usize,
    /// The activation of every hidden layer.
    pub act: Act,
    /// The normalisation of every hidden layer.
    pub norm: Norm,
    /// Where it goes (see the formulas above).
    pub place: Place,
    /// Whether every hidden layer adds its result to its input (a skip connection).
    pub residual: bool,
    /// s in `x + s * f(x)`: 1 is the plain skip connection; 1 / √depth keeps the
    /// stream's size from growing with depth.
    pub residual_scale: f64,
    /// How every weight matrix starts (the input map's and the read-out's too).
    pub init: Init,
    /// The factor on every starting weight matrix (see [`Init`]).
    pub gain: f64,
    /// Whether the linear layers have biases (and layer and batch normalisations their
    /// own; their gains are always there).
    pub bias: bool,
}

impl Default for MlpSpec {
    /// Four ReLU layers of 32 on points in the plane in three classes, He's start, no
    /// normalisation and no skips.
    fn default() -> Self {
        MlpSpec {
            inputs: 2,
            width: 32,
            depth: 4,
            outputs: 3,
            act: Act::Relu,
            norm: Norm::None,
            place: Place::Pre,
            residual: false,
            residual_scale: 1.0,
            init: Init::He,
            gain: 1.0,
            bias: true,
        }
    }
}

impl MlpSpec {
    /// The same with every size at least 1 (what a model is built with).
    pub fn clamped(&self) -> MlpSpec {
        MlpSpec { inputs: self.inputs.max(1), width: self.width.max(1), depth: self.depth.max(1), outputs: self.outputs.max(1), ..*self }
    }
}

/// A decoder-only transformer, as small modern ones are. The stream x (one row of
/// `d_model` numbers per position) starts as the sum of a token embedding and a learnt
/// position embedding; then come `layers` blocks, each attention and then an MLP;
/// then (pre-norm only) a final normalisation; then the unembedding, which gives one
/// logit per token of the vocabulary at every position.
///
/// A block, with norm the chosen normalisation (the identity for `Norm::None`):
///
/// - pre: `x = x + attn(norm(x))`, then `x = x + mlp(norm(x))`
/// - post: `x = norm(x + attn(x))`, then `x = norm(x + mlp(x))`
/// - without `residual`, the `x +` is left out.
///
/// `attn` is `heads` heads of width d_head = `d_model / heads` (rounded down, at least
/// 1), each with its own query, key, value and output matrices (Wq, Wk, Wv of d_head
/// rows by d_model, Wo of d_model rows by d_head), causal, their outputs summed. For a
/// head whose input is h (one row per position): Q = h Wqᵀ, K = h Wkᵀ and V = h Wvᵀ; the
/// scores are Q Kᵀ / √d_head (entry (i, j) the query of position i against the key of
/// position j); the pattern is their causal softmax (row i a softmax over positions 0
/// to i, so position i attends only to itself and the positions before it); and the
/// head's output is (pattern V) Woᵀ. `mlp` is `W2 act(W1 h + b1) + b2`, from `d_model`
/// to `d_mlp` units and back. There are no biases in attention.
///
/// Starting weights: every matrix normal with variance `init_gain`² / its fan-in
/// (the output matrices of the heads counting the heads together: fan-in heads ×
/// d_head); the embeddings normal with variance `init_gain`² / d_model, so a token's
/// vector has length about `init_gain`; biases 0, gains 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransformerSpec {
    /// Different tokens (0 to vocab − 1).
    pub vocab: usize,
    /// The longest sequence it reads (the rows of the position embedding).
    pub context: usize,
    /// The width of the stream.
    pub d_model: usize,
    /// Heads in every attention layer.
    pub heads: usize,
    /// Blocks.
    pub layers: usize,
    /// Hidden units of every block's MLP.
    pub d_mlp: usize,
    /// The activation of the MLPs.
    pub act: Act,
    /// The normalisation in every block (and the final one, pre-norm).
    pub norm: Norm,
    /// Pre-norm or post-norm (see the formulas above).
    pub place: Place,
    /// Whether every attention layer and MLP adds to the stream (skip connections).
    pub residual: bool,
    /// The factor on every starting weight (see above).
    pub init_gain: f64,
    /// Whether the unembedding is the token embedding itself (logits `x Eᵀ`) rather
    /// than a matrix of its own.
    pub tied_unembed: bool,
}

impl Default for TransformerSpec {
    /// Two pre-norm blocks of two heads on runs of ten tokens: big enough to learn the
    /// repeated runs of [`repeats`].
    fn default() -> Self {
        TransformerSpec {
            vocab: 10,
            context: 12,
            d_model: 32,
            heads: 2,
            layers: 2,
            d_mlp: 64,
            act: Act::Gelu,
            norm: Norm::Layer,
            place: Place::Pre,
            residual: true,
            init_gain: 1.0,
            tied_unembed: false,
        }
    }
}

impl TransformerSpec {
    /// The same with every size at least 1.
    pub fn clamped(&self) -> TransformerSpec {
        TransformerSpec {
            vocab: self.vocab.max(1),
            context: self.context.max(1),
            d_model: self.d_model.max(1),
            heads: self.heads.max(1),
            layers: self.layers.max(1),
            d_mlp: self.d_mlp.max(1),
            ..*self
        }
    }

    /// The width of each head: `d_model / heads`, rounded down, at least 1.
    pub fn d_head(&self) -> usize {
        (self.d_model.max(1) / self.heads.max(1)).max(1)
    }
}

/// Which kind of model, and its specification (with sizes clamped).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arch {
    /// A multilayer perceptron.
    Mlp(MlpSpec),
    /// A decoder-only transformer.
    Transformer(TransformerSpec),
}

/// What a parameter group is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// The matrix of a linear layer (one row per output unit).
    Weight,
    /// The bias of a linear layer.
    Bias,
    /// A normalisation's gain.
    Gain,
    /// A normalisation's bias.
    Shift,
    /// The token embedding: one row per token of the vocabulary.
    Embedding,
    /// The position embedding: one row per position.
    Position,
    /// The unembedding: one row per token (absent when tied to the embedding).
    Unembedding,
    /// A head's query matrix (d_head rows by d_model).
    Query,
    /// A head's key matrix.
    Key,
    /// A head's value matrix.
    Value,
    /// A head's output matrix (d_model rows by d_head).
    HeadOut,
}

/// Which part of the network a parameter group or a stage belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    /// An MLP's data, and its input map (residual networks only); layer 0.
    Input,
    /// A hidden layer of an MLP.
    Layer,
    /// An MLP's read-out; the layer after the last.
    Readout,
    /// A transformer's embeddings; layer 0.
    Embedding,
    /// A block's attention (its normalisation and heads).
    Attention,
    /// A block's MLP (its normalisation and two layers).
    Mlp,
    /// A transformer's final normalisation and unembedding; the layer after the last.
    Top,
}

/// One named group of parameters: a matrix of weights, a bias, a gain, an embedding.
/// `value` and `grad` are `rows` by `cols`, row-major; `grad` is what
/// [`Model::accumulate`] has added up since [`Model::zero_grads`]. A scene may change
/// the values (to scale a layer's weights, say) but not their number.
#[derive(Clone, Debug)]
pub struct Group {
    /// "layer 3 weights", "block 2 head 1 query", "token embedding", ...
    pub name: String,
    /// What it is.
    pub kind: Kind,
    /// Which part of the network it is in.
    pub part: Part,
    /// See the module's documentation for the numbering.
    pub layer: usize,
    /// For a head's matrices, which head (from 1); otherwise 0.
    pub head: usize,
    /// Rows (a weight matrix: output units; a bias or gain: 1).
    pub rows: usize,
    /// Columns (a weight matrix: input units).
    pub cols: usize,
    /// The present values, row-major.
    pub value: Vec<f64>,
    /// The accumulated gradient, row-major.
    pub grad: Vec<f64>,
    /// Adam's running averages of the gradient and of its square.
    m: Vec<f64>,
    v: Vec<f64>,
}

impl Group {
    /// A group named `name`, of the kind, part, layer and head `at`, of `shape` (rows,
    /// columns), starting at `value`.
    fn new(name: String, at: (Kind, Part, usize, usize), shape: (usize, usize), value: Vec<f64>) -> Group {
        let ((kind, part, layer, head), (rows, cols)) = (at, shape);
        let n = rows * cols;
        Group { name, kind, part, layer, head, rows, cols, value, grad: vec![0.0; n], m: vec![0.0; n], v: vec![0.0; n] }
    }
}

/// What a forward pass computes or records at one point of the network. The tables in
/// [`Trace`] say which appear where and what each is exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stage {
    // ---- the MLP
    /// What a layer receives (layer 0: the data).
    Input,
    /// A linear layer's output, `W · + b`.
    Linear,
    /// A normalisation's output (gain and bias included).
    Normed,
    /// The input of the activation.
    PreAct,
    /// The output of the activation.
    PostAct,
    /// A residual post-norm layer's sum before its normalisation.
    Sum,
    /// What a layer hands on (the read-out's: the network's output).
    Output,
    // ---- the transformer
    /// Each position's token's row of the embedding.
    TokenEmbedding,
    /// Each position's row of the position embedding.
    PositionEmbedding,
    /// The residual stream: after the embeddings (layer 0) and after each block.
    Stream,
    /// Attention's normalised input (pre-norm).
    AttnNormed,
    /// A head's queries, `h Wqᵀ`: positions by d_head.
    Query,
    /// A head's keys, `h Wkᵀ`.
    Key,
    /// A head's values, `h Wvᵀ`.
    Value,
    /// A head's scores, `Q Kᵀ / √d_head`, positions by positions (the entries above the
    /// diagonal are worked out but masked, and get no gradient).
    Scores,
    /// A head's attention pattern: the causal softmax of the scores.
    Pattern,
    /// `pattern V`: what the head gathers, d_head wide.
    Mix,
    /// What the head adds to the stream, `mix Woᵀ`.
    HeadOut,
    /// What attention adds to the stream: the sum of the heads' outputs.
    Attn,
    /// Post-norm: the sum before attention's normalisation.
    AttnSum,
    /// The stream after attention.
    Mid,
    /// The MLP's normalised input (pre-norm).
    MlpNormed,
    /// The MLP's hidden pre-activations, `W1 h + b1`.
    MlpPre,
    /// The MLP's hidden activations.
    MlpHidden,
    /// What the MLP adds to the stream.
    Mlp,
    /// Post-norm: the sum before the MLP's normalisation.
    MlpSum,
    /// The final normalisation's output (pre-norm).
    FinalNormed,
    /// One logit per token of the vocabulary at every position: the unembedding of the
    /// last stream (normalised, pre-norm).
    Logits,
}

impl Stage {
    /// A name for a label ("pre-activation", "pattern", ...).
    pub fn name(self) -> &'static str {
        match self {
            Stage::Input => "input",
            Stage::Linear => "linear output",
            Stage::Normed => "normalised",
            Stage::PreAct => "pre-activation",
            Stage::PostAct => "post-activation",
            Stage::Sum => "sum",
            Stage::Output => "output",
            Stage::TokenEmbedding => "token embedding",
            Stage::PositionEmbedding => "position embedding",
            Stage::Stream => "stream",
            Stage::AttnNormed => "attention input",
            Stage::Query => "query",
            Stage::Key => "key",
            Stage::Value => "value",
            Stage::Scores => "scores",
            Stage::Pattern => "pattern",
            Stage::Mix => "mix",
            Stage::HeadOut => "head output",
            Stage::Attn => "attention output",
            Stage::AttnSum => "attention sum",
            Stage::Mid => "stream after attention",
            Stage::MlpNormed => "MLP input",
            Stage::MlpPre => "MLP pre-activation",
            Stage::MlpHidden => "MLP hidden",
            Stage::Mlp => "MLP output",
            Stage::MlpSum => "MLP sum",
            Stage::FinalNormed => "final normalised",
            Stage::Logits => "logits",
        }
    }
}

/// One recorded stage: which, where, and the tape's handle to its value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Entry {
    /// Which stage.
    pub stage: Stage,
    /// The part of the network it is in.
    pub part: Part,
    /// Its layer or block (see the module's documentation for the numbering).
    pub layer: usize,
    /// For a head's stages, which head (from 1); otherwise 0.
    pub head: usize,
    /// The tape's handle to its value (and gradient).
    pub var: Var,
    /// For an activation's input or output: the activation, and the handle of its
    /// input, whose slopes are the gates (what [`Probe`] counts as shut).
    pub gate: Option<(Act, Var)>,
}

impl Entry {
    /// A label: "layer 3 pre-activation", "block 2 head 1 pattern", "data", ...
    pub fn label(&self) -> String {
        let name = self.stage.name();
        match self.part {
            Part::Input if self.stage == Stage::Input => "data".to_string(),
            Part::Input => format!("input map {name}"),
            Part::Layer => format!("layer {} {name}", self.layer),
            Part::Readout => format!("read-out {name}"),
            Part::Embedding if self.stage == Stage::Stream => "stream after the embeddings".to_string(),
            Part::Embedding | Part::Top => name.to_string(),
            Part::Attention | Part::Mlp if self.head > 0 => format!("block {} head {} {name}", self.layer, self.head),
            Part::Attention | Part::Mlp => format!("block {} {name}", self.layer),
        }
    }
}

/// What a forward pass recorded on the tape: a handle for every stage of the network,
/// the parameters, the output and the loss. After `tape.backward(trace.loss)` the
/// values and gradients of any of them are `tape.value(var)` and `tape.grad(var)`.
///
/// An MLP records, for hidden layer l (1 to depth), with x its Input:
///
/// ```text
///                   Normed            Linear            PreAct    PostAct       Sum           Output
///   plain, post     norm(Linear)      W x + b           Normed    act(PreAct)   -             PostAct
///   plain, pre      norm(x)           W Normed + b      Linear    act(PreAct)   -             PostAct
///   residual, pre   norm(x)           W Normed + b      Linear    act(PreAct)   -             x + s PostAct
///   residual, post  norm(Sum)         W x + b           Linear    act(PreAct)   x + s PostAct Normed
/// ```
///
/// where Normed is there only with a normalisation, and not in the first layer of the
/// plain pre arrangement (which reads the data as it is). Without it, norm is the
/// identity: in plain post PreAct is Linear, in the pre rows Linear is `W x + b`, and in
/// residual post Output is Sum. Several stages may be the same handle (PreAct and
/// Linear, say): every layer has Input, Linear, PreAct, PostAct and Output. Layer 0 has Input (the data, examples by `inputs`) and, in a residual
/// network, Linear and Output (the input map's result). The read-out, layer depth + 1,
/// has Input (the last hidden Output) and Output (the network's output, `trace.output`).
///
/// A transformer records Stream at every layer from 0 (after the embeddings, with
/// TokenEmbedding and PositionEmbedding) to `layers` (after each block); in block b
/// AttnNormed (pre-norm, with a normalisation), for each head Query, Key, Value,
/// Scores, Pattern, Mix and HeadOut, then Attn, AttnSum (post-norm, with a
/// normalisation), Mid, MlpNormed (pre-norm, with one), MlpPre, MlpHidden, Mlp and MlpSum
/// (post-norm, with one); and at layer `layers + 1` FinalNormed (pre-norm, with one) and
/// Logits (`trace.output`, positions by vocabulary).
///
/// How a stream's gradient is made up, in a residual pre-norm block (x' = x + f(norm
/// x)): the gradient at the stream entering a part is the gradient at the stream
/// leaving it (the skip road, handed back unchanged) plus what came back through the
/// part. So in block b the attention road's share of the gradient at `Stream` b − 1 is
/// that gradient minus the one at `Mid` b, and the MLP road's share of the gradient at
/// `Mid` b is that minus the one at `Stream` b. (In the post-norm arrangement every road
/// first passes back through the normalisation, from `Mid` to `AttnSum`, say.)
#[derive(Clone, Debug)]
pub struct Trace {
    /// Every recorded stage, in the order computed (from the input up).
    pub entries: Vec<Entry>,
    /// One handle for each of the model's parameter groups, in the same order as
    /// [`Model::groups`]: their gradients are the parameters'.
    pub params: Vec<Var>,
    /// The network's output: the read-out (examples by `outputs`), or the logits.
    pub output: Var,
    /// The loss, 1 by 1 (NaN for a pass without targets).
    pub loss: Var,
    /// Examples in the batch, or positions in the sequence.
    pub rows: usize,
    /// Hidden layers, or blocks.
    pub layers: usize,
}

impl Trace {
    fn new(rows: usize, layers: usize, params: Vec<Var>, placeholder: Var, per_layer: usize) -> Trace {
        Trace { entries: Vec::with_capacity(per_layer * layers + 8), params, output: placeholder, loss: placeholder, rows, layers }
    }

    fn put(&mut self, stage: Stage, part: Part, layer: usize, head: usize, var: Var) {
        self.entries.push(Entry { stage, part, layer, head, var, gate: None });
    }

    fn gated(&mut self, stage: Stage, part: Part, layer: usize, var: Var, act: Act, input: Var) {
        self.entries.push(Entry { stage, part, layer, head: 0, var, gate: Some((act, input)) });
    }

    /// The entry of a stage at a layer (for a head's stages, head 1's).
    pub fn entry(&self, stage: Stage, layer: usize) -> Option<&Entry> {
        self.entries.iter().find(|e| e.stage == stage && e.layer == layer && e.head <= 1)
    }

    /// The handle of a stage at a layer (for a head's stages, head 1's), if the network
    /// has it there.
    pub fn get(&self, stage: Stage, layer: usize) -> Option<Var> {
        self.entry(stage, layer).map(|e| e.var)
    }

    /// The handle of a head's stage (heads from 1).
    pub fn head(&self, stage: Stage, layer: usize, head: usize) -> Option<Var> {
        self.entries.iter().find(|e| e.stage == stage && e.layer == layer && e.head == head).map(|e| e.var)
    }

    /// Every entry of one stage, from the bottom of the network up (Stream at layers 0
    /// to `layers`, say).
    pub fn all(&self, stage: Stage) -> impl Iterator<Item = &Entry> + '_ {
        self.entries.iter().filter(move |e| e.stage == stage)
    }

    /// Every entry of one layer, in the order computed.
    pub fn of_layer(&self, layer: usize) -> impl Iterator<Item = &Entry> + '_ {
        self.entries.iter().filter(move |e| e.layer == layer)
    }
}

/// What the loss of an MLP is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Loss {
    /// The mean over every entry of the squared difference between output and target
    /// (targets: examples by `outputs` numbers).
    Mse,
    /// Softmax of the outputs, then cross-entropy: the mean over the examples of minus
    /// the log of the probability given to the right class. Targets: one class index
    /// per example, as a whole number (0.0, 1.0, ...); one that is not a class (negative,
    /// too large, not whole, NaN) leaves its example out.
    CrossEntropy,
}

// ----- the model ------------------------------------------------------------------------

/// Where a linear layer's parameters are among the groups.
#[derive(Clone, Copy, Debug)]
struct Linear {
    w: usize,
    b: Option<usize>,
}

/// Where a normalisation's parameters are.
#[derive(Clone, Copy, Debug)]
struct NormAt {
    gain: usize,
    shift: Option<usize>,
}

#[derive(Clone, Debug)]
struct HeadAt {
    q: usize,
    k: usize,
    v: usize,
    o: usize,
}

#[derive(Clone, Debug)]
struct BlockAt {
    attn_norm: Option<NormAt>,
    heads: Vec<HeadAt>,
    mlp_norm: Option<NormAt>,
    mlp_in: Linear,
    mlp_out: Linear,
}

#[derive(Clone, Debug)]
enum Plan {
    Mlp { input_map: Option<Linear>, layers: Vec<(Linear, Option<NormAt>)>, readout: Linear },
    Transformer { tok: usize, pos: usize, blocks: Vec<BlockAt>, final_norm: Option<NormAt>, unembed: Option<usize> },
}

/// A network and its parameters, built from a specification: the parameters as named
/// groups ([`Group`]), their accumulated gradients, and Adam's state. Its passes are
/// recorded on a [`Tape`] the scene keeps, and return a [`Trace`]. A clone is an
/// independent copy (a design kept for comparison, a starting point to return to).
#[derive(Clone, Debug)]
pub struct Model {
    arch: Arch,
    groups: Vec<Group>,
    plan: Plan,
    /// Adam steps taken.
    steps: u32,
    /// 0, 1, 2, ...: the position embedding's rows, looked up every pass.
    positions: Vec<usize>,
}

/// Normal draws with the given standard deviation.
fn normals(n: usize, sd: f64, rng: &mut Rng) -> Vec<f64> {
    (0..n).map(|_| rng.normal() as f64 * sd).collect()
}

/// A starting weight matrix of `rows` (fan-out) by `cols` (fan-in), drawn as `init`
/// says, times `gain`.
fn draw(init: Init, gain: f64, rows: usize, cols: usize, rng: &mut Rng) -> Vec<f64> {
    if init != Init::Orthogonal {
        return normals(rows * cols, gain * init.variance(cols, rows).sqrt(), rng);
    }
    // Gram-Schmidt on a normal matrix's columns gives an evenly drawn orthogonal set
    if rows >= cols {
        let mut a = normals(rows * cols, 1.0, rng);
        linalg::orthonormalise(&mut a, rows, cols);
        a.iter_mut().for_each(|x| *x *= gain);
        a
    } else {
        let mut a = normals(cols * rows, 1.0, rng);
        linalg::orthonormalise(&mut a, cols, rows);
        // the transpose of a cols by rows matrix with orthonormal columns
        (0..rows * cols).map(|k| gain * a[(k % cols) * rows + k / cols]).collect()
    }
}

impl Model {
    fn add(&mut self, name: String, at: (Kind, Part, usize, usize), shape: (usize, usize), value: Vec<f64>) -> usize {
        self.groups.push(Group::new(name, at, shape, value));
        self.groups.len() - 1
    }

    /// A normalisation's gain (and bias, if it has one) for `width` units.
    fn norm_params(&mut self, norm: Norm, with_shift: bool, prefix: &str, part: Part, layer: usize, width: usize) -> Option<NormAt> {
        if norm == Norm::None {
            return None;
        }
        let gain = self.add(format!("{prefix}norm gain"), (Kind::Gain, part, layer, 0), (1, width), vec![1.0; width]);
        let shift =
            (with_shift && norm != Norm::Rms).then(|| self.add(format!("{prefix}norm bias"), (Kind::Shift, part, layer, 0), (1, width), vec![0.0; width]));
        Some(NormAt { gain, shift })
    }

    fn empty(arch: Arch, positions: usize) -> Model {
        Model {
            arch,
            groups: Vec::new(),
            plan: Plan::Mlp { input_map: None, layers: Vec::new(), readout: Linear { w: 0, b: None } },
            steps: 0,
            positions: (0..positions).collect(),
        }
    }

    /// An MLP as `spec` describes (sizes taken as at least 1), its weights drawn from
    /// `rng`. Its groups, in this order: "input map weights" and "input map bias"
    /// (residual networks only, layer 0), then for each hidden layer l "layer l weights",
    /// "layer l bias", "layer l norm gain", "layer l norm bias" (each where it exists),
    /// then "read-out weights" and "read-out bias" (layer depth + 1).
    pub fn mlp(spec: &MlpSpec, rng: &mut Rng) -> Model {
        let s = spec.clamped();
        let mut model = Model::empty(Arch::Mlp(s), 0);
        let linear = |model: &mut Model, prefix: String, part: Part, layer: usize, rows: usize, cols: usize, rng: &mut Rng| {
            let w = model.add(format!("{prefix}weights"), (Kind::Weight, part, layer, 0), (rows, cols), draw(s.init, s.gain, rows, cols, rng));
            let b = s.bias.then(|| model.add(format!("{prefix}bias"), (Kind::Bias, part, layer, 0), (1, rows), vec![0.0; rows]));
            Linear { w, b }
        };
        let input_map = s.residual.then(|| linear(&mut model, "input map ".into(), Part::Input, 0, s.width, s.inputs, rng));
        let mut layers = Vec::with_capacity(s.depth);
        for l in 1..=s.depth {
            let fan_in = if l == 1 && !s.residual { s.inputs } else { s.width };
            let lin = linear(&mut model, format!("layer {l} "), Part::Layer, l, s.width, fan_in, rng);
            // (the plain pre arrangement does not normalise the data)
            let normalised = !(l == 1 && !s.residual && s.place == Place::Pre);
            let norm = if normalised { model.norm_params(s.norm, s.bias, &format!("layer {l} "), Part::Layer, l, s.width) } else { None };
            layers.push((lin, norm));
        }
        let readout = linear(&mut model, "read-out ".into(), Part::Readout, s.depth + 1, s.outputs, s.width, rng);
        model.plan = Plan::Mlp { input_map, layers, readout };
        model
    }

    /// A transformer as `spec` describes (sizes taken as at least 1), its weights drawn
    /// from `rng`. Its groups, in this order: "token embedding", "position embedding"
    /// (layer 0); for each block b "block b attention norm gain" (and "... norm bias"),
    /// for each head h "block b head h query", "... key", "... value", "... output",
    /// then "block b MLP norm gain" (and bias), "block b MLP in weights", "block b MLP in
    /// bias", "block b MLP out weights", "block b MLP out bias"; then "final norm gain"
    /// (and bias; pre-norm only) and "unembedding" (unless tied), layer `layers + 1`.
    pub fn transformer(spec: &TransformerSpec, rng: &mut Rng) -> Model {
        let s = spec.clamped();
        let (d, dh, g) = (s.d_model, s.d_head(), s.init_gain);
        let mut model = Model::empty(Arch::Transformer(s), s.context);
        let sd = |fan_in: usize| g / (fan_in as f64).sqrt();
        let tok = model.add("token embedding".into(), (Kind::Embedding, Part::Embedding, 0, 0), (s.vocab, d), normals(s.vocab * d, sd(d), rng));
        let pos = model.add("position embedding".into(), (Kind::Position, Part::Embedding, 0, 0), (s.context, d), normals(s.context * d, sd(d), rng));
        let mut blocks = Vec::with_capacity(s.layers);
        for b in 1..=s.layers {
            let attn_norm = model.norm_params(s.norm, true, &format!("block {b} attention "), Part::Attention, b, d);
            let heads = (1..=s.heads)
                .map(|h| {
                    let mut matrix = |what: &str, kind: Kind, rows: usize, cols: usize, sd: f64| {
                        model.add(format!("block {b} head {h} {what}"), (kind, Part::Attention, b, h), (rows, cols), normals(rows * cols, sd, rng))
                    };
                    let q = matrix("query", Kind::Query, dh, d, sd(d));
                    let k = matrix("key", Kind::Key, dh, d, sd(d));
                    let v = matrix("value", Kind::Value, dh, d, sd(d));
                    let o = matrix("output", Kind::HeadOut, d, dh, sd(s.heads * dh));
                    HeadAt { q, k, v, o }
                })
                .collect();
            let mlp_norm = model.norm_params(s.norm, true, &format!("block {b} MLP "), Part::Mlp, b, d);
            let mut linear = |what: &str, rows: usize, cols: usize| {
                let w = model.add(format!("block {b} MLP {what} weights"), (Kind::Weight, Part::Mlp, b, 0), (rows, cols), normals(rows * cols, sd(cols), rng));
                let b = model.add(format!("block {b} MLP {what} bias"), (Kind::Bias, Part::Mlp, b, 0), (1, rows), vec![0.0; rows]);
                Linear { w, b: Some(b) }
            };
            let mlp_in = linear("in", s.d_mlp, d);
            let mlp_out = linear("out", d, s.d_mlp);
            blocks.push(BlockAt { attn_norm, heads, mlp_norm, mlp_in, mlp_out });
        }
        let top = s.layers + 1;
        let final_norm = if s.place == Place::Pre { model.norm_params(s.norm, true, "final ", Part::Top, top, d) } else { None };
        let unembed =
            (!s.tied_unembed).then(|| model.add("unembedding".into(), (Kind::Unembedding, Part::Top, top, 0), (s.vocab, d), normals(s.vocab * d, sd(d), rng)));
        model.plan = Plan::Transformer { tok, pos, blocks, final_norm, unembed };
        model
    }

    /// Which kind of model, with the specification it was built from (sizes clamped).
    pub fn arch(&self) -> &Arch {
        &self.arch
    }

    /// The specification of an MLP (sizes clamped); `None` for a transformer.
    pub fn mlp_spec(&self) -> Option<&MlpSpec> {
        match &self.arch {
            Arch::Mlp(s) => Some(s),
            Arch::Transformer(_) => None,
        }
    }

    /// The specification of a transformer (sizes clamped); `None` for an MLP.
    pub fn transformer_spec(&self) -> Option<&TransformerSpec> {
        match &self.arch {
            Arch::Transformer(s) => Some(s),
            Arch::Mlp(_) => None,
        }
    }

    /// The parameter groups, in the order of the constructor's documentation.
    pub fn groups(&self) -> &[Group] {
        &self.groups
    }

    /// The same, to change values (not their number).
    pub fn groups_mut(&mut self) -> &mut [Group] {
        &mut self.groups
    }

    /// Where the group of this name is in [`Model::groups`], which is also where its
    /// handle is in a trace's `params`.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.groups.iter().position(|g| g.name == name)
    }

    /// The group of this name.
    pub fn group(&self, name: &str) -> Option<&Group> {
        self.groups.iter().find(|g| g.name == name)
    }

    /// The group of this name, to change its values.
    pub fn group_mut(&mut self, name: &str) -> Option<&mut Group> {
        self.groups.iter_mut().find(|g| g.name == name)
    }

    /// The number of parameters, every group's entries together.
    pub fn n_params(&self) -> usize {
        self.groups.iter().map(|g| g.value.len()).sum()
    }

    /// Record every group as a parameter on the tape.
    fn params_on(&self, tape: &mut Tape) -> Vec<Var> {
        self.groups.iter().map(|g| tape.param(g.rows, g.cols, &g.value)).collect()
    }

    /// The normalisation `norm` with its gain and bias.
    fn normalise(norm: Norm, tape: &mut Tape, x: Var, at: NormAt, p: &[Var]) -> Var {
        let y = match norm {
            Norm::None => return x,
            Norm::Batch => tape.batch_norm(x, EPS),
            Norm::Layer => tape.layer_norm(x, EPS),
            Norm::Rms => tape.rms_norm(x, EPS),
        };
        let y = tape.mul_row(y, p[at.gain]);
        match at.shift {
            Some(s) => tape.add_row(y, p[s]),
            None => y,
        }
    }

    fn linear(tape: &mut Tape, x: Var, at: Linear, p: &[Var]) -> Var {
        let z = tape.matmul_t(x, p[at.w]);
        match at.b {
            Some(b) => tape.add_row(z, p[b]),
            None => z,
        }
    }

    /// A pass that cannot be made (the wrong kind of model): no stages, NaN output.
    fn no_pass(tape: &mut Tape, rows: usize) -> Trace {
        tape.clear();
        let nan = tape.scalar(f64::NAN);
        Trace::new(rows, 0, Vec::new(), nan, 0)
    }

    /// Record an MLP's forward pass on `tape` (which is cleared first) for a batch of
    /// examples, `inputs` holding `spec.inputs` numbers for each, one example after
    /// another. Returns the trace; its loss is NaN (see [`Model::loss_mlp`]). On a
    /// transformer, a trace with no stages.
    pub fn forward_mlp(&self, tape: &mut Tape, inputs: &[f64]) -> Trace {
        let (Arch::Mlp(s), Plan::Mlp { input_map, layers, readout }) = (&self.arch, &self.plan) else {
            debug_assert!(matches!(self.arch, Arch::Mlp(_)), "forward_mlp on a transformer");
            return Model::no_pass(tape, 0);
        };
        tape.clear();
        let n = inputs.len() / s.inputs;
        debug_assert!(n * s.inputs == inputs.len(), "{} numbers are not a whole number of examples of {}", inputs.len(), s.inputs);
        let p = self.params_on(tape);
        let x0 = tape.constant(n, s.inputs, &inputs[..n * s.inputs]);
        let mut tr = Trace::new(n, s.depth, p, x0, 7);
        let p = std::mem::take(&mut tr.params);
        tr.put(Stage::Input, Part::Input, 0, 0, x0);
        let mut x = x0;
        if let Some(map) = input_map {
            x = Model::linear(tape, x, *map, &p);
            tr.put(Stage::Linear, Part::Input, 0, 0, x);
            tr.put(Stage::Output, Part::Input, 0, 0, x);
        }
        let scaled = |tape: &mut Tape, a: Var| if s.residual_scale == 1.0 { a } else { tape.scale(a, s.residual_scale) };
        for (i, (lin, norm_at)) in layers.iter().enumerate() {
            let l = i + 1;
            let part = Part::Layer;
            tr.put(Stage::Input, part, l, 0, x);
            let norm = |tape: &mut Tape, tr: &mut Trace, v: Var| match norm_at {
                Some(at) => {
                    let y = Model::normalise(s.norm, tape, v, *at, &p);
                    tr.put(Stage::Normed, part, l, 0, y);
                    y
                }
                None => v,
            };
            let out = match (s.residual, s.place) {
                (false, Place::Post) => {
                    let z = Model::linear(tape, x, *lin, &p);
                    tr.put(Stage::Linear, part, l, 0, z);
                    let pre = norm(tape, &mut tr, z);
                    let a = tape.act(pre, s.act);
                    tr.gated(Stage::PreAct, part, l, pre, s.act, pre);
                    tr.gated(Stage::PostAct, part, l, a, s.act, pre);
                    a
                }
                (false, Place::Pre) | (true, Place::Pre) => {
                    let h = norm(tape, &mut tr, x);
                    let z = Model::linear(tape, h, *lin, &p);
                    let a = tape.act(z, s.act);
                    tr.put(Stage::Linear, part, l, 0, z);
                    tr.gated(Stage::PreAct, part, l, z, s.act, z);
                    tr.gated(Stage::PostAct, part, l, a, s.act, z);
                    if s.residual {
                        let branch = scaled(tape, a);
                        tape.add(x, branch)
                    } else {
                        a
                    }
                }
                (true, Place::Post) => {
                    let z = Model::linear(tape, x, *lin, &p);
                    let a = tape.act(z, s.act);
                    tr.put(Stage::Linear, part, l, 0, z);
                    tr.gated(Stage::PreAct, part, l, z, s.act, z);
                    tr.gated(Stage::PostAct, part, l, a, s.act, z);
                    let branch = scaled(tape, a);
                    let sum = tape.add(x, branch);
                    tr.put(Stage::Sum, part, l, 0, sum);
                    norm(tape, &mut tr, sum)
                }
            };
            tr.put(Stage::Output, part, l, 0, out);
            x = out;
        }
        let top = s.depth + 1;
        tr.put(Stage::Input, Part::Readout, top, 0, x);
        let y = Model::linear(tape, x, *readout, &p);
        tr.put(Stage::Output, Part::Readout, top, 0, y);
        tr.output = y;
        tr.loss = tape.scalar(f64::NAN);
        tr.params = p;
        tr
    }

    /// The forward pass of [`Model::forward_mlp`] and its loss against `targets`: for
    /// [`Loss::Mse`] `outputs` numbers per example, row-major, for
    /// [`Loss::CrossEntropy`] one class index per example (see [`Loss`]); a different
    /// number of targets is a mistake (NaN in a release build). Then
    /// `tape.backward(trace.loss)` differentiates.
    pub fn loss_mlp(&self, tape: &mut Tape, inputs: &[f64], targets: &[f64], loss: Loss) -> Trace {
        let mut tr = self.forward_mlp(tape, inputs);
        let (n, outputs) = tape.shape(tr.output);
        if self.mlp_spec().is_none() {
            return tr;
        }
        tr.loss = match loss {
            Loss::Mse => {
                let y = tape.constant(n, outputs, targets);
                tape.mse(tr.output, y)
            }
            Loss::CrossEntropy => {
                let class = |t: f64| if t >= 0.0 && t < outputs as f64 && t.fract() == 0.0 { t as usize } else { NO_TARGET };
                tape.cross_entropy_with(tr.output, targets.len(), targets.iter().map(|&t| class(t)))
            }
        };
        tr
    }

    /// Record a transformer's forward pass on `tape` (cleared first) for one sequence
    /// of tokens (at most `context` of them: any beyond are left out), one row per
    /// position. Returns the trace, its loss NaN (see [`Model::loss_transformer`]). A
    /// token that is not in the vocabulary gives NaN. A batch of sequences is a loop:
    /// a pass, a backward sweep and [`Model::accumulate`] for each. On an MLP, a trace
    /// with no stages.
    pub fn forward_transformer(&self, tape: &mut Tape, tokens: &[usize]) -> Trace {
        let (Arch::Transformer(s), Plan::Transformer { tok, pos, blocks, final_norm, unembed }) = (&self.arch, &self.plan) else {
            debug_assert!(matches!(self.arch, Arch::Transformer(_)), "forward_transformer on an MLP");
            return Model::no_pass(tape, 0);
        };
        tape.clear();
        let t = tokens.len().min(s.context);
        let p = self.params_on(tape);
        let embed = tape.gather_rows(p[*tok], &tokens[..t]);
        let mut tr = Trace::new(t, s.layers, Vec::new(), embed, 7 * s.heads + 12);
        tr.put(Stage::TokenEmbedding, Part::Embedding, 0, 0, embed);
        let place = tape.gather_rows(p[*pos], &self.positions[..t]);
        tr.put(Stage::PositionEmbedding, Part::Embedding, 0, 0, place);
        let mut x = tape.add(embed, place);
        tr.put(Stage::Stream, Part::Embedding, 0, 0, x);
        let (pre, post) = (s.place == Place::Pre, s.place == Place::Post);
        let score_scale = 1.0 / (s.d_head() as f64).sqrt();
        for (i, block) in blocks.iter().enumerate() {
            let b = i + 1;
            // attention
            let part = Part::Attention;
            let h = match (pre, block.attn_norm) {
                (true, Some(at)) => {
                    let y = Model::normalise(s.norm, tape, x, at, &p);
                    tr.put(Stage::AttnNormed, part, b, 0, y);
                    y
                }
                _ => x,
            };
            let mut attn = None;
            for (k, head) in block.heads.iter().enumerate() {
                let k = k + 1;
                let q = tape.matmul_t(h, p[head.q]);
                let key = tape.matmul_t(h, p[head.k]);
                let v = tape.matmul_t(h, p[head.v]);
                let scores = tape.matmul_t(q, key);
                let scores = tape.scale(scores, score_scale);
                let pattern = tape.softmax_rows_causal(scores);
                let mix = tape.matmul(pattern, v);
                let out = tape.matmul_t(mix, p[head.o]);
                for (stage, var) in [
                    (Stage::Query, q),
                    (Stage::Key, key),
                    (Stage::Value, v),
                    (Stage::Scores, scores),
                    (Stage::Pattern, pattern),
                    (Stage::Mix, mix),
                    (Stage::HeadOut, out),
                ] {
                    tr.put(stage, part, b, k, var);
                }
                attn = Some(match attn {
                    Some(sum) => tape.add(sum, out),
                    None => out,
                });
            }
            let attn = attn.unwrap_or(h);
            tr.put(Stage::Attn, part, b, 0, attn);
            let sum = if s.residual { tape.add(x, attn) } else { attn };
            x = match (post, block.attn_norm) {
                (true, Some(at)) => {
                    tr.put(Stage::AttnSum, part, b, 0, sum);
                    Model::normalise(s.norm, tape, sum, at, &p)
                }
                _ => sum,
            };
            tr.put(Stage::Mid, part, b, 0, x);
            // the MLP
            let part = Part::Mlp;
            let h = match (pre, block.mlp_norm) {
                (true, Some(at)) => {
                    let y = Model::normalise(s.norm, tape, x, at, &p);
                    tr.put(Stage::MlpNormed, part, b, 0, y);
                    y
                }
                _ => x,
            };
            let z = Model::linear(tape, h, block.mlp_in, &p);
            let a = tape.act(z, s.act);
            tr.gated(Stage::MlpPre, part, b, z, s.act, z);
            tr.gated(Stage::MlpHidden, part, b, a, s.act, z);
            let y = Model::linear(tape, a, block.mlp_out, &p);
            tr.put(Stage::Mlp, part, b, 0, y);
            let sum = if s.residual { tape.add(x, y) } else { y };
            x = match (post, block.mlp_norm) {
                (true, Some(at)) => {
                    tr.put(Stage::MlpSum, part, b, 0, sum);
                    Model::normalise(s.norm, tape, sum, at, &p)
                }
                _ => sum,
            };
            tr.put(Stage::Stream, part, b, 0, x);
        }
        let top = s.layers + 1;
        if let Some(at) = final_norm {
            x = Model::normalise(s.norm, tape, x, *at, &p);
            tr.put(Stage::FinalNormed, Part::Top, top, 0, x);
        }
        let logits = tape.matmul_t(x, p[unembed.unwrap_or(*tok)]);
        tr.put(Stage::Logits, Part::Top, top, 0, logits);
        tr.output = logits;
        tr.loss = tape.scalar(f64::NAN);
        tr.params = p;
        tr
    }

    /// The forward pass of [`Model::forward_transformer`] and its loss: cross-entropy of
    /// the logits at every position that has a target, averaged over those positions.
    /// `targets[t]` is the token wanted at position t (the next token, for a language
    /// model); a position without one ([`NO_TARGET`], a number not in the vocabulary, or
    /// beyond the end of `targets`) is not scored. To differentiate the loss at one
    /// position alone, give every other position `NO_TARGET`.
    pub fn loss_transformer(&self, tape: &mut Tape, tokens: &[usize], targets: &[usize]) -> Trace {
        let mut tr = self.forward_transformer(tape, tokens);
        if self.transformer_spec().is_none() {
            return tr;
        }
        let t = tr.rows;
        tr.loss = tape.cross_entropy_with(tr.output, t, (0..t).map(|i| targets.get(i).copied().unwrap_or(NO_TARGET)));
        tr
    }

    // ----- training ----------------------------------------------------------------

    /// Add the gradients the tape holds for the trace's parameters into the groups'
    /// `grad` (after `tape.backward`). Called once per example or sequence of a batch,
    /// it sums their gradients; [`Model::zero_grads`] starts again.
    pub fn accumulate(&mut self, tape: &Tape, trace: &Trace) {
        for (group, &v) in self.groups.iter_mut().zip(&trace.params) {
            for (d, &g) in group.grad.iter_mut().zip(tape.grad(v)) {
                *d += g;
            }
        }
    }

    /// Set every accumulated gradient to 0.
    pub fn zero_grads(&mut self) {
        self.groups.iter_mut().for_each(|g| g.grad.fill(0.0));
    }

    /// The length of the whole accumulated gradient (every group's entries as one
    /// vector).
    pub fn grad_norm(&self) -> f64 {
        self.groups.iter().flat_map(|g| &g.grad).map(|g| g * g).sum::<f64>().sqrt()
    }

    /// Scale the accumulated gradient down to length `max_norm` if it is longer, and
    /// return its length before. (A `max_norm` of 0 or below zeroes it; infinity or NaN
    /// leaves it alone.)
    pub fn clip(&mut self, max_norm: f64) -> f64 {
        let norm = self.grad_norm();
        if norm > max_norm {
            let k = if max_norm > 0.0 { max_norm / norm } else { 0.0 };
            self.groups.iter_mut().flat_map(|g| &mut g.grad).for_each(|g| *g *= k);
        }
        norm
    }

    /// One step of plain gradient descent: every parameter p becomes p − lr × its
    /// accumulated gradient.
    pub fn step_sgd(&mut self, lr: f64) {
        for g in &mut self.groups {
            for (p, &d) in g.value.iter_mut().zip(&g.grad) {
                *p -= lr * d;
            }
        }
    }

    /// One step of Adam (Kingma and Ba, in PyTorch's form) with the accumulated
    /// gradient g: m ← 0.9 m + 0.1 g and v ← 0.999 v + 0.001 g², then p ← p − lr m̂ /
    /// (√v̂ + 1e-8), where m̂ = m / (1 − 0.9^t) and v̂ = v / (1 − 0.999^t) undo the
    /// averages' start at zero (t the number of steps so far, this one included). Each
    /// parameter moves by about lr, against the sign its gradient has kept, whatever
    /// the gradient's size; at the first step by lr |g| / (|g| + 1e-8), which is lr but
    /// for a gradient smaller than about 1e-6.
    pub fn step_adam(&mut self, lr: f64) {
        // (saturating: a count that wrapped to 0 would make the corrections 0/0)
        self.steps = self.steps.saturating_add(1);
        let t = self.steps.min(100_000) as i32;
        let (c1, c2) = (1.0 - 0.9f64.powi(t), 1.0 - 0.999f64.powi(t));
        // (numbers this small are flushed to 0: below the normal range of doubles,
        // arithmetic becomes very slow, in WebAssembly most of all)
        let flush = |x: f64| if x.abs() < 1e-300 { 0.0 } else { x };
        for g in &mut self.groups {
            for (((p, &d), m), v) in g.value.iter_mut().zip(&g.grad).zip(g.m.iter_mut()).zip(g.v.iter_mut()) {
                *m = flush(0.9 * *m + 0.1 * d);
                *v = flush(0.999 * *v + 0.001 * d * d);
                *p -= lr * (*m / c1) / ((*v / c2).sqrt() + 1e-8);
            }
        }
    }

    /// Forget Adam's running averages and step count (for a fresh start of training
    /// from the present weights).
    pub fn reset_adam(&mut self) {
        self.steps = 0;
        for g in &mut self.groups {
            g.m.fill(0.0);
            g.v.fill(0.0);
        }
    }

    /// Adam steps taken since the start or [`Model::reset_adam`].
    pub fn adam_steps(&self) -> u32 {
        self.steps
    }
}

// ----- measuring ------------------------------------------------------------------------

/// The numbers a scene shows for one recorded stage after a backward pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageProbe {
    /// The root mean square of the stage's values, over every entry (examples and
    /// units).
    pub rms: f64,
    /// Their mean.
    pub mean: f64,
    /// The root mean square of the gradient there.
    pub grad_rms: f64,
    /// For an activation's input or output: the share of entries (example by unit)
    /// whose gate is shut or nearly so, the slope of the activation there below 0.01
    /// in size: saturated, or (ReLU) switched off. (GELU's and SiLU's slopes pass
    /// through 0 near −0.75 and −1.28, so entries close to those count too; leaky ReLU's
    /// 0.01 does not.) `None` for other stages.
    pub shut: Option<f64>,
    /// For an activation's input or output: the share of units shut for EVERY example
    /// of the batch (every position of the sequence): dead units. A dead unit passes
    /// no gradient back through itself on this data, so (unless a normalisation
    /// between its weights and its activation mixes it with the other units, as in the
    /// plain post arrangement) none reaches its weights and nothing reopens it.
    pub dead: Option<f64>,
}

/// The numbers a scene shows for one parameter group.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupProbe {
    /// The length of the group's values (all its entries as one vector).
    pub value_norm: f64,
    /// The length of this pass's gradient for it.
    pub grad_norm: f64,
}

impl GroupProbe {
    /// The size of a plain gradient-descent step at rate `lr` relative to the weights:
    /// lr × gradient length / value length (∞ for weights that are all 0, NaN if the
    /// gradient is 0 too): the number practitioners watch. (Adam's steps are about lr
    /// per entry instead, whatever the gradient's size.)
    pub fn relative_step(&self, lr: f64) -> f64 {
        lr * self.grad_norm / self.value_norm
    }
}

/// What a scene shows after a forward and backward pass, worked out from the tape and
/// the trace in one place so that every scene measures the same thing: a
/// [`StageProbe`] for each entry of the trace (in the same order as
/// `trace.entries`) and a [`GroupProbe`] for each parameter group (in the order of
/// `trace.params`, which is the model's). The gradients are the tape's, so they are
/// those of whatever was last differentiated (the loss, normally), for this pass only.
#[derive(Clone, Debug)]
pub struct Probe {
    /// One for each entry of the trace, in its order.
    pub entries: Vec<StageProbe>,
    /// One for each parameter group, in the model's order.
    pub groups: Vec<GroupProbe>,
}

/// The root mean square (0 for nothing).
fn rms(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        (v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64).sqrt()
    }
}

impl Probe {
    /// Measure every stage and group of `trace` from the values and gradients on
    /// `tape` (after a backward sweep; before one, the gradients are 0).
    pub fn new(tape: &Tape, trace: &Trace) -> Probe {
        let entries = trace
            .entries
            .iter()
            .map(|e| {
                let values = tape.value(e.var);
                let mean = if values.is_empty() { 0.0 } else { values.iter().sum::<f64>() / values.len() as f64 };
                let (shut, dead) = match e.gate {
                    Some((act, input)) => {
                        let (z, units) = (tape.value(input), tape.cols(input));
                        let closed = |x: f64| act.df(x).abs() < 0.01;
                        let shut = if z.is_empty() { 0.0 } else { z.iter().filter(|&&x| closed(x)).count() as f64 / z.len() as f64 };
                        let examples = tape.rows(input);
                        let dead_units = if examples == 0 { 0 } else { (0..units).filter(|&j| (0..examples).all(|i| closed(z[i * units + j]))).count() };
                        (Some(shut), Some(if units == 0 { 0.0 } else { dead_units as f64 / units as f64 }))
                    }
                    None => (None, None),
                };
                StageProbe { rms: rms(values), mean, grad_rms: rms(tape.grad(e.var)), shut, dead }
            })
            .collect();
        let length = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let groups = trace.params.iter().map(|&v| GroupProbe { value_norm: length(tape.value(v)), grad_norm: length(tape.grad(v)) }).collect();
        Probe { entries, groups }
    }

    /// The probe of the stage at a layer (head 1's for a head's stage), if recorded.
    pub fn of(&self, trace: &Trace, stage: Stage, layer: usize) -> Option<&StageProbe> {
        let i = trace.entries.iter().position(|e| e.stage == stage && e.layer == layer && e.head <= 1)?;
        self.entries.get(i)
    }
}

/// The share of examples classified right: those whose largest output (of `classes`
/// per example, row-major, as in a trace's output) is at the column of their target
/// class (targets as [`Loss::CrossEntropy`] takes them). A tie for the largest, a NaN, or
/// a target that is not a class counts as wrong; no examples, 0.
pub fn share_right(outputs: &[f64], classes: usize, targets: &[f64]) -> f64 {
    let classes = classes.max(1);
    let n = (outputs.len() / classes).min(targets.len());
    let right = outputs
        .chunks_exact(classes)
        .zip(targets)
        .filter(|(o, &t)| {
            let c = t as usize;
            t >= 0.0 && t.fract() == 0.0 && c < classes && (0..classes).all(|k| k == c || o[k] < o[c])
        })
        .count();
    if n == 0 {
        0.0
    } else {
        right as f64 / n as f64
    }
}

/// Which map [`input_output_jacobian`] differentiates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Span {
    /// The output of the last hidden layer with respect to the input of the first (in
    /// a residual network, the input map's output; otherwise the data): the stack.
    Hidden,
    /// The network's output with respect to the data: the whole map.
    Whole,
    /// The output of hidden layer `to` with respect to the input of hidden layer `from`
    /// (both from 1, `from` at most `to`; clamped to the network's layers).
    Layers(usize, usize),
}

/// The Jacobian of an MLP's map for one example, for the scenes that show its singular
/// values (`crate::math::linalg::svd`): returns (matrix row-major, rows, cols), a row
/// for each unit of the map's output and a column for each number of its input, entry
/// (k, l) being how much output k moves per unit change of input l.
///
/// `inputs` is the example (`inputs` numbers), or a batch whose first row is the
/// example: the rest matter only with batch normalisation, whose output for one
/// example depends on the others (they are held fixed; alone in a batch, an example is
/// normalised to nothing). One backward sweep per output unit, at the model's present
/// weights. For a transformer, or no example, returns an empty matrix with 0 rows and
/// columns.
pub fn input_output_jacobian(model: &Model, inputs: &[f64], span: Span) -> (Vec<f64>, usize, usize) {
    let Some(spec) = model.mlp_spec() else { return (Vec::new(), 0, 0) };
    let mut tape = Tape::new();
    let trace = model.forward_mlp(&mut tape, &inputs[..inputs.len() / spec.inputs * spec.inputs]);
    if trace.rows == 0 {
        return (Vec::new(), 0, 0);
    }
    let depth = spec.depth;
    let (from, to) = match span {
        Span::Hidden => ((Stage::Input, 1), (Stage::Output, depth)),
        Span::Whole => ((Stage::Input, 0), (Stage::Output, depth + 1)),
        Span::Layers(a, b) => {
            let b = b.clamp(1, depth);
            ((Stage::Input, a.clamp(1, b)), (Stage::Output, b))
        }
    };
    let (Some(input), Some(output)) = (trace.get(from.0, from.1), trace.get(to.0, to.1)) else { return (Vec::new(), 0, 0) };
    let (rows, cols) = (tape.cols(output), tape.cols(input));
    let mut seed = vec![0.0; tape.value(output).len()];
    let mut jac = vec![0.0; rows * cols];
    for k in 0..rows {
        seed[k] = 1.0;
        tape.backward_seeded(output, &seed);
        seed[k] = 0.0;
        jac[k * cols..(k + 1) * cols].copy_from_slice(&tape.grad(input)[..cols]);
    }
    (jac, rows, cols)
}

// ----- toy data ------------------------------------------------------------------------

/// `n` points in the plane on `classes` (2 or 3; other numbers are taken as the
/// nearer) interleaved spiral arms, with their labels: returns (inputs, two numbers per
/// point, row-major; targets, the arm of each point as a class index 0.0, 1.0, ...),
/// the shapes [`Model::loss_mlp`] takes with [`Loss::CrossEntropy`]. Point i is on arm
/// i mod classes, so any run of points has every arm about equally. On arm c, a point
/// at t (uniform in [0, 1)) is at radius 0.1 + 0.9 t and angle 2π c / classes + 2.5π t,
/// plus a normal angular jitter of 0.1 radian: every arm turns one and a quarter times
/// round, inside the unit disc. (`n` is taken as at least 1.)
pub fn spirals(n: usize, classes: usize, rng: &mut Rng) -> (Vec<f64>, Vec<f64>) {
    let (n, classes) = (n.max(1), classes.clamp(2, 3));
    let mut inputs = Vec::with_capacity(2 * n);
    let mut targets = Vec::with_capacity(n);
    for i in 0..n {
        let c = i % classes;
        let t = rng.f32() as f64;
        let r = 0.1 + 0.9 * t;
        let angle = std::f64::consts::TAU * c as f64 / classes as f64 + 2.5 * std::f64::consts::PI * t + 0.1 * rng.normal() as f64;
        inputs.extend([r * angle.cos(), r * angle.sin()]);
        targets.push(c as f64);
    }
    (inputs, targets)
}

/// The fixed network that [`teacher`] imitates for `inputs` numbers per example: 16
/// tanh units with weights of variance 1 / inputs and biases of variance 1/4, then one
/// output with weights of variance 2.5 / 16 (so that its output has a spread of about
/// 1 on normal inputs), all drawn from a seed of its own.
fn teacher_net(inputs: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(0x7EAC_4E12 ^ inputs as u64);
    let w1 = normals(16 * inputs, (1.0 / inputs as f64).sqrt(), &mut rng);
    let b1 = normals(16, 0.5, &mut rng);
    let w2 = normals(16, (2.5f64 / 16.0).sqrt(), &mut rng);
    (w1, b1, w2)
}

/// A regression: `n` examples of `inputs` numbers each (independent standard normal
/// draws from `rng`) and, as targets, the outputs of a small network fixed once and
/// for all (16 tanh units and one linear output; the same function for every call with
/// the same number of inputs, whatever `rng`), with no noise added: the task can be
/// learnt exactly. Returns (inputs, row-major; targets, one per example), the shapes
/// [`Model::loss_mlp`] takes with [`Loss::Mse`] and `outputs: 1`. (Sizes are taken
/// as at least 1.)
pub fn teacher(n: usize, inputs: usize, rng: &mut Rng) -> (Vec<f64>, Vec<f64>) {
    let (n, k) = (n.max(1), inputs.max(1));
    let (w1, b1, w2) = teacher_net(k);
    let x = normals(n * k, 1.0, rng);
    let y = x.chunks_exact(k).map(|x| (0..16).map(|j| w2[j] * (b1[j] + (0..k).map(|i| w1[j * k + i] * x[i]).sum::<f64>()).tanh()).sum()).collect();
    (x, y)
}

/// The length of the run that [`repeats`] repeats in a sequence of `context` tokens:
/// half of it, rounded down, at least 1.
pub fn run_length(context: usize) -> usize {
    (context / 2).max(1)
}

/// One sequence for a transformer to learn from: a random run of L = `run_length(
/// context)` tokens followed by the same run again (and, for an odd `context`, its
/// first token once more), so that from position L on, each next token can be predicted
/// by finding the earlier copy of the present token and reading what followed it (what
/// an induction head does); before that, it cannot. The run's tokens are all different
/// when the vocabulary allows (`vocab` ≥ L), and drawn independently otherwise.
///
/// Returns (tokens, `context` of them; targets, the next token at each position and
/// [`NO_TARGET`] at the last), the shapes [`Model::loss_transformer`] takes. The second
/// copy is positions L to `context` − 2 (their targets are predictable); to score only
/// it, set the targets before L to `NO_TARGET`. (Sizes are taken as at least 1.)
pub fn repeats(context: usize, vocab: usize, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
    let (context, vocab) = (context.max(1), vocab.max(1));
    let run = run_length(context);
    let chosen: Vec<usize> = if vocab >= run {
        // the first `run` entries of a random shuffle of the vocabulary
        let mut all: Vec<usize> = (0..vocab).collect();
        for i in 0..run {
            let j = i + rng.below(vocab - i);
            all.swap(i, j);
        }
        all.truncate(run);
        all
    } else {
        (0..run).map(|_| rng.below(vocab)).collect()
    };
    let tokens: Vec<usize> = (0..context).map(|t| chosen[t % run]).collect();
    let targets = (0..context).map(|t| tokens.get(t + 1).copied().unwrap_or(NO_TARGET)).collect();
    (tokens, targets)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::nn;

    fn spec(act: Act, norm: Norm, place: Place, residual: bool) -> MlpSpec {
        MlpSpec { inputs: 3, width: 5, depth: 3, outputs: 3, act, norm, place, residual, residual_scale: 0.7, init: Init::Xavier, gain: 1.3, bias: true }
    }

    fn tspec(act: Act, norm: Norm, place: Place, residual: bool, tied: bool) -> TransformerSpec {
        TransformerSpec { vocab: 7, context: 6, d_model: 8, heads: 2, layers: 2, d_mlp: 12, act, norm, place, residual, init_gain: 1.0, tied_unembed: tied }
    }

    /// Give every parameter a random value (so that no bias or gain sits at its start,
    /// where some gradients vanish by symmetry).
    fn jitter(model: &mut Model, rng: &mut Rng) {
        for g in model.groups_mut() {
            for v in g.value.iter_mut() {
                *v += 0.3 * rng.normal() as f64;
            }
        }
    }

    /// The loss a model gives (a fresh pass).
    type Pass<'a> = &'a dyn Fn(&Model, &mut Tape) -> Trace;

    /// The model's gradient, from the tape and `accumulate`, against central differences
    /// of the loss for a handful of entries of every group (the first, the last and two
    /// at random), to a relative 1e-5 (with a floor of 1e-8 of the largest gradient).
    fn check_model(name: &str, model: &Model, pass: Pass, rng: &mut Rng) {
        let mut tape = Tape::new();
        let trace = pass(model, &mut tape);
        tape.backward(trace.loss);
        let mut m = model.clone();
        m.zero_grads();
        m.accumulate(&tape, &trace);
        let largest = m.groups().iter().flat_map(|g| &g.grad).fold(0.0f64, |a, b| a.max(b.abs()));
        let loss_of = |m: &Model| {
            let mut tape = Tape::new();
            let trace = pass(m, &mut tape);
            tape.value(trace.loss)[0]
        };
        for gi in 0..m.groups().len() {
            let n = m.groups()[gi].value.len();
            for e in [0, n - 1, rng.below(n), rng.below(n)] {
                let mut moved = model.clone();
                let x = moved.groups()[gi].value[e];
                let h = 1e-5 * (1.0 + x.abs());
                moved.groups_mut()[gi].value[e] = x + h;
                let up = loss_of(&moved);
                moved.groups_mut()[gi].value[e] = x - h;
                let down = loss_of(&moved);
                let fd = (up - down) / (2.0 * h);
                let g = m.groups()[gi].grad[e];
                assert!(
                    (fd - g).abs() <= 1e-5 * g.abs().max(fd.abs()) + 1e-8 * largest.max(1e-3),
                    "{name}: {} entry {e}: tape {g}, differences {fd}",
                    m.groups()[gi].name
                );
            }
        }
    }

    #[test]
    fn every_mlp_has_the_gradient_of_its_finite_differences() {
        let mut rng = Rng::new(11);
        let x: Vec<f64> = (0..6 * 3).map(|_| rng.normal() as f64).collect();
        let classes = [0.0, 2.0, 1.0, 1.0, 0.0, 2.0];
        let values: Vec<f64> = (0..6 * 3).map(|_| rng.normal() as f64).collect();
        for act in [Act::Tanh, Act::Relu, Act::Gelu] {
            for norm in Norm::ALL {
                for place in Place::ALL {
                    for residual in [false, true] {
                        let mut model = Model::mlp(&spec(act, norm, place, residual), &mut rng);
                        jitter(&mut model, &mut rng);
                        let name = format!("{} {} {} {}", act.name(), norm.name(), place.name(), if residual { "residual" } else { "plain" });
                        check_model(&format!("{name}, cross-entropy"), &model, &|m, t| m.loss_mlp(t, &x, &classes, Loss::CrossEntropy), &mut rng);
                        check_model(&format!("{name}, squared error"), &model, &|m, t| m.loss_mlp(t, &x, &values, Loss::Mse), &mut rng);
                    }
                }
            }
        }
    }

    #[test]
    fn every_transformer_has_the_gradient_of_its_finite_differences() {
        let mut rng = Rng::new(12);
        let tokens = [3usize, 1, 4, 1, 5, 6];
        let targets = [1usize, 4, 1, NO_TARGET, 6, 2];
        for (act, tied) in [(Act::Gelu, false), (Act::Relu, true)] {
            for norm in Norm::ALL {
                for place in Place::ALL {
                    for residual in [false, true] {
                        let mut model = Model::transformer(&tspec(act, norm, place, residual, tied), &mut rng);
                        jitter(&mut model, &mut rng);
                        let name = format!("{} {} {} {}", act.name(), norm.name(), place.name(), if residual { "residual" } else { "plain" });
                        check_model(&name, &model, &|m, t| m.loss_transformer(t, &tokens, &targets), &mut rng);
                    }
                }
            }
        }
    }

    /// A plain MLP and `nn::Mlp`, the site's hand-written network, given the same
    /// weights and data, have the same gradient to f32 accuracy (that network computes
    /// in f32; ReLU is the activation both compute exactly, the others there being
    /// fast approximations).
    #[test]
    fn a_plain_mlp_has_the_gradient_of_the_hand_written_network() {
        let mut rng = Rng::new(13);
        // (one width for every hidden layer: the spec has only one)
        let sizes = [3usize, 6, 6, 4];
        let spec = MlpSpec { inputs: 3, width: 6, depth: 2, outputs: 4, act: Act::Relu, ..MlpSpec::default() };
        for loss in [Loss::CrossEntropy, Loss::Mse] {
            // (its biases start random, so that they are tested too)
            let mut theirs = nn::Mlp::new(&sizes, nn::Act::Relu, &mut rng);
            let mut ours = Model::mlp(&spec, &mut rng);
            let order = ["layer 1 weights", "layer 1 bias", "layer 2 weights", "layer 2 bias", "read-out weights", "read-out bias"];
            for (l, pair) in order.chunks(2).enumerate() {
                ours.group_mut(pair[0]).unwrap().value = theirs.w[l].iter().map(|&v| v as f64).collect();
                ours.group_mut(pair[1]).unwrap().value = theirs.b[l].iter().map(|&v| v as f64).collect();
            }
            let n = 7;
            let x32: Vec<f32> = (0..n * 3).map(|_| rng.normal()).collect();
            let y32: Vec<f32> = match loss {
                Loss::CrossEntropy => (0..n).map(|i| (i % 4) as f32).collect(),
                Loss::Mse => (0..n * 4).map(|_| rng.normal()).collect(),
            };
            // nn::Mlp keeps outputs one column per neuron; our targets are row-major
            let targets: Vec<f64> = match loss {
                Loss::CrossEntropy => y32.iter().map(|&v| v as f64).collect(),
                Loss::Mse => (0..n * 4).map(|k| y32[(k % 4) * n + k / 4] as f64).collect(),
            };
            theirs.forward(&x32, n);
            let their_loss = theirs.backward(&y32, if loss == Loss::Mse { nn::Loss::Mse } else { nn::Loss::Softmax });
            let their_grad = theirs.grad();
            let x: Vec<f64> = x32.iter().map(|&v| v as f64).collect();
            let mut tape = Tape::new();
            let trace = ours.loss_mlp(&mut tape, &x, &targets, loss);
            tape.backward(trace.loss);
            ours.accumulate(&tape, &trace);
            assert!((tape.value(trace.loss)[0] - their_loss as f64).abs() < 1e-5 * (1.0 + their_loss.abs() as f64), "{loss:?} loss");
            let our_grad: Vec<f64> = order.iter().flat_map(|name| ours.group(name).unwrap().grad.clone()).collect();
            assert_eq!(our_grad.len(), their_grad.len());
            let largest = our_grad.iter().fold(0.0f64, |a, b| a.max(b.abs()));
            for (k, (a, &b)) in our_grad.iter().zip(&their_grad).enumerate() {
                assert!((a - b as f64).abs() <= 1e-5 * a.abs() + 1e-6 * largest, "{loss:?} entry {k}: ours {a}, theirs {b}");
            }
        }
    }

    // ----- what the layers compute, by hand ---------------------------------------------

    /// x Wᵀ + b for x of `n` rows.
    fn by_hand_linear(x: &[f64], w: &Group, b: Option<&Group>) -> Vec<f64> {
        let (rows, cols) = (w.rows, w.cols);
        x.chunks(cols)
            .flat_map(|x| (0..rows).map(move |j| (0..cols).map(|p| x[p] * w.value[j * cols + p]).sum::<f64>() + b.map_or(0.0, |b| b.value[j])))
            .collect()
    }

    /// norm(x) with gain and bias, x being `n` rows of `cols`.
    fn by_hand_norm(norm: Norm, x: &[f64], cols: usize, gain: &Group, shift: Option<&Group>) -> Vec<f64> {
        let n = x.len() / cols;
        let mut y = x.to_vec();
        let mut standardise = |idx: Vec<usize>, centre: bool| {
            let m = if centre { idx.iter().map(|&k| x[k]).sum::<f64>() / idx.len() as f64 } else { 0.0 };
            let v = idx.iter().map(|&k| (x[k] - m).powi(2)).sum::<f64>() / idx.len() as f64;
            for &k in &idx {
                y[k] = (x[k] - m) / (v + EPS).sqrt();
            }
        };
        match norm {
            Norm::None => return y,
            Norm::Layer | Norm::Rms => (0..n).for_each(|i| standardise((0..cols).map(|j| i * cols + j).collect(), norm == Norm::Layer)),
            Norm::Batch => (0..cols).for_each(|j| standardise((0..n).map(|i| i * cols + j).collect(), true)),
        }
        for (k, y) in y.iter_mut().enumerate() {
            *y = *y * gain.value[k % cols] + shift.map_or(0.0, |s| s.value[k % cols]);
        }
        y
    }

    fn assert_close(a: &[f64], b: &[f64], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: lengths");
        for (k, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() <= 1e-12 * (1.0 + y.abs()), "{what}: entry {k}: {x} against {y}");
        }
    }

    /// Every hidden layer of every arrangement computes exactly the formula of
    /// `MlpSpec`'s documentation, stage by stage as `Trace`'s table says, worked out
    /// here by hand from the layer's input and its parameters.
    #[test]
    fn the_layers_compute_what_the_docs_say() {
        let mut rng = Rng::new(14);
        let x: Vec<f64> = (0..5 * 3).map(|_| rng.normal() as f64).collect();
        for norm in Norm::ALL {
            for place in Place::ALL {
                for residual in [false, true] {
                    let spec = spec(Act::Gelu, norm, place, residual);
                    let mut model = Model::mlp(&spec, &mut rng);
                    jitter(&mut model, &mut rng);
                    let mut tape = Tape::new();
                    let tr = model.forward_mlp(&mut tape, &x);
                    let val = |stage: Stage, l: usize| tape.value(tr.get(stage, l).unwrap_or_else(|| panic!("{stage:?} missing at {l}"))).to_vec();
                    let g = |name: String| model.group(&name);
                    let what = format!("{} {} {}", norm.name(), place.name(), residual);
                    if residual {
                        let mapped = by_hand_linear(&x, g("input map weights".into()).unwrap(), g("input map bias".into()));
                        assert_close(&val(Stage::Output, 0), &mapped, &what);
                    }
                    for l in 1..=spec.depth {
                        let input = val(Stage::Input, l);
                        let (w, b) = (g(format!("layer {l} weights")).unwrap(), g(format!("layer {l} bias")));
                        let gain = g(format!("layer {l} norm gain"));
                        let shift = g(format!("layer {l} norm bias"));
                        let normalise = |v: &[f64]| match gain {
                            Some(gain) => by_hand_norm(norm, v, spec.width, gain, shift),
                            None => v.to_vec(),
                        };
                        let act = |v: &[f64]| v.iter().map(|&z| spec.act.f(z)).collect::<Vec<f64>>();
                        let first_plain_pre = l == 1 && !residual && place == Place::Pre;
                        assert_eq!(gain.is_some(), norm != Norm::None && !first_plain_pre, "{what}: layer {l} normalisation");
                        assert_eq!(tr.get(Stage::Normed, l).is_some(), gain.is_some());
                        let s = spec.residual_scale;
                        let out = match (residual, place) {
                            (false, Place::Post) => act(&normalise(&by_hand_linear(&input, w, b))),
                            (false, Place::Pre) => act(&by_hand_linear(&normalise(&input), w, b)),
                            (true, Place::Pre) => {
                                let a = act(&by_hand_linear(&normalise(&input), w, b));
                                input.iter().zip(&a).map(|(x, a)| x + s * a).collect()
                            }
                            (true, Place::Post) => {
                                let a = act(&by_hand_linear(&input, w, b));
                                let sum: Vec<f64> = input.iter().zip(&a).map(|(x, a)| x + s * a).collect();
                                assert_close(&val(Stage::Sum, l), &sum, &what);
                                normalise(&sum)
                            }
                        };
                        assert_close(&val(Stage::Output, l), &out, &format!("{what}, layer {l}"));
                        assert_close(&val(Stage::PostAct, l), &act(&val(Stage::PreAct, l)), &what);
                        // every layer has the five stages, and the next layer's input is this one's output
                        for stage in [Stage::Input, Stage::Linear, Stage::PreAct, Stage::PostAct, Stage::Output] {
                            assert!(tr.get(stage, l).is_some(), "{what}: {stage:?} at {l}");
                        }
                        assert_eq!(tr.get(Stage::Output, l), tr.get(Stage::Input, l + 1));
                    }
                    let top = spec.depth + 1;
                    let readout = by_hand_linear(&val(Stage::Input, top), g("read-out weights".into()).unwrap(), g("read-out bias".into()));
                    assert_close(&val(Stage::Output, top), &readout, &what);
                    assert_eq!(Some(tr.output), tr.get(Stage::Output, top));
                    assert!(tape.value(tr.loss)[0].is_nan());
                }
            }
        }
    }

    /// A transformer computes what `TransformerSpec`'s documentation says: the heads'
    /// scores, causal patterns and outputs, attention as their sum, the two sums of a
    /// block (pre and post, with and without skips), and the logits.
    #[test]
    fn a_transformer_computes_what_the_docs_say() {
        let mut rng = Rng::new(15);
        let tokens = [2usize, 6, 0, 6, 3];
        for norm in Norm::ALL {
            for place in Place::ALL {
                for residual in [false, true] {
                    for tied in [false, true] {
                        let spec = tspec(Act::Silu, norm, place, residual, tied);
                        let mut model = Model::transformer(&spec, &mut rng);
                        jitter(&mut model, &mut rng);
                        let mut tape = Tape::new();
                        let tr = model.forward_transformer(&mut tape, &tokens);
                        let what = format!("{} {} {} {}", norm.name(), place.name(), residual, tied);
                        let val = |stage: Stage, l: usize| tape.value(tr.get(stage, l).unwrap_or_else(|| panic!("{what}: {stage:?} missing at {l}"))).to_vec();
                        let g = |name: &str| model.group(name).unwrap_or_else(|| panic!("no group {name}"));
                        let (t, d, dh) = (tokens.len(), spec.d_model, spec.d_head());
                        let embed: Vec<f64> =
                            (0..t * d).map(|k| g("token embedding").value[tokens[k / d] * d + k % d] + g("position embedding").value[k]).collect();
                        assert_close(&val(Stage::Stream, 0), &embed, &what);
                        let add = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<f64>>();
                        for b in 1..=spec.layers {
                            let x = val(Stage::Stream, b - 1);
                            let normalise = |v: &[f64], which: &str| match model.group(&format!("block {b} {which} norm gain")) {
                                Some(gain) => by_hand_norm(norm, v, d, gain, model.group(&format!("block {b} {which} norm bias"))),
                                None => v.to_vec(),
                            };
                            let pre = place == Place::Pre;
                            let h = if pre { normalise(&x, "attention") } else { x.clone() };
                            let mut attn = vec![0.0; t * d];
                            for head in 1..=spec.heads {
                                let m = |what: &str| g(&format!("block {b} head {head} {what}"));
                                let (q, k, v) =
                                    (by_hand_linear(&h, m("query"), None), by_hand_linear(&h, m("key"), None), by_hand_linear(&h, m("value"), None));
                                let scores: Vec<f64> =
                                    (0..t * t).map(|e| (0..dh).map(|c| q[e / t * dh + c] * k[e % t * dh + c]).sum::<f64>() / (dh as f64).sqrt()).collect();
                                assert_close(tape.value(tr.head(Stage::Scores, b, head).unwrap()), &scores, &what);
                                let pattern = tape.value(tr.head(Stage::Pattern, b, head).unwrap());
                                for i in 0..t {
                                    let row = &pattern[i * t..(i + 1) * t];
                                    assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12 && row[i + 1..].iter().all(|&p| p == 0.0));
                                    let top = scores[i * t..i * t + i + 1].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                                    let total: f64 = scores[i * t..i * t + i + 1].iter().map(|s| (s - top).exp()).sum();
                                    for j in 0..=i {
                                        assert!((row[j] - (scores[i * t + j] - top).exp() / total).abs() < 1e-12);
                                    }
                                }
                                let mix: Vec<f64> = (0..t * dh).map(|e| (0..t).map(|j| pattern[e / dh * t + j] * v[j * dh + e % dh]).sum()).collect();
                                let out = by_hand_linear(&mix, m("output"), None);
                                assert_close(tape.value(tr.head(Stage::HeadOut, b, head).unwrap()), &out, &what);
                                attn = add(&attn, &out);
                            }
                            assert_close(&val(Stage::Attn, b), &attn, &what);
                            let sum = if residual { add(&x, &attn) } else { attn };
                            let mid = if pre { sum } else { normalise(&sum, "attention") };
                            assert_close(&val(Stage::Mid, b), &mid, &what);
                            let h = if pre { normalise(&mid, "MLP") } else { mid.clone() };
                            let z = by_hand_linear(&h, g(&format!("block {b} MLP in weights")), Some(g(&format!("block {b} MLP in bias"))));
                            let a: Vec<f64> = z.iter().map(|&z| spec.act.f(z)).collect();
                            let y = by_hand_linear(&a, g(&format!("block {b} MLP out weights")), Some(g(&format!("block {b} MLP out bias"))));
                            assert_close(&val(Stage::Mlp, b), &y, &what);
                            let sum = if residual { add(&mid, &y) } else { y };
                            let out = if pre { sum } else { normalise(&sum, "MLP") };
                            assert_close(&val(Stage::Stream, b), &out, &what);
                        }
                        let x = val(Stage::Stream, spec.layers);
                        let x = match model.group("final norm gain") {
                            Some(gain) => by_hand_norm(norm, &x, d, gain, model.group("final norm bias")),
                            None => x,
                        };
                        assert_eq!(model.group("final norm gain").is_some(), place == Place::Pre && norm != Norm::None);
                        let unembed = if tied { g("token embedding") } else { g("unembedding") };
                        assert_close(&val(Stage::Logits, spec.layers + 1), &by_hand_linear(&x, unembed, None), &what);
                        assert_eq!(tape.shape(tr.output), (t, spec.vocab));
                    }
                }
            }
        }
    }

    /// The names, kinds and numbers of the groups are those the constructors' docs list.
    #[test]
    fn groups_are_named_and_numbered_as_documented() {
        let mut rng = Rng::new(16);
        let m = Model::mlp(&MlpSpec { depth: 2, residual: true, norm: Norm::Layer, ..MlpSpec::default() }, &mut rng);
        let names: Vec<&str> = m.groups().iter().map(|g| g.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "input map weights",
                "input map bias",
                "layer 1 weights",
                "layer 1 bias",
                "layer 1 norm gain",
                "layer 1 norm bias",
                "layer 2 weights",
                "layer 2 bias",
                "layer 2 norm gain",
                "layer 2 norm bias",
                "read-out weights",
                "read-out bias"
            ]
        );
        let layers: Vec<usize> = m.groups().iter().map(|g| g.layer).collect();
        assert_eq!(layers, [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3]);
        assert_eq!((m.group("layer 2 weights").unwrap().rows, m.group("layer 2 weights").unwrap().cols), (32, 32));
        assert_eq!(m.n_params(), 2 * 32 + 32 + 2 * (32 * 32 + 3 * 32) + 3 * 32 + 3);
        let t = Model::transformer(&TransformerSpec { layers: 2, heads: 2, norm: Norm::Rms, ..TransformerSpec::default() }, &mut rng);
        let q = t.group("block 2 head 1 query").unwrap();
        assert_eq!((q.kind, q.part, q.layer, q.head, q.rows, q.cols), (Kind::Query, Part::Attention, 2, 1, 16, 32));
        let o = t.group("block 1 head 2 output").unwrap();
        assert_eq!((o.kind, o.head, o.rows, o.cols), (Kind::HeadOut, 2, 32, 16));
        assert!(t.group("block 1 attention norm gain").is_some() && t.group("block 1 attention norm bias").is_none());
        let u = t.group("unembedding").unwrap();
        assert_eq!((u.kind, u.part, u.layer), (Kind::Unembedding, Part::Top, 3));
        assert_eq!(t.groups().first().unwrap().name, "token embedding");
        let tied = Model::transformer(&TransformerSpec { tied_unembed: true, ..TransformerSpec::default() }, &mut rng);
        assert!(tied.group("unembedding").is_none());
        // labels of stages
        let mut tape = Tape::new();
        let tr = t.forward_transformer(&mut tape, &[1, 2, 3]);
        let label = |stage, layer, head| tr.entries.iter().find(|e| e.stage == stage && e.layer == layer && e.head == head).unwrap().label();
        assert_eq!(label(Stage::Pattern, 2, 1), "block 2 head 1 pattern");
        assert_eq!(label(Stage::Stream, 0, 0), "stream after the embeddings");
        assert_eq!(label(Stage::Stream, 2, 0), "block 2 stream");
        let tr = m.forward_mlp(&mut tape, &[0.1, 0.2]);
        assert_eq!(tr.entry(Stage::PreAct, 2).unwrap().label(), "layer 2 pre-activation");
        assert_eq!(tr.entry(Stage::Input, 0).unwrap().label(), "data");
        assert_eq!(tr.all(Stage::Output).count(), 4);
        assert_eq!(tr.of_layer(3).count(), 2);
    }

    /// The starting weights have the variances `Init` promises (each times gain²), and
    /// orthogonal ones are orthogonal: Wᵀ W = gain² I for a square layer, orthonormal
    /// columns or rows times gain otherwise. The transformer's are as its spec says.
    #[test]
    fn the_starting_weights_have_the_promised_variances() {
        let mut rng = Rng::new(17);
        for init in [Init::Lecun, Init::Xavier, Init::He] {
            for gain in [1.0, 1.7] {
                let spec = MlpSpec { inputs: 48, width: 64, depth: 3, outputs: 32, init, gain, ..MlpSpec::default() };
                let m = Model::mlp(&spec, &mut rng);
                for g in m.groups().iter().filter(|g| g.kind == Kind::Weight) {
                    let n = g.value.len() as f64;
                    let mean = g.value.iter().sum::<f64>() / n;
                    let var = g.value.iter().map(|v| v * v).sum::<f64>() / n;
                    let want = gain * gain * init.variance(g.cols, g.rows);
                    // (the standard error of a variance from n draws is var √(2/n): 3 % here)
                    assert!((var / want - 1.0).abs() < 0.15, "{} {gain} {}: variance {var}, promised {want}", init.name(), g.name);
                    assert!(mean.abs() < 4.0 * (want / n).sqrt());
                }
                assert!(m.groups().iter().filter(|g| g.kind == Kind::Bias).all(|g| g.value.iter().all(|&b| b == 0.0)));
            }
        }
        assert_eq!((Init::Lecun.variance(10, 30), Init::Xavier.variance(10, 30), Init::He.variance(10, 30)), (0.1, 0.05, 0.2));
        let gain = 1.3;
        let spec = MlpSpec { inputs: 48, width: 64, depth: 2, outputs: 32, init: Init::Orthogonal, gain, ..MlpSpec::default() };
        let m = Model::mlp(&spec, &mut rng);
        for g in m.groups().iter().filter(|g| g.kind == Kind::Weight) {
            let (r, c) = (g.rows, g.cols);
            // the Gram matrix of the short side: columns of a tall matrix, rows of a wide one
            let k = r.min(c);
            for a in 0..k {
                for b in 0..k {
                    let dot: f64 = if r >= c {
                        (0..r).map(|i| g.value[i * c + a] * g.value[i * c + b]).sum()
                    } else {
                        (0..c).map(|j| g.value[a * c + j] * g.value[b * c + j]).sum()
                    };
                    let want = if a == b { gain * gain } else { 0.0 };
                    assert!((dot - want).abs() < 1e-12, "{}: ({a}, {b}) = {dot}", g.name);
                }
            }
        }
        // the transformer: embeddings of variance g²/d, every matrix g² / fan-in, the
        // heads' outputs counting the heads together
        let g = 0.8;
        let spec = TransformerSpec { vocab: 60, context: 40, d_model: 64, heads: 4, layers: 1, d_mlp: 128, init_gain: g, ..TransformerSpec::default() };
        let t = Model::transformer(&spec, &mut rng);
        let variance = |name: &str| {
            let v = &t.group(name).unwrap().value;
            v.iter().map(|x| x * x).sum::<f64>() / v.len() as f64
        };
        for (name, fan_in) in [
            ("token embedding", 64),
            ("position embedding", 64),
            ("block 1 head 1 query", 64),
            ("block 1 head 2 output", 64),
            ("block 1 MLP in weights", 64),
            ("block 1 MLP out weights", 128),
            ("unembedding", 64),
        ] {
            let want = g * g / fan_in as f64;
            assert!((variance(name) / want - 1.0).abs() < 0.2, "{name}: {} against {want}", variance(name));
        }
    }

    /// One full-batch Adam step on an MLP; returns the loss before it.
    fn train_step(model: &mut Model, tape: &mut Tape, x: &[f64], y: &[f64], loss: Loss, lr: f64) -> f64 {
        let trace = model.loss_mlp(tape, x, y, loss);
        tape.backward(trace.loss);
        model.zero_grads();
        model.accumulate(tape, &trace);
        model.step_adam(lr);
        tape.value(trace.loss)[0]
    }

    /// Adam drives the loss on the spirals to under half its starting value in a few
    /// hundred steps, for a plain tanh network and a residual, normalised ReLU one;
    /// and the teacher's regression too.
    #[test]
    fn adam_learns_the_spirals_and_the_teacher() {
        let mut rng = Rng::new(18);
        let (x, y) = spirals(240, 3, &mut rng);
        let designs = [
            MlpSpec { inputs: 2, width: 32, depth: 3, outputs: 3, act: Act::Tanh, init: Init::Xavier, ..MlpSpec::default() },
            MlpSpec { inputs: 2, width: 32, depth: 6, outputs: 3, act: Act::Relu, norm: Norm::Layer, residual: true, ..MlpSpec::default() },
        ];
        for spec in designs {
            let mut model = Model::mlp(&spec, &mut Rng::new(1));
            let mut tape = Tape::new();
            let first = train_step(&mut model, &mut tape, &x, &y, Loss::CrossEntropy, 0.01);
            let mut last = first;
            for _ in 0..300 {
                last = train_step(&mut model, &mut tape, &x, &y, Loss::CrossEntropy, 0.01);
            }
            assert!(last < 0.5 * first, "{} {}: loss {first} to {last}", spec.act.name(), spec.norm.name());
            // and it classifies most points right
            let trace = model.forward_mlp(&mut tape, &x);
            let out = tape.value(trace.output);
            let right = out.chunks(3).zip(&y).filter(|(o, &c)| (0..3).all(|k| o[k] <= o[c as usize])).count();
            assert!(right as f64 > 0.8 * y.len() as f64, "{right} of {} right", y.len());
        }
        let (x, y) = teacher(200, 4, &mut rng);
        let mut model =
            Model::mlp(&MlpSpec { inputs: 4, width: 32, depth: 2, outputs: 1, act: Act::Tanh, init: Init::Xavier, ..MlpSpec::default() }, &mut Rng::new(2));
        let mut tape = Tape::new();
        let first = train_step(&mut model, &mut tape, &x, &y, Loss::Mse, 0.01);
        let mut last = first;
        for _ in 0..300 {
            last = train_step(&mut model, &mut tape, &x, &y, Loss::Mse, 0.01);
        }
        assert!(last < 0.1 * first, "teacher: loss {first} to {last}");
    }

    /// The loss on the second copy of a run (positions the first copy makes predictable).
    fn second_copy_loss(model: &Model, tape: &mut Tape, rng: &mut Rng, sequences: usize) -> f64 {
        let spec = model.transformer_spec().unwrap();
        let run = run_length(spec.context);
        let mut total = 0.0;
        for _ in 0..sequences {
            let (tokens, mut targets) = repeats(spec.context, spec.vocab, rng);
            targets[..run].fill(NO_TARGET);
            let trace = model.loss_transformer(tape, &tokens, &targets);
            total += tape.value(trace.loss)[0];
        }
        total / sequences as f64
    }

    /// A two-block transformer trained with Adam on repeated runs learns to predict the
    /// second copy from the first: its loss there falls from about ln(vocab) to well
    /// under it.
    #[test]
    fn adam_teaches_a_transformer_the_repeats() {
        let spec = TransformerSpec::default();
        let mut model = Model::transformer(&spec, &mut Rng::new(3));
        let (mut tape, mut rng, mut test) = (Tape::new(), Rng::new(4), Rng::new(5));
        let before = second_copy_loss(&model, &mut tape, &mut test, 32);
        let chance = (spec.vocab as f64).ln();
        assert!((before / chance - 1.0).abs() < 0.3, "before training: {before} against ln(vocab) {chance}");
        for _ in 0..600 {
            model.zero_grads();
            for _ in 0..8 {
                let (tokens, targets) = repeats(spec.context, spec.vocab, &mut rng);
                let trace = model.loss_transformer(&mut tape, &tokens, &targets);
                tape.backward(trace.loss);
                model.accumulate(&tape, &trace);
            }
            model.clip(1.0);
            model.step_adam(3e-3);
        }
        let after = second_copy_loss(&model, &mut tape, &mut test, 64);
        assert!(after < 0.2 * chance, "after training: {after} against ln(vocab) {chance}");
    }

    /// Every number of the probe, against the same quantity worked out by hand from the
    /// tape; and a unit that is shut for every example counts as dead.
    #[test]
    fn the_probe_measures_what_it_says() {
        let mut rng = Rng::new(19);
        let spec = MlpSpec { inputs: 2, width: 6, depth: 3, outputs: 3, act: Act::Relu, norm: Norm::Layer, place: Place::Post, ..MlpSpec::default() };
        let mut model = Model::mlp(&spec, &mut rng);
        // unit 0 of layer 2 shut for every example: its normalised input pushed far down
        model.group_mut("layer 2 norm bias").unwrap().value[0] = -50.0;
        let (x, y) = spirals(9, 3, &mut rng);
        let mut tape = Tape::new();
        let trace = model.loss_mlp(&mut tape, &x, &y, Loss::CrossEntropy);
        tape.backward(trace.loss);
        let probe = Probe::new(&tape, &trace);
        assert_eq!((probe.entries.len(), probe.groups.len()), (trace.entries.len(), model.groups().len()));
        for (e, p) in trace.entries.iter().zip(&probe.entries) {
            let (v, g) = (tape.value(e.var), tape.grad(e.var));
            let n = v.len() as f64;
            assert!((p.rms - (v.iter().map(|a| a * a).sum::<f64>() / n).sqrt()).abs() < 1e-14);
            assert!((p.mean - v.iter().sum::<f64>() / n).abs() < 1e-14);
            assert!((p.grad_rms - (g.iter().map(|a| a * a).sum::<f64>() / n).sqrt()).abs() < 1e-14);
            let gated = matches!(e.stage, Stage::PreAct | Stage::PostAct);
            assert_eq!(p.shut.is_some(), gated, "{}", e.label());
            if let Some((act, input)) = e.gate {
                let (z, cols) = (tape.value(input), tape.cols(input));
                let shut = z.iter().filter(|&&z| act.df(z).abs() < 0.01).count() as f64 / z.len() as f64;
                let dead = (0..cols).filter(|&j| z.iter().skip(j).step_by(cols).all(|&z| act.df(z).abs() < 0.01)).count() as f64 / cols as f64;
                assert_eq!((p.shut, p.dead), (Some(shut), Some(dead)));
                // (for ReLU, shut is "not above 0")
                assert_eq!(shut, z.iter().filter(|&&z| z <= 0.0).count() as f64 / z.len() as f64);
            }
        }
        for (g, (&v, p)) in model.groups().iter().zip(trace.params.iter().zip(&probe.groups)) {
            let length = |x: &[f64]| x.iter().map(|a| a * a).sum::<f64>().sqrt();
            assert!((p.value_norm - length(&g.value)).abs() < 1e-14 && (p.grad_norm - length(tape.grad(v))).abs() < 1e-14);
            assert_eq!(p.relative_step(0.1), 0.1 * p.grad_norm / p.value_norm);
        }
        let dead = probe.of(&trace, Stage::PostAct, 2).unwrap().dead.unwrap();
        assert!(dead >= 1.0 / 6.0, "dead share {dead}");
        // without a normalisation in the way, a dead unit's weights and bias get nothing
        let plain = MlpSpec { norm: Norm::None, ..spec };
        let mut model2 = Model::mlp(&plain, &mut rng);
        model2.group_mut("layer 2 bias").unwrap().value[0] = -50.0;
        let trace2 = model2.loss_mlp(&mut tape, &x, &y, Loss::CrossEntropy);
        tape.backward(trace2.loss);
        assert!(Probe::new(&tape, &trace2).of(&trace2, Stage::PreAct, 2).unwrap().dead.unwrap() >= 1.0 / 6.0);
        let (w, b) = (trace2.params[model2.index_of("layer 2 weights").unwrap()], trace2.params[model2.index_of("layer 2 bias").unwrap()]);
        assert!(tape.grad(w)[..6].iter().all(|&g| g == 0.0) && tape.grad(b)[0] == 0.0);
        assert!(tape.grad(w)[6..].iter().any(|&g| g != 0.0));
        // and the share right, from the outputs
        let trace = model.forward_mlp(&mut tape, &x);
        let out = tape.value(trace.output);
        let right = out.chunks(3).zip(&y).filter(|(o, &c)| (0..3).all(|k| k == c as usize || o[k] < o[c as usize])).count();
        assert_eq!(share_right(out, 3, &y), right as f64 / 9.0);
        assert_eq!(share_right(&[1.0, f64::NAN, 0.0, 2.0], 2, &[0.0, 1.0]), 0.5);
        assert_eq!(share_right(&[], 3, &[]), 0.0);
    }

    /// The Jacobian is the derivative of the map (against finite differences), the
    /// pieces chain (the whole map's is the read-out's weights times the stack's times
    /// the input map's), and a deep linear stack of orthogonal layers has every
    /// singular value equal to gain^depth.
    #[test]
    fn the_jacobian_of_an_mlp_is_the_derivative_of_its_map() {
        let mut rng = Rng::new(20);
        let spec =
            MlpSpec { inputs: 3, width: 5, depth: 4, outputs: 2, act: Act::Tanh, norm: Norm::Layer, residual: true, residual_scale: 0.5, ..MlpSpec::default() };
        let mut model = Model::mlp(&spec, &mut rng);
        jitter(&mut model, &mut rng);
        let x = [0.3, -0.8, 1.1];
        let (whole, rows, cols) = input_output_jacobian(&model, &x, Span::Whole);
        assert_eq!((rows, cols), (2, 3));
        let output = |x: &[f64]| {
            let mut tape = Tape::new();
            let tr = model.forward_mlp(&mut tape, x);
            tape.value(tr.output).to_vec()
        };
        for l in 0..3 {
            let (mut up, mut down) = (x, x);
            up[l] += 1e-6;
            down[l] -= 1e-6;
            let (a, b) = (output(&up), output(&down));
            for k in 0..2 {
                let fd = (a[k] - b[k]) / 2e-6;
                assert!((fd - whole[k * 3 + l]).abs() < 1e-8, "({k}, {l}): {} against {fd}", whole[k * 3 + l]);
            }
        }
        let (hidden, hr, hc) = input_output_jacobian(&model, &x, Span::Hidden);
        assert_eq!((hr, hc), (5, 5));
        let product = |a: &[f64], b: &[f64], m: usize, k: usize, n: usize| -> Vec<f64> {
            (0..m * n).map(|e| (0..k).map(|p| a[e / n * k + p] * b[p * n + e % n]).sum()).collect()
        };
        let chained =
            product(&product(&model.group("read-out weights").unwrap().value, &hidden, 2, 5, 5), &model.group("input map weights").unwrap().value, 2, 5, 3);
        assert_close(&whole, &chained, "chain");
        // the stack is its two halves, one after the other
        let (top, _, _) = input_output_jacobian(&model, &x, Span::Layers(3, 4));
        let (bottom, _, _) = input_output_jacobian(&model, &x, Span::Layers(1, 2));
        assert_close(&hidden, &product(&top, &bottom, 5, 5, 5), "halves");
        // orthogonal and linear: every singular value gain^depth
        let spec = MlpSpec { inputs: 8, width: 8, depth: 6, outputs: 3, act: Act::Identity, init: Init::Orthogonal, gain: 1.1, ..MlpSpec::default() };
        let model = Model::mlp(&spec, &mut rng);
        let (jac, r, c) = input_output_jacobian(&model, &[0.5; 8], Span::Hidden);
        let (_, sv, _) = linalg::svd(&jac, r, c);
        assert!(sv.iter().all(|s| (s - 1.1f64.powi(6)).abs() < 1e-12), "{sv:?}");
        // batch normalisation: the example's map depends on the batch it is in
        let spec = MlpSpec { inputs: 2, width: 4, depth: 2, outputs: 2, norm: Norm::Batch, act: Act::Tanh, ..MlpSpec::default() };
        let model = Model::mlp(&spec, &mut rng);
        let (a, _, _) = input_output_jacobian(&model, &[0.1, 0.2, 0.5, -0.4, -0.3, 0.9], Span::Whole);
        let (b, _, _) = input_output_jacobian(&model, &[0.1, 0.2, 0.7, -0.1, -0.3, 0.9], Span::Whole);
        assert!(a.iter().zip(&b).any(|(a, b)| (a - b).abs() > 1e-6));
        assert_eq!(input_output_jacobian(&model, &[], Span::Whole).1, 0);
    }

    /// Accumulating adds, zeroing zeroes, plain descent steps by −lr times the gradient,
    /// clipping returns the length before and leaves it at the cap, and Adam's first
    /// step moves every parameter by the rate against its gradient's sign.
    #[test]
    fn the_optimisers_and_clipping_do_what_they_say() {
        let mut rng = Rng::new(21);
        let (x, y) = teacher(8, 3, &mut rng);
        let mut model = Model::mlp(&MlpSpec { inputs: 3, width: 4, depth: 2, outputs: 1, act: Act::Tanh, ..MlpSpec::default() }, &mut rng);
        let mut tape = Tape::new();
        let trace = model.loss_mlp(&mut tape, &x, &y, Loss::Mse);
        tape.backward(trace.loss);
        model.accumulate(&tape, &trace);
        let once: Vec<f64> = model.groups().iter().flat_map(|g| g.grad.clone()).collect();
        model.accumulate(&tape, &trace);
        let twice: Vec<f64> = model.groups().iter().flat_map(|g| g.grad.clone()).collect();
        assert!(once.iter().zip(&twice).all(|(a, b)| 2.0 * a == *b));
        let before: Vec<f64> = model.groups().iter().flat_map(|g| g.value.clone()).collect();
        let mut sgd = model.clone();
        sgd.step_sgd(0.1);
        let after: Vec<f64> = sgd.groups().iter().flat_map(|g| g.value.clone()).collect();
        assert!(before.iter().zip(&after).zip(&twice).all(|((p, q), g)| *q == p - 0.1 * g));
        let mut adam = model.clone();
        adam.step_adam(0.01);
        let after: Vec<f64> = adam.groups().iter().flat_map(|g| g.value.clone()).collect();
        for ((p, q), g) in before.iter().zip(&after).zip(&twice) {
            if g.abs() > 1e-6 {
                assert!((q - p + 0.01 * g.signum()).abs() < 1e-7, "Adam's first step: {p} to {q} for gradient {g}");
            }
        }
        assert_eq!(adam.adam_steps(), 1);
        // the second step, against the paper's formula written out
        let (g1, g2) = (twice[3], -0.5 * twice[3]);
        let at = |m: &Model| m.groups().iter().flat_map(|g| g.value.clone()).nth(3).unwrap();
        let p1 = at(&adam);
        adam.groups_mut().iter_mut().for_each(|g| g.grad.iter_mut().for_each(|d| *d *= -0.5));
        adam.step_adam(0.01);
        let (m, v) = (0.9 * 0.1 * g1 + 0.1 * g2, 0.999 * 0.001 * g1 * g1 + 0.001 * g2 * g2);
        let want = p1 - 0.01 * (m / (1.0 - 0.81)) / ((v / (1.0 - 0.999f64.powi(2))).sqrt() + 1e-8);
        assert!((at(&adam) - want).abs() < 1e-15, "{} against {want}", at(&adam));
        adam.reset_adam();
        assert_eq!(adam.adam_steps(), 0);
        // clipping
        let norm = model.grad_norm();
        assert!((norm - twice.iter().map(|g| g * g).sum::<f64>().sqrt()).abs() < 1e-15);
        assert_eq!(model.clip(f64::INFINITY), norm);
        assert_eq!(model.grad_norm(), norm);
        assert_eq!(model.clip(norm / 4.0), norm);
        assert!((model.grad_norm() - norm / 4.0).abs() < 1e-15 * norm);
        model.clip(0.0);
        assert_eq!(model.grad_norm(), 0.0);
        model.accumulate(&tape, &trace);
        model.zero_grads();
        assert_eq!(model.grad_norm(), 0.0);
    }

    /// Every setting a slider can reach, and data a scene could hand over, runs without
    /// a panic: sizes of 0, a gain of 0 or NaN or 1e10, a batch of none or of one, NaN
    /// data, tokens outside the vocabulary, sequences longer than the context, targets
    /// that run short.
    #[test]
    fn nothing_panics_on_any_setting() {
        let mut rng = Rng::new(22);
        let mut tape = Tape::new();
        for act in Act::ALL {
            for norm in Norm::ALL {
                for place in Place::ALL {
                    for residual in [false, true] {
                        for (gain, init, size) in [(0.0, Init::He, 0), (f64::NAN, Init::Orthogonal, 1), (1e10, Init::Lecun, 2)] {
                            let spec = MlpSpec {
                                inputs: size,
                                width: size,
                                depth: size,
                                outputs: size,
                                act,
                                norm,
                                place,
                                residual,
                                residual_scale: 0.0,
                                init,
                                gain,
                                bias: size != 1,
                            };
                            let mut model = Model::mlp(&spec, &mut rng);
                            let k = model.mlp_spec().unwrap().inputs;
                            for (n, fill) in [(0, 0.5), (1, 0.5), (3, f64::NAN), (2, f64::INFINITY)] {
                                let x = vec![fill; n * k];
                                for (loss, y) in [(Loss::CrossEntropy, vec![0.0; n]), (Loss::Mse, vec![fill; n * model.mlp_spec().unwrap().outputs])] {
                                    let trace = model.loss_mlp(&mut tape, &x, &y, loss);
                                    tape.backward(trace.loss);
                                    let _ = Probe::new(&tape, &trace);
                                    model.zero_grads();
                                    model.accumulate(&tape, &trace);
                                    model.clip(0.0);
                                    model.step_sgd(0.0);
                                    model.step_adam(f64::NAN);
                                }
                                let _ = input_output_jacobian(&model, &x, Span::Layers(0, 99));
                            }
                        }
                    }
                }
            }
        }
        for norm in Norm::ALL {
            for place in Place::ALL {
                for (vocab, context, d_model, heads) in [(1, 1, 1, 5), (3, 2, 2, 1), (4, 3, 5, 2)] {
                    let spec =
                        TransformerSpec { vocab, context, d_model, heads, layers: 0, d_mlp: 0, norm, place, init_gain: 0.0, ..TransformerSpec::default() };
                    let mut model = Model::transformer(&spec, &mut rng);
                    for (tokens, targets) in [(vec![], vec![]), (vec![0], vec![]), (vec![0, 9, 1, 0, 2], vec![1, 2]), (vec![0; 9], vec![NO_TARGET; 9])] {
                        let trace = model.loss_transformer(&mut tape, &tokens, &targets);
                        tape.backward(trace.loss);
                        let _ = Probe::new(&tape, &trace);
                        model.accumulate(&tape, &trace);
                        model.clip(1.0);
                        model.step_adam(0.1);
                    }
                }
            }
        }
        for (context, vocab) in [(0, 0), (1, 5), (7, 1), (3, 2)] {
            let (tokens, targets) = repeats(context, vocab, &mut rng);
            assert_eq!(tokens.len(), context.max(1));
            assert!(tokens.iter().all(|&t| t < vocab.max(1)) && targets.last() == Some(&NO_TARGET));
        }
        assert_eq!(spirals(0, 0, &mut rng).0.len(), 2);
        assert_eq!(teacher(0, 0, &mut rng).0.len(), 1);
    }

    /// Asking an MLP for a transformer's pass (or the reverse) is a mistake; in a
    /// release build it gives a trace with no stages and a NaN loss, not a panic.
    #[test]
    fn the_wrong_kind_of_pass_is_not_a_panic() {
        if cfg!(debug_assertions) {
            return;
        }
        let mut rng = Rng::new(23);
        let (mlp, tf) = (Model::mlp(&MlpSpec::default(), &mut rng), Model::transformer(&TransformerSpec::default(), &mut rng));
        let mut tape = Tape::new();
        for trace in [mlp.loss_transformer(&mut tape, &[1, 2], &[2, 3]), tf.loss_mlp(&mut tape, &[0.0, 1.0], &[1.0], Loss::CrossEntropy)] {
            assert!(trace.entries.is_empty() && tape.value(trace.loss)[0].is_nan());
        }
        assert_eq!(input_output_jacobian(&tf, &[0.0, 1.0], Span::Whole), (Vec::new(), 0, 0));
        // and inputs that are not a whole number of examples: the whole ones are used
        let trace = mlp.forward_mlp(&mut tape, &[0.1, 0.2, 0.3]);
        assert_eq!(trace.rows, 1);
    }

    /// The toy data have the shapes and properties their docs promise.
    #[test]
    fn the_toy_data_are_as_described() {
        let (x, y) = spirals(301, 3, &mut Rng::new(24));
        assert_eq!((x.len(), y.len()), (602, 301));
        assert_eq!((x.clone(), y.clone()), spirals(301, 3, &mut Rng::new(24)));
        for (i, (p, &c)) in x.chunks(2).zip(&y).enumerate() {
            let r = (p[0] * p[0] + p[1] * p[1]).sqrt();
            assert!((0.1..=1.0).contains(&r) && c == (i % 3) as f64);
        }
        assert!(spirals(10, 7, &mut Rng::new(1)).1.iter().all(|&c| c < 3.0));
        assert!(spirals(10, 1, &mut Rng::new(1)).1.contains(&1.0));
        // the teacher: the same fixed function whatever the generator
        let (x, y) = teacher(500, 3, &mut Rng::new(25));
        let (w1, b1, w2) = teacher_net(3);
        for (x, &y) in x.chunks(3).zip(&y) {
            let want: f64 = (0..16).map(|j| w2[j] * (b1[j] + (0..3).map(|i| w1[j * 3 + i] * x[i]).sum::<f64>()).tanh()).sum();
            assert_eq!(y, want);
        }
        let spread = (y.iter().map(|v| v * v).sum::<f64>() / 500.0).sqrt();
        assert!((0.4..2.5).contains(&spread), "teacher's outputs spread {spread}");
        let (x2, _) = teacher(500, 3, &mut Rng::new(26));
        assert_ne!(x, x2);
        // the repeated runs
        let mut rng = Rng::new(27);
        for (context, vocab) in [(12, 10), (13, 10), (12, 3), (2, 5)] {
            let (tokens, targets) = repeats(context, vocab, &mut rng);
            let run = run_length(context);
            assert_eq!((tokens.len(), targets.len()), (context, context));
            for t in run..context {
                assert_eq!(tokens[t], tokens[t - run]);
            }
            for t in 0..context - 1 {
                assert_eq!(targets[t], tokens[t + 1]);
            }
            assert_eq!(targets[context - 1], NO_TARGET);
            if vocab >= run {
                let mut seen = tokens[..run].to_vec();
                seen.sort();
                seen.dedup();
                assert_eq!(seen.len(), run, "the run's tokens differ");
            }
        }
        assert_eq!((run_length(12), run_length(13), run_length(1)), (6, 6, 1));
    }

    /// The cost of one forward and backward pass (and accumulating its gradient), for
    /// the scenes to budget by: run with `cargo test --release --lib math::arch --
    /// --ignored --nocapture`. Each figure is the best of five runs of a few hundred
    /// passes, and the median of the five beside it.
    #[test]
    #[ignore]
    fn the_cost_of_a_pass() {
        let time = |mut pass: Box<dyn FnMut()>| {
            for _ in 0..50 {
                pass();
            }
            let mut runs: Vec<f64> = (0..5)
                .map(|_| {
                    let start = std::time::Instant::now();
                    for _ in 0..300 {
                        pass();
                    }
                    start.elapsed().as_secs_f64() * 1e6 / 300.0
                })
                .collect();
            runs.sort_by(f64::total_cmp);
            (runs[0], runs[2])
        };
        let mut rng = Rng::new(28);
        let (x, y) = spirals(16, 3, &mut rng);
        for (act, norm, residual) in [(Act::Relu, Norm::None, false), (Act::Tanh, Norm::Layer, true), (Act::Gelu, Norm::Batch, false)] {
            let spec = MlpSpec { inputs: 2, width: 32, depth: 20, outputs: 3, act, norm, residual, ..MlpSpec::default() };
            let mut model = Model::mlp(&spec, &mut rng);
            let (x, y) = (x.clone(), y.clone());
            let mut tape = Tape::new();
            let (best, median) = time(Box::new(move || {
                let trace = model.loss_mlp(&mut tape, &x, &y, Loss::CrossEntropy);
                tape.backward(trace.loss);
                model.accumulate(&tape, &trace);
            }));
            println!(
                "MLP 20 x 32, batch 16, {} {} {}: {best:.0} us (median {median:.0})",
                act.name(),
                norm.name(),
                if residual { "residual" } else { "plain" }
            );
        }
        for (act, d_mlp, heads) in [(Act::Gelu, 128, 4), (Act::Relu, 128, 4), (Act::Gelu, 64, 2)] {
            let spec = TransformerSpec { vocab: 16, context: 12, d_model: 32, heads, layers: 4, d_mlp, act, ..TransformerSpec::default() };
            let mut model = Model::transformer(&spec, &mut rng);
            let (tokens, targets) = repeats(12, 16, &mut rng);
            let mut tape = Tape::new();
            let (best, median) = time(Box::new(move || {
                let trace = model.loss_transformer(&mut tape, &tokens, &targets);
                tape.backward(trace.loss);
                model.accumulate(&tape, &trace);
            }));
            println!("transformer 4 blocks x 32, {heads} heads, MLP {d_mlp}, 12 tokens, {}: {best:.0} us (median {median:.0})", act.name());
        }
    }
}

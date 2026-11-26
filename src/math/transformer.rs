//! A small transformer that can be trained in a frame's spare time and then taken
//! apart: every intermediate quantity of the forward pass is kept, and parts can be
//! knocked out or overwritten on the way through.
//!
//! The model is a decoder-only transformer without normalisation layers:
//!
//! ```text
//!     x0[t]  = E[token t] + P[t]                          the residual stream
//!     x(l+1) = xl + sum over heads (O_h z_h) + MLP_l       each part ADDS to it
//!     z_h[t] = sum over s <= t of A_h[t, s] (V_h xl[s])    attention moves information
//!     A_h[t, .] = softmax over s <= t of (Q_h xl[t]) . (K_h xl[s]) / sqrt(d_head)
//!     logits[t] = U xL[t]
//! ```
//!
//! With no normalisation the final stream is exactly the sum of what the embeddings,
//! each head and each MLP wrote, so "what did this part contribute to that logit" has
//! an exact answer (direct logit attribution), and reading the stream part-way up with
//! the unembedding (the logit lens) is the same linear read-off as at the top.
//!
//! Storage: activations are kept one feature per column over all rows of a batch
//! (`x[i * rows + row]`, `row = sequence * len + position`), so that every matrix
//! product is a handful of long `y += a x` loops (see `nn::axpy4`).

use crate::math::nn::{adam_step_size, adam_update, axpy, axpy4, dot, fast_exp, sum};
use crate::math::rng::Rng;
use std::ops::Range;

/// The shape of a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Config {
    /// Number of different tokens.
    pub vocab: usize,
    /// Longest sequence it can read.
    pub ctx: usize,
    /// Width of the residual stream.
    pub d: usize,
    pub layers: usize,
    /// Attention heads per layer; `d` must be a multiple of it (each head works in
    /// `d / heads` dimensions).
    pub heads: usize,
    /// Hidden width of each layer's MLP, or 0 for an attention-only model.
    pub mlp: usize,
}

impl Config {
    pub fn d_head(&self) -> usize {
        self.d / self.heads
    }
}

/// How a head is knocked out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Knock {
    /// Its output is replaced by nothing at all.
    Zero,
    /// Its output is replaced, at every position, by its average over the batch: the
    /// head still says what it says on average, but nothing about this input.
    Mean,
}

/// Overwrite (or add to) the residual stream at one place on the way through.
/// `point` 0 is the stream before the first layer, `point` l the stream after layer
/// l - 1, so `point == layers` is the final stream.
#[derive(Clone, Debug)]
pub struct Patch {
    pub point: usize,
    pub seq: usize,
    pub pos: usize,
    pub value: Vec<f32>,
    /// Add `value` to the stream instead of replacing it.
    pub add: bool,
}

/// Interventions applied during a forward pass.
#[derive(Clone, Debug, Default)]
pub struct Edits {
    /// `(layer, head, how)`.
    pub knocked: Vec<(usize, usize, Knock)>,
    /// `(layer, head, value)`: heads whose output is replaced, at every position, by
    /// the given vector.
    pub fixed: Vec<(usize, usize, Vec<f32>)>,
    pub patches: Vec<Patch>,
}

impl Edits {
    pub fn none() -> Self {
        Edits::default()
    }

    pub fn knock(mut self, layer: usize, head: usize, how: Knock) -> Self {
        self.knocked.push((layer, head, how));
        self
    }

    /// Replace what a head writes by `value`, the same at every position: its
    /// average measured beforehand on other sequences, say (see
    /// [`Transformer::head_mean`]), which unlike [`Knock::Mean`] does not depend on
    /// what else is in the batch.
    /// (`value` is one vector of the stream's width.)
    pub fn fix(mut self, layer: usize, head: usize, value: &[f32]) -> Self {
        self.fixed.push((layer, head, value.to_vec()));
        self
    }

    /// Replace the stream at `point` (see [`Patch`]) of sequence `seq`, position `pos`.
    pub fn replace(mut self, point: usize, seq: usize, pos: usize, value: &[f32]) -> Self {
        self.patches.push(Patch { point, seq, pos, value: value.to_vec(), add: false });
        self
    }

    /// Add `value` to the stream there.
    pub fn add(mut self, point: usize, seq: usize, pos: usize, value: &[f32]) -> Self {
        self.patches.push(Patch { point, seq, pos, value: value.to_vec(), add: true });
        self
    }

    pub fn is_empty(&self) -> bool {
        self.knocked.is_empty() && self.fixed.is_empty() && self.patches.is_empty()
    }
}

// ----- tasks --------------------------------------------------------------------------

/// Something a model can be trained to predict the next token of.
#[allow(clippy::len_without_is_empty)] // (a task's sequences are never empty)
pub trait Task {
    fn vocab(&self) -> usize;
    /// Length of the sequences it makes.
    fn len(&self) -> usize;
    /// Write one sequence into `tokens`, and into `scored` a 1 at every position
    /// whose NEXT token the model is asked to predict (0 elsewhere; the last position
    /// has no next token and is always 0).
    fn sample(&self, rng: &mut Rng, tokens: &mut [u32], scored: &mut [f32]);
}

/// A run of different tokens, repeated until the sequence is full: `C A F B  C A F B
/// C A ...`. The run's length changes from sequence to sequence, so the only way to
/// predict a repeat is to find the earlier copy of the current token and look at
/// what followed it. (If the run length were fixed, looking a fixed distance back
/// would do, and no token would ever have to be compared with another.) Positions
/// are scored from the start of the second copy on, where the current token has an
/// earlier copy. (Not at the last token of the first run: every token so far is new
/// there, so whether the run ends or goes on cannot be told from the tokens, and
/// scoring only the rows where it ends would teach the model to guess an end.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Repeat {
    pub vocab: usize,
    pub len: usize,
    /// Shortest and longest run.
    pub shortest: usize,
    pub longest: usize,
}

impl Repeat {
    /// One sequence with the given run length (the scene uses this to show a
    /// sequence of a chosen period).
    pub fn with_run(&self, run: usize, rng: &mut Rng, tokens: &mut [u32], scored: &mut [f32]) {
        let run = run.clamp(1, self.vocab.min(self.len).max(1));
        // the first `run` entries of a shuffled alphabet
        let mut letters: Vec<u32> = (0..self.vocab as u32).collect();
        for i in 0..run {
            let j = i + rng.below(self.vocab - i);
            letters.swap(i, j);
        }
        for t in 0..self.len {
            tokens[t] = letters[t % run];
            scored[t] = if t >= run && t + 1 < self.len { 1.0 } else { 0.0 };
        }
    }
}

impl Task for Repeat {
    fn vocab(&self) -> usize {
        self.vocab
    }

    fn len(&self) -> usize {
        self.len
    }

    fn sample(&self, rng: &mut Rng, tokens: &mut [u32], scored: &mut [f32]) {
        let run = self.shortest + rng.below(self.longest - self.shortest + 1);
        self.with_run(run, rng, tokens, scored);
    }
}

/// Sequences laid end to end, with each position's target (the next token) and
/// whether it is scored.
#[derive(Clone, Debug, Default)]
pub struct Batch {
    pub count: usize,
    pub len: usize,
    pub tokens: Vec<u32>,
    pub targets: Vec<u32>,
    pub mask: Vec<f32>,
}

impl Batch {
    pub fn sample(task: &dyn Task, count: usize, rng: &mut Rng) -> Self {
        let mut b = Batch::default();
        b.refill(task, count, rng);
        b
    }

    /// Draw `count` fresh sequences.
    pub fn refill(&mut self, task: &dyn Task, count: usize, rng: &mut Rng) {
        let len = task.len();
        (self.count, self.len) = (count, len);
        self.tokens.resize(count * len, 0);
        self.targets.resize(count * len, 0);
        self.mask.resize(count * len, 0.0);
        for b in 0..count {
            let at = b * len..(b + 1) * len;
            task.sample(rng, &mut self.tokens[at.clone()], &mut self.mask[at]);
        }
        self.set_targets();
    }

    /// One sequence given outright.
    pub fn single(tokens: &[u32], scored: &[f32]) -> Self {
        let mut b = Batch { count: 1, len: tokens.len(), tokens: tokens.to_vec(), targets: vec![0; tokens.len()], mask: scored.to_vec() };
        b.set_targets();
        b
    }

    fn set_targets(&mut self) {
        let len = self.len;
        for b in 0..self.count {
            for t in 0..len {
                let n = b * len + t;
                if t + 1 < len {
                    self.targets[n] = self.tokens[n + 1];
                } else {
                    self.targets[n] = 0;
                    self.mask[n] = 0.0;
                }
            }
        }
    }
}

// ----- the model ----------------------------------------------------------------------

#[derive(Clone, Debug)]
struct LayerAt {
    q: Range<usize>,
    k: Range<usize>,
    v: Range<usize>,
    o: Range<usize>,
    w_in: Range<usize>,
    b_in: Range<usize>,
    w_out: Range<usize>,
    b_out: Range<usize>,
}

/// Where each group of parameters lives in the flat parameter vector.
#[derive(Clone, Debug)]
struct Layout {
    /// `tok[i * vocab + token]`: component `i` of a token's embedding.
    tok: Range<usize>,
    /// `pos[i * ctx + t]`.
    pos: Range<usize>,
    layers: Vec<LayerAt>,
    /// `unembed[token * d + i]`.
    unembed: Range<usize>,
    total: usize,
}

impl Layout {
    fn new(c: &Config) -> Self {
        let mut at = 0;
        let mut take = |n: usize| {
            let r = at..at + n;
            at += n;
            r
        };
        let (d, m) = (c.d, c.mlp);
        let tok = take(c.vocab * d);
        let pos = take(c.ctx * d);
        let layers = (0..c.layers)
            .map(|_| LayerAt {
                q: take(d * d),
                k: take(d * d),
                v: take(d * d),
                o: take(d * d),
                w_in: take(m * d),
                b_in: take(m),
                w_out: take(d * m),
                b_out: take(if m > 0 { d } else { 0 }),
            })
            .collect();
        let unembed = take(c.vocab * d);
        Layout { tok, pos, layers, unembed, total: at }
    }
}

/// Everything a forward pass computed (see the module comment for the layout).
#[derive(Default)]
struct Cache {
    rows: usize,
    len: usize,
    count: usize,
    tokens: Vec<u32>,
    /// `layers + 1` streams of `d * rows`.
    resid: Vec<Vec<f32>>,
    /// The stream after attention and before the MLP (only kept if there is an MLP).
    mid: Vec<Vec<f32>>,
    q: Vec<Vec<f32>>,
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    z: Vec<Vec<f32>>,
    /// Per layer, `att[(head * rows + row) * len + s]`.
    att: Vec<Vec<f32>>,
    /// Per layer, `head_out[(head * d + i) * rows + row]`.
    head_out: Vec<Vec<f32>>,
    pre: Vec<Vec<f32>>,
    hid: Vec<Vec<f32>>,
    mlp_out: Vec<Vec<f32>>,
    logits: Vec<f32>,
    probs: Vec<f32>,
    /// Which heads the last pass knocked out (and whether any by its average), and
    /// which (point, row) it overwrote.
    knocked: Vec<bool>,
    averaged: bool,
    replaced: Vec<(usize, usize)>,
}

pub struct Transformer {
    pub cfg: Config,
    at: Layout,
    p: Vec<f32>,
    g: Vec<f32>,
    /// Adam's running averages and step count.
    mean: Vec<f32>,
    square: Vec<f32>,
    steps: u32,
    /// The stretches of `p` that weight decay shrinks: all but the MLP biases.
    decayed: Vec<std::ops::Range<usize>>,
    c: Cache,
    // scratch for the backward pass
    dx: Vec<f32>,
    dl: Vec<f32>,
    dz: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
    dh: Vec<f32>,
    row: Vec<f32>,
    row2: Vec<f32>,
}

/// `y_j += sum_i w[j * n_in + i] x_i` for every output `j` (`x` is `n_in` columns of
/// `rows`, `y` is `n_out` columns).
fn matmul_add(y: &mut [f32], w: &[f32], x: &[f32], n_out: usize, n_in: usize, rows: usize) {
    for j in 0..n_out {
        let yj = &mut y[j * rows..(j + 1) * rows];
        let wj = &w[j * n_in..(j + 1) * n_in];
        let mut i = 0;
        while i + 4 <= n_in {
            let col = |k: usize| &x[(i + k) * rows..(i + k + 1) * rows];
            axpy4(yj, [wj[i], wj[i + 1], wj[i + 2], wj[i + 3]], [col(0), col(1), col(2), col(3)]);
            i += 4;
        }
        while i < n_in {
            axpy(yj, wj[i], &x[i * rows..(i + 1) * rows]);
            i += 1;
        }
    }
}

/// The transpose: `x_i += sum_j w[j * n_in + i] y_j` (how a gradient at the outputs
/// reaches the inputs).
fn matmul_t_add(x: &mut [f32], w: &[f32], y: &[f32], n_out: usize, n_in: usize, rows: usize) {
    for i in 0..n_in {
        let xi = &mut x[i * rows..(i + 1) * rows];
        let mut j = 0;
        while j + 4 <= n_out {
            let col = |k: usize| &y[(j + k) * rows..(j + k + 1) * rows];
            axpy4(xi, [w[j * n_in + i], w[(j + 1) * n_in + i], w[(j + 2) * n_in + i], w[(j + 3) * n_in + i]], [col(0), col(1), col(2), col(3)]);
            j += 4;
        }
        while j < n_out {
            axpy(xi, w[j * n_in + i], &y[j * rows..(j + 1) * rows]);
            j += 1;
        }
    }
}

/// `dw[j * n_in + i] += dy_j . x_i`.
fn weight_grad(dw: &mut [f32], dy: &[f32], x: &[f32], n_out: usize, n_in: usize, rows: usize) {
    for j in 0..n_out {
        let dyj = &dy[j * rows..(j + 1) * rows];
        for i in 0..n_in {
            dw[j * n_in + i] += dot(dyj, &x[i * rows..(i + 1) * rows]);
        }
    }
}

/// Softmax of each row of a table kept one column per class.
fn softmax_columns(logits: &[f32], probs: &mut [f32], classes: usize, rows: usize, scratch: &mut Vec<f32>) {
    scratch.clear();
    scratch.resize(2 * rows, 0.0);
    let (top, total) = scratch.split_at_mut(rows);
    top.fill(f32::MIN);
    for c in 0..classes {
        for (m, &l) in top.iter_mut().zip(&logits[c * rows..(c + 1) * rows]) {
            *m = m.max(l);
        }
    }
    for c in 0..classes {
        let (l, p) = (&logits[c * rows..(c + 1) * rows], &mut probs[c * rows..(c + 1) * rows]);
        for n in 0..rows {
            let e = fast_exp(l[n] - top[n]);
            p[n] = e;
            total[n] += e;
        }
    }
    for t in total.iter_mut() {
        *t = 1.0 / *t;
    }
    for c in 0..classes {
        for (p, &inv) in probs[c * rows..(c + 1) * rows].iter_mut().zip(total.iter()) {
            *p *= inv;
        }
    }
}

impl Transformer {
    pub fn new(cfg: Config, rng: &mut Rng) -> Self {
        assert!(cfg.heads >= 1 && cfg.d.is_multiple_of(cfg.heads), "d must be a multiple of the number of heads");
        let at = Layout::new(&cfg);
        let mut p = vec![0.0; at.total];
        let d = cfg.d as f32;
        let mut fill = |r: &Range<usize>, std: f32| p[r.clone()].iter_mut().for_each(|w| *w = rng.normal() * std);
        // Embeddings of length about 1; attention scores start small (nearly even
        // attention); what a head or MLP writes starts small next to the stream; and
        // the first predictions are nearly even over the vocabulary.
        fill(&at.tok, 1.0 / d.sqrt());
        fill(&at.pos, 0.5 / d.sqrt());
        for l in &at.layers {
            fill(&l.q, 0.7 / d.sqrt());
            fill(&l.k, 0.7 / d.sqrt());
            fill(&l.v, 0.7 / d.sqrt());
            fill(&l.o, 0.7 / d.sqrt());
            fill(&l.w_in, 1.0 / d.sqrt());
            fill(&l.w_out, 0.5 / (cfg.mlp.max(1) as f32).sqrt());
        }
        fill(&at.unembed, 0.5 / d.sqrt());
        let n = at.total;
        // everything between the biases
        let mut spared: Vec<std::ops::Range<usize>> = at.layers.iter().flat_map(|l| [l.b_in.clone(), l.b_out.clone()]).collect();
        spared.sort_by_key(|r| r.start);
        let (mut decayed, mut from) = (Vec::new(), 0);
        for r in spared.iter().chain(std::iter::once(&(n..n))) {
            if from < r.start {
                decayed.push(from..r.start);
            }
            from = from.max(r.end);
        }
        Transformer {
            cfg,
            at,
            p,
            g: vec![0.0; n],
            mean: vec![0.0; n],
            square: vec![0.0; n],
            steps: 0,
            decayed,
            c: Cache::default(),
            dx: Vec::new(),
            dl: Vec::new(),
            dz: Vec::new(),
            dq: Vec::new(),
            dk: Vec::new(),
            dv: Vec::new(),
            dh: Vec::new(),
            row: Vec::new(),
            row2: Vec::new(),
        }
    }

    pub fn n_params(&self) -> usize {
        self.p.len()
    }

    pub fn params(&self) -> &[f32] {
        &self.p
    }

    pub fn params_mut(&mut self) -> &mut [f32] {
        &mut self.p
    }

    /// The gradient left by the last [`Transformer::backward`].
    pub fn grads(&self) -> &[f32] {
        &self.g
    }

    /// Named parameter groups: `(name, range in the parameter vector)`.
    pub fn groups(&self) -> Vec<(String, Range<usize>)> {
        let mut out = vec![("token embedding".to_string(), self.at.tok.clone()), ("position embedding".to_string(), self.at.pos.clone())];
        for (l, a) in self.at.layers.iter().enumerate() {
            for (name, r) in [
                ("Q", &a.q),
                ("K", &a.k),
                ("V", &a.v),
                ("O", &a.o),
                ("MLP in", &a.w_in),
                ("MLP in bias", &a.b_in),
                ("MLP out", &a.w_out),
                ("MLP out bias", &a.b_out),
            ] {
                if !r.is_empty() {
                    out.push((format!("layer {l} {name}"), r.clone()));
                }
            }
        }
        out.push(("unembedding".to_string(), self.at.unembed.clone()));
        out
    }

    fn size_cache(&mut self, count: usize, len: usize) {
        let cfg = self.cfg;
        let rows = count * len;
        let c = &mut self.c;
        if c.rows == rows && c.len == len && c.resid.len() == cfg.layers + 1 {
            return;
        }
        (c.rows, c.len, c.count) = (rows, len, count);
        // The buffers are kept and only resized. A scene that trains on sixteen rows,
        // shows one and tests on twelve changes the shape several times a frame, and
        // letting go of a megabyte and asking for it again each time was a twentieth
        // of a training step.
        fn per(v: &mut Vec<Vec<f32>>, n: usize, size: usize) {
            v.resize_with(n, Vec::new);
            v.iter_mut().for_each(|x| zeroed(x, size));
        }
        fn zeroed(x: &mut Vec<f32>, size: usize) {
            x.clear();
            x.resize(size, 0.0);
        }
        let m = if cfg.mlp > 0 { cfg.layers } else { 0 };
        per(&mut c.resid, cfg.layers + 1, cfg.d * rows);
        per(&mut c.mid, m, cfg.d * rows);
        per(&mut c.q, cfg.layers, cfg.d * rows);
        per(&mut c.k, cfg.layers, cfg.d * rows);
        per(&mut c.v, cfg.layers, cfg.d * rows);
        per(&mut c.z, cfg.layers, cfg.d * rows);
        per(&mut c.att, cfg.layers, cfg.heads * rows * len);
        per(&mut c.head_out, cfg.layers, cfg.heads * cfg.d * rows);
        per(&mut c.pre, m, cfg.mlp * rows);
        per(&mut c.hid, m, cfg.mlp * rows);
        per(&mut c.mlp_out, m, cfg.d * rows);
        zeroed(&mut c.logits, cfg.vocab * rows);
        zeroed(&mut c.probs, cfg.vocab * rows);
    }

    /// Run `count` sequences laid end to end in `tokens`, keeping everything.
    pub fn forward(&mut self, tokens: &[u32], count: usize, edits: &Edits) {
        let cfg = self.cfg;
        let len = tokens.len() / count.max(1);
        assert!(count >= 1 && len >= 1 && len * count == tokens.len() && len <= cfg.ctx, "bad batch shape");
        self.size_cache(count, len);
        let rows = count * len;
        let (d, heads, dh, m) = (cfg.d, cfg.heads, cfg.d_head(), cfg.mlp);
        let Transformer { p, at, c, row, .. } = self;
        c.tokens.clear();
        c.tokens.extend_from_slice(tokens);
        c.knocked.clear();
        c.knocked.resize(cfg.layers * heads, false);
        // (set where a head is in fact replaced by its average, below)
        c.averaged = false;
        assert!(edits.fixed.iter().all(|e| e.2.len() == d), "a head is fixed to a vector of the stream's width");
        c.replaced.clear();

        let patch = |point: usize, x: &mut [f32], replaced: &mut Vec<(usize, usize)>| {
            for e in edits.patches.iter().filter(|e| e.point == point) {
                let n = e.seq * len + e.pos;
                assert!(e.seq < count && e.pos < len && e.value.len() == d, "patch out of range");
                for i in 0..d {
                    if e.add {
                        x[i * rows + n] += e.value[i];
                    } else {
                        x[i * rows + n] = e.value[i];
                    }
                }
                if !e.add {
                    replaced.push((point, n));
                }
            }
        };

        // the embeddings
        {
            let x = &mut c.resid[0];
            let (tok, pos) = (&p[at.tok.clone()], &p[at.pos.clone()]);
            for i in 0..d {
                let (ti, pi) = (&tok[i * cfg.vocab..(i + 1) * cfg.vocab], &pos[i * cfg.ctx..(i + 1) * cfg.ctx]);
                let xi = &mut x[i * rows..(i + 1) * rows];
                for b in 0..count {
                    for t in 0..len {
                        xi[b * len + t] = ti[tokens[b * len + t] as usize] + pi[t];
                    }
                }
            }
            patch(0, x, &mut c.replaced);
        }

        let scale = 1.0 / (dh as f32).sqrt();
        row.clear();
        row.resize(len, 0.0);
        for l in 0..cfg.layers {
            let la = &at.layers[l];
            let (before, after) = c.resid.split_at_mut(l + 1);
            let (x, out) = (&before[l], &mut after[0]);
            let (q, k, v, z) = (&mut c.q[l], &mut c.k[l], &mut c.v[l], &mut c.z[l]);
            q.fill(0.0);
            k.fill(0.0);
            v.fill(0.0);
            matmul_add(q, &p[la.q.clone()], x, d, d, rows);
            matmul_add(k, &p[la.k.clone()], x, d, d, rows);
            matmul_add(v, &p[la.v.clone()], x, d, d, rows);

            // where each position looks, and what it collects from there
            let att = &mut c.att[l];
            for h in 0..heads {
                for b in 0..count {
                    let base = b * len;
                    for t in 0..len {
                        let n = base + t;
                        let live = &mut row[..=t];
                        live.fill(0.0);
                        for comp in h * dh..(h + 1) * dh {
                            axpy(live, q[comp * rows + n] * scale, &k[comp * rows + base..comp * rows + base + t + 1]);
                        }
                        let top = live.iter().fold(f32::MIN, |a, &b| a.max(b));
                        let mut total = 0.0;
                        for s in live.iter_mut() {
                            *s = fast_exp(*s - top);
                            total += *s;
                        }
                        let inv = 1.0 / total;
                        let a = &mut att[(h * rows + n) * len..(h * rows + n + 1) * len];
                        for s in 0..len {
                            a[s] = if s <= t { live[s] * inv } else { 0.0 };
                        }
                        for comp in h * dh..(h + 1) * dh {
                            z[comp * rows + n] = dot(&a[..=t], &v[comp * rows + base..comp * rows + base + t + 1]);
                        }
                    }
                }
            }

            // what each head writes to the stream
            let ho = &mut c.head_out[l];
            ho.fill(0.0);
            let o = &p[la.o.clone()];
            for h in 0..heads {
                let knock = edits.knocked.iter().find(|e| e.0 == l && e.1 == h).map(|e| e.2);
                if knock == Some(Knock::Zero) {
                    c.knocked[l * heads + h] = true;
                    continue;
                }
                if let Some((_, _, value)) = edits.fixed.iter().find(|e| e.0 == l && e.1 == h) {
                    for j in 0..d {
                        ho[(h * d + j) * rows..(h * d + j + 1) * rows].fill(value[j]);
                    }
                    c.knocked[l * heads + h] = true;
                    continue;
                }
                for j in 0..d {
                    let yj = &mut ho[(h * d + j) * rows..(h * d + j + 1) * rows];
                    let wj = &o[j * d + h * dh..j * d + (h + 1) * dh];
                    let zc = |comp: usize| &z[(h * dh + comp) * rows..(h * dh + comp + 1) * rows];
                    let mut comp = 0;
                    while comp + 4 <= dh {
                        axpy4(yj, [wj[comp], wj[comp + 1], wj[comp + 2], wj[comp + 3]], [zc(comp), zc(comp + 1), zc(comp + 2), zc(comp + 3)]);
                        comp += 4;
                    }
                    while comp < dh {
                        axpy(yj, wj[comp], zc(comp));
                        comp += 1;
                    }
                    if knock == Some(Knock::Mean) {
                        let average = sum(yj) / rows as f32;
                        yj.fill(average);
                    }
                }
                if knock.is_some() {
                    (c.knocked[l * heads + h], c.averaged) = (true, true);
                }
            }

            // the stream after attention
            let stream: &mut Vec<f32> = if m > 0 { &mut c.mid[l] } else { &mut *out };
            stream.copy_from_slice(x);
            for h in 0..heads {
                for j in 0..d {
                    axpy(&mut stream[j * rows..(j + 1) * rows], 1.0, &ho[(h * d + j) * rows..(h * d + j + 1) * rows]);
                }
            }
            if m > 0 {
                let (mid, pre, hid, mo) = (&c.mid[l], &mut c.pre[l], &mut c.hid[l], &mut c.mlp_out[l]);
                let (b_in, b_out) = (&p[la.b_in.clone()], &p[la.b_out.clone()]);
                for (j, col) in pre.chunks_exact_mut(rows).enumerate() {
                    col.fill(b_in[j]);
                }
                matmul_add(pre, &p[la.w_in.clone()], mid, m, d, rows);
                for (h, &z) in hid.iter_mut().zip(pre.iter()) {
                    *h = z.max(0.0);
                }
                for (j, col) in mo.chunks_exact_mut(rows).enumerate() {
                    col.fill(b_out[j]);
                }
                matmul_add(mo, &p[la.w_out.clone()], hid, d, m, rows);
                for ((o, &a), &b) in out.iter_mut().zip(mid.iter()).zip(mo.iter()) {
                    *o = a + b;
                }
            }
            patch(l + 1, out, &mut c.replaced);
        }

        c.logits.fill(0.0);
        matmul_add(&mut c.logits, &p[at.unembed.clone()], &c.resid[cfg.layers], cfg.vocab, d, rows);
        softmax_columns(&c.logits, &mut c.probs, cfg.vocab, rows, row);
    }

    /// The loss (average of minus the log of the probability given to the target,
    /// over the scored rows) and the share of scored rows whose most likely token is
    /// the target, for the last forward pass.
    pub fn loss(&self, targets: &[u32], mask: &[f32]) -> (f32, f32) {
        let (rows, vocab) = (self.c.rows, self.cfg.vocab);
        let (mut total, mut right, mut weight) = (0.0f64, 0.0f64, 0.0f64);
        for n in 0..rows {
            if mask[n] <= 0.0 {
                continue;
            }
            let p = self.c.probs[targets[n] as usize * rows + n];
            total += mask[n] as f64 * -(p.max(1e-30) as f64).ln();
            let best = (0..vocab).all(|v| self.c.probs[v * rows + n] <= p);
            right += if best { mask[n] as f64 } else { 0.0 };
            weight += mask[n] as f64;
        }
        if weight == 0.0 {
            return (0.0, 0.0);
        }
        ((total / weight) as f32, (right / weight) as f32)
    }

    /// The gradient of the loss of the last forward pass (which must have been run
    /// without edits). Left in [`Transformer::grads`].
    pub fn backward(&mut self, targets: &[u32], mask: &[f32]) {
        let cfg = self.cfg;
        let (rows, len, count) = (self.c.rows, self.c.len, self.c.count);
        let (d, heads, dh, m, vocab) = (cfg.d, cfg.heads, cfg.d_head(), cfg.mlp, cfg.vocab);
        assert!(self.c.knocked.iter().all(|k| !k) && self.c.replaced.is_empty(), "backward after an edited forward pass");
        let Transformer { p, g, at, c, dx, dl, dz, dq, dk, dv, dh: dhid, row, row2, .. } = self;
        g.fill(0.0);
        let weight: f32 = mask.iter().sum::<f32>().max(1e-12);

        // the loss with respect to the logits
        dl.clear();
        dl.extend_from_slice(&c.probs);
        for v in 0..vocab {
            for (x, &w) in dl[v * rows..(v + 1) * rows].iter_mut().zip(mask) {
                *x *= w / weight;
            }
        }
        for n in 0..rows {
            dl[targets[n] as usize * rows + n] -= mask[n] / weight;
        }
        let last = &c.resid[cfg.layers];
        weight_grad(&mut g[at.unembed.clone()], dl, last, vocab, d, rows);
        dx.clear();
        dx.resize(d * rows, 0.0);
        matmul_t_add(dx, &p[at.unembed.clone()], dl, vocab, d, rows);

        let scale = 1.0 / (dh as f32).sqrt();
        for buffer in [&mut *dz, &mut *dq, &mut *dk, &mut *dv] {
            buffer.clear();
            buffer.resize(d * rows, 0.0);
        }
        row.clear();
        row.resize(len, 0.0);
        row2.clear();
        row2.resize(len, 0.0);
        for l in (0..cfg.layers).rev() {
            let la = &at.layers[l];
            if m > 0 {
                // out = mid + W_out relu(W_in mid + b_in) + b_out
                let (mid, pre, hid) = (&c.mid[l], &c.pre[l], &c.hid[l]);
                weight_grad(&mut g[la.w_out.clone()], dx, hid, d, m, rows);
                for (j, gb) in g[la.b_out.clone()].iter_mut().enumerate() {
                    *gb += sum(&dx[j * rows..(j + 1) * rows]);
                }
                dhid.clear();
                dhid.resize(m * rows, 0.0);
                matmul_t_add(dhid, &p[la.w_out.clone()], dx, d, m, rows);
                for (g, &z) in dhid.iter_mut().zip(pre.iter()) {
                    if z <= 0.0 {
                        *g = 0.0;
                    }
                }
                weight_grad(&mut g[la.w_in.clone()], dhid, mid, m, d, rows);
                for (j, gb) in g[la.b_in.clone()].iter_mut().enumerate() {
                    *gb += sum(&dhid[j * rows..(j + 1) * rows]);
                }
                matmul_t_add(dx, &p[la.w_in.clone()], dhid, m, d, rows);
            }
            // (dx is now the gradient at the stream after attention)
            let (x, q, k, v, z, att) = (&c.resid[l], &c.q[l], &c.k[l], &c.v[l], &c.z[l], &c.att[l]);
            weight_grad(&mut g[la.o.clone()], dx, z, d, d, rows);
            dz.fill(0.0);
            matmul_t_add(dz, &p[la.o.clone()], dx, d, d, rows);
            dq.fill(0.0);
            dk.fill(0.0);
            dv.fill(0.0);
            for h in 0..heads {
                for b in 0..count {
                    let base = b * len;
                    for t in 0..len {
                        let n = base + t;
                        let a = &att[(h * rows + n) * len..(h * rows + n) * len + t + 1];
                        // through z = sum_s a[s] v[s]
                        let da = &mut row[..=t];
                        da.fill(0.0);
                        for comp in h * dh..(h + 1) * dh {
                            let gz = dz[comp * rows + n];
                            let span = comp * rows + base..comp * rows + base + t + 1;
                            axpy(da, gz, &v[span.clone()]);
                            axpy(&mut dv[span], gz, a);
                        }
                        // through the softmax
                        let mean = dot(a, da);
                        let ds = &mut row2[..=t];
                        for s in 0..=t {
                            ds[s] = a[s] * (da[s] - mean) * scale;
                        }
                        // through score[s] = q . k[s]
                        for comp in h * dh..(h + 1) * dh {
                            let span = comp * rows + base..comp * rows + base + t + 1;
                            dq[comp * rows + n] = dot(ds, &k[span.clone()]);
                            axpy(&mut dk[span], q[comp * rows + n], ds);
                        }
                    }
                }
            }
            weight_grad(&mut g[la.q.clone()], dq, x, d, d, rows);
            weight_grad(&mut g[la.k.clone()], dk, x, d, d, rows);
            weight_grad(&mut g[la.v.clone()], dv, x, d, d, rows);
            matmul_t_add(dx, &p[la.q.clone()], dq, d, d, rows);
            matmul_t_add(dx, &p[la.k.clone()], dk, d, d, rows);
            matmul_t_add(dx, &p[la.v.clone()], dv, d, d, rows);
        }

        // the embeddings
        let (tok_at, pos_at) = (at.tok.start, at.pos.start);
        for i in 0..d {
            let dxi = &dx[i * rows..(i + 1) * rows];
            for b in 0..count {
                for t in 0..len {
                    let n = b * len + t;
                    g[tok_at + i * vocab + c.tokens[n] as usize] += dxi[n];
                    g[pos_at + i * cfg.ctx + t] += dxi[n];
                }
            }
        }
    }

    /// One step of Adam with decoupled weight decay (the decay is not applied to
    /// the MLP biases).
    pub fn adam(&mut self, lr: f32, decay: f32) {
        self.steps = self.steps.saturating_add(1);
        let step = adam_step_size(self.steps, lr);
        if decay > 0.0 {
            // (never below zero, which would flip every weight's sign)
            let keep = (1.0 - lr * decay).max(0.0);
            for r in &self.decayed {
                self.p[r.clone()].iter_mut().for_each(|w| *w *= keep);
            }
        }
        adam_update(step, &mut self.p, &self.g, &mut self.mean, &mut self.square);
    }

    /// Forget Adam's running averages.
    pub fn reset_optimizer(&mut self) {
        self.mean.fill(0.0);
        self.square.fill(0.0);
        self.steps = 0;
    }

    /// One training step on `batch`; returns the loss and accuracy before the step.
    pub fn train_step(&mut self, batch: &Batch, lr: f32, decay: f32) -> (f32, f32) {
        self.forward(&batch.tokens, batch.count, &Edits::default());
        let result = self.loss(&batch.targets, &batch.mask);
        self.backward(&batch.targets, &batch.mask);
        self.adam(lr, decay);
        result
    }

    /// Loss and accuracy on `batch` with the given interventions.
    pub fn evaluate(&mut self, batch: &Batch, edits: &Edits) -> (f32, f32) {
        self.forward(&batch.tokens, batch.count, edits);
        self.loss(&batch.targets, &batch.mask)
    }

    // ----- reading the last forward pass ----------------------------------------------

    /// Rows in the last forward pass (`sequences * length`), and the length.
    pub fn shape(&self) -> (usize, usize) {
        (self.c.rows, self.c.len)
    }

    fn column(buffer: &[f32], rows: usize, row: usize, out: &mut [f32]) {
        for (i, o) in out.iter_mut().enumerate() {
            *o = buffer[i * rows + row];
        }
    }

    /// The residual stream at `point` (0 = before the first layer, `layers` = final)
    /// in row `row`.
    pub fn stream(&self, point: usize, row: usize, out: &mut [f32]) {
        Self::column(&self.c.resid[point], self.c.rows, row, out);
    }

    /// What head `head` of `layer` wrote to the stream in row `row`.
    pub fn head_vector(&self, layer: usize, head: usize, row: usize, out: &mut [f32]) {
        let (d, rows) = (self.cfg.d, self.c.rows);
        Self::column(&self.c.head_out[layer][head * d * rows..(head + 1) * d * rows], rows, row, out);
    }

    /// The average of what a head wrote, over every position of every sequence of
    /// the last pass.
    pub fn head_mean(&self, layer: usize, head: usize, out: &mut [f32]) {
        let (d, rows) = (self.cfg.d, self.c.rows);
        let written = &self.c.head_out[layer][head * d * rows..(head + 1) * d * rows];
        for (j, o) in out.iter_mut().enumerate().take(d) {
            *o = sum(&written[j * rows..(j + 1) * rows]) / rows.max(1) as f32;
        }
    }

    /// What the MLP of `layer` wrote (zero for an attention-only model).
    pub fn mlp_vector(&self, layer: usize, row: usize, out: &mut [f32]) {
        if self.cfg.mlp == 0 {
            out.fill(0.0);
        } else {
            Self::column(&self.c.mlp_out[layer], self.c.rows, row, out);
        }
    }

    /// A token's embedding.
    pub fn token_vector(&self, token: usize, out: &mut [f32]) {
        Self::column(&self.p[self.at.tok.clone()], self.cfg.vocab, token, out);
    }

    /// A position's embedding.
    pub fn position_vector(&self, pos: usize, out: &mut [f32]) {
        Self::column(&self.p[self.at.pos.clone()], self.cfg.ctx, pos, out);
    }

    /// Where head `head` of `layer` looks from row `row`: one weight per position of
    /// that row's sequence (zero for later positions).
    pub fn attention(&self, layer: usize, head: usize, row: usize) -> &[f32] {
        let (rows, len) = (self.c.rows, self.c.len);
        &self.c.att[layer][(head * rows + row) * len..(head * rows + row + 1) * len]
    }

    pub fn logit(&self, token: usize, row: usize) -> f32 {
        self.c.logits[token * self.c.rows + row]
    }

    /// The probability the model gives `token` as the next one after row `row`.
    pub fn prob(&self, token: usize, row: usize) -> f32 {
        self.c.probs[token * self.c.rows + row]
    }

    /// The most likely next token after row `row`.
    pub fn best(&self, row: usize) -> usize {
        (0..self.cfg.vocab).max_by(|&a, &b| self.prob(a, row).total_cmp(&self.prob(b, row))).unwrap_or(0)
    }

    /// The logit a vector in the stream gives `token` when read by the unembedding.
    /// Reading is linear, so the logits of a sum are the sum of the logits: this is
    /// what direct logit attribution adds up.
    pub fn read_logit(&self, vector: &[f32], token: usize) -> f32 {
        let d = self.cfg.d;
        let u = &self.p[self.at.unembed.start + token * d..self.at.unembed.start + (token + 1) * d];
        u.iter().zip(vector).map(|(a, b)| a * b).sum()
    }

    /// The probabilities read off a vector in the stream (the logit lens when the
    /// vector is the stream part-way up).
    pub fn read_probs(&self, vector: &[f32], out: &mut [f32]) {
        for (token, o) in out.iter_mut().enumerate() {
            *o = self.read_logit(vector, token);
        }
        let top = out.iter().fold(f32::MIN, |a, &b| a.max(b));
        let mut total = 0.0;
        for o in out.iter_mut() {
            *o = fast_exp(*o - top);
            total += *o;
        }
        out.iter_mut().for_each(|o| *o /= total);
    }

    // ----- Jacobians ------------------------------------------------------------------

    /// How the final residual stream of sequence `seq` of the last forward pass
    /// changes, per unit, when the stream at `point` (see [`Patch`]), position `pos`,
    /// is nudged along `direction`: the Jacobian of the rest of the network applied
    /// to that direction, computed exactly by carrying the nudge forward. `out` gets
    /// `len * d` numbers, `out[t * d + i]` being component `i` at position `t`
    /// (zero before `pos`: attention only looks back).
    pub fn tangent(&self, seq: usize, point: usize, pos: usize, direction: &[f32], out: &mut Vec<f32>) {
        let cfg = self.cfg;
        let (rows, len) = (self.c.rows, self.c.len);
        let (d, heads, dh, m) = (cfg.d, cfg.heads, cfg.d_head(), cfg.mlp);
        let base = seq * len;
        assert!(point <= cfg.layers && pos < len && base + len <= rows && direction.len() == d);
        let c = &self.c;
        // (a head replaced by its average still answers a nudge, by the average of
        // its answers over the batch, which this does not carry)
        assert!(!c.averaged, "tangent after a pass with a head replaced by its average");
        // row-major here: dx[t * d + i]
        out.clear();
        out.resize(len * d, 0.0);
        out[pos * d..(pos + 1) * d].copy_from_slice(direction);
        let scale = 1.0 / (dh as f32).sqrt();
        let (mut dq, mut dk, mut dv, mut dz) = (vec![0.0f32; len * d], vec![0.0f32; len * d], vec![0.0f32; len * d], vec![0.0f32; len * d]);
        let mut ds = vec![0.0f32; len];
        let mut dmid = vec![0.0f32; len * d];
        let mut dpre = vec![0.0f32; m];
        let apply = |w: &[f32], x: &[f32], y: &mut [f32], n_out: usize, n_in: usize| {
            for j in 0..n_out {
                y[j] = dot(&w[j * n_in..(j + 1) * n_in], x);
            }
        };
        for l in point..cfg.layers {
            let la = &self.at.layers[l];
            let (q, k, v, att) = (&c.q[l], &c.k[l], &c.v[l], &c.att[l]);
            for t in pos..len {
                let x = &out[t * d..(t + 1) * d];
                apply(&self.p[la.q.clone()], x, &mut dq[t * d..(t + 1) * d], d, d);
                apply(&self.p[la.k.clone()], x, &mut dk[t * d..(t + 1) * d], d, d);
                apply(&self.p[la.v.clone()], x, &mut dv[t * d..(t + 1) * d], d, d);
            }
            dz.fill(0.0);
            for h in 0..heads {
                if c.knocked[l * heads + h] {
                    continue;
                }
                for t in pos..len {
                    let n = base + t;
                    let a = &att[(h * rows + n) * len..(h * rows + n) * len + t + 1];
                    // change in each score, then in each attention weight
                    for s in 0..=t {
                        let mut acc = 0.0;
                        for comp in h * dh..(h + 1) * dh {
                            acc += dq[t * d + comp] * k[comp * rows + base + s];
                            if s >= pos {
                                acc += q[comp * rows + n] * dk[s * d + comp];
                            }
                        }
                        ds[s] = acc * scale;
                    }
                    let mean: f32 = (0..=t).map(|s| a[s] * ds[s]).sum();
                    for comp in h * dh..(h + 1) * dh {
                        let mut acc = 0.0;
                        for s in 0..=t {
                            acc += a[s] * (ds[s] - mean) * v[comp * rows + base + s];
                            if s >= pos {
                                acc += a[s] * dv[s * d + comp];
                            }
                        }
                        dz[t * d + comp] = acc;
                    }
                }
            }
            dmid.copy_from_slice(out);
            let o = &self.p[la.o.clone()];
            for t in pos..len {
                for j in 0..d {
                    dmid[t * d + j] += dot(&o[j * d..(j + 1) * d], &dz[t * d..(t + 1) * d]);
                }
            }
            if m > 0 {
                let (w_in, w_out, pre) = (&self.p[la.w_in.clone()], &self.p[la.w_out.clone()], &c.pre[l]);
                for t in pos..len {
                    let n = base + t;
                    apply(w_in, &dmid[t * d..(t + 1) * d], &mut dpre, m, d);
                    for (j, g) in dpre.iter_mut().enumerate() {
                        if pre[j * rows + n] <= 0.0 {
                            *g = 0.0;
                        }
                    }
                    for j in 0..d {
                        dmid[t * d + j] += dot(&w_out[j * m..(j + 1) * m], &dpre);
                    }
                }
            }
            out.copy_from_slice(&dmid);
            // a place the last pass overwrote does not respond to anything upstream
            for &(at_point, n) in &c.replaced {
                if at_point == l + 1 && n >= base && n < base + len {
                    out[(n - base) * d..(n - base + 1) * d].fill(0.0);
                }
            }
        }
    }

    /// The same for a sequence given outright (runs it first): the change in the
    /// final residual stream at every position, `[t * d + i]`, per unit nudge of the
    /// stream at `point`, position `pos`, along `direction`.
    pub fn jvp(&mut self, tokens: &[u32], point: usize, pos: usize, direction: &[f32]) -> Vec<f32> {
        self.forward(tokens, 1, &Edits::default());
        let mut out = Vec::new();
        self.tangent(0, point, pos, direction, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL: Config = Config { vocab: 7, ctx: 6, d: 8, layers: 2, heads: 2, mlp: 12 };

    fn random_batch(cfg: &Config, count: usize, len: usize, rng: &mut Rng) -> Batch {
        let tokens: Vec<u32> = (0..count * len).map(|_| rng.below(cfg.vocab) as u32).collect();
        let mut b =
            Batch { count, len, tokens, targets: vec![0; count * len], mask: (0..count * len).map(|_| if rng.bool(0.7) { 1.0 } else { 0.0 }).collect() };
        b.set_targets();
        b
    }

    /// A model with weights big enough that every part matters to the output.
    fn lively(cfg: Config, seed: u64) -> Transformer {
        let mut rng = Rng::new(seed);
        let mut net = Transformer::new(cfg, &mut rng);
        net.params_mut().iter_mut().for_each(|w| *w *= 1.8);
        let biases: Vec<Range<usize>> = net.at.layers.iter().flat_map(|l| [l.b_in.clone(), l.b_out.clone()]).collect();
        for r in biases {
            net.p[r].iter_mut().for_each(|b| *b = rng.normal() * 0.2);
        }
        net
    }

    #[test]
    fn gradient_matches_finite_differences_in_every_group() {
        for cfg in [SMALL, Config { mlp: 0, heads: 1, ..SMALL }] {
            let mut rng = Rng::new(11);
            let mut net = lively(cfg, 3);
            let batch = random_batch(&cfg, 3, 6, &mut rng);
            net.forward(&batch.tokens, batch.count, &Edits::none());
            net.backward(&batch.targets, &batch.mask);
            let analytic = net.grads().to_vec();
            for (name, range) in net.groups() {
                let mut size = 0.0f32;
                for i in range {
                    let keep = net.p[i];
                    let mut numeric = |h: f32| {
                        net.p[i] = keep + h;
                        let up = net.evaluate(&batch, &Edits::none()).0;
                        net.p[i] = keep - h;
                        let down = net.evaluate(&batch, &Edits::none()).0;
                        net.p[i] = keep;
                        (up - down) / (2.0 * h)
                    };
                    // (two step sizes: a ReLU that switches inside the wider step
                    // spoils that estimate, not the gradient)
                    let estimates = [numeric(2e-3), numeric(4e-4)];
                    let close = |n: f32| (n - analytic[i]).abs() < 2e-3 + 0.03 * n.abs();
                    assert!(estimates.iter().any(|&n| close(n)), "{name}, parameter {i}: numeric {estimates:?} vs analytic {}", analytic[i]);
                    size = size.max(analytic[i].abs());
                }
                // every group has a gradient worth checking
                assert!(size > 3e-3, "{name}: largest gradient only {size}");
            }
        }
    }

    #[test]
    fn later_tokens_do_not_change_earlier_outputs() {
        let mut rng = Rng::new(5);
        let mut net = lively(SMALL, 4);
        let a = random_batch(&SMALL, 2, 6, &mut rng);
        net.forward(&a.tokens, 2, &Edits::none());
        let before: Vec<f32> = (0..SMALL.vocab).flat_map(|v| (0..12).map(move |n| (v, n))).map(|(v, n)| net.logit(v, n)).collect();
        // change the last token of the first sequence and the fourth of the second
        let mut tokens = a.tokens.clone();
        tokens[5] = (tokens[5] + 1) % SMALL.vocab as u32;
        tokens[6 + 3] = (tokens[6 + 3] + 2) % SMALL.vocab as u32;
        net.forward(&tokens, 2, &Edits::none());
        let mut changed = 0;
        for v in 0..SMALL.vocab {
            for n in 0..12 {
                let same = net.logit(v, n) == before[v * 12 + n];
                let may_differ = n == 5 || n >= 6 + 3;
                assert!(same || may_differ, "row {n} changed");
                changed += usize::from(!same);
            }
        }
        assert!(changed > 0);
    }

    #[test]
    fn the_parts_written_to_the_stream_add_up_to_the_logits() {
        let mut rng = Rng::new(8);
        for cfg in [SMALL, Config { mlp: 0, ..SMALL }] {
            let mut net = lively(cfg, 6);
            let batch = random_batch(&cfg, 2, 6, &mut rng);
            net.forward(&batch.tokens, 2, &Edits::none());
            let mut part = vec![0.0; cfg.d];
            for n in 0..12 {
                for token in 0..cfg.vocab {
                    net.token_vector(batch.tokens[n] as usize, &mut part);
                    let mut total = net.read_logit(&part, token);
                    net.position_vector(n % 6, &mut part);
                    total += net.read_logit(&part, token);
                    for l in 0..cfg.layers {
                        for h in 0..cfg.heads {
                            net.head_vector(l, h, n, &mut part);
                            total += net.read_logit(&part, token);
                        }
                        net.mlp_vector(l, n, &mut part);
                        total += net.read_logit(&part, token);
                    }
                    assert!((total - net.logit(token, n)).abs() < 2e-4, "row {n} token {token}: {total} vs {}", net.logit(token, n));
                }
            }
        }
    }

    #[test]
    fn with_every_head_knocked_out_only_the_embeddings_speak() {
        let cfg = Config { mlp: 0, ..SMALL };
        let mut rng = Rng::new(9);
        let mut net = lively(cfg, 7);
        let batch = random_batch(&cfg, 2, 6, &mut rng);
        let mut edits = Edits::none();
        for l in 0..cfg.layers {
            for h in 0..cfg.heads {
                edits = edits.knock(l, h, Knock::Zero);
            }
        }
        net.forward(&batch.tokens, 2, &edits);
        let (mut tok, mut pos) = (vec![0.0; cfg.d], vec![0.0; cfg.d]);
        for n in 0..12 {
            net.token_vector(batch.tokens[n] as usize, &mut tok);
            net.position_vector(n % 6, &mut pos);
            for token in 0..cfg.vocab {
                let want = net.read_logit(&tok, token) + net.read_logit(&pos, token);
                assert!((net.logit(token, n) - want).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn a_mean_knocked_head_says_the_same_everywhere() {
        let mut rng = Rng::new(10);
        let mut net = lively(SMALL, 8);
        let batch = random_batch(&SMALL, 3, 6, &mut rng);
        net.forward(&batch.tokens, 3, &Edits::none());
        let mut average = [0.0f32; SMALL.d];
        let mut part = vec![0.0; SMALL.d];
        for n in 0..18 {
            net.head_vector(1, 0, n, &mut part);
            average.iter_mut().zip(&part).for_each(|(a, p)| *a += p / 18.0);
        }
        net.forward(&batch.tokens, 3, &Edits::none().knock(1, 0, Knock::Mean));
        for n in 0..18 {
            net.head_vector(1, 0, n, &mut part);
            for i in 0..SMALL.d {
                assert!((part[i] - average[i]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn patching_every_position_from_another_run_reproduces_that_run() {
        let mut rng = Rng::new(12);
        let mut net = lively(SMALL, 9);
        let clean = random_batch(&SMALL, 1, 6, &mut rng);
        let other = random_batch(&SMALL, 1, 6, &mut rng);
        net.forward(&clean.tokens, 1, &Edits::none());
        let want: Vec<f32> = (0..SMALL.vocab).flat_map(|v| (0..6).map(move |n| (v, n))).map(|(v, n)| net.logit(v, n)).collect();
        for point in 0..=SMALL.layers {
            net.forward(&clean.tokens, 1, &Edits::none());
            let mut edits = Edits::none();
            let mut x = vec![0.0; SMALL.d];
            for t in 0..6 {
                net.stream(point, t, &mut x);
                edits = edits.replace(point, 0, t, &x);
            }
            net.forward(&other.tokens, 1, &edits);
            for v in 0..SMALL.vocab {
                for n in 0..6 {
                    assert!((net.logit(v, n) - want[v * 6 + n]).abs() < 1e-5, "point {point}");
                }
            }
        }
        // and adding nothing changes nothing
        net.forward(&clean.tokens, 1, &Edits::none().add(1, 0, 2, &[0.0; SMALL.d]));
        assert!((0..SMALL.vocab).all(|v| (0..6).all(|n| net.logit(v, n) == want[v * 6 + n])));
    }

    #[test]
    fn the_tangent_matches_nudging_the_stream() {
        let mut rng = Rng::new(13);
        for cfg in [SMALL, Config { mlp: 0, heads: 1, layers: 3, ..SMALL }] {
            let mut net = lively(cfg, 10);
            let batch = random_batch(&cfg, 1, 6, &mut rng);
            let d = cfg.d;
            for (point, pos) in [(0usize, 0usize), (0, 2), (1, 1), (1, 4), (cfg.layers, 3)] {
                let direction: Vec<f32> = (0..d).map(|_| rng.normal()).collect();
                let exact = net.jvp(&batch.tokens, point, pos, &direction);
                let h = 1e-2;
                let run = |net: &mut Transformer, sign: f32| {
                    let nudge: Vec<f32> = direction.iter().map(|x| x * sign * h).collect();
                    net.forward(&batch.tokens, 1, &Edits::none().add(point, 0, pos, &nudge));
                    let mut all = vec![0.0; 6 * d];
                    for t in 0..6 {
                        net.stream(cfg.layers, t, &mut all[t * d..(t + 1) * d]);
                    }
                    all
                };
                let (up, down) = (run(&mut net, 1.0), run(&mut net, -1.0));
                let mut size = 0.0f32;
                for i in 0..6 * d {
                    let numeric = (up[i] - down[i]) / (2.0 * h);
                    assert!((numeric - exact[i]).abs() < 3e-3 + 0.03 * numeric.abs(), "point {point} pos {pos} entry {i}: {numeric} vs {}", exact[i]);
                    size = size.max(exact[i].abs());
                    if i < pos * d {
                        assert_eq!(exact[i], 0.0, "attention only looks back");
                    }
                }
                assert!(size > 0.1);
            }
        }
    }

    /// The same with interventions in force: a silenced head and a head held at a
    /// fixed vector no longer answer the nudge, and a row that was overwritten
    /// forgets it.
    #[test]
    fn the_tangent_follows_the_interventions_of_the_pass() {
        let mut rng = Rng::new(17);
        let cfg = SMALL;
        let mut net = lively(cfg, 10);
        let batch = random_batch(&cfg, 1, 6, &mut rng);
        let d = cfg.d;
        let mut vector = || (0..d).map(|_| rng.normal()).collect::<Vec<f32>>();
        let (held, written, direction) = (vector(), vector(), vector());
        let edits = Edits::none().knock(0, 1, Knock::Zero).fix(1, 0, &held).replace(1, 0, 4, &written);
        let (point, pos) = (0, 2);
        let plain = net.jvp(&batch.tokens, point, pos, &direction);
        net.forward(&batch.tokens, 1, &edits);
        let mut exact = Vec::new();
        net.tangent(0, point, pos, &direction, &mut exact);
        let h = 1e-2;
        let mut run = |sign: f32| {
            let nudge: Vec<f32> = direction.iter().map(|x| x * sign * h).collect();
            net.forward(&batch.tokens, 1, &edits.clone().add(point, 0, pos, &nudge));
            let mut all = vec![0.0; 6 * d];
            for t in 0..6 {
                net.stream(cfg.layers, t, &mut all[t * d..(t + 1) * d]);
            }
            all
        };
        let (up, down) = (run(1.0), run(-1.0));
        let mut apart = 0.0f32;
        for i in 0..6 * d {
            let numeric = (up[i] - down[i]) / (2.0 * h);
            assert!((numeric - exact[i]).abs() < 3e-3 + 0.03 * numeric.abs(), "entry {i}: {numeric} vs {}", exact[i]);
            apart = apart.max((exact[i] - plain[i]).abs());
        }
        // (the interventions matter here, so ignoring them would have failed above)
        assert!(apart > 0.05, "the interventions changed the tangent by at most {apart}");
    }

    #[test]
    fn the_repeat_task_scores_exactly_the_predictable_positions() {
        let task = Repeat { vocab: 12, len: 20, shortest: 4, longest: 9 };
        let mut rng = Rng::new(2);
        let (mut tokens, mut scored) = (vec![0u32; 20], vec![0.0f32; 20]);
        let mut runs = std::collections::BTreeSet::new();
        for _ in 0..200 {
            task.sample(&mut rng, &mut tokens, &mut scored);
            // the run length is where the first token comes back
            let run = (1..20).find(|&t| tokens[t] == tokens[0]).unwrap();
            runs.insert(run);
            assert!(tokens[..run].iter().collect::<std::collections::BTreeSet<_>>().len() == run, "tokens of a run differ");
            for t in 0..20 {
                assert_eq!(tokens[t], tokens[t % run]);
                // scored exactly when the current token has an earlier copy: only
                // then does what followed that copy tell the next token. (That the
                // next token has been seen before is not enough: at the end of the
                // first run it is the first token only if the run ends there.)
                let predictable = t + 1 < 20 && tokens[..t].contains(&tokens[t]);
                assert_eq!(scored[t] > 0.0, predictable, "position {t}, run {run}");
            }
        }
        assert_eq!(runs.into_iter().collect::<Vec<_>>(), (4..=9).collect::<Vec<_>>());
    }

    #[test]
    fn training_lowers_the_loss() {
        let task = Repeat { vocab: 10, len: 12, shortest: 3, longest: 5 };
        let cfg = Config { vocab: 10, ctx: 12, d: 16, layers: 2, heads: 1, mlp: 0 };
        let mut rng = Rng::new(1);
        let mut net = Transformer::new(cfg, &mut rng);
        let mut batch = Batch::sample(&task, 16, &mut rng);
        let first = net.evaluate(&batch, &Edits::none()).0;
        // an untrained model is about as unsure as the vocabulary is large
        assert!((first - (10.0f32).ln()).abs() < 0.3, "{first}");
        let mut last = 0.0;
        for _ in 0..150 {
            batch.refill(&task, 16, &mut rng);
            last = net.train_step(&batch, 0.01, 0.0).0;
        }
        assert!(last < first - 0.5, "{first} -> {last}");
        assert!(net.params().iter().all(|w| w.is_finite()));
    }

    /// What a training step and a forward pass cost for the shape the scene starts
    /// with (run with `--ignored --nocapture`).
    #[test]
    #[ignore]
    fn cost() {
        let task = Repeat { vocab: 16, len: 20, shortest: 4, longest: 9 };
        let cfg = Config { vocab: 16, ctx: 20, d: 32, layers: 2, heads: 1, mlp: 0 };
        let mut rng = Rng::new(1);
        let mut net = Transformer::new(cfg, &mut rng);
        let mut b = Batch::sample(&task, 16, &mut rng);
        let start = std::time::Instant::now();
        for _ in 0..2000 {
            b.refill(&task, 16, &mut rng);
            net.train_step(&b, 0.005, 0.0);
        }
        let step = start.elapsed().as_secs_f64() / 2000.0;
        let one = Batch::sample(&task, 1, &mut rng);
        let start = std::time::Instant::now();
        for _ in 0..5000 {
            net.forward(&one.tokens, 1, &Edits::none());
        }
        let forward = start.elapsed().as_secs_f64() / 5000.0;
        let direction = vec![1.0; 32];
        let mut out = Vec::new();
        let start = std::time::Instant::now();
        for _ in 0..5000 {
            net.tangent(0, 0, 3, &direction, &mut out);
        }
        let tangent = start.elapsed().as_secs_f64() / 5000.0;
        println!(
            "{} weights; training step (16 rows of 20): {:.2} ms = {:.0} steps/s; forward pass of one row: {:.0} us; one tangent from the embeddings: {:.0} us",
            net.n_params(),
            step * 1e3,
            1.0 / step,
            forward * 1e6,
            tangent * 1e6
        );
    }

    /// How fast, and whether, various shapes learn the repeat task (run with
    /// `--ignored --nocapture`).
    #[test]
    #[ignore]
    #[allow(clippy::type_complexity)]
    fn explore() {
        // (d, heads, batch, lr, vocab, len, shortest, longest, decay)
        let trials: &[(usize, usize, usize, f32, usize, usize, usize, usize, f32)] = &[
            (32, 1, 16, 0.01, 16, 20, 4, 9, 0.0),
            (32, 2, 16, 0.01, 16, 20, 4, 9, 0.0),
            (32, 1, 32, 0.01, 16, 20, 4, 9, 0.0),
            (32, 2, 32, 0.01, 16, 20, 4, 9, 0.0),
            (24, 1, 16, 0.01, 16, 20, 4, 9, 0.0),
            (16, 1, 16, 0.01, 16, 20, 4, 9, 0.0),
            (32, 1, 16, 0.003, 16, 20, 4, 9, 0.0),
            (32, 1, 16, 0.02, 16, 20, 4, 9, 0.0),
            (32, 1, 16, 0.01, 12, 16, 3, 7, 0.0),
        ];
        for &(d, heads, batch, lr, vocab, len, shortest, longest, decay) in trials {
            for seed in 0..3u64 {
                let task = Repeat { vocab, len, shortest, longest };
                let cfg = Config { vocab, ctx: len, d, layers: 2, heads, mlp: 0 };
                let mut rng = Rng::new(100 + seed);
                let mut net = Transformer::new(cfg, &mut rng);
                let mut b = Batch::sample(&task, batch, &mut rng);
                let test = Batch::sample(&task, 64, &mut Rng::new(7));
                let start = std::time::Instant::now();
                let (mut reached, mut trace) = (None, String::new());
                let total = 6000;
                for step in 0..total {
                    b.refill(&task, batch, &mut rng);
                    net.train_step(&b, lr, decay);
                    if (step + 1) % 250 == 0 {
                        let (loss, acc) = net.evaluate(&test, &Edits::none());
                        if (step + 1) % 500 == 0 {
                            trace += &format!(" {loss:.2}");
                        }
                        if acc > 0.97 && reached.is_none() {
                            reached = Some(step + 1);
                        }
                    }
                }
                let ms = start.elapsed().as_secs_f64() * 1000.0 / total as f64;
                println!("d={d} h={heads} B={batch} lr={lr} V={vocab} T={len} run {shortest}-{longest} seed {seed}: {ms:.2} ms/step, 97% at {reached:?}; loss{trace}");
            }
        }
    }

    #[test]
    fn decay_shrinks_the_weights_and_spares_the_biases() {
        let cfg = Config { mlp: 6, ..SMALL };
        let mut net = Transformer::new(cfg, &mut Rng::new(2));
        net.params_mut().iter_mut().for_each(|w| *w = 1.0);
        net.g.fill(0.0);
        // lr * decay = 1: every weight goes to exactly zero, every bias stays put
        net.adam(0.5, 2.0);
        let biases: usize = net.at.layers.iter().map(|l| l.b_in.len() + l.b_out.len()).sum();
        assert!(biases > 0);
        assert_eq!(net.params().iter().filter(|&&w| w == 1.0).count(), biases);
        assert_eq!(net.params().iter().filter(|&&w| w == 0.0).count(), net.n_params() - biases);
        // and a decay stronger than that cannot do more
        net.params_mut().iter_mut().for_each(|w| *w = 1.0);
        net.adam(1.0, 5.0);
        assert!(net.params().iter().all(|w| w.is_finite() && *w >= 0.0));
    }

    #[test]
    fn a_fixed_head_writes_the_given_vector_everywhere() {
        let mut rng = Rng::new(8);
        let mut net = Transformer::new(SMALL, &mut rng);
        let task = Repeat { vocab: SMALL.vocab, len: SMALL.ctx, shortest: 2, longest: 3 };
        let batch = Batch::sample(&task, 3, &mut rng);
        net.forward(&batch.tokens, batch.count, &Edits::none());
        let mut mean = vec![0.0f32; SMALL.d];
        net.head_mean(0, 1, &mut mean);
        // fixing the head at its own batch average is the same as knocking it out by its average
        net.forward(&batch.tokens, batch.count, &Edits::none().knock(0, 1, Knock::Mean));
        let by_average = net.logit(1, 4);
        net.forward(&batch.tokens, batch.count, &Edits::none().fix(0, 1, &mean));
        assert!((net.logit(1, 4) - by_average).abs() < 1e-4);
        let mut written = vec![0.0f32; SMALL.d];
        for row in [0, 5, batch.count * SMALL.ctx - 1] {
            net.head_vector(0, 1, row, &mut written);
            assert!(written.iter().zip(&mean).all(|(a, b)| (a - b).abs() < 1e-6));
        }
        // and, unlike the average, it does not change with what else is in the batch
        net.forward(&batch.tokens[..SMALL.ctx], 1, &Edits::none().fix(0, 1, &mean));
        let alone = net.logit(1, 4);
        net.forward(&batch.tokens, batch.count, &Edits::none().fix(0, 1, &mean));
        assert!((net.logit(1, 4) - alone).abs() < 1e-5);
    }
}

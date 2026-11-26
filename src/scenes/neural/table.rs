//! The playground's second task: learning an arithmetic table, with part of it held
//! out. The network sees a pair (a, b) as two one-hot codes and must name the answer.

use super::arith::{self, Defaults, Score, Task, MODULI, OPS};
use super::session::Problem;
use super::training::Training;
use super::*;
use crate::clock;

const MAX_HIDDEN: usize = 2;
const MAX_WIDTH: usize = 128;
const VIEWS: [&str; 3] = ["answers", "right or wrong", "truth"];

/// Colours of the "right or wrong" view: trained-on right, held-out right,
/// trained-on wrong, held-out wrong.
const VERDICT: [Color; 4] = [Color::hex(0x1E5C45), Color::hex(0x2EE6A6), Color::hex(0x6B2630), Color::hex(0xFF5D5D)];

pub struct Table {
    kind: usize,
    modulus: usize,
    fraction: f32,
    split: u64,
    /// Scale of the initial weights relative to the usual choice.
    gain: f32,
    task: Task,
    t: Training,
    rng: Rng,
    /// Every picture, side by side: tile 0 is the big table, then one tile per
    /// neuron (layer by layer, the outputs last), each p by p pixels.
    atlas: Image,
    atlas_cols: usize,
    view: usize,
    /// The neuron under the pointer: (layer, index), the outputs being the last layer.
    hover: Option<(usize, usize)>,
    pred: Vec<u8>,
    score: Score,
    /// Each first-layer neuron's shares of the frequencies, and the picture of them.
    spectra: Vec<f32>,
    order: Vec<u32>,
    freq_map: Image,
    concentration: f32,
    stale: bool,
    /// For each frequency 1..=p/2: where the first layer puts the numbers 0..p along
    /// it (see `arith::ring`), and how strong that wave is.
    rings: Vec<(Vec<Vec2>, f32)>,
    /// For each frequency, the share of first-layer neurons devoted to it (see
    /// `arith::devoted`): how brightly its ring is drawn.
    devoted: Vec<f32>,
    /// Frequencies removed from the first layer as the network is drawn and scored
    /// (training goes on with all of them).
    cut: Vec<bool>,
    /// The score with those frequencies out.
    cut_score: Score,
    /// Right answers over the whole table with each frequency alone removed, the
    /// training step each figure was worked out at (one is brought up to date on
    /// every third redraw while training, and on every redraw while paused), and the
    /// right answers with every frequency in place at that same step.
    alone: Vec<f32>,
    alone_at: Vec<u64>,
    alone_base: Vec<f32>,
    next_alone: usize,
    alone_turn: u32,
    /// The frequency whose ring is under the pointer.
    hover_ring: Option<usize>,
    scratch: Vec<u8>,
}

/// Where the row of rings (one per frequency) sits: left edge, right edge, the height
/// of the ring centres.
const RINGS: (f32, f32, f32) = (-3.75, 4.1, -2.7);

/// A ring's "without it" figure is shown only while it still describes the network:
/// while the share of the table right has moved less than this since it was worked
/// out, and it is from the last quarter of the run (networks change fast early on,
/// and more and more slowly later).
const ALONE_FRESH: f32 = 0.03;

/// The view that holds everything this task draws.
pub const VIEW: View = View::Fit { center: Vec2::new(0.15, -0.45), half: Vec2::new(4.15, 2.95) };

/// A group of tiles on screen: one layer's neurons.
#[derive(Clone, Copy, Default)]
struct Block {
    count: usize,
    /// Top-left corner of the grid of tiles, tiles per row, and the grid's pitch.
    corner: Vec2,
    cols: usize,
    pitch: f32,
    /// Left edge and width of the space the block was given (for its title).
    left: f32,
}

impl Block {
    /// Bottom-left corner and side of tile `j`.
    fn tile(&self, j: usize) -> (Vec2, f32) {
        let (col, row) = (j % self.cols, j / self.cols);
        let side = self.pitch * 0.92;
        (self.corner + Vec2::new(col as f32 * self.pitch, -(row as f32 + 1.0) * self.pitch) + Vec2::splat(self.pitch * 0.04), side)
    }

    /// The tile under `m`, if any.
    fn pick(&self, m: Vec2) -> Option<usize> {
        let q = (m - self.corner) * Vec2::new(1.0, -1.0) / self.pitch;
        if q.x < 0.0 || q.y < 0.0 || q.x >= self.cols as f32 {
            return None;
        }
        let j = q.y as usize * self.cols + q.x as usize;
        (j < self.count).then_some(j)
    }
}

impl Table {
    pub fn new(gfx: &Gfx) -> Self {
        // Settings under which the network memorises the cells it is shown within
        // about a hundred steps and generalises to the rest over the next few
        // thousand; a test in `arith.rs` holds them to that.
        let d = Defaults::default();
        let mut t = Training::new(d.hidden, d.width);
        t.act = Act::ALL.iter().position(|&a| a == d.act).unwrap_or(0);
        t.lr = d.lr;
        t.decay = d.decay;
        t.batch = usize::MAX;
        t.rate = 1200;
        let (kind, modulus, fraction, split) = (d.kind, MODULI.iter().position(|&m| m == d.p).unwrap_or(0), d.fraction, d.split);
        let task = Task::new(kind, MODULI[modulus], fraction, split);
        let mut rng = Rng::new(d.seed);
        t.load(Mlp::new(&task.sizes(t.hidden, t.width), t.act(), &mut rng), Problem::Table(task.clone()));
        Table {
            kind,
            modulus,
            fraction,
            split,
            gain: 1.0,
            task,
            t,
            rng,
            atlas: Image::new(gfx, 1, 1),
            atlas_cols: 1,
            view: 0,
            hover: None,
            pred: Vec::new(),
            score: Score::default(),
            spectra: Vec::new(),
            order: Vec::new(),
            freq_map: Image::new(gfx, 1, 1),
            concentration: 0.0,
            stale: true,
            rings: Vec::new(),
            devoted: Vec::new(),
            cut: Vec::new(),
            cut_score: Score::default(),
            alone: Vec::new(),
            alone_at: Vec::new(),
            alone_base: Vec::new(),
            next_alone: 0,
            alone_turn: 0,
            hover_ring: None,
            scratch: Vec::new(),
        }
    }

    pub fn running(&self) -> bool {
        self.t.busy()
    }

    fn rebuild(&mut self) {
        let mut net = Mlp::new(&self.task.sizes(self.t.hidden, self.t.width), self.t.act(), &mut self.rng);
        net.scale_weights(self.gain);
        self.t.load(net, Problem::Table(self.task.clone()));
        // a new network: nothing has been removed from it
        self.cut.clear();
        self.alone_at.clear();
        self.stale = true;
    }

    /// One of the "things to try": the default run with a table, a share of it to
    /// train on, and a weight decay of the visitor's choosing.
    fn set_up(&mut self, kind: usize, fraction: f32, decay: f32) {
        let d = Defaults::default();
        (self.kind, self.fraction, self.split, self.gain, self.view) = (kind, fraction, d.split, 1.0, 0);
        self.modulus = MODULI.iter().position(|&m| m == d.p).unwrap_or(0);
        (self.t.hidden, self.t.width, self.t.lr, self.t.decay, self.t.optim) = (d.hidden, d.width, d.lr, decay, 5);
        self.t.act = Act::ALL.iter().position(|&a| a == d.act).unwrap_or(0);
        (self.t.batch, self.t.running) = (usize::MAX, true);
        self.rng = Rng::new(d.seed);
        self.task = Task::new(self.kind, MODULI[self.modulus], self.fraction, self.split);
        self.rebuild();
    }

    /// Right answers over the whole table.
    fn overall(&self, score: &Score) -> f32 {
        (score.accuracy[0] * self.task.train.len() as f32 + score.accuracy[1] * self.task.test.len() as f32) / self.task.cells().max(1) as f32
    }

    /// Centre and radius of frequency `k`'s ring (k counted from 0) in the row of rings.
    fn ring_place(&self, k: usize) -> (Vec2, f32) {
        let count = (self.task.p / 2).max(1) as f32;
        let pitch = ((RINGS.1 - RINGS.0) / count).min(1.3);
        let left = (RINGS.0 + RINGS.1 - pitch * count) * 0.5;
        (Vec2::new(left + (k as f32 + 0.5) * pitch, RINGS.2), (pitch * 0.38).min(0.4))
    }

    /// A new table or a new split: the old weights answer a different question.
    fn retask(&mut self) {
        // (a full batch stays a full batch when the number of training pairs
        // changes: `Training` keeps the batch as it was asked for)
        self.task = Task::new(self.kind, MODULI[self.modulus], self.fraction, self.split);
        self.rebuild();
    }

    /// Where each layer's tiles go: side by side across the top right of the view.
    fn blocks(&self) -> ([Block; MAX_HIDDEN + 1], usize) {
        let n = self.t.net().layers();
        let (x0, x1, top, bottom, gap) = (0.3f32, 4.1f32, 1.98f32, -0.72f32, 0.22f32);
        let counts = &self.t.net().sizes[1..];
        // wider layers get more room, but no layer is squeezed out
        let largest = counts.iter().map(|&c| (c as f32).sqrt()).fold(0.0, f32::max);
        let weight = |c: usize| (c as f32).sqrt().max(0.45 * largest);
        let total: f32 = counts.iter().map(|&c| weight(c)).sum();
        let room = x1 - x0 - gap * (n - 1) as f32;
        let mut out = [Block::default(); MAX_HIDDEN + 1];
        let mut left = x0;
        for (block, &count) in out.iter_mut().zip(counts) {
            let (bw, bh) = (room * weight(count) / total, top - bottom);
            // the tiling with the largest tiles
            let (mut cols, mut pitch) = (1, 0.0f32);
            for c in 1..=count {
                let s = (bw / c as f32).min(bh / count.div_ceil(c) as f32);
                if s > pitch {
                    (cols, pitch) = (c, s);
                }
            }
            let pitch = pitch.min(0.62);
            *block = Block { count, corner: Vec2::new(left + (bw - cols as f32 * pitch) * 0.5, top), cols, pitch, left };
            left += bw + gap;
        }
        (out, n)
    }

    /// Recompute the scores and every picture from the current weights.
    fn refresh(&mut self, gfx: &Gfx) {
        let p = self.task.p;
        let k_max = p / 2;
        if self.cut.len() != k_max || self.alone_at.len() != k_max {
            self.cut = vec![false; k_max];
            self.alone = vec![0.0; k_max];
            self.alone_at = vec![u64::MAX; k_max];
            self.alone_base = vec![0.0; k_max];
            self.next_alone = 0;
        }
        // what one frequency is worth: the score with it alone removed. One frequency
        // is brought up to date per redraw, so the cost stays small.
        let steps = self.t.steps;
        self.alone_turn = (self.alone_turn + 1) % 3;
        let due = (0..k_max).map(|i| (self.next_alone + i) % k_max).find(|&k| self.alone_at[k] != steps);
        // (every third redraw while training; at once for a figure never worked out)
        let updated = due.filter(|&k| self.alone_turn == 0 || self.alone_at[k] == u64::MAX || !self.t.running);
        if let Some(k) = updated {
            let (task, scratch) = (&self.task, &mut self.scratch);
            let score = self.t.with_edited(|net| arith::remove_frequency(net, p, k + 1), |net| arith::evaluate(net, task, scratch));
            self.alone[k] = self.overall(&score);
            self.alone_at[k] = steps;
            self.next_alone = (k + 1) % k_max;
        }

        // the network as it is trained: its scores are what the charts follow
        self.score = arith::evaluate(self.t.net_mut(), &self.task, &mut self.pred);
        self.t.record(self.score.loss, self.score.accuracy);
        if let Some(k) = updated {
            // what that figure is to be compared with: the score at the same step
            self.alone_base[k] = self.overall(&self.score);
        }
        self.rings.clear();
        self.rings.extend((1..=k_max).map(|k| arith::ring(self.t.net(), p, k)));
        arith::spectra(self.t.net(), p, &mut self.spectra);
        arith::devoted(&self.spectra, k_max, &mut self.devoted);

        // the network as it is drawn: with the removed frequencies out
        self.cut_score = self.score;
        if self.cut.iter().any(|&c| c) {
            let (task, pred, cut) = (&self.task, &mut self.pred, &self.cut);
            self.cut_score = self
                .t
                .with_edited(|net| (0..k_max).filter(|&k| cut[k]).for_each(|k| arith::remove_frequency(net, p, k + 1)), |net| arith::evaluate(net, task, pred));
        }

        let net = self.t.net();
        let layers = net.layers();
        let tiles = 1 + net.sizes[1..].iter().sum::<usize>();
        let cols = (tiles as f32).sqrt().ceil() as usize;
        self.atlas.ensure_size(gfx, (cols * p) as u32, (tiles.div_ceil(cols) * p) as u32);
        self.atlas_cols = cols;
        let pixels = self.atlas.pixels_mut();
        let mut tile = 1;
        for l in 0..layers {
            for j in 0..net.sizes[l + 1] {
                paint_tile(pixels, cols, p, tile, net.neuron(l, j));
                tile += 1;
            }
        }
        self.paint_big();

        // which frequency each first-layer neuron uses: one column per neuron (sorted
        // by its strongest frequency), one row per frequency
        let neurons = self.t.net().sizes[1];
        self.concentration = arith::concentration(&self.spectra, k_max);
        let share = |j: usize| &self.spectra[j * k_max..(j + 1) * k_max];
        let strongest = |j: usize| (0..k_max).fold(0, |best, k| if share(j)[k] > share(j)[best] { k } else { best });
        self.order.clear();
        self.order.extend(0..neurons as u32);
        self.order.sort_by(|&a, &b| {
            let (ka, kb) = (strongest(a as usize), strongest(b as usize));
            ka.cmp(&kb).then(share(b as usize)[kb].total_cmp(&share(a as usize)[ka]))
        });
        self.freq_map.ensure_size(gfx, neurons as u32, k_max.max(1) as u32);
        let glow = Cmap::Magma.table();
        let pixels = self.freq_map.pixels_mut();
        for (x, &j) in self.order.iter().enumerate() {
            for k in 0..k_max {
                pixels[k * neurons + x] = glow[table_index(share(j as usize)[k])];
            }
        }
    }

    /// Paint the big table (tile 0): the neuron under the pointer, or else the chosen
    /// view of the answers, from what the last recomputation left. False if that is
    /// not to be had: the network's activations are no longer those over the table.
    fn paint_big(&mut self) -> bool {
        let (p, cols) = (self.task.p, self.atlas_cols);
        let net = self.t.net();
        let pixels = self.atlas.pixels_mut();
        if pixels.len() < cols * p * p {
            return false;
        }
        match self.hover {
            Some((l, j)) => {
                let values = if l < net.layers() && j < net.sizes[l + 1] { net.neuron(l, j) } else { &[] };
                if values.len() != p * p {
                    return false;
                }
                paint_tile(pixels, cols, p, 0, values);
            }
            None => {
                if self.pred.len() != p * p {
                    return false;
                }
                let classes = if arith::cyclic(self.task.kind) { Cmap::Twilight } else { Cmap::Viridis }.table();
                let class = |c: u8| classes[table_index((c as f32 + 0.5) / p as f32)];
                for s in 0..p * p {
                    let (right, held_out) = (self.pred[s] == self.task.answer[s], self.task.held_out[s]);
                    pixels[(s / p) * cols * p + s % p] = match self.view {
                        0 => class(self.pred[s]),
                        1 => VERDICT[(!right) as usize * 2 + held_out as usize],
                        _ => class(self.task.answer[s]),
                    };
                }
            }
        }
        true
    }

    /// Draw frequency `k`'s ring (k counted from 0): frequency k's part of the
    /// embedding, the numbers 0..p where the first layer puts them along that
    /// frequency, joined in order (see `arith::ring`). It is drawn to a scale on which
    /// a pure wave puts number `a` at angle 2 pi k a / p on the unit circle (drawn
    /// faint), a regular star; but nearly any layer comes out close to that star,
    /// random ones included. What says whether the layer uses the frequency is the
    /// share of its neurons devoted to it, and that sets how brightly the ring is drawn.
    fn draw_ring(&self, f: &mut Frame, k: usize, centre: Vec2, radius: f32, labelled: bool) {
        let (points, _) = &self.rings[k];
        let p = points.len();
        // full strength when the frequency has its fair share of the neurons (every
        // neuron devoted to one frequency, all of them used alike), faint when it has
        // next to none, as with random weights
        let weight = (self.devoted.get(k).copied().unwrap_or(0.0) * self.rings.len() as f32).clamp(0.15, 1.0);
        f.circle(centre.extend(0.0), Vec3::Z, radius, 1.0, pal::FAINT);
        let place = |q: Vec2| centre + q.clamp_length_max(1.45) * radius;
        let path: Vec<Vec3> = points.iter().map(|&q| place(q).extend(0.005)).collect();
        if p >= 3 {
            f.dl.lines.closed(&path, if labelled { 1.5 } else { 1.0 }, pal::DIM.a(0.75 * weight));
        }
        let colours = Cmap::Twilight.table();
        f.dl.discs.style.outline_px = 0.0;
        for (a, &q) in points.iter().enumerate() {
            let at = place(q);
            let colour = colours[table_index((a as f32 + 0.5) / p as f32)];
            f.dot(at.extend(0.01), if labelled { 8.0 } else { 4.5 }, colour.a(weight));
            if labelled {
                let out = (at - centre).normalize_or_zero();
                f.text_off(at, Vec2::new(out.x, -out.y) * 13.0, a.to_string(), 11.5, pal::TEXT, Align2::CENTER_CENTER);
            }
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui::note(ui, "A network learning an arithmetic table. Every pair $(a, b)$ is one cell, and some cells are never shown to it. The big table is its answer for every pair; each small tile is one neuron over the same table (hover one to enlarge it). The default run shows grokking: the network first memorises the cells it trains on, and only much later finds the rule, at which point the held-out cells come right.");
        if self.t.ui_run(ui) {
            self.rebuild();
        }

        ui::section(ui, "Things to try");
        if ui::experiment(ui, "Memorise first, understand later", "Within about a hundred steps every trained-on cell is right (cyan) and every held-out cell wrong (orange). Over the next few thousand steps the orange curve climbs. Meanwhile the first-layer neurons each settle on a single frequency and the rings at the bottom brighten: half of the neurons within the first hundred steps, all of them by the time most held-out cells are right.") {
            self.set_up(0, 0.8, 1.0);
        }
        if ui::experiment(ui, "Without weight decay", "It memorises and stays there: even after twenty thousand steps the held-out cells are still wrong, and only about half of the first-layer neurons have settled on a single frequency (with weight decay, all of them do), so the rings stay dimmer. The pull towards small weights is what makes the rule win over the memorised table.") {
            self.set_up(0, 0.8, 0.0);
        }
        if ui::experiment(ui, "Only half the table", "Too few examples to pin the rule down: after thousands of steps the held-out cells are still wrong.") {
            self.set_up(0, 0.5, 1.0);
        }
        if ui::experiment(ui, "Multiplication", "It comes right too, but the neurons do not settle on single frequencies, so the rings stay faint: multiplying the numbers other than $0$ mod a prime $p$ is adding, mod $p - 1$, their places in a different ordering, and the waves run along that ordering.") {
            self.set_up(2, 0.8, 1.0);
        }
        if ui::experiment(
            ui,
            "A table with no surprise",
            r"$\max(a, b)$: nearby numbers have nearby answers, so the held-out cells are right almost at once. No two phases, and no waves needed.",
        ) {
            self.set_up(5, 0.8, 1.0);
        }
        ui::note(ui, "Once $a + b$ has come right, click the rings at the bottom to take frequencies out of the network. One alone costs little, because the answer is a vote of all of them; each further one costs more, and with all of them gone the network is guessing.");

        ui::section(ui, "Table");
        let tables: Vec<String> = OPS.iter().map(|op| format!("${op}$")).collect();
        let tables: Vec<&str> = tables.iter().map(String::as_str).collect();
        let mut changed = ui::choice(ui, &tables, &mut self.kind);
        let p = MODULI[self.modulus];
        ui::note(ui, &format!("All arithmetic is mod {p}, so there are {p} possible answers."));
        let names: Vec<String> = MODULI.iter().map(|m| format!("$p = {m}$")).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        changed |= ui::choice(ui, &names, &mut self.modulus);
        changed |= ui::slider(ui, "fraction of the table trained on", &mut self.fraction, 0.2..=0.95);
        if ui.button("new split").clicked() {
            self.split += 1;
            changed = true;
        }
        if changed {
            self.retask();
        }
        match self.kind {
            0 | 1 => ui::note(ui, r"How it does it (the circuit it has found): the first layer places each number on circles, one circle per frequency $k$, with number $a$ at angle $2\pi k a / p$ (the rings at the bottom). Adding two numbers is then adding two angles, which a layer of neurons can do, and the output reads the answer off the summed angle. The strip under the table shows each first-layer neuron settling on a single frequency (one bright cell per column) as the held-out cells come right."),
            2 => ui::note(ui, "Multiplication mod a prime $p$ is, for the numbers other than $0$, addition mod $p - 1$ in disguise (relabel each number by its discrete logarithm), so it is learned the same way; but its waves run along that relabelling and do not show in the spectrum here."),
            5 | 6 => ui::note(ui, "Not modular: nearby numbers have nearby answers, so no waves are needed."),
            _ => {}
        }

        ui::section(ui, "The big table shows");
        if ui::choice(ui, &VIEWS, &mut self.view) {
            self.stale = true;
        }
        match self.view {
            0 => ui::note(ui, "One colour per answer. White dots mark wrong answers; small dark dots mark the pairs held out of training."),
            1 => ui::note(ui, "Green is right and red is wrong; the bright cells are the held-out pairs."),
            _ => ui::note(ui, "What a perfect network would show."),
        }

        ui::section(ui, "Network");
        let mut arch = self.t.ui_network(ui, MAX_HIDDEN, MAX_WIDTH);
        arch |= ui::slider_log(ui, "initial weight scale", &mut self.gain, 0.1..=4.0);
        arch |= self.t.ui_activation(ui);
        if arch {
            self.rebuild();
        }
        self.t.ui_optimizer(ui);
        self.t.ui_training(ui, self.task.train.len());
        ui::note(ui, "Weight decay is what turns memorising into generalising here: without it the held-out cells stay wrong.");

        ui::section(ui, "Frequencies removed");
        let removed = self.cut.iter().filter(|&&c| c).count();
        ui::readout(ui, "removed from the first layer", format!("{removed} of {}", self.cut.len()));
        if removed > 0 {
            ui::readout(ui, "right answers with them out", format!("{:.1} %", self.overall(&self.cut_score) * 100.0));
            if ui.button("put them all back").clicked() {
                self.cut.iter_mut().for_each(|c| *c = false);
                self.stale = true;
            }
            ui::note(ui, "Training goes on with every frequency in place; removing them changes only how the network is read here.");
        } else {
            ui::note(ui, "Click a ring at the bottom of the picture to remove its frequency.");
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::CYAN, "cyan curves and numbers: the pairs it trains on"),
                (pal::ORANGE, "orange: the pairs held out of training, which it never sees"),
                (
                    Cmap::CoolWarm.table()[230],
                    "small tiles: one neuron each over the whole table, red where its value is high, blue where it is below zero, each on its own scale",
                ),
                (Cmap::Magma.table()[200], "the strip under the table: a column per first-layer neuron, bright at the frequency it uses"),
                (Cmap::Twilight.table()[60], "dots on a ring: the numbers $0$ to $p - 1$, each in its own colour"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(
            ui,
            "Held-out pairs:",
            "cells of the table the network is never trained on. Getting them right means it has found the rule, not just stored the answers.",
        );
        ui::term(ui, "Grokking:", "generalising long after memorising: the held-out answers come right thousands of steps after the trained-on ones.");
        ui::term(ui, "Frequency $k$:", "a wave that goes round $k$ times as the number goes from $0$ to $p$. The network's solution is built from such waves.");
        ui::term(
            ui,
            "Embedding:",
            "where the first layer puts a number: the weights from that number's input to each first-layer neuron. Each ring at the bottom is one frequency's part of it, drawn the brighter the more neurons are devoted to that frequency (give it most of their weight).",
        );
        ui::term(ui, "Weight decay:", "a steady pull of every weight towards zero during training.");

        ui::section(ui, "Numbers");
        self.t.ui_numbers(ui);
        ui::readout(ui, "pairs trained on / held out", format!("{} / {}", self.task.train.len(), self.task.test.len()));
        ui::readout(ui, "frequency concentration", format!("{:.2}", self.concentration));
        let random = arith::random_concentration(self.task.p).map(|c| format!(" Random weights give about {c:.2}.")).unwrap_or_default();
        ui::note(
            ui,
            &format!(
                "Frequency concentration: the average share a first-layer neuron gives its strongest frequency. 1 means every neuron is a single wave.{random}"
            ),
        );
        ui::readout_colored(ui, "right, trained-on pairs", format!("{:.1} %", self.score.accuracy[0] * 100.0), pal::CYAN);
        ui::readout_colored(ui, "right, held-out pairs", format!("{:.1} %", self.score.accuracy[1] * 100.0), pal::ORANGE);
        ui::readout_colored(ui, "loss, trained-on pairs", loss_number(self.score.loss[0]), pal::CYAN);
        ui::readout_colored(ui, "loss, held-out pairs", loss_number(self.score.loss[1]), pal::ORANGE);
        ui::see_also(
            ui,
            &[("mlp", "Network playground"), ("superposition", "Superposition"), ("transformer", "A transformer, opened up"), ("dft", "The Fourier basis")],
        );
    }

    pub fn frame(&mut self, f: &mut Frame) {
        self.t.advance(f.dt, f.real_dt);
        let (blocks, n_blocks) = self.blocks();
        let hover = f.mouse().and_then(|m| blocks[..n_blocks].iter().enumerate().find_map(|(l, b)| b.pick(m).map(|j| (l, j))));
        let k_max = self.task.p / 2;
        self.hover_ring = f.mouse().and_then(|m| {
            (0..k_max.min(self.rings.len())).find(|&k| {
                let (centre, radius) = self.ring_place(k);
                (m - centre).abs().max_element() < radius * 1.15
            })
        });
        if let Some(k) = self.hover_ring.filter(|_| f.clicked_canvas()) {
            if let Some(c) = self.cut.get_mut(k) {
                *c = !*c;
                self.stale = true;
            }
        }
        // Only the pointer moved: the big table is repainted from what the last
        // recomputation left, if the weights are still those it was made from.
        // (Otherwise everything is recomputed, as it would be for a new step.)
        let moved = hover != self.hover;
        self.hover = hover;
        let repainted = moved && !self.stale && self.t.drawn_is_current() && self.paint_big();
        if self.t.refresh_due(self.stale || (moved && !repainted)) {
            let start = clock::now();
            self.refresh(f.gfx);
            self.t.refreshed(clock::now() - start);
            self.stale = false;
        }

        let p = self.task.p;
        let layers = self.t.net().layers();
        let tile_part = |tile: usize, cols: usize| [(tile % cols) * p, (tile / cols) * p, p, p].map(|v| v as u32);

        // the big table
        let (min, side) = (Vec2::new(-3.75, -1.0), 3.05);
        let max = min + Vec2::splat(side);
        let big_ring = self.hover_ring.filter(|&k| k < self.rings.len());
        match big_ring {
            Some(k) => {
                inset(f, min, max);
                self.draw_ring(f, k, (min + max) * 0.5, side * 0.34, true);
            }
            None => f.image_part(&mut self.atlas, tile_part(0, self.atlas_cols), min, max, -0.01),
        }
        frame_rect(f, min, max, 1.4, pal::DIM);
        if self.hover.is_none() && big_ring.is_none() && self.view == 0 && self.pred.len() == p * p {
            // logical pixels per world unit, to size the marks to the cells
            let unit = match (f.cam.project(Vec3::ZERO), f.cam.project(Vec3::X)) {
                (Some(a), Some(b)) => (b - a).length() / f.px,
                _ => 100.0,
            };
            let cell = side / p as f32;
            f.dl.discs.style.outline_px = 0.8;
            for s in 0..p * p {
                let centre = min + Vec2::new((s % p) as f32 + 0.5, (s / p) as f32 + 0.5) * cell;
                if self.pred[s] != self.task.answer[s] {
                    f.dot(centre, (0.4 * cell * unit).max(2.5), pal::WHITE);
                }
                if self.task.held_out[s] {
                    f.dot(centre + Vec2::splat(0.3 * cell), (0.2 * cell * unit).max(1.6), Color::hex(0x0A0D16));
                }
            }
        }
        let removed = self.cut.iter().filter(|&&c| c).count();
        let title = match self.hover {
            _ if big_ring.is_some() => format!("frequency {}'s part of the embedding", big_ring.unwrap_or(0) + 1),
            Some((l, j)) if l + 1 < layers => format!("layer {}, neuron {}: its activity for every pair", l + 1, j + 1),
            Some((_, j)) => format!("the output for answer {j}: how strongly each pair votes for it"),
            None if removed > 0 && self.view != 2 => {
                format!("the network's answers with {removed} {} removed", if removed == 1 { "frequency" } else { "frequencies" })
            }
            None => [
                format!(r"the network's answer to every pair:  ${} \;\mathrm{{mod}}\; {p}$", OPS[self.kind]),
                "right or wrong, pair by pair".to_string(),
                format!(r"the true table:  ${} \;\mathrm{{mod}}\; {p}$", OPS[self.kind]),
            ][self.view]
                .clone(),
        };
        f.text(Vec2::new(min.x, max.y + 0.06), title, 14.0, pal::TEXT, Align2::LEFT_BOTTOM);
        match big_ring {
            Some(k) => {
                let neurons = self.t.net().sizes[1];
                let devoted = (self.devoted.get(k).copied().unwrap_or(0.0) * neurons as f32).round() as usize;
                f.text(
                    Vec2::new(min.x, min.y - 0.05),
                    format!("{devoted} of the {neurons} first-layer neurons give this frequency most of their weight"),
                    12.0,
                    pal::DIM,
                    Align2::LEFT_TOP,
                )
            }
            None => {
                f.text(Vec2::new((min.x + max.x) * 0.5, min.y - 0.05), "$a$ →", 12.0, pal::DIM, Align2::CENTER_TOP);
                f.text(Vec2::new(min.x - 0.06, (min.y + max.y) * 0.5), "$b$ ↑", 12.0, pal::DIM, Align2::RIGHT_CENTER);
            }
        }

        // how the first layer's weights divide among frequencies
        let origin = Vec2::new(min.x, -1.85);
        let size = Vec2::new(side, 0.42);
        let what = match self.hover {
            Some((0, j)) if k_max > 0 => {
                // the hovered neuron alone: its share of each frequency, as bars
                let heights: Vec<f32> = self.spectra[j * k_max..(j + 1) * k_max].iter().map(|v| v * size.y).collect();
                inset(f, origin - Vec2::splat(0.05), origin + size + Vec2::splat(0.05));
                f.line(origin, origin + Vec2::new(size.x, 0.0), 1.0, pal::DIM.a(0.8));
                bars(f, origin.x, size.x / k_max as f32, &heights, origin.y, pal::PURPLE);
                format!("neuron {}: how its input weights divide among frequencies 1 to {k_max}", j + 1)
            }
            _ => {
                f.image(&mut self.freq_map, origin, origin + size, 0.0);
                frame_rect(f, origin, origin + size, 1.0, pal::FAINT);
                format!("frequency (1 to {k_max}, upwards) used by each first-layer neuron (sorted)")
            }
        };
        f.text(origin + Vec2::new(0.0, size.y + 0.07), what, 12.0, pal::DIM, Align2::LEFT_BOTTOM);

        // one tile per neuron, layer by layer
        let mut tile = 1;
        for (l, block) in blocks[..n_blocks].iter().enumerate() {
            for j in 0..block.count {
                let (corner, s) = block.tile(j);
                f.image_part(&mut self.atlas, tile_part(tile, self.atlas_cols), corner, corner + Vec2::splat(s), 0.0);
                tile += 1;
            }
            if let Some((hl, j)) = self.hover {
                if hl == l {
                    let (c, s) = block.tile(j);
                    f.dl.lines.closed(&[c, c + Vec2::new(s, 0.0), c + Vec2::splat(s), c + Vec2::new(0.0, s)].map(|q| q.extend(0.01)), 2.2, pal::WHITE);
                }
            }
            let label = if l + 1 < layers { format!("layer {}: {} neurons", l + 1, block.count) } else { "outputs: one per answer".to_string() };
            f.text(Vec2::new(block.left, block.corner.y + 0.06), label, 12.5, pal::TEXT, Align2::LEFT_BOTTOM);
        }

        // the first layer one frequency at a time: where it puts the numbers
        if k_max > 0 && self.rings.len() == k_max {
            let base = self.overall(&self.score);
            for k in 0..k_max {
                let (centre, radius) = self.ring_place(k);
                self.draw_ring(f, k, centre, radius, false);
                let (cut, hot) = (self.cut[k], self.hover_ring == Some(k));
                if cut {
                    let d = Vec2::splat(radius * 0.8);
                    f.line((centre - d).extend(0.02), (centre + d).extend(0.02), 2.2, pal::RED);
                    f.line((centre + Vec2::new(-d.x, d.y)).extend(0.02), (centre + Vec2::new(d.x, -d.y)).extend(0.02), 2.2, pal::RED);
                }
                if hot {
                    f.circle(centre.extend(0.02), Vec3::Z, radius * 1.12, 1.6, pal::WHITE);
                }
                let below = centre - Vec2::new(0.0, radius * 1.18);
                f.text(below, format!("$k = {}$", k + 1), 11.5, if cut { pal::RED } else { pal::TEXT }, Align2::CENTER_TOP);
                // (each figure against the score at the step it was worked out at)
                let age = self.t.steps.saturating_sub(self.alone_at[k]);
                if self.alone_at[k] != u64::MAX && 4 * age <= self.t.steps && (base - self.alone_base[k]).abs() <= ALONE_FRESH {
                    let drop = self.alone_base[k] - self.alone[k];
                    let colour = if drop > 0.1 {
                        pal::RED
                    } else if drop > 0.03 {
                        pal::ORANGE
                    } else {
                        pal::DIM
                    };
                    // (the words only where the columns are wide enough for them)
                    let wide = self.ring_place(1.min(k_max - 1)).0.x - self.ring_place(0).0.x;
                    let label = if k_max == 1 || wide * pixels_per_unit(f) > 96.0 {
                        format!("{:.0} % without it", self.alone[k] * 100.0)
                    } else {
                        format!("{:.0} %", self.alone[k] * 100.0)
                    };
                    f.text_off(below, Vec2::new(0.0, 15.0), label, 10.5, colour, Align2::CENTER_TOP);
                }
            }
            let what = format!("each frequency's part of the embedding of the numbers 0 to {}, the brighter the more neurons use it", p - 1);
            // one line above the rings: as much of it as the canvas has room for
            let at = Vec2::new(RINGS.0, RINGS.2 + 0.52);
            let room = (RINGS.1 - RINGS.0) * pixels_per_unit(f);
            let fits = |text: &str| text_width(text, 12.0) <= room;
            if removed > 0 {
                let with_out = format!("{removed} removed: {:.0} % of the table right (was {:.0} %)", self.overall(&self.cut_score) * 100.0, base * 100.0);
                if fits(&format!("{what}      {with_out}")) {
                    f.text(at, what, 12.0, pal::DIM, Align2::LEFT_BOTTOM);
                    f.text(Vec2::new(RINGS.1, at.y), with_out, 12.0, pal::RED, Align2::RIGHT_BOTTOM);
                } else {
                    f.text(at, with_out, 12.0, pal::RED, Align2::LEFT_BOTTOM);
                }
            } else {
                let long = format!("{what}   ·   hover to enlarge   ·   click to remove");
                let short = "the first layer, one frequency at a time   ·   click to remove".to_string();
                f.text(
                    at,
                    if fits(&long) {
                        long
                    } else if fits(&what) {
                        what
                    } else {
                        short
                    },
                    12.0,
                    pal::DIM,
                    Align2::LEFT_BOTTOM,
                );
            }
        }

        // loss and accuracy as training goes
        let csize = Vec2::new(1.72, 0.6);
        let (o1, o2) = (Vec2::new(0.3, -1.85), Vec2::new(2.38, -1.85));
        chart(f, o1, csize, &[(self.t.loss[0].points(), pal::CYAN), (self.t.loss[1].points(), pal::ORANGE)], true);
        chart_in(f, o2, csize, &[(self.t.accuracy[0].points(), pal::CYAN), (self.t.accuracy[1].points(), pal::ORANGE)], false, Some([0.0, 1.0]));
        // the two values of each chart, in the colours of its curves. (A loss below
        // 10⁻³ is typeset as a power of ten, wider than the room kept for it: measured
        // as drawn, the value on the left moves over for the one on the right, and the
        // chart's name is shortened if the three no longer fit on the line.)
        let ppu = pixels_per_unit(f);
        let mut caption = |origin: Vec2, name: &str, short: &str, trained: String, held_out: String| {
            let y = origin.y + csize.y + 0.07;
            let right = origin.x + csize.x;
            let trained_at = (right - 0.52).min(right - f.text_size(&held_out, 12.0).x / ppu - 0.12);
            let room = trained_at - f.text_size(&trained, 12.0).x / ppu - 0.08 - origin.x;
            let name = if f.text_size(name, 12.0).x / ppu <= room { name } else { short };
            f.text(Vec2::new(origin.x, y), name, 12.0, pal::DIM, Align2::LEFT_BOTTOM);
            f.text(Vec2::new(trained_at, y), trained, 12.0, pal::CYAN, Align2::RIGHT_BOTTOM);
            f.text(Vec2::new(right, y), held_out, 12.0, pal::ORANGE, Align2::RIGHT_BOTTOM);
        };
        caption(o1, "loss (log scale)", "loss", loss_number(self.score.loss[0]), loss_number(self.score.loss[1]));
        caption(o2, "right answers", "right", format!("{:.0} %", self.score.accuracy[0] * 100.0), format!("{:.0} %", self.score.accuracy[1] * 100.0));
    }
}

/// One neuron over the table, into tile `tile` of an atlas `cols` tiles wide, on a
/// scale set by its own largest value.
fn paint_tile(pixels: &mut [Color], cols: usize, p: usize, tile: usize, values: &[f32]) {
    let heat = Cmap::CoolWarm.table();
    let (width, largest) = (cols * p, values.iter().fold(0.0f32, |m, v| m.max(v.abs())));
    let k = if largest > 1e-12 { 0.5 / largest } else { 0.0 };
    let origin = (tile / cols) * p * width + (tile % cols) * p;
    for (b, row) in values.chunks_exact(p).enumerate() {
        if let Some(line) = pixels.get_mut(origin + b * width..origin + b * width + p) {
            for (px, &v) in line.iter_mut().zip(row) {
                *px = heat[table_index(0.5 + v * k)];
            }
        }
    }
}

//! A small transformer, trained in view and then taken apart: where its heads look,
//! at what depth the answer appears, which parts write it, and which are needed.
//!
//! Everything is drawn in screen space (logical pixels), laid out afresh for the
//! size of the view: one column per position of the sequence, and the pictures that
//! are "one value per position" (the model's guesses, the logit lens, patching)
//! share those columns.

use super::*;
use crate::clock;
use crate::math::transformer::{Batch, Config, Edits, Knock, Repeat, Transformer};
use egui::RichText;

pub const DEF: SceneDef = SceneDef {
    id: "transformer",
    title: "A transformer, opened up",
    group: super::GROUP,
    blurb: "Train a tiny transformer, then see where its heads look and which parts do the work.",
    make: |_| Box::new(Opened::new()),
};

const LETTERS: [&str; 26] = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"];
const HEAD_LETTERS: [&str; 4] = ["a", "b", "c", "d"];

/// Sequences the loss, the knock-outs and the heads' habits are measured on.
const EVAL: usize = 12;

/// With "pause once it has learned" ticked, training also pauses after this many
/// steps without success (several times what the default model needs).
const GIVE_UP: u64 = 3000;

/// The model and the task, as the controls set them.
#[derive(Clone, Copy, PartialEq, Debug)]
struct Shape {
    layers: usize,
    heads: usize,
    width: usize,
    mlp: bool,
    vocab: usize,
    len: usize,
    shortest: usize,
    longest: usize,
}

/// Two layers of one head each: the least that can do the task, so each head has
/// one job. It learns the task from every seed tried, in a few hundred steps (see
/// the tests).
const DEFAULT: Shape = Shape { layers: 2, heads: 1, width: 32, mlp: false, vocab: 16, len: 20, shortest: 4, longest: 9 };
/// How it is trained to begin with (and in every experiment): learning rate, and
/// rows per step.
const LR: f32 = 0.005;
const ROWS: usize = 16;

impl Shape {
    fn config(&self) -> Config {
        Config { vocab: self.vocab, ctx: self.len, d: self.width, layers: self.layers, heads: self.heads, mlp: if self.mlp { 2 * self.width } else { 0 } }
    }

    fn task(&self) -> Repeat {
        Repeat { vocab: self.vocab, len: self.len, shortest: self.shortest, longest: self.longest }
    }

    fn head_count(&self) -> usize {
        self.layers * self.heads
    }

    /// "2a": the layer (counted from 1) and a letter for the head.
    fn head_name(&self, k: usize) -> String {
        format!("{}{}", k / self.heads + 1, HEAD_LETTERS[k % self.heads])
    }

    /// Keep the task possible: a run fits twice into the sequence, and a letter is
    /// left over (patching swaps one in).
    fn tidy(&mut self) {
        self.longest = self.longest.min(self.vocab - 1).min(self.len / 2).max(2);
        self.shortest = self.shortest.clamp(2, self.longest);
    }
}

/// What a head looks at, as average shares of its attention over the evaluation set:
///
/// - `[0]` the position just before, measured within the first copy of the run
///   (where each letter is new, and where what a head collects is later looked up);
/// - `[1]` positions holding the right answer (what followed an earlier copy of the
///   current letter), measured in the repeated part;
/// - `[2]` earlier copies of the current letter, measured in the repeated part.
///
/// The last forward pass must have been `batch`, with the edits (if any) whose
/// effect the habits are to show.
fn habits(net: &Transformer, batch: &Batch, shape: &Shape) -> Vec<[f32; 3]> {
    let len = batch.len;
    let mut out = vec![[0.0f32; 3]; shape.head_count()];
    let (mut first, mut later) = (0.0f32, 0.0f32);
    for b in 0..batch.count {
        let seq = &batch.tokens[b * len..(b + 1) * len];
        for t in 1..len {
            let n = b * len + t;
            // (the last position is never scored either, but it is not in the first copy)
            let in_first_copy = batch.mask[n] <= 0.0 && t + 1 < len;
            if in_first_copy {
                first += 1.0;
            }
            if batch.mask[n] > 0.0 {
                later += 1.0;
            }
            for (k, o) in out.iter_mut().enumerate() {
                let a = net.attention(k / shape.heads, k % shape.heads, n);
                if in_first_copy {
                    o[0] += a[t - 1];
                }
                if batch.mask[n] > 0.0 {
                    for s in 0..=t {
                        if seq[s] == batch.targets[n] {
                            o[1] += a[s];
                        }
                        if s < t && seq[s] == seq[t] {
                            o[2] += a[s];
                        }
                    }
                }
            }
        }
    }
    for o in &mut out {
        o[0] /= first.max(1.0);
        o[1] /= later.max(1.0);
        o[2] /= later.max(1.0);
    }
    out
}

/// A head's strongest habit, if it has one: its index in [`habits`] and its share.
fn habit(h: [f32; 3]) -> Option<(usize, f32)> {
    let k = (0..3).max_by(|&a, &b| h[a].total_cmp(&h[b])).unwrap_or(0);
    (h[k] > 0.4).then_some((k, h[k]))
}

/// (The last two: no habit while training, and none with heads knocked out.)
const HABIT_LINES: [[&str; 3]; 5] = [["looks at the", "letter before", "(early in the row)"], ["looks at what", "came after the", "earlier copy"], ["looks at the", "earlier copy", "of this letter"], ["no fixed", "habit (yet)", ""], ["no fixed", "habit", ""]];
/// The same for maps too small to write much under.
const HABIT_SHORT: [[&str; 3]; 5] = [["letter", "before", ""], ["after the", "copy", ""], ["earlier", "copy", ""], ["no habit", "", ""], ["no habit", "", ""]];

#[derive(Clone, Copy, PartialEq)]
enum Job {
    /// Loss and accuracy on the evaluation set, and the heads' habits.
    Eval,
    /// The loss with one more head knocked out.
    Solo(usize),
    /// One row of the patching picture.
    Patch(usize),
}

/// Everything the picture shows that is worked out from the model.
#[derive(Default)]
struct Shown {
    /// The probability given to the right next letter at each position of the
    /// sequence on show (NaN where the next letter cannot be known), and the
    /// model's guess there.
    right: Vec<f32>,
    guess: Vec<usize>,
    /// Per head, `len * len` attention weights (`[from * len + to]`).
    att: Vec<Vec<f32>>,
    /// The logit lens: `[point * len + position]`, the probability of the right
    /// letter when the stream is read at that depth; and the guess read there.
    lens: Vec<f32>,
    lens_guess: Vec<usize>,
    /// What each part adds to the right letter's lead over the average letter, at
    /// the position in focus: `(name, value, head)`.
    parts: Vec<(String, f32, Option<usize>)>,
    parts_total: f32,
    /// On the evaluation set: loss and accuracy with nothing knocked out, with the
    /// current knock-outs, and the loss with each head knocked out as well.
    intact: (f32, f32),
    base: (f32, f32),
    solo: Vec<f32>,
    /// The heads' habits with nothing knocked out (which head does what), and with
    /// the knock-outs in force (what the maps show).
    habits: Vec<[f32; 3]>,
    habits_now: Vec<[f32; 3]>,
    /// Patching, for the selected position: `[point * len + position]`, the share of
    /// the lost answer that comes back (NaN where there is nothing to show).
    patch: Vec<f32>,
    /// The letter swapped in; how sure the model is of the answer before the swap
    /// (with the knock-outs in force), and after it; and whether there is anything
    /// to trace (the answer was found, and after the swap it is under even odds).
    swapped: Option<usize>,
    found: f32,
    damaged: f32,
    traced: bool,
}

struct Opened {
    shape: Shape,
    net: Transformer,
    seed: u64,
    rng: Rng,
    batch: Batch,
    eval: Batch,
    /// The sequence on show, and its run length.
    seq: Batch,
    run: usize,
    running: bool,
    stop_when_learned: bool,
    stopped: bool,
    /// Steps taken since training was last started by hand, and whether it was
    /// paused for getting nowhere in that many.
    since: u64,
    gave_up: bool,
    /// Steps per second asked for, and achieved.
    rate: usize,
    speed: f32,
    pace: clock::Pace,
    steps: u64,
    /// Rows trained on since the last reset (to tell whether the test rows can still
    /// be new to the model).
    rows_drawn: u64,
    lr: f32,
    batch_size: usize,
    loss: f32,
    accuracy: f32,
    /// Loss and accuracy as training went.
    history: [History; 2],
    hover: Option<usize>,
    selected: usize,
    knocked: Vec<bool>,
    how: Knock,
    /// Each head's average output on the evaluation rows with nothing knocked out:
    /// what "replace with its average" puts in its place.
    means: Vec<Vec<f32>>,
    /// Training was started by hand on a model that had already learned: do not
    /// pause it again for having learned.
    keep_going: bool,
    /// Frames to leave out the measurements for, after a round of them that took long.
    rest: clock::Rest,
    /// An experiment waiting for the model to finish learning.
    pending: Option<usize>,
    /// Why the last experiment could not knock out the head it is about, if it could not.
    missing: Option<&'static str>,
    view: Shown,
    stale_shown: bool,
    stale_jobs: bool,
    queue: Vec<Job>,
    focus: usize,
}

impl Opened {
    fn new() -> Self {
        Self::with(DEFAULT, 1)
    }

    fn with(shape: Shape, seed: u64) -> Self {
        let mut rng = Rng::new(seed);
        let net = Transformer::new(shape.config(), &mut rng);
        let task = shape.task();
        let mut scene = Opened {
            shape,
            net,
            seed,
            batch: Batch::sample(&task, ROWS, &mut rng),
            rng,
            eval: Batch::sample(&task, EVAL, &mut Rng::new(77)),
            seq: Batch::default(),
            run: 0,
            running: true,
            stop_when_learned: true,
            stopped: false,
            since: 0,
            gave_up: false,
            rate: 60,
            speed: 0.0,
            pace: clock::Pace::default(),
            steps: 0,
            rows_drawn: 0,
            lr: LR,
            batch_size: ROWS,
            loss: 0.0,
            accuracy: 0.0,
            history: [History::new(400), History::new(400)],
            hover: None,
            selected: 0,
            knocked: vec![false; shape.head_count()],
            how: Knock::Mean,
            means: Vec::new(),
            keep_going: false,
            rest: clock::Rest::default(),
            pending: None,
            missing: None,
            view: Shown::default(),
            stale_shown: true,
            stale_jobs: true,
            queue: Vec::new(),
            focus: 0,
        };
        scene.show_run(6.clamp(shape.shortest, shape.longest), &mut Rng::new(5));
        scene
    }

    /// Put a sequence with the given run length on show, and select a position in
    /// its second copy (the first place the answer can be looked up).
    fn show_run(&mut self, run: usize, rng: &mut Rng) {
        let len = self.shape.len;
        let (mut tokens, mut scored) = (vec![0u32; len], vec![0.0f32; len]);
        self.shape.task().with_run(run, rng, &mut tokens, &mut scored);
        self.seq = Batch::single(&tokens, &scored);
        self.run = run;
        self.selected = (run + 2).min(len - 2);
        self.hover = None;
        self.changed();
    }

    /// The model or what is asked of it changed: everything shown must be redone.
    fn changed(&mut self) {
        self.missing = None;
        self.stale_shown = true;
        self.stale_jobs = true;
        self.queue.clear();
    }

    /// Fresh weights for the current shape.
    fn reset_weights(&mut self) {
        self.seed += 1;
        let mut rng = Rng::new(self.seed);
        self.net = Transformer::new(self.shape.config(), &mut rng);
        self.rng = rng;
        self.pending = None;
        self.keep_going = false;
        self.means.clear();
        self.steps = 0;
        self.rows_drawn = 0;
        self.loss = 0.0;
        self.accuracy = 0.0;
        self.history.iter_mut().for_each(History::clear);
        self.stopped = false;
        self.since = 0;
        self.gave_up = false;
        self.knocked = vec![false; self.shape.head_count()];
        self.view = Shown::default();
        self.changed();
    }

    /// The controls have set the shape (it was `before`): keep the task possible, and
    /// start again only if that leaves a different model or task. (A slider held past
    /// where `tidy` lets it go would otherwise throw the trained model away on every
    /// frame.)
    fn reshape(&mut self, before: Shape) {
        self.shape.tidy();
        if self.shape != before {
            self.rebuild();
            self.running = true;
        }
    }

    /// The shape or the task changed: new model, new sequences.
    fn rebuild(&mut self) {
        self.shape.tidy();
        // (what a step costs is for the pace to measure afresh)
        self.pace = clock::Pace::default();
        let task = self.shape.task();
        self.eval = Batch::sample(&task, EVAL, &mut Rng::new(77));
        self.reset_weights();
        self.show_run(6.clamp(self.shape.shortest, self.shape.longest), &mut Rng::new(5));
    }

    fn step_once(&mut self) {
        let task = self.shape.task();
        self.batch.refill(&task, self.batch_size, &mut self.rng);
        let (loss, accuracy) = self.net.train_step(&self.batch, self.lr, 0.0);
        let blend = if self.steps == 0 { 1.0 } else { 0.1 };
        self.loss += (loss - self.loss) * blend;
        self.accuracy += (accuracy - self.accuracy) * blend;
        self.steps += 1;
        self.rows_drawn += self.batch_size as u64;
        self.since += 1;
        self.history[0].push(self.loss);
        self.history[1].push(self.accuracy);
        self.stale_shown = true;
        self.stale_jobs = true;
    }

    /// The knock-outs in force, plus one more head if asked.
    fn edits(&self, also: Option<usize>) -> Edits {
        self.edits_for(|k| self.knocked[k] || also == Some(k))
    }

    /// Edits that knock out the heads `out` says.
    fn edits_for(&self, out: impl Fn(usize) -> bool) -> Edits {
        let mut e = Edits::none();
        for k in 0..self.shape.head_count() {
            if out(k) {
                let (layer, head) = (k / self.shape.heads, k % self.shape.heads);
                e = match (self.how, self.means.get(k)) {
                    // (the same average in every measurement, whatever rows it is run on)
                    (Knock::Mean, Some(mean)) => e.fix(layer, head, mean),
                    _ => e.knock(layer, head, Knock::Zero),
                };
            }
        }
        e
    }

    /// How many different rows the task can make: a run of each allowed length, of
    /// different letters in some order.
    fn possible_rows(&self) -> f64 {
        let Shape { vocab, shortest, longest, .. } = self.shape;
        (shortest..=longest).map(|run| (0..run).map(|i| vocab.saturating_sub(i) as f64).product::<f64>()).sum()
    }

    fn any_knocked(&self) -> bool {
        self.knocked.iter().any(|k| *k)
    }

    /// The position the arcs and the "what each part adds" bars are about.
    fn in_focus(&self) -> usize {
        self.hover.unwrap_or(self.selected)
    }

    /// Run the sequence on show and read off everything that is cheap to read.
    fn refresh_shown(&mut self) {
        let Shape { layers, heads, width: d, vocab, len, .. } = self.shape;
        let edits = self.edits(None);
        self.net.forward(&self.seq.tokens, 1, &edits);
        let (net, seq, v) = (&self.net, &self.seq, &mut self.view);
        let scored = |t: usize| seq.mask[t] > 0.0;
        v.right = (0..len).map(|t| if scored(t) { net.prob(seq.targets[t] as usize, t) } else { f32::NAN }).collect();
        v.guess = (0..len).map(|t| net.best(t)).collect();
        v.att = (0..layers * heads).map(|k| (0..len).flat_map(|t| net.attention(k / heads, k % heads, t)[..len].iter().copied()).collect()).collect();
        let (mut x, mut p) = (vec![0.0; d], vec![0.0; vocab]);
        v.lens.clear();
        v.lens_guess.clear();
        for point in 0..=layers {
            for t in 0..len {
                net.stream(point, t, &mut x);
                net.read_probs(&x, &mut p);
                v.lens.push(if scored(t) { p[seq.targets[t] as usize] } else { f32::NAN });
                v.lens_guess.push((0..vocab).max_by(|&a, &b| p[a].total_cmp(&p[b])).unwrap_or(0));
            }
        }

        // what each part adds to the right letter's lead over the average letter
        let t = self.hover.unwrap_or(self.selected);
        self.focus = t;
        v.parts.clear();
        v.parts_total = 0.0;
        if scored(t) {
            let answer = seq.targets[t] as usize;
            let lead = |x: &[f32]| net.read_logit(x, answer) - (0..vocab).map(|u| net.read_logit(x, u)).sum::<f32>() / vocab as f32;
            net.token_vector(seq.tokens[t] as usize, &mut x);
            v.parts.push((format!("the letter here ({})", LETTERS[seq.tokens[t] as usize]), lead(&x), None));
            net.position_vector(t, &mut x);
            v.parts.push(("its place in the row".to_string(), lead(&x), None));
            for l in 0..layers {
                for h in 0..heads {
                    net.head_vector(l, h, t, &mut x);
                    v.parts.push((format!("head {}", self.shape.head_name(l * heads + h)), lead(&x), Some(l * heads + h)));
                }
                if self.shape.mlp {
                    net.mlp_vector(l, t, &mut x);
                    v.parts.push((format!("MLP of layer {}", l + 1), lead(&x), None));
                }
            }
            v.parts_total = v.parts.iter().map(|p| p.1).sum();
        }
        self.stale_shown = false;
    }

    /// Every earlier copy of the selected letter swapped for a letter the sequence
    /// does not use: the model can then no longer look the answer up.
    fn corrupted(&self) -> Option<(Vec<u32>, usize)> {
        let t = self.selected;
        if self.seq.mask[t] <= 0.0 {
            return None;
        }
        // (no earlier copy of the letter: nothing to swap)
        if !self.seq.tokens[..t].contains(&self.seq.tokens[t]) {
            return None;
        }
        let spare = (0..self.shape.vocab as u32).find(|u| !self.seq.tokens.contains(u))?;
        let mut tokens = self.seq.tokens.clone();
        for s in 0..t {
            if tokens[s] == self.seq.tokens[t] {
                tokens[s] = spare;
            }
        }
        Some((tokens, spare as usize))
    }

    /// One row of the patching picture: run the damaged sequence once per position,
    /// each time putting the undamaged stream back at that one position of `point`.
    fn patch_row(&mut self, point: usize) {
        let Shape { layers, width: d, len, .. } = self.shape;
        if self.view.patch.len() != (layers + 1) * len {
            self.view.patch = vec![f32::NAN; (layers + 1) * len];
        }
        let t = self.selected;
        let row = point * len..(point + 1) * len;
        self.view.patch[row.clone()].fill(f32::NAN);
        let Some((damaged, spare)) = self.corrupted() else {
            self.view.swapped = None;
            self.view.traced = false;
            return;
        };
        let answer = self.seq.targets[t] as usize;
        let mut edits = self.edits(None);
        self.net.forward(&self.seq.tokens, 1, &edits);
        let clean = self.net.prob(answer, t).max(1e-30).ln();
        let mut x = vec![0.0; d];
        for pos in 0..=t {
            self.net.stream(point, pos, &mut x);
            edits = edits.replace(point, pos, pos, &x);
        }
        // sequence `pos` has position `pos` put back; the last one is left damaged
        let count = t + 2;
        let tokens: Vec<u32> = (0..count).flat_map(|_| damaged.iter().copied()).collect();
        self.net.forward(&tokens, count, &edits);
        let lost = self.net.prob(answer, (t + 1) * len + t).max(1e-30).ln();
        self.view.swapped = Some(spare);
        self.view.found = clean.exp();
        self.view.damaged = lost.exp();
        // nothing to trace unless the answer was found and the swap lost it: no
        // longer more likely than not, and a clear drop (or the shares are noise)
        self.view.traced = self.view.found >= 0.5 && self.view.damaged < 0.5 && clean - lost >= 0.05;
        if !self.view.traced {
            return;
        }
        for pos in 0..=t {
            let back = self.net.prob(answer, pos * len + t).max(1e-30).ln();
            self.view.patch[point * len + pos] = ((back - lost) / (clean - lost)).clamp(0.0, 1.0);
        }
    }

    fn run_job(&mut self, job: Job) {
        match job {
            Job::Eval => {
                self.net.forward(&self.eval.tokens, self.eval.count, &Edits::none());
                // each head's average, for the knock-outs that put it in the head's place
                let mut mean = vec![0.0; self.shape.width];
                self.means.clear();
                for k in 0..self.shape.head_count() {
                    self.net.head_mean(k / self.shape.heads, k % self.shape.heads, &mut mean);
                    self.means.push(mean.clone());
                }
                if self.how == Knock::Mean && self.any_knocked() {
                    self.stale_shown = true;
                }
                self.view.habits = habits(&self.net, &self.eval, &self.shape);
                self.view.intact = self.net.loss(&self.eval.targets, &self.eval.mask);
                if self.any_knocked() {
                    self.view.base = self.net.evaluate(&self.eval, &self.edits(None));
                    // (what is written under the maps describes the maps, which show
                    // the model with its knock-outs)
                    self.view.habits_now = habits(&self.net, &self.eval, &self.shape);
                } else {
                    self.view.base = self.view.intact;
                    self.view.habits_now = self.view.habits.clone();
                }
            }
            Job::Solo(k) => {
                self.view.solo.resize(self.shape.head_count(), f32::NAN);
                self.view.solo[k] = if self.knocked[k] { f32::NAN } else { self.net.evaluate(&self.eval, &self.edits(Some(k))).0 };
            }
            Job::Patch(point) => self.patch_row(point),
        }
    }

    /// The slower measurements, one job per frame, so that a frame never waits for
    /// all of them (a job takes under a millisecond for the default model, about ten
    /// for the largest); the round is run again for as long as the model keeps
    /// changing.
    fn pump(&mut self) {
        if self.queue.is_empty() {
            if !self.stale_jobs {
                return;
            }
            self.stale_jobs = false;
            // (popped from the end: the evaluation first, since it measures the
            // averages the others knock heads out with; then patching, which
            // answers a click)
            self.queue.extend((0..self.shape.head_count()).rev().map(Job::Solo));
            self.queue.extend((0..=self.shape.layers).rev().map(Job::Patch));
            self.queue.push(Job::Eval);
        }
        if let Some(job) = self.queue.pop() {
            self.run_job(job);
        }
    }

    fn settled(&self) -> bool {
        self.queue.is_empty() && !self.stale_jobs && !self.stale_shown
    }

    /// Has it learned the task? Judged by the last measurement on the evaluation set
    /// (at most a few frames old while training) and by the recent training rows.
    fn learned(&self) -> bool {
        self.view.intact.1 > 0.97 && self.accuracy > 0.975 && self.steps >= 100
    }

    /// Have the habits and knock-out losses been measured since the last reset?
    fn measured(&self) -> bool {
        self.view.habits.len() == self.shape.head_count() && self.view.solo.len() == self.shape.head_count()
    }

    /// A good moment to stop training: it has learned the task, and it gets every
    /// repeat of the row on show right. (A trained model still slips on a row in a
    /// hundred or so, and which rows changes from step to step; this keeps the
    /// picture it stops on from being one of them.)
    fn ready_to_stop(&self) -> bool {
        // (with heads knocked out the row on show says nothing about the model itself)
        let row_right = self.any_knocked() || self.view.right.iter().filter(|p| p.is_finite()).all(|&p| p > 0.5);
        self.learned() && self.measured() && row_right
    }

    /// The head, beyond the first layer, that looks at the answer.
    fn finder(&self) -> Option<usize> {
        let h = &self.view.habits;
        (self.shape.heads..h.len()).max_by(|&a, &b| h[a][1].total_cmp(&h[b][1])).filter(|&k| h[k][1] > 0.4)
    }

    /// The head in an earlier layer that the finder most depends on: the one whose
    /// loss, knocked out alone (every other head in), hurts most. (Measured here, not
    /// read from the knock-out losses on show, which include the knock-outs in force.)
    fn helper(&mut self) -> Option<usize> {
        let finder = self.finder()?;
        let before = finder / self.shape.heads * self.shape.heads;
        let mut best: Option<(usize, f32)> = None;
        for k in 0..before {
            let edits = self.edits_for(|j| j == k);
            let loss = self.net.evaluate(&self.eval, &edits).0;
            if loss.is_finite() && best.is_none_or(|(_, most)| loss > most) {
                best = Some((k, loss));
            }
        }
        best.map(|(k, _)| k)
    }

    /// Once it has learned: stop (if asked to), and run a waiting experiment. And
    /// give up, for now, on a model that is getting nowhere.
    fn consider_stopping(&mut self) {
        if self.ready_to_stop() {
            if let Some(k) = self.pending.take() {
                self.running = false;
                self.stopped = self.stop_when_learned;
                self.try_experiment(k);
            } else if self.running && self.stop_when_learned && !self.keep_going {
                self.running = false;
                self.stopped = true;
            }
        } else if self.running && self.stop_when_learned && self.since >= GIVE_UP && self.pending.is_none() {
            // a model that cannot learn the task would otherwise train for ever
            self.running = false;
            self.stopped = true;
            self.gave_up = true;
        }
    }

    const EXPERIMENTS: [(&'static str, &'static str); 6] = [
        ("watch it learn", "Fresh weights. The loss starts where a blind guess would be and creeps down while the model learns only that the next letter is one it has seen. Then, within a couple of hundred steps, the two heads lock together and the repeats come right. Training pauses by itself after about 300 steps."),
        ("see where the heads look", "The ringed letter is in the second copy. The second-layer head points straight at the letter that followed the earlier copy: that letter is the answer."),
        ("knock out the head that finds the answer", "The repeats go wrong, and the lens rows show the answer no longer appearing after the last layer."),
        ("knock out its first-layer helper", "This head never writes the answer itself (its bar in \"what each part adds\" is next to nothing), yet without it the second-layer head no longer looks at the answer (see its map, and the words under it), and the repeats go wrong all the same."),
        ("trace the answer's path", "All heads back in. Patching shows which places carry what is needed: the swapped letter at first; the position after it once the first layer has run; only the final position at the end."),
        ("try it with one layer", "It stalls with only a little over half of the repeats right, however long it trains. A single head can find the earlier copy of a letter, but the answer is the letter after that, and getting there takes a second step."),
    ];

    /// Set an experiment up. Those that need a trained model wait for one.
    fn try_experiment(&mut self, k: usize) {
        // (what the experiments say is about the default model, trained the default way)
        self.lr = LR;
        self.batch_size = ROWS;
        match k {
            0 => {
                if self.shape != DEFAULT {
                    self.shape = DEFAULT;
                    self.rebuild();
                } else {
                    self.reset_weights();
                }
                self.rate = 60;
                self.running = true;
            }
            5 => {
                self.shape = Shape { layers: 1, ..DEFAULT };
                self.rebuild();
                self.rate = 240;
                self.running = true;
            }
            _ if self.shape != DEFAULT || !self.learned() || !self.measured() => {
                // train it first, quickly, then come back
                if self.shape != DEFAULT {
                    self.shape = DEFAULT;
                    self.rebuild();
                }
                self.pending = Some(k);
                self.rate = self.rate.max(240);
                self.running = true;
                self.stopped = false;
                self.gave_up = false;
                self.since = 0;
            }
            1 | 4 => {
                self.knocked.fill(false);
                self.hover = None;
                self.selected = (self.run + 2).min(self.shape.len - 2);
                self.changed();
            }
            _ => {
                let head = if k == 2 { self.finder() } else { self.helper() };
                self.knocked.fill(false);
                self.changed();
                match head {
                    Some(head) => self.knocked[head] = true,
                    None => self.missing = Some("No head looks mostly at the answer in this model, so there is no head for this experiment: nothing was knocked out."),
                }
            }
        }
    }
}

// ----- drawing ------------------------------------------------------------------------

const BACKING: Color = Color::hex(0x0B0F1A);
const CELL: Color = Color::hex(0x161C2B);

/// Where everything goes, for one size of view.
struct Lay {
    view: Vec2,
    phone: bool,
    pad: f32,
    /// Left edge of the first column, and the width of a column.
    x0: f32,
    col: f32,
    head: f32,
    arcs: (f32, f32),
    tokens: f32,
    tile: f32,
    guess: f32,
    guess_h: f32,
    sure: f32,
    sure_h: f32,
    lens_head: f32,
    lens: f32,
    row: f32,
    patch_head: f32,
    patch: f32,
    /// The cards underneath: the heads' maps (top-left corner of each, and their
    /// side), the "what each part adds" bars (and how many rows of them there is room
    /// for), and the training chart (if there is room for it).
    side: f32,
    maps: Vec<Vec2>,
    parts: (Vec2, Vec2),
    part_rows: usize,
    chart: Option<(Vec2, Vec2)>,
}

/// Room for the names of the lens and patching rows on a phone, left of the columns.
const PHONE_LABELS: f32 = 38.0;

/// How wide a line of this scene's small print is: the kit's estimate allows for
/// wide letters, and these mostly lower-case lines take about a tenth less.
fn line_width(text: &str, size: f32) -> f32 {
    text_width(text, size) * 0.92
}

/// A loss as the picture and the panel write it: as mathematics, since `fmt` writes
/// a small one as 1.5e-4 (which is then set as a power of ten); and as it is if it is
/// not a number at all.
fn loss_text(loss: f32) -> String {
    if loss.is_finite() {
        format!("${}$", fmt(loss))
    } else {
        fmt(loss)
    }
}

/// Height of what is written under a head's map.
const UNDER_MAP: f32 = 66.0;
const OVER_MAP: f32 = 17.0;

impl Lay {
    /// The size of the small print.
    fn small(&self) -> f32 {
        if self.phone {
            10.5
        } else {
            11.5
        }
    }

    /// `top`: how much of the top of the view the application's buttons cover.
    fn new(view: Vec2, top: f32, shape: &Shape, part_rows: usize) -> Self {
        let (w, h) = (view.x, view.y);
        let phone = w < 520.0;
        let pad = if phone { 10.0 } else { 22.0 };
        let mut y = (top + 4.0).max(12.0);
        let labels = if phone { PHONE_LABELS } else { (w * 0.17).clamp(96.0, 136.0) };
        let x0 = pad + labels;
        let col = ((w - pad - x0) / shape.len as f32).min(36.0);
        let k = (h / 800.0).clamp(0.72, 1.4);
        let head = y;
        y += 18.0;
        let arcs = (y, y + 84.0 * k);
        y = arcs.1 + 3.0;
        let tile = col.clamp(10.0, 15.0 + 14.0 * k);
        let tokens = y;
        y += tile + 2.0;
        let guess_h = (tile * 0.82).max(11.0);
        let guess = y;
        y += guess_h + 2.0;
        let sure_h = 13.0 * k.max(0.85);
        let sure = y;
        // (a phone has no room for labels beside the rows: a line under them instead)
        y += sure_h + 10.0 * k + if phone { 14.0 } else { 0.0 };
        let row = (col * 0.66).clamp(11.0, 8.0 + 12.0 * k);
        let lens_head = y;
        y += 17.0;
        let lens = y;
        y += row * (shape.layers + 1) as f32 + 10.0 * k;
        let patch_head = y;
        y += 17.0;
        let patch = y;
        y += row * (shape.layers + 1) as f32 + 14.0 * k;

        // the cards: the largest maps for which everything still fits
        let n = shape.head_count();
        let inner = w - 2.0 * pad;
        let avail = h - y - 8.0;
        let flow = |side: f32, rows: usize, chart_on: bool| {
            // (a line more when some rows are left out, to say so)
            let parts_h = 44.0 + 15.0 * (rows + 1 + usize::from(rows < part_rows)) as f32;
            let per_row = (((inner + 10.0) / (side + 10.0)).floor() as usize).clamp(1, n);
            let rows = n.div_ceil(per_row);
            let maps: Vec<Vec2> = (0..n).map(|k| Vec2::new(pad + (k % per_row) as f32 * (side + 10.0), y + OVER_MAP + (k / per_row) as f32 * (side + OVER_MAP + UNDER_MAP))).collect();
            let maps_w = per_row as f32 * (side + 10.0);
            let maps_h = rows as f32 * (side + OVER_MAP + UNDER_MAP);
            // the bars and the chart go beside the maps if there is room, else below
            let beside = w - pad - (pad + maps_w + 14.0);
            let (parts, chart);
            if !phone && beside >= 240.0 {
                let x = pad + maps_w + 14.0;
                let width = beside.min(480.0);
                parts = (Vec2::new(x, y), Vec2::new(x + width, y + parts_h));
                let room = (maps_h - parts_h - 10.0).clamp(96.0, 230.0);
                chart = (Vec2::new(x, y + parts_h + 10.0), Vec2::new(x + width, y + parts_h + 10.0 + room));
            } else {
                let top = y + maps_h + 6.0;
                let width = if phone { inner } else { (inner * 0.45).clamp(240.0, 340.0) };
                parts = (Vec2::new(pad, top), Vec2::new(pad + width, top + parts_h));
                chart = if phone { (Vec2::new(pad, top + parts_h + 8.0), Vec2::new(w - pad, top + parts_h + 100.0)) } else { (Vec2::new(pad + width + 14.0, top), Vec2::new(w - pad, top + parts_h.max(110.0))) };
            }
            let chart = chart_on.then_some(chart);
            let bottom = parts.1.y.max(chart.map_or(y, |c| c.1.y)).max(y + maps_h);
            (maps, parts, chart, bottom - y)
        };
        // If nothing fits even with the smallest maps (many heads, a short view),
        // smaller maps are no help: the bars keep only the parts that add most (down
        // to three), then the training chart is left out (the panel has its numbers),
        // and then the bars are cut further.
        let fits = |side: f32, rows: usize, chart_on: bool| flow(side, rows, chart_on).3 <= avail;
        let most_rows = |chart_on: bool, fewest: usize| {
            let mut rows = part_rows;
            while rows > fewest.min(part_rows) && !fits(48.0, rows, chart_on) {
                rows -= 1;
            }
            rows
        };
        let mut rows = most_rows(true, 3);
        let chart_on = fits(48.0, rows, true);
        if !chart_on {
            rows = most_rows(false, 1);
        }
        let mut side = 296.0;
        while side > 48.0 && !fits(side, rows, chart_on) {
            side -= 8.0;
        }
        let (maps, parts, chart, _) = flow(side, rows, chart_on);
        Lay { view, phone, pad, x0, col, head, arcs, tokens, tile, guess, guess_h, sure, sure_h, lens_head, lens, row, patch_head, patch, side, maps, parts, part_rows: rows, chart }
    }

    fn centre(&self, column: usize) -> f32 {
        self.x0 + (column as f32 + 0.5) * self.col
    }

    /// The column under a point, if it is over the part of the picture that is laid
    /// out in columns.
    fn column(&self, p: Vec2, shape: &Shape) -> Option<usize> {
        let bottom = self.patch + self.row * (shape.layers + 1) as f32;
        let i = ((p.x - self.x0) / self.col).floor();
        (p.y >= self.arcs.0 && p.y <= bottom && i >= 0.0 && (i as usize) < shape.len).then_some(i as usize)
    }

    /// The head whose map (or the text under it) is under a point.
    fn map(&self, p: Vec2) -> Option<usize> {
        self.maps.iter().position(|m| p.x >= m.x && p.x <= m.x + self.side && p.y >= m.y - OVER_MAP && p.y <= m.y + self.side + UNDER_MAP - 8.0)
    }
}

impl Opened {
    fn head_color(&self, k: usize) -> Color {
        pal::SERIES[k % pal::SERIES.len()]
    }

    fn token_color(&self, token: usize) -> Color {
        Cmap::Turbo.color(0.06 + 0.88 * token as f32 / (self.shape.vocab.max(2) - 1) as f32).mix(BACKING, 0.5)
    }

    fn draw(&self, f: &mut Frame, lay: &Lay) {
        let Shape { layers, len, .. } = self.shape;
        let (seq, v) = (&self.seq, &self.view);
        let ready = v.right.len() == len;
        let focus = self.in_focus();
        let good = Cmap::Viridis.table();
        let (text, dim) = (pal::TEXT, pal::DIM);
        let small = lay.small();
        let label_x = lay.x0 - 9.0;
        let right_edge = lay.x0 + lay.col * len as f32;
        hud_rect_at(f, 0, Vec2::ZERO, lay.view, BACKING);

        // ----- the sequence, the guesses, and where the heads look
        let learned = if self.steps == 0 { "untrained".to_string() } else { format!("after {} training steps", count(self.steps as usize)) };
        let wide = lay.view.x >= 900.0;
        let heading = if lay.phone {
            format!("a run of letters, repeated ({learned})")
        } else if wide {
            format!("a run of {} letters, repeated. The model reads from the left and guesses each next letter ({learned}).", self.run)
        } else {
            format!("a run of {} letters, repeated; the model guesses each next letter ({} steps)", self.run, count(self.steps as usize))
        };
        f.text_px(Vec2::new(lay.pad, lay.head), heading, small + 0.5, text, Align2::LEFT_TOP);
        if !lay.phone {
            let mid = |a: f32, b: f32| (a + b) * 0.5;
            f.text_px(Vec2::new(label_x, lay.arcs.0 + 4.0), "where each head", small, dim, Align2::RIGHT_TOP);
            f.text_px(Vec2::new(label_x, lay.arcs.0 + 4.0 + small + 2.0), "looks from the", small, dim, Align2::RIGHT_TOP);
            f.text_px(Vec2::new(label_x, lay.arcs.0 + 4.0 + 2.0 * (small + 2.0)), "ringed letter", small, dim, Align2::RIGHT_TOP);
            f.text_px(Vec2::new(label_x, mid(lay.tokens, lay.tokens + lay.tile)), "the sequence", small, text, Align2::RIGHT_CENTER);
            let roomy = lay.x0 - lay.pad > 128.0;
            f.text_px(Vec2::new(label_x, mid(lay.guess, lay.guess + lay.guess_h)), if roomy { "its guess for the next" } else { "its next guess" }, small, dim, Align2::RIGHT_CENTER);
            f.text_px(Vec2::new(label_x, mid(lay.sure, lay.sure + lay.sure_h)), if roomy { "how sure of the right one" } else { "how sure of it" }, small - 1.0, dim, Align2::RIGHT_CENTER);
        } else {
            f.text_px(Vec2::new(lay.pad, lay.sure + lay.sure_h + 4.0), "row 2: its guess for the next letter (green: right)   row 3: how sure", 9.5, dim, Align2::LEFT_TOP);
        }
        for t in 0..len {
            let (a, b) = (lay.x0 + t as f32 * lay.col + 1.0, lay.x0 + (t + 1) as f32 * lay.col - 1.0);
            let token = seq.tokens[t] as usize;
            hud_rect_at(f, 1, Vec2::new(a, lay.tokens), Vec2::new(b, lay.tokens + lay.tile), self.token_color(token));
            f.text_px(Vec2::new(lay.centre(t), lay.tokens + lay.tile * 0.5), LETTERS[token], (lay.tile * 0.56).clamp(9.0, 16.0), pal::WHITE, Align2::CENTER_CENTER);
            if !ready {
                continue;
            }
            // the guess, green if right and red if wrong; grey where nothing can be known yet
            let scored = seq.mask[t] > 0.0;
            let right = scored && v.guess[t] == seq.targets[t] as usize;
            let back = if !scored { CELL } else if right { Color::hex(0x1F7A4D) } else { Color::hex(0x9C2F3F) };
            hud_rect_at(f, 1, Vec2::new(a, lay.guess), Vec2::new(b, lay.guess + lay.guess_h), back);
            if t + 1 < len {
                let ink = if scored { pal::WHITE } else { pal::FAINT };
                f.text_px(Vec2::new(lay.centre(t), lay.guess + lay.guess_h * 0.5), LETTERS[v.guess[t]], (lay.guess_h * 0.6).clamp(8.5, 14.0), ink, Align2::CENTER_CENTER);
            }
            hud_rect_at(f, 1, Vec2::new(a, lay.sure), Vec2::new(b, lay.sure + lay.sure_h), CELL);
            if scored {
                let p = v.right[t].clamp(0.0, 1.0);
                hud_rect_at(f, 2, Vec2::new(a, lay.sure + lay.sure_h * (1.0 - p)), Vec2::new(b, lay.sure + lay.sure_h), good[table_index(0.15 + 0.85 * p)]);
            }
        }
        // the first run cannot be predicted (every letter in it is new): say so under it
        let first_run = lay.col * self.run as f32;
        if let Some(words) = ["nothing to go on yet", "no clue yet"].into_iter().find(|w| ready && line_width(w, small - 2.0) + 6.0 <= first_run) {
            f.text_px(Vec2::new(lay.x0 + first_run * 0.5, lay.sure + lay.sure_h * 0.5), words, small - 2.0, pal::FAINT.mix(dim, 0.5), Align2::CENTER_CENTER);
        }
        // rings: the selected position (white) and the one under the pointer
        let ring = |f: &mut Frame, t: usize, width: f32, color: Color| {
            let (a, b) = (lay.x0 + t as f32 * lay.col, lay.x0 + (t + 1) as f32 * lay.col);
            hud_outline(f, Vec2::new(a, lay.tokens - 1.0), Vec2::new(b, lay.tokens + lay.tile + 1.0), width, color);
        };
        ring(f, self.selected, 2.2, pal::WHITE);
        if let Some(t) = self.hover.filter(|&t| t != self.selected) {
            ring(f, t, 1.4, pal::WHITE.a(0.7));
        }

        // arcs from the position in focus to where each head looks
        if ready {
            let base = lay.arcs.1;
            let room = lay.arcs.1 - lay.arcs.0 - 6.0;
            for (k, att) in v.att.iter().enumerate() {
                let color = if self.knocked[k] { pal::FAINT } else { self.head_color(k) };
                for s in 0..=focus {
                    let a = att[focus * len + s];
                    if a < 0.04 {
                        continue;
                    }
                    let (from, to) = (lay.centre(focus), lay.centre(s));
                    let lift = k as f32 * 3.0;
                    if s == focus {
                        // looking at itself: a short stalk
                        hud_line(f, Vec2::new(from + lift - 4.0, base), Vec2::new(from + lift - 4.0, base - 9.0 - 10.0 * a), 1.0 + 3.0 * a, color.a(0.35 + 0.65 * a));
                        continue;
                    }
                    let height = (room * (0.22 + 0.78 * ((focus - s) as f32 / (len - 1) as f32).sqrt()) - lift).max(8.0);
                    let pts = (0..=18).map(|i| {
                        let u = i as f32 / 18.0;
                        Vec2::new(from + (to - from) * u, base - height * 4.0 * u * (1.0 - u))
                    });
                    hud_path(f, pts, 1.0 + 3.6 * a, color.a(0.3 + 0.7 * a));
                    hud_rect_at(f, 2, Vec2::new(to - 2.0 - 2.0 * a, base - 3.0 - 3.0 * a), Vec2::new(to + 2.0 + 2.0 * a, base + 1.0), color);
                }
            }
            // which colour is which head
            let mut x = if lay.phone { lay.pad } else { lay.x0 + 4.0 };
            for k in 0..self.shape.head_count() {
                let name = format!("head {}", self.shape.head_name(k));
                hud_rect_at(f, 2, Vec2::new(x, lay.arcs.0 + 5.0), Vec2::new(x + 9.0, lay.arcs.0 + 14.0), self.head_color(k));
                f.text_px(Vec2::new(x + 13.0, lay.arcs.0 + 9.5), &name, small - 1.0, if self.knocked[k] { pal::FAINT } else { dim }, Align2::LEFT_CENTER);
                x += 22.0 + name.len() as f32 * (small - 1.0) * 0.56;
                if x > right_edge - 60.0 {
                    break;
                }
            }
            // with heads knocked out: what that costs, on the test rows
            if self.any_knocked() && v.base.1.is_finite() && self.measured() {
                let names: Vec<String> = (0..self.shape.head_count()).filter(|&k| self.knocked[k]).map(|k| self.shape.head_name(k)).collect();
                let what = if lay.phone { format!("without {}: {:.0} % right", names.join(", "), v.base.1 * 100.0) } else { format!("with head {} knocked out it gets {:.0} % of repeats right (with it, {:.0} %)", names.join(" and "), v.base.1 * 100.0, v.intact.1 * 100.0) };
                if x + what.len() as f32 * small * 0.5 < right_edge {
                    f.text_px(Vec2::new(right_edge, lay.arcs.0 + 9.5), what, small, pal::RED.mix(pal::WHITE, 0.25), Align2::RIGHT_CENTER);
                }
            }
        }

        // ----- the logit lens and patching: one cell per place in the stream
        let bar_length = if lay.phone { 44.0 } else { 70.0 };
        let scale_bar = |f: &mut Frame, y: f32, what: &str| {
            // "what 0 [gradient] 1", right-aligned
            let (x1, wide) = (right_edge, bar_length);
            f.text_px(Vec2::new(x1, y + 7.0), "1", small - 1.0, dim, Align2::RIGHT_CENTER);
            for i in 0..14 {
                let (a, b) = (x1 - 10.0 - wide + wide * i as f32 / 14.0, x1 - 10.0 - wide + wide * (i + 1) as f32 / 14.0);
                hud_rect_at(f, 1, Vec2::new(a, y + 2.0), Vec2::new(b + 0.5, y + 12.0), good[table_index((i as f32 + 0.5) / 14.0)]);
            }
            f.text_px(Vec2::new(x1 - 14.0 - wide, y + 7.0), format!("{what} 0"), small - 1.0, dim, Align2::RIGHT_CENTER);
        };
        let point_name = |point: usize| match (point, lay.phone) {
            (0, false) if lay.x0 - lay.pad > 128.0 => "after the embeddings".to_string(),
            (0, false) => "embeddings only".to_string(),
            (0, true) => "start".to_string(),
            (l, false) => format!("after layer {l}"),
            (l, true) => format!("layer {l}"),
        };
        let row_labels = |f: &mut Frame, top: f32| {
            for point in 0..=layers {
                let y = top + (point as f32 + 0.5) * lay.row;
                if lay.phone {
                    f.text_px(Vec2::new(lay.x0 - 4.0, y), point_name(point), 9.5, dim, Align2::RIGHT_CENTER);
                } else {
                    f.text_px(Vec2::new(label_x, y), point_name(point), small - 0.5, dim, Align2::RIGHT_CENTER);
                }
            }
        };
        let cell = |t: usize, top: f32, point: usize| (Vec2::new(lay.x0 + t as f32 * lay.col + 1.0, top + point as f32 * lay.row + 1.0), Vec2::new(lay.x0 + (t + 1) as f32 * lay.col - 1.0, top + (point + 1) as f32 * lay.row));
        // A heading and the scale at its right share the line as far as the columns
        // go: the first of the forms (longest first) that fits there with its scale.
        let heading = |f: &mut Frame, y: f32, forms: &[(&str, Option<&str>)]| {
            let room = right_edge - lay.pad;
            let width = |(words, bar): &(&str, Option<&str>)| line_width(words, small + 0.5) + bar.map_or(0.0, |b| 18.0 + bar_length + line_width(&format!("{b} 0"), small - 1.0));
            let Some(&(words, bar)) = forms.iter().find(|form| width(form) <= room).or(forms.last()) else {
                return;
            };
            f.text_px(Vec2::new(lay.pad, y), words, small + 0.5, text, Align2::LEFT_TOP);
            if let Some(what) = bar {
                scale_bar(f, y, what);
            }
        };

        let lens_forms = [
            ("logit lens: read the stream before the model is finished. At what depth is the answer there?", Some("chance it gives the right letter:")),
            ("logit lens: the stream read early. Is the answer there yet?", Some("sure:")),
            ("logit lens: the stream read early", Some("sure:")),
            ("logit lens: the stream read early", None),
        ];
        heading(f, lay.lens_head, &lens_forms);
        row_labels(f, lay.lens);
        if ready && v.lens.len() == (layers + 1) * len {
            for point in 0..=layers {
                for t in 0..len {
                    let (min, max) = cell(t, lay.lens, point);
                    let p = v.lens[point * len + t];
                    if p.is_nan() {
                        hud_rect_at(f, 1, min, max, CELL.mix(BACKING, 0.5));
                        continue;
                    }
                    hud_rect_at(f, 1, min, max, good[table_index(p)]);
                    if lay.row >= 16.0 && lay.col >= 16.0 {
                        let ink = if p > 0.55 { Color::hex(0x10141F) } else { pal::WHITE.a(0.8) };
                        f.text_px((min + max) * 0.5, LETTERS[v.lens_guess[point * len + t]], (lay.row * 0.62).min(11.0), ink, Align2::CENTER_CENTER);
                    }
                }
            }
        }

        let answer = seq.targets[self.selected] as usize;
        let here = LETTERS[seq.tokens[self.selected] as usize];
        let patch_heading = match (v.swapped, seq.mask[self.selected] > 0.0) {
            (_, false) => vec!["patching: click a letter in the repeated part to trace how its answer is found".to_string()],
            (None, true) => vec![format!("patching: there is no earlier {here} in this row to swap, so nothing to trace from here")],
            (Some(spare), _) if v.found >= 0.5 && !v.traced => {
                vec![format!("patching: swapping every earlier {here} for {} does not lose the answer {} here ({:.0} % sure still), so nothing to trace", LETTERS[spare], LETTERS[answer], v.damaged * 100.0)]
            }
            (Some(_), _) if v.found < 0.5 && lay.phone => vec![format!("patching: the answer {} is not found, so nothing to trace", LETTERS[answer])],
            (Some(_), _) if v.found < 0.5 => vec![format!("patching: nothing to trace here, since the model does not find the answer {} in the first place", LETTERS[answer])],
            (Some(spare), _) => vec![
                format!(
                    "patching: swap every earlier {here} for {} (red frame) and it is only {:.0} % sure of the answer {}. Put ONE place of the stream back: how much returns (yellow: all)?",
                    LETTERS[spare],
                    v.damaged * 100.0,
                    LETTERS[answer]
                ),
                format!("patching: earlier {here} swapped (red frame), ONE place put back: how much returns (yellow: all)?"),
                format!("patching: earlier {here} swapped for {}", LETTERS[spare]),
            ],
        };
        // (a trace has a scale; the long forms are worth more than the scale, which
        // the colour key in the panel also gives)
        let patch_forms: Vec<(&str, Option<&str>)> = match patch_heading.as_slice() {
            [long, medium, short] => vec![(long, Some("share that returns:")), (long, None), (medium, Some("back:")), (medium, None), (short, Some("back:")), (short, None)],
            forms => forms.iter().map(|words| (words.as_str(), None)).collect(),
        };
        heading(f, lay.patch_head, &patch_forms);
        row_labels(f, lay.patch);
        if v.patch.len() == (layers + 1) * len {
            for point in 0..=layers {
                for t in 0..len {
                    let (min, max) = cell(t, lay.patch, point);
                    let p = v.patch[point * len + t];
                    if p.is_nan() {
                        hud_rect_at(f, 1, min, max, CELL.mix(BACKING, 0.5));
                    } else {
                        hud_rect_at(f, 1, min, max, good[table_index(p)]);
                    }
                }
            }
            // which letters were swapped
            if v.swapped.is_some() && v.traced {
                for s in 0..self.selected {
                    if seq.tokens[s] == seq.tokens[self.selected] {
                        let x = lay.x0 + s as f32 * lay.col;
                        hud_outline(f, Vec2::new(x + 1.0, lay.patch + 1.0), Vec2::new(x + lay.col - 1.0, lay.patch + lay.row), 2.0, pal::RED);
                    }
                }
            }
        }
        // the selected column, through both grids
        let x = lay.x0 + self.selected as f32 * lay.col;
        for top in [lay.lens, lay.patch] {
            hud_outline(f, Vec2::new(x, top), Vec2::new(x + lay.col, top + lay.row * (layers + 1) as f32 + 1.0), 1.2, pal::WHITE.a(0.75));
        }

        self.draw_maps(f, lay, focus);
        self.draw_parts(f, lay);
        self.draw_chart(f, lay);
    }

    /// Each head's whole attention pattern, what it has turned into, and what is
    /// lost without it.
    fn draw_maps(&self, f: &mut Frame, lay: &Lay, focus: usize) {
        let len = self.shape.len;
        let v = &self.view;
        let side = lay.side;
        let hot = Cmap::Magma.table();
        let tiny = if side < 100.0 { 9.5 } else { 11.0 };
        let worst = (self.shape.vocab as f32).ln();
        for (k, &at) in lay.maps.iter().enumerate() {
            let out = self.knocked[k];
            let color = self.head_color(k);
            f.text_px(at + Vec2::new(0.0, -3.0), format!("head {}", self.shape.head_name(k)), tiny + 1.0, if out { pal::FAINT } else { color }, Align2::LEFT_BOTTOM);
            if side >= 215.0 {
                f.text_px(at + Vec2::new(side, -3.0), "row: looking from, column: at", tiny - 1.0, pal::FAINT.mix(pal::DIM, 0.6), Align2::RIGHT_BOTTOM);
            }
            hud_rect_at(f, 1, at, at + Vec2::splat(side), Color::hex(0x05060B));
            let c = side / len as f32;
            if let Some(att) = v.att.get(k).filter(|a| a.len() == len * len) {
                for t in 0..len {
                    for s in 0..=t {
                        let a = att[t * len + s];
                        if a < 0.03 {
                            continue;
                        }
                        let shade = hot[table_index(0.12 + 0.88 * a.sqrt())];
                        hud_rect_at(f, 2, at + Vec2::new(s as f32 * c, t as f32 * c), at + Vec2::new((s + 1) as f32 * c, (t + 1) as f32 * c), if out { shade.mix(BACKING, 0.75) } else { shade });
                    }
                }
            }
            // the row of the position in focus
            let y = at.y + (focus as f32 + 0.5) * c;
            hud_line(f, Vec2::new(at.x - 4.0, y), Vec2::new(at.x - 0.5, y), 2.0, pal::WHITE);
            hud_outline(f, at, at + Vec2::splat(side), 1.0, if out { pal::FAINT } else { color.a(0.8) });

            let mut y = at.y + side + 4.0;
            if out {
                f.text_px(Vec2::new(at.x, y), "knocked out", tiny, pal::RED, Align2::LEFT_TOP);
                if side >= 92.0 {
                    f.text_px(Vec2::new(at.x, y + tiny + 2.0), "(click to restore)", tiny - 0.5, pal::DIM, Align2::LEFT_TOP);
                }
                continue;
            }
            let h = v.habits_now.get(k).copied().unwrap_or([0.0; 3]);
            let which = habit(h).map_or(if self.any_knocked() { 4 } else { 3 }, |(which, _)| which);
            let lines = if side >= 92.0 { HABIT_LINES[which] } else { HABIT_SHORT[which] };
            for line in lines.iter().filter(|l| !l.is_empty()) {
                f.text_px(Vec2::new(at.x, y), *line, tiny, pal::TEXT, Align2::LEFT_TOP);
                y += tiny + 1.5;
            }
            // what is lost without it
            let y = at.y + side + UNDER_MAP - 24.0;
            if let Some(&loss) = v.solo.get(k).filter(|l| l.is_finite()) {
                if side >= 92.0 {
                    f.text_px(Vec2::new(at.x, y), format!("loss without it: {loss:.2}"), tiny - 0.5, pal::DIM, Align2::LEFT_TOP);
                }
                // the bar is the rise over the loss with the head in place
                let share = ((loss - v.base.0) / worst).clamp(0.0, 1.0);
                hud_rect_at(f, 1, Vec2::new(at.x, y + tiny + 3.0), Vec2::new(at.x + side, y + tiny + 8.0), CELL);
                hud_rect_at(f, 2, Vec2::new(at.x, y + tiny + 3.0), Vec2::new(at.x + side * share, y + tiny + 8.0), pal::RED.mix(pal::YELLOW, 1.0 - share));
            }
        }
    }

    /// What each part of the model adds to the right answer at the position in focus.
    fn draw_parts(&self, f: &mut Frame, lay: &Lay) {
        let v = &self.view;
        let (min, max) = lay.parts;
        let small = lay.small();
        hud_rect_at(f, 1, min, max, Color::hex(0x0E1320));
        let t = self.focus;
        let x = min.x + 9.0;
        if v.parts.is_empty() {
            f.text_px(Vec2::new(x, min.y + 8.0), "what each part adds to the right letter", small, pal::TEXT, Align2::LEFT_TOP);
            f.text_px(Vec2::new(x, min.y + 10.0 + small + 2.0), "(point at a letter in the repeated part)", small - 0.5, pal::DIM, Align2::LEFT_TOP);
            return;
        }
        let answer = LETTERS[self.seq.targets[t] as usize];
        f.text_px(Vec2::new(x, min.y + 7.0), format!("what each part adds to the score of {answer},"), small, pal::TEXT, Align2::LEFT_TOP);
        f.text_px(Vec2::new(x, min.y + 8.0 + small + 2.0), format!("the right letter after position {}", t + 1), small, pal::TEXT, Align2::LEFT_TOP);
        let top = min.y + 40.0;
        let names = (max.x - min.x) * 0.43;
        let (zero, reach) = (min.x + names + (max.x - min.x - names - 44.0) * 0.3, (max.x - min.x - names - 44.0) * 0.7);
        let most = v.parts.iter().map(|p| p.1.abs()).fold(1e-3f32, f32::max);
        // (where there is no room for every part: those that add most, in their order)
        let mut shown: Vec<usize> = (0..v.parts.len()).collect();
        if lay.part_rows < shown.len() {
            shown.sort_by(|&a, &b| v.parts[b].1.abs().total_cmp(&v.parts[a].1.abs()));
            shown.truncate(lay.part_rows);
            shown.sort_unstable();
        }
        for (i, (name, value, head)) in shown.iter().map(|&k| &v.parts[k]).enumerate() {
            let y = top + i as f32 * 15.0;
            let ink = match head {
                Some(k) if self.knocked[*k] => pal::FAINT,
                Some(k) => self.head_color(*k),
                None => pal::DIM,
            };
            f.text_px(Vec2::new(min.x + names - 6.0, y + 7.0), name, small - 0.5, ink, Align2::RIGHT_CENTER);
            let end = zero + reach * value / most;
            hud_rect_at(f, 2, Vec2::new(zero.min(end), y + 2.0), Vec2::new(zero.max(end).max(zero.min(end) + 1.0), y + 12.0), if *value >= 0.0 { pal::GREEN } else { pal::PINK });
            f.text_px(Vec2::new(max.x - 6.0, y + 7.0), signed(*value, 1), small - 0.5, pal::TEXT, Align2::RIGHT_CENTER);
        }
        let bottom = top + shown.len() as f32 * 15.0;
        hud_line(f, Vec2::new(zero, top - 1.0), Vec2::new(zero, bottom), 1.0, pal::DIM);
        f.text_px(Vec2::new(x, bottom + 9.0), "sum: its lead over an average letter", small - 1.0, pal::DIM, Align2::LEFT_CENTER);
        f.text_px(Vec2::new(max.x - 6.0, bottom + 9.0), signed(v.parts_total, 1), small - 0.5, pal::WHITE, Align2::RIGHT_CENTER);
        if shown.len() < v.parts.len() {
            f.text_px(Vec2::new(x, bottom + 24.0), format!("(the {} largest of {} parts; the sum is of all)", shown.len(), v.parts.len()), small - 1.0, pal::DIM, Align2::LEFT_CENTER);
        }
    }

    /// Loss and accuracy over training.
    fn draw_chart(&self, f: &mut Frame, lay: &Lay) {
        let Some((min, max)) = lay.chart else {
            return;
        };
        let small = lay.small();
        hud_rect_at(f, 1, min, max, Color::hex(0x0E1320));
        let (a, b) = (min + Vec2::new(10.0, 34.0), max - Vec2::new(12.0, 18.0));
        f.text_px(min + Vec2::new(9.0, 7.0), "training:", small, pal::TEXT, Align2::LEFT_TOP);
        f.text_px(min + Vec2::new(9.0 + small * 5.2, 7.0), format!("loss {}", loss_text(self.loss)), small, pal::CYAN, Align2::LEFT_TOP);
        f.text_px(Vec2::new(max.x - 9.0, min.y + 7.0), format!("repeats right {:.0} %", self.accuracy * 100.0), small, pal::YELLOW, Align2::RIGHT_TOP);
        let worst = (self.shape.vocab as f32).ln();
        // the level of a blind guess, and the floor
        hud_line(f, Vec2::new(a.x, b.y), Vec2::new(b.x, b.y), 1.0, pal::DIM.a(0.8));
        let blind = b.y - (b.y - a.y) * (worst / (worst * 1.08));
        for i in 0..24 {
            let (x0, x1) = (a.x + (b.x - a.x) * i as f32 / 24.0, a.x + (b.x - a.x) * (i as f32 + 0.5) / 24.0);
            hud_line(f, Vec2::new(x0, blind), Vec2::new(x1, blind), 1.0, pal::FAINT);
        }
        // (named under the chart, where no curve can run through the words)
        let dashed = if b.x - a.x >= 270.0 { "dashed line: the loss of a blind guess" } else { "dashed: a blind guess" };
        f.text_px(Vec2::new((a.x + b.x) * 0.5, b.y + 3.0), dashed, small - 2.0, pal::FAINT.mix(pal::DIM, 0.7), Align2::CENTER_TOP);
        f.text_px(Vec2::new(a.x, b.y + 3.0), "step 0", small - 2.0, pal::DIM, Align2::LEFT_TOP);
        f.text_px(Vec2::new(b.x, b.y + 3.0), format!("step {}", count(self.steps as usize)), small - 2.0, pal::DIM, Align2::RIGHT_TOP);
        let (losses, right) = (self.history[0].points(), self.history[1].points());
        let n = losses.len();
        if n >= 2 {
            let at = |i: usize, share: f32| Vec2::new(a.x + (b.x - a.x) * i as f32 / (n - 1) as f32, b.y - (b.y - a.y) * share.clamp(0.0, 1.0));
            hud_path(f, right.iter().enumerate().map(|(i, &v)| at(i, v)), 1.6, pal::YELLOW.a(0.9));
            hud_path(f, losses.iter().enumerate().map(|(i, &v)| at(i, v / (worst * 1.08))), 1.8, pal::CYAN);
        }
    }
}

impl Scene for Opened {
    fn view(&self) -> View {
        View::Fit { center: Vec2::ZERO, half: Vec2::ONE }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = 0.0;
        s.vignette = 0.0;
    }

    /// Nothing moves unless the model is training or measurements are still coming in.
    fn animated(&self) -> bool {
        self.running || !self.settled() || self.pending.is_some()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("point at a letter: where the heads look    click it: study it    click a head's map: knock it out")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        let shape = self.shape;
        let made_of = format!(
            "{} with {}{}, {} weights",
            if shape.layers == 1 { "one layer".to_string() } else { format!("{} layers", ["", "one", "two", "three"][shape.layers.min(3)]) },
            if shape.heads == 1 { "one attention head".to_string() } else { format!("{} attention heads", shape.heads) },
            if shape.layers > 1 { " each" } else { "" },
            count(self.net.n_params())
        );
        ui::note(ui, &format!("A transformer small enough to read: {made_of}. {}", "It is shown rows of letters in which a short run repeats, and learns to predict the repeats. Nobody tells it how; the pictures show the method it finds."));
        ui.horizontal_wrapped(|ui| {
            if ui.button(if self.running { "pause" } else { "train" }).clicked() {
                self.running = !self.running;
                self.stopped = false;
                self.gave_up = false;
                self.since = 0;
                self.pending = None;
                // asked to train a model that has learned: then train, and do not
                // pause again a moment later for the same reason
                self.keep_going = self.running && self.learned();
            }
            if ui.button("step").clicked() {
                self.running = false;
                self.pending = None;
                self.step_once();
            }
            if ui.button("new weights").clicked() {
                self.reset_weights();
                self.running = true;
            }
            if ui.button("another row").clicked() {
                let run = self.shape.shortest + self.rng.below(self.shape.longest - self.shape.shortest + 1);
                let mut rng = Rng::new(self.rng.next_u64());
                self.show_run(run, &mut rng);
            }
        });
        // what is going on, always two lines or so, so that nothing below jumps
        let status = match (self.pending, self.running, self.stopped) {
            _ if self.missing.is_some() => self.missing.unwrap_or_default().to_string(),
            (Some(k), _, _) => format!("Training first; \"{}\" follows as soon as it has learned.", Self::EXPERIMENTS[k].0),
            (None, true, _) if self.keep_going => "Training on, although it already gets the repeats right; press \"pause\" to stop.".to_string(),
            (None, true, _) if self.stop_when_learned => "Training. It pauses by itself once it gets the repeats right; press \"pause\" to stop it sooner.".to_string(),
            (None, true, _) => "Training without end: it stops only when you press \"pause\", or tick the box below.".to_string(),
            (None, false, true) if self.gave_up => format!("Paused: {} steps on, it still does not get the repeats right. Press \"train\" to let it keep trying.", count(GIVE_UP as usize)),
            (None, false, true) => "It now gets the repeats right, so training has paused by itself. Press \"train\" to carry on.".to_string(),
            (None, false, false) => "Paused. Press \"train\" to carry on, or \"step\" to take one training step at a time.".to_string(),
        };
        ui::note(ui, &status);
        if ui.checkbox(&mut self.stop_when_learned, "pause once it has learned").changed() {
            self.stopped = false;
        }

        ui::section(ui, "Things to try");
        for (k, (label, look_for)) in Self::EXPERIMENTS.iter().enumerate() {
            if ui::experiment(ui, label, look_for) {
                self.try_experiment(k);
            }
        }

        ui::section(ui, "The words");
        ui::term(ui, "Residual stream:", "the running total kept at each position. Every part of the model reads from it and adds its result back to it; the last total is turned into the guess.");
        ui::term(ui, "Attention head:", "the part that moves information between positions. From each position it looks back at earlier ones (its pattern, the square maps), and copies something from where it looks.");
        ui::term(ui, "Induction head:", "a head that finds an earlier copy of the current letter and looks at what came after it, which is the letter to guess. The default model here grows one in its second layer; language models use the same trick to carry on repeated text.");
        ui::term(ui, "Logit lens:", "turning the stream into a guess before the model is finished, to see at what depth the answer appears.");
        ui::term(ui, "Direct logit attribution:", "splitting the final score of a letter into what each part added. Exact here, because the stream is a plain sum.");
        ui::term(ui, "Knocking out (ablation):", "silencing one part and seeing what breaks: a test of what the model needs, not just of what it shows.");
        ui::term(ui, "Patching:", "running the model on a damaged input and putting back one piece of the undamaged run, to find where the needed information sits.");

        ui::section(ui, "Knock out heads");
        ui.horizontal_wrapped(|ui| {
            for k in 0..self.shape.head_count() {
                if ui.selectable_label(self.knocked[k], RichText::new(format!("head {}", self.shape.head_name(k))).size(12.5)).clicked() {
                    self.knocked[k] = !self.knocked[k];
                    self.changed();
                }
            }
            if self.any_knocked() && ui.button(RichText::new("restore all").size(12.5)).clicked() {
                self.knocked.fill(false);
                self.changed();
            }
        });
        let mut how = if self.how == Knock::Mean { 0 } else { 1 };
        if ui::choice(ui, &["replace with its average", "replace with nothing"], &mut how) {
            self.how = if how == 0 { Knock::Mean } else { Knock::Zero };
            self.changed();
        }
        ui::note(ui, "A knocked-out head's output is replaced at every position by its average over many rows (it still says what it always says, but nothing about this row), or by nothing at all, which is harsher.");
        let mut position = self.selected + 1;
        if ui::slider_int(ui, "position studied", &mut position, 1..=self.shape.len - 1) {
            self.selected = position - 1;
            self.hover = None;
            self.changed();
        }

        ui::section(ui, "The model");
        let before = self.shape;
        ui::slider_int(ui, "layers", &mut self.shape.layers, 1..=3);
        let mut heads = [1, 2, 4].iter().position(|&h| h == self.shape.heads).unwrap_or(1);
        ui.horizontal(|ui| {
            ui.label(RichText::new("heads per layer").size(12.5).color(pal::DIM.egui()));
            ui::choice(ui, &["1", "2", "4"], &mut heads);
        });
        self.shape.heads = [1, 2, 4][heads];
        let mut width = [16, 32, 48].iter().position(|&w| w == self.shape.width).unwrap_or(1);
        ui.horizontal(|ui| {
            ui.label(RichText::new("width of the stream").size(12.5).color(pal::DIM.egui()));
            ui::choice(ui, &["16", "32", "48"], &mut width);
        });
        self.shape.width = [16, 32, 48][width];
        ui.checkbox(&mut self.shape.mlp, "an MLP after each layer's attention");
        ui::note(ui, "Attention alone is enough for this task, and leaves less to look at. An MLP works on one position at a time; it cannot move information between positions.");
        ui::section(ui, "The task");
        ui::slider_int(ui, "different letters", &mut self.shape.vocab, 6..=26);
        ui::slider_int(ui, "letters in a row", &mut self.shape.len, 10..=32);
        ui::slider_int(ui, "shortest run", &mut self.shape.shortest, 2..=12);
        ui::slider_int(ui, "longest run", &mut self.shape.longest, 2..=12);
        ui::note(ui, "The run's length changes from row to row. That is what forces the model to compare letters: with one fixed length it could simply look a fixed distance back.");
        self.reshape(before);
        ui::section(ui, "Training");
        ui::slider_log(ui, "learning rate", &mut self.lr, 0.0005..=0.02);
        ui::slider_int_log(ui, "rows per step", &mut self.batch_size, 4..=64);
        ui::slider_int_log(ui, "steps per second", &mut self.rate, 5..=2000);
        ui::note(ui, "Slow enough to watch by default. Training shares each frame with the drawing, so a high setting gives as many steps as fit.");
        if self.pace.spacing_out() {
            ui::note(
                ui,
                &format!(
                    "One training step of a model this size takes about {:.0} ms, more than a frame can spare. So steps are spaced out to take at most a third of the time: training runs slower than asked (see the numbers below), and the frames between steps are not held up.",
                    self.pace.step_cost() * 1e3
                ),
            );
        }

        ui::section(ui, "Colours");
        ui::key(
            ui,
            &[
                (Color::hex(0x1F7A4D), "guess row: the guess is right"),
                (Color::hex(0x9C2F3F), "guess row: the guess is wrong"),
                (Cmap::Viridis.color(0.95), "lens and patching: near 1 (dark purple is near 0)"),
                (Cmap::Magma.color(0.85), "square maps: a large share of the head's attention"),
                (pal::GREEN, "bars: this part raises the right letter's score"),
                (pal::PINK, "bars: this part lowers it"),
                (pal::RED, "bar under a head's map: how far the loss rises with that head knocked out"),
            ],
        );
        ui::note(ui, "Each head has its own colour, used for its arcs, its map and its bar. Letters have colours only so that repeats are easy to spot.");

        ui::section(ui, "Numbers");
        ui::readout(ui, "training steps", count(self.steps as usize));
        ui::readout(ui, "steps per second, achieved", if self.running { format!("{:.0}", self.speed) } else { "-".to_string() });
        ui::readout(ui, &format!("loss on {EVAL} test rows"), loss_text(self.view.intact.0));
        ui::readout(ui, &format!("repeats right on {EVAL} test rows"), format!("{:.1} %", self.view.intact.1 * 100.0));
        if self.any_knocked() {
            ui::readout(ui, "the same, with the knock-outs", format!("{:.1} %", self.view.base.1 * 100.0));
        }
        ui::readout(ui, "weights", count(self.net.n_params()));
        // (the test rows are drawn like the training rows; with few letters and short
        // rows, training soon draws every row there is)
        let possible = self.possible_rows();
        if possible <= self.rows_drawn as f64 {
            ui::note(ui, &format!("With these settings only {} different rows are possible, and training has drawn {}: it has probably seen the test rows too, so they no longer show how it does on rows that are new to it.", count(possible as usize), count(self.rows_drawn as usize)));
        }
        ui::see_also(ui, &[("superposition", "superposition"), ("sae", "sparse autoencoders"), ("grokking", "grokking"), ("belief", "the Jacobian lens"), ("mlp", "network playground")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the pointer belongs to the picture: there is nothing to pan or zoom
        f.handles.capture();

        // train, within the frame's allowance
        if self.running && f.dt > 0.0 {
            let mut turn = self.pace.begin(self.rate as f32, f.dt, f.real_dt, 0.004);
            while turn.next() {
                self.step_once();
            }
            let done = self.pace.end(turn);
            if f.real_dt > 0.0 {
                self.speed += (done as f32 / f.real_dt - self.speed) * 0.05;
            }
        }

        let lay = Lay::new(f.view_size(), f.top_inset, &self.shape, 2 + self.shape.head_count() + if self.shape.mlp { self.shape.layers } else { 0 });
        // the pointer: a column under it is in focus, a click selects it; a click
        // on a head's map knocks that head out or restores it
        let at = f.ptr.pos.map(|p| p / f.px);
        let hover = at.and_then(|p| lay.column(p, &self.shape)).filter(|&t| t + 1 < self.shape.len);
        if hover != self.hover {
            self.hover = hover;
            self.stale_shown = true;
        }
        if f.ptr.clicked {
            if let Some(t) = hover {
                if t != self.selected {
                    self.selected = t;
                    self.changed();
                }
            } else if let Some(k) = at.and_then(|p| lay.map(p)) {
                self.knocked[k] = !self.knocked[k];
                self.changed();
            }
        }

        // the pictures and the slower measurements: left out for a while after they
        // have taken long (a large model), so that they never set the frame rate
        if !self.running || self.rest.over() {
            let start = clock::now();
            if self.stale_shown {
                self.refresh_shown();
            }
            self.pump();
            self.rest = clock::Rest::after(clock::now() - start);
        
        }

        self.consider_stopping();
        self.draw(f, &lay);
    }
}


/// How fast the default model trains in this build, in steps per second (for
/// `eidos bench-training` and its browser twin, `tools/wasm-bench.mjs`).
pub fn benchmark() -> String {
    let (cfg, task) = (DEFAULT.config(), DEFAULT.task());
    let mut rng = Rng::new(1);
    let mut net = Transformer::new(cfg, &mut rng);
    let mut batch = Batch::sample(&task, 16, &mut rng);
    for _ in 0..5 {
        net.train_step(&batch, 0.005, 0.0);
    }
    let (start, mut steps) = (crate::clock::now(), 0u32);
    while crate::clock::now() - start < 0.5 {
        batch.refill(&task, 16, &mut rng);
        net.train_step(&batch, 0.005, 0.0);
        steps += 1;
    }
    let rate = steps as f64 / (crate::clock::now() - start);
    format!("  {:<52} {rate:>9.0}\n", "transformer, 2 layers x 1 head, width 32, 16 rows")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Take every measurement the picture shows.
    fn measure(scene: &mut Opened) {
        scene.changed();
        // (as a frame does: the pictures when they are stale, then one job)
        while !scene.settled() {
            if scene.stale_shown {
                scene.refresh_shown();
            }
            scene.pump();
        }
    }

    /// Train for a fixed number of steps, then measure.
    fn trained(shape: Shape, seed: u64, steps: usize) -> Opened {
        let mut scene = Opened::with(shape, seed);
        for _ in 0..steps {
            scene.step_once();
        }
        measure(&mut scene);
        scene
    }

    /// Train until the scene would pause by itself (looking every tenth step);
    /// returns the step it stopped at.
    fn train_until_learned(scene: &mut Opened) -> u64 {
        while scene.steps < 3000 {
            scene.step_once();
            if scene.steps.is_multiple_of(10) {
                measure(scene);
                if scene.ready_to_stop() {
                    return scene.steps;
                }
            }
        }
        panic!("not learned after {} steps: loss {}, right {}", scene.steps, scene.loss, scene.accuracy);
    }

    /// The scene's text, held to account on the run a visitor sees first.
    #[test]
    fn the_default_model_learns_the_task_the_way_the_text_says() {
        let mut scene = Opened::new();
        let len = scene.shape.len;
        // the loss starts where a blind guess would be, is slow to move, then falls away
        scene.step_once();
        assert!((scene.loss - (scene.shape.vocab as f32).ln()).abs() < 0.25, "first loss {}", scene.loss);
        for _ in 1..100 {
            scene.step_once();
        }
        assert!(scene.loss > 1.6, "loss after 100 steps {}", scene.loss);
        let stopped = train_until_learned(&mut scene);
        assert!(stopped > 200 && stopped < 350, "learned after {stopped} steps");
        assert!(scene.loss < 0.1 && scene.view.intact.1 > 0.97, "loss {}, right on the test rows {}", scene.loss, scene.view.intact.1);
        assert!(scene.net.n_params() > 9_000 && scene.net.n_params() < 11_000, "about ten thousand weights");

        // the first run is not scored, so nothing is claimed there ("nothing to go on
        // yet"): every letter in it is new, and where the run ends cannot be told
        let (run, v) = (scene.run, &scene.view);
        assert!(v.right[..run].iter().all(|p| p.is_nan()) && v.right[run..len - 1].iter().all(|p| p.is_finite()), "{:?}", v.right);
        assert!(v.right[run..len - 1].iter().all(|&p| p > 0.5), "{:?}", v.right);

        // one head per job: the first looks a letter back, the second at the answer
        let roles = (scene.finder(), scene.helper());
        let v = &scene.view;
        assert_eq!(habit(v.habits[0]).map(|h| h.0), Some(0), "{:?}", v.habits[0]);
        assert_eq!(habit(v.habits[1]).map(|h| h.0), Some(1), "{:?}", v.habits[1]);
        assert!(v.habits[0][0] > 0.55 && v.habits[1][1] > 0.9, "{:?}", v.habits);
        assert_eq!(roles, (Some(1), Some(0)));
        // "points straight at the letter that followed the earlier copy"
        let (t, s) = (scene.selected, scene.selected - scene.run);
        assert!(v.att[1][t * len + s + 1] > 0.9, "head 2a at the answer: {}", v.att[1][t * len + s + 1]);
        // both are needed (without either, worse than a blind guess at 2.77)...
        assert!(v.solo[0] > 3.0 && v.solo[1] > 2.5, "{:?}", v.solo);
        // ...but only the second writes the answer
        let part = |name: &str| v.parts.iter().find(|p| p.0 == name).map(|p| p.1).unwrap();
        assert!(part("head 2a") > 8.0 && part("head 1a").abs() < 1.0, "{:?}", v.parts);
        assert!(v.parts.iter().filter(|p| p.2.is_none()).all(|p| p.1.abs() < 0.6), "{:?}", v.parts);

        // the logit lens: the answer is not there until the last layer has run
        let scored: Vec<usize> = (0..len).filter(|&t| scene.seq.mask[t] > 0.0).collect();
        let sure = |point: usize| scored.iter().map(|&t| v.lens[point * len + t]).sum::<f32>() / scored.len() as f32;
        assert!(sure(0) < 0.15 && sure(1) < 0.2 && sure(2) > 0.95, "{} {} {}", sure(0), sure(1), sure(2));

        // patching: the swapped letter, then the position after it, then the last position
        assert_eq!(scene.seq.tokens[s], scene.seq.tokens[t]);
        assert!(v.traced && v.damaged < 0.4, "the swap loses the answer: {}", v.damaged);
        let patch = |point: usize, pos: usize| v.patch[point * len + pos];
        assert!(patch(0, s) > 0.9 && patch(1, s + 1) > 0.9, "{:?}", v.patch);
        for (point, on_path) in [(0, s), (1, s + 1)] {
            for pos in (0..=t).filter(|&pos| pos != on_path) {
                assert!(patch(point, pos) < 0.5, "point {point} position {pos}: {}", patch(point, pos));
            }
        }
        // (at the end only the final position matters: putting back the final stream
        // there restores the answer by construction, and no other position's final
        // stream reaches it; a check of the arithmetic, not of the model)
        assert!(patch(2, t) > 0.999 && (0..t).all(|pos| patch(2, pos) < 1e-3), "{:?}", v.patch);
    }

    #[test]
    fn knocking_out_either_head_breaks_it_and_restoring_them_mends_it() {
        let mut scene = Opened::new();
        train_until_learned(&mut scene);
        let before = (scene.view.solo.clone(), scene.view.habits_now.clone(), scene.view.right.clone(), scene.view.lens.clone());
        let len = scene.shape.len;
        // the head that finds the answer
        scene.try_experiment(2);
        assert_eq!(scene.knocked, [false, true]);
        measure(&mut scene);
        assert!(scene.view.base.1 < 0.25, "right without the finder: {}", scene.view.base.1);
        let last_row = &scene.view.lens[2 * len..];
        assert!(last_row.iter().filter(|p| p.is_finite()).all(|&p| p < 0.4), "the answer no longer appears: {last_row:?}");
        assert!(scene.view.found < 0.5, "nothing to trace by patching");
        // its helper in the first layer
        scene.try_experiment(3);
        assert_eq!(scene.knocked, [true, false]);
        measure(&mut scene);
        assert!(scene.view.base.1 < 0.25, "right without the helper: {}", scene.view.base.1);
        // the second-layer head no longer looks at the answer, and what is written
        // under its map (measured with the knock-out, like the map) says so
        let (now, then) = (scene.view.habits_now[1], scene.view.habits[1]);
        assert!(now[1] < 0.3 && then[1] > 0.9 && habit(now).map(|h| h.0) != Some(1), "with the helper out {now:?}, with it {then:?}");
        // everything back: every measurement is as it was before the knock-outs
        scene.try_experiment(4);
        assert_eq!(scene.knocked, [false, false]);
        measure(&mut scene);
        let after = (scene.view.solo.clone(), scene.view.habits_now.clone(), scene.view.right.clone(), scene.view.lens.clone());
        assert!(format!("{before:?}") == format!("{after:?}"), "not mended: {before:?} vs {after:?}");
        // the harsher kind of knock-out breaks it too
        scene.how = Knock::Zero;
        scene.knocked[0] = true;
        measure(&mut scene);
        assert!(scene.view.base.1 < 0.25, "{}", scene.view.base.1);

        // with no head that looks at the answer, the experiments that knock it (or its
        // helper) out say so and knock out nothing
        scene.knocked.fill(false);
        measure(&mut scene);
        scene.view.habits.iter_mut().for_each(|h| *h = [0.3, 0.3, 0.3]);
        for k in [2, 3] {
            scene.try_experiment(k);
            assert!(scene.knocked.iter().all(|k| !k) && scene.missing.is_some());
        }
    }

    /// The first-layer head the finder depends on is picked with nothing else
    /// knocked out, whatever is knocked out at the time.
    #[test]
    fn the_helper_is_measured_alone() {
        let mut scene = Opened::with(Shape { heads: 2, ..DEFAULT }, 1);
        train_until_learned(&mut scene);
        let helper = scene.helper();
        assert!(helper.is_some_and(|k| k < 2), "{helper:?}");
        for out in [vec![2, 3], vec![0], vec![1], vec![0, 1]] {
            scene.knocked.fill(false);
            out.iter().for_each(|&k| scene.knocked[k] = true);
            measure(&mut scene);
            assert_eq!(scene.helper(), helper, "with {out:?} knocked out");
        }
    }

    /// The experiments are written about the default model: from any other, they
    /// go back to it (and to the default training) first.
    #[test]
    fn the_experiments_go_back_to_the_default_model() {
        // two heads a layer, learned: "knock out the head that finds the answer"
        // starts again with the model its caption is about
        let mut scene = Opened::with(Shape { heads: 2, ..DEFAULT }, 1);
        train_until_learned(&mut scene);
        scene.try_experiment(2);
        assert_eq!((scene.shape, scene.pending, scene.knocked.len()), (DEFAULT, Some(2), 2));
        // and every experiment trains the default way
        for k in 0..6 {
            let mut scene = Opened::with(Shape { heads: 2, width: 48, ..DEFAULT }, 1);
            (scene.lr, scene.batch_size) = (0.02, 64);
            scene.try_experiment(k);
            let want = if k == 5 { Shape { layers: 1, ..DEFAULT } } else { DEFAULT };
            assert_eq!((scene.shape, scene.lr, scene.batch_size), (want, LR, ROWS), "experiment {k}");
            assert_eq!(scene.pending, (1..5).contains(&k).then_some(k), "experiment {k}");
        }
    }

    /// A slider held past where `tidy` lets it go starts again once, when the
    /// value changes, and not on every frame after that.
    #[test]
    fn a_slider_held_past_its_limit_keeps_the_model() {
        let mut scene = trained(DEFAULT, 1, 20);
        let seed = scene.seed;
        let hold = |scene: &mut Opened, shortest: usize, longest: usize| {
            for _ in 0..3 {
                let before = scene.shape;
                (scene.shape.shortest, scene.shape.longest) = (shortest, longest);
                scene.reshape(before);
                scene.step_once();
            }
        };
        // "shortest run" dragged past the longest (9): one new model, of runs 9 to 9
        hold(&mut scene, 12, 9);
        assert_eq!((scene.shape.shortest, scene.seed, scene.steps), (9, seed + 1, 3));
        // "longest run" dragged past 10, half the row: one more
        hold(&mut scene, 9, 12);
        assert_eq!((scene.shape.longest, scene.seed, scene.steps), (10, seed + 2, 3));
        // and held there, nothing more
        hold(&mut scene, 9, 12);
        assert_eq!((scene.seed, scene.steps), (seed + 2, 6));
    }

    #[test]
    fn one_layer_does_not_learn_it() {
        let scene = trained(Shape { layers: 1, ..DEFAULT }, 1, 700);
        let (loss, right) = scene.view.intact;
        assert!(right > 0.35 && right < 0.6 && loss > 1.0, "one layer: loss {loss}, right {right}");
        assert!(scene.finder().is_none() && !scene.learned());
    }

    /// "It stalls with only a little over half of the repeats right, however long it
    /// trains" (run with `--ignored`: it trains for a while).
    #[test]
    #[ignore]
    fn one_layer_stalls_however_long_it_trains() {
        let shape = Shape { layers: 1, ..DEFAULT };
        // (more rows than the scene's twelve, for a steadier measure)
        let test = Batch::sample(&shape.task(), 400, &mut Rng::new(1234));
        for seed in 1..=3 {
            let mut scene = Opened::with(shape, seed);
            let mut right = Vec::new();
            for steps in [GIVE_UP, 2 * GIVE_UP] {
                while scene.steps < steps {
                    scene.step_once();
                }
                right.push(scene.net.evaluate(&test, &Edits::none()).1);
            }
            assert!(right.iter().all(|&r| r > 0.5 && r < 0.65) && right[1] - right[0] < 0.08, "seed {seed}: right after {GIVE_UP} and twice as many steps: {right:?}");
        }
    }

    /// "It pauses by itself after about 300 steps", from whatever seed "new weights"
    /// or "watch it learn" gives it (run with `--ignored`).
    #[test]
    #[ignore]
    fn the_default_model_learns_in_about_300_steps_from_any_seed() {
        let stopped: Vec<u64> = (2..=9).map(|seed| train_until_learned(&mut Opened::with(DEFAULT, seed))).collect();
        assert!(stopped.iter().all(|&s| (200..=350).contains(&s)), "{stopped:?}");
    }

    #[test]
    fn training_pauses_by_itself_when_learned_or_when_getting_nowhere() {
        let mut scene = Opened::new();
        train_until_learned(&mut scene);
        scene.consider_stopping();
        assert!(!scene.running && scene.stopped && !scene.gave_up);
        // a model that cannot learn the task is given up on, not trained for ever
        let mut one = Opened::with(Shape { layers: 1, ..DEFAULT }, 1);
        for _ in 0..50 {
            one.step_once();
        }
        measure(&mut one);
        one.consider_stopping();
        assert!(one.running);
        one.since = GIVE_UP;
        one.consider_stopping();
        assert!(!one.running && one.gave_up);
        // unless the box is unticked
        one.running = true;
        one.stop_when_learned = false;
        one.consider_stopping();
        assert!(one.running);
    }

    #[test]
    fn an_experiment_that_needs_a_trained_model_waits_for_one() {
        let mut scene = Opened::new();
        scene.running = false;
        scene.try_experiment(2);
        assert_eq!(scene.pending, Some(2));
        assert!(scene.running && scene.knocked.iter().all(|k| !k));
        // from a one-layer model it goes back to two layers first
        let mut scene = Opened::with(Shape { layers: 1, ..DEFAULT }, 1);
        scene.try_experiment(3);
        assert_eq!((scene.shape, scene.pending), (DEFAULT, Some(3)));
    }

    #[test]
    fn everything_fits_in_the_view() {
        let largest = Shape { layers: 3, heads: 4, width: 48, mlp: true, vocab: 26, len: 32, shortest: 4, longest: 12 };
        let shapes = [DEFAULT, Shape { heads: 2, ..DEFAULT }, Shape { heads: 4, ..DEFAULT }, Shape { layers: 3, heads: 2, ..DEFAULT }, largest];
        // (with the canvas of a 1600 x 1000 and of a 1920 x 1080 window, both panels open)
        let views = [(980.0, 940.0), (580.0, 700.0), (390.0, 800.0), (1400.0, 1000.0), (760.0, 620.0), (1070.0, 1000.0), (1398.0, 1080.0), (3000.0, 700.0), (1070.0, 600.0)];
        for (w, h) in views {
            let view = Vec2::new(w, h);
            for shape in shapes {
                let rows = 2 + shape.head_count() + if shape.mlp { shape.layers } else { 0 };
                let lay = Lay::new(view, if view.x < 900.0 { 46.0 } else { 0.0 }, &shape, rows);
                let inside = |min: Vec2, max: Vec2| min.x >= 0.0 && min.y >= 0.0 && max.x <= view.x + 0.5 && max.y <= view.y + 0.5 && max.x > min.x && max.y > min.y;
                assert!(lay.x0 + lay.col * shape.len as f32 <= view.x - lay.pad + 0.5, "{view}: the columns");
                for m in &lay.maps {
                    assert!(inside(*m - Vec2::new(0.0, OVER_MAP), *m + Vec2::new(lay.side, lay.side + UNDER_MAP - 8.0)), "{view} {shape:?}: a map at {m} of side {}", lay.side);
                }
                assert!(inside(lay.parts.0, lay.parts.1), "{view} {shape:?}: the bars {:?}", lay.parts);
                if let Some((min, max)) = lay.chart {
                    assert!(inside(min, max), "{view} {shape:?}: the chart {:?}", lay.chart);
                }
                // what the scene starts with is shown whole, everywhere; and a desktop
                // window of the usual size shows every part of every model
                if shape == DEFAULT || (w, h) == (1070.0, 1000.0) {
                    assert!(lay.part_rows == rows && lay.chart.is_some(), "{view} {shape:?}: {} of {rows} bars, chart {}", lay.part_rows, lay.chart.is_some());
                }
                // (a short view that also gives up its top to the application's buttons
                // has room only for the smallest maps: that they fit is checked above)
                if shape.head_count() <= 4 && view.y >= 700.0 {
                    assert!(lay.side >= 76.0, "{view} {shape:?}: maps of side {}", lay.side);
                }
                // the maps are not squeezed for a card that is too tall beside them
                if (w, h) == (3000.0, 700.0) {
                    assert!(lay.side >= 200.0, "{view} {shape:?}: maps of side {}", lay.side);
                }
                // the pointer finds what is drawn
                assert_eq!(lay.column(Vec2::new(lay.centre(7), lay.tokens + 3.0), &shape), Some(7));
                assert_eq!(lay.column(Vec2::new(lay.centre(7), lay.patch + 3.0), &shape), Some(7));
                assert_eq!(lay.map(*lay.maps.last().unwrap() + Vec2::splat(5.0)), Some(shape.head_count() - 1));
                assert_eq!(lay.map(Vec2::new(lay.centre(7), lay.tokens + 3.0)), None);
            }
        }
    }

    /// "Only so many different rows are possible": the count is that of the rows
    /// the task really makes.
    #[test]
    fn the_possible_rows_are_counted_right() {
        let shape = Shape { vocab: 6, len: 10, shortest: 2, longest: 3, ..DEFAULT };
        let scene = Opened::with(shape, 1);
        let (task, mut rng) = (shape.task(), Rng::new(3));
        let rows: std::collections::HashSet<Vec<u32>> = (0..5000).map(|_| Batch::sample(&task, 1, &mut rng).tokens).collect();
        assert_eq!((scene.possible_rows(), rows.len()), (150.0, 150));
        // six letters, ten in a row (runs of 4 and 5 once tidied)
        let mut shape = Shape { vocab: 6, len: 10, ..DEFAULT };
        shape.tidy();
        assert_eq!(Opened::with(shape, 1).possible_rows(), 1080.0);
    }

    #[test]
    fn the_task_stays_possible_whatever_the_sliders_say() {
        let mut shape = Shape { vocab: 6, len: 10, shortest: 12, longest: 12, ..DEFAULT };
        shape.tidy();
        assert_eq!((shape.shortest, shape.longest), (5, 5));
        // a scene of that shape runs, and patching has a spare letter to swap in
        let scene = trained(shape, 3, 5);
        assert!(scene.corrupted().is_some());
    }

    /// What a run of training looks like, step by step, and what the heads become
    /// (run with `--ignored --nocapture`).
    #[test]
    #[ignore]
    fn explore() {
        for (heads, lr) in [(1usize, 0.005f32), (2, 0.005)] {
            for seed in 1..=4u64 {
                let shape = Shape { heads, ..DEFAULT };
                let mut scene = Opened::with(shape, seed);
                scene.lr = lr;
                let mut trace = String::new();
                for step in 1..=1500 {
                    scene.step_once();
                    if step % 20 == 0 && step <= 600 {
                        trace += &format!(" {:.2}", scene.loss);
                    }
                }
                measure(&mut scene);
                let (finder, helper) = (scene.finder(), scene.helper());
                let v = &scene.view;
                println!("heads {heads} lr {lr} seed {seed}: loss every 20 steps:{trace}");
                println!("   test rows: loss {:.3}, right {:.3}; finder {:?} helper {:?}", v.intact.0, v.intact.1, finder.map(|k| scene.shape.head_name(k)), helper.map(|k| scene.shape.head_name(k)));
                for k in 0..shape.head_count() {
                    println!("   head {}: before {:.2} answer {:.2} copy {:.2}; loss without it {:.2}", scene.shape.head_name(k), v.habits[k][0], v.habits[k][1], v.habits[k][2], v.solo[k]);
                }
                let len = shape.len;
                for point in 0..=shape.layers {
                    let row: Vec<String> = (0..=scene.selected).map(|t| if v.patch[point * len + t].is_nan() { " . ".to_string() } else { format!("{:.2}", v.patch[point * len + t]) }).collect();
                    println!("   patch at point {point}: {}", row.join(" "));
                }
                let parts: Vec<String> = v.parts.iter().map(|p| format!("{} {:+.1}", p.0, p.1)).collect();
                println!("   parts: {}", parts.join("; "));
            }
        }
    }
}

//! Probing: fit a simple read-out to a layer's activations to find out what can be
//! read off them. And the caution that goes with it: that something can be read off
//! a layer does not show that the network uses it. Intervening does: push every
//! point's activations across the read-out's boundary, so that the read-out says the
//! opposite of what it said, and see whether the network's answers change.

use super::*;
use crate::clock::Pace;
use crate::math::linalg;
use crate::math::nn::{Act, Loss, Mlp};
use crate::scenes::neural::{make_data, Data, CLASS, DATASETS};

pub const DEF: SceneDef = SceneDef {
    id: "probe",
    title: "Probing a network",
    group: GROUP,
    blurb: "What can be read off each layer, and whether the network actually uses it.",
    make: |g| Box::new(Probing::new(g)),
};

const HIDDEN: usize = 3;
const WIDTH: usize = 8;
/// Input, the hidden layers, output.
const LAYERS: usize = HIDDEN + 2;
const LAYER_NAMES: [&str; LAYERS] = ["input", "layer 1", "layer 2", "layer 3", "output"];
const FIELD_N: usize = 56;
const DATA_R: f32 = 1.15;
const N_TRAIN: usize = 260;
const N_TEST: usize = 200;

/// The properties probed for: the name in the table, what the dots of the picture
/// then show (one line under it), and what it is in full (the panel).
const PROPERTIES: [(&str, &str, &str); 5] = [
    ("its class", "which class the point belongs to", "the class the network is trained to tell: blue or orange"),
    ("right half", "whether the point is right of the centre", r"whether the point is right of the centre ($x_1 > 0$)"),
    ("far from centre", "whether the point is in the outer half", "whether the point is further from the centre than half of the points are"),
    ("same-sign corner", r"whether $x_1$ and $x_2$ have the same sign", r"whether $x_1$ and $x_2$ have the same sign: the top-right and bottom-left quarters"),
    ("random labels", "a coin tossed for each point", "a coin tossed for each point. There is nothing to find, so this row shows what failure looks like"),
];

/// A read-out counts as working from this accuracy (the two kinds counting equally,
/// see [`Probe::accuracy`]). Only a working one is pushed across: where a straight
/// line cannot tell the two kinds apart, there is no "across".
const READABLE: f32 = 0.8;
/// What pushing the points across does is measured as the share of the network's
/// answers that change; from this share on, the verdict says "most".
const MOST: f32 = 0.5;

/// The network and data every "thing to try" uses, so that what a caption says is
/// what is seen. ("new network" draws others; the tests check the captions on a
/// dozen of them.)
const SEED: u64 = 5;
/// The noise in the data and the learning rate they start from, and that every
/// "thing to try" goes back to.
const NOISE: f32 = 0.15;
const RATE: f32 = 0.02;

fn dim(layer: usize) -> usize {
    match layer {
        0 => 2,
        l if l <= HIDDEN => WIDTH,
        _ => 1,
    }
}

/// One point's activations, layer by layer: `[0]` is the input, `[LAYERS - 1][0]` the
/// output. Only the first `dim(layer)` numbers of each are used.
type Acts = [[f32; WIDTH]; LAYERS];

/// Run the network on from layer `from`, whose activations are already in `acts`;
/// fills in every later layer and returns the output.
fn run(net: &Mlp, from: usize, acts: &mut Acts) -> f32 {
    for l in from..LAYERS - 1 {
        let (n_in, n_out) = (dim(l), dim(l + 1));
        let prev = acts[l];
        for j in 0..n_out {
            let z = net.b[l][j] + net.w[l][j * n_in..(j + 1) * n_in].iter().zip(&prev).map(|(w, h)| w * h).sum::<f32>();
            acts[l + 1][j] = if l + 2 == LAYERS { z } else { net.act.f(z) };
        }
    }
    acts[LAYERS - 1][0]
}

/// Every point of `data` through the network at once: each point's activations at
/// every layer. (The network's own batch pass; `run` is only for carrying on from a
/// layer whose activations have been changed.)
fn activations(net: &mut Mlp, data: &Data) -> Vec<Acts> {
    let n = data.len();
    net.forward(&data.x, n);
    let mut all = vec![[[0.0f32; WIDTH]; LAYERS]; n];
    for (i, acts) in all.iter_mut().enumerate() {
        acts[0][0] = data.x[2 * i];
        acts[0][1] = data.x[2 * i + 1];
    }
    for layer in 1..LAYERS {
        for j in 0..dim(layer) {
            for (acts, &value) in all.iter_mut().zip(net.neuron(layer - 1, j)) {
                acts[layer][j] = value;
            }
        }
    }
    all
}

/// Does point `p` have property `k`? (`coin` is the point's random label.)
fn truth(k: usize, p: Vec2, class: f32, median_radius: f32, coin: bool) -> bool {
    match k {
        0 => class > 0.5,
        1 => p.x > 0.0,
        2 => p.length() > median_radius,
        3 => p.x * p.y > 0.0,
        _ => coin,
    }
}

/// A linear read-out of one layer: "yes" where `w . h + b > 0`.
#[derive(Clone, Copy, Default)]
struct Probe {
    w: [f32; WIDTH],
    b: f32,
    /// How often it is right on the held-out points, the two kinds counting equally:
    /// the average of its hit rates on the points that have the property and on those
    /// that do not. So always giving the same answer scores 50 %, however rare the
    /// property is.
    accuracy: f32,
    /// The average activations of the points that have the property, less those of
    /// the points that do not: the line the points are pushed along.
    shift: [f32; WIDTH],
    /// The share of the network's answers on the held-out points that change once
    /// every one of them has been pushed across this read-out's boundary (see
    /// [`Probe::push`]). Not a number where nothing is pushed: where the read-out does
    /// not work, and at the output, after which there is no network left to run.
    changed: f32,
}

impl Probe {
    fn read(&self, h: &[f32; WIDTH], d: usize) -> f32 {
        self.b + self.w[..d].iter().zip(h).map(|(w, h)| w * h).sum::<f32>()
    }

    /// How often it is right about the points whose activations are `acts` (at
    /// `layer`), the two kinds counting equally: its hit rate on each kind, averaged.
    /// (A kind with no points here is left out of the average.)
    fn score(&self, acts: &[Acts], layer: usize, truth: &[bool]) -> f32 {
        let (mut hits, mut count) = ([0.0f32; 2], [0.0f32; 2]);
        for (a, &t) in acts.iter().zip(truth) {
            count[t as usize] += 1.0;
            hits[t as usize] += ((self.read(&a[layer], dim(layer)) > 0.0) == t) as u8 as f32;
        }
        let rates: Vec<f32> = (0..2).filter(|&t| count[t] > 0.0).map(|t| hits[t] / count[t]).collect();
        rates.iter().sum::<f32>() / rates.len().max(1) as f32
    }

    /// How far apart the read-out puts the two kinds on average: what one step of
    /// `shift` changes its reading by.
    fn gap(&self, d: usize) -> f32 {
        self.w[..d].iter().zip(&self.shift).map(|(w, s)| w * s).sum()
    }

    /// Push one point's activations across the boundary: along `shift` (the line
    /// joining the two kinds' averages, where the activations of the two kinds
    /// actually differ), just as far as it takes for the read-out to read the opposite
    /// of what it read, as firmly. Each point is moved by its own amount, so a group
    /// made of several clusters crosses as well as one made of a single cluster.
    /// (Moving every point by the same `shift` instead would slide the whole cloud,
    /// and change more than the read-out looks at.)
    fn push(&self, h: &mut [f32; WIDTH], d: usize) {
        let along = 2.0 * self.read(h, d) / self.gap(d);
        h[..d].iter_mut().zip(&self.shift).for_each(|(h, s)| *h -= along * s);
    }

    /// The least-squares read-out of `truth` (as +1 / -1) from the `d` numbers in `h`,
    /// with the two kinds counting equally however many of each there are (as they do
    /// when it is scored), and a little ridge so that it is defined even for a layer
    /// whose neurons are stuck together.
    fn fit(h: &[[f32; WIDTH]], truth: &[bool], d: usize) -> Probe {
        let n = d + 1;
        let yes = truth.iter().filter(|t| **t).count();
        let share = |t: bool| 0.5 * h.len() as f64 / (if t { yes } else { h.len() - yes }).max(1) as f64;
        let (mut a, mut rhs) = (vec![0.0f64; n * n], vec![0.0f64; n]);
        for (row, &t) in h.iter().zip(truth) {
            let (y, weight) = (if t { 1.0 } else { -1.0 }, share(t));
            let value = |i: usize| if i < d { row[i] as f64 } else { 1.0 };
            for i in 0..n {
                rhs[i] += weight * value(i) * y;
                for j in 0..n {
                    a[i * n + j] += weight * value(i) * value(j);
                }
            }
        }
        for i in 0..d {
            a[i * n + i] += 1e-3 * h.len() as f64;
        }
        let mut probe = Probe::default();
        if let Some(x) = linalg::solve(&mut a, &mut rhs, n) {
            for i in 0..d {
                probe.w[i] = x[i] as f32;
            }
            probe.b = x[d] as f32;
        }
        probe
    }
}

/// The data, the network, and every read-out of it.
struct Lab {
    net: Mlp,
    train: Data,
    test: Data,
    /// Each point's random label (training points, then held-out points).
    coins: [Vec<bool>; 2],
    median: f32,
    /// Whether each point has each property, `truths[which][property][point]`
    /// (`which`: 0 for the training points, 1 for those held out). Training does not
    /// change these.
    truths: [[Vec<bool>; PROPERTIES.len()]; 2],
    /// `probes[property][layer]`.
    probes: [[Probe; LAYERS]; PROPERTIES.len()],
    /// The network's own accuracy on the held-out points, and its loss on the
    /// training points.
    accuracy: f32,
    loss: f32,
    /// Is each property, on this data, just the class under another name?
    same_as_class: [bool; PROPERTIES.len()],
    /// Has the network been trained at a learning rate other than [`RATE`], at which
    /// the "things to try" were measured?
    off_rate: bool,
}

impl Lab {
    fn new(kind: usize, noise: f32, seed: u64) -> Lab {
        let mut rng = Rng::new(seed);
        let train = make_data(kind, N_TRAIN, noise, &mut rng);
        let test = make_data(kind, N_TEST, noise, &mut rng);
        let coins = [(0..N_TRAIN).map(|_| rng.bool(0.5)).collect(), (0..N_TEST).map(|_| rng.bool(0.5)).collect()];
        let mut radii: Vec<f32> = (0..train.len()).map(|i| train.point(i).length()).collect();
        radii.sort_by(f32::total_cmp);
        let median = radii[radii.len() / 2];
        let net = Mlp::new(&[2, WIDTH, WIDTH, WIDTH, 1], Act::Tanh, &mut rng);
        let mut lab = Lab { net, train, test, coins, median, truths: Default::default(), probes: Default::default(), accuracy: 0.0, loss: 0.0, same_as_class: [false; PROPERTIES.len()], off_rate: false };
        for which in 0..2 {
            let data = lab.set(which);
            let truths: [Vec<bool>; PROPERTIES.len()] = std::array::from_fn(|k| (0..data.len()).map(|i| truth(k, data.point(i), data.y[i], lab.median, lab.coins[which][i])).collect());
            lab.truths[which] = truths;
        }
        for k in 1..PROPERTIES.len() {
            let agree = (0..lab.train.len()).filter(|&i| lab.truth(k, 0, i) == lab.truth(0, 0, i)).count() as f32 / lab.train.len() as f32;
            lab.same_as_class[k] = !(0.04..=0.96).contains(&agree);
        }
        lab.probe();
        lab
    }

    fn set(&self, which: usize) -> &Data {
        if which == 0 {
            &self.train
        } else {
            &self.test
        }
    }

    /// Property `k` of point `i` of the training (0) or held-out (1) points.
    fn truth(&self, k: usize, which: usize, i: usize) -> bool {
        self.truths[which][k][i]
    }

    /// One training step on all the training points.
    fn step(&mut self, lr: f32) {
        self.off_rate |= lr != RATE;
        self.loss = self.net.train(&self.train.x, &self.train.y, self.train.len(), lr, Loss::Logistic);
    }

    /// Fit every read-out to the training points and score it on the held-out ones;
    /// then, for each one that works, push the held-out points across it in turn and
    /// count the network's answers that change.
    fn probe(&mut self) {
        let train = activations(&mut self.net, &self.train);
        let test = activations(&mut self.net, &self.test);
        let says = |a: &Acts| a[LAYERS - 1][0] > 0.0;
        self.accuracy = test.iter().zip(&self.test.y).filter(|(a, y)| says(a) == (**y > 0.5)).count() as f32 / test.len() as f32;
        let mut probes = [[Probe::default(); LAYERS]; PROPERTIES.len()];
        let mut column = vec![[0.0f32; WIDTH]; train.len()];
        for layer in 0..LAYERS {
            let d = dim(layer);
            column.iter_mut().zip(&train).for_each(|(c, a)| *c = a[layer]);
            for (k, row) in probes.iter_mut().enumerate() {
                let targets = &self.truths[0][k];
                let mut probe = Probe::fit(&column, targets, d);
                probe.accuracy = probe.score(&test, layer, &self.truths[1][k]);
                // where the two kinds sit on average
                let (mut mean, mut count) = ([[0.0f32; WIDTH]; 2], [0.0f32; 2]);
                for (h, &t) in column.iter().zip(targets) {
                    count[t as usize] += 1.0;
                    (0..d).for_each(|j| mean[t as usize][j] += h[j]);
                }
                for j in 0..d {
                    probe.shift[j] = mean[1][j] / count[1].max(1.0) - mean[0][j] / count[0].max(1.0);
                }
                // (a working read-out reads the two kinds' averages well apart: about
                // 2, as it is fitted to +1 and -1; the guard is only against dividing
                // by nearly nothing)
                let pushed = layer + 1 < LAYERS && probe.accuracy >= READABLE && probe.gap(d) > 0.1;
                probe.changed = if !pushed {
                    f32::NAN
                } else {
                    let changed = test
                        .iter()
                        .filter(|a| {
                            let mut acts = **a;
                            probe.push(&mut acts[layer], d);
                            (run(&self.net, layer, &mut acts) > 0.0) != says(a)
                        })
                        .count();
                    changed as f32 / test.len() as f32
                };
                row[layer] = probe;
            }
        }
        self.probes = probes;
    }
}

struct Probing {
    lab: Lab,
    data_kind: usize,
    noise: f32,
    seed: u64,
    lr: f32,
    /// Steps per second asked for.
    rate: usize,
    running: bool,
    steps: u64,
    pace: Pace,
    /// The network's loss as it trains.
    history: History,
    /// The read-out on show: (property, layer).
    picked: (usize, usize),
    hover: Option<(usize, usize)>,
    field: Surface,
    /// The places of the plane the picture is worked out at, as the network takes them.
    grid: Vec<f32>,
    /// The weights or the data changed: every read-out is to be fitted again.
    stale: bool,
    /// The picture is to be drawn again (another read-out was chosen, or they were refitted).
    repaint: bool,
    /// Seconds of training since the read-outs were last fitted: they are fitted
    /// again once [`REFIT`] seconds have passed and the network has changed, however
    /// often the display redraws.
    since_fit: f32,
    /// Whether the view is phone-sized (not known until the first frame), and whether
    /// the camera should be told.
    narrow: Option<bool>,
    new_view: bool,
}

/// How often (in seconds of training) the read-outs are fitted again and the pushes
/// redone: all 25 read-outs, up to 20 pushes and the picture take a millisecond or so.
const REFIT: f32 = 0.1;

/// Where things are drawn: the big picture's centre and side; the table's top-left
/// corner and the size of one cell.
const FIELD: (Vec2, f32) = (Vec2::new(-2.25, 0.1), 3.7);
const TABLE: (Vec2, Vec2) = (Vec2::new(1.55, 1.6), Vec2::new(0.56, 0.56));

impl Probing {
    fn new(gfx: &Gfx) -> Self {
        let mut field = Surface::new(gfx, FIELD_N as u32, FIELD_N as u32);
        field.fill(|u, v| [u - 0.5, v - 0.5, 0.0, 0.0]);
        field.style.lit = 0.0;
        field.style.cmap = Some(Cmap::CoolWarm);
        field.style.contour = 1.0;
        field.style.contour_alpha = 0.9;
        field.style.contour_reach = 0.5;
        field.style.line_color = pal::WHITE;
        field.style.range = [-1.6, 1.6];
        let place = |i: usize| i as f32 / (FIELD_N - 1) as f32 * 2.0 * DATA_R - DATA_R;
        let grid = (0..FIELD_N * FIELD_N).flat_map(|i| [place(i % FIELD_N), place(i / FIELD_N)]).collect();
        Probing {
            lab: Lab::new(1, NOISE, SEED),
            data_kind: 1,
            noise: NOISE,
            seed: SEED,
            lr: RATE,
            rate: 240,
            running: true,
            steps: 0,
            pace: Pace::default(),
            history: History::new(600),
            picked: (1, 2),
            hover: None,
            field,
            grid,
            stale: true,
            repaint: true,
            since_fit: 0.0,
            narrow: None,
            new_view: false,
        }
    }

    fn restart(&mut self) {
        self.lab = Lab::new(self.data_kind, self.noise, self.seed);
        self.steps = 0;
        self.history.clear();
        self.pace.reset();
        self.stale = true;
    }

    /// One of the "things to try": always on the same network and data (see [`SEED`]),
    /// started afresh unless that is what is on show already and training may go on.
    fn set_up(&mut self, data: usize, picked: (usize, usize), running: bool) {
        // (the noise and the learning rate too: what the captions say was checked at
        // these, so a network trained at another rate is not the same network)
        let moved = self.noise != NOISE || self.lab.off_rate;
        (self.noise, self.lr) = (NOISE, RATE);
        if data != self.data_kind || self.seed != SEED || moved || !running {
            (self.data_kind, self.seed) = (data, SEED);
            self.restart();
        }
        (self.picked, self.running, self.repaint) = (picked, running, true);
    }

    fn cell(k: usize, layer: usize) -> (Vec2, Vec2) {
        let min = TABLE.0 + Vec2::new(layer as f32 * TABLE.1.x, -(k as f32 + 1.0) * TABLE.1.y);
        (min, min + TABLE.1 * Vec2::new(0.95, 0.92))
    }

    /// What the chosen read-out says at every place in the plane.
    fn paint(&mut self) {
        let (k, layer) = self.picked;
        let (probe, d) = (self.lab.probes[k][layer], dim(layer));
        self.lab.net.forward(&self.grid, FIELD_N * FIELD_N);
        let net = &self.lab.net;
        // the layer's numbers at every place: the two inputs themselves, or one
        // column per neuron from the pass just made
        let columns: Vec<&[f32]> = if layer == 0 { Vec::new() } else { (0..d).map(|j| net.neuron(layer - 1, j)).collect() };
        let mut h = [0.0f32; WIDTH];
        for (i, p) in self.field.data_mut().iter_mut().enumerate() {
            if layer == 0 {
                (h[0], h[1]) = (self.grid[2 * i], self.grid[2 * i + 1]);
            } else {
                for (value, column) in h.iter_mut().zip(&columns) {
                    *value = column[i];
                }
            }
            p[3] = probe.read(&h, d);
        }
    }

    /// What was measured about a read-out, in words: how well it reads, and what
    /// pushing the points across did to the network's answers.
    /// It says no more than that: in particular not what the network "uses".
    fn verdict(&self, k: usize, layer: usize) -> (&'static str, Color) {
        verdict(self.lab.probes[k][layer], layer)
    }
}

/// What was measured about a read-out, in words. How many of the network's answers
/// pushing the points across changed is said in a word (the count is shown beside it):
/// "none" exactly when no red bar is drawn.
fn verdict(p: Probe, layer: usize) -> (&'static str, Color) {
    let readable = p.accuracy >= READABLE;
    if layer + 1 == LAYERS {
        return if readable { ("the output alone tells it", pal::GREEN) } else { ("the output alone does not tell it", pal::DIM) };
    }
    if !readable {
        return if layer == 0 { ("no straight line through the raw input tells it", pal::DIM) } else { ("not readable here", pal::DIM) };
    }
    if !p.changed.is_finite() {
        return ("readable here", pal::YELLOW);
    }
    match p.changed {
        c if c <= 0.0 => ("readable here, yet pushing the points across changes none of the network's answers", pal::YELLOW),
        c if c < 0.1 => ("readable here, yet pushing the points across changes only a few of the network's answers", pal::YELLOW),
        c if c < MOST => ("readable here; pushing the points across changes some of the network's answers", pal::YELLOW),
        c if c < 0.95 => ("readable here, and pushing the points across changes most of the network's answers", pal::GREEN),
        c if c < 1.0 => ("readable here, and pushing the points across changes nearly all of the network's answers", pal::GREEN),
        _ => ("readable here, and pushing the points across changes all of the network's answers", pal::GREEN),
    }
}

/// One of the "things to try": the button, what to look for, the data set, the
/// read-out to show (property, layer) and whether the network trains (false: fresh
/// weights, paused).
type Experiment = (&'static str, &'static str, usize, (usize, usize), bool);

/// Each caption is checked by `what_the_experiments_say_is_so`.
const EXPERIMENTS: [Experiment; 6] = [
    (
        "Readable is not the same as used",
        "Left or right of centre can be read off layer 2 most of the time. Yet pushing every point across there changes none of the network's answers (no red bar): to tell a ring from its middle, left and right do not matter.",
        1,
        (1, 2),
        true,
    ),
    ("What the network does use", "The class can be read off layer 2, and pushing the points across there changes every one of the network's answers (a full red bar).", 1, (0, 2), true),
    ("The answer appears with depth", "At the input no straight line separates a ring from its middle, so the read-out fails. One layer on, it works: the layers exist to make the answer easy to read.", 1, (0, 0), true),
    (
        "A network that has learned nothing",
        "Fresh random weights, training paused: the network itself is guessing, yet the class can be read off its layer 2 almost perfectly. The probe did the work, not the network: pushing the points across there changes only a few of the network's answers. A probe's success says little by itself.",
        1,
        (0, 2),
        false,
    ),
    ("The control row", "Random labels: nothing to find, so about 50 %. A read-out scoring near this has found nothing, and there is nothing to push across (no red bars in this row).", 1, (4, 2), true),
    (
        "Used early, then no longer",
        "On the xor data the class depends on left or right. Pushing the points across at layer 1 changes all of the network's answers. Now click the same row at layer 3: none change. By then left or right has been folded into the answer, and the network no longer looks at it. And the class itself, at layer 1? A straight line reads it, yet pushing across there changes none of the answers: the network does not get its answer from that line.",
        3,
        (1, 1),
        true,
    ),
];

impl Scene for Probing {
    fn view(&self) -> View {
        if self.narrow == Some(true) {
            // on a phone: start on the table, the rest is a drag away
            View::Fit { center: Vec2::new(2.1, 0.2), half: Vec2::new(2.35, 2.4) }
        } else {
            View::Fit { center: Vec2::new(0.17, 0.0), half: Vec2::new(4.45, 2.7) }
        }
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn animated(&self) -> bool {
        self.running
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A common way to ask what a network knows: take one layer's activations and fit a simple straight-line read-out (a probe) that tries to tell some property of the input from them. The table shows how well that works for five properties at every layer of a small network, live, as it trains. Then the catch: a property can be easy to read off a layer and still play no part in what the network does. To find out, intervene: push every point's activations across the read-out's boundary, so that the read-out now says the opposite, and see whether the network's answers change.");
        ui.horizontal(|ui| {
            if ui.button(if self.running { "pause" } else { "train" }).clicked() {
                self.running = !self.running;
                self.stale = true;
            }
            if ui.button("new network").clicked() {
                self.seed += 1;
                self.restart();
            }
        });

        ui::section(ui, "Things to try");
        for (label, look_for, data, picked, running) in EXPERIMENTS {
            if ui::experiment(ui, label, look_for) {
                self.set_up(data, picked, running);
            }
        }
        ui::note(ui, "Each of these starts from the same network, so that it shows what it says. \"new network\" draws another: the numbers move, and now and then a caption fits less well.");

        ui::section(ui, "Data");
        let mut changed = ui::choice(ui, &DATASETS, &mut self.data_kind);
        changed |= ui::slider(ui, "noise", &mut self.noise, 0.0..=1.0);
        if changed {
            self.restart();
        }
        ui::section(ui, "Training");
        ui::slider_log(ui, "learning rate", &mut self.lr, 0.001..=0.1);
        ui::slider_int_log(ui, "steps per second", &mut self.rate, 10..=2000);
        ui::note(ui, r"The network has three hidden layers of eight $\tanh$ neurons. Read-outs are fitted to the training points and scored on points held out.");

        ui::section(ui, "The five properties");
        for (name, _, what) in PROPERTIES {
            ui::term(ui, &format!("{name}:"), what);
        }

        ui::section(ui, "Words");
        ui::term(ui, "Probe:", r"a simple read-out (here a straight line: a weighted sum $w \cdot h$ of the layer's activations $h$, compared with a threshold) trained to tell a property from a layer.");
        ui::term(ui, "Pushing across:", "moving each point's activations over the read-out's boundary to the matching place on the far side, so that the read-out now says the opposite of what it said, just as firmly; then letting the rest of the network run on from there. Each point moves along the line joining the two kinds' average activations, the way the two kinds actually differ. If none of the network's answers change, it was not using where the points sat along that line; if they change, it was. Only a read-out that works is pushed across: where a straight line cannot tell the two kinds apart, there is no \"across\".");
        ui::term(ui, "Right, and chance:", "a read-out is scored on the points that have the property and on those that do not, and the two hit rates are averaged. So chance is 50 % for every property, even one that nearly every point has: always saying \"yes\" scores 50 %.");

        ui::section(ui, "Colours");
        ui::key(ui, &[(CLASS[1], "orange dots: points that have the chosen property; red ground: where the read-out says yes"), (CLASS[0], "blue dots: points that do not; blue ground: where it says no"), (pal::RED, "bar in a cell: the share of the network's answers that change when the points are pushed across there (none: no bar)")]);

        ui::section(ui, "Numbers");
        ui::readout(ui, "training steps", self.steps.to_string());
        ui::readout(ui, "network right, held-out points", format!("{:.0} %", self.lab.accuracy * 100.0));
        let (k, layer) = self.picked;
        let p = self.lab.probes[k][layer];
        ui::readout(ui, "chosen read-out right", format!("{:.0} %", p.accuracy * 100.0));
        if p.changed.is_finite() {
            let n = self.lab.test.len();
            ui::readout(ui, "answers changed by pushing across", format!("{} of {n}", (p.changed * n as f32).round()));
        }
        ui::see_also(ui, &[("mlp", "Network playground"), ("fold", "How layers fold space"), ("sae", "Sparse autoencoders"), ("transformer", "A transformer, opened up")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let narrow = narrow_view(f);
        if self.narrow != Some(narrow) {
            self.new_view |= self.narrow.is_some() || narrow;
            self.narrow = Some(narrow);
        }
        // train within a time allowance
        let mut stepped = false;
        if self.running && f.dt > 0.0 {
            let mut turn = self.pace.begin(self.rate as f32, f.dt, f.real_dt, 0.004);
            while turn.next() {
                self.lab.step(self.lr);
                self.steps += 1;
                self.history.push(self.lab.loss);
            }
            stepped = self.pace.end(turn) > 0;
            self.since_fit += f.real_dt;
        }

        // which cell of the table the pointer is on; a click chooses it
        let hover = f.mouse().and_then(|m| {
            (0..PROPERTIES.len()).flat_map(|k| (0..LAYERS).map(move |l| (k, l))).find(|&(k, l)| {
                let (min, max) = Self::cell(k, l);
                m.cmpge(min).all() && m.cmple(max).all()
            })
        });
        self.hover = hover;
        if let Some(cell) = hover.filter(|_| f.clicked_canvas()) {
            // (the read-outs are as they were: only the picture changes)
            self.picked = cell;
            self.repaint = true;
        }
        if self.stale || (stepped && self.since_fit >= REFIT) {
            self.lab.probe();
            (self.stale, self.repaint, self.since_fit) = (false, true, 0.0);
        }
        if self.repaint {
            self.paint();
            self.repaint = false;
        }

        // the big picture: what the chosen read-out says at every place
        let (k, layer) = self.picked;
        let (centre, side) = FIELD;
        let half = side * 0.5;
        self.field.style.model = Mat4::from_translation(centre.extend(-0.2)) * Mat4::from_scale(Vec3::new(side, side, 1.0));
        f.surface(&mut self.field);
        frame_rect(f, centre - Vec2::splat(half), centre + Vec2::splat(half), 1.4, pal::DIM);
        f.dl.discs.style.outline_px = 1.1;
        for i in 0..self.lab.train.len() {
            let p = self.lab.train.point(i);
            if p.abs().max_element() < DATA_R {
                f.dot(centre + p / DATA_R * half, 6.0, CLASS[self.lab.truth(k, 0, i) as usize]);
            }
        }
        let (name, shown, _) = PROPERTIES[k];
        f.text(centre + Vec2::new(-half, half + 0.24), format!("what a straight-line read-out of {} says:  \"{name}\"?", LAYER_NAMES[layer]), 14.0, pal::TEXT, Align2::LEFT_BOTTOM);
        f.text(centre + Vec2::new(-half, half + 0.06), format!("dots: the truth ({shown})"), 11.5, pal::DIM, Align2::LEFT_BOTTOM);
        ramp_key(f, Vec2::new(centre.x, centre.y - half - 0.165), 0.7, 0.07, Cmap::CoolWarm, ("read-out says no", CLASS[0]), ("read-out says yes", CLASS[1]), "white line: its boundary");

        // the table: one row per property, one column per layer
        let (origin, cell) = TABLE;
        // (on a phone the words keep their size while the table shrinks: shorter ones there)
        let narrow = self.narrow == Some(true);
        let heading = if narrow { "how often each layer's read-out is right (chance: 50 %)" } else { "how often a read-out of each layer is right, on held-out points (chance: 50 %)" };
        f.text(origin + Vec2::new(-1.6, 0.52), heading, 12.5, pal::TEXT, Align2::LEFT_BOTTOM);
        f.text(
            origin + Vec2::new(-1.6, 0.33),
            if narrow { "on held-out points, the two kinds counting equally   ·   click a cell" } else { "the two kinds count equally   ·   click a cell to see it on the left" },
            11.0,
            pal::DIM,
            Align2::LEFT_BOTTOM,
        );
        for (l, name) in LAYER_NAMES.iter().enumerate() {
            f.text(origin + Vec2::new((l as f32 + 0.475) * cell.x, 0.06), *name, 11.5, if l == layer { pal::WHITE } else { pal::DIM }, Align2::CENTER_BOTTOM);
        }
        let shade = Cmap::Viridis.table();
        for (k2, (name, ..)) in PROPERTIES.iter().enumerate() {
            let (row_min, row_max) = Self::cell(k2, 0);
            let label = if self.lab.same_as_class[k2] { format!("{name} ({})", if narrow { "= class" } else { "= class here" }) } else { name.to_string() };
            f.text(Vec2::new(row_min.x - 0.1, (row_min.y + row_max.y) * 0.5), label, 12.0, if k2 == k { pal::WHITE } else { pal::TEXT }, Align2::RIGHT_CENTER);
            for l in 0..LAYERS {
                let (min, max) = Self::cell(k2, l);
                let p = self.lab.probes[k2][l];
                let level = ((p.accuracy - 0.5) / 0.5).clamp(0.0, 1.0);
                f.dl.tris_alpha.rect(min, max, 0.0, shade[table_index(0.08 + 0.84 * level)]);
                let ink = if level > 0.7 { Color::hex(0x0A0D16) } else { pal::WHITE };
                f.text((min + max) * 0.5 + Vec2::new(0.0, 0.05), format!("{:.0}", p.accuracy * 100.0), 13.0, ink, Align2::CENTER_CENTER);
                // how many of the network's answers change when the points are pushed
                // across here (any at all: a bar; none: no bar, as the verdict says)
                if p.changed > 0.0 {
                    let (a, b) = (Vec2::new(min.x + 0.03, min.y + 0.03), Vec2::new(min.x + 0.03 + (max.x - min.x - 0.06) * p.changed.min(1.0), min.y + 0.09));
                    f.dl.tris_alpha.rect(a, b, 0.01, pal::RED);
                }
                let chosen = (k2, l) == self.picked;
                if chosen || self.hover == Some((k2, l)) {
                    let c = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)];
                    f.dl.lines.closed(&c.map(|q| q.extend(0.02)), if chosen { 2.4 } else { 1.4 }, pal::WHITE);
                }
            }
        }
        // under the table: what the red bars are, then the chosen read-out in words
        // (lines of text, one under the other, however many each wraps to)
        let (below, _) = Self::cell(PROPERTIES.len() - 1, 0);
        let (at, width) = (Vec2::new(origin.x - 1.6, below.y - 0.07), 4.6);
        let mut down = 0.0;
        let mut say = |f: &mut Frame, text: &str, size: f32, color: Color, gap: f32| {
            let anchor = at - Vec2::new(0.0, down / pixels_per_unit(f));
            down += wrapped(f, anchor, width, text, size, color) as f32 * (size + 3.0) + gap;
        };
        say(f, "red bar: the share of the network's answers that change when the points are pushed across there", 11.0, pal::DIM, 9.0);
        let p = self.lab.probes[k][layer];
        let (verdict, colour) = self.verdict(k, layer);
        say(f, &format!("\"{name}\" read off {}: right {:.0} %", LAYER_NAMES[layer], p.accuracy * 100.0), 12.5, pal::TEXT, 2.0);
        let n = self.lab.test.len();
        let pushed = if layer + 1 == LAYERS {
            "nothing is pushed at the output: there is no network left to run".to_string()
        } else if p.changed.is_finite() {
            let count = (p.changed * n as f32).round();
            if count > 0.0 {
                format!("pushing the points across there changes {count} of the network's {n} answers")
            } else {
                format!("pushing the points across there changes none of the network's {n} answers")
            }
        } else if p.accuracy < READABLE {
            "nothing is pushed: this read-out cannot tell the two kinds apart".to_string()
        } else {
            "nothing is pushed: the read-out puts the two kinds' averages too close together".to_string()
        };
        say(f, &pushed, 12.5, pal::TEXT, 4.0);
        say(f, &format!("→ {verdict}"), 12.5, colour, 0.0);

        // the network's loss as it trains: at the foot of the picture, or under the
        // words if (on a small canvas) they wrap that far down; left out if that
        // would put it below the bottom of the view
        let per_unit = pixels_per_unit(f);
        let size = Vec2::new(4.3, 0.3);
        let words_end = at.y - down / per_unit;
        let o = Vec2::new(origin.x - 1.6, (-2.55f32).min(words_end - 24.0 / per_unit - size.y - 0.06));
        if o.y > f.cam.flat_bounds().0.y.min(-2.75) {
            chart(f, o, size, &[(self.history.points(), pal::CYAN)], true);
            f.text(o + Vec2::new(0.0, size.y + 0.06), format!("the network's loss, log scale   ·   it is right on {:.0} % of held-out points", self.lab.accuracy * 100.0), 11.5, pal::DIM, Align2::LEFT_BOTTOM);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trained(kind: usize, seed: u64, steps: usize) -> Lab {
        let mut lab = Lab::new(kind, 0.15, seed);
        for _ in 0..steps {
            lab.step(RATE);
        }
        lab.probe();
        lab
    }

    #[test]
    fn a_probe_reads_a_linear_property() {
        let lab = Lab::new(1, 0.15, SEED);
        // "right half" is a straight line at the input, and its two groups differ
        // along x₁ only
        let p = lab.probes[1][0];
        assert!(p.accuracy > 0.97, "{}", p.accuracy);
        assert!(p.shift[0] > 0.5 && p.shift[1].abs() < 0.15, "{:?}", &p.shift[..2]);
    }

    /// The network's batch pass and the hand-written pass that carries on from a
    /// layer are the same network.
    #[test]
    fn carrying_on_from_a_layer_gives_the_network_s_own_output() {
        let mut lab = trained(0, SEED, 300);
        let all = activations(&mut lab.net, &lab.test);
        for a in all.iter().take(40) {
            for from in 0..LAYERS - 1 {
                let mut acts = *a;
                // (everything after `from` is worked out again)
                acts[from + 1..].iter_mut().for_each(|layer| layer.fill(f32::NAN));
                let output = run(&lab.net, from, &mut acts);
                assert!((output - a[LAYERS - 1][0]).abs() < 1e-4, "from layer {from}: {output} against {}", a[LAYERS - 1][0]);
            }
        }
    }

    /// "Chance: 50 %", whatever the property. On the blobs nearly every point is in a
    /// same-sign corner: a read-out that always says yes (or no) scores 50 there, and
    /// the raw input is no longer said to tell it (it used to score 94 at every layer
    /// by saying yes to everything).
    #[test]
    fn chance_is_fifty_percent_whatever_the_property() {
        let mut lab = trained(4, SEED, 1500);
        let test = activations(&mut lab.net, &lab.test);
        let corner = &lab.truths[1][3];
        let share = corner.iter().filter(|t| **t).count() as f32 / corner.len() as f32;
        assert!(share > 0.9, "a same-sign corner on the blobs: {share}");
        for layer in 0..LAYERS {
            for b in [1.0, -1.0] {
                assert_eq!(Probe { b, ..Probe::default() }.score(&test, layer, corner), 0.5);
            }
        }
        let input = lab.probes[3][0];
        assert!(input.accuracy < 0.6, "{}", input.accuracy);
        assert_eq!(verdict(input, 0).0, "no straight line through the raw input tells it");
        // and on the moons, where a third of the points are in one, the same
        let moons = trained(2, SEED, 1500);
        let share = moons.truths[1][3].iter().filter(|t| **t).count() as f32 / N_TEST as f32;
        assert!((0.25..0.45).contains(&share) && moons.probes[3][0].accuracy < 0.6, "{share} {}", moons.probes[3][0].accuracy);
    }

    /// Pushing across: every point ends on the other side of the read-out, as far
    /// from its boundary as it was, having moved only along the line joining the two
    /// kinds' averages. Also where a kind is made of two opposite clusters (the class
    /// on the xor data), whose averages nearly coincide: there the old push (by the
    /// difference of the averages) moved nothing at all.
    #[test]
    fn pushing_across_puts_every_point_on_the_other_side() {
        for (kind, k, layer) in [(1, 1, 1), (3, 0, 1), (1, 0, 2), (3, 1, 3), (1, 1, 0)] {
            let mut lab = trained(kind, SEED, 1500);
            let (p, d) = (lab.probes[k][layer], dim(layer));
            assert!(p.changed.is_finite(), "{} {k} {layer}", DATASETS[kind]);
            let length = p.shift[..d].iter().map(|s| s * s).sum::<f32>().sqrt();
            let mut moved = 0.0;
            let test = activations(&mut lab.net, &lab.test);
            for a in &test {
                let mut h = a[layer];
                let before = p.read(&h, d);
                p.push(&mut h, d);
                let after = p.read(&h, d);
                assert!((after + before).abs() < 1e-3 * (1.0 + before.abs()), "{before} became {after}");
                let step: Vec<f32> = (0..d).map(|j| h[j] - a[layer][j]).collect();
                let along = step.iter().zip(&p.shift).map(|(s, t)| s * t).sum::<f32>() / length;
                let size = step.iter().map(|s| s * s).sum::<f32>().sqrt();
                assert!((size - along.abs()).abs() < 1e-3 * (1.0 + size), "moved {size}, {along} of it along the line");
                moved += size / test.len() as f32;
            }
            // (each point moves its own distance, which is not the distance between the
            // averages)
            assert!(moved > 0.1, "{} {k} {layer}: moved {moved} on average, the averages {length} apart", DATASETS[kind]);
        }
        let xor = trained(3, SEED, 1500).probes[0][1];
        assert!(xor.shift[..WIDTH].iter().map(|s| s * s).sum::<f32>().sqrt() < 0.5);
    }

    /// The verdict says what was measured and no more; its words follow the numbers,
    /// and the red bar (drawn when `changed > 0`) is there exactly when it does not
    /// say "none".
    #[test]
    fn the_verdict_says_only_what_was_measured() {
        let lab = trained(1, SEED, 1500);
        for row in &lab.probes {
            assert!(row[LAYERS - 1].changed.is_nan());
            assert!(row[..LAYERS - 1].iter().all(|p| p.changed.is_finite() == (p.accuracy >= READABLE)));
        }
        let says = |accuracy: f32, changed: f32, layer: usize| verdict(Probe { accuracy, changed, ..Probe::default() }, layer).0;
        assert_eq!(says(0.9, 0.0, 2), "readable here, yet pushing the points across changes none of the network's answers");
        assert_eq!(says(0.9, 0.005, 2), "readable here, yet pushing the points across changes only a few of the network's answers");
        assert_eq!(says(0.9, 0.3, 2), "readable here; pushing the points across changes some of the network's answers");
        assert_eq!(says(0.9, 0.5, 2), "readable here, and pushing the points across changes most of the network's answers");
        assert_eq!(says(0.9, 0.97, 1), "readable here, and pushing the points across changes nearly all of the network's answers");
        assert_eq!(says(0.9, 1.0, 0), "readable here, and pushing the points across changes all of the network's answers");
        assert_eq!(says(0.6, f32::NAN, 2), "not readable here");
        assert_eq!(says(0.55, f32::NAN, 0), "no straight line through the raw input tells it");
        assert_eq!(says(1.0, f32::NAN, LAYERS - 1), "the output alone tells it");
        assert_eq!(says(0.5, f32::NAN, LAYERS - 1), "the output alone does not tell it");
        for changed in [0.0, 0.005, 0.04, 0.3, 0.6, 1.0] {
            assert_eq!(says(0.9, changed, 2).contains("none"), changed <= 0.0, "{changed}");
            assert_eq!(says(0.9, changed, 2).contains("most") || says(0.9, changed, 2).contains("all"), changed >= MOST, "{changed}");
        }
    }

    /// A network trained at another learning rate is not the one the "things to try"
    /// were measured on: they start afresh rather than show it.
    #[test]
    fn a_network_trained_at_another_rate_is_noticed() {
        let mut lab = Lab::new(1, NOISE, SEED);
        lab.step(RATE);
        assert!(!lab.off_rate);
        lab.step(0.05);
        lab.step(RATE);
        assert!(lab.off_rate);
    }

    /// The "things to try", held to account on twenty networks: what each caption
    /// says must hold on (nearly) all of them, and to the letter on the one the
    /// experiments use.
    #[test]
    fn what_the_experiments_say_is_so() {
        let (mut networks, mut side_readable, mut side_none) = (0, 0, 0);
        let (mut xor_all, mut xor_late_none, mut class_read, mut class_none) = (0, 0, 0, 0);
        for seed in 5..25u64 {
            networks += 1;
            let ring = trained(1, seed, 1500);
            assert!(ring.accuracy > 0.95, "seed {seed}: ring network {}", ring.accuracy);
            // left or right: read off layer 2 most of the time, and pushing the
            // points across there changes none of the network's answers
            let side = ring.probes[1][2];
            assert!(side.accuracy > 0.75, "seed {seed}: right half at layer 2: {}", side.accuracy);
            if side.accuracy >= READABLE {
                side_readable += 1;
                side_none += (side.changed == 0.0) as usize;
                assert!(side.changed < 0.03, "seed {seed}: right half pushed at layer 2: {}", side.changed);
            }
            // (one layer earlier, where pushing used to slide the whole ring sideways
            // and so change the class: some answers change, not most)
            assert!(ring.probes[1][1].changed < 0.25, "seed {seed}: right half pushed at layer 1: {}", ring.probes[1][1].changed);
            // the class: not readable at the input, readable one layer on, and
            // pushing across at layer 2 changes every answer
            assert!(ring.probes[0][0].accuracy < 0.65, "seed {seed}: class at the input: {}", ring.probes[0][0].accuracy);
            assert!(ring.probes[0][1].accuracy > 0.9 && ring.probes[0][2].accuracy > 0.95, "seed {seed}");
            assert_eq!(ring.probes[0][2].changed, 1.0, "seed {seed}: class pushed at layer 2");
            // the control row finds nothing anywhere, and so is pushed nowhere
            for layer in 0..LAYERS {
                let p = ring.probes[4][layer];
                assert!((0.38..0.62).contains(&p.accuracy) && p.changed.is_nan(), "seed {seed}: random labels at layer {layer}: {}", p.accuracy);
            }
            // an untrained network is guessing, yet its layer 2 lets the class be
            // read off; pushing across there changes only a few of its answers
            let fresh = Lab::new(1, 0.15, seed);
            let class = fresh.probes[0][2];
            assert!(fresh.accuracy < 0.75 && class.accuracy > 0.85 && class.changed < 0.15, "seed {seed}: untrained: network {} read-out {} changed {}", fresh.accuracy, class.accuracy, class.changed);

            // xor: pushing left or right across at layer 1 changes (nearly) all the
            // answers, and at layer 3 none; the class at layer 1 is read, yet pushing
            // it across changes none
            let xor = trained(3, seed, 1500);
            assert!(xor.same_as_class[3] && xor.accuracy > 0.95, "seed {seed}: xor network {}", xor.accuracy);
            xor_all += (xor.probes[1][1].changed >= 0.95) as usize;
            let late = xor.probes[1][3].changed;
            assert!(late < 0.02, "seed {seed}: xor, right half pushed at layer 3: {late}");
            xor_late_none += (late == 0.0) as usize;
            let class = xor.probes[0][1];
            if class.accuracy >= READABLE {
                class_read += 1;
                // (none, or a stray answer or two)
                class_none += (class.changed < 0.02) as usize;
            }

            if seed == SEED {
                // the network the experiments show: each lands on the verdict its caption describes
                let says = |lab: &Lab, (k, layer): (usize, usize)| verdict(lab.probes[k][layer], layer).0;
                assert_eq!(says(&ring, EXPERIMENTS[0].3), "readable here, yet pushing the points across changes none of the network's answers");
                // ("no red bar", by the bar's own rule)
                assert!(side.changed <= 0.0);
                assert_eq!(says(&ring, EXPERIMENTS[1].3), "readable here, and pushing the points across changes all of the network's answers");
                assert_eq!(says(&ring, EXPERIMENTS[2].3), "no straight line through the raw input tells it");
                assert_eq!(says(&fresh, EXPERIMENTS[3].3), "readable here, yet pushing the points across changes only a few of the network's answers");
                assert!(fresh.probes[0][2].accuracy > 0.97);
                assert_eq!(says(&ring, EXPERIMENTS[4].3), "not readable here");
                assert_eq!(says(&xor, EXPERIMENTS[5].3), "readable here, and pushing the points across changes all of the network's answers");
                assert_eq!(says(&xor, (1, 3)), "readable here, yet pushing the points across changes none of the network's answers");
                assert_eq!(says(&xor, (0, 1)), "readable here, yet pushing the points across changes none of the network's answers");
            }
        }
        // "most of the time", "now and then a caption fits less well": all but a
        // network or two
        assert!(side_readable >= networks - 2 && side_none >= side_readable - 2, "left or right readable at layer 2 on {side_readable} of {networks} networks, untouched by the push on {side_none}");
        assert!(xor_all >= networks - 2 && xor_late_none >= networks - 2, "xor answers all changed at layer 1 on {xor_all} of {networks} networks, none at layer 3 on {xor_late_none}");
        assert!(class_read >= networks - 4 && class_none >= class_read - 1, "xor class read at layer 1 on {class_read} networks, untouched by the push on {class_none}");
        // the experiments are set up as the table above assumes
        assert_eq!(EXPERIMENTS.map(|e| (e.2, e.4)), [(1, true), (1, true), (1, true), (1, false), (1, true), (3, true)]);
    }

    fn cell(p: &Probe) -> String {
        if p.changed.is_finite() {
            format!("{:>3.0} ch{:>3.0}", p.accuracy * 100.0, p.changed * 100.0)
        } else {
            format!("{:>3.0}      ", p.accuracy * 100.0)
        }
    }

    /// The whole table for a few settings (run with --ignored --nocapture): how often
    /// each read-out is right, and the share of the network's answers that pushing
    /// across it changes ("ch").
    #[test]
    #[ignore]
    fn explore() {
        for (kind, steps) in [(1usize, 0usize), (1, 1500), (3, 1500), (0, 3000), (2, 1500), (4, 1500)] {
            for seed in [5, 6, 7] {
                let lab = trained(kind, seed, steps);
                println!("{} after {steps} steps, seed {seed}: network right {:.0} %", DATASETS[kind], lab.accuracy * 100.0);
                for (k, (name, ..)) in PROPERTIES.iter().enumerate() {
                    let cells: Vec<String> = (0..LAYERS).map(|l| cell(&lab.probes[k][l])).collect();
                    println!("    {name:<18} {}", cells.join(" | "));
                }
            }
        }
    }

    /// What each caption's numbers are, seed by seed (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn captions_seed_by_seed() {
        println!("seed | ring net | side@1 @2 read ch | class@0 @1 @2 read, @2 ch | control min..max | fresh net, class@2 read ch | xor net, side@1 ch, @3 ch, class@1 read ch");
        for seed in 5..25u64 {
            let ring = trained(1, seed, 1500);
            let fresh = Lab::new(1, 0.15, seed);
            let xor = trained(3, seed, 1500);
            let control: Vec<f32> = (0..LAYERS).map(|l| ring.probes[4][l].accuracy).collect();
            let (side1, side2, class) = (ring.probes[1][1], ring.probes[1][2], ring.probes[0]);
            println!(
                "{seed:>4} | {:.2} | {:.2} {:.2} {:.2} {:.2} | {:.2} {:.2} {:.2}, {:.2} | {:.2}..{:.2} | {:.2}, {:.2} {:.2} | {:.2}, {:.2}, {:.2}, {:.2} {:.2}",
                ring.accuracy,
                side1.accuracy,
                side1.changed,
                side2.accuracy,
                side2.changed,
                class[0].accuracy,
                class[1].accuracy,
                class[2].accuracy,
                class[2].changed,
                control.iter().copied().fold(1.0, f32::min),
                control.iter().copied().fold(0.0, f32::max),
                fresh.accuracy,
                fresh.probes[0][2].accuracy,
                fresh.probes[0][2].changed,
                xor.accuracy,
                xor.probes[1][1].changed,
                xor.probes[1][3].changed,
                xor.probes[0][1].accuracy,
                xor.probes[0][1].changed
            );
        }
    }
}

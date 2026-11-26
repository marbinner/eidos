//! Sparse autoencoders: finding the features a network has packed into too few
//! dimensions, given nothing but its activations.

use super::autoencoder::*;
use super::*;
use crate::clock::Pace;
use crate::math::nn::dot;

pub const DEF: SceneDef = SceneDef {
    id: "sae",
    title: "Sparse autoencoders",
    group: GROUP,
    blurb: "Pulling a network's features back out of superposition.",
    make: |_| Box::new(SaeScene::new()),
};

/// The hidden dimensions on offer, and for each one the number of features, their
/// sparsity and the penalty at which the autoencoder recovers them all (see the tests).
const DIMS: [usize; 4] = [2, 3, 8, 16];
const DEFAULTS: [(usize, f32, f32); 4] = [(5, 0.9, 0.3), (8, 0.9, 0.3), (16, 0.9, 0.15), (32, 0.95, 0.1)];
const MAX_FEATURES: [usize; 4] = [12, 16, 24, 32];
const MAX_LATENTS: usize = 48;
/// Samples the statistics are measured on, and how many of them are drawn as dots.
const EVAL: usize = 800;
const CLOUD: usize = 420;
const SEED: u64 = 11;
/// The learning rate and batch size every experiment starts from (what its caption
/// was measured at).
const LR: f32 = 0.01;
const BATCH: usize = 256;
/// How often (in seconds) the measurements on show are brought up to date while
/// training: measuring a large dictionary takes a good part of a millisecond.
const MEASURE: f32 = 0.03;
/// How long (in seconds of training) each sample is taken apart before the next,
/// where samples take turns.
const TURN: f32 = 2.5;

/// Everything an experiment sets.
#[derive(Clone, Copy)]
struct Setup {
    /// Index into [`DIMS`].
    dims: usize,
    n: usize,
    k: usize,
    sparsity: f32,
    lambda: f32,
    pca: bool,
}

const fn setup(dims: usize, n: usize, k: usize, sparsity: f32, lambda: f32, pca: bool) -> Setup {
    Setup { dims, n, k, sparsity, lambda, pca }
}

/// "Things to try": (button, what to look for, what it sets). Every claim made here
/// is checked by a test of the same setting.
const TRY: [(&str, &str, Setup); 8] = [
    (
        "five features in a plane",
        "Each bold arrow swings onto one of the faint spokes, and the matrix becomes one bright cell per row: all five found, though the autoencoder was never told where they are.",
        setup(0, 5, 5, 0.9, 0.3, false),
    ),
    (
        r"no penalty ($\lambda = 0$)",
        "The rebuilding becomes exact, yet two or three latents now fire for almost every activation, even the empty ones, and the arrows stop wherever they like: with no price on firing, nothing ties a latent to a feature.",
        setup(0, 5, 5, 0.9, 0.0, false),
    ),
    (
        r"too much penalty ($\lambda = 1.2$)",
        r"Most arrows still find a feature, but the latents report a small part of the true size (the scatter sags far below the dashed line) and most of the variance is left unexplained. Towards $\lambda = 2$ they stop firing altogether.",
        setup(0, 5, 5, 0.9, 1.2, false),
    ),
    (
        "too few latents (3 for 5 features)",
        "Three directions cannot cover five features. Each latent ends up on a feature or between two and has to answer for its neighbours as well, so none counts as found, and about a quarter of the variance stays unexplained.",
        setup(0, 5, 3, 0.9, 0.3, false),
    ),
    (
        "too many latents (10 for 5 features)",
        "All five are still found. Each spare latent either goes dead or becomes a second copy of a feature, sharing its activation with the first: two rows lit in one column.",
        setup(0, 5, 10, 0.9, 0.3, false),
    ),
    (
        "32 features in 16 dimensions",
        "Too many dimensions to draw, so read the matrix. Within a few thousand steps it is a clean diagonal: every one of the 32 found.",
        setup(3, 32, 32, 0.95, 0.1, false),
    ),
    (
        "the same, features often together",
        "Sparsity 0.5: sixteen features present at a time. The diagonal never forms. The method rests on features rarely being present together.",
        setup(3, 32, 32, 0.5, 0.1, false),
    ),
    (
        "principal components instead",
        "The white lines are the cloud's principal components. There are only two, at right angles, and this cloud is about as wide in every direction, so they point wherever chance puts them. Five features cannot be read off them.",
        setup(0, 5, 5, 0.9, 0.3, true),
    ),
];

/// What a latent is called: A, B, C, ... then a, b, c, ...
fn letter(j: usize) -> String {
    match j {
        0..=25 => ((b'A' + j as u8) as char).to_string(),
        26..=51 => ((b'a' + (j - 26) as u8) as char).to_string(),
        _ => (j + 1).to_string(),
    }
}

/// A direction of the hidden space as a point of the picture (2 or 3 dimensions).
fn v3(d: &[f32]) -> Vec3 {
    Vec3::new(d[0], d[1], if d.len() > 2 { d[2] } else { 0.0 })
}

const DOT: Color = Color::hex(0xC9D3E8);
const SPOKE: Color = Color::hex(0x8A96AE);

struct SaeScene {
    dims: usize,
    n: usize,
    k: usize,
    sparsity: f32,
    noise: f32,
    lambda: f32,
    lr: f32,
    batch: usize,
    /// Steps per second asked for.
    rate: usize,
    pca: bool,
    show_encoder: bool,
    features: Features,
    sae: Sae,
    rng: Rng,
    running: bool,
    steps: u64,
    pace: Pace,
    /// The training batch (feature values, activations).
    bx: Vec<f32>,
    bh: Vec<f32>,
    /// The fixed set everything shown is measured on, and what is known about it.
    ex: Vec<f32>,
    eh: Vec<f32>,
    variance: f32,
    mean: Vec<f32>,
    pcs: Vec<f32>,
    pc_spread: Vec<f32>,
    /// Which of those samples are drawn as dots: `cloud`, the ones that hold some
    /// feature, and `quiet`, the ones that hold none (only noise), drawn fainter. The
    /// two together are the first few hundred samples, so the dots show the share of
    /// empty activations the autoencoder is given.
    cloud: Vec<usize>,
    quiet: Vec<usize>,
    report: Report,
    /// The step count the report is for, and the seconds since it was measured.
    shown: u64,
    since: f32,
    stale: bool,
    /// Running averages over recent batches: error as a share of the variance, and
    /// latents firing per activation; and their histories.
    error: f32,
    firing: f32,
    history: [History; 2],
    /// The sample that is taken apart (an index into the evaluation set), and how
    /// often "next" has been asked for where samples cycle by themselves.
    selected: usize,
    skipped: usize,
    /// Where samples cycle by themselves: the seconds of training they have been
    /// cycling for (so that pausing keeps the one on show).
    turning: f32,
    /// The sample taken apart in the picture, and the order its latents were laid end
    /// to end in the last frame (see `laid`).
    laid: (usize, Vec<usize>),
    /// Where each latent's arrow starts in the sample taken apart, as drawn and on the
    /// way there (None for a latent that does not fire): it follows where the order
    /// puts it (`follow_starts`).
    starts: Vec<Option<(Vec3, Vec3)>>,
    /// The weights as they were before the newest training step, and as the picture
    /// draws them (see `Drawn`); and the training steps owed, counted as `pace`
    /// counts them, whose share of a step is how far the picture is into the newest.
    before: Drawn,
    drawn: Drawn,
    owed: f32,
    new_view: bool,
    /// The shape of the view, as the camera's starting position depends on it:
    /// panels under the picture (a phone), or a view too short for the full legend.
    stacked: bool,
    short: bool,
}

/// Is the view narrow and upright (a phone)? The panels then go under the picture
/// rather than beside it. (Judged by shape as well as width: a narrow view that is
/// wider than it is tall, a phone on its side, has no height to stack things in.)
fn stacked(view: Vec2) -> bool {
    view.x < 720.0 && view.y > view.x
}

/// A view with less height than this (below the application's strip) gets the legend
/// in one line, and the picture a little higher up.
const SHORT: f32 = 560.0;

/// The autoencoder's weights as the picture draws them. It trains 240 steps a second,
/// a whole number of them in a frame: four in every frame at 60 frames a second, but
/// three or four at 75, one or two at 144, and drawn as the steps left them the
/// dictionary moved by one step's worth in one frame and two in the next. So the
/// picture draws the weights between those before the newest step and those after it,
/// by the share of a step the scene's time has gone since that step was due: one step
/// behind, and moving in every frame by about the same number of steps. Every step's
/// own shake is still there. (`SaeScene::sae` is the autoencoder itself: the numbers,
/// the matrix, the scatter and the charts are all its own.)
#[derive(Default)]
struct Drawn {
    m: usize,
    k: usize,
    enc: Vec<f32>,
    be: Vec<f32>,
    dec: Vec<f32>,
    bd: Vec<f32>,
}

impl Drawn {
    /// The weights of `sae` as they are.
    fn keep(&mut self, sae: &Sae) {
        (self.m, self.k) = (sae.m, sae.k);
        self.enc.clone_from(&sae.enc);
        self.be.clone_from(&sae.be);
        self.dec.clone_from(&sae.dec);
        self.bd.clone_from(&sae.bd);
    }

    /// A share `t` of the way from the weights `before` to those of `sae` (all the way,
    /// if `before` is not of the same shape).
    fn between(&mut self, before: &Drawn, sae: &Sae, t: f32) {
        self.keep(sae);
        if (before.m, before.k) != (sae.m, sae.k) || t >= 1.0 {
            return;
        }
        for (now, was) in [(&mut self.enc, &before.enc), (&mut self.be, &before.be), (&mut self.dec, &before.dec), (&mut self.bd, &before.bd)] {
            for (v, w) in now.iter_mut().zip(was) {
                *v = w + (*v - w) * t;
            }
        }
    }

    fn decoder(&self, j: usize) -> &[f32] {
        &self.dec[j * self.m..(j + 1) * self.m]
    }

    fn encoder(&self, j: usize) -> &[f32] {
        &self.enc[j * self.m..(j + 1) * self.m]
    }

    /// How strongly each latent fires for activation `h` with these weights: the same
    /// sum as `Sae::run` (the tests hold the two together).
    fn fired(&self, h: &[f32]) -> Vec<f32> {
        let c: Vec<f32> = h.iter().zip(&self.bd).map(|(h, b)| h - b).collect();
        (0..self.k).map(|j| (dot(self.encoder(j), &c) + self.be[j]).max(0.0)).collect()
    }
}

/// How closely the arrows of the sample taken apart follow their places in the chain,
/// in seconds of the scene's time (see `SaeScene::follow_starts`).
const SLIDE: f32 = 0.05;

/// How much more strongly a latent must fire than the one laid before it in the
/// sample taken apart, as a share, for the two to change places.
const OVERTAKE: f32 = 0.25;

/// The order in which the latents that fire for a sample are laid end to end: the
/// strongest first, but each keeps its place from the frame before (`kept`, for the
/// same sample) until the one after it fires more strongly by `OVERTAKE`. Training
/// shakes the latents, and two that fire about equally changed places back and forth
/// from one frame to the next: the joint between them jumped between two places (the
/// twelve jumps in six seconds as the scene opens). The sum they reach is the same in
/// any order, and so are the numbers given for them.
fn laid(kept: &[usize], fired: &[f32]) -> Vec<usize> {
    let fires = |j: usize| fired.get(j).is_some_and(|v| *v > 0.0);
    let mut order: Vec<usize> = kept.iter().copied().filter(|&j| fires(j)).collect();
    // (those that have begun to fire join in order of strength, after the rest)
    let mut new: Vec<usize> = (0..fired.len()).filter(|&j| fires(j) && !order.contains(&j)).collect();
    new.sort_by(|&a, &b| fired[b].total_cmp(&fired[a]));
    order.extend(new);
    // neighbours change places only when the later is clearly the stronger (which no
    // swap can undo, so this ends)
    let mut swapped = true;
    while swapped {
        swapped = false;
        for i in 1..order.len() {
            if fired[order[i]] > fired[order[i - 1]] * (1.0 + OVERTAKE) {
                order.swap(i - 1, i);
                swapped = true;
            }
        }
    }
    order
}

/// What the autoencoder makes of one activation: its latents and its reconstruction.
struct Reading {
    fired: Vec<f32>,
    out: Vec<f32>,
}

/// Where everything that is drawn in screen space goes (logical pixels).
struct Layout {
    stacked: bool,
    /// Too little room for the legend's full text: one line instead.
    brief: bool,
    /// Top-left of the matrix's cells, the size of a cell, the width of the bars.
    matrix: Vec2,
    cell: f32,
    bars: f32,
    /// Top-left and side of the scatter's plotting square.
    scatter: Vec2,
    side: f32,
    charts: [Vec2; 2],
    chart: Vec2,
    /// The "one activation taken apart" panel (only when there is no picture).
    apart: Vec2,
    apart_size: Vec2,
    key: Vec2,
}

impl Layout {
    /// The same layout `by` pixels further down.
    fn lowered(mut self, by: f32) -> Layout {
        for at in [&mut self.matrix, &mut self.scatter, &mut self.apart, &mut self.key].into_iter().chain(&mut self.charts) {
            at.y += by;
        }
        self
    }
}

impl SaeScene {
    fn new() -> Self {
        let first = TRY[0].2;
        let mut rng = Rng::new(SEED);
        let features = Features::new(first.n, DIMS[first.dims], &mut rng);
        let sae = Sae::new(DIMS[first.dims], first.k, &mut rng);
        let mut scene = SaeScene {
            dims: first.dims,
            n: first.n,
            k: first.k,
            sparsity: first.sparsity,
            noise: 0.0,
            lambda: first.lambda,
            lr: LR,
            batch: BATCH,
            rate: 240,
            pca: first.pca,
            show_encoder: false,
            features,
            sae,
            rng,
            running: true,
            steps: 0,
            pace: Pace::default(),
            bx: Vec::new(),
            bh: Vec::new(),
            ex: Vec::new(),
            eh: Vec::new(),
            variance: 1.0,
            mean: Vec::new(),
            pcs: Vec::new(),
            pc_spread: Vec::new(),
            cloud: Vec::new(),
            quiet: Vec::new(),
            report: Report::default(),
            shown: 0,
            since: 0.0,
            stale: true,
            error: 1.0,
            firing: 0.0,
            history: [History::new(600), History::new(600)],
            selected: 0,
            skipped: 0,
            turning: 0.0,
            laid: (usize::MAX, Vec::new()),
            starts: Vec::new(),
            before: Drawn::default(),
            drawn: Drawn::default(),
            owed: 0.0,
            new_view: false,
            stacked: false,
            short: false,
        };
        scene.apply(first);
        scene
    }

    fn m(&self) -> usize {
        DIMS[self.dims]
    }

    /// Set an experiment up from scratch (always the same way: experiments repeat).
    fn apply(&mut self, s: Setup) {
        self.new_view |= s.dims != self.dims;
        (self.dims, self.n, self.k, self.sparsity, self.lambda, self.pca) = (s.dims, s.n, s.k, s.sparsity, s.lambda, s.pca);
        // (and the settings the caption does not name, at which it was measured)
        (self.noise, self.lr, self.batch) = (0.0, LR, BATCH);
        self.rng = Rng::new(SEED);
        self.running = true;
        self.rebuild();
    }

    /// New features (the same ones for the same numbers), new samples, new weights.
    fn rebuild(&mut self) {
        let (n, m) = (self.n, self.m());
        self.features = Features::new(n, m, &mut Rng::new(SEED + (n * 100 + m) as u64));
        self.resample();
        self.restart();
    }

    /// Fresh weights for the autoencoder; the data stay.
    fn restart(&mut self) {
        self.sae = Sae::new(self.m(), self.k, &mut self.rng);
        self.steps = 0;
        self.error = 1.0;
        self.firing = 0.0;
        self.history.iter_mut().for_each(History::clear);
        self.pace.reset();
        // (new weights: nothing to draw them coming from)
        (self.before, self.owed) = (Drawn::default(), 0.0);
        self.stale = true;
    }

    /// Draw the evaluation set again (after the data's settings change). It always
    /// uses the same random numbers, so a slider thins or thickens the same cloud.
    fn resample(&mut self) {
        let m = self.m();
        let mut rng = Rng::new(SEED + 7);
        self.features.sample(EVAL, self.sparsity, self.noise, &mut rng, &mut self.ex, &mut self.eh);
        (self.mean, self.pcs, self.pc_spread) = principal_components(&self.eh, m);
        self.variance = spread(&self.eh, m);
        let present = |s: usize| self.ex[s * self.n..(s + 1) * self.n].iter().filter(|x| **x > 0.0).count();
        // (the first samples, in order, until CLOUD of them hold some feature)
        (self.cloud, self.quiet) = (Vec::new(), Vec::new());
        for s in 0..EVAL {
            if self.cloud.len() == CLOUD {
                break;
            }
            if present(s) > 0 {
                self.cloud.push(s);
            } else {
                self.quiet.push(s);
            }
        }
        // start with a plain case: one feature, clearly present
        let good = |s: &&usize| present(**s) == 1 && self.ex[**s * self.n..(**s + 1) * self.n].iter().any(|&x| x > 0.7);
        self.selected = self.cloud.iter().find(good).or(self.cloud.first()).copied().unwrap_or(0);
        (self.skipped, self.turning) = (0, 0.0);
        self.stale = true;
    }

    fn train(&mut self, steps: usize) {
        for _ in 0..steps {
            self.features.sample(self.batch, self.sparsity, self.noise, &mut self.rng, &mut self.bx, &mut self.bh);
            let loss = self.sae.step(&self.bh, self.batch, self.lambda, self.lr);
            let share = loss.error / self.variance.max(1e-9);
            let blend = if self.steps == 0 { 1.0 } else { 0.05 };
            self.error += (share - self.error) * blend;
            self.firing += (loss.firing - self.firing) * blend;
            self.steps += 1;
            // (the histories keep themselves evenly spaced over the run)
            self.history[0].push(self.error);
            self.history[1].push(self.firing);
        }
    }

    fn refresh(&mut self) {
        self.report = Report::of(&self.sae, &self.features, &self.ex, &self.eh, self.variance);
        self.shown = self.steps;
        (self.stale, self.since) = (false, 0.0);
    }

    /// Point every dead latent at what is left over of an activation the autoencoder
    /// rebuilds badly (a different one each), so that it has something to learn.
    fn reseed_dead(&mut self) {
        let (m, k) = (self.m(), self.k);
        self.refresh();
        let (mut f, mut out) = (vec![0.0f32; k], vec![0.0f32; m]);
        let mut left: Vec<(f32, Vec<f32>)> = self
            .eh
            .chunks_exact(m)
            .map(|h| {
                self.sae.run(h, &mut f, &mut out);
                let rest: Vec<f32> = h.iter().zip(&out).map(|(a, b)| a - b).collect();
                (dot(&rest, &rest), rest)
            })
            .collect();
        left.sort_by(|a, b| b.0.total_cmp(&a.0));
        let dead: Vec<usize> = (0..k).filter(|&j| self.report.rate[j] < DEAD).collect();
        for (j, (_, rest)) in dead.into_iter().zip(left) {
            self.sae.reseed(j, &rest);
        }
        self.stale = true;
    }

    fn color(&self, j: usize) -> Color {
        const FEW: [Color; 8] = [pal::CYAN, pal::ORANGE, pal::GREEN, pal::PINK, pal::YELLOW, pal::PURPLE, pal::BLUE, Color::hex(0xB4E05A)];
        if self.k <= FEW.len() {
            FEW[j]
        } else {
            Cmap::Turbo.color(0.06 + 0.9 * j as f32 / (self.k - 1) as f32)
        }
    }

    fn the_view(&self) -> View {
        match (self.m(), self.stacked) {
            // the picture sits left of the panels (or, on a narrow screen, above them)
            // (room is kept above for the captions and below for the charts)
            // (in a short view the charts' titles come higher up: the picture moves up a little)
            (2, false) if self.short => View::Fit { center: Vec2::new(0.85, -0.3), half: Vec2::new(2.45, 2.3) },
            (2, false) => View::Fit { center: Vec2::new(0.85, 0.075), half: Vec2::new(2.45, 2.175) },
            (2, true) => View::Fit { center: Vec2::new(0.0, -1.35), half: Vec2::new(1.45, 3.5) },
            (3, false) => View::Orbit { target: Vec3::ZERO, yaw: 0.7, pitch: 0.38, dist: 8.6 },
            (3, true) => View::Orbit { target: Vec3::new(0.0, 0.0, -1.6), yaw: 0.7, pitch: 0.3, dist: 12.0 },
            _ => View::Fit { center: Vec2::ZERO, half: Vec2::splat(1.0) },
        }
    }

    /// Rows of the matrix panel: one per latent, and the principal components under
    /// them when those are shown.
    fn rows(&self) -> f32 {
        self.k as f32 + if self.pca { self.m() as f32 + 1.8 } else { 0.0 }
    }

    /// `top`: how much of the top of the view the application's own buttons cover
    /// (see `Frame::top_inset`); everything starts below that.
    fn layout(&self, view: Vec2, top: f32) -> Layout {
        let stacked = stacked(view);
        let pictured = self.m() <= 3;
        let n = self.n as f32;
        let rows = self.rows();
        if !stacked {
            // laid out in the part of the view below the strip, then moved down by it
            let (w, h) = (view.x, view.y - top);
            let brief = h < SHORT;
            let col = (0.30 * w).clamp(240.0, 340.0);
            let x0 = w - col - 12.0;
            let chart = Vec2::new((0.21 * w).clamp(150.0, 240.0), (0.12 * h).clamp(56.0, 96.0));
            let charts = [Vec2::new(30.0, h - chart.y - 30.0), Vec2::new(30.0 + chart.x + 44.0, h - chart.y - 30.0)];
            let lay = if pictured {
                let bars = 62.0;
                let cell = ((col - 34.0 - bars - 26.0) / n).min(0.40 * h / rows).min(30.0).max(2.0);
                let matrix = Vec2::new(x0 + 34.0, 80.0);
                let bottom = matrix.y + rows * cell;
                // (in a view too short for it, the scatter is left out: side 0)
                let room = (col - 84.0).min(h - bottom - 170.0);
                let side = if room < 50.0 { 0.0 } else { room.min(240.0) };
                Layout { stacked, brief, matrix, cell, bars, scatter: Vec2::new(x0 + 44.0, bottom + 122.0), side, charts, chart, apart: Vec2::ZERO, apart_size: Vec2::ZERO, key: Vec2::new(22.0, 20.0) }
            } else {
                let bars = 84.0;
                let cell = ((x0 - 70.0 - bars - 50.0) / n).min((h - 100.0 - chart.y - 150.0) / rows).min(24.0).max(2.0);
                let side = (col - 84.0).min(0.28 * h).clamp(50.0, 230.0);
                let scatter = Vec2::new(x0 + 44.0, 84.0);
                let apart = Vec2::new(x0 + 16.0, scatter.y + side + 104.0);
                let apart_size = Vec2::new(col - 32.0, (0.2 * h).clamp(84.0, 150.0));
                // (in a view too short for it, "taken apart" is left out: no size)
                let apart_size = if apart.y + apart_size.y + 36.0 > h { Vec2::ZERO } else { apart_size };
                Layout { stacked, brief, matrix: Vec2::new(64.0, 100.0), cell, bars, scatter, side, charts, chart, apart, apart_size, key: Vec2::new(22.0, 20.0) }
            };
            lay.lowered(top)
        } else {
            let (w, h) = (view.x, view.y);
            let chart = Vec2::new((w - 90.0) / 2.0, (0.09 * h).clamp(48.0, 80.0));
            let charts = [Vec2::new(24.0, h - chart.y - 24.0), Vec2::new(24.0 + chart.x + 42.0, h - chart.y - 24.0)];
            let floor = charts[0].y - 50.0;
            let key = Vec2::new(12.0, top + 6.0);
            if pictured {
                // the picture above; matrix and scatter side by side under it
                let (under, bars) = (0.47 * h + 62.0, 44.0);
                let side = (0.33 * w).min(floor - under - 16.0).max(40.0);
                let scatter = Vec2::new(w - side - 18.0, under);
                let room = scatter.x - 30.0 - 6.0 - (34.0 + 10.0 + bars + 12.0);
                let cell = (room / n).min((floor - under - 40.0) / rows).min(26.0).max(2.0);
                Layout { stacked, brief: true, matrix: Vec2::new(34.0, under), cell, bars, scatter, side, charts, chart, apart: Vec2::ZERO, apart_size: Vec2::ZERO, key }
            } else {
                // the matrix above; scatter and "taken apart" side by side under it
                let (under, bars) = (top + 78.0, 50.0);
                // (the matrix leaves the scatter under it a fair height)
                let keep = (0.22 * w).clamp(96.0, 130.0);
                let cell = ((w - 46.0 - bars - 14.0) / n).min((floor - under - 136.0 - keep) / rows).min(20.0).max(2.0);
                let below = under + rows * cell + 112.0;
                let side = (0.34 * w).min(floor - below - 24.0).clamp(40.0, 170.0);
                let apart = Vec2::new(34.0 + side + 36.0, below - 14.0);
                let apart_size = Vec2::new(w - apart.x - 14.0, side + 14.0);
                Layout { stacked, brief: true, matrix: Vec2::new(34.0, under), cell, bars, scatter: Vec2::new(34.0, below), side, charts, chart, apart, apart_size, key }
            }
        }
    }

    /// What the autoencoder makes of sample `s` of the evaluation set.
    fn read(&self, s: usize) -> Reading {
        let (m, k) = (self.m(), self.k);
        let mut reading = Reading { fired: vec![0.0; k], out: vec![0.0; m] };
        self.sae.run(&self.eh[s * m..(s + 1) * m], &mut reading.fired, &mut reading.out);
        reading
    }

    /// The sample that is being taken apart this frame.
    fn chosen(&self, hover: Option<usize>) -> usize {
        if self.m() <= 3 || self.cloud.is_empty() {
            hover.unwrap_or(self.selected)
        } else {
            // with no picture to point at, the samples take turns while it trains
            // (paused, the one on show stays)
            let turn = (self.turning / TURN) as usize;
            self.cloud[(turn + self.skipped) % self.cloud.len()]
        }
    }

    /// Bring the starts of the arrows of the sample taken apart `dt` seconds of the
    /// scene's time nearer to where the order (`laid`) puts them: base of the
    /// dictionary, then each latent's arrow from the tip of the one before. When two
    /// change places the arrows slide past each other (through two eases of `SLIDE`
    /// one after the other, exactly there in the end), rather than the joint between
    /// them jumping. With `at_once` (another sample, or training paused) they are put
    /// there at once; so is a latent that has just begun to fire.
    fn follow_starts(&mut self, fired: &[f32], at_once: bool, dt: f32) {
        let k = self.k;
        let mut to = vec![None; k];
        let mut at = v3(&self.drawn.bd);
        for &j in &self.laid.1 {
            to[j] = Some(at);
            at += v3(self.drawn.decoder(j)) * fired[j];
        }
        if at_once || self.starts.len() != k {
            self.starts = to.iter().map(|t| t.map(|t| (t, t))).collect();
            return;
        }
        let ease = 1.0 - (-dt.max(0.0) / SLIDE).exp();
        for (kept, to) in self.starts.iter_mut().zip(to) {
            *kept = match (*kept, to) {
                (Some((shown, between)), Some(to)) => {
                    let between = between + (to - between) * ease;
                    let shown = shown + (between - shown) * ease;
                    // (exactly there in the end: the picture then stops changing)
                    if (to - shown).length().max((to - between).length()) < 1e-4 {
                        Some((to, to))
                    } else {
                        Some((shown, between))
                    }
                }
                (_, to) => to.map(|t| (t, t)),
            };
        }
    }

    /// The cloud, the true directions, the dictionary, and one sample taken apart: its
    /// latents laid end to end, each from where `starts` has it, each as strong as
    /// `fired` says. (The dictionary and the sample's latents as the weights drawn
    /// have them: see `Drawn`.)
    fn picture(&self, f: &mut Frame, chosen: usize, fired: &[f32]) {
        let (n, m, k) = (self.n, self.m(), self.k);
        let flat = m == 2;
        let facing = -f.cam.fwd;
        f.circle(Vec3::ZERO, Vec3::Z, 1.0, 1.0, pal::FAINT.a(0.8));
        if !flat {
            f.circle(Vec3::ZERO, Vec3::X, 1.0, 1.0, pal::FAINT.a(0.5));
            f.circle(Vec3::ZERO, Vec3::Y, 1.0, 1.0, pal::FAINT.a(0.5));
        }
        // the truth: one spoke per feature, numbered
        for i in 0..n {
            let d = v3(self.features.dir(i));
            f.dashed(Vec3::ZERO, d * 1.2, 1.3, SPOKE.a(0.9));
            f.text(d * 1.31, (i + 1).to_string(), 12.0, SPOKE, Align2::CENTER_CENTER);
        }
        // what the autoencoder is given: the activations that hold some feature, and
        // fainter, those that hold none (at zero noise, all of them at the centre)
        for &s in &self.quiet {
            f.dot(v3(&self.eh[s * m..(s + 1) * m]), if flat { 4.5 } else { 4.0 }, DOT.a(0.3));
        }
        for &s in &self.cloud {
            f.dot(v3(&self.eh[s * m..(s + 1) * m]), if flat { 5.5 } else { 5.0 }, DOT.a(0.75));
        }
        if self.pca {
            let mean = v3(&self.mean);
            for c in 0..m {
                let reach = v3(&self.pcs[c * m..(c + 1) * m]) * (2.5 * self.pc_spread[c]).max(0.25);
                f.line(mean - reach, mean + reach, 2.4, pal::WHITE);
                f.text(mean + reach * 1.12, format!("PC {}", c + 1), 11.5, pal::WHITE, Align2::CENTER_CENTER);
            }
        }
        // the dictionary: one unit arrow per latent, from the offset the decoder adds
        let base = v3(&self.drawn.bd);
        for j in 0..k {
            let dead = self.report.rate.get(j).is_some_and(|r| *r < DEAD);
            let c = self.color(j).a(if dead { 0.3 } else { 1.0 });
            let tip = base + v3(self.drawn.decoder(j));
            f.arrow(base, tip, 3.2, c);
            if k <= 16 {
                f.text(base + (tip - base) * 1.09, letter(j), 12.0, c, Align2::CENTER_CENTER);
            }
            if self.show_encoder {
                let hears = base + v3(self.drawn.encoder(j));
                f.line(base, hears, 1.4, pal::WHITE.a(0.8));
                f.dl.overlay_discs.push(hears, 8.0, c);
            }
        }
        // one activation taken apart: the latents that fire, each scaled along its
        // arrow and laid end to end, reach the reconstruction; red is what is missed
        if self.cloud.is_empty() {
            return;
        }
        let h = &self.eh[chosen * m..(chosen + 1) * m];
        // (each latent in turn, whatever its place in the chain: drawn in the order of
        // the chain, two that change places would swap their places in the list drawn)
        let mut at = base;
        for j in (0..k).filter(|&j| fired[j] > 0.0) {
            let v = v3(self.drawn.decoder(j)) * fired[j];
            let start = self.starts.get(j).copied().flatten().map_or(at, |(shown, _)| shown);
            f.arrow(start, start + v, 5.5, pal::WHITE);
            f.arrow(start, start + v, 3.0, self.color(j));
            at += v;
        }
        let target = v3(h);
        f.line(at, target, 2.2, pal::RED);
        f.dot(at, 8.0, pal::WHITE);
        let r = f.cam.world_per_px(target) * f.px * 9.0;
        f.circle(target, if flat { Vec3::Z } else { facing }, r, 2.0, pal::WHITE);
    }

    /// Which true feature each latent has found: the matrix of cosines, each row with
    /// how often its latent fires.
    fn matrix(&self, f: &mut Frame, lay: &Layout) {
        let (n, m, k) = (self.n, self.m(), self.k);
        let r = &self.report;
        if r.order.len() != k || r.cosine.len() != k * n {
            return;
        }
        let (o, cell) = (lay.matrix, lay.cell);
        let size = Vec2::new(n as f32 * cell, k as f32 * cell);
        let rows = self.rows();
        let bars_at = o.x + size.x + 10.0;
        // (on a narrow screen the panel is no wider than its matrix, and says less)
        let short = lay.stacked;
        let right = if short { bars_at + lay.bars + 12.0 } else { (bars_at + lay.bars + 12.0).max(o.x + 262.0).min(f.view_size().x - 6.0) };
        hud_panel(f, o - Vec2::new(30.0, 62.0), Vec2::new(right, o.y + rows * cell + 58.0));
        let title = if short { "what each latent has found" } else { "which true feature has each latent found?" };
        let sub = if short { "rows: latents   columns: features" } else { "rows: latents   columns: true features   bright: same direction" };
        f.text_px(o + Vec2::new(-22.0, -47.0), title, 11.5, pal::TEXT, Align2::LEFT_CENTER);
        f.text_px(o + Vec2::new(-22.0, -31.0), sub, 9.5, pal::DIM, Align2::LEFT_CENTER);
        let heat = Cmap::Magma.table();
        hud_heat(f, o, size, k, n, |i, j| heat[table_index(0.1 + 0.9 * r.cosine[r.order[i] * n + j].clamp(0.0, 1.0))]);
        if cell >= 13.0 {
            for i in 0..n {
                f.text_px(o + Vec2::new((i as f32 + 0.5) * cell, -8.0), (i + 1).to_string(), 9.5, SPOKE, Align2::CENTER_CENTER);
            }
        }
        // how often each latent fires, against how often a true feature is present
        let present = 1.0 - self.sparsity;
        let full = r.rate.iter().copied().fold(present, f32::max).max(1e-4) * 1.12;
        for (row, &j) in r.order.iter().enumerate() {
            let y = o.y + row as f32 * cell;
            let c = self.color(j);
            hud_rect(f, Vec2::new(o.x - 8.0, y), Vec2::new(o.x - 3.0, y + (cell - 1.0).max(1.0)), c);
            if cell >= 11.0 {
                f.text_px(Vec2::new(o.x - 15.0, y + 0.5 * cell), letter(j), cell.min(11.0), c, Align2::CENTER_CENTER);
            }
            if r.rate[j] < DEAD {
                if cell >= 9.0 {
                    f.text_px(Vec2::new(bars_at, y + 0.5 * cell), "dead", cell.min(10.5), pal::RED, Align2::LEFT_CENTER);
                } else {
                    hud_rect(f, Vec2::new(bars_at, y), Vec2::new(bars_at + 5.0, y + (cell - 1.0).max(1.0)), pal::RED);
                }
            } else {
                hud_rect(f, Vec2::new(bars_at, y + 0.15 * cell), Vec2::new(bars_at + lay.bars * r.rate[j] / full, y + (0.85 * cell).max(1.0)), c);
            }
        }
        let mark = bars_at + lay.bars * present / full;
        hud_line(f, Vec2::new(mark, o.y - 3.0), Vec2::new(mark, o.y + size.y + 3.0), 1.2, pal::TEXT.a(0.8));
        f.text_px(Vec2::new(bars_at, o.y - 8.0), "fires in", 9.5, pal::DIM, Align2::LEFT_CENTER);
        let mut foot = o.y + size.y + 13.0;
        if self.pca {
            // the principal components, measured against the same features
            let top = o.y + (k as f32 + 1.8) * cell;
            let title = if short { "principal components instead" } else { "principal components against the same features" };
            f.text_px(Vec2::new(o.x - 22.0, top - 0.9 * cell.max(11.0)), title, 9.5, pal::DIM, Align2::LEFT_CENTER);
            let grey = Cmap::Grays.table();
            hud_heat(f, Vec2::new(o.x, top), Vec2::new(size.x, m as f32 * cell), m, n, |c, i| {
                grey[table_index(0.08 + 0.92 * dot(&self.pcs[c * m..(c + 1) * m], self.features.dir(i)).abs())]
            });
            foot = top + m as f32 * cell + 13.0;
        }
        let (bars, line) = if short {
            ("bars: how often each fires".to_string(), format!("line: one feature's rate ({:.0} %)", present * 100.0))
        } else {
            ("bars: the share of activations each latent fires in".to_string(), format!("white line: how often one true feature is present ({:.0} %)", present * 100.0))
        };
        f.text_px(Vec2::new(o.x - 22.0, foot), bars, 9.5, pal::DIM, Align2::LEFT_CENTER);
        f.text_px(Vec2::new(o.x - 22.0, foot + 13.0), line, 9.5, pal::DIM, Align2::LEFT_CENTER);
        f.text_px(Vec2::new(o.x - 22.0, foot + 30.0), format!("features found: {} of {}", r.recovered, n), 11.5, pal::TEXT, Align2::LEFT_CENTER);
    }

    /// True size of a feature against what its latents report: shrinkage, made visible.
    fn scatter(&self, f: &mut Frame, lay: &Layout) {
        let (o, s) = (lay.scatter, lay.side);
        if s <= 0.0 {
            return;
        }
        hud_panel(f, o - Vec2::new(30.0, 54.0), o + Vec2::new(s + 18.0, s + 34.0));
        let short = lay.stacked;
        let title = if short { "is the true size reported?" } else { "do latents report a feature's true size?" };
        let sub = if short { "across: true   up: reported" } else { "across: its true size   up: what its latents report" };
        f.text_px(o + Vec2::new(-22.0, -40.0), title, 11.5, pal::TEXT, Align2::LEFT_CENTER);
        f.text_px(o + Vec2::new(-22.0, -24.0), sub, 9.5, pal::DIM, Align2::LEFT_CENTER);
        let at = |x: f32, y: f32| o + Vec2::new(x.clamp(0.0, 1.0) * s, (1.0 - y.clamp(0.0, 1.0)) * s);
        hud_line(f, at(0.0, 0.0), at(1.0, 0.0), 1.0, pal::DIM.a(0.8));
        hud_line(f, at(0.0, 0.0), at(0.0, 1.0), 1.0, pal::DIM.a(0.8));
        // the line a faithful reading would lie on
        for d in 0..12 {
            let (a, b) = (d as f32 / 12.0, (d as f32 + 0.55) / 12.0);
            hud_line(f, at(a, a), at(b, b), 1.2, pal::TEXT.a(0.7));
        }
        f.text_px(at(1.0, 1.0) + Vec2::new(4.0, -8.0), "dashed: exact", 9.5, pal::TEXT.a(0.8), Align2::RIGHT_CENTER);
        f.text_px(at(0.0, 0.0) + Vec2::new(-5.0, 9.0), "0", 9.5, pal::DIM, Align2::CENTER_CENTER);
        f.text_px(at(1.0, 0.0) + Vec2::new(0.0, 9.0), "1", 9.5, pal::DIM, Align2::CENTER_CENTER);
        f.text_px(at(0.0, 1.0) + Vec2::new(-8.0, 0.0), "1", 9.5, pal::DIM, Align2::CENTER_CENTER);
        let r = &self.report;
        for &(x, got, i) in &r.pairs {
            let p = at(x, got);
            let c = r.best.get(i as usize).map_or(DOT, |b| self.color(b.0));
            hud_rect(f, p - Vec2::splat(1.5), p + Vec2::splat(1.5), c);
        }
        let verdict = match (r.pairs.is_empty(), short) {
            (true, false) => "no latent stands for a feature yet".to_string(),
            (true, true) => "nothing to compare yet".to_string(),
            (false, false) => format!("on average they report {:.0} % of it", r.read * 100.0),
            (false, true) => format!("reported: {:.0} % on average", r.read * 100.0),
        };
        f.text_px(o + Vec2::new(-22.0, s + 20.0), verdict, 11.0, pal::TEXT, Align2::LEFT_CENTER);
    }

    /// Reconstruction error and number of latents firing, over training.
    fn charts(&self, f: &mut Frame, lay: &Layout) {
        let small = if lay.stacked { 10.0 } else { 11.5 };
        let [a, b] = lay.charts;
        hud_chart(f, a, lay.chart, &[(self.history[0].points(), pal::CYAN)], true, None);
        f.text_px(a + Vec2::new(-2.0, -18.0), format!("{} (log scale)  {:.1} %", if lay.stacked { "unexplained" } else { "variance left unexplained" }, self.report.unexplained * 100.0), small, pal::CYAN, Align2::LEFT_CENTER);
        // how many features are truly present per activation: what L0 should come down to
        let truly = self.n as f32 * (1.0 - self.sparsity);
        let top = self.history[1].points().iter().copied().fold(truly, f32::max).max(0.05) * 1.15;
        hud_chart(f, b, lay.chart, &[(self.history[1].points(), pal::YELLOW)], false, Some([0.0, top]));
        let y = b.y + lay.chart.y * (1.0 - truly / top);
        for d in 0..16 {
            let x = |t: f32| b.x + lay.chart.x * t / 16.0;
            hud_line(f, Vec2::new(x(d as f32), y), Vec2::new(x(d as f32 + 0.5), y), 1.1, pal::TEXT.a(0.7));
        }
        f.text_px(Vec2::new(b.x + lay.chart.x, y + 9.0), format!("dashed: features truly present ({truly:.1})"), 9.5, pal::TEXT.a(0.85), Align2::RIGHT_CENTER);
        f.text_px(b + Vec2::new(-2.0, -18.0), format!("latents firing per activation (L0)  {:.2}", self.report.l0), small, pal::YELLOW, Align2::LEFT_CENTER);
        if !lay.stacked {
            f.text_px(a + Vec2::new(lay.chart.x, lay.chart.y + 15.0), "training steps →", 9.5, pal::DIM, Align2::RIGHT_CENTER);
            f.text_px(b + Vec2::new(lay.chart.x, lay.chart.y + 15.0), "training steps →", 9.5, pal::DIM, Align2::RIGHT_CENTER);
        }
    }

    /// Without a picture: one activation as bars, the features truly in it above the
    /// line and what the latents standing for them report below it.
    fn apart(&self, f: &mut Frame, lay: &Layout, chosen: usize, reading: &Reading) {
        let (n, k) = (self.n, self.k);
        let (o, size) = (lay.apart, lay.apart_size);
        hud_panel(f, o - Vec2::new(8.0, 40.0), o + size + Vec2::new(8.0, 36.0));
        f.text_px(o + Vec2::new(0.0, -26.0), "one activation, taken apart", 11.5, pal::TEXT, Align2::LEFT_CENTER);
        let sub = if lay.stacked { "up: truly in it   down: reported" } else { "up: features truly in it   down: what their latents report" };
        f.text_px(o + Vec2::new(0.0, -10.0), sub, 9.5, pal::DIM, Align2::LEFT_CENTER);
        let fired = &reading.fired;
        let (mid, half, bar) = (o.y + 0.5 * size.y, 0.5 * size.y - 2.0, size.x / n as f32);
        let mut stray = 0;
        let mut reported = vec![0.0f32; n];
        for j in 0..k {
            match self.report.owner.get(j).copied().flatten() {
                Some(i) => reported[i] += fired[j],
                None => stray += (fired[j] > 0.0) as usize,
            }
        }
        for i in 0..n {
            let (x0, x1) = (o.x + i as f32 * bar + 0.5, o.x + (i as f32 + 1.0) * bar - 0.5);
            let truly = self.ex[chosen * n + i].min(1.0);
            if truly > 0.0 {
                hud_rect(f, Vec2::new(x0, mid - truly * half), Vec2::new(x1, mid), pal::TEXT);
            }
            if reported[i] > 0.0 {
                let c = self.report.best.get(i).map_or(DOT, |b| self.color(b.0));
                hud_rect(f, Vec2::new(x0, mid), Vec2::new(x1, mid + reported[i].min(1.0) * half), c);
            }
        }
        hud_line(f, Vec2::new(o.x, mid), Vec2::new(o.x + size.x, mid), 1.0, pal::DIM);
        f.text_px(o + Vec2::new(0.0, size.y + 10.0), format!("feature 1 … {n} →"), 9.5, pal::DIM, Align2::LEFT_CENTER);
        let text = if lay.stacked { format!("firing for no feature: {stray}") } else { format!("latents firing that stand for no feature: {stray}") };
        f.text_px(o + Vec2::new(0.0, size.y + 24.0), text, 10.5, if stray > 0 { pal::RED } else { pal::DIM }, Align2::LEFT_CENTER);
    }

    /// What the marks in the picture mean, and the chosen activation in words.
    fn captions(&self, f: &mut Frame, lay: &Layout, chosen: usize, reading: &Reading) {
        let (n, m, k) = (self.n, self.m(), self.k);
        if m > 3 {
            let text = if lay.stacked {
                format!("{n} features in {m} dimensions, {k} latents looking for them")
            } else {
                format!("{n} features hidden in {m} dimensions, {k} latents looking for them (too many dimensions to draw)")
            };
            f.caption_px(lay.key, text, if lay.stacked { 10.5 } else { 12.0 }, pal::TEXT, Align2::LEFT_TOP);
            return;
        }
        if lay.brief {
            f.caption_px(lay.key, "dots: activations   dashed: true features   arrows: what it found", 10.0, pal::TEXT, Align2::LEFT_TOP);
            return;
        }
        // (an empty line is a small gap)
        let mut lines: Vec<(String, Color)> = vec![
            ("dots: activations, all the autoencoder is given".into(), DOT),
            ("dashed spokes 1, 2, …: the true features, which it never sees".into(), SPOKE),
            ("bold arrows A, B, …: the directions it has learned (its dictionary)".into(), pal::CYAN),
        ];
        if !self.quiet.is_empty() {
            lines.insert(1, ("faint dots: activations holding no feature, only noise (with no noise, all at the centre)".into(), DOT.a(0.6)));
        }
        if self.pca {
            lines.push(("white lines: principal components of the cloud".into(), pal::WHITE));
        }
        if !self.cloud.is_empty() {
            // the chosen activation in words: what was in it, and what the autoencoder says
            // (each a formula: so many times a feature, plus so many times another)
            let term = |v: f32, name: String| format!(r"{v:.2} \times \text{{{name}}}");
            let mut truly: Vec<(f32, usize)> = (0..n).map(|i| (self.ex[chosen * n + i], i)).filter(|t| t.0 > 0.0).collect();
            truly.sort_by(|a, b| b.0.total_cmp(&a.0));
            let (fired, out) = (&reading.fired, &reading.out);
            let h = &self.eh[chosen * m..(chosen + 1) * m];
            let mut read: Vec<(f32, usize)> = (0..k).map(|j| (fired[j], j)).filter(|t| t.0 > 0.0).collect();
            read.sort_by(|a, b| b.0.total_cmp(&a.0));
            let join = |terms: Vec<String>| if terms.is_empty() { "nothing".to_string() } else { format!("${}$", terms.join(" + ")) };
            let missed = h.iter().zip(out).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
            lines.push((String::new(), pal::WHITE));
            lines.push((format!("the ringed activation is truly  {}", join(truly.iter().take(4).map(|&(v, i)| term(v, format!("feature {}", i + 1))).collect())), pal::WHITE));
            lines.push((format!("the autoencoder reads it as  {}", join(read.iter().take(4).map(|&(v, j)| term(v, format!("latent {}", letter(j)))).collect())), pal::WHITE));
            lines.push((format!("red: what that misses ({missed:.2}).  Point at a dot to take it apart; click to keep it."), pal::RED));
        }
        // (a sum too long to end before the matrix's panel goes on a line of its own,
        // under the words that bring it in)
        let room = lay.matrix.x - 36.0 - lay.key.x;
        let mut y = lay.key.y;
        for (text, color) in lines {
            if text.is_empty() {
                y += 6.0;
                continue;
            }
            if let Some((words, sum)) = text.split_once("  $").filter(|_| f.text_size(&text, 11.5).x > room) {
                f.caption_px(Vec2::new(lay.key.x, y), words, 11.5, color, Align2::LEFT_TOP);
                f.caption_px(Vec2::new(lay.key.x + 14.0, y + 18.0), format!("${sum}"), 11.5, color, Align2::LEFT_TOP);
                y += 36.0;
                continue;
            }
            f.caption_px(Vec2::new(lay.key.x, y), text, 11.5, color, Align2::LEFT_TOP);
            y += 18.0;
        }
    }
}

impl Scene for SaeScene {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        // charts and matrices read better without a halo
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.running || self.m() == 3
    }

    /// In space the arrangement is a solid: turn it slowly so its shape can be read.
    fn auto_rotate(&self) -> f32 {
        if self.m() == 3 {
            0.12
        } else {
            0.0
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A network can hold more features than it has dimensions by giving them overlapping directions (see Superposition). Here we are handed only its activations and must find those directions. A sparse autoencoder rewrites every activation as a sum of a few directions from a list it learns, and is charged for each one it uses. That charge is what makes the list line up with the features.");
        ui::note(ui, r"Latent $j$ listens along $e_j$ and writes back along its arrow $d_j$, of length 1. Given an activation $h$, it fires $f_j = \operatorname{ReLU}(e_j \cdot (h - b) + c_j)$, and $h$ is rebuilt as");
        ui::equation(ui, r"\hat h = b + \sum_j f_j d_j");
        ui::note(ui, "Training makes the error and the penalty together small, on average:");
        ui::equation(ui, r"\lVert h - \hat h \rVert^2 + \lambda \sum_j f_j");
        match ui::buttons(ui, &[if self.running { "pause" } else { "train" }, "new weights", "re-seed dead latents"]) {
            Some(0) => self.running = !self.running,
            Some(1) => self.restart(),
            Some(2) => self.reseed_dead(),
            _ => {}
        }
        ui::note(ui, "Re-seeding points every dead latent at an activation that is rebuilt badly, to give it something to learn.");
        ui::section(ui, "The words");
        ui::term(ui, "feature", "one thing the network represents, stored as a direction. An activation is the sum of the few features present in it.");
        ui::term(ui, "latent", r"one unit of the autoencoder. It fires ($f_j > 0$) when it detects its direction in an activation.");
        ui::term(ui, "dictionary", r"the list of directions $d_j$, one per latent, that activations are rebuilt from.");
        ui::term(ui, r"L1 penalty $\lambda$", "the price charged for every unit of latent activation.");
        ui::term(ui, "L0", "how many latents fire for one activation, on average.");
        ui::term(ui, "dead latent", "one that never fires, and so never learns.");
        ui::term(ui, "shrinkage", "latents reporting less than was truly there, because every unit they report is charged for.");

        ui::section(ui, "Things to try");
        for (label, look_for, setup) in TRY {
            if ui::experiment(ui, label, look_for) {
                self.apply(setup);
            }
        }

        if self.m() == 2 {
            ui::note(ui, "In the plane, also point at a dot that lies between two spokes: it holds two features at once. Two that point nearly opposite ways partly cancel, and the autoencoder reads what is left as the feature between them.");
        }

        ui::section(ui, "The hidden features");
        let labels: Vec<String> = DIMS.iter().map(|d| format!("{d} dimensions")).collect();
        let labels: Vec<&str> = labels.iter().map(String::as_str).collect();
        let mut dims = self.dims;
        if ui::choice(ui, &labels, &mut dims) {
            let (n, sparsity, lambda) = DEFAULTS[dims];
            self.apply(setup(dims, n, n, sparsity, lambda, self.pca));
        }
        if ui::slider_int(ui, "features", &mut self.n, 2..=MAX_FEATURES[self.dims]) {
            self.rebuild();
        }
        let mut data = ui::slider(ui, "sparsity", &mut self.sparsity, 0.0..=0.99);
        ui::note(ui, "The chance that a feature is absent from any one activation.");
        data |= ui::slider(ui, "noise", &mut self.noise, 0.0..=0.2);
        if data {
            self.resample();
        }

        ui::section(ui, "The autoencoder");
        if ui::slider_int(ui, "latents (dictionary size)", &mut self.k, 1..=MAX_LATENTS) {
            self.restart();
        }
        ui::note(ui, "A real network does not say how many features it holds, so this is always a guess.");
        ui::slider_log_or_zero(ui, r"L1 penalty $\lambda$", &mut self.lambda, 0.01, 2.0);
        ui::note(ui, "At zero only the rebuilding counts. The larger it is, the fewer latents fire and the less they report.");

        ui::section(ui, "Show");
        if ui.checkbox(&mut self.pca, "principal components (PCA)").changed() {
            self.stale = true;
        }
        if self.m() <= 3 {
            ui.checkbox(&mut self.show_encoder, "what each latent listens for");
            ui::note(ui, r"A thin line ending in a dot, per latent: the direction $e_j$ it listens along (its encoder), which need not be the direction $d_j$ it writes back (its arrow).");
        }

        ui::section(ui, "Training");
        ui::slider_log(ui, "learning rate", &mut self.lr, 0.001..=0.1);
        ui::slider_int_log(ui, "batch size", &mut self.batch, 32..=1024);
        ui::slider_int_log(ui, "steps per second", &mut self.rate, 10..=5000);

        ui::section(ui, "Colours");
        ui::key(
            ui,
            &[
                (DOT, "a dot: one activation, all the autoencoder is given; a faint one holds no feature, only noise (with no noise, these all sit at the centre)"),
                (SPOKE, "dashed spokes and numbers: the true features, hidden from it"),
                (self.color(0), "each latent has a colour and a letter: its arrow, its row of the matrix, its dots in the scatter"),
                (pal::RED, "what the autoencoder misses of the ringed activation; a dead latent"),
            ],
        );

        ui::section(ui, "Numbers");
        let r = &self.report;
        ui::readout(ui, "training steps", count(self.steps as usize));
        ui::readout(ui, "variance left unexplained", format!("{:.1} %", r.unexplained * 100.0));
        ui::readout(ui, "latents firing per activation (L0)", format!("{:.2}", r.l0));
        ui::readout(ui, "features truly present per activation", fmt(self.n as f32 * (1.0 - self.sparsity)));
        ui::readout(ui, "features found", format!("{} of {}", r.recovered, self.n));
        ui::readout(ui, "average best cosine", format!("{:.3}", r.mean_best));
        ui::readout(ui, "dead latents", format!("{} of {}", r.dead, self.k));
        ui::readout(ui, "latents report", if r.pairs.is_empty() { "–".to_string() } else { format!("{:.0} % of true size", r.read * 100.0) });
        ui::note(ui, "A feature counts as found when some latent's direction is within 18° of it (cosine above 0.95) and that latent answers to it rather than to its neighbours: shown the features one at a time, at least three quarters of its response is to its own. Where features crowd one another (five or more in the plane, seven or more in space), that rules out a direction that merely happens to point the right way before training. Where they are few and far apart, there is nothing nearby to mistake a feature for, so a latent that happens to point at one already answers to it alone, and counts, trained or not. Even a found feature is read low: partly the penalty, and partly that crowded features force each latent to ignore small activations so as not to fire for its neighbours.");

        ui::see_also(ui, &[("superposition", "Superposition: where overlapping features come from"), ("sensing", "Compressed sensing: the same 1-norm idea, where it can be proved to work"), ("transformer", "A transformer, opened up"), ("mlp", "Network playground")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        let (stacked, short) = (stacked(view), view.y - f.top_inset < SHORT);
        if (stacked, short) != (self.stacked, self.short) {
            (self.stacked, self.short) = (stacked, short);
            self.new_view = true;
        }
        // train, within a time allowance (keeping the weights from before each step, for
        // the picture: see `Drawn`)
        if self.running && f.dt > 0.0 {
            let rate = self.rate as f32;
            let mut turn = self.pace.begin(rate, f.dt, f.real_dt, 0.004);
            while turn.next() {
                self.before.keep(&self.sae);
                self.train(1);
            }
            let taken = self.pace.end(turn);
            // (what the pace owes, counted the same way: it grows with the scene's time,
            // up to the same cap, and each step taken pays one)
            self.owed = ((self.owed + rate * f.dt).min((rate * 0.1).max(2.0)) - taken as f32).max(0.0);
        }
        // paused, the picture is the weights as they are
        let share = if self.running { self.owed.min(1.0) } else { 1.0 };
        self.drawn.between(&self.before, &self.sae, share);
        // measure: a small dictionary in every frame that trained, a large one not
        // more often than every MEASURE seconds, however often the display redraws
        // (headless renders step real time evenly, so they repeat). At that pace a
        // display of 60 frames a second gets new numbers in every other frame: the
        // chart and the bars of the small dictionaries, which cost next to nothing to
        // measure, moved thirty times a second in a picture drawn sixty.
        self.since += f.real_dt;
        let small = self.k * self.m() <= 64;
        if self.stale || (self.steps != self.shown && (small || self.since >= MEASURE)) {
            self.refresh();
        }
        if self.running {
            self.turning += f.dt;
        }

        let pictured = self.m() <= 3;
        // the dot under the pointer, if any
        let hover = f.ptr.pos.filter(|_| pictured).and_then(|p| {
            let m = self.m();
            let near = |s: &usize| on_screen(f, v3(&self.eh[*s * m..(*s + 1) * m])).map(|q| (q - p / f.px).length());
            self.cloud.iter().chain(&self.quiet).filter_map(|s| near(s).map(|d| (d, *s))).filter(|t| t.0 < 12.0).min_by(|a, b| a.0.total_cmp(&b.0)).map(|t| t.1)
        });
        if !pictured {
            // (a click anywhere moves on to the next example there, so both clicks of a
            // double-click are the scene's: R still resets the view)
            f.handles.claim_clicks();
        }
        if f.clicked_canvas() {
            match hover {
                Some(s) => self.selected = s,
                None if !pictured => self.skipped += 1,
                None => {}
            }
        }
        let chosen = self.chosen(hover);
        let reading = self.read(chosen);
        let lay = self.layout(view, f.top_inset);
        if pictured {
            // (the sample as the weights drawn read it; a new sample is laid out afresh,
            // strongest first)
            let m = self.m();
            let fired = self.drawn.fired(&self.eh[chosen * m..(chosen + 1) * m]);
            let kept = if self.laid.0 == chosen { std::mem::take(&mut self.laid.1) } else { Vec::new() };
            let again = kept.is_empty();
            self.laid = (chosen, laid(&kept, &fired));
            self.follow_starts(&fired, again || !self.running, f.dt);
            self.picture(f, chosen, &fired);
        }
        self.matrix(f, &lay);
        self.scatter(f, &lay);
        self.charts(f, &lay);
        if !pictured && !self.cloud.is_empty() && lay.apart_size.y > 0.0 {
            self.apart(f, &lay, chosen, &reading);
        }
        self.captions(f, &lay, chosen, &reading);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scene itself, set up as an experiment and trained for `steps`.
    fn run(setup: Setup, steps: usize) -> SaeScene {
        let mut scene = SaeScene::new();
        scene.apply(setup);
        scene.train(steps);
        scene.refresh();
        scene
    }

    /// The same with other starting weights (`again` restarts first).
    fn rerun(setup: Setup, steps: usize, again: usize) -> SaeScene {
        let mut scene = SaeScene::new();
        scene.apply(setup);
        for _ in 0..again {
            scene.restart();
        }
        scene.train(steps);
        scene.refresh();
        scene
    }

    // ----- the scene's text, held to account ------------------------------------------

    #[test]
    fn the_default_finds_all_five_features() {
        for again in 0..3 {
            let s = rerun(TRY[0].2, 1500, again);
            let r = &s.report;
            assert_eq!(r.recovered, 5, "weights {again}: best cosines {:?}", r.best);
            assert_eq!(r.dead, 0);
            // one latent per feature: about as many fire as there are features present
            assert!((r.l0 - 0.5).abs() < 0.12, "L0 {}", r.l0);
            assert!(r.unexplained < 0.2, "unexplained {}", r.unexplained);
        }
    }

    #[test]
    fn without_the_penalty_the_rebuilding_is_exact_but_nothing_lines_up() {
        for again in 0..3 {
            let s = rerun(TRY[1].2, 4000, again);
            let r = &s.report;
            assert!(r.unexplained < 0.01, "unexplained {}", r.unexplained);
            assert!(r.recovered < 5, "all five found without a penalty");
            // "two or three latents fire for almost every activation", against half a
            // latent by default
            assert!(r.l0 > 1.8 && r.l0 < 3.2, "L0 {}", r.l0);
            assert!(r.rate.iter().any(|rate| *rate > 0.7), "no latent fires nearly always: {:?}", r.rate);
        }
    }

    #[test]
    fn too_much_penalty_shrinks_the_reading_and_then_silences_it() {
        let usual = run(TRY[0].2, 3000).report.read;
        for again in 0..3 {
            let s = rerun(TRY[2].2, 4000, again);
            let r = &s.report;
            assert!(r.recovered >= 3, "found only {}", r.recovered);
            assert!(r.read < 0.3 && r.read < usual - 0.25, "reads {} against {usual}", r.read);
            assert!(r.unexplained > 0.5, "unexplained {}", r.unexplained);
        }
        let silent = run(Setup { lambda: 2.0, ..TRY[2].2 }, 4000);
        assert!(silent.report.l0 < 0.05, "L0 {}", silent.report.l0);
        assert!(silent.report.unexplained > 0.9);
    }

    #[test]
    fn the_reading_falls_as_the_penalty_grows() {
        let read = |lambda: f32| run(Setup { lambda, ..TRY[0].2 }, 3000).report.read;
        let (a, b, c) = (read(0.3), read(0.6), read(1.0));
        assert!(a > b + 0.08 && b > c + 0.08, "{a} {b} {c}");
        assert!(a < 0.8, "the default already reads low: {a}");
    }

    /// The note under the numbers: crowding raises the threshold beyond what the
    /// penalty alone asks for (half of lambda, when features are at right angles).
    #[test]
    fn crowded_features_force_a_higher_threshold() {
        let threshold = |s: &SaeScene| (0..s.k).map(|j| -s.sae.be[j] / dot(s.sae.encoder(j), s.sae.encoder(j)).sqrt()).sum::<f32>() / s.k as f32;
        let square = run(Setup { n: 4, k: 4, ..TRY[0].2 }, 3000);
        let pentagon = run(TRY[0].2, 3000);
        assert_eq!(square.report.recovered, 4);
        let (t4, t5) = (threshold(&square), threshold(&pentagon));
        assert!((t4 - 0.15).abs() < 0.03, "right angles: threshold {t4}, half the penalty is 0.15");
        assert!(t5 > 1.4 * t4, "pentagon {t5} against {t4}");
        assert!(pentagon.report.read < square.report.read - 0.03);
    }

    #[test]
    fn three_latents_cannot_cover_five_features() {
        let usual = run(TRY[0].2, 3000).report.unexplained;
        for again in 0..3 {
            let s = rerun(TRY[3].2, 4000, again);
            let r = &s.report;
            // every latent answers for a neighbour as well as for the feature it is
            // nearest to, so none of them counts as having found it
            assert!(r.telling.iter().all(|&t| t > 0.3 && t < TELLING), "weights {again}: {:?}", r.telling);
            assert_eq!(r.recovered, 0);
            assert!(r.unexplained > 0.17 && r.unexplained < 0.33, "unexplained {}", r.unexplained);
            assert!(r.unexplained > 1.6 * usual);
        }
    }

    #[test]
    fn spare_latents_die_or_copy_a_feature() {
        for again in 0..3 {
            let s = rerun(TRY[4].2, 4000, again);
            let r = &s.report;
            assert_eq!(r.recovered, 5);
            let live: Vec<usize> = (0..s.k).filter(|&j| r.rate[j] >= DEAD).collect();
            assert!(live.iter().all(|&j| r.nearest[j].1 > RECOVERED), "a live latent sits on no feature: {:?}", r.nearest);
            let copies = (0..5).filter(|&i| live.iter().filter(|&&j| r.nearest[j].0 == i).count() >= 2).count();
            assert!(copies >= 1, "no feature has two latents");
            assert_eq!(live.len() + r.dead, 10);
            // the copies share the activation: more latents fire than features are present
            assert!(r.l0 > 0.6, "L0 {}", r.l0);
        }
    }

    #[test]
    fn thirty_two_features_are_found_in_sixteen_dimensions_unless_they_come_together() {
        let sparse = run(TRY[5].2, 3000);
        assert_eq!(sparse.report.recovered, 32, "best cosines {:?}", sparse.report.best);
        let dense = run(TRY[6].2, 3000);
        assert!(dense.report.recovered <= 3, "found {} of 32 dense features", dense.report.recovered);
        assert!((dense.n as f32 * (1.0 - dense.sparsity) - 16.0).abs() < 1e-3);
    }

    #[test]
    fn principal_components_are_not_the_features() {
        // the pentagon's cloud is about as wide in every direction
        let plane = run(TRY[7].2, 10);
        assert!(plane.pca && plane.pc_spread[0] / plane.pc_spread[1] < 1.2, "spreads {:?}", plane.pc_spread);
        // and in sixteen dimensions no component points along any feature
        let s = run(TRY[5].2, 10);
        let m = s.m();
        let best = (0..m * s.n).map(|e| dot(&s.pcs[(e / s.n) * m..(e / s.n + 1) * m], s.features.dir(e % s.n)).abs()).fold(0.0, f32::max);
        assert!(best < 0.8, "a principal component lies along a feature (cosine {best})");
        for c in 0..m {
            assert!((dot(&s.pcs[c * m..(c + 1) * m], &s.pcs[c * m..(c + 1) * m]) - 1.0).abs() < 1e-3);
        }
    }

    /// The hint about dots between the spokes.
    #[test]
    fn two_features_that_nearly_cancel_are_read_as_the_one_between() {
        let s = run(TRY[0].2, 3000);
        let (d1, d3) = (s.features.dir(0), s.features.dir(2));
        let h = [0.75 * d1[0] + 0.6 * d3[0], 0.75 * d1[1] + 0.6 * d3[1]];
        let (mut f, mut out) = (vec![0.0; s.k], vec![0.0; 2]);
        s.sae.run(&h, &mut f, &mut out);
        let strongest = (0..s.k).max_by(|&a, &b| f[a].total_cmp(&f[b])).unwrap();
        assert!(f[strongest] > 0.0);
        assert_eq!(s.report.owner[strongest], Some(1), "read as {:?} with {f:?}", s.report.owner[strongest]);
    }

    #[test]
    fn reseeding_gives_dead_latents_something_to_fire_for() {
        let mut s = run(TRY[4].2, 4000);
        let before = s.report.dead;
        assert!(before >= 1, "this setting should leave a dead latent");
        s.reseed_dead();
        s.refresh();
        assert_eq!(s.report.dead, 0, "{} were dead, {} still are", before, s.report.dead);
    }

    #[test]
    fn what_is_shown_is_what_the_autoencoder_does() {
        // the report's numbers against a direct count
        let s = run(TRY[0].2, 800);
        let (m, k) = (s.m(), s.k);
        let (mut f, mut out) = (vec![0.0; k], vec![0.0; m]);
        let mut firing = 0;
        for h in s.eh.chunks_exact(m) {
            s.sae.run(h, &mut f, &mut out);
            firing += f.iter().filter(|v| **v > 0.0).count();
            // the reconstruction is the offset plus the latents along their unit arrows
            for i in 0..m {
                let sum = s.sae.bd[i] + (0..k).map(|j| f[j] * s.sae.decoder(j)[i]).sum::<f32>();
                assert!((sum - out[i]).abs() < 1e-5);
            }
        }
        assert!((s.report.l0 - firing as f32 / EVAL as f32).abs() < 1e-4);
        for j in 0..k {
            assert!((dot(s.sae.decoder(j), s.sae.decoder(j)) - 1.0).abs() < 1e-4);
        }
        assert_eq!(s.report.order.len(), k);
    }

    /// How reliably each setting does what the text says, over ten seeds, and what a
    /// step and a measurement cost (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn survey() {
        let cases: &[(usize, usize, usize, f32, f32, usize)] = &[
            (5, 2, 5, 0.9, 0.3, 1000),
            (5, 2, 5, 0.9, 0.0, 4000),
            (5, 2, 5, 0.9, 1.2, 4000),
            (5, 2, 3, 0.9, 0.3, 4000),
            (5, 2, 10, 0.9, 0.3, 4000),
            (5, 2, 5, 0.3, 0.3, 4000),
            (8, 3, 8, 0.9, 0.3, 4000),
            (16, 8, 16, 0.9, 0.15, 4000),
            (32, 16, 32, 0.95, 0.1, 3000),
            (32, 16, 48, 0.95, 0.1, 3000),
            (32, 16, 32, 0.5, 0.1, 3000),
        ];
        for &(n, m, k, sparsity, lambda, steps) in cases {
            let (mut all, mut recovered, mut unexplained, mut read, mut l0, mut dead) = (0, 0.0, 0.0, 0.0, 0.0, 0.0);
            let start = crate::clock::now();
            for seed in 0..10 {
                let mut rng = Rng::new(100 + seed);
                let features = Features::new(n, m, &mut rng);
                let mut sae = Sae::new(m, k, &mut rng);
                let (mut x, mut h) = (Vec::new(), Vec::new());
                for _ in 0..steps {
                    features.sample(256, sparsity, 0.0, &mut rng, &mut x, &mut h);
                    sae.step(&h, 256, lambda, 0.01);
                }
                features.sample(2048, sparsity, 0.0, &mut rng, &mut x, &mut h);
                let r = Report::of(&sae, &features, &x, &h, spread(&h, m));
                all += (r.recovered == n) as usize;
                recovered += r.recovered as f32 / 10.0;
                unexplained += r.unexplained / 10.0;
                read += r.read / 10.0;
                l0 += r.l0 / 10.0;
                dead += r.dead as f32 / 10.0;
            }
            let per_step = (crate::clock::now() - start) / (10 * steps) as f64 * 1e6;
            println!("n={n} m={m} k={k} S={sparsity} lambda={lambda} steps={steps}: all found in {all}/10  found {recovered:.1}/{n}  unexplained {unexplained:.3}  L0 {l0:.2}  read {read:.2}  dead {dead:.1}   ({per_step:.0} us per step)");
        }
        let mut s = run(setup(3, 32, 48, 0.95, 0.1, true), 100);
        let start = crate::clock::now();
        for _ in 0..200 {
            s.refresh();
        }
        println!("measuring the largest dictionary: {:.0} us", (crate::clock::now() - start) / 200.0 * 1e6);
    }

    /// How "telling" the latents are that point along a feature, before and after
    /// training (run with --ignored --nocapture): what the threshold has to separate.
    #[test]
    #[ignore]
    fn telling_fresh_and_trained() {
        let aligned = |s: &SaeScene| -> Vec<f32> {
            let r = &s.report;
            (0..s.k).filter(|&j| r.nearest[j].1 > RECOVERED && r.rate[j] >= DEAD).map(|j| r.telling[j]).collect()
        };
        let show = |name: &str, s: &SaeScene| {
            let t = aligned(s);
            let (lo, hi) = (t.iter().copied().fold(1.0, f32::min), t.iter().copied().fold(0.0, f32::max));
            let old = s.report.best.iter().filter(|b| b.1 > RECOVERED).count();
            println!("{name:<44} aligned live latents {:>2}: telling {lo:.2}..{hi:.2}   found now {} (by direction alone {old}) of {}", t.len(), s.report.recovered, s.n);
        };
        for (label, _, setup) in TRY {
            for again in 0..3 {
                show(&format!("fresh  {label} #{again}"), &rerun(setup, 0, again));
            }
            let steps = if setup.dims == 3 { 3000 } else { 4000 };
            for again in 0..2 {
                show(&format!("trained {label} #{again}"), &rerun(setup, steps, again));
            }
        }
        for k in [10, 24, 48] {
            for again in 0..3 {
                show(&format!("fresh  5 features, {k} latents #{again}"), &rerun(setup(0, 5, k, 0.9, 0.3, false), 0, again));
                show(&format!("fresh  8 in 3-D, {k} latents #{again}"), &rerun(setup(1, 8, k, 0.9, 0.3, false), 0, again));
            }
        }
        for steps in [20, 50, 100, 200, 400, 800, 1500] {
            show(&format!("default after {steps} steps"), &run(TRY[0].2, steps));
        }
        for (dims, &(n, sparsity, lambda)) in DEFAULTS.iter().enumerate() {
            for again in 0..3 {
                show(&format!("trained default of {} dimensions #{again}", DIMS[dims]), &rerun(setup(dims, n, n, sparsity, lambda, false), 4000, again));
            }
        }
        for again in 0..3 {
            show(&format!("trained too few latents #{again}"), &rerun(TRY[3].2, 4000, again));
            show(&format!("trained 12 features in the plane #{again}"), &rerun(setup(0, 12, 12, 0.9, 0.3, false), 4000, again));
            show(&format!("trained 16 features in 3-D #{again}"), &rerun(setup(1, 16, 16, 0.9, 0.3, false), 4000, again));
        }
    }

    /// "Features found" used to count any direction within 18 degrees of a feature,
    /// which chance provides before a single step: two or three of five by default,
    /// all five with ten latents or more. Now, where features crowd one another (five
    /// or more in the plane, seven or more in space), nothing counts until it is
    /// learned; where they are few and far apart, a fresh latent that points at one
    /// already answers to it alone, and counts (as the note under the numbers says).
    #[test]
    fn nothing_counts_as_found_before_training_where_features_crowd() {
        for setting in [TRY[0].2, TRY[4].2, setup(0, 5, 48, 0.9, 0.3, false), setup(1, 8, 8, 0.9, 0.3, false), setup(1, 8, 48, 0.9, 0.3, false), TRY[5].2] {
            let mut by_direction = 0;
            for again in 0..4 {
                let s = rerun(setting, 0, again);
                by_direction += s.report.best.iter().filter(|b| b.1 > RECOVERED).count();
                assert_eq!(s.report.recovered, 0, "{} latents on {} features in {} dimensions, weights {again}: telling {:?}", s.k, s.n, s.m(), s.report.telling);
            }
            // (and in the plane, chance does point latents at features)
            assert!(DIMS[setting.dims] != 2 || by_direction > 0, "no chance alignments to rule out");
        }
        // every crowded setting, with as many latents as features and with the most
        for (dims, crowded) in [(0, 5), (1, 7)] {
            for n in crowded..=MAX_FEATURES[dims] {
                for k in [n, MAX_LATENTS] {
                    for again in 0..3 {
                        let s = rerun(setup(dims, n, k, 0.9, 0.3, false), 0, again);
                        assert_eq!(s.report.recovered, 0, "{n} features in {} dimensions, {k} fresh latents", DIMS[dims]);
                    }
                }
            }
        }
        // and the few, far apart: fresh latents that point the right way count
        for (dims, few) in [(0, 4), (1, 6)] {
            for n in 2..=few {
                let found: usize = (0..6).map(|again| rerun(setup(dims, n, MAX_LATENTS, 0.9, 0.3, false), 0, again).report.recovered).sum();
                assert!(found > 0, "{n} features in {} dimensions: no fresh latent counts", DIMS[dims]);
            }
        }
        // a dozen features in a plane: every direction is within 18 degrees of one,
        // so direction alone "finds" nearly all of them, trained or not; answering to
        // one's own feature and not the next is what they cannot do
        let crowded = run(setup(0, 12, 12, 0.9, 0.3, false), 3000);
        assert!(crowded.report.best.iter().filter(|b| b.1 > RECOVERED).count() >= 9);
        assert_eq!(crowded.report.recovered, 0);
        // while every default setting, trained, still finds all of its features
        for (dims, &(n, sparsity, lambda)) in DEFAULTS.iter().enumerate() {
            let s = run(setup(dims, n, n, sparsity, lambda, false), 4000);
            assert_eq!(s.report.recovered, n, "{n} features in {} dimensions: telling {:?}", DIMS[dims], s.report.telling);
        }
    }

    /// An experiment repeats: it puts back the learning rate and batch size too,
    /// which its caption was measured at, whatever the sliders were left at.
    #[test]
    fn an_experiment_puts_the_training_settings_back() {
        let mut s = SaeScene::new();
        (s.lr, s.batch, s.noise) = (0.1, 64, 0.2);
        s.train(100);
        s.apply(TRY[0].2);
        assert_eq!((s.lr, s.batch, s.noise), (LR, BATCH, 0.0));
        s.train(500);
        let fresh = run(TRY[0].2, 500);
        assert_eq!(s.sae.dec, fresh.sae.dec);
    }

    /// Where the samples take turns (no picture), pausing keeps the one on show, and
    /// training on starts from it again.
    #[test]
    fn pausing_keeps_the_sample_on_show() {
        let mut s = run(TRY[5].2, 10);
        s.turning = 7.6;
        let shown = s.chosen(None);
        assert_ne!(shown, s.cloud[0], "(the turns have moved on from the first)");
        s.running = false;
        assert_eq!(s.chosen(None), shown);
        s.running = true;
        assert_eq!(s.chosen(None), shown);
    }

    /// The dots are what the autoencoder is given: the activations that hold no
    /// feature (only noise) are drawn too, in their true share.
    #[test]
    fn the_dots_include_the_empty_activations_in_their_true_share() {
        let mut s = run(TRY[0].2, 1);
        s.noise = 0.2;
        s.resample();
        let mut drawn: Vec<usize> = s.cloud.iter().chain(&s.quiet).copied().collect();
        drawn.sort();
        assert!(drawn.iter().enumerate().all(|(i, &d)| i == d), "the dots are the first samples, in order");
        let share = s.quiet.len() as f32 / drawn.len() as f32;
        // (none of five features, each absent nine times in ten)
        let empty = 0.9f32.powi(5);
        assert!((share - empty).abs() < 0.04, "{share} of the dots empty, against {empty}");
        assert!(s.quiet.iter().all(|&q| s.ex[q * s.n..(q + 1) * s.n].iter().all(|x| *x == 0.0)));
        assert!(s.cloud.iter().all(|&c| s.ex[c * s.n..(c + 1) * s.n].iter().any(|x| *x > 0.0)));
    }

    /// A re-seeded latent's optimiser starts afresh: its first step is the learning
    /// rate, as at the very start (it used to be three times that, and some ten steps
    /// on six times), so that at a high rate it settles rather than swinging about and
    /// dying again.
    #[test]
    fn a_re_seeded_latent_starts_with_small_steps_and_lives() {
        let mut s = run(TRY[4].2, 4000);
        let dead: Vec<usize> = (0..s.k).filter(|&j| s.report.rate[j] < DEAD).collect();
        s.reseed_dead();
        let (m, j) = (s.m(), dead[0]);
        let (enc, be) = (s.sae.encoder(j).to_vec(), s.sae.be[j]);
        s.train(1);
        let moved = s.sae.encoder(j).iter().zip(&enc).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(moved <= 1.01 * s.lr && moved > 0.9 * s.lr, "the encoder moved {moved} at learning rate {}", s.lr);
        assert!(((s.sae.be[j] - be).abs() - s.lr).abs() < 0.01 * s.lr, "its bias moved {}", s.sae.be[j] - be);
        assert_eq!(m, 2);
        // at three times the usual rate, the re-seeded latents are all alive 200 steps
        // on (they all died again within 50 when the first steps were too large)
        for again in 0..3 {
            let mut s = rerun(TRY[4].2, 3000, again);
            s.lr = 0.03;
            s.train(500);
            s.refresh();
            let dead: Vec<usize> = (0..s.k).filter(|&j| s.report.rate[j] < DEAD).collect();
            assert!(!dead.is_empty());
            s.reseed_dead();
            s.train(200);
            s.refresh();
            assert!(dead.iter().all(|&j| s.report.rate[j] >= DEAD), "weights {again}: re-seeded {dead:?}, rates {:?}", s.report.rate);
        }
    }

    /// The charts' histories stand for the whole run, evenly: as many steps to every point.
    #[test]
    fn the_histories_cover_the_run_evenly() {
        let s = run(TRY[0].2, 2500);
        for history in &s.history {
            let (points, every) = (history.points().len() as u64, history.every() as u64);
            assert!(points < 600 && every > 1, "{points} points of {every} steps");
            assert!((points * every).abs_diff(s.steps) < every, "{points} points of {every} steps for {} steps", s.steps);
        }
    }

    /// The panels keep below the strip the application's buttons take, and a narrow
    /// view that is wider than tall is not stacked (it has no height for that).
    #[test]
    fn the_layout_keeps_clear_of_the_top_strip_and_fits_short_views() {
        assert!(stacked(Vec2::new(390.0, 800.0)) && !stacked(Vec2::new(600.0, 400.0)) && !stacked(Vec2::new(1500.0, 940.0)) && !stacked(Vec2::new(768.0, 1024.0)));
        for dims in [0usize, 3] {
            let (n, sparsity, lambda) = DEFAULTS[dims];
            let s = run(setup(dims, n, n, sparsity, lambda, false), 1);
            for view in [Vec2::new(1500.0, 940.0), Vec2::new(978.0, 940.0), Vec2::new(768.0, 1024.0), Vec2::new(600.0, 400.0), Vec2::new(844.0, 390.0), Vec2::new(390.0, 800.0)] {
                for top in [0.0, 46.0] {
                    let lay = s.layout(view, top);
                    let what = format!("{} dimensions, view {view}, strip {top}", DIMS[dims]);
                    // the legend, and the backings of the matrix and of the scatter
                    assert!(lay.key.y >= top, "{what}: legend at {}", lay.key.y);
                    assert!(lay.matrix.y - 62.0 >= top, "{what}: matrix panel from {}", lay.matrix.y - 62.0);
                    if lay.side > 0.0 {
                        assert!(lay.scatter.y - 54.0 >= top, "{what}: scatter panel from {}", lay.scatter.y - 54.0);
                        assert!(lay.scatter.y + lay.side <= view.y, "{what}: scatter down to {}", lay.scatter.y + lay.side);
                    }
                    if lay.apart_size.y > 0.0 {
                        assert!(lay.apart.y - 40.0 >= top && lay.apart.y + lay.apart_size.y + 36.0 <= view.y, "{what}: \"taken apart\" from {} to {}", lay.apart.y - 40.0, lay.apart.y + lay.apart_size.y + 36.0);
                    }
                    // the charts are inside the view, and the matrix has cells worth the name
                    assert!(lay.charts.iter().all(|c| c.y >= top && c.y + lay.chart.y <= view.y), "{what}: charts at {:?}", lay.charts);
                    // (five rows can be large in any of these views; 32 rows in a view
                    // 400 high cannot, and are only asked to fit, below)
                    assert!(dims != 0 || lay.cell >= 10.0, "{what}: cells of {} px", lay.cell);
                    // the matrix, with the lines of text under it, ends above the charts' titles
                    let foot = lay.matrix.y + s.rows() * lay.cell + 58.0;
                    let ceiling = if lay.stacked || dims == 0 { view.y } else { lay.charts[0].y - 26.0 };
                    assert!(foot <= ceiling, "{what}: matrix panel down to {foot}, room to {ceiling}");
                }
            }
        }
    }

    /// The weights the picture draws read a sample as the autoencoder does, are those
    /// before the newest step at its start and the autoencoder's own at its end, and are
    /// the autoencoder's own once paused.
    #[test]
    fn the_weights_drawn_are_the_autoencoders_between_two_steps() {
        let mut s = run(TRY[0].2, 300);
        let (m, k) = (s.m(), s.k);
        let mut before = Drawn::default();
        before.keep(&s.sae);
        s.train(1);
        let mut drawn = Drawn::default();
        for sample in [0, 7, 99] {
            let h = &s.eh[sample * m..(sample + 1) * m];
            drawn.between(&before, &s.sae, 1.0);
            let (mut fired, mut out) = (vec![0.0; k], vec![0.0; m]);
            s.sae.run(h, &mut fired, &mut out);
            assert_eq!(drawn.fired(h), fired);
            drawn.between(&before, &s.sae, 0.0);
            assert!(drawn.dec == before.dec && drawn.enc == before.enc && drawn.be == before.be && drawn.bd == before.bd);
            assert!(drawn.dec != s.sae.dec);
        }
        // a frame at 60 frames a second takes four steps, and the picture is one behind
        let mut kept = Drawn::default();
        for _ in 0..4 {
            kept.keep(&s.sae);
            s.train(1);
        }
        let owed = (s.rate as f32 / 60.0 - 4.0).max(0.0);
        drawn.between(&kept, &s.sae, owed.min(1.0));
        assert!(owed < 1e-3 && drawn.dec == kept.dec);
    }

    /// Two latents that fire about equally keep their places in the sample taken
    /// apart, however the noise of training tips them; one that comes to fire clearly
    /// more strongly moves ahead; one that begins to fire joins after the rest, and
    /// moves up only if it is clearly stronger.
    #[test]
    fn latents_keep_their_places_until_one_clearly_overtakes() {
        assert_eq!(laid(&[], &[0.0, 0.5, 0.2, 0.7]), [3, 1, 2]);
        let mut order = laid(&[], &[0.06, 0.07, 0.8]);
        assert_eq!(order, [2, 1, 0]);
        for (a, b) in [(0.07, 0.06), (0.065, 0.06), (0.05, 0.06), (0.074, 0.06)] {
            order = laid(&order, &[a, b, 0.8]);
            assert_eq!(order, [2, 1, 0], "{a} {b}");
        }
        assert_eq!(laid(&order, &[0.08, 0.06, 0.8]), [2, 0, 1]);
        assert_eq!(laid(&[2, 1], &[0.0, 0.3, 0.8, 0.31]), [2, 1, 3]);
        assert_eq!(laid(&[2, 1], &[0.0, 0.3, 0.8, 0.5]), [2, 3, 1]);
        assert_eq!(laid(&[2, 1], &[0.0, 0.3, 0.8, 0.9]), [2, 3, 1]);
        assert_eq!(laid(&[2, 1], &[0.0, 0.3, 0.8, 1.1]), [3, 2, 1]);
        assert_eq!(laid(&[2, 1], &[0.0, 0.0, 0.0, 0.0]), Vec::<usize>::new());
    }
}

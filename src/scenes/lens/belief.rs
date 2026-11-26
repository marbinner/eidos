//! The Jacobian lens on a belief filter: the toy model of Ostrow, Qian, Liu and Fiete,
//! "A Mathematical Theory of the Jacobian Lens" (2026), running live, with the lens's
//! three factors drawn as what they are: lengths of chains of arrows.
//!
//! Everything is drawn in screen space (logical pixels), laid out afresh for the
//! size of the view, so the same panels serve a wide window and a phone.

use super::page::{width, Rect};
use super::ring::{self, Chain, Direction, Factors, Filter, Latent, Measured, Nudge, Probe, Process};
use crate::math::complex::C32;
use crate::scenes::prelude::*;
use std::collections::VecDeque;

pub const DEF: SceneDef = SceneDef {
    id: "belief",
    title: "The Jacobian lens on a belief filter",
    group: super::GROUP,
    blurb: "Hidden dials, a filter that tracks them, and the three numbers that decide what a Jacobian lens sees.",
    make: |_| Box::new(Belief::new()),
};

/// The most dials there can be.
const DIALS: usize = 3;
const RING: [Color; DIALS] = [pal::CYAN, pal::YELLOW, pal::PINK];
const NAME: [&str; DIALS] = ["A", "B", "C"];
const SIZES: [usize; 4] = [2, 4, 8, 16];
const DARK: Color = Color::hex(0x121724);
/// A cell of the token strip in which nothing was shown.
const CELL: Color = Color::hex(0x19202F);

/// Tokens kept for the strip.
const STRIP: usize = 80;
/// The measuring run: tokens a second, taken in batches (so that the clock is not
/// read for every token) within at most `LAB_MOST` seconds of a frame; the tokens a
/// frame in reproducible renders instead (the same at sixty frames a second); and the
/// number of nudges after which it stops.
const LAB_RATE: f32 = 24_000.0;
const LAB_BATCH: usize = 50;
const LAB_MOST: f64 = 0.002;
const LAB_TOKENS: usize = 400;
const LAB_CONTEXTS: u64 = 40_000;
/// Points on the curves against ω and against λ.
const CURVE: usize = 96;

/// A setting of the whole scene.
struct Preset {
    label: &'static str,
    look_for: &'static str,
    /// Sites round each dial, and the diffusion q.
    sites: usize,
    q: f32,
    /// Each dial's (persistence λ, velocity v, key bias ψ).
    dials: &'static [(f32, usize, Option<f32>)],
    /// The dial to study, and the lens's tuning ω as a fraction of π.
    study: usize,
    omega: f32,
}

/// The paper's experiments, with its parameters (appendix E), and one more for the
/// tuned lens. What each shows is checked by the tests.
const PRESETS: [Preset; 5] = [
    Preset {
        label: "memory: the magnitude sweep",
        look_for: r"Three dials that stand still but forget at different rates ($\lambda = 0.8, 0.9, 0.95$). Every chain is straight; only its length differs. The lens lists the longest memory first: C, B, A.",
        sites: 8,
        q: 0.25,
        dials: &[(0.8, 0, None), (0.9, 0, None), (0.95, 0, None)],
        study: 2,
        omega: 0.0,
    },
    Preset {
        label: "turning: the lag sweep",
        look_for: "Same memory, different speeds: A flips half a turn every token, B makes a quarter turn, C stands still. A turning dial's chain curls up, so little of its length survives. Listed: C, B, A.",
        sites: 8,
        q: 0.25,
        dials: &[(0.9, 4, None), (0.9, 2, None), (0.9, 0, None)],
        study: 1,
        omega: 0.0,
    },
    Preset {
        label: "context: the key sweep",
        look_for: r"Dials with two sites, each shown true or reversed depending on a key ($\psi = 0, 0.75, 1$). Reversed glimpses give reversed chains, which cancel in the average. With a fair-coin key (A) nothing survives at all.",
        sites: 2,
        q: 0.0,
        dials: &[(0.95, 0, Some(0.0)), (0.95, 0, Some(0.75)), (0.95, 0, Some(1.0))],
        study: 1,
        omega: 0.0,
    },
    Preset {
        label: "the crossing",
        look_for: r"Two dials: A stands still, B turns a quarter turn and remembers well ($\lambda = 0.95$). Raise dial A's persistence past $\lambda^* \approx 0.65$ (bottom right) and watch A overtake B in the list.",
        sites: 8,
        q: 0.25,
        dials: &[(0.55, 0, None), (0.95, 2, None)],
        study: 0,
        omega: 0.0,
    },
    Preset {
        label: "tune the lens to fast things",
        look_for: r"The turning dials again, but with the lens tuned to $\omega = \pi$: it now adds up responses that flip every token and cancels the steady ones. The order reverses: A, B, C.",
        sites: 8,
        q: 0.25,
        dials: &[(0.9, 4, None), (0.9, 2, None), (0.9, 0, None)],
        study: 0,
        omega: 1.0,
    },
];

/// How wide a line of this scene's labels comes out when drawn, near enough. (The
/// toolkit's `text_width` allows for wider letters than they have: measured,
/// "dial A:  λ 0.90,  turns 180° a token" at 11.5 pixels is drawn 167 pixels wide,
/// against an estimate of 195.) A line with formulas in it is measured as it is drawn,
/// which costs little for a line measured in every frame.
fn drawn_width(f: &Frame, text: &str, size: f32) -> f32 {
    if crate::typeset::has_math(text) {
        f.text_size(text, size).x
    } else {
        0.87 * text_width(text, size)
    }
}

/// A turn in whole degrees, half a turn being +180 whichever way rounding has it.
fn degrees(angle: f32) -> f32 {
    let d = angle.to_degrees().round();
    if d <= -180.0 {
        180.0
    } else {
        d
    }
}

/// The panels, in the order they are drawn (`TOKENS`: the strip of tokens).
const RINGS: usize = 0;
const TOKENS: usize = 1;
const CHAIN: usize = 2;
const CLOUD: usize = 3;
const BARS: usize = 4;
const TUNING: usize = 5;
const SWEEP: usize = 6;
const PANELS: usize = 7;

/// The least size of each panel at which everything in it is drawn as it should be,
/// worked out from what its `draw_` function puts where.
const LEAST: [Vec2; PANELS] = [
    // the heading, rings 28 pixels across, and their names under them
    Vec2::new(200.0, 100.0),
    // the heading and three rows of cells 8 pixels high
    Vec2::new(200.0, 56.0),
    // 104 pixels of words above and below a picture 40 high (the cloud's take 96)
    Vec2::new(140.0, 144.0),
    Vec2::new(140.0, 144.0),
    // 116 pixels of words around bars 34 high
    Vec2::new(200.0, 150.0),
    // 82 pixels of words and axes around a chart 40 high
    Vec2::new(140.0, 122.0),
    Vec2::new(140.0, 122.0),
];

/// Which panels are kept first when there is not room for all of them: the dials,
/// the factors and their product, then the chains that explain them, then the rest.
const KEEP: [usize; PANELS] = [RINGS, BARS, CHAIN, CLOUD, TOKENS, TUNING, SWEEP];

/// The room under the panels for the line that says some were left out.
const NOTE: f32 = 10.0;

/// Where each panel goes for a view of a given size: nowhere, for a panel there is
/// no room for. (Drawing and clicking both go by this, so they always agree.)
struct Layout {
    panels: [Option<Rect>; PANELS],
    /// How many panels were left out, and the top-left corner of the line that says so.
    left_out: usize,
    note: Vec2,
}

/// A row of panels side by side: which, how wide each is, and the row's share of
/// the height in [`Layout::shared`].
type Row = (Vec<usize>, Vec<f32>, f32);

impl Layout {
    /// `inset`: how much of the top of the view the application's buttons cover.
    fn new(view: Vec2, inset: f32) -> Self {
        let (m, gap, top) = (10.0, 8.0, inset.max(40.0));
        let (w, room) = (view.x - 2.0 * m, view.y - top - 12.0);
        let half = (w - gap) * 0.5;
        let shared = Self::shared(view, top, room);
        if shared.iter().zip(LEAST).all(|(r, least)| r.w() >= least.x && r.h() >= least.y) {
            return Layout { panels: shared.map(Some), left_out: 0, note: Vec2::ZERO };
        }
        // Too short (or too narrow) for that: the same rows, each given at least
        // what its panels need, and as many of them as fit, the most important first.
        let side = ((w - 2.0 * gap) * 0.28).floor();
        let rows: Vec<Row> = if view.x >= 560.0 {
            vec![
                (vec![RINGS], vec![w], 0.2475),
                (vec![TOKENS], vec![w], 0.1275),
                (vec![CHAIN, CLOUD, BARS], vec![side, side, w - 2.0 * (side + gap)], 0.34),
                (vec![TUNING, SWEEP], vec![half, half], 0.285),
            ]
        } else if half >= LEAST[CHAIN].x {
            vec![
                (vec![RINGS], vec![w], 0.19),
                (vec![TOKENS], vec![w], 0.105),
                (vec![CHAIN, CLOUD], vec![half, half], 0.225),
                (vec![BARS], vec![w], 0.25),
                (vec![TUNING, SWEEP], vec![half, half], 0.23),
            ]
        } else {
            // too narrow for two abreast: one panel to a row
            [0.19, 0.105, 0.225, 0.225, 0.25, 0.23, 0.23].into_iter().enumerate().map(|(p, share)| (vec![p], vec![w], share)).collect()
        };
        let least = |row: &Row| row.0.iter().map(|&p| LEAST[p].y).fold(0.0, f32::max);
        let wide_enough = |row: &Row| row.0.iter().zip(&row.1).all(|(&p, &width)| width >= LEAST[p].x);
        let all = rows.iter().all(wide_enough) && rows.iter().map(|row| least(row) + gap).sum::<f32>() - gap <= room;
        let room = if all { room } else { room - NOTE };
        let mut kept = vec![false; rows.len()];
        let mut used = -gap;
        for p in KEEP {
            if let Some(k) = rows.iter().position(|row| row.0.contains(&p)) {
                if !kept[k] && wide_enough(&rows[k]) && used + gap + least(&rows[k]) <= room {
                    kept[k] = true;
                    used += gap + least(&rows[k]);
                }
            }
        }
        let shown: Vec<&Row> = rows.iter().zip(&kept).filter(|r| *r.1).map(|r| r.0).collect();
        // Share out the height as the layout above would, except that a row given
        // less than it needs is given just that, and the rest shared among the others.
        let free = room - gap * shown.len().saturating_sub(1) as f32;
        let mut pinned = vec![false; shown.len()];
        let mut height = vec![0.0f32; shown.len()];
        loop {
            let left = free - shown.iter().zip(&pinned).filter(|r| *r.1).map(|r| least(r.0)).sum::<f32>();
            let shares: f32 = shown.iter().zip(&pinned).filter(|r| !*r.1).map(|r| r.0 .2).sum();
            let mut again = false;
            for (s, row) in shown.iter().enumerate() {
                height[s] = if pinned[s] { least(row) } else { (left * row.2 / shares).floor() };
                if height[s] < least(row) {
                    (pinned[s], again) = (true, true);
                }
            }
            if !again {
                break;
            }
        }
        let mut panels = [None; PANELS];
        let mut y = top;
        for (row, h) in shown.iter().zip(height) {
            let mut x = m;
            for (&p, &width) in row.0.iter().zip(&row.1) {
                panels[p] = Some(Rect::sized(x, y, width, h));
                x += width + gap;
            }
            y += h + gap;
        }
        let left_out = panels.iter().filter(|p| p.is_none()).count();
        let note = Vec2::new(m, if shown.is_empty() { top } else { y - gap + 5.0 });
        Layout { panels, left_out, note }
    }

    /// The layout that serves most windows: each row a fixed share of the height.
    fn shared(view: Vec2, top: f32, h: f32) -> [Rect; PANELS] {
        let (m, gap) = (10.0, 8.0);
        let w = view.x - 2.0 * m;
        let half = (w - gap) * 0.5;
        if view.x >= 560.0 {
            // three rows: the process; one nudge, many nudges and the factors; two sweeps
            let (a, b) = ((0.375 * h).floor(), (0.34 * h).floor());
            let c = h - a - b - 2.0 * gap;
            let rings = (a * 0.66).floor();
            let side = ((w - 2.0 * gap) * 0.28).floor();
            let (yb, yc) = (top + a + gap, top + a + b + 2.0 * gap);
            [
                Rect::sized(m, top, w, rings),
                Rect::sized(m, top + rings + gap, w, a - rings - gap),
                Rect::sized(m, yb, side, b),
                Rect::sized(m + side + gap, yb, side, b),
                Rect::sized(m + 2.0 * (side + gap), yb, w - 2.0 * (side + gap), b),
                Rect::sized(m, yc, half, c),
                Rect::sized(m + half + gap, yc, half, c),
            ]
        } else {
            // a narrow screen: the same panels in a single column, two abreast where
            // they are a pair
            let share = [0.19, 0.105, 0.225, 0.25];
            let hs: Vec<f32> = share.iter().map(|s| (s * h).floor()).collect();
            let rest = h - hs.iter().sum::<f32>() - 4.0 * gap;
            let mut y = top;
            let mut next = |height: f32| {
                let at = y;
                y += height + gap;
                at
            };
            let (y0, y1, y2, y3, y4) = (next(hs[0]), next(hs[1]), next(hs[2]), next(hs[3]), next(rest));
            [
                Rect::sized(m, y0, w, hs[0]),
                Rect::sized(m, y1, w, hs[1]),
                Rect::sized(m, y2, half, hs[2]),
                Rect::sized(m + half + gap, y2, half, hs[2]),
                Rect::sized(m, y3, w, hs[3]),
                Rect::sized(m, y4, half, rest),
                Rect::sized(m + half + gap, y4, half, rest),
            ]
        }
    }

    /// The line that says how many panels were left out, as long as `room` pixels allow.
    fn note_text(&self, view: Vec2, room: f32) -> String {
        let n = self.left_out;
        let panels = if n == 1 { "1 more panel needs".to_string() } else { format!("{n} more panels need") };
        let upright = if view.x > view.y { ", or turn the phone upright" } else { "" };
        let forms = [format!("{panels} a taller window: try full screen{upright}"), format!("{panels} a taller window"), format!("{n} more in a taller window")];
        forms.iter().find(|t| text_width(t, 11.0) <= room).unwrap_or(&forms[2]).clone()
    }
}

/// One token of the run that is shown.
#[derive(Clone, Copy)]
struct Tick {
    revealed: bool,
    /// The first Fourier mode of each ring's forecast after this token.
    m: [C32; DIALS],
}

/// The run that is shown as it stood before its newest token: what the picture draws
/// gliding to the run as it stands now (see [`Belief::glided`]).
struct Before {
    /// Where each dial was, the filter's forecast for it and that forecast's first mode.
    x: Vec<usize>,
    forecast: Vec<Vec<f32>>,
    mode: Vec<C32>,
    /// How many links each dial's chain of responses had, and whether a key had
    /// reversed it.
    links: Vec<usize>,
    reversed: [bool; DIALS],
}

/// The share of the time between two tokens over which the picture glides from the
/// run as it stood before the newest token to the run as it stands now.
const GLIDE_SHARE: f32 = 0.5;

/// The partial sums of the responses to one nudge, lag by lag, and how likely the
/// chain is to end at each (a reveal wipes the nudge out).
struct Sums {
    /// `points[k]` is the sum of the responses at lags 1 to k (so `points[0]` is zero).
    points: Vec<C32>,
    /// The chance that the chain ends at `points[k]`.
    chance: Vec<f32>,
    /// The share of contexts in which the key reverses the whole chain.
    reversed: f32,
    /// How many links to draw: all but the rarest few chains end within them. (A dial
    /// with a long memory could go on many times further, but hardly ever does.)
    drawn: usize,
}

struct Belief {
    process: Process,
    /// Lags added up by the lens (the paper's H).
    window: usize,
    /// The frequency the lens is tuned to (0 for the plain lens).
    omega: f32,
    selected: usize,
    rate: f32,
    running: bool,

    // the run that is shown
    rng: Rng,
    chain: Chain,
    filter: Filter,
    ticks: VecDeque<Tick>,
    /// Tokens since the last reveal (none before the first). Counted, not read off
    /// the strip, which a long wait between reveals outruns.
    since: Option<usize>,
    /// For each ring, the nudge given at its last reveal (if there has been one),
    /// whether the key reversed it, and the filter's responses at lags 1, 2, ...
    nudges: Vec<Option<Nudge>>,
    reversed: [bool; DIALS],
    responses: Vec<Vec<C32>>,
    /// How far the run is from its newest token to the next, from 0 to 1.
    clock: f32,
    /// The run as it stood before its newest token. The run (`chain`, `filter`,
    /// `responses`) and every number and word are the run as it stands; the dials,
    /// their forecasts and the ends of the chain are drawn gliding to it from this
    /// over the first part of the time to the next token.
    before: Option<Before>,

    // the measuring run: the same process and filter at full speed, nudged at every reveal
    lab_rng: Rng,
    lab_chain: Chain,
    lab_filter: Filter,
    probes: Vec<Probe>,
    measured: Vec<Measured>,
    /// Tokens of the measuring run, and the surprise (minus the logarithm of the
    /// probability given to the token that came) added up over them: for the filter
    /// and for a predictor with no memory.
    lab_tokens: u64,
    surprise: [f64; 2],
    lab_pace: crate::clock::Pace,

    // worked out from the settings whenever they change
    stale: bool,
    /// The three factors of each ring's first Fourier mode: its position, the arrow
    /// of the picture, and the mode the paper compares latents on. Everything drawn
    /// (bars, chains, the list, the sweep and its crossings) is about this mode.
    theory: Vec<Factors>,
    /// The rings whose strongest direction in the lens is a finer pattern than that
    /// (two or more waves round the ring), with that direction.
    finer: Vec<Direction>,
    /// Each ring's lens weight against the tuning ω, from 0 to π.
    tuned: Vec<Vec<f32>>,
    /// The studied ring's lens weight against its own persistence, at λ = 0,
    /// 1/`CURVE`, ..., 1. (Where it draws level with another ring is in `switches`.)
    sweep: Vec<f32>,
    /// Where the studied dial's weight passes another dial's as its λ grows: the other
    /// dial, the λ, and whether it rises through (overtakes) or falls back below.
    switches: Vec<(usize, f32, bool)>,
    /// Each dial's weight through an endless window under the plain lens: what the
    /// paper's closed forms are about.
    endless: Vec<f32>,
    sums: Sums,
}

impl Belief {
    fn new() -> Self {
        Self::with(1)
    }

    fn process_of(preset: &Preset) -> Process {
        Process { latents: preset.dials.iter().map(|&(lambda, v, psi)| Latent { n: preset.sites, lambda, v, psi }).collect(), q: preset.q, epsilon: 0.2 }
    }

    /// The scene set up as one of the presets.
    fn with(preset: usize) -> Self {
        let process = Self::process_of(&PRESETS[preset]);
        let mut rng = Rng::new(1);
        let chain = Chain::new(&process, &mut rng);
        let mut scene = Belief {
            filter: Filter::new(&process),
            lab_filter: Filter::new(&process),
            lab_tokens: 0,
            surprise: [0.0; 2],
            lab_pace: crate::clock::Pace::default(),
            lab_chain: chain.clone(),
            chain,
            process,
            window: 128,
            omega: 0.0,
            selected: 0,
            rate: 3.0,
            running: true,
            rng: rng.clone(),
            ticks: VecDeque::new(),
            since: None,
            nudges: Vec::new(),
            reversed: [false; DIALS],
            responses: Vec::new(),
            clock: 0.0,
            before: None,
            lab_rng: rng,
            probes: Vec::new(),
            measured: Vec::new(),
            stale: true,
            theory: Vec::new(),
            finer: Vec::new(),
            tuned: Vec::new(),
            sweep: Vec::new(),
            switches: Vec::new(),
            endless: Vec::new(),
            sums: Sums { points: Vec::new(), chance: Vec::new(), reversed: 0.0, drawn: 0 },
        };
        scene.apply(preset);
        scene
    }

    fn rings(&self) -> usize {
        self.process.latents.len()
    }

    fn apply(&mut self, preset: usize) {
        let preset = &PRESETS[preset];
        self.process = Self::process_of(preset);
        self.window = 128;
        self.omega = preset.omega * PI;
        self.selected = preset.study;
        self.restart();
    }

    /// Begin both runs again (after any change to the process).
    fn restart(&mut self) {
        let k = self.rings();
        self.selected = self.selected.min(k - 1);
        self.rng = Rng::new(20);
        self.chain = Chain::new(&self.process, &mut self.rng);
        self.filter = Filter::new(&self.process);
        self.ticks.clear();
        self.since = None;
        self.nudges = vec![None; k];
        self.reversed = [false; DIALS];
        self.responses = vec![Vec::new(); k];
        // (start with a strip's worth of history rather than an empty picture)
        for _ in 0..STRIP {
            self.tick();
        }
        self.before = None;
        self.remeasure();
    }

    /// Begin the measuring run again (after a change to the process).
    fn remeasure(&mut self) {
        self.lab_rng = Rng::new(21);
        self.lab_chain = Chain::new(&self.process, &mut self.lab_rng);
        self.lab_filter = Filter::new(&self.process);
        self.lab_tokens = 0;
        self.surprise = [0.0; 2];
        self.relens();
    }

    /// Measure the lens afresh (after a change to the lens alone: the run of tokens
    /// and how well the filter predicts them are no different for it).
    fn relens(&mut self) {
        let k = self.rings();
        self.probes = vec![Probe::default(); k];
        self.measured = vec![Measured::default(); k];
        self.lab_pace.reset();
        self.stale = true;
    }

    /// The tuning as ring `i`'s first mode feels it (see [`Process::tuned`]).
    fn tuning(&self, i: usize) -> f32 {
        self.process.tuned(i, 1, self.omega)
    }

    /// One token of the run that is shown.
    fn tick(&mut self) {
        let k = self.rings();
        let mut before = [false; DIALS];
        before[..k].copy_from_slice(&self.chain.minus);
        let (token, revealed) = self.chain.emit(&self.process, &mut self.rng);
        self.filter.observe(&self.process, token);
        self.since = if revealed { Some(0) } else { self.since.map(|n| n + 1) };
        let mut tick = Tick { revealed, m: [C32::ZERO; DIALS] };
        for (i, l) in self.process.latents.iter().enumerate() {
            tick.m[i] = self.filter.mode(i, 1);
            if revealed {
                self.nudges[i] = Some(Nudge::start(l, self.process.q, 1, before[i]));
                self.reversed[i] = before[i] && l.psi.is_some();
                self.responses[i].clear();
            } else if let Some(nudge) = &mut self.nudges[i] {
                nudge.advance(l, self.process.q);
                if self.responses[i].len() < 400 {
                    self.responses[i].push(nudge.response());
                }
            }
        }
        self.ticks.push_back(tick);
        if self.ticks.len() > STRIP {
            self.ticks.pop_front();
        }
    }

    /// The run as it stands, to glide from once the next token has come.
    fn before_next(&self) -> Before {
        let k = self.rings();
        Before {
            x: self.chain.x.clone(),
            forecast: (0..k).map(|i| self.filter.forecast(i).to_vec()).collect(),
            mode: (0..k).map(|i| self.filter.mode(i, 1)).collect(),
            links: (0..k).map(|i| self.live_len(i)).collect(),
            reversed: self.reversed,
        }
    }

    /// How far the picture has glided from the run before its newest token to the run
    /// now: 0 as the token comes, 1 a share `GLIDE_SHARE` of the way to the next, eased
    /// in and out. (Drawn as they came, a token every third of a second moved the dials
    /// and the ends of the chains at once, and nothing in between.) Paused, the picture
    /// is the run as it stands.
    fn glided(&self) -> Option<(&Before, f32)> {
        let t = (self.clock / GLIDE_SHARE).clamp(0.0, 1.0);
        self.before.as_ref().filter(|_| self.running && t < 1.0).map(|b| (b, t * t * (3.0 - 2.0 * t)))
    }

    /// Where dial `i` is drawn, as an angle round its ring: where it is, or on its way
    /// there the shorter way round from where it was before the newest token.
    fn dial_shown(&self, i: usize) -> f32 {
        let turn = |x: usize| TAU * x as f32 / self.process.latents[i].n as f32;
        match self.glided().filter(|(b, _)| b.x.len() == self.chain.x.len()) {
            Some((b, g)) => {
                let (from, to) = (turn(b.x[i]), turn(self.chain.x[i]));
                from + g * (PI - (PI - (to - from)).rem_euclid(TAU))
            }
            None => turn(self.chain.x[i]),
        }
    }

    /// The tokens that `dt` seconds of the scene's time ask for (at most four in a
    /// frame), noting each time how the run stood before.
    fn advance(&mut self, dt: f32) {
        self.clock += dt * self.rate;
        let mut done = 0;
        while self.clock >= 1.0 && done < 4 {
            self.clock -= 1.0;
            self.before = Some(self.before_next());
            self.tick();
            done += 1;
        }
        self.clock = self.clock.min(1.0);
    }

    /// The end of the chain being built for ring `i` as it is drawn: the link it has
    /// got to, or part way to it from the one before as a new token adds a link. (A
    /// reveal, which starts the chain afresh, is drawn at once.) Returns the link
    /// before, the link now, and how far between them.
    fn live_end(&self, i: usize) -> (usize, usize, f32) {
        let now = self.live_len(i).min(self.sums.drawn);
        match self.glided() {
            Some((b, g)) if b.links[i] + 1 == self.live_len(i) && b.reversed[i] == self.reversed[i] => (b.links[i].min(self.sums.drawn), now, g),
            _ => (now, now, 1.0),
        }
    }

    /// Some more tokens of the measuring run.
    fn measure(&mut self, tokens: usize) {
        let k = self.rings();
        for _ in 0..tokens {
            let mut before = [false; DIALS];
            before[..k].copy_from_slice(&self.lab_chain.minus);
            let (token, revealed) = self.lab_chain.emit(&self.process, &mut self.lab_rng);
            self.surprise[0] -= (self.lab_filter.chance_of(&self.process, token).max(1e-30) as f64).ln();
            self.surprise[1] -= (self.process.memoryless_chance(token) as f64).ln();
            self.lab_tokens += 1;
            self.lab_filter.observe(&self.process, token);
            for i in 0..k {
                let (l, tuned) = (self.process.latents[i], self.tuning(i));
                self.probes[i].token(&l, self.process.q, 1, revealed, before[i], self.window, tuned, &mut self.measured[i]);
            }
        }
    }

    /// A frame's share of the measuring run, `real_dt` seconds after the last, until
    /// it has seen enough: so many tokens a second, whatever the display's frame rate.
    fn lab_frame(&mut self, real_dt: f32) {
        if self.measured[0].contexts >= LAB_CONTEXTS {
            return;
        }
        if crate::clock::reproducible() {
            self.measure(LAB_TOKENS);
            return;
        }
        let mut turn = self.lab_pace.begin(LAB_RATE / LAB_BATCH as f32, real_dt, real_dt, LAB_MOST);
        while turn.next() {
            self.measure(LAB_BATCH);
        }
        self.lab_pace.end(turn);
    }

    /// Everything that follows from the settings alone.
    fn refresh(&mut self) {
        self.stale = false;
        let (k, window) = (self.rings(), Some(self.window));
        self.theory = (0..k).map(|i| self.process.lens(i, 1, window, self.omega)).collect();
        // a dial's finer patterns can outrank its first mode in the lens: note which
        let strongest = self.process.spectrum(window, self.omega);
        self.finer = (0..k).filter_map(|i| strongest.iter().find(|d| d.latent == i).filter(|d| d.mode != 1 && d.sigma > 1.001 * self.theory[i].sigma()).copied()).collect();
        self.tuned = (0..k).map(|i| (0..=CURVE).map(|s| self.process.lens(i, 1, window, PI * s as f32 / CURVE as f32).sigma()).collect()).collect();
        self.sweep = self.process.sweep(self.selected, CURVE, window, self.omega);
        self.switches = (0..k)
            .filter(|&o| o != self.selected)
            .flat_map(|o| self.process.crossings(self.selected, &self.sweep, self.theory[o].sigma(), window, self.omega).into_iter().map(move |(at, rising)| (o, at, rising)))
            .collect();
        self.endless = (0..k).map(|i| self.process.lens(i, 1, None, 0.0).sigma()).collect();
        self.sums = self.partial_sums();
    }

    /// Whether the paper's closed form for λ* applies to ring `i`: it is for a dial
    /// that stands still, with no key, read by the plain lens (and through an endless
    /// window, which the read-out says); and it divides by the diffusion factor,
    /// which must be above zero.
    fn formula_applies(&self, i: usize) -> bool {
        let l = self.process.latents[i];
        l.v == 0 && l.psi.is_none() && self.omega == 0.0 && self.process.diffusion_factor(i, 1) > 1e-3
    }

    /// The paper's closed form for dial `i`'s weight through an endless window, and
    /// what goes into it, as the panel writes them: |α|; φ, how far the forecast turns
    /// per token whichever way round (the lens follows the dial's pattern or its mirror
    /// image, whichever turns forwards: see [`Process::tuned`]); r; and σ.
    fn closed_form(&self, i: usize) -> [String; 4] {
        let alpha = self.process.alpha(i, 1);
        let phi = alpha.arg().abs();
        let r = alpha.abs() * (1.0 - self.process.epsilon);
        let psi = self.process.latents[i].psi.unwrap_or(1.0);
        let sigma = ring::lens_weight(C32::polar(alpha.abs(), phi), self.process.epsilon, psi, self.omega);
        [format!("{:.3}", alpha.abs()), format!("{:.0}°", degrees(phi)), format!("{r:.3}"), format!("{sigma:.3}")]
    }

    /// The paper's closed form for where the studied dial overtakes dial `other`:
    /// through an endless window, whatever window the scene is set to (the two agree
    /// once the window is long).
    fn formula(&self, other: usize) -> f32 {
        ring::crossing_static(self.process.diffusion_factor(self.selected, 1), self.process.epsilon, self.endless[other])
    }

    /// The chain of responses to a nudge of the studied ring if no reveal ever came,
    /// and the chance that a reveal ends it at each link.
    fn partial_sums(&self) -> Sums {
        let (i, keep) = (self.selected, 1.0 - self.process.epsilon);
        let alpha = self.process.alpha(i, 1);
        let beta = alpha * C32::polar(1.0, -self.tuning(i));
        let (mut points, mut chance) = (vec![C32::ZERO], Vec::new());
        let (mut term, mut left) = (alpha * beta, 1.0f32);
        // (stop at the window, or once a longer chain is too rare or too short to draw)
        while points.len() < 2 || (points.len() <= self.window.min(160) && left > 2e-4 && term.abs() > 1e-4) {
            chance.push(left * self.process.epsilon);
            left *= keep;
            points.push(*points.last().unwrap() + term);
            term = term * beta;
        }
        chance.push(left);
        let psi = self.process.latents[i].psi.unwrap_or(1.0);
        let mut within = 0.0;
        let enough = chance.iter().position(|c| {
            within += c;
            within > 0.94
        });
        let drawn = enough.unwrap_or(usize::MAX).max(3).min(points.len() - 1);
        Sums { points, chance, reversed: 0.5 * (1.0 - psi), drawn }
    }

    /// Ring `i`'s responses so far, turned back by the tuning and cut at the window:
    /// the links of the chain being built.
    fn live_links(&self, i: usize) -> Vec<C32> {
        let tuned = self.tuning(i);
        self.responses[i].iter().take(self.window).enumerate().map(|(t, &k)| k * C32::polar(1.0, -tuned * (t + 1) as f32)).collect()
    }

    /// How many links the chain being built for ring `i` has so far.
    fn live_len(&self, i: usize) -> usize {
        self.responses[i].len().min(self.window)
    }

    /// The dials in the order the lens lists them, each with its weight σ: by first
    /// Fourier mode, the mode every panel is about (as in the paper's comparison of
    /// latents). Equal weights keep the dials' own order.
    fn listing(&self) -> Vec<(usize, f32)> {
        let mut dials: Vec<(usize, f32)> = self.theory.iter().map(Factors::sigma).enumerate().collect();
        dials.sort_by(|a, b| b.1.total_cmp(&a.1));
        dials
    }

    /// The crossings the sweep chart marks, from left to right: all but those hard
    /// against the left edge (λ* ≤ 0.03), whose marks would be written over the
    /// axis's "λ = 0".
    fn marked(&self) -> Vec<(usize, f32)> {
        let mut marks: Vec<(usize, f32)> = self.switches.iter().map(|s| (s.0, s.1)).filter(|s| s.1 > 0.03).collect();
        marks.sort_by(|a, b| a.1.total_cmp(&b.1));
        marks
    }

    /// The listing in words, as long as `room` pixels allow at text size `size`.
    fn listing_text(&self, room: f32, size: f32) -> String {
        let listing = self.listing();
        let names = listing.iter().map(|d| NAME[d.0]).collect::<Vec<_>>().join(", then ");
        let full = listing.iter().map(|d| format!("dial {} ({:.2})", NAME[d.0], d.1)).collect::<Vec<_>>().join(",  then ");
        let forms: Vec<String> = if self.finer.is_empty() {
            vec![format!("the lens lists: {full}"), format!("the lens lists: {names}")]
        } else {
            // some dial has a finer pattern that the lens ranks above its first one
            let long = self.finer.iter().map(|d| format!("{}'s {}-wave pattern is stronger: {:.2}", NAME[d.latent], d.mode, d.sigma)).collect::<Vec<_>>().join(", ");
            let brief = self.finer.iter().map(|d| format!("{}'s {}-wave: {:.2}", NAME[d.latent], d.mode, d.sigma)).collect::<Vec<_>>().join(", ");
            vec![
                format!("by 1-wave pattern the lens lists: {full}   ·   {long}"),
                format!("by 1-wave pattern: {names}   ·   {long}"),
                format!("by 1-wave pattern: {names}   ·   {brief}"),
                format!("1-wave: {names}   ·   {brief}"),
                format!("1-wave: {names}"),
            ]
        };
        forms.iter().find(|t| text_width(t, size) <= room).unwrap_or(&forms[forms.len() - 1]).clone()
    }

    fn turn_text(&self, i: usize) -> String {
        let turn = degrees(self.process.latents[i].theta());
        if turn == 0.0 {
            "stands still".to_string()
        } else {
            format!("turns {turn:.0}° a token")
        }
    }

    // ----- drawing ------------------------------------------------------------------

    fn draw_rings(&self, f: &mut Frame, r: Rect) {
        hud_panel(f, r.min, r.max);
        let k = self.rings();
        let roomy = r.w() > 520.0;
        f.text_px(r.min + Vec2::new(10.0, 8.0), if roomy { "the hidden dials, and what the filter expects of each" } else { "hidden dials and the filter's forecast" }, 12.0, pal::TEXT, Align2::LEFT_TOP);
        if roomy {
            f.text_px(Vec2::new(r.max.x - 10.0, r.min.y + 9.0), "click a dial to study it below", 10.5, pal::DIM, Align2::RIGHT_TOP);
        }
        let cell = r.w() / k as f32;
        let radius = (cell * 0.5 - 26.0).min((r.h() - 96.0) * 0.5).max(14.0);
        // each dial's name with its persistence and its turn, if every one of them
        // fits its share of the width (with a little room between neighbours)
        let long: Vec<String> = (0..k).map(|i| format!(r"dial {}:  $\lambda = {:.2}$,  {}", NAME[i], self.process.latents[i].lambda, self.turn_text(i))).collect();
        let long = long.iter().all(|t| drawn_width(f, t, 11.5) + 12.0 <= cell).then_some(long);
        for (i, l) in self.process.latents.iter().enumerate() {
            let c = Vec2::new(r.min.x + cell * (i as f32 + 0.5), r.min.y + 40.0 + radius);
            let site = |x: usize| {
                let a = TAU * x as f32 / l.n as f32;
                c + radius * Vec2::new(a.cos(), -a.sin())
            };
            let chosen = i == self.selected;
            hud_circle(f, c, radius, if chosen { 1.6 } else { 1.0 }, if chosen { RING[i].a(0.8) } else { pal::DIM.a(0.45) });
            // (as drawn: gliding from where things stood before the newest token)
            let glided = self.glided().filter(|(b, _)| b.forecast.get(i).is_some_and(|p| p.len() == l.n));
            // the forecast: a disc on every site, its area the probability
            let forecast = self.filter.forecast(i);
            let largest = (TAU * radius / l.n as f32 * 0.78).min(radius * 0.55).min(30.0);
            for (x, &p) in forecast.iter().enumerate() {
                let p = glided.map_or(p, |(b, g)| b.forecast[i][x] + (p - b.forecast[i][x]) * g).clamp(0.0, 1.0);
                f.dl.overlay_discs.push(f.hud(site(x)), 3.0 + (largest - 3.0) * p.sqrt(), DARK.mix(RING[i], 0.3 + 0.7 * p.sqrt()));
            }
            // where the dial really is
            let a = self.dial_shown(i);
            hud_circle(f, c + radius * Vec2::new(a.cos(), -a.sin()), largest * 0.5 + 3.5, 1.8, pal::WHITE);
            // the forecast's first Fourier mode: its centre of mass on the ring
            let m = self.filter.mode(i, 1);
            let m = glided.map_or(m, |(b, g)| b.mode[i] + (m - b.mode[i]).scale(g));
            if m.abs() > 0.02 {
                hud_arrow(f, c, c + radius * Vec2::new(m.re, -m.im), 2.2, pal::WHITE);
            }
            f.dl.overlay_discs.push(f.hud(c), 5.0, pal::WHITE);
            // what it is, and how long since it was last shown
            let base = c.y + radius + largest * 0.5 + 10.0;
            let what = match &long {
                Some(long) => long[i].clone(),
                None => format!(r"dial {}  $\lambda = {:.2}$", NAME[i], l.lambda),
            };
            f.text_px(Vec2::new(c.x, base), what, 11.5, RING[i], Align2::CENTER_TOP);
            let mut status = match self.since {
                Some(0) => "shown by the last token".to_string(),
                Some(1) => "last shown 1 token ago".to_string(),
                Some(n) => format!("last shown {n} tokens ago"),
                None => "not shown yet".to_string(),
            };
            if let Some(psi) = l.psi {
                // (the wait is the same for every dial: where there is no room for
                // both, the key's bias is what this dial has of its own)
                let both = format!(r"key bias $\psi = {psi:.2}$  ·  {status}");
                status = if drawn_width(f, &both, 10.5) + 12.0 <= cell { both } else { format!(r"key bias $\psi = {psi:.2}$") };
            }
            if base + 28.0 < r.max.y && cell > 150.0 {
                f.text_px(Vec2::new(c.x, base + 15.0), status, 10.5, pal::DIM, Align2::CENTER_TOP);
            }
        }
    }

    fn draw_strip(&self, f: &mut Frame, r: Rect) {
        hud_panel(f, r.min, r.max);
        let k = self.rings();
        let roomy = r.w() > 520.0;
        f.text_px(r.min + Vec2::new(10.0, 7.0), "the tokens so far, newest on the right", 12.0, pal::TEXT, Align2::LEFT_TOP);
        if r.w() > 780.0 {
            f.text_px(Vec2::new(r.max.x - 10.0, r.min.y + 8.0), "lit column: the token showed the dials  ·  arrow: the forecast after it (which way, how sure)", 10.5, pal::DIM, Align2::RIGHT_TOP);
        } else if roomy {
            f.text_px(Vec2::new(r.max.x - 10.0, r.min.y + 8.0), "lit: dials shown  ·  arrow: the forecast", 10.5, pal::DIM, Align2::RIGHT_TOP);
        }
        let lags = r.h() > 26.0 + 14.0 * k as f32 + 16.0;
        let cell = ((r.h() - 30.0 - if lags { 14.0 } else { 0.0 }) / k as f32).clamp(8.0, 24.0);
        let (x0, y0) = (r.min.x + 30.0, r.min.y + 27.0);
        let cols = (((r.max.x - 10.0 - x0) / cell).floor() as usize).min(self.ticks.len());
        let right = x0 + cols as f32 * cell;
        // (the lags are numbered only while the reveal they are counted from is in view)
        let since = self.since.filter(|&s| s < cols);
        // As a token comes the strip slides a cell to the left, rather than stepping
        // there at once: the token that leaves it fades out at the left edge, the new one
        // fades in at the right (see `glided`).
        let glided = self.glided().map(|(_, g)| g);
        let leaving = (glided.is_some() && self.ticks.len() > cols) as usize;
        let shift = glided.map_or(0.0, |g| (1.0 - g) * cell);
        for (c, tick) in self.ticks.iter().skip(self.ticks.len() - cols - leaving).enumerate() {
            // (counted from the leftmost column of the strip, the one leaving it before that)
            let c = c as isize - leaving as isize;
            let x = x0 + c as f32 * cell + shift;
            let ink = match glided {
                Some(g) if c < 0 => 1.0 - g,
                Some(g) if c + 1 == cols as isize => g,
                _ => 1.0,
            };
            // (the one leaving is cut off at the strip's left edge, where the dials' names
            // are, and its arrows shrink with what is left of it, to nothing as the strip
            // comes to rest)
            let hi = x + cell - 1.0;
            let lo = x.max(x0).min(hi);
            let part = (hi - lo) / (cell - 1.0);
            for i in 0..k {
                let at = Vec2::new(lo, y0 + i as f32 * cell);
                hud_rect(f, at, Vec2::new(hi, at.y + cell - 1.0), if tick.revealed { DARK.mix(RING[i], 0.42) } else { CELL }.a(ink));
                let m = tick.m[i];
                if m.abs() > 0.04 {
                    let mid = Vec2::new(0.5 * (lo + hi), at.y + (cell - 1.0) * 0.5);
                    let v = Vec2::new(m.re, -m.im) * (cell - 4.0) * 0.5 * part;
                    hud_arrow(f, mid - v, mid + v, 1.2, if tick.revealed { pal::WHITE } else { RING[i] }.a(ink));
                }
            }
            // the lag of each token since the last reveal: the τ of the panels below
            let age = cols as isize - 1 - c;
            if lags && cell >= 13.0 {
                if let Some(s) = since.filter(|&s| age >= 0 && age as usize <= s) {
                    let age = age as usize;
                    let text = if age == s { "▲".to_string() } else { (s - age).to_string() };
                    f.text_px(Vec2::new(x + (cell - 1.0) * 0.5, y0 + k as f32 * cell + 2.0), text, 9.0, pal::DIM, Align2::CENTER_TOP);
                }
            }
        }
        for i in 0..k {
            f.text_px(Vec2::new(x0 - 8.0, y0 + (i as f32 + 0.5) * cell), NAME[i], 11.0f32.min(cell + 1.0), RING[i], Align2::RIGHT_CENTER);
        }
        if lags && cell >= 13.0 && roomy {
            let under = y0 + k as f32 * cell + 2.0;
            match (since, self.since) {
                (Some(s), _) => f.text_px(Vec2::new(right - (s as f32 + 1.0) * cell - 6.0 + shift, under), r"last reveal, then lags $\tau =$", 9.0, pal::DIM, Align2::RIGHT_TOP),
                (None, Some(s)) => f.text_px(Vec2::new(right, under), format!("the last reveal was {s} tokens ago, off the strip"), 9.0, pal::DIM, Align2::RIGHT_TOP),
                (None, None) => {}
            }
        }
    }

    /// The corners of the smallest upright box that holds the chain as drawn, for
    /// each of the given signs (+1 the chain itself, −1 the chain reversed by a key).
    fn reach(&self, signs: &[f32]) -> (Vec2, Vec2) {
        let (mut lo, mut hi) = (Vec2::ZERO, Vec2::ZERO);
        for p in &self.sums.points[..=self.sums.drawn] {
            for &sign in signs {
                lo = lo.min(sign * p.vec2());
                hi = hi.max(sign * p.vec2());
            }
        }
        (lo, hi)
    }

    /// The map from the complex plane to the inside of a panel that keeps the box
    /// `lo..hi` in view, with room left at the top and bottom for words.
    fn plane(lo: Vec2, hi: Vec2, r: Rect, top: f32, bottom: f32) -> impl Fn(C32) -> Vec2 {
        let inner = Rect::new(r.min.x + 26.0, r.min.y + top, r.max.x - 26.0, r.max.y - bottom);
        // (a chain that lies along a line is given some breadth, in proportion to its length)
        let span = hi - lo;
        let span = span.max(Vec2::splat(0.05 * span.max_element().max(1e-3)));
        let scale = (inner.w() / span.x).min(inner.h() / span.y).min(inner.w().min(inner.h()) / 0.25);
        let (mid, centre) = (0.5 * (lo + hi), inner.center());
        move |z: C32| centre + scale * Vec2::new(z.re - mid.x, -(z.im - mid.y))
    }

    fn draw_chain(&self, f: &mut Frame, r: Rect) {
        hud_panel(f, r.min, r.max);
        let i = self.selected;
        let roomy = r.w() > 250.0;
        f.text_px(r.min + Vec2::new(10.0, 8.0), format!("one nudge to dial {}", NAME[i]), 12.0, pal::TEXT, Align2::LEFT_TOP);
        f.text_px(r.min + Vec2::new(10.0, 24.0), if roomy { "its effect 1, 2, 3 … tokens later, tip to tail" } else { "its effects, tip to tail" }, 10.5, pal::DIM, Align2::LEFT_TOP);
        let sign = if self.reversed[i] { -1.0 } else { 1.0 };
        let (lo, hi) = self.reach(&[sign]);
        let to = Self::plane(lo, hi, r, 52.0, 52.0);
        let links = self.live_links(i);
        // every link of the chain: bright if that lag has passed since the last reveal,
        // faint for the lags a reveal may yet cut off. A link that doubles straight
        // back over the one before it is drawn a little to one side, so that a chain
        // that goes to and fro shows as a ladder rather than as a single line.
        let at = |k: usize| to(self.sums.points[k].scale(sign));
        let folds = |k: usize| k >= 2 && (at(k) - at(k - 1)).normalize_or_zero().dot((at(k - 1) - at(k - 2)).normalize_or_zero()) < -0.95;
        let rungs = (2..=self.sums.drawn).filter(|&k| folds(k)).count();
        let step = {
            let across = (at(1) - at(0)).perp().normalize_or_zero();
            let across = if across.y < 0.0 { -across } else { across };
            across * 9.0f32.min((r.h() - 112.0) / (rungs as f32 + 1.0)).max(2.0)
        };
        let mut aside = -0.5 * rungs as f32 * step;
        let (start, mut end_aside) = (at(0) + aside, aside);
        for k in 1..=self.sums.drawn {
            if folds(k) {
                hud_line(f, at(k - 1) + aside, at(k - 1) + aside + step, 1.0, RING[i].a(0.38));
                aside += step;
            }
            let (a, b) = (at(k - 1) + aside, at(k) + aside);
            let (width, color) = if k <= links.len() { (2.2, RING[i]) } else { (1.2, RING[i].a(0.38)) };
            if (b - a).length() > 9.0 {
                hud_arrow(f, a, b, width, color);
            } else {
                hud_line(f, a, b, width, color);
            }
            if k <= 3 && (b - a).length() > 24.0 {
                // (beside a ladder's rung rather than between two of them)
                let beside = if rungs > 0 { (b - a).normalize() * ((b - a).length() * 0.5 + 16.0) } else { -(b - a).perp().normalize() * 9.0 };
                f.text_px(0.5 * (a + b) + beside, if k == 1 && (b - a).length() > 50.0 { "lag 1".to_string() } else { k.to_string() }, 9.5, pal::DIM, Align2::CENTER_CENTER);
            }
            if k == links.len().min(self.sums.drawn) {
                end_aside = aside;
            }
        }
        // from the start to where the chain has got to (drawn beside it, so that a
        // straight chain does not hide it; and gliding to the link a new token adds)
        let (path, end) = links.iter().fold((0.0, C32::ZERO), |(path, end), k| (path + k.abs(), end + *k));
        let (from, to, g) = self.live_end(i);
        let (a, b) = (at(0) + end_aside, at(from).lerp(at(to), g) + end_aside);
        if (b - a).length() > 6.0 {
            let beside = (b - a).perp().normalize() * if rungs > 0 { -4.0 } else { 5.0 };
            hud_arrow(f, a + beside, b + beside, 1.4, pal::WHITE);
        }
        let a = start;
        f.dl.overlay_discs.push(f.hud(a), 6.0, pal::WHITE);
        let heading = (at(1) - at(0)).normalize_or_zero();
        let (off, align) = if heading.x > 0.5 {
            (Vec2::new(2.0, -6.0), Align2::RIGHT_BOTTOM)
        } else if heading.x < -0.5 {
            (Vec2::new(-2.0, -6.0), Align2::LEFT_BOTTOM)
        } else {
            (Vec2::new(0.0, -8.0 * heading.y.signum()), if heading.y > 0.0 { Align2::CENTER_BOTTOM } else { Align2::CENTER_TOP })
        };
        f.text_px(a + off, "start", 9.5, pal::DIM, align);
        let t = self.theory[i];
        let now = if self.nudges[i].is_none() {
            "waiting for the first reveal".to_string()
        } else if links.is_empty() {
            (if roomy { "a reveal has just come: nothing added yet" } else { "a reveal has just come" }).to_string()
        } else if roomy {
            format!("this one so far: path {:.2}, start to end {:.2} (white)", path, end.abs())
        } else {
            format!("so far: {:.2} of path {:.2}", end.abs(), path)
        };
        f.text_px(Vec2::new(r.min.x + 10.0, r.max.y - 38.0), now, 10.5, pal::TEXT, Align2::LEFT_TOP);
        let average = if roomy {
            format!(r"on average: path $a = {:.2}$, start to end {:.2}: $\rho_\tau = {:.2}$", t.a, t.a * t.rho_tau, t.rho_tau)
        } else {
            format!(r"average: {:.2} of {:.2}: $\rho_\tau = {:.2}$", t.a * t.rho_tau, t.a, t.rho_tau)
        };
        f.text_px(Vec2::new(r.min.x + 10.0, r.max.y - 22.0), average, 10.5, RING[i], Align2::LEFT_TOP);
    }

    fn draw_cloud(&self, f: &mut Frame, r: Rect) {
        hud_panel(f, r.min, r.max);
        let i = self.selected;
        let roomy = r.w() > 250.0;
        f.text_px(r.min + Vec2::new(10.0, 8.0), "many nudges", 12.0, pal::TEXT, Align2::LEFT_TOP);
        f.text_px(r.min + Vec2::new(10.0, 24.0), if roomy { "where each chain ends: bigger = more often" } else { "where the chains end" }, 10.5, pal::DIM, Align2::LEFT_TOP);
        let t = self.theory[i];
        // the average distance of an end from the start is drawn as a circle: keep it in view
        let far = t.a * t.rho_tau;
        let (lo, hi) = self.reach(if self.sums.reversed > 0.0 { &[1.0, -1.0] } else { &[1.0] });
        let to = Self::plane(lo.min(Vec2::splat(-far)), hi.max(Vec2::splat(far)), r, 48.0, 48.0);
        let origin = to(C32::ZERO);
        let scale = (to(C32::ONE) - origin).length();
        hud_circle(f, origin, far * scale, 1.0, pal::DIM.a(0.9));
        hud_line(f, origin - Vec2::new(5.0, 0.0), origin + Vec2::new(5.0, 0.0), 1.0, pal::DIM);
        hud_line(f, origin - Vec2::new(0.0, 5.0), origin + Vec2::new(0.0, 5.0), 1.0, pal::DIM);
        // every possible end, sized by its chance (reversed by a key of −1)
        let straight = 1.0 - self.sums.reversed;
        let mut mean = C32::ZERO;
        for (k, (p, &chance)) in self.sums.points.iter().zip(&self.sums.chance).enumerate() {
            for (sign, share) in [(1.0, straight), (-1.0, self.sums.reversed)] {
                let weight = chance * share;
                if weight > 1e-4 && k <= self.sums.drawn {
                    f.dl.overlay_discs.push(f.hud(to(p.scale(sign))), 3.0 + 24.0 * weight.sqrt(), if sign > 0.0 { RING[i] } else { RING[i].mix(pal::WHITE, 0.55) });
                }
                mean = mean + p.scale(sign * weight);
            }
        }
        // the end of the chain being built now (gliding to the link a new token adds)
        let (from, now, g) = self.live_end(i);
        let sign = if self.reversed[i] { -1.0 } else { 1.0 };
        hud_circle(f, to(self.sums.points[from].scale(sign)).lerp(to(self.sums.points[now].scale(sign)), g), 8.0, 1.4, pal::WHITE);
        // their average, as an arrow
        // (if the ends all lie along one line the arrow would be buried under them: it
        // is then drawn just below that line)
        let along = mean.vec2().normalize_or_zero();
        let in_line = self.sums.points[..=self.sums.drawn].iter().all(|p| p.vec2().perp_dot(along).abs() < 0.02 * far.max(1e-3));
        let below = if in_line && mean.abs() * scale > 3.0 { Vec2::new(-along.y, along.x).abs() * 15.0 } else { Vec2::ZERO };
        if mean.abs() * scale > 3.0 {
            hud_arrow(f, origin + below, to(mean) + below, 2.6, pal::WHITE);
        }
        f.dl.overlay_discs.push(f.hud(to(mean) + below), 8.5, pal::WHITE);
        let (line1, line2) = if roomy {
            (format!(r"white arrow: the average end, length $\sigma = {:.2}$", t.sigma()), format!(r"circle: the ends' average distance, {:.2}: $\rho_c = {:.2}$", far, t.rho_c))
        } else {
            (format!(r"arrow (average): $\sigma = {:.2}$", t.sigma()), format!(r"circle {:.2}: $\rho_c = {:.2}$", far, t.rho_c))
        };
        f.text_px(Vec2::new(r.min.x + 10.0, r.max.y - 38.0), line1, 10.5, pal::TEXT, Align2::LEFT_TOP);
        f.text_px(Vec2::new(r.min.x + 10.0, r.max.y - 22.0), line2, 10.5, RING[i], Align2::LEFT_TOP);
        if self.sums.reversed > 0.0 && roomy {
            f.text_px(Vec2::new(r.max.x - 10.0, r.min.y + 8.0), "pale: key −1", 10.0, RING[i].mix(pal::WHITE, 0.55), Align2::RIGHT_TOP);
        }
    }

    fn draw_bars(&self, f: &mut Frame, r: Rect) {
        hud_panel(f, r.min, r.max);
        let k = self.rings();
        let roomy = r.w() > 330.0;
        let (lift, top, base) = Self::bar_span(r);
        let tall = lift == 0.0;
        let done = self.measured[0].contexts;
        let marks = format!("bars: theory  ·  white marks: measured on {} nudges of the filter", count(done as usize));
        if tall {
            f.text_px(r.min + Vec2::new(10.0, 8.0), if roomy { "three factors and their product, for each dial" } else { "three factors and their product" }, 12.0, pal::TEXT, Align2::LEFT_TOP);
            f.text_px(r.min + Vec2::new(10.0, 24.0), if roomy { marks } else { format!("bars: theory · marks: {} nudges", count(done as usize)) }, 10.5, pal::DIM, Align2::LEFT_TOP);
        } else {
            let (title, marks) = ("three factors and their product", "white marks: measured");
            f.text_px(r.min + Vec2::new(10.0, 8.0), title, 12.0, pal::TEXT, Align2::LEFT_TOP);
            // (beside it, where there is room)
            if text_width(title, 12.0) + text_width(marks, 10.5) + 30.0 <= r.w() {
                f.text_px(Vec2::new(r.max.x - 10.0, r.min.y + 9.0), marks, 10.5, pal::DIM, Align2::RIGHT_TOP);
            }
        }
        // (two lines of heading, the factor itself, whether it is a share of one)
        type Group = (&'static str, &'static str, fn(&Factors) -> f32, bool);
        let groups: [Group; 4] = [
            ("magnitude", "$a$", |t| t.a, false),
            ("lag", r"coherence $\rho_\tau$", |t| t.rho_tau, true),
            ("context", r"coherence $\rho_c$", |t| t.rho_c, true),
            ("lens weight", r"$\sigma = a \cdot \rho_\tau \cdot \rho_c$", |t| t.sigma(), false),
        ];
        let floor = self.theory.iter().map(|t| 0.2 * t.a).fold(1e-3, f32::max);
        let gw = (r.w() - 20.0) / 4.0;
        let bw = ((gw - 18.0) / k as f32).min(30.0);
        for (g, (name, symbol, value, unit)) in groups.iter().enumerate() {
            let x0 = r.min.x + 10.0 + g as f32 * gw;
            let mid = x0 + gw * 0.5;
            f.text_px(Vec2::new(mid, r.min.y + 43.0 - lift), *name, 10.5, pal::TEXT, Align2::CENTER_TOP);
            f.text_px(Vec2::new(mid, r.min.y + 57.0 - lift), if gw > 84.0 { *symbol } else { symbol.trim_start_matches("coherence ") }, 10.5, if g == 0 || g == 3 { pal::DIM } else { pal::TEXT }, Align2::CENTER_TOP);
            // (a group with nothing in it, every weight zero for instance, is drawn on
            // the scale of the magnitudes: on a scale of its own, the measured marks
            // would show their sampling noise at full height)
            let full = if *unit { 1.0 } else { self.theory.iter().map(value).fold(floor, f32::max) * 1.05 };
            let left = mid - bw * k as f32 * 0.5;
            if *unit {
                hud_line(f, Vec2::new(left - 3.0, top), Vec2::new(left + bw * k as f32 + 3.0, top), 1.0, pal::FAINT.a(0.7));
            }
            hud_line(f, Vec2::new(left - 3.0, base), Vec2::new(left + bw * k as f32 + 3.0, base), 1.0, pal::DIM.a(0.8));
            for i in 0..k {
                let v = value(&self.theory[i]);
                let x = left + i as f32 * bw;
                let y = base - (base - top) * (v / full).clamp(0.0, 1.0);
                hud_rect(f, Vec2::new(x + 2.0, y), Vec2::new(x + bw - 2.0, base), RING[i].a(if i == self.selected { 1.0 } else { 0.72 }));
                // (bars too thin to carry a number each: only the studied ring's is written)
                if bw >= 21.0 || i == self.selected {
                    f.text_px(Vec2::new(x + bw * 0.5, y - 2.0), format!("{v:.2}"), if bw > 26.0 { 10.0 } else { 9.0 }, if bw >= 21.0 { pal::TEXT } else { RING[i] }, Align2::CENTER_BOTTOM);
                }
                f.text_px(Vec2::new(x + bw * 0.5, base + 3.0), NAME[i], 10.0, RING[i], Align2::CENTER_TOP);
                if done >= 200 {
                    let m = value(&self.measured[i].factors());
                    let ym = base - (base - top) * (m / full).clamp(0.0, 1.0);
                    hud_line(f, Vec2::new(x, ym), Vec2::new(x + bw, ym), 2.0, pal::WHITE);
                }
            }
        }
        // the order the lens lists them in
        f.text_px(Vec2::new(r.min.x + 10.0, r.max.y - 20.0), self.listing_text(r.w() - 20.0, 11.5), 11.5, pal::WHITE, Align2::LEFT_TOP);
    }

    /// The frame of a small chart inside a panel: its plotting area.
    fn plot_area(r: Rect) -> Rect {
        Rect::new(r.min.x + 36.0, r.min.y + 56.0, r.max.x - 14.0, r.max.y - 26.0)
    }

    /// The bars of the factors in a panel: how far the headings are lifted (a short
    /// panel has no room for their second line), and the heights of a full bar's top
    /// and of the bars' base.
    fn bar_span(r: Rect) -> (f32, f32, f32) {
        let lift = if r.h() > 200.0 { 0.0 } else { 14.0 };
        let top = r.min.y + 88.0 - lift;
        (lift, top, (r.max.y - 42.0).max(top + 1.0))
    }

    /// The axes of such a chart of lens weight σ: from zero to `top` up the side.
    fn sigma_axes(f: &mut Frame, p: Rect, top: f32) {
        hud_line(f, p.min, Vec2::new(p.min.x, p.max.y), 1.0, pal::DIM.a(0.8));
        hud_line(f, Vec2::new(p.min.x, p.max.y), p.max, 1.0, pal::DIM.a(0.8));
        f.text_px(Vec2::new(p.min.x - 5.0, p.min.y), format!("{top:.1}"), 9.5, pal::DIM, Align2::RIGHT_CENTER);
        f.text_px(Vec2::new(p.min.x - 5.0, p.max.y), "0", 9.5, pal::DIM, Align2::RIGHT_CENTER);
        f.text_px(Vec2::new(p.min.x - 22.0, p.center().y), r"$\sigma$", 10.5, pal::DIM, Align2::CENTER_CENTER);
    }

    fn draw_tuning(&self, f: &mut Frame, r: Rect) {
        hud_panel(f, r.min, r.max);
        let roomy = r.w() > 330.0;
        f.text_px(r.min + Vec2::new(10.0, 8.0), if roomy { r"tuning the lens: each dial's weight $\sigma$ against the frequency $\omega$" } else { r"tuning: weight $\sigma$ against $\omega$" }, 12.0, pal::TEXT, Align2::LEFT_TOP);
        let caption = r"a lens tuned to $\omega$ adds up responses that turn by $\omega$ per token; each dial peaks at its forecast's turn ▲";
        if drawn_width(f, caption, 10.5) + 20.0 <= r.w() {
            f.text_px(r.min + Vec2::new(10.0, 24.0), caption, 10.5, pal::DIM, Align2::LEFT_TOP);
        }
        let p = Self::plot_area(r);
        let top = self.tuned.iter().flatten().fold(1e-3f32, |a, &b| a.max(b)) * 1.2;
        let at = |omega: f32, v: f32| Vec2::new(p.min.x + p.w() * omega / PI, p.max.y - p.h() * (v / top).clamp(0.0, 1.0));
        Self::sigma_axes(f, p, top);
        for (frac, wide, narrow) in [(0.0, r"$\omega = 0$: steady", "0"), (0.5, r"$\pi/2$", r"$\pi/2$"), (1.0, r"$\pi$: flips every token", r"$\pi$")] {
            let align = if frac == 0.0 { Align2::LEFT_TOP } else if frac == 1.0 { Align2::RIGHT_TOP } else { Align2::CENTER_TOP };
            f.text_px(Vec2::new(p.min.x + p.w() * frac, p.max.y + 4.0), if roomy { wide } else { narrow }, 9.5, pal::DIM, align);
        }
        // where the lens is tuned now
        let x = at(self.omega, 0.0).x;
        hud_line(f, Vec2::new(x, p.min.y), Vec2::new(x, p.max.y), 1.2, pal::WHITE.a(0.75));
        if roomy {
            f.text_px(Vec2::new(x + if self.omega > 0.8 * PI { 2.0 } else { -2.0 }, p.min.y - 3.0), "the lens now", 9.5, pal::WHITE, if self.omega > 0.8 * PI { Align2::RIGHT_BOTTOM } else { Align2::LEFT_BOTTOM });
        }
        for (i, curve) in self.tuned.iter().enumerate() {
            let pts: Vec<Vec3> = curve.iter().enumerate().map(|(s, &v)| f.hud(at(PI * s as f32 / CURVE as f32, v))).collect();
            f.dl.overlay.polyline(pts, if i == self.selected { 2.2 } else { 1.6 }, RING[i]);
            // its own turn per token, marked on the axis (side by side where rings share one)
            let turn = self.process.turn(i, 1).abs();
            let same = (0..i).filter(|&o| (self.process.turn(o, 1).abs() - turn).abs() < 1e-3).count();
            let nudge = if turn < 0.1 { 6.0 } else if turn > PI - 0.1 { -6.0 } else { 0.0 } + 9.0 * same as f32 * if turn > PI - 0.1 { -1.0 } else { 1.0 };
            f.text_px(at(turn, 0.0) + Vec2::new(nudge, -1.0), "▲", 9.0, RING[i], Align2::CENTER_BOTTOM);
            // and its name at its peak
            let peak = (0..=CURVE).max_by(|&a, &b| curve[a].total_cmp(&curve[b])).unwrap_or(0);
            let named = at(PI * peak as f32 / CURVE as f32, curve[peak]) + Vec2::new(if peak == 0 { 22.0 } else if peak == CURVE { -10.0 } else { 0.0 }, -4.0);
            f.text_px(named, NAME[i], 10.5, RING[i], Align2::CENTER_BOTTOM);
            f.dl.overlay_discs.push(f.hud(at(self.omega, self.theory[i].sigma())), 7.0, RING[i]);
        }
    }

    fn draw_sweep(&self, f: &mut Frame, r: Rect) {
        hud_panel(f, r.min, r.max);
        let i = self.selected;
        let roomy = r.w() > 330.0;
        f.text_px(r.min + Vec2::new(10.0, 8.0), if roomy { format!(r"if dial {} remembered more: its weight $\sigma$ against its persistence $\lambda$", NAME[i]) } else { format!(r"dial {}: $\sigma$ against $\lambda$", NAME[i]) }, 12.0, pal::TEXT, Align2::LEFT_TOP);
        let caption = "level lines: the other dials. Where the curve crosses one, the lens's order switches";
        if drawn_width(f, caption, 10.5) + 20.0 <= r.w() {
            f.text_px(r.min + Vec2::new(10.0, 24.0), caption, 10.5, pal::DIM, Align2::LEFT_TOP);
        }
        let p = Self::plot_area(r);
        let others = (0..self.rings()).filter(|&o| o != i).map(|o| self.theory[o].sigma());
        let top = self.sweep.iter().copied().chain(others).fold(1e-3f32, f32::max) * 1.08;
        let at = |lambda: f32, v: f32| Vec2::new(p.min.x + p.w() * lambda, p.max.y - p.h() * (v / top).clamp(0.0, 1.0));
        Self::sigma_axes(f, p, top);
        f.text_px(Vec2::new(p.min.x, p.max.y + 4.0), r"$\lambda = 0$", 9.5, pal::DIM, Align2::LEFT_TOP);
        f.text_px(Vec2::new(p.max.x, p.max.y + 4.0), "1", 9.5, pal::DIM, Align2::RIGHT_TOP);
        let mut others: Vec<usize> = (0..self.rings()).filter(|&o| o != i).collect();
        others.sort_by(|&a, &b| self.theory[b].sigma().total_cmp(&self.theory[a].sigma()));
        let mut above = f32::MIN;
        for o in others {
            let level = self.theory[o].sigma();
            hud_line(f, at(0.0, level), at(1.0, level), 1.4, RING[o].a(0.9));
            // (the name sits on its line; under it, if the line above is too close)
            let y = at(0.0, level).y;
            let (dy, align) = if y - above < 14.0 { (2.0, Align2::LEFT_TOP) } else { (-2.0, Align2::LEFT_BOTTOM) };
            above = y;
            f.text_px(Vec2::new(p.min.x + 4.0, y + dy), format!("dial {}", NAME[o]), 9.5, RING[o], align);
        }
        let pts: Vec<Vec3> = self.sweep.iter().enumerate().map(|(s, &v)| f.hud(at(s as f32 / CURVE as f32, v))).collect();
        f.dl.overlay.polyline(pts, 2.2, RING[i]);
        // Each λ* is written under the axis, centred on its line, if there is room
        // there (as "λ* 0.65" if "λ* = 0.65" is too wide). Where a neighbour or the
        // axis's own "λ = 0" and "1" are in the way, it goes beside its line, in the
        // first free row above the axis (and leftwards at the right-hand end, so as to
        // stay inside the chart). `taken`: how far right each row is written so far,
        // the row under the axis first.
        // (A crossing hard against the left edge is not marked: see `marked`.)
        let mut taken = [p.min.x + width(r"$\lambda = 0$", 9.5) + 6.0, f32::MIN, f32::MIN, f32::MIN];
        for (o, lambda) in self.marked() {
            let top_of = at(lambda, self.theory[o].sigma());
            hud_line(f, Vec2::new(top_of.x, p.max.y), top_of, 1.0, pal::WHITE.a(0.6));
            let label = format!(r"$\lambda^* = {lambda:.2}$");
            let under = [label.clone(), format!(r"$\lambda^*$ {lambda:.2}")].into_iter().find(|t| {
                let half = 0.5 * width(t, 9.5);
                top_of.x - half >= taken[0] && top_of.x + half <= p.max.x - 12.0
            });
            if let Some(text) = under {
                taken[0] = top_of.x + 0.5 * width(&text, 9.5) + 5.0;
                f.text_px(Vec2::new(top_of.x, p.max.y + 4.0), text, 9.5, pal::WHITE, Align2::CENTER_TOP);
                continue;
            }
            let w = width(&label, 9.5);
            let leftwards = top_of.x + 3.0 + w > p.max.x;
            let (from, to) = if leftwards { (top_of.x - 3.0 - w, top_of.x - 3.0) } else { (top_of.x + 3.0, top_of.x + 3.0 + w) };
            let row = (1..taken.len()).find(|&k| from >= taken[k]).unwrap_or(taken.len() - 1);
            taken[row] = to + 5.0;
            let y = p.max.y - 3.0 - 12.0 * (row - 1) as f32;
            let (x, align) = if leftwards { (to, Align2::RIGHT_BOTTOM) } else { (from, Align2::LEFT_BOTTOM) };
            f.text_px(Vec2::new(x, y), label, 9.5, pal::WHITE, align);
        }
        let here = at(self.process.latents[i].lambda, self.theory[i].sigma());
        f.dl.overlay_discs.push(f.hud(here), 9.0, pal::WHITE);
        f.text_px(here + Vec2::new(-7.0, -5.0), "now", 9.5, pal::WHITE, Align2::RIGHT_BOTTOM);
    }
}

impl Scene for Belief {
    fn view(&self) -> View {
        View::Fit { center: Vec2::ZERO, half: Vec2::splat(1.0) }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn animated(&self) -> bool {
        self.running || self.measured[0].contexts < LAB_CONTEXTS
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("click a dial to study it")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "Hidden dials turn on their own. Most tokens show nothing; about one in five shows where every dial is. A filter keeps the best possible forecast of each dial from those glimpses. The Jacobian lens asks one question of it: if a glimpse had shown something slightly different, how would the forecast respond over the tokens that follow? It adds that response over the lags, averages it over contexts, and ranks each dial by how much is left. The picture shows what is left, and why.");
        ui::note(ui, "This is the toy model of Ostrow, Qian, Liu and Fiete, \"A Mathematical Theory of the Jacobian Lens\" (2026); their symbols are given beside the plain names.");

        ui::section(ui, "Things to try");
        for (p, preset) in PRESETS.iter().enumerate() {
            if ui::experiment(ui, preset.label, preset.look_for) {
                self.apply(p);
            }
        }

        ui::section(ui, "Run");
        ui.horizontal(|ui| {
            if ui.button(if self.running { "pause" } else { "run" }).clicked() {
                self.running = !self.running;
            }
            if ui.button("one token").clicked() {
                self.tick();
            }
            if ui.button("start again").clicked() {
                self.restart();
            }
        });
        ui::slider(ui, "tokens per second", &mut self.rate, 0.5..=20.0);

        ui::section(ui, "Study one dial");
        let k = self.rings();
        let names: Vec<String> = (0..k).map(|i| format!("dial {}", NAME[i])).collect();
        let names: Vec<&str> = names.iter().map(String::as_str).collect();
        self.stale |= ui::choice(ui, &names, &mut self.selected);
        ui::note(ui, "The two chain panels in the middle and the chart at the bottom right are about this dial. Clicking a dial in the picture chooses it too.");

        ui::section(ui, "The dials");
        let mut changed = false;
        let mut count = k - 2;
        if ui::choice(ui, &["2 dials", "3 dials"], &mut count) {
            let model = *self.process.latents.last().unwrap();
            self.process.latents.resize(count + 2, model);
            changed = true;
        }
        let mut size = SIZES.iter().position(|&n| n == self.process.latents[0].n).unwrap_or(2);
        ui::words(ui, "sites round each dial ($n$)", 12.5, pal::DIM);
        if ui::choice(ui, &["2", "4", "8", "16"], &mut size) {
            for l in &mut self.process.latents {
                l.n = SIZES[size];
                l.v %= l.n;
            }
            changed = true;
        }
        ui::term(ui, r"Persistence $\lambda$:", "the chance per token that a dial carries on from where it was. Otherwise it jumps to a random site and all memory of it is worthless.");
        ui::term(ui, "Velocity $v$:", r"the sites it moves per token while it carries on. Its turn per token is $\theta = 360° \cdot v / n$.");
        ui::term(ui, r"Key, and its bias $\psi$:", r"every token then carries a +1 or a −1 for the dial, and a glimpse that comes straight after a −1 shows the dial half a turn away from where it is (only the key of the token just before the glimpse counts). $\psi$ is how lopsided the key is: 1 is always +1, 0 is a fair coin.");
        let n = self.process.latents[0].n;
        for (i, l) in self.process.latents.iter_mut().enumerate() {
            ui.add_space(4.0);
            ui.label(egui::RichText::new(format!("■ dial {}", NAME[i])).size(13.0).strong().color(RING[i].egui()));
            changed |= ui::slider(ui, r"persistence $\lambda$", &mut l.lambda, 0.3..=0.99);
            changed |= ui::slider_int(ui, "velocity $v$ (sites per token)", &mut l.v, 0..=n - 1);
            let mut keyed = l.psi.is_some();
            if ui.checkbox(&mut keyed, "has a key").changed() {
                l.psi = keyed.then_some(0.75);
                changed = true;
            }
            if let Some(psi) = &mut l.psi {
                changed |= ui::slider(ui, r"key bias $\psi$", psi, 0.0..=1.0);
            }
        }
        ui::section(ui, "Shared by all dials");
        changed |= ui::slider(ui, r"reveal probability $\varepsilon$", &mut self.process.epsilon, 0.05..=0.6);
        ui::note(ui, "The chance that a token shows the dials. A reveal replaces everything the filter believed, so it also wipes out any nudge.");
        changed |= ui::slider(ui, "diffusion $q$", &mut self.process.q, 0.0..=0.45);
        ui::note(ui, "The chance of slipping one site forward, and the same of slipping back, at each token on which a dial carries on.");
        if changed {
            self.restart();
        }

        ui::section(ui, "The lens");
        let mut lens = ui::slider_int_log(ui, "window $H$ (lags added up)", &mut self.window, 1..=256);
        lens |= ui::slider(ui, r"tuning $\omega$", &mut self.omega, 0.0..=PI);
        ui::note(ui, r"The plain lens ($\omega = 0$) adds the responses at lags 1 to $H$ as they are. Tuned to $\omega$, it first turns the response at lag $\tau$ back by $\omega \cdot \tau$, so a response that itself turns by $\omega$ per token adds up straight. (The paper's $\omega$-$J$-lens.)");
        if lens {
            self.relens();
        }

        ui::section(ui, "Colours");
        let mut entries: Vec<(Color, String)> = (0..k).map(|i| (RING[i], format!("dial {}: its forecast (discs), its chain and its bars", NAME[i]))).collect();
        entries.push((pal::WHITE, "white circle: where a dial really is. White arrows: which way a forecast leans (top), one chain from start to end (middle left), the average of all chain ends (middle)".to_string()));
        let entries: Vec<(Color, &str)> = entries.iter().map(|(c, t)| (*c, t.as_str())).collect();
        ui::key(ui, &entries);

        ui::section(ui, "Words");
        ui::term(ui, "Token:", "one item of the sequence a model reads. Here a token either shows where every dial is (a glimpse, or reveal) or shows nothing.");
        ui::term(ui, "Hidden Markov model:", "a process with a state nobody sees directly (the dials) that moves at random by fixed rules, and emits tokens that depend on it.");
        ui::term(ui, "Forecast (belief):", "the probability of each site for a dial's next position, given every token so far. The paper's belief state $m$ is its first Fourier mode: the arrow in each ring, long when the filter is sure.");
        ui::term(ui, "Bayes filter:", r"the rule that keeps the forecast exactly right. Here: after a glimpse, start from what was shown; at every token, push the forecast one step along the dial's own motion, which multiplies $m$ by $\alpha$ (shrink by $|\alpha|$, turn by $\theta$).");
        ui::term(ui, "Jacobian:", r"how much an output changes per unit change of an input. Here the input is what a glimpse showed and the output is the forecast $\tau$ tokens later. The answer is $\alpha^{\tau + 1}$ until the next glimpse and zero after it (and reversed, if the key said the glimpse was shown half a turn off).");
        ui::term(ui, r"Lag $\tau$:", "how many tokens after the nudge the response is read.");
        ui::term(ui, "Context:", "everything else that varies from one nudge to the next: here, how soon the next glimpse comes, and the key.");
        ui::term(ui, "Magnitude $a$:", "the lengths of the responses added over the lags, on average: the chain's path length.");
        ui::term(ui, r"Lag coherence $\rho_\tau$:", "the chains' average start-to-end distance over their average path length. 1 if every chain is straight, small if they curl up or double back.");
        ui::term(ui, r"Context coherence $\rho_c$:", "the length of the average chain end over the average distance of the ends from the start. 1 if the chains all end in the same direction, 0 if as many end one way as the opposite way.");
        ui::term(ui, r"Lens weight $\sigma$:", r"the product $a \cdot \rho_\tau \cdot \rho_c$: the length of the average chain end, and the singular value the lens gives this dial. The lens lists dials by it.");
        ui::term(ui, "Pattern (Fourier mode):", "everything here is about a dial's position: one wave round its ring, the arrow in the picture, the paper's first Fourier mode, on which it compares dials. The filter also carries finer patterns of each dial, with two or more waves round the ring. They usually fade faster. But for a dial that turns fast one of them can turn more slowly, and with much slipping (a large $q$) one of them can outlast the first; the lens then ranks it above the dial's position, and the list under the bars says so when that happens.");

        ui::section(ui, "Numbers for the studied dial");
        let i = self.selected.min(k - 1);
        let (t, m) = (self.theory.get(i).copied().unwrap_or_default(), self.measured[i].factors());
        let [shrink, phi, memory, endless] = self.closed_form(i);
        ui::readout(ui, r"shrink per token $|\alpha|$", shrink);
        ui::readout(ui, r"turn per token $\theta$", format!("{:.0}°", degrees(self.process.latents[i].theta())));
        if self.process.diffusion_factor(i, 1) < 0.0 {
            ui::note(ui, "With this much slipping the forecast of the dial's position also changes sign every token (the diffusion factor is below zero): to the lens that is half a turn more.");
        }
        ui::readout(ui, r"memory per token $r = |\alpha|(1 - \varepsilon)$", memory);
        ui::note(ui, "Theory, then measured:");
        ui::readout(ui, "magnitude $a$", format!("{:.3}   {:.3}", t.a, m.a));
        ui::readout(ui, r"lag coherence $\rho_\tau$", format!("{:.3}   {:.3}", t.rho_tau, m.rho_tau));
        ui::readout(ui, r"context coherence $\rho_c$", format!("{:.3}   {:.3}", t.rho_c, m.rho_c));
        ui::readout(ui, r"lens weight $\sigma$", format!("{:.3}   {:.3}", t.sigma(), m.sigma()));
        ui::readout(ui, "nudges measured", crate::frame::count(self.measured[i].contexts as usize));
        ui::readout(ui, r"turn of the forecast, either way $\varphi$", phi);
        ui::readout(ui, r"$\sigma$ for an endless window", endless);
        ui::note(ui, r"That last number is the paper's closed form ($\psi = 1$ for a dial without a key):");
        ui::equation(ui, r"\frac{|\psi| \cdot |\alpha| \cdot r}{\left|1 - r \cdot e^{i(\varphi - \omega)}\right|}");
        ui::note(ui, r"Memory in the numerator, turning in the denominator. $\varphi$ is how far the forecast's arrow turns per token, whichever way round (0° to 180°): $\theta$ without its sign, unless slipping reverses the forecast every token, which turns it half a turn more.");
        if let Some(d) = self.finer.iter().find(|d| d.latent == i) {
            ui::readout(ui, &format!(r"its {}-wave pattern, $\sigma$", d.mode), format!("{:.3}", d.sigma));
            ui::note(ui, "A finer pattern of this dial outranks its position in the lens (see \"Pattern\" above). The panels stay with its position.");
        }
        let formula = self.formula_applies(i);
        for &(o, lambda, rising) in &self.switches {
            ui::readout(ui, &format!(r"{} dial {} above $\lambda^*$", if rising { "overtakes" } else { "falls behind" }, NAME[o]), format!("{lambda:.3}"));
            if formula && rising {
                ui::readout(ui, "   the paper's formula (endless window)", format!("{:.3}", self.formula(o)));
            }
        }

        ui::section(ui, "How good is the filter?");
        let per = |total: f64| if self.lab_tokens > 0 { format!("{:.3}", total / self.lab_tokens as f64) } else { "…".to_string() };
        ui::readout(ui, "surprise per token: the filter", per(self.surprise[0]));
        ui::readout(ui, "surprise per token: no memory", per(self.surprise[1]));
        let tokens = self.process.vocab();
        ui::readout(ui, "a blind guess among all tokens", format!("{:.3}", (tokens as f32).ln()));
        ui::readout(ui, "different tokens", crate::frame::count(tokens));
        ui::note(ui, "Surprise is minus the logarithm of the probability a predictor gave to the token that came (in nats; lower is better). No predictor can beat the filter; one that ignores the past and knows only how often each token occurs does worse. A network trained to predict these tokens is judged between the two.");

        ui::see_also(ui, &[("coherence", "what averaging Jacobians keeps"), ("lyapunov", "how nudges grow through layers"), ("transformer", "a transformer, opened up"), ("signal", "signal propagation")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        if self.stale {
            self.refresh();
        }
        // the run that is shown: a few tokens a second
        if self.running && f.dt > 0.0 {
            self.advance(f.dt);
        }
        self.lab_frame(f.real_dt);

        let view = f.view_size();
        let lay = Layout::new(view, f.top_inset);

        // a click on a dial chooses it for study
        if f.clicked_canvas() {
            if let (Some(r), Some(at)) = (lay.panels[RINGS], f.ptr.pos.map(|p| p / f.px)) {
                let cell = r.w() / self.rings() as f32;
                if at.y > r.min.y && at.y < r.max.y && at.x > r.min.x && at.x < r.max.x {
                    let hit = (((at.x - r.min.x) / cell) as usize).min(self.rings() - 1);
                    if hit != self.selected {
                        self.selected = hit;
                        self.refresh();
                    }
                }
            }
        }

        let draw: [fn(&Self, &mut Frame, Rect); PANELS] =
            [Self::draw_rings, Self::draw_strip, Self::draw_chain, Self::draw_cloud, Self::draw_bars, Self::draw_tuning, Self::draw_sweep];
        for (r, draw) in lay.panels.into_iter().zip(draw) {
            if let Some(r) = r {
                draw(self, f, r);
            }
        }
        if lay.left_out > 0 {
            f.text_px(lay.note, lay.note_text(view, view.x - 20.0), 11.0, pal::TEXT, Align2::LEFT_TOP);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene(preset: usize) -> Belief {
        let mut s = Belief::with(preset);
        s.refresh();
        s
    }

    fn listed(s: &Belief) -> Vec<usize> {
        s.listing().iter().map(|d| d.0).collect()
    }

    /// What the "things to try" say is what happens.
    #[test]
    fn the_presets_show_what_they_say() {
        // memory: the longest memory first, every chain straight
        let s = scene(0);
        assert_eq!(listed(&s), [2, 1, 0]);
        assert!(s.theory.iter().all(|t| (t.rho_tau - 1.0).abs() < 1e-3 && (t.rho_c - 1.0).abs() < 1e-3));
        // turning: the still dial first, the one that flips last, with equal magnitudes
        let s = scene(1);
        assert_eq!(listed(&s), [2, 1, 0]);
        assert!((s.theory[0].a - s.theory[2].a).abs() < 1e-4);
        assert_eq!([s.process.latents[0].theta(), s.process.latents[1].theta(), s.process.latents[2].theta()], [PI, PI / 2.0, 0.0]);
        // context: listed by key bias, and nothing left of the fair-coin dial
        let s = scene(2);
        assert_eq!(listed(&s), [2, 1, 0]);
        assert!(s.theory[0].sigma() < 1e-6 && (s.theory[1].rho_c - 0.75).abs() < 1e-4);
        // tuned to π the order of the turning dials reverses
        let s = scene(4);
        assert_eq!(listed(&s), [0, 1, 2]);
        // and in none of them does a finer pattern of a dial outrank its position:
        // the list is then the lens's own order, whichever mode it is taken by
        for p in 0..PRESETS.len() {
            let s = scene(p);
            assert!(s.finer.is_empty(), "preset {p}");
            let by_strongest: Vec<usize> = {
                let mut seen = [false; DIALS];
                s.process.spectrum(Some(s.window), s.omega).iter().filter(|d| !std::mem::replace(&mut seen[d.latent], true)).map(|d| d.latent).collect()
            };
            // (the fair-coin dial of the key sweep has no weight in any mode: its place is a tie)
            let weighty = |order: &[usize]| order.iter().copied().filter(|&i| s.theory[i].sigma() > 1e-6).collect::<Vec<_>>();
            assert_eq!(weighty(&listed(&s)), weighty(&by_strongest), "preset {p}");
        }
    }

    /// The bars, the list, the level lines and the crossings all speak of one weight
    /// per dial (its first mode's), also where a finer pattern of a dial is stronger;
    /// and the list then says that it is.
    #[test]
    fn one_weight_per_dial_everywhere() {
        let mut s = scene(1);
        s.process.q = 0.15;
        s.restart();
        s.refresh();
        // listed by the same numbers the σ bars show
        assert_eq!(listed(&s), [2, 1, 0]);
        for (dial, weight) in s.listing() {
            assert_eq!(weight, s.theory[dial].sigma());
        }
        // dial A flips every token, but its 2-wave pattern stands still, and is stronger
        assert_eq!(s.finer.len(), 1);
        let finer = s.finer[0];
        assert!(finer.latent == 0 && finer.mode == 2 && finer.sigma > 1.5 * s.theory[0].sigma(), "{finer:?}");
        assert!(s.listing_text(2000.0, 11.5).contains("A's 2-wave pattern is stronger"));
        // the crossing marked for the studied dial (B) against A is where the list switches
        let &(_, star, rising) = s.switches.iter().find(|x| x.0 == 0).expect("B crosses A");
        assert!(rising);
        for (lambda, order) in [(star - 0.03, [2, 0, 1]), (star + 0.03, [2, 1, 0])] {
            s.process.latents[1].lambda = lambda;
            s.restart();
            s.refresh();
            assert_eq!(listed(&s), order, "λ = {lambda}");
        }
    }

    /// With a short window and the lens tuned to what flips, more memory first raises
    /// a still dial's weight and then lowers it: two crossings of each level, both marked.
    #[test]
    fn a_weight_that_rises_and_falls_has_two_crossings() {
        let mut s = scene(0);
        (s.process.q, s.window, s.omega) = (0.0, 2, PI);
        s.restart();
        s.refresh();
        for other in [0, 1] {
            let found: Vec<(f32, bool)> = s.switches.iter().filter(|x| x.0 == other).map(|x| (x.1, x.2)).collect();
            assert_eq!(found.len(), 2, "against dial {other}: {found:?}");
            assert!(found[0].1 && !found[1].1 && found[0].0 < found[1].0, "it overtakes, then falls behind: {found:?}");
        }
        // the paper's formula is for the plain lens: not offered here
        assert!(!s.formula_applies(2));
    }

    #[test]
    fn nothing_is_overtaken_at_zero_and_the_formula_keeps_to_its_case() {
        // the key sweep: dial A has no weight at all, so there is no level to cross
        let s = scene(2);
        assert!(s.theory[0].sigma() < 1e-6 && s.switches.iter().all(|x| x.0 != 0), "{:?}", s.switches);
        // the paper's closed form: offered for the crossing preset's still dial ...
        assert!(scene(3).formula_applies(0) && !scene(3).formula_applies(1));
        // ... not where slipping cancels the first mode (two sites, q = 1/4) or reverses it
        for q in [0.25, 0.4] {
            let mut s = scene(0);
            for l in &mut s.process.latents {
                l.n = 2;
            }
            s.process.q = q;
            s.restart();
            s.refresh();
            assert!(!s.formula_applies(2), "q = {q}");
            assert!(s.switches.iter().all(|x| x.1.is_finite() && x.1 > 0.0));
        }
    }

    /// "Last shown n tokens ago" is counted, so it is right however long the wait
    /// (the strip holds only the last eighty tokens).
    #[test]
    fn the_wait_since_the_last_reveal_outlasts_the_strip() {
        let mut s = scene(1);
        s.process.epsilon = 0.05;
        s.restart();
        let (mut count, mut longest) = (s.since, 0);
        for _ in 0..20_000 {
            s.tick();
            count = if s.ticks.back().unwrap().revealed { Some(0) } else { count.map(|n| n + 1) };
            assert_eq!(s.since, count);
            longest = longest.max(count.unwrap_or(0));
        }
        assert!(longest > STRIP, "the longest wait was only {longest}");
    }

    #[test]
    fn words_fit_and_half_a_turn_is_180_degrees() {
        assert_eq!([degrees(PI), degrees(-PI), degrees(-PI / 2.0), degrees(0.0)], [180.0, 180.0, -90.0, 0.0]);
        // dial A of the turning preset flips half a turn: said the same way in both places
        let s = scene(1);
        assert_eq!(degrees(s.process.alpha(0, 1).arg()), 180.0);
        assert_eq!(degrees(s.process.latents[0].theta()), 180.0);
        assert_eq!(s.turn_text(0), "turns 180° a token");
        // and where slipping reverses the forecast (two sites, q = 0.4) the turn that
        // is named is still the dial's own, not the forecast's
        let mut flipped = scene(0);
        for l in &mut flipped.process.latents {
            (l.n, l.v) = (2, 0);
        }
        flipped.process.q = 0.4;
        assert!(flipped.process.diffusion_factor(0, 1) < 0.0);
        assert_eq!(degrees(flipped.process.alpha(0, 1).arg()), 180.0);
        assert_eq!((degrees(flipped.process.latents[0].theta()), flipped.turn_text(0).as_str()), (0.0, "stands still"));
        // the list is cut to the room there is (a phone's panel is about 350 pixels)
        let mut finer = scene(1);
        finer.process.q = 0.15;
        finer.restart();
        finer.refresh();
        for s in (0..PRESETS.len()).map(scene).chain([finer]) {
            for room in [350.0, 420.0, 700.0] {
                let text = s.listing_text(room, 11.5);
                assert!(text_width(&text, 11.5) <= room, "{text:?} in {room}");
            }
        }
    }

    #[test]
    fn the_crossing_is_where_the_text_says() {
        let mut s = scene(3);
        assert_eq!(listed(&s), [1, 0]);
        let (other, at, rising) = s.switches[0];
        assert!(other == 1 && rising);
        assert!((at - 0.65).abs() < 0.005, "λ* = {at}");
        // the paper's closed form gives the same point (the preset's window is long)
        assert!((at - s.formula(1)).abs() < 1e-3);
        s.process.latents[0].lambda = 0.7;
        s.refresh();
        assert_eq!(listed(&s), [0, 1]);
        // Through a short window the scene's own crossing moves, and the formula,
        // which is for an endless one, stays where it was: the paper's 0.649.
        let endless = s.formula(1);
        for (window, least) in [(2, 0.80), (1, 0.90)] {
            s.window = window;
            s.refresh();
            let at = s.switches.iter().find(|x| x.0 == 1).expect("still a crossing").1;
            assert!(at > least, "window {window}: λ* = {at}");
            assert!((s.formula(1) - endless).abs() < 1e-6 && (endless - 0.649).abs() < 0.002, "window {window}: {}", s.formula(1));
        }
    }

    /// The chain drawn link by link is the filter's own response, and it is what the
    /// faint full chain predicts.
    #[test]
    fn the_live_chain_follows_the_predicted_one() {
        for preset in 0..PRESETS.len() {
            let mut s = scene(preset);
            for _ in 0..300 {
                s.tick();
                for i in 0..s.rings() {
                    if i != s.selected {
                        continue;
                    }
                    let sign = if s.reversed[i] { -1.0 } else { 1.0 };
                    let sum = s.live_links(i).iter().fold(C32::ZERO, |a, k| a + *k);
                    let links = s.live_links(i).len();
                    if links < s.sums.points.len() {
                        assert!((sum - s.sums.points[links].scale(sign)).abs() < 1e-4, "preset {preset}, {links} links");
                    }
                }
            }
        }
    }

    /// The picture's own numbers: the cloud's average is σ, the average distance of
    /// its dots is a ρ_τ, and the chances add up to one.
    #[test]
    fn the_cloud_draws_the_factors() {
        for preset in 0..PRESETS.len() {
            let s = scene(preset);
            let t = s.theory[s.selected];
            let total: f32 = s.sums.chance.iter().sum();
            assert!((total - 1.0).abs() < 1e-4);
            let (mut mean, mut distance) = (C32::ZERO, 0.0);
            for (p, &c) in s.sums.points.iter().zip(&s.sums.chance) {
                mean = mean + p.scale(c * (1.0 - 2.0 * s.sums.reversed));
                distance += c * p.abs();
            }
            assert!((mean.abs() - t.sigma()).abs() < 2e-3, "preset {preset}: {} against {}", mean.abs(), t.sigma());
            assert!((distance - t.a * t.rho_tau).abs() < 2e-3, "preset {preset}");
        }
    }

    /// How long the measuring run takes per token (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn cost() {
        for preset in [1, 2] {
            let mut s = scene(preset);
            let start = std::time::Instant::now();
            s.measure(600_000);
            println!("preset {preset}: {:.0} ns per token of the measuring run", start.elapsed().as_secs_f64() * 1e9 / 600_000.0);
            let start = std::time::Instant::now();
            for _ in 0..2000 {
                s.refresh();
            }
            println!("preset {preset}: {:.3} ms to work the theory out again", start.elapsed().as_secs_f64() * 1e3 / 2000.0);
        }
    }

    /// Whatever the sliders are set to, the numbers stay finite and the pictures have
    /// something to draw.
    #[test]
    fn no_setting_breaks_the_scene() {
        let mut s = scene(1);
        let mut rng = Rng::new(9);
        for trial in 0..400 {
            let n = SIZES[rng.below(4)];
            let count = 2 + rng.below(2);
            s.process.latents = (0..count).map(|_| Latent { n, lambda: rng.range(0.3, 0.99), v: rng.below(n), psi: rng.bool(0.3).then(|| rng.range(0.0, 1.0)) }).collect();
            s.process.q = [0.0, 0.25, 0.45, rng.range(0.0, 0.45)][rng.below(4)];
            s.process.epsilon = [0.05, 0.6, rng.range(0.05, 0.6)][rng.below(3)];
            s.window = [1, 2, 7, 128, 256][rng.below(5)];
            s.omega = [0.0, PI, rng.range(0.0, PI)][rng.below(3)];
            s.selected = rng.below(3);
            s.restart();
            s.refresh();
            for _ in 0..30 {
                s.tick();
            }
            s.measure(300);
            assert!(s.sums.points.len() >= 2 && s.sums.drawn >= 1 && s.sums.drawn < s.sums.points.len(), "trial {trial}");
            assert!((s.sums.chance.iter().sum::<f32>() - 1.0).abs() < 1e-3, "trial {trial}");
            for i in 0..s.rings() {
                let (t, m) = (s.theory[i], s.measured[i].factors());
                for v in [t.a, t.rho_tau, t.rho_c, m.a, m.rho_tau, m.rho_c] {
                    assert!(v.is_finite() && v >= 0.0, "trial {trial}: {t:?} {m:?}");
                }
                assert!(t.rho_tau <= 1.0 + 1e-4 && t.rho_c <= 1.0 + 1e-4, "trial {trial}: {t:?}");
                assert!(s.tuned[i].iter().chain(&s.sweep).all(|v| v.is_finite()), "trial {trial}");
                assert!(s.live_links(i).len() <= s.window);
            }
        }
    }

    #[test]
    fn the_measuring_run_reaches_the_theory() {
        for preset in 0..PRESETS.len() {
            let mut s = scene(preset);
            while s.measured[0].contexts < LAB_CONTEXTS {
                s.measure(LAB_TOKENS);
            }
            for i in 0..s.rings() {
                let (t, m) = (s.theory[i], s.measured[i].factors());
                assert!((t.a - m.a).abs() < 0.03 * t.a.max(0.3), "preset {preset} ring {i}: a {} against {}", m.a, t.a);
                assert!((t.rho_tau - m.rho_tau).abs() < 0.02 && (t.rho_c - m.rho_c).abs() < 0.03, "preset {preset} ring {i}: {m:?} against {t:?}");
            }
        }
    }

    /// The measuring run (and so how fast the white marks settle) keeps to the clock,
    /// not to the display: the same tokens in a second at 30 frames a second as at 240.
    #[test]
    fn the_measuring_run_keeps_time_whatever_the_frame_rate() {
        let mut tokens = Vec::new();
        for hz in [30.0f32, 60.0, 144.0, 240.0] {
            let mut s = scene(1);
            for _ in 0..hz as usize {
                s.lab_frame(1.0 / hz);
            }
            tokens.push(s.lab_tokens);
        }
        assert!(tokens.iter().all(|&t| (t as f32 - LAB_RATE).abs() <= 2.0 * LAB_BATCH as f32), "tokens in a second: {tokens:?}");
    }

    /// What the key's term says: a glimpse shows the dial half a turn away exactly
    /// when the token just before it carried −1, whatever the keys before that.
    #[test]
    fn a_glimpse_goes_by_the_key_of_the_token_just_before_it() {
        let p = Process { latents: vec![Latent { n: 8, lambda: 0.9, v: 1, psi: Some(0.0) }], q: 0.1, epsilon: 0.3 };
        let mut rng = Rng::new(4);
        let mut chain = Chain::new(&p, &mut rng);
        // each token so far: whether it was a glimpse, and whether its key was −1
        let mut past: Vec<(bool, bool)> = Vec::new();
        let (mut turned, mut not_after_a_gap) = (0, 0);
        for _ in 0..20_000 {
            let x = chain.x[0];
            let (token, revealed) = chain.emit(&p, &mut rng);
            if revealed {
                let mut shown = [0usize];
                p.shown(token, &mut shown);
                let half = shown[0] == (x + 4) % 8;
                assert!(half || shown[0] == x);
                let before = past.last().copied();
                assert_eq!(half, before.is_some_and(|b| b.1));
                turned += half as usize;
                // a −1, then a token that shows nothing and carries +1: not turned
                if past.len() >= 2 && past[past.len() - 2].1 && before == Some((false, false)) {
                    not_after_a_gap += 1;
                }
            }
            past.push((revealed, p.key_minus(token, 0)));
        }
        assert!(turned > 500 && not_after_a_gap > 300, "{turned} turned, {not_after_a_gap} not turned after a gap");
    }

    /// Put the numbers the panel gives into the formula it gives and out comes the
    /// weight it gives, also for a dial that turns backwards and for one whose
    /// forecast slipping reverses; and the tuning chart peaks at that turn φ, where
    /// its ▲ is.
    #[test]
    fn the_closed_form_is_what_its_numbers_give() {
        let mut backwards = scene(1);
        (backwards.process.latents[0].v, backwards.omega, backwards.selected) = (6, PI / 2.0, 0);
        let mut reversed = scene(0);
        for l in &mut reversed.process.latents {
            l.n = 2;
        }
        (reversed.process.q, reversed.selected) = (0.4, 1);
        let mut tuned = Belief { process: reversed.process.clone(), ..scene(0) };
        (tuned.omega, tuned.selected) = (PI, 1);
        for (case, (mut s, want)) in [(backwards, 1.225), (reversed, 0.163), (tuned, 0.411)].into_iter().enumerate() {
            s.restart();
            s.refresh();
            let i = s.selected;
            let [shrink, phi, memory, endless] = s.closed_form(i);
            // (the dial's own turn θ is not the angle the formula takes)
            assert_ne!(format!("{:.0}°", degrees(s.process.latents[i].theta())), phi, "case {case}");
            let number = |t: &str| t.trim_end_matches('°').parse::<f32>().unwrap();
            let (a, phi, r, sigma) = (number(&shrink), number(&phi).to_radians(), number(&memory), number(&endless));
            let psi = s.process.latents[i].psi.unwrap_or(1.0);
            let formula = psi.abs() * a * r / (C32::ONE - C32::polar(r, phi - s.omega)).abs();
            assert!((formula - sigma).abs() < 0.006 && (sigma - want).abs() < 1e-3, "case {case}: {formula} by the formula, {sigma} shown");
            let curve = &s.tuned[i];
            let peak = (0..=CURVE).max_by(|&a, &b| curve[a].total_cmp(&curve[b])).unwrap();
            assert!((PI * peak as f32 / CURVE as f32 - phi).abs() <= PI / CURVE as f32, "case {case}: peak at {peak}");
            assert!((s.process.turn(i, 1).abs() - phi).abs() < 0.01, "case {case}: ▲ at {}", s.process.turn(i, 1));
        }
    }

    /// Every window from a small phone's to a large screen: each panel shown has the
    /// room it needs, nothing is drawn upside down or over another panel, and the
    /// chains' map is finite and keeps them inside their panels. Panels left out are
    /// said to be, in a line under the others.
    #[test]
    fn every_window_gets_panels_that_fit() {
        // the published pictures: the view in a window of 1600 by 1000 and of 1100 by
        // 700 with the panels docked, and a phone's, keep every panel where it was
        for (view, inset) in [(Vec2::new(1078.0, 1000.0), 0.0), (Vec2::new(578.0, 700.0), 0.0), (Vec2::new(390.0, 800.0), 46.0)] {
            let (lay, top) = (Layout::new(view, inset), inset.max(40.0));
            let shared = Layout::shared(view, top, view.y - top - 12.0);
            assert!(lay.panels.iter().zip(shared).all(|(a, b)| a.is_some_and(|a| a.min == b.min && a.max == b.max)), "{view}");
        }
        let grid = (240..=2400).step_by(13).flat_map(|w| (180..=1400).step_by(11).map(move |h| (w, h)));
        let named = [(240, 180), (800, 360), (520, 480), (375, 548), (844, 390), (390, 800), (1600, 1000), (1100, 700), (2400, 1400)];
        for (w, h) in grid.chain(named) {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w as f32, h as f32);
                let lay = Layout::new(view, inset);
                let shown: Vec<(usize, Rect)> = lay.panels.iter().enumerate().filter_map(|(p, r)| r.map(|r| (p, r))).collect();
                assert_eq!(lay.left_out, PANELS - shown.len());
                for &(p, r) in &shown {
                    let at = format!("{w}x{h} (inset {inset}), panel {p}: {r:?}");
                    assert!(r.w() >= LEAST[p].x && r.h() >= LEAST[p].y, "{at}");
                    assert!(r.min.x >= 0.0 && r.min.y >= inset && r.max.x <= view.x && r.max.y <= view.y, "{at}");
                    let apart = |s: Rect| r.max.x <= s.min.x || s.max.x <= r.min.x || r.max.y <= s.min.y || s.max.y <= r.min.y;
                    assert!(shown.iter().all(|&(o, s)| o == p || apart(s)), "{at}");
                    match p {
                        TUNING | SWEEP => {
                            let chart = Belief::plot_area(r);
                            assert!(chart.h() >= 40.0 && chart.w() >= 40.0 && chart.max.y <= r.max.y, "{at}");
                        }
                        BARS => {
                            let (_, top, base) = Belief::bar_span(r);
                            assert!(base - top >= 34.0 && base <= r.max.y, "{at}");
                        }
                        CHAIN | CLOUD => {
                            let margin = if p == CHAIN { 52.0 } else { 48.0 };
                            let boxes = [(Vec2::ZERO, Vec2::new(2.0, 0.0)), (Vec2::ZERO, Vec2::ZERO), (Vec2::new(0.0, -3.0), Vec2::ZERO), (Vec2::new(-1.0, -0.5), Vec2::new(1.5, 0.7))];
                            for (lo, hi) in boxes {
                                let to = Belief::plane(lo, hi, r, margin, margin);
                                let scale = (to(C32::ONE) - to(C32::ZERO)).x;
                                assert!(scale.is_finite() && scale > 0.0, "{at}: scale {scale}");
                                for corner in [lo, hi] {
                                    let q = to(C32::new(corner.x, corner.y));
                                    assert!(q.x >= r.min.x && q.x <= r.max.x && q.y >= r.min.y && q.y <= r.max.y, "{at}: {corner} drawn at {q}");
                                }
                            }
                        }
                        _ => {}
                    }
                }
                if lay.left_out > 0 {
                    let text = lay.note_text(view, view.x - 20.0);
                    assert!(shown.iter().all(|s| s.1.max.y <= lay.note.y) && lay.note.y + 14.0 <= view.y && text_width(&text, 11.0) <= view.x - 20.0, "{w}x{h}: {text}");
                }
            }
        }
        // a phone on its side shows the dials and the factors, at the least
        let side = Layout::new(Vec2::new(844.0, 390.0), 46.0);
        assert!([RINGS, CHAIN, CLOUD, BARS].iter().all(|&p| side.panels[p].is_some()) && side.left_out > 0);
    }

    /// The dials are drawn turning to where each token puts them, not jumping there:
    /// at 3 tokens a second on displays of 60 and 144 frames a second, no dial drawn
    /// turns in a frame by more than a third of what the dial itself turned at the
    /// token, and halfway to the next token each is drawn exactly where it is.
    #[test]
    fn the_dials_are_drawn_turning_to_where_each_token_puts_them() {
        // (the turning dials: a half turn, a quarter turn, none, each token)
        let turning = PRESETS.iter().position(|p| p.label.starts_with("turning")).unwrap();
        for hz in [60.0f32, 144.0] {
            let mut s = scene(turning);
            let dt = 1.0 / hz;
            let k = s.rings();
            let mut drawn: Vec<f32> = (0..k).map(|i| s.dial_shown(i)).collect();
            let (mut most, mut moved) = (0.0f32, 0.0f32);
            for _ in 0..(4.0 * hz) as usize {
                let was = s.chain.x.clone();
                s.advance(dt);
                for i in 0..k {
                    let now = s.dial_shown(i);
                    // (the shorter way round, as the dial is drawn turning)
                    let apart = |a: f32, b: f32| (PI - (PI - (a - b)).rem_euclid(TAU)).abs();
                    let n = s.process.latents[i].n as f32;
                    moved = moved.max(apart(TAU * was[i] as f32 / n, TAU * s.chain.x[i] as f32 / n));
                    most = most.max(apart(now, drawn[i]));
                    drawn[i] = now;
                    if s.clock >= GLIDE_SHARE {
                        assert_eq!(now, TAU * s.chain.x[i] as f32 / n, "{hz} Hz: dial {i} not where it is");
                    }
                }
            }
            assert!(moved > 1.0 && most < moved / 3.0, "{hz} Hz: drawn turning {most} in a frame, the dials {moved} at a token");
        }
    }
}

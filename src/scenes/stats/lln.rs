//! The law of large numbers: the average of independent draws settling on the mean,
//! its typical distance from the mean shrinking like σ/√n, what the law does not say
//! (that a count catches up after a run of bad luck), and a source with no mean.

use super::plot::{self, Glide, Plot, Rect, AXES, CURVES, MARKS, OWN};
use super::*;
use crate::math::stats::{self as st, Dist};

pub const DEF: SceneDef = SceneDef {
    id: "lln",
    title: "The law of large numbers",
    group: CHANCE,
    blurb: "Averages settle on the mean as a sample grows, at the pace of one over the square root of its size, unless the tails are too heavy for there to be a mean.",
    make: |_| Box::new(Lln::create()),
};

// ----- where the draws come from ----------------------------------------------------------

/// How many draws a run has.
const LONGEST: usize = 10_000;
/// How many tails the coin's runs begin with, when they are made to.
const TAILS: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Source {
    Die,
    Coin,
    Wait,
    Humps,
    Cauchy,
    Darts,
}

const SOURCES: [Source; 6] = [Source::Die, Source::Coin, Source::Wait, Source::Humps, Source::Cauchy, Source::Darts];

/// The two humps: half the draws from each.
const HUMPS: [Dist; 2] = [Dist::Normal { mean: -2.0, sd: 0.5 }, Dist::Normal { mean: 2.0, sd: 0.5 }];
/// The waiting time for something that happens once a unit of time on average.
const WAIT: Dist = Dist::Exponential { rate: 1.0 };
const CAUCHY: Dist = Dist::Cauchy { centre: 0.0, scale: 1.0 };
/// A dart thrown at random into a square lands inside the circle drawn in it with the
/// circle's share of the square's area.
const INSIDE: f64 = std::f64::consts::FRAC_PI_4;
const FACES: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

impl Source {
    /// The name on its button.
    fn name(self) -> &'static str {
        match self {
            Source::Die => "die",
            Source::Coin => "coin",
            Source::Wait => "wait",
            Source::Humps => "two humps",
            Source::Cauchy => "Cauchy",
            Source::Darts => r"darts for $\pi$",
        }
    }

    /// The mean of one draw, where there is one. (The die and the two humps are not
    /// among `Dist`'s distributions: their moments are worked out from what they are
    /// made of.)
    fn mean(self, p: f64) -> Option<f64> {
        match self {
            Source::Die => Some(FACES.iter().sum::<f64>() / 6.0),
            Source::Coin => Dist::Bernoulli { p }.mean(),
            Source::Wait => WAIT.mean(),
            Source::Humps => Some(0.5 * (HUMPS[0].mean()? + HUMPS[1].mean()?)),
            Source::Cauchy => CAUCHY.mean(),
            Source::Darts => Dist::Bernoulli { p: INSIDE }.mean().map(|m| 4.0 * m),
        }
    }

    /// The standard deviation of one draw, where it is finite.
    fn sd(self, p: f64) -> Option<f64> {
        match self {
            Source::Die => {
                let m = self.mean(p)?;
                Some((FACES.iter().map(|k| (k - m) * (k - m)).sum::<f64>() / 6.0).sqrt())
            }
            Source::Coin => Dist::Bernoulli { p }.sd(),
            Source::Wait => WAIT.sd(),
            Source::Humps => {
                // (the mean square of a mixture is the mixture of its parts' mean squares)
                let m = self.mean(p)?;
                let square = 0.5 * (HUMPS[0].variance()? + HUMPS[0].mean()?.powi(2)) + 0.5 * (HUMPS[1].variance()? + HUMPS[1].mean()?.powi(2));
                Some((square - m * m).max(0.0).sqrt())
            }
            Source::Cauchy => CAUCHY.sd(),
            Source::Darts => Dist::Bernoulli { p: INSIDE }.sd().map(|s| 4.0 * s),
        }
    }

    /// The density of one draw (for those with one).
    fn density(self, x: f64) -> f64 {
        match self {
            Source::Wait => WAIT.pdf(x),
            Source::Humps => 0.5 * (HUMPS[0].pdf(x) + HUMPS[1].pdf(x)),
            Source::Cauchy => CAUCHY.pdf(x),
            _ => f64::NAN,
        }
    }

    fn draw(self, p: f64, rng: &mut Rng) -> f64 {
        match self {
            Source::Die => 1.0 + (st::unit(rng) * 6.0).floor().min(5.0),
            Source::Coin => Dist::Bernoulli { p }.sample(rng),
            Source::Wait => WAIT.sample(rng),
            Source::Humps => HUMPS[(st::unit(rng) < 0.5) as usize].sample(rng),
            Source::Cauchy => CAUCHY.sample(rng),
            Source::Darts => {
                let (u, v) = throw(rng);
                if u * u + v * v <= 1.0 {
                    4.0
                } else {
                    0.0
                }
            }
        }
    }

    /// The `n`th draw of a run (from 1): the coin's first ten are tails when `tails`.
    fn value(self, p: f64, tails: bool, n: usize, rng: &mut Rng) -> f64 {
        if tails && self == Source::Coin && n <= TAILS {
            0.0
        } else {
            self.draw(p, rng)
        }
    }

    /// The range of averages the main chart shows.
    fn range(self) -> [f64; 2] {
        match self {
            Source::Die => [1.0, 6.0],
            Source::Coin => [0.0, 1.0],
            Source::Wait => [0.0, 3.0],
            Source::Humps => [-4.0, 4.0],
            Source::Cauchy => [-6.0, 6.0],
            Source::Darts => [0.0, 4.0],
        }
    }

    /// The bins the bold run's draws are counted in: one for each face of the die and
    /// each side of the coin, and for the darts outside and inside.
    fn bins(self) -> st::Histogram {
        match self {
            Source::Die => st::Histogram::new(0.5, 6.5, 6),
            Source::Coin => st::Histogram::new(-0.5, 1.5, 2),
            Source::Wait => st::Histogram::new(0.0, 6.0, 30),
            Source::Humps => st::Histogram::new(-4.5, 4.5, 36),
            Source::Cauchy => st::Histogram::new(-8.0, 8.0, 32),
            Source::Darts => st::Histogram::new(-2.0, 6.0, 2),
        }
    }

    /// What one draw counts in the total that the second chart follows: the darts'
    /// draws count 4 for a point inside, and that chart counts points.
    fn unit(self) -> f64 {
        if self == Source::Darts {
            4.0
        } else {
            1.0
        }
    }
}

/// A point thrown at random into the square from −1 to 1 each way.
fn throw(rng: &mut Rng) -> (f64, f64) {
    (2.0 * st::unit(rng) - 1.0, 2.0 * st::unit(rng) - 1.0)
}

/// The n at which each run's average is kept: every n up to 100, then about a hundred
/// and fifty to a tenfold, at round numbers (so that 200, 250, 400 and 1 000 are among
/// them), up to 10 000.
fn grid() -> &'static [u32] {
    static GRID: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    GRID.get_or_init(|| {
        let mut g: Vec<u32> = (1..=100).collect();
        let mantissas = (100..200).step_by(2).chain((200..400).step_by(5)).chain((400..1000).step_by(10));
        for decade in [1u32, 10] {
            for m in mantissas.clone() {
                let n = m * decade;
                if g.last().is_none_or(|last| n > *last) {
                    g.push(n);
                }
            }
        }
        g.push(LONGEST as u32);
        g
    })
}

/// The place in the grid nearest to n.
fn grid_index(n: f64) -> usize {
    let g = grid();
    let i = g.partition_point(|&m| (m as f64) < n);
    if i == 0 {
        0
    } else if i >= g.len() {
        g.len() - 1
    } else if (g[i] as f64 - n) < (n - g[i - 1] as f64) {
        i
    } else {
        i - 1
    }
}

/// A run's averages at the n of the grid: every draw is made and added, and the
/// average kept where the grid has a place.
fn averages(source: Source, p: f64, tails: bool, seed: u64) -> Vec<f64> {
    let grid = grid();
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(grid.len());
    let mut sum = 0.0;
    for n in 1..=LONGEST {
        sum += source.value(p, tails, n, &mut rng);
        if grid.get(out.len()) == Some(&(n as u32)) {
            out.push(sum / n as f64);
        }
    }
    out
}

fn run_seed(seed: u64, k: usize) -> u64 {
    seed.wrapping_mul(1_000_003).wrapping_add(k as u64)
}

fn bold_seed(seed: u64, k: u64) -> u64 {
    seed.wrapping_mul(1_000_003).wrapping_add(500_000).wrapping_add(k)
}

/// The share of a bell outside ±z is 5 %: the funnel's width in standard errors.
fn z95() -> f64 {
    st::normal_quantile(0.975)
}

// ----- the run being drawn -----------------------------------------------------------------

/// The most draws the bold run makes in one frame (far more than its pace asks for, but
/// for a frame after a long stall: it then falls behind instead of holding the frame up).
const MOST_DRAWS: usize = 5000;
/// The most of the darts' throws drawn.
const THROWS_DRAWN: usize = 5000;

struct Bold {
    rng: Rng,
    /// Draws made, their sum, and the newest of them.
    n: usize,
    sum: f64,
    last: f64,
    /// The averages at the n of the grid reached so far.
    at: Vec<f64>,
    /// How far along it is drawn: between n − 1 and n (see `head`).
    p: f64,
    /// Its draws, counted by face, by side or in bins.
    hist: st::Histogram,
    /// The darts' first throws, and whether each landed inside.
    throws: Vec<(f32, f32, bool)>,
}

impl Bold {
    /// A run with its first draw made: an average of one.
    fn new(source: Source, p: f64, tails: bool, seed: u64) -> Bold {
        let mut b = Bold { rng: Rng::new(seed), n: 0, sum: 0.0, last: 0.0, at: Vec::new(), p: 1.0, hist: source.bins(), throws: Vec::new() };
        b.step(source, p, tails);
        b
    }

    fn step(&mut self, source: Source, p: f64, tails: bool) {
        let n = self.n + 1;
        let x = if source == Source::Darts {
            let (u, v) = throw(&mut self.rng);
            let inside = u * u + v * v <= 1.0;
            if self.throws.len() < THROWS_DRAWN {
                self.throws.push((u as f32, v as f32, inside));
            }
            if inside {
                4.0
            } else {
                0.0
            }
        } else {
            source.value(p, tails, n, &mut self.rng)
        };
        (self.n, self.last) = (n, x);
        self.sum += x;
        self.hist.add(x);
        if grid().get(self.at.len()) == Some(&(n as u32)) {
            self.at.push(self.sum / n as f64);
        }
    }

    fn done(&self) -> bool {
        self.p >= LONGEST as f64
    }

    /// Draws a second at this point of the run: a run takes `seconds` to cross the
    /// chart evenly, so on a logarithmic axis they come faster as n grows.
    fn rate(&self, seconds: f32, log: bool) -> f64 {
        let span = seconds.max(0.5) as f64;
        if self.done() {
            0.0
        } else if log {
            self.p * (LONGEST as f64).ln() / span
        } else {
            (LONGEST as f64 - 1.0) / span
        }
    }

    /// Go `dt` seconds of the scene's time further, making the draws that takes.
    fn advance(&mut self, dt: f32, seconds: f32, log: bool, source: Source, p: f64, tails: bool) {
        if self.done() || dt <= 0.0 {
            return;
        }
        let (end, span, dt) = (LONGEST as f64, seconds.max(0.5) as f64, dt as f64);
        let to = if log { self.p * (end.ln() * dt / span).exp() } else { self.p + (end - 1.0) * dt / span }.min(end);
        let need = (to.ceil() as usize).min(LONGEST);
        let mut made = 0;
        while self.n < need && made < MOST_DRAWS {
            self.step(source, p, tails);
            made += 1;
        }
        self.p = if self.n < need { self.n as f64 } else { to };
    }

    /// Where it is drawn: how far along n, the average there, and the total less n μ
    /// there. The newest draw is drawn coming in: the head moves along the line from
    /// the average before it to the average with it, as the scene's time goes from
    /// when the draw was due to when the next one is, so that it moves in every frame
    /// however the draws fall into frames.
    fn head(&self, mean: f64) -> (f64, f64, f64) {
        let n = self.n.max(1) as f64;
        let t = (self.p - (n - 1.0)).max(0.0).min(1.0);
        let before = self.sum - self.last;
        let (avg0, avg1) = if self.n > 1 { (before / (n - 1.0), self.sum / n) } else { (self.sum, self.sum) };
        let (gap0, gap1) = (before - (n - 1.0) * mean, self.sum - n * mean);
        let gap0 = if self.n > 1 { gap0 } else { gap1 };
        (self.p.max(1.0), avg0 + (avg1 - avg0) * t, gap0 + (gap1 - gap0) * t)
    }

    fn average(&self) -> f64 {
        self.sum / self.n.max(1) as f64
    }
}

// ----- the experiments ----------------------------------------------------------------------

struct Trial {
    button: &'static str,
    caption: &'static str,
    source: Source,
    p: f64,
    tails: bool,
    runs: usize,
    marks: [u32; 2],
    seed: u64,
}

const TRIALS: [Trial; 5] = [
    Trial {
        button: "a die",
        caption: r"Roll a die again and again, keeping the average of the rolls so far. At first it jumps with each roll; then it settles on the mean, 3.5, as the green funnel narrows. Its typical distance from 3.5 is $\sigma/\sqrt{n}$: 0.54 after 10 rolls and 0.054 after 1 000 (the markers; the table measures it over the thin runs). A hundred times the rolls, a tenth of the distance. By 1 000 rolls over 95 % of runs are within 0.11 of 3.5.",
        source: Source::Die,
        p: 0.5,
        tails: false,
        runs: 12,
        marks: [10, 1000],
        seed: 7,
    },
    Trial {
        button: "many runs",
        caption: r"Two hundred runs fill the funnel: at each $n$ past the first few, about 95 % of runs are inside it (here 191 and 193 of the 200 at the markers). Four times the rolls, half the distance: the typical distance from 3.5 is 0.17 at $n = 100$ and 0.085 at $n = 400$. And the funnel is about each $n$ alone: 14 of these runs are inside it at one marker and outside at the other.",
        source: Source::Die,
        p: 0.5,
        tails: false,
        runs: 200,
        marks: [100, 400],
        seed: 4,
    },
    Trial {
        button: "after ten tails",
        caption: r"Every run begins with ten tails; after them the coin is fair, and heads still comes half the time: it remembers nothing. The share of heads (top) climbs back to one half only because the ten tails become a smaller and smaller part of the tosses. The count does not catch up: heads minus half the tosses (below) starts 5 short and wanders: it comes back to 0 again and again, yet its typical distance from 0 grows, to 6.9 at 100 tosses and 50 at 10 000, and on average it stays 5 short.",
        source: Source::Coin,
        p: 0.5,
        tails: true,
        runs: 12,
        marks: [100, 10_000],
        seed: 13,
    },
    Trial {
        button: "no mean",
        caption: r"The Cauchy distribution has no mean, and the law needs one. Its averages never settle: now and then a single draw lands so far out that it throws the average somewhere else, at any $n$ (two runs in three still move by more than 1 between draws 1 000 and 10 000). An average of 1 000 draws is spread exactly as one draw is: the middle half of the runs is about 2 wide at both markers.",
        source: Source::Cauchy,
        p: 0.5,
        tails: false,
        runs: 200,
        marks: [1, 1000],
        seed: 11,
    },
    Trial {
        button: r"$\pi$ by chance",
        caption: r"Throw points at random into a square with a circle in it: each lands inside with probability $\pi/4$, the circle's share of the area. Count 4 for a point inside and 0 for one outside, and the average closes in on $\pi$: its typical error is 0.16 after 100 points and 0.016 after 10 000. A hundred times the points, a tenth of the error. An area worked out as an average of random draws: the Monte Carlo method.",
        source: Source::Darts,
        p: 0.5,
        tails: false,
        runs: 12,
        marks: [100, 10_000],
        seed: 18,
    },
];

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;
/// How long a run takes to cross the chart, unless the visitor says otherwise (seconds).
const SECONDS: f32 = 12.0;
/// The most runs, and how many come in a second of the scene's time while they are
/// being made (each is ten thousand draws).
const MOST_RUNS: usize = 200;
const RUN_RATE: f32 = 400.0;
/// How long a new run takes to fade in (seconds of the scene's time).
const FADE: f32 = 0.3;

// ----- the scene ----------------------------------------------------------------------------

/// A run already made: its averages at the n of the grid, and how long it has been
/// there.
struct Run {
    avg: Vec<f64>,
    age: f32,
}

/// What the runs depend on: when it changes they are made again.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Key {
    source: Source,
    p: u64,
    tails: bool,
    seed: u64,
}

/// The thin runs' lines as last laid out (see `Lln::thin_runs`), and what for.
#[derive(Default)]
struct Laid {
    made_for: Option<LaidFor>,
    main: Vec<LinePoint>,
    gap: Vec<LinePoint>,
}

#[derive(Clone, Copy, PartialEq)]
struct LaidFor {
    main: Rect,
    gap: Option<Rect>,
    t: u64,
    runs: usize,
    made_for: Option<Key>,
    hud: [[Vec3; 3]; 3],
}

/// What glides: the bold run's bars, the markers (where they are along n, as the
/// logarithm of n), and the axis (0 logarithmic, 1 ordinary).
#[derive(Default)]
struct Look {
    bars: Glide,
    marks: Glide,
    axis: Glide,
    /// Where each run crosses each marker, as drawn.
    dots: [Dots; 2],
    /// Where the markers' names are across the main chart, and how many lines up each
    /// is (the second goes up a line when the two would cover each other).
    label_x: Glide,
    label_up: Glide,
    /// The bold run's head as drawn (the logarithm of n, the average, the total less
    /// n μ), a point half-way there that it follows, and which run it is the head of.
    head: Option<[f64; 3]>,
    toward: [f64; 3],
    head_of: Option<(Key, u64)>,
}

/// The averages of the runs where they cross a marker, as drawn, a point half-way to
/// each that it follows, and the runs they are of (the key they were made for, and how
/// many there are).
#[derive(Default)]
struct Dots {
    at: Vec<f64>,
    toward: Vec<f64>,
    of: Option<(Option<Key>, usize)>,
}

/// How quickly the head drawn follows the run: through two eases one after the other,
/// each this long (seconds: the time to go two thirds of the way). The run itself is a
/// zigzag, a corner at every draw, and n goes evenly; a head drawn exactly at its end
/// turns each corner in a frame and changes speed there, several times a second over
/// the first hundred draws. Eased, it rounds the corners off and its speed changes
/// smoothly, and it is never more than a few pixels behind.
const HEAD_EASE: f64 = 0.03;
/// The same for the dots where the runs cross a marker being dragged (see
/// `Lln::ease_dots`): the hand moves slowly, and the corners come closer together.
const DOT_EASE: f64 = 0.05;

pub struct Lln {
    source: Source,
    /// The coin's chance of heads, and whether its runs begin with ten tails.
    p: f64,
    tails: bool,
    /// How many thin runs there are to be.
    wanted: usize,
    /// How long the bold run takes to cross the chart (seconds).
    seconds: f32,
    log: bool,
    seed: u64,
    /// How many bold runs have been started since the seed was set.
    bolds: u64,
    runs: Vec<Run>,
    made_for: Option<Key>,
    pace: Pace,
    bold: Bold,
    bold_for: Option<(Key, u64)>,
    /// The two markers' places in the grid, and where each is while it is held (the
    /// logarithm of n).
    marks: [usize; 2],
    held: [Option<f32>; 2],
    look: Look,
    laid: Laid,
    /// Is anything still to change without the visitor doing anything?
    moving: bool,
}

impl Lln {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Lln {
        let t = &TRIALS[OPENING];
        let mut s = Lln {
            source: t.source,
            p: t.p,
            tails: t.tails,
            wanted: t.runs,
            seconds: SECONDS,
            log: true,
            seed: t.seed,
            bolds: 0,
            runs: Vec::new(),
            made_for: None,
            pace: Pace::default(),
            bold: Bold::new(t.source, t.p, t.tails, bold_seed(t.seed, 0)),
            bold_for: None,
            marks: t.marks.map(|n| grid_index(n as f64)),
            held: [None; 2],
            look: Look::default(),
            laid: Laid::default(),
            moving: true,
        };
        s.refresh();
        s
    }

    /// Set up one of the experiments in `TRIALS`: every setting its lesson depends on,
    /// and the bold run from its first draw.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        (self.source, self.p, self.tails, self.wanted, self.seed, self.bolds) = (t.source, t.p, t.tails, t.runs, t.seed, 0);
        (self.seconds, self.log) = (SECONDS, true);
        self.marks = t.marks.map(|n| grid_index(n as f64));
        // (the markers go to their places at once: the runs are made anew, and markers
        // gliding there would carry their dots along paths that are not yet drawn)
        (self.look.marks, self.look.label_x, self.look.label_up) = (Glide::default(), Glide::default(), Glide::default());
        self.bold_for = None;
        self.refresh();
    }

    fn key(&self) -> Key {
        Key { source: self.source, p: self.p.to_bits(), tails: self.tails, seed: self.seed }
    }

    /// Make the runs again if what they depend on has changed, and keep as many as are
    /// wanted.
    fn refresh(&mut self) {
        self.p = if self.p.is_finite() { self.p.max(0.0).min(1.0) } else { 0.5 };
        self.wanted = self.wanted.clamp(1, MOST_RUNS);
        let key = self.key();
        if self.made_for != Some(key) {
            self.runs.clear();
            self.pace.reset();
            self.made_for = Some(key);
        }
        self.runs.truncate(self.wanted);
        if self.bold_for != Some((key, self.bolds)) {
            self.bold = Bold::new(self.source, self.p, self.tails, bold_seed(self.seed, self.bolds));
            self.bold_for = Some((key, self.bolds));
        }
    }

    /// Make the next thin run.
    fn add_run(&mut self) {
        let k = self.runs.len();
        self.runs.push(Run { avg: averages(self.source, self.p, self.tails, run_seed(self.seed, k)), age: 0.0 });
    }

    /// The runs owed and the bold run's draws, for `dt` seconds of the scene's time.
    fn work(&mut self, dt: f32, real_dt: f32) {
        if dt <= 0.0 {
            return;
        }
        for r in &mut self.runs {
            r.age += dt;
        }
        if self.runs.len() < self.wanted {
            let mut turn = scene_turn(&mut self.pace, RUN_RATE, dt, real_dt);
            while self.runs.len() < self.wanted && turn.next() {
                self.add_run();
            }
            self.pace.end(turn);
        } else {
            self.pace.reset();
        }
        self.bold.advance(dt, self.seconds, self.log, self.source, self.p, self.tails);
    }

    /// Everything made at once, and the bold run to its end (for the tests).
    #[cfg(test)]
    fn settle(&mut self) {
        self.refresh();
        while self.runs.len() < self.wanted {
            self.add_run();
        }
        while !self.bold.done() {
            self.bold.advance(1.0, self.seconds, self.log, self.source, self.p, self.tails);
        }
    }

    fn mean(&self) -> Option<f64> {
        self.source.mean(self.p)
    }

    fn sd(&self) -> Option<f64> {
        self.source.sd(self.p)
    }

    /// What the runs show at grid place `g` (None before there are any).
    fn reading(&self, g: usize) -> Option<Reading> {
        let n = *grid().get(g)?;
        let values: Vec<f64> = self.runs.iter().filter_map(|r| r.avg.get(g).copied()).collect();
        if values.is_empty() {
            return None;
        }
        let count = values.len();
        let sorted = st::sorted(&values);
        let quartiles = (st::quantile_sorted(&sorted, 0.25), st::quantile_sorted(&sorted, 0.75));
        let (inside, rms) = match (self.mean(), self.sd()) {
            (Some(m), Some(_)) => {
                let inside = values.iter().filter(|v| !self.outside(**v, g)).count();
                let rms = (values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / count as f64).sqrt();
                (Some(inside), Some(rms))
            }
            _ => (None, None),
        };
        Some(Reading { n, count, inside, rms, quartiles })
    }

    /// Is an average `avg` of the first n draws, at the grid's place `g`, outside the
    /// funnel there? (Never, for a source without a funnel.)
    fn outside(&self, avg: f64, g: usize) -> bool {
        match (self.mean(), self.sd(), grid().get(g)) {
            // (a hair of room: a source with no spread has averages exactly on the mean, and
            // a funnel of no width)
            (Some(m), Some(s), Some(&n)) => (avg - m).abs() > z95() * s / (n as f64).sqrt() * (1.0 + 1e-9) + 1e-12,
            _ => false,
        }
    }
}

/// The runs at one n.
#[derive(Clone, Copy, Debug)]
struct Reading {
    n: u32,
    count: usize,
    /// How many are inside the funnel, and their root mean square distance from the
    /// mean (for a source with a mean and a finite spread).
    inside: Option<usize>,
    rms: Option<f64>,
    /// The quarter and three quarters of the way up the runs' averages.
    quartiles: (f64, f64),
}

// ----- the layout ---------------------------------------------------------------------------

/// Where everything goes on a canvas (logical pixels).
#[derive(Clone, Copy, Debug)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Text sizes: titles, words, small print.
    big: f32,
    text: f32,
    small: f32,
    sources: [Rect; 6],
    again: Rect,
    /// The charts' rectangles (the data's: their numbers and names go round them).
    main: Rect,
    gap: Option<Rect>,
    sample: Option<Rect>,
    /// The markers' numbers, and the key.
    table: Option<Rect>,
    key: Option<Rect>,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 700.0;
        let (big, text, small) = if narrow { (12.0, 11.0, 10.5) } else { (14.0, 12.5, 11.5) };
        let side = if narrow { 8.0 } else { 20.0 };
        let top = inset.max(0.0) + if narrow { 4.0 } else { 12.0 };
        let bh = if narrow { 24.0 } else { 26.0 };
        let rect = |x0: f32, y0: f32, x1: f32, y1: f32| (Vec2::new(x0, y0), Vec2::new(x1.max(x0), y1.max(y0)));
        // room round a chart: for its title and axis name above, its numbers to the left
        // and its numbers and axis name below
        let (above, left, below) = if narrow { (36.0, 30.0, 34.0) } else { (44.0, 40.0, 36.0) };
        let small_view = view.y < 420.0 || view.x < 240.0;
        // (a window too short for the buttons and a chart has the chart alone)
        let none = (Vec2::ZERO, Vec2::ZERO);
        if view.y < 300.0 {
            let main = rect(side + left, top + above, view.x - side - 14.0, view.y - below);
            return Layout { view, narrow, big, text, small, sources: [none; 6], again: none, main, gap: None, sample: None, table: None, key: None };
        }
        if narrow {
            let row1 = plot::even_row(4, side, view.x - side, top, bh, 5.0);
            let row2 = plot::even_row(3, side, view.x - side, top + bh + 5.0, bh, 5.0);
            let sources = [row1[0], row1[1], row1[2], row1[3], row2[0], row2[1]];
            let again = row2[2];
            let y0 = top + 2.0 * bh + 5.0 + 6.0;
            // (room at the right for half of the last number along the axis)
            let (x0, x1) = (side + left, view.x - side - 14.0);
            if small_view {
                let main = rect(x0, y0 + above, x1, view.y - below);
                return Layout { view, narrow, big, text, small, sources, again, main, gap: None, sample: None, table: None, key: None };
            }
            // the table: a heading, the markers, and four rows
            let table_h = 6.0 * (small + 5.0);
            let fixed = 3.0 * (above + below) + table_h + 10.0;
            let rest = (view.y - y0 - fixed - 6.0).max(60.0);
            let (hm, hg, hs) = (rest * 0.45, rest * 0.29, rest * 0.26);
            let main = rect(x0, y0 + above, x1, y0 + above + hm);
            let y = main.1.y + below;
            let gap = rect(x0, y + above, x1, y + above + hg);
            let y = gap.1.y + below;
            let table = rect(side, y + 4.0, view.x - side, y + 4.0 + table_h);
            let y = table.1.y + 6.0;
            // (room at the right for the die's 1/6; the darts' square, as tall as it is
            // wide, is put in the middle)
            let sample = rect(x0, y + above, x1 - 12.0, y + above + hs);
            return Layout { view, narrow, big, text, small, sources, again, main, gap: Some(gap), sample: Some(sample), table: Some(table), key: None };
        }
        let width = view.x - 2.0 * side;
        let split = side + width * 0.64;
        let sources_row = plot::even_row(6, side, split, top, bh, 6.0);
        let sources = [sources_row[0], sources_row[1], sources_row[2], sources_row[3], sources_row[4], sources_row[5]];
        let right = split + 30.0;
        let again = rect(right + left, top, view.x - side, top + bh);
        let y0 = top + bh + 10.0;
        if small_view {
            let main = rect(side + left, y0 + above, view.x - side, view.y - below);
            return Layout { view, narrow, big, text, small, sources, again, main, gap: None, sample: None, table: None, key: None };
        }
        let rest = (view.y - y0 - 2.0 * (above + below) - 8.0).max(60.0);
        let (hm, hg) = (rest * 0.62, rest * 0.38);
        let main = rect(side + left, y0 + above, split, y0 + above + hm);
        let gap = rect(side + left, main.1.y + below + above, split, main.1.y + below + above + hg);
        // the right-hand column: the bold run's draws (with room at the right for the
        // die's 1/6), as tall as the room left by the table and the key allows, up to
        // the main chart's height
        let (table_h, key_h) = (6.0 * (small + 7.0), 6.0 * (small + 6.0));
        let hs = (view.y - 10.0 - main.0.y - below - 8.0 - table_h - 18.0 - key_h - 10.0).min(hm).max(hm * 0.5);
        let sample = rect(right + left, main.0.y, view.x - side - 26.0, main.0.y + hs);
        let table = rect(right, sample.1.y + below + 8.0, view.x - side, sample.1.y + below + 8.0 + table_h);
        let key = rect(right, table.1.y + 18.0, view.x - side, table.1.y + 18.0 + key_h);
        // (in a short window what does not fit is left out)
        let fits = |r: Rect| r.1.y <= view.y - 4.0;
        Layout { view, narrow, big, text, small, sources, again, main, gap: Some(gap), sample: Some(sample), table: Some(table).filter(|r| fits(*r)), key: Some(key).filter(|r| fits(*r)) }
    }
}

// ----- drawing helpers ----------------------------------------------------------------------

/// Where n goes across a chart of n, from 0 at the left to 1 at the right: on a
/// logarithmic axis (`t` = 0), an ordinary one (`t` = 1), or a share of the way from
/// the one to the other while the axis glides.
fn across(n: f64, t: f64) -> f64 {
    let (n, end) = (n.max(1.0), LONGEST as f64);
    (1.0 - t) * n.ln() / end.ln() + t * n / end
}

/// The n at which a chart of n is `u` of the way across (the inverse of `across`).
fn n_at(u: f64, t: f64) -> f64 {
    let (mut lo, mut hi) = (0.0f64, (LONGEST as f64).ln());
    for _ in 0..50 {
        let mid = 0.5 * (lo + hi);
        if across(mid.exp(), t) < u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi)).exp()
}

/// The two places of the grid either side of n, and how far n is from the first to the
/// second along a chart's axis (`t` as for `across`), for reading a run's path there.
fn between(n: f64, t: f64) -> (usize, usize, f64) {
    let g = grid();
    let i = g.partition_point(|&m| (m as f64) <= n).max(1).min(g.len() - 1);
    let (a, b) = (across(g[i - 1] as f64, t), across(g[i] as f64, t));
    let share = if b > a { ((across(n, t) - a) / (b - a)).max(0.0).min(1.0) } else { 0.0 };
    (i - 1, i, share)
}

/// The n at which the funnels and bands are drawn: spread evenly along both kinds of
/// axis, so that they are smooth on either and while the axis glides.
fn band_ns() -> &'static [f64] {
    static NS: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
    NS.get_or_init(|| {
        let end = LONGEST as f64;
        let mut ns: Vec<f64> = (0..=120).map(|k| end.powf(k as f64 / 120.0)).chain((1..=60).map(|k| end * k as f64 / 60.0)).collect();
        ns.sort_by(f64::total_cmp);
        ns.dedup_by(|a, b| (*a - *b).abs() < 1e-9 * b.abs());
        ns
    })
}

/// A shaded band between `lo` and `hi` at each x, kept inside the chart.
fn band(f: &mut Frame, plot: &Plot, columns: &[(f64, f64, f64)], color: Color) {
    if plot.tiny() {
        return;
    }
    let (top, bottom) = (plot.rect.0.y, plot.rect.1.y);
    let column = |&(x, lo, hi): &(f64, f64, f64)| {
        (x.is_finite() && lo.is_finite() && hi.is_finite()).then(|| (plot.px_x(x), plot.px_y(hi).max(top).min(bottom), plot.px_y(lo).max(top).min(bottom)))
    };
    for pair in columns.windows(2) {
        if let (Some(l), Some(r)) = (column(&pair[0]), column(&pair[1])) {
            let c = [Vec2::new(l.0, l.2), Vec2::new(l.0, l.1), Vec2::new(r.0, r.1), Vec2::new(r.0, r.2)].map(|p| f.hud_at(1, p));
            f.dl.tris_alpha.quad(c[0], c[1], c[2], c[3], color);
        }
    }
}

/// The axis of n along the bottom of a chart whose x goes from 0 to 1 (see `across`):
/// 1, 10, 100, … on a logarithmic axis, round numbers on an ordinary one, each set
/// fading as the axis glides from the one to the other.
fn n_axis(f: &mut Frame, plot: &Plot, t: f64, name: &str, size: f32) {
    if plot.tiny() {
        return;
    }
    let (min, max) = plot.rect;
    plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
    let end = LONGEST as f64;
    let tick = |f: &mut Frame, n: f64, long: bool, label: bool, alpha: f32| {
        let x = plot.px_x(across(n, t));
        let reach = if long { 4.0 } else { 2.5 };
        plot::line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + reach), 1.0, pal::DIM.a(0.8 * alpha));
        if label {
            f.text_px(Vec2::new(x, max.y + 6.0), count(n.round() as usize), size, pal::DIM.a(alpha), Align2::CENTER_TOP);
        }
    };
    if t < 1.0 {
        let alpha = (1.0 - t) as f32;
        let mut decade = 1.0;
        while decade <= end {
            for m in 1..=9 {
                let n = decade * m as f64;
                if n <= end {
                    tick(f, n, m == 1, m == 1, alpha);
                }
            }
            decade *= 10.0;
        }
    }
    if t > 0.0 {
        let ticks = plot::Ticks::of(0.0, end, plot.width());
        for v in ticks.values() {
            tick(f, v.max(1.0), true, true, t as f32);
        }
    }
    if !name.is_empty() {
        f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 8.0 + size), name, size, pal::DIM, Align2::CENTER_TOP);
    }
}

/// A number to two significant figures (a dash for none).
fn sig(v: f64) -> String {
    if !v.is_finite() {
        return "—".to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let decimals = (1.0 - v.abs().log10().floor()).max(0.0).min(5.0) as usize;
    num(v, decimals)
}

/// A number to three decimals at most, without the zeros after its last digit.
fn tidy(v: f64) -> String {
    let text = num(v, 3);
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

/// The first of `options` that fits in `room` pixels at `size` (the last if none does).
fn fitting<'a>(f: &Frame, options: &'a [String], room: f32, size: f32) -> &'a str {
    options.iter().find(|t| f.text_size(t, size).x <= room).or(options.last()).map_or("", String::as_str)
}

/// A batch of discs of their own (marks whose number changes: the markers' dots, the
/// darts), told from the usual one by a hair of their size.
fn discs_of<'b>(f: &'b mut Frame<'_>, kind: u8) -> &'b mut crate::gfx::draw::PointBatch {
    f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, size_scale: 1.0 + 1e-4 * (kind as f32 + 1.0), ..Default::default() })
}

// ----- the scene's frame ----------------------------------------------------------------------

/// Kinds of line batch (see `plot::apart`): the thin runs, the bold run, and the same
/// two in the chart of the total.
const THIN: u8 = OWN;
const BOLD: u8 = OWN + 1;
const THIN_GAP: u8 = OWN + 2;
const BOLD_GAP: u8 = OWN + 3;
/// The dark edges under the bold runs.
const UNDER: u8 = OWN + 4;
const UNDER_GAP: u8 = OWN + 5;
const UNDERLAY: Color = Color::hex(0x080B14);

/// Line batches are drawn in the order they are first asked for: so, before anything
/// is drawn, the thin runs come first, the funnels' edges and the markers over them,
/// and the bold runs over all.
fn in_order(f: &mut Frame) {
    for kind in [AXES, THIN, THIN_GAP, CURVES, MARKS, UNDER, BOLD, UNDER_GAP, BOLD_GAP] {
        plot::apart(f, kind, false);
    }
}

/// How bright each thin run is: fainter the more there are, so that two hundred fill
/// the funnel without hiding its edges.
fn thin_alpha(runs: usize) -> f32 {
    (0.55 * (12.0 / runs.max(1) as f32).powf(0.6)).clamp(0.07, 0.55)
}

impl Scene for Lln {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.moving
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("white marks on the axis of $n$: drag them along    buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Draw from a distribution again and again, each draw independent of the others, and keep the average of the draws so far. The law of large numbers: as the number of draws $n$ grows, the average settles on the distribution's mean $\mu$, if it has one. How fast: its typical distance from $\mu$ is $\sigma/\sqrt{n}$, where $\sigma$ is the standard deviation of a single draw (if that is finite), so four times the draws halve it. Choose where the draws come from with the buttons over the picture.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Settings");
        ui::slider_int_log(ui, "runs", &mut self.wanted, 1..=MOST_RUNS);
        ui::note(ui, "The thin paths: each another run of 10 000 draws. The numbers at the markers are counted over them.");
        ui::slider_log(ui, "seconds for a run", &mut self.seconds, 2.0..=60.0);
        ui::note(ui, "How long the bold run takes to cross the chart. On the logarithmic axis the draws come faster as $n$ grows, so that it crosses evenly: about one a second at first, thousands a second at the end.");
        let mut axis = (!self.log) as usize;
        if ui::choice(ui, &[r"$n$ on a logarithmic axis", r"on an ordinary axis"], &mut axis) {
            self.log = axis == 0;
        }
        if self.source == Source::Coin {
            let mut p = self.p as f32;
            if ui::slider(ui, r"chance of heads $p$", &mut p, 0.0..=1.0) {
                self.p = p as f64;
            }
            crate::scenes::stats::checkbox(ui, &mut self.tails, "every run begins with ten tails");
        }
        ui.horizontal(|ui| {
            if ui.button("another run").clicked() {
                self.bolds += 1;
            }
            if ui.button("start over").clicked() {
                self.seed = self.seed.wrapping_add(100);
                self.bolds = 0;
            }
        });
        ui::note(ui, "Another run: a new bold run, the thin ones kept. Start over: new runs for all.");
        self.refresh();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (ESTIMATE, "yellow: the average of the draws so far, one path for each run; bold, the run being drawn now"),
                (TRUTH, r"green: the mean $\mu$ and the funnel round it; the true chances or density of one draw"),
                (DATA, "cyan: the bold run's draws so far"),
                (OTHER, "orange: the total of the draws less what the mean predicts for it"),
                (ALARM, "pink: a run outside the funnel at a marker"),
                (pal::WHITE, r"white: the two markers, to drag along $n$; beside each, a bracket from $\mu$ less to $\mu$ plus the runs' typical distance from it there (without a mean: the middle half of the runs)"),
            ],
        );

        ui::section(ui, "Words");
        let z = z95();
        ui::term(ui, "Independent draws:", "each made in the same way, and none affected by the others: a die does not remember its last roll.");
        ui::term(ui, "Sample mean:", r"the average of the draws so far, $\bar x_n = (x_1 + \dots + x_n)/n$.");
        ui::term(ui, "Law of large numbers:", r"for independent draws from a distribution with a mean $\mu$, the sample mean comes within any margin of $\mu$, however small, and stays within it from some $n$ on, with probability 1. It does not say how soon.");
        ui::term(ui, "Standard error:", r"the typical distance of the sample mean from $\mu$ (the square root of its mean square, over many runs): exactly $\sigma/\sqrt{n}$, if a single draw has a finite standard deviation $\sigma$.");
        let funnel = format!(
            r"$\mu \pm {}\,\sigma/\sqrt{{n}}$. At each $n$ about 95 % of runs are inside it once the averages are spread like a bell (the next scene says when); at any $n$ at least $1 - 1/{}^2$, about {} %, are (Chebyshev's inequality). It is about each $n$ alone: a run inside at one $n$ can be outside at another.",
            num(z, 2),
            num(z, 2),
            num(100.0 * (1.0 - 1.0 / (z * z)), 0)
        );
        ui::term(ui, "The funnel:", &funnel);
        ui::term(ui, "Gap in the total:", r"the total of the first $n$ draws less $n\mu$, what the mean predicts for it. Its typical size is $\sigma\sqrt{n}$, which grows; the average's distance from $\mu$ is the gap divided by $n$, and that shrinks.");
        ui::term(ui, "Monte Carlo:", r"working a number out (an area, a probability, an integral) as the average of random draws; its error falls like $1/\sqrt{n}$.");

        ui::section(ui, "Numbers");
        let b = &self.bold;
        ui::readout(ui, "draws in the bold run", count(b.n));
        ui::readout(ui, "their average", format!("${}$", num(b.average(), 3)));
        match (self.mean(), self.sd()) {
            (Some(m), Some(s)) => {
                ui::readout(ui, r"the mean $\mu$", format!("${}$", tidy(m)));
                ui::readout(ui, r"the average's distance from $\mu$", format!("${}$", num((b.average() - m).abs(), 3)));
                ui::readout(ui, r"$\sigma/\sqrt{n}$ at this $n$", format!("${}$", num(s / (b.n.max(1) as f64).sqrt(), 3)));
            }
            _ => ui::readout(ui, r"the mean $\mu$", "none"),
        }
        ui::readout(ui, "draws a second now", count(b.rate(self.seconds, self.log).round() as usize));
        ui::see_also(ui, &[("clt", "the shape of the averages inside the funnel"), ("intervals", "the funnel turned round: a confidence interval"), ("estimators", "the average as an estimate, and its error"), ("mcmc", "averages of draws that are not independent")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 160.0;
        if !tiny {
            self.buttons(f, &lay);
        }
        self.refresh();
        self.work(f.dt, f.real_dt);

        // the glides: the markers and the axis answer the hand (by the clock), the bars
        // the draws (by the scene's time)
        let axis = [if self.log { 0.0 } else { 1.0 }];
        let mut moving = self.look.axis.follow(&axis, f.real_dt, false);
        let t = self.look.axis.get(0).max(0.0).min(1.0) as f64;
        let plots = Plots::of(&lay, self.source, self.p);
        if !tiny {
            self.handles(f, &plots, t);
        }
        let marks: Vec<f32> = (0..2).map(|k| self.held[k].unwrap_or_else(|| (grid()[self.marks[k]] as f32).log10())).collect();
        // (a marker in the hand is where the hand is)
        if self.held.iter().any(Option::is_some) {
            self.look.marks.snap(&marks);
        } else {
            moving |= self.look.marks.follow(&marks, f.real_dt, true);
        }
        moving |= self.ease_head(f.dt);
        moving |= self.ease_dots(f.real_dt, t);
        moving |= self.place_names(f, &lay, &plots, t);
        let bars = self.bar_heights();
        moving |= self.look.bars.follow(&bars, f.dt, true);
        let fading = self.runs.iter().any(|r| r.age < FADE);
        self.moving = moving || fading || !self.bold.done() || self.runs.len() < self.wanted;
        if tiny {
            return;
        }
        in_order(f);
        self.thin_runs(f, &plots, t);
        self.draw_main(f, &lay, &plots, t);
        if let Some(gap) = &plots.gap {
            self.draw_gap(f, &lay, gap, t);
        }
        if let Some(sample) = &plots.sample {
            self.draw_sample(f, &lay, sample);
        }
        if let Some(table) = lay.table {
            self.draw_table(f, &lay, table);
        }
        if let Some(key) = lay.key {
            self.draw_key(f, &lay, key);
        }
    }
}

/// The charts on this canvas.
struct Plots {
    main: Plot,
    gap: Option<Plot>,
    sample: Option<Plot>,
}

impl Plots {
    fn of(lay: &Layout, source: Source, p: f64) -> Plots {
        let main = Plot::new(lay.main, [0.0, 1.0], source.range());
        let gap = lay.gap.map(|r| {
            let (end, z) = (LONGEST as f64, z95());
            let reach = source.sd(p).map_or(1.0, |s| z * s * end.sqrt() / source.unit());
            // (room for the ten tails' head start, even where the band has no width)
            let half = (reach * 1.15).max(12.0);
            Plot::new(r, [0.0, 1.0], [-half, half])
        });
        let sample = lay.sample.map(|r| match source {
            Source::Die => Plot::new(r, [0.5, 6.5], [0.0, 1.0]),
            Source::Coin => Plot::new(r, [-0.5, 1.5], [0.0, 1.0]),
            Source::Darts => {
                // a square, in the middle of its room
                let (min, max) = r;
                let side = (max.x - min.x).min(max.y - min.y);
                let mid = (min.x + max.x) * 0.5;
                Plot::new((Vec2::new(mid - side * 0.5, min.y), Vec2::new(mid + side * 0.5, min.y + side)), [-1.0, 1.0], [-1.0, 1.0])
            }
            _ => {
                let h = source.bins();
                let peak = (0..=200).map(|i| source.density(h.lo + (h.hi - h.lo) * i as f64 / 200.0)).fold(0.0, f64::max);
                Plot::new(r, [h.lo, h.hi], [0.0, peak * 1.3])
            }
        });
        Plots { main, gap, sample }
    }
}

impl Lln {
    /// The buttons over the picture: where the draws come from, and another run.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        for (k, r) in lay.sources.iter().enumerate() {
            let on = self.source == SOURCES[k];
            if plot::button(f, *r, SOURCES[k].name(), lay.text, on) && !on {
                self.source = SOURCES[k];
            }
        }
        if plot::button(f, lay.again, "another run", lay.text, false) {
            self.bolds += 1;
        }
    }

    /// The markers, dragged along n.
    ///
    /// The numbers are read at the place of the grid nearest the hand; the marker itself
    /// follows the hand exactly while it is held, and glides to that place when let go.
    fn handles(&mut self, f: &mut Frame, plots: &Plots, t: f64) {
        let main = &plots.main;
        for k in 0..2 {
            let shown = 10f64.powf(self.look.marks.get(k) as f64);
            let x = across(shown, t);
            if let Some((u, _)) = main.handle(f, 60 + k as u32, x, main.y[0], pal::WHITE) {
                let n = n_at(u, t);
                self.marks[k] = grid_index(n);
                self.held[k] = Some(n.max(1.0).log10() as f32);
            }
        }
        if !f.handles.dragging() {
            self.held = [None; 2];
        }
    }

    /// Bring the head drawn `dt` seconds of the scene's time nearer to the bold run's
    /// own (there at once for a new run). Says whether it is still on its way.
    fn ease_head(&mut self, dt: f32) -> bool {
        let (n, avg, gap) = self.bold.head(self.mean().unwrap_or(0.0));
        let to = [n.max(1.0).ln(), avg, gap];
        let of = self.bold_for;
        let look = &mut self.look;
        let Some(head) = look.head.as_mut().filter(|_| look.head_of == of && to.iter().all(|v| v.is_finite())) else {
            (look.head, look.toward, look.head_of) = (Some(to), to, of);
            return false;
        };
        let k = 1.0 - (-(dt.max(0.0) as f64) / HEAD_EASE).exp();
        for ((h, w), t) in head.iter_mut().zip(look.toward.iter_mut()).zip(to) {
            *w += (t - *w) * k;
            *h += (*w - *h) * k;
        }
        // (exactly there in the end, so that the picture comes to rest)
        let near = |a: &[f64; 3]| (a[0] - to[0]).abs() < 1e-7 && (a[1] - to[1]).abs() < 1e-7 * (1.0 + to[1].abs()) && (a[2] - to[2]).abs() < 1e-7 * (1.0 + to[2].abs());
        if (near(head) && near(&look.toward)) || !head.iter().chain(&look.toward).all(|v| v.is_finite()) {
            (*head, look.toward) = (to, to);
            return false;
        }
        true
    }

    /// Bring the dots where the runs cross the markers `dt` seconds nearer to where
    /// they are, through two eases like the head's. A marker dragged along n carries its dots
    /// along the runs' paths; those are zigzags with a corner at every place of the grid,
    /// and a dot drawn exactly on them changes speed at each, as often as every frame
    /// where the places are close. Eased, a dot rounds the corners off and comes to rest
    /// exactly on its path when the marker does. (New runs: there at once.)
    fn ease_dots(&mut self, dt: f32, t: f64) -> bool {
        let of = Some((self.made_for, self.runs.len()));
        let k = 1.0 - (-(dt.max(0.0) as f64) / DOT_EASE).exp();
        let mut moving = false;
        for m in 0..2 {
            let (lo, hi, share) = between(10f64.powf(self.look.marks.get(m) as f64), t);
            let to: Vec<f64> = self.runs.iter().map(|r| r.avg.get(lo).zip(r.avg.get(hi)).map_or(f64::NAN, |(a, b)| a + (b - a) * share)).collect();
            let dots = &mut self.look.dots[m];
            if dots.of != of || dots.at.len() != to.len() {
                (dots.at, dots.toward, dots.of) = (to.clone(), to, of);
                continue;
            }
            let mut near = true;
            for ((a, w), t) in dots.at.iter_mut().zip(dots.toward.iter_mut()).zip(&to) {
                *w += (t - *w) * k;
                *a += (*w - *a) * k;
                if !(a.is_finite() && w.is_finite()) {
                    (*a, *w) = (*t, *t);
                }
                near &= ((*a - t).abs() <= 1e-7 * (1.0 + t.abs()) && (*w - t).abs() <= 1e-7 * (1.0 + t.abs())) || !t.is_finite();
            }
            if near {
                (dots.at, dots.toward) = (to.clone(), to);
            } else {
                moving = true;
            }
        }
        moving
    }

    /// Where the markers' names go: under the top of the chart's axis, each over its
    /// marker but kept inside the chart, and the second a line higher where the two
    /// would cover each other. They glide there (by the clock: they answer the hand).
    fn place_names(&mut self, f: &Frame, lay: &Layout, plots: &Plots, t: f64) -> bool {
        let plot = &plots.main;
        let (min, max) = plot.rect;
        let mut xs = [0.0f32; 2];
        let mut halves = [0.0f32; 2];
        for k in 0..2 {
            let label = format!("$n = {}$", count(grid()[self.marks[k]] as usize));
            halves[k] = f.text_size(&label, lay.small).x * 0.5 + 4.0;
            let x = plot.px_x(across(10f64.powf(self.look.marks.get(k) as f64), t));
            xs[k] = x.max(min.x + halves[k]).min(max.x - halves[k]);
        }
        let crowded = (xs[0] - xs[1]).abs() < halves[0] + halves[1] + 4.0;
        let ups = [0.0, if crowded { 1.0 } else { 0.0 }];
        let mut moving = self.look.label_x.follow(&xs, f.real_dt, false);
        moving |= self.look.label_up.follow(&ups, f.real_dt, false);
        // (a marker in the hand has its name with it)
        if self.held.iter().any(Option::is_some) {
            self.look.label_x.snap(&xs);
        }
        moving
    }

    /// The head as drawn: n, the average and the total less n μ.
    fn head(&self) -> (f64, f64, f64) {
        match self.look.head {
            Some([ln_n, avg, gap]) => (ln_n.exp().max(1.0).min(LONGEST as f64), avg, gap),
            None => self.bold.head(self.mean().unwrap_or(0.0)),
        }
    }

    /// What the bars beside the main chart are to be: the share of each face or side,
    /// or the density in each bin.
    fn bar_heights(&self) -> Vec<f32> {
        let h = &self.bold.hist;
        match self.source {
            Source::Die | Source::Coin | Source::Darts => h.counts.iter().map(|&c| c as f32 / h.total.max(1) as f32).collect(),
            _ => (0..h.counts.len()).map(|i| h.density(i) as f32).collect(),
        }
    }

    /// The thin runs, in both charts: laid out afresh when what they look like has
    /// changed (the runs, the canvas, the axis, the view, or runs still fading in), and
    /// otherwise copied as they were last laid out. (Two hundred runs of a few hundred
    /// points each, in two charts, take milliseconds to lay out, and while the bold run
    /// is drawn nothing about them changes.)
    fn thin_runs(&mut self, f: &mut Frame, plots: &Plots, t: f64) {
        let fading = self.runs.iter().any(|r| r.age < FADE);
        let wanted = LaidFor { main: plots.main.rect, gap: plots.gap.map(|g| g.rect), t: t.to_bits(), runs: self.runs.len(), made_for: self.made_for, hud: f.hud };
        if !fading && self.laid.made_for == Some(wanted) {
            plot::apart(f, THIN, false).pts.extend_from_slice(&self.laid.main);
            plot::apart(f, THIN_GAP, false).pts.extend_from_slice(&self.laid.gap);
            return;
        }
        let start = [THIN, THIN_GAP].map(|k| plot::apart(f, k, false).pts.len());
        let alpha = thin_alpha(self.runs.len());
        // (many runs are a haze in which a point every two and a half pixels is enough)
        let apart = if self.runs.len() > 40 { 2.5 } else { 1.5 };
        let plot = &plots.main;
        if !plot.tiny() {
            let drawn = Self::drawn(plot, t, apart);
            let us: Vec<f64> = drawn.iter().map(|&i| across(grid()[i] as f64, t)).collect();
            for r in &self.runs {
                let a = alpha * (r.age / FADE).min(1.0);
                plot.path(f, THIN, drawn.iter().zip(&us).filter_map(|(&i, &u)| r.avg.get(i).map(|v| (u, *v))), 1.0, ESTIMATE.a(a));
            }
        }
        if let (Some(plot), Some(m)) = (plots.gap.filter(|p| !p.tiny()), self.mean()) {
            let drawn = Self::drawn(&plot, t, apart);
            let us: Vec<f64> = drawn.iter().map(|&i| across(grid()[i] as f64, t)).collect();
            let unit = self.source.unit();
            for r in &self.runs {
                let a = alpha * (r.age / FADE).min(1.0);
                let gap = |i: usize, avg: f64| (avg - m) * grid()[i] as f64 / unit;
                plot.path(f, THIN_GAP, drawn.iter().zip(&us).filter_map(|(&i, &u)| r.avg.get(i).map(|v| (u, gap(i, *v)))), 1.0, OTHER.a(a));
            }
        }
        self.laid.main = plot::apart(f, THIN, false).pts.get(start[0]..).unwrap_or_default().to_vec();
        self.laid.gap = plot::apart(f, THIN_GAP, false).pts.get(start[1]..).unwrap_or_default().to_vec();
        self.laid.made_for = (!fading).then_some(wanted);
    }

    /// The grid's places drawn along a path: no two closer than `apart` pixels.
    fn drawn(plot: &Plot, t: f64, apart: f32) -> Vec<usize> {
        let g = grid();
        let mut out = Vec::with_capacity(g.len());
        let mut last = f32::NEG_INFINITY;
        for (i, &n) in g.iter().enumerate() {
            let x = plot.px_x(across(n as f64, t));
            if x - last >= apart || i + 1 == g.len() {
                out.push(i);
                last = x;
            }
        }
        out
    }

    fn draw_main(&self, f: &mut Frame, lay: &Layout, plots: &Plots, t: f64) {
        let plot = &plots.main;
        if plot.tiny() {
            return;
        }
        let (min, max) = plot.rect;
        plot.backing(f, if lay.narrow { 30.0 } else { 40.0 }, if lay.narrow { 32.0 } else { 36.0 }, if lay.narrow { 36.0 } else { 44.0 }, 8.0);
        let room = max.x - min.x + 30.0;
        let runs = self.runs.len();
        let titles = [
            format!("the average of the first $n$ draws: {runs} runs, and one being drawn now (bold)"),
            format!("average of the first $n$ draws: {runs} runs, and one (bold)"),
            "average of the first $n$ draws".to_string(),
        ];
        plot.title(f, fitting(f, &titles, room, lay.big), lay.big);
        plot.y_grid(f);
        plot.y_axis(f, "average", lay.small);
        let axis_name = if t < 0.5 { "$n$, the number of draws (logarithmic scale)" } else { "$n$, the number of draws" };
        n_axis(f, plot, t, axis_name, lay.small);

        // the funnel, the mean ± 1.96 standard errors: shaded under the runs, its edges
        // over them (their batch comes later: see `in_order`)
        let (mean, sd, z) = (self.mean(), self.sd(), z95());
        if let (Some(m), Some(s)) = (mean, sd) {
            let columns: Vec<(f64, f64, f64)> = band_ns().iter().map(|&n| (across(n, t), m - z * s / n.sqrt(), m + z * s / n.sqrt())).collect();
            band(f, plot, &columns, TRUTH.a(0.035));
            plot.path(f, CURVES, columns.iter().map(|c| (c.0, c.1)), 1.1, TRUTH.a(0.8));
            plot.path(f, CURVES, columns.iter().map(|c| (c.0, c.2)), 1.1, TRUTH.a(0.8));
            plot.hline(f, MARKS, m, 1.4, TRUTH, false);
        } else {
            // no mean: the centre, which is the median, and no funnel
            plot.hline(f, MARKS, 0.0, 1.2, TRUTH.a(0.8), true);
        }

        // the bold run (the thin ones are `thin_runs`'), on a dark edge of its own, to be
        // seen among them
        let drawn = Self::drawn(plot, t, 1.5);
        let us: Vec<f64> = drawn.iter().map(|&i| across(grid()[i] as f64, t)).collect();
        let b = &self.bold;
        let (head_n, head_avg, _) = self.head();
        let path: Vec<(f64, f64)> = drawn
            .iter()
            .zip(&us)
            .filter(|(&i, _)| grid()[i] as f64 <= head_n)
            .filter_map(|(&i, &u)| b.at.get(i).map(|v| (u, *v)))
            .chain(std::iter::once((across(head_n, t), head_avg)))
            .collect();
        plot.path(f, UNDER, path.iter().copied(), 4.6, UNDERLAY);
        plot.path(f, BOLD, path, 2.6, ESTIMATE);
        plot.dot(f, across(head_n, t), head_avg, 9.0, ESTIMATE);

        // the markers: a line each, and a dot where each run crosses it (on the run's
        // path, so that a marker on its way to a new n carries the dots along the paths)
        let shown: [f64; 2] = [0, 1].map(|k| 10f64.powf(self.look.marks.get(k) as f64));
        let size = if lay.narrow { 4.0 } else { 5.0 };
        for (k, &n) in shown.iter().enumerate() {
            let u = across(n, t);
            plot.vline(f, MARKS, u, 1.2, pal::WHITE.a(0.55), true);
            let g = self.marks[k];
            let x = plot.px_x(u);
            let half = match (mean, sd) {
                (Some(_), Some(s)) => z * s / n.max(1.0).sqrt(),
                _ => f64::INFINITY,
            };
            for &v in &self.look.dots[k].at {
                let at = Vec2::new(x, plot.px_y(v));
                if plot::within(at, plot.rect, 0.5) && at.is_finite() {
                    let outside = mean.is_some_and(|m| (v - m).abs() > half * (1.0 + 1e-9) + 1e-12);
                    let p = f.hud(at);
                    discs_of(f, 0).push(p, size, if outside { ALARM } else { ESTIMATE }.a(0.85));
                }
            }
            // beside it, how far the runs typically are from the mean there: a bracket from
            // μ less to μ plus that (for a source without a mean, the middle half of the
            // runs), measured on the dots as drawn, so that it changes as smoothly as they do
            let dots: Vec<f64> = self.look.dots[k].at.iter().copied().filter(|v| v.is_finite()).collect();
            if !dots.is_empty() {
                let (lo, hi) = match (mean, sd) {
                    (Some(m), Some(_)) => {
                        let d = (dots.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / dots.len() as f64).sqrt();
                        (m - d, m + d)
                    }
                    _ => {
                        let sorted = st::sorted(&dots);
                        (st::quantile_sorted(&sorted, 0.25), st::quantile_sorted(&sorted, 0.75))
                    }
                };
                let side = if x + 12.0 < max.x { 7.0 } else { -7.0 };
                let (top, bottom) = (plot.px_y(hi).max(min.y).min(max.y), plot.px_y(lo).max(min.y).min(max.y));
                let bx = x + side;
                plot::line_of(f, MARKS, Vec2::new(bx, top), Vec2::new(bx, bottom), 2.0, pal::WHITE);
                for y in [top, bottom] {
                    plot::line_of(f, MARKS, Vec2::new(bx - 3.0, y), Vec2::new(bx + 3.0, y), 2.0, pal::WHITE);
                }
            }
            if max.y - min.y >= 120.0 {
                let label = format!("$n = {}$", count(grid()[g] as usize));
                let x = self.look.label_x.get(k);
                let up = self.look.label_up.get(k) * (lay.small + 9.0);
                f.caption_px(Vec2::new(x, max.y - 12.0 - up), label, lay.small, pal::WHITE, Align2::CENTER_BOTTOM);
            }
        }

        // the bold run's numbers, over the chart's right-hand corner (where the axis's
        // name is at the left), and the names of what is drawn, where there is room
        let readout = format!("bold run: $n = {}$, average ${}$", count(b.n), num(b.average(), 3));
        f.text_px(Vec2::new(max.x, min.y - 6.0), readout, lay.small, ESTIMATE, Align2::RIGHT_BOTTOM);
        if max.y - min.y < 120.0 {
            return;
        }
        match (mean, sd) {
            (Some(m), Some(s)) => {
                let half = z * s / (LONGEST as f64).sqrt();
                let above = plot.px_y(m + half).max(min.y + lay.small + 12.0);
                let below = plot.px_y(m - half).min(max.y - lay.small - 20.0);
                if s > 0.0 {
                    f.caption_px(Vec2::new(max.x - 6.0, above - 4.0), format!(r"funnel: $\mu \pm {}\,\sigma/\sqrt{{n}}$", num(z, 2)), lay.small, TRUTH, Align2::RIGHT_BOTTOM);
                    f.caption_px(Vec2::new(max.x - 6.0, below + 4.0), format!(r"mean $\mu = {}$", tidy(m)), lay.small, TRUTH, Align2::RIGHT_TOP);
                } else {
                    // (the mean may be on the chart's edge: the name goes on the side with room)
                    let y = plot.px_y(m);
                    let (at, align) = if y - min.y > max.y - y { (y - 4.0, Align2::RIGHT_BOTTOM) } else { (y + 4.0, Align2::RIGHT_TOP) };
                    f.caption_px(Vec2::new(max.x - 6.0, at), format!(r"mean $\mu = {}$; no spread, so the funnel has no width", tidy(m)), lay.small, TRUTH, align);
                }
            }
            _ => {
                let y = plot.px_y(0.0);
                let names = ["the centre (the median): there is no mean, and no funnel".to_string(), "the centre: no mean, no funnel".to_string()];
                let name = fitting(f, &names, max.x - min.x - 12.0, lay.small).to_string();
                f.caption_px(Vec2::new(max.x - 6.0, y - 4.0), name, lay.small, TRUTH, Align2::RIGHT_BOTTOM);
            }
        }
    }

    fn draw_gap(&self, f: &mut Frame, lay: &Layout, plot: &Plot, t: f64) {
        if plot.tiny() {
            return;
        }
        let (min, max) = plot.rect;
        plot.backing(f, if lay.narrow { 30.0 } else { 40.0 }, if lay.narrow { 32.0 } else { 36.0 }, if lay.narrow { 36.0 } else { 44.0 }, 8.0);
        let room = max.x - min.x + 30.0;
        let (Some(m), Some(s)) = (self.mean(), self.sd()) else {
            plot.title(f, "the total of the draws: nothing to compare it with", lay.big);
            let mid = (min + max) * 0.5;
            f.text_px(mid - Vec2::new(0.0, 2.0), "The Cauchy distribution has no mean,", lay.small, pal::DIM, Align2::CENTER_BOTTOM);
            f.text_px(mid + Vec2::new(0.0, 2.0), r"so there is no $n\mu$ to compare the total with.", lay.small, pal::DIM, Align2::CENTER_TOP);
            return;
        };
        let unit = self.source.unit();
        let title = match self.source {
            Source::Coin if self.p == 0.5 => vec!["the count that does not catch up: heads minus half the tosses".to_string(), "heads minus half the tosses".to_string()],
            Source::Coin => vec![format!(r"heads minus $p$ times the tosses ($p = {}$)", tidy(self.p))],
            Source::Darts => vec![r"points inside the circle minus $\pi/4$ of all the points".to_string(), r"points inside less $\pi/4$ of all".to_string()],
            Source::Die => vec![format!(r"the total of the rolls minus ${}n$", num(m, 1)), "the total less $3.5n$".to_string()],
            Source::Wait => vec![r"the total of the waits minus $n$ (their mean, $1$, each)".to_string(), "the total of the waits minus $n$".to_string()],
            _ => vec![r"the total of the draws (their mean is $0$)".to_string(), "the total".to_string()],
        };
        plot.title(f, fitting(f, &title, room, lay.big), lay.big);
        plot.y_grid(f);
        plot.y_axis(f, "", lay.small);
        // (a short chart gets one round number from the kit, the 0: it is given one either
        // side as well, so that the band's size can be read)
        if plot::Ticks::of(plot.y[0], plot.y[1], plot.height() * 1.6).values().count() < 3 {
            let v = nice_step((plot.y[1] * 0.5) as f32) as f64;
            for y in [-v, v] {
                let at = plot.px_y(y);
                plot::line_of(f, AXES, Vec2::new(min.x - 4.0, at), Vec2::new(min.x, at), 1.0, pal::DIM.a(0.8));
                f.text_px(Vec2::new(min.x - 6.0, at), num(y, 0), lay.small, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
        n_axis(f, plot, t, "", lay.small);
        let z = z95();
        let columns: Vec<(f64, f64, f64)> = band_ns().iter().map(|&n| (across(n, t), -z * s * n.sqrt() / unit, z * s * n.sqrt() / unit)).collect();
        band(f, plot, &columns, TRUTH.a(0.035));
        plot.path(f, CURVES, columns.iter().map(|c| (c.0, c.1)), 1.1, TRUTH.a(0.8));
        plot.path(f, CURVES, columns.iter().map(|c| (c.0, c.2)), 1.1, TRUTH.a(0.8));
        plot.hline(f, MARKS, 0.0, 1.2, TRUTH, false);

        let drawn = Self::drawn(plot, t, 1.5);
        let us: Vec<f64> = drawn.iter().map(|&i| across(grid()[i] as f64, t)).collect();
        let gap = |i: usize, avg: f64| (avg - m) * grid()[i] as f64 / unit;
        let b = &self.bold;
        let (head_n, _, head_gap) = self.head();
        let path: Vec<(f64, f64)> = drawn
            .iter()
            .zip(&us)
            .filter(|(&i, _)| grid()[i] as f64 <= head_n)
            .filter_map(|(&i, &u)| b.at.get(i).map(|v| (u, gap(i, *v))))
            .chain(std::iter::once((across(head_n, t), head_gap / unit)))
            .collect();
        plot.path(f, UNDER_GAP, path.iter().copied(), 4.4, UNDERLAY);
        plot.path(f, BOLD_GAP, path, 2.4, OTHER);
        plot.dot(f, across(head_n, t), head_gap / unit, 8.0, OTHER);
        for k in 0..2 {
            plot.vline(f, MARKS, across(10f64.powf(self.look.marks.get(k) as f64), t), 1.2, pal::WHITE.a(0.55), true);
        }
        let band_name = if s > 0.0 { format!(r"the band $\pm {}\,\sigma\sqrt{{n}}$ widens like $\sqrt{{n}}$", num(z, 2)) } else { "no spread: no gap ever opens".to_string() };
        f.caption_px(Vec2::new(min.x + 6.0, min.y + 6.0), band_name, lay.small, TRUTH, Align2::LEFT_TOP);
        // after ten tails the count is that many heads short of p n, and that is where it
        // stays on average: each toss after them adds 1 − p or takes away p, nothing on average
        if self.source == Source::Coin && self.tails && self.p > 0.0 {
            let short = -(TAILS as f64) * self.p;
            plot.hline(f, MARKS, short, 1.2, OTHER, true);
            let y = plot.px_y(short);
            let label = format!("${}$: where the ten tails left it, and where it stays on average", num(short, if short.fract() == 0.0 { 0 } else { 1 }));
            f.caption_px(Vec2::new(min.x + 6.0, y + 5.0), label, lay.small, OTHER, Align2::LEFT_TOP);
        }
    }

    fn draw_sample(&self, f: &mut Frame, lay: &Layout, plot: &Plot) {
        if plot.tiny() {
            return;
        }
        let (min, max) = plot.rect;
        let b = &self.bold;
        let bars = &self.look.bars;
        let room = lay.sample.map_or(max.x - min.x, |r| r.1.x - r.0.x) + 56.0;
        match self.source {
            Source::Darts => {
                hud_panel(f, min - Vec2::splat(6.0), max + Vec2::splat(6.0));
                let inside = b.hist.counts.get(1).copied().unwrap_or(0);
                let titles = [format!("the bold run's points: {} of {} inside the circle", count(inside as usize), count(b.n)), format!("{} of {} inside", count(inside as usize), count(b.n))];
                f.text_px(Vec2::new(lay.sample.map_or(min.x, |r| r.0.x) - 30.0, min.y - 10.0), fitting(f, &titles, room, lay.big), lay.big, pal::TEXT, Align2::LEFT_BOTTOM);
                plot.path(f, AXES, [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0)], 1.0, pal::DIM.a(0.8));
                plot.path(f, CURVES, (0..=96).map(|k| (k as f64 / 96.0 * std::f64::consts::TAU).sin_cos()).map(|(s, c)| (c, s)), 1.4, TRUTH);
                let size = if lay.narrow { 2.2 } else { 3.0 };
                let drawn = b.throws.len().min(b.n);
                for &(u, v, inside) in &b.throws[..drawn] {
                    let at = plot.at(u as f64, v as f64);
                    let p = f.hud(at);
                    discs_of(f, 1).push(p, size, if inside { DATA.a(0.8) } else { pal::DIM.a(0.6) });
                }
                // what the average is made of
                let sum = format!(r"$4 \times {} / {} = {}$: the average", inside, b.n, num(b.average(), 3));
                f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 9.0), sum, lay.small, ESTIMATE, Align2::CENTER_TOP);
                if b.n > THROWS_DRAWN {
                    f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 11.0 + lay.small + 4.0), format!("(the first {} points drawn)", count(THROWS_DRAWN)), lay.small, pal::DIM, Align2::CENTER_TOP);
                }
            }
            Source::Die | Source::Coin => {
                plot.backing(f, 34.0, 26.0, 36.0, 8.0);
                let what = if self.source == Source::Die { "rolls: the share of each face" } else { "tosses: the share of tails and of heads" };
                let titles = [format!("the bold run's {what}"), format!("its {what}"), "the share of each".to_string()];
                plot.title(f, fitting(f, &titles, room, lay.big), lay.big);
                plot.y_grid(f);
                plot.y_axis(f, "", lay.small);
                plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
                let heights: Vec<f64> = (0..b.hist.counts.len()).map(|i| bars.get(i) as f64).collect();
                let lo = if self.source == Source::Die { 0.5 } else { -0.5 };
                plot.bars(f, lo, 1.0, &heights, 4.0, DATA.a(0.7));
                if self.source == Source::Die {
                    for (k, face) in FACES.iter().enumerate() {
                        f.text_px(Vec2::new(plot.px_x(*face), max.y + 5.0), format!("{}", k + 1), lay.small, pal::DIM, Align2::CENTER_TOP);
                    }
                    plot.hline(f, MARKS, 1.0 / 6.0, 1.4, TRUTH, true);
                    f.text_px(Vec2::new(max.x + 5.0, plot.px_y(1.0 / 6.0)), "$1/6$", lay.small, TRUTH, Align2::LEFT_CENTER);
                } else {
                    for (x, name, chance) in [(0.0, "tails", 1.0 - self.p), (1.0, "heads", self.p)] {
                        f.text_px(Vec2::new(plot.px_x(x), max.y + 5.0), name, lay.small, pal::DIM, Align2::CENTER_TOP);
                        let (a, c) = (plot.at(x - 0.45, chance), plot.at(x + 0.45, chance));
                        plot::line_of(f, MARKS, a, c, 1.6, TRUTH);
                        f.caption_px(Vec2::new(c.x, c.y - 4.0), format!("${}$", tidy(chance)), lay.small, TRUTH, Align2::RIGHT_BOTTOM);
                    }
                }
            }
            _ => {
                plot.backing(f, 34.0, 26.0, 36.0, 8.0);
                let titles = ["the bold run's draws, and (green) the density of one".to_string(), "the bold run's draws".to_string()];
                plot.title(f, fitting(f, &titles, room, lay.big), lay.big);
                plot.x_axis(f, "", lay.small);
                let h = &b.hist;
                let heights: Vec<f64> = (0..h.counts.len()).map(|i| bars.get(i) as f64).collect();
                plot.bars(f, h.lo, h.width(), &heights, 0.5, DATA.a(0.7));
                let source = self.source;
                plot.curve(f, CURVES, 160, 1.6, TRUTH, |x| source.density(x));
                let beyond = h.total - h.counts.iter().map(|&c| c as u64).sum::<u64>();
                if beyond > 0 {
                    let text = format!("and {} beyond the chart", count(beyond as usize));
                    f.text_px(Vec2::new(max.x - 4.0, min.y + 4.0), text, lay.small, pal::DIM, Align2::RIGHT_TOP);
                }
            }
        }
    }

    fn draw_table(&self, f: &mut Frame, lay: &Layout, (min, max): Rect) {
        let row = lay.small + if lay.narrow { 5.0 } else { 7.0 };
        let width = max.x - min.x;
        let cols = [min.x + width * 0.62, min.x + width * 0.86];
        let runs = self.runs.len();
        f.text_px(min, format!("at the markers, over the {runs} thin runs"), lay.small, pal::TEXT, Align2::LEFT_TOP);
        let readings = self.marks.map(|g| self.reading(g));
        for (k, &g) in self.marks.iter().enumerate() {
            f.text_px(Vec2::new(cols[k], min.y + row), format!("$n = {}$", count(grid()[g] as usize)), lay.small, pal::WHITE, Align2::CENTER_TOP);
        }
        let mut rows: Vec<(String, [String; 2], Color)> = Vec::new();
        match (self.mean(), self.sd()) {
            (Some(_), Some(s)) => {
                rows.push(("inside the funnel".to_string(), readings.map(|r| r.and_then(|r| r.inside.map(|i| format!("{i} of {}", r.count))).unwrap_or_else(|| "—".to_string())), ESTIMATE));
                rows.push((r"typical distance from $\mu$".to_string(), readings.map(|r| r.and_then(|r| r.rms).map_or("—".to_string(), |v| format!("${}$", sig(v)))), ESTIMATE));
                rows.push((r"$\sigma/\sqrt{n}$".to_string(), self.marks.map(|g| format!("${}$", sig(s / (grid()[g] as f64).sqrt()))), TRUTH));
                let unit = self.source.unit();
                let name = if self.source == Source::Coin { "typical gap in heads" } else if self.source == Source::Darts { "typical gap in points inside" } else { "typical gap in the total" };
                rows.push((name.to_string(), readings.map(|r| r.and_then(|r| r.rms.map(|v| format!("${}$", sig(v * r.n as f64 / unit)))).unwrap_or_else(|| "—".to_string())), OTHER));
            }
            _ => {
                rows.push(("middle half of the averages: width".to_string(), readings.map(|r| r.map_or("—".to_string(), |r| format!("${}$", sig(r.quartiles.1 - r.quartiles.0)))), ESTIMATE));
                let one = CAUCHY.quantile(0.75) - CAUCHY.quantile(0.25);
                rows.push(("a single draw's middle half".to_string(), [format!("${}$", sig(one)), format!("${}$", sig(one))], TRUTH));
            }
        }
        for (i, (name, values, color)) in rows.iter().enumerate() {
            let y = min.y + (i + 2) as f32 * row + 2.0;
            f.text_px(Vec2::new(min.x, y), name.as_str(), lay.small, pal::DIM, Align2::LEFT_TOP);
            for k in 0..2 {
                f.text_px(Vec2::new(cols[k], y), values[k].as_str(), lay.small, *color, Align2::CENTER_TOP);
            }
        }
    }

    fn draw_key(&self, f: &mut Frame, lay: &Layout, (min, _max): Rect) {
        let entries: Vec<(Color, &str)> = vec![
            (ESTIMATE, "the average so far, in each run"),
            (TRUTH, "the truth: the mean, the funnel"),
            (DATA, "the bold run's draws"),
            (OTHER, "the total less what the mean predicts"),
            (ALARM, "a run outside the funnel, at a marker"),
            (pal::WHITE, "markers (drag them along $n$), with the runs' typical distance"),
        ];
        hud_key(f, min, &entries, lay.small);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(n: u32) -> usize {
        grid().binary_search(&n).expect("on the grid")
    }

    /// Many runs' averages at the given n (each a column).
    fn many(source: Source, p: f64, tails: bool, seed: u64, runs: usize) -> Vec<Vec<f64>> {
        (0..runs).map(|k| averages(source, p, tails, run_seed(seed, k))).collect()
    }

    /// The runs' averages at n.
    fn column(runs: &[Vec<f64>], n: u32) -> Vec<f64> {
        runs.iter().map(|r| r[at(n)]).collect()
    }

    /// The root mean square distance of the values from `m`, and their mean distance.
    fn rms(values: &[f64], m: f64) -> f64 {
        (values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / values.len() as f64).sqrt()
    }

    /// The share of the values within `half` of `m`.
    fn within(values: &[f64], m: f64, half: f64) -> f64 {
        values.iter().filter(|v| (*v - m).abs() <= half).count() as f64 / values.len() as f64
    }

    /// The distance between the quartiles.
    fn middle_half(values: &[f64]) -> f64 {
        let s = st::sorted(values);
        st::quantile_sorted(&s, 0.75) - st::quantile_sorted(&s, 0.25)
    }

    /// An experiment as its button sets it up, its runs all made.
    fn trial(k: usize) -> Lln {
        let mut s = Lln::create();
        s.set_up(k);
        s.settle();
        s
    }

    /// How many runs these tests count over (the captions' numbers were measured over
    /// 8 000: `explore`).
    const MANY: usize = 2000;

    /// The sources' means and spreads, from `Dist` and from what the die and the humps
    /// are made of, are what their draws show; the Cauchy has neither.
    #[test]
    fn the_sources_have_the_moments_their_draws_show() {
        assert_eq!(Source::Die.mean(0.5), Some(3.5));
        assert!((Source::Die.sd(0.5).unwrap() - (35.0f64 / 12.0).sqrt()).abs() < 1e-12);
        assert!((Source::Humps.sd(0.5).unwrap() - 4.25f64.sqrt()).abs() < 1e-12);
        assert!((Source::Darts.mean(0.5).unwrap() - std::f64::consts::PI).abs() < 1e-12);
        assert_eq!(Source::Coin.sd(0.3), Dist::Bernoulli { p: 0.3 }.sd());
        assert_eq!((Source::Cauchy.mean(0.5), Source::Cauchy.sd(0.5)), (None, None));
        let mut rng = Rng::new(9);
        for source in SOURCES {
            for p in [0.3, 0.5] {
                let x: Vec<f64> = (0..200_000).map(|_| source.draw(p, &mut rng)).collect();
                assert!(x.iter().all(|v| v.is_finite()), "{source:?}");
                match (source.mean(p), source.sd(p)) {
                    (Some(m), Some(s)) => {
                        let se = s / (x.len() as f64).sqrt();
                        assert!((st::mean(&x) - m).abs() < 4.0 * se, "{source:?}: mean {} for {m}", st::mean(&x));
                        assert!((st::sd(&x) / s - 1.0).abs() < 0.02, "{source:?}: sd {} for {s}", st::sd(&x));
                    }
                    _ => assert!((middle_half(&x) - 2.0).abs() < 0.03, "Cauchy: quartiles {} apart", middle_half(&x)),
                }
            }
        }
    }

    /// Every n up to 100, then round numbers (among them those the experiments mark),
    /// rising; and a place found from any n is the nearest.
    #[test]
    fn the_grid_has_its_round_numbers() {
        let g = grid();
        assert!(g.windows(2).all(|w| w[0] < w[1]) && g[0] == 1 && *g.last().unwrap() == LONGEST as u32);
        assert!((1..=100).all(|n| g.binary_search(&n).is_ok()));
        for n in [200u32, 250, 400, 1000, 2500, 4000, 10_000] {
            assert_eq!(grid_index(n as f64), at(n));
        }
        assert_eq!((grid_index(0.0), grid_index(1e9), grid_index(f64::NAN)), (0, g.len() - 1, 0));
        assert_eq!(g[grid_index(1003.0)], 1000);
        // and the axis can be read back, on either scale and between them
        for t in [0.0, 0.4, 1.0] {
            for n in [1.0, 7.0, 400.0, 9999.0] {
                assert!((n_at(across(n, t), t) / n - 1.0).abs() < 1e-6, "{n} at {t}");
            }
        }
    }

    /// A run's averages are its draws added up and divided by their number, and the
    /// bold run, drawn a draw at a time, comes to the same averages from the same seed.
    #[test]
    fn a_run_is_the_running_average_of_its_draws() {
        let mut rng = Rng::new(5);
        let (mut sum, mut by_hand) = (0.0, Vec::new());
        for n in 1..=LONGEST {
            sum += Source::Die.draw(0.5, &mut rng);
            if grid().binary_search(&(n as u32)).is_ok() {
                by_hand.push(sum / n as f64);
            }
        }
        assert_eq!(averages(Source::Die, 0.5, false, 5), by_hand);
        for source in SOURCES {
            let mut b = Bold::new(source, 0.5, true, 11);
            while b.n < LONGEST {
                b.step(source, 0.5, true);
            }
            assert_eq!(b.at, averages(source, 0.5, true, 11), "{source:?}");
            assert_eq!(b.hist.total, LONGEST as u64);
        }
        // the coin's ten tails, when asked for, and only for the coin
        let tails = averages(Source::Coin, 0.5, true, 3);
        assert!(tails[..TAILS].iter().all(|a| *a == 0.0));
        assert!(averages(Source::Die, 0.5, true, 3)[0] >= 1.0);
    }

    /// The die: 0.54 from 3.5 after 10 rolls, 0.054 after 1 000, and over 95 % of runs
    /// within 0.11 by then; the opening's dozen runs show about that.
    #[test]
    fn a_die_settles_as_its_caption_says() {
        let runs = many(Source::Die, 0.5, false, 77, MANY);
        let (ten, thousand) = (column(&runs, 10), column(&runs, 1000));
        let (r10, r1000) = (rms(&ten, 3.5), rms(&thousand, 3.5));
        assert!((r10 - 0.54).abs() < 0.02 && (r1000 - 0.054).abs() < 0.002, "{r10} and {r1000}");
        let near = within(&thousand, 3.5, 0.11);
        assert!(near > 0.95 && near < 0.97, "{near} within 0.11 at 1 000");
        // the opening
        let s = trial(OPENING);
        assert_eq!((s.source, s.runs.len(), s.marks.map(|g| grid()[g])), (Source::Die, 12, [10, 1000]));
        let sd = s.sd().unwrap();
        for g in s.marks {
            let r = s.reading(g).unwrap();
            let theory = sd / (r.n as f64).sqrt();
            assert!((r.rms.unwrap() / theory - 1.0).abs() < 0.08, "the dozen runs at {}: {} for {theory}", r.n, r.rms.unwrap());
            assert_eq!(r.inside, Some(12));
        }
        let created = Lln::create();
        assert_eq!((created.source, created.seed, created.wanted), (s.source, s.seed, s.wanted));
    }

    /// Two hundred runs: about 95 % inside the funnel at each n, the typical distance
    /// halved by four times the rolls, and runs inside at one marker and outside at the
    /// other.
    #[test]
    fn many_runs_fill_the_funnel_as_the_caption_says() {
        let s = trial(1);
        assert_eq!((s.runs.len(), s.marks.map(|g| grid()[g])), (200, [100, 400]));
        let [a, b] = s.marks.map(|g| s.reading(g).unwrap());
        assert_eq!((a.inside, b.inside), (Some(191), Some(193)));
        assert_eq!((sig(a.rms.unwrap()), sig(b.rms.unwrap())), ("0.17".to_string(), "0.085".to_string()));
        let switched = s.runs.iter().filter(|r| s.outside(r.avg[s.marks[0]], s.marks[0]) != s.outside(r.avg[s.marks[1]], s.marks[1])).count();
        assert_eq!(switched, 14);
        // over many more runs
        let runs = many(Source::Die, 0.5, false, 78, MANY);
        let (sd, z) = (Source::Die.sd(0.5).unwrap(), z95());
        for n in [10u32, 100, 400, 1000] {
            let share = within(&column(&runs, n), 3.5, z * sd / (n as f64).sqrt());
            assert!((share - 0.95).abs() < 0.01, "{share} inside at {n}");
        }
        let ratio = rms(&column(&runs, 400), 3.5) / rms(&column(&runs, 100), 3.5);
        assert!((ratio - 0.5).abs() < 0.02, "four times the rolls: {ratio} of the distance");
        // whatever n, at least 1 − 1/z² are inside (Chebyshev), for every source with a mean
        for source in [Source::Die, Source::Coin, Source::Wait, Source::Humps, Source::Darts] {
            let runs = many(source, 0.5, false, 79, 400);
            let (m, sd) = (source.mean(0.5).unwrap(), source.sd(0.5).unwrap());
            for n in [1u32, 2, 3, 5, 10, 100] {
                let share = within(&column(&runs, n), m, z * sd / (n as f64).sqrt() + 1e-12);
                assert!(share >= 1.0 - 1.0 / (z * z) - 0.03, "{source:?} at {n}: {share}");
            }
        }
    }

    /// After ten tails: the eleventh toss is heads half the time, the share closes on
    /// one half, and the count of heads less half the tosses comes back to 0 again and
    /// again, while its typical distance from 0 grows (6.9 at 100, 50 at 10 000); on
    /// average it stays 5 short.
    #[test]
    fn after_ten_tails_the_count_does_not_catch_up() {
        let runs = many(Source::Coin, 0.5, true, 80, MANY);
        let gaps = |n: u32| column(&runs, n).iter().map(|a| (a - 0.5) * n as f64).collect::<Vec<f64>>();
        let (g100, g10k) = (gaps(100), gaps(10_000));
        let (r100, r10k) = (rms(&g100, 0.0), rms(&g10k, 0.0));
        assert!((r100 - 6.9).abs() < 0.3 && (r10k - 50.0).abs() < 2.5, "typical gaps {r100} and {r10k}");
        let (m100, m10k) = (st::mean(&g100), st::mean(&g10k));
        // (the mean of 2 000 gaps of spread 50 is good to about 1.1)
        assert!((m100 + 5.0).abs() < 0.4 && (m10k + 5.0).abs() < 3.5, "mean gaps {m100} and {m10k}");
        // back to 0 again and again: the share of runs in which the count is 0, or
        // changes sign, between `from` and `to` tosses (seen at the grid's tosses only,
        // so if anything too few)
        let returns = |from: u32, to: u32| {
            let back = |r: &Vec<f64>| {
                let g: Vec<f64> = grid().iter().zip(r).filter(|(n, _)| (from..=to).contains(*n)).map(|(&n, a)| (a - 0.5) * n as f64).collect();
                g.iter().any(|v| v.abs() < 1e-9) || g.windows(2).any(|w| w[0] * w[1] < 0.0)
            };
            runs.iter().filter(|r| back(r)).count() as f64 / runs.len() as f64
        };
        let (early, late) = (returns(11, 1000), returns(1000, 10_000));
        assert!(early > 0.65 && late > 0.65, "back to 0 in {early} of the runs by 1 000 tosses, in {late} between 1 000 and 10 000");
        // the eleventh toss: heads in half the runs
        let heads = column(&runs, 11).iter().filter(|a| (*a * 11.0 - 1.0).abs() < 1e-9).count() as f64 / MANY as f64;
        assert!((heads - 0.5).abs() < 0.035, "{heads} heads on the eleventh toss");
        // and the share of heads is back near one half
        let share = st::mean(&column(&runs, 10_000));
        assert!((share - 0.5).abs() < 0.001, "share {share}");
        // the experiment
        let s = trial(2);
        assert!(s.source == Source::Coin && s.tails && s.p == 0.5 && s.marks.map(|g| grid()[g]) == [100, 10_000]);
        assert!(s.bold.at[..TAILS].iter().all(|a| *a == 0.0));
    }

    /// The Cauchy: the middle half of the averages is 2 wide at 1 draw and at 1 000, and
    /// two runs in three still move by more than 1 between 1 000 and 10 000 draws (a die
    /// never does).
    #[test]
    fn the_cauchy_averages_never_settle() {
        let runs = many(Source::Cauchy, 0.5, false, 81, MANY);
        let (one, thousand) = (middle_half(&column(&runs, 1)), middle_half(&column(&runs, 1000)));
        assert!((one - 2.0).abs() < 0.15 && (thousand - 2.0).abs() < 0.15, "{one} and {thousand}");
        let moved = |runs: &[Vec<f64>]| runs.iter().filter(|r| (r[at(10_000)] - r[at(1000)]).abs() > 1.0).count() as f64 / runs.len() as f64;
        let share = moved(&runs);
        assert!((share - 0.677).abs() < 0.035, "{share} moved by more than 1");
        assert_eq!(moved(&many(Source::Die, 0.5, false, 81, 200)), 0.0);
        // the experiment's 200 runs show it too
        let s = trial(3);
        for g in s.marks {
            let r = s.reading(g).unwrap();
            let width = r.quartiles.1 - r.quartiles.0;
            assert!(r.inside.is_none() && r.rms.is_none() && (width - 2.0).abs() < 0.1, "{width} at {}", r.n);
            assert_eq!(sig(width), "2.0");
        }
    }

    /// π from points in a square: typically 0.16 off after 100 points, 0.016 after 10 000.
    #[test]
    fn darts_find_pi_as_the_caption_says() {
        let runs = many(Source::Darts, 0.5, false, 82, MANY);
        let pi = std::f64::consts::PI;
        let (r100, r10k) = (rms(&column(&runs, 100), pi), rms(&column(&runs, 10_000), pi));
        assert!((r100 - 0.164).abs() < 0.008 && (r10k - 0.0164).abs() < 0.0008, "{r100} and {r10k}");
        let s = trial(4);
        assert!(s.source == Source::Darts && s.marks.map(|g| grid()[g]) == [100, 10_000]);
        // the throws drawn are the bold run's own: four times the share inside is its average
        let inside = s.bold.hist.counts[1] as f64;
        assert!((4.0 * inside / LONGEST as f64 - s.bold.average()).abs() < 1e-12);
        assert_eq!(s.bold.throws.len(), THROWS_DRAWN);
        assert_eq!(s.bold.throws.iter().filter(|t| t.2).count(), s.bold.throws.iter().filter(|t| t.0 * t.0 + t.1 * t.1 <= 1.0).count());
    }

    /// A coin that always lands heads has no spread: its funnel has no width, every
    /// average is exactly 1, every run is inside, and the total never leaves n p.
    #[test]
    fn a_source_with_no_spread_is_a_level_line() {
        let mut s = Lln::create();
        (s.source, s.p) = (Source::Coin, 1.0);
        s.settle();
        assert_eq!(s.sd(), Some(0.0));
        assert!(s.runs.iter().all(|r| r.avg.iter().all(|a| *a == 1.0)));
        for g in s.marks {
            let r = s.reading(g).unwrap();
            assert_eq!((r.inside, r.rms), (Some(s.runs.len()), Some(0.0)));
        }
        let (n, avg, gap) = s.bold.head(1.0);
        assert_eq!((n, avg, gap), (LONGEST as f64, 1.0, 0.0));
        // with ten tails first it is outside the funnel of no width all the way, and still finite
        s.tails = true;
        s.settle();
        let r = s.reading(s.marks[1]).unwrap();
        assert_eq!(r.inside, Some(0));
        assert!(s.bold.head(1.0).2 == -(TAILS as f64));
    }

    /// The bold run crosses the chart evenly at every display rate: each frame it moves
    /// the frame's share of the chart, on either axis, and it ends when it should.
    #[test]
    fn the_bold_run_moves_evenly() {
        for log in [true, false] {
            for hz in [60.0f32, 75.0, 120.0, 144.0] {
                let mut s = Lln::create();
                s.log = log;
                let t = if log { 0.0 } else { 1.0 };
                let dt = 1.0 / hz;
                let (mut last, mut frames) = (across(s.bold.head(3.5).0, t), 0);
                while !s.bold.done() {
                    s.bold.advance(dt, s.seconds, s.log, s.source, s.p, s.tails);
                    let (n, avg, _) = s.bold.head(3.5);
                    let x = across(n, t);
                    frames += 1;
                    let step = (x - last) * s.seconds as f64 / dt as f64;
                    // (the last frame takes only what is left)
                    assert!(s.bold.done() || (step - if log { 1.0 } else { 1.0 - 1.0 / LONGEST as f64 }).abs() < 1e-6, "{hz} Hz, frame {frames}: {step}");
                    assert!((1.0..=6.0).contains(&avg) && n <= LONGEST as f64 && s.bold.n as f64 >= n);
                    last = x;
                }
                let seconds = frames as f32 * dt;
                assert!((seconds - s.seconds).abs() <= dt, "{hz} Hz: {seconds} s");
            }
        }
    }

    /// The head is drawn following the run through a short ease: it moves in every
    /// frame at every display rate, never more than a few hundredths of a second
    /// behind, and comes exactly to the run's end, where the picture comes to rest.
    #[test]
    fn the_head_drawn_follows_the_run_closely() {
        for hz in [60.0f32, 75.0, 120.0, 144.0] {
            let mut s = Lln::create();
            let dt = 1.0 / hz;
            // (the first frame puts the head where the run starts)
            assert!(!s.ease_head(dt));
            let mut last = s.head().0;
            let per_second = (LONGEST as f64).ln() / s.seconds as f64;
            let mut frames = 0;
            loop {
                s.bold.advance(dt, s.seconds, s.log, s.source, s.p, s.tails);
                let moving = s.ease_head(dt);
                let (n, avg, gap) = s.head();
                frames += 1;
                if !moving {
                    assert!(s.bold.done() && (n, avg, gap) == s.bold.head(3.5), "{hz} Hz: at rest before the end");
                    break;
                }
                assert!(n > last, "{hz} Hz, frame {frames}: no move");
                // (two eases of 0.03 s: about 0.06 s behind)
                let behind = (s.bold.p.ln() - n.ln()) / per_second;
                assert!(behind < 0.08, "{hz} Hz, frame {frames}: {behind} s behind");
                last = n;
                assert!(frames as f32 * dt < s.seconds + 1.0, "{hz} Hz: still moving");
            }
        }
    }

    /// However long a frame, the bold run makes a bounded number of draws in it (and
    /// falls behind instead), and new runs come at their pace.
    #[test]
    fn the_work_in_a_frame_is_bounded() {
        let mut s = Lln::create();
        s.set_up(1);
        s.seconds = 2.0;
        s.work(1.0 / 60.0, 1.0 / 60.0);
        assert!(s.runs.len() <= (RUN_RATE / 60.0).ceil() as usize + 1, "{} runs in a frame", s.runs.len());
        let before = s.bold.n;
        s.bold.advance(30.0, 2.0, true, s.source, s.p, s.tails);
        assert!(s.bold.n - before <= MOST_DRAWS && s.bold.p == s.bold.n as f64);
    }

    /// Another run starts the bold run afresh and keeps the thin ones; start over makes
    /// them all again; a changed source makes them for it.
    #[test]
    fn another_run_keeps_the_thin_ones() {
        let mut s = trial(OPENING);
        let (thin, bold) = (s.runs[3].avg.clone(), s.bold.at.clone());
        s.bolds += 1;
        s.refresh();
        assert_eq!(s.runs.len(), 12);
        assert!(s.runs[3].avg == thin && s.bold.n == 1);
        s.settle();
        assert!(s.bold.at != bold);
        s.source = Source::Wait;
        s.refresh();
        assert!(s.runs.is_empty() && s.bold.n == 1);
    }

    /// Every experiment sets everything its lesson depends on, whatever was done before.
    #[test]
    fn the_experiments_set_every_setting() {
        for k in 0..TRIALS.len() {
            let mut a = Lln::create();
            a.set_up(k);
            let mut b = Lln::create();
            (b.source, b.p, b.tails, b.wanted, b.seconds, b.log, b.seed, b.bolds) = (Source::Humps, 0.2, true, 77, 3.0, false, 999, 4);
            b.marks = [3, 7];
            b.set_up(k);
            assert_eq!((a.source, a.p, a.tails, a.wanted, a.seconds, a.log, a.seed, a.bolds, a.marks), (b.source, b.p, b.tails, b.wanted, b.seconds, b.log, b.seed, b.bolds, b.marks), "{k}");
            a.settle();
            b.settle();
            assert!(a.bold.at == b.bold.at && a.runs.iter().zip(&b.runs).all(|(x, y)| x.avg == y.avg));
        }
    }

    /// The captions, buttons and names are markup that sets.
    #[test]
    fn the_words_are_markup_that_sets() {
        let texts = TRIALS.iter().flat_map(|t| [t.button, t.caption]).chain(SOURCES.iter().map(|s| s.name()));
        for text in texts {
            let found = crate::typeset::inspect(text);
            assert!(found.error.is_none() && found.words.is_empty(), "{text}: {found:?}");
        }
    }

    /// Every part of the picture is inside the window, and no two charts cover each
    /// other, from a desktop to a phone to a sliver.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h, inset) in [(1078.0, 1000.0, 0.0), (1600.0, 1000.0, 0.0), (700.0, 500.0, 0.0), (390.0, 800.0, 44.0), (390.0, 400.0, 44.0), (240.0, 180.0, 0.0), (1.0, 1.0, 0.0)] {
            let view = Vec2::new(w, h);
            let lay = Layout::of(view, inset);
            let mut rects = vec![lay.main, lay.again];
            rects.extend(lay.sources);
            rects.extend([lay.gap, lay.sample, lay.table, lay.key].into_iter().flatten());
            for (min, max) in &rects {
                assert!(min.is_finite() && max.is_finite() && max.x >= min.x && max.y >= min.y, "{view}: {min} {max}");
            }
            if w < 300.0 || h < 300.0 {
                continue;
            }
            for (min, max) in &rects {
                assert!(min.x >= 0.0 && min.y >= inset && max.x <= w && max.y <= h, "{view}: {min} {max} outside");
            }
            let charts: Vec<Rect> = [Some(lay.main), lay.gap, lay.sample, lay.table].into_iter().flatten().collect();
            for (i, a) in charts.iter().enumerate() {
                for b in &charts[i + 1..] {
                    let apart = a.1.x <= b.0.x || b.1.x <= a.0.x || a.1.y <= b.0.y || b.1.y <= a.0.y;
                    assert!(apart, "{view}: {a:?} and {b:?} overlap");
                }
            }
        }
    }

    /// What each experiment's runs show at its markers, for a few seeds (run with
    /// --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore_seeds() {
        for k in 0..TRIALS.len() {
            for seed in 1..=24u64 {
                let mut s = Lln::create();
                s.set_up(k);
                s.seed = seed;
                s.refresh();
                while s.runs.len() < s.wanted {
                    s.add_run();
                }
                let r = s.marks.map(|g| s.reading(g).unwrap());
                let theory = s.marks.map(|g| s.sd().map_or(f64::NAN, |sd| sd / (grid()[g] as f64).sqrt()));
                let switched = s.runs.iter().filter(|run| s.outside(run.avg[s.marks[0]], s.marks[0]) != s.outside(run.avg[s.marks[1]], s.marks[1])).count();
                println!(
                    "{:<16} seed {seed:>2}: inside {:?} {:?}, rms/theory {:.2} {:.2}, gap rms {:.1} {:.1}, iqr {:.2} {:.2}, switched {switched}",
                    TRIALS[k].button,
                    r[0].inside,
                    r[1].inside,
                    r[0].rms.unwrap_or(f64::NAN) / theory[0],
                    r[1].rms.unwrap_or(f64::NAN) / theory[1],
                    r[0].rms.unwrap_or(f64::NAN) * r[0].n as f64,
                    r[1].rms.unwrap_or(f64::NAN) * r[1].n as f64,
                    r[0].quartiles.1 - r[0].quartiles.0,
                    r[1].quartiles.1 - r[1].quartiles.0,
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn explore() {
        let z = st::normal_quantile(0.975);
        for (source, p, tails) in [(Source::Die, 0.5, false), (Source::Coin, 0.5, false), (Source::Coin, 0.5, true), (Source::Wait, 0.5, false), (Source::Humps, 0.5, false), (Source::Darts, 0.5, false)] {
            let (m, s) = (source.mean(p).unwrap(), source.sd(p).unwrap());
            let runs = many(source, p, tails, 77, 8000);
            println!("{source:?} tails {tails}: mean {m}, sd {s}");
            for n in [1u32, 10, 100, 400, 1000, 10000] {
                let g = at(n);
                let d: Vec<f64> = runs.iter().map(|r| r[g] - m).collect();
                let rms = (d.iter().map(|v| v * v).sum::<f64>() / d.len() as f64).sqrt();
                let inside = d.iter().filter(|v| v.abs() <= z * s / (n as f64).sqrt() + 1e-12).count() as f64 / d.len() as f64;
                let gap: Vec<f64> = d.iter().map(|v| v * n as f64).collect();
                let gap_rms = (gap.iter().map(|v| v * v).sum::<f64>() / gap.len() as f64).sqrt();
                let gap_mean = gap.iter().sum::<f64>() / gap.len() as f64;
                let within01 = d.iter().filter(|v| v.abs() <= 0.1).count() as f64 / d.len() as f64;
                println!(
                    "  n {n:>5}: rms {rms:.4} (σ/√n {:.4}), inside {:.4}, gap rms {gap_rms:.2} mean {gap_mean:.2}, within 0.1 {within01:.4}",
                    s / (n as f64).sqrt(),
                    inside
                );
            }
        }
        // the Cauchy: the middle half of the averages
        let runs = many(Source::Cauchy, 0.5, false, 77, 8000);
        for n in [1u32, 10, 100, 1000, 10000] {
            let g = at(n);
            let s = st::sorted(&runs.iter().map(|r| r[g]).collect::<Vec<_>>());
            println!("Cauchy n {n:>5}: quartiles {:.3} {:.3}, width {:.3}", st::quantile_sorted(&s, 0.25), st::quantile_sorted(&s, 0.75), st::quantile_sorted(&s, 0.75) - st::quantile_sorted(&s, 0.25));
        }
    }
}

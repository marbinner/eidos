//! Confidence intervals: sample after sample from a population whose truth is known,
//! each made into an interval by one recipe, in a column with the truth standing still
//! through it; the count of those that cover the truth; and one of them taken apart.

use super::plot::{self, Glide, Plot, Rect};
use super::*;
use crate::clock::Budget;
use crate::math::stats::{self as st, Dist};

pub const DEF: SceneDef = SceneDef {
    id: "intervals",
    title: "Confidence intervals",
    group: ESTIMATION,
    blurb: "A hundred intervals from a hundred samples: what 95 % does and does not promise, and what decides how wide an interval is.",
    make: |_| Box::new(Intervals::create()),
};

// ----- populations and recipes ---------------------------------------------------------

/// What the samples are drawn from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Population {
    /// Normal, mean μ, standard deviation [`BELL_SD`].
    Bell,
    /// Log-normal with mean m: most values small, a few very large.
    Tail,
    /// Each value 1 with probability p, else 0.
    Share,
}

impl Population {
    const ALL: [Population; 3] = [Population::Bell, Population::Tail, Population::Share];

    fn index(self) -> usize {
        match self {
            Population::Bell => 0,
            Population::Tail => 1,
            Population::Share => 2,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Population::Bell => "a bell",
            Population::Tail => "a long right tail",
            Population::Share => "a share",
        }
    }
}

/// The bell's standard deviation.
const BELL_SD: f64 = 2.0;

/// The ranges of the true values, and where each population starts.
const TRUTH_RANGE: [(f64, f64); 3] = [(0.0, 20.0), (1.0, 20.0), (0.0, 1.0)];
const TRUTH_START: [f64; 3] = [10.0, 10.0, 0.3];

/// The recipe that makes an interval from a sample.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Method {
    /// x̄ ± z σ/√n, with the population's σ known.
    KnownZ,
    /// x̄ ± t s/√n: σ estimated, Student's t.
    StudentT,
    /// x̄ ± z s/√n: σ estimated, and the normal critical value used anyway.
    PlugInZ,
    /// p̂ ± z √(p̂(1 − p̂)/n): the usual interval for a share.
    Wald,
    /// Wilson's interval for a share.
    Wilson,
}

impl Method {
    const FOR_MEANS: [Method; 3] = [Method::KnownZ, Method::StudentT, Method::PlugInZ];
    const FOR_SHARES: [Method; 2] = [Method::Wald, Method::Wilson];

    /// Its name on a button.
    fn name(self) -> &'static str {
        match self {
            Method::KnownZ => r"$\sigma$ known: $z$",
            Method::StudentT => r"$s$ and Student's $t$",
            Method::PlugInZ => r"$s$, but $z$",
            Method::Wald => r"the usual: $\hat p \pm z\,\text{SE}$",
            Method::Wilson => "Wilson's",
        }
    }

    /// The recipe as a formula.
    fn formula(self) -> &'static str {
        match self {
            Method::KnownZ => r"$\bar x \pm z \cdot \sigma/\sqrt{n}$",
            Method::StudentT => r"$\bar x \pm t_{n-1} \cdot s/\sqrt{n}$",
            Method::PlugInZ => r"$\bar x \pm z \cdot s/\sqrt{n}$",
            Method::Wald => r"$\hat p \pm z \sqrt{\hat p(1 - \hat p)/n}$",
            Method::Wilson => "Wilson's interval",
        }
    }

    /// Does it need the spread estimated from the sample (two values at least)?
    fn estimates_spread(self) -> bool {
        matches!(self, Method::StudentT | Method::PlugInZ)
    }
}

/// What a critical value depends on: the recipe, n and the level.
type CritKey = (Method, usize, usize);

/// The confidence levels offered.
const LEVELS: [f64; 6] = [0.5, 0.8, 0.9, 0.95, 0.99, 1.0];
const LEVEL_NAMES: [&str; 6] = ["50 %", "80 %", "90 %", "95 %", "99 %", "100 %"];
/// The index of 95 %.
const L95: usize = 3;

/// Everything an interval depends on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    pop: Population,
    /// The true mean (bell, tail) or share.
    truth: f64,
    n: usize,
    level: usize,
    method: Method,
}

impl Setup {
    fn level(&self) -> f64 {
        LEVELS[self.level.min(LEVELS.len() - 1)]
    }

    /// The population's standard deviation.
    fn sigma(&self) -> f64 {
        match self.pop {
            Population::Bell => BELL_SD,
            Population::Tail => self.truth * (std::f64::consts::E - 1.0).sqrt(),
            Population::Share => (self.truth * (1.0 - self.truth)).max(0.0).sqrt(),
        }
    }

    /// The critical value: how many standard errors the interval reaches either side,
    /// from the normal distribution or Student's t with n − 1 degrees of freedom (∞ at a
    /// level of 100 %; not a number for t with fewer than two values).
    fn critical(&self) -> f64 {
        let upper = 1.0 - (1.0 - self.level()) / 2.0;
        match self.method {
            Method::StudentT if self.n < 2 => f64::NAN,
            Method::StudentT => Dist::StudentT { df: (self.n - 1) as f64 }.quantile(upper),
            _ => Dist::STANDARD_NORMAL.quantile(upper),
        }
    }

    /// No interval at all: the spread is to be estimated from a single value.
    fn no_interval(&self) -> bool {
        self.method.estimates_spread() && self.n < 2
    }

    /// The interval this recipe makes of a sample (None if it makes none).
    #[cfg(test)]
    fn read(&self, s: Summary) -> Option<Iv> {
        self.read_at(s, self.critical())
    }

    /// The same with the critical value given (a t value is found by a search: worked
    /// out once, not for every sample).
    fn read_at(&self, s: Summary, c: f64) -> Option<Iv> {
        if self.no_interval() {
            return None;
        }
        let n = self.n as f64;
        let (mean, sd) = self.scaled(s);
        Some(match self.method {
            Method::KnownZ | Method::StudentT | Method::PlugInZ => {
                let spread = if self.method == Method::KnownZ { self.sigma() } else { sd };
                let m = margin(c, spread / n.sqrt());
                Iv { est: mean, lo: mean - m, hi: mean + m }
            }
            Method::Wald => {
                let m = margin(c, (mean * (1.0 - mean) / n).max(0.0).sqrt());
                Iv { est: mean, lo: mean - m, hi: mean + m }
            }
            Method::Wilson => {
                let (centre, half) = wilson(mean, n, c);
                Iv { est: mean, lo: (centre - half).max(0.0), hi: (centre + half).min(1.0) }
            }
        })
    }

    /// A sample's mean and standard deviation in the population's units.
    fn scaled(&self, s: Summary) -> (f64, f64) {
        match self.pop {
            Population::Bell => (self.truth + BELL_SD * s.mean, BELL_SD * s.sd),
            Population::Tail => (self.truth * s.mean, self.truth * s.sd),
            Population::Share => (s.mean, s.sd),
        }
    }

    /// One value of the population, from a raw draw (a standard normal number, or for a
    /// share an even one between 0 and 1).
    fn value(&self, r: f64) -> f64 {
        value_of(self.pop, self.truth, r)
    }
}

fn value_of(pop: Population, truth: f64, r: f64) -> f64 {
    match pop {
        Population::Bell => truth + BELL_SD * r,
        Population::Tail => truth * (r - 0.5).exp(),
        Population::Share => {
            if r < truth {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Critical value times standard error; nothing when there is no spread, whatever the
/// critical value (also an infinite one: the limit of the interval is then its point).
fn margin(c: f64, se: f64) -> f64 {
    if se == 0.0 {
        0.0
    } else {
        c * se
    }
}

/// Wilson's interval for a share: its centre and half-width (at an infinite critical
/// value, the limit: all of 0 to 1).
fn wilson(p: f64, n: f64, c: f64) -> (f64, f64) {
    if c.is_infinite() {
        return (0.5, 0.5);
    }
    let a = c * c / n;
    let centre = (p + a / 2.0) / (1.0 + a);
    let half = c / (1.0 + a) * (p * (1.0 - p) / n + a / (4.0 * n)).max(0.0).sqrt();
    (centre, half)
}

/// An interval: the estimate and the two ends.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Iv {
    est: f64,
    lo: f64,
    hi: f64,
}

/// Where an interval stands to the truth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    Covers,
    Below,
    Above,
}

impl Iv {
    fn verdict(&self, truth: f64) -> Verdict {
        if self.hi < truth {
            Verdict::Below
        } else if self.lo > truth {
            Verdict::Above
        } else {
            Verdict::Covers
        }
    }

    fn zero_width(&self) -> bool {
        self.hi - self.lo == 0.0
    }
}

// ----- samples ---------------------------------------------------------------------------

/// What is kept of each sample: its mean and standard deviation, for the bell in
/// standard units (so that the true mean can move without drawing again) and for the
/// tail with a true mean of 1; for a share, the share of ones (and no spread).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Summary {
    mean: f64,
    sd: f64,
}

/// The seed of every sample's own stream: sample `k` of the run with seed `seed` is
/// the same sample however many came before it, whatever the recipe, and with more
/// values asked of it its first values stay as they were.
fn sample_rng(seed: u64, k: usize) -> Rng {
    Rng::new(seed.wrapping_mul(0x0000_0001_0000_0001).wrapping_add(k as u64))
}

/// Sample `k`'s raw draws: standard normal numbers, or for a share even ones from 0 to 1.
fn raw(pop: Population, seed: u64, k: usize, n: usize, out: &mut Vec<f64>) {
    let mut rng = sample_rng(seed, k);
    out.clear();
    out.extend((0..n).map(|_| if pop == Population::Share { st::unit(&mut rng) } else { st::normal(&mut rng) }));
}

/// What sample `k` is kept as (see [`Summary`]); `share` is the true share.
fn summary(pop: Population, share: f64, seed: u64, k: usize, n: usize, buf: &mut Vec<f64>) -> Summary {
    raw(pop, seed, k, n, buf);
    match pop {
        Population::Bell => {}
        Population::Tail => buf.iter_mut().for_each(|r| *r = (*r - 0.5).exp()),
        Population::Share => buf.iter_mut().for_each(|r| *r = value_of(Population::Share, share, *r)),
    }
    Summary { mean: st::mean(buf), sd: if pop == Population::Share { 0.0 } else { st::sd(buf) } }
}

/// What the kept samples depend on: the population, the seed, n, and for a share the
/// true share (a bell's or a tail's samples are kept in units that do not need it).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Key {
    pop: Population,
    seed: u64,
    n: usize,
    share: u64,
}

impl Key {
    fn of(s: &Setup, seed: u64) -> Key {
        Key { pop: s.pop, seed, n: s.n, share: if s.pop == Population::Share { s.truth.to_bits() } else { 0 } }
    }
}

// ----- the run -----------------------------------------------------------------------------

/// The column holds this many intervals at rest (and one more on its way in or out).
const ROWS: usize = 100;
const SLOTS: usize = ROWS + 1;
/// Where a run stops, one after another: "go on" takes it to the next.
const GOALS: [usize; 3] = [100, 1000, 10_000];
/// The seed every run starts from ("new samples" moves it on).
const SEED: u64 = 196;

/// The counts for the samples so far, read by the setup.
#[derive(Clone, Debug, Default, PartialEq)]
struct Tally {
    count: usize,
    covered: usize,
    below: usize,
    above: usize,
    /// Intervals of no width at all (a share of 0 or 1 by the usual recipe).
    zero: usize,
    /// Of those, the ones that missed.
    zero_missed: usize,
    /// How many had covered after each sample.
    cum: Vec<u32>,
}

impl Tally {
    fn add(&mut self, iv: Option<Iv>, truth: f64) {
        self.count += 1;
        if let Some(iv) = iv {
            let v = iv.verdict(truth);
            match v {
                Verdict::Covers => self.covered += 1,
                Verdict::Below => self.below += 1,
                Verdict::Above => self.above += 1,
            }
            if iv.zero_width() {
                self.zero += 1;
                if v != Verdict::Covers {
                    self.zero_missed += 1;
                }
            }
        }
        self.cum.push(self.covered as u32);
    }

    /// The share covered so far (NaN before the first).
    fn share(&self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.covered as f64 / self.count as f64
        }
    }
}

/// The experiments: one setting each.
struct Trial {
    name: &'static str,
    look_for: &'static str,
    pop: Population,
    truth: f64,
    n: usize,
    level: usize,
    method: Method,
    rate: f32,
    goal: usize,
    show_truth: bool,
    /// Show what a single interval does not say.
    statement: bool,
}

const TRIALS: [Trial; 6] = [
    Trial {
        name: "a hundred intervals",
        look_for: r"A new sample of ten values from a bell every tenth of a second, each made into a 95 % interval ($\sigma$ known). The green line, the true mean $\mu = 10$, stays where it is; the intervals move. Of these hundred, 95 cover it and 5 miss: 3 wholly below, 2 wholly above. Which ones miss is luck. Press “go on” on the picture: by 10 000 samples the share that covers has settled near 95 %.",
        pop: Population::Bell,
        truth: 10.0,
        n: 10,
        level: L95,
        method: Method::KnownZ,
        rate: 10.0,
        goal: 100,
        show_truth: true,
        statement: false,
    },
    Trial {
        name: r"too narrow: $n = 5$ and $z$",
        look_for: r"Five values a sample, $\sigma$ estimated by the sample's own $s$, and the normal critical value 1.96 used anyway: in the long run only 88 % cover, not 95 %. With five values $s$ comes out below $\sigma$ in 59 % of samples, and then the interval is narrower than $\sigma$ would make it. Click “$s$ and Student's $t$” on the picture: the same samples, every interval 1.42 times as wide (2.78 against 1.96), and 95 % cover. Student's $t$ is exactly the correction for an estimated spread, when the values come from a bell.",
        pop: Population::Bell,
        truth: 10.0,
        n: 5,
        level: L95,
        method: Method::PlugInZ,
        rate: 100.0,
        goal: 1000,
        show_truth: true,
        statement: false,
    },
    Trial {
        name: "what sets the width",
        look_for: r"Twenty values a sample, $\sigma$ known: every interval reaches $1.96\,\sigma/\sqrt{20}$ either side of its mean. Set $n$ to 80 below: the same samples with sixty more values each, and every interval is exactly half as wide (four times the data, half the width), with 95 % still covering. Then click 99 % on the picture: 1.31 times as wide again; 50 %: about a third of the 95 % width, and in the long run half miss. Data buy narrowness; confidence is paid for in width.",
        pop: Population::Bell,
        truth: 10.0,
        n: 20,
        level: L95,
        method: Method::KnownZ,
        rate: 10.0,
        goal: 100,
        show_truth: true,
        statement: false,
    },
    Trial {
        name: "a long right tail",
        look_for: r"Ten values a sample from a population with a long right tail, and Student's $t$: only 84 % cover, and nearly every miss is below $\mu$ (16 % of samples miss below, 0.1 % above). A sample seldom holds one of the rare large values: the mean comes out below $\mu$ in 59 % of samples, and in 40 000 samples every interval that missed below had an $s$ smaller than $\sigma$: too low and too narrow at once. Student's $t$ assumes a bell.",
        pop: Population::Tail,
        truth: 10.0,
        n: 10,
        level: L95,
        method: Method::StudentT,
        rate: 100.0,
        goal: 1000,
        show_truth: true,
        statement: false,
    },
    Trial {
        name: "a share near zero",
        look_for: r"A share $p = 0.05$, thirty tries a sample, and the usual interval $\hat p \pm 1.96\,\text{SE}$: only 78 % cover. In 21 % of samples there is no success at all; then $\hat p = 0$, its SE is 0, and the interval is the single point 0, which misses. Click “Wilson's” on the picture: the same samples read again; over many samples 94 % cover, and none is a single point.",
        pop: Population::Share,
        truth: 0.05,
        n: 30,
        level: L95,
        method: Method::Wald,
        rate: 100.0,
        goal: 1000,
        show_truth: true,
        statement: false,
    },
    Trial {
        name: "what 95 % does not mean",
        look_for: r"One interval, with the truth hidden as it is in practice. “The true mean is in here with probability 95 %” is wrong: $\mu$ is a fixed number, in this interval or not, and once the sample is drawn there is no chance left in it. Press “show the truth”: it is in, and it was in before you looked. The 95 % belongs to the recipe: of the intervals it makes, 95 % cover. (A probability for $\mu$ itself needs a prior: see Bayesian updating.)",
        pop: Population::Bell,
        truth: 10.0,
        n: 10,
        level: L95,
        method: Method::KnownZ,
        rate: 10.0,
        goal: 1,
        show_truth: false,
        statement: true,
    },
];

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;

/// One place in the column: the sample drawn there (so that a new one is not taken
/// for the old one moved), and its interval as drawn (low end, high end, estimate).
#[derive(Clone, Debug, Default)]
struct Slot {
    k: Option<usize>,
    glide: Glide,
    /// The sample was new to this place in this frame.
    fresh: bool,
    /// Where its interval stands to the truth (None: there is none), and whether it is
    /// drawn where it is going (then that, not the drawn ends, gives its colour).
    verdict: Option<Verdict>,
    zero: bool,
    resting: bool,
}

/// The sample taken apart: which (and which before it), and its values as drawn, on
/// their way from the last one taken apart (a share of the way, `t`: from 0 to 1 on
/// the way, and on to `REST` while it is held there to be read).
#[derive(Clone, Debug, Default)]
struct Anatomy {
    k: Option<usize>,
    before: Option<usize>,
    /// The values (x) of the sample before and now, its mean, and its interval's ends.
    from: Vec<f32>,
    to: Vec<f32>,
    t: f32,
    /// The setup and n it was worked out for.
    key: Option<(Setup, u64)>,
}

impl Anatomy {
    /// How long the picture takes to go from one sample to the next, and how long it
    /// is then held before it goes on to another (seconds): at most three samples a
    /// second, however fast they come.
    const MOVE: f32 = 0.12;
    const HOLD: f32 = 0.2;
    const REST: f32 = 1.0 + Self::HOLD / Self::MOVE;

    /// The sample whose words and colours are shown: the one the picture is nearer.
    fn shown(&self) -> Option<usize> {
        if self.t >= 0.5 {
            self.k
        } else {
            self.before.or(self.k)
        }
    }

    fn drawn(&self, i: usize) -> f32 {
        let (a, b) = (self.from.get(i).copied().unwrap_or(0.0), self.to.get(i).copied().unwrap_or(0.0));
        a + (b - a) * self.t.clamp(0.0, 1.0)
    }

    /// How many values the sample has (each is two numbers: where across, where up).
    fn dots(&self) -> usize {
        self.to.len().saturating_sub(3) / 2
    }

    fn drawn_x(&self, i: usize) -> f32 {
        self.drawn(2 * i)
    }

    fn drawn_y(&self, i: usize) -> f32 {
        self.drawn(2 * i + 1)
    }

    /// The estimate, and the interval's two ends, as drawn.
    fn drawn_interval(&self) -> (f32, f32, f32) {
        let n = 2 * self.dots();
        (self.drawn(n), self.drawn(n + 1), self.drawn(n + 2))
    }
}

pub struct Intervals {
    setup: Setup,
    /// Each population's true value, kept while another is shown.
    truths: [f64; 3],
    /// The recipe last chosen for means, and for shares.
    mean_method: Method,
    share_method: Method,
    seed: u64,
    key: Key,
    /// The samples so far (see [`Summary`]); those below `stale` were kept for another
    /// key and are being worked out again, newest first.
    sums: Vec<Summary>,
    stale: usize,
    tally: Tally,
    /// The setup the tally was counted for.
    tallied: Option<Setup>,
    /// The run stops when it has this many samples.
    goal: usize,
    /// Samples a second.
    rate: f32,
    pace: Pace,
    phase: Phase,
    /// How far the column has slid towards the next sample (0 to 1).
    slide: f32,
    show_truth: bool,
    statement: bool,
    slots: Vec<Slot>,
    /// The coverage curve's heights as drawn, and its end (across and up).
    curve: Glide,
    end: Glide,
    /// The sample the pointer is over, and the one clicked to be kept.
    hovered: Option<usize>,
    pinned: Option<usize>,
    anatomy: Anatomy,
    /// Something drawn is still on its way somewhere.
    moving: bool,
    /// The critical value, and the recipe, n and level it is for.
    crit: std::cell::Cell<Option<(CritKey, f64)>>,
    buf: Vec<f64>,
}

impl Intervals {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Intervals {
        let mut s = Intervals {
            setup: Setup { pop: Population::Bell, truth: 10.0, n: 10, level: L95, method: Method::KnownZ },
            truths: TRUTH_START,
            mean_method: Method::KnownZ,
            share_method: Method::Wald,
            seed: SEED,
            key: Key { pop: Population::Bell, seed: SEED, n: 0, share: 0 },
            sums: Vec::new(),
            stale: 0,
            tally: Tally::default(),
            tallied: None,
            goal: GOALS[0],
            rate: 10.0,
            pace: Pace::default(),
            phase: Phase::default(),
            slide: 1.0,
            show_truth: true,
            statement: false,
            slots: vec![Slot::default(); SLOTS],
            curve: Glide::default(),
            end: Glide::default(),
            hovered: None,
            pinned: None,
            anatomy: Anatomy::default(),
            moving: false,
            crit: std::cell::Cell::new(None),
            buf: Vec::new(),
        };
        s.set_up(OPENING);
        s
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.truths[t.pop.index()] = t.truth;
        if t.pop == Population::Share {
            self.share_method = t.method;
        } else {
            self.mean_method = t.method;
        }
        self.setup = Setup { pop: t.pop, truth: t.truth, n: t.n, level: t.level, method: t.method };
        (self.rate, self.show_truth, self.statement, self.seed) = (t.rate, t.show_truth, t.statement, SEED);
        self.restart();
        self.goal = t.goal;
    }

    /// Back to no samples, with the seed as it is.
    fn restart(&mut self) {
        self.key = Key::of(&self.setup, self.seed);
        self.sums.clear();
        self.stale = 0;
        self.tally = Tally::default();
        self.tallied = Some(self.setup);
        self.goal = GOALS[0];
        self.pace.reset();
        self.phase.reset();
        self.slide = 1.0;
        self.pinned = None;
        self.hovered = None;
        for slot in &mut self.slots {
            slot.k = None;
        }
    }

    fn count(&self) -> usize {
        self.sums.len()
    }

    fn running(&self) -> bool {
        self.count() < self.goal
    }

    /// The next place to stop after this many samples (None: the run is at its end).
    fn next_goal(&self) -> Option<usize> {
        GOALS.iter().copied().find(|g| *g > self.count())
    }

    /// Choose a population: its own true value, and the recipe last used for its kind.
    fn choose_population(&mut self, pop: Population) {
        if pop == self.setup.pop {
            return;
        }
        self.setup.pop = pop;
        self.setup.truth = self.truths[pop.index()];
        self.setup.method = if pop == Population::Share { self.share_method } else { self.mean_method };
        self.statement = false;
        self.restart();
    }

    fn choose_method(&mut self, m: Method) {
        self.setup.method = m;
        if self.setup.pop == Population::Share {
            self.share_method = m;
        } else {
            self.mean_method = m;
        }
    }

    /// After a change of setting: keep the samples if they are still the ones the
    /// setting asks for, else work them out again (the newest at once, the rest a
    /// little in each frame), and count again.
    fn refresh(&mut self) {
        let lo_hi = TRUTH_RANGE[self.setup.pop.index()];
        self.setup.truth = self.setup.truth.max(lo_hi.0).min(lo_hi.1);
        self.truths[self.setup.pop.index()] = self.setup.truth;
        self.setup.n = self.setup.n.clamp(1, 100);
        self.setup.level = self.setup.level.min(LEVELS.len() - 1);
        let key = Key::of(&self.setup, self.seed);
        if key.pop != self.key.pop || key.seed != self.key.seed {
            self.restart();
            return;
        }
        if key != self.key {
            self.key = key;
            // the samples in the column first, so that it changes at once
            let count = self.count();
            for k in count.saturating_sub(SLOTS)..count {
                self.sums[k] = summary(key.pop, self.setup.truth, key.seed, k, key.n, &mut self.buf);
            }
            self.stale = count.saturating_sub(SLOTS);
            self.anatomy.key = None;
            self.tallied = None;
        }
        if self.stale == 0 && self.tallied != Some(self.setup) {
            self.recount();
        }
    }

    /// Work out again the samples kept for another key, newest first, until the
    /// budget is spent (all of them, without one).
    fn rework(&mut self, budget: Option<&Budget>) {
        let key = self.key;
        while self.stale > 0 {
            let k = self.stale - 1;
            self.sums[k] = summary(key.pop, self.setup.truth, key.seed, k, key.n, &mut self.buf);
            self.stale -= 1;
            if (self.stale & 63) == 0 && budget.is_some_and(|b| b.spent()) {
                break;
            }
        }
        if self.stale == 0 {
            self.recount();
        }
    }

    /// Count all the samples again, read by the setup as it is.
    fn recount(&mut self) {
        let mut tally = Tally { cum: Vec::with_capacity(self.sums.len()), ..Tally::default() };
        let c = self.critical();
        for s in &self.sums {
            tally.add(self.setup.read_at(*s, c), self.setup.truth);
        }
        self.tally = tally;
        self.tallied = Some(self.setup);
    }

    /// One more sample.
    fn step(&mut self) {
        let k = self.count();
        let s = summary(self.key.pop, self.setup.truth, self.key.seed, k, self.key.n, &mut self.buf);
        self.sums.push(s);
        if self.stale == 0 {
            self.tally.add(self.read(s), self.setup.truth);
        }
    }

    /// Go on to the next stop, from where the column rests: the first sample comes at
    /// once, so that the column slides on from where it stands.
    fn go_on(&mut self) {
        if let Some(g) = self.next_goal() {
            self.goal = g;
            self.statement = false;
            self.pace.reset();
            self.phase.reset();
            self.step();
            self.slide = 0.0;
        }
    }

    /// Stop where the run is.
    fn pause(&mut self) {
        self.goal = self.count();
    }

    /// The samples `dt` seconds of the scene's time ask for, and how far the column
    /// has slid towards the next.
    fn advance(&mut self, dt: f32, real_dt: f32) {
        if dt <= 0.0 {
            return;
        }
        let mut turn = scene_turn(&mut self.pace, self.rate, dt, real_dt);
        while self.running() && turn.next() {
            self.step();
        }
        let taken = self.pace.end(turn);
        self.slide = self.phase.after(self.rate, dt, taken);
    }

    /// The interval of sample `k` as the setup reads it.
    fn interval(&self, k: usize) -> Option<Iv> {
        self.sums.get(k).and_then(|s| self.read(*s))
    }

    /// The setup's interval of a sample.
    fn read(&self, s: Summary) -> Option<Iv> {
        self.setup.read_at(s, self.critical())
    }

    /// The setup's critical value, kept while the recipe, n and the level stay.
    fn critical(&self) -> f64 {
        let key = (self.setup.method, self.setup.n, self.setup.level);
        match self.crit.get() {
            Some((k, c)) if k == key => c,
            _ => {
                let c = self.setup.critical();
                self.crit.set(Some((key, c)));
                c
            }
        }
    }

    /// What the charts are centred on: the truth, or while it is hidden the first
    /// sample's estimate (an axis centred on the truth would give it away).
    fn centre(&self) -> f64 {
        match self.sums.first() {
            Some(s) if !self.show_truth => {
                let (mean, _) = self.setup.scaled(*s);
                let lo_hi = TRUTH_RANGE[self.setup.pop.index()];
                ((mean * 10.0).round() / 10.0).max(lo_hi.0.max(0.1)).min(lo_hi.1)
            }
            _ => self.setup.truth,
        }
    }

    /// The range of values the column shows.
    fn axis_range(&self) -> [f64; 2] {
        let t = self.centre();
        match self.setup.pop {
            Population::Bell => [t - 2.5 * BELL_SD, t + 2.5 * BELL_SD],
            Population::Tail => [-0.25 * t, 3.0 * t],
            Population::Share => {
                // (a stretch of 0.6 round the truth, within −0.1 to 1.1)
                let hi = (t + 0.3).max(0.5).min(1.1);
                [hi - 0.6, hi]
            }
        }
    }

    /// The range of values the chart of the sample taken apart shows.
    fn sample_range(&self) -> [f64; 2] {
        let t = self.centre();
        match self.setup.pop {
            Population::Bell => [t - 4.0 * BELL_SD, t + 4.0 * BELL_SD],
            Population::Tail => [-0.25 * t, 7.0 * t],
            Population::Share => [-0.1, 1.1],
        }
    }

    /// The coverage chart's range of shares: from 0.4 below the promised level, or a
    /// little above it, to 1.
    fn cover_span(&self) -> [f64; 2] {
        let hi = (self.setup.level() + 0.1).min(1.0);
        [hi - 0.4, hi]
    }

    /// The sample whose interval is taken apart: the one kept, else the one under the
    /// pointer, else the newest.
    fn focus(&self) -> Option<usize> {
        let count = self.count();
        self.pinned.filter(|k| *k < count).or(self.hovered.filter(|k| *k < count)).or(count.checked_sub(1))
    }
}

// ----- the layout -------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Layout {
    view: Vec2,
    narrow: bool,
    big: f32,
    text: f32,
    small: f32,
    button: f32,
    /// The title's place (none on a phone), and the rows of buttons (left, right,
    /// top; none in a window too short for them).
    title: Option<Vec2>,
    methods: Option<(f32, f32, f32)>,
    levels: Option<(f32, f32, f32)>,
    button_h: f32,
    /// The tops of the two lines of counts over the column.
    head: f32,
    head2: f32,
    /// The column of intervals, the sample taken apart (all of it, and the height of
    /// its chart), and the coverage chart (the rectangle its curve is drawn in): the
    /// last two left out, the coverage chart first, where the window has no room.
    column: Rect,
    anatomy: Option<Rect>,
    chart_h: f32,
    cover: Option<Rect>,
    /// The key under the column.
    key: Option<Vec2>,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        // (a phone held upright, or a window as narrow or too short to put things side
        // by side: one thing under another)
        let narrow = view.x < 760.0 && (view.x < 1.2 * view.y || view.y < 420.0);
        let (big, text, small, button) = if narrow || view.x < 900.0 { (13.0, 11.5, 10.5, 11.0) } else { (15.0, 13.0, 11.5, 12.5) };
        let side = if narrow { 10.0 } else { 22.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 14.0 };
        let button_h = if narrow { 24.0 } else { 26.0 };
        // (the lines of the arithmetic, and the room between them)
        let line = text + if narrow { 6.0 } else { 9.0 };
        let sane = |(a, b): Rect| (a, b.max(a));
        if narrow {
            let w = view.x - 2.0 * side;
            let buttons = view.y - top >= 400.0;
            let methods = buttons.then_some((side, view.x - side, top));
            let levels = buttons.then_some((side, view.x - side, top + button_h + 5.0));
            let head = if buttons { top + 2.0 * button_h + 13.0 } else { top };
            let head2 = head + button_h - 2.0;
            let col_top = head2 + small + text + 12.0;
            // from the bottom up: the coverage chart, the sample taken apart, the key;
            // whatever leaves the column too short is left out
            let room = view.y - col_top;
            let cover_h = (room * 0.13).clamp(30.0, 80.0);
            let chart_h = (room * 0.09).clamp(30.0, 60.0);
            let anat_h = text + 10.0 + chart_h + 24.0 + 2.0 * line;
            let key_h = small + 12.0;
            let axis_h = 36.0;
            let cover_all = cover_h + text + 16.0 + 34.0;
            let (show_cover, show_anat) = match room - axis_h - key_h {
                r if r - cover_all - anat_h >= 170.0 => (true, true),
                r if r - anat_h >= 170.0 => (false, true),
                _ => (false, false),
            };
            let cover_bottom = view.y - 34.0;
            let cover = show_cover.then(|| sane((Vec2::new(side + 36.0, cover_bottom - cover_h), Vec2::new(view.x - side - 18.0, cover_bottom))));
            let anat_bottom = if show_cover { cover_bottom - cover_h - text - 16.0 } else { view.y - 8.0 };
            let anatomy = show_anat.then(|| sane((Vec2::new(side, anat_bottom - anat_h), Vec2::new(view.x - side, anat_bottom))));
            let key_y = if show_anat { anat_bottom - anat_h - key_h } else { view.y - key_h };
            let key = (room > 120.0).then_some(Vec2::new(side, key_y));
            let col_bottom = (if key.is_some() { key_y - axis_h } else { view.y - axis_h }).max(col_top + 1.0);
            let column = sane((Vec2::new(side, col_top), Vec2::new(side + w.max(0.0), col_bottom)));
            Layout { view, narrow, big, text, small, button, title: None, methods, levels, button_h, head, head2, column, anatomy, chart_h, cover, key }
        } else {
            let gap = 46.0;
            let left_w = ((view.x - 2.0 * side - gap) * 0.58).max(10.0);
            let right_x = side + left_w + gap;
            let right_w = view.x - side - right_x;
            let title = Vec2::new(side, top);
            let methods = (side, side + left_w, top + big + 12.0);
            let levels = (side, side + left_w, methods.2 + button_h + 6.0);
            let head = levels.2 + button_h + 14.0;
            let head2 = head + button_h;
            let col_top = head2 + small + text + 14.0;
            let col_bottom = (view.y - 80.0).max(col_top + 1.0);
            let column = sane((Vec2::new(side, col_top), Vec2::new(side + left_w, col_bottom)));
            let key = Vec2::new(side, col_bottom + 44.0);
            let room = col_bottom - methods.2;
            let chart_h = (room * 0.3).clamp(60.0, 260.0);
            let anat_h = text + 20.0 + chart_h + 34.0 + 5.0 * line;
            let anatomy = (right_w >= 220.0 && anat_h <= room).then(|| sane((Vec2::new(right_x, methods.2), Vec2::new(right_x + right_w, methods.2 + anat_h))));
            let cover_top = methods.2 + anat_h + text + 44.0;
            let cover = (anatomy.is_some() && col_bottom - cover_top >= 70.0).then(|| sane((Vec2::new(right_x + 44.0, cover_top), Vec2::new(right_x + right_w - 18.0, col_bottom))));
            Layout { view, narrow, big, text, small, button, title: Some(title), methods: Some(methods), levels: Some(levels), button_h, head, head2, column, anatomy, chart_h, cover, key: Some(key) }
        }
    }

    /// The height of one row of the column (it holds `ROWS` at rest).
    fn row(&self) -> f32 {
        (self.column.1.y - self.column.0.y) / ROWS as f32
    }
}

/// Words that wrap, in screen space: `kit::wrapped`, which works in the plane's units,
/// at the plane's point under `at`. Returns the height taken.
fn wrapped_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> f32 {
    let Some(world) = f.cam.pick_ground(at * f.px) else { return 0.0 };
    let ppu = pixels_per_unit(f).max(1e-6);
    wrapped(f, world.truncate(), width / ppu, text, size, color);
    wrapped_height(f, width / ppu, text, size)
}

/// A row of colour swatches and their meanings, from `at` to the right (and on to
/// further lines below, as room runs out before `right`). Returns the height taken.
fn key_row(f: &mut Frame, at: Vec2, right: f32, entries: &[(Color, &str)], size: f32) -> f32 {
    let (mut x, mut y) = (at.x, at.y);
    for (color, meaning) in entries {
        let w = f.text_size(meaning, size).x + size + 22.0;
        if x > at.x && x + w > right {
            x = at.x;
            y += size + 7.0;
        }
        hud_rect(f, Vec2::new(x, y + 1.0), Vec2::new(x + size - 2.0, y + size - 1.0), *color);
        f.text_px(Vec2::new(x + size + 4.0, y + size * 0.5), *meaning, size, pal::DIM, Align2::LEFT_CENTER);
        x += w;
    }
    y + size - at.y
}

/// A screen point the motion of whatever is drawn there cannot be followed through:
/// off the picture. A place in a batch that has nothing to show this frame is drawn
/// here, so that every place keeps its place in the batch.
const AWAY: Vec2 = Vec2::new(-4000.0, -4000.0);

/// The kinds of line this scene draws in batches of their own: the column (the same
/// number of marks in every frame, each place's own), the coverage curve (which grows),
/// and the ticks along the interval taken apart (as many as it has room for).
const COLUMN: u8 = plot::OWN;
const CURVE: u8 = plot::OWN + 1;
const SAMPLE: u8 = plot::OWN + 2;

impl Scene for Intervals {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.running()
            || self.stale > 0
            || self.slide < 1.0
            || self.anatomy.t < 1.0
            || self.anatomy.k != self.focus()
            || self.slots.iter().any(|s| s.fresh)
            || self.moving
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("point at an interval to take it apart; click it to keep it there    buttons on the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A confidence interval is a range worked out from a sample by a recipe that, used on sample after sample, catches the true value in a stated share of them: 95 %, for a 95 % interval, when the recipe's assumptions hold. Here the truth is known (the green line), and sample after sample is drawn and made into an interval. The truth stays put; the intervals move, and each one either covers the truth or misses it. Every recipe here has one shape:");
        ui::equation(ui, r"\text{estimate} \pm \text{critical value} \times \text{standard error}");

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.name, t.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The population");
        let mut pop = self.setup.pop.index();
        let names: Vec<&str> = Population::ALL.iter().map(|p| p.name()).collect();
        if ui::choice(ui, &names, &mut pop) {
            self.choose_population(Population::ALL[pop]);
        }
        let (lo, hi) = TRUTH_RANGE[self.setup.pop.index()];
        let mut truth = self.setup.truth as f32;
        let label = if self.setup.pop == Population::Share { r"true share $p$" } else { r"true mean $\mu$" };
        if ui::slider(ui, label, &mut truth, lo as f32..=hi as f32) {
            self.setup.truth = truth as f64;
        }
        ui::note(
            ui,
            match self.setup.pop {
                Population::Bell => r"Normal, with standard deviation $\sigma = 2$.",
                Population::Tail => r"Log-normal: 69 % of its values are below its mean, and a few are far above it. Its standard deviation is 1.31 times its mean.",
                Population::Share => r"Each value is 1 (a success) with probability $p$, else 0; the share of ones estimates $p$.",
            },
        );

        ui::section(ui, "The intervals");
        let methods: &[Method] = if self.setup.pop == Population::Share { &Method::FOR_SHARES } else { &Method::FOR_MEANS };
        let mut m = methods.iter().position(|m| *m == self.setup.method).unwrap_or(0);
        let names: Vec<&str> = methods.iter().map(|m| m.name()).collect();
        if ui::choice(ui, &names, &mut m) {
            self.choose_method(methods[m]);
        }
        ui::note(ui, "The recipe. Changing it reads the same samples again: watch the same data cover or miss under each.");
        ui::choice(ui, &LEVEL_NAMES, &mut self.setup.level);
        ui::note(ui, "The level: the share of samples whose interval the recipe promises will cover the truth.");
        let mut n = self.setup.n;
        if ui::slider_int_log(ui, "values a sample ($n$)", &mut n, 1..=100) {
            self.setup.n = n;
        }
        ui::note(ui, "A sample keeps its first values when $n$ changes: more are added, or the last are taken away.");
        ui::slider_log(ui, "samples a second", &mut self.rate, 1.0..=300.0);
        ui.horizontal(|ui| {
            if self.running() {
                if ui.button("pause").clicked() {
                    self.pause();
                }
            } else if let Some(g) = self.next_goal() {
                if ui.button(format!("go on to {}", count(g))).clicked() {
                    self.go_on();
                }
            }
            if ui.button("start over").clicked() {
                self.restart();
                self.go_on();
            }
            if ui.button("new samples").clicked() {
                self.seed = self.seed.wrapping_add(1);
                self.restart();
                self.go_on();
            }
        });
        checkbox(ui, &mut self.show_truth, "show the truth");
        ui::note(ui, "In practice the truth is hidden: then nothing on the picture says which intervals cover it.");
        self.refresh();

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (TRUTH, r"green: the truth, $\mu$ or $p$, and the population the samples come from"),
                (ESTIMATE, "yellow: an interval that covers the truth; the tick on it is the sample's estimate"),
                (ALARM, "pink: an interval that misses, wholly below or wholly above the truth"),
                (NEUTRAL, "blue: an interval while the truth is hidden"),
                (DATA, "cyan: the values of the sample taken apart"),
                (pal::WHITE, "white band: the interval taken apart; dashed line: the promised level"),
                (pal::DIM, "grey: shares below 0 or above 1, which cannot be"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Confidence interval:", "a range made from a sample by a recipe that, over repeated samples, covers the true value a stated share of the time.");
        ui::term(ui, "Confidence level:", "that share, chosen beforehand: 95 % here unless changed.");
        ui::term(ui, "Coverage:", "the share of intervals that really do cover the truth. The level is what the recipe promises; the coverage is what it does. They agree when the recipe's assumptions hold, and may not when they do not.");
        ui::term(ui, "Standard error:", r"the standard deviation of the estimate from sample to sample: $\sigma/\sqrt{n}$ for a mean, estimated by $s/\sqrt{n}$.");
        ui::term(ui, "Critical value:", r"how many standard errors the interval reaches either side: $z = 1.96$ for 95 % from the normal distribution.");
        ui::term(ui, "Margin of error:", "critical value times standard error: half the interval's width.");
        ui::term(ui, "Student's $t$:", r"the distribution of $(\bar x - \mu)/(s/\sqrt{n})$ for values from a bell. Its tails are heavier than the normal's, because $s$ wanders too; with $n - 1$ degrees of freedom, $t = 2.78$ for 95 % at $n = 5$, falling to 1.96 as $n$ grows.");
        ui::term(ui, "Wilson's interval:", r"for a share, the values of $p$ a test would not reject: never outside 0 to 1, and never a single point.");
        ui::equation(ui, r"\frac{\hat p + z^2/2n}{1 + z^2/n} \pm \frac{z}{1 + z^2/n}\sqrt{\frac{\hat p(1 - \hat p)}{n} + \frac{z^2}{4n^2}}");

        ui::section(ui, "Numbers");
        let t = &self.tally;
        ui::readout(ui, "samples so far", count(t.count));
        if self.setup.no_interval() {
            ui::readout(ui, "intervals", "none: one value has no spread");
        } else if self.stale > 0 {
            ui::readout(ui, "covered", "counting");
        } else if t.count > 0 {
            ui::readout(ui, "covered", format!("{} ({} %)", count(t.covered), num(100.0 * t.share(), 1)));
            ui::readout(ui, "missed below, above", format!("{}, {}", count(t.below), count(t.above)));
            if self.setup.pop == Population::Share {
                ui::readout(ui, "single points", count(t.zero));
            }
        }
        let c = self.critical();
        ui::readout(ui, "critical value", if c.is_nan() { "none".to_string() } else if c.is_infinite() { r"$\infty$".to_string() } else { num(c, 3) });
        ui::see_also(
            ui,
            &[
                ("estimators", "the standard error, and the spread of an estimate"),
                ("likelihood", "another way to make an interval"),
                ("bootstrap", "and another, from one sample alone"),
                ("bayes", "a credible interval, which does say “with probability”"),
                ("testing", "an interval is the values a test would not reject"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 160.0;
        // the picture is charts on the screen: nothing in it is moved by dragging
        f.handles.capture();
        if !tiny {
            self.draw_buttons(f, &lay);
        }
        self.refresh();
        if self.stale > 0 {
            let budget = Budget::new(0.003);
            self.rework(Some(&budget));
        }
        self.advance(f.dt, f.real_dt);
        if !tiny {
            self.point(f, &lay);
        }
        self.follow(f);
        if tiny {
            return;
        }
        self.draw_head(f, &lay);
        self.draw_column(f, &lay);
        if let Some(rect) = lay.anatomy {
            self.draw_anatomy(f, &lay, rect);
        }
        if let Some(rect) = lay.cover {
            self.draw_cover(f, &lay, rect);
        }
    }
}

impl Intervals {
    /// The title, the counts over the column, and the run's buttons.
    fn draw_head(&mut self, f: &mut Frame, lay: &Layout) {
        let s = self.setup;
        if let Some(at) = lay.title {
            let what = if s.pop == Population::Share { r"the share $p$" } else { r"the mean $\mu$" };
            let title = format!("{} intervals for {what}, from samples of $n = {}$:   {}", LEVEL_NAMES[s.level], s.n, s.method.formula());
            f.text_px(at, title, lay.big, pal::TEXT, Align2::LEFT_TOP);
        }
        let (min, max) = lay.column;
        let (y, h, size) = (lay.head, lay.button_h - 4.0, lay.button);
        // the run's buttons, at the right of the first line
        let run = if self.running() {
            "pause".to_string()
        } else if let Some(g) = self.next_goal() {
            format!("go on to {}", count(g))
        } else {
            "start over".to_string()
        };
        let truth = if self.show_truth { "hide the truth" } else { "show the truth" };
        // (each as wide as the widest of its words, so that it keeps its place as they change)
        let widest = |words: &[&str]| words.iter().map(|w| f.text_size(w, size).x).fold(0.0f32, f32::max) + 18.0;
        let widths = [widest(&["pause", "go on to 10 000", "start over"]), widest(&["hide the truth", "show the truth"])];
        let right = max.x;
        let truth_rect = (Vec2::new(right - widths[1], y - 3.0), Vec2::new(right, y - 3.0 + h));
        let run_rect = (Vec2::new(truth_rect.0.x - 6.0 - widths[0], y - 3.0), Vec2::new(truth_rect.0.x - 6.0, y - 3.0 + h));
        // (in a window too small for them, the buttons are left to the panel)
        let buttons = max.x - min.x >= 330.0;
        if buttons && plot::button(f, run_rect, &run, size, false) {
            if self.running() {
                self.pause();
            } else if self.next_goal().is_some() {
                self.go_on();
            } else {
                self.restart();
                self.go_on();
            }
        }
        if buttons && plot::button(f, truth_rect, truth, size, false) {
            self.show_truth = !self.show_truth;
        }
        let t = &self.tally;
        let room = if buttons { run_rect.0.x } else { max.x } - min.x - 10.0;
        let (first, color) = if s.no_interval() {
            (vec!["no intervals: one value has no spread".to_string(), "no intervals".to_string()], ALARM)
        } else if !self.show_truth {
            (vec!["how many cover: hidden with the truth".to_string(), "covered: hidden".to_string()], pal::DIM)
        } else if self.stale > 0 {
            (vec!["counting again…".to_string()], pal::DIM)
        } else if t.count == 0 {
            (vec!["the first sample is on its way".to_string(), "".to_string()], pal::DIM)
        } else {
            let share = num(100.0 * t.share(), 1);
            (vec![format!("covered {} of {} ({share} %)", count(t.covered), count(t.count)), format!("{} of {} cover", count(t.covered), count(t.count))], ESTIMATE)
        };
        let first = first.iter().find(|w| f.text_size(w, lay.text).x <= room).or(first.last()).cloned().unwrap_or_default();
        f.text_px(Vec2::new(min.x, y + (h - 6.0) * 0.5), first, lay.text, color, Align2::LEFT_CENTER);
        // the second line: where the misses fell
        let y2 = lay.head2;
        let mut second = Vec::new();
        if self.show_truth && self.stale == 0 && !s.no_interval() && t.count > 0 {
            second.push((format!("missed: {} below, {} above", count(t.below), count(t.above)), ALARM));
            if t.zero > 0 {
                let points = if t.zero == 1 { "1 is a single point".to_string() } else { format!("{} are single points", count(t.zero)) };
                second.push((points, ALARM));
            }
        }
        second.push(("newest at the top".to_string(), pal::DIM));
        let mut x = min.x;
        for (text, color) in second {
            let w = f.text_size(&text, lay.small).x;
            if x > min.x && x + w > max.x {
                break;
            }
            f.text_px(Vec2::new(x, y2), text, lay.small, color, Align2::LEFT_TOP);
            x += w + 18.0;
        }
    }

    /// The column of intervals, newest at the top, and the truth through it.
    fn draw_column(&mut self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.column;
        let plot = Plot::new(lay.column, self.axis_range(), [0.0, 1.0]);
        plot.backing(f, 8.0, 34.0, 8.0, 8.0);
        let name = if self.setup.pop == Population::Share { "the share" } else { "the mean" };
        plot.x_axis(f, name, lay.small);
        let row = lay.row();
        let width = (row * 0.55).clamp(1.2, 4.0);
        let count = self.count();
        let truth = self.setup.truth;
        let y_of = |age: usize, slide: f32| min.y + (age as f32 + slide - 0.5) * row;
        // the interval taken apart, on a band
        if let Some(k) = self.anatomy.shown().filter(|k| *k < count && count - 1 - k < ROWS) {
            let y = y_of(count - 1 - k, self.slide);
            plot::fill(f, (Vec2::new(min.x - 6.0, y - row * 0.5), Vec2::new(max.x + 6.0, y + row * 0.5)), pal::WHITE.a(if self.pinned == Some(k) { 0.16 } else { 0.07 }));
        }
        let none = self.setup.no_interval();
        // what has happened, where the column cannot show it by itself: in a box the
        // rows behind it leave clear
        let note = if none {
            Some(r"One value has no spread: $s$ divides by $n - 1 = 0$. With the spread to be estimated there is no interval; with $\sigma$ known there is (“$\sigma$ known: $z$”).")
        } else if self.setup.level() >= 1.0 && count > 0 {
            Some(match self.setup.method {
                Method::Wilson => "At 100 % the critical value is infinite, and every interval is all of 0 to 1: it always covers, and says nothing.",
                Method::Wald => "At 100 % the critical value is infinite, and every interval is the whole line: it says nothing. Only a sample with no spread still gives a single point, which can miss.",
                _ => "At 100 % the critical value is infinite, and every interval is the whole line: it always covers, and says nothing.",
            })
        } else {
            None
        };
        let note_w = (max.x - min.x - 40.0).max(40.0);
        let note_at = Vec2::new(min.x + 20.0, min.y + (max.y - min.y) * 0.4);
        let note_box = note.map(|n| (note_at.y - 10.0, note_at.y + kit_height(f, note_w, n, lay.text) + 6.0));
        // what a single interval does not say, over the empty column (the truth's line
        // goes round it)
        let boxed = if self.statement && count <= 1 { self.draw_statement(f, lay) } else { None };
        if self.show_truth {
            let x = plot.px_x(truth);
            if x >= min.x && x <= max.x {
                let (a, b) = boxed.or(note_box).unwrap_or(((min.y + max.y) * 0.5, (min.y + max.y) * 0.5));
                plot::line_of(f, plot::MARKS, Vec2::new(x, min.y), Vec2::new(x, a), 2.0, TRUTH);
                plot::line_of(f, plot::MARKS, Vec2::new(x, b), Vec2::new(x, max.y), 2.0, TRUTH);
            }
            if x >= min.x - 1.0 && x <= max.x + 1.0 {
                let label = if self.setup.pop == Population::Share { format!("$p = {}$", num(truth, 2)) } else { format!(r"$\mu = {}$", num(truth, 1)) };
                f.text_px(Vec2::new(x, min.y - row * 0.5 - 5.0), label, lay.text, TRUTH, Align2::CENTER_BOTTOM);
            }
        }
        if self.setup.pop == Population::Share {
            // no share is below 0 or above 1
            for (a, b, words, align) in [(plot.x[0], 0.0, "below 0: no share", Align2::RIGHT_BOTTOM), (1.0, plot.x[1], "above 1: no share", Align2::LEFT_BOTTOM)] {
                let (xa, xb) = (plot.px_x(a).max(min.x), plot.px_x(b).min(max.x));
                if xb - xa > 1.0 {
                    plot::fill(f, (Vec2::new(xa, min.y), Vec2::new(xb, max.y)), pal::DIM.a(0.08));
                    if f.text_size(words, lay.small).x + 8.0 < xb - xa {
                        let x = if align == Align2::RIGHT_BOTTOM { xb - 4.0 } else { xa + 4.0 };
                        f.text_px(Vec2::new(x, max.y - 4.0), words, lay.small, pal::DIM, align);
                    }
                }
            }
        }
        let (lo_edge, hi_edge) = (plot.x[0] as f32, plot.x[1] as f32);
        // Each place in the column draws the same marks in every frame, those it has no
        // use for off the picture: its interval, the estimate's tick, an arrowhead at
        // either end (for an interval that runs on beyond the column) and a ring (for
        // one of no width). A mark that came and went would leave the others of the
        // batch out of step from one frame to the next.
        let away = |f: &mut Frame, n: usize| {
            for _ in 0..n {
                plot::line_of(f, COLUMN, AWAY, AWAY + Vec2::X, 1.0, Color::CLEAR);
            }
        };
        let ring_away = |f: &mut Frame| plot::path_of(f, COLUMN, &[AWAY, AWAY + Vec2::X, AWAY + Vec2::ONE, AWAY + Vec2::Y, AWAY], 1.0, Color::CLEAR);
        for slot in &self.slots {
            let shown = slot.k.filter(|_| !slot.fresh);
            let Some(k) = shown else {
                // (nothing here this frame: the place is kept, off the picture)
                away(f, 6);
                ring_away(f);
                continue;
            };
            let age = count - 1 - k;
            let r = age as f32 + self.slide;
            let alpha = r.min(1.0).min(SLOTS as f32 - r).clamp(0.0, 1.0);
            let y = y_of(age, self.slide);
            if note_box.is_some_and(|(a, b)| y > a - row && y < b + row) {
                away(f, 6);
                ring_away(f);
                continue;
            }
            let (lo, hi, est) = (slot.glide.get(0), slot.glide.get(1), slot.glide.get(2));
            let t = truth as f32;
            let verdict = if slot.resting { slot.verdict } else { slot.verdict.map(|_| if hi < t { Verdict::Below } else if lo > t { Verdict::Above } else { Verdict::Covers }) };
            let color = match (self.show_truth, verdict) {
                (_, None) => pal::DIM,
                (false, _) => NEUTRAL,
                (true, Some(Verdict::Covers)) => ESTIMATE,
                (true, Some(_)) => ALARM,
            }
            .a(alpha);
            let x = |v: f32| plot.px_x(v as f64).max(min.x).min(max.x);
            plot::line_of(f, COLUMN, Vec2::new(x(lo), y), Vec2::new(x(hi), y), width, color);
            let tick = (row * 0.48).max(width * 0.5 + 1.5);
            plot::line_of(f, COLUMN, Vec2::new(x(est), y - tick), Vec2::new(x(est), y + tick), 1.5, if none { pal::DIM.a(alpha) } else { color });
            // ends beyond the column: an arrowhead, for "and on"; no width: a ring
            let head = (row * 0.5).max(2.5);
            for (beyond, edge, back) in [(lo < lo_edge, min.x, head), (hi > hi_edge, max.x, -head)] {
                if beyond && !none {
                    plot::line_of(f, COLUMN, Vec2::new(edge + back, y - head), Vec2::new(edge, y), 1.2, color);
                    plot::line_of(f, COLUMN, Vec2::new(edge + back, y + head), Vec2::new(edge, y), 1.2, color);
                } else {
                    away(f, 2);
                }
            }
            if slot.zero && !none {
                let c = Vec2::new(x(est), y);
                let d = (row * 0.6).max(3.0);
                let ring = [c - Vec2::X * d, c - Vec2::Y * d, c + Vec2::X * d, c + Vec2::Y * d, c - Vec2::X * d];
                plot::path_of(f, COLUMN, &ring, 1.2, color);
            } else {
                ring_away(f);
            }
        }

        if let (Some(note), Some((top, bottom))) = (note, note_box) {
            hud_rect(f, Vec2::new(note_at.x - 10.0, top), Vec2::new(note_at.x + note_w + 10.0, bottom), Color::hex(0x10141F).a(0.94));
            wrapped_px(f, note_at, note_w, note, lay.text, pal::TEXT);
        }
        // the key, under the axis
        let mut key: Vec<(Color, &str)> = if self.show_truth {
            vec![(ESTIMATE, "covers the truth"), (ALARM, "misses it"), (TRUTH, "the truth")]
        } else {
            vec![(NEUTRAL, "an interval: whether it covers, nothing here says")]
        };
        if self.setup.pop == Population::Share && plot.x[0] < 0.0 {
            key.push((pal::DIM.a(0.5), "no share is here"));
        }
        if lay.anatomy.is_some() && !lay.narrow {
            key.push((pal::WHITE.a(0.4), "taken apart, right"));
        }
        if let Some(at) = lay.key {
            key_row(f, at, max.x, &key, lay.small);
        }
    }

    /// What a single interval does not say, over the empty column.
    /// Returns the box's top and bottom.
    fn draw_statement(&mut self, f: &mut Frame, lay: &Layout) -> Option<(f32, f32)> {
        let (min, max) = lay.column;
        let iv = self.interval(0)?;
        let level = LEVEL_NAMES[self.setup.level];
        let d = if self.setup.pop == Population::Share { 3 } else { 2 };
        let range = format!("$[{}, {}]$", num(iv.lo, d), num(iv.hi, d));
        let mu = if self.setup.pop == Population::Share { "$p$" } else { r"$\mu$" };
        let w = (max.x - min.x - 48.0).max(60.0);
        let at = Vec2::new(min.x + 24.0, min.y + lay.row() * 6.0 + 14.0);
        let text = [
            (format!("Is {mu} in {range}?"), pal::TEXT, lay.big),
            (format!("Not “with probability {level}”. {mu} is a fixed number: it is in this interval or it is not, and once the sample is drawn no chance is left in it."), pal::DIM, lay.text),
            (format!("The {level} belongs to the recipe: of the intervals it makes from new samples, {level} cover {mu}. Which ones, nobody can tell without knowing {mu}."), pal::DIM, lay.text),
        ];
        let mut y = at.y;
        let mut heights = Vec::new();
        for (t, _, size) in &text {
            heights.push(kit_height(f, w, t, *size));
        }
        let verdict = if self.show_truth {
            Some(match iv.verdict(self.setup.truth) {
                Verdict::Covers => (format!("Here it is in: {mu} $= {}$.", num(self.setup.truth, d - 1)), ESTIMATE),
                Verdict::Below => (format!("Here it is not: {mu} is above it."), ALARM),
                Verdict::Above => (format!("Here it is not: {mu} is below it."), ALARM),
            })
        } else {
            None
        };
        let total = heights.iter().sum::<f32>() + 10.0 * text.len() as f32 + if verdict.is_some() { lay.text + 10.0 } else { 0.0 };
        hud_rect(f, at - Vec2::splat(12.0), Vec2::new(at.x + w + 12.0, at.y + total + 4.0), Color::hex(0x10141F).a(0.94));
        for ((t, color, size), h) in text.iter().zip(&heights) {
            wrapped_px(f, Vec2::new(at.x, y), w, t, *size, *color);
            y += h + 10.0;
        }
        if let Some((t, color)) = verdict {
            f.text_px(Vec2::new(at.x, y), t, lay.text, color, Align2::LEFT_TOP);
        }
        Some((at.y - 12.0, at.y + total + 4.0))
    }

    /// The sample taken apart: its values, the population they come from, their mean,
    /// its interval measured in standard errors, and the arithmetic written out.
    fn draw_anatomy(&mut self, f: &mut Frame, lay: &Layout, (min, max): Rect) {
        let Some(k) = self.anatomy.shown().filter(|k| *k < self.count()) else {
            f.text_px(min, "the first sample is on its way", lay.text, pal::DIM, Align2::LEFT_TOP);
            return;
        };
        let s = self.setup;
        let title = if self.pinned == Some(k) {
            format!("sample {}, kept: click it again to let it go", count(k + 1))
        } else if self.hovered == Some(k) {
            format!("sample {}, under the pointer", count(k + 1))
        } else if k + 1 == self.count() {
            format!("sample {}, the newest, taken apart", count(k + 1))
        } else {
            format!("sample {}, taken apart", count(k + 1))
        };
        f.text_px(min, title, lay.text, pal::TEXT, Align2::LEFT_TOP);
        let chart_top = min.y + lay.text + if lay.narrow { 10.0 } else { 20.0 };
        let chart_h = lay.chart_h;
        let rect = (Vec2::new(min.x + 8.0, chart_top), Vec2::new(max.x - 8.0, chart_top + chart_h));
        let range = self.sample_range();
        let plot = Plot::new(rect, range, [0.0, 1.0]);
        plot.backing(f, 8.0, 30.0, 6.0, 8.0);
        plot.x_axis(f, "", lay.small);
        let a = &self.anatomy;
        let n = a.dots();
        // the population (not while the truth is hidden: it would show it)
        match s.pop {
            _ if !self.show_truth => {}
            Population::Share => {
                for (x, h) in [(0.0, 1.0 - s.truth), (1.0, s.truth)] {
                    plot.bar(f, x - 0.035, x + 0.035, 0.25, 0.25 + 0.7 * h, 0.0, TRUTH.a(0.12));
                }
            }
            _ => {
                let dist = if s.pop == Population::Bell { Dist::Normal { mean: s.truth, sd: BELL_SD } } else { Dist::LogNormal { mu: s.truth.ln() - 0.5, sigma: 1.0 } };
                let peak = if s.pop == Population::Bell { dist.pdf(s.truth) } else { dist.pdf((s.truth.ln() - 1.5).exp()) };
                let g = |x: f64| 0.25 + 0.7 * dist.pdf(x) / peak.max(1e-12);
                plot.area(f, range[0], range[1], 0.25, 90, TRUTH.a(0.035), g);
                plot.curve(f, plot::CURVES, 90, 1.2, TRUTH.a(0.55), g);
            }
        }
        if self.show_truth {
            plot.vline(f, plot::MARKS, s.truth, 1.6, TRUTH, false);
        }
        // the values: on a band of their own, or for a share in two piles
        let (lo_x, hi_x) = (rect.0.x, rect.1.x);
        let dot = if lay.narrow { 4.5 } else { 6.0 };
        for i in 0..n {
            let (x, y) = (a.drawn_x(i), a.drawn_y(i));
            let at = plot.at(x as f64, y as f64);
            plot::disc(f, Vec2::new(at.x.max(lo_x).min(hi_x), at.y), dot, DATA);
        }
        let (est, lo, hi) = a.drawn_interval();
        let iv = self.interval(k);
        let color = match (self.show_truth, iv.map(|iv| iv.verdict(s.truth))) {
            (_, None) => pal::DIM,
            (false, _) => NEUTRAL,
            (true, Some(Verdict::Covers)) => ESTIMATE,
            (true, Some(_)) => ALARM,
        };
        let bar_y = plot.px_y(if lay.narrow { 0.13 } else { 0.12 });
        let x = |v: f32| plot.px_x(v as f64).max(lo_x).min(hi_x);
        if iv.is_some() {
            plot::line_of(f, plot::MARKS, Vec2::new(x(lo), bar_y), Vec2::new(x(hi), bar_y), 3.0, color);
            for end in [lo, hi] {
                plot::line_of(f, plot::MARKS, Vec2::new(x(end), bar_y - 6.0), Vec2::new(x(end), bar_y + 6.0), 1.5, color);
            }
            // the margin measured in standard errors: a tick at each whole one
            let c = self.critical();
            let ruled = s.method != Method::Wilson && c.is_finite() && c > 0.0;
            let se = (hi - est) / c as f32;
            let whole = if ruled && se * plot.width() / (range[1] - range[0]) as f32 > 3.0 { c.floor().min(4.0) as usize } else { 0 };
            for j in 1..=4usize {
                for side in [-1.0f32, 1.0] {
                    let (a, b) = if j <= whole {
                        let v = x(est + side * j as f32 * se);
                        (Vec2::new(v, bar_y - 4.5), Vec2::new(v, bar_y + 4.5))
                    } else {
                        (AWAY, AWAY + Vec2::Y)
                    };
                    plot::line_of(f, SAMPLE, a, b, 1.3, color);
                }
            }
            if ruled && hi > lo {
                let label = format!(r"${} \times \text{{SE}}$", num(c, 2));
                let w = f.text_size(&label, lay.small).x;
                if x(hi) - x(est) > w + 6.0 && !lay.narrow {
                    f.text_px(Vec2::new((x(est) + x(hi)) * 0.5, bar_y + 5.0), label, lay.small, color, Align2::CENTER_TOP);
                } else {
                    f.text_px(Vec2::new(x(hi) + 6.0, bar_y), label, lay.small, color, Align2::LEFT_CENTER);
                }
            }
        }
        // the estimate, through the values
        plot::line_of(f, plot::MARKS, Vec2::new(x(est), bar_y - 8.0), Vec2::new(x(est), plot.px_y(0.62)), 1.5, ESTIMATE.a(0.85));
        let name = if s.pop == Population::Share { r"$\hat p$" } else { r"$\bar x$" };
        f.text_px(Vec2::new(x(est), plot.px_y(0.62) - 2.0), name, lay.small, ESTIMATE, Align2::CENTER_BOTTOM);
        // the arithmetic
        let lines = self.arithmetic(k, lay.narrow);
        let mut y = rect.1.y + if lay.narrow { 24.0 } else { 34.0 };
        for (text, color) in lines {
            if y + lay.text > max.y + 4.0 {
                break;
            }
            f.text_px(Vec2::new(min.x, y), text, lay.text, color, Align2::LEFT_TOP);
            y += lay.text + if lay.narrow { 6.0 } else { 9.0 };
        }
    }

    /// The arithmetic of sample `k`'s interval, a line at a time.
    fn arithmetic(&self, k: usize, narrow: bool) -> Vec<(String, Color)> {
        let s = self.setup;
        let Some(sum) = self.sums.get(k).copied() else { return Vec::new() };
        let (mean, sd) = s.scaled(sum);
        let n = s.n;
        let c = self.critical();
        let level = LEVEL_NAMES[s.level];
        let crit = if c.is_infinite() { r"\infty".to_string() } else { num(c, 3) };
        let mut out = Vec::new();
        let d = if s.pop == Population::Share { 3 } else { 2 };
        match s.method {
            Method::KnownZ | Method::StudentT | Method::PlugInZ => {
                let known = s.method == Method::KnownZ;
                if known {
                    out.push((format!(r"$n = {n}$ values, mean $\bar x = {}$", num(mean, d)), pal::TEXT));
                } else if n < 2 {
                    out.push((format!(r"$n = 1$ value, $\bar x = {}$; $s$ needs two", num(mean, d)), pal::TEXT));
                    return out;
                } else {
                    out.push((format!(r"$n = {n}$ values, mean $\bar x = {}$, $s = {}$", num(mean, d), num(sd, d)), pal::TEXT));
                }
                let (spread, name) = if known { (s.sigma(), r"\sigma") } else { (sd, "s") };
                let se = spread / (n as f64).sqrt();
                out.push((format!(r"standard error ${name}/\sqrt{{n}} = {}/\sqrt{{{n}}} = {}$", num(spread, if known && s.pop == Population::Bell { 0 } else { d }), num(se, 3)), pal::TEXT));
                let which = match s.method {
                    Method::StudentT => format!(r"$t = {crit}$, with {} degrees of freedom", n - 1),
                    Method::PlugInZ => format!(r"$z = {crit}$, as if $\sigma$ were known"),
                    _ => format!("$z = {crit}$"),
                };
                out.push((format!("critical value for {level}: {which}"), pal::TEXT));
                let m = margin(c, se);
                let ends = if m.is_infinite() { r"(-\infty, \infty)".to_string() } else { format!("[{}, {}]", num(mean - m, d), num(mean + m, d)) };
                out.push((format!(r"${} \pm {crit} \times {} = {ends}$", num(mean, d), num(se, 3)), pal::TEXT));
            }
            Method::Wald | Method::Wilson => {
                let hits = (mean * n as f64).round() as usize;
                let tries = if hits == 1 { "1 success".to_string() } else { format!("{hits} successes") };
                out.push((format!(r"$n = {n}$ tries, {tries}: $\hat p = {}$", num(mean, d)), pal::TEXT));
                if s.method == Method::Wald {
                    let se = (mean * (1.0 - mean) / n as f64).max(0.0).sqrt();
                    out.push((format!(r"standard error $\sqrt{{\hat p(1 - \hat p)/n}} = {}$", num(se, 3)), pal::TEXT));
                    out.push((format!("critical value for {level}: $z = {crit}$"), pal::TEXT));
                    let m = margin(c, se);
                    let ends = if m.is_infinite() { r"(-\infty, \infty)".to_string() } else { format!("[{}, {}]", num(mean - m, d), num(mean + m, d)) };
                    out.push((format!(r"${} \pm {crit} \times {} = {ends}$", num(mean, d), num(se, 3)), pal::TEXT));
                    if se == 0.0 {
                        out.push((r"no spread in the sample: $\text{SE} = 0$, a single point".to_string(), ALARM));
                    }
                } else {
                    let (centre, half) = wilson(mean, n as f64, c);
                    out.push((format!("critical value for {level}: $z = {crit}$"), pal::TEXT));
                    out.push((format!(r"centre $(\hat p + z^2/2n)/(1 + z^2/n) = {}$", num(centre, d)), pal::TEXT));
                    let lo = (centre - half).max(0.0);
                    let hi = (centre + half).min(1.0);
                    out.push((format!(r"${} \pm {} = [{}, {}]$", num(centre, d), num(half, d), num(lo, d), num(hi, d)), pal::TEXT));
                }
            }
        }
        if let Some(iv) = self.read(sum) {
            let t = num(s.truth, if s.pop == Population::Share { 2 } else { 1 });
            let mu = if s.pop == Population::Share { "p" } else { r"\mu" };
            let line = if !self.show_truth {
                ("covers the truth, or not: hidden".to_string(), pal::DIM)
            } else {
                match iv.verdict(s.truth) {
                    Verdict::Covers => (format!("it covers ${mu} = {t}$"), ESTIMATE),
                    Verdict::Below => (format!("it misses: all of it is below ${mu} = {t}$"), ALARM),
                    Verdict::Above => (format!("it misses: all of it is above ${mu} = {t}$"), ALARM),
                }
            };
            out.push(line);
        }
        if narrow {
            // (a phone has room for the interval and the verdict)
            let len = out.len();
            out = out.into_iter().skip(len.saturating_sub(2)).collect();
        }
        out
    }

    /// The share covered so far against the number of samples, and the promise.
    fn draw_cover(&mut self, f: &mut Frame, lay: &Layout, rect: Rect) {
        let [lo, hi] = self.cover_span();
        let plot = Plot::new(rect, [0.0, 4.0], [lo, hi]);
        if plot.tiny() {
            return;
        }
        plot.backing(f, 40.0, 34.0, 30.0, 26.0);
        let (min, max) = rect;
        let t = &self.tally;
        let level = self.setup.level();
        let title = if self.setup.no_interval() || !self.show_truth || t.count == 0 || self.stale > 0 {
            "the share that covered, so far".to_string()
        } else {
            format!("the share that covered, so far: {} %", num(100.0 * t.share(), 1))
        };
        f.text_px(Vec2::new(min.x - 30.0, min.y - 10.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        // the axes: samples on a log scale, shares in percent
        plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        for (e, name) in ["1", "10", "100", "1000", "10 000"].iter().enumerate() {
            let x = plot.px_x(e as f64);
            plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 6.0), *name, lay.small, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 8.0 + lay.small), "samples so far (log scale)", lay.small, pal::DIM, Align2::CENTER_TOP);
        let step = if hi - lo > 0.25 { 0.1 } else { 0.05 };
        let mut v = (lo / step).ceil() * step;
        while v <= hi + 1e-9 {
            let y = plot.px_y(v);
            plot::line_of(f, plot::AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8));
            plot::line_of(f, plot::AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.12));
            f.text_px(Vec2::new(min.x - 6.0, y), format!("{} %", num(100.0 * v, 0)), lay.small, pal::DIM, Align2::RIGHT_CENTER);
            v += step;
        }
        plot.hline(f, plot::MARKS, level, 1.4, pal::WHITE.a(0.7), true);
        let promise = format!("promised: {}", LEVEL_NAMES[self.setup.level]);
        let py = plot.px_y(level.min(hi));
        f.text_px(Vec2::new(max.x - 4.0, py - 4.0), promise, lay.small, pal::WHITE.a(0.8), Align2::RIGHT_BOTTOM);
        if self.setup.no_interval() || !self.show_truth {
            let why = if self.setup.no_interval() { "no intervals to count" } else { "the truth is hidden: nothing to count against" };
            f.text_px((min + max) * 0.5 + Vec2::new(0.0, 12.0), why, lay.small, pal::DIM, Align2::CENTER_CENTER);
            return;
        }
        let ks = chart_counts(t.count);
        let end = (self.end.get(0) as f64, self.end.get(1) as f64);
        let mut pts: Vec<(f64, f64)> = ks.iter().enumerate().map(|(i, k)| ((*k as f64).log10(), self.curve.get(i) as f64)).collect();
        if let Some(last) = pts.last_mut() {
            *last = end;
        }
        plot.path(f, CURVE, pts.iter().copied(), 1.8, ESTIMATE);
        if !ks.is_empty() {
            plot::disc(f, plot.at(end.0, end.1.max(lo).min(hi)), 7.0, ESTIMATE);
        }
    }
}

/// How tall `wrapped_px` draws a text.
fn kit_height(f: &Frame, width: f32, text: &str, size: f32) -> f32 {
    let ppu = pixels_per_unit(f).max(1e-6);
    wrapped_height(f, width / ppu, text, size)
}

/// The colour of an interval when the truth is hidden: neither covering nor missing.
const NEUTRAL: Color = pal::LINK;

/// Places along the coverage chart's axis of samples, evenly on a log scale.
fn chart_counts(count: usize) -> Vec<usize> {
    let mut ks: Vec<usize> = Vec::with_capacity(170);
    for j in 0..=160 {
        let k = 10f64.powf(j as f64 / 40.0).round() as usize;
        if k >= count {
            break;
        }
        if ks.last() != Some(&k) {
            ks.push(k);
        }
    }
    if count > 0 {
        ks.push(count);
    }
    ks
}

impl Intervals {
    /// The buttons on the picture: the recipe and the level.
    fn draw_buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let methods: &[Method] = if self.setup.pop == Population::Share { &Method::FOR_SHARES } else { &Method::FOR_MEANS };
        let names: Vec<&str> = methods.iter().map(|m| m.name()).collect();
        let Some((x0, x1, y)) = lay.methods else { return };
        for (i, r) in fitted_row(f, &names, x0, x1, y, lay.button_h, lay.button).iter().enumerate() {
            if plot::button(f, *r, names[i], lay.button, methods[i] == self.setup.method) {
                self.choose_method(methods[i]);
            }
        }
        let Some((x0, x1, y)) = lay.levels else { return };
        for (i, r) in fitted_row(f, &LEVEL_NAMES, x0, x1, y, lay.button_h, lay.button).iter().enumerate() {
            if plot::button(f, *r, LEVEL_NAMES[i], lay.button, i == self.setup.level) {
                self.setup.level = i;
            }
        }
    }

    /// Which sample the pointer is over in the column, and a click that keeps it.
    fn point(&mut self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.column;
        let row = lay.row();
        self.hovered = None;
        let Some(p) = plot::pointer(f).filter(|p| plot::within(*p, lay.column, 0.0)) else { return };
        f.handles.claim_clicks();
        let count = self.count();
        let age = ((p.y - min.y) / row.max(1e-6) + 0.5 - self.slide).round();
        let k = (age >= 0.0 && (age as usize) < count.min(ROWS)).then(|| count - 1 - age as usize);
        self.hovered = k;
        if f.ptr.clicked && p.x <= max.x {
            self.pinned = if self.pinned == k { None } else { k };
        }
    }

    /// Bring what is drawn towards what it shows: each place in the column, the
    /// column's range, the coverage chart, and the sample taken apart.
    fn follow(&mut self, f: &mut Frame) {
        let dt = f.real_dt;
        let mut moving = false;
        let axis = self.axis_range();
        let span = axis[1] - axis[0];
        let (far_lo, far_hi) = (axis[0] - 0.06 * span, axis[1] + 0.06 * span);
        let count = self.count();
        let first = count.saturating_sub(SLOTS);
        let c = self.critical();
        for (j, slot) in self.slots.iter_mut().enumerate() {
            // the sample in this place: of the last SLOTS, the one whose number is j more
            // than a multiple of SLOTS
            let k = first + (j + SLOTS - first % SLOTS) % SLOTS;
            let k = (k < count).then_some(k);
            slot.fresh = false;
            let Some(k) = k else {
                slot.k = None;
                continue;
            };
            let s = self.sums[k];
            let (mean, _) = self.setup.scaled(s);
            let iv = self.setup.read_at(s, c);
            let to = match iv {
                Some(iv) => [iv.lo.max(far_lo).min(far_hi) as f32, iv.hi.max(far_lo).min(far_hi) as f32, iv.est.max(far_lo).min(far_hi) as f32],
                None => [mean as f32; 3],
            };
            slot.verdict = iv.map(|iv| iv.verdict(self.setup.truth));
            slot.zero = iv.is_some_and(|iv| iv.zero_width());
            if slot.k != Some(k) {
                slot.k = Some(k);
                slot.glide.snap(&to);
                slot.fresh = true;
                slot.resting = true;
            } else {
                slot.resting = !slot.glide.follow(&to, dt, false);
                moving |= !slot.resting;
            }
        }
        if self.stale == 0 {
            let ks = chart_counts(self.tally.count);
            let ys: Vec<f32> = ks.iter().map(|k| self.tally.cum[k - 1] as f32 / *k as f32).collect();
            moving |= self.curve.follow(&ys, dt, true);
            // the end, where the newest sample has brought the share: a point of its own,
            // gliding there (the curve gains a point now and then, which would start
            // where it is going)
            let end = ks.last().map_or([0.0, 0.0], |k| [(*k as f32).log10(), ys.last().copied().unwrap_or(0.0)]);
            moving |= self.end.follow(&end, dt, ks.len() > 1);
        }
        self.follow_anatomy(f);
        self.moving = moving;
    }

    /// The sample taken apart: when it is another than the one drawn, its values go
    /// there from where they are, evenly, in `Anatomy::MOVE` seconds (by the scene's
    /// clock while it runs, else by the real one), and are held there for
    /// `Anatomy::HOLD` before another may take its place: a picture that is never
    /// still could not be read against its arithmetic.
    fn follow_anatomy(&mut self, f: &Frame) {
        let clock = if f.dt > 0.0 { f.dt } else { f.real_dt };
        let want = self.focus();
        let key = (self.setup, self.seed);
        let c = self.critical();
        let a = &mut self.anatomy;
        a.t = (a.t + clock / Anatomy::MOVE).min(Anatomy::REST);
        let changed = a.key != Some(key);
        if (a.k != want && (a.t >= Anatomy::REST || a.k.is_none())) || changed {
            let to = match want {
                Some(k) => anatomy_values(&self.setup, c, self.seed, k, self.sums.get(k).copied(), &mut self.buf),
                None => Vec::new(),
            };
            let drawn: Vec<f32> = (0..a.to.len()).map(|i| a.drawn(i)).collect();
            // (a sample of another size, or another setting, is drawn where it is)
            let glide = drawn.len() == to.len() && !changed;
            a.from = if glide { drawn } else { to.clone() };
            a.t = if glide { 0.0 } else { Anatomy::REST };
            a.before = if glide { a.k } else { want };
            a.to = to;
            a.k = want;
            a.key = Some(key);
        }
    }
}

/// Where the anatomy draws sample `k`: each value across and up (on a band of their
/// own; a share's ones and noughts in two piles), then its estimate and its interval's
/// ends (the estimate thrice where there is no interval).
fn anatomy_values(setup: &Setup, c: f64, seed: u64, k: usize, s: Option<Summary>, buf: &mut Vec<f64>) -> Vec<f32> {
    raw(setup.pop, seed, k, setup.n, buf);
    let mut out: Vec<f32> = Vec::with_capacity(2 * buf.len() + 3);
    let columns = ((buf.len() as f32).sqrt().ceil() as usize).clamp(1, 10);
    let mut piles = [0usize; 2];
    for (i, r) in buf.iter().enumerate() {
        let v = setup.value(*r);
        if setup.pop == Population::Share {
            let pile = &mut piles[(v > 0.5) as usize];
            let (col, row) = (*pile % columns, *pile / columns);
            *pile += 1;
            out.push((v + (col as f64 - (columns - 1) as f64 / 2.0) * 0.022) as f32);
            out.push(0.29 + 0.062 * row as f32);
        } else {
            out.push(v as f32);
            out.push(0.29 + 0.26 * (0.618_034 * i as f32 + 0.2).fract());
        }
    }
    let iv = s.and_then(|s| setup.read_at(s, c));
    let (mean, _) = s.map(|s| setup.scaled(s)).unwrap_or((f64::NAN, 0.0));
    let far = 1e6f32;
    match iv {
        Some(iv) => out.extend([iv.est as f32, (iv.lo as f32).max(-far), (iv.hi as f32).min(far)]),
        None => out.extend([mean as f32; 3]),
    }
    out
}

/// Buttons side by side from x0, each as wide as its words; squeezed to equal widths
/// filling x0 to x1 if they do not fit.
fn fitted_row(f: &Frame, words: &[&str], x0: f32, x1: f32, y: f32, h: f32, size: f32) -> Vec<Rect> {
    let gap = 6.0;
    let rects = plot::button_row(f, words, x0, y, h, size, gap);
    if rects.last().is_some_and(|r| r.1.x > x1) {
        plot::even_row(words.len(), x0, x1, y, h, 4.0)
    } else {
        rects
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, run to `samples` samples.
    fn run(k: usize, samples: usize) -> Intervals {
        let mut s = Intervals::create();
        s.set_up(k);
        s.goal = samples;
        while s.running() {
            s.step();
        }
        s.refresh();
        s.rework(None);
        s
    }

    /// The same with the seed given.
    fn run_seeded(k: usize, seed: u64, samples: usize) -> Intervals {
        let mut s = Intervals::create();
        s.set_up(k);
        s.seed = seed;
        s.restart();
        s.goal = samples;
        while s.running() {
            s.step();
        }
        s
    }

    /// The share of `s`'s samples that cover, fall below and fall above, read by `setup`.
    fn shares(s: &Intervals) -> (f64, f64, f64) {
        let t = &s.tally;
        let n = t.count as f64;
        (t.covered as f64 / n, t.below as f64 / n, t.above as f64 / n)
    }

    /// The opening: of the first hundred, 95 cover, 3 miss below and 2 above, and the
    /// first sample (the one the last experiment shows alone) covers; run on to 10 000,
    /// the share settles near 95 %.
    #[test]
    fn the_opening_hundred() {
        let s = run(OPENING, 100);
        assert_eq!((s.tally.covered, s.tally.below, s.tally.above), (95, 3, 2), "{:?}", s.tally);
        assert_eq!(s.interval(0).map(|iv| iv.verdict(s.setup.truth)), Some(Verdict::Covers));
        let s = run(OPENING, 10_000);
        assert!((s.tally.share() - 0.95).abs() < 0.006, "{}", s.tally.share());
        // the opening state is this experiment's
        let fresh = Intervals::create();
        assert_eq!((fresh.setup, fresh.goal, fresh.rate), (s.setup, 100, 10.0));
    }

    /// The critical values come from the distributions, not from a table: 1.96, 2.78 at
    /// n = 5, and the ratios the captions quote.
    #[test]
    fn critical_values_and_widths() {
        let mut s = Intervals::create();
        s.set_up(1);
        let z = s.setup.critical();
        s.choose_method(Method::StudentT);
        let t = s.setup.critical();
        assert!((z - 1.959_964).abs() < 1e-5 && (t - 2.776_445).abs() < 1e-5, "{z} {t}");
        assert!((t / z - 1.42).abs() < 0.005, "{}", t / z);
        s.choose_method(Method::KnownZ);
        let known = s.setup;
        let at = |level: usize| Setup { level, ..known }.critical() / Setup { level: L95, ..known }.critical();
        assert!((at(4) - 1.31).abs() < 0.005, "{}", at(4));
        assert!((at(0) - 0.344).abs() < 0.001 && at(0) > 1.0 / 3.0 && at(0) < 0.35, "{}", at(0));
        assert!(Setup { level: 5, ..s.setup }.critical().is_infinite());
    }

    /// With the values from a bell, Student's t covers what it promises, at every n and
    /// every level; and so does z with the spread known.
    #[test]
    fn t_covers_what_it_promises_for_a_bell() {
        for n in [2usize, 3, 5, 10, 30] {
            let mut s = Intervals::create();
            s.set_up(1);
            s.setup.n = n;
            s.refresh();
            s.goal = 20_000;
            while s.running() {
                s.step();
            }
            for method in [Method::StudentT, Method::KnownZ] {
                s.choose_method(method);
                for level in 0..5 {
                    s.setup.level = level;
                    s.recount();
                    let l = LEVELS[level];
                    let sd = (l * (1.0 - l) / 20_000.0).sqrt();
                    let c = s.tally.share();
                    assert!((c - l).abs() < 3.5 * sd, "n = {n}, {method:?} at {l}: {c}");
                }
            }
        }
    }

    /// Too narrow: five values, s, and z cover 88 % (s is below sigma in 59 % of
    /// samples); Student's t on the same samples is 1.42 times as wide and covers 95 %.
    #[test]
    fn too_narrow_with_z() {
        let mut s = run(1, 40_000);
        let (c, below, above) = shares(&s);
        assert!((c - 0.88).abs() < 0.006 && (below - above).abs() < 0.008, "{c} {below} {above}");
        let small = s.sums.iter().filter(|x| x.sd < 1.0).count() as f64 / 40_000.0;
        assert!((small - 0.59).abs() < 0.008, "{small}");
        let before: Vec<Iv> = (0..100).filter_map(|k| s.interval(k)).collect();
        s.choose_method(Method::StudentT);
        s.refresh();
        let (c, _, _) = shares(&s);
        assert!((c - 0.95).abs() < 0.004, "{c}");
        for (k, b) in before.iter().enumerate() {
            let a = s.interval(k).expect("an interval");
            assert!(a.est == b.est && ((a.hi - a.est) / (b.hi - b.est) - 1.4166).abs() < 1e-3, "{a:?} {b:?}");
        }
        // what the scene shows after its first thousand
        let s = run(1, 1000);
        assert!((s.tally.share() - 0.876).abs() < 0.0005, "{}", s.tally.share());
    }

    /// What sets the width: four times the data, exactly half the width on the same
    /// samples, and still 95 %; 99 % is 1.31 times as wide, 50 % a third (the ratios
    /// are held above), and in the long run half miss.
    #[test]
    fn what_sets_the_width() {
        let mut s = run(2, 100);
        let wide: Vec<Iv> = (0..100).filter_map(|k| s.interval(k)).collect();
        s.setup.n = 80;
        s.refresh();
        s.rework(None);
        for (k, w) in wide.iter().enumerate() {
            let a = s.interval(k).expect("an interval");
            assert!(((a.hi - a.lo) / (w.hi - w.lo) - 0.5).abs() < 1e-12, "{a:?} {w:?}");
        }
        // the first twenty values of each sample are the ones it had
        let mut buf = Vec::new();
        let mut short = Vec::new();
        raw(Population::Bell, SEED, 7, 80, &mut buf);
        raw(Population::Bell, SEED, 7, 20, &mut short);
        assert_eq!(&buf[..20], &short[..]);
        let mut s = run(2, 20_000);
        s.setup.n = 80;
        s.refresh();
        s.rework(None);
        assert!((s.tally.share() - 0.95).abs() < 0.005, "{}", s.tally.share());
        s.setup.level = 0;
        s.refresh();
        assert!((s.tally.share() - 0.5).abs() < 0.012, "{}", s.tally.share());
    }

    /// A long right tail, ten values, Student's t: 84 % cover, 16 % miss below and 0.1 %
    /// above; the mean is below the truth in 59 % of samples, and of 40 000 samples every
    /// interval that missed below had an s below sigma. (The population: 69 % of its
    /// values below its mean, a standard deviation 1.31 times it.)
    #[test]
    fn a_long_tail_misses_below() {
        let s = run(3, 40_000);
        let (c, below, above) = shares(&s);
        assert!((c - 0.84).abs() < 0.006 && (below - 0.16).abs() < 0.006 && (above - 0.001).abs() < 0.0006, "{c} {below} {above}");
        let sigma = s.setup.sigma();
        let (mut low, mut small_when_low) = (0usize, true);
        for (k, sum) in s.sums.iter().enumerate() {
            let (mean, sd) = s.setup.scaled(*sum);
            low += (mean < s.setup.truth) as usize;
            if s.interval(k).map(|iv| iv.verdict(s.setup.truth)) == Some(Verdict::Below) {
                small_when_low &= sd < sigma;
            }
        }
        assert!((low as f64 / 40_000.0 - 0.59).abs() < 0.008 && small_when_low, "{low}");
        // the standard deviation the panel quotes, against the draws
        let draws: Vec<f64> = (0..200).flat_map(|k| {
            let mut b = Vec::new();
            raw(Population::Tail, 5, k, 200, &mut b);
            b.into_iter().map(|r| value_of(Population::Tail, 1.0, r)).collect::<Vec<_>>()
        }).collect();
        assert!((st::mean(&draws) - 1.0).abs() < 0.03 && (st::sd(&draws) - 1.31).abs() < 0.08, "{} {}", st::mean(&draws), st::sd(&draws));
        assert!((s.setup.sigma() / s.setup.truth - 1.311).abs() < 0.001);
        // 69 % of its values below its mean
        let below = draws.iter().filter(|v| **v < 1.0).count() as f64 / draws.len() as f64;
        assert!((st::normal_cdf(0.5) - 0.69).abs() < 0.002 && (below - 0.69).abs() < 0.01, "{below}");
    }

    /// A share near zero: the usual interval covers 78 %; in 21 % of samples there is no
    /// success and the interval is the single point 0, which misses; Wilson's covers 94 %
    /// and is never a single point.
    #[test]
    fn a_share_near_zero() {
        let mut s = run(4, 40_000);
        let (c, _, _) = shares(&s);
        let zero = s.tally.zero as f64 / 40_000.0;
        assert!((c - 0.78).abs() < 0.006 && (zero - 0.2146).abs() < 0.006 && s.tally.zero_missed == s.tally.zero, "{c} {zero}");
        assert!((0.95f64.powi(30) - 0.2146).abs() < 1e-4);
        s.choose_method(Method::Wilson);
        s.refresh();
        let (c, _, _) = shares(&s);
        assert!((c - 0.94).abs() < 0.006 && s.tally.zero == 0, "{c} {}", s.tally.zero);
    }

    /// What 95 % does not mean: one interval, the truth hidden; it covers.
    #[test]
    fn one_interval_alone() {
        let mut s = Intervals::create();
        s.set_up(5);
        assert!(!s.show_truth && s.statement && s.goal == 1);
        while s.running() {
            s.step();
        }
        assert_eq!(s.count(), 1);
        assert_eq!(s.interval(0).map(|iv| iv.verdict(s.setup.truth)), Some(Verdict::Covers));
    }

    /// The ends of every range: one value with the spread to estimate (no interval),
    /// a share of 0 or 1, a level of 100 %.
    #[test]
    fn degenerate_settings_are_told_honestly() {
        let one = |method| Setup { pop: Population::Bell, truth: 10.0, n: 1, level: L95, method };
        let s = Summary { mean: 0.3, sd: 0.0 };
        assert!(one(Method::StudentT).read(s).is_none() && one(Method::PlugInZ).read(s).is_none());
        assert!(one(Method::StudentT).critical().is_nan());
        let iv = one(Method::KnownZ).read(s).expect("sigma known needs no spread");
        assert!((iv.hi - iv.lo - 2.0 * 1.959_964 * BELL_SD).abs() < 1e-4);
        // a share of 0: every sample all noughts
        for (p, ones) in [(0.0, 0.0), (1.0, 1.0)] {
            let mut sc = Intervals::create();
            sc.set_up(4);
            sc.setup.truth = p;
            sc.refresh();
            sc.goal = 50;
            while sc.running() {
                sc.step();
            }
            assert!(sc.sums.iter().all(|x| x.mean == ones));
            assert_eq!((sc.tally.covered, sc.tally.zero), (50, 50), "p = {p}");
            sc.choose_method(Method::Wilson);
            sc.refresh();
            let iv = sc.interval(0).expect("an interval");
            assert!(sc.tally.covered == 50 && sc.tally.zero == 0 && iv.lo >= 0.0 && iv.hi <= 1.0, "{iv:?}");
        }
        // 100 %: the whole line (for a share, all of 0 to 1), and always covering; but a
        // sample with no spread is still a point
        let all = Setup { level: 5, ..one(Method::KnownZ) };
        let iv = all.read(s).expect("an interval");
        assert!(iv.lo == f64::NEG_INFINITY && iv.hi == f64::INFINITY && iv.verdict(10.0) == Verdict::Covers);
        let share = Setup { pop: Population::Share, truth: 0.3, n: 10, level: 5, method: Method::Wilson };
        assert_eq!(share.read(Summary { mean: 0.0, sd: 0.0 }).map(|iv| (iv.lo, iv.hi)), Some((0.0, 1.0)));
        let wald = Setup { method: Method::Wald, ..share };
        assert!(wald.read(Summary { mean: 0.0, sd: 0.0 }).is_some_and(|iv| iv.zero_width()));
        assert!(wald.read(Summary { mean: 0.4, sd: 0.0 }).is_some_and(|iv| iv.hi.is_infinite()));
        // the anatomy's numbers stay finite at every edge
        for setup in [all, one(Method::StudentT), share, wald] {
            let v = anatomy_values(&setup, setup.critical(), 3, 0, Some(s), &mut Vec::new());
            assert!(v.iter().all(|x| x.is_finite()), "{setup:?}: {v:?}");
        }
    }

    /// A change of n reads the same samples again: the same counts as a run made with
    /// that n from the start, newest first and then the rest.
    #[test]
    fn a_change_of_n_reads_the_samples_again() {
        let mut s = run(2, 600);
        s.setup.n = 37;
        s.refresh();
        assert!(s.stale == 600 - SLOTS);
        let budget = Budget::new(0.0);
        s.rework(Some(&budget));
        assert!(s.stale < 600 - SLOTS);
        s.rework(None);
        let mut fresh = Intervals::create();
        fresh.set_up(2);
        fresh.setup.n = 37;
        fresh.refresh();
        fresh.goal = 600;
        while fresh.running() {
            fresh.step();
        }
        assert_eq!(s.sums, fresh.sums);
        assert_eq!(s.tally, fresh.tally);
    }

    /// The layout fits every window: everything inside it, and in order down a phone.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h, inset) in [(1075.0, 1000.0, 0.0), (1400.0, 760.0, 0.0), (390.0, 756.0, 44.0), (700.0, 500.0, 0.0), (240.0, 180.0, 0.0), (1.0, 1.0, 0.0), (3000.0, 2000.0, 0.0)] {
            let lay = Layout::of(Vec2::new(w, h), inset);
            let rects: Vec<Rect> = [Some(lay.column), lay.anatomy, lay.cover].into_iter().flatten().collect();
            assert!(rects.iter().all(|(a, b)| a.is_finite() && b.is_finite() && b.y >= a.y && b.x >= a.x), "{w}x{h}: {lay:?}");
            if w >= 240.0 {
                for (a, b) in &rects {
                    assert!(a.x >= 0.0 && a.y >= inset && b.x <= w && b.y <= h, "{w}x{h}: {a} {b}");
                }
                // one under another on a phone, side by side on a desktop, nothing over anything
                if let (Some(a), Some(c)) = (lay.anatomy, lay.cover) {
                    assert!(a.1.y < c.0.y, "{w}x{h}: {lay:?}");
                }
                if let Some(a) = lay.anatomy {
                    let key = lay.key.unwrap_or(Vec2::splat(f32::MAX));
                    assert!(if lay.narrow { lay.column.1.y + 30.0 < key.y && key.y + lay.small < a.0.y } else { lay.column.1.x < a.0.x }, "{w}x{h}: {lay:?}");
                }
                assert!(lay.row() > 0.5, "{w}x{h}: rows of {}", lay.row());
            }
            // a phone and a desktop show everything
            if (w, h) == (390.0, 756.0) || (w, h) == (1075.0, 1000.0) || (w, h) == (1400.0, 760.0) {
                assert!(lay.anatomy.is_some() && lay.cover.is_some() && lay.methods.is_some() && lay.row() > 2.5, "{w}x{h}: {lay:?}");
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore_numbers() {
        // the opening, run on
        let s = run(0, 10_000);
        println!("opening at 10 000: {:.4}", s.tally.share());
        // the width experiment at 50 %
        let mut s = run(2, 100);
        s.setup.level = 0;
        s.refresh();
        println!("width at 50 %, first 100: covered {}", s.tally.covered);
        // the tail: how often the mean is low, and how small s is when an interval misses below
        let s = run(3, 40_000);
        let (mut low, mut missed, mut small) = (0usize, 0usize, 0usize);
        let sigma = s.setup.sigma();
        for (k, sum) in s.sums.iter().enumerate() {
            let (mean, sd) = s.setup.scaled(*sum);
            if mean < s.setup.truth {
                low += 1;
            }
            if s.interval(k).map(|iv| iv.verdict(s.setup.truth)) == Some(Verdict::Below) {
                missed += 1;
                if sd < sigma {
                    small += 1;
                }
            }
        }
        println!("tail: coverage {:.4}, below {:.4}, above {:.5}, mean low in {:.3}, misses below with s < sigma {:.3}", s.tally.share(), s.tally.below as f64 / 4e4, s.tally.above as f64 / 4e4, low as f64 / 4e4, small as f64 / missed as f64);
        // with five values from a bell, how often s is below sigma
        let s = run(1, 40_000);
        let below = s.sums.iter().filter(|x| x.sd < 1.0).count() as f64 / 4e4;
        println!("n = 5: s below sigma in {below:.3}; coverage with z {:.4}", s.tally.share());
        let s = run(4, 40_000);
        println!("share: wald {:.4}, single points {:.4} missed {}", s.tally.share(), s.tally.zero as f64 / 4e4, s.tally.zero_missed);
    }

    /// Seeds whose first hundred intervals, as the scene opens, cover 95 of 100 with the
    /// first of them covering, and what the other experiments show after a thousand
    /// with each (how `SEED` was chosen; run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore_seeds() {
        for seed in 1..400u64 {
            let s = run_seeded(0, seed, 100);
            let t = &s.tally;
            let first = s.interval(0).map(|iv| iv.verdict(10.0));
            if t.covered == 95 && first == Some(Verdict::Covers) {
                let counts: Vec<String> = (1..5)
                    .map(|k| {
                        let r = run_seeded(k, seed, 1000);
                        format!("{:.1}", 100.0 * r.tally.share())
                    })
                    .collect();
                println!("seed {seed}: below {} above {}; first 1000 of the others: {}", t.below, t.above, counts.join(" "));
            }
        }
    }
}

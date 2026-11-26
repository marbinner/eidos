//! Seeing and doing: the units that happen to have X = x, against every unit set to x.

use super::*;
use crate::math::causal::{line, mean};

pub const DEF: SceneDef = SceneDef {
    id: "dosee",
    title: "Seeing and doing",
    group: GROUP,
    blurb: "Watching who happens to take a treatment is not the same as giving it: what an intervention cuts.",
    make: |_| Box::new(DoSee::new()),
};

/// The most units a population can have, and how many it has to begin with.
const MOST: usize = 2000;
const UNITS: usize = 1000;
/// The spread of what the arrows leave unexplained in X, and in Y (each unit's own
/// noise, times these).
const NOISE_X: f64 = 0.6;
const NOISE_Y: f64 = 0.5;
/// In an experiment X is dealt at random, evenly between −DEALT and DEALT.
const DEALT: f64 = 2.5;
/// Seeing picks out the units whose X is within this of x, to begin with.
const WINDOW: f64 = 0.25;
/// The population every setting starts from.
const SEED: u64 = 8;
/// The histograms' bins: this wide in Y, from −SPAN to SPAN.
const BIN: f64 = 0.25;
const SPAN: f64 = 12.0;
const BINS: usize = (2.0 * SPAN / BIN) as usize;
/// The seeing curve is left out where fewer units than this are seen.
const FEW: usize = 8;
/// Fewer units seen than this, and the words say that their mean says little.
const HANDFUL: usize = 10;
/// How long the units take to travel to x when it is set, and back (seconds).
const TRAVEL: f32 = 1.3;
/// How long a glide to new settings takes, about (each of two eases, in seconds).
const GLIDE: f64 = 0.09;

// ----- the model ----------------------------------------------------------------------

/// Which way the arrow between X and Y points.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Arrow {
    /// X causes Y.
    Forward,
    /// Y causes X, and X does nothing to Y.
    Backward,
}

/// How each unit's X and Y are made from its Z and its own noise: with the arrow
/// forward, X = aZ + noise and Y = bX + cZ + noise; with it backward, Y = cZ + noise
/// and X = bY + aZ + noise. `dealt`: an experiment, in which X is dealt at random
/// instead (and nothing else has a say in it).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Model {
    arrow: Arrow,
    a: f64,
    b: f64,
    c: f64,
    dealt: bool,
}

/// One unit's own randomness: its Z, the noise in its X and in its Y, and the number
/// an experiment would deal it (from −1 to 1).
#[derive(Clone, Copy, Debug)]
struct Unit {
    z: f64,
    nx: f64,
    ny: f64,
    deal: f64,
}

impl Model {
    /// The unit's X and Y as they come, with nothing set.
    fn natural(&self, u: &Unit) -> (f64, f64) {
        match self.arrow {
            Arrow::Forward => {
                let x = if self.dealt { DEALT * u.deal } else { self.a * u.z + NOISE_X * u.nx };
                (x, self.b * x + self.c * u.z + NOISE_Y * u.ny)
            }
            Arrow::Backward => {
                let y = self.c * u.z + NOISE_Y * u.ny;
                let x = if self.dealt { DEALT * u.deal } else { self.b * y + self.a * u.z + NOISE_X * u.nx };
                (x, y)
            }
        }
    }

    /// The unit's Y with X set to `x` by hand: its own Z and its own noise, and only X
    /// different. Every arrow into X is cut, and the rest is left as it was.
    fn done(&self, u: &Unit, x: f64) -> f64 {
        match self.arrow {
            Arrow::Forward => self.b * x + self.c * u.z + NOISE_Y * u.ny,
            Arrow::Backward => self.c * u.z + NOISE_Y * u.ny,
        }
    }

    /// How far a unit's Y moves for each step its X is moved by hand: b if X causes Y,
    /// nothing if it does not. (So each unit travels along a line of this slope.)
    fn response(&self) -> f64 {
        match self.arrow {
            Arrow::Forward => self.b,
            Arrow::Backward => 0.0,
        }
    }
}

/// A population: each unit's own randomness, the same for every model, so that a
/// change of the arrows moves the same units.
fn population(seed: u64) -> Vec<Unit> {
    let mut rng = Rng::new(seed);
    (0..MOST).map(|_| Unit { z: rng.normal() as f64, nx: rng.normal() as f64, ny: rng.normal() as f64, deal: rng.range(-1.0, 1.0) as f64 }).collect()
}

/// What looking at the units says at one x.
#[derive(Clone, Copy, Debug)]
struct Look {
    units: usize,
    /// How many units have X within the window of x, and the means of their Y and Z
    /// (0 when there are none).
    seen: usize,
    seen_y: f64,
    seen_z: f64,
    /// The means of Y and of Z over every unit, with X set to x.
    done_y: f64,
    done_z: f64,
}

/// The units of `model` (the first `count` of `pop`) seen and set at `x`.
fn look(model: &Model, pop: &[Unit], count: usize, x: f64, window: f64) -> Look {
    let units = &pop[..count.min(pop.len())];
    let (mut seen, mut sy, mut sz) = (0usize, 0.0, 0.0);
    let (mut dy, mut dz) = (0.0, 0.0);
    for u in units {
        let (ux, uy) = model.natural(u);
        if (ux - x).abs() <= window {
            seen += 1;
            sy += uy;
            sz += u.z;
        }
        dy += model.done(u, x);
        dz += u.z;
    }
    let per = |sum: f64, n: usize| if n > 0 { sum / n as f64 } else { 0.0 };
    Look { units: units.len(), seen, seen_y: per(sy, seen), seen_z: per(sz, seen), done_y: per(dy, units.len()), done_z: per(dz, units.len()) }
}

/// The trend of what is seen: the slope of the least-squares line of Y against X,
/// over every unit as it comes.
fn trend(model: &Model, pop: &[Unit], count: usize) -> f64 {
    let (x, y): (Vec<f64>, Vec<f64>) = pop[..count.min(pop.len())].iter().map(|u| model.natural(u)).unzip();
    line(&x, &y).0
}

/// The average causal effect: how much the mean of Y over every unit rises when X is
/// set one higher (the slope of the line of doing; the same at every x here, since
/// every effect is a straight line).
fn effect(model: &Model, pop: &[Unit], count: usize) -> f64 {
    let units = &pop[..count.min(pop.len())];
    let at = |x: f64| mean(&units.iter().map(|u| model.done(u, x)).collect::<Vec<f64>>());
    at(1.0) - at(0.0)
}

/// The mean of Y among the units whose X is within a window of any x: the units in
/// order of X, and the running sums of their Y.
struct Seeing {
    xs: Vec<f64>,
    sums: Vec<f64>,
}

impl Seeing {
    fn of(xs: &[f64], ys: &[f64]) -> Seeing {
        let mut order: Vec<usize> = (0..xs.len().min(ys.len())).collect();
        order.sort_by(|&i, &j| xs[i].total_cmp(&xs[j]));
        let mut sums = Vec::with_capacity(order.len() + 1);
        sums.push(0.0);
        for &i in &order {
            sums.push(sums[sums.len() - 1] + ys[i]);
        }
        Seeing { xs: order.iter().map(|&i| xs[i]).collect(), sums }
    }

    /// How many units have X within `window` of `x`, and their mean Y (none if none).
    fn at(&self, x: f64, window: f64) -> Option<(usize, f64)> {
        let lo = self.xs.partition_point(|v| *v < x - window);
        let hi = self.xs.partition_point(|v| *v <= x + window);
        (hi > lo).then(|| (hi - lo, (self.sums[hi] - self.sums[lo]) / (hi - lo) as f64))
    }
}

/// The bin a value of Y falls in, if any.
fn bin(y: f64) -> Option<usize> {
    let k = ((y + SPAN) / BIN).floor();
    (k >= 0.0 && k < BINS as f64).then_some(k as usize)
}

/// The share of `values` in each bin, per unit of Y (a density: the bars' area is 1).
fn density(values: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut counts = vec![0.0; BINS];
    let mut n = 0usize;
    for v in values {
        n += 1;
        if let Some(k) = bin(v) {
            counts[k] += 1.0;
        }
    }
    if n > 0 {
        counts.iter_mut().for_each(|c| *c /= n as f64 * BIN);
    }
    counts
}

/// The value below which a share `q` of `v` lies (`v` sorted; 0 if empty).
fn quantile(v: &[f64], q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let at = (q * (v.len() - 1) as f64).round() as usize;
    v[at.min(v.len() - 1)]
}

// ----- what is set ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    See,
    Do,
}

/// Everyday names for Z, X and Y, or none.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Names {
    Plain,
    Medicine,
}

impl Names {
    /// Z's, X's and Y's.
    fn of(self) -> Option<[&'static str; 3]> {
        match self {
            Names::Plain => None,
            Names::Medicine => Some(["illness", "dose", "recovery"]),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    model: Model,
    x: f64,
    window: f64,
    units: usize,
    seed: u64,
    mode: Mode,
    names: Names,
}

const fn setup(arrow: Arrow, [a, b, c]: [f64; 3], dealt: bool, x: f64, mode: Mode, names: Names) -> Setup {
    Setup { model: Model { arrow, a, b, c, dealt }, x, window: WINDOW, units: UNITS, seed: SEED, mode, names }
}

/// What the scene opens with: Z pushes X and Y both, and x is picked by seeing.
const OPENING: Setup = setup(Arrow::Forward, [1.0, 0.5, 1.0], false, 1.5, Mode::See, Names::Plain);

/// The experiments. Every number in a caption is checked by a test of the same setting.
const TRY: [(&str, &str, Setup); 5] = [
    (
        "seeing is not doing",
        r"The units that happen to have $X$ near $1.5$ (pink) have a mean $Y$ of $1.84$. Set $X$ to $1.5$ for every unit instead, and the mean $Y$ is $0.77$ (green). Those seen are not a fair sample: $Z$ pushed their $X$ up, so their mean $Z$ is $1.07$, against $0.02$ over everyone, and $Z$ raises $Y$ by itself. Watch the units travel: each keeps its own $Z$ and noise, and so moves along a line of slope $b = 0.5$.",
        setup(Arrow::Forward, [1.0, 0.5, 1.0], false, 1.5, Mode::Do, Names::Plain),
    ),
    (
        "no confounder",
        r"$Z$ still pushes $X$, but no longer touches $Y$ ($c = 0$). The units seen near $1.5$ still have a high $Z$ (mean $1.07$), but now that does nothing to their $Y$: seeing gives a mean $Y$ of $0.77$ and doing $0.75$, and the two histograms nearly mirror each other. A confounder has to reach both.",
        setup(Arrow::Forward, [1.0, 0.5, 0.0], false, 1.5, Mode::Do, Names::Plain),
    ),
    (
        "a randomised experiment",
        r"The units of the first, but each dealt its $X$ at random, so that $Z$ has no say in it (its arrow is cut). Now merely seeing the results gives doing's answer: the 89 units dealt an $X$ near $1.5$ have a mean $Y$ of $0.74$, against doing's $0.77$ (apart only by the luck of which units were dealt it), and what is seen rises by $0.52$ per unit of $X$, against the causal $0.50$.",
        setup(Arrow::Forward, [1.0, 0.5, 1.0], true, 1.5, Mode::See, Names::Plain),
    ),
    (
        "the arrow the other way",
        r"Here $Y$ causes $X$, and $X$ does nothing to $Y$. Seeing shows a clear trend: $Y$ rises by $0.77$ per unit of $X$, and the units with $X$ near $1.5$ have a mean $Y$ of $1.16$. Doing shows none: set $X$ by hand and every unit keeps its $Y$, so the units slide straight across and the green line is flat, at $0.02$.",
        setup(Arrow::Backward, [0.0, 1.0, 1.0], false, 1.5, Mode::Do, Names::Plain),
    ),
    (
        "a treatment that seems to harm",
        r"$Z$ is how ill a patient is, $X$ the dose given and $Y$ the recovery, each measured from its average. Doctors give more to the sicker, and the sicker recover less, so those seen with high doses do worse: seen, recovery falls by $0.60$ per unit of dose. Yet a higher dose helps every patient alike: each moves up along a line of slope $0.5$ as the dose is raised, and the green line rises by $0.50$.",
        setup(Arrow::Forward, [1.0, 0.5, -1.5], false, 1.5, Mode::Do, Names::Medicine),
    ),
];

// ----- what is drawn follows what is set ------------------------------------------------

/// A copy of some numbers that follows them: two eases one after the other (so that it
/// sets off gently rather than at full speed), and exactly there in the end.
#[derive(Clone, Debug, Default)]
struct Follow {
    mid: Vec<f64>,
    shown: Vec<f64>,
}

impl Follow {
    /// There at once.
    fn jump(&mut self, v: &[f64]) {
        self.mid = v.to_vec();
        self.shown = v.to_vec();
    }

    fn get(&self, i: usize) -> f64 {
        self.shown.get(i).copied().unwrap_or(0.0)
    }

    /// `dt` seconds nearer to `to`; true while still on the way (it arrives when within
    /// `close` of it). A number new to the list starts where it is going.
    fn toward(&mut self, to: &[f64], dt: f64, close: f64) -> bool {
        let had = self.shown.len().min(self.mid.len()).min(to.len());
        self.mid.truncate(had);
        self.shown.truncate(had);
        self.mid.extend_from_slice(&to[had..]);
        self.shown.extend_from_slice(&to[had..]);
        let k = 1.0 - (-dt.max(0.0) / GLIDE).exp();
        let mut apart = 0.0f64;
        for ((m, s), t) in self.mid.iter_mut().zip(self.shown.iter_mut()).zip(to) {
            *m += (t - *m) * k;
            *s += (*m - *s) * k;
            apart = apart.max((t - *m).abs()).max((t - *s).abs());
        }
        // (exactly there in the end, not nearly: the picture then stops changing)
        if apart < close || !apart.is_finite() {
            self.jump(to);
            false
        } else {
            true
        }
    }
}

/// Where everything is going: each unit's X and Y as it comes, the slope it travels
/// along when X is set, and the ranges of X and Y the scatter shows.
struct Goal {
    xs: Vec<f64>,
    ys: Vec<f64>,
    slope: f64,
    /// Lowest and highest X, lowest and highest Y.
    frame: [f64; 4],
    /// The units in order of X, for the means of those seen.
    seeing: Seeing,
    /// The least-squares slope of what is seen, and the average causal effect.
    trend: f64,
    effect: f64,
}

/// How many pieces the curve of what is seen is drawn in.
const STEPS: usize = 160;

// ----- the scene ------------------------------------------------------------------------

struct DoSee {
    model: Model,
    x: f64,
    window: f64,
    units: usize,
    seed: u64,
    mode: Mode,
    names: Names,
    pop: Vec<Unit>,
    goal: Goal,
    /// Each unit's X and Y as drawn, the slope they travel along, and the ranges shown.
    xs: Follow,
    ys: Follow,
    slope: Follow,
    frame: Follow,
    /// x and the window's half-width as drawn: they glide to an experiment's, and
    /// follow the hand at once.
    at: Follow,
    /// How far the units have travelled towards x: 0 where they came, 1 all at x.
    travel: f32,
    /// Going back to where they came before setting off again (an experiment of doing
    /// chosen while the units are at x already, so that it is seen to happen).
    again: bool,
    /// How brightly each unit is lit as one that seeing picks out (0 to 1).
    lit: Vec<f32>,
    /// The histograms as drawn: the seen units' Y, then every unit's with X set to x.
    bars: Follow,
    /// The curve of the mean of Y among the units seen at each x, as drawn: the x of
    /// each of its points, then their Y, then how clearly each is drawn (0 where too
    /// few units are seen to say). It follows the units' goal, not the units on their
    /// way, whose means would jump from frame to frame as units pass in and out.
    curve: Follow,
    /// The mean of the seen units' Y at x as drawn, and how clearly it is drawn.
    marks: Follow,
    /// x being dragged across the scatter.
    holding: bool,
    /// Something is still on its way.
    moving: bool,
}

impl DoSee {
    fn new() -> DoSee {
        let mut s = DoSee {
            model: OPENING.model,
            x: OPENING.x,
            window: OPENING.window,
            units: OPENING.units,
            seed: OPENING.seed,
            mode: OPENING.mode,
            names: OPENING.names,
            pop: population(OPENING.seed),
            goal: Goal { xs: Vec::new(), ys: Vec::new(), slope: 0.0, frame: [-3.0, 3.0, -3.0, 3.0], seeing: Seeing::of(&[], &[]), trend: 0.0, effect: 0.0 },
            xs: Follow::default(),
            ys: Follow::default(),
            slope: Follow::default(),
            frame: Follow::default(),
            at: Follow::default(),
            travel: 0.0,
            again: false,
            lit: Vec::new(),
            bars: Follow::default(),
            curve: Follow::default(),
            marks: Follow::default(),
            holding: false,
            moving: true,
        };
        s.apply(OPENING);
        s.settle();
        s
    }

    /// Set an experiment up (the units then glide to their new places).
    fn apply(&mut self, s: Setup) {
        if s.seed != self.seed || self.pop.is_empty() {
            self.pop = population(s.seed);
        }
        (self.model, self.x, self.window, self.units, self.seed, self.mode, self.names) = (s.model, s.x, s.window, s.units, s.seed, s.mode, s.names);
        // (doing again from the start, so that it is seen to happen: the units first go
        // back to where they came, if they are at x)
        self.again = s.mode == Mode::Do && self.travel > 0.0;
        self.refresh();
    }

    /// Work out where everything is going, after any change of what is set.
    fn refresh(&mut self) {
        self.units = self.units.clamp(1, MOST);
        let units = &self.pop[..self.units.min(self.pop.len())];
        let (xs, ys): (Vec<f64>, Vec<f64>) = units.iter().map(|u| self.model.natural(u)).unzip();
        // X: as far either way as nearly every unit reaches, and not less than 3
        let mut reach: Vec<f64> = xs.iter().map(|v| v.abs()).collect();
        reach.sort_by(f64::total_cmp);
        let rx = (quantile(&reach, 0.995) * 1.08).max(3.0);
        let rx = (rx * 2.0).ceil() / 2.0;
        // Y: nearly every unit as it comes, and as it would be with X set at either end
        let mut all: Vec<f64> = ys.clone();
        for end in [-rx, rx] {
            all.extend(units.iter().map(|u| self.model.done(u, end)));
        }
        all.sort_by(f64::total_cmp);
        let (lo, hi) = (quantile(&all, 0.003), quantile(&all, 0.997));
        let pad = 0.06 * (hi - lo).max(1.0);
        let (lo, hi) = (((lo - pad) * 2.0).floor() / 2.0, ((hi + pad) * 2.0).ceil() / 2.0);
        let (lo, hi) = (lo.min(-1.0).max(-SPAN), hi.max(1.0).min(SPAN));
        let seeing = Seeing::of(&xs, &ys);
        let (trend, effect) = (trend(&self.model, &self.pop, self.units), effect(&self.model, &self.pop, self.units));
        self.goal = Goal { xs, ys, slope: self.model.response(), frame: [-rx, rx, lo, hi], seeing, trend, effect };
        self.x = self.x.max(-rx).min(rx);
        self.moving = true;
    }

    /// Everything drawn, there at once (and the tests' way to see the scene at rest).
    fn settle(&mut self) {
        self.xs.jump(&self.goal.xs);
        self.ys.jump(&self.goal.ys);
        self.slope.jump(&[self.goal.slope]);
        self.frame.jump(&self.goal.frame);
        self.at.jump(&[self.x, self.window]);
        self.travel = if self.mode == Mode::Do { 1.0 } else { 0.0 };
        self.again = false;
        self.lit = self.lit_target();
        let bars = self.bars_target();
        self.bars.jump(&bars);
        self.curve.jump(&self.curve_target());
        self.marks.jump(&self.marks_target());
        self.moving = false;
    }

    /// Which units seeing picks out now, as drawn: 1 if its X is within the window.
    fn lit_target(&self) -> Vec<f32> {
        let (at, window) = (self.at.get(0), self.at.get(1));
        self.xs.shown.iter().map(|x| if (x - at).abs() <= window { 1.0 } else { 0.0 }).collect()
    }

    /// Where each unit is drawn with X set to x: its own Y moved along its slope.
    fn done_y(&self, i: usize) -> f64 {
        self.ys.get(i) + self.slope.get(0) * (self.at.get(0) - self.xs.get(i))
    }

    /// The histograms the drawn units make: the seen ones' Y, then every unit's Y
    /// with X set to x.
    fn bars_target(&self) -> Vec<f64> {
        let n = self.xs.shown.len().min(self.ys.shown.len());
        let (at, window) = (self.at.get(0), self.at.get(1));
        let seen = density((0..n).filter(|&i| (self.xs.get(i) - at).abs() <= window).map(|i| self.ys.get(i)));
        let done = density((0..n).map(|i| self.done_y(i)));
        seen.into_iter().chain(done).collect()
    }

    /// Where the curve of what is seen is going: at each of evenly many x across the
    /// scatter, the mean Y of the units seen there (and, where fewer than `FEW` are,
    /// the height of its neighbour, drawn clear).
    fn curve_target(&self) -> Vec<f64> {
        let [x0, x1, ..] = self.goal.frame;
        let at: Vec<(f64, Option<f64>)> = (0..=STEPS)
            .map(|k| {
                let x = x0 + (x1 - x0) * k as f64 / STEPS as f64;
                (x, self.goal.seeing.at(x, self.window).filter(|(count, _)| *count >= FEW).map(|(_, y)| y))
            })
            .collect();
        let mut held = at.iter().find_map(|(_, y)| *y).unwrap_or(0.0);
        let mut out = Vec::with_capacity(3 * at.len());
        out.extend(at.iter().map(|(x, _)| *x));
        for (_, y) in &at {
            held = y.unwrap_or(held);
            out.push(held);
        }
        out.extend(at.iter().map(|(_, y)| if y.is_some() { 1.0 } else { 0.0 }));
        out
    }

    /// Where the mark of what is seen at x is going: the mean Y of the units seen
    /// there, and 1 (or, where there are none, where it was, and 0: drawn clear).
    fn marks_target(&self) -> Vec<f64> {
        let seen = self.goal.seeing.at(self.x, self.window).map(|(_, y)| y);
        vec![seen.unwrap_or(self.marks.get(0)), if seen.is_some() { 1.0 } else { 0.0 }]
    }

    /// The mean Y of every unit with X set to `x`, as drawn: a straight line, since
    /// each unit's Y moves along its slope.
    fn done_shown(&self, x: f64) -> f64 {
        let n = self.xs.shown.len().min(self.ys.shown.len());
        mean(&self.ys.shown[..n]) + self.slope.get(0) * (x - mean(&self.xs.shown[..n]))
    }

    /// Move everything drawn nearer to where it is going: the travel to x and back by
    /// the scene's time, the rest by the clock (it answers the hand, also while the
    /// scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        if self.again && self.travel <= 0.0 {
            self.again = false;
        }
        let to = if self.mode == Mode::Do && !self.again { 1.0 } else { 0.0 };
        let mut moving = self.again;
        if self.travel != to {
            let step = dt.max(0.0) / TRAVEL;
            self.travel = if to > self.travel { (self.travel + step).min(1.0) } else { (self.travel - step).max(0.0) };
            moving |= self.travel != to;
        }
        let dt = real_dt.max(0.0) as f64;
        if self.holding {
            self.at.jump(&[self.x, self.window]);
        } else {
            moving |= self.at.toward(&[self.x, self.window], dt, 1e-4);
        }
        moving |= self.xs.toward(&self.goal.xs, dt, 1e-4);
        moving |= self.ys.toward(&self.goal.ys, dt, 1e-4);
        moving |= self.slope.toward(&[self.goal.slope], dt, 1e-5);
        moving |= self.frame.toward(&self.goal.frame, dt, 1e-4);
        // (a unit lights up as the window reaches it, quickly but not in a flash)
        let lit = self.lit_target();
        self.lit.resize(lit.len(), 0.0);
        let k = 1.0 - (-(dt as f32) / 0.06).exp();
        for (l, t) in self.lit.iter_mut().zip(&lit) {
            *l += (t - *l) * k;
            if (t - *l).abs() < 0.01 {
                *l = *t;
            } else {
                moving = true;
            }
        }
        let bars = self.bars_target();
        moving |= self.bars.toward(&bars, dt, 1e-4);
        let curve = self.curve_target();
        moving |= self.curve.toward(&curve, dt, 1e-4);
        let marks = self.marks_target();
        moving |= self.marks.toward(&marks, dt, 1e-4);
        self.moving = moving;
    }

    /// What looking at the units says now (the settings, not the picture on its way).
    fn look(&self) -> Look {
        look(&self.model, &self.pop, self.units, self.x, self.window)
    }

    /// The least-squares slope of what is seen, and the average causal effect (worked
    /// out when the units change).
    fn trend(&self) -> f64 {
        self.goal.trend
    }

    fn effect(&self) -> f64 {
        self.goal.effect
    }

    /// The names the picture gives Z, X and Y: the letters, with an everyday name after
    /// them in the experiments that have one.
    fn name(&self, k: usize) -> String {
        let letter = ["Z", "X", "Y"][k.min(2)];
        match self.names.of() {
            Some(names) => format!("${letter}$: {}", names[k.min(2)]),
            None => format!("${letter}$"),
        }
    }
}

// ----- where everything goes ------------------------------------------------------------

/// A rectangle on the screen, logical pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    min: Vec2,
    max: Vec2,
}

impl Rect {
    fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
        // (never upside down, and never of no size: a tiny window squeezes, it does not fold)
        let (lo_x, hi_x) = (x0.min(x1), x0.max(x1).max(x0.min(x1) + 1.0));
        let (lo_y, hi_y) = (y0.min(y1), y0.max(y1).max(y0.min(y1) + 1.0));
        Rect { min: Vec2::new(lo_x, lo_y), max: Vec2::new(hi_x, hi_y) }
    }

    fn w(&self) -> f32 {
        self.max.x - self.min.x
    }

    fn h(&self) -> f32 {
        self.max.y - self.min.y
    }

    fn contains(&self, p: Vec2, margin: f32) -> bool {
        p.x >= self.min.x - margin && p.x <= self.max.x + margin && p.y >= self.min.y - margin && p.y <= self.max.y + margin
    }
}

/// Where everything goes on a canvas of `view` logical pixels, the top `inset` of
/// which lies under the application's buttons: the tabs and the graph at the top, the
/// words beside the graph (under the picture on a narrow canvas), and the scatter
/// with the two histograms beside it, sharing its Y.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    narrow: bool,
    text: f32,
    small: f32,
    tabs: Rect,
    graph: Rect,
    words: Rect,
    /// What the colours of the dots say.
    key: Rect,
    plot: Rect,
    hist: Rect,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::ONE);
        let narrow = view.x < 620.0;
        let (text, small) = if narrow { (11.5, 10.5) } else { (13.0, 11.5) };
        let side = if narrow { 10.0 } else { 18.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let tabs = Rect::new(side, top, view.x - side, top + if narrow { 26.0 } else { 28.0 });
        let graph_top = tabs.max.y + 8.0;
        // the scatter's columns: numbers along its left, the histograms at its right
        let hist_w = if narrow { (view.x * 0.25).clamp(64.0, 120.0) } else { (view.x * 0.2).clamp(140.0, 260.0) };
        let left = side + if narrow { 24.0 } else { 34.0 };
        let bare = view.x < 300.0 || view.y < 300.0 || (view.x < 480.0 && view.y < 560.0);
        let (graph, words, key, main_top, bottom, left) = if bare {
            // a window too small for more than the scatter and its histograms
            let none = Rect::new(0.0, 0.0, 0.0, 0.0);
            (none, none, none, tabs.max.y + 24.0, view.y - 30.0, left)
        } else if view.y < 620.0 && view.x >= 480.0 {
            // a canvas too short for a row above the scatter (a phone held sideways): the
            // graph, the words and the key in a column at its left
            let column = (view.x * 0.27).clamp(190.0, 300.0);
            let head = (view.y * 0.3).clamp(80.0, column / 1.7);
            let graph = Rect::new(side, graph_top, side + head * 1.7, graph_top + head);
            let key = Rect::new(side, view.y - 2.0 * small - 18.0, side + column, view.y - 6.0);
            let words = Rect::new(side, graph.max.y + 10.0, side + column, key.min.y - 8.0);
            (graph, words, key, tabs.max.y + 36.0, view.y - 40.0, side + column + 44.0)
        } else if narrow {
            // a phone: the graph, with the key to the dots' colours beside it; the
            // scatter; and the words under the picture, on as many lines as they take
            let head = (view.y * 0.16).clamp(96.0, 124.0);
            let graph = Rect::new(side, graph_top, side + head * 1.75, graph_top + head);
            let key = Rect::new(graph.max.x + 8.0, graph.min.y + 4.0, view.x - side, graph.min.y + 2.0 * small + 16.0);
            let words_h = 6.0 * (small + 4.0) + 10.0;
            let words = Rect::new(side, view.y - 6.0 - words_h, view.x - side, view.y - 6.0);
            (graph, words, key, graph.max.y + 30.0, words.min.y - 34.0, left)
        } else {
            // the graph at the left of a row above the scatter, the words and the key
            // beside it
            let head = (view.y * 0.17).clamp(118.0, 168.0);
            let graph = Rect::new(side, graph_top, side + head * 1.7, graph_top + head);
            let right = Rect::new(graph.max.x + 24.0, graph_top, view.x - side, graph.max.y);
            let key = Rect::new(right.min.x, right.max.y - small - 2.0, right.max.x, right.max.y);
            (graph, Rect::new(right.min.x, right.min.y, right.max.x, key.min.y - 8.0), key, graph.max.y + 40.0, view.y - 40.0, left)
        };
        let bottom = bottom.max(main_top + 40.0);
        let hist = Rect::new(view.x - side - hist_w, main_top, view.x - side, bottom);
        let plot = Rect::new(left, main_top, hist.min.x - if narrow { 14.0 } else { 28.0 }, bottom);
        Layout { narrow, text, small, tabs, graph, words, key, plot, hist }
    }

    /// The tabs: seeing, doing, and the experiment's switch, left to right.
    fn tab(&self, k: usize) -> Rect {
        let w = if self.narrow { ((self.tabs.w() - 12.0) / 3.0).max(30.0) } else { 112.0 };
        let gap = if k == 2 { 12.0 } else { 0.0 };
        let x = self.tabs.min.x + k as f32 * (w + 4.0) + gap;
        let w = if k == 2 && !self.narrow { 150.0 } else { w };
        Rect::new(x, self.tabs.min.y, x + w, self.tabs.max.y)
    }
}

/// From the scatter's X and Y to the screen and back.
#[derive(Clone, Copy, Debug)]
struct Map {
    plot: Rect,
    /// Lowest and highest X and Y shown.
    frame: [f64; 4],
}

impl Map {
    fn to(&self, x: f64, y: f64) -> Vec2 {
        let [x0, x1, y0, y1] = self.frame;
        let sx = ((x - x0) / (x1 - x0).max(1e-9)) as f32;
        let sy = ((y - y0) / (y1 - y0).max(1e-9)) as f32;
        Vec2::new(self.plot.min.x + sx * self.plot.w(), self.plot.max.y - sy * self.plot.h())
    }

    /// The X at a place across the screen.
    fn x_at(&self, px: f32) -> f64 {
        let [x0, x1, ..] = self.frame;
        x0 + ((px - self.plot.min.x) / self.plot.w()) as f64 * (x1 - x0)
    }

    /// The height on the screen of a value of Y.
    fn y_px(&self, y: f64) -> f32 {
        self.to(self.frame[0], y).y
    }

    fn shows(&self, p: Vec2) -> bool {
        self.plot.contains(p, 0.5)
    }
}

// ----- drawing --------------------------------------------------------------------------

/// The colour of a unit with this Z: dark purple for a low one, pale for a high one.
fn z_color(z: f64) -> Color {
    let t = ((z + 2.0) / 4.0).clamp(0.0, 1.0) as f32;
    Color::hex(0x5A3AA8).mix(Color::hex(0xF0E8FF), t)
}

/// A translucent fill on the screen, on one of the sheets of `Frame::hud_at`.
fn hud_fill(f: &mut Frame, level: usize, min: Vec2, max: Vec2, color: Color) {
    let c = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)].map(|p| f.hud_at(level, p));
    f.dl.tris_alpha.quad(c[0], c[1], c[2], c[3], color);
}

/// A line on the screen behind the dots (the grid): on the back sheet, depth-tested.
fn hud_back_line(f: &mut Frame, a: Vec2, b: Vec2, width: f32, color: Color) {
    let (a, b) = (f.hud_at(0, a), f.hud_at(0, b));
    f.dl.lines.segment(a, b, width, color);
}

/// A dashed line on the screen, on top of everything.
fn hud_dashed(f: &mut Frame, a: Vec2, b: Vec2, width: f32, color: Color) {
    let (a, b) = (f.hud(a), f.hud(b));
    f.dl.lines_with(LineKind::Overlay, LineStyle { dash: 7.0, duty: 0.6, depth_bias: 0.0, ..Default::default() }).segment(a, b, width, color);
}

/// A tab over the picture: true when clicked.
fn hud_tab(f: &mut Frame, r: Rect, text: &str, on: bool, pointer: Option<Vec2>, size: f32) -> bool {
    let over = pointer.is_some_and(|p| r.contains(p, 0.0));
    let fill = match (on, over) {
        (true, _) => Color::hex(0x2B3B60),
        (false, true) => Color::hex(0x2A3349),
        (false, false) => Color::hex(0x161C2B),
    };
    hud_rect(f, r.min, r.max, fill);
    if on {
        hud_line(f, Vec2::new(r.min.x, r.max.y - 1.0), Vec2::new(r.max.x, r.max.y - 1.0), 2.0, pal::LINK);
    }
    f.text_px((r.min + r.max) * 0.5, text, size, if on { pal::TEXT } else { pal::DIM }, Align2::CENTER_CENTER);
    over && f.ptr.clicked
}

/// The words of a text, each with its formulas whole.
fn words_of(text: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let (mut start, mut inside) = (0, false);
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 1,
            b'$' => inside = !inside,
            b' ' if !inside => {
                words.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    words.push(&text[start.min(text.len())..]);
    words
}

/// Text broken into lines no wider than `width` pixels, the first line's top-left at
/// `at`. Returns the height taken.
fn wrap_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> f32 {
    let mut lines: Vec<String> = vec![String::new()];
    for word in words_of(text) {
        let Some(last) = lines.last_mut() else { break };
        if last.is_empty() {
            last.push_str(word);
        } else if f.text_size(&format!("{last} {word}"), size).x > width {
            lines.push(word.to_string());
        } else {
            last.push(' ');
            last.push_str(word);
        }
    }
    let line = size + 4.0;
    for (i, l) in lines.iter().enumerate() {
        f.text_px(at + Vec2::new(0.0, i as f32 * line), l.as_str(), size, color, Align2::LEFT_TOP);
    }
    lines.len() as f32 * line
}

impl DoSee {
    fn map(&self, lay: &Layout) -> Map {
        Map { plot: lay.plot, frame: [self.frame.get(0), self.frame.get(1), self.frame.get(2), self.frame.get(3)] }
    }

    /// The tabs over the picture: seeing, doing, and an experiment in place of nature.
    fn tabs(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let names = if lay.narrow { ["seeing", "doing", "random $X$"] } else { ["seeing", "doing", "deal $X$ at random"] };
        for (k, name) in names.into_iter().enumerate() {
            let on = match k {
                0 => self.mode == Mode::See,
                1 => self.mode == Mode::Do,
                _ => self.model.dealt,
            };
            if hud_tab(f, lay.tab(k), name, on, pointer, lay.small) {
                match k {
                    0 => (self.mode, self.again) = (Mode::See, false),
                    1 => (self.mode, self.again) = (Mode::Do, false),
                    _ => {
                        self.model.dealt = !self.model.dealt;
                        self.refresh();
                    }
                }
                self.moving = true;
            }
        }
    }

    /// The causal graph: Z above, X and Y below; in doing, every arrow into X cut and
    /// a hand on X.
    fn graph(&self, f: &mut Frame, lay: &Layout) {
        let g = lay.graph;
        if g.h() < 40.0 {
            return;
        }
        let size = lay.small;
        let r = (g.h() * 0.17).clamp(12.0, 20.0);
        let low = g.max.y - r - size - 8.0;
        // (room at the left of X for the hand that sets it, and above that for the die)
        let (z, x, y) = (Vec2::new(g.min.x + g.w() * 0.56, g.min.y + r + 2.0), Vec2::new(g.min.x + r + 36.0, low), Vec2::new(g.max.x - r - 4.0, low));
        let die = Vec2::new(g.min.x + r + 2.0, g.min.y + r + 2.0);
        let doing = self.mode == Mode::Do;
        let m = self.model;
        // (an arrow into X is cut by doing; nature's are also cut by an experiment)
        let into_x = |nature: bool| if doing || (nature && m.dealt) { Link::Cut } else { Link::Arrow };
        let mut nodes = vec![Node { at: z, name: "$Z$", color: THIRD }, Node { at: x, name: "$X$", color: CAUSE }, Node { at: y, name: "$Y$", color: EFFECT }];
        let mut links: Vec<(usize, usize, Link, Color, f32)> = Vec::new();
        if m.a != 0.0 {
            links.push((0, 1, into_x(true), THIRD, 1.8));
        }
        if m.c != 0.0 {
            links.push((0, 2, Link::Arrow, THIRD, 1.8));
        }
        if m.b != 0.0 {
            match m.arrow {
                Arrow::Forward => links.push((1, 2, Link::Arrow, TRUTH, 2.2)),
                Arrow::Backward => links.push((2, 1, into_x(true), TRUTH, 2.2)),
            }
        }
        if m.dealt {
            nodes.push(Node { at: die, name: "", color: pal::TEXT });
            links.push((3, 1, into_x(false), pal::TEXT, 1.8));
        }
        hud_graph(f, &nodes, &links, r);
        if m.dealt {
            // a die, its face of three
            let s = r * 0.55;
            hud_outline(f, die - Vec2::splat(s), die + Vec2::splat(s), 1.4, pal::TEXT);
            for k in [-1.0, 0.0, 1.0] {
                let at = f.hud_at(2, die + Vec2::new(k, -k) * s * 0.55);
                f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, ..Default::default() }).push(at, (r * 0.26).max(3.5), pal::TEXT);
            }
        }
        // the strengths beside their arrows: a and c outside the triangle, b inside it
        let centre = (z + x + y) / 3.0;
        // (clear of the stroke across an arrow that is cut, too)
        let cut = |link: Link| if link == Link::Cut { 9.0 } else { 3.0 };
        let label = |f: &mut Frame, from: Vec2, to: Vec2, text: String, color: Color, outside: bool, clear: f32| {
            let mid = (from + to) * 0.5;
            let mut n = Vec2::new(-(to - from).y, (to - from).x).normalize_or_zero();
            if ((mid - centre).dot(n) < 0.0) == outside {
                n = -n;
            }
            let half = f.text_size(&text, size) * 0.5;
            let reach = n.x.abs() * half.x + n.y.abs() * half.y + clear;
            f.text_px(mid + n * reach, text, size, color, Align2::CENTER_CENTER);
        };
        if m.a != 0.0 {
            label(f, z, x, format!("$a = {}$", num(m.a, 2)), THIRD, true, cut(into_x(true)));
        }
        if m.c != 0.0 {
            label(f, z, y, format!("$c = {}$", num(m.c, 2)), THIRD, true, 3.0);
        }
        if m.b != 0.0 {
            let link = if m.arrow == Arrow::Backward { into_x(true) } else { Link::Arrow };
            label(f, x, y, format!("$b = {}$", num(m.b, 2)), TRUTH, false, cut(link));
        }
        // what sets X in doing: a hand
        let under = |p: Vec2| p + Vec2::new(0.0, r + 4.0);
        if doing {
            f.text_px(x - Vec2::new(r + 3.0, 0.0), "☞", r * 1.5, pal::TEXT, Align2::RIGHT_CENTER);
            f.text_px(under(x), format!(r"$\operatorname{{do}}(X = {})$", num(self.x, 2)), size, CAUSE, Align2::CENTER_TOP);
        } else if let Some(names) = self.names.of() {
            f.text_px(under(x), names[1], size, CAUSE, Align2::CENTER_TOP);
        }
        if let Some(names) = self.names.of() {
            f.text_px(under(y), names[2], size, EFFECT, Align2::CENTER_TOP);
            f.text_px(z + Vec2::new(r + 6.0, -r * 0.5), names[0], size, THIRD, Align2::LEFT_CENTER);
        }
    }

    /// The scatter: the units, the window seeing looks through, the curves of the two
    /// means, and the handle for x.
    fn scatter(&self, f: &mut Frame, lay: &Layout, map: &Map) {
        let p = lay.plot;
        hud_panel(f, p.min - Vec2::splat(4.0), p.max + Vec2::splat(4.0));
        // the grid, with its numbers
        let [x0, x1, y0, y1] = map.frame;
        for (lo, hi, along_x) in [(x0, x1, true), (y0, y1, false)] {
            let span_px = if along_x { p.w() } else { p.h() } as f64;
            let apart = if lay.narrow { 52.0 } else { 80.0 };
            let step = nice_step((apart * (hi - lo) / span_px.max(1.0)) as f32) as f64;
            let first = (lo / step).ceil() as i64;
            let last = (hi / step).floor() as i64;
            for k in first..=last.min(first + 60) {
                let v = k as f64 * step;
                let (a, b) = if along_x { (map.to(v, y0), map.to(v, y1)) } else { (map.to(x0, v), map.to(x1, v)) };
                hud_back_line(f, a, b, 1.0, if k == 0 { pal::FAINT.a(0.9) } else { pal::FAINT.a(0.35) });
                let text = fmt_tick(v as f32, step as f32);
                if along_x {
                    f.text_px(a + Vec2::new(0.0, 6.0), text, lay.small, pal::DIM, Align2::CENTER_TOP);
                } else {
                    f.text_px(a - Vec2::new(6.0, 0.0), text, lay.small, pal::DIM, Align2::RIGHT_CENTER);
                }
            }
        }
        f.text_px(Vec2::new(p.max.x, p.max.y + 22.0), self.name(1), lay.text, CAUSE, Align2::RIGHT_TOP);
        f.text_px(Vec2::new(p.min.x, p.min.y - 8.0), self.name(2), lay.text, EFFECT, Align2::LEFT_BOTTOM);

        // the window seeing looks through (faint once the units have been moved)
        let t = smooth(self.travel);
        let (at, window) = (self.at.get(0), self.at.get(1));
        let (wl, wr) = (map.to(at - window, y0).x.max(p.min.x), map.to(at + window, y0).x.min(p.max.x));
        if wr > wl {
            // (blended in linear light, where a little goes a long way on a dark ground)
            hud_fill(f, 0, Vec2::new(wl, p.min.y), Vec2::new(wr, p.max.y), NAIVE.a(0.012 * (1.0 - 0.6 * t)));
            for e in [wl, wr] {
                hud_back_line(f, Vec2::new(e, p.min.y), Vec2::new(e, p.max.y), 1.0, NAIVE.a(0.5 * (1.0 - 0.6 * t)));
            }
        }

        // the units: where they came (faint, once they have moved), a ring round each one
        // seeing picks out, and each where it is now: on its way to x along its slope
        let n = self.xs.shown.len().min(self.ys.shown.len()).min(self.pop.len());
        let big = if lay.narrow { 4.2 } else { 5.2 };
        let slope = self.slope.get(0);
        // (the trail each unit leaves is brightest on the way, and stays faintly once it
        // is there: every trail has the same slope)
        let trail = 0.06 * t + 0.3 * 4.0 * t * (1.0 - t);
        // (rings and the faint places the units came from have no dark rim: a rim is
        // drawn whole however faint the disc, and would ring every unit; the units
        // themselves are a batch of their own, made after these and so drawn over them)
        let plain = PointStyle { outline_px: 0.0, ..Default::default() };
        let rimmed = PointStyle { outline_px: 1.25, ..Default::default() };
        let _ = f.dl.points_with(PointKind::Disc, plain);
        for i in 0..n {
            let (ux, uy) = (self.xs.get(i), self.ys.get(i));
            let home = map.to(ux, uy);
            let now = map.to(ux + t as f64 * (at - ux), uy + t as f64 * slope * (at - ux));
            let c = z_color(self.pop[i].z);
            let lit = self.lit.get(i).copied().unwrap_or(0.0);
            let (shows_home, shows_now) = (map.shows(home), map.shows(now));
            // (every unit draws the same things in every frame, hidden where it is off the
            // picture: the motion of each can then be followed from frame to frame)
            let (back, front) = (f.hud_at(1, home), f.hud_at(2, now));
            let marks = f.dl.points_with(PointKind::Disc, plain);
            marks.push(back, big + if lay.narrow { 3.5 } else { 5.0 }, NAIVE.a(if shows_home { 0.9 * lit * (1.0 - 0.8 * t) } else { 0.0 }));
            marks.push(back, big * 0.75, c.a(if shows_home { 0.35 * t } else { 0.0 }));
            f.dl.lines.segment(back, f.hud_at(1, now), 1.0, c.a(if shows_home && shows_now { trail } else { 0.0 }));
            let alpha = 0.55 + 0.45 * lit.max(t);
            // (a little larger once set to x, where they stand in one column)
            f.dl.points_with(PointKind::Disc, rimmed).push(front, big * (1.0 + 0.45 * t), c.a(if shows_now { alpha } else { 0.0 }));
        }

        // the two curves: the mean of Y among the units seen at each x, and over every
        // unit with X set to each x
        // (the same number of points in every frame, drawn clear where too few are seen
        // to say: so each point of the curve can be followed from frame to frame, and
        // the curve fades out where it ends)
        let count = STEPS + 1;
        let points: Vec<(Vec2, f32)> = (0..count)
            .map(|k| {
                let q = map.to(self.curve.get(k), self.curve.get(count + k));
                let shown = if map.shows(q) { self.curve.get(2 * count + k) as f32 } else { 0.0 };
                (q.clamp(p.min, p.max), shown.clamp(0.0, 1.0))
            })
            .collect();
        let placed: Vec<(Vec3, f32)> = points.iter().map(|&(q, a)| (f.hud(q), a)).collect();
        let pink = f.dl.lines_with(LineKind::Overlay, LineStyle { depth_bias: 0.0, ..Default::default() });
        pink.brk();
        for (at, alpha) in placed {
            pink.push(at, 2.6, NAIVE.a(alpha));
        }
        pink.brk();
        let curve_end = points.iter().rev().find(|(_, a)| *a > 0.5).map(|(q, _)| *q);
        let done_at = |x: f64| self.done_shown(x);
        let (a, b) = clip_line(map, x0, x1, done_at);
        if let (Some(a), Some(b)) = (a, b) {
            hud_line(f, a, b, 2.6, TRUTH);
        }
        // their names, at their right-hand ends, kept apart
        let mut ends: Vec<(Vec2, &str, Color, u32)> = Vec::new();
        if let Some(q) = curve_end {
            ends.push((q, "seen", NAIVE, 0));
        }
        if let Some(b) = b {
            ends.push((b, "done", TRUTH, 1));
        }
        if ends.len() == 2 && (ends[0].0.y - ends[1].0.y).abs() < lay.small + 6.0 {
            let mid = (ends[0].0.y + ends[1].0.y) * 0.5;
            let up = if ends[0].0.y <= ends[1].0.y { -1.0 } else { 1.0 };
            ends[0].0.y = mid + up * (lay.small * 0.5 + 3.0);
            ends[1].0.y = mid - up * (lay.small * 0.5 + 3.0);
        }
        // (gliding to a new end, not jumping there)
        let corner = Vec2::new(p.max.x, p.min.y);
        for (at, text, color, number) in ends {
            let half = f.text_size(text, lay.small) * 0.5;
            let centre = at + Vec2::new(-8.0 - half.x, -5.0 - half.y);
            let centre = centre.max(p.min + half).min(p.max - half);
            f.text_placed("end", number, corner, centre - corner, text, lay.small, color);
        }

        // x: the line through the scatter, the handle on the axis, and the two means there
        let top = map.to(at, y1);
        let foot = map.to(at, y0);
        // (behind the units, which gather on it in doing)
        hud_back_line(f, top, foot, 1.2, CAUSE.a(0.6));
        // (both marks drawn in every frame, clear where there is nothing to mark)
        let plain = PointStyle { outline_px: 0.0, ..Default::default() };
        let mid = lay.hist.min.x + lay.hist.w() * 0.5;
        let seen_q = map.to(at, self.marks.get(0));
        let done_q = map.to(at, done_at(at));
        for (q, alpha, color, to) in [(seen_q, self.marks.get(1) as f32, NAIVE, mid), (done_q, 1.0, TRUTH, lay.hist.max.x)] {
            let alpha = if map.shows(q) { alpha.clamp(0.0, 1.0) } else { 0.0 };
            let at = q.clamp(p.min, p.max);
            let mark = f.hud_at(2, at);
            f.dl.points_with(PointKind::Overlay, plain).push(mark, 9.0, color.a(alpha));
            hud_dashed(f, at, Vec2::new(to, at.y), 1.2, color.a(0.7 * alpha));
        }
        f.caption_px(top + Vec2::new(0.0, 4.0), format!("$x = {}$", num(self.x, 2)), lay.small, CAUSE, Align2::CENTER_TOP);
        f.dl.overlay_discs.push(f.hud_at(2, foot), if self.holding { 17.0 } else { 13.0 }, CAUSE);
        if self.holding {
            f.dl.glows.push(f.hud_at(2, foot), 40.0, CAUSE.a(0.55));
        }
    }

    /// The two histograms back to back, on the scatter's scale of Y: the seen units'
    /// to the left of the middle, every unit's with X set to x to the right.
    fn histograms(&self, f: &mut Frame, lay: &Layout, map: &Map, look: &Look) {
        let h = lay.hist;
        hud_panel(f, h.min - Vec2::splat(4.0), h.max + Vec2::splat(4.0));
        let mid = h.min.x + h.w() * 0.5;
        let half = h.w() * 0.5 - 4.0;
        // (a density of 1 reaches the edge, or the highest bar of doing if that is higher)
        let top = (0..BINS).map(|k| self.bars.get(BINS + k)).fold(1.0f64, f64::max) * 1.05;
        let scale = half as f64 / top;
        for k in 0..BINS {
            let lo = -SPAN + k as f64 * BIN;
            let (a, b) = (map.y_px(lo + BIN), map.y_px(lo));
            if b < h.min.y || a > h.max.y {
                continue;
            }
            let (a, b) = (a.max(h.min.y) + 0.5, b.min(h.max.y) - 0.5);
            let seen = (self.bars.get(k) * scale).min(half as f64) as f32;
            let done = (self.bars.get(BINS + k) * scale).min(half as f64) as f32;
            if seen > 0.2 {
                hud_fill(f, 1, Vec2::new(mid - seen, a), Vec2::new(mid, b), NAIVE.a(0.75));
            }
            if done > 0.2 {
                hud_fill(f, 1, Vec2::new(mid, a), Vec2::new(mid + done, b), TRUTH.a(0.75));
            }
        }
        hud_line(f, Vec2::new(mid, h.min.y), Vec2::new(mid, h.max.y), 1.0, pal::DIM.a(0.8));
        // what each half is, and the mean of each across it
        f.text_px(Vec2::new(mid - 6.0, h.min.y - 8.0), "seen", lay.text, NAIVE, Align2::RIGHT_BOTTOM);
        f.text_px(Vec2::new(mid + 6.0, h.min.y - 8.0), "done", lay.text, TRUTH, Align2::LEFT_BOTTOM);
        let foot = h.max.y + 6.0;
        let seen_count = if look.seen == 1 { "1 unit".to_string() } else { format!("{} units", look.seen) };
        f.text_px(Vec2::new(mid - 6.0, foot), seen_count, lay.small, NAIVE, Align2::RIGHT_TOP);
        f.text_px(Vec2::new(mid + 6.0, foot), format!("all {}", look.units), lay.small, TRUTH, Align2::LEFT_TOP);
        // (the means as drawn on the scatter, gliding as they do; each line drawn in every
        // frame, clear where there is nothing to draw)
        let seen_y = map.y_px(self.marks.get(0)).max(h.min.y).min(h.max.y);
        let seen_alpha = if h.contains(Vec2::new(mid, map.y_px(self.marks.get(0))), 0.0) { self.marks.get(1) as f32 } else { 0.0 };
        hud_line(f, Vec2::new(h.min.x, seen_y), Vec2::new(mid, seen_y), 2.0, NAIVE.a(seen_alpha.clamp(0.0, 1.0)));
        if look.seen == 0 {
            f.caption_px(Vec2::new(mid - 4.0, h.min.y + h.h() * 0.5), "nobody\nseen here", lay.small, NAIVE, Align2::RIGHT_CENTER);
        }
        let done = map.y_px(self.done_shown(self.at.get(0)));
        let done_alpha = if h.contains(Vec2::new(mid, done), 0.0) { 1.0 } else { 0.0 };
        let done = done.max(h.min.y).min(h.max.y);
        hud_line(f, Vec2::new(mid, done), Vec2::new(h.max.x, done), 2.0, TRUTH.a(done_alpha));
    }

    /// The numbers, in words, beside the graph (under the picture on a phone).
    fn words(&self, f: &mut Frame, lay: &Layout, look: &Look) {
        let r = lay.words;
        if r.w() < 120.0 {
            return;
        }
        let size = if lay.narrow { lay.small } else { lay.text };
        let mut y = r.min.y + 2.0;
        for (text, color) in self.sentences(look) {
            if y + size > r.max.y + 2.0 {
                break;
            }
            y += wrap_px(f, Vec2::new(r.min.x, y), r.w(), &text, size, color) + 3.0;
        }
    }

    /// What the numbers say, in three sentences: what is seen, what is done, and the
    /// two slopes.
    fn sentences(&self, look: &Look) -> [(String, Color); 3] {
        let (x, w) = (num(self.x, 2), num(self.window, 2));
        let (y, z) = (num(look.seen_y, 2), num(look.seen_z, 2));
        let seen = match look.seen {
            0 => format!(r"Seen: no unit has $X$ within ${w}$ of ${x}$, so seeing has nothing to say here."),
            1 => format!(r"Seen: only 1 unit has $X$ within ${w}$ of ${x}$, too few to say much: its $Y$ is ${y}$ (its $Z$, ${z}$)."),
            n if n < HANDFUL => format!(r"Seen: only {n} units have $X$ within ${w}$ of ${x}$, too few to say much: their mean $Y$ is ${y}$ (mean $Z = {z}$)."),
            n => format!(r"Seen: the {n} units whose $X$ happens to be within ${w}$ of ${x}$ have mean $Y = {y}$ (and mean $Z = {z}$)."),
        };
        let done = format!(r"Done: all {} units, with $X$ set to ${x}$, have mean $Y = {}$ (and mean $Z = {}$).", look.units, num(look.done_y, 2), num(look.done_z, 2));
        let slopes = format!(r"Per unit of $X$, $Y$ rises by ${}$ as seen (the least-squares slope), and by ${}$ as done: the average causal effect.", num(self.trend(), 2), num(self.effect(), 2));
        [(seen, NAIVE), (done, TRUTH), (slopes, pal::DIM)]
    }

    /// What the colour of a dot says: its Z, on a strip from low to high.
    fn z_key(&self, f: &mut Frame, lay: &Layout) {
        let r = lay.key;
        if r.w() < 100.0 {
            return;
        }
        let size = lay.small;
        // (the strip beside the words where there is room for both on one line, under them otherwise)
        let stacked = r.h() > 2.0 * size + 8.0;
        let title = if lay.narrow { "dots: units, by their $Z$" } else { "each dot a unit, coloured by its $Z$:" };
        let at = Vec2::new(r.min.x, r.min.y + size * 0.5 + 1.0);
        f.text_px(at, title, size, pal::DIM, Align2::LEFT_CENTER);
        let strip = if stacked { Vec2::new(r.min.x + 22.0, r.min.y + size + 10.0) } else { Vec2::new(at.x + f.text_size(title, size).x + 34.0, r.min.y + 3.0) };
        let w = (r.max.x - strip.x - 26.0).clamp(40.0, 150.0);
        let pieces = 24;
        for k in 0..pieces {
            let (a, b) = (k as f32 / pieces as f32, (k + 1) as f32 / pieces as f32);
            hud_rect(f, strip + Vec2::new(w * a, 0.0), strip + Vec2::new(w * b + 0.5, size - 2.0), z_color(-2.0 + 4.0 * (a + b) as f64 * 0.5));
        }
        let mid = strip.y + (size - 2.0) * 0.5;
        f.text_px(Vec2::new(strip.x - 5.0, mid), "−2", size, pal::DIM, Align2::RIGHT_CENTER);
        f.text_px(Vec2::new(strip.x + w + 5.0, mid), "2", size, pal::DIM, Align2::LEFT_CENTER);
    }
}

/// The part of the line y = `at(x)`, for x from `x0` to `x1`, that the scatter shows:
/// its two ends on the screen (none if it is not on the picture).
fn clip_line(map: &Map, x0: f64, x1: f64, at: impl Fn(f64) -> f64) -> (Option<Vec2>, Option<Vec2>) {
    let [_, _, y0, y1] = map.frame;
    let (a, b) = (at(x0), at(x1));
    // the stretch of x over which the line is between y0 and y1
    let (mut lo, mut hi) = (x0, x1);
    if (b - a).abs() > 1e-12 {
        let cross = |y: f64| x0 + (y - a) / (b - a) * (x1 - x0);
        let (p, q) = (cross(y0), cross(y1));
        lo = lo.max(p.min(q));
        hi = hi.min(p.max(q));
    } else if a < y0 || a > y1 {
        return (None, None);
    }
    // (and nothing where the numbers are not numbers at all)
    if lo.is_nan() || hi.is_nan() || lo >= hi {
        return (None, None);
    }
    (Some(map.to(lo, at(lo))), Some(map.to(hi, at(hi))))
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The rules each unit is made by, as the panel writes them (the first is replaced by
/// dealing in an experiment).
fn rules(model: &Model) -> [&'static str; 2] {
    const DEALT_X: &str = r"X \text{ dealt at random, from } {-2.5} \text{ to } 2.5";
    match (model.arrow, model.dealt) {
        (Arrow::Forward, false) => [r"X = aZ + 0.6\,N_X", r"Y = bX + cZ + 0.5\,N_Y"],
        (Arrow::Forward, true) => [DEALT_X, r"Y = bX + cZ + 0.5\,N_Y"],
        (Arrow::Backward, false) => [r"Y = cZ + 0.5\,N_Y", r"X = bY + aZ + 0.6\,N_X"],
        (Arrow::Backward, true) => [r"Y = cZ + 0.5\,N_Y", DEALT_X],
    }
}

impl Scene for DoSee {
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
        Some("drag across the scatter: choose $x$    tabs over the picture: seeing or doing, and an experiment")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Two ways to ask what $X$ does to $Y$. Seeing: pick out the units that happen to have $X$ near $x$, and look at their $Y$. Doing: set $X$ to $x$ for every unit, whatever it would have been, and look at $Y$ then. If something else pushes both $X$ and $Y$, as $Z$ does here, the answers differ: here the units seen with a high $X$ are mostly those with a high $Z$, and $Z$ raises $Y$ by itself. Each unit is made by these rules:");
        for rule in rules(&self.model) {
            ui::equation(ui, rule);
        }
        ui::note(ui, r"with its own $Z$, $N_X$ and $N_Y$, each drawn from a bell curve (mean $0$, standard deviation $1$). Doing replaces the rule for $X$ by $X = x$: it cuts every arrow into $X$ and leaves the rest alone, each unit's $Z$ and noise included. Drag across the scatter to choose $x$.");

        ui::section(ui, "Things to try");
        for (label, look_for, setup) in TRY {
            if ui::experiment(ui, label, look_for) {
                self.apply(setup);
            }
        }

        ui::section(ui, "Seeing or doing");
        let mut mode = (self.mode == Mode::Do) as usize;
        if ui::choice(ui, &["seeing", "doing"], &mut mode) {
            (self.mode, self.again) = (if mode == 1 { Mode::Do } else { Mode::See }, false);
            self.moving = true;
        }
        let mut x = self.x as f32;
        let reach = self.goal.frame[1] as f32;
        if ui::slider(ui, "$x$", &mut x, -reach..=reach) {
            self.x = x as f64;
            self.moving = true;
        }
        let mut window = self.window as f32;
        if ui::slider(ui, "window", &mut window, 0.05..=1.0) {
            self.window = window as f64;
            self.moving = true;
        }
        ui::note(ui, r"Seeing picks out the units whose $X$ is within this of $x$. Narrower is closer to $X = x$ exactly, but catches fewer units.");
        let mut dealt = self.model.dealt as usize;
        if ui::choice(ui, &["$X$ as it comes", "$X$ dealt at random"], &mut dealt) {
            self.model.dealt = dealt == 1;
            self.refresh();
        }
        ui::note(ui, r"An experiment: each unit's $X$ dealt at random, evenly from $-2.5$ to $2.5$ (as if by a die), in place of the rule for $X$.");

        ui::section(ui, "The arrows");
        let mut arrow = (self.model.arrow == Arrow::Backward) as usize;
        if ui::choice(ui, &[r"$X \to Y$", r"$Y \to X$"], &mut arrow) {
            self.model.arrow = if arrow == 1 { Arrow::Backward } else { Arrow::Forward };
            self.names = Names::Plain;
            self.refresh();
        }
        let mut changed = false;
        for (name, v) in [(r"$a$: $Z \to X$", &mut self.model.a), (r"$b$: between $X$ and $Y$", &mut self.model.b), (r"$c$: $Z \to Y$", &mut self.model.c)] {
            let mut s = *v as f32;
            if ui::slider(ui, name, &mut s, -1.5..=1.5) {
                *v = s as f64;
                changed = true;
            }
        }
        ui::note(ui, "How strongly each arrow pushes, as in the rules above: a strength of 0 is no arrow at all.");
        let mut units = self.units;
        if ui::slider_int_log(ui, "units", &mut units, 10..=MOST) {
            self.units = units;
            changed = true;
        }
        if ui.button("new units").clicked() {
            self.seed = self.seed.wrapping_add(1);
            self.pop = population(self.seed);
            changed = true;
        }
        ui::note(ui, r"New units: every unit's $Z$ and noise drawn again.");
        if changed {
            self.refresh();
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (z_color(1.2), r"dots: one unit each, paler the higher its $Z$; in doing, faint where each came from, with a faint trail"),
                (NAIVE, r"pink: seeing. The window round $x$, the units in it (ringed), the mean $Y$ of those seen at each $x$, and the histogram of their $Y$"),
                (TRUTH, r"green: doing. The mean $Y$ with $X$ set to each $x$ for every unit, the histogram of every unit's $Y$ with $X$ set to $x$, and the arrow between $X$ and $Y$"),
                (CAUSE, r"cyan: $X$, and $x$ (drag it); the hand at $X$: $X$ set by hand"),
                (THIRD, r"purple: $Z$ and its arrows. Any arrow drawn dashed and struck through has been cut"),
                (pal::TEXT, r"white ring with a die (in an experiment): chance, dealing $X$"),
            ],
        );
        ui::note(ui, "Each histogram is scaled to the same area, so that a group of 80 can be set against one of 1000 (a bar too long for its half is cut off at the edge).");
        ui::section(ui, "Words");
        ui::term(ui, "Unit:", "one of the things measured: a patient, a town, a day. Each has its own values of the variables.");
        ui::term(ui, "Seeing:", r"the distribution of $Y$ among the units with $X = x$, written $P(Y \mid X = x)$. Here, those with $X$ within the window of $x$, since hardly any has $X$ exactly $x$.");
        ui::term(ui, "Doing, an intervention:", r"setting $X$ to $x$ by hand for every unit, written $\operatorname{do}(X = x)$: every arrow into $X$ is cut, and nothing else changes. The distribution of $Y$ then is $P(Y \mid \operatorname{do}(X = x))$.");
        ui::term(ui, "Confounder:", r"a common cause of $X$ and $Y$, like $Z$ here. It makes seeing differ from doing.");
        ui::term(ui, "Randomised experiment:", r"$X$ dealt by chance. Then nothing else can be a cause of $X$, so what is seen in the results is what doing gives, up to the luck of the draw.");
        ui::term(ui, "Average causal effect:", r"how much the mean of $Y$ under $\operatorname{do}(X = x)$ changes as $x$ goes up by one: the slope of the green line, $b$ when $X$ causes $Y$ and $0$ when it does not.");
        ui::term(ui, "Least-squares slope:", r"the slope of the straight line that fits the scatter best: how much $Y$ rises per unit of $X$ among the units as they come.");

        ui::section(ui, "Numbers");
        let look = self.look();
        if look.seen > 0 {
            ui::readout_colored(ui, r"mean $Y$ seen near $x$", format!("${}$", num(look.seen_y, 3)), NAIVE);
            ui::readout_colored(ui, r"mean $Z$ seen near $x$", format!("${}$", num(look.seen_z, 3)), NAIVE);
        } else {
            ui::readout_colored(ui, r"mean $Y$ seen near $x$", "nobody seen", NAIVE);
        }
        ui::readout_colored(ui, "units seen", look.seen.to_string(), NAIVE);
        ui::readout_colored(ui, r"mean $Y$ with $X$ set to $x$", format!("${}$", num(look.done_y, 3)), TRUTH);
        ui::readout_colored(ui, r"mean $Z$ then", format!("${}$", num(look.done_z, 3)), TRUTH);
        ui::readout(ui, "slope of what is seen", format!("${}$", num(self.trend(), 3)));
        ui::readout_colored(ui, "average causal effect", format!("${}$", num(self.effect(), 3)), TRUTH);
        ui::see_also(
            ui,
            &[
                ("confound", "why the seen slope is off, and by how much"),
                ("adjust", "getting doing's answer from data only seen"),
                ("counterfactual", "what would have happened to one unit"),
                ("regress", "least-squares lines"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        let pointer = f.ptr.pos.map(|p| p / f.px);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone:
        // nothing legible fits, and the picture is left as it is for when they close)
        let covered = view.x < 160.0 || view.y < 120.0;
        if !covered {
            // the tabs are for clicking, and the scatter for choosing x: not for moving the view
            let on_tabs = pointer.is_some_and(|p| lay.tabs.contains(p, 2.0));
            let on_plot = pointer.is_some_and(|p| lay.plot.contains(p, 10.0));
            if on_tabs || on_plot || self.holding {
                f.handles.capture();
                f.handles.claim_clicks();
            }
            if f.ptr.pressed && on_plot {
                self.holding = true;
            }
            if !f.ptr.down {
                self.holding = false;
            }
            if self.holding {
                if let Some(p) = pointer {
                    let reach = self.goal.frame[1];
                    let x = self.map(&lay).x_at(p.x).max(-reach).min(reach);
                    if x != self.x && x.is_finite() {
                        self.x = x;
                        self.moving = true;
                    }
                }
            }
            self.tabs(f, &lay, pointer);
        }
        self.advance(f.dt, f.real_dt);
        if covered {
            return;
        }
        let map = self.map(&lay);
        let look = self.look();
        self.graph(f, &lay);
        self.scatter(f, &lay, &map);
        self.histograms(f, &lay, &map, &look);
        self.words(f, &lay, &look);
        self.z_key(f, &lay);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scene set up as experiment `k`, at rest.
    fn run(k: usize) -> DoSee {
        let mut s = DoSee::new();
        s.apply(TRY[k].2);
        s.settle();
        s
    }

    /// What a look says, as the captions write it: units seen, their mean Y and Z,
    /// and the mean Y and Z with X set (to two places).
    fn quoted(l: &Look) -> (usize, String, String, String, String) {
        (l.seen, num(l.seen_y, 2), num(l.seen_z, 2), num(l.done_y, 2), num(l.done_z, 2))
    }

    /// The captions of the "things to try", held to account on the units they set up.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let s = |v: &str| v.to_string();
        // seeing is not doing: 1.84 against 0.77; their Z 1.07 against 0.02; each unit
        // moves along a line of slope b = 0.5
        let first = run(0);
        assert_eq!(first.mode, Mode::Do);
        assert_eq!(quoted(&first.look()), (76, s("1.84"), s("1.07"), s("0.77"), s("0.02")));
        assert_eq!(first.model.response(), 0.5);
        // no confounder: their Z is still 1.07; seeing 0.77, doing 0.75
        let second = run(1);
        assert_eq!(quoted(&second.look()), (76, s("0.77"), s("1.07"), s("0.75"), s("0.02")));
        // a randomised experiment: 89 units dealt an X near 1.5, 0.74 against 0.77;
        // slopes 0.52 and 0.50
        let third = run(2);
        assert!(third.model.dealt && third.mode == Mode::See);
        let l = third.look();
        assert_eq!((l.seen, num(l.seen_y, 2), num(l.done_y, 2)), (89, s("0.74"), s("0.77")));
        assert_eq!((num(third.trend(), 2), num(third.effect(), 2)), (s("0.52"), s("0.50")));
        // the arrow the other way: a trend of 0.77, 1.16 seen; nothing done, flat at 0.02
        let fourth = run(3);
        let l = fourth.look();
        assert_eq!((num(fourth.trend(), 2), num(l.seen_y, 2), num(l.done_y, 2)), (s("0.77"), s("1.16"), s("0.02")));
        assert_eq!(fourth.effect(), 0.0);
        for k in 0..fourth.units {
            let u = &fourth.pop[k];
            assert_eq!(fourth.model.done(u, -2.0), fourth.model.natural(u).1, "unit {k} keeps its Y");
        }
        // a treatment that seems to harm: seen, recovery falls by 0.60 per unit of dose;
        // each unit's own line, and the green, rise by 0.50
        let fifth = run(4);
        assert_eq!(fifth.names.of(), Some(["illness", "dose", "recovery"]));
        assert!(fifth.model.a > 0.0 && fifth.model.c < 0.0, "more to the sicker, and the sicker recover less");
        assert_eq!((num(fifth.trend(), 2), num(fifth.effect(), 2), fifth.model.response()), (s("-0.60").replace('-', "−"), s("0.50"), 0.5));
        // (each of these opens on the same units, and in the mode its caption describes)
        for (k, mode) in [Mode::Do, Mode::Do, Mode::See, Mode::Do, Mode::Do].into_iter().enumerate() {
            assert_eq!((run(k).mode, run(k).units, run(k).seed), (mode, UNITS, SEED), "experiment {k}");
        }
    }

    /// Doing moves each unit along a line of slope b (none, if X does not cause Y),
    /// keeping its own Z and noise: setting X to what it was anyway changes nothing, and
    /// the mean of Y over every unit set to x rises by b for each step of x.
    #[test]
    fn doing_moves_each_unit_along_its_own_line() {
        for (_, _, setup) in TRY {
            for dealt in [false, true] {
                let m = Model { dealt, ..setup.model };
                let pop = population(5);
                for u in &pop[..300] {
                    let (x, y) = m.natural(u);
                    assert!((m.done(u, x) - y).abs() < 1e-12, "{m:?}: X set to what it was");
                    for to in [-3.0, 0.0, 2.5] {
                        assert!((m.done(u, to) - (y + m.response() * (to - x))).abs() < 1e-12, "{m:?}");
                    }
                }
                assert!((effect(&m, &pop, 1000) - m.response()).abs() < 1e-9);
            }
        }
    }

    /// What the opening note says of the confounded units, in general: the units seen
    /// near x have a mean Y of b x + c E[Z | X = x], with E[Z | X = x] = a x / (a² + 0.6²);
    /// with X set to x it is b x. Over a large population, at several x.
    #[test]
    fn seeing_adds_what_z_brings_with_it() {
        let pop = population(41);
        let m = TRY[0].2.model;
        let pull = m.a / (m.a * m.a + NOISE_X * NOISE_X);
        for x in [-1.0, 0.0, 0.8, 1.5] {
            let l = look(&m, &pop, MOST, x, 0.25);
            let (seen, done) = (m.b * x + m.c * pull * x, m.b * x);
            // (a few hundred units seen, with Y spread by about 0.7: their mean within 0.12)
            assert!((l.seen_y - seen).abs() < 0.12 && (l.done_y - done).abs() < 0.06, "at {x}: seen {} (theory {seen}), done {} (theory {done})", l.seen_y, l.done_y);
        }
        // and the least-squares slope of what is seen is b + c a / (a² + 0.6²)
        assert!((trend(&m, &pop, MOST) - (m.b + m.c * pull)).abs() < 0.05, "{}", trend(&m, &pop, MOST));
    }

    /// Without a confounder, and in a randomised experiment, seeing agrees with doing
    /// up to the luck of the sample: over many populations the gap averages nothing, and
    /// is as large as a sample of its size makes it. With the confounder it does not.
    #[test]
    fn an_experiment_makes_seeing_agree_with_doing() {
        let confounded = TRY[0].2.model;
        for model in [TRY[1].2.model, Model { dealt: true, ..confounded }] {
            let (mut gaps, mut z) = (Vec::new(), Vec::new());
            for seed in 100..160u64 {
                let pop = population(seed);
                let l = look(&model, &pop, UNITS, 1.5, WINDOW);
                // the spread of the seen mean: Y's spread among those seen, over their number
                let seen: Vec<f64> = pop[..UNITS].iter().map(|u| model.natural(u)).filter(|(x, _)| (x - 1.5).abs() <= WINDOW).map(|(_, y)| y).collect();
                let sd = (crate::math::causal::variance(&seen) / seen.len().max(1) as f64).sqrt();
                gaps.push(l.seen_y - l.done_y);
                z.push((l.seen_y - l.done_y) / sd);
            }
            let within = z.iter().filter(|v| v.abs() < 2.5).count();
            assert!(mean(&gaps).abs() < 0.05 && within >= 55, "{model:?}: mean gap {}, {within} of 60 within 2.5 spreads", mean(&gaps));
            let confounded_gap = look(&confounded, &population(100), UNITS, 1.5, WINDOW);
            assert!(confounded_gap.seen_y - confounded_gap.done_y > 0.8);
        }
    }

    /// The arrow the other way: seeing shows a trend and doing none, and the harmful
    /// treatment looks harmful and helps, on every population, not only the one shown.
    #[test]
    fn the_reversals_hold_on_every_population() {
        for seed in 100..130u64 {
            let pop = population(seed);
            let (back, harm) = (TRY[3].2.model, TRY[4].2.model);
            assert!(trend(&back, &pop, UNITS) > 0.65 && effect(&back, &pop, UNITS) == 0.0, "seed {seed}");
            assert!(trend(&harm, &pop, UNITS) < -0.45 && (effect(&harm, &pop, UNITS) - 0.5).abs() < 1e-9, "seed {seed}");
        }
    }

    /// The picture's account of the numbers: at rest, the pink curve at x is the mean
    /// of the units seen there, the green line at x the mean with X set to x, and each
    /// histogram has area 1 (all the values are on it) and its mean where it should be.
    #[test]
    fn the_picture_agrees_with_the_numbers() {
        for k in 0..TRY.len() {
            let s = run(k);
            let l = s.look();
            let n = s.units;
            let seeing = Seeing::of(&s.xs.shown[..n], &s.ys.shown[..n]);
            let (count, y) = seeing.at(s.x, s.window).expect("units are seen");
            assert!(count == l.seen && (y - l.seen_y).abs() < 1e-9, "experiment {k}");
            let done = mean(&(0..n).map(|i| s.done_y(i)).collect::<Vec<f64>>());
            assert!((done - l.done_y).abs() < 1e-9, "experiment {k}: {done} against {}", l.done_y);
            for (half, want) in [(&s.bars.shown[..BINS], l.seen_y), (&s.bars.shown[BINS..], l.done_y)] {
                let area: f64 = half.iter().sum::<f64>() * BIN;
                let centre: f64 = half.iter().enumerate().map(|(j, d)| d * BIN * (-SPAN + (j as f64 + 0.5) * BIN)).sum::<f64>();
                assert!((area - 1.0).abs() < 1e-9 && (centre - want).abs() < 0.5 * BIN, "experiment {k}: area {area}, centre {centre} against {want}");
            }
            // every unit is on the picture's scale of Y, at x and as it came
            let [x0, x1, y0, y1] = s.goal.frame;
            let on = (0..n).filter(|&i| (y0..=y1).contains(&s.ys.get(i)) && (y0..=y1).contains(&s.done_y(i)) && (x0..=x1).contains(&s.xs.get(i))).count();
            assert!(on as f64 >= 0.99 * n as f64, "experiment {k}: {on} of {n} on the picture");
        }
    }

    /// The opening note's rules are the ones the units are made by.
    #[test]
    fn the_rules_written_are_the_rules_used() {
        let forward = Model { arrow: Arrow::Forward, a: 1.0, b: 1.0, c: 1.0, dealt: false };
        assert_eq!(rules(&forward), [format!(r"X = aZ + {}\,N_X", NOISE_X), format!(r"Y = bX + cZ + {}\,N_Y", NOISE_Y)]);
        assert_eq!(rules(&Model { arrow: Arrow::Backward, ..forward }), [format!(r"Y = cZ + {}\,N_Y", NOISE_Y), format!(r"X = bY + aZ + {}\,N_X", NOISE_X)]);
        for arrow in [Arrow::Forward, Arrow::Backward] {
            let dealt = rules(&Model { arrow, dealt: true, ..forward });
            assert!(dealt.iter().any(|r| r.contains(&format!("from }} {{-{DEALT}}} \\text{{ to }} {DEALT}"))), "{dealt:?}");
        }
        let u = Unit { z: 0.7, nx: -1.1, ny: 0.4, deal: 0.3 };
        let m = Model { arrow: Arrow::Forward, a: 0.9, b: -0.4, c: 1.3, dealt: false };
        let x = 0.9 * 0.7 + 0.6 * -1.1;
        assert_eq!(m.natural(&u), (x, -0.4 * x + 1.3 * 0.7 + 0.5 * 0.4));
        let back = Model { arrow: Arrow::Backward, ..m };
        let y = 1.3 * 0.7 + 0.5 * 0.4;
        assert_eq!(back.natural(&u), (-0.4 * y + 0.9 * 0.7 + 0.6 * -1.1, y));
        // an experiment deals X evenly from −2.5 to 2.5, whatever Z is
        assert_eq!(Model { dealt: true, ..m }.natural(&u).0, 2.5 * 0.3);
        let deals: Vec<f64> = population(3).iter().map(|u| u.deal).collect();
        assert!(deals.iter().all(|d| (-1.0..1.0).contains(d)) && mean(&deals).abs() < 0.05);
    }

    /// Nobody in the window, one unit, a handful, ten units in all, every arrow at 0, a
    /// window at either end of its slider: every number stays finite, and the words say
    /// what has happened.
    #[test]
    fn degenerate_settings_are_said_plainly() {
        let mut s = DoSee::new();
        // far out, where no unit happens to be
        s.window = 0.05;
        s.x = s.goal.frame[1];
        s.settle();
        let l = s.look();
        assert_eq!(l.seen, 0);
        let [seen, done, slopes] = s.sentences(&l);
        assert!(seen.0.contains("nothing to say") && done.0.contains("1000 units"), "{}", seen.0);
        assert!(!format!("{}{}{}", seen.0, done.0, slopes.0).contains('?'));
        // a handful, and one: further out than most units go
        s.window = 0.2;
        for want in [HANDFUL - 1, 1] {
            s.x = (0..400).map(|k| 1.5 + k as f64 * 0.01).find(|&x| look(&s.model, &s.pop, s.units, x, s.window).seen == want).expect("a place with that many");
            let l = s.look();
            let words = &s.sentences(&l)[0].0;
            assert!(words.contains("too few") && words.contains(if want == 1 { "only 1 unit has" } else { "units have" }), "{words}");
        }
        // ten units, every arrow 0, the window at both ends
        for (units, model, window) in [(10, Model { arrow: Arrow::Forward, a: 0.0, b: 0.0, c: 0.0, dealt: false }, 0.05), (10, TRY[3].2.model, 1.0), (MOST, Model { arrow: Arrow::Backward, a: 1.5, b: 1.5, c: 1.5, dealt: false }, 1.0)] {
            let mut s = DoSee::new();
            (s.units, s.model, s.window) = (units, model, window);
            s.refresh();
            s.settle();
            for x in [s.goal.frame[0], 0.0, s.goal.frame[1]] {
                s.x = x;
                let l = s.look();
                let numbers = [l.seen_y, l.seen_z, l.done_y, l.done_z, s.trend(), s.effect()];
                assert!(numbers.iter().all(|v| v.is_finite()), "{model:?} at {x}: {numbers:?}");
                assert!(s.goal.frame.iter().all(|v| v.is_finite()) && s.goal.frame[0] < s.goal.frame[1] && s.goal.frame[2] < s.goal.frame[3]);
                assert!(s.bars_target().iter().all(|v| v.is_finite()));
            }
        }
        // (and nothing breaks on an empty population)
        assert_eq!(look(&TRY[0].2.model, &[], 10, 0.0, 0.25).units, 0);
        assert_eq!(trend(&TRY[0].2.model, &[], 10), 0.0);
        assert!(Seeing::of(&[], &[]).at(0.0, 1.0).is_none());
    }

    /// Everything drawn comes to rest after any change, by the clock, at 60 frames a
    /// second and at 144: the travel in its time, the glides soon after.
    #[test]
    fn the_picture_comes_to_rest() {
        for hz in [60.0f32, 144.0] {
            for k in 0..TRY.len() {
                let mut s = DoSee::new();
                s.apply(TRY[k].2);
                let mut frames = 0;
                while s.moving {
                    s.advance(1.0 / hz, 1.0 / hz);
                    frames += 1;
                    assert!(frames as f32 / hz < TRAVEL + 1.5, "experiment {k} still moving at {hz} Hz");
                }
                assert_eq!(s.travel, if TRY[k].2.mode == Mode::Do { 1.0 } else { 0.0 });
                // where it rests is exactly where it is going
                assert!(s.xs.shown == s.goal.xs && s.ys.shown == s.goal.ys);
            }
        }
        // from one experiment to another, the units never jump: an experiment of doing
        // chosen while they are at x takes them back first, and then to x again
        let mut s = run(0);
        for k in [2, 0, 1, 3, 0] {
            s.apply(TRY[k].2);
            let (mut was, mut back) = (s.travel, s.travel == 0.0);
            for _ in 0..600 {
                s.advance(1.0 / 60.0, 1.0 / 60.0);
                assert!((s.travel - was).abs() <= 1.0 / 60.0 / TRAVEL + 1e-6, "to experiment {k}: travel {was} to {}", s.travel);
                back |= s.travel == 0.0;
                was = s.travel;
            }
            assert!(!s.moving && back && s.travel == if TRY[k].2.mode == Mode::Do { 1.0 } else { 0.0 }, "experiment {k}");
        }
        // the number of units changed while they glide: the new ones start in place
        let mut s = DoSee::new();
        s.apply(TRY[3].2);
        s.advance(0.05, 0.05);
        s.units = 1500;
        s.refresh();
        for _ in 0..400 {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        assert!(!s.moving && s.xs.shown.len() == 1500 && s.xs.shown == s.goal.xs);
    }

    /// The parts of the picture keep to the canvas and clear of each other, whatever
    /// its shape.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (x, y) in [(1078.0f32, 1000.0f32), (1100.0, 700.0), (570.0, 700.0), (1600.0, 1000.0), (390.0, 800.0), (360.0, 640.0), (400.0, 600.0), (600.0, 900.0), (600.0, 400.0), (640.0, 480.0), (844.0, 390.0), (2400.0, 1300.0)] {
            let inset = if x < 900.0 { 46.0 } else { 0.0 };
            let lay = Layout::of(Vec2::new(x, y), inset);
            let rects = [("tabs", lay.tabs), ("graph", lay.graph), ("words", lay.words), ("key", lay.key), ("plot", lay.plot), ("histograms", lay.hist)];
            for (name, r) in rects {
                assert!(r.min.is_finite() && r.max.is_finite() && r.min.x >= 0.0 && r.min.y >= inset && r.max.x <= x && r.max.y <= y, "{x}x{y}: {name} {r:?}");
            }
            // (the words and the key are left out where they have no room)
            let solid = [lay.tabs, lay.graph, lay.plot, lay.hist];
            for (i, a) in solid.iter().enumerate() {
                for b in &solid[i + 1..] {
                    assert!(a.max.x <= b.min.x || b.max.x <= a.min.x || a.max.y <= b.min.y || b.max.y <= a.min.y, "{x}x{y}: {a:?} and {b:?} overlap");
                }
            }
            assert!(lay.plot.w() >= 150.0 && lay.plot.h() >= 150.0, "{x}x{y}: the scatter is {:?}", lay.plot);
            for k in 0..3 {
                assert!(lay.tab(k).max.x <= x, "{x}x{y}: tab {k}");
            }
        }
        // a small window has the scatter and its histograms alone, side by side, and one
        // far too small still has no rectangle turned inside out
        for (x, y) in [(240.0f32, 180.0f32), (300.0, 300.0), (420.0, 500.0)] {
            let lay = Layout::of(Vec2::new(x, y), 46.0);
            for r in [lay.tabs, lay.graph, lay.words, lay.key, lay.plot, lay.hist] {
                assert!(r.w() >= 1.0 && r.h() >= 1.0 && r.min.is_finite() && r.max.is_finite());
            }
            assert!(lay.graph.h() < 40.0 && lay.words.w() < 120.0 && lay.plot.max.x <= lay.hist.min.x && lay.hist.max.x <= x, "{x}x{y}");
        }
    }

    /// The numbers behind the captions, for several populations (run with --ignored
    /// --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for seed in 1u64..=12 {
            let pop = population(seed);
            let mut line = format!("seed {seed:>2}:");
            for (_, _, s) in TRY {
                let m = s.model;
                let l = look(&m, &pop, UNITS, s.x, WINDOW);
                line += &format!("  [{:>3} {:.2} {:.2} | {:.2} {:.2} | {:.2} {:.2}]", l.seen, l.seen_y, l.seen_z, l.done_y, l.done_z, trend(&m, &pop, UNITS), effect(&m, &pop, UNITS));
            }
            println!("{line}");
        }
    }
}

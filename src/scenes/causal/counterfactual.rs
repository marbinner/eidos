//! What would have happened: a counterfactual, worked out in three steps for one unit
//! from a model of how its Y is made, beside what the whole population would do.

use super::*;
use crate::math::causal::{mean, variance};
use glam::DVec2;

pub const DEF: SceneDef = SceneDef {
    id: "counterfactual",
    title: "What would have happened",
    group: GROUP,
    blurb: "The same unit with one thing changed: the three steps of a counterfactual.",
    make: |_| Box::new(Counterfactual::create()),
};

// ----- the model ----------------------------------------------------------------------

/// How many units the population has, and the seed they are drawn from.
const UNITS: usize = 200;
const SEED: u64 = 5;
/// What the picture shows of X and of Y.
const X_MAX: f64 = 6.0;
const Y_MAX: f64 = 7.0;
/// The spread of noise that is added; and of a factor that multiplies, on a
/// logarithmic scale.
const SIGMA: f64 = 0.55;
const SPREAD: f64 = 0.15;
/// The middle 90 % of a normal variable lies within this many standard deviations of
/// its mean.
const Z90: f64 = 1.644_853_626_951_472;
/// Points along each curve, and how many other units' paths can be shown.
const SAMPLES: usize = 121;
const OTHERS: usize = 9;
/// The lowest Y a unit can be given by hand: a factor that multiplies must be positive.
const Y_LEAST: f64 = 0.05;

/// The shape of f, the average Y when X is set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Levels,
    Hump,
    Line,
    Flat,
}

impl Shape {
    const ALL: [Shape; 4] = [Shape::Levels, Shape::Hump, Shape::Line, Shape::Flat];

    fn name(self) -> &'static str {
        match self {
            Shape::Levels => "levels off",
            Shape::Hump => "rises and falls",
            Shape::Line => "straight",
            Shape::Flat => "flat: no effect",
        }
    }

    fn f(self, x: f64) -> f64 {
        match self {
            Shape::Levels => 1.8 + 2.7 * (1.0 - (-x / 1.6).exp()),
            Shape::Hump => {
                let u = x / 2.4;
                1.8 + 2.5 * u * (1.0 - u).exp()
            }
            Shape::Line => 1.8 + 0.44 * x,
            Shape::Flat => 3.0,
        }
    }
}

/// How a unit's noise enters its Y.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Enters {
    /// Y = f(X) + N.
    Added,
    /// Y = f(X) M, with M > 0 and on average 1.
    Multiplied,
}

/// A model of how Y is made: the curve, how the noise enters, and whether a unit's
/// noise would be the same at another X (or a new one, drawn afresh for every value).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Model {
    shape: Shape,
    enters: Enters,
    keeps: bool,
}

impl Model {
    fn f(&self, x: f64) -> f64 {
        self.shape.f(x)
    }

    /// Y at X = x for a unit whose noise is z (on the standard scale: z = 1 is one
    /// standard deviation above the mean).
    fn y(&self, x: f64, z: f64) -> f64 {
        match self.enters {
            Enters::Added => self.f(x) + SIGMA * z,
            Enters::Multiplied => self.f(x) * (SPREAD * z - 0.5 * SPREAD * SPREAD).exp(),
        }
    }

    /// Step 1: the noise of a unit seen at (x, y), on the standard scale. (A factor
    /// needs y above 0, which the handle keeps it.)
    fn noise_of(&self, x: f64, y: f64) -> f64 {
        match self.enters {
            Enters::Added => (y - self.f(x)) / SIGMA,
            Enters::Multiplied => ((y.max(1e-9) / self.f(x)).ln() + 0.5 * SPREAD * SPREAD) / SPREAD,
        }
    }

    /// Steps 2 and 3: what a unit seen at (x, y) would have had at X = x2.
    fn answer(&self, x: f64, y: f64, x2: f64) -> Answer {
        if x2 == x {
            return Answer::Point(y);
        }
        if self.keeps {
            Answer::Point(self.y(x2, self.noise_of(x, y)))
        } else {
            Answer::Spread(self.y(x2, -Z90), self.y(x2, Z90))
        }
    }
}

/// What the model says would have happened.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Answer {
    /// This, and nothing else.
    Point(f64),
    /// Anything the population could have, with its odds: the middle 90 % of it.
    Spread(f64, f64),
}

/// The units: each one's X as it was seen, its noise on the standard scale, and where
/// it is drawn across the strips.
struct Population {
    x: Vec<f64>,
    z: Vec<f64>,
    jitter: Vec<f32>,
    /// Nine units at evenly spaced ranks of their noise, whose paths can be shown.
    others: [usize; OTHERS],
}

impl Population {
    fn new() -> Population {
        let mut rng = Rng::new(SEED);
        let mut x: Vec<f64> = (0..UNITS).map(|i| 0.15 + (X_MAX - 0.3) * (i as f64 + rng.f32() as f64) / UNITS as f64).collect();
        rng.shuffle(&mut x);
        let raw: Vec<f64> = (0..UNITS).map(|_| rng.normal() as f64).collect();
        // (exactly mean 0 and spread 1, so that the average of everyone at any x is
        // the curve itself when the noise is added)
        let (m, sd) = (mean(&raw), variance(&raw).sqrt());
        let z: Vec<f64> = raw.iter().map(|v| (v - m) / sd).collect();
        let jitter = (0..UNITS).map(|_| rng.f32()).collect();
        let mut order: Vec<usize> = (0..UNITS).collect();
        order.sort_by(|a, b| z[*a].total_cmp(&z[*b]));
        let others = std::array::from_fn(|k| order[(2 * k + 1) * UNITS / (2 * OTHERS)]);
        Population { x, z, jitter, others }
    }

    /// How many units' noise is below `z`: how many of them this unit is above, at
    /// any one X, when everyone is given that X.
    fn below(&self, z: f64) -> usize {
        self.z.iter().filter(|v| **v < z).count()
    }
}

// ----- the numbers as written -------------------------------------------------------------

/// A number to two places, from a count of hundredths, with a true minus sign.
fn cents(v: i64) -> String {
    let sign = if v < 0 { "−" } else { "" };
    format!("{sign}{}.{:02}", v.abs() / 100, v.abs() % 100)
}

/// The same with its sign always written: +1.20, −0.35, 0.00.
fn signed_cents(v: i64) -> String {
    if v > 0 {
        format!("+{}", cents(v))
    } else {
        cents(v)
    }
}

fn to_cents(v: f64) -> i64 {
    (v * 100.0).round().clamp(-1e12, 1e12) as i64
}

/// The numbers of the three steps as the scene writes them. Each is worked out from
/// the numbers written before it, so that the sums on the screen are right as written
/// (the picture uses the numbers in full; they differ by less than a hundredth).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sums {
    x: i64,
    y: i64,
    x2: i64,
    fx: i64,
    fx2: i64,
    /// Added: the noise n = y − f(x), in hundredths. Multiplied: the factor m = y / f(x),
    /// in thousandths.
    noise: i64,
    /// The answer: one value, or the middle 90 % of what it could be (hundredths).
    answer: Result<i64, (i64, i64)>,
    /// x′ is x: nothing is changed.
    same: bool,
}

impl Sums {
    fn of(model: &Model, unit: DVec2, x2: f64) -> Sums {
        let (x, y, fx, fx2) = (to_cents(unit.x), to_cents(unit.y), to_cents(model.f(unit.x)), to_cents(model.f(x2)));
        let noise = match model.enters {
            Enters::Added => y - fx,
            Enters::Multiplied => ((y as f64 / fx.max(1) as f64) * 1000.0).round() as i64,
        };
        let same = x2 == unit.x;
        let answer = match model.answer(unit.x, unit.y, x2) {
            _ if same => Ok(y),
            Answer::Point(_) => Ok(match model.enters {
                Enters::Added => fx2 + noise,
                Enters::Multiplied => ((noise * fx2) as f64 / 1000.0).round() as i64,
            }),
            Answer::Spread(lo, hi) => Err((to_cents(lo), to_cents(hi))),
        };
        Sums { x, y, x2: to_cents(x2), fx, fx2, noise, answer, same }
    }

    /// The noise as written: n = 0.50, or m = 1.165.
    fn noise_text(&self, enters: Enters) -> String {
        match enters {
            Enters::Added => cents(self.noise),
            Enters::Multiplied => format!("{}.{:03}", self.noise / 1000, self.noise % 1000),
        }
    }
}

// ----- what is drawn --------------------------------------------------------------------

/// Everything the picture shows, in the units of X and Y: what a setting comes to, and
/// what a change of setting is drawn passing through (one picture blended into the next).
#[derive(Clone, Debug, PartialEq)]
struct Geometry {
    curve: Vec<DVec2>,
    units: Vec<DVec2>,
    /// Every unit's Y with X set to this unit's x, and to x′: the two strips.
    at_x: Vec<f64>,
    at_x2: Vec<f64>,
    unit: DVec2,
    x2: f64,
    fx: f64,
    fx2: f64,
    /// Where the answer is drawn (the middle of the band when there is one).
    y2: f64,
    /// 1 when the answer is one value, 0 when it is a band.
    point: f64,
    band: [f64; 2],
    /// The unit's would-have-beens at every x′ (the curve itself when it has none).
    path: Vec<DVec2>,
    others: Vec<Vec<DVec2>>,
    /// How much the other units' paths show.
    others_shown: f64,
}

fn sample_x(i: usize) -> f64 {
    X_MAX * i as f64 / (SAMPLES - 1) as f64
}

impl Geometry {
    fn of(model: &Model, pop: &Population, unit: DVec2, x2: f64, show_others: bool) -> Geometry {
        let curve: Vec<DVec2> = (0..SAMPLES).map(|i| DVec2::new(sample_x(i), model.f(sample_x(i)))).collect();
        let units = pop.x.iter().zip(&pop.z).map(|(x, z)| DVec2::new(*x, model.y(*x, *z))).collect();
        let at_x = pop.z.iter().map(|z| model.y(unit.x, *z)).collect();
        let at_x2 = pop.z.iter().map(|z| model.y(x2, *z)).collect();
        let z = model.noise_of(unit.x, unit.y);
        let (y2, point, band) = match model.answer(unit.x, unit.y, x2) {
            Answer::Point(v) => (v, 1.0, [v, v]),
            Answer::Spread(lo, hi) => (model.f(x2), 0.0, [lo, hi]),
        };
        let path = if model.keeps { (0..SAMPLES).map(|i| DVec2::new(sample_x(i), model.y(sample_x(i), z))).collect() } else { curve.clone() };
        let others = pop.others.iter().map(|&k| (0..SAMPLES).map(|i| DVec2::new(sample_x(i), model.y(sample_x(i), pop.z[k]))).collect()).collect();
        let others_shown = if show_others && model.keeps { 1.0 } else { 0.0 };
        Geometry { curve, units, at_x, at_x2, unit, x2, fx: model.f(unit.x), fx2: model.f(x2), y2, point, band, path, others, others_shown }
    }

    /// A share `s` of the way from `a` to `b`.
    fn between(a: &Geometry, b: &Geometry, s: f64) -> Geometry {
        let v = |p: &[DVec2], q: &[DVec2]| p.iter().zip(q).map(|(p, q)| p.lerp(*q, s)).collect::<Vec<_>>();
        let n = |p: &[f64], q: &[f64]| p.iter().zip(q).map(|(p, q)| p + (q - p) * s).collect::<Vec<_>>();
        let l = |p: f64, q: f64| p + (q - p) * s;
        Geometry {
            curve: v(&a.curve, &b.curve),
            units: v(&a.units, &b.units),
            at_x: n(&a.at_x, &b.at_x),
            at_x2: n(&a.at_x2, &b.at_x2),
            unit: a.unit.lerp(b.unit, s),
            x2: l(a.x2, b.x2),
            fx: l(a.fx, b.fx),
            fx2: l(a.fx2, b.fx2),
            y2: l(a.y2, b.y2),
            point: l(a.point, b.point),
            band: [l(a.band[0], b.band[0]), l(a.band[1], b.band[1])],
            path: v(&a.path, &b.path),
            others: a.others.iter().zip(&b.others).map(|(p, q)| v(p, q)).collect(),
            others_shown: l(a.others_shown, b.others_shown),
        }
    }

    /// The unit's would-have-been at X = t, read off its path.
    fn path_at(&self, t: f64) -> f64 {
        let i = (t / X_MAX * (SAMPLES - 1) as f64).max(0.0).min((SAMPLES - 1) as f64);
        let (k, r) = ((i.floor() as usize).min(SAMPLES - 2), i - i.floor());
        let (a, b) = (self.path.get(k).map_or(0.0, |p| p.y), self.path.get(k + 1).map_or(0.0, |p| p.y));
        a + (b - a) * r
    }

    fn curve_at(&self, t: f64) -> f64 {
        let i = (t / X_MAX * (SAMPLES - 1) as f64).max(0.0).min((SAMPLES - 1) as f64);
        let (k, r) = ((i.floor() as usize).min(SAMPLES - 2), i - i.floor());
        let (a, b) = (self.curve.get(k).map_or(0.0, |p| p.y), self.curve.get(k + 1).map_or(0.0, |p| p.y));
        a + (b - a) * r
    }
}

// ----- the three steps, played ----------------------------------------------------------

/// When each step plays (seconds from the start of the first), and when it is all done.
const READ: (f32, f32) = (0.0, 1.2);
const SET: (f32, f32) = (1.6, 2.9);
const WORK: (f32, f32) = (3.3, 4.8);
const SHOW: (f32, f32) = (4.8, 5.5);

/// The three steps being played: the time on the clock (negative before the first
/// begins), and whether the answer as it was drawn before fades out first.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Play {
    t: f32,
    from: f32,
    fade: bool,
}

/// How far each part of the play has got, from 0 to 1 (all 1 when nothing plays), and
/// which step is playing (0 before the first, 4 when they are done).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Progress {
    /// What is left of the answer drawn before the play began.
    before: f32,
    read: f32,
    set: f32,
    work: f32,
    show: f32,
    step: usize,
}

fn smooth(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Progress {
    const DONE: Progress = Progress { before: 0.0, read: 1.0, set: 1.0, work: 1.0, show: 1.0, step: 4 };

    fn of(play: Option<Play>) -> Progress {
        let Some(p) = play else { return Progress::DONE };
        let at = |(a, b): (f32, f32)| smooth((p.t - a) / (b - a));
        let step = if p.t < READ.0 {
            0
        } else if p.t < SET.0 {
            1
        } else if p.t < WORK.0 {
            2
        } else if p.t < SHOW.1 {
            3
        } else {
            4
        };
        let before = if p.fade { 1.0 - smooth((p.t - p.from) / 0.35) } else { 0.0 };
        Progress { before, read: at(READ), set: at(SET), work: at(WORK), show: at(SHOW), step }
    }

    /// How much a part revealed by `part` (one of the fields above) shows: what is left
    /// of it from before the play until its own part begins.
    fn shown(&self, part: f32) -> f32 {
        if part > 0.0 {
            part
        } else {
            self.before
        }
    }
}

// ----- where things go ------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    min: Vec2,
    max: Vec2,
}

impl Rect {
    fn new(min: Vec2, max: Vec2) -> Rect {
        Rect { min, max: max.max(min + Vec2::ONE) }
    }

    fn w(&self) -> f32 {
        self.max.x - self.min.x
    }

    fn contains(&self, p: Vec2) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }
}

/// Where everything goes on a canvas, in logical pixels from its top-left corner. On a
/// wide canvas: the steps and the model above, the scatter below them with the strips
/// of everyone at x and at x′ beside it. On a phone held upright: the scatter with the
/// strip at x′ beside it, the steps under them.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    /// Logical pixels per unit of the stage.
    ppu: f32,
    narrow: bool,
    plot: Rect,
    /// Everyone at x, and everyone at x′ (none: no room).
    strips: [Option<Rect>; 2],
    steps: Option<Rect>,
    /// The model: its graph and its equation.
    card: Option<Rect>,
    /// The button that plays the three steps.
    play: Option<Rect>,
    /// Room for the caption along the bottom.
    caption: bool,
    size: f32,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let ppu = (view.y / STAGE_H).min(view.x / 2.0).max(1e-3);
        let top = inset.max(0.0) + 8.0;
        // a canvas squeezed to a sliver (the controls open over a phone's picture): the
        // scatter alone, until it opens out again
        if view.x < 200.0 {
            let plot = Rect::new(Vec2::new(30.0, top + 30.0), Vec2::new(view.x - 6.0, view.y - 30.0));
            return Layout { view, ppu, narrow: true, plot, strips: [None, None], steps: None, card: None, play: None, caption: false, size: 12.0 };
        }
        let narrow = view.x < 620.0 && view.y >= 520.0;
        if narrow {
            let strip_w = 62.0;
            let steps_h = (view.y * 0.29).clamp(170.0, 240.0);
            let plot = Rect::new(Vec2::new(34.0, top + 40.0), Vec2::new(view.x - strip_w - 20.0, view.y - steps_h - 34.0));
            let strip = Rect::new(Vec2::new(view.x - strip_w - 8.0, plot.min.y), Vec2::new(view.x - 8.0, plot.max.y));
            let steps = Rect::new(Vec2::new(8.0, view.y - steps_h - 6.0), Vec2::new(view.x - 8.0, view.y - 6.0));
            let play = Rect::new(Vec2::new(steps.max.x - 118.0, steps.min.y + 6.0), Vec2::new(steps.max.x - 6.0, steps.min.y + 30.0));
            return Layout { view, ppu, narrow, plot, strips: [None, Some(strip)], steps: Some(steps), card: None, play: Some(play), caption: false, size: 12.0 };
        }
        let below = if view.y >= 420.0 { 60.0 } else { 26.0 };
        let right = view.x - 12.0;
        if view.y < 560.0 && view.x >= 700.0 {
            // wide and short (a phone held sideways): the steps in a column on the right,
            // the strip of everyone at x′ beside the scatter
            let column = (view.x * 0.36).clamp(260.0, 380.0);
            let steps = Rect::new(Vec2::new(right - column, top), Vec2::new(right, view.y - 8.0));
            let strip_w = 70.0;
            let plot_top = top + 40.0;
            // (room under the strip for this unit's rank, two lines)
            let below = below.max(40.0);
            let strip = Rect::new(Vec2::new(steps.min.x - 16.0 - strip_w, plot_top), Vec2::new(steps.min.x - 16.0, view.y - below));
            let plot = Rect::new(Vec2::new(44.0, plot_top), Vec2::new(strip.min.x - 22.0, view.y - below));
            let play = Rect::new(Vec2::new(steps.max.x - 126.0, steps.max.y - 32.0), Vec2::new(steps.max.x - 8.0, steps.max.y - 8.0));
            return Layout { view, ppu, narrow: true, plot, strips: [None, Some(strip)], steps: Some(steps), card: None, play: Some(play), caption: false, size: 12.0 };
        }
        // a head for the steps and the model, if the canvas is tall enough for it
        let head = (view.y >= 560.0 && view.x >= 560.0).then_some(126.0);
        let plot_top = top + head.map_or(40.0, |h| h + 44.0);
        let strip_w = 96.0;
        let both = view.x >= 900.0;
        let one = view.x >= 520.0;
        let second = one.then(|| Rect::new(Vec2::new(right - strip_w, plot_top), Vec2::new(right, view.y - below)));
        let first = both.then(|| Rect::new(Vec2::new(right - 2.0 * strip_w - 14.0, plot_top), Vec2::new(right - strip_w - 14.0, view.y - below)));
        let plot_right = first.or(second).map_or(view.x - 14.0, |s| s.min.x - 26.0);
        let plot = Rect::new(Vec2::new(50.0, plot_top), Vec2::new(plot_right, view.y - below));
        let (steps, card) = match head {
            Some(h) => {
                let card_left = first.or(second).map_or(view.x, |s| s.min.x);
                let card = (card_left < view.x - 150.0).then(|| Rect::new(Vec2::new(card_left, top), Vec2::new(right, top + h)));
                (Some(Rect::new(Vec2::new(12.0, top), Vec2::new(plot.max.x, top + h))), card)
            }
            None => (None, None),
        };
        let play = match steps {
            Some(s) => Some(Rect::new(Vec2::new(s.max.x - 176.0, s.min.y + 8.0), Vec2::new(s.max.x - 8.0, s.min.y + 34.0))),
            None => (view.x >= 300.0 && view.y >= 200.0).then(|| Rect::new(Vec2::new(plot.max.x - 168.0, plot.min.y - 22.0), Vec2::new(plot.max.x, plot.min.y - 2.0))),
        };
        Layout { view, ppu, narrow: false, plot, strips: [first, second], steps, card, play, caption: view.y >= 420.0, size: 13.0 }
    }

    /// A point of the canvas (logical pixels) on the stage.
    fn stage(&self, px: Vec2) -> Vec2 {
        Vec2::new(px.x - self.view.x * 0.5, self.view.y * 0.5 - px.y) / self.ppu
    }

    /// The room kept inside the plot's edge, so that a dot on the edge of the range is
    /// drawn whole.
    const PAD: f32 = 10.0;

    /// From X and Y to the stage.
    fn map(&self) -> Map {
        let p = self.plot;
        let (left, right) = (self.stage(Vec2::new(p.min.x + Self::PAD, 0.0)).x, self.stage(Vec2::new(p.max.x - Self::PAD, 0.0)).x);
        let (bottom, top) = (self.stage(Vec2::new(0.0, p.max.y - Self::PAD)).y, self.stage(Vec2::new(0.0, p.min.y + Self::PAD)).y);
        let k = Vec2::new(((right - left) / X_MAX as f32).max(1e-6), ((top - bottom) / Y_MAX as f32).max(1e-6));
        Map { o: Vec2::new(left, bottom), k }
    }

    /// A rectangle of the canvas on the stage (lower left, upper right).
    fn on_stage(&self, r: Rect) -> (Vec2, Vec2) {
        let (a, b) = (self.stage(r.min), self.stage(r.max));
        (a.min(b), a.max(b))
    }
}

/// From the units of X and Y to the stage: o + p k.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Map {
    o: Vec2,
    k: Vec2,
}

impl Map {
    fn to(&self, p: DVec2) -> Vec2 {
        self.o + p.as_vec2() * self.k
    }

    fn at(&self, x: f64, y: f64) -> Vec2 {
        self.to(DVec2::new(x, y))
    }

    fn back(&self, s: Vec2) -> DVec2 {
        ((s - self.o) / self.k).as_dvec2()
    }
}

// ----- the scene ------------------------------------------------------------------------

/// Seconds a change of setting takes to play.
const CHANGE_TIME: f32 = 0.8;
/// A dragged handle within this many pixels of a special place (the unit on the curve,
/// x′ on x) is put there.
const SNAP: f32 = 7.0;
/// How quickly a value put in its place by the hand is drawn arriving there.
const GLIDE: f64 = 0.06;

/// A setting: the model, the unit as it was seen (its x and its y), x′, and whether the
/// other units' paths are shown.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    model: Model,
    unit: (f64, f64),
    x2: f64,
    others: bool,
}

const fn added(shape: Shape, keeps: bool) -> Model {
    Model { shape, enters: Enters::Added, keeps }
}

const fn multiplied(shape: Shape) -> Model {
    Model { shape, enters: Enters::Multiplied, keeps: true }
}

/// The scene as it opens, and as the first experiment sets it up. (Each unit's y is
/// chosen so that its noise comes out round as written: here n = 3.72 − 3.22.)
const OPENING: Setup = Setup { model: added(Shape::Levels, true), unit: (1.2, 3.72), x2: 4.4, others: false };

/// Experiments: (button, what to look for, setting).
const TRIALS: [(&str, &str, Setup); 5] = [
    (
        "the three steps",
        r"One unit, seen at $x = 1.20$ with $y = 3.72$, half a unit above the curve. Watch the steps play. Its noise is read off as its height above the curve, $n = 0.50$ (purple). Its $X$ is set to $x' = 4.40$, and nothing else is changed. $Y$ is worked out again as $f(x') + n$, the same height above the curve there: $y' = 4.83$. The dotted path, the curve moved up by $n$, is what its $Y$ would have been at every $x'$.",
        OPENING,
    ),
    (
        "a lucky unit stays lucky",
        r"Seen at $x = 4.70$ with $y = 5.46$, far above the curve: had all 200 units been given that $X$, this one would be higher than 195 of them (left strip). Had it been given $x' = 1.00$ instead, its $Y$ would have fallen with the curve, to 4.15, and still been higher than 195 of the 200 given $x'$ (right strip): above the average there by the same $n = 1.10$.",
        Setup { model: added(Shape::Levels, true), unit: (4.7, 5.46), x2: 1.0, others: false },
    ),
    (
        "noise that multiplies",
        r"Now each unit's $Y$ is the curve times a factor $M$ of its own (on average 1), and the spread grows where the curve is high. This unit's factor is $m = 1.200$: its dotted path is the curve stretched, not moved up, and the faint paths of nine other units fan out instead of running side by side. The same change of $X$ changes each unit's $Y$ by a different amount: this one's by $+1.56$, the average by $+1.30$.",
        Setup { model: multiplied(Shape::Levels), unit: (1.0, 3.66), x2: 4.6, others: true },
    ),
    (
        "the individual and the average disagree",
        r"Raising $X$ from 0.50 to 5.50 raises the average $Y$ by 1.89: that is the average effect. This unit gets 0.75 times what the curve gives, at every $x$, so its own $Y$ would have risen by only 1.42, three quarters as much. The average effect answers for the population; the counterfactual, for this unit.",
        Setup { model: multiplied(Shape::Levels), unit: (0.5, 1.89), x2: 5.5, others: false },
    ),
    (
        "same data, different answers",
        r"The unit and the scatter of the first experiment, dot for dot, but a model that draws each unit's noise afresh for every value of $X$. Both models give the same data under every intervention, since each unit is seen at one $X$ only. But now what was seen says nothing about the noise at $x'$: $y'$ is anything the population could have there, between 3.42 and 5.23 with 9 chances in 10 (orange). Switch 'its noise at another $x$' back to 'the same': not one dot of the data moves.",
        Setup { model: added(Shape::Levels, false), unit: (1.2, 3.72), x2: 4.4, others: false },
    ),
];

impl Setup {
    fn unit(&self) -> DVec2 {
        DVec2::new(self.unit.0, self.unit.1)
    }
}

pub struct Counterfactual {
    model: Model,
    /// The unit as it was seen: X = x, Y = y.
    unit: DVec2,
    x2: f64,
    show_others: bool,
    pop: Population,
    /// A change of setting under way: the picture it started from, and how far it has
    /// got (0 to 1).
    change: Option<(Geometry, f32)>,
    play: Option<Play>,
    /// What a snap moved the unit and x′ by, still to be drawn closing up (X and Y
    /// units): the picture glides into the place the hand put it.
    glide_unit: DVec2,
    glide_x2: f64,
    /// Where the handles are held (on the stage), while they are.
    grip: [Option<Vec2>; 2],
}

impl Counterfactual {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let mut s = Counterfactual {
            model: OPENING.model,
            unit: OPENING.unit(),
            x2: OPENING.x2,
            show_others: OPENING.others,
            pop: Population::new(),
            change: None,
            play: None,
            glide_unit: DVec2::ZERO,
            glide_x2: 0.0,
            grip: [None; 2],
        };
        // the steps play as the scene opens (with nothing drawn of them before)
        s.play = Some(Play { t: -0.3, from: -0.3, fade: false });
        s
    }

    fn goal(&self) -> Geometry {
        Geometry::of(&self.model, &self.pop, self.unit, self.x2, self.show_others)
    }

    /// What is drawn: on its way from the last setting while a change plays, and with
    /// the hand's snaps closing up.
    fn shown(&self) -> Geometry {
        let goal = self.goal();
        let mut g = match &self.change {
            Some((from, t)) => Geometry::between(from, &goal, smooth(*t) as f64),
            None => goal,
        };
        if self.glide_unit != DVec2::ZERO || self.glide_x2 != 0.0 {
            // (the unit and x′ are drawn where they have got to, and everything that
            // hangs on them as the model has it there)
            g = Geometry::of(&self.model, &self.pop, g.unit + self.glide_unit, g.x2 + self.glide_x2, self.show_others);
        }
        g
    }

    fn sums(&self) -> Sums {
        Sums::of(&self.model, self.unit, self.x2)
    }

    /// Take a setting, and play the way to it from what is drawn now.
    fn apply(&mut self, setup: &Setup) {
        let from = self.shown();
        self.model = setup.model;
        self.unit = setup.unit();
        self.x2 = setup.x2;
        self.show_others = setup.others;
        (self.glide_unit, self.glide_x2) = (DVec2::ZERO, 0.0);
        self.change = Some((from, 0.0));
        self.play = None;
    }

    fn set_up(&mut self, k: usize) {
        self.apply(&TRIALS[k].2);
        if k == 0 {
            self.replay(true);
        }
    }

    /// Play the three steps (after the change of setting under way, if any).
    fn replay(&mut self, after_change: bool) {
        let lead = if after_change && self.change.is_some() { CHANGE_TIME + 0.1 } else { 0.45 };
        self.play = Some(Play { t: -lead, from: -lead, fade: true });
    }

    /// A change of setting from the panel: played from what is drawn now (a snap's
    /// glide, if one was closing up, is in that already).
    fn changed(&mut self, from: Geometry) {
        self.change = Some((from, 0.0));
        (self.glide_unit, self.glide_x2) = (DVec2::ZERO, 0.0);
        self.play = None;
    }

    /// Move the change of setting and the play on by `dt` seconds of the scene's time
    /// (a change all the way if its time stands still: a click must still be
    /// answered), and the hand's glides by `real` seconds of the clock.
    fn advance(&mut self, dt: f32, real: f32) {
        if let Some((_, t)) = self.change.as_mut() {
            *t = if dt > 0.0 { *t + dt / CHANGE_TIME } else { 1.0 };
            if *t >= 1.0 {
                self.change = None;
            }
        }
        if let Some(p) = self.play.as_mut() {
            p.t += dt.max(0.0);
            if p.t >= SHOW.1 {
                self.play = None;
            }
        }
        let k = (-(real.max(0.0) as f64) / GLIDE).exp();
        self.glide_unit *= k;
        self.glide_x2 *= k;
        // (exactly there in the end, not nearly: the picture then stops changing)
        if self.glide_unit.length() < 1e-3 || !self.glide_unit.is_finite() {
            self.glide_unit = DVec2::ZERO;
        }
        if self.glide_x2.abs() < 1e-3 || !self.glide_x2.is_finite() {
            self.glide_x2 = 0.0;
        }
    }

    /// The unit put where the hand has it: within the picture, kept above 0 (a factor
    /// that multiplies is positive), and on the curve if it is within a few pixels of
    /// it. Going onto the curve or off it, the picture glides there from where it was
    /// drawn; otherwise it follows the hand exactly.
    fn put_unit(&mut self, p: DVec2, per_px: f64) {
        let hand = DVec2::new(p.x.max(0.0).min(X_MAX), p.y.max(Y_LEAST).min(Y_MAX));
        if !hand.is_finite() {
            return;
        }
        let f = self.model.f(hand.x);
        let snapped = (hand.y - f).abs() < SNAP as f64 * per_px;
        let to = if snapped { DVec2::new(hand.x, f) } else { hand };
        let was = self.unit.y == self.model.f(self.unit.x);
        if snapped && !was {
            self.glide_unit = hand - to;
        } else if !snapped && was {
            self.glide_unit += self.unit - to;
        }
        self.unit = to;
    }

    /// x′ put where the hand has it: within the picture, and on x if it is within a
    /// few pixels of it (gliding there, and away, as the unit does onto its curve).
    fn put_x2(&mut self, x: f64, per_px: f64) {
        let hand = x.max(0.0).min(X_MAX);
        if !hand.is_finite() {
            return;
        }
        let snapped = (hand - self.unit.x).abs() < SNAP as f64 * per_px;
        let to = if snapped { self.unit.x } else { hand };
        let was = self.x2 == self.unit.x;
        if snapped && !was {
            self.glide_x2 = hand - to;
        } else if !snapped && was {
            self.glide_x2 += self.x2 - to;
        }
        self.x2 = to;
    }
}

// ----- drawing --------------------------------------------------------------------------

/// Where a piece of the picture goes when it is not to be seen: far off any canvas.
/// Each piece is drawn in every frame, seen or not, so that what is drawn keeps its
/// order from frame to frame (`eidos motion` compares frames point by point).
const NOWHERE: Vec2 = Vec2::new(-1e4, -1e4);

/// The part of the curve `pts` within the range of Y the picture shows, as pieces.
fn clipped(pts: &[DVec2]) -> Vec<Vec<DVec2>> {
    let mut pieces: Vec<Vec<DVec2>> = Vec::new();
    let mut piece: Vec<DVec2> = Vec::new();
    let inside = |p: DVec2| p.y >= 0.0 && p.y <= Y_MAX && p.is_finite();
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        if !(a.is_finite() && b.is_finite()) {
            continue;
        }
        // the share of the way from a to b at which it crosses each edge
        let cross = |level: f64| if (a.y - level) * (b.y - level) < 0.0 { Some((level - a.y) / (b.y - a.y)) } else { None };
        let mut ts: Vec<f64> = [cross(0.0), cross(Y_MAX)].into_iter().flatten().collect();
        ts.sort_by(f64::total_cmp);
        let mut from = a;
        for t in ts {
            let at = a.lerp(b, t);
            if inside(from.lerp(at, 0.5)) {
                if piece.is_empty() {
                    piece.push(from);
                }
                piece.push(at);
                pieces.push(std::mem::take(&mut piece));
            }
            from = at;
        }
        if inside(from.lerp(b, 0.5)) {
            if piece.is_empty() {
                piece.push(from);
            }
            piece.push(b);
        } else if !piece.is_empty() {
            pieces.push(std::mem::take(&mut piece));
        }
    }
    if piece.len() >= 2 {
        pieces.push(piece);
    }
    pieces.retain(|p| p.len() >= 2);
    pieces
}

/// A point of the picture, or nowhere if it is outside the range of Y shown.
fn in_range(map: &Map, p: DVec2) -> Option<Vec2> {
    (p.y >= -1e-9 && p.y <= Y_MAX + 1e-9 && p.is_finite()).then(|| map.to(p))
}

/// The answer's colour, and the noise's, the curve's and X's (the section's colours).
const UNIT: Color = EFFECT;
const NOISE: Color = THIRD;
const CURVE: Color = TRUTH;
const XS: Color = CAUSE;
/// The population.
const CROWD: Color = pal::WHITE;

struct Look {
    progress: Progress,
    /// A change of setting is playing: the names wait until it is still.
    moving: bool,
    wpp: f32,
}

impl Counterfactual {
    /// The handles: the unit (orange) and x′ (cyan, on the axis). Neither is offered
    /// while a setting changes; moving one while the steps play stops the play, and the
    /// answer is then drawn whole.
    fn handles(&mut self, f: &mut Frame, lay: &Layout, map: &Map) {
        if !f.handles.dragging() {
            self.grip = [None; 2];
        }
        if self.change.is_some() {
            return;
        }
        let per_px = 1.0 / (map.k.y * lay.ppu) as f64;
        let per_px_x = 1.0 / (map.k.x * lay.ppu) as f64;
        let g = self.shown();
        let mut at = self.grip[0].unwrap_or(map.to(g.unit));
        if f.drag2(1, &mut at, UNIT) {
            self.grip[0] = Some(at);
            self.put_unit(map.back(at), per_px);
            self.play = None;
        }
        let axis = lay.stage(Vec2::new(0.0, lay.plot.max.y)).y;
        let mut at = self.grip[1].unwrap_or(Vec2::new(map.at(g.x2, 0.0).x, axis));
        if f.drag2(2, &mut at, XS) {
            self.grip[1] = Some(at);
            self.put_x2(map.back(at).x, per_px_x);
            self.play = None;
        }
    }

    fn draw(&mut self, f: &mut Frame, lay: &Layout, map: &Map, g: &Geometry, look: &Look) {
        let mut room = Room::new(f);
        let pr = look.progress;
        let s3 = |p: Vec2| p.extend(0.0);
        // ----- backings, the grid, the numbers along the edges
        let (plo, phi) = lay.on_stage(lay.plot);
        f.dl.tris_alpha.rect(plo, phi, 0.0, Color::hex(0x0B0F1A).a(0.9));
        frame_rect(f, plo, phi, 1.0, pal::FAINT.a(0.7));
        for strip in lay.strips.iter().flatten() {
            let (lo, hi) = lay.on_stage(*strip);
            f.dl.tris_alpha.rect(lo, hi, 0.0, Color::hex(0x0B0F1A).a(0.9));
            frame_rect(f, lo, hi, 1.0, pal::FAINT.a(0.7));
            room.wall(strip.min - Vec2::new(4.0, 40.0), strip.max + Vec2::new(4.0, 40.0));
        }
        // (a number at every whole value, or every second one where they would crowd)
        let every = |px_per_unit: f32, least: f32| if px_per_unit >= least { 1 } else { 2 };
        let (every_x, every_y) = (every(map.k.x * lay.ppu, 24.0), every(map.k.y * lay.ppu, 16.0));
        for x in 0..=X_MAX as i32 {
            let (k, x) = (x, x as f64);
            f.line(map.at(x, 0.0) - Vec2::Y * (Layout::PAD / lay.ppu), map.at(x, Y_MAX) + Vec2::Y * (Layout::PAD / lay.ppu), 1.0, pal::FAINT.a(0.32));
            if k % every_x == 0 {
                f.text_off(map.at(x, 0.0), Vec2::new(0.0, Layout::PAD + 4.0), num(x, 0), 11.0, pal::DIM, Align2::CENTER_TOP);
            }
        }
        for y in 0..=Y_MAX as i32 {
            let (k, y) = (y, y as f64);
            let left = lay.stage(Vec2::new(lay.plot.min.x, 0.0)).x;
            let right = lay.strips.iter().flatten().last().map_or(phi.x, |s| lay.stage(s.max).x);
            f.line(Vec2::new(left, map.at(0.0, y).y), Vec2::new(right, map.at(0.0, y).y), 1.0, pal::FAINT.a(if y == 0.0 { 0.6 } else { 0.25 }));
            if k % every_y == 0 {
                f.text_off(Vec2::new(left, map.at(0.0, y).y), Vec2::new(-6.0, 0.0), num(y, 0), 11.0, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
        // the axes' names, at their far ends
        f.text_off(map.at(X_MAX, 0.0), Vec2::new(Layout::PAD - 5.0, Layout::PAD - 3.0), "$X$", 14.0, XS, Align2::RIGHT_BOTTOM);
        f.text_off(Vec2::new(plo.x, map.at(0.0, Y_MAX).y), Vec2::new(-22.0, -Layout::PAD - 2.0), "$Y$", 14.0, UNIT, Align2::CENTER_BOTTOM);
        room.wall(Vec2::new(lay.plot.min.x - 50.0, lay.plot.min.y - 4.0), Vec2::new(lay.plot.min.x + 2.0, lay.plot.max.y + 40.0));
        room.wall(Vec2::new(lay.plot.min.x - 50.0, lay.plot.max.y - 2.0), Vec2::new(lay.plot.max.x + 4.0, lay.plot.max.y + 40.0));

        // ----- the other units' paths, faint
        let others = g.others_shown as f32;
        for (k, path) in g.others.iter().enumerate() {
            let pieces = clipped(path);
            for piece in &pieces {
                f.path(piece.iter().map(|p| map.to(*p)), 1.3, CROWD.a(0.3 * others));
            }
            // (their own dots, a little brighter)
            let i = self.pop.others[k];
            match g.units.get(i).and_then(|p| in_range(map, *p)) {
                Some(p) if others > 0.0 => f.dot(p, 6.0, CROWD.a(0.85 * others)),
                _ => f.dot(NOWHERE, 6.0, CROWD.a(0.0)),
            }
        }

        // ----- the population, in the scatter and in the strips
        for p in &g.units {
            match in_range(map, *p) {
                Some(p) => f.dot(p, 4.5, CROWD.a(0.42)),
                None => f.dot(NOWHERE, 4.5, CROWD.a(0.0)),
            }
        }
        for (strip, values) in lay.strips.iter().zip([&g.at_x, &g.at_x2]) {
            let Some(strip) = strip else { continue };
            let (lo, hi) = lay.on_stage(*strip);
            let pad = 10.0 / lay.ppu;
            for (v, j) in values.iter().zip(&self.pop.jitter) {
                let x = lo.x + pad + (hi.x - lo.x - 2.0 * pad) * j;
                match in_range(map, DVec2::new(0.0, *v)) {
                    Some(p) => f.dot(Vec2::new(x, p.y), 4.0, CROWD.a(0.42)),
                    None => f.dot(NOWHERE, 4.0, CROWD.a(0.0)),
                }
            }
        }

        // ----- the curve: the average Y when X is set
        for piece in clipped(&g.curve) {
            f.path(piece.iter().map(|p| map.to(*p)), 2.6, CURVE);
        }
        for piece in clipped(&g.curve).iter().take(1) {
            for w in piece.windows(2).step_by(6) {
                room.line(f, s3(map.to(w[0])), s3(map.to(w[1])));
            }
        }

        // ----- x and x′, and the change of X between them
        let (x, y, x2) = (g.unit.x, g.unit.y, g.x2);
        let bottom = map.at(0.0, 0.0).y - Layout::PAD / lay.ppu;
        // (before its step begins, what is left of the last answer fades where it was)
        let set_to = if pr.set > 0.0 { x + (x2 - x) * pr.set as f64 } else { x2 };
        f.dashed(Vec2::new(map.at(x, 0.0).x, bottom), map.at(x, y.min(Y_MAX)), 1.3, XS.a(0.55));
        f.dashed(Vec2::new(map.at(x2, 0.0).x, bottom), map.at(x2, Y_MAX), 1.4, XS.a(0.8));
        room.line(f, s3(Vec2::new(map.at(x2, 0.0).x, bottom)), s3(map.at(x2, Y_MAX)));
        room.line(f, s3(Vec2::new(map.at(x, 0.0).x, bottom)), s3(map.at(x, y.min(Y_MAX))));
        // the action: X moved from x to x′ (an arrow along the bottom), and where the
        // curve is at the X it has got to
        let set_alpha = pr.shown(pr.set);
        let lift = 24.0 / lay.ppu;
        let (a, b) = (Vec2::new(map.at(x, 0.0).x, bottom + lift), Vec2::new(map.at(set_to, 0.0).x, bottom + lift));
        if (b - a).length() * lay.ppu > 6.0 && set_alpha > 0.0 {
            f.arrow(a, b, 2.2, XS.a(set_alpha * if pr.step == 2 { 1.0 } else { 0.7 }));
        } else {
            f.line(NOWHERE, NOWHERE, 2.2, XS.a(0.0));
        }
        let ring_at = DVec2::new(set_to, g.curve_at(set_to));
        match in_range(map, ring_at) {
            Some(p) if set_alpha > 0.0 => f.circle(p, Vec3::Z, 5.5 * look.wpp, 2.0, CURVE.a(set_alpha)),
            _ => f.circle(NOWHERE, Vec3::Z, 5.5 * look.wpp, 2.0, CURVE.a(0.0)),
        }

        // ----- step 1: the noise, read off as the stroke from the curve to the unit
        let read = pr.shown(pr.read) as f64;
        let stroke_top = g.fx + (y - g.fx) * if pr.read > 0.0 { pr.read as f64 } else { 1.0 };
        let stroke = |f: &mut Frame, at: f64, from: f64, to: f64, alpha: f32| match (in_range(map, DVec2::new(at, from.clamp(0.0, Y_MAX))), in_range(map, DVec2::new(at, to.clamp(0.0, Y_MAX)))) {
            (Some(p), Some(q)) if alpha > 0.0 && (q - p).length() * lay.ppu > 0.5 => f.line(p, q, 3.2, NOISE.a(alpha)),
            _ => f.line(NOWHERE, NOWHERE, 3.2, NOISE.a(0.0)),
        };
        stroke(f, x, g.fx, stroke_top, read as f32);
        if read > 0.0 {
            room.line(f, s3(map.at(x, g.fx.clamp(0.0, Y_MAX))), s3(map.at(x, stroke_top.clamp(0.0, Y_MAX))));
        }

        // ----- step 3: Y worked out again with the same noise
        let work = pr.shown(pr.work) as f64;
        let point = g.point as f32;
        // (while it plays, the unit travels along its path from x to x′, the path
        // drawn behind it; the noise's copy stands on the curve under it)
        let travel = if pr.work > 0.0 && pr.step <= 3 { x + (x2 - x) * pr.work as f64 } else { x2 };
        let top = if pr.work > 0.0 && pr.step <= 3 { g.path_at(travel) } else { g.y2 };
        let copy_alpha = if pr.work > 0.0 { (pr.work * 6.0).min(1.0) } else { work as f32 } * point;
        let copy_foot = if pr.work > 0.0 && pr.step <= 3 { g.curve_at(travel) } else { g.fx2 };
        stroke(f, travel, copy_foot, top, copy_alpha);
        // the path, dotted (what would have been): as far as the unit has travelled
        // while it plays, all of it after. (As many dots as the plot's width makes
        // about nine pixels apart, each at its own X, so that the dots stay the same
        // dots from frame to frame however the path changes.)
        let show = pr.shown(pr.show) * point;
        let path = clipped(&g.path);
        let (lo, hi) = (x.min(travel), x.max(travel));
        let during = if pr.work > 0.0 && pr.step <= 3 { point } else { 0.0 };
        let dots = ((lay.plot.w() / 9.0) as usize).clamp(16, 200);
        for k in 0..dots {
            let t = X_MAX * k as f64 / (dots - 1) as f64;
            let passed = t >= lo - 1e-9 && t <= hi + 1e-9;
            let alpha = 0.9 * show.max(if passed { during } else { 0.0 });
            match in_range(map, DVec2::new(t, g.path_at(t))) {
                Some(p) if alpha > 0.0 => f.dot(p, 4.6, UNIT.a(alpha)),
                _ => f.dot(NOWHERE, 4.6, UNIT.a(0.0)),
            }
        }
        if show > 0.3 {
            for piece in path.iter() {
                for w in piece.windows(2).step_by(6) {
                    room.line(f, s3(map.to(w[0])), s3(map.to(w[1])));
                }
            }
        }
        // the band of what it could have been, when its noise is drawn afresh
        let band = (1.0 - point) * pr.shown(pr.work);
        let mid = g.fx2;
        let opened = |v: f64| mid + (v - mid) * if pr.work > 0.0 { pr.work as f64 } else { 1.0 };
        let (b0, b1) = (opened(g.band[0]).clamp(0.0, Y_MAX), opened(g.band[1]).clamp(0.0, Y_MAX));
        let half = 7.0 / lay.ppu;
        let (q0, q1) = (map.at(x2, b0), map.at(x2, b1));
        if band > 0.0 && (q1.y - q0.y) * lay.ppu > 0.5 {
            f.dl.tris_alpha.rect(Vec2::new(q0.x - half, q0.y), Vec2::new(q0.x + half, q1.y), 0.0, UNIT.a(0.16 * band));
            frame_rect(f, Vec2::new(q0.x - half, q0.y), Vec2::new(q0.x + half, q1.y), 1.6, UNIT.a(band));
            room.boxed(on_screen(f, s3(Vec2::new(q0.x - half, q1.y))).unwrap_or(Vec2::ZERO), on_screen(f, s3(Vec2::new(q0.x + half, q0.y))).unwrap_or(Vec2::ZERO));
        } else {
            frame_rect(f, NOWHERE, NOWHERE + Vec2::ONE, 1.6, UNIT.a(0.0));
        }

        // ----- the counterfactual point, where the unit lands
        // (the unit is seen whole as soon as it sets off)
        let landed = if pr.work > 0.0 && pr.step <= 3 { (pr.work * 6.0).min(1.0) } else { pr.shown(pr.work) } * point;
        match in_range(map, DVec2::new(travel, top)) {
            Some(p) if landed > 0.0 => {
                f.circle(p, Vec3::Z, 8.0 * look.wpp, 2.4, UNIT.a(landed));
                f.dot(p, 7.0, UNIT.a(landed));
            }
            _ => {
                f.circle(NOWHERE, Vec3::Z, 8.0 * look.wpp, 2.4, UNIT.a(0.0));
                f.dot(NOWHERE, 7.0, UNIT.a(0.0));
            }
        }
        // off the picture: say where
        if landed > 0.5 && !(0.0..=Y_MAX).contains(&top) {
            let edge = if top > Y_MAX { Y_MAX } else { 0.0 };
            let (dir, align) = if top > Y_MAX { (-1.0, Align2::CENTER_TOP) } else { (1.0, Align2::CENTER_BOTTOM) };
            f.text_off(map.at(travel, edge), Vec2::new(0.0, dir * 4.0), format!("{} $y' = {}$", if top > Y_MAX { "↑" } else { "↓" }, num(top, 2)), 12.0, UNIT, align);
        }

        // ----- the strips: everyone at x, everyone at x′, and this unit among them
        let marks = [(0usize, y, pr.shown(pr.read)), (1, g.y2, pr.shown(pr.show) * point)];
        for (k, v, alpha) in marks {
            let Some(strip) = lay.strips[k] else { continue };
            let (lo, hi) = lay.on_stage(strip);
            let mean_v = if k == 0 { g.fx } else { g.fx2 };
            match in_range(map, DVec2::new(0.0, mean_v)) {
                Some(p) => f.line(Vec2::new(lo.x, p.y), Vec2::new(hi.x, p.y), 2.0, CURVE.a(0.9)),
                None => f.line(NOWHERE, NOWHERE, 2.0, CURVE.a(0.0)),
            }
            match in_range(map, DVec2::new(0.0, v)) {
                Some(p) if alpha > 0.0 => {
                    f.line(Vec2::new(lo.x, p.y), Vec2::new(hi.x, p.y), 2.4, UNIT.a(alpha));
                    f.dot(Vec2::new((lo.x + hi.x) * 0.5, p.y), 9.0, UNIT.a(alpha));
                }
                _ => {
                    f.line(NOWHERE, NOWHERE, 2.4, UNIT.a(0.0));
                    f.dot(NOWHERE, 9.0, UNIT.a(0.0));
                }
            }
            if k == 1 {
                // the band, across the strip
                let (r0, r1) = (map.at(0.0, b0).y, map.at(0.0, b1).y);
                if band > 0.0 && (r1 - r0) * lay.ppu > 0.5 {
                    f.dl.tris_alpha.rect(Vec2::new(lo.x, r0), Vec2::new(hi.x, r1), 0.0, UNIT.a(0.07 * band));
                    frame_rect(f, Vec2::new(lo.x, r0), Vec2::new(hi.x, r1), 1.4, UNIT.a(band));
                } else {
                    frame_rect(f, NOWHERE, NOWHERE + Vec2::ONE, 1.4, UNIT.a(0.0));
                }
            }
        }

        // ----- the unit as it was seen (its handle is drawn over this)
        if let Some(p) = in_range(map, g.unit) {
            f.circle(p, Vec3::Z, 8.0 * look.wpp, 1.6, UNIT.a(0.5));
        }
        // (names keep off the unit and off where it would have been)
        for at in [g.unit, DVec2::new(g.x2, g.y2)] {
            if let Some(p) = in_range(map, at).and_then(|p| on_screen(f, s3(p))) {
                room.boxed(p - Vec2::splat(9.0), p + Vec2::splat(9.0));
            }
        }

        // ----- names, where there is room for them (not while a setting changes)
        if !look.moving {
            self.names(f, &mut room, lay, map, g, look);
        }
    }

    /// The names of the things in the picture, each beside what it names.
    fn names(&self, f: &mut Frame, room: &mut Room, lay: &Layout, map: &Map, g: &Geometry, look: &Look) {
        let pr = look.progress;
        let sums = self.sums();
        let s3 = |p: Vec2| p.extend(0.0);
        let size = lay.size;
        let (x, y, x2) = (g.unit.x, g.unit.y, g.x2);
        // (Each label is written whether or not it finds a clear place, the short form
        // where the long one does not fit: the same labels in the same order from frame
        // to frame, so that a label is never taken for another.)
        let px_y = (map.k.y * lay.ppu) as f64;
        // the side of a point of the path away from its stroke: across the path on the
        // screen, away from the curve
        let across = |f: &Frame, t: f64, top: f64, foot: f64| {
            let at = |t: f64, v: f64| on_screen(f, s3(map.at(t, v)));
            let slope = (g.path_at((t + 0.1).min(X_MAX)) - g.path_at((t - 0.1).max(0.0))) / ((t + 0.1).min(X_MAX) - (t - 0.1).max(0.0));
            match (at(t, top), at(t + 0.1, top + 0.1 * slope), at(t, foot)) {
                (Some(a), Some(b), Some(c)) => {
                    let d = (b - a).normalize_or_zero();
                    let n = Vec2::new(d.y, -d.x);
                    if n.dot(c - a) > 0.0 {
                        -n
                    } else {
                        n
                    }
                }
                _ => Vec2::new(-1.0, -1.0),
            }
        };
        let (noise_text, short_noise) = match self.model.enters {
            Enters::Added => (format!("$n = {}$", sums.noise_text(Enters::Added)), "$n$"),
            Enters::Multiplied => (format!("$m = {}$", sums.noise_text(Enters::Multiplied)), "$m$"),
        };
        // beside a stroke, on the side where the curve falls away
        let beside = |t: f64| {
            let rise = g.curve_at((t + 0.2).min(X_MAX)) - g.curve_at((t - 0.2).max(0.0));
            Vec2::new(if rise >= 0.0 { -1.0 } else { 1.0 }, 0.0)
        };
        if pr.read > 0.5 && (y - g.fx).abs() * px_y > 4.0 {
            let mid = s3(map.to(DVec2::new(x, ((g.fx + y) * 0.5).clamp(0.0, Y_MAX))));
            if !room.label(f, "noise", mid, &noise_text, size, NOISE, beside(x), 7.0, true) {
                room.label(f, "noise", mid, short_noise, size + 1.0, NOISE, beside(x), 6.0, false);
            }
        }
        // the answer
        if pr.step >= 4 {
            match sums.answer {
                Ok(v) => {
                    if let Some(p) = in_range(map, DVec2::new(x2, g.y2)) {
                        // up (or down, away from its stroke) and to the left, clear of the
                        // line of x′ through it and of the path
                        // (always the short form: a name that changes its length as x′
                        // is dragged would jump about)
                        let up = if g.y2 >= g.fx2 { -0.5 } else { 0.5 };
                        room.label(f, "answer", s3(p), &format!("$y' = {}$", cents(v)), size + 1.0, UNIT, Vec2::new(-1.0, up), 14.0, false);
                    }
                    // the same noise, at x′
                    if g.point > 0.5 && !sums.same && (g.y2 - g.fx2).abs() * px_y > 16.0 {
                        let mid = DVec2::new(x2, ((g.fx2 + g.y2) * 0.5).clamp(0.0, Y_MAX));
                        room.label(f, "copy", s3(map.to(mid)), &format!("same {short_noise}"), size, NOISE, beside(x2), 7.0, false);
                    }
                }
                Err(_) => {
                    if let Some(p) = in_range(map, DVec2::new(x2, g.band[1].min(Y_MAX))) {
                        room.label(f, "answer", s3(p), "$y'$: in here, 9 chances in 10", size, UNIT, Vec2::new(1.0, -1.0), 10.0, false);
                    }
                }
            }
        }
        // the unit as seen, across its path from its stroke (a short name: one that
        // moves to another side as the unit is dragged must not have far to go)
        if let Some(p) = in_range(map, g.unit) {
            room.label(f, "unit", s3(p), "$(x, y)$", size + 1.0, UNIT, across(f, x, y, g.fx), 10.0, false);
        }
        // x and x′, at the ends of the arrow of the change between them
        let level = map.at(0.0, 0.0).y + (24.0 - Layout::PAD) / lay.ppu;
        let (to_x2, to_x) = if x2 >= x { (Align2::LEFT_CENTER, Align2::RIGHT_CENTER) } else { (Align2::RIGHT_CENTER, Align2::LEFT_CENTER) };
        let out = if x2 >= x { 1.0 } else { -1.0 };
        if sums.same {
            f.text_off(Vec2::new(map.at(x2, 0.0).x, level), Vec2::new(8.0, 0.0), "$x' = x$", 15.0, XS, Align2::LEFT_CENTER);
        } else {
            f.text_off(Vec2::new(map.at(x2, 0.0).x, level), Vec2::new(8.0 * out, 0.0), "$x'$", 15.0, XS, to_x2);
            f.text_off(Vec2::new(map.at(x, 0.0).x, level), Vec2::new(-8.0 * out, 0.0), "$x$", 15.0, XS, to_x);
        }
        // the curve and the path, at their right-hand ends, each on its side of the other
        let end = g.curve.last().copied().unwrap_or(DVec2::ZERO);
        let path_end = g.path.last().copied().unwrap_or(end);
        let below = if path_end.y >= end.y { 1.0 } else { -1.0 };
        if let Some(p) = in_range(map, end) {
            room.label(f, "f", s3(p), "$f(X)$", size + 1.0, CURVE, Vec2::new(-0.3, below), 10.0, false);
        }
        if pr.step >= 4 && g.point > 0.5 && !sums.same {
            if let Some(p) = in_range(map, path_end) {
                // (the last of the names, and the only one left out where it has no room:
                // the key says what the dots are)
                room.label(f, "path", s3(p), "its $y'$ for every $x'$", size - 1.0, UNIT, Vec2::new(-0.3, -below), 10.0, true);
            }
        }
        // the strips' titles and this unit's rank in each
        let z = self.model.noise_of(self.unit.x, self.unit.y);
        let above = self.pop.below(z);
        for (k, strip) in lay.strips.iter().enumerate() {
            let Some(strip) = strip else { continue };
            let title = if k == 0 { "at $X = x$" } else { "at $X = x'$" };
            let top = Vec2::new((strip.min.x + strip.max.x) * 0.5, strip.min.y - 5.0);
            f.text_px(top, title, size - 1.0, pal::DIM, Align2::CENTER_BOTTOM);
            f.text_px(top - Vec2::new(0.0, size + 3.0), "everyone", size - 1.0, pal::DIM, Align2::CENTER_BOTTOM);
            let shown = if k == 0 { pr.read } else { pr.show };
            if shown <= 0.5 {
                continue;
            }
            let (rank, of) = match (k, sums.answer) {
                (1, Err(_)) => ("anywhere".to_string(), "among them".to_string()),
                _ => (format!("above {above}"), format!("of {UNITS}")),
            };
            let at = Vec2::new((strip.min.x + strip.max.x) * 0.5, strip.max.y + 6.0);
            f.text_px(at, rank, size - 1.0, UNIT, Align2::CENTER_TOP);
            f.text_px(at + Vec2::new(0.0, size + 3.0), of, size - 1.0, pal::DIM, Align2::CENTER_TOP);
        }
    }

    /// The steps, one to a line, with their numbers: the one playing bright, those to
    /// come not yet written.
    fn steps(&self, f: &mut Frame, lay: &Layout, look: &Look) {
        let Some(r) = lay.steps else { return };
        let (lo, hi) = lay.on_stage(r);
        f.dl.tris_alpha.rect(lo, hi, 0.0, Color::hex(0x0B0F1A).a(0.9));
        frame_rect(f, lo, hi, 1.0, pal::FAINT.a(0.7));
        let pr = look.progress;
        let sums = self.sums();
        let size = lay.size;
        let enters = self.model.enters;
        let tint = |c: Color, body: &str| kit_tint(c, body);
        let noise_name = match enters {
            Enters::Added => tint(NOISE, "n"),
            Enters::Multiplied => tint(NOISE, "m"),
        };
        let read = match enters {
            Enters::Added => format!("${} = y - f(x) = {} - {} = {}$", noise_name, cents(sums.y), cents(sums.fx), tint(NOISE, &cents(sums.noise))),
            Enters::Multiplied => format!("${} = y / f(x) = {}$", noise_name, tint(NOISE, &sums.noise_text(enters))),
        };
        let set = format!("$x = {} \\to {} = {}$", cents(sums.x), tint(XS, "x'"), tint(XS, &cents(sums.x2)));
        let y2 = tint(UNIT, "y'");
        let work = match sums.answer {
            _ if sums.same => format!("$x' = x$: nothing is changed, so ${y2} = y = {}$", cents(sums.y)),
            Ok(v) => match enters {
                Enters::Added => format!("${y2} = f(x') + {noise_name} = {} + {} = {}$", cents(sums.fx2), cents(sums.noise), tint(UNIT, &cents(v))),
                Enters::Multiplied => format!("${y2} = {noise_name}\\,f(x') = {} \\times {} = {}$", sums.noise_text(enters), cents(sums.fx2), tint(UNIT, &cents(v))),
            },
            Err((a, b)) => format!("its noise at $x'$ is new: ${y2}$ is from ${}$ to ${}$, with 9 chances in 10", cents(a), cents(b)),
        };
        let title = "Had $X$ been $x'$, what would this unit's $Y$ have been?";
        let names = ["1  Read off its noise", "2  Set $X$ to $x'$, nothing else", if self.model.keeps { "3  Work out $Y$ again, same noise" } else { "3  Work out $Y$ again" }];
        let formulas = [read, set, work];
        let color = |k: usize| {
            if pr.step < k {
                None
            } else if pr.step == k || pr.step >= 4 {
                Some(pal::TEXT)
            } else {
                Some(pal::DIM)
            }
        };
        // (placed through the stage, so that the words move with their box)
        let shift = on_screen(f, lay.stage(r.min).extend(0.0)).map_or(Vec2::ZERO, |p| p - r.min);
        let left = r.min.x + 14.0;
        let width = r.w() - 28.0;
        let line = size + 8.0;
        let mut y = r.min.y + 10.0;
        // (the title keeps clear of the play button, at the top right)
        let beside = lay.play.filter(|b| b.min.y < r.min.y + 40.0).map_or(width, |b| b.min.x - 10.0 - left);
        for text in wrap(f, title, size, beside) {
            f.text_px(Vec2::new(left, y) + shift, text, size, pal::TEXT, Align2::LEFT_TOP);
            y += line;
        }
        y += 6.0;
        // on a wide canvas each step on one line, its sum in a column; else the sum
        // under the step's name
        let column = (r.w() * 0.36).clamp(180.0, 300.0);
        let one_line = width >= 540.0;
        for k in 0..3 {
            let sum = if one_line { vec![formulas[k].clone()] } else { wrap(f, &formulas[k], size, width - 16.0) };
            let Some(c) = color(k + 1) else {
                y += if one_line { line } else { line * (1 + sum.len()) as f32 };
                continue;
            };
            if pr.step == k + 1 {
                f.text_px(Vec2::new(left - 8.0, y + size * 0.5) + shift, "▸", size, pal::TEXT, Align2::CENTER_CENTER);
            }
            f.text_px(Vec2::new(left, y) + shift, names[k], size, c, Align2::LEFT_TOP);
            if one_line {
                f.text_px(Vec2::new(left + column, y) + shift, formulas[k].as_str(), size, c, Align2::LEFT_TOP);
                y += line;
            } else {
                y += line;
                for text in sum {
                    f.text_px(Vec2::new(left + 16.0, y - 2.0) + shift, text, size, c, Align2::LEFT_TOP);
                    y += line;
                }
            }
        }
        // the average change and this unit's, once it is all done
        if pr.step >= 4 {
            let average = sums.fx2 - sums.fx;
            let theirs = match sums.answer {
                Ok(v) => format!("this unit's ${} - y = {}$", y2, tint(UNIT, &signed_cents(v - sums.y))),
                Err((a, b)) => format!("this unit's: from ${}$ to ${}$, 9 chances in 10", signed_cents(a - sums.y), signed_cents(b - sums.y)),
            };
            let text = format!("Average change ${} = {}$;   {theirs}", tint(CURVE, "f(x') - f(x)"), signed_cents(average));
            y += 4.0;
            for text in wrap(f, &text, size - 0.5, width) {
                if y + size > r.max.y {
                    break;
                }
                f.text_px(Vec2::new(left, y) + shift, text, size - 0.5, pal::DIM, Align2::LEFT_TOP);
                y += line - 1.0;
            }
        }
    }

    /// The model: its graph, and its equation.
    fn card(&self, f: &mut Frame, lay: &Layout) {
        let Some(r) = lay.card else { return };
        let (lo, hi) = lay.on_stage(r);
        f.dl.tris_alpha.rect(lo, hi, 0.0, Color::hex(0x0B0F1A).a(0.9));
        frame_rect(f, lo, hi, 1.0, pal::FAINT.a(0.7));
        // (placed through the stage, so that it moves with the picture)
        let shift = on_screen(f, lay.stage(r.min).extend(0.0)).map_or(Vec2::ZERO, |p| p - r.min);
        let at = |px: Vec2| px + shift;
        let mid = (r.min.x + r.max.x) * 0.5;
        let noise = match self.model.enters {
            Enters::Added => "N",
            Enters::Multiplied => "M",
        };
        let nodes = [
            Node { at: at(Vec2::new(mid - 60.0, r.min.y + 30.0)), name: "X", color: XS },
            Node { at: at(Vec2::new(mid, r.min.y + 30.0)), name: "Y", color: UNIT },
            Node { at: at(Vec2::new(mid + 60.0, r.min.y + 30.0)), name: noise, color: NOISE },
        ];
        hud_graph(f, &nodes, &[(0, 1, Link::Arrow, CURVE, 2.0), (2, 1, Link::Arrow, NOISE, 2.0)], 14.0);
        let equation = match (self.model.enters, self.model.keeps) {
            (Enters::Added, true) => "$Y = f(X) + N$",
            (Enters::Multiplied, true) => r"$Y = f(X)\,M$",
            (Enters::Added, false) => "$Y = f(X) + N_X$",
            (Enters::Multiplied, false) => r"$Y = f(X)\,M_X$",
        };
        let note = if self.model.keeps { "the same noise at every $x$" } else { "a new noise for every $x$" };
        f.text_px(at(Vec2::new(mid, r.min.y + 58.0)), equation, 14.0, pal::TEXT, Align2::CENTER_TOP);
        f.text_px(at(Vec2::new(mid, r.min.y + 82.0)), note, 11.5, pal::DIM, Align2::CENTER_TOP);
        f.text_px(at(Vec2::new(mid, r.min.y + 100.0)), "the model", 11.0, pal::FAINT.mix(pal::DIM, 0.5), Align2::CENTER_TOP);
    }
}

/// `kit::tint`, by a name that does not shadow the closures that use it.
fn kit_tint(c: Color, body: &str) -> String {
    tint(c, body)
}

/// The words of a text, each formula kept whole.
fn words(text: &str) -> Vec<&str> {
    let (mut out, mut start, mut inside) = (Vec::new(), 0, false);
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 1,
            b'$' => inside = !inside,
            b' ' if !inside => {
                if i > start {
                    out.push(&text[start..i]);
                }
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// A text broken into lines no wider than `room` logical pixels, as the frame measures
/// them, formulas kept whole.
fn wrap(f: &Frame, text: &str, size: f32, room: f32) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for word in words(text) {
        match lines.last_mut() {
            Some(last) if f.text_size(&format!("{last} {word}"), size).x <= room => {
                last.push(' ');
                last.push_str(word);
            }
            _ => lines.push(word.to_string()),
        }
    }
    lines
}

/// A button drawn over the picture (logical pixels): true when it is clicked.
fn hud_button(f: &mut Frame, r: Rect, text: &str, pointer: Option<Vec2>) -> bool {
    let over = pointer.is_some_and(|p| r.contains(p));
    hud_rect(f, r.min, r.max, if over { Color::hex(0x33405E) } else { Color::hex(0x222B40) });
    f.text_px((r.min + r.max) * 0.5, text, 12.0, pal::TEXT, Align2::CENTER_CENTER);
    over && f.ptr.clicked
}

impl Scene for Counterfactual {
    fn view(&self) -> View {
        STAGE_VIEW
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    /// Moving only while the steps play, a setting changes, or a snap closes up.
    fn animated(&self) -> bool {
        self.play.is_some() || self.change.is_some() || self.glide_unit != DVec2::ZERO || self.glide_x2 != 0.0
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("orange dot: the unit, as it was seen    cyan dot: the other $X$ asked about    drag elsewhere: move    scroll: zoom")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A counterfactual asks about one unit, after the fact. This unit had $X = x$, and its $Y$ came out as $y$: what would its $Y$ have been had $X$ been $x'$? (A patient given dose $x$ of a drug, whose blood pressure fell by $y$: how far would it have fallen at dose $x'$?) That is not the average $Y$ when $X$ is set to $x'$ for everyone (an intervention): that answer is the same for every unit, and this one keeps whatever made this unit different.");
        ui::note(ui, r"With a model of how $Y$ is made, here $Y = f(X) + N$, where $N$ is the unit's own noise (all its other causes), it takes three steps:");
        ui::note(ui, r"1. Read its noise off what was seen: $n = y - f(x)$.");
        ui::note(ui, r"2. Set $X$ to $x'$, and change nothing else.");
        ui::note(ui, r"3. Work out $Y$ again with the same noise: $y' = f(x') + n$.");
        ui::note(ui, r"(In the model here nothing that sets $X$ also acts on $Y$, as when a trial deals out the doses, so the curve through the scatter is $f$.) Drag the orange dot (the unit, as it was seen) and the cyan dot under the picture.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The three steps");
        if ui.button("▶ play them again").clicked() {
            self.replay(false);
        }
        ui::note(ui, "With their names, in the box over the picture, and the sums for the model chosen below.");
        ui::section(ui, "The model");
        let from = self.shown();
        let mut shape = Shape::ALL.iter().position(|s| *s == self.model.shape).unwrap_or(0);
        let names: Vec<&str> = Shape::ALL.iter().map(|s| s.name()).collect();
        let mut changed = false;
        ui::note(ui, r"The curve $f$: the average $Y$ when $X$ is set to $x$ for everyone.");
        if ui::choice(ui, &names, &mut shape) {
            self.model.shape = Shape::ALL[shape];
            changed = true;
        }
        let mut enters = if self.model.enters == Enters::Added { 0 } else { 1 };
        ui::note(ui, "How the noise enters:");
        if ui::choice(ui, &[r"added: $Y = f(X) + N$", r"multiplied: $Y = f(X)\,M$"], &mut enters) {
            self.model.enters = if enters == 0 { Enters::Added } else { Enters::Multiplied };
            changed = true;
        }
        ui::note(ui, r"Added, a unit is the curve moved up or down by its $N$. Multiplied, it is the curve times its own factor $M$ (on average 1), so a unit with a large $M$ responds more to $X$.");
        let mut keeps = if self.model.keeps { 0 } else { 1 };
        ui::note(ui, r"Its noise at another $x$:");
        if ui::choice(ui, &["the same", "drawn afresh"], &mut keeps) {
            self.model.keeps = keeps == 0;
            changed = true;
        }
        ui::note(ui, r"No data can say which: each unit is seen at one $X$ only, and both give the same data under every intervention.");
        let mut others = self.show_others;
        ui.checkbox(&mut others, "show nine other units' paths");
        if others != self.show_others {
            self.show_others = others;
            changed = true;
        }
        if changed {
            // (the unit keeps what was seen of it: its X and its Y)
            self.unit.y = self.unit.y.max(Y_LEAST);
            self.changed(from);
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (CROWD, "white dots: the units, each seen at one $X$; in the strips, the spread of everyone's $Y$ with $X$ set to $x$, and to $x'$"),
                (CURVE, r"green: $f$, the average $Y$ when $X$ is set; in the strips, the average there"),
                (UNIT, r"orange: this unit as it was seen, $(x, y)$, and where it would have been, $(x', y')$; dotted, its $y'$ for every $x'$; a bar, where it would have been with 9 chances in 10, when its noise is drawn afresh"),
                (NOISE, r"purple: its noise, from the curve to the unit, and the same noise at $x'$"),
                (XS, r"cyan: its $x$, and the $x'$ asked about (drag it); the arrow, $X$ being set"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Unit:", "one member of the population: a patient, a pupil, a field, a day.");
        ui::term(ui, "Counterfactual:", "what would have happened to a unit had one thing been different, given what did happen to it.");
        ui::term(ui, "Noise:", r"what the model leaves unexplained in a unit's $Y$: all the other causes that set it apart from the average at its $X$.");
        ui::term(ui, "Intervention:", r"setting $X$ to a value for everyone, whatever it would have been. Its answer is about the population: the strip at $x'$.");
        ui::term(ui, "Abduction, action, prediction:", "the usual names of the three steps (Judea Pearl's): the noise worked out from what was seen; $X$ changed; $Y$ worked out again.");
        ui::term(ui, "Average effect:", r"$f(x') - f(x)$: how much the average $Y$ changes when $X$ is changed for everyone. A unit's own effect is $y' - y$.");

        ui::section(ui, "Numbers");
        let sums = self.sums();
        let z = self.model.noise_of(self.unit.x, self.unit.y);
        ui::readout_colored(ui, "seen: $x$, $y$", format!("${}$, ${}$", cents(sums.x), cents(sums.y)), UNIT);
        match self.model.enters {
            Enters::Added => ui::readout_colored(ui, "its noise $n$", format!("${}$", cents(sums.noise)), NOISE),
            Enters::Multiplied => ui::readout_colored(ui, "its factor $m$", format!("${}$", sums.noise_text(Enters::Multiplied)), NOISE),
        }
        ui::readout_colored(ui, "$x'$", format!("${}$", cents(sums.x2)), XS);
        match sums.answer {
            Ok(v) => ui::readout_colored(ui, "would have been $y'$", format!("${}$", cents(v)), UNIT),
            Err((a, b)) => ui::readout_colored(ui, "$y'$, 9 chances in 10", format!("${}$ to ${}$", cents(a), cents(b)), UNIT),
        }
        ui::readout_colored(ui, "average at $x'$, $f(x')$", format!("${}$", cents(sums.fx2)), CURVE);
        ui::readout(ui, "above how many of the 200, given the same $X$", match sums.answer {
            Err(_) => format!("{} at $x$; any at $x'$", self.pop.below(z)),
            Ok(_) => format!("{}, at $x$ and at $x'$", self.pop.below(z)),
        });
        let notes = self.notes();
        for note in notes {
            ui::note(ui, &note);
        }
        ui::see_also(
            ui,
            &[
                ("dosee", "seeing and doing: the population's answer"),
                ("confound", "correlation is not causation"),
                ("direction", "which way the arrow points, from the same kind of model"),
                ("regress", "fitting a curve to a scatter"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset);
        let map = lay.map();
        self.advance(f.dt, f.real_dt);
        let pointer = f.ptr.pos.map(|p| p / f.px);
        // the play button: its own clicks, and no moving the picture from there
        let mut pressed = false;
        if let Some(b) = lay.play {
            let b = match on_screen(f, lay.stage(b.min).extend(0.0)) {
                Some(at) => Rect::new(at, at + (b.max - b.min)),
                None => b,
            };
            if pointer.is_some_and(|p| b.contains(p)) {
                f.handles.capture();
                f.handles.claim_clicks();
            }
            pressed = hud_button(f, b, if lay.narrow { "▶ play the steps" } else { "▶ play the three steps" }, pointer);
        }
        if pressed {
            self.replay(false);
        }
        self.handles(f, &lay, &map);
        let g = self.shown();
        let look = Look { progress: Progress::of(self.play), moving: self.change.is_some(), wpp: 1.0 / pixels_per_unit(f) };
        self.draw(f, &lay, &map, &g, &look);
        self.steps(f, &lay, &look);
        self.card(f, &lay);
        if lay.caption {
            bottom_caption(f, &["white: the units, each seen once", "green: the average when $X$ is set", "purple: this unit's noise", "orange: this unit, and what it would have been"], 12.0);
        }
    }
}

impl Counterfactual {
    /// What the setting comes to, in words: the cases where the picture says something
    /// special.
    fn notes(&self) -> Vec<String> {
        let sums = self.sums();
        let mut out = Vec::new();
        if sums.same {
            out.push(r"$x' = x$: nothing is changed, so what would have happened is what did happen, in every model.".to_string());
            return out;
        }
        if self.model.shape == Shape::Flat {
            out.push(if self.model.keeps {
                r"The curve is flat: setting $X$ changes nothing on average, and with its noise kept, nothing for this unit either: $y' = y$, whatever $x'$ is.".to_string()
            } else {
                r"The curve is flat: setting $X$ changes nothing on average. Yet with the noise drawn afresh, changing $X$ changes every unit's $Y$, in ways that cancel out over the population. The data cannot tell this from no effect at all.".to_string()
            });
        }
        // (exactly on it: the handle puts it there when it is dragged close)
        if self.model.keeps && self.unit.y == self.model.f(self.unit.x) {
            out.push(r"This unit is on the curve, exactly average at its $x$: its path is the curve itself.".to_string());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, with the change of setting and the
    /// steps played to the end at 60 frames a second.
    fn run(k: usize) -> Counterfactual {
        let mut s = Counterfactual::create();
        s.set_up(k);
        for _ in 0..1000 {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        assert!(s.change.is_none() && s.play.is_none() && !s.animated());
        s
    }

    /// How many of the population a unit with noise z is above.
    fn rank(s: &Counterfactual) -> usize {
        s.pop.below(s.model.noise_of(s.unit.x, s.unit.y))
    }

    /// The captions of the "things to try", held to account: every number they quote,
    /// as the scene writes it.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // the three steps: n = 0.50 above the curve at x = 1.20, y = 3.72; y′ = 4.83 at 4.40
        let s = run(0);
        let sums = s.sums();
        assert_eq!((sums.x, sums.y, sums.fx, sums.noise, sums.x2, sums.fx2, sums.answer), (120, 372, 322, 50, 440, 433, Ok(483)), "{sums:?}");
        assert!((s.goal().path_at(s.x2) - s.goal().curve_at(s.x2) - (s.unit.y - s.model.f(s.unit.x))).abs() < 1e-9);
        // and the scene opens on the same setting
        let opened = Counterfactual::create();
        assert_eq!((opened.model, opened.unit, opened.x2), (s.model, s.unit, s.x2));

        // a lucky unit: above 195 of 200 at x = 4.70 and at x′ = 1.00, its Y falling to 4.15
        let s = run(1);
        let sums = s.sums();
        assert_eq!((sums.x, sums.y, sums.noise, sums.x2, sums.answer), (470, 546, 110, 100, Ok(415)), "{sums:?}");
        let g = s.goal();
        let above = |values: &[f64], v: f64| values.iter().filter(|w| **w < v).count();
        assert_eq!((rank(&s), above(&g.at_x, s.unit.y), above(&g.at_x2, g.y2)), (195, 195, 195));
        assert!(g.y2 < s.unit.y && g.y2 - g.fx2 > 1.0 && (g.y2 - g.fx2 - (s.unit.y - g.fx)).abs() < 1e-9);

        // noise that multiplies: m = 1.200, the path the curve stretched (not moved), the
        // other paths fanning out, and the changes +1.56 against +1.30 on average
        let s = run(2);
        let sums = s.sums();
        assert_eq!((sums.x, sums.y, sums.fx, sums.noise, sums.x2, sums.fx2, sums.answer), (100, 366, 305, 1200, 460, 435, Ok(522)), "{sums:?}");
        let ok = sums.answer.unwrap_or(0);
        assert_eq!((ok - sums.y, sums.fx2 - sums.fx), (156, 130));
        let g = s.goal();
        assert!(s.show_others && g.others_shown == 1.0);
        let ratio: Vec<f64> = g.path.iter().zip(&g.curve).map(|(p, c)| p.y / c.y).collect();
        let gap: Vec<f64> = g.path.iter().zip(&g.curve).map(|(p, c)| p.y - c.y).collect();
        assert!(ratio.iter().all(|r| (r - ratio[0]).abs() < 1e-9) && gap.iter().fold(0.0f64, |m, v| m.max(*v)) > gap.iter().fold(9.0f64, |m, v| m.min(*v)) + 0.15);
        // (the other paths fan out: their gaps from one another grow with the curve)
        let spread = |i: usize| g.others.iter().map(|p| p[i].y).fold(f64::MIN, f64::max) - g.others.iter().map(|p| p[i].y).fold(f64::MAX, f64::min);
        assert!(spread(SAMPLES - 1) > 1.3 * spread(0), "{} against {}", spread(SAMPLES - 1), spread(0));
        // and with the noise added instead, every unit's change is the same
        let mut added = run(2);
        added.model.enters = Enters::Added;
        let g = added.goal();
        let change = |p: &Vec<DVec2>| p[SAMPLES - 1].y - p[0].y;
        assert!(g.others.iter().all(|p| (change(p) - change(&g.curve)).abs() < 1e-9));

        // the individual and the average: +1.89 on average, +1.42 for this unit (m = 0.75)
        let s = run(3);
        let sums = s.sums();
        assert_eq!((sums.x, sums.y, sums.fx, sums.noise, sums.x2, sums.fx2, sums.answer), (50, 189, 252, 750, 550, 441, Ok(331)), "{sums:?}");
        let ok = sums.answer.unwrap_or(0);
        assert_eq!((sums.fx2 - sums.fx, ok - sums.y), (189, 142));
        assert!(((ok - sums.y) as f64 / (sums.fx2 - sums.fx) as f64 - 0.75).abs() < 0.005);

        // same data, different answers: the scatter of the first experiment, dot for dot,
        // and y′ between 3.42 and 5.23 with 9 chances in 10
        let s = run(4);
        let sums = s.sums();
        assert_eq!((sums.y, sums.x2, sums.answer), (372, 440, Err((342, 523))), "{sums:?}");
        assert_eq!(s.goal().units, run(0).goal().units);
        let mut back = run(4);
        back.model.keeps = true;
        assert_eq!(back.goal().units, s.goal().units);
        assert_eq!(back.sums().answer, Ok(483));
    }

    /// The three steps are the model: with the noise kept, a unit keeps its distance
    /// from the curve (added) or its ratio to it (multiplied), at every x′; its path
    /// goes through what was seen and through the answer; and its rank among everyone
    /// given the same X is the same at x and at x′.
    #[test]
    fn a_kept_noise_is_kept_at_every_x() {
        let mut rng = Rng::new(3);
        let pop = Population::new();
        for shape in Shape::ALL {
            for enters in [Enters::Added, Enters::Multiplied] {
                let model = Model { shape, enters, keeps: true };
                for _ in 0..50 {
                    let x = rng.range(0.0, X_MAX as f32) as f64;
                    let y = rng.range(Y_LEAST as f32, Y_MAX as f32) as f64;
                    let x2 = rng.range(0.0, X_MAX as f32) as f64;
                    let Answer::Point(v) = model.answer(x, y, x2) else { panic!("{model:?}: no single answer") };
                    match enters {
                        Enters::Added => assert!((v - model.f(x2) - (y - model.f(x))).abs() < 1e-9),
                        Enters::Multiplied => assert!((v / model.f(x2) - y / model.f(x)).abs() < 1e-9),
                    }
                    let g = Geometry::of(&model, &pop, DVec2::new(x, y), x2, false);
                    assert!((g.path_at(x) - y).abs() < 0.02 * y.abs().max(1.0) && (g.path_at(x2) - v).abs() < 0.02 * v.abs().max(1.0), "{model:?} at {x}, {y} to {x2}");
                    let above = |values: &[f64], v: f64| values.iter().filter(|w| **w < v).count();
                    assert_eq!(above(&g.at_x, y), above(&g.at_x2, v), "{model:?} at {x}, {y} to {x2}");
                    assert_eq!(above(&g.at_x, y), pop.below(model.noise_of(x, y)));
                }
            }
        }
    }

    /// With the noise drawn afresh, the answer is the population's spread at x′ (9 in
    /// 10 of everyone given x′ fall in the band), and the data are the same as with
    /// the noise kept, under every intervention: for every unit, at every x.
    #[test]
    fn fresh_noise_gives_the_same_data_and_the_population_s_odds() {
        let pop = Population::new();
        for shape in Shape::ALL {
            for enters in [Enters::Added, Enters::Multiplied] {
                let kept = Model { shape, enters, keeps: true };
                let fresh = Model { keeps: false, ..kept };
                for x2 in [0.0, 2.5, 6.0] {
                    let (a, b) = (Geometry::of(&kept, &pop, DVec2::new(1.0, 3.0), x2, false), Geometry::of(&fresh, &pop, DVec2::new(1.0, 3.0), x2, false));
                    assert!(a.units == b.units && a.at_x == b.at_x && a.at_x2 == b.at_x2 && a.curve == b.curve);
                    let Answer::Spread(lo, hi) = fresh.answer(1.0, 3.0, x2) else { panic!("one answer under {fresh:?}") };
                    let inside = b.at_x2.iter().filter(|v| **v >= lo && **v <= hi).count();
                    assert!((174..=186).contains(&inside), "{fresh:?} at {x2}: {inside} of 200 in the band");
                    assert!(lo < b.fx2 && b.fx2 < hi && b.point == 0.0);
                }
            }
        }
    }

    /// The population: its noise exactly mean 0 and spread 1 (so that, added, the
    /// average of everyone at any x is the curve itself), and every unit within the
    /// picture, whatever the model and the X it is given.
    #[test]
    fn the_population_is_all_in_the_picture() {
        let pop = Population::new();
        assert!(mean(&pop.z).abs() < 1e-12 && (variance(&pop.z) - 1.0).abs() < 1e-12);
        for shape in Shape::ALL {
            for enters in [Enters::Added, Enters::Multiplied] {
                let model = Model { shape, enters, keeps: true };
                for i in 0..SAMPLES {
                    let x = sample_x(i);
                    let column: Vec<f64> = pop.z.iter().map(|z| model.y(x, *z)).collect();
                    assert!(column.iter().all(|v| *v > 0.1 && *v < Y_MAX - 0.1), "{model:?} at {x}: {:?}", column.iter().fold((9.0f64, 0.0f64), |m, v| (m.0.min(*v), m.1.max(*v))));
                    if enters == Enters::Added {
                        assert!((mean(&column) - model.f(x)).abs() < 1e-12);
                    }
                }
            }
        }
        // the nine whose paths can be shown are at evenly spaced ranks, lowest to highest
        let ranks: Vec<usize> = pop.others.iter().map(|&k| pop.below(pop.z[k])).collect();
        assert_eq!(ranks, (0..OTHERS).map(|k| (2 * k + 1) * UNITS / (2 * OTHERS)).collect::<Vec<_>>());
    }

    /// The sums written on the screen add up as written, and differ from the numbers
    /// the picture uses by less than the last place written can show.
    #[test]
    fn the_sums_on_the_screen_add_up() {
        let mut rng = Rng::new(9);
        for _ in 0..2000 {
            let model = Model { shape: Shape::ALL[rng.below(4)], enters: if rng.bool(0.5) { Enters::Added } else { Enters::Multiplied }, keeps: true };
            let unit = DVec2::new(rng.range(0.0, 6.0) as f64, rng.range(Y_LEAST as f32, Y_MAX as f32) as f64);
            let x2 = rng.range(0.0, 6.0) as f64;
            let sums = Sums::of(&model, unit, x2);
            let Ok(v) = sums.answer else { panic!("{sums:?}") };
            let Answer::Point(exact) = model.answer(unit.x, unit.y, x2) else { panic!() };
            match model.enters {
                Enters::Added => {
                    assert!(sums.y - sums.fx == sums.noise && sums.fx2 + sums.noise == v);
                    assert!((v as f64 / 100.0 - exact).abs() <= 0.015 + 1e-9, "{sums:?}: {exact}");
                }
                Enters::Multiplied => {
                    assert_eq!(sums.noise, ((sums.y as f64 / sums.fx as f64) * 1000.0).round() as i64);
                    assert_eq!(v, (sums.noise as f64 * sums.fx2 as f64 / 1000.0).round() as i64);
                    // (a factor read off two rounded numbers, times a third)
                    assert!((v as f64 / 100.0 - exact).abs() <= 0.015 + exact.abs() * 0.005, "{sums:?}: {exact}");
                }
            }
        }
        assert_eq!((cents(-5), cents(1234), signed_cents(5), signed_cents(0), signed_cents(-120)), ("−0.05".to_string(), "12.34".to_string(), "+0.05".to_string(), "0.00".to_string(), "−1.20".to_string()));
    }

    /// The degenerate settings: x′ on x (nothing is changed, in any model), a unit on
    /// the curve (its path is the curve), a flat curve (with the noise kept nothing
    /// changes for anyone); and the unit and x′ at the ends of their ranges, where every
    /// number drawn or written stays finite.
    #[test]
    fn degenerate_settings_say_what_has_happened() {
        let pop = Population::new();
        for shape in Shape::ALL {
            for enters in [Enters::Added, Enters::Multiplied] {
                for keeps in [true, false] {
                    let model = Model { shape, enters, keeps };
                    assert_eq!(model.answer(2.0, 3.3, 2.0), Answer::Point(3.3));
                    let mut s = Counterfactual::create();
                    (s.model, s.unit, s.x2) = (model, DVec2::new(2.0, 3.3), 2.0);
                    assert!(s.sums().same && s.sums().answer == Ok(330) && s.notes()[0].contains("nothing is changed"));
                    // on the curve: the path is the curve itself
                    if keeps {
                        let on = DVec2::new(2.0, model.f(2.0));
                        let g = Geometry::of(&model, &pop, on, 4.0, false);
                        assert!(g.path.iter().zip(&g.curve).all(|(p, c)| (p.y - c.y).abs() < 1e-9));
                        s.unit = on;
                        s.x2 = 4.0;
                        assert!(s.notes().iter().any(|n| n.contains("exactly average")), "{:?}", s.notes());
                    }
                    for unit in [DVec2::new(0.0, Y_LEAST), DVec2::new(X_MAX, Y_MAX), DVec2::new(0.0, Y_MAX), DVec2::new(X_MAX, Y_LEAST)] {
                        for x2 in [0.0, X_MAX, 3.0] {
                            let g = Geometry::of(&model, &pop, unit, x2, true);
                            let finite = |v: &[DVec2]| v.iter().all(|p| p.is_finite());
                            assert!(finite(&g.curve) && finite(&g.units) && finite(&g.path) && g.others.iter().all(|p| finite(p)), "{model:?} {unit} {x2}");
                            assert!(g.y2.is_finite() && g.band.iter().all(|v| v.is_finite()) && g.at_x.iter().chain(&g.at_x2).all(|v| v.is_finite()));
                            let sums = Sums::of(&model, unit, x2);
                            assert!(!format!("{sums:?}").contains("NaN"));
                        }
                    }
                }
            }
            // flat: with the noise kept, y′ = y whatever x′ is
            let flat = Model { shape: Shape::Flat, enters: Enters::Added, keeps: true };
            assert_eq!(flat.answer(1.0, 3.7, 5.0), Answer::Point(3.7));
            let flat = Model { enters: Enters::Multiplied, ..flat };
            let Answer::Point(v) = flat.answer(1.0, 3.7, 5.0) else { panic!() };
            assert!((v - 3.7).abs() < 1e-12);
        }
        let mut s = Counterfactual::create();
        s.model.shape = Shape::Flat;
        assert!(s.notes().iter().any(|n| n.contains("nothing for this unit")));
        s.model.keeps = false;
        assert!(s.notes().iter().any(|n| n.contains("changes every unit")));
    }

    /// A unit dragged within a few pixels of the curve is put on it, and x′ within a
    /// few pixels of x on x; a little further, they are where the hand put them. Either
    /// way the picture closes up on them, and comes to rest.
    #[test]
    fn snaps_put_things_exactly_in_place() {
        let mut s = Counterfactual::create();
        s.play = None;
        let per_px = 0.01;
        let x = 2.0;
        s.put_unit(DVec2::new(x, s.model.f(x) + 0.05), per_px);
        assert_eq!(s.unit.y, s.model.f(x));
        s.put_unit(DVec2::new(x, s.model.f(x) + 0.09), per_px);
        assert!((s.unit.y - s.model.f(x) - 0.09).abs() < 1e-12);
        s.put_x2(x + 0.06, per_px);
        assert_eq!(s.x2, x);
        s.put_x2(x + 0.3, per_px);
        assert!((s.x2 - x - 0.3).abs() < 1e-12);
        // out of range: kept in
        s.put_unit(DVec2::new(-3.0, -5.0), per_px);
        assert_eq!(s.unit, DVec2::new(0.0, Y_LEAST));
        s.put_x2(99.0, per_px);
        assert_eq!(s.x2, X_MAX);
        for _ in 0..200 {
            s.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        assert!(!s.animated() && s.glide_unit == DVec2::ZERO && s.glide_x2 == 0.0);
    }

    /// The steps play in their order, each to the end, and the play stops: at 30, 60
    /// and 144 frames a second, in the same scene time.
    #[test]
    fn the_steps_play_in_order() {
        for hz in [30.0f32, 60.0, 144.0] {
            let mut s = Counterfactual::create();
            let mut seen = Vec::new();
            let mut last = Progress::of(s.play);
            for _ in 0..(10.0 * hz) as usize {
                s.advance(1.0 / hz, 1.0 / hz);
                let p = Progress::of(s.play);
                assert!(p.read >= last.read && p.set >= last.set && p.work >= last.work && p.show >= last.show && p.step >= last.step, "{p:?} after {last:?}");
                if seen.last() != Some(&p.step) {
                    seen.push(p.step);
                }
                last = p;
            }
            assert_eq!(seen, [0, 1, 2, 3, 4], "at {hz} Hz");
            assert!(!s.animated() && last == Progress::DONE);
            // each step ends before the next begins
            assert!(READ.1 < SET.0 && SET.1 < WORK.0 && WORK.1 <= SHOW.0);
        }
        // a paused scene does not play on, but a change of setting is still answered
        let mut s = Counterfactual::create();
        s.set_up(2);
        s.advance(0.0, 0.0);
        assert!(s.change.is_none());
    }

    /// The layout, on every kind of canvas: everything inside it and finite, the plot
    /// clear of the strips and of the box of steps.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1078.0, 1000.0), (1600.0, 1000.0), (390.0, 800.0), (390.0, 754.0), (240.0, 180.0), (800.0, 390.0), (600.0, 480.0), (1.0, 1.0), (3000.0, 2000.0), (300.0, 600.0)] {
            for inset in [0.0, 46.0] {
                let lay = Layout::of(Vec2::new(w, h), inset);
                let map = lay.map();
                assert!(map.o.is_finite() && map.k.is_finite() && map.k.x > 0.0 && map.k.y > 0.0);
                let all = [Some(lay.plot), lay.strips[0], lay.strips[1], lay.steps, lay.card, lay.play];
                for r in all.iter().flatten() {
                    assert!(r.min.is_finite() && r.max.is_finite() && r.max.cmpgt(r.min).all(), "{w}x{h}: {r:?}");
                }
                if w >= 300.0 && h >= 300.0 {
                    for r in all.iter().flatten() {
                        assert!(r.min.x >= 0.0 && r.max.x <= w + 0.5 && r.max.y <= h + 0.5, "{w}x{h}: {r:?}");
                    }
                    for other in [lay.strips[0], lay.strips[1], lay.steps, lay.card].iter().flatten() {
                        let apart = lay.plot.max.x <= other.min.x || other.max.x <= lay.plot.min.x || lay.plot.max.y <= other.min.y || other.max.y <= lay.plot.min.y;
                        assert!(apart, "{w}x{h}: the plot {:?} and {other:?}", lay.plot);
                    }
                    // a point of the plot goes there and back
                    let p = DVec2::new(2.5, 3.5);
                    assert!((map.back(map.to(p)) - p).length() < 1e-4);
                }
            }
        }
    }
}

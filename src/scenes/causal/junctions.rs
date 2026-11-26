//! Chains, forks and colliders: three variables wired three ways, and what holding the
//! middle one fixed does to the other two.
//!
//! Every unit of a sample is dealt one bell-curve noise per variable, once; each
//! junction wires the same noises into X, Y and a third variable, so that a unit keeps
//! its noise when the wiring changes. Holding the third variable fixed is done as it is
//! done with data: only the units in a band of it are looked at (or, as selection
//! happens in life, only those above a threshold).

use super::*;
use crate::math::causal::{correlation, line, mean, partial_correlation, variance};

pub const DEF: SceneDef = SceneDef {
    id: "junctions",
    title: "Chains, forks and colliders",
    group: GROUP,
    blurb: "The three ways two arrows can wire three variables, and what holding the middle one fixed does to the other two.",
    make: |_| Box::new(Junctions::create()),
};

/// The most units there can be (the slider's top): each is dealt its noise once.
const MOST: usize = 4000;
/// The units at the start, and in every experiment.
const UNITS: usize = 1000;
/// The sample the scene opens with, and that every experiment uses.
const SEED: u64 = 398;
/// Units drawn in each small picture (its numbers use them all).
const SMALL_DOTS: usize = 500;
/// How far either way of the mean the pictures reach, in standard deviations.
const REACH: f64 = 3.3;
/// No band narrower than this (standard deviations).
const NARROWEST: f32 = 0.02;
/// With fewer units than this a correlation wanders a long way from sample to sample.
const FEW: usize = 30;
/// How long a change takes to glide into place: the time to cover two thirds of the way.
const GLIDE: f32 = 0.11;
/// The same for each of the two eases a line follows its place through.
const FOLLOW: f32 = 0.06;
/// The arrows' strengths at the start, and in every experiment.
const STRENGTHS: Strengths = Strengths { a: 1.5, b: 1.5, k: 1.0 };
/// The band at the start (standard deviations either side of the mean).
const BAND: [f32; 2] = [-0.2, 0.2];
/// The threshold of the experiments that select: about the top quarter of the units.
const TOP_QUARTER: f32 = 0.67;
/// How much wider a collider's effect's graph is than the others' (for D, beside C).
const EFFECT_WIDER: f32 = 1.3;

/// The four wirings. A collider's effect is the collider with one more variable, `D`,
/// that the collider causes, and `D` is the one held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Chain,
    Fork,
    Collider,
    Effect,
}

impl Kind {
    const ALL: [Kind; 4] = [Kind::Chain, Kind::Fork, Kind::Collider, Kind::Effect];

    fn index(self) -> usize {
        self as usize
    }

    /// Its name in the small pictures and the panel.
    fn name(self) -> &'static str {
        match self {
            Kind::Chain => "chain",
            Kind::Fork => "fork",
            Kind::Collider => "collider",
            Kind::Effect => "collider's effect",
        }
    }

    /// Its name over the large picture.
    fn heading(self) -> &'static str {
        match self {
            Kind::Chain => "A chain",
            Kind::Fork => "A fork",
            Kind::Collider => "A collider",
            Kind::Effect => "A collider's effect",
        }
    }

    /// The variable that is held: the middle one, or what the collider causes.
    fn held(self) -> &'static str {
        match self {
            Kind::Chain => "M",
            Kind::Fork => "Z",
            Kind::Collider => "C",
            Kind::Effect => "D",
        }
    }

    fn wiring(self) -> &'static str {
        match self {
            Kind::Chain => r"$X \to M \to Y$",
            Kind::Fork => r"$X \leftarrow Z \to Y$",
            Kind::Collider => r"$X \to C \leftarrow Y$",
            Kind::Effect => r"$X \to C \leftarrow Y,\ C \to D$",
        }
    }

    /// What the junction does, by the rule (d-separation), with these strengths: an
    /// arrow of strength 0 is no arrow.
    fn role(self, s: Strengths) -> &'static str {
        if s.a == 0.0 || s.b == 0.0 {
            return match self {
                Kind::Chain => r"An arrow of strength 0 is no arrow: nothing passes between $X$ and $Y$, whether $M$ is held or not.",
                Kind::Fork => r"An arrow of strength 0 is no arrow: nothing passes between $X$ and $Y$, whether $Z$ is held or not.",
                Kind::Collider => r"An arrow of strength 0 is no arrow: nothing passes between $X$ and $Y$, whether $C$ is held or not.",
                Kind::Effect => r"An arrow of strength 0 is no arrow: nothing passes between $X$ and $Y$, whether $D$ is held or not.",
            };
        }
        if self == Kind::Effect && s.k == 0.0 {
            return r"With $k = 0$, $C$ does not cause $D$, and holding $D$ opens nothing.";
        }
        match self {
            Kind::Chain => r"$M$ passes on what $X$ does to $Y$. Holding $M$ blocks the path from $X$ to $Y$.",
            Kind::Fork => r"$Z$ causes both $X$ and $Y$. Holding $Z$ blocks the path between them.",
            Kind::Collider => r"Both $X$ and $Y$ cause $C$. The path is blocked at $C$; holding $C$ opens it.",
            Kind::Effect => r"$C$ causes $D$. Holding $D$ opens the path at $C$ too, though less than holding $C$.",
        }
    }

    /// An everyday reading of the letters.
    fn story(self) -> &'static str {
        match self {
            Kind::Chain => r"For instance: $X$ a drug's dose, $M$ its level in the blood, $Y$ its effect.",
            Kind::Fork => r"For instance: $Z$ a child's age, $X$ shoe size, $Y$ reading score.",
            Kind::Collider => r"For instance: $X$ talent at maths, $Y$ talent at music, $C$ how strong a candidate is.",
            Kind::Effect => r"For instance: the talents and $C$ as before, $D$ an interview mark that $C$ and luck decide.",
        }
    }

    /// The wiring as equations (each $\varepsilon$ a noise of its own).
    fn model(self) -> &'static str {
        match self {
            Kind::Chain => r"M = aX + \varepsilon_M, \quad Y = bM + \varepsilon_Y",
            Kind::Fork => r"X = aZ + \varepsilon_X, \quad Y = bZ + \varepsilon_Y",
            Kind::Collider => r"C = aX + bY + \varepsilon_C",
            Kind::Effect => r"C = aX + bY + \varepsilon_C, \quad D = kC + \varepsilon_D",
        }
    }

    /// How far the mean of Y moves when X is set one unit higher, every arrow into X
    /// cut: the true effect of X on Y.
    fn effect(self, s: Strengths) -> f64 {
        match self {
            Kind::Chain => s.a as f64 * s.b as f64,
            _ => 0.0,
        }
    }

    /// Does the one path between X and Y carry dependence, with the held variable held
    /// exactly (`held`) or not at all? The rule (d-separation): a chain or fork variable
    /// blocks the path when it is held; a collider blocks it unless it, or something it
    /// causes, is held.
    fn open(self, held: bool) -> bool {
        match self {
            Kind::Chain | Kind::Fork => !held,
            Kind::Collider | Kind::Effect => held,
        }
    }

    /// Are X and Y dependent, with the held variable held exactly (`held`) or not at
    /// all? Where the path is open, yes, unless an arrow on it has strength 0: then it
    /// is not there, and nothing passes.
    fn dependent(self, s: Strengths, held: bool) -> bool {
        let on_path = match self {
            Kind::Effect if held => s.a != 0.0 && s.b != 0.0 && s.k != 0.0,
            _ => s.a != 0.0 && s.b != 0.0,
        };
        self.open(held) && on_path
    }

    /// The variables of its graph: where each sits in a box from (0, 0) at the top left
    /// to (1, 1), its name and colour; X, Y, the middle one and (for `Effect`) D. X and Y
    /// are at the bottom corners and the middle one above between them in every
    /// junction, so that they differ only in which way the arrows point. (A collider's
    /// effect has D at the top right, past Y, its box a share `EFFECT_WIDER` wider.)
    fn nodes(self) -> Vec<(Vec2, &'static str, Color)> {
        let y = if self == Kind::Effect { 1.0 / EFFECT_WIDER } else { 1.0 };
        let mut nodes = vec![
            (Vec2::new(0.0, 1.0), "$X$", CAUSE),
            (Vec2::new(y, 1.0), "$Y$", EFFECT),
            (Vec2::new(y * 0.5, 0.0), ["$M$", "$Z$", "$C$", "$C$"][self.index()], THIRD),
        ];
        if self == Kind::Effect {
            nodes.push((Vec2::new(1.0, 0.0), "$D$", THIRD));
        }
        nodes
    }

    /// How wide its graph's box is, for a box `h` high (logical pixels).
    fn graph_width(self, h: f32) -> f32 {
        h * if self == Kind::Effect { 1.7 * EFFECT_WIDER } else { 1.7 }
    }

    /// Its arrows: from, to (as indices into `nodes`), and the strength.
    fn arrows(self, s: Strengths) -> Vec<(usize, usize, f32)> {
        match self {
            Kind::Chain => vec![(0, 2, s.a), (2, 1, s.b)],
            Kind::Fork => vec![(2, 0, s.a), (2, 1, s.b)],
            Kind::Collider => vec![(0, 2, s.a), (1, 2, s.b)],
            Kind::Effect => vec![(0, 2, s.a), (1, 2, s.b), (2, 3, s.k)],
        }
    }

    /// The index in `nodes` of the variable held.
    fn held_node(self) -> usize {
        if self == Kind::Effect {
            3
        } else {
            2
        }
    }
}

/// The strengths of the arrows: `a` and `b` of the two arrows of every junction, `k` of
/// the collider's arrow to its effect.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Strengths {
    a: f32,
    b: f32,
    k: f32,
}

/// What is done with the held variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hold {
    /// Every unit is looked at.
    Nothing,
    /// Only the units with it in a band.
    Band,
    /// Only the units with it above a threshold: selection.
    Above,
}

impl Hold {
    const ALL: [Hold; 3] = [Hold::Nothing, Hold::Band, Hold::Above];
}

/// Which units are looked at: the held variable, in standard deviations from its mean,
/// in the band or above the threshold.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Window {
    hold: Hold,
    band: [f32; 2],
    above: f32,
}

impl Window {
    fn lit(&self, z: f64) -> bool {
        // (a band taken to an end of the strip runs on past it: the strip shows only so
        // far, and a band dragged across all of it holds every unit)
        let edge = |v: f32, end: f64| if (v as f64 - end).abs() < 1e-6 { end.signum() * f64::INFINITY } else { v as f64 };
        match self.hold {
            Hold::Nothing => true,
            Hold::Band => z >= edge(self.band[0], -REACH) && z <= edge(self.band[1], REACH),
            Hold::Above => z >= self.above as f64,
        }
    }
}

/// Each unit's noise: one bell-curve draw for each of the four variables a junction can
/// have.
fn deal(seed: u64) -> Vec<[f64; 4]> {
    let mut rng = Rng::new(seed);
    (0..MOST).map(|_| [(); 4].map(|_| rng.normal() as f64)).collect()
}

/// X, Y and the held variable of every unit: each variable is its causes, each times
/// its arrow's strength, plus its own noise.
fn wire(kind: Kind, s: Strengths, noise: &[[f64; 4]]) -> [Vec<f64>; 3] {
    let (a, b, k) = (s.a as f64, s.b as f64, s.k as f64);
    let mut out = [Vec::with_capacity(noise.len()), Vec::with_capacity(noise.len()), Vec::with_capacity(noise.len())];
    for e in noise {
        let (x, y, held) = match kind {
            Kind::Chain => {
                let m = a * e[0] + e[1];
                (e[0], b * m + e[2], m)
            }
            Kind::Fork => (a * e[1] + e[0], b * e[1] + e[2], e[1]),
            Kind::Collider => (e[0], e[2], a * e[0] + b * e[2] + e[1]),
            Kind::Effect => {
                let c = a * e[0] + b * e[2] + e[1];
                (e[0], e[2], k * c + e[3])
            }
        };
        out[0].push(x);
        out[1].push(y);
        out[2].push(held);
    }
    out
}

/// Values in standard units (less their mean, over their standard deviation; all 0 if
/// they have no spread), with the mean and standard deviation.
fn standard(v: &[f64]) -> (Vec<f64>, f64, f64) {
    let (m, sd) = (mean(v), variance(v).sqrt());
    let scale = if sd > 1e-12 && sd.is_finite() { 1.0 / sd } else { 0.0 };
    (v.iter().map(|x| (x - m) * scale).collect(), m, sd)
}

/// A straight line fitted to some of the units, in standard units: how many, their
/// correlation, the line's slope and intercept, and the stretch of X it is drawn over:
/// where most of those units are (2.2 standard deviations either way of their mean,
/// which moves little when one unit comes or goes, as the least and greatest X would).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Fit {
    n: usize,
    r: f64,
    slope: f64,
    intercept: f64,
    from: f64,
    to: f64,
}

/// The line through the units `keep` picks (none for fewer than three).
fn fit(x: &[f64], y: &[f64], keep: &[bool]) -> Option<Fit> {
    let (xs, ys): (Vec<f64>, Vec<f64>) = x.iter().zip(y).zip(keep).filter(|(_, k)| **k).map(|((a, b), _)| (*a, *b)).unzip();
    if xs.len() < 3 {
        return None;
    }
    let (slope, intercept) = line(&xs, &ys);
    let (m, spread) = (mean(&xs), 2.2 * variance(&xs).sqrt());
    let (from, to) = (m - spread, m + spread);
    Some(Fit { n: xs.len(), r: correlation(&xs, &ys), slope, intercept, from, to })
}

/// What one junction's sample shows, with what is held.
struct Look {
    /// X, Y and the held variable of every unit, in standard units.
    x: Vec<f64>,
    y: Vec<f64>,
    z: Vec<f64>,
    /// The units looked at.
    lit: Vec<bool>,
    lit_count: usize,
    /// The line through every unit, and through the lit ones (none if nothing is held,
    /// or too few are lit).
    all: Option<Fit>,
    held: Option<Fit>,
    /// The correlation of X and Y with the held variable held exactly (by regression).
    partial: f64,
    /// What X really does to Y, as a line in standard units (slope, intercept).
    truth: (f64, f64),
}

fn look(kind: Kind, s: Strengths, noise: &[[f64; 4]], window: Window) -> Look {
    let [x, y, h] = wire(kind, s, noise);
    let partial = partial_correlation(&x, &y, &[&h]);
    let (x, mx, sx) = standard(&x);
    let (y, my, sy) = standard(&y);
    let (z, _, _) = standard(&h);
    let lit: Vec<bool> = z.iter().map(|v| window.lit(*v)).collect();
    let lit_count = lit.iter().filter(|l| **l).count();
    let all = fit(&x, &y, &vec![true; x.len()]);
    let held = if window.hold == Hold::Nothing { None } else { fit(&x, &y, &lit) };
    // the mean of Y with X set to x is effect × x (every noise has mean 0): in
    // standard units, a line of slope effect × sx / sy
    let effect = kind.effect(s);
    let truth = if sy > 1e-12 { (effect * sx / sy, (effect * mx - my) / sy) } else { (0.0, 0.0) };
    Look { x, y, z, lit, lit_count, all, held, partial: if partial.is_finite() { partial } else { 0.0 }, truth }
}

/// What the looks were worked out for.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Key {
    s: Strengths,
    n: usize,
    seed: u64,
    window: Window,
}

/// An experiment: the button, what to look for, and what it sets.
struct Trial {
    button: &'static str,
    caption: &'static str,
    kind: Kind,
    hold: Hold,
    truth: bool,
}

const TRIALS: [Trial; 5] = [
    Trial {
        button: "a chain: dose, blood level, effect",
        caption: r"Over all patients a larger dose goes with a larger effect: $r = 0.78$. Among those in the band, whose blood levels are about the same, the dose says almost nothing more: $r = 0.06$. The dose acts only through the blood, so holding the blood level holds all that the dose does.",
        kind: Kind::Chain,
        hold: Hold::Band,
        truth: false,
    },
    Trial {
        button: "a fork: age, shoe size, reading",
        caption: r"Over all children, bigger feet go with better reading, $r = 0.68$, because older children have both. Among children of about the same age (the band) the trend is gone: $r = 0.00$. Holding the common cause takes away the trend it made: this is adjusting for a confounder.",
        kind: Kind::Fork,
        hold: Hold::Band,
        truth: false,
    },
    Trial {
        button: "a collider: who gets in",
        caption: r"Over all candidates the two talents are unrelated: $r = 0.00$. A school takes the top quarter by $C$, the 249 bright dots, and among them the better at maths tend to be the worse at music: $r = -0.48$. Nobody's talent changed: to get in with little of one, a candidate needed much of the other.",
        kind: Kind::Collider,
        hold: Hold::Above,
        truth: false,
    },
    Trial {
        button: "selecting on what a collider causes",
        caption: r"The school sees only $D$, an interview mark that $C$ and luck decide, and takes the top quarter by it. The talents of those it takes are related again, more weakly: $r = -0.36$, against $-0.48$ when it took them by $C$ itself (the small picture of the collider). Holding or selecting on anything a collider causes opens the path too.",
        kind: Kind::Effect,
        hold: Hold::Above,
        truth: false,
    },
    Trial {
        button: "should the middle one be held?",
        caption: r"The green dashed lines are what $X$ really does to $Y$. In the chain the line through all units follows it, and holding $M$ hides a real effect. In the fork the band's line follows it: holding $Z$ is right. In the collider the line through all units follows it, and holding $C$ makes up an effect that is not there. Whether to hold the middle one depends on which way its arrows point.",
        kind: Kind::Chain,
        hold: Hold::Band,
        truth: true,
    },
];

// ----- where things go -----------------------------------------------------------

/// A rectangle on the canvas, in logical pixels from its top-left corner, y downwards.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Area {
    min: Vec2,
    max: Vec2,
}

impl Area {
    fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Area {
        let min = Vec2::new(x0, y0);
        Area { min, max: Vec2::new(x1, y1).max(min + Vec2::ONE) }
    }

    fn w(&self) -> f32 {
        self.max.x - self.min.x
    }

    fn h(&self) -> f32 {
        self.max.y - self.min.y
    }

    fn contains(&self, p: Vec2) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }

    /// The point a share `t` of the way across and `s` of the way down.
    fn at(&self, t: f32, s: f32) -> Vec2 {
        self.min + Vec2::new(t * self.w(), s * self.h())
    }
}

/// The three ways the picture is laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    /// The heading over the scatter and the strip under it, the four small pictures in
    /// a column beside them (a desktop).
    Wide,
    /// All in one column, the four small pictures in a row at the bottom (a phone held
    /// upright).
    Column,
    /// Three columns: the heading, the key and the buttons; the scatter and its strip;
    /// the four small pictures, two by two (a phone held sideways, a short window).
    Short,
}

/// Where everything goes on a canvas. Worked out in logical pixels for the view as it
/// opens, and drawn on the stage (`STAGE_VIEW`), so that zooming and panning move it
/// all together.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    /// Logical pixels per unit of the stage.
    ppu: f32,
    shape: Shape,
    /// The large picture's graph, and the words beside it.
    graph: Area,
    words: Area,
    /// The key to the lines, over the scatter.
    legend: Area,
    plot: Area,
    /// The held variable's spread, with the band on it.
    strip: Area,
    /// What to hold: nothing, a band, above a threshold.
    modes: [Area; 3],
    /// The four junctions, small: a column of cards beside the large picture, or tiles.
    cards: [Area; 4],
    /// Text sizes.
    size: f32,
    small: f32,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let ppu = (view.y / STAGE_H).min(view.x / 2.0).max(1e-3);
        if view.x < 620.0 && view.y >= 520.0 {
            Layout::column(view, ppu, inset)
        } else if view.y < 560.0 {
            Layout::short(view, ppu, inset)
        } else {
            Layout::wide(view, ppu, inset)
        }
    }

    /// Small pictures as tiles (name, scatter, numbers), not as cards with a graph.
    fn tiles(&self) -> bool {
        self.shape != Shape::Wide
    }

    fn short(view: Vec2, ppu: f32, inset: f32) -> Layout {
        let (w, h) = (view.x, view.y);
        let pad = 10.0;
        let top = inset.max(0.0) + 6.0;
        // the first column: the graph, the words under it, the key, the buttons at the foot
        let first = (w * 0.3).clamp(150.0, 300.0);
        let graph = Area::new(pad, top, pad + Kind::Effect.graph_width(56.0), top + 64.0);
        let modes = Layout::buttons(pad + first, h - pad - 22.0);
        let legend = Area::new(pad + 4.0, modes[0].min.y - 66.0, pad + first, modes[0].min.y - 8.0);
        let words = Area::new(pad, graph.max.y + 6.0, pad + first, legend.min.y - 4.0);
        // the second: the scatter, the name of X under it, and the strip
        let x0 = pad + first + 30.0;
        let side = (h - top - pad - 22.0 - 24.0 - 14.0).min(w * 0.42).max(40.0);
        let plot = Area::new(x0, top + 4.0, x0 + side, top + 4.0 + side);
        let strip = Area::new(plot.min.x, plot.max.y + 22.0, plot.max.x, plot.max.y + 46.0);
        // the third: the four small pictures, two by two, under a line saying what they are
        let (gap, left) = (6.0, plot.max.x + 20.0);
        let tile_w = ((w - pad - left - gap) * 0.5).max(10.0);
        let tile_h = ((h - top - 16.0 - pad - gap) * 0.5).max(10.0);
        let cards = [0, 1, 2, 3].map(|k| {
            let (x, y) = (left + (k % 2) as f32 * (tile_w + gap), top + 16.0 + (k / 2) as f32 * (tile_h + gap));
            Area::new(x, y, x + tile_w, y + tile_h)
        });
        Layout { view, ppu, shape: Shape::Short, graph, words, legend, plot, strip, modes, cards, size: 12.0, small: 11.0 }
    }

    fn wide(view: Vec2, ppu: f32, inset: f32) -> Layout {
        let (w, h) = (view.x, view.y);
        let pad = 16.0;
        let top = inset.max(0.0) + 12.0;
        let card_w = (w * 0.28).clamp(150.0, 320.0);
        let cards_x = w - pad - card_w;
        let (x0, x1) = (pad + 24.0, (cards_x - 30.0).max(pad + 80.0));
        let graph_h = (h * 0.13).clamp(56.0, 120.0);
        // (as wide as the widest graph needs: `draw_header` takes what the one shown does)
        let graph = Area::new(x0 - 4.0, top, x0 - 4.0 + Kind::Effect.graph_width(graph_h), top + graph_h);
        let words = Area::new(graph.max.x + 22.0, top, x1, top + graph_h);
        let legend = Area::new(x0, graph.max.y + 12.0, x1, graph.max.y + 34.0);
        // under the scatter: the name of X, the row of buttons, the strip, the words under it
        let strip_h = (h * 0.075).clamp(34.0, 72.0);
        let below = 22.0 + 30.0 + strip_h + 30.0;
        let room = (h - pad - below - legend.max.y - 6.0).max(40.0);
        let side = (x1 - x0).min(room).max(40.0);
        let left = x0 + ((x1 - x0) - side).max(0.0) * 0.5;
        let plot = Area::new(left, legend.max.y + 6.0, left + side, legend.max.y + 6.0 + side);
        let row = plot.max.y + 22.0;
        let strip = Area::new(plot.min.x, row + 30.0, plot.max.x, row + 30.0 + strip_h);
        let modes = Layout::buttons(plot.max.x, row + 2.0);
        // (under a line saying what they are)
        let (gap, cards_top) = (10.0, top + 22.0);
        let card_h = ((h - cards_top - pad - 3.0 * gap) / 4.0).max(20.0);
        let cards = [0, 1, 2, 3].map(|k| {
            let y = cards_top + k as f32 * (card_h + gap);
            Area::new(cards_x, y, w - pad, y + card_h)
        });
        Layout { view, ppu, shape: Shape::Wide, graph, words, legend, plot, strip, modes, cards, size: 13.0, small: 12.0 }
    }

    fn column(view: Vec2, ppu: f32, inset: f32) -> Layout {
        let (w, h) = (view.x, view.y);
        let pad = 10.0;
        let top = inset.max(0.0) + 8.0;
        let graph = Area::new(pad + 4.0, top, pad + 4.0 + Kind::Effect.graph_width(66.0), top + 80.0);
        let words = Area::new(graph.max.x + 14.0, top, w - pad, top + 80.0);
        let legend = Area::new(pad + 18.0, graph.max.y + 8.0, w - pad, graph.max.y + 44.0);
        // the four wirings in a row of small pictures at the bottom (under a line saying
        // what they are), and the strip over them
        let gap = 6.0;
        let tile_w = ((w - 2.0 * pad - 3.0 * gap) / 4.0).max(10.0);
        let tile_h = 18.0 + (tile_w - 10.0).min(h * 0.13).clamp(10.0, 110.0) + 34.0;
        let tiles_top = h - pad - tile_h;
        let strip_h = 38.0;
        let below = 22.0 + 30.0 + strip_h + 30.0;
        let room = (tiles_top - 22.0 - below - legend.max.y - 4.0).max(40.0);
        let side = (w - 2.0 * pad - 18.0).min(room).max(40.0);
        let left = pad + 18.0 + ((w - 2.0 * pad - 18.0) - side).max(0.0) * 0.5;
        let plot = Area::new(left, legend.max.y + 4.0, left + side, legend.max.y + 4.0 + side);
        let row = plot.max.y + 22.0;
        let strip = Area::new(plot.min.x, row + 30.0, plot.max.x, row + 30.0 + strip_h);
        let modes = Layout::buttons(plot.max.x, row + 2.0);
        let cards = [0, 1, 2, 3].map(|k| {
            let x = pad + k as f32 * (tile_w + gap);
            Area::new(x, tiles_top, x + tile_w, h - pad)
        });
        Layout { view, ppu, shape: Shape::Column, graph, words, legend, plot, strip, modes, cards, size: 12.0, small: 11.0 }
    }

    /// The three buttons for what to hold, ending at `right`, at the height `y`.
    fn buttons(right: f32, y: f32) -> [Area; 3] {
        let widths = [70.0, 60.0, 60.0];
        let mut x = right - widths.iter().sum::<f32>() - 8.0;
        widths.map(|w| {
            let a = Area::new(x, y, x + w, y + 22.0);
            x += w + 4.0;
            a
        })
    }

    /// A point of the canvas on the stage.
    fn at(&self, px: Vec2) -> Vec2 {
        Vec2::new(px.x - self.view.x * 0.5, self.view.y * 0.5 - px.y) / self.ppu
    }

    /// A point of the stage on the canvas (as it opens).
    fn px(&self, stage: Vec2) -> Vec2 {
        Vec2::new(stage.x * self.ppu + self.view.x * 0.5, self.view.y * 0.5 - stage.y * self.ppu)
    }
}

/// Where a point (standard units) is drawn in a scatter's square.
fn in_plot(plot: Area, u: f64, v: f64) -> Vec2 {
    let t = |a: f64| ((a.max(-REACH).min(REACH) + REACH) / (2.0 * REACH)) as f32;
    Vec2::new(plot.min.x + t(u) * plot.w(), plot.max.y - t(v) * plot.h())
}

/// The part of the line v = slope u + intercept from u = `from` to u = `to` that is
/// inside the scatter's square, if any.
fn clipped(slope: f64, intercept: f64, from: f64, to: f64) -> Option<[(f64, f64); 2]> {
    let (from, to) = (from.max(-REACH), to.min(REACH));
    if !(slope.is_finite() && intercept.is_finite() && from < to) {
        return None;
    }
    // (Liang–Barsky, against the top and bottom of the square)
    let (a, b) = ((from, intercept + slope * from), (to, intercept + slope * to));
    let d = (b.0 - a.0, b.1 - a.1);
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for (p, q) in [(-d.1, a.1 + REACH), (d.1, REACH - a.1)] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else if p < 0.0 {
            t0 = t0.max(q / p);
        } else {
            t1 = t1.min(q / p);
        }
    }
    let at = |t: f64| (a.0 + d.0 * t, (a.1 + d.1 * t).max(-REACH).min(REACH));
    (t0 < t1).then(|| [at(t0), at(t1)])
}

/// The colours of the held variable's values: from blue (low) through the section's
/// purple to a pale pink (high), two standard deviations either way.
fn shades() -> Vec<Color> {
    let (low, high) = (Color::hex(0x3F6BFF), Color::hex(0xFFB8EC));
    (0..=64).map(|i| i as f32 / 64.0).map(|t| if t < 0.5 { low.mix(THIRD, t * 2.0) } else { THIRD.mix(high, t * 2.0 - 1.0) }).collect()
}

fn shade(table: &[Color], z: f64) -> Color {
    let t = (z / 2.0 + 1.0) * 0.5;
    let i = (t.max(0.0).min(1.0) * (table.len() - 1) as f64).round() as usize;
    table.get(i).copied().unwrap_or(THIRD)
}

/// How many lines `wrapped` breaks `text` into at `width` logical pixels: the same
/// words, kept whole with their formulas, measured the same way.
fn lines_needed(f: &Frame, text: &str, width: f32, size: f32) -> usize {
    let mut words: Vec<String> = Vec::new();
    let mut inside = false;
    for word in text.split(' ') {
        match words.last_mut() {
            Some(last) if inside => {
                last.push(' ');
                last.push_str(word);
            }
            _ => words.push(word.to_string()),
        }
        inside ^= word.matches('$').count() % 2 == 1;
    }
    let mut lines = vec![String::new()];
    for word in words {
        let Some(last) = lines.last_mut() else { break };
        if last.is_empty() {
            last.push_str(&word);
        } else if f.text_size(last, size).x + f.text_size(&word, size).x + size * 0.47 > width {
            lines.push(word);
        } else {
            last.push(' ');
            last.push_str(&word);
        }
    }
    lines.len()
}

/// A correlation as the picture writes it.
fn r_text(fit: Option<Fit>) -> String {
    match fit {
        Some(fit) => format!("$r = {}$", num(fit.r, 2)),
        None => "too few for a line".to_string(),
    }
}

// ----- the scene ------------------------------------------------------------------

/// A line as drawn (slope, intercept, and the X it runs from and to), following where
/// it belongs through two eases one after the other: a unit coming into the band or
/// leaving it moves the line a step, and the picture follows the steps evenly.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Line {
    between: [f32; 4],
    at: [f32; 4],
}

impl Line {
    /// `dt` seconds nearer to `to`; false once it is there.
    fn follow(&mut self, to: [f32; 4], dt: f32) -> bool {
        let k = 1.0 - (-dt.max(0.0) / FOLLOW).exp();
        let mut apart = 0.0f32;
        for j in 0..4 {
            self.between[j] += (to[j] - self.between[j]) * k;
            self.at[j] += (self.between[j] - self.at[j]) * k;
            apart = apart.max((to[j] - self.at[j]).abs()).max((to[j] - self.between[j]).abs());
        }
        // (exactly there in the end, not nearly: the picture then stops changing)
        if apart < 1e-3 || !apart.is_finite() {
            *self = Line { between: to, at: to };
            return false;
        }
        true
    }
}

struct Junctions {
    kind: Kind,
    s: Strengths,
    hold: Hold,
    band: [f32; 2],
    above: f32,
    n: usize,
    seed: u64,
    /// Draw what X really does to Y.
    truth: bool,
    noise: Vec<[f64; 4]>,
    /// What each junction shows, in `Kind::ALL`'s order, and what for.
    looks: Vec<Look>,
    key: Option<Key>,
    /// The large scatter as drawn: each unit's X, Y and held value (standard units),
    /// gliding to where they belong.
    shown: Vec<[f32; 3]>,
    /// The small scatters as drawn.
    small: [Vec<Vec2>; 4],
    /// The lines as drawn: every unit, the lit ones, the truth; in the large scatter
    /// and in each small one.
    lines: [Option<Line>; 3],
    small_lines: [[Option<Line>; 3]; 4],
    moving: bool,
    shades: Vec<Color>,
}

impl Junctions {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let mut s = Junctions {
            kind: Kind::Collider,
            s: STRENGTHS,
            hold: Hold::Band,
            band: BAND,
            above: TOP_QUARTER,
            n: UNITS,
            seed: SEED,
            truth: false,
            noise: deal(SEED),
            looks: Vec::new(),
            key: None,
            shown: Vec::new(),
            small: Default::default(),
            lines: [None; 3],
            small_lines: [[None; 3]; 4],
            moving: false,
            shades: shades(),
        };
        s.refresh();
        s.settle();
        s
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let Some(t) = TRIALS.get(k) else { return };
        (self.kind, self.hold, self.truth) = (t.kind, t.hold, t.truth);
        (self.s, self.band, self.above, self.n) = (STRENGTHS, BAND, TOP_QUARTER, UNITS);
        self.reseed(SEED);
        self.refresh();
    }

    fn reseed(&mut self, seed: u64) {
        if seed != self.seed {
            self.seed = seed;
            self.noise = deal(seed);
        }
    }

    fn window(&self) -> Window {
        Window { hold: self.hold, band: self.band, above: self.above }
    }

    /// Work out what every junction shows, if anything it depends on has changed.
    fn refresh(&mut self) {
        let key = Key { s: self.s, n: self.n.clamp(1, MOST), seed: self.seed, window: self.window() };
        if self.key == Some(key) {
            return;
        }
        self.key = Some(key);
        let noise = &self.noise[..key.n.min(self.noise.len())];
        self.looks = Kind::ALL.iter().map(|k| look(*k, self.s, noise, key.window)).collect();
        self.moving = true;
    }

    fn current(&self) -> Option<&Look> {
        self.looks.get(self.kind.index())
    }

    /// Are too few units held for their correlation to mean much?
    fn few(&self) -> bool {
        self.hold != Hold::Nothing && self.current().is_some_and(|l| l.lit_count < FEW)
    }

    /// Where a junction's lines belong.
    fn line_goals(&self, look: &Look) -> [Option<[f32; 4]>; 3] {
        let as_line = |f: Option<Fit>| f.map(|f| [f.slope, f.intercept, f.from, f.to].map(|v| v as f32));
        let r = REACH as f32;
        [as_line(look.all), as_line(look.held), self.truth.then_some([look.truth.0 as f32, look.truth.1 as f32, -r, r])]
    }

    /// Move everything drawn `dt` seconds nearer to where it belongs.
    fn glide(&mut self, dt: f32) {
        let k = 1.0 - (-dt.max(0.0) / GLIDE).exp();
        let mut moving = false;
        let mut ease = |at: &mut f32, to: f32| {
            let next = *at + (to - *at) * k;
            if (to - next).abs() < 2e-3 || !next.is_finite() {
                *at = to;
            } else {
                *at = next;
                moving = true;
            }
        };
        if let Some(look) = self.looks.get(self.kind.index()) {
            let n = look.x.len();
            self.shown.truncate(n);
            for i in 0..n {
                let to = [look.x[i] as f32, look.y[i] as f32, look.z[i] as f32];
                match self.shown.get_mut(i) {
                    Some(at) => (0..3).for_each(|j| ease(&mut at[j], to[j])),
                    None => self.shown.push(to),
                }
            }
        }
        for (small, look) in self.small.iter_mut().zip(&self.looks) {
            let n = look.x.len().min(SMALL_DOTS);
            small.truncate(n);
            for i in 0..n {
                let to = Vec2::new(look.x[i] as f32, look.y[i] as f32);
                match small.get_mut(i) {
                    Some(at) => {
                        ease(&mut at.x, to.x);
                        ease(&mut at.y, to.y);
                    }
                    None => small.push(to),
                }
            }
        }
        let main = self.current().map_or([None; 3], |look| self.line_goals(look));
        let small: Vec<[Option<[f32; 4]>; 3]> = self.looks.iter().map(|look| self.line_goals(look)).collect();
        let pairs = self.lines.iter_mut().zip(main).chain(self.small_lines.iter_mut().zip(small).flat_map(|(at, to)| at.iter_mut().zip(to)));
        for (at, to) in pairs {
            match (at.as_mut(), to) {
                (Some(at), Some(to)) => moving |= at.follow(to, dt),
                _ => *at = to.map(|to| Line { between: to, at: to }),
            }
        }
        self.moving = moving;
    }

    /// Everything where it belongs at once.
    fn settle(&mut self) {
        self.shown.clear();
        self.small.iter_mut().for_each(Vec::clear);
        self.lines = [None; 3];
        self.small_lines = [[None; 3]; 4];
        self.glide(1.0);
        self.moving = false;
    }

    // ----- input on the picture ---------------------------------------------------

    /// The small pictures and the buttons over the picture: clicked, they choose.
    fn clicks(&mut self, f: &mut Frame, lay: &Layout) {
        let Some(p) = f.mouse().map(|m| lay.px(m)) else { return };
        let card = lay.cards.iter().position(|a| a.contains(p));
        let mode = lay.modes.iter().position(|a| a.contains(p));
        if card.is_none() && mode.is_none() {
            return;
        }
        // (a click there is for choosing, and two quick ones are two choices)
        f.handles.claim_clicks();
        if f.ptr.clicked && !f.handles.on_handle() {
            if let Some(k) = card {
                self.kind = Kind::ALL[k];
                self.moving = true;
            }
            if let Some(m) = mode {
                self.hold = Hold::ALL[m];
            }
        }
    }

    /// The handles on the strip: the band's two edges and its middle, or the threshold.
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        let strip = lay.strip;
        let x_of = |z: f32| strip.min.x + ((z as f64 + REACH) / (2.0 * REACH)) as f32 * strip.w();
        let z_of = |px: f32| (((px - strip.min.x) / strip.w().max(1.0)) as f64 * 2.0 * REACH - REACH) as f32;
        let r = REACH as f32;
        match self.hold {
            Hold::Nothing => {}
            Hold::Band => {
                let [lo, hi] = self.band;
                let mut a = lay.at(Vec2::new(x_of(lo), strip.max.y));
                if f.drag2(1, &mut a, pal::YELLOW) {
                    self.band[0] = z_of(lay.px(a).x).max(-r).min(hi - NARROWEST);
                }
                let mut b = lay.at(Vec2::new(x_of(hi), strip.max.y));
                if f.drag2(2, &mut b, pal::YELLOW) {
                    self.band[1] = z_of(lay.px(b).x).max(lo + NARROWEST).min(r);
                }
                let mut m = lay.at(Vec2::new(x_of((lo + hi) * 0.5), strip.min.y));
                if f.drag2(3, &mut m, pal::YELLOW) {
                    let half = (hi - lo) * 0.5;
                    let mid = z_of(lay.px(m).x).max(-r + half).min(r - half);
                    self.band = [mid - half, mid + half];
                }
            }
            Hold::Above => {
                let mut a = lay.at(Vec2::new(x_of(self.above), strip.max.y));
                if f.drag2(4, &mut a, pal::YELLOW) {
                    self.above = z_of(lay.px(a).x).max(-r).min(r);
                }
            }
        }
    }

    // ----- drawing ----------------------------------------------------------------

    /// The large scatter: every unit, coloured by the held variable, the lit ones
    /// bright; the lines; the names of the axes.
    fn draw_plot(&self, f: &mut Frame, lay: &Layout) {
        let Some(look) = self.current() else { return };
        let plot = lay.plot;
        let at = |p: Vec2| lay.at(p).extend(0.0);
        f.dl.tris_alpha.rect(lay.at(plot.min), lay.at(plot.max), 0.0, Color::hex(0x0A0D16).a(0.85));
        frame_rect(f, lay.at(plot.min), lay.at(plot.max), 1.0, pal::FAINT);
        // the means of X and of Y
        for (a, b) in [(in_plot(plot, 0.0, -REACH), in_plot(plot, 0.0, REACH)), (in_plot(plot, -REACH, 0.0), in_plot(plot, REACH, 0.0))] {
            f.line(lay.at(a), lay.at(b), 1.0, pal::FAINT.a(0.6));
        }
        let size = (plot.w() / 150.0).clamp(2.6, 4.4);
        // two layers of one dot per unit, in the same order every frame: the faint one
        // under, the lit one over
        f.dl.discs.style.outline_px = 0.0;
        for (i, p) in self.shown.iter().enumerate() {
            let lit = look.lit.get(i).copied().unwrap_or(false);
            let c = shade(&self.shades, p[2] as f64);
            f.dl.discs.push(at(in_plot(plot, p[0] as f64, p[1] as f64)), size, c.a(if lit { 0.0 } else { 0.22 }));
        }
        let bright = f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 1.0, ..Default::default() });
        for (i, p) in self.shown.iter().enumerate() {
            let lit = look.lit.get(i).copied().unwrap_or(false);
            let c = shade(&self.shades, p[2] as f64);
            bright.push(at(in_plot(plot, p[0] as f64, p[1] as f64)), if lit { size + 2.2 } else { 0.0 }, c.mix(pal::WHITE, 0.25));
        }
        // the lines, over the dots
        for (k, (line, color, width)) in self.lines.iter().zip([(pal::WHITE, 2.4), (pal::YELLOW, 2.8), (TRUTH, 2.2)]).map(|(l, (c, w))| (l, c, w)).enumerate() {
            let Some([m, c, from, to]) = line.map(|l| l.at.map(|v| v as f64)) else { continue };
            let Some([p, q]) = clipped(m, c, from, to) else { continue };
            let (p, q) = (at(in_plot(plot, p.0, p.1)), at(in_plot(plot, q.0, q.1)));
            if k == 2 {
                f.dl.lines_with(LineKind::Overlay, LineStyle { dash: 10.0, duty: 0.55, ..Default::default() }).segment(p, q, width, color);
            } else {
                // (a dark edge, so that the line reads over dots of any colour)
                f.dl.overlay.segment(p, q, width + 2.5, Color::hex(0x080B14).a(0.7));
                f.dl.overlay.segment(p, q, width, color);
            }
        }
        // the axes' names
        let story = lay.shape == Shape::Wide;
        let x_name = if story { format!("{}  {}", "$X$", self.axis_word(0)) } else { "$X$".to_string() };
        f.text(lay.at(Vec2::new(plot.max.x, plot.max.y + 4.0)), x_name, lay.size + 1.0, CAUSE, Align2::RIGHT_TOP);
        f.text(lay.at(Vec2::new(plot.min.x - 6.0, plot.min.y)), "$Y$", lay.size + 2.0, EFFECT, Align2::RIGHT_TOP);
        if story {
            f.text(lay.at(Vec2::new(plot.min.x + 6.0, plot.min.y + 4.0)), self.axis_word(1), lay.small, EFFECT.a(0.8), Align2::LEFT_TOP);
        }
    }

    /// What X (0) and Y (1) stand for in the junction's everyday reading.
    fn axis_word(&self, which: usize) -> &'static str {
        match (self.kind, which) {
            (Kind::Chain, 0) => "the dose",
            (Kind::Chain, _) => "the effect",
            (Kind::Fork, 0) => "shoe size",
            (Kind::Fork, _) => "reading score",
            (_, 0) => "talent at maths",
            (_, _) => "talent at music",
        }
    }

    /// The key to the lines, with their correlations, over the scatter.
    fn draw_legend(&self, f: &mut Frame, lay: &Layout) {
        let Some(look) = self.current() else { return };
        let held = self.kind.held();
        let lit = match self.hold {
            Hold::Nothing => None,
            Hold::Band => Some(format!("the {} in the band of ${held}$: {}", count(look.lit_count), r_text(look.held))),
            Hold::Above => Some(format!("the {} with ${held}$ above the threshold: {}", count(look.lit_count), r_text(look.held))),
        };
        // (each: the colour of its line, the colour of its words, the words, dashed)
        let mut items = vec![(pal::WHITE, pal::TEXT, format!("all {} units: {}", count(look.x.len()), r_text(look.all)), false)];
        if let Some(lit) = lit {
            if self.few() {
                items.push((pal::YELLOW, pal::RED, format!("{lit} (too few to trust)"), false));
            } else {
                items.push((pal::YELLOW, pal::YELLOW, lit, false));
            }
        }
        if self.truth {
            let effect = self.kind.effect(self.s);
            let what = if effect == 0.0 { r"what $X$ does to $Y$: nothing".to_string() } else { r"what $X$ does to $Y$".to_string() };
            items.push((TRUTH, TRUTH, what, true));
        }
        let size = lay.small;
        let (mut x, mut y) = (lay.legend.min.x, lay.legend.min.y + 8.0);
        for (color, ink, text, dashed) in items {
            let w = text_width(&text, size) + 30.0;
            if x > lay.legend.min.x && x + w > lay.legend.max.x {
                (x, y) = (lay.legend.min.x, y + size + 6.0);
            }
            let (a, b) = (lay.at(Vec2::new(x, y)), lay.at(Vec2::new(x + 20.0, y)));
            if dashed {
                for k in 0..3 {
                    let s = k as f32 * 7.5;
                    f.dl.overlay.segment(lay.at(Vec2::new(x + s, y)).extend(0.0), lay.at(Vec2::new(x + s + 4.5, y)).extend(0.0), 2.2, color);
                }
            } else {
                f.dl.overlay.segment(a.extend(0.0), b.extend(0.0), 2.6, color);
            }
            f.text(lay.at(Vec2::new(x + 26.0, y)), text, size, ink, Align2::LEFT_CENTER);
            x += w + 16.0;
        }
    }

    /// The held variable's spread, coloured as the dots are, with the band (or the
    /// threshold) on it and the buttons for what to hold.
    fn draw_strip(&self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let Some(look) = self.current() else { return };
        let strip = lay.strip;
        let held = self.kind.held();
        f.dl.tris_alpha.rect(lay.at(strip.min), lay.at(strip.max), 0.0, Color::hex(0x0A0D16).a(0.85));
        // how many units have each value, and how many of those are lit
        let bins = 44;
        let (mut total, mut lit) = (vec![0usize; bins], vec![0usize; bins]);
        // (the few beyond the ends counted in the end bins)
        for (z, l) in look.z.iter().zip(&look.lit) {
            let i = ((((z + REACH) / (2.0 * REACH)) * bins as f64).floor().max(0.0) as usize).min(bins - 1);
            total[i] += 1;
            lit[i] += *l as usize;
        }
        let tallest = total.iter().copied().max().unwrap_or(1).max(1) as f32;
        let bar = strip.w() / bins as f32;
        for i in 0..bins {
            let centre = -REACH + (i as f64 + 0.5) * 2.0 * REACH / bins as f64;
            let c = shade(&self.shades, centre);
            let x = strip.min.x + i as f32 * bar;
            let height = |count: usize| (strip.h() - 4.0) * count as f32 / tallest;
            for (count, alpha) in [(total[i], 0.28), (lit[i], 0.95)] {
                if count > 0 {
                    let (lo, hi) = (Vec2::new(x + bar * 0.08, strip.max.y - height(count)), Vec2::new(x + bar * 0.92, strip.max.y));
                    f.dl.tris_alpha.rect(lay.at(lo), lay.at(hi), 0.0, c.a(alpha));
                }
            }
        }
        let x_of = |z: f32| strip.min.x + ((z as f64 + REACH) / (2.0 * REACH)) as f32 * strip.w();
        let r = REACH as f32;
        let span = match self.hold {
            Hold::Nothing => None,
            Hold::Band => Some((self.band[0].max(-r), self.band[1].min(r), true)),
            Hold::Above => Some((self.above.max(-r).min(r), r, false)),
        };
        if let Some((lo, hi, both)) = span {
            let (a, b) = (x_of(lo), x_of(hi));
            f.dl.tris_alpha.rect(lay.at(Vec2::new(a, strip.min.y)), lay.at(Vec2::new(b.max(a + 1.0), strip.max.y)), 0.0, pal::YELLOW.a(0.07));
            f.line(lay.at(Vec2::new(a, strip.min.y)), lay.at(Vec2::new(a, strip.max.y)), 1.6, pal::YELLOW);
            if both {
                f.line(lay.at(Vec2::new(b, strip.min.y)), lay.at(Vec2::new(b, strip.max.y)), 1.6, pal::YELLOW);
            }
        }
        // the words under the strip (clear of the handles on its lower edge; in three
        // columns there is no room, and the key says how many)
        if lay.shape != Shape::Short {
            let words_y = strip.max.y + 19.0;
            f.text(lay.at(Vec2::new(strip.min.x, words_y)), format!("lower ${held}$"), lay.small, pal::DIM, Align2::LEFT_CENTER);
            f.text(lay.at(Vec2::new(strip.max.x, words_y)), format!("higher ${held}$"), lay.small, pal::DIM, Align2::RIGHT_CENTER);
            let share = match self.hold {
                Hold::Nothing => "every unit counts".to_string(),
                Hold::Band => format!("{} of {} units in the band", count(look.lit_count), count(look.x.len())),
                Hold::Above => format!("{} of {} units above the threshold", count(look.lit_count), count(look.x.len())),
            };
            let few = self.few();
            let share = if few { format!("{share}: too few to trust") } else { share };
            f.text(lay.at(Vec2::new((strip.min.x + strip.max.x) * 0.5, words_y)), share, lay.small, if few { pal::RED } else { pal::YELLOW }, Align2::CENTER_CENTER);
        }
        let row = lay.modes[0].min.y + 11.0;
        if lay.shape == Shape::Wide {
            f.text(lay.at(Vec2::new(strip.min.x, row)), format!("every unit's ${held}$, in the colour of its dot"), lay.small, THIRD, Align2::LEFT_CENTER);
        }
        f.text(lay.at(Vec2::new(lay.modes[0].min.x - 8.0, row)), format!("hold ${held}$:"), lay.size, pal::TEXT, Align2::RIGHT_CENTER);
        for (k, (area, name)) in lay.modes.iter().zip(["nothing", "a band", "above"]).enumerate() {
            let on = Hold::ALL[k] == self.hold;
            let over = pointer.is_some_and(|p| area.contains(p));
            let fill = if on { Color::hex(0x4A3F1A) } else if over { Color::hex(0x33405E) } else { Color::hex(0x222B40) };
            f.dl.tris_alpha.rect(lay.at(area.min), lay.at(area.max), 0.0, fill);
            if on {
                frame_rect(f, lay.at(area.min), lay.at(area.max), 1.2, pal::YELLOW);
            }
            f.text(lay.at((area.min + area.max) * 0.5), name, lay.small, if on { pal::YELLOW } else { pal::TEXT }, Align2::CENTER_CENTER);
        }
    }

    /// A junction's graph in `area`, its nodes of radius `radius`, the held one ringed;
    /// with `strengths`, each arrow's strength beside it.
    fn draw_graph(&self, f: &mut Frame, lay: &Layout, kind: Kind, area: Area, radius: f32, strengths: bool) {
        let inner = Area::new(area.min.x + radius + 3.0, area.min.y + radius + 3.0, area.max.x - radius - 3.0, area.max.y - radius - 3.0);
        let place = |p: Vec2| on_screen(f, lay.at(inner.at(p.x, p.y)).extend(0.0));
        let mut nodes = Vec::new();
        for (p, name, color) in kind.nodes() {
            let Some(at) = place(p) else { return };
            nodes.push(Node { at, name, color });
        }
        let links: Vec<(usize, usize, Link, Color, f32)> = kind
            .arrows(self.s)
            .iter()
            .map(|&(a, b, s)| {
                let strong = (s.abs() / 1.5).min(1.0);
                (a, b, Link::Arrow, TRUTH.a(0.25 + 0.75 * strong), 1.2 + 1.4 * strong)
            })
            .collect();
        hud_graph(f, &nodes, &links, radius);
        if self.hold != Hold::Nothing {
            if let Some(n) = nodes.get(kind.held_node()) {
                hud_circle(f, n.at, radius + 4.5, 2.0, pal::YELLOW);
                // named beside its ring, to the right of the middle one (D has the
                // arrow from C on one side and the words on the other: its ring alone)
                if strengths && kind != Kind::Effect {
                    let word = if self.hold == Hold::Band { "held" } else { "selected" };
                    f.text_px(n.at + Vec2::new(radius + 10.0, 0.0), word, lay.small, pal::YELLOW, Align2::LEFT_CENTER);
                }
            }
        }
        // (only where the graph is large: in a small one they would crowd the arrows,
        // and the panel has them)
        if strengths && lay.shape == Shape::Wide {
            for (&(a, b, s), letter) in kind.arrows(self.s).iter().zip(["a", "b", "k"]) {
                let (Some(p), Some(q)) = (nodes.get(a), nodes.get(b)) else { continue };
                let along = (q.at - p.at).normalize_or_zero();
                // on the outer side of the arrow, away from the middle of the graph
                let mut side = Vec2::new(along.y, -along.x);
                let centre = nodes.iter().map(|n| n.at).sum::<Vec2>() / nodes.len().max(1) as f32;
                if side.dot((p.at + q.at) * 0.5 - centre) < 0.0 {
                    side = -side;
                }
                let at = (p.at + q.at) * 0.5 + side * 12.0;
                let align = if side.x >= 0.0 { Align2::LEFT_CENTER } else { Align2::RIGHT_CENTER };
                let align = if side.x.abs() < 0.3 { if side.y > 0.0 { Align2::CENTER_TOP } else { Align2::CENTER_BOTTOM } } else { align };
                f.text_px(at, format!("${letter} = {}$", num(s as f64, 1)), lay.small, TRUTH.a(0.9), align);
            }
        }
    }

    /// The heading over the large picture: the graph, its name, what it does, a reading.
    fn draw_header(&self, f: &mut Frame, lay: &Layout) {
        let radius = match lay.shape {
            Shape::Column => 14.0,
            _ => (lay.graph.h() * 0.17).clamp(11.0, 19.0),
        };
        let box_h = match lay.shape {
            Shape::Column => 66.0,
            Shape::Short => 56.0,
            Shape::Wide => lay.graph.h(),
        };
        let graph = Area::new(lay.graph.min.x, lay.graph.min.y, lay.graph.min.x + self.kind.graph_width(box_h), lay.graph.max.y);
        self.draw_graph(f, lay, self.kind, graph, radius, true);
        // the words start beside the graph shown, which may be narrower than the room
        // kept for the widest (under it, in three columns)
        let words = if lay.shape == Shape::Short {
            lay.words
        } else {
            let gap = lay.words.min.x - lay.graph.max.x;
            Area::new(graph.max.x + gap, lay.words.min.y, lay.words.max.x, lay.words.max.y)
        };
        let title = format!("{}   {}", self.kind.heading(), self.kind.wiring());
        let big = if lay.shape == Shape::Wide { 17.0 } else { 14.0 };
        let width = words.w() / lay.ppu;
        let lines = wrapped(f, lay.at(words.min), width, &title, big, pal::TEXT);
        let mut y = words.min.y + lines as f32 * (big + 3.0) + 5.0;
        let lines = wrapped(f, lay.at(Vec2::new(words.min.x, y)), width, self.kind.role(self.s), lay.small, pal::DIM);
        y += lines as f32 * (lay.small + 3.0) + 4.0;
        // (the everyday reading where there is room for all of it)
        let story = self.kind.story();
        if y + lines_needed(f, story, words.w(), lay.small) as f32 * (lay.small + 3.0) <= words.max.y + 6.0 {
            wrapped(f, lay.at(Vec2::new(words.min.x, y)), width, story, lay.small, pal::MUTED);
        }
    }

    /// The four junctions side by side, small: each its graph, its scatter and its two
    /// correlations (on a phone, its name, its scatter and the two numbers). Click one
    /// to see it large.
    fn draw_cards(&self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let first = lay.cards[0];
        let heading = if lay.tiles() { "every wiring, held alike (choose one):" } else { "every wiring, held alike (click one to see it large):" };
        f.text(lay.at(Vec2::new(first.min.x, first.min.y - 6.0)), heading, lay.small, pal::MUTED, Align2::LEFT_BOTTOM);
        for (k, kind) in Kind::ALL.into_iter().enumerate() {
            let (Some(card), Some(look)) = (lay.cards.get(k), self.looks.get(k)) else { continue };
            let chosen = kind == self.kind;
            let over = pointer.is_some_and(|p| card.contains(p));
            f.dl.tris_alpha.rect(lay.at(card.min), lay.at(card.max), 0.0, Color::hex(if over && !chosen { 0x121A2C } else { 0x0C111D }).a(0.92));
            frame_rect(f, lay.at(card.min), lay.at(card.max), if chosen { 1.6 } else { 1.0 }, if chosen { pal::TEXT.a(0.8) } else { pal::FAINT });
            if lay.tiles() {
                self.draw_tile(f, lay, k, *card, look);
                continue;
            }
            let pad = 10.0;
            f.text(lay.at(card.min + Vec2::new(pad, pad)), kind.name(), 13.0, if chosen { pal::TEXT } else { pal::DIM }, Align2::LEFT_TOP);
            // the scatter on the right, as tall as the card allows
            let side = (card.h() - 2.0 * pad).min(card.w() * 0.52).max(10.0);
            let plot = Area::new(card.max.x - pad - side, card.min.y + pad, card.max.x - pad, card.min.y + pad + side);
            self.draw_small_plot(f, lay, k, plot);
            // the graph and the numbers on the left
            let left = Area::new(card.min.x + pad, card.min.y + 30.0, plot.min.x - 10.0, card.max.y - pad);
            let numbers_h = 2.0 * (lay.small + 7.0);
            let graph = Area::new(left.min.x, left.min.y, left.max.x, (left.max.y - numbers_h - 4.0).max(left.min.y + 10.0));
            let radius = (graph.h() * 0.2).min(graph.w() * 0.12).clamp(6.0, 12.0);
            self.draw_graph(f, lay, kind, graph, radius, false);
            let mut y = left.max.y - numbers_h + lay.small * 0.5;
            f.text(lay.at(Vec2::new(left.min.x, y)), format!("all: {}", r_text(look.all)), lay.small, pal::TEXT, Align2::LEFT_CENTER);
            y += lay.small + 7.0;
            if self.hold != Hold::Nothing {
                let label = if self.hold == Hold::Band { "band" } else { "above" };
                f.text(lay.at(Vec2::new(left.min.x, y)), format!("{label}: {}", r_text(look.held)), lay.small, pal::YELLOW, Align2::LEFT_CENTER);
            }
        }
    }

    /// A small picture on a phone: the name over the scatter, the numbers under it.
    fn draw_tile(&self, f: &mut Frame, lay: &Layout, k: usize, tile: Area, look: &Look) {
        let chosen = Kind::ALL[k] == self.kind;
        let name = ["chain", "fork", "collider", "its effect"][k];
        f.text(lay.at(Vec2::new(tile.min.x + 5.0, tile.min.y + 4.0)), name, lay.small, if chosen { pal::TEXT } else { pal::DIM }, Align2::LEFT_TOP);
        let side = (tile.w() - 10.0).min(tile.h() - 52.0).max(8.0);
        let left = tile.min.x + (tile.w() - side) * 0.5;
        let plot = Area::new(left, tile.min.y + 18.0, left + side, tile.min.y + 18.0 + side);
        self.draw_small_plot(f, lay, k, plot);
        let value = |fit: Option<Fit>| fit.map_or("–".to_string(), |fit| format!("${}$", num(fit.r, 2)));
        let y = plot.max.y + 4.0;
        f.text(lay.at(Vec2::new(tile.min.x + 5.0, y)), format!("all {}", value(look.all)), lay.small, pal::TEXT, Align2::LEFT_TOP);
        if self.hold != Hold::Nothing {
            let label = if self.hold == Hold::Band { "band" } else { "above" };
            f.text(lay.at(Vec2::new(tile.min.x + 5.0, y + lay.small + 3.0)), format!("{label} {}", value(look.held)), lay.small, pal::YELLOW, Align2::LEFT_TOP);
        }
    }

    /// One small scatter: the units, the lit ones bright, and the lines.
    fn draw_small_plot(&self, f: &mut Frame, lay: &Layout, k: usize, plot: Area) {
        let (Some(look), Some(dots)) = (self.looks.get(k), self.small.get(k)) else { return };
        let at = |p: Vec2| lay.at(p).extend(0.0);
        f.dl.tris_alpha.rect(lay.at(plot.min), lay.at(plot.max), 0.0, Color::hex(0x06080F));
        frame_rect(f, lay.at(plot.min), lay.at(plot.max), 1.0, pal::FAINT.a(0.7));
        for (i, p) in dots.iter().enumerate() {
            let lit = look.lit.get(i).copied().unwrap_or(false);
            let c = shade(&self.shades, look.z.get(i).copied().unwrap_or(0.0));
            f.dl.discs.push(at(in_plot(plot, p.x as f64, p.y as f64)), 2.2, c.a(if lit { 0.0 } else { 0.3 }));
        }
        let bright = f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 0.6, ..Default::default() });
        for (i, p) in dots.iter().enumerate() {
            let lit = look.lit.get(i).copied().unwrap_or(false);
            let c = shade(&self.shades, look.z.get(i).copied().unwrap_or(0.0));
            bright.push(at(in_plot(plot, p.x as f64, p.y as f64)), if lit { 3.0 } else { 0.0 }, c);
        }
        let lines = self.small_lines.get(k).copied().unwrap_or([None; 3]);
        for (j, (line, color)) in lines.iter().zip([pal::WHITE, pal::YELLOW, TRUTH]).enumerate() {
            let Some([m, c, from, to]) = line.map(|l| l.at.map(|v| v as f64)) else { continue };
            let Some([p, q]) = clipped(m, c, from, to) else { continue };
            let (p, q) = (at(in_plot(plot, p.0, p.1)), at(in_plot(plot, q.0, q.1)));
            if j == 2 {
                f.dl.lines_with(LineKind::Overlay, LineStyle { dash: 6.0, duty: 0.55, ..Default::default() }).segment(p, q, 1.6, color);
            } else {
                f.dl.overlay.segment(p, q, 1.8, color);
            }
        }
    }
}

impl Scene for Junctions {
    fn view(&self) -> View {
        STAGE_VIEW
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.moving
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("yellow dots: move the band    small pictures: choose one    drag elsewhere: move    scroll: zoom")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"With two arrows, three variables can be wired in three ways. In a chain the middle one passes an effect along, $X \to M \to Y$; in a fork it causes both ends, $X \leftarrow Z \to Y$; in a collider both ends cause it, $X \to C \leftarrow Y$. Each dot is one unit (a patient, a child, a candidate): its $Y$ against its $X$, coloured by the third variable. To hold the middle one fixed, look only at the units in a narrow band of it (the bright dots), and see what becomes of the trend between $X$ and $Y$.");
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The wiring");
        let mut kind = self.kind.index();
        if ui::choice(ui, &Kind::ALL.map(|k| k.name()), &mut kind) {
            self.kind = Kind::ALL[kind.min(3)];
            self.moving = true;
        }
        ui::equation(ui, self.kind.model());
        ui::note(ui, r"Each $\varepsilon$ is a noise of the variable's own, a bell curve of spread 1, and so is each variable with no arrow into it. The strengths are the same in every junction:");
        ui::slider(ui, r"strength $a$", &mut self.s.a, -2.0..=2.0);
        ui::slider(ui, r"strength $b$", &mut self.s.b, -2.0..=2.0);
        ui::slider(ui, r"strength $k$ of $C \to D$", &mut self.s.k, -2.0..=2.0);
        ui::note(ui, r"At 0 an arrow is not there at all, and nothing passes along the path. ($k$ is used by the collider's effect alone.)");

        ui::section(ui, "Holding");
        let mut hold = Hold::ALL.iter().position(|h| *h == self.hold).unwrap_or(1);
        let held = self.kind.held();
        let names = ["nothing".to_string(), format!("a band of ${held}$"), format!("${held}$ above a threshold")];
        if ui::choice(ui, &names.iter().map(String::as_str).collect::<Vec<_>>(), &mut hold) {
            self.hold = Hold::ALL[hold.min(2)];
        }
        ui::note(ui, "Drag the yellow dots under the scatter: the band's two edges, and its middle to move it whole; or the threshold. The band is measured in standard deviations of the held variable, so the same band is used for every wiring.");
        ui::slider_int_log(ui, "units", &mut self.n, 20..=MOST);
        if ui.button("new sample").clicked() {
            self.reseed(self.seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) % 1_000_000_007);
        }
        ui::note(ui, "A fresh draw of every noise: the numbers change a little, the patterns do not.");
        ui.checkbox(&mut self.truth, "show the true effect (green, dashed)");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (THIRD, r"dots: one unit each, coloured by the held variable: blue where it is low, purple in the middle, pale pink where it is high (the strip under the scatter shows how many units have each value)"),
                (pal::YELLOW, "yellow: the band, or the threshold. The units in it are bright, the rest faint; the yellow line is the trend among the bright ones"),
                (pal::WHITE, "white line: the trend among all the units"),
                (TRUTH, r"green: the arrows, and (dashed, when shown) what $X$ really does to $Y$: how the mean of $Y$ moves if $X$ is set by hand"),
                (CAUSE, r"$X$, across"),
                (EFFECT, r"$Y$, up"),
                (THIRD, "purple: the third variable; ringed in yellow when it is held"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Correlation $r$:", r"how closely the units follow a straight line, from $-1$ to $1$; 0 when there is no straight-line trend. Here every effect is a straight line and every noise a bell curve, so over all units, or with a variable held, $r = 0$ means no dependence of any kind.");
        ui::term(ui, "Standard deviation:", "the typical distance of a value from its mean. The band's edges are counted in it, so that a band holds about the same share of the units whatever the variable.");
        ui::term(ui, "Holding fixed:", r"comparing only units with the same value of a variable (also called conditioning on it, or controlling for it). With data, a narrow band of it; by regression, taking out of $X$ and $Y$ what it explains, which gives the partial correlation.");
        ui::term(ui, "Chain:", r"$X \to M \to Y$; $M$ is a mediator, a go-between.");
        ui::term(ui, "Fork:", r"$X \leftarrow Z \to Y$; $Z$ is a common cause, a confounder.");
        ui::term(ui, "Collider:", r"$X \to C \leftarrow Y$; $C$ is a common effect, where the two arrowheads collide.");
        ui::term(ui, "Selection bias:", "a trend made by looking only at units chosen for a common effect of two variables, or for something it causes. Also called Berkson's paradox; explaining away is the same seen in one unit: it was chosen, one cause is weak, so the other is likely strong.");
        ui::term(
            ui,
            "$d$-separation:",
            r"the rule these add up to. A path between two variables (a route along the arrows, whichever way they point) carries dependence unless it passes through a chain or fork variable that is held, or through a collider that is not held and nothing it causes is held either. If every path is blocked the two are independent, whatever the strengths; if one is open they are dependent, unless the strengths happen to cancel.",
        );

        ui::section(ui, "Numbers");
        if let Some(look) = self.current() {
            ui::readout_colored(ui, "all units: $r$", r_value(look.all), pal::WHITE);
            let lit_label = match self.hold {
                Hold::Nothing => None,
                Hold::Band => Some("the units in the band: $r$"),
                Hold::Above => Some("the units above the threshold: $r$"),
            };
            if let Some(label) = lit_label {
                ui::readout_colored(ui, label, r_value(look.held), pal::YELLOW);
                ui::readout(ui, "how many", format!("{} of {}", count(look.lit_count), count(look.x.len())));
            }
            ui::readout(ui, &format!("held exactly: $r$ given ${held}$"), format!("${}$", num(look.partial, 2)));
            ui::note(ui, &format!(r"The partial correlation: what a band of ${held}$ gives as it narrows, with more and more units."));
            let effect = self.kind.effect(self.s);
            if self.kind == Kind::Chain {
                ui::readout_colored(ui, r"what $X$ does to $Y$: $ab$", format!("${}$", num(effect, 2)), TRUTH);
                ui::note(ui, r"Set $X$ one higher by hand, every arrow into it cut, and the mean of $Y$ moves by $ab$.");
            } else {
                ui::readout_colored(ui, r"what $X$ does to $Y$", "nothing", TRUTH);
            }
            let rule = |dependent: bool| if dependent { "dependent" } else { "independent" };
            ui::readout(ui, r"by the rule, $X$ and $Y$ are", rule(self.kind.dependent(self.s, false)));
            ui::readout(ui, &format!(r"and with ${held}$ held exactly"), rule(self.kind.dependent(self.s, true)));
        }
        ui::see_also(ui, &[("confound", "confounding, at length"), ("adjust", "what to control for"), ("pc", "finding colliders in data"), ("gauss2", "holding fixed in a bell curve")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: there is no room to draw in)
        if view.x < 160.0 || view.y < 140.0 {
            self.refresh();
            self.glide(f.real_dt);
            return;
        }
        let lay = Layout::of(view, f.top_inset);
        let pointer = f.mouse().map(|m| lay.px(m));
        self.clicks(f, &lay);
        self.handles(f, &lay);
        self.refresh();
        self.glide(f.real_dt);
        self.draw_header(f, &lay);
        self.draw_legend(f, &lay);
        self.draw_plot(f, &lay);
        self.draw_strip(f, &lay, pointer);
        self.draw_cards(f, &lay, pointer);
    }
}

/// A correlation as the panel writes it.
fn r_value(fit: Option<Fit>) -> String {
    match fit {
        Some(fit) => format!("${}$", num(fit.r, 2)),
        None => "too few units".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, after other settings were made first
    /// (an experiment sets everything it depends on).
    fn run(k: usize) -> Junctions {
        let mut s = Junctions::create();
        (s.kind, s.hold, s.band, s.above, s.n, s.truth) = (Kind::Fork, Hold::Nothing, [1.0, 2.5], -1.0, 57, !TRIALS[k].truth);
        s.s = Strengths { a: -0.3, b: 2.0, k: 0.0 };
        s.reseed(5);
        s.refresh();
        s.set_up(k);
        s
    }

    fn look_of(s: &Junctions, kind: Kind) -> &Look {
        &s.looks[kind.index()]
    }

    fn r(fit: Option<Fit>) -> f64 {
        fit.map_or(f64::NAN, |f| f.r)
    }

    fn slope(fit: Option<Fit>) -> f64 {
        fit.map_or(f64::NAN, |f| f.slope)
    }

    /// A number as a caption writes it (an ASCII minus in the markup).
    fn written(v: f64) -> String {
        num(v, 2).replace('−', "-")
    }

    fn says(k: usize, text: &str) {
        assert!(TRIALS[k].caption.contains(text), "{}: the caption should say {text:?}", TRIALS[k].button);
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // a chain: a trend over all patients, almost none among those with about the
        // same blood level
        let s = run(0);
        assert!(s.kind == Kind::Chain && s.hold == Hold::Band && !s.truth && s.n == UNITS);
        let l = look_of(&s, Kind::Chain);
        says(0, &format!("larger effect: $r = {}$", written(r(l.all))));
        says(0, &format!("almost nothing more: $r = {}$", written(r(l.held))));
        assert!(r(l.all) > 0.75 && r(l.held).abs() < 0.08 && l.lit_count > 120, "{} {} ({} units)", r(l.all), r(l.held), l.lit_count);

        // a fork: the same pattern
        let s = run(1);
        assert!(s.kind == Kind::Fork && s.hold == Hold::Band);
        let l = look_of(&s, Kind::Fork);
        says(1, &format!("better reading, $r = {}$", written(r(l.all))));
        says(1, &format!("the trend is gone: $r = {}$", written(r(l.held))));
        assert!(r(l.all) > 0.65 && r(l.held).abs() < 0.02, "{} {}", r(l.all), r(l.held));

        // a collider: unrelated over everyone, related among the top quarter by C
        let s = run(2);
        assert!(s.kind == Kind::Collider && s.hold == Hold::Above);
        let l = look_of(&s, Kind::Collider);
        let share = l.lit_count as f64 / l.x.len() as f64;
        says(2, &format!("unrelated: $r = {}$", written(r(l.all))));
        says(2, &format!("the {} bright dots", l.lit_count));
        says(2, &format!("worse at music: $r = {}$", written(r(l.held))));
        says(2, "top quarter");
        assert!(r(l.all).abs() < 0.01 && r(l.held) < -0.45 && slope(l.held) < -0.4 && (0.23..0.27).contains(&share), "{} {} {share}", r(l.all), r(l.held));

        // what the collider causes: the same, more weakly, and the collider's own
        // small picture shows the stronger trend beside it
        let s = run(3);
        assert!(s.kind == Kind::Effect && s.hold == Hold::Above);
        let (effect, collider) = (look_of(&s, Kind::Effect), look_of(&s, Kind::Collider));
        says(3, &format!("more weakly: $r = {}$", written(r(effect.held))));
        says(3, &format!("against ${}$ when it took", written(r(collider.held))));
        says(3, "top quarter");
        assert!(r(effect.all).abs() < 0.01 && r(effect.held) < -0.3 && r(effect.held) > r(collider.held) + 0.08, "{} against {}", r(effect.held), r(collider.held));
        assert!((0.23..0.27).contains(&(effect.lit_count as f64 / effect.x.len() as f64)));

        // should the middle one be held? The true effect is followed by the line of all
        // units in the chain and the collider, and by the band's line in the fork; the
        // other line is far from it
        let s = run(4);
        assert!(s.truth && s.kind == Kind::Chain && s.hold == Hold::Band);
        let near = |fit: Option<Fit>, look: &Look| (slope(fit) - look.truth.0).abs();
        let (chain, fork, collider) = (look_of(&s, Kind::Chain), look_of(&s, Kind::Fork), look_of(&s, Kind::Collider));
        assert!(chain.truth.0 > 0.7 && near(chain.all, chain) < 0.03 && near(chain.held, chain) > 0.5, "{:?} {:?} {:?}", chain.truth, chain.all, chain.held);
        assert!(fork.truth.0 == 0.0 && near(fork.held, fork) < 0.02 && near(fork.all, fork) > 0.5, "{:?} {:?}", fork.all, fork.held);
        assert!(collider.truth.0 == 0.0 && near(collider.all, collider) < 0.02 && near(collider.held, collider) > 0.5, "{:?} {:?}", collider.all, collider.held);
        // (and the lines cross the square near its middle, as the truth does: no offset)
        for look in [chain, fork, collider] {
            assert!(look.truth.1.abs() < 0.05, "{:?}", look.truth);
        }
    }

    /// The patterns of the captions are not the luck of one sample: in a dozen others,
    /// holding the middle of a chain or a fork leaves little trend, and holding or
    /// selecting on a collider, or what it causes, makes a clear one; and selecting on
    /// the middle of a chain or fork (above a threshold) only weakens its trend.
    #[test]
    fn the_patterns_hold_in_other_samples() {
        for seed in 1..=12u64 {
            for hold in [Hold::Band, Hold::Above] {
                let mut s = Junctions::create();
                s.reseed(seed);
                s.hold = hold;
                s.refresh();
                let [chain, fork, collider, effect] = [0, 1, 2, 3].map(|k| &s.looks[k]);
                assert!(r(chain.all) > 0.7 && r(fork.all) > 0.6 && r(collider.all).abs() < 0.1 && r(effect.all).abs() < 0.1, "seed {seed}");
                if hold == Hold::Band {
                    assert!(r(chain.held).abs() < 0.3 && r(fork.held).abs() < 0.3, "seed {seed}: {} {}", r(chain.held), r(fork.held));
                    assert!(r(collider.held) < -0.5 && r(effect.held) < -0.3, "seed {seed}: {} {}", r(collider.held), r(effect.held));
                } else {
                    assert!(r(chain.held) > 0.2 && r(chain.held) < r(chain.all) - 0.15, "seed {seed}: chain above {}", r(chain.held));
                    assert!(r(fork.held) > 0.15 && r(fork.held) < r(fork.all) - 0.15, "seed {seed}: fork above {}", r(fork.held));
                    assert!(r(collider.held) < -0.25 && r(effect.held) < -0.15, "seed {seed}: {} {}", r(collider.held), r(effect.held));
                }
                // held exactly, what C causes says less than C itself
                assert!(effect.partial.abs() < collider.partial.abs(), "seed {seed}: {} {}", effect.partial, collider.partial);
            }
        }
    }

    /// The rule (d-separation), against many units: where it says independent the
    /// correlation is near 0, where it says dependent it is clearly not, over every
    /// unit and with the held variable held exactly; and an arrow of strength 0 breaks
    /// the path.
    #[test]
    fn the_rule_says_which_pairs_depend() {
        let noise = deal(11);
        let window = Window { hold: Hold::Band, band: BAND, above: TOP_QUARTER };
        let settings = [(1.5, 1.5, 1.0), (0.5, -1.0, 2.0), (-2.0, 0.7, 0.5), (0.0, 1.5, 1.0), (1.5, 0.0, 1.0), (1.0, 1.0, 0.0), (0.6, 0.8, 1.0)];
        let mut seen = [0usize; 2];
        for (a, b, d) in settings {
            let s = Strengths { a, b, k: d };
            for kind in Kind::ALL {
                let l = look(kind, s, &noise, window);
                for (held, value) in [(false, r(l.all)), (true, l.partial)] {
                    let dependent = kind.dependent(s, held);
                    seen[dependent as usize] += 1;
                    assert!(if dependent { value.abs() > 0.1 } else { value.abs() < 0.06 }, "{kind:?} at {a} {b} {d}, held {held}: r = {value}, the rule says dependent {dependent}");
                }
            }
        }
        assert!(seen[0] > 15 && seen[1] > 15, "{seen:?}");
        // what the heading says, against the rule: dependence comes and goes as it says
        for (a, b, k) in settings {
            let s = Strengths { a, b, k };
            for kind in Kind::ALL {
                let role = kind.role(s);
                let (free, held) = (kind.dependent(s, false), kind.dependent(s, true));
                if role.contains("nothing passes") || role.contains("opens nothing") {
                    assert!(!free && !held, "{kind:?} at {a} {b} {k}: {role}");
                } else if role.contains("blocks the path") {
                    assert!(free && !held, "{kind:?} at {a} {b} {k}: {role}");
                } else {
                    assert!(role.contains("opens") && !free && held, "{kind:?} at {a} {b} {k}: {role}");
                }
            }
        }
        // the rule as the panel words it, for the strengths the scene opens with
        for kind in Kind::ALL {
            assert_eq!(kind.dependent(STRENGTHS, false), matches!(kind, Kind::Chain | Kind::Fork));
            assert_eq!(kind.dependent(STRENGTHS, true), matches!(kind, Kind::Collider | Kind::Effect));
        }
    }

    /// "Holding D opens the path at C too, though less than holding C": with
    /// straight-line effects and bell-curve noise the correlation of X and Y given D is
    /// −ab / √((V − a² + 1/k²)(V − b² + 1/k²)), V the variance of C and k the strength of
    /// C → D; given C itself it is the limit as k grows, −ab / √((1 + a²)(1 + b²)), and always larger. Checked
    /// against many units, and for strengths all over the sliders.
    #[test]
    fn holding_what_a_collider_causes_opens_the_path_less() {
        let given_d = |a: f64, b: f64, d: f64| {
            let v = 1.0 + a * a + b * b;
            -a * b / ((v - a * a + 1.0 / (d * d)) * (v - b * b + 1.0 / (d * d))).sqrt()
        };
        let given_c = |a: f64, b: f64| -a * b / ((1.0 + a * a) * (1.0 + b * b)).sqrt();
        let noise = deal(3);
        let window = Window { hold: Hold::Nothing, band: BAND, above: TOP_QUARTER };
        for (a, b, d) in [(1.5f32, 1.5f32, 1.0f32), (0.7, -1.2, 0.4), (2.0, 1.0, 2.0)] {
            let s = Strengths { a, b, k: d };
            let (c, e) = (look(Kind::Collider, s, &noise, window).partial, look(Kind::Effect, s, &noise, window).partial);
            let (a, b, d) = (a as f64, b as f64, d as f64);
            assert!((c - given_c(a, b)).abs() < 0.03 && (e - given_d(a, b, d)).abs() < 0.03, "{a} {b} {d}: {c} {e} against {} {}", given_c(a, b), given_d(a, b, d));
        }
        let steps = [-2.0, -1.3, -0.6, -0.1, 0.1, 0.6, 1.3, 2.0];
        for a in steps {
            for b in steps {
                for d in steps {
                    assert!(given_d(a, b, d).abs() < given_c(a, b).abs(), "{a} {b} {d}");
                }
            }
        }
    }

    /// Every number drawn or read out stays finite, and the picture says what has
    /// happened: no arrows at all, a band holding a handful of units or none, a band
    /// that holds everyone, a threshold above every unit, as few units as there can be.
    #[test]
    fn degenerate_settings_stay_finite() {
        let cases: [(Strengths, Hold, [f32; 2], f32, usize); 7] = [
            (Strengths { a: 0.0, b: 0.0, k: 0.0 }, Hold::Band, BAND, TOP_QUARTER, UNITS),
            (STRENGTHS, Hold::Band, [3.2, 3.22], TOP_QUARTER, UNITS),
            (STRENGTHS, Hold::Band, [0.0, NARROWEST], TOP_QUARTER, 20),
            (STRENGTHS, Hold::Band, [-3.3, 3.3], TOP_QUARTER, UNITS),
            (STRENGTHS, Hold::Above, BAND, 3.3, UNITS),
            (Strengths { a: 2.0, b: -2.0, k: 2.0 }, Hold::Above, BAND, -3.3, 20),
            (STRENGTHS, Hold::Nothing, BAND, TOP_QUARTER, MOST),
        ];
        for (k, (st, hold, band, above, n)) in cases.into_iter().enumerate() {
            let mut s = Junctions::create();
            (s.s, s.hold, s.band, s.above, s.n, s.truth) = (st, hold, band, above, n, true);
            s.refresh();
            s.glide(1.0 / 60.0);
            for (kind, l) in Kind::ALL.iter().zip(&s.looks) {
                let fits = [l.all, l.held].into_iter().flatten();
                for fit in fits {
                    assert!([fit.r, fit.slope, fit.intercept, fit.from, fit.to].iter().all(|v| v.is_finite()) && fit.n >= 3, "case {k}, {kind:?}: {fit:?}");
                }
                assert!(l.partial.is_finite() && l.truth.0.is_finite() && l.truth.1.is_finite(), "case {k}, {kind:?}");
                assert!(l.x.iter().chain(&l.y).chain(&l.z).all(|v| v.is_finite()) && l.lit_count <= l.x.len());
                // a band of a handful of units draws no line through them, or one, as the count allows
                assert_eq!(l.held.is_some(), hold != Hold::Nothing && l.lit_count >= 3, "case {k}, {kind:?}");
            }
            assert!(s.shown.iter().flatten().all(|v| v.is_finite()) && s.lines.iter().chain(s.small_lines.iter().flatten()).flatten().all(|l| l.at.iter().chain(&l.between).all(|v| v.is_finite())));
        }
        // a band that holds every unit holds nothing: the same trend as all of them
        let mut s = Junctions::create();
        (s.band, s.kind) = ([-3.3, 3.3], Kind::Chain);
        s.refresh();
        let l = look_of(&s, Kind::Chain);
        assert!(l.lit_count == l.x.len() && (r(l.held) - r(l.all)).abs() < 1e-12);
        // no arrows: nothing to see anywhere, and the true effect is none
        let mut s = Junctions::create();
        s.s = Strengths { a: 0.0, b: 0.0, k: 0.0 };
        s.refresh();
        for l in &s.looks {
            assert!(r(l.all).abs() < 0.08 && l.truth.0 == 0.0, "{}", r(l.all));
        }
        // the lines are cut to the square, whatever they are
        for (m, c) in [(0.0, 0.0), (1e9, 0.0), (-3.0, 50.0), (f64::NAN, 0.0), (0.0, f64::INFINITY), (2.0, -1.0)] {
            if let Some([p, q]) = clipped(m, c, -REACH, REACH) {
                assert!([p.0, p.1, q.0, q.1].iter().all(|v| v.is_finite() && v.abs() <= REACH + 1e-9), "{m} {c}: {p:?} {q:?}");
            }
        }
        assert!(clipped(1.0, 0.0, 1.0, 1.0).is_none() && clipped(0.0, 10.0, -1.0, 1.0).is_none());
    }

    /// In every window the pieces lie on the canvas and clear of each other, and the
    /// scatter keeps a useful size where there is room for one.
    #[test]
    fn the_layout_fits_every_window() {
        let inside = |a: &Area, view: Vec2| a.min.x >= -0.5 && a.min.y >= -0.5 && a.max.x <= view.x + 0.5 && a.max.y <= view.y + 0.5;
        let apart = |a: &Area, b: &Area| a.max.x <= b.min.x || b.max.x <= a.min.x || a.max.y <= b.min.y || b.max.y <= a.min.y;
        // the desktop canvas with the panels beside it, without them, the motion
        // check's window, a phone (with the buttons over the picture), a phone held
        // sideways, and a tiny window
        for (view, inset, side) in [
            (Vec2::new(1078.0, 1000.0), 0.0, 500.0),
            (Vec2::new(1600.0, 1000.0), 0.0, 500.0),
            (Vec2::new(578.0, 700.0), 0.0, 260.0),
            (Vec2::new(390.0, 800.0), 46.0, 330.0),
            (Vec2::new(800.0, 390.0), 46.0, 250.0),
            (Vec2::new(240.0, 180.0), 46.0, 0.0),
        ] {
            let lay = Layout::of(view, inset);
            let parts = [lay.graph, lay.words, lay.legend, lay.plot, lay.strip];
            assert!(lay.plot.w() >= side && (lay.plot.w() - lay.plot.h()).abs() < 0.5, "{view}: {:?}", lay.plot);
            if side > 0.0 {
                for (i, a) in parts.iter().chain(&lay.cards).chain(&lay.modes).enumerate() {
                    assert!(inside(a, view), "{view}: part {i} {a:?} off the canvas");
                }
                assert!(parts.iter().all(|p| p.min.y >= inset), "{view}: under the buttons");
                for (i, a) in parts.iter().enumerate() {
                    for b in &parts[i + 1..] {
                        assert!(apart(a, b), "{view}: {a:?} and {b:?} overlap");
                    }
                    for c in &lay.cards {
                        assert!(apart(a, c), "{view}: {a:?} and the small picture {c:?} overlap");
                    }
                }
                for m in &lay.modes {
                    assert!(apart(m, &lay.plot) && apart(m, &lay.strip), "{view}: a button {m:?} overlaps");
                }
            }
            // a point of the canvas and the stage are one and the same
            for p in [Vec2::ZERO, view, Vec2::new(37.0, 410.0)] {
                assert!((lay.px(lay.at(p)) - p).length() < 1e-3);
            }
        }
    }

    /// A change of setting glides into place and then comes to rest, in about the same
    /// time at any frame rate.
    #[test]
    fn a_change_glides_and_comes_to_rest() {
        for hz in [30.0f32, 60.0, 144.0] {
            let mut s = Junctions::create();
            assert!(!s.moving);
            s.set_up(0);
            let mut frames = 0;
            let start = s.shown.clone();
            while s.moving || frames == 0 {
                s.glide(1.0 / hz);
                frames += 1;
                assert!(frames as f32 / hz < 1.5, "still gliding after {frames} frames at {hz} Hz");
                if frames == 1 {
                    // the first frame moves the dots some way, not all the way
                    let target = look_of(&s, Kind::Chain);
                    let (moved, left) = (s.shown[5][0] - start[5][0], target.x[5] as f32 - s.shown[5][0]);
                    assert!(moved.abs() > 0.0 && left.abs() > 0.0 || (target.x[5] as f32 - start[5][0]).abs() < 1e-3, "{moved} {left}");
                }
            }
            let l = look_of(&s, Kind::Chain);
            assert!(s.shown.iter().zip(&l.x).all(|(p, x)| p[0] == *x as f32), "{hz} Hz: not where it belongs at rest");
            assert!(frames as f32 / hz > 0.4, "{hz} Hz: done in {frames} frames");
        }
    }

    /// The numbers behind the captions, for a few samples (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for seed in [SEED, 1, 2, 3] {
            for (k, t) in TRIALS.iter().enumerate() {
                let mut s = run(k);
                s.reseed(seed);
                s.refresh();
                let mut line = format!("seed {seed:>3} {:<36}", t.button);
                for kind in Kind::ALL {
                    let l = look_of(&s, kind);
                    line += &format!(
                        " | {:<9} all {:>6} lit {:>6} ({:>3}) partial {:>6} slopes {:>6} {:>6} truth {:>6}",
                        kind.name().split(' ').next().unwrap_or(""),
                        num(r(l.all), 3),
                        num(r(l.held), 3),
                        l.lit_count,
                        num(l.partial, 3),
                        num(slope(l.all), 3),
                        num(slope(l.held), 3),
                        num(l.truth.0, 3)
                    );
                }
                println!("{line}");
            }
        }
    }
}

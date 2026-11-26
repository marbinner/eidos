//! Correlation is not causation: a common cause makes two things go together whether
//! or not one causes the other, and comparing only units alike in it takes that away.

use super::*;
use crate::math::causal::{correlation, line, regress, residuals, variance};
use glam::DVec2;

pub const DEF: SceneDef = SceneDef {
    id: "confound",
    title: "Correlation is not causation",
    group: GROUP,
    blurb: "Two things that move together, and the hidden third that moves them both.",
    make: |_| Box::new(Confound::create()),
};

/// The values Z takes, each as likely as the others: its mean is 0 and its variance 2.
/// (A few values rather than a continuum, so that "units with the same Z" is exactly
/// that, and a line can be fitted to each group of them.)
const LEVELS: [f64; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
const VAR_Z: f64 = 2.0;
/// The most units a sample can have. Every sample is drawn this large and its first so
/// many are shown, so that a change in their number keeps the units already there.
const MOST: usize = 1000;
/// The fewest units of one Z that a line is fitted to.
const FEWEST: usize = 3;
/// How fast the picture follows a change: the time constant, in seconds, of each of
/// the two eases it goes through (it sets off gently and arrives gently).
const FOLLOW: f64 = 0.08;
/// The colour of a cleared background behind a line, so that it stands out from the
/// dots it crosses.
const UNDER: Color = Color::hex(0x080B14);

/// The colour of the units with the `k`-th value of Z, lowest first: from a deep
/// blue-violet through the topic's purple to a pale lilac, all of them Z's colour, in
/// steps of even lightness.
fn level_color(k: usize) -> Color {
    const RAMP: [u32; 5] = [0x4F5BD5, 0x7B73E7, 0xA78BFA, 0xCDBDFC, 0xF4EEFF];
    Color::hex(RAMP[k.min(4)])
}

// ----- the model -------------------------------------------------------------------------

/// What makes the units: the strengths of the three arrows, and how much X and Y vary
/// on their own. Each unit has a Z, and then
///   X = a Z + sx εX,   Y = b X + c Z + sy εY,
/// with εX and εY its own standard normal draws.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Model {
    a: f64,
    b: f64,
    c: f64,
    sx: f64,
    sy: f64,
}

impl Model {
    fn array(&self) -> [f64; 5] {
        [self.a, self.b, self.c, self.sx, self.sy]
    }

    fn of(v: [f64; 5]) -> Model {
        Model { a: v[0], b: v[1], c: v[2], sx: v[3], sy: v[4] }
    }

    /// One unit's X and Y.
    fn point(&self, d: &Draw) -> DVec2 {
        let z = d.z();
        let x = self.a * z + self.sx * d.u;
        DVec2::new(x, self.b * x + self.c * z + self.sy * d.v)
    }

    /// Var(X) = a² Var(Z) + sx², as the model makes it.
    fn var_x(&self) -> f64 {
        self.a * self.a * VAR_Z + self.sx * self.sx
    }

    /// What Z adds to the slope of the line through all units, as they grow in number:
    /// c a Var(Z) / Var(X). None if X does not vary at all.
    fn added(&self) -> Option<f64> {
        let v = self.var_x();
        (v > 1e-12).then(|| self.c * self.a * VAR_Z / v)
    }

    /// The slope the line through all units tends to: b + c a Var(Z) / Var(X).
    fn naive_limit(&self) -> Option<f64> {
        self.added().map(|d| self.b + d)
    }

    /// Does Z push on both X and Y?
    fn confounded(&self) -> bool {
        self.a != 0.0 && self.c != 0.0
    }
}

/// One unit's own draws: which value of Z it has, and its own variation in X and in Y.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Draw {
    level: usize,
    u: f64,
    v: f64,
}

impl Draw {
    fn z(&self) -> f64 {
        LEVELS[self.level.min(LEVELS.len() - 1)]
    }
}

/// A sample of `MOST` units, the same every time for the same seed.
fn sample(seed: u64) -> Vec<Draw> {
    let mut rng = Rng::new(seed);
    (0..MOST).map(|_| Draw { level: rng.below(LEVELS.len()).min(LEVELS.len() - 1), u: rng.normal() as f64, v: rng.normal() as f64 }).collect()
}

/// The standard error of a slope fitted by least squares, from what is left of the
/// regressor once the other regressors are taken out (`x`, centred) and the fit's
/// residuals `r`, with `k` coefficients fitted: the sandwich form, which holds whether
/// or not the scatter about the line is as wide everywhere (here it is not: the
/// line through all units leaves Z's part in what it misses).
fn standard_error(x: &[f64], r: &[f64], k: usize) -> Option<f64> {
    let n = x.len().min(r.len());
    let sxx: f64 = x.iter().map(|v| v * v).sum();
    if n <= k || sxx <= 0.0 {
        return None;
    }
    let meat: f64 = x.iter().zip(r).map(|(x, r)| x * x * r * r).sum();
    let se = (meat * n as f64 / (n - k) as f64).sqrt() / sxx;
    se.is_finite().then_some(se)
}

/// What the sample as asked for says: the numbers of the scene.
#[derive(Clone, Debug, PartialEq)]
struct Stats {
    n: usize,
    /// The line through all units: slope, intercept and the slope's standard error.
    /// None if X does not vary.
    all: Option<(f64, f64, f64)>,
    /// Y fitted on X and Z together: the coefficient of X and its standard error. None
    /// if X does not vary among units with the same Z.
    held: Option<(f64, f64)>,
    /// The line within each group of units with the same Z (slope, intercept), and how
    /// many units each has. None for a group of fewer than `FEWEST`, or with no spread in X.
    groups: [Option<(f64, f64)>; 5],
    counts: [usize; 5],
    correlation: f64,
}

/// Is there any spread in these values?
fn spread(x: &[f64]) -> bool {
    variance(x) > 1e-12
}

fn stats(m: &Model, draws: &[Draw], n: usize) -> Stats {
    let n = n.min(draws.len());
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut zs = Vec::with_capacity(n);
    let mut by: [(Vec<f64>, Vec<f64>); 5] = Default::default();
    for d in &draws[..n] {
        let p = m.point(d);
        xs.push(p.x);
        ys.push(p.y);
        zs.push(d.z());
        let group = &mut by[d.level.min(4)];
        group.0.push(p.x);
        group.1.push(p.y);
    }
    let all = (n >= FEWEST && spread(&xs)).then(|| {
        let (slope, intercept) = line(&xs, &ys);
        let mx = xs.iter().sum::<f64>() / n as f64;
        let centred: Vec<f64> = xs.iter().map(|x| x - mx).collect();
        let r: Vec<f64> = xs.iter().zip(&ys).map(|(x, y)| y - slope * x - intercept).collect();
        (slope, intercept, standard_error(&centred, &r, 2).unwrap_or(f64::NAN))
    });
    // (X less what Z says of it: nothing left means that X is fixed by Z)
    let apart = residuals(&[&zs], &xs);
    let held = if n > FEWEST && spread(&apart) && variance(&apart) > 1e-9 * variance(&xs) {
        regress(&[&xs, &zs], &ys).map(|w| {
            let r = residuals(&[&xs, &zs], &ys);
            (w[0], standard_error(&apart, &r, 3).unwrap_or(f64::NAN))
        })
    } else {
        None
    };
    let groups = by.each_ref().map(|(x, y)| (x.len() >= FEWEST && spread(x)).then(|| line(x, y)));
    let counts = by.each_ref().map(|(x, _)| x.len());
    Stats { n, all, held, groups, counts, correlation: correlation(&xs, &ys) }
}

// ----- the settings the scene opens with and the things to try -----------------------------

/// Everyday names for the variables, for the things to try.
struct Story {
    /// What one unit is.
    unit: &'static str,
    x: &'static str,
    y: &'static str,
    z: &'static str,
    /// What Z's lowest and highest values stand for.
    low: &'static str,
    high: &'static str,
}

const STORIES: [Story; 3] = [
    Story { unit: "day", x: "ice cream sold", y: "drownings", z: "temperature", low: "cold", high: "hot" },
    Story { unit: "person", x: "sunscreen used", y: "sunburn", z: "hours in the sun", low: "few", high: "many" },
    Story { unit: "patient", x: "dose of medicine", y: "health after", z: "how ill", low: "slightly", high: "gravely" },
];

/// Everything a thing to try sets.
#[derive(Clone, Copy, Debug)]
struct Setup {
    story: Option<usize>,
    a: f32,
    b: f32,
    c: f32,
    sx: f32,
    sy: f32,
    n: usize,
    hold: bool,
    seed: u64,
}

const OPENING: Setup = Setup { story: None, a: 1.0, b: 0.5, c: 1.0, sx: 0.6, sy: 0.4, n: 400, hold: true, seed: 1 };

const TRIALS: [(&str, &str, Setup); 5] = [
    (
        "a trend with no cause",
        r"Each dot is a day. Hot days bring more ice cream sales and more swimming, so more drownings. Ice cream drowns nobody ($b = 0$: no arrow from $X$ to $Y$), yet the line through all days rises (slope $0.88$): temperature's doing, but for sampling noise. Among days of one temperature (one colour) the lines are flat but for sampling noise; holding temperature fixed gives $0.00 \pm 0.03$.",
        Setup { story: Some(0), a: 1.0, b: 0.0, c: 1.0, sx: 0.6, sy: 0.4, n: 400, hold: true, seed: 2 },
    ),
    (
        "an effect hidden",
        r"Those who spend longer in the sun use more sunscreen, and burn more. Sunscreen protects ($b = -0.85$), yet the line through everyone is flat (slope $-0.02$): those who use the most are those most in the sun. Among people with equal hours in the sun, more sunscreen goes with less burn: holding $Z$ fixed gives $-0.85 \pm 0.03$.",
        Setup { story: Some(1), a: 1.0, b: -0.85, c: 1.0, sx: 0.6, sy: 0.4, n: 400, hold: true, seed: 3 },
    ),
    (
        "Simpson's reversal",
        r"Doctors give more of a medicine to the more ill, and the more ill do worse. Over all patients more medicine goes with worse health (slope $-1.07$), yet within every grade of illness it goes with better health ($0.55 \pm 0.03$ holding $Z$ fixed; the true effect is $0.6$). Which is right is not in the data but in the arrows: here illness decides the dose, so compare like with like.",
        Setup { story: Some(2), a: 1.0, b: 0.6, c: -2.0, sx: 0.6, sy: 0.4, n: 400, hold: true, seed: 4 },
    ),
    (
        "no confounder",
        r"The same medicine, but each dose is drawn by lot: no arrow from $Z$ to $X$. Illness still changes health but no longer goes with the dose, so the line through all patients has the true slope but for sampling noise ($0.53 \pm 0.08$, against $b = 0.6$). Holding illness fixed finds it more precisely: $0.58 \pm 0.01$. A randomised trial works like this.",
        Setup { story: Some(2), a: 0.0, b: 0.6, c: -2.0, sx: 1.0, sy: 0.4, n: 1000, hold: true, seed: 5 },
    ),
    (
        "chance alone",
        "Nothing causes anything: no arrows at all. Yet these ten units go together (correlation 0.65, slope 0.66), by chance. Press \"new sample\" a few times: about two samples in five slope by 0.3 or more, up or down. With a thousand units the slope stays near 0.",
        Setup { story: None, a: 0.0, b: 0.0, c: 0.0, sx: 1.0, sy: 1.0, n: 10, hold: false, seed: 31 },
    ),
];

// ----- following a change ----------------------------------------------------------------

/// Numbers as drawn, on their way to the ones asked for: two eases one after the
/// other (`follow` in `linalg/packing.rs`), exactly there in the end.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Follow<const N: usize> {
    between: [f64; N],
    shown: [f64; N],
}

impl<const N: usize> Follow<N> {
    fn at(v: [f64; N]) -> Self {
        Follow { between: v, shown: v }
    }

    /// Move on by `dt` seconds towards `to` (all the way if no time passes: a change
    /// must still be answered while the scene's time stands still), and be exactly
    /// there once within `close` of it. True while still on the way.
    fn toward(&mut self, to: [f64; N], dt: f64, close: f64) -> bool {
        if *self == Follow::at(to) {
            return false;
        }
        if dt <= 0.0 || !dt.is_finite() {
            *self = Follow::at(to);
            return false;
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        let mut apart = 0.0f64;
        for i in 0..N {
            self.between[i] += (to[i] - self.between[i]) * k;
            self.shown[i] += (self.between[i] - self.shown[i]) * k;
            apart = apart.max((to[i] - self.between[i]).abs()).max((to[i] - self.shown[i]).abs());
        }
        // (exactly there in the end; and there at once if anything went wrong on the way)
        if apart < close || !apart.is_finite() {
            *self = Follow::at(to);
            return false;
        }
        true
    }

    fn rests_at(&self, to: [f64; N]) -> bool {
        *self == Follow::at(to)
    }
}

// ----- what is drawn ---------------------------------------------------------------------

/// A unit as drawn: where it is, its value of Z, and how much of it there is (less
/// than 1 while it fades in or out).
#[derive(Clone, Copy, Debug)]
struct Unit {
    p: DVec2,
    level: usize,
    w: f64,
}

/// Weighted sums for a line through units that may be partly there: the picture fits
/// its lines with them, so that a unit fading in moves the lines a little at a time.
/// (With every weight 1 they give exactly the least-squares line.)
#[derive(Clone, Copy, Debug, Default)]
struct Sums {
    w: f64,
    x: f64,
    y: f64,
    xx: f64,
    xy: f64,
}

impl Sums {
    fn add(&mut self, p: DVec2, w: f64) {
        self.w += w;
        self.x += w * p.x;
        self.y += w * p.y;
        self.xx += w * p.x * p.x;
        self.xy += w * p.x * p.y;
    }

    fn mean(&self) -> Option<DVec2> {
        (self.w > 1e-9).then(|| DVec2::new(self.x, self.y) / self.w)
    }

    /// The variance of x.
    fn var_x(&self) -> f64 {
        self.mean().map_or(0.0, |m| (self.xx / self.w - m.x * m.x).max(0.0))
    }

    /// The least-squares slope; None if x does not vary.
    fn slope(&self) -> Option<f64> {
        let m = self.mean()?;
        let vxx = self.xx / self.w - m.x * m.x;
        let slope = (self.xy / self.w - m.x * m.y) / vxx;
        (vxx > 1e-12 * (1.0 + m.x * m.x) && slope.is_finite()).then_some(slope)
    }
}

/// The lines the picture draws: through all units, and within each value of Z.
struct Fits {
    all: Sums,
    groups: [Sums; 5],
}

impl Fits {
    fn of(units: &[Unit]) -> Fits {
        let mut fits = Fits { all: Sums::default(), groups: [Sums::default(); 5] };
        for u in units {
            fits.all.add(u.p, u.w);
            fits.groups[u.level.min(4)].add(u.p, u.w);
        }
        fits
    }
}

// ----- where everything goes ---------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct Area {
    min: Vec2,
    max: Vec2,
}

impl Area {
    /// (Never upside down, and never of no size: a tiny window squeezes, it does not fold.)
    fn new(a: Vec2, b: Vec2) -> Area {
        let min = a.min(b);
        Area { min, max: a.max(b).max(min + Vec2::ONE) }
    }

    fn size(&self) -> Vec2 {
        self.max - self.min
    }

    fn mid(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    fn holds(&self, p: Vec2) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }
}

/// Where everything goes on a canvas, in logical pixels. On a wide canvas the graph,
/// the buttons and the numbers make a column on the left and the scatter is on the
/// right; on one about as tall as it is wide, the graph and the numbers make a row on
/// top with the scatter under it; on a narrow one (a phone held upright) the graph is
/// on top with the buttons beside it, the scatter under them and the numbers under that.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    narrow: bool,
    /// Text sizes.
    size: f32,
    small: f32,
    graph: Area,
    radius: f32,
    /// "hold Z fixed" and "new sample".
    buttons: [Area; 2],
    plot: Area,
    /// Z's key and Y's name go on one line above the scatter, or (narrow) on two.
    head: [f32; 2],
    /// The room right of the scatter for the names of its lines.
    margin: f32,
    /// None on a canvas too small to spare the room.
    numbers: Option<Area>,
}

/// The scatter in the room from `lo` to `hi`: no more than `wide` times as wide as it
/// is tall nor `tall` times as tall as wide (a scatter stretched either way makes every
/// slope look steeper or flatter than it is), in the middle across and at the top.
fn scatter_in(lo: Vec2, hi: Vec2, wide: f32, tall: f32) -> Area {
    let room = (hi - lo).max(Vec2::splat(10.0));
    let (w, h) = (room.x.min(room.y * wide), room.y.min(room.x * tall));
    let x = lo.x + (room.x - w) * 0.5;
    Area::new(Vec2::new(x, lo.y), Vec2::new(x + w, lo.y + h))
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let inset = inset.max(0.0);
        // (room under the scatter for its numbers and X's name, and left of it for Y's numbers)
        let (under, beside) = (44.0, 46.0);
        if view.x >= 1.3 * view.y && view.x >= 760.0 {
            let (pad, top) = (18.0, inset + 14.0);
            let col = (view.x * 0.3).clamp(250.0, 360.0);
            let gh = (view.y * 0.26).clamp(130.0, 230.0);
            let graph = Area::new(Vec2::new(pad, top), Vec2::new(pad + col, top + gh));
            let by = graph.max.y + 8.0;
            let bw = (col - 10.0) * 0.5;
            let buttons = [Area::new(Vec2::new(pad, by), Vec2::new(pad + bw, by + 28.0)), Area::new(Vec2::new(pad + bw + 10.0, by), Vec2::new(pad + col, by + 28.0))];
            let numbers_top = by + 28.0 + 22.0;
            let numbers = (view.y - pad - numbers_top >= 120.0).then(|| Area::new(Vec2::new(pad, numbers_top), Vec2::new(pad + col, view.y - pad)));
            let margin = 100.0;
            let plot = scatter_in(Vec2::new(pad + col + 40.0 + beside, top + 30.0), Vec2::new(view.x - pad - margin, view.y - pad - under), 1.3, 1.05);
            Layout { narrow: false, size: 13.0, small: 11.5, graph, radius: 19.0, buttons, plot, head: [plot.min.y - 16.0; 2], margin, numbers }
        } else if view.x >= 560.0 {
            let (pad, top) = (18.0, inset + 14.0);
            let gw = (view.x * 0.38).clamp(240.0, 380.0);
            let gh = (view.y * 0.22).clamp(130.0, 220.0);
            let graph = Area::new(Vec2::new(pad, top), Vec2::new(pad + gw, top + gh));
            let by = graph.max.y + 6.0;
            let bw = (gw - 10.0) * 0.5;
            let buttons = [Area::new(Vec2::new(pad, by), Vec2::new(pad + bw, by + 28.0)), Area::new(Vec2::new(pad + bw + 10.0, by), Vec2::new(pad + gw, by + 28.0))];
            let numbers = Some(Area::new(Vec2::new(pad + gw + 32.0, top + 4.0), Vec2::new(view.x - pad, by + 28.0)));
            let margin = 100.0;
            let plot = scatter_in(Vec2::new(pad + beside, by + 28.0 + 46.0), Vec2::new(view.x - pad - margin, view.y - pad - under), 1.3, 1.05);
            Layout { narrow: false, size: 13.0, small: 11.5, graph, radius: 19.0, buttons, plot, head: [plot.min.y - 16.0; 2], margin, numbers }
        } else {
            let (pad, top) = (12.0, inset + 8.0);
            let bw = (view.x * 0.3).clamp(84.0, 124.0);
            let gh = (view.y * 0.19).clamp(60.0, 140.0);
            let graph = Area::new(Vec2::new(pad, top), Vec2::new(view.x - pad - bw - 8.0, top + gh));
            let (bx, mid) = (view.x - pad - bw, graph.mid().y);
            let buttons = [Area::new(Vec2::new(bx, mid - 31.0), Vec2::new(view.x - pad, mid - 5.0)), Area::new(Vec2::new(bx, mid + 5.0), Vec2::new(view.x - pad, mid + 31.0))];
            let head = [graph.max.y + 12.0, graph.max.y + 31.0];
            let numbers_h = if view.y >= 560.0 { 136.0 } else { 0.0 };
            let margin = 66.0;
            let plot = scatter_in(Vec2::new(pad + 34.0, head[1] + 14.0), Vec2::new(view.x - pad - margin, view.y - pad - numbers_h - 40.0), 1.3, 1.15);
            let numbers = (numbers_h > 0.0).then(|| Area::new(Vec2::new(pad, plot.max.y + 44.0), Vec2::new(view.x - pad, view.y - pad)));
            Layout { narrow: true, size: 11.5, small: 10.5, graph, radius: 15.0, buttons, plot, head, margin, numbers }
        }
    }
}

/// A button drawn over the picture (logical pixels), lit when `chosen`: true when clicked.
fn hud_button(f: &mut Frame, at: Area, text: &str, chosen: bool, pointer: Option<Vec2>, size: f32) -> bool {
    let over = pointer.is_some_and(|p| at.holds(p));
    let fill = match (chosen, over) {
        (true, _) => Color::hex(0x3A4C78),
        (false, true) => Color::hex(0x33405E),
        (false, false) => Color::hex(0x222B40),
    };
    hud_rect(f, at.min, at.max, fill);
    f.text_px(at.mid(), text, size, pal::TEXT, Align2::CENTER_CENTER);
    over && f.ptr.clicked
}

/// The part of the segment from `a` to `b` inside the box `lo`–`hi`, if any.
fn clip(a: DVec2, b: DVec2, lo: DVec2, hi: DVec2) -> Option<(DVec2, DVec2)> {
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    let d = b - a;
    for (p, q) in [(-d.x, a.x - lo.x), (d.x, hi.x - a.x), (-d.y, a.y - lo.y), (d.y, hi.y - a.y)] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
        }
    }
    (t0 <= t1 && t0.is_finite() && t1.is_finite()).then(|| (a + d * t0, a + d * t1))
}

/// The largest "nice" number (1, 2 or 5 times a power of ten) that is at most `x`.
fn nice_below(x: f32) -> f32 {
    let up = nice_step(x);
    if up <= x {
        up
    } else if (up / 10f32.powf(up.log10().floor()) - 5.0).abs() < 0.01 {
        up * 0.4
    } else {
        up * 0.5
    }
}

/// A number written for a product: in brackets if it is negative.
fn factor(v: f64) -> String {
    let text = num(v, 2);
    if text.starts_with('−') {
        format!("({text})")
    } else {
        text
    }
}

/// A text broken into lines no wider than `width` logical pixels (formulas kept whole).
fn wrap(f: &Frame, text: &str, size: f32, width: f32) -> Vec<String> {
    let mut words: Vec<String> = Vec::new();
    let mut inside = false;
    let mut word = String::new();
    for ch in text.chars() {
        if ch == '$' {
            inside = !inside;
        }
        if ch == ' ' && !inside {
            words.push(std::mem::take(&mut word));
        } else {
            word.push(ch);
        }
    }
    words.push(word);
    let mut lines: Vec<String> = Vec::new();
    for w in words.into_iter().filter(|w| !w.is_empty()) {
        match lines.last_mut() {
            Some(last) if f.text_size(&format!("{last} {w}"), size).x <= width => {
                last.push(' ');
                last.push_str(&w);
            }
            _ => lines.push(w),
        }
    }
    lines
}

// ----- the scene ---------------------------------------------------------------------------

pub struct Confound {
    /// The settings asked for, as the controls hold them.
    a: f32,
    b: f32,
    c: f32,
    sx: f32,
    sy: f32,
    n: usize,
    hold: bool,
    story: Option<usize>,
    seed: u64,
    draws: Vec<Draw>,
    /// The sample being replaced, while it fades out.
    old: Option<Vec<Draw>>,
    /// What is drawn, on its way to what is asked for: the model; how much of each
    /// unit is there (1 for the first `n`, 0 for the rest: those that come or go do so
    /// together, so that the lines move evenly); how far the new sample has faded in
    /// over the old; how far the lines within each Z have faded in.
    model: Follow<5>,
    weights: Vec<Follow<1>>,
    fresh: Follow<1>,
    held: Follow<1>,
    /// The part of the plane the scatter shows (lowest x, lowest y, highest x, highest
    /// y): where it is heading, and as drawn. Not yet framed before the first frame.
    aim: [f64; 4],
    range: Follow<4>,
    framed: bool,
    /// The numbers of the sample as asked for, and what they were worked out from.
    known: Option<([f32; 5], usize, u64, Stats)>,
    /// Where each line was last drawn on the screen (the five within each Z, the true
    /// effect's and the one through all units).
    kept: [(Vec2, Vec2); 7],
}

impl Confound {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let mut s = Confound {
            a: 0.0,
            b: 0.0,
            c: 0.0,
            sx: 0.0,
            sy: 0.0,
            n: 0,
            hold: false,
            story: None,
            seed: 0,
            draws: Vec::new(),
            old: None,
            model: Follow::at([0.0; 5]),
            weights: vec![Follow::at([0.0]); MOST],
            fresh: Follow::at([1.0]),
            held: Follow::at([0.0]),
            aim: [-1.0, -1.0, 1.0, 1.0],
            range: Follow::at([-1.0, -1.0, 1.0, 1.0]),
            framed: false,
            known: None,
            kept: [(Vec2::ZERO, Vec2::ZERO); 7],
        };
        s.apply(&OPENING);
        s.settle();
        s
    }

    /// The model asked for.
    fn goal(&self) -> Model {
        Model { a: self.a as f64, b: self.b as f64, c: self.c as f64, sx: self.sx as f64, sy: self.sy as f64 }
    }

    /// Take every setting of `s` (the picture then follows).
    fn apply(&mut self, s: &Setup) {
        (self.a, self.b, self.c, self.sx, self.sy, self.n, self.hold, self.story) = (s.a, s.b, s.c, s.sx, s.sy, s.n.clamp(5, MOST), s.hold, s.story);
        if s.seed != self.seed || self.draws.len() != MOST {
            self.resample(s.seed);
        }
    }

    fn set_up(&mut self, k: usize) {
        if let Some((_, _, setup)) = TRIALS.get(k) {
            self.apply(setup);
        }
    }

    /// Draw the units afresh from `seed`; the old ones fade out as the new fade in.
    fn resample(&mut self, seed: u64) {
        let fresh = sample(seed);
        if self.draws.len() == MOST {
            self.old = Some(std::mem::replace(&mut self.draws, fresh));
            self.fresh = Follow::at([0.0]);
        } else {
            self.draws = fresh;
        }
        self.seed = seed;
    }

    /// Put the picture where the settings say at once (as the scene opens).
    fn settle(&mut self) {
        self.model = Follow::at(self.goal().array());
        for (i, w) in self.weights.iter_mut().enumerate() {
            *w = Follow::at([if i < self.n { 1.0 } else { 0.0 }]);
        }
        self.fresh = Follow::at([1.0]);
        self.old = None;
        self.held = Follow::at([if self.hold { 1.0 } else { 0.0 }]);
    }

    /// The numbers of the sample as asked for.
    fn stats(&mut self) -> Stats {
        let key = ([self.a, self.b, self.c, self.sx, self.sy], self.n, self.seed);
        match &self.known {
            Some((p, n, seed, s)) if (*p, *n, *seed) == key => s.clone(),
            _ => {
                let s = stats(&self.goal(), &self.draws, self.n);
                self.known = Some((key.0, key.1, key.2, s.clone()));
                s
            }
        }
    }

    /// Move what is drawn on by `dt` seconds towards what is asked for.
    fn advance(&mut self, dt: f64) {
        self.model.toward(self.goal().array(), dt, 1e-5);
        for (i, w) in self.weights.iter_mut().enumerate() {
            w.toward([if i < self.n { 1.0 } else { 0.0 }], dt, 1e-4);
        }
        self.held.toward([if self.hold { 1.0 } else { 0.0 }], dt, 1e-3);
        if self.old.is_some() && !self.fresh.toward([1.0], dt, 1e-4) {
            self.old = None;
        }
    }

    fn moving(&self) -> bool {
        !self.model.rests_at(self.goal().array())
            || self.weights.iter().enumerate().any(|(i, w)| !w.rests_at([if i < self.n { 1.0 } else { 0.0 }]))
            || self.old.is_some()
            || !self.held.rests_at([if self.hold { 1.0 } else { 0.0 }])
            || !self.range.rests_at(self.aim)
    }

    /// The units as drawn: those of the sample that are there or on their way (every
    /// one of these, also one that is all but gone, so that the same units are drawn
    /// all the while some come or go), and those of the old sample while it fades out.
    fn units(&self) -> Vec<Unit> {
        let m = Model::of(self.model.shown);
        let fresh = self.fresh.shown[0].clamp(0.0, 1.0);
        let mut units = Vec::with_capacity(2 * self.n);
        let mut add = |draws: &[Draw], share: f64| {
            for (d, w) in draws.iter().zip(&self.weights).filter(|(_, w)| w.shown[0] > 0.0 || w.between[0] > 0.0) {
                units.push(Unit { p: m.point(d), level: d.level.min(4), w: share * w.shown[0].clamp(0.0, 1.0) });
            }
        };
        add(&self.draws, if self.old.is_some() { fresh } else { 1.0 });
        if let Some(old) = &self.old {
            add(old, 1.0 - fresh);
        }
        units
    }

    /// The part of the plane the scatter should show for the settings asked for (lowest
    /// x, lowest y, highest x, highest y): all of their units and where the model puts
    /// nearly all units, with a little room round them.
    fn wanted(&self) -> [f64; 4] {
        let m = self.goal();
        let hx = (2.0 * m.a.abs() + 2.3 * m.sx).max(0.5);
        let hy = (2.0 * (m.a * m.b + m.c).abs() + 2.3 * (m.b * m.b * m.sx * m.sx + m.sy * m.sy).sqrt()).max(0.5);
        let (mut lo, mut hi) = (DVec2::new(-hx, -hy), DVec2::new(hx, hy));
        for d in self.draws.iter().take(self.n) {
            let p = m.point(d);
            lo = lo.min(p);
            hi = hi.max(p);
        }
        let pad = (hi - lo) * 0.04;
        [lo.x - pad.x, lo.y - pad.y, hi.x + pad.x, hi.y + pad.y]
    }

    /// Head the axes for what the settings asked for will show, and move them on by `dt`
    /// seconds. What they are heading for is kept while it still shows that well, so
    /// that small changes (a slider moved a little) do not move the axes at all; a
    /// change of setting is followed in one glide, not in several.
    fn frame_on(&mut self, dt: f64) {
        let [x0, y0, x1, y1] = self.wanted();
        let (lo, hi) = (DVec2::new(x0, y0), DVec2::new(x1, y1));
        let [a0, b0, a1, b1] = self.aim;
        let (alo, ahi) = (DVec2::new(a0, b0), DVec2::new(a1, b1));
        let shows = alo.cmple(lo).all() && ahi.cmpge(hi).all() && ((ahi - alo) / (hi - lo)).max_element() < 1.6;
        if !shows && lo.is_finite() && hi.is_finite() {
            // (a little more than is needed, so that the next small change is not a new frame)
            let extra = (hi - lo) * 0.04;
            self.aim = [lo.x - extra.x, lo.y - extra.y, hi.x + extra.x, hi.y + extra.y];
        }
        if self.framed {
            self.range.toward(self.aim, dt, 2e-5 * (hi - lo).max_element());
        } else {
            self.range = Follow::at(self.aim);
            self.framed = true;
        }
    }
}

impl Scene for Confound {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    /// Moving only while the picture follows a change.
    fn animated(&self) -> bool {
        self.moving()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("the buttons over the picture: compare like with like, or draw a new sample")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "Two things can go together because one causes the other, because something else causes both, or by chance (and, as a later scene shows, because of how the units were picked). The trend in the data alone does not say which. $X$ causes $Y$ if changing $X$, and nothing else, changes $Y$. Here a third variable $Z$ pushes on both:");
        ui::equation(ui, r"X = aZ + \sigma_X \varepsilon_X, \qquad Y = bX + cZ + \sigma_Y \varepsilon_Y");
        ui::note(ui, r"Each dot is one unit (a day, a patient), coloured by its $Z$: one of $-2, -1, 0, 1, 2$, each as likely. $\varepsilon_X$ and $\varepsilon_Y$ are the unit's own variation, bell-shaped. The pink line is fitted to all units; the green one has the slope $b$, the true effect of $X$ on $Y$.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The arrows");
        let mut arrows = ui::slider(ui, r"$a$, the arrow $Z \to X$", &mut self.a, -2.0..=2.0);
        arrows |= ui::slider(ui, r"$b$, the arrow $X \to Y$", &mut self.b, -2.0..=2.0);
        arrows |= ui::slider(ui, r"$c$, the arrow $Z \to Y$", &mut self.c, -2.0..=2.0);
        if arrows {
            // (to the hundredth the arrows show, so that 0 is reached and is 0; and the
            // everyday names belong to the numbers of their thing to try)
            for v in [&mut self.a, &mut self.b, &mut self.c] {
                *v = (*v * 100.0).round() / 100.0;
            }
            self.story = None;
        }
        ui::note(ui, "Each says how much the variable at the arrow's head rises when the one at its tail rises by 1 and nothing else changes. At 0 the arrow is gone.");
        ui::section(ui, "The units");
        ui::slider_int_log(ui, "units", &mut self.n, 5..=MOST);
        if ui::slider(ui, r"$\sigma_X$, how much $X$ varies on its own", &mut self.sx, 0.0..=1.5) {
            self.sx = (self.sx * 100.0).round() / 100.0;
        }
        if ui::slider(ui, r"$\sigma_Y$, how much $Y$ varies on its own", &mut self.sy, 0.0..=1.5) {
            self.sy = (self.sy * 100.0).round() / 100.0;
        }
        let mut hold = usize::from(self.hold);
        if ui::choice(ui, &["all units together", "hold $Z$ fixed"], &mut hold) {
            self.hold = hold == 1;
        }
        ui::note(ui, r"Holding $Z$ fixed: a line is fitted to each group of units with the same $Z$, in their colour. It can only be done if $Z$ was measured.");
        if ui.button("new sample").clicked() {
            self.resample(self.seed + 1);
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (CAUSE, r"$X$, the cause asked about, across"),
                (EFFECT, r"$Y$, its effect, up"),
                (THIRD, r"$Z$, the third variable: the dots' colours, from $-2$ (deep) to $2$ (pale)"),
                (NAIVE, r"pink: the line fitted to all units, what the data suggest read naively"),
                (TRUTH, r"green: the slope $b$, the true effect, drawn through the middle of the cloud; in the graph, the arrow $X \to Y$"),
                (level_color(3), r"lines in the dots' colours: fitted within one value of $Z$"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Cause:", r"$X$ causes $Y$ if changing $X$, and nothing else, changes $Y$.");
        ui::term(ui, "Confounder:", r"a common cause of $X$ and $Y$, here $Z$. It makes them go together whether or not one causes the other.");
        ui::term(ui, "Holding fixed, adjusting for:", r"comparing only units with the same $Z$, or fitting $Y$ on $X$ and $Z$ together. Here that takes the confounding away; it needs $Z$ to have been measured.");
        ui::term(ui, "Simpson's paradox:", "a trend that goes one way within every group and the other way when the groups are put together.");
        ui::term(ui, "Sampling noise:", r"how much a slope fitted to a sample changes from one sample to the next. The number after $\pm$ is its typical size (the standard error).");

        ui::section(ui, "Numbers");
        let s = self.stats();
        let m = self.goal();
        let slope = |v: Option<(f64, f64)>| v.map_or("none".to_string(), |(v, se)| format!(r"${} \pm {}$", num(v, 2), num(se, 2)));
        ui::readout_colored(ui, "slope of the line through all units", slope(s.all.map(|(v, _, se)| (v, se))), NAIVE);
        ui::readout_colored(ui, "true effect $b$", format!("${}$", num(m.b, 2)), TRUTH);
        ui::readout_colored(ui, "slope holding $Z$ fixed", slope(s.held), level_color(3));
        ui::readout(ui, "correlation of $X$ and $Y$", format!("${}$", num(s.correlation, 2)));
        ui::readout(ui, r"$\operatorname{Var}(X) = a^2 \operatorname{Var}(Z) + \sigma_X^2$", format!("${}$", num(m.var_x(), 2)));
        ui::readout(ui, r"$b + c\,a \operatorname{Var}(Z) / \operatorname{Var}(X)$", m.naive_limit().map_or("none".to_string(), |v| format!("${}$", num(v, 2))));
        ui::note(ui, r"The last is the slope the pink line tends to as the units grow in number: the effect plus what the confounder adds. $\operatorname{Var}(Z) = 2$. The slope holding $Z$ fixed is the coefficient of $X$ when $Y$ is fitted on $X$ and $Z$ together.");
        ui::see_also(ui, &[("dosee", "seeing and doing: what an intervention is"), ("junctions", "chains, forks and colliders"), ("adjust", "what to control for, and what not"), ("regress", "least-squares fitting")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset);
        let pointer = f.ptr.pos.map(|p| p / f.px);
        if pointer.is_some_and(|p| lay.buttons.iter().any(|b| b.holds(p))) {
            f.handles.claim_clicks();
        }
        if hud_button(f, lay.buttons[0], "hold $Z$ fixed", self.hold, pointer, lay.small + 0.5) {
            self.hold = !self.hold;
        }
        if hud_button(f, lay.buttons[1], "new sample", false, pointer, lay.small + 0.5) {
            self.resample(self.seed + 1);
        }
        let dt = f.dt as f64;
        self.advance(dt);
        let units = self.units();
        self.frame_on(dt);
        let fits = Fits::of(&units);
        let stats = self.stats();
        self.draw_graph(f, &lay);
        self.draw_plot(f, &lay, &units, &fits);
        self.draw_numbers(f, &lay, &stats);
    }
}

impl Confound {
    /// The causal graph: Z above, X and Y below, each arrow as thick as it is strong
    /// and gone at 0, its strength written beside it.
    fn draw_graph(&self, f: &mut Frame, lay: &Layout) {
        let (g, r) = (lay.graph, lay.radius);
        let story = self.story.and_then(|k| STORIES.get(k));
        let cx = g.mid().x;
        let z_at = Vec2::new(cx, g.min.y + r + 4.0);
        let low = g.max.y - r - lay.small - 8.0;
        let half = ((g.size().x * 0.5 - r - 8.0).min((low - z_at.y) * 1.3)).max(r);
        let nodes = [
            Node { at: Vec2::new(cx - half, low), name: "$X$", color: CAUSE },
            Node { at: Vec2::new(cx + half, low), name: "$Y$", color: EFFECT },
            Node { at: z_at, name: "$Z$", color: THIRD },
        ];
        let m = Model::of(self.model.shown);
        let width = |s: f64| 1.3 + 1.4 * s.abs().min(2.0) as f32;
        // (an arrow fades out as its strength goes to 0, and at 0 is not seen at all; it
        // is still drawn, so that what is drawn stays the same while the others change)
        let links = [(2, 0, m.a, THIRD), (2, 1, m.c, THIRD), (0, 1, m.b, TRUTH)].map(|(from, to, s, color)| (from, to, Link::Arrow, color.a((s.abs() / 0.15).min(1.0) as f32), width(s)));
        hud_graph(f, &nodes, &links, r);
        // the strengths, outside the triangle
        let centre = (nodes[0].at + nodes[1].at + nodes[2].at) / 3.0;
        for (from, to, name, value, color) in [(2, 0, "a", self.a, THIRD), (2, 1, "c", self.c, THIRD), (0, 1, "b", self.b, TRUTH)] {
            let (p, q) = (nodes[from].at, nodes[to].at);
            let mid = (p + q) * 0.5;
            let mut out = (q - p).perp().normalize_or_zero();
            if out.dot(mid - centre) < 0.0 {
                out = -out;
            }
            let text = format!("${name} = {}$", num(value as f64, 2));
            let color = if value == 0.0 { pal::FAINT } else { color };
            let align = if out.x < -0.3 {
                Align2::RIGHT_CENTER
            } else if out.x > 0.3 {
                Align2::LEFT_CENTER
            } else {
                Align2::CENTER_TOP
            };
            f.text_px(mid + out * 7.0, text, lay.size, color, align);
        }
        // the everyday names, under X and Y (kept on the canvas) and beside Z
        if let Some(s) = story {
            let view = f.view_size();
            for (node, name) in [(nodes[0], s.x), (nodes[1], s.y)] {
                let half = f.text_size(name, lay.small).x * 0.5;
                let x = node.at.x.max(half + 4.0).min(view.x - half - 4.0);
                f.text_px(Vec2::new(x, node.at.y + r + 4.0), name, lay.small, node.color, Align2::CENTER_TOP);
            }
            f.text_px(z_at + Vec2::new(r + 8.0, 0.0), s.z, lay.small, THIRD, Align2::LEFT_CENTER);
        }
    }

    /// The scatter of Y against X, its lines, and their names.
    fn draw_plot(&mut self, f: &mut Frame, lay: &Layout, units: &[Unit], fits: &Fits) {
        let p = lay.plot;
        let story = self.story.and_then(|k| STORIES.get(k));
        hud_panel(f, p.min - Vec2::splat(6.0), p.max + Vec2::splat(6.0));
        let [x0, y0, x1, y1] = self.range.shown;
        let (lo, hi) = (DVec2::new(x0, y0), DVec2::new(x1, y1));
        let span = (hi - lo).max(DVec2::splat(1e-9));
        let size = p.size();
        let to = |q: DVec2| Vec2::new(p.min.x + ((q.x - lo.x) / span.x) as f32 * size.x, p.max.y - ((q.y - lo.y) / span.y) as f32 * size.y);

        // the scale: faint lines at round numbers, under the dots, and the numbers beside
        // them (not while the axes glide: the numbers would slide past and change)
        let still = self.range.rests_at(self.aim);
        for (axis, along) in [(0usize, size.x), (1, size.y)] {
            let (a, b) = if axis == 0 { (lo.x, hi.x) } else { (lo.y, hi.y) };
            // (a number every 90 pixels or more, but at least three of them)
            let mut step = nice_step(((b - a) as f32 * 90.0 / along.max(1.0)).max(1e-6));
            if ((b - a) as f32) < 2.5 * step {
                step = nice_below(((b - a) as f32 / 2.5).max(1e-6));
            }
            let first = (a as f32 / step).ceil() as i32;
            for k in first..first + 40 {
                let v = k as f32 * step;
                if v as f64 > b {
                    break;
                }
                let at = if axis == 0 { to(DVec2::new(v as f64, lo.y)).x } else { to(DVec2::new(lo.x, v as f64)).y };
                let faint = pal::FAINT.a(if k == 0 { 0.55 } else { 0.25 });
                if axis == 0 {
                    hud_rect_at(f, 1, Vec2::new(at - 0.5, p.min.y), Vec2::new(at + 0.5, p.max.y), faint);
                    if still {
                        f.text_px(Vec2::new(at, p.max.y + 9.0), fmt_tick(v, step), lay.small, pal::DIM, Align2::CENTER_TOP);
                    }
                } else {
                    hud_rect_at(f, 1, Vec2::new(p.min.x, at - 0.5), Vec2::new(p.max.x, at + 0.5), faint);
                    if still {
                        f.text_px(Vec2::new(p.min.x - 10.0, at), fmt_tick(v, step), lay.small, pal::DIM, Align2::RIGHT_CENTER);
                    }
                }
            }
        }

        // the units (every one of them, also those faded out or off the edge, which are
        // drawn at the edge and not seen: so the dots drawn stay the same ones while
        // anything glides)
        let count = (units.iter().map(|u| u.w).sum::<f64>() as f32).max(1.0);
        let dot = ((size.x * size.y / count).sqrt() * 0.45).clamp(8.0, 15.0);
        for u in units {
            let at = to(u.p);
            let inside = p.holds(at) && at.is_finite();
            let at = if at.is_finite() { at.max(p.min).min(p.max) } else { p.min };
            f.dot(f.hud_at(2, at), dot, level_color(u.level).a(if inside { u.w as f32 } else { 0.0 }));
        }

        // the lines, each over a cleared band so that it stands out from the dots. Each
        // is always drawn, as nothing where it has no place (and then where it was last,
        // so that it never seems to leap away)
        let mut line_px = |f: &mut Frame, k: usize, ends: Option<(DVec2, DVec2)>, width: f32, color: Color| {
            let seen = ends.and_then(|(a, b)| clip(a, b, lo, hi)).map(|(a, b)| (to(a), to(b))).filter(|(a, b)| a.is_finite() && b.is_finite());
            let (a, b) = seen.unwrap_or(self.kept[k]);
            self.kept[k] = (a, b);
            let alpha = if seen.is_some() { color.0[3] as f32 / 255.0 } else { 0.0 };
            hud_line(f, a, b, width + 5.0, UNDER.a(0.95 * alpha));
            hud_line(f, a, b, width, color.a(alpha));
        };
        let held = self.held.shown[0].clamp(0.0, 1.0) as f32;
        for (k, g) in fits.groups.iter().enumerate() {
            let ends = g.mean().zip(g.slope()).map(|(mean, slope)| {
                let reach = 2.0 * g.var_x().sqrt();
                (DVec2::new(mean.x - reach, mean.y - slope * reach), DVec2::new(mean.x + reach, mean.y + slope * reach))
            });
            // (a group fades in as its units do: a line needs FEWEST of them)
            let there = ((g.w - (FEWEST - 1) as f64).clamp(0.0, 1.0)) as f32;
            line_px(f, k, ends.filter(|_| held * there > 0.0), 2.8, level_color(k).a(held * there));
        }
        let m = Model::of(self.model.shown);
        let mean = fits.all.mean();
        let far = |slope: f64| mean.map(|c| (DVec2::new(lo.x, c.y + slope * (lo.x - c.x)), DVec2::new(hi.x, c.y + slope * (hi.x - c.x))));
        let (truth, naive) = (far(m.b), fits.all.slope().and_then(far));
        line_px(f, 5, truth, 3.6, TRUTH);
        line_px(f, 6, naive, 2.2, NAIVE);
        let mut names: Vec<(&'static str, &str, DVec2, Color)> = Vec::new();
        if let Some((_, end)) = truth {
            names.push(("truth", "true effect", end, TRUTH));
        }
        if let Some((_, end)) = naive {
            names.push(("naive", "all units", end, NAIVE));
        }
        // their names, right of the scatter where each meets its right-hand edge (or
        // level with where it leaves), set apart if they would touch
        let mut placed: Vec<(&'static str, &str, f32, f32, Color)> = names.iter().map(|(tag, text, end, color)| (*tag, *text, to(*end).y.clamp(p.min.y, p.max.y), 0.0, *color)).collect();
        if let [first, second] = placed.as_mut_slice() {
            let apart = lay.size + 4.0;
            let gap = (second.2 - first.2).abs();
            if gap < apart {
                let push = (apart - gap) * 0.5;
                let (up, down) = if first.2 <= second.2 { (first, second) } else { (second, first) };
                (up.3, down.3) = (-push, push);
            }
        }
        for (tag, text, y, shift, color) in placed {
            let from = Vec2::new(p.max.x + 10.0, y);
            let w = f.text_size(text, lay.small).x;
            f.text_placed(tag, 0, from, Vec2::new(w * 0.5, shift), text, lay.small, color);
        }

        // what the axes are, and what a dot is
        let x_name = story.map_or("$X$".to_string(), |s| format!("$X$: {}", s.x));
        let x_at = Vec2::new(p.mid().x, p.max.y + lay.small + 17.0);
        f.text_px(x_at, x_name.as_str(), lay.size, CAUSE, Align2::CENTER_TOP);
        if let Some(s) = story {
            let unit = format!("each dot: one {}", s.unit);
            let (w, room) = (f.text_size(&unit, lay.small).x, p.max.x - (x_at.x + f.text_size(&x_name, lay.size).x * 0.5) - 16.0);
            if w <= room {
                f.text_px(Vec2::new(p.max.x, x_at.y + 1.0), unit, lay.small, pal::DIM, Align2::RIGHT_TOP);
            }
        }
        let y_name = story.map_or("$Y$".to_string(), |s| format!("$Y$: {}", s.y));
        f.text_px(Vec2::new(p.min.x - 32.0, lay.head[1]), y_name, lay.size, EFFECT, Align2::LEFT_CENTER);
        // and Z's colours: at the right above the scatter, or on a line of their own
        let z_name = story.map_or("$Z$:".to_string(), |s| format!("$Z$, {}:", s.z));
        let (low, high) = story.map_or(("$-2$", "$2$"), |s| (s.low, s.high));
        let box_side = lay.small - 1.0;
        let width = f.text_size(&z_name, lay.small).x + 8.0 + f.text_size(low, lay.small).x + 5.0 + 5.0 * (box_side + 3.0) + 2.0 + f.text_size(high, lay.small).x;
        let (mut x, y) = if lay.narrow { (p.min.x - 32.0, lay.head[0]) } else { (p.max.x - width, lay.head[0]) };
        f.text_px(Vec2::new(x, y), z_name.as_str(), lay.small, THIRD, Align2::LEFT_CENTER);
        x += f.text_size(&z_name, lay.small).x + 8.0;
        f.text_px(Vec2::new(x, y), low, lay.small, pal::DIM, Align2::LEFT_CENTER);
        x += f.text_size(low, lay.small).x + 5.0;
        for k in 0..5 {
            hud_rect(f, Vec2::new(x, y - box_side * 0.5), Vec2::new(x + box_side, y + box_side * 0.5), level_color(k));
            x += box_side + 3.0;
        }
        f.text_px(Vec2::new(x + 2.0, y), high, lay.small, pal::DIM, Align2::LEFT_CENTER);
    }

    /// The numbers: the slopes, the true effect, and what the confounder adds.
    fn draw_numbers(&self, f: &mut Frame, lay: &Layout, s: &Stats) {
        let Some(area) = lay.numbers else { return };
        let (size, m) = (lay.size, self.goal());
        let row = size + 8.0;
        let mut y = area.min.y + size * 0.5;
        let x = area.min.x + 26.0;
        let swatch = |f: &mut Frame, y: f32, colors: &[Color]| {
            let w = 18.0 / colors.len() as f32;
            for (i, c) in colors.iter().enumerate() {
                hud_rect(f, Vec2::new(area.min.x + i as f32 * w, y - 1.5), Vec2::new(area.min.x + (i + 1) as f32 * w, y + 1.5), *c);
            }
        };
        let slope = |v: f64, se: f64| format!(r"${} \pm {}$", num(v, 2), num(se, 2));
        swatch(f, y, &[NAIVE]);
        let all = s.all.map_or("all units: no line, $X$ does not vary".to_string(), |(v, _, se)| format!("all units: slope {}", slope(v, se)));
        f.text_px(Vec2::new(x, y), all, size, pal::TEXT, Align2::LEFT_CENTER);
        y += row;
        swatch(f, y, &[TRUTH]);
        f.text_px(Vec2::new(x, y), format!("true effect: $b = {}$", num(m.b, 2)), size, pal::TEXT, Align2::LEFT_CENTER);
        y += row;
        swatch(f, y, &[0, 1, 2, 3, 4].map(level_color));
        let held = match (self.hold, s.held) {
            (false, _) => ("holding $Z$ fixed: off".to_string(), pal::DIM),
            (true, Some((v, se))) => (format!("holding $Z$ fixed: slope {}", slope(v, se)), pal::TEXT),
            (true, None) => ("holding $Z$ fixed: nothing to compare".to_string(), pal::TEXT),
        };
        f.text_px(Vec2::new(x, y), held.0, size, held.1, Align2::LEFT_CENTER);
        y += row + 4.0;
        // the formula, with the numbers in it
        let width = area.size().x;
        if let Some(limit) = m.naive_limit() {
            f.text_px(Vec2::new(area.min.x, y), "with many units, the pink slope is", lay.small, pal::DIM, Align2::LEFT_CENTER);
            y += lay.small + 8.0;
            let (big, flat) = (
                format!(
                    r"$b + \dfrac{{c\,a \operatorname{{Var}}(Z)}}{{\operatorname{{Var}}(X)}} = {} + \dfrac{{{} \cdot {} \cdot 2}}{{{}}} = {}$",
                    num(m.b, 2),
                    factor(m.c),
                    factor(m.a),
                    num(m.var_x(), 2),
                    num(limit, 2)
                ),
                format!(r"$b + c\,a \operatorname{{Var}}(Z)/\operatorname{{Var}}(X) = {}$", num(limit, 2)),
            );
            let text = if f.text_size(&big, size).x <= width { big } else { flat };
            let h = f.text_size(&text, size).y;
            f.text_px(Vec2::new(area.min.x, y + h * 0.5 - size * 0.5), text, size, pal::TEXT, Align2::LEFT_CENTER);
            y += h.max(size) + 6.0;
        }
        // what it comes to, in as many whole sentences as there is room for
        for sentence in verdict(&m, s) {
            let lines = wrap(f, &sentence, lay.small, width);
            if y + (lines.len() as f32 - 0.5) * (lay.small + 5.0) > area.max.y {
                break;
            }
            for text in lines {
                f.text_px(Vec2::new(area.min.x, y), text, lay.small, pal::DIM, Align2::LEFT_CENTER);
                y += lay.small + 5.0;
            }
        }
    }
}

/// What the numbers come to, in words: the sentences, the most important first.
fn verdict(m: &Model, s: &Stats) -> Vec<String> {
    let Some(added) = m.added() else {
        return vec!["$X$ never changes here, so nothing can show what changing it would do.".into()];
    };
    let mut said = vec![if m.confounded() {
        let limit = m.b + added;
        let what = if m.b == 0.0 {
            "There is no effect at all: the trend is all the confounder's."
        } else if limit.abs() < 0.25 * m.b.abs() {
            "It all but hides the effect."
        } else if limit * m.b < 0.0 {
            "It turns the trend round: Simpson's paradox."
        } else if limit.abs() > m.b.abs() {
            "The effect looks larger than it is."
        } else {
            "The effect looks smaller than it is."
        };
        format!(r"$Z$ adds ${}$ to the slope. {what}", num(added, 2))
    } else {
        let why = match (m.a == 0.0, m.c == 0.0) {
            (true, true) => r"$Z$ moves neither $X$ nor $Y$",
            (true, false) => r"$Z$ does not move $X$ ($a = 0$)",
            _ => r"$Z$ does not move $Y$ ($c = 0$)",
        };
        format!(r"{why}: nothing confounds, and the pink slope is $b$ but for sampling noise.")
    }];
    if m.sx == 0.0 && m.a != 0.0 {
        said.push(r"$X$ is fixed by $Z$: units with the same $Z$ all have the same $X$, so holding $Z$ fixed leaves nothing to compare.".into());
    }
    if s.n < 30 {
        said.push(r"With so few units the slopes wander from sample to sample, by about the number after $\pm$.".into());
    }
    said
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scene with a thing to try set up and its picture settled.
    fn scene(k: usize) -> Confound {
        let mut s = Confound::create();
        s.set_up(k);
        s.settle();
        s
    }

    fn two(v: f64) -> String {
        num(v, 2)
    }

    /// The settings of the things to try and of the opening, and a few from the far
    /// corners of the sliders.
    fn models() -> Vec<Model> {
        let mut all: Vec<Model> = TRIALS.iter().map(|t| t.2).chain([OPENING]).map(|s| Model { a: s.a as f64, b: s.b as f64, c: s.c as f64, sx: s.sx as f64, sy: s.sy as f64 }).collect();
        all.push(Model { a: -1.5, b: 0.3, c: 2.0, sx: 0.2, sy: 1.0 });
        all.push(Model { a: 2.0, b: -1.0, c: -0.5, sx: 1.5, sy: 0.1 });
        all.push(Model { a: 0.4, b: 2.0, c: -2.0, sx: 1.0, sy: 1.5 });
        all
    }

    /// A far larger sample than the scene draws, from its model: X, Y, Z, and which
    /// value of Z.
    fn big(m: &Model, n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
        let mut rng = Rng::new(seed);
        let draws: Vec<Draw> = (0..n).map(|_| Draw { level: rng.below(5).min(4), u: rng.normal() as f64, v: rng.normal() as f64 }).collect();
        let points: Vec<DVec2> = draws.iter().map(|d| m.point(d)).collect();
        (points.iter().map(|p| p.x).collect(), points.iter().map(|p| p.y).collect(), draws.iter().map(|d| d.z()).collect(), draws.iter().map(|d| d.level).collect())
    }

    /// The formula on the screen: with many units, the line through all of them has
    /// the slope b + c a Var(Z) / Var(X), where Var(Z) = 2 and Var(X) = a² Var(Z) + sx².
    #[test]
    fn the_line_through_all_units_tends_to_the_effect_plus_what_z_adds() {
        for m in models() {
            let (x, y, z, _) = big(&m, 400_000, 7);
            assert!((variance(&z) - VAR_Z).abs() < 0.01, "{m:?}: Var(Z) {}", variance(&z));
            assert!((variance(&x) / m.var_x() - 1.0).abs() < 0.01, "{m:?}: Var(X) {} against {}", variance(&x), m.var_x());
            let formula = m.b + m.c * m.a * VAR_Z / (m.a * m.a * VAR_Z + m.sx * m.sx);
            let limit = m.naive_limit().unwrap();
            assert!((limit - formula).abs() < 1e-12);
            // (within three of its standard errors, which at this size are 0.005 at most)
            let (slope, intercept) = line(&x, &y);
            let mx = x.iter().sum::<f64>() / x.len() as f64;
            let centred: Vec<f64> = x.iter().map(|v| v - mx).collect();
            let r: Vec<f64> = x.iter().zip(&y).map(|(x, y)| y - slope * x - intercept).collect();
            let se = standard_error(&centred, &r, 2).unwrap();
            assert!(se < 0.005 && (slope - limit).abs() < 3.0 * se, "{m:?}: the line through 400 000 units has slope {slope} ± {se}, the formula says {limit}");
        }
    }

    /// Among units with the same Z the trend of Y against X is b, and fitting Y on X
    /// and Z together finds it too.
    #[test]
    fn holding_z_fixed_finds_the_effect() {
        for m in models() {
            let (x, y, z, level) = big(&m, 400_000, 8);
            let w = regress(&[&x, &z], &y).unwrap();
            assert!((w[0] - m.b).abs() < 0.006, "{m:?}: fitted on X and Z, {}", w[0]);
            for k in 0..5 {
                let (xs, ys): (Vec<f64>, Vec<f64>) = (0..x.len()).filter(|&i| level[i] == k).map(|i| (x[i], y[i])).unzip();
                let slope = line(&xs, &ys).0;
                // (within three and a half standard errors: X varies only by sx within a group)
                let room = 3.5 * m.sy / (m.sx * (xs.len() as f64).sqrt());
                assert!((slope - m.b).abs() < room.max(0.01), "{m:?}: within Z = {}, {slope}", LEVELS[k]);
            }
        }
    }

    /// The captions of the things to try, held to account on the samples they set up.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let all = |s: &Stats| s.all.unwrap();
        let held = |s: &Stats| s.held.unwrap();
        let slopes = |s: &Stats| s.groups.map(|g| g.unwrap().0);

        // a trend with no cause: no arrow from X to Y, the line through all days rises
        // (0.88, near what temperature adds, 0.85), the lines within each temperature
        // are flat but for sampling noise, and holding temperature fixed gives 0.00 ± 0.03
        let mut s = scene(0);
        let (m, st) = (s.goal(), s.stats());
        assert!(m.b == 0.0 && m.confounded() && STORIES[s.story.unwrap()].x == "ice cream sold");
        assert_eq!((two(all(&st).0), two(m.added().unwrap())), ("0.88".into(), "0.85".into()));
        assert_eq!((two(held(&st).0), two(held(&st).1)), ("0.00".into(), "0.03".into()));
        // (each within twice its own standard error of 0: about 0.4 / (√80 × 0.6) = 0.075)
        assert!(slopes(&st).iter().all(|v| v.abs() < 0.15), "{:?}", slopes(&st));
        assert!(verdict(&m, &st).concat().contains("no effect at all"));

        // an effect hidden: sunscreen protects, the line through everyone is flat, and
        // within each group more sunscreen goes with less burn
        let mut s = scene(1);
        let (m, st) = (s.goal(), s.stats());
        assert!((m.b + 0.85).abs() < 1e-6 && m.naive_limit().unwrap().abs() < 0.005);
        assert_eq!(two(all(&st).0), "−0.02");
        assert!(all(&st).0.abs() < 1.5 * all(&st).2, "{:?}", st.all);
        assert_eq!((two(held(&st).0), two(held(&st).1)), ("−0.85".into(), "0.03".into()));
        assert!(slopes(&st).iter().all(|v| *v < -0.7), "{:?}", slopes(&st));
        assert!(verdict(&m, &st).concat().contains("hides the effect"));

        // Simpson's reversal: the whole goes down, every grade of illness goes up
        let mut s = scene(2);
        let (m, st) = (s.goal(), s.stats());
        assert!((m.b - 0.6).abs() < 1e-6 && m.a > 0.0 && m.c < 0.0 && m.naive_limit().unwrap() < 0.0);
        assert_eq!(two(all(&st).0), "−1.07");
        assert_eq!((two(held(&st).0), two(held(&st).1)), ("0.55".into(), "0.03".into()));
        assert!(slopes(&st).iter().all(|v| *v > 0.4), "{:?}", slopes(&st));
        assert!(verdict(&m, &st).concat().contains("Simpson"));

        // no confounder: the true slope but for sampling noise, and more precisely
        // holding illness fixed
        let mut s = scene(3);
        let (m, st) = (s.goal(), s.stats());
        assert!(m.a == 0.0 && (m.b - 0.6).abs() < 1e-6 && m.c == -2.0 && !m.confounded() && m.naive_limit() == Some(m.b));
        assert_eq!((two(all(&st).0), two(all(&st).2)), ("0.53".into(), "0.08".into()));
        assert!((all(&st).0 - m.b).abs() < all(&st).2);
        assert_eq!((two(held(&st).0), two(held(&st).1)), ("0.58".into(), "0.01".into()));
        assert!(held(&st).1 < all(&st).2 / 5.0);

        // and as the scene opens, the confounder makes the effect look larger
        let mut s = Confound::create();
        assert!(verdict(&s.goal(), &s.stats()).concat().contains("larger") && s.goal().naive_limit().unwrap() > 2.0 * s.goal().b);

        // chance alone: no arrows, yet these ten units go together
        let mut s = scene(4);
        let (m, st) = (s.goal(), s.stats());
        assert!(m.a == 0.0 && m.b == 0.0 && m.c == 0.0 && st.n == 10 && !s.hold);
        assert_eq!((two(st.correlation), two(all(&st).0)), ("0.65".into(), "0.66".into()));
        assert!(verdict(&m, &st).concat().contains("so few units"));
    }

    /// "About two samples in five slope by 0.3 or more, up or down. With a thousand units
    /// the slope stays near 0."
    #[test]
    fn chance_makes_trends_that_come_and_go() {
        let m = Model { a: 0.0, b: 0.0, c: 0.0, sx: 1.0, sy: 1.0 };
        let slopes: Vec<f64> = (1..=600).map(|seed| stats(&m, &sample(seed), 10).all.unwrap().0).collect();
        let share = |keep: &dyn Fn(f64) -> bool| slopes.iter().filter(|v| keep(**v)).count() as f64 / slopes.len() as f64;
        let (steep, up, down) = (share(&|v| v.abs() >= 0.3), share(&|v| v >= 0.3), share(&|v| v <= -0.3));
        assert!((0.36..0.46).contains(&steep) && up > 0.17 && down > 0.17, "{steep} of the samples slope by 0.3 or more ({up} up, {down} down)");
        let most = (1..=100).map(|seed| stats(&m, &sample(seed), MOST).all.unwrap().0.abs()).fold(0.0, f64::max);
        assert!(most < 0.1, "with a thousand units a slope of {most}");
    }

    /// The number after ± is how far the slope moves from one sample to the next: the
    /// spread of the slopes of many samples, against the standard errors they report.
    #[test]
    fn the_number_after_plus_minus_is_the_spread_from_sample_to_sample() {
        for setup in [OPENING, TRIALS[0].2, TRIALS[2].2, TRIALS[3].2] {
            let m = Model { a: setup.a as f64, b: setup.b as f64, c: setup.c as f64, sx: setup.sx as f64, sy: setup.sy as f64 };
            let runs: Vec<Stats> = (100..400).map(|seed| stats(&m, &sample(seed), setup.n)).collect();
            let spread = |v: &[f64]| variance(v).sqrt();
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let (all, all_se): (Vec<f64>, Vec<f64>) = runs.iter().map(|r| (r.all.unwrap().0, r.all.unwrap().2)).unzip();
            let (held, held_se): (Vec<f64>, Vec<f64>) = runs.iter().map(|r| r.held.unwrap()).unzip();
            for (name, v, se) in [("all units", &all, &all_se), ("Z held", &held, &held_se)] {
                let ratio = mean(se) / spread(v);
                assert!((0.88..1.12).contains(&ratio), "{m:?}, {name}: the ± says {}, the slopes spread by {}", mean(se), spread(v));
            }
            // and the slopes centre on what the formula says, and on b
            assert!((mean(&all) - m.naive_limit().unwrap()).abs() < 0.3 * spread(&all), "{m:?}: {}", mean(&all));
            assert!((mean(&held) - m.b).abs() < 0.3 * spread(&held), "{m:?}: {}", mean(&held));
        }
    }

    /// No spread in X, X fixed by Z, too few units, an arrow at 0: nothing that is not
    /// a number, and the words say what has happened.
    #[test]
    fn degenerate_settings_say_what_has_happened() {
        // X fixed by Z: a line through all units, none within a Z, nothing held
        let mut s = Confound::create();
        s.sx = 0.0;
        let (m, st) = (s.goal(), s.stats());
        assert!(st.all.is_some() && st.held.is_none() && st.groups.iter().all(Option::is_none), "{st:?}");
        assert!(verdict(&m, &st).concat().contains("fixed by"));
        // X never changes
        s.a = 0.0;
        let (m, st) = (s.goal(), s.stats());
        assert!(st.all.is_none() && st.held.is_none() && m.naive_limit().is_none() && st.correlation == 0.0);
        assert!(verdict(&m, &st).concat().contains("never changes"));
        // no confounding, said either way
        let mut s = Confound::create();
        s.c = 0.0;
        assert!(verdict(&s.goal(), &s.stats()).concat().contains("does not move $Y$"));
        s.a = 0.0;
        assert!(verdict(&s.goal(), &s.stats()).concat().contains("neither"));
        // five units: groups too small for a line are left without one, and the slopes
        // are said to wander
        let mut s = Confound::create();
        s.n = 5;
        let st = s.stats();
        assert!(st.counts.iter().sum::<usize>() == 5 && st.groups.iter().zip(st.counts).all(|(g, n)| g.is_some() == (n >= FEWEST)));
        assert!(verdict(&s.goal(), &st).concat().contains("so few units"));
        // whatever the sliders say, every number is finite or not there at all
        for a in [-2.0, 0.0, 0.01, 2.0] {
            for sx in [0.0, 0.01, 1.5] {
                for n in [5, 6, 30, MOST] {
                    let mut s = Confound::create();
                    (s.a, s.b, s.c, s.sx, s.sy, s.n) = (a, -2.0, 2.0, sx, 0.0, n);
                    let st = s.stats();
                    let fine = |v: f64| v.is_finite();
                    assert!(st.all.is_none_or(|(v, k, se)| fine(v) && fine(k) && fine(se)) && st.held.is_none_or(|(v, se)| fine(v) && fine(se)), "{a} {sx} {n}: {st:?}");
                    assert!(st.groups.iter().flatten().all(|(v, k)| fine(*v) && fine(*k)) && fine(st.correlation));
                    s.advance(1.0 / 60.0);
                    s.frame_on(1.0 / 60.0);
                    assert!(s.units().iter().all(|u| u.p.is_finite() && u.w.is_finite()) && s.range.shown.iter().all(|v| v.is_finite()));
                }
            }
        }
    }

    /// The lines the picture draws (fitted with weights, so that units can fade in and
    /// out) are the lines the numbers come from, once the picture has come to rest.
    #[test]
    fn the_lines_drawn_are_the_lines_fitted() {
        for k in 0..TRIALS.len() {
            let mut s = scene(k);
            let (units, st) = (s.units(), s.stats());
            let fits = Fits::of(&units);
            assert!((fits.all.w - s.n as f64).abs() < 1e-9);
            assert!((fits.all.slope().unwrap() - st.all.unwrap().0).abs() < 1e-9);
            // (a group of fewer than FEWEST may have a line in the picture, but it is not seen)
            for (g, want) in fits.groups.iter().zip(st.groups) {
                match want {
                    Some((slope, _)) => assert!((g.slope().unwrap() - slope).abs() < 1e-9),
                    None => assert!(g.w < FEWEST as f64 || g.slope().is_none()),
                }
            }
        }
    }

    /// After a change the picture follows, and comes to rest exactly where it was
    /// asked to in a second or so, at any rate of frames; with the scene's time
    /// standing still it is there at once.
    #[test]
    fn the_picture_follows_a_change_and_comes_to_rest() {
        for hz in [30.0, 60.0, 144.0] {
            for k in 0..TRIALS.len() {
                let mut s = Confound::create();
                s.frame_on(0.0);
                s.set_up(k);
                let mut frames = 0;
                while s.moving() {
                    s.advance(1.0 / hz);
                    s.frame_on(1.0 / hz);
                    frames += 1;
                    assert!(frames as f64 / hz < 2.0, "{} still moving after {frames} frames at {hz} Hz", TRIALS[k].0);
                }
                assert!(frames as f64 / hz > 0.3, "{}: over in {frames} frames", TRIALS[k].0);
                assert!(s.model.shown == s.goal().array() && s.old.is_none() && s.range.shown == s.aim);
            }
        }
        let mut s = Confound::create();
        s.frame_on(0.0);
        s.set_up(2);
        s.advance(0.0);
        s.frame_on(0.0);
        assert!(!s.moving());
    }

    /// Every canvas, from a desktop's to a tiny one, has a layout: every part of it
    /// finite and the right way up; and where there is room, the parts apart and on
    /// the canvas.
    #[test]
    fn every_canvas_has_a_layout() {
        for (w, h) in [(1080.0, 1000.0), (1600.0, 1000.0), (390.0, 800.0), (800.0, 390.0), (600.0, 700.0), (700.0, 900.0), (240.0, 180.0), (1.0, 1.0), (5000.0, 300.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let mut parts = vec![lay.graph, lay.plot, lay.buttons[0], lay.buttons[1]];
                parts.extend(lay.numbers);
                assert!(parts.iter().all(|a| a.min.is_finite() && a.max.is_finite() && a.min.cmple(a.max).all()), "{view}: {lay:?}");
                if w >= 390.0 && h >= 560.0 {
                    let on = |a: &Area| a.min.cmpge(Vec2::ZERO).all() && a.max.cmple(view).all();
                    let apart = |a: &Area, b: &Area| a.max.x <= b.min.x || b.max.x <= a.min.x || a.max.y <= b.min.y || b.max.y <= a.min.y;
                    assert!(parts.iter().all(on) && lay.numbers.is_some(), "{view}: {lay:?}");
                    for i in 0..parts.len() {
                        for j in 0..i {
                            assert!(apart(&parts[i], &parts[j]), "{view}: parts {i} and {j} overlap: {lay:?}");
                        }
                    }
                    assert!(lay.plot.max.x + lay.margin <= w && lay.plot.min.y > inset && lay.plot.size().min_element() > 250.0, "{view}: {lay:?}");
                }
            }
        }
    }

    /// Where the units are on a desktop's scatter, in pixels, as drawn now.
    fn on_screen(s: &Confound) -> Vec<Vec2> {
        let plot = Layout::of(Vec2::new(1080.0, 1000.0), 0.0).plot;
        let [x0, y0, x1, y1] = s.range.shown;
        let size = plot.size().as_dvec2();
        s.units().iter().map(|u| (plot.min.as_dvec2() + DVec2::new((u.p.x - x0) / (x1 - x0) * size.x, (y1 - u.p.y) / (y1 - y0) * size.y)).as_vec2()).collect()
    }

    /// A slider dragged steadily (b from 0.5 to 2 and back, the units from 400 to 1000,
    /// σ_X to 0) moves the picture steadily too: the dots go no further in one frame
    /// than a few times as far as in the frames either side, though the axes change
    /// on the way, and no more than a few pixels.
    #[test]
    fn a_slider_dragged_moves_the_picture_evenly() {
        for hz in [60.0, 120.0] {
            let mut s = Confound::create();
            s.frame_on(0.0);
            let frames = (2.0 * hz) as usize;
            let mut moves = Vec::new();
            let mut was = on_screen(&s);
            for i in 0..frames + (hz as usize) {
                let t = (i as f32 / frames as f32).min(1.0);
                s.b = 0.5 + 1.5 * (1.0 - (2.0 * t - 1.0).abs());
                s.sx = 0.6 * (1.0 - t);
                s.n = 400 + (600.0 * t) as usize;
                s.advance(1.0 / hz);
                s.frame_on(1.0 / hz);
                // (units are added at the end of the list: those there before are where they were in it)
                let now = on_screen(&s);
                moves.push(now.iter().zip(&was).map(|(a, b)| a.distance(*b)).fold(0.0f32, f32::max));
                was = now;
            }
            let worst = moves.windows(3).filter(|w| w[1] > 2.0).map(|w| w[1] / w[0].max(w[2]).max(0.05)).fold(0.0f32, f32::max);
            let most = moves.iter().copied().fold(0.0f32, f32::max);
            // (measured: 1.01 and 6.5 pixels at 60 Hz, 1.00 and 3.4 at 120)
            assert!(worst < 1.3 && most < 10.0 * 60.0 / hz as f32, "at {hz} Hz a frame moved {worst} times as far as its neighbours, and one {most} pixels");
        }
    }

    #[test]
    fn nice_numbers_below() {
        for (x, want) in [(2.84f32, 2.0f32), (2.0, 2.0), (1.9, 1.0), (7.0, 5.0), (0.3, 0.2), (12.0, 10.0), (0.049, 0.02)] {
            assert!((nice_below(x) - want).abs() < 1e-5 * want, "{x}: {}", nice_below(x));
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for (k, (name, _, setup)) in TRIALS.iter().enumerate() {
            let mut s = scene(k);
            let st = s.stats();
            let m = s.goal();
            println!("{name}: seed {} limit {:?} added {:?}", setup.seed, m.naive_limit(), m.added());
            println!("  all {:?}\n  held {:?}\n  groups {:?}\n  counts {:?} r {:.3}", st.all, st.held, st.groups, st.counts, st.correlation);
        }
        for m in models() {
            let (x, y, z, level) = big(&m, 400_000, 7);
            let (x8, y8, z8, l8) = big(&m, 400_000, 8);
            let w = regress(&[&x8, &z8], &y8).unwrap();
            let worst = (0..5).map(|k| {
                let (xs, ys): (Vec<f64>, Vec<f64>) = (0..x8.len()).filter(|&i| l8[i] == k).map(|i| (x8[i], y8[i])).unzip();
                let room = m.sy / (m.sx * (xs.len() as f64).sqrt());
                (line(&xs, &ys).0 - m.b) / room
            }).fold(0.0f64, |a, v| a.max(v.abs()));
            let _ = (z, level);
            println!("{m:?}: line {:+.4} held {:+.4} groups worst {worst:.2} se", line(&x, &y).0 - m.naive_limit().unwrap(), w[0] - m.b);
        }
        for setup in [OPENING, TRIALS[0].2, TRIALS[2].2, TRIALS[3].2] {
            let m = Model { a: setup.a as f64, b: setup.b as f64, c: setup.c as f64, sx: setup.sx as f64, sy: setup.sy as f64 };
            let runs: Vec<Stats> = (100..400).map(|seed| stats(&m, &sample(seed), setup.n)).collect();
            let spread = |v: &[f64]| variance(v).sqrt();
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let (all, all_se): (Vec<f64>, Vec<f64>) = runs.iter().map(|r| (r.all.unwrap().0, r.all.unwrap().2)).unzip();
            let (held, held_se): (Vec<f64>, Vec<f64>) = runs.iter().map(|r| r.held.unwrap()).unzip();
            println!("{m:?}: all ratio {:.3} held ratio {:.3}; centre {:.3} {:.3}", mean(&all_se) / spread(&all), mean(&held_se) / spread(&held), (mean(&all) - m.naive_limit().unwrap()) / spread(&all), (mean(&held) - m.b) / spread(&held));
        }
    }
}

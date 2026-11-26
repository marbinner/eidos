//! Two flows at once, or one after the other. Read Ax and Bx as velocities: flowing
//! with both at once for a time t is the matrix e^(t(A+B)); flowing by B for that time
//! and then by A is e^(tA) e^(tB). The two agree when AB = BA. When they do not, n short
//! rounds of each in turn close the gap like 1/n, and the symmetric round (half of A,
//! B, half of A) like 1/n².

use super::*;
use crate::scenes::stats::plot::{self, Plot};
use glam::{DMat2, DVec2};

pub const DEF: SceneDef = SceneDef {
    id: "expsum",
    title: "Two flows at once, or one after the other",
    group: EXPONENTIAL,
    blurb: "The exponential of a sum against the product of the exponentials: equal when the matrices commute, and what small steps do when they do not.",
    make: |_| Box::new(Expsum::new()),
};

// ----- the arithmetic, in double precision -------------------------------------------

/// The longest time the flows are followed for.
const T_MAX: f64 = 2.0;
/// The most rounds.
const N_MAX: usize = 64;
/// The largest entry of A or B either way. No eigenvalue of A + B is then beyond 12 in
/// size, so nothing drawn or written is beyond e^(12 × 2), about 3 × 10¹⁰, times the start.
const ENTRY_MAX: f64 = 3.0;
/// How far out the start x₀ may be dragged.
const X0_MAX: f64 = 20.0;

/// A matrix by its rows, [[a, b], [c, d]], the way matrices are written.
fn rows([a, b, c, d]: [f64; 4]) -> DMat2 {
    DMat2::from_cols(DVec2::new(a, c), DVec2::new(b, d))
}

fn trace(m: DMat2) -> f64 {
    m.x_axis.x + m.y_axis.y
}

/// (tr A / 2)² − det A, worked out as ((a − d)/2)² + bc, so that a matrix with a double
/// eigenvalue gives 0 exactly.
fn quarter_disc(m: DMat2) -> f64 {
    let half = 0.5 * (m.x_axis.x - m.y_axis.y);
    half * half + m.y_axis.x * m.x_axis.y
}

/// e^(At), in closed form (as in the scene "expm"). A = sI + B with s = tr A / 2, and B
/// squares to qI (q = `quarter_disc`), so e^(At) = e^(st) (c I + g B) with c = cosh(√q t)
/// and g = sinh(√q t) / √q (cos and sin for q < 0; 1 and t for q = 0). Near q t² = 0 the
/// three are one power series, summed as one: nothing is divided by a number near zero.
fn expm(a: DMat2, t: f64) -> DMat2 {
    let s = 0.5 * trace(a);
    let b = a - DMat2::IDENTITY * s;
    let q = quarter_disc(a);
    let z = q * t * t;
    let (c, g) = if z.abs() < 1.0 {
        // c = Σ zᵏ/(2k)!, g = t Σ zᵏ/(2k + 1)!: twelve terms are exact to the last digit
        let (mut c, mut g, mut tc, mut tg) = (1.0, 1.0, 1.0, 1.0);
        for k in 1..=12 {
            let k = k as f64;
            tc *= z / ((2.0 * k - 1.0) * (2.0 * k));
            tg *= z / ((2.0 * k) * (2.0 * k + 1.0));
            c += tc;
            g += tg;
        }
        (c, g * t)
    } else if q > 0.0 {
        let r = q.sqrt();
        ((r * t).cosh(), (r * t).sinh() / r)
    } else {
        let r = (-q).sqrt();
        ((r * t).cos(), (r * t).sin() / r)
    };
    (DMat2::IDENTITY * c + b * g) * (s * t).exp()
}

/// The size of a matrix: the square root of the sum of the squares of its entries (its
/// length as a list of four numbers).
fn size(m: DMat2) -> f64 {
    m.to_cols_array().iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// AB − BA.
fn bracket(a: DMat2, b: DMat2) -> DMat2 {
    a * b - b * a
}

/// What the symmetric round gets wrong first, to the third power of its length h:
/// e^(hA/2) e^(hB) e^(hA/2) = e^(h(A+B) + h³ C + …) with C = [B, [B, A]]/12 − [A, [A, B]]/24.
fn third(a: DMat2, b: DMat2) -> DMat2 {
    bracket(b, bracket(b, a)) * (1.0 / 12.0) - bracket(a, bracket(a, b)) * (1.0 / 24.0)
}

/// One round of length h as one matrix: B for h and then A for h (the matrix on the
/// right acts first), or symmetric: A for h/2, B for h, A for h/2.
fn round(a: DMat2, b: DMat2, h: f64, sym: bool) -> DMat2 {
    if sym {
        let half = expm(a, 0.5 * h);
        half * expm(b, h) * half
    } else {
        expm(a, h) * expm(b, h)
    }
}

/// n rounds of length t/n: (e^(tA/n) e^(tB/n))ⁿ, or the symmetric one's.
fn rounds(a: DMat2, b: DMat2, t: f64, n: usize, sym: bool) -> DMat2 {
    let n = n.max(1);
    let r = round(a, b, t / n as f64, sym);
    let mut m = DMat2::IDENTITY;
    for _ in 0..n {
        m = r * m;
    }
    m
}

/// How far apart the n rounds and both at once are, as matrices.
fn gap(a: DMat2, b: DMat2, t: f64, n: usize, sym: bool) -> f64 {
    size(rounds(a, b, t, n, sym) - expm(a + b, t))
}

/// Whether a gap is nothing but rounding: a millionth of a millionth of the size of
/// what it is the gap between.
fn no_gap(gap: f64, scale: f64) -> bool {
    // (and a gap that is not a number is no gap to draw)
    gap.is_nan() || gap <= 1e-12 * scale.max(1.0)
}

/// The pieces of the zig-zag in the order they are flowed: whether each is the outer
/// matrix's (A's, in the usual order), and for how long. Plain: inner, outer, inner,
/// outer, …; symmetric: half of outer, inner, outer, …, inner, half of outer (two
/// halves in a row are one whole piece).
fn pieces(t: f64, n: usize, sym: bool) -> Vec<(bool, f64)> {
    let n = n.max(1);
    let h = t / n as f64;
    let mut out = Vec::with_capacity(2 * n + 1);
    if sym {
        out.push((true, 0.5 * h));
        for k in 0..n {
            out.push((false, h));
            out.push((true, if k + 1 < n { h } else { 0.5 * h }));
        }
    } else {
        for _ in 0..n {
            out.push((false, h));
            out.push((true, h));
        }
    }
    out
}

/// Segments the zig-zag is drawn with, whatever n is: evenly spaced in the time spent
/// flowing (2t in all, t by each matrix). For every n up to 64 that divides 256 every
/// corner is one of their ends, in both kinds of round; for other n a corner is within
/// a 1024th of the path of one.
const U: usize = 1024;
/// Segments of the path of both at once.
const SMOOTH: usize = 256;

/// The zig-zag: its points, and how much of the outer matrix's colour each has (1)
/// against the inner one's (0).
#[derive(Clone, Debug, Default)]
struct Zig {
    points: Vec<DVec2>,
    mix: Vec<f32>,
}

/// The path of x₀ under n rounds of `inner` and `outer` in turns over the time t, at
/// U + 1 moments evenly spaced in the time spent flowing. Within a piece each point is
/// the last carried on by one small step of its flow: a few exponentials a piece.
fn zig(outer: DMat2, inner: DMat2, x0: DVec2, t: f64, n: usize, sym: bool) -> Zig {
    let pieces = pieces(t, n, sym);
    let total = 2.0 * t;
    let step = total / U as f64;
    let steps = [expm(inner, step), expm(outer, step)];
    let mut points = Vec::with_capacity(U + 1);
    let mut mix = Vec::with_capacity(U + 1);
    // the piece now flowed, when it began, where x₀ had got to then
    let (mut k, mut began, mut start) = (0usize, 0.0f64, x0);
    let mut last: Option<(usize, DVec2)> = None;
    for i in 0..=U {
        let tau = total * i as f64 / U as f64;
        // (a moment at a corner belongs to the piece it ends)
        while k + 1 < pieces.len() && tau > began + pieces[k].1 + 1e-9 * step {
            let (outer_piece, d) = pieces[k];
            start = expm(if outer_piece { outer } else { inner }, d) * start;
            began += d;
            k += 1;
        }
        let outer_piece = pieces[k].0;
        let p = match last {
            Some((was, p)) if was == k => steps[outer_piece as usize] * p,
            _ => expm(if outer_piece { outer } else { inner }, (tau - began).max(0.0)) * start,
        };
        last = Some((k, p));
        points.push(p);
        mix.push(if outer_piece { 1.0 } else { 0.0 });
    }
    Zig { points, mix }
}

/// The path of x₀ with both velocities at once, from 0 to the time t.
fn smooth(a: DMat2, b: DMat2, x0: DVec2, t: f64) -> Vec<DVec2> {
    let m = expm(a + b, t / SMOOTH as f64);
    let mut p = x0;
    let mut out = Vec::with_capacity(SMOOTH + 1);
    out.push(p);
    for _ in 0..SMOOTH {
        p = m * p;
        out.push(p);
    }
    out
}

/// Where round k (of n) ends on the grid of the zig-zag: after 2kt/n of the 2t of
/// flowing.
fn round_end(k: usize, n: usize) -> usize {
    ((k * U) as f64 / n.max(1) as f64).round() as usize
}

// ----- pairs and experiments ----------------------------------------------------------

struct Pair {
    name: &'static str,
    a: [f64; 4],
    b: [f64; 4],
    note: &'static str,
}

const PAIRS: [Pair; 5] = [
    Pair {
        name: "a stretch and a turn",
        a: [1.0, 0.0, 0.0, -1.0],
        b: [0.0, -1.0, 1.0, 0.0],
        note: r"$A$ stretches along $x$ and squeezes along $y$; $B$ turns. $AB - BA$ is not $0$: its size is 2.83. Both at once they balance: $(A + B)^2 = 0$, so $e^{t(A+B)} = I + t(A + B)$, and every point slides along a straight line.",
    },
    Pair {
        name: "an even stretch and a turn",
        a: [0.5, 0.0, 0.0, 0.5],
        b: [0.0, -1.0, 1.0, 0.0],
        note: r"$A = 0.5I$ grows every arrow by the same factor, whichever way it points, so it commutes with every matrix: $AB - BA = 0$. Both at once is a spiral, and every round ends on it.",
    },
    Pair {
        name: "two stretches, same axes",
        a: [0.8, 0.0, 0.0, -0.4],
        b: [-0.6, 0.0, 0.0, 0.7],
        note: r"Both act along the $x$ and $y$ axes, each by its own amounts: $AB = BA$, and there is no gap.",
    },
    Pair {
        name: "two shears",
        a: [0.0, 1.0, 0.0, 0.0],
        b: [0.0, 0.0, 1.0, 0.0],
        note: r"$A$ slides the plane along the $x$ axis, $B$ along the $y$ axis. $AB - BA$ stretches along $x$ and squeezes along $y$; both at once, $A + B$ stretches along the line $y = x$ and squeezes along $y = -x$.",
    },
    Pair {
        name: "two stretches, 45° apart",
        a: [1.0, 0.0, 0.0, -1.0],
        b: [0.0, 1.0, 1.0, 0.0],
        note: r"$A$ stretches along $x$, $B$ along the line $y = x$, and each squeezes across. $AB - BA$ is a turn: a quarter turn clockwise, times 2.",
    },
];

/// What a chart shows the gap against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Chart {
    /// The number of rounds n, at the time t.
    Rounds,
    /// The time t, with n rounds.
    Time,
}

/// What an experiment sets going once its setting has been glided to.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Then {
    Rest,
    /// t plays from where it is set to, up to this.
    Play(f64),
    /// n doubles, from 1 up to `COUNT_TO`.
    Count,
}

struct Trial {
    label: &'static str,
    look_for: &'static str,
    pair: usize,
    x0: [f64; 2],
    t: f64,
    n: usize,
    sym: bool,
    chart: Chart,
    then: Then,
}

/// The start every experiment uses.
const X0: [f64; 2] = [1.0, -0.6];

const TRIALS: [Trial; 5] = [
    Trial {
        label: "a stretch and a turn",
        look_for: r"$A$ stretches along $x$ and squeezes along $y$; $B$ turns. Both at once (white), they balance into a shear: $x_0$ slides along a straight line to $e^{A+B} x_0 = (2.60, 1.00)$. Turning for a unit of time and then stretching (blue, then orange) ends at $e^A e^B x_0 = (2.84, 0.19)$ instead, 0.845 away (red). The order matters too: $e^B e^A$ is another matrix again. $AB - BA$ is not $0$, and $e^A e^B \ne e^{A+B}$.",
        pair: 0,
        x0: X0,
        t: 1.0,
        n: 1,
        sym: false,
        chart: Chart::Rounds,
        then: Then::Rest,
    },
    Trial {
        label: "a pair that commutes",
        look_for: r"An even stretch, $A = 0.5I$, and a turn: $AB = BA$. Four rounds, each a turn (blue) and then an even growth (orange), for $t/4$ each: every round ends on the white spiral of both at once (the dots), and the last where it ends. No gap, for any $n$ and in either kind of round: the chart's curves lie on its floor.",
        pair: 1,
        x0: X0,
        t: 1.0,
        n: 4,
        sym: false,
        chart: Chart::Rounds,
        then: Then::Rest,
    },
    Trial {
        label: "small steps close the gap",
        look_for: r"Back to the stretch and the turn, now in $n$ rounds, each $B$ for the time $t/n$ and then $A$ for $t/n$, with $n$ doubling from 1 to 32. The zig-zag closes in on the straight white path. Each doubling about halves the gap between the matrices: 0.355, 0.177, 0.088, 0.044 for $n = 4, 8, 16, 32$, parallel to the chart's dashed line of slope $-1$. The gap falls like $1/n$.",
        pair: 0,
        x0: X0,
        t: 1.0,
        n: 1,
        sym: false,
        chart: Chart::Rounds,
        then: Then::Count,
    },
    Trial {
        label: "the symmetric split",
        look_for: r"Each round is now $A$ for $t/2n$, $B$ for $t/n$ and $A$ for $t/2n$, four times over (two halves of $A$ in a row make one piece, so there is one piece more than before). The gap is 0.042, against 0.355 for the plain rounds. Each doubling of $n$ now quarters it: 0.042, 0.0105, 0.0026, 0.00066 for $n = 4, 8, 16, 32$, parallel to the dashed line of slope $-2$: it falls like $1/n^2$. Many simulations are stepped this way when each part is easy to follow alone.",
        pair: 0,
        x0: X0,
        t: 1.0,
        n: 4,
        sym: true,
        chart: Chart::Rounds,
        then: Then::Rest,
    },
    Trial {
        label: "the gap starts as $t^2$",
        look_for: r"One round, $n = 1$, with $t$ played from 0, and the chart against $t$. For a short time the gap between the matrices is about $(t^2/2) \lVert AB - BA \rVert$: at $t = 0.1$ it is 0.01416, and that gives 0.01414. On the chart the gap runs along that dashed line, of slope 2: halve $t$ and the gap falls to a quarter. The two matrices' series agree in $I$ and in $t(A + B)$, and first differ in their terms in $t^2$, by $(t^2/2)(AB - BA)$. By $t = 1$ the later terms have grown: 1.567, against 1.414.",
        pair: 0,
        x0: X0,
        t: 0.0,
        n: 1,
        sym: false,
        chart: Chart::Time,
        then: Then::Play(1.0),
    },
];

/// Units of time a second, as t plays.
const PLAY_RATE: f64 = 0.25;
/// Seconds between one doubling of n and the next, as an experiment counts up, and the
/// n it stops at.
const COUNT_EVERY: f32 = 1.4;
const COUNT_TO: usize = 32;

// ----- what glides -------------------------------------------------------------------

/// What glides when a button changes it: the two matrices, the time and the start.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    a: DMat2,
    b: DMat2,
    t: f64,
    x0: DVec2,
}

impl Setting {
    fn numbers(&self) -> [f64; 11] {
        let (a, b) = (self.a.to_cols_array(), self.b.to_cols_array());
        [a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3], self.t, self.x0.x, self.x0.y]
    }

    fn from_numbers(v: [f64; 11]) -> Setting {
        Setting { a: DMat2::from_cols_array(&[v[0], v[1], v[2], v[3]]), b: DMat2::from_cols_array(&[v[4], v[5], v[6], v[7]]), t: v[8], x0: DVec2::new(v[9], v[10]) }
    }
}

/// How fast a change is glided to: each of the two eases one after the other covers two
/// thirds of what is left in this many seconds.
const FOLLOW: f64 = 0.09;

/// A setting as asked for, and as drawn: the drawn one follows it through `between`, two
/// eases one after the other, so that it sets off gently and arrives exactly.
#[derive(Clone, Copy, Debug)]
struct Glide {
    to: Setting,
    between: Setting,
    shown: Setting,
}

impl Glide {
    fn at(s: Setting) -> Glide {
        Glide { to: s, between: s, shown: s }
    }

    fn arrived(&self) -> bool {
        self.shown == self.to && self.between == self.to
    }

    /// Change part of the setting at once, in all three: what a hand moves.
    fn now(&mut self, change: impl Fn(&mut Setting)) {
        for s in [&mut self.to, &mut self.between, &mut self.shown] {
            change(s);
        }
    }

    fn step(&mut self, dt: f64) {
        if self.arrived() || dt <= 0.0 {
            return;
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        let (to, mut between, mut shown) = (self.to.numbers(), self.between.numbers(), self.shown.numbers());
        let mut apart = 0.0f64;
        for i in 0..11 {
            between[i] += (to[i] - between[i]) * k;
            shown[i] += (between[i] - shown[i]) * k;
            apart = apart.max((to[i] - between[i]).abs()).max((to[i] - shown[i]).abs());
        }
        // (exactly there in the end: the picture then stops changing)
        if apart < 2e-4 || !apart.is_finite() {
            *self = Glide::at(self.to);
        } else {
            (self.between, self.shown) = (Setting::from_numbers(between), Setting::from_numbers(shown));
        }
    }
}

/// A zig-zag as drawn. For the same number of rounds and the same kind of round it is
/// the zig-zag of the setting as drawn, point for point; when either changes, the drawn
/// one glides from where it was to the new one (each point of the grid to the point of
/// the same moment), through two eases, as `Glide` does. What is kept is how far each
/// point still is from where it is going, so that a hand moving x₀ meanwhile moves the
/// whole path with it.
#[derive(Default)]
struct Morph {
    /// What the target was worked out from, to the bit (the setting, n and the kind of
    /// round); and the count and kind alone.
    of: Vec<u64>,
    key: Option<(usize, bool)>,
    target: Zig,
    /// Each point's distance still to go, in the two eases (the first, then the drawn).
    off: [Vec<DVec2>; 2],
    mix_off: [Vec<f32>; 2],
    /// The furthest any point had to go when it last set off, and has now.
    start: f64,
    now: f64,
}

impl Morph {
    fn aim(&mut self, key: (usize, bool), target: Zig) {
        let len = target.points.len();
        let same_grid = self.target.points.len() == len && self.off[0].len() == len;
        if same_grid && self.key.is_some_and(|k| k != key) {
            // (where each point is drawn stays where it is: only what it is going to changes)
            for s in 0..2 {
                for i in 0..len {
                    self.off[s][i] += self.target.points[i] - target.points[i];
                    self.mix_off[s][i] += self.target.mix[i] - target.mix[i];
                }
            }
            self.now = self.furthest();
            self.start = self.now;
        } else if !same_grid {
            self.off = [vec![DVec2::ZERO; len], vec![DVec2::ZERO; len]];
            self.mix_off = [vec![0.0; len], vec![0.0; len]];
            (self.start, self.now) = (0.0, 0.0);
        }
        self.key = Some(key);
        self.target = target;
    }

    fn furthest(&self) -> f64 {
        self.off.iter().flatten().fold(0.0f64, |m, d| m.max(d.length()))
    }

    fn moving(&self) -> bool {
        self.now > 0.0
    }

    /// `dt` seconds nearer to where it is going.
    fn step(&mut self, dt: f64) {
        if !self.moving() || dt <= 0.0 {
            return;
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        let k32 = k as f32;
        let [first, drawn] = &mut self.off;
        for (b, s) in first.iter_mut().zip(drawn.iter_mut()) {
            *b *= 1.0 - k;
            *s += (*b - *s) * k;
        }
        let [first, drawn] = &mut self.mix_off;
        for (b, s) in first.iter_mut().zip(drawn.iter_mut()) {
            *b *= 1.0 - k32;
            *s += (*b - *s) * k32;
        }
        self.now = self.furthest();
        // (exactly there in the end: a ten-thousandth of the path's own size is nothing)
        let scale = self.target.points.iter().fold(1.0f64, |m, p| m.max(p.abs().max_element()));
        if self.now < 1e-4 * scale || !self.now.is_finite() {
            for s in 0..2 {
                self.off[s].iter_mut().for_each(|d| *d = DVec2::ZERO);
                self.mix_off[s].iter_mut().for_each(|d| *d = 0.0);
            }
            self.now = 0.0;
        }
    }

    fn point(&self, i: usize) -> DVec2 {
        self.target.points[i] + self.off[1].get(i).copied().unwrap_or(DVec2::ZERO)
    }

    fn mix(&self, i: usize) -> f32 {
        (self.target.mix[i] + self.mix_off[1].get(i).copied().unwrap_or(0.0)).clamp(0.0, 1.0)
    }

    /// How far along the glide is: 0 as it sets off, 1 once there.
    fn arrived_share(&self) -> f32 {
        if self.now <= 0.0 || self.start <= 0.0 {
            1.0
        } else {
            (1.0 - self.now / self.start).clamp(0.0, 1.0) as f32
        }
    }
}

// ----- the chart's numbers ------------------------------------------------------------

/// Times the chart against t is worked out at: from a hundredth to `T_MAX`, evenly
/// spread on its logarithmic axis.
const T_SAMPLES: usize = 81;
const T_LOW: f64 = 0.01;

fn sample_time(i: usize) -> f64 {
    T_LOW * (T_MAX / T_LOW).powf(i as f64 / (T_SAMPLES - 1) as f64)
}

/// The gaps a chart draws, kept with what they were worked out for, to the bit.
struct Curves {
    of: Vec<u64>,
    /// Along the chart: n = 1 to `N_MAX`, or the times `sample_time`.
    xs: Vec<f64>,
    /// The gap of the plain rounds and of the symmetric ones at each; and the size of
    /// what it is the gap between (for telling a gap from rounding).
    gaps: [Vec<f64>; 2],
    scale: Vec<f64>,
}

impl Curves {
    /// The gap of a kind of round (0 plain, 1 symmetric) as a power of ten, at a place
    /// across the chart (the power of ten of n or of t), between the points worked out;
    /// far below the chart where there is no gap.
    fn at(&self, kind: usize, x: f64) -> f64 {
        let value = |i: usize| {
            let g = self.gaps[kind][i];
            if no_gap(g, self.scale[i]) {
                -300.0
            } else {
                g.log10()
            }
        };
        let last = self.xs.len().saturating_sub(1);
        let i = self.xs.iter().rposition(|v| v.log10() <= x).unwrap_or(0).min(last.saturating_sub(1));
        if self.xs.len() < 2 {
            return if self.xs.is_empty() { -300.0 } else { value(0) };
        }
        let (x0, x1) = (self.xs[i].log10(), self.xs[i + 1].log10());
        let share = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0);
        value(i) + (value(i + 1) - value(i)) * share
    }
}

fn curves_of(a: DMat2, b: DMat2, t: f64, n: usize, chart: Chart) -> Curves {
    let mut of: Vec<u64> = a.to_cols_array().iter().chain(b.to_cols_array().iter()).map(|v| v.to_bits()).collect();
    let (mut xs, mut plain, mut sym, mut scale) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    match chart {
        Chart::Rounds => {
            of.extend([0, t.to_bits()]);
            let both = expm(a + b, t);
            for k in 1..=N_MAX {
                xs.push(k as f64);
                plain.push(size(rounds(a, b, t, k, false) - both));
                sym.push(size(rounds(a, b, t, k, true) - both));
                scale.push(size(both));
            }
        }
        Chart::Time => {
            of.extend([1, n as u64]);
            for i in 0..T_SAMPLES {
                let s = sample_time(i);
                let both = expm(a + b, s);
                xs.push(s);
                plain.push(size(rounds(a, b, s, n, false) - both));
                sym.push(size(rounds(a, b, s, n, true) - both));
                scale.push(size(both));
            }
        }
    }
    Curves { of, xs, gaps: [plain, sym], scale }
}

// ----- where things go ---------------------------------------------------------------

/// The part of the scene the view is fitted to (its centre, and half its size): the
/// paths of every experiment lie in it, and so does 0, which the turns go round.
const FRAME_CENTRE: Vec2 = Vec2::new(1.5, 0.25);
const FRAME_HALF: Vec2 = Vec2::new(1.65, 1.15);

/// The screen panel's parts, in logical pixels.
#[derive(Clone, Debug)]
struct Panel {
    min: Vec2,
    max: Vec2,
    /// The size of the numbers in the matrices, and the width of a column of them.
    text: f32,
    column: f32,
    /// Top-left of each matrix's name, two to a row: A, B; both at once, the rounds; and
    /// (with room) AB − BA, the other order.
    cells: Vec<Vec2>,
    /// The lines of numbers under the matrices, and how many there are.
    lines: Vec2,
    line_count: usize,
    /// The row of buttons for n, and the time track: its two ends at its middle, and
    /// the play button.
    n_row: Vec2,
    bar: (Vec2, Vec2),
    play: (Vec2, Vec2),
    /// The chart's buttons, and the rectangle it draws in (none without the room).
    chart_buttons: Vec2,
    chart: Option<plot::Rect>,
}

const CAP_H: f32 = 16.0;
const LINE_H: f32 = 17.0;
const ROW_H: f32 = 24.0;
const BAR_H: f32 = 28.0;
/// The chart's room above its rectangle (title and buttons) and below it (the numbers
/// along its axis and the axis's name), and to its left (the numbers up its side).
const CHART_ABOVE: f32 = 50.0;
const CHART_BELOW: f32 = 34.0;
const CHART_LEFT: f32 = 40.0;

#[derive(Clone, Debug)]
struct Layout {
    /// A phone held upright: the panel goes below the picture.
    narrow: bool,
    picture: (Vec2, Vec2),
    panel: Option<Panel>,
}

/// The widest `cell` can be, in characters of its number (a minus sign counted as two).
const CELL: usize = 6;

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let narrow = view.x < 620.0 && view.y >= 560.0;
        let pad = if narrow { 10.0 } else { 12.0 };
        let width = if narrow { view.x } else { 340.0 };
        // the matrices: four to a row below a picture wide enough for that, two
        // otherwise, as large as fits, up to 13
        let count = if narrow { 4 } else { 6 };
        let per_row = if narrow && width - 2.0 * pad + 12.0 >= 4.0 * (matrix_size(2, 2, CELL, 13.0).x + 16.0) { 4 } else { 2 };
        let column = (width - 2.0 * pad - 12.0 * (per_row - 1) as f32) / per_row as f32;
        let mut text = 13.0f32;
        while matrix_size(2, 2, CELL, text).x > column && text > 9.5 {
            text -= 0.5;
        }
        let row = CAP_H + matrix_size(2, 2, CELL, text).y + 8.0;
        let rows = count / per_row;
        let line_count = if narrow { 2 } else { 3 };
        let above = pad + rows as f32 * row + line_count as f32 * LINE_H + 8.0 + ROW_H + 8.0 + BAR_H + 10.0;
        let whole = Layout { narrow, picture: (Vec2::new(0.0, inset), view), panel: None };
        let place = |min: Vec2, width: f32, plot_h: Option<f32>| {
            let x = min.x + pad;
            let mut y = min.y + pad;
            let mut cells = Vec::new();
            for _ in 0..rows {
                for j in 0..per_row {
                    cells.push(Vec2::new(x + j as f32 * (column + 12.0), y));
                }
                y += row;
            }
            let lines = Vec2::new(x, y);
            y += line_count as f32 * LINE_H + 8.0;
            let n_row = Vec2::new(x, y);
            y += ROW_H + 8.0;
            let play = (Vec2::new(min.x + width - pad - 30.0, y), Vec2::new(min.x + width - pad, y + BAR_H));
            let bar = (Vec2::new(x + 16.0, y + BAR_H * 0.5), Vec2::new(play.0.x - 78.0, y + BAR_H * 0.5));
            y += BAR_H + 10.0;
            let chart_buttons = Vec2::new(x, y + 20.0);
            let chart = plot_h.map(|h| (Vec2::new(x + CHART_LEFT, y + CHART_ABOVE), Vec2::new(min.x + width - pad - 8.0, y + CHART_ABOVE + h)));
            let bottom = chart.map_or(y, |c| c.1.y + CHART_BELOW);
            Panel { min, max: Vec2::new(min.x + width, bottom + pad), text, column, cells, lines, line_count, n_row, bar, play, chart_buttons, chart }
        };
        let chart_extra = CHART_ABOVE + CHART_BELOW;
        if narrow {
            // the panel below the picture, which keeps at least the top 44 %
            let room = view.y * 0.58;
            let plot_h = (room - above - chart_extra - pad).min(150.0);
            let plot_h = (plot_h >= 70.0).then_some(plot_h);
            if room < above {
                return whole;
            }
            let height = above + plot_h.map_or(0.0, |h| h + chart_extra) + pad;
            let top = view.y - height;
            let mut panel = place(Vec2::new(0.0, top), view.x, plot_h);
            panel.max = view;
            return Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, top)), panel: Some(panel) };
        }
        let top = inset + 12.0;
        let room = view.y - 12.0 - top;
        let plot_h = (room - above - chart_extra - pad).min(170.0);
        let plot_h = (plot_h >= 70.0).then_some(plot_h);
        if view.x < 600.0 || room < above + pad {
            return whole;
        }
        let panel = place(Vec2::new(view.x - width - 12.0, top), width, plot_h);
        let picture = (Vec2::new(0.0, inset), Vec2::new(panel.min.x - 8.0, view.y));
        Layout { narrow, picture, panel: Some(panel) }
    }

    /// The part of the canvas the scene's frame is fitted into: the picture's, less the
    /// heading at its top and the caption at its foot.
    fn frame_room(&self) -> (Vec2, Vec2) {
        let (min, max) = self.picture;
        let foot = if self.narrow { 12.0 } else { 56.0 };
        (min + Vec2::new(12.0, 58.0), max - Vec2::new(12.0, foot))
    }
}

fn inside_rect(p: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

/// A number for a matrix on the screen, in five characters at most (so that the
/// brackets never move as the numbers change): two decimals, then one, then none, and
/// a power of ten beyond 9999.
fn cell(v: f64) -> String {
    let a = v.abs();
    let number = if !v.is_finite() {
        "?".into()
    } else if a < 9.995 {
        num(v, 2)
    } else if a < 99.95 {
        num(v, 1)
    } else if a < 9999.5 {
        num(v, 0)
    } else {
        format!("{v:.0e}").replace('-', "−")
    };
    format!("${number}$")
}

/// A gap or a size for the screen: three figures, in powers of ten when it is small or
/// large, and 0 for no gap at all.
fn figures(v: f64) -> String {
    if v == 0.0 {
        "0".into()
    } else if !v.is_finite() {
        "?".into()
    } else if (0.01..1000.0).contains(&v.abs()) {
        let decimals = (2 - v.abs().log10().floor() as i32).clamp(0, 4) as usize;
        num(v, decimals)
    } else {
        format!("{v:.2e}").replace('-', "−")
    }
}

/// The parts of the picture drawn as lines whose number of points can change from one
/// frame to the next (or that come and go), each in a batch of its own: different
/// from the usual batch by a hair of their width and nothing else, so that `eidos
/// motion` can still compare the rest point for point.
#[derive(Clone, Copy, PartialEq)]
enum Part {
    Smooth = 1,
    Zig,
    Other,
    Field,
    Gap,
}

fn lines_for<'a>(f: &'a mut Frame, part: Part) -> &'a mut crate::gfx::draw::LineBatch {
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 1e-4 * part as u8 as f32, ..Default::default() })
}

/// Line kinds on the chart (see `plot::apart`): what keeps its points, and what may
/// gain or lose some as it crosses the chart's edge.
const CURVE_KIND: [u8; 2] = [plot::OWN, plot::OWN + 1];
const SLOPE_KIND: [u8; 2] = [plot::OWN + 2, plot::OWN + 3];

/// The furthest the drawing goes beyond the view, as a share of its size: what runs
/// further is kept at that box (the same number of points, as it swings in and out).
fn keep_in(p: DVec2, lo: DVec2, hi: DVec2) -> Vec3 {
    let q = if p.is_finite() { p.max(lo).min(hi) } else { lo };
    q.as_vec2().extend(0.0)
}

fn inside_box(p: DVec2, lo: DVec2, hi: DVec2) -> bool {
    p.is_finite() && p.cmpge(lo).all() && p.cmple(hi).all()
}

/// The box a label takes on the screen, nearly: its text as wide as it is reckoned to be.
fn label_box(l: &crate::gfx::draw::Label) -> (Vec2, Vec2) {
    let size = Vec2::new(text_width(&l.text, l.size), l.size + 2.0);
    let share = |a: egui::Align| match a {
        egui::Align::Min => 0.0,
        egui::Align::Center => 0.5,
        egui::Align::Max => 1.0,
    };
    let min = l.pos - size * Vec2::new(share(l.align.x()), share(l.align.y()));
    (min, min + size)
}

/// Whether two boxes on the screen meet.
fn overlap(a: (Vec2, Vec2), b: (Vec2, Vec2)) -> bool {
    a.0.x < b.1.x && b.0.x < a.1.x && a.0.y < b.1.y && b.0.y < a.1.y
}

/// Keep names off the numbers the axes are marked with (where `Frame::axes2` puts them).
fn keep_ticks_clear(f: &Frame, room: &mut Room) {
    let (lo, hi) = f.cam.flat_bounds();
    let view = f.view_size();
    let wpp = (hi.y - lo.y) / view.y.max(1.0);
    if !(wpp.is_finite() && wpp > 0.0) {
        return;
    }
    let step = nice_step(wpp * 95.0);
    let margin = 16.0 * wpp;
    let y_anchor = 0.0f32.max(lo.y + margin * 1.6).min(hi.y - margin * 0.4);
    let x_anchor = 0.0f32.max(lo.x + margin * 0.4).min(hi.x - margin * 2.4);
    let count = |lo: f32, hi: f32| ((hi - lo) / step).floor().clamp(0.0, 200.0) as i32;
    let first = |lo: f32| (lo / step).ceil();
    for i in 0..=count(lo.x, hi.x) {
        let x = (first(lo.x) + i as f32) * step;
        if let Some(p) = on_screen(f, Vec3::new(x, y_anchor, 0.0)) {
            room.boxed(p + Vec2::new(-12.0, 4.0), p + Vec2::new(12.0, 18.0));
        }
    }
    for i in 0..=count(lo.y, hi.y) {
        let y = (first(lo.y) + i as f32) * step;
        if let Some(p) = on_screen(f, Vec3::new(x_anchor, y, 0.0)) {
            room.boxed(p + Vec2::new(5.0, -7.0), p + Vec2::new(30.0, 7.0));
        }
    }
}

/// "n = 4 rounds", "n = 1 round".
fn rounds_count(n: usize) -> String {
    format!("$n = {n}$ round{}", if n == 1 { "" } else { "s" })
}

// ----- the scene ---------------------------------------------------------------------

pub struct Expsum {
    glide: Glide,
    /// The number of rounds, and whether each is symmetric.
    n: usize,
    sym: bool,
    /// The pair last chosen from the list (none once A or B is changed by hand).
    pair: Option<usize>,
    playing: bool,
    /// Where playing stops; and play once the glide to a new setting has arrived.
    stop_at: f64,
    play_when_there: bool,
    /// Doubling n: seconds until the next doubling.
    counting: Option<f32>,
    /// Show the other order (A first, then B) too, and a velocity field (0 none, 1 A's,
    /// 2 B's, 3 both at once).
    other: bool,
    field: usize,
    chart: Chart,
    zig: Morph,
    other_zig: Morph,
    curves: Option<Curves>,
    /// Where the dot on the chart is drawn across it (gliding to where n is, or following
    /// a hand that drags it), and the hand's place while it does.
    dot: plot::Glide,
    dot_moving: bool,
    dot_drag: Option<f64>,
    /// The chart's vertical range, as powers of ten (its bottom and its top), gliding.
    range: plot::Glide,
    range_moving: bool,
    /// The canvas last drawn on and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and so where the picture is.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
}

impl Expsum {
    fn new() -> Self {
        let start = Setting { a: rows(PAIRS[0].a), b: rows(PAIRS[0].b), t: 1.0, x0: DVec2::from(X0) };
        let mut scene = Expsum {
            glide: Glide::at(start),
            n: 1,
            sym: false,
            pair: Some(0),
            playing: false,
            stop_at: T_MAX,
            play_when_there: false,
            counting: None,
            other: false,
            field: 0,
            chart: Chart::Rounds,
            zig: Morph::default(),
            other_zig: Morph::default(),
            curves: None,
            dot: plot::Glide::default(),
            dot_moving: false,
            dot_drag: None,
            range: plot::Glide::default(),
            range_moving: false,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
        };
        scene.set_up(0);
        // (opening, there is nothing to glide from)
        scene.glide = Glide::at(scene.glide.to);
        scene
    }

    /// Set up experiment `k` of `TRIALS`: glide to its setting, then do what it does.
    fn set_up(&mut self, k: usize) {
        let trial = &TRIALS[k];
        let pair = &PAIRS[trial.pair];
        self.glide.to = Setting { a: rows(pair.a), b: rows(pair.b), t: trial.t, x0: DVec2::from(trial.x0) };
        self.pair = Some(trial.pair);
        (self.n, self.sym) = (trial.n, trial.sym);
        self.set_chart(trial.chart);
        (self.playing, self.play_when_there, self.stop_at, self.counting) = (false, false, T_MAX, None);
        match trial.then {
            Then::Rest => {}
            Then::Play(to) => (self.play_when_there, self.stop_at) = (true, to),
            Then::Count => self.counting = Some(COUNT_EVERY),
        }
        (self.other, self.field) = (false, 0);
    }

    fn choose_pair(&mut self, k: usize) {
        self.glide.to.a = rows(PAIRS[k].a);
        self.glide.to.b = rows(PAIRS[k].b);
        self.pair = Some(k);
    }

    /// Chart the gap against n or against t (the dot on it starts where it belongs).
    fn set_chart(&mut self, chart: Chart) {
        if chart != self.chart {
            (self.chart, self.dot, self.dot_drag) = (chart, plot::Glide::default(), None);
        }
    }

    fn set_n(&mut self, n: usize) {
        self.n = n.clamp(1, N_MAX);
        self.counting = None;
    }

    /// The setting as drawn.
    fn shown(&self) -> Setting {
        self.glide.shown
    }

    /// Play from where t is (from 0, if it is at the end).
    fn play(&mut self) {
        self.stop_at = T_MAX;
        if self.glide.to.t >= T_MAX - 1e-9 {
            self.glide.to.t = 0.0;
            self.play_when_there = true;
        } else {
            self.playing = true;
        }
    }

    fn stop(&mut self) {
        (self.playing, self.play_when_there) = (false, false);
    }

    /// Move everything on: `dt` seconds of the scene's time (playing, counting) and
    /// `real_dt` of the clock's (a change of setting glides by the clock, so that it is
    /// seen through even while the scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        self.glide.step(real_dt as f64);
        if self.play_when_there && self.glide.arrived() {
            (self.play_when_there, self.playing) = (false, true);
        }
        if self.playing {
            let t = (self.glide.to.t + dt as f64 * PLAY_RATE).min(self.stop_at);
            self.glide.now(|s| s.t = t);
            if t >= self.stop_at {
                self.playing = false;
            }
        }
        if let Some(left) = self.counting {
            if self.glide.arrived() {
                let left = left - dt;
                if left <= 0.0 {
                    self.n = (self.n * 2).min(N_MAX);
                    self.counting = (self.n < COUNT_TO).then_some(left + COUNT_EVERY);
                } else {
                    self.counting = Some(left);
                }
            }
        }
        // the zig-zags, for the setting as it is now drawn (worked out again only when
        // that has changed), and gliding to a new count or kind of round
        let s = self.shown();
        let of: Vec<u64> = s.numbers().iter().map(|v| v.to_bits()).chain([self.n as u64, self.sym as u64]).collect();
        if self.zig.of != of {
            self.zig.aim((self.n, self.sym), zig(s.a, s.b, s.x0, s.t, self.n, self.sym));
            self.zig.of = of.clone();
        }
        self.zig.step(real_dt as f64);
        if self.other {
            if self.other_zig.of != of {
                self.other_zig.aim((self.n, self.sym), zig(s.b, s.a, s.x0, s.t, self.n, self.sym));
                self.other_zig.of = of;
            }
            self.other_zig.step(real_dt as f64);
        } else {
            self.other_zig = Morph::default();
        }
    }

    /// The canonical view: the picture's frame in the middle of the part of the canvas
    /// it has to itself.
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset);
        let (min, max) = lay.frame_room();
        let room = (max - min).max(Vec2::splat(60.0));
        let per_unit = (room.x / (2.0 * FRAME_HALF.x)).min(room.y / (2.0 * FRAME_HALF.y)).max(1e-3);
        // (screen y runs down, the scene's up)
        let off = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        View::Fit { center: Vec2::new(FRAME_CENTRE.x - off.x, FRAME_CENTRE.y + off.y), half: self.canvas.max(Vec2::ONE) * (0.5 / per_unit) }
    }

    /// The gaps the chart draws, for the setting as drawn (worked out again only when
    /// that has changed).
    fn curves(&mut self) -> &Curves {
        let Setting { a, b, t, .. } = self.shown();
        let fresh = curves_of_key(a, b, t, self.n, self.chart);
        if self.curves.as_ref().is_none_or(|c| c.of != fresh) {
            self.curves = Some(curves_of(a, b, t, self.n, self.chart));
        }
        self.curves.get_or_insert_with(|| curves_of(a, b, t, 1, Chart::Rounds))
    }
}

/// What `curves_of` works the curves out from, as it keeps it.
fn curves_of_key(a: DMat2, b: DMat2, t: f64, n: usize, chart: Chart) -> Vec<u64> {
    let mut of: Vec<u64> = a.to_cols_array().iter().chain(b.to_cols_array().iter()).map(|v| v.to_bits()).collect();
    match chart {
        Chart::Rounds => of.extend([0, t.to_bits()]),
        Chart::Time => of.extend([1, n as u64]),
    }
    of
}

/// The matrix of a 2×2 to drag or type, within ±`ENTRY_MAX`. Returns (changed, typed):
/// a number typed in glides the picture to it, one dragged moves it at once.
fn matrix_edit(ui: &mut egui::Ui, id: &str, m: &mut DMat2) -> (bool, bool) {
    let (mut changed, mut typed) = (false, false);
    let mut a = [[m.x_axis.x, m.y_axis.x], [m.x_axis.y, m.y_axis.y]];
    egui::Grid::new(id).spacing([6.0, 4.0]).show(ui, |ui| {
        for row in &mut a {
            for v in row.iter_mut() {
                let r = ui.add(egui::DragValue::new(v).speed(0.01).fixed_decimals(2).range(-ENTRY_MAX..=ENTRY_MAX));
                changed |= r.changed();
                typed |= r.changed() && !r.dragged();
            }
            ui.end_row();
        }
    });
    if changed {
        *m = rows([a[0][0], a[0][1], a[1][0], a[1][1]]);
    }
    (changed, typed)
}

/// The name of the rounds' product, as a formula: e^(tA) e^(tB) for one plain round.
fn rounds_name(n: usize, sym: bool, other: bool) -> String {
    let (p, q) = if other { ("B", "A") } else { ("A", "B") };
    match (n, sym) {
        (1, false) => format!("e^{{t{p}}} e^{{t{q}}}"),
        (1, true) => format!("e^{{t{p}/2}} e^{{t{q}}} e^{{t{p}/2}}"),
        (_, false) => format!(r"(e^{{t{p}/n}} e^{{t{q}/n}})^n"),
        (_, true) => format!(r"(e^{{t{p}/2n}} e^{{t{q}/n}} e^{{t{p}/2n}})^n"),
    }
}

/// The matrix of the drawn setting's rounds, in the order asked for.
fn rounds_of(s: &Setting, n: usize, sym: bool, other: bool) -> DMat2 {
    if other {
        rounds(s.b, s.a, s.t, n, sym)
    } else {
        rounds(s.a, s.b, s.t, n, sym)
    }
}

impl Scene for Expsum {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.10;
        s.axis_alpha = 0.35;
    }

    fn animated(&self) -> bool {
        self.playing || self.play_when_there || !self.glide.arrived() || self.counting.is_some() || self.zig.moving() || self.other_zig.moving() || self.range_moving || self.dot_moving
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"For numbers, $e^{a+b} = e^a e^b$. For matrices, read $Ax$ and $Bx$ as velocities, as in the scene on the matrix exponential. Flowing with both at once, the velocity $Ax + Bx$, for a time $t$ takes $x_0$ to $e^{t(A+B)} x_0$ (white). Flowing by $B$ for that time and then by $A$ takes it to $e^{tA} e^{tB} x_0$ (blue, then orange). The two agree when $AB = BA$; in general they do not, and the red line is the gap.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The pair $A$, $B$");
        let names: Vec<&str> = PAIRS.iter().map(|p| p.name).collect();
        if let Some(k) = ui::buttons(ui, &names) {
            self.choose_pair(k);
        }
        if let Some(p) = self.pair.map(|k| &PAIRS[k]) {
            ui::note(ui, p.note);
        }
        for (name, id, which) in [(r"$A$ (orange)", "expsum_a", 0), (r"$B$ (blue)", "expsum_b", 1)] {
            ui.label(ui::label_text(ui, name, 12.5, Some(pal::DIM), false));
            let mut m = if which == 0 { self.glide.to.a } else { self.glide.to.b };
            let (changed, typed) = matrix_edit(ui, id, &mut m);
            if changed {
                self.pair = None;
                if typed {
                    if which == 0 {
                        self.glide.to.a = m;
                    } else {
                        self.glide.to.b = m;
                    }
                } else {
                    self.glide.now(|s| if which == 0 { s.a = m } else { s.b = m });
                }
            }
        }

        ui::section(ui, "Time and rounds");
        ui.horizontal(|ui| {
            let busy = self.playing || self.play_when_there;
            if ui.button(if busy { "❚❚ pause" } else { "▶ play" }).clicked() {
                if busy {
                    self.stop();
                } else {
                    self.play();
                }
            }
            if ui.button(ui::label_text(ui, "back to $t = 0$", 12.5, None, false)).clicked() {
                self.glide.to.t = 0.0;
                self.stop();
            }
        });
        let mut t = self.glide.to.t as f32;
        if ui::slider(ui, "time $t$", &mut t, 0.0..=T_MAX as f32) {
            self.glide.now(|s| s.t = t as f64);
            self.stop();
        }
        ui::note(ui, "How long each flow is followed for. Play runs it on at a quarter of a unit a second.");
        let mut n = self.n;
        if ui::slider_int_log(ui, "rounds $n$", &mut n, 1..=N_MAX) {
            self.set_n(n);
        }
        ui.horizontal(|ui| {
            if ui.button(ui::label_text(ui, "double $n$ from 1", 12.5, None, false)).clicked() {
                self.set_n(1);
                self.counting = Some(COUNT_EVERY);
            }
        });
        let mut kind = self.sym as usize;
        if ui::choice(ui, &["plain: $B$, then $A$", r"symmetric: $\tfrac12 A$, $B$, $\tfrac12 A$"], &mut kind) {
            self.sym = kind == 1;
        }
        ui::note(ui, r"A round is $B$ for the time $t/n$ and then $A$ for $t/n$; or, symmetric, $A$ for $t/2n$, $B$ for $t/n$ and $A$ for $t/2n$. The zig-zag is $n$ rounds.");

        ui::section(ui, "Show");
        ui::checkbox(ui, &mut self.other, "the other order too: $A$ first, then $B$");
        ui::choice(ui, &["no field", "$Ax$", "$Bx$", "$Ax + Bx$"], &mut self.field);
        ui::note(ui, "The velocity field: an arrow at each point of a grid, how far that velocity would carry it in a fifth of a unit of time.");

        ui::section(ui, "Reading the picture");
        let mut key = vec![
            (pal::WHITE, r"white: the path of $x_0$ (drag it) with both velocities at once, $Ax + Bx$, for the time $t$; it ends at $e^{t(A+B)} x_0$"),
            (pal::BLUE, r"blue: flowing by $B$ alone, velocity $Bx$"),
            (pal::ORANGE, r"orange: flowing by $A$ alone, velocity $Ax$. Blue and orange in turn are the zig-zag: $n$ rounds"),
            (pal::TEXT, "dots on it: where each round ends"),
            (pal::RED, r"red: the gap between where the two end; on the chart, the gap between the matrices, and $AB - BA$, where it comes from"),
        ];
        if self.other {
            key.push((pal::DIM, r"faint: the other order, $A$ first, then $B$"));
        }
        key.push((pal::TEXT, r"the chart: the gap between the matrices against $n$ (or $t$) on logarithmic axes, red for the kind of round drawn and grey for the other (drag the red dot). Dashed: lines of a fixed slope"));
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, "Flow:", r"what $\mathrm{d}x/\mathrm{d}t = Ax$ does to the plane: every point moves with the velocity $A$ times where it is. Followed for a time $t$, it is the matrix $e^{tA}$.");
        ui::term(ui, "Both at once:", r"the flow of $A + B$, whose velocity is $Ax + Bx$: $e^{t(A+B)}$.");
        ui::term(ui, "One after the other:", r"$e^{tA} e^{tB}$: the matrix on the right acts first, so this is $B$ for the time $t$ and then $A$.");
        ui::term(ui, "Commute:", r"$A$ and $B$ commute when $AB = BA$. Then $e^{tA} e^{tB} = e^{t(A+B)}$ for every $t$; and if those agree for every $t$, then $AB = BA$, since the difference is $(t^2/2)(AB - BA)$ plus terms in $t^3$ and higher powers.");
        ui::term(ui, "Shear:", r"a flow in which every point slides parallel to one line, at a speed in proportion to its distance from it. Both at once, the stretch and the turn make one: $(A + B)x$ points along the line $y = x$, and its size is in proportion to the distance of $x$ from that line.");
        ui::term(ui, r"$AB - BA$:", r"the commutator, often written $[A, B]$: how far the two orders of applying $A$ and $B$ differ.");
        ui::term(ui, r"Size $\lVert M \rVert$:", "the square root of the sum of the squares of the entries: the matrix's length as a list of four numbers.");
        ui::term(ui, "Rounds:", r"following $A + B$ by short turns of each alone (splitting it). The plain round is called Lie–Trotter splitting, the symmetric one Strang splitting.");
        ui::term(ui, "Logarithmic axes:", r"each step along an axis multiplies by the same factor. A gap that falls like $1/n$ is a straight line of slope $-1$ there; like $1/n^2$, of slope $-2$.");

        ui::section(ui, "Numbers");
        let s = self.shown();
        let both = expm(s.a + s.b, s.t);
        let ours = rounds_of(&s, self.n, self.sym, false);
        let c = size(bracket(s.a, s.b));
        ui::readout(ui, "$t$, $n$", format!("${}$, ${}$", num(s.t, 2), self.n));
        ui::readout(ui, r"$\lVert AB - BA \rVert$", format!("${}$", figures(c)));
        let matrix_gap = size(ours - both);
        let gap_of = |g: f64, scale: f64| if no_gap(g, scale) { "0".to_string() } else { format!("${}$", figures(g)) };
        ui::readout_colored(ui, "gap between the matrices", gap_of(matrix_gap, size(both)), pal::RED);
        let other_kind = gap(s.a, s.b, s.t, self.n, !self.sym);
        ui::readout(ui, if self.sym { "the same, plain rounds" } else { "the same, symmetric rounds" }, gap_of(other_kind, size(both)));
        let at = (ours * s.x0 - both * s.x0).length();
        ui::readout_colored(ui, r"gap at $x_0$", gap_of(at, (both * s.x0).length()), pal::RED);
        if !self.sym {
            ui::readout(ui, r"$(t^2/2n) \lVert AB - BA \rVert$", format!("${}$", figures(s.t * s.t / (2.0 * self.n as f64) * c)));
        } else {
            ui::readout(ui, r"$(t^3/n^2) \lVert C \rVert$", format!("${}$", figures(s.t.powi(3) / (self.n * self.n) as f64 * size(third(s.a, s.b)))));
            ui::note(ui, r"$C = [B, [B, A]]/12 - [A, [A, B]]/24$: what one symmetric round of length $h$ gets wrong first, $h^3 C$.");
        }
        if self.other {
            let theirs = rounds_of(&s, self.n, self.sym, true);
            ui::readout(ui, "gap, the other order", gap_of(size(theirs - both), size(both)));
        }
        ui::see_also(ui, &[("expm", "the matrix exponential: one flow"), ("commute", "matrices that commute"), ("symskew", "a matrix as a stretch plus a spin"), ("expskew", "spins in space, and the turns they make")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        let before = Layout::of(self.canvas, self.inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.panel.is_some() != before.panel.is_some() || (view - self.canvas).abs().max_element() > 24.0);
        if !covered && (changed || !self.measured) {
            self.new_view |= changed;
            (self.canvas, self.measured) = (view, true);
        }
        if !covered {
            self.new_view |= (f.top_inset - self.inset).abs() > 1.0;
            self.inset = f.top_inset;
        }

        let pointer = f.ptr.pos.map(|p| p / f.px);
        if let Some(panel) = lay.panel.as_ref().filter(|_| !covered) {
            if pointer.is_some_and(|p| inside_rect(p, (panel.min, panel.max))) {
                // the panel is for its own handles and buttons: no moving the view from
                // there, and two quick clicks are two clicks
                f.handles.capture();
                f.handles.claim_clicks();
            }
            self.panel_input(f, panel);
        }
        self.advance(f.dt, f.real_dt);

        let labels_before = f.dl.labels.len();
        let plates = self.picture(f, &lay, covered);
        let numbers_from = f.dl.labels.len();
        f.axes2();
        // no number of the axes is written over the heading or the caption
        let mut k = 0;
        f.dl.labels.retain(|l| {
            k += 1;
            k <= numbers_from || !plates.iter().flatten().any(|plate| overlap(label_box(l), *plate))
        });
        if covered {
            f.dl.labels.truncate(labels_before);
            return;
        }
        // nothing written into the picture may show through the panel
        if let Some(p) = &lay.panel {
            let (lo, hi) = (p.min - Vec2::splat(2.0), p.max + Vec2::splat(2.0));
            let mut k = 0;
            f.dl.labels.retain(|l| {
                k += 1;
                k <= labels_before || l.backed || !inside_rect(l.pos, (lo, hi))
            });
            self.panel(f, p);
        } else {
            self.range_moving = false;
        }
    }
}

impl Expsum {
    /// The panel's buttons and handles: n, the kind of round, the time's knob and play
    /// button, the chart's buttons and the dot on the chart.
    fn panel_input(&mut self, f: &mut Frame, p: &Panel) {
        // n, as a row of buttons for its powers of two
        let size = 12.0;
        let label_w = 24.0;
        let room = p.max.x - p.n_row.x - label_w - 10.0;
        let counts = [1usize, 2, 4, 8, 16, 32, 64];
        let boxes = plot::even_row(counts.len(), p.n_row.x + label_w, p.n_row.x + label_w + room, p.n_row.y, ROW_H, 4.0);
        for (k, r) in counts.iter().zip(boxes) {
            if plot::button(f, r, &k.to_string(), size, self.n == *k) {
                self.set_n(*k);
            }
        }
        // the time track: its knob is a handle
        let (b0, b1) = p.bar;
        let t = self.shown().t;
        let knob = b0 + (b1 - b0) * (t / T_MAX).clamp(0.0, 1.0) as f32;
        if let Some(to) = plot::screen_handle(f, 2, knob, pal::WHITE) {
            let t = (((to.x - b0.x) / (b1.x - b0.x).max(1.0)) as f64 * T_MAX).clamp(0.0, T_MAX);
            self.glide.now(|s| s.t = t);
            self.stop();
        }
        let pointer = plot::pointer(f);
        if plot::button(f, p.play, if self.playing || self.play_when_there { "❚❚" } else { "▶" }, 13.0, false) {
            if self.playing || self.play_when_there {
                self.stop();
            } else {
                self.play();
            }
        } else if let Some(at) = pointer.filter(|_| f.clicked_canvas()) {
            if inside_rect(at, (b0 - Vec2::new(8.0, 12.0), b1 + Vec2::new(8.0, 12.0))) {
                // a click on the track: t glides there
                self.glide.to.t = (((at.x - b0.x) / (b1.x - b0.x).max(1.0)) as f64 * T_MAX).clamp(0.0, T_MAX);
                self.stop();
            }
        }
        // the chart's buttons: which kind of round, and against what
        let Some(rect) = p.chart else { return };
        let y = p.chart_buttons.y;
        let words = ["plain", "symmetric"];
        let left = plot::button_row(f, &words, p.chart_buttons.x, y, 22.0, 12.0, 4.0);
        for (k, r) in left.into_iter().enumerate() {
            if plot::button(f, r, words[k], 12.0, self.sym == (k == 1)) {
                self.sym = k == 1;
            }
        }
        let against = ["against $n$", "against $t$"];
        let widths: Vec<f32> = against.iter().map(|w| f.text_size(w, 12.0).x + 18.0).collect();
        let mut x = rect.1.x + 4.0 - widths.iter().sum::<f32>() - 4.0;
        for (k, w) in widths.iter().enumerate() {
            let r = (Vec2::new(x, y), Vec2::new(x + w, y + 22.0));
            let which = if k == 0 { Chart::Rounds } else { Chart::Time };
            if plot::button(f, r, against[k], 12.0, self.chart == which) {
                self.set_chart(which);
            }
            x += w + 4.0;
        }
        // the dot on the chart, for the kind of round drawn: drag it along to change n
        // (or t). While it is dragged it follows the hand, and n follows it in whole
        // steps; let go, it glides to where n is.
        let plot = self.plot(rect);
        let target = match (self.chart, self.dot_drag) {
            (Chart::Rounds, Some(x)) => x,
            (Chart::Rounds, None) => (self.n as f64).log10(),
            (Chart::Time, _) => self.shown().t.max(T_LOW).log10(),
        };
        // (t moves smoothly of itself: only a change of n is glided)
        if self.dot_drag.is_some() || self.chart == Chart::Time || self.dot.shown.len() != 1 {
            self.dot.snap(&[target as f32]);
            self.dot_moving = false;
        } else {
            self.dot_moving = self.dot.follow(&[target as f32], f.real_dt, false);
        }
        let x = self.dot.get(0) as f64;
        let kind = self.sym as usize;
        let y = self.curves().at(kind, x).max(plot.y[0]).min(plot.y[1]);
        match plot.handle(f, 3, x, y, pal::RED) {
            Some((to, _)) => match self.chart {
                Chart::Rounds => {
                    let to = to.clamp(0.0, (N_MAX as f64).log10());
                    self.dot_drag = Some(to);
                    let n = 10f64.powf(to).round() as usize;
                    if n != self.n {
                        self.set_n(n);
                    }
                }
                Chart::Time => {
                    let t = 10f64.powf(to).clamp(T_LOW, T_MAX);
                    self.glide.now(|s| s.t = t);
                    self.stop();
                }
            },
            None if !f.handles.dragging() => self.dot_drag = None,
            None => {}
        }
    }

    /// The chart's axes: n or t across, and the gap up, both by powers of ten.
    fn plot(&self, rect: plot::Rect) -> Plot {
        let x = match self.chart {
            Chart::Rounds => [-0.06, (N_MAX as f64).log10() + 0.06],
            Chart::Time => [T_LOW.log10(), T_MAX.log10()],
        };
        let y = [self.range.get(0) as f64, self.range.get(1) as f64];
        Plot::new(rect, x, if y[1] > y[0] { y } else { [-3.0, 1.0] })
    }

    /// Everything drawn in the scene's own coordinates, and the names written beside it.
    /// Returns the boxes of the heading and of the caption, which are written over it.
    fn picture(&mut self, f: &mut Frame, lay: &Layout, covered: bool) -> [Option<(Vec2, Vec2)>; 2] {
        let (lo, hi) = f.cam.flat_bounds();
        // what is drawn is kept a view's width beyond the view, whatever the flows do
        let span = (hi - lo).as_dvec2();
        let (box_lo, box_hi) = (lo.as_dvec2() - span, hi.as_dvec2() + span);

        // the start, which can be dragged (unless it lies under the panel)
        let mut x0 = self.shown().x0.as_vec2();
        let under_panel = lay.panel.as_ref().is_some_and(|p| on_screen(f, x0.extend(0.0)).is_some_and(|q| inside_rect(q, (p.min, p.max))));
        if (!under_panel || f.handles.dragging()) && f.drag2(1, &mut x0, pal::WHITE) {
            let x = x0.as_dvec2().clamp(DVec2::splat(-X0_MAX), DVec2::splat(X0_MAX));
            self.glide.now(|s| s.x0 = x);
        }
        let s = self.shown();
        let Setting { a, b, t, x0 } = s;

        // (the batches of the parts, made in the same order every frame)
        for part in [Part::Smooth, Part::Zig, Part::Other, Part::Field, Part::Gap] {
            lines_for(f, part);
        }

        // the velocity field, if asked for
        if self.field > 0 {
            let (m, c) = match self.field {
                1 => (a, pal::ORANGE),
                2 => (b, pal::BLUE),
                _ => (a + b, pal::WHITE),
            };
            let wpp = (hi.y - lo.y) / f.view_size().y.max(1.0);
            let step = nice_step(wpp * 56.0).max(1e-6);
            let count = |lo: f32, hi: f32| ((hi - lo) / step).floor().clamp(0.0, 80.0) as i32;
            let (i0, j0) = ((lo.x / step).ceil(), (lo.y / step).ceil());
            for i in 0..=count(lo.x, hi.x) {
                for j in 0..=count(lo.y, hi.y) {
                    let p = Vec2::new((i0 + i as f32) * step, (j0 + j as f32) * step);
                    let v = (m * p.as_dvec2()).as_vec2() * 0.2;
                    // (no longer than most of the way to the next point of the grid)
                    let v = v.clamp_length_max(step * 0.85);
                    if v.length() < 0.06 * step {
                        f.dl.discs.push(p.extend(0.0), 2.0, c.a(0.35));
                        continue;
                    }
                    let head = (5.0 * wpp).min(v.length() * 0.45);
                    let d = v.normalize();
                    let tip = p + v;
                    lines_for(f, Part::Field).segment(p.extend(0.0), (tip - d * head * 0.8).extend(0.0), 1.1, c.a(0.4));
                    f.dl.tris_alpha.tri(tip.extend(0.0), (tip - d * head + d.perp() * head * 0.45).extend(0.0), (tip - d * head - d.perp() * head * 0.45).extend(0.0), c.a(0.5));
                }
            }
        }

        // both at once
        let path = smooth(a, b, x0, t);
        lines_for(f, Part::Smooth).polyline(path.iter().map(|p| keep_in(*p, box_lo, box_hi)), 2.6, pal::WHITE.a(0.9));
        let both = *path.last().unwrap_or(&x0);

        // the other order, faint, under the zig-zag
        if self.other && !self.other_zig.target.points.is_empty() {
            let batch = lines_for(f, Part::Other);
            batch.brk();
            for i in 0..=U {
                // (its outer pieces are B's here)
                let c = pal::ORANGE.mix(pal::BLUE, self.other_zig.mix(i)).mix(pal::DIM, 0.45);
                batch.push(keep_in(self.other_zig.point(i), box_lo, box_hi), 1.4, c.a(0.55));
            }
            batch.brk();
        }

        // the zig-zag, blue for B and orange for A
        let batch = lines_for(f, Part::Zig);
        batch.brk();
        for i in 0..=U.min(self.zig.target.points.len().saturating_sub(1)) {
            let c = pal::BLUE.mix(pal::ORANGE, self.zig.mix(i));
            batch.push(keep_in(self.zig.point(i), box_lo, box_hi), 2.2, c);
        }
        batch.brk();
        let end = self.zig.target.points.last().map_or(x0, |_| self.zig.point(U.min(self.zig.target.points.len() - 1)));
        let end_exact = rounds(a, b, t, self.n, self.sym) * x0;
        // where each round ends (not while the zig-zag glides to a new count, nor when
        // they would be too close together to tell apart)
        let share = self.zig.arrived_share();
        if self.n <= 16 && share > 0.0 && !self.zig.target.points.is_empty() {
            let dots = f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 1.25 + 1e-4, ..Default::default() });
            for k in 1..self.n {
                let p = self.zig.point(round_end(k, self.n).min(U));
                if inside_box(p, box_lo, box_hi) {
                    dots.push(p.as_vec2().extend(0.0), 8.0, pal::TEXT.a(0.9 * share * share));
                }
            }
        }

        // the gap
        let scale = (both.length()).max(x0.length());
        let apart = (end_exact - both).length();
        let none = no_gap(apart, scale);
        if !none {
            lines_for(f, Part::Gap).segment(keep_in(both, box_lo, box_hi), keep_in(end, box_lo, box_hi), 2.4, pal::RED);
        }
        if inside_box(end, box_lo, box_hi) {
            f.dot(end.as_vec2(), 9.0, pal::ORANGE);
        }
        if inside_box(both, box_lo, box_hi) {
            f.dot(both.as_vec2(), 11.0, pal::WHITE);
        }
        let other_end = (self.other && !self.other_zig.target.points.is_empty()).then(|| self.other_zig.point(U));
        if let Some(p) = other_end.filter(|p| inside_box(*p, box_lo, box_hi)) {
            f.dot(p.as_vec2(), 8.0, pal::DIM);
        }
        let heading = self.heading(f, lay);
        let caption = self.caption(f, lay);
        let plates = [Some(heading), caption];
        if covered {
            return plates;
        }

        // the names, where there is room for them
        let mut room = Room::new(f);
        keep_ticks_clear(f, &mut room);
        if let Some(p) = &lay.panel {
            room.wall(p.min - Vec2::splat(4.0), p.max + Vec2::splat(4.0));
        }
        for plate in plates.iter().flatten() {
            room.wall(plate.0, plate.1);
        }
        for w in path.windows(2).step_by(8) {
            room.line(f, keep_in(w[0], box_lo, box_hi), keep_in(w[1], box_lo, box_hi));
        }
        let mut last = keep_in(x0, box_lo, box_hi);
        for i in (0..=U).step_by(8) {
            let p = keep_in(self.zig.point(i.min(self.zig.target.points.len().saturating_sub(1))), box_lo, box_hi);
            room.line(f, last, p);
            last = p;
        }
        if !none {
            room.line(f, keep_in(both, box_lo, box_hi), keep_in(end, box_lo, box_hi));
        }
        let away = |v: DVec2| Vec2::new(v.x as f32, -(v.y as f32));
        let seen = |f: &Frame, p: DVec2| p.is_finite() && on_screen(f, p.as_vec2().extend(0.0)).is_some_and(|q| inside_rect(q, (Vec2::ZERO, f.view_size())));
        if seen(f, x0) {
            room.label(f, "x0", x0.as_vec2().extend(0.0), "$x_0$", 16.0, pal::TEXT, -away((a + b) * x0), 10.0, false);
        }
        let ours = rounds_name(self.n, self.sym, false);
        // (the unit direction on the screen from one point of the scene to another)
        let towards = |p: DVec2, q: DVec2| away(q - p).normalize_or_zero();
        if none {
            if seen(f, both) {
                let text = format!(r"$e^{{t(A+B)}} x_0 = {ours} x_0$");
                room.label(f, "both", both.as_vec2().extend(0.0), &text, 14.0, pal::WHITE, away((a + b) * both), 10.0, false);
            }
        } else {
            let apart_px = on_screen(f, both.as_vec2().extend(0.0)).zip(on_screen(f, end.as_vec2().extend(0.0))).map_or(0.0, |(p, q)| p.distance(q));
            if seen(f, both) {
                let ahead = away((a + b) * both).normalize_or_zero();
                room.label(f, "both", both.as_vec2().extend(0.0), r"$e^{t(A+B)} x_0$", 14.0, pal::WHITE, ahead + towards(end, both), 10.0, false);
            }
            // (two ends close together on the screen are named as one, by the gap: the
            // rounds' end is named on the panel)
            if seen(f, end) && apart_px >= 48.0 {
                let text = format!("${ours} x_0$");
                room.label(f, "end", end.as_vec2().extend(0.0), &text, 14.0, pal::ORANGE.mix(pal::TEXT, 0.3), towards(both, end), 10.0, false);
            }
            let mid = (both + end) * 0.5;
            if seen(f, mid) && apart_px > 3.0 {
                let across = away((end - both).perp()).normalize_or_zero();
                room.label(f, "gap", mid.as_vec2().extend(0.0), &format!("gap ${}$", figures(apart)), 13.0, pal::RED, across, 8.0 + (24.0 - apart_px).max(0.0) * 0.5, false);
            }
        }
        // (the other order's end is named where there is room, and not beside the others)
        let px = |f: &Frame, p: DVec2| on_screen(f, p.as_vec2().extend(0.0));
        let clear = |f: &Frame, p: DVec2| [both, end].iter().all(|q| px(f, p).zip(px(f, *q)).is_some_and(|(p, q)| p.distance(q) >= 48.0));
        if let Some(p) = other_end.filter(|p| seen(f, *p) && clear(f, *p)) {
            let text = format!("${} x_0$", rounds_name(self.n, self.sym, true));
            room.label(f, "other", p.as_vec2().extend(0.0), &text, 13.0, pal::DIM, away(p - both), 10.0, true);
        }
        plates
    }

    /// The heading at the top of the picture: the pair, whether it commutes, and what
    /// the zig-zag is. Returns the box it takes.
    fn heading(&self, f: &mut Frame, lay: &Layout) -> (Vec2, Vec2) {
        let s = self.shown();
        let at = lay.picture.0 + Vec2::new(16.0, 14.0);
        // (the pair asked for: while the matrices glide to it, the heading is already its)
        let to = self.glide.to;
        let name = self.pair.map_or("your own pair", |k| PAIRS[k].name);
        let commutes = no_gap(size(bracket(to.a, to.b)), size(to.a) * size(to.b));
        let first = format!("{name}: {}", if commutes { "$AB = BA$" } else { r"$AB \ne BA$" });
        let count = rounds_count(self.n);
        let second = match (self.sym, lay.narrow) {
            (true, false) => format!(r"white: both at once, for $t = {}$. Orange and blue: half of $A$, $B$, half of $A$, in {count}", num(s.t, 2)),
            (false, false) => format!(r"white: both at once, for $t = {}$. Blue, then orange: $B$, then $A$, in {count}", num(s.t, 2)),
            (true, true) => format!(r"half of $A$, $B$, half of $A$: {count}"),
            (false, true) => format!(r"blue, then orange: $B$, then $A$: {count}"),
        };
        let room = (lay.picture.1.x - lay.picture.0.x - 32.0).max(40.0);
        let size = if f.text_size(&first, 15.0).x <= room { 15.0 } else { 13.0 };
        let small = if f.text_size(&second, 12.5).x <= room { 12.5 } else { 11.0 };
        let wide = f.text_size(&first, size).x.max(f.text_size(&second, small).x).min(room);
        f.caption_px(at, first.as_str(), size, pal::TEXT, Align2::LEFT_TOP);
        f.caption_px(at + Vec2::new(0.0, size + 8.0), second, small, pal::DIM, Align2::LEFT_TOP);
        (at - Vec2::splat(6.0), at + Vec2::new(wide + 12.0, size + small + 16.0))
    }

    /// The caption along the foot of the picture (none on a phone, where the panel is
    /// there). Returns the box it takes.
    fn caption(&self, f: &mut Frame, lay: &Layout) -> Option<(Vec2, Vec2)> {
        if lay.narrow || f.view_size().y < 300.0 {
            return None;
        }
        let parts = [r"white: both at once, velocity $Ax + Bx$", "blue: flowing by $B$", "orange: flowing by $A$", "red: the gap", "dots: where each round ends"];
        let size = 12.0;
        let (left, right, bottom) = (lay.picture.0.x, lay.picture.1.x, lay.picture.1.y - 16.0);
        let room = right - left - 36.0;
        let gap = "      ";
        let mut lines: Vec<String> = Vec::new();
        for part in parts {
            match lines.last_mut() {
                Some(last) if f.text_size(last, size).x + text_width(gap, size) + f.text_size(part, size).x <= room => {
                    last.push_str(gap);
                    last.push_str(part);
                }
                _ => lines.push(part.to_string()),
            }
        }
        let count = lines.len();
        let mut widest = 0.0f32;
        for (k, line) in lines.into_iter().enumerate() {
            widest = widest.max(f.text_size(&line, size).x);
            let up = (count - 1 - k) as f32 * (size + 7.0);
            f.caption_px(Vec2::new(left + 18.0, bottom - up), line, size, pal::DIM, Align2::LEFT_BOTTOM);
        }
        Some((Vec2::new(left + 10.0, bottom - count as f32 * (size + 7.0) - 4.0), Vec2::new(left + 26.0 + widest, bottom + 6.0)))
    }

    /// The screen panel: the matrices, the sizes and gaps, the buttons for n, the time
    /// track and the chart.
    fn panel(&mut self, f: &mut Frame, p: &Panel) {
        hud_panel(f, p.min, p.max);
        let s = self.shown();
        let both = expm(s.a + s.b, s.t);
        let ours = rounds_of(&s, self.n, self.sym, false);
        let theirs = rounds_of(&s, self.n, self.sym, true);
        let c = bracket(s.a, s.b);
        let entries = |m: DMat2, color: Color| -> Vec<Vec<(String, Color)>> { (0..2).map(|i| (0..2).map(|j| (cell(m.col(j)[i]), color)).collect()).collect() };
        let zig_colour = pal::ORANGE.mix(pal::TEXT, 0.45);
        let mut cells: Vec<(String, DMat2, Color)> = vec![
            ("$A$".into(), s.a, pal::ORANGE),
            ("$B$".into(), s.b, pal::BLUE),
            (r"both at once: $e^{t(A+B)}$".into(), both, pal::WHITE),
            (format!("${}$", rounds_name(self.n, self.sym, false)), ours, zig_colour),
        ];
        if p.cells.len() >= 6 {
            cells.push(("$AB - BA$".into(), c, pal::RED.mix(pal::TEXT, 0.35)));
            cells.push((format!("other order: ${}$", rounds_name(self.n, self.sym, true)), theirs, if self.other { pal::TEXT } else { pal::DIM }));
        }
        let text = p.text;
        for ((name, m, color), at) in cells.iter().zip(&p.cells) {
            // (a name wider than its column is set smaller, to fit)
            let wide = f.text_size(name, 12.0).x;
            let room = p.column + 8.0;
            let size = if wide > room { (12.0 * room / wide).max(8.5) } else { 12.0 };
            f.text_px(*at, name.as_str(), size, *color, Align2::LEFT_TOP);
            hud_matrix(f, *at + Vec2::new(0.0, CAP_H), &entries(*m, *color), CELL, text, pal::DIM);
        }

        // the sizes and the gaps
        let small = 12.0;
        let gap_text = |g: f64, scale: f64| if no_gap(g, scale) { "0".to_string() } else { figures(g) };
        let lines = [
            (format!(r"$\lVert AB - BA \rVert = {}$", figures(size(c))), pal::RED.mix(pal::TEXT, 0.35)),
            (format!(r"gap $\lVert {} - e^{{t(A+B)}} \rVert = {}$", rounds_name(self.n, self.sym, false), gap_text(size(ours - both), size(both))), pal::RED),
            (format!(r"the other order's: ${}$", gap_text(size(theirs - both), size(both))), pal::DIM),
        ];
        for (k, (line, color)) in lines.iter().take(p.line_count).enumerate() {
            let at = p.lines + Vec2::new(0.0, k as f32 * LINE_H);
            let wide = f.text_size(line, small).x;
            let room = p.max.x - p.lines.x - 10.0;
            let size = if wide > room { (small * room / wide).max(8.5) } else { small };
            f.text_px(at, line.as_str(), size, *color, Align2::LEFT_TOP);
        }

        // n's row (its buttons are drawn with the panel's input)
        f.text_px(p.n_row + Vec2::new(0.0, ROW_H * 0.5), "$n$", 14.0, pal::TEXT, Align2::LEFT_CENTER);

        // the time track
        let (b0, b1) = p.bar;
        f.text_px(b0 - Vec2::new(14.0, 0.0), "$t$", 14.0, pal::TEXT, Align2::CENTER_CENTER);
        hud_line(f, b0, b1, 2.0, pal::DIM);
        for (v, name) in [(0.0, "$0$"), (1.0, "$1$"), (2.0, "$2$")] {
            let x = b0.x + (b1.x - b0.x) * (v / T_MAX) as f32;
            hud_line(f, Vec2::new(x, b0.y - 4.0), Vec2::new(x, b0.y + 4.0), 1.2, pal::DIM);
            f.text_px(Vec2::new(x, b0.y + 7.0), name, 10.5, pal::DIM, Align2::CENTER_TOP);
        }
        let knob = b0 + (b1 - b0) * (s.t / T_MAX).clamp(0.0, 1.0) as f32;
        hud_line(f, b0, knob, 3.0, pal::WHITE.a(0.8));
        f.text_px(Vec2::new(b1.x + 14.0, b0.y), format!("$t = {}$", num(s.t, 2)), 13.0, pal::TEXT, Align2::LEFT_CENTER);

        match p.chart {
            Some(rect) => self.draw_chart(f, rect),
            None => self.range_moving = false,
        }
    }

    /// The chart of the gap between the matrices, against n or t, on logarithmic axes:
    /// both kinds of round, the one drawn bright red, with dashed lines of their slopes.
    fn draw_chart(&mut self, f: &mut Frame, rect: plot::Rect) {
        let chart = self.chart;
        let (n, sym) = (self.n, self.sym);
        let s = self.shown();
        let curves = self.curves();
        let xs = curves.xs.clone();
        let gaps = curves.gaps.clone();
        let scale = curves.scale.clone();
        // the vertical range: from the smallest gap to the largest, in whole powers of
        // ten, six at most (gliding when it changes)
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        for (g, sc) in gaps.iter().flat_map(|g| g.iter().zip(&scale)) {
            if !no_gap(*g, *sc) && g.is_finite() {
                lo = lo.min(g.log10());
                hi = hi.max(g.log10());
            }
        }
        let any = hi >= lo;
        let (lo, hi) = if any { (lo.floor(), hi.ceil().max(lo.floor() + 1.0)) } else { (-3.0, 0.0) };
        let lo = lo.max(hi - 6.0);
        self.range_moving = self.range.follow(&[lo as f32, hi as f32], f.real_dt, false);
        let plot = self.plot(rect);
        let (min, max) = rect;
        f.text_px(Vec2::new(min.x - CHART_LEFT, min.y - CHART_ABOVE), "the gap between the matrices", 12.0, pal::TEXT, Align2::LEFT_TOP);

        // the axes, marked at powers of ten (and at n's powers of two)
        plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
        plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let marks: Vec<(f64, String)> = match chart {
            Chart::Rounds => [1, 2, 4, 8, 16, 32, 64].iter().map(|k| ((*k as f64).log10(), format!("${k}$"))).collect(),
            Chart::Time => [(0.01, "0.01"), (0.1, "0.1"), (1.0, "1"), (2.0, "2")].iter().map(|(v, name)| (f64::log10(*v), format!("${name}$"))).collect(),
        };
        for (v, name) in &marks {
            let x = plot.px_x(*v);
            plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, max.y + 6.0), name.as_str(), 10.5, pal::DIM, Align2::CENTER_TOP);
        }
        let across = match chart {
            Chart::Rounds => "rounds $n$",
            Chart::Time => "time $t$",
        };
        f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 19.0), across, 11.0, pal::DIM, Align2::CENTER_TOP);
        // (none when every gap is 0: there is nothing to measure)
        let (y0, y1) = if any { (plot.y[0].ceil() as i32, plot.y[1].floor() as i32) } else { (1, 0) };
        for k in y0..=y1 {
            let y = plot.px_y(k as f64);
            plot::line_of(f, plot::AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8));
            plot::line_of(f, plot::AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.12));
            let name = match k {
                0 => "$1$".to_string(),
                1 => "$10$".to_string(),
                -1 => "$0.1$".to_string(),
                _ => format!("$10^{{{k}}}$"),
            };
            f.text_px(Vec2::new(min.x - 6.0, y), name, 10.5, pal::DIM, Align2::RIGHT_CENTER);
        }

        // the two curves: the kind drawn in red, the other dim
        let xv = |x: f64| match chart {
            Chart::Rounds => x.log10(),
            Chart::Time => x.log10(),
        };
        let floor = plot.y[0];
        for (k, g) in gaps.iter().enumerate() {
            let color = if (k == 1) == sym { pal::RED } else { pal::DIM.a(0.7) };
            let pts = xs.iter().zip(g).zip(&scale).map(|((x, g), sc)| (xv(*x), if no_gap(*g, *sc) { floor } else { g.log10().max(floor) }));
            plot.path(f, CURVE_KIND[k], pts, if (k == 1) == sym { 2.2 } else { 1.5 }, color);
        }
        // the slopes: through the curve's last point against n; the first term of the
        // gap against t
        let c = size(bracket(s.a, s.b));
        let c3 = size(third(s.a, s.b));
        for k in 0..2 {
            let (from, to, at) = match chart {
                Chart::Rounds => {
                    let last = gaps[k][N_MAX - 1];
                    if no_gap(last, scale[N_MAX - 1]) {
                        (None, None, 0.0)
                    } else {
                        // (beside the curve, three times above the plain one and a third
                        // of the symmetric one, where it can be seen)
                        let slope = -(k as f64 + 1.0);
                        let beside = if k == 0 { 3f64.log10() } else { -(3f64.log10()) };
                        let line = |x: f64| last.log10() + beside + slope * (x - (N_MAX as f64).log10());
                        let (x0, x1) = (2f64.log10(), (N_MAX as f64).log10());
                        (Some((x0, line(x0))), Some((x1, line(x1))), slope)
                    }
                }
                Chart::Time => {
                    let coeff = if k == 0 { c / (2.0 * n as f64) } else { c3 / (n * n) as f64 };
                    if coeff <= 1e-12 {
                        (None, None, 0.0)
                    } else {
                        let power = k as f64 + 2.0;
                        let line = |x: f64| coeff.log10() + power * x;
                        let (x0, x1) = (plot.x[0], plot.x[1]);
                        (Some((x0, line(x0))), Some((x1, line(x1))), power)
                    }
                }
            };
            let (Some(p), Some(q)) = (from, to) else { continue };
            let pts = plot::clip(&[plot.at(p.0, p.1), plot.at(q.0, q.1)], rect);
            for piece in pts.split(|v| !v.is_finite()) {
                if piece.len() >= 2 {
                    let (a, b) = (piece[0], piece[piece.len() - 1]);
                    // (over the curves, on the panel's second level: where a line and its
                    // curve are one, the dashes show on it)
                    let (p, q) = (f.hud_at(2, a), f.hud_at(2, b));
                    plot::apart(f, SLOPE_KIND[k], true).segment(p, q, 1.4, pal::TEXT.a(0.7));
                    // the slope named a third of the way along, on the side of the line
                    // it does not run to there (above the plain one's, below the
                    // symmetric one's)
                    let (left, right) = if a.x <= b.x { (a, b) } else { (b, a) };
                    let at_point = left + (right - left) * 0.3;
                    let rising = right.y < left.y;
                    let (off, align) = match (k == 0, rising) {
                        (true, false) => (Vec2::new(4.0, -4.0), Align2::LEFT_BOTTOM),
                        (false, false) => (Vec2::new(-4.0, 4.0), Align2::RIGHT_TOP),
                        (true, true) => (Vec2::new(-4.0, -4.0), Align2::RIGHT_BOTTOM),
                        (false, true) => (Vec2::new(4.0, 4.0), Align2::LEFT_TOP),
                    };
                    f.text_px(at_point + off, format!("slope ${}$", num(at, 0)), 10.5, pal::TEXT.a(0.8), align);
                }
            }
        }
        if !any {
            let text = match chart {
                Chart::Rounds => "no gap: $0$ for every $n$",
                Chart::Time => "no gap: $0$ at every $t$",
            };
            f.text_px((min + max) * 0.5, text, 12.0, pal::TEXT, Align2::CENTER_CENTER);
        }
        // the other kind's dot, beside the drawn kind's (which is a handle: see
        // `panel_input`)
        let here = self.dot.get(0) as f64;
        if here >= plot.x[0] - 1e-9 {
            let y = self.curves().at(!sym as usize, here).max(floor).min(plot.y[1]);
            plot.dot(f, here, y, 7.0, pal::DIM);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(m: DMat2, n: DMat2, tol: f64) -> bool {
        (m - n).to_cols_array().iter().all(|v| v.abs() <= tol)
    }

    /// e^(At) by its series, summed until the terms are nothing.
    fn by_series(a: DMat2, t: f64) -> DMat2 {
        let (mut sum, mut term) = (DMat2::IDENTITY, DMat2::IDENTITY);
        for k in 1..=80 {
            term = term * a * (t / k as f64);
            sum += term;
        }
        sum
    }

    fn pair(k: usize) -> (DMat2, DMat2) {
        (rows(PAIRS[k].a), rows(PAIRS[k].b))
    }

    /// The closed form agrees with the series, for every pair and for many matrices, on
    /// both sides of a double eigenvalue and right at it.
    #[test]
    fn the_exponential_is_the_series() {
        let mut rng = Rng::new(7);
        let mut cases: Vec<DMat2> = PAIRS.iter().flat_map(|p| [rows(p.a), rows(p.b), rows(p.a) + rows(p.b)]).collect();
        cases.extend((0..300).map(|_| rows([0; 4].map(|_| rng.range(-3.0, 3.0) as f64))));
        for eps in [0.0, 1e-14, -1e-14, 1e-8, -1e-8, 1e-3, -1e-3] {
            cases.push(rows([-1.0, 1.0, eps, -1.0]));
        }
        cases.push(DMat2::ZERO);
        for a in cases {
            for t in [0.0, 0.1, 0.5, 1.0, 2.0, -0.7] {
                let (e, s) = (expm(a, t), by_series(a, t));
                assert!(close(e, s, 1e-11 * size(s).max(1.0)), "{a:?} at {t}");
            }
        }
    }

    /// The zig-zag is what its matrices say: it starts at x0, every round ends where
    /// that many rounds take x0, it ends where all n do, and it spends t in each flow.
    #[test]
    fn the_zig_zag_is_the_rounds() {
        let x0 = DVec2::from(X0);
        for k in 0..PAIRS.len() {
            let (a, b) = pair(k);
            for sym in [false, true] {
                for (n, t) in [(1, 1.0), (2, 0.7), (4, 1.0), (5, 1.3), (16, 2.0), (64, 1.0), (3, 0.0)] {
                    let z = zig(a, b, x0, t, n, sym);
                    assert_eq!(z.points.len(), U + 1);
                    assert_eq!(z.points[0], x0);
                    let r = round(a, b, t / n as f64, sym);
                    let mut p = x0;
                    for j in 1..=n {
                        p = r * p;
                        let i = round_end(j, n);
                        // (n = 5 has its round ends between points of the grid: within a step)
                        let tol = if U.is_multiple_of(n) { 1e-10 } else { 0.05 * t.max(0.1) * 3.0 };
                        assert!((z.points[i] - p).length() <= tol * p.length().max(1.0), "pair {k} sym {sym} n {n}: round {j}");
                    }
                    assert!((z.points[U] - rounds(a, b, t, n, sym) * x0).length() < 1e-10 * x0.length().max(1.0));
                    let pieces = pieces(t, n, sym);
                    assert_eq!(pieces.len(), if sym { 2 * n + 1 } else { 2 * n });
                    for which in [false, true] {
                        let spent: f64 = pieces.iter().filter(|p| p.0 == which).map(|p| p.1).sum();
                        assert!((spent - t).abs() < 1e-12);
                    }
                    // the colours: blue (0) in B's pieces, orange (1) in A's
                    if t > 0.0 {
                        assert_eq!(z.mix[1], if sym { 1.0 } else { 0.0 });
                        assert_eq!(z.mix[U], 1.0);
                    }
                }
            }
        }
    }

    /// For pairs that commute there is no gap at all, for any n and either kind of
    /// round, and every round ends on the path of both at once; for those that do not,
    /// there is.
    #[test]
    fn a_gap_exactly_when_ab_and_ba_differ() {
        for k in 0..PAIRS.len() {
            let (a, b) = pair(k);
            let commutes = no_gap(size(bracket(a, b)), size(a) * size(b));
            assert_eq!(commutes, k == 1 || k == 2, "pair {k}");
            for t in [0.3, 1.0, 2.0] {
                let scale = size(expm(a + b, t));
                for n in 1..=N_MAX {
                    for sym in [false, true] {
                        assert_eq!(no_gap(gap(a, b, t, n, sym), scale), commutes, "pair {k}, t {t}, n {n}, {sym}");
                    }
                }
            }
        }
        // the even stretch and the turn: four rounds end on the white spiral
        let (a, b) = pair(1);
        let x0 = DVec2::from(X0);
        let z = zig(a, b, x0, 1.0, 4, false);
        for j in 1..=4 {
            let on = expm(a + b, j as f64 / 4.0) * x0;
            assert!((z.points[round_end(j, 4)] - on).length() < 1e-12, "round {j}");
        }
    }

    /// The pairs' notes.
    #[test]
    fn the_pairs_are_what_their_notes_say() {
        // a stretch and a turn: AB - BA of size 2.83, and (A + B)^2 = 0, so e^(t(A+B)) = I + t(A + B)
        let (a, b) = pair(0);
        assert_eq!(figures(size(bracket(a, b))), "2.83");
        assert_eq!((a + b) * (a + b), DMat2::ZERO);
        for t in [0.5, 1.0, 2.0] {
            assert!(close(expm(a + b, t), DMat2::IDENTITY + (a + b) * t, 1e-15));
        }
        // (A + B)(x, y) = (x - y)(1, 1): along the line y = x, in proportion to the distance from it
        let mut rng = Rng::new(9);
        for _ in 0..50 {
            let p = DVec2::new(rng.range(-3.0, 3.0) as f64, rng.range(-3.0, 3.0) as f64);
            assert!(((a + b) * p - DVec2::ONE * (p.x - p.y)).length() < 1e-15);
        }
        // an even stretch commutes with anything
        let (a, _) = pair(1);
        let mut rng = Rng::new(5);
        for _ in 0..50 {
            let m = rows([0; 4].map(|_| rng.range(-3.0, 3.0) as f64));
            assert_eq!(bracket(a, m), DMat2::ZERO);
        }
        // two stretches along the axes
        let (a, b) = pair(2);
        assert!(a.x_axis.y == 0.0 && a.y_axis.x == 0.0 && b.x_axis.y == 0.0 && b.y_axis.x == 0.0 && bracket(a, b) == DMat2::ZERO);
        // two shears: AB - BA stretches along x and squeezes along y; A + B stretches
        // along y = x and squeezes along y = -x
        let (a, b) = pair(3);
        assert_eq!(a * DVec2::Y, DVec2::X);
        assert_eq!(b * DVec2::X, DVec2::Y);
        assert_eq!(bracket(a, b), rows([1.0, 0.0, 0.0, -1.0]));
        assert_eq!((a + b) * DVec2::ONE, DVec2::ONE);
        assert_eq!((a + b) * DVec2::new(1.0, -1.0), DVec2::new(-1.0, 1.0));
        // two stretches 45 degrees apart: AB - BA is a quarter turn clockwise, times 2
        let (a, b) = pair(4);
        assert_eq!(b * DVec2::ONE, DVec2::ONE);
        assert_eq!(b * DVec2::new(1.0, -1.0), DVec2::new(-1.0, 1.0));
        assert_eq!(bracket(a, b) * DVec2::X, DVec2::new(0.0, -2.0));
        assert_eq!(bracket(a, b) * DVec2::Y, DVec2::new(2.0, 0.0));
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// it has done what it does (or for `seconds` at most).
    fn run(k: usize, seconds: f32) -> Expsum {
        let mut scene = Expsum::new();
        scene.set_up(k);
        for _ in 0..(seconds * 60.0) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.glide.arrived() && !scene.playing && !scene.play_when_there && scene.counting.is_none() && !scene.zig.moving() {
                break;
            }
        }
        scene
    }

    #[test]
    fn the_stretch_and_the_turn_do_what_the_caption_says() {
        let scene = run(0, 30.0);
        let s = scene.shown();
        assert!(s.t == 1.0 && scene.n == 1 && !scene.sym && scene.pair == Some(0));
        let x0 = s.x0;
        // both at once: a straight line, to (2.60, 1.00)
        let path = smooth(s.a, s.b, x0, s.t);
        let dir = ((s.a + s.b) * x0).normalize();
        assert!(path.iter().all(|p| (*p - x0).perp_dot(dir).abs() < 1e-12));
        let both = expm(s.a + s.b, 1.0) * x0;
        assert_eq!(tuple(&[both.x, both.y], 2), "(2.60, 1.00)");
        // B, then A: to (2.84, 0.19), 0.845 away
        let end = expm(s.a, 1.0) * expm(s.b, 1.0) * x0;
        assert_eq!(tuple(&[end.x, end.y], 2), "(2.84, 0.19)");
        assert_eq!(figures((end - both).length()), "0.845");
        // the other order is another matrix again
        let other = expm(s.b, 1.0) * expm(s.a, 1.0);
        assert!(size(other - expm(s.a, 1.0) * expm(s.b, 1.0)) > 1.0 && size(other - expm(s.a + s.b, 1.0)) > 1.0);
        // the zig-zag drawn turns first (blue), then stretches (orange)
        assert!(scene.zig.target.mix[..U / 2].iter().all(|m| *m == 0.0) && scene.zig.target.mix[U / 2 + 1..].iter().all(|m| *m == 1.0));
        assert!((scene.zig.point(U) - end).length() < 1e-12);
    }

    #[test]
    fn the_pair_that_commutes_does_what_the_caption_says() {
        let scene = run(1, 30.0);
        let s = scene.shown();
        assert!(s.t == 1.0 && scene.n == 4 && !scene.sym && scene.pair == Some(1));
        assert_eq!(s.a, DMat2::IDENTITY * 0.5);
        assert_eq!(bracket(s.a, s.b), DMat2::ZERO);
        // each round a turn, then an even growth, for t/4 each, ending on the spiral
        assert_eq!(pieces(s.t, 4, false), [(false, 0.25), (true, 0.25)].repeat(4));
        for j in 1..=4 {
            let on = expm(s.a + s.b, s.t * j as f64 / 4.0) * s.x0;
            assert!((scene.zig.point(round_end(j, 4)) - on).length() < 1e-12);
        }
        // no gap at any n, either kind: the curves on the floor
        let curves = curves_of(s.a, s.b, s.t, 4, Chart::Rounds);
        assert!(curves.gaps.iter().all(|g| g.iter().zip(&curves.scale).all(|(g, sc)| no_gap(*g, *sc))));
    }

    #[test]
    fn small_steps_close_the_gap_as_the_caption_says() {
        let scene = run(2, 60.0);
        let s = scene.shown();
        // n doubled from 1 to 32, and stopped there
        assert!(scene.n == COUNT_TO && scene.counting.is_none() && !scene.sym && s.t == 1.0 && scene.pair == Some(0));
        let g = |n: usize| gap(s.a, s.b, s.t, n, false);
        assert_eq!([4, 8, 16, 32].map(|n| num(g(n), 3)), ["0.355", "0.177", "0.088", "0.044"]);
        // each doubling about halves it
        for n in [1, 2, 4, 8, 16, 32] {
            let ratio = g(n) / g(2 * n);
            assert!((ratio - 2.0).abs() < if n == 1 { 0.18 } else { 0.035 }, "{n}: {ratio}");
        }
        // and the zig-zag closes in on the white line: its furthest point from it shrinks
        let x0 = s.x0;
        let dir = ((s.a + s.b) * x0).normalize();
        let off = |n: usize| zig(s.a, s.b, x0, s.t, n, false).points.iter().map(|p| (*p - x0).perp_dot(dir).abs()).fold(0.0, f64::max);
        assert!([1, 2, 4, 8, 16, 32].windows(2).all(|w| off(w[1]) < 0.76 * off(w[0])) && off(32) < 0.06 * off(1), "{:?}", [1, 2, 4, 8, 16, 32].map(off));
    }

    #[test]
    fn the_symmetric_split_does_what_the_caption_says() {
        let scene = run(3, 30.0);
        let s = scene.shown();
        assert!(scene.n == 4 && scene.sym && s.t == 1.0 && scene.pair == Some(0));
        let g = |n: usize| gap(s.a, s.b, s.t, n, true);
        assert_eq!(num(g(4), 3), "0.042");
        assert_eq!(num(gap(s.a, s.b, s.t, 4, false), 3), "0.355");
        assert_eq!([num(g(4), 3), num(g(8), 4), num(g(16), 4), num(g(32), 5)], ["0.042", "0.0105", "0.0026", "0.00066"]);
        for n in [4, 8, 16, 32] {
            let ratio = g(n) / g(2 * n);
            assert!((ratio - 4.0).abs() < 0.03, "{n}: {ratio}");
        }
        // one piece more than the plain rounds: the two halves in a row are one piece
        let merged = |sym: bool| {
            let p = pieces(s.t, 4, sym);
            p.len()
        };
        assert_eq!((merged(false), merged(true)), (8, 9));
    }

    #[test]
    fn the_gap_starts_as_t_squared_as_the_caption_says() {
        let scene = run(4, 60.0);
        let s = scene.shown();
        // played from 0 to 1, one round, against t
        assert!(s.t == 1.0 && scene.n == 1 && !scene.sym && scene.chart == Chart::Time && !scene.playing);
        let c = size(bracket(s.a, s.b));
        let at = |t: f64| (gap(s.a, s.b, t, 1, false), t * t / 2.0 * c);
        let (g, line) = at(0.1);
        assert_eq!((num(g, 5), num(line, 5)), ("0.01416".to_string(), "0.01414".to_string()));
        let (g, line) = at(1.0);
        assert_eq!((num(g, 3), num(line, 3)), ("1.567".to_string(), "1.414".to_string()));
        // halve t and the gap falls to a quarter, for short times
        for t in [0.1, 0.05, 0.02] {
            let ratio = at(t).0 / at(t / 2.0).0;
            assert!((ratio - 4.0).abs() < 0.01, "{t}: {ratio}");
        }
        // the series agree in I and t(A + B), and differ first in t^2, by (t^2/2)(AB - BA)
        let t = 1e-4;
        let diff = (expm(s.a, t) * expm(s.b, t) - expm(s.a + s.b, t)) * (1.0 / (t * t));
        assert!(close(diff, bracket(s.a, s.b) * 0.5, 1e-3), "{diff:?}");
        // and the play started from t = 0
        let mut fresh = Expsum::new();
        fresh.set_up(4);
        assert!(fresh.glide.to.t == 0.0 && fresh.play_when_there && fresh.stop_at == 1.0);
    }

    /// The rates the chart's dashed lines stand for, for every pair that does not
    /// commute: 1/n and 1/n² for many rounds, t² and t³ for short times.
    #[test]
    fn the_rates_are_one_over_n_and_one_over_n_squared() {
        for k in [0, 3, 4] {
            let (a, b) = pair(k);
            for t in [0.5, 1.0] {
                for n in [16, 32] {
                    let plain = gap(a, b, t, n, false) / gap(a, b, t, 2 * n, false);
                    let sym = gap(a, b, t, n, true) / gap(a, b, t, 2 * n, true);
                    assert!((plain - 2.0).abs() < 0.01 && (sym - 4.0).abs() < 0.02, "pair {k}, t {t}, n {n}: {plain} {sym}");
                }
            }
            // for short times, the first terms: (t^2/2n)|[A, B]| and (t^3/n^2)|C|
            let (c, c3) = (size(bracket(a, b)), size(third(a, b)));
            for n in [1, 4] {
                let t = 0.01;
                let plain = gap(a, b, t, n, false) / (t * t / (2.0 * n as f64) * c);
                let sym = gap(a, b, t, n, true) / (t.powi(3) / (n * n) as f64 * c3);
                assert!((plain - 1.0).abs() < 1e-3 && (sym - 1.0).abs() < 1e-3, "pair {k}, n {n}: {plain} {sym}");
            }
        }
    }

    /// What each experiment is about lies in the part of the scene the view is fitted to:
    /// the start, both ends and every point of the zig-zag.
    #[test]
    fn every_experiment_is_in_view() {
        let (lo, hi) = ((FRAME_CENTRE - FRAME_HALF).as_dvec2(), (FRAME_CENTRE + FRAME_HALF).as_dvec2());
        for k in 0..TRIALS.len() {
            let scene = run(k, 60.0);
            let s = scene.shown();
            for n in [1, 2, 4, 8, 16, 32] {
                for sym in [false, true] {
                    let z = zig(s.a, s.b, s.x0, s.t, n, sym);
                    assert!(z.points.iter().all(|p| inside_box(*p, lo, hi)), "trial {k}, n {n}, {sym}");
                }
            }
            assert!(smooth(s.a, s.b, s.x0, s.t).iter().all(|p| inside_box(*p, lo, hi)) && inside_box(DVec2::ZERO, lo, hi), "trial {k}");
        }
    }

    /// A change of n glides the drawn zig-zag from the old to the new, smoothly, and
    /// comes exactly to rest there; a hand moving x0 meanwhile moves it with it.
    #[test]
    fn a_new_count_glides_and_comes_to_rest() {
        let mut scene = run(0, 30.0);
        assert!(!scene.zig.moving());
        let before: Vec<DVec2> = (0..=U).map(|i| scene.zig.point(i)).collect();
        scene.set_n(8);
        let mut last = before.clone();
        let mut frames = 0;
        loop {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            let now: Vec<DVec2> = (0..=U).map(|i| scene.zig.point(i)).collect();
            let step = now.iter().zip(&last).map(|(p, q)| p.distance(*q)).fold(0.0, f64::max);
            // (no point moves more than a tenth of a unit in a frame)
            assert!(step < 0.1, "frame {frames}: {step}");
            last = now;
            frames += 1;
            if !scene.zig.moving() {
                break;
            }
            assert!(frames < 300);
        }
        assert!(frames > 8);
        let target = zig(scene.shown().a, scene.shown().b, scene.shown().x0, 1.0, 8, false);
        assert!((0..=U).all(|i| scene.zig.point(i) == target.points[i]));
        // a hand on x0 in the middle of a glide carries the drawn path with it
        scene.set_n(2);
        scene.advance(1.0 / 60.0, 1.0 / 60.0);
        let drawn0 = scene.zig.point(0);
        scene.glide.now(|s| s.x0 += DVec2::new(0.3, 0.0));
        scene.advance(0.0, 0.0);
        assert!((scene.zig.point(0) - drawn0 - DVec2::new(0.3, 0.0)).length() < 1e-12 && scene.zig.moving());
    }

    /// Degenerate input: every number drawn or read out is finite, whatever A, B, t, n
    /// and x0 are.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        let big = ENTRY_MAX;
        let mats = [DMat2::ZERO, rows([big; 4]), rows([big, -big, -big, big]), rows([-big; 4]), rows([0.0, big, 0.0, 0.0]), DMat2::IDENTITY * 1e-13];
        for a in mats {
            for b in mats {
                for t in [0.0, 1e-9, 1.0, T_MAX] {
                    for x0 in [DVec2::ZERO, DVec2::splat(X0_MAX), DVec2::new(-X0_MAX, 3.0)] {
                        for (n, sym) in [(1, false), (7, true), (N_MAX, false), (N_MAX, true)] {
                            let z = zig(a, b, x0, t, n, sym);
                            assert!(z.points.iter().all(|p| p.as_vec2().is_finite()), "{a:?} {b:?} {t} {n}");
                            let g = gap(a, b, t, n, sym);
                            assert!(g.is_finite() && (g as f32).is_finite());
                            for text in [figures(g), cell(rounds(a, b, t, n, sym).x_axis.x), figures(size(bracket(a, b)))] {
                                assert!(!text.contains("NaN") && !text.contains("inf") && !text.contains('?'), "{text}");
                            }
                        }
                        assert!(smooth(a, b, x0, t).iter().all(|p| p.as_vec2().is_finite()));
                    }
                }
                for chart in [Chart::Rounds, Chart::Time] {
                    let c = curves_of(a, b, 1.0, 8, chart);
                    assert!(c.gaps.iter().flatten().all(|g| g.is_finite()));
                }
            }
        }
        assert_eq!(figures(0.0), "0");
        assert_eq!(figures(0.0441973), "0.0442");
        assert_eq!(figures(2.828), "2.83");
        assert_eq!(figures(1.567), "1.57");
    }

    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1600.0, 1000.0), (1100.0, 700.0), (1080.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (240.0, 180.0), (620.0, 480.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let (min, max) = lay.picture;
                assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max - min).min_element() > 50.0, "{w}x{h}: {lay:?}");
                if let Some(p) = &lay.panel {
                    let within = |q: Vec2| q.cmpge(p.min - Vec2::splat(0.5)).all() && q.cmple(p.max + Vec2::splat(0.5)).all();
                    assert!(p.min.cmpge(Vec2::ZERO).all() && p.max.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h}: {p:?}");
                    assert!(within(p.play.0) && within(p.play.1) && within(p.bar.0) && p.bar.1.x > p.bar.0.x + 60.0, "{w}x{h}: {p:?}");
                    if let Some(c) = p.chart {
                        assert!(within(c.0) && within(c.1) && c.1.y - c.0.y >= 70.0, "{w}x{h}: {p:?}");
                    }
                    assert!(p.min.y >= inset || lay.narrow, "{w}x{h}: under the buttons");
                    // the matrices fit their columns
                    let column = matrix_size(2, 2, CELL, p.text).x;
                    assert!(p.cells.iter().all(|c| c.x + column <= p.max.x), "{w}x{h}");
                }
            }
        }
        // on a desktop and on a phone, the chart is there
        assert!(Layout::of(Vec2::new(1600.0, 1000.0), 0.0).panel.is_some_and(|p| p.chart.is_some()));
        assert!(Layout::of(Vec2::new(390.0, 800.0), 46.0).panel.is_some_and(|p| p.chart.is_some()));
        assert!(Layout::of(Vec2::new(1100.0, 700.0), 0.0).panel.is_some_and(|p| p.chart.is_some()));
        // (the canvas of a window that size with the panels beside it)
        assert!(Layout::of(Vec2::new(580.0, 700.0), 0.0).panel.is_some_and(|p| p.chart.is_some()));
    }
}

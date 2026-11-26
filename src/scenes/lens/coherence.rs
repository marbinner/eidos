//! What averaging Jacobians keeps: the main result of Ostrow, Qian, Liu and Fiete,
//! "A Mathematical Theory of the Jacobian Lens" (2026), in a residual stream of two
//! dimensions, where every quantity in it can be seen at once.

use super::page::{width, world, write, write_foot, Page, Panel, Rect};
use super::*;
use crate::math::complex::C32;

// ----- the model ----------------------------------------------------------------------

/// One mode of the network's response: a direction of the residual stream, and what
/// becomes of a nudge along it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Mode {
    /// How strongly the nudge comes out at the last layer, at the token where it was
    /// made (the paper's e^{Tχ}: growth through the layers).
    pub strength: f32,
    /// θ: how far the response turns from one token to the next, in radians.
    pub turn: f32,
    /// The factor by which the response fades from one token to the next (e^{μ} ≤ 1).
    pub fade: f32,
    /// κ: the response has its usual sign in a share (1 + κ)/2 of contexts and the
    /// opposite sign in the rest. 1: every context agrees; 0: a coin toss.
    pub agree: f32,
    /// The standard deviation of the activations along the mode.
    pub spread: f32,
}

/// The paper's model of shared and context-specific modes (its Eq. 5), cut down to
/// two dimensions: the response at the last layer, `lag` tokens after a nudge `v`
/// to the residual stream in context `c`, is
///
/// ```text
///     K(c, lag) v = sum over modes i of  ±strength_i fade_i^lag cos(turn_i lag) (e_i . v) e_i
/// ```
///
/// where the sign is the context's (see [`Mode::agree`]) and e₁, e₂ are orthonormal.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Model {
    pub modes: [Mode; 2],
    /// The angle of the first mode's direction.
    pub tilt: f32,
    /// H: how many token lags the lens averages over.
    pub lags: usize,
    /// ω: the frequency the lens is tuned to. 0 is the plain average of the paper's
    /// Eq. 1; otherwise lag τ is weighted by e^{-iωτ} (its section 1.2).
    pub omega: f32,
}

/// The paper's three numbers for one direction (its Eq. 10).
#[derive(Clone, Copy, Debug)]
pub struct Factors {
    /// Magnitude: the average size of the response, over contexts and lags.
    pub a: f32,
    /// Lag coherence: what is left of that after averaging over lags, as a fraction.
    pub rho_tau: f32,
    /// Context coherence: what is left of that after averaging over contexts.
    pub rho_c: f32,
}

impl Factors {
    /// Their product, which is the size of what the lens returns: |J_H v|.
    pub fn lens(&self) -> f32 {
        self.a * self.rho_tau * self.rho_c
    }
}

/// What the lens picks out of the stream (see [`Table::pick`]).
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Pick {
    /// One direction above all others: the lens's first right singular vector.
    One(Vec2),
    /// Every direction alike: the two singular values are equal (to a part in a
    /// thousand), as when both modes survive alike.
    Alike,
    /// Next to nothing gets through at all.
    Nothing,
}

/// A model with everything that does not depend on the direction worked out once.
pub struct Table {
    pub axes: [Vec2; 2],
    /// The four contexts: the sign each mode has in it, and how likely it is.
    pub contexts: [([f32; 2], f32); 4],
    /// For each lag, each mode's `strength fade^lag cos(turn lag)`.
    gains: Vec<[f32; 2]>,
    /// For each lag, the lens's weight e^{-iω lag} as (cos, -sin).
    weights: Vec<(f32, f32)>,
}

impl Model {
    /// The directions e₁, e₂ of the two modes.
    pub fn axes(&self) -> [Vec2; 2] {
        let e = Vec2::from_angle(self.tilt);
        [e, e.perp()]
    }

    pub fn table(&self) -> Table {
        let likely = |i: usize, sign: f32| 0.5 * (1.0 + sign * self.modes[i].agree);
        Table {
            axes: self.axes(),
            contexts: [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]].map(|s| (s, likely(0, s[0]) * likely(1, s[1]))),
            gains: (0..self.lags).map(|lag| self.modes.map(|m| m.strength * m.fade.powi(lag as i32) * (m.turn * lag as f32).cos())).collect(),
            weights: (0..self.lags).map(|lag| ((self.omega * lag as f32).cos(), -(self.omega * lag as f32).sin())).collect(),
        }
    }

    /// What the lens multiplies mode `i` by: its strength times the lag average of
    /// `fade^lag cos(turn lag) e^{-iω lag}`. For a steady mode and the plain lens this
    /// is the paper's `e^{Tχ} f_H(γ)`.
    pub fn gain(&self, i: usize) -> C32 {
        let m = self.modes[i];
        let sum = (0..self.lags).fold(C32::ZERO, |sum, lag| sum + C32::polar(m.fade.powi(lag as i32) * (m.turn * lag as f32).cos(), -self.omega * lag as f32));
        sum.scale(m.strength / self.lags as f32)
    }

    /// The share of the activations' variance that lies along the unit direction `v`.
    pub fn variance_share(&self, v: Vec2) -> f32 {
        let (e, [m1, m2]) = (self.axes(), self.modes);
        let (v1, v2) = (m1.spread * m1.spread, m2.spread * m2.spread);
        (v1 * e[0].dot(v).powi(2) + v2 * e[1].dot(v).powi(2)) / (v1 + v2).max(1e-12)
    }

    /// What the lens makes of a token that is read out at the output along `w`: the
    /// direction j = J* w of the residual stream (the paper's Eq. 12). Each mode's part
    /// of `w` is scaled by what survives of that mode. (A tuned lens is complex; its
    /// parts are then given by their sizes.)
    pub fn token_direction(&self, w: Vec2) -> Vec2 {
        let e = self.axes();
        (0..2).fold(Vec2::ZERO, |j, i| {
            let gain = self.gain(i);
            let kept = self.modes[i].agree * if self.omega == 0.0 { gain.re } else { gain.abs() };
            j + e[i] * (kept * e[i].dot(w))
        })
    }

    /// The direction the activations vary most along (their first principal component),
    /// if there is one: not when they vary as much along both modes (to a part in a
    /// thousand), and so as much in every direction.
    pub fn principal(&self) -> Option<Vec2> {
        let e = self.axes();
        let [v1, v2] = self.modes.map(|m| m.spread * m.spread);
        if (v1 - v2).abs() <= 1e-3 * v1.max(v2) {
            None
        } else if v1 > v2 {
            Some(e[0])
        } else {
            Some(e[1])
        }
    }
}

impl Table {
    /// K(c, lag) v: the response `lag` tokens on, in the context with these signs.
    pub fn response(&self, signs: [f32; 2], lag: usize, v: Vec2) -> Vec2 {
        let g = self.gains[lag];
        self.axes[0] * (signs[0] * g[0] * self.axes[0].dot(v)) + self.axes[1] * (signs[1] * g[1] * self.axes[1].dot(v))
    }

    /// G_c v: the lens's average of the responses over lags, in one context. A pair
    /// (real part, imaginary part); the imaginary part is zero for the plain lens.
    pub fn lag_average(&self, signs: [f32; 2], v: Vec2) -> (Vec2, Vec2) {
        let (mut re, mut im) = (Vec2::ZERO, Vec2::ZERO);
        for (lag, w) in self.weights.iter().enumerate() {
            let k = self.response(signs, lag, v);
            re += k * w.0;
            im += k * w.1;
        }
        let h = self.weights.len().max(1) as f32;
        (re / h, im / h)
    }

    /// The three factors for the unit direction `v`, straight from their definitions.
    pub fn factors(&self, v: Vec2) -> Factors {
        let h = self.weights.len().max(1) as f32;
        let (mut a, mut typical) = (0.0, 0.0);
        let (mut re, mut im) = (Vec2::ZERO, Vec2::ZERO);
        for (signs, p) in self.contexts {
            if p <= 0.0 {
                continue;
            }
            a += p * (0..self.weights.len()).map(|lag| self.response(signs, lag, v).length()).sum::<f32>() / h;
            let g = self.lag_average(signs, v);
            typical += p * (g.0.length_squared() + g.1.length_squared()).sqrt();
            re += g.0 * p;
            im += g.1 * p;
        }
        let lens = (re.length_squared() + im.length_squared()).sqrt();
        Factors { a, rho_tau: if a > 1e-9 { typical / a } else { 0.0 }, rho_c: if typical > 1e-9 { lens / typical } else { 0.0 } }
    }

    /// J_H v: what the lens returns for `v` (real and imaginary parts).
    pub fn lens_of(&self, v: Vec2) -> (Vec2, Vec2) {
        self.contexts.iter().fold((Vec2::ZERO, Vec2::ZERO), |sum, (signs, p)| {
            let g = self.lag_average(*signs, v);
            (sum.0 + g.0 * *p, sum.1 + g.1 * *p)
        })
    }

    /// The lens as a matrix: J_H = A + iB.
    pub fn lens(&self) -> (Mat2, Mat2) {
        let (x, y) = (self.lens_of(Vec2::X), self.lens_of(Vec2::Y));
        (Mat2::from_cols(x.0, y.0), Mat2::from_cols(x.1, y.1))
    }

    /// Which mode (0 or 1) lies nearest the lens's first direction, if it has one.
    pub fn first_mode(&self) -> Option<usize> {
        match self.pick() {
            Pick::One(v1) => Some((self.axes[1].dot(v1).abs() > self.axes[0].dot(v1).abs()) as usize),
            _ => None,
        }
    }

    /// What the lens picks out: its first direction, unless no direction comes first.
    pub fn pick(&self) -> Pick {
        let (s1, s2, v1) = self.singular();
        // (the strongest answer to a nudge, at lag 0)
        let strongest = self.gains.first().map_or(0.0, |g| g[0].abs().max(g[1].abs()));
        if s1 <= 1e-4 * strongest {
            Pick::Nothing
        } else if s1 - s2 <= 1e-3 * s1 {
            Pick::Alike
        } else {
            Pick::One(v1)
        }
    }

    /// The lens's two singular values, and the (real) direction it responds to most:
    /// its first right singular vector. (When the two are equal that direction is
    /// rounding, not a choice: see [`Table::pick`].)
    pub fn singular(&self) -> (f32, f32, Vec2) {
        let (a, b) = self.lens();
        // |J v|² = v . (AᵀA + BᵀB) v for a real v
        let m = a.transpose() * a + b.transpose() * b;
        let (p, q, r) = (m.x_axis.x, m.x_axis.y, m.y_axis.y);
        let (mean, half) = (0.5 * (p + r), (0.25 * (p - r) * (p - r) + q * q).sqrt());
        ((mean + half).max(0.0).sqrt(), (mean - half).max(0.0).sqrt(), Vec2::from_angle(0.5 * (2.0 * q).atan2(p - r)))
    }
}

// ----- the scene ----------------------------------------------------------------------

pub const DEF: SceneDef = SceneDef {
    id: "coherence",
    title: "What averaging Jacobians keeps",
    group: super::GROUP,
    blurb: "A lens that averages a network's responses keeps what is strong, steady and agreed on.",
    make: |_| Box::new(Coherence::new()),
};

const STEADY: Mode = Mode { strength: 1.0, turn: 0.0, fade: 1.0, agree: 1.0, spread: 0.7 };

/// Where the scene starts: mode 2 is the stronger and carries more of the variance,
/// but it turns and the contexts disagree about it.
const START: Model = Model {
    modes: [Mode { strength: 0.7, spread: 0.55, ..STEADY }, Mode { strength: 1.8, turn: 0.9, fade: 1.0, agree: 0.6, spread: 1.3 }],
    tilt: 0.42,
    lags: 16,
    omega: 0.0,
};

/// Settings worth seeing: (button, what to look for, the model, token directions
/// shown). What each note says is checked by the tests.
const EXPERIMENTS: [(&str, &str, Model, bool); 6] = [
    (
        "Strong but unsteady loses",
        "Mode 2 (orange) answers a nudge more strongly than mode 1 (blue): the outer curve reaches 1.7 times as far along it. It also holds most of the variance. But its answer swings to and fro from token to token and the contexts disagree about it, so little of it survives the averaging. The lens's first direction is mode 1.",
        START,
        false,
    ),
    (
        "The paper's own example",
        "Both modes answer equally strongly: the outer curve is a circle. But mode 2 flips sign from each token to the next, and from context to context. The inner curve is mode 1 alone, although mode 2 holds 97 % of the variance. What the lens picks has nothing to do with variance.",
        Model { modes: [Mode { spread: 0.3, ..STEADY }, Mode { turn: PI, agree: 0.0, spread: 1.7, ..STEADY }], ..START },
        false,
    ),
    (
        "Only the lags differ",
        "The contexts all agree, so the middle and inner curves coincide. Mode 2 turns slowly: its answer points the same way for eight tokens, then swings round to the other side. Drag $v$ onto mode 2 and see the chain double back. Raise 'tokens averaged over' to 16 and the chain ends where it began: a longer look keeps only what holds still for longer.",
        Model { modes: [STEADY, Mode { turn: 0.2, ..STEADY }], lags: 12, ..START },
        false,
    ),
    (
        "Only the contexts differ",
        "Nothing turns, so the outer and middle curves coincide. But mode 2 has the opposite sign in a third of the contexts: along it the inner curve is pinched to a third of the middle one. Drag $v$ onto mode 2 to see the contexts pull against each other.",
        Model { modes: [STEADY, Mode { agree: 1.0 / 3.0, ..STEADY }], ..START },
        false,
    ),
    (
        "Tune the lens to the turning mode",
        r"Mode 2 makes a quarter turn per token, and the plain lens would not see it at all. This lens is tuned to that frequency ($\omega = \pi/2$): it unwinds a quarter turn per token before adding up. Now the steady mode is the one that cancels, and the lens's first direction is mode 2.",
        Model { modes: [STEADY, Mode { turn: PI / 2.0, ..STEADY }], omega: PI / 2.0, ..START },
        false,
    ),
    (
        "What the lens makes of tokens",
        "The dashed green arrows are the directions three tokens are read out along at the output. The solid ones are what the lens turns them into: directions to look along in the residual stream. All three are pulled onto the mode that survives. Whatever the token, the lens reads the stream along nearly the same line.",
        START,
        true,
    ),
];

/// The unembedding directions of three made-up tokens (angles in the plane).
const TOKENS: [f32; 3] = [0.15, 1.2, 2.9];
const TOKEN_COLORS: [Color; 3] = [Color::hex(0x2EE6A6), Color::hex(0xA3E635), Color::hex(0x5EEAD4)];
const MODE_COLORS: [Color; 2] = [pal::BLUE, pal::ORANGE];
/// How many directions the curves are drawn through.
const AROUND: usize = 240;
/// Radii of the picture, in units of the longest response: where the handle for v
/// and the names of the modes go.
const HANDLE: f32 = 1.13;
const NAMES: f32 = 1.31;

/// How to anchor a label that sits at the far end of a line from the centre in the
/// direction `d`, so that its text runs on outwards and not back over the picture.
fn outward(d: Vec2) -> Align2 {
    if d.x > 0.35 {
        Align2::LEFT_CENTER
    } else if d.x < -0.35 {
        Align2::RIGHT_CENTER
    } else if d.y > 0.0 {
        Align2::CENTER_BOTTOM
    } else {
        Align2::CENTER_TOP
    }
}

/// A name at the point `p` of a picture, at the far end of a line from the centre in
/// the direction `d`: the text runs outwards, and is moved sideways if it would
/// otherwise leave the panel `area`.
fn name(f: &mut Frame, area: Rect, p: Vec2, d: Vec2, text: &str, size: f32, color: Color) {
    let align = outward(d);
    let width = width(text, size) * 1.1;
    let (left, right) = if align == Align2::LEFT_CENTER {
        (p.x, p.x + width)
    } else if align == Align2::RIGHT_CENTER {
        (p.x - width, p.x)
    } else {
        (p.x - 0.5 * width, p.x + 0.5 * width)
    };
    let shift = (area.min.x + 2.0 - left).max(0.0) - (right - (area.max.x - 2.0)).max(0.0);
    f.text(Vec2::new(p.x + shift, p.y), text, size, color, align);
}

struct Coherence {
    m: Model,
    /// The angle of the direction v being looked at.
    probe: f32,
    cloud: bool,
    tokens: bool,
    /// For each direction round the circle: a, a ρτ, a ρτ ρc.
    curves: Vec<[f32; 3]>,
    /// The longest response, a(v), over all directions: the scale of the picture.
    longest: f32,
    /// The model with everything that does not depend on v worked out, and the
    /// model it (and the curves) were worked out for.
    table: Table,
    drawn: Option<Model>,
    /// Standard normal pairs: the activations, before they are spread along the modes.
    normals: Vec<Vec2>,
    page: Page,
}

impl Coherence {
    fn new() -> Self {
        let mut rng = Rng::new(12);
        Coherence { m: START, probe: 1.25, cloud: true, tokens: false, curves: Vec::new(), longest: 1.0, table: START.table(), drawn: None, normals: (0..260).map(|_| Vec2::new(rng.normal(), rng.normal())).collect(), page: Page::new() }
    }

    fn v(&self) -> Vec2 {
        Vec2::from_angle(self.probe)
    }

    /// Bring the table and the curves up to date with the model (after any change
    /// to it: at the start of a frame, and in the panel once its controls have run).
    fn sync(&mut self) {
        if self.drawn == Some(self.m) {
            return;
        }
        self.table = self.m.table();
        self.curves = (0..AROUND)
            .map(|k| {
                let fac = self.table.factors(Vec2::from_angle(k as f32 / AROUND as f32 * TAU));
                [fac.a, fac.a * fac.rho_tau, fac.lens()]
            })
            .collect();
        self.longest = self.curves.iter().fold(0.0f32, |m, c| m.max(c[0])).max(1e-4);
        self.drawn = Some(self.m);
    }

    // ----- the main picture: every direction at once

    fn picture(&mut self, f: &mut Frame, r: Rect) {
        let small = r.w() < 430.0;
        let table = &self.table;
        let mut head = Panel::open(f, r, 12.0);
        let title = if small { "nudge the stream along $v$: how strongly the output answers, for every direction $v$" } else { "nudge the residual stream along a direction $v$: how strongly the output answers, for every direction $v$ at once (distance from the centre)" };
        head.text(f, title, 12.0, pal::TEXT, 3.0);
        let mut entries = vec![(pal::TEXT.a(0.75), "one nudge, on average: $a$"), (pal::YELLOW, r"after averaging over lags: $a \rho_\tau$"), (pal::PINK, r"and over contexts: $a \rho_\tau \rho_c = |J v|$, what the lens sees")];
        if self.cloud {
            entries.push((pal::PURPLE, "dots: the activations"));
        }
        head.key(f, &entries, 2.0);
        let inner = head.inner;
        let area = Rect::new(inner.min.x, head.y, inner.max.x, inner.max.y);
        let unit = 0.5 * area.w().min(area.h()) / (NAMES + 0.12);
        let o = world(area.center());
        let at = |q: Vec2| o + q * unit;
        let scale = 1.0 / self.longest.max(1e-6);

        // rings at round sizes of response
        let step = nice_step(self.longest / 2.6);
        let mut ring = step;
        while ring <= self.longest * 1.001 {
            f.circle(o.extend(0.0), Vec3::Z, ring * scale * unit, 1.0, pal::FAINT.a(0.5));
            // (numbered half-way between two of the modes' lines, clear of both)
            f.text(at(Vec2::from_angle(self.m.tilt - PI / 4.0) * ring * scale), fmt_tick(ring, step), 10.0, pal::DIM, Align2::LEFT_TOP);
            ring += step;
        }
        // the activations, and the direction they vary most along
        if self.cloud {
            let [m1, m2] = self.m.modes;
            let grow = 0.4 / m1.spread.max(m2.spread).max(0.3);
            for z in &self.normals {
                let p = (table.axes[0] * (m1.spread * z.x) + table.axes[1] * (m2.spread * z.y)) * grow;
                if p.length() < HANDLE - 0.05 {
                    f.dot(at(p), 4.0, pal::PURPLE.a(0.8));
                }
            }
            // (when they vary as much every way, no direction is drawn as theirs)
            if let Some(pc) = self.m.principal() {
                f.dashed(at(-pc * (NAMES - 0.1)), at(pc * (HANDLE - 0.08)), 1.5, pal::PURPLE);
                name(f, area, at(-pc * (NAMES - 0.06)), -pc, "most variance (PC 1)", 11.0, pal::PURPLE);
            }
        }
        // the two modes
        for i in 0..2 {
            let e = table.axes[i];
            f.line(at(-e * 1.04), at(e * (NAMES - 0.1)), 1.2, MODE_COLORS[i].a(0.55));
            name(f, area, at(e * (NAMES - 0.06)), e, &format!("mode {}", i + 1), 12.0, MODE_COLORS[i]);
        }
        // the three curves
        let around = |k: usize| Vec2::from_angle(k as f32 / AROUND as f32 * TAU);
        for (which, width, color) in [(0, 1.8, pal::TEXT.a(0.75)), (1, 2.2, pal::YELLOW), (2, 3.0, pal::PINK)] {
            let pts: Vec<Vec3> = (0..AROUND).map(|k| at(around(k) * (self.curves[k][which] * scale)).extend(0.0)).collect();
            f.dl.lines.closed(&pts, width, color);
        }
        // the direction the lens responds to most, if one direction comes first
        let (s1, _, _) = table.singular();
        if let (Pick::One(v1), true) = (table.pick(), s1 * scale > 0.02) {
            f.line(at(-v1 * (s1 * scale)), at(v1 * (s1 * scale)), 1.6, pal::PINK.a(0.8));
            // named beyond the far end of the line, a little to one side of the mode's own line
            let side = v1.perp() * (9.0 / unit) * if v1.perp().y > 0.0 { 1.0 } else { -1.0 };
            name(f, area, at(-v1 * (s1 * scale + 0.05) + side), if v1.x.abs() > 0.35 { -v1 } else { Vec2::X }, "the lens's first direction", 11.0, pal::PINK);
        }
        // what the lens makes of three tokens' output directions
        if self.tokens {
            for (q, angle) in TOKENS.iter().enumerate() {
                let w = Vec2::from_angle(*angle);
                let j = self.m.token_direction(w);
                f.dashed(o, at(w * 0.98), 1.4, TOKEN_COLORS[q].a(0.8));
                name(f, area, at(w * 1.02), w, &format!("$w_{}$", q + 1), 11.0, TOKEN_COLORS[q]);
                if j.length() * scale > 0.01 {
                    f.arrow(o, at(j * scale), 3.0, TOKEN_COLORS[q]);
                    f.text(at(j * scale + j.normalize_or_zero() * 0.07), format!("$j_{}$", q + 1), 11.0, TOKEN_COLORS[q], Align2::CENTER_CENTER);
                }
            }
        }
        // the direction being looked at: drag it round
        let v = self.v();
        let k = ((self.probe.rem_euclid(TAU) / TAU * AROUND as f32).round() as usize) % AROUND;
        f.line(o, at(v * HANDLE), 1.4, pal::WHITE.a(0.7));
        for (which, color) in [(0, pal::TEXT), (1, pal::YELLOW), (2, pal::PINK)] {
            f.dot(at(v * (self.curves[k][which] * scale)), 8.0, color);
        }
        let mut handle = at(v * HANDLE);
        if f.drag2(1, &mut handle, pal::WHITE) {
            let d = handle - o;
            if d.length() > 1e-3 {
                self.probe = d.y.atan2(d.x);
            }
        }
        name(f, area, at(v * (HANDLE + 0.06)), v, "$v$ (drag)", 11.5, pal::WHITE);
    }

    // ----- the three side pictures, all for the direction v

    /// The responses at lags 0, 1, 2, ... laid tip to tail: what averaging over lags does.
    fn chain(&self, f: &mut Frame, r: Rect) {
        let (small, table, v) = (r.w() < 250.0, &self.table, self.v());
        let tuned = self.m.omega != 0.0;
        let mut head = Panel::open(f, r, 11.0);
        head.text(f, "one context, lag by lag", 12.0, pal::TEXT, 1.0);
        let caption = match (small, tuned) {
            (true, false) => "the answers at lags 0, 1, 2, … laid tip to tail",
            (false, false) => r"the output's answers $K(c, \tau) v$ at lags $\tau = 0, 1, 2, \ldots$ after a nudge along $v$, laid tip to tail",
            (true, true) => r"the answers, each first turned back by $\omega\tau$ (bold: $\cos$ part, faint: $\sin$ part)",
            (false, true) => r"the answers $K(c, \tau) v$, each first turned back by $\omega\tau$ as the tuned lens does (bold: $\cos$ part, faint: $\sin$ part)",
        };
        head.text(f, caption, 10.5, pal::DIM, 2.0);
        let (inner, y) = (head.inner, head.y);

        // the chains: running totals of the (weighted) responses in the context where both modes have their usual sign
        let lags = self.m.lags;
        let (mut along, mut across) = (vec![Vec2::ZERO], vec![Vec2::ZERO]);
        let mut walked = 0.0;
        for lag in 0..lags {
            let (k, w) = (table.response([1.0, 1.0], lag, v), table.weights[lag]);
            walked += k.length();
            along.push(along[lag] + k * w.0);
            across.push(across[lag] + k * w.1);
        }
        let got = (along[lags].length_squared() + across[lags].length_squared()).sqrt();
        let ratio = if walked > 1e-9 { got / walked } else { 0.0 };
        let note = if small { format!(r"walked ${}$, got ${}$: $\rho_\tau = {:.2}$", fmt(walked), fmt(got), ratio) } else { format!(r"walked ${}$, got ${}$ from the start: lag coherence $\rho_\tau = {:.2}$", fmt(walked), fmt(got), ratio) };
        let foot = write_foot(f, inner, 0.0, &note, 11.0, pal::YELLOW) + 2.0;
        let area = Rect::new(inner.min.x, y, inner.max.x, inner.max.y - foot - 2.0).shrink(8.0);

        // fit everything walked through into the area
        let none: &[Vec2] = &[];
        let points = along.iter().chain(if tuned { across.iter() } else { none.iter() });
        let (mut lo, mut hi) = (Vec2::ZERO, Vec2::ZERO);
        for p in points {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let extent = (hi - lo).max(Vec2::splat(1e-6));
        let unit = (area.w() / extent.x).min(area.h() / extent.y);
        let o = world(area.center()) - (lo + hi) * 0.5 * unit;
        let at = |p: Vec2| o + p * unit;
        for i in 0..2 {
            let reach = table.axes[i] * (0.5 * extent.max_element());
            f.line(at(-reach * 0.25), at(reach * 0.25), 1.0, MODE_COLORS[i].a(0.45));
        }
        let fade = |lag: usize| 1.0 - 0.6 * lag as f32 / lags.max(1) as f32;
        if tuned {
            for lag in 0..lags {
                f.line(at(across[lag]), at(across[lag + 1]), 1.2, pal::TEXT.a(0.35 * fade(lag)));
            }
            f.arrow(at(Vec2::ZERO), at(across[lags]), 1.6, pal::YELLOW.a(0.55));
        }
        for lag in 0..lags {
            f.line(at(along[lag]), at(along[lag + 1]), 2.0, pal::TEXT.a(0.85 * fade(lag)));
            f.dot(at(along[lag + 1]), 3.5, pal::TEXT.a(fade(lag)));
        }
        f.arrow(at(Vec2::ZERO), at(along[lags]), 2.6, pal::YELLOW);
        f.dot(at(Vec2::ZERO), 6.0, pal::WHITE);
    }

    /// The lag averages in each context, and their mean: what averaging over contexts does.
    fn contexts(&self, f: &mut Frame, r: Rect) {
        let (small, table, v) = (r.w() < 250.0, &self.table, self.v());
        let mut head = Panel::open(f, r, 11.0);
        head.text(f, "context by context", 12.0, pal::TEXT, 1.0);
        let caption = if small { "each context's lag average (thicker: likelier), and their mean (pink)" } else { "the lag average $G v$ in each context (yellow; thicker: likelier), and the mean of them all (pink)" };
        head.text(f, caption, 10.5, pal::DIM, 2.0);
        let (inner, y) = (head.inner, head.y);

        // (a tuned lens gives complex averages: their sizes are drawn, mode by mode,
        // which leaves the lengths and the cancelling between contexts as they are)
        let size = |i: usize| if self.m.omega == 0.0 { self.m.gain(i).re } else { self.m.gain(i).abs() };
        let average = |signs: [f32; 2]| table.axes[0] * (signs[0] * size(0) * table.axes[0].dot(v)) + table.axes[1] * (signs[1] * size(1) * table.axes[1].dot(v));
        let mean = table.contexts.iter().fold(Vec2::ZERO, |sum, (signs, p)| sum + average(*signs) * *p);
        let typical = average([1.0, 1.0]).length();
        let ratio = if typical > 1e-9 { mean.length() / typical } else { 0.0 };
        let note = if small { format!(r"each ${}$, their mean ${}$: $\rho_c = {:.2}$", fmt(typical), fmt(mean.length()), ratio) } else { format!(r"each is ${}$ long, their mean ${}$: context coherence $\rho_c = {:.2}$", fmt(typical), fmt(mean.length()), ratio) };
        let foot = write_foot(f, inner, 0.0, &note, 11.0, pal::PINK) + 2.0;
        let area = Rect::new(inner.min.x, y, inner.max.x, inner.max.y - foot - 2.0).shrink(6.0);

        let o = world(area.center());
        let unit = 0.5 * area.w().min(area.h()) / typical.max(1e-6);
        for i in 0..2 {
            let reach = table.axes[i] * (0.5 * area.w().min(area.h()));
            f.line(o - reach, o + reach, 1.0, MODE_COLORS[i].a(0.4));
        }
        for (signs, p) in table.contexts {
            if p > 0.002 {
                f.arrow(o, o + average(signs) * unit, 1.2 + 3.6 * p, pal::YELLOW.a(0.4 + 0.6 * p));
            }
        }
        if mean.length() * unit > 2.0 {
            f.arrow(o, o + mean * unit, 3.0, pal::PINK);
        } else {
            f.dot(o, 7.0, pal::PINK);
        }
    }

    /// The three sizes along v as bars: the radii of the three curves there.
    fn bars(&self, f: &mut Frame, r: Rect) {
        let fac = self.table.factors(self.v());
        let mut head = Panel::open(f, r, 11.0);
        head.text(f, "along the chosen direction $v$", 12.0, pal::TEXT, 4.0);
        let (inner, mut y) = (head.inner, head.y);
        let rows = [
            (format!("one nudge, on average: $a = {}$", fmt(fac.a)), fac.a, pal::TEXT.a(0.75)),
            (format!(r"$\times$ lag coherence $\rho_\tau = {:.2}$, leaving ${}$", fac.rho_tau, fmt(fac.a * fac.rho_tau)), fac.a * fac.rho_tau, pal::YELLOW),
            (format!(r"$\times$ context coherence $\rho_c = {:.2}$, leaving $|J v| = {}$", fac.rho_c, fmt(fac.lens())), fac.lens(), pal::PINK),
        ];
        let room = (inner.max.y - y) / 3.0;
        for (text, value, color) in rows {
            let used = write(f, Vec2::new(inner.min.x, y), inner.w(), &text, 11.0, color.a(1.0));
            let bar = (room - used - 7.0).clamp(4.0, 12.0);
            let top = y + used + 1.0;
            Rect::new(inner.min.x, top, inner.max.x, top + bar).fill(f, pal::FAINT.a(0.25));
            let length = inner.w() * (value / self.longest.max(1e-6)).clamp(0.0, 1.0);
            if length > 0.5 {
                Rect::new(inner.min.x, top, inner.min.x + length, top + bar).fill(f, color);
            }
            y += room;
        }
    }
}

fn mode_controls(ui: &mut egui::Ui, m: &mut Mode) {
    ui::slider(ui, "strength", &mut m.strength, 0.0..=2.0);
    ui::slider(ui, r"turn per token $\theta$", &mut m.turn, 0.0..=PI);
    let turn = if m.turn < 0.01 {
        "The answer points the same way at every later token.".to_string()
    } else {
        format!(r"$\theta = {:.2}\pi$: the answer swings to and fro, once every {:.1} tokens.", m.turn / PI, TAU / m.turn)
    };
    ui::note(ui, &turn);
    ui::slider(ui, "fade per token", &mut m.fade, 0.5..=1.0);
    ui::slider(ui, r"agreement between contexts $\kappa$", &mut m.agree, 0.0..=1.0);
    ui::note(ui, &format!("The usual sign in {:.0} % of contexts, the opposite in the rest.", 50.0 * (1.0 + m.agree)));
    ui::slider(ui, "spread of the activations", &mut m.spread, 0.1..=2.0);
}

impl Scene for Coherence {
    fn view(&self) -> View {
        self.page.view()
    }

    fn take_view(&mut self) -> Option<View> {
        self.page.take_view()
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    /// Nothing moves here unless something is dragged.
    fn animated(&self) -> bool {
        false
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A Jacobian lens reads a network's middle layers by asking how the output answers a small nudge there. It averages the answer over the tokens that follow the nudge (the lags) and over many different texts (the contexts). This is a network small enough to see what that averaging keeps: its residual stream has just two directions, called modes, and each answers a nudge in its own way. Drag $v$ round the picture.");
        ui::section(ui, "Things to try");
        for (label, look_for, model, tokens) in EXPERIMENTS {
            if ui::experiment(ui, label, look_for) {
                self.m = model;
                self.tokens = tokens;
            }
        }
        ui::section(ui, "Mode 1 (blue)");
        ui.push_id("mode 1", |ui| mode_controls(ui, &mut self.m.modes[0]));
        ui::section(ui, "Mode 2 (orange)");
        ui.push_id("mode 2", |ui| mode_controls(ui, &mut self.m.modes[1]));
        ui::section(ui, "The lens");
        ui::slider_int(ui, "tokens averaged over, $H$", &mut self.m.lags, 1..=64);
        ui::slider(ui, r"tuned to frequency $\omega$", &mut self.m.omega, 0.0..=PI);
        ui::note(ui, r"At $\omega = 0$ this is the plain average. Otherwise the answer at lag $\tau$ is turned back by $\omega\tau$ before it is added, which picks out answers that turn at that rate (the paper's $\omega$-lens).");
        ui::section(ui, "The picture");
        let mut tilt = self.m.tilt.to_degrees();
        if ui::slider(ui, "tilt of the modes (degrees)", &mut tilt, 0.0..=180.0) {
            self.m.tilt = tilt.to_radians();
        }
        let mut probe = self.probe.rem_euclid(TAU).to_degrees();
        if ui::slider(ui, "direction $v$ (degrees)", &mut probe, 0.0..=360.0) {
            self.probe = probe.to_radians();
        }
        ui.checkbox(&mut self.cloud, "the activations and their main direction");
        ui.checkbox(&mut self.tokens, "what the lens makes of three tokens");
        if self.tokens {
            ui::note(ui, r"A token $q$ is read out at the output along a direction $w$ (dashed). The lens turns it into a direction of the residual stream, $j = J^\top w$ (solid): each mode's part of $w$, scaled by what survives of that mode.");
        }
        ui::section(ui, "What the words mean");
        ui::term(ui, r"Lag $\tau$", "how many tokens after the nudge the output is looked at.");
        ui::term(ui, "Context $c$", "the text the network is reading when it is nudged. The same nudge can be answered differently in different texts.");
        ui::term(ui, r"Jacobian $K(c, \tau)$", r"the table that turns a small nudge $v$ into the output's answer $K v$, in context $c$ at lag $\tau$.");
        ui::term(ui, "The lens $J$", "the average of $K$ over lags and contexts (the paper's $J_H$, for $H$ lags). Its singular values rank the directions.");
        ui::term(ui, "Magnitude $a$", "the size of the answer to one nudge, on average.");
        ui::term(ui, r"Lag coherence $\rho_\tau$ (rho-tau)", "the share of that left after averaging over lags: 1 if the answer points the same way at every lag.");
        ui::term(ui, r"Context coherence $\rho_c$", "the share of that left after averaging over contexts: 1 if every context answers alike.");
        ui::note(ui, "The two coherences are each the share left of what comes before them, so the three multiply to the size of what the lens returns, for every direction $v$:");
        ui::equation(ui, r"|J v| = a \cdot \rho_\tau \cdot \rho_c");
        ui::note(ui, "The lens's largest singular value is the largest that product gets.");
        ui::section(ui, "Colours");
        ui::key(ui, &[
            (pal::TEXT.a(0.75), "the output's answer to one nudge, on average ($a$)"),
            (pal::YELLOW, "what is left of it after averaging over lags"),
            (pal::PINK, "what is left after averaging over contexts too: what the lens sees"),
            (MODE_COLORS[0], "mode 1"),
            (MODE_COLORS[1], "mode 2"),
            (pal::PURPLE, "the activations, and the direction they vary most along"),
            (TOKEN_COLORS[0], "tokens: read-out directions (dashed) and what the lens makes of them"),
        ]);
        ui::section(ui, "Numbers");
        self.sync();
        let table = &self.table;
        let fac = table.factors(self.v());
        ui::readout(ui, "magnitude $a$, along $v$", format!("${}$", fmt(fac.a)));
        ui::readout(ui, r"lag coherence $\rho_\tau$", format!("${:.3}$", fac.rho_tau));
        ui::readout(ui, r"context coherence $\rho_c$", format!("${:.3}$", fac.rho_c));
        ui::readout(ui, "their product, $|J v|$", format!("${}$", fmt(fac.lens())));
        let (s1, s2, _) = table.singular();
        ui::readout(ui, "the lens's singular values", format!("${}, {}$", fmt(s1), fmt(s2)));
        match (table.pick(), table.first_mode()) {
            (Pick::One(v1), Some(first)) => {
                ui::readout_colored(ui, "the lens's first direction", format!("mode {}", first + 1), MODE_COLORS[first]);
                ui::readout(ui, "share of the variance along it", format!("{:.0} %", 100.0 * self.m.variance_share(v1)));
            }
            (pick, _) => {
                ui::readout(ui, "the lens's first direction", "none");
                ui::note(ui, if pick == Pick::Nothing { "Next to nothing survives the averaging." } else { "Both modes survive alike, so no single direction comes first." });
            }
        }
        match self.m.principal() {
            Some(pc) => ui::readout(ui, "share along the main direction", format!("{:.0} %", 100.0 * self.m.variance_share(pc))),
            None => {
                ui::readout(ui, "the activations' main direction", "none");
                ui::note(ui, "They vary as much along both modes, and so as much in every direction.");
            }
        }
        ui::section(ui, "In the paper");
        ui::note(ui, r#"This is section 1 of Ostrow, Qian, Liu and Fiete, "A Mathematical Theory of the Jacobian Lens" (2026), in two dimensions. The modes are its Eq. 5; $a$, $\rho_\tau$ and $\rho_c$ are Eq. 10; their product and the largest singular value are Eq. 11; the example with the variance is Corollary 1.1; the tokens are Eq. 12; the tuned lens is section 1.2. Here the modes are at right angles, and a mode's answer swings to and fro ($\cos \theta\tau$) where the paper lets it turn in the complex plane."#);
        ui::see_also(ui, &[("lyapunov", "Lyapunov exponents"), ("belief", "The Jacobian lens on a belief filter"), ("transformer", "A transformer, opened up")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        self.page.fit(f);
        self.sync();

        let (full, size, gap) = (self.page.usable(), self.page.size, 5.0);
        if size.x >= 720.0 && size.x >= 0.9 * size.y {
            // the picture on the left, the three side pictures down the right
            let column = (0.31 * full.w()).clamp(250.0, 330.0);
            let split = full.max.x - column;
            self.picture(f, Rect::new(full.min.x, full.min.y, split - gap, full.max.y));
            let bars = 152.0;
            let half = full.min.y + bars + (full.h() - bars) * 0.5;
            self.bars(f, Rect::new(split + gap, full.min.y, full.max.x, full.min.y + bars - gap));
            self.chain(f, Rect::new(split + gap, full.min.y + bars + gap, full.max.x, half - gap));
            self.contexts(f, Rect::new(split + gap, half + gap, full.max.x, full.max.y));
        } else if size.x >= 560.0 {
            // the picture on top, the side pictures in a row underneath
            let below = (0.36 * full.h()).clamp(230.0, 320.0);
            let split = full.max.y - below;
            self.picture(f, Rect::new(full.min.x, full.min.y, full.max.x, split - gap));
            let third = full.w() / 3.0;
            self.bars(f, Rect::new(full.min.x, split + gap, full.min.x + third - gap, full.max.y));
            self.chain(f, Rect::new(full.min.x + third + gap, split + gap, full.min.x + 2.0 * third - gap, full.max.y));
            self.contexts(f, Rect::new(full.min.x + 2.0 * third + gap, split + gap, full.max.x, full.max.y));
        } else {
            // a narrow screen: the bars under the picture, then the other two side by side
            let (bars, pair) = (118.0, (0.27 * full.h()).clamp(170.0, 240.0));
            let split = full.max.y - bars - pair;
            self.picture(f, Rect::new(full.min.x, full.min.y, full.max.x, split - gap));
            self.bars(f, Rect::new(full.min.x, split + gap, full.max.x, split + bars - gap));
            let mid = full.center().x;
            self.chain(f, Rect::new(full.min.x, split + bars + gap, mid - gap, full.max.y));
            self.contexts(f, Rect::new(mid + gap, split + bars + gap, full.max.x, full.max.y));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_model(rng: &mut Rng) -> Model {
        let mut mode = || Mode { strength: rng.range(0.1, 2.0), turn: rng.range(0.0, PI), fade: rng.range(0.6, 1.0), agree: rng.f32(), spread: rng.range(0.2, 2.0) };
        let modes = [mode(), mode()];
        Model { modes, tilt: rng.range(0.0, PI), lags: 1 + rng.below(40), omega: if rng.bool(0.5) { 0.0 } else { rng.range(0.0, PI) } }
    }

    fn length(v: (Vec2, Vec2)) -> f32 {
        (v.0.length_squared() + v.1.length_squared()).sqrt()
    }

    /// The three factors multiply to what the lens returns (the step before the paper's Eq. 11).
    #[test]
    fn the_three_factors_multiply_to_the_size_of_the_lens() {
        let mut rng = Rng::new(1);
        for _ in 0..200 {
            let table = random_model(&mut rng).table();
            let v = Vec2::from_angle(rng.range(0.0, TAU));
            let fac = table.factors(v);
            assert!((0.0..=1.0001).contains(&fac.rho_tau) && (0.0..=1.0001).contains(&fac.rho_c), "{fac:?}");
            let lens = length(table.lens_of(v));
            assert!((fac.lens() - lens).abs() < 1e-4 * (1.0 + lens), "{} vs {lens}", fac.lens());
            // and the matrix says the same as the averages it was built from
            let (a, b) = table.lens();
            assert!((length((a * v, b * v)) - lens).abs() < 1e-4);
        }
    }

    /// The paper's Eq. 11: the lens's largest singular value is the largest the
    /// product gets over all directions, and it gets there along the first singular
    /// direction.
    #[test]
    fn the_largest_singular_value_is_the_largest_product() {
        let mut rng = Rng::new(2);
        for _ in 0..60 {
            let table = random_model(&mut rng).table();
            let best = (0..3600).map(|k| table.factors(Vec2::from_angle(k as f32 * TAU / 3600.0)).lens()).fold(0.0, f32::max);
            let (s1, s2, v1) = table.singular();
            assert!((s1 - best).abs() < 2e-4 * (1.0 + best), "{s1} vs {best}");
            assert!((table.factors(v1).lens() - s1).abs() < 2e-4 * (1.0 + s1));
            assert!((table.factors(v1.perp()).lens() - s2).abs() < 2e-4 * (1.0 + s1));
        }
    }

    /// With orthonormal modes the singular values are each mode's gain times its
    /// agreement, and a steady mode's gain is the paper's f_H.
    #[test]
    fn each_mode_survives_by_its_gain_times_its_agreement() {
        let mut rng = Rng::new(3);
        for _ in 0..60 {
            let m = random_model(&mut rng);
            let (s1, s2, _) = m.table().singular();
            let mut want = [0, 1].map(|i| m.modes[i].agree * m.gain(i).abs());
            want.sort_by(|a, b| b.total_cmp(a));
            assert!((s1 - want[0]).abs() < 1e-4 && (s2 - want[1]).abs() < 1e-4, "{s1} {s2} vs {want:?}");
        }
        let m = Model { modes: [Mode { strength: 1.3, fade: 0.9, ..STEADY }, STEADY], tilt: 0.3, lags: 12, omega: 0.0 };
        let f_h = (1.0 - 0.9f32.powi(12)) / (12.0 * (1.0 - 0.9));
        assert!((m.gain(0).re - 1.3 * f_h).abs() < 1e-5 && m.gain(0).im.abs() < 1e-6);
    }

    /// The paper's Corollary 1.1 (proved in its appendix D): the lens keeps mode 1 alone, whatever
    /// share of the variance mode 1 has.
    #[test]
    fn the_papers_example_keeps_one_mode_whatever_the_variance() {
        let base = EXPERIMENTS[1].2;
        for spreads in [(0.3, 1.7), (1.7, 0.3), (0.05, 2.0)] {
            let mut m = base;
            (m.modes[0].spread, m.modes[1].spread) = spreads;
            let table = m.table();
            let e = table.axes;
            let (a, b) = table.lens();
            // J = s e1 e1ᵀ exactly, with s = 1
            let want = Mat2::from_cols(e[0] * e[0].x, e[0] * e[0].y);
            assert!((a - want).to_cols_array().iter().all(|d| d.abs() < 1e-5) && b.to_cols_array().iter().all(|d| d.abs() < 1e-6));
            let (s1, s2, v1) = table.singular();
            assert!((s1 - 1.0).abs() < 1e-5 && s2 < 1e-5 && v1.dot(e[0]).abs() > 0.9999);
            let share = m.variance_share(v1);
            let want = spreads.0 * spreads.0 / (spreads.0 * spreads.0 + spreads.1 * spreads.1);
            assert!((share - want).abs() < 1e-4);
            // the answer to one nudge is as strong in every direction
            let sizes: Vec<f32> = (0..12).map(|k| table.factors(Vec2::from_angle(k as f32 * 0.5)).a).collect();
            assert!(sizes.iter().all(|a| (a - 1.0).abs() < 1e-5), "{sizes:?}");
        }
        assert!((base.variance_share(base.table().axes[1]) - 0.97).abs() < 0.005);
    }

    /// The ω-lens answers most to a mode that turns at its own frequency.
    #[test]
    fn a_tuned_lens_responds_most_to_a_mode_turning_at_its_frequency() {
        for turn in [0.4, 1.0, PI / 2.0, 2.2] {
            let gain = |omega: f32| Model { modes: [Mode { turn, ..STEADY }, STEADY], tilt: 0.0, lags: 48, omega }.gain(0).abs();
            let best = (0..=300).map(|k| k as f32 * PI / 300.0).max_by(|a, b| gain(*a).total_cmp(&gain(*b))).unwrap();
            assert!((best - turn).abs() < 0.06, "turn {turn}: best at {best}");
            // it catches half of a to-and-fro: the half that turns its way
            assert!((gain(turn) - 0.5).abs() < 0.06, "turn {turn}: gain {}", gain(turn));
        }
        // a mode that flips at every token is caught whole at ω = π
        let m = Model { modes: [Mode { turn: PI, ..STEADY }, STEADY], tilt: 0.0, lags: 16, omega: PI };
        assert!((m.gain(0).abs() - 1.0).abs() < 1e-4 && m.gain(1).abs() < 1e-4);
    }

    /// What the notes of the "things to try" say, held to account.
    #[test]
    fn the_experiments_show_what_their_notes_say() {
        let look = |k: usize| {
            let m = EXPERIMENTS[k].2;
            let table = m.table();
            let (s1, s2, _) = table.singular();
            let first = table.first_mode();
            (m, table, s1, s2, first)
        };
        // strong but unsteady loses
        let (m, table, s1, s2, first) = look(0);
        let e = table.axes;
        // "1.7 times as far": 1.8 times the average of |cos 0.9τ| over 16 lags, against 0.7
        let ratio = table.factors(e[1]).a / table.factors(e[0]).a;
        assert!((ratio - 1.7).abs() < 0.05, "{ratio}");
        assert!(m.variance_share(e[1]) > 0.8);
        assert!(first == Some(0) && s1 > 4.0 * s2, "{s1} {s2}");
        // the paper's example: see the test above; here, that the lens picks mode 1
        assert_eq!(look(1).4, Some(0));
        // only the lags differ
        let (m, table, _, _, first) = look(2);
        let e = table.axes;
        for k in 0..24 {
            let fac = table.factors(Vec2::from_angle(k as f32 * 0.3));
            assert!((fac.rho_c - 1.0).abs() < 1e-4);
        }
        let along = table.factors(e[1]);
        assert!(first == Some(0) && along.rho_tau < 0.75 && along.rho_tau > 0.5, "{along:?}");
        // the same way for eight tokens (lags 0 to 7), then the other
        assert!((0..12).all(|lag| (table.response([1.0, 1.0], lag, e[1]).dot(e[1]) > 0.0) == (lag < 8)));
        // a longer look keeps less of it: at 16 lags the chain is nearly back at its start
        let longer = Model { lags: 16, ..m }.table().factors(e[1]);
        assert!(longer.rho_tau < 0.1, "{} then {}", along.rho_tau, longer.rho_tau);
        // only the contexts differ
        let (_, table, _, _, first) = look(3);
        let e = table.axes;
        for k in 0..24 {
            assert!((table.factors(Vec2::from_angle(k as f32 * 0.3)).rho_tau - 1.0).abs() < 1e-4);
        }
        assert!(first == Some(0) && (table.factors(e[1]).rho_c - 1.0 / 3.0).abs() < 1e-4);
        // the tuned lens picks the turning mode, and the plain one would not see it
        let (m, table, s1, s2, first) = look(4);
        assert!(first == Some(1) && (s1 - 0.5).abs() < 1e-4 && s2 < 1e-3, "{s1} {s2}");
        let plain = Model { omega: 0.0, ..m }.table();
        assert!(plain.factors(table.axes[1]).lens() < 1e-3 && (plain.singular().0 - 1.0).abs() < 1e-4);
        // tokens: every lens direction lies along the surviving mode, near enough
        let (_, table, ..) = look(5);
        let (a, _) = table.lens();
        for angle in TOKENS {
            let j = EXPERIMENTS[5].2.token_direction(Vec2::from_angle(angle));
            // (the same thing as the transpose of the lens's matrix applied to w)
            assert!((j - a.transpose() * Vec2::from_angle(angle)).length() < 1e-5);
            assert!(j.normalize().dot(table.axes[0]).abs() > 0.97, "token at {angle}: {j:?}");
        }
    }

    /// When both modes survive alike, no direction is the lens's first (it would be
    /// whichever way rounding tipped it); nor is any the activations' main one when they
    /// vary as much along both modes.
    #[test]
    fn with_both_modes_alike_no_direction_comes_first() {
        // "Only the contexts differ", then mode 2's agreement up to 1: the lens is the identity
        let mut m = EXPERIMENTS[3].2;
        m.modes[1].agree = 1.0;
        let table = m.table();
        let (s1, s2, _) = table.singular();
        assert!((s1 - 1.0).abs() < 1e-5 && (s2 - 1.0).abs() < 1e-5, "{s1} {s2}");
        assert!(table.pick() == Pick::Alike && table.first_mode().is_none());
        assert!(m.principal().is_none());
        // a little apart, and there is a first direction again, along the stronger mode
        m.modes[1].agree = 0.99;
        assert_eq!(m.table().first_mode(), Some(0));
        m.modes[1].spread = 0.71;
        assert_eq!(m.principal(), Some(m.axes()[1]));
        // nothing survives when neither mode is agreed on, nor when neither answers
        let mut none = EXPERIMENTS[1].2;
        none.modes[0].agree = 0.0;
        assert_eq!(none.table().pick(), Pick::Nothing);
        let quiet = Model { modes: [Mode { strength: 0.0, ..STEADY }; 2], ..START };
        assert_eq!(quiet.table().pick(), Pick::Nothing);
        // and every experiment that names a first direction has one
        for (k, (_, _, model, _)) in EXPERIMENTS.iter().enumerate() {
            assert!(matches!(model.table().pick(), Pick::One(_)), "experiment {k}");
        }
    }

    /// In this model the lag average is as long in every context (only its direction
    /// differs), which is why the side pictures can show one context and stand for all.
    #[test]
    fn the_lag_average_is_as_long_in_every_context() {
        let mut rng = Rng::new(4);
        for _ in 0..40 {
            let table = random_model(&mut rng).table();
            let v = Vec2::from_angle(rng.range(0.0, TAU));
            let lengths: Vec<f32> = table.contexts.iter().map(|(signs, _)| length(table.lag_average(*signs, v))).collect();
            assert!(lengths.iter().all(|l| (l - lengths[0]).abs() < 1e-5), "{lengths:?}");
        }
    }
}

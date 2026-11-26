//! The pseudoinverse: the best substitute for the inverse of a matrix that squashes the
//! plane. From the SVD, every singular value that is not 0 turned upside down and every
//! 0 left as 0. What it gives is the nearest answer and, of those, the shortest; and a
//! singular value that is small but not 0 is better cut off or damped than turned
//! upside down.

use super::*;
use glam::{DMat2, DVec2};

pub const DEF: SceneDef = SceneDef {
    id: "pinv",
    title: "The pseudoinverse",
    group: FACTOR,
    blurb: "How far a matrix without an inverse can be undone: the nearest answer and the shortest, from the SVD.",
    make: |_| Box::new(Pinv::create()),
};

// ----- colours: one each, in both planes, in the chart and in the key ------------------

/// v₁ and u₁, the directions A stretches most; their lines are the row space and the
/// column space when σ₂ = 0.
const FIRST: Color = pal::ORANGE;
/// v₂ and u₂; the line of v₂ is the null space when σ₂ = 0.
const SECOND: Color = pal::PURPLE;
/// The unit circle, and where A takes it.
const CIRCLE: Color = pal::YELLOW;
/// b, which the visitor drags.
const POINT: Color = pal::WHITE;
/// A⁺, a second matrix beside A, and what it gives: x⁺ = A⁺b, and p = Ax⁺.
const PINV: Color = pal::BLUE;
/// The answer with the small singular values cut off.
const CUT: Color = pal::CYAN;
/// The damped answer (as ridge is drawn in the scene of sensitive equations).
const DAMP: Color = pal::GREEN;
/// What is left over: the part of b that no Ax reaches.
const LEFT: Color = pal::RED;
/// The panes' backing (the hole of a ring is drawn in it).
const BACKING: Color = Color::hex(0x0B0F1A);

// ----- the arithmetic, in double precision ----------------------------------------------

/// The largest σ₁.
const S_MAX: f64 = 2.0;
/// How far out b may be dragged, along each axis.
const B_MAX: f64 = 2.4;
/// Each plane shows at least this far from 0 every way.
const R_FIT: f64 = 2.5;
/// The largest cut-off and damping the sliders give.
const TAU_MAX: f32 = 1.0;
const LAMBDA_MAX: f32 = 1.0;
/// The top of the chart of what each σ becomes.
const Y_TOP: f64 = 5.0;

/// What the scene is set to: A by its SVD (σ₁ ≥ σ₂ ≥ 0, and the directions of u₁ and v₁
/// in radians), b, the cut-off τ and the damping λ.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    s1: f64,
    s2: f64,
    tu: f64,
    tv: f64,
    b: DVec2,
    tau: f64,
    lambda: f64,
}

impl Setting {
    fn numbers(&self) -> [f64; 8] {
        [self.s1, self.s2, self.tu, self.tv, self.b.x, self.b.y, self.tau, self.lambda]
    }

    fn from_numbers(v: [f64; 8]) -> Setting {
        Setting { s1: v[0], s2: v[1], tu: v[2], tv: v[3], b: DVec2::new(v[4], v[5]), tau: v[6], lambda: v[7] }
    }

    fn svd(&self) -> Svd {
        let (u1, v1) = (DVec2::from_angle(self.tu), DVec2::from_angle(self.tv));
        Svd { s: [self.s1, self.s2], u: [u1, u1.perp()], v: [v1, v1.perp()] }
    }
}

/// A = σ₁ u₁ v₁ᵀ + σ₂ u₂ v₂ᵀ: U and V are turns (so det A = σ₁σ₂ is never negative here).
#[derive(Clone, Copy, Debug)]
struct Svd {
    s: [f64; 2],
    u: [DVec2; 2],
    v: [DVec2; 2],
}

/// How a singular value σ is turned upside down.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Undo {
    /// The pseudoinverse: 1/σ, and 0 for σ = 0.
    Exact,
    /// The same, with every σ below τ treated as 0.
    Cut(f64),
    /// σ/(σ² + λ): ridge.
    Damp(f64),
}

impl Undo {
    fn factor(self, s: f64) -> f64 {
        if s <= 0.0 || s.is_nan() {
            return 0.0;
        }
        match self {
            Undo::Exact => 1.0 / s,
            Undo::Cut(tau) => {
                if s >= tau {
                    1.0 / s
                } else {
                    0.0
                }
            }
            Undo::Damp(lambda) => s / (s * s + lambda.max(0.0)),
        }
    }

    /// Whether σ is turned upside down at all (else it is treated as 0).
    fn keeps(self, s: f64) -> bool {
        self.factor(s) != 0.0
    }
}

/// The matrix a bᵀ.
fn outer(a: DVec2, b: DVec2) -> DMat2 {
    DMat2::from_cols(a * b.x, a * b.y)
}

impl Svd {
    fn a(&self) -> DMat2 {
        outer(self.u[0], self.v[0]) * self.s[0] + outer(self.u[1], self.v[1]) * self.s[1]
    }

    /// How many singular values are not 0.
    fn rank(&self) -> usize {
        self.s.iter().filter(|s| **s > 0.0).count()
    }

    /// V Σ⁺ Uᵀ, each σ turned upside down as `how` says.
    fn undo(&self, how: Undo) -> DMat2 {
        outer(self.v[0], self.u[0]) * how.factor(self.s[0]) + outer(self.v[1], self.u[1]) * how.factor(self.s[1])
    }

    /// The same applied to b: each part of b along a uᵢ, turned into a part along vᵢ.
    fn solve(&self, how: Undo, b: DVec2) -> DVec2 {
        self.v[0] * (how.factor(self.s[0]) * self.u[0].dot(b)) + self.v[1] * (how.factor(self.s[1]) * self.u[1].dot(b))
    }

    /// Where A takes the answer `how` gives for b: b's part along each uᵢ, times σᵢ times
    /// what σᵢ becomes. Worked out from b, not from the answer: with σ₂ a hair above 0,
    /// A⁻¹b is so long along v₂ that its part along v₁ is lost in rounding (and A's
    /// entries cannot hold σ₂ at all), but σ₂ times 1/σ₂ is still 1.
    fn reached(&self, how: Undo, b: DVec2) -> DVec2 {
        self.u[0] * (self.s[0] * how.factor(self.s[0]) * self.u[0].dot(b)) + self.u[1] * (self.s[1] * how.factor(self.s[1]) * self.u[1].dot(b))
    }

    /// The projections onto the column space (the uᵢ with σᵢ > 0) and the row space:
    /// AA⁺ and A⁺A, worked out from the SVD for the same reason.
    fn onto_columns(&self) -> DMat2 {
        (0..2).filter(|&i| self.s[i] > 0.0).fold(DMat2::ZERO, |m, i| m + outer(self.u[i], self.u[i]))
    }

    fn onto_rows(&self) -> DMat2 {
        (0..2).filter(|&i| self.s[i] > 0.0).fold(DMat2::ZERO, |m, i| m + outer(self.v[i], self.v[i]))
    }
}

/// Everything the picture shows, worked out from a setting.
#[derive(Clone, Copy, Debug)]
struct Answers {
    svd: Svd,
    rank: usize,
    b: DVec2,
    /// x⁺ = A⁺b, and p = Ax⁺, the point of the column space nearest to b.
    x: DVec2,
    p: DVec2,
    /// The cut-off answer, and how many σ the cut-off keeps; the damped answer.
    cut: DVec2,
    kept: usize,
    damp: DVec2,
    /// The cut-off τ and the damping λ, and whether those answers are shown (τ > 0, λ > 0).
    tau: f64,
    lambda: f64,
    cutting: bool,
    damping: bool,
}

impl Answers {
    fn of(s: &Setting) -> Answers {
        let svd = s.svd();
        let x = svd.solve(Undo::Exact, s.b);
        let cut = Undo::Cut(s.tau);
        Answers {
            svd,
            rank: svd.rank(),
            b: s.b,
            x,
            p: svd.reached(Undo::Exact, s.b),
            cut: svd.solve(cut, s.b),
            kept: svd.s.iter().filter(|v| cut.keeps(**v)).count(),
            damp: svd.solve(Undo::Damp(s.lambda), s.b),
            tau: s.tau,
            lambda: s.lambda,
            cutting: s.tau > 0.0,
            damping: s.lambda > 0.0,
        }
    }
}

// ----- experiments ----------------------------------------------------------------------

/// The directions of u₁ and v₁ every experiment uses, in degrees.
const TU: f64 = 15.0;
const TV: f64 = -20.0;

/// b sliding along u₂ and back: by `amp` × sin(2πt/`period`) × sin(πt/`length`) at the
/// time t, for `length` seconds. It starts and stops at rest, where it began.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Film {
    amp: f64,
    period: f32,
    length: f32,
}

impl Film {
    fn at(&self, t: f32) -> f64 {
        let t = t.max(0.0).min(self.length) as f64;
        let (p, l) = (self.period as f64, self.length as f64);
        self.amp * (std::f64::consts::TAU * t / p).sin() * (std::f64::consts::PI * t / l).sin()
    }
}

struct Trial {
    label: &'static str,
    look_for: &'static str,
    s: [f64; 2],
    /// b by its parts along u₁ and u₂.
    b: [f64; 2],
    tau: f64,
    lambda: f64,
    film: Option<Film>,
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "no inverse: the nearest, then the shortest",
        look_for: r"$\sigma_2 = 0$: $A$ squashes the plane onto the orange line among the outputs, its column space, so $Ax = b$ has no answer for this $b$, which is off the line. The point of the line nearest to $b$ is $p$, where the red left-over $b - p$ (0.90 long) meets the line at a right angle. Every $x$ on the dashed blue line among the inputs reaches $p$: they differ by steps along the null space (purple), which $A$ sends to $0$. $x^+ = A^+ b$ is the shortest of them, 1.50 long, at the corner mark on the row space. A step of 1 along the line from it is $\sqrt{1.50^2 + 1^2} = 1.80$ long.",
        s: [1.2, 0.0],
        b: [1.8, 0.9],
        tau: 0.0,
        lambda: 0.0,
        film: None,
    },
    Trial {
        label: "slide $b$ across the column space",
        look_for: r"$b$ slides along $u_2$, at right angles to the column space: out to 1.80 from it, across to 0.70 on the other side, out to 1.80 again, and back to where it began. $p$ and $x^+$ do not move at all, because $A^+$ sends $u_2$ to $0$. Only the red left-over $b - p$ changes, and it is $0$ each time $b$ crosses the line.",
        s: [1.2, 0.0],
        b: [1.8, 0.9],
        tau: 0.0,
        lambda: 0.0,
        film: Some(Film { amp: 1.6, period: 4.0, length: 6.0 }),
    },
    Trial {
        label: "an inverse: $A^+ = A^{-1}$",
        look_for: r"$\sigma_2 = 0.9$: no direction is squashed flat, so $A$ has an inverse, and $A^+$ is that inverse: $AA^+ = A^+A = I$. The column space is the whole plane, so $p = b$, and exactly one $x$ reaches it: $x^+ = A^{-1}b$. Its part along $v_2$ is $b$'s part along $u_2$, 0.90, divided by $\sigma_2$: 1.00. With $\sigma_2 = 0$ that part was dropped.",
        s: [1.2, 0.9],
        b: [1.8, 0.9],
        tau: 0.0,
        lambda: 0.0,
        film: None,
    },
    Trial {
        label: "nearly flat: the inverse runs away",
        look_for: r"$\sigma_2 = 0.03$: $A$ has an inverse, but $A^{-1}$ divides $b$'s part along $u_2$, 0.90, by 0.03. Its answer is 30.0 from $0$, far off the picture (the blue arrow leaves it). The cut-off answer (cyan) treats $\sigma_2$ as $0$, since it is below $\tau = 0.1$: it is the 1.50-long answer of the first experiment. The damped answer (green, $\lambda = 0.04$) multiplies by $\sigma_2/(\sigma_2^2 + \lambda) = 0.73$ instead of $1/\sigma_2 = 33.3$, and stays near too.",
        s: [1.2, 0.03],
        b: [1.8, 0.9],
        tau: 0.1,
        lambda: 0.04,
        film: None,
    },
    Trial {
        label: "noise in $b$, blown up by $1/\\sigma_2$",
        look_for: r"$\sigma_2 = 0.25$, and $b$ wobbles along $u_2$ by up to 0.15, as noise in a measurement would move it. $A^{-1}$ multiplies that by $1/\sigma_2 = 4$: its answer (blue) swings by up to 0.60. The cut-off answer (cyan, $\tau = 0.3$) ignores $u_2$ altogether and does not move. The damped one (green, $\lambda = 0.1$) multiplies by 1.54, and swings by up to 0.23. The price: neither of them solves $Ax = b$. Their $Ax$, the small dots among the outputs, miss $b$.",
        s: [1.2, 0.25],
        b: [1.8, 0.2],
        tau: 0.3,
        lambda: 0.1,
        film: Some(Film { amp: 0.15, period: 1.2, length: 6.6 }),
    },
];

impl Trial {
    fn setting(&self) -> Setting {
        let (tu, tv) = (TU.to_radians(), TV.to_radians());
        let u1 = DVec2::from_angle(tu);
        Setting { s1: self.s[0], s2: self.s[1], tu, tv, b: u1 * self.b[0] + u1.perp() * self.b[1], tau: self.tau, lambda: self.lambda }
    }
}

// ----- numbers for the screen -----------------------------------------------------------

/// A number for the screen: two decimals and a true minus sign, and in powers of ten
/// once it is too long for that.
fn short(v: f64) -> String {
    if v.is_finite() && v.abs() >= 9999.5 {
        format!("{v:.1e}").replace('-', "−")
    } else {
        num(v, 2)
    }
}

/// A setting's value for the screen (σ, τ, λ): as `short` writes it, but in powers of ten
/// below 0.005, so that a value that is not 0 is never written as 0.
fn value(v: f64) -> String {
    if v != 0.0 && v.abs() < 0.005 && v.is_finite() {
        format!("{v:.1e}").replace('-', "−")
    } else if v.abs() < 0.1 && v.is_finite() {
        // (with figures enough that 1 over it, written beside it, is believed; no
        // more zeros at the end than two decimals need)
        let mut text = format!("{v:.4}").replace('-', "−");
        while text.ends_with('0') && text.split('.').nth(1).is_some_and(|d| d.len() > 2) {
            text.pop();
        }
        text
    } else {
        short(v)
    }
}

/// A number for a matrix on the screen, as a formula, in five characters at most (so
/// that the brackets never move as the numbers change): two decimals, then one, then
/// none, and a power of ten beyond 9999.
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

/// The widest `cell` can be, in characters of its number (a minus sign counted as two).
const CELL: usize = 6;

/// A point, as a formula's markup.
fn point(p: DVec2) -> String {
    format!("({}, {})", short(p.x), short(p.y))
}

// ----- the layout -----------------------------------------------------------------------

/// Room above a pane for its title (logical pixels).
const TITLE: f32 = 20.0;

/// One of the two planes: where it is on the canvas (logical pixels), and how its own
/// units are drawn there.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Pane {
    lo: Vec2,
    hi: Vec2,
    /// Logical pixels per unit of the plane (the same in both).
    scale: f32,
    /// The canvas, and logical pixels per unit of the stage (`STAGE_VIEW`).
    view: Vec2,
    ppu: f32,
}

impl Pane {
    /// Where 0 of the plane is, on the canvas.
    fn zero(&self) -> Vec2 {
        (self.lo + self.hi) * 0.5
    }

    /// A point of the canvas (logical pixels) on the stage.
    fn stage(&self, px: Vec2) -> Vec2 {
        Vec2::new(px.x - self.view.x * 0.5, self.view.y * 0.5 - px.y) / self.ppu
    }

    /// A point of the plane on the canvas (only for points in or near what it shows).
    fn px(&self, p: DVec2) -> Vec2 {
        let z = self.zero();
        Vec2::new(z.x + p.x as f32 * self.scale, z.y - p.y as f32 * self.scale)
    }

    /// A point of the plane on the stage.
    fn to(&self, p: DVec2) -> Vec2 {
        self.stage(self.px(p))
    }

    /// The point of the plane at a point of the stage.
    fn back(&self, s: Vec2) -> DVec2 {
        let px = Vec2::new(s.x * self.ppu + self.view.x * 0.5, self.view.y * 0.5 - s.y * self.ppu);
        let d = (px - self.zero()) / self.scale.max(1e-6);
        DVec2::new(d.x as f64, -d.y as f64)
    }

    /// The part of the plane the pane shows (a pixel in from its edges).
    fn seen(&self) -> (DVec2, DVec2) {
        let half = (((self.hi - self.lo) * 0.5 - Vec2::ONE).max(Vec2::ONE) / self.scale.max(1e-6)).as_dvec2();
        (-half, half)
    }

    fn shows(&self, p: DVec2) -> bool {
        let (lo, hi) = self.seen();
        p.is_finite() && p.cmpge(lo).all() && p.cmple(hi).all()
    }
}

/// Where everything goes on a canvas, in logical pixels from its top-left corner.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    ppu: f32,
    /// One plane above the other (a phone held upright), not side by side.
    narrow: bool,
    /// The heading's top-left corner and the room it has across (none on a small canvas).
    heading: Option<(Vec2, f32)>,
    /// The plane of the inputs x, then that of the outputs.
    panes: [Pane; 2],
    /// The chart of what each σ becomes, and the matrices (none where there is no room).
    chart: Option<(Vec2, Vec2)>,
    mats: Option<(Vec2, Vec2)>,
}

/// The size of the matrices' numbers, and the room the four of them take.
const MAT_TEXT: f32 = 13.0;

fn mats_size() -> Vec2 {
    let m = matrix_size(2, 2, CELL, MAT_TEXT);
    let lead = text_width("$A^+A =$", MAT_TEXT) + 8.0;
    Vec2::new(2.0 * (lead + m.x) + 28.0 + 20.0, 2.0 * m.y + 16.0 + 18.0 + 28.0)
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::ONE);
        let ppu = (view.y / STAGE_H).min(view.x / 2.0).max(1e-3);
        let narrow = view.x < 620.0 && view.y > view.x * 1.3;
        let top = inset.max(0.0) + if narrow { 6.0 } else { 8.0 };
        let pane = |lo: Vec2, hi: Vec2| Pane { lo, hi: hi.max(lo + Vec2::ONE), scale: 1.0, view, ppu };
        let (heading, mut panes, chart, mats);
        if narrow {
            let head_h = if view.y >= 640.0 { 24.0 } else { 0.0 };
            heading = (head_h > 0.0).then_some((Vec2::new(10.0, top), view.x - 20.0));
            let strip = if view.y >= 640.0 { (view.y * 0.17).clamp(110.0, 150.0) } else { 0.0 };
            let gap = 46.0;
            let first = top + head_h + TITLE;
            let bottom = view.y - 8.0 - if strip > 0.0 { strip + 8.0 } else { 0.0 };
            let h = ((bottom - first - gap) * 0.5).max(10.0);
            panes = [pane(Vec2::new(8.0, first), Vec2::new(view.x - 8.0, first + h)), pane(Vec2::new(8.0, first + h + gap), Vec2::new(view.x - 8.0, first + 2.0 * h + gap))];
            chart = (strip > 0.0).then(|| (Vec2::new(8.0, view.y - 8.0 - strip), Vec2::new(view.x - 8.0, view.y - 8.0)));
            mats = None;
        } else {
            let head_h = if view.y >= 480.0 && view.x >= 560.0 { 46.0 } else { 0.0 };
            heading = (head_h > 0.0).then_some((Vec2::new(14.0, top), view.x - 28.0));
            let first = top + head_h + TITLE;
            let gap = (view.x * 0.07).clamp(46.0, 84.0);
            let w = ((view.x - 24.0 - gap) * 0.5).max(10.0);
            let strip = if view.y >= 560.0 { (view.y * 0.22).clamp(150.0, 220.0) } else { 0.0 };
            let h = (view.y - 10.0 - first - if strip > 0.0 { strip + 14.0 } else { 0.0 }).max(10.0);
            let strip_top = first + h + 14.0;
            panes = [pane(Vec2::new(12.0, first), Vec2::new(12.0 + w, first + h)), pane(Vec2::new(view.x - 12.0 - w, first), Vec2::new(view.x - 12.0, first + h))];
            let (lo, hi) = (Vec2::new(12.0, strip_top), Vec2::new(view.x - 12.0, strip_top + strip));
            if strip < 100.0 {
                (chart, mats) = (None, None);
            } else {
                let need = mats_size();
                if hi.x - lo.x >= need.x + 20.0 + 280.0 && strip >= need.y {
                    chart = Some((lo, Vec2::new(hi.x - need.x - 20.0, hi.y)));
                    mats = Some((Vec2::new(hi.x - need.x, lo.y), Vec2::new(hi.x, lo.y + need.y)));
                } else {
                    (chart, mats) = (Some((lo, hi)), None);
                }
            }
        }
        // the same scale in both planes, showing at least R_FIT every way
        let least = panes.iter().map(|p| (p.hi - p.lo).min_element()).fold(f32::MAX, f32::min);
        let scale = ((least * 0.5 - 14.0) / R_FIT as f32).max(2.0);
        for p in &mut panes {
            p.scale = scale;
        }
        Layout { view, ppu, narrow, heading, panes, chart, mats }
    }

    /// A point of the canvas on the stage.
    fn stage(&self, px: Vec2) -> Vec2 {
        Vec2::new(px.x - self.view.x * 0.5, self.view.y * 0.5 - px.y) / self.ppu
    }
}

// ----- the glide from one setting to another ---------------------------------------------

/// How fast a change of setting is glided to: each of the two eases one after the other
/// covers two thirds of what is left in this many seconds.
const FOLLOW: f64 = 0.09;

/// A setting as asked for, and as drawn: the drawn one follows it through `between`,
/// two eases one after the other, so that it sets off gently and arrives exactly.
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

    /// Aim at a new setting, its angles taken the short way round from where they are.
    fn aim(&mut self, mut s: Setting) {
        let near = |to: f64, from: f64| from + (to - from + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
        s.tu = near(s.tu, self.shown.tu);
        s.tv = near(s.tv, self.shown.tv);
        self.to = s;
    }

    /// Move the drawn setting `dt` seconds nearer to the one asked for.
    fn step(&mut self, dt: f64) {
        if self.arrived() || dt.is_nan() || dt <= 0.0 {
            return;
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        let (to, mut between, mut shown) = (self.to.numbers(), self.between.numbers(), self.shown.numbers());
        let mut apart = 0.0f64;
        for i in 0..to.len() {
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

/// A film under way: b slides along u₂ from `centre` (waiting until the glide to its
/// setting has arrived).
#[derive(Clone, Copy, Debug)]
struct Playing {
    film: Film,
    t: f32,
    centre: Option<DVec2>,
}

// ----- the scene ------------------------------------------------------------------------

pub struct Pinv {
    glide: Glide,
    playing: Option<Playing>,
    /// The heading's sizes as last chosen, with what they were chosen for.
    heading: Option<(String, String, f32, (f32, f32))>,
}

impl Pinv {
    fn create() -> Self {
        let mut scene = Pinv { glide: Glide::at(TRIALS[0].setting()), playing: None, heading: None };
        scene.set_up(0);
        scene.glide = Glide::at(scene.glide.to);
        scene
    }

    /// Set up experiment `k` of `TRIALS`: glide to its setting, then play its film.
    fn set_up(&mut self, k: usize) {
        let trial = &TRIALS[k];
        self.glide.aim(trial.setting());
        self.playing = trial.film.map(|film| Playing { film, t: 0.0, centre: None });
    }

    fn shown(&self) -> Setting {
        self.glide.shown
    }

    /// Move everything on: `dt` seconds of the scene's time (the film) and `real_dt` of
    /// the clock's (a change of setting glides by the clock, so that it is seen through
    /// even while the scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        self.glide.step(real_dt as f64);
        let Some(play) = &mut self.playing else { return };
        if play.centre.is_none() {
            if !self.glide.arrived() {
                return;
            }
            play.centre = Some(self.glide.to.b);
        }
        let centre = play.centre.unwrap_or(self.glide.to.b);
        play.t += if dt.is_finite() { dt.max(0.0) } else { 0.0 };
        let done = play.t >= play.film.length;
        let b = if done { centre } else { centre + self.glide.to.svd().u[1] * play.film.at(play.t) };
        self.glide.now(|s| s.b = b);
        if done {
            self.playing = None;
        }
    }

    /// The handles: b, σ₁u₁ and σ₂u₂ among the outputs, v₁ among the inputs. A hand moves the
    /// setting at once.
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        let s = self.shown();
        let svd = s.svd();
        let [inputs, outputs] = lay.panes;
        let mut at = outputs.to(s.b);
        if f.drag2(1, &mut at, POINT) {
            let (lo, hi) = outputs.seen();
            // (kept in view, and within ±B_MAX; max then min, which no bound can upset)
            let b = outputs.back(at).max(lo * 0.97).min(hi * 0.97).max(DVec2::splat(-B_MAX)).min(DVec2::splat(B_MAX));
            let b = if b.is_finite() { b } else { s.b };
            self.glide.now(|g| g.b = b);
            self.playing = None;
        }
        let mut at = outputs.to(svd.u[0] * svd.s[0]);
        if f.drag2(2, &mut at, FIRST) {
            let p = outputs.back(at);
            let s1 = p.length().min(S_MAX);
            let tu = if p.length() > 1e-6 { p.y.atan2(p.x) } else { s.tu };
            self.glide.now(|g| {
                g.s1 = s1;
                g.s2 = g.s2.min(s1);
                g.tu = tu;
            });
        }
        if s.s1 > 0.0 {
            let mut at = outputs.to(svd.u[1] * svd.s[1]);
            if f.drag2(3, &mut at, SECOND) {
                let s2 = outputs.back(at).dot(svd.u[1]).max(0.0).min(s.s1);
                self.glide.now(|g| g.s2 = s2);
            }
        }
        let mut at = inputs.to(svd.v[0]);
        if f.drag2(4, &mut at, FIRST) {
            let p = inputs.back(at);
            if p.length() > 1e-6 {
                let tv = p.y.atan2(p.x);
                self.glide.now(|g| g.tv = tv);
            }
        }
    }
}

impl Scene for Pinv {
    fn view(&self) -> View {
        STAGE_VIEW
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        !self.glide.arrived() || self.playing.is_some()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("white dot: move $b$    orange and purple dots: change $A$    drag elsewhere: move    scroll: zoom")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A matrix that squashes the plane has no inverse: some $b$ are reached by no $x$, and others by many. $A$ here is $2 \times 2$, given by its SVD: it takes the direction $v_1$ to $\sigma_1 u_1$ and $v_2$ to $\sigma_2 u_2$ (scene svd), so $\sigma_2 = 0$ (with $\sigma_1 > 0$) squashes the plane onto the line of $u_1$. Its best substitute for an inverse is the pseudoinverse,");
        ui::equation(ui, r"A = U \Sigma V^\top, \quad A^+ = V \Sigma^+ U^\top,");
        ui::note(ui, r"where $\Sigma^+$ has $1/\sigma$ for each $\sigma$ that is not $0$, and $0$ for each $0$. Then $x^+ = A^+ b$ makes $Ax$ as near to $b$ as it can be, and is the shortest $x$ that does.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, r"The matrix $A$");
        let to = self.glide.to;
        let mut s1 = to.s1 as f32;
        if ui::slider(ui, r"$\sigma_1$", &mut s1, 0.0..=S_MAX as f32) {
            let s1 = s1 as f64;
            self.glide.aim(Setting { s1, s2: to.s2.min(s1), ..to });
        }
        let mut s2 = to.s2 as f32;
        if ui::slider(ui, r"$\sigma_2$", &mut s2, 0.0..=(to.s1 as f32).max(1e-6)) {
            self.glide.aim(Setting { s2: (s2 as f64).max(0.0).min(to.s1), ..to });
        }
        ui::note(ui, r"Or drag the dots on the picture: among the outputs, the orange and purple ends of the yellow ellipse are $\sigma_1 u_1$ and $\sigma_2 u_2$; among the inputs, the orange one on the circle turns $v_1$.");

        ui::section(ui, r"Taming a small $\sigma$");
        let mut tau = to.tau as f32;
        if ui::slider_log_or_zero(ui, r"cut-off $\tau$", &mut tau, 0.001, TAU_MAX) {
            self.glide.aim(Setting { tau: tau as f64, ..to });
        }
        ui::note(ui, r"Every $\sigma$ below $\tau$ is treated as $0$ (the cyan answer). At $0$: off.");
        let mut lambda = to.lambda as f32;
        if ui::slider_log_or_zero(ui, r"damping $\lambda$", &mut lambda, 0.0001, LAMBDA_MAX) {
            self.glide.aim(Setting { lambda: lambda as f64, ..to });
        }
        ui::note(ui, r"Each $1/\sigma$ becomes $\sigma/(\sigma^2 + \lambda)$ (the green answer). At $0$: off.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (CIRCLE, r"yellow: the unit circle (inputs), and where $A$ takes it (outputs): an ellipse with half-axes $\sigma_1 u_1$ and $\sigma_2 u_2$, flat when $\sigma_2 = 0$"),
                (FIRST, r"orange: $v_1$ (inputs), the direction $A$ stretches most, and $u_1$ (outputs), where it goes: $Av_1 = \sigma_1 u_1$. When $\sigma_2 = 0 < \sigma_1$ their lines are the row space and the column space"),
                (SECOND, r"purple: $v_2$ and $u_2$, with $Av_2 = \sigma_2 u_2$. When $\sigma_2 = 0 < \sigma_1$ the line of $v_2$ is the null space, every $x$ that $A$ sends to $0$"),
                (POINT, "white: $b$ (drag it)"),
                (PINV, r"blue: $x^+ = A^+b$ (inputs), and $p = Ax^+$ (outputs), the point $A$ can reach nearest to $b$; dashed, every $x$ that $A$ also takes to $p$"),
                (LEFT, r"red: $b - p$, the part of $b$ that no $Ax$ reaches"),
                (CUT, r"cyan: $x_\tau$, the answer with every $\sigma$ below $\tau$ treated as $0$, and (small dot) $Ax_\tau$"),
                (DAMP, r"green: $x_\lambda$, the damped answer, and (small dot) $Ax_\lambda$"),
                (pal::TEXT, r"the chart: what each $\sigma$ becomes in $A^+$ (blue: $1/\sigma$, and $0$ for $0$), cut off (cyan) and damped (green); the purple line is $\sigma_2$, the orange $\sigma_1$"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, r"Pseudoinverse $A^+$:", r"$V \Sigma^+ U^\top$, the SVD with each $\sigma$ that is not $0$ turned upside down and each $0$ left as $0$. If $A$ has an inverse, $A^+ = A^{-1}$.");
        ui::term(ui, "Column space:", r"every output $Ax$ that $A$ can reach: the line of $u_1$ when $\sigma_2 = 0 < \sigma_1$, the whole plane when $\sigma_2 > 0$.");
        ui::term(ui, "Row space, null space:", r"when $\sigma_2 = 0 < \sigma_1$, the lines of $v_1$ and of $v_2$. $A$ sends the null space to $0$, so adding a step along it to $x$ changes nothing in $Ax$.");
        ui::term(ui, "Projection:", r"a map that drops each point at right angles onto a line (or plane) through $0$; doing it twice changes nothing more. $AA^+$ projects onto the column space, $A^+A$ onto the row space.");
        ui::term(ui, "Least squares:", r"the $x$ that makes $|Ax - b|$ as small as it can be (scene lsq).");
        ui::term(ui, "Not square:", r"the recipe $V\Sigma^+U^\top$ works for a matrix of any shape. With independent columns $A^+ = (A^\top A)^{-1} A^\top$, the least-squares formula; with independent rows $A^+ = A^\top (A A^\top)^{-1}$, which gives the shortest answer (scene underdet).");

        ui::section(ui, "Numbers");
        let s = self.shown();
        let ans = Answers::of(&s);
        let number = |v: f64| format!("${}$", value(v));
        ui::readout_colored(ui, r"$\sigma_1$", number(s.s1), FIRST);
        ui::readout_colored(ui, r"$\sigma_2$", number(s.s2), SECOND);
        ui::readout(ui, r"$1/\sigma_2$", if s.s2 > 0.0 { number(1.0 / s.s2) } else { r"none: $\sigma_2 = 0$".into() });
        ui::readout_colored(ui, r"$x^+ = A^+ b$", format!("${}$", point(ans.x)), PINV);
        ui::readout_colored(ui, r"$|x^+|$", number(ans.x.length()), PINV);
        ui::readout_colored(ui, r"$|b - p|$, what is left over", number((ans.b - ans.p).length()), LEFT);
        if ans.cutting {
            ui::readout_colored(ui, r"$x_\tau$, cut off", format!("${}$", point(ans.cut)), CUT);
        }
        if ans.damping {
            ui::readout_colored(ui, r"$x_\lambda$, damped", format!("${}$", point(ans.damp)), DAMP);
        }
        let rows = |m: DMat2| [[m.x_axis.x, m.y_axis.x], [m.x_axis.y, m.y_axis.y]];
        let a = ans.svd.a();
        let pinv = ans.svd.undo(Undo::Exact);
        let big = |m: DMat2| m.to_cols_array().iter().any(|v| v.abs() >= 999.5);
        if big(pinv) {
            ui::equation(ui, &format!(r"A = {}", bmatrix(&rows(a), 2)));
            ui::note(ui, &format!(r"$A^+$ has entries as large as ${}$.", short(pinv.to_cols_array().iter().fold(0.0f64, |m, v| m.max(v.abs())))));
        } else {
            ui::equation(ui, &format!(r"A = {}, \quad A^+ = {}", bmatrix(&rows(a), 2), bmatrix(&rows(pinv), 2)));
        }
        ui::equation(ui, &format!(r"AA^+ = {}, \quad A^+A = {}", bmatrix(&rows(ans.svd.onto_columns()), 2), bmatrix(&rows(ans.svd.onto_rows()), 2)));
        ui::see_also(
            ui,
            &[
                ("svd", r"where $\sigma_1, \sigma_2$, the $u$'s and the $v$'s come from"),
                ("lsq", "the nearest point, as a projection"),
                ("underdet", "the shortest answer, and other ways to choose one"),
                ("condition", r"how much an error in $b$ grows in $x$; ridge"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset);
        self.advance(f.dt, f.real_dt);
        // (a canvas squeezed to a sliver, beside the controls on a phone, is not touched)
        let roomy = lay.view.x >= 160.0 && lay.view.y >= 120.0;
        if roomy {
            self.handles(f, &lay);
        }
        // (names only where the panes have room for them)
        let named = roomy && lay.panes.iter().all(|p| (p.hi - p.lo).min_element() >= 150.0);
        let ans = Answers::of(&self.shown());
        let mut room = Room::new(f);
        walls(&mut room, &lay);
        draw_inputs(f, &mut room, &lay, &ans, named);
        draw_outputs(f, &mut room, &lay, &ans, named);
        between(f, &lay);
        titles(f, &lay);
        if let Some((at, width)) = lay.heading {
            self.heading(f, at, width, &self.shown(), lay.narrow);
        }
        if let Some(rect) = lay.chart {
            chart(f, &lay, rect, &self.shown(), &ans);
        }
        if let Some(rect) = lay.mats {
            matrices(f, rect, &ans);
        }
    }
}

// ----- drawing ----------------------------------------------------------------------------

/// Where a piece of the picture goes when it is not shown: far off any canvas (stage
/// units), see-through. Each piece is drawn in every frame, shown or not, so that the
/// list of what is drawn keeps its order from frame to frame (`eidos motion` compares
/// frames point by point). A piece that jumps where the mathematics jumps (x⁺, as σ₂
/// leaves 0) is drawn as one copy for each side of the jump, the other copy nowhere:
/// the jump is then a piece leaving and another arriving, not a piece flying across.
const NOWHERE: Vec2 = Vec2::new(-1e4, -1e4);

/// The part of the segment from `a` to `b` inside the box.
fn segment_in(a: DVec2, b: DVec2, (lo, hi): (DVec2, DVec2)) -> Option<(DVec2, DVec2)> {
    if !(a.is_finite() && b.is_finite()) {
        return None;
    }
    let d = b - a;
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
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
    (t0 <= t1).then(|| (if t0 <= 0.0 { a } else { a + d * t0 }, if t1 >= 1.0 { b } else { a + d * t1 }))
}

/// A segment of a pane's plane, cut to what it shows: always one segment (nowhere if
/// `shown` is false or none of it is in view). Returns where it was drawn, on the stage.
fn seg(f: &mut Frame, pane: &Pane, a: DVec2, b: DVec2, shown: bool, width: f32, color: Color) -> Option<(Vec2, Vec2)> {
    match segment_in(a, b, pane.seen()).filter(|_| shown) {
        Some((p, q)) => {
            let (p, q) = (pane.to(p), pane.to(q));
            f.line(p, q, width, color);
            Some((p, q))
        }
        None => {
            f.line(NOWHERE, NOWHERE, width, color.a(0.0));
            None
        }
    }
}

/// An arrow of a pane's plane from `a` to `b`: one segment always (as `seg`), and a head
/// if its tip is in view. Returns the shaft's ends drawn, and whether the tip is in view.
fn arrow(f: &mut Frame, pane: &Pane, a: DVec2, b: DVec2, shown: bool, width: f32, color: Color) -> Option<(Vec2, Vec2, bool)> {
    let tip_in = pane.shows(b);
    let Some((p, q)) = segment_in(a, b, pane.seen()).filter(|_| shown && (b - a).length() * pane.scale as f64 > 2.0) else {
        f.line(NOWHERE, NOWHERE, width, color.a(0.0));
        return None;
    };
    let (sp, sq) = (pane.to(p), pane.to(q));
    let len = (sq - sp).length();
    let mut end = sq;
    if tip_in && len > 1e-6 {
        let dir = (sq - sp) / len;
        let head = ((width * 3.2 + 5.0) / pane.ppu).min(len * 0.6);
        let side = dir.perp() * (head * 0.42);
        let base = sq - dir * head;
        f.dl.tris_alpha.tri(sq.extend(0.0), (base + side).extend(0.0), (base - side).extend(0.0), color);
        end = sq - dir * (head * 0.8);
    }
    f.line(sp, end, width, color);
    Some((sp, sq, tip_in))
}

/// A dot of a pane's plane: where it is if `shown` and in view, else nowhere.
fn dot(f: &mut Frame, pane: &Pane, p: DVec2, shown: bool, size: f32, color: Color) -> bool {
    if shown && pane.shows(p) {
        f.dot(pane.to(p), size, color);
        true
    } else {
        f.dot(NOWHERE, size, color.a(0.0));
        false
    }
}

/// The small square corner that marks a right angle at `at` between the directions `d1`
/// and `d2`, its sides `size` pixels long: always three points (nowhere if not `shown`).
#[allow(clippy::too_many_arguments)]
fn corner(f: &mut Frame, pane: &Pane, at: DVec2, d1: DVec2, d2: DVec2, shown: bool, size: f32, color: Color) {
    let len = size as f64 / pane.scale.max(1e-6) as f64;
    let (a, b) = (d1.normalize_or_zero() * len, d2.normalize_or_zero() * len);
    let pts = [at + a, at + a + b, at + b];
    if shown && pts.iter().all(|p| pane.shows(*p)) && a != DVec2::ZERO && b != DVec2::ZERO {
        f.path(pts.map(|p| pane.to(p)), 1.4, color);
    } else {
        f.path([NOWHERE; 3], 1.4, color.a(0.0));
    }
}

/// The whole line through 0 along `d`, as far as the pane shows it.
fn line_through_zero(f: &mut Frame, room: &mut Room, pane: &Pane, d: DVec2, width: f32, color: Color) {
    let far = d.normalize_or_zero() * 1e3;
    if let Some((p, q)) = seg(f, pane, -far, far, far != DVec2::ZERO, width, color) {
        room.line(f, p.extend(0.0), q.extend(0.0));
    }
}

/// Where along the ray from 0 in the unit direction `d` the pane's edge is, less `back`
/// pixels: a place for the name of a line, near where it leaves the picture.
fn near_edge(pane: &Pane, d: DVec2, back: f32) -> DVec2 {
    let (_, hi) = pane.seen();
    let t = [(hi.x, d.x), (hi.y, d.y)].iter().filter(|(_, c)| c.abs() > 1e-9).map(|(h, c)| h / c.abs()).fold(f64::MAX, f64::min);
    let t = if t.is_finite() { t } else { 1.0 };
    d * (t - back as f64 / pane.scale.max(1e-6) as f64).max(0.0)
}

/// Which end of each line through 0 (v₁, v₂, u₁, u₂) its name goes at: away from where
/// x⁺, b and p are in the experiments, which have b's part along u₁ positive. (Always the
/// same end: a name that changed ends would leap across the picture.)
const ENDS: [f64; 4] = [-1.0, 1.0, -1.0, -1.0];

/// How far off a line a name must be put, towards the side `n` (a unit direction on the
/// screen, at right angles to the line), for its box to keep a few pixels clear of the
/// line: `Room` puts a box `gap` pixels from its point towards `n`, and a box beside a
/// sloping line reaches back towards it at one corner.
fn gap_off_line(n: Vec2, text: &str, size: f32, at_least: f32) -> f32 {
    let n = n.normalize_or_zero();
    let (hx, hy) = (text_width(text, size) * 0.5 + 4.0, size * 0.5 + 1.0);
    at_least + hx * (n.x.abs() - n.x * n.x) + hy * (n.y.abs() - n.y * n.y)
}

/// The side of the line along `d` that is up on the screen.
fn up_from(d: DVec2) -> Vec2 {
    let across = Vec2::new(d.y as f32, d.x as f32);
    if across.y > 0.0 {
        -across
    } else {
        across
    }
}

/// A pane's backing, its grid and the numbers along its edges, and the names of its axes.
fn backdrop(f: &mut Frame, room: &mut Room, pane: &Pane, k: u32, names: [&str; 2], labels: bool) {
    let (lo, hi) = (pane.stage(pane.lo), pane.stage(pane.hi));
    f.dl.tris_alpha.rect(lo.min(hi), lo.max(hi), 0.0, BACKING.a(0.9));
    frame_rect(f, lo.min(hi), lo.max(hi), 1.0, pal::FAINT.a(0.7));
    let (slo, shi) = pane.seen();
    let step = nice_step(80.0 / pane.scale.max(1e-3)) as f64;
    if !(step > 0.0 && step.is_finite()) {
        return;
    }
    let ticks = |a: f64, b: f64| {
        let first = (a / step).ceil();
        let n = (((b - a) / step).floor().max(0.0) as usize).min(60);
        (0..=n).map(move |i| (first + i as f64) * step).filter(move |v| *v <= b)
    };
    // (thin lines of the grid as slivers of fill: the backdrop, not part of the picture)
    let sliver = |f: &mut Frame, a: Vec2, b: Vec2, width: f32, color: Color| {
        let across = (b - a).normalize_or_zero().perp() * (width * 0.5 / pane.ppu);
        f.dl.tris_alpha.quad((a - across).extend(0.0), (b - across).extend(0.0), (b + across).extend(0.0), (a + across).extend(0.0), color);
    };
    let px = 1.0 / pane.scale as f64;
    for x in ticks(slo.x, shi.x) {
        let zero = x.abs() < step * 1e-6;
        sliver(f, pane.to(DVec2::new(x, slo.y)), pane.to(DVec2::new(x, shi.y)), if zero { 1.4 } else { 1.0 }, pal::FAINT.a(if zero { 0.9 } else { 0.3 }));
        if labels && !zero && x > slo.x + 30.0 * px && x < shi.x - 16.0 * px {
            f.text_px(pane.px(DVec2::new(x, slo.y)) + Vec2::new(0.0, -3.0), fmt_tick(x as f32, step as f32), 10.5, pal::DIM, Align2::CENTER_BOTTOM);
        }
    }
    for y in ticks(slo.y, shi.y) {
        let zero = y.abs() < step * 1e-6;
        sliver(f, pane.to(DVec2::new(slo.x, y)), pane.to(DVec2::new(shi.x, y)), if zero { 1.4 } else { 1.0 }, pal::FAINT.a(if zero { 0.9 } else { 0.3 }));
        if labels && !zero && y > slo.y + 20.0 * px && y < shi.y - 12.0 * px {
            f.text_px(pane.px(DVec2::new(slo.x, y)) + Vec2::new(4.0, 0.0), fmt_tick(y as f32, step as f32), 10.5, pal::DIM, Align2::LEFT_CENTER);
        }
    }
    // names keep off the numbers along the edges
    room.wall(Vec2::new(pane.lo.x, pane.hi.y - 16.0), pane.hi);
    room.wall(pane.lo, Vec2::new(pane.lo.x + 30.0, pane.hi.y));
    if labels {
        room.label(f, ("axis", 2 * k), pane.to(DVec2::new(shi.x, 0.0)).extend(0.0), names[0], 12.0, pal::DIM, Vec2::new(-1.0, 1.0), 4.0, true);
        room.label(f, ("axis", 2 * k + 1), pane.to(DVec2::new(0.0, shi.y)).extend(0.0), names[1], 12.0, pal::DIM, Vec2::new(1.0, 1.0), 4.0, true);
    }
}

/// Names keep to the panes: everything round them and between them is walled off.
fn walls(room: &mut Room, lay: &Layout) {
    let [a, b] = lay.panes;
    let v = lay.view;
    let (left, right) = (a.lo.x.min(b.lo.x), a.hi.x.max(b.hi.x));
    let (top, bottom) = (a.lo.y.min(b.lo.y), a.hi.y.max(b.hi.y));
    room.wall(Vec2::ZERO, Vec2::new(v.x, top));
    room.wall(Vec2::new(0.0, bottom), v);
    room.wall(Vec2::ZERO, Vec2::new(left, v.y));
    room.wall(Vec2::new(right, 0.0), v);
    if lay.narrow {
        room.wall(Vec2::new(0.0, a.hi.y), Vec2::new(v.x, b.lo.y));
    } else {
        room.wall(Vec2::new(a.hi.x, 0.0), Vec2::new(b.lo.x, v.y));
    }
}

/// Name a point of a pane (if it is in view).
#[allow(clippy::too_many_arguments)]
fn name(f: &mut Frame, room: &mut Room, tag: (&'static str, u32), pane: &Pane, at: DVec2, text: &str, size: f32, color: Color, towards: Vec2, gap: f32, optional: bool) {
    if pane.shows(at) {
        room.label(f, tag, pane.to(at).extend(0.0), text, size, color, towards, gap, optional);
    }
}

/// The name of a line through 0, near where it leaves the picture at the end `d` (a unit
/// direction), on its side `n`: put by that rule alone, not where `Room` finds room, so
/// that it moves only as the line does and never leaps to another side. The names put
/// after it keep off it.
fn line_name(f: &mut Frame, room: &mut Room, tag: (&'static str, u32), pane: &Pane, d: DVec2, text: &str, color: Color) {
    let size = 13.0;
    let n = up_from(d).normalize_or_zero();
    let gap = gap_off_line(n, text, size, 7.0);
    let at = pane.px(near_edge(pane, d, 44.0));
    let half = Vec2::new(text_width(text, size) * 0.5 + 4.0, size * 0.5 + 1.0);
    let centre = at + n * gap + Vec2::new(n.x * half.x, n.y * half.y);
    // (kept inside the pane)
    let centre = centre.max(pane.lo + half + Vec2::splat(2.0)).min(pane.hi - half - Vec2::new(2.0, 18.0));
    if !centre.is_finite() {
        return;
    }
    room.boxed(centre - half, centre + half);
    f.text_placed(tag.0, tag.1, at, centre - at, text, size, color);
}

/// Keep names off a dot of a pane (`size` pixels across).
fn keep_off(room: &mut Room, pane: &Pane, at: DVec2, size: f32) {
    if pane.shows(at) {
        let c = pane.px(at);
        room.boxed(c - Vec2::splat(size * 0.5), c + Vec2::splat(size * 0.5));
    }
}

/// The direction on the screen of a step `d` in a pane.
fn screen(d: DVec2) -> Vec2 {
    Vec2::new(d.x as f32, -d.y as f32)
}

/// The plane of the inputs: the unit circle, v₁ and v₂ with their lines, every x that
/// reaches p, x⁺, and the cut-off and damped answers.
fn draw_inputs(f: &mut Frame, room: &mut Room, lay: &Layout, ans: &Answers, labels: bool) {
    let pane = &lay.panes[0];
    backdrop(f, room, pane, 0, ["$x_1$", "$x_2$"], labels);
    let svd = ans.svd;
    let [v1, v2] = svd.v;
    let rank = ans.rank;

    // the unit circle
    let circle: Vec<Vec2> = (0..=96).map(|i| pane.to(DVec2::from_angle(i as f64 / 96.0 * std::f64::consts::TAU))).collect();
    f.path(circle.iter().copied(), 1.4, CIRCLE.a(0.6));
    for pair in circle.windows(12).step_by(11) {
        room.line(f, pair[0].extend(0.0), pair[11].extend(0.0));
    }

    // the lines of v₁ and v₂: the row space and the null space when σ₂ = 0
    let strong = |on: bool| if on { 0.9 } else { 0.4 };
    line_through_zero(f, room, pane, v1, if rank == 1 { 2.0 } else { 1.2 }, FIRST.a(strong(rank == 1)));
    line_through_zero(f, room, pane, v2, if rank == 1 { 2.0 } else { 1.2 }, SECOND.a(strong(rank <= 1)));
    // v₁ and v₂ themselves (v₁'s tip is a handle)
    arrow(f, pane, DVec2::ZERO, v1, true, 2.2, FIRST);
    arrow(f, pane, DVec2::ZERO, v2, true, 2.2, SECOND);
    keep_off(room, pane, v1, 16.0);
    keep_off(room, pane, v2, 8.0);

    // every x that A takes to p: x⁺ plus any step along the null space (rank 1)
    let far = v2 * 1e3;
    match segment_in(ans.x - far, ans.x + far, pane.seen()).filter(|_| rank == 1) {
        Some((p, q)) => {
            let (p, q) = (pane.to(p), pane.to(q));
            f.dashed(p, q, 1.6, PINV.a(0.8));
            room.line(f, p.extend(0.0), q.extend(0.0));
        }
        None => f.dashed(NOWHERE, NOWHERE, 1.6, PINV.a(0.0)),
    }
    // x⁺ is the one at right angles to it from 0
    corner(f, pane, ans.x, -ans.x, v2, rank == 1 && ans.x.length() * pane.scale as f64 > 24.0, 10.0, pal::WHITE.a(0.8));

    // the answers: the cut-off one under x⁺, the damped one over it (copies: see NOWHERE)
    for k in 0..3 {
        // (a ring, so that it shows round x⁺ where the two are the same)
        dot(f, pane, ans.cut, ans.cutting && ans.kept == k, 17.0, CUT);
        dot(f, pane, ans.cut, ans.cutting && ans.kept == k, 9.0, BACKING);
    }
    let mut x_end = None;
    for r in 0..3 {
        if let Some(drawn) = arrow(f, pane, DVec2::ZERO, ans.x, rank == r, 2.6, PINV) {
            x_end = Some(drawn);
        }
        dot(f, pane, ans.x, rank == r, 11.0, PINV);
    }
    dot(f, pane, ans.damp, ans.damping, 8.0, DAMP);
    for p in [ans.x, ans.cut, ans.damp] {
        keep_off(room, pane, p, 14.0);
    }
    if !labels {
        return;
    }

    // names: the lines near where they leave the picture
    let line_names = match rank {
        1 => ["row space", "null space"],
        2 => ["$v_1$", "$v_2$"],
        _ => ["", ""],
    };
    for (i, (d, text, c)) in [(v1 * ENDS[0], line_names[0], FIRST), (v2 * ENDS[1], line_names[1], SECOND)].into_iter().enumerate() {
        if !text.is_empty() {
            line_name(f, room, ("in line", i as u32), pane, d, text, c);
        }
    }
    name(f, room, ("unit", 0), pane, DVec2::from_angle(-2.4), "unit circle", 12.0, CIRCLE.a(0.85), Vec2::new(-1.0, 1.0), 4.0, true);
    // (short where it sits between two lines; and not the same words on the two sides
    // of a jump, where it is not the same point)
    let x_text = match rank {
        0 => "$x^+ = 0$",
        1 => "$x^+$",
        _ => "$x^+ = A^{-1}b$",
    };
    match x_end {
        Some((_, _, true)) | None => {
            let up = if v2.y >= 0.0 { v2 } else { -v2 };
            name(f, room, ("x+", 0), pane, ans.x, x_text, 14.0, PINV.mix(pal::WHITE, 0.35), screen(ans.x.normalize_or_zero() - up), 9.0, false)
        }
        Some((_, q, false)) => {
            // off the picture: its length, where the arrow leaves it
            let text = format!(r"$|x^+| = {}$", short(ans.x.length()));
            room.label(f, ("x far", 0u32), q.extend(0.0), &text, 13.0, PINV.mix(pal::WHITE, 0.35), -screen(ans.x) + Vec2::new(1.0, 0.0), 12.0, false);
        }
    }
    if rank == 1 && ans.x.length() * pane.scale as f64 > 24.0 {
        // the dashed line's name, near where it leaves the picture upwards, on the side
        // away from 0
        let along = if v2.dot(DVec2::Y) >= 0.0 { v2 } else { -v2 };
        if let Some((_, q)) = segment_in(ans.x, ans.x + along * 1e3, pane.seen()) {
            let at = q - along * (60.0 / pane.scale as f64);
            if (at - ans.x).dot(along) > 0.0 {
                let (text, side) = ("all give $Ax = p$", screen(ans.x));
                name(f, room, ("every x", 0), pane, at, text, 12.0, PINV.mix(pal::WHITE, 0.2), side, gap_off_line(side, text, 12.0, 6.0), true);
            }
        }
    }
    if ans.cutting && ans.kept != rank {
        name(f, room, ("cut", 0), pane, ans.cut, "cut off", 12.5, CUT, screen(ans.cut - ans.x) + Vec2::new(0.0, 0.6), 10.0, false);
    } else if ans.cutting {
        name(f, room, ("cut", 1), pane, ans.cut, "cut off: the same", 12.5, CUT, -screen(ans.x) + Vec2::new(0.0, 0.6), 10.0, false);
    }
    if ans.damping {
        name(f, room, ("damp", 0), pane, ans.damp, "damped", 12.5, DAMP, screen(ans.damp - ans.x) + Vec2::new(0.0, -0.6), 8.0, false);
    }
    for (p, c, text) in [(ans.cut, CUT, "cut off"), (ans.damp, DAMP, "damped")] {
        let on = if c == CUT { ans.cutting } else { ans.damping };
        if on && !pane.shows(p) {
            // off the picture: which way, and how far
            let edge = near_edge(pane, p.normalize_or_zero(), 24.0);
            let label = format!(r"{text}: ${}$ away", short(p.length()));
            name(f, room, ("far", (c == CUT) as u32), pane, edge, &label, 12.0, c, -screen(p), 6.0, true);
        }
    }
}

/// The plane of the outputs: the ellipse A makes of the unit circle, u₁ and u₂ with their
/// lines, b, p and the left-over b − p.
fn draw_outputs(f: &mut Frame, room: &mut Room, lay: &Layout, ans: &Answers, labels: bool) {
    let pane = &lay.panes[1];
    backdrop(f, room, pane, 1, ["$b_1$", "$b_2$"], labels);
    let svd = ans.svd;
    let a = svd.a();
    let [u1, u2] = svd.u;
    let rank = ans.rank;

    // the ellipse: where A takes the unit circle
    // the lines of u₁ and u₂: the column space when σ₂ = 0
    let strong = |on: bool| if on { 0.9 } else { 0.4 };
    line_through_zero(f, room, pane, u1, if rank == 1 { 2.0 } else { 1.2 }, FIRST.a(strong(rank == 1)));
    line_through_zero(f, room, pane, u2, 1.2, SECOND.a(0.4));

    // the ellipse, over them: where A takes the unit circle (thicker as it flattens, so
    // that a flat one shows on the column space as the stretch of it that it is)
    let ring: Vec<Vec2> = (0..=96).map(|i| pane.to(a * DVec2::from_angle(i as f64 / 96.0 * std::f64::consts::TAU))).collect();
    let middle = pane.to(DVec2::ZERO);
    for pair in ring.windows(2) {
        f.dl.tris_alpha.tri(middle.extend(0.0), pair[0].extend(0.0), pair[1].extend(0.0), CIRCLE.a(0.12));
    }
    let thin = (svd.s[1] * pane.scale as f64 / 8.0).min(1.0) as f32;
    f.path(ring.iter().copied(), 1.6 + 2.4 * (1.0 - thin), CIRCLE.a(0.8));
    for pair in ring.windows(9).step_by(8) {
        room.line(f, pair[0].extend(0.0), pair[8].extend(0.0));
    }
    // σ₁u₁ and σ₂u₂, the ellipse's half-axes (their tips are handles)
    let (h1, h2) = (u1 * svd.s[0], u2 * svd.s[1]);
    arrow(f, pane, DVec2::ZERO, h1, true, 2.2, FIRST);
    arrow(f, pane, DVec2::ZERO, h2, true, 2.2, SECOND);
    for p in [h1, h2, ans.b] {
        keep_off(room, pane, p, 18.0);
    }

    // the left-over b − p, at right angles to the column space, and p (copies by rank)
    for r in 0..3 {
        if let Some((p, q)) = seg(f, pane, ans.p, ans.b, rank == r && r < 2, 2.4, LEFT) {
            room.line(f, p.extend(0.0), q.extend(0.0));
        }
    }
    let leftover = ans.b - ans.p;
    for r in 0..3 {
        corner(f, pane, ans.p, ans.b - ans.p, u1, rank == r && r == 1 && leftover.length() * pane.scale as f64 > 24.0, 10.0, pal::WHITE.a(0.8));
        dot(f, pane, ans.p, rank == r && r < 2, 10.0, PINV);
    }
    keep_off(room, pane, ans.p, 14.0);
    // where A takes the cut-off and the damped answers (copies, as the answers)
    for k in 0..3 {
        dot(f, pane, svd.reached(Undo::Cut(ans.tau), ans.b), ans.cutting && ans.kept == k, 7.0, CUT);
    }
    dot(f, pane, svd.reached(Undo::Damp(ans.lambda), ans.b), ans.damping, 6.0, DAMP);
    if !labels {
        return;
    }

    let line_names = match rank {
        1 => ["column space", "$u_2$"],
        2 => ["$u_1$", "$u_2$"],
        _ => ["", ""],
    };
    for (i, (d, text, c)) in [(u1 * ENDS[2], line_names[0], FIRST), (u2 * ENDS[3], line_names[1], SECOND)].into_iter().enumerate() {
        if !text.is_empty() {
            line_name(f, room, ("out line", i as u32), pane, d, text, c);
        }
    }
    let seen_px = |v: DVec2| v.length() * pane.scale as f64;
    if seen_px(h1) > 24.0 {
        name(f, room, ("tip", 0), pane, h1, r"$\sigma_1 u_1$", 13.0, FIRST, -up_from(u1), 9.0, true);
    }
    if seen_px(h2) > 24.0 {
        name(f, room, ("tip", 1), pane, h2, r"$\sigma_2 u_2$", 13.0, SECOND, -screen(u1), 9.0, true);
    }
    // b's name across the column space from it, on b's side (it is often near the line,
    // and a name beside it along the line would have to change sides as it moves); p's
    // on the other side
    let side = screen(u2) * if ans.b.dot(u2) < 0.0 { -1.0 } else { 1.0 };
    name(f, room, ("b", 0), pane, ans.b, "$b$", 16.0, POINT, side, gap_off_line(side, "$b$", 16.0, 9.0), false);
    if rank < 2 {
        let p_text = if rank == 1 { "$p = AA^+b$" } else { "$p = 0$" };
        name(f, room, ("p", 0), pane, ans.p, p_text, 13.5, PINV.mix(pal::WHITE, 0.35), -side, gap_off_line(-side, p_text, 13.5, 8.0), false);
        if seen_px(leftover) > 40.0 {
            name(f, room, ("left", 0), pane, (ans.p + ans.b) * 0.5, "$b - p$", 13.0, LEFT, screen(leftover.perp()), 8.0, true);
        }
    }
    if seen_px(h1) > 40.0 {
        // (on the far side from σ₁u₁, where the other names are fewer)
        let at = if rank == 1 { -h1 * 0.5 } else { a * DVec2::from_angle(2.4) };
        let text = if rank == 1 { "the circle, squashed flat" } else { "the circle, carried by $A$" };
        let (towards, gap) = if rank == 1 { (up_from(u1), gap_off_line(up_from(u1), text, 12.0, 8.0)) } else { (screen(at), 8.0) };
        name(f, room, ("ellipse", rank as u32), pane, at, text, 12.0, CIRCLE.a(0.9), towards, gap, true);
    }
}

/// The arrows between the panes: A takes the inputs to the outputs, A⁺ brings them back.
fn between(f: &mut Frame, lay: &Layout) {
    let [a, b] = lay.panes;
    let blue = PINV.mix(pal::WHITE, 0.25);
    if lay.narrow {
        let (y0, y1) = (a.hi.y + 6.0, b.lo.y - 6.0);
        if y1 - y0 < 18.0 {
            return;
        }
        let (x1, x2) = (a.hi.x - 64.0, a.hi.x - 26.0);
        hud_arrow(f, Vec2::new(x1, y0), Vec2::new(x1, y1), 1.8, pal::TEXT);
        f.text_px(Vec2::new(x1 - 7.0, (y0 + y1) * 0.5), "$A$", 14.0, pal::TEXT, Align2::RIGHT_CENTER);
        hud_arrow(f, Vec2::new(x2, y1), Vec2::new(x2, y0), 1.8, blue);
        f.text_px(Vec2::new(x2 + 7.0, (y0 + y1) * 0.5), "$A^+$", 14.0, blue, Align2::LEFT_CENTER);
    } else {
        let (x0, x1) = (a.hi.x + 6.0, b.lo.x - 6.0);
        if x1 - x0 < 18.0 {
            return;
        }
        let mid = (a.lo.y + a.hi.y) * 0.5;
        let (y1, y2) = (mid - 26.0, mid + 26.0);
        hud_arrow(f, Vec2::new(x0, y1), Vec2::new(x1, y1), 1.8, pal::TEXT);
        f.text_px(Vec2::new((x0 + x1) * 0.5, y1 - 7.0), "$A$", 14.0, pal::TEXT, Align2::CENTER_BOTTOM);
        hud_arrow(f, Vec2::new(x1, y2), Vec2::new(x0, y2), 1.8, blue);
        f.text_px(Vec2::new((x0 + x1) * 0.5, y2 + 7.0), "$A^+$", 14.0, blue, Align2::CENTER_TOP);
    }
}

/// What each pane shows, above it.
fn titles(f: &mut Frame, lay: &Layout) {
    let choices: [&[&str]; 2] = [&["the inputs $x$", "inputs $x$", "$x$"], &["the outputs $Ax$, and $b$ (drag it)", "the outputs $Ax$, and $b$", "outputs, $b$", "$b$"]];
    for (pane, texts) in lay.panes.iter().zip(choices) {
        let room = pane.hi.x - pane.lo.x - if lay.narrow { 110.0 } else { 4.0 };
        if let Some(t) = texts.iter().find(|t| text_width(t, 13.0) <= room) {
            f.text_px(Vec2::new(pane.lo.x + 2.0, pane.lo.y - 6.0), *t, 13.0, pal::TEXT, Align2::LEFT_BOTTOM);
        }
    }
}

impl Pinv {
    /// The heading over the panes: what A is now, and what that means (the second line
    /// left out on a phone).
    fn heading(&mut self, f: &mut Frame, at: Vec2, width: f32, s: &Setting, narrow: bool) {
        let (first, second) = verdict(s);
        let kept = self.heading.as_ref().filter(|h| h.0 == first && h.1 == second && h.2 == width).map(|h| h.3);
        let (big, small) = match kept {
            Some(sizes) => sizes,
            None => {
                let fits = |text: &str, size: f32| crate::typeset::measure(f.ctx, text, size).map(|m| m.x <= width);
                match (fits(&first, 15.0), fits(&first, 13.0), fits(&second, 12.5)) {
                    (Some(a), Some(b), Some(c)) => {
                        let sizes = (if a { 15.0 } else if b { 13.0 } else { 11.5 }, if c { 12.5 } else { 11.0 });
                        self.heading = Some((first.clone(), second.clone(), width, sizes));
                        sizes
                    }
                    // (the formulas' fonts not in place yet: the first frame)
                    _ => (13.0, 11.0),
                }
            }
        };
        f.text_px(at, first.as_str(), big, pal::TEXT, Align2::LEFT_TOP);
        if !narrow {
            f.text_px(at + Vec2::new(0.0, big + 7.0), second.as_str(), small, pal::DIM, Align2::LEFT_TOP);
        }
    }
}

/// What A is, in a line, and what that means for Ax = b, in another.
fn verdict(s: &Setting) -> (String, String) {
    let svd = s.svd();
    match svd.rank() {
        0 => (r"$A = 0$: it sends every $x$ to $0$".into(), r"so $A^+ = 0$ too: $p = 0$ is all $A$ reaches, and $x^+ = 0$ the shortest $x$ that does".into()),
        1 => (
            r"$\sigma_2 = 0$: $A$ squashes the plane onto a line, and has no inverse".into(),
            r"$A^+b$ is the shortest $x$ that $A$ takes to $p$, the point of the line nearest to $b$".into(),
        ),
        _ if s.s2 >= s.s1 => (
            format!(r"$\sigma_1 = \sigma_2 = {}$: $A$ has an inverse, and $A^+ = A^{{-1}}$", value(s.s1)),
            r"$A$ stretches every direction alike, so any two directions at right angles serve as $v_1$ and $v_2$".into(),
        ),
        _ if s.s2 >= 0.25 * s.s1 => (format!(r"$\sigma_2 = {}$: $A$ has an inverse, and $A^+ = A^{{-1}}$", value(s.s2)), r"every $b$ is reached by exactly one $x$, $x^+ = A^{-1}b$".into()),
        _ => (
            format!(r"$\sigma_2 = {}$: $A$ has an inverse, but $1/\sigma_2 = {}$", value(s.s2), short(1.0 / s.s2)),
            format!(r"$A^{{-1}}$ multiplies $b$'s part along $u_2$ by ${}$ to make $x$'s part along $v_2$: noise there grows ${}$ times", short(1.0 / s.s2), short(1.0 / s.s2)),
        ),
    }
}

/// The chart of what each σ becomes: 1/σ (and 0 for 0), cut off below τ, damped; with
/// σ₁ and σ₂ marked, and what σ₂ becomes under each, written in its corner.
fn chart(f: &mut Frame, lay: &Layout, (lo, hi): (Vec2, Vec2), s: &Setting, ans: &Answers) {
    let st = |px: Vec2| lay.stage(px);
    let (a, b) = (st(lo), st(hi));
    f.dl.tris_alpha.rect(a.min(b), a.max(b), 0.0, BACKING.a(0.9));
    frame_rect(f, a.min(b), a.max(b), 1.0, pal::FAINT.a(0.7));
    let small = hi.x - lo.x < 420.0;
    let text = if small { 10.5 } else { 11.5 };
    let (left, right, top, bottom) = (lo.x + 30.0, hi.x - 12.0, lo.y + if small { 8.0 } else { 24.0 }, hi.y - 20.0);
    if right - left < 40.0 || bottom - top < 30.0 {
        return;
    }
    let at = |sigma: f64, y: f64| st(Vec2::new(left + (right - left) * (sigma / S_MAX).clamp(0.0, 1.0) as f32, bottom - (bottom - top) * (y / Y_TOP).clamp(0.0, 1.0) as f32));
    if !small {
        f.text_px(Vec2::new(lo.x + 8.0, lo.y + 6.0), r"what each $\sigma$ becomes", text + 0.5, pal::TEXT, Align2::LEFT_TOP);
    }
    // the axes, and their numbers
    f.line(at(0.0, 0.0), at(S_MAX, 0.0), 1.2, pal::DIM.a(0.8));
    f.line(at(0.0, 0.0), at(0.0, Y_TOP), 1.2, pal::DIM.a(0.8));
    let x_of = |v: f64| left + (right - left) * (v / S_MAX).clamp(0.0, 1.0) as f32;
    let (x1, x2) = (x_of(s.s1), x_of(s.s2));
    for k in 1..=4 {
        let x = x_of(k as f64 * 0.5);
        // (not where the name of σ₁ or σ₂ is)
        if (x - x1).abs() > 22.0 && (x - x2).abs() > 22.0 {
            f.text_px(Vec2::new(x, bottom + 3.0), fmt_tick(k as f32 * 0.5, 0.5), 10.0, pal::DIM, Align2::CENTER_TOP);
        }
    }
    for k in 1..=Y_TOP as usize {
        let y = bottom - (bottom - top) * (k as f32 / Y_TOP as f32);
        f.text_px(Vec2::new(left - 5.0, y), k.to_string(), 10.0, pal::DIM, Align2::RIGHT_CENTER);
    }
    f.text_px(Vec2::new(left - 5.0, bottom), "0", 10.0, pal::DIM, Align2::RIGHT_CENTER);
    f.text_px(Vec2::new(right + 2.0, bottom - 3.0), r"$\sigma$", 12.0, pal::DIM, Align2::RIGHT_BOTTOM);

    // σ₁ and σ₂, marked
    for (v, c) in [(s.s1, FIRST), (s.s2, SECOND)] {
        f.line(at(v, 0.0), at(v, Y_TOP), 1.2, c.a(0.55));
    }
    // 1/σ (blue), and 0 at σ = 0
    let n = 64;
    let from = 1.0 / Y_TOP;
    f.path((0..=n).map(|i| from + (S_MAX - from) * i as f64 / n as f64).map(|x| at(x, 1.0 / x)), 2.6, PINV);
    f.dot(at(0.0, 0.0), 8.0, PINV);
    // damped (green): σ/(σ² + λ)
    let damp = Undo::Damp(s.lambda);
    f.path((0..=n).map(|i| S_MAX * i as f64 / n as f64).map(|x| at(x, damp.factor(x))), 2.0, DAMP.a(if ans.damping { 1.0 } else { 0.0 }));
    // cut off (cyan): 0 below τ, a jump at τ, then 1/σ (over the blue)
    let tau = s.tau.min(S_MAX);
    let cut_alpha = if ans.cutting { 1.0 } else { 0.0 };
    f.path([at(0.0, 0.0), at(tau, 0.0), at(tau, if tau > 0.0 { 1.0 / tau } else { Y_TOP })], 2.0, CUT.a(cut_alpha));
    let start = tau.max(from);
    f.path((0..=n).map(|i| start + (S_MAX - start) * i as f64 / n as f64).map(|x| at(x, 1.0 / x)), 1.2, CUT.a(cut_alpha));

    // what σ₂ becomes under each (copies for the jumps at 0 and at τ)
    let s2 = s.s2;
    for zero in [false, true] {
        let here = (s2 <= 0.0) == zero;
        let p = if zero { at(0.0, 0.0) } else { at(s2, 1.0 / s2.max(1e-300)) };
        f.dot(if here { p } else { NOWHERE }, 9.0, PINV.a(if here { 1.0 } else { 0.0 }));
    }
    let cut = Undo::Cut(s.tau);
    for kept in [false, true] {
        let here = ans.cutting && cut.keeps(s2) == kept;
        let p = at(s2, cut.factor(s2));
        f.dot(if here { p } else { NOWHERE }, 8.0, CUT.a(if here { 1.0 } else { 0.0 }));
    }
    f.dot(if ans.damping { at(s2, damp.factor(s2)) } else { NOWHERE }, 7.0, DAMP.a(if ans.damping { 1.0 } else { 0.0 }));

    // the σ's names under the axis (kept apart when they are close)
    let close = (x1 - x2).abs() < 30.0;
    f.text_px(Vec2::new(x1 + if close { 3.0 } else { 0.0 }, bottom + 3.0), r"$\sigma_1$", 11.5, FIRST, if close { Align2::LEFT_TOP } else { Align2::CENTER_TOP });
    f.text_px(Vec2::new(x2 - if close { 3.0 } else { 0.0 }, bottom + 3.0), r"$\sigma_2$", 11.5, SECOND, if close { Align2::RIGHT_TOP } else { Align2::CENTER_TOP });

    // what σ₂ becomes, in the top right-hand corner
    let mut lines: Vec<(String, Color)> = Vec::new();
    lines.push((if s2 > 0.0 { format!(r"$\sigma_2 = {}$ becomes:", value(s2)) } else { r"$\sigma_2 = 0$ becomes:".to_string() }, SECOND));
    lines.push((if s2 > 0.0 { format!(r"$1/\sigma_2 = {}$ in $A^+$", short(1.0 / s2)) } else { r"$0$ in $A^+$, not $1/0$".into() }, PINV.mix(pal::WHITE, 0.3)));
    if ans.cutting {
        let v = cut.factor(s2);
        let v = if v == 0.0 { "0".to_string() } else { short(v) };
        lines.push((format!(r"${v}$ cut off at $\tau = {}$", value(s.tau)), CUT));
    }
    if ans.damping {
        lines.push((format!(r"${}$ damped, $\lambda = {}$", short(damp.factor(s2)), value(s.lambda)), DAMP));
    }
    let x = right - 6.0;
    let line = text + 7.0;
    for (i, (t, c)) in lines.iter().enumerate() {
        f.caption_px(Vec2::new(x, top + 4.0 + i as f32 * line), t.as_str(), text, *c, Align2::RIGHT_TOP);
    }
}

/// A matrix of numbers on the screen, as `hud_matrix` draws it, with a number too wide
/// for its cell (a large power of ten) drawn smaller, to fit: so that the brackets stay
/// where they are whatever the numbers.
fn matrix(f: &mut Frame, top_left: Vec2, m: DMat2, color: Color, size: f32) -> Vec2 {
    let rows: Vec<Vec<(String, Color)>> = (0..2).map(|i| (0..2).map(|j| (cell(m.col(j)[i]), color)).collect()).collect();
    let room = matrix_cell(CELL, size).x - 2.0;
    let mut fitted = rows.clone();
    let mut smaller = Vec::new();
    for (i, row) in fitted.iter_mut().enumerate() {
        for (j, (text, c)) in row.iter_mut().enumerate() {
            if !text.contains('e') {
                continue;
            }
            let wide = f.text_size(text, size).x;
            if wide > room {
                smaller.push((i, j, std::mem::take(text), *c, size * room / wide));
            }
        }
    }
    let total = hud_matrix(f, top_left, &fitted, CELL, size, pal::DIM);
    for (i, j, text, c, s) in smaller {
        let (lo, hi) = matrix_entry(top_left, CELL, size, i, j);
        f.text_px((lo + hi) * 0.5, text, s, c, Align2::CENTER_CENTER);
    }
    total
}

/// A, A⁺, AA⁺ and A⁺A written out, with what the last two are.
fn matrices(f: &mut Frame, (lo, hi): (Vec2, Vec2), ans: &Answers) {
    hud_panel(f, lo, hi);
    let size = MAT_TEXT;
    let svd = ans.svd;
    let (a, pinv) = (svd.a(), svd.undo(Undo::Exact));
    let m = matrix_size(2, 2, CELL, size);
    let lead = text_width("$A^+A =$", size) + 8.0;
    let col = lead + m.x + 28.0;
    let (x0, y0) = (lo.x + 12.0, lo.y + 10.0);
    let blue = PINV.mix(pal::WHITE, 0.3);
    let rows = [[("$A =$", a, pal::TEXT), ("$A^+ =$", pinv, blue)], [("$AA^+ =$", svd.onto_columns(), pal::TEXT), ("$A^+A =$", svd.onto_rows(), pal::TEXT)]];
    for (r, row) in rows.iter().enumerate() {
        let y = y0 + r as f32 * (m.y + 34.0);
        for (c, (lead_text, mat, colour)) in row.iter().enumerate() {
            let x = x0 + c as f32 * col;
            f.text_px(Vec2::new(x + lead - 6.0, y + m.y * 0.5), *lead_text, size, if c == 1 && r == 0 { blue } else { pal::TEXT }, Align2::RIGHT_CENTER);
            matrix(f, Vec2::new(x + lead, y), *mat, *colour, size);
        }
    }
    // what the products are
    let y = y0 + 2.0 * m.y + 34.0 + 6.0;
    let small = 11.5;
    let (left, right) = match ans.rank {
        2 => (r"$AA^+ = I$: $A^+ = A^{-1}$", "$A^+A = I$"),
        1 => ("projects onto the column space", "projects onto the row space"),
        _ => ("$AA^+ = 0$", "$A^+A = 0$"),
    };
    f.text_px(Vec2::new(x0, y), left, small, pal::DIM, Align2::LEFT_TOP);
    f.text_px(Vec2::new(x0 + col, y), right, small, pal::DIM, Align2::LEFT_TOP);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A matrix by its rows.
    /// Ax, by the SVD: σᵢ times x's part along vᵢ, along uᵢ.
    fn apply(svd: &Svd, x: DVec2) -> DVec2 {
        svd.u[0] * (svd.s[0] * svd.v[0].dot(x)) + svd.u[1] * (svd.s[1] * svd.v[1].dot(x))
    }

    fn close(m: DMat2, n: DMat2, tol: f64) -> bool {
        (m - n).to_cols_array().iter().all(|v| v.abs() <= tol)
    }

    /// Settings of every kind: random ones, and the degenerate ones (A = 0, σ₂ = 0, σ₂
    /// as small as a slider can make it, σ₁ = σ₂, b = 0, a cut-off above both σ).
    fn settings() -> Vec<Setting> {
        let mut rng = Rng::new(7);
        let mut out: Vec<Setting> = TRIALS.iter().map(Trial::setting).collect();
        for _ in 0..400 {
            let s1 = rng.range(0.0, S_MAX as f32) as f64;
            let s2 = match rng.below(3) {
                0 => 0.0,
                1 => s1 * rng.f32() as f64 * 0.05,
                _ => s1 * rng.f32() as f64,
            };
            let b = DVec2::new(rng.range(-2.4, 2.4) as f64, rng.range(-2.4, 2.4) as f64);
            out.push(Setting { s1, s2, tu: rng.range(-4.0, 4.0) as f64, tv: rng.range(-4.0, 4.0) as f64, b, tau: rng.range(0.0, 1.0) as f64, lambda: rng.range(0.0, 1.0) as f64 });
        }
        let base = TRIALS[0].setting();
        for (s1, s2) in [(0.0, 0.0), (1.2, 0.0), (1.2, f32::from_bits(1) as f64), (1.2, 1e-30), (2.0, 2.0), (0.7, 0.7), (S_MAX, 1e-9)] {
            for b in [DVec2::ZERO, base.b, DVec2::splat(B_MAX), DVec2::new(-B_MAX, B_MAX)] {
                for (tau, lambda) in [(0.0, 0.0), (1.0, 1.0), (0.001, 0.0001)] {
                    out.push(Setting { s1, s2, b, tau, lambda, ..base });
                }
            }
        }
        out
    }

    /// Whether A's entries hold σ₂ to some digits (they cannot once it is under about a
    /// millionth of a millionth of σ₁, and products of A's entries are then rounding).
    fn held(s: &Setting) -> bool {
        s.s2 == 0.0 || s.s2 >= 1e-6 * s.s1
    }

    /// A⁺ is what it must be, for every setting: A A⁺ A = A and A⁺ A A⁺ = A⁺; A A⁺ and
    /// A⁺ A are symmetric, and projections (twice is once) onto the column space and the
    /// row space; A⁺ = A⁻¹ when A has one, and 0 when A = 0. (By the SVD for every σ₂, as
    /// the scene works them out; and as products of the entries where they hold σ₂.)
    #[test]
    fn the_pseudoinverse_is_what_it_should_be() {
        let mut rng = Rng::new(3);
        for s in settings() {
            let svd = s.svd();
            // σ σ⁺ σ = σ and σ⁺ σ σ⁺ = σ⁺, for each σ: A A⁺ A = A and A⁺ A A⁺ = A⁺
            for v in svd.s {
                let w = Undo::Exact.factor(v);
                assert!((v * w * v - v).abs() <= 1e-15 * v && (w * v * w - w).abs() <= 1e-15 * w, "{s:?}");
            }
            for _ in 0..10 {
                let x = DVec2::new(rng.normal() as f64, rng.normal() as f64);
                let (cols, rows) = (svd.onto_columns(), svd.onto_rows());
                assert!((svd.reached(Undo::Exact, x) - cols * x).length() < 1e-12, "{s:?}");
                assert!((cols * (cols * x) - cols * x).length() < 1e-12 && (rows * (rows * x) - rows * x).length() < 1e-12);
                assert!(close(cols, cols.transpose(), 1e-15) && close(rows, rows.transpose(), 1e-15));
            }
            if !held(&s) {
                continue;
            }
            let (a, p) = (svd.a(), svd.undo(Undo::Exact));
            let (size_a, size_p) = (a.abs().to_cols_array().iter().cloned().fold(1.0, f64::max), p.abs().to_cols_array().iter().cloned().fold(1.0, f64::max));
            let tol = 1e-12 * size_a * size_p;
            assert!(close(a * p * a, a, tol * size_a), "{s:?}");
            assert!(close(p * a * p, p, tol * size_p), "{s:?}");
            let (ap, pa) = (a * p, p * a);
            for m in [ap, pa] {
                assert!(close(m, m.transpose(), tol) && close(m * m, m, tol), "{s:?}: {m:?}");
            }
            assert!(close(ap, svd.onto_columns(), tol) && close(pa, svd.onto_rows(), tol), "{s:?}");
            match svd.rank() {
                2 => {
                    assert!(close(ap, DMat2::IDENTITY, tol) && close(pa, DMat2::IDENTITY, tol), "{s:?}");
                    let inv = a.inverse();
                    assert!(close(p, inv, 1e-9 * size_p), "{s:?}: {p:?} against {inv:?}");
                }
                0 => assert!(p == DMat2::ZERO && a == DMat2::ZERO),
                _ => assert!(a.determinant().abs() <= 1e-12 * size_a * size_a, "{s:?}"),
            }
            assert!(p.to_cols_array().iter().all(|v| v.is_finite()), "{s:?}");
        }
    }

    /// The pseudoinverse of an `m` by `n` matrix (by rows) from the general SVD of
    /// `math::linalg`: an independent route to the same thing. A singular value under
    /// a trillionth of the largest is rounding, and taken as 0.
    fn general_pinv(a: &[f64], m: usize, n: usize) -> Vec<f64> {
        let (u, s, v) = crate::math::linalg::svd(a, m, n);
        let r = m.min(n);
        let big = s.iter().cloned().fold(0.0, f64::max);
        let mut out = vec![0.0; n * m];
        for k in 0..r {
            if s[k] > 1e-12 * big && s[k] > 0.0 {
                for j in 0..n {
                    for i in 0..m {
                        out[j * m + i] += v[j * r + k] * u[i * r + k] / s[k];
                    }
                }
            }
        }
        out
    }

    /// The product of an `m` by `k` and a `k` by `n` matrix, by rows.
    fn times(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        (0..m * n).map(|ij| (0..k).map(|l| a[ij / n * k + l] * b[l * n + ij % n]).sum()).collect()
    }

    fn transposed(a: &[f64], m: usize, n: usize) -> Vec<f64> {
        (0..n * m).map(|ji| a[ji % m * n + ji / m]).collect()
    }

    fn inverse2(a: &[f64]) -> Vec<f64> {
        let det = a[0] * a[3] - a[1] * a[2];
        vec![a[3] / det, -a[1] / det, -a[2] / det, a[0] / det]
    }

    /// The scene's A⁺ is the general one; and what the panel and the primer say of
    /// matrices of other shapes holds: with independent columns A⁺ = (AᵀA)⁻¹Aᵀ, with
    /// independent rows A⁺ = Aᵀ(AAᵀ)⁻¹. And the primer's example: the one equation
    /// x + 2y = 2 has the shortest answer (0.4, 0.8).
    #[test]
    fn the_formulas_for_independent_columns_and_rows() {
        for s in settings() {
            let svd = s.svd();
            let a = svd.a();
            let by_rows = [a.x_axis.x, a.y_axis.x, a.x_axis.y, a.y_axis.y];
            let size = by_rows.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            // (the general SVD tells a zero σ from rounding only where σ₂ is not tiny)
            if size == 0.0 || (s.s2 > 0.0 && s.s2 < 1e-6 * size) {
                continue;
            }
            let p = svd.undo(Undo::Exact);
            let general = general_pinv(&by_rows, 2, 2);
            let scale = p.abs().to_cols_array().iter().cloned().fold(1.0, f64::max);
            for (mine, theirs) in [p.x_axis.x, p.y_axis.x, p.x_axis.y, p.y_axis.y].iter().zip(&general) {
                assert!((mine - theirs).abs() < 1e-7 * scale, "{s:?}: {p:?} against {general:?}");
            }
        }
        let mut rng = Rng::new(9);
        for _ in 0..200 {
            // three equations in two unknowns, and two in three
            let tall: Vec<f64> = (0..6).map(|_| rng.normal() as f64).collect();
            let ata = times(&transposed(&tall, 3, 2), &tall, 2, 3, 2);
            let columns = times(&inverse2(&ata), &transposed(&tall, 3, 2), 2, 2, 3);
            for (x, y) in general_pinv(&tall, 3, 2).iter().zip(&columns) {
                assert!((x - y).abs() < 1e-8 * y.abs().max(1.0), "{tall:?}");
            }
            let wide: Vec<f64> = (0..6).map(|_| rng.normal() as f64).collect();
            let aat = times(&wide, &transposed(&wide, 2, 3), 2, 3, 2);
            let rows_formula = times(&transposed(&wide, 2, 3), &inverse2(&aat), 3, 2, 2);
            for (x, y) in general_pinv(&wide, 2, 3).iter().zip(&rows_formula) {
                assert!((x - y).abs() < 1e-8 * y.abs().max(1.0), "{wide:?}");
            }
        }
        let one = general_pinv(&[1.0, 2.0], 1, 2);
        let x = [one[0] * 2.0, one[1] * 2.0];
        assert!((x[0] - 0.4).abs() < 1e-12 && (x[1] - 0.8).abs() < 1e-12, "{x:?}");
        assert!((one[0] - 0.2).abs() < 1e-12 && (one[1] - 0.4).abs() < 1e-12);
    }

    /// x⁺ = A⁺b comes as near to b as any x can (A x⁺ = p, with b − p at right angles to
    /// the column space), and is the shortest x that does: every other one is x⁺ plus a
    /// step along the null space, and |x|² = |x⁺|² + |step|². It has no part in the null
    /// space.
    #[test]
    fn the_answer_is_the_nearest_and_then_the_shortest() {
        let mut rng = Rng::new(5);
        for s in settings() {
            let ans = Answers::of(&s);
            let svd = ans.svd;
            let a = svd.a();
            let best = (ans.p - ans.b).length();
            if held(&s) {
                assert!((a * ans.x - ans.p).length() < 1e-9 * ans.x.length().max(1.0), "{s:?}");
            }
            for _ in 0..20 {
                let x = DVec2::new(rng.range(-5.0, 5.0) as f64, rng.range(-5.0, 5.0) as f64);
                assert!((apply(&svd, x) - ans.b).length() >= best - 1e-9, "{s:?}: {x:?} comes nearer");
            }
            match ans.rank {
                1 => {
                    assert!((ans.b - ans.p).dot(svd.u[0]).abs() < 1e-9 && ans.x.dot(svd.v[1]).abs() < 1e-9 * ans.x.length().max(1.0), "{s:?}");
                    for t in [-2.0, -0.3, 0.5, 1.0, 4.0] {
                        let other = ans.x + svd.v[1] * t;
                        assert!((a * other - ans.p).length() < 1e-9 * ans.x.length().max(1.0) && (apply(&svd, other) - ans.p).length() < 1e-9 * ans.x.length().max(1.0), "{s:?}");
                        assert!((other.length_squared() - ans.x.length_squared() - t * t).abs() < 1e-9 * ans.x.length_squared().max(1.0));
                    }
                }
                2 => assert!((ans.p - ans.b).length() < 1e-9 * ans.x.length().max(1.0), "{s:?}"),
                _ => assert!(ans.x == DVec2::ZERO && ans.p == DVec2::ZERO),
            }
        }
    }

    /// The damped answer is ridge's, (AᵀA + λI)⁻¹Aᵀb; the cut-off answer turns upside
    /// down only the σ at or above τ; and the factors the chart draws are these.
    #[test]
    fn damping_is_ridge_and_cutting_keeps_the_large() {
        for s in settings() {
            let ans = Answers::of(&s);
            let svd = ans.svd;
            let a = svd.a();
            if s.lambda > 1e-3 && held(&s) {
                let ridge = (a.transpose() * a + DMat2::IDENTITY * s.lambda).inverse() * a.transpose() * s.b;
                assert!((ridge - ans.damp).length() < 1e-9 * ridge.length().max(1.0), "{s:?}");
            }
            let kept: Vec<usize> = (0..2).filter(|&i| svd.s[i] > 0.0 && svd.s[i] >= s.tau).collect();
            let want = kept.iter().fold(DVec2::ZERO, |x, &i| x + svd.v[i] * (svd.u[i].dot(s.b) / svd.s[i]));
            assert!((want - ans.cut).length() < 1e-9 * want.length().max(1.0) && ans.kept == kept.len(), "{s:?}");
        }
        // a few factors by hand
        assert_eq!(Undo::Exact.factor(0.0), 0.0);
        assert_eq!(Undo::Exact.factor(0.25), 4.0);
        assert_eq!(Undo::Cut(0.3).factor(0.25), 0.0);
        assert_eq!(Undo::Cut(0.3).factor(0.3), 1.0 / 0.3);
        assert!((Undo::Damp(0.1).factor(0.25) - 0.25 / 0.1625).abs() < 1e-15);
        assert_eq!(Undo::Damp(0.0).factor(0.5), 2.0);
        assert_eq!(Undo::Damp(0.1).factor(0.0), 0.0);
    }

    /// The scene after experiment `k`'s button, at 60 frames a second, with what was
    /// shown in every frame of it (until it rests, or for 30 seconds at most).
    /// (Only the frames from when the glide to the experiment's setting has arrived.)
    fn run(k: usize) -> (Pinv, Vec<Setting>) {
        let mut scene = Pinv::create();
        scene.set_up(k);
        let mut seen = Vec::new();
        for _ in 0..30 * 60 {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if scene.glide.arrived() {
                seen.push(scene.shown());
            }
            if !scene.animated() {
                break;
            }
        }
        assert!(!scene.animated(), "experiment {k} does not come to rest");
        (scene, seen)
    }

    /// Each experiment ends exactly where it says, and at rest.
    #[test]
    fn every_experiment_arrives_and_rests() {
        for k in 0..TRIALS.len() {
            let (scene, _) = run(k);
            let (s, want) = (scene.shown(), TRIALS[k].setting());
            assert!(s.s1 == want.s1 && s.s2 == want.s2 && s.tau == want.tau && s.lambda == want.lambda, "{k}");
            assert!((s.b - want.b).length() < 1e-12 && (DVec2::from_angle(s.tu) - DVec2::from_angle(want.tu)).length() < 1e-12, "{k}");
            // (and from any other experiment, the same)
            for j in 0..TRIALS.len() {
                let mut from = Pinv::create();
                from.set_up(j);
                for _ in 0..600 {
                    from.advance(1.0 / 60.0, 1.0 / 60.0);
                }
                from.set_up(k);
                for _ in 0..1200 {
                    from.advance(1.0 / 60.0, 1.0 / 60.0);
                }
                let s = from.shown();
                assert!(!from.animated() && s.s2 == want.s2 && (s.b - want.b).length() < 1e-12, "{j} then {k}");
            }
        }
    }

    #[test]
    fn no_inverse_does_what_its_caption_says() {
        let (scene, _) = run(0);
        let s = scene.shown();
        let ans = Answers::of(&s);
        let svd = ans.svd;
        assert!(s.s2 == 0.0 && ans.rank == 1);
        // b is off the column space; p is its shadow, the left-over at right angles, 0.90
        assert!((ans.b - ans.p).length() > 0.5 && (ans.b - ans.p).dot(svd.u[0]).abs() < 1e-12);
        assert_eq!(num((ans.b - ans.p).length(), 2), "0.90");
        // x⁺, 1.50 long, on the row space; a step of 1 along the line, 1.80
        assert!(ans.x.perp_dot(svd.v[0]).abs() < 1e-12 && num(ans.x.length(), 2) == "1.50");
        assert_eq!(num((ans.x + svd.v[1]).length(), 2), "1.80");
        assert!(((1.5f64 * 1.5 + 1.0).sqrt() - (ans.x + svd.v[1]).length()).abs() < 1e-12);
        // the null space is what A sends to 0
        assert!((svd.a() * svd.v[1]).length() < 1e-15);
    }

    #[test]
    fn sliding_b_does_what_its_caption_says() {
        let (scene, seen) = run(1);
        let start = TRIALS[1].setting();
        let first = Answers::of(&start);
        let u2 = start.svd().u[1];
        let (mut lo, mut hi, mut crossings) = (f64::MAX, f64::MIN, 0);
        let mut last = start.b.dot(u2);
        for s in &seen {
            let ans = Answers::of(s);
            // p and x⁺ do not move: A⁺ sends u₂ to 0
            assert!((ans.x - first.x).length() < 1e-12 && (ans.p - first.p).length() < 1e-12);
            let along = s.b.dot(u2);
            (lo, hi) = (lo.min(along), hi.max(along));
            crossings += (along.signum() != last.signum()) as usize;
            last = along;
            assert!(((ans.b - ans.p).length() - along.abs()).abs() < 1e-12);
        }
        assert!((s_u2(&start) - 0.9).abs() < 1e-12);
        assert!(start.svd().undo(Undo::Exact) * u2 == DVec2::ZERO || (start.svd().undo(Undo::Exact) * u2).length() < 1e-15);
        // out to 1.80 on one side, 0.70 on the other, crossing the line twice (as drawn,
        // and as the film goes, between the frames)
        assert!(num(hi, 2) == "1.80" && num(lo, 2) == "−0.70" && crossings == 2, "{lo} {hi} {crossings}");
        let film = TRIALS[1].film.unwrap_or(Film { amp: 0.0, period: 1.0, length: 1.0 });
        let fine: Vec<f64> = (0..=60_000).map(|i| 0.9 + film.at(film.length * i as f32 / 60_000.0)).collect();
        let (most, least) = (fine.iter().cloned().fold(f64::MIN, f64::max), fine.iter().cloned().fold(f64::MAX, f64::min));
        assert!((most - 1.8).abs() < 1e-6 && (least + 0.7).abs() < 1e-6, "{least} {most}");
        // and back where it began
        assert!((scene.shown().b - start.b).length() < 1e-12);
    }

    fn s_u2(s: &Setting) -> f64 {
        s.b.dot(s.svd().u[1])
    }

    #[test]
    fn the_inverse_does_what_its_caption_says() {
        let (scene, _) = run(2);
        let s = scene.shown();
        let ans = Answers::of(&s);
        let svd = ans.svd;
        let (a, p) = (svd.a(), svd.undo(Undo::Exact));
        assert!(s.s2 == 0.9 && ans.rank == 2);
        assert!(close(p, a.inverse(), 1e-12) && close(a * p, DMat2::IDENTITY, 1e-12) && close(p * a, DMat2::IDENTITY, 1e-12));
        // what the screen shows of them
        for m in [a * p, p * a] {
            assert_eq!([m.x_axis.x, m.y_axis.x, m.x_axis.y, m.y_axis.y].map(cell), ["$1.00$", "$0.00$", "$0.00$", "$1.00$"]);
        }
        assert!((ans.p - ans.b).length() < 1e-12);
        assert!(num(s_u2(&s), 2) == "0.90" && num(ans.x.dot(svd.v[1]), 2) == "1.00");
        assert!((ans.x.dot(svd.v[1]) - s_u2(&s) / s.s2).abs() < 1e-12);
    }

    #[test]
    fn nearly_flat_does_what_its_caption_says() {
        let (scene, _) = run(3);
        let s = scene.shown();
        let ans = Answers::of(&s);
        assert!(s.s2 == 0.03 && ans.rank == 2 && s.tau == 0.1 && s.lambda == 0.04);
        // 0.90 divided by 0.03: 30.0 from 0, off the picture on any canvas
        assert!(num(s_u2(&s), 2) == "0.90" && num(ans.x.length(), 1) == "30.0");
        for view in [Vec2::new(1080.0, 1000.0), Vec2::new(1600.0, 1000.0), Vec2::new(390.0, 800.0), Vec2::new(580.0, 700.0), Vec2::new(2400.0, 700.0)] {
            assert!(!Layout::of(view, 0.0).panes[0].shows(ans.x), "{view}");
        }
        // the cut-off answer is the first experiment's x⁺, 1.50 long
        let first = Answers::of(&TRIALS[0].setting());
        assert!((ans.cut - first.x).length() < 1e-12 && num(ans.cut.length(), 2) == "1.50" && ans.kept == 1);
        // the damped one multiplies by 0.73 where the inverse does by 33.3, and is near
        assert!(num(Undo::Damp(0.04).factor(0.03), 2) == "0.73" && num(1.0 / 0.03, 1) == "33.3");
        assert!(ans.damp.length() < R_FIT);
    }

    #[test]
    fn noise_does_what_its_caption_says() {
        let (_, seen) = run(4);
        let start = TRIALS[4].setting();
        let svd = start.svd();
        let centre = Answers::of(&start);
        assert!(start.s2 == 0.25 && num(1.0 / start.s2, 0) == "4" && start.tau == 0.3 && start.lambda == 0.1);
        let (mut wobble, mut swing, mut damped) = (0.0f64, 0.0f64, 0.0f64);
        for s in &seen {
            let ans = Answers::of(s);
            wobble = wobble.max((s_u2(s) - s_u2(&start)).abs());
            swing = swing.max((ans.x - centre.x).length());
            damped = damped.max((ans.damp - centre.damp).length());
            // the cut-off answer does not move; neither it nor the damped one solves Ax = b
            assert!((ans.cut - centre.cut).length() < 1e-12 && ans.kept == 1);
            assert!((svd.a() * ans.cut - s.b).length() > 0.04 && (svd.a() * ans.damp - s.b).length() > 0.04);
            // the inverse's answer stays on the picture
            assert!(Layout::of(Vec2::new(390.0, 800.0), 46.0).panes[0].shows(ans.x));
        }
        assert!((wobble - 0.15).abs() < 1e-6 && num(swing, 2) == "0.60" && num(damped, 2) == "0.23", "{wobble} {swing} {damped}");
        assert_eq!(num(Undo::Damp(0.1).factor(0.25), 2), "1.54");
    }

    /// A value that is not 0 is never written as 0; a large one in powers of ten.
    #[test]
    fn numbers_are_written_truly() {
        assert_eq!([value(0.0), value(0.003), value(0.03), value(0.0257), value(1.5)], ["0.00", "3.0e−3", "0.03", "0.0257", "1.50"]);
        assert_eq!([short(-0.004), short(12345.0)], ["0.00", "1.2e4"]);
        assert_eq!([cell(0.5), cell(-12.34), cell(345.6), cell(7.0e44)], ["$0.50$", "$−12.3$", "$346$", "$7e44$"]);
        let (first, _) = verdict(&Setting { s2: 0.003, ..TRIALS[0].setting() });
        assert!(first.contains("3.0e−3") && !first.contains("0.00"), "{first}");
    }

    /// Whatever the setting, every number drawn or written is finite, and so is every
    /// place a thing is drawn at.
    #[test]
    fn degenerate_settings_leave_everything_finite() {
        for s in settings() {
            let ans = Answers::of(&s);
            for p in [ans.x, ans.p, ans.cut, ans.damp, ans.svd.a() * ans.cut] {
                assert!(p.is_finite(), "{s:?}");
            }
            let (first, second) = verdict(&s);
            for text in [first, second, short(ans.x.length()), point(ans.x)] {
                assert!(!text.contains("NaN") && !text.contains("inf") && !text.contains('?'), "{s:?}: {text}");
            }
            for v in ans.svd.undo(Undo::Exact).to_cols_array() {
                assert!(!cell(v).contains('?'), "{s:?}");
            }
            for lay in [Layout::of(Vec2::new(1080.0, 1000.0), 0.0), Layout::of(Vec2::new(390.0, 800.0), 46.0)] {
                let pane = lay.panes[0];
                if let Some((p, q)) = segment_in(DVec2::ZERO, ans.x, pane.seen()) {
                    assert!(pane.to(p).is_finite() && pane.to(q).is_finite(), "{s:?}");
                }
            }
        }
    }

    /// A point of a plane goes to the stage and back to itself; the panes keep to the
    /// canvas and off each other, show at least R_FIT every way (on a canvas big enough),
    /// and leave room for the chart and the matrices where they are drawn.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (1600.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (240.0, 180.0), (620.0, 480.0), (2400.0, 700.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let [a, b] = lay.panes;
                let inside = |lo: Vec2, hi: Vec2| lo.cmpge(Vec2::ZERO).all() && hi.cmple(view + Vec2::splat(0.5)).all();
                for p in [a, b] {
                    assert!(p.lo.x >= 0.0 && p.hi.x <= w && p.hi.y <= h + 0.5 && p.scale > 0.0 && p.scale.is_finite(), "{w}x{h}: {p:?}");
                    if w >= 300.0 && h >= 400.0 {
                        let (lo, hi) = p.seen();
                        assert!(inside(p.lo, p.hi) && hi.min_element() >= R_FIT && lo.max_element() <= -R_FIT, "{w}x{h}: {p:?}");
                        assert!(p.lo.y >= inset, "{w}x{h}: under the buttons");
                    }
                    for q in [DVec2::new(1.3, -0.7), DVec2::ZERO, DVec2::new(-2.0, 2.4)] {
                        assert!((p.back(p.to(q)) - q).length() < 1e-3, "{w}x{h}");
                    }
                }
                // side by side, or one above the other, never on top of each other
                assert!(a.hi.x <= b.lo.x || a.hi.y <= b.lo.y, "{w}x{h}");
                for (lo, hi) in lay.chart.into_iter().chain(lay.mats) {
                    assert!(inside(lo, hi) && lo.y >= a.hi.y.max(b.hi.y), "{w}x{h}");
                }
                if let (Some(c), Some(m)) = (lay.chart, lay.mats) {
                    assert!(c.1.x <= m.0.x && m.1.y <= h, "{w}x{h}");
                }
            }
        }
    }
}

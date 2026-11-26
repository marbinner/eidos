//! When eigenvectors run out: a repeated eigenvalue with a single eigenline. The best a
//! change of basis can do for such a matrix is the Jordan block J = [λ 1; 0 λ], so that
//! A = P J P⁻¹; its powers carry a shear that keeps growing, kλ^(k−1), on top of λᵏ; it
//! is where two eigenlines meet as they close like scissors; and a change ε below J's
//! diagonal moves its eigenvalues by √ε.

use super::*;
use crate::scenes::stats::plot::{self, Plot};
use glam::{DMat2, DVec2};

pub const DEF: SceneDef = SceneDef {
    id: "jordan",
    title: "When eigenvectors run out: Jordan blocks",
    group: FACTOR,
    blurb: "A repeated eigenvalue with too few eigenvectors: what such a matrix does, and why it is so sensitive.",
    make: |_| Box::new(Jordan::new()),
};

// ----- the arithmetic, in double precision -------------------------------------------

/// The steps drawn: x, Ax, …, Aᴷx.
const K: usize = 40;
/// How far λ and ε go either way.
const LAMBDA_MAX: f64 = 1.2;
const EPS_MAX: f64 = 0.25;
/// How far out the start x may be taken, and how long and how short v and w may be.
const X_MAX: f64 = 6.0;
const ARROW_MIN: f64 = 0.25;
const ARROW_MAX: f64 = 3.0;
/// The least |det P| / (|v| |w|), the sine of the angle between v and w (about 8°): P
/// stays well away from flattening the plane, so P⁻¹ stays of a size to write.
const APART: f64 = 0.14;
/// Steps a second while playing.
const RATE: f64 = 4.0;
/// The bar of the ε slider lets go onto 0 within this of it: the block itself.
const EPS_SNAP: f64 = 0.003;

/// The chain the scene opens with: v and w at right angles and of one length, so that
/// A behaves exactly as J does, turned; and the start x = 1.25 w.
const V0: [f64; 2] = [0.6, 0.8];
const W0: [f64; 2] = [-0.8, 0.6];
const X0: [f64; 2] = [-1.0, 0.75];

/// A matrix by its rows, [[a, b], [c, d]], the way matrices are written.
fn rows([a, b, c, d]: [f64; 4]) -> DMat2 {
    DMat2::from_cols(DVec2::new(a, c), DVec2::new(b, d))
}

/// J_ε = [λ 1; ε λ]: the Jordan block J with ε put below its diagonal (J itself at ε = 0).
fn block(lambda: f64, eps: f64) -> DMat2 {
    rows([lambda, 1.0, eps, lambda])
}

/// Jᵏ in closed form: λᵏ on the diagonal and kλ^(k−1) above it. (J = λI + N with N² = 0,
/// so the binomial theorem stops after two terms: Jᵏ = λᵏI + kλ^(k−1)N.)
fn jordan_power(lambda: f64, k: usize) -> DMat2 {
    let diagonal = lambda.powi(k as i32);
    let shear = if k == 0 { 0.0 } else { k as f64 * lambda.powi(k as i32 - 1) };
    rows([diagonal, shear, 0.0, diagonal])
}

/// What kind of matrix J_ε is (and A with it), by the sign of ε.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// ε > 0: two eigenvalues λ ± √ε, two eigenlines: diagonalisable.
    Two,
    /// ε = 0: λ twice with one eigenline: defective, a Jordan block.
    One,
    /// ε < 0: λ ± i√−ε, no eigenline: a turn with a stretch, in a suitable basis.
    Complex,
}

fn kind(eps: f64) -> Kind {
    if eps > 0.0 {
        Kind::Two
    } else if eps < 0.0 {
        Kind::Complex
    } else {
        Kind::One
    }
}

/// The eigenvalues of J_ε, as (real part, imaginary part), the larger first: λ ± √ε,
/// or λ ± i√−ε.
fn eigenvalues(lambda: f64, eps: f64) -> [(f64, f64); 2] {
    let r = eps.abs().sqrt();
    if eps >= 0.0 {
        [(lambda + r, 0.0), (lambda - r, 0.0)]
    } else {
        [(lambda, r), (lambda, -r)]
    }
}

/// The largest size of an eigenvalue.
fn spectral_radius(lambda: f64, eps: f64) -> f64 {
    eigenvalues(lambda, eps).iter().map(|(re, im)| re.hypot(*im)).fold(0.0, f64::max)
}

/// What is drawn: λ, ε, the chain v and w (P's columns) and the start x.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setting {
    lambda: f64,
    eps: f64,
    v: DVec2,
    w: DVec2,
    x: DVec2,
}

impl Setting {
    fn numbers(&self) -> [f64; 8] {
        [self.lambda, self.eps, self.v.x, self.v.y, self.w.x, self.w.y, self.x.x, self.x.y]
    }

    fn from_numbers(n: [f64; 8]) -> Setting {
        Setting { lambda: n[0], eps: n[1], v: DVec2::new(n[2], n[3]), w: DVec2::new(n[4], n[5]), x: DVec2::new(n[6], n[7]) }
    }

    fn p(&self) -> DMat2 {
        DMat2::from_cols(self.v, self.w)
    }
}

/// Whether v and w make a basis this scene will draw: each of a length between the
/// limits, and far enough from parallel.
fn chain_ok(v: DVec2, w: DVec2) -> bool {
    let (a, b) = (v.length(), w.length());
    let fine = |l: f64| l.is_finite() && (ARROW_MIN..=ARROW_MAX).contains(&l);
    fine(a) && fine(b) && v.perp_dot(w).abs() >= APART * a * b
}

/// Everything worked out from a setting: P, P⁻¹, J_ε, A = P J_ε P⁻¹, x in the basis v, w
/// (c = P⁻¹x), and the steps: their coordinates J_εᵏ c, and where they are, xₖ = P J_εᵏ c.
struct Shape {
    p: DMat2,
    p_inv: DMat2,
    b: DMat2,
    a: DMat2,
    c: DVec2,
    coords: [DVec2; K + 1],
    steps: [DVec2; K + 1],
}

impl Shape {
    fn of(s: &Setting) -> Shape {
        let p = s.p();
        let det = p.determinant();
        // (a setting always has a P well away from flat; this keeps the numbers finite if not)
        let p_inv = if det.abs() > 1e-12 { p.inverse() } else { DMat2::IDENTITY };
        let b = block(s.lambda, s.eps);
        let c = p_inv * s.x;
        let mut coords = [DVec2::ZERO; K + 1];
        coords[0] = c;
        for k in 1..=K {
            coords[k] = b * coords[k - 1];
        }
        let steps = coords.map(|q| p * q);
        Shape { p, p_inv, b, a: p * b * p_inv, c, coords, steps }
    }

    /// The step `k` as drawn between two whole steps: the straight way from one to the next.
    fn step_at(&self, k: f64) -> (DVec2, DVec2) {
        let k = k.clamp(0.0, K as f64);
        let i = (k.floor() as usize).min(K - 1);
        let u = k - i as f64;
        (self.steps[i].lerp(self.steps[i + 1], u), self.coords[i].lerp(self.coords[i + 1], u))
    }

    /// |xₖ| for every k.
    fn sizes(&self) -> [f64; K + 1] {
        self.steps.map(|p| p.length())
    }
}

/// The k at which |xₖ| is largest (the first, if two are equal), and that size.
fn largest(sizes: &[f64]) -> (usize, f64) {
    sizes.iter().enumerate().fold((0, f64::MIN), |best, (k, &n)| if n > best.1 { (k, n) } else { best })
}

/// The eigenvectors of A when ε > 0: v ± √ε w (the larger eigenvalue's first), at
/// length 1.
fn unit_eigenvectors(s: &Setting) -> Option<(DVec2, DVec2)> {
    if s.eps <= 0.0 {
        return None;
    }
    let g = s.eps.sqrt();
    let (u1, u2) = ((s.v + s.w * g).normalize_or_zero(), (s.v - s.w * g).normalize_or_zero());
    (u1 != DVec2::ZERO && u2 != DVec2::ZERO).then_some((u1, u2))
}

/// For ε > 0: the angle between the two eigenlines (degrees, from 0 to 90), |det V| for
/// V the matrix of unit eigenvectors (the yellow area, which is the sine of that angle),
/// and the largest entry of V⁻¹ in size. det V is worked out as it simplifies,
/// −2√ε det P / (|v + √ε w| |v − √ε w|), so that it is not the difference of two nearly
/// equal numbers when the lines nearly meet.
fn scissors(s: &Setting) -> Option<(f64, f64, f64)> {
    let (u1, u2) = unit_eigenvectors(s)?;
    let g = s.eps.sqrt();
    let det = (2.0 * g * s.p().determinant() / ((s.v + s.w * g).length() * (s.v - s.w * g).length())).abs().min(1.0);
    // V⁻¹ = [u2.y, −u2.x; −u1.y, u1.x] / det V
    let largest = [u2.y, u2.x, u1.y, u1.x].iter().fold(0.0f64, |m, e| m.max(e.abs())) / det.max(1e-300);
    Some((det.asin().to_degrees(), det, largest.min(1e300)))
}

/// A number for a matrix on the screen, as a formula, in five characters at most (so
/// that the brackets never move as the numbers change): two decimals, then one, then
/// none, and a power of ten beyond 9999. A number too small for two decimals but not 0
/// (an ε of 0.0001) is written in a power of ten too, rather than as 0.00.
fn cell(v: f64) -> String {
    let a = v.abs();
    let number = if !v.is_finite() {
        "?".into()
    } else if a != 0.0 && a < 0.005 {
        format!("{v:.0e}").replace('-', "−")
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

/// The widest `cell` can be, in characters of its number, a minus sign counted as two.
const CELL: usize = 6;

/// A small number as it is typed and read: up to four places after the point, and in
/// powers of ten below that (0.0001, 2.5e−5), 0 for 0.
fn small(v: f64) -> String {
    if v == 0.0 {
        "0".into()
    } else if v.abs() >= 0.0001 - 1e-12 {
        let text = format!("{v:.4}");
        let text = text.trim_end_matches('0');
        let text = text.strip_suffix('.').unwrap_or(text);
        text.replace('-', "−")
    } else {
        format!("{v:.1e}").replace('-', "−")
    }
}

// ----- tours, experiments ------------------------------------------------------------

/// A course for ε to take: the values it stops at, each with the seconds it stays, and
/// the seconds from one to the next (eased in and out).
#[derive(Clone, Copy, Debug)]
struct Tour {
    stops: &'static [(f64, f32)],
    leg: f32,
}

impl Tour {
    fn length(&self) -> f32 {
        self.stops.iter().map(|s| s.1).sum::<f32>() + self.stops.len().saturating_sub(1) as f32 * self.leg
    }

    /// Where ε is, `s` seconds in.
    fn at(&self, s: f32) -> f64 {
        let mut t = s.max(0.0);
        for (i, &(eps, stay)) in self.stops.iter().enumerate() {
            if t <= stay {
                return eps;
            }
            t -= stay;
            let Some(&(next, _)) = self.stops.get(i + 1) else { return eps };
            if t <= self.leg {
                let u = smoothstep(t / self.leg.max(1e-6)) as f64;
                return eps + (next - eps) * u;
            }
            t -= self.leg;
        }
        self.stops.last().map_or(0.0, |s| s.0)
    }
}

struct Trial {
    label: &'static str,
    look_for: &'static str,
    lambda: f64,
    /// ε as set (where a tour starts).
    eps: f64,
    v: [f64; 2],
    w: [f64; 2],
    x: [f64; 2],
    /// The step shown, and where playing on from there stops (if it plays).
    k: usize,
    play_to: Option<usize>,
    tour: Option<Tour>,
}

const SCISSORS: Tour = Tour { stops: &[(0.16, 0.8), (0.0, 0.0)], leg: 5.0 };
const TINY: Tour = Tour { stops: &[(0.0, 0.6), (0.0001, 0.0)], leg: 2.5 };
const KINDS: Tour = Tour { stops: &[(0.04, 1.6), (0.0, 2.0), (-0.04, 0.0)], leg: 2.6 };

const TRIALS: [Trial; 5] = [
    Trial {
        label: "two eigenlines closing",
        look_for: r"$\lambda = 0.5$, and $\varepsilon$ plays from $0.16$ down to $0$. The eigenvalues $0.5 \pm \sqrt{\varepsilon}$ start at $0.9$ and $0.1$ and meet at $0.5$, and the orange and purple eigenlines close like scissors onto the line of $v$. Until they meet, $A$ is diagonalisable, $A = VDV^{-1}$, but its two eigenvectors grow nearly parallel: the yellow area $|\det V|$ falls from $0.69$ to $0.20$ at $\varepsilon = 0.01$, and to $0.02$ at $\varepsilon = 0.0001$, where $V^{-1}$ has entries as large as 40. At $\varepsilon = 0$ one eigenline is left: $A$ is defective.",
        lambda: 0.5,
        eps: 0.16,
        v: V0,
        w: W0,
        x: X0,
        k: 3,
        play_to: None,
        tour: Some(SCISSORS),
    },
    Trial {
        label: r"a shear: $\lambda = 1$",
        look_for: r"Here $P = I$, so $A = J$ with $\lambda = 1$: a shear, the simplest Jordan block. $J^k$ has ones on its diagonal and $k$ above it, so $A^k$ slides every point along the $x$ axis by $k$ times its height. The dots from $x = (0, 0.4)$ step $0.4$ to the right each time, evenly spaced, and the $x$ axis, the one eigenline, stays put. $|A^k x| = 0.4\sqrt{k^2 + 1}$ grows like $k$: by adding, not by multiplying.",
        lambda: 1.0,
        eps: 0.0,
        v: [1.0, 0.0],
        w: [0.0, 1.0],
        x: [0.0, 0.4],
        k: 0,
        play_to: Some(7),
        tour: None,
    },
    Trial {
        label: "it grows before it shrinks",
        look_for: r"$\lambda = 0.9$, so $\lambda^k$ alone shrinks at every step (the orange line on the chart). But $x = 1.25w$, so $A^k x = 1.25(k\lambda^{k-1} v + \lambda^k w)$, and the part along $v$ grows first. $|A^k x|$ climbs to $3.89$ times $|x|$ at $k = 9$, and is back below $|x|$ only at $k = 35$. It does go to $0$ in the end, as $|\lambda| < 1$. The height of the hump depends on $v$ and $w$: halve $w$ (dragging its tip, $x$ left where it is) and $|A^k x|$ climbs to $7.76$ times $|x|$.",
        lambda: 0.9,
        eps: 0.0,
        v: V0,
        w: W0,
        x: X0,
        k: 0,
        play_to: Some(K),
        tour: None,
    },
    Trial {
        label: "a tiny change, a large effect",
        look_for: r"$\varepsilon$ goes from $0$ to $0.0001$ below the diagonal of $J$, with $\lambda = 0.9$. $A$'s entries change by less than $0.0001$, too little to show in its numbers, but its eigenvalues split to $0.91$ and $0.89$: each moves by $\sqrt{\varepsilon} = 0.01$, a hundred times the change. A symmetric matrix changed by a symmetric one of that size (the square root of the sum of its squared entries) has its eigenvalues moved by $0.0001$ at most. On the chart of how far they move, both scales logarithmic, $\sqrt{|\varepsilon|}$ is a line of slope $1/2$: the smaller the change, the more it is magnified.",
        lambda: 0.9,
        eps: 0.0,
        v: V0,
        w: W0,
        x: X0,
        k: 9,
        play_to: None,
        tour: Some(TINY),
    },
    Trial {
        label: r"the three kinds of $2 \times 2$ matrix",
        look_for: r"$\lambda = 0.75$, and $\varepsilon$ goes from $0.04$ through $0$ to $-0.04$. Up to a change of basis, a real $2 \times 2$ matrix is one of three kinds. Two eigenlines (here $0.95$ and $0.55$): diagonal in the basis of its eigenvectors. One ($0.75$ twice): a Jordan block. None ($0.75 \pm 0.2i$): a turn with a stretch, and the steps spiral in. ($\lambda I$, which keeps every line, is diagonal already.) $\tr A = 1.5$ throughout, while $\det A$ rises through $(\tr A)^2/4$: below the parabola of the map in the matrix exponential's scene, on it, above it.",
        lambda: 0.75,
        eps: 0.04,
        v: V0,
        w: W0,
        x: X0,
        k: K,
        play_to: None,
        tour: Some(KINDS),
    },
];

/// The experiment the scene opens at (the hump), and the step it shows: the highest.
const OPENING: usize = 2;
const OPENING_K: usize = 9;

// ----- gliding from one setting to the next ------------------------------------------

/// How fast a change of setting is glided to: each of the two eases one after the
/// other covers two thirds of what is left in this many seconds.
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

    /// Move the drawn setting `dt` seconds nearer to the one asked for.
    fn step(&mut self, dt: f64) {
        if self.arrived() || dt <= 0.0 {
            return;
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        let (to, mut between, mut shown) = (self.to.numbers(), self.between.numbers(), self.shown.numbers());
        let mut apart = 0.0f64;
        for i in 0..8 {
            between[i] += (to[i] - between[i]) * k;
            shown[i] += (between[i] - shown[i]) * k;
            // (ε is followed to a small share of itself: an ε of 0.0001 is what matters)
            let scale = if i == 1 { (to[i].abs().max(1e-6) * 1e2).min(1.0) } else { 1.0 };
            apart = apart.max((to[i] - between[i]).abs() / scale).max((to[i] - shown[i]).abs() / scale);
        }
        // (exactly there in the end: the picture then stops changing)
        if apart < 2e-4 || !apart.is_finite() {
            *self = Glide::at(self.to);
        } else {
            (self.between, self.shown) = (Setting::from_numbers(between), Setting::from_numbers(shown));
        }
    }
}

// ----- where things go ---------------------------------------------------------------

/// The part of the plane the view is fitted to (its centre, and half its size): the
/// steps of every experiment lie in it, as far as they are worth seeing.
const FRAME_CENTRE: Vec2 = Vec2::new(0.75, 2.0);
const FRAME_HALF: Vec2 = Vec2::new(2.25, 2.5);

/// The screen panel's parts, in logical pixels.
#[derive(Clone, Debug)]
struct Panel {
    min: Vec2,
    max: Vec2,
    /// The size of the numbers in the matrices.
    text: f32,
    /// Top-left corners of the row "A = [ ]  Jᵏ = [ ]", of the line "A = P J P⁻¹", and of
    /// the row of P, J and P⁻¹.
    row1: Vec2,
    title: Vec2,
    row2: Vec2,
    /// The charts: where their data is drawn (their titles go above, their numbers
    /// below and to the left).
    hump: (Vec2, Vec2),
    play: (Vec2, Vec2),
    eig: (Vec2, Vec2),
    moved: (Vec2, Vec2),
}

#[derive(Clone, Debug)]
struct Layout {
    /// A phone held upright: the panel goes below the picture.
    narrow: bool,
    /// The part of the canvas the picture has to itself.
    picture: (Vec2, Vec2),
    /// None on a canvas too small to spare the room (the numbers are in the side panel).
    panel: Option<Panel>,
}

/// Rows of the panel, in logical pixels: a line of words, a chart's title, the room
/// under a chart for its numbers, and the room to its left for them.
const LINE_H: f32 = 17.0;
const CHART_TITLE: f32 = 20.0;
const CHART_BELOW: f32 = 20.0;
const CHART_LEFT: f32 = 30.0;

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let narrow = view.x < 620.0 && view.y >= 560.0;
        let pad = if narrow { 10.0 } else { 12.0 };
        let width = if narrow { view.x } else { 372.0 };
        let inner = width - 2.0 * pad;
        // the numbers' size: as large as lets P, J and P⁻¹ fit across, up to 13
        let three = |s: f32| 3.0 * matrix_size(2, 2, CELL, s).x + 2.0 * 10.0;
        let mut text = 13.0f32;
        while three(text) > inner && text > 9.0 {
            text -= 0.5;
        }
        let mat_h = matrix_size(2, 2, CELL, text).y;
        let names_h = 16.0;
        let rows_h = mat_h + 8.0 + LINE_H + mat_h + names_h;
        // everything but the charts' own heights
        let fixed = pad + rows_h + 10.0 + CHART_TITLE + CHART_BELOW + 6.0 + CHART_TITLE + CHART_BELOW + pad;
        // the charts as tall as the canvas lets them be on a desktop, between these
        let (hump_h, small_h) = if narrow {
            (58.0, 96.0)
        } else {
            let spare = (view.y - inset - 20.0 - fixed - 190.0).max(0.0).min(110.0);
            (80.0 + spare * 50.0 / 110.0, 110.0 + spare * 60.0 / 110.0)
        };
        let total = fixed + hump_h + small_h;
        let whole = Layout { narrow, picture: (Vec2::new(0.0, inset), view), panel: None };
        let place = |min: Vec2| {
            let x = min.x + pad;
            let mut y = min.y + pad;
            let row1 = Vec2::new(x, y);
            y += mat_h + 8.0;
            let title = Vec2::new(x, y);
            y += LINE_H;
            let row2 = Vec2::new(x, y);
            y += mat_h + names_h + 10.0 + CHART_TITLE;
            let right = min.x + width - pad;
            let hump = (Vec2::new(x + CHART_LEFT, y), Vec2::new(right - 4.0, y + hump_h));
            let play = (Vec2::new(right - 30.0, y - CHART_TITLE - 2.0), Vec2::new(right, y - 2.0));
            y += hump_h + CHART_BELOW + 6.0 + CHART_TITLE;
            let half = (right - x - 14.0) * 0.5;
            let eig = (Vec2::new(x, y), Vec2::new(x + half - CHART_LEFT * 0.3, y + small_h));
            let moved = (Vec2::new(x + half + 14.0 + CHART_LEFT, y), Vec2::new(right - 4.0, y + small_h));
            y += small_h + CHART_BELOW + pad;
            Panel { min, max: Vec2::new(min.x + width, y), text, row1, title, row2, hump, play, eig, moved }
        };
        if narrow {
            // the panel below the picture; the picture keeps at least the top 42 %
            let below = view.y - total;
            if below < view.y * 0.42 {
                return whole;
            }
            let mut panel = place(Vec2::new(0.0, below));
            panel.max = view;
            return Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, below)), panel: Some(panel) };
        }
        let top = inset + 12.0;
        if view.x < 760.0 || view.y - top - 8.0 < total {
            return whole;
        }
        let panel = place(Vec2::new(view.x - width - 12.0, top));
        Layout { narrow, picture: (Vec2::new(0.0, inset), Vec2::new(panel.min.x - 8.0, view.y)), panel: Some(panel) }
    }

    /// The part of the canvas the scene's frame is fitted into: the picture's, less the
    /// heading at its top and the caption at its foot.
    fn frame_room(&self) -> (Vec2, Vec2) {
        let (min, max) = self.picture;
        let foot = if self.narrow { 8.0 } else { 40.0 };
        (min + Vec2::new(14.0, 58.0), max - Vec2::new(14.0, foot))
    }
}

fn inside_rect(p: Vec2, (min, max): (Vec2, Vec2)) -> bool {
    p.cmpge(min).all() && p.cmple(max).all()
}

/// A matrix of numbers on the panel, as `hud_matrix` draws it, with a number too wide
/// for its cell (a power of ten) drawn smaller, to fit: so that the brackets stay where
/// they are whatever the numbers.
fn matrix(f: &mut Frame, top_left: Vec2, rows: &[Vec<(String, Color)>], size: f32) -> Vec2 {
    let room = matrix_cell(CELL, size).x - 4.0;
    let mut fitted = rows.to_vec();
    let mut smaller = Vec::new();
    for (i, row) in fitted.iter_mut().enumerate() {
        for (j, (text, color)) in row.iter_mut().enumerate() {
            // (only a power of ten can be too wide)
            if !text.contains('e') {
                continue;
            }
            let wide = f.text_size(text, size).x;
            if wide > room {
                smaller.push((i, j, std::mem::take(text), *color, size * room / wide));
            }
        }
    }
    let total = hud_matrix(f, top_left, &fitted, CELL, size, pal::DIM);
    for (i, j, text, color, s) in smaller {
        let (lo, hi) = matrix_entry(top_left, CELL, size, i, j);
        f.text_px((lo + hi) * 0.5, text, s, color, Align2::CENTER_CENTER);
    }
    total
}

/// A matrix's entries as `matrix` takes them, each column in its own colour.
fn entries(m: DMat2, cols: [Color; 2]) -> Vec<Vec<(String, Color)>> {
    (0..2).map(|i| (0..2).map(|j| (cell(m.col(j)[i]), cols[j])).collect()).collect()
}

/// The parts of the picture drawn as lines whose number of points changes as the
/// picture moves: each in a batch of its own (different from the usual one by a hair of
/// its width, and nothing else), so that from one frame to the next the parts that stay
/// the same can still be compared point for point (`eidos motion`).
#[derive(Clone, Copy, PartialEq)]
enum Part {
    /// The parts of xₖ along v and w, which vanish when xₖ has no part along one.
    Parts = 1,
    /// The yellow parallelogram's outline, there only when there are two eigenlines.
    Sliver,
    /// The line joining the steps up to the one shown, and on from it: they gain and
    /// lose points as the step moves on.
    Before,
    After,
}

fn lines_for<'a>(f: &'a mut Frame, part: Part) -> &'a mut crate::gfx::draw::LineBatch {
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 1e-4 * part as u8 as f32, ..Default::default() })
}

/// An arrow from `a` to `b` (as `Frame::arrow` draws one), its shaft in the batch of `part`.
fn arrow_in(f: &mut Frame, part: Part, a: Vec2, b: Vec2, width: f32, c: Color) {
    let (a3, b3) = (a.extend(0.0), b.extend(0.0));
    let d = b3 - a3;
    let len = d.length();
    if !(len > 1e-6 && len.is_finite()) {
        return;
    }
    let dir = d / len;
    let wpp = f.cam.world_per_px(b3) * f.px;
    let head = ((width * 3.2 + 5.0) * wpp).min(len * 0.6);
    let side = dir.cross(Vec3::Z).normalize_or_zero() * (head * 0.42);
    let base = b3 - dir * head;
    f.dl.tris_alpha.tri(b3, base + side, base - side, c);
    lines_for(f, part).segment(a3, b3 - dir * (head * 0.8), width, c);
}

/// The part of the line through 0 along `d` inside the box `lo..hi` (if it crosses it).
fn line_in_box(d: Vec2, lo: Vec2, hi: Vec2) -> Option<(Vec2, Vec2)> {
    let d = d.normalize_or_zero();
    if d == Vec2::ZERO {
        return None;
    }
    let far = 4.0 * (hi - lo).length() + lo.length() + hi.length();
    let (a, b) = (-d * far, d * far);
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    let e = b - a;
    for (p, q) in [(-e.x, a.x - lo.x), (e.x, hi.x - a.x), (-e.y, a.y - lo.y), (e.y, hi.y - a.y)] {
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
    (t0 < t1).then(|| (a + e * t0, a + e * t1))
}

// ----- the scene ---------------------------------------------------------------------

pub struct Jordan {
    glide: Glide,
    /// The step asked for, and the step drawn, which follows it (and is it, while playing).
    k: f64,
    k_shown: f64,
    /// Where the drawn step is on its way, between the two eases it follows k through.
    k_between: f64,
    /// The mark on the chart is being dragged: k follows it between whole steps, and
    /// goes to the nearest one when it is let go.
    scrubbing: bool,
    playing: bool,
    /// Play once the glide to a new setting has arrived; and where playing stops.
    play_when_there: bool,
    play_to: f64,
    /// The course ε is on, and the seconds into it (it starts once the glide has arrived).
    tour: Option<(Tour, f32)>,
    grid: bool,
    parts: bool,
    /// The canvas last drawn on and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and so where the picture is.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
}

impl Jordan {
    fn new() -> Self {
        let mut scene = Jordan {
            glide: Glide::at(Setting { lambda: 0.0, eps: 0.0, v: DVec2::X, w: DVec2::Y, x: DVec2::ONE }),
            k: 0.0,
            k_shown: 0.0,
            k_between: 0.0,
            scrubbing: false,
            playing: false,
            play_when_there: false,
            play_to: 0.0,
            tour: None,
            grid: true,
            parts: true,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
        };
        // (opening, there is nothing to glide from: the hump, at its highest)
        scene.set_up(OPENING);
        scene.glide = Glide::at(scene.glide.to);
        (scene.k, scene.k_shown, scene.k_between, scene.play_when_there) = (OPENING_K as f64, OPENING_K as f64, OPENING_K as f64, false);
        scene
    }

    /// Set up experiment `i` of `TRIALS`: glide to its setting, then play it.
    fn set_up(&mut self, i: usize) {
        let t = &TRIALS[i];
        self.glide.to = Setting { lambda: t.lambda, eps: t.eps, v: DVec2::from(t.v), w: DVec2::from(t.w), x: DVec2::from(t.x) };
        self.k = t.k as f64;
        self.playing = false;
        self.play_when_there = t.play_to.is_some();
        self.play_to = t.play_to.unwrap_or(t.k) as f64;
        self.tour = t.tour.map(|tour| (tour, 0.0));
        (self.grid, self.parts) = (true, true);
    }

    /// The setting as drawn.
    fn shown(&self) -> Setting {
        self.glide.shown
    }

    /// Play the steps from where k is (from 0, if it is at the end).
    fn play(&mut self) {
        if self.k >= K as f64 - 1e-9 {
            self.k = 0.0;
            self.play_when_there = true;
        } else {
            self.playing = true;
        }
        self.play_to = K as f64;
    }

    fn pause(&mut self) {
        (self.playing, self.play_when_there) = (false, false);
        self.k = self.k.round();
    }

    /// Choose the step k by hand.
    fn choose_step(&mut self, k: f64) {
        (self.playing, self.play_when_there) = (false, false);
        self.k = k.round().clamp(0.0, K as f64);
    }

    /// Move everything on: `dt` seconds of the scene's time (playing, the tour) and
    /// `real_dt` of the clock's (a change of setting glides by the clock, so that it is
    /// seen through even while the scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        self.glide.step(real_dt as f64);
        if self.glide.arrived() {
            if let Some((tour, s)) = self.tour {
                let s = s + dt.max(0.0);
                let eps = tour.at(s);
                self.glide.now(|g| g.eps = eps);
                self.tour = (s < tour.length()).then_some((tour, s));
            }
        }
        if self.play_when_there && self.glide.arrived() && self.k_shown == self.k {
            (self.play_when_there, self.playing) = (false, true);
        }
        if !self.scrubbing && !self.playing {
            self.k = self.k.round();
        }
        if self.playing || self.scrubbing {
            if self.playing {
                self.k = (self.k + RATE * dt.max(0.0) as f64).min(self.play_to);
                self.playing = self.k < self.play_to;
            }
            (self.k_shown, self.k_between) = (self.k, self.k);
        } else if self.k_shown != self.k {
            // a step chosen by hand is glided to, through two eases one after the other
            let ease = 1.0 - (-(real_dt.max(0.0) as f64) / 0.05).exp();
            self.k_between += (self.k - self.k_between) * ease;
            self.k_shown += (self.k_between - self.k_shown) * ease;
            if (self.k - self.k_shown).abs().max((self.k - self.k_between).abs()) < 2e-3 {
                (self.k_shown, self.k_between) = (self.k, self.k);
            }
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
}

/// A slider for ε that writes small values in full (0.0001, not 0.000), with a bar that
/// lets go onto 0 itself within a hair of it: the block with nothing changed. Returns
/// true when ε was changed.
fn eps_slider(ui: &mut egui::Ui, label: &str, v: &mut f64) -> bool {
    let (mut typed, mut bar) = (*v, *v);
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(ui::label_text(ui, label, 12.5, Some(pal::DIM), false));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let value = egui::DragValue::new(&mut typed).speed(0.0005).range(-EPS_MAX..=EPS_MAX).custom_formatter(|v, _| small(v).replace('−', "-"));
            changed |= ui.add(value).changed();
        });
    });
    ui.scope(|ui| {
        ui.spacing_mut().slider_width = ui.available_width();
        if ui.add(egui::Slider::new(&mut bar, -EPS_MAX..=EPS_MAX).show_value(false)).changed() {
            changed = true;
            if bar.abs() < EPS_SNAP {
                bar = 0.0;
            }
        }
    });
    if typed != *v {
        *v = typed;
    } else if bar != *v {
        *v = bar;
    }
    changed
}

/// The heading's words for a kind: what it is, and what that means (a shorter version
/// for a narrow picture).
fn heading_text(s: &Setting) -> (String, [&'static str; 2]) {
    let [(r1, i1), (r2, _)] = eigenvalues(s.lambda, s.eps);
    match kind(s.eps) {
        Kind::One => (
            format!(r"one eigenline: $\lambda = {}$ twice", num(s.lambda, 2)),
            [r"defective: no change of basis makes $A$ diagonal; the best one gives $A = PJP^{-1}$", r"defective: the best is $A = PJP^{-1}$"],
        ),
        Kind::Two => (
            format!(r"two eigenlines: $\lambda = {}$ and ${}$", num(r1, 2), num(r2, 2)),
            [r"diagonalisable: $A = VDV^{-1}$, with the unit eigenvectors as $V$'s columns", r"diagonalisable: $A = VDV^{-1}$"],
        ),
        Kind::Complex => (
            format!(r"no eigenline: $\lambda = {} \pm {}i$", num(r1, 2), num(i1, 2)),
            [r"complex eigenvalues: in a suitable basis $A$ is a turn with a stretch", "a turn with a stretch, in a suitable basis"],
        ),
    }
}

impl Scene for Jordan {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.05;
        s.axis_alpha = 0.3;
    }

    fn animated(&self) -> bool {
        self.playing || self.play_when_there || !self.glide.arrived() || self.tour.is_some() || self.k_shown != self.k
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A matrix can be made diagonal by a change of basis only if it has enough independent eigenvectors: two, for a $2 \times 2$ matrix. With an eigenvalue $\lambda$ repeated it may have only one eigenline. It is then called defective, and the best a change of basis $P$ can do is the Jordan block $J$:");
        ui::equation(ui, r"A = PJP^{-1}, \quad J = \begin{bmatrix} \lambda & 1 \\ 0 & \lambda \end{bmatrix}");
        ui::note(ui, r"$P$'s first column is the eigenvector $v$; its second is a vector $w$ that $A$ sends to $\lambda w + v$. So $A$ multiplies the coordinates of $x$ along $v$ and $w$ by $\lambda$ and, on top of that, adds the one along $w$ to the one along $v$: the powers of $J$ have a second term besides $\lambda^k$, a shear that keeps growing:");
        ui::equation(ui, r"J^k = \begin{bmatrix} \lambda^k & k\lambda^{k-1} \\ 0 & \lambda^k \end{bmatrix}");

        ui::section(ui, "Things to try");
        for (i, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(i);
            }
        }

        ui::section(ui, "Steps");
        ui.horizontal(|ui| {
            let busy = self.playing || self.play_when_there;
            if ui.button(if busy { "❚❚ pause" } else { "▶ play" }).clicked() {
                if busy {
                    self.pause();
                } else {
                    self.play();
                }
            }
            if ui.button(ui::label_text(ui, "back to $k = 0$", 12.5, None, false)).clicked() {
                self.choose_step(0.0);
            }
        });
        let mut k = self.k.round() as usize;
        if ui::slider_int(ui, "step $k$", &mut k, 0..=K) {
            self.choose_step(k as f64);
        }
        ui::note(ui, r"The step shown: $x_k = A^k x$, with its parts along $v$ and $w$. Or drag the white mark along the chart of $|A^k x|$.");

        ui::section(ui, "The matrix");
        let mut lambda = self.glide.to.lambda as f32;
        if ui::slider(ui, r"eigenvalue $\lambda$", &mut lambda, -LAMBDA_MAX as f32..=LAMBDA_MAX as f32) {
            self.glide.now(|s| s.lambda = lambda as f64);
        }
        let mut eps = self.glide.to.eps;
        if eps_slider(ui, r"$\varepsilon$, below $J$'s diagonal", &mut eps) {
            self.glide.now(|s| s.eps = eps);
            self.tour = None;
        }
        ui::note(ui, r"$\varepsilon = 0$ is the Jordan block itself. Above 0 the eigenvalues are $\lambda \pm \sqrt{\varepsilon}$, two of them; below, $\lambda \pm i\sqrt{-\varepsilon}$. Type a value as small as you like.");
        ui::note(ui, r"Drag the tips of $v$ and $w$ to choose $P$: $A = PJP^{-1}$ is built from them.");
        if let Some(i) = ui::buttons(ui, &["the first $v$ and $w$", "$P = I$"]) {
            let (v, w) = if i == 0 { (V0, W0) } else { ([1.0, 0.0], [0.0, 1.0]) };
            self.glide.to.v = DVec2::from(v);
            self.glide.to.w = DVec2::from(w);
        }

        ui::section(ui, "Show");
        ui::checkbox(ui, &mut self.grid, "the grid of $v$ and $w$");
        ui::checkbox(ui, &mut self.parts, "the parts of $x_k$ along $v$ and $w$");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::ORANGE, r"orange, dashed: the eigenline (with $\varepsilon > 0$, the larger eigenvalue's); purple: the second one, when there are two"),
                (pal::PINK, r"pink arrow $v$: $P$'s first column, the eigenvector when $\varepsilon = 0$ (drag its tip)"),
                (pal::GREEN, r"green arrow $w$: $P$'s second column, which $A$ sends to $\lambda w + v$ (drag its tip)"),
                (pal::WHITE, r"white: the start $x$ (drag it) and its steps $x_k = A^k x$, joined in order, bright up to the step shown"),
                (pal::DIM, r"faint grid: whole steps along $v$ and $w$; the pink and green arrows to $x_k$ are its parts along them"),
                (pal::YELLOW, r"yellow: the parallelogram of the two unit eigenvectors, when there are two; its area is $|\det V|$"),
                (pal::TEXT, r"the bar chart: $|A^k x|$ for each $k$ (drag the white mark to choose $k$), and in orange $|\lambda|^k\,|x|$, what the growth $\lambda^k$ alone would give"),
                (pal::TEXT, r"the complex plane: the eigenvalues as orange and purple dots, $\lambda$ as a ring, and the circle $|\lambda| = 1$"),
                (pal::RED, r"red: the change $\varepsilon$, and how far it moves the eigenvalues"),
                (pal::BLUE, r"blue, on the log chart: the most the same change can move a symmetric matrix's eigenvalues"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Defective:", r"having fewer independent eigenvectors than the matrix has columns. A defective $2 \times 2$ matrix has one eigenvalue twice and a single eigenline. No change of basis makes it diagonal.");
        ui::term(ui, "Jordan block:", r"$J$ above: $\lambda$ twice on the diagonal and a 1 just above it. Every defective $2 \times 2$ matrix is $PJP^{-1}$ for some $P$.");
        ui::term(ui, "Chain:", r"$v$ and $w$ with $(A - \lambda I)v = 0$ and $(A - \lambda I)w = v$; that is, $Av = \lambda v$ and $Aw = \lambda w + v$, which is $AP = PJ$ column by column. In a larger matrix a chain can be longer ($A - \lambda I$ sending $w_2$ to $w_1$ and $w_1$ to $v$), and its block then has $\lambda$ all along its diagonal and a 1 in every place just above it.");
        ui::term(ui, "The flow:", r"$e^{Jt} = e^{\lambda t}(I + tN)$, where $N = J - \lambda I$ has the single 1 and $N^2 = 0$: the same shear, growing like $t$, on top of $e^{\lambda t}$.");
        ui::term(ui, "Diagonalisable:", r"$A = VDV^{-1}$ with $D$ diagonal: $V$'s columns are eigenvectors, and they must be independent for $V^{-1}$ to exist.");
        ui::term(ui, r"$J_\varepsilon$:", r"$J$ with $\varepsilon$ put in place of the 0 below its diagonal. Its eigenvalues are $\lambda \pm \sqrt{\varepsilon}$ (for $\varepsilon < 0$, $\lambda \pm i\sqrt{-\varepsilon}$), and the picture shows $A = PJ_\varepsilon P^{-1}$.");
        ui::term(ui, "Complex eigenvalues:", r"$a \pm bi$: no line is kept. In a suitable basis $A$ is the matrix with rows $(a, -b)$ and $(b, a)$: a turn combined with a stretch.");

        ui::section(ui, "Numbers");
        let s = self.shown();
        let shape = Shape::of(&s);
        let [(r1, i1), (r2, _)] = eigenvalues(s.lambda, s.eps);
        let k = self.k_shown.round() as usize;
        ui::readout(
            ui,
            "kind",
            match kind(s.eps) {
                Kind::One => "one eigenline: defective",
                Kind::Two => "two eigenlines",
                Kind::Complex => "complex eigenvalues",
            },
        );
        ui::readout(
            ui,
            "eigenvalues",
            match kind(s.eps) {
                Kind::One => format!("${}$ twice", num(r1, 4)),
                Kind::Two => format!("${}, {}$", num(r1, 4), num(r2, 4)),
                Kind::Complex => format!(r"${} \pm {}i$", num(r1, 4), num(i1, 4)),
            },
        );
        let tr = shape.a.x_axis.x + shape.a.y_axis.y;
        ui::readout(ui, r"$\tr A$, $\det A$", format!("${}, {}$", num(tr, 4), num(shape.a.determinant(), 4)));
        ui::readout(ui, r"$(\tr A)^2/4$", format!("${}$", num(tr * tr / 4.0, 4)));
        ui::readout(ui, r"$x$ in the basis $v, w$", format!("${}$", tuple(&[shape.c.x, shape.c.y], 2)));
        let sizes = shape.sizes();
        ui::readout(ui, &format!(r"$|A^k x| / |x|$ at $k = {k}$"), format!("${}$", num(sizes[k.min(K)] / sizes[0].max(1e-300), 2)));
        let (top_k, top) = largest(&sizes);
        ui::readout(ui, r"largest $|A^k x| / |x|$", format!("${}$ at $k = {top_k}$", num(top / sizes[0].max(1e-300), 2)));
        if let Some((angle, det, inv)) = scissors(&s) {
            ui::readout_colored(ui, "angle between the eigenlines", format!("{}°", num(angle, 2)), pal::ORANGE);
            ui::readout_colored(ui, r"$|\det V|$", format!("${}$", small_or(det)), pal::YELLOW);
            let inv = if inv < 9999.5 { num(inv, 1) } else { format!("{inv:.1e}") };
            ui::readout_colored(ui, r"largest entry of $V^{-1}$", format!("${inv}$"), pal::YELLOW);
        }
        if s.eps != 0.0 {
            ui::readout_colored(ui, r"eigenvalues moved, $\sqrt{|\varepsilon|}$", format!("${}$", small(s.eps.abs().sqrt())), pal::RED);
            ui::readout_colored(ui, r"symmetric: at most $|\varepsilon|$", format!("${}$", small(s.eps.abs())), pal::BLUE);
        }
        ui::see_also(
            ui,
            &[
                ("eigen", "finding eigenvalues, and a shear's single eigenline"),
                ("powers", "powers of a matrix with two eigenlines"),
                ("expm", r"the flow $e^{At}$, and the map of every $2 \times 2$ matrix"),
                ("basis", "a change of basis, and the eigenvector basis"),
                ("schur", "the triangular form every matrix has"),
            ],
        );
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
        }
        self.advance(f.dt, f.real_dt);

        let labels_before = f.dl.labels.len();
        let plates = self.picture(f, &lay);
        let numbers_from = f.dl.labels.len();
        f.axes2();
        // no number of the axes is written over the heading or the caption
        let mut i = 0;
        f.dl.labels.retain(|l| {
            i += 1;
            i <= numbers_from || !plates.iter().flatten().any(|plate| inside_rect(l.pos, (plate.0 - Vec2::splat(10.0), plate.1 + Vec2::splat(10.0))))
        });
        if covered {
            f.dl.labels.truncate(labels_before);
            return;
        }
        // nothing written into the picture may show through the panel
        if let Some(p) = &lay.panel {
            let (lo, hi) = (p.min - Vec2::splat(2.0), p.max + Vec2::splat(2.0));
            let mut i = 0;
            f.dl.labels.retain(|l| {
                i += 1;
                i <= labels_before || l.backed || !inside_rect(l.pos, (lo, hi))
            });
            self.panel(f, p);
        }
    }
}

/// |det V| for the screen: two decimals, or two figures in a power of ten below that.
fn small_or(v: f64) -> String {
    if v >= 0.005 || v == 0.0 {
        num(v, 2)
    } else {
        format!("{v:.1e}").replace('-', "−")
    }
}

impl Jordan {
    /// Everything drawn in the plane, and the names written beside it. Returns the boxes
    /// of the heading and of the caption, which are written over it.
    fn picture(&mut self, f: &mut Frame, lay: &Layout) -> [Option<(Vec2, Vec2)>; 2] {
        let (lo, hi) = f.cam.flat_bounds();
        let under_panel = |f: &Frame, p: DVec2| lay.panel.as_ref().is_some_and(|panel| on_screen(f, p.as_vec2().extend(0.0)).is_some_and(|q| inside_rect(q, (panel.min, panel.max))));

        // the handles: x, and the tips of v and w (not where they lie under the panel)
        let s = self.shown();
        let dragging = f.handles.dragging();
        let mut x = s.x.as_vec2();
        if (!under_panel(f, s.x) || dragging) && f.drag2(1, &mut x, pal::WHITE) {
            let x = x.as_dvec2().clamp(DVec2::splat(-X_MAX), DVec2::splat(X_MAX));
            self.glide.now(|g| g.x = x);
        }
        let mut v = s.v.as_vec2();
        if (!under_panel(f, s.v) || dragging) && f.drag2(2, &mut v, pal::PINK) {
            let v = v.as_dvec2();
            if chain_ok(v, s.w) {
                self.glide.now(|g| g.v = v);
            }
        }
        let mut w = s.w.as_vec2();
        if (!under_panel(f, s.w) || dragging) && f.drag2(3, &mut w, pal::GREEN) {
            let w = w.as_dvec2();
            if chain_ok(s.v, w) {
                self.glide.now(|g| g.w = w);
            }
        }

        let s = self.shown();
        let shape = Shape::of(&s);
        let (vf, wf) = (s.v.as_vec2(), s.w.as_vec2());
        // the batches whose number of points may change, made in the same order every frame
        for part in [Part::Parts, Part::Sliver, Part::Before, Part::After] {
            lines_for(f, part);
        }

        let mut room = Room::new(f);
        if let Some(p) = &lay.panel {
            room.wall(p.min - Vec2::splat(4.0), p.max + Vec2::splat(4.0));
        }
        let heading = self.heading(f, lay, &s);
        room.wall(heading.0, heading.1);
        let mut plates = [Some(heading), None];
        if !lay.narrow && f.view_size().y >= 300.0 {
            let parts = [r"white dots: $x_k = A^k x$, $k = 0, 1, \ldots, 40$", "faint grid: whole steps along $v$ and $w$", "dashed: the eigenlines"];
            let (left, right, bottom) = (lay.picture.0.x, lay.picture.1.x, lay.picture.1.y - 14.0);
            let text = parts.join("      ");
            let size = 12.0;
            let wide = f.text_size(&text, size).x;
            if wide + 36.0 <= right - left {
                f.caption_px(Vec2::new(left + 18.0, bottom), text, size, pal::DIM, Align2::LEFT_BOTTOM);
                let plate = (Vec2::new(left + 10.0, bottom - size - 8.0), Vec2::new(left + 26.0 + wide, bottom + 6.0));
                room.wall(plate.0, plate.1);
                plates[1] = Some(plate);
            }
        }

        // the grid of v and w: whole steps along each
        if self.grid {
            let n = 8;
            for i in -n..=n {
                let t = i as f32;
                if i == 0 {
                    continue;
                }
                f.line(vf * t - wf * n as f32, vf * t + wf * n as f32, 1.0, pal::DIM.a(0.16));
                f.line(wf * t - vf * n as f32, wf * t + vf * n as f32, 1.0, pal::DIM.a(0.16));
            }
            // (the lines through 0 along v and w a little stronger)
            f.line(-wf * n as f32, wf * n as f32, 1.2, pal::GREEN.a(0.22));
            if kind(s.eps) != Kind::One {
                f.line(-vf * n as f32, vf * n as f32, 1.2, pal::PINK.a(0.22));
            }
        }

        // the eigenlines, dashed (the dashed batch holds them alone, so their coming and
        // going with ε disturbs nothing else)
        let lines: Vec<(DVec2, f64, Color)> = match kind(s.eps) {
            Kind::One => vec![(s.v, s.lambda, pal::ORANGE)],
            Kind::Two => {
                let [(l1, _), (l2, _)] = eigenvalues(s.lambda, s.eps);
                let g = s.eps.sqrt();
                vec![(s.v + s.w * g, l1, pal::ORANGE), (s.v - s.w * g, l2, pal::PURPLE)]
            }
            Kind::Complex => Vec::new(),
        };
        for (d, _, c) in &lines {
            if let Some((a, b)) = line_in_box(d.as_vec2(), lo, hi) {
                f.dashed(a, b, 1.7, c.a(0.9));
                room.line(f, a.extend(0.0), b.extend(0.0));
            }
        }

        // the parallelogram of the unit eigenvectors: flat when the lines meet
        let sliver = unit_eigenvectors(&s).map(|(u1, u2)| (u1.as_vec2(), u2.as_vec2()));
        if let Some((u1, u2)) = sliver {
            let corners = [Vec2::ZERO, u1, u1 + u2, u2].map(|p| p.extend(0.0));
            f.dl.tris_alpha.quad(corners[0], corners[1], corners[2], corners[3], pal::YELLOW.a(0.22));
            lines_for(f, Part::Sliver).polyline([corners[0], corners[1], corners[2], corners[3], corners[0]], 1.4, pal::YELLOW.a(0.85));
            room.shape(f, &corners);
            for k in 0..4 {
                room.line(f, corners[k], corners[(k + 1) % 4]);
            }
        }

        // the steps: dots joined in order, bright up to the step shown
        let k_shown = self.k_shown;
        let pts: Vec<Vec3> = shape.steps.iter().map(|p| p.as_vec2().extend(0.0)).collect();
        // (kept to a box well beyond the view: a far step is not drawn as a far point)
        let span = (hi - lo).length();
        // (max then min rather than clamp: bounds worked out from the view are never trusted to be in order)
        let keep = |p: Vec3| p.truncate().max(lo - Vec2::splat(span)).min(hi + Vec2::splat(span)).extend(0.0);
        let reached = (k_shown.floor() as usize).min(K);
        let (now, now_coords) = shape.step_at(k_shown);
        let nowf = now.as_vec2();
        // the joining line, bright up to the step shown and faint on from it: the two
        // meet where the step shown is drawn
        let mut before: Vec<Vec3> = pts[..=reached].iter().map(|p| keep(*p)).collect();
        before.push(keep(nowf.extend(0.0)));
        let mut after = vec![keep(nowf.extend(0.0))];
        after.extend(pts[(reached + 1).min(K)..].iter().map(|p| keep(*p)));
        if reached >= K {
            after.push(after[0]);
        }
        lines_for(f, Part::Before).polyline(before, 1.6, pal::WHITE.a(0.55));
        lines_for(f, Part::After).polyline(after, 1.0, pal::WHITE.a(0.2));
        for (i, p) in pts.iter().enumerate() {
            let on = i as f64 <= k_shown + 1e-9;
            f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 1.25 + 1e-4, ..Default::default() }).push(keep(*p), if i == 0 { 7.0 } else { 6.0 }, pal::WHITE.a(if on { 0.85 } else { 0.25 }));
        }
        for pair in pts.windows(2) {
            room.line(f, keep(pair[0]), keep(pair[1]));
        }

        // the parts of the step shown along v and w: from 0 along w, then along v
        let along_w = wf * now_coords.y as f32;
        if self.parts {
            arrow_in(f, Part::Parts, Vec2::ZERO, along_w, 2.0, pal::GREEN.a(0.8));
            arrow_in(f, Part::Parts, along_w, nowf, 2.0, pal::PINK.a(0.8));
            room.line(f, Vec3::ZERO, along_w.extend(0.0));
            room.line(f, along_w.extend(0.0), nowf.extend(0.0));
        }
        f.dot(nowf, 12.0, pal::WHITE);

        // the chain
        f.arrow(Vec2::ZERO, vf, 2.8, pal::PINK);
        f.arrow(Vec2::ZERO, wf, 2.8, pal::GREEN);
        room.line(f, Vec3::ZERO, vf.extend(0.0));
        room.line(f, Vec3::ZERO, wf.extend(0.0));

        // the names, once everything they must keep clear of is drawn; the most needed first
        let away = |f: &Frame, p: Vec2| on_screen(f, p.extend(0.0)).zip(on_screen(f, Vec3::ZERO)).map_or(Vec2::NEG_Y, |(a, b)| a - b);
        let seen = |f: &Frame, p: Vec2| p.is_finite() && on_screen(f, p.extend(0.0)).is_some_and(|q| inside_rect(q, (Vec2::ZERO, f.view_size())));
        let long_px = |f: &Frame, a: Vec2, b: Vec2| on_screen(f, a.extend(0.0)).zip(on_screen(f, b.extend(0.0))).map_or(0.0, |(p, q)| p.distance(q));
        if seen(f, vf) {
            room.label(f, "v", vf.extend(0.0), "$v$", 16.0, pal::PINK, away(f, vf), 9.0, false);
        }
        if seen(f, wf) {
            room.label(f, "w", wf.extend(0.0), "$w$", 16.0, pal::GREEN, away(f, wf), 9.0, false);
        }
        let xf = s.x.as_vec2();
        if seen(f, xf) {
            room.label(f, "x", xf.extend(0.0), "$x$", 16.0, pal::TEXT, away(f, xf), 10.0, false);
        }
        let k_now = k_shown.round() as usize;
        if seen(f, nowf) && long_px(f, nowf, xf) > 20.0 {
            room.label(f, "now", nowf.extend(0.0), &format!("$x_{{{k_now}}}$"), 15.0, pal::WHITE, away(f, nowf), 10.0, true);
        }
        // each eigenline beside itself, well inside the picture, nearer its upper end if
        // there is room there (the orange one's on one side, the purple one's on the other;
        // well off the line: a wide name beside a slanting line needs the room)
        let (min, max) = lay.frame_room();
        let room_lo = on_screen_inverse(f, Vec2::new(min.x, max.y)).unwrap_or(lo);
        let room_hi = on_screen_inverse(f, Vec2::new(max.x, min.y)).unwrap_or(hi);
        // (a single line's name goes on the side the steps are not on, the side of w they
        // are not: the steps keep the sign of their part along w when λ > 0)
        let w_side = away(f, wf) * if lines.len() == 1 && shape.c.y > 0.0 { -1.0 } else { 1.0 };
        for (i, (d, l, c)) in lines.iter().enumerate() {
            let up_end = d.as_vec2() * if d.y < 0.0 { -1.0 } else { 1.0 };
            if let Some((a, b)) = line_in_box(up_end, room_lo, room_hi) {
                let points: Vec<Vec3> = [0.8, 0.62, 0.45].iter().map(|t| b * *t).chain([0.8, 0.62, 0.45].iter().map(|t| a * *t)).map(|p| p.extend(0.0)).collect();
                let text = if lines.len() == 1 { format!(r"eigenline, $\lambda = {}$", num(*l, 2)) } else { format!(r"$\lambda = {}$", num(*l, 2)) };
                let side = if i == 1 { -w_side } else { w_side };
                room.label_any(f, ("eigen", i as u32), &points, &text, 13.0, *c, side, 16.0, true);
            }
        }
        if self.parts {
            // (well off their arrows, for the same reason)
            if long_px(f, Vec2::ZERO, along_w) > 46.0 {
                let mid = along_w * 0.5;
                room.label(f, "part w", mid.extend(0.0), &format!(r"${}\,w$", num(now_coords.y, 2)), 13.0, pal::GREEN, -away(f, vf), 14.0, true);
            }
            if long_px(f, along_w, nowf) > 46.0 {
                let mid = (along_w + nowf) * 0.5;
                room.label(f, "part v", mid.extend(0.0), &format!(r"${}\,v$", num(now_coords.x, 2)), 13.0, pal::PINK, away(f, wf), 14.0, true);
            }
        }
        if let (Some((u1, u2)), Some((_, det, _))) = (sliver, scissors(&s)) {
            let far = u1 + u2;
            if long_px(f, Vec2::ZERO, far) > 50.0 {
                room.label(f, "area", far.extend(0.0), &format!(r"$|\det V| = {}$", small_or(det)), 12.5, pal::YELLOW, away(f, far), 8.0, true);
            }
        }
        plates
    }

    /// The heading at the top of the picture: the kind of matrix and its eigenvalues, and
    /// in a line under it what that means. Returns the box it takes.
    fn heading(&self, f: &mut Frame, lay: &Layout, s: &Setting) -> (Vec2, Vec2) {
        let at = lay.picture.0 + Vec2::new(16.0, 14.0);
        let (first, second) = heading_text(s);
        let room = (lay.picture.1.x - lay.picture.0.x - 32.0).max(40.0);
        let size = if f.text_size(&first, 15.0).x <= room { 15.0 } else { 13.0 };
        let second = if f.text_size(second[0], 12.5).x <= room { second[0] } else { second[1] };
        let small = if f.text_size(second, 12.5).x <= room { 12.5 } else { 11.0 };
        let wide = f.text_size(&first, size).x.max(f.text_size(second, small).x).min(room);
        f.caption_px(at, first, size, pal::TEXT, Align2::LEFT_TOP);
        f.caption_px(at + Vec2::new(0.0, size + 8.0), second, small, pal::DIM, Align2::LEFT_TOP);
        (at - Vec2::splat(6.0), at + Vec2::new(wide + 12.0, size + small + 16.0))
    }

    /// The screen panel: A, Jᵏ, A = P J P⁻¹ written out, the chart of |Aᵏx|, the
    /// eigenvalues in the complex plane, and how far they move against ε.
    fn panel(&mut self, f: &mut Frame, p: &Panel) {
        hud_panel(f, p.min, p.max);
        let s = self.shown();
        let shape = Shape::of(&s);
        let t = p.text;
        let k = self.k_shown.round() as usize;
        let changed = s.eps != 0.0;
        let j_name = if changed { r"J_\varepsilon" } else { "J" };

        // A = [ ]   Jᵏ = [ ]
        let msize = matrix_size(2, 2, CELL, t);
        let mid = p.row1.y + msize.y * 0.5;
        let mut x = p.row1.x;
        let lead = |f: &Frame, text: &str| f.text_size(text, t).x;
        f.text_px(Vec2::new(x, mid), "$A =$", t, pal::TEXT, Align2::LEFT_CENTER);
        x += lead(f, "$A =$") + 6.0;
        x += matrix(f, Vec2::new(x, p.row1.y), &entries(shape.a, [pal::TEXT, pal::TEXT]), t).x + 14.0;
        let power = format!("${j_name}^{{{k}}} =$");
        f.text_px(Vec2::new(x, mid), power.as_str(), t, pal::TEXT, Align2::LEFT_CENTER);
        x += lead(f, &power) + 6.0;
        // (J's own by the formula the panel gives; J_ε's by multiplying)
        let jk = if changed { shape.b_power(k) } else { jordan_power(s.lambda, k) };
        matrix(f, Vec2::new(x, p.row1.y), &entries(jk, [pal::TEXT, pal::TEXT]), t);

        // A = P J P⁻¹, written out
        f.text_px(p.title, format!(r"$A = P\,{j_name}\,P^{{-1}}$:"), 12.0, pal::DIM, Align2::LEFT_TOP);
        let mut x = p.row2.x;
        let j_label = format!("${j_name}$");
        let names = [r"$P = [\,v\;\; w\,]$", j_label.as_str(), r"$P^{-1}$"];
        let mut j_rows = entries(shape.b, [pal::TEXT, pal::TEXT]);
        j_rows[1][0].1 = if changed { pal::RED } else { pal::DIM };
        let mats = [entries(shape.p, [pal::PINK, pal::GREEN]), j_rows, entries(shape.p_inv, [pal::DIM.mix(pal::TEXT, 0.5), pal::DIM.mix(pal::TEXT, 0.5)])];
        for (m, name) in mats.iter().zip(names) {
            let size = matrix(f, Vec2::new(x, p.row2.y), m, t);
            f.text_px(Vec2::new(x + size.x * 0.5, p.row2.y + size.y + 2.0), name, 11.5, pal::DIM, Align2::CENTER_TOP);
            x += size.x + 10.0;
        }

        self.hump_chart(f, p, &shape);
        eigen_chart(f, p.eig, &s);
        moved_chart(f, p.moved, &s);
    }

    /// The chart of |Aᵏx| against k, with a mark at the step shown that can be dragged,
    /// and the play button.
    fn hump_chart(&mut self, f: &mut Frame, p: &Panel, shape: &Shape) {
        let sizes = shape.sizes();
        let s = self.shown();
        let rho = spectral_radius(s.lambda, s.eps);
        let top = sizes.iter().fold(0.0f64, |m, v| m.max(*v)).max(1e-9) * 1.18;
        let plot = Plot::new(p.hump, [-0.7, K as f64 + 0.7], [0.0, top]);
        // (the button first: a click on it is its own, and not a click on the chart)
        let busy = self.playing || self.play_when_there;
        if plot::button(f, p.play, if busy { "❚❚" } else { "▶" }, 13.0, false) {
            if busy {
                self.pause();
            } else {
                self.play();
            }
        }
        // the title says what the bars are, and the key beside it what the orange line is
        title(f, p.hump, CHART_LEFT, r"bars: $|A^k x|$ at each step $k$");
        let key = if kind(s.eps) == Kind::One { r"orange: $|\lambda|^k\,|x|$" } else { r"orange: $|\lambda_1|^k\,|x|$" };
        f.text_px(Vec2::new(p.play.0.x - 10.0, p.hump.0.y - 7.0), key, 11.5, pal::ORANGE, Align2::RIGHT_BOTTOM);
        plot.y_grid(f);
        plot.x_axis(f, "", 10.5);
        plot.y_axis(f, "", 10.5);
        for (i, n) in sizes.iter().enumerate() {
            let on = i as f64 <= self.k_shown + 1e-9;
            plot.bar(f, i as f64 - 0.3, i as f64 + 0.3, 0.0, *n, 0.0, pal::WHITE.a(if on { 0.7 } else { 0.22 }));
        }
        // what the eigenvalues alone would give: (largest |λ|)ᵏ |x|
        let start = sizes[0];
        let pts: Vec<Vec2> = (0..=K).map(|i| plot.at(i as f64, rho.powi(i as i32) * start)).collect();
        plot::path_of(f, plot::OWN, &plot::clip(&pts, plot.rect), 1.6, pal::ORANGE.a(0.9));
        // the highest step, if it is a hump: higher than the start, and before the end.
        // Named to its right, or to its left near the chart's right end, and gliding
        // from the one to the other (`Frame::text_placed`).
        let (top_k, top_n) = largest(&sizes);
        if !plot.tiny() && top_k > 0 && top_k < K && top_n > 1.02 * sizes[0] {
            let (min, max) = p.hump;
            let at = plot.at(top_k as f64, top_n);
            let text = format!("highest: $k = {top_k}$");
            let half = f.text_size(&text, 10.5) * 0.5;
            let right = at.x < min.x + (max.x - min.x) * 0.6;
            let centre = at + Vec2::new(if right { 9.0 + half.x } else { -9.0 - half.x }, -1.0 - half.y);
            f.text_placed("highest", 0, min, centre - min, text, 10.5, pal::TEXT);
        }
        // the step shown: a mark to drag along the chart
        let ks = self.k_shown.clamp(0.0, K as f64);
        let i = (ks.floor() as usize).min(K - 1);
        let height = sizes[i] + (sizes[i + 1] - sizes[i]) * (ks - i as f64);
        plot.vline(f, plot::MARKS, ks, 1.2, pal::WHITE.a(0.5), false);
        // (while it is dragged, k follows it between whole steps, as when playing)
        if let Some((to, _)) = plot.handle(f, 4, ks, height, pal::WHITE) {
            (self.playing, self.play_when_there, self.scrubbing) = (false, false, true);
            self.k = to.clamp(0.0, K as f64);
        }
        if !f.handles.dragging() {
            self.scrubbing = false;
        }
        // a click on the chart chooses that step too
        if f.clicked_canvas() {
            if let Some((at, _)) = plot.under_pointer(f) {
                self.choose_step(at);
            }
        }
    }
}

/// A chart's title, above its rectangle and over the numbers to its left (`left` pixels).
fn title(f: &mut Frame, (min, _): (Vec2, Vec2), left: f32, text: &str) {
    f.text_px(Vec2::new(min.x - left, min.y - 7.0), text, 12.0, pal::TEXT, Align2::LEFT_BOTTOM);
}

impl Shape {
    /// J_εᵏ, by multiplying.
    fn b_power(&self, k: usize) -> DMat2 {
        (0..k.min(K)).fold(DMat2::IDENTITY, |m, _| m * self.b)
    }
}

/// The scene's point under the screen point `px`, for the flat view.
fn on_screen_inverse(f: &Frame, px: Vec2) -> Option<Vec2> {
    f.cam.pick_ground(px * f.px).map(|p| p.truncate())
}

/// The eigenvalues in the complex plane: the unit circle, λ (J's double eigenvalue) as a
/// ring, and the eigenvalues of J_ε as dots, with the red way they have come from λ.
fn eigen_chart(f: &mut Frame, rect: (Vec2, Vec2), s: &Setting) {
    let (min, max) = rect;
    let size = max - min;
    // (one scale both ways, as large as shows the real axis from −1.75 to 1.75 and the
    // imaginary one from −1.15 to 1.15: the circle |λ| = 1 whole, whatever the chart's shape)
    let per = (size.x as f64 / 3.5).min(size.y as f64 / 2.3).max(1e-6);
    let (half_re, half_im) = (size.x as f64 * 0.5 / per, size.y as f64 * 0.5 / per);
    let plot = Plot::new(rect, [-half_re, half_re], [-half_im, half_im]);
    title(f, rect, 0.0, "the eigenvalues");
    if plot.tiny() {
        return;
    }
    plot::line_of(f, plot::AXES, plot.at(-half_re, 0.0), plot.at(half_re, 0.0), 1.0, pal::DIM.a(0.6));
    plot::line_of(f, plot::AXES, plot.at(0.0, -half_im), plot.at(0.0, half_im), 1.0, pal::DIM.a(0.6));
    let circle: Vec<Vec2> = (0..=72).map(|i| { let a = i as f64 / 72.0 * std::f64::consts::TAU; plot.at(a.cos(), a.sin()) }).collect();
    plot::path_of(f, plot::AXES, &plot::clip(&circle, rect), 1.0, pal::DIM.a(0.55));
    f.text_px(plot.at(1.0, 0.0) + Vec2::new(3.0, 2.0), "$1$", 10.0, pal::DIM, Align2::LEFT_TOP);
    f.text_px(plot.at(-1.0, 0.0) + Vec2::new(-3.0, 2.0), "$-1$", 10.0, pal::DIM, Align2::RIGHT_TOP);
    if half_im > 1.05 {
        f.text_px(plot.at(0.0, 1.0) + Vec2::new(4.0, -2.0), "$i$", 10.0, pal::DIM, Align2::LEFT_BOTTOM);
    }
    f.text_px(Vec2::new(max.x - 2.0, min.y + 2.0), r"$|\lambda| = 1$", 10.0, pal::DIM, Align2::RIGHT_TOP);
    // λ, and where J_ε's eigenvalues have gone from it
    let centre = plot.at(s.lambda, 0.0);
    let eig = eigenvalues(s.lambda, s.eps);
    for (k, (re, im)) in eig.iter().enumerate() {
        let to = plot.at(*re, *im);
        let pts = plot::clip(&[centre, to], rect);
        plot::path_of(f, plot::OWN + 1 + k as u8, &pts, 2.0, pal::RED.a(0.85));
    }
    if inside_rect(centre, rect) {
        hud_circle(f, centre, 6.0, 1.3, pal::TEXT.a(0.8));
    }
    let colours = [pal::ORANGE, pal::PURPLE];
    for (k, (re, im)) in eig.iter().enumerate().rev() {
        plot.dot(f, *re, *im, 8.0, colours[k]);
    }
    let label = match kind(s.eps) {
        Kind::One => format!(r"$\lambda = {}$, twice", num(s.lambda, 2)),
        Kind::Two => format!(r"${}$ and ${}$", num(eig[0].0, 2), num(eig[1].0, 2)),
        Kind::Complex => format!(r"${} \pm {}i$", num(eig[0].0, 2), num(eig[0].1, 2)),
    };
    f.text_px(Vec2::new(min.x + 2.0, max.y + 4.0), label, 11.0, pal::TEXT, Align2::LEFT_TOP);
}

/// How far the eigenvalues move against the size of the change, both on logarithmic
/// scales from 10⁻⁶ to 1: √|ε| for J_ε (a line of slope ½), and at most |ε| for a
/// symmetric matrix changed as much (slope 1), with the change now as a red mark.
fn moved_chart(f: &mut Frame, rect: (Vec2, Vec2), s: &Setting) {
    let plot = Plot::new(rect, [-6.0, 0.0], [-6.0, 0.0]);
    title(f, rect, CHART_LEFT, r"how far they move, against $|\varepsilon|$");
    if plot.tiny() {
        return;
    }
    let (min, max) = rect;
    plot::line_of(f, plot::AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
    plot::line_of(f, plot::AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
    for (v, name) in [(-6.0, "$10^{-6}$"), (-4.0, "$10^{-4}$"), (-2.0, "$10^{-2}$"), (0.0, "$1$")] {
        let x = plot.px_x(v);
        plot::line_of(f, plot::AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
        f.text_px(Vec2::new(x, max.y + 6.0), name, 9.5, pal::DIM, Align2::CENTER_TOP);
        let y = plot.px_y(v);
        plot::line_of(f, plot::AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8));
        f.text_px(Vec2::new(min.x - 6.0, y), name, 9.5, pal::DIM, Align2::RIGHT_CENTER);
        plot::line_of(f, plot::AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.12));
    }
    plot.path(f, plot::CURVES, [(-6.0, -3.0), (0.0, 0.0)], 1.8, pal::ORANGE);
    plot.path(f, plot::CURVES, [(-6.0, -6.0), (0.0, 0.0)], 1.8, pal::BLUE);
    // (each name beside its own line: above the orange one, below the blue one; what ε
    // is now in the empty corner at the top left)
    f.text_px(plot.at(-5.8, -2.0), r"$J_\varepsilon$: $\sqrt{|\varepsilon|}$", 10.5, pal::ORANGE, Align2::LEFT_BOTTOM);
    f.text_px(plot.at(-2.5, -3.7), "symmetric:", 10.5, pal::BLUE, Align2::LEFT_TOP);
    f.text_px(plot.at(-2.5, -3.7) + Vec2::new(0.0, 13.0), r"at most $|\varepsilon|$", 10.5, pal::BLUE, Align2::LEFT_TOP);
    if s.eps == 0.0 {
        f.text_px(Vec2::new(min.x + 5.0, min.y + 3.0), r"$\varepsilon = 0$: no change", 10.5, pal::DIM, Align2::LEFT_TOP);
    } else {
        let at = s.eps.abs().log10().clamp(-6.0, 0.0);
        // (what ε is, at the top of its mark, on whichever side of it there is room)
        let x = plot.px_x(at);
        let (side, align) = if x < min.x + 0.55 * (max.x - min.x) { (4.0, Align2::LEFT_TOP) } else { (-4.0, Align2::RIGHT_TOP) };
        f.text_px(Vec2::new(x + side, min.y + 3.0), format!(r"$\varepsilon = {}$", small(s.eps)), 10.5, pal::RED, align);
        plot.vline(f, plot::MARKS, at, 1.3, pal::RED.a(0.8), false);
        plot.dot(f, at, at * 0.5, 7.0, pal::ORANGE);
        plot.dot(f, at, at, 7.0, pal::BLUE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(m: DMat2, n: DMat2, tol: f64) -> bool {
        (m - n).to_cols_array().iter().all(|v| v.abs() <= tol)
    }

    /// The scene as an experiment's button leaves it, run at 60 frames a second until
    /// everything has come to rest (or for `seconds` at most).
    fn run(i: usize, seconds: f32) -> Jordan {
        let mut scene = Jordan::new();
        scene.set_up(i);
        for _ in 0..(seconds * 60.0) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if !scene.animated() {
                break;
            }
        }
        scene
    }

    /// Jᵏ in closed form is J multiplied by itself k times, for every λ the slider
    /// allows, 0 among them.
    #[test]
    fn the_powers_of_j_are_what_the_formula_says() {
        for lambda in [-1.2, -0.9, -0.3, 0.0, 0.5, 0.9, 1.0, 1.2] {
            let mut m = DMat2::IDENTITY;
            for k in 0..=K {
                assert!(close(jordan_power(lambda, k), m, 1e-12 * m.to_cols_array().iter().fold(1.0f64, |a, v| a.max(v.abs()))), "λ = {lambda}, k = {k}");
                m *= block(lambda, 0.0);
            }
        }
    }

    /// e^(Jt) = e^(λt) [1 t; 0 1], against the series.
    #[test]
    fn the_flow_of_j_is_what_the_formula_says() {
        for lambda in [-1.0, -0.2, 0.0, 0.9] {
            for t in [0.0, 0.5, 1.0, 2.5] {
                let j = block(lambda, 0.0);
                let (mut sum, mut term) = (DMat2::IDENTITY, DMat2::IDENTITY);
                for n in 1..60 {
                    term = term * j * (t / n as f64);
                    sum += term;
                }
                let formula = rows([1.0, t, 0.0, 1.0]) * (lambda * t).exp();
                assert!(close(sum, formula, 1e-12 * formula.to_cols_array().iter().fold(1.0f64, |a, v| a.max(v.abs()))), "λ = {lambda}, t = {t}");
            }
        }
    }

    /// A = P J_ε P⁻¹ is what the scene says of it: at ε = 0, Av = λv, Aw = λw + v and
    /// (A − λI) is not 0 but squares to 0 (one eigenline); for ε ≠ 0 its eigenvalues are
    /// λ ± √ε by trace and determinant, with eigenvectors v ± √ε w when ε > 0.
    #[test]
    fn a_is_built_from_its_chain() {
        let mut rng = Rng::new(7);
        for _ in 0..300 {
            let v = DVec2::new(rng.range(-2.0, 2.0) as f64, rng.range(-2.0, 2.0) as f64);
            let w = DVec2::new(rng.range(-2.0, 2.0) as f64, rng.range(-2.0, 2.0) as f64);
            if !chain_ok(v, w) {
                continue;
            }
            let lambda = rng.range(-1.2, 1.2) as f64;
            for eps in [0.0, 1e-8, 1e-4, 0.04, 0.25, -1e-4, -0.04, -0.25] {
                let s = Setting { lambda, eps, v, w, x: DVec2::ONE };
                let a = Shape::of(&s).a;
                let n = a - DMat2::IDENTITY * lambda;
                let scale = a.to_cols_array().iter().fold(1.0f64, |m, e| m.max(e.abs()));
                if eps == 0.0 {
                    assert!((a * v - v * lambda).length() < 1e-12 * scale * v.length());
                    assert!((a * w - w * lambda - v).length() < 1e-12 * scale * (v.length() + w.length()));
                    assert!(close(n * n, DMat2::ZERO, 1e-11 * scale * scale) && n.to_cols_array().iter().any(|e| e.abs() > 1e-3));
                }
                let [(r1, i1), (r2, i2)] = eigenvalues(lambda, eps);
                let tr = a.x_axis.x + a.y_axis.y;
                assert!((tr - (r1 + r2)).abs() < 1e-11 * scale, "{eps}");
                let det = r1 * r2 - i1 * i2;
                assert!((a.determinant() - det).abs() < 1e-11 * scale * scale, "{eps}: {} against {det}", a.determinant());
                if eps > 0.0 {
                    let g = eps.sqrt();
                    for (u, l) in [(v + w * g, r1), (v - w * g, r2)] {
                        assert!((a * u - u * l).length() < 1e-11 * scale * u.length(), "{eps}");
                    }
                }
            }
        }
    }

    /// The steps drawn are the powers of A applied to x.
    #[test]
    fn the_steps_are_the_powers_of_a() {
        for i in 0..TRIALS.len() {
            let s = run(i, 60.0).shown();
            let shape = Shape::of(&s);
            let mut p = s.x;
            for k in 0..=K {
                assert!((shape.steps[k] - p).length() < 1e-9 * p.length().max(1.0), "trial {i}, step {k}");
                p = shape.a * p;
            }
            // the power written on the panel: for J itself, the closed form
            if s.eps == 0.0 {
                assert!(close(shape.b_power(9), jordan_power(s.lambda, 9), 1e-12));
            }
        }
    }

    #[test]
    fn the_scissors_close_as_the_caption_says() {
        let mut scene = Jordan::new();
        scene.set_up(0);
        let mut seen = Vec::new();
        for _ in 0..(60.0 * 12.0) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            // (once the glide from where it was has arrived)
            if scene.glide.arrived() {
                seen.push(scene.shown().eps);
            }
        }
        // from 0.16 down to 0, never up, and ending at 0 exactly: one eigenline
        assert!(seen[0] == 0.16 && seen.windows(2).all(|w| w[1] <= w[0]), "{seen:?}");
        let s = scene.shown();
        assert!(s.eps == 0.0 && kind(s.eps) == Kind::One && s.lambda == 0.5 && scene.tour.is_none());
        let at = |eps: f64| Setting { eps, ..s };
        assert_eq!(eigenvalues(0.5, 0.16).map(|e| num(e.0, 2)), ["0.90", "0.10"]);
        let numbers = |eps: f64| scissors(&at(eps)).unwrap();
        assert!(num(numbers(0.16).1, 2) == "0.69" && num(numbers(0.01).1, 2) == "0.20" && num(numbers(0.0001).1, 2) == "0.02", "{:?}", [numbers(0.16), numbers(0.01), numbers(0.0001)]);
        assert!(numbers(0.0001).2 > 40.0 && numbers(0.0001).2 < 41.0, "{:?}", numbers(0.0001));
        // the eigenlines close onto v's line: the angle falls to nothing with ε
        assert!(numbers(0.16).0 > 43.0 && numbers(0.0001).0 < 1.2 && numbers(1e-10).0 < 2e-3);
        assert!(unit_eigenvectors(&at(1e-10)).is_some_and(|(u1, u2)| (u1 - s.v.normalize()).length() < 1e-4 && (u2 - s.v.normalize()).length() < 1e-4));
    }

    #[test]
    fn the_shear_does_what_its_caption_says() {
        let scene = run(1, 60.0);
        let s = scene.shown();
        assert!(scene.k == 7.0 && !scene.playing);
        assert!(s.lambda == 1.0 && s.eps == 0.0 && s.v == DVec2::X && s.w == DVec2::Y && s.x == DVec2::new(0.0, 0.4));
        let shape = Shape::of(&s);
        assert!(close(shape.a, rows([1.0, 1.0, 0.0, 1.0]), 0.0));
        for k in 0..=K {
            assert!(close(jordan_power(1.0, k), rows([1.0, k as f64, 0.0, 1.0]), 0.0));
            let p = shape.steps[k];
            assert!((p - DVec2::new(0.4 * k as f64, 0.4)).length() < 1e-12, "{k}: {p}");
            assert!((p.length() - 0.4 * ((k * k + 1) as f64).sqrt()).abs() < 1e-12);
        }
        // the x axis stays put
        assert_eq!(shape.a * DVec2::new(2.5, 0.0), DVec2::new(2.5, 0.0));
    }

    #[test]
    fn the_hump_is_where_the_caption_says() {
        let scene = run(2, 60.0);
        let s = scene.shown();
        assert!(scene.k == K as f64 && !scene.playing && s.lambda == 0.9 && s.eps == 0.0);
        assert!((s.x - s.w * 1.25).length() < 1e-15);
        let shape = Shape::of(&s);
        // A^k x = 1.25 (kλ^(k−1) v + λ^k w)
        for k in 0..=K {
            let want = (s.v * if k == 0 { 0.0 } else { k as f64 * 0.9f64.powi(k as i32 - 1) } + s.w * 0.9f64.powi(k as i32)) * 1.25;
            assert!((shape.steps[k] - want).length() < 1e-12);
        }
        let sizes = shape.sizes();
        let (top_k, top) = largest(&sizes);
        assert!(top_k == 9 && num(top / sizes[0], 2) == "3.89", "{top_k} {}", top / sizes[0]);
        let below = (1..=K).find(|k| sizes[*k] < sizes[0]);
        assert_eq!(below, Some(35));
        assert!(spectral_radius(0.9, 0.0) < 1.0 && sizes[K] < sizes[34]);
        // λ^k alone shrinks at every step
        assert!((1..=K).all(|k| 0.9f64.powi(k as i32) < 0.9f64.powi(k as i32 - 1)));
        // halve w, x left where it is: the hump climbs to 7.76
        let halved = Shape::of(&Setting { w: s.w * 0.5, ..s }).sizes();
        let (_, top) = largest(&halved);
        assert_eq!(num(top / halved[0], 2), "7.76");
    }

    #[test]
    fn the_tiny_change_does_what_its_caption_says() {
        let scene = run(3, 60.0);
        let s = scene.shown();
        assert!(s.eps == 0.0001 && s.lambda == 0.9 && scene.tour.is_none() && kind(s.eps) == Kind::Two);
        let [(r1, _), (r2, _)] = eigenvalues(s.lambda, s.eps);
        assert!(num(r1, 2) == "0.91" && num(r2, 2) == "0.89" && (r1 - 0.9 - 0.01).abs() < 1e-15 && (0.9 - r2 - 0.01).abs() < 1e-15);
        // A's entries change by less than 0.0001, and its numbers on the screen not at all
        let (before, after) = (Shape::of(&Setting { eps: 0.0, ..s }).a, Shape::of(&s).a);
        assert!((after - before).to_cols_array().iter().all(|d| d.abs() < 1e-4));
        assert!(before.to_cols_array().iter().zip(after.to_cols_array()).all(|(p, q)| cell(*p) == cell(q)));
        // the numbers computed from A agree: the eigenvalues of A by its trace and determinant
        let (tr, det) = (after.x_axis.x + after.y_axis.y, after.determinant());
        let disc = (tr * tr / 4.0 - det).sqrt();
        assert!((tr / 2.0 + disc - r1).abs() < 1e-9 && (tr / 2.0 - disc - r2).abs() < 1e-9);
        // a symmetric matrix changed by 0.0001 (in size) moves its eigenvalues by at most
        // that: tried on symmetric matrices with every kind of change of that size
        let mut rng = Rng::new(9);
        for _ in 0..500 {
            let (a, b, c) = (rng.range(-1.0, 1.0) as f64, rng.range(-1.0, 1.0) as f64, rng.range(-1.0, 1.0) as f64);
            let sym = rows([a, b, b, c]);
            let e = rows([0; 4].map(|_| rng.range(-1.0, 1.0) as f64));
            // (scaled to size 1e-4: its largest stretch, the square root of the larger
            // eigenvalue of EᵀE)
            let ete = e.transpose() * e;
            let (t, d) = (ete.x_axis.x + ete.y_axis.y, ete.determinant());
            let big = (0.5 * (t + (t * t - 4.0 * d).max(0.0).sqrt())).sqrt();
            let e = e * (1e-4 / big.max(1e-12));
            let ev = |m: DMat2| -> [(f64, f64); 2] {
                let (tr, det) = (m.x_axis.x + m.y_axis.y, m.determinant());
                let q = tr * tr / 4.0 - det;
                if q >= 0.0 {
                    [(tr / 2.0 + q.sqrt(), 0.0), (tr / 2.0 - q.sqrt(), 0.0)]
                } else {
                    [(tr / 2.0, (-q).sqrt()), (tr / 2.0, -(-q).sqrt())]
                }
            };
            let (before, after) = (ev(sym), ev(sym + e));
            for (re, im) in after {
                let nearest = before.iter().map(|(r, _)| (re - r).hypot(im)).fold(f64::MAX, f64::min);
                assert!(nearest <= 1e-4 * 1.01 + 1e-12, "{nearest}");
            }
        }
    }

    #[test]
    fn the_three_kinds_are_what_the_caption_says() {
        let mut scene = Jordan::new();
        scene.set_up(4);
        let mut kinds: Vec<Kind> = Vec::new();
        for _ in 0..(60.0 * 20.0) as usize {
            scene.advance(1.0 / 60.0, 1.0 / 60.0);
            if !scene.glide.arrived() {
                continue;
            }
            let s = scene.shown();
            let k = kind(s.eps);
            if kinds.last() != Some(&k) {
                kinds.push(k);
            }
            // the trace stays 1.5 all the way
            let a = Shape::of(&s).a;
            assert!((a.x_axis.x + a.y_axis.y - 1.5).abs() < 1e-12);
        }
        assert_eq!(kinds, [Kind::Two, Kind::One, Kind::Complex]);
        let s = scene.shown();
        assert!(s.eps == -0.04 && s.lambda == 0.75 && scene.tour.is_none());
        assert_eq!(eigenvalues(0.75, 0.04).map(|e| num(e.0, 2)), ["0.95", "0.55"]);
        assert_eq!(eigenvalues(0.75, -0.04), [(0.75, 0.2), (0.75, -0.2)]);
        // det A rises through (tr A)²/4 = 0.5625: below, on, above
        for (eps, sign) in [(0.04, -1.0), (0.0, 0.0), (-0.04, 1.0)] {
            let a = Shape::of(&Setting { eps, ..s }).a;
            let gap = a.determinant() - 0.5625;
            assert!(if sign == 0.0 { gap.abs() < 1e-12 } else { gap * sign > 0.03 }, "{eps}: {gap}");
        }
        // with complex eigenvalues the steps go round 0: the angle of x_k keeps turning,
        // the same way, while |x_k| shrinks to nothing
        let shape = Shape::of(&s);
        let turns: Vec<f64> = shape.steps.windows(2).map(|w| w[0].perp_dot(w[1])).collect();
        assert!(turns.iter().all(|t| *t > 0.0) || turns.iter().all(|t| *t < 0.0), "{turns:?}");
        assert!(shape.sizes()[K] < 1e-3 * shape.sizes()[0]);
    }

    /// What each experiment is about lies in the part of the plane the view is fitted
    /// to: the start, v and w, and the steps as far as the caption speaks of them.
    #[test]
    fn every_experiment_is_in_view() {
        let (lo, hi) = ((FRAME_CENTRE - FRAME_HALF).as_dvec2(), (FRAME_CENTRE + FRAME_HALF).as_dvec2());
        let inside = |p: DVec2| p.cmpge(lo).all() && p.cmple(hi).all();
        for (i, upto) in [(0, K), (1, 7), (2, K), (3, K), (4, K)] {
            let s = run(i, 60.0).shown();
            let shape = Shape::of(&s);
            assert!(inside(s.v) && inside(s.w) && inside(DVec2::ZERO), "trial {i}");
            for k in 0..=upto {
                assert!(inside(shape.steps[k]), "trial {i}, step {k}: {}", shape.steps[k]);
            }
        }
    }

    /// Degenerate input: every number drawn or read out is finite, whatever λ, ε, v, w
    /// and x are within their limits.
    #[test]
    fn degenerate_input_leaves_everything_finite() {
        let chains = [(DVec2::X, DVec2::Y), (DVec2::new(3.0, 0.0), DVec2::new(0.42, 0.42 * 0.99)), (DVec2::new(0.25, 0.0), DVec2::new(-3.0, 0.43)), (DVec2::new(0.6, 0.8), DVec2::new(-0.8, 0.6))];
        for (v, w) in chains {
            for lambda in [-LAMBDA_MAX, -1.0, 0.0, 1e-9, 1.0, LAMBDA_MAX] {
                for eps in [-EPS_MAX, -1e-12, 0.0, 1e-300, 1e-12, 1e-4, EPS_MAX] {
                    for x in [DVec2::ZERO, DVec2::splat(X_MAX), DVec2::new(-X_MAX, 1e-9)] {
                        let s = Setting { lambda, eps, v, w, x };
                        let shape = Shape::of(&s);
                        assert!(shape.steps.iter().all(|p| p.as_vec2().is_finite()) && shape.a.as_mat2().is_finite() && shape.p_inv.as_mat2().is_finite());
                        assert!(shape.sizes().iter().all(|n| n.is_finite()));
                        for k in [0.0, 0.5, 17.3, K as f64, 1e9, -3.0] {
                            let (p, c) = shape.step_at(k);
                            assert!(p.as_vec2().is_finite() && c.as_vec2().is_finite());
                        }
                        if let Some((angle, det, inv)) = scissors(&s) {
                            assert!(angle.is_finite() && det.is_finite() && inv.is_finite() && (0.0..=180.0).contains(&angle), "{eps}: {angle} {det} {inv}");
                        }
                        let text = [cell(shape.a.x_axis.x), small(eps), small_or(eps.abs()), heading_text(&s).0].join(" ");
                        assert!(!text.contains("NaN") && !text.contains("inf") && !text.contains('?'), "{text}");
                    }
                }
            }
        }
    }

    /// A drag that would make v and w nearly parallel, or too long or short, is refused.
    #[test]
    fn the_chain_stays_a_basis() {
        assert!(chain_ok(DVec2::from(V0), DVec2::from(W0)) && chain_ok(DVec2::X, DVec2::Y));
        assert!(!chain_ok(DVec2::X, DVec2::new(1.0, 0.1)) && !chain_ok(DVec2::X, DVec2::ZERO) && !chain_ok(DVec2::X * 4.0, DVec2::Y));
        assert!(!chain_ok(DVec2::X, DVec2::new(f64::NAN, 1.0)));
        for t in TRIALS.iter() {
            assert!(chain_ok(DVec2::from(t.v), DVec2::from(t.w)), "{}", t.label);
        }
    }

    /// A tour starts at its first stop, stays, moves on smoothly and ends at its last.
    #[test]
    fn a_tour_goes_where_it_says() {
        for tour in [SCISSORS, TINY, KINDS] {
            assert_eq!(tour.at(0.0), tour.stops[0].0);
            assert_eq!(tour.at(tour.length() + 1.0), tour.stops.last().unwrap().0);
            let mut last = tour.at(0.0);
            for i in 1..=2000 {
                let e = tour.at(tour.length() * i as f32 / 2000.0);
                assert!((e - last).abs() < 0.01, "a jump in ε");
                last = e;
            }
        }
    }

    /// The panel and the picture fit the canvas, from a phone to a desktop.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (1600.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (390.0, 560.0), (800.0, 600.0), (240.0, 180.0), (620.0, 480.0), (1100.0, 700.0)] {
            for inset in [0.0, 46.0] {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, inset);
                let (min, max) = lay.picture;
                assert!(min.cmpge(Vec2::ZERO).all() && max.cmple(view).all() && (max - min).min_element() > 50.0, "{w}x{h}: {lay:?}");
                if let Some(p) = &lay.panel {
                    let within = |q: Vec2| q.cmpge(p.min - Vec2::splat(0.5)).all() && q.cmple(p.max + Vec2::splat(0.5)).all();
                    assert!(p.min.cmpge(Vec2::ZERO).all() && p.max.cmple(view + Vec2::splat(0.5)).all(), "{w}x{h}: {p:?}");
                    for r in [p.hump, p.play, p.eig, p.moved] {
                        assert!(within(r.0) && within(r.1) && r.1.x > r.0.x + 40.0 - 20.0 * (r == p.play) as i32 as f32, "{w}x{h}: {r:?}");
                    }
                    assert!(p.min.y >= inset || lay.narrow, "{w}x{h}: under the buttons");
                    let row2 = 3.0 * matrix_size(2, 2, CELL, p.text).x + 20.0;
                    assert!(p.row2.x + row2 <= p.max.x, "{w}x{h}: the matrices run off");
                }
            }
        }
    }
}

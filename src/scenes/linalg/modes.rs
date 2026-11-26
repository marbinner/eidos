//! Coupled springs: normal modes.
//!
//! Masses in a row between two walls, joined by springs, obey x'' = −Kx (every mass is
//! 1), with K symmetric. The masses are stepped here as masses on springs, knowing
//! nothing of modes; beside them are the eigenvectors of K, worked out on their own:
//! the motions in which every mass swings in step, each at the square root of its
//! eigenvalue. The tests hold the one to the other.

use super::*;
use crate::math::linalg::eig_sym;
use crate::scenes::stats::plot::{self, Rect};
use std::collections::VecDeque;

pub const DEF: SceneDef = SceneDef {
    id: "modes",
    title: "Coupled springs: normal modes",
    group: EXPONENTIAL,
    blurb: "Masses joined by springs: every motion is a sum of a few pure ones, the eigenvectors of a symmetric matrix, each at its own frequency.",
    make: |_| Box::new(Springs::new()),
};

// ----- the springs and their matrix ------------------------------------------------

/// The most masses, and the fewest.
const MOST: usize = 6;
const FEWEST: usize = 2;
/// The stiffness of every spring but the middle one. (Every mass is 1.)
const K: f64 = 9.0;
/// The middle spring's stiffness at its least and most. At the least, with the most
/// friction, the slowest mode of any row still swings (`spectrum` has why).
const KAPPA_LEAST: f64 = 0.25;
const KAPPA_MOST: f64 = 18.0;
/// The most friction.
const GAMMA_MOST: f64 = 1.0;

/// Whether spring `j` of a row of `n` masses is a middle one: the spring in the middle
/// for an even number of masses, the two either side of the middle mass for an odd
/// number. (Spring j joins mass j − 1 to mass j, counting from 0; "mass −1" and "mass
/// n" are the walls.)
fn is_middle(n: usize, j: usize) -> bool {
    if n.is_multiple_of(2) {
        j == n / 2
    } else {
        j == n / 2 || j == n / 2 + 1
    }
}

/// Each spring's stiffness, from the left wall to the right (those past spring n unused).
fn springs(n: usize, kappa: f64) -> [f64; MOST + 1] {
    std::array::from_fn(|j| if j <= n && is_middle(n, j) { kappa } else { K })
}

/// K, row by row (n by n): the force on mass i is −(Kx)ᵢ, where xᵢ is how far mass i
/// is from where it rests. Spring j is stretched by xⱼ − xⱼ₋₁ and pulls the two masses
/// it joins towards each other with its stiffness times that: it adds its stiffness to
/// both their places on the diagonal and takes it from the two places that join them.
/// So K is symmetric: a spring pulls the masses at its two ends equally hard.
fn stiffness(n: usize, kappa: f64) -> Vec<f64> {
    let s = springs(n, kappa);
    let mut k = vec![0.0; n * n];
    for (j, &s) in s.iter().enumerate().take(n + 1) {
        if j >= 1 {
            k[(j - 1) * n + j - 1] += s;
        }
        if j < n {
            k[j * n + j] += s;
        }
        if j >= 1 && j < n {
            k[(j - 1) * n + j] -= s;
            k[j * n + j - 1] -= s;
        }
    }
    k
}

/// The normal modes of a row: ω² (an eigenvalue of K) and a unit eigenvector for
/// each, slowest first, each turned so that its first number is positive.
#[derive(Clone, Debug, PartialEq)]
struct Spectrum {
    n: usize,
    w2: [f64; MOST],
    v: [[f64; MOST]; MOST],
}

/// The modes of `n` masses with the middle spring at `kappa`.
///
/// No two of them ever have the same frequency over the springs allowed (the tests
/// look), so each mode's shape is a single line, and changes smoothly with `kappa`;
/// and the slowest is never slower than 0.68 a second, so the most friction allowed,
/// which slows a swing to √(ω² − γ²/4), leaves every mode swinging.
fn spectrum(n: usize, kappa: f64) -> Spectrum {
    let (values, vectors) = eig_sym(&stiffness(n, kappa), n);
    let mut s = Spectrum { n, w2: [0.0; MOST], v: [[0.0; MOST]; MOST] };
    for m in 0..n {
        // (`eig_sym` gives the largest first)
        let j = n - 1 - m;
        let v = &vectors[j * n..(j + 1) * n];
        // The first number of an eigenvector of a row is never 0: the first row of
        // Kv = λv gives the second number from the first, the second row the third, and
        // so on (no spring is 0), so a 0 there would make them all 0. Its sign is then a
        // fixed choice that changes smoothly with the springs.
        let sign = if v[0] < 0.0 { -1.0 } else { 1.0 };
        s.w2[m] = values[j];
        for i in 0..n {
            s.v[m][i] = sign * v[i];
        }
    }
    s
}

impl Spectrum {
    fn omega(&self, m: usize) -> f64 {
        self.w2[m].max(0.0).sqrt()
    }

    /// How fast mode `m` turns with friction γ: √(ω² − γ²/4).
    fn damped(&self, m: usize, gamma: f64) -> f64 {
        (self.w2[m] - 0.25 * gamma * gamma).max(1e-12).sqrt()
    }

    /// How far along each mode `x` lies: x · v.
    fn coords(&self, x: &[f64; MOST]) -> [f64; MOST] {
        std::array::from_fn(|m| if m < self.n { (0..self.n).map(|i| self.v[m][i] * x[i]).sum() } else { 0.0 })
    }

    /// Each mode's position q, velocity q' and its pointer: (q, −(q' + γq/2)/ω_d),
    /// which goes round a circle at the rate ω_d and shrinks as e^(−γt/2) (without
    /// friction, (q, −q'/ω)).
    fn pointers(&self, s: &State, gamma: f64) -> [(f64, f64, Vec2); MOST] {
        let (q, p) = (self.coords(&s.x), self.coords(&s.v));
        std::array::from_fn(|m| {
            if m >= self.n {
                return (0.0, 0.0, Vec2::ZERO);
            }
            let w = -(p[m] + 0.5 * gamma * q[m]) / self.damped(m, gamma);
            (q[m], p[m], Vec2::new(q[m] as f32, w as f32))
        })
    }

    /// The energy in each mode: ½(q'² + ω²q²). Together they are the whole energy, as
    /// the modes are perpendicular and K turns each into a multiple of itself.
    fn energies(&self, s: &State) -> [f64; MOST] {
        let (q, p) = (self.coords(&s.x), self.coords(&s.v));
        std::array::from_fn(|m| if m < self.n { 0.5 * (p[m] * p[m] + self.w2[m] * q[m] * q[m]) } else { 0.0 })
    }
}

// ----- the motion ----------------------------------------------------------------

/// Where the masses are (each from its resting place, in units of the distance between
/// two resting places) and how fast they go.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct State {
    x: [f64; MOST],
    v: [f64; MOST],
}

/// The accelerations: −Kx − γv, each mass being 1.
fn accel(k: &[f64], n: usize, s: &State, gamma: f64) -> [f64; MOST] {
    std::array::from_fn(|i| if i < n { -(0..n).map(|j| k[i * n + j] * s.x[j]).sum::<f64>() - gamma * s.v[i] } else { 0.0 })
}

/// The time one step of the masses takes (scene seconds).
const STEP: f64 = 1e-3;

/// One step of length `h`: the springs by the velocity Verlet method (half a kick, a
/// drift, half a kick), which neither gains nor loses energy over any length of time,
/// only wobbles about it by a part in (ωh)²; the friction, exactly, half a step either
/// side. (Each mode is then stepped on its own, by the same method: the step is the
/// same in the modes' coordinates.)
fn step(s: &mut State, k: &[f64], n: usize, gamma: f64, h: f64) {
    let damp = (-0.5 * gamma * h).exp();
    for i in 0..n {
        s.v[i] *= damp;
    }
    let a = accel(k, n, s, 0.0);
    for i in 0..n {
        s.v[i] += 0.5 * h * a[i];
        s.x[i] += h * s.v[i];
    }
    let a = accel(k, n, s, 0.0);
    for i in 0..n {
        s.v[i] += 0.5 * h * a[i];
        s.v[i] *= damp;
    }
}

/// ½ Σ vᵢ² + ½ xᵀKx: the masses' energy of motion and the springs' of stretch.
fn energy(s: &State, k: &[f64], n: usize) -> f64 {
    let kx: f64 = (0..n).map(|i| s.x[i] * (0..n).map(|j| k[i * n + j] * s.x[j]).sum::<f64>()).sum();
    0.5 * ((0..n).map(|i| s.v[i] * s.v[i]).sum::<f64>() + kx)
}

/// A block's width, as a share of the distance between two resting places.
const BLOCK: f64 = 0.24;
/// The furthest a mass may be pulled aside.
const REACH: f64 = 0.45;
/// The shortest any spring may ever get, as the same share: a start is only allowed
/// if no motion that can follow from it brings two blocks, or a block and a wall,
/// nearer than this.
const SHORTEST: f64 = 0.12;

/// Whether masses let go from rest at `x` can never come nearer one another, or a wall,
/// than `SHORTEST` allows.
fn safe(spec: &Spectrum, x: &[f64; MOST]) -> bool {
    safe_moving(spec, &State { x: *x, v: [0.0; MOST] })
}

/// The same for masses moving as `s` says. Mode m's energy, ½(q'² + ω²q²), can only
/// stay the same or fall (with friction), so it never swings further than
/// aₘ = √(q² + q'²/ω²); and the stretch of spring j, xⱼ − xⱼ₋₁, is never more in size
/// than Σ aₘ |vₘⱼ − vₘⱼ₋₁| over the modes.
fn safe_moving(spec: &Spectrum, s: &State) -> bool {
    let n = spec.n;
    let (q, p) = (spec.coords(&s.x), spec.coords(&s.v));
    let a: [f64; MOST] = std::array::from_fn(|m| if m < n { (q[m] * q[m] + p[m] * p[m] / spec.w2[m].max(1e-12)).sqrt() } else { 0.0 });
    let at = |m: usize, i: isize| if i < 0 || i >= n as isize { 0.0 } else { spec.v[m][i as usize] };
    (0..=n).all(|j| {
        let most: f64 = (0..n).map(|m| a[m].abs() * (at(m, j as isize) - at(m, j as isize - 1)).abs()).sum();
        let wall = j == 0 || j == n;
        let room = if wall { 1.0 - 0.5 * BLOCK } else { 1.0 - BLOCK };
        most <= room - SHORTEST
    })
}

/// `x` scaled down, if need be, until it is `safe`.
fn made_safe(spec: &Spectrum, x: [f64; MOST]) -> [f64; MOST] {
    if safe(spec, &x) {
        return x;
    }
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..30 {
        let mid = 0.5 * (lo + hi);
        if safe(spec, &x.map(|v| v * mid)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    x.map(|v| v * lo)
}

/// Where the masses are at time t when let go from rest at `start`: each mode on its
/// own, aₘ e^(−γt/2) (cos ω_d t + γ/(2ω_d) sin ω_d t) along vₘ. (For the tests: the
/// scene steps the masses as masses on springs.)
#[cfg(test)]
fn closed_form(spec: &Spectrum, start: &[f64; MOST], gamma: f64, t: f64) -> [f64; MOST] {
    let a = spec.coords(start);
    let mut x = [0.0; MOST];
    for m in 0..spec.n {
        let wd = spec.damped(m, gamma);
        let q = a[m] * (-0.5 * gamma * t).exp() * ((wd * t).cos() + 0.5 * gamma / wd * (wd * t).sin());
        for i in 0..spec.n {
            x[i] += q * spec.v[m][i];
        }
    }
    x
}

/// The start with the left mass pulled aside: what a new number of masses begins with.
fn pluck() -> [f64; MOST] {
    let mut x = [0.0; MOST];
    x[0] = 0.4;
    x
}

// ----- experiments -----------------------------------------------------------------

struct Trial {
    label: &'static str,
    look_for: &'static str,
    n: usize,
    kappa: f64,
    start: [f64; MOST],
}

const TRIALS: [Trial; 5] = [
    Trial {
        label: "together",
        look_for: r"Both masses pulled the same way, the same distance, and let go. They swing as one, in step, at $\omega = 3.00$: there and back every 2.09 seconds. The middle spring keeps its length the whole time, so each mass feels its outer spring alone: $\omega^2 = k/m = 9$. This is mode 1, along $(1, 1)$: mode 2's pointer stays at $0$, and a change of the middle spring does not change the motion.",
        n: 2,
        kappa: K,
        start: [0.35, 0.35, 0.0, 0.0, 0.0, 0.0],
    },
    Trial {
        label: "against each other",
        look_for: r"Moved towards each other, the same distance each, and let go. They swing in opposite directions, mirror images of each other, at $\omega = \sqrt{27} = 5.20$: there and back every 1.21 seconds. The middle spring is squeezed or stretched by twice as much as either mass moves, so on each mass it pushes or pulls twice as hard as an outer spring would: $\omega^2 = (k + 2k)/m = 27$. This is mode 2, along $(1, -1)$.",
        n: 2,
        kappa: K,
        start: [0.3, -0.3, 0.0, 0.0, 0.0, 0.0],
    },
    Trial {
        label: "pluck one mass",
        look_for: r"The left mass alone pulled aside. $(1, 0)$ is half of $(1, 1)$ plus half of $(1, -1)$, so both modes start with the same amplitude, and keep it: the two pointers stay as long as each other, each turning at its own rate. Their energies differ, 25 % and 75 %, as energy goes with $\omega^2$ and $\omega_2^2 = 3\omega_1^2$. The masses' motion never repeats exactly, as $\omega_2 / \omega_1 = \sqrt 3$ is not a fraction.",
        n: 2,
        kappa: K,
        start: [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    Trial {
        label: "a weak middle spring: beats",
        look_for: r"The middle spring at $\kappa = 1.5$, a sixth of the others, and the left mass pulled. Now $\omega_1 = 3.00$ and $\omega_2 = \sqrt{12} = 3.46$ are close, and the two modes drift out of step and back: in the chart, the left mass's swing shrinks to nothing at 6.77 seconds while the right one's grows to the full size of the start, and at 13.54 seconds it is the other way round again. That beat period is $2\pi / (\omega_2 - \omega_1)$.",
        n: 2,
        kappa: 1.5,
        start: [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
    Trial {
        label: "five masses: sine shapes",
        look_for: r"Five masses, six equal springs, the left mass pulled. In each row the stems lie on a sine curve (dashed) that is $0$ at both walls: for mode $j$ they are in proportion to $\sin(30° j), \sin(60° j), \ldots, \sin(150° j)$. The slowest mode is the smoothest, all one way; the curve of mode $j$ changes sign $j - 1$ times between the walls. The frequencies are $\omega_j = 6 \sin(15° j)$: 1.55, 3.00, 4.24, 5.20 and 5.80. These are waves like those of the Fourier basis, held at $0$ by the walls.",
        n: 5,
        kappa: K,
        start: [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
    },
];

/// The experiment the scene opens with: the left mass plucked.
const OPENING: usize = 2;

// ----- what glides -----------------------------------------------------------------

/// How long a change glides for (seconds).
const GLIDE: f64 = 0.45;

/// Numbers as asked for, and as drawn: the drawn ones go from where they were to where
/// they are asked to be along an ease (setting off gently and arriving gently), in
/// `GLIDE` seconds, and are then exactly there.
#[derive(Clone, Debug, PartialEq)]
struct Ease {
    from: Vec<f64>,
    to: Vec<f64>,
    shown: Vec<f64>,
    /// How far along the way, from 0 to 1.
    done: f64,
}

impl Ease {
    fn at(v: Vec<f64>) -> Ease {
        Ease { from: v.clone(), to: v.clone(), shown: v, done: 1.0 }
    }

    fn arrived(&self) -> bool {
        self.done >= 1.0
    }

    /// Ask for `v`, from wherever the drawn numbers are.
    fn aim(&mut self, v: Vec<f64>) {
        if v != self.to || !self.arrived() {
            (self.from, self.done) = (self.shown.clone(), 0.0);
        }
        self.to = v;
        if self.from == self.to {
            *self = Ease::at(self.to.clone());
        }
    }

    /// Move the drawn numbers `dt` seconds nearer to those asked for.
    fn step(&mut self, dt: f64) {
        if self.arrived() || dt.is_nan() || dt <= 0.0 {
            return;
        }
        self.done = (self.done + dt / GLIDE).min(1.0);
        if self.arrived() {
            *self = Ease::at(self.to.clone());
            return;
        }
        let u = self.done * self.done * (3.0 - 2.0 * self.done);
        for i in 0..self.to.len() {
            self.shown[i] = self.from[i] + (self.to[i] - self.from[i]) * u;
        }
    }
}

/// The row as drawn: for each of the `MOST` places a mass can have, where it rests
/// (a share of the way from the left wall to the right) and how much it is there
/// (0 to 1), then the distance between two resting places (the same share), and the
/// middle spring's stiffness (as its logarithm). A place without a mass rests against
/// the right wall, and a change in the number of masses slides them out from it or
/// back into it.
fn look_of(n: usize, kappa: f64) -> Vec<f64> {
    let mut v = Vec::with_capacity(2 * MOST + 2);
    v.extend((0..MOST).map(|i| if i < n { (i + 1) as f64 / (n + 1) as f64 } else { 1.0 }));
    v.extend((0..MOST).map(|i| if i < n { 1.0 } else { 0.0 }));
    v.push(1.0 / (n + 1) as f64);
    v.push(kappa.ln());
    v
}

/// A place's resting share, how much it is there, the spacing and the middle spring,
/// out of the numbers of `look_of`.
fn rest_of(look: &[f64], i: usize) -> f64 {
    look[i]
}

fn presence(look: &[f64], i: usize) -> f64 {
    look[MOST + i]
}

fn spacing_of(look: &[f64]) -> f64 {
    look[2 * MOST]
}

fn kappa_of(look: &[f64]) -> f64 {
    look[2 * MOST + 1].exp()
}

// ----- the record of the motion ----------------------------------------------------

/// Where every place's mass was at one moment (a share of the way from wall to wall)
/// and how much it was there.
#[derive(Clone, Copy, Debug)]
struct Record {
    t: f64,
    at: [f32; MOST],
    there: [f32; MOST],
}

/// Steps between two records: forty a second.
const RECORD_EVERY: u64 = 25;
/// How long the record is kept (seconds).
const KEPT: f64 = 32.0;
/// The most steps a frame takes: a frame far longer than that falls behind instead.
const MOST_STEPS: usize = 2000;
/// Less energy than this is none: the masses are put exactly at rest.
const NO_ENERGY: f64 = 1e-13;

// ----- the scene -------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Hold {
    /// Moving.
    Free,
    /// Held by a hand on one of the masses: all of them stay where they are.
    Hand,
    /// Gliding to a new start, to be let go from there.
    Placing,
}

pub struct Springs {
    n: usize,
    kappa: f64,
    gamma: f64,
    /// How fast the scene's time runs, on top of the application's time speed.
    speed: f32,
    /// K and its modes, for the middle spring as drawn.
    k: Vec<f64>,
    spec: Spectrum,
    state: State,
    hold: Hold,
    /// Where the masses were last let go from.
    start: [f64; MOST],
    /// While placing: where the masses are, gliding to the new start.
    placing: Option<Ease>,
    /// The row as drawn (`look_of`).
    look: Ease,
    /// Steps taken, and the scene time owed since the last one.
    steps: u64,
    owed: f64,
    history: VecDeque<Record>,
    /// The energy and the size of the motion (the length of all the pointers together)
    /// when the masses were let go: the rows' bars and circles are measured by them.
    energy_ref: f64,
    size_ref: f64,
    /// How long the masses have been exactly at rest (scene seconds), and when they
    /// were last let go (the scene's time).
    still_for: f64,
    let_go_at: f64,
    /// How much the rows of modes are there: they fade in when their number changes.
    rows_in: f32,
    /// When the number of masses last changed (the scene's time), and how much of the
    /// record from before then is still shown: it fades away.
    cut: Option<(f64, f32)>,
}

impl Springs {
    fn new() -> Self {
        let n = TRIALS[OPENING].n;
        let mut scene = Springs {
            n,
            kappa: K,
            gamma: 0.0,
            speed: 1.0,
            k: stiffness(n, K),
            spec: spectrum(n, K),
            state: State::default(),
            hold: Hold::Free,
            start: [0.0; MOST],
            placing: None,
            look: Ease::at(look_of(n, K)),
            steps: 0,
            owed: 0.0,
            history: VecDeque::new(),
            energy_ref: 0.0,
            size_ref: 0.0,
            still_for: 0.0,
            let_go_at: 0.0,
            rows_in: 1.0,
            cut: None,
        };
        scene.set_up(OPENING);
        // (opening, there is nothing to glide from: let go at once)
        scene.look = Ease::at(scene.look.to.clone());
        if let Some(p) = scene.placing.take() {
            scene.state.x = to_array(&p.to);
        }
        scene.let_go();
        scene.rows_in = 1.0;
        scene
    }

    /// Set up experiment `t`: its row and springs, no friction, time at its usual
    /// speed, and the masses glided to its start and let go.
    fn set_up(&mut self, t: usize) {
        let trial = &TRIALS[t];
        self.gamma = 0.0;
        self.speed = 1.0;
        self.set_row(trial.n, trial.kappa);
        self.place(trial.start);
    }

    /// A new number of masses or middle spring: the row glides to it.
    fn set_row(&mut self, n: usize, kappa: f64) {
        let n = n.clamp(FEWEST, MOST);
        let kappa = if kappa.is_finite() { kappa.max(KAPPA_LEAST).min(KAPPA_MOST) } else { K };
        if n != self.n {
            self.rows_in = 0.0;
            // (a record already fading is let go of at once)
            if let Some((t, _)) = self.cut {
                self.history.retain(|r| r.t >= t);
            }
            self.cut = Some((self.now(), 1.0));
        }
        (self.n, self.kappa) = (n, kappa);
        self.look.aim(look_of(n, kappa));
        self.springs_as_drawn();
    }

    /// K and its modes for the row as asked for, with the middle spring as drawn.
    fn springs_as_drawn(&mut self) {
        let kappa = kappa_of(&self.look.shown);
        if self.spec.n != self.n || (kappa - spectrum_kappa(&self.k, self.n)).abs() > 0.0 {
            self.k = stiffness(self.n, kappa);
            self.spec = spectrum(self.n, kappa);
        }
    }

    /// Glide the masses from where they are drawn to `target` (made safe), holding
    /// them; they are let go from there when they and the row have arrived.
    fn place(&mut self, target: [f64; MOST]) {
        let from = self.drawn().x;
        let mut target = target;
        for v in target.iter_mut().skip(self.n) {
            *v = 0.0;
        }
        // (safe for the springs as asked for, which is what they will be)
        let spec = spectrum(self.n, self.kappa);
        let target = made_safe(&spec, target.map(|v| v.max(-REACH).min(REACH)));
        let mut ease = Ease::at(from.to_vec());
        ease.aim(target.to_vec());
        self.placing = Some(ease);
        self.state = State { x: from, v: [0.0; MOST] };
        self.hold = Hold::Placing;
        self.still_for = 0.0;
    }

    /// Start the masses from rest where they are now (once the row is as asked for, and
    /// from a safe start: from one that is not, they glide to the nearest that is).
    fn let_go(&mut self) {
        self.look = Ease::at(self.look.to.clone());
        self.springs_as_drawn();
        self.state.v = [0.0; MOST];
        for v in self.state.x.iter_mut().skip(self.n) {
            *v = 0.0;
        }
        if !safe(&self.spec, &self.state.x) {
            self.place(self.state.x);
            return;
        }
        self.start = self.state.x;
        self.hold = Hold::Free;
        self.placing = None;
        self.let_go_at = self.steps as f64 * STEP;
        self.measure();
    }

    /// The middle spring changed by hand: followed at once, not glided. The masses go on
    /// moving under the new springs where that is safe, and are let go again from rest,
    /// where they are, where it is not.
    fn set_kappa(&mut self, kappa: f64) {
        self.set_row(self.n, kappa);
        self.look = Ease::at(self.look.to.clone());
        self.springs_as_drawn();
        match self.hold {
            Hold::Free if safe_moving(&self.spec, &self.state) => self.measure(),
            Hold::Free => self.place(self.drawn().x),
            Hold::Placing => {
                let to = self.placing.as_ref().map_or(self.state.x, |p| to_array(&p.to));
                self.place(to);
            }
            Hold::Hand => {}
        }
        self.still_for = 0.0;
    }

    /// The energy and size of the motion as it now is, to measure the rows by.
    fn measure(&mut self) {
        self.energy_ref = energy(&self.state, &self.k, self.n);
        self.size_ref = self.spec.pointers(&self.state, self.gamma).iter().map(|p| p.2.length_squared() as f64).sum::<f64>().sqrt();
    }

    /// Start in mode `m` alone, as far out as is safe and no mass further than 0.35.
    fn start_in_mode(&mut self, m: usize) {
        if m >= self.n {
            return;
        }
        // (of the springs as asked for: the drawn ones may still be on their way there)
        let v = spectrum(self.n, self.kappa).v[m];
        let biggest = v.iter().take(self.n).fold(0.0f64, |b, x| b.max(x.abs())).max(1e-9);
        self.place(v.map(|x| x * 0.35 / biggest));
    }

    /// The scene's time now.
    fn now(&self) -> f64 {
        self.steps as f64 * STEP + self.owed
    }

    /// The masses as drawn: between two steps, carried on by the time owed (to the
    /// second power of it: as smooth as the steps themselves).
    fn drawn(&self) -> State {
        if self.hold != Hold::Free {
            return self.state;
        }
        let t = self.owed;
        let a = accel(&self.k, self.n, &self.state, self.gamma);
        State { x: std::array::from_fn(|i| self.state.x[i] + t * self.state.v[i] + 0.5 * t * t * a[i]), v: std::array::from_fn(|i| self.state.v[i] + t * a[i]) }
    }

    /// Nothing moves, and nothing has for as long as the chart shows: the picture no
    /// longer changes.
    fn resting(&self) -> bool {
        self.hold == Hold::Free && self.still_for > KEPT && self.look.arrived()
    }

    /// Move everything on: `dt` seconds of the scene's time (the masses) and `real_dt`
    /// of the clock's (a change of setting glides by the clock, so that it is seen
    /// through even while the scene's time stands still).
    fn advance(&mut self, dt: f32, real_dt: f32) {
        let real = if real_dt.is_finite() { real_dt.clamp(0.0, 0.25) as f64 } else { 0.0 };
        self.look.step(real);
        self.springs_as_drawn();
        if self.rows_in < 1.0 {
            self.rows_in += (1.0 - self.rows_in) * (1.0 - (-real as f32 / 0.12).exp()) + 1e-3;
            if self.rows_in > 0.998 {
                self.rows_in = 1.0;
            }
        }
        if let Some((t, fading)) = &mut self.cut {
            *fading *= (-real as f32 / 0.15).exp();
            if *fading < 0.02 {
                let t = *t;
                self.history.retain(|r| r.t >= t);
                self.cut = None;
            }
        }
        if let Some(p) = &mut self.placing {
            p.step(real);
            self.state.x = to_array(&p.shown);
            if p.arrived() && self.look.arrived() {
                self.let_go();
            }
        }
        if self.resting() {
            return;
        }
        // (the scene's time stands still while the masses are held, by a hand or on
        // their way to a new start: the chart goes on from where it was once they are
        // let go)
        if self.hold != Hold::Free {
            return;
        }
        let dt = if dt.is_finite() { (dt.max(0.0) * self.speed) as f64 } else { 0.0 };
        self.owed += dt;
        let mut taken = 0;
        while self.owed >= STEP && taken < MOST_STEPS {
            self.one_step();
            self.owed -= STEP;
            taken += 1;
        }
        if taken == MOST_STEPS {
            self.owed = self.owed.min(STEP);
        }
    }

    fn one_step(&mut self) {
        if self.hold == Hold::Free {
            step(&mut self.state, &self.k, self.n, self.gamma, STEP);
            if self.state != State::default() && energy(&self.state, &self.k, self.n) < NO_ENERGY {
                self.state = State::default();
            }
        }
        self.still_for = if self.hold == Hold::Free && self.state == State::default() { self.still_for + STEP } else { 0.0 };
        self.steps += 1;
        if self.steps.is_multiple_of(RECORD_EVERY) {
            self.record();
        }
    }

    /// Note where every mass is now.
    fn record(&mut self) {
        let t = self.steps as f64 * STEP;
        let look = &self.look.shown;
        let spacing = spacing_of(look);
        let at = std::array::from_fn(|i| (rest_of(look, i) + self.state.x[i] * spacing) as f32);
        // (none while the row is changing its number of masses: the masses slide along
        // it then, and their traces would sweep across the chart)
        let settled = look[..=2 * MOST] == self.look.to[..=2 * MOST];
        let there = std::array::from_fn(|i| if settled { presence(look, i) as f32 } else { 0.0 });
        self.history.push_back(Record { t, at, there });
        while self.history.front().is_some_and(|r| r.t < t - KEPT) {
            self.history.pop_front();
        }
    }

    /// What the masses are doing, in a few words, for the heading.
    fn doing(&self) -> String {
        if self.hold == Hold::Hand {
            return "held: let go, and the masses start from rest".to_string();
        }
        let s = match &self.placing {
            Some(p) => State { x: to_array(&p.to), v: [0.0; MOST] },
            None => self.drawn(),
        };
        let sizes: Vec<f64> = self.spec.pointers(&s, self.gamma).iter().take(self.n).map(|p| p.2.length() as f64).collect();
        let total = sizes.iter().map(|a| a * a).sum::<f64>().sqrt();
        if total < 1e-9 {
            return "at rest: drag a mass aside and let go".to_string();
        }
        let present: Vec<usize> = (0..self.n).filter(|&m| sizes[m] > 0.005 * total).collect();
        let mut words = match present.as_slice() {
            [m] if self.n == 2 => format!("mode {} alone: the masses swing {}", m + 1, if *m == 0 { "together" } else { "against each other" }),
            [m] => format!(r"mode {} alone: every mass swings in step, at $\omega = {}$", m + 1, num(self.spec.omega(*m), 2)),
            // (two close frequencies: the masses' swings grow and shrink, all the way from
            // one mass to the other where the two modes are about as large)
            [0, 1] if self.n == 2 && self.spec.omega(1) < 1.25 * self.spec.omega(0) => {
                let even = sizes[0] < 1.5 * sizes[1] && sizes[1] < 1.5 * sizes[0];
                let what = if even { "the swing passes to the other mass and back" } else { "each mass's swing grows and shrinks" };
                format!("modes 1 and 2, close in frequency: {what}, every {} seconds", num(beat_period(&self.spec), 2))
            }
            all if all.len() == self.n && self.n == 2 => "both modes at once, each at its own frequency".to_string(),
            all if all.len() == self.n => format!("all {} modes at once, each at its own frequency", WORDS[self.n]),
            some => format!("modes {} at once, each at its own frequency", listed(some)),
        };
        if self.gamma > 0.0 {
            words.push_str(r"; with friction, every mode dies away as $e^{-\gamma t/2}$");
        }
        words
    }

    /// Whether every spring is the same, as drawn and as asked for.
    fn all_equal(&self) -> bool {
        self.kappa == K && (kappa_of(&self.look.shown) - K).abs() < 1e-9
    }
}

/// The middle spring K was made with (that of a row of `n`).
fn spectrum_kappa(k: &[f64], n: usize) -> f64 {
    if k.len() != n * n {
        return f64::NAN;
    }
    // (a middle spring is the stiffness taken from between its two masses; with an odd
    // row, between the middle mass and the one before it)
    let j = n / 2;
    -k[(j - 1) * n + j]
}

/// The beat period of two masses: 2π/(ω₂ − ω₁).
fn beat_period(spec: &Spectrum) -> f64 {
    std::f64::consts::TAU / (spec.omega(1) - spec.omega(0)).max(1e-9)
}

const WORDS: [&str; 8] = ["no", "one", "two", "three", "four", "five", "six", "seven"];

/// "1 and 3", "1, 3 and 5".
fn listed(modes: &[usize]) -> String {
    let names: Vec<String> = modes.iter().map(|m| (m + 1).to_string()).collect();
    match names.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
        _ => names.join(""),
    }
}

fn to_array(v: &[f64]) -> [f64; MOST] {
    std::array::from_fn(|i| v.get(i).copied().unwrap_or(0.0))
}

/// A mode's colour: orange for the slowest, purple for the fastest, between for the rest.
fn mode_colour(m: usize, n: usize) -> Color {
    if n <= 1 {
        return pal::ORANGE;
    }
    pal::ORANGE.mix(pal::PURPLE, m as f32 / (n - 1) as f32)
}

/// A stiffness for the screen: a whole number as one, anything else to two decimals.
fn stiffness_text(s: f64) -> String {
    if (s - s.round()).abs() < 5e-3 {
        num(s, 0)
    } else {
        num(s, 2)
    }
}

// ----- where things go -------------------------------------------------------------

/// The rows of modes: their place and the columns in them, in logical pixels.
#[derive(Clone, Debug)]
struct Rows {
    /// The backing.
    min: Vec2,
    max: Vec2,
    /// The two lines saying what the rows are (none on a phone, where one line goes
    /// over the columns' names).
    header: Option<Vec2>,
    /// Top-left of K written out (none where there is no room).
    matrix: Option<Vec2>,
    /// Where the columns' names go, the first row's top, and each row's height.
    names_y: f32,
    top: f32,
    height: f32,
    /// The button's left and width, the frequency's left, the small row's two ends,
    /// the circle's middle and radius, and the bar's two ends.
    button: (f32, f32),
    freq: f32,
    mini: (f32, f32),
    circle: (f32, f32),
    bar: (f32, f32),
    /// Top-left of the sum of the modes, written out (none where there is no room).
    sum: Option<Vec2>,
    /// Text sizes.
    size: f32,
    small: f32,
}

#[derive(Clone, Debug)]
struct Layout {
    narrow: bool,
    /// The heading's top-left, its sizes and the room across it (none: no room).
    heading: Option<(Vec2, f32, f32, f32)>,
    /// The walls (their inner faces) and the middle of the blocks, across and down.
    left: f32,
    right: f32,
    y: f32,
    /// Half the walls' height.
    wall: f32,
    /// Where the masses' names and the springs' numbers go (none: no room).
    names: Option<(f32, f32)>,
    /// The chart of the masses' positions (time runs down it), and how many pixels
    /// stand for a second.
    chart: Option<(Rect, f32)>,
    rows: Option<Rows>,
    size: f32,
    small: f32,
}

impl Layout {
    fn of(view: Vec2, inset: f32, n: usize) -> Layout {
        let n = n.clamp(FEWEST, MOST) as f32;
        if view.x >= 760.0 && view.y >= 460.0 {
            Self::wide(view, inset, n)
        } else if view.y >= 400.0 && view.x >= 300.0 {
            Self::narrow(view, inset, n)
        } else {
            // too small for anything but the masses (and a heading, if it fits)
            let pad = 8.0;
            let heading = (view.y >= 200.0).then_some((Vec2::new(pad, inset + 6.0), 12.5, 11.0, view.x - 2.0 * pad));
            let top = inset + if heading.is_some() { 44.0 } else { 6.0 };
            let y = (top + 0.5 * (view.y - top)).max(top + 14.0);
            Layout { narrow: true, heading, left: pad + 4.0, right: (view.x - pad - 4.0).max(pad + 40.0), y, wall: 14.0, names: None, chart: None, rows: None, size: 11.0, small: 10.0 }
        }
    }

    fn wide(view: Vec2, inset: f32, n: f32) -> Layout {
        let pad = 18.0;
        let right_w = (view.x * 0.4).clamp(350.0, 470.0);
        let split = view.x - right_w - 10.0;
        let heading = Some((Vec2::new(pad, inset + 14.0), 15.0, 12.5, split - 2.0 * pad));
        let names_y = inset + 86.0;
        let y = names_y + 50.0;
        let (left, right) = (pad + 50.0, split - 24.0);
        let chart_top = y + 46.0;
        let chart = (Vec2::new(left, chart_top), Vec2::new(right, view.y - 40.0));
        // the rows' column, from the top: what they are, K, the columns' names, the rows
        let (min, max) = (Vec2::new(split, inset + 12.0), Vec2::new(view.x - 10.0, view.y - 12.0));
        let x0 = min.x + 14.0;
        let header = Some(Vec2::new(x0, min.y + 12.0));
        let matrix_top = min.y + 62.0;
        let mat_h = matrix_size(n as usize, n as usize, 5, 12.0).y;
        let names_row = matrix_top + mat_h + 22.0;
        let top = names_row + 18.0;
        let height = ((max.y - 8.0 - top) / n).min(96.0);
        let inner = max.x - 14.0 - x0;
        let mini = (x0 + 86.0, x0 + 86.0 + (inner - 86.0 - 70.0 - 96.0).max(90.0));
        let circle = (mini.1 + 40.0, (height * 0.3).min(26.0));
        let bar = (circle.0 + 40.0, max.x - 54.0);
        // the sum of the modes written out, under the rows, where there is room
        let below = top + n * height + 18.0;
        let sum = (max.y - below >= 120.0).then_some(Vec2::new(x0, below));
        let rows = Rows {
            min,
            max,
            header,
            matrix: Some(Vec2::new(x0, matrix_top)),
            names_y: names_row,
            top,
            height,
            button: (x0, 68.0),
            freq: x0,
            mini,
            circle,
            bar,
            sum,
            size: 12.0,
            small: 10.5,
        };
        Layout { narrow: false, heading, left, right, y, wall: 30.0, names: Some((names_y, names_y + 17.0)), chart: Some((chart, 40.0)), rows: Some(rows), size: 12.5, small: 11.0 }
    }

    fn narrow(view: Vec2, inset: f32, n: f32) -> Layout {
        let pad = 10.0;
        let heading = Some((Vec2::new(pad, inset + 8.0), 13.5, 11.5, view.x - 2.0 * pad));
        let names_y = inset + 74.0;
        let y = names_y + 42.0;
        let (left, right) = (pad + 36.0, view.x - pad - 6.0);
        // the rows at the foot, each a line; the chart between them and the masses
        let row_h = 40.0;
        let rows_h = 18.0 + n * row_h;
        let rows_min = Vec2::new(0.0, view.y - rows_h - 14.0);
        let chart_top = y + 36.0;
        let chart_bottom = rows_min.y - 26.0;
        let fits_rows = rows_min.y > y + 40.0;
        let chart = (fits_rows && chart_bottom - chart_top >= 90.0).then_some(((Vec2::new(left, chart_top), Vec2::new(right, chart_bottom)), 28.0));
        let rows = fits_rows.then(|| {
            let x0 = pad + 2.0;
            let top = rows_min.y + 22.0;
            let w = view.x - 2.0 * pad - 4.0;
            let button = (x0, 26.0);
            let freq = x0 + 32.0;
            let mini = (x0 + 98.0, x0 + 98.0 + (w - 98.0 - 34.0 - 74.0).max(60.0));
            let circle = (mini.1 + 20.0, 12.0);
            let bar = (circle.0 + 24.0, view.x - pad - 32.0);
            Rows {
                min: rows_min,
                max: Vec2::new(view.x, view.y),
                header: None,
                matrix: None,
                names_y: rows_min.y + 6.0,
                top,
                height: row_h,
                button,
                freq,
                mini,
                circle,
                bar,
                sum: None,
                size: 11.5,
                small: 10.0,
            }
        });
        Layout { narrow: true, heading, left, right, y, wall: 24.0, names: Some((names_y, names_y + 15.0)), chart, rows, size: 11.5, small: 10.5 }
    }
}

/// The parts of the picture drawn as lines, a batch for each (see `plot::apart`): one
/// whose number of points can change from frame to frame (the chart's traces) has a
/// batch of its own, so that the others can be followed point for point.
const WALLS: u8 = plot::AXES;
const COILS: u8 = plot::CURVES;
const BLOCKS: u8 = plot::MARKS;
const SHAPES: u8 = 4;
const POINTERS: u8 = 5;
const TRACES: u8 = plot::OWN;

/// The zigzags of a spring: this many turns, each two strokes.
const TURNS: usize = 7;

/// A spring from `a` to `b` (on one level): straight a little at each end, zigzag
/// between, `amp` either side of the line, always the same number of points.
fn spring_points(a: Vec2, b: Vec2, amp: f32) -> Vec<Vec2> {
    let len = b.x - a.x;
    let lead = (len * 0.14).clamp(0.0, 8.0);
    let (p, q) = (a.x + lead, b.x - lead);
    let mut pts = Vec::with_capacity(2 * TURNS + 3);
    pts.push(a);
    pts.push(Vec2::new(p, a.y));
    for c in 1..2 * TURNS {
        let side = if c % 2 == 1 { -amp } else { amp };
        pts.push(Vec2::new(p + (q - p) * c as f32 / (2 * TURNS) as f32, a.y + side));
    }
    pts.push(Vec2::new(q, a.y));
    pts.push(b);
    pts
}

impl Scene for Springs {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.25);
    }

    fn animated(&self) -> bool {
        !self.resting() || self.rows_in < 1.0 || self.placing.is_some() || self.cut.is_some()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("white dots on the masses: drag one aside, let go    a mode's button: start in that mode alone")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A row of masses joined by springs, between two walls. Pull the masses aside by $x = (x_1, \ldots, x_n)$ from where they rest, and each spring pulls back with its stiffness times its stretch; together the springs push mass $i$ with the force $-(Kx)_i$, where $K$ is the symmetric matrix on the picture. Every mass here is $1$, so without friction Newton's law says");
        ui::equation(ui, r"x'' = -Kx");
        ui::note(ui, r"Try a motion in which every mass swings in step, $x = v \cos(\omega t)$ for one fixed vector $v$. Then $x'' = -\omega^2 x$, so it is a motion of the masses exactly when");
        ui::equation(ui, r"Kv = \omega^2 v");
        ui::note(ui, r"The eigenvectors of $K$ are the pure motions, the normal modes, each swinging at its own frequency $\omega$, the square root of its eigenvalue. Every motion is a sum of them, and without friction each keeps its own amplitude for ever.");

        ui::section(ui, "Things to try");
        for (t, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(t);
            }
        }

        ui::section(ui, "The springs");
        let mut n = self.n;
        if ui::slider_int(ui, "masses $n$", &mut n, FEWEST..=MOST) && n != self.n {
            self.set_row(n, self.kappa);
            self.place(pluck());
        }
        let mut kappa = self.kappa as f32;
        if ui::slider_log(ui, r"middle spring $\kappa$", &mut kappa, KAPPA_LEAST as f32..=KAPPA_MOST as f32) {
            self.set_kappa(kappa as f64);
        }
        ui::note(ui, r"Every other spring has $k = 9$. With an odd number of masses the middle is a mass, and the two springs beside it both have $\kappa$. The masses carry on under the new springs; if that could make two of them collide, they start again from rest instead, near enough to their resting places that it cannot.");
        let mut gamma = self.gamma as f32;
        if ui::slider(ui, r"friction $\gamma$", &mut gamma, 0.0..=GAMMA_MOST as f32) {
            self.gamma = gamma as f64;
            self.still_for = 0.0;
        }
        ui::note(ui, r"A drag of $\gamma$ times each mass's velocity, the same on every mass. The modes then still keep their shapes, and each pointer turns into a spiral: it shrinks as $e^{-\gamma t/2}$ while it turns at $\sqrt{\omega^2 - \gamma^2/4}$. Friction that differed from mass to mass would in general mix the modes.");
        ui::slider_log(ui, "speed of time", &mut self.speed, 0.1..=4.0);
        ui::note(ui, "On top of the time speed under View: at 1, the scene's time runs at a second a second.");
        ui.horizontal(|ui| {
            if ui.button("let go again").clicked() {
                self.place(self.start);
            }
            if ui.button("pull the left mass").clicked() {
                self.place(pluck());
            }
        });

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::WHITE, "white: the masses (drag the dot on one aside, then let go: they start from rest; while one is held, time stands still), and under them each mass's position over time, now at the top and earlier further down"),
                (pal::DIM, "zigzags: the springs; the number over each is its stiffness"),
                (pal::ORANGE, "orange to purple: one colour for each mode, slowest first"),
                (pal::ORANGE.mix(pal::PURPLE, 0.33), "in a mode's row, the stems: its shape $v$, one for each mass, up for a swing to the right and down for one to the left; dashed, the sine curve they lie on when all the springs are the same"),
                (pal::WHITE, "the small masses under them: that mode's part of the motion alone, drawn smaller. The big masses' motion is the sum of these parts"),
                (pal::ORANGE.mix(pal::PURPLE, 0.67), "the circle: the mode's amplitude, as a pointer turning at $\\omega$; its shadow across is how far along its shape the mode is at that moment"),
                (pal::PURPLE, "the bar: the mode's energy, as a share of the whole energy at the start (when the masses were let go, or the springs last changed)"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Normal mode:", r"a motion in which all the masses swing in step at one frequency, keeping one shape: $x = v \cos(\omega t)$ with $Kv = \omega^2 v$.");
        ui::term(ui, "Stiffness matrix $K$:", r"entry $(i, j)$ is minus the force on mass $i$ when mass $j$ alone is moved by $1$. It is symmetric because a spring pulls the two masses it joins equally hard, in opposite directions.");
        ui::term(ui, r"Frequency $\omega$:", r"in radians a second: a mode goes there and back once every $2\pi/\omega$ seconds.");
        ui::term(ui, "Amplitude:", r"how far a mode swings. Let go from rest at $x_0$, the mode along the unit vector $v$ has amplitude $v \cdot x_0$: with equal masses the modes are perpendicular, so these dot products are the coordinates of $x_0$ along them.");
        ui::term(ui, "Beats:", r"two modes of close frequencies drift out of step and back, so the swing they make together grows and shrinks, once every $2\pi/(\omega_2 - \omega_1)$ seconds.");
        ui::term(ui, "Pointer:", r"a mode's coordinate $q = v \cdot x$ (how far along its shape the masses are) and its rate $q'$, drawn as the point $(q, -q'/\omega)$ (without friction). It goes round a circle at the rate $\omega$, and its shadow across is $q$.");
        ui::term(ui, "The motion as a flow:", r"the state $(x, x')$ moves with the velocity $\begin{bmatrix} 0 & I \\ -K & -\gamma I \end{bmatrix}$ times itself, so the matrix exponential carries it. Along the modes that matrix splits into one block $\begin{bmatrix} 0 & 1 \\ -\omega^2 & -\gamma \end{bmatrix}$ for each, with eigenvalues $-\gamma/2 \pm i\sqrt{\omega^2 - \gamma^2/4}$: a turn, and with friction a spiral.");

        ui::section(ui, "Numbers");
        ui::readout(ui, r"$k$, $\kappa$, $\gamma$", format!("{}, {}, {}", num(K, 0), num(self.kappa, 2), num(self.gamma, 2)));
        for m in 0..self.n {
            let w = self.spec.omega(m);
            ui::readout_colored(ui, &format!(r"mode {}: $\omega$, and seconds there and back", m + 1), format!("{}, {}", num(w, 2), num(std::f64::consts::TAU / w.max(1e-9), 2)), mode_colour(m, self.n));
        }
        if self.n == 2 {
            ui::readout(ui, r"modes back in step every $2\pi/(\omega_2 - \omega_1)$, seconds", num(beat_period(&self.spec), 2));
        }
        let e = energy(&self.drawn(), &self.k, self.n);
        ui::readout(ui, "energy: now, when let go", format!("{}, {}", num(e, 3), num(self.energy_ref, 3)));
        ui::see_also(ui, &[("spectral", "symmetric matrices: perpendicular eigenvectors"), ("expm", "the matrix exponential, a flow in the plane"), ("dft", "waves as a basis"), ("pendulum", "a double pendulum: coupled, but not linear, and chaotic")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt, f.real_dt);
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the masses move on, unseen)
        if view.x < 160.0 {
            return;
        }
        let lay = Layout::of(view, f.top_inset, self.n);
        // (every batch made in the same order in every frame, whatever is drawn in it:
        // one left out would move every later one up a place, and from one frame to the
        // next they could not be followed point for point)
        for kind in [WALLS, COILS, BLOCKS, plot::BUTTONS, SHAPES, POINTERS, TRACES, TRACES + 1] {
            plot::apart(f, kind, false);
            plot::apart(f, kind, true);
        }
        // (and the points of `plot::disc`)
        f.dl.points_with(PointKind::Overlay, PointStyle { outline_px: 0.0, ..Default::default() });
        self.hands(f, &lay);
        if let Some(rows) = &lay.rows {
            self.rows(f, rows);
        }
        let d = self.drawn();
        self.heading(f, &lay);
        self.chain(f, &lay, &d);
        if let Some((rect, pps)) = lay.chart {
            self.chart(f, &lay, rect, pps, &d);
        }
    }
}

impl Springs {
    /// Where place `i`'s mass is on the screen, across: its resting place and how far
    /// it is from it.
    fn across(&self, lay: &Layout, i: usize, x: f64) -> f32 {
        let look = &self.look.shown;
        let w = lay.right - lay.left;
        lay.left + w * (rest_of(look, i) + x * spacing_of(look)) as f32
    }

    /// Half a block's width on the screen.
    fn half_block(&self, lay: &Layout) -> f32 {
        let w = (lay.right - lay.left) * spacing_of(&self.look.shown) as f32;
        (0.5 * BLOCK as f32 * w).clamp(7.0, 17.0)
    }

    /// The masses as handles: a hand on one holds them all; moving it pulls that one
    /// aside; letting go starts them from rest.
    fn hands(&mut self, f: &mut Frame, lay: &Layout) {
        let d = self.drawn();
        let mut moved = None;
        for i in 0..self.n {
            if presence(&self.look.shown, i) < 0.5 {
                continue;
            }
            let at = Vec2::new(self.across(lay, i, d.x[i]), lay.y);
            if let Some(p) = plot::screen_handle(f, 10 + i as u32, at, pal::WHITE) {
                moved = Some((i, p));
            }
        }
        if f.handles.dragging() && self.hold != Hold::Hand {
            self.state = State { x: d.x, v: [0.0; MOST] };
            (self.hold, self.placing, self.still_for) = (Hold::Hand, None, 0.0);
        }
        if let (Some((i, p)), Hold::Hand) = (moved, self.hold) {
            let spacing = (lay.right - lay.left) as f64 * spacing_of(&self.look.shown);
            let rest = self.across(lay, i, 0.0) as f64;
            let want = ((p.x as f64 - rest) / spacing.max(1e-6)).max(-REACH).min(REACH);
            self.pull(i, want);
        }
        if self.hold == Hold::Hand && !f.handles.dragging() {
            self.let_go();
        }
    }

    /// Mass `i` pulled to `want`, or as near it as is safe.
    fn pull(&mut self, i: usize, want: f64) {
        let mut x = self.state.x;
        x[i] = want;
        if safe(&self.spec, &x) {
            self.state.x = x;
            return;
        }
        let (mut ok, mut bad) = (self.state.x[i], want);
        for _ in 0..24 {
            x[i] = 0.5 * (ok + bad);
            if safe(&self.spec, &x) {
                ok = x[i];
            } else {
                bad = x[i];
            }
        }
        self.state.x[i] = ok;
    }

    fn heading(&self, f: &mut Frame, lay: &Layout) {
        let Some((at, big, small, room)) = lay.heading else { return };
        let first = format!("{} masses, {} springs: $x'' = -Kx$", capital(WORDS[self.n]), WORDS[self.n + 1]);
        f.text_px(at, first, big, pal::TEXT, Align2::LEFT_TOP);
        // what the masses are doing, on two lines if one will not hold it
        let mut y = at.y + big + 8.0;
        for line in lines_of(f, &self.doing(), small, room).into_iter().take(2) {
            f.text_px(Vec2::new(at.x, y), line, small, pal::DIM, Align2::LEFT_TOP);
            y += small + 5.0;
        }
    }

    /// The walls, springs and masses, with their names.
    fn chain(&self, f: &mut Frame, lay: &Layout, d: &State) {
        let look = &self.look.shown;
        let (y, wall) = (lay.y, lay.wall);
        // the walls, hatched on the far side
        for (x, out) in [(lay.left, -1.0f32), (lay.right, 1.0)] {
            plot::line_of(f, WALLS, Vec2::new(x, y - wall), Vec2::new(x, y + wall), 3.0, pal::TEXT.a(0.8));
            for k in 0..6 {
                let yy = y - wall + 4.0 + k as f32 * (2.0 * wall - 8.0) / 5.0;
                plot::line_of(f, WALLS, Vec2::new(x, yy), Vec2::new(x + out * 9.0, yy - 7.0), 1.2, pal::DIM.a(0.7));
            }
        }
        let half = self.half_block(lay);
        let centre = |i: usize| self.across(lay, i, d.x[i]);
        // the springs: spring j from the left wall or mass j − 1, to mass j or the right
        // wall; a place without a mass sits at the right wall with no width
        let amp = (half * 0.6).clamp(4.0, 8.0);
        for j in 0..=MOST {
            let there = if j == 0 { 1.0 } else { presence(look, j - 1) as f32 };
            let a = if j == 0 { lay.left } else { centre(j - 1) + half * presence(look, j - 1) as f32 };
            let b = if j == MOST { lay.right } else { centre(j) - half * presence(look, j) as f32 };
            let stiff = if j <= self.n && is_middle(self.n, j) { kappa_of(look) } else { K };
            let width = (1.5 * (stiff / K).powf(0.3) as f32).clamp(0.6, 2.4);
            let pts = spring_points(Vec2::new(a, y), Vec2::new(b.max(a), y), amp);
            plot::path_of(f, COILS, &pts, width, pal::TEXT.a(0.75 * there.min(1.0)));
        }
        // the masses: a block each, its rest mark below, and an arm down to the chart
        for i in 0..MOST {
            let there = presence(look, i) as f32;
            let c = centre(i);
            let h = half * there;
            let (lo, hi) = (Vec2::new(c - h, y - h), Vec2::new(c + h, y + h));
            plot::fill(f, (lo, hi), Color::hex(0x2A3350).a(0.95 * there));
            plot::path_of(f, BLOCKS, &[lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y), lo], 1.4, pal::WHITE.a(0.85 * there));
            let rest = self.across(lay, i, 0.0);
            plot::line_of(f, BLOCKS, Vec2::new(rest, y + half + 5.0), Vec2::new(rest, y + half + 11.0), 1.2, pal::DIM.a(there));
            if let Some(((top, _), _)) = lay.chart {
                plot::line_of(f, BLOCKS, Vec2::new(c, y + h), Vec2::new(c, top.y), 1.0, pal::WHITE.a(0.22 * there));
            }
        }
        // the names of the masses, and each spring's stiffness, over their resting places
        if let Some((names, numbers)) = lay.names {
            let n = self.n;
            let w = lay.right - lay.left;
            let place = |i: f64| lay.left + w * (i / (n + 1) as f64) as f32;
            let s = springs(n, self.kappa);
            for i in 0..n {
                f.text_px(Vec2::new(place((i + 1) as f64), names), format!("$x_{}$", i + 1), lay.size + 1.0, pal::TEXT, Align2::CENTER_BOTTOM);
            }
            for (j, s) in s.iter().enumerate().take(n + 1) {
                f.text_px(Vec2::new(place(j as f64 + 0.5), numbers), stiffness_text(*s), lay.small, pal::DIM, Align2::CENTER_BOTTOM);
            }
        }
    }

    /// Each mass's position against time: now at the top, earlier further down.
    fn chart(&self, f: &mut Frame, lay: &Layout, (min, max): Rect, pps: f32, d: &State) {
        let w = lay.right - lay.left;
        let now = self.now();
        let small = lay.small;
        // the resting places, and the times down the side
        for i in 0..self.n {
            let x = lay.left + w * (i + 1) as f32 / (self.n + 1) as f32;
            plot::dashed_of(f, WALLS, Vec2::new(x, min.y), Vec2::new(x, max.y), 1.0, pal::DIM.a(0.3));
        }
        plot::line_of(f, WALLS, Vec2::new(min.x - 8.0, min.y), Vec2::new(min.x - 8.0, max.y), 1.0, pal::DIM.a(0.6));
        let span = (max.y - min.y) / pps;
        let every = if span > 12.0 { 5.0 } else { 2.0 };
        let mut s = 0.0;
        while s <= span + 1e-3 {
            let yy = min.y + s * pps;
            plot::line_of(f, WALLS, Vec2::new(min.x - 12.0, yy), Vec2::new(min.x - 8.0, yy), 1.0, pal::DIM.a(0.6));
            let text = if s == 0.0 { "now".to_string() } else { format!(r"$-{}\,\text{{s}}$", s) };
            f.text_px(Vec2::new(min.x - 14.0, yy), text, small, pal::DIM, Align2::RIGHT_CENTER);
            s += every;
        }
        let words = if lay.narrow { "each mass's position, over time" } else { "each mass's position over time: now at the top, earlier further down" };
        f.text_px(Vec2::new(min.x, max.y + 8.0), words, small, pal::DIM, Align2::LEFT_TOP);
        // the traces, from where each mass is now back through the record; the part
        // from before the number of masses last changed fading away
        let look = &self.look.shown;
        let since = self.cut.map_or(f64::MIN, |c| c.0);
        let mut pts: [Vec<Vec2>; 2] = [Vec::with_capacity(MOST * (self.history.len() + 3)), Vec::new()];
        for i in 0..MOST {
            for part in &mut pts {
                if !part.is_empty() {
                    part.push(Vec2::NAN);
                }
            }
            if presence(look, i) >= 0.5 {
                pts[0].push(Vec2::new(self.across(lay, i, d.x[i]), min.y));
            }
            for r in self.history.iter().rev() {
                let y = min.y + (now - r.t) as f32 * pps;
                let part = &mut pts[(r.t < since) as usize];
                part.push(if r.there[i] < 0.5 { Vec2::NAN } else { Vec2::new(lay.left + w * r.at[i], y) });
                if y > max.y {
                    break;
                }
            }
        }
        plot::path_of(f, TRACES, &plot::clip(&pts[0], (min, max)), 1.5, pal::WHITE.a(0.8));
        if let Some((_, fading)) = self.cut {
            plot::path_of(f, TRACES + 1, &plot::clip(&pts[1], (min, max)), 1.5, pal::WHITE.a(0.8 * fading));
        }
    }

    /// The rows of modes: for each, a button to start in it, its frequency, its shape
    /// and its part of the motion, its pointer and its energy.
    fn rows(&mut self, f: &mut Frame, r: &Rows) {
        let n = self.n;
        let fade = self.rows_in;
        let d = self.drawn();
        hud_panel(f, r.min, r.max);
        if let Some(at) = r.header {
            f.text_px(at, "The pure motions: eigenvectors $v$ of $K$", 13.0, pal::TEXT, Align2::LEFT_TOP);
            f.text_px(at + Vec2::new(0.0, 20.0), r"$Kv = \omega^2 v$; a button starts the masses in that mode alone", 11.5, pal::DIM, Align2::LEFT_TOP);
        }
        if let Some(at) = r.matrix {
            let k = &self.k;
            let rows: Vec<Vec<(String, Color)>> =
                (0..n).map(|i| (0..n).map(|j| { let v = k[i * n + j]; (format!("${}$", stiffness_text(v)), if v == 0.0 { pal::DIM } else { pal::TEXT }) }).collect()).collect();
            let size = matrix_size(n, n, 5, 12.0);
            f.text_px(Vec2::new(at.x, at.y + 0.5 * size.y), "$K =$", 13.0, pal::TEXT, Align2::LEFT_CENTER);
            hud_matrix(f, Vec2::new(at.x + 40.0, at.y), &rows, 5, 12.0, pal::DIM);
        }
        let size = r.size;
        let wide = r.header.is_some();
        if wide {
            f.text_px(Vec2::new(r.freq, r.names_y), "mode", r.small, pal::DIM, Align2::LEFT_TOP);
        }
        let long = "shape, and its part of the motion";
        let shape = if f.text_size(long, r.small).x < r.mini.1 - r.mini.0 { long } else { "shape" };
        f.text_px(Vec2::new(r.mini.0, r.names_y), shape, r.small, pal::DIM, Align2::LEFT_TOP);
        f.text_px(Vec2::new(r.circle.0, r.names_y), "amplitude", r.small, pal::DIM, Align2::CENTER_TOP);
        f.text_px(Vec2::new(0.5 * (r.bar.0 + r.bar.1) + 10.0, r.names_y), "energy", r.small, pal::DIM, Align2::CENTER_TOP);
        let pointers = self.spec.pointers(&d, self.gamma);
        let energies = self.spec.energies(&d);
        let (size_ref, energy_ref) = if self.hold == Hold::Free {
            (self.size_ref, self.energy_ref)
        } else {
            (pointers.iter().map(|p| p.2.length_squared() as f64).sum::<f64>().sqrt(), energy(&d, &self.k, n))
        };
        for m in 0..n {
            let c = mode_colour(m, n).a(fade);
            let top = r.top + m as f32 * r.height;
            let mid = top + 0.5 * r.height;
            // the button, and the frequency beside it or under it
            let (bx, bw) = r.button;
            let label = if r.header.is_some() { format!("mode {}", m + 1) } else { format!("{}", m + 1) };
            let (blo, bhi) = if r.header.is_some() { (Vec2::new(bx, top + 4.0), Vec2::new(bx + bw, top + 26.0)) } else { (Vec2::new(bx, mid - 11.0), Vec2::new(bx + bw, mid + 11.0)) };
            if plot::button(f, (blo, bhi), &label, size, false) {
                self.start_in_mode(m);
            }
            let w = self.spec.omega(m);
            let period = std::f64::consts::TAU / w.max(1e-9);
            if r.header.is_some() {
                f.text_px(Vec2::new(r.freq, top + 32.0), format!(r"$\omega = {}$", num(w, 2)), size, c, Align2::LEFT_TOP);
                f.text_px(Vec2::new(r.freq, top + 50.0), format!(r"$2\pi/\omega = {}\,\text{{s}}$", num(period, 2)), r.small, pal::DIM.a(fade), Align2::LEFT_TOP);
            } else {
                f.text_px(Vec2::new(r.freq, mid - 7.0), format!(r"$\omega = {}$", num(w, 2)), size, c, Align2::LEFT_CENTER);
                f.text_px(Vec2::new(r.freq, mid + 8.0), format!(r"${}\,\text{{s}}$", num(period, 2)), r.small, pal::DIM.a(fade), Align2::LEFT_CENTER);
            }
            // the shape, as a graph over the masses' places: a stem for each mass, up for a
            // swing to the right, down for one to the left (on a sine curve, dashed, when
            // every spring is the same); and under it a small row of masses moving with
            // this mode's part of the motion alone
            let (x0, x1) = r.mini;
            let spacing = (x1 - x0) / (n + 1) as f32;
            let graph_y = top + 0.38 * r.height;
            let reach = 0.25 * r.height;
            let line_y = top + 0.82 * r.height;
            plot::line_of(f, SHAPES, Vec2::new(x0, graph_y), Vec2::new(x1, graph_y), 1.0, pal::DIM.a(0.35 * fade));
            for x in [x0, x1] {
                plot::line_of(f, SHAPES, Vec2::new(x, line_y - 6.0), Vec2::new(x, line_y + 6.0), 1.6, pal::DIM.a(0.8 * fade));
            }
            plot::line_of(f, SHAPES, Vec2::new(x0, line_y), Vec2::new(x1, line_y), 1.0, pal::DIM.a(0.35 * fade));
            let v = &self.spec.v[m];
            let biggest = v.iter().take(n).fold(0.0f64, |b, x| b.max(x.abs())).max(1e-9);
            if self.all_equal() {
                // (mode m + 1 of equal springs is sin((m + 1)πx) across the row, scaled as
                // the stems are: its largest stem the full reach)
                let pi = std::f32::consts::PI;
                let j = (m + 1) as f32;
                let top_stem = (1..=n).map(|i| (pi * j * i as f32 / (n + 1) as f32).sin().abs()).fold(0.0f32, f32::max).max(1e-6);
                let curve: Vec<Vec3> = (0..=48)
                    .map(|k| {
                        let u = k as f32 / 48.0;
                        f.hud(Vec2::new(x0 + (x1 - x0) * u, graph_y - reach * (pi * j * u).sin() / top_stem))
                    })
                    .collect();
                plot::apart(f, SHAPES, true).polyline(curve, 1.0, c.a(0.55 * fade));
            }
            let q = pointers[m].0;
            let dot = (3.5f32).max(spacing * 0.1).min(6.0);
            for i in 0..n {
                let rest = x0 + spacing * (i + 1) as f32;
                let tip = Vec2::new(rest, graph_y - reach * (v[i] / biggest) as f32);
                plot::line_of(f, SHAPES, Vec2::new(rest, graph_y), tip, 2.0, c);
                plot::disc(f, tip, 6.0, c);
                let at = rest + (q * v[i]) as f32 * spacing;
                plot::fill(f, (Vec2::new(at - dot, line_y - dot), Vec2::new(at + dot, line_y + dot)), pal::WHITE.a(0.8 * fade));
            }
            // the pointer: on a circle as long as the whole motion's size, its shadow
            // across, and the energy's share
            let (cx, rad) = r.circle;
            let centre = Vec2::new(cx, mid);
            hud_circle(f, centre, rad, 1.0, pal::DIM.a(0.6 * fade));
            plot::line_of(f, POINTERS, centre - Vec2::new(rad, 0.0), centre + Vec2::new(rad, 0.0), 1.0, pal::DIM.a(0.3 * fade));
            let p = pointers[m].2;
            let scale = if size_ref > 1e-9 { rad / size_ref as f32 } else { 0.0 };
            let tip = centre + Vec2::new(p.x, -p.y) * scale;
            let tip = if tip.is_finite() { tip } else { centre };
            arrow(f, POINTERS, centre, tip, 2.0, c);
            plot::dashed_of(f, POINTERS, tip, Vec2::new(tip.x, mid), 1.0, c.a(0.6 * fade));
            let (b0, b1) = r.bar;
            let share = if energy_ref > 1e-12 { (energies[m] / energy_ref).clamp(0.0, 1.0) as f32 } else { 0.0 };
            plot::fill(f, (Vec2::new(b0, mid - 4.0), Vec2::new(b1, mid + 4.0)), pal::DIM.a(0.18 * fade));
            plot::fill(f, (Vec2::new(b0, mid - 4.0), Vec2::new(b0 + (b1 - b0) * share, mid + 4.0)), c.a(0.85 * fade));
            f.text_px(Vec2::new(b1 + 6.0, mid), format!("{} %", num(100.0 * share as f64, 0)), r.small, pal::TEXT.a(fade), Align2::LEFT_CENTER);
        }
        if let Some(at) = r.sum {
            let mut y = at.y;
            for (k, text) in self.sum_lines().into_iter().enumerate() {
                let (size, colour) = if k == 0 { (12.0, pal::DIM) } else { (13.0, pal::TEXT) };
                let x = at.x + if k > 1 { 16.0 } else { 0.0 };
                for line in lines_of(f, &text, size, r.max.x - 12.0 - x) {
                    let h = f.text_size(&line, size).y;
                    f.text_px(Vec2::new(x, y), line, size, colour.a(fade), Align2::LEFT_TOP);
                    y += h + 6.0;
                }
            }
        }
    }

    /// The motion as a sum of the modes, written out: what it is let go from, then the
    /// sum (with its numbers, for a few masses and no friction).
    fn sum_lines(&self) -> Vec<String> {
        let n = self.n;
        let x0 = match (self.hold, &self.placing) {
            (Hold::Placing, Some(p)) => to_array(&p.to),
            (Hold::Hand, _) => self.state.x,
            _ => self.start,
        };
        let mut lines = vec![format!("let go from rest at $x_0 = {}$, the masses move as", tuple(&x0[..n], 2))];
        if self.gamma > 0.0 {
            lines.push(r"$x(t) = \sum_m (v_m \cdot x_0)\, c_m(t)\, v_m$".to_string());
            lines.push(r"$c_m(t)$: a swing at $\sqrt{\omega_m^2 - \gamma^2/4}$ that dies away as $e^{-\gamma t/2}$".to_string());
            return lines;
        }
        if n > 3 {
            lines.push(r"$x(t) = \sum_m (v_m \cdot x_0) \cos(\omega_m t)\, v_m$".to_string());
            lines.push(r"each mode's part, $v_m \cdot x_0$ times its shape, swinging at its own rate".to_string());
            return lines;
        }
        // (the springs as asked for: the ones drawn may still be on their way there)
        let sp = spectrum(n, self.kappa);
        let a = sp.coords(&x0);
        for m in 0..n {
            let lead = match (m, a[m] < 0.0 && num(a[m], 2) != "0.00") {
                (0, false) => "x(t) = ".to_string(),
                (0, true) => "x(t) = -".to_string(),
                (_, false) => "+ ".to_string(),
                (_, true) => "- ".to_string(),
            };
            let colour = mode_colour(m, n);
            lines.push(format!(r"${lead}{} \cos({}t) {}$", tint(colour, &num(a[m].abs(), 2)), num(sp.omega(m), 2), tint(colour, &column(&sp.v[m][..n], 2))));
        }
        lines
    }
}

/// An arrow on the screen from `a` to `b` in a batch of its kind, always of the same
/// number of points (one of no length is drawn as nothing at all).
fn arrow(f: &mut Frame, kind: u8, a: Vec2, b: Vec2, width: f32, c: Color) {
    let d = b - a;
    let len = d.length();
    let dir = if len > 1e-6 { d / len } else { Vec2::ZERO };
    let head = (3.0 + 2.0 * width).min(0.6 * len);
    let side = dir.perp() * head * 0.55;
    let base = b - dir * head;
    plot::path_of(f, kind, &[a, b, base + side, b, base - side], width, c);
}

/// A text broken into lines no wider than `room` pixels, at spaces outside its formulas.
fn lines_of(f: &Frame, text: &str, size: f32, room: f32) -> Vec<String> {
    let mut words: Vec<&str> = Vec::new();
    let (mut start, mut inside) = (0, false);
    for (i, c) in text.char_indices() {
        match c {
            '$' => inside = !inside,
            ' ' if !inside => {
                words.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    words.push(&text[start..]);
    let mut lines: Vec<String> = Vec::new();
    for word in words {
        match lines.last_mut() {
            Some(line) if f.text_size(&format!("{line} {word}"), size).x <= room => {
                line.push(' ');
                line.push_str(word);
            }
            _ => lines.push(word.to_string()),
        }
    }
    lines
}

fn capital(word: &str) -> String {
    let mut c = word.chars();
    c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    const TAU64: f64 = std::f64::consts::TAU;

    /// The scene as an experiment's button leaves it, at `hz` frames a second, once the
    /// masses have glided to its start and been let go.
    fn set_up(t: usize, hz: f64) -> Springs {
        let mut scene = Springs::new();
        scene.set_up(t);
        settle(&mut scene, hz);
        scene
    }

    fn settle(scene: &mut Springs, hz: f64) {
        for _ in 0..(hz * 5.0) as usize {
            if scene.hold == Hold::Free {
                return;
            }
            scene.advance((1.0 / hz) as f32, (1.0 / hz) as f32);
        }
        panic!("never let go");
    }

    /// Run on for `seconds` at `hz`, calling `each` with the scene after every frame.
    fn run(scene: &mut Springs, seconds: f64, hz: f64, mut each: impl FnMut(&Springs)) {
        for _ in 0..(seconds * hz).round() as usize {
            scene.advance((1.0 / hz) as f32, (1.0 / hz) as f32);
            each(scene);
        }
    }

    /// The time since the masses were let go.
    fn since(scene: &Springs) -> f64 {
        scene.now() - scene.let_go_at
    }

    fn springs_tried() -> Vec<(usize, f64)> {
        let mut out = Vec::new();
        for n in FEWEST..=MOST {
            for i in 0..=40 {
                let kappa = KAPPA_LEAST * (KAPPA_MOST / KAPPA_LEAST).powf(i as f64 / 40.0);
                out.push((n, kappa));
            }
            out.push((n, K));
        }
        out
    }

    /// K is the springs: symmetric, each row's numbers adding up to the stiffness of the
    /// wall springs at that mass (none, inside the row), and two masses' matrix as it is
    /// written in the books.
    #[test]
    fn k_is_made_of_the_springs() {
        for (n, kappa) in springs_tried() {
            let k = stiffness(n, kappa);
            let s = springs(n, kappa);
            for i in 0..n {
                let row: f64 = (0..n).map(|j| k[i * n + j]).sum();
                let walls = if i == 0 { s[0] } else { 0.0 } + if i == n - 1 { s[n] } else { 0.0 };
                assert!((row - walls).abs() < 1e-12, "{n} {kappa}: row {i}");
                for j in 0..n {
                    assert_eq!(k[i * n + j], k[j * n + i]);
                }
            }
            // the middle spring is the one in the middle
            let middles: Vec<usize> = (0..=n).filter(|&j| is_middle(n, j)).collect();
            assert!(middles.iter().all(|&j| j + j == n || j + j + 1 == n || j + j == n + 1) && !middles.is_empty(), "{n}: {middles:?}");
            assert_eq!(spectrum_kappa(&k, n), kappa);
        }
        assert_eq!(stiffness(2, 1.5), vec![10.5, -1.5, -1.5, 10.5]);
        assert_eq!(stiffness(2, K), vec![18.0, -9.0, -9.0, 18.0]);
    }

    /// The modes are unit eigenvectors of K, perpendicular, slowest first, each with its
    /// first number positive; no two frequencies are ever the same, and the slowest
    /// is never so slow that the most friction stops it swinging.
    #[test]
    fn the_modes_are_the_eigenvectors_of_k() {
        let mut slowest = f64::MAX;
        for (n, kappa) in springs_tried() {
            let k = stiffness(n, kappa);
            let sp = spectrum(n, kappa);
            for m in 0..n {
                let v = &sp.v[m];
                for i in 0..n {
                    let kv: f64 = (0..n).map(|j| k[i * n + j] * v[j]).sum();
                    assert!((kv - sp.w2[m] * v[i]).abs() < 1e-10, "{n} {kappa}: mode {m}");
                }
                for l in 0..n {
                    let d: f64 = (0..n).map(|i| v[i] * sp.v[l][i]).sum();
                    assert!((d - if l == m { 1.0 } else { 0.0 }).abs() < 1e-12);
                }
                assert!(v[0] > 1e-3, "{n} {kappa}: mode {m} starts at {}", v[0]);
                if m > 0 {
                    assert!(sp.w2[m] > sp.w2[m - 1] + 1e-3, "{n} {kappa}: modes {m} and {} close: {:?}", m - 1, sp.w2);
                }
            }
            slowest = slowest.min(sp.omega(0));
        }
        // (and the most friction, which slows a swing to √(ω² − γ²/4), leaves it swinging)
        assert!(slowest > 0.68 && 0.5 * GAMMA_MOST < 0.68, "{slowest}");
    }

    /// With every spring the same, mode j's shape is the sine curve sin(ijπ/(n + 1)),
    /// and its frequency 2√k sin(jπ/(2(n + 1))): the five masses' caption.
    #[test]
    fn equal_springs_swing_in_sine_shapes() {
        for n in FEWEST..=MOST {
            let sp = spectrum(n, K);
            let pi = std::f64::consts::PI;
            for m in 0..n {
                let j = (m + 1) as f64;
                let scale = (2.0 / (n + 1) as f64).sqrt();
                for i in 0..n {
                    assert!((sp.v[m][i] - scale * ((i + 1) as f64 * j * pi / (n + 1) as f64).sin()).abs() < 1e-10, "{n}: mode {m}");
                }
                assert!((sp.omega(m) - 2.0 * K.sqrt() * (j * pi / (2.0 * (n + 1) as f64)).sin()).abs() < 1e-10);
            }
        }
        // the caption's numbers: sin(30 j), ..., sin(150 j), 6 sin(15 j), and j - 1 turns
        let scene = set_up(4, 60.0);
        let sp = &scene.spec;
        assert!(scene.n == 5 && scene.kappa == K);
        let freqs: Vec<String> = (0..5).map(|m| num(sp.omega(m), 2)).collect();
        assert_eq!(freqs, ["1.55", "3.00", "4.24", "5.20", "5.80"]);
        for m in 0..5 {
            let j = (m + 1) as f64;
            assert!((sp.omega(m) - 6.0 * (15.0 * j).to_radians().sin()).abs() < 1e-12);
            let biggest = sp.v[m].iter().fold(0.0f64, |b, x| b.max(x.abs()));
            for i in 0..5 {
                let want = (30.0 * (i + 1) as f64 * j).to_radians().sin();
                assert!((sp.v[m][i] / biggest - want / (0..5).map(|i| (30.0 * (i + 1) as f64 * j).to_radians().sin().abs()).fold(0.0, f64::max)).abs() < 1e-9);
            }
            // turns: sign changes along the row, skipping masses that stay still
            let signs: Vec<f64> = sp.v[m].iter().take(5).filter(|x| x.abs() > 1e-9).map(|x| x.signum()).collect();
            let turns = signs.windows(2).filter(|w| w[0] != w[1]).count();
            assert_eq!(turns, m, "mode {}", m + 1);
        }
        // the slowest is all one way
        assert!(sp.v[0].iter().take(5).all(|x| *x > 0.0));
    }

    /// The masses, stepped as masses on springs, move as the modes say they must: each
    /// mode's start along it, aₘ = vₘ · x₀, times cos(ωₘt) (with friction, a dying
    /// swing at √(ω² − γ²/4)), to the accuracy of the stepping; at any frame rate.
    #[test]
    fn the_masses_follow_the_sum_of_the_modes() {
        let mut worst = 0.0f64;
        for (t, gamma, hz) in [(2, 0.0, 60.0), (3, 0.0, 144.0), (4, 0.0, 30.0), (2, 0.4, 60.0), (4, GAMMA_MOST, 75.0)] {
            let mut scene = set_up(t, hz);
            scene.gamma = gamma;
            scene.measure();
            let size = scene.start.iter().map(|x| x * x).sum::<f64>().sqrt();
            run(&mut scene, 60.0, hz, |s| {
                let want = closed_form(&s.spec, &s.start, s.gamma, since(s));
                let got = s.drawn().x;
                let off = (0..MOST).map(|i| (got[i] - want[i]).abs()).fold(0.0, f64::max) / size;
                worst = worst.max(off);
            });
        }
        assert!(worst < 3e-4, "the masses strayed {worst} of the start's size from the sum of the modes");
    }

    /// Without friction the energy stays what it was, within a part in a hundred
    /// thousand, over a thousand seconds, and does not drift; with friction it never
    /// rises above where it started, and falls as e^(−γt).
    #[test]
    fn energy_is_kept_and_friction_only_takes_it() {
        for n in [2, 6] {
            let mut scene = Springs::new();
            scene.set_row(n, K);
            scene.place(pluck());
            settle(&mut scene, 60.0);
            let e0 = energy(&scene.state, &scene.k, n);
            let (mut most, mut early, mut late) = (0.0f64, 0.0, 0.0);
            for k in 0..1_000_000u32 {
                step(&mut scene.state, &scene.k, n, 0.0, STEP);
                let e = energy(&scene.state, &scene.k, n);
                most = most.max((e - e0).abs() / e0);
                if k < 100_000 {
                    early += e / 1e5;
                } else if k >= 900_000 {
                    late += e / 1e5;
                }
            }
            assert!(most < 1e-5, "{n} masses: energy off by {most}");
            assert!(((late - early) / e0).abs() < 1e-8, "{n} masses: a drift of {}", (late - early) / e0);
        }
        for gamma in [0.2, GAMMA_MOST] {
            let mut scene = set_up(4, 60.0);
            scene.gamma = gamma;
            let e0 = energy(&scene.state, &scene.k, scene.n);
            let (mut low, mut high) = (f64::MAX, 0.0f64);
            for k in 1..=20_000u32 {
                step(&mut scene.state, &scene.k, scene.n, gamma, STEP);
                let e = energy(&scene.state, &scene.k, scene.n);
                assert!(e <= e0 * (1.0 + 1e-6), "{gamma}: rose to {}", e / e0);
                let ratio = e / (e0 * (-gamma * k as f64 * STEP).exp());
                (low, high) = (low.min(ratio), high.max(ratio));
            }
            // (each mode's energy swings between moving and stretching as it falls, by about
            // γ/(2ω) of itself either way)
            assert!(1.0 - low < 0.13 * gamma && high - 1.0 < 0.13 * gamma, "{gamma}: {low} to {high} of e^(-γt)");
        }
    }

    /// Each mode's pointer goes round at its own rate and keeps its length without
    /// friction; with friction it shrinks as e^(−γt/2), turning at √(ω² − γ²/4).
    #[test]
    fn pointers_turn_at_their_rates() {
        for gamma in [0.0, 0.6] {
            let mut scene = set_up(4, 60.0);
            scene.gamma = gamma;
            let (first, from) = (scene.spec.pointers(&scene.drawn(), gamma), since(&scene));
            run(&mut scene, 10.0, 60.0, |_| {});
            let t = since(&scene) - from;
            let now = scene.spec.pointers(&scene.drawn(), gamma);
            for m in 0..scene.n {
                let (a, b) = (first[m].2.as_dvec2(), now[m].2.as_dvec2());
                assert!((b.length() / a.length() - (-0.5 * gamma * t).exp()).abs() < 1e-4, "{gamma}: mode {m}");
                let turned = a.angle_to(b);
                let want = (scene.spec.damped(m, gamma) * t + std::f64::consts::PI).rem_euclid(TAU64) - std::f64::consts::PI;
                assert!((turned - want).abs() < 2e-3, "{gamma}: mode {m} turned {turned}, not {want}");
            }
        }
    }

    /// "together": mode 1 alone at 3.00, every 2.09 s; the middle spring keeps its
    /// length; ω² = k; and a change of the middle spring leaves the motion as it was.
    #[test]
    fn together_swings_as_its_caption_says() {
        let mut scene = set_up(0, 60.0);
        let sp = scene.spec.clone();
        assert_eq!((num(sp.omega(0), 2).as_str(), num(TAU64 / sp.omega(0), 2).as_str()), ("3.00", "2.09"));
        assert!((sp.w2[0] - K).abs() < 1e-12);
        let a = sp.coords(&scene.start);
        assert!(a[1].abs() < 1e-15 && a[0] > 0.3 && (sp.v[0][0] - sp.v[0][1]).abs() < 1e-15);
        assert_eq!(scene.doing(), "mode 1 alone: the masses swing together");
        run(&mut scene, 20.0, 60.0, |s| {
            let x = s.drawn().x;
            assert!((x[0] - x[1]).abs() < 1e-12);
            assert!(s.spec.pointers(&s.drawn(), 0.0)[1].2.length() < 1e-9);
        });
        // the middle spring changed by hand, in mid-swing: the motion goes on as it was
        run(&mut scene, 0.4, 60.0, |_| {});
        let before = scene.drawn();
        scene.set_kappa(1.5);
        assert_eq!(scene.hold, Hold::Free);
        assert_eq!(scene.drawn().x, before.x);
        run(&mut scene, 10.0, 60.0, |s| {
            let want = closed_form(&sp, &s.start, 0.0, since(s));
            assert!((s.drawn().x[0] - want[0]).abs() < 1e-4 && (s.drawn().x[0] - s.drawn().x[1]).abs() < 1e-12);
        });
    }

    /// "against each other": mode 2 alone at √27 = 5.20, every 1.21 s; the middle spring
    /// stretched by twice what either mass moves; ω² = k + 2k.
    #[test]
    fn against_each_other_swings_as_its_caption_says() {
        let mut scene = set_up(1, 60.0);
        let sp = scene.spec.clone();
        assert_eq!((num(sp.omega(1), 2).as_str(), num(TAU64 / sp.omega(1), 2).as_str()), ("5.20", "1.21"));
        assert!((sp.w2[1] - 3.0 * K).abs() < 1e-12 && (27f64.sqrt() - sp.omega(1)).abs() < 1e-12);
        assert!(scene.start[0] > 0.0 && scene.start[1] == -scene.start[0], "moved towards each other");
        assert_eq!(scene.doing(), "mode 2 alone: the masses swing against each other");
        run(&mut scene, 20.0, 60.0, |s| {
            let x = s.drawn().x;
            assert!((x[0] + x[1]).abs() < 1e-12);
            assert!(s.spec.pointers(&s.drawn(), 0.0)[0].2.length() < 1e-9);
        });
    }

    /// "pluck one mass": equal amplitudes, kept; 25 % and 75 % of the energy;
    /// ω₂/ω₁ = √3.
    #[test]
    fn a_plucked_mass_swings_as_its_caption_says() {
        let mut scene = set_up(2, 60.0);
        let sp = scene.spec.clone();
        assert!(scene.start[1] == 0.0 && scene.start[0] > 0.0);
        let a = sp.coords(&scene.start);
        assert!((a[0] - a[1]).abs() < 1e-15);
        assert!((sp.omega(1) / sp.omega(0) - 3f64.sqrt()).abs() < 1e-12);
        let e = sp.energies(&scene.drawn());
        let shares: Vec<String> = (0..2).map(|m| num(100.0 * e[m] / (e[0] + e[1]), 0)).collect();
        assert_eq!(shares, ["25", "75"]);
        assert_eq!(scene.doing(), "both modes at once, each at its own frequency");
        let size = scene.size_ref;
        run(&mut scene, 30.0, 60.0, |s| {
            let p = s.spec.pointers(&s.drawn(), 0.0);
            assert!((p[0].2.length() - p[1].2.length()).abs() < 1e-5 * size as f32);
            let e = s.spec.energies(&s.drawn());
            assert!((e[0] / s.energy_ref - 0.25).abs() < 1e-5 && (e[1] / s.energy_ref - 0.75).abs() < 1e-5);
        });
    }

    /// "a weak middle spring: beats": 3.00 and √12 = 3.46; the left mass's swing at its
    /// smallest at 6.77 s, when the right one's is as large as the start, and the other
    /// way round at 13.54 s; 2π/(ω₂ − ω₁).
    #[test]
    fn a_weak_middle_spring_beats_as_its_caption_says() {
        let mut scene = set_up(3, 120.0);
        let sp = scene.spec.clone();
        assert!((scene.kappa - 1.5).abs() < 1e-12 && (scene.kappa - K / 6.0).abs() < 1e-12);
        assert_eq!((num(sp.omega(0), 2).as_str(), num(sp.omega(1), 2).as_str()), ("3.00", "3.46"));
        assert!((sp.omega(1) - 12f64.sqrt()).abs() < 1e-9);
        let beat = beat_period(&sp);
        assert_eq!((num(beat, 2).as_str(), num(beat / 2.0, 2).as_str()), ("13.54", "6.77"));
        assert_eq!(scene.doing(), "modes 1 and 2, close in frequency: the swing passes to the other mass and back, every 13.54 seconds");
        // the largest swing of each mass over half a second either side of a time
        let mut seen: Vec<(f64, [f64; 2])> = Vec::new();
        run(&mut scene, 15.0, 120.0, |s| seen.push((since(s), [s.drawn().x[0], s.drawn().x[1]])));
        let swing = |at: f64, i: usize| seen.iter().filter(|(t, _)| (t - at).abs() <= 0.5).fold(0.0f64, |m, (_, x)| m.max(x[i].abs()));
        let d = scene.start[0];
        // (half a second off, the dying swing is sin(0.46 / 4) = 0.116 of the start: within
        // the second round each time it is less than that, and nothing in the middle)
        assert!(swing(6.77, 0) < 0.12 * d && swing(6.77, 1) > 0.99 * d, "{} {}", swing(6.77, 0), swing(6.77, 1));
        assert!(swing(13.54, 1) < 0.12 * d && swing(13.54, 0) > 0.99 * d, "{} {}", swing(13.54, 0), swing(13.54, 1));
        let middle = |at: f64, i: usize| seen.iter().filter(|(t, _)| (t - at).abs() <= 0.02).fold(0.0f64, |m, (_, x)| m.max(x[i].abs()));
        assert!(middle(6.77, 0) < 0.01 * d && middle(13.54, 1) < 0.01 * d, "{} {}", middle(6.77, 0), middle(13.54, 1));
        // and the left mass's swing is never smaller than there
        let least = (1..150).map(|k| k as f64 * 0.1).filter(|t| *t < 13.0).map(|t| swing(t, 0)).fold(f64::MAX, f64::min);
        assert!((least - swing(6.77, 0)).abs() < 0.02 * d, "{least}");
        // from a start mostly along one mode, the swing does not pass over: the heading
        // says only that the swings grow and shrink
        scene.place([0.4, 0.2, 0.0, 0.0, 0.0, 0.0]);
        settle(&mut scene, 60.0);
        assert!(scene.doing().contains("each mass's swing grows and shrinks"), "{}", scene.doing());
    }

    /// Each experiment's start is safe as it is (not scaled down), and the row's
    /// heading says what is moving.
    #[test]
    fn every_experiment_starts_where_it_says() {
        for (t, trial) in TRIALS.iter().enumerate() {
            let scene = set_up(t, 60.0);
            assert_eq!(scene.start, trial.start, "{}", trial.label);
            assert_eq!(scene.n, trial.n);
            assert!(scene.gamma == 0.0 && scene.speed == 1.0);
        }
    }

    /// No motion ever brings two blocks, or a block and a wall, nearer than the
    /// shortest spring allowed: from random starts as safe as a hand may set them, with
    /// and without friction.
    #[test]
    fn the_blocks_never_meet() {
        let mut rng = Rng::new(7);
        for trial in 0..60 {
            let n = FEWEST + trial % (MOST - FEWEST + 1);
            let kappa = KAPPA_LEAST * (KAPPA_MOST / KAPPA_LEAST).powf(rng.f32() as f64);
            let sp = spectrum(n, kappa);
            let k = stiffness(n, kappa);
            let x = made_safe(&sp, std::array::from_fn(|i| if i < n { rng.range(-REACH as f32, REACH as f32) as f64 } else { 0.0 }));
            let mut s = State { x, v: [0.0; MOST] };
            let gamma = if trial % 3 == 0 { 0.3 } else { 0.0 };
            for _ in 0..30_000 {
                step(&mut s, &k, n, gamma, STEP);
                for j in 0..=n {
                    let (a, b) = (if j == 0 { 0.0 } else { s.x[j - 1] }, if j == n { 0.0 } else { s.x[j] });
                    let room = if j == 0 || j == n { 1.0 - 0.5 * BLOCK } else { 1.0 - BLOCK };
                    assert!((b - a).abs() <= room - SHORTEST + 1e-6, "{n} {kappa}: spring {j}");
                }
            }
        }
    }

    /// A pure mode started by its button stays that mode.
    #[test]
    fn a_modes_button_starts_it_alone() {
        for n in FEWEST..=MOST {
            for m in 0..n {
                let mut scene = Springs::new();
                scene.set_row(n, 3.0);
                settle(&mut scene, 60.0);
                scene.start_in_mode(m);
                settle(&mut scene, 60.0);
                let a = scene.spec.coords(&scene.start);
                assert!((0..n).all(|l| l == m || a[l].abs() < 1e-12) && a[m] > 0.1, "{n}: mode {m}: {a:?}");
                assert!(scene.doing().starts_with(&format!("mode {} alone", m + 1)), "{}", scene.doing());
                assert!(scene.start.iter().all(|x| x.abs() <= 0.35 + 1e-12));
            }
        }
    }

    /// Friction left on: the masses come exactly to rest, the chart empties, and then
    /// the scene stops changing.
    #[test]
    fn friction_brings_everything_to_rest() {
        let mut scene = set_up(2, 60.0);
        scene.gamma = GAMMA_MOST;
        run(&mut scene, 200.0, 30.0, |_| {});
        assert!(!scene.animated() && scene.state == State::default());
        let (steps, records) = (scene.steps, scene.history.len());
        run(&mut scene, 5.0, 60.0, |_| {});
        assert!(scene.steps == steps && scene.history.len() == records);
        // a hand on a mass wakes it
        scene.place(pluck());
        assert!(scene.animated());
    }

    /// Whatever is asked for, everything drawn or read out is finite: the ends of every
    /// slider, frames of every length, no motion at all.
    #[test]
    fn the_ends_of_everything_are_finite() {
        let mut scene = Springs::new();
        for (k, (n, kappa, gamma)) in [(2, KAPPA_LEAST, 0.0), (6, KAPPA_MOST, GAMMA_MOST), (3, KAPPA_LEAST, GAMMA_MOST), (5, f64::NAN, 0.0), (6, 1e9, 0.0), (2, -3.0, 0.5)].into_iter().enumerate() {
            scene.set_row(n, kappa);
            scene.gamma = gamma;
            scene.place(if k % 2 == 0 { [REACH; MOST] } else { [0.0; MOST] });
            for dt in [1.0 / 60.0, 0.0, 0.27, f32::NAN, 1.0 / 240.0, 4.0 / 15.0 * 4.0, -1.0] {
                for _ in 0..20 {
                    scene.advance(dt, dt);
                    let d = scene.drawn();
                    assert!(d.x.iter().chain(&d.v).all(|v| v.is_finite()));
                    assert!(scene.spec.pointers(&d, scene.gamma).iter().all(|p| p.2.is_finite()));
                    assert!(scene.size_ref.is_finite() && scene.energy_ref.is_finite());
                    assert!(!scene.doing().contains("NaN") && !scene.doing().contains('?'));
                    assert!(scene.history.iter().all(|r| r.at.iter().all(|v| v.is_finite())));
                }
            }
        }
    }

    /// The layout fits every window, from a phone to a desktop to one far too small.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h) in [(1080.0, 1000.0), (1600.0, 1000.0), (390.0, 800.0), (390.0, 560.0), (780.0, 460.0), (620.0, 480.0), (800.0, 600.0), (240.0, 180.0), (320.0, 400.0)] {
            for inset in [0.0, 46.0] {
                for n in FEWEST..=MOST {
                    let view = Vec2::new(w, h);
                    let lay = Layout::of(view, inset, n);
                    let inside = |p: Vec2| p.x >= -0.5 && p.y >= -0.5 && p.x <= w + 0.5 && p.y <= h + 0.5;
                    assert!(lay.left < lay.right && inside(Vec2::new(lay.left, lay.y)) && inside(Vec2::new(lay.right, lay.y + lay.wall)), "{w}x{h}");
                    if let Some(((min, max), _)) = lay.chart {
                        assert!(inside(min) && inside(max) && max.y - min.y >= 90.0 && min.y > lay.y, "{w}x{h}: chart");
                    }
                    if let Some(r) = &lay.rows {
                        assert!(inside(r.min) && inside(r.max) && r.top + n as f32 * r.height <= h + 0.5, "{w}x{h} {n}: rows");
                        assert!(r.mini.0 < r.mini.1 && r.mini.1 < r.circle.0 - r.circle.1 && r.circle.0 + r.circle.1 < r.bar.0 && r.bar.0 < r.bar.1, "{w}x{h}: {r:?}");
                    }
                }
            }
        }
    }
}


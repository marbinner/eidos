//! The Fourier basis. A signal of N samples is a vector in N dimensions; the waves
//! that fit a whole number of times into the N samples are N directions at right
//! angles to one another, and the signal's coordinates along them, its dot products
//! with them, are its discrete Fourier transform. An operation that treats every
//! position alike (a blur) does nothing in that basis but multiply each wave by a number.

use super::*;
use std::f64::consts::PI as PI64;

pub const DEF: SceneDef = SceneDef {
    id: "dft",
    title: "The Fourier basis",
    group: "Linear algebra: data and dimensions",
    blurb: "A signal as a vector, and waves as perpendicular axes to measure it along.",
    make: |_| Box::new(Dft::new()),
};

/// The sizes the signal can have.
const N_MIN: usize = 8;
const N_MAX: usize = 64;
/// The size the scene opens with, and every experiment's.
const N_START: usize = 32;
/// The samples stay within ±AMP (a dragged one stops there).
const AMP: f64 = 2.0;
/// The widths a blur can average over (1: no blur).
const WIDTHS: [usize; 5] = [1, 3, 5, 9, 15];
/// How long a glide takes: each of its two eases covers about two thirds of what is
/// left of the way in this many seconds.
const GLIDE: f64 = 0.07;

// The colours: one thing each.
const SIGNAL: Color = pal::WHITE;
const COS: Color = pal::CYAN;
const SIN: Color = pal::PINK;
const UP: Color = pal::YELLOW;
const DOWN: Color = pal::PURPLE;
const REBUILT: Color = pal::GREEN;
const BLUR: Color = pal::ORANGE;
const SIZE: Color = pal::DIM;
const HEAT: Cmap = Cmap::IceFire;

// ----- the waves ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// 1, 1, 1, …
    Constant,
    Cos,
    Sin,
    /// +1, −1, +1, …: for even N, the cosine of frequency N/2 (whose sine is 0 at every sample).
    Alternating,
}

/// One wave of the basis: its frequency (how many times it fits into the N samples)
/// and its kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Wave {
    k: usize,
    kind: Kind,
}

impl Wave {
    /// Wave j of the basis for n samples. They come in order of frequency: the
    /// constant; then the cosine and the sine of each frequency from 1 up; and, when n
    /// is even, the alternating wave last.
    fn of(j: usize, n: usize) -> Wave {
        if j == 0 {
            Wave { k: 0, kind: Kind::Constant }
        } else if j % 2 == 1 {
            let k = j.div_ceil(2);
            Wave { k, kind: if 2 * k == n { Kind::Alternating } else { Kind::Cos } }
        } else {
            Wave { k: j / 2, kind: Kind::Sin }
        }
    }

    /// Whether the basis for n samples has this wave.
    fn fits(self, n: usize) -> bool {
        match self.kind {
            Kind::Constant => true,
            Kind::Cos | Kind::Sin => self.k >= 1 && 2 * self.k < n,
            Kind::Alternating => n.is_multiple_of(2),
        }
    }

    /// Its place in the basis for n samples.
    fn index(self, n: usize) -> usize {
        match self.kind {
            Kind::Constant => 0,
            Kind::Cos => (2 * self.k).saturating_sub(1),
            Kind::Sin => 2 * self.k,
            Kind::Alternating => n.saturating_sub(1),
        }
    }

    /// Its value where its angle is `phase` (radians): scaled to length 1 over the n samples.
    fn value(self, n: usize, phase: f64) -> f64 {
        let n = n.max(1) as f64;
        match self.kind {
            Kind::Constant => 1.0 / n.sqrt(),
            Kind::Cos => (2.0 / n).sqrt() * phase.cos(),
            Kind::Sin => (2.0 / n).sqrt() * phase.sin(),
            Kind::Alternating => phase.cos() / n.sqrt(),
        }
    }

    /// Its value at position t: a sample number, or anywhere between (the smooth curve
    /// through its samples).
    fn at(self, n: usize, t: f64) -> f64 {
        self.value(n, 2.0 * PI64 * self.k as f64 * t / n.max(1) as f64)
    }

    /// Its value at sample i, the angle reduced to less than a turn first, so that it is
    /// as exact as the arithmetic allows.
    fn sample(self, n: usize, i: usize) -> f64 {
        let n1 = n.max(1);
        self.value(n, 2.0 * PI64 * ((self.k * i) % n1) as f64 / n1 as f64)
    }

    /// Its name as a formula, to go between dollars: `\cos 3`, `\text{the constant}`.
    fn symbol(self) -> String {
        match self.kind {
            Kind::Constant => r"\text{the constant}".to_string(),
            Kind::Cos => format!(r"\cos {}", self.k),
            Kind::Sin => format!(r"\sin {}", self.k),
            Kind::Alternating => r"\text{the alternating wave}".to_string(),
        }
    }

    /// Its name in a sentence: "$\cos 3$", "the constant".
    fn name(self) -> String {
        match self.kind {
            Kind::Constant => "the constant".to_string(),
            Kind::Cos | Kind::Sin => format!("${}$", self.symbol()),
            Kind::Alternating => "the alternating wave".to_string(),
        }
    }

    /// What it is called in running text: "the wave $\cos 3$", "the constant".
    fn called(self) -> String {
        match self.kind {
            Kind::Cos | Kind::Sin => format!("the wave {}", self.name()),
            _ => self.name(),
        }
    }

    /// A short name, for a label beside its bar: inside a formula, like `symbol`.
    fn short(self) -> String {
        match self.kind {
            Kind::Constant => r"\text{constant}".to_string(),
            Kind::Alternating => r"\text{alternating}".to_string(),
            _ => self.symbol(),
        }
    }

    fn color(self) -> Color {
        if self.kind == Kind::Sin {
            SIN
        } else {
            COS
        }
    }
}

/// How many frequencies the basis for n samples has: 0 to n/2, rounded down.
fn frequencies(n: usize) -> usize {
    n / 2 + 1
}

/// The waves of frequency k, as places in the basis: the constant alone, a cosine and
/// a sine, or the alternating wave alone.
fn members(k: usize, n: usize) -> std::ops::Range<usize> {
    if k == 0 {
        0..1
    } else if 2 * k == n {
        n - 1..n
    } else {
        (2 * k - 1).min(n)..(2 * k + 1).min(n)
    }
}

/// The waves for n samples as the rows of an n × n matrix W. Its rows are at right
/// angles to one another and of length 1, so W x is x's coordinates and Wᵀ takes
/// coordinates back to samples.
struct Basis {
    n: usize,
    w: Vec<f64>,
}

impl Basis {
    fn new(n: usize) -> Basis {
        let w = (0..n * n).map(|e| Wave::of(e / n, n).sample(n, e % n)).collect();
        Basis { n, w }
    }

    fn row(&self, j: usize) -> &[f64] {
        let n = self.n;
        self.w.get(j * n..(j + 1) * n).unwrap_or(&[])
    }

    /// The coordinates of x: its dot product with each wave.
    fn forward(&self, x: &[f64]) -> Vec<f64> {
        (0..self.n).map(|j| self.row(j).iter().zip(x).map(|(w, x)| w * x).sum()).collect()
    }

    /// The signal with coordinates c: each wave times its coordinate, added up.
    fn back(&self, c: &[f64]) -> Vec<f64> {
        let mut x = vec![0.0; self.n];
        for (j, cj) in c.iter().enumerate().take(self.n) {
            if *cj != 0.0 {
                for (xi, w) in x.iter_mut().zip(self.row(j)) {
                    *xi += cj * w;
                }
            }
        }
        x
    }

    /// The matrix A as it is in the waves' basis: W A Wᵀ (A n × n, row after row).
    fn in_waves(&self, a: &[f64]) -> Vec<f64> {
        let n = self.n;
        // W A first, then (W A) Wᵀ
        let mut wa = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                wa[j * n + i] = (0..n).map(|m| self.w[j * n + m] * a[m * n + i]).sum();
            }
        }
        let mut out = vec![0.0; n * n];
        for j in 0..n {
            for l in 0..n {
                out[j * n + l] = (0..n).map(|i| wa[j * n + i] * self.w[l * n + i]).sum();
            }
        }
        out
    }
}

/// The dot products of every wave with every wave, W Wᵀ (n × n, row after row).
fn gram(b: &Basis) -> Vec<f64> {
    let n = b.n;
    (0..n * n).map(|e| b.row(e / n).iter().zip(b.row(e % n)).map(|(p, q)| p * q).sum()).collect()
}

/// The coordinates as they were `s` shifts of one sample ago (with s between whole
/// numbers, part of the way): a shift turns each frequency's cosine and sine
/// coordinates together through the angle 2πk/N, like the hand of a clock, and so
/// keeps their combined size. The constant is left as it is, and the alternating wave,
/// with no sine to turn into, can only shrink and flip.
fn turned(c: &[f64], n: usize, s: f64) -> Vec<f64> {
    let mut out = c.to_vec();
    if s == 0.0 || c.len() != n {
        return out;
    }
    for k in 1..frequencies(n) {
        let r = members(k, n);
        let angle = -2.0 * PI64 * k as f64 * s / n as f64;
        let (sin, cos) = angle.sin_cos();
        if r.len() == 2 {
            let (a, b) = (c[r.start], c[r.start + 1]);
            out[r.start] = a * cos - b * sin;
            out[r.start + 1] = a * sin + b * cos;
        } else if r.len() == 1 {
            out[r.start] = c[r.start] * cos;
        }
    }
    out
}

/// How many samples either side a blur of this width averages over, for n samples
/// (never more than goes round once).
fn reach(n: usize, width: usize) -> usize {
    ((width.max(1) - 1) / 2).min(n.saturating_sub(1) / 2)
}

/// The number a blur multiplies frequency k by: averaging over 2r + 1 neighbours,
/// (1 + 2 cos(2πk/N) + 2 cos(2π·2k/N) + … + 2 cos(2π·rk/N)) / (2r + 1).
fn factor(n: usize, width: usize, k: usize) -> f64 {
    let r = reach(n, width);
    let sum: f64 = (1..=r).map(|m| (2.0 * PI64 * ((k * m) % n.max(1)) as f64 / n.max(1) as f64).cos()).sum();
    (1.0 + 2.0 * sum) / (2 * r + 1) as f64
}

/// The blur as a matrix (n × n, row after row): row i averages the samples from i − r
/// to i + r, going round the end. Every row is the one above it shifted along by one:
/// a circulant matrix.
fn blur_matrix(n: usize, width: usize) -> Vec<f64> {
    let r = reach(n, width) as i64;
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        for m in -r..=r {
            let col = (i as i64 + m).rem_euclid(n as i64) as usize;
            a[i * n + col] += 1.0 / (2 * r + 1) as f64;
        }
    }
    a
}

/// The blur done directly on the samples (what the tests hold the scene's way, through
/// the waves, to).
#[cfg(test)]
fn blurred(x: &[f64], width: usize) -> Vec<f64> {
    let n = x.len();
    let r = reach(n, width) as i64;
    (0..n).map(|i| (-r..=r).map(|m| x[(i as i64 + m).rem_euclid(n as i64) as usize]).sum::<f64>() / (2 * r + 1) as f64).collect()
}

fn length(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// The first of 1, 1.5, 2, 3, 4, 6, 8 times a power of ten that is at least `v`: a
/// height for a panel that changes only now and then while the signal is dragged.
fn ladder(v: f64) -> f64 {
    let v = if v.is_finite() { v.clamp(1e-6, 1e6) } else { 1.0 };
    let decade = 10f64.powf(v.log10().floor());
    [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0].into_iter().map(|m| m * decade).find(|h| *h >= v * (1.0 - 1e-12)).unwrap_or(10.0 * decade)
}

/// The height of the panel of bars above and below 0, in the units of a coordinate:
/// more than the signal's length, which no coordinate can exceed (a coordinate is the
/// length of a shadow), so that a bar reaches the top only if the signal is nearly
/// all one wave.
fn scale_for(length: f64) -> f64 {
    ladder((length * 1.08).max(0.5))
}

/// How far the panel of products reaches below 0 and above it: past the lowest and the
/// highest of the running sums (and a little either way, however one-sided they are).
fn range_for(sums: &[f64]) -> [f64; 2] {
    let top = sums.iter().fold(0.0f64, |m, s| m.max(*s));
    let bottom = sums.iter().fold(0.0f64, |m, s| m.max(-s));
    let big = top.max(bottom);
    [ladder((bottom * 1.08).max(0.15 * big).max(0.05)), ladder((top * 1.08).max(0.15 * big).max(0.05))]
}

/// The running sums of the products of x and w, sample by sample.
fn running(x: &[f64], w: &[f64]) -> Vec<f64> {
    let mut total = 0.0;
    x.iter().zip(w).map(|(x, w)| {
        total += x * w;
        total
    }).collect()
}

// ----- the signals --------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Preset {
    Chord,
    Wave,
    Spike,
    Bump,
    Step,
    NoisyBump,
    Noise,
    Zero,
}

impl Preset {
    const ALL: [Preset; 8] = [Preset::Chord, Preset::Wave, Preset::Spike, Preset::Bump, Preset::Step, Preset::NoisyBump, Preset::Noise, Preset::Zero];

    fn name(self) -> &'static str {
        match self {
            Preset::Chord => "four waves",
            Preset::Wave => "one wave",
            Preset::Spike => "a spike",
            Preset::Bump => "a smooth bump",
            Preset::Step => "a step",
            Preset::NoisyBump => "a noisy bump",
            Preset::Noise => "noise",
            Preset::Zero => "zero",
        }
    }

    /// Where the spike is, for n samples.
    fn spike_at(n: usize) -> usize {
        n * 5 / 16
    }

    fn samples(self, n: usize) -> Vec<f64> {
        let nf = n as f64;
        let turn = |k: f64, i: usize| 2.0 * PI64 * k * i as f64 / nf;
        let bump = |i: usize| {
            // (the distance round the circle of samples, from 0.4 of the way along)
            let d = (i as f64 - 0.4 * nf + 0.5 * nf).rem_euclid(nf) - 0.5 * nf;
            1.8 * (-0.5 * (d / (0.1 * nf)).powi(2)).exp()
        };
        let mut rng = Rng::new(11);
        let noise: Vec<f64> = (0..n).map(|_| rng.normal() as f64).collect();
        (0..n)
            .map(|i| match self {
                Preset::Chord => 0.95 * turn(2.0, i).cos() + 0.7 * turn(3.0, i).sin() - 0.4 * turn(5.0, i).cos() + 0.25 * turn(7.0, i).sin(),
                Preset::Wave => 1.5 * turn(3.0, i).cos(),
                Preset::Spike => {
                    if i == Self::spike_at(n) {
                        2.0
                    } else {
                        0.0
                    }
                }
                Preset::Bump => bump(i),
                Preset::Step => {
                    if i < n / 2 {
                        -1.0
                    } else {
                        1.0
                    }
                }
                Preset::NoisyBump => bump(i) - 0.5 + 0.35 * noise[i],
                Preset::Noise => 0.8 * noise[i],
                Preset::Zero => 0.0,
            })
            .map(|v| v.clamp(-AMP, AMP))
            .collect()
    }
}

/// Which waves the rebuilding keeps first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Order {
    /// The largest coordinates.
    Largest,
    /// The lowest frequencies.
    Lowest,
}

/// Everything an experiment sets (with N = 32).
#[derive(Clone, Copy, Debug)]
struct Setup {
    preset: Preset,
    /// The wave picked, as a place in the basis.
    picked: usize,
    keep: usize,
    order: Order,
    width: usize,
}

const fn setup(preset: Preset, picked: usize, keep: usize, order: Order, width: usize) -> Setup {
    Setup { preset, picked, keep, order, width }
}

/// What the scene opens with: four waves, the largest picked, the three largest kept.
const OPENING: Setup = setup(Preset::Chord, 3, 3, Order::Largest, 1);

/// The experiments. Every claim in a caption is checked by a test of the same setting.
const TRY: [(&str, &str, Setup); 5] = [
    (
        "one wave, one coordinate",
        r"The signal is a copy of the wave $\cos 3$, 1.5 high. The two are above 0 together and below 0 together, so every product is positive, and the stack climbs to 6.00: the coordinate of $\cos 3$, and the signal's whole length (it lies along that one axis). Every other coordinate is 0, and that one wave alone rebuilds the signal exactly.",
        setup(Preset::Wave, 5, 1, Order::Largest, 1),
    ),
    (
        "two different waves cancel",
        r"The same signal against $\cos 4$. Where the two waves have the same sign the product is yellow, where not, purple. The stack climbs to 1.17, is back at 0 at sample 16, sinks to −0.80 and ends at 0: their dot product is 0. So is that of every two different waves (the map on the right is 1 on its diagonal and 0 everywhere else): the waves are at right angles.",
        setup(Preset::Wave, 7, 1, Order::Largest, 1),
    ),
    (
        "a spike needs every wave",
        r"2 at sample 10, 0 everywhere else. Every product is 0 but one, so each coordinate is twice that wave's value at sample 10, and none is small: the grey lids are all level, every frequency from 1 to 15 holding the same 6.25 % of the length$^2$ (the constant and the alternating wave half that). The 8 largest coordinates hold only 45 % of it: the green reaches 0.91 at the spike, and elsewhere strays as far as 0.60 from 0.",
        setup(Preset::Spike, 1, 8, Order::Largest, 1),
    ),
    (
        "a smooth bump needs few",
        r"A smooth bump. Its coordinates shrink fast as the frequency rises: from frequency 4 up all are below 0.2, from 7 up below 0.001. The green uses only the 7 slowest waves (the constant and frequencies 1 to 3) and holds 99.9 % of the length$^2$: it is within 0.05 of the white everywhere. Smooth signals need few waves.",
        setup(Preset::Bump, 1, 7, Order::Lowest, 1),
    ),
    (
        "a blur multiplies each wave",
        r"Each sample is replaced by the average of 5, itself and two neighbours each side (the orange curve). That treats every position alike: its matrix $C$ (left) is circulant, one row shifted along again and again. In the wave basis the same blur is diagonal (right): each wave is just multiplied by a number, the orange dots, and its coordinate with it (the orange bars). 1 for the constant; frequencies 6 and up are cut to a quarter or less, and 7 to 12 come out upside down, like the ringed $\sin 9$.",
        setup(Preset::NoisyBump, 18, 32, Order::Lowest, 5),
    ),
];

// ----- what is drawn follows what is set ----------------------------------------------

/// A copy of some numbers that follows them: two eases one after the other (so that it
/// sets off gently rather than at full speed), and exactly there in the end.
#[derive(Clone, Debug, Default)]
struct Follow {
    mid: Vec<f64>,
    shown: Vec<f64>,
}

impl Follow {
    fn at(v: &[f64]) -> Follow {
        Follow { mid: v.to_vec(), shown: v.to_vec() }
    }

    /// There at once.
    fn jump(&mut self, v: &[f64]) {
        self.mid = v.to_vec();
        self.shown = v.to_vec();
    }

    /// Put number i at v at once (it follows the hand).
    fn set(&mut self, i: usize, v: f64) {
        if let (Some(a), Some(b)) = (self.mid.get_mut(i), self.shown.get_mut(i)) {
            (*a, *b) = (v, v);
        }
    }

    fn get(&self, i: usize) -> f64 {
        self.shown.get(i).copied().unwrap_or(0.0)
    }

    /// `dt` seconds nearer to `to`; true while still on the way (it arrives when within
    /// `close` of it).
    fn toward(&mut self, to: &[f64], dt: f64, close: f64) -> bool {
        if self.shown.len() != to.len() || self.mid.len() != to.len() {
            self.jump(to);
            return false;
        }
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

// ----- the scene ----------------------------------------------------------------------

pub struct Dft {
    n: usize,
    basis: Basis,
    x: Vec<f64>,
    /// What the signal was made as, to make it again at another N (none once a sample
    /// has been moved by hand, or it has been shifted).
    preset: Option<Preset>,
    picked: usize,
    keep: usize,
    order: Order,
    width: usize,
    /// The samples as drawn (they glide to new values), and how many samples' worth
    /// they are still sliding after a shift.
    shown: Follow,
    slide: Follow,
    /// The picked wave as drawn: its coordinates (all 0 but a 1 at the picked one, at
    /// rest), and where the ring round its bar is.
    wave: Follow,
    ring: Follow,
    /// How much of each wave the rebuilding takes (1 kept, 0 left out, at rest).
    kept: Follow,
    /// The blur's factor for each frequency (all 1 without a blur), and how much of the
    /// blur is drawn (1 with one, 0 without).
    factor: Follow,
    blur_on: Follow,
    /// The height of the panel of bars, and how far the panel of products reaches below
    /// and above 0, in the units of a coordinate.
    scale: Follow,
    range: Follow,
    /// Where the handle for the number of waves kept is along its track (0 to 1).
    knob: Follow,
    knob_held: bool,
    /// Picking a bar by pressing on the bars and moving along them.
    scrubbing: bool,
    /// Something is still on its way.
    moving: bool,
    /// The heat maps, and the size and the blur they were painted for.
    heat: Option<Image>,
    painted: Option<(usize, usize)>,
    /// The few sentences on why, as broken into lines.
    note: Note,
}

impl Dft {
    fn new() -> Dft {
        let n = N_START;
        let mut s = Dft {
            n,
            basis: Basis::new(n),
            x: vec![0.0; n],
            preset: None,
            picked: 0,
            keep: n,
            order: Order::Largest,
            width: 1,
            shown: Follow::default(),
            slide: Follow::at(&[0.0]),
            wave: Follow::default(),
            ring: Follow::default(),
            kept: Follow::default(),
            factor: Follow::default(),
            blur_on: Follow::default(),
            scale: Follow::default(),
            range: Follow::default(),
            knob: Follow::default(),
            knob_held: false,
            scrubbing: false,
            moving: true,
            heat: None,
            painted: None,
            note: Note::default(),
        };
        s.apply(OPENING);
        s.settle();
        s
    }

    /// Set an experiment up.
    fn apply(&mut self, s: Setup) {
        if self.n != N_START {
            self.resize(N_START);
        }
        self.preset = Some(s.preset);
        self.x = s.preset.samples(self.n);
        self.picked = s.picked.min(self.n - 1);
        self.keep = s.keep.min(self.n);
        self.order = s.order;
        self.width = s.width;
        self.moving = true;
    }

    /// Take a new signal of the same size.
    fn take(&mut self, preset: Preset) {
        self.preset = Some(preset);
        self.x = preset.samples(self.n);
        self.moving = true;
    }

    /// Another number of samples: the signal made again (or, if drawn by hand, read off
    /// at the new places), the same wave picked if there is one, and the picture there at once.
    fn resize(&mut self, n: usize) {
        let n = n.clamp(N_MIN, N_MAX);
        if n == self.n {
            return;
        }
        let old = self.n;
        let wave = Wave::of(self.picked, old);
        self.x = match self.preset {
            Some(p) => p.samples(n),
            None => (0..n)
                .map(|i| {
                    let t = i as f64 * old as f64 / n as f64;
                    let (a, u) = (t.floor() as usize % old, t.fract());
                    self.x.get(a).copied().unwrap_or(0.0) * (1.0 - u) + self.x.get((a + 1) % old).copied().unwrap_or(0.0) * u
                })
                .collect(),
        };
        self.picked = if wave.fits(n) { wave.index(n) } else { wave.index(n).min(n - 1) };
        self.keep = ((self.keep * n) as f64 / old as f64).round() as usize;
        self.keep = self.keep.min(n);
        self.n = n;
        self.basis = Basis::new(n);
        self.settle();
        self.moving = true;
    }

    /// Shift the signal one sample along, round the end: the last sample becomes the first.
    fn shift(&mut self) {
        let n = self.n;
        if self.x.len() != n || self.shown.shown.len() != n {
            return;
        }
        self.x.rotate_right(1);
        self.shown.mid.rotate_right(1);
        self.shown.shown.rotate_right(1);
        // drawn where the samples were, and sliding from there (no more than a few
        // turns' worth piled up by a quick hand)
        let s = (self.slide.get(0) + 1.0).min(3.0 * n as f64);
        self.slide.jump(&[s]);
        self.preset = None;
        self.moving = true;
    }

    /// The signal's coordinates: its dot product with each wave.
    fn coords(&self) -> Vec<f64> {
        self.basis.forward(&self.x)
    }

    /// The waves in the order the rebuilding takes them.
    fn keep_order(&self, c: &[f64]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..self.n).collect();
        if self.order == Order::Largest {
            // (a stable sort: equal sizes in order of frequency)
            order.sort_by(|&a, &b| c[b].abs().total_cmp(&c[a].abs()));
        }
        order
    }

    /// 1 for each wave kept, 0 for each left out.
    fn kept_target(&self, c: &[f64]) -> Vec<f64> {
        let mut kept = vec![0.0; self.n];
        for j in self.keep_order(c).into_iter().take(self.keep) {
            kept[j] = 1.0;
        }
        kept
    }

    /// The signal rebuilt from the kept waves alone (at rest).
    fn rebuilt(&self) -> Vec<f64> {
        let c = self.coords();
        let kept = self.kept_target(&c);
        self.basis.back(&c.iter().zip(&kept).map(|(c, k)| c * k).collect::<Vec<_>>())
    }

    /// The share of the signal's length² that the kept waves hold (none for the zero signal).
    fn share_kept(&self) -> Option<f64> {
        let c = self.coords();
        let all: f64 = c.iter().map(|v| v * v).sum();
        let kept: f64 = c.iter().zip(self.kept_target(&c)).map(|(c, k)| c * c * k).sum();
        (all > 1e-24).then(|| kept / all)
    }

    fn factors(&self) -> Vec<f64> {
        (0..frequencies(self.n)).map(|k| factor(self.n, self.width, k)).collect()
    }

    /// Where the drawn things are going.
    fn targets(&self) -> Targets {
        let c = self.coords();
        let mut wave = vec![0.0; self.n];
        wave[self.picked.min(self.n - 1)] = 1.0;
        Targets {
            kept: self.kept_target(&c),
            wave,
            ring: slot(self.picked, self.n),
            factor: self.factors(),
            blur_on: if self.width > 1 && reach(self.n, self.width) > 0 { 1.0 } else { 0.0 },
            scale: scale_for(length(&self.x)),
            range: range_for(&running(&self.x, self.basis.row(self.picked))),
        }
    }

    /// Everything drawn, there at once (and the tests' way to see the scene at rest).
    fn settle(&mut self) {
        let t = self.targets();
        self.shown.jump(&self.x);
        self.slide.jump(&[0.0]);
        self.wave.jump(&t.wave);
        self.ring.jump(&[t.ring]);
        self.kept.jump(&t.kept);
        self.factor.jump(&t.factor);
        self.blur_on.jump(&[t.blur_on]);
        self.scale.jump(&[t.scale]);
        self.range.jump(&t.range);
        self.knob.jump(&[self.keep as f64 / self.n as f64]);
    }

    /// Move everything drawn `dt` seconds nearer to where it is going.
    fn advance(&mut self, dt: f64) {
        let t = self.targets();
        let mut moving = false;
        moving |= self.shown.toward(&self.x, dt, 1e-4);
        moving |= self.slide.toward(&[0.0], dt, 1e-4);
        moving |= self.wave.toward(&t.wave, dt, 1e-4);
        moving |= self.ring.toward(&[t.ring], dt, 1e-3);
        moving |= self.kept.toward(&t.kept, dt, 1e-4);
        moving |= self.factor.toward(&t.factor, dt, 1e-4);
        moving |= self.blur_on.toward(&[t.blur_on], dt, 1e-4);
        moving |= self.scale.toward(&[t.scale], dt, 1e-4 * t.scale);
        moving |= self.range.toward(&t.range, dt, 1e-4 * (t.range[0] + t.range[1]));
        if !self.knob_held {
            moving |= self.knob.toward(&[self.keep as f64 / self.n as f64], dt, 1e-4);
        }
        self.moving = moving;
    }

    /// What is drawn this frame, worked out from the drawn copies.
    fn drawn(&self) -> Drawn {
        let b = &self.basis;
        let xs = self.shown.shown.clone();
        let c0 = b.forward(&xs);
        let s = self.slide.get(0);
        let c = turned(&c0, self.n, s);
        let xr = if s == 0.0 { xs.clone() } else { b.back(&c) };
        let wave = b.back(&self.wave.shown);
        let kept = self.kept.shown.clone();
        let rebuilt = b.back(&c0.iter().zip(&kept).map(|(c, k)| c * k).collect::<Vec<_>>());
        let factor = self.factor.shown.clone();
        let by_wave: Vec<f64> = (0..self.n).map(|j| factor.get(Wave::of(j, self.n).k).copied().unwrap_or(1.0)).collect();
        let blurred = b.back(&c0.iter().zip(&by_wave).map(|(c, f)| c * f).collect::<Vec<_>>());
        Drawn { s, xs, c, xr, wave, kept, rebuilt, blurred, factor, by_wave, blur: self.blur_on.get(0), scale: self.scale.get(0).max(1e-6), range: [self.range.get(0).max(1e-6), self.range.get(1).max(1e-6)], ring: self.ring.get(0) }
    }

    /// Paint the heat maps: without a blur, W (the waves as its rows, each drawn as
    /// bright as it goes) and W Wᵀ; with one, the blur's matrix C (each of its non-zero
    /// entries, all the same, drawn as 1) and W C Wᵀ.
    fn paint(&mut self, gfx: &Gfx) {
        let n = self.n;
        let key = (n, if reach(n, self.width) > 0 { self.width } else { 1 });
        if self.painted == Some(key) && self.heat.is_some() {
            return;
        }
        let (w, h) = (2 * n + 1, n);
        let image = self.heat.get_or_insert_with(|| Image::new(gfx, w as u32, h as u32));
        image.ensure_size(gfx, w as u32, h as u32);
        if image.size() != [w as u32, h as u32] {
            return;
        }
        let (left, right) = if key.1 > 1 {
            let c = blur_matrix(n, key.1);
            let d = self.basis.in_waves(&c);
            let width = (2 * reach(n, key.1) + 1) as f64;
            (c.iter().map(|v| v * width).collect::<Vec<f64>>(), d)
        } else {
            let top = (2.0 / n as f64).sqrt();
            (self.basis.w.iter().map(|v| v / top).collect(), gram(&self.basis))
        };
        let table = HEAT.table();
        let pixels = image.pixels_mut();
        for i in 0..n {
            // (pixel rows count from the bottom, matrix rows from the top)
            let row = &mut pixels[(n - 1 - i) * w..(n - i) * w];
            for j in 0..n {
                row[j] = table[heat_index(left[i * n + j])];
                row[n + 1 + j] = table[heat_index(right[i * n + j])];
            }
            row[n] = Color::CLEAR;
        }
        self.painted = Some(key);
    }
}

/// Where a value between −1 and 1 goes in the heat maps' colours (not all the way to
/// the ends, which are bright enough to shout).
fn heat_index(v: f64) -> usize {
    // (small values brightened a little, so that a quarter is not lost in the dark middle)
    let v = v.clamp(-1.0, 1.0);
    table_index(0.5 + 0.45 * (v.signum() * v.abs().powf(0.6)) as f32)
}

struct Targets {
    kept: Vec<f64>,
    wave: Vec<f64>,
    ring: f64,
    factor: Vec<f64>,
    blur_on: f64,
    scale: f64,
    range: [f64; 2],
}

/// What a frame draws.
struct Drawn {
    /// How far the samples are still sliding (in samples).
    s: f64,
    /// The signal as drawn after the shift (drawn s samples to the left of its places).
    xs: Vec<f64>,
    /// The coordinates as drawn: turned back part of the way while sliding.
    c: Vec<f64>,
    /// The signal those coordinates make (whose products with the wave are drawn).
    xr: Vec<f64>,
    /// The picked wave.
    wave: Vec<f64>,
    kept: Vec<f64>,
    /// Like `xs`: the rebuilt signal and the blurred one.
    rebuilt: Vec<f64>,
    blurred: Vec<f64>,
    /// The blur's factor for each frequency, and for each wave.
    factor: Vec<f64>,
    by_wave: Vec<f64>,
    /// How much of the blur is shown (0 to 1).
    blur: f64,
    scale: f64,
    /// How far the panel of products reaches below 0 and above it.
    range: [f64; 2],
    /// Where the ring round the picked bar is (a place among the bars).
    ring: f64,
}

/// Where wave j's bar is among the bars, in widths of a bar: the bars of one frequency
/// side by side, half a bar between frequencies.
fn slot(j: usize, n: usize) -> f64 {
    j as f64 + 0.5 * Wave::of(j, n).k as f64 + 0.5
}

/// How many bar widths all the bars take.
fn slots(n: usize) -> f64 {
    n as f64 + 0.5 * (frequencies(n) - 1) as f64
}

// ----- where everything goes ----------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct Area {
    min: Vec2,
    max: Vec2,
}

impl Area {
    fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Area {
        // (never upside down, and never of no size: a tiny window squeezes, it does not fold)
        let (lo_x, hi_x) = (x0.min(x1), x0.max(x1).max(x0.min(x1) + 1e-3));
        let (lo_y, hi_y) = (y0.min(y1), y0.max(y1).max(y0.min(y1) + 1e-3));
        Area { min: Vec2::new(lo_x, lo_y), max: Vec2::new(hi_x, hi_y) }
    }

    fn w(&self) -> f32 {
        self.max.x - self.min.x
    }

    fn h(&self) -> f32 {
        self.max.y - self.min.y
    }

    fn mid(&self) -> f32 {
        0.5 * (self.min.y + self.max.y)
    }

    fn contains(&self, p: Vec2, margin: f32) -> bool {
        p.x >= self.min.x - margin && p.x <= self.max.x + margin && p.y >= self.min.y - margin && p.y <= self.max.y + margin
    }
}

/// Where everything goes, in the units of the stage (five high, as wide as the canvas
/// makes it, centred on 0). On a wide canvas, the signal's side (samples) in a column on
/// the left and the waves' side (coordinates) on the right; on a narrow one, all in one column.
struct Layout {
    wide: bool,
    /// Text sizes in logical pixels, and units per logical pixel.
    size: f32,
    small: f32,
    px: f32,
    signal: Area,
    wave: Area,
    products: Area,
    coords: Area,
    rebuild: Area,
    /// The ends of the track of the handle for how many waves are kept.
    track: (Vec2, Vec2),
    /// The two heat maps (none: no room), side by side or one above the other.
    heat: Option<[Area; 2]>,
    /// Where a few sentences on why go: top-left corner and width (none: no room).
    note: Option<(Vec2, f32)>,
}

impl Layout {
    fn of(w: f32, ppu: f32, inset: f32) -> Layout {
        let px = 1.0 / ppu.max(1e-3);
        let wide = w >= 4.6;
        let (size, small) = if wide { (13.0, 11.0) } else { (11.0, 9.5) };
        let title = (size + 10.0) * px;
        let gap = (if wide { 12.0 } else { 7.0 }) * px;
        let top = 2.5 - (inset + 8.0) * px;
        let bottom = -2.5 + 8.0 * px;
        let numbers = 15.0 * px;
        let track_h = 24.0 * px;
        let label = (small + 5.0) * px;
        if wide {
            let left = -0.5 * w + 36.0 * px;
            let split = -0.5 * w + 0.54 * w;
            let right = 0.5 * w - 26.0 * px;
            let c_left = split + 76.0 * px;
            // the left column: four panels, each under its title, the numbers of the
            // samples under the first, and the handle's track above the last
            let fixed = 4.0 * title + 3.0 * gap + 2.0 * numbers + track_h;
            let room = (top - bottom - fixed).max(0.2);
            let [hs, hw, hp, hr] = [0.31, 0.12, 0.27, 0.30].map(|s| s * room);
            let y = top - title;
            let signal = Area::new(left, y - hs, split, y);
            let y = signal.min.y - numbers - gap - title;
            let wave = Area::new(left, y - hw, split, y);
            let y = wave.min.y - gap - title;
            let products = Area::new(left, y - hp, split, y);
            let y = products.min.y - gap - title;
            let track = (Vec2::new(left + 44.0 * px, y - 0.5 * track_h), Vec2::new(split - 150.0 * px, y - 0.5 * track_h));
            let y = y - track_h;
            let rebuild = Area::new(left, y - hr, split, y);
            // the right column: the bars under two lines of title, the frequencies under
            // them, then the heat maps side by side, each under two lines of its own
            let y = top - 2.0 * title;
            let coords = Area::new(c_left, y - (top - bottom) * 0.42, right, y);
            let heat_top = coords.min.y - 2.0 * numbers - 2.0 * gap - 2.0 * label;
            let apart = 36.0 * px;
            let side = (heat_top - bottom - 4.0 * px).min((right - c_left - apart) * 0.5);
            let first = Area::new(c_left, heat_top - side, c_left + side, heat_top);
            let second = Area::new(c_left + side + apart, heat_top - side, c_left + 2.0 * side + apart, heat_top);
            let heat = (side * ppu >= 48.0).then_some([first, second]);
            let note_top = first.min.y - 2.0 * gap;
            let note = (note_top - bottom > 7.0 * label && heat.is_some()).then_some((Vec2::new(c_left, note_top), right - c_left));
            Layout { wide, size, small, px, signal, wave, products, coords, rebuild, track, heat, note }
        } else {
            let left = -0.5 * w + 28.0 * px;
            let right = 0.5 * w - 10.0 * px;
            let fixed = 5.0 * title + label + 5.0 * gap + 3.0 * numbers + track_h;
            let room = (top - bottom - fixed).max(0.2);
            let [hs, hw, hp, hc, hr] = [0.22, 0.09, 0.19, 0.29, 0.21].map(|s| s * room);
            let y = top - title;
            let signal = Area::new(left, y - hs, right, y);
            let y = signal.min.y - numbers - gap - title;
            let wave = Area::new(left, y - hw, right, y);
            let y = wave.min.y - gap - title;
            let products = Area::new(left, y - hp, right, y);
            // the bars under two lines of title, with the heat maps one above the other
            // in a column beside them
            let y0 = products.min.y - gap;
            let block = title + label + hc + 2.0 * numbers;
            let side = ((block - 2.0 * label - gap - 8.0 * px) * 0.5).min((right - left) * 0.3);
            let heat_left = right - side;
            let y = y0 - title - label;
            let coords = Area::new(left, y - hc, heat_left - 16.0 * px, y);
            let first = Area::new(heat_left, y0 - label - 4.0 * px - side, right, y0 - label - 4.0 * px);
            let second = Area::new(heat_left, first.min.y - gap - label - 4.0 * px - side, right, first.min.y - gap - label - 4.0 * px);
            let heat = (side * ppu >= 36.0).then_some([first, second]);
            let y = coords.min.y - 2.0 * numbers - gap - title;
            let track = (Vec2::new(left + 40.0 * px, y - 0.5 * track_h), Vec2::new(right - 120.0 * px, y - 0.5 * track_h));
            let y = y - track_h;
            let rebuild = Area::new(left, y - hr, right, y);
            Layout { wide, size, small, px, signal, wave, products, coords, rebuild, track, heat, note: None }
        }
    }

    /// Where sample position t (0 to N − 1, or between) is across the signal's panels.
    fn at(&self, t: f64, n: usize) -> f32 {
        let a = self.signal;
        a.min.x + (t as f32 + 0.5) * a.w() / n.max(1) as f32
    }

    /// Where bar place p is across the panel of bars.
    fn bar(&self, p: f64, n: usize) -> f32 {
        let a = self.coords;
        a.min.x + p as f32 * a.w() / slots(n) as f32
    }
}

/// `text_width`, remembered for the texts with formulas in them: measuring a formula
/// means laying it out, too dear to do for every label in every frame, and the labels
/// are mostly the same from one frame to the next.
fn width(text: &str, size: f32) -> f32 {
    thread_local! {
        static KNOWN: std::cell::RefCell<std::collections::HashMap<(String, u32), f32>> = std::cell::RefCell::new(std::collections::HashMap::new());
    }
    if !crate::typeset::has_math(text) {
        return text_width(text, size);
    }
    KNOWN.with(|known| {
        let mut known = known.borrow_mut();
        let key = (text.to_string(), size.to_bits());
        if let Some(w) = known.get(&key) {
            return *w;
        }
        // (a label with a number in it is a new text whenever the number changes)
        if known.len() >= 256 {
            known.clear();
        }
        let w = text_width(text, size);
        known.insert(key, w);
        w
    })
}

/// The first of `options` that fits in `room` units at this zoom (the last if none does).
fn pick<'a>(options: &[&'a str], room: f32, px: f32, size: f32) -> &'a str {
    options.iter().copied().find(|t| width(t, size) * px < room).unwrap_or(options.last().copied().unwrap_or(""))
}

/// Words in their colours side by side, a key, starting at `at` (the left end of the
/// line, at its middle).
fn key_row(f: &mut Frame, at: Vec2, parts: &[(Color, &str)], size: f32, px: f32) {
    let mut x = at.x;
    for (color, text) in parts {
        f.text((x, at.y), *text, size, *color, Align2::LEFT_CENTER);
        x += (width(text, size) + 18.0) * px;
    }
}

/// The height of the panel of bars, as written beside it: 8, 1.5, 0.75, −4.
fn scale_text(v: f64) -> String {
    let text = num(v, 2);
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

/// A few sentences on the picture, broken into lines as `wrapped` breaks them, and kept
/// until the text or the room for it changes. (`wrapped` measures every word and every
/// line of a text with formulas in it again in every frame: more work than all the rest
/// of the picture.)
#[derive(Default)]
struct Note {
    /// What the lines were broken for: the text, the room in logical pixels, the size.
    key: (String, f32, f32),
    lines: Vec<String>,
}

impl Note {
    fn draw(&mut self, f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) {
        let room = width * pixels_per_unit(f);
        if !crate::typeset::has_math(text) {
            wrapped(f, at, width, text, size, color);
            return;
        }
        if (self.key.0.as_str(), self.key.1, self.key.2) != (text, room, size) {
            match note_lines(f, text, room, size) {
                Some(lines) => (self.key, self.lines) = ((text.to_string(), room, size), lines),
                // (the formulas cannot be measured before their fonts are in place)
                None => {
                    wrapped(f, at, width, text, size, color);
                    return;
                }
            }
        }
        for (i, line) in self.lines.iter().enumerate() {
            f.text_off(at, Vec2::new(0.0, i as f32 * (size + 3.0)), line.as_str(), size, color, Align2::LEFT_TOP);
        }
    }
}

/// A text with formulas in it broken into lines no wider than `room` logical pixels,
/// exactly as `wrapped` breaks it; none if a formula cannot be measured yet.
fn note_lines(f: &Frame, text: &str, room: f32, size: f32) -> Option<Vec<String>> {
    let measure = |t: &str| if crate::typeset::has_math(t) { crate::typeset::measure(f.ctx, t, size).map(|s| s.x) } else { Some(f.text_size(t, size).x) };
    let mut lines = vec![String::new()];
    for word in note_words(text) {
        let last = lines.last_mut()?;
        if last.is_empty() {
            last.push_str(word);
        } else if measure(last)? + measure(word)? + size * 0.47 > room {
            lines.push(word.to_string());
        } else {
            last.push(' ');
            last.push_str(word);
        }
    }
    Some(lines)
}

/// The words of a text, each formula whole (the words `wrapped` breaks lines between).
fn note_words(text: &str) -> Vec<&str> {
    let (mut words, mut start, mut inside) = (Vec::new(), 0, false);
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

/// A button drawn over the picture (logical pixels): true when it is clicked.
fn hud_button(f: &mut Frame, min: Vec2, max: Vec2, text: &str, pointer: Option<Vec2>) -> bool {
    let over = pointer.is_some_and(|p| p.cmpge(min).all() && p.cmple(max).all());
    hud_rect(f, min, max, if over { Color::hex(0x33405E) } else { Color::hex(0x222B40) });
    f.text_px((min + max) * 0.5, text, 12.0, pal::TEXT, Align2::CENTER_CENTER);
    over && f.ptr.clicked
}

/// The places a sample drawn at position q (in samples, while sliding) shows at: itself
/// and, near the ends, its copy round the other end, each with how much of it shows
/// (fading out as it leaves the row).
fn copies(q: f64, n: usize) -> [(f64, f32); 2] {
    let nf = n as f64;
    let q = q.rem_euclid(nf);
    let show = |q: f64| (1.0 - (-q).max(q - (nf - 1.0)).max(0.0)).clamp(0.0, 1.0) as f32;
    [(q, show(q)), (q - nf, show(q - nf))]
}

/// The points of a periodic row of values drawn `s` samples to the left, for a curve
/// through them that slides through the ends of the row: in order across the row, cut
/// off half a sample beyond the first and last places. The same number of points in the
/// same order all through a slide (a sample leaving at one end is not moved round to the
/// other end of the list), so that the curve is drawn moving evenly.
fn periodic_points(v: &[f64], s: f64) -> Vec<(f64, f64)> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f64;
    // the samples from one before the row to two after it, at their places now (for a
    // slide of s, sample j + c is at j + c − s, which is between j − 1 and j)
    let c = if s > 0.0 { s.ceil() - 1.0 } else { 0.0 };
    let raw: Vec<(f64, f64)> = (-1..=n as i64 + 1).map(|j| (j as f64 + c - s, v[(j + c as i64).rem_euclid(n as i64) as usize])).collect();
    // where the curve crosses a place t between two of them
    let along = |t: f64| {
        raw.windows(2).find(|w| w[0].0 <= t && t <= w[1].0).map_or(raw[0].1, |w| {
            let span = (w[1].0 - w[0].0).max(1e-12);
            w[0].1 + (w[1].1 - w[0].1) * (t - w[0].0) / span
        })
    };
    let (lo, hi) = (-0.5, nf - 0.5);
    raw.iter().map(|&(q, y)| if q < lo { (lo, along(lo)) } else if q > hi { (hi, along(hi)) } else { (q, y) }).collect()
}

impl Dft {
    /// The signal: stems with handles, and the blurred signal over it.
    fn draw_signal(&mut self, f: &mut Frame, lay: &Layout, d: &Drawn, handles: bool, pointer: Option<Vec2>) {
        let (n, a, px) = (self.n, lay.signal, lay.px);
        let half = 0.5 * a.h();
        let y_of = |v: f64| a.mid() + (v / (AMP * 1.1)) as f32 * half;
        inset(f, a.min - Vec2::splat(4.0 * px), a.max + Vec2::splat(4.0 * px));
        for v in [-2.0, -1.0, 1.0, 2.0] {
            f.line((a.min.x, y_of(v)), (a.max.x, y_of(v)), 1.0, pal::FAINT.a(0.45));
            f.text((a.min.x - 6.0 * px, y_of(v)), num(v, 0), lay.small, pal::DIM, Align2::RIGHT_CENTER);
        }
        f.line((a.min.x, y_of(0.0)), (a.max.x, y_of(0.0)), 1.0, pal::DIM.a(0.8));
        f.text((a.min.x - 6.0 * px, y_of(0.0)), "0", lay.small, pal::DIM, Align2::RIGHT_CENTER);
        // which sample is which, under the row
        let every = [1usize, 2, 4, 8, 16].into_iter().find(|e| a.w() / n as f32 * *e as f32 / px >= 22.0).unwrap_or(16);
        for i in (0..n).step_by(every) {
            f.text((lay.at(i as f64, n), a.min.y - 3.0 * px), i.to_string(), lay.small, pal::DIM, Align2::CENTER_TOP);
        }
        let pitch_px = a.w() / n as f32 / px;
        // the blur, after it
        if d.blur > 0.0 {
            let pts = periodic_points(&d.blurred, d.s);
            f.path(pts.iter().map(|(q, v)| Vec2::new(lay.at(*q, n), y_of(*v))), 2.2, BLUR.a(d.blur as f32));
        }
        // the samples: a stem each, with a handle on top (but not while they slide)
        let stem = (0.22 * pitch_px).clamp(1.2, 2.6);
        let head = (0.55 * pitch_px).clamp(4.0, 11.0);
        for (i, &v) in d.xs.iter().enumerate() {
            for (q, show) in copies(i as f64 - d.s, n) {
                if show <= 0.0 {
                    continue;
                }
                let xq = lay.at(q, n);
                f.line((xq, y_of(0.0)), (xq, y_of(v)), stem, SIGNAL.a(0.75 * show));
                // (the handles are the heads, where there are handles)
                if !handles {
                    f.dot((xq, y_of(v)), head, SIGNAL.a(show));
                }
            }
        }
        // what it is, and the button that shifts it
        let button_w = 112.0;
        let bmax = on_screen(f, Vec3::new(a.max.x, a.max.y + 4.0 * px, 0.0)).unwrap_or(Vec2::ZERO);
        let bmin = bmax - Vec2::new(button_w, 22.0);
        if hud_button(f, bmin, bmax, "shift by one ▸", pointer) {
            self.shift();
        }
        let room = a.w() - (button_w + 12.0) * px;
        let long = format!("$x$, the signal: {n} numbers, one per sample (drag the white dots)");
        let short = format!("$x$, the signal: {n} numbers");
        let title = pick(&[&long, &short, "$x$"], room, px, lay.size);
        f.text((a.min.x, a.max.y + 6.0 * px), title, lay.size, SIGNAL, Align2::LEFT_BOTTOM);
        let len2: f64 = self.x.iter().map(|v| v * v).sum();
        f.text((a.max.x - 6.0 * px, a.max.y - 4.0 * px), format!(r"$\text{{length}}^2 = \sum x_n^2 = {}$", num(len2, 2)), lay.small, pal::DIM, Align2::RIGHT_TOP);
    }

    /// The handles on the samples (not while they slide): taken before the picture is
    /// worked out, so that the stem under a dragged handle is drawn where the handle now is.
    /// Returns whether there were handles to drag this frame.
    fn drag_samples(&mut self, f: &mut Frame, lay: &Layout) -> bool {
        let (n, a, px) = (self.n, lay.signal, lay.px);
        if self.slide.get(0) != 0.0 {
            return false;
        }
        let half = 0.5 * a.h();
        let head = (0.55 * a.w() / n as f32 / px).clamp(4.0, 11.0);
        for i in 0..n {
            let v = self.shown.get(i);
            let mut p = Vec3::new(lay.at(i as f64, n), a.mid() + (v / (AMP * 1.1)) as f32 * half, 0.0);
            if f.drag3_sized(100 + i as u32, &mut p, Some(Vec3::Z), SIGNAL, head) {
                let new = (((p.y - a.mid()) / half.max(1e-6)) as f64 * AMP * 1.1).clamp(-AMP, AMP);
                if new.is_finite() && i < self.x.len() {
                    self.x[i] = new;
                    self.shown.set(i, new);
                    self.preset = None;
                    self.moving = true;
                }
            }
        }
        true
    }

    /// The picked wave, sample by sample, over the smooth curve it is the samples of.
    fn draw_wave(&self, f: &mut Frame, lay: &Layout, d: &Drawn) {
        let (n, a, px) = (self.n, lay.wave, lay.px);
        let top = (2.0 / n as f64).sqrt();
        let y_of = |v: f64| a.mid() + (v / (top * 1.15)) as f32 * 0.5 * a.h();
        let wave = Wave::of(self.picked, n);
        let color = wave.color();
        f.line((a.min.x, y_of(0.0)), (a.max.x, y_of(0.0)), 1.0, pal::DIM.a(0.8));
        // the smooth curve: the drawn wave's coordinates, at every point between the samples
        let parts: Vec<(Wave, f64)> = self.wave.shown.iter().enumerate().filter(|(_, a)| a.abs() > 1e-6).map(|(j, a)| (Wave::of(j, n), *a)).collect();
        let steps = (n * 6).min(320);
        f.path(
            (0..=steps).map(|s| {
                let t = -0.5 + n as f64 * s as f64 / steps as f64;
                Vec2::new(lay.at(t, n), y_of(parts.iter().map(|(w, a)| a * w.at(n, t)).sum()))
            }),
            1.2,
            color.a(0.35),
        );
        let pitch_px = a.w() / n as f32 / px;
        let stem = (0.22 * pitch_px).clamp(1.2, 2.6);
        for (i, &v) in d.wave.iter().enumerate() {
            let x = lay.at(i as f64, n);
            f.line((x, y_of(0.0)), (x, y_of(v)), stem, color.a(0.9));
            f.dot((x, y_of(v)), (0.4 * pitch_px).clamp(3.0, 7.0), color);
        }
        let size = if wave.kind == Kind::Constant || wave.kind == Kind::Alternating { 1.0 / (n as f64).sqrt() } else { top };
        f.text((a.min.x - 6.0 * px, y_of(top)), format!("+{}", num(top, 2)), lay.small, pal::DIM, Align2::RIGHT_CENTER);
        f.text((a.min.x - 6.0 * px, y_of(-top)), num(-top, 2), lay.small, pal::DIM, Align2::RIGHT_CENTER);
        let long = format!(r"$\times$ {}, of length 1: its numbers are at most $\pm {}$", wave.called(), num(size, 2));
        let short = format!(r"$\times$ {}", wave.called());
        f.text((a.min.x, a.max.y + 6.0 * px), pick(&[&long, &short], a.w(), px, lay.size), lay.size, color, Align2::LEFT_BOTTOM);
    }

    /// The products of the signal and the wave, sample by sample, each stacked on the
    /// ones before: the last level is their sum, the coordinate.
    fn draw_products(&self, f: &mut Frame, lay: &Layout, d: &Drawn) {
        let (n, a, px) = (self.n, lay.products, lay.px);
        let [below, above] = d.range;
        let y_of = |v: f64| a.min.y + (((v + below) / (below + above)).clamp(-0.03, 1.03)) as f32 * a.h();
        inset(f, a.min - Vec2::splat(4.0 * px), a.max + Vec2::splat(4.0 * px));
        f.line((a.min.x, y_of(0.0)), (a.max.x, y_of(0.0)), 1.0, pal::DIM.a(0.8));
        f.text((a.min.x - 6.0 * px, y_of(0.0)), "0", lay.small, pal::DIM, Align2::RIGHT_CENTER);
        let pitch = a.w() / n as f32;
        let mut total = 0.0;
        for (i, (x, w)) in d.xr.iter().zip(&d.wave).enumerate() {
            let p = x * w;
            let (from, to) = (total, total + p);
            let xc = lay.at(i as f64, n);
            let (lo, hi) = (y_of(from.min(to)), y_of(from.max(to)));
            let color = if p >= 0.0 { UP } else { DOWN };
            f.dl.tris_alpha.rect(Vec2::new(xc - 0.36 * pitch, lo), Vec2::new(xc + 0.36 * pitch, hi.max(lo + 1.0 * px)), 0.0, color.a(0.85));
            // the level reached, carried on to the next bar
            if i + 1 < n {
                f.line((xc + 0.36 * pitch, y_of(to)), (xc + 0.64 * pitch, y_of(to)), 1.0, pal::DIM.a(0.9));
            }
            total = to;
        }
        let end = y_of(total);
        let wave = Wave::of(self.picked, n);
        let sum = format!("$= {}$", num(total, 2));
        if lay.wide {
            f.dashed((a.min.x, end), (a.max.x + 8.0 * px, end), 1.4, UP.a(0.9));
            f.text((a.max.x + 10.0 * px, end), sum, lay.size, UP, Align2::LEFT_CENTER);
        } else {
            // (no room beside the panel: inside it at the right, on the side of the line
            // with more room)
            f.dashed((a.min.x, end), (a.max.x, end), 1.4, UP.a(0.9));
            let above = end < a.mid() + 0.25 * a.h();
            let y = if above { end + 3.0 * px } else { end - 3.0 * px };
            f.caption_px(
                on_screen(f, Vec3::new(a.max.x - 4.0 * px, y, 0.0)).unwrap_or(Vec2::ZERO),
                sum,
                lay.size,
                UP,
                if above { Align2::RIGHT_BOTTOM } else { Align2::RIGHT_TOP },
            );
        }
        // (the numbers at the ends, where they are not on top of the 0)
        for v in [-below, above] {
            if (y_of(v) - y_of(0.0)).abs() > (lay.small + 2.0) * px {
                f.text((a.min.x - 6.0 * px, y_of(v)), scale_text(v), lay.small, pal::DIM, Align2::RIGHT_CENTER);
            }
        }
        let long = format!(r"$=$ the products $x_n \times w_n$, each stacked on the ones before: their sum is $x \cdot {}$", wave.symbol());
        let short = r"$=$ the products $x_n \times w_n$, stacked: their sum is the dot product".to_string();
        f.text((a.min.x, a.max.y + 6.0 * px), pick(&[&long, &short, "products, stacked"], a.w(), px, lay.size), lay.size, UP, Align2::LEFT_BOTTOM);
    }

    /// The coordinates, one bar for each wave, in order of frequency.
    fn draw_coords(&self, f: &mut Frame, lay: &Layout, d: &Drawn) {
        let (n, a, px) = (self.n, lay.coords, lay.px);
        let half = 0.5 * a.h();
        let y_of = |v: f64| a.mid() + ((v / d.scale).clamp(-1.05, 1.05)) as f32 * half;
        let pitch = a.w() / slots(n) as f32;
        inset(f, a.min - Vec2::splat(4.0 * px), a.max + Vec2::splat(4.0 * px));
        f.line((a.min.x, y_of(0.0)), (a.max.x, y_of(0.0)), 1.0, pal::DIM.a(0.8));
        for v in [-d.scale, d.scale] {
            f.line((a.min.x, y_of(v)), (a.max.x, y_of(v)), 1.0, pal::FAINT.a(0.45));
            f.text((a.min.x - 6.0 * px, y_of(v)), scale_text(v), lay.small, pal::DIM, Align2::RIGHT_CENTER);
        }
        f.text((a.min.x - 6.0 * px, y_of(0.0)), "0", lay.small, pal::DIM, Align2::RIGHT_CENTER);
        let mut room = Room::new(f);
        // each frequency's size, how far its cosine and sine bars reach together, as a
        // grey lid over the pair: neither bar can be longer, and a shift only trades
        // length between them under it
        for k in 0..frequencies(n) {
            let r = members(k, n);
            let size = r.clone().map(|j| d.c.get(j).map_or(0.0, |c| c * c)).sum::<f64>().sqrt();
            let (x0, x1) = (lay.bar(slot(r.start, n) - 0.5, n), lay.bar(slot(r.end - 1, n) + 0.5, n));
            let (top, tick) = (y_of(size), (4.0 * px).min(0.5 * (y_of(size) - y_of(0.0)).abs()));
            f.path([Vec2::new(x0, top - tick), Vec2::new(x0, top), Vec2::new(x1, top), Vec2::new(x1, top - tick)], 1.4, SIZE);
            room.line(f, Vec3::new(x0, y_of(size), 0.0), Vec3::new(x1, y_of(size), 0.0));
        }
        for j in 0..n {
            let wave = Wave::of(j, n);
            let c = d.c.get(j).copied().unwrap_or(0.0);
            let kept = d.kept.get(j).copied().unwrap_or(1.0) as f32;
            let xc = lay.bar(slot(j, n), n);
            let (lo, hi) = (y_of(c.min(0.0)), y_of(c.max(0.0)));
            f.dl.tris_alpha.rect(Vec2::new(xc - 0.4 * pitch, lo), Vec2::new(xc + 0.4 * pitch, hi.max(lo + px)), 0.0, wave.color().a(0.22 + 0.68 * kept));
            room.line(f, Vec3::new(xc, y_of(0.0), 0.0), Vec3::new(xc, y_of(c), 0.0));
            if d.blur > 0.0 {
                let b = c * d.by_wave.get(j).copied().unwrap_or(1.0);
                let (lo, hi) = (y_of(b.min(0.0)), y_of(b.max(0.0)));
                f.dl.tris_alpha.rect(Vec2::new(xc - 0.18 * pitch, lo), Vec2::new(xc + 0.18 * pitch, hi.max(lo + px)), 0.0, BLUR.a(d.blur as f32));
            }
        }
        // the blur's factors: 1 at the top line, 0 at the middle one
        if d.blur > 0.0 {
            let pts: Vec<Vec2> = (0..frequencies(n))
                .map(|k| {
                    let r = members(k, n);
                    let x = lay.bar(0.5 * (slot(r.start, n) + slot(r.end - 1, n)), n);
                    Vec2::new(x, a.mid() + d.factor.get(k).copied().unwrap_or(1.0).clamp(-1.0, 1.0) as f32 * half)
                })
                .collect();
            f.path(pts.iter().copied(), 1.4, BLUR.a(0.75 * d.blur as f32));
            for p in &pts {
                f.dot(*p, 5.5, BLUR.a(d.blur as f32));
            }
            for w in pts.windows(2) {
                room.line(f, w[0].extend(0.0), w[1].extend(0.0));
            }
            let says = pick(&[r"orange dots: each frequency's factor, $\times 1$ at the top line, $\times 0$ at the middle", r"dots: factor, top $\times 1$, middle $\times 0$", ""], 0.75 * a.w(), px, lay.small);
            f.text((a.max.x - 6.0 * px, a.max.y - 8.0 * px - lay.small * px), says, lay.small, BLUR.a(d.blur as f32), Align2::RIGHT_TOP);
        }
        // the picked one, ringed, with its name and its coordinate
        let rx = lay.bar(d.ring, n);
        // (as tall as the bar it rings, gliding from one bar's height to the next's with it)
        let picked_c: f64 = self.wave.shown.iter().zip(&d.c).map(|(a, c)| a * c).sum();
        let (lo, hi) = (y_of(picked_c.min(0.0)) - 3.0 * px, y_of(picked_c.max(0.0)) + 3.0 * px);
        frame_rect(f, Vec2::new(rx - 0.62 * pitch, lo), Vec2::new(rx + 0.62 * pitch, hi), 1.6, UP);
        for (p, q) in [((-0.62, lo), (-0.62, hi)), ((0.62, lo), (0.62, hi)), ((-0.62, hi), (0.62, hi)), ((-0.62, lo), (0.62, lo))] {
            room.line(f, Vec3::new(rx + p.0 * pitch, p.1, 0.0), Vec3::new(rx + q.0 * pitch, q.1, 0.0));
        }
        let wave = Wave::of(self.picked, n);
        // (it stays inside the panel: the titles above it, the numbers beside and below it
        // and the neighbouring panels are no place for it)
        if let (Some(p), Some(q)) = (on_screen(f, Vec3::new(a.min.x, a.max.y, 0.0)), on_screen(f, Vec3::new(a.max.x, a.min.y, 0.0))) {
            let far = 1e5;
            room.wall(Vec2::new(-far, -far), Vec2::new(far, p.y));
            room.wall(Vec2::new(-far, q.y), Vec2::new(far, far));
            room.wall(Vec2::new(-far, -far), Vec2::new(p.x, far));
            room.wall(Vec2::new(q.x, -far), Vec2::new(far, far));
        }
        let up = if picked_c >= 0.0 { Vec2::new(0.0, -1.0) } else { Vec2::new(0.0, 1.0) };
        let away = if picked_c >= 0.0 { 1.0 } else { -1.0 };
        let tip = |more: f32| Vec3::new(rx, if picked_c >= 0.0 { hi } else { lo } + away * more * px, 0.0);
        let value = d.c.get(self.picked).copied().unwrap_or(0.0);
        let text = if d.blur > 0.0 {
            let after = value * d.by_wave.get(self.picked).copied().unwrap_or(1.0);
            format!(r"${} = {} \to {}$", wave.short(), num(value, 2), num(after, 2))
        } else {
            format!("${} = {}$", wave.short(), num(value, 2))
        };
        // (nor the words at the top right of the panel)
        if let Some(corner) = on_screen(f, a.max.extend(0.0)) {
            let (lines, wide) = if d.blur > 0.0 { (2.0, 0.75 * a.w() / px) } else { (1.0, width(r"$\sum \text{coordinate}^2 = 00.00$", lay.small) + 12.0) };
            room.wall(corner - Vec2::new(wide, 0.0), corner + Vec2::new(0.0, 6.0 + lines * (lay.small + 6.0)));
        }
        // (beside the end of the bar, or, where that is crowded, further along its line)
        // and failing all that, at the top left of the panel
        let corner = Vec3::new(a.min.x + (width(&text, lay.small + 1.0) * 0.5 + 8.0) * px, a.max.y - (lay.small + 4.0) * px, 0.0);
        room.label_any(f, "picked", &[tip(0.0), tip(24.0), tip(48.0), tip(72.0), corner], &text, lay.small + 1.0, UP, up, 5.0, false);
        // which frequency is which, under the bars
        let every = [1usize, 2, 4, 8].into_iter().find(|e| a.w() / (frequencies(n) as f32) * *e as f32 / px >= 18.0).unwrap_or(8);
        for k in (0..frequencies(n)).step_by(every) {
            let r = members(k, n);
            let x = lay.bar(0.5 * (slot(r.start, n) + slot(r.end - 1, n)), n);
            f.text((x, a.min.y - 3.0 * px), k.to_string(), lay.small, pal::DIM, Align2::CENTER_TOP);
        }
        let axis = pick(&["frequency $k$: how many times the wave fits into the $N$ samples →", "frequency $k$ →"], a.w(), px, lay.small);
        f.text((a.max.x, a.min.y - (lay.small + 6.0) * px), axis, lay.small, pal::DIM, Align2::RIGHT_TOP);
        // what it is, above
        let line1 = pick(
            &[r"$x$'s coordinates in the wave basis: one bar for each wave, the dot product $x \cdot \text{wave}$", r"its coordinates: one bar per wave, $x \cdot \text{wave}$", r"coordinates, $x \cdot \text{wave}$"],
            a.w(),
            px,
            lay.size,
        );
        f.text((a.min.x, a.max.y + 4.0 * px + (lay.small + 5.0) * px), line1, lay.size, pal::TEXT, Align2::LEFT_BOTTOM);
        let long: [(Color, &str); 3] = [(COS, "cosines, the constant and the alternating wave"), (SIN, "sines"), (SIZE, r"lid: each frequency's size $\sqrt{\cos^2 + \sin^2}$")];
        let short: [(Color, &str); 3] = [(COS, r"$\cos$"), (SIN, r"$\sin$"), (SIZE, "lid: size")];
        let fits = |parts: &[(Color, &str)]| parts.iter().map(|(_, t)| width(t, lay.small) + 18.0).sum::<f32>() * px < a.w();
        let parts: &[(Color, &str)] = if fits(&long) { &long } else { &short };
        key_row(f, Vec2::new(a.min.x, a.max.y + 4.0 * px + 0.5 * (lay.small + 2.0) * px), parts, lay.small, px);
        let len2: f64 = self.coords().iter().map(|v| v * v).sum();
        f.text((a.max.x - 6.0 * px, a.max.y - 4.0 * px), format!(r"$\sum \text{{coordinate}}^2 = {}$", num(len2, 2)), lay.small, pal::DIM, Align2::RIGHT_TOP);
    }

    /// The signal rebuilt from the kept waves, over the signal itself.
    fn draw_rebuild(&mut self, f: &mut Frame, lay: &Layout, d: &Drawn, pointer: Option<Vec2>) {
        let (n, a, px) = (self.n, lay.rebuild, lay.px);
        let half = 0.5 * a.h();
        let y_of = |v: f64| a.mid() + ((v / (AMP * 1.1)).clamp(-1.1, 1.1)) as f32 * half;
        inset(f, a.min - Vec2::splat(4.0 * px), a.max + Vec2::splat(4.0 * px));
        f.line((a.min.x, y_of(0.0)), (a.max.x, y_of(0.0)), 1.0, pal::DIM.a(0.8));
        f.text((a.min.x - 6.0 * px, y_of(0.0)), "0", lay.small, pal::DIM, Align2::RIGHT_CENTER);
        let signal = periodic_points(&d.xs, d.s);
        f.path(signal.iter().map(|(q, v)| Vec2::new(lay.at(*q, n), y_of(*v))), 4.0, SIGNAL.a(0.28));
        let rebuilt = periodic_points(&d.rebuilt, d.s);
        f.path(rebuilt.iter().map(|(q, v)| Vec2::new(lay.at(*q, n), y_of(*v))), 2.0, REBUILT);
        let pitch_px = a.w() / n as f32 / px;
        for (i, &v) in d.rebuilt.iter().enumerate() {
            for (q, show) in copies(i as f64 - d.s, n) {
                if show > 0.0 {
                    f.dot((lay.at(q, n), y_of(v)), (0.35 * pitch_px).clamp(3.0, 6.0), REBUILT.a(show));
                }
            }
        }
        // how many are kept: a handle on a track, and the order they are taken in
        let (t0, t1) = lay.track;
        f.line(t0, t1, 2.0, pal::FAINT);
        for k in [0, n] {
            let x = t0.x + (t1.x - t0.x) * k as f32 / n as f32;
            f.line((x, t0.y - 4.0 * px), (x, t0.y + 4.0 * px), 1.0, pal::DIM);
        }
        f.text((t0.x - 8.0 * px, t0.y), "keep", lay.small, pal::DIM, Align2::RIGHT_CENTER);
        let mut knob = Vec2::new(t0.x + (t1.x - t0.x) * self.knob.get(0) as f32, t0.y);
        let held = f.drag2(1, &mut knob, REBUILT);
        if held {
            let share = ((knob.x - t0.x) / (t1.x - t0.x).max(1e-6)).clamp(0.0, 1.0) as f64;
            self.knob.jump(&[share]);
            let keep = (share * n as f64).round() as usize;
            if keep != self.keep {
                self.keep = keep.min(n);
            }
            self.moving = true;
        }
        self.knob_held = held || (self.knob_held && f.handles.dragging());
        let order = match self.order {
            Order::Largest => "largest first",
            Order::Lowest => "lowest first",
        };
        let bmin = on_screen(f, Vec3::new(t1.x + 30.0 * px, t0.y, 0.0)).unwrap_or(Vec2::ZERO) - Vec2::new(0.0, 11.0);
        if hud_button(f, bmin, bmin + Vec2::new(112.0, 22.0), &format!("⇄ {order}"), pointer) {
            self.order = if self.order == Order::Largest { Order::Lowest } else { Order::Largest };
            self.moving = true;
        }
        let share = self.share_kept();
        let which = match self.order {
            Order::Largest => "largest",
            Order::Lowest => "lowest-frequency",
        };
        let long = match share {
            Some(s) => format!("$x$ rebuilt from the {} {which} of its {n} coordinates: {} % of its length$^2$", self.keep, num(s * 100.0, 1)),
            None => format!("$x$ rebuilt from {} of its {n} coordinates: the zero signal has nothing to rebuild", self.keep),
        };
        let short = match share {
            Some(s) => format!("rebuilt from {}: {} % of length$^2$", waves(self.keep), num(s * 100.0, 1)),
            None => format!("rebuilt from {}", waves(self.keep)),
        };
        f.text((a.min.x, t0.y + 13.0 * px), pick(&[&long, &short], a.w(), px, lay.size), lay.size, REBUILT, Align2::LEFT_BOTTOM);
    }

    /// The two heat maps: without a blur, the waves as the rows of W and their dot
    /// products W Wᵀ; with one, the blur's matrix and the same blur in the wave basis.
    fn draw_heat(&mut self, f: &mut Frame, lay: &Layout) {
        let Some([first, second]) = lay.heat else { return };
        let (n, px) = (self.n, lay.px);
        self.paint(f.gfx);
        let blur = reach(n, self.width) > 0;
        if let Some(image) = &mut self.heat {
            if image.size() == [(2 * n + 1) as u32, n as u32] {
                f.image_part(image, [0, 0, n as u32, n as u32], first.min, first.max, 0.0);
                f.image_part(image, [n as u32 + 1, 0, n as u32, n as u32], second.min, second.max, 0.0);
            }
        }
        for area in [first, second] {
            frame_rect(f, area.min, area.max, 1.0, pal::DIM.a(0.7));
        }
        // the picked wave's row: in W, the wave itself; in W Wᵀ, its dot products with
        // every wave; in W C Wᵀ, what the blur makes of it
        let row = |area: Area, j: f32| {
            let cell = area.h() / n as f32;
            (Vec2::new(area.min.x - 2.0 * px, area.max.y - (j + 1.0) * cell), Vec2::new(area.max.x + 2.0 * px, area.max.y - j * cell))
        };
        // (gliding from one row to the next with the picked wave)
        let j = self.wave.shown.iter().enumerate().map(|(j, a)| j as f64 * a).sum::<f64>().clamp(0.0, (n - 1) as f64) as f32;
        for area in if blur { vec![second] } else { vec![first, second] } {
            let (lo, hi) = row(area, j);
            frame_rect(f, lo, hi, 1.4, UP);
        }
        let worst = format!("largest off it: {}", tiny(self.off_diagonal()));
        let (one, two): ([&str; 2], [String; 2]) = if blur {
            let w = 2 * reach(n, self.width) + 1;
            (
                [r"$C$: the blur, one row per sample", r"$W C W^\top$: the blur in the wave basis"],
                [format!("each bright cell: $1/{w}$"), "diagonal: each wave times its factor".to_string()],
            )
        } else {
            ([r"$W$: one wave per row", r"$W W^\top$: every wave $\cdot$ every wave"], [r"$Wx$ is $x$'s coordinates".to_string(), format!("1 on the diagonal, 0 off it ({worst})")])
        };
        let middle: [&str; 2] = if blur { [r"$C$: the blur", r"$W C W^\top$: in the wave basis"] } else { [r"$W$: the waves", r"$W W^\top$: wave $\cdot$ wave"] };
        let short: [&str; 2] = if blur { [r"$C$, the blur", r"$W C W^\top$"] } else { [r"$W$: the waves", r"$W W^\top$"] };
        for (k, area) in [first, second].into_iter().enumerate() {
            // (as much as the map is wide, plus the gap beside it on a wide canvas)
            // (the words' widths are guessed, and a guess a fifth short would run into the next title)
            let room = (area.w() + if lay.wide { 30.0 * px } else { 0.0 }) / 1.2;
            if lay.wide {
                f.text((area.min.x, area.max.y + 4.0 * px + (lay.small + 5.0) * px), pick(&[one[k], middle[k], short[k]], room, px, lay.small), lay.small, pal::TEXT, Align2::LEFT_BOTTOM);
                f.text((area.min.x, area.max.y + 4.0 * px), pick(&[&two[k], ""], room, px, lay.small - 1.0), lay.small - 1.0, pal::DIM, Align2::LEFT_BOTTOM);
            } else {
                f.text((area.min.x, area.max.y + 4.0 * px), pick(&[short[k], ""], room + 12.0 * px, px, lay.small), lay.small, pal::TEXT, Align2::LEFT_BOTTOM);
            }
        }
    }

    /// Why the picture is as it is, in a few sentences.
    fn why(&self) -> String {
        let r = reach(self.n, self.width);
        if r == 0 {
            return r"Why two different waves have dot product 0: their product is two other waves, of the sum and of the difference of their frequencies ($\cos a \cos b = \frac{1}{2} \cos(a - b) + \frac{1}{2} \cos(a + b)$), and a wave that goes a whole number of times round the $N$ samples adds up to 0. A wave times itself is $\frac{1}{2} + \frac{1}{2} \cos 2a$, which leaves only the $\frac{1}{2}$: $N/2$ in all, which is why each wave is scaled by $\sqrt{2/N}$, to length 1.".to_string();
        }
        let w = 2 * r + 1;
        let (around, factor) = match r {
            1 => (r"\cos(a - \theta) + \cos a + \cos(a + \theta)".to_string(), r"1 + 2\cos\theta".to_string()),
            2 => (r"\cos(a - 2\theta) + \cos(a - \theta) + \cos a + \cos(a + \theta) + \cos(a + 2\theta)".to_string(), r"1 + 2\cos\theta + 2\cos 2\theta".to_string()),
            _ => (format!(r"\cos(a - {r}\theta) + \cdots + \cos a + \cdots + \cos(a + {r}\theta)"), format!(r"1 + 2\cos\theta + \cdots + 2\cos {r}\theta")),
        };
        // (the two sides of the equation are two formulas, so that the note on the
        // picture, which keeps each formula on one line, can break the line between them)
        format!(r"Why the blur is diagonal here: averaging a wave over its neighbours gives back the same wave, smaller. With $\theta = 2\pi k/N$, the wave's step from one sample to the next, $({around}) / {w}$ $= \cos a \times ({factor}) / {w}$, and $\sin a$ is multiplied by the same number. So each frequency is only multiplied by that number: its orange dot.")
    }

    /// The largest dot product of two different waves (rounding).
    fn off_diagonal(&self) -> f64 {
        let n = self.n;
        let g = gram(&self.basis);
        (0..n * n).filter(|e| e / n != e % n).map(|e| g[e].abs()).fold(0.0, f64::max)
    }
}

/// "1 wave", "3 waves".
fn waves(k: usize) -> String {
    if k == 1 {
        "1 wave".to_string()
    } else {
        format!("{k} waves")
    }
}

/// A small number written as a power of ten, as a formula: $3 \times 10^{-16}$.
fn tiny(v: f64) -> String {
    if v.is_nan() || v <= 0.0 {
        return "$0$".to_string();
    }
    let e = v.log10().floor();
    let mantissa = (v / 10f64.powf(e)).round() as i64;
    let (mantissa, e) = if mantissa >= 10 { (1, e + 1.0) } else { (mantissa, e) };
    format!(r"${mantissa} \times 10^{{{}}}$", e as i64)
}

impl Scene for Dft {
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
        Some("white dots: drag a sample    bars: tap or slide along to pick a wave    green dot: how many waves to keep    drag elsewhere: move    scroll: zoom")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A signal of $N$ numbers is a vector in $N$ dimensions, one axis per sample. The waves that fit a whole number of times into the $N$ samples are $N$ other axes, all at right angles to one another. Measured along them, by a dot product with each, the signal has $N$ new coordinates: its discrete Fourier transform. Adding the waves back up, each times its coordinate, gives the signal again. Drag the white dots to change the signal; press on a bar to pick a wave.");

        ui::section(ui, "Things to try");
        for (label, look_for, setup) in TRY {
            if ui::experiment(ui, label, look_for) {
                self.apply(setup);
            }
        }

        ui::section(ui, "The signal");
        let names: Vec<&str> = Preset::ALL.iter().map(|p| p.name()).collect();
        if let Some(i) = ui::buttons(ui, &names) {
            self.take(Preset::ALL[i]);
        }
        let mut n = self.n;
        if ui::slider_int(ui, "samples $N$", &mut n, N_MIN..=N_MAX) {
            self.resize(n);
        }
        if ui.button("shift by one ▸").clicked() {
            self.shift();
        }
        ui::note(ui, r"Moves every sample one place to the right; the last comes round to the front. Every $\cos$ and $\sin$ bar changes, but no grey lid moves: a shift turns each frequency's pair of coordinates through $k/N$ of a turn, like the hand of a clock, which keeps its length. Which single coordinates are the largest can change, though, and with them what the green keeps: the opening signal shifted once has 78 % of its length$^2$ in its 3 largest coordinates, not 96 %.");

        ui::section(ui, "The waves");
        let mut picked = self.picked;
        if ui::slider_int(ui, "picked wave", &mut picked, 0..=self.n - 1) {
            self.picked = picked.min(self.n - 1);
            self.moving = true;
        }
        ui::note(ui, &format!("Now: {}. Or press on its bar.", Wave::of(self.picked, self.n).name()));
        let mut keep = self.keep;
        if ui::slider_int(ui, "waves kept", &mut keep, 0..=self.n) {
            self.keep = keep.min(self.n);
            self.moving = true;
        }
        let mut order = (self.order == Order::Lowest) as usize;
        if ui::choice(ui, &["largest first", "lowest frequencies first"], &mut order) {
            self.order = if order == 1 { Order::Lowest } else { Order::Largest };
            self.moving = true;
        }
        ui::note(ui, "How many of the waves the green signal is rebuilt from (or drag the green dot), and which: those with the largest coordinates, or the slowest.");
        let mut w = WIDTHS.iter().position(|&w| w == self.width).unwrap_or(0);
        if ui::choice(ui, &["no blur", "blur 3", "blur 5", "blur 9", "blur 15"], &mut w) {
            self.width = WIDTHS[w.min(WIDTHS.len() - 1)];
            self.moving = true;
        }
        ui::note(ui, r"A blur replaces each sample by the average of itself and its neighbours (3, 5, … samples in all, going round the end). Any operation that treats every position alike keeps each frequency to itself: in the wave basis it can only scale and turn each frequency's $(\cos, \sin)$ pair, as the shift turns it. One that is the same both ways round, like this blur, only scales: its matrix there is diagonal.");

        ui::section(ui, "Reading the picture");
        let white = format!(r"white: the signal $x$, its samples $x_0, \ldots, x_{{{}}}$", self.n - 1);
        ui::key(
            ui,
            &[
                (SIGNAL, &white),
                (COS, "cyan: the cosine waves, with the constant and the alternating wave, and their coordinates (the bars)"),
                (SIN, "pink: the sine waves and their coordinates"),
                (SIZE, r"grey lid over each pair of bars: the size of that frequency, $\sqrt{\cos^2 + \sin^2}$"),
                (UP, r"yellow: a product $x_n w_n$ above 0, and the dot product $x \cdot \text{wave}$ (the ringed bar)"),
                (DOWN, "purple: a product below 0"),
                (REBUILT, "green: $x$ rebuilt from the kept waves alone (the faded bars are left out)"),
                (BLUR, "orange: $x$ blurred, its coordinates, and the number the blur multiplies each frequency by"),
                (HEAT.table()[heat_index(0.8)], "heat maps: red to yellow above 0, dark at 0, blue below"),
            ],
        );
        ui::note(ui, &self.why());

        ui::section(ui, "Words");
        ui::term(ui, "Sample:", "one of the $N$ numbers that make up the signal: its value at one place.");
        ui::term(ui, "Dot product:", r"$x \cdot w = x_0 w_0 + x_1 w_1 + \cdots$, the products sample by sample, added up. Length: $\sqrt{x \cdot x}$.");
        ui::term(ui, r"Wave $\cos k$, $\sin k$:", r"the $N$ numbers $\cos(2\pi k n/N)$ or $\sin(2\pi k n/N)$, $n = 0, \ldots, N - 1$, scaled to length 1. They fit $k$ whole times into the $N$ samples: $k$ is the frequency.");
        ui::term(ui, "The basis:", r"for even $N$, the constant, $\cos k$ and $\sin k$ for $k = 1$ to $N/2 - 1$, and the alternating wave $+1, -1, +1, \ldots$ ($k = N/2$; its sine is 0 at every sample). $N$ waves in all. For odd $N$, no alternating wave.");
        ui::term(ui, "Orthonormal:", "at right angles to each other (dot product 0), and each of length 1.");
        ui::term(ui, "$W$:", r"the matrix with the waves as its rows. $Wx$ is $x$'s coordinates, and $W^\top$ takes coordinates back to samples, since $W W^\top = I$. Going over to the wave basis turns $N$-space rigidly (a rotation, perhaps with a mirror): every length and angle is kept, so $\sum x_n^2 = \sum \text{coordinate}^2$.");
        ui::term(ui, "Coordinate:", "how far along one wave the signal reaches: its dot product with that wave, since the waves are orthonormal.");
        ui::term(ui, "Discrete Fourier transform:", r"the $N$ coordinates of the signal in the wave basis. It is usually written with complex numbers, $X_k = \sum_n x_n e^{-2\pi i k n/N}$; for a real signal the real part of $X_k$ is $\sqrt{N/2}$ times the coordinate of $\cos k$, and the imaginary part $-\sqrt{N/2}$ times that of $\sin k$.");
        ui::term(ui, "Circulant:", "a matrix each of whose rows is the one above shifted along by one: an operation that treats every position alike.");

        ui::section(ui, "Numbers");
        let c = self.coords();
        let len2: f64 = self.x.iter().map(|v| v * v).sum();
        let coord2: f64 = c.iter().map(|v| v * v).sum();
        ui::readout(ui, r"length$^2$ of $x$, $\sum x_n^2$", num(len2, 4));
        ui::readout(ui, r"$\sum \text{coordinate}^2$", num(coord2, 4));
        let wave = Wave::of(self.picked, self.n);
        ui::readout_colored(ui, &format!("coordinate of {}", wave.name()), num(c.get(self.picked).copied().unwrap_or(0.0), 4), UP);
        match self.share_kept() {
            Some(s) => ui::readout_colored(ui, "length$^2$ kept by the green", format!("{} %", num(s * 100.0, 2)), REBUILT),
            None => ui::readout_colored(ui, "length$^2$ kept by the green", "none to keep", REBUILT),
        }
        let error = length(&self.x.iter().zip(self.rebuilt()).map(|(x, r)| x - r).collect::<Vec<_>>());
        ui::readout_colored(ui, "length of what the green misses", num(error, 4), REBUILT);
        ui::readout(ui, "largest dot product of two different waves", tiny(self.off_diagonal()));
        ui::see_also(
            ui,
            &[
                ("basis", "change of basis: coordinates along axes at right angles are dot products"),
                ("fourier", "Fourier series: a closed curve as a sum of turning circles"),
                ("lowrank", "another basis that compresses: the SVD of a picture"),
                ("grokking", "a network that learns to add with these waves"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone:
        // a column of panels a few pixels wide would only be a heap of letters)
        if f.view_size().x < 160.0 {
            self.advance(f.real_dt as f64);
            return;
        }
        let (w, ppu) = (stage_width(f), pixels_per_unit(f));
        let lay = Layout::of(w, ppu, f.top_inset);
        let pointer = f.ptr.pos.map(|p| p / f.px);
        let n = self.n;

        // picking a bar: press on the bars, and slide along them
        let mouse = f.mouse();
        let over = mouse.is_some_and(|m| lay.coords.contains(m, 6.0 * lay.px));
        if over || self.scrubbing {
            f.handles.capture();
            f.handles.claim_clicks();
        }
        if f.ptr.pressed && over && !f.handles.on_handle() {
            self.scrubbing = true;
        }
        if !f.ptr.down {
            self.scrubbing = false;
        }
        if self.scrubbing || (f.ptr.clicked && over) {
            if let Some(m) = mouse {
                let at = ((m.x - lay.coords.min.x) / lay.coords.w() * slots(n) as f32) as f64;
                let nearest = (0..n).min_by(|&a, &b| (slot(a, n) - at).abs().total_cmp(&(slot(b, n) - at).abs())).unwrap_or(0);
                if nearest != self.picked {
                    self.picked = nearest;
                    self.moving = true;
                }
            }
        }
        // (the buttons over the picture are for clicking, not for moving the view)
        if pointer.is_some_and(|p| {
            let near = |q: Vec3| on_screen(f, q).is_some_and(|s| (s - p).abs().cmple(Vec2::new(130.0, 20.0)).all());
            near(Vec3::new(lay.signal.max.x, lay.signal.max.y, 0.0)) || near(Vec3::new(lay.track.1.x + 60.0 * lay.px, lay.track.1.y, 0.0))
        }) {
            f.handles.capture();
            f.handles.claim_clicks();
        }

        self.advance(f.real_dt as f64);
        let handles = self.drag_samples(f, &lay);
        let d = self.drawn();
        self.draw_signal(f, &lay, &d, handles, pointer);
        self.draw_wave(f, &lay, &d);
        self.draw_products(f, &lay, &d);
        self.draw_coords(f, &lay, &d);
        self.draw_rebuild(f, &lay, &d, pointer);
        self.draw_heat(f, &lay);
        if let Some((at, width)) = lay.note {
            let why = self.why();
            self.note.draw(f, at, width, &why, lay.small, pal::DIM);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scene set up as an experiment, at rest.
    fn run(k: usize) -> Dft {
        let mut s = Dft::new();
        s.apply(TRY[k].2);
        s.settle();
        s
    }

    /// The running sums of the picked wave's products with the signal.
    fn stack(s: &Dft) -> Vec<f64> {
        running(&s.x, s.basis.row(s.picked))
    }

    fn coordinate(s: &Dft) -> f64 {
        s.coords()[s.picked]
    }

    /// The largest coordinate other than those listed.
    fn largest_but(c: &[f64], but: &[usize]) -> f64 {
        c.iter().enumerate().filter(|(j, _)| !but.contains(j)).map(|(_, v)| v.abs()).fold(0.0, f64::max)
    }

    /// The size of each frequency, √(cos² + sin²).
    fn sizes(c: &[f64], n: usize) -> Vec<f64> {
        (0..frequencies(n)).map(|k| members(k, n).map(|j| c[j] * c[j]).sum::<f64>().sqrt()).collect()
    }

    fn signals(n: usize) -> Vec<Vec<f64>> {
        let mut rng = Rng::new(n as u64);
        let mut all: Vec<Vec<f64>> = Preset::ALL.iter().map(|p| p.samples(n)).collect();
        all.extend((0..4).map(|_| (0..n).map(|_| rng.range(-2.0, 2.0) as f64).collect()));
        all
    }

    /// The waves are orthonormal for every N: the basis is exactly what the panel says
    /// it is (the constant, a cosine and a sine for each frequency 1 to N/2 − 1, and for
    /// even N the alternating wave), and a wave times itself adds up to N/2 before it is
    /// scaled (the note under the heat maps).
    #[test]
    fn the_waves_are_an_orthonormal_basis() {
        for n in N_MIN..=N_MAX {
            let b = Basis::new(n);
            let g = gram(&b);
            for e in 0..n * n {
                let want = if e / n == e % n { 1.0 } else { 0.0 };
                assert!((g[e] - want).abs() < 1e-13, "N {n}: {} · {} = {}", Wave::of(e / n, n).name(), Wave::of(e % n, n).name(), g[e]);
            }
            for j in 0..n {
                let w = Wave::of(j, n);
                assert_eq!(w.index(n), j);
                assert!(w.fits(n) && w.k <= n / 2);
                assert_eq!(w.kind == Kind::Alternating, n % 2 == 0 && j == n - 1, "N {n}, wave {j}");
                if w.kind == Kind::Alternating {
                    assert!((0..n).all(|i| (b.row(j)[i] * (n as f64).sqrt() - if i % 2 == 0 { 1.0 } else { -1.0 }).abs() < 1e-12));
                }
                // (the smooth curve passes through the samples)
                assert!((0..n).all(|i| (w.at(n, i as f64) - b.row(j)[i]).abs() < 1e-12));
            }
            assert_eq!((0..frequencies(n)).map(|k| members(k, n).len()).sum::<usize>(), n);
            for k in 1..n.div_ceil(2) {
                let squares: f64 = (0..n).map(|i| (2.0 * PI64 * (k * i) as f64 / n as f64).cos().powi(2)).sum();
                assert!((squares - n as f64 / 2.0).abs() < 1e-10, "N {n}, k {k}: {squares}");
            }
        }
    }

    /// The coordinates are the dot products, the last running sum is the coordinate,
    /// all the waves rebuild the signal exactly, and the length is the same in both bases.
    #[test]
    fn lengths_are_kept_and_the_waves_rebuild_the_signal() {
        for n in [N_MIN, 9, 16, 31, N_START, 47, N_MAX] {
            let b = Basis::new(n);
            for x in signals(n) {
                let c = b.forward(&x);
                let back = b.back(&c);
                assert!(x.iter().zip(&back).all(|(p, q)| (p - q).abs() < 1e-12), "N {n}");
                assert!((length(&x) - length(&c)).abs() < 1e-12 * length(&x).max(1.0));
                for j in 0..n {
                    let sums = running(&x, b.row(j));
                    assert!((sums[n - 1] - c[j]).abs() < 1e-12);
                    // (and no running sum is longer than the signal: the panel of bars always has room)
                    assert!(sums.iter().all(|v| v.abs() <= length(&x) + 1e-12));
                }
            }
        }
    }

    /// The opening picture: four waves, exactly four coordinates not 0, the largest
    /// picked; three kept hold 96.1 % of the length².
    #[test]
    fn the_opening_picture_is_four_waves() {
        let s = Dft::new();
        assert_eq!((s.n, s.picked, s.keep, s.order, s.width), (32, 3, 3, Order::Largest, 1));
        assert!(s.x.iter().all(|v| v.abs() < 1.95), "no sample is clipped");
        let c = s.coords();
        let four = [3, 6, 9, 14];
        assert_eq!(four.map(|j| Wave::of(j, 32).name()), [r"$\cos 2$", r"$\sin 3$", r"$\cos 5$", r"$\sin 7$"]);
        assert!(largest_but(&c, &four) < 1e-13);
        assert!(four.iter().all(|&j| c[j].abs() > 0.9) && largest_but(&c, &[]) == c[3].abs());
        assert_eq!(num(s.share_kept().unwrap() * 100.0, 1), "96.1");
    }

    /// "One wave, one coordinate": every product positive, the stack climbs to 6.00, the
    /// signal's length; every other coordinate 0; one wave rebuilds it exactly.
    #[test]
    fn one_wave_one_coordinate() {
        let s = run(0);
        assert_eq!(Wave::of(s.picked, s.n).name(), r"$\cos 3$");
        let w = s.basis.row(s.picked);
        assert!(s.x.iter().zip(w).all(|(x, w)| x * w > 0.0));
        let c = s.coords();
        assert_eq!(num(c[s.picked], 2), "6.00");
        assert!((length(&s.x) - 6.0).abs() < 1e-12 && (s.x.iter().fold(0.0f64, |m, v| m.max(v.abs())) - 1.5).abs() < 1e-12);
        assert!(largest_but(&c, &[s.picked]) < 1e-13);
        assert_eq!(s.share_kept().map(|v| num(v * 100.0, 1)), Some("100.0".to_string()));
        assert!(s.x.iter().zip(s.rebuilt()).all(|(x, r)| (x - r).abs() < 1e-12));
    }

    /// "Two different waves cancel": against cos 4 the stack climbs to 1.17, is back at 0
    /// at sample 16, sinks to −0.80 and ends at 0; the products' signs are the waves' signs.
    #[test]
    fn two_different_waves_cancel() {
        let s = run(1);
        assert_eq!(Wave::of(s.picked, s.n).name(), r"$\cos 4$");
        assert_eq!(s.x, run(0).x);
        let sums = stack(&s);
        let top = sums.iter().fold(f64::MIN, |m, v| m.max(*v));
        let bottom = sums.iter().fold(f64::MAX, |m, v| m.min(*v));
        assert_eq!((num(top, 2), num(bottom, 2)), ("1.17".to_string(), "−0.80".to_string()));
        assert!(sums[16].abs() < 1e-12 && sums[31].abs() < 1e-12, "{} {}", sums[16], sums[31]);
        assert!(sums[..16].iter().all(|v| *v > 0.0) && sums[17..31].iter().all(|v| *v < 0.0));
        assert_eq!(num(coordinate(&s), 2), "0.00");
        assert_eq!(UP, pal::YELLOW);
        assert_eq!(DOWN, pal::PURPLE);
    }

    /// "A spike needs every wave": each coordinate is 2 × the wave at sample 10; every
    /// frequency 1 to 15 holds 6.25 % of the length², the ends half that; the 8 largest
    /// hold 45 %, the green reaches 0.91 at the spike and strays as far as 0.60 elsewhere.
    #[test]
    fn a_spike_needs_every_wave() {
        let s = run(2);
        let n = s.n;
        assert_eq!(Preset::spike_at(n), 10);
        assert!(s.x.iter().enumerate().all(|(i, v)| *v == if i == 10 { 2.0 } else { 0.0 }));
        let c = s.coords();
        for j in 0..n {
            assert!((c[j] - 2.0 * s.basis.row(j)[10]).abs() < 1e-12);
            let products = s.x.iter().zip(s.basis.row(j)).filter(|(x, w)| *x * *w != 0.0).count();
            assert!(products <= 1);
        }
        let len2 = 4.0;
        for (k, size) in sizes(&c, n).into_iter().enumerate() {
            let share = size * size / len2;
            let want = if k == 0 || 2 * k == n { 0.03125 } else { 0.0625 };
            assert!((share - want).abs() < 1e-12, "frequency {k}: {share}");
        }
        let share = s.share_kept().unwrap();
        assert_eq!(num(share * 100.0, 0), "45");
        let green = s.rebuilt();
        assert_eq!(num(green[10], 2), "0.91");
        let stray = green.iter().enumerate().filter(|(i, _)| *i != 10).map(|(_, v)| v.abs()).fold(0.0, f64::max);
        assert_eq!(num(stray, 2), "0.60");
    }

    /// "A smooth bump needs few": from frequency 4 up every coordinate is below 0.2, from 7
    /// up below 0.001; the 7 slowest waves (the constant and frequencies 1 to 3) hold
    /// 99.9 % of the length², and the green is within 0.05 of the white everywhere.
    #[test]
    fn a_smooth_bump_needs_few() {
        let s = run(3);
        let n = s.n;
        let c = s.coords();
        assert!((7..n).all(|j| Wave::of(j, n).k >= 4) && (0..7).all(|j| Wave::of(j, n).k <= 3));
        assert!((0..n).filter(|&j| Wave::of(j, n).k >= 4).all(|j| c[j].abs() < 0.2));
        assert!((0..n).filter(|&j| Wave::of(j, n).k >= 7).all(|j| c[j].abs() < 1e-3));
        assert!((0..n).filter(|&j| Wave::of(j, n).k == 6).any(|j| c[j].abs() > 1e-3));
        assert_eq!(s.kept_target(&c), (0..n).map(|j| if j < 7 { 1.0 } else { 0.0 }).collect::<Vec<_>>());
        assert_eq!(num(s.share_kept().unwrap() * 100.0, 1), "99.9");
        let worst = s.x.iter().zip(s.rebuilt()).map(|(x, r)| (x - r).abs()).fold(0.0, f64::max);
        assert!(worst < 0.05 && worst > 0.04, "{worst}");
    }

    /// "A blur multiplies each wave": the blur's matrix is circulant, it is diagonal in
    /// the wave basis with the factors on the diagonal, and the orange bars are the
    /// coordinates of the blurred signal. 1 for the constant; a quarter or less from
    /// frequency 6 up; below 0 from 7 to 12 only; the ringed sin 9 flips.
    #[test]
    fn a_blur_multiplies_each_wave() {
        let s = run(4);
        let n = s.n;
        assert_eq!((s.width, reach(n, s.width)), (5, 2));
        let c = blur_matrix(n, s.width);
        for i in 1..n {
            for j in 0..n {
                assert_eq!(c[i * n + j], c[(i - 1) * n + (j + n - 1) % n], "each row the one above, shifted");
            }
        }
        let d = s.basis.in_waves(&c);
        for j in 0..n {
            for l in 0..n {
                let want = if j == l { factor(n, s.width, Wave::of(j, n).k) } else { 0.0 };
                assert!((d[j * n + l] - want).abs() < 1e-13, "W C Wᵀ ({j}, {l}) = {}", d[j * n + l]);
            }
        }
        let after = s.basis.forward(&blurred(&s.x, s.width));
        let before = s.coords();
        for j in 0..n {
            assert!((after[j] - before[j] * factor(n, s.width, Wave::of(j, n).k)).abs() < 1e-12);
        }
        let f: Vec<f64> = (0..frequencies(n)).map(|k| factor(n, s.width, k)).collect();
        assert_eq!(f[0], 1.0);
        assert!(f[6..].iter().all(|v| v.abs() <= 0.25) && f[5] > 0.25);
        assert_eq!((0..frequencies(n)).filter(|&k| f[k] < 0.0).collect::<Vec<_>>(), (7..=12).collect::<Vec<_>>());
        assert_eq!(Wave::of(s.picked, n).name(), r"$\sin 9$");
        assert!(before[s.picked] < 0.0 && after[s.picked] > 0.0);
        assert_eq!((num(before[s.picked], 2), num(after[s.picked], 2)), ("−0.37".to_string(), "0.09".to_string()));
    }

    /// Every blur, at every N: diagonal in the wave basis, the same number for a
    /// frequency's cosine and sine, and the factor the note's formula gives.
    #[test]
    fn every_blur_is_diagonal_in_the_wave_basis() {
        for n in [N_MIN, 9, 16, 21, N_MAX] {
            let b = Basis::new(n);
            for width in WIDTHS {
                let d = b.in_waves(&blur_matrix(n, width));
                let r = reach(n, width);
                for j in 0..n {
                    for l in 0..n {
                        let k = Wave::of(j, n).k;
                        let note = (1.0 + 2.0 * (1..=r).map(|m| (2.0 * PI64 * (k * m) as f64 / n as f64).cos()).sum::<f64>()) / (2 * r + 1) as f64;
                        let want = if j == l { note } else { 0.0 };
                        assert!((d[j * n + l] - want).abs() < 1e-12, "N {n}, width {width}");
                    }
                }
            }
        }
    }

    /// The note under the blur: any circulant matrix, symmetric or not, keeps each
    /// frequency to itself, scaling and turning its (cos, sin) pair; a symmetric one only
    /// scales. And W keeps lengths and angles (Wᵀ W = I as well as W Wᵀ = I).
    #[test]
    fn a_circulant_keeps_each_frequency_to_itself() {
        let mut rng = Rng::new(3);
        for n in [N_MIN, 9, 16, 21] {
            let b = Basis::new(n);
            for symmetric in [false, true] {
                let mut h: Vec<f64> = (0..n).map(|_| rng.normal() as f64).collect();
                if symmetric {
                    for m in 1..n {
                        h[m] = h[n - m];
                    }
                }
                let c: Vec<f64> = (0..n * n).map(|e| h[(e % n + n - e / n) % n]).collect();
                let d = b.in_waves(&c);
                for j in 0..n {
                    for l in 0..n {
                        let (wj, wl) = (Wave::of(j, n), Wave::of(l, n));
                        if wj.k != wl.k {
                            assert!(d[j * n + l].abs() < 1e-12, "N {n}: {} and {}", wj.name(), wl.name());
                        } else if j != l {
                            // a scaled turn: equal on the diagonal, opposite off it, and 0 if symmetric
                            let (cc, ss, cs, sc) = (d[(j.min(l)) * (n + 1)], d[(j.max(l)) * (n + 1)], d[j.min(l) * n + j.max(l)], d[j.max(l) * n + j.min(l)]);
                            assert!((cc - ss).abs() < 1e-12 && (cs + sc).abs() < 1e-12, "N {n}, k {}", wj.k);
                            assert!(!symmetric || cs.abs() < 1e-12);
                        }
                    }
                }
            }
            for i in 0..n {
                for m in 0..n {
                    let dot: f64 = (0..n).map(|j| b.row(j)[i] * b.row(j)[m]).sum();
                    assert!((dot - if i == m { 1.0 } else { 0.0 }).abs() < 1e-13);
                }
            }
        }
    }

    /// A shift turns each frequency's pair of coordinates through k/N of a turn (the
    /// alternating wave flips), so no frequency's size changes; and the picture starts
    /// from where it was: the coordinates drawn right after it are the ones before.
    #[test]
    fn a_shift_turns_each_frequency_and_keeps_its_size() {
        for n in [N_MIN, 9, N_START, 33, N_MAX] {
            for x in signals(n) {
                let mut s = Dft::new();
                s.resize(n);
                s.x = x.clone();
                s.settle();
                let before = s.coords();
                s.shift();
                let after = s.coords();
                assert_eq!(s.x[1..], x[..n - 1]);
                assert_eq!(s.x[0], x[n - 1]);
                for (p, q) in sizes(&before, n).iter().zip(sizes(&after, n)) {
                    assert!((p - q).abs() < 1e-12, "N {n}");
                }
                let back = turned(&after, n, 1.0);
                assert!(back.iter().zip(&before).all(|(p, q)| (p - q).abs() < 1e-12));
                let d = s.drawn();
                assert!(d.c.iter().zip(&before).all(|(p, q)| (p - q).abs() < 1e-12), "the coordinates drawn jumped");
                // part of the way, every size is kept but the alternating wave's
                for t in [0.25, 0.5, 0.75] {
                    let mid = sizes(&turned(&after, n, t), n);
                    for (k, (p, q)) in mid.iter().zip(sizes(&after, n)).enumerate() {
                        assert!(2 * k == n || (p - q).abs() < 1e-12);
                    }
                }
                for _ in 0..200 {
                    s.advance(1.0 / 60.0);
                }
                assert!(!s.moving && s.slide.get(0) == 0.0 && s.drawn().c.iter().zip(&after).all(|(p, q)| (p - q).abs() < 1e-12));
            }
        }
    }

    /// The note under the shift button: the opening signal, shifted once, has 78 % of its
    /// length² in its 3 largest coordinates, not 96 %.
    #[test]
    fn a_shift_can_change_which_coordinates_are_largest() {
        let mut s = Dft::new();
        assert_eq!(num(s.share_kept().unwrap() * 100.0, 0), "96");
        s.shift();
        assert_eq!(num(s.share_kept().unwrap() * 100.0, 0), "78");
    }

    /// A curve through a sliding row: the same points in the same order all through a
    /// slide, each moving a little at a time, inside the row, through the samples at rest,
    /// and at a slide of one where the row was before it was shifted.
    #[test]
    fn a_sliding_curve_moves_evenly() {
        for n in [N_MIN, 9, N_START] {
            let v: Vec<f64> = (0..n).map(|i| ((i * 7) % 5) as f64 - 2.0).collect();
            let rest = periodic_points(&v, 0.0);
            assert_eq!(rest.len(), n + 3);
            for (i, &x) in v.iter().enumerate() {
                assert!(rest.iter().any(|&(q, y)| q == i as f64 && y == x));
            }
            let mut shifted = v.clone();
            shifted.rotate_right(1);
            let start = periodic_points(&shifted, 1.0);
            for &(q, y) in &start {
                let back = periodic_points(&v, 0.0);
                assert!(back.iter().any(|&(p, x)| (p - q).abs() < 1e-12 && (x - y).abs() < 1e-12), "{q} {y}");
            }
            let mut before = start;
            for step in 1..=200 {
                let now = periodic_points(&shifted, 1.0 - step as f64 / 200.0);
                assert_eq!(now.len(), before.len());
                for (p, q) in before.iter().zip(&now) {
                    assert!((p.0 - q.0).abs() <= 0.0051 && (p.1 - q.1).abs() < 0.05, "step {step}: {p:?} to {q:?}");
                    assert!(q.0 >= -0.5 && q.0 <= n as f64 - 0.5);
                }
                before = now;
            }
        }
    }

    /// After an experiment, a pick, a shift or a new N, everything drawn arrives where it
    /// is going within a second, exactly, and then stops, at any frame rate.
    #[test]
    fn glides_arrive_and_stop() {
        for hz in [30.0, 60.0, 144.0] {
            let mut s = Dft::new();
            for step in 0..9 {
                match step {
                    0..=4 => s.apply(TRY[step].2),
                    5 => s.picked = 20,
                    6 => s.shift(),
                    7 => s.width = 15,
                    _ => s.keep = 0,
                }
                let mut frames = 0;
                loop {
                    s.advance(1.0 / hz);
                    frames += 1;
                    if !s.moving {
                        break;
                    }
                    assert!(frames as f64 / hz < 1.0, "step {step} still moving after {frames} frames at {hz} Hz");
                }
                let t = s.targets();
                assert_eq!(s.shown.shown, s.x);
                assert_eq!((s.wave.shown.clone(), s.kept.shown.clone(), s.factor.shown.clone()), (t.wave, t.kept, t.factor));
                assert_eq!((s.scale.get(0), s.range.shown.clone(), s.ring.get(0)), (t.scale, t.range.to_vec(), t.ring));
            }
        }
    }

    /// The zero signal, N at its ends and odd, nothing kept or everything, every blur
    /// and every pick: every number drawn or written is finite, and the zero signal is
    /// said to have nothing to keep.
    #[test]
    fn degenerate_settings_leave_everything_finite() {
        for n in [N_MIN, 9, N_START, 63, N_MAX] {
            for preset in Preset::ALL {
                for (keep, width) in [(0, 1), (n, 15), (1, 3), (n / 2, 9)] {
                    let mut s = Dft::new();
                    s.resize(n);
                    s.take(preset);
                    (s.keep, s.width, s.picked) = (keep, width, n - 1);
                    s.shift();
                    s.advance(0.02);
                    let d = s.drawn();
                    for v in [&d.xs, &d.c, &d.xr, &d.wave, &d.kept, &d.rebuilt, &d.blurred, &d.factor, &d.by_wave] {
                        assert!(v.iter().all(|x| x.is_finite()));
                    }
                    assert!(d.scale.is_finite() && d.scale > 0.0 && d.range.iter().all(|r| r.is_finite() && *r > 0.0) && d.ring.is_finite());
                    assert!(periodic_points(&d.xs, d.s).iter().all(|(q, v)| q.is_finite() && v.is_finite()));
                    match preset {
                        Preset::Zero => assert_eq!(s.share_kept(), None),
                        _ => assert!(s.share_kept().is_some_and(|v| (0.0..=1.0 + 1e-12).contains(&v))),
                    }
                    assert!(!s.why().is_empty() && s.off_diagonal() < 1e-13);
                }
            }
        }
        // a hand-drawn signal is read off at a new N, not lost
        let mut s = Dft::new();
        s.x[5] = 1.7;
        s.preset = None;
        s.resize(64);
        assert_eq!(s.x.len(), 64);
        assert!((s.x[10] - 1.7).abs() < 1e-12);
        assert_eq!(Wave::of(s.picked, 64).name(), r"$\cos 2$");
        s.resize(8);
        assert!(s.picked < 8 && s.keep <= 8 && s.x.len() == 8);
        // the ladder of panel heights
        let steps = [0.0, 0.5, 0.9, 1.0, 1.2, 2.5, 3.5, 5.0, 7.0, 9.0, 33.0].map(ladder);
        let want = [1e-6, 0.6, 1.0, 1.0, 1.5, 3.0, 4.0, 6.0, 8.0, 10.0, 40.0];
        assert!(steps.iter().zip(want).all(|(p, q)| (p - q).abs() < 1e-12 * q), "{steps:?}");
        assert_eq!(ladder(f64::NAN), 1.0);
        assert!((scale_for(0.0) - 0.6).abs() < 1e-12);
    }

    /// Every canvas, from a wide desktop to a phone and far too small: every number of
    /// the layout finite; on the desktop and the phone, the panels inside the stage and
    /// clear of one another, in the order the scene reads.
    #[test]
    fn the_layout_fits_every_canvas() {
        // (stage width, pixels per unit, inset): 1080 × 1000, 1100 × 700, a phone, a tiny window
        for (w, ppu, inset, check) in [(5.4, 200.0, 0.0, true), (7.86, 140.0, 0.0, true), (2.44, 160.0, 46.0, true), (3.2, 70.0, 46.0, false), (2.0, 1.0, 46.0, false)] {
            let lay = Layout::of(w, ppu, inset);
            let areas = [lay.signal, lay.wave, lay.products, lay.coords, lay.rebuild];
            let mut all: Vec<Area> = areas.to_vec();
            all.extend(lay.heat.iter().flatten());
            assert!(all.iter().all(|a| a.min.is_finite() && a.max.is_finite()) && lay.track.0.is_finite() && lay.track.1.is_finite());
            if check {
                for a in &all {
                    assert!(a.min.x >= -0.5 * w && a.max.x <= 0.5 * w && a.min.y >= -2.5 && a.max.y <= 2.5 - inset / ppu, "{a:?} at width {w}");
                    assert!(a.h() * ppu > 30.0, "{a:?} at width {w}");
                }
                for (i, a) in all.iter().enumerate() {
                    for b in &all[i + 1..] {
                        let apart = a.max.x <= b.min.x || b.max.x <= a.min.x || a.max.y <= b.min.y || b.max.y <= a.min.y;
                        assert!(apart, "{a:?} and {b:?} at width {w}");
                    }
                }
                assert!(lay.heat.is_some());
            }
        }
    }

    /// The Words: for a real signal, the real part of the complex transform
    /// Xₖ = Σ xₙ e^(−2πikn/N) is √(N/2) times the coordinate of cos k, and the imaginary
    /// part −√(N/2) times that of sin k.
    #[test]
    fn the_complex_transform_says_the_same() {
        for n in [N_MIN, 9, N_START, N_MAX] {
            let b = Basis::new(n);
            for x in signals(n) {
                let c = b.forward(&x);
                for k in 1..n.div_ceil(2) {
                    let angle = |i: usize| -2.0 * PI64 * ((k * i) % n) as f64 / n as f64;
                    let re: f64 = x.iter().enumerate().map(|(i, v)| v * angle(i).cos()).sum();
                    let im: f64 = x.iter().enumerate().map(|(i, v)| v * angle(i).sin()).sum();
                    let scale = (n as f64 / 2.0).sqrt();
                    assert!((re - scale * c[2 * k - 1]).abs() < 1e-11 && (im + scale * c[2 * k]).abs() < 1e-11, "N {n}, k {k}");
                }
            }
        }
    }

    /// The note on the picture is broken into lines at spaces, never inside a formula, as
    /// `wrapped` breaks it: each word of every note is whole markup.
    #[test]
    fn the_notes_words_keep_each_formula_whole() {
        assert_eq!(note_words(r"so $a + b = c$ and $\$ x$ ends"), ["so", "$a + b = c$", "and", r"$\$ x$", "ends"]);
        for n in [N_MIN, 9, N_START] {
            for width in WIDTHS {
                let mut s = Dft::new();
                s.resize(n);
                s.width = width;
                for word in note_words(&s.why()) {
                    assert!(!word.is_empty() && word.replace(r"\$", "").matches('$').count() % 2 == 0, "{word}");
                    assert!(crate::typeset::pieces(word).is_ok(), "{word}");
                }
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for (k, (label, _, _)) in TRY.iter().enumerate() {
            let s = run(k);
            let c = s.coords();
            let green = s.rebuilt();
            let worst = s.x.iter().zip(&green).map(|(x, r)| (x - r).abs()).fold(0.0, f64::max);
            println!("{label}: coordinate {}, share kept {:?}, largest miss {worst:.4}", num(c[s.picked], 3), s.share_kept());
            println!("  green: {:?}", green.iter().map(|v| num(*v, 2)).collect::<Vec<_>>());
            println!("  stack: {:?}", stack(&s).iter().map(|v| num(*v, 2)).collect::<Vec<_>>());
        }
    }
}

//! High dimensions: in many dimensions two random directions are nearly at right
//! angles, the length of a random vector hardly varies, and a random projection into
//! far fewer dimensions keeps every distance nearly as it was.

use super::*;
use crate::clock::{Budget, Pace};
use std::cell::RefCell;
use std::f64::consts::{LN_2, PI as PI64, SQRT_2};

pub const DEF: SceneDef = SceneDef {
    id: "highdim",
    title: "High dimensions",
    group: "Linear algebra: data and dimensions",
    blurb: "Why random directions in many dimensions are almost perpendicular, and what that makes room for.",
    make: |_| Box::new(HighDim::new()),
};

/// The most dimensions, and the most directions in the heat map.
const N_MAX: usize = 1000;
const M_MAX: usize = 1000;
/// Pairs of directions drawn for the large chart (then it stops), and how many a
/// second of the scene's time.
const TOTAL: u32 = 20_000;
const RATE: f32 = 8_000.0;
/// The time in which a drawn bar covers two thirds of the way to its count (seconds).
const FOLLOW: f32 = 0.12;
/// The small charts side by side: their n, the pairs behind each, and their bars.
const SMALL_N: [usize; 4] = [2, 3, 10, 100];
const SMALL_PAIRS: u32 = 6_000;
const SMALL_BINS: usize = 30;
/// How long the heat map takes to fill in, whatever its size (seconds).
const FILL: f32 = 1.5;
/// The lengths chart: from 0 to this, in bars this many.
const LEN_MAX: f64 = 34.0;
const LEN_BINS: usize = 170;
/// The lengths drawn faintly for comparison.
const LEN_REF: [usize; 4] = [2, 10, 100, 1000];
/// The points of the projection view: so many, and the first at this distance from
/// the centre, the last at `FAR` (evenly between), so that their distances from one
/// another vary.
const POINTS: usize = 50;
const NEAR: f64 = 1.0;
const FAR: f64 = 6.0;
/// How far up the projection's chart goes, as a multiple of how far across (the two
/// scales are the same, so that "after = before" is a diagonal).
const REACH: f32 = 2.0;
/// The time in which glides that answer the visitor's hand cover two thirds of the
/// way (seconds).
const GLIDE: f32 = 0.1;

/// The family of seeds of the heat map's directions.
const MANY: u64 = 10;

/// The colours: one thing each.
const DRAWN: Color = pal::CYAN;
const EXACT: Color = pal::WHITE;
const TYPICAL: Color = pal::ORANGE;
const LARGEST: Color = pal::GREEN;
const CHOSEN: Color = pal::YELLOW;
const HEAT: Cmap = Cmap::IceFire;

/// What the picture shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    /// Two random directions, and many: their cosines.
    Angles,
    /// The length of a random vector.
    Lengths,
    /// Points pushed into fewer dimensions.
    Projection,
}

const TABS: [(&str, Tab); 3] = [("angles", Tab::Angles), ("lengths", Tab::Lengths), ("projection", Tab::Projection)];

/// Everything an experiment sets.
#[derive(Clone, Copy, Debug)]
struct Setup {
    tab: Tab,
    /// Dimensions of the directions, and how many directions the heat map has.
    n: usize,
    m: usize,
    /// Dimensions of the points before and after the projection.
    big_n: usize,
    k: usize,
}

const fn angles(n: usize, m: usize) -> Setup {
    Setup { tab: Tab::Angles, n, m, big_n: 1000, k: 50 }
}

/// The scene opens on the third. Every number here is checked by a test of the same
/// setting.
const TRY: [(&str, &str, Setup); 5] = [
    (
        "the plane",
        r"Two random directions in the plane. The angle between them is equally likely to be anything, but its cosine piles up at $\pm 1$, where the white curve grows without limit (it is cut off). Half the pairs (50.0 % of 20 000) are within 45° of lying along one line: $|\text{cosine}|$ above 0.71. In the heat map of 40 directions, 70 % of the cells are bright, $|\text{cosine}|$ above ½.",
        angles(2, 40),
    ),
    (
        "space",
        r"In three dimensions every cosine from −1 to 1 is equally likely: the bars come up level, at height ½. (It is Archimedes' discovery that a sphere's area is spread evenly along its axis.) Only 20 % of pairs are within 11.5° of a right angle, $|\text{cosine}|$ below 0.2.",
        angles(3, 40),
    ),
    (
        "a hundred dimensions",
        r"In 100 dimensions the cosine is a narrow bell, its spread (the orange marks) $1/\sqrt{n} = 0.1$: 95.5 % of 20 000 pairs have $|\text{cosine}|$ below 0.2, an angle between 78.5° and 101.5°. In the heat map of 40 directions every cell off the diagonal is dim: the largest $|\text{cosine}|$ of the 780 pairs (framed in green) is 0.34.",
        angles(100, 40),
    ),
    (
        "a thousand directions in a hundred dimensions",
        r"Ten times as many directions as dimensions. Only 100 could be exactly at right angles, yet the largest $|\text{cosine}|$ among all 499 500 pairs is 0.46: every two are at least 62.7° apart. The green curve is the largest among the first $r$: 0.37 at $r = 100$, as many as there are dimensions, and still only 0.46 at $r = 1000$.",
        angles(100, 1000),
    ),
    (
        "a thousand dimensions squashed into fifty",
        r"50 points in 1000 dimensions, multiplied by a random $50 \times 1000$ matrix (divided by $\sqrt{50}$). Every one of the 1225 distances between them survives, changed by a factor between 0.71 and 1.37 (the grey lines); 95 % of them by a factor between 0.80 and 1.22, and six in ten by less than 10 % either way (between the orange lines, $1 \pm 1/\sqrt{2k}$). Make $k$ smaller to see them scatter.",
        Setup { tab: Tab::Projection, n: 100, m: 40, big_n: 1000, k: 50 },
    ),
];

/// The setting the scene opens with.
const OPENING: usize = 2;

// ----- the arithmetic -----------------------------------------------------------------

/// ln Γ(x) for x ≥ 1/2 (Lanczos, accurate to about 15 digits).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let x = x - 1.0;
    let t = x + 7.5;
    let sum = C.iter().enumerate().skip(1).fold(C[0], |s, (i, c)| s + c / (x + i as f64));
    0.5 * (2.0 * PI64).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// The density of the cosine between two random directions in n dimensions,
/// proportional to (1 − c²)^((n − 3)/2). (Any real n from 2 up: the picture glides
/// from one whole n to the next.) Infinite at ±1 for n below 3.
fn cos_density(n: f64, c: f64) -> f64 {
    CosLaw::new(n).at(c)
}

/// That density for one n, with the part that does not depend on the cosine worked
/// out once: a curve of it has hundreds of points, and that part is two logarithms of
/// the gamma function.
#[derive(Clone, Copy)]
struct CosLaw {
    scale: f64,
    power: f64,
}

impl CosLaw {
    fn new(n: f64) -> CosLaw {
        let n = n.max(2.0);
        CosLaw { scale: (ln_gamma(n / 2.0) - ln_gamma((n - 1.0) / 2.0)).exp() / PI64.sqrt(), power: (n - 3.0) / 2.0 }
    }

    fn at(&self, c: f64) -> f64 {
        self.scale * (1.0 - c * c).max(0.0).powf(self.power)
    }
}

/// The density at the top of a chart of the cosine: some way above the peak where
/// there is one (n of 3 or more), leaving room for the names at the top, and for n
/// below 3, where there is none, three times as far above the density at 0 in the
/// plane, a little less on the way to 3.
fn cos_top(n: f64) -> f64 {
    let middle = cos_density(n, 0.0);
    if n >= 3.0 {
        1.3 * middle
    } else {
        1.3 * middle * (1.0 + 2.0 * (3.0 - n))
    }
}

/// The density of the length of a vector of n independent standard normal numbers.
#[cfg(test)]
fn len_density(n: f64, r: f64) -> f64 {
    LenLaw::new(n).at(r)
}

/// That density for one n, with the terms without r worked out once (as `CosLaw`).
/// They are kept apart and subtracted one at a time, in the formula's order, so that
/// every point is rounded just as the formula written out in one line would round it.
#[derive(Clone, Copy)]
struct LenLaw {
    n: f64,
    shift: f64,
    log_gamma: f64,
}

impl LenLaw {
    fn new(n: f64) -> LenLaw {
        LenLaw { n, shift: (0.5 * n - 1.0) * LN_2, log_gamma: ln_gamma(0.5 * n) }
    }

    fn at(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return 0.0;
        }
        ((self.n - 1.0) * r.ln() - 0.5 * r * r - self.shift - self.log_gamma).exp()
    }
}

/// The average of that length, and its standard deviation.
fn len_mean(n: f64) -> f64 {
    SQRT_2 * (ln_gamma(0.5 * (n + 1.0)) - ln_gamma(0.5 * n)).exp()
}

fn len_sd(n: f64) -> f64 {
    (n - len_mean(n).powi(2)).max(0.0).sqrt()
}

/// The rough rule for the largest |cosine| among m random directions: √(2 ln(m²/2)/n)
/// (the tail of a bell of spread 1/√n, against m²/2 pairs).
fn rule(n: usize, m: usize) -> f64 {
    let pairs = (m as f64).powi(2) / 2.0;
    (2.0 * pairs.max(1.0).ln() / n as f64).sqrt()
}

/// Fill `out` with standard normal numbers; returns their length.
fn normals(rng: &mut Rng, out: &mut [f64]) -> f64 {
    let mut sum = 0.0;
    for v in out.iter_mut() {
        *v = rng.normal() as f64;
        sum += *v * *v;
    }
    sum.sqrt().max(1e-300)
}

/// A dot product in single precision, eight sums at a time (which the compiler can
/// vectorise; one running sum it may not reorder).
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let tail: f32 = ca.remainder().iter().zip(cb.remainder()).map(|(x, y)| x * y).sum();
    for (x, y) in ca.zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// How many bars the chart of the cosine has for n: about a sixth of the spread 1/√n
/// each, so that the bell is always some thirty bars across and its tallest bar
/// counts about a fifteenth of the pairs.
fn cos_bins(n: usize) -> usize {
    ((12.0 * (n as f64).sqrt()).round() as usize).clamp(40, 400)
}

/// A seed for the `i`-th thing of a kind, in n dimensions: the same thing whatever
/// else has been asked for (one more direction leaves the others as they were).
fn seed(kind: u64, n: usize, i: usize) -> u64 {
    kind.wrapping_mul(1_000_003).wrapping_add((n as u64).wrapping_mul(10_007)).wrapping_add(i as u64)
}

// ----- histograms that fill in --------------------------------------------------------

/// Bars counting values between `lo` and `hi`, drawn as densities (in the units of
/// the exact curve) of what they will be once all `planned` values are in: so they
/// rise from nothing to meet the curve as values arrive.
#[derive(Clone, Debug)]
struct Hist {
    lo: f64,
    hi: f64,
    counts: Vec<u32>,
    planned: u32,
    /// The heights drawn, and a step between them and the counts' own: two eases
    /// one after the other, so that a bar whose count comes in a few at a time, more
    /// in one frame than the next, still rises evenly.
    between: Vec<f32>,
    shown: Vec<f32>,
}

impl Hist {
    fn new(lo: f64, hi: f64, bins: usize, planned: u32) -> Hist {
        let bins = bins.max(1);
        Hist { lo, hi, counts: vec![0; bins], planned: planned.max(1), between: vec![0.0; bins], shown: vec![0.0; bins] }
    }

    fn bins(&self) -> usize {
        self.counts.len()
    }

    fn width(&self) -> f64 {
        (self.hi - self.lo) / self.bins() as f64
    }

    /// Count a value (one outside the range goes in no bar).
    fn add(&mut self, x: f64) {
        let t = (x - self.lo) / (self.hi - self.lo);
        if t.is_finite() && (0.0..=1.0).contains(&t) {
            let i = ((t * self.bins() as f64) as usize).min(self.bins() - 1);
            self.counts[i] += 1;
        }
    }

    /// The height bar `i` has come to.
    fn height(&self, i: usize) -> f32 {
        self.counts.get(i).map_or(0.0, |&c| (c as f64 / (self.planned as f64 * self.width())) as f32)
    }

    /// Bring the drawn bars `dt` seconds nearer to their counts.
    fn follow(&mut self, dt: f32) {
        if dt <= 0.0 || self.caught_up() {
            return;
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        let (mut apart, mut most) = (0.0f32, 0.0f32);
        for i in 0..self.bins() {
            let h = self.height(i);
            self.between[i] += (h - self.between[i]) * k;
            self.shown[i] += (self.between[i] - self.shown[i]) * k;
            apart = apart.max((self.shown[i] - h).abs()).max((self.between[i] - h).abs());
            most = most.max(h).max(self.shown[i]);
        }
        // (exactly there in the end, not nearly: the picture then stops changing)
        if apart < 1e-4 * most.max(1e-3) {
            self.catch_up();
        }
    }

    fn catch_up(&mut self) {
        for i in 0..self.bins() {
            let h = self.height(i);
            (self.between[i], self.shown[i]) = (h, h);
        }
    }

    fn caught_up(&self) -> bool {
        (0..self.bins()).all(|i| self.shown[i] == self.height(i) && self.between[i] == self.height(i))
    }

    /// Start counting afresh in `bins` new bars. What is drawn keeps its shape,
    /// carried over to the new bars, and sinks from there while the new counts rise.
    fn restart(&mut self, lo: f64, hi: f64, bins: usize) {
        let old = std::mem::replace(self, Hist::new(lo, hi, bins, self.planned));
        for i in 0..self.bins() {
            let x = self.lo + (i as f64 + 0.5) * self.width();
            let t = (x - old.lo) / (old.hi - old.lo);
            if (0.0..1.0).contains(&t) {
                let j = ((t * old.bins() as f64) as usize).min(old.bins() - 1);
                (self.between[i], self.shown[i]) = (old.between[j], old.shown[j]);
            }
        }
    }
}

// ----- random directions, one after another --------------------------------------------

/// Random vectors in n dimensions, drawn one at a time, each paired with the one
/// before: a cosine (of the two directions), a dot product (of the two vectors as
/// drawn) and a length (of the new one) each time.
///
/// Every pair is two independent random directions. Successive pairs share one, but
/// they are independent all the same: the cosine of a pair has the same spread
/// whichever its first direction is, so knowing the cosine of the last pair tells
/// nothing about the next.
struct Stream {
    n: usize,
    rng: Rng,
    /// The last vector as drawn, and scaled to length 1.
    last: Vec<f64>,
    last_unit: Vec<f64>,
    fresh: Vec<f64>,
    pairs: u32,
    planned: u32,
    cos: Hist,
    len: Hist,
    /// Sums for the read-outs: of cosines squared, lengths, lengths squared, dot
    /// products squared; and every cosine drawn.
    c2: f64,
    l1: f64,
    l2: f64,
    d2: f64,
    cosines: Vec<f32>,
    /// For `share_below`: the bound last asked about, how many cosines have been
    /// looked at for it, and how many of those passed.
    below: (f64, usize, usize),
}

impl Stream {
    fn new(n: usize, seed: u64, planned: u32, bins: usize) -> Stream {
        let n = n.clamp(2, N_MAX);
        let mut rng = Rng::new(seed);
        let mut last = vec![0.0; n];
        let length = normals(&mut rng, &mut last);
        let last_unit = last.iter().map(|v| v / length).collect();
        Stream {
            n,
            rng,
            last,
            last_unit,
            fresh: vec![0.0; n],
            pairs: 0,
            planned,
            cos: Hist::new(-1.0, 1.0, bins, planned),
            len: Hist::new(0.0, LEN_MAX, LEN_BINS, planned),
            c2: 0.0,
            l1: 0.0,
            l2: 0.0,
            d2: 0.0,
            cosines: Vec::with_capacity(planned as usize),
            below: (f64::NAN, 0, 0),
        }
    }

    /// The stream the large charts are drawn from.
    fn main(n: usize) -> Stream {
        Stream::new(n, seed(1, n, 0), TOTAL, cos_bins(n))
    }

    /// Start again in n dimensions, the bars drawn so far sinking from where they are.
    fn renew(&mut self, n: usize) {
        let (mut cos, mut len) = (std::mem::replace(&mut self.cos, Hist::new(0.0, 1.0, 1, 1)), std::mem::replace(&mut self.len, Hist::new(0.0, 1.0, 1, 1)));
        *self = Stream::main(n);
        cos.restart(-1.0, 1.0, cos_bins(self.n));
        len.restart(0.0, LEN_MAX, LEN_BINS);
        (self.cos, self.len) = (cos, len);
    }

    fn complete(&self) -> bool {
        self.pairs >= self.planned
    }

    /// One more vector, paired with the last.
    fn step(&mut self) {
        let length = normals(&mut self.rng, &mut self.fresh);
        let (mut c, mut d) = (0.0, 0.0);
        for i in 0..self.n {
            let u = self.fresh[i] / length;
            c += u * self.last_unit[i];
            d += self.fresh[i] * self.last[i];
            self.last_unit[i] = u;
        }
        std::mem::swap(&mut self.last, &mut self.fresh);
        self.cos.add(c);
        self.len.add(length);
        self.c2 += c * c;
        self.l1 += length;
        self.l2 += length * length;
        self.d2 += d * d;
        self.cosines.push(c as f32);
        self.pairs += 1;
    }

    /// The share of the cosines drawn that pass `test`.
    #[cfg(test)]
    fn share(&self, test: impl Fn(f64) -> bool) -> f64 {
        self.cosines.iter().filter(|c| test(**c as f64)).count() as f64 / self.cosines.len().max(1) as f64
    }

    /// The share of the cosines drawn whose size is below `bound`, as `share` gives
    /// it, looking only at those drawn since it was last asked about the same bound
    /// (the panel asks in every frame, and there are up to 20 000).
    fn share_below(&mut self, bound: f64) -> f64 {
        if self.below.0 != bound {
            self.below = (bound, 0, 0);
        }
        let (_, looked, passed) = &mut self.below;
        *passed += self.cosines[*looked..].iter().filter(|c| (**c as f64).abs() < bound).count();
        *looked = self.cosines.len();
        *passed as f64 / self.cosines.len().max(1) as f64
    }

    /// The standard deviation of the cosines drawn (their average is 0).
    fn spread(&self) -> f64 {
        (self.c2 / self.pairs.max(1) as f64).sqrt()
    }

    fn mean_length(&self) -> f64 {
        self.l1 / self.pairs.max(1) as f64
    }

    fn length_spread(&self) -> f64 {
        let k = self.pairs.max(1) as f64;
        (self.l2 / k - (self.l1 / k).powi(2)).max(0.0).sqrt()
    }

    /// The typical size of the dot product of two vectors as drawn.
    fn dot_spread(&self) -> f64 {
        (self.d2 / self.pairs.max(1) as f64).sqrt()
    }
}

/// The small chart for n: its pairs all drawn at once.
fn small(n: usize) -> Hist {
    let mut s = Stream::new(n, seed(2, n, 0), SMALL_PAIRS, SMALL_BINS);
    while !s.complete() {
        s.step();
    }
    s.cos.catch_up();
    s.cos
}

// ----- many directions ----------------------------------------------------------------

/// m random directions in n dimensions and every dot product among them, worked out
/// one direction at a time: the next one against all those before it.
struct Many {
    /// The family of seeds the directions come from.
    kind: u64,
    n: usize,
    m: usize,
    /// The directions so far, one after another.
    vecs: Vec<f32>,
    rows: usize,
    /// The largest |cosine| so far, with its sign, and its pair (i < j).
    best: Option<(f32, usize, usize)>,
    /// The largest |cosine| among the first r + 1 directions, for each r so far.
    by_r: Vec<f32>,
    /// The heat map, row 0 at the top (cells not yet worked out are clear).
    pixels: Vec<Color>,
    /// How many of these directions the picture on the GPU shows (None: none, as a
    /// picture that is new or shows other directions). Direction r changes only row r
    /// and column r of the heat map (`step`), so the picture is brought up to date by
    /// copying those, not all m × m cells: with 1000 directions, a million.
    painted: Option<usize>,
    pace: Pace,
}

impl Many {
    fn new(n: usize, m: usize) -> Many {
        Many::drawn(n, m, MANY)
    }

    /// The same with directions from another family of seeds.
    fn drawn(n: usize, m: usize, kind: u64) -> Many {
        let (n, m) = (n.clamp(2, N_MAX), m.clamp(1, M_MAX));
        Many { kind, n, m, vecs: vec![0.0; m * n], rows: 0, best: None, by_r: Vec::with_capacity(m), pixels: vec![Color::CLEAR; m * m], painted: None, pace: Pace::default() }
    }

    fn complete(&self) -> bool {
        self.rows >= self.m
    }

    /// The next direction, and its dot products with all before it.
    fn step(&mut self) {
        let (n, m, r) = (self.n, self.m, self.rows);
        if r >= m {
            return;
        }
        let mut g = vec![0.0; n];
        let length = normals(&mut Rng::new(seed(self.kind, n, r)), &mut g);
        let (done, rest) = self.vecs.split_at_mut(r * n);
        for (v, x) in rest.iter_mut().zip(&g) {
            *v = (x / length) as f32;
        }
        let new = &rest[..n];
        let table = HEAT.table();
        for j in 0..r {
            let c = dot(&done[j * n..(j + 1) * n], new);
            let color = table[heat_index(c)];
            self.pixels[r * m + j] = color;
            self.pixels[j * m + r] = color;
            if self.best.is_none_or(|b| c.abs() > b.0.abs()) {
                self.best = Some((c, j, r));
            }
        }
        self.pixels[r * m + r] = table[heat_index(1.0)];
        self.by_r.push(self.best.map_or(0.0, |b| b.0.abs()));
        self.rows += 1;
    }

    /// Bring `picture` (m × m, its rows counting from the bottom) up to date with the
    /// heat map: all of it if it holds none of these directions, else the rows and
    /// columns of the directions worked out since.
    fn paint(&mut self, picture: &mut [Color]) {
        let m = self.m;
        match self.painted {
            None => {
                // (picture rows count from the bottom, the matrix's from the top)
                for i in 0..m {
                    picture[(m - 1 - i) * m..(m - i) * m].copy_from_slice(&self.pixels[i * m..(i + 1) * m]);
                }
            }
            Some(done) => {
                for r in done..self.rows {
                    let row = (m - 1 - r) * m;
                    picture[row..row + r + 1].copy_from_slice(&self.pixels[r * m..r * m + r + 1]);
                    for j in 0..r {
                        picture[(m - 1 - j) * m + r] = self.pixels[j * m + r];
                    }
                }
            }
        }
        self.painted = Some(self.rows);
    }

    /// The blocks of the picture that `paint` will change (`[x, y, width, height]`,
    /// rows counting from the bottom): the rows of the directions worked out since it
    /// was last painted, and their columns. None if it will change all of it.
    fn unpainted(&self) -> Option<[[u32; 4]; 2]> {
        let (m, rows, done) = (self.m as u32, self.rows as u32, self.painted? as u32);
        let new = rows.saturating_sub(done);
        Some([[0, m.saturating_sub(rows), m, new], [done, m.saturating_sub(rows), new, rows.min(m)]])
    }

    /// Pairs among the directions so far.
    fn pairs(&self) -> usize {
        self.rows * self.rows.saturating_sub(1) / 2
    }
}

/// Where a cosine goes in the heat map's colours.
fn heat_index(c: f32) -> usize {
    table_index(0.5 + 0.5 * c.clamp(-1.0, 1.0))
}

// ----- the projection -----------------------------------------------------------------

/// `POINTS` points in N dimensions, and their images under a random k × N matrix of
/// standard normal numbers divided by √k, one row of it at a time.
struct Projection {
    big_n: usize,
    k: usize,
    points: Vec<f64>,
    pairs: Vec<(usize, usize)>,
    before: Vec<f64>,
    /// Each row of the matrix applied to every point, row after row.
    coords: Vec<f64>,
    rows: usize,
    /// For each pair, the sum of its squared differences over the first `used` rows.
    sums: Vec<f64>,
    used: usize,
    /// Every ratio of distance after to distance before, smallest first, and their
    /// `summary`: what the picture, its words and the panel all read in every frame.
    /// Worked out again in `work` whenever `used` changes, which is the only place
    /// where it or the sums do.
    sorted: Vec<f64>,
    stats: Option<[f64; 5]>,
}

impl Projection {
    fn new(big_n: usize, k: usize) -> Projection {
        let big_n = big_n.clamp(2, N_MAX);
        let k = k.clamp(1, big_n);
        let mut points = vec![0.0; POINTS * big_n];
        for (i, p) in points.chunks_exact_mut(big_n).enumerate() {
            let length = normals(&mut Rng::new(seed(4, big_n, i)), p);
            let radius = NEAR + (FAR - NEAR) * i as f64 / (POINTS - 1) as f64;
            p.iter_mut().for_each(|v| *v *= radius / length);
        }
        let pairs: Vec<(usize, usize)> = (0..POINTS).flat_map(|i| (i + 1..POINTS).map(move |j| (i, j))).collect();
        let before = pairs
            .iter()
            .map(|&(i, j)| {
                let (a, b) = (&points[i * big_n..(i + 1) * big_n], &points[j * big_n..(j + 1) * big_n]);
                a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
            })
            .collect();
        Projection { big_n, k, points, sums: vec![0.0; pairs.len()], pairs, before, coords: Vec::new(), rows: 0, used: 0, sorted: Vec::new(), stats: None }
    }

    /// One more row of the matrix, applied to every point.
    fn add_row(&mut self) {
        let n = self.big_n;
        let mut a = vec![0.0; n];
        normals(&mut Rng::new(seed(5, n, self.rows)), &mut a);
        for p in self.points.chunks_exact(n) {
            self.coords.push(p.iter().zip(&a).map(|(x, y)| x * y).sum());
        }
        self.rows += 1;
    }

    /// Carry on towards k rows, within `budget` (at least one row a call).
    fn work(&mut self, budget: &Budget) {
        let was = self.used;
        if self.used > self.k {
            // fewer rows than were summed: sum again from the start
            self.sums.iter_mut().for_each(|s| *s = 0.0);
            self.used = 0;
        }
        let mut made = 0;
        while self.rows < self.k && !(made > 0 && budget.spent()) {
            self.add_row();
            made += 1;
        }
        while self.used < self.rows.min(self.k) {
            let row = &self.coords[self.used * POINTS..(self.used + 1) * POINTS];
            for (s, &(i, j)) in self.sums.iter_mut().zip(&self.pairs) {
                *s += (row[i] - row[j]).powi(2);
            }
            self.used += 1;
        }
        // (summing again from the start always ends with fewer rows than before)
        if self.used != was {
            let mut r: Vec<f64> = (0..self.pairs.len()).filter_map(|p| self.after(p).map(|a| a / self.before[p])).collect();
            r.sort_by(f64::total_cmp);
            self.stats = summary(&r);
            self.sorted = r;
        }
    }

    fn done(&self) -> bool {
        self.used == self.k
    }

    /// The distance after, for each pair (none before a row has been applied).
    fn after(&self, p: usize) -> Option<f64> {
        (self.used > 0).then(|| (self.sums[p] / self.used as f64).sqrt())
    }

    /// Every ratio of distance after to distance before, smallest first.
    fn ratios(&self) -> &[f64] {
        &self.sorted
    }

    /// Their `summary`.
    fn summary(&self) -> Option<[f64; 5]> {
        self.stats
    }
}

/// The smallest, the largest, the middle 95 % and the standard deviation of ratios
/// sorted smallest first.
fn summary(r: &[f64]) -> Option<[f64; 5]> {
    let (first, last) = (r.first()?, r.last()?);
    let at = |q: f64| r[((q * (r.len() - 1) as f64).round() as usize).min(r.len() - 1)];
    let mean = r.iter().sum::<f64>() / r.len() as f64;
    let sd = (r.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / r.len() as f64).sqrt();
    Some([*first, *last, at(0.025), at(0.975), sd])
}

// ----- the scene ----------------------------------------------------------------------

pub struct HighDim {
    tab: Tab,
    n: usize,
    m: usize,
    big_n: usize,
    k: usize,
    stream: Stream,
    pace: Pace,
    smalls: Vec<Hist>,
    many: Many,
    heat: Option<Image>,
    proj: Projection,
    /// The n the curves are drawn for: it glides to n, in proportion.
    n_shown: f64,
    /// Where the frame round the largest |cosine| is drawn, as a share of the heat
    /// map's side across and down, and a step between it and where it is going: it
    /// glides to each new largest (two eases one after the other, so that it sets off
    /// gently).
    ring: Option<Vec2>,
    ring_between: Vec2,
    /// The projection's pairs as drawn (distance before, distance after), gliding to
    /// where they are, and the distance at the right-hand edge of the chart.
    dots: Vec<Vec2>,
    span: f32,
    /// The factors after ÷ before, counted, the bars gliding to their counts; and the
    /// k the exact law beside them is drawn for (gliding to k, in proportion).
    factors: Hist,
    k_shown: f64,
    /// The exact laws' curves as last drawn.
    sketches: RefCell<Sketches>,
    /// The words under the lengths and the projection as last broken into lines, a
    /// paragraph each (the two views' paragraphs share the places, one view at a time).
    broken: RefCell<Vec<Broken>>,
    /// What the counts and the dots' places were last worked out for: N, the rows of
    /// the matrix summed, and (for the counts) the number of bars. The points and
    /// every row of the matrix follow from N and the row's number alone (`seed`), so
    /// these two numbers say everything the distances after depend on.
    counted: Option<(usize, usize, usize)>,
    goals: Option<((usize, usize), Vec<Vec2>, f32)>,
}

impl HighDim {
    fn new() -> HighDim {
        let s = TRY[OPENING].2;
        let mut scene = HighDim {
            tab: s.tab,
            n: s.n,
            m: s.m,
            big_n: s.big_n,
            k: s.k,
            stream: Stream::main(s.n),
            pace: Pace::default(),
            smalls: SMALL_N.iter().map(|&n| small(n)).collect(),
            many: Many::new(s.n, s.m),
            heat: None,
            proj: Projection::new(s.big_n, s.k),
            n_shown: s.n as f64,
            ring: None,
            ring_between: Vec2::ZERO,
            dots: Vec::new(),
            span: 10.0,
            factors: Hist::new(0.0, 2.0, factor_bins(s.k), 1),
            k_shown: s.k as f64,
            counted: None,
            goals: None,
            sketches: RefCell::default(),
            broken: RefCell::default(),
        };
        scene.apply(s);
        scene
    }

    /// Set an experiment up (and run its random draws again from the start, so that
    /// the picture fills in as it did the first time).
    fn apply(&mut self, s: Setup) {
        self.tab = s.tab;
        (self.n, self.m) = (s.n.clamp(2, N_MAX), s.m.clamp(1, M_MAX));
        self.stream.renew(self.n);
        self.pace.reset();
        self.many = Many::new(self.n, self.m);
        self.ring = None;
        self.set_projection(s.big_n, s.k);
    }

    /// A new n: new directions, drawn afresh.
    fn set_n(&mut self, n: usize) {
        let n = n.clamp(2, N_MAX);
        if n != self.stream.n {
            self.n = n;
            self.stream.renew(n);
            self.pace.reset();
            self.many = Many::new(n, self.m);
            self.ring = None;
        }
    }

    /// A new m: the first directions are the same, but the heat map is worked out
    /// again from the start.
    fn set_m(&mut self, m: usize) {
        let m = m.clamp(1, M_MAX);
        if m != self.many.m {
            self.m = m;
            self.many = Many::new(self.n, m);
            self.ring = None;
        }
    }

    fn set_projection(&mut self, big_n: usize, k: usize) {
        self.big_n = big_n.clamp(2, N_MAX);
        self.k = k.clamp(1, self.big_n);
        if self.big_n != self.proj.big_n {
            self.proj = Projection::new(self.big_n, self.k);
        }
        self.proj.k = self.k;
    }

    /// Draw pairs at the rate asked for, within the frame's allowance.
    fn sample(&mut self, dt: f32, real_dt: f32) {
        if dt <= 0.0 || self.stream.complete() {
            return;
        }
        let mut turn = self.pace.begin(RATE, dt, real_dt, 0.002);
        while !self.stream.complete() && turn.next() {
            self.stream.step();
        }
        self.pace.end(turn);
    }

    /// Fill the heat map in at its rate, within the frame's allowance.
    fn fill(&mut self, dt: f32, real_dt: f32) {
        if dt <= 0.0 || self.many.complete() {
            return;
        }
        let rate = (self.many.m as f32 / FILL).max(2.0);
        let mut turn = self.many.pace.begin(rate, dt, real_dt, 0.002);
        while !self.many.complete() && turn.next() {
            self.many.step();
        }
        self.many.pace.end(turn);
    }

    /// Everything a frame moves on, `dt` seconds of scene time and `real_dt` of real
    /// time after the last (the tests run the scene through this too).
    fn advance(&mut self, dt: f32, real_dt: f32, budget: &Budget) {
        match self.tab {
            Tab::Angles | Tab::Lengths => {
                self.sample(dt, real_dt);
                if self.tab == Tab::Angles {
                    self.fill(dt, real_dt);
                }
                self.stream.cos.follow(dt);
                self.stream.len.follow(dt);
                self.glide_ring(dt);
            }
            // (the projection answers the settings: it goes on while the scene is paused)
            Tab::Projection => {
                self.proj.work(budget);
                self.count_factors(real_dt);
            }
        }
        self.n_shown = glide_in_proportion(self.n_shown, self.n as f64, real_dt);
        self.k_shown = glide_in_proportion(self.k_shown, self.k as f64, real_dt);
        self.glide_dots(real_dt);
    }

    /// Count the factors after ÷ before as they now are, and let the bars glide to
    /// the counts (by the clock: they answer the visitor's hand).
    fn count_factors(&mut self, dt: f32) {
        let bins = factor_bins(self.k);
        if self.factors.bins() != bins {
            self.factors.restart(0.0, 2.0, bins);
        }
        self.factors.planned = self.proj.pairs.len().max(1) as u32;
        let now = Some((self.proj.big_n, self.proj.used, bins));
        if self.counted != now {
            self.factors.counts.iter_mut().for_each(|c| *c = 0);
            for p in 0..self.proj.pairs.len() {
                if let Some(after) = self.proj.after(p) {
                    self.factors.add(after / self.proj.before[p]);
                }
            }
            self.counted = now;
        }
        self.factors.follow(dt);
    }

    /// Where the frame round the largest should be: the cell of the largest |cosine|, above the diagonal.
    fn ring_goal(&self) -> Option<Vec2> {
        let (_, i, j) = self.many.best?;
        let m = self.many.m as f32;
        Some(Vec2::new((j as f32 + 0.5) / m, (i as f32 + 0.5) / m))
    }

    fn glide_ring(&mut self, dt: f32) {
        let Some(goal) = self.ring_goal() else {
            self.ring = None;
            return;
        };
        let k = 1.0 - (-dt.max(0.0) / GLIDE).exp();
        let Some(at) = self.ring else {
            (self.ring, self.ring_between) = (Some(goal), goal);
            return;
        };
        self.ring_between += (goal - self.ring_between) * k;
        let mut at = at + (self.ring_between - at) * k;
        if ((goal - at).length() < 2e-4 && (goal - self.ring_between).length() < 2e-4) || !at.is_finite() || !self.ring_between.is_finite() {
            (at, self.ring_between) = (goal, goal);
        }
        self.ring = Some(at);
    }

    /// Where the projection's pairs are now, and the distance the chart should reach
    /// across: a round number a little beyond the farthest pair.
    fn dot_goals(&self) -> (Vec<Vec2>, f32) {
        let goals = (0..self.proj.pairs.len())
            .map(|p| {
                let before = self.proj.before[p];
                let after = self.proj.after(p).unwrap_or(before);
                Vec2::new(before as f32, after as f32)
            })
            .collect();
        let far = self.proj.before.iter().fold(0.0f64, |a, b| a.max(*b)) as f32 * 1.04;
        let step = nice_step(far / 4.0);
        (goals, (far / step).ceil().max(1.0) * step)
    }

    fn glide_dots(&mut self, dt: f32) {
        let key = (self.proj.big_n, self.proj.used);
        if self.goals.as_ref().is_none_or(|g| g.0 != key) {
            let (goals, span) = self.dot_goals();
            self.goals = Some((key, goals, span));
        }
        let Some((_, goals, span)) = &self.goals else { return };
        let span = *span;
        if self.dots.len() != goals.len() {
            (self.dots, self.span) = (goals.clone(), span);
            return;
        }
        let k = 1.0 - (-dt.max(0.0) / GLIDE).exp();
        let mut apart = 0.0f32;
        for (d, g) in self.dots.iter_mut().zip(goals) {
            *d += (*g - *d) * k;
            apart = apart.max((*g - *d).length());
        }
        self.span += (span - self.span) * k;
        if apart < 2e-3 && (span - self.span).abs() < 2e-3 {
            self.dots.clone_from(goals);
            self.span = span;
        }
    }

    fn dots_moving(&self) -> bool {
        match &self.goals {
            Some((key, goals, span)) if *key == (self.proj.big_n, self.proj.used) => self.dots != *goals || self.span != *span,
            _ => {
                let (goals, span) = self.dot_goals();
                self.dots != goals || self.span != span
            }
        }
    }

    fn bars_moving(&self) -> bool {
        !self.stream.complete() || !self.stream.cos.caught_up() || !self.stream.len.caught_up()
    }
}

/// `dt` seconds of a glide from `from` to `to` that covers the same share of the way
/// in proportion each moment (from 2 to 4 as from 200 to 400), exactly there at
/// the end.
fn glide_in_proportion(from: f64, to: f64, dt: f32) -> f64 {
    let k = 1.0 - (-(dt.max(0.0) / GLIDE) as f64).exp();
    let at = (from.max(1e-9).ln() + (to.max(1e-9).ln() - from.max(1e-9).ln()) * k).exp();
    if !at.is_finite() || (at.ln() - to.max(1e-9).ln()).abs() < 1e-4 {
        to
    } else {
        at
    }
}

/// How many bars count the factors after ÷ before, from 0 to 2: about a quarter of
/// their spread 1/√(2k) each, between 20 and 200.
fn factor_bins(k: usize) -> usize {
    ((8.0 * (2.0 * k as f64).sqrt()).round() as usize).clamp(20, 200)
}

/// The exact laws' curves as last drawn (on the screen), each kept with what it was
/// drawn for: the n (or k) of its law and the corners of its chart, to the bit. A
/// curve is hundreds of points of a power, or of a logarithm and an exponential, and
/// in most frames (the bars rising, a heat map filling in, the pointer moving) it is
/// the same as in the last. Each place a curve is drawn has a slot of its own.
#[derive(Default)]
struct Sketches {
    kept: Vec<Option<Sketch>>,
}

/// A curve's points (on the screen, each with its colour), and the n of its law and
/// its chart's corners, as bits.
type Sketch = ((u64, [u32; 4]), Vec<(Vec2, Color)>);

/// The slots: the large chart of the cosine, the small ones, the lengths' laws (for
/// each n in `LEN_REF`, and the one drawn bright) and the factors' law.
const MAIN_SKETCH: usize = 0;
const SMALL_SKETCH: usize = 1;
const LEN_SKETCH: usize = SMALL_SKETCH + SMALL_N.len();
const FACTOR_SKETCH: usize = LEN_SKETCH + LEN_REF.len() + 1;

impl Sketches {
    /// The curve in `slot` for a law of `n` in the chart `min..max`, drawn by `make`
    /// unless it is the one kept there.
    fn curve(&mut self, slot: usize, n: f64, min: Vec2, max: Vec2, make: impl FnOnce() -> Vec<(Vec2, Color)>) -> &[(Vec2, Color)] {
        let of = (n.to_bits(), [min.x, min.y, max.x, max.y].map(f32::to_bits));
        if self.kept.len() <= slot {
            self.kept.resize_with(slot + 1, || None);
        }
        let kept = &mut self.kept[slot];
        if kept.as_ref().is_none_or(|k| k.0 != of) {
            *kept = Some((of, make()));
        }
        kept.as_ref().map_or(&[], |k| &k.1)
    }
}

/// The angle in degrees between two lines with this |cosine|.
fn angle_apart(c: f64) -> f64 {
    c.clamp(0.0, 1.0).acos().to_degrees()
}

/// How wide a text is on the screen: one with mathematics in it exactly (and kept from
/// frame to frame, as `Frame::text_size` keeps it), words alone as `text_width` judges.
/// None for one with mathematics in it before the fonts are in place (the first frame).
fn measured(f: &Frame, text: &str, size: f32) -> Option<f32> {
    if crate::typeset::has_math(text) {
        crate::typeset::measure(f.ctx, text, size).map(|s| s.x)
    } else {
        Some(text_width(text, size))
    }
}

/// The same, and before the fonts are in place as `text_width` judges it.
fn width(f: &Frame, text: &str, size: f32) -> f32 {
    measured(f, text, size).unwrap_or_else(|| text_width(text, size))
}

/// The first of `options` that fits in `room` logical pixels (the last if none does).
fn pick<'a>(f: &Frame, options: &[&'a str], room: f32, size: f32) -> &'a str {
    options.iter().copied().find(|t| width(f, t, size) <= room).unwrap_or(options[options.len() - 1])
}

/// A translucent rectangle in screen space, with its colour fading from `bottom` to
/// `top` (on the level of what lies on a panel).
fn hud_bar(f: &mut Frame, min: Vec2, max: Vec2, bottom: Color, top: Color) {
    let p = [Vec2::new(min.x, max.y), max, Vec2::new(max.x, min.y), min].map(|p| f.hud_at(1, p));
    f.dl.tris_alpha.quad_grad(p, [bottom, bottom, top, top]);
}

/// A line in screen space on top of everything, in a batch of its own: for lines
/// whose number of points changes from frame to frame (a curve that grows, a ring
/// that comes and goes, a curve broken where it is cut off), kept apart from the
/// others so that `eidos motion` can still follow those, which keep theirs.
/// Each kind of such line has a batch to itself (`batch`: 0 the exact curves, 1 the
/// arrows where they are cut off, 2 the diagonal of the heat map, 3 the frame round the largest, 4 the
/// curve of the largest).
fn hud_path_apart(f: &mut Frame, batch: u8, points: impl Iterator<Item = Vec2>, width: f32, color: Color) {
    let points: Vec<Vec3> = points.map(|p| f.hud(p)).collect();
    // (the fading is for trails: in a line batch it only tells one batch from another)
    let style = LineStyle { depth_bias: 0.0, fade_power: 10.0 + batch as f32, ..Default::default() };
    f.dl.lines_with(LineKind::Overlay, style).polyline(points, width, color);
}

/// The same with a colour for each point (a part drawn clear is not seen).
fn hud_curve_apart(f: &mut Frame, batch: u8, points: impl Iterator<Item = (Vec2, Color)>, width: f32) {
    let points: Vec<(Vec3, Color)> = points.map(|(p, c)| (f.hud(p), c)).collect();
    let style = LineStyle { depth_bias: 0.0, fade_power: 10.0 + batch as f32, ..Default::default() };
    let lines = f.dl.lines_with(LineKind::Overlay, style);
    lines.brk();
    for (p, c) in points {
        lines.push(p, width, c);
    }
    lines.brk();
}

/// A dashed line in screen space, on top of everything.
fn hud_dashed(f: &mut Frame, a: Vec2, b: Vec2, width: f32, color: Color) {
    let (a, b) = (f.hud(a), f.hud(b));
    f.dl.lines_with(LineKind::Overlay, LineStyle { dash: 7.0, duty: 0.55, depth_bias: 0.0, ..Default::default() }).segment(a, b, width, color);
}

/// A button over the picture: true when clicked.
fn hud_tab(f: &mut Frame, min: Vec2, max: Vec2, text: &str, on: bool, pointer: Option<Vec2>, size: f32) -> bool {
    let over = pointer.is_some_and(|p| p.cmpge(min).all() && p.cmple(max).all());
    let fill = match (on, over) {
        (true, _) => Color::hex(0x2B3B60),
        (false, true) => Color::hex(0x2A3349),
        (false, false) => Color::hex(0x161C2B),
    };
    hud_rect(f, min, max, fill);
    if on {
        hud_line(f, Vec2::new(min.x, max.y - 1.0), Vec2::new(max.x, max.y - 1.0), 2.0, pal::LINK);
    }
    f.text_px((min + max) * 0.5, text, size, if on { pal::TEXT } else { pal::DIM }, Align2::CENTER_CENTER);
    over && f.ptr.clicked
}

/// Coloured swatches with their meanings, left to right from `at`, onto further lines
/// when they reach `room`. Returns the height taken.
fn legend(f: &mut Frame, at: Vec2, room: f32, parts: &[(Color, &str)], size: f32) -> f32 {
    let line = size + 5.0;
    let mut p = at;
    for (color, text) in parts {
        let w = size * 0.8 + 5.0 + width(f, text, size);
        if p.x > at.x && p.x + w > at.x + room {
            p = Vec2::new(at.x, p.y + line);
        }
        let s = size * 0.7;
        hud_rect(f, p + Vec2::new(0.0, (line - s) * 0.5 - 2.0), p + Vec2::new(s, (line + s) * 0.5 - 2.0), *color);
        f.text_px(p + Vec2::new(s + 5.0, line * 0.5 - 2.0), *text, size, *color, Align2::LEFT_CENTER);
        p.x += w + size * 1.4;
    }
    p.y + line - at.y
}

/// How wide a legend is on one line.
fn legend_width(f: &Frame, parts: &[(Color, &str)], size: f32) -> f32 {
    parts.iter().map(|(_, text)| size * 2.2 + 5.0 + width(f, text, size)).sum::<f32>() - size * 1.4
}

/// The screen rectangle `min`..`max` (logical pixels) in the scene's flat
/// coordinates, as the camera now shows them: where a picture must go to lie there.
fn world_rect(f: &Frame, min: Vec2, max: Vec2) -> (Vec2, Vec2) {
    let (lo, hi) = f.cam.flat_bounds();
    let view = f.view_size().max(Vec2::ONE);
    let to = |p: Vec2| Vec2::new(lo.x + p.x / view.x * (hi.x - lo.x), hi.y - p.y / view.y * (hi.y - lo.y));
    let (a, b) = (to(min), to(max));
    (Vec2::new(a.x, b.y), Vec2::new(b.x, a.y))
}

/// Where everything goes on a canvas of `view` logical pixels, the top `inset` of
/// which lies under the application's buttons.
#[derive(Clone, Copy, Debug)]
struct Layout {
    narrow: bool,
    /// Text sizes: titles, words, small print.
    big: f32,
    text: f32,
    small: f32,
    /// The row of tabs (left end, width, height), and where the title starts below it.
    tabs: (Vec2, f32, f32),
    title: Vec2,
    /// How wide the column of the main chart is.
    column: f32,
    /// The main chart (top-left, bottom-right corners of the plot itself).
    main: (Vec2, Vec2),
    /// The small charts, side by side (none where there is no room).
    smalls: Option<(Vec2, Vec2)>,
    /// The heat map: top-left corner and side; and the chart beside or under it.
    heat: Option<(Vec2, f32)>,
    best: Option<(Vec2, Vec2)>,
    /// Where words and numbers go under or beside the main chart, and how wide.
    words: Option<(Vec2, f32)>,
    /// A second chart beside the main one (the projection's factors, counted).
    side: Option<(Vec2, Vec2)>,
}

impl Layout {
    fn of(tab: Tab, view: Vec2, inset: f32) -> Layout {
        let narrow = view.x < 620.0;
        let (big, text, small) = if narrow { (13.0, 11.5, 10.5) } else { (15.0, 12.5, 11.0) };
        let side = if narrow { 14.0 } else { 24.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let tab_w = if narrow { ((view.x - 2.0 * side - 8.0) / 3.0).max(30.0) } else { 104.0 };
        let tabs = (Vec2::new(side, top), tab_w, if narrow { 26.0 } else { 28.0 });
        let title = Vec2::new(side, top + tabs.2 + if narrow { 10.0 } else { 16.0 });
        // the room taken by the title, its second line and the legend under it
        let head = if narrow { 66.0 } else { 70.0 };
        let left = side + if narrow { 18.0 } else { 30.0 };
        match tab {
            Tab::Angles => {
                let column = if narrow { view.x - side } else { (view.x * 0.6).max(left + 100.0) };
                let main_top = title.y + head;
                // under the chart: two rows of numbers and a line of words; then the
                // small charts with a line above; then the heat map (on a phone)
                let below = if narrow { 50.0 } else { 58.0 };
                let small_h = if narrow { 42.0 } else { (view.y * 0.085).clamp(40.0, 86.0) };
                // (above the small charts: a line of words, and their names; under
                // them, on a wide canvas, what each shows)
                let small_room = small_h + 2.0 * (small + 7.0) + if narrow { 6.0 } else { small + 12.0 };
                let heat_room = if narrow { (view.x * 0.42).min(view.y * 0.22) + 96.0 } else { 0.0 };
                let h = (view.y - main_top - below - small_room - heat_room - 16.0).clamp(60.0, if narrow { 260.0 } else { 540.0 });
                let main = (Vec2::new(left, main_top), Vec2::new(column - 20.0, main_top + h));
                let small_top = main.1.y + below + 2.0 * (small + 7.0);
                let smalls = (small_top + small_h < view.y - 4.0).then_some((Vec2::new(left, small_top), Vec2::new(column - 20.0, small_top + small_h)));
                let (heat, best) = if narrow {
                    let s = (view.x * 0.42).min(view.y * 0.22);
                    let y = small_top + small_h + 44.0;
                    let fits = y + s + 46.0 < view.y && smalls.is_some();
                    let best_left = side + s + 40.0;
                    (fits.then_some((Vec2::new(side, y), s)), (fits && view.x - side - best_left > 80.0).then_some((Vec2::new(best_left, y + 18.0), Vec2::new(view.x - side - 4.0, y + s - 4.0))))
                } else {
                    let x = column + 14.0;
                    let w = view.x - side - x;
                    let s = (w - 10.0).min(view.y * 0.34).max(0.0);
                    let y = main_top;
                    let best_top = y + s + 118.0;
                    let best_h = (view.y - best_top - 40.0).min(240.0);
                    (
                        (s >= 60.0).then_some((Vec2::new(x, y), s)),
                        (s >= 60.0 && best_h >= 50.0).then_some((Vec2::new(x + 30.0, best_top), Vec2::new(view.x - side - 8.0, best_top + best_h))),
                    )
                };
                Layout { narrow, big, text, small, tabs, title, column, main, smalls, heat, best, words: None, side: None }
            }
            Tab::Lengths => {
                let column = view.x - side;
                let main_top = title.y + head;
                let words_h = if narrow { 170.0 } else { 120.0 };
                let h = (view.y - main_top - 50.0 - words_h - 16.0).clamp(60.0, if narrow { 320.0 } else { 540.0 });
                let main = (Vec2::new(left, main_top), Vec2::new(column - 8.0, main_top + h));
                let words_top = main.1.y + 52.0;
                let words = (words_top + 40.0 < view.y).then_some((Vec2::new(side, words_top), column - side));
                Layout { narrow, big, text, small, tabs, title, column, main, smalls: None, heat: None, best: None, words, side: None }
            }
            Tab::Projection => {
                let main_top = title.y + head;
                // a chart twice as tall as it is wide, and the numbers beside it or under it
                let (main, words, chart) = if narrow {
                    let h = ((view.x - left - side) * REACH).min(view.y - main_top - 220.0).max(60.0);
                    let main = (Vec2::new(left, main_top), Vec2::new(left + h / REACH, main_top + h));
                    let words_top = main.1.y + 46.0;
                    (main, (words_top + 40.0 < view.y).then_some((Vec2::new(side, words_top), view.x - 2.0 * side)), None)
                } else {
                    let h = (view.y - main_top - 60.0).min((view.x * 0.5 - left) * REACH).max(60.0);
                    let main = (Vec2::new(left, main_top), Vec2::new(left + h / REACH, main_top + h));
                    let x = main.1.x + 60.0;
                    let w = (view.x - side - x).min(520.0);
                    // the factors counted, under the words, level with the foot of the chart
                    let top = main_top + 330.0;
                    let chart = (w >= 200.0 && main.1.y - top >= 110.0).then_some((Vec2::new(x + 24.0, top), Vec2::new(x + w - 10.0, main.1.y)));
                    (main, (w >= 160.0).then_some((Vec2::new(x, main_top), w)), chart)
                };
                Layout { narrow, big, text, small, tabs, title, column: view.x - side, main, smalls: None, heat: None, best: None, words, side: chart }
            }
        }
    }
}

impl HighDim {
    /// The tabs over the picture.
    fn draw_tabs(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let (at, w, h) = lay.tabs;
        for (k, (name, tab)) in TABS.iter().enumerate() {
            let min = at + Vec2::new(k as f32 * (w + 4.0), 0.0);
            if hud_tab(f, min, min + Vec2::new(w, h), name, self.tab == *tab, pointer, lay.text) {
                self.tab = *tab;
            }
        }
    }

    /// The title and the legend under it; returns nothing (the layout leaves room).
    /// The title, and under it the legend: `long` if it fits on one line, else `short`.
    /// (Whether it fits is judged by `widest`, the long legend at its widest, so that
    /// a count that gains a digit does not make the legend change halfway.)
    fn heading(f: &mut Frame, lay: &Layout, title: &[&str], long: &[(Color, &str)], short: &[(Color, &str)], widest: &[(Color, &str)]) {
        let room = lay.column - lay.title.x;
        let t = pick(f, title, room, lay.big);
        f.text_px(lay.title, t, lay.big, pal::TEXT, Align2::LEFT_TOP);
        let parts = if legend_width(f, widest, lay.small) <= room { long } else { short };
        legend(f, lay.title + Vec2::new(0.0, lay.big + 8.0), room, parts, lay.small);
    }

    // ----- angles -----

    fn draw_angles(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let n = self.n;
        let (min, max) = lay.main;
        let title_long = format!("The angle between two random directions in $n = {n}$ dimensions");
        let title_short = format!("Two random directions in $n = {n}$ dimensions");
        let title_tiny = format!("$n = {n}$ dimensions");
        let drawn = format!("{} random pairs so far", count(self.stream.pairs as usize));
        let drawn_short = format!("{} pairs", count(self.stream.pairs as usize));
        let law = r"the exact law, $\propto (1 - c^2)^{(n - 3)/2}$ at cosine $c$";
        let long = [(DRAWN, drawn.as_str()), (EXACT, law), (TYPICAL, r"$\pm 1/\sqrt{n}$, the typical size")];
        let short = [(DRAWN, drawn_short.as_str()), (EXACT, "exact law"), (TYPICAL, r"$\pm 1/\sqrt{n}$")];
        let most = format!("{} random pairs so far", count(TOTAL as usize));
        let widest = [(DRAWN, most.as_str()), (EXACT, law), (TYPICAL, r"$\pm 1/\sqrt{n}$, the typical size")];
        Self::heading(f, lay, &[&title_long, &title_short, &title_tiny], &long, &short, &widest);
        f.text_px(Vec2::new(min.x, min.y - 5.0), "how often (density)", lay.small, pal::DIM, Align2::LEFT_BOTTOM);
        self.cos_chart(f, min, max, &self.stream.cos, self.n_shown, MAIN_SKETCH, lay);

        // under it: the cosine, the angle, and what was measured
        let x_of = |c: f64| min.x + (c as f32 + 1.0) * 0.5 * (max.x - min.x);
        for (c, deg) in [(-1.0, "180°"), (-0.5, "120°"), (0.0, "90°"), (0.5, "60°"), (1.0, "0°")] {
            let label = if c == 0.0 { "0".to_string() } else { num(c, if c == -1.0 || c == 1.0 { 0 } else { 1 }) };
            f.text_px(Vec2::new(x_of(c), max.y + 4.0), label, lay.small, pal::DIM, Align2::CENTER_TOP);
            f.text_px(Vec2::new(x_of(c), max.y + 4.0 + lay.small + 3.0), deg, lay.small, pal::MUTED, Align2::CENTER_TOP);
        }
        let row = max.y + 4.0 + 2.0 * (lay.small + 3.0) + 2.0;
        let measured = if self.stream.pairs > 0 {
            let root = num(1.0 / (n as f64).sqrt(), 3);
            let long = |spread: &str| format!(r"across: the cosine (and under it the angle).   Spread of the cosines drawn: {spread}, and $1/\sqrt{{n}} = {root}$");
            let short = |spread: &str| format!(r"spread drawn {spread}, $1/\sqrt{{n}} = {root}$");
            // (whether the long one fits is judged with the spread's digits all 0: so the
            // line measured is set once for an n, not again in every frame as the spread
            // drawn changes)
            let spread = num(self.stream.spread(), 3);
            if width(f, &long("0.000"), lay.small) <= max.x - lay.title.x + 20.0 {
                long(&spread)
            } else {
                short(&spread)
            }
        } else {
            "across: the cosine of the angle (and under it the angle)".to_string()
        };
        f.text_px(Vec2::new(lay.title.x, row), measured, lay.small, pal::DIM, Align2::LEFT_TOP);

        if let Some((smin, smax)) = lay.smalls {
            self.draw_smalls(f, lay, smin, smax, pointer);
        }
        if let Some((at, side)) = lay.heat {
            self.draw_heat(f, lay, at, side, pointer);
        }
        if let Some((bmin, bmax)) = lay.best {
            self.draw_best(f, lay, bmin, bmax);
        }
    }

    /// A chart of the cosine for n: bars, the exact curve, and (for the large one,
    /// `slot` `MAIN_SKETCH`) the orange marks at ±1/√n with their names.
    #[allow(clippy::too_many_arguments)]
    fn cos_chart(&self, f: &mut Frame, min: Vec2, max: Vec2, hist: &Hist, n: f64, slot: usize, lay: &Layout) {
        let marks = slot == MAIN_SKETCH;
        let top = cos_top(n) as f32;
        let (w, h) = (max.x - min.x, max.y - min.y);
        if w < 4.0 || h < 4.0 {
            return;
        }
        let law = CosLaw::new(n);
        let x_of = |c: f64| min.x + (c as f32 + 1.0) * 0.5 * w;
        let y_of = |d: f32| max.y - (d / top).clamp(0.0, 1.0) * h;
        // faint lines at the cosines named under the chart
        for c in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            hud_line(f, Vec2::new(x_of(c), min.y), Vec2::new(x_of(c), max.y), 1.0, pal::FAINT.a(if c == 0.0 { 0.7 } else { 0.35 }));
        }
        // one height named on the left: the law's at a cosine of 0 (its peak, for n of 3
        // or more, which is always as high on the chart)
        if marks {
            let d = law.at(0.0) as f32;
            hud_line(f, Vec2::new(min.x, y_of(d)), Vec2::new(max.x, y_of(d)), 1.0, pal::FAINT.a(0.45));
            f.text_px(Vec2::new(min.x - 4.0, y_of(d)), num(d as f64, if d < 10.0 { 2 } else { 1 }), lay.small, pal::DIM, Align2::RIGHT_CENTER);
            f.text_px(Vec2::new(min.x - 4.0, max.y), "0", lay.small, pal::DIM, Align2::RIGHT_CENTER);
        }
        // the bars, and their tops as one line
        let bw = hist.width();
        let mut outline = Vec::with_capacity(2 * hist.bins() + 2);
        for i in 0..hist.bins() {
            let (a, b) = (x_of(hist.lo + i as f64 * bw), x_of(hist.lo + (i + 1) as f64 * bw));
            let y = y_of(hist.shown[i]);
            if y < max.y - 0.05 {
                hud_bar(f, Vec2::new(a, y), Vec2::new(b, max.y), DRAWN.a(0.22), DRAWN.a(0.62));
            }
            outline.push(Vec2::new(a, y));
            outline.push(Vec2::new(b, y));
        }
        hud_path(f, outline.into_iter(), 1.3, DRAWN.a(0.9));
        hud_line(f, Vec2::new(min.x, max.y), Vec2::new(max.x, max.y), 1.0, pal::DIM.a(0.8));
        // the exact law, cut off at the top of the chart (sampled short of
        // ±1 itself, where for n a hair above 3 it drops to 0 and for n of 3 it does
        // not: the ends would leap as n glides to 3)
        let steps = (w as usize).clamp(60, 600);
        let at = |s: usize| -1.0 + 2.0 * (s as f64 + 0.5) / (steps + 1) as f64;
        // (where it is above the top, it is drawn along the top edge, clear: so that it
        // has as many points whatever n is)
        let curve = || {
            (0..=steps)
                .map(|s| {
                    let c = at(s);
                    let d = law.at(c) as f32;
                    let seen = d.is_finite() && d <= top;
                    (Vec2::new(x_of(c), y_of(if seen { d } else { top })), if seen { EXACT } else { EXACT.a(0.0) })
                })
                .collect()
        };
        hud_curve_apart(f, 0, self.sketches.borrow_mut().curve(slot, n, min, max, curve).iter().copied(), 2.0);
        // where bars and curve go on above the chart: an arrow up at each end
        if n < 2.999 {
            for side in [-1.0f32, 1.0] {
                let x = x_of((side * 0.975) as f64);
                let tip = Vec2::new(x, min.y + 2.0);
                hud_path_apart(f, 1, [Vec2::new(x, min.y + 16.0), tip].into_iter(), 1.6, EXACT);
                hud_path_apart(f, 1, [tip + Vec2::new(-4.0, 6.0), tip, tip + Vec2::new(4.0, 6.0)].into_iter(), 1.6, EXACT);
            }
            if marks {
                let words = pick(f, &[r"both go on up without limit at $\pm 1$: cut off", r"cut off: no limit at $\pm 1$", "cut off"], w * 0.5, lay.small);
                f.caption_px(Vec2::new(max.x - 18.0, min.y + 4.0), words, lay.small, EXACT, Align2::RIGHT_TOP);
            }
        }
        if !marks {
            return;
        }
        // the typical size, ±1/√n
        let s = 1.0 / n.sqrt();
        let (xl, xr) = (x_of(-s), x_of(s));
        for x in [xl, xr] {
            hud_dashed(f, Vec2::new(x, min.y + 20.0), Vec2::new(x, max.y), 1.6, TYPICAL);
        }
        let value = num(s, if s < 0.1 { 3 } else { 2 });
        let both = format!(r"$\pm 1/\sqrt{{n}} = \pm {value}$");
        let (right, left) = (format!(r"$+1/\sqrt{{n}} = {value}$"), r"$-1/\sqrt{n}$");
        if xr - xl > width(f, &right, lay.small) * 2.0 + 24.0 {
            f.caption_px(Vec2::new(xr + 4.0, min.y + 20.0), right, lay.small, TYPICAL, Align2::LEFT_TOP);
            f.caption_px(Vec2::new(xl - 4.0, min.y + 20.0), left, lay.small, TYPICAL, Align2::RIGHT_TOP);
        } else {
            f.caption_px(Vec2::new(xr + 4.0, min.y + 20.0), both, lay.small, TYPICAL, Align2::LEFT_TOP);
        }
    }

    /// The same for n = 2, 3, 10, 100, side by side: click one to show it large.
    fn draw_smalls(&mut self, f: &mut Frame, lay: &Layout, min: Vec2, max: Vec2, pointer: Option<Vec2>) {
        let gap = if lay.narrow { 10.0 } else { 18.0 };
        let count = SMALL_N.len() as f32;
        let w = (max.x - min.x - gap * (count - 1.0)) / count;
        if w < 20.0 {
            return;
        }
        let head = pick(f, &["The same for other $n$, side by side (click one to show it above):", "The same for other $n$ (click one):", "Other $n$ (click):"], max.x - min.x, lay.small);
        f.text_px(Vec2::new(lay.title.x, min.y - 2.0 * (lay.small + 7.0)), head, lay.small, pal::DIM, Align2::LEFT_TOP);
        for (k, &n) in SMALL_N.iter().enumerate() {
            let a = Vec2::new(min.x + k as f32 * (w + gap), min.y);
            let b = Vec2::new(a.x + w, max.y);
            let chosen = self.n == n;
            let frame = (a - Vec2::new(4.0, lay.small + 8.0), b + Vec2::splat(4.0));
            let over = pointer.is_some_and(|p| p.cmpge(frame.0).all() && p.cmple(frame.1).all());
            if over {
                hud_rect(f, frame.0, frame.1, Color::hex(0x1A2133));
                f.handles.claim_clicks();
                if f.ptr.clicked {
                    self.set_n(n);
                }
            }
            let hist = &self.smalls[k];
            self.cos_chart(f, a, b, hist, n as f64, SMALL_SKETCH + k, lay);
            if chosen {
                hud_outline(f, frame.0, frame.1, 1.4, CHOSEN);
            }
            f.text_px(Vec2::new(a.x, a.y - 3.0), format!("$n = {n}$"), lay.small, if chosen { CHOSEN } else { pal::TEXT }, Align2::LEFT_BOTTOM);
            if !lay.narrow {
                let says = [r"piles up at $\pm 1$", "level: all alike", "a dome", "a narrow bell"][k];
                f.text_px(Vec2::new(a.x + w * 0.5, b.y + 8.0), pick(f, &[says, ""], w + gap, lay.small), lay.small, pal::DIM, Align2::CENTER_TOP);
            }
        }
    }

    /// The dot products among m random directions, as a heat map.
    fn draw_heat(&mut self, f: &mut Frame, lay: &Layout, at: Vec2, side: f32, pointer: Option<Vec2>) {
        let (n, m) = (self.many.n, self.many.m);
        let many = if m == 1 { "direction" } else { "directions" };
        let title = format!("$m = {m}$ random {many} in $n = {n}$: every dot product");
        let short = format!("{m} {many}: every dot product");
        let room = f.view_size().x - at.x - 8.0;
        f.text_px(at - Vec2::new(0.0, 34.0), pick(f, &[&title, &short], room, lay.text), lay.text, pal::TEXT, Align2::LEFT_TOP);
        let sub = pick(f, &["row $i$, column $j$: the cosine between directions $i$ and $j$", "cell $i, j$: cosine of $i$ and $j$"], side + 60.0, lay.small);
        f.text_px(at - Vec2::new(0.0, 17.0), sub, lay.small, pal::DIM, Align2::LEFT_TOP);
        // the picture, painted as the dot products are worked out (a new picture, of
        // another size, holds none of them)
        if self.heat.as_ref().is_none_or(|i| i.size() != [m as u32, m as u32]) {
            self.many.painted = None;
        }
        if self.many.painted != Some(self.many.rows) {
            let image = self.heat.get_or_insert_with(|| Image::new(f.gfx, m as u32, m as u32));
            image.ensure_size(f.gfx, m as u32, m as u32);
            if image.size() == [m as u32, m as u32] {
                // (only the new directions' rows and columns need sending to the GPU)
                match self.many.unpainted() {
                    Some(parts) => self.many.paint(image.pixels_mut_in(&parts)),
                    None => self.many.paint(image.pixels_mut()),
                }
            }
        }
        let max = at + Vec2::splat(side);
        if let Some(image) = &mut self.heat {
            if image.size() == [m as u32, m as u32] {
                let (lo, hi) = world_rect(f, at, max);
                f.image(image, lo, hi, 0.0);
            }
        }
        // (cells smaller than a pixel are shown one in so many: the diagonal, drawn so,
        // would come out dotted)
        if m as f32 > side && self.many.rows > 0 {
            let done = self.many.rows as f32 / m as f32 * side;
            hud_path_apart(f, 2, [at, at + Vec2::splat(done)].into_iter(), 1.2, HEAT.table()[heat_index(1.0)]);
        }
        hud_outline(f, at - Vec2::ONE, max + Vec2::ONE, 1.0, pal::DIM.a(0.7));
        // the largest, framed
        if let Some(ring) = self.ring {
            // (a square round the cell, a few pixels clear of it, on a dark edge that
            // keeps it seen on bright cells)
            let c = at + ring * side;
            let r = (0.5 * side / m as f32 + 3.0).max(6.0);
            let corners = [Vec2::new(-r, -r), Vec2::new(r, -r), Vec2::new(r, r), Vec2::new(-r, r), Vec2::new(-r, -r)];
            hud_path_apart(f, 3, corners.iter().map(|d| c + *d), 4.5, Color::hex(0x05070C));
            hud_path_apart(f, 3, corners.iter().map(|d| c + *d), 2.0, LARGEST);
        }
        // the cell pointed at, read out beside the pointer
        if let Some(p) = pointer.filter(|p| p.cmpge(at).all() && p.cmplt(max).all()) {
            let cell = ((p - at) / side * m as f32).floor();
            let (i, j) = ((cell.y.max(0.0) as usize).min(m - 1), (cell.x.max(0.0) as usize).min(m - 1));
            if i < self.many.rows && j < self.many.rows {
                let c = dot(&self.many.vecs[i * n..(i + 1) * n], &self.many.vecs[j * n..(j + 1) * n]).clamp(-1.0, 1.0) as f64;
                let text = if i == j {
                    format!("direction {} with itself: cosine 1", i + 1)
                } else {
                    format!("directions {} and {}: cosine {}, {:.1}° apart", i + 1, j + 1, num(c, 3), c.acos().to_degrees())
                };
                let left = p.x + 14.0 + text_width(&text, lay.small) > f.view_size().x - 6.0;
                let place = if left { Vec2::new(p.x - 14.0, p.y - 14.0) } else { Vec2::new(p.x + 14.0, p.y - 14.0) };
                f.caption_px(place, text, lay.small, pal::TEXT, if left { Align2::RIGHT_BOTTOM } else { Align2::LEFT_BOTTOM });
            }
        }
        // the colours, and the largest
        let y = max.y + 10.0;
        let ramp = (side * 0.6).min(160.0);
        let table = HEAT.table();
        for i in 0..32 {
            let (a, b) = (i as f32 / 32.0, (i + 1) as f32 / 32.0);
            hud_rect(f, Vec2::new(at.x + ramp * a, y), Vec2::new(at.x + ramp * b + 0.5, y + 7.0), table[table_index((a + b) * 0.5)]);
        }
        for (t, s) in [(0.0, "−1"), (0.5, "0"), (1.0, "+1")] {
            f.text_px(Vec2::new(at.x + ramp * t, y + 9.0), s, lay.small, pal::DIM, Align2::CENTER_TOP);
        }
        if !lay.narrow {
            f.text_px(Vec2::new(at.x + ramp + 10.0, y + 3.5), "(the diagonal: each with itself)", lay.small, pal::DIM, Align2::LEFT_CENTER);
        }
        let line1 = match (self.many.best, self.many.complete()) {
            (None, _) if m == 1 => "one direction: no pair to compare".to_string(),
            (None, _) => "working out the dot products".to_string(),
            (Some((c, _, _)), true) if self.many.pairs() == 1 => format!(r"$|\text{{cosine}}|$ of the one pair (framed): {}", num(c.abs() as f64, 2)),
            (Some((c, _, _)), true) => format!(r"largest $|\text{{cosine}}|$ of the {} pairs (framed): {}", count(self.many.pairs()), num(c.abs() as f64, 2)),
            (Some((c, _, _)), false) => format!("largest so far, of {} pairs (framed): {}", count(self.many.pairs()), num(c.abs() as f64, 2)),
        };
        let line2 = match self.many.best {
            Some((c, _, _)) => {
                let a = angle_apart(c.abs() as f64);
                if a < 0.1 {
                    "so two of them lie along one line, to within 0.1°".to_string()
                } else if a < 5.0 {
                    format!("so two of them lie nearly along one line ({a:.1}° off it)")
                } else {
                    format!("so every two are between {a:.1}° and {:.1}° apart", 180.0 - a)
                }
            }
            None => String::new(),
        };
        let wide = lay.column.max(side + 120.0);
        f.text_px(Vec2::new(at.x, y + 28.0), pick(f, &[&line1], wide, lay.text), lay.text, LARGEST, Align2::LEFT_TOP);
        f.text_px(Vec2::new(at.x, y + 28.0 + lay.text + 4.0), line2, lay.small, pal::DIM, Align2::LEFT_TOP);
    }

    /// The largest |cosine| among the first r directions, against r.
    fn draw_best(&self, f: &mut Frame, lay: &Layout, min: Vec2, max: Vec2) {
        let m = self.many.m;
        let (w, h) = (max.x - min.x, max.y - min.y);
        if m < 3 || w < 40.0 || h < 30.0 {
            return;
        }
        let title = pick(f, &[r"largest $|\text{cosine}|$ among the first $r$ directions", r"largest $|\text{cos}|$, first $r$"], w + 30.0, lay.small);
        f.text_px(Vec2::new(min.x - 30.0, min.y - 18.0), title, lay.small, pal::TEXT, Align2::LEFT_TOP);
        let span = (m as f32).ln() - 2f32.ln();
        let x_of = |r: f32| min.x + ((r.ln() - 2f32.ln()) / span).clamp(0.0, 1.0) * w;
        let y_of = |v: f32| max.y - v.clamp(0.0, 1.0) * h;
        for (v, s) in [(0.0, "0"), (0.5, "0.5"), (1.0, "1")] {
            hud_line(f, Vec2::new(min.x, y_of(v)), Vec2::new(max.x, y_of(v)), 1.0, if v == 0.0 { pal::DIM.a(0.8) } else { pal::FAINT.a(0.5) });
            f.text_px(Vec2::new(min.x - 5.0, y_of(v)), s, lay.small, pal::DIM, Align2::RIGHT_CENTER);
        }
        for r in [2usize, 10, 100, 1000] {
            if r <= m {
                f.text_px(Vec2::new(x_of(r as f32), max.y + 3.0), r.to_string(), lay.small, pal::DIM, Align2::CENTER_TOP);
            }
        }
        f.text_px(Vec2::new(max.x, max.y + 3.0 + lay.small + 2.0), "$r$ →", lay.small, pal::DIM, Align2::RIGHT_TOP);
        // r = n: up to there, directions could all be exactly at right angles
        let n = self.many.n;
        if n >= 2 && n <= m {
            let x = x_of(n as f32);
            hud_dashed(f, Vec2::new(x, min.y), Vec2::new(x, max.y), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x + 4.0, max.y - 3.0), "$r = n$", lay.small, pal::DIM, Align2::LEFT_BOTTOM);
        }
        let by_r = &self.many.by_r;
        if by_r.len() >= 2 {
            let pts = (1..by_r.len()).map(|i| Vec2::new(x_of((i + 1) as f32), y_of(by_r[i])));
            hud_path_apart(f, 4, pts, 2.0, LARGEST);
        }
    }

    // ----- lengths -----

    fn draw_lengths(&self, f: &mut Frame, lay: &Layout) {
        let n = self.n;
        let (min, max) = lay.main;
        let (w, h) = (max.x - min.x, max.y - min.y);
        let title_long = format!(r"The length of a random vector: $n = {n}$ standard normal numbers $g_1, \ldots, g_n$");
        let title_short = format!("The length of $n = {n}$ random numbers");
        let drawn = format!("{} random vectors so far", count(self.stream.pairs as usize));
        let long = [(DRAWN, drawn.as_str()), (EXACT, "the exact law (faint: other $n$)"), (TYPICAL, r"$\sqrt{n}$, and $1/\sqrt{2}$ either side")];
        let short = [(DRAWN, drawn.as_str()), (EXACT, "exact law"), (TYPICAL, r"$\sqrt{n} \pm 1/\sqrt{2}$")];
        let most = format!("{} random vectors so far", count(TOTAL as usize));
        let widest = [(DRAWN, most.as_str()), (EXACT, "the exact law (faint: other $n$)"), (TYPICAL, r"$\sqrt{n}$, and $1/\sqrt{2}$ either side")];
        Self::heading(f, lay, &[&title_long, &title_short], &long, &short, &widest);
        if w < 20.0 || h < 20.0 {
            return;
        }
        f.text_px(Vec2::new(min.x, min.y - 5.0), "how often (density)", lay.small, pal::DIM, Align2::LEFT_BOTTOM);
        let top = 0.72f32;
        let x_of = |r: f64| min.x + (r / LEN_MAX) as f32 * w;
        let y_of = |d: f32| max.y - (d / top).clamp(0.0, 1.0) * h;
        for r in (0..=30).step_by(5) {
            let x = x_of(r as f64);
            hud_line(f, Vec2::new(x, min.y), Vec2::new(x, max.y), 1.0, pal::FAINT.a(0.35));
            f.text_px(Vec2::new(x, max.y + 4.0), r.to_string(), lay.small, pal::DIM, Align2::CENTER_TOP);
        }
        let axis = pick(f, &[r"across: the length $\sqrt{g_1^2 + \cdots + g_n^2}$", "length"], w, lay.small);
        f.text_px(Vec2::new(max.x, max.y + 6.0 + lay.small), axis, lay.small, pal::DIM, Align2::RIGHT_TOP);
        // √n named here; it and 1/√2 either side of it are marked over the bars, below
        let root = (self.n_shown).sqrt();
        let (a, b) = (x_of(root - 0.5f64.sqrt()), x_of(root + 0.5f64.sqrt()));
        let mark = format!(r"$\sqrt{{n}} = {}$", num(root, if root < 10.0 { 2 } else { 1 }));
        let right = x_of(root) + 4.0 + width(f, &mark, lay.small) < max.x;
        f.text_px(Vec2::new(if right { x_of(root) + 4.0 } else { x_of(root) - 4.0 }, min.y + 18.0), mark, lay.small, TYPICAL, if right { Align2::LEFT_TOP } else { Align2::RIGHT_TOP });
        // the bars
        let hist = &self.stream.len;
        let bw = hist.width();
        let mut outline = Vec::with_capacity(2 * hist.bins());
        for i in 0..hist.bins() {
            let (a, b) = (x_of(hist.lo + i as f64 * bw), x_of(hist.lo + (i + 1) as f64 * bw));
            let y = y_of(hist.shown[i]);
            if y < max.y - 0.05 {
                hud_bar(f, Vec2::new(a, y), Vec2::new(b, max.y), DRAWN.a(0.22), DRAWN.a(0.62));
            }
            outline.push(Vec2::new(a, y));
            outline.push(Vec2::new(b, y));
        }
        hud_path(f, outline.into_iter(), 1.3, DRAWN.a(0.9));
        hud_line(f, Vec2::new(min.x, max.y), Vec2::new(max.x, max.y), 1.0, pal::DIM.a(0.8));
        // the exact laws: the others faint, named over their peaks
        let steps = (w as usize).clamp(80, 900);
        let mut sketches = self.sketches.borrow_mut();
        let mut curve = |f: &mut Frame, slot: usize, n: f64, width: f32, color: Color| {
            let law = LenLaw::new(n);
            let make = || (0..=steps).map(|s| LEN_MAX * s as f64 / steps as f64).map(|r| (Vec2::new(x_of(r), y_of(law.at(r) as f32)), color)).collect();
            hud_path(f, sketches.curve(slot, n, min, max, make).iter().map(|p| p.0), width, color);
            law
        };
        for (k, other) in LEN_REF.into_iter().enumerate() {
            if other == n {
                continue;
            }
            let law = curve(f, LEN_SKETCH + k, other as f64, 1.2, EXACT.a(0.35));
            let peak = ((other - 1) as f64).sqrt();
            let p = Vec2::new(x_of(peak), y_of(law.at(peak) as f32));
            // (the first two are close together: their names go to either side)
            let (nudge, align) = match other {
                2 => (Vec2::new(2.0, -4.0), Align2::RIGHT_BOTTOM),
                10 => (Vec2::new(-2.0, -4.0), Align2::LEFT_BOTTOM),
                _ => (Vec2::new(0.0, -4.0), Align2::CENTER_BOTTOM),
            };
            f.text_px(p + nudge, format!("$n = {other}$"), lay.small, pal::DIM, align);
        }
        let law = curve(f, LEN_SKETCH + LEN_REF.len(), self.n_shown, 2.0, EXACT);
        // half way up the curve: a bar 1/√2 either side of √n
        let y = y_of(0.5 * law.at(root) as f32);
        hud_dashed(f, Vec2::new(x_of(root), min.y + 18.0), Vec2::new(x_of(root), max.y), 1.6, TYPICAL);
        hud_line(f, Vec2::new(a, y), Vec2::new(b, y), 2.0, TYPICAL);
        for x in [a, b] {
            hud_line(f, Vec2::new(x, y - 5.0), Vec2::new(x, y + 5.0), 2.0, TYPICAL);
        }
    }

    /// Under the lengths: why the angles come out as they do.
    fn draw_why(&self, f: &mut Frame, lay: &Layout) {
        let Some((at, width)) = lay.words else { return };
        let n = self.n as f64;
        let s = &self.stream;
        let measured = if s.pairs > 0 {
            format!(r"Measured here: average length {}, $\sqrt{{n}} = {}$; spread {} either way, $1/\sqrt{{2}} = 0.707$.", num(s.mean_length(), 2), num(n.sqrt(), 2), num(s.length_spread(), 3))
        } else {
            String::new()
        };
        let dots = if s.pairs > 0 { format!(r" (measured: $\pm {}$, and $\sqrt{{n}} = {}$)", num(s.dot_spread(), 1), num(n.sqrt(), 1)) } else { String::new() };
        let lines = [
            (pal::TEXT, "Why two random directions are nearly at right angles:".to_string()),
            (pal::DIM, r"• each vector's length is close to $\sqrt{n}$, as the chart shows: within about $1/\sqrt{2}$ either way, whatever $n$ is, so ever closer in proportion.".to_string()),
            (pal::DIM, format!(r"• their dot product $g \cdot h = g_1 h_1 + \cdots + g_n h_n$ adds $n$ terms of random sign, each about 1 in size; like a walk of $n$ random steps it ends up about $\sqrt{{n}}$ from $0$, either way{dots}.")),
            (pal::DIM, r"• so the cosine $g \cdot h / (|g|\,|h|)$ is about $\pm\sqrt{n} / (\sqrt{n} \cdot \sqrt{n}) = \pm 1/\sqrt{n}$.".to_string()),
            (DRAWN, measured),
        ];
        let mut y = at.y;
        let mut broken = self.broken.borrow_mut();
        let places = broken.len().max(lines.len());
        broken.resize_with(places, || None);
        for ((color, text), kept) in lines.into_iter().zip(broken.iter_mut()) {
            y += wrap_px(f, Vec2::new(at.x, y), width, &text, lay.small, color, kept) + 3.0;
        }
    }

    // ----- projection -----

    fn draw_projection(&self, f: &mut Frame, lay: &Layout) {
        let (big_n, k) = (self.big_n, self.k);
        let (min, max) = lay.main;
        let pairs = self.proj.pairs.len();
        let title_long = format!("{POINTS} points in $N = {big_n}$ dimensions, pushed into $k = {k}$ by a random matrix");
        let title_short = format!("{POINTS} points: $N = {big_n}$ dimensions into $k = {k}$");
        let each = format!("each dot: one of the {pairs} pairs");
        let long = [(DRAWN, each.as_str()), (EXACT, r"$\text{after} = \text{before}$"), (TYPICAL, r"$1 \pm 1/\sqrt{2k}$"), (pal::DIM, "the extremes")];
        let short = [(DRAWN, "a pair"), (EXACT, r"$\text{after} = \text{before}$"), (TYPICAL, r"$1 \pm 1/\sqrt{2k}$"), (pal::DIM, "extremes")];
        Self::heading(f, lay, &[&title_long, &title_short], &long, &short, &long);
        let (w, h) = (max.x - min.x, max.y - min.y);
        if w < 30.0 || h < 30.0 {
            return;
        }
        // the same scale both ways: across a little beyond the farthest pair, up twice that
        let span_x = self.span.max(1e-3);
        let span_y = span_x * REACH;
        let x_of = |v: f32| min.x + (v / span_x).clamp(0.0, 1.0) * w;
        let y_of = |v: f32| max.y - (v / span_y).clamp(0.0, 1.0) * h;
        let step = nice_step(span_x / 4.0);
        let mut v = 0.0;
        while v <= span_x * 1.0001 {
            hud_line(f, Vec2::new(x_of(v), min.y), Vec2::new(x_of(v), max.y), 1.0, pal::FAINT.a(0.35));
            f.text_px(Vec2::new(x_of(v), max.y + 4.0), fmt_tick(v, step), lay.small, pal::DIM, Align2::CENTER_TOP);
            v += step;
        }
        let step_y = nice_step(span_y / 5.0);
        let mut v = 0.0;
        while v <= span_y * 1.0001 {
            hud_line(f, Vec2::new(min.x, y_of(v)), Vec2::new(max.x, y_of(v)), 1.0, pal::FAINT.a(0.35));
            f.text_px(Vec2::new(min.x - 5.0, y_of(v)), fmt_tick(v, step_y), lay.small, pal::DIM, Align2::RIGHT_CENTER);
            v += step_y;
        }
        hud_line(f, Vec2::new(min.x, max.y), Vec2::new(max.x, max.y), 1.0, pal::DIM.a(0.8));
        hud_line(f, Vec2::new(min.x, min.y), Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let across = format!("distance before, in $N = {big_n}$ dimensions →");
        f.text_px(Vec2::new(max.x, max.y + 6.0 + lay.small), pick(f, &[&across, "before →"], w, lay.small), lay.small, pal::DIM, Align2::RIGHT_TOP);
        let up = format!("↑ distance after, in $k = {k}$");
        f.text_px(Vec2::new(min.x - 30.0, min.y - 5.0), up, lay.small, pal::DIM, Align2::LEFT_BOTTOM);
        // a line through the origin with slope `s`, as far as the chart goes
        let ray = |s: f32| {
            let end = if s * span_x <= span_y { Vec2::new(span_x, s * span_x) } else { Vec2::new(span_y / s.max(1e-6), span_y) };
            (Vec2::new(x_of(0.0), y_of(0.0)), Vec2::new(x_of(end.x), y_of(end.y)))
        };
        // the extremes and the typical change, then the dots, then the line they lie along
        if let Some([lo, hi, ..]) = self.proj.summary() {
            for (s, name) in [(lo as f32, format!(r"$\times {}$", num(lo, 2))), (hi as f32, format!(r"$\times {}$", num(hi, 2)))] {
                let (a, b) = ray(s);
                hud_dashed(f, a, b, 1.2, pal::DIM);
                f.text_px(b + Vec2::new(4.0, if s * span_x <= span_y { 0.0 } else { -2.0 }), name, lay.small, pal::DIM, if s * span_x <= span_y { Align2::LEFT_CENTER } else { Align2::CENTER_BOTTOM });
            }
        }
        let typical = (1.0 / (2.0 * self.k_shown).sqrt()) as f32;
        for s in [1.0 - typical, 1.0 + typical] {
            if s > 0.0 {
                let (a, b) = ray(s);
                hud_dashed(f, a, b, 1.4, TYPICAL);
            }
        }
        let size = if pairs > 600 { 3.4 } else { 4.5 };
        // (a pair beyond the top is left out of sight: drawn there, but clear, so
        // that the dots stay as many from frame to frame)
        let placed: Vec<(Vec3, bool)> = self.dots.iter().map(|d| (f.hud_at(2, Vec2::new(x_of(d.x), y_of(d.y))), d.y <= span_y)).collect();
        let batch = f.dl.points_with(PointKind::Overlay, PointStyle::default());
        for (at, seen) in placed {
            batch.push(at, size, DRAWN.a(if seen { 0.75 } else { 0.0 }));
        }
        let (a, b) = ray(1.0);
        hud_line(f, a, b, 1.6, EXACT);
        let off = self.dots.iter().filter(|d| d.y > span_y).count();
        if off > 0 {
            f.text_px(Vec2::new(max.x - 4.0, min.y + 4.0), format!("{off} beyond the top"), lay.small, DRAWN, Align2::RIGHT_TOP);
        }
    }

    fn draw_projection_words(&self, f: &mut Frame, lay: &Layout) {
        let Some((at, width)) = lay.words else { return };
        let ratios = self.proj.ratios();
        let k = self.k as f64;
        let mut lines: Vec<(Color, String)> = vec![(pal::TEXT, format!("Distance after ÷ distance before, for all {} pairs:", count(self.proj.pairs.len())))];
        if let Some([lo, hi, q1, q2, sd]) = self.proj.summary() {
            if !self.proj.done() {
                lines.push((pal::DIM, format!("(working: {} of the $k = {}$ rows of the matrix so far)", self.proj.used, self.k)));
            }
            lines.push((pal::DIM, format!("• between {} and {} (the grey lines)", num(lo, 2), num(hi, 2))));
            lines.push((pal::DIM, format!("• the middle 95 % between {} and {}", num(q1, 2), num(q2, 2))));
            lines.push((TYPICAL, format!(r"• spread (standard deviation) {}; and $1/\sqrt{{2k}} = {}$", num(sd, 3), num(1.0 / (2.0 * k).sqrt(), 3))));
            let inside = ratios.iter().filter(|r| (**r - 1.0).abs() < 1.0 / (2.0 * k).sqrt()).count();
            lines.push((TYPICAL, format!("• {:.0} % of them inside the orange lines", 100.0 * inside as f64 / ratios.len().max(1) as f64)));
        }
        lines.push((
            pal::DIM,
            r#"Why: for any one pair, the matrix turns the difference between the two points into $k$ random numbers whose spread is its length divided by $\sqrt{k}$. So the factor is the length of $k$ standard normal numbers divided by $\sqrt{k}$ (the "lengths" view, in $k$ dimensions): close to 1, within about $1/\sqrt{2k}$. $N$ does not come into it."#.to_string(),
        ));
        if self.k < 4 {
            lines.push((pal::DIM, "With so few dimensions left, the factor is anything from nearly 0 to 2 or more: points far apart can land on top of each other.".to_string()));
        }
        let mut y = at.y;
        let mut broken = self.broken.borrow_mut();
        let places = broken.len().max(lines.len());
        broken.resize_with(places, || None);
        for ((color, text), kept) in lines.into_iter().zip(broken.iter_mut()) {
            y += wrap_px(f, Vec2::new(at.x, y), width, &text, lay.small + if color == pal::TEXT { 1.0 } else { 0.0 }, color, kept) + 4.0;
        }
    }
}

impl HighDim {
    /// Beside the projection: the factors after ÷ before, counted, with the exact law
    /// of one (the length of k standard normal numbers divided by √k).
    fn draw_factors(&self, f: &mut Frame, lay: &Layout) {
        let Some((min, max)) = lay.side else { return };
        let (w, h) = (max.x - min.x, max.y - min.y);
        let k = self.k_shown;
        f.text_px(Vec2::new(min.x - 24.0, min.y - 22.0), "The factors after ÷ before, counted", lay.text, pal::TEXT, Align2::LEFT_TOP);
        let hist = &self.factors;
        let bins = hist.bins();
        let beyond = self.proj.ratios().iter().filter(|r| **r >= 2.0).count();
        let length = LenLaw::new(k);
        let law = |r: f64| length.at(r * k.sqrt()) * k.sqrt();
        // (its peak: at √((k − 1)/k), or for k = 1 at 0, where the law is not 0 but √(2/π))
        let peak = law(((k - 1.0).max(0.0) / k).sqrt().max(1e-9));
        let top = (1.15 * peak) as f32;
        let x_of = |r: f64| min.x + (r / 2.0) as f32 * w;
        let y_of = |d: f32| max.y - (d / top).clamp(0.0, 1.0) * h;
        for r in [0.0, 0.5, 1.0, 1.5, 2.0] {
            hud_line(f, Vec2::new(x_of(r), min.y), Vec2::new(x_of(r), max.y), 1.0, pal::FAINT.a(if r == 1.0 { 0.8 } else { 0.35 }));
            f.text_px(Vec2::new(x_of(r), max.y + 4.0), num(r, 1), lay.small, pal::DIM, Align2::CENTER_TOP);
        }
        let mut outline = Vec::with_capacity(2 * bins);
        for i in 0..bins {
            let (a, b) = (x_of(i as f64 * hist.width()), x_of((i + 1) as f64 * hist.width()));
            let y = y_of(hist.shown[i]);
            if y < max.y - 0.05 {
                hud_bar(f, Vec2::new(a, y), Vec2::new(b, max.y), DRAWN.a(0.22), DRAWN.a(0.62));
            }
            outline.push(Vec2::new(a, y));
            outline.push(Vec2::new(b, y));
        }
        hud_path(f, outline.into_iter(), 1.3, DRAWN.a(0.9));
        hud_line(f, Vec2::new(min.x, max.y), Vec2::new(max.x, max.y), 1.0, pal::DIM.a(0.8));
        let steps = (w as usize).clamp(60, 500);
        let make = || (0..=steps).map(|s| 2.0 * (s as f64 + 0.5) / (steps + 1) as f64).map(|r| (Vec2::new(x_of(r), y_of(law(r) as f32)), EXACT)).collect();
        hud_path(f, self.sketches.borrow_mut().curve(FACTOR_SKETCH, k, min, max, make).iter().map(|p| p.0), 2.0, EXACT);
        let s = 1.0 / (2.0 * k).sqrt();
        for r in [1.0 - s, 1.0 + s] {
            if r > 0.0 {
                hud_dashed(f, Vec2::new(x_of(r), min.y), Vec2::new(x_of(r), max.y), 1.4, TYPICAL);
            }
        }
        let numbers = if self.k == 1 { "1 normal number".to_string() } else { format!("{} normal numbers", self.k) };
        let axis = format!(r"white: the exact law of one factor, the length of {numbers} ÷ $\sqrt{{k}}$");
        f.text_px(Vec2::new(min.x - 24.0, max.y + 6.0 + lay.small), pick(f, &[&axis, "white: the exact law of one factor"], w + 24.0, lay.small), lay.small, pal::DIM, Align2::LEFT_TOP);
        if beyond > 0 {
            f.text_px(Vec2::new(max.x, min.y + 2.0), format!("{beyond} beyond 2 →"), lay.small, DRAWN, Align2::RIGHT_TOP);
        }
    }
}

/// The words of a text, a formula between dollars one word however many spaces it has:
/// a line is never broken inside one.
fn words(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let (mut words, mut start, mut inside, mut i) = (Vec::new(), 0, false, 0);
    while i < bytes.len() {
        match bytes[i] {
            // (a `\$` is a dollar sign, not the end of a formula)
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

/// Text broken into lines no wider than `room` logical pixels, each word as wide as
/// `measure` says (none if it cannot say).
fn lines_of(text: &str, room: f32, size: f32, measure: impl Fn(&str) -> Option<f32>) -> Option<Vec<String>> {
    let space = size * 0.47;
    let mut lines: Vec<String> = vec![String::new()];
    // (the last line's width, word by word: a formula is measured once, not again with
    // every word that follows it on its line)
    let mut wide = 0.0;
    for word in words(text) {
        let w = measure(word)?;
        let last = lines.last_mut()?;
        if last.is_empty() {
            last.push_str(word);
            wide = w;
        } else if wide + w + space > room {
            lines.push(word.to_string());
            wide = w;
        } else {
            last.push(' ');
            last.push_str(word);
            wide += space + w;
        }
    }
    Some(lines)
}

/// A text as last broken into lines by `wrap_px`, kept with the text, the room and the
/// size it was broken for. (Its formulas are measured exactly, which is work for every
/// frame if not kept.)
type Broken = Option<((String, f32, f32), Vec<String>)>;

/// Text broken into lines no wider than `room` logical pixels, its first line's
/// top-left corner at `at`, the lines kept in `kept` for as long as the text, the room
/// and the size are the same. Returns the height taken.
#[allow(clippy::too_many_arguments)]
fn wrap_px(f: &mut Frame, at: Vec2, room: f32, text: &str, size: f32, color: Color, kept: &mut Broken) -> f32 {
    if text.is_empty() {
        return 0.0;
    }
    if kept.as_ref().is_none_or(|(of, _)| (of.0.as_str(), of.1, of.2) != (text, room, size)) {
        // (none kept before the formulas' fonts are in place, the first frame)
        *kept = lines_of(text, room, size, |word| measured(f, word, size)).map(|lines| ((text.to_string(), room, size), lines));
    }
    let lines = match kept {
        Some((_, lines)) => lines.clone(),
        None => lines_of(text, room, size, |word| Some(width(f, word, size))).unwrap_or_default(),
    };
    let line = size + 4.0;
    for (i, l) in lines.iter().enumerate() {
        f.text_px(at + Vec2::new(0.0, i as f32 * line), l.as_str(), size, color, Align2::LEFT_TOP);
    }
    lines.len() as f32 * line
}

impl Scene for HighDim {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    /// Moving while pairs are being drawn and the bars rise to them, while the heat
    /// map fills in, while the projection is worked out, and while anything glides.
    fn animated(&self) -> bool {
        let n_gliding = self.n_shown != self.n as f64;
        match self.tab {
            Tab::Angles => self.bars_moving() || !self.many.complete() || self.ring != self.ring_goal() || n_gliding,
            Tab::Lengths => self.bars_moving() || n_gliding,
            Tab::Projection => !self.proj.done() || self.dots_moving() || !self.factors.caught_up() || self.k_shown != self.k as f64,
        }
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("Click a tab over the picture to change what it shows, and a small chart to show it large. Point at a cell of the heat map to read it.")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Pick two directions at random in $n$ dimensions and look at the angle between them. In the plane every angle is equally likely. In many dimensions the two are nearly always almost at right angles: the cosine of the angle (their dot product, as vectors of length 1) is spread round $0$ with a typical size of $1/\sqrt{n}$. So $n$ dimensions have room for far more than $n$ directions that are nearly at right angles to one another, which is what superposition and compressed sensing live on. The same reasoning makes lengths nearly fixed, and lets a random projection squash points into far fewer dimensions with every distance nearly kept.");

        ui::section(ui, "Things to try");
        for (label, look_for, setup) in TRY {
            if ui::experiment(ui, label, look_for) {
                self.apply(setup);
            }
        }

        ui::section(ui, "What the picture shows");
        let mut tab = TABS.iter().position(|t| t.1 == self.tab).unwrap_or(0);
        let names: Vec<&str> = TABS.iter().map(|t| t.0).collect();
        if ui::choice(ui, &names, &mut tab) {
            self.tab = TABS[tab].1;
        }
        ui::note(ui, "Angles: the cosine between random directions, two at a time and many at once. Lengths: how long a random vector is. Projection: points pushed into fewer dimensions. The same tabs are over the picture.");

        ui::section(ui, "Directions");
        let mut n = self.n;
        if ui::slider_int_log(ui, "dimensions $n$", &mut n, 2..=N_MAX) {
            self.set_n(n);
        }
        let mut m = self.m;
        if ui::slider_int_log(ui, "directions in the heat map, $m$", &mut m, 1..=M_MAX) {
            self.set_m(m);
        }
        ui::note(ui, "A random direction: $n$ independent standard normal numbers, divided by their length, so that no direction is more likely than another. Each new $n$ draws everything afresh.");

        ui::section(ui, "Projection");
        let (mut big_n, mut k) = (self.big_n, self.k);
        let mut changed = ui::slider_int_log(ui, "dimensions before, $N$", &mut big_n, 2..=N_MAX);
        changed |= ui::slider_int_log(ui, "dimensions after, $k$", &mut k, 1..=big_n.max(1));
        if changed {
            self.set_projection(big_n, k);
        }
        ui::note(ui, r"The points: 50 of them, each in a random direction in $N$ dimensions, at distances from the centre spread evenly from 1 to 6. The matrix: $k \times N$ standard normal numbers, divided by $\sqrt{k}$. One more $k$ adds a row and keeps the others.");

        ui::section(ui, "Reading the picture");
        let table = HEAT.table();
        ui::key(
            ui,
            &[
                (DRAWN, "cyan bars and dots: what the random draws gave"),
                (EXACT, r"white curve: the exact law they are drawn from (in projection: $\text{distance after} = \text{distance before}$)"),
                (TYPICAL, r"orange: the typical size, $\pm 1/\sqrt{n}$ for a cosine, $\sqrt{n} \pm 1/\sqrt{2}$ for a length, $1 \pm 1/\sqrt{2k}$ for a change of distance"),
                (LARGEST, r"green: the largest $|\text{cosine}|$ among $m$ directions, framed in the heat map"),
                (CHOSEN, "yellow frame: the small chart that is shown large"),
                (table[heat_index(-0.9)], "heat map: blue cells are cosines below 0, red ones above, dark ones near 0"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Dimensions $n$:", "how many numbers make up each vector.");
        ui::term(ui, "Cosine:", "of the angle between two directions; for vectors of length 1 it is their dot product. 1: the same way; 0: at right angles; −1: opposite.");
        ui::term(ui, "Standard normal numbers:", "random numbers spread in the bell shape, averaging 0, with standard deviation 1.");
        ui::term(ui, "Standard deviation, spread:", "the typical distance of random numbers from their average: the square root of the average squared distance.");
        ui::term(ui, "Density:", "the height of bars and curves: the share of draws per unit across, so the area between two places is the share of draws that fall between them.");
        ui::term(ui, "Random projection:", "multiplying by a matrix of random numbers with fewer rows than columns, to put points in fewer dimensions.");

        ui::section(ui, "Numbers");
        match self.tab {
            Tab::Angles => {
                let two = 2.0 / (self.n as f64).sqrt();
                let below = self.stream.share_below(two);
                let s = &self.stream;
                ui::readout(ui, "pairs drawn", format!("{} of {}", count(s.pairs as usize), count(TOTAL as usize)));
                ui::readout_colored(ui, "spread of their cosines", num(s.spread(), 4), DRAWN);
                ui::readout_colored(ui, r"$1/\sqrt{n}$", num(1.0 / (self.n as f64).sqrt(), 4), TYPICAL);
                ui::readout(ui, r"pairs with $|\text{cosine}|$ below $2/\sqrt{n}$", format!("{:.1} %", 100.0 * below));
                match self.many.best {
                    Some((c, _, _)) => {
                        ui::readout_colored(ui, r"largest $|\text{cosine}|$ in the heat map", num(c.abs() as f64, 3), LARGEST);
                        let a = angle_apart(c.abs() as f64);
                        ui::readout(ui, "every two directions apart by", format!("{a:.1}° to {:.1}°", 180.0 - a));
                    }
                    None => ui::readout_colored(ui, r"largest $|\text{cosine}|$ in the heat map", "no pairs", LARGEST),
                }
                if self.many.m >= 2 {
                    ui::readout(ui, r"rule $\sqrt{2 \ln(m^2/2)/n}$", num(rule(self.many.n, self.many.m), 3));
                }
                ui::note(ui, r"The spread of the cosine is exactly $1/\sqrt{n}$: the $n$ coordinates of a direction have squares adding up to $1$, so each is $1/n$ on average. The rule for the largest $|\text{cosine}|$ comes from the tail of a bell of that spread. For $n$ of 100 or more it is a fair guide, most often a little above what is measured; for small $n$ it is no guide at all (it passes 1).");
            }
            Tab::Lengths => {
                let s = &self.stream;
                let n = self.n as f64;
                // (the lengths as formulas, every one alike: two of them are written with ±)
                ui::readout(ui, "vectors drawn", count(s.pairs as usize));
                ui::readout_colored(ui, "average length", format!("${}$", num(s.mean_length(), 3)), DRAWN);
                ui::readout_colored(ui, r"$\sqrt{n}$", format!("${}$", num(n.sqrt(), 3)), TYPICAL);
                ui::readout_colored(ui, "spread of the lengths", format!("${}$", num(s.length_spread(), 3)), DRAWN);
                ui::readout(ui, "exactly, for this $n$", format!(r"${} \pm {}$", num(len_mean(n), 3), num(len_sd(n), 3)));
                ui::readout(ui, "spread ÷ average", format!("{:.1} %", 100.0 * len_sd(n) / len_mean(n)));
                ui::readout(ui, r"typical dot product $g \cdot h$", format!(r"$\pm {}$", num(s.dot_spread(), 2)));
            }
            Tab::Projection => {
                if let Some([lo, hi, q1, q2, sd]) = self.proj.summary() {
                    ui::readout(ui, "smallest factor", num(lo, 3));
                    ui::readout(ui, "largest factor", num(hi, 3));
                    ui::readout(ui, "middle 95 %", format!("{} to {}", num(q1, 3), num(q2, 3)));
                    ui::readout_colored(ui, "spread", num(sd, 3), DRAWN);
                    ui::readout_colored(ui, r"$1/\sqrt{2k}$", num(1.0 / (2.0 * self.k as f64).sqrt(), 3), TYPICAL);
                }
            }
        }
        ui::see_also(
            ui,
            &[
                ("packing", r"spreading directions apart: how small the largest $|\text{cosine}|$ can be made"),
                ("superposition", "a network packing more features than it has dimensions"),
                ("sensing", "compressed sensing: random measurements of a sparse signal"),
                ("clt", "the central limit theorem: why sums of many random terms are bell-shaped"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt, f.real_dt, &Budget::new(0.004));
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: nothing legible fits)
        if view.x < 200.0 || view.y < 150.0 {
            return;
        }
        let lay = Layout::of(self.tab, view, f.top_inset);
        let pointer = f.ptr.pos.map(|p| p / f.px);
        let (at, w, h) = lay.tabs;
        if pointer.is_some_and(|p| p.cmpge(at).all() && p.cmple(at + Vec2::new(3.0 * w + 8.0, h)).all()) {
            f.handles.claim_clicks();
        }
        self.draw_tabs(f, &lay, pointer);
        match self.tab {
            Tab::Angles => self.draw_angles(f, &lay, pointer),
            Tab::Lengths => {
                self.draw_lengths(f, &lay);
                self.draw_why(f, &lay);
            }
            Tab::Projection => {
                self.draw_projection(f, &lay);
                self.draw_projection_words(f, &lay);
                self.draw_factors(f, &lay);
            }
        }
        std::hint::black_box(self.animated());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scene set up as experiment `k`, its random draws all made.
    fn run(k: usize) -> HighDim {
        let mut s = HighDim::new();
        s.apply(TRY[k].2);
        settle(&mut s);
        s
    }

    /// Run the scene to rest, at 60 frames a second.
    fn settle(s: &mut HighDim) {
        for _ in 0..20_000 {
            s.advance(1.0 / 60.0, 1.0 / 60.0, &Budget::new(1.0));
            if !s.animated() {
                return;
            }
        }
        panic!("still moving after 20 000 frames");
    }

    #[test]
    fn ln_gamma_is_accurate() {
        assert!((ln_gamma(0.5) - PI64.sqrt().ln()).abs() < 1e-13);
        assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-13);
        assert!((ln_gamma(1.0)).abs() < 1e-13);
        // Stirling's series for a large argument
        let x = 500.5f64;
        let stirling = (x - 0.5) * x.ln() - x + 0.5 * (2.0 * PI64).ln() + 1.0 / (12.0 * x) - 1.0 / (360.0 * x.powi(3));
        assert!((ln_gamma(x) - stirling).abs() < 1e-10, "{} {stirling}", ln_gamma(x));
    }

    /// The exact laws are densities: each has area 1, and the spread it gives the
    /// cosine is 1/√n; the lengths' law has the average and spread worked out for it.
    #[test]
    fn the_laws_are_densities() {
        for n in [2.0, 2.5, 3.0, 4.0, 10.0, 100.0, 1000.0] {
            // (in the angle, so that the ends at n = 2, infinite, are no trouble)
            let steps = 200_000;
            let (mut area, mut second) = (0.0, 0.0);
            for i in 0..steps {
                let t = PI64 * (i as f64 + 0.5) / steps as f64;
                let c = t.cos();
                let w = cos_density(n, c) * t.sin() * PI64 / steps as f64;
                area += w;
                second += w * c * c;
            }
            assert!((area - 1.0).abs() < 1e-6, "n {n}: area {area}");
            assert!((second - 1.0 / n).abs() < 1e-6, "n {n}: second moment {second}");
            let (mut area, mut mean, mut sq) = (0.0, 0.0, 0.0);
            let dr = 60.0 / steps as f64;
            for i in 0..steps {
                let r = (i as f64 + 0.5) * dr;
                let d = len_density(n, r) * dr;
                area += d;
                mean += d * r;
                sq += d * r * r;
            }
            assert!((area - 1.0).abs() < 1e-6 && (mean - len_mean(n)).abs() < 1e-6 && (sq - n).abs() < 1e-5, "n {n}: {area} {mean} {sq}");
        }
        // flat in space; and the spread of a length tends to 1/√2
        assert!((cos_density(3.0, 0.7) - 0.5).abs() < 1e-12 && (cos_density(3.0, 1.0) - 0.5).abs() < 1e-12);
        for (n, lo, hi) in [(2.0, 0.65, 0.66), (10.0, 0.697, 0.699), (100.0, 0.706, 0.7063), (1000.0, 0.707, std::f64::consts::FRAC_1_SQRT_2)] {
            assert!((lo..hi).contains(&len_sd(n)), "n {n}: {}", len_sd(n));
        }
    }

    /// What every chart of the cosine shows: the spread of the cosines drawn is
    /// 1/√n, and the bars come to the exact law.
    #[test]
    fn the_cosines_drawn_follow_the_exact_law() {
        for n in [2usize, 3, 10, 100, 1000] {
            let mut s = Stream::main(n);
            while !s.complete() {
                s.step();
            }
            let spread = s.spread() * (n as f64).sqrt();
            assert!((spread - 1.0).abs() < 0.012, "n {n}: spread × √n = {spread}");
            // each bar holds about the share of pairs the law gives it (where that is
            // more than a few pairs, and leaving out the ends at n = 2)
            let h = &s.cos;
            for i in 0..h.bins() {
                let (a, b) = (h.lo + i as f64 * h.width(), h.lo + (i + 1) as f64 * h.width());
                let steps = 400;
                let law: f64 = (0..steps).map(|j| cos_density(n as f64, a + (b - a) * (j as f64 + 0.5) / steps as f64)).sum::<f64>() / steps as f64 * (b - a);
                let expected = law * TOTAL as f64;
                if expected > 20.0 && (n > 2 || i > 0 && i + 1 < h.bins()) {
                    let seen = h.counts[i] as f64;
                    assert!((seen - expected).abs() < 5.0 * expected.sqrt() + 2.0, "n {n}, bar {i}: {seen} against {expected:.1}");
                }
            }
        }
    }

    /// The share of the heat map's pairs whose |cosine| passes `test`.
    fn cells(m: &Many, test: impl Fn(f32) -> bool) -> f64 {
        let n = m.n;
        let pass = (0..m.rows).flat_map(|i| (0..i).map(move |j| (i, j))).filter(|&(i, j)| test(dot(&m.vecs[i * n..(i + 1) * n], &m.vecs[j * n..(j + 1) * n]).abs())).count();
        pass as f64 / m.pairs().max(1) as f64
    }

    /// The scene opens on "a hundred dimensions", and comes to rest.
    #[test]
    fn the_scene_opens_on_a_hundred_dimensions() {
        let mut s = HighDim::new();
        let o = TRY[OPENING].2;
        assert_eq!((s.tab, s.n, s.m, s.big_n, s.k), (o.tab, o.n, o.m, o.big_n, o.k));
        assert_eq!(TRY[OPENING].0, "a hundred dimensions");
        assert!(s.animated());
        settle(&mut s);
        assert!(s.stream.complete() && s.many.complete() && s.stream.cos.caught_up());
    }

    /// "The plane": half the pairs are within 45° of lying along one line; the
    /// curve is cut off at the top; in the heat map 70 % of the cells are bright.
    #[test]
    fn the_plane() {
        let s = run(0);
        assert_eq!((s.n, s.m, s.tab, s.stream.pairs), (2, 40, Tab::Angles, TOTAL));
        let share = s.stream.share(|c| c.abs() > 0.5f64.sqrt());
        assert!(TRY[0].1.contains("(50.0 % of 20 000)") && format!("{:.1}", share * 100.0) == "50.0", "{share}");
        assert!((0.5f64.sqrt().acos().to_degrees() - 45.0).abs() < 1e-9);
        // the curve goes off the top of the chart near both ends, and the bars there too
        assert!(cos_density(2.0, 0.97) > cos_top(2.0) && cos_density(2.0, 0.95) < cos_top(2.0));
        let h = &s.stream.cos;
        assert!(h.shown[0] as f64 > cos_top(2.0) && h.shown[h.bins() - 1] as f64 > cos_top(2.0));
        let bright = cells(&s.many, |c| c > 0.5);
        assert!(TRY[0].1.contains("70 % of the cells are bright") && format!("{:.0}", bright * 100.0) == "70", "{bright}");
    }

    /// "Space": the bars come up level at ½ (every one within 0.06 of it), and 20 %
    /// of pairs are within 11.5° of a right angle.
    #[test]
    fn space() {
        let s = run(1);
        assert_eq!((s.n, s.tab), (3, Tab::Angles));
        let h = &s.stream.cos;
        for i in 0..h.bins() {
            assert!((h.shown[i] - 0.5).abs() < 0.06, "bar {i}: {}", h.shown[i]);
        }
        let near = s.stream.share(|c| c.abs() < 0.2);
        assert!(TRY[1].1.contains("Only 20 % of pairs are within 11.5°") && format!("{:.0}", near * 100.0) == "20", "{near}");
        assert!((90.0 - 0.2f64.acos().to_degrees() - 11.5).abs() < 0.05);
    }

    /// "A hundred dimensions": 95.5 % of pairs have |cosine| below 0.2, between
    /// 78.5° and 101.5°; the spread is 1/√n = 0.1; in the heat map the largest
    /// |cosine| of the 780 pairs is 0.34.
    #[test]
    fn a_hundred_dimensions() {
        let s = run(2);
        assert_eq!((s.n, s.m), (100, 40));
        let near = s.stream.share(|c| c.abs() < 0.2);
        assert!(TRY[2].1.contains("95.5 % of 20 000 pairs") && format!("{:.1}", near * 100.0) == "95.5", "{near}");
        assert!(format!("{:.1}", 0.2f64.acos().to_degrees()) == "78.5" && format!("{:.1}", (-0.2f64).acos().to_degrees()) == "101.5");
        assert!((s.stream.spread() - 0.1).abs() < 0.002, "{}", s.stream.spread());
        let (c, _, _) = s.many.best.expect("pairs");
        assert_eq!(s.many.pairs(), 780);
        assert!(TRY[2].1.contains("780 pairs (framed in green) is 0.34") && num(c.abs() as f64, 2) == "0.34", "{c}");
        // dim: no cell off the diagonal reaches a third of the way to full brightness
        assert!(cells(&s.many, |c| c > 0.34) == 0.0);
    }

    /// "A thousand directions in a hundred dimensions": the largest |cosine| of the
    /// 499 500 pairs is 0.46, so every two are at least 62.7° apart; among the first
    /// 100 it is 0.37.
    #[test]
    fn a_thousand_directions_in_a_hundred_dimensions() {
        let s = run(3);
        assert_eq!((s.n, s.m, s.many.pairs()), (100, 1000, 499_500));
        let (c, i, j) = s.many.best.expect("pairs");
        let c = c.abs() as f64;
        let words = TRY[3].1;
        assert!(words.contains("all 499 500 pairs is 0.46") && num(c, 2) == "0.46", "{c}");
        assert!(words.contains("at least 62.7° apart") && format!("{:.1}", angle_apart(c)) == "62.7");
        assert!(words.contains("0.37 at $r = 100$") && num(s.many.by_r[99] as f64, 2) == "0.37", "{}", s.many.by_r[99]);
        assert!(words.contains("still only 0.46 at $r = 1000$") && s.many.by_r[999] as f64 == c);
        // the frame is round that pair, above the diagonal
        assert!(i < j && s.ring == s.ring_goal() && s.ring.is_some());
        // and the curve only ever rises
        assert!(s.many.by_r.windows(2).all(|w| w[1] >= w[0]));
    }

    /// "A thousand dimensions squashed into fifty": every factor between 0.71 and
    /// 1.37, the middle 95 % between 0.80 and 1.22, and six in ten within 10 %.
    #[test]
    fn a_thousand_dimensions_squashed_into_fifty() {
        let s = run(4);
        assert_eq!((s.tab, s.big_n, s.k, s.proj.pairs.len()), (Tab::Projection, 1000, 50, 1225));
        assert!(s.proj.done());
        let r = s.proj.ratios();
        let [lo, hi, q1, q2, sd] = summary(r).expect("pairs");
        let words = TRY[4].1;
        assert!(words.contains("between 0.71 and 1.37") && num(lo, 2) == "0.71" && num(hi, 2) == "1.37", "{lo} {hi}");
        assert!(words.contains("between 0.80 and 1.22") && num(q1, 2) == "0.80" && num(q2, 2) == "1.22", "{q1} {q2}");
        let within = r.iter().filter(|v| (**v - 1.0).abs() < 0.1).count() as f64 / r.len() as f64;
        assert!(words.contains("six in ten") && (0.57..0.65).contains(&within), "{within}");
        assert!((1.0 / (2.0f64 * 50.0).sqrt() - 0.1).abs() < 1e-12);
        // its spread is about 1/√(2k)
        assert!((sd - 0.1).abs() < 0.015, "{sd}");
        // the bars beside them count every factor (all below 2 here), and have come to
        // their counts
        assert!(s.factors.counts.iter().sum::<u32>() as usize == r.len() && s.factors.caught_up() && s.k_shown == 50.0);
        // the dots drawn are where the pairs are
        assert!(s.dots.iter().zip(0..).all(|(d, p)| (d.x as f64 - s.proj.before[p]).abs() < 1e-4 && (d.y as f64 - s.proj.after(p).unwrap_or(0.0)).abs() < 1e-4));
    }

    /// The lengths view: a vector of n standard normal numbers is close to √n long,
    /// within about 1/√2 either way whatever n is; the dot product of two is about
    /// ±√n; as drawn and as worked out exactly.
    #[test]
    fn lengths_are_nearly_fixed() {
        for n in [2usize, 3, 10, 100, 1000] {
            let mut st = Stream::main(n);
            while !st.complete() {
                st.step();
            }
            let root = (n as f64).sqrt();
            assert!((st.mean_length() - len_mean(n as f64)).abs() < 0.02, "n {n}: {}", st.mean_length());
            assert!((len_mean(n as f64) - root).abs() < 0.17 && (0.64..0.72).contains(&st.length_spread()), "n {n}: {} ± {}", st.mean_length(), st.length_spread());
            assert!((st.dot_spread() / root - 1.0).abs() < 0.02, "n {n}: dot products ±{}", st.dot_spread());
        }
        // ever closer in proportion
        assert!(len_sd(1000.0) / len_mean(1000.0) < 0.023 && len_sd(10.0) / len_mean(10.0) > 0.22);
    }

    /// The panel's words on the rule √(2 ln(m²/2)/n): for n of 100 or more a fair
    /// guide, most often a little above what is measured; for small n no guide (it
    /// passes 1).
    #[test]
    fn the_rule_for_the_largest() {
        let (mut above, mut all) = (0, 0);
        for kind in [MANY, 11, 12, 13, 14, 15] {
            for (n, m) in [(100usize, 40usize), (100, 300), (300, 300), (1000, 200)] {
                let mut many = Many::drawn(n, m, kind);
                while !many.complete() {
                    many.step();
                }
                let ratio = many.best.map_or(0.0, |b| b.0.abs() as f64) / rule(n, m);
                assert!((0.75..1.06).contains(&ratio), "n {n}, m {m}, seeds {kind}: measured ÷ rule = {ratio}");
                above += (ratio < 1.0) as u32;
                all += 1;
            }
        }
        assert!(above * 10 >= all * 8, "the rule above what is measured {above} times of {all}");
        assert!(rule(10, 40) > 1.0 && rule(3, 10) > 1.0);
    }

    /// The words under the projection: with k below 4 the factors run from nearly 0
    /// to 2 or more; and N does not come into the spread.
    #[test]
    fn few_dimensions_scramble_distances() {
        for k in 1..4 {
            let mut p = Projection::new(1000, k);
            p.work(&Budget::new(10.0));
            let [lo, hi, ..] = summary(p.ratios()).expect("pairs");
            assert!(lo < 0.2 && hi > 2.0, "k {k}: {lo} to {hi}");
        }
        for big_n in [100usize, 300, 1000] {
            let mut p = Projection::new(big_n, 50);
            p.work(&Budget::new(10.0));
            let sd = summary(p.ratios()).expect("pairs")[4];
            assert!((sd - 0.1).abs() < 0.02, "N {big_n}: spread {sd}");
        }
    }

    /// One more direction leaves the others as they were, and one more row of the
    /// matrix the other rows; fewer rows sum again from the start.
    #[test]
    fn more_adds_and_keeps_the_rest() {
        let (mut a, mut b) = (Many::new(30, 20), Many::new(30, 50));
        while !a.complete() {
            a.step();
        }
        while !b.complete() {
            b.step();
        }
        assert_eq!(a.vecs[..], b.vecs[..20 * 30]);
        assert_eq!(a.by_r[..], b.by_r[..20]);
        let mut p = Projection::new(200, 10);
        p.work(&Budget::new(10.0));
        let ten: Vec<f64> = (0..p.pairs.len()).filter_map(|q| p.after(q)).collect();
        p.k = 20;
        p.work(&Budget::new(10.0));
        let coords = p.coords[..10 * POINTS].to_vec();
        p.k = 10;
        p.work(&Budget::new(10.0));
        assert_eq!(p.coords[..10 * POINTS], coords[..]);
        let again: Vec<f64> = (0..p.pairs.len()).filter_map(|q| p.after(q)).collect();
        assert!(ten.iter().zip(&again).all(|(x, y)| (x - y).abs() < 1e-12));
    }

    /// Settings at their ends and every window: nothing panics, everything on show
    /// is finite, and the scene comes to rest.
    #[test]
    fn the_ends_of_the_sliders_are_fine() {
        for (n, m, big_n, k) in [(2, 1, 2, 1), (2, 2, 2, 2), (1000, 1000, 1000, 1000), (0, 0, 0, 0), (5000, 5000, 5000, 9000), (7, 3, 3, 1)] {
            for tab in [Tab::Angles, Tab::Lengths, Tab::Projection] {
                let mut s = HighDim::new();
                s.apply(Setup { tab, n, m, big_n, k });
                settle(&mut s);
                assert!(s.n >= 2 && s.n <= N_MAX && s.m >= 1 && s.m <= M_MAX && s.k >= 1 && s.k <= s.big_n);
                let st = &s.stream;
                assert!([st.spread(), st.mean_length(), st.length_spread(), st.dot_spread()].iter().all(|v| v.is_finite()));
                assert!(s.dots.iter().all(|d| d.is_finite()) && s.span.is_finite() && s.n_shown.is_finite());
                if tab == Tab::Angles {
                    // one direction has no pair; two have one
                    assert_eq!(s.many.best.is_some(), s.many.m >= 2);
                    assert!(s.many.by_r.iter().all(|v| v.is_finite()));
                }
            }
        }
        for tab in [Tab::Angles, Tab::Lengths, Tab::Projection] {
            for (x, y, inset) in [(1600.0, 1000.0, 0.0), (1100.0, 700.0, 0.0), (390.0, 800.0, 46.0), (240.0, 180.0, 46.0), (200.0, 150.0, 46.0), (800.0, 300.0, 46.0), (3000.0, 400.0, 0.0)] {
                let lay = Layout::of(tab, Vec2::new(x, y), inset);
                let mut numbers = vec![lay.big, lay.text, lay.small, lay.tabs.0.x, lay.tabs.1, lay.tabs.2, lay.title.x, lay.title.y, lay.column, lay.main.0.x, lay.main.0.y, lay.main.1.x, lay.main.1.y];
                for (a, b) in [lay.smalls, lay.best, lay.side].into_iter().flatten() {
                    numbers.extend([a.x, a.y, b.x, b.y]);
                }
                if let Some((a, side)) = lay.heat {
                    numbers.extend([a.x, a.y, side]);
                }
                if let Some((a, w)) = lay.words {
                    numbers.extend([a.x, a.y, w]);
                }
                assert!(numbers.iter().all(|v| v.is_finite()), "{tab:?} at {x} × {y}");
                // what is drawn stays on the canvas
                assert!(lay.main.1.x <= x && lay.heat.is_none_or(|(a, s)| a.x + s <= x && a.y + s <= y), "{tab:?} at {x} × {y}: {lay:?}");
            }
        }
        // the laws at the ends of n stay finite where they are drawn
        for n in [2.0, 2.0001, 2.999, 3.0, 1000.0] {
            assert!(cos_top(n).is_finite() && cos_density(n, 0.3).is_finite() && len_density(n, 5.0).is_finite() && len_mean(n).is_finite());
        }
    }

    /// The bars rise to their counts without overshooting, and are exactly there in
    /// the end; a new n carries the drawn shape over and lets it sink.
    #[test]
    fn the_bars_rise_evenly() {
        let mut s = Stream::main(100);
        let mut last = vec![0.0f32; s.cos.bins()];
        let mut steps = Vec::new();
        for frame in 0..600 {
            for _ in 0..(RATE / 60.0) as usize {
                if !s.complete() {
                    s.step();
                }
            }
            s.cos.follow(1.0 / 60.0);
            for i in 0..s.cos.bins() {
                assert!(s.cos.shown[i] <= s.cos.height(i) + 1e-6 && s.cos.shown[i] >= last[i] - 1e-6, "frame {frame}, bar {i}");
            }
            let middle = s.cos.bins() / 2;
            steps.push(s.cos.shown[middle] - last[middle]);
            last.clone_from(&s.cos.shown);
        }
        assert!(s.cos.caught_up());
        // while the pairs come in, the middle bar rises by nearly the same each frame
        // (the counts come in unevenly; the drawn bar does not)
        for w in steps[30..140].windows(2) {
            assert!((w[1] - w[0]).abs() < 0.15 * w[0].max(w[1]), "{w:?}");
        }
        let before = s.cos.shown.clone();
        s.renew(2);
        assert_eq!(s.cos.bins(), cos_bins(2));
        let middle = (before[59] + before[60]) * 0.5;
        assert!(s.cos.shown[s.cos.bins() / 2] > 0.5 * middle && s.cos.counts.iter().all(|c| *c == 0));
    }

    /// The heat map's picture, brought up to date a few directions at a time by
    /// copying only their rows and columns, is always the whole heat map: through new
    /// n, new m (a picture of another size), an experiment, and back.
    #[test]
    fn the_heat_maps_picture_follows_every_direction() {
        // the whole heat map as the picture holds it (its rows from the bottom)
        let whole = |many: &Many| (0..many.m).rev().flat_map(|i| many.pixels[i * many.m..(i + 1) * many.m].to_vec()).collect::<Vec<Color>>();
        let mut s = HighDim::new();
        let mut picture = Vec::new();
        let changes: [fn(&mut HighDim); 7] = [|_| {}, |s| s.set_n(30), |s| s.set_m(70), |s| s.set_m(20), |s| s.apply(TRY[0].2), |s| s.set_n(100), |s| s.set_m(40)];
        for (i, change) in changes.into_iter().enumerate() {
            change(&mut s);
            for few in [0, 1, 3, 7, 2, 50, 1] {
                for _ in 0..few {
                    s.many.step();
                }
                // (as `draw_heat` does it: a picture of another size is a new one,
                // holding no directions)
                let m = s.many.m;
                if picture.len() != m * m {
                    picture = vec![Color::CLEAR; m * m];
                    s.many.painted = None;
                }
                if s.many.painted != Some(s.many.rows) {
                    s.many.paint(&mut picture);
                }
                assert!(picture == whole(&s.many), "change {i}, {} directions", s.many.rows);
            }
        }
    }

    /// What is kept of the projection (the ratios sorted and summed up, the factors
    /// counted, the dots' places) is what working it out afresh gives, half way
    /// through the rows of the matrix and at the end, after every change of N and k
    /// and an experiment.
    #[test]
    fn what_is_kept_of_the_projection_is_what_it_is() {
        let check = |s: &mut HighDim, i: usize| {
            let p = &s.proj;
            let mut r: Vec<f64> = (0..p.pairs.len()).filter_map(|q| p.after(q).map(|a| a / p.before[q])).collect();
            r.sort_by(f64::total_cmp);
            assert!(p.ratios() == r.as_slice() && p.summary() == summary(&r), "change {i}, {} rows", p.used);
            let mut counted = Hist::new(0.0, 2.0, s.factors.bins(), 1);
            r.iter().for_each(|v| counted.add(*v));
            assert_eq!(s.factors.counts, counted.counts, "change {i}, {} rows", p.used);
            assert_eq!(s.proj.ratios().iter().filter(|r| **r >= 2.0).count(), (0..p.pairs.len()).filter(|&q| p.after(q).is_some_and(|a| a / p.before[q] >= 2.0)).count());
            let (goals, span) = s.dot_goals();
            assert_eq!(s.dots_moving(), s.dots != goals || s.span != span, "change {i}");
        };
        let mut s = HighDim::new();
        s.apply(TRY[4].2);
        let changes: [fn(&mut HighDim); 6] =
            [|_| {}, |s| s.set_projection(1000, 20), |s| s.set_projection(1000, 300), |s| s.set_projection(300, 80), |s| s.set_projection(1000, 50), |s| s.apply(TRY[4].2)];
        for (i, change) in changes.into_iter().enumerate() {
            change(&mut s);
            // (a frame with no time to spare takes one row of the matrix)
            s.advance(1.0 / 60.0, 1.0 / 60.0, &Budget::new(0.0));
            check(&mut s, i);
            settle(&mut s);
            check(&mut s, i);
            let (goals, span) = s.dot_goals();
            assert!(s.dots == goals && s.span == span, "change {i}");
        }
    }

    /// A curve kept for a slot is the one its law and chart make: made again when n or
    /// a corner of the chart changes in the least, kept while neither does.
    #[test]
    fn a_kept_curve_is_the_one_its_law_and_chart_make() {
        let mut kept = Sketches::default();
        let made = std::cell::Cell::new(0);
        let (a, b) = (Vec2::new(10.0, 20.0), Vec2::new(300.0, 200.0));
        let a2 = Vec2::new(10.0, 20.000002);
        let cases = [(0, 3.0, a, b, true), (0, 3.0, a, b, false), (0, 3.0 + 1e-12, a, b, true), (0, 3.0, a2, b, true), (0, 3.0, a2, b + Vec2::X, true), (2, 3.0, a2, b + Vec2::X, true), (0, 3.0, a2, b + Vec2::X, false)];
        for (slot, n, min, max, new) in cases {
            let before = made.get();
            let curve = kept.curve(slot, n, min, max, || {
                made.set(made.get() + 1);
                vec![(min * n as f32 + max, EXACT)]
            });
            assert_eq!(curve, &[(min * n as f32 + max, EXACT)]);
            assert_eq!(made.get() - before, new as u32, "slot {slot}, n {n}, {min} {max}");
        }
    }

    /// The share of the cosines below a bound, counted only over those drawn since it
    /// was last asked for, is the share of all of them: as pairs are drawn, for
    /// another bound, and after a new n.
    #[test]
    fn the_share_below_counts_every_cosine() {
        let mut st = Stream::main(100);
        for (pairs, bound) in [(0, 0.2), (50, 0.2), (1000, 0.2), (0, 0.1), (10, 0.1), (5000, 0.2)] {
            for _ in 0..pairs {
                st.step();
            }
            assert_eq!(st.share_below(bound), st.share(|c| c.abs() < bound));
        }
        st.renew(10);
        for _ in 0..30 {
            st.step();
        }
        assert_eq!(st.share_below(0.2), st.share(|c| c.abs() < 0.2));
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for (k, (label, _, setup)) in TRY.iter().enumerate() {
            let s = run(k);
            let st = &s.stream;
            let n = s.n as f64;
            println!("{label}: n {} m {} N {} k {}", s.n, s.m, s.big_n, s.k);
            println!(
                "  spread {:.4} (1/√n {:.4}); |c|>0.7071 {:.4}; |c|<0.2 {:.4}; |c|<2/√n {:.4}; mean length {:.3} sd {:.3}; dot spread {:.2}",
                st.spread(),
                1.0 / n.sqrt(),
                st.share(|c| c.abs() > 0.5f64.sqrt()),
                st.share(|c| c.abs() < 0.2),
                st.share(|c| c.abs() < 2.0 / n.sqrt()),
                st.mean_length(),
                st.length_spread(),
                st.dot_spread()
            );
            let h = &st.cos;
            let heights: Vec<f32> = (0..h.bins()).map(|i| h.height(i)).collect();
            let (lo, hi) = heights.iter().fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
            println!("  bars {} from {lo:.3} to {hi:.3}", h.bins());
            let m = &s.many;
            if let Some((c, i, j)) = m.best {
                let bright = (0..m.m).flat_map(|i| (0..i).map(move |j| (i, j))).filter(|&(i, j)| dot(&m.vecs[i * m.n..(i + 1) * m.n], &m.vecs[j * m.n..(j + 1) * m.n]).abs() > 0.5).count();
                println!(
                    "  heat: largest {:.4} ({i}, {j}) angle {:.2}; rule {:.3}; pairs {}; share |c|>0.5 {:.3}; by r=10 {:.3} r=100 {:.3} r=m {:.3}",
                    c.abs(),
                    angle_apart(c.abs() as f64),
                    rule(m.n, m.m),
                    m.pairs(),
                    bright as f64 / m.pairs() as f64,
                    m.by_r.get(9).copied().unwrap_or(0.0),
                    m.by_r.get(99).copied().unwrap_or(0.0),
                    m.by_r.last().copied().unwrap_or(0.0)
                );
            }
            let mut p = Projection::new(setup.big_n, setup.k);
            p.work(&Budget::new(10.0));
            let r = p.ratios();
            if let Some([lo, hi, q1, q2, sd]) = summary(r) {
                let inside = r.iter().filter(|v| (**v - 1.0).abs() < 1.0 / (2.0 * setup.k as f64).sqrt()).count() as f64 / r.len() as f64;
                let within10 = r.iter().filter(|v| (**v - 1.0).abs() < 0.1).count() as f64 / r.len() as f64;
                println!("  projection: {lo:.3}..{hi:.3}, 95 % {q1:.3}..{q2:.3}, sd {sd:.4}, inside {inside:.3}, within 10 % {within10:.3}");
            }
        }
        for n in [2usize, 3, 10, 100, 1000] {
            let mut st = Stream::main(n);
            while !st.complete() {
                st.step();
            }
            println!("n {n}: spread {:.4} × √n = {:.4}; length {:.3} ± {:.3} (exact {:.3} ± {:.3}); dot spread {:.2}", st.spread(), st.spread() * (n as f64).sqrt(), st.mean_length(), st.length_spread(), len_mean(n as f64), len_sd(n as f64), st.dot_spread());
        }
        for (big_n, k) in [(1000usize, 1usize), (1000, 2), (1000, 5), (1000, 10), (1000, 20), (1000, 50), (1000, 200), (1000, 1000), (100, 50), (10, 5), (2, 1)] {
            let mut p = Projection::new(big_n, k);
            p.work(&Budget::new(10.0));
            let r = p.ratios();
            let [lo, hi, q1, q2, sd] = summary(r).unwrap();
            let inside = r.iter().filter(|v| (**v - 1.0).abs() < 1.0 / (2.0 * k as f64).sqrt()).count() as f64 / r.len() as f64;
            println!("N {big_n} k {k}: {lo:.3}..{hi:.3}, 95 % {q1:.3}..{q2:.3}, sd {sd:.4} (1/√(2k) {:.4}), inside {inside:.3}", 1.0 / (2.0 * k as f64).sqrt());
        }
        for (n, m) in [(100usize, 1000usize), (1000, 1000), (10, 1000), (100, 100), (300, 1000)] {
            let mut many = Many::new(n, m);
            while !many.complete() {
                many.step();
            }
            println!("n {n} m {m}: largest {:.4}, rule {:.4}", many.best.unwrap().0.abs(), rule(n, m));
        }
    }

    /// How the heat map's numbers vary from one family of seeds to another.
    #[test]
    #[ignore]
    fn seeds() {
        for kind in 3..40u64 {
            let mut a = Many::drawn(100, 40, kind);
            while !a.complete() {
                a.step();
            }
            let mut b = Many::drawn(100, 1000, kind);
            while !b.complete() {
                b.step();
            }
            let mut c = Many::drawn(2, 40, kind);
            while !c.complete() {
                c.step();
            }
            let bright = (0..40).flat_map(|i| (0..i).map(move |j| (i, j))).filter(|&(i, j)| dot(&c.vecs[i * 2..i * 2 + 2], &c.vecs[j * 2..j * 2 + 2]).abs() > 0.5).count() as f64 / 780.0;
            println!("kind {kind}: n 100 m 40 largest {:.3}; m 1000: r=10 {:.3} r=40 {:.3} r=100 {:.3} r=300 {:.3} r=1000 {:.3}; plane bright {bright:.3}", a.best.unwrap().0.abs(), b.by_r[9], b.by_r[39], b.by_r[99], b.by_r[299], b.by_r[999]);
        }
    }
}

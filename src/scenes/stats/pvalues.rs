//! Many tests, and peeking.
//!
//! When nothing is going on, a test's p-value is equally likely to be anywhere from 0
//! to 1: that is all a level α rests on, and it is a promise about one look at one
//! test. Three pictures of what becomes of it: many studies (where their p-values
//! fall, and what the significant ones estimated); a family of tests done at once,
//! with the corrections of Bonferroni and of Benjamini and Hochberg; and one study
//! tested again after every new value and stopped as soon as p ≤ α.
//!
//! Every test here is the same one: n values, each normal with spread 1, and the null
//! hypothesis that their mean is 0, tested two-sided with z = (their mean) × √n, the
//! spread being known. A study's mean is drawn directly, as a normal number with
//! spread 1/√n: the same in distribution as drawing its n values and averaging them.
//! The studies that are peeked at do draw their values one by one, since each look
//! sees one more.

use super::plot::{self, Glide, Plot, Rect, AXES, CURVES, MARKS, OWN};
use super::*;
use crate::math::stats as st;

pub const DEF: SceneDef = SceneDef {
    id: "pvalues",
    title: "Many tests, and peeking",
    group: TESTING,
    blurb: "When nothing is going on, $p$-values are spread evenly from 0 to 1: what that does to twenty tests at once, and to stopping as soon as a result looks good.",
    make: |_| Box::new(PValues::create()),
};

// ----- colours ------------------------------------------------------------------------------

/// A hypothesis whose null is true: nothing is going on.
const NULL: Color = ASSUMED;
/// A hypothesis that is a real effect (here the truth is known), and the effect's true size.
const REAL: Color = TRUTH;
/// The level α and what lies beyond it: a result called significant; one that is not
/// real is a false alarm.
const SIGNIFICANT: Color = ALARM;
/// The p-value of one study after each new value.
const PATH: Color = DATA;
/// The region beyond α, faintly. (Colours mix in linear light: a few per cent of pink
/// over the charts' near-black comes out a strong dark red.)
fn faint_significant() -> Color {
    SIGNIFICANT.a(0.009)
}

// ----- the test ---------------------------------------------------------------------------------

/// The two-sided p-value of a z statistic: the chance, if the null is true, of a z at
/// least as far from 0.
fn p_of(z: f64) -> f64 {
    (2.0 * st::normal_cdf(-z.abs())).min(1.0)
}

/// How far from 0 a z must be for a two-sided test at level α to call it significant.
fn critical(alpha: f64) -> f64 {
    st::normal_quantile(1.0 - 0.5 * alpha)
}

/// The chance that the test calls a real effect of size `effect` significant (its
/// power), with n values: z is then normal about effect × √n.
fn power(effect: f64, n: usize, alpha: f64) -> f64 {
    let (shift, c) = (effect * (n as f64).sqrt(), critical(alpha));
    st::normal_cdf(shift - c) + st::normal_cdf(-shift - c)
}

/// Benjamini and Hochberg's rule, on p-values sorted in rising order: how many of them
/// to declare. That is the largest rank k whose p-value is at most k q / m, the line
/// through 0 that reaches q at the last rank; it and every smaller one are declared,
/// also those above the line before it. 0 if no p-value is under the line.
fn bh_count(sorted: &[f64], q: f64) -> usize {
    let m = sorted.len() as f64;
    (1..=sorted.len()).rev().find(|&k| sorted[k - 1] <= k as f64 * q / m).unwrap_or(0)
}

/// The three ways of deciding which tests of a family to declare.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    /// Each at α, as if it were alone.
    Plain,
    /// Each at α/m.
    Bonferroni,
    /// Benjamini and Hochberg's, at q.
    Bh,
}

impl Rule {
    const ALL: [Rule; 3] = [Rule::Plain, Rule::Bonferroni, Rule::Bh];

    fn name(self) -> &'static str {
        match self {
            Rule::Plain => "no correction",
            Rule::Bonferroni => "Bonferroni",
            Rule::Bh => "Benjamini–Hochberg",
        }
    }

    /// The p-value at or below which this rule declares a test of the family (sorted
    /// rising); none declared below 0.
    fn threshold(self, sorted: &[f64], alpha: f64, q: f64) -> f64 {
        match self {
            Rule::Plain => alpha,
            Rule::Bonferroni => alpha / sorted.len().max(1) as f64,
            Rule::Bh => match bh_count(sorted, q) {
                0 => -1.0,
                k => sorted[k - 1],
            },
        }
    }
}

// ----- the settings -------------------------------------------------------------------------

/// Everything the three pictures depend on.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    /// Values per test.
    n: usize,
    /// The size of a real effect: its mean, in spreads of one value.
    effect: f64,
    /// The share of the hypotheses tested that are real effects.
    real: f64,
    /// The level: a test calls p ≤ α significant.
    alpha: f64,
    /// Tests in a family, and the false discovery rate Benjamini and Hochberg are asked for.
    m: usize,
    q: f64,
    /// The first and the last look at a study that is peeked at (values so far).
    first: usize,
    last: usize,
}

impl Setup {
    /// Everything in its range, whatever a slider or a test has done.
    fn tidy(&mut self) {
        let finite = |v: f64, or: f64| if v.is_finite() { v } else { or };
        self.n = self.n.clamp(1, 400);
        self.effect = finite(self.effect, 0.5).max(0.0).min(1.5);
        self.real = finite(self.real, 0.0).max(0.0).min(1.0);
        self.alpha = finite(self.alpha, 0.05).max(0.001).min(0.2);
        self.m = self.m.clamp(1, 200);
        self.q = finite(self.q, 0.1).max(0.01).min(0.3);
        self.first = self.first.clamp(1, 200);
        self.last = self.last.max(self.first).min(1000);
    }

    /// Where a real effect's z is centred.
    fn shift(&self) -> f64 {
        self.effect * (self.n as f64).sqrt()
    }

    /// How many of a family's tests are real effects.
    fn real_tests(&self) -> usize {
        ((self.real * self.m as f64).round() as usize).min(self.m)
    }
}

/// The three pictures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tab {
    Studies,
    Family,
    Peeking,
}

impl Tab {
    const ALL: [Tab; 3] = [Tab::Studies, Tab::Family, Tab::Peeking];

    fn index(self) -> usize {
        self as usize
    }
}

// ----- many studies -----------------------------------------------------------------------------

/// The bins of the p-values' histogram: twentieths of 0 to 1.
const BINS: usize = 20;

#[derive(Clone, Copy, Debug, PartialEq)]
struct StudyKey {
    n: usize,
    effect: f64,
    real: f64,
    seed: u64,
}

/// Studies one after another, each of a real effect with the chance `real`.
struct Studies {
    key: StudyKey,
    rng: Rng,
    /// Each study's z, and whether it is of a real effect. (Its estimate of the effect
    /// is z / √n and its p-value `p_of(z)`; kept as z, they can be counted again at
    /// any level.)
    z: Vec<f32>,
    real: Vec<bool>,
}

impl Studies {
    fn new(key: StudyKey) -> Studies {
        Studies { key, rng: Rng::new(key.seed), z: Vec::new(), real: Vec::new() }
    }

    fn step(&mut self) {
        let real = st::unit(&mut self.rng) < self.key.real;
        let shift = if real { self.key.effect * (self.key.n as f64).sqrt() } else { 0.0 };
        self.z.push((shift + st::normal(&mut self.rng)) as f32);
        self.real.push(real);
    }
}

/// What the studies so far come to, at a level: counted as they come, and counted
/// again from the start when the level changes (dragged, say).
#[derive(Clone, Debug, Default)]
struct Counts {
    key: Option<(StudyKey, u64)>,
    counted: usize,
    /// The p-values in twentieths of 0 to 1: true nulls, real effects.
    p_bins: [[u32; BINS]; 2],
    /// The estimates' bins, in units of z: bin j runs from (first + j) w to
    /// (first + j + 1) w, and the critical value is k w exactly, so that each bin is
    /// either significant or not. Counts of true nulls and real effects.
    first: i64,
    w: f64,
    k: i64,
    est: Vec<[u32; 2]>,
    /// The range the chart shows, in units of z: what the bins are cut from, which
    /// moves smoothly with the level (the bins' own ends jump when their width does).
    lo: f64,
    hi: f64,
    /// Studies, and significant ones: true nulls, real effects.
    total: [u64; 2],
    sig: [u64; 2],
    /// The significant real effects: their estimates added up, and how many of them
    /// have the wrong sign.
    sig_sum: f64,
    sig_wrong: u64,
}

impl Counts {
    /// Count the studies not yet counted (all of them again if the level or the
    /// studies have changed).
    fn catch_up(&mut self, s: &Studies, alpha: f64) {
        let key = (s.key, alpha.to_bits());
        if self.key != Some(key) {
            let c = critical(alpha);
            let shift = s.key.effect * (s.key.n as f64).sqrt();
            // from below the lowest hump to above the highest, 0 and the critical
            // values always among them
            let centre_lo = if s.key.real > 0.0 { shift.min(0.0) } else { 0.0 };
            let centre_hi = if s.key.real > 0.0 { shift.max(0.0) } else { 0.0 };
            let (lo, hi) = (centre_lo.min(-c) - 3.4, centre_hi.max(c) + 3.4);
            let k = ((c / ((hi - lo) / 40.0)).round() as i64).max(1);
            let w = c / k as f64;
            let first = (lo / w).floor() as i64;
            let bins = (((hi / w).ceil() as i64 - first).max(1) as usize).min(400);
            *self = Counts { key: Some(key), first, w, k, est: vec![[0; 2]; bins], lo, hi, ..Counts::default() };
        }
        let c = critical(alpha);
        let root = (s.key.n as f64).sqrt();
        for i in self.counted..s.z.len().min(s.real.len()) {
            let (z, r) = (s.z[i] as f64, s.real[i] as usize);
            self.p_bins[r][((p_of(z) * BINS as f64) as usize).min(BINS - 1)] += 1;
            let j = (z / self.w).floor() as i64 - self.first;
            if let Some(bin) = usize::try_from(j).ok().and_then(|j| self.est.get_mut(j)) {
                bin[r] += 1;
            }
            self.total[r] += 1;
            if z.abs() >= c {
                self.sig[r] += 1;
                if r == 1 {
                    self.sig_sum += z / root;
                    self.sig_wrong += (z < 0.0) as u64;
                }
            }
        }
        self.counted = s.z.len();
    }

    fn studies(&self) -> u64 {
        self.total[0] + self.total[1]
    }

    /// Is the estimates' bin j beyond the critical values?
    fn bin_significant(&self, j: usize) -> bool {
        let g = self.first + j as i64;
        // (bin g runs from g w to (g + 1) w: beyond c = k w, or wholly below −c)
        g >= self.k || g < -self.k
    }

    /// The share of the significant studies that are real effects (None if there are none).
    fn real_share(&self) -> Option<f64> {
        let sig = self.sig[0] + self.sig[1];
        (sig > 0).then(|| self.sig[1] as f64 / sig as f64)
    }

    /// The average estimate of the significant real effects (None if there are none).
    fn sig_mean(&self) -> Option<f64> {
        (self.sig[1] > 0).then(|| self.sig_sum / self.sig[1] as f64)
    }

    /// The chi-square statistic of the true nulls' p-values against twenty equal
    /// bars, and how many of them there are.
    fn level_check(&self) -> (f64, u64) {
        let total = self.total[0];
        let expect = total as f64 / BINS as f64;
        if total == 0 {
            return (0.0, 0);
        }
        (self.p_bins[0].iter().map(|&o| (o as f64 - expect).powi(2) / expect).sum(), total)
    }
}

// ----- a family of tests --------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct FamilyKey {
    m: usize,
    real: usize,
    shift: f64,
    alpha: f64,
    q: f64,
    seed: u64,
}

impl FamilyKey {
    fn of(s: &Setup, seed: u64) -> FamilyKey {
        FamilyKey { m: s.m, real: s.real_tests(), shift: s.shift(), alpha: s.alpha, q: s.q, seed }
    }
}

/// One rule's results over the families so far.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct RuleTally {
    /// Families with at least one false alarm.
    any_false: u64,
    /// The share of false alarms among the declared, added up over the families (a
    /// family that declares nothing counts 0).
    false_share: f64,
    /// Real effects declared, and tests declared.
    found: u64,
    declared: u64,
}

/// What one rule declares in one family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Verdict {
    declared: usize,
    false_alarms: usize,
    missed: usize,
}

/// Families of m tests, the first `real` of each a real effect.
struct Families {
    key: FamilyKey,
    rng: Rng,
    done: u64,
    tally: [RuleTally; 3],
    /// The newest family's p-values, test by test, and the same sorted.
    latest: Vec<f64>,
    sorted: Vec<f64>,
}

impl Families {
    fn new(key: FamilyKey) -> Families {
        Families { key, rng: Rng::new(key.seed), done: 0, tally: [RuleTally::default(); 3], latest: Vec::new(), sorted: Vec::new() }
    }

    fn step(&mut self) {
        let k = self.key;
        self.latest.clear();
        for i in 0..k.m {
            let shift = if i < k.real { k.shift } else { 0.0 };
            self.latest.push(p_of(shift + st::normal(&mut self.rng)));
        }
        self.sorted.clone_from(&self.latest);
        self.sorted.sort_by(f64::total_cmp);
        for (r, rule) in Rule::ALL.iter().enumerate() {
            let v = verdict(&self.latest, &self.sorted, k.real, *rule, k.alpha, k.q);
            let t = &mut self.tally[r];
            t.any_false += (v.false_alarms > 0) as u64;
            t.false_share += v.false_alarms as f64 / v.declared.max(1) as f64;
            t.found += (v.declared - v.false_alarms) as u64;
            t.declared += v.declared as u64;
        }
        self.done += 1;
    }
}

/// What `rule` declares among the p-values `p` (the first `real` of them real
/// effects; `sorted` the same in rising order).
fn verdict(p: &[f64], sorted: &[f64], real: usize, rule: Rule, alpha: f64, q: f64) -> Verdict {
    let t = rule.threshold(sorted, alpha, q);
    let declared = p.iter().filter(|&&v| v <= t).count();
    let found = p.iter().take(real).filter(|&&v| v <= t).count();
    Verdict { declared, false_alarms: declared - found, missed: real.min(p.len()) - found }
}

// ----- peeking ------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct PeekKey {
    first: usize,
    last: usize,
    alpha: f64,
    seed: u64,
}

/// The z of one study at each look, from the first (its p-value is `p_of(z)`), and
/// the first look (as a place in that list) at which its p-value was below α.
#[derive(Clone, Debug, Default, PartialEq)]
struct Path {
    z: Vec<f32>,
    dip: Option<usize>,
}

/// How many paths are drawn: the newest growing, the ones before it fainter.
const SLOTS: usize = 6;

/// How far down the peeking chart's p-value axis goes, in z: to p = 0.004 (lower
/// p-values are drawn at its foot). The axis is even in z, not in p: near p = 1 the
/// p-value swings with every value, while z, and so the evidence, barely moves.
const PEEK_DEPTH: f64 = 2.9;

/// The p-values named on that axis.
const PEEK_TICKS: [f64; 6] = [1.0, 0.5, 0.2, 0.1, 0.05, 0.01];

/// Studies of pure noise, each tested at every look from `first` to `last`.
struct Peeks {
    key: PeekKey,
    done: u64,
    /// How many studies first went below α at each look, from the first look on; and
    /// how many were below it at the last look.
    dips: Vec<u32>,
    below_last: u64,
    /// The paths on display, and which of them is the newest.
    paths: Vec<Path>,
    newest: Option<usize>,
    /// How long the newest has been growing (seconds of the scene's time), and how long
    /// it takes to grow.
    age: f32,
    grow: f32,
}

impl Peeks {
    fn new(key: PeekKey) -> Peeks {
        Peeks { key, done: 0, dips: vec![0; key.last + 1 - key.first], below_last: 0, paths: Vec::new(), newest: None, age: 0.0, grow: 1.0 }
    }

    /// Is the newest path drawn in full, so that the next study can be watched?
    fn wants_path(&self) -> bool {
        self.newest.is_none() || self.age >= self.grow
    }

    /// Run one more study; with `record`, keep its path to be drawn.
    fn step(&mut self, record: bool, grow: f32) {
        let (dip, below, path) = peek(self.key, self.done, record);
        if let Some(d) = dip {
            if let Some(c) = self.dips.get_mut(d) {
                *c += 1;
            }
        }
        self.below_last += below as u64;
        if let Some(path) = path {
            let slot = match self.newest {
                Some(k) if self.paths.len() >= SLOTS => (k + 1) % SLOTS,
                _ => self.paths.len(),
            };
            if slot < self.paths.len() {
                self.paths[slot] = path;
            } else {
                self.paths.push(path);
            }
            (self.newest, self.age, self.grow) = (Some(slot), 0.0, grow);
        }
        self.done += 1;
    }

    /// The share of the studies that have been below α by look `n`.
    fn share_by(&self, n: usize) -> f64 {
        if self.done == 0 || n < self.key.first {
            return 0.0;
        }
        let upto = (n - self.key.first + 1).min(self.dips.len());
        self.dips[..upto].iter().map(|&c| c as u64).sum::<u64>() as f64 / self.done as f64
    }

    /// Of the studies that went below α at some look, the share above it again at the
    /// last look (None if none went below).
    fn back_above(&self) -> Option<f64> {
        let dipped: u64 = self.dips.iter().map(|&c| c as u64).sum();
        (dipped > 0).then(|| (dipped - self.below_last.min(dipped)) as f64 / dipped as f64)
    }

    /// Is the newest path still growing?
    fn growing(&self) -> bool {
        self.newest.is_some() && self.age < self.grow
    }
}

/// Study number `i` of pure noise, looked at after every value from the first look
/// to the last: the first look below α (as a place from the first look), whether it
/// is below α at the last look, and its path if asked for. Each study has a generator
/// of its own, so that the count does not depend on which studies are drawn.
fn peek(key: PeekKey, i: u64, record: bool) -> (Option<usize>, bool, Option<Path>) {
    let mut rng = Rng::new(key.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i.wrapping_mul(0xD1B5_4A32_D192_ED03));
    let c = critical(key.alpha);
    let mut path = record.then(|| Path { z: Vec::with_capacity(key.last + 1 - key.first), dip: None });
    let (mut sum, mut dip, mut below) = (0.0f64, None, false);
    for n in 1..=key.last {
        sum += rng.normal() as f64;
        if n < key.first {
            continue;
        }
        let z = sum / (n as f64).sqrt();
        below = z.abs() >= c;
        if dip.is_none() && below {
            dip = Some(n - key.first);
        }
        if let Some(path) = &mut path {
            path.z.push(z as f32);
        }
    }
    if let Some(path) = &mut path {
        path.dip = dip;
    }
    (dip, below, path)
}

/// The dots where paths first went below α: a point batch of their own, since they come
/// and go with the paths (in the batch of the share chart's marks they would leave the
/// marks to be taken for one another from frame to frame).
const DIP_DOTS: PointStyle = PointStyle { size_scale: 1.0001, intensity: 1.0, outline_px: 0.0, world_size: false, outline: Color::hex(0x0B0E14) };

/// A dot in the batch of [`DIP_DOTS`], at a point of the screen.
fn dip_dot(f: &mut Frame, at: Vec2, size: f32, color: Color) {
    if at.is_finite() {
        let p = f.hud(at);
        f.dl.points_with(PointKind::Overlay, DIP_DOTS).push(p, size, color);
    }
}

/// The looks the share of studies below α is drawn at: evenly spaced in log n.
const SHARE_POINTS: usize = 90;

/// The looks whose shares are written on the picture (those within the range).
const MARK_LOOKS: [usize; 4] = [10, 50, 100, 500];

// ----- the experiments --------------------------------------------------------------------

struct Trial {
    button: &'static str,
    look_for: &'static str,
    tab: Tab,
    setup: Setup,
    rule: Rule,
    /// How many studies or families to run, and how many a second.
    goal: usize,
    rate: f32,
}

/// The settings the scene opens with, and the experiments start from.
const OPENING: Setup = Setup { n: 16, effect: 0.7, real: 0.2, alpha: 0.05, m: 20, q: 0.1, first: 10, last: 500 };

/// The seed every experiment runs with, so that it shows the same thing every time.
const FIRST_SEED: u64 = 1;

const TRIALS: [Trial; 6] = [
    Trial {
        button: "nothing going on",
        look_for: r"Ten thousand studies, and in every one the null is true: the mean is 0. Their $p$-values fill the twenty bars evenly, a twentieth of the studies in each twentieth of the range, give or take the wobble of chance (here from 4.3 % to 5.4 %; the chi-square test under Numbers finds them no more uneven than chance makes bars). So the 5 % below $\alpha = 0.05$ are false alarms, and each looks exactly like a discovery.",
        tab: Tab::Studies,
        setup: Setup { real: 0.0, ..OPENING },
        rule: Rule::Bonferroni,
        goal: 10_000,
        rate: 2500.0,
    },
    Trial {
        button: r"what “significant” is worth",
        look_for: r"One hypothesis in ten is a real effect, and the test of each has power 0.80 ($n = 16$, effect 0.70). Then $0.1 \times 0.8 = 0.08$ of all the studies are real effects that come out significant, and $0.9 \times 0.05 = 0.045$ are true nulls that do. So only $0.08 / 0.125 = 0.64$ of the significant results are real, and the picture finds 64 % of them real. Lower the effect to 0.28 (power 0.20): 0.02 against 0.045, and the share falls to 31 %, in the picture too.",
        tab: Tab::Studies,
        setup: Setup { real: 0.1, effect: 0.7, n: 16, ..OPENING },
        rule: Rule::Bonferroni,
        goal: 10_000,
        rate: 2500.0,
    },
    Trial {
        button: "the winner’s curse",
        look_for: r"Every study measures a real but small effect, 0.25, with $n = 16$: power 0.17. The 17 % that come out significant estimate it at 0.61 on average, more than twice its size, and a few (under 1 %) even have the wrong sign. Significance picked out the studies whose noise happened to push the estimate past the critical value.",
        tab: Tab::Studies,
        setup: Setup { real: 1.0, effect: 0.25, n: 16, ..OPENING },
        rule: Rule::Bonferroni,
        goal: 10_000,
        rate: 2500.0,
    },
    Trial {
        button: "twenty tests of nothing",
        look_for: r"Families of twenty tests in which every null is true. At least one test comes out significant in $1 - 0.95^{20} = 0.64$ of the families (here 65 % of 4 000). Each test keeps its promise of 5 %; the family does not.",
        tab: Tab::Family,
        setup: Setup { real: 0.0, m: 20, ..OPENING },
        rule: Rule::Plain,
        goal: 4000,
        rate: 800.0,
    },
    Trial {
        button: "correcting for twenty tests",
        look_for: r"Twenty tests, five of them real effects, each with power 0.71 at $\alpha = 0.05$ ($n = 25$, effect 0.5). Uncorrected, 54 % of the families have a false alarm. Bonferroni tests each at $\alpha/m = 0.0025$: now 3.8 % do, under 5 %, but it finds only 30 % of the real effects. Benjamini–Hochberg at $q = 0.1$ finds 54 % of them, and false alarms are 7.6 % of what it declares, on average. Click a rule in the table to light up what it declares.",
        tab: Tab::Family,
        setup: Setup { real: 0.25, m: 20, n: 25, effect: 0.5, q: 0.1, ..OPENING },
        rule: Rule::Bonferroni,
        goal: 4000,
        rate: 800.0,
    },
    Trial {
        button: "peeking",
        look_for: r"Studies of pure noise, each tested again after every new value from the 10th to the 500th, and stopped at the first $p \le 0.05$. Any one look has a 5 % chance of a false alarm, but by the 100th value 28 % of the studies have dipped below 0.05 at some look, and by the 500th, 40 %. Most of those that dip (here 89 %) are back above 0.05 at the last look: the peeker stopped at a passing low.",
        tab: Tab::Peeking,
        setup: Setup { first: 10, last: 500, ..OPENING },
        rule: Rule::Bonferroni,
        goal: 4000,
        rate: 800.0,
    },
];

// ----- what is drawn, gliding ---------------------------------------------------------------

/// The numbers the picture is drawn from, gliding to where the counts have gone.
#[derive(Default)]
struct Look {
    /// The p-values' bars (true nulls', then real effects' shares), the estimates' bars
    /// (the same), and the two charts' tops with the average significant estimate.
    p_bars: Glide,
    e_bars: Glide,
    marks: Glide,
    /// The shown family: its sorted p-values (log10, by rank), how lit each cell is,
    /// and where Benjamini and Hochberg's last declared rank is.
    dots: Glide,
    cells: Glide,
    cut: Glide,
    /// The share of peeked-at studies that have been below α by each look, and the top
    /// of its chart.
    share: Glide,
    moving: bool,
    /// The tops of the charts that grow (p-values, estimates, the peeking share), as
    /// last chosen: see [`top_of`].
    tops: [f32; 3],
    /// Where the line of the significant estimates' average is aimed: following the
    /// running average through a slower ease, since that moves by a random step with
    /// every study (its label says the number itself).
    aim: Option<f32>,
}

/// A chart's top for its highest value `high`: rounded up to a round number, and kept
/// where it was (`kept`) while that still has room and is not far too high, so that
/// the axis does not creep with every new study but moves now and then, gliding.
/// `fresh`: a run just begun, whose counts are still all over the place, keeps the top
/// it has (a bar taller than the chart is cut at its top for those few frames).
fn top_of(high: f32, kept: f32, fresh: bool) -> f32 {
    // (a new top leaves room to grow into, so that the next study does not need another)
    let roomy = high * 1.12;
    let step = nice_step((roomy / 12.0).max(1e-6));
    let round = (roomy / step).ceil().max(1.0) * step;
    if kept > 0.0 && (fresh || (kept >= high && kept <= high * 1.35)) {
        kept
    } else {
        round
    }
}

// ----- the scene ----------------------------------------------------------------------------

pub struct PValues {
    tab: Tab,
    s: Setup,
    rule: Rule,
    /// For each picture: things a second, how many in all, the seed, and the pace.
    rates: [f32; 3],
    goals: [usize; 3],
    seeds: [u64; 3],
    paces: [Pace; 3],
    studies: Studies,
    counts: Counts,
    families: Families,
    /// The family on display, test by test, and how long it has been shown (seconds of
    /// the scene's time); and how many families had been run when it was taken.
    shown: Vec<f64>,
    shown_age: f32,
    shown_done: u64,
    peeks: Peeks,
    look: Look,
    wraps: Wraps,
    /// The y axes of the charts whose tops glide (p-values, estimates, the peeking
    /// share), as last drawn: see [`Grid`].
    grids: [Grid; 3],
    /// Whether each chart's grid was chosen afresh this frame, and how many points their
    /// batches and how many labels the last frame drew (see the end of `frame`).
    regridded: [bool; 3],
    drawn_last: ([usize; 3], usize),
}

/// The line batches of the three gliding charts' grids (the paths of the studies peeked
/// at are `OWN + 1` on).
const GRIDS: [u8; 3] = [OWN + 10, OWN + 11, OWN + 12];

/// The family on display changes at most this often (seconds): the counts take in
/// every family, the eye a few a second.
const SHOW_EVERY: f32 = 0.45;

impl PValues {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> PValues {
        let s = OPENING;
        let seeds = [FIRST_SEED; 3];
        let mut scene = PValues {
            tab: Tab::Studies,
            s,
            rule: Rule::Bh,
            rates: [2500.0, 800.0, 800.0],
            goals: [10_000, 4000, 4000],
            seeds,
            paces: [Pace::default(), Pace::default(), Pace::default()],
            studies: Studies::new(StudyKey { n: s.n, effect: s.effect, real: s.real, seed: seeds[0] }),
            counts: Counts::default(),
            families: Families::new(FamilyKey::of(&s, seeds[1])),
            shown: Vec::new(),
            shown_age: 0.0,
            shown_done: 0,
            peeks: Peeks::new(PeekKey { first: s.first, last: s.last, alpha: s.alpha, seed: seeds[2] }),
            look: Look::default(),
            wraps: Wraps::default(),
            grids: [Grid::default(); 3],
            regridded: [false; 3],
            drawn_last: ([0; 3], 0),
        };
        scene.refresh();
        scene
    }

    /// Set up one of the experiments in `TRIALS`: its picture, every setting, and its
    /// seed, run from the start.
    fn set_up(&mut self, k: usize) {
        let Some(t) = TRIALS.get(k) else { return };
        self.tab = t.tab;
        self.s = t.setup;
        self.rule = t.rule;
        let i = t.tab.index();
        (self.goals[i], self.rates[i], self.seeds[i]) = (t.goal, t.rate, FIRST_SEED);
        self.refresh();
        // (the same seed and settings as before still start again from nothing)
        self.restart(t.tab);
    }

    /// Run the picture's studies or families again from nothing, with its present seed.
    fn restart(&mut self, tab: Tab) {
        match tab {
            Tab::Studies => self.studies = Studies::new(self.studies.key),
            Tab::Family => {
                self.families = Families::new(self.families.key);
                (self.shown, self.shown_done) = (Vec::new(), 0);
            }
            Tab::Peeking => self.peeks = Peeks::new(self.peeks.key),
        }
        self.paces[tab.index()].reset();
    }

    /// The settings in their ranges, and each picture's studies started again if what
    /// they depend on has changed.
    fn refresh(&mut self) {
        self.s.tidy();
        for i in 0..3 {
            self.rates[i] = if self.rates[i].is_finite() { self.rates[i].max(1.0).min(20_000.0) } else { 100.0 };
            self.goals[i] = self.goals[i].clamp(10, 100_000);
        }
        let s = self.s;
        let study = StudyKey { n: s.n, effect: s.effect, real: s.real, seed: self.seeds[0] };
        if self.studies.key != study {
            self.studies = Studies::new(study);
            self.paces[0].reset();
        }
        let family = FamilyKey::of(&s, self.seeds[1]);
        if self.families.key != family {
            self.families = Families::new(family);
            (self.shown, self.shown_done) = (Vec::new(), 0);
            self.paces[1].reset();
        }
        let peek = PeekKey { first: s.first, last: s.last, alpha: s.alpha, seed: self.seeds[2] };
        if self.peeks.key != peek {
            self.peeks = Peeks::new(peek);
            self.paces[2].reset();
        }
        self.counts.catch_up(&self.studies, s.alpha);
    }

    /// Is the picture's run still going?
    fn running(&self) -> bool {
        let goal = self.goals[self.tab.index()] as u64;
        match self.tab {
            Tab::Studies => (self.studies.z.len() as u64) < goal,
            Tab::Family => self.families.done < goal,
            Tab::Peeking => self.peeks.done < goal,
        }
    }

    /// The picture's studies or families that `dt` seconds of the scene's time ask for,
    /// within the frame's allowance.
    fn work(&mut self, dt: f32, real_dt: f32) {
        let i = self.tab.index();
        let (rate, goal) = (self.rates[i], self.goals[i] as u64);
        let mut turn = scene_turn(&mut self.paces[i], rate, dt, real_dt);
        match self.tab {
            Tab::Studies => {
                while (self.studies.z.len() as u64) < goal && turn.next() {
                    self.studies.step();
                }
            }
            Tab::Family => {
                while self.families.done < goal && turn.next() {
                    self.families.step();
                }
            }
            Tab::Peeking => {
                // (each path grows for as long as a study takes at this rate, and for
                // a second at the least, so that it can be watched)
                let grow = (1.0 / rate).max(1.0).min(3.0);
                while self.peeks.done < goal && turn.next() {
                    let record = self.peeks.wants_path();
                    self.peeks.step(record, grow);
                }
            }
        }
        self.paces[i].end(turn);
        self.counts.catch_up(&self.studies, self.s.alpha);
    }

    /// Run the picture's studies or families to the end at once (the tests of this file).
    #[cfg(test)]
    fn settle(&mut self) {
        self.refresh();
        let goal = self.goals[self.tab.index()] as u64;
        match self.tab {
            Tab::Studies => {
                while (self.studies.z.len() as u64) < goal {
                    self.studies.step();
                }
            }
            Tab::Family => {
                while self.families.done < goal {
                    self.families.step();
                }
            }
            Tab::Peeking => {
                while self.peeks.done < goal {
                    self.peeks.step(false, 1.0);
                }
            }
        }
        self.counts.catch_up(&self.studies, self.s.alpha);
    }

    /// The scene's time moves on: the family on display, and the path that grows.
    fn advance(&mut self, dt: f32) {
        let dt = dt.max(0.0);
        self.shown_age += dt;
        if self.families.done > self.shown_done && (self.shown_age >= SHOW_EVERY || self.shown.is_empty()) {
            self.shown.clone_from(&self.families.latest);
            (self.shown_age, self.shown_done) = (0.0, self.families.done);
        }
        if self.tab == Tab::Peeking {
            self.peeks.age = (self.peeks.age + dt).min(self.peeks.grow);
        }
    }

    /// What the picture shows, in words, under the buttons that choose it.
    fn headline(&self) -> &'static str {
        match self.tab {
            Tab::Studies => r"Studies one after another, each testing whether its $n$ values have mean 0. In the purple ones the null is true: nothing is going on. The green ones are of a real effect.",
            Tab::Family => r"A family of $m$ tests done at once, the first of them real effects. Which ones does each rule declare?",
            Tab::Peeking => r"Studies of pure noise, each tested again after every new value. A peeker stops at the first look with $p \le \alpha$.",
        }
    }

    /// The shown family sorted: (p-value, test) by rank.
    fn shown_sorted(&self) -> Vec<(f64, usize)> {
        let mut v: Vec<(f64, usize)> = self.shown.iter().copied().zip(0..).collect();
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        v
    }

    /// The lowest power of ten on the family's p-value axis.
    fn p_floor(&self) -> f64 {
        let lowest = (self.s.alpha / self.s.m as f64 * 0.4).min(1e-4);
        lowest.log10().floor().max(-8.0)
    }

    /// The numbers the picture glides to, and the glide. `room`: how much taller than
    /// its highest bar the estimates' chart is, to leave its labels clear of the bars.
    fn follow(&mut self, dt: f32, room: f32) {
        let mut moving = false;
        match self.tab {
            Tab::Studies => {
                let (c, look) = (&self.counts, &mut self.look);
                // (shares in percent: a glide is there when within 1e-4 of where it goes,
                // which must be well under a pixel)
                let total = c.studies().max(1) as f32 / 100.0;
                let p: Vec<f32> = (0..2).flat_map(|r| c.p_bins[r].iter().map(move |&k| k as f32 / total)).collect();
                let e: Vec<f32> = (0..2).flat_map(|r| c.est.iter().map(move |b| b[r] as f32 / total)).collect();
                let fresh = c.studies() < 1000;
                let p_top = top_of((0..BINS).map(|j| p[j] + p[BINS + j]).fold(160.0 / BINS as f32, f32::max) * 1.08, look.tops[0], fresh);
                let bins = c.est.len();
                let e_top = top_of((0..bins).map(|j| e[j] + e[bins + j]).fold(2.0, f32::max) * room, look.tops[1], fresh);
                look.tops[0] = p_top;
                look.tops[1] = e_top;
                let mean = c.sig_mean().unwrap_or(self.s.effect) as f32;
                let aim = match look.aim {
                    Some(a) if !fresh && (mean - a).abs() > 1e-5 => a + (mean - a) * (1.0 - (-dt.max(0.0) / 0.3).exp()),
                    _ => mean,
                };
                look.aim = Some(aim);
                // (a new layout of the bars starts where it is, not from the last one's)
                let keep = look.e_bars.shown.len() == e.len();
                moving |= look.p_bars.follow(&p, dt, true);
                moving |= look.e_bars.follow(&e, dt, keep);
                moving |= look.marks.follow(&[p_top, e_top, aim], dt, true) || aim != mean;
            }
            Tab::Family => {
                let floor = self.p_floor();
                let sorted = self.shown_sorted();
                let dots: Vec<f32> = sorted.iter().map(|(p, _)| p.max(10f64.powf(floor)).log10() as f32).collect();
                let ps: Vec<f64> = sorted.iter().map(|(p, _)| *p).collect();
                let t = self.rule.threshold(&ps, self.s.alpha, self.s.q);
                let cells: Vec<f32> = if self.shown.is_empty() { vec![0.0; self.s.m] } else { self.shown.iter().map(|&p| (p <= t) as u8 as f32).collect() };
                let cut = bh_count(&ps, self.s.q) as f32 + 0.5;
                let look = &mut self.look;
                moving |= look.dots.follow(&dots, dt, true);
                moving |= look.cells.follow(&cells, dt, true);
                moving |= look.cut.follow(&[cut], dt, true);
            }
            Tab::Peeking => {
                let looks = share_looks(&self.peeks.key);
                // (in percent, as above)
                let mut share: Vec<f32> = looks.iter().map(|&n| 100.0 * self.peeks.share_by(n) as f32).collect();
                let top = top_of(share.iter().copied().fold(8.0, f32::max) * 1.08, self.look.tops[2], self.peeks.done < 300).min(105.0);
                self.look.tops[2] = top;
                share.push(top);
                share.extend(MARK_LOOKS.iter().map(|&n| 100.0 * self.peeks.share_by(n) as f32));
                let look = &mut self.look;
                moving |= look.share.follow(&share, dt, true);
            }
        }
        self.look.moving = moving;
    }
}

/// The looks the share curve is drawn at: from the first to the last, evenly in log n.
fn share_looks(key: &PeekKey) -> Vec<usize> {
    let (a, b) = ((key.first as f64).ln(), (key.last as f64).ln());
    (0..SHARE_POINTS).map(|i| (a + (b - a) * i as f64 / (SHARE_POINTS - 1) as f64).exp().round() as usize).map(|n| n.max(key.first).min(key.last)).collect()
}

// ----- words on the picture -------------------------------------------------------------------

/// A share as a percentage: one decimal below 10 %, none above.
fn pct(v: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    let x = v * 100.0;
    format!("{}\u{a0}%", num(x, if x < 9.95 { 1 } else { 0 }))
}

/// A small number with up to three significant digits and no trailing zeros: 0.05,
/// 0.0025, 0.1.
fn short(v: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    if v == 0.0 {
        return "0".to_string();
    }
    let digits = (2 - v.abs().log10().floor() as i32).clamp(0, 9) as usize;
    let text = num(v, digits);
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

/// Texts broken into lines no wider than a width, kept while they are drawn.
#[derive(Default)]
struct Wraps {
    kept: Vec<(String, u32, u32, Vec<String>)>,
}

impl Wraps {
    fn lines(&mut self, f: &Frame, text: &str, size: f32, width: f32) -> Vec<String> {
        let key = (size.to_bits(), width.round().to_bits());
        if let Some(found) = self.kept.iter().find(|k| k.0 == text && (k.1, k.2) == key) {
            return found.3.clone();
        }
        let mut lines: Vec<String> = vec![String::new()];
        for word in words_of(text) {
            let Some(last) = lines.last_mut() else { break };
            if last.is_empty() {
                last.push_str(word);
                continue;
            }
            let longer = format!("{last} {word}");
            if f.text_size(&longer, size).x > width {
                lines.push(word.to_string());
            } else {
                *last = longer;
            }
        }
        if self.kept.len() > 32 {
            self.kept.clear();
        }
        self.kept.push((text.to_string(), key.0, key.1, lines.clone()));
        lines
    }

    /// Draw `text` from `at` down, in lines no wider than `width`; the height taken.
    #[allow(clippy::too_many_arguments)]
    fn draw(&mut self, f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> f32 {
        let lines = self.lines(f, text, size, width);
        let step = size + 5.0;
        for (k, line) in lines.iter().enumerate() {
            f.text_px(at + Vec2::new(0.0, k as f32 * step), line.as_str(), size, color, Align2::LEFT_TOP);
        }
        lines.len() as f32 * step
    }
}

/// The words of a text, each with its formulas whole.
fn words_of(text: &str) -> Vec<&str> {
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

/// The first of `options` that fits in `room` pixels at `size` (the last if none does).
fn pick<'t>(f: &Frame, options: &[&'t str], room: f32, size: f32) -> &'t str {
    options.iter().copied().find(|t| f.text_size(t, size).x <= room).or(options.last().copied()).unwrap_or("")
}

// ----- the layout -------------------------------------------------------------------------------

/// Where everything goes on a canvas.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    /// A narrow canvas (smaller words), and one whose charts go one under the other (a
    /// phone held upright; one held on its side has them side by side).
    narrow: bool,
    stacked: bool,
    /// Text sizes: headings, words, small print.
    big: f32,
    text: f32,
    small: f32,
    /// The three buttons that choose the picture.
    tabs: [Rect; 3],
    /// The line under them that says what the picture shows.
    head: Vec2,
    head_w: f32,
    /// The two charts' rectangles (their axes' numbers and names go round them).
    charts: [Rect; 2],
    /// The numbers and the key, under the charts.
    below: Rect,
    /// Too small for the charts: the numbers alone.
    compact: bool,
}

impl Layout {
    /// The layout of a canvas `view` (logical pixels), below `inset` pixels taken by the
    /// application's own buttons, with `head_lines` lines of words under the buttons.
    fn of(view: Vec2, inset: f32, head_lines: usize) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 760.0;
        let (big, text, small) = if narrow { (12.5, 11.5, 10.5) } else { (14.0, 12.5, 11.5) };
        let side = if narrow { 10.0 } else { 24.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let tab_h = if narrow { 26.0 } else { 28.0 };
        let tabs_w = (view.x - 2.0 * side).min(if narrow { 560.0 } else { 640.0 });
        let row = plot::even_row(3, side, side + tabs_w, top, tab_h, 6.0);
        let tabs = [row[0], row[1], row[2]];
        let head = Vec2::new(side, top + tab_h + 10.0);
        let head_w = view.x - 2.0 * side;
        // (two lines of words under the buttons, then each chart's title and the name of
        // its y axis above it, its numbers to its left and below it)
        let head_h = head_lines.max(1) as f32 * (text + 5.0);
        let (above, under, left) = (42.0, 34.0, 40.0);
        let stacked = !(view.x >= 760.0 || (view.x >= 440.0 && view.x > view.y));
        // (the numbers under the charts: fewer of them on a short canvas, the rows that do
        // not fit left out)
        let below_h = if stacked { 112.0 } else { (view.y * 0.2).max(44.0).min(128.0) };
        let charts_top = head.y + head_h + 6.0 + above;
        let (charts, below) = if stacked {
            let each = ((view.y - charts_top - 2.0 * under - above - below_h - 6.0) * 0.5).max(0.0);
            let a = (Vec2::new(side + left, charts_top), Vec2::new(view.x - side - 6.0, charts_top + each));
            let y = a.1.y + under + above;
            let b = (Vec2::new(side + left, y), Vec2::new(view.x - side - 6.0, y + each));
            let y = b.1.y + under;
            ([a, b], (Vec2::new(side, y), Vec2::new(view.x - side, view.y - 6.0)))
        } else {
            let height = (view.y - charts_top - under - below_h - 10.0).max(0.0);
            let gap = 44.0;
            let width = ((view.x - 2.0 * side - gap) * 0.5).max(0.0);
            let a = (Vec2::new(side + left, charts_top), Vec2::new(side + width - 8.0, charts_top + height));
            let x = side + width + gap;
            let b = (Vec2::new(x + left, charts_top), Vec2::new(view.x - side - 8.0, charts_top + height));
            let y = charts_top + height + under + 4.0;
            ([a, b], (Vec2::new(side, y), Vec2::new(view.x - side, view.y - 8.0)))
        };
        let small_chart = |r: &Rect| r.1.x - r.0.x < 80.0 || r.1.y - r.0.y < 50.0;
        let compact = charts.iter().any(small_chart);
        Layout { view, narrow, stacked, big, text, small, tabs, head, head_w, charts, below, compact }
    }
}

// ----- the panel ----------------------------------------------------------------------------

const INTRO: &str = r"A test's $p$-value is the chance, if nothing is going on (the null hypothesis is true), of a result at least as far from nothing as the one seen. So when nothing is going on, the $p$-value is equally likely to be anywhere from 0 to 1, and a test that calls $p \le \alpha$ significant gives a false alarm with probability $\alpha$. That is a promise about one test, looked at once. Here: many studies, many tests at once, and one study looked at again and again.";

const THE_TEST: &str = r"Every test here takes $n$ values, each normal with spread 1, and asks whether their mean $\bar x$ is 0: $z = \bar x \sqrt{n}$, and $p = 2\,\Phi(-|z|)$. A real effect is a mean of $d$ in place of 0.";

impl Scene for PValues {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.running() || self.look.moving || (self.tab == Tab::Peeking && self.peeks.growing()) || (self.tab == Tab::Family && self.families.done > self.shown_done)
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some(r"buttons on the picture: click    pink dot under the $p$-values (many studies): drag to move $\alpha$")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, INTRO);

        ui::section(ui, "Things to try");
        for (k, t) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, t.button, t.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The picture");
        let mut tab = self.tab.index();
        if ui::choice(ui, &["many studies", "many tests at once", "peeking"], &mut tab) {
            self.tab = Tab::ALL[tab.min(2)];
        }
        let i = self.tab.index();
        if self.tab != Tab::Peeking {
            ui::note(ui, THE_TEST);
            if self.tab == Tab::Family {
                let mut m = self.s.m;
                if ui::slider_int(ui, "tests in a family, $m$", &mut m, 1..=200) {
                    self.s.m = m;
                }
            }
            let mut real = self.s.real as f32;
            if ui::slider(ui, "share of real effects", &mut real, 0.0..=1.0) {
                self.s.real = real as f64;
            }
            let mut effect = self.s.effect as f32;
            if ui::slider(ui, "size of a real effect, $d$", &mut effect, 0.0..=1.5) {
                self.s.effect = effect as f64;
            }
            ui::note(ui, "In spreads of one value. At 0 a “real” effect is no effect at all, and its tests are true nulls too.");
            let mut n = self.s.n;
            if ui::slider_int_log(ui, "values per test, $n$", &mut n, 1..=400) {
                self.s.n = n;
            }
        } else {
            ui::note(ui, r"Each study draws values of pure noise (mean 0, spread 1) one at a time, and after each one from the first look on tests whether their mean is 0.");
            ui::note(ui, r"With no last look at all, every such study would dip below any $\alpha$ sooner or later: for independent values of the same spread, $z$ wanders ever further now and then (the law of the iterated logarithm).");
            let (mut first, mut last) = (self.s.first, self.s.last);
            if ui::slider_int_log(ui, "first look (values so far)", &mut first, 1..=200) {
                self.s.first = first;
                self.s.last = self.s.last.max(first);
            }
            if ui::slider_int_log(ui, "last look", &mut last, 1..=1000) {
                self.s.last = last.max(self.s.first);
            }
        }
        let mut alpha = self.s.alpha as f32;
        if ui::slider_log(ui, r"level $\alpha$", &mut alpha, 0.001..=0.2) {
            self.s.alpha = alpha as f64;
        }
        if self.tab == Tab::Studies {
            ui::note(ui, r"Or drag the pink dot under the $p$-values. The studies stay as they are; only what is called significant changes.");
        }
        if self.tab == Tab::Family {
            let mut q = self.s.q as f32;
            if ui::slider(ui, "false discovery rate asked for, $q$", &mut q, 0.01..=0.3) {
                self.s.q = q as f64;
            }
            let mut rule = Rule::ALL.iter().position(|r| *r == self.rule).unwrap_or(0);
            if ui::choice(ui, &[Rule::Plain.name(), Rule::Bonferroni.name(), Rule::Bh.name()], &mut rule) {
                self.rule = Rule::ALL[rule.min(2)];
            }
            ui::note(ui, "Which rule's verdicts light the cells. The table under the picture counts all three.");
            if self.s.m == 1 {
                ui::note(ui, r"With one test there is nothing to correct for: Bonferroni is the plain test, and Benjamini–Hochberg the plain test at level $q$.");
            }
        }
        let (per, all) = match self.tab {
            Tab::Studies => ("studies per second", "studies in all"),
            Tab::Family => ("families per second", "families in all"),
            Tab::Peeking => ("studies per second", "studies in all"),
        };
        let range = if self.tab == Tab::Studies { 10.0..=20_000.0 } else { 1.0..=2000.0 };
        ui::slider_log(ui, per, &mut self.rates[i], range);
        let most = if self.tab == Tab::Studies { 100_000 } else { 20_000 };
        ui::slider_int_log(ui, all, &mut self.goals[i], 10..=most);
        if ui.button("run again, with new random numbers").clicked() {
            self.seeds[i] = self.seeds[i].wrapping_add(1);
        }
        self.refresh();

        ui::section(ui, "Reading the picture");
        match self.tab {
            Tab::Studies => ui::key(
                ui,
                &[
                    (NULL, "purple: studies in which the null is true (nothing is going on)"),
                    (REAL, "green: studies of a real effect, and its true size"),
                    (SIGNIFICANT, r"pink: the level $\alpha$, and the critical values; beyond them a result is significant"),
                    (ESTIMATE, "yellow: the average estimate of the significant real effects"),
                ],
            ),
            Tab::Family => ui::key(
                ui,
                &[
                    (REAL, "bright green: a real effect declared (a true discovery)"),
                    (SIGNIFICANT, "pink: a true null declared (a false alarm)"),
                    (REAL.a(0.35), "dim green: a real effect not declared (a miss)"),
                    (NULL.a(0.45), "dim purple: a true null, rightly not declared"),
                    (pal::WHITE, "white lines: the three rules' thresholds, the chosen one bright"),
                ],
            ),
            Tab::Peeking => ui::key(
                ui,
                &[
                    (PATH, r"cyan: one study's $p$-value after each new value"),
                    (SIGNIFICANT, r"pink: below $\alpha$; the dot is the first such look, where a peeker stops; and the share of studies that have been below by each look"),
                    (pal::DIM, "grey: how the study would have gone on"),
                ],
            ),
        }

        ui::section(ui, "Words");
        ui::term(ui, "$p$-value:", "the chance, if the null hypothesis is true, of a result at least as far from it as the one seen.");
        ui::term(ui, "Null hypothesis:", "here, that the mean of the values is 0: nothing is going on. A true null is one that is right.");
        ui::term(ui, "Significant:", r"$p \le \alpha$, for a level $\alpha$ chosen beforehand. A true null called significant is a false alarm.");
        ui::term(ui, "Power:", "the chance that the test of a real effect comes out significant.");
        ui::term(ui, "Critical value:", r"how far from 0 an estimate must be to come out significant: $\Phi^{-1}(1 - \alpha/2) / \sqrt{n}$ either side, 0.49 for $n = 16$ at $\alpha = 0.05$.");
        ui::term(ui, r"$z$ and $\Phi$:", r"$z$ is the mean of the $n$ values over its spread $1/\sqrt{n}$; if the null is true it is normal with mean 0 and spread 1. $\Phi(x)$ is the share of that distribution below $x$.");
        ui::term(ui, "Multiple comparisons:", "many tests at once. The more there are, the likelier it is that some come out significant by chance alone.");
        ui::term(ui, "Family-wise error rate:", "the chance of at least one false alarm in a family of tests.");
        ui::term(ui, "Bonferroni correction:", r"testing each of $m$ tests at $\alpha/m$. The chance of any false alarm is then at most $\alpha$, however the tests depend on each other: it is at most the sum of their chances.");
        ui::term(ui, "False discovery rate:", "the share of false alarms among the tests declared, on average over families (a family that declares nothing counts as 0).");
        ui::term(ui, "Benjamini–Hochberg:", r"sort the $m$ $p$-values; find the last rank $k$ whose $p$-value is at most $kq/m$; declare it and every smaller one, also those above the line. For independent tests the false discovery rate is then $q$ times the share of true nulls, never more than $q$.");
        ui::term(ui, "Optional stopping:", "looking at the data as they come and stopping as soon as the result is significant. Each look is another chance of a false alarm.");
        ui::term(ui, "Winner's curse:", "studies picked out for being significant overestimate their effects on average, the more so the lower their power.");

        ui::section(ui, "Numbers");
        let s = self.s;
        match self.tab {
            Tab::Studies => {
                let c = &self.counts;
                ui::readout(ui, "studies so far", count(c.studies() as usize));
                ui::readout(ui, "power of each test against a real effect", num(power(s.effect, s.n, s.alpha), 3));
                ui::readout(ui, "true nulls called significant", if c.total[0] > 0 { pct(c.sig[0] as f64 / c.total[0] as f64) } else { "none yet".into() });
                ui::readout(ui, "real effects called significant", if c.total[1] > 0 { pct(c.sig[1] as f64 / c.total[1] as f64) } else { "none yet".into() });
                ui::readout(ui, "share of the significant that are real", c.real_share().map_or("none significant".into(), pct));
                ui::readout(ui, "significant real effects' average estimate", c.sig_mean().map_or("none yet".into(), |v| format!("${}$", num(v, 3))));
                let (chi, nulls) = c.level_check();
                if nulls >= 100 {
                    let p = 1.0 - st::Dist::ChiSquare { df: (BINS - 1) as f64 }.cdf(chi);
                    ui::readout(ui, "true nulls' bars against level", format!(r"$\chi^2 = {}$, $p = {}$", num(chi, 1), num(p, 2)));
                    ui::note(ui, "A test of the test: the chi-square statistic of the purple bars against twenty equal ones, on 19 degrees of freedom. A large $p$ says they are no more uneven than chance makes bars.");
                }
            }
            Tab::Family => {
                let fam = &self.families;
                ui::readout(ui, "families so far", count(fam.done as usize));
                ui::readout(ui, "real effects in each", format!("{} of {}", s.real_tests(), s.m));
                ui::readout(ui, r"power of each test at $\alpha$", num(power(s.effect, s.n, s.alpha), 3));
                ui::readout(ui, r"power of each test at $\alpha/m$", num(power(s.effect, s.n, s.alpha / s.m as f64), 3));
                let nulls = s.m - s.real_tests();
                ui::readout(ui, "any false alarm, uncorrected: by formula", pct(1.0 - (1.0 - s.alpha).powi(nulls as i32)));
                ui::readout(ui, "any false alarm, Bonferroni: by formula", pct(1.0 - (1.0 - s.alpha / s.m as f64).powi(nulls as i32)));
            }
            Tab::Peeking => {
                let p = &self.peeks;
                ui::readout(ui, "studies so far", count(p.done as usize));
                ui::readout(ui, r"below $\alpha$ at some look", pct(p.share_by(s.last)));
                ui::readout(ui, r"below $\alpha$ at the last look", if p.done > 0 { pct(p.below_last as f64 / p.done as f64) } else { "none yet".into() });
                ui::readout(ui, "of those that dipped, back above at the last", p.back_above().map_or("none dipped".into(), pct));
            }
        }
        ui::see_also(ui, &[("testing", "one test, its level and its power"), ("chance", "the same arithmetic as a medical test's"), ("anova", "one test of several groups instead of many"), ("modelselect", "the same curse when choosing a model")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let head = self.headline();
        let lay = Layout::of(f.view_size(), f.top_inset, 1);
        let head_lines = self.wraps.lines(f, head, lay.text, lay.head_w).len();
        let lay = Layout::of(f.view_size(), f.top_inset, head_lines);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let tiny = lay.view.x < 160.0 || lay.view.y < 160.0;
        if !tiny {
            self.draw_tabs(f, &lay);
        }
        self.refresh();
        if f.dt > 0.0 {
            self.work(f.dt, f.real_dt);
        }
        self.advance(f.dt);
        // (three rows of labels at the top of the estimates' chart, over no bar)
        let height = lay.charts[1].1.y - lay.charts[1].0.y;
        let room = (height / (height - 3.0 * (lay.small + 8.0) - 8.0).max(height * 0.4)).max(1.12);
        self.follow(f.real_dt, room);
        if tiny {
            return;
        }
        let head = self.headline();
        self.wraps.draw(f, lay.head, lay.head_w, head, lay.text, pal::TEXT);
        if lay.compact {
            let below = lay.head + Vec2::new(0.0, (head_lines as f32 + 0.6) * (lay.text + 5.0));
            if below.y + lay.small < lay.view.y {
                f.text_px(below, "(a larger window shows the charts)", lay.small, pal::DIM, Align2::LEFT_TOP);
            }
            return;
        }
        self.regridded = [false; 3];
        match self.tab {
            Tab::Studies => self.draw_studies(f, &lay),
            Tab::Family => self.draw_family(f, &lay),
            Tab::Peeking => self.draw_peeking(f, &lay),
        }
        // new round values on an axis: make sure the axes' batch and the labels are not
        // as many as in the last frame, so that nothing takes the new ones for the old
        // ones moved (one chart's axis may have lost a value as another gained one)
        let count = |f: &mut Frame| (GRIDS.map(|k| plot::apart(f, k, false).pts.iter().filter(|p| p.width >= 0.0).count()), f.dl.labels.len());
        let (grids, labels) = count(f);
        for k in 0..3 {
            if self.regridded[k] && grids[k] == self.drawn_last.0[k] {
                plot::line_of(f, GRIDS[k], Vec2::ZERO, Vec2::ZERO, 1.0, Color::CLEAR);
            }
        }
        if self.regridded.iter().any(|r| *r) && labels == self.drawn_last.1 {
            f.text_px(Vec2::ZERO, "", 1.0, Color::CLEAR, Align2::LEFT_TOP);
        }
        self.drawn_last = count(f);
    }
}

// ----- drawing ------------------------------------------------------------------------------

/// A y axis of powers of ten, for a chart of the logarithm (base 10) of a number.
fn log_y_axis(plot: &Plot, f: &mut Frame, name: &str, size: f32) {
    if plot.tiny() {
        return;
    }
    let (min, max) = plot.rect;
    plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
    let (lo, hi) = (plot.y[0].ceil() as i64, plot.y[1].floor() as i64);
    // (every power of ten while there is room for it, else every other)
    let every = if (hi - lo) as f32 * 2.2 * size > plot.height() { 2 } else { 1 };
    for k in lo..=hi {
        if (hi - k) % every != 0 {
            continue;
        }
        let y = plot.px_y(k as f64);
        plot::line_of(f, AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8));
        plot::line_of(f, AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.14));
        f.text_px(Vec2::new(min.x - 6.0, y), num(10f64.powi(k as i32), (-k).max(0) as usize), size, pal::DIM, Align2::RIGHT_CENTER);
    }
    if !name.is_empty() {
        f.text_px(Vec2::new(min.x - 30.0, min.y - 6.0), name, size, pal::DIM, Align2::LEFT_BOTTOM);
    }
}

/// An x axis for a chart of ln n, with round values of n.
fn log_x_axis(plot: &Plot, f: &mut Frame, name: &str, size: f32) {
    if plot.tiny() {
        return;
    }
    let (min, max) = plot.rect;
    plot::line_of(f, AXES, Vec2::new(min.x, max.y), max, 1.0, pal::DIM.a(0.8));
    // 1, 2, 5, 10, 20, 50, ...: the 2s left out where they would crowd
    let all: Vec<f64> = (0..4).flat_map(|e| [1.0, 2.0, 5.0].map(|m| m * 10f64.powi(e))).collect();
    let crowded = |step: &[f64]| step.windows(2).any(|w| plot.px_x(w[1].ln()) - plot.px_x(w[0].ln()) < 30.0);
    let inside = |v: &f64| v.ln() >= plot.x[0] - 1e-9 && v.ln() <= plot.x[1] + 1e-9;
    let mut ticks: Vec<f64> = all.iter().copied().filter(inside).collect();
    if crowded(&ticks) {
        ticks.retain(|v| (v.log10().fract() - 2f64.log10()).abs() > 1e-6);
    }
    if crowded(&ticks) {
        ticks.retain(|v| v.log10().fract().abs() < 1e-6);
    }
    for v in ticks {
        let x = plot.px_x(v.ln());
        plot::line_of(f, AXES, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, pal::DIM.a(0.8));
        f.text_px(Vec2::new(x, max.y + 6.0), num(v, 0), size, pal::DIM, Align2::CENTER_TOP);
    }
    if !name.is_empty() {
        f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 8.0 + size), name, size, pal::DIM, Align2::CENTER_TOP);
    }
}

/// The round values on the y axis of a chart whose top glides: chosen from where the
/// top is going (`goal`), with a few beyond it. They stay the same while the axis glides
/// there: each line and number keeps its place in its batch, sliding with the axis,
/// unseen while it is off the chart. (Chosen afresh from the axis as drawn, a line
/// would come in at one end as another went out at the other, and the lines would be
/// taken for one another from frame to frame.)
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Grid {
    step: f64,
    count: usize,
}

impl Grid {
    fn of(goal: f64, pixels: f32) -> Grid {
        let step = plot::Ticks::of(0.0, goal, pixels * 1.6).step;
        if !(step.is_finite() && step > 0.0 && goal.is_finite() && goal > 0.0) {
            return Grid::default();
        }
        Grid { step, count: ((goal / step).floor() as usize + 4).min(64) }
    }

    /// The grid lines, tick marks and numbers on the y axis of `plot`, in the line batch
    /// of `kind` (each such axis has one of its own: one chart's axis given new values
    /// leaves another's, gliding, to be followed from frame to frame).
    fn draw(&self, plot: &Plot, f: &mut Frame, kind: u8, name: &str, size: f32) {
        let (min, max) = plot.rect;
        plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
        let decimals = (-self.step.log10().floor()).max(0.0).min(6.0) as usize;
        for i in 0..self.count {
            let v = i as f64 * self.step;
            let y = plot.px_y(v);
            let seen = y >= min.y - 0.5 && y <= max.y + 0.5;
            let y = y.max(min.y).min(max.y);
            let alpha = if seen { 1.0 } else { 0.0 };
            plot::line_of(f, kind, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8 * alpha));
            plot::line_of(f, kind, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.14 * alpha));
            let text = if seen { num(v, decimals) } else { String::new() };
            f.text_px(Vec2::new(min.x - 6.0, y), text, size, pal::DIM, Align2::RIGHT_CENTER);
        }
        if !name.is_empty() {
            f.text_px(Vec2::new(min.x - 30.0, min.y - 6.0), name, size, pal::DIM, Align2::LEFT_BOTTOM);
        }
    }
}

/// A label on a plate beside a vertical line at screen x, `row` lines down from the
/// top of the chart: on the side asked for (`left`), or on the other where it would
/// run off the chart; the first of `texts` (each shorter than the last) that fits on
/// either side.
#[allow(clippy::too_many_arguments)]
fn line_label(f: &mut Frame, plot: &Plot, x: f32, row: f32, texts: &[String], size: f32, color: Color, left: bool) {
    let (min, max) = plot.rect;
    let y = min.y + 5.0 + row * (size + 8.0);
    let sides = |text: &str| {
        let width = f.text_size(text, size).x;
        (x - 6.0 - width >= min.x, x + 6.0 + width <= max.x)
    };
    let text = texts.iter().find(|t| sides(t).0 || sides(t).1).or(texts.last()).cloned().unwrap_or_default();
    let (fits_left, fits_right) = sides(&text);
    if fits_left && (left || !fits_right) {
        plate(f, Vec2::new(x - 6.0, y), text, size, color, Align2::RIGHT_TOP);
    } else {
        plate(f, Vec2::new(x + 6.0, y), text, size, color, Align2::LEFT_TOP);
    }
}

/// A label on a plate; an empty one draws no plate (but is still a label, so that the
/// picture has as many from frame to frame).
fn plate(f: &mut Frame, at: Vec2, text: impl Into<String>, size: f32, color: Color, align: Align2) {
    let text = text.into();
    if text.is_empty() {
        f.text_px(at, text, size, color, align);
    } else {
        f.caption_px(at, text, size, color, align);
    }
}

/// A p-value as a label writes it: two decimals, or as many as a small one needs.
fn p_text(p: f64) -> String {
    if p >= 0.01 {
        num(p, 2)
    } else {
        short(p)
    }
}

/// A y axis of p-values for a chart of −|z|: even in z, named in p.
fn p_axis(plot: &Plot, f: &mut Frame, name: &str, size: f32) {
    if plot.tiny() {
        return;
    }
    let (min, max) = plot.rect;
    plot::line_of(f, AXES, min, Vec2::new(min.x, max.y), 1.0, pal::DIM.a(0.8));
    let mut last = f32::NEG_INFINITY;
    for p in PEEK_TICKS {
        let y = plot.px_y(-critical(p));
        if !(y >= min.y - 0.5 && y <= max.y + 0.5) || y - last < size + 2.0 {
            continue;
        }
        last = y;
        plot::line_of(f, AXES, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, pal::DIM.a(0.8));
        plot::line_of(f, AXES, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.14));
        f.text_px(Vec2::new(min.x - 6.0, y), short(p), size, pal::DIM, Align2::RIGHT_CENTER);
    }
    if !name.is_empty() {
        f.text_px(Vec2::new(min.x - 30.0, min.y - 6.0), name, size, pal::DIM, Align2::LEFT_BOTTOM);
    }
}

/// Lines of words down the rectangle `area`, each in its colour: as many as fit in it.
fn lines(f: &mut Frame, (min, max): Rect, rows: &[(String, Color)], size: f32) {
    let step = size + 6.0;
    for (k, (text, color)) in rows.iter().enumerate() {
        let at = min + Vec2::new(0.0, k as f32 * step);
        if at.y + size > max.y + 2.0 {
            break;
        }
        f.text_px(at, text.as_str(), size, *color, Align2::LEFT_TOP);
    }
}

impl PValues {
    /// The buttons that choose the picture.
    fn draw_tabs(&mut self, f: &mut Frame, lay: &Layout) {
        let names = [["1 · many studies", "many studies", "studies"], ["2 · many tests at once", "many tests", "tests"], ["3 · peeking", "peeking", "peeking"]];
        let room = lay.tabs[0].1.x - lay.tabs[0].0.x - 12.0;
        // (the first style in which every name fits)
        let style = (0..3).find(|&k| names.iter().all(|n| f.text_size(n[k], lay.text).x <= room)).unwrap_or(2);
        for (k, rect) in lay.tabs.iter().enumerate() {
            let text = names[k][style];
            if plot::button(f, *rect, text, lay.text, self.tab.index() == k) {
                self.tab = Tab::ALL[k];
            }
        }
    }

    /// The y axis of gliding chart `k` (see [`Grid`]), its values chosen from the
    /// percentage the chart's top is going to.
    fn grid(&mut self, f: &mut Frame, k: usize, plot: &Plot, name: &str, size: f32) {
        let grid = Grid::of(self.look.tops[k] as f64 / 100.0, plot.height());
        grid.draw(plot, f, GRIDS[k], name, size);
        self.regridded[k] = grid != self.grids[k];
        self.grids[k] = grid;
    }

    fn draw_studies(&mut self, f: &mut Frame, lay: &Layout) {
        let s = self.s;
        let (big, small) = (lay.big, lay.small);
        let [pr, er] = lay.charts;
        let look = &self.look;

        // where the p-values fall
        let p_top = (look.marks.get(0) as f64 / 100.0).max(0.01);
        let plot = Plot::new(pr, [0.0, 1.0], [0.0, p_top]);
        plot.backing(f, 40.0, 34.0, 44.0, 10.0);
        plot.title(f, "where the studies’ $p$-values fall", big);
                plot.bar(f, 0.0, s.alpha, 0.0, p_top, 0.0, faint_significant());
        for j in 0..BINS {
            let (x0, x1) = (j as f64 / BINS as f64, (j + 1) as f64 / BINS as f64);
            let (a, b) = (look.p_bars.get(j) as f64 / 100.0, look.p_bars.get(BINS + j) as f64 / 100.0);
            plot.bar(f, x0, x1, 0.0, a, 1.0, NULL.a(0.8));
            plot.bar(f, x0, x1, a, a + b, 1.0, REAL.a(0.85));
        }
        let level = (1.0 - s.real) / BINS as f64;
        let level_color = NULL.mix(pal::WHITE, 0.35);
        if s.real < 1.0 {
            plot.hline(f, MARKS, level, 1.5, level_color, true);
        }
        let level_text = if s.real <= 0.0 {
            "level: a twentieth in each bar"
        } else if s.real < 1.0 {
            "the true nulls alone: level"
        } else {
            ""
        };
        plate(f, Vec2::new(pr.1.x - 4.0, plot.px_y(level) - 5.0), level_text, small, level_color, Align2::RIGHT_BOTTOM);
        plot.vline(f, MARKS, s.alpha, 1.8, SIGNIFICANT, false);
        line_label(f, &plot, plot.px_x(s.alpha), 0.0, &[format!(r"$\alpha = {}$", short(s.alpha))], lay.text, SIGNIFICANT, false);
        if let Some((x, _)) = plot.handle(f, 1, s.alpha, 0.0, SIGNIFICANT) {
            self.s.alpha = x.max(0.001).min(0.2);
        }
        plot.x_axis(f, "$p$-value", small);
        self.grid(f, 0, &plot, "share of studies", small);

        // what each study estimated
        let c = &self.counts;
        let look = &self.look;
        let root = (s.n as f64).sqrt();
        let bins = c.est.len();
        let (x0, x1) = (c.lo / root, c.hi / root);
        let e_top = (look.marks.get(1) as f64 / 100.0).max(0.005);
        let plot = Plot::new(er, [x0, x1], [0.0, e_top]);
        plot.backing(f, 40.0, 34.0, 44.0, 10.0);
        plot.title(f, "what each study estimated", big);
                let cv = critical(s.alpha) / root;
        plot.bar(f, x0, -cv, 0.0, e_top, 0.0, faint_significant());
        plot.bar(f, cv, x1, 0.0, e_top, 0.0, faint_significant());
        if look.e_bars.shown.len() == 2 * bins {
            for j in 0..bins {
                let lo = (c.first + j as i64) as f64 * c.w / root;
                let hi = lo + c.w / root;
                let (a, b) = (look.e_bars.get(j) as f64 / 100.0, look.e_bars.get(bins + j) as f64 / 100.0);
                let strength = if c.bin_significant(j) { 0.95 } else { 0.32 };
                plot.bar(f, lo, hi, 0.0, a, 0.5, NULL.a(strength));
                plot.bar(f, lo, hi, a, a + b, 0.5, REAL.a(strength));
            }
        }
        plot.vline(f, MARKS, -cv, 1.4, SIGNIFICANT, true);
        plot.vline(f, MARKS, cv, 1.4, SIGNIFICANT, true);
        line_label(f, &plot, plot.px_x(-cv), 2.0, &[format!(r"significant beyond $\pm{}$", num(cv, 2)), format!(r"$\pm{}$", num(cv, 2))], small, SIGNIFICANT, true);
        let real = s.real > 0.0;
        if real {
            plot.vline(f, MARKS, s.effect, 2.0, REAL, false);
        }
        let texts = if real { vec![format!("true effect {}", num(s.effect, 2)), format!("true {}", num(s.effect, 2))] } else { vec![String::new()] };
        line_label(f, &plot, plot.px_x(s.effect), 0.0, &texts, small, REAL, true);
        let mean = look.marks.get(2) as f64;
        let average = c.sig_mean().filter(|_| real);
        if average.is_some() {
            plot.vline(f, MARKS, mean, 2.0, ESTIMATE, false);
        }
        let texts = average.map_or(vec![String::new()], |m| vec![format!("the significant ones’ average {}", num(m, 2)), format!("their average {}", num(m, 2)), num(m, 2)]);
        line_label(f, &plot, plot.px_x(mean), 1.0, &texts, small, ESTIMATE, false);
        plot.x_axis(f, "estimated effect: the mean of the $n$ values", small);
        self.grid(f, 1, &plot, "share of studies", small);

        self.draw_study_numbers(f, lay);
    }

    /// What the studies so far come to, under the charts, and a key beside them.
    fn draw_study_numbers(&mut self, f: &mut Frame, lay: &Layout) {
        let (s, c) = (self.s, &self.counts);
        let (min, max) = lay.below;
        let size = lay.text;
        let width = if lay.narrow { max.x - min.x } else { (max.x - min.x) * 0.62 };
        let n = count(c.studies() as usize);
        let real = count(c.total[1] as usize);
        let first = match (c.total[1], c.total[0]) {
            (_, 0) if c.studies() > 0 => vec![format!("{n} studies, every one of a real effect")],
            (0, _) => vec![format!("{n} studies, all of nothing")],
            _ => vec![format!("{n} studies, {real} of them of a real effect"), format!("{n} studies, {real} real")],
        };
        let sig = count((c.sig[0] + c.sig[1]) as usize);
        let (sr, sn) = (count(c.sig[1] as usize), count(c.sig[0] as usize));
        let a = short(s.alpha);
        // (what there is to say depends on whether there are true nulls, real effects, or both)
        let (nulls, reals) = (s.real < 1.0, s.real > 0.0);
        let second = match (nulls, reals) {
            (true, true) => vec![format!(r"significant ($p \le {a}$): {sig}, of which {sr} real effects and {sn} true nulls"), format!(r"significant ($p \le {a}$): {sig}: {sr} real, {sn} null")],
            (true, false) => vec![format!(r"significant ($p \le {a}$): {sig}, every one a false alarm")],
            _ => vec![format!(r"significant ($p \le {a}$): {sig}, every one a real effect")],
        };
        let third = match (nulls, reals, c.real_share()) {
            (true, true, Some(share)) => format!("of the significant, real: {}", pct(share)),
            (true, true, None) => "none of them significant yet".to_string(),
            (true, false, _) => "there are no real effects to find".to_string(),
            _ => "there are no true nulls to mistake".to_string(),
        };
        let rate = |k: usize| if c.total[k] > 0 { pct(c.sig[k] as f64 / c.total[k] as f64) } else { "none yet".to_string() };
        let (r0, r1) = (rate(0), rate(1));
        let pw = num(power(s.effect, s.n, s.alpha), 2);
        let fourth = match (nulls, reals) {
            (true, true) => vec![format!("called significant: {r0} of the true nulls (false alarms), {r1} of the real effects (power {pw} by formula)"), format!("significant: {r0} of true nulls, {r1} of real effects")],
            (true, false) => vec![format!(r"called significant: {r0} of them ($\alpha = {a}$)")],
            _ => vec![format!("called significant: {r1} of them (power {pw} by formula)")],
        };
        let fifth = match c.sig_mean() {
            Some(m) if s.real > 0.0 => {
                let (m, t) = (num(m, 2), num(s.effect, 2));
                vec![format!("the significant real effects’ average estimate: {m}, the true effect {t}"), format!("significant real ones’ average: {m} (true {t})")]
            }
            _ => vec![String::new()],
        };
        let fit = |f: &Frame, options: &[String]| options.iter().find(|t| f.text_size(t, size).x <= width).or(options.last()).cloned().unwrap_or_default();
        let rows = [(fit(f, &first), pal::TEXT), (fit(f, &second), SIGNIFICANT), (third, REAL), (fit(f, &fourth), pal::DIM), (fit(f, &fifth), ESTIMATE)];
        lines(f, (min, max), &rows, size);
        if !lay.narrow {
            let key = [
                (NULL, "a true null: nothing is going on"),
                (REAL, "a real effect, and its true size"),
                (SIGNIFICANT, r"significant: $p \le \alpha$, beyond the critical values"),
                (ESTIMATE, "the significant real effects’ average estimate"),
            ];
            let rows = (((max.y - min.y) / (lay.small + 6.0)) as usize).min(key.len());
            hud_key(f, Vec2::new(min.x + width + 24.0, min.y + 2.0), &key[..rows], lay.small);
        }
    }

    fn draw_family(&mut self, f: &mut Frame, lay: &Layout) {
        let s = self.s;
        let (big, small) = (lay.big, lay.small);
        let [gr, pr] = lay.charts;
        let m = s.m;
        let real = s.real_tests();
        let sorted = self.shown_sorted();
        let ps: Vec<f64> = sorted.iter().map(|(p, _)| *p).collect();
        let threshold = self.rule.threshold(&ps, s.alpha, s.q);

        // the family, cell by cell, with its key under it
        hud_panel(f, gr.0 - Vec2::new(40.0, 44.0), gr.1 + Vec2::new(10.0, 34.0));
        let title = match real {
            0 => [format!("the family: {m} tests, none of them a real effect"), format!("{m} tests, none real")],
            r if r == m => [format!("the family: {m} tests, every one a real effect"), format!("{m} tests, all real")],
            1 => [format!("the family: {m} tests, the first a real effect"), format!("{m} tests, the first real")],
            r => [format!("the family: {m} tests, the first {r} real effects"), format!("{m} tests, the first {r} real")],
        };
        let title = if f.text_size(&title[0], big).x <= gr.1.x - gr.0.x + 30.0 { title[0].clone() } else { title[1].clone() };
        f.text_px(Vec2::new(gr.0.x - 30.0, gr.0.y - 24.0), title, big, pal::TEXT, Align2::LEFT_BOTTOM);
        let row = small + 7.0;
        let entries = [
            (REAL, "real effect, declared"),
            (SIGNIFICANT, "true null, declared: a false alarm"),
            (REAL.a(0.3), "real effect, missed"),
            (NULL.a(0.35), "true null, not declared"),
        ];
        // (two entries to a row where they fit, else one)
        let half = (gr.1.x - gr.0.x) * 0.5;
        let across = if entries.iter().all(|(_, t)| f.text_size(t, small).x + small + 12.0 <= half) { 2 } else { 1 };
        let key_h = (1 + entries.len().div_ceil(across)) as f32 * row + 6.0;
        let (w, h) = (gr.1.x - gr.0.x, (gr.1.y - gr.0.y - key_h).max(4.0));
        let cols = ((m as f32 * w / h).sqrt().ceil() as usize).max(1).min(m);
        let rows = m.div_ceil(cols);
        let cell = (w / cols as f32).min(h / rows as f32).min(84.0);
        let corner = gr.0 + Vec2::new((w - cell * cols as f32) * 0.5, (h - cell * rows as f32) * 0.5);
        let gap = if cell > 10.0 { 2.0 } else { 1.0 };
        for i in 0..m {
            let at = corner + Vec2::new((i % cols) as f32 * cell, (i / cols) as f32 * cell);
            let (lo, hi) = (at, at + Vec2::splat(cell - gap));
            let is_real = i < real;
            hud_rect(f, lo, hi, if is_real { REAL.a(0.24) } else { NULL.a(0.28) });
            let lit = self.look.cells.get(i);
            if lit > 0.004 {
                hud_rect_at(f, 2, lo, hi, if is_real { REAL } else { SIGNIFICANT }.a(0.92 * lit.min(1.0)));
            }
        }
        let (d, v) = if self.shown.is_empty() {
            (0, 0)
        } else {
            let vd = verdict(&self.shown, &ps, real, self.rule, s.alpha, s.q);
            (vd.declared, vd.false_alarms)
        };
        let key_top = gr.1.y - key_h + 8.0;
        let words = if self.shown.is_empty() {
            "the first family is on its way".to_string()
        } else {
            let alarms = if v == 1 { "1 false alarm".to_string() } else { format!("{v} false alarms") };
            format!("lit by {}: {d} declared, {alarms}", self.rule.name())
        };
        f.text_px(Vec2::new(gr.0.x, key_top), words, small, pal::TEXT, Align2::LEFT_TOP);
        for (k, (color, text)) in entries.iter().enumerate() {
            let at = Vec2::new(gr.0.x + (k % across) as f32 * half, key_top + row * (1 + k / across) as f32);
            hud_rect(f, at + Vec2::new(0.0, 2.0), at + Vec2::new(small - 1.0, small + 1.0), *color);
            f.text_px(at + Vec2::new(small + 5.0, small * 0.5 + 1.5), *text, small, pal::DIM, Align2::LEFT_CENTER);
        }

        // its p-values, smallest first
        let floor = self.p_floor();
        let top = m as f64 + 0.5;
        let plot = Plot::new(pr, [0.5, top], [floor, 0.0]);
        plot.backing(f, 40.0, 34.0, 44.0, 10.0);
        plot.title(f, "its $p$-values, smallest first", big);
        log_y_axis(&plot, f, "$p$-value", small);
        plot.x_axis(f, "rank", small);
        let lit = |r: Rule| if r == self.rule { (2.0, pal::WHITE) } else { (1.2, pal::DIM.a(0.8)) };
        let (wa, ca) = lit(Rule::Plain);
        plot.hline(f, MARKS, s.alpha.log10(), wa, ca, false);
        let (wb, cb) = lit(Rule::Bonferroni);
        let bonferroni = s.alpha / m as f64;
        plot.hline(f, MARKS, bonferroni.log10(), wb, cb, false);
        let (wh, ch) = lit(Rule::Bh);
        plot.curve(f, CURVES, 64, wh, ch, |k| (k.max(1e-9) * s.q / m as f64).log10());
        let cut = self.look.cut.get(0) as f64;
        plot.vline(f, MARKS, cut, 1.0, ch.a(0.6), true);
        let end = |y: f64| Vec2::new(pr.1.x - 4.0, plot.px_y(y));
        f.caption_px(end(s.alpha.log10()) + Vec2::new(0.0, 4.0), format!(r"$\alpha = {}$: no correction", short(s.alpha)), small, ca, Align2::RIGHT_TOP);
        f.caption_px(end(bonferroni.log10()) + Vec2::new(0.0, 4.0), format!(r"$\alpha/m = {}$: Bonferroni", short(bonferroni)), small, cb, Align2::RIGHT_TOP);
        f.caption_px(end((top * s.q / m as f64).log10()) - Vec2::new(0.0, 4.0), format!(r"$kq/m$, $q = {}$: Benjamini–Hochberg", short(s.q)), small, ch, Align2::RIGHT_BOTTOM);
        let k_cut = bh_count(&ps, s.q);
        let cut_text = if self.shown.is_empty() {
            String::new()
        } else if k_cut == 0 {
            "no point under the line".to_string()
        } else {
            format!("the last point under the line: rank {k_cut}")
        };
        let cut_x = plot.px_x(cut.max(0.5)).min(pr.1.x - f.text_size(&cut_text, small).x - 8.0).max(pr.0.x + 4.0);
        plate(f, Vec2::new(cut_x + 4.0, pr.1.y - 4.0), cut_text, small, ch, Align2::LEFT_BOTTOM);
        let dot = if m <= 40 { 8.0 } else if m <= 100 { 6.0 } else { 4.5 };
        for (k, (p, test)) in sorted.iter().enumerate() {
            let is_real = *test < real;
            let color = match (*p <= threshold, is_real) {
                (true, true) => REAL,
                (true, false) => SIGNIFICANT,
                (false, true) => REAL.a(0.5),
                (false, false) => NULL.a(0.85),
            };
            plot.dot(f, (k + 1) as f64, self.look.dots.get(k) as f64, dot, color);
        }

        self.draw_family_table(f, lay);
    }

    /// The three rules over all the families so far, each row a button that lights
    /// its verdicts.
    fn draw_family_table(&mut self, f: &mut Frame, lay: &Layout) {
        let s = self.s;
        let fam = &self.families;
        let (min, max) = lay.below;
        let size = if lay.narrow { lay.small } else { lay.text };
        let done = fam.done.max(1) as f64;
        let real = s.real_tests();
        let row_h = size + 11.0;
        let label_w = if lay.narrow { 104.0 } else { 190.0 };
        let col_w = ((max.x - min.x - label_w - 8.0) / 4.0).max(10.0);
        let heads = [["declared, on average", "declared", "declared"], ["families with a false alarm", "any false alarm", "any false"], ["false share of the declared", "false share", "false share"], ["real effects found (power)", "power", "power"]];
        // (one style for all four headings: the first in which they all fit)
        let style = (0..3).find(|&k| heads.iter().all(|h| f.text_size(h[k], size).x <= col_w - 8.0)).unwrap_or(2);
        f.text_px(min, format!("over {} families of tests:", count(fam.done as usize)), size, pal::TEXT, Align2::LEFT_TOP);
        let head_y = min.y + row_h;
        for (j, h) in heads.iter().enumerate() {
            let x = min.x + label_w + 8.0 + (j as f32 + 1.0) * col_w;
            f.text_px(Vec2::new(x - 4.0, head_y), h[style], size, pal::DIM, Align2::RIGHT_TOP);
        }
        for (r, rule) in Rule::ALL.iter().enumerate() {
            let y = head_y + row_h * (r as f32 + 1.0);
            if y + row_h - 6.0 > max.y + 2.0 {
                break;
            }
            let rect = (Vec2::new(min.x, y - 4.0), Vec2::new(min.x + label_w, y + row_h - 7.0));
            let names: &[&str] = match rule {
                Rule::Plain => &["no correction", "none"],
                Rule::Bonferroni => &["Bonferroni"],
                Rule::Bh => &["Benjamini–Hochberg", "BH"],
            };
            let text = pick(f, names, label_w - 12.0, size);
            if plot::button(f, rect, text, size, self.rule == *rule) {
                self.rule = *rule;
            }
            let t = fam.tally[r];
            let cells = [
                num(t.declared as f64 / done, 1),
                pct(t.any_false as f64 / done),
                pct(t.false_share / done),
                if real == 0 { "none real".to_string() } else { pct(t.found as f64 / (done * real as f64)) },
            ];
            let color = if fam.done == 0 { pal::FAINT } else { pal::TEXT };
            for (j, text) in cells.iter().enumerate() {
                let x = min.x + label_w + 8.0 + (j as f32 + 1.0) * col_w;
                f.text_px(Vec2::new(x - 4.0, y), text.as_str(), size, color, Align2::RIGHT_TOP);
            }
        }
    }

    fn draw_peeking(&mut self, f: &mut Frame, lay: &Layout) {
        let s = self.s;
        let key = self.peeks.key;
        let (big, size, small) = (lay.big, lay.text, lay.small);
        let [pr, sr] = lay.charts;
        let (a, b) = ((key.first as f64).ln(), (key.last as f64).ln());

        // one study's p-value after each new value, on a scale even in z
        let floor = -PEEK_DEPTH;
        let y_of = |z: f64| -z.abs().min(PEEK_DEPTH);
        let y_alpha = -critical(s.alpha);
        let plot = Plot::new(pr, [a, b], [floor, 0.0]);
        plot.backing(f, 40.0, 34.0, 44.0, 10.0);
        plot.title(f, "the $p$-value after each new value", big);
        plot.bar(f, plot.x[0], plot.x[1], floor, y_alpha, 0.0, faint_significant());
        plot.hline(f, MARKS, y_alpha, 1.4, SIGNIFICANT, false);
        // (every slot's batch is there in every frame, in the same order, even while it
        // holds no path or a path of one point: a batch that came and went would leave
        // the ones after it to be taken for one another from frame to frame)
        for slot in 0..SLOTS {
            plot::line_of(f, OWN + 1 + slot as u8, pr.0, pr.0, 1.0, Color::CLEAR);
        }
        // (and the dips' batch likewise)
        dip_dot(f, pr.0, 0.0, Color::CLEAR);
        let peeks = &self.peeks;
        let at = |i: usize| ((key.first + i) as f64).ln();
        let (mut tip, mut stop) = (None, None);
        for slot in 0..SLOTS {
            let Some(path) = peeks.paths.get(slot) else { continue };
            let kind = OWN + 1 + slot as u8;
            let newest = peeks.newest == Some(slot);
            // the looks drawn: about one to each pixel and a half across (far to the right
            // hundreds of looks share a few pixels, and a line through each of them would
            // be a smear of zigzags inside single columns), the first and the last, and
            // the one where the path first dips below α
            let mut keep: Vec<usize> = Vec::with_capacity(path.z.len().min(1000));
            let mut last_x = f32::NEG_INFINITY;
            for i in 0..path.z.len() {
                let x = plot.px_x(at(i));
                if x - last_x >= 1.5 || Some(i) == path.dip || i + 1 == path.z.len() {
                    keep.push(i);
                    last_x = x;
                }
            }
            let all: Vec<(f64, f64)> = keep.iter().map(|&i| (at(i), y_of(path.z[i] as f64))).collect();
            let dip_at = path.dip.and_then(|d| keep.iter().position(|&i| i == d));
            // the newest gains its values as a pen moving at an even speed along it on
            // the screen reaches them, in the time it grows for: each value whole, as it
            // comes (evenly in n, or in log n, it would race up and down the steep
            // stretches and crawl along the flat)
            let pts: Vec<(f64, f64)> = if newest && peeks.grow > 0.0 && all.len() >= 2 {
                let screen: Vec<Vec2> = all.iter().map(|&(x, y)| plot.at(x, y)).collect();
                let lengths: Vec<f32> = screen.windows(2).map(|w| (w[1] - w[0]).length()).collect();
                let total: f32 = lengths.iter().sum();
                let mut pen = (peeks.age / peeks.grow).min(1.0) * total;
                let mut reached = 1;
                for &l in &lengths {
                    if pen < l {
                        break;
                    }
                    pen -= l;
                    reached += 1;
                }
                // (at the end, every value: what is left of the pen's way is rounding)
                if peeks.age >= peeks.grow {
                    reached = all.len();
                }
                all[..reached.min(all.len())].to_vec()
            } else {
                all
            };
            let split = dip_at.map_or(pts.len(), |d| (d + 1).min(pts.len()));
            let before = PATH.a(if newest { 1.0 } else { 0.32 });
            let after = pal::DIM.a(if newest { 0.8 } else { 0.25 });
            let width = if newest { 2.0 } else { 1.2 };
            plot.path(f, kind, pts[..split].iter().copied(), width, before);
            if split < pts.len() {
                plot.path(f, kind, pts[split.saturating_sub(1)..].iter().copied(), width * 0.8, after);
            }
            // (the dip's dot once the pen has reached it)
            if let (Some(d), Some(at_d)) = (path.dip, dip_at.filter(|a| *a < pts.len())) {
                let (x, y) = pts[at_d];
                dip_dot(f, plot.at(x, y), if newest { 9.0 } else { 5.0 }, SIGNIFICANT.a(if newest { 1.0 } else { 0.55 }));
                if newest {
                    stop = Some((x, key.first + d));
                }
            }
            if newest {
                tip = pts.last().copied().zip(keep.get(pts.len().saturating_sub(1)).map(|i| key.first + i));
            }
        }
        // (the newest study told in words at fixed places: a label riding its tip would
        // jerk with every value)
        let stop_text = stop.map_or(String::new(), |(_, n)| format!("the newest study first dips below after {n} values: a peeker stops there"));
        let stop_text = if f.text_size(&stop_text, small).x > pr.1.x - pr.0.x - 12.0 { stop.map_or(String::new(), |(_, n)| format!("a peeker stops after {n} values")) } else { stop_text };
        plate(f, Vec2::new(pr.0.x + 6.0, plot.px_y(floor) - 6.0), stop_text, small, SIGNIFICANT, Align2::LEFT_BOTTOM);
        plate(f, Vec2::new(pr.1.x - 4.0, plot.px_y(y_alpha) - 4.0), format!(r"$\alpha = {}$", short(s.alpha)), small, SIGNIFICANT, Align2::RIGHT_BOTTOM);
        p_axis(&plot, f, "$p$-value", small);
        log_x_axis(&plot, f, "values so far: each a look, and a test", small);

        // the share of studies that have been below α at some look so far
        let looks = share_looks(&key);
        let top = (self.look.share.get(looks.len()) as f64 / 100.0).max(0.05);
        let plot = Plot::new(sr, [a, b], [0.0, top]);
        plot.backing(f, 40.0, 34.0, 44.0, 10.0);
        plot.title(f, r"share of the studies below $\alpha$ at some look so far", big);
                plot.hline(f, MARKS, s.alpha, 1.4, SIGNIFICANT.a(0.7), true);
        f.caption_px(Vec2::new(sr.1.x - 4.0, plot.px_y(s.alpha) + 4.0), format!("one look alone: {}", pct(s.alpha)), small, SIGNIFICANT.a(0.9), Align2::RIGHT_TOP);
        // (held to the chart rather than cut at its top: early in a run the share can stand
        // above the top for a moment, and a curve cut short has fewer points)
        let ceiling = plot.y[1];
        plot.path(f, CURVES, looks.iter().enumerate().map(|(i, &n)| ((n as f64).ln(), (self.look.share.get(i) as f64 / 100.0).min(ceiling))), 2.2, SIGNIFICANT);
        for (j, n) in MARK_LOOKS.iter().enumerate() {
            let inside = *n >= key.first && *n <= key.last;
            let x = (*n as f64).ln();
            let y = self.look.share.get(looks.len() + 1 + j) as f64 / 100.0;
            let text = if inside { format!("by {n}: {}", pct(self.peeks.share_by(*n))) } else { String::new() };
            if inside {
                plot.dot(f, x, y, 6.0, SIGNIFICANT);
            }
            let at = plot.at(x, y) - Vec2::new(0.0, 9.0);
            let w = f.text_size(&text, small).x;
            let at = Vec2::new(at.x.min(sr.1.x - w * 0.5 - 2.0).max(sr.0.x + w * 0.5 + 2.0), at.y);
            plate(f, at, text, small, pal::TEXT, Align2::CENTER_BOTTOM);
        }
        self.grid(f, 2, &plot, "share", small);
        log_x_axis(&plot, f, "values so far", small);

        // the numbers, and a key beside them
        let (min, max) = lay.below;
        let width = if lay.narrow { max.x - min.x } else { (max.x - min.x) * 0.62 };
        let p = &self.peeks;
        let done = count(p.done as usize);
        let first = [
            format!("{done} studies of pure noise, each looked at after every value from the {} to the {}", ordinal(key.first), ordinal(key.last)),
            format!("{done} studies of noise, looks {} to {}", key.first, key.last),
        ];
        let dipped = p.share_by(key.last);
        let second = [format!(r"below $\alpha$ at some look: {} of them; at any one look: {}", pct(dipped), pct(s.alpha)), format!(r"below $\alpha$ at some look: {}", pct(dipped))];
        let back = p.back_above().map_or(String::new(), |b| format!(r"of those, above $\alpha$ again at the last look: {}", pct(b)));
        let now = tip.map_or(String::new(), |((_, y), n)| format!("the newest study: $p = {}$ after {n} values", p_text(p_of(y))));
        let fit = |f: &Frame, options: &[String]| options.iter().find(|t| f.text_size(t, size).x <= width).or(options.last()).cloned().unwrap_or_default();
        lines(f, (min, max), &[(fit(f, &first), pal::TEXT), (fit(f, &second), SIGNIFICANT), (back, pal::DIM), (now, PATH)], size);
        if !lay.narrow {
            let entries = [(PATH, "one study’s $p$-value"), (SIGNIFICANT, r"below $\alpha$: the dot is where a peeker stops"), (pal::DIM, "how the study would have gone on")];
            let rows = (((max.y - min.y) / (small + 6.0)) as usize).min(entries.len());
            hud_key(f, Vec2::new(min.x + width + 24.0, min.y + 2.0), &entries[..rows], small);
        }
    }
}

/// 1st, 2nd, 3rd, 4th, ..., 11th, 12th, 13th, 21st.
fn ordinal(n: usize) -> String {
    let suffix = match (n % 10, n % 100) {
        (_, 11..=13) => "th",
        (1, _) => "st",
        (2, _) => "nd",
        (3, _) => "rd",
        _ => "th",
    };
    format!("{n}{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What each experiment's seeded run gives, as its caption says it (measured by `explore`).
    const SHOWN_WORTH: f64 = 0.6391;
    const SHOWN_WORTH_LOW: f64 = 0.3090;
    const SHOWN_CURSE: f64 = 0.6121;
    /// Uncorrected and Bonferroni, families with a false alarm; Bonferroni's and
    /// Benjamini–Hochberg's power; Benjamini–Hochberg's false share.
    const SHOWN_CORRECTING: [f64; 5] = [0.5440, 0.0382, 0.2962, 0.5448, 0.0761];
    /// By the 100th look and by the 500th; back above at the last.
    const SHOWN_PEEKING: [f64; 3] = [0.2775, 0.4040, 0.8892];

    /// An experiment run to its end, as its button sets it up (with another seed, if one
    /// is given).
    fn run_with(k: usize, seed: Option<u64>) -> PValues {
        let mut s = PValues::create();
        s.set_up(k);
        if let Some(seed) = seed {
            s.seeds[s.tab.index()] = seed;
        }
        s.settle();
        s
    }

    fn run(k: usize) -> PValues {
        run_with(k, None)
    }

    /// The share of families in which a rule had a false alarm, the share of false
    /// alarms among what it declared (on average), and its power.
    fn rates(s: &PValues, rule: Rule) -> (f64, f64, f64) {
        let fam = &s.families;
        let t = fam.tally[Rule::ALL.iter().position(|r| *r == rule).unwrap_or(0)];
        let done = fam.done as f64;
        (t.any_false as f64 / done, t.false_share / done, t.found as f64 / (done * s.s.real_tests().max(1) as f64))
    }

    /// Benjamini and Hochberg's rule as defined, on lists worked out by hand.
    #[test]
    fn benjamini_hochberg_by_hand() {
        // m = 5, q = 0.1: the line is 0.02, 0.04, 0.06, 0.08, 0.10. The second p-value
        // (0.05) is above it, the fourth (0.07) the last under it: four are declared,
        // the second among them
        assert_eq!(bh_count(&[0.01, 0.05, 0.055, 0.07, 0.5], 0.1), 4);
        // ten at q = 0.05: the line rises by 0.005 a rank, and only the first two are under it
        assert_eq!(bh_count(&[0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.216], 0.05), 2);
        // none under the line, all of them, and none at all
        assert_eq!(bh_count(&[0.2, 0.3, 0.9], 0.1), 0);
        assert_eq!(bh_count(&[0.001, 0.002, 0.003], 0.1), 3);
        assert_eq!(bh_count(&[], 0.1), 0);
        // what each rule declares of the first list, unsorted, its first two real effects
        let p = [0.055, 0.5, 0.01, 0.07, 0.05];
        let mut sorted = p.to_vec();
        sorted.sort_by(f64::total_cmp);
        // BH: all but 0.5, so one real effect found, three false alarms, one miss
        assert_eq!(verdict(&p, &sorted, 2, Rule::Bh, 0.05, 0.1), Verdict { declared: 4, false_alarms: 3, missed: 1 });
        // at α = 0.05: 0.01 and 0.05, both true nulls
        assert_eq!(verdict(&p, &sorted, 2, Rule::Plain, 0.05, 0.1), Verdict { declared: 2, false_alarms: 2, missed: 2 });
        // at α/m = 0.01: 0.01 alone
        assert_eq!(verdict(&p, &sorted, 2, Rule::Bonferroni, 0.05, 0.1), Verdict { declared: 1, false_alarms: 1, missed: 2 });
        // one test: Bonferroni is the plain test, and Benjamini–Hochberg the plain test at q
        assert_eq!(Rule::Bonferroni.threshold(&[0.03], 0.05, 0.1), 0.05);
        assert_eq!((bh_count(&[0.07], 0.05), bh_count(&[0.07], 0.1)), (0, 1));
    }

    /// The test itself: p-values and critical values from the normal distribution, and
    /// the power as a formula agrees with counting.
    #[test]
    fn the_test_is_the_two_sided_z_test() {
        assert!((critical(0.05) - 1.959_963_984_540_054).abs() < 1e-9);
        assert!((p_of(1.959_963_984_540_054) - 0.05).abs() < 1e-9 && (p_of(-1.959_963_984_540_054) - 0.05).abs() < 1e-9);
        assert_eq!(p_of(0.0), 1.0);
        // the critical value the Words give as an example
        assert_eq!(num(critical(0.05) / 16f64.sqrt(), 2), "0.49");
        assert!(power(0.0, 25, 0.05) - 0.05 < 1e-12);
        // counted: a real effect of 0.5 with n = 25
        let mut rng = Rng::new(9);
        let hits = (0..100_000).filter(|_| p_of(2.5 + st::normal(&mut rng)) <= 0.05).count() as f64 / 100_000.0;
        let pw = power(0.5, 25, 0.05);
        assert!((pw - 0.705).abs() < 0.001 && (hits - pw).abs() < 0.005, "power {pw}, counted {hits}");
    }

    /// "Nothing going on": the p-values of studies of nothing are level.
    #[test]
    fn nothing_going_on_fills_the_bars_evenly() {
        let s = run(0);
        let c = &s.counts;
        assert_eq!((c.studies(), c.total[1]), (10_000, 0));
        // a chi-square test of the twenty bars against level
        let (chi, nulls) = c.level_check();
        let p = 1.0 - st::Dist::ChiSquare { df: 19.0 }.cdf(chi);
        assert!(nulls == 10_000 && p > 0.05, "chi-square {chi} on 19 degrees of freedom: p = {p}");
        // "here from 4.3 % to 5.4 %"
        let shares: Vec<f64> = c.p_bins[0].iter().map(|&k| k as f64 / 10_000.0).collect();
        let (low, high) = (shares.iter().copied().fold(1.0, f64::min), shares.iter().copied().fold(0.0, f64::max));
        assert_eq!((pct(low), pct(high)), ("4.3\u{a0}%".to_string(), "5.4\u{a0}%".to_string()));
        // and 5 % of them are false alarms
        let alarms = c.sig[0] as f64 / 10_000.0;
        assert!((alarms - 0.05).abs() < 0.001, "{alarms}");
        // the same for other seeds: level by the chi-square test
        for seed in 2..8 {
            let c = run_with(0, Some(seed)).counts;
            let (chi, _) = c.level_check();
            let p = 1.0 - st::Dist::ChiSquare { df: 19.0 }.cdf(chi);
            assert!(p > 0.001, "seed {seed}: chi-square {chi}, p = {p}");
        }
    }

    /// "What significant is worth": the arithmetic, and the studies counted.
    #[test]
    fn what_significant_is_worth() {
        let t = TRIALS[1].setup;
        let pw = power(t.effect, t.n, t.alpha);
        let share = |pw: f64| 0.1 * pw / (0.1 * pw + 0.9 * 0.05);
        assert!((pw - 0.80).abs() < 0.001, "power {pw}");
        assert!((1000.0 * 0.1 * pw - 80.0).abs() < 0.5 && (1000.0 * 0.9 * t.alpha - 45.0).abs() < 1e-9);
        assert!((share(pw) - 0.64).abs() < 0.001, "{}", share(pw));
        let low = power(0.28, t.n, t.alpha);
        assert!((low - 0.20).abs() < 0.002 && (1000.0 * 0.1 * low - 20.0).abs() < 0.5 && (share(low) - 0.31).abs() < 0.002, "power {low}, share {}", share(low));
        // counted, as shown: SHOWN_WORTH of the significant are real
        let s = run(1);
        let real = s.counts.real_share().unwrap_or(0.0);
        assert!((real - SHOWN_WORTH).abs() < 0.0005 && pct(real) == "64\u{a0}%", "{real}");
        // lowering the effect to 0.28, as the caption asks
        let mut s = run(1);
        s.s.effect = 0.28;
        s.settle();
        let low = s.counts.real_share().unwrap_or(0.0);
        assert!((low - SHOWN_WORTH_LOW).abs() < 0.0005 && pct(low) == "31\u{a0}%", "{low}");
    }

    /// "The winner's curse": the significant estimates are far above the truth.
    #[test]
    fn the_winners_curse() {
        let t = TRIALS[2].setup;
        let pw = power(t.effect, t.n, t.alpha);
        assert!((pw - 0.17).abs() < 0.002, "power {pw}");
        let s = run(2);
        let c = &s.counts;
        let sig = c.sig[1] as f64 / c.total[1] as f64;
        let mean = c.sig_mean().unwrap_or(0.0);
        assert!((sig - 0.17).abs() < 0.008 && (mean - SHOWN_CURSE).abs() < 0.0005 && num(mean, 2) == "0.61" && mean > 2.0 * t.effect, "{sig} significant, average {mean}");
        // a few of the wrong sign, under 1 %
        assert!(c.sig_wrong > 0 && (c.sig_wrong as f64) < 0.01 * c.sig[1] as f64, "{} of {}", c.sig_wrong, c.sig[1]);
    }

    /// "Twenty tests of nothing": at least one false alarm in 64 % of the families.
    #[test]
    fn twenty_tests_of_nothing() {
        assert!((1.0 - 0.95f64.powi(20) - 0.64).abs() < 0.002);
        let s = run(3);
        let (any, _, _) = rates(&s, Rule::Plain);
        // (here 65 % of 4 000: 64 % is within the wobble of chance, about 0.8 % either way)
        assert!(s.families.done == 4000 && pct(any) == "65\u{a0}%" && (any - 0.64).abs() < 2.0 * (0.64f64 * 0.36 / 4000.0).sqrt(), "{any}");
        // each test alone keeps to 5 %
        let declared = s.families.tally[0].declared as f64 / (s.families.done as f64 * 20.0);
        assert!((declared - 0.05).abs() < 0.003, "{declared}");
    }

    /// "Correcting for twenty tests": the three rules, as the caption says.
    #[test]
    fn correcting_for_twenty_tests() {
        let t = TRIALS[4].setup;
        assert_eq!((t.m, t.real_tests()), (20, 5));
        assert_eq!(num(power(t.effect, t.n, t.alpha), 2), "0.71");
        assert!((t.alpha / t.m as f64 - 0.0025).abs() < 1e-12);
        let s = run(4);
        let (plain_any, _, plain_power) = rates(&s, Rule::Plain);
        let (bonf_any, _, bonf_power) = rates(&s, Rule::Bonferroni);
        let (_, bh_share, bh_power) = rates(&s, Rule::Bh);
        let shown = SHOWN_CORRECTING;
        for (name, got, want) in [("uncorrected, any false alarm", plain_any, shown[0]), ("Bonferroni, any false alarm", bonf_any, shown[1]), ("Bonferroni's power", bonf_power, shown[2]), ("Benjamini–Hochberg's power", bh_power, shown[3]), ("Benjamini–Hochberg's false share", bh_share, shown[4])] {
            assert!((got - want).abs() < 0.0005, "{name}: {got}, the caption {want}");
        }
        let said = [pct(plain_any), pct(bonf_any), pct(bonf_power), pct(bh_power), pct(bh_share)];
        assert_eq!(said, ["54", "3.8", "30", "54", "7.6"].map(|v| format!("{v}\u{a0}%")));
        assert!(bonf_any < 0.05 && bh_share < 0.1 && bh_power > bonf_power && plain_power > bh_power);
        // by formula: 15 true nulls at 0.05 and at 0.0025
        assert!((1.0 - 0.95f64.powi(15) - 0.537).abs() < 0.001 && (1.0 - 0.9975f64.powi(15) - 0.0368).abs() < 0.0005);
    }

    /// "Peeking": the share of studies of nothing that dip below 0.05 at some look.
    #[test]
    fn peeking() {
        let s = run(5);
        let p = &s.peeks;
        let (by100, by500, back) = (p.share_by(100), p.share_by(500), p.back_above().unwrap_or(0.0));
        let shown = SHOWN_PEEKING;
        assert!((by100 - shown[0]).abs() < 0.0005 && (by500 - shown[1]).abs() < 0.0005 && (back - shown[2]).abs() < 0.0005, "{by100} by 100, {by500} by 500, {back} back above");
        assert_eq!([pct(by100), pct(by500), pct(back)], ["28", "40", "89"].map(|v| format!("{v}\u{a0}%")));
        // the first look alone is one test: 5 %
        assert!((p.share_by(10) - 0.05).abs() < 0.01, "{}", p.share_by(10));
        // the share never falls as the looks go on
        assert!((10..500).all(|n| p.share_by(n) <= p.share_by(n + 1)));
    }

    /// Watching a study does not change what it counts, and its path is the same
    /// study: its dip is its first look below α.
    #[test]
    fn watching_a_study_does_not_change_the_count() {
        let key = PeekKey { first: 10, last: 200, alpha: 0.05, seed: 3 };
        for i in 0..60 {
            let (dip, below, _) = peek(key, i, false);
            let (dip2, below2, path) = peek(key, i, true);
            assert_eq!((dip, below), (dip2, below2));
            let path = path.expect("a path");
            assert_eq!((path.dip, path.z.len()), (dip, 191));
            let c = critical(0.05) as f32;
            let first = path.z.iter().position(|z| z.abs() >= c - 1e-5);
            assert!(first == dip || dip.is_some_and(|d| first.is_some_and(|f| f.abs_diff(d) <= 1)), "study {i}: {first:?} {dip:?}");
            if let Some(z) = path.z.last().filter(|z| (z.abs() - c).abs() > 1e-4) {
                assert_eq!(below, z.abs() >= c, "study {i}");
            }
        }
    }

    /// Changing the level keeps the studies and counts them again; changing it for a
    /// family starts the families again; a new seed gives new numbers.
    #[test]
    fn a_change_starts_the_counts_afresh() {
        let mut s = run(0);
        let before = s.studies.z.clone();
        s.s.alpha = 0.01;
        s.refresh();
        assert_eq!(s.studies.z, before);
        let by_hand = before.iter().filter(|z| p_of(**z as f64) <= 0.01).count() as u64;
        assert_eq!(s.counts.sig[0], by_hand);
        let mut s = run(3);
        s.s.alpha = 0.01;
        s.refresh();
        assert_eq!(s.families.done, 0);
        let mut s = run(0);
        s.seeds[0] += 1;
        s.settle();
        assert_ne!(s.studies.z, before);
    }

    /// Settings at their ends, and the cases with nothing to count: every number stays
    /// finite and says what has happened.
    #[test]
    fn degenerate_settings_are_told_honestly() {
        for (m, real) in [(1, 0.0), (1, 1.0), (200, 0.0), (200, 1.0), (20, 0.5)] {
            let mut s = PValues::create();
            s.tab = Tab::Family;
            (s.s.m, s.s.real, s.goals[1]) = (m, real, 50);
            s.settle();
            s.advance(1.0);
            s.follow(0.1, 1.3);
            assert!(s.look.dots.shown.iter().chain(&s.look.cells.shown).chain(&s.look.cut.shown).all(|v| v.is_finite()));
            for (r, t) in s.families.tally.iter().enumerate() {
                let share = t.false_share / s.families.done as f64;
                assert!((0.0..=1.0).contains(&share), "m {m}, real {real}, rule {r}: {share}");
                if real == 0.0 {
                    // every discovery a false alarm: the false share is 1 in every family
                    // that declares anything, and 0 (not 0/0) in one that declares nothing
                    assert_eq!(t.found, 0);
                    assert!((t.false_share - t.any_false as f64).abs() < 1e-9);
                }
                if real == 1.0 {
                    assert_eq!(t.any_false, 0);
                }
            }
        }
        // studies all real, none real, an effect of nothing, α at its ends
        for (real, effect, alpha) in [(1.0, 1.5, 0.2), (0.0, 0.5, 0.001), (0.5, 0.0, 0.05), (1.0, 0.0, 0.001)] {
            let mut s = PValues::create();
            (s.s.real, s.s.effect, s.s.alpha, s.goals[0]) = (real, effect, alpha, 500);
            s.settle();
            s.follow(0.1, 1.3);
            let c = &s.counts;
            assert!(c.est.len() >= 10 && c.est.len() <= 400 && c.w > 0.0);
            assert!(s.look.p_bars.shown.iter().chain(&s.look.e_bars.shown).chain(&s.look.marks.shown).all(|v| v.is_finite()));
            if real == 0.0 {
                assert!(c.sig_mean().is_none());
            }
        }
        // one look only: one test, and a share of about α
        let mut s = PValues::create();
        s.tab = Tab::Peeking;
        (s.s.first, s.s.last, s.goals[2]) = (50, 20, 2000);
        s.settle();
        assert_eq!((s.s.first, s.s.last), (50, 50));
        assert!((s.peeks.share_by(50) - 0.05).abs() < 0.015);
        s.follow(0.1, 1.3);
        assert!(s.look.share.shown.iter().all(|v| v.is_finite()));
        // and nothing at all in range outside it
        let mut bad = PValues::create();
        (bad.s.alpha, bad.s.q, bad.s.effect, bad.rates[0]) = (f64::NAN, -1.0, f64::INFINITY, f32::NAN);
        bad.refresh();
        assert!(bad.s.alpha == 0.05 && bad.s.q == 0.01 && bad.s.effect == 0.5 && bad.rates[0].is_finite());
    }

    /// Every window gets a layout with everything on it, the charts apart from the
    /// numbers under them; a window too small for charts is told so.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h) in [(1600.0, 1000.0), (1100.0, 700.0), (1340.0, 1000.0), (390.0, 800.0), (390.0, 560.0), (390.0, 300.0), (500.0, 400.0), (800.0, 390.0), (240.0, 180.0), (8.0, 8.0), (2400.0, 500.0)] {
            for lines in 1..=4 {
                let view = Vec2::new(w, h);
                let lay = Layout::of(view, 40.0, lines);
                let rects = [lay.charts[0], lay.charts[1], lay.below, lay.tabs[0], lay.tabs[2]];
                assert!(rects.iter().all(|r| r.0.is_finite() && r.1.is_finite()), "{view}");
                if !lay.compact {
                    for r in rects {
                        assert!(r.0.x >= 0.0 && r.0.y >= 0.0 && r.1.x <= w + 0.5 && r.1.y <= h + 0.5 && r.1.x >= r.0.x && r.1.y >= r.0.y, "{view}, {lines} lines: {r:?}");
                    }
                    // the x axis's numbers and name (34 pixels) above the numbers
                    assert!(lay.charts.iter().all(|c| c.1.y + 30.0 <= lay.below.0.y), "{view}: {lay:?}");
                    if lay.stacked {
                        assert!(lay.charts[0].1.y + 60.0 <= lay.charts[1].0.y, "{view}");
                    } else {
                        assert!(lay.charts[0].1.x + 60.0 <= lay.charts[1].0.x, "{view}");
                    }
                }
            }
        }
        // a desktop and a phone have their charts
        assert!(!Layout::of(Vec2::new(1100.0, 700.0), 0.0, 2).compact && !Layout::of(Vec2::new(390.0, 760.0), 44.0, 3).compact);
        // and so does a phone on its side
        let side = Layout::of(Vec2::new(800.0, 390.0), 44.0, 2);
        assert!(!side.compact && !side.stacked, "{side:?}");
    }

    /// Every experiment comes to rest: its run reaches its end, the picture glides to
    /// it and stops, and the scene stops asking for frames.
    #[test]
    fn every_experiment_comes_to_rest() {
        for k in 0..TRIALS.len() {
            let mut s = run(k);
            s.advance(1.0);
            let mut frames = 0;
            loop {
                s.follow(1.0 / 60.0, 1.3);
                if !s.look.moving {
                    break;
                }
                frames += 1;
                assert!(frames < 300, "experiment {k} still gliding");
            }
            s.advance(10.0);
            assert!(!s.animated(), "experiment {k}");
        }
    }

    /// The words the panel and the picture show are markup that sets.
    #[test]
    fn the_words_are_markup_that_sets() {
        let texts = TRIALS.iter().flat_map(|t| [t.button, t.look_for]).chain([INTRO, THE_TEST]);
        for text in texts {
            let found = crate::typeset::inspect(text);
            assert!(found.error.is_none() && found.words.is_empty(), "{text}: {found:?}");
        }
        assert_eq!(pct(0.0368), "3.7\u{a0}%");
        assert_eq!(pct(0.64), "64\u{a0}%");
        assert_eq!((short(0.05), short(0.0025), short(0.1), p_text(0.508), p_text(0.0034)), ("0.05".into(), "0.0025".into(), "0.1".into(), "0.51".into(), "0.0034".into()));
        assert_eq!((ordinal(1), ordinal(2), ordinal(13), ordinal(500), ordinal(22)), ("1st".into(), "2nd".into(), "13th".into(), "500th".into(), "22nd".into()));
    }

    /// The rates in the captions hold beyond the seed each experiment shows: with six
    /// other seeds, within sampling error.
    #[test]
    fn the_rates_hold_for_other_seeds() {
        for seed in 2..8 {
            let s = run_with(1, Some(seed));
            let real = s.counts.real_share().unwrap_or(0.0);
            assert!((real - 0.64).abs() < 0.04, "seed {seed}: {real}");
            let s = run_with(2, Some(seed));
            let mean = s.counts.sig_mean().unwrap_or(0.0);
            assert!((mean - 0.61).abs() < 0.03, "seed {seed}: {mean}");
            let s = run_with(3, Some(seed));
            let (any, _, _) = rates(&s, Rule::Plain);
            assert!((any - 0.64).abs() < 0.035, "seed {seed}: {any}");
            let s = run_with(4, Some(seed));
            let (bonf_any, _, bonf_power) = rates(&s, Rule::Bonferroni);
            let (_, bh_share, bh_power) = rates(&s, Rule::Bh);
            assert!(bonf_any < 0.05 && (bonf_power - 0.30).abs() < 0.02 && (bh_power - 0.54).abs() < 0.02 && (bh_share - 0.076).abs() < 0.012, "seed {seed}: {bonf_any} {bonf_power} {bh_power} {bh_share}");
            let s = run_with(5, Some(seed));
            let (by100, by500) = (s.peeks.share_by(100), s.peeks.share_by(500));
            assert!((by100 - 0.275).abs() < 0.025 && (by500 - 0.40).abs() < 0.025, "seed {seed}: {by100} {by500}");
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            println!("{}: {:?}", TRIALS[k].button, s.tab);
            let c = &s.counts;
            match s.tab {
                Tab::Studies => {
                    let counts: Vec<String> = (0..BINS).map(|j| c.p_bins[0][j].to_string()).collect();
                    println!("  true nulls' bars {}", counts.join(" "));
                    println!("  level check {:?}", c.level_check());
                    println!("  sig {:?} total {:?} real share {:?} sig mean {:?} wrong {}", c.sig, c.total, c.real_share(), c.sig_mean(), c.sig_wrong);
                    println!("  power {}", power(s.s.effect, s.s.n, s.s.alpha));
                }
                Tab::Family => {
                    for rule in Rule::ALL {
                        let (any, share, pw) = rates(&s, rule);
                        println!("  {rule:?}: any false {any:.4} false share {share:.4} power {pw:.4}");
                    }
                }
                Tab::Peeking => {
                    for n in [10, 20, 50, 100, 200, 500] {
                        println!("  by {n}: {:.4}", s.peeks.share_by(n));
                    }
                    println!("  back above at the last look: {:?}", s.peeks.back_above());
                }
            }
        }
        let mut s = run(1);
        s.s.effect = 0.28;
        s.settle();
        println!("worth, effect 0.28: {:?}", s.counts.real_share());
    }
}

//! Chance and conditional probability: a thousand people in a square, split by two
//! events. A probability is a share of them, the area of a part of the square; a
//! conditional probability is a share of one part of them; and Bayes' rule is counting.

use super::plot::{self, Glide, Rect};
use super::*;
use crate::math::stats::unit;

pub const DEF: SceneDef = SceneDef {
    id: "chance",
    title: "Chance and conditional probability",
    group: CHANCE,
    blurb: "A thousand people in a square: events as regions of it, conditional probability, independence, and Bayes' rule as counting.",
    make: |_| Box::new(Chance::create()),
};

// ----- the probabilities ----------------------------------------------------------------

/// How many people the square holds.
const PEOPLE: u32 = 1000;

/// The three numbers everything here follows from: the share of everyone in $A$, and
/// the share in $B$ on each side of the divider.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Probs {
    /// $P(A)$: in a test, the base rate.
    a: f64,
    /// $P(B \mid A)$: in a test, its sensitivity.
    b_a: f64,
    /// $P(B \mid \text{not } A)$: in a test, its false alarm rate.
    b_n: f64,
}

/// The four regions of the square, in the order their numbers are kept: in $A$ and in
/// $B$; in $A$ only; in $B$ only; in neither.
const BOTH: usize = 0;
const A_ONLY: usize = 1;
const B_ONLY: usize = 2;
const NEITHER: usize = 3;

impl Probs {
    /// Each kept between 0 and 1 (one that is not a number becomes 0).
    fn kept(self) -> Probs {
        let keep = |v: f64| if v.is_nan() { 0.0 } else { v.max(0.0).min(1.0) };
        Probs { a: keep(self.a), b_a: keep(self.b_a), b_n: keep(self.b_n) }
    }

    /// The probability of each region: its area.
    fn joint(&self) -> [f64; 4] {
        let (a, n) = (self.a, 1.0 - self.a);
        [a * self.b_a, a * (1.0 - self.b_a), n * self.b_n, n * (1.0 - self.b_n)]
    }

    /// $P(B)$: the two regions of $B$ together.
    fn b(&self) -> f64 {
        let j = self.joint();
        j[BOTH] + j[B_ONLY]
    }

    /// $P(A \text{ or } B)$: everything but the region in neither.
    fn either(&self) -> f64 {
        let j = self.joint();
        j[BOTH] + j[A_ONLY] + j[B_ONLY]
    }

    /// $P(B \mid A)$, if there are any $A$'s to look at.
    fn b_given_a(&self) -> Option<f64> {
        (self.a > 0.0).then_some(self.b_a)
    }

    /// $P(B \mid \text{not } A)$, if there is anyone outside $A$.
    fn b_given_not_a(&self) -> Option<f64> {
        (self.a < 1.0).then_some(self.b_n)
    }

    /// $P(A \mid B)$, the share of the $B$'s that are $A$'s (Bayes' rule), if there are
    /// any $B$'s.
    fn a_given_b(&self) -> Option<f64> {
        let b = self.b();
        (b > 0.0).then(|| (self.joint()[BOTH] / b).min(1.0))
    }

    /// Are $A$ and $B$ independent: is $P(A \text{ and } B) = P(A)\,P(B)$? The two differ
    /// by $P(A)\,P(\text{not } A)\,(P(B \mid A) - P(B \mid \text{not } A))$: so exactly when
    /// the two levels are equal, or when everyone or no one is in $A$.
    fn independent(&self) -> bool {
        self.a == 0.0 || self.a == 1.0 || self.b_a == self.b_n
    }

    /// The likelihood ratio, $P(B \mid A) / P(B \mid \text{not } A)$, when it is a number.
    fn ratio(&self) -> Option<f64> {
        (self.b_n > 0.0).then(|| self.b_a / self.b_n)
    }
}

/// Odds, $p : (1 - p)$ as one number, when they are finite.
fn odds(p: f64) -> Option<f64> {
    (p < 1.0).then(|| p / (1.0 - p))
}

/// A probability set by hand: kept between 0 and 1, to `steps` steps.
fn snap(v: f64, steps: f64) -> f64 {
    let v = if v.is_nan() { 0.0 } else { v.max(0.0).min(1.0) };
    (v * steps).round() / steps
}

// ----- the people ---------------------------------------------------------------------------

/// Whole people in each region, in the order of [`Probs::joint`]; they add up to 1 000.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counts([u32; 4]);

impl Counts {
    /// Each region's expected number of the 1 000, rounded so that they add up to it:
    /// the people in $A$ first (their share of 1 000, rounded), then the $B$'s among
    /// them and among the rest (their shares of those, rounded).
    fn expected(p: &Probs) -> Counts {
        let whole = |v: f64, most: u32| (v.round().max(0.0) as u32).min(most);
        let in_a = whole(PEOPLE as f64 * p.a, PEOPLE);
        let rest = PEOPLE - in_a;
        let ab = whole(in_a as f64 * p.b_a, in_a);
        let nb = whole(rest as f64 * p.b_n, rest);
        Counts([ab, in_a - ab, nb, rest - nb])
    }

    /// 1 000 people drawn at random: each in $A$ with probability $P(A)$, and in $B$
    /// with the probability of their side of the divider. The same seed gives the same
    /// people, so a probability changed a little moves only the few it must.
    fn drawn(p: &Probs, seed: u64) -> Counts {
        let mut rng = Rng::new(seed);
        let mut c = [0u32; 4];
        for _ in 0..PEOPLE {
            let in_a = unit(&mut rng) < p.a;
            let in_b = unit(&mut rng) < if in_a { p.b_a } else { p.b_n };
            c[region(in_a, in_b)] += 1;
        }
        Counts(c)
    }

    fn a(&self) -> u32 {
        self.0[BOTH] + self.0[A_ONLY]
    }

    fn not_a(&self) -> u32 {
        self.0[B_ONLY] + self.0[NEITHER]
    }

    fn b(&self) -> u32 {
        self.0[BOTH] + self.0[B_ONLY]
    }

    fn not_b(&self) -> u32 {
        self.0[A_ONLY] + self.0[NEITHER]
    }

    fn either(&self) -> u32 {
        self.0[BOTH] + self.0[A_ONLY] + self.0[B_ONLY]
    }

    fn total(&self) -> u32 {
        self.0.iter().sum()
    }
}

fn region(in_a: bool, in_b: bool) -> usize {
    match (in_a, in_b) {
        (true, true) => BOTH,
        (true, false) => A_ONLY,
        (false, true) => B_ONLY,
        (false, false) => NEITHER,
    }
}

/// Where the square is divided as drawn: by the probabilities themselves, or, for a
/// random draw, by the shares of the people drawn (a side with no one in it keeps the
/// probability's level, a line of no length).
fn split(truth: &Probs, counts: &Counts, drawn: bool) -> Probs {
    if !drawn {
        return *truth;
    }
    let share = |k: u32, of: u32, or: f64| if of > 0 { k as f64 / of as f64 } else { or };
    Probs { a: counts.a() as f64 / PEOPLE as f64, b_a: share(counts.0[BOTH], counts.a(), truth.b_a), b_n: share(counts.0[B_ONLY], counts.not_a(), truth.b_n) }
}

/// The four regions as rectangles of the square (0 to 1 each way, y up): x from, x to,
/// y from, y to.
fn rects(a: f32, b_a: f32, b_n: f32) -> [[f32; 4]; 4] {
    [[0.0, a, 0.0, b_a], [0.0, a, b_a, 1.0], [a, 1.0, 0.0, b_n], [a, 1.0, b_n, 1.0]]
}

/// The region of the square a point is in, for the divisions given.
fn region_at(x: f32, y: f32, a: f32, b_a: f32, b_n: f32) -> usize {
    match (x < a, y < if x < a { b_a } else { b_n }) {
        (true, true) => BOTH,
        (true, false) => A_ONLY,
        (false, true) => B_ONLY,
        (false, false) => NEITHER,
    }
}

/// How many places there are in the square for people to stand.
const SEATS: usize = 16_384;

/// The places people stand, fixed points of the square (0 to 1 each way, y up) spread
/// evenly over it however many are taken from the start of the list: the R2 sequence
/// (Roberts, 2018), steps of the inverse of the plastic number and of its square.
fn seats() -> &'static [[f32; 2]] {
    static SEATED: std::sync::OnceLock<Vec<[f32; 2]>> = std::sync::OnceLock::new();
    SEATED.get_or_init(|| {
        let g = 1.324_717_957_244_746_f64;
        let (s1, s2) = (1.0 / g, 1.0 / (g * g));
        (0..SEATS).map(|n| [(0.5 + s1 * n as f64).fract() as f32, (0.5 + s2 * n as f64).fract() as f32]).collect()
    })
}

/// Who stands where: of the places in each region, the first `counts` of them in the
/// list (so as evenly spread as the list is), for the divisions given. One more person
/// in a region takes its next place and moves no one; a division moved hands the
/// places it passes to the region on their side. Returns, for each place, its region
/// and whether anyone stands there; a region too small to hold its people among all
/// the places (none can be, at rest) stands the rest at its middle, in `extra`.
fn crowd(a: f32, b_a: f32, b_n: f32, counts: [u32; 4], taken: &mut Vec<(u8, bool)>) -> [u32; 4] {
    taken.clear();
    let mut seated = [0u32; 4];
    for &[x, y] in seats() {
        let k = region_at(x, y, a, b_a, b_n);
        let here = seated[k] < counts[k];
        seated[k] += u32::from(here);
        taken.push((k as u8, here));
    }
    [0, 1, 2, 3].map(|k| counts[k].saturating_sub(seated[k]))
}

// ----- the stories ------------------------------------------------------------------------

/// What $A$ and $B$ are called: the same numbers, told three ways.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Story {
    Test,
    Spam,
    Events,
}

const STORIES: [Story; 3] = [Story::Test, Story::Spam, Story::Events];

/// What a story calls things.
struct Words {
    button: &'static str,
    short: &'static str,
    people: &'static str,
    drawn: &'static str,
    what: &'static str,
    /// The two sides of the divider, and the two levels, as the table names them.
    a: &'static str,
    not_a: &'static str,
    b: &'static str,
    not_b: &'static str,
    regions: [&'static str; 4],
    key: [&'static str; 4],
    /// The two bars.
    of_a: &'static str,
    of_b: &'static str,
    no_a: &'static str,
    no_b: &'static str,
    all_a: &'static str,
    /// Bayes' rule in odds.
    odds_of: &'static str,
    learning: &'static str,
    retest: &'static str,
    retest_short: &'static str,
}

const WORDS: [Words; 3] = [
    Words {
        button: "a medical test",
        short: "test",
        people: "people",
        drawn: "people drawn at random",
        what: "who has a condition ($A$), and who tests positive ($B$)",
        a: "have it ($A$)",
        not_a: "do not",
        b: "positive ($B$)",
        not_b: "negative",
        regions: ["found", "missed", "false alarm", "all clear"],
        key: ["has it and tests positive: found", "has it, tests negative: missed", "does not have it, tests positive: a false alarm", "does not have it, tests negative: all clear"],
        of_a: "Of those who have it ($A$), the share who test positive",
        of_b: "Of those who test positive ($B$), the share who have it",
        no_a: "No one has it",
        no_b: "No one tests positive",
        all_a: "Everyone has it",
        odds_of: "The odds of having it",
        learning: "a positive test",
        retest: "test the positives again",
        retest_short: "test again",
    },
    Words {
        button: "a spam filter",
        short: "spam",
        people: "emails",
        drawn: "emails drawn at random",
        what: "which are spam ($A$), and which the filter flags ($B$)",
        a: "spam ($A$)",
        not_a: "not spam",
        b: "flagged ($B$)",
        not_b: "not flagged",
        regions: ["caught", "missed", "false alarm", "passed"],
        key: ["spam, flagged: caught", "spam, not flagged: missed", "not spam, flagged: a false alarm", "not spam, not flagged: passed"],
        of_a: "Of the spam ($A$), the share flagged",
        of_b: "Of the flagged emails ($B$), the share that are spam",
        no_a: "There is no spam",
        no_b: "Nothing is flagged",
        all_a: "Every email is spam",
        odds_of: "The odds that an email is spam",
        learning: "being flagged",
        retest: "filter the flagged emails again",
        retest_short: "filter again",
    },
    Words {
        button: "two events",
        short: "events",
        people: "equally likely outcomes",
        drawn: "outcomes drawn at random",
        what: "two events, $A$ and $B$, each true of some of them",
        a: "$A$",
        not_a: "not $A$",
        b: "$B$",
        not_b: "not $B$",
        regions: ["$A$ and $B$", "$A$ only", "$B$ only", "neither"],
        key: ["in $A$ and in $B$", "in $A$, not in $B$", "in $B$, not in $A$", "in neither"],
        of_a: "Of the outcomes in $A$, the share in $B$",
        of_b: "Of the outcomes in $B$, the share in $A$",
        no_a: "No outcome is in $A$",
        no_b: "No outcome is in $B$",
        all_a: "Every outcome is in $A$",
        odds_of: "The odds of $A$",
        learning: "learning that $B$ is true",
        retest: r"make $P(A \mid B)$ the new $P(A)$",
        retest_short: "update",
    },
];

/// Which group the picture looks at: a conditional probability is a share of one group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Focus {
    Everyone,
    /// Only the $A$'s: the probabilities given $A$.
    A,
    /// Only the $B$'s: the probabilities given $B$.
    B,
    /// Those in $A$ or $B$ or both.
    Either,
}

const FOCI: [Focus; 4] = [Focus::Everyone, Focus::A, Focus::B, Focus::Either];
const FOCUS_NAMES: [&str; 4] = ["everyone", "only $A$", "only $B$", "$A$ or $B$"];

// ----- numbers as words ---------------------------------------------------------------------

/// `v` to `decimals` places with its trailing zeros dropped.
fn trimmed(v: f64, decimals: usize) -> String {
    let s = num(v, decimals);
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

/// `v` with the digits it needs, starting from `decimals` places: more where fewer
/// would round a number between the ends onto an end (never 100 for what is not
/// certain, nor 0 for what can happen).
fn honest(v: f64, decimals: usize, end: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    let mut d = decimals;
    while d < 8 {
        let s = trimmed(v, d);
        let shown: f64 = s.replace('−', "-").parse().unwrap_or(v);
        if (shown == end && v < end) || (shown == 0.0 && v > 0.0) {
            d += 1;
        } else {
            return s;
        }
    }
    trimmed(v, d)
}

/// A probability as a percentage, with the digits it needs: 90, 9.2, 0.45.
fn pct(p: f64) -> String {
    let v = p * 100.0;
    let whole = (v - v.round()).abs() < 1e-7;
    let d = if whole {
        0
    } else if !(1.0..=99.0).contains(&v) {
        2
    } else {
        1
    };
    honest(v, d, 100.0)
}

/// A probability as a decimal, to three significant digits: 0.4, 0.12, 0.0917.
fn dec(p: f64) -> String {
    if p == 0.0 || p == 1.0 {
        return trimmed(p, 0);
    }
    let d = (2.0 - p.abs().log10().floor()).max(1.0).min(8.0) as usize;
    honest(p, d, 1.0)
}

/// A number to three significant digits (for odds and ratios), thousands set apart.
fn sig(v: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    if v >= 1000.0 {
        return count(v.round() as usize);
    }
    let d = if v == 0.0 { 0 } else { (2.0 - v.abs().log10().floor()).max(0.0).min(8.0) as usize };
    trimmed(v, d)
}

/// A whole number inside a formula, its thousands set apart: $1\,000$.
fn count_math(n: usize) -> String {
    count(n).replace('\u{2009}', r"\,")
}

/// Odds written as two numbers, one of them 1: $1 : 99$, $1.01 : 1$.
fn odds_text(o: f64) -> String {
    if o < 1.0 {
        format!("$1 : {}$", sig(1.0 / o))
    } else {
        format!("${} : 1$", sig(o))
    }
}

/// A probability that may be undefined, as the panel reads it out.
fn pct_or(p: Option<f64>) -> String {
    p.map_or("undefined".to_string(), |p| format!("{} %", pct(p)))
}

// ----- the experiments ----------------------------------------------------------------------

/// One of the "things to try".
struct Trial {
    button: &'static str,
    caption: &'static str,
    truth: Probs,
    story: Story,
    focus: Focus,
    /// Then test the positives again, once.
    again: bool,
}

/// A rare condition and a good test.
const RARE: Probs = Probs { a: 0.01, b_a: 0.9, b_n: 0.09 };

const TRIALS: [Trial; 5] = [
    Trial {
        button: "a rare condition",
        caption: r"1 % have the condition ($A$). The test finds 90 % of them, and wrongly flags 9 % of the rest. Of 1 000 people, 10 have it and 9 of those test positive; 990 do not, and 89 of those test positive too. So only 9 of the 98 positives have it: $P(A \mid B) = 9.2\%$, though $P(B \mid A) = 90\%$. Most positives are false alarms; the two bars are two different numbers.",
        truth: RARE,
        story: Story::Test,
        focus: Focus::B,
        again: false,
    },
    Trial {
        button: "a common condition",
        caption: r"The same test where 30 % have it: now 270 of the 333 positives have it, $P(A \mid B) = 81\%$. The test has not changed. What a positive result means depends on how common the condition was to begin with, the base rate.",
        truth: Probs { a: 0.3, b_a: 0.9, b_n: 0.09 },
        story: Story::Test,
        focus: Focus::B,
        again: false,
    },
    Trial {
        button: "test the positives again",
        caption: r"Take the positives of the rare condition and test them again, with a second test like the first whose mistakes are independent of the first's. Of them 9.2 % have it, so that is the base rate now, and after a second positive about 50 % do. In odds: $1 : 99$ before any test; each positive multiplies the odds by 10 ($0.9 / 0.09$, the likelihood ratio), to $10 : 99$ (or $1 : 9.9$) after one and $100 : 99$ (or $1.01 : 1$) after two.",
        truth: RARE,
        story: Story::Test,
        focus: Focus::B,
        again: true,
    },
    Trial {
        button: "independent",
        caption: r"The level of $B$ is the same on both sides: 30 % of the $A$'s and 30 % of the rest. Then $P(B \mid A) = P(B) = 30\%$, and learning that $A$ is true tells nothing about $B$. The areas multiply: $P(A \text{ and } B) = P(A)\,P(B) = 0.4 \times 0.3 = 0.12$, 120 of the 1 000. It holds the other way round too: $P(A \mid B) = P(A) = 40\%$. Drag one level off the other, and none of this is true any more.",
        truth: Probs { a: 0.4, b_a: 0.3, b_n: 0.3 },
        story: Story::Events,
        focus: Focus::A,
        again: false,
    },
    Trial {
        button: "either or both",
        caption: r"$A$ or $B$ (or both) is three regions: $300 + 200 + 100 = 600$ of the 1 000. Adding $P(A) = 0.5$ and $P(B) = 0.4$ gives 0.9, too much: the 300 in both are counted once in each. Take them away once: $P(A \text{ or } B) = 0.5 + 0.4 - 0.3 = 0.6$. Or take those in neither, 400, from everyone: $1 - 0.4 = 0.6$.",
        truth: Probs { a: 0.5, b_a: 0.6, b_n: 0.2 },
        story: Story::Events,
        focus: Focus::Either,
        again: false,
    },
];

/// The scene opens as the first experiment sets it up.
const OPENING: usize = 0;
/// The seed of the first random draw.
const FIRST_SEED: u64 = 13;

// ----- colours ------------------------------------------------------------------------------

/// The regions: blue for $A$ (bright where it is $B$ too), pink for $B$ without $A$ (in
/// a test, a false alarm), grey for neither.
const REGION: [Color; 4] = [pal::BLUE, Color::hex(0x4A64A0), ALARM, Color::hex(0x666F86)];
/// The same, light enough to read as words on the dark backing.
const REGION_TEXT: [Color; 4] = [Color::hex(0x86AEFF), Color::hex(0xA3B6DD), Color::hex(0xFF86B3), Color::hex(0xAEB5C5)];
/// How strongly each region's area is tinted.
const TINT: [f32; 4] = [0.2, 0.1, 0.2, 0.06];
/// How long a person takes to fade in or out (seconds: about two thirds of the way).
const FADE: f32 = 0.06;
/// How brightly a region the picture does not look at is drawn.
const UNLIT: f32 = 0.42;

/// The kinds of line, each in a batch of its own (see `plot::apart`): over everything,
/// and (see [`under`]) under the people.
const EDGES: u8 = plot::AXES;
const BAR_LINES: u8 = plot::MARKS;
const DIVISIONS: u8 = 0;
const DASHES: u8 = 1;
const RINGS: u8 = 2;
const RING_DASHES: u8 = 3;

/// A line in screen space under the people (who are drawn over it: in a region as
/// narrow as a person, they would otherwise vanish behind its edges), in a batch of
/// its kind.
fn under(f: &mut Frame, kind: u8, a: Vec2, b: Vec2, width: f32, color: Color, dashed: bool) {
    if !(a.is_finite() && b.is_finite()) {
        return;
    }
    let (a, b) = (f.hud_at(1, a), f.hud_at(1, b));
    let fade_power = 40.0 + kind as f32;
    let style = if dashed { LineStyle { dash: 7.0, duty: 0.55, depth_bias: 0.0, fade_power, ..Default::default() } } else { LineStyle { depth_bias: 0.0, fade_power, ..Default::default() } };
    f.dl.lines_with(LineKind::Blend, style).segment(a, b, width, color);
}

/// Where a region's count goes across: over the region's middle if it is wide enough,
/// and otherwise out past its `outer` edge (the square's) by as much as it lacks room,
/// so that the count never covers the people it counts. `inner` is its edge at the
/// divider, `w` the count's width. (A continuous function of the region's width, so
/// that a count moved out never jumps.) Returns the middle of the count, and how far
/// out it is (0 inside, 1 wholly outside).
fn count_x(outer: f32, inner: f32, w: f32) -> (f32, f32) {
    let room = (inner - outer).abs();
    let toward = if inner >= outer { 1.0 } else { -1.0 };
    let t = (room / (w + 14.0)).min(1.0);
    let outside = outer - toward * (10.0 + 0.5 * w);
    let middle = outer + toward * 0.5 * room;
    (outside + (middle - outside) * t, 1.0 - t)
}

// ----- what the picture glides through --------------------------------------------------------

/// The places in the list of numbers the picture glides through (besides the people).
/// The square's divisions as drawn: $P(A)$, and the two levels (3).
const SPLIT: usize = 0;
/// The probabilities themselves (3): where the handles and the green lines are.
const TRUE: usize = 3;
/// 1 while a random draw is shown.
const DRAWN: usize = 6;
/// How brightly each region is drawn (4).
const LIT: usize = 7;
/// The outlines round $A$ and round $B$.
const RING_A: usize = 11;
const RING_B: usize = 12;
/// Each bar's share (2), whether it has one (2), and how brightly it is drawn (2).
const SHARE: usize = 13;
const KNOWN: usize = 15;
const BAR_LIT: usize = 17;
/// The marks on the bars: the share among everyone (2), and the probability (2).
const EVERYONE: usize = 19;
const TRUE_SHARE: usize = 21;
/// How many people each region holds (4): gliding too, so that a region growing as its
/// division glides takes in its people as it goes.
const COUNT: usize = 23;
const NUMBERS: usize = 27;

/// What the picture is aimed at for one set of settings.
struct Aim {
    key: (Probs, bool, u64, Focus),
    counts: Counts,
    numbers: Vec<f32>,
}

// ----- the scene ----------------------------------------------------------------------------

pub struct Chance {
    truth: Probs,
    story: Story,
    /// A random draw of the people instead of the expected numbers, and its seed.
    drawn: bool,
    seed: u64,
    focus: Focus,
    aim: Option<Aim>,
    numbers: Glide,
    /// Each place in the square (see [`seats`]): its region as drawn and whether anyone
    /// stands there, and how brightly they are drawn (fading in and out).
    taken: Vec<(u8, bool)>,
    shown: Vec<f32>,
    /// People with no place in their region (see [`crowd`]).
    extra: [u32; 4],
    /// What the places were last given out for (the divisions and numbers as drawn), and
    /// whether any was still fading then.
    seated: Option<([f32; 3], [u32; 4])>,
    fading: bool,
    moving: bool,
}

impl Chance {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Chance {
        let mut s = Chance {
            truth: RARE,
            story: Story::Test,
            drawn: false,
            seed: FIRST_SEED,
            focus: Focus::Everyone,
            aim: None,
            numbers: Glide::default(),
            taken: Vec::with_capacity(SEATS),
            shown: vec![0.0; SEATS],
            extra: [0; 4],
            seated: None,
            fading: false,
            moving: false,
        };
        s.set_up(OPENING);
        s
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let Some(t) = TRIALS.get(k) else { return };
        self.truth = t.truth;
        self.story = t.story;
        self.focus = t.focus;
        self.drawn = false;
        if t.again {
            self.retest();
        }
    }

    /// Test the positives again: the share of them who have it is the new base rate.
    fn retest(&mut self) {
        if let Some(p) = self.truth.a_given_b() {
            self.truth.a = p;
        }
    }

    fn words(&self) -> &'static Words {
        &WORDS[STORIES.iter().position(|s| *s == self.story).unwrap_or(0)]
    }

    /// The people as shown: the expected numbers or this draw.
    fn counts(&self) -> Counts {
        if self.drawn {
            Counts::drawn(&self.truth, self.seed)
        } else {
            Counts::expected(&self.truth)
        }
    }

    /// Aim the picture at the settings as they are (worked out only when one changes).
    fn aim(&mut self) {
        self.truth = self.truth.kept();
        let key = (self.truth, self.drawn, self.seed, self.focus);
        if self.aim.as_ref().is_some_and(|a| a.key == key) {
            return;
        }
        let counts = self.counts();
        let split = split(&self.truth, &counts, self.drawn);
        let numbers = self.numbers_for(&counts, &split);
        self.aim = Some(Aim { key, counts, numbers });
    }

    /// The numbers the picture glides to, besides the people (see `SPLIT` and the rest).
    fn numbers_for(&self, counts: &Counts, split: &Probs) -> Vec<f32> {
        let t = &self.truth;
        let mut v = vec![0.0f32; NUMBERS];
        for (i, x) in [split.a, split.b_a, split.b_n, t.a, t.b_a, t.b_n].into_iter().enumerate() {
            v[SPLIT + i] = x as f32;
        }
        v[DRAWN] = if self.drawn { 1.0 } else { 0.0 };
        let lit = match self.focus {
            Focus::Everyone => [1.0, 1.0, 1.0, 1.0],
            Focus::A => [1.0, 1.0, UNLIT, UNLIT],
            Focus::B => [1.0, UNLIT, 1.0, UNLIT],
            Focus::Either => [1.0, 1.0, 1.0, UNLIT],
        };
        v[LIT..LIT + 4].copy_from_slice(&lit);
        v[RING_A] = if matches!(self.focus, Focus::A | Focus::Either) { 1.0 } else { 0.0 };
        v[RING_B] = if matches!(self.focus, Focus::B | Focus::Either) { 1.0 } else { 0.0 };
        let shares = self.shares(counts);
        for w in 0..2 {
            v[SHARE + w] = shares[w].unwrap_or(0.0) as f32;
            v[KNOWN + w] = if shares[w].is_some() { 1.0 } else { 0.0 };
        }
        v[BAR_LIT] = if matches!(self.focus, Focus::B | Focus::Either) { 0.5 } else { 1.0 };
        v[BAR_LIT + 1] = if matches!(self.focus, Focus::A | Focus::Either) { 0.5 } else { 1.0 };
        v[EVERYONE] = t.b() as f32;
        v[EVERYONE + 1] = t.a as f32;
        v[TRUE_SHARE] = t.b_given_a().unwrap_or(0.0) as f32;
        v[TRUE_SHARE + 1] = t.a_given_b().unwrap_or(0.0) as f32;
        // (in thousands, as the rest are shares: a glide comes to rest within a fixed
        // distance of its goal, and that should be a small part of a person)
        for k in 0..4 {
            v[COUNT + k] = counts.0[k] as f32 / PEOPLE as f32;
        }
        v
    }

    /// Seat the people as the picture now stands (its divisions and numbers as drawn),
    /// and fade each place towards being taken or empty over `dt` seconds (at once
    /// with `snap`). Says whether any is still fading.
    fn seat(&mut self, dt: f32, snap: bool) -> bool {
        let n = &self.numbers;
        let counts = [0, 1, 2, 3].map(|k| (n.get(COUNT + k) * PEOPLE as f32).round().max(0.0) as u32);
        let key = ([n.get(SPLIT), n.get(SPLIT + 1), n.get(SPLIT + 2)], counts);
        // (nothing to do for a picture at rest: the same places, none of them fading)
        if self.seated == Some(key) && !self.fading && !snap {
            return false;
        }
        self.seated = Some(key);
        self.extra = crowd(key.0[0], key.0[1], key.0[2], counts, &mut self.taken);
        let k = 1.0 - (-dt.max(0.0) / FADE).exp();
        let mut fading = false;
        for (shown, &(_, here)) in self.shown.iter_mut().zip(&self.taken) {
            let to = if here { 1.0 } else { 0.0 };
            *shown = if snap { to } else { *shown + (to - *shown) * k };
            if (to - *shown).abs() < 0.004 {
                *shown = to;
            } else {
                fading = true;
            }
        }
        self.fading = fading;
        fading
    }

    /// The bars' shares: of the $A$'s, the share in $B$; of the $B$'s, the share in $A$.
    /// The probabilities themselves, or the shares of this draw; none for a group with
    /// no one in it.
    fn shares(&self, counts: &Counts) -> [Option<f64>; 2] {
        if self.drawn {
            let of = |k: u32, n: u32| (n > 0).then(|| k as f64 / n as f64);
            [of(counts.0[BOTH], counts.a()), of(counts.0[BOTH], counts.b())]
        } else {
            [self.truth.b_given_a(), self.truth.a_given_b()]
        }
    }

    /// The picture at its goal at once (the tests of this file).
    #[cfg(test)]
    fn settle(&mut self) {
        self.aim();
        if let Some(aim) = &self.aim {
            self.numbers.snap(&aim.numbers);
        }
        self.seat(0.0, true);
    }

    // ----- what is written -----------------------------------------------------------------

    /// The line over the picture.
    fn title(&self) -> String {
        let w = self.words();
        format!("{} {}: {}", count(PEOPLE as usize), if self.drawn { w.drawn } else { w.people }, w.what)
    }

    /// What one bar says under it.
    fn bar_value(&self, which: usize, counts: &Counts) -> String {
        let w = self.words();
        let (part, of) = (counts.0[BOTH], if which == 0 { counts.a() } else { counts.b() });
        let name = if which == 0 { r"P(B \mid A)" } else { r"P(A \mid B)" };
        if self.drawn {
            if of == 0 {
                return format!("{} in this draw: no share to take.", if which == 0 { w.no_a } else { w.no_b });
            }
            return format!("in this draw: {} of {}, {} %", part, of, pct(part as f64 / of as f64));
        }
        match self.shares(counts)[which] {
            Some(p) => format!(r"${name} = {}\%$   ({} of {})", pct(p), part, of),
            None => format!(r"{}: ${name}$ is undefined.", if which == 0 { w.no_a } else { w.no_b }),
        }
    }

    /// What the mark on a bar is called: the share among everyone, or (under a random
    /// draw) the probability.
    fn mark_names(&self, which: usize) -> [String; 2] {
        let t = &self.truth;
        let everyone = if which == 0 { t.b() } else { t.a };
        let truth = if which == 0 { t.b_given_a() } else { t.a_given_b() };
        [format!("everyone: {} %", pct(everyone)), truth.map_or(String::new(), |p| format!("probability: {} %", pct(p)))]
    }

    /// The words beside the bars: Bayes' rule in odds and whether the events are
    /// independent; or, looking at $A$ or $B$, how the two add up.
    fn reasons(&self, counts: &Counts) -> String {
        let (t, w) = (&self.truth, self.words());
        if self.focus == Focus::Either {
            let c = &counts.0;
            let m = |v: u32| count_math(v as usize);
            return format!(
                r"$P(A \text{{ or }} B) = P(A) + P(B) - P(A \text{{ and }} B)$: those in both are counted in $P(A)$ and again in $P(B)$, so they are taken away once. In people: ${} + {} - {} = {}$ of the {}. Or take those in neither from everyone: ${} - {} = {}$.",
                m(counts.a()),
                m(counts.b()),
                m(c[BOTH]),
                m(counts.either()),
                count(PEOPLE as usize),
                m(PEOPLE),
                m(c[NEITHER]),
                m(counts.either())
            );
        }
        let bayes = if t.b() <= 0.0 {
            format!("{}: there is nothing to learn from, and $P(A \\mid B)$ is undefined.", w.no_b)
        } else if t.a <= 0.0 {
            format!("{}: the odds are 0 before {} and stay 0 after.", w.no_a, w.learning)
        } else if t.a >= 1.0 {
            format!("{}: there is nothing left for {} to change.", w.all_a, w.learning)
        } else if t.b_n <= 0.0 {
            r"$P(B \mid \text{not } A) = 0$: every $B$ is an $A$, so $P(A \mid B) = 100\%$.".to_string()
        } else if t.b_a <= 0.0 {
            r"$P(B \mid A) = 0$: no $B$ is an $A$, so $P(A \mid B) = 0\%$.".to_string()
        } else {
            let before = odds(t.a).unwrap_or(0.0);
            let ratio = t.ratio().unwrap_or(1.0);
            format!(
                "{} are {} at first; {} multiplies them by the likelihood ratio, ${} / {} = {}$, to {}.",
                w.odds_of,
                odds_text(before),
                w.learning,
                dec(t.b_a),
                dec(t.b_n),
                sig(ratio),
                odds_text(before * ratio)
            )
        };
        let independence = if t.a <= 0.0 || t.a >= 1.0 {
            String::new()
        } else if t.independent() {
            // (the product written out where it is exact: both to a tenth of a per cent)
            let whole = |v: f64| (v * 1000.0 - (v * 1000.0).round()).abs() < 1e-9;
            let product = if whole(t.a) && whole(t.b_a) {
                format!(r" = {} \times {} = {}", dec(t.a), dec(t.b()), trimmed(t.a * t.b(), 6))
            } else {
                String::new()
            };
            format!(r" The levels are equal: $A$ and $B$ are independent, $P(A \text{{ and }} B) = P(A)\,P(B){product}$.")
        } else {
            " The levels differ: $A$ and $B$ are not independent.".to_string()
        };
        format!("{bayes}{independence}")
    }
}

// ----- the layout ---------------------------------------------------------------------------

const MODES: [&str; 3] = ["expected numbers", "a random draw", "draw again"];
const MODES_SHORT: [&str; 3] = ["expected", "random draw", "draw again"];

/// How far under the square the divider's dot is.
const STEM: f32 = 15.0;

/// Where everything goes on a canvas (logical pixels, y down).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Too small for more than the square (and, `bare`, for its names round it).
    compact: bool,
    bare: bool,
    /// Too small for anything.
    tiny: bool,
    big: f32,
    text: f32,
    small: f32,
    /// The line over everything (a wide canvas only).
    title: Option<Vec2>,
    stories: [Rect; 3],
    modes: [Rect; 3],
    square: Rect,
    /// Room beside the square, left and right, for the levels' names.
    margin: [f32; 2],
    /// Room over the square for the two sides' names.
    head: f32,
    table: Rect,
    /// Each bar's block (its title, the bar and what it says), and the bar itself.
    blocks: [Rect; 2],
    bars: [Rect; 2],
    /// The words beside the bars (none where there is no room).
    reasons: Option<Rect>,
    again: Rect,
    looks: [Rect; 4],
    /// Where "look at" is written (a wide canvas only).
    look_label: Option<Vec2>,
}

const NO_RECT: Rect = (Vec2::ZERO, Vec2::ZERO);

/// Rectangles for buttons side by side from (x, y), each as wide as its words (as
/// `plot::button_row`, measured without a frame, so that the tests can lay out too).
fn row_of(words: &[&str], x: f32, y: f32, h: f32, size: f32, gap: f32) -> Vec<Rect> {
    let mut left = x;
    words
        .iter()
        .map(|w| {
            let width = text_width(w, size) + 18.0;
            let r = (Vec2::new(left, y), Vec2::new(left + width, y + h));
            left += width + gap;
            r
        })
        .collect()
}

fn three(v: Vec<Rect>) -> [Rect; 3] {
    [0, 1, 2].map(|i| v.get(i).copied().unwrap_or(NO_RECT))
}

fn four(v: Vec<Rect>) -> [Rect; 4] {
    [0, 1, 2, 3].map(|i| v.get(i).copied().unwrap_or(NO_RECT))
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 760.0;
        let (big, text, small) = if narrow { (13.0, 11.5, 10.5) } else { (15.0, 12.5, 11.5) };
        let side = if narrow { 8.0 } else { 20.0 };
        let top = inset.max(0.0) + if narrow { 6.0 } else { 12.0 };
        let row = 24.0;
        let mut lay = Layout {
            view,
            narrow,
            compact: false,
            bare: false,
            tiny: view.x < 60.0 || view.y < 60.0,
            big,
            text,
            small,
            title: None,
            stories: [NO_RECT; 3],
            modes: [NO_RECT; 3],
            square: NO_RECT,
            margin: [0.0; 2],
            head: 0.0,
            table: NO_RECT,
            blocks: [NO_RECT; 2],
            bars: [NO_RECT; 2],
            reasons: None,
            again: NO_RECT,
            looks: [NO_RECT; 4],
            look_label: None,
        };
        let story_words: Vec<&str> = WORDS.iter().map(|w| if narrow { w.short } else { w.button }).collect();
        if narrow {
            lay.stories = three(plot::even_row(3, side, view.x - side, top, row, 5.0));
            lay.margin = [38.0, 38.0];
            lay.head = 2.0 * (small + 4.0) + 8.0;
            let square_top = top + row + 8.0 + lay.head;
            // under the square: the divider's dot and name, the table, the bars, two rows
            // of buttons, and the words if there is room
            let table_h = 4.0 * (small + 7.0);
            let block_h = 2.0 * (small + 4.0) + 4.0 + 14.0 + small + 8.0;
            let below = STEM + small + 16.0 + table_h + 8.0 + 2.0 * (block_h + 6.0) + 2.0 * (row + 6.0) + 6.0;
            let reasons_h = 3.0 * (small + 3.0) + 8.0;
            let room = view.y - square_top - below - 4.0;
            let wide = view.x - 2.0 * side - lay.margin[0] - lay.margin[1];
            let s = wide.min(room).max(0.0);
            // (a square too small to read shrinks no further: the words make way first)
            let s = if room - reasons_h >= wide.min(260.0) { wide.min(room - reasons_h) } else { s };
            let left = side + lay.margin[0] + (wide - s) * 0.5;
            lay.square = (Vec2::new(left, square_top), Vec2::new(left + s, square_top + s));
            let mut y = square_top + s + STEM + small + 16.0;
            lay.table = (Vec2::new(side, y), Vec2::new(view.x - side, y + table_h));
            y += table_h + 8.0;
            for w in 0..2 {
                lay.blocks[w] = (Vec2::new(side, y), Vec2::new(view.x - side, y + block_h));
                let bar_top = y + 2.0 * (small + 4.0) + 4.0;
                lay.bars[w] = (Vec2::new(side + 4.0, bar_top), Vec2::new(view.x - side - 4.0, bar_top + 14.0));
                y += block_h + 6.0;
            }
            let modes = plot::even_row(4, side, view.x - side, y + 6.0, row, 5.0);
            lay.modes = three(modes.clone());
            lay.again = modes.get(3).copied().unwrap_or(NO_RECT);
            y += row + 6.0;
            lay.looks = four(plot::even_row(4, side, view.x - side, y + 6.0, row, 5.0));
            y += row + 12.0;
            if view.y - y >= reasons_h - 8.0 {
                lay.reasons = Some((Vec2::new(side, y), Vec2::new(view.x - side, view.y - 4.0)));
            }
            lay.compact = s < 150.0;
        } else {
            lay.title = Some(Vec2::new(side, top));
            let y = top + big + 12.0;
            let stories = row_of(&story_words, side, y, row, small, 6.0);
            let end = stories.last().map_or(side, |r| r.1.x);
            lay.stories = three(stories);
            lay.modes = three(row_of(&MODES, end + 26.0, y, row, small, 6.0));
            lay.margin = [86.0, 108.0];
            lay.head = 2.0 * (text + 5.0) + 10.0;
            let heads_top = y + row + 16.0;
            let square_top = heads_top + lay.head;
            let column_w = (view.x * 0.27).max(270.0).min(340.0);
            let wide = view.x - 2.0 * side - lay.margin[0] - lay.margin[1] - 24.0 - column_w;
            let high = view.y - square_top - STEM - 2.0 * small - 40.0;
            let s = wide.min(high).max(0.0);
            let left = side + lay.margin[0];
            lay.square = (Vec2::new(left, square_top), Vec2::new(left + s, square_top + s));
            // the right-hand column: everything the square leaves
            let x0 = left + s + lay.margin[1] + 24.0;
            let x1 = view.x - side;
            let mut y = heads_top;
            let table_h = (small + 10.0) + 4.0 * (text + 10.0);
            lay.table = (Vec2::new(x0, y), Vec2::new(x1, y + table_h));
            y += table_h + 22.0;
            let block_h = 2.0 * (small + 4.0) + (small + 6.0) + 18.0 + text + 10.0;
            for w in 0..2 {
                lay.blocks[w] = (Vec2::new(x0, y), Vec2::new(x1, y + block_h));
                let bar_top = y + 2.0 * (small + 4.0) + small + 6.0;
                lay.bars[w] = (Vec2::new(x0, bar_top), Vec2::new(x1, bar_top + 18.0));
                y += block_h + 14.0;
            }
            let reasons_h = 6.0 * (small + 3.0) + 6.0;
            lay.reasons = Some((Vec2::new(x0, y), Vec2::new(x1, y + reasons_h)));
            y += reasons_h + 12.0;
            let w = WORDS.iter().map(|w| text_width(w.retest, small)).fold(0.0, f32::max) + 18.0;
            lay.again = (Vec2::new(x0, y), Vec2::new(x0 + w.min(x1 - x0), y + row));
            y += row + 16.0;
            lay.look_label = Some(Vec2::new(x0, y));
            y += small + 8.0;
            lay.looks = four(row_of(&FOCUS_NAMES, x0, y, row, small, 6.0));
            lay.compact = s < 150.0 || x1 - x0 < 200.0;
        }
        if lay.compact {
            // the square alone, as large as the canvas allows, with its names round it
            // where there is room for them, and with nothing at all where there is not
            lay.margin = [38.0, 38.0];
            lay.head = 2.0 * (small + 4.0) + 8.0;
            let below = STEM + small + 18.0;
            let named = (view.x - 2.0 * side - 76.0).min(view.y - top - lay.head - below - side).max(0.0);
            lay.bare = named < 120.0;
            let (s, up) = if lay.bare { ((view.x - 2.0 * side).min(view.y - top - 2.0 * side).max(0.0), 0.0) } else { (named, lay.head) };
            let corner = Vec2::new((view.x - s) * 0.5, top + up + (view.y - top - up - s - if lay.bare { 0.0 } else { below }) * 0.5);
            lay.square = (corner, corner + Vec2::splat(s));
            lay.tiny |= s < 40.0;
        }
        lay
    }
}

// ----- the scene's frame ----------------------------------------------------------------------

impl Scene for Chance {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.moving
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("green dots: drag the divider and the two levels    buttons over the picture: click")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(
            ui,
            r"A probability is a share. Here 1 000 people stand in a square. The left part holds those who have a condition, $A$, and is as wide as their share, $P(A)$. In each part, the bottom holds those who test positive, $B$, as high as their share of that part. So every region's area is the share of everyone in it, and its dots are its people. A conditional probability, such as $P(A \mid B)$, is a share of one group only: look at the $B$'s alone, and ask what share of them are $A$'s.",
        );

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The story");
        let mut story = STORIES.iter().position(|s| *s == self.story).unwrap_or(0);
        let names: Vec<&str> = WORDS.iter().map(|w| w.button).collect();
        if ui::choice(ui, &names, &mut story) {
            self.story = STORIES[story.min(2)];
        }
        ui::note(ui, "The same numbers, told three ways.");

        ui::section(ui, "The probabilities");
        let mut a = self.truth.a as f32;
        if ui::slider(ui, r"$P(A)$", &mut a, 0.0..=1.0) {
            self.truth.a = snap(a as f64, 1000.0);
        }
        ui::note(ui, "The base rate: how common the condition is. Or drag the green dot under the square.");
        let mut b_a = self.truth.b_a as f32;
        if ui::slider(ui, r"$P(B \mid A)$", &mut b_a, 0.0..=1.0) {
            self.truth.b_a = snap(b_a as f64, 1000.0);
        }
        ui::note(ui, "The sensitivity: the share of those who have it whom the test finds. Or drag the level on the left.");
        let mut b_n = self.truth.b_n as f32;
        if ui::slider(ui, r"$P(B \mid \text{not } A)$", &mut b_n, 0.0..=1.0) {
            self.truth.b_n = snap(b_n as f64, 1000.0);
        }
        ui::note(ui, "The false alarm rate: the share of the rest whom it flags anyway. Or drag the level on the right.");
        let w = self.words();
        if ui::buttons(ui, &[w.retest]).is_some() {
            self.retest();
        }
        ui::note(ui, r"The share of the positives who have it, $P(A \mid B)$, becomes the new $P(A)$: the positives take a second test of the same kind, whose mistakes are independent of the first's.");

        ui::section(ui, "The people");
        let mut mode = usize::from(self.drawn);
        if ui::choice(ui, &MODES[..2], &mut mode) {
            self.drawn = mode == 1;
        }
        if ui::buttons(ui, &[MODES[2]]).is_some() {
            self.drawn = true;
            self.seed = self.seed.wrapping_add(1);
        }
        ui::note(ui, "Expected numbers: each region holds its share of the 1 000, rounded to whole people. A random draw: each person is in $A$ with probability $P(A)$, and in $B$ with the probability of their side, so the counts come out differently every time, round the expected numbers.");

        ui::section(ui, "Look at");
        let mut focus = FOCI.iter().position(|f| *f == self.focus).unwrap_or(0);
        if ui::choice(ui, &FOCUS_NAMES, &mut focus) {
            self.focus = FOCI[focus.min(3)];
        }
        ui::note(ui, r"Only $A$: the probabilities given $A$ are shares of this group alone. Only $B$: the same for $B$. $A$ or $B$: the three regions of either or both.");

        ui::section(ui, "Reading the picture");
        let w = self.words();
        let keys: Vec<String> = ["blue", "dim blue", "pink", "grey"].iter().zip(w.key).map(|(c, k)| format!("{c}: {k}")).collect();
        ui::key(
            ui,
            &[
                (REGION[BOTH], keys[0].as_str()),
                (REGION[A_ONLY], keys[1].as_str()),
                (REGION[B_ONLY], keys[2].as_str()),
                (REGION[NEITHER], keys[3].as_str()),
                (TRUTH, "green: the probabilities you set, the divider and the level of $B$ on each side; drag their dots"),
                (DATA, "cyan: in a random draw, where the people drawn divide the square (the probabilities are then dashed)"),
                (pal::WHITE, "white: the group looked at; on a bar, the share among everyone"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(ui, "Event:", "something that is true or not of each person: here $A$, having the condition, and $B$, testing positive.");
        ui::term(ui, "Probability:", r"the share of everyone of whom an event is true: a share of the 1 000, the area of its part of the square. Its complement, $P(\text{not } A) = 1 - P(A)$, is the rest.");
        ui::term(ui, "Joint probability:", r"$P(A \text{ and } B)$, the share of whom both are true: one region.");
        ui::term(ui, "Conditional probability:", r"$P(B \mid A) = P(A \text{ and } B) / P(A)$, the share of $B$'s among the $A$'s alone. Undefined when there are no $A$'s.");
        ui::term(ui, "Independence:", r"$P(A \text{ and } B) = P(A)\,P(B)$. With some $A$'s and some not, that is the two levels being equal: knowing whether $A$ is true changes nothing about $B$.");
        ui::term(ui, "Base rate (prior):", r"$P(A)$, how common the condition is before any test.");
        ui::term(ui, "Sensitivity:", r"$P(B \mid A)$, the share of those who have it whom the test finds.");
        ui::term(ui, "False alarm rate:", r"$P(B \mid \text{not } A)$, the share of the rest whom it flags anyway.");
        ui::term(ui, "Bayes' rule:", r"$P(A \mid B) = P(B \mid A)\,P(A) / P(B)$: the region in both, over the two regions of $B$.");
        ui::term(ui, "Odds:", r"a probability $p$ as $p : (1 - p)$; 1 % is odds of $1 : 99$.");
        ui::term(ui, "Likelihood ratio:", r"$P(B \mid A) / P(B \mid \text{not } A)$, how many times likelier $B$ is with $A$ than without. The odds after learning $B$ are the odds before times this.");

        ui::section(ui, "Numbers");
        let t = self.truth;
        ui::readout(ui, r"$P(A)$", pct_or(Some(t.a)));
        ui::readout(ui, r"$P(B \mid A)$", pct_or(t.b_given_a()));
        ui::readout(ui, r"$P(B \mid \text{not } A)$", pct_or(t.b_given_not_a()));
        ui::readout(ui, r"$P(B)$", pct_or(Some(t.b())));
        ui::readout(ui, r"$P(A \text{ and } B)$", pct_or(Some(t.joint()[BOTH])));
        ui::readout(ui, r"$P(A \mid B)$", pct_or(t.a_given_b()));
        ui::readout(ui, r"$P(A \text{ or } B)$", pct_or(Some(t.either())));
        ui::readout(ui, "independent", if t.independent() { "yes" } else { "no" });
        if let (Some(before), Some(ratio)) = (odds(t.a).filter(|o| *o > 0.0), t.ratio()) {
            ui::readout(ui, "odds before", odds_text(before));
            ui::readout(ui, "likelihood ratio", sig(ratio));
            ui::readout(ui, "odds after", odds_text(before * ratio));
        }
        if self.drawn {
            let c = self.counts();
            ui::readout(ui, r"this draw: $A$, $B$, both", format!("{}, {}, {}", c.a(), c.b(), c.0[BOTH]));
        }
        ui::see_also(
            ui,
            &[
                ("distributions", "the counts in a random draw are binomial"),
                ("bayes", "the same rule for an unknown that can be any number"),
                ("pvalues", "what share of significant findings are real: the same calculation"),
                ("confound", "when two events go together without one causing the other"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let lay = Layout::of(f.view_size(), f.top_inset);
        if !lay.tiny && !lay.bare {
            self.handles(f, &lay);
        }
        if !lay.tiny && !lay.compact {
            self.buttons(f, &lay);
        }
        self.aim();
        let Some(aim) = &self.aim else { return };
        let a = self.numbers.follow(&aim.numbers, f.real_dt, false);
        // (the places are taken as the picture stands, divisions and numbers gliding)
        let b = self.seat(f.real_dt, false);
        self.moving = a || b;
        if lay.tiny {
            return;
        }
        self.draw_square(f, &lay);
        if !lay.bare {
            self.draw_names(f, &lay);
        }
        if !lay.compact {
            if let Some(at) = lay.title {
                f.text_px(at, self.title(), lay.big, pal::TEXT, Align2::LEFT_TOP);
            }
            self.draw_table(f, &lay);
            for which in 0..2 {
                self.draw_bar(f, &lay, which);
            }
            self.draw_reasons(f, &lay);
        }
    }
}

// ----- input --------------------------------------------------------------------------------

impl Chance {
    /// The three green dots: the divider (on a stem under the square) and the level of
    /// $B$ on each side. They set the probabilities, which snap to whole tenths of a
    /// per cent ($P(A)$) and whole per cents (the levels), so that equal levels can be
    /// found by hand.
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.square;
        let side = (max.x - min.x).max(1.0);
        let n = &self.numbers;
        let (a, b_a, b_n) = (n.get(TRUE), n.get(TRUE + 1), n.get(TRUE + 2));
        let x = |u: f32| min.x + u * side;
        let y = |v: f32| max.y - v * side;
        let up = |p: Vec2| ((max.y - p.y) / side) as f64;
        if let Some(p) = plot::screen_handle(f, 1, Vec2::new(x(a), max.y + STEM), TRUTH) {
            self.truth.a = snap(((p.x - min.x) / side) as f64, 1000.0);
        }
        if let Some(p) = plot::screen_handle(f, 2, Vec2::new(x(0.5 * a), y(b_a)), TRUTH) {
            self.truth.b_a = snap(up(p), 100.0);
        }
        if let Some(p) = plot::screen_handle(f, 3, Vec2::new(x(0.5 + 0.5 * a), y(b_n)), TRUTH) {
            self.truth.b_n = snap(up(p), 100.0);
        }
    }

    /// The buttons over the picture.
    fn buttons(&mut self, f: &mut Frame, lay: &Layout) {
        let size = lay.small;
        for (k, w) in WORDS.iter().enumerate() {
            let text = if lay.narrow { w.short } else { w.button };
            if plot::button(f, lay.stories[k], text, size, self.story == STORIES[k]) {
                self.story = STORIES[k];
            }
        }
        let modes = if lay.narrow { MODES_SHORT } else { MODES };
        for (k, text) in modes.iter().enumerate() {
            let on = (k == 0 && !self.drawn) || (k == 1 && self.drawn);
            if plot::button(f, lay.modes[k], text, size, on) {
                match k {
                    0 => self.drawn = false,
                    1 => self.drawn = true,
                    _ => {
                        self.drawn = true;
                        self.seed = self.seed.wrapping_add(1);
                    }
                }
            }
        }
        let w = self.words();
        // (with no positives there is no one to test again, and the button says so)
        let again = match (self.truth.b() > 0.0, lay.narrow) {
            (true, true) => w.retest_short,
            (true, false) => w.retest,
            (false, true) => "none to test",
            (false, false) => "no one to test again",
        };
        if plot::button(f, lay.again, again, size, false) {
            self.retest();
        }
        for (k, text) in FOCUS_NAMES.iter().enumerate() {
            if plot::button(f, lay.looks[k], text, size, self.focus == FOCI[k]) {
                self.focus = FOCI[k];
            }
        }
        if let Some(at) = lay.look_label {
            f.text_px(at, "look at", size, pal::DIM, Align2::LEFT_TOP);
        }
    }
}

// ----- drawing ------------------------------------------------------------------------------

impl Chance {
    /// The square: its regions, its people, its dividing lines, and the outlines of the
    /// group looked at.
    fn draw_square(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.square;
        let side = (max.x - min.x).max(1.0);
        let n = &self.numbers;
        let at = |u: f32, v: f32| Vec2::new(min.x + u * side, max.y - v * side);
        let px = |r: [f32; 4]| (at(r[0], r[3]), at(r[1], r[2]));
        if lay.bare {
            hud_panel(f, min - Vec2::splat(6.0), max + Vec2::splat(6.0));
        } else {
            let below = STEM + lay.small + 18.0 + if lay.narrow || lay.compact { 0.0 } else { lay.small + 10.0 };
            hud_panel(f, min - Vec2::new(lay.margin[0] - 6.0, lay.head + 6.0), max + Vec2::new(lay.margin[1] - 6.0, below));
        }
        let all = rects(n.get(SPLIT), n.get(SPLIT + 1), n.get(SPLIT + 2));
        for k in 0..4 {
            plot::fill(f, px(all[k]), REGION[k].a(TINT[k] * n.get(LIT + k)));
        }

        // the people, at their places, over the lines: never much smaller than their
        // full size, so that a region as narrow as a person still shows who is in it
        let size = (0.56 * side / (PEOPLE as f32).sqrt()).max(3.5);
        let lit = [0, 1, 2, 3].map(|k| 0.95 * n.get(LIT + k));
        // (every place up to the last one anyone stands in, the empty ones unseen: so
        // that a place keeps its place in the batch while people come and go)
        let last = self.shown.iter().rposition(|v| *v > 0.0).map_or(0, |i| i + 1);
        let mut dots: Vec<(Vec3, Color)> = Vec::with_capacity(last + 4);
        for ((&[x, y], &(k, _)), &shown) in seats().iter().zip(&self.taken).zip(&self.shown).take(last) {
            let k = (k as usize).min(3);
            dots.push((f.hud_at(2, at(x, y)), REGION[k].a(lit[k] * shown)));
        }
        // (and any a region had no room for, at its middle: never, at rest)
        for k in 0..4 {
            if self.extra[k] > 0 {
                let [x0, x1, y0, y1] = all[k];
                dots.push((f.hud_at(2, at(0.5 * (x0 + x1), 0.5 * (y0 + y1))), REGION[k].a(lit[k])));
            }
        }
        let style = PointStyle { outline_px: 0.0, size_scale: 1.0001, ..Default::default() };
        let batch = f.dl.points_with(PointKind::Disc, style);
        for (p, color) in dots {
            batch.push(p, size, color);
        }

        // the edge of the square
        let edge = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
        for i in 0..4 {
            under(f, DIVISIONS, edge[i], edge[i + 1], 1.0, pal::DIM.a(0.55), false);
        }

        // the probabilities (solid, or dashed under a random draw) and the draw's own divisions
        let d = n.get(DRAWN);
        let (ta, tba, tbn) = (n.get(TRUE), n.get(TRUE + 1), n.get(TRUE + 2));
        let truth_lines = [(at(ta, 0.0) + Vec2::new(0.0, if lay.bare { 0.0 } else { STEM }), at(ta, 1.0)), (at(0.0, tba), at(ta, tba)), (at(ta, tbn), at(1.0, tbn))];
        for (a, b) in truth_lines {
            under(f, DIVISIONS, a, b, 2.0, TRUTH.a(1.0 - d), false);
            under(f, DASHES, a, b, 1.8, TRUTH.a(0.9 * d), true);
        }
        let (a, ba, bn) = (n.get(SPLIT), n.get(SPLIT + 1), n.get(SPLIT + 2));
        for (p, q) in [(at(a, 0.0), at(a, 1.0)), (at(0.0, ba), at(a, ba)), (at(a, bn), at(1.0, bn))] {
            under(f, DIVISIONS, p, q, 1.6, DATA.a(d), false);
        }

        // the group looked at: A within a solid line, B within a dashed one
        let ring = |r: [f32; 4]| {
            let (lo, hi) = px(r);
            [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y), lo]
        };
        let (ring_a, ring_b) = (n.get(RING_A), n.get(RING_B));
        let around_a = ring([0.0, a, 0.0, 1.0]);
        for i in 0..4 {
            under(f, RINGS, around_a[i], around_a[i + 1], 2.0, pal::WHITE.a(0.9 * ring_a), false);
        }
        for r in [all[BOTH], all[B_ONLY]] {
            let pts = ring(r);
            for i in 0..4 {
                under(f, RING_DASHES, pts[i], pts[i + 1], 2.0, pal::WHITE.a(0.9 * ring_b), true);
            }
        }
    }

    /// The names round the square: the two sides over it, each region's count in it, the
    /// levels beside it and $P(A)$ under it.
    fn draw_names(&self, f: &mut Frame, lay: &Layout) {
        let Some(aim) = &self.aim else { return };
        let (min, max) = lay.square;
        let side = (max.x - min.x).max(1.0);
        let n = &self.numbers;
        let at = |u: f32, v: f32| Vec2::new(min.x + u * side, max.y - v * side);
        let (w, t) = (self.words(), &self.truth);
        let c = &aim.counts;

        // the two sides over the square, each over a bracket as wide as it is
        let a = n.get(SPLIT);
        let bracket_y = min.y - 8.0;
        for (u0, u1) in [(0.0, a), (a, 1.0)] {
            let (p, q) = (at(u0, 1.0), at(u1, 1.0));
            let (p, q) = (Vec2::new(p.x + 1.0, bracket_y), Vec2::new(q.x - 1.0, bracket_y));
            plot::line_of(f, EDGES, p, Vec2::new(q.x.max(p.x), q.y), 1.0, pal::DIM.a(0.8));
            plot::line_of(f, EDGES, p, p + Vec2::new(0.0, 4.0), 1.0, pal::DIM.a(0.8));
            plot::line_of(f, EDGES, Vec2::new(q.x.max(p.x), q.y), Vec2::new(q.x.max(p.x), q.y + 4.0), 1.0, pal::DIM.a(0.8));
        }
        let heads = [format!("{}: {}", w.a, count(c.a() as usize)), format!("{}: {}", w.not_a, count(c.not_a() as usize))];
        let widths = heads.each_ref().map(|h| f.text_size(h, lay.text).x);
        let centres = [at(0.5 * a, 1.0).x, at(0.5 + 0.5 * a, 1.0).x];
        let mut left = [centres[0] - 0.5 * widths[0], centres[1] - 0.5 * widths[1]];
        // (apart where they would meet, and inside the backing: each a continuous
        // function of the divider, so that neither ever jumps)
        let short = left[0] + widths[0] + 14.0 - left[1];
        if short > 0.0 {
            left[0] -= 0.5 * short;
            left[1] += 0.5 * short;
        }
        let (lo, hi) = (min.x - lay.margin[0] + 10.0, max.x + lay.margin[1] - 10.0);
        let push = (lo - left[0]).max(0.0) - (left[1] + widths[1] - hi).max(0.0);
        for k in 0..2 {
            f.text_px(Vec2::new(left[k] + push, bracket_y - 5.0), heads[k].as_str(), lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        }

        // each region's count, in it (or out past the square's edge where it is too
        // narrow), clear of the dot on its level, and with its name where there is room
        let (ba, bn) = (n.get(SPLIT + 1), n.get(SPLIT + 2));
        let (tba, tbn) = (n.get(TRUE + 1), n.get(TRUE + 2));
        let all = rects(a, ba, bn);
        for k in 0..4 {
            let r = all[k];
            let (lo, hi) = (at(r[0], r[3]), at(r[1], r[2]));
            let number = if c.0[k] > 0 { count(c.0[k] as usize) } else { String::new() };
            let size = lay.text + 1.0;
            let wide = f.text_size(&number, size).x;
            let (x, out) = if k < 2 { count_x(lo.x, hi.x, wide) } else { count_x(hi.x, lo.x, wide) };
            // (looking at A or B, those in both are the ones counted twice)
            let region_name = if self.focus == Focus::Either && k == BOTH { "in both: counted twice" } else { w.regions[k] };
            let named = out == 0.0 && hi.x - lo.x > f.text_size(region_name, lay.small).x + 10.0 && hi.y - lo.y > 2.0 * lay.text + 14.0;
            let clear = if named { 30.0 } else { 18.0 + 10.0 * out };
            let level = if k < 2 { at(0.0, tba).y } else { at(0.0, tbn).y };
            let middle = 0.5 * (lo.y + hi.y);
            let y = if k == BOTH || k == B_ONLY { middle.max(level + clear) } else { middle.min(level - clear) };
            let lift = if named { 0.5 * lay.text + 1.0 } else { 0.0 };
            let lit = n.get(LIT + k).max(0.45);
            f.caption_px(Vec2::new(x, y - lift), number, size, REGION_TEXT[k].a(lit), Align2::CENTER_CENTER);
            let at = Vec2::new(x, y + lift + 2.0);
            if named && c.0[k] > 0 {
                let color = if self.focus == Focus::Either && k == BOTH { pal::TEXT } else { pal::DIM };
                f.caption_px(at, region_name, lay.small, color.a(lit), Align2::CENTER_CENTER);
            } else {
                // (an empty label in its place, so that the labels keep their places in the list)
                f.text_px(at, "", lay.small, pal::DIM, Align2::CENTER_CENTER);
            }
        }

        // the levels, beside the square at their heights (on a wide canvas, the name
        // over the line and the value under it)
        let values = [t.b_given_a(), t.b_given_not_a()].map(|p| p.map_or("undefined".to_string(), |p| format!("{} %", pct(p))));
        let ends = [(min.x - 8.0, at(0.0, tba).y, Align2::RIGHT_BOTTOM, Align2::RIGHT_TOP), (max.x + 8.0, at(0.0, tbn).y, Align2::LEFT_BOTTOM, Align2::LEFT_TOP)];
        let names = [r"$P(B \mid A)$", r"$P(B \mid \text{not } A)$"];
        for k in 0..2 {
            let (x, y, above, below) = ends[k];
            if lay.narrow || lay.compact {
                let align = if k == 0 { Align2::RIGHT_CENTER } else { Align2::LEFT_CENTER };
                f.text_px(Vec2::new(x, y), values[k].as_str(), lay.small, TRUTH, align);
            } else {
                f.text_px(Vec2::new(x, y - 1.0), names[k], lay.small, TRUTH, above);
                f.text_px(Vec2::new(x, y + 2.0), values[k].as_str(), lay.small, TRUTH, below);
            }
        }

        // P(A), under the divider's dot
        let name = format!(r"$P(A) = {}\%$", pct(t.a));
        let half = 0.5 * f.text_size(&name, lay.small).x;
        let x = at(n.get(TRUE), 0.0).x.max(min.x - lay.margin[0] + 10.0 + half).min(max.x + lay.margin[1] - 10.0 - half);
        f.text_px(Vec2::new(x, max.y + STEM + 10.0), name, lay.small, TRUTH, Align2::CENTER_TOP);

        // what the lines are, under it all (a wide canvas only)
        if !lay.narrow && !lay.compact {
            let note = if self.drawn {
                "dashed green: the probabilities, set by the green dots    cyan: how the people of this draw divide the square"
            } else {
                "each dot is one of the 1 000; each region's area is its probability    drag the green dots"
            };
            f.text_px(Vec2::new(min.x - lay.margin[0] + 10.0, max.y + STEM + lay.small + 22.0), note, lay.small, pal::DIM, Align2::LEFT_TOP);
        }
    }

    /// The four counts again, with the totals of each row and column.
    fn draw_table(&self, f: &mut Frame, lay: &Layout) {
        let Some(aim) = &self.aim else { return };
        let (min, max) = lay.table;
        let (w, c) = (self.words(), &aim.counts);
        hud_panel(f, min - Vec2::splat(6.0), max + Vec2::splat(6.0));
        let mut y = min.y;
        if !lay.narrow {
            let title = if self.drawn { "this random draw" } else { "the expected numbers, rounded to whole people" };
            f.text_px(Vec2::new(min.x, y), title, lay.small, pal::DIM, Align2::LEFT_TOP);
            y += lay.small + 10.0;
        }
        let row_h = (max.y - y) / 4.0;
        let width = max.x - min.x;
        let first = width * 0.27;
        let column = (width - first) / 3.0;
        let right = |j: usize| min.x + first + column * (j as f32 + 1.0) - 6.0;
        let size = if lay.narrow { lay.small } else { lay.text };
        let heads = [w.b, w.not_b, "total"];
        for (j, h) in heads.iter().enumerate() {
            f.text_px(Vec2::new(right(j), y + 0.5 * row_h), *h, lay.small, pal::DIM, Align2::RIGHT_CENTER);
        }
        let rows: [(&str, [(u32, Color); 3]); 3] = [
            (w.a, [(c.0[BOTH], REGION_TEXT[BOTH]), (c.0[A_ONLY], REGION_TEXT[A_ONLY]), (c.a(), pal::TEXT)]),
            (w.not_a, [(c.0[B_ONLY], REGION_TEXT[B_ONLY]), (c.0[NEITHER], REGION_TEXT[NEITHER]), (c.not_a(), pal::TEXT)]),
            ("total", [(c.b(), pal::TEXT), (c.not_b(), pal::TEXT), (c.total(), pal::TEXT)]),
        ];
        for (i, (name, cells)) in rows.iter().enumerate() {
            let cy = y + (i as f32 + 1.5) * row_h;
            f.text_px(Vec2::new(min.x, cy), *name, lay.small, pal::DIM, Align2::LEFT_CENTER);
            for (j, (v, color)) in cells.iter().enumerate() {
                f.text_px(Vec2::new(right(j), cy), count(*v as usize), size, *color, Align2::RIGHT_CENTER);
            }
        }
        let line_y = y + 3.0 * row_h + 1.0;
        plot::line_of(f, EDGES, Vec2::new(min.x, line_y), Vec2::new(max.x, line_y), 1.0, pal::DIM.a(0.4));
    }

    /// One bar: the $A$'s split into $B$ and not (0), or the $B$'s split into $A$ and not
    /// (1), each as long as the other, so that the two shares can be compared.
    fn draw_bar(&self, f: &mut Frame, lay: &Layout, which: usize) {
        let Some(aim) = &self.aim else { return };
        let (bmin, bmax) = lay.blocks[which];
        let (min, max) = lay.bars[which];
        let n = &self.numbers;
        let w = self.words();
        hud_panel(f, bmin - Vec2::splat(6.0), bmax + Vec2::splat(4.0));
        let lit = n.get(BAR_LIT + which);
        let title = if which == 0 { w.of_a } else { w.of_b };
        let title = if f.text_size(title, lay.small).x <= bmax.x - bmin.x { title.to_string() } else { title.split(", the share").next().unwrap_or(title).to_string() };
        f.text_px(bmin, title, lay.small, pal::TEXT.a(0.4 + 0.6 * lit), Align2::LEFT_TOP);

        let width = max.x - min.x;
        let x_at = |share: f32| min.x + share.max(0.0).min(1.0) * width;
        let (share, known) = (n.get(SHARE + which), n.get(KNOWN + which));
        hud_rect(f, min, max, Color::hex(0x161C2B));
        let colors = if which == 0 { [REGION[BOTH], REGION[A_ONLY]] } else { [REGION[BOTH], REGION[B_ONLY]] };
        let split_x = x_at(share);
        hud_rect_at(f, 2, min, Vec2::new(split_x, max.y), colors[0].a(0.9 * known * lit));
        hud_rect_at(f, 2, Vec2::new(split_x, min.y), max, colors[1].a(0.75 * known * lit));
        let edge = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y), min];
        plot::path_of(f, BAR_LINES, &edge, 1.0, pal::DIM.a(0.6));

        // the mark: the share among everyone, or (under a random draw) the probability
        let d = n.get(DRAWN);
        let marks = [(n.get(EVERYONE + which), pal::WHITE.a(1.0 - d)), (n.get(TRUE_SHARE + which), TRUTH.a(d * known))];
        let names = self.mark_names(which);
        for (k, (v, color)) in marks.iter().enumerate() {
            let x = x_at(*v);
            plot::line_of(f, BAR_LINES, Vec2::new(x, min.y - 4.0), Vec2::new(x, max.y + 4.0), 2.0, *color);
            let half = 0.5 * f.text_size(&names[k], lay.small).x;
            let x = x.max(min.x + half).min(max.x - half);
            f.text_px(Vec2::new(x, min.y - 5.0), names[k].as_str(), lay.small, *color, Align2::CENTER_BOTTOM);
        }
        let value = self.bar_value(which, &aim.counts);
        f.text_px(Vec2::new(min.x, max.y + 6.0), value, if lay.narrow { lay.small } else { lay.text }, pal::TEXT.a(0.4 + 0.6 * lit), Align2::LEFT_TOP);
    }

    /// The words beside the bars (see [`Chance::reasons`]).
    fn draw_reasons(&self, f: &mut Frame, lay: &Layout) {
        let (Some((min, max)), Some(aim)) = (lay.reasons, &self.aim) else { return };
        let text = self.reasons(&aim.counts);
        hud_panel(f, min - Vec2::splat(6.0), max + Vec2::splat(4.0));
        wrap_px(f, min, max.x - min.x, &text, lay.small, pal::TEXT);
    }
}

/// `kit::wrapped` in screen space: a text broken into lines no wider than `width`
/// logical pixels, its first line's top-left corner at `at`. (`wrapped` measures in the
/// plane's units; in this flat view those are a fixed number of pixels, and a point of
/// the screen is a point of the plane.)
fn wrap_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) {
    let ppu = pixels_per_unit(f).max(1e-6);
    let Some(world) = f.cam.pick_ground(at * f.px) else { return };
    wrapped(f, world.truncate(), width / ppu, text, size, color);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, with the picture at its goal.
    fn run(k: usize) -> Chance {
        let mut s = Chance::create();
        s.set_up(k);
        s.settle();
        s
    }

    /// Probabilities on a grid that takes in the ends.
    fn grid() -> Vec<Probs> {
        let values = [0.0, 0.0004, 0.0005, 0.001, 0.01, 0.09, 0.3, 0.5, 0.9, 0.999, 0.9995, 1.0];
        let mut out = Vec::new();
        for a in values {
            for b_a in values {
                for b_n in values {
                    out.push(Probs { a, b_a, b_n });
                }
            }
        }
        out
    }

    /// The four counts always add up to 1 000, each within rounding of its expected
    /// number; and the draw's add up too.
    #[test]
    fn the_counts_add_up() {
        for p in grid() {
            let c = Counts::expected(&p);
            assert_eq!(c.total(), PEOPLE, "{p:?}");
            assert_eq!(c.a(), (1000.0 * p.a).round() as u32, "{p:?}");
            let j = p.joint();
            for k in 0..4 {
                assert!((c.0[k] as f64 - 1000.0 * j[k]).abs() <= 1.0 + 1e-9, "{p:?}: region {k} has {}, expected {}", c.0[k], 1000.0 * j[k]);
            }
            assert_eq!(Counts::drawn(&p, 3).total(), PEOPLE);
        }
    }

    /// Bayes' rule is counting the regions: the share of the $B$ area that is also $A$;
    /// the odds form gives the same; and the rounded people agree to within a person.
    #[test]
    fn bayes_rule_is_counting() {
        for p in grid() {
            let j = p.joint();
            match p.a_given_b() {
                Some(post) => {
                    let by_area = j[BOTH] / (j[BOTH] + j[B_ONLY]);
                    let by_rule = p.b_a * p.a / p.b();
                    assert!((post - by_area).abs() < 1e-12 && (post - by_rule).abs() < 1e-12, "{p:?}");
                    if let (Some(before), Some(ratio), Some(after)) = (odds(p.a), p.ratio(), odds(post)) {
                        assert!((after - before * ratio).abs() <= 1e-9 * after.max(1.0), "{p:?}: {after} against {}", before * ratio);
                    }
                    let c = Counts::expected(&p);
                    if c.b() > 0 {
                        // each count is within a person of its expected number
                        let (lo, hi) = (((j[BOTH] * 1000.0) - 1.0).max(0.0), j[BOTH] * 1000.0 + 1.0);
                        let (blo, bhi) = (((p.b() * 1000.0) - 2.0).max(1e-9), p.b() * 1000.0 + 2.0);
                        let share = c.0[BOTH] as f64 / c.b() as f64;
                        assert!(share >= lo / bhi - 1e-9 && share <= (hi / blo).min(1.0) + 1e-9, "{p:?}: {share} against {post}");
                    }
                }
                None => assert!(p.b() == 0.0 && Counts::expected(&p).b() == 0, "{p:?}"),
            }
            // the complement and the sum rule
            assert!((p.either() - (p.a + p.b() - j[BOTH])).abs() < 1e-12 && (p.either() - (1.0 - j[NEITHER])).abs() < 1e-12);
            let c = Counts::expected(&p);
            assert_eq!(c.either(), c.a() + c.b() - c.0[BOTH]);
            assert_eq!(c.either(), PEOPLE - c.0[NEITHER]);
        }
    }

    /// Independent exactly when the two levels are equal (with some $A$'s and some not),
    /// which is exactly when the areas multiply.
    #[test]
    fn independent_exactly_when_the_levels_are_equal() {
        for p in grid() {
            let product = (p.joint()[BOTH] - p.a * p.b()).abs() < 1e-12;
            assert_eq!(p.independent(), product, "{p:?}");
            if p.a > 0.0 && p.a < 1.0 {
                assert_eq!(p.independent(), p.b_a == p.b_n, "{p:?}");
                if p.independent() {
                    assert!((p.b_given_a().unwrap_or(-1.0) - p.b()).abs() < 1e-12);
                    if let Some(q) = p.a_given_b() {
                        assert!((q - p.a).abs() < 1e-12, "{p:?}");
                    }
                }
            }
        }
    }

    /// What the "things to try" say, held to the scene as each sets it up.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // a rare condition: 10 have it, 9 of them positive; 89 of the 990 positive too;
        // 9 of 98 positives, 9.2 %, though 90 % of those who have it test positive
        let rare = run(0);
        let c = rare.aim.as_ref().map(|a| a.counts).unwrap_or(Counts([0; 4]));
        assert_eq!(c.0, [9, 1, 89, 901]);
        assert_eq!((c.a(), c.not_a(), c.b()), (10, 990, 98));
        assert_eq!(pct(rare.truth.a_given_b().unwrap_or(0.0)), "9.2");
        assert_eq!(pct(rare.truth.b_a), "90");
        assert!(rare.truth.a_given_b().unwrap_or(1.0) < 0.5, "most positives are false alarms");
        assert!((rare.story, rare.focus, rare.drawn) == (Story::Test, Focus::B, false));
        assert!(TRIALS[0].caption.contains("9 of the 98") && TRIALS[0].caption.contains(r"9.2\%") && TRIALS[0].caption.contains("89 of those"));

        // a common condition: 270 of 333 positives, 81 %
        let common = run(1);
        let c = Counts::expected(&common.truth);
        assert_eq!((c.0[BOTH], c.b()), (270, 333));
        assert_eq!(format!("{:.0}", 100.0 * common.truth.a_given_b().unwrap_or(0.0)), "81");
        assert!(TRIALS[1].caption.contains("270 of the 333") && TRIALS[1].caption.contains(r"81\%"));

        // test the positives again: the base rate is now 9.2 %; after a second positive
        // about 50 %; odds 1 : 99, times 10 each time, 10 : 99 and 100 : 99
        let again = run(2);
        assert_eq!(pct(again.truth.a), "9.2");
        let post = again.truth.a_given_b().unwrap_or(0.0);
        assert_eq!(format!("{:.0}", 100.0 * post), "50");
        assert!((again.truth.ratio().unwrap_or(0.0) - 10.0).abs() < 1e-9);
        assert!((odds(RARE.a).unwrap_or(0.0) - 1.0 / 99.0).abs() < 1e-12);
        assert!((odds(again.truth.a).unwrap_or(0.0) - 10.0 / 99.0).abs() < 1e-9);
        assert!((odds(post).unwrap_or(0.0) - 100.0 / 99.0).abs() < 1e-9);
        assert_eq!(odds_text(odds(RARE.a).unwrap_or(0.0)), "$1 : 99$");
        assert_eq!(odds_text(odds(again.truth.a).unwrap_or(0.0)), "$1 : 9.9$");
        assert_eq!(odds_text(odds(post).unwrap_or(0.0)), "$1.01 : 1$");
        assert!(again.reasons(&Counts::expected(&again.truth)).contains("$1 : 9.9$ at first") && again.reasons(&Counts::expected(&again.truth)).contains("to $1.01 : 1$"));

        // independent: 30 % on both sides; P(B | A) = P(B) = 30 %; 0.4 × 0.3 = 0.12, 120
        // of 1 000; P(A | B) = P(A) = 40 %
        let ind = run(3);
        let t = ind.truth;
        assert!(t.independent() && t.b_a == t.b_n && pct(t.b()) == "30" && pct(t.b_a) == "30");
        assert_eq!(Counts::expected(&t).0[BOTH], 120);
        assert_eq!(dec(t.joint()[BOTH]), "0.12");
        assert_eq!(pct(t.a_given_b().unwrap_or(0.0)), "40");
        assert!(ind.reasons(&Counts::expected(&t)).contains(r"= 0.4 \times 0.3 = 0.12"));
        // and one level moved off the other: none of it holds
        let off = Probs { b_n: 0.31, ..t };
        assert!(!off.independent() && (off.joint()[BOTH] - off.a * off.b()).abs() > 1e-4);

        // either or both: 300 + 200 + 100 = 600; 0.5 + 0.4 - 0.3 = 0.6; 400 in neither
        let either = run(4);
        let c = Counts::expected(&either.truth);
        assert_eq!(c.0, [300, 200, 100, 400]);
        assert_eq!(c.either(), 600);
        let t = either.truth;
        assert_eq!([dec(t.a), dec(t.b()), dec(t.joint()[BOTH]), dec(t.either())], ["0.5", "0.4", "0.3", "0.6"]);
        assert!(either.reasons(&c).contains("$500 + 400 - 300 = 600$") && either.reasons(&c).contains(r"$1\,000 - 400 = 600$"), "{}", either.reasons(&c));
    }

    /// The numbers a random draw gives wobble round the expected ones as the binomial
    /// distribution says: over many draws, their means and spreads are the binomial
    /// ones; and the first draw shown is an ordinary one.
    #[test]
    fn a_random_draw_wobbles_as_the_binomial_says() {
        let runs = 2000;
        let (mut a, mut b, mut both) = (Vec::with_capacity(runs), Vec::with_capacity(runs), Vec::with_capacity(runs));
        for seed in 0..runs as u64 {
            let c = Counts::drawn(&RARE, seed);
            a.push(c.a() as f64);
            b.push(c.b() as f64);
            both.push(c.0[BOTH] as f64);
        }
        use crate::math::stats::{mean, variance, Dist};
        for (v, p) in [(&a, RARE.a), (&b, RARE.b()), (&both, RARE.joint()[BOTH])] {
            let bin = Dist::Binomial { n: PEOPLE, p };
            let (m, var) = (bin.mean().unwrap_or(0.0), bin.variance().unwrap_or(0.0));
            let se = (var / runs as f64).sqrt();
            assert!((mean(v) - m).abs() < 2.0 * se, "mean {} against {m}", mean(v));
            assert!((variance(v) / var - 1.0).abs() < 0.06, "variance {} against {var}", variance(v));
        }
        // the first draw: inside the middle 99.8 % of each count's binomial
        let c = Counts::drawn(&RARE, FIRST_SEED);
        for (k, p) in [(c.a(), RARE.a), (c.b(), RARE.b())] {
            let bin = Dist::Binomial { n: PEOPLE, p };
            assert!((k as f64) >= bin.quantile(0.001) && (k as f64) <= bin.quantile(0.999), "{k} for p = {p}");
        }
        // the same seed, the same people: more of A takes in more people, never fewer
        let mut last = 0;
        for k in 0..=100 {
            let n = Counts::drawn(&Probs { a: k as f64 / 100.0, ..RARE }, 11).a();
            assert!(n >= last);
            last = n;
        }
        assert_eq!(last, PEOPLE);
    }

    /// Every region holds as many people as it counts, all inside it and evenly spread;
    /// one person more takes one more place and moves no one; and a small move of the
    /// divider changes the places of only a few.
    #[test]
    fn the_people_stand_in_their_regions() {
        let mut taken = Vec::new();
        for p in grid().into_iter().step_by(3) {
            for drawn in [false, true] {
                let c = if drawn { Counts::drawn(&p, 5) } else { Counts::expected(&p) };
                let s = split(&p, &c, drawn);
                let (a, b_a, b_n) = (s.a as f32, s.b_a as f32, s.b_n as f32);
                let extra = crowd(a, b_a, b_n, c.0, &mut taken);
                assert_eq!(extra, [0; 4], "{p:?} {drawn}: some had no place");
                let mut seen = [0u32; 4];
                for (&[x, y], &(k, here)) in seats().iter().zip(&taken) {
                    assert_eq!(k as usize, region_at(x, y, a, b_a, b_n));
                    seen[k as usize] += u32::from(here);
                }
                assert_eq!(seen, c.0, "{p:?} {drawn}");
            }
        }
        // evenly: in the opening picture, no two people closer than half the spacing of
        // a square grid of 1 000, and every quarter of the false alarms' band holds a
        // quarter of them, near enough
        let (a, b_a, b_n) = (RARE.a as f32, RARE.b_a as f32, RARE.b_n as f32);
        let c = Counts::expected(&RARE);
        crowd(a, b_a, b_n, c.0, &mut taken);
        let people: Vec<[f32; 2]> = seats().iter().zip(&taken).filter(|(_, t)| t.1).map(|(p, _)| *p).collect();
        assert_eq!(people.len(), PEOPLE as usize);
        let pitch = 1.0 / (PEOPLE as f32).sqrt();
        for i in 0..people.len() {
            for j in 0..i {
                let d = ((people[i][0] - people[j][0]).powi(2) + (people[i][1] - people[j][1]).powi(2)).sqrt();
                assert!(d > 0.5 * pitch, "{:?} and {:?}", people[i], people[j]);
            }
        }
        let alarms: Vec<f32> = seats().iter().zip(&taken).filter(|(_, t)| t.1 && t.0 as usize == B_ONLY).map(|(p, _)| p[0]).collect();
        for q in 0..4 {
            let lo = a + (1.0 - a) * q as f32 / 4.0;
            let inside = alarms.iter().filter(|x| **x >= lo && **x < lo + (1.0 - a) / 4.0).count();
            assert!((inside as f32 - 89.0 / 4.0).abs() <= 4.0, "quarter {q}: {inside} of 89");
        }
        // one more positive among those who have it, one fewer negative: one place
        // taken among the positives, one left among the negatives, no other change
        let before = taken.clone();
        crowd(a, b_a, b_n, [10, 0, 89, 901], &mut taken);
        let changed: Vec<((u8, bool), (u8, bool))> = before.iter().zip(&taken).filter(|(x, y)| x != y).map(|(x, y)| (*x, *y)).collect();
        assert_eq!(changed.len(), 2, "{changed:?}");
        assert!(changed.contains(&((A_ONLY as u8, true), (A_ONLY as u8, false))) && changed.contains(&((BOTH as u8, false), (BOTH as u8, true))), "{changed:?}");
        // the divider moved by a tenth of a per cent (one person), anywhere along the
        // square: at most six places change hands (a place that changes region with no
        // one standing in it shows nothing)
        let mut after = Vec::new();
        for k in (1..999).step_by(7) {
            for (j, t) in [(k, &mut taken), (k + 1, &mut after)] {
                let p = Probs { a: j as f64 / 1000.0, ..RARE };
                crowd(p.a as f32, b_a, b_n, Counts::expected(&p).0, t);
            }
            let changed = taken.iter().zip(&after).filter(|(x, y)| x != y && (x.1 || y.1)).count();
            assert!(changed <= 6, "from {k} to {} in A: {changed} places changed", k + 1);
        }
    }

    /// No one in $A$, everyone in $A$, a test never or always positive, no false alarms:
    /// every number drawn is finite, and the words say what has happened.
    #[test]
    fn degenerate_settings_are_told_honestly() {
        let ends = [0.0, 0.0005, 0.01, 0.5, 0.999, 1.0];
        let mut corners = Vec::new();
        for a in ends {
            for b_a in ends {
                for b_n in ends {
                    corners.push(Probs { a, b_a, b_n });
                }
            }
        }
        for p in corners {
            for drawn in [false, true] {
                for focus in FOCI {
                    let mut s = Chance::create();
                    (s.truth, s.drawn, s.focus) = (p, drawn, focus);
                    s.settle();
                    let aim = s.aim.as_ref().expect("aimed");
                    assert!(aim.numbers.iter().chain(&s.shown).all(|v| v.is_finite()) && s.extra == [0; 4], "{p:?}");
                    let texts = [s.reasons(&aim.counts), s.bar_value(0, &aim.counts), s.bar_value(1, &aim.counts), s.title()];
                    for t in texts.iter().chain(s.mark_names(0).iter()).chain(s.mark_names(1).iter()) {
                        assert!(!t.contains('?') && !t.contains("NaN") && !t.contains("inf"), "{p:?}: {t}");
                    }
                }
            }
        }
        let say = |p: Probs| {
            let mut s = Chance::create();
            s.truth = p;
            s.settle();
            let c = Counts::expected(&p);
            (s.reasons(&c), s.bar_value(0, &c), s.bar_value(1, &c))
        };
        let (_, no_a, _) = say(Probs { a: 0.0, ..RARE });
        assert!(no_a.contains("No one has it") && no_a.contains("undefined"), "{no_a}");
        let (never, _, no_b) = say(Probs { b_a: 0.0, b_n: 0.0, ..RARE });
        assert!(no_b.contains("undefined") && never.contains("No one tests positive"), "{never} / {no_b}");
        let (always, _, _) = say(Probs { b_a: 1.0, b_n: 1.0, ..RARE });
        assert!(always.contains("multiplies them by the likelihood ratio, $1 / 1 = 1$") && always.contains("independent"), "{always}");
        let (sure, _, _) = say(Probs { b_n: 0.0, ..RARE });
        assert!(sure.contains(r"P(A \mid B) = 100\%"), "{sure}");
        let (all, _, _) = say(Probs { a: 1.0, ..RARE });
        assert!(all.contains("Everyone has it"), "{all}");
        // never 100 % for what is not certain, nor 0 % for what can happen
        assert_eq!(pct(0.99999), "99.999");
        assert_eq!(pct(0.000004), "0.0004");
        assert_eq!(pct(1.0), "100");
        assert_eq!(dec(0.0917431), "0.0917");
    }

    /// The layout of every window: finite, inside the canvas, the square square, and
    /// nothing on top of anything else.
    #[test]
    fn the_layout_fits_every_window() {
        for (w, h, inset) in [(1075.0, 1000.0, 0.0), (1400.0, 760.0, 0.0), (876.0, 700.0, 0.0), (390.0, 756.0, 44.0), (700.0, 500.0, 0.0), (700.0, 500.0, 44.0), (390.0, 300.0, 44.0), (240.0, 180.0, 0.0), (1.0, 1.0, 0.0), (3000.0, 2000.0, 0.0)] {
            let lay = Layout::of(Vec2::new(w, h), inset);
            let mut all: Vec<Rect> = vec![lay.square, lay.table, lay.again];
            all.extend(lay.stories.iter().chain(&lay.modes).chain(&lay.blocks).chain(&lay.bars).chain(&lay.looks));
            assert!(all.iter().all(|(a, b)| a.is_finite() && b.is_finite()), "{w}x{h}");
            let (min, max) = lay.square;
            assert!(((max.x - min.x) - (max.y - min.y)).abs() < 1e-3, "{w}x{h}: not square");
            if lay.compact {
                // the square and its names on the canvas, or the square alone
                let room = if lay.bare { (min, max) } else { (min - Vec2::new(lay.margin[0], lay.head), max + Vec2::new(lay.margin[1], STEM + lay.small + 12.0)) };
                assert!(lay.tiny || (room.0.x >= 0.0 && room.0.y >= inset && room.1.x <= w + 0.5 && room.1.y <= h + 0.5), "{w}x{h}: {room:?}");
                continue;
            }
            assert!(w >= 390.0, "{w}x{h} is not compact");
            let inside = |(a, b): Rect| a.x >= -0.5 && a.y >= inset - 0.5 && b.x <= w + 0.5 && b.y <= h + 0.5;
            assert!(all.iter().all(|r| inside(*r)), "{w}x{h}");
            assert!(max.x - min.x >= 250.0, "{w}x{h}: the square is {}", max.x - min.x);
            // the rows of buttons stay on the canvas, and the square clear of everything else
            let overlap = |(a, b): Rect, (c, d): Rect| a.x < d.x && c.x < b.x && a.y < d.y && c.y < b.y;
            let others: Vec<Rect> = [lay.table, lay.again].into_iter().chain(lay.stories).chain(lay.modes).chain(lay.blocks).chain(lay.looks).collect();
            let around = (min - Vec2::new(lay.margin[0], lay.head), max + Vec2::new(lay.margin[1], STEM + lay.small + 12.0));
            for r in &others {
                assert!(!overlap(around, *r), "{w}x{h}: {r:?} over the square");
            }
            if let Some(r) = lay.reasons {
                assert!(r.1.y > r.0.y && !overlap(around, r));
            }
        }
    }

    /// After a change, the picture glides to the new one and comes to rest exactly
    /// there, at any frame rate.
    #[test]
    fn glides_arrive_and_stop() {
        for hz in [30.0f32, 60.0, 144.0] {
            let mut s = run(0);
            s.set_up(1);
            s.aim();
            let mut frames = 0;
            loop {
                let goal = s.aim.as_ref().map(|aim| aim.numbers.clone()).unwrap_or_default();
                let a = s.numbers.follow(&goal, 1.0 / hz, false);
                let b = s.seat(1.0 / hz, false);
                frames += 1;
                if !(a || b) {
                    assert!(s.numbers.shown == goal && s.shown.iter().zip(&s.taken).all(|(v, t)| *v == if t.1 { 1.0 } else { 0.0 }));
                    break;
                }
                assert!((frames as f32) < 2.0 * hz, "still gliding after two seconds at {hz} Hz");
            }
            assert!(frames as f32 > 0.15 * hz, "arrived in {frames} frames at {hz} Hz: no glide");
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            let c = Counts::expected(&s.truth);
            println!("{}: {:?} counts {:?}, P(A|B) {:?}, P(B) {}, either {}", TRIALS[k].button, s.truth, c.0, s.truth.a_given_b(), s.truth.b(), s.truth.either());
            println!("    {}", s.reasons(&c));
            println!("    {} / {}", s.bar_value(0, &c), s.bar_value(1, &c));
        }
        for seed in FIRST_SEED..FIRST_SEED + 5 {
            println!("draw {seed}: {:?}", Counts::drawn(&RARE, seed).0);
        }
        // the spread of many draws against the binomial's
        use crate::math::stats::{mean, variance};
        let draws: Vec<Counts> = (0..2000u64).map(|seed| Counts::drawn(&RARE, seed)).collect();
        for (name, v, p) in [("A", draws.iter().map(|c| c.a() as f64).collect::<Vec<_>>(), RARE.a), ("B", draws.iter().map(|c| c.b() as f64).collect(), RARE.b())] {
            println!("{name}: mean {:.3} against {:.3}, variance {:.3} against {:.3}", mean(&v), 1000.0 * p, variance(&v), 1000.0 * p * (1.0 - p));
        }
        // how many people change places when the divider moves by one person, over the range
        let (mut most, mut taken, mut before) = (0, Vec::new(), Vec::new());
        for k in 1..999 {
            for (j, t) in [(k, &mut before), (k + 1, &mut taken)] {
                let p = Probs { a: j as f64 / 1000.0, ..RARE };
                crowd(p.a as f32, p.b_a as f32, p.b_n as f32, Counts::expected(&p).0, t);
            }
            most = most.max(before.iter().zip(&taken).filter(|(x, y)| x != y && (x.1 || y.1)).count());
        }
        println!("most places changed by one person more in A: {most}");
    }
}

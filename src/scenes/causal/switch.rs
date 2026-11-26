//! Two vaccines, one switch: a close look at Taquet and others (Nature Medicine, 2024),
//! who compared people first vaccinated against shingles in the three years before and
//! the three years after the United States moved from the live vaccine to the
//! recombinant one, and found fewer dementia diagnoses after. What that comparison
//! assumes, what the paper found and checked, and a toy model of the mechanisms other
//! than the vaccine that could, or could not, give the same pattern.
//!
//! Every fact about the paper comes from the dossier compiled on 25 November 2026, and
//! every number shown is in one place here (the constants under "what the paper
//! reports"), each with the dossier's section and the paper's page. Everything the toy
//! model draws is made up, and the picture says so wherever it is drawn.

use super::*;
use std::collections::HashMap;

pub const DEF: SceneDef = SceneDef {
    id: "switch",
    title: "Two vaccines, one switch",
    group: crate::scenes::CASE_STUDIES,
    blurb: "Taquet and others (2024) compared people vaccinated before and after the United States changed shingles vaccine: what that comparison assumes, and what else could give its result.",
    make: |_| Box::new(Switch::new()),
};

// ----- dates --------------------------------------------------------------------------

/// The first day of month `m` (1 to 12) of `year`, as a decimal year.
const fn month(year: i32, m: i32) -> f64 {
    year as f64 + (m - 1) as f64 / 12.0
}

/// A window of first doses: the first day of its first month, and how many months.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Window {
    from: f64,
    months: usize,
}

impl Window {
    fn to(self) -> f64 {
        self.from + self.months as f64 / 12.0
    }
}

/// The cohorts [Methods, PDF p. 6] (dossier §1.4): first dose from 1 October 2014 to
/// 30 September 2017, and from 1 November 2017 to 31 October 2020. October 2017, the
/// month of the switch, is left out.
const EARLIER: Window = Window { from: month(2014, 10), months: 36 };
const LATER: Window = Window { from: month(2017, 11), months: 36 };
/// The six-month windows of the secondary analysis, "Apr 17–Sep 17" and "Nov 17–Apr 18"
/// [Suppl. Table 6, Suppl. p. 12; Extended Data Fig. 3F] (dossier §1.4).
const EARLIER_SIX: Window = Window { from: month(2017, 4), months: 6 };
const LATER_SIX: Window = Window { from: month(2017, 11), months: 6 };
/// The month the pandemic is taken to start (dossier T2).
const PANDEMIC: f64 = month(2020, 3);
/// The United States moved from ICD-9-CM to ICD-10-CM diagnosis codes on 1 October 2015
/// (CMS/CDC; dossier T1).
const ICD10: f64 = month(2015, 10);
/// The paper does not state when its data end (dossier §1.5: "not verified"); its
/// reporting summary was last updated on 13 March 2024 [PDF p. 14]. The picture draws
/// the end there, saying so, and the toy model assumes it.
const DATA_END: f64 = month(2024, 3);

// ----- what the paper reports (the dossier's values) ------------------------------------

/// A published estimate and its 95 % interval.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Figure {
    value: f64,
    low: f64,
    high: f64,
}

const fn fig(value: f64, low: f64, high: f64) -> Figure {
    Figure { value, low, high }
}

impl Figure {
    /// "0.83 (0.79–0.87)", to `decimals` places.
    fn text(self, decimals: usize) -> String {
        format!("{} ({}–{})", num(self.value, decimals), num(self.low, decimals), num(self.high, decimals))
    }
}

/// Table 1 of the paper [p. 2779] (dossier §1.9): ratios of restricted mean time lost,
/// later cohort over earlier, with 95 % intervals, each with the name the forest gives it.
const TABLE1: [(&str, Figure); 13] = [
    ("primary analysis", fig(0.83, 0.79, 0.87)),
    ("women", fig(0.78, 0.73, 0.83)),
    ("men", fig(0.87, 0.81, 0.94)),
    ("six-month windows", fig(0.83, 0.76, 0.92)),
    ("aligned follow-up", fig(0.83, 0.79, 0.87)),
    ("predominant vaccine only", fig(0.82, 0.79, 0.86)),
    ("excluding those given both", fig(0.79, 0.74, 0.83)),
    ("matched on deprivation too", fig(0.84, 0.80, 0.88)),
    ("coarsened exact matching", fig(0.82, 0.77, 0.87)),
    ("dementia or death", fig(0.93, 0.91, 0.96)),
    ("mortality", fig(0.98, 0.95, 1.01)),
    ("negative-control outcome", fig(0.97, 0.91, 1.03)),
    ("shingles (positive control)", fig(0.65, 0.61, 0.69)),
];

/// Where a row of `TABLE1` is.
const PRIMARY: usize = 0;
const SIX_MONTHS: usize = 3;
const CEM: usize = 8;
const MORTALITY: usize = 10;
const NEGATIVE: usize = 11;
const SHINGLES: usize = 12;
/// The rows a narrow canvas has room for.
const TABLE1_SHORT: [usize; 8] = [0, 1, 2, 3, 9, 10, 11, 12];

/// The analysis with follow-up entirely before the pandemic (six-month windows, follow-up
/// to 18 months): a hazard ratio, not a ratio of time lost [p. 2779; Methods, PDF p. 7]
/// (dossier §1.9, §1.10 row 4).
const BEFORE_PANDEMIC_HR: Figure = fig(0.74, 0.62, 0.90);
/// "164 (95% CI 124–205) additional diagnosis-free days among those affected" [Table 1,
/// p. 2779] (dossier §1.9; the text on p. 2779 prints 124–202, dossier §1.13).
const DAYS_AMONG_AFFECTED: Figure = fig(164.0, 124.0, 205.0);
/// From the authors' posted data [OSF] (dossier §1.8): the difference in restricted mean
/// survival time, in days, and the unrounded ratio of time lost.
const RMST_DIFFERENCE_DAYS: Figure = fig(9.75, 7.35, 12.15);
const RMTL_RATIO_POSTED: f64 = 0.8319;
/// The window, in days after the first dose, in the posted curves [OSF] (dossier §1.5).
const WINDOW_DAYS: (f64, f64) = (90.0, 2191.0);
/// Median (interquartile range) follow-up in years [p. 2779] (dossier §1.5).
const MEDIAN_LATER: Figure = fig(4.15, 3.16, 4.99);
const MEDIAN_EARLIER: Figure = fig(6.0, 5.2, 6.0);
/// Shares given each vaccine, in percent [p. 2779] (dossier §1.4).
const LIVE_BEFORE: f64 = 98.0;
const RECOMBINANT_AFTER: f64 = 95.0;
/// People in each matched cohort, and the pools they were drawn from [Suppl. Table 1,
/// Suppl. p. 5] (dossier §1.6, §1.4); the health-care organisations in the network
/// [Methods, PDF p. 6] (dossier §1.3); the covariates matched on [Methods, PDF p. 6].
const MATCHED: usize = 103_837;
const POOL_LATER: usize = 386_023;
const POOL_EARLIER: usize = 103_958;
const ORGANISATIONS: usize = 62;
const COVARIATES: usize = 60;
/// Women against men: 22 % against 13 % more time diagnosis-free, permutation test
/// P = 0.017 [p. 2779] (dossier §1.9).
const MORE_TIME_WOMEN_MEN: (f64, f64, f64) = (22.0, 13.0, 0.017);
/// The same group's 2026 cardiovascular paper ran a placebo switch with Tdap recipients:
/// RMTL ratio 1.01 (0.99–1.03) (dossier §1.14).
const CARDIO_PLACEBO: Figure = fig(1.01, 0.99, 1.03);

/// Cumulative incidence, in percent, after (later) and before (earlier) the switch, at
/// whole years after the first dose, from the authors' posted data [OSF] (dossier §1.9).
struct Marks {
    years: &'static [f64],
    later: &'static [f64],
    earlier: &'static [f64],
}

const DEMENTIA_MARKS: Marks = Marks { years: &[1.0, 2.0, 3.0, 5.0, 6.0], later: &[0.46, 1.12, 1.97, 4.06, 5.75], earlier: &[0.63, 1.57, 2.55, 4.70, 5.94] };
const DEATH_MARKS: Marks = Marks { years: &[3.0, 6.0], later: &[3.72, 9.85], earlier: &[3.70, 10.45] };
const SHINGLES_MARKS: Marks = Marks { years: &[2.0, 3.0, 6.0], later: &[0.72, 1.12, 2.51], earlier: &[1.15, 1.81, 3.61] };

/// Person-years, cases and people lost to follow-up in each year after the first dose,
/// earlier cohort then later [Suppl. Table 2, Suppl. p. 8 (PDF p. 9)] (dossier §1.5).
const SUPPL_TABLE_2: [[(f64, f64, f64); 2]; 6] = [
    [(101_849.0, 636.0, 3_304.0), (100_576.0, 459.0, 5_956.0)],
    [(97_872.0, 932.0, 3_072.0), (94_437.0, 625.0, 5_463.0)],
    [(93_639.0, 940.0, 3_696.0), (86_882.0, 752.0, 9_321.0)],
    [(88_581.0, 954.0, 4_488.0), (69_162.0, 671.0, 23_867.0)],
    [(82_731.0, 947.0, 5_371.0), (40_394.0, 470.0, 30_524.0)],
    [(75_421.0, 988.0, 7_860.0), (11_815.0, 180.0, 24_557.0)],
];

/// The crude rate ratios the dossier calculates from that table (§1.5), to two places:
/// what the picture writes, held against the arithmetic by a test.
const CRUDE_RATIOS: [f64; 6] = [0.73, 0.69, 0.86, 0.90, 1.02, 1.16];

/// Where the pandemic falls in each cohort's follow-up, in years after the first dose,
/// as the dossier calculates it from the cohort dates (T2): from the last of the earlier
/// cohort to the first, and from the first of the later cohort to the last (negative:
/// before the first dose).
const PANDEMIC_EARLIER: (f64, f64) = (2.5, 5.4);
const PANDEMIC_LATER: (f64, f64) = (2.3, -0.6);

/// The paper's own account of the time course [p. 2779], quoted (dossier §1.9).
const TIME_COURSE: &str = "In the paper's words: “The time-varying HRs became significantly lower than 1 within the first year of follow-up and then progressively approached (and, in some but not all robustness analyses, exceeded) 1 towards the end the 6-year follow-up” (page 2779).";

/// The sources line every view carries, long and short.
const SOURCES: &str = "Sources as read on 25 November 2026: Taquet, Dercon, Todd and Harrison, Nature Medicine 30, 2777–2781 (2024), doi:10.1038/s41591-024-03201-5, with its supplement and the authors' posted data.";
const SOURCES_SHORT: &str = "Sources as read on 25 November 2026: Taquet and others, Nat Med 2024, doi:10.1038/s41591-024-03201-5";

/// What the authors themselves give as limits [p. 2780; Suppl. Note 1, Suppl. p. 2]
/// (dossier §1.11), and the case they make from their checks [p. 2780] (dossier §1.12),
/// quoted.
const LIMITS: [&str; 10] = [
    "“This study is observational, and causality cannot be demonstrated.” (page 2780)",
    "“This study has several limitations in addition to those inherent to studies based on EHR data (such as no validation of diagnoses and sparse information on socioeconomic and lifestyle factors; see Supplementary Note 1).” (page 2780)",
    "“First, being diagnosis-free does not imply being disease-free, because there can be delays in diagnosis. However, if it is assumed that diagnostic delays are similar between cohorts, then differences in disease-free time will follow differences in diagnosis-free time.” (page 2780)",
    "“Second, we did not investigate the impact of multiple vaccine doses.” (page 2780)",
    "“Third, the number of people who received a shingles vaccine increased between before and after the introduction of the recombinant vaccine, justifying the need for additional control of covariates (as achieved here using matching). However, the fact that the association was maintained when the exposure window was reduced to 6 months on either side of the step change in recombinant-vaccine uptake argues strongly against the possibility that drifts in population characteristics could explain the main findings.” (page 2780)",
    "“Fourth, the paired nature of the data was not accounted for in the estimation of CIs within the primary analysis ... when this was accounted for in the secondary analysis using coarsened exact matching, little difference was observed in the estimated confidence intervals.” (page 2780)",
    "“The observation that the time-varying HR became greater than 1 towards the end of the follow-up could imply that the vaccine delays, rather than prevents, dementia onset. However, this was not robustly observed across analyses (Extended Data Fig. 5) and therefore requires replication.” (page 2780)",
    "“Despite the matching and use of various comparison cohorts, there may well be residual confounding, particularly related to social and economic factors which are not well captured in EHR networks and which might influence outcomes post-vaccination as well as choice of zoster vaccine.” (Supplementary Note 1, page 2)",
    "“A patient may be seen in different HCOs for different parts of their care, and if one HCO is not part of the federated network then part of their medical records may not be available.” (Supplementary Note 1, page 2)",
    "“Historical data before the start of EHRs (or the addition of an HCO to the network) may well be incomplete.” (Supplementary Note 1, page 2)",
];
const THEIR_CASE: &str = "Their case from the checks: “The observation that all-cause mortality was very similar between cohorts, the lack of association with a composite negative control outcome and the robustness of the findings across several secondary analyses support the absence of obvious residual confounding.” (page 2780)";

/// What would decide the open points (the dossier's summary table in §1.15, and §5), as
/// questions.
const QUESTIONS: [&str; 6] = [
    "Calendar time: does the same before-and-after comparison, among people given a vaccine that did not change in 2017 (Tdap or influenza), find no difference in dementia? Do the cohorts still differ when compared in the same calendar months?",
    "Early timing and selection: did the cohorts differ before vaccination, in health-care use and cognitive complaints? Does the difference survive when the first years of follow-up are left out?",
    "Records and look-back: does the result hold among people with long, comparable records before vaccination, matched also on their number of visits?",
    "Leaving the records: who left early in each cohort (1.8 times as many in the later one in the first two years, calculated from Supplementary Table 2), and does the result survive weighting for it, or bounds that assume they were at higher risk?",
    "Crossing over: what does an analysis that stops following people when they get the other vaccine, weighted for who stops, give?",
    "Randomised trials of the recombinant vaccine with dementia as a primary outcome began in 2026, in Denmark (against no vaccine, open label) and Finland (against placebo, blinded); neither has reported, and neither compares it with the live vaccine.",
];

// ----- the toy model ------------------------------------------------------------------
//
// Two cohorts, each a mix of monthly groups of first doses spread evenly over the
// cohort's window, followed from the first dose in half-month steps until the end of
// the data. Every group has, in each step, a rate of recorded dementia diagnosis and a
// rate of leaving the records. Nothing is sampled: what is computed is what the analysis
// would give with endlessly many people, so whatever changes in the picture is the
// mechanism's doing, never chance.
//
// The rate of diagnosis of someone vaccinated at calendar time v, at time t after the
// first dose (in follow-up year k, from 0), is
//
//     base(k) × vaccine(k) × adopters × pandemic(v + t) × drift(v + t)
//
// of which a share `leaving` (in the later cohort) leave the records before the
// diagnosis is made; and a further share of people have dementia already, unrecorded,
// when vaccinated, and are diagnosed at the rate 1 a year (times the same calendar
// factors). Each factor is 1 unless its mechanism is on. The analysis is the paper's:
// the cumulative incidence from three months to six years (Kaplan–Meier, in the limit
// of many people), the ratio of restricted mean time lost over that window, and the
// crude rate ratio in each year of follow-up.

/// Steps per year of follow-up (half a month each), and the steps in a month.
const PER_YEAR: usize = 24;
const MONTH_STEPS: usize = PER_YEAR / 12;
const DT: f64 = 1.0 / PER_YEAR as f64;
/// Six years of follow-up, counted from three months.
const YEARS: usize = 6;
const STEPS: usize = YEARS * PER_YEAR;
const COUNT_FROM: usize = PER_YEAR / 4;
/// The points of a curve from three months to six years.
const CURVE: usize = STEPS - COUNT_FROM + 1;
/// The before-the-pandemic check follows for 18 months.
const SHORT: usize = PER_YEAR * 3 / 2;
/// The calendar the toy's factors are laid out on: from the first month of the earlier
/// cohort, a step at a time, past the end of the data.
const CAL0: f64 = month(2014, 10);
const CAL_STEPS: usize = 240;

/// The toy's baseline: the rate of recorded diagnosis rises by this factor each year of
/// follow-up (with age), and is set so that, with nothing switched on, this share is
/// diagnosed by six years (about the paper's earlier cohort, 5.94 %).
const RISE: f64 = 1.12;
const SIX_YEAR_SHARE: f64 = 0.06;
/// The toy pandemic: diagnoses fewer by the strength set, from March 2020 for two years.
const PANDEMIC_END: f64 = month(2022, 3);
/// Drift is measured from the switch.
const DRIFT_FROM: f64 = month(2017, 11);
/// Dementia already present at the first dose is recorded at this rate (a year).
const RECORDED: f64 = 1.0;
/// The share with dementia already present falls steadily from the strength set, among
/// those vaccinated in October 2014, to none among those vaccinated in October 2020.
const PRESENT_TO: f64 = month(2020, 11);
/// A delay holds off diagnoses for the first three years.
const HELD_YEARS: usize = 3;
/// The other outcomes of the checks: deaths (rising by DEATH_RISE a year, 10 % by six
/// years), the painful conditions of the negative control (2 % by six years) and
/// shingles (3.6 % by six years, its rate times SHINGLES_HR with the newer vaccine).
const DEATH_RISE: f64 = 1.10;
const DEATH_SHARE: f64 = 0.10;
const PAINFUL_SHARE: f64 = 0.02;
const SHINGLES_SHARE: f64 = 0.036;
const SHINGLES_HR: f64 = 0.6;

/// The mechanisms of the toy model, each with its strength (`Toy::OFF` has them off).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Toy {
    /// The newer vaccine's true hazard ratio on dementia: 1 for no effect.
    theta: f64,
    /// A delay: the share of the diagnoses the newer vaccine holds off in the first three
    /// years that are made in years four to six instead.
    delay: f64,
    /// The pandemic: diagnoses fewer by this share from March 2020 to March 2022.
    pandemic: f64,
    /// Drift: diagnosis more ready by this much each calendar year (a rate: the factor
    /// is e^(drift × years from the switch)).
    drift: f64,
    /// Healthier early adopters: the later cohort's risk lower by this share ...
    adopters: f64,
    /// ... in general health (deaths and the other illnesses too), or in cognition only.
    general: bool,
    /// Leaving the records: in the later cohort, this share of those on the way to a
    /// diagnosis leave the records before it is made.
    leaving: f64,
    /// Already present: the share, among those vaccinated in October 2014, who have
    /// dementia not yet in their records (fewer among those vaccinated later).
    present: f64,
}

impl Toy {
    const OFF: Toy = Toy { theta: 1.0, delay: 0.0, pandemic: 0.0, drift: 0.0, adopters: 0.0, general: false, leaving: 0.0, present: 0.0 };

    /// The same true effect of the vaccine, and nothing else.
    fn vaccine_only(self) -> Toy {
        Toy { theta: self.theta, delay: self.delay, ..Toy::OFF }
    }

    /// The strengths kept in their ranges (anything not a number taken as off).
    fn tidy(self) -> Toy {
        let within = |v: f64, lo: f64, hi: f64, off: f64| if v.is_finite() { v.max(lo).min(hi) } else { off };
        Toy {
            theta: within(self.theta, THETA_LEAST, 1.0, 1.0),
            delay: within(self.delay, 0.0, 1.0, 0.0),
            pandemic: within(self.pandemic, 0.0, PANDEMIC_MOST, 0.0),
            drift: within(self.drift, -DRIFT_MOST, DRIFT_MOST, 0.0),
            adopters: within(self.adopters, 0.0, ADOPTERS_MOST, 0.0),
            general: self.general,
            leaving: within(self.leaving, 0.0, LEAVING_MOST, 0.0),
            present: within(self.present, 0.0, PRESENT_MOST, 0.0),
        }
    }

    /// What of it can change what `check` finds (the rest is set to off, so that two
    /// toys that differ only there need not be compared again).
    fn relevant(self, check: Check) -> Toy {
        let t = self.tidy();
        let d = check.design();
        match d.outcome {
            Outcome::Dementia if !d.vaccine => Toy { theta: 1.0, delay: 0.0, adopters: 0.0, general: false, ..t },
            Outcome::Dementia => Toy { general: false, ..t },
            _ => Toy { adopters: if t.general { t.adopters } else { 0.0 }, general: t.general, ..Toy::OFF },
        }
    }
}

/// The ends of the strengths' sliders.
const THETA_LEAST: f64 = 0.5;
const PANDEMIC_MOST: f64 = 0.8;
const DRIFT_MOST: f64 = 0.12;
const ADOPTERS_MOST: f64 = 0.5;
const LEAVING_MOST: f64 = 0.5;
const PRESENT_MOST: f64 = 0.03;

/// The rate of recorded dementia in follow-up year `k` (from 0) with nothing switched on.
fn base(k: usize) -> f64 {
    // (the first year counts from three months)
    let years: f64 = 0.75 + (1..YEARS).map(|k| RISE.powi(k as i32)).sum::<f64>();
    -(1.0 - SIX_YEAR_SHARE).ln() / years * RISE.powi(k as i32)
}

/// The death rate in follow-up year `k`, set to give its share by six years.
fn death_rate(k: usize) -> f64 {
    let years: f64 = 0.75 + (1..YEARS).map(|k| DEATH_RISE.powi(k as i32)).sum::<f64>();
    -(1.0 - DEATH_SHARE).ln() / years * DEATH_RISE.powi(k as i32)
}

/// A constant rate that gives `share` by six years, counting from three months.
fn flat_rate(share: f64) -> f64 {
    -(1.0 - share).ln() / (YEARS as f64 - 0.25)
}

/// The newer vaccine's factor on the rate of dementia in follow-up year `k`: θ; with a
/// delay, θ for the first three years and, after, enough more that a share `delay` of
/// what was held off is made up by six years (all of it, at 1: with nothing else on,
/// the cumulative incidences then meet at six years).
fn vaccine_factor(toy: &Toy, k: usize) -> f64 {
    if k < HELD_YEARS || toy.delay <= 0.0 {
        return toy.theta;
    }
    let held: f64 = (0..HELD_YEARS).map(|k| base(k) * if k == 0 { 0.75 } else { 1.0 }).sum();
    let after: f64 = (HELD_YEARS..YEARS).map(base).sum();
    toy.theta + toy.delay * (1.0 - toy.theta) * (1.0 + held / after)
}

/// The share with dementia already present, unrecorded, among those vaccinated at `v`.
fn present_share(toy: &Toy, v: f64) -> f64 {
    toy.present * ((PRESENT_TO - v) / (PRESENT_TO - EARLIER.from)).max(0.0).min(1.0)
}

/// 1 − e^(−x): the share gone in a step at total rate x per step. (A short series where
/// x is small, as it nearly always is here: this is most of the model's arithmetic.)
fn lost(x: f64) -> f64 {
    if x < 0.05 {
        x * (1.0 - x * (0.5 - x * (1.0 / 6.0 - x * (1.0 / 24.0 - x / 120.0))))
    } else {
        -(-x).exp_m1()
    }
}

/// What is counted.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Outcome {
    Dementia,
    Death,
    /// The negative control: acutely painful conditions not linked with dementia.
    Painful,
    Shingles,
}

/// The main comparison, and each check the paper ran, as the toy runs them, with the
/// placebo switch, which the paper did not run for dementia.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Check {
    Main,
    SixMonths,
    BeforePandemic,
    Negative,
    Mortality,
    Shingles,
    Matching,
    Placebo,
}

const CHECKS: [Check; 8] =
    [Check::Main, Check::SixMonths, Check::BeforePandemic, Check::Negative, Check::Mortality, Check::Shingles, Check::Matching, Check::Placebo];

/// How a comparison is made: the two windows of first doses, how long they are followed
/// (steps), what is counted, whether the vaccine differs between them (not in the
/// placebo switch, where both had a vaccine that did not change), whether the later
/// cohort's adopters were selected (not in the placebo switch either: in this toy that
/// selection comes with the new vaccine, not with the period), and whether the result is
/// a hazard ratio rather than a ratio of time lost.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Design {
    windows: [Window; 2],
    horizon: usize,
    outcome: Outcome,
    vaccine: bool,
    adopters: bool,
    hazard_ratio: bool,
}

impl Check {
    fn design(self) -> Design {
        let main = Design { windows: [EARLIER, LATER], horizon: STEPS, outcome: Outcome::Dementia, vaccine: true, adopters: true, hazard_ratio: false };
        match self {
            Check::Main | Check::Matching => main,
            Check::SixMonths => Design { windows: [EARLIER_SIX, LATER_SIX], ..main },
            Check::BeforePandemic => Design { windows: [EARLIER_SIX, LATER_SIX], horizon: SHORT, hazard_ratio: true, ..main },
            Check::Negative => Design { outcome: Outcome::Painful, ..main },
            Check::Mortality => Design { outcome: Outcome::Death, ..main },
            Check::Shingles => Design { outcome: Outcome::Shingles, ..main },
            Check::Placebo => Design { vaccine: false, adopters: false, ..main },
        }
    }

    /// Its name in the table and over the charts.
    fn name(self) -> &'static str {
        match self {
            Check::Main => "main comparison",
            Check::SixMonths => "six-month windows",
            Check::BeforePandemic => "before the pandemic",
            Check::Negative => "negative-control outcome",
            Check::Mortality => "mortality",
            Check::Shingles => "shingles (positive control)",
            Check::Matching => "another matching method",
            Check::Placebo => "placebo switch",
        }
    }

    /// What it counts, for the charts' titles.
    fn counts(self) -> &'static str {
        match self.design().outcome {
            Outcome::Dementia => "recorded dementia",
            Outcome::Death => "deaths",
            Outcome::Painful => "the painful conditions of the negative control",
            Outcome::Shingles => "shingles",
        }
    }

    /// What the paper reports for it, if anything.
    fn paper(self) -> Option<Figure> {
        match self {
            Check::Main => Some(TABLE1[PRIMARY].1),
            Check::SixMonths => Some(TABLE1[SIX_MONTHS].1),
            Check::BeforePandemic => Some(BEFORE_PANDEMIC_HR),
            Check::Negative => Some(TABLE1[NEGATIVE].1),
            Check::Mortality => Some(TABLE1[MORTALITY].1),
            Check::Shingles => Some(TABLE1[SHINGLES].1),
            Check::Matching => Some(TABLE1[CEM].1),
            Check::Placebo => None,
        }
    }

    /// The paper's cumulative incidences for its outcome, where they were posted.
    fn marks(self) -> Option<&'static Marks> {
        match self {
            Check::Main => Some(&DEMENTIA_MARKS),
            Check::Mortality => Some(&DEATH_MARKS),
            Check::Shingles => Some(&SHINGLES_MARKS),
            _ => None,
        }
    }
}

/// The calendar factors on the rate of dementia diagnosis (pandemic and drift), step by
/// step from `CAL0`, each at the middle of its step.
fn calendar(toy: &Toy) -> Vec<f64> {
    (0..CAL_STEPS)
        .map(|j| {
            let c = CAL0 + (j as f64 + 0.5) * DT;
            let pandemic = if (PANDEMIC..PANDEMIC_END).contains(&c) { 1.0 - toy.pandemic } else { 1.0 };
            pandemic * (toy.drift * (c - DRIFT_FROM)).exp()
        })
        .collect()
}

/// One cohort as the analysis sees it.
#[derive(Clone, Debug, Default, PartialEq)]
struct Side {
    /// Cumulative incidence at each step's end from three months: `curve[0]` is 0 (at
    /// three months), `curve[n]` is after n steps.
    curve: Vec<f64>,
    /// Cases and person-years in each year after the first dose (the part counted).
    cases: [f64; YEARS],
    years: [f64; YEARS],
    /// Cases and person-time in each counted step, for the hazard ratio.
    events: Vec<f64>,
    exposure: Vec<f64>,
    /// Restricted mean time lost over the window, in years.
    rmtl: f64,
}

/// One group of people: its share of the cohort, where its first doses fall on the
/// calendar of `calendar` (in steps), how many steps it is followed before the data end,
/// and whether it has dementia already.
struct Group {
    weight: f64,
    at: usize,
    steps: usize,
    present: bool,
}

/// One side of a comparison (`later`: the cohort after the switch).
fn side(toy: &Toy, design: &Design, later: bool, cal: &[f64]) -> Side {
    let window = design.windows[later as usize];
    let dementia = design.outcome == Outcome::Dementia;
    let mut groups = Vec::with_capacity(2 * window.months);
    for m in 0..window.months {
        // (first doses in the middle of the month)
        let v = window.from + (m as f64 + 0.5) / 12.0;
        let at = ((window.from - CAL0) / DT).round().max(0.0) as usize + m * MONTH_STEPS + MONTH_STEPS / 2;
        let steps = (((DATA_END - v) / DT).round().max(0.0) as usize).min(design.horizon);
        let share = if dementia { present_share(toy, v) } else { 0.0 };
        let weight = 1.0 / window.months as f64;
        groups.push(Group { weight: weight * (1.0 - share), at, steps, present: false });
        if share > 0.0 {
            groups.push(Group { weight: weight * share, at, steps, present: true });
        }
    }
    let newer = later && design.vaccine;
    let selected = if later && design.adopters { 1.0 - toy.adopters } else { 1.0 };
    let general = if toy.general { selected } else { 1.0 };
    // each year's rate for those without dementia at the first dose, before the calendar
    let yearly: [f64; YEARS] = std::array::from_fn(|k| match design.outcome {
        Outcome::Dementia => base(k) * selected * if newer { vaccine_factor(toy, k) } else { 1.0 },
        Outcome::Death => death_rate(k) * general,
        Outcome::Painful => flat_rate(PAINFUL_SHARE) * general,
        Outcome::Shingles => flat_rate(SHINGLES_SHARE) * general * if newer { SHINGLES_HR } else { 1.0 },
    });
    let leaving = if later && dementia { toy.leaving } else { 0.0 };
    let mut alive: Vec<f64> = groups.iter().map(|g| g.weight).collect();
    let mut out = Side { curve: Vec::with_capacity(CURVE), ..Side::default() };
    out.curve.push(0.0);
    let mut survival = 1.0;
    for i in 0..design.horizon {
        let k = i / PER_YEAR;
        let (mut events, mut exposure) = (0.0, 0.0);
        for (g, a) in groups.iter().zip(alive.iter_mut()) {
            if i >= g.steps || *a <= 0.0 {
                continue;
            }
            let cal = if dementia { cal.get(g.at + i).copied().unwrap_or(1.0) } else { 1.0 };
            // everyone on the way to a diagnosis goes: most diagnosed, some leaving first
            let total = if g.present { RECORDED } else { yearly[k] } * cal;
            let gone = *a * lost(total * DT);
            let time = if total > 0.0 { gone / total } else { *a * DT };
            events += total * (1.0 - leaving) * time;
            exposure += time;
            *a -= gone;
        }
        if i < COUNT_FROM {
            continue;
        }
        // the hazard among those still seen, and the curve it gives
        let hazard = if exposure > 0.0 { events / exposure } else { 0.0 };
        let fall = lost(hazard * DT);
        out.rmtl += if hazard > 0.0 { DT - survival * fall / hazard } else { DT * (1.0 - survival) };
        survival *= 1.0 - fall;
        out.curve.push(1.0 - survival);
        out.cases[k] += events;
        out.years[k] += exposure;
        out.events.push(events);
        out.exposure.push(exposure);
    }
    out
}

/// Both sides of a comparison, and what it gives: the ratio of time lost (or the hazard
/// ratio, for the check before the pandemic) and the crude rate ratio in each year.
#[derive(Clone, Debug, PartialEq)]
struct Comparison {
    sides: [Side; 2],
    ratio: f64,
    by_year: [Option<f64>; YEARS],
}

fn compare(toy: &Toy, check: Check) -> Comparison {
    let toy = toy.tidy();
    let design = check.design();
    let cal = calendar(&toy);
    let sides = [side(&toy, &design, false, &cal), side(&toy, &design, true, &cal)];
    let ratio = if design.hazard_ratio { hazard_ratio(&sides[0], &sides[1]) } else { quotient(sides[1].rmtl, sides[0].rmtl) };
    let by_year = std::array::from_fn(|k| {
        let rate = |s: &Side| if s.years[k] > 0.0 && s.cases[k] > 0.0 { Some(s.cases[k] / s.years[k]) } else { None };
        match (rate(&sides[0]), rate(&sides[1])) {
            (Some(a), Some(b)) => Some(b / a),
            _ => None,
        }
    });
    Comparison { sides, ratio, by_year }
}

/// a / b, and 1 where there is nothing to divide.
fn quotient(a: f64, b: f64) -> f64 {
    if b > 0.0 && a.is_finite() && (a / b).is_finite() {
        a / b
    } else {
        1.0
    }
}

/// The hazard ratio a Cox model would find with endlessly many people: the β at which
/// the later cohort's cases equal what its share of the person-time at risk leads one
/// to expect, step by step (found by halving; 1 where there are no cases).
fn hazard_ratio(earlier: &Side, later: &Side) -> f64 {
    let n = earlier.events.len().min(later.events.len()).min(earlier.exposure.len()).min(later.exposure.len());
    let score = |beta: f64| {
        let e = beta.exp();
        let mut u = 0.0;
        for i in 0..n {
            let (y0, y1) = (earlier.exposure[i], later.exposure[i]);
            if y1 * e + y0 <= 0.0 {
                continue;
            }
            u += later.events[i] - y1 * e * (earlier.events[i] + later.events[i]) / (y1 * e + y0);
        }
        u
    };
    let total: f64 = earlier.events[..n].iter().chain(&later.events[..n]).sum();
    if total <= 0.0 || !total.is_finite() {
        return 1.0;
    }
    let (mut lo, mut hi) = (-6.0f64, 6.0f64);
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if score(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (0.5 * (lo + hi)).exp()
}

/// Every comparison for one toy, and each again with the vaccine's own effect alone.
#[derive(Clone, Debug, PartialEq)]
struct Results {
    toy: Toy,
    runs: Vec<Comparison>,
    alone: Vec<Comparison>,
}

impl Results {
    fn of(toy: &Toy) -> Results {
        let only = toy.vaccine_only();
        Results { toy: *toy, runs: CHECKS.iter().map(|c| compare(toy, *c)).collect(), alone: CHECKS.iter().map(|c| compare(&only, *c)).collect() }
    }

    /// The same for another toy, working out again only the comparisons it changes.
    fn update(&mut self, toy: &Toy) {
        if *toy == self.toy {
            return;
        }
        let (was, was_only, only) = (self.toy, self.toy.vaccine_only(), toy.vaccine_only());
        for (i, c) in CHECKS.iter().enumerate() {
            if toy.relevant(*c) != was.relevant(*c) {
                self.runs[i] = compare(toy, *c);
            }
            if only.relevant(*c) != was_only.relevant(*c) {
                self.alone[i] = compare(&only, *c);
            }
        }
        self.toy = *toy;
    }

    fn index(check: Check) -> usize {
        CHECKS.iter().position(|c| *c == check).unwrap_or(0)
    }

    fn run(&self, check: Check) -> &Comparison {
        &self.runs[Self::index(check).min(self.runs.len() - 1)]
    }

    fn alone(&self, check: Check) -> &Comparison {
        &self.alone[Self::index(check).min(self.alone.len() - 1)]
    }
}

// ----- the experiments ----------------------------------------------------------------

/// What the picture shows: the comparison on the calendar, what the paper found, or the
/// toy model.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Page {
    Design,
    Paper,
    Toy,
}

const PAGES: [Page; 3] = [Page::Design, Page::Paper, Page::Toy];

/// Everything an experiment sets.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Setup {
    page: Page,
    toy: Toy,
    check: Check,
}

const TRY: [(&str, &str, Setup); 5] = [
    (
        "the switch, on the calendar",
        "The opening picture: calendar time across, time since the first dose up, so that each person climbs a diagonal. The pandemic (March 2020) cuts the earlier cohort's band between 2.5 and 5.4 years after the first dose, and the later cohort's in its first 2.3 years or before it began. The later band is cut short by the end of the data. Whatever else changed over the decade falls on the two bands differently.",
        Setup { page: Page::Design, toy: Toy::OFF, check: Check::Main },
    ),
    (
        "a real effect, and nothing else",
        "Toy model: the newer vaccine's true hazard ratio is 0.80 and every other mechanism is off. The pink bars stand at 0.80 in every year, on the green line, and the ratio of time lost is 0.80. The yellow marks (crude ratios calculated from the paper's table) are not level: 0.69 in year 2, 1.16 in year 6. In this toy a constant effect alone does not give that shape.",
        Setup { page: Page::Toy, toy: Toy { theta: 0.8, ..Toy::OFF }, check: Check::Main },
    ),
    (
        "no effect, and a pandemic",
        "Toy model: no true effect, and 40 % fewer diagnoses from March 2020 for two years. The ratio by year falls below 1 early (0.74 in year 2) and rises above it late (1.33 in year 5), as the crude ratios from the paper's table do, and the ratio of time lost is 0.89, with no effect at all. Now look at the checks: the one that ends before the pandemic reads 1.00, where the paper found a hazard ratio of 0.74 (0.62–0.90), and the six-month windows read 0.97 (the paper's 0.83). This toy pandemic alone cannot give these.",
        Setup { page: Page::Toy, toy: Toy { pandemic: 0.4, ..Toy::OFF }, check: Check::Main },
    ),
    (
        "healthier early adopters",
        "Toy model: no true effect, but the later cohort's risk is 20 % lower, in general health, for reasons the records do not hold. Every year's ratio is 0.80, as with a real effect. The checks that should read 1 do not: mortality 0.81 and the negative control 0.80 (the paper's were 0.98 and 0.97). Set them to differ in cognition only: then every check, the placebo switch included, reads exactly what a true hazard ratio of 0.80 would give. No check here tells the two apart.",
        Setup { page: Page::Toy, toy: Toy { adopters: 0.2, general: true, ..Toy::OFF }, check: Check::Mortality },
    ),
    (
        "a placebo switch",
        "Toy model: no true effect, a pandemic (40 % fewer diagnoses for two years) and adopters healthier in cognition only (15 %). The main comparison gives 0.76. The placebo switch, the same comparison among people given a vaccine that did not change in 2017, gives 0.89: the pandemic's part, which comes with the period. The adopters' part does not show in it, because in this toy it comes with the new vaccine. No such check was reported for dementia.",
        Setup { page: Page::Toy, toy: Toy { pandemic: 0.4, adopters: 0.15, ..Toy::OFF }, check: Check::Placebo },
    ),
];

/// What each mechanism does alone in the toy, at the strength a click on the picture
/// gives it (held by `each_mechanism_has_its_signature`).
const SIGNATURES: &str = "What each does alone, at the strength a click on the picture gives it. A true effect (0.80), healthier adopters (20 %) or leaving the records (20 %): the same ratio, 0.80, in every year. A drift (−5 % a year): nearly the same in every year (0.86 to 0.91). The pandemic (40 %): below 1 in the first three years, above 1 in years 5 and 6. Dementia already there (1 %): below 1 in year 1, fading to 1. A delay (all made up): the true effect for three years, above 1 after, and curves that meet at six years. Only the pandemic and the delay put the late years above 1.";

/// Which checks see which mechanism, in the toy (held by `which_checks_see_what`).
const CAUGHT: &str = "In this toy: the six-month windows and the check before the pandemic bring the calendar's mechanisms (pandemic, drift, dementia already there) most of the way back to what the vaccine alone gives, and leave the rest as they are. Mortality and the negative control move only with adopters healthier in general. The placebo switch shows everything that comes with the period (pandemic, drift, leaving, dementia already there) and nothing that comes with the new vaccine (a true effect, a delay, healthier adopters).";

/// The mechanisms as the switches over the toy's picture list them: name, what it is in
/// the graph, and the strength a click turns it on at.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mechanism {
    Effect,
    Delay,
    Pandemic,
    Drift,
    Adopters,
    Leaving,
    Present,
}

const MECHANISMS: [Mechanism; 7] =
    [Mechanism::Effect, Mechanism::Delay, Mechanism::Pandemic, Mechanism::Drift, Mechanism::Adopters, Mechanism::Leaving, Mechanism::Present];

impl Mechanism {
    fn name(self) -> &'static str {
        match self {
            Mechanism::Effect => "a true effect of the newer vaccine",
            Mechanism::Delay => "a delay, not prevention",
            Mechanism::Pandemic => "the pandemic: fewer diagnoses",
            Mechanism::Drift => "a drift in diagnosis and coding",
            Mechanism::Adopters => "healthier early adopters",
            Mechanism::Leaving => "leaving the records first",
            Mechanism::Present => "dementia already there, unrecorded",
        }
    }

    fn short(self) -> &'static str {
        match self {
            Mechanism::Effect => "true effect",
            Mechanism::Delay => "delay",
            Mechanism::Pandemic => "pandemic",
            Mechanism::Drift => "drift",
            Mechanism::Adopters => "adopters",
            Mechanism::Leaving => "leaving",
            Mechanism::Present => "already there",
        }
    }

    fn on(self, t: &Toy) -> bool {
        match self {
            Mechanism::Effect => t.theta < 1.0,
            Mechanism::Delay => t.delay > 0.0,
            Mechanism::Pandemic => t.pandemic > 0.0,
            Mechanism::Drift => t.drift != 0.0,
            Mechanism::Adopters => t.adopters > 0.0,
            Mechanism::Leaving => t.leaving > 0.0,
            Mechanism::Present => t.present > 0.0,
        }
    }

    /// Its strength as the switch writes it.
    fn strength(self, t: &Toy) -> String {
        if !self.on(t) {
            return "off".into();
        }
        let percent = |v: f64| format!("{} %", num(100.0 * v, 0));
        match self {
            Mechanism::Effect => format!("hazard ratio {}", num(t.theta, 2)),
            Mechanism::Delay if t.theta >= 1.0 => "nothing to delay".into(),
            Mechanism::Delay => format!("{} made up later", percent(t.delay)),
            Mechanism::Pandemic => format!("{} fewer, 2 years", percent(t.pandemic)),
            Mechanism::Drift => format!("{}{} a year", if t.drift > 0.0 { "+" } else { "" }, percent(t.drift)),
            Mechanism::Adopters => format!("{} lower, {}", percent(t.adopters), if t.general { "general" } else { "cognition" }),
            Mechanism::Leaving => format!("{} of those due", percent(t.leaving)),
            Mechanism::Present => format!("{} % in Oct 2014", num(100.0 * t.present, 1)),
        }
    }

    /// Switch it on (at a strength that shows) or off.
    fn toggle(self, t: &mut Toy) {
        let on = !self.on(t);
        match self {
            Mechanism::Effect => t.theta = if on { 0.8 } else { 1.0 },
            Mechanism::Delay => {
                // (a delay is of a true effect: with none, one comes with it)
                t.delay = if on { 1.0 } else { 0.0 };
                if on && t.theta >= 1.0 {
                    t.theta = 0.8;
                }
            }
            Mechanism::Pandemic => t.pandemic = if on { 0.4 } else { 0.0 },
            Mechanism::Drift => t.drift = if on { -0.05 } else { 0.0 },
            Mechanism::Adopters => t.adopters = if on { 0.2 } else { 0.0 },
            Mechanism::Leaving => t.leaving = if on { 0.2 } else { 0.0 },
            Mechanism::Present => t.present = if on { 0.01 } else { 0.0 },
        }
    }

    /// The colour of its arrows: the vaccine's own effect is what is really so (green);
    /// every other mechanism is a third road (purple).
    fn color(self) -> Color {
        match self {
            Mechanism::Effect | Mechanism::Delay => TRUTH,
            _ => THIRD,
        }
    }
}

// ----- what the picture follows -------------------------------------------------------

/// How long a glide to new settings takes, about (each of two eases, in seconds).
const GLIDE: f64 = 0.08;

/// A list of numbers the picture follows: two eases one after the other, so that a
/// change of setting is drawn moving evenly, and exactly there in the end.
#[derive(Clone, Debug, Default)]
struct Follow {
    mid: Vec<f64>,
    shown: Vec<f64>,
}

impl Follow {
    fn jump(&mut self, v: &[f64]) {
        self.mid = v.to_vec();
        self.shown = v.to_vec();
    }

    fn get(&self, i: usize) -> f64 {
        self.shown.get(i).copied().unwrap_or(0.0)
    }

    /// `dt` seconds nearer to `to`; true while still on the way. A number new to the
    /// list starts where it is going.
    fn toward(&mut self, to: &[f64], dt: f64) -> bool {
        let had = self.shown.len().min(self.mid.len()).min(to.len());
        self.mid.truncate(had);
        self.shown.truncate(had);
        self.mid.extend_from_slice(&to[had..]);
        self.shown.extend_from_slice(&to[had..]);
        let k = 1.0 - (-dt.max(0.0) / GLIDE).exp();
        let mut apart = 0.0f64;
        for ((m, s), t) in self.mid.iter_mut().zip(self.shown.iter_mut()).zip(to) {
            *m += (t - *m) * k;
            *s += (*m - *s) * k;
            apart = apart.max((t - *m).abs()).max((t - *s).abs());
        }
        // (exactly there in the end, not nearly: the picture then stops changing)
        if apart < 1e-5 || !apart.is_finite() {
            self.jump(to);
            false
        } else {
            true
        }
    }
}

/// What the toy's charts are drawn from: each cohort's curve (padded to six years with
/// its last value), the ratio in each year (the toy's, then the vaccine's alone: 1 where
/// there is none), the ratio over the window (toy, vaccine alone), and the charts'
/// scales (the top of the curves' chart, and the ends of the two ratio scales).
#[derive(Clone, Debug, Default)]
struct Shown {
    curves: [Follow; 2],
    years: Follow,
    ratio: Follow,
    scales: Follow,
}

/// The numbers of `Shown` for a check of `results`.
struct Targets {
    curves: [Vec<f64>; 2],
    years: Vec<f64>,
    ratio: [f64; 2],
    scales: [f64; 5],
}

fn targets(results: &Results, check: Check) -> Targets {
    let run = results.run(check);
    let alone = results.alone(check);
    let curve = |s: &Side| {
        let mut c = s.curve.clone();
        let last = c.last().copied().unwrap_or(0.0);
        c.resize(CURVE, last);
        c
    };
    let years: Vec<f64> = run.by_year.iter().chain(alone.by_year.iter()).map(|y| y.unwrap_or(1.0)).collect();
    // the scales: room for every curve and every published mark, and for every ratio
    let mut top: f64 = run.sides.iter().flat_map(|s| s.curve.iter()).fold(0.0f64, |m, v| m.max(*v)) * 100.0;
    if let Some(m) = check.marks() {
        top = m.later.iter().chain(m.earlier).fold(top, |a, b| a.max(*b));
    }
    let top = nice_top(top);
    let mut spread: Vec<f64> = years.clone();
    if check == Check::Main {
        spread.extend(CRUDE_RATIOS);
    }
    let (lo, hi) = ratio_scale(&spread, 0.6, 1.4);
    let mut points = vec![run.ratio, alone.ratio];
    if let Some(p) = check.paper() {
        points.extend([p.low, p.high]);
    }
    let (rlo, rhi) = ratio_scale(&points, 0.6, 1.2);
    Targets { curves: [curve(&run.sides[0]), curve(&run.sides[1])], years, ratio: [run.ratio, alone.ratio], scales: [top, lo, hi, rlo, rhi] }
}

/// The top of the curves' chart, in percent: a round number a little above the highest.
fn nice_top(v: f64) -> f64 {
    let v = if v.is_finite() { v.max(0.5) } else { 1.0 };
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0].into_iter().find(|t| *t >= v * 1.08).unwrap_or(30.0)
}

/// A scale for ratios: from at most `lo` to at least `hi`, wide enough for every value,
/// in steps of a tenth.
fn ratio_scale(values: &[f64], lo: f64, hi: f64) -> (f64, f64) {
    let (mut a, mut b) = (lo, hi);
    for v in values.iter().filter(|v| v.is_finite()) {
        a = a.min(*v);
        b = b.max(*v);
    }
    ((a * 10.0 - 0.6).floor() / 10.0, (b * 10.0 + 0.6).ceil() / 10.0)
}

// ----- the scene ----------------------------------------------------------------------

struct Switch {
    page: Page,
    toy: Toy,
    check: Check,
    results: Results,
    shown: Shown,
    moving: bool,
    /// Line breaks of the texts on the picture, kept (measuring them is costly).
    wrap: Wrap,
}

impl Switch {
    fn new() -> Switch {
        let toy = Toy::OFF;
        let results = Results::of(&toy);
        let mut s = Switch { page: Page::Design, toy, check: Check::Main, results, shown: Shown::default(), moving: false, wrap: Wrap::default() };
        s.settle();
        s
    }

    fn apply(&mut self, setup: Setup) {
        self.page = setup.page;
        self.toy = setup.toy.tidy();
        self.check = setup.check;
        self.refresh();
    }

    /// The results for the settings as they are now; the picture glides to them.
    fn refresh(&mut self) {
        self.toy = self.toy.tidy();
        self.results.update(&self.toy);
        self.moving = true;
    }

    /// Everything drawn where it is going, at once.
    fn settle(&mut self) {
        let t = targets(&self.results, self.check);
        let s = &mut self.shown;
        s.curves[0].jump(&t.curves[0]);
        s.curves[1].jump(&t.curves[1]);
        s.years.jump(&t.years);
        s.ratio.jump(&t.ratio);
        s.scales.jump(&t.scales);
        self.moving = false;
    }

    fn advance(&mut self, dt: f32) {
        let t = targets(&self.results, self.check);
        let dt = dt.max(0.0) as f64;
        let s = &mut self.shown;
        let mut moving = s.curves[0].toward(&t.curves[0], dt);
        moving |= s.curves[1].toward(&t.curves[1], dt);
        moving |= s.years.toward(&t.years, dt);
        moving |= s.ratio.toward(&t.ratio, dt);
        moving |= s.scales.toward(&t.scales, dt);
        self.moving = moving;
    }
}

// ----- geometry -----------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    min: Vec2,
    max: Vec2,
}

impl Rect {
    const NONE: Rect = Rect { min: Vec2::ZERO, max: Vec2::ZERO };

    fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
        Rect { min: Vec2::new(x0, y0), max: Vec2::new(x1.max(x0), y1.max(y0)) }
    }

    fn w(&self) -> f32 {
        self.max.x - self.min.x
    }

    fn h(&self) -> f32 {
        self.max.y - self.min.y
    }

    fn contains(&self, p: Vec2, margin: f32) -> bool {
        p.x >= self.min.x - margin && p.x <= self.max.x + margin && p.y >= self.min.y - margin && p.y <= self.max.y + margin
    }

    fn shows(&self) -> bool {
        self.w() >= 20.0 && self.h() >= 20.0
    }
}

/// Where everything goes on a canvas of `view` logical pixels, the top `inset` of which
/// lies under the application's buttons. A wide canvas has columns; a narrow one (a phone
/// held upright) stacks the parts; one too small for more has only each page's main
/// picture. Each chart's rectangle is its plotting area: its title goes above it and its
/// numbers to its left and below.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    view: Vec2,
    wide: bool,
    bare: bool,
    text: f32,
    small: f32,
    tabs: Rect,
    /// Where the sources line sits (its bottom left).
    foot: Vec2,
    // the comparison on the calendar
    lexis: Rect,
    graph: Rect,
    legend: Rect,
    words: Rect,
    // what the paper found
    survival: Rect,
    sums: Rect,
    forest: Rect,
    years: Rect,
    // the toy model
    frame: Rect,
    mini: Rect,
    toy_graph: Rect,
    toy_legend: Rect,
    mechs: Rect,
    switches: Rect,
    cum: Rect,
    ratios: Rect,
    strip: Rect,
    table: Rect,
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::ONE);
        let (w, h) = (view.x, view.y);
        let bare = w < 300.0 || h < 300.0;
        let wide = !bare && w >= 640.0 && w >= 0.7 * h;
        let (text, small) = if wide { (13.0, 11.5) } else { (11.5, 10.5) };
        let side = if wide { 18.0 } else { 10.0 };
        let top = inset.max(0.0).min(h * 0.3) + if wide { 12.0 } else { 6.0 };
        let tabs = Rect::new(side, top, w - side, top + if wide { 28.0 } else { 26.0 });
        let foot = Vec2::new(side, h - 6.0);
        let body = tabs.max.y + if wide { 14.0 } else { 10.0 };
        // (room for the sources line: two lines on a narrow canvas)
        let bottom = h - if bare {
            6.0
        } else if wide {
            small + 16.0
        } else {
            2.0 * small + 18.0
        };
        let bh = (bottom - body).max(10.0);
        let none = Rect::NONE;
        let mut lay = Layout {
            view,
            wide,
            bare,
            text,
            small,
            tabs,
            foot,
            lexis: none,
            graph: none,
            legend: none,
            words: none,
            survival: none,
            sums: none,
            forest: none,
            years: none,
            frame: none,
            mini: none,
            toy_graph: none,
            toy_legend: none,
            mechs: none,
            switches: none,
            cum: none,
            ratios: none,
            strip: none,
            table: none,
        };
        if bare {
            // each page's main picture alone
            lay.lexis = Rect::new(side + 30.0, body + small + 8.0, w - side, bottom - small - 10.0);
            lay.forest = Rect::new(side, body + small + 8.0, w - side, bottom);
            lay.ratios = Rect::new(side + 30.0, body + small + 8.0, w - side, bottom - small - 8.0);
            return lay;
        }
        let line = |n: f32, size: f32| n * (size + 5.0);
        if wide && bh < 520.0 {
            // a wide canvas too short for rows of parts (a phone held sideways): each page's
            // two main parts side by side
            let half = side + 0.5 * (w - 2.0 * side);
            lay.lexis = Rect::new(side + 54.0, body + line(2.0, text) + 6.0, w - side - 8.0, bottom - line(2.0, small) - 24.0);
            lay.survival = Rect::new(side + 46.0, body + line(2.0, text) + 4.0, half - 20.0, bottom - line(2.0, small) - 10.0);
            lay.forest = Rect::new(half + 20.0, body + line(1.0, text) + 6.0, w - side, bottom);
            lay.frame = Rect::new(side - 8.0, body, w - side + 8.0, bottom);
            let inner = body + line(1.0, text) + 14.0;
            let top = inner + line(3.0, small) + 18.0;
            // (the ratio by year at the left; the checks at the right, and the ratio over
            // the window under them where there is room)
            lay.ratios = Rect::new(side + 40.0, top, half - 20.0, bottom - line(1.0, small) - 10.0);
            lay.table = Rect::new(half + 10.0, inner, w - side - 6.0, (inner + 9.0 * (small + 6.0)).min(bottom - 8.0));
            let st = lay.table.max.y + line(2.0, small) + 12.0;
            if st + 16.0 + small + 10.0 <= bottom {
                lay.strip = Rect::new(half + 40.0, st, w - side - 8.0, st + 16.0);
            }
        } else if wide {
            // the comparison: the calendar across the top, the graph and the words under it
            let split = body + (0.54 * bh).max(220.0).min(bh - 170.0);
            lay.lexis = Rect::new(side + 54.0, body + line(2.0, text) + 6.0, w - side - 8.0, split - line(2.0, small) - 24.0);
            let gw = (0.47 * w).min(560.0);
            lay.graph = Rect::new(side, split + line(1.0, small) + 6.0, side + gw, bottom - line(4.0, small) - 6.0);
            lay.legend = Rect::new(side, lay.graph.max.y + 4.0, side + gw, bottom);
            lay.words = Rect::new(side + gw + 28.0, split, w - side, bottom - 4.0);
            // the paper: the curves and the years at the left, the forest and the sums at the right
            let col = side + (0.46 * w).max(300.0);
            lay.survival = Rect::new(side + 46.0, body + line(2.0, text) + 4.0, col, body + 0.40 * bh);
            lay.years = Rect::new(side + 46.0, lay.survival.max.y + line(2.0, small) + line(2.0, text) + 22.0, col, bottom - line(4.0, small) - 10.0);
            lay.forest = Rect::new(col + 40.0, body + line(1.0, text) + 6.0, w - side, body + (0.52 * bh).max(200.0));
            lay.sums = Rect::new(col + 40.0, lay.forest.max.y + 14.0, w - side, bottom);
            // the toy model, in a frame: the calendar, the graph and the switches at the
            // left; the charts, the ratio and the checks at the right (the calendar and
            // the switches only where there is room for them)
            lay.frame = Rect::new(side - 8.0, body, w - side + 8.0, bottom);
            let inner = body + line(1.0, text) + 14.0;
            let fh = (bottom - inner).max(10.0);
            let left = (0.37 * w).max(260.0);
            let top = inner + line(1.0, small) + 6.0;
            let tall = fh >= 700.0;
            let g = if tall {
                lay.mini = Rect::new(side + 34.0, top, side + left, top + 0.2 * fh);
                lay.mini.max.y + line(3.0, small) + 22.0
            } else {
                top + 4.0
            };
            lay.toy_graph = Rect::new(side, g, side + left, g + if tall { 0.24 } else { 0.3 } * fh);
            lay.toy_legend = Rect::new(side, lay.toy_graph.max.y + 2.0, side + left, lay.toy_graph.max.y + line(3.0, small) + 4.0);
            let sw = lay.toy_legend.max.y + line(1.0, small) + 8.0;
            if bottom - 8.0 - sw >= 7.0 * 15.0 {
                lay.switches = Rect::new(side, sw, side + left, bottom - 8.0);
            }
            let rx = side + left + 70.0;
            lay.cum = Rect::new(rx, top, w - side - 56.0, inner + 0.27 * fh);
            let r = lay.cum.max.y + line(3.0, small) + 18.0;
            lay.ratios = Rect::new(rx, r, w - side - 6.0, r + 0.18 * fh);
            let st = lay.ratios.max.y + line(3.0, small) + 12.0;
            lay.strip = Rect::new(rx, st, w - side - 6.0, st + 20.0);
            lay.table = Rect::new(rx - 36.0, lay.strip.max.y + line(1.0, small) + 14.0, w - side - 6.0, bottom - 8.0);
        } else {
            // a phone: the parts one under another
            lay.lexis = Rect::new(side + 30.0, body + line(2.0, small) + 4.0, w - side - 4.0, body + 0.37 * bh);
            let g = lay.lexis.max.y + line(3.0, small) + 22.0;
            // (tall enough for the variables above the row and below it to stand apart)
            lay.graph = Rect::new(side, g, w - side, g + (0.22 * bh).max(180.0));
            lay.legend = Rect::new(side, lay.graph.max.y + 2.0, w - side, lay.graph.max.y + line(5.0, small) + 4.0);
            lay.words = Rect::new(side, lay.legend.max.y + 6.0, w - side, bottom);
            lay.survival = Rect::new(side + 34.0, body + line(2.0, small) + 4.0, w - side - 4.0, body + 0.25 * bh);
            let fo = lay.survival.max.y + line(2.0, small) + 24.0;
            lay.forest = Rect::new(side, fo, w - side, fo + 0.38 * bh);
            lay.years = Rect::new(side + 34.0, lay.forest.max.y + line(2.0, small) + 14.0, w - side - 4.0, bottom - line(4.0, small) - 4.0);
            lay.frame = Rect::new(side - 6.0, body, w - side + 6.0, bottom);
            let inner = body + line(1.0, small) + 10.0;
            let fh = (bottom - inner).max(10.0);
            lay.mechs = Rect::new(side, inner, w - side, inner + line(2.0, small));
            let c = lay.mechs.max.y + line(1.0, small) + 8.0;
            lay.cum = Rect::new(side + 32.0, c, w - side - 44.0, c + 0.22 * fh);
            let r = lay.cum.max.y + line(4.0, small) + 12.0;
            lay.ratios = Rect::new(side + 32.0, r, w - side - 4.0, r + 0.19 * fh);
            let s = lay.ratios.max.y + line(3.0, small) + 8.0;
            lay.strip = Rect::new(side + 32.0, s, w - side - 8.0, s + 16.0);
            lay.table = Rect::new(side, lay.strip.max.y + line(1.0, small) + 12.0, w - side, bottom - 4.0);
        }
        lay
    }

    /// The tabs: one per page.
    fn tab(&self, k: usize) -> Rect {
        let n = PAGES.len() as f32;
        let wd = if self.wide { 172.0f32.min((self.tabs.w() - 8.0) / n) } else { (self.tabs.w() - 8.0) / n };
        let x = self.tabs.min.x + k as f32 * (wd + 4.0);
        Rect::new(x, self.tabs.min.y, x + wd, self.tabs.max.y)
    }
}

// ----- drawing helpers ----------------------------------------------------------------

/// A translucent shape on the screen, on one of the sheets of `Frame::hud_at`: the
/// corners in order round it (convex).
fn hud_shape(f: &mut Frame, level: usize, corners: &[Vec2], color: Color) {
    if corners.len() < 3 || corners.iter().any(|c| !c.is_finite()) {
        return;
    }
    let p: Vec<Vec3> = corners.iter().map(|c| f.hud_at(level, *c)).collect();
    for i in 1..p.len() - 1 {
        f.dl.tris_alpha.tri(p[0], p[i], p[i + 1], color);
    }
}

fn hud_fill(f: &mut Frame, level: usize, min: Vec2, max: Vec2, color: Color) {
    hud_shape(f, level, &[min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)], color);
}

/// A grid line of a chart, on the screen: a batch of its own, so that a chart whose
/// scale glides and gains a line on the way does not change the batch its curves are in.
fn hud_grid(f: &mut Frame, a: Vec2, b: Vec2, width: f32, color: Color) {
    let (a, b) = (f.hud(a), f.hud(b));
    f.dl.lines_with(LineKind::Overlay, LineStyle { depth_bias: 0.0, intensity: 0.999, ..Default::default() }).segment(a, b, width, color);
}

/// A dashed line on the screen, on top of everything.
fn hud_dashed(f: &mut Frame, a: Vec2, b: Vec2, width: f32, color: Color) {
    let (a, b) = (f.hud(a), f.hud(b));
    f.dl.lines_with(LineKind::Overlay, LineStyle { dash: 7.0, duty: 0.55, depth_bias: 0.0, ..Default::default() }).segment(a, b, width, color);
}

/// A dashed arrow on the screen: the head solid.
fn hud_dashed_arrow(f: &mut Frame, a: Vec2, b: Vec2, width: f32, color: Color) {
    let d = (b - a).normalize_or_zero();
    let head = (4.0 + 2.5 * width).min((b - a).length() * 0.5);
    hud_dashed(f, a, b - d * head * 0.5, width, color);
    let n = Vec2::new(-d.y, d.x);
    hud_line(f, b, b - d * head + n * head * 0.5, width, color);
    hud_line(f, b, b - d * head - n * head * 0.5, width, color);
}

/// A tab over the picture: true when clicked.
fn hud_tab(f: &mut Frame, r: Rect, text: &str, on: bool, pointer: Option<Vec2>, size: f32) -> bool {
    let over = pointer.is_some_and(|p| r.contains(p, 0.0));
    let fill = match (on, over) {
        (true, _) => Color::hex(0x2B3B60),
        (false, true) => Color::hex(0x2A3349),
        (false, false) => Color::hex(0x161C2B),
    };
    hud_rect(f, r.min, r.max, fill);
    if on {
        hud_line(f, Vec2::new(r.min.x, r.max.y - 1.0), Vec2::new(r.max.x, r.max.y - 1.0), 2.0, pal::LINK);
    }
    f.text_px((r.min + r.max) * 0.5, text, size, if on { pal::TEXT } else { pal::DIM }, Align2::CENTER_CENTER);
    over && f.ptr.clicked
}

/// The first of `options` that fits in `room` pixels at `size` (the last if none does).
fn pick(f: &Frame, options: &[&str], room: f32, size: f32) -> String {
    options.iter().copied().find(|t| f.text_size(t, size).x <= room).or_else(|| options.last().copied()).unwrap_or("").to_string()
}

/// The words of a text, each formula whole.
fn words_of(text: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let (mut start, mut inside) = (0, false);
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

/// Texts broken into lines, kept from frame to frame: measuring every word of a long
/// text again in every frame would cost more than the rest of the picture.
#[derive(Default)]
struct Wrap {
    kept: HashMap<(String, u32, u32), Vec<String>>,
}

impl Wrap {
    /// `text` in lines no wider than `width` pixels at `size`.
    fn lines(&mut self, f: &Frame, text: &str, width: f32, size: f32) -> Vec<String> {
        let key = (text.to_string(), width.round() as u32, (size * 10.0).round() as u32);
        if let Some(lines) = self.kept.get(&key) {
            return lines.clone();
        }
        // (formulas can be measured only once their fonts are in place: until then the
        // lines are broken by an estimate, and not kept)
        let exact = !crate::typeset::has_math(text) || crate::typeset::measure(f.ctx, text, size).is_some();
        let measure = |t: &str| if exact { f.text_size(t, size).x } else { text_width(t, size) };
        let mut lines: Vec<String> = vec![String::new()];
        for word in words_of(text) {
            let Some(last) = lines.last_mut() else { break };
            // (a per cent sign stays with its number)
            if word.starts_with('%') && !last.is_empty() {
                last.push(' ');
                last.push_str(word);
                continue;
            }
            if last.is_empty() {
                last.push_str(word);
            } else if measure(&format!("{last} {word}")) > width {
                lines.push(word.to_string());
            } else {
                last.push(' ');
                last.push_str(word);
            }
        }
        if exact {
            if self.kept.len() > 600 {
                self.kept.clear();
            }
            self.kept.insert(key, lines.clone());
        }
        lines
    }

    /// Draw it, the first line's top left at `at`, if all of it fits above `bottom`;
    /// returns the height it took.
    #[allow(clippy::too_many_arguments)]
    fn draw(&mut self, f: &mut Frame, at: Vec2, width: f32, bottom: f32, text: &str, size: f32, color: Color) -> f32 {
        let lines = self.lines(f, text, width, size);
        let step = size + 4.0;
        // (all of it or none: a sentence cut off half way would say something else)
        if at.y + lines.len() as f32 * step > bottom + 0.5 {
            return 0.0;
        }
        let mut y = at.y;
        for l in lines {
            f.text_px(Vec2::new(at.x, y), l, size, color, Align2::LEFT_TOP);
            y += step;
        }
        y - at.y
    }
}

/// The colours of the two cohorts: the earlier (mostly the live vaccine) grey, the later
/// (mostly the recombinant vaccine) the colour of the cause asked about.
const EARLY: Color = Color::hex(0xB8C0D0);
const LATE: Color = CAUSE;
/// What the paper reports, wherever it is set beside the toy model's numbers.
const PUBLISHED: Color = pal::YELLOW;
/// The frame round the toy model.
const TOY_FRAME: Color = Color::hex(0xC9A3FF);

/// A convex polygon cut to the part with x at most `cut` (`keep_left`) or at least it.
fn clip_x(points: &[Vec2], cut: f32, keep_left: bool) -> Vec<Vec2> {
    let inside = |p: Vec2| if keep_left { p.x <= cut } else { p.x >= cut };
    let mut out = Vec::new();
    for i in 0..points.len() {
        let (a, b) = (points[i], points[(i + 1) % points.len()]);
        if inside(a) {
            out.push(a);
        }
        if inside(a) != inside(b) && (b.x - a.x).abs() > 1e-6 {
            let t = (cut - a.x) / (b.x - a.x);
            out.push(a + (b - a) * t);
        }
    }
    out
}

// ----- the causal graph ---------------------------------------------------------------

/// The variables of the graph (dossier §4.3): the period of the first dose, the vaccine,
/// dementia, its recorded diagnosis; the recorded traits (matched), the traits the
/// records do not hold, the calendar years the follow-up falls in, how far the records
/// reach back before the first dose, and leaving the records.
#[derive(Clone, Copy, Debug, PartialEq)]
enum V {
    Era,
    Vaccine,
    Dementia,
    Recorded,
    Traits,
    Unrecorded,
    Calendar,
    LookBack,
    Leaving,
}

/// Each variable: where it is in a box 1 by 1 (laid out so that no two arrows cross),
/// its letter, what it is, and whether the analysis holds it (drawn boxed: matched on,
/// or counted only while in the records).
const VARS: [(V, f32, f32, &str, &str, bool); 9] = [
    (V::Era, 0.03, 0.40, "E", "period of first dose", false),
    (V::Vaccine, 0.32, 0.40, "T", "vaccine", false),
    (V::Dementia, 0.62, 0.40, "D", "dementia", false),
    (V::Recorded, 0.95, 0.88, "D^*", "recorded diagnosis", false),
    (V::Traits, 0.42, 0.0, "X", "recorded traits (matched)", true),
    (V::Unrecorded, 0.42, 0.18, "U", "traits not recorded", false),
    (V::Leaving, 0.62, 0.66, "C", "still in the records", true),
    (V::LookBack, 0.34, 0.76, "L", "records before the dose", false),
    (V::Calendar, 0.14, 0.97, "K", "calendar years followed", false),
];

/// The colour of a variable of the graph: the vaccine is the cause asked about, dementia
/// and its record the effect, the recorded traits grey, the rest third variables.
fn var_color(v: V) -> Color {
    match v {
        V::Vaccine => CAUSE,
        V::Dementia | V::Recorded => EFFECT,
        V::Traits => pal::DIM,
        _ => THIRD,
    }
}

/// The roads between them: the effect asked about, and each other road from the period
/// to the recorded diagnosis, with the mechanism that opens it in the toy.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Road {
    /// The period decides the vaccine, the vaccine (perhaps) dementia, dementia its record.
    Asked,
    /// Through the recorded traits: closed by matching.
    Matched,
    /// Through who came forward in each period.
    Adopters,
    /// Through leaving the records on the way to a diagnosis.
    Leaving,
    /// Through how far the records reach back.
    LookBack,
    /// Through the calendar years of follow-up: the pandemic, drift.
    Calendar,
}

const EDGES: [(V, V, Road); 15] = [
    (V::Era, V::Vaccine, Road::Asked),
    (V::Vaccine, V::Dementia, Road::Asked),
    (V::Dementia, V::Recorded, Road::Asked),
    (V::Era, V::Traits, Road::Matched),
    (V::Traits, V::Dementia, Road::Matched),
    (V::Era, V::Unrecorded, Road::Adopters),
    (V::Unrecorded, V::Dementia, Road::Adopters),
    (V::Era, V::Leaving, Road::Leaving),
    (V::Dementia, V::Leaving, Road::Leaving),
    (V::Leaving, V::Recorded, Road::Leaving),
    (V::Era, V::LookBack, Road::LookBack),
    (V::LookBack, V::Recorded, Road::LookBack),
    (V::Era, V::Calendar, Road::Calendar),
    (V::Calendar, V::Recorded, Road::Calendar),
    (V::Era, V::Era, Road::Asked),
];

impl Road {
    /// The toy's mechanisms that open it.
    fn mechanisms(self) -> &'static [Mechanism] {
        match self {
            Road::Asked | Road::Matched => &[],
            Road::Adopters => &[Mechanism::Adopters],
            Road::Leaving => &[Mechanism::Leaving],
            Road::LookBack => &[Mechanism::Present],
            Road::Calendar => &[Mechanism::Pandemic, Mechanism::Drift],
        }
    }
}

// ----- drawing --------------------------------------------------------------------------

impl Switch {
    /// The tabs over the picture.
    fn tabs(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let names: [&str; 3] = if lay.wide { ["the comparison", "what the paper found", "toy model"] } else { ["comparison", "the paper", "toy model"] };
        for (k, name) in names.into_iter().enumerate() {
            if hud_tab(f, lay.tab(k), name, self.page == PAGES[k], pointer, lay.small) && self.page != PAGES[k] {
                self.page = PAGES[k];
                self.moving = true;
            }
        }
    }

    /// The sources line, at the foot of every page.
    fn sources(&mut self, f: &mut Frame, lay: &Layout) {
        if lay.bare {
            return;
        }
        let room = lay.view.x - 2.0 * lay.foot.x;
        let size = lay.small - 1.0;
        if f.text_size(SOURCES, size).x <= room || f.text_size(SOURCES_SHORT, size).x <= room {
            let text = pick(f, &[SOURCES, SOURCES_SHORT], room, size);
            f.text_px(lay.foot, text, size, pal::DIM, Align2::LEFT_BOTTOM);
        } else {
            // (on two lines where one is too short)
            let lines = self.wrap.lines(f, SOURCES_SHORT, room, size);
            let n = lines.len().min(2);
            for (k, l) in lines.into_iter().take(2).enumerate() {
                f.text_px(lay.foot - Vec2::new(0.0, (n - 1 - k) as f32 * (size + 3.0)), l, size, pal::DIM, Align2::LEFT_BOTTOM);
            }
        }
    }

    /// The causal graph in `r`. `toy`: light each road the toy has switched on; without
    /// it, the roads the design needs absent are dashed.
    fn graph(&mut self, f: &mut Frame, r: Rect, size: f32, toy: Option<&Toy>) {
        if !r.shows() {
            return;
        }
        // (no flatter than this, or the variables above and below the row crowd it)
        let r = Rect::new(r.min.x, r.min.y, r.min.x + r.w().min(2.6 * r.h()), r.max.y);
        let radius = (r.h() * 0.065).clamp(9.0, 16.0);
        let inset = Vec2::new(radius + 6.0, radius + 6.0);
        let (lo, hi) = (r.min + inset, r.max - inset);
        let at = |v: V| VARS.iter().find(|x| x.0 == v).map_or(r.min, |x| Vec2::new(lo.x + x.1 * (hi.x - lo.x), lo.y + x.2 * (hi.y - lo.y)));
        let lit = |road: Road| toy.is_some_and(|t| road.mechanisms().iter().any(|m| m.on(t)));
        for (a, b, road) in EDGES {
            if a == b {
                continue;
            }
            let (pa, pb) = (at(a), at(b));
            let d = (pb - pa).normalize_or_zero();
            let (p, q) = (pa + d * (radius + 3.0), pb - d * (radius + 5.0));
            if (q - p).dot(d) <= 4.0 {
                continue;
            }
            match road {
                Road::Asked if a == V::Vaccine => match toy {
                    Some(t) if t.theta >= 1.0 => hud_dashed_arrow(f, p, q, 1.3, pal::DIM.a(0.6)),
                    Some(_) => hud_arrow(f, p, q, 2.6, TRUTH),
                    None => hud_arrow(f, p, q, 2.2, TRUTH.mix(pal::DIM, 0.25)),
                },
                Road::Asked => hud_arrow(f, p, q, 2.0, if a == V::Dementia { EFFECT } else { CAUSE }),
                Road::Matched => hud_arrow(f, p, q, 1.4, pal::DIM.a(0.75)),
                _ if lit(road) => hud_arrow(f, p, q, 2.6, THIRD),
                _ => hud_dashed_arrow(f, p, q, 1.3, THIRD.a(if toy.is_some() { 0.35 } else { 0.8 })),
            }
        }
        for (v, _, _, letter, _, held) in VARS {
            let p = at(v);
            let color = var_color(v);
            hud_fill(f, 0, p - Vec2::splat(radius), p + Vec2::splat(radius), Color::hex(0x0B0F1A).a(0.9));
            hud_circle(f, p, radius, 1.8, color);
            if held {
                hud_outline(f, p - Vec2::splat(radius + 4.0), p + Vec2::splat(radius + 4.0), 1.2, color);
            }
            f.text_px(p, format!("${letter}$"), (radius * 0.95).max(11.0), color, Align2::CENTER_CENTER);
        }
        // the effect asked about, named on its arrow
        let (pt, pd) = (at(V::Vaccine), at(V::Dementia));
        let label = match toy {
            None => "effect asked about".to_string(),
            Some(t) if t.theta >= 1.0 => "none".to_string(),
            Some(t) if t.delay > 0.0 => format!(r"$\theta = {}$, delayed", num(t.theta, 2)),
            Some(t) => format!(r"$\theta = {}$", num(t.theta, 2)),
        };
        let room = (pd - pt).length() - 2.0 * radius - 8.0;
        let label = pick(f, &[&label, &label.replace(", delayed", ""), ""], room, size);
        let color = if toy.is_none_or(|t| t.theta < 1.0) { TRUTH } else { pal::DIM };
        f.text_px((pt + pd) * 0.5 - Vec2::new(0.0, 5.0), label, size, color, Align2::CENTER_BOTTOM);
        // in the toy, the open roads named beside their variables
        if let Some(t) = toy {
            for (v, dir) in [
                (V::Unrecorded, Vec2::new(1.0, 0.0)),
                (V::Leaving, Vec2::new(1.0, 0.0)),
                (V::LookBack, Vec2::new(0.0, -1.0)),
                (V::Calendar, Vec2::new(1.0, 0.0)),
            ] {
                let road = EDGES.iter().find(|e| e.0 == V::Era && e.1 == v).map(|e| e.2);
                let on: Vec<&str> = road.map(|r| r.mechanisms().iter().filter(|m| m.on(t)).map(|m| m.short()).collect()).unwrap_or_default();
                if on.is_empty() {
                    continue;
                }
                let p = at(v) + dir * (radius + 8.0);
                let align = if dir.x > 0.0 { Align2::LEFT_CENTER } else { Align2::CENTER_BOTTOM };
                f.caption_px(p, on.join(", "), size, THIRD, align);
            }
        }
    }

    /// What each letter of the graph is, in the letter's colour.
    fn legend() -> String {
        VARS.iter().map(|(v, _, _, letter, what, _)| format!("${}$ {what}", tint(var_color(*v), letter))).collect::<Vec<_>>().join(" · ")
    }

    /// The calendar: calendar time across, time since the first dose up, so that each
    /// person climbs a diagonal. `windows` and `years`: the cohorts drawn; `toy`: shade
    /// the toy's pandemic; `full`: every label (the opening picture).
    #[allow(clippy::too_many_arguments)]
    fn lexis(&mut self, f: &mut Frame, r: Rect, lay: &Layout, windows: [Window; 2], years: f64, toy: Option<&Toy>, full: bool) {
        if !r.shows() {
            return;
        }
        let (small, text) = (lay.small, lay.text);
        let (c0, c1, top) = (2014.0f64, 2027.0f64, 6.5f64);
        let x = |c: f64| r.min.x + ((c - c0) / (c1 - c0)) as f32 * r.w();
        let y = |t: f64| r.max.y - (t / top) as f32 * r.h();
        let px = |c: f64, t: f64| Vec2::new(x(c), y(t));
        let per_year = r.w() / (c1 - c0) as f32;
        // the grid: each year, and each year of follow-up
        for k in 0..=13 {
            let c = c0 + k as f64;
            hud_line(f, px(c, 0.0), px(c, top), 1.0, pal::FAINT.a(0.25));
        }
        for t in 0..=6 {
            hud_line(f, px(c0, t as f64), px(c1, t as f64), 1.0, pal::FAINT.a(if t == 0 { 0.8 } else { 0.25 }));
        }
        // the toy's pandemic
        if let Some(t) = toy.filter(|t| t.pandemic > 0.0) {
            hud_fill(f, 0, px(PANDEMIC, top), px(PANDEMIC_END, 0.0), THIRD.a((0.08 + 0.3 * t.pandemic) as f32));
        }
        // the cohorts: what the data show, and the rest of the six years, not yet seen
        let end = x(DATA_END);
        for (window, color) in windows.into_iter().zip([EARLY, LATE]) {
            let band = [px(window.from, 0.0), px(window.to(), 0.0), px(window.to() + years, years), px(window.from + years, years)];
            hud_shape(f, 0, &clip_x(&band, end, true), color.a(0.22));
            hud_shape(f, 0, &clip_x(&band, end, false), color.a(0.06));
            let seen = clip_x(&band, end, true);
            for i in 0..seen.len() {
                hud_line(f, seen[i], seen[(i + 1) % seen.len()], 1.2, color.a(0.8));
            }
        }
        // counting starts at three months
        hud_dashed(f, px(c0, 0.25), px(c1, 0.25), 1.0, pal::DIM.a(0.8));
        // the switch (the month left out)
        hud_fill(f, 0, px(month(2017, 10), top), px(month(2017, 11), 0.0), pal::TEXT.a(0.12));
        hud_line(f, px(month(2017, 10) + 1.0 / 24.0, 0.0), px(month(2017, 10) + 1.0 / 24.0, top), 1.4, pal::TEXT.a(0.6));
        // the pandemic, and where it falls in each cohort's follow-up
        hud_line(f, px(PANDEMIC, 0.0), px(PANDEMIC, top), 1.6, THIRD);
        let (e0, e1) = (PANDEMIC - windows[0].to(), PANDEMIC - windows[0].from);
        let (l0, l1) = ((PANDEMIC - windows[1].to()).max(0.0), PANDEMIC - windows[1].from);
        if e1 > 0.0 {
            hud_line(f, px(PANDEMIC, e0.max(0.0).min(years)), px(PANDEMIC, e1.min(years)), 5.0, THIRD.a(0.85));
        }
        if l1 > 0.0 {
            hud_line(f, px(PANDEMIC, l0.min(years)), px(PANDEMIC, l1.min(years)), 5.0, THIRD.a(0.85));
        }
        // the end of the data
        hud_dashed(f, px(DATA_END, 0.0), px(DATA_END, top), 1.4, pal::TEXT.a(0.8));
        // the axes' numbers
        let every = if per_year >= 34.0 {
            1
        } else if per_year >= 17.0 {
            2
        } else {
            4
        };
        for k in (0..=13).step_by(every) {
            let c = c0 + k as f64;
            if k < 13 {
                f.text_px(px(c + 0.5, 0.0) + Vec2::new(0.0, 4.0), format!("{}", c as i32), small, pal::DIM, Align2::CENTER_TOP);
            }
        }
        for t in 0..=6 {
            f.text_px(px(c0, t as f64) - Vec2::new(6.0, 0.0), t.to_string(), small, pal::DIM, Align2::RIGHT_CENTER);
        }
        if full {
            f.text_px(Vec2::new(r.min.x - 6.0, r.min.y - 4.0), "years since first dose", small, pal::DIM, Align2::LEFT_BOTTOM);
        }
        // the first doses of each cohort, under the axis
        let under = r.max.y + small + 10.0;
        for (k, (window, color)) in windows.into_iter().zip([EARLY, LATE]).enumerate() {
            let (a, b) = (x(window.from), x(window.to()));
            hud_line(f, Vec2::new(a, under), Vec2::new(b, under), 2.0, color);
            hud_line(f, Vec2::new(a, under - 4.0), Vec2::new(a, under + 4.0), 2.0, color);
            hud_line(f, Vec2::new(b, under - 4.0), Vec2::new(b, under + 4.0), 2.0, color);
            let words: Vec<String> = if full {
                if k == 0 {
                    vec![
                        format!("first doses Oct 2014 – Sep 2017: {} % live vaccine", LIVE_BEFORE),
                        format!("Oct 2014 – Sep 2017: {} % live", LIVE_BEFORE),
                        "earlier".into(),
                    ]
                } else {
                    vec![
                        format!("Nov 2017 – Oct 2020: {} % recombinant", RECOMBINANT_AFTER),
                        format!("Nov 2017 – Oct 2020: {} % recomb.", RECOMBINANT_AFTER),
                        "later".into(),
                    ]
                }
            } else if k == 0 {
                vec!["earlier cohort's first doses".into(), "earlier".into()]
            } else {
                vec!["later cohort's first doses".into(), "later".into()]
            };
            let words: Vec<&str> = words.iter().map(String::as_str).collect();
            if r.w() < 300.0 {
                continue;
            }
            let room = if k == 0 { x(windows[1].from) - a - 8.0 } else { r.max.x - a };
            let say = pick(f, &words, room, small);
            f.text_px(Vec2::new(a, under + 5.0), say, small, color, Align2::LEFT_TOP);
        }
        if r.w() < 300.0 {
            // (too small for words: the bands, the lines and the numbers only)
            return;
        }
        // the labels
        let lines = small + 4.0;
        let switch_label =
            pick(f, &["October 2017: the switch (that month left out)", "Oct 2017: the switch", "switch"], x(PANDEMIC) - x(month(2017, 10)) + 40.0, small);
        f.caption_px(px(month(2017, 10), top) + Vec2::new(0.0, 2.0), switch_label, small, pal::TEXT, Align2::RIGHT_TOP);
        let pandemic_label = if toy.is_some_and(|t| t.pandemic > 0.0) { "toy pandemic: fewer diagnoses for 2 years" } else { "March 2020: the pandemic" };
        let pandemic_label = pick(f, &[pandemic_label, "March 2020: pandemic", "pandemic"], x(DATA_END) - x(PANDEMIC) - 6.0, small);
        f.caption_px(px(PANDEMIC, top) + Vec2::new(4.0, 2.0), pandemic_label, small, THIRD, Align2::LEFT_TOP);
        let room = lay.view.x - x(DATA_END) - 10.0;
        let end_label = if full {
            pick(
                f,
                &[
                    "end of the data: not stated; drawn at March 2024, the date of the reporting summary",
                    "end of the data: not stated (drawn at March 2024)",
                    "end of the data",
                ],
                room,
                small,
            )
        } else {
            pick(f, &["toy: data end March 2024", "data end"], x(c1) - x(DATA_END) - 4.0, small)
        };
        if full {
            // (on as many as four lines where one is too short)
            let long = "end of the data: not stated; drawn at March 2024, when the reporting summary was last updated";
            let broken = self.wrap.lines(f, long, room.max(40.0), small);
            let said: Vec<String> = if f.text_size(&end_label, small).x <= room && end_label != "end of the data" {
                vec![end_label]
            } else if broken.len() <= 4 {
                broken
            } else {
                vec!["end of the data".into(), "(not stated)".into()]
            };
            for (k, l) in said.into_iter().enumerate() {
                f.caption_px(px(DATA_END, top) + Vec2::new(4.0, 2.0 + (k + 1) as f32 * lines), l, small, pal::TEXT, Align2::LEFT_TOP);
            }
        } else {
            f.caption_px(px(DATA_END, 1.0) + Vec2::new(4.0, 0.0), end_label, small, pal::TEXT, Align2::LEFT_CENTER);
        }
        if full {
            // where the pandemic falls (the dossier's numbers, from the cohort dates)
            let e_mid = px(PANDEMIC, 0.5 * (e0 + e1).min(years));
            let e_text = pick(
                f,
                &[
                    &format!("{} to {} years after the first dose", num(PANDEMIC_EARLIER.0, 1), num(PANDEMIC_EARLIER.1, 1)),
                    &format!("{}–{} years in", num(PANDEMIC_EARLIER.0, 1), num(PANDEMIC_EARLIER.1, 1)),
                ],
                x(PANDEMIC) - r.min.x - 10.0,
                small,
            );
            f.caption_px(e_mid - Vec2::new(8.0, 0.0), e_text, small, THIRD, Align2::RIGHT_CENTER);
            let l_mid = px(PANDEMIC, 0.5 * (l0 + l1).min(years));
            let l_text = pick(
                f,
                &[
                    &format!("up to {} years after the first dose, or before it", num(PANDEMIC_LATER.0, 1)),
                    &format!("first {} years, or before", num(PANDEMIC_LATER.0, 1)),
                ],
                r.max.x - x(PANDEMIC) - 12.0,
                small,
            );
            f.caption_px(l_mid + Vec2::new(8.0, 0.0), l_text, small, THIRD, Align2::LEFT_CENTER);
            // the cohorts' names and how long they are followed
            let mid_e = px(0.5 * (windows[0].from + windows[0].to()) + 1.0, 1.0);
            f.text_px(mid_e, "earlier cohort", text, EARLY, Align2::CENTER_CENTER);
            let mid_l = px(0.5 * (windows[1].from + windows[1].to()) + 3.0, 3.0);
            f.text_px(mid_l, "later cohort", text, LATE, Align2::CENTER_CENTER);
            let e_med = format!("median follow-up {} years", num(MEDIAN_EARLIER.value, 1));
            let e_short = format!("median {} years", num(MEDIAN_EARLIER.value, 1));
            let e_say = pick(f, &[&e_med, &e_short, &format!("{} y", num(MEDIAN_EARLIER.value, 1))], 3.8 * per_year, small);
            f.caption_px(px(0.5 * (windows[0].from + windows[0].to()) + 5.5, 5.5), e_say, small, EARLY, Align2::CENTER_CENTER);
            let l_med = format!("median {} years", num(MEDIAN_LATER.value, 2));
            f.caption_px(
                px(DATA_END, 4.6) + Vec2::new(-6.0, 0.0),
                pick(f, &[&l_med, &format!("{} y", num(MEDIAN_LATER.value, 2))], 3.8 * per_year, small),
                small,
                LATE,
                Align2::RIGHT_CENTER,
            );
            // one person
            let (born, t1) = (2015.5, 4.6);
            hud_dashed(f, px(born, 0.0), px(born + t1, t1), 1.2, pal::WHITE.a(0.85));
            hud_circle(f, px(born, 0.0), 3.0, 2.0, pal::WHITE);
            let who = pick(f, &["one person: a year older, a year further on the calendar", "one person"], 3.2 * per_year, small);
            f.caption_px(px(born + t1, t1) + Vec2::new(-4.0, -2.0), who, small, pal::WHITE, Align2::RIGHT_BOTTOM);
            // the diagnosis codes changed in October 2015
            hud_line(f, px(ICD10, 0.0), px(ICD10, 0.4), 2.0, pal::DIM);
            if per_year >= 60.0 {
                let icd = pick(f, &["ICD-10 codes from Oct 2015", "ICD-10"], 2.0 * per_year, small);
                f.text_px(px(ICD10, 0.4) + Vec2::new(3.0, -2.0), icd, small, pal::DIM, Align2::LEFT_BOTTOM);
            }
            // the part not yet seen
            if r.max.x - end > 60.0 {
                f.text_px(px(DATA_END + 0.15, 2.0), pick(f, &["not yet seen", ""], r.max.x - end - 8.0, small), small, pal::DIM, Align2::LEFT_CENTER);
            }
        }
        f.text_px(
            px(c1, 0.25) + Vec2::new(-4.0, -2.0),
            pick(f, &["counted from 3 months", "3 months"], 4.0 * per_year, small),
            small,
            pal::DIM,
            Align2::RIGHT_BOTTOM,
        );
    }

    // ----- the comparison ------------------------------------------------------------

    fn draw_design(&mut self, f: &mut Frame, lay: &Layout) {
        let (text, small) = (lay.text, lay.small);
        let title = pick(
            f,
            &[
                "Who is compared: people first vaccinated at 65 or over, in the three years before the switch and the three after",
                "First vaccinated at 65 or over: three years before the switch, three after",
                "The two cohorts on the calendar",
            ],
            lay.lexis.max.x - lay.lexis.min.x + 50.0,
            text,
        );
        if lay.lexis.shows() && !lay.bare {
            f.text_px(
                Vec2::new(lay.lexis.min.x - 40.0f32.min(lay.lexis.min.x - 4.0), lay.lexis.min.y - small - 14.0),
                title,
                text,
                pal::TEXT,
                Align2::LEFT_BOTTOM,
            );
        }
        self.lexis(f, lay.lexis, lay, [EARLIER, LATER], YEARS as f64, None, true);
        if lay.bare {
            return;
        }
        // the graph, and what it says
        let g = lay.graph;
        if g.shows() {
            let head = pick(f, &["The comparison as a causal graph", "As a graph"], g.w(), small + 1.0);
            f.text_px(Vec2::new(g.min.x, g.min.y - 4.0), head, small + 1.0, pal::TEXT, Align2::LEFT_BOTTOM);
            self.graph(f, g, small, None);
            let key = if lay.wide {
                format!("{}. Solid: the road the comparison is meant to measure; dashed: roads it needs absent; boxed: held fixed (matched on, or counted only while in the records).", Self::legend())
            } else {
                format!("{}. Dashed: needed absent; boxed: held fixed.", Self::legend())
            };
            let l = lay.legend;
            self.wrap.draw(f, l.min, l.w(), l.max.y, &key, small, pal::DIM);
        }
        let w = lay.words;
        if w.shows() {
            let paragraphs = self.design_words(lay.wide);
            let mut y = w.min.y;
            for (i, p) in paragraphs.iter().enumerate() {
                let color = if i == 0 { pal::TEXT } else { pal::DIM };
                y += self.wrap.draw(f, Vec2::new(w.min.x, y), w.w(), w.max.y, p, small, color) + 6.0;
            }
        }
    }

    /// The words beside the graph (under it on a narrow canvas, where they are shorter).
    fn design_words(&self, wide: bool) -> Vec<String> {
        if !wide {
            return vec![format!(
                "Compared: first vaccinated at 65 or over, three years either side of the switch, matched on {} recorded characteristics; counted, a recorded dementia diagnosis from 3 months to 6 years. The claim: once matched, the period changes the diagnosis only through the vaccine; each dashed road would carry a difference with no vaccine effect.",
                COVARIATES
            )];
        }
        let mut out = vec![format!(
            "{} people first vaccinated against shingles at 65 or over in the three years after the switch, and as many in the three years before, matched on {} recorded characteristics, in the health records of {} US health-care organisations; counted, a first recorded dementia diagnosis from 3 months to 6 years after the first dose. Matching kept {} % of the {} earlier vaccinees and {} % of the {} later ones (calculated from Supplementary Table 1).",
            count(MATCHED),
            COVARIATES,
            ORGANISATIONS,
            num(100.0 * MATCHED as f64 / POOL_EARLIER as f64, 1),
            count(POOL_EARLIER),
            num(100.0 * MATCHED as f64 / POOL_LATER as f64, 1),
            count(POOL_LATER)
        )];
        out.push("The claim to a causal reading: once those characteristics are matched, being vaccinated just after rather than just before October 2017 changes the recorded diagnosis only through which vaccine was given. Every dashed road would carry a difference with no vaccine effect at all.".into());
        out.push(format!(
            "Besides the vaccine, calendar time differs. The pandemic falls early in the later cohort's follow-up and late in the earlier cohort's; the later cohort is followed for less long (median {} against {} years); whatever changed in diagnosis and coding over the decade falls on the two differently.",
            num(MEDIAN_LATER.value, 2),
            num(MEDIAN_EARLIER.value, 1)
        ));
        if wide {
            out.push(format!("Found: a ratio of restricted mean time lost of {}, later over earlier (see “what the paper found”).", TABLE1[PRIMARY].1.text(2)));
        }
        out
    }

    // ----- what the paper found --------------------------------------------------------

    fn draw_paper(&mut self, f: &mut Frame, lay: &Layout) {
        let (text, small) = (lay.text, lay.small);
        let below_forest = self.forest(f, lay);
        if lay.bare {
            return;
        }
        // the share still without a diagnosis: the area between it and 100 % is the time lost
        let r = lay.survival;
        if r.shows() {
            let title = pick(f, &["Still without a recorded diagnosis: the paper's two cohorts", "Still without a diagnosis"], r.w() + 40.0, text);
            f.text_px(Vec2::new(r.min.x - 40.0, r.min.y - small - 12.0), title, text, pal::TEXT, Align2::LEFT_BOTTOM);
            let sub =
                pick(f, &["the authors' posted points, joined by straight lines; shaded: time lost", "posted points; shaded: time lost"], r.w() + 40.0, small);
            f.text_px(Vec2::new(r.min.x - 40.0, r.min.y - 6.0), sub, small, pal::DIM, Align2::LEFT_BOTTOM);
            let lowest = 93.0;
            let x = |t: f64| r.min.x + (t / 6.0) as f32 * r.w();
            let y = |p: f64| r.min.y + ((100.0 - p) / (100.0 - lowest)) as f32 * r.h();
            for p in [94.0, 96.0, 98.0, 100.0] {
                hud_line(f, Vec2::new(r.min.x, y(p)), Vec2::new(r.max.x, y(p)), 1.0, pal::FAINT.a(if p == 100.0 { 0.9 } else { 0.35 }));
                f.text_px(Vec2::new(r.min.x - 6.0, y(p)), format!("{} %", p as i32), small, pal::DIM, Align2::RIGHT_CENTER);
            }
            for t in 0..=6 {
                f.text_px(Vec2::new(x(t as f64), r.max.y + 3.0), t.to_string(), small, pal::DIM, Align2::CENTER_TOP);
            }
            f.text_px(Vec2::new(r.max.x, r.max.y + small + 6.0), "years after the first dose", small, pal::DIM, Align2::RIGHT_TOP);
            hud_dashed(f, Vec2::new(x(0.25), r.min.y), Vec2::new(x(0.25), r.max.y), 1.0, pal::DIM.a(0.6));
            let m = &DEMENTIA_MARKS;
            for (values, color) in [(m.earlier, EARLY), (m.later, LATE)] {
                let mut pts = vec![Vec2::new(x(0.25), y(100.0))];
                pts.extend(m.years.iter().zip(values).map(|(t, v)| Vec2::new(x(*t), y(100.0 - v))));
                for w in pts.windows(2) {
                    hud_shape(f, 0, &[w[0], w[1], Vec2::new(w[1].x, y(100.0)), Vec2::new(w[0].x, y(100.0))], color.a(0.16));
                }
                hud_path(f, pts.iter().copied(), 2.0, color);
                for p in &pts[1..] {
                    hud_circle(f, *p, 3.0, 1.6, color);
                }
            }
            // the areas, in days
            let (before, after) = rmtl_days();
            let e_text = format!("earlier: about {} days lost per person (the cyan area and the grey band)", num(before, 1));
            let l_text = format!("later: about {} days lost per person", num(after, 1));
            let e_mid = format!("earlier: about {} days lost per person", num(before, 1));
            f.caption_px(
                Vec2::new(x(6.0) - 4.0, y(100.0 - 5.94) + 6.0),
                pick(f, &[&e_text, &e_mid, &format!("{} days", num(before, 1))], r.w() * 0.95, small),
                small,
                EARLY,
                Align2::RIGHT_TOP,
            );
            f.caption_px(
                Vec2::new(x(4.2), y(100.0 - 1.6)),
                pick(f, &[&l_text, &format!("{} days", num(after, 1))], r.w() * 0.5, small),
                small,
                LATE,
                Align2::CENTER_CENTER,
            );
            f.caption_px(
                Vec2::new(x(0.4), y(lowest) - 4.0),
                format!("ratio of the areas: {}", num(TABLE1[PRIMARY].1.value, 2)),
                small,
                PUBLISHED,
                Align2::LEFT_BOTTOM,
            );
        }
        // the rate ratio year by year
        let r = lay.years;
        if r.shows() {
            let title = pick(
                f,
                &["Crude rate ratio in each year after the first dose, later over earlier", "Crude rate ratio, each year", "Rate ratio by year"],
                r.w() + 40.0,
                if lay.wide { text } else { small + 1.0 },
            );
            let sub = pick(
                f,
                &[
                    "calculated from the paper's Supplementary Table 2; the paper does not report these ratios",
                    "calculated from Suppl. Table 2 (not reported as such)",
                    "from Suppl. Table 2",
                ],
                r.w() + 40.0,
                small,
            );
            f.text_px(Vec2::new(r.min.x - 40.0, r.min.y - small - 12.0), title, if lay.wide { text } else { small + 1.0 }, pal::TEXT, Align2::LEFT_BOTTOM);
            f.text_px(Vec2::new(r.min.x - 40.0, r.min.y - 6.0), sub, small, pal::DIM, Align2::LEFT_BOTTOM);
            let (lo, hi) = (0.55f64, 1.25f64);
            let ybar = |v: f64| r.max.y - ((v - lo) / (hi - lo)) as f32 * r.h();
            for v in [0.75, 1.0, 1.25] {
                hud_line(f, Vec2::new(r.min.x, ybar(v)), Vec2::new(r.max.x, ybar(v)), 1.0, pal::FAINT.a(if v == 1.0 { 0.9 } else { 0.35 }));
                f.text_px(Vec2::new(r.min.x - 6.0, ybar(v)), num(v, 2), small, pal::DIM, Align2::RIGHT_CENTER);
            }
            let slot = r.w() / YEARS as f32;
            for (k, ratio) in CRUDE_RATIOS.iter().enumerate() {
                let cx = r.min.x + (k as f32 + 0.5) * slot;
                let (a, b) = (ybar(1.0), ybar(*ratio));
                hud_fill(f, 1, Vec2::new(cx - 0.3 * slot, a.min(b)), Vec2::new(cx + 0.3 * slot, a.max(b).max(a.min(b) + 1.0)), PUBLISHED.a(0.75));
                let above = *ratio >= 1.0;
                f.text_px(
                    Vec2::new(cx, if above { b - 3.0 } else { b + 3.0 }),
                    num(*ratio, 2),
                    small,
                    PUBLISHED,
                    if above { Align2::CENTER_BOTTOM } else { Align2::CENTER_TOP },
                );
                f.text_px(Vec2::new(cx, r.max.y + 3.0), format!("year {}", k + 1), small, pal::DIM, Align2::CENTER_TOP);
                // how much each year rests on: person-years, earlier then later
                let [e, l] = SUPPL_TABLE_2[k];
                f.text_px(Vec2::new(cx, r.max.y + small + 7.0), short_count(e.0), small - 0.5, EARLY, Align2::CENTER_TOP);
                f.text_px(Vec2::new(cx, r.max.y + 2.0 * small + 11.0), short_count(l.0), small - 0.5, LATE, Align2::CENTER_TOP);
            }
            let py = pick(
                f,
                &[
                    "thousands of person-years in each year, earlier (grey) and later (cyan): the later cohort's last years rest on few people",
                    "person-years, earlier and later (thousands)",
                    "person-years",
                ],
                r.w() + 40.0,
                small - 0.5,
            );
            f.text_px(Vec2::new(r.min.x - 40.0, r.max.y + 3.0 * small + 16.0), py, small - 0.5, pal::DIM, Align2::LEFT_TOP);
        }
        // on a narrow canvas, the sum behind the paper's 164 days, between the forest and the years
        if !lay.wide && lay.years.shows() {
            let says = format!(
                "The paper's “{} additional diagnosis-free days among those affected” is {} days per person divided by the {} % of the earlier cohort diagnosed by six years (calculated from the posted data).",
                DAYS_AMONG_AFFECTED.value as i32,
                num(RMST_DIFFERENCE_DAYS.value, 2),
                num(DEMENTIA_MARKS.earlier[4], 2)
            );
            let top = below_forest + 2.0;
            let bottom = lay.years.min.y - 2.0 * small - 14.0;
            self.wrap.draw(f, Vec2::new(lay.forest.min.x, top), lay.forest.w(), bottom, &says, small, pal::DIM);
        }
        // the sums, and the time course in the paper's words
        let s = lay.sums;
        if s.shows() {
            let (before, after) = rmtl_days();
            let window = WINDOW_DAYS.1 - WINDOW_DAYS.0;
            let paragraphs = [
                format!(
                    "Restricted mean time lost: over the window from 3 months to 6 years (day {} to day {}), the area between a cohort's curve and 100 %, which is the average time spent with a diagnosis. About {} days in the earlier cohort and {} in the later, of about {}; their ratio is the paper's {} ({}).",
                    WINDOW_DAYS.0 as i32,
                    WINDOW_DAYS.1 as i32,
                    num(before, 1),
                    num(after, 1),
                    count(((window / 100.0).round() * 100.0) as usize),
                    num(TABLE1[PRIMARY].1.value, 2),
                    "95 % CI 0.79–0.87"
                ),
                format!(
                    "Per person the difference is {} days without a diagnosis ({} % of the window). The paper's “{} additional diagnosis-free days among those affected” is the same difference divided by the {} % of the earlier cohort diagnosed by six years: $9.75 / 0.0594 \\approx 164$. (Days calculated from the authors' posted data.)",
                    num(RMST_DIFFERENCE_DAYS.value, 2),
                    num(100.0 * RMST_DIFFERENCE_DAYS.value / window, 1),
                    DAYS_AMONG_AFFECTED.value as i32,
                    num(DEMENTIA_MARKS.earlier[4], 2)
                ),
                format!("By six years the curves nearly meet: {} % against {} %, where the later cohort's estimate rests on few people.", num(DEMENTIA_MARKS.later[4], 2), num(DEMENTIA_MARKS.earlier[4], 2)),
                TIME_COURSE.to_string(),
            ];
            let mut y = below_forest.max(lay.forest.min.y) + 16.0;
            for (i, p) in paragraphs.iter().enumerate() {
                y += self.wrap.draw(f, Vec2::new(s.min.x, y), s.w(), s.max.y, p, small, if i == 3 { pal::TEXT } else { pal::DIM }) + 6.0;
            }
        }
    }

    /// Table 1 as a forest: every ratio of time lost on one scale, the hazard ratio of
    /// the check before the pandemic apart from it.
    fn forest(&mut self, f: &mut Frame, lay: &Layout) -> f32 {
        let r = lay.forest;
        if !r.shows() {
            return r.min.y;
        }
        let small = lay.small;
        let mut rows: Vec<usize> = if lay.wide { (0..TABLE1.len()).collect() } else { TABLE1_SHORT.to_vec() };
        if lay.bare {
            // (a window too small for every row: the first that fit)
            let fit = ((r.h() - small - 12.0) / (small + 2.0)).max(1.0) as usize;
            rows.truncate(fit);
        }
        let title = pick(
            f,
            &[
                "Ratio of restricted mean time lost, later over earlier, 95 % intervals (Table 1, page 2779)",
                "Ratio of time lost, later ÷ earlier (Table 1)",
                "Table 1",
            ],
            r.w(),
            if lay.wide { lay.text } else { small + 1.0 },
        );
        f.text_px(Vec2::new(r.min.x, r.min.y - 6.0), title, if lay.wide { lay.text } else { small + 1.0 }, pal::TEXT, Align2::LEFT_BOTTOM);
        let label_w = rows.iter().map(|&i| f.text_size(TABLE1[i].0, small).x).fold(0.0f32, f32::max) + 10.0;
        let value_w = f.text_size("0.00 (0.00–0.00)", small).x + 10.0;
        let (ax0, ax1) = (r.min.x + label_w.min(r.w() * 0.45), r.max.x - value_w);
        let (lo, hi) = (0.6f64, 1.1f64);
        let xv = |v: f64| ax0 + ((v - lo) / (hi - lo)) as f32 * (ax1 - ax0);
        let extra = if lay.bare { 0.0 } else { 2.0 * (small + 5.0) + 8.0 };
        let row_h = ((r.h() - small - 8.0 - extra) / rows.len() as f32).clamp(10.0, 22.0);
        let y0 = r.min.y + 6.0;
        // (every tenth numbered where there is room, every fifth where there is not)
        let every = if (ax1 - ax0) / 5.0 >= 28.0 { 1 } else { 2 };
        for (k, v) in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1].into_iter().enumerate() {
            let x = xv(v);
            hud_line(f, Vec2::new(x, y0), Vec2::new(x, y0 + rows.len() as f32 * row_h), 1.0, pal::FAINT.a(if v == 1.0 { 0.9 } else { 0.3 }));
            if k % every == 0 {
                f.text_px(Vec2::new(x, y0 + rows.len() as f32 * row_h + 3.0), num(v, 1), small, pal::DIM, Align2::CENTER_TOP);
            }
        }
        for (n, &i) in rows.iter().enumerate() {
            let (name, fg) = TABLE1[i];
            let cy = y0 + (n as f32 + 0.5) * row_h;
            let color = if i == PRIMARY { PUBLISHED } else { PUBLISHED.mix(pal::DIM, 0.25) };
            f.text_px(Vec2::new(ax0 - 8.0, cy), name, small, if i == PRIMARY { pal::TEXT } else { pal::DIM }, Align2::RIGHT_CENTER);
            hud_line(f, Vec2::new(xv(fg.low), cy), Vec2::new(xv(fg.high), cy), 1.6, color);
            let d = if i == PRIMARY { 5.0 } else { 3.8 };
            hud_shape(
                f,
                1,
                &[Vec2::new(xv(fg.value), cy - d), Vec2::new(xv(fg.value) + d, cy), Vec2::new(xv(fg.value), cy + d), Vec2::new(xv(fg.value) - d, cy)],
                color,
            );
            f.text_px(Vec2::new(r.max.x, cy), fg.text(2), small, color, Align2::RIGHT_CENTER);
        }
        if !lay.bare {
            let y = y0 + rows.len() as f32 * row_h + small + 10.0;
            let hr = format!(
                "before the pandemic (six-month windows, 18 months): hazard ratio {}, a different measure, so not on this scale",
                BEFORE_PANDEMIC_HR.text(2)
            );
            let hr_short = format!("before the pandemic: hazard ratio {} (another measure)", BEFORE_PANDEMIC_HR.text(2));
            let say = pick(f, &[&hr, &hr_short, &format!("pre-pandemic HR {}", BEFORE_PANDEMIC_HR.text(2))], r.w(), small);
            f.text_px(Vec2::new(r.min.x, y), say, small, PUBLISHED, Align2::LEFT_TOP);
            let sex = format!(
                "Women {} % and men {} % more time diagnosis-free (permutation test $P = {}$).",
                num(MORE_TIME_WOMEN_MEN.0, 0),
                num(MORE_TIME_WOMEN_MEN.1, 0),
                MORE_TIME_WOMEN_MEN.2
            );
            if lay.wide {
                f.text_px(Vec2::new(r.min.x, y + small + 5.0), sex, small, pal::DIM, Align2::LEFT_TOP);
                return y + 2.0 * (small + 5.0);
            }
            return y + small + 5.0;
        }
        y0 + rows.len() as f32 * row_h + small + 6.0
    }

    // ----- the toy model ----------------------------------------------------------------

    fn draw_toy(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let (text, small) = (lay.text, lay.small);
        // the frame that says what this is
        let fr = lay.frame;
        if fr.shows() {
            hud_dashed(f, fr.min, Vec2::new(fr.max.x, fr.min.y), 1.4, TOY_FRAME.a(0.8));
            hud_dashed(f, Vec2::new(fr.max.x, fr.min.y), fr.max, 1.4, TOY_FRAME.a(0.8));
            hud_dashed(f, fr.max, Vec2::new(fr.min.x, fr.max.y), 1.4, TOY_FRAME.a(0.8));
            hud_dashed(f, Vec2::new(fr.min.x, fr.max.y), fr.min, 1.4, TOY_FRAME.a(0.8));
            let says = pick(
                f,
                &[
                    "TOY MODEL: two made-up cohorts, not the study's data; worked out exactly, with no sampling. Yellow marks are the paper's, for reference.",
                    "TOY MODEL: made-up numbers, not the study's data; yellow: the paper's",
                    "TOY MODEL: made-up, not the study's data",
                    "TOY MODEL",
                ],
                fr.w() - 16.0,
                if lay.wide { text } else { small },
            );
            f.caption_px(fr.min + Vec2::new(8.0, 6.0), says, if lay.wide { text } else { small }, TOY_FRAME, Align2::LEFT_TOP);
        }
        let check = self.check;
        let design = check.design();
        if lay.mini.shows() {
            let head = pick(
                f,
                &["Toy cohorts on the calendar (up: years since the first dose)", "Toy cohorts on the calendar", "Toy cohorts"],
                lay.mini.w() + 30.0,
                small,
            );
            f.text_px(Vec2::new(lay.mini.min.x - 34.0, lay.mini.min.y - 6.0), head, small, pal::TEXT, Align2::LEFT_BOTTOM);
            let toy = self.toy;
            self.lexis(f, lay.mini, lay, design.windows, design.horizon as f64 / PER_YEAR as f64, Some(&toy), false);
        }
        if lay.toy_graph.shows() {
            let toy = self.toy;
            f.text_px(
                Vec2::new(lay.toy_graph.min.x, lay.toy_graph.min.y - 4.0),
                pick(f, &["Toy: which roads are open (solid purple)", "Toy: open roads"], lay.toy_graph.w(), small),
                small,
                pal::TEXT,
                Align2::LEFT_BOTTOM,
            );
            self.graph(f, lay.toy_graph, small - 0.5, Some(&toy));
            let l = lay.toy_legend;
            self.wrap.draw(f, l.min, l.w(), l.max.y, &Self::legend(), small - 0.5, pal::DIM);
        }
        if lay.mechs.w() > 20.0 && lay.mechs.h() > 8.0 {
            let on: Vec<String> = MECHANISMS.iter().filter(|m| m.on(&self.toy)).map(|m| format!("{} ({})", m.short(), m.strength(&self.toy))).collect();
            let says = if on.is_empty() { "Mechanisms on: none (switch them in the panel)".to_string() } else { format!("Mechanisms on: {}", on.join("; ")) };
            let m = lay.mechs;
            self.wrap.draw(f, m.min, m.w(), m.max.y, &says, small, pal::TEXT);
        }
        if lay.switches.shows() {
            self.switches(f, lay, pointer);
        }
        self.curves(f, lay);
        self.ratios(f, lay);
        self.strip(f, lay);
        self.table(f, lay, pointer);
    }

    /// The mechanisms, each a switch: a click turns it on or off.
    fn switches(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let r = lay.switches;
        let small = lay.small;
        f.text_px(
            Vec2::new(r.min.x, r.min.y - 4.0),
            pick(f, &["Mechanisms (click to switch; strengths in the panel)", "Mechanisms"], r.w(), small),
            small,
            pal::TEXT,
            Align2::LEFT_BOTTOM,
        );
        let row = ((r.h()) / MECHANISMS.len() as f32).clamp(12.0, 24.0);
        for (k, m) in MECHANISMS.iter().enumerate() {
            let y = r.min.y + k as f32 * row;
            let rr = Rect::new(r.min.x, y, r.max.x, y + row - 2.0);
            let on = m.on(&self.toy);
            let over = pointer.is_some_and(|p| rr.contains(p, 0.0));
            if over {
                hud_rect(f, rr.min, rr.max, Color::hex(0x1E2638));
            }
            let s = (row * 0.5).min(11.0);
            let bx = Vec2::new(rr.min.x + 4.0, rr.min.y + 0.5 * (row - 2.0) - 0.5 * s);
            hud_outline(f, bx, bx + Vec2::splat(s), 1.2, m.color());
            if on {
                hud_rect_at(f, 2, bx + Vec2::splat(2.0), bx + Vec2::splat(s - 2.0), m.color());
            }
            let cy = rr.min.y + 0.5 * (row - 2.0);
            f.text_px(Vec2::new(bx.x + s + 8.0, cy), m.name(), small, if on { pal::TEXT } else { pal::DIM }, Align2::LEFT_CENTER);
            f.text_px(Vec2::new(rr.max.x - 4.0, cy), m.strength(&self.toy), small, if on { m.color() } else { pal::FAINT }, Align2::RIGHT_CENTER);
            if over && f.ptr.clicked {
                m.toggle(&mut self.toy);
                self.refresh();
            }
        }
    }

    /// The cumulative incidence of the comparison shown, both toy cohorts, with the
    /// paper's posted points where there are some.
    fn curves(&mut self, f: &mut Frame, lay: &Layout) {
        let r = lay.cum;
        if !r.shows() {
            return;
        }
        let small = lay.small;
        let check = self.check;
        let title = format!("Toy: cumulative incidence of {}, {}", check.counts(), check.name());
        let short = format!("Toy: {}, {}", check.counts(), check.name());
        f.text_px(
            Vec2::new(r.min.x - 32.0, r.min.y - 6.0),
            pick(f, &[&title, &short, "Toy: cumulative incidence"], r.w() + 32.0, small),
            small,
            pal::TEXT,
            Align2::LEFT_BOTTOM,
        );
        let top = self.shown.scales.get(0).max(0.5);
        let x = |t: f64| r.min.x + (t / YEARS as f64) as f32 * r.w();
        let y = |p: f64| r.max.y - (p / top) as f32 * r.h();
        let step = if top > 12.0 {
            5.0
        } else if top > 5.0 {
            2.0
        } else if top > 2.0 {
            1.0
        } else {
            0.5
        };
        let mut v = 0.0;
        while v <= top + 1e-9 {
            hud_grid(f, Vec2::new(r.min.x, y(v)), Vec2::new(r.max.x, y(v)), 1.0, pal::FAINT.a(if v == 0.0 { 0.9 } else { 0.3 }));
            f.text_px(Vec2::new(r.min.x - 5.0, y(v)), format!("{} %", num(v, if step < 1.0 { 1 } else { 0 })), small, pal::DIM, Align2::RIGHT_CENTER);
            v += step;
        }
        for t in 0..=YEARS {
            f.text_px(Vec2::new(x(t as f64), r.max.y + 3.0), t.to_string(), small, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(r.max.x, r.max.y + small + 5.0), "years after the first dose", small, pal::DIM, Align2::RIGHT_TOP);
        let horizon = check.design().horizon;
        let n = (horizon - COUNT_FROM + 1).min(CURVE);
        for (k, color) in [(0usize, EARLY), (1, LATE)] {
            let c = &self.shown.curves[k];
            let pts = (0..n).map(|i| Vec2::new(x((COUNT_FROM + i) as f64 * DT), y(100.0 * c.get(i))));
            hud_path(f, pts, 2.2, color);
        }
        if horizon < STEPS {
            let xe = x(horizon as f64 * DT);
            hud_dashed(f, Vec2::new(xe, r.min.y), Vec2::new(xe, r.max.y), 1.0, pal::DIM);
            f.text_px(Vec2::new(xe + 4.0, r.min.y + 2.0), "this check stops at 18 months", small, pal::DIM, Align2::LEFT_TOP);
        }
        // the ends named
        let end = |k: usize| self.shown.curves[k].get(n.saturating_sub(1)) * 100.0;
        let (ye, yl) = (y(end(0)), y(end(1)));
        let apart = if (ye - yl).abs() < small + 2.0 { 0.5 * (small + 2.0 - (ye - yl).abs()) } else { 0.0 };
        let (ye, yl) = if ye <= yl { (ye - apart, yl + apart) } else { (ye + apart, yl - apart) };
        let xe = x((horizon as f64 * DT).min(YEARS as f64)) + 6.0;
        if horizon >= STEPS {
            f.text_px(Vec2::new(xe, ye), "earlier", small, EARLY, Align2::LEFT_CENTER);
            f.text_px(Vec2::new(xe, yl), "later", small, LATE, Align2::LEFT_CENTER);
        }
        if let Some(m) = check.marks() {
            for (values, color) in [(m.earlier, EARLY), (m.later, LATE)] {
                for (t, v) in m.years.iter().zip(values) {
                    let p = Vec2::new(x(*t), y(*v));
                    // a published point: a yellow ring with the cohort's colour inside
                    hud_circle(f, p, 4.5, 1.8, PUBLISHED);
                    hud_circle(f, p, 1.6, 1.6, color);
                }
            }
            let say =
                pick(f, &["rings: the paper's posted points (yellow), for its two cohorts (grey, cyan)", "rings: the paper's points", ""], r.w() * 0.6, small);
            f.caption_px(Vec2::new(r.min.x + 6.0, r.min.y + 4.0), say, small, PUBLISHED, Align2::LEFT_TOP);
        }
    }

    /// The crude rate ratio in each year of follow-up: the toy's (pink bars), the
    /// vaccine's alone (green), and the paper's, calculated from its table (yellow).
    fn ratios(&mut self, f: &mut Frame, lay: &Layout) {
        let r = lay.ratios;
        if !r.shows() {
            return;
        }
        let small = lay.small;
        let check = self.check;
        let title = pick(
            f,
            &[
                "Toy: crude rate ratio in each year after the first dose, later over earlier",
                "Toy: rate ratio in each year, later ÷ earlier",
                "Toy: rate ratio by year",
            ],
            r.w() + 32.0,
            small,
        );
        if !lay.bare {
            f.text_px(Vec2::new(r.min.x - 32.0, r.min.y - small - 18.0), title, small, pal::TEXT, Align2::LEFT_BOTTOM);
        }
        let (lo, hi) = (self.shown.scales.get(1), self.shown.scales.get(2).max(self.shown.scales.get(1) + 0.1));
        let y = |v: f64| r.max.y - ((v - lo) / (hi - lo)) as f32 * r.h();
        let step = if hi - lo > 1.6 { 0.5 } else { 0.25 };
        let mut v = (lo / step).ceil() * step;
        while v <= hi + 1e-9 {
            let one = (v - 1.0).abs() < 1e-9;
            hud_grid(f, Vec2::new(r.min.x, y(v)), Vec2::new(r.max.x, y(v)), 1.0, pal::FAINT.a(if one { 0.95 } else { 0.3 }));
            f.text_px(Vec2::new(r.min.x - 5.0, y(v)), num(v, 2), small, if one { pal::TEXT } else { pal::DIM }, Align2::RIGHT_CENTER);
            v += step;
        }
        let slot = r.w() / YEARS as f32;
        let run = self.results.run(check);
        for k in 0..YEARS {
            let cx = r.min.x + (k as f32 + 0.5) * slot;
            f.text_px(Vec2::new(cx, r.max.y + 3.0), format!("year {}", k + 1), small, pal::DIM, Align2::CENTER_TOP);
            if run.by_year[k].is_none() {
                continue;
            }
            let (toy, alone) = (self.shown.years.get(k), self.shown.years.get(YEARS + k));
            let (a, b) = (y(1.0), y(toy));
            hud_fill(f, 1, Vec2::new(cx - 0.28 * slot, a.min(b)), Vec2::new(cx + 0.28 * slot, a.max(b).max(a.min(b) + 1.0)), NAIVE.a(0.8));
            hud_line(f, Vec2::new(cx - 0.42 * slot, y(alone)), Vec2::new(cx + 0.42 * slot, y(alone)), 2.4, TRUTH);
            // the toy's value at the end of its bar, and the paper's beside it
            let above = toy >= 1.0 - 1e-9;
            let at = Vec2::new(cx, if above { b - 2.0 } else { b + 2.0 });
            f.text_px(at, num(toy, 2), small, NAIVE, if above { Align2::CENTER_BOTTOM } else { Align2::CENTER_TOP });
            if check == Check::Main {
                let (p, px) = (y(CRUDE_RATIOS[k]), cx + 0.36 * slot);
                hud_shape(f, 2, &[Vec2::new(px - 5.0, p), Vec2::new(px, p - 5.0), Vec2::new(px + 5.0, p), Vec2::new(px, p + 5.0)], PUBLISHED);
            }
        }
        let key = if check == Check::Main {
            "pink: toy; green: the vaccine alone (drag the green dot); yellow: calculated from the paper's Supplementary Table 2"
        } else {
            "pink: toy; green: the vaccine alone"
        };
        let say = pick(
            f,
            &[key, "pink toy · green vaccine alone · yellow from the paper's table", "pink toy · green vaccine alone · yellow paper", "toy · alone · paper"],
            r.w() + 32.0,
            small,
        );
        f.text_px(Vec2::new(r.min.x - 32.0, r.min.y - if lay.bare { 4.0 } else { 12.0 }), say, small, pal::DIM, Align2::LEFT_BOTTOM);
        // the true effect, by hand: the green line of the first year
        if check.design().vaccine && check.design().outcome == Outcome::Dementia {
            let cx = r.min.x + 0.5 * slot;
            let at = Vec2::new(cx - 0.42 * slot, y(self.shown.years.get(YEARS)));
            let mut p = f.hud_at(2, at);
            if f.drag3_sized(1, &mut p, None, TRUTH, 12.0) {
                if let Some(s) = f.cam.project(p) {
                    let py = s.y / f.px;
                    let v = lo + ((r.max.y - py) / r.h().max(1.0)) as f64 * (hi - lo);
                    if v.is_finite() {
                        let theta = (v.max(THETA_LEAST).min(1.0) * 100.0).round() / 100.0;
                        if theta != self.toy.theta {
                            self.toy.theta = theta;
                            self.refresh();
                        }
                    }
                }
            }
        }
    }

    /// The ratio over the whole window: toy, vaccine alone, and the paper's.
    fn strip(&mut self, f: &mut Frame, lay: &Layout) {
        let r = lay.strip;
        if r.w() < 20.0 || r.h() < 8.0 {
            return;
        }
        let small = lay.small;
        let check = self.check;
        let measure = if check.design().hazard_ratio { "hazard ratio, 3 to 18 months" } else { "ratio of restricted mean time lost" };
        let title = format!("Toy: {measure}, {}", check.name());
        f.text_px(
            Vec2::new(r.min.x - 32.0, r.min.y - small - 6.0),
            pick(f, &[&title, &format!("Toy: {measure}")], r.w() + 32.0, small),
            small,
            pal::TEXT,
            Align2::LEFT_BOTTOM,
        );
        let (lo, hi) = (self.shown.scales.get(3), self.shown.scales.get(4).max(self.shown.scales.get(3) + 0.1));
        let x = |v: f64| r.min.x + ((v - lo) / (hi - lo)) as f32 * r.w();
        let cy = r.min.y + 0.5 * r.h();
        hud_line(f, Vec2::new(r.min.x, cy), Vec2::new(r.max.x, cy), 1.0, pal::FAINT);
        let mut v = (lo * 10.0).ceil() / 10.0;
        while v <= hi + 1e-9 {
            let one = (v - 1.0).abs() < 1e-9;
            hud_grid(
                f,
                Vec2::new(x(v), cy - if one { 7.0 } else { 3.0 }),
                Vec2::new(x(v), cy + if one { 7.0 } else { 3.0 }),
                1.0,
                if one { pal::TEXT } else { pal::FAINT },
            );
            f.text_px(Vec2::new(x(v), cy + 8.0), num(v, 1), small - 0.5, pal::DIM, Align2::CENTER_TOP);
            v += 0.1;
        }
        let (toy, alone) = (self.shown.ratio.get(0), self.shown.ratio.get(1));
        if let Some(p) = check.paper() {
            hud_line(f, Vec2::new(x(p.low), cy), Vec2::new(x(p.high), cy), 3.0, PUBLISHED.a(0.7));
            let d = 6.0;
            hud_shape(
                f,
                2,
                &[Vec2::new(x(p.value), cy - d), Vec2::new(x(p.value) + d, cy), Vec2::new(x(p.value), cy + d), Vec2::new(x(p.value) - d, cy)],
                PUBLISHED,
            );
        }
        hud_circle(f, Vec2::new(x(alone), cy), 7.0, 2.0, TRUTH);
        hud_circle(f, Vec2::new(x(toy), cy), 3.5, 3.5, NAIVE);
        // the numbers, in a row above
        let mut parts = vec![
            (format!("toy {}", num(self.results.run(check).ratio, 2)), NAIVE),
            (format!("vaccine alone {}", num(self.results.alone(check).ratio, 2)), TRUTH),
        ];
        match check.paper() {
            Some(p) => parts.push((format!("paper {}", p.text(2)), PUBLISHED)),
            None => parts.push(("paper: not reported for dementia".to_string(), PUBLISHED)),
        }
        let mut xx = r.max.x;
        for (t, c) in parts.iter().rev() {
            let wd = f.text_size(t, small).x;
            f.text_px(Vec2::new(xx, r.min.y - 6.0), t.as_str(), small, *c, Align2::RIGHT_BOTTOM);
            xx -= wd + 14.0;
        }
    }

    /// Every check at once: the toy's result, the vaccine's alone, and the paper's. A
    /// click on a row shows that check in the charts.
    fn table(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let r = lay.table;
        if !r.shows() {
            return;
        }
        let small = lay.small;
        let cols = if lay.wide { [0.0, 0.44, 0.58, 0.76] } else { [0.0, 0.45, 0.6, 0.76] };
        let col = |k: usize| r.min.x + cols[k] * r.w();
        let row = (r.h() / (CHECKS.len() + 1) as f32).clamp(11.0, 22.0);
        let head = r.min.y + 0.5 * row;
        f.text_px(
            Vec2::new(col(0) + 4.0, head),
            pick(f, &["Toy: each check (click to show it)", "Toy: checks"], col(1) - col(0) - 4.0, small),
            small,
            pal::TEXT,
            Align2::LEFT_CENTER,
        );
        f.text_px(Vec2::new(col(1), head), "toy", small, NAIVE, Align2::LEFT_CENTER);
        f.text_px(Vec2::new(col(2), head), pick(f, &["vaccine alone", "alone"], col(3) - col(2) - 4.0, small), small, TRUTH, Align2::LEFT_CENTER);
        f.text_px(Vec2::new(col(3), head), "paper", small, PUBLISHED, Align2::LEFT_CENTER);
        hud_line(f, Vec2::new(r.min.x, r.min.y + row), Vec2::new(r.max.x, r.min.y + row), 1.0, pal::FAINT);
        for (k, c) in CHECKS.iter().enumerate() {
            let y0 = r.min.y + (k + 1) as f32 * row;
            let rr = Rect::new(r.min.x, y0 + 1.0, r.max.x, y0 + row - 1.0);
            let over = pointer.is_some_and(|p| rr.contains(p, 0.0));
            if *c == self.check {
                hud_rect(f, rr.min, rr.max, Color::hex(0x24304A));
            } else if over {
                hud_rect(f, rr.min, rr.max, Color::hex(0x1A2133));
            }
            let cy = y0 + 0.5 * row;
            let hr = c.design().hazard_ratio;
            let name = if hr { format!("{} (hazard ratio)", c.name()) } else { c.name().to_string() };
            let name = pick(f, &[&name, c.name()], col(1) - col(0) - 8.0, small);
            f.text_px(Vec2::new(col(0) + 4.0, cy), name, small, if *c == self.check { pal::TEXT } else { pal::DIM }, Align2::LEFT_CENTER);
            f.text_px(Vec2::new(col(1), cy), num(self.results.run(*c).ratio, 2), small, NAIVE, Align2::LEFT_CENTER);
            f.text_px(Vec2::new(col(2), cy), num(self.results.alone(*c).ratio, 2), small, TRUTH, Align2::LEFT_CENTER);
            let paper = match c.paper() {
                Some(p) => p.text(2),
                None => "not reported".into(),
            };
            let paper = pick(f, &[&paper, &c.paper().map_or("—".to_string(), |p| num(p.value, 2))], r.max.x - col(3), small);
            f.text_px(Vec2::new(col(3), cy), paper, small, PUBLISHED, Align2::LEFT_CENTER);
            if over && f.ptr.clicked && self.check != *c {
                self.check = *c;
                self.moving = true;
            }
        }
    }
}

/// The restricted mean time lost of each cohort, in days, from the posted ratio and
/// difference (dossier §1.8: "about 58.0 days" and "48.3 days", calculated).
fn rmtl_days() -> (f64, f64) {
    let before = RMST_DIFFERENCE_DAYS.value / (1.0 - RMTL_RATIO_POSTED);
    (before, before - RMST_DIFFERENCE_DAYS.value)
}

/// A count in thousands, "101.8k" style, for the narrow rows of person-years.
fn short_count(v: f64) -> String {
    format!("{}k", num(v / 1000.0, 1))
}

impl Scene for Switch {
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
        Some("tabs over the picture: the comparison, what the paper found, the toy model    toy model: click a mechanism to switch it, a check to show it; drag the green dot to set the true effect")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A close look at one study: Taquet and others (Nature Medicine, 2024). In October 2017 the United States licensed and recommended a new shingles vaccine, the recombinant one, and use moved rapidly from the live vaccine to it. The study compares people first vaccinated (at 65 or over) in the three years after the switch with people first vaccinated in the three years before, matched on recorded characteristics, and finds fewer recorded dementia diagnoses in the later group over the next six years. Here: what that comparison assumes, what it found and checked, and, in a toy model with made-up numbers, which other mechanisms could or could not give the same pattern.");
        ui::note(ui, "This scene is about how to read such evidence. It is not a verdict on whether either vaccine prevents dementia, and it is not medical advice. Randomised trials of the recombinant vaccine with dementia as a primary outcome began in 2026 and have not reported.");

        ui::section(ui, "Things to try");
        for (label, look_for, setup) in TRY {
            if ui::experiment(ui, label, look_for) {
                self.apply(setup);
            }
        }

        ui::section(ui, "Show");
        let mut page = PAGES.iter().position(|p| *p == self.page).unwrap_or(0);
        if ui::choice(ui, &["the comparison", "what the paper found", "toy model"], &mut page) {
            self.page = PAGES[page.min(PAGES.len() - 1)];
            self.moving = true;
        }

        ui::section(ui, "The toy model's mechanisms");
        ui::note(ui, "Two made-up cohorts laid out on the paper's calendar (first doses spread evenly over each window, data ending in March 2024), analysed as the paper analyses, from 3 months to 6 years, with endlessly many people, so that nothing in it is chance. With nothing switched on, 6 % are diagnosed by six years, about as in the paper's earlier cohort; everything else is made up. Someone vaccinated at calendar time $v$ is diagnosed, $t$ years later (in year $k$ of follow-up), at the rate");
        ui::equation(ui, r"\lambda = b_k \, \theta \, (1 - u) \, p(v + t) \, e^{\delta (v + t - t_0)}");
        ui::note(
            ui,
            r"where $b_k$ is the baseline (up 12 % a year), $\theta$ the true effect and $u$ the healthier adopters (both in the later cohort only), $p$ is $1 - s$ during the toy pandemic and 1 otherwise, and $\delta$ is the drift, from $t_0$, November 2017. A share $q$ of those due leave the records first (the later cohort), and some have dementia already, unrecorded (more of the earlier vaccinees, whose records reach back less far), recorded at the rate 1 a year. Each mechanism below is a road in the graph; each strength at its left end is off.",
        );
        let mut changed = false;
        let mut v = self.toy.theta as f32;
        if ui::slider(ui, r"true effect: hazard ratio $\theta$", &mut v, THETA_LEAST as f32..=1.0) {
            self.toy.theta = v as f64;
            changed = true;
        }
        ui::note(
            ui,
            r"On the rate of dementia among those not yet diagnosed: $\theta = 1$ is no effect, $0.8$ is a fifth fewer. (The green dot on the picture sets it too.)",
        );
        let mut v = self.toy.delay as f32;
        if ui::slider(ui, "delay: share made up later", &mut v, 0.0..=1.0) {
            self.toy.delay = v as f64;
            changed = true;
        }
        ui::note(ui, "Of the diagnoses the true effect holds off in years 1 to 3, the share made in years 4 to 6 instead (at 1, every one: the curves meet at six years). With no true effect there is nothing to delay. A vaccine that delayed only the diagnosis, the disease unchanged, would put exactly the same records in these cohorts. The authors raise this reading themselves: the late hazard ratio above 1 “could imply that the vaccine delays, rather than prevents, dementia onset”, but “this was not robustly observed across analyses” (page 2780).");
        let mut v = self.toy.pandemic as f32;
        if ui::slider(ui, "pandemic: diagnoses not made", &mut v, 0.0..=PANDEMIC_MOST as f32) {
            self.toy.pandemic = v as f64;
            changed = true;
        }
        ui::note(ui, "From March 2020 to March 2022, in both cohorts alike, at whatever point of their follow-up that falls. The paper's check: the six-month windows followed for 18 months only, entirely before the pandemic, gave a hazard ratio of 0.74 (0.62–0.90). The later years of follow-up cannot be checked in this way: for the later cohort they all fall after March 2020.");
        let mut v = self.toy.drift as f32;
        if ui::slider(ui, "drift: change a year", &mut v, -DRIFT_MOST as f32..=DRIFT_MOST as f32) {
            self.toy.drift = v as f64;
            changed = true;
        }
        ui::note(ui, "Diagnosis or coding more (or less) ready by this rate each calendar year, in both cohorts alike. The direction over the decade is not known: the slider goes both ways. The paper's checks: the six-month windows gave 0.83 (0.76–0.92) and the negative control 0.97 (0.91–1.03); no placebo switch was reported for dementia.");
        let mut v = self.toy.adopters as f32;
        if ui::slider(ui, "healthier adopters: lower risk", &mut v, 0.0..=ADOPTERS_MOST as f32) {
            self.toy.adopters = v as f64;
            changed = true;
        }
        let mut general = self.toy.general as usize;
        if ui::choice(ui, &["in cognition only", "in general health"], &mut general) {
            self.toy.general = general == 1;
            changed = true;
        }
        ui::note(ui, "The later cohort's risk lower for reasons the records do not hold. In general health, its deaths, painful conditions and shingles are lower too; in cognition only, its dementia alone. In this toy the selection comes with the new vaccine, not with the period, so people given an unchanged vaccine after the switch (the placebo switch) do not have it. The paper matched on 60 recorded characteristics, prior influenza vaccination among them, and found mortality 0.98 (0.95–1.01) and the negative control 0.97 (0.91–1.03): evidence against large differences in general health, not against differences in cognition or health-care engagement alone.");
        let mut v = self.toy.leaving as f32;
        if ui::slider(ui, "leaving the records first", &mut v, 0.0..=LEAVING_MOST as f32) {
            self.toy.leaving = v as f64;
            changed = true;
        }
        ui::note(ui, "In the later cohort, this share of the people on their way to a diagnosis leave the records (care elsewhere, a move) before it is made. In this toy it comes with the period, so the placebo switch has it too. The paper aligned the cohorts' lengths of follow-up (0.83, 0.79–0.87); it reported no analysis of who left. In the first two years the later cohort lost 1.8 times as many people as the earlier (calculated from Supplementary Table 2).");
        let mut v = (self.toy.present * 100.0) as f32;
        if ui::slider(ui, "already there: % in Oct 2014", &mut v, 0.0..=(PRESENT_MOST * 100.0) as f32) {
            self.toy.present = v as f64 / 100.0;
            changed = true;
        }
        ui::note(ui, "Dementia present but not in the records at the first dose, because the records reach back less far, so not excluded, and then recorded during follow-up as if new. Its share falls steadily from this value among those vaccinated in October 2014 to none among those vaccinated in October 2020. The paper left out anyone with dementia (or another neurodegenerative disease) recorded before or up to a month after the first dose, and counted from 3 months; it reported no analysis by the length of the records before it, and notes that history before a record began “may well be incomplete”.");
        if ui.button("every mechanism off").clicked() {
            self.toy = Toy::OFF;
            changed = true;
        }
        ui::note(ui, SIGNATURES);
        if changed {
            self.refresh();
        }
        ui::section(ui, "Run a check on the toy cohorts");
        let names: Vec<&str> = CHECKS.iter().map(|c| c.name()).collect();
        let mut k = Results::index(self.check);
        if ui::choice(ui, &names, &mut k) {
            self.check = CHECKS[k.min(CHECKS.len() - 1)];
            self.moving = true;
        }
        ui::note(ui, CAUGHT);
        ui::note(ui, "The paper's checks, as the toy runs them: the cohorts vaccinated in the six months either side of the switch; those again, followed for 18 months only, all before the pandemic (a hazard ratio); outcomes the vaccine should not change (painful conditions needing care, unlinked to dementia; deaths); shingles, which the newer vaccine is known to prevent better (in the toy, by a hazard ratio of 0.6); another matching method (the toy has no recorded traits, so it changes nothing here). And one not reported for dementia: the placebo switch, the same comparison among people given, in the same two periods, a vaccine that did not change.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (EARLY, "grey: the earlier cohort, first dose October 2014 to September 2017 (98 % live vaccine)"),
                (LATE, "cyan: the later cohort, first dose November 2017 to October 2020 (95 % recombinant), and the vaccine in the graph"),
                (THIRD, "purple: the period's other roads to the recorded diagnosis, and the pandemic; dashed, the design needs them absent; solid, the toy has them open"),
                (EFFECT, "orange: dementia and its recorded diagnosis"),
                (TRUTH, "green: the toy's true effect, and what its comparison would give if nothing but the vaccine differed"),
                (NAIVE, "pink: what the toy's comparison gives"),
                (PUBLISHED, "yellow: what the paper reports (diamonds and rings), set beside the toy's numbers for reference"),
                (TOY_FRAME, "the dashed frame: the toy model; everything inside it is made up but the yellow marks, which are the paper's"),
            ],
        );
        ui::note(ui, "In the graph, a boxed variable is held fixed by the analysis: the recorded traits, matched on, and staying in the records, since only those still seen are counted. Holding a common effect opens a road, as in “three junctions”.");

        ui::section(ui, "Words");
        ui::term(ui, "Natural experiment:", "a comparison of groups formed by an event the people did not choose (here the date of a switch in vaccines), read as if the event had assigned them.");
        ui::term(ui, "Lexis diagram:", "calendar time across, time since an event up: each person climbs a diagonal, so a calendar event is a vertical line that each cohort meets at its own point of follow-up.");
        ui::term(
            ui,
            "Cumulative incidence:",
            "the share diagnosed by each time, here estimated as Kaplan–Meier does: people who leave the records count for as long as they are seen.",
        );
        ui::term(ui, "Restricted mean time lost:", "over a window (here 3 months to 6 years), the area between the curve of those still without a diagnosis and 100 %: the average time spent after a diagnosis within the window. The paper's measure is the later cohort's divided by the earlier's.");
        ui::term(ui, "Crude rate ratio:", "diagnoses per person-year in one cohort divided by the other's, in the same year of follow-up.");
        ui::term(ui, "Hazard ratio:", "the ratio of the rates at which people not yet diagnosed are diagnosed, at the same time after the first dose.");
        ui::term(ui, "Propensity-score matching:", "pairing people across the cohorts whose recorded characteristics make them equally likely to be in either, so that those characteristics balance. It cannot balance what is not recorded.");
        ui::term(ui, "Negative-control outcome:", "an outcome the vaccine should not change but that shares the suspected biases: a difference in it signals bias. The paper's: acutely painful conditions not linked with dementia, which almost always need care.");
        ui::term(
            ui,
            "Positive control:",
            "an outcome the vaccine is known to change (shingles): it checks that the design can see an effect, not that it is free of bias.",
        );
        ui::term(ui, "Placebo switch:", "the same before-and-after comparison among people given a vaccine that did not change: whatever came with the period, and not with the new vaccine, shows in it.");
        ui::term(
            ui,
            "Informative censoring:",
            "leaving the records for reasons tied to the outcome, such as early decline: those who stay are then not like those who left.",
        );
        ui::term(ui, "Toy model:", "a made-up model whose mechanisms are known, to see what each could produce; it cannot show what did.");

        ui::section(ui, "Numbers");
        let run = self.results.run(self.check);
        let alone = self.results.alone(self.check);
        let measure = if self.check.design().hazard_ratio { "the hazard ratio" } else { "the ratio of time lost" };
        ui::note(ui, &format!("The toy's {}, {measure}:", self.check.name()));
        ui::readout_colored(ui, "toy", num(run.ratio, 3), NAIVE);
        ui::readout_colored(ui, "the vaccine alone", num(alone.ratio, 3), TRUTH);
        let years: Vec<String> = run.by_year.iter().map(|y| y.map_or("—".to_string(), |y| num(y, 2))).collect();
        ui::readout(ui, "toy, ratio by year", years.join(" "));
        let six = |s: &Side| s.curve.last().copied().unwrap_or(0.0) * 100.0;
        ui::readout(ui, "toy, diagnosed by the end", format!("{} %, {} %", num(six(&run.sides[0]), 2), num(six(&run.sides[1]), 2)));
        ui::readout_colored(ui, "paper, main ratio", TABLE1[PRIMARY].1.text(2), PUBLISHED);

        ui::section(ui, "The paper's own limits, quoted");
        ui::note(ui, THEIR_CASE);
        egui::CollapsingHeader::new("their list").id_salt("switch limits").show(ui, |ui| {
            for q in LIMITS {
                ui::note(ui, q);
            }
        });
        ui::section(ui, "What would decide the open points");
        egui::CollapsingHeader::new("the questions").id_salt("switch questions").show(ui, |ui| {
            for q in QUESTIONS {
                ui::note(ui, q);
            }
            ui::note(ui, &format!("The same group ran a placebo switch (Tdap recipients) for cardiovascular outcomes in 2026: ratio of time lost {}. As far as the sources read on 25 November 2026 show, none has been published for dementia.", CARDIO_PLACEBO.text(2)));
        });

        ui::section(ui, "Sources");
        ui::note(ui, SOURCES);
        egui::CollapsingHeader::new("funding and interests").id_salt("switch funding").show(ui, |ui| {
            ui::note(ui, "Funding: “This research was supported by the NIHR (https://www.nihr.ac.uk/) Oxford Health Biomedical Research Centre (NIHR203316).” “The funder of the study had no role in study design, data collection, data analysis, data interpretation or writing of the paper.”");
            ui::note(ui, "Competing interests: “J.A.T. is a consultant for GSK and co-director of the Oxford-GSK Institute for Molecular and Computational Medicine. GSK had no involvement of any kind in this study and were not aware of it until after the paper was accepted. The other authors declare no competing interests.”");
        });
        ui::see_also(
            ui,
            &[
                ("shingles", "the other studies and their designs"),
                ("confound", "confounding: a common cause"),
                ("junctions", "why holding a common effect opens a road"),
                ("adjust", "what matching can and cannot close"),
                ("dosee", "why a randomised trial settles it"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        let pointer = f.ptr.pos.map(|p| p / f.px);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone:
        // nothing legible fits, and the picture is left as it is for when they close)
        let covered = view.x < 120.0 || view.y < 100.0;
        if !covered {
            // the picture is buttons and charts drawn on the screen: the pointer is never
            // for moving the view
            f.handles.capture();
            if pointer.is_some_and(|p| lay.tabs.contains(p, 2.0) || lay.switches.contains(p, 2.0) || lay.table.contains(p, 2.0)) {
                f.handles.claim_clicks();
            }
            self.tabs(f, &lay, pointer);
        }
        let was = (self.toy, self.check);
        if !covered {
            match self.page {
                Page::Design => self.draw_design(f, &lay),
                Page::Paper => self.draw_paper(f, &lay),
                Page::Toy => self.draw_toy(f, &lay, pointer),
            }
            self.sources(f, &lay);
        }
        if (self.toy, self.check) != was {
            self.moving = true;
        }
        self.advance(f.real_dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scene set up as experiment `k`, at rest.
    fn run(k: usize) -> Switch {
        let mut s = Switch::new();
        s.apply(TRY[k].2);
        s.settle();
        s
    }

    /// A ratio as the captions write it.
    fn two(v: f64) -> String {
        num(v, 2)
    }

    /// The toy with one mechanism switched on as a click on the picture switches it.
    fn clicked(m: Mechanism) -> Toy {
        let mut t = Toy::OFF;
        m.toggle(&mut t);
        t
    }

    /// Every switch off and no true effect: the two curves are one, and every ratio is 1
    /// (the positive control apart: there the newer vaccine has its known effect).
    #[test]
    fn with_nothing_on_the_cohorts_agree() {
        let r = Results::of(&Toy::OFF);
        for (check, run) in CHECKS.iter().zip(&r.runs) {
            if *check == Check::Shingles {
                assert!((run.ratio - r.alone(Check::Shingles).ratio).abs() < 1e-12 && run.ratio < 0.65);
                continue;
            }
            assert!((run.ratio - 1.0).abs() < 1e-9, "{check:?}: {}", run.ratio);
            for (a, b) in run.sides[0].curve.iter().zip(&run.sides[1].curve) {
                assert!((a - b).abs() < 1e-12, "{check:?}");
            }
            for (k, y) in run.by_year.iter().enumerate() {
                if let Some(y) = y {
                    assert!((y - 1.0).abs() < 1e-9, "{check:?} year {}: {y}", k + 1);
                }
            }
        }
        // the baseline gives its share by six years, near the paper's earlier cohort
        let main = r.run(Check::Main);
        let six = main.sides[0].curve.last().copied().unwrap_or(0.0);
        assert!((six - SIX_YEAR_SHARE).abs() < 1e-9 && (100.0 * six - DEMENTIA_MARKS.earlier[4]).abs() < 0.1);
        assert_eq!(main.sides[0].curve.len(), CURVE);
        // and the other outcomes theirs
        let at_six = |c: Check| r.run(c).sides[0].curve.last().copied().unwrap_or(0.0);
        assert!(
            (at_six(Check::Mortality) - DEATH_SHARE).abs() < 1e-9
                && (at_six(Check::Negative) - PAINFUL_SHARE).abs() < 1e-9
                && (at_six(Check::Shingles) - SHINGLES_SHARE).abs() < 1e-9
        );
    }

    /// A true hazard ratio θ and nothing else: the crude rate ratio is θ in every year,
    /// and the hazard ratio before the pandemic is θ.
    #[test]
    fn a_true_effect_alone_is_level() {
        for theta in [0.5, 0.7, 0.83, 1.0] {
            let toy = Toy { theta, ..Toy::OFF };
            let main = compare(&toy, Check::Main);
            for (k, y) in main.by_year.iter().enumerate() {
                let y = y.expect("every year has cases");
                assert!((y - theta).abs() < 1e-9, "θ = {theta}, year {}: {y}", k + 1);
            }
            let short = compare(&toy, Check::BeforePandemic);
            assert!((short.ratio - theta).abs() < 1e-6, "θ = {theta}: {}", short.ratio);
            // the placebo switch knows nothing of it
            assert!((compare(&toy, Check::Placebo).ratio - 1.0).abs() < 1e-9);
        }
    }

    /// A delay of every diagnosis held off: θ for three years, more after, and curves that
    /// meet at six years; half of them, half way; with no true effect, nothing at all.
    #[test]
    fn a_delay_makes_the_curves_meet() {
        let six = |c: &Comparison, k: usize| c.sides[k].curve.last().copied().unwrap_or(0.0);
        for theta in [0.5, 0.8, 0.95] {
            let all = compare(&Toy { theta, delay: 1.0, ..Toy::OFF }, Check::Main);
            assert!((six(&all, 0) - six(&all, 1)).abs() < 1e-9, "θ = {theta}");
            for (k, y) in all.by_year.iter().enumerate() {
                let y = y.expect("cases");
                assert!(if k < HELD_YEARS { (y - theta).abs() < 1e-9 } else { y > 1.0 }, "θ = {theta}, year {}: {y}", k + 1);
            }
            let half = compare(&Toy { theta, delay: 0.5, ..Toy::OFF }, Check::Main);
            let none = compare(&Toy { theta, ..Toy::OFF }, Check::Main);
            assert!(six(&none, 1) < six(&half, 1) && six(&half, 1) < six(&all, 1));
        }
        let inert = Results::of(&Toy { delay: 1.0, ..Toy::OFF });
        assert_eq!(inert.runs, Results::of(&Toy::OFF).runs);
        // a click on "a delay" brings a true effect with it, and says so when there is none
        assert_eq!(clicked(Mechanism::Delay), Toy { theta: 0.8, delay: 1.0, ..Toy::OFF });
        assert_eq!(Mechanism::Delay.strength(&Toy { delay: 1.0, ..Toy::OFF }), "nothing to delay");
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let says = |k: usize, part: &str| assert!(TRY[k].1.contains(part), "experiment {k} does not say {part:?}");
        // the opening picture: where the pandemic falls (checked against the dates in
        // `the_dossiers_arithmetic`)
        let s = run(0);
        assert_eq!((s.page, s.toy, s.check), (Page::Design, Toy::OFF, Check::Main));
        says(0, &format!("between {} and {} years", num(PANDEMIC_EARLIER.0, 1), num(PANDEMIC_EARLIER.1, 1)));
        says(0, &format!("first {} years", num(PANDEMIC_LATER.0, 1)));
        // a real effect: level at 0.80, ratio of time lost 0.80; the paper's crude ratios
        let s = run(1);
        let main = s.results.run(Check::Main);
        assert!(main.by_year.iter().all(|y| y.map(two).as_deref() == Some("0.80")));
        says(1, "true hazard ratio is 0.80");
        says(1, &format!("the ratio of time lost is {}", two(main.ratio)));
        says(1, &format!("{} in year 2, {} in year 6", two(CRUDE_RATIOS[1]), two(CRUDE_RATIOS[5])));
        // a pandemic and no effect: below 1 early, above late, as the paper's crude ratios
        let s = run(2);
        let (main, alone) = (s.results.run(Check::Main), s.results.alone(Check::Main));
        let y: Vec<f64> = main.by_year.iter().map(|y| y.unwrap_or(f64::NAN)).collect();
        assert!(y[..3].iter().all(|v| *v < 1.0) && y[4..].iter().all(|v| *v > 1.0), "{y:?}");
        assert!(CRUDE_RATIOS[..2].iter().all(|v| *v < 1.0) && CRUDE_RATIOS[4..].iter().all(|v| *v > 1.0));
        says(2, "40 % fewer diagnoses");
        assert_eq!(s.toy.pandemic, 0.4);
        says(2, &format!("({} in year 2)", two(y[1])));
        says(2, &format!("({} in year 5)", two(y[4])));
        says(2, &format!("the ratio of time lost is {}", two(main.ratio)));
        assert_eq!(two(alone.ratio), "1.00");
        let before = s.results.run(Check::BeforePandemic).ratio;
        says(2, &format!("reads {}, where the paper found a hazard ratio of {}", two(before), BEFORE_PANDEMIC_HR.text(2)));
        // ("alone cannot give both": before the pandemic the toy reads 1 whatever its strength)
        for k in 0..=8 {
            let t = Toy { pandemic: k as f64 * 0.1, ..Toy::OFF };
            assert!((compare(&t, Check::BeforePandemic).ratio - 1.0).abs() < 1e-9);
        }
        says(2, &format!("the six-month windows read {} (the paper's {})", two(s.results.run(Check::SixMonths).ratio), two(TABLE1[SIX_MONTHS].1.value)));
        // healthier adopters, in general health: level at 0.80; mortality and the negative
        // control move with them
        let s = run(3);
        assert!(s.toy.general && s.toy.adopters == 0.2 && s.check == Check::Mortality);
        assert!(s.results.run(Check::Main).by_year.iter().all(|y| y.map(two).as_deref() == Some("0.80")));
        let (mortality, negative) = (s.results.run(Check::Mortality).ratio, s.results.run(Check::Negative).ratio);
        says(3, &format!("mortality {} and the negative control {}", two(mortality), two(negative)));
        says(3, &format!("the paper's were {} and {}", two(TABLE1[MORTALITY].1.value), two(TABLE1[NEGATIVE].1.value)));
        // in cognition only: every check reads exactly what a true effect of 0.80 gives
        let cognition = Results::of(&Toy { adopters: 0.2, ..Toy::OFF });
        let real = Results::of(&Toy { theta: 0.8, ..Toy::OFF });
        for c in CHECKS {
            assert!((cognition.run(c).ratio - real.run(c).ratio).abs() < 1e-9, "{c:?}");
        }
        says(3, "true hazard ratio of 0.80");
        // a placebo switch: the main comparison, and the placebo's, which is the pandemic's alone
        let s = run(4);
        assert!(!s.toy.general && s.toy.adopters == 0.15 && s.toy.pandemic == 0.4 && s.check == Check::Placebo);
        let (main, placebo) = (s.results.run(Check::Main).ratio, s.results.run(Check::Placebo).ratio);
        says(4, &format!("main comparison gives {}", two(main)));
        says(4, &format!("gives {}: the pandemic's part", two(placebo)));
        assert!((placebo - compare(&Toy { pandemic: 0.4, ..Toy::OFF }, Check::Placebo).ratio).abs() < 1e-12);
        assert!((placebo - compare(&Toy { pandemic: 0.4, ..Toy::OFF }, Check::Main).ratio).abs() < 1e-12);
        says(4, "15 %");
        // every experiment opens on the comparison its caption speaks of
        for (k, (_, _, setup)) in TRY.iter().enumerate() {
            let s = run(k);
            assert_eq!((s.page, s.check), (setup.page, setup.check), "experiment {k}");
        }
    }

    /// The panel's note on what each mechanism does alone (`SIGNATURES`), at the strength
    /// a click on the picture gives it.
    #[test]
    fn each_mechanism_has_its_signature() {
        let years = |t: &Toy| -> Vec<f64> { compare(t, Check::Main).by_year.iter().map(|y| y.expect("cases")).collect() };
        for m in [Mechanism::Effect, Mechanism::Adopters, Mechanism::Leaving] {
            let y = years(&clicked(m));
            assert!(y.iter().all(|v| two(*v) == "0.80"), "{m:?}: {y:?}");
        }
        assert!(SIGNATURES.contains("the same ratio, 0.80, in every year"));
        let drift = years(&clicked(Mechanism::Drift));
        let (lo, hi) = drift.iter().fold((f64::MAX, f64::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
        assert!(SIGNATURES.contains(&format!("({} to {})", two(lo), two(hi))) && clicked(Mechanism::Drift).drift == -0.05, "{drift:?}");
        let pandemic = years(&clicked(Mechanism::Pandemic));
        assert!(pandemic[..3].iter().all(|v| *v < 1.0) && pandemic[4..].iter().all(|v| *v > 1.0), "{pandemic:?}");
        let present = years(&clicked(Mechanism::Present));
        assert!(present[0] < 1.0 && present.windows(2).all(|w| w[1] >= w[0] - 1e-12) && two(present[5]) == "1.00", "{present:?}");
        let delay = years(&clicked(Mechanism::Delay));
        assert!(delay[..3].iter().all(|v| two(*v) == "0.80") && delay[3..].iter().all(|v| *v > 1.0), "{delay:?}");
        // only the pandemic and the delay put the late years above 1
        for m in MECHANISMS {
            let late = years(&clicked(m))[4..].iter().any(|v| *v > 1.0 + 1e-9);
            assert_eq!(late, matches!(m, Mechanism::Pandemic | Mechanism::Delay), "{m:?}");
        }
        // the strengths the note gives
        for (m, says) in [
            (Mechanism::Effect, "(0.80)"),
            (Mechanism::Adopters, "(20 %)"),
            (Mechanism::Leaving, "(20 %)"),
            (Mechanism::Pandemic, "(40 %)"),
            (Mechanism::Present, "(1 %)"),
        ] {
            assert!(SIGNATURES.contains(says), "{m:?}");
        }
        assert_eq!(
            (
                clicked(Mechanism::Adopters).adopters,
                clicked(Mechanism::Leaving).leaving,
                clicked(Mechanism::Pandemic).pandemic,
                clicked(Mechanism::Present).present
            ),
            (0.2, 0.2, 0.4, 0.01)
        );
    }

    /// The panel's note on which checks see which mechanism (`CAUGHT`).
    #[test]
    fn which_checks_see_what() {
        let bias = |r: &Results, c: Check| r.run(c).ratio - r.alone(c).ratio;
        for m in [Mechanism::Pandemic, Mechanism::Drift, Mechanism::Present] {
            let r = Results::of(&clicked(m));
            let main = bias(&r, Check::Main);
            for c in [Check::SixMonths, Check::BeforePandemic] {
                assert!(bias(&r, c).abs() < 0.5 * main.abs(), "{m:?}, {c:?}: {} against {main}", bias(&r, c));
            }
        }
        for t in [clicked(Mechanism::Leaving), clicked(Mechanism::Adopters), Toy { adopters: 0.2, general: true, ..Toy::OFF }] {
            let r = Results::of(&t);
            let main = bias(&r, Check::Main);
            for c in [Check::SixMonths, Check::BeforePandemic] {
                assert!((bias(&r, c) - main).abs() < 0.01, "{t:?}, {c:?}");
            }
        }
        // mortality and the negative control move only with adopters healthier in general
        let mut toys: Vec<Toy> = MECHANISMS.iter().map(|m| clicked(*m)).collect();
        toys.push(Toy { adopters: 0.2, general: true, ..Toy::OFF });
        for t in &toys {
            let r = Results::of(t);
            for c in [Check::Mortality, Check::Negative] {
                let moved = (r.run(c).ratio - 1.0).abs() > 0.05;
                assert_eq!(moved, t.general, "{t:?}, {c:?}: {}", r.run(c).ratio);
                assert!(moved || (r.run(c).ratio - 1.0).abs() < 1e-9);
            }
            // the placebo switch: the whole of what comes with the period, none of what comes with the vaccine
            let period = t.pandemic > 0.0 || t.drift != 0.0 || t.leaving > 0.0 || t.present > 0.0;
            let placebo = r.run(Check::Placebo).ratio;
            if period {
                assert!((placebo - r.run(Check::Main).ratio).abs() < 1e-12 && (placebo - 1.0).abs() > 0.05, "{t:?}");
            } else {
                assert!((placebo - 1.0).abs() < 1e-9, "{t:?}");
            }
        }
        assert!(CAUGHT.contains("(pandemic, drift, leaving, dementia already there)") && CAUGHT.contains("(a true effect, a delay, healthier adopters)"));
    }

    /// Working out again only what a change touches gives what working it all out gives.
    #[test]
    fn an_update_is_a_fresh_computation() {
        let toys = [
            Toy::OFF,
            Toy { pandemic: 0.3, ..Toy::OFF },
            Toy { pandemic: 0.3, theta: 0.7, ..Toy::OFF },
            Toy { pandemic: 0.3, theta: 0.7, adopters: 0.2, ..Toy::OFF },
            Toy { pandemic: 0.3, theta: 0.7, adopters: 0.2, general: true, ..Toy::OFF },
            Toy { drift: 0.05, leaving: 0.1, present: 0.02, delay: 0.5, theta: 0.6, ..Toy::OFF },
            Toy::OFF,
        ];
        let mut r = Results::of(&toys[0]);
        for t in toys {
            r.update(&t);
            assert_eq!(r, Results::of(&t), "{t:?}");
        }
    }

    /// The numbers the picture writes that the dossier calculated from the paper's
    /// tables, worked out again from those tables; and the published ones as given.
    #[test]
    fn the_dossiers_arithmetic() {
        // the crude rate ratios, from Supplementary Table 2
        for (k, [e, l]) in SUPPL_TABLE_2.iter().enumerate() {
            assert_eq!(two((l.1 / l.0) / (e.1 / e.0)), two(CRUDE_RATIOS[k]), "year {}", k + 1);
        }
        let total = |side: usize| SUPPL_TABLE_2.iter().fold((0.0, 0.0), |(c, p), row| (c + row[side].1, p + row[side].0));
        assert_eq!((total(0), total(1)), ((5_397.0, 540_093.0), (3_157.0, 403_266.0)));
        // 1.8 times as many lost in the first two years
        let lost = |side: usize| SUPPL_TABLE_2[0][side].2 + SUPPL_TABLE_2[1][side].2;
        assert_eq!((lost(0), lost(1)), (6_376.0, 11_419.0));
        assert_eq!(num(lost(1) / lost(0), 1), "1.8");
        assert!(QUESTIONS[3].contains("1.8 times as many"));
        // the time lost, in days: about 58.0 and 48.3; 9.75 days of about 2,100, 0.5 %
        let (before, after) = rmtl_days();
        assert_eq!((num(before, 1), num(after, 1)), ("58.0".to_string(), "48.3".to_string()));
        assert!((after / before - RMTL_RATIO_POSTED).abs() < 1e-12 && two(RMTL_RATIO_POSTED) == two(TABLE1[PRIMARY].1.value));
        let window = WINDOW_DAYS.1 - WINDOW_DAYS.0;
        assert_eq!((window, num(100.0 * RMST_DIFFERENCE_DAYS.value / window, 1)), (2101.0, "0.5".to_string()));
        // the paper's 164 days is 9.75 days over the earlier cohort's 5.94 %
        assert_eq!((RMST_DIFFERENCE_DAYS.value / (DEMENTIA_MARKS.earlier[4] / 100.0)).round(), DAYS_AMONG_AFFECTED.value);
        // where the pandemic falls in each cohort's follow-up, from the dates
        let near = |a: f64, b: f64| (a - b).abs() < 0.1;
        assert!(near(PANDEMIC - EARLIER.to(), PANDEMIC_EARLIER.0) && near(PANDEMIC - EARLIER.from, PANDEMIC_EARLIER.1));
        assert!(near(PANDEMIC - LATER.from, PANDEMIC_LATER.0) && near(PANDEMIC - LATER.to(), PANDEMIC_LATER.1));
        // matching kept 99.9 % and 26.9 % of the pools
        assert_eq!(
            (num(100.0 * MATCHED as f64 / POOL_EARLIER as f64, 1), num(100.0 * MATCHED as f64 / POOL_LATER as f64, 1)),
            ("99.9".to_string(), "26.9".to_string())
        );
        // the windows: three years each side, October 2017 left out; six months each side
        assert_eq!((EARLIER.to(), LATER.from, EARLIER.to() + 1.0 / 12.0), (month(2017, 10), month(2017, 11), LATER.from));
        assert_eq!((EARLIER_SIX.to(), LATER_SIX.to()), (month(2017, 10), month(2018, 5)));
        // the published figures as the picture writes them
        assert_eq!(TABLE1[PRIMARY].1.text(2), "0.83 (0.79–0.87)");
        assert_eq!(BEFORE_PANDEMIC_HR.text(2), "0.74 (0.62–0.90)");
        assert_eq!(TABLE1.iter().map(|r| r.1.text(2)).collect::<Vec<_>>().join(" "), "0.83 (0.79–0.87) 0.78 (0.73–0.83) 0.87 (0.81–0.94) 0.83 (0.76–0.92) 0.83 (0.79–0.87) 0.82 (0.79–0.86) 0.79 (0.74–0.83) 0.84 (0.80–0.88) 0.82 (0.77–0.87) 0.93 (0.91–0.96) 0.98 (0.95–1.01) 0.97 (0.91–1.03) 0.65 (0.61–0.69)");
        assert_eq!(CARDIO_PLACEBO.text(2), "1.01 (0.99–1.03)");
        assert_eq!((MEDIAN_LATER.text(2), MEDIAN_EARLIER.value), ("4.15 (3.16–4.99)".to_string(), 6.0));
        assert_eq!(MORE_TIME_WOMEN_MEN, (22.0, 13.0, 0.017));
        assert_eq!((count(MATCHED), COVARIATES, ORGANISATIONS, LIVE_BEFORE, RECOMBINANT_AFTER), ("103\u{2009}837".to_string(), 60, 62, 98.0, 95.0));
    }

    /// Strengths at their ends, alone and together, and settings that are not numbers:
    /// every number stays finite and every ratio positive.
    #[test]
    fn degenerate_settings_stay_finite() {
        let ends = [
            Toy::OFF,
            Toy { theta: THETA_LEAST, delay: 1.0, ..Toy::OFF },
            Toy { pandemic: PANDEMIC_MOST, ..Toy::OFF },
            Toy { drift: DRIFT_MOST, ..Toy::OFF },
            Toy { drift: -DRIFT_MOST, ..Toy::OFF },
            Toy { adopters: ADOPTERS_MOST, general: true, ..Toy::OFF },
            Toy { leaving: LEAVING_MOST, ..Toy::OFF },
            Toy { present: PRESENT_MOST, ..Toy::OFF },
            Toy {
                theta: THETA_LEAST,
                delay: 1.0,
                pandemic: PANDEMIC_MOST,
                drift: DRIFT_MOST,
                adopters: ADOPTERS_MOST,
                general: true,
                leaving: LEAVING_MOST,
                present: PRESENT_MOST,
            },
            Toy { theta: f64::NAN, delay: f64::INFINITY, pandemic: -3.0, drift: 99.0, adopters: 2.0, general: false, leaving: f64::NAN, present: -1.0 },
        ];
        for t in ends {
            let r = Results::of(&t);
            for (c, run) in CHECKS.iter().zip(r.runs.iter().chain(&r.alone)) {
                assert!(run.ratio.is_finite() && run.ratio > 0.0, "{t:?} {c:?}: {}", run.ratio);
                assert!(run.by_year.iter().flatten().all(|v| v.is_finite() && *v > 0.0), "{t:?} {c:?}");
                for s in &run.sides {
                    assert!(s.curve.iter().all(|v| v.is_finite() && (0.0..1.0).contains(v)) && s.rmtl.is_finite(), "{t:?} {c:?}");
                }
            }
            let mut s = Switch::new();
            s.toy = t;
            s.refresh();
            for c in CHECKS {
                s.check = c;
                let tg = targets(&s.results, c);
                assert!(tg.curves.iter().flatten().chain(&tg.years).chain(&tg.ratio).chain(&tg.scales).all(|v| v.is_finite()), "{t:?} {c:?}");
                assert!(tg.scales[1] < tg.scales[2] && tg.scales[3] < tg.scales[4] && tg.scales[0] > 0.0);
            }
            assert!(s.toy.tidy() == s.toy && s.toy.theta >= THETA_LEAST);
        }
        // nothing to divide
        assert_eq!((quotient(1.0, 0.0), quotient(f64::NAN, 1.0)), (1.0, 1.0));
        assert_eq!(hazard_ratio(&Side::default(), &Side::default()), 1.0);
        assert!(nice_top(f64::NAN).is_finite() && nice_top(-1.0) > 0.0);
    }

    /// After any change the picture comes to rest, by the clock, at 60 frames a second and
    /// at 144, exactly where it was going.
    #[test]
    fn the_picture_comes_to_rest() {
        for hz in [60.0f32, 144.0] {
            let mut s = Switch::new();
            for k in [1usize, 2, 4, 3, 0, 2] {
                s.apply(TRY[k].2);
                let mut frames = 0;
                while s.moving {
                    s.advance(1.0 / hz);
                    frames += 1;
                    assert!((frames as f32) / hz < 2.0, "experiment {k} still moving at {hz} Hz");
                }
                let t = targets(&s.results, s.check);
                assert!(s.shown.curves[1].shown == t.curves[1] && s.shown.years.shown == t.years && s.shown.ratio.shown == t.ratio);
            }
            // a check chosen glides there too
            for c in CHECKS {
                s.check = c;
                s.moving = true;
                let mut frames = 0;
                while s.moving {
                    s.advance(1.0 / hz);
                    frames += 1;
                    assert!((frames as f32) / hz < 2.0);
                }
            }
        }
    }

    /// The parts of each page keep to the canvas, below the application's buttons, and
    /// clear of one another, whatever its shape.
    #[test]
    fn the_layout_fits_every_canvas() {
        for (x, y) in [
            (1078.0f32, 1000.0f32),
            (1600.0, 1000.0),
            (1100.0, 700.0),
            (578.0, 700.0),
            (390.0, 800.0),
            (390.0, 754.0),
            (360.0, 640.0),
            (600.0, 900.0),
            (844.0, 390.0),
            (640.0, 480.0),
            (2400.0, 1300.0),
            (760.0, 560.0),
        ] {
            let inset = if x < 900.0 { 46.0 } else { 0.0 };
            let lay = Layout::of(Vec2::new(x, y), inset);
            let all = [
                ("tabs", lay.tabs),
                ("lexis", lay.lexis),
                ("graph", lay.graph),
                ("legend", lay.legend),
                ("words", lay.words),
                ("survival", lay.survival),
                ("sums", lay.sums),
                ("forest", lay.forest),
                ("years", lay.years),
                ("frame", lay.frame),
                ("mini", lay.mini),
                ("toy graph", lay.toy_graph),
                ("toy legend", lay.toy_legend),
                ("mechanisms", lay.mechs),
                ("switches", lay.switches),
                ("curves", lay.cum),
                ("ratios", lay.ratios),
                ("strip", lay.strip),
                ("table", lay.table),
            ];
            for (name, r) in all {
                if r == Rect::NONE {
                    continue;
                }
                assert!(r.min.is_finite() && r.max.is_finite() && r.min.x >= 0.0 && r.min.y >= inset && r.max.x <= x && r.max.y <= y, "{x}x{y}: {name} {r:?}");
            }
            for page in [
                vec![lay.lexis, lay.graph, lay.legend, lay.words],
                vec![lay.survival, lay.forest, lay.years],
                vec![lay.mini, lay.toy_graph, lay.toy_legend, lay.switches, lay.cum, lay.ratios, lay.strip, lay.table],
            ] {
                let solid: Vec<Rect> = page.into_iter().filter(|r| *r != Rect::NONE).collect();
                for (i, a) in solid.iter().enumerate() {
                    for b in &solid[i + 1..] {
                        assert!(a.max.x <= b.min.x || b.max.x <= a.min.x || a.max.y <= b.min.y || b.max.y <= a.min.y, "{x}x{y}: {a:?} and {b:?} overlap");
                    }
                }
            }
            if !lay.bare {
                // the main picture of each page has room to be read
                let short = y < 700.0;
                for (name, r, least) in [("lexis", lay.lexis, 120.0), ("forest", lay.forest, 120.0), ("ratios", lay.ratios, 50.0), ("curves", lay.cum, 60.0)] {
                    if short && name == "curves" && r == Rect::NONE {
                        continue;
                    }
                    assert!(r.w() >= 200.0 && r.h() >= least, "{x}x{y}: {name} {r:?}");
                }
            }
            for k in 0..PAGES.len() {
                assert!(lay.tab(k).max.x <= x - 4.0, "{x}x{y}: tab {k}");
            }
        }
        // a window too small for more has each page's main picture alone
        for (x, y) in [(240.0f32, 180.0f32), (300.0, 260.0), (1.0, 1.0)] {
            let lay = Layout::of(Vec2::new(x, y), 46.0);
            assert!(lay.bare && lay.graph == Rect::NONE && lay.table == Rect::NONE);
            for r in [lay.tabs, lay.lexis, lay.forest, lay.ratios] {
                assert!(r.min.is_finite() && r.max.is_finite() && r.w() >= 0.0 && r.h() >= 0.0);
            }
        }
    }

    /// Every text the scene shows, the parts of the panel that open only on a click
    /// included, parses as markup, with no sign of mathematics left outside dollars.
    #[test]
    fn every_text_is_set_right() {
        let s = Switch::new();
        let mut texts: Vec<String> =
            vec![SOURCES.into(), SOURCES_SHORT.into(), TIME_COURSE.into(), THEIR_CASE.into(), SIGNATURES.into(), CAUGHT.into(), Switch::legend()];
        texts.extend(LIMITS.iter().chain(&QUESTIONS).map(|t| t.to_string()));
        texts.extend(TRY.iter().flat_map(|(a, b, _)| [a.to_string(), b.to_string()]));
        texts.extend(MECHANISMS.iter().flat_map(|m| [m.name().to_string(), m.short().to_string()]));
        texts.extend(CHECKS.iter().map(|c| c.name().to_string()));
        texts.extend(TABLE1.iter().map(|r| r.0.to_string()));
        texts.extend(s.design_words(true).into_iter().chain(s.design_words(false)));
        for t in &texts {
            let pieces = crate::typeset::pieces(t);
            assert!(pieces.is_ok(), "does not parse: {t}");
            // (the panel's folded parts are not seen by `eidos mathlint`: the signs of
            // mathematics it looks for, outside dollars, looked for here)
            for piece in pieces.unwrap_or_default() {
                if let crate::typeset::Piece::Text(words) = piece {
                    assert!(!words.chars().any(|c| "=^_<>≤≥≈×θλσαβμ".contains(c)), "mathematics outside dollars in: {t}");
                    assert!(!words.contains(" p. ") && !words.contains("(p. "), "a lone p reads as a variable: {t}");
                }
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        let show = |name: &str, toy: Toy| {
            let r = Results::of(&toy);
            let main = r.run(Check::Main);
            let years: Vec<String> = main.by_year.iter().map(|y| y.map_or("-".into(), |y| format!("{y:.3}"))).collect();
            let checks: Vec<String> = CHECKS.iter().map(|c| format!("{:.3}/{:.3}", r.run(*c).ratio, r.alone(*c).ratio)).collect();
            let six = |s: &Side| s.curve.last().copied().unwrap_or(0.0) * 100.0;
            println!("{name:<28} years [{}]  6y {:.2}% {:.2}%  checks {}", years.join(" "), six(&main.sides[0]), six(&main.sides[1]), checks.join(" "));
        };
        show("off", Toy::OFF);
        for m in MECHANISMS {
            show(&format!("{m:?}"), clicked(m));
        }
        show("adopters 0.2 general", Toy { adopters: 0.2, general: true, ..Toy::OFF });
        for s in [0.2, 0.4, 0.6, 0.8] {
            show(&format!("pandemic {s}"), Toy { pandemic: s, ..Toy::OFF });
        }
        show("pandemic .4 + adopters .15", Toy { pandemic: 0.4, adopters: 0.15, ..Toy::OFF });
        show("theta .85 + pandemic .3", Toy { theta: 0.85, pandemic: 0.3, ..Toy::OFF });
        let t = std::time::Instant::now();
        for _ in 0..20 {
            let _ = Results::of(&Toy { pandemic: 0.4, present: 0.01, leaving: 0.1, ..Toy::OFF });
        }
        println!("Results::of: {:.2} ms", t.elapsed().as_secs_f64() * 1000.0 / 20.0);
        let mut r = Results::of(&Toy::OFF);
        let t = std::time::Instant::now();
        for k in 0..20 {
            r.update(&Toy { pandemic: 0.3 + 0.01 * k as f64, ..Toy::OFF });
        }
        println!("Results::update (pandemic): {:.2} ms", t.elapsed().as_secs_f64() * 1000.0 / 20.0);
    }
}

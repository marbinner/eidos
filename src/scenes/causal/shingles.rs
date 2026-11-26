//! Shingles vaccine and dementia: a map of the published studies, the causal graph
//! each kind of study assumes, and what each design can and cannot rule out.
//!
//! This is a case study about real papers by named researchers. Everything factual
//! in it comes from the dossier compiled for this section on 25 November 2026 from
//! the papers and the trial registries; `FACTS` holds every number shown, each with
//! where the dossier gives it, and the tests hold the texts to it. Nothing is
//! simulated. What the scene adds is the drawing: one causal graph for the whole
//! question, coloured by what each design closes and leaves open and by the arrows
//! each explanation would need, a table of designs against explanations, and the
//! studies grouped by design, each estimate in its own unit.

use super::*;
use std::collections::HashMap;

pub const DEF: SceneDef = SceneDef {
    id: "shingles",
    title: "Shingles vaccine and dementia",
    group: crate::scenes::CASE_STUDIES,
    blurb: "A real question with no finished trial: the studies, the causal graph each of them assumes, and what each design can and cannot rule out.",
    make: |_| Box::new(Shingles::new()),
};

// ----- the graph ------------------------------------------------------------------------

/// The variables, by their place in `VARS`.
mod v {
    pub const E: usize = 0;
    pub const P: usize = 1;
    pub const V: usize = 2;
    pub const T: usize = 3;
    pub const R: usize = 4;
    pub const I: usize = 5;
    pub const D: usize = 6;
    pub const DX: usize = 7;
    pub const H: usize = 8;
    pub const K: usize = 9;
    pub const F: usize = 10;
    pub const A: usize = 11;
    pub const S: usize = 12;
    pub const B: usize = 13;
    pub const O: usize = 14;
    pub const Z: usize = 15;
    pub const L: usize = 16;
    pub const M: usize = 17;
    pub const C: usize = 18;
    pub const N: usize = 19;
}

/// Which variables are drawn: the core graph always, and two layers the visitor can
/// turn on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layer {
    Core,
    /// More causes of who is vaccinated.
    Who,
    /// How a diagnosis comes to be recorded, and who stays in the data.
    Record,
}

/// What a variable is in the question: the exposure, the outcome, or something else.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Part {
    Exposure,
    Outcome,
    Other,
}

/// Whether the variable is in the data the studies use.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Seen {
    Yes,
    Partly,
    No,
}

/// Where a variable's name is written beside its disc.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
    Above,
    Below,
    Left,
    Right,
}

struct Var {
    sym: &'static str,
    name: &'static str,
    /// The name on a narrow canvas.
    short: &'static str,
    /// Its place in the unit square the graph is drawn in (x across, y down).
    at: (f32, f32),
    side: Side,
    layer: Layer,
    part: Part,
    seen: Seen,
    /// What it is (the line under the graph, when the pointer is on it).
    note: &'static str,
}

#[allow(clippy::too_many_arguments)]
const fn var(
    sym: &'static str,
    name: &'static str,
    short: &'static str,
    at: (f32, f32),
    side: Side,
    layer: Layer,
    part: Part,
    seen: Seen,
    note: &'static str,
) -> Var {
    Var { sym, name, short, at, side, layer, part, seen, note }
}

const VARS: [Var; 20] = [
    var("$E$", "eligibility", "eligible", (0.02, 0.36), Side::Below, Layer::Core, Part::Other, Seen::Yes, "Eligibility for a free vaccine, by date of birth, age band or the start of a programme. Nothing a person does can change which side of a birth-date cut-off they are on."),
    var("$P$", "period", "period", (0.02, 0.64), Side::Below, Layer::Core, Part::Other, Seen::Yes, "The period in which someone was vaccinated: before or after the US switch of October 2017."),
    var("$V$", "vaccinated", "vaccinated", (0.2, 0.34), Side::Above, Layer::Core, Part::Exposure, Seen::Yes, "Vaccination against shingles, with its date."),
    var("$T$", "which vaccine", "vaccine", (0.22, 0.66), Side::Below, Layer::Core, Part::Exposure, Seen::Yes, "Which vaccine: the recombinant one (a protein of the virus with the AS01B adjuvant, two doses) or the live one (one dose)."),
    var("$R$", "reactivation", "reactivation", (0.42, 0.38), Side::Above, Layer::Core, Part::Other, Seen::No, "Reactivation of the varicella-zoster virus, with symptoms or without: mostly unobserved."),
    var("$I$", "immune stimulation", "immune", (0.4, 0.64), Side::Below, Layer::Core, Part::Other, Seen::No, "Effects on the immune system that do not go through the virus (the adjuvant, trained immunity, effects on other infections): a hypothesis."),
    var("$D$", "dementia", "dementia", (0.66, 0.48), Side::Left, Layer::Core, Part::Outcome, Seen::No, "The dementia process itself, its pathology and its symptoms: never observed directly."),
    var("$D^*$", "recorded diagnosis", "diagnosis", (0.94, 0.52), Side::Above, Layer::Core, Part::Outcome, Seen::Yes, "A recorded diagnosis of dementia and its date: what every study actually analyses."),
    var("$H$", "health, engagement, frailty", "health", (0.6, 0.04), Side::Above, Layer::Core, Part::Other, Seen::Partly, "Health-seeking behaviour, frailty and function (the healthy vaccinee): only partly in the records."),
    var("$K$", "health-care contact", "contact", (0.84, 0.84), Side::Below, Layer::Core, Part::Other, Seen::Yes, "Contact with the health service, and how intensely dementia is looked for and coded."),
    var("$F$", "calendar years followed", "calendar", (0.4, 0.98), Side::Left, Layer::Core, Part::Other, Seen::Yes, "The calendar years someone is followed through: the COVID-19 pandemic, changes in coding."),
    var("$A$", "age, sex", "age, sex", (0.32, 0.02), Side::Left, Layer::Who, Part::Other, Seen::Yes, "Age, which drives uptake, dementia and death; and sex (several studies report larger associations in women)."),
    var("$S$", "income, education", "income", (0.07, 0.07), Side::Above, Layer::Who, Part::Other, Seen::No, "Socio-economic position and education: largely unmeasured in health records."),
    var("$B$", "early decline", "decline", (0.8, 0.28), Side::Above, Layer::Who, Part::Other, Seen::No, "Cognitive decline before any diagnosis (prodromal): unmeasured."),
    var("$O$", "other vaccines", "others", (0.86, 0.1), Side::Above, Layer::Who, Part::Other, Seen::Yes, "Other vaccinations: influenza, Tdap, pneumococcal, RSV."),
    var("$Z$", "recorded shingles", "shingles", (0.54, 0.74), Side::Below, Layer::Record, Part::Other, Seen::Yes, "A recorded episode of shingles: the visible part of reactivation."),
    var("$L$", "site", "site", (0.34, 0.84), Side::Above, Layer::Record, Part::Other, Seen::Partly, "The clinic, hospital or region: hidden from researchers in TriNetX."),
    var("$\\dagger$", "death", "death", (0.96, 0.2), Side::Right, Layer::Record, Part::Other, Seen::Yes, "Death, a competing event: no diagnosis can follow it."),
    var("$C$", "leaves the record", "leaves", (0.98, 0.78), Side::Below, Layer::Record, Part::Other, Seen::Yes, "Leaving the record (lost to follow-up): moving, changing provider, ending contact."),
    var("$N$", "second dose", "dose 2", (0.0, 0.24), Side::Above, Layer::Record, Part::Other, Seen::Yes, "A second dose of the recombinant vaccine: an event after the start of follow-up."),
];

struct Arrow {
    from: usize,
    to: usize,
    /// How far the arrow bows sideways, as a share of its length (positive: to the
    /// right of the way it points, as seen on the screen).
    bend: f32,
    /// Drawn only in a design that must assume it away (dashed, with a "?").
    assumed: bool,
    /// Why it is there, and what supports it.
    note: &'static str,
}

const fn arrow(from: usize, to: usize, bend: f32, note: &'static str) -> Arrow {
    Arrow { from, to, bend, assumed: false, note }
}

const fn assumed(from: usize, to: usize, bend: f32, note: &'static str) -> Arrow {
    Arrow { from, to, bend, assumed: true, note }
}

const ARROWS: [Arrow; 40] = [
    arrow(v::E, v::V, 0.0, "Programmes offer a free vaccine to eligible birth cohorts: uptake jumped by 47.2 percentage points in Wales, 16.4 in Australia and 27.4 in Ontario (a survey)."),
    arrow(v::P, v::T, 0.0, "The US switch of October 2017: 98% of first doses were of the live vaccine before it, 95% of the recombinant vaccine after."),
    arrow(v::P, v::V, 0.0, "Recommendations, shortages and novelty change who is vaccinated in each period: recorded first doses a year roughly quadrupled after the switch (calculated for this map from Taquet 2024, Supplementary Table 3)."),
    arrow(v::P, v::F, 0.0, "People vaccinated in different periods are followed through different calendar years: the pandemic falls early in the later group's follow-up and late in the earlier group's."),
    arrow(v::H, v::V, -0.2, "Healthier, more engaged people get vaccinated (shown for influenza vaccine by Jackson and others, 2006; recombinant-vaccine completion is higher with better health and more visits, Ackerson and others, 2021)."),
    arrow(v::H, v::D, -0.2, "Frailty and health behaviour may relate to the risk of dementia: plausible, not cited for dementia specifically."),
    arrow(v::H, v::K, -0.12, "Engaged people see clinicians more, so problems are found earlier: plausible."),
    arrow(v::V, v::R, 0.12, "Vaccines reduce reactivation of the virus."),
    arrow(v::T, v::R, 0.0, "The recombinant vaccine reduces reactivation more than the live one: in separate trials, efficacy against shingles of 97.4% at 60 to 69 and 91.3% at 70 or older, against 64% and 38% for the live vaccine."),
    arrow(v::V, v::I, 0.0, "Immune effects that do not go through the virus, such as trained immunity: a hypothesis (the Welsh authors write of a VZV-independent immunomodulatory effect of the live vaccine)."),
    arrow(v::T, v::I, -0.2, "The recombinant vaccine's AS01 adjuvant stimulates the immune system: a hypothesis for an effect on dementia (Taquet 2024; Taquet 2025)."),
    arrow(v::R, v::D, 0.0, "Hypothesised: the virus damages the brain or its vessels. Shingles is associated with dementia (pooled risk ratio 1.14, 1.04–1.25, Marra 2025); debated: a Danish cohort found no association."),
    arrow(v::I, v::D, 0.0, "Hypothesised: immune stimulation protects against dementia."),
    arrow(v::D, v::DX, 0.0, "Dementia is diagnosed and recorded, late and incompletely: what every study analyses is $D^*$, not $D$."),
    arrow(v::K, v::DX, 0.2, "More contact with the health service, more chances for dementia to be diagnosed and coded."),
    arrow(v::V, v::K, 0.0, "The vaccination visit itself, and a second visit for the recombinant vaccine, is contact with the health service: plausible; it would push towards more diagnoses among the vaccinated."),
    arrow(v::F, v::K, 0.0, "Calendar time changes diagnosis: new dementia registrations fell in the pandemic in Sweden, England's estimated dementia diagnosis rate fell by 5.4%, and the US coding system changed on 1 October 2015."),
    arrow(v::A, v::V, 0.0, "Age drives uptake."),
    arrow(v::A, v::D, -0.2, "Age drives the risk of dementia."),
    arrow(v::S, v::V, 0.12, "Resources and education shape access to vaccines (completion of the recombinant series is higher with higher income and education, Ackerson 2021)."),
    arrow(v::S, v::D, -0.15, "Education and resources may shape the risk of dementia: widely hypothesised, not cited here."),
    arrow(v::B, v::V, 0.14, "People in early, undiagnosed decline may be less likely to seek or complete vaccination (shown for diagnosed dementia, Ackerson 2021)."),
    arrow(v::B, v::DX, 0.2, "Early decline comes before the diagnosis."),
    arrow(v::H, v::O, 0.12, "Vaccinated people get other vaccines too: associations with dementia of similar size for Tdap, pneumococcal and influenza vaccines (Harris 2023; Wiemken 2022; Lophatananon 2023)."),
    arrow(v::R, v::Z, 0.12, "Clinical shingles is the visible part of reactivation."),
    arrow(v::Z, v::K, 0.0, "A shingles episode is a clinic visit that can reveal dementia (the Welsh authors tested this route and argue it is too small to explain their effect)."),
    arrow(v::L, v::T, -0.2, "Sites differ in how they record vaccines: plausible."),
    arrow(v::L, v::K, 0.0, "Sites differ in how they code dementia: plausible; TriNetX hides which site."),
    arrow(v::D, v::M, 0.0, "Dementia shortens life."),
    arrow(v::H, v::M, 0.12, "Frailty shortens life."),
    arrow(v::M, v::DX, 0.2, "Death is a competing event: someone who has died cannot be diagnosed later."),
    arrow(v::D, v::C, 0.0, "People with emerging dementia may leave the record (care homes, other providers): plausible."),
    arrow(v::K, v::C, -0.2, "Whether people stay in contact decides whether they stay in the record."),
    arrow(v::C, v::DX, 0.0, "Leaving the record hides any later diagnosis."),
    arrow(v::V, v::N, 0.0, "A second dose of the recombinant vaccine follows a first."),
    arrow(v::H, v::N, 0.0, "Completion of the recombinant series is higher with better health and more visits (Ackerson 2021)."),
    assumed(v::E, v::K, -0.2, "Must be absent: eligibility must bring no contact with the health service of its own (the exclusion restriction). A visit prompted by eligibility would push towards more diagnoses."),
    assumed(v::E, v::DX, -0.5, "Must be absent: nothing else that affects recorded dementia may change at the cut-off date (continuity)."),
    assumed(v::P, v::C, -0.3, "Must be absent: leaving the record must not differ by period. In years 1 and 2 the later group lost about 1.8 times as many people (calculated for this map from Taquet 2024, Supplementary Table 2)."),
    assumed(v::O, v::D, 0.4, "Must be absent, or known: the comparator vaccine's own effect on dementia. Several comparators are themselves associated with lower dementia (Harris 2023; Lophatananon 2023)."),
];

/// The arrow from `from` to `to`, if the graph has one.
#[cfg(test)]
fn arrow_at(from: usize, to: usize) -> Option<usize> {
    ARROWS.iter().position(|a| a.from == from && a.to == to)
}

// ----- the designs ----------------------------------------------------------------------

/// What a picture beside the text shows: how a design works, or one study's numbers.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
enum Pic {
    None,
    Cohort,
    Comparator,
    Switch,
    Birth,
    Clone,
    Trial {
        blind: bool,
    },
    /// Rayens 2026: one study, one unit, two comparison groups.
    Rayens,
    /// Hamilton 2026: the positive control and the outcome, both in percentage points.
    Hamilton,
    /// Kivimäki 2026: the implied effects per vaccinee, one paper, one unit.
    Kivimaki,
}

type Pairs = &'static [(usize, usize)];

struct Design {
    /// The row of the table.
    name: &'static str,
    /// The row on a narrow canvas.
    short: &'static str,
    title: &'static str,
    gist: &'static str,
    who: &'static str,
    assumption: &'static str,
    graph: &'static str,
    checks: &'static str,
    cannot: &'static str,
    /// What the studies of this design show, where the dossier draws a pattern.
    seen: &'static str,
    pic: Pic,
    /// What is compared, what is held (matched, adjusted, or the same for everyone
    /// compared) and held only as far as the records show it.
    compared: &'static [usize],
    held: &'static [usize],
    partly: &'static [usize],
    /// Arrows the design removes; arrows on the path it estimates; on paths it leaves
    /// open; on paths it closes only in part; and arrows it must assume absent.
    cut: Pairs,
    effect: Pairs,
    open: Pairs,
    weak: Pairs,
    assumes: Pairs,
    /// A layer whose variables the design cannot be drawn without.
    needs: Option<Layer>,
}

/// Every arrow into the vaccination, which randomisation removes.
const INTO_V: Pairs = &[(v::E, v::V), (v::P, v::V), (v::H, v::V), (v::A, v::V), (v::S, v::V), (v::B, v::V), (v::P, v::T), (v::L, v::T)];
const INTO_V_AND_VISIT: Pairs = &[(v::E, v::V), (v::P, v::V), (v::H, v::V), (v::A, v::V), (v::S, v::V), (v::B, v::V), (v::P, v::T), (v::L, v::T), (v::V, v::K)];
const VACCINE_EFFECT: Pairs = &[(v::V, v::R), (v::V, v::I), (v::R, v::D), (v::I, v::D), (v::D, v::DX)];
const VACCINE_TYPE_EFFECT: Pairs = &[(v::V, v::R), (v::V, v::I), (v::T, v::R), (v::T, v::I), (v::R, v::D), (v::I, v::D), (v::D, v::DX)];

const COHORT: usize = 0;
const COMPARATOR: usize = 1;
const SWITCH: usize = 2;
const BIRTH: usize = 3;
const EMULATION: usize = 4;
const OPEN_TRIAL: usize = 5;
const BLIND_TRIAL: usize = 6;

const DESIGNS: [Design; 7] = [
    Design {
        name: "cohort: vaccinated or not",
        short: "cohort",
        title: "Cohort: vaccinated against unvaccinated",
        gist: r"People who chose to be vaccinated against people who did not, adjusted for what the records show. Health and engagement that the records miss stay a back door: $V \gets H \to D$.",
        who: "People who were vaccinated against people who were not, matched, weighted or adjusted on recorded traits (age, sex, illnesses, use of care).",
        assumption: "Within levels of the measured covariates, vaccinated and unvaccinated people would have had the same risk of dementia without vaccination (conditional exchangeability); also positivity, consistency, and outcomes recorded as well in both groups.",
        graph: r"Adjustment closes the back doors through what is recorded: age and sex ($A$, among the further causes), and health only as far as the records show it ($H$, faint pink). What the records miss stays open, and with more causes shown so do income and education ($S$) and early decline ($B$). The vaccination visit is itself contact with the health service ($V \to K$).",
        checks: "Covariate balance; negative-control outcomes (Rayens: 0.94); negative-control exposures; active comparators; E-values (Rayens: 3.50); analyses with the start of follow-up shifted earlier (Polisky).",
        cannot: "No check can prove that no unmeasured common cause, such as health engagement or cognition, remains. A pitfall: immortal time, when the vaccinated must survive to a dose and the unvaccinated start at another event (Dos Reis: a preventive visit).",
        seen: "In the same data, Rayens and others report a hazard ratio of 0.49 against unvaccinated people and 0.73 against people given the Tdap vaccine: the association moves towards 1 when the comparison group also chose a vaccine. Harris and others found near-identical associations for the zoster, Tdap and pneumococcal vaccines.",
        pic: Pic::Cohort,
        compared: &[v::V],
        held: &[v::A],
        partly: &[v::H],
        cut: &[],
        effect: VACCINE_EFFECT,
        open: &[(v::S, v::V), (v::S, v::D), (v::B, v::V), (v::B, v::DX), (v::V, v::K), (v::K, v::DX), (v::R, v::Z), (v::Z, v::K)],
        weak: &[(v::H, v::V), (v::H, v::D), (v::H, v::K), (v::H, v::M), (v::M, v::DX)],
        assumes: &[],
        needs: None,
    },
    Design {
        name: "active comparator",
        short: "comparator",
        title: "Active comparator: one vaccine against another",
        gist: r"People given a zoster vaccine against people given another vaccine. Both chose a vaccine, so the healthy-vaccinee back door narrows; the comparator must itself leave dementia alone ($O \to D$).",
        who: "People given a zoster vaccine against people given another vaccine (influenza, Tdap, pneumococcal), matched on recorded traits; or one zoster vaccine against the other.",
        assumption: "Given the measured covariates, people who chose one vaccine are exchangeable with those who chose the other; and the comparator vaccine has no effect on dementia, or a known one.",
        graph: r"Health and engagement decide strongly whether people are vaccinated at all, and more weakly which vaccine they choose: what remains of $V \gets H \to D$ is that weaker part (faint pink). The comparator vaccine's own effect, $O \to D$, must be absent.",
        checks: "Several different comparators; negative-control outcomes; analyses with the start of follow-up shifted earlier.",
        cannot: "Cannot prove that the choice between vaccines is unconfounded. Several comparator vaccines are themselves associated with lower dementia (Harris 2023; Lophatananon 2023), so a comparator's own effect is uncertain in both directions. Cannot isolate the adjuvant, because the antigens and the people differ too.",
        seen: "Two direct comparisons of the recombinant with the live vaccine: Taquet 2024 (a ratio of restricted mean time lost, 0.83) and Polisky 2025 (risk 18% lower at 5 years). They point the same way, and both rest on US health records and on who chose a new vaccine.",
        pic: Pic::Comparator,
        compared: &[v::V],
        held: &[v::A],
        partly: &[v::H],
        cut: &[],
        effect: VACCINE_EFFECT,
        open: &[(v::R, v::Z), (v::Z, v::K), (v::K, v::DX)],
        weak: &[(v::H, v::V), (v::H, v::D), (v::H, v::K), (v::S, v::V), (v::S, v::D), (v::B, v::V), (v::B, v::DX), (v::H, v::M), (v::M, v::DX)],
        assumes: &[(v::O, v::D)],
        needs: Some(Layer::Who),
    },
    Design {
        name: "calendar switch, 2017",
        short: "switch",
        title: "Calendar switch: before and after October 2017",
        gist: "Vaccinated people before the US switch (98% live vaccine) against vaccinated people after it (95% recombinant). It assumes the period changes recorded dementia only through which vaccine was given.",
        who: "People first vaccinated at 65 or older between November 2017 and October 2020 (95% recombinant) against those first vaccinated between October 2014 and September 2017 (98% live), matched one to one on 60 recorded characteristics.",
        assumption: "Once measured covariates are matched, being vaccinated just after rather than just before October 2017 is unrelated to a dementia diagnosis except through the type of vaccine.",
        graph: r"Everyone compared was vaccinated, so $V$ is held, and who came forward in each period ($P \to V \gets H$) can differ; matching closes that path only for what the records show. The two groups are followed through different calendar years ($P \to F \to K \to D^*$, the pandemic among them), and leaving the record must not differ by period ($P \to C$).",
        checks: "Done: six-month windows either side of the switch, 0.83 (0.76–0.92); follow-up ended before the pandemic, hazard ratio 0.74 (0.62–0.90); shingles itself as a positive control, 0.65 (0.61–0.69); mortality, 0.98 (0.95–1.01); a negative-control outcome, 0.97 (0.91–1.03); another matching method, 0.82 (0.77–0.87). Not done for dementia: the same comparison of periods among people given a vaccine that did not change (the authors ran it for heart outcomes in 2026: 1.01, 0.99–1.03).",
        cannot: "Selection by period on unmeasured traits, and differences in measurement between the periods, cannot be tested in these data. It compares the recombinant vaccine with the live one, not with no vaccine. The next scene, Two vaccines, one switch, takes this study apart.",
        seen: "Corsi-Zuelli and others (2026), from the same group, used the switch for heart and vessel outcomes, not dementia, and added checks: among Tdap recipients, a vaccine that did not change, the two periods gave 1.01 (0.99–1.03).",
        pic: Pic::Switch,
        compared: &[v::P],
        held: &[v::V, v::A],
        partly: &[v::H],
        cut: &[],
        effect: &[(v::P, v::T), (v::T, v::R), (v::T, v::I), (v::R, v::D), (v::I, v::D), (v::D, v::DX)],
        open: &[(v::P, v::F), (v::F, v::K), (v::K, v::DX), (v::S, v::V), (v::S, v::D), (v::B, v::V), (v::B, v::DX), (v::R, v::Z), (v::Z, v::K), (v::C, v::DX)],
        weak: &[(v::P, v::V), (v::H, v::V), (v::H, v::D), (v::H, v::K), (v::H, v::M), (v::M, v::DX)],
        assumes: &[(v::P, v::C)],
        needs: None,
    },
    Design {
        name: "birth-date cut-off",
        short: "birth date",
        title: "Birth-date cut-off (regression discontinuity)",
        gist: "People born just after a cut-off date (eligible) against just before (not eligible). Choice and health do not decide which side of a birthday someone falls, so the healthy-vaccinee paths stay closed.",
        who: "People born just after a date that made them eligible for a free vaccine against people born just before, who were not. Close to the cut-off the two sides differ in nothing but eligibility.",
        assumption: "Nothing else that affects recorded dementia changes abruptly at the cut-off date (continuity). The jump in diagnoses there is then the effect of eligibility. Dividing it by the jump in uptake gives the effect among those vaccinated because eligible, if eligibility acts only through vaccination (exclusion), nobody is vaccinated because ineligible (monotonicity), and uptake is measured accurately.",
        graph: r"No arrow leads into $E$: birth dates cannot be chosen. Health still decides who of the eligible take the vaccine ($H \to V$), but the comparison is by birth date, not by vaccination, so that path is closed at $V$, where two arrows meet. Both sides live through the same calendar time. Must be absent: other changes at the same date ($E \to D^*$) and visits brought by eligibility itself ($E \to K$).",
        checks: "Balance of prior diagnoses and covariates either side; placebo cut-offs at other dates; other diseases as placebo outcomes; the years before the programme; different bandwidths and grace periods; replication in other data.",
        cannot: "Cannot prove that nothing unobserved changes at the cut-off. The division by uptake is open to mis-measured uptake (Australia, Canada) and to eligibility acting in other ways; Kivimäki and others test one consequence, that implied effects per vaccinee be at most 100%. Estimates hold near the cut-off: ages about 79 to 80 in Wales and Australia, about 70 in Ontario. Two studies add difference-in-differences, which assumes parallel trends instead.",
        seen: "Wales, Australia and Ontario report fewer diagnoses; the largest study, of English hospital records, finds no effect on hospital-coded dementia. Dividing by uptake implies 18% in Wales but more than 100% in Australia and in one Canadian analysis (Kivimäki and others, 2026).",
        pic: Pic::Birth,
        compared: &[v::E],
        held: &[],
        partly: &[],
        cut: &[],
        effect: &[(v::E, v::V), (v::V, v::R), (v::V, v::I), (v::R, v::D), (v::I, v::D), (v::D, v::DX)],
        open: &[(v::V, v::K), (v::K, v::DX), (v::R, v::Z), (v::Z, v::K)],
        weak: &[],
        assumes: &[(v::E, v::K), (v::E, v::DX)],
        needs: None,
    },
    Design {
        name: "target-trial emulation",
        short: "emulation",
        title: "Target-trial emulation",
        gist: "Observational data analysed as a trial would be: one start of follow-up for everyone, each person copied into both strategies and censored when their own course leaves one. It fixes timing, not unmeasured confounding.",
        who: "Patients newly admitted to skilled nursing given the recombinant vaccine within 12 months against those not given it, each cloned into both strategies, censored on deviating and weighted (Hayes 2026).",
        assumption: "At each time, vaccination is as good as random given the measured history (sequential exchangeability), with positivity and correctly specified censoring weights.",
        graph: "Drawn here it looks like the cohort: what it fixes is timing (immortal time, and a start of follow-up that differs between groups), which a graph of this kind does not show. The back doors through unmeasured traits stay open.",
        checks: "Negative-control analyses, which the authors write 'suggest some residual confounding'.",
        cannot: "Unmeasured confounding. In a frail population of whom few were vaccinated (1.73%), who is vaccinated may be strongly selected.",
        seen: "",
        pic: Pic::Clone,
        compared: &[v::V],
        held: &[v::A],
        partly: &[v::H],
        cut: &[],
        effect: VACCINE_EFFECT,
        open: &[(v::S, v::V), (v::S, v::D), (v::B, v::V), (v::B, v::DX), (v::V, v::K), (v::K, v::DX), (v::R, v::Z), (v::Z, v::K)],
        weak: &[(v::H, v::V), (v::H, v::D), (v::H, v::K), (v::H, v::M), (v::M, v::DX)],
        assumes: &[],
        needs: None,
    },
    Design {
        name: "trial, open label",
        short: "open trial",
        title: "Randomised trial, open label (DAN-ZOSTER)",
        gist: r"Chance decides who is vaccinated, which cuts every arrow into $V$. Participants know their arm, so contact with the health service after vaccination can still differ ($V \to K$): a problem of measurement, not of confounding.",
        who: "168,460 people aged 65 or older in Denmark, randomised to two doses of the recombinant vaccine or to no vaccination, with no placebo; outcomes from national registries.",
        assumption: "Randomisation holds; follow-up is complete; outcomes are recorded alike in both arms.",
        graph: r"Randomisation cuts every arrow into $V$ and $T$: health, eligibility and period no longer decide who is vaccinated. $V \to K \to D^*$ stays open, because people know whether they were vaccinated.",
        checks: "Started 28 April 2026; active, not recruiting; primary completion estimated June 2029. New dementia at about 3 years is one of two primary outcomes.",
        cannot: "It compares the recombinant vaccine with none, not with the live vaccine, and cannot separate the antigen from the adjuvant. Not yet reported.",
        seen: "",
        pic: Pic::Trial { blind: false },
        compared: &[v::V, v::T],
        held: &[],
        partly: &[],
        cut: INTO_V,
        effect: VACCINE_TYPE_EFFECT,
        open: &[(v::V, v::K), (v::K, v::DX), (v::R, v::Z), (v::Z, v::K)],
        weak: &[],
        assumes: &[],
        needs: None,
    },
    Design {
        name: "trial, blinded placebo",
        short: "blind trial",
        title: "Randomised trial, blinded, with placebo (Finland)",
        gist: r"Chance decides who is vaccinated, and the placebo arm has the same injections and visits without knowing it, so the vaccination visit no longer differs between arms.",
        who: "An estimated 33,609 people aged 76 or older in Finland, randomised 3 to 1 to the recombinant vaccine or placebo, observer-blind.",
        assumption: "Randomisation holds; follow-up is complete; outcomes are recorded alike in both arms, which blinding helps to ensure.",
        graph: r"Randomisation cuts every arrow into $V$ and $T$, and placebo with blinding cuts $V \to K$. Fewer shingles episodes still mean fewer visits ($R \to Z \to K$), a route the Welsh authors tested and argue is too small to explain their effect.",
        checks: "Started 31 March 2026; recruiting; primary completion estimated 1 April 2030. Primary outcome: the hazard ratio of a dementia diagnosis, up to 10 years.",
        cannot: "Its comparator is placebo, not the live vaccine; it cannot separate the antigen from the adjuvant. It excludes nursing-home residents and people under 76. Not yet reported.",
        seen: "",
        pic: Pic::Trial { blind: true },
        compared: &[v::V, v::T],
        held: &[],
        partly: &[],
        cut: INTO_V_AND_VISIT,
        effect: VACCINE_TYPE_EFFECT,
        open: &[(v::R, v::Z), (v::Z, v::K), (v::K, v::DX)],
        weak: &[],
        assumes: &[],
        needs: None,
    },
];

// ----- the explanations -----------------------------------------------------------------

struct Story {
    /// Its number in the dossier, and its name in two lines for the table's header.
    label: &'static str,
    head: [&'static str; 2],
    name: &'static str,
    title: &'static str,
    /// A real effect of the vaccine on dementia (green), or a difference without one (pink).
    real: bool,
    arrows: Pairs,
    /// The arrows by which it enters: with a design chosen, all its arrows are drawn in
    /// the state the design leaves the most open of these (a healthy-vaccinee story is
    /// closed where health no longer decides who is compared, whatever else is open).
    gates: Pairs,
    gist: &'static str,
    predicts: &'static str,
    evidence: &'static str,
    /// A layer whose variables the explanation cannot be drawn without.
    needs: Option<Layer>,
}

const STORIES: [Story; 8] = [
    Story {
        label: "1",
        head: ["fewer", "reactivations"],
        name: "fewer reactivations",
        title: "1. A true effect through fewer reactivations of the virus",
        real: true,
        arrows: &[(v::V, v::R), (v::T, v::R), (v::R, v::D), (v::D, v::DX)],
        gates: &[(v::V, v::R), (v::T, v::R)],
        gist: r"Vaccination keeps the virus from waking up ($V, T \to R$), and reactivation harms the brain ($R \to D$).",
        predicts: "A larger effect for the recombinant vaccine than for the live one, roughly in proportion to their protection against shingles; an effect that wanes as that protection wanes; larger in people at higher risk of shingles; a lag between vaccination and fewer diagnoses; no effect from vaccines that do not act on the virus.",
        evidence: "For it: gradients with protection against shingles (Polisky; Taquet), and the time lag and sex pattern in Wales. Clinical shingles alone looks too rare: in Taquet's data recorded shingles fell by 0.43 percentage points by 2 years and recorded dementia by 0.45 (calculated for this map from the authors' posted data), so nearly every averted episode would have to avert a diagnosis; reactivation without symptoms is not ruled out.",
        needs: None,
    },
    Story {
        label: "2",
        head: ["immune", "stimulation"],
        name: "immune stimulation",
        title: "2. A true effect through general immune stimulation",
        real: true,
        arrows: &[(v::T, v::I), (v::V, v::I), (v::I, v::D), (v::D, v::DX)],
        gates: &[(v::V, v::I), (v::T, v::I)],
        gist: r"Something in the vaccine other than its action on the virus (for the recombinant vaccine, the AS01 adjuvant) acts on the immune system, and through it on dementia ($T \to I \to D$).",
        predicts: "Effects from other adjuvanted vaccines too (an AS01 vaccine against RSV), and not from matched vaccines without the adjuvant; an effect not proportional to protection against shingles; possibly larger in women.",
        evidence: "For it: recipients of AS01 vaccines against RSV had a lower 18-month dementia risk than recipients of influenza vaccine (Taquet 2025). Contested by Williams and others (2025): the RSV brand was recorded for only 44%, the RSV vaccine has half the adjuvant dose yet a similar association, and many vaccines without an adjuvant show similar associations.",
        needs: None,
    },
    Story {
        label: "3",
        head: ["healthy", "vaccinee"],
        name: "healthy vaccinee",
        title: "3. Healthy-vaccinee confounding",
        real: false,
        arrows: &[(v::H, v::V), (v::H, v::D), (v::H, v::K), (v::K, v::DX), (v::H, v::O), (v::H, v::M)],
        gates: &[(v::H, v::V)],
        gist: r"Healthier, more engaged people are the ones who get vaccinated ($H \to V$), and the same traits lower their risk of dementia or change how it is diagnosed ($H \to D$, $H \to K$).",
        predicts: "Similar 'protection' from every vaccine; associations with lower mortality and with unrelated outcomes; associations that shrink with active comparators; no effect in birth-date studies or randomised trials.",
        evidence: "Rayens and others: 0.49 against unvaccinated people, 0.73 against Tdap recipients, in the same data. Harris and others: near-identical associations for zoster, Tdap and pneumococcal vaccines. The birth-date studies are designed to exclude it: Wales, Australia and Ontario are positive, and England's hospital-data study is null.",
        needs: None,
    },
    Story {
        label: "4",
        head: ["delayed", "diagnosis"],
        name: "delayed diagnosis",
        title: "4. Delayed diagnosis rather than delayed disease",
        real: false,
        arrows: &[(v::V, v::K), (v::K, v::DX), (v::R, v::Z), (v::Z, v::K)],
        gates: &[(v::V, v::K), (v::Z, v::K)],
        gist: r"Vaccination changes when dementia is diagnosed, not when it develops: through contact with the health service ($V \to K \to D^*$), or through shingles episodes averted and the visits that would have come with them ($R \to Z \to K$).",
        predicts: "Effects on the diagnosis of other conditions that depend on being looked for; no effect on hard end points that do not depend on a diagnosis; cumulative incidence curves that converge over long follow-up.",
        evidence: "The Welsh study checked other diagnoses and found no effect except on rheumatological ones. Effects on dementia deaths (Michalik; Xie) point against it, though deaths too depend partly on a recorded diagnosis; converging curves in Taquet 2024 point the other way.",
        needs: None,
    },
    Story {
        label: "4b",
        head: ["delayed", "onset"],
        name: "delayed onset",
        title: "4b. Delayed onset rather than prevention",
        real: true,
        arrows: &[(v::V, v::R), (v::T, v::R), (v::V, v::I), (v::T, v::I), (v::R, v::D), (v::I, v::D), (v::D, v::DX)],
        gates: &[(v::V, v::R), (v::T, v::R), (v::V, v::I), (v::T, v::I)],
        gist: "A real effect on the disease that only postpones it: fewer cases early, as many in the end. The arrows are those of a true effect; what differs is timing.",
        predicts: "A hazard ratio below 1 early and above 1 later; cumulative curves that meet (Taquet 2024: 5.75% against 5.94% at 6 years, from the authors' posted data).",
        evidence: "Open. Taquet's curves converge by 6 years, where its later group's data are thin; the Welsh effects persist to 7 to 8 years; Polisky reports that the live vaccine's association wanes with its protection against shingles.",
        needs: None,
    },
    Story {
        label: "5",
        head: ["selection,", "follow-up"],
        name: "selection",
        title: "5. Selection through survival, follow-up or later events",
        real: false,
        arrows: &[(v::D, v::C), (v::K, v::C), (v::C, v::DX), (v::D, v::M), (v::M, v::DX), (v::V, v::N), (v::H, v::N)],
        gates: &[(v::C, v::DX), (v::M, v::DX), (v::H, v::N)],
        gist: r"Who stays in the data, or who is counted, depends on dementia, health or later behaviour: people leave the record ($C$), die first ($\dagger$), or are counted only once they have had a second dose ($N$).",
        predicts: "Results that move with censoring weights and with the composite of dementia or death; stronger estimates when later behaviour is conditioned on (Taquet 2024, excluding those who received both vaccines: 0.79).",
        evidence: "Rayens and Dos Reis start follow-up at the second dose, which requires surviving to it. In Taquet 2024 the later group lost about 1.8 times as many people in years 1 and 2 (calculated for this map from the paper's Supplementary Table 2); the paper does not comment on it.",
        needs: Some(Layer::Record),
    },
    Story {
        label: "6",
        head: ["calendar", "time"],
        name: "calendar time",
        title: "6. Calendar-time artefacts",
        real: false,
        arrows: &[(v::P, v::F), (v::F, v::K), (v::K, v::DX)],
        gates: &[(v::P, v::F)],
        gist: r"Groups vaccinated in different periods are followed through different calendar years, and diagnosis and coding change over the years ($P \to F \to K \to D^*$): the pandemic, a new coding system.",
        predicts: "The same 'effect' in a placebo switch, a comparison of periods among people given a vaccine that did not change; a hazard ratio over time that tracks calendar events such as the pandemic.",
        evidence: "In the 2017 switch the pandemic falls early in the later group's follow-up and late in the earlier group's. Its analysis with follow-up ended before the pandemic still found a reduction (hazard ratio 0.74, 0.62–0.90); a placebo switch for dementia has not been reported.",
        needs: None,
    },
    Story {
        label: "7",
        head: ["chance,", "flexibility"],
        name: "chance",
        title: "7. Chance, flexibility and publication bias",
        real: false,
        arrows: &[],
        gates: &[],
        gist: "No arrows: random error, and the choice among many reasonable analyses, can produce an association with nothing behind it.",
        predicts: "Large replications that regress to the null; implied effects implausibly big; heterogeneity across settings.",
        evidence: r"Cannot be dismissed: Kivimäki and others report implausibly large implied effects, the large English birth-date study is null, and the meta-analyses show $I^2$ of about 97 to 100%. Chance is an unlikely explanation for the primary result of the 2017 switch itself.",
        needs: None,
    },
];

// ----- what each design can tell apart --------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mark {
    /// The design can tell this explanation from the others.
    Can,
    /// In part, or only under a condition.
    Partly,
    Cannot,
    /// The design does not decide it: it depends on the follow-up or the analysis,
    /// or the dossier's table does not place the design here.
    Open,
}

impl Mark {
    fn says(self) -> &'static str {
        match self {
            Mark::Can => "can tell it apart",
            Mark::Partly => "partly, or only if",
            Mark::Cannot => "cannot tell it apart",
            Mark::Open => "not decided by the design",
        }
    }
}

struct Cell {
    mark: Mark,
    text: &'static str,
}

const fn cell(mark: Mark, text: &'static str) -> Cell {
    Cell { mark, text }
}

const NOT_PLACED: &str = "Not decided by the kind of design: it depends on how the particular study was run and analysed.";
const FOLLOW_UP: &str = "Depends on the length of follow-up, not on the design: short studies cannot see curves meet.";

/// Rows: `DESIGNS`; columns: `STORIES`. From the dossier's table of explanations (what
/// designs can and cannot tell each apart) and its account of each design.
const CELLS: [[Cell; 8]; 7] = [
    [
        cell(Mark::Cannot, "Cannot: comparisons of vaccinated with unvaccinated people are confounded."),
        cell(Mark::Cannot, "Cannot: it compares a whole vaccine with none, so nothing separates the immune route from the others, and the comparison is confounded."),
        cell(Mark::Cannot, "Cannot: adjusting for more covariates cannot prove that no unmeasured trait remains. A negative-control outcome is a partial check (Rayens: 0.94, 0.89–1.00)."),
        cell(Mark::Cannot, r"Cannot by design: as in an open-label trial, the vaccinated had a vaccination visit that the others did not ($V \to K$)."),
        cell(Mark::Open, FOLLOW_UP),
        cell(Mark::Open, "Depends on the analysis: censoring weights, or intention-to-treat against per-protocol. Starting follow-up at the second dose (Rayens; Dos Reis) requires surviving to it."),
        cell(Mark::Open, NOT_PLACED),
        cell(Mark::Open, NOT_PLACED),
    ],
    [
        cell(Mark::Partly, "Partly: comparing vaccines that protect against shingles to different degrees is one test of this explanation, if the choice of vaccine is unconfounded."),
        cell(Mark::Partly, "Partly: only a comparison of the same antigen with and without the adjuvant would tell (an AS01 vaccine against RSV against one without it); against influenza vaccine the antigens and the people differ too."),
        cell(Mark::Partly, "Partly: both groups chose a vaccine, so the bias shrinks; whatever decides which vaccine people choose remains."),
        cell(Mark::Open, NOT_PLACED),
        cell(Mark::Open, FOLLOW_UP),
        cell(Mark::Open, "Depends on the analysis; Polisky's comparisons of doses condition on completing the series."),
        cell(Mark::Open, NOT_PLACED),
        cell(Mark::Open, NOT_PLACED),
    ],
    [
        cell(Mark::Partly, "Partly: the two vaccines differ in protection against shingles (shingles itself: 0.65, 0.61–0.69), which this explanation predicts should matter; they differ in adjuvant too."),
        cell(Mark::Cannot, "Cannot: the recombinant vaccine brings antigen and adjuvant together, and the comparison has nothing that separates them."),
        cell(Mark::Partly, "Partly: everyone compared was vaccinated, and mortality (0.98) and a negative-control outcome (0.97) were similar; who came forward in each period may still differ in ways the records miss."),
        cell(Mark::Cannot, "Cannot rule it out: the early reduction could be an effect on the timing of diagnosis of disease already present. The authors assume that delays in diagnosis are similar in the two groups."),
        cell(Mark::Cannot, "Cannot: a delay, a pandemic artefact and an artefact of dementia already present at the start all fit the late rise of the hazard ratio above 1. The authors note that the rise was not seen in every analysis and needs replication."),
        cell(Mark::Cannot, "Cannot from these data: health records do not say why people leave them, and the later group lost about 1.8 times as many people in years 1 and 2 (calculated for this map from Supplementary Table 2). Done: follow-up aligned between the groups, 0.83 (0.79–0.87); dementia or death counted together, 0.93 (0.91–0.96)."),
        cell(Mark::Cannot, "Cannot by design: a comparison of periods without a placebo series. Checks done: six-month windows, 0.83 (0.76–0.92); follow-up ended before the pandemic, 0.74 (0.62–0.90). In 2026 the same group ran a placebo switch (a vaccine that did not change) for cardiovascular outcomes, 1.01 (0.99–1.03); none has been published for dementia."),
        cell(Mark::Partly, "Chance is an unlikely explanation for the primary result; no protocol or pre-registration is mentioned, so the choice of windows and covariates cannot be audited."),
    ],
    [
        cell(Mark::Partly, "Can tell a true effect from bias, if eligibility acts on dementia only through vaccination (the exclusion restriction), but not this route from the immune one."),
        cell(Mark::Cannot, "Cannot: it measures the whole effect of one vaccine against none, whatever the route."),
        cell(Mark::Can, "Can: eligibility, not choice, changes at the cut-off, so health and engagement are balanced across it."),
        cell(Mark::Partly, "Partly: with checks on other diagnoses (Wales: no effect except on rheumatological ones); effects on dementia deaths (Michalik; Xie) help, though deaths too depend partly on a recorded diagnosis."),
        cell(Mark::Partly, "Can, with long follow-up: the Welsh effects persist to 7 to 8 years."),
        cell(Mark::Open, "Depends on the analysis: censoring weights, or intention-to-treat against per-protocol."),
        cell(Mark::Can, "Can: both sides of the cut-off live through the same calendar time."),
        cell(Mark::Partly, "Small or exploratory analyses cannot; the largest replication (England, 6.3 million people, hospital records) found no effect on hospital-coded dementia."),
    ],
    [
        cell(Mark::Cannot, "Cannot: like other observational cohorts, it is open to unmeasured confounding."),
        cell(Mark::Cannot, "Cannot: one vaccine against none, antigen and adjuvant together."),
        cell(Mark::Cannot, "Cannot: it does not fix unmeasured confounding; the authors write that their negative controls 'suggest some residual confounding'."),
        cell(Mark::Cannot, "Cannot by design: the vaccinated had a vaccination visit that the others did not."),
        cell(Mark::Open, FOLLOW_UP),
        cell(Mark::Partly, "Partly: cloning and censoring weights handle when people start and stop, if the weights are right."),
        cell(Mark::Open, NOT_PLACED),
        cell(Mark::Open, NOT_PLACED),
    ],
    [
        cell(Mark::Partly, "Can tell a true effect from bias, but not this route from the immune one: the trial cannot separate antigen from adjuvant."),
        cell(Mark::Cannot, "Cannot: the trial gives antigen and adjuvant together, against no vaccine, so it cannot separate them."),
        cell(Mark::Can, "Can: chance, not health, decides who is vaccinated."),
        cell(Mark::Cannot, "Cannot: participants know their arm, so contact with the health service after vaccination can differ."),
        cell(Mark::Partly, "Only with long follow-up; its dementia outcome is at about 3 years."),
        cell(Mark::Partly, "Partly: randomisation does not stop people leaving; attrition remains a threat, though outcomes come from national registries."),
        cell(Mark::Can, "Can: both arms live through the same calendar time."),
        cell(Mark::Can, "Can: a large trial, registered before it started."),
    ],
    [
        cell(Mark::Partly, "Can tell a true effect from bias, but not this route from the immune one: the trial cannot separate antigen from adjuvant."),
        cell(Mark::Cannot, "Cannot: the trial gives antigen and adjuvant together, against placebo, so it cannot separate them. Only a trial with factorial or active arms could."),
        cell(Mark::Can, "Can: chance, not health, decides who is vaccinated."),
        cell(Mark::Can, r"Can, for the vaccination visit and later contact: placebo and blinding keep them alike. Fewer shingles episodes would still mean fewer visits ($R \to Z \to K$), a route the Welsh authors tested and argue is too small."),
        cell(Mark::Can, "Can: dementia is followed for up to 10 years."),
        cell(Mark::Partly, "Partly: randomisation does not stop people leaving; attrition remains a threat."),
        cell(Mark::Can, "Can: both arms live through the same calendar time."),
        cell(Mark::Can, "Can: a large trial, registered before it started."),
    ],
];

// ----- the studies ----------------------------------------------------------------------

/// How much of a source the dossier could read.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Read {
    Full,
    Part,
    Abstract,
    Registry,
    Title,
}

impl Read {
    /// In a word or two, for the panel's read-out.
    fn brief(self) -> &'static str {
        match self {
            Read::Full => "in full",
            Read::Part => "in part",
            Read::Abstract => "abstract",
            Read::Registry => "registry",
            Read::Title => "title only",
        }
    }

    fn says(self) -> &'static str {
        match self {
            Read::Full => "read in full, with its supplement and posted data",
            Read::Part => "read in part",
            Read::Abstract => "abstract only",
            Read::Registry => "registry record",
            Read::Title => "content not verified: title and disclosures only",
        }
    }
}

/// The groups of the list of studies, in order: one for each kind of design, and
/// one for the work that pools or questions the others.
const GROUPS: [&str; 7] = [
    "cohorts: vaccinated or not",
    "active comparators",
    "calendar switch",
    "birth-date cut-offs",
    "target-trial emulation",
    "randomised trials (not yet reported)",
    "reviews, re-analyses, commentaries",
];

struct Study {
    /// Its entry in `FACTS`.
    key: &'static str,
    /// First author and year, as the list shows it.
    short: &'static str,
    /// Its authors as the card names them.
    by: &'static str,
    group: usize,
    /// The design whose graph it is drawn on (none for a review).
    design: Option<usize>,
    place: &'static str,
    /// Where, from what data, how many.
    meta: &'static str,
    /// Its main estimate, in its own unit, for the list.
    estimate: &'static str,
    blocks: &'static [(&'static str, &'static str)],
    read: Read,
    cite: &'static str,
    doi: &'static str,
    /// Funding and competing interests, exactly as the dossier gives them.
    disclosure: &'static str,
    /// What the study itself holds beyond its design (a second dose), and the arrows
    /// that opens.
    held: &'static [usize],
    open: Pairs,
    pic: Option<Pic>,
}

const NO_DOI: &str = "";

const STUDIES: [Study; 34] = [
    Study {
        key: "eyting",
        short: "Eyting 2025",
        by: "Eyting, Xie, Michalik, Heß, Chung and Geldsetzer, 2025",
        group: 3,
        design: Some(BIRTH),
        place: "Wales",
        meta: "Wales, SAIL: about 80% of GP practices, linked to hospital and death records · 282,541 people",
        estimate: "receipt: 3.5 pp fewer (0.6–7.1)",
        blocks: &[
            ("who against whom", "People born just after 2 September 1933 against just before. Wales offered the live vaccine from 1 September 2013 to people aged 79; people already 80 were never eligible. Uptake: 47.2% among those born in the week after, 0.01% in the week before."),
            ("outcome", "A new dementia diagnosis in primary care, hospital or death records over 7 years."),
            ("reported", "Effect of eligibility: 1.3 pp fewer (0.2–2.7), 8.5% relative. Effect of receiving the vaccine: 3.5 pp fewer (0.6–7.1), 20.0% relative (6.5–33.4). A difference-in-differences version: −3.1 pp (−5.8 to −0.4). Stronger in women (the numbers by sex, in a supplementary table, were not read for this map)."),
            ("checks", "Balance of prior diagnoses, preventive care and census education; placebo cut-offs in other years; no jump before the programme; no effect on other common diseases; bandwidths from 0.25 to 2 times the main one; grace periods of 0 to 12 months; dementia defined by prescriptions instead; adjustment for health-care use."),
            ("possible threats", "Under-diagnosis (the authors argue it is alike either side of the cut-off); valid near ages 79 to 80; chance; the difference between the sexes is exploratory."),
            ("re-analysis", "Kivimäki and others: an implied reduction of 18% (−0 to 35) among the vaccinated."),
        ],
        read: Read::Part,
        cite: "Nature 641:438–446 (2025); preprint medRxiv 2023",
        doi: "10.1038/s41586-025-08800-x",
        disclosure: "Declares no competing interests.",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "michalik",
        short: "Michalik 2023",
        by: "Michalik and others, 2023 (preprint)",
        group: 3,
        design: Some(BIRTH),
        place: "England, Wales",
        meta: "England and Wales, death certificates by month of birth · 5,077,426 people",
        estimate: "deaths: 0.38 pp fewer (0.08–0.68)",
        blocks: &[
            ("who against whom", "Birth cohorts eligible for the live vaccine against those not eligible, either side of 2 September 1933, by month of birth."),
            ("outcome", "Deaths with dementia as the underlying cause, over 9 years."),
            ("reported", "Effect of eligibility: 0.38 pp fewer (0.08–0.68), 4.8% relative. Women −0.62 (−1.06 to −0.19); men −0.11 (−0.51 to 0.28)."),
            ("checks", "Balance of deaths before the programme; the effect is present 'only since the year in which the HZ vaccination program was implemented'."),
            ("possible threats", "How causes of death are coded; birth month, not week, decides the side."),
        ],
        read: Read::Abstract,
        cite: "medRxiv 2023",
        doi: "10.1101/2023.09.08.23295225",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "xie",
        short: "Xie 2025",
        by: "Xie, Eyting, Bommer, Ahmed and Geldsetzer, 2025",
        group: 3,
        design: Some(BIRTH),
        place: "Wales",
        meta: "Wales, SAIL",
        estimate: "impairment: 3.1 pp fewer (1.0–6.2)",
        blocks: &[
            ("who against whom", "As in Eyting and others: born either side of 2 September 1933."),
            ("outcome", "A new diagnosis of mild cognitive impairment over 9 years; and, among 14,350 people already living with dementia, deaths due to dementia."),
            ("reported", "Mild cognitive impairment: effect of eligibility 1.5 pp fewer (0.5–2.9), of receipt 3.1 pp fewer (1.0–6.2). Dementia deaths: eligibility 8.5 pp fewer (0.6–18.5), receipt 29.5 pp fewer (0.6–62.9). All-cause mortality among people with dementia: receipt 22.7 pp fewer (6.5–42.8). Larger in women; in men 'statistically indistinguishable from zero' (tests of the difference: P = 0.029 for impairment, 0.039 for deaths)."),
            ("checks", "Impairment defined to exclude dementia within 3 to 6 months; adjustment for an imbalance; difference in differences."),
            ("possible threats", "Very wide intervals; large implied effects (Kivimäki and others: 41% for impairment, 56% for dementia mortality); an imbalance in prior ischaemic heart disease, which the authors judge likely to be chance."),
        ],
        read: Read::Part,
        cite: "Cell 188:7049–7064 (2025)",
        doi: "10.1016/j.cell.2025.11.007",
        disclosure: "Declares no competing interests.",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "pomirchy25",
        short: "Pomirchy 2025",
        by: "Pomirchy, Bommer, Pradella, Michalik, Peters and Geldsetzer, 2025",
        group: 3,
        design: Some(BIRTH),
        place: "Australia",
        meta: "Australia, GP records of 65 practices · 101,219 people",
        estimate: "ITT: 1.8 pp fewer (0.4–3.3)",
        blocks: &[
            ("who against whom", "People born either side of 2 November 1936, by week of birth (people aged 70 to 79 became eligible on 1 November 2016). Eligibility raised ever-vaccination by 16.4 pp (13.2–19.5)."),
            ("outcome", "A new dementia diagnosis in GP records over 7.4 years."),
            ("reported", "Effect of eligibility: 1.8 pp fewer (0.4–3.3). Not divided by uptake, because vaccination was under-recorded. No significant difference between the sexes."),
            ("checks", "Placebo cut-offs 3 years either side null; no effect on 15 common diagnoses or on preventive services; bandwidths of 1 to 9 years; grace periods of 0 to 156 weeks."),
            ("possible threats", "Under-recording (1.4% with a recorded diagnosis against about 8.4% estimated prevalence over 65); practices not chosen at random; possible collider bias from restrictions on the sample; a large bandwidth."),
            ("re-analysis", "Kivimäki and others divide the effect by the uptake jump: an implied reduction of 177% (28–301) among the vaccinated."),
        ],
        read: Read::Part,
        cite: "JAMA 333:2083–2092 (2025)",
        doi: "10.1001/jama.2025.5013",
        disclosure: "Declares no competing interests, apart from an outside grant to Dr Peters.",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "pomirchy26",
        short: "Pomirchy 2026",
        by: "Pomirchy, Chung, Bommer, Strobel and Geldsetzer, 2026",
        group: 3,
        design: Some(BIRTH),
        place: "Ontario",
        meta: "Canada (Ontario), primary-care records · 232,124 people born in Ontario",
        estimate: "ITT: 2.0 pp fewer (0.4–3.5)",
        blocks: &[
            ("who against whom", "People born either side of 1 January 1946 (the main cut-off) and 1 January 1945. Uptake jumps, from a survey of other people: 27.4 pp (0.0–54.7) and 13.9 pp (−5.8 to 33.6). Also synthetic difference-in-differences and synthetic control against other provinces."),
            ("outcome", "A new dementia diagnosis over 5.5 years."),
            ("reported", "Effect of eligibility: 2.0 pp fewer (0.4–3.5) at the 1946 cut-off; 2.0 pp fewer (0.2–3.8) at 1945. Ontario's eligible cohorts lower than the same cohorts in other provinces."),
            ("checks", "The same cut-offs in provinces without the programme: −0.93 pp (−3.12 to 1.26) and +0.9 (−1.71 to 3.52); placebo cut-offs; 15 common diagnoses; trends before the programme."),
            ("possible threats", "Uptake measured in a different population, so no estimate per vaccinee; 5.5 years in the early 70s; changes over time that differ between provinces."),
            ("re-analysis", "Kivimäki and others: implied reductions of 84% (18–219) and 111% (15–303) among the vaccinated. Letters in Lancet Neurology (Taquet, Todd and Harrison; Kivimäki, Livingston and Sipilä) and a reply by Geldsetzer: contents not verified."),
        ],
        read: Read::Part,
        cite: "Lancet Neurol 25:170–180 (2026)",
        doi: "10.1016/S1474-4422(25)00455-7",
        disclosure: "Declares no competing interests.",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "hamilton",
        short: "Hamilton 2026",
        by: "Hamilton, Pinot de Moira, Bracher-Smith, Michalik and others, with Geldsetzer and Davey Smith, 2026 (preprint)",
        group: 3,
        design: Some(BIRTH),
        place: "England",
        meta: "England, Hospital Episode Statistics · 6,340,214 people · not peer reviewed",
        estimate: "ITT: −0.06 pp (−0.40 to 0.27)",
        blocks: &[
            ("who against whom", "Birth months either side of September 1933; the effect of eligibility only."),
            ("outcome", "Dementia coded in hospital records over 8 years."),
            ("reported", "Shingles: −0.12 pp (−0.153 to −0.079): the design finds the effect it should. Dementia: −0.06 pp (−0.40 to 0.27), roughly −3% to +2% of a 13.6% risk. In the Welsh data the authors reproduce −1.30 (−2.71 to −0.19); with hospital codes only it is −0.87 (−2.13 to 0.26)."),
            ("checks", "Placebo cut-offs; negative controls; a second hospital extract, also near null."),
            ("possible threats", "Hospital codes only (no primary-care, death or prescription data); birth month, not week; about 18 months of look-back; denominators about 32% above official population estimates."),
            ("how it reads", "The authors say the Welsh and English results may be 'concordant', but that the Ontario and Australian effects 'cannot easily be reconciled' with theirs. Two authors of the Welsh study are among its authors."),
        ],
        read: Read::Part,
        cite: "medRxiv July 2026 (not peer reviewed)",
        doi: "10.64898/2026.07.20.26358345",
        disclosure: "Declares no competing interests.",
        held: &[],
        open: &[],
        pic: Some(Pic::Hamilton),
    },
    Study {
        key: "taquet24",
        short: "Taquet 2024",
        by: "Taquet, Dercon, Todd and Harrison, 2024",
        group: 2,
        design: Some(SWITCH),
        place: "USA",
        meta: "USA, TriNetX health records of 62 organisations · 103,837 in each matched group",
        estimate: "RMTL ratio 0.83 (0.79–0.87)",
        blocks: &[
            ("who against whom", "People first vaccinated at 65 or older from November 2017 to October 2020 (95% recombinant) against October 2014 to September 2017 (98% live); those vaccinated in October 2017 left out; matched one to one on 60 recorded characteristics."),
            ("outcome", "A first dementia diagnosis from 3 months to 6 years after vaccination."),
            ("reported", "Ratio of restricted mean time lost 0.83 (0.79–0.87): within the window, the later group spent on average 17% less time after a diagnosis. The paper also reports 164 (124–205) additional diagnosis-free days among those affected. Women 0.78, men 0.87."),
            ("checks", "Shingles as a positive control: 0.65 (0.61–0.69). Mortality: 0.98 (0.95–1.01). Dementia or death: 0.93 (0.91–0.96). A negative-control outcome: 0.97 (0.91–1.03). Six-month windows: 0.83 (0.76–0.92). Follow-up ended before the pandemic: hazard ratio 0.74 (0.62–0.90). Another matching method: 0.82 (0.77–0.87)."),
            ("open questions", "Not yet published for dementia: a placebo switch with an unchanged vaccine; follow-up aligned in calendar time; trajectories before vaccination; why people left the record (the later group lost about 1.8 times as many in years 1 and 2, calculated for this map from Supplementary Table 2). The authors write: 'This study is observational, and causality cannot be demonstrated.'"),
            ("next", "The next scene, Two vaccines, one switch, takes this study apart."),
        ],
        read: Read::Full,
        cite: "Nat Med 30:2777–2781 (2024)",
        doi: "10.1038/s41591-024-03201-5",
        disclosure: "Funding: 'This research was supported by the NIHR (https://www.nihr.ac.uk/) Oxford Health Biomedical Research Centre (NIHR203316).' 'The funder of the study had no role in study design, data collection, data analysis, data interpretation or writing of the paper.' Competing interests: 'J.A.T. is a consultant for GSK and co-director of the Oxford-GSK Institute for Molecular and Computational Medicine. GSK had no involvement of any kind in this study and were not aware of it until after the paper was accepted. The other authors declare no competing interests.'",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "taquet25",
        short: "Taquet 2025",
        by: "Taquet, Todd and Harrison, 2025",
        group: 1,
        design: Some(COMPARATOR),
        place: "USA",
        meta: "USA, TriNetX · 436,788 people",
        estimate: "AS01 RSV vaccine 0.71 (0.61–0.83)",
        blocks: &[
            ("who against whom", "People aged 60 or older from May 2023 given an AS01-adjuvanted vaccine (against RSV, mostly Arexvy, the brand recorded for only 44%; the recombinant shingles vaccine; or both) against people given influenza vaccine, matched on 66 covariates."),
            ("outcome", "A dementia diagnosis from 3 to 18 months."),
            ("reported", "Against influenza vaccine: RSV vaccine 0.71 (0.61–0.83); recombinant shingles vaccine 0.82 (0.74–0.91); both 0.63 (0.55–0.72) (the measure was not noted when this source was read). The authors: 'No difference was observed between the two AS01-adjuvanted vaccines, suggesting that the AS01 adjuvant itself plays a direct role in lowering dementia risk'."),
            ("checks", "A negative-control outcome (null); shingles as a positive control."),
            ("the exchange", "Williams and others (2025) reply: the RSV brand was recorded for only 44%, with no analysis by brand; the RSV vaccine has half the AS01 dose of the shingles vaccine yet a similar association, 'no evidence for a dose-response'; many vaccines without an adjuvant show similar associations; 'residual confounding … cannot be ruled out'. The authors answer that analysis by brand lacked power, and suggest a 'saturation effect'."),
        ],
        read: Read::Part,
        cite: "npj Vaccines 10:130 (2025); critique 10:255; reply 10:256",
        doi: "10.1038/s41541-025-01172-3",
        disclosure: "The critique by Williams and others was written by Pfizer employees; Pfizer makes the non-adjuvanted RSV vaccine Abrysvo.",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "polisky",
        short: "Polisky 2025",
        by: "Polisky and others, with Geldsetzer, van Duijn and Schwab, 2025",
        group: 1,
        design: Some(COMPARATOR),
        place: "USA",
        meta: "USA, Optum health records of over 100 million people",
        estimate: "18% lower risk than live (2–31%)",
        blocks: &[
            ("who against whom", "Matched on about 400 covariates: the live or recombinant vaccine against the pneumococcal vaccine PPSV23; two or more doses of the recombinant against the live vaccine; two or more doses against one; recurrent against single shingles."),
            ("outcome", "A dementia diagnosis, up to 9 years."),
            ("reported", "Recombinant (two or more doses) against live: risk 18% lower at 5 years (2–31%). Live against PPSV23: 33% lower (30–35%) at 3 years. Recombinant (two or more) against PPSV23: 27% lower (20–34%). Two or more doses against one: 24% lower (18–29%) at 3 years. Recurrent shingles: 7 to 9% higher risk than a single episode."),
            ("checks", "A 'negative control by temporal shift' (the start moved a year earlier): 'little evidence for residual confounding … for most pairs'."),
            ("possible threats", "Residual confounding; comparisons of doses condition on completing the series."),
        ],
        read: Read::Part,
        cite: "Nat Med 31:4172–4179 (2025)",
        doi: "10.1038/s41591-025-03972-5",
        disclosure: "Most authors are GSK employees. 'P.G. serves on the GSK advisory board, \"Herpes zoster and dementia\"'; the start date of that role is not given in what was read for this map.",
        held: &[v::N],
        open: &[(v::V, v::N), (v::H, v::N)],
        pic: None,
    },
    Study {
        key: "scherrer",
        short: "Scherrer 2021",
        by: "Scherrer, Salas, Wiemken, Hoft, Jacobs and Morley, 2021",
        group: 0,
        design: Some(COHORT),
        place: "USA",
        meta: "USA, Veterans Health Administration (136,016) and MarketScan claims (172,790)",
        estimate: "HR 0.69 (0.67–0.72)",
        blocks: &[
            ("who against whom", "People aged 65 or older given the live vaccine against unvaccinated people, weighted by propensity score."),
            ("outcome", "Incident dementia."),
            ("reported", "Hazard ratio 0.69 (0.67–0.72) among veterans; 0.65 (0.57–0.74) in MarketScan; none at ages 65 to 69 in MarketScan."),
            ("checks", "Replication in two data sets; E-values (in a conference abstract, 2022)."),
            ("possible threats", "The healthy-vaccinee effect; the veterans' cohort is 4% female."),
        ],
        read: Read::Abstract,
        cite: "PLoS One 16:e0257405 (2021)",
        doi: "10.1371/journal.pone.0257405",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "wiemken",
        short: "Wiemken 2022",
        by: "Wiemken and others, 2022",
        group: 0,
        design: Some(COHORT),
        place: "USA",
        meta: "USA, veterans and MarketScan",
        estimate: "both vaccines 0.50 (0.43–0.59)",
        blocks: &[
            ("who against whom", "People given both the zoster and the Tdap vaccine, one of them, or neither; boosted propensity scores."),
            ("outcome", "Incident dementia."),
            ("reported", "Both against neither: 0.50 (0.43–0.59) among veterans; 0.58 (0.38–0.89) in MarketScan (the measure was not noted when this source was read)."),
            ("possible threats", "The healthy-vaccinee effect: more vaccines, lower risk."),
        ],
        read: Read::Abstract,
        cite: "J Am Geriatr Soc 70:1157–1168 (2022)",
        doi: "10.1111/jgs.17606",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "schnier",
        short: "Schnier 2022",
        by: "Schnier, Janbek, Lathe and Haas, 2022",
        group: 0,
        design: Some(COHORT),
        place: "Wales",
        meta: "Wales, national data, 2013 to 2020",
        estimate: "aHR 0.72 (0.69–0.75)",
        blocks: &[
            ("who against whom", "Vaccinated against unvaccinated people."),
            ("outcome", "Incident dementia."),
            ("reported", "Adjusted hazard ratio 0.72 (0.69–0.75)."),
            ("possible threats", "The authors: it 'may reflect selection bias'; vaccination was also associated with lower all-cause mortality."),
        ],
        read: Read::Abstract,
        cite: "Alzheimers Dement (TRCI) 8:e12293 (2022)",
        doi: "10.1002/trc2.12293",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "lophatananon21",
        short: "Lophatananon 2021",
        by: "Lophatananon and others, 2021",
        group: 0,
        design: Some(COHORT),
        place: "UK",
        meta: "UK Biobank, a case-control study nested in it",
        estimate: "OR 0.808 (0.657–0.993)",
        blocks: &[
            ("who against whom", "The live vaccine (Zostavax) against no zoster vaccine, in a case-control design nested in UK Biobank."),
            ("outcome", "Dementia 3 years or more after assessment."),
            ("reported", "Odds ratio 0.808 (0.657–0.993)."),
            ("possible threats", "Healthy-volunteer and healthy-vaccinee effects."),
        ],
        read: Read::Abstract,
        cite: "BMJ Open 11:e045871 (2021)",
        doi: "10.1136/bmjopen-2020-045871",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "lophatananon23",
        short: "Lophatananon 2023",
        by: "Lophatananon and others, 2023",
        group: 0,
        design: Some(COHORT),
        place: "England",
        meta: "England, CPRD Aurum primary care with hospital and death records",
        estimate: "aHR 0.78 (0.77–0.79)",
        blocks: &[
            ("who against whom", "854,745 zoster-vaccinated people against matched unvaccinated people."),
            ("outcome", "Dementia."),
            ("reported", "Adjusted hazard ratio 0.78 (0.77–0.79); for influenza vaccine, used as a comparison exposure, 0.96 (0.94–0.97)."),
            ("possible threats", "The healthy-vaccinee effect."),
        ],
        read: Read::Abstract,
        cite: "BMC Public Health 23:1903 (2023)",
        doi: "10.1186/s12889-023-16768-4",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "harris",
        short: "Harris 2023",
        by: "Harris and others, 2023",
        group: 0,
        design: Some(COHORT),
        place: "USA",
        meta: "USA, Optum Clinformatics claims",
        estimate: "RR 0.75 (0.73–0.76), Alzheimer's",
        blocks: &[
            ("who against whom", "People given a zoster, Tdap or Td, or pneumococcal vaccine against unvaccinated people, matched by propensity score."),
            ("outcome", "Alzheimer's disease over 8 years."),
            ("reported", "Risk ratio 0.75 (0.73–0.76) for the zoster vaccine; 0.70 for Tdap; 0.73 for pneumococcal."),
            ("possible threats", "The healthy-vaccinee effect: associations of similar size for every vaccine."),
        ],
        read: Read::Abstract,
        cite: "J Alzheimers Dis 95:703–718 (2023)",
        doi: "10.3233/JAD-221231",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "ukraintseva",
        short: "Ukraintseva 2024",
        by: "Ukraintseva and others, 2024",
        group: 0,
        design: Some(COHORT),
        place: "USA",
        meta: "USA, Health and Retirement Study with Medicare",
        estimate: "15–21% lower risk, no interval",
        blocks: &[
            ("who against whom", "People given a shingles or pneumococcal vaccine at ages 65 to 75 against those not, in a 'pseudorandomized sample'."),
            ("outcome", "Alzheimer's disease."),
            ("reported", "15 to 21% lower risk (the abstract's range across both vaccines). No interval appears in what was read for this map."),
            ("possible threats", "The healthy-vaccinee effect."),
        ],
        read: Read::Abstract,
        cite: "Exp Gerontol 190:112411 (2024)",
        doi: "10.1016/j.exger.2024.112411",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "tang",
        short: "Tang 2025",
        by: "Tang, Ray, Arnold and Acharya, 2025",
        group: 0,
        design: Some(COHORT),
        place: "USA",
        meta: "USA, Optum Labs · 4.5 million people",
        estimate: "full series aHR 0.68 (0.67–0.70)",
        blocks: &[
            ("who against whom", "People aged 50 or older with the full or partial recombinant series against unvaccinated people, exposure counted as it happens from 1 January 2018."),
            ("outcome", "Dementia."),
            ("reported", "Full series: adjusted hazard ratio 0.68 (0.67–0.70); partial: 0.89 (0.87–0.92)."),
            ("checks", "An analysis stratified by birth cohort."),
            ("possible threats", "The authors list possible immortal-time and healthy-vaccinee bias."),
        ],
        read: Read::Part,
        cite: "Vaccine 46:126673 (2025)",
        doi: "10.1016/j.vaccine.2024.126673",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "rayens",
        short: "Rayens 2026",
        by: "Rayens and others, 2026",
        group: 0,
        design: Some(COHORT),
        place: "USA",
        meta: "USA, Kaiser Permanente Southern California",
        estimate: "aHR 0.49 (0.46–0.51)",
        blocks: &[
            ("who against whom", "65,800 people aged 65 or older with two doses of the recombinant vaccine against 263,200 unvaccinated people, matched and weighted; follow-up starts at the second dose."),
            ("outcome", "Dementia."),
            ("reported", "Adjusted hazard ratio 0.49 (0.46–0.51) against the unvaccinated; 0.73 (0.67–0.79) against Tdap recipients."),
            ("checks", "A composite negative-control outcome: 0.94 (0.89–1.00); an E-value of 3.50; chart review confirmed 94 of 100 dementia codes."),
            ("possible threats", "The healthy-vaccinee effect; starting at the second dose requires surviving to it."),
        ],
        read: Read::Part,
        cite: "Nat Commun 17:2056 (2026)",
        doi: "10.1038/s41467-026-69289-0",
        disclosure: "GSK co-authors.",
        held: &[v::N],
        open: &[(v::V, v::N), (v::H, v::N)],
        pic: Some(Pic::Rayens),
    },
    Study {
        key: "dosreis",
        short: "Dos Reis 2026",
        by: "Dos Reis and others, 2026",
        group: 0,
        design: Some(COHORT),
        place: "USA",
        meta: "USA, Medicare",
        estimate: "HR 0.67 (0.65–0.68), first 3 years",
        blocks: &[
            ("who against whom", "People with two doses of the recombinant vaccine (followed from the second) against unvaccinated people (followed from a preventive visit), matched and weighted."),
            ("outcome", "Dementia, Alzheimer's disease, vascular dementia."),
            ("reported", "Hazard ratio 0.67 (0.65–0.68) up to 3 years; 0.74 (0.69–0.79) after 3 years."),
            ("possible threats", "Different events start follow-up in the two groups; the healthy-vaccinee effect."),
        ],
        read: Read::Abstract,
        cite: "Alzheimers Dement 22:e71407 (2026)",
        doi: "10.1002/alz.71407",
        disclosure: "GSK co-authors.",
        held: &[v::N],
        open: &[(v::V, v::N), (v::H, v::N)],
        pic: None,
    },
    Study {
        key: "hayes",
        short: "Hayes 2026",
        by: "Hayes and others, 2026",
        group: 4,
        design: Some(EMULATION),
        place: "USA",
        meta: "USA, Medicare and nursing-home records · 509,926 people",
        estimate: "RR 0.76 (0.69–0.84)",
        blocks: &[
            ("who against whom", "Patients newly admitted to skilled nursing: at least one dose of the recombinant vaccine within 12 months against none (1.73% vaccinated, 87% of them after discharge); clones, censoring and weights."),
            ("outcome", "A validated dementia diagnosis over 4 years."),
            ("reported", "Risk 18.8% against 24.6%: 5.8 pp lower (3.9–7.5); risk ratio 0.76 (0.69–0.84). Weaker in men and in people who had had the live vaccine."),
            ("checks", "Negative controls, which the authors write 'suggest some residual confounding' (their details were not verified for this map)."),
            ("possible threats", "A frail population, in which healthy-vaccinee selection may be strong."),
        ],
        read: Read::Abstract,
        cite: "Ann Intern Med 179:958–967 (2026)",
        doi: "10.7326/ANNALS-25-04689",
        disclosure: "GSK funded.",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "oh",
        short: "Oh 2026",
        by: "Oh and others, 2026",
        group: 0,
        design: Some(COHORT),
        place: "South Korea",
        meta: "South Korea, linked national databases",
        estimate: "aHR 0.88 (0.87–0.90)",
        blocks: &[
            ("who against whom", "People aged 50 or older given the live vaccine against unvaccinated people, with overlap weighting."),
            ("outcome", "Memory disorders; Alzheimer's disease."),
            ("reported", "Adjusted hazard ratio 0.88 (0.87–0.90); Alzheimer's 0.75 (0.71–0.78); restricted mean survival time 15.5 and 11.3 days longer per decade. Attenuation over time reported."),
            ("possible threats", "The healthy-vaccinee effect."),
        ],
        read: Read::Abstract,
        cite: "Alzheimers Dement 22:e71410 (2026)",
        doi: "10.1002/alz.71410",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "rose",
        short: "Rose 2026",
        by: "Rose and others, 2026 (preprint)",
        group: 0,
        design: Some(COHORT),
        place: "USA",
        meta: "USA, N3C · 1.48 million people",
        estimate: "HR 0.63 (0.60–0.65)",
        blocks: &[
            ("who against whom", "People given the recombinant vaccine against not, with and without COVID-19; adjusted only for age, sex and race."),
            ("outcome", "Late-onset Alzheimer's disease over 5 years."),
            ("reported", "Hazard ratio 0.63 (0.60–0.65) without COVID-19; 0.77 (0.73–0.81) with it."),
            ("possible threats", "Minimal adjustment."),
        ],
        read: Read::Abstract,
        cite: "medRxiv 2026",
        doi: "10.64898/2026.01.21.26344555",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "shah",
        short: "Shah 2024",
        by: "Shah and others, 2024",
        group: 6,
        design: None,
        place: "meta-analysis",
        meta: "Meta-analysis, random effects",
        estimate: "pooled OR 0.84 (0.50–1.43)",
        blocks: &[("reported", r"Pooled odds ratio 0.84 (0.50–1.43), P = 0.53, $I^2 = 100\%$: the interval includes 1."), ("what pooling adds", "No identification of its own: pooling studies that share a possible bias gives a precise estimate with the same bias.")],
        read: Read::Abstract,
        cite: "Brain Behav 14:e3415 (2024)",
        doi: "10.1002/brb3.3415",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "yin",
        short: "Yin 2025",
        by: "Yin, Deng and Liu, 2025",
        group: 6,
        design: None,
        place: "meta-analysis",
        meta: "Meta-analysis of 5 cohorts",
        estimate: "pooled HR 0.71 (0.66–0.76)",
        blocks: &[
            ("reported", r"Pooled hazard ratio 0.71 (0.66–0.76), $I^2 = 97.15\%$."),
            ("what pooling adds", "The cohorts pooled compare vaccinated with unvaccinated people, so they share the possibility of healthy-vaccinee bias; pooling would carry it into a precise estimate."),
        ],
        read: Read::Abstract,
        cite: "J Alzheimers Dis 106:1232–1241 (2025)",
        doi: "10.1177/13872877251351593",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "marra",
        short: "Marra 2025",
        by: "Marra and others, 2025",
        group: 6,
        design: None,
        place: "review",
        meta: "Systematic review of 18 studies",
        estimate: "pooled RR 0.68 (0.56–0.83)",
        blocks: &[("reported", r"Vaccination: pooled risk ratio 0.68 (0.56–0.83), $I^2 = 99\%$. Shingles infection: 1.14 (1.04–1.25).")],
        read: Read::Abstract,
        cite: "Hum Vaccin Immunother 21:2546741 (2025)",
        doi: "10.1080/21645515.2025.2546741",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "zhang",
        short: "Zhang 2026",
        by: "Zhang and others, 2026",
        group: 6,
        design: None,
        place: "meta-analysis",
        meta: "Meta-analysis of 13 studies, one estimate per data source, registered in advance (PROSPERO)",
        estimate: "pooled 0.72 (0.65–0.81)",
        blocks: &[
            ("reported", r"Pooled 0.72 (0.65–0.81), $I^2 = 97.1\%$ (the measure was not noted when this source was read). The two direct comparisons of the recombinant with the live vaccine, reported separately: 0.83 (Taquet, a ratio of restricted mean time lost) and a risk ratio of 0.82 (0.69–0.98) (Polisky)."),
            ("the authors", "Residual confounding 'preclude[s] causal interpretation'; quasi-experimental estimates are reported separately."),
        ],
        read: Read::Abstract,
        cite: "Vaccines 14:685 (2026), PROSPERO CRD420261326042",
        doi: "10.3390/vaccines14080685",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "heen",
        short: "Heen 2026",
        by: "Heen and others, 2026 (Norwegian and Swedish public-health institutes)",
        group: 6,
        design: None,
        place: "review",
        meta: "Systematic review of the recombinant vaccine, graded for certainty (GRADE)",
        estimate: "'may be associated with a reduced risk'",
        blocks: &[("conclusion", "'RZV may be associated with a reduced risk of dementia (low CoE)': low certainty of evidence. Healthy-vaccinee bias noted.")],
        read: Read::Abstract,
        cite: "Rev Med Virol 36:e70203 (2026)",
        doi: "10.1002/rmv.70203",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "hiqa",
        short: "HIQA 2026",
        by: "HIQA (Ireland), 2026",
        group: 6,
        design: None,
        place: "review",
        meta: "Scoping review, 20 May 2026",
        estimate: "'lacks robustness and causal certainty'",
        blocks: &[("conclusion", "'the current evidence base lacks robustness and causal certainty'.")],
        read: Read::Part,
        cite: "HIQA (Ireland), scoping review, 20 May 2026",
        doi: NO_DOI,
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "danzoster",
        short: "DAN-ZOSTER",
        by: "DAN-ZOSTER, NCT07485283",
        group: 5,
        design: Some(OPEN_TRIAL),
        place: "Denmark",
        meta: "Denmark · 168,460 enrolled",
        estimate: "not yet reported",
        blocks: &[
            ("design", "Pragmatic, randomised, parallel, open label: two doses of the recombinant vaccine, 2 to 6 months apart, against no vaccination; outcomes from national registries."),
            ("who", "Aged 65 or older; excludes prior dementia, prior zoster vaccination, and chronic inflammatory rheumatic disease treated with immunosuppression."),
            ("primary outcomes", "Hospitalisation for major cardiovascular events (about 1 year) and new dementia (Alzheimer's, vascular or unspecified; about 3 years)."),
            ("status", "Started 28 April 2026; active, not recruiting; primary completion estimated June 2029 (registry updated 14 October 2026). Not yet reported."),
        ],
        read: Read::Registry,
        cite: "ClinicalTrials.gov NCT07485283; EU CT 2025-524944-37-00",
        doi: NO_DOI,
        disclosure: "Sponsor-investigator: Tor Biering-Sørensen (Herlev and Gentofte Hospital, Denmark). Collaborator: GlaxoSmithKline.",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "finland",
        short: "Finnish trial",
        by: "GSK study 300889 in Finland, NCT07502560",
        group: 5,
        design: Some(BLIND_TRIAL),
        place: "Finland",
        meta: "Finland · 33,609 planned",
        estimate: "not yet reported",
        blocks: &[
            ("design", "Phase IV, pragmatic, randomised 3 to 1 to the recombinant vaccine or placebo, observer-blind."),
            ("who", "Aged 76 or older; excludes dementia, an ongoing memory work-up, severe immunocompromise, nursing-home residents and prior zoster vaccination."),
            ("primary outcome", "The hazard ratio of an incident dementia diagnosis, from day 1, up to 10 years."),
            ("status", "Started 31 March 2026; recruiting; primary completion estimated 1 April 2030, study completion 31 March 2037 (registry updated 12 May 2026). Not yet reported."),
        ],
        read: Read::Registry,
        cite: "ClinicalTrials.gov NCT07502560; EU CT 2025-524598-17-00",
        doi: NO_DOI,
        disclosure: "Sponsor: GlaxoSmithKline. Collaborators: Finnish Vaccine Research; Dr Arto Palmu.",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "kivimaki",
        short: "Kivimäki 2026",
        by: "Kivimäki, Frank, Livingston and Sipilä, 2026",
        group: 6,
        design: Some(BIRTH),
        place: "re-analysis",
        meta: "A re-analysis of published numbers; no new data",
        estimate: "implied per vaccinee: 18% to 177%",
        blocks: &[
            ("what it does", "Divides each published effect of eligibility by its jump in uptake, relative to the baseline risk: the implied reduction among those vaccinated because eligible."),
            ("why it matters", "Under the assumptions of the division, the effect of eligibility equals the uptake jump times the effect among those vaccinated because eligible, and that effect cannot exceed their own baseline risk: an implied reduction above 100% is impossible."),
            ("reported", "Wales 18% (−0 to 35); Australia 177% (28–301); Canada 84% (18–219) and 111% (15–303); impairment 41%; dementia mortality 56%. The authors call some of these 'impossible or biologically implausible'."),
            ("the arithmetic", r"With Australia's two published numbers, $1.8 / 0.164 \approx 11.0$: about 11.0 pp fewer diagnoses per person vaccinated because eligible (calculated here)."),
            ("ways out", "Vaccination under-recorded, so the true uptake jump is larger; eligibility acting in other ways; chance; bias. The critique concerns scaling and plausibility; it does not identify a specific bias in the designs."),
            ("also", "It reports that 'no corresponding decline was observed in national data from the UK, Australia, or Canada' (not verified for this map). A reply by Geldsetzer: content not verified."),
        ],
        read: Read::Abstract,
        cite: "Brain Behav Immun 139:107021 (2026)",
        doi: "10.1016/j.bbi.2026.107021",
        disclosure: "",
        held: &[],
        open: &[],
        pic: Some(Pic::Kivimaki),
    },
    Study {
        key: "hamiltondaveysmith",
        short: "Hamilton, Davey Smith 2026",
        by: "Hamilton and Davey Smith, 2026",
        group: 6,
        design: None,
        place: "commentary",
        meta: "Commentary",
        estimate: "content not verified",
        blocks: &[
            ("title", "'Shingles vaccination and dementia: Do the numbers add up?'"),
            ("what is known", "Nothing of its content: only its title and disclosures were read, so this map says nothing about its arguments."),
        ],
        read: Read::Title,
        cite: "Brain Behav Immun 140:107072 (2026)",
        doi: "10.1016/j.bbi.2026.107072",
        disclosure: "University of Bristol has received GSK research funding unrelated to this work; Davey Smith has advised GSK and others.",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "geldsetzer",
        short: "Geldsetzer 2026",
        by: "Geldsetzer, 2026",
        group: 6,
        design: None,
        place: "commentary",
        meta: "Commentary",
        estimate: "content not verified",
        blocks: &[
            ("title", "'Why large-scale randomized trials of live-attenuated shingles vaccination for dementia prevention are urgently needed'"),
            ("what is known", "Only its title and disclosure were read. No randomised trial of the live vaccine for dementia is registered, as far as a search of ClinicalTrials.gov and ISRCTN made for this map found."),
        ],
        read: Read::Title,
        cite: "Nat Med 32:2720–2723 (2026)",
        doi: "10.1038/s41591-026-04474-8",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
    Study {
        key: "corsizuelli",
        short: "Corsi-Zuelli 2026",
        by: "Corsi-Zuelli and others, with Harrison and Taquet, 2026",
        group: 2,
        design: Some(SWITCH),
        place: "USA",
        meta: "The same 2017 switch, for heart and vessel outcomes, not dementia",
        estimate: "Tdap placebo switch 1.01 (0.99–1.03)",
        blocks: &[
            ("what it is", "The group behind the 2017 comparison used the same switch for cardiovascular outcomes, with ages 60 or older and April to September 2017 against April to September 2018."),
            ("checks it added", "A comparison of the same two periods among people given Tdap, a vaccine that did not change (ratio of restricted mean time lost 1.01, 0.99–1.03); trajectories before vaccination; narrower windows for covariates with adjustment for encounters; an E-value; competing-risk models; a formal statement of the instrumental-variable assumptions."),
            ("for dementia", "These checks have not, as far as the reading for this map found, been published for dementia. The paper itself notes: 'Early divergence of Kaplan−Meier curves, while biologically plausible, can be a sign of unmeasured confounding'."),
        ],
        read: Read::Part,
        cite: "Nat Med 32:3612–3616 (2026)",
        doi: "10.1038/s41591-026-04606-0",
        disclosure: "",
        held: &[],
        open: &[],
        pic: None,
    },
];

/// Where a study is in `STUDIES`, by its key (for the experiments and the tests).
fn study(key: &str) -> Option<usize> {
    STUDIES.iter().position(|s| s.key == key)
}

// ----- the numbers ------------------------------------------------------------------------

/// A source's numbers, as the dossier gives them, typed a second time: a number shown
/// in a study's texts that is not here, or a citation that differs, fails the tests,
/// so that a slip in either cannot put a wrong figure on the screen.
#[cfg(test)]
struct Fact {
    key: &'static str,
    cite: &'static str,
    doi: &'static str,
    numbers: &'static [&'static str],
}

/// The dossier is DOSSIER.md of 25 November 2026; "§2.1 #n" is row n of its table of
/// studies. Pages are the papers' own where the dossier gives them (only for Taquet 2024).
#[cfg(test)]
const FACTS: &[Fact] = &[
    // §2.1 #1; §2.2 (Eyting); §4.4; Kivimäki's figure from §2.1 #30. Read in part.
    Fact {
        key: "eyting",
        cite: "Nature 641:438–446 (2025); preprint medRxiv 2023",
        doi: "10.1038/s41586-025-08800-x",
        numbers: &[
            "2025", "80", "282,541", "2", "1933", "1", "2013", "79", "47.2", "0.01", "7", "1.3", "0.2", "2.7", "8.5", "3.5", "0.6", "7.1", "20.0", "6.5",
            "33.4", "3.1", "5.8", "0.4", "0.25", "0", "12", "18", "35", "641", "438", "446", "2023",
        ],
    },
    // §2.1 #2; §2.2. Abstract only.
    Fact {
        key: "michalik",
        cite: "medRxiv 2023",
        doi: "10.1101/2023.09.08.23295225",
        numbers: &["2023", "5,077,426", "2", "1933", "9", "0.38", "0.08", "0.68", "4.8", "0.62", "1.06", "0.19", "0.11", "0.51", "0.28"],
    },
    // §2.1 #3; §2.2; Kivimäki's figures from §2.1 #30. Read in part.
    Fact {
        key: "xie",
        cite: "Cell 188:7049–7064 (2025)",
        doi: "10.1016/j.cell.2025.11.007",
        numbers: &[
            "2025", "2", "1933", "9", "14,350", "1.5", "0.5", "2.9", "3.1", "1.0", "6.2", "8.5", "0.6", "18.5", "29.5", "62.9", "22.7", "6.5", "42.8", "0.029",
            "0.039", "3", "6", "41", "56", "188", "7049", "7064",
        ],
    },
    // §2.1 #4; §2.2 (Pomirchy); Kivimäki's figure from §2.1 #30. Read in part.
    Fact {
        key: "pomirchy25",
        cite: "JAMA 333:2083–2092 (2025)",
        doi: "10.1001/jama.2025.5013",
        numbers: &[
            "2025", "65", "101,219", "2", "1936", "70", "79", "1", "2016", "16.4", "13.2", "19.5", "7.4", "1.8", "0.4", "3.3", "3", "15", "9", "0", "156",
            "1.4", "8.4", "177", "28", "301", "333", "2083", "2092",
        ],
    },
    // §2.1 #5; §2.2; §5 item 7; the letters from §2.1, "Other items". Read in part.
    Fact {
        key: "pomirchy26",
        cite: "Lancet Neurol 25:170–180 (2026)",
        doi: "10.1016/S1474-4422(25)00455-7",
        numbers: &[
            "2026", "232,124", "1", "1946", "1945", "27.4", "0.0", "54.7", "13.9", "5.8", "33.6", "5.5", "2.0", "0.4", "3.5", "0.2", "3.8", "0.93", "3.12",
            "1.26", "0.9", "1.71", "3.52", "15", "70", "84", "18", "219", "111", "303", "25", "170", "180",
        ],
    },
    // §2.1 #6; §2.2 (Hamilton). Not peer reviewed. Read in part.
    Fact {
        key: "hamilton",
        cite: "medRxiv July 2026 (not peer reviewed)",
        doi: "10.64898/2026.07.20.26358345",
        numbers: &[
            "2026",
            "6,340,214",
            "1933",
            "8",
            "0.12",
            "0.153",
            "0.079",
            "0.06",
            "0.40",
            "0.27",
            "3",
            "2",
            "13.6",
            "1.30",
            "2.71",
            "0.19",
            "0.87",
            "2.13",
            "0.26",
            "18",
            "32",
        ],
    },
    // §1 (Taquet 2024): p. 2779 (shares 95% and 98%; median follow-up), Methods PDF p. 6
    // (cohort dates, age, 3 months to 6 years, 60 covariates, 1:1), Table 1 p. 2779
    // (0.83, 0.79–0.87, 164, 124–205; women 0.78, men 0.87; 0.65; 0.98; 0.97; 0.83
    // 0.76–0.92; CEM 0.82 0.77–0.87; excluding both 0.79), p. 2779 (pre-pandemic HR
    // 0.74, 0.62–0.90), p. 2780 (the quoted sentence), Suppl. Table 2 (the 1.8 times:
    // the dossier's calculation, §1.5 and T9), §1.14 (2026: 1.01, 0.99–1.03).
    Fact {
        key: "taquet24",
        cite: "Nat Med 30:2777–2781 (2024)",
        doi: "10.1038/s41591-024-03201-5",
        numbers: &[
            "2024", "62", "103,837", "65", "2017", "2020", "95", "2014", "98", "60", "3", "6", "0.83", "0.79", "0.87", "17", "164", "124", "205", "0.78",
            "0.65", "0.61", "0.69", "0.98", "0.95", "1.01", "0.97", "0.91", "1.03", "0.76", "0.92", "0.74", "0.62", "0.90", "0.82", "0.77", "1.8", "1", "2",
            "203316", "30", "2777", "2781", "0.93", "0.96",
        ],
    },
    // §2.1 #8; §2.2 (the AS01 paper and the exchange it prompted). Read in part.
    Fact {
        key: "taquet25",
        cite: "npj Vaccines 10:130 (2025); critique 10:255; reply 10:256",
        doi: "10.1038/s41541-025-01172-3",
        numbers: &[
            "2025", "436,788", "0.71", "0.61", "0.83", "60", "2023", "44", "66", "3", "18", "0.82", "0.74", "0.91", "0.63", "0.55", "0.72", "10", "130", "255",
            "256",
        ],
    },
    // §2.1 #9; §2.2; §3.2 (recurrent shingles 7–9%); disclosures after §2.1. Read in part.
    Fact {
        key: "polisky",
        cite: "Nat Med 31:4172–4179 (2025)",
        doi: "10.1038/s41591-025-03972-5",
        numbers: &["2025", "100", "400", "9", "18", "5", "2", "31", "33", "3", "30", "35", "27", "20", "34", "24", "29", "7", "4172", "4179"],
    },
    // §2.1 #10. Abstract only (E-values in a 2022 conference abstract).
    Fact {
        key: "scherrer",
        cite: "PLoS One 16:e0257405 (2021)",
        doi: "10.1371/journal.pone.0257405",
        numbers: &["2021", "136,016", "172,790", "65", "0.69", "0.67", "0.72", "0.65", "0.57", "0.74", "69", "2022", "4", "16"],
    },
    // §2.1 #11. Abstract only.
    Fact {
        key: "wiemken",
        cite: "J Am Geriatr Soc 70:1157–1168 (2022)",
        doi: "10.1111/jgs.17606",
        numbers: &["2022", "0.50", "0.43", "0.59", "0.58", "0.38", "0.89", "70", "1157", "1168"],
    },
    // §2.1 #12. Abstract only.
    Fact {
        key: "schnier",
        cite: "Alzheimers Dement (TRCI) 8:e12293 (2022)",
        doi: "10.1002/trc2.12293",
        numbers: &["2022", "2013", "2020", "0.72", "0.69", "0.75", "8"],
    },
    // §2.1 #13. Abstract only.
    Fact {
        key: "lophatananon21",
        cite: "BMJ Open 11:e045871 (2021)",
        doi: "10.1136/bmjopen-2020-045871",
        numbers: &["2021", "3", "0.808", "0.657", "0.993", "11"],
    },
    // §2.1 #14. Abstract only.
    Fact {
        key: "lophatananon23",
        cite: "BMC Public Health 23:1903 (2023)",
        doi: "10.1186/s12889-023-16768-4",
        numbers: &["2023", "854,745", "0.78", "0.77", "0.79", "0.96", "0.94", "0.97", "23", "1903"],
    },
    // §2.1 #15. Abstract only.
    Fact {
        key: "harris",
        cite: "J Alzheimers Dis 95:703–718 (2023)",
        doi: "10.3233/JAD-221231",
        numbers: &["2023", "8", "0.75", "0.73", "0.76", "0.70", "95", "703", "718"],
    },
    // §2.1 #16. Abstract only.
    Fact {
        key: "ukraintseva",
        cite: "Exp Gerontol 190:112411 (2024)",
        doi: "10.1016/j.exger.2024.112411",
        numbers: &["2024", "65", "75", "15", "21", "190", "112411"],
    },
    // §2.1 #17. Read in part.
    Fact {
        key: "tang",
        cite: "Vaccine 46:126673 (2025)",
        doi: "10.1016/j.vaccine.2024.126673",
        numbers: &["2025", "4.5", "50", "1", "2018", "0.68", "0.67", "0.70", "0.89", "0.87", "0.92", "46", "126673"],
    },
    // §2.1 #18; §2.2; §4.1 (E-value). Read in part.
    Fact {
        key: "rayens",
        cite: "Nat Commun 17:2056 (2026)",
        doi: "10.1038/s41467-026-69289-0",
        numbers: &[
            "2026", "65,800", "65", "263,200", "0.49", "0.46", "0.51", "0.73", "0.67", "0.79", "0.94", "0.89", "1.00", "3.50", "94", "100", "17", "2056",
        ],
    },
    // §2.1 #19. Abstract only.
    Fact {
        key: "dosreis",
        cite: "Alzheimers Dement 22:e71407 (2026)",
        doi: "10.1002/alz.71407",
        numbers: &["2026", "0.67", "0.65", "0.68", "3", "0.74", "0.69", "0.79", "22"],
    },
    // §2.1 #20; §2.2 (87% after discharge). Abstract only.
    Fact {
        key: "hayes",
        cite: "Ann Intern Med 179:958–967 (2026)",
        doi: "10.7326/ANNALS-25-04689",
        numbers: &["2026", "509,926", "12", "1.73", "87", "4", "18.8", "24.6", "5.8", "3.9", "7.5", "0.76", "0.69", "0.84", "179", "958", "967"],
    },
    // §2.1 #21. Abstract only.
    Fact {
        key: "oh",
        cite: "Alzheimers Dement 22:e71410 (2026)",
        doi: "10.1002/alz.71410",
        numbers: &["2026", "50", "0.88", "0.87", "0.90", "0.75", "0.71", "0.78", "15.5", "11.3", "22"],
    },
    // §2.1 #22. Abstract only.
    Fact {
        key: "rose",
        cite: "medRxiv 2026",
        doi: "10.64898/2026.01.21.26344555",
        numbers: &["2026", "1.48", "19", "5", "0.63", "0.60", "0.65", "0.77", "0.73", "0.81"],
    },
    // §2.1 #23; §4.7. Abstract only.
    Fact {
        key: "shah",
        cite: "Brain Behav 14:e3415 (2024)",
        doi: "10.1002/brb3.3415",
        numbers: &["2024", "0.84", "0.50", "1.43", "0.53", "2", "100", "1", "14"],
    },
    // §2.1 #24; §4.7. Abstract only.
    Fact {
        key: "yin",
        cite: "J Alzheimers Dis 106:1232–1241 (2025)",
        doi: "10.1177/13872877251351593",
        numbers: &["2025", "5", "0.71", "0.66", "0.76", "2", "97.15", "106", "1232", "1241"],
    },
    // §2.1 #25. Abstract only.
    Fact {
        key: "marra",
        cite: "Hum Vaccin Immunother 21:2546741 (2025)",
        doi: "10.1080/21645515.2025.2546741",
        numbers: &["2025", "18", "0.68", "0.56", "0.83", "2", "99", "1.14", "1.04", "1.25", "21", "2546741"],
    },
    // §2.1 #26. Abstract only.
    Fact {
        key: "zhang",
        cite: "Vaccines 14:685 (2026), PROSPERO CRD420261326042",
        doi: "10.3390/vaccines14080685",
        numbers: &["2026", "13", "0.72", "0.65", "0.81", "2", "97.1", "0.83", "0.82", "0.69", "0.98", "14", "685"],
    },
    // §2.1 #27. Abstract only.
    Fact { key: "heen", cite: "Rev Med Virol 36:e70203 (2026)", doi: "10.1002/rmv.70203", numbers: &["2026", "36"] },
    // §2.1 #28. Read in part.
    Fact { key: "hiqa", cite: "HIQA (Ireland), scoping review, 20 May 2026", doi: NO_DOI, numbers: &["2026", "20"] },
    // §0 (the registry record, read 25 November 2026).
    Fact {
        key: "danzoster",
        cite: "ClinicalTrials.gov NCT07485283; EU CT 2025-524944-37-00",
        doi: NO_DOI,
        numbers: &["168,460", "2", "6", "65", "1", "3", "28", "2026", "2029", "14"],
    },
    // §0 (the registry record, read 25 November 2026).
    Fact {
        key: "finland",
        cite: "ClinicalTrials.gov NCT07502560; EU CT 2025-524598-17-00",
        doi: NO_DOI,
        numbers: &["300889", "33,609", "3", "1", "76", "10", "31", "2026", "2030", "2037", "12"],
    },
    // §2.1 #30; §2.2 (Kivimäki); the arithmetic 1.8 / 0.164 is calculated here from
    // §2.1 #4 and tested. Abstract only.
    Fact {
        key: "kivimaki",
        cite: "Brain Behav Immun 139:107021 (2026)",
        doi: "10.1016/j.bbi.2026.107021",
        numbers: &["2026", "100", "18", "0", "35", "177", "28", "301", "84", "219", "111", "15", "303", "41", "56", "1.8", "0.164", "11.0", "139"],
    },
    // §2.1, "Other items found but not tabulated"; disclosures there. Title only.
    Fact { key: "hamiltondaveysmith", cite: "Brain Behav Immun 140:107072 (2026)", doi: "10.1016/j.bbi.2026.107072", numbers: &["2026", "140"] },
    // §0. Title and disclosure only.
    Fact { key: "geldsetzer", cite: "Nat Med 32:2720–2723 (2026)", doi: "10.1038/s41591-026-04474-8", numbers: &["2026", "32"] },
    // §1.14 (read in part).
    Fact {
        key: "corsizuelli",
        cite: "Nat Med 32:3612–3616 (2026)",
        doi: "10.1038/s41591-026-04606-0",
        numbers: &["2017", "2026", "60", "2018", "1.01", "0.99", "1.03"],
    },
    // The numbers the designs, explanations, arrows and the opening card use that are
    // not one study's own: §0 (trials), §1.4 (dates of the switch), §3.2 (uptake
    // jumps; trial efficacy against shingles from MMWR 2018; Hazan 5.4%; ICD-10-CM
    // on 1 October 2015; Marra 1.14), §3.3 (I² 97–100%), T19 (0.43, 0.45: the
    // dossier's calculation from the posted data), §1.9 (5.75%, 5.94%), and the
    // date the sources were read.
    Fact {
        key: "shared",
        cite: "",
        doi: NO_DOI,
        numbers: &[
            "25", "2026", "2029", "2030", "2017", "47.2", "16.4", "27.4", "98", "95", "97.4", "60", "69", "91.3", "70", "64", "38", "5.4", "1", "2015", "1.14",
            "1.04", "1.25", "97", "100", "0.43", "0.45", "2", "5.75", "5.94", "6", "1.8", "3", "7", "8", "79", "80", "18", "0.49", "0.73", "0.94", "3.50",
            "0.83", "0.76", "0.92", "0.74", "0.62", "0.90", "0.65", "0.61", "0.69", "0.98", "0.95", "1.01", "0.97", "0.91", "1.03", "0.82", "0.77", "0.87",
            "0.99", "0.89", "1.00", "1.73", "168,460", "28", "33,609", "31", "76", "10", "44", "0.79", "4", "6.3", "2014", "2020", "65", "12", "1933", "2013",
            "0.01", "3.5", "0.6", "7.1", "0.12", "0.06", "0.40", "0.27", "177", "111", "2023", "2024", "2025", "2021", "2022", "2006", "5", "0.93", "0.96",
        ],
    },
];

#[cfg(test)]
fn fact(key: &str) -> Option<&'static Fact> {
    FACTS.iter().find(|f| f.key == key)
}

// ----- the experiments --------------------------------------------------------------------

struct Trial {
    button: &'static str,
    caption: &'static str,
    design: Option<usize>,
    study: Option<&'static str>,
    story: Option<usize>,
}

const TRIALS: [Trial; 5] = [
    Trial {
        button: "the healthy vaccinee",
        caption: r"A cohort of vaccinated against unvaccinated people (Rayens 2026). Faint pink: the back door $V \gets H \to D$, which adjustment closes only as far as the records show health. In the same data the paper reports a hazard ratio of 0.49 against the unvaccinated and 0.73 against Tdap recipients: the association moves towards 1 when the comparison group also chose a vaccine.",
        design: Some(COHORT),
        study: Some("rayens"),
        story: Some(2),
    },
    Trial {
        button: "a birthday decides",
        caption: r"Wales offered the live vaccine from 1 September 2013 to people aged 79; people already 80 were never eligible. Uptake was 0.01% among people born in the week before 2 September 1933 and 47.2% in the week after. Health still decides who of the eligible took it, but not who was born on which side: the healthy-vaccinee arrows now lie on closed, grey paths. The design assumes that nothing else changes at that date (dashed). The paper reports 3.5 percentage points fewer new diagnoses over 7 years among those vaccinated (0.6–7.1).",
        design: Some(BIRTH),
        study: Some("eyting"),
        story: Some(2),
    },
    Trial {
        button: "same design, another answer",
        caption: "The same cut-off in England's hospital records, 6.3 million people (Hamilton 2026, a preprint). The design finds the known effect on shingles (0.12 percentage points fewer) and none on hospital-coded dementia: −0.06 points (−0.40 to 0.27). Wales, Australia and Ontario found fewer diagnoses. Kivimäki and others divided each effect of eligibility by its jump in uptake: an implied reduction among the vaccinated of 18% in Wales, but 177% in Australia and 111% in one Canadian analysis, where more than 100% is impossible if the division's assumptions hold.",
        design: Some(BIRTH),
        study: Some("hamilton"),
        story: Some(7),
    },
    Trial {
        button: "vaccine against vaccine",
        caption: r"Taquet and others (2024) compared people first vaccinated in the three years after the US switch of October 2017 (95% recombinant) with those in the three years before (98% live). Everyone compared was vaccinated, which the authors chose to avoid the healthy-vaccinee bias of comparisons with the unvaccinated; the paper reports a ratio of restricted mean time lost of 0.83 (0.79–0.87). But the period changes more than the vaccine: the two groups are followed through different calendar years ($P \to F \to K$), with the pandemic early in one group's follow-up and late in the other's. The next scene takes this study apart.",
        design: Some(SWITCH),
        study: Some("taquet24"),
        story: Some(6),
    },
    Trial {
        button: "two ways it could be true",
        caption: r"If vaccination protects, it may be through fewer reactivations of the virus ($T \to R \to D$) or by stimulating the immune system in general, as the adjuvant might ($T \to I \to D$, lit). The first predicts protection in step with protection against shingles; the second, protection from other adjuvanted vaccines too. The Finnish trial is randomised, blinded and placebo-controlled, which cuts every arrow into $V$ and keeps contact alike; but, like the Danish trial, it gives both routes at once, and it compares the vaccine with placebo, not with the live vaccine.",
        design: Some(BLIND_TRIAL),
        study: Some("finland"),
        story: Some(1),
    },
];

// ----- what the scene shows ---------------------------------------------------------------

/// What the visitor has chosen. A study brings its design's graph with it.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug, Hash)]
struct Choice {
    design: Option<usize>,
    study: Option<usize>,
    story: Option<usize>,
}

impl Choice {
    /// The design whose graph is drawn.
    fn graph(&self) -> Option<usize> {
        match self.study {
            Some(s) => STUDIES.get(s).and_then(|s| s.design),
            None => self.design,
        }
    }
}

/// The card: the text that goes with the choice.
#[derive(Clone, Debug, PartialEq)]
struct Card {
    title: String,
    meta: String,
    pic: Pic,
    blocks: Vec<(String, String)>,
    foot: String,
}

const OPENING: [(&str, &str); 5] = [
    (
        "no trial has reported",
        "No randomised trial has reported an effect of any shingles vaccine on dementia. Two trials of the recombinant vaccine began in 2026, in Denmark (open label, against no vaccine) and Finland (blinded, against placebo); their registries estimate primary completion in June 2029 and April 2030. Until then every estimate rests on an assumption that the data cannot fully test.",
    ),
    (
        "what is measured",
        r"Every study counts a recorded diagnosis, $D^*$, not the disease, $D$ (dashed: never observed). A difference in diagnoses is a difference in disease only if diagnosis is as complete, and as early, in the groups compared.",
    ),
    (
        "what would settle it",
        "Randomisation removes the healthy-vaccinee bias in who is vaccinated; blinding with placebo also keeps diagnosis alike in both arms. Before the trials report: for the 2017 switch, a placebo switch and the other checks not yet run; for the birth-date studies, an account of why they disagree (how dementia is recorded, week or month of birth, under-recorded uptake, chance).",
    ),
    ("what the trials will not settle", "Neither trial compares the recombinant vaccine with the live one, and neither separates the antigen (a protein of the virus) from the adjuvant. No trial of the live vaccine is registered, as far as a search of ClinicalTrials.gov and ISRCTN made for this map found."),
    ("how to read this map", "Choose a design (a row of the table), an explanation (a column), both (a cell), or a study (the list). The graph shows what the choice closes and leaves open; point at an arrow or a variable to see why it is there."),
];

fn card(choice: Choice) -> Card {
    let story = choice.story.and_then(|s| STORIES.get(s));
    if let Some(k) = choice.study {
        let Some(s) = STUDIES.get(k) else { return card(Choice::default()) };
        let mut blocks: Vec<(String, String)> = s.blocks.iter().map(|(h, b)| (h.to_string(), b.to_string())).collect();
        if let (Some(d), Some(j)) = (s.design, choice.story) {
            let c = &CELLS[d][j];
            blocks.push((format!("explanation {}: {}", STORIES[j].label, c.mark.says()), c.text.to_string()));
        } else if let Some(st) = story {
            blocks.push((format!("explanation {}", st.label), st.gist.to_string()));
        }
        let design = s.design.map_or("no design of its own".to_string(), |d| DESIGNS[d].name.to_string());
        let source = if s.doi.is_empty() { s.cite.to_string() } else { format!("{}. doi:{}", s.cite, s.doi) };
        return Card {
            title: s.by.to_string(),
            meta: format!("{} · {} · {}", s.meta, design, s.read.says()),
            pic: s.pic.unwrap_or(s.design.map_or(Pic::None, |d| DESIGNS[d].pic)),
            blocks,
            foot: source,
        };
    }
    match (choice.design, story) {
        (Some(d), Some(st)) => {
            let (des, j) = (&DESIGNS[d], choice.story.unwrap_or(0));
            let c = &CELLS[d][j];
            Card {
                title: format!("{} · explanation {}", des.title, st.label),
                meta: format!("{}: {}", st.title, c.mark.says()),
                pic: des.pic,
                blocks: vec![
                    ("verdict".to_string(), c.text.to_string()),
                    ("the explanation".to_string(), st.gist.to_string()),
                    ("it predicts".to_string(), st.predicts.to_string()),
                    ("the design assumes".to_string(), des.assumption.to_string()),
                ],
                foot: String::new(),
            }
        }
        (Some(d), None) => {
            let des = &DESIGNS[d];
            let mut blocks = vec![
                ("who against whom".to_string(), des.who.to_string()),
                ("what it assumes".to_string(), des.assumption.to_string()),
                ("in the graph".to_string(), des.graph.to_string()),
                ("checks that support it".to_string(), des.checks.to_string()),
                ("what it cannot rule out".to_string(), des.cannot.to_string()),
            ];
            if !des.seen.is_empty() {
                blocks.push(("in these studies".to_string(), des.seen.to_string()));
            }
            let mine: Vec<&str> = STUDIES.iter().filter(|s| s.design == Some(d) && s.group != 6).map(|s| s.short).collect();
            Card {
                title: des.title.to_string(),
                meta: format!("{} on the map: {}", if mine.len() == 1 { "1 study" } else { "studies" }, mine.join(", ")),
                pic: des.pic,
                blocks,
                foot: String::new(),
            }
        }
        (None, Some(st)) => {
            let j = choice.story.unwrap_or(0);
            let list = |m: &[Mark]| -> String {
                let names: Vec<&str> = DESIGNS.iter().enumerate().filter(|(d, _)| m.contains(&CELLS[*d][j].mark)).map(|(_, d)| d.name).collect();
                if names.is_empty() {
                    "none".to_string()
                } else {
                    names.join("; ")
                }
            };
            Card {
                title: st.title.to_string(),
                meta: if st.real { "green: a real effect on dementia".to_string() } else { "pink: a difference in diagnoses without one".to_string() },
                pic: Pic::None,
                blocks: vec![
                    ("the arrows".to_string(), st.gist.to_string()),
                    ("it predicts".to_string(), st.predicts.to_string()),
                    ("where the evidence sits".to_string(), st.evidence.to_string()),
                    ("designs that can tell it apart, at least in part".to_string(), list(&[Mark::Can, Mark::Partly])),
                    ("designs that cannot".to_string(), list(&[Mark::Cannot])),
                ],
                foot: String::new(),
            }
        }
        (None, None) => Card {
            title: "Why the question is hard".to_string(),
            meta: "A map of the studies, not a verdict on the vaccine, and not medical advice.".to_string(),
            pic: Pic::None,
            blocks: OPENING.iter().map(|(h, b)| (h.to_string(), b.to_string())).collect(),
            foot: String::new(),
        },
    }
}

/// What a narrow canvas says under the graph: the choice in a sentence or two.
fn gist(choice: Choice) -> (String, String) {
    if let Some(s) = choice.study.and_then(|k| STUDIES.get(k)) {
        let design = s.design.map_or("", |d| DESIGNS[d].name);
        return (format!("{} · {}", s.short, design), format!("{}: {}", s.place, s.estimate));
    }
    match (choice.design, choice.story) {
        (Some(d), Some(j)) => (format!("{} · explanation {}", DESIGNS[d].name, STORIES[j].label), CELLS[d][j].text.to_string()),
        (Some(d), None) => (DESIGNS[d].title.to_string(), DESIGNS[d].gist.to_string()),
        (None, Some(j)) => (STORIES[j].title.to_string(), STORIES[j].gist.to_string()),
        (None, None) => (
            "No randomised trial has reported.".to_string(),
            "Every study rests on an assumption it cannot fully test. Choose a design, a study or an explanation.".to_string(),
        ),
    }
}

// ----- how the graph is drawn -------------------------------------------------------------

/// What an arrow carries in the design chosen.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Carries {
    /// No design chosen.
    Idle,
    /// On the path the design estimates: the effect on dementia itself.
    Effect,
    /// On a path the design leaves open, which would give a difference without one.
    Open,
    /// On such a path, closed only as far as the records go.
    Weak,
    /// An arrow the design must assume absent.
    Assumed,
    /// Removed by the design.
    Cut,
    /// On no path the design leaves open.
    Closed,
}

/// The colour of an arrow on no path any design leaves open, and of a closed path.
const CLOSED: Color = Color::hex(0x5A6478);
const IDLE: Color = Color::hex(0x7C88A3);

/// How an arrow is drawn: what it carries in the design chosen, whether the
/// explanation chosen uses it, and whether it is drawn at all.
fn carries(choice: Choice, k: usize) -> Carries {
    let Some(a) = ARROWS.get(k) else { return Carries::Idle };
    let Some(d) = choice.graph().and_then(|d| DESIGNS.get(d)) else {
        return Carries::Idle;
    };
    let has = |list: Pairs| list.contains(&(a.from, a.to));
    let extra: Pairs = choice.study.and_then(|s| STUDIES.get(s)).map_or(&[], |s| s.open);
    if has(d.cut) {
        Carries::Cut
    } else if has(d.assumes) {
        Carries::Assumed
    } else if has(d.effect) {
        Carries::Effect
    } else if has(d.open) || has(extra) {
        Carries::Open
    } else if has(d.weak) {
        Carries::Weak
    } else {
        Carries::Closed
    }
}

/// Which variables are drawn: the core, the layers turned on, and any layer the choice
/// cannot be drawn without.
fn shown_layers(choice: Choice, layers: [bool; 2]) -> [bool; 2] {
    let mut on = layers;
    let mut need = |l: Option<Layer>| match l {
        Some(Layer::Who) => on[0] = true,
        Some(Layer::Record) => on[1] = true,
        _ => {}
    };
    need(choice.graph().and_then(|d| DESIGNS.get(d)).and_then(|d| d.needs));
    need(choice.story.and_then(|s| STORIES.get(s)).and_then(|s| s.needs));
    on
}

fn layer_on(layer: Layer, on: [bool; 2]) -> bool {
    match layer {
        Layer::Core => true,
        Layer::Who => on[0],
        Layer::Record => on[1],
    }
}

/// Is the arrow drawn at all: both its ends shown, and, if it is one a design assumes
/// away, that design chosen.
fn arrow_shown(choice: Choice, on: [bool; 2], k: usize) -> bool {
    let Some(a) = ARROWS.get(k) else { return false };
    let ends = layer_on(VARS[a.from].layer, on) && layer_on(VARS[a.to].layer, on);
    ends && (!a.assumed || carries(choice, k) == Carries::Assumed)
}

/// What each arrow is drawn as carrying with these layers shown. An arrow the design
/// leaves open is drawn so only where the open path through it can be seen whole: from
/// what is compared, through arrows drawn, to the recorded diagnosis. (With the
/// recording layer hidden, a path through recorded shingles has nothing to show for
/// it but its last arrow, which would be pink with nothing leading to it.)
fn drawn_carries(choice: Choice, on: [bool; 2]) -> Vec<Carries> {
    let mut c: Vec<Carries> = (0..ARROWS.len()).map(|k| if arrow_shown(choice, on, k) { carries(choice, k) } else { Carries::Closed }).collect();
    let Some(d) = choice.graph().and_then(|d| DESIGNS.get(d)) else { return c };
    let passes = |c: Carries| matches!(c, Carries::Effect | Carries::Open | Carries::Weak | Carries::Assumed);
    // the variables reached along such arrows, either way, from `from`, not through `avoid`
    let reach = |c: &[Carries], from: &[usize], avoid: &[usize]| -> Vec<bool> {
        let mut seen = vec![false; VARS.len()];
        let mut todo: Vec<usize> = from.to_vec();
        for &i in from {
            seen[i] = true;
        }
        while let Some(i) = todo.pop() {
            for (k, a) in ARROWS.iter().enumerate() {
                if !passes(c[k]) {
                    continue;
                }
                let next = if a.from == i {
                    a.to
                } else if a.to == i {
                    a.from
                } else {
                    continue;
                };
                if !seen[next] && !avoid.contains(&next) {
                    seen[next] = true;
                    todo.push(next);
                }
            }
        }
        seen
    };
    let from_compared = reach(&c, d.compared, &[v::DX]);
    let to_diagnosis = reach(&c, &[v::DX], d.compared);
    for (k, a) in ARROWS.iter().enumerate() {
        if matches!(c[k], Carries::Open | Carries::Weak | Carries::Assumed) {
            let live = (from_compared[a.from] && to_diagnosis[a.to]) || (from_compared[a.to] && to_diagnosis[a.from]);
            if !live {
                c[k] = Carries::Closed;
            }
        }
    }
    c
}

/// The colour and width of an arrow that carries `c`.
fn look_of(c: Carries) -> (Color, f32) {
    match c {
        Carries::Idle => (IDLE, 1.8),
        Carries::Effect => (TRUTH, 2.4),
        Carries::Open => (NAIVE, 2.4),
        Carries::Weak => (NAIVE.a(0.42), 1.7),
        Carries::Assumed => (NAIVE, 1.8),
        Carries::Cut => (CLOSED, 1.8),
        Carries::Closed => (CLOSED, 1.6),
    }
}

/// How open a kind of arrow is, for choosing the most open of several.
fn openness(c: Carries) -> u8 {
    match c {
        Carries::Idle => 0,
        Carries::Cut => 1,
        Carries::Closed => 2,
        Carries::Assumed => 3,
        Carries::Weak => 4,
        Carries::Open => 5,
        Carries::Effect => 6,
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct ArrowLook {
    color: Color,
    width: f32,
    /// How far it is there (0: not drawn).
    alpha: f32,
    carries: Carries,
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct VarLook {
    alpha: f32,
    /// How held it is drawn (a box), and whether only as far as the records go.
    held: f32,
    partly: bool,
    compared: f32,
}

#[derive(Clone, PartialEq, Debug)]
struct Look {
    arrows: Vec<ArrowLook>,
    vars: Vec<VarLook>,
}

impl Look {
    fn of(choice: Choice, layers: [bool; 2]) -> Look {
        let on = shown_layers(choice, layers);
        let design = choice.graph().and_then(|d| DESIGNS.get(d));
        let study = choice.study.and_then(|s| STUDIES.get(s));
        let story = choice.story.and_then(|s| STORIES.get(s));
        let drawn = drawn_carries(choice, on);
        let gate = story.filter(|_| design.is_some()).map_or(Carries::Idle, |s| {
            s.gates
                .iter()
                .filter_map(|&(a, b)| ARROWS.iter().position(|x| x.from == a && x.to == b))
                .map(|k| drawn[k])
                .max_by_key(|c| openness(*c))
                .unwrap_or(Carries::Closed)
        });
        let arrows = (0..ARROWS.len())
            .map(|k| {
                let a = &ARROWS[k];
                let c = drawn[k];
                let lit = story.map(|s| s.arrows.contains(&(a.from, a.to)));
                let (mut color, mut width) = look_of(c);
                // an explanation chosen alone draws its own arrows in its colour; with a
                // design, in the state the design leaves the way it enters by
                match (story, c, lit) {
                    (Some(s), Carries::Idle, Some(true)) => color = if s.real { TRUTH } else { NAIVE },
                    (Some(_), _, Some(true)) if gate != Carries::Idle => color = look_of(gate).0,
                    _ => {}
                }
                let mut alpha = if arrow_shown(choice, on, k) { 1.0 } else { 0.0 };
                match lit {
                    Some(true) => width += if c == Carries::Weak { 0.8 } else { 1.4 },
                    Some(false) => alpha *= 0.3,
                    None => {}
                }
                ArrowLook { color, width, alpha, carries: c }
            })
            .collect();
        let held = |i: usize| design.is_some_and(|d| d.held.contains(&i)) || study.is_some_and(|s| s.held.contains(&i));
        let vars = (0..VARS.len())
            .map(|i| VarLook {
                alpha: if layer_on(VARS[i].layer, on) { 1.0 } else { 0.0 },
                held: if held(i) || design.is_some_and(|d| d.partly.contains(&i)) { 1.0 } else { 0.0 },
                partly: !held(i) && design.is_some_and(|d| d.partly.contains(&i)),
                compared: if design.is_some_and(|d| d.compared.contains(&i)) { 1.0 } else { 0.0 },
            })
            .collect();
        Look { arrows, vars }
    }

    /// Part of the way from `self` to `to`: colours, widths and alphas glide, the kind
    /// of line is the new one at once.
    fn towards(&self, to: &Look, t: f32) -> Look {
        let t = t.clamp(0.0, 1.0);
        let lerp = |a: f32, b: f32| a + (b - a) * t;
        Look {
            arrows: self
                .arrows
                .iter()
                .zip(&to.arrows)
                .map(|(a, b)| ArrowLook { color: a.color.mix(b.color, t), width: lerp(a.width, b.width), alpha: lerp(a.alpha, b.alpha), carries: b.carries })
                .collect(),
            vars: self
                .vars
                .iter()
                .zip(&to.vars)
                .map(|(a, b)| VarLook { alpha: lerp(a.alpha, b.alpha), held: lerp(a.held, b.held), partly: b.partly, compared: lerp(a.compared, b.compared) })
                .collect(),
        }
    }
}

/// How long a change of choice takes to glide into place: the time in which it covers
/// about two thirds of the way (seconds).
const GLIDE: f32 = 0.11;

// ----- where things go --------------------------------------------------------------------

type Rect = (Vec2, Vec2);

fn inside(p: Vec2, (lo, hi): Rect) -> bool {
    p.cmpge(lo).all() && p.cmple(hi).all()
}

/// The views of a narrow canvas, one at a time.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
enum Pane {
    Graph,
    Details,
    Table,
    Studies,
}

const PANES: [(Pane, &str); 4] = [(Pane::Graph, "graph"), (Pane::Details, "details"), (Pane::Table, "table"), (Pane::Studies, "studies")];

/// What the lower part of a wide canvas shows.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
enum Chooser {
    Table,
    Studies,
}

/// Where the parts of the picture go (logical pixels from the canvas's top-left corner).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Lay {
    wide: bool,
    /// Text sizes.
    text: f32,
    small: f32,
    /// The tabs of a narrow canvas, or of the chooser on a wide one.
    tabs: Option<Rect>,
    graph: Option<Rect>,
    /// Under the graph: the layer switches, and the line that says what the pointer is on.
    switches: Option<Rect>,
    status: Option<Rect>,
    /// The few lines that say what is chosen, under the graph of a narrow canvas.
    gist: Option<Rect>,
    card: Option<Rect>,
    table: Option<Rect>,
    list: Option<Rect>,
    /// Where the line about the sources goes.
    footer: Option<Vec2>,
    radius: f32,
}

/// A canvas laid out with everything at once, not one view at a time.
fn is_wide(view: Vec2) -> bool {
    view.x >= 900.0 && view.y >= 640.0
}

/// A canvas too small for anything but the graph.
fn is_tiny(view: Vec2) -> bool {
    view.x < 300.0 || view.y < 300.0
}

impl Lay {
    fn of(view: Vec2, inset: f32, pane: Pane, chooser: Chooser) -> Lay {
        let view = view.max(Vec2::ONE);
        let side = 16.0;
        let top = inset.max(0.0) + 10.0;
        if is_tiny(view) {
            let graph = (Vec2::new(6.0, top), Vec2::new(view.x - 6.0, (view.y - 6.0).max(top + 10.0)));
            return Lay {
                wide: false,
                text: 11.5,
                small: 10.0,
                tabs: None,
                graph: Some(graph),
                switches: None,
                status: None,
                gist: None,
                card: None,
                table: None,
                list: None,
                footer: None,
                radius: 9.0,
            };
        }
        if is_wide(view) {
            let foot = view.y - 12.0;
            let gw = ((view.x - 3.0 * side) * 0.56).round();
            let gh = (view.y * 0.5).clamp(300.0, 560.0).round();
            let graph = (Vec2::new(side, top), Vec2::new(side + gw, top + gh));
            let switches = (Vec2::new(side, graph.1.y + 6.0), Vec2::new(side + gw, graph.1.y + 50.0));
            let status = (Vec2::new(side, switches.1.y + 6.0), Vec2::new(side + gw, switches.1.y + 46.0));
            let card = (Vec2::new(side * 2.0 + gw + 6.0, top), Vec2::new(view.x - side, status.1.y));
            let tabs = (Vec2::new(side, status.1.y + 14.0), Vec2::new(view.x - side, status.1.y + 40.0));
            let below = (Vec2::new(side, tabs.1.y + 10.0), Vec2::new(view.x - side, foot - 18.0));
            let (table, list) = match chooser {
                Chooser::Table => (Some(below), None),
                Chooser::Studies => (None, Some(below)),
            };
            return Lay {
                wide: true,
                text: 13.0,
                small: 11.5,
                tabs: Some(tabs),
                graph: Some(graph),
                switches: Some(switches),
                status: Some(status),
                gist: None,
                card: Some(card),
                table,
                list,
                footer: Some(Vec2::new(side, foot)),
                radius: 17.0,
            };
        }
        // one view at a time, chosen by the tabs at the top
        let tabs = (Vec2::new(10.0, top), Vec2::new(view.x - 10.0, top + 28.0));
        let foot = view.y - 10.0;
        let area = (Vec2::new(12.0, tabs.1.y + 10.0), Vec2::new(view.x - 12.0, foot - 16.0));
        let (text, small) = if view.x < 480.0 { (12.0, 10.5) } else { (12.5, 11.0) };
        let mut lay = Lay {
            wide: false,
            text,
            small,
            tabs: Some(tabs),
            graph: None,
            switches: None,
            status: None,
            gist: None,
            card: None,
            table: None,
            list: None,
            footer: Some(Vec2::new(12.0, foot)),
            radius: if view.x < 420.0 { 13.0 } else { 15.0 },
        };
        match pane {
            Pane::Graph => {
                // the graph first: under it the switches, what the pointer is on and
                // what is chosen, as far as there is room for them
                let room = area.1.y - area.0.y;
                let gh = if room >= 420.0 {
                    (room - 150.0).min(view.x * 1.35)
                } else if room >= 200.0 {
                    room - 66.0
                } else {
                    room
                };
                let graph = (area.0, Vec2::new(area.1.x, area.0.y + gh.max(10.0)));
                lay.graph = Some(graph);
                if room >= 200.0 {
                    let switches = (Vec2::new(area.0.x, graph.1.y + 4.0), Vec2::new(area.1.x, graph.1.y + 64.0));
                    lay.switches = Some(switches);
                    if room >= 420.0 {
                        let status = (Vec2::new(area.0.x, switches.1.y + 4.0), Vec2::new(area.1.x, switches.1.y + 40.0));
                        lay.status = Some(status);
                        lay.gist = (area.1.y - status.1.y >= 40.0).then(|| (Vec2::new(area.0.x, status.1.y + 6.0), area.1));
                    }
                }
            }
            Pane::Details => lay.card = Some(area),
            Pane::Table => lay.table = Some(area),
            Pane::Studies => lay.list = Some(area),
        }
        lay
    }

    /// Every part that is drawn, for the tests.
    #[cfg(test)]
    fn parts(&self) -> Vec<Rect> {
        [self.tabs, self.graph, self.switches, self.status, self.gist, self.card, self.table, self.list].into_iter().flatten().collect()
    }
}

/// Where the variables go in the rectangle `r`: their places in the unit square,
/// stretched to fill it, with room at the edges for discs of `radius` and their names.
fn places((lo, hi): Rect, radius: f32, names: bool) -> Vec<Vec2> {
    let margin = Vec2::new(if names { radius + 8.0 } else { radius + 3.0 }, if names { radius + 6.0 } else { radius + 3.0 });
    let size = (hi - lo - 2.0 * margin).max(Vec2::ONE);
    // (not wider than half as wide again as it is tall, or the graph reads as a strip)
    let w = size.x.min(size.y * 1.7);
    let x0 = lo.x + margin.x + (size.x - w) * 0.5;
    VARS.iter().map(|v| Vec2::new(x0 + v.at.0 * w, lo.y + margin.y + v.at.1 * size.y)).collect()
}

/// The points of an arrow from the rim of one disc to the rim of the other, bowed by
/// `bend` of its length (nothing if the discs touch).
fn arrow_points(a: Vec2, b: Vec2, bend: f32, radius: f32) -> Vec<Vec2> {
    let d = b - a;
    let len = d.length();
    if len.is_nan() || len <= 2.0 * radius + 8.0 {
        return Vec::new();
    }
    let across = Vec2::new(-d.y, d.x) / len;
    let control = (a + b) * 0.5 + across * bend * len;
    let steps = if bend == 0.0 { 1 } else { 28 };
    let at = |t: f32| a * (1.0 - t) * (1.0 - t) + control * 2.0 * t * (1.0 - t) + b * t * t;
    let mut pts: Vec<Vec2> = Vec::with_capacity(steps + 3);
    // (on a curve, the rims are found among the points: none inside either disc)
    if steps == 1 {
        let along = d / len;
        pts.push(a + along * (radius + 3.0));
        pts.push(b - along * (radius + 4.0));
        return pts;
    }
    for i in 0..=steps * 4 {
        let p = at(i as f32 / (steps * 4) as f32);
        if p.distance(a) >= radius + 3.0 && p.distance(b) >= radius + 4.0 && (i % 4 == 0 || pts.is_empty()) {
            pts.push(p);
        }
    }
    // the last point exactly on the rim's edge of room, so that the head sits there
    if let Some(i) = (0..=steps * 4).rev().map(|i| at(i as f32 / (steps * 4) as f32)).position(|p| p.distance(b) >= radius + 4.0) {
        let t = 1.0 - i as f32 / (steps * 4) as f32;
        let end = at(t);
        if pts.last().is_none_or(|l| l.distance(end) > 0.5) {
            pts.push(end);
        }
    }
    if pts.len() < 2 {
        pts.clear();
    }
    pts
}

/// Distance from `p` to a polyline.
fn path_distance(p: Vec2, pts: &[Vec2]) -> f32 {
    pts.windows(2).map(|w| seg_dist(p, w[0], w[1])).fold(f32::MAX, f32::min)
}

/// The middle of a polyline, by length.
fn path_middle(pts: &[Vec2]) -> Vec2 {
    let total: f32 = pts.windows(2).map(|w| w[0].distance(w[1])).sum();
    let mut left = total * 0.5;
    for w in pts.windows(2) {
        let l = w[0].distance(w[1]);
        if l >= left && l > 0.0 {
            return w[0].lerp(w[1], left / l);
        }
        left -= l;
    }
    pts.first().copied().unwrap_or(Vec2::ZERO)
}

/// A dashed polyline in screen space: dashes `on` long every `period` along it.
fn dashed_px(f: &mut Frame, pts: &[Vec2], width: f32, color: Color, period: f32, on: f32) {
    let mut s = 0.0f32;
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let l = a.distance(b);
        if l <= 0.0 {
            continue;
        }
        let mut t = 0.0;
        while t < l {
            let phase = (s + t) % period;
            let (start, stop) = if phase < on { (t, (t + on - phase).min(l)) } else { (t, (t + period - phase).min(l)) };
            if phase < on {
                hud_line(f, a.lerp(b, start / l), a.lerp(b, stop / l), width, color);
            }
            t = stop.max(t + 0.01);
        }
        s += l;
    }
}

/// A ring of dashes in screen space.
fn dashed_ring(f: &mut Frame, centre: Vec2, radius: f32, width: f32, color: Color) {
    let n = 14;
    for k in 0..n {
        let (a, b) = (k as f32 / n as f32 * TAU, (k as f32 + 0.55) / n as f32 * TAU);
        hud_path(f, (0..=4).map(|i| centre + Vec2::from_angle(a + (b - a) * i as f32 / 4.0) * radius), width, color);
    }
}

/// A filled disc in screen space.
fn disc(f: &mut Frame, centre: Vec2, radius: f32, color: Color) {
    let n = 24;
    let c = f.hud_at(1, centre);
    for i in 0..n {
        let (a, b) = (
            f.hud_at(1, centre + Vec2::from_angle(i as f32 / n as f32 * TAU) * radius),
            f.hud_at(1, centre + Vec2::from_angle((i + 1) as f32 / n as f32 * TAU) * radius),
        );
        f.dl.tris_alpha.tri(c, a, b, color);
    }
}

/// A button drawn over the picture, lit when `chosen`: true when it is clicked.
fn hud_button(f: &mut Frame, (min, max): Rect, text: &str, size: f32, chosen: bool, pointer: Option<Vec2>) -> bool {
    let over = pointer.is_some_and(|p| inside(p, (min, max)));
    let fill = match (chosen, over) {
        (true, _) => Color::hex(0x3A4C78),
        (false, true) => Color::hex(0x33405E),
        (false, false) => Color::hex(0x222B40),
    };
    hud_rect(f, min, max, fill);
    f.text_px((min + max) * 0.5, text, size, if chosen { pal::WHITE } else { pal::TEXT }, Align2::CENTER_CENTER);
    over && f.ptr.clicked
}

/// A text broken into lines no wider than `width`, formulas kept whole, measured by
/// `measure` (the frame's own measure on the screen; an estimate in the tests).
fn wrap(text: &str, size: f32, width: f32, measure: &dyn Fn(&str, f32) -> f32) -> Vec<String> {
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
    words.push(&text[start.min(text.len())..]);
    let mut lines: Vec<String> = Vec::new();
    for word in words.into_iter().filter(|w| !w.is_empty()) {
        match lines.last_mut() {
            Some(last) if measure(&format!("{last} {word}"), size) <= width => {
                last.push(' ');
                last.push_str(word);
            }
            _ => lines.push(word.to_string()),
        }
    }
    lines
}

/// One line of a laid-out card.
#[derive(Clone, Debug, PartialEq)]
enum Row {
    Text { text: String, size: f32, color: Color, y: f32 },
    Pic { pic: Pic, y: f32, h: f32 },
}

/// The height a picture takes in a card of this width.
fn pic_height(pic: Pic, width: f32) -> f32 {
    match pic {
        Pic::None => 0.0,
        _ if width < 260.0 => 0.0,
        Pic::Kivimaki | Pic::Rayens | Pic::Hamilton => 112.0,
        _ => 96.0,
    }
}

/// A card laid out in lines `width` wide: the rows, and how tall it is.
fn lay_out(card: &Card, width: f32, text: f32, small: f32, measure: &dyn Fn(&str, f32) -> f32) -> (Vec<Row>, f32) {
    let mut rows = Vec::new();
    let mut y = 0.0;
    let add = |rows: &mut Vec<Row>, y: &mut f32, words: &str, size: f32, color: Color| {
        for line in wrap(words, size, width, measure) {
            rows.push(Row::Text { text: line, size, color, y: *y });
            *y += size + 4.0;
        }
    };
    add(&mut rows, &mut y, &card.title, text + 1.0, pal::WHITE);
    add(&mut rows, &mut y, &card.meta, small, pal::DIM);
    let h = pic_height(card.pic, width);
    if h > 0.0 {
        y += 6.0;
        rows.push(Row::Pic { pic: card.pic, y, h });
        y += h + 2.0;
    }
    for (head, body) in &card.blocks {
        y += 6.0;
        add(&mut rows, &mut y, &head.to_uppercase(), small - 1.0, pal::MUTED);
        add(&mut rows, &mut y, body, small + 0.5, pal::TEXT);
    }
    if !card.foot.is_empty() {
        y += 6.0;
        add(&mut rows, &mut y, &card.foot, small - 0.5, pal::DIM);
    }
    (rows, y)
}

/// A card laid out to fit `height`: as it is, or without its picture, or in smaller
/// type, whichever fits first (the last if none does).
fn fit_card(card: &Card, width: f32, height: f32, text: f32, small: f32, measure: &dyn Fn(&str, f32) -> f32) -> (Vec<Row>, f32) {
    let laid = lay_out(card, width, text, small, measure);
    if laid.1 <= height {
        return laid;
    }
    let bare = Card { pic: Pic::None, ..card.clone() };
    let laid = lay_out(&bare, width, text, small, measure);
    if laid.1 <= height {
        return laid;
    }
    lay_out(&bare, width, text - 1.0, small - 1.0, measure)
}

// ----- the scene --------------------------------------------------------------------------

/// What the pointer is on in the graph.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum On {
    Var(usize),
    Arrow(usize),
}

struct Shingles {
    choice: Choice,
    /// The two layers of variables the visitor has turned on.
    layers: [bool; 2],
    pane: Pane,
    chooser: Chooser,
    /// The group of studies open in a list too long to show whole.
    group: usize,
    /// What is drawn: on its way from `from` to `to`, `t` of the way.
    from: Look,
    to: Look,
    t: f32,
    /// What the pointer is on, and what was tapped (kept until the next tap).
    hover: Option<On>,
    pinned: Option<On>,
    /// The size of the canvas the picture is laid out for.
    canvas: Vec2,
    /// How lit each arrow and each variable is drawn (0 to 1), easing towards whatever
    /// the pointer is on; and whether any is still on its way.
    lit: Vec<f32>,
    lighting: bool,
    /// Cards laid out, by what they show and how wide.
    cards: HashMap<(Choice, i32, i32), (Vec<Row>, f32)>,
    /// Texts broken into lines, by their text, size and width.
    lines: HashMap<(String, i32, i32), Vec<String>>,
}

impl Shingles {
    fn new() -> Shingles {
        let look = Look::of(Choice::default(), [false; 2]);
        Shingles {
            choice: Choice::default(),
            layers: [false; 2],
            pane: Pane::Graph,
            chooser: Chooser::Table,
            group: 0,
            from: look.clone(),
            to: look,
            t: 1.0,
            hover: None,
            pinned: None,
            canvas: Vec2::ZERO,
            lit: vec![0.0; ARROWS.len() + VARS.len()],
            lighting: false,
            cards: HashMap::new(),
            lines: HashMap::new(),
        }
    }

    /// Change what is chosen, and glide the graph there.
    fn choose(&mut self, choice: Choice) {
        if choice == self.choice {
            return;
        }
        self.choice = choice;
        if let Some(s) = choice.study.and_then(|s| STUDIES.get(s)) {
            self.group = s.group;
        } else if let Some(d) = choice.design {
            self.group = group_of(d);
        }
        self.restyle();
    }

    fn restyle(&mut self) {
        let now = self.from.towards(&self.to, self.t);
        self.from = now;
        self.to = Look::of(self.choice, self.layers);
        self.t = if self.from == self.to { 1.0 } else { 0.0 };
    }

    fn set_layer(&mut self, k: usize, on: bool) {
        if let Some(l) = self.layers.get_mut(k) {
            if *l != on {
                *l = on;
                self.restyle();
            }
        }
    }

    /// Set up one of the experiments in `TRIALS`: its design, study and explanation,
    /// the layers off, and the graph in view.
    fn set_up(&mut self, k: usize) {
        let Some(trial) = TRIALS.get(k) else { return };
        let study = trial.study.and_then(study);
        self.layers = [false; 2];
        self.pinned = None;
        self.pane = Pane::Graph;
        self.choice = Choice { design: trial.design, study, story: trial.story };
        if let Some(s) = study.and_then(|s| STUDIES.get(s)) {
            self.group = s.group;
        }
        self.restyle();
    }

    /// What is drawn this frame.
    fn shown(&self) -> Look {
        if self.t >= 1.0 {
            self.to.clone()
        } else {
            self.from.towards(&self.to, self.t)
        }
    }

    /// Bring the glide `dt` seconds on; true while anything is still on its way.
    fn glide(&mut self, dt: f32) -> bool {
        if self.t >= 1.0 {
            return false;
        }
        self.t += (1.0 - self.t) * (1.0 - (-dt.max(0.0) / GLIDE).exp());
        // (exactly there in the end: the picture then stops changing)
        if 1.0 - self.t < 0.004 || !self.t.is_finite() {
            self.t = 1.0;
            self.from = self.to.clone();
        }
        self.t < 1.0
    }

    fn wrapped(&mut self, f: &Frame, text: &str, size: f32, width: f32) -> Vec<String> {
        let key = (text.to_string(), (size * 10.0) as i32, width as i32);
        if let Some(lines) = self.lines.get(&key) {
            return lines.clone();
        }
        if self.lines.len() > 400 {
            self.lines.clear();
        }
        let lines = wrap(text, size, width, &|t, s| f.text_size(t, s).x);
        self.lines.insert(key, lines.clone());
        lines
    }
}

/// The list's group for a design.
fn group_of(design: usize) -> usize {
    match design {
        COHORT => 0,
        COMPARATOR => 1,
        SWITCH => 2,
        BIRTH => 3,
        EMULATION => 4,
        _ => 5,
    }
}

impl Scene for Shingles {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 3.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    /// Still, unless the graph is gliding to a new choice.
    fn animated(&self) -> bool {
        self.t < 1.0 || self.lighting
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("click a row, a column or a cell of the table, or a study    point at an arrow or a variable: why it is there")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "Does vaccination against shingles lower the risk of dementia? No randomised trial has reported: two began in 2026, and their registries estimate primary completion in 2029 and 2030. Every published estimate rests on an assumption that its data cannot fully test, and what every study analyses is a recorded diagnosis, not the disease itself.");
        ui::note(ui, "This is a map of the published studies: the causal graph each kind of study assumes, the arrows its design cuts or leaves open, and which explanations it can and cannot rule out. It teaches how to read the evidence; it gives no verdict on whether the vaccine works, and it is not medical advice.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "What to show");
        let mut names: Vec<&str> = vec!["no design"];
        names.extend(DESIGNS.iter().map(|d| d.name));
        let mut d = self.choice.graph().map_or(0, |d| d + 1);
        if ui::choice(ui, &names, &mut d) {
            self.choose(Choice { design: d.checked_sub(1), study: None, story: self.choice.story });
        }
        ui::note(ui, "A design colours the graph by what it closes and what it leaves open.");
        let mut names: Vec<&str> = vec!["no explanation"];
        names.extend(STORIES.iter().map(|s| s.name));
        let mut s = self.choice.story.map_or(0, |s| s + 1);
        if ui::choice(ui, &names, &mut s) {
            self.choose(Choice { story: s.checked_sub(1), ..self.choice });
        }
        ui::note(ui, "An explanation lights the arrows it needs. With a design too, they take the state the design leaves the arrow the explanation enters by, and the table's cell says whether that design can tell it apart.");
        let mut who = self.layers[0];
        if ui.checkbox(&mut who, "more causes of who is vaccinated").changed() {
            self.set_layer(0, who);
        }
        let mut record = self.layers[1];
        if ui.checkbox(&mut record, "recording and follow-up").changed() {
            self.set_layer(1, record);
        }
        ui::note(ui, "The core graph has eleven variables; these add nine more. A choice that needs some of them shows them.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (CAUSE, r"the exposure: vaccination $V$ and which vaccine $T$; a double ring marks what a design compares"),
                (EFFECT, r"dementia $D$ and its recorded diagnosis $D^*$; a dashed ring: not in the data"),
                (THIRD, "everything else; a box: held (matched, adjusted, or the same for everyone compared); a dashed box: held only as far as the records go"),
                (TRUTH, "green: a path through dementia itself (what a design estimates; an explanation in which the vaccine works)"),
                (NAIVE, "pink: a path that would give a difference in diagnoses without the vaccine changing dementia; faint, closed only in part; dashed with ?, an arrow the design must assume absent"),
                (CLOSED, "grey: a path the design closes; a stroke across an arrow: cut by randomisation"),
            ],
        );

        ui::section(ui, "Words");
        ui::term(
            ui,
            "Recorded diagnosis:",
            r"what the studies count, $D^*$: a dementia code in health records, death certificates or registries, with its date. It can lag the disease, $D$, or miss it.",
        );
        ui::term(
            ui,
            "Back door:",
            "a path from what is compared to the outcome that starts with an arrow into what is compared. It carries association that is not an effect.",
        );
        ui::term(ui, "Held:", "fixed by the design: matched, adjusted for, or the same for everyone compared. Holding a variable where two arrows meet opens the path through it.");
        ui::term(ui, "Healthy-vaccinee effect:", "people who get vaccinated tend to be healthier and more engaged with health care than those who do not, which can make any vaccine look protective.");
        ui::term(ui, "Regression discontinuity:", "comparing people just either side of a threshold (here a date of birth) that changes who is eligible. The jump in the outcome there is the effect of eligibility (intention to treat, ITT); divided by the jump in uptake, the effect of receiving the vaccine among those vaccinated because eligible.");
        ui::term(ui, "Active comparator:", "a comparison group given another vaccine instead of none.");
        ui::term(ui, "Target-trial emulation:", "analysing observational data as a trial would be analysed, with one start of follow-up for everyone.");
        ui::term(ui, "Hazard ratio (HR; aHR, adjusted):", "the ratio of the rates of new diagnoses in two groups among people not yet diagnosed.");
        ui::term(ui, "Risk ratio (RR), odds ratio (OR):", "the ratio of the risks, or of the odds, of a diagnosis in two groups.");
        ui::term(
            ui,
            "Restricted mean time lost (RMTL):",
            "the average time lived after a diagnosis within a fixed window; its ratio divides one group's by the other's.",
        );
        ui::term(ui, "Percentage points (pp):", "a difference between two percentages: 20% against 17% is 3 pp.");
        ui::term(ui, "Negative-control outcome:", "an outcome the vaccine should not affect but that shares the suspected biases; an association with it signals bias. A positive control is one it is known to affect (here, shingles itself).");
        ui::term(ui, "Immortal time:", "follow-up during which, by the way exposure is defined, the outcome cannot occur or be counted for one group.");
        ui::term(ui, r"$I^2$:", "in a meta-analysis, the share of the variation between studies' estimates that is more than chance would give.");
        ui::term(ui, "Recombinant and live vaccines:", "the recombinant vaccine (Shingrix, RZV) is a protein of the virus with the AS01B adjuvant, two doses; the live vaccine (Zostavax, ZVL) is a weakened virus, one dose. An adjuvant is added to a vaccine to strengthen the immune response.");

        ui::section(ui, "Numbers");
        match self.choice.study.and_then(|s| STUDIES.get(s)) {
            Some(s) => {
                // (short values only: a long one does not wrap, and would push the panel wider)
                ui::readout(ui, "study", s.short);
                ui::note(ui, &format!("Its main estimate: {}.", s.estimate));
                ui::readout(ui, "read for this map", s.read.brief());
            }
            None => {
                let read = STUDIES.iter().filter(|s| matches!(s.read, Read::Full | Read::Part)).count();
                ui::readout(ui, "sources on the map", STUDIES.len().to_string());
                ui::readout(ui, "read in full or in part", read.to_string());
                ui::readout(ui, "randomised trials reported", "none");
            }
        }
        ui::note(ui, "Each estimate is in its own unit: a hazard ratio, a risk ratio, a ratio of restricted mean time lost and a difference in percentage points are different quantities, and are not put on one scale here.");

        egui::CollapsingHeader::new("Sources as read on 25 November 2026").id_salt("shingles-sources").show(ui, |ui| {
            ui::note(ui, "Everything here comes from notes compiled on 25 November 2026 from the papers and the trial registries. For each source: first author, year, journal, DOI, how much of it was read, and its funding or competing interests where the notes record them, as printed.");
            for s in &STUDIES {
                let doi = if s.doi.is_empty() { String::new() } else { format!(" doi:{}", s.doi) };
                ui::note(ui, &format!("{}. {}.{} ({})", s.by, s.cite, doi, s.read.says()));
                if !s.disclosure.is_empty() {
                    ui::words(ui, s.disclosure, 11.0, pal::FAINT);
                }
            }
        });

        ui::see_also(
            ui,
            &[
                ("switch", "two vaccines, one switch: the 2017 study taken apart"),
                ("confound", "confounding: a third variable behind both"),
                ("adjust", "what to control for, and what not"),
                ("junctions", "chains, forks and colliders"),
                ("dosee", "seeing and doing: what randomisation does"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        // the pointer belongs to the picture: there is nothing to pan or zoom
        f.handles.capture();
        f.handles.claim_clicks();
        // (laid out again only when the canvas changes by more than a pixel either way)
        let seen = f.view_size();
        if (seen - self.canvas).abs().max_element() >= 1.5 {
            self.canvas = seen;
            self.cards.clear();
        }
        let view = self.canvas;
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        if view.x < 150.0 || view.y < 120.0 {
            self.glide(f.real_dt);
            return;
        }
        let lay = Lay::of(view, f.top_inset, self.pane, self.chooser);
        let pointer = f.ptr.pos.map(|p| p / f.px);
        self.draw_tabs(f, &lay, pointer);
        self.draw_graph(f, &lay, pointer);
        self.draw_switches(f, &lay, pointer);
        self.draw_card(f, &lay);
        self.draw_table(f, &lay, pointer);
        self.draw_list(f, &lay, pointer);
        if let Some(at) = lay.footer {
            let text = if lay.wide {
                "Sources as read on 25 November 2026: each study's first author, journal, year and DOI are in its card and in the panel's Sources."
            } else {
                "Sources as read on 25 November 2026: in each study's card and the panel."
            };
            f.text_px(at, text, lay.small - 0.5, pal::FAINT, Align2::LEFT_BOTTOM);
        }
        self.glide(f.real_dt);
    }
}

impl Shingles {
    fn draw_tabs(&mut self, f: &mut Frame, lay: &Lay, pointer: Option<Vec2>) {
        let Some((lo, hi)) = lay.tabs else { return };
        if lay.wide {
            let names = [(Chooser::Table, "designs and explanations"), (Chooser::Studies, "the studies, by design")];
            let mut x = lo.x;
            for (c, name) in names {
                let w = f.text_size(name, lay.small).x + 28.0;
                if hud_button(f, (Vec2::new(x, lo.y), Vec2::new(x + w, hi.y)), name, lay.small, self.chooser == c, pointer) {
                    self.chooser = c;
                }
                x += w + 8.0;
            }
            if self.choice != Choice::default() {
                let w = f.text_size("clear the choice", lay.small).x + 24.0;
                if hud_button(f, (Vec2::new(hi.x - w, lo.y), Vec2::new(hi.x, hi.y)), "clear the choice", lay.small, false, pointer) {
                    self.choose(Choice::default());
                }
            }
            return;
        }
        let w = (hi.x - lo.x - 6.0 * (PANES.len() - 1) as f32) / PANES.len() as f32;
        for (k, (pane, name)) in PANES.iter().enumerate() {
            let x = lo.x + k as f32 * (w + 6.0);
            if hud_button(f, (Vec2::new(x, lo.y), Vec2::new(x + w, hi.y)), name, lay.small, self.pane == *pane, pointer) {
                self.pane = *pane;
            }
        }
    }

    fn draw_graph(&mut self, f: &mut Frame, lay: &Lay, pointer: Option<Vec2>) {
        let Some(rect) = lay.graph else { return };
        let size = rect.1 - rect.0;
        let roomy = size.x >= 300.0 && size.y >= 220.0;
        let r = if roomy { lay.radius } else { (size.y / 24.0).clamp(6.0, lay.radius) };
        let at = places(rect, r, roomy);
        let look = self.shown();
        // the arrows' shapes, for drawing and for finding the one under the pointer
        let shapes: Vec<Vec<Vec2>> = ARROWS.iter().map(|a| arrow_points(at[a.from], at[a.to], a.bend, r)).collect();
        // what the pointer is on: a variable before an arrow
        self.hover = pointer.filter(|p| inside(*p, rect)).and_then(|p| {
            let var = (0..VARS.len()).filter(|&i| look.vars[i].alpha > 0.5).find(|&i| at[i].distance(p) <= r + 4.0);
            let arrow = || {
                (0..ARROWS.len())
                    .filter(|&k| look.arrows[k].alpha > 0.5)
                    .map(|k| (k, path_distance(p, &shapes[k])))
                    .filter(|(_, d)| *d <= 6.0)
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .map(|(k, _)| k)
            };
            var.map(On::Var).or_else(|| arrow().map(On::Arrow))
        });
        if f.ptr.clicked && pointer.is_some_and(|p| inside(p, rect)) {
            self.pinned = if self.hover == self.pinned { None } else { self.hover };
        }
        let focus = self.hover.or(self.pinned);
        // what the pointer is on lights up, and goes out again, by the clock (it answers
        // the hand), not at once
        let ease = 1.0 - (-f.real_dt.max(0.0) / 0.06).exp();
        self.lighting = false;
        for (j, x) in self.lit.iter_mut().enumerate() {
            let on = if j < ARROWS.len() { focus == Some(On::Arrow(j)) } else { focus == Some(On::Var(j - ARROWS.len())) };
            let to = if on { 1.0 } else { 0.0 };
            *x += (to - *x) * ease;
            if (to - *x).abs() < 0.01 || !x.is_finite() {
                *x = to;
            } else {
                self.lighting = true;
            }
        }

        // the arrows
        for k in 0..ARROWS.len() {
            let l = look.arrows[k];
            let pts = &shapes[k];
            if l.alpha < 0.01 || pts.len() < 2 {
                continue;
            }
            // (lit by brightening, not widening: the line keeps its place and its width)
            let lit = self.lit.get(k).copied().unwrap_or(0.0);
            let width = l.width;
            let base = l.color.mix(pal::WHITE, 0.45 * lit);
            let color = base.a(l.alpha * (l.color.0[3] as f32 / 255.0 + (1.0 - l.color.0[3] as f32 / 255.0) * lit));
            match l.carries {
                Carries::Assumed => dashed_px(f, pts, width, color, 9.0, 5.0),
                Carries::Cut => dashed_px(f, pts, width, color.a(color.0[3] as f32 / 255.0 * 0.6), 12.0, 6.0),
                _ => hud_path(f, pts.iter().copied(), width, color),
            }
            // the head, along the last stretch
            let (Some(&tip), Some(&before)) = (pts.last(), pts.get(pts.len().saturating_sub(2))) else { continue };
            let along = (tip - before).normalize_or_zero();
            let side = Vec2::new(-along.y, along.x);
            let head = 4.0 + 2.5 * width.min(3.0);
            hud_line(f, tip, tip - along * head + side * head * 0.5, width, color);
            hud_line(f, tip, tip - along * head - side * head * 0.5, width, color);
            let mid = path_middle(pts);
            if l.carries == Carries::Cut {
                let across = (side + along * 0.6).normalize_or_zero();
                hud_line(f, mid - across * 7.0, mid + across * 7.0, width + 0.6, CLOSED.mix(pal::WHITE, 0.25).a(l.alpha));
            }
            if l.carries == Carries::Assumed && roomy {
                disc(f, mid, 8.0, Color::hex(0x0A0D16).a(l.alpha));
                hud_circle(f, mid, 8.0, 1.4, NAIVE.a(l.alpha));
                f.text_px(mid, "?", lay.text, NAIVE.a(l.alpha), Align2::CENTER_CENTER);
            }
        }

        // the variables
        for (i, v) in VARS.iter().enumerate() {
            let l = look.vars[i];
            if l.alpha < 0.01 {
                continue;
            }
            let color = match v.part {
                Part::Exposure => CAUSE,
                Part::Outcome => EFFECT,
                Part::Other => THIRD,
            }
            .a(l.alpha);
            let p = at[i];
            let lit = self.lit.get(ARROWS.len() + i).copied().unwrap_or(0.0);
            if v.seen == Seen::Yes {
                hud_circle(f, p, r, 1.8, color.mix(pal::WHITE, 0.45 * lit));
            } else {
                dashed_ring(f, p, r, 1.8, color.mix(pal::WHITE, 0.45 * lit));
            }
            if l.compared > 0.01 {
                hud_circle(f, p, r + 4.5, 1.8, color.a(l.alpha * l.compared));
            }
            if l.held > 0.01 {
                let half = Vec2::splat(r + if l.compared > 0.01 { 9.0 } else { 5.0 });
                let c = pal::WHITE.a(0.85 * l.held * l.alpha);
                let corners = [p - half, Vec2::new(p.x + half.x, p.y - half.y), p + half, Vec2::new(p.x - half.x, p.y + half.y), p - half];
                if l.partly {
                    dashed_px(f, &corners, 1.4, c, 7.0, 4.0);
                } else {
                    hud_path(f, corners.into_iter(), 1.4, c);
                }
            }
            f.text_px(p, v.sym, (r * 0.95).max(9.0), color, Align2::CENTER_CENTER);
            if roomy {
                let name = if size.x >= 480.0 { v.name } else { v.short };
                let gap = r + if l.held > 0.01 { 8.0 } else { 4.0 };
                let (off, align) = match v.side {
                    Side::Above => (Vec2::new(0.0, -gap), Align2::CENTER_BOTTOM),
                    Side::Below => (Vec2::new(0.0, gap), Align2::CENTER_TOP),
                    Side::Left => (Vec2::new(-gap - 2.0, 0.0), Align2::RIGHT_CENTER),
                    Side::Right => (Vec2::new(gap + 2.0, 0.0), Align2::LEFT_CENTER),
                };
                f.text_px(p + off, name, lay.small - 0.5, pal::DIM.a(l.alpha), align);
            }
        }

        // what the pointer is on
        if let Some((lo, hi)) = lay.status {
            let text = match focus {
                Some(On::Var(i)) => format!("{} {}: {}", VARS[i].sym, VARS[i].name, VARS[i].note),
                Some(On::Arrow(k)) => {
                    let a = &ARROWS[k];
                    let what = match look.arrows[k].carries {
                        Carries::Idle | Carries::Closed => "",
                        Carries::Effect => " In this design: on the path it estimates.",
                        Carries::Open => " In this design: on a path it leaves open.",
                        Carries::Weak => " In this design: on a path closed only as far as the records go.",
                        Carries::Assumed => " In this design: assumed absent.",
                        Carries::Cut => " In this design: cut by randomisation.",
                    };
                    format!("${} \\to {}$: {}{}", VARS[a.from].sym.trim_matches('$'), VARS[a.to].sym.trim_matches('$'), a.note, what)
                }
                None => "Point at an arrow or a variable to see why it is there.".to_string(),
            };
            let lines = self.wrapped(f, &text, lay.small, hi.x - lo.x);
            let rows = (((hi.y - lo.y) / (lay.small + 3.0)).floor() as usize).max(1);
            for (k, line) in lines.iter().take(rows).enumerate() {
                let line = if k + 1 == rows && lines.len() > rows { format!("{line} …") } else { line.clone() };
                f.text_px(Vec2::new(lo.x, lo.y + k as f32 * (lay.small + 3.0)), line, lay.small, pal::DIM, Align2::LEFT_TOP);
            }
        }

        // a narrow canvas: what is chosen, in brief
        if let Some((lo, hi)) = lay.gist {
            let (title, body) = gist(self.choice);
            let mut y = lo.y;
            for (text, size, color) in [(title, lay.text, pal::WHITE), (body, lay.small, pal::TEXT)] {
                for line in self.wrapped(f, &text, size, hi.x - lo.x) {
                    if y + size > hi.y {
                        break;
                    }
                    f.text_px(Vec2::new(lo.x, y), line, size, color, Align2::LEFT_TOP);
                    y += size + 3.0;
                }
                y += 3.0;
            }
            if y + lay.small <= hi.y {
                f.text_px(Vec2::new(lo.x, y), "the whole text: the details tab", lay.small - 0.5, pal::FAINT, Align2::LEFT_TOP);
            }
        }
    }

    /// The layer switches, and under them a key to the lines.
    fn draw_switches(&mut self, f: &mut Frame, lay: &Lay, pointer: Option<Vec2>) {
        let Some((lo, hi)) = lay.switches else { return };
        let on = shown_layers(self.choice, self.layers);
        let names = if hi.x - lo.x >= 520.0 { ["+ who is vaccinated", "+ recording, follow-up"] } else { ["+ who", "+ recording"] };
        let mut x = lo.x;
        for k in 0..2 {
            let w = f.text_size(names[k], lay.small).x + 18.0;
            if hud_button(f, (Vec2::new(x, lo.y), Vec2::new(x + w, lo.y + 24.0)), names[k], lay.small, on[k], pointer) {
                self.set_layer(k, !self.layers[k]);
            }
            x += w + 6.0;
        }
        // the kinds of line, on as many rows as they need
        let size = lay.small - 0.5;
        let entries: [(Color, &str, bool); 5] = [
            (TRUTH, "through dementia", false),
            (NAIVE, "without it", false),
            (NAIVE.a(0.42), "closed in part", false),
            (CLOSED, "closed", false),
            (NAIVE, "assumed absent", true),
        ];
        let (mut x, mut y) = (lo.x, lo.y + 24.0 + 4.0 + size * 0.5 + 1.0);
        for (color, text, dashed) in entries {
            let w = f.text_size(text, size).x + 26.0;
            if x + w > hi.x && x > lo.x {
                x = lo.x;
                y += size + 5.0;
            }
            if y + size * 0.5 > hi.y + 1.0 {
                break;
            }
            let (a, b) = (Vec2::new(x, y), Vec2::new(x + 16.0, y));
            if dashed {
                dashed_px(f, &[a, b], 2.0, color, 6.0, 3.5);
            } else {
                hud_line(f, a, b, 2.4, color);
            }
            f.text_px(Vec2::new(x + 21.0, y), text, size, pal::DIM, Align2::LEFT_CENTER);
            x += w + 8.0;
        }
    }

    fn draw_card(&mut self, f: &mut Frame, lay: &Lay) {
        let Some((lo, hi)) = lay.card else { return };
        let width = hi.x - lo.x;
        let key = (self.choice, width as i32, (lay.text * 10.0) as i32);
        if !self.cards.contains_key(&key) {
            if self.cards.len() > 64 {
                self.cards.clear();
            }
            let laid = fit_card(&card(self.choice), width, hi.y - lo.y, lay.text, lay.small, &|t, s| f.text_size(t, s).x);
            self.cards.insert(key, laid);
        }
        let Some((rows, _)) = self.cards.get(&key).cloned() else { return };
        let bottom = hi.y;
        let mut cut = false;
        for row in &rows {
            match row {
                Row::Text { text, size, color, y } => {
                    if lo.y + y + size > bottom {
                        cut = true;
                        break;
                    }
                    f.text_px(Vec2::new(lo.x, lo.y + y), text.as_str(), *size, *color, Align2::LEFT_TOP);
                }
                Row::Pic { pic, y, h } => {
                    if lo.y + y + h > bottom {
                        cut = true;
                        break;
                    }
                    draw_pic(f, *pic, (Vec2::new(lo.x, lo.y + y), Vec2::new(hi.x, lo.y + y + h)), lay.small);
                }
            }
        }
        if cut {
            f.text_px(Vec2::new(hi.x, bottom), "… (more in a larger window)", lay.small - 0.5, pal::FAINT, Align2::RIGHT_BOTTOM);
        }
    }

    fn draw_table(&mut self, f: &mut Frame, lay: &Lay, pointer: Option<Vec2>) {
        let Some((lo, hi)) = lay.table else { return };
        let small = lay.small;
        let names: Vec<&str> = DESIGNS.iter().map(|d| if lay.wide { d.name } else { d.short }).collect();
        let first = names.iter().map(|n| f.text_size(n, small).x).fold(0.0, f32::max) + 14.0;
        let cols = STORIES.len() as f32;
        let cw = ((hi.x - lo.x - first) / cols).max(10.0);
        let compact = cw < 74.0;
        let head_h = if compact { small + 10.0 } else { 2.0 * small + 12.0 };
        let legend_h = if compact { 4.0 * (small + 3.0) } else { 0.0 };
        let note_h = small + 6.0;
        let rh = ((hi.y - lo.y - head_h - legend_h - note_h) / DESIGNS.len() as f32).clamp(16.0, 34.0);
        let x_of = |j: usize| lo.x + first + cw * (j as f32 + 0.5);
        let y_of = |i: usize| lo.y + head_h + rh * (i as f32 + 0.5);
        let graph = self.choice.graph();
        let story = self.choice.story;
        // the chosen row and column, lit
        if let Some(i) = graph.filter(|&i| i < DESIGNS.len()) {
            hud_rect(f, Vec2::new(lo.x - 4.0, y_of(i) - rh * 0.5), Vec2::new(lo.x + first + cw * cols, y_of(i) + rh * 0.5), Color::hex(0x1C2538));
        }
        if let Some(j) = story {
            hud_rect(f, Vec2::new(x_of(j) - cw * 0.5, lo.y), Vec2::new(x_of(j) + cw * 0.5, lo.y + head_h + rh * DESIGNS.len() as f32), Color::hex(0x1C2538));
        }
        // the pointer over a header or a cell
        let over = pointer.and_then(|p| {
            let row = ((p.y - lo.y - head_h) / rh).floor();
            let col = ((p.x - lo.x - first) / cw).floor();
            let in_rows = (0.0..DESIGNS.len() as f32).contains(&row);
            let in_cols = (0.0..cols).contains(&col);
            if p.x >= lo.x - 4.0 && p.x < lo.x + first && in_rows {
                Some((Some(row as usize), None))
            } else if in_cols && p.y >= lo.y && p.y < lo.y + head_h {
                Some((None, Some(col as usize)))
            } else if in_rows && in_cols {
                Some((Some(row as usize), Some(col as usize)))
            } else {
                None
            }
        });
        if let Some((i, j)) = over {
            let (a, b) = match (i, j) {
                (Some(i), Some(j)) => (Vec2::new(x_of(j) - cw * 0.5, y_of(i) - rh * 0.5), Vec2::new(x_of(j) + cw * 0.5, y_of(i) + rh * 0.5)),
                (Some(i), None) => (Vec2::new(lo.x - 4.0, y_of(i) - rh * 0.5), Vec2::new(lo.x + first - 4.0, y_of(i) + rh * 0.5)),
                (None, Some(j)) => (Vec2::new(x_of(j) - cw * 0.5, lo.y), Vec2::new(x_of(j) + cw * 0.5, lo.y + head_h)),
                (None, None) => (Vec2::ZERO, Vec2::ZERO),
            };
            hud_outline(f, a, b, 1.2, pal::WHITE.a(0.5));
            if f.ptr.clicked {
                let c = self.choice;
                let next = match (i, j) {
                    (Some(i), Some(j)) if graph == Some(i) && story == Some(j) && c.study.is_none() => Choice { design: Some(i), study: None, story: None },
                    (Some(i), Some(j)) => Choice { design: Some(i), study: None, story: Some(j) },
                    (Some(i), None) if graph == Some(i) && c.study.is_none() => Choice { design: None, study: None, story: c.story },
                    (Some(i), None) => Choice { design: Some(i), study: None, story: c.story },
                    (None, Some(j)) if story == Some(j) => Choice { story: None, ..c },
                    (None, Some(j)) => Choice { story: Some(j), ..c },
                    (None, None) => c,
                };
                self.choose(next);
                if !lay.wide {
                    self.pane = Pane::Graph;
                }
            }
        }
        // the headers
        for (j, s) in STORIES.iter().enumerate() {
            let color = if j == 7 {
                pal::DIM
            } else if s.real {
                TRUTH
            } else {
                NAIVE
            };
            if compact {
                f.text_px(Vec2::new(x_of(j), lo.y + head_h * 0.5), s.label, small, color, Align2::CENTER_CENTER);
            } else {
                f.text_px(Vec2::new(x_of(j), lo.y + 4.0), s.head[0], small, color, Align2::CENTER_TOP);
                f.text_px(Vec2::new(x_of(j), lo.y + 6.0 + small), s.head[1], small, color, Align2::CENTER_TOP);
            }
        }
        f.text_px(Vec2::new(lo.x, lo.y + head_h * 0.5), "design", small, pal::MUTED, Align2::LEFT_CENTER);
        for (i, name) in names.iter().enumerate() {
            let chosen = graph == Some(i);
            f.text_px(Vec2::new(lo.x, y_of(i)), *name, small, if chosen { pal::WHITE } else { pal::TEXT }, Align2::LEFT_CENTER);
            hud_line(f, Vec2::new(lo.x, y_of(i) + rh * 0.5), Vec2::new(lo.x + first + cw * cols, y_of(i) + rh * 0.5), 1.0, pal::FAINT.a(0.4));
            for j in 0..STORIES.len() {
                draw_mark(f, Vec2::new(x_of(j), y_of(i)), CELLS[i][j].mark, (rh * 0.22).clamp(4.0, 6.5));
            }
        }
        // what the marks mean, and (compact) what the numbers are
        let mut y = lo.y + head_h + rh * DESIGNS.len() as f32 + 6.0;
        let mut x = lo.x;
        for m in [Mark::Can, Mark::Partly, Mark::Cannot, Mark::Open] {
            let w = f.text_size(m.says(), small - 0.5).x + 22.0;
            if x + w > hi.x {
                x = lo.x;
                y += small + 5.0;
            }
            draw_mark(f, Vec2::new(x + 6.0, y + small * 0.5), m, 4.5);
            f.text_px(Vec2::new(x + 16.0, y), m.says(), small - 0.5, pal::DIM, Align2::LEFT_TOP);
            x += w + 8.0;
        }
        if compact {
            y += small + 6.0;
            let legend: Vec<String> = STORIES.iter().map(|s| format!("{} {}", s.label, s.name)).collect();
            for line in self.wrapped(f, &legend.join(" · "), small - 0.5, hi.x - lo.x) {
                if y + small > hi.y + 12.0 {
                    break;
                }
                f.text_px(Vec2::new(lo.x, y), line, small - 0.5, pal::DIM, Align2::LEFT_TOP);
                y += small + 2.0;
            }
        }
    }

    fn draw_list(&mut self, f: &mut Frame, lay: &Lay, pointer: Option<Vec2>) {
        let Some((lo, hi)) = lay.list else { return };
        let small = lay.small;
        let rh = small + 7.0;
        let ncol = (((hi.x - lo.x) / 340.0).floor() as usize).clamp(1, 3);
        let cw = (hi.x - lo.x) / ncol as f32;
        let per = (((hi.y - lo.y) / rh).floor() as usize).max(1);
        // the rows: each group's heading and its studies; when they do not all fit,
        // only the open group's studies, under every heading
        let all: usize = GROUPS.len() + STUDIES.len();
        let whole = all <= per * ncol;
        let mut rows: Vec<(Option<usize>, usize)> = Vec::new();
        for (g, _) in GROUPS.iter().enumerate() {
            rows.push((None, g));
            if whole || g == self.group {
                rows.extend(STUDIES.iter().enumerate().filter(|(_, s)| s.group == g).map(|(k, _)| (Some(k), g)));
            }
        }
        // (columns of about equal length, and a heading not left alone at the foot of one)
        let per = if whole { rows.len().div_ceil(ncol).max(1).min(per) + 1 } else { per };
        let mut placed: Vec<(Vec2, Option<usize>, usize, usize)> = Vec::new();
        let (mut col, mut line) = (0usize, 0usize);
        for (k, (study, g)) in rows.iter().enumerate() {
            let next_is_study = rows.get(k + 1).is_some_and(|r| r.0.is_some());
            if line >= per || (study.is_none() && next_is_study && line + 1 >= per && line > 0) {
                col += 1;
                line = 0;
            }
            if col >= ncol {
                break;
            }
            placed.push((Vec2::new(lo.x + col as f32 * cw, lo.y + line as f32 * rh), *study, *g, col));
            line += 1;
        }
        // each column's estimates start after its widest name, but for a name much
        // longer than the rest, which pushes only its own
        let mut name_w = vec![0.0f32; ncol];
        for (_, study, _, c) in &placed {
            if let (Some(k), Some(w)) = (study, name_w.get_mut(*c)) {
                let own = f.text_size(STUDIES[*k].short, small).x;
                if own <= 0.4 * cw {
                    *w = w.max(own);
                }
            }
        }
        for (at, study, g, c) in placed {
            let row = (at, at + Vec2::new(cw - 10.0, rh));
            let over = pointer.is_some_and(|p| inside(p, row));
            match study {
                None => {
                    let open = whole || g == self.group;
                    let count = STUDIES.iter().filter(|s| s.group == g).count();
                    let text = format!("{} {} ({})", if open { "▾" } else { "▸" }, GROUPS[g], count);
                    f.text_px(at + Vec2::new(0.0, rh * 0.5), text, small, if over { pal::WHITE } else { pal::MUTED }, Align2::LEFT_CENTER);
                    if over && f.ptr.clicked {
                        self.group = g;
                    }
                }
                Some(k) => {
                    let s = &STUDIES[k];
                    let chosen = self.choice.study == Some(k);
                    if chosen || over {
                        hud_rect(f, row.0 - Vec2::new(4.0, 0.0), row.1, if chosen { Color::hex(0x26324C) } else { Color::hex(0x1C2538) });
                    }
                    let x2 = at.x + 12.0 + name_w.get(c).copied().unwrap_or(0.0).max(f.text_size(s.short, small).x) + 14.0;
                    f.text_px(at + Vec2::new(12.0, rh * 0.5), s.short, small, if chosen { pal::WHITE } else { pal::TEXT }, Align2::LEFT_CENTER);
                    let room = at.x + cw - 14.0 - x2;
                    let est = self.wrapped(f, s.estimate, small - 0.5, room.max(20.0));
                    let mut text = est.first().cloned().unwrap_or_default();
                    if est.len() > 1 {
                        text.push_str(" …");
                    }
                    f.text_px(
                        Vec2::new(x2, at.y + rh * 0.5),
                        text,
                        small - 0.5,
                        if s.read == Read::Title { pal::FAINT } else { pal::DIM },
                        Align2::LEFT_CENTER,
                    );
                    if over && f.ptr.clicked {
                        let next =
                            if chosen { Choice { study: None, ..self.choice } } else { Choice { design: s.design, study: Some(k), story: self.choice.story } };
                        self.choose(next);
                        if !lay.wide && !chosen {
                            self.pane = Pane::Graph;
                        }
                    }
                }
            }
        }
    }
}

/// A mark of the table: a full disc (can), a half (partly), a ring (cannot), a dash
/// (not decided by the design).
fn draw_mark(f: &mut Frame, at: Vec2, mark: Mark, r: f32) {
    match mark {
        Mark::Can => disc(f, at, r, pal::TEXT),
        Mark::Partly => {
            hud_circle(f, at, r, 1.3, pal::TEXT);
            let n = 12;
            let c = f.hud_at(1, at);
            for i in 0..n {
                let (a, b) = (i as f32 / n as f32 * PI + PI * 0.5, (i + 1) as f32 / n as f32 * PI + PI * 0.5);
                let (p, q) = (f.hud_at(1, at + Vec2::from_angle(a) * r), f.hud_at(1, at + Vec2::from_angle(b) * r));
                f.dl.tris_alpha.tri(c, p, q, pal::TEXT);
            }
        }
        Mark::Cannot => hud_circle(f, at, r, 1.3, pal::DIM),
        Mark::Open => hud_line(f, at - Vec2::new(r, 0.0), at + Vec2::new(r, 0.0), 1.4, pal::FAINT),
    }
}

/// Estimates with their intervals on one scale, a row each with its name above it, the
/// axis under them with its ticks, and the value of no difference marked on every row.
#[allow(clippy::too_many_arguments)]
fn interval_rows(
    f: &mut Frame,
    (lo, hi): Rect,
    s: f32,
    x: &dyn Fn(f32) -> f32,
    ticks: &[f32],
    none: f32,
    rows: &[(f32, f32, f32, &str)],
    tick: impl Fn(f32) -> String,
) {
    let axis = hi.y - 18.0;
    hud_line(f, Vec2::new(x(ticks[0]), axis), Vec2::new(x(*ticks.last().unwrap_or(&none)), axis), 1.0, pal::DIM);
    for &v in ticks {
        hud_line(f, Vec2::new(x(v), axis), Vec2::new(x(v), axis + 3.0), 1.0, pal::DIM);
        f.text_px(Vec2::new(x(v), axis + 4.0), tick(v), s, if v == none { pal::DIM } else { pal::FAINT }, Align2::CENTER_TOP);
    }
    let step = (axis - lo.y - 22.0) / rows.len().max(1) as f32;
    for (k, &(v, a, b, name)) in rows.iter().enumerate() {
        let y = lo.y + 22.0 + step * (k as f32 + 0.75);
        dashed_px(f, &[Vec2::new(x(none), y - 7.0), Vec2::new(x(none), y + 7.0)], 1.0, pal::DIM, 4.0, 2.0);
        hud_line(f, Vec2::new(x(a), y), Vec2::new(x(b), y), 2.0, pal::WHITE.a(0.55));
        disc(f, Vec2::new(x(v), y), 3.5, pal::WHITE);
        let width = f.text_size(name, s).x;
        let left = x(a).min(hi.x - width).max(lo.x);
        f.text_px(Vec2::new(left, y - 6.0), name, s, pal::TEXT, Align2::LEFT_BOTTOM);
    }
}

/// The picture beside a card's text.
fn draw_pic(f: &mut Frame, pic: Pic, (lo, hi): Rect, small: f32) {
    let s = small - 1.0;
    let w = hi.x - lo.x;
    let mid = (lo.y + hi.y) * 0.5;
    let dots = |f: &mut Frame, x: f32, y: f32, n: usize, color: Color| {
        for k in 0..n {
            disc(f, Vec2::new(x + k as f32 * 11.0, y), 3.2, color);
        }
    };
    match pic {
        Pic::None => {}
        Pic::Cohort | Pic::Comparator => {
            let (top, bottom) = (lo.y + 22.0, lo.y + 54.0);
            let other = if pic == Pic::Cohort { ("not vaccinated", pal::DIM) } else { ("given another vaccine", THIRD) };
            f.text_px(Vec2::new(lo.x, lo.y + 2.0), "how it works (schematic)", s, pal::FAINT, Align2::LEFT_TOP);
            dots(f, lo.x + 6.0, top, 8, CAUSE);
            f.text_px(Vec2::new(lo.x + 98.0, top), "given a zoster vaccine", s, CAUSE, Align2::LEFT_CENTER);
            dots(f, lo.x + 6.0, bottom, 8, other.1);
            f.text_px(Vec2::new(lo.x + 98.0, bottom), other.0, s, other.1, Align2::LEFT_CENTER);
            let note = if pic == Pic::Cohort {
                "who is in which group: their own choice, and their health"
            } else {
                "both groups chose a vaccine; which one, they chose too"
            };
            f.text_px(Vec2::new(lo.x, lo.y + 76.0), note, s, NAIVE.mix(pal::DIM, 0.3), Align2::LEFT_TOP);
        }
        Pic::Switch => {
            // calendar years across: who was vaccinated when, and when the pandemic came
            let (y0, y1) = (2014.0f32, 2022.0f32);
            let x = |year: f32| lo.x + 10.0 + (w - 20.0) * (year - y0) / (y1 - y0);
            let axis = lo.y + 70.0;
            hud_line(f, Vec2::new(x(y0), axis), Vec2::new(x(y1), axis), 1.0, pal::DIM);
            for year in [2014, 2016, 2018, 2020, 2022] {
                f.text_px(Vec2::new(x(year as f32), axis + 4.0), year.to_string(), s, pal::FAINT, Align2::CENTER_TOP);
            }
            let before = (x(2014.75), x(2017.67));
            let after = (x(2017.83), x(2020.83));
            hud_rect(f, Vec2::new(before.0, lo.y + 34.0), Vec2::new(before.1, lo.y + 52.0), THIRD.a(0.35));
            hud_rect(f, Vec2::new(after.0, lo.y + 34.0), Vec2::new(after.1, lo.y + 52.0), CAUSE.a(0.35));
            f.text_px(Vec2::new((before.0 + before.1) * 0.5, lo.y + 43.0), "98% live", s, pal::TEXT, Align2::CENTER_CENTER);
            f.text_px(Vec2::new((after.0 + after.1) * 0.5, lo.y + 43.0), "95% recombinant", s, pal::TEXT, Align2::CENTER_CENTER);
            let pandemic = x(2020.17);
            dashed_px(f, &[Vec2::new(pandemic, lo.y + 18.0), Vec2::new(pandemic, axis)], 1.2, NAIVE.a(0.8), 6.0, 3.0);
            f.text_px(Vec2::new(pandemic + 4.0, lo.y + 18.0), "pandemic", s, NAIVE, Align2::LEFT_TOP);
            let switch = x(2017.75);
            hud_line(f, Vec2::new(switch, lo.y + 26.0), Vec2::new(switch, axis), 1.4, pal::WHITE.a(0.8));
            f.text_px(Vec2::new(switch, lo.y + 24.0), "switch", s, pal::WHITE, Align2::CENTER_BOTTOM);
            f.text_px(Vec2::new(lo.x, lo.y + 2.0), "first doses by calendar year", s, pal::FAINT, Align2::LEFT_TOP);
        }
        Pic::Birth => {
            // two little plots against date of birth: uptake, and new diagnoses
            let cut = lo.x + w * 0.5;
            let span = (w * 0.42).min(170.0);
            let (u0, u1) = (lo.y + 22.0, lo.y + 46.0);
            let (d0, d1) = (lo.y + 56.0, lo.y + 80.0);
            dashed_px(f, &[Vec2::new(cut, lo.y + 16.0), Vec2::new(cut, lo.y + 84.0)], 1.2, pal::WHITE.a(0.7), 6.0, 3.0);
            hud_line(f, Vec2::new(cut - span, u1), Vec2::new(cut, u1), 2.0, CAUSE);
            hud_line(f, Vec2::new(cut, u0 + 2.0), Vec2::new(cut + span, u0 + 2.0), 2.0, CAUSE);
            f.text_px(Vec2::new(cut - span, u0 + 6.0), "uptake", s, CAUSE, Align2::LEFT_CENTER);
            hud_line(f, Vec2::new(cut - span, d0 + 6.0), Vec2::new(cut, d0 + 8.0), 2.0, EFFECT);
            hud_line(f, Vec2::new(cut, d0 + 14.0), Vec2::new(cut + span, d0 + 16.0), 2.0, EFFECT);
            f.text_px(Vec2::new(cut - span, d1 + 2.0), "new diagnoses", s, EFFECT, Align2::LEFT_TOP);
            f.text_px(Vec2::new(cut - 4.0, lo.y + 12.0), "born before: not eligible", s, pal::DIM, Align2::RIGHT_BOTTOM);
            f.text_px(Vec2::new(cut + 4.0, lo.y + 12.0), "after: eligible", s, pal::DIM, Align2::LEFT_BOTTOM);
            f.text_px(Vec2::new(cut + span, d1 + 2.0), "schematic, not data", s, pal::FAINT, Align2::RIGHT_TOP);
        }
        Pic::Clone => {
            let start = Vec2::new(lo.x + 16.0, mid);
            disc(f, start, 4.0, pal::TEXT);
            f.text_px(start + Vec2::new(0.0, 10.0), "admitted", s, pal::DIM, Align2::CENTER_TOP);
            let (a, b) = (Vec2::new(lo.x + w * 0.75, lo.y + 26.0), Vec2::new(lo.x + w * 0.75, lo.y + 70.0));
            hud_arrow(f, start, a, 1.6, CAUSE);
            hud_arrow(f, start, b, 1.6, pal::DIM);
            f.text_px(a + Vec2::new(6.0, 0.0), "copy: vaccinated within 12 months", s, CAUSE, Align2::LEFT_CENTER);
            f.text_px(b + Vec2::new(6.0, 0.0), "copy: not vaccinated", s, pal::DIM, Align2::LEFT_CENTER);
            f.text_px(
                Vec2::new(lo.x, lo.y + 2.0),
                "each person copied into both; a copy is censored when the person's course leaves its strategy (schematic)",
                s,
                pal::FAINT,
                Align2::LEFT_TOP,
            );
        }
        Pic::Trial { blind } => {
            let start = Vec2::new(lo.x + 16.0, mid + 6.0);
            disc(f, start, 4.0, pal::TEXT);
            f.text_px(start + Vec2::new(0.0, 10.0), "by chance", s, pal::DIM, Align2::CENTER_TOP);
            let (a, b) = (Vec2::new(lo.x + w * 0.5, lo.y + 32.0), Vec2::new(lo.x + w * 0.5, lo.y + 76.0));
            hud_arrow(f, start, a, 1.6, CAUSE);
            hud_arrow(f, start, b, 1.6, pal::DIM);
            f.text_px(a + Vec2::new(6.0, 0.0), "recombinant vaccine", s, CAUSE, Align2::LEFT_CENTER);
            let other = if blind { "placebo; neither knows which" } else { "no vaccine; both know" };
            f.text_px(b + Vec2::new(6.0, 0.0), other, s, pal::DIM, Align2::LEFT_CENTER);
            f.text_px(Vec2::new(lo.x, lo.y + 2.0), "how it works (schematic)", s, pal::FAINT, Align2::LEFT_TOP);
        }
        Pic::Rayens => {
            // one study's two hazard ratios on one scale: the same unit, the same data
            let x = |v: f32| lo.x + 12.0 + (w - 24.0) * (v - 0.4) / 0.7;
            interval_rows(
                f,
                (lo, hi),
                s,
                &x,
                &[0.4, 0.6, 0.8, 1.0],
                1.0,
                &[(0.49, 0.46, 0.51, "0.49 against unvaccinated people"), (0.73, 0.67, 0.79, "0.73 against Tdap recipients")],
                |v| if v == 1.0 { "1: no difference".to_string() } else { format!("{v:.1}") },
            );
            f.text_px(Vec2::new(lo.x, lo.y + 2.0), "Rayens 2026: adjusted hazard ratios, one study, two comparison groups", s, pal::FAINT, Align2::LEFT_TOP);
        }
        Pic::Hamilton => {
            // one study's two differences, in percentage points
            let x = |v: f32| lo.x + 12.0 + (w - 24.0) * (v + 0.5) / 0.8;
            interval_rows(
                f,
                (lo, hi),
                s,
                &x,
                &[-0.4, -0.2, 0.0, 0.2],
                0.0,
                &[(-0.12, -0.153, -0.079, "shingles: −0.12 (−0.153 to −0.079)"), (-0.06, -0.40, 0.27, "dementia: −0.06 (−0.40 to 0.27)")],
                |v| if v == 0.0 { "0: no difference".to_string() } else { num(v as f64, 1) },
            );
            f.text_px(Vec2::new(lo.x, lo.y + 2.0), "Hamilton 2026: effects of eligibility, in percentage points", s, pal::FAINT, Align2::LEFT_TOP);
        }
        Pic::Kivimaki => {
            // one paper's implied effects, all in its own unit
            let x = |v: f32| lo.x + 12.0 + (w - 24.0) * (v + 10.0) / 330.0;
            let axis = lo.y + 92.0;
            hud_line(f, Vec2::new(x(-10.0), axis), Vec2::new(x(320.0), axis), 1.0, pal::DIM);
            for v in [0.0f32, 100.0, 200.0, 300.0] {
                f.text_px(Vec2::new(x(v), axis + 3.0), format!("{v:.0}%"), s, pal::FAINT, Align2::CENTER_TOP);
            }
            dashed_px(f, &[Vec2::new(x(100.0), lo.y + 18.0), Vec2::new(x(100.0), axis)], 1.2, NAIVE, 6.0, 3.0);
            f.text_px(Vec2::new(x(100.0) + 4.0, lo.y + 18.0), "100%: the most possible", s, NAIVE, Align2::LEFT_TOP);
            for (k, (v, a, b, name)) in [
                (18.0f32, 0.0, 35.0, "Wales 18%"),
                (84.0, 18.0, 219.0, "Canada 84%"),
                (111.0, 15.0, 303.0, "Canada 111%"),
                (177.0, 28.0, 301.0, "Australia 177%"),
            ]
            .into_iter()
            .enumerate()
            {
                let y = lo.y + 34.0 + k as f32 * 14.0;
                hud_line(f, Vec2::new(x(a), y), Vec2::new(x(b), y), 1.6, pal::WHITE.a(0.45));
                disc(f, Vec2::new(x(v), y), 3.0, pal::WHITE);
                f.text_px(Vec2::new(x(a) - 6.0, y), name, s, pal::TEXT, Align2::RIGHT_CENTER);
            }
            f.text_px(Vec2::new(lo.x, lo.y + 2.0), "Kivimäki 2026: implied reduction among the vaccinated", s, pal::FAINT, Align2::LEFT_TOP);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scene set up as experiment `k`, with the graph glided into place.
    fn run(k: usize) -> Shingles {
        let mut s = Shingles::new();
        s.set_up(k);
        for _ in 0..400 {
            s.glide(1.0 / 60.0);
        }
        s
    }

    fn carried(s: &Shingles, from: usize, to: usize) -> Carries {
        carries(s.choice, arrow_at(from, to).unwrap_or_else(|| panic!("no arrow {from} -> {to}")))
    }

    /// Every number written in `text` (digits, with their decimal points and thousands
    /// commas), leaving out digits that are part of a name (AS01, COVID-19).
    fn numbers_in(text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            if chars[i].is_ascii_digit() {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit() || ((chars[i] == '.' || chars[i] == ',') && chars.get(i + 1).is_some_and(|c| c.is_ascii_digit())))
                {
                    i += 1;
                }
                let glued_before =
                    start > 0 && (chars[start - 1].is_alphabetic() || (chars[start - 1] == '-' && start > 1 && chars[start - 2].is_alphabetic()));
                let glued_after = chars.get(i).is_some_and(|c| c.is_alphabetic());
                if !glued_before && !glued_after {
                    out.push(chars[start..i].iter().collect());
                }
            } else {
                i += 1;
            }
        }
        out
    }

    /// "Rayens 2026 … 0.49 against the unvaccinated and 0.73 against Tdap recipients:
    /// the association moves towards 1 …" with the back door faint pink.
    #[test]
    fn the_healthy_vaccinee() {
        let s = run(0);
        assert_eq!(s.choice, Choice { design: Some(COHORT), study: study("rayens"), story: Some(2) });
        assert_eq!(carried(&s, v::H, v::V), Carries::Weak);
        assert_eq!(carried(&s, v::H, v::D), Carries::Weak);
        assert_eq!(carried(&s, v::V, v::R), Carries::Effect);
        let rayens = fact("rayens").expect("Rayens");
        assert!(rayens.numbers.contains(&"0.49") && rayens.numbers.contains(&"0.73"));
        // the two ratios as the card writes them: nearer 1 against Tdap recipients
        let reported = STUDIES[study("rayens").unwrap_or(0)].blocks.iter().find(|(h, _)| *h == "reported").map_or("", |(_, b)| *b);
        let ratios: Vec<f64> = numbers_in(reported).iter().filter_map(|n| n.parse().ok()).collect();
        assert_eq!(ratios, [0.49, 0.46, 0.51, 0.73, 0.67, 0.79]);
        assert!(ratios[0] < ratios[3] && ratios[3] < 1.0);
        // the explanation lights the arrows of the back door
        assert!(STORIES[2].arrows.contains(&(v::H, v::V)) && STORIES[2].arrows.contains(&(v::H, v::D)));
        assert_eq!(CELLS[COHORT][2].mark, Mark::Cannot);
        // and all of them faint pink: closed only in part
        let look = Look::of(s.choice, s.layers);
        for (a, b) in [(v::H, v::V), (v::H, v::D), (v::H, v::K), (v::K, v::DX)] {
            assert_eq!(look.arrows[arrow_at(a, b).unwrap_or(0)].color, look_of(Carries::Weak).0, "{a} -> {b}");
        }
        // the second dose is held by this study, and opens a path
        let look = Look::of(s.choice, [false, true]);
        assert!(look.vars[v::N].held > 0.5 && !look.vars[v::N].partly);
        assert_eq!(carried(&s, v::V, v::N), Carries::Open);
    }

    /// "… the healthy-vaccinee arrows now lie on closed, grey paths. The design assumes
    /// that nothing else changes at that date (dashed). … 3.5 percentage points fewer
    /// (0.6–7.1)."
    #[test]
    fn a_birthday_decides() {
        let s = run(1);
        assert_eq!(s.choice.graph(), Some(BIRTH));
        for (a, b) in [(v::H, v::V), (v::H, v::D), (v::H, v::K)] {
            assert_eq!(carried(&s, a, b), Carries::Closed, "{a} -> {b}");
        }
        assert_eq!(carried(&s, v::E, v::V), Carries::Effect);
        assert_eq!(carried(&s, v::E, v::DX), Carries::Assumed);
        assert_eq!(carried(&s, v::E, v::K), Carries::Assumed);
        // nothing leads into eligibility
        assert!(!ARROWS.iter().any(|a| a.to == v::E));
        assert_eq!(CELLS[BIRTH][2].mark, Mark::Can);
        // the explanation's arrows are all drawn closed, even where another path is open
        let look = Look::of(s.choice, s.layers);
        for &(a, b) in STORIES[2].arrows.iter().filter(|(a, b)| VARS[*a].layer == Layer::Core && VARS[*b].layer == Layer::Core) {
            assert_eq!(look.arrows[arrow_at(a, b).unwrap_or(0)].color, CLOSED, "{a} -> {b}");
        }
        assert_eq!(carried(&s, v::K, v::DX), Carries::Open);
        let e = fact("eyting").expect("Eyting");
        for n in ["0.01", "47.2", "3.5", "0.6", "7.1", "1933", "2013"] {
            assert!(e.numbers.contains(&n), "{n}");
        }
    }

    /// "… −0.06 points (−0.40 to 0.27) … 18% in Wales, but 177% in Australia and 111%
    /// in one Canadian analysis, where more than 100% is impossible …"; and the
    /// division the Kivimäki card works through.
    #[test]
    fn same_design_another_answer() {
        let s = run(2);
        assert_eq!((s.choice.graph(), s.choice.study, s.choice.story), (Some(BIRTH), study("hamilton"), Some(7)));
        let h = fact("hamilton").expect("Hamilton");
        for n in ["0.06", "0.40", "0.27", "0.12", "6,340,214"] {
            assert!(h.numbers.contains(&n), "{n}");
        }
        // the interval of the dementia effect contains 0; that of shingles does not
        let reported = STUDIES[study("hamilton").unwrap_or(0)].blocks.iter().find(|(h, _)| *h == "reported").map_or("", |(_, b)| *b);
        let signed = |text: &str| -> Vec<f64> {
            text.replace('−', "-").split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-')).filter_map(|t| t.parse().ok()).collect()
        };
        let shingles = signed(reported.split(':').nth(1).unwrap_or(""));
        let dementia = signed(reported.split("Dementia:").nth(1).unwrap_or("").split(',').next().unwrap_or(""));
        assert_eq!((&shingles[..3], &dementia[..3]), (&[-0.12, -0.153, -0.079][..], &[-0.06, -0.40, 0.27][..]));
        assert!(shingles[2] < 0.0 && dementia[1] < 0.0 && dementia[2] > 0.0);
        let k = fact("kivimaki").expect("Kivimäki");
        let implied = [("18", false), ("177", true), ("111", true), ("84", false)];
        for (n, over) in implied {
            assert!(k.numbers.contains(&n));
            assert_eq!(n.parse::<f64>().unwrap_or(0.0) > 100.0, over, "{n}");
        }
        // Australia: 1.8 pp divided by 16.4 pp is 11.0 pp per person vaccinated because eligible
        assert_eq!(num(1.8 / 0.164, 1), "11.0");
        assert!(fact("pomirchy25").is_some_and(|f| f.numbers.contains(&"1.8") && f.numbers.contains(&"16.4")));
        // 6,340,214 is "6.3 million"
        assert_eq!(num(6_340_214.0 / 1e6, 1), "6.3");
    }

    /// "… compared people first vaccinated in the three years after … (95% recombinant)
    /// with those in the three years before (98% live) … 0.83 (0.79–0.87) … followed
    /// through different calendar years ($P \to F \to K$)."
    #[test]
    fn vaccine_against_vaccine() {
        let s = run(3);
        assert_eq!(s.choice.graph(), Some(SWITCH));
        let look = Look::of(s.choice, s.layers);
        assert!(look.vars[v::P].compared > 0.5 && look.vars[v::V].held > 0.5 && !look.vars[v::V].partly && look.vars[v::H].partly);
        for (a, b) in [(v::P, v::F), (v::F, v::K), (v::K, v::DX)] {
            assert_eq!(carried(&s, a, b), Carries::Open);
        }
        for (a, b) in [(v::P, v::T), (v::T, v::R), (v::T, v::I)] {
            assert_eq!(carried(&s, a, b), Carries::Effect);
        }
        // who came forward in each period: open through the held V, closed in part by matching
        assert_eq!(carried(&s, v::P, v::V), Carries::Weak);
        assert_eq!(CELLS[SWITCH][6].mark, Mark::Cannot);
        let t = fact("taquet24").expect("Taquet 2024");
        for n in ["95", "98", "0.83", "0.79", "0.87"] {
            assert!(t.numbers.contains(&n), "{n}");
        }
    }

    /// "… randomised, blinded and placebo-controlled, which cuts every arrow into $V$
    /// and keeps contact alike; … gives both routes at once …"
    #[test]
    fn two_ways_it_could_be_true() {
        let s = run(4);
        assert_eq!(s.choice.graph(), Some(BLIND_TRIAL));
        for (k, a) in ARROWS.iter().enumerate() {
            if (a.to == v::V || a.to == v::T) && !a.assumed {
                assert_eq!(carries(s.choice, k), Carries::Cut, "{} -> {}", a.from, a.to);
            }
        }
        assert_eq!(carried(&s, v::V, v::K), Carries::Cut);
        assert_eq!(carried(&s, v::T, v::R), Carries::Effect);
        assert_eq!(carried(&s, v::T, v::I), Carries::Effect);
        assert_eq!(CELLS[BLIND_TRIAL][1].mark, Mark::Cannot);
        assert_eq!(CELLS[OPEN_TRIAL][1].mark, Mark::Cannot);
        // in the open-label trial the visit stays open
        assert_eq!(carries(Choice { design: Some(OPEN_TRIAL), ..Choice::default() }, arrow_at(v::V, v::K).unwrap_or(0)), Carries::Open);
    }

    /// An open path is drawn pink only where it can be seen whole: in the blinded trial
    /// the only open path into the diagnosis runs through recorded shingles, so with
    /// that layer hidden its last arrow is drawn closed; shown, it is open again. And
    /// what the designs leave open in the core graph is drawn so with no layer shown.
    #[test]
    fn an_open_path_is_drawn_only_whole() {
        let k = arrow_at(v::K, v::DX).unwrap_or(0);
        let blind = Choice { design: Some(BLIND_TRIAL), ..Choice::default() };
        assert_eq!(carries(blind, k), Carries::Open);
        assert_eq!(drawn_carries(blind, [false, false])[k], Carries::Closed);
        assert_eq!(drawn_carries(blind, [false, true])[k], Carries::Open);
        for (d, a, b) in [
            (COHORT, v::H, v::V),
            (COHORT, v::V, v::K),
            (SWITCH, v::P, v::F),
            (SWITCH, v::P, v::V),
            (BIRTH, v::E, v::DX),
            (BIRTH, v::E, v::K),
            (OPEN_TRIAL, v::V, v::K),
        ] {
            let c = Choice { design: Some(d), ..Choice::default() };
            let k = arrow_at(a, b).unwrap_or(0);
            assert_eq!(drawn_carries(c, [false, false])[k], carries(c, k), "design {d}: {a} -> {b}");
        }
    }

    /// The arrows of the designs, explanations and studies are arrows of the graph; each
    /// design has its effect path from what it compares to the recorded diagnosis; an
    /// arrow is in at most one list of a design.
    #[test]
    fn the_graph_data_hang_together() {
        let exists = |p: &(usize, usize)| arrow_at(p.0, p.1).is_some();
        for (d, des) in DESIGNS.iter().enumerate() {
            let lists = [des.cut, des.effect, des.open, des.weak, des.assumes];
            for l in lists {
                assert!(l.iter().all(exists), "design {d}: {l:?}");
            }
            for (k, a) in ARROWS.iter().enumerate() {
                let n = lists.iter().filter(|l| l.contains(&(a.from, a.to))).count();
                assert!(n <= 1, "design {d}: arrow {k} in {n} lists");
                // an arrow assumed away is assumed only by the designs that list it
                if a.assumed {
                    assert_eq!(des.assumes.contains(&(a.from, a.to)), carries(Choice { design: Some(d), ..Choice::default() }, k) == Carries::Assumed);
                }
            }
            // a directed path of effect arrows from something compared to D*
            let mut reach: Vec<usize> = des.compared.to_vec();
            let mut grew = true;
            while grew {
                grew = false;
                for &(a, b) in des.effect {
                    if reach.contains(&a) && !reach.contains(&b) {
                        reach.push(b);
                        grew = true;
                    }
                }
            }
            assert!(reach.contains(&v::D) && reach.contains(&v::DX), "design {d} reaches {reach:?}");
        }
        for s in &STORIES {
            assert!(s.arrows.iter().all(exists) && s.gates.iter().all(exists), "{}", s.name);
            assert!(s.gates.iter().all(|g| s.arrows.contains(g)) || s.label == "5" || s.label == "4", "{}: a gate that is not its arrow", s.name);
        }
        for s in &STUDIES {
            assert!(s.open.iter().all(exists), "{}", s.key);
            assert!(s.design.is_none_or(|d| d < DESIGNS.len()) && s.group < GROUPS.len());
        }
        // every group has a study, and every design one at least
        for g in 0..GROUPS.len() {
            assert!(STUDIES.iter().any(|s| s.group == g), "group {g}");
        }
        for d in 0..DESIGNS.len() {
            assert!(STUDIES.iter().any(|s| s.design == Some(d)), "design {d}");
        }
        // assumed arrows are drawn only where a design needs them
        for (k, a) in ARROWS.iter().enumerate() {
            if a.assumed {
                assert!(!arrow_shown(Choice::default(), [true, true], k));
            }
        }
    }

    /// Every text the scene can show: its markup parses and no word is set in italic.
    fn every_text() -> Vec<String> {
        let mut texts: Vec<String> = Vec::new();
        for s in &STUDIES {
            texts.extend([s.short, s.by, s.place, s.meta, s.estimate, s.cite, s.disclosure].map(String::from));
            texts.extend(s.blocks.iter().flat_map(|(h, b)| [h.to_string(), b.to_string()]));
        }
        for d in &DESIGNS {
            texts.extend([d.name, d.short, d.title, d.gist, d.who, d.assumption, d.graph, d.checks, d.cannot, d.seen].map(String::from));
        }
        for s in &STORIES {
            texts.extend([s.name, s.title, s.gist, s.predicts, s.evidence].map(String::from));
        }
        for row in &CELLS {
            texts.extend(row.iter().map(|c| c.text.to_string()));
        }
        for v in &VARS {
            texts.extend([v.sym, v.name, v.short, v.note].map(String::from));
        }
        for a in &ARROWS {
            texts.push(a.note.to_string());
        }
        for t in &TRIALS {
            texts.extend([t.button, t.caption].map(String::from));
        }
        texts.extend(OPENING.iter().flat_map(|(h, b)| [h.to_string(), b.to_string()]));
        texts
    }

    #[test]
    fn every_text_is_typeset() {
        for text in every_text() {
            let found = crate::typeset::inspect(&text);
            assert!(found.error.is_none() && found.words.is_empty(), "{text}: {found:?}");
        }
    }

    /// Every number a study's texts show is one the dossier gives for it, typed again in
    /// `FACTS`; every number elsewhere is one the dossier gives somewhere; and every
    /// citation and DOI matches its second typing.
    #[test]
    fn every_number_shown_is_the_dossiers() {
        let mut missing: Vec<String> = Vec::new();
        for s in &STUDIES {
            let f = fact(s.key).unwrap_or_else(|| panic!("no facts for {}", s.key));
            assert_eq!((s.cite, s.doi), (f.cite, f.doi), "{}", s.key);
            let texts = [s.short, s.meta, s.estimate].into_iter().chain(s.blocks.iter().map(|(_, b)| *b));
            for text in texts {
                for n in numbers_in(text) {
                    if !f.numbers.contains(&n.as_str()) {
                        missing.push(format!("{}: {n} in \"{text}\"", s.key));
                    }
                }
            }
        }
        let shared = fact("shared").expect("shared numbers");
        let mut others: Vec<&str> = Vec::new();
        for d in &DESIGNS {
            others.extend([d.gist, d.who, d.assumption, d.graph, d.checks, d.cannot, d.seen]);
        }
        for s in &STORIES {
            others.extend([s.gist, s.predicts, s.evidence]);
        }
        for row in &CELLS {
            others.extend(row.iter().map(|c| c.text));
        }
        for a in &ARROWS {
            others.push(a.note);
        }
        for t in &TRIALS {
            others.push(t.caption);
        }
        others.extend(OPENING.iter().map(|(_, b)| *b));
        for text in others {
            for n in numbers_in(text) {
                if !shared.numbers.contains(&n.as_str()) {
                    missing.push(format!("shared: {n} in \"{text}\""));
                }
            }
        }
        assert!(missing.is_empty(), "numbers shown that FACTS does not hold:\n{}", missing.join("\n"));
        // and FACTS holds nothing twice for one source
        for f in FACTS {
            let mut seen: Vec<&str> = Vec::new();
            for n in f.numbers {
                assert!(!seen.contains(n), "{}: {n} twice", f.key);
                seen.push(n);
            }
        }
        // the tokens are read as written
        assert_eq!(numbers_in("282,541 people, 0.83 (0.79–0.87), AS01 and COVID-19, −3.1 pp"), ["282,541", "0.83", "0.79", "0.87", "3.1"]);
    }

    /// The numbers the scene works out from the dossier's, worked out again.
    #[test]
    fn the_arithmetic_holds() {
        // Supplementary Table 2 of Taquet 2024 (dossier §1.5): losses in years 1 and 2
        let (after, before) = (5_956.0 + 5_463.0, 3_304.0 + 3_072.0);
        assert_eq!((after, before), (11_419.0, 6_376.0));
        assert_eq!(num(after / before, 1), "1.8");
        // the dossier's T19: falls by 2 years, from the posted cumulative incidences
        assert_eq!(num(1.57 - 1.12, 2), "0.45");
        assert_eq!(num(1.15 - 0.72, 2), "0.43");
        // Supplementary Table 3: first doses a year roughly quadrupled
        let (early, late) = (40_568.0 + 36_484.0 + 33_175.0, 143_862.0 + 155_711.0 + 113_168.0);
        assert_eq!((early, late), (110_227.0, 412_741.0));
        assert!((3.5..4.5).contains(&(late / early)));
        // Australia's division (Kivimäki's method), as the card writes it
        assert_eq!(num(1.8 / 0.164, 1), "11.0");
    }

    /// An experiment shows the same thing whatever was done before it.
    #[test]
    fn an_experiment_sets_everything_it_depends_on() {
        for k in 0..TRIALS.len() {
            let fresh = run(k);
            let mut s = Shingles::new();
            s.choose(Choice { design: Some(OPEN_TRIAL), study: study("tang"), story: Some(5) });
            s.set_layer(0, true);
            s.set_layer(1, true);
            s.pane = Pane::Studies;
            s.pinned = Some(On::Var(3));
            s.set_up(k);
            for _ in 0..400 {
                s.glide(1.0 / 60.0);
            }
            assert_eq!((s.choice, s.layers, s.pane, s.pinned), (fresh.choice, fresh.layers, fresh.pane, fresh.pinned));
            assert_eq!(s.shown(), fresh.shown());
            assert!(!s.glide(1.0 / 60.0), "experiment {k} still moving");
        }
    }

    /// A change glides: most of the way in a quarter of a second, exactly there soon
    /// after, the same at 60 and at 144 frames a second.
    #[test]
    fn a_change_glides_into_place() {
        for hz in [60.0f32, 144.0] {
            let mut s = Shingles::new();
            s.set_up(3);
            let mut frames = 0;
            while s.glide(1.0 / hz) {
                frames += 1;
                assert!(frames as f32 / hz < 1.2, "still gliding after {frames} frames at {hz}");
            }
            assert!(frames as f32 / hz > 0.3, "{frames} frames at {hz}");
            assert_eq!(s.shown(), Look::of(s.choice, s.layers));
        }
    }

    /// Nothing chosen, every single choice, every pair: a card with a title and text,
    /// a look with every number finite.
    #[test]
    fn every_choice_has_a_card_and_a_finite_look() {
        let mut choices = vec![Choice::default()];
        for d in 0..DESIGNS.len() {
            for s in 0..=STORIES.len() {
                choices.push(Choice { design: Some(d), study: None, story: s.checked_sub(1) });
            }
        }
        for k in 0..STUDIES.len() {
            for s in [None, Some(0), Some(5)] {
                choices.push(Choice { design: STUDIES[k].design, study: Some(k), story: s });
            }
        }
        for s in 0..STORIES.len() {
            choices.push(Choice { story: Some(s), ..Choice::default() });
        }
        // and out of range, as a careless caller might ask
        choices.push(Choice { design: None, study: Some(999), story: None });
        for c in choices {
            let cd = card(c);
            assert!(!cd.title.is_empty() && !cd.blocks.is_empty(), "{c:?}");
            for layers in [[false, false], [true, true]] {
                let look = Look::of(c, layers);
                assert!(look.arrows.iter().all(|a| a.width.is_finite() && a.alpha.is_finite()) && look.vars.iter().all(|v| v.alpha.is_finite()));
            }
            let (title, body) = gist(c);
            assert!(!title.is_empty() && !body.is_empty());
        }
    }

    /// A source the dossier could not read says so, and says nothing more.
    #[test]
    fn an_unread_source_says_what_is_not_known() {
        for key in ["hamiltondaveysmith", "geldsetzer"] {
            let k = study(key).expect(key);
            let s = &STUDIES[k];
            assert_eq!(s.read, Read::Title);
            let c = card(Choice { study: Some(k), design: s.design, story: None });
            assert!(c.meta.contains("not verified"), "{}", c.meta);
            assert!(c.blocks.iter().all(|(h, _)| h != "reported"));
        }
        // and a number the dossier marks as not verified is said to be so where it is shown
        let kiv = &STUDIES[study("kivimaki").unwrap_or(0)];
        assert!(kiv.blocks.iter().any(|(_, b)| b.contains("national data") && b.contains("not verified")));
    }

    /// The parts of the picture lie inside the canvas and apart, on every canvas; the
    /// graph has room; and every card fits where it is shown on a desktop and a phone.
    #[test]
    fn the_parts_of_the_picture_keep_apart() {
        let views = [
            Vec2::new(1078.0, 1000.0),
            Vec2::new(1100.0, 700.0),
            Vec2::new(578.0, 700.0),
            Vec2::new(390.0, 754.0),
            Vec2::new(390.0, 330.0),
            Vec2::new(240.0, 180.0),
            Vec2::new(900.0, 640.0),
        ];
        for view in views {
            for pane in PANES.map(|p| p.0) {
                for chooser in [Chooser::Table, Chooser::Studies] {
                    let lay = Lay::of(view, if view.x < 700.0 { 46.0 } else { 0.0 }, pane, chooser);
                    let parts = lay.parts();
                    for (i, a) in parts.iter().enumerate() {
                        assert!(a.0.cmpge(Vec2::ZERO).all() && a.1.cmple(view + 0.5).all() && a.0.cmple(a.1).all(), "{view} {pane:?}: {a:?}");
                        for b in &parts[..i] {
                            assert!(a.1.x <= b.0.x || b.1.x <= a.0.x || a.1.y <= b.0.y || b.1.y <= a.0.y, "{view} {pane:?}: {a:?} and {b:?}");
                        }
                    }
                    if let Some(g) = lay.graph {
                        let size = g.1 - g.0;
                        assert!(size.x >= 100.0 && size.y >= 100.0, "{view} {pane:?}: graph {size}");
                    }
                }
            }
        }
        // the phone's graph is large enough to read
        let phone = Lay::of(Vec2::new(390.0, 754.0), 46.0, Pane::Graph, Chooser::Table);
        assert!(phone.graph.is_some_and(|g| g.1.y - g.0.y >= 330.0), "{phone:?}");
        // every card fits on a desktop, and on a phone's details view
        // (capitals are wider than the estimate allows for, other text narrower)
        let measure = |t: &str, s: f32| text_width(t, s) * if t.chars().any(|c| c.is_lowercase()) { 1.05 } else { 1.45 };
        let mut choices = vec![Choice::default()];
        choices.extend((0..DESIGNS.len()).map(|d| Choice { design: Some(d), ..Choice::default() }));
        choices.extend((0..STORIES.len()).map(|s| Choice { story: Some(s), ..Choice::default() }));
        for s in 0..=STORIES.len() {
            choices.extend((0..STUDIES.len()).map(|k| Choice { study: Some(k), design: STUDIES[k].design, story: s.checked_sub(1) }));
        }
        for d in 0..DESIGNS.len() {
            choices.extend((0..STORIES.len()).map(|s| Choice { design: Some(d), story: Some(s), study: None }));
        }
        for (view, inset, pane) in [(Vec2::new(1078.0, 1000.0), 0.0, Pane::Graph), (Vec2::new(390.0, 754.0), 46.0, Pane::Details)] {
            let lay = Lay::of(view, inset, pane, Chooser::Table);
            let (lo, hi) = lay.card.expect("a card");
            for c in &choices {
                let (_, h) = fit_card(&card(*c), hi.x - lo.x, hi.y - lo.y, lay.text, lay.small, &measure);
                assert!(h <= hi.y - lo.y, "{view}: {c:?} is {h} tall in {}", hi.y - lo.y);
            }
        }
    }

    /// Where the variables go: inside the graph's rectangle, never two on top of each
    /// other, every arrow long enough to be seen, on a desktop and a phone.
    #[test]
    fn the_variables_keep_apart() {
        for (view, inset) in [(Vec2::new(1078.0, 1000.0), 0.0), (Vec2::new(390.0, 754.0), 46.0), (Vec2::new(578.0, 700.0), 0.0)] {
            let lay = Lay::of(view, inset, Pane::Graph, Chooser::Table);
            let rect = lay.graph.expect("a graph");
            let at = places(rect, lay.radius, true);
            for (i, p) in at.iter().enumerate() {
                assert!(inside(*p, rect), "{view}: {i} at {p}");
                for q in &at[..i] {
                    assert!(p.distance(*q) > 2.0 * lay.radius + 10.0, "{view}: {i} too near another");
                }
            }
            for a in &ARROWS {
                let pts = arrow_points(at[a.from], at[a.to], a.bend, lay.radius);
                assert!(pts.len() >= 2 && pts.iter().all(|p| p.is_finite()), "{view}: {} -> {}", a.from, a.to);
                // an arrow passes no other disc
                for (i, p) in at.iter().enumerate() {
                    if i != a.from && i != a.to {
                        assert!(
                            path_distance(*p, &pts) > lay.radius + 2.0,
                            "{view}: {} -> {} passes over {}",
                            VARS[a.from].name,
                            VARS[a.to].name,
                            VARS[i].name
                        );
                    }
                }
            }
        }
        // two discs on top of each other, or a sliver of a window: no arrow, nothing infinite
        assert!(arrow_points(Vec2::ZERO, Vec2::ZERO, 0.3, 10.0).is_empty());
        assert!(arrow_points(Vec2::ZERO, Vec2::new(5.0, 0.0), 0.0, 10.0).is_empty());
        let tiny = places((Vec2::ZERO, Vec2::splat(1.0)), 9.0, false);
        assert!(tiny.iter().all(|p| p.is_finite()));
    }
}

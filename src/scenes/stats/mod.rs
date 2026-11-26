//! Probability and statistics: sampling, the normal distribution, fitting, MCMC.

use crate::clock::{Pace, Turn};
use crate::math::linalg::{poly_eval, poly_fit};
use crate::scenes::prelude::*;

pub mod plot;

// The scenes of this topic, in the order they appear in the catalogue: a course in
// four parts (the headings below). A scene not yet written holds its place here with a
// stand-in (`Unwritten`) and is listed nowhere a visitor looks (`scenes::UNWRITTEN`).
crate::scenes::topic!(
    // chance and distributions
    chance,
    distributions,
    moments,
    lln,
    clt,
    gauss2,
    // from data to estimates
    describe,
    estimators,
    likelihood,
    intervals,
    bootstrap,
    bayes,
    mcmc,
    // tests
    testing,
    pvalues,
    twogroups,
    anova,
    chisquare,
    // models
    correlation,
    regress,
    multireg,
    logistic,
    modelselect,
    entropy,
);

/// The four headings of the statistics course.
pub const CHANCE: &str = "Statistics: chance and distributions";
pub const ESTIMATION: &str = "Statistics: from data to estimates";
#[allow(dead_code)]
pub const TESTING: &str = "Statistics: tests";
pub const MODELS: &str = "Statistics: models";

// The course's colours, one meaning each in every scene of it. (A scene uses the ones
// it needs: hence the allowances.)
/// The truth: the distribution the data come from, a parameter's true value.
#[allow(dead_code)]
pub const TRUTH: Color = pal::GREEN;
/// The data: a sample, its points and its histogram.
#[allow(dead_code)]
pub const DATA: Color = pal::CYAN;
/// What is worked out from the data: an estimate, a fitted curve, a statistic.
#[allow(dead_code)]
pub const ESTIMATE: Color = pal::YELLOW;
/// What is assumed for the sake of argument: a null hypothesis, a prior.
#[allow(dead_code)]
pub const ASSUMED: Color = pal::PURPLE;
/// What is surprising or wrong: a tail beyond the statistic, an error, a miss.
#[allow(dead_code)]
pub const ALARM: Color = pal::PINK;
/// A second group or a second thing to compare the first with.
#[allow(dead_code)]
pub const OTHER: Color = pal::ORANGE;

/// A checkbox whose label may have mathematics in it (egui's own sets its label as
/// plain text).
fn checkbox(ui: &mut egui::Ui, on: &mut bool, label: &str) -> egui::Response {
    crate::typeset::seen(label);
    let text = crate::ui::label_text(ui, label, 12.5, None, false);
    ui.checkbox(on, text)
}

fn normal_pdf(x: f32, mu: f32, sd: f32) -> f32 {
    let z = (x - mu) / sd;
    (-0.5 * z * z).exp() / (sd * (TAU).sqrt())
}

/// This frame's share of work asked for at `rate` a second of scene time, which
/// follows the app's time speed, within the frame's allowance of real time (see
/// [`Pace`]): `dt` and `real_dt` are the frame's (`Frame::dt`, `Frame::real_dt`). Call
/// it only while scene time runs (`dt > 0`): a paused scene takes nothing.
fn scene_turn(pace: &mut Pace, rate: f32, dt: f32, real_dt: f32) -> Turn {
    pace.begin(rate, dt, real_dt, 0.004)
}

/// How far the scene's time has gone from the newest step a [`Pace`] took towards the
/// next: 0 just as that step was due, 1 when the next is. Whole steps come one in a
/// frame and none in the next, or one and then two; a picture of what one step does
/// is drawn this share of the way through it, so that it moves in every frame. (The
/// pace counts what it owes but does not say; this counts it again the same way:
/// what is owed grows with the scene's time, up to the same cap, and every step taken
/// pays one.)
#[derive(Clone, Copy, Debug, Default)]
struct Phase {
    owed: f32,
}

impl Phase {
    /// After a frame of `dt` seconds of the scene's time in which `taken` steps were
    /// taken at `rate` a second: the share of the way to the next step.
    fn after(&mut self, rate: f32, dt: f32, taken: usize) -> f32 {
        let owed = (self.owed + rate * dt.max(0.0)).min((rate * 0.1).max(2.0));
        self.owed = (owed - taken as f32).max(0.0);
        self.share()
    }

    /// The share now (1 if a step is owed and was not taken: the frame's time ran out).
    fn share(&self) -> f32 {
        self.owed.min(1.0)
    }

    /// Before the first step of a new run.
    fn reset(&mut self) {
        self.owed = 0.0;
    }
}

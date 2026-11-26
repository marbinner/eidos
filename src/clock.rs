//! A monotonic clock that also works in the browser, time budgets for work that must
//! not hold up a frame, and the switch that keeps headless renders reproducible.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;
use web_time::Instant;

static START: OnceLock<Instant> = OnceLock::new();
static REPRODUCIBLE: AtomicBool = AtomicBool::new(false);

/// Seconds since the program started.
pub fn now() -> f64 {
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// In reproducible mode (headless renders, benchmarks) nothing may depend on how
/// fast the machine is: budgets never run out, so the same input gives the same image.
pub fn set_reproducible(on: bool) {
    REPRODUCIBLE.store(on, Ordering::Relaxed);
}

pub fn reproducible() -> bool {
    REPRODUCIBLE.load(Ordering::Relaxed)
}

/// An allowance of time, started when it is created.
pub struct Budget {
    start: f64,
    seconds: f64,
}

impl Budget {
    pub fn new(seconds: f64) -> Self {
        Budget { start: now(), seconds }
    }

    /// Has the allowance been used up? (Never, in reproducible mode.)
    pub fn spent(&self) -> bool {
        !reproducible() && now() - self.start >= self.seconds
    }
}

/// The allowance for work that shares the frame with drawing (training, a
/// simulation): a share of the time between frames, so that a fast display keeps its
/// frame rate, and never more than `most` seconds.
pub fn frame_budget(frame_dt: f32, most: f64) -> Budget {
    Budget::new((0.35 * frame_dt as f64).clamp(0.001, most))
}

/// The rest after a piece of display work (recomputing a picture that is costly to
/// recompute): twice as long as the work took, so that repeated as often as this
/// allows it takes a third of the time at most, whatever the display's frame rate,
/// and a slow recomputation never sets the frame rate. Quick work rests for less
/// than a frame, which is no rest at all. (None in reproducible mode.)
#[derive(Clone, Copy, Debug, Default)]
pub struct Rest {
    until: f64,
}

impl Rest {
    /// Start resting after work that took `seconds`.
    pub fn after(seconds: f64) -> Self {
        Rest { until: now() + 2.0 * seconds.max(0.0) }
    }

    /// May the work be done again?
    pub fn over(&self) -> bool {
        reproducible() || self.over_at(now())
    }

    fn over_at(&self, time: f64) -> bool {
        time >= self.until
    }
}

/// Paces work that is asked for at a rate (training steps per second, say) and has
/// to share the frame with drawing: so many steps are owed per frame, at least one is
/// taken if any is owed, and no more are started once the frame's allowance of time
/// is spent (see [`frame_budget`]). What is owed and not taken is kept, up to a tenth
/// of a second's worth, so a slow frame is made up for and a long stall is not.
///
/// A single step can cost more than a frame's whole allowance (a large model). Then
/// "at least one" would let the step set the frame rate, so the pace remembers what a
/// step costs and, while that is more than the allowance, waits twice as long as each
/// step took before starting another: such work gets at most a third of the time, and
/// it is the work that slows down, not the frames in between.
///
/// ```ignore
/// let mut turn = self.pace.begin(rate, f.dt, f.real_dt, 0.004);
/// while turn.next() {
///     self.step();
/// }
/// let done = self.pace.end(turn);
/// ```
///
/// (A `Turn` borrows nothing, so the loop body is free to use the whole scene. In
/// reproducible mode the time allowance never runs out, nothing waits, and the count
/// alone decides.)
#[derive(Clone, Copy, Debug, Default)]
pub struct Pace {
    owed: f32,
    /// What one step takes, in seconds: a running average over the frames that took
    /// any (0 before the first).
    cost: f64,
    /// The allowance of the last frame that took a step.
    allowance: f64,
    /// While steps cost more than the allowance: no step starts before this time.
    after: f64,
}

/// One frame's share of the work: see [`Pace`].
pub struct Turn {
    owed: f32,
    budget: Budget,
    done: usize,
    /// A costly step was taken a moment ago: none now.
    waiting: bool,
}

impl Pace {
    /// Start a frame: `rate` steps for each second of the scene's time, of which `dt`
    /// has passed since the last frame (`Frame::dt`: none while the scene is paused,
    /// more or less than real time as the time-speed control says), and at most `most`
    /// seconds of work, out of the `real_dt` seconds a frame really lasts.
    pub fn begin(&mut self, rate: f32, dt: f32, real_dt: f32, most: f64) -> Turn {
        // (at least two: with a cap of one, at a slow rate the fraction of a step
        // left over after each step would be thrown away)
        let owed = (self.owed + rate * dt.max(0.0)).min((rate * 0.1).max(2.0));
        let budget = frame_budget(real_dt, most);
        Turn { owed, waiting: !reproducible() && budget.start < self.after, budget, done: 0 }
    }

    /// End the frame; returns how many steps were taken.
    pub fn end(&mut self, turn: Turn) -> usize {
        self.owed = turn.owed;
        if turn.done > 0 {
            let took = now() - turn.budget.start;
            let each = took / turn.done as f64;
            self.cost = if self.cost > 0.0 { self.cost + (each - self.cost) * 0.3 } else { each };
            self.allowance = turn.budget.seconds;
            if self.spacing_out() {
                self.after = now() + 2.0 * took;
            }
        }
        turn.done
    }

    /// Forget what is owed (after a pause or a restart).
    pub fn reset(&mut self) {
        self.owed = 0.0;
        self.after = 0.0;
    }

    /// Are steps being spaced out, because one costs more than a frame can spare?
    /// (Never in reproducible mode.)
    pub fn spacing_out(&self) -> bool {
        !reproducible() && self.cost > self.allowance
    }

    /// What one step takes, in seconds, as far as has been measured.
    pub fn step_cost(&self) -> f64 {
        self.cost
    }
}

impl Turn {
    /// May another step be taken? (Counts it if so.)
    #[allow(clippy::should_implement_trait)] // (it reads as a loop condition, not as an iterator)
    pub fn next(&mut self) -> bool {
        if self.owed < 1.0 || self.waiting || (self.done > 0 && self.budget.spent()) {
            return false;
        }
        self.owed -= 1.0;
        self.done += 1;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pace_takes_what_is_owed_and_keeps_the_change() {
        // (the steps here take no time, so the allowance of time never runs out and
        // only the count decides; no test may set the global reproducible flag,
        // which the tests of the background thread depend on)
        let mut pace = Pace::default();
        let mut steps = Vec::new();
        for _ in 0..6 {
            let mut turn = pace.begin(90.0, 1.0 / 60.0, 1.0 / 60.0, 0.004);
            while turn.next() {}
            steps.push(pace.end(turn));
        }
        // 1.5 steps a frame: one, two, one, two, ...
        assert_eq!(steps, [1, 2, 1, 2, 1, 2]);
        // a stall is not made up for beyond a tenth of a second's worth
        let mut turn = pace.begin(90.0, 30.0, 30.0, 0.004);
        while turn.next() {}
        assert_eq!(pace.end(turn), 9);
        // nothing is owed while paused (no time passes for it)
        let mut turn = pace.begin(90.0, 0.0, 0.0, 0.004);
        assert!(!turn.next());
        pace.end(turn);
        // and steps this cheap are never spaced out
        assert!(!pace.spacing_out());
    }

    #[test]
    fn a_step_too_long_for_a_frame_gets_a_third_of_the_time() {
        // 60 steps a second asked for, frames of 5 ms (a fast display), and a step
        // that takes 12 ms, three times the most a frame allows it: without spacing
        // out, every frame would take a step and training would take 70 % of the time
        let step = std::time::Duration::from_millis(12);
        let mut pace = Pace::default();
        let (start, mut working, mut steps) = (now(), 0.0, 0);
        while now() - start < 0.6 {
            let mut turn = pace.begin(60.0, 0.005, 0.005, 0.004);
            while turn.next() {
                let t = now();
                std::thread::sleep(step);
                working += now() - t;
            }
            steps += pace.end(turn);
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let share = working / (now() - start);
        assert!(pace.spacing_out() && pace.step_cost() >= 0.012, "cost {}", pace.step_cost());
        // a third, give or take a frame's rounding (a loaded machine stretches the
        // sleeps, and the waits with them)
        assert!(share > 0.2 && share < 0.37, "share of the time {share}, {steps} steps");
        // the frames went on meanwhile: a step every few of them, not every one
        assert!(steps >= 8, "{steps} steps");
    }

    /// Asked for a few steps a second, a pace takes that many at any frame rate: the
    /// part of a step still owed after each one is kept, not thrown away.
    #[test]
    fn a_slow_rate_is_kept_at_every_frame_rate() {
        for rate in [5.0f32, 7.0, 9.0, 10.0] {
            for hz in [30.0f32, 60.0, 120.0, 144.0, 240.0] {
                let mut pace = Pace::default();
                let mut steps = 0;
                for _ in 0..(60.0 * hz) as usize {
                    let mut turn = pace.begin(rate, 1.0 / hz, 1.0 / hz, 0.004);
                    while turn.next() {}
                    steps += pace.end(turn);
                }
                let achieved = steps as f32 / 60.0;
                assert!((achieved - rate).abs() < 0.05, "asked for {rate} steps a second at {hz} Hz, got {achieved}");
            }
        }
    }

    /// Display work repeated as often as its rest allows takes a third of the time at
    /// most on any display, and quick work is not held back at all.
    #[test]
    fn work_and_its_rest_share_the_time_one_to_two() {
        // A display that shows a frame every 1/hz seconds: the work starts with a
        // frame, the rest starts when the work ends, and the work is done again in
        // the first frame that begins after the rest is over.
        let share = |seconds: f64, hz: f64| {
            let frame = 1.0 / hz;
            let rest = Rest { until: seconds + 2.0 * seconds };
            let mut again = (seconds / frame).ceil().max(1.0) * frame;
            while !rest.over_at(again) {
                again += frame;
            }
            seconds / again
        };
        let displays = [30.0, 60.0, 120.0, 144.0, 240.0];
        for ms in [0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 20.0, 35.0, 60.0, 150.0] {
            for hz in displays {
                let s = share(ms / 1000.0, hz);
                assert!(s <= 1.0 / 3.0 + 1e-9, "{ms} ms of work at {hz} Hz takes {:.0} % of the time", s * 100.0);
            }
        }
        // 12 ms of work (waiting a number of frames gave it 7 % at 30 Hz and 41 % at 240 Hz)
        for hz in displays {
            let s = share(0.012, hz);
            assert!((0.17..=0.34).contains(&s), "12 ms of work at {hz} Hz takes {:.1} % of the time", s * 100.0);
        }
        // a millisecond's work is done again in the very next frame at 60 Hz
        assert!((share(0.001, 60.0) - 0.06).abs() < 1e-9);
        assert!(!Rest::after(10.0).over_at(now() + 1.0) && Rest::default().over_at(0.0));
    }
}

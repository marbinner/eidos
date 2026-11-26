//! Double pendulum chaos.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "pendulum",
    title: "Double pendulum chaos",
    group: "Dynamics",
    blurb: "Hundreds of nearly identical pendulums that do not stay identical.",
    make: |g| Box::new(DoublePendulum::new(g)),
};

/// State of one double pendulum: angles and angular velocities.
#[derive(Clone, Copy)]
struct Pend {
    a1: f64,
    a2: f64,
    w1: f64,
    w2: f64,
}

/// Experiments: (button, what to look for, first angle, second angle, angle difference
/// across the set, friction). Each also sets 600 pendulums, gravity 9.81 and equal masses.
const TRIALS: [(&str, &str, f32, f32, f32, f32); 4] = [
    ("released together", "The 600 differ by a thousandth of a radian in all. For about four and a half seconds they swing as one; then the trails fan out into a rainbow and the tips go their own ways.", 2.2, 2.6, 1e-3, 0.0),
    ("a thousand times closer", "The same with a difference a thousand times smaller. They stay together twice as long, about nine seconds, and then part all the same: a smaller difference only buys time.", 2.2, 2.6, 1e-6, 0.0),
    ("low and gentle", "Released from low down, with little energy. The motion is regular, not chaotic: after a minute and a half the tips are still within a hundredth of a rod of each other.", 0.5, 0.6, 1e-3, 0.0),
    ("with friction", "Friction drains the energy (watch it fall to −29.43, its value hanging at rest) before the small difference has had time to grow: the 600 never part, and after about 40 seconds all hang still.", 2.2, 2.6, 1e-3, 0.2),
];

/// The pendulums and their settings: everything but the drawing (the tests run it).
struct Swing {
    count: usize,
    angle1: f32,
    angle2: f32,
    spread: f32,
    gravity: f32,
    mass_ratio: f32,
    damping: f32,
    p: Vec<Pend>,
    /// Simulated seconds since the trails last got a row.
    since_row: f64,
}

impl Swing {
    fn new() -> Self {
        let mut s = Swing { count: 600, angle1: 2.2, angle2: 2.6, spread: 1e-3, gravity: 9.81, mass_ratio: 1.0, damping: 0.0, p: Vec::new(), since_row: 0.0 };
        s.reset();
        s
    }

    /// Release every pendulum again from the angles set.
    fn reset(&mut self) {
        let n = self.count.max(1);
        self.p = (0..n)
            .map(|i| {
                let off = if n > 1 { (i as f64 / (n - 1) as f64 - 0.5) * self.spread as f64 } else { 0.0 };
                Pend { a1: self.angle1 as f64, a2: self.angle2 as f64 + off, w1: 0.0, w2: 0.0 }
            })
            .collect();
        self.since_row = 0.0;
    }

    /// Set up experiment `k` of `TRIALS` and release.
    fn set_up(&mut self, k: usize) {
        let (_, _, angle1, angle2, spread, damping) = TRIALS[k];
        (self.count, self.angle1, self.angle2, self.spread, self.gravity, self.mass_ratio, self.damping) = (600, angle1, angle2, spread, 9.81, 1.0, damping);
        self.reset();
    }

    /// Equations of motion for unit rod lengths and masses m1 = 1, m2 = `mu`.
    fn accel(&self, s: Pend) -> (f64, f64) {
        let (g, mu, c) = (self.gravity as f64, self.mass_ratio as f64, self.damping as f64);
        let d = s.a1 - s.a2;
        let den = 1.0 + mu * d.sin() * d.sin();
        let a1 = (-g * (1.0 + mu) * s.a1.sin() + mu * g * s.a2.sin() * d.cos() - mu * d.sin() * (s.w2 * s.w2 + s.w1 * s.w1 * d.cos())) / den;
        let a2 = ((1.0 + mu) * (s.w1 * s.w1 * d.sin() - g * s.a2.sin() + g * s.a1.sin() * d.cos()) + mu * s.w2 * s.w2 * d.sin() * d.cos()) / den;
        (a1 - c * s.w1, a2 - c * s.w2)
    }

    fn rk4(&self, s: Pend, h: f64) -> Pend {
        let f = |q: Pend| {
            let (a1, a2) = self.accel(q);
            (q.w1, q.w2, a1, a2)
        };
        let add = |q: Pend, k: (f64, f64, f64, f64), h: f64| Pend { a1: q.a1 + k.0 * h, a2: q.a2 + k.1 * h, w1: q.w1 + k.2 * h, w2: q.w2 + k.3 * h };
        let k1 = f(s);
        let k2 = f(add(s, k1, h * 0.5));
        let k3 = f(add(s, k2, h * 0.5));
        let k4 = f(add(s, k3, h));
        let mix = (
            (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0) / 6.0,
            (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1) / 6.0,
            (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2) / 6.0,
            (k1.3 + 2.0 * k2.3 + 2.0 * k3.3 + k4.3) / 6.0,
        );
        add(s, mix, h)
    }

    /// Advance every pendulum by `h` seconds.
    fn step(&mut self, h: f64) {
        for i in 0..self.p.len() {
            self.p[i] = self.rk4(self.p[i], h);
        }
    }

    fn energy(&self, s: Pend) -> f64 {
        let (g, mu) = (self.gravity as f64, self.mass_ratio as f64);
        let kinetic = 0.5 * (1.0 + mu) * s.w1 * s.w1 + 0.5 * mu * s.w2 * s.w2 + mu * s.w1 * s.w2 * (s.a1 - s.a2).cos();
        let potential = -(1.0 + mu) * g * s.a1.cos() - mu * g * s.a2.cos();
        kinetic + potential
    }

    fn joints(s: Pend) -> (Vec2, Vec2) {
        let p1 = Vec2::new(s.a1.sin() as f32, -(s.a1.cos() as f32));
        let p2 = p1 + Vec2::new(s.a2.sin() as f32, -(s.a2.cos() as f32));
        (p1, p2)
    }

    /// The distance between the first and the last pendulum's tips, in rod lengths.
    fn separation(&self) -> f32 {
        match (self.p.first(), self.p.last()) {
            (Some(a), Some(b)) => (Self::joints(*a).1 - Self::joints(*b).1).length(),
            _ => 0.0,
        }
    }
}

/// The longest step that keeps the energy true. RK4's error in the energy grows like
/// the fifth power of the step; it is larger in stronger gravity (everything happens
/// faster, and a minute holds more swings) and much larger with a heavy lower bob,
/// which whips the light upper one about. Measured at the corners of the sliders
/// (`the_energy_stays_put_at_every_setting`), this keeps the change in a minute under
/// three ten-thousandths of the energy; at the default settings the step is as long
/// as it always was.
fn longest_step(gravity: f32, mass_ratio: f32) -> f64 {
    let (g, mu) = (gravity.max(0.1) as f64, mass_ratio.max(0.0) as f64);
    (0.067 * g.powf(-0.6) * (1.0 + mu).powf(-1.3)).min(0.0045)
}

/// The most steps one frame takes. A time allowance (see `frame`) normally stops it
/// sooner; where none applies (a reproducible render) this alone bounds the frame.
const MOST_STEPS: usize = 64;

/// How a frame of `dt` seconds is integrated in steps no longer than `longest`:
/// (number of steps, step size). The steps add up to `dt`, so the pendulums keep real
/// time at any frame rate and any setting of the time-speed control, up to
/// `MOST_STEPS` (a very slow frame, or a very high speed, then falls behind rather
/// than taking for ever).
fn substeps(dt: f32, longest: f64) -> (usize, f64) {
    let dt = dt.max(0.0) as f64;
    let steps = ((dt / longest).ceil() as usize).clamp(1, MOST_STEPS);
    (steps, (dt / steps as f64).min(longest))
}

/// The trails get a row every so much simulated time (two a frame at sixty frames a
/// second), whatever the length of the steps: counted in steps, a display that asks
/// for one short step a frame would never be given a row at all.
const ROW_EVERY: f64 = 1.0 / 120.0;

/// Whether a row is due after a step of `h`, given the time `since` the last one.
fn row_due(since: &mut f64, h: f64) -> bool {
    *since += h;
    // (a hair early, so that two steps of half the interval always make one row)
    let due = *since >= ROW_EVERY * 0.999;
    if due {
        // What is over counts towards the next row, so that rows keep their rate
        // however the steps fall. (No step is as long as a row, so this never owes
        // more than one.)
        *since = (*since - ROW_EVERY).min(ROW_EVERY);
    }
    due
}

/// Where pendulum `i` of `n` is in the colour map: blue for the first, red for the last.
fn tint(i: usize, n: usize) -> f32 {
    0.08 + 0.84 * i as f32 / n.max(2) as f32
}

pub struct DoublePendulum {
    swing: Swing,
    /// The number of pendulums the slider asks for; the set is made anew once the
    /// slider is let go.
    wanted: usize,
    /// The number of pendulums the trails were made for.
    built: usize,
    trails: TrailSet,
    length: u32,
    show_all_rods: bool,
}

impl DoublePendulum {
    pub fn new(gfx: &Gfx) -> Self {
        let swing = Swing::new();
        let n = swing.count;
        DoublePendulum { swing, wanted: n, built: n, trails: TrailSet::new(gfx, n as u32, 260, false), length: 260, show_all_rods: false }
    }

    /// Release again, with trails for as many pendulums as there now are.
    fn release(&mut self, gfx: &Gfx) {
        self.swing.reset();
        let n = self.swing.p.len();
        if n != self.built {
            self.trails = TrailSet::new(gfx, n as u32, self.length, false);
            self.built = n;
        } else {
            self.trails.clear();
        }
    }
}

impl Scene for DoublePendulum {
    fn view(&self) -> View {
        View::Flat { center: Vec2::new(0.0, -0.1), half_height: 2.55 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.06;
        s.axis_alpha = 0.0;
        s.bloom = s.bloom.max(0.7);
    }

    fn ui(&mut self, ui: &mut egui::Ui, gfx: &Gfx) {
        ui::note(ui, "A double pendulum is a pendulum hanging from the end of another. Here hundreds of them are released together, differing only by a hair in the second angle. They move as one, and then they do not: this is chaos.");
        let mut again = false;
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.swing.set_up(k);
                self.wanted = self.swing.count;
                again = true;
            }
        }

        ui::section(ui, "Release");
        let s = &mut self.swing;
        again |= ui::slider(ui, "first angle", &mut s.angle1, -PI..=PI);
        again |= ui::slider(ui, "second angle", &mut s.angle2, -PI..=PI);
        ui::note(ui, "Angles are in radians, measured from hanging straight down.");
        again |= ui::slider_log(ui, "angle difference across the set", &mut s.spread, 1e-7..=0.5);
        ui::slider_int_log(ui, "pendulums", &mut self.wanted, 1..=6000);
        // (new trails are a new buffer on the GPU: made once the slider is let go,
        // not on every frame of a drag)
        if self.wanted != self.swing.count && !ui.input(|i| i.pointer.any_down()) {
            self.swing.count = self.wanted;
            again = true;
        }
        ui::section(ui, "Physics");
        let s = &mut self.swing;
        again |= ui::slider(ui, "gravity", &mut s.gravity, 0.5..=25.0);
        again |= ui::slider(ui, "mass ratio $m_2/m_1$", &mut s.mass_ratio, 0.1..=5.0);
        ui::note(ui, "The lower bob's mass over the upper one's.");
        ui::slider(ui, "friction", &mut s.damping, 0.0..=0.5);
        ui.checkbox(&mut self.show_all_rods, "draw every rod");
        if ui.button("release again").clicked() || again {
            self.release(gfx);
        }

        ui::section(ui, "Reading the picture");
        let colours = Cmap::Turbo;
        ui::key(
            ui,
            &[
                (colours.color(tint(0, 2)), "blue: the tip of the first pendulum and its trail"),
                (colours.color(tint(2, 2)), "red: the last one; the colours between are the pendulums between"),
                (pal::WHITE, "white: the rods of the first, middle and last pendulum (or of all, with \"draw every rod\")"),
                (pal::DIM, "grey dot: the pivot they all hang from"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Chaos:", "motion that follows exact rules, with nothing left to chance, yet magnifies the smallest difference in the start until two futures have nothing in common.");
        if let Some(a) = self.swing.p.first() {
            ui::section(ui, "Numbers");
            ui::readout(ui, "energy of the first pendulum", format!("${}$", fmt(self.swing.energy(*a) as f32)));
            ui::readout(ui, "tip separation, first to last", format!("${}$", fmt(self.swing.separation())));
            ui::note(ui, "The separation is in rod lengths. Without friction the energy stays constant, however wild the motion; friction only takes it away.");
        }
        ui::see_also(ui, &[("attractor", "chaos in three equations, traced by thousands of particles"), ("lyapunov", "how fast a nudge grows, measured layer by layer in a network"), ("waves", "a whole membrane of oscillators")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let n = self.swing.p.len();
        let (sub, h) = substeps(f.dt, longest_step(self.swing.gravity, self.swing.mass_ratio));
        // (each pendulum keeps its colour: looked up, not worked out, for every row)
        let colours = Cmap::Turbo.table();
        let colour = |i: usize| colours[table_index(tint(i, n))];
        if f.dt > 0.0 {
            // as many steps as the frame has time for: past that the pendulums fall
            // behind real time, and the frame rate does not suffer
            let budget = crate::clock::frame_budget(f.real_dt, 0.006);
            for k in 0..sub {
                if k > 0 && budget.spent() {
                    break;
                }
                self.swing.step(h);
                if row_due(&mut self.swing.since_row, h) {
                    let alpha = 0.6;
                    self.trails.style.intensity = 1.25;
                    for (i, (point, s)) in self.trails.row_mut().iter_mut().zip(&self.swing.p).enumerate() {
                        let tip = Swing::joints(*s).1;
                        *point = LinePoint { pos: [tip.x, tip.y, 0.0], width: 1.5, color: colour(i).a(alpha) };
                    }
                    self.trails.commit(f.gfx);
                }
            }
        }
        f.trails(&mut self.trails);

        let rods: Vec<usize> = if self.show_all_rods { (0..n).collect() } else { vec![0, n / 2, n.saturating_sub(1)] };
        let fade = if self.show_all_rods { (8.0 / n as f32).clamp(0.03, 0.8) } else { 0.85 };
        for i in rods {
            let Some(&s) = self.swing.p.get(i) else { continue };
            let (p1, p2) = Swing::joints(s);
            // rod by rod: as one path, the first rod is not drawn whenever the second
            // folds back on it (the join in lines.wgsl cuts it away)
            f.line(Vec2::ZERO, p1, 2.2, pal::WHITE.a(fade));
            f.line(p1, p2, 2.2, pal::WHITE.a(fade));
            if !self.show_all_rods {
                f.dot(p1, 9.0, pal::WHITE);
                f.dot(p2, 11.0, colour(i));
            }
        }
        f.dot(Vec2::ZERO, 8.0, pal::DIM);
        bottom_caption(f, &["each colour: one pendulum's tip and its trail", "blue the first, red the last"], 12.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_is_integrated_for_as_long_as_it_lasts() {
        let default = longest_step(9.81, 1.0);
        // 60 Hz at the default settings: four steps, as there always were
        let (steps, h) = substeps(1.0 / 60.0, default);
        assert!(steps == 4 && (steps as f64 * h - 1.0 / 60.0).abs() < 1e-9);
        // 30 Hz, 120 Hz and time speed 4 at 60 Hz, and the same with the shortest
        // steps the sliders lead to: still the whole frame, in steps no longer than asked
        for longest in [default, longest_step(25.0, 5.0)] {
            for dt in [1.0f32 / 30.0, 1.0 / 120.0, 4.0 / 60.0] {
                let (steps, h) = substeps(dt, longest);
                let whole = (steps as f64 * h - dt as f64).abs() < 1e-7;
                // (except time speed 4 in the shortest steps, which falls a tenth behind)
                assert!((whole || steps == MOST_STEPS) && h <= longest + 1e-12, "dt {dt}: {steps} x {h}");
            }
        }
        // a frame of half a second is not chased
        let (steps, h) = substeps(0.5, default);
        assert!(steps == MOST_STEPS && h <= default + 1e-12);
        assert_eq!(substeps(0.0, default).1, 0.0);
        assert!(substeps(f32::NAN, default).1 == 0.0);
    }

    /// The trails grow at the same rate in simulated time on any display and at any
    /// time speed, to within a row or two: also where a frame is a single short step (a
    /// 240 Hz display, or the time-speed control turned down), which once left them
    /// standing still, and where the steps do not divide a row evenly (50 or 144 Hz, a
    /// display a little off 60 Hz, frames that jitter), which once lost a third of them.
    #[test]
    fn trails_get_their_rows_at_any_frame_rate() {
        let mut rng = Rng::new(3);
        for longest in [longest_step(9.81, 1.0), longest_step(25.0, 5.0), longest_step(0.5, 0.1)] {
            for hz in [30.0f32, 50.0, 59.94, 60.0, 60.1, 72.0, 75.0, 90.0, 100.0, 120.0, 144.0, 165.0, 240.0, 15.0, 960.0] {
                for jitter in [0.0f32, 0.01] {
                    let (mut since, mut rows, mut time) = (0.0, 0, 0.0);
                    for _ in 0..3000 {
                        let dt = (1.0 + jitter * rng.range(-1.0, 1.0)) / hz;
                        let (steps, h) = substeps(dt, longest);
                        for _ in 0..steps {
                            rows += row_due(&mut since, h) as u32;
                            time += h;
                        }
                    }
                    let expected = time / ROW_EVERY;
                    assert!((rows as f64 - expected).abs() <= 2.0, "{hz} Hz (jitter {jitter}), steps up to {longest}: {rows} rows where {expected:.1} are due");
                }
            }
        }
        // sixty frames a second: exactly two rows a frame, as before
        let (mut since, mut rows) = (0.0, 0);
        for _ in 0..100 {
            let (steps, h) = substeps(1.0 / 60.0, longest_step(9.81, 1.0));
            rows += (0..steps).filter(|_| row_due(&mut since, h)).count();
        }
        assert_eq!(rows, 200);
    }

    /// Run one pendulum released from `angles` for `seconds` as the scene does at 60
    /// frames a second: the largest change in its energy relative to the start.
    fn energy_drift(gravity: f32, mass_ratio: f32, angles: (f32, f32), seconds: f32) -> f64 {
        let mut s = Swing::new();
        (s.count, s.gravity, s.mass_ratio, s.angle1, s.angle2) = (1, gravity, mass_ratio, angles.0, angles.1);
        s.reset();
        let e0 = s.energy(s.p[0]);
        let (steps, h) = substeps(1.0 / 60.0, longest_step(gravity, mass_ratio));
        let mut worst = 0.0f64;
        for _ in 0..(seconds * 60.0) as usize {
            for _ in 0..steps {
                s.step(h);
            }
            worst = worst.max((s.energy(s.p[0]) - e0).abs());
        }
        worst / e0.abs()
    }

    /// Without friction the energy stays put (the panel says so) at every setting of
    /// the sliders, not only the default ones: over a minute it changes by less than
    /// three ten-thousandths of itself. (A step of 1/240 s at every setting lost up to a
    /// fifth in a minute with strong gravity and a heavy lower bob.)
    #[test]
    fn the_energy_stays_put_at_every_setting() {
        for gravity in [0.5f32, 9.81, 25.0] {
            for mass_ratio in [0.1f32, 1.0, 5.0] {
                for angles in [(2.2, 2.6), (3.0, 3.1), (3.1, -3.1), (PI, 0.0), (1.5, 1.5)] {
                    let drift = energy_drift(gravity, mass_ratio, angles, 60.0);
                    assert!(drift < 3e-4, "gravity {gravity}, mass ratio {mass_ratio}, angles {angles:?}: the energy changed by {drift:.2e} of itself");
                }
            }
        }
    }

    /// Experiment `k` as its button sets it up, run for `seconds` at sixty frames a
    /// second; `look` sees it after every frame. Each pendulum moves on its own, so
    /// every tenth of them and the last (the separation is first to last) are all that
    /// is kept: the same motion in a tenth of the time.
    fn run_trial(k: usize, seconds: f32, mut look: impl FnMut(f32, &Swing)) {
        let mut s = Swing::new();
        s.set_up(k);
        let last = *s.p.last().unwrap();
        s.p = s.p.iter().step_by(10).copied().chain([last]).collect();
        let (steps, h) = substeps(1.0 / 60.0, longest_step(s.gravity, s.mass_ratio));
        for frame in 1..=(seconds * 60.0) as usize {
            for _ in 0..steps {
                s.step(h);
            }
            look(frame as f32 / 60.0, &s);
        }
    }

    /// When the first and last tips are first more than half a rod apart (within
    /// `seconds`), and the furthest apart they get.
    fn parting(k: usize, seconds: f32) -> (Option<f32>, f32) {
        let (mut when, mut widest) = (None, 0.0f32);
        run_trial(k, seconds, |t, s| {
            let d = s.separation();
            widest = widest.max(d);
            if when.is_none() && d > 0.5 {
                when = Some(t);
            }
        });
        (when, widest)
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // released together: as one for about four and a half seconds, then apart
        let (Some(together), widest) = parting(0, 20.0) else { panic!("released together, they never parted") };
        assert!((4.2..4.9).contains(&together) && widest > 3.0, "released together: parted after {together} s, at most {widest} apart");
        // a thousand times closer: twice as long, about nine seconds
        let (Some(closer), widest) = parting(1, 20.0) else { panic!("a thousand times closer, they never parted") };
        assert!((8.4..9.4).contains(&closer) && (1.8..2.2).contains(&(closer / together)) && widest > 3.0, "a thousand times closer: parted after {closer} s");
        // low and gentle: within a hundredth of a rod after a minute and a half
        let (_, widest) = parting(2, 90.0);
        assert!(widest < 0.01, "low and gentle: {widest} apart");
        // with friction: the energy only falls, to its value at rest; the pendulums
        // never part, and after about forty seconds every one hangs still
        let rest = -(1.0 + 2.0 * 1.0) * 9.81;
        let (mut energy, mut rose, mut widest, mut still_from) = (f64::MAX, false, 0.0f32, None);
        run_trial(3, 60.0, |t, s| {
            let e = s.energy(s.p[0]);
            rose |= e > energy + 1e-6;
            energy = e;
            widest = widest.max(s.separation());
            let furthest = s.p.iter().map(|p| (Swing::joints(*p).1 - Vec2::new(0.0, -2.0)).length()).fold(0.0f32, f32::max);
            if furthest >= 0.05 {
                still_from = None;
            } else if still_from.is_none() {
                still_from = Some(t);
            }
        });
        assert!(!rose && (energy - rest).abs() < 0.01, "with friction: the energy ends at {energy}, and rose on the way: {rose}");
        assert!(widest < 0.08, "with friction: {widest} apart");
        assert!(still_from.is_some_and(|t| (38.0..46.0).contains(&t)), "with friction: all hang still from {still_from:?} s");
    }

    /// The experiments' numbers (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for hz in [60.0f32, 144.0, 30.0] {
            for k in 0..TRIALS.len() {
                for damping in [None, Some(0.05f32), Some(0.1), Some(0.15)] {
                    if damping.is_some() && k != 3 {
                        continue;
                    }
                    let mut s = Swing::new();
                    s.set_up(k);
                    if let Some(c) = damping {
                        s.damping = c;
                    }
                    let e0 = s.energy(s.p[0]);
                    let (steps, h) = substeps(1.0 / hz, longest_step(s.gravity, s.mass_ratio));
                    let (mut parted, mut settled, mut widest) = (None, None, 0.0f32);
                    let mut energies = Vec::new();
                    let rest = Vec2::new(0.0, -2.0);
                    for frame in 1..=(90.0 * hz) as usize {
                        for _ in 0..steps {
                            s.step(h);
                        }
                        let t = frame as f32 / hz;
                        let sep = s.separation();
                        widest = widest.max(sep);
                        if parted.is_none() && sep > 0.5 {
                            parted = Some(t);
                        }
                        let far = s.p.iter().map(|p| (Swing::joints(*p).1 - rest).length()).fold(0.0f32, f32::max);
                        if far < 0.05 && settled.is_none() {
                            settled = Some(t);
                        }
                        if far >= 0.05 {
                            settled = None;
                        }
                        if frame % (10.0 * hz) as usize == 0 {
                            energies.push(format!("{:.2}", s.energy(s.p[0])));
                        }
                    }
                    println!("{hz} Hz {:<24} friction {:.2}: parted {parted:?}, widest {widest:.4}, all at rest from {settled:?}, energy {e0:.2} then {}", TRIALS[k].0, s.damping, energies.join(" "));
                }
            }
        }
        for gravity in [0.5f32, 9.81, 25.0] {
            for mass_ratio in [0.1f32, 1.0, 2.0, 5.0] {
                let worst = [(2.2, 2.6), (3.0, 3.1), (3.1, -3.1), (PI, 0.0), (1.5, 1.5), (0.0, PI)].map(|a| energy_drift(gravity, mass_ratio, a, 60.0)).into_iter().fold(0.0, f64::max);
                println!("gravity {gravity:5.2} mass ratio {mass_ratio:3.1}: step {:.5}, worst change in a minute {worst:.2e}", longest_step(gravity, mass_ratio));
            }
        }
    }
}

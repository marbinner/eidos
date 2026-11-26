//! Strange attractors.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "attractor",
    title: "Strange attractors",
    group: "Dynamics",
    blurb: "Lorenz and friends, traced by thousands of glowing particles.",
    make: |g| Box::new(Attractor::new(g)),
};

struct System {
    name: &'static str,
    /// Parameter names, default values and slider ranges.
    params: &'static [(&'static str, f32, f32, f32)],
    /// One step for every particle, with the system's equations compiled into the loop.
    step: fn(&mut Cloud, f32),
    /// The same three equations, written out for the panel (markup, for a display).
    equations: &'static str,
    /// Centre and scale that bring the attractor to roughly unit size.
    center: Vec3,
    scale: f32,
    start: Vec3,
    dt: f32,
}

const SYSTEMS: [System; 6] = [
    System {
        name: "Lorenz",
        params: &[(r"$\sigma$", 10.0, 0.0, 30.0), (r"$\rho$", 28.0, 0.0, 60.0), (r"$\beta$", 2.667, 0.1, 6.0)],
        step: |c, h| c.step_with(lorenz, h),
        equations: r"\begin{cases} \dot x = \sigma (y - x) \\ \dot y = x (\rho - z) - y \\ \dot z = x y - \beta z \end{cases}",
        center: Vec3::new(0.0, 0.0, 25.0),
        scale: 0.085,
        start: Vec3::new(1.0, 1.0, 20.0),
        dt: 0.006,
    },
    System {
        name: "Rössler",
        params: &[("$a$", 0.2, 0.0, 0.5), ("$b$", 0.2, 0.0, 2.0), ("$c$", 5.7, 1.0, 12.0)],
        step: |c, h| c.step_with(rossler, h),
        equations: r"\begin{cases} \dot x = -y - z \\ \dot y = x + a y \\ \dot z = b + z (x - c) \end{cases}",
        center: Vec3::new(0.0, 0.0, 6.0),
        scale: 0.17,
        start: Vec3::new(4.0, 0.0, 0.5),
        dt: 0.03,
    },
    System {
        name: "Aizawa",
        params: &[("$a$", 0.95, 0.5, 1.2), ("$b$", 0.7, 0.3, 1.0), ("$c$", 0.6, 0.2, 1.0)],
        step: |c, h| c.step_with(aizawa, h),
        equations: r"\begin{cases} \dot x = (z - b) x - 3.5 y \\ \dot y = 3.5 x + (z - b) y \\ \dot z = c + a z - z^3/3 \\ \qquad {} - (x^2 + y^2)(1 + 0.25 z) \\ \qquad {} + 0.1 z x^3 \end{cases}",
        center: Vec3::new(0.0, 0.0, 0.6),
        scale: 1.55,
        start: Vec3::new(0.1, 0.0, 0.0),
        dt: 0.012,
    },
    System {
        name: "Thomas",
        params: &[("$b$", 0.208, 0.05, 0.4), ("", 0.0, 0.0, 0.0), ("", 0.0, 0.0, 0.0)],
        step: |c, h| c.step_with(thomas, h),
        equations: r"\begin{cases} \dot x = \sin y - b x \\ \dot y = \sin z - b y \\ \dot z = \sin x - b z \end{cases}",
        center: Vec3::ZERO,
        scale: 0.5,
        start: Vec3::new(1.1, 1.1, -0.01),
        dt: 0.06,
    },
    System {
        name: "Halvorsen",
        params: &[("$a$", 1.4, 1.0, 2.2), ("", 0.0, 0.0, 0.0), ("", 0.0, 0.0, 0.0)],
        step: |c, h| c.step_with(halvorsen, h),
        equations: r"\begin{cases} \dot x = -a x - 4 y - 4 z - y^2 \\ \dot y = -a y - 4 z - 4 x - z^2 \\ \dot z = -a z - 4 x - 4 y - x^2 \end{cases}",
        center: Vec3::new(-2.5, -2.5, -2.5),
        scale: 0.2,
        start: Vec3::new(-5.0, 0.0, 0.0),
        dt: 0.006,
    },
    System {
        name: "Chen",
        params: &[("$a$", 35.0, 30.0, 45.0), ("$b$", 3.0, 1.0, 6.0), ("$c$", 28.0, 20.0, 30.0)],
        step: |c, h| c.step_with(chen, h),
        equations: r"\begin{cases} \dot x = a (y - x) \\ \dot y = (c - a) x - x z + c y \\ \dot z = x y - b z \end{cases}",
        center: Vec3::new(0.0, 0.0, 22.0),
        scale: 0.085,
        start: Vec3::new(-3.0, 2.0, 20.0),
        dt: 0.0025,
    },
];

// The systems' equations. Each is compiled into a loop of its own (`System::step`):
// called through a pointer, four times a step for every particle, they made a step
// three times as slow.

fn lorenz(p: Vec3, k: &[f32; 3]) -> Vec3 {
    Vec3::new(k[0] * (p.y - p.x), p.x * (k[1] - p.z) - p.y, p.x * p.y - k[2] * p.z)
}

fn rossler(p: Vec3, k: &[f32; 3]) -> Vec3 {
    Vec3::new(-p.y - p.z, p.x + k[0] * p.y, k[1] + p.z * (p.x - k[2]))
}

fn aizawa(p: Vec3, k: &[f32; 3]) -> Vec3 {
    let (d, e, ff) = (3.5, 0.25, 0.1);
    Vec3::new(
        (p.z - k[1]) * p.x - d * p.y,
        d * p.x + (p.z - k[1]) * p.y,
        k[2] + k[0] * p.z - p.z * p.z * p.z / 3.0 - (p.x * p.x + p.y * p.y) * (1.0 + e * p.z) + ff * p.z * p.x * p.x * p.x,
    )
}

fn thomas(p: Vec3, k: &[f32; 3]) -> Vec3 {
    Vec3::new(p.y.sin() - k[0] * p.x, p.z.sin() - k[0] * p.y, p.x.sin() - k[0] * p.z)
}

fn halvorsen(p: Vec3, k: &[f32; 3]) -> Vec3 {
    Vec3::new(
        -k[0] * p.x - 4.0 * p.y - 4.0 * p.z - p.y * p.y,
        -k[0] * p.y - 4.0 * p.z - 4.0 * p.x - p.z * p.z,
        -k[0] * p.z - 4.0 * p.x - 4.0 * p.y - p.x * p.x,
    )
}

fn chen(p: Vec3, k: &[f32; 3]) -> Vec3 {
    Vec3::new(k[0] * (p.y - p.x), (k[2] - k[0]) * p.x - p.x * p.z + k[2] * p.y, p.x * p.y - k[1] * p.z)
}

/// Experiments: (button, what to look for, ρ, particles, starting cloud size, trail
/// length, colour by: 0 speed, 1 particle). Each is the Lorenz system with σ = 10 and
/// β = 2.667, run at speed 1.
const TRIALS: [(&str, &str, f32, usize, f32, u32, usize); 3] = [
    ("two starts a hair apart", "Two particles in two colours, started closer together than any screen could show. For ten seconds or so they trace a single path; then they part, and each goes its own way round the same butterfly, changing wings at moments of its own.", 28.0, 2, 1e-5, 600, 1),
    ("a speck of 2,400", "All 2,400 particles start in a speck too small to see. For about ten seconds it travels as one bright streak; then it is drawn out into a thread along the wings, and by 45 seconds it is spread over the whole butterfly, half on each wing.", 28.0, 2400, 1e-5, 160, 0),
    (r"$\rho = 10$: no chaos", r"With $\rho$ turned down to $10$ there is no butterfly. Every particle spirals into one of two resting points, the centres of the wings, roughly half to each, and within about fifteen seconds all of them have come to rest.", 10.0, 2400, 0.6, 160, 0),
];

pub struct Attractor {
    cloud: Cloud,
    /// The particles and trail length the sliders ask for: the trails are made anew
    /// for them once the slider is let go.
    want_count: usize,
    want_length: u32,
    length: u32,
    color_by: usize,
    cmap: usize,
    trails: TrailSet,
    spin: bool,
}

const ATTRACTOR_CMAPS: [Cmap; 5] = [Cmap::Plasma, Cmap::Aurora, Cmap::Turbo, Cmap::Twilight, Cmap::IceFire];
/// What the two ends of the part of each map the trails use look like, in words (for
/// the caption on the picture).
const ATTRACTOR_CMAP_ENDS: [(&str, &str); 5] = [("purple", "yellow"), ("dark blue", "green"), ("dark blue", "red"), ("pale", "red"), ("light blue", "orange")];

/// The most line segments the trails may have between them (particles times trail
/// length): about as many as a frame can draw at a comfortable rate. (The default
/// has 384,000; the buffer they live in would hold six times this many.)
const MAX_SEGMENTS: usize = 2_000_000;

/// The longest trail that many particles can have.
fn longest_trail(particles: usize) -> u32 {
    (MAX_SEGMENTS / particles.max(1)).clamp(8, 600) as u32
}

/// The most steps one frame takes (speed 4 at fifteen frames a second). A time
/// allowance normally stops it sooner (see `frame`); where none applies (a
/// reproducible render) this alone bounds the frame.
const MOST_STEPS: usize = 32;

/// How far to advance this frame: (steps, step size). At speed 1 on a 60 Hz display
/// that is two steps of the system's own `dt`; other speeds and frame times scale the
/// total. No step is longer than `dt`, so that no step owes the trails more than one
/// row (see [`row_due`]), and no frame takes more than `MOST_STEPS`: a very slow frame
/// falls behind rather than taking for ever. Nothing moves at speed zero.
fn advance(system_dt: f32, speed: f32, frame_dt: f32) -> (usize, f32) {
    let total = speed * 2.0 * (frame_dt * 60.0);
    if total.is_nan() || total <= 0.0 {
        return (0, 0.0);
    }
    // (a hair under, so that exactly two steps' worth is two steps and not three)
    let steps = ((total - 1e-3).ceil() as usize).clamp(1, MOST_STEPS);
    (steps, system_dt * (total / steps as f32).min(1.0))
}

/// Whether the trails are owed a row after a step of `h` (in steps of the system's
/// own `dt`), given the system's time `since` the last one. A row for every `dt`,
/// whatever the display and the speed, so that a trail of a given length always
/// covers the same stretch of the system's time. (Counted in steps, a 240 Hz display
/// made trails half as long as a 60 Hz one.)
fn row_due(since: &mut f32, h: f32) -> bool {
    *since += h;
    // (a hair early, so that steps of exactly `dt` make a row each)
    let due = *since >= 0.999;
    if due {
        // what is over counts towards the next row (no step is longer than a row, so
        // this never owes more than one)
        *since = (*since - 1.0).min(1.0);
    }
    due
}

/// One fourth-order Runge-Kutta step from `p`: the new point and the mean velocity
/// over the step.
#[inline(always)]
fn rk4(deriv: impl Fn(Vec3, &[f32; 3]) -> Vec3, k: &[f32; 3], p: Vec3, h: f32) -> (Vec3, Vec3) {
    let k1 = deriv(p, k);
    let k2 = deriv(p + k1 * (h * 0.5), k);
    let k3 = deriv(p + k2 * (h * 0.5), k);
    let k4 = deriv(p + k3 * h, k);
    let d = (k1 + k2 * 2.0 + k3 * 2.0 + k4) / 6.0;
    (p + d * h, d)
}

/// The speed to colour as fastest, from a sample of the particles' speeds: the one
/// that two in a hundred exceed. Not the very fastest: a particle on its way out to
/// infinity is thousands of times faster than anything on the attractor, and one of
/// those would turn every other trail the colour of "slow".
fn typical_top_speed(sample: &mut [f32]) -> Option<f32> {
    if sample.is_empty() {
        return None;
    }
    let k = ((sample.len() as f32 * 0.98) as usize).min(sample.len() - 1);
    Some(*sample.select_nth_unstable_by(k, f32::total_cmp).1)
}

/// Where `t` (0 to 1) falls in the part of the colour map the trails use: they stay
/// off its pale end, which under additive blending washes out.
fn shade(t: f32) -> usize {
    table_index(if t.is_finite() { 0.06 + 0.8 * t.clamp(0.0, 1.0) } else { 0.06 })
}

/// The shade for a speed `x` times the top one: `shade(x^0.8)` (which spreads the
/// slower speeds a little), looked up in 4096 steps of `x` rather than worked out
/// with a power for every particle and row, which cost as much as the step itself.
fn speed_shade(x: f32) -> usize {
    static TABLE: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| (0..4096).map(|j| shade((j as f32 / 4095.0).powf(0.8)) as u8).collect());
    // (NaN counts as the fastest, as `min` makes it)
    table[((x.min(1.0).max(0.0) * 4095.0 + 0.5) as usize).min(4095)] as usize
}

/// Has a particle left for good (some systems throw off part of a wide starting cloud)?
fn escaped(p: Vec3) -> bool {
    !p.is_finite() || p.length() > 1e4
}

/// The particles and the system they follow: everything but the drawing (the tests
/// run it).
struct Cloud {
    sys: usize,
    k: [f32; 3],
    spread: f32,
    speed: f32,
    pos: Vec<Vec3>,
    /// Each particle's speed over its last step.
    vel: Vec<f32>,
    /// Particles that left since the trails last got a row and were put back in the
    /// starting cloud: their next row cuts the trail (a negative width), so that no
    /// streak joins the old place to the new.
    cut: Vec<bool>,
    rng: Rng,
    /// The system's time since the trails last got a row, in steps of its own `dt`.
    since_row: f32,
    /// A few hundred of the speeds of this frame's steps (see [`typical_top_speed`]).
    speeds: Vec<f32>,
    /// The speed that "by speed" colours as fastest.
    vmax: f32,
}

impl Cloud {
    fn new(count: usize) -> Self {
        let mut c = Cloud { sys: 0, k: [0.0; 3], spread: 0.6, speed: 1.0, pos: vec![Vec3::ZERO; count], vel: Vec::new(), cut: Vec::new(), rng: Rng::new(5), since_row: 0.0, speeds: Vec::new(), vmax: 1.0 };
        c.load_system();
        c
    }

    fn system(&self) -> &'static System {
        &SYSTEMS[self.sys]
    }

    /// The chosen system with its own parameters, and every particle in its starting cloud.
    fn load_system(&mut self) {
        let s = self.system();
        for i in 0..3 {
            self.k[i] = s.params[i].1;
        }
        // (each system has its own range of speeds)
        self.vmax = 1.0;
        self.reset();
    }

    /// A starting point: somewhere in the starting cloud.
    fn spawn(&mut self) -> Vec3 {
        let s = self.system();
        s.start + Vec3::new(self.rng.normal(), self.rng.normal(), self.rng.normal()) * (self.spread / s.scale)
    }

    /// Every particle back to the starting cloud.
    fn reset(&mut self) {
        let n = self.pos.len();
        self.pos = (0..n).map(|_| self.spawn()).collect();
        self.vel = vec![0.0; n];
        self.cut = vec![false; n];
    }

    /// Make it `count` particles: those there are stay where they are, and new ones
    /// start in the starting cloud.
    fn resize(&mut self, count: usize) {
        let n = self.pos.len();
        for _ in n..count {
            let p = self.spawn();
            self.pos.push(p);
        }
        self.pos.truncate(count);
        self.vel.resize(count, 0.0);
        self.cut.resize(count, false);
    }

    /// Set up experiment `k` of `TRIALS`: all but the drawing.
    fn set_up(&mut self, k: usize) {
        let (_, _, rho, count, spread, _, _) = TRIALS[k];
        self.sys = 0;
        self.load_system();
        (self.k, self.spread, self.speed) = ([10.0, rho, 2.667], spread, 1.0);
        self.pos.resize(count, Vec3::ZERO);
        self.reset();
    }

    /// One step of `h` for every particle. One that leaves starts again somewhere in
    /// the starting cloud.
    fn step(&mut self, h: f32) {
        (self.system().step)(self, h);
    }

    fn step_with(&mut self, deriv: impl Fn(Vec3, &[f32; 3]) -> Vec3 + Copy, h: f32) {
        let n = self.pos.len();
        // (a few hundred particles are enough to judge the speeds by)
        let every = (n / 400).max(1);
        for i in 0..n {
            let (mut p, d) = rk4(deriv, &self.k, self.pos[i], h);
            let v = d.length();
            if escaped(p) {
                p = self.spawn();
                self.cut[i] = true;
            } else if v.is_finite() && i % every == 0 {
                self.speeds.push(v);
            }
            self.pos[i] = p;
            self.vel[i] = v;
        }
    }

    /// Bring the speed scale towards the speeds of this frame's steps, over about a
    /// fifth of a second of the scene's time.
    fn ease_speed_scale(&mut self, dt: f32) {
        if let Some(top) = typical_top_speed(&mut self.speeds).filter(|v| *v > 0.0) {
            self.vmax += (top - self.vmax) * (1.0 - (-dt.max(0.0) / 0.2).exp());
        }
        self.speeds.clear();
    }
}

impl Attractor {
    pub fn new(gfx: &Gfx) -> Self {
        Attractor { cloud: Cloud::new(2400), want_count: 2400, want_length: 160, length: 160, color_by: 0, cmap: 0, trails: TrailSet::new(gfx, 2400, 160, true), spin: true }
    }

    /// Every particle back to the starting cloud, with no trails.
    fn restart(&mut self) {
        self.cloud.reset();
        self.trails.clear();
    }

    /// New trails for the particles and length asked for. The particles there are
    /// stay where they are: changing the length does not start the picture again.
    fn rebuild(&mut self, gfx: &Gfx) {
        // (many particles cannot also have the longest trails: see `MAX_SEGMENTS`)
        self.length = self.want_length.min(longest_trail(self.want_count));
        self.trails = TrailSet::new(gfx, self.want_count as u32, self.length, true);
        self.cloud.resize(self.want_count);
    }

    /// Set up experiment `k` of `TRIALS`.
    fn set_up(&mut self, k: usize, gfx: &Gfx) {
        let (_, _, _, count, _, length, color_by) = TRIALS[k];
        self.cloud.set_up(k);
        (self.want_count, self.want_length, self.color_by) = (count, length, color_by);
        self.rebuild(gfx);
    }
}

impl Scene for Attractor {
    fn view(&self) -> View {
        View::Orbit { target: Vec3::ZERO, yaw: -0.6, pitch: 0.28, dist: 9.2 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.max(0.75);
    }

    fn auto_rotate(&self) -> f32 {
        if self.spin {
            0.12
        } else {
            0.0
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, gfx: &Gfx) {
        ui::note(ui, "Three equations say how a point moves through space. From almost anywhere nearby it is drawn onto one shape, the attractor (for the Lorenz system, shown first, a butterfly with two wings), and wanders over it for ever without repeating itself. Each glowing thread is the recent path of one particle. Two neighbours on it drift apart exponentially fast: chaos.");
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k, gfx);
            }
        }

        ui::section(ui, "System");
        let names: Vec<&str> = SYSTEMS.iter().map(|s| s.name).collect();
        if ui::choice(ui, &names, &mut self.cloud.sys) {
            self.cloud.load_system();
            self.trails.clear();
        }
        let s = self.cloud.system();
        ui::note(ui, "Each particle moves by");
        ui::equation(ui, s.equations);
        for i in 0..3 {
            let (name, _, lo, hi) = s.params[i];
            if !name.is_empty() {
                ui::slider(ui, name, &mut self.cloud.k[i], lo..=hi);
            }
        }
        ui::section(ui, "Particles");
        ui::slider_int_log(ui, "particles", &mut self.want_count, 1..=30000);
        let mut len = self.want_length as usize;
        ui::slider_int_log(ui, "trail length", &mut len, 8..=600);
        self.want_length = (len as u32).min(longest_trail(self.want_count));
        // (the trails are one buffer on the GPU: made anew once the slider is let go,
        // not on every frame of a drag)
        if (self.want_count, self.want_length) != (self.cloud.pos.len(), self.length) && !ui.input(|i| i.pointer.any_down()) {
            self.rebuild(gfx);
        }
        ui::note(ui, "A trail's length is counted in steps of the system's own time.");
        if longest_trail(self.want_count) < 600 {
            ui::note(ui, &format!("With {} particles a trail can be at most {} long: any longer would be more lines than a frame can draw.", count(self.want_count), longest_trail(self.want_count)));
        }
        if ui::slider_log(ui, "starting cloud size", &mut self.cloud.spread, 1e-5..=1.5) {
            self.restart();
        }
        ui::slider(ui, "speed", &mut self.cloud.speed, 0.0..=4.0);
        ui::note(ui, "How fast the system's time runs. If a frame has no time for it all, the particles slow down instead.");
        if ui.button("restart").clicked() {
            self.restart();
        }
        ui::section(ui, "Look");
        ui::choice(ui, &["by speed", "by particle", "by height"], &mut self.color_by);
        let names: Vec<&str> = ATTRACTOR_CMAPS.iter().map(|c| c.name()).collect();
        ui::choice(ui, &names, &mut self.cmap);
        ui.checkbox(&mut self.spin, "slowly rotate");

        ui::section(ui, "Reading the picture");
        // the colours in use, and what the two ends of the scale stand for
        let (low, high) = [("slow", "fast"), ("first particle", "last particle"), ("low", "high")][self.color_by];
        let (strip, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 8.0), egui::Sense::hover());
        let cmap = ATTRACTOR_CMAPS[self.cmap];
        let cells = 48;
        for i in 0..cells {
            let x = |j: usize| strip.left() + strip.width() * j as f32 / cells as f32;
            let cell = egui::Rect::from_min_max(egui::pos2(x(i), strip.top()), egui::pos2(x(i + 1) + 0.5, strip.bottom()));
            // the same part of the map as the trails use
            ui.painter().rect_filled(cell, 0.0, cmap.color(0.06 + 0.8 * (i as f32 + 0.5) / cells as f32).egui());
        }
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(low).size(12.0).color(pal::DIM.egui()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(egui::RichText::new(high).size(12.0).color(pal::DIM.egui()));
            });
        });
        ui::key(
            ui,
            &[
                (cmap.color(0.46), "each thread: the recent path of one particle, in the colours of the strip above"),
                (pal::WHITE, "white dots: where each particle is now (with 1,500 particles or fewer)"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Attractor:", "the shape that a system's paths are drawn onto and then stay on.");
        ui::term(ui, "Chaos:", "motion that follows exact rules, yet magnifies the smallest difference in the start until two paths have nothing in common.");
        ui::section(ui, "Numbers");
        ui::readout(ui, "line segments drawn", count(self.cloud.pos.len() * self.length as usize));
        ui::see_also(ui, &[("pendulum", "chaos in a machine you could build"), ("field", "particles riding a flow in the plane"), ("lyapunov", "how fast a nudge grows, in a deep network")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let s = self.cloud.system();
        // (the table is the fast way to a colour: this runs for every particle and row)
        let colours = ATTRACTOR_CMAPS[self.cmap].table();
        let (steps, h) = advance(s.dt, self.cloud.speed, f.dt);
        let n = self.cloud.pos.len();
        // Additive trails pile up, so the brightness of each one falls with how many
        // there are and how long they are; the picture keeps the same exposure.
        self.trails.style.intensity = (3.4 * (40.0 / n as f32).powf(0.75) * (160.0 / self.length as f32).sqrt()).clamp(0.008, 1.8);
        let vmax = self.cloud.vmax.max(1e-6);
        // as many steps as the frame has time for: past that the particles fall behind,
        // and the frame rate does not suffer
        let budget = crate::clock::frame_budget(f.real_dt, 0.008);
        for k in 0..steps {
            if k > 0 && budget.spent() {
                break;
            }
            self.cloud.step(h);
            if !row_due(&mut self.cloud.since_row, h / s.dt) {
                continue;
            }
            let c = &mut self.cloud;
            let last = n.saturating_sub(1).max(1) as f32;
            for (i, (point, ((p, v), cut))) in self.trails.row_mut().iter_mut().zip(c.pos.iter().zip(&c.vel).zip(&mut c.cut)).enumerate() {
                let q = (*p - s.center) * s.scale;
                let shade = match self.color_by {
                    0 => speed_shade(v / vmax),
                    1 => shade(i as f32 / last),
                    _ => shade(q.z * 0.25 + 0.5),
                };
                *point = LinePoint { pos: q.to_array(), width: if *cut { -1.0 } else { 1.5 }, color: colours[shade] };
                *cut = false;
            }
            self.trails.commit(f.gfx);
        }
        self.cloud.ease_speed_scale(f.dt);
        f.trails(&mut self.trails);
        // bright heads, when there are few enough particles to see them individually
        if n <= 1500 {
            for p in &self.cloud.pos {
                f.glow((*p - s.center) * s.scale, 6.0, pal::WHITE.a(0.35));
            }
        }
        let (low, high) = ATTRACTOR_CMAP_ENDS[self.cmap];
        let colour = match self.color_by {
            0 => format!("colour: speed, {low} slow, {high} fast"),
            1 => format!("colour: one per particle, {low} the first, {high} the last"),
            _ => format!("colour: height, {low} low, {high} high"),
        };
        bottom_caption(f, &["each thread: the recent path of one particle", &colour], 12.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every system's equations, and its parameters' names, are markup that sets: one
    /// that did not would be shown as it is written, dollars and all, and only once
    /// its system is chosen.
    #[test]
    fn every_system_is_written_in_markup_that_sets() {
        for s in &SYSTEMS {
            for text in [format!("${}$", s.equations)].into_iter().chain(s.params.iter().map(|p| p.0.to_string()).filter(|n| !n.is_empty())) {
                let found = crate::typeset::inspect(&text);
                assert!(found.error.is_none() && found.words.is_empty() && text.starts_with('$'), "{}: {text} {found:?}", s.name);
            }
        }
    }

    /// Many particles cannot also have the longest trails: no more line segments than
    /// a frame can draw, and never under eight rows.
    #[test]
    fn the_trails_stay_drawable() {
        for particles in [1usize, 7, 2400, 3333, 3334, 13_000, 19_999, 20_000, 22_371, 30_000] {
            let rows = longest_trail(particles) as usize;
            assert!(rows * particles <= MAX_SEGMENTS && rows >= 8, "{particles} particles, {rows} rows");
        }
        assert_eq!(longest_trail(2400), 600);
        assert_eq!(longest_trail(30_000), 66);
    }

    #[test]
    fn the_frame_advances_by_what_the_speeds_ask() {
        // speed 1 at 60 Hz: two steps of the system's own dt (as it always was)
        assert_eq!(advance(0.006, 1.0, 1.0 / 60.0), (2, 0.006));
        // nothing moves at speed zero, or while paused (dt = 0)
        assert_eq!(advance(0.006, 0.0, 1.0 / 60.0).0, 0);
        assert_eq!(advance(0.006, 1.0, 0.0).0, 0);
        assert_eq!(advance(0.006, 1.0, f32::NAN).0, 0);
        // the same simulated time per second at 30 Hz, and four times it at time speed
        // 4, in steps no longer than the system's dt
        for (speed, dt, total) in [(1.0, 1.0 / 30.0, 4.0), (1.0, 4.0 / 60.0, 8.0), (4.0, 1.0 / 60.0, 8.0), (0.2, 1.0 / 60.0, 0.4), (1.0, 1.0 / 90.0, 4.0 / 3.0)] {
            let (steps, h) = advance(0.01, speed, dt);
            assert!((steps as f32 * h / 0.01 - total).abs() < 1e-4 && h <= 0.01 + 1e-9, "speed {speed}, dt {dt}: {steps} x {h}");
        }
        // never more than MOST_STEPS steps
        let (steps, h) = advance(0.01, 4.0, 0.5);
        assert!(steps == MOST_STEPS && (h - 0.01).abs() < 1e-9);
    }

    /// A trail of a given length covers the same stretch of the system's time on any
    /// display and at any speed: a row for every `dt` of it, to within a row. (One row
    /// a step made trails half as long at 240 Hz as at 60, and shorter at low speeds.)
    #[test]
    fn trails_cover_the_same_time_on_any_display() {
        let mut rng = Rng::new(3);
        for hz in [15.0f32, 30.0, 50.0, 59.94, 60.0, 60.1, 72.0, 90.0, 120.0, 144.0, 165.0, 240.0] {
            for speed in [0.05f32, 0.25, 0.75, 1.0, 1.7, 4.0] {
                for jitter in [0.0f32, 0.01] {
                    let (mut since, mut rows, mut time) = (0.0f32, 0, 0.0f64);
                    for _ in 0..2000 {
                        let (steps, h) = advance(0.006, speed, (1.0 + jitter * rng.range(-1.0, 1.0)) / hz);
                        for _ in 0..steps {
                            rows += row_due(&mut since, h / 0.006) as u32;
                            time += (h / 0.006) as f64;
                        }
                    }
                    assert!((rows as f64 - time).abs() <= 1.5, "{hz} Hz, speed {speed}, jitter {jitter}: {rows} rows for {time:.1} dt");
                }
            }
        }
        // sixty frames a second at speed 1: two rows a frame, one a step, as before
        let (mut since, mut rows) = (0.0f32, 0);
        for _ in 0..100 {
            let (steps, h) = advance(0.006, 1.0, 1.0 / 60.0);
            rows += (0..steps).filter(|_| row_due(&mut since, h / 0.006)).count();
        }
        assert_eq!(rows, 200);
    }

    /// The lookup gives the colour the power would have given, or the next one.
    #[test]
    fn the_speed_shade_is_the_power_looked_up() {
        for j in 0..=20_000 {
            let x = j as f32 / 20_000.0;
            assert!(speed_shade(x).abs_diff(shade(x.powf(0.8))) <= 1, "{x}");
        }
        assert_eq!(speed_shade(f32::NAN), shade(1.0));
        assert_eq!(speed_shade(7.0), shade(1.0));
        assert_eq!(speed_shade(0.0), shade(0.0));
    }

    /// More particles, or a longer trail, do not start the picture again: the
    /// particles there are keep their places, and new ones start in the cloud.
    #[test]
    fn resizing_keeps_the_particles_there_are() {
        let mut c = Cloud::new(100);
        for _ in 0..50 {
            c.step(0.006);
        }
        let before = c.pos.clone();
        c.resize(150);
        assert_eq!(&c.pos[..100], &before[..]);
        assert!(c.pos.len() == 150 && c.vel.len() == 150 && c.cut.len() == 150);
        assert!(c.pos[100..].iter().all(|p| (*p - c.system().start).length() < 40.0));
        c.resize(30);
        assert_eq!(&c.pos[..], &before[..30]);
    }

    /// What the scene does each step, without the drawing: particles that leave are
    /// put back somewhere in the starting cloud, not all on one point; and the speed
    /// that colours are scaled by stays a speed found on the attractor, however fast
    /// the ones on their way out are going. (It used to become NaN with the first
    /// escape, and every trail one flat colour from then on.)
    #[test]
    fn escapees_come_back_apart_and_do_not_set_the_speed_scale() {
        for sys in 0..SYSTEMS.len() {
            let mut c = Cloud::new(2400);
            c.sys = sys;
            c.load_system();
            let system = c.system();
            let (steps, h) = advance(system.dt, 1.0, 1.0 / 60.0);
            let (mut lost_ever, mut on_attractor) = (0, 0.0f32);
            for frame in 0..300 {
                for _ in 0..steps {
                    c.step(h);
                    lost_ever += c.cut.iter().filter(|x| **x).count();
                    c.cut.fill(false);
                }
                // the fastest of those that are staying, late in the run
                if frame >= 250 {
                    on_attractor = c.pos.iter().zip(&c.vel).filter(|(p, _)| ((**p - system.center) * system.scale).length() < 3.0).map(|(_, v)| *v).fold(on_attractor, f32::max);
                }
                c.ease_speed_scale(1.0 / 60.0);
                assert!(c.vmax.is_finite() && c.vmax > 0.0, "{}: speed scale {}", system.name, c.vmax);
            }
            assert!(c.pos.iter().all(|p| p.is_finite()), "{}", system.name);
            assert!(c.vmax < 1.5 * on_attractor && c.vmax > 0.2 * on_attractor, "{}: scale {}, fastest on the attractor {on_attractor}", system.name, c.vmax);
            // no two particles share a place (they would share a path for good)
            let mut sorted: Vec<[u32; 3]> = c.pos.iter().map(|p| p.to_array().map(f32::to_bits)).collect();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), c.pos.len(), "{}: coincident particles", system.name);
            // (Halvorsen does throw part of this cloud off: the case is exercised)
            if sys == 4 {
                assert!(lost_ever > 0);
            }
        }
    }

    /// Experiment `k` as its button sets it up, with the particles' starting places
    /// drawn from `seed`, run for `seconds` at sixty frames a second; `look` sees it
    /// after every frame.
    fn run_trial(k: usize, seed: u64, seconds: f32, mut look: impl FnMut(f32, &Cloud)) {
        let mut c = Cloud::new(1);
        c.rng = Rng::new(seed);
        c.set_up(k);
        let (steps, h) = advance(c.system().dt, c.speed, 1.0 / 60.0);
        for frame in 1..=(seconds * 60.0) as usize {
            for _ in 0..steps {
                c.step(h);
            }
            look(frame as f32 / 60.0, &c);
        }
    }

    /// The size of a cloud on the screen: the root-mean-square distance from its centre.
    fn size(c: &Cloud) -> f32 {
        let mean = c.pos.iter().copied().sum::<Vec3>() / c.pos.len() as f32;
        (c.pos.iter().map(|p| (*p - mean).length_squared()).sum::<f32>() / c.pos.len() as f32).sqrt() * c.system().scale
    }

    /// The share of the particles on the wing where x > 0.
    fn right_wing(c: &Cloud) -> f32 {
        c.pos.iter().filter(|p| p.x > 0.0).count() as f32 / c.pos.len() as f32
    }

    /// The captions of the "things to try", held to account (several starts each).
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let scale = SYSTEMS[0].scale;
        for seed in 1..=6u64 {
            // two starts: one path for ten seconds or so, then apart (a tenth of a unit
            // on the screen, about the width of a wing's hole); after that each changes
            // wings when it will, so that they are as often on different wings as not
            let (mut parted, mut apart, mut frames) = (None, 0, 0);
            run_trial(0, seed, 90.0, |t, c| {
                if parted.is_none() && (c.pos[0] - c.pos[1]).length() * scale > 0.1 {
                    parted = Some(t);
                }
                if t > 30.0 {
                    frames += 1;
                    apart += (c.pos[0].x.signum() != c.pos[1].x.signum()) as usize;
                }
            });
            let apart = apart as f32 / frames as f32;
            assert!(parted.is_some_and(|t| (7.0..18.0).contains(&t)) && (0.35..0.65).contains(&apart), "seed {seed}: two starts part at {parted:?} s, then on different wings {apart} of the time");
            // rho = 10: all at rest at one of the two resting points within about
            // fifteen seconds (within a hundredth of a unit on the screen: a pixel or
            // two), roughly half at each
            let rest = (2.667f32 * 9.0).sqrt();
            let points = [Vec3::new(rest, rest, 9.0), Vec3::new(-rest, -rest, 9.0)];
            let (mut still_from, mut share) = (None, 0.0);
            run_trial(2, seed, 20.0, |t, c| {
                if c.pos.iter().all(|p| points.iter().any(|q| (*p - *q).length() * scale < 0.01)) {
                    still_from.get_or_insert(t);
                } else {
                    still_from = None;
                }
                share = right_wing(c);
            });
            assert!(still_from.is_some_and(|t| (11.0..16.0).contains(&t)) && (0.4..0.6).contains(&share), "seed {seed}: rho 10, all at rest from {still_from:?} s, {share} on the right");
        }
        // the speck: a streak at ten seconds, a thread at fifteen, spread over the
        // butterfly (as widely as a cloud that has had minutes to mix) at 45, half on
        // each wing
        let mut mixed = Cloud::new(2400);
        for _ in 0..12_000 {
            mixed.step(SYSTEMS[0].dt);
        }
        let mixed = size(&mixed);
        for seed in [1u64, 2] {
            let mut seen = Vec::new();
            run_trial(1, seed, 45.0, |t, c| {
                if [10.0f32, 15.0, 45.0].iter().any(|m| (t - m).abs() < 1e-3) {
                    seen.push((size(c), right_wing(c)));
                }
            });
            let [(ten, _), (fifteen, _), (end, share)] = seen[..] else { panic!("{seen:?}") };
            assert!(ten < 0.03 && fifteen > 0.3 && (end / mixed - 1.0).abs() < 0.06 && (0.45..0.55).contains(&share), "seed {seed}: the speck at 10, 15 and 45 s: {seen:?}, mixed {mixed}");
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        let s = &SYSTEMS[0];
        for seed in 1..=8u64 {
            // two starts: when they are first a tenth of a unit apart on the screen, and
            // how often they change wings
            let (mut parted, mut switches, mut wing) = (None, 0, 0.0f32);
            run_trial(0, seed, 60.0, |t, c| {
                let d = (c.pos[0] - c.pos[1]).length() * s.scale;
                if parted.is_none() && d > 0.1 {
                    parted = Some(t);
                }
                if c.pos[0].x.signum() != wing {
                    switches += 1;
                    wing = c.pos[0].x.signum();
                }
            });
            // the speck: its size (distance from its centre, on the screen) and how many on each wing
            let mut sizes = Vec::new();
            run_trial(1, seed, 60.0, |t, c| {
                if [1.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0].iter().any(|m| (t - m).abs() < 1e-3) {
                    let mean = c.pos.iter().copied().sum::<Vec3>() / c.pos.len() as f32;
                    let rms = (c.pos.iter().map(|p| (*p - mean).length_squared()).sum::<f32>() / c.pos.len() as f32).sqrt() * s.scale;
                    let right = c.pos.iter().filter(|p| p.x > 0.0).count() as f32 / c.pos.len() as f32;
                    sizes.push(format!("{t:.0}s {rms:.3}/{right:.2}"));
                }
            });
            // rho 10: when all are within a twentieth of a unit (on the screen) of one of the two points
            let rest = (2.667f32 * 9.0).sqrt();
            let points = [Vec3::new(rest, rest, 9.0), Vec3::new(-rest, -rest, 9.0)];
            let (mut settled, mut share) = (None, 0.0);
            run_trial(2, seed, 60.0, |t, c| {
                let far = c.pos.iter().map(|p| points.iter().map(|q| (*p - *q).length()).fold(f32::MAX, f32::min)).fold(0.0f32, f32::max) * s.scale;
                if far < 0.01 {
                    settled.get_or_insert(t);
                } else {
                    settled = None;
                }
                share = c.pos.iter().filter(|p| p.x > 0.0).count() as f32 / c.pos.len() as f32;
            });
            println!("seed {seed}: two starts part at {parted:?} s, {switches} wing changes in a minute; speck {}; rho 10: all settled from {settled:?} s, {share:.2} on the right", sizes.join(" "));
        }
        // a cloud of the default kind, well mixed, for comparison
        let mut c = Cloud::new(2400);
        for _ in 0..7200 {
            c.step(0.006);
        }
        let mean = c.pos.iter().copied().sum::<Vec3>() / c.pos.len() as f32;
        let rms = (c.pos.iter().map(|p| (*p - mean).length_squared()).sum::<f32>() / c.pos.len() as f32).sqrt() * s.scale;
        println!("a well mixed cloud: size {rms:.3}");
    }
}

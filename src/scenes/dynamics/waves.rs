//! Waves and interference.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "waves",
    title: "Waves and interference",
    group: "Dynamics",
    blurb: "The wave equation on a membrane you can touch.",
    make: |g| Box::new(Waves::new(g)),
};

const WAVE_N: usize = 192;

/// The update is stable only while c² (in grid cells per step, squared) stays under
/// one half; beyond that the shortest ripples grow without bound and the membrane
/// turns to NaN within seconds. The slider's range keeps under this; so does a clamp.
const C2_MAX: f32 = 0.49;
/// c at wave speed 1.
const C_PER_SPEED: f32 = 0.48;
const SPEED_MAX: f32 = 1.45;

/// One leapfrog step of u_tt = c² ∇²u on an `n` by `n` grid with the edges held
/// fixed: `v` is the change of `u` per step, `sponge` what is left of it at each cell
/// (the absorbing band at the edges), and nothing moves where there is a `wall`.
fn wave_step(u: &mut [f32], v: &mut [f32], wall: &[bool], sponge: &[f32], n: usize, c2: f32, damp: f32) {
    for j in 1..n - 1 {
        let row = j * n;
        for i in 1..n - 1 {
            let k = row + i;
            let lap = u[k - 1] + u[k + 1] + u[k - n] + u[k + n] - 4.0 * u[k];
            v[k] = (v[k] + c2 * lap) * damp * sponge[k];
        }
    }
    for k in 0..n * n {
        if wall[k] {
            v[k] = 0.0;
            u[k] = 0.0;
        } else {
            u[k] += v[k];
        }
    }
}

/// Experiments: (button, what to look for, sources: 0 none, 1 one, 2 two, frequency,
/// whether the edges absorb waves). Each also takes the barrier away, sets wave speed
/// 1 and damping 0.15, and stills the water first.
const TRIALS: [(&str, &str, usize, f32, bool); 3] = [
    ("two sources in step", "Two sources pushed up and down together. On the line straight out from between them their crests arrive together and add up. Along two still spokes either side of that line (measured: 20° from it halfway across the membrane, 23° further out), the crests of one always meet the troughs of the other, and the water hardly moves: interference.", 2, 3.0, true),
    ("a quicker rhythm", "The same at twice the frequency. The waves are half as long, and there are twice as many still spokes, closer together: about 12° and 36° either side of the middle line.", 2, 6.0, true),
    ("make your own", "No sources and still water: drag across it, then again across your first stroke. Rings spread out from each, and where they meet they pass through each other unchanged: the water simply adds them up.", 0, 3.0, true),
];

/// How far a finger pushes the membrane down for every grid cell of its stroke.
const POKE: f32 = -0.012;
/// A finger held still (or moving less than this many cells a step) pushes as hard as
/// one moving this many cells a step: one poke every sixtieth of a second of the
/// scene's time, as it always did at sixty frames a second.
const STILL: f32 = 60.0 / STEPS_PER_SECOND;

/// The membrane and what drives it: everything but the drawing (the tests run it).
struct Membrane {
    u: Vec<f32>,
    v: Vec<f32>,
    wall: Vec<bool>,
    sponge: Vec<f32>,
    speed: f32,
    damping: f32,
    sources: usize,
    freq: f32,
    barrier: usize,
    absorb: bool,
    time: f32,
    /// Steps of the membrane that are due and not yet taken (see [`steps_due`]).
    owed: f32,
    /// The picture no longer shows the membrane as it is (it moved, or the walls did).
    stale: bool,
}

impl Membrane {
    fn new() -> Self {
        let n = WAVE_N;
        let mut m = Membrane {
            u: vec![0.0; n * n],
            v: vec![0.0; n * n],
            wall: vec![false; n * n],
            sponge: vec![1.0; n * n],
            speed: 1.0,
            damping: 0.15,
            sources: 2,
            freq: 3.0,
            barrier: 0,
            absorb: true,
            time: 0.0,
            owed: 0.0,
            stale: true,
        };
        m.build_walls();
        m
    }

    fn build_walls(&mut self) {
        let n = WAVE_N;
        self.stale = true;
        self.wall.fill(false);
        // a band near the edges that soaks waves up, so they leave instead of echoing
        let band = 22.0;
        for j in 0..n {
            for i in 0..n {
                let d = i.min(j).min(n - 1 - i).min(n - 1 - j) as f32;
                let t = ((band - d) / band).max(0.0);
                self.sponge[j * n + i] = if self.absorb { 1.0 - 0.09 * t * t } else { 1.0 };
            }
        }
        let row = n * 5 / 9;
        let mid = n / 2;
        let open = |i: usize| match self.barrier {
            1 => i.abs_diff(mid) < 5,
            2 => i.abs_diff(mid - 14) < 4 || i.abs_diff(mid + 14) < 4,
            _ => true,
        };
        if self.barrier > 0 {
            for i in 0..n {
                if !open(i) {
                    self.wall[row * n + i] = true;
                    self.wall[(row + 1) * n + i] = true;
                }
            }
        }
    }

    fn clear(&mut self) {
        self.u.fill(0.0);
        self.v.fill(0.0);
        self.time = 0.0;
        self.stale = true;
    }

    /// Set up experiment `k` of `TRIALS`, on still water.
    fn set_up(&mut self, k: usize) {
        let (_, _, sources, freq, absorb) = TRIALS[k];
        (self.sources, self.freq, self.barrier, self.speed, self.damping, self.absorb) = (sources, freq, 0, 1.0, 0.15, absorb);
        self.build_walls();
        self.clear();
    }

    fn poke(&mut self, at: Vec2, amount: f32) {
        // `at` in grid coordinates
        let n = WAVE_N as i32;
        let r = 5;
        for dj in -r..=r {
            for di in -r..=r {
                let (i, j) = (at.x as i32 + di, at.y as i32 + dj);
                if i > 0 && j > 0 && i < n - 1 && j < n - 1 {
                    let d2 = (di * di + dj * dj) as f32;
                    self.v[(j * n + i) as usize] += amount * (-d2 / 6.0).exp();
                }
            }
        }
    }

    /// A finger that moves from `a` to `b` (grid coordinates) during one step pushes
    /// the membrane down along the way, in proportion to the distance, and as hard as
    /// a still finger where it hardly moves; in at most `most` pokes.
    fn press(&mut self, a: Vec2, b: Vec2, most: usize) {
        let len = (b - a).length();
        if !len.is_finite() {
            return;
        }
        let pokes = (len.ceil() as usize).clamp(1, most.max(1));
        let each = POKE * len.max(STILL) / len.ceil().max(1.0);
        for j in 1..=pokes {
            self.poke(a.lerp(b, j as f32 / pokes as f32), each);
        }
    }

    fn step(&mut self, c2: f32, damp: f32) {
        wave_step(&mut self.u, &mut self.v, &self.wall, &self.sponge, WAVE_N, c2, damp);
        self.stale = true;
    }

    /// Advance by a frame of `dt` seconds of the scene's time: the steps that are due,
    /// each with the sources' push, and with its share of a finger's `stroke` (from,
    /// to, in grid coordinates) over the frame. So a still finger pushes as hard per
    /// second on any display, and while the scene is paused it pushes nothing. Returns
    /// the number of steps taken.
    fn advance(&mut self, dt: f32, stroke: Option<(Vec2, Vec2)>) -> usize {
        if dt.is_nan() || dt <= 0.0 {
            return 0;
        }
        let n = WAVE_N;
        let c2 = (C_PER_SPEED * self.speed).powi(2).min(C2_MAX);
        let damp = 1.0 - 0.012 * self.damping;
        let steps = steps_due(&mut self.owed, dt);
        for s in 0..steps {
            self.time += 1.0 / STEPS_PER_SECOND;
            // each source drives a small soft patch, not a single cell (no needles)
            let push = (self.time * self.freq * std::f32::consts::TAU).sin() * 0.16;
            let (j, cols) = (WAVE_SOURCE_ROW, source_columns(self.sources));
            for &c in cols {
                for dj in -4i32..=4 {
                    for di in -4i32..=4 {
                        let w = (-((di * di + dj * dj) as f32) / 5.0).exp();
                        let k = ((j as i32 + dj) as usize) * n + (c as i32 + di) as usize;
                        // push on the membrane rather than pinning it: no kink at the source
                        self.v[k] += push * 0.05 * w;
                    }
                }
            }
            if let Some((from, to)) = stroke {
                let at = |s: usize| from.lerp(to, s as f32 / steps as f32);
                // (no more pokes in a frame than there always were: forty)
                self.press(at(s), at(s + 1), 40 / steps);
            }
            self.step(c2, damp);
        }
        steps
    }
}

pub struct Waves {
    m: Membrane,
    surf: Surface,
    cmap: usize,
    /// How far along the finger's path the membrane has been pushed.
    last_poke: Option<Vec2>,
    level: f32,
}

const WAVE_CMAPS: [Cmap; 4] = [Cmap::IceFire, Cmap::Ocean, Cmap::CoolWarm, Cmap::Twilight];

/// Grid row of the driven patches.
const WAVE_SOURCE_ROW: usize = WAVE_N / 6;

/// Grid columns of the driven patches, for none, one or two sources.
fn source_columns(sources: usize) -> &'static [usize] {
    const MID: usize = WAVE_N / 2;
    match sources {
        1 => &[MID],
        2 => &[MID - 18, MID + 18],
        _ => &[],
    }
}

impl Waves {
    pub fn new(gfx: &Gfx) -> Self {
        Waves { m: Membrane::new(), surf: Surface::new(gfx, WAVE_N as u32, WAVE_N as u32), cmap: 0, last_poke: None, level: 0.2 }
    }
}

/// The membrane takes this many steps for every second of the scene's time, on any
/// display. (Three for every frame, as it once was, made the waves run twice as fast
/// on a display twice as fast, while the sources kept their rhythm: another pattern.)
const STEPS_PER_SECOND: f32 = 180.0;

/// How quickly the colours follow the loudness of the water: they move a share
/// 1 - exp(-t / LEVEL_EASE) of the way in `t` seconds of the membrane's time.
const LEVEL_EASE: f32 = 0.55;

/// How many steps to take in a frame of `dt` seconds, given what was already `owed`:
/// what is due, less what is left for the next frame, and never more than twelve (a
/// very slow frame, or the time speed turned far up, then falls behind instead).
fn steps_due(owed: &mut f32, dt: f32) -> usize {
    *owed = (*owed + dt.max(0.0) * STEPS_PER_SECOND).min(12.0);
    // (a hair early, so that sixty frames a second is three steps every frame)
    let steps = (*owed + 1e-3).floor();
    *owed -= steps;
    steps as usize
}

impl Scene for Waves {
    fn pointer_hint(&self) -> Option<&'static str> {
        Some("drag: make waves    right-drag: turn    scroll: zoom")
    }

    fn view(&self) -> View {
        // far enough back, and aimed low enough, that all four corners are in the picture
        View::Orbit { target: Vec3::new(-0.4, -0.15, -1.0), yaw: -1.2, pitch: 0.78, dist: 16.6 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A stretched membrane, like a drum skin. Every point is pulled towards the level of its neighbours, so a disturbance spreads outwards as waves. This rule is the wave equation, for the height $u$ of each point and the wave speed $c$:");
        ui::equation(ui, r"\frac{\partial^2 u}{\partial t^2} = c^2 \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)");
        ui::note(ui, "Drag across the surface to make some.");
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.m.set_up(k);
            }
        }
        let m = &mut self.m;
        ui::section(ui, "Sources");
        ui::choice(ui, &["none", "one", "two"], &mut m.sources);
        ui::slider(ui, "frequency", &mut m.freq, 0.5..=6.0);
        ui::note(ui, "Swings a second. A source swinging twice as fast makes waves half as long.");
        ui::section(ui, "Barrier");
        if ui::choice(ui, &["none", "single slit", "double slit"], &mut m.barrier) {
            m.build_walls();
            m.clear();
        }
        ui::section(ui, "Medium");
        ui::slider(ui, "wave speed", &mut m.speed, 0.2..=SPEED_MAX);
        ui::slider(ui, "damping", &mut m.damping, 0.0..=1.0);
        ui::note(ui, "How quickly the water loses its motion.");
        if ui.checkbox(&mut m.absorb, "edges absorb waves (off: they reflect)").changed() {
            m.build_walls();
        }
        if ui.button("still the water").clicked() {
            m.clear();
        }
        ui::section(ui, "Reading the picture");
        let names: Vec<&str> = WAVE_CMAPS.iter().map(|c| c.name()).collect();
        ui::choice(ui, &names, &mut self.cmap);
        let cmap = WAVE_CMAPS[self.cmap];
        // in the order of WAVE_CMAPS: what the middle of each map looks like
        let middle = ["black", "teal", "pale grey", "dark purple"][self.cmap];
        ui::key(
            ui,
            &[
                (cmap.color(0.9), "crests: water above its resting level"),
                (cmap.color(0.1), "troughs: water below it"),
                (cmap.color(0.5), &format!("{middle}: at rest, or between a crest and a trough")),
                (pal::WHITE, "\"source\": a patch pushed up and down in a steady rhythm"),
            ],
        );
        ui::note(ui, "The height is exaggerated, and the colours follow how loud the water is, so that quiet and loud water both read. A barrier is a raised wall.");
        ui::section(ui, "Words");
        ui::term(ui, "Interference:", "where two waves meet they add up: crest on crest makes a bigger crest, crest on trough cancels.");
        ui::term(ui, "Diffraction:", "waves spreading out round an edge or through a gap, into places no straight line from the source reaches.");
        ui::see_also(ui, &[("fourier", "any repeating shape as a sum of simple waves"), ("pendulum", "one swinging thing, rather than a whole sheet of them")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let n = WAVE_N;
        let half = 3.6;
        let to_grid = |p: Vec3| Vec2::new((p.x / half * 0.5 + 0.5) * (n - 1) as f32, (p.y / half * 0.5 + 0.5) * (n - 1) as f32);

        // dragging pushes the surface down along the stroke
        f.handles.capture();
        let hit = f.ptr.pos.and_then(|p| f.cam.pick_ground(p)).filter(|p| p.x.abs() < half && p.y.abs() < half);
        let mut stroke = None;
        if f.ptr.down {
            if let Some(p) = hit {
                let g = to_grid(p);
                stroke = Some((self.last_poke.unwrap_or(g), g));
            }
        } else {
            self.last_poke = None;
        }
        let taken = self.m.advance(f.dt, stroke);
        // The stroke has been pressed in up to the finger once a step has taken it (a
        // fast display has frames without one: the stroke waits for the next). While
        // paused it presses nothing, and its path is not saved up for later.
        if let Some((_, to)) = stroke {
            if taken > 0 || f.dt <= 0.0 {
                self.last_poke = Some(to);
            }
        }

        // (nothing to redo, and nothing to send to the GPU, while the membrane is still:
        // paused, with no hand on it)
        if std::mem::take(&mut self.m.stale) {
            // colour and height follow the typical amplitude, so quiet and loud water
            // both read (eased over the membrane's time, not by frames)
            let rms = (self.m.u.iter().map(|v| v * v).sum::<f32>() / (n * n) as f32).sqrt();
            let ease = 1.0 - (-(taken as f32 / STEPS_PER_SECOND) / LEVEL_EASE).exp();
            self.level += ((rms * 3.2).max(0.06) - self.level) * ease;
            let scale = 0.12 / self.level;
            let (u, wall) = (&self.m.u, &self.m.wall);
            for (k, p) in self.surf.data_mut().iter_mut().enumerate() {
                let (i, j) = (k % n, k / n);
                let x = (i as f32 / (n - 1) as f32 * 2.0 - 1.0) * half;
                let y = (j as f32 / (n - 1) as f32 * 2.0 - 1.0) * half;
                let h = if wall[k] { 0.16 } else { u[k] };
                *p = [x, y, h * 2.2 * scale, if wall[k] { 0.0 } else { u[k] }];
            }
        }
        let level = self.level;
        let st = &mut self.surf.style;
        st.cmap = Some(WAVE_CMAPS[self.cmap]);
        st.range = [-level, level];
        st.specular = 0.9;
        st.rim = 0.7;
        f.surface(&mut self.surf);

        // name the driven patches: they swing further than the colours reach
        for &c in source_columns(self.m.sources) {
            let at = |i: usize| (i as f32 / (n - 1) as f32 * 2.0 - 1.0) * half;
            if let Some(px) = on_screen(f, Vec3::new(at(c), at(WAVE_SOURCE_ROW), 0.0)) {
                f.caption_px(px + Vec2::new(0.0, 34.0), "source", 12.0, pal::WHITE, Align2::CENTER_TOP);
            }
        }

        // what the colours mean, and (since plain dragging does something else here
        // than in every other scene) what a drag does
        let colours = [("red: above the resting level", "blue: below"), ("pale: crests", "dark: troughs"), ("red: above the resting level", "blue: below"), ("red: above the resting level", "blue: below")][self.cmap];
        bottom_caption(f, &[colours.0, colours.1, "drag on the surface to make waves", "right-drag to turn the view"], 12.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A splash, then many steps: how large the membrane's swing gets.
    fn largest_swing(c2: f32, steps: usize) -> f32 {
        let n = 64;
        let (mut u, mut v) = (vec![0.0f32; n * n], vec![0.0f32; n * n]);
        let (wall, sponge) = (vec![false; n * n], vec![1.0f32; n * n]);
        // a sharp poke: it has every wavelength in it, the shortest included
        v[(n / 2) * n + n / 2] = -0.5;
        v[(n / 2) * n + n / 2 + 1] = 0.3;
        let mut largest = 0.0f32;
        for _ in 0..steps {
            wave_step(&mut u, &mut v, &wall, &sponge, n, c2, 1.0);
            largest = u.iter().fold(largest, |m, x| m.max(x.abs()));
            if !largest.is_finite() {
                break;
            }
        }
        largest
    }

    /// The waves cover the same ground in a second of the scene's time whatever the
    /// display's frame rate and the time speed: three steps a frame at sixty frames a
    /// second, as there always were, and the same 180 a second at any other pace.
    #[test]
    fn the_membrane_keeps_time_at_any_frame_rate() {
        let mut owed = 0.0;
        assert!((0..200).all(|_| steps_due(&mut owed, 1.0 / 60.0) == 3));
        for dt in [1.0f32 / 120.0, 1.0 / 144.0, 1.0 / 30.0, 0.25 / 60.0, 2.0 / 60.0] {
            let mut owed = 0.0;
            let taken: usize = (0..1000).map(|_| steps_due(&mut owed, dt)).sum();
            let due = 1000.0 * dt * STEPS_PER_SECOND;
            assert!((taken as f32 - due).abs() <= 1.5, "dt {dt}: {taken} steps where {due} are due");
        }
        // a frame of a quarter of a second is not chased
        assert_eq!(steps_due(&mut 0.0, 0.25), 12);
        assert_eq!(steps_due(&mut 0.0, 0.0), 0);
    }

    #[test]
    fn the_membrane_is_stable_at_every_speed_the_slider_offers() {
        // undamped, reflecting edges: the worst case
        let top = (C_PER_SPEED * SPEED_MAX).powi(2);
        assert!(top <= C2_MAX);
        assert!(largest_swing(top, 6000) < 5.0, "at the slider's top: {}", largest_swing(top, 6000));
        assert!(largest_swing(C2_MAX, 6000) < 5.0);
        // and why the limit is where it is: just past one half it blows up
        let beyond = largest_swing(0.59, 6000);
        assert!(beyond.is_nan() || beyond > 1e6, "the old top of the slider (speed 1.6) was unstable: {beyond}");
    }

    /// A still finger and a moving one push the membrane as hard per second of the
    /// scene's time on any display; nothing is pushed while paused, and nothing is
    /// saved up for later. (One poke a frame made a still finger four times as strong
    /// at 240 Hz as at 60, and pokes made while paused all landed at once on resuming.)
    #[test]
    fn a_finger_pushes_as_hard_at_any_frame_rate() {
        // the push a finger gives in a second at `hz` frames a second, still or moving
        // at 100 cells a second, done as the scene does it: on a membrane that does not
        // carry the push away (no wave speed, no damping), so all of it is still there
        let pushed = |hz: f32, moving: bool| {
            let mut m = Membrane::new();
            m.set_up(2);
            (m.speed, m.damping, m.absorb) = (0.0, 0.0, false);
            m.build_walls();
            let mut last: Option<Vec2> = None;
            for frame in 0..hz as usize {
                let at = Vec2::new(40.0 + if moving { 100.0 * frame as f32 / hz } else { 0.0 }, 96.0);
                if m.advance(1.0 / hz, Some((last.unwrap_or(at), at))) > 0 {
                    last = Some(at);
                }
            }
            m.v.iter().sum::<f32>()
        };
        for moving in [false, true] {
            let at60 = pushed(60.0, moving);
            for hz in [30.0f32, 50.0, 72.0, 144.0, 240.0] {
                let other = pushed(hz, moving);
                assert!((other / at60 - 1.0).abs() < 0.03, "moving {moving}: a second at {hz} Hz pushes {other}, at 60 Hz {at60}");
            }
        }
        // paused: a finger on the membrane does nothing, then or later
        let mut m = Membrane::new();
        m.set_up(2);
        let at = Vec2::splat(96.0);
        for _ in 0..60 {
            assert_eq!(m.advance(0.0, Some((at, at))), 0);
        }
        assert!(m.v.iter().all(|v| *v == 0.0) && m.u.iter().all(|u| *u == 0.0));
    }

    /// Experiment `k` as its button sets it up, run at sixty frames a second: how
    /// strongly each cell swings (its root-mean-square height) from `from` to `to`
    /// seconds.
    fn swing_map(k: usize, from: f32, to: f32) -> Vec<f32> {
        let mut m = Membrane::new();
        m.set_up(k);
        let n = WAVE_N;
        let mut sum = vec![0.0f32; n * n];
        let mut frames = 0;
        for frame in 1..=(to * 60.0) as usize {
            m.advance(1.0 / 60.0, None);
            if frame as f32 / 60.0 > from {
                frames += 1;
                for (s, u) in sum.iter_mut().zip(&m.u) {
                    *s += u * u;
                }
            }
        }
        sum.iter().map(|s| (s / frames as f32).sqrt()).collect()
    }

    /// The swing along an arc of radius `r` cells round grid point `centre`, at
    /// `angle` degrees from the line straight out from the sources (towards higher
    /// rows).
    fn on_arc(map: &[f32], centre: (f32, f32), r: f32, angle: f32) -> f32 {
        let a = angle.to_radians();
        let (i, j) = ((centre.0 + r * a.sin()).round() as usize, (centre.1 + r * a.cos()).round() as usize);
        map[j * WAVE_N + i]
    }

    /// The still spokes along an arc: the angles (degrees, 1° apart, within `reach` of
    /// the middle line) where the swing is a local minimum below `deep` of the largest.
    fn spokes(map: &[f32], centre: (f32, f32), r: f32, reach: i32, deep: f32) -> Vec<i32> {
        let at = |d: i32| on_arc(map, centre, r, d as f32);
        let top = (-reach..=reach).map(at).fold(0.0f32, f32::max);
        (-reach + 2..=reach - 2).filter(|&d| at(d) < deep * top && (d - 2..=d + 2).all(|e| at(e) >= at(d))).collect()
    }

    /// The length of the waves leaving the sources: twice the mean distance between
    /// the places where the water crosses its resting level, along the line straight
    /// ahead from between the sources.
    fn wavelength(m: &Membrane) -> f32 {
        let n = WAVE_N;
        let line: Vec<f32> = (WAVE_SOURCE_ROW + 20..n - 30).map(|j| m.u[j * n + n / 2]).collect();
        let crossings: Vec<f32> = line.windows(2).enumerate().filter(|(_, w)| w[0].signum() != w[1].signum()).map(|(i, w)| i as f32 + w[0] / (w[0] - w[1])).collect();
        2.0 * (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f32
    }

    /// Run experiment `k` for `seconds` at sixty frames a second, with a finger's
    /// `strokes` (from, to, in grid cells, or none) in the first frames, one a frame.
    fn run(k: usize, seconds: f32, strokes: &[Option<(Vec2, Vec2)>]) -> Membrane {
        let mut m = Membrane::new();
        m.set_up(k);
        for &stroke in strokes {
            m.advance(1.0 / 60.0, stroke);
        }
        for _ in 0..(seconds * 60.0) as usize {
            m.advance(1.0 / 60.0, None);
        }
        m
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let between = (96.0, WAVE_SOURCE_ROW as f32);
        // two sources in step: two still spokes either side of the line straight out
        // from between them, where the water swings a fraction of what it does on that
        // line: 20° from it 60 cells out (halfway across the membrane, which is 192
        // cells, from the sources' row at 32), 23° at 90 cells. (Far out they would be
        // where the paths from the two sources, 36 cells apart, differ by half a wave of
        // 28.8 cells: at 23.6°.)
        let map = swing_map(0, 8.0, 10.0);
        for (r, angle) in [(60.0, 20), (90.0, 23)] {
            let found = spokes(&map, between, r, 45, 0.3);
            assert!(matches!(found[..], [a, b] if (a + angle).abs() <= 1 && (b - angle).abs() <= 1), "two sources, {r} cells out: spokes at {found:?}");
            assert!(on_arc(&map, between, r, 0.0) > 0.85 * (-45..=45).map(|d| on_arc(&map, between, r, d as f32)).fold(0.0f32, f32::max));
        }
        // a quicker rhythm: waves half as long, twice as many spokes, about 12° and
        // 36° either side
        let map = swing_map(1, 8.0, 10.0);
        for r in [60.0, 90.0] {
            let found = spokes(&map, between, r, 45, 0.25);
            assert!(matches!(found[..], [a, b, c, d] if (-40..=-32).contains(&a) && (-15..=-9).contains(&b) && (9..=15).contains(&c) && (32..=40).contains(&d)), "a quicker rhythm, {r} cells out: spokes at {found:?}");
        }
        let ratio = wavelength(&run(1, 8.0, &[])) / wavelength(&run(0, 8.0, &[]));
        assert!((ratio - 0.5).abs() < 0.03, "twice the frequency: waves {ratio} as long");
        // make your own: rings spread out from a stroke (as far in every direction),
        // and two strokes pass through each other unchanged: the water is the sum of
        // what each would have made alone
        let (p, q) = (Vec2::new(70.0, 80.0), Vec2::new(120.0, 110.0));
        let (one, other) = ((p, p + Vec2::new(4.0, 0.0)), (q, q + Vec2::new(0.0, 3.0)));
        let m = run(2, 0.4, &[Some((p, p))]);
        let ring: Vec<f32> = (0..36).map(|a| on_arc(&m.u, (p.x, p.y), 25.0, a as f32 * 10.0).abs()).collect();
        assert!(ring.iter().cloned().fold(f32::MAX, f32::min) > 0.9 * ring.iter().cloned().fold(0.0, f32::max) && ring[0] > 1e-5, "a ring 25 cells out: {ring:?}");
        let (a, b, both) = (run(2, 6.0, &[Some(one), None]), run(2, 6.0, &[None, Some(other)]), run(2, 6.0, &[Some(one), Some(other)]));
        let largest = both.u.iter().fold(0.0f32, |m, u| m.max(u.abs()));
        let apart = both.u.iter().zip(a.u.iter().zip(&b.u)).map(|(c, (x, y))| (c - x - y).abs()).fold(0.0f32, f32::max);
        assert!(largest > 1e-4 && apart < 1e-4 * largest, "two strokes together differ from their sum by {apart} (of {largest})");
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        let between = (96.0, WAVE_SOURCE_ROW as f32);
        for k in [0usize, 1] {
            let map = swing_map(k, 8.0, 10.0);
            for r in [60.0f32, 90.0] {
                let top = (-45..=45).map(|d| on_arc(&map, between, r, d as f32)).fold(0.0f32, f32::max);
                let depths: Vec<String> = spokes(&map, between, r, 45, 0.5).iter().map(|d| format!("{d}°: {:.2}", on_arc(&map, between, r, *d as f32) / top)).collect();
                println!("{} at {r} cells: on the middle line {:.2} of the largest; spokes {}", TRIALS[k].0, on_arc(&map, between, r, 0.0) / top, depths.join(", "));
            }
            println!("   wavelength {:.1} cells", wavelength(&run(k, 8.0, &[])));
        }
        let p = Vec2::new(70.0, 80.0);
        let m = run(2, 0.4, &[Some((p, p))]);
        let ring: Vec<f32> = (0..36).map(|a| on_arc(&m.u, (p.x, p.y), 25.0, a as f32 * 10.0).abs()).collect();
        println!("a ring 25 cells out: weakest {:.2} of the strongest", ring.iter().cloned().fold(f32::MAX, f32::min) / ring.iter().cloned().fold(0.0, f32::max));
    }
}

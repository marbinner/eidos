//! Vector fields and flow.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "field",
    title: "Vector fields and flow",
    group: "Calculus",
    blurb: "Thousands of particles riding a differential equation.",
    make: |g| Box::new(VectorField::new(g)),
};

const FIELD_PRESETS: [(&str, &str, &str, f32); 9] = [
    ("vortex", "-y", "x", 3.2),
    ("saddle", "x", "-y", 3.2),
    ("spiral sink", "-x - 2y", "2x - y", 3.2),
    ("pendulum", "y", "-sin(x) - 0.15 y", 4.2),
    ("Van der Pol", "y", "1.5 (1 - x^2) y - x", 4.2),
    ("predator-prey", "x (1 - y)", "y (x - 1)", 3.2),
    ("dipole", "(x+1)/((x+1)^2+y^2+0.1) - (x-1)/((x-1)^2+y^2+0.1)", "y/((x+1)^2+y^2+0.1) - y/((x-1)^2+y^2+0.1)", 3.0),
    ("double gyre", "-sin(pi x) cos(pi y) - 0.4 sin(t) cos(pi x) sin(pi y)", "cos(pi x) sin(pi y)", 1.6),
    ("waves", "sin(y + t)", "sin(x - t)", 4.5),
];

/// What each preset shows, in the order of `FIELD_PRESETS`.
const FIELD_NOTES: [&str; 9] = [
    "Vortex: every particle circles the origin, all at the same angular speed.",
    "Saddle: the flow comes in along the $y$ axis and leaves along the $x$ axis. Only the origin stays put.",
    "Spiral sink: everything winds inwards and comes to rest at the origin.",
    r"A pendulum with a little friction: $x$ is its angle and $y$ its angular velocity. The eyes at $x = 0, \pm 2\pi, \pm 4\pi$ and so on are all the pendulum hanging at rest (each $2\pi$ is one more full turn), and every particle is heading for one of them: the faster it starts, the more times it swings over the top on the way. The pinch points at $x = \pm\pi$ are the pendulum balanced upside down.",
    "Van der Pol oscillator: whether a particle starts inside or outside, it is drawn onto the same closed loop, a limit cycle.",
    "Predator and prey (Lotka–Volterra): $x$ is the prey and $y$ the predators. Their numbers chase each other round the balance point $(1, 1)$, for ever.",
    "Dipole: the flow streams out of a source at $(-1, 0)$ and into a sink at $(1, 0)$.",
    "Gyres: a chessboard of cells turning in alternate directions, shaken from side to side in time, so that particles leak from one cell to the next.",
    "Waves: this field changes with time ($t$), so a particle's path is not one of the curves you see at any one instant.",
];

/// One of the things to try.
struct Trial {
    label: &'static str,
    look_for: &'static str,
    /// The field: an index into `FIELD_PRESETS`.
    preset: usize,
    /// The background, as in the background choice.
    background: usize,
    /// A particle to release (at time 0) and follow.
    release: Option<[f32; 2]>,
}

const TRIALS: [Trial; 4] = [
    Trial {
        label: "a source and a sink",
        look_for: "The background shows the divergence: red where the flow spreads out, blue where it gathers. That happens round the source at $(-1, 0)$ and the sink at $(1, 0)$, and fades away from them: elsewhere the flow mostly just passes through.",
        preset: 6,
        background: 2,
        release: None,
    },
    Trial {
        label: "a vortex spins everywhere",
        look_for: "The background shows the curl, and it is the same red all over, not only at the centre: a tiny paddle wheel dropped anywhere in this flow turns at the same rate as the whole.",
        preset: 0,
        background: 3,
        release: None,
    },
    Trial {
        label: "a loop that draws everything in",
        look_for: "The white line is the path from a point near the centre: it spirals out onto the loop. Click outside the loop and the path from there spirals in onto the same loop.",
        preset: 4,
        background: 1,
        release: Some([0.2, 0.0]),
    },
    Trial {
        label: "a flow that changes with time",
        look_for: "The cells sway from side to side. The white line is the path ahead of one particle released at $(0.9, 0.5)$, and the white dot travels it: watch it leave its cell for the next.",
        preset: 7,
        background: 1,
        release: Some([0.9, 0.5]),
    },
];

/// The streaks get a row every so much scene time (one a frame at sixty frames a
/// second), so that a streak is always the same stretch of motion: counted in frames,
/// it was four times shorter at 240 Hz than at 60, and dimmer.
const ROW_EVERY: f32 = 1.0 / 60.0;
/// Rows in a streak: with `ROW_EVERY`, the last three quarters of a second.
const ROWS: u32 = 44;
/// The most steps one frame takes: a very long frame (or a high time speed) falls
/// behind rather than taking longer still.
const MOST_STEPS: usize = 16;
/// A background that changes with time is worked out again at most this often (in
/// seconds of scene time).
const BACKDROP_EVERY: f32 = 1.0 / 15.0;
/// The background's grid.
const BACKDROP_SIZE: (usize, usize) = (144, 96);
/// How far ahead, in the field's own time, the path of a released particle is traced.
const PATH_AHEAD: f32 = 12.0;

/// How many steps of `ROW_EVERY` are due after a frame of `dt` seconds, given what is
/// `owed` from before. What is left over is kept for the next frame; what a very long
/// frame owes beyond `MOST_STEPS` is let go.
fn steps_due(owed: &mut f32, dt: f32) -> usize {
    *owed += dt.max(0.0);
    // (a hair early, so that rounding never makes a frame of exactly ROW_EVERY wait)
    let due = (*owed / ROW_EVERY + 1e-3).floor();
    *owed -= due * ROW_EVERY;
    (due as usize).min(MOST_STEPS)
}

/// Positions and velocities of many points, for formulas evaluated over all of them
/// at once.
#[derive(Default)]
struct Points {
    xs: Vec<f32>,
    ys: Vec<f32>,
    us: Vec<f32>,
    vs: Vec<f32>,
    batch: Batch,
}

impl Points {
    /// One midpoint step: the velocity each point at `(xs, ys)` moves with over a
    /// step of `h` from time `t`, into `(us, vs)`. (`xs` and `ys` are left at the
    /// half-way points.)
    fn midpoint(&mut self, p: &Expr, q: &Expr, t: f32, h: f32) {
        p.eval_batch(&[&self.xs, &self.ys, &[t]], &mut self.us, &mut self.batch);
        q.eval_batch(&[&self.xs, &self.ys, &[t]], &mut self.vs, &mut self.batch);
        for i in 0..self.xs.len() {
            self.xs[i] += 0.5 * h * self.us[i];
            self.ys[i] += 0.5 * h * self.vs[i];
        }
        p.eval_batch(&[&self.xs, &self.ys, &[t + 0.5 * h]], &mut self.us, &mut self.batch);
        q.eval_batch(&[&self.xs, &self.ys, &[t + 0.5 * h]], &mut self.vs, &mut self.batch);
    }
}

/// The background's numbers: a grid of points and what is shown at each, with room
/// to work them out in, kept from frame to frame.
#[derive(Default)]
struct Backdrop {
    xs: Vec<f32>,
    ys: Vec<f32>,
    values: Vec<f32>,
    /// Their sizes, sorted in part to find the colour scale.
    mags: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    plus: Vec<f32>,
    minus: Vec<f32>,
    batch: Batch,
}

impl Backdrop {
    /// Points `nx` by `ny` over the rectangle from `lo` to `hi`.
    fn grid(&mut self, lo: Vec2, hi: Vec2, nx: usize, ny: usize) {
        let size = hi - lo;
        let (du, dv) = (1.0 / (nx - 1) as f32, 1.0 / (ny - 1) as f32);
        self.xs.clear();
        self.ys.clear();
        for j in 0..ny {
            for i in 0..nx {
                self.xs.push(lo.x + size.x * (i as f32 * du));
                self.ys.push(lo.y + size.y * (j as f32 * dv));
            }
        }
    }

    /// What the background shows at every point of the grid at time `t`, into
    /// `values`: the speed (`mode` 1), the divergence (2) or the curl (3). These two
    /// are worked out from differences `h` either side of each point.
    fn work_out(&mut self, p: &Expr, q: &Expr, mode: usize, t: f32, h: f32) {
        let n = self.xs.len();
        self.values.clear();
        self.values.resize(n, 0.0);
        // (the evaluations fill as many as their output holds)
        self.a.resize(n, 0.0);
        self.b.resize(n, 0.0);
        if mode == 1 {
            p.eval_batch(&[&self.xs, &self.ys, &[t]], &mut self.a, &mut self.batch);
            q.eval_batch(&[&self.xs, &self.ys, &[t]], &mut self.b, &mut self.batch);
            for k in 0..n {
                self.values[k] = self.a[k].hypot(self.b[k]);
            }
            return;
        }
        // divergence: dp/dx + dq/dy; curl: dq/dx - dp/dy
        let (first, second, sign) = if mode == 2 { ((p, true), (q, false), 1.0) } else { ((q, true), (p, false), -1.0) };
        self.add_slope(first.0, first.1, 1.0, t, h);
        self.add_slope(second.0, second.1, sign, t, h);
    }

    /// Add `sign` times the slope of `e` along x (or y) at every point to `values`.
    fn add_slope(&mut self, e: &Expr, along_x: bool, sign: f32, t: f32, h: f32) {
        let base = if along_x { &self.xs } else { &self.ys };
        self.plus.clear();
        self.plus.extend(base.iter().map(|v| v + h));
        self.minus.clear();
        self.minus.extend(base.iter().map(|v| v - h));
        if along_x {
            e.eval_batch(&[&self.plus, &self.ys, &[t]], &mut self.a, &mut self.batch);
            e.eval_batch(&[&self.minus, &self.ys, &[t]], &mut self.b, &mut self.batch);
        } else {
            e.eval_batch(&[&self.xs, &self.plus, &[t]], &mut self.a, &mut self.batch);
            e.eval_batch(&[&self.xs, &self.minus, &[t]], &mut self.b, &mut self.batch);
        }
        for k in 0..self.values.len() {
            // (divided by the distance the two points really are apart, after rounding)
            self.values[k] += sign * (self.a[k] - self.b[k]) / (self.plus[k] - self.minus[k]);
        }
    }

    /// The colour scale: the 97th percentile of the sizes, so that one singular point
    /// cannot wash out the map.
    fn scale(&mut self) -> f32 {
        self.mags.clear();
        self.mags.extend(self.values.iter().filter(|v| v.is_finite()).map(|v| v.abs()));
        if self.mags.is_empty() {
            return 1e-9;
        }
        let idx = self.mags.len() * 97 / 100;
        self.mags.select_nth_unstable_by(idx, f32::total_cmp);
        self.mags[idx].max(1e-9)
    }
}

/// The step of the differences for the divergence and the curl over the view from
/// `lo` to `hi`: small beside the view, and never so small that rounding the
/// coordinates spoils it.
fn difference_step(lo: Vec2, hi: Vec2) -> f32 {
    (1e-3 * (hi.y - lo.y)).max(2e-4 * lo.abs().max(hi.abs()).max_element())
}

/// A path drawn where the visitor clicked (or an experiment released a particle).
struct Probe {
    path: Vec<Vec3>,
    /// For a field that changes with time, the field's time at each point of the
    /// path, so that a marker can travel it; otherwise empty.
    times: Vec<f32>,
}

impl Probe {
    /// Where the released particle is at time `t`, if it is on the path.
    fn at(&self, t: f32) -> Option<Vec3> {
        let k = self.times.partition_point(|&s| s <= t);
        if k == 0 || k >= self.times.len() {
            return None;
        }
        let (t0, t1) = (self.times[k - 1], self.times[k]);
        let u = ((t - t0) / (t1 - t0).max(1e-9)).clamp(0.0, 1.0);
        Some(self.path[k - 1].lerp(self.path[k], u))
    }
}

pub struct VectorField {
    p: ExprField,
    q: ExprField,
    count: usize,
    /// The count on the slider: taken up when the slider is let go, since a new count
    /// means new streaks (a buffer of up to 37 MB) and a new set of particles.
    wanted: usize,
    pos: Vec<Vec2>,
    life: Vec<f32>,
    fresh: Vec<bool>,
    /// (None only in the tests, which have no graphics device; so for `bg`.)
    trails: Option<TrailSet>,
    rng: Rng,
    bg: Option<Surface>,
    bg_mode: usize,
    /// The view the background was worked out for (NaN: work it out again).
    bg_view: [f32; 4],
    /// Scene time since the background was worked out.
    bg_age: f32,
    backdrop: Backdrop,
    arrows: bool,
    speed: f32,
    vmax: f32,
    /// The field's own time, the t in the formulas: scene time times `speed`.
    clock: f32,
    /// Scene time not yet stepped through.
    owed: f32,
    points: Points,
    probes: Vec<Probe>,
    /// A particle an experiment asked to release (in the next frame, which knows the view).
    release: Option<Vec2>,
    half_height: f32,
    /// A preset has asked for a view of another size.
    new_view: bool,
}

impl VectorField {
    pub fn new(gfx: &Gfx) -> Self {
        let mut s = Self::create();
        s.trails = Some(Self::make_trails(gfx, s.count));
        s.bg = Some(Surface::new(gfx, BACKDROP_SIZE.0 as u32, BACKDROP_SIZE.1 as u32));
        s
    }

    /// The scene without its pictures on the graphics device (the tests make it so).
    fn create() -> Self {
        let count = 5000;
        VectorField {
            p: ExprField::new(FIELD_PRESETS[3].1, &["x", "y", "t"]),
            q: ExprField::new(FIELD_PRESETS[3].2, &["x", "y", "t"]),
            count,
            wanted: count,
            pos: Vec::new(),
            life: Vec::new(),
            fresh: Vec::new(),
            trails: None,
            rng: Rng::new(3),
            bg: None,
            bg_mode: 1,
            bg_view: [f32::NAN; 4],
            bg_age: 0.0,
            backdrop: Backdrop::default(),
            arrows: false,
            speed: 1.0,
            vmax: 1.0,
            clock: 0.0,
            owed: 0.0,
            points: Points::default(),
            probes: Vec::new(),
            release: None,
            half_height: FIELD_PRESETS[3].3,
            new_view: false,
        }
    }

    /// The streaks: one short tapering trail per particle.
    fn make_trails(gfx: &Gfx, count: usize) -> TrailSet {
        let mut trails = TrailSet::new(gfx, count as u32, ROWS, true);
        trails.style.intensity = 1.25;
        trails.style.fade_power = 1.6;
        trails.style.taper = 0.5;
        trails
    }

    fn time_dependent(&self) -> bool {
        self.p.expr.uses(2) || self.q.expr.uses(2)
    }

    /// Show preset `i`.
    fn load(&mut self, i: usize) {
        self.p.set(FIELD_PRESETS[i].1);
        self.q.set(FIELD_PRESETS[i].2);
        self.new_view = self.half_height != FIELD_PRESETS[i].3;
        self.half_height = FIELD_PRESETS[i].3;
        self.formula_changed();
    }

    fn formula_changed(&mut self) {
        self.probes.clear();
        self.bg_view = [f32::NAN; 4];
        self.vmax = 1e-3;
    }

    /// Set up experiment `k` of `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let trial = &TRIALS[k];
        self.load(trial.preset);
        self.bg_mode = trial.background;
        (self.speed, self.arrows, self.clock) = (1.0, false, 0.0);
        self.release = trial.release.map(Vec2::from);
    }

    fn field_at(&self, p: Vec2, t: f32) -> Vec2 {
        Vec2::new(self.p.expr.eval(&[p.x, p.y, t]), self.q.expr.eval(&[p.x, p.y, t]))
    }

    /// The path through `start`, both ways, of a field that does not change with
    /// time: every particle that passes there follows it.
    fn streamline(&self, start: Vec2, lo: Vec2, hi: Vec2) -> Vec<Vec3> {
        // (`lo..hi` is three views across: a step of a hundredth of a view, fine enough
        // that a particle never strays visibly from the curve where the flow turns sharply)
        let h = 0.004 * (hi.y - lo.y);
        let mut out = Vec::new();
        for dir in [-1.0f32, 1.0] {
            let mut p = start;
            let mut part = vec![p.extend(0.0)];
            for _ in 0..2100 {
                // classical Runge-Kutta, with the step normalised to constant arc length
                let unit = |q: Vec2| self.field_at(q, 0.0).normalize_or_zero() * dir;
                let k1 = unit(p);
                let k2 = unit(p + k1 * h * 0.5);
                let k3 = unit(p + k2 * h * 0.5);
                let k4 = unit(p + k3 * h);
                let d = (k1 + k2 * 2.0 + k3 * 2.0 + k4) / 6.0;
                if d.length() < 1e-4 {
                    break;
                }
                p += d * h;
                if !p.is_finite() || p.x < lo.x || p.x > hi.x || p.y < lo.y || p.y > hi.y {
                    break;
                }
                part.push(p.extend(0.0));
            }
            if dir < 0.0 {
                part.reverse();
                out = part;
            } else {
                out.extend(part.into_iter().skip(1));
            }
        }
        out
    }

    /// The path ahead of a particle released at `start` at time `t` in a field that
    /// changes with time, for `PATH_AHEAD` of the field's time (or until it leaves
    /// `lo..hi`), with the time at each point.
    fn pathline(&self, start: Vec2, t: f32, lo: Vec2, hi: Vec2) -> Probe {
        // steps a fixed distance long where the flow is fast, a fixed time where it is slow
        let ds = 0.004 * (hi.y - lo.y);
        let (mut p, mut s) = (start, t);
        let (mut path, mut times) = (vec![p.extend(0.0)], vec![s]);
        while s < t + PATH_AHEAD && path.len() < 6000 {
            let speed = self.field_at(p, s).length();
            let h = if speed > 0.0 { (ds / speed).min(0.05) } else { 0.05 };
            // classical Runge-Kutta, in time
            let k1 = self.field_at(p, s);
            let k2 = self.field_at(p + k1 * h * 0.5, s + h * 0.5);
            let k3 = self.field_at(p + k2 * h * 0.5, s + h * 0.5);
            let k4 = self.field_at(p + k3 * h, s + h);
            p += (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (h / 6.0);
            s += h;
            if !p.is_finite() || p.x < lo.x || p.x > hi.x || p.y < lo.y || p.y > hi.y {
                break;
            }
            path.push(p.extend(0.0));
            times.push(s);
        }
        Probe { path, times }
    }

    /// Draw the path through `at`, in the view from `lo` to `hi`.
    fn trace(&mut self, at: Vec2, lo: Vec2, hi: Vec2) {
        let size = hi - lo;
        let probe = if self.time_dependent() {
            self.pathline(at, self.clock, lo - size, hi + size)
        } else {
            Probe { path: self.streamline(at, lo - size, hi + size), times: Vec::new() }
        };
        self.probes.push(probe);
        if self.probes.len() > 12 {
            self.probes.remove(0);
        }
    }

    /// New particles scattered over the view from `lo` to `hi`.
    fn scatter(&mut self, lo: Vec2, hi: Vec2) {
        let (n, size) = (self.count, hi - lo);
        self.pos = (0..n).map(|_| lo + Vec2::new(self.rng.f32(), self.rng.f32()) * size).collect();
        self.life = (0..n).map(|_| self.rng.range(0.5, 5.0)).collect();
        self.fresh = vec![true; n];
        for v in [&mut self.points.xs, &mut self.points.ys, &mut self.points.us, &mut self.points.vs] {
            v.resize(n, 0.0);
        }
        if let Some(trails) = &mut self.trails {
            trails.clear();
        }
    }

    /// Move every particle on by one step (`ROW_EVERY` of scene time, `speed` times
    /// that of the field's), with the midpoint method and the whole population at
    /// once, and write where they are into the streaks' newest row. Those that leave
    /// the view from `lo` to `hi`, or have lived their time, start again somewhere new.
    fn step(&mut self, lo: Vec2, hi: Vec2) {
        let n = self.pos.len();
        let h = ROW_EVERY * self.speed;
        for i in 0..n {
            self.points.xs[i] = self.pos[i].x;
            self.points.ys[i] = self.pos[i].y;
        }
        self.points.midpoint(&self.p.expr, &self.q.expr, self.clock, h);
        let size = hi - lo;
        let pad = size * 0.06;
        let mut step_max = 0.0f32;
        // (from the ready-made table: this is done for every particle, every step)
        let colours = Cmap::Aurora.table();
        let mut row = self.trails.as_mut().map(|t| t.row_mut());
        for i in 0..n {
            let vel = Vec2::new(self.points.us[i], self.points.vs[i]);
            let speed = vel.length();
            let mut p = self.pos[i] + vel * h;
            self.life[i] -= ROW_EVERY;
            let out = p.x < lo.x - pad.x || p.x > hi.x + pad.x || p.y < lo.y - pad.y || p.y > hi.y + pad.y;
            let respawn = out || self.life[i] < 0.0 || !p.is_finite();
            if respawn {
                p = lo + Vec2::new(self.rng.f32(), self.rng.f32()) * size;
                self.life[i] = self.rng.range(1.5, 6.0);
            }
            if speed.is_finite() {
                step_max = step_max.max(speed);
            }
            if let Some(slot) = row.as_deref_mut().and_then(|r| r.get_mut(i)) {
                let c = colours[table_index(0.2 + 0.8 * (speed / self.vmax).clamp(0.0, 1.0).sqrt())];
                // a negative width cuts the trail, so a respawn never draws a streak
                *slot = LinePoint { pos: [p.x, p.y, 0.0], width: if respawn || self.fresh[i] { -1.0 } else { 1.7 }, color: c.a(0.62) };
            }
            self.fresh[i] = false;
            self.pos[i] = p;
        }
        // the colour scale follows the fastest particle, easing by 5 % a step (steps
        // come every ROW_EVERY of scene time: a time constant of a third of a second)
        self.vmax += (step_max.max(1e-3) - self.vmax) * 0.05;
        self.clock += h;
    }

    /// Work the background out again if the view or the formulas have changed, or
    /// (at most every `BACKDROP_EVERY`) if the field changes with time.
    fn update_backdrop(&mut self, dt: f32, lo: Vec2, hi: Vec2) {
        let Some(bg) = &mut self.bg else { return };
        self.bg_age += dt;
        let view = [lo.x, lo.y, hi.x, hi.y];
        let moved = view != self.bg_view;
        let time_dep = self.p.expr.uses(2) || self.q.expr.uses(2);
        let due = moved || (time_dep && self.bg_age >= BACKDROP_EVERY);
        if !due {
            return;
        }
        let (nx, ny) = bg.size();
        if moved || self.backdrop.xs.len() != nx * ny {
            self.backdrop.grid(lo, hi, nx, ny);
        }
        self.bg_view = view;
        self.bg_age = 0.0;
        self.backdrop.work_out(&self.p.expr, &self.q.expr, self.bg_mode, self.clock, difference_step(lo, hi));
        let peak = self.backdrop.scale();
        let b = &self.backdrop;
        // (where the field has no value the background has a hole)
        for (k, v) in bg.data_mut().iter_mut().enumerate() {
            *v = [b.xs[k], b.ys[k], -0.5, b.values[k]];
        }
        let st = &mut bg.style;
        st.lit = 0.0;
        st.opacity = 0.55;
        if self.bg_mode == 1 {
            st.cmap = Some(Cmap::Ocean);
            st.range = [0.0, peak * 1.6];
        } else {
            st.cmap = Some(Cmap::CoolWarm);
            st.range = [-peak, peak];
        }
    }
}

impl Scene for VectorField {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: self.half_height }
    }

    /// Each preset is shown at the scale that suits it.
    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.10;
        s.axis_alpha = 0.3;
    }

    fn ui(&mut self, ui: &mut egui::Ui, gfx: &Gfx) {
        ui::note(ui, r"A vector field gives every point a velocity: the two formulas below say how fast a particle at $(x, y)$ moves across ($\mathrm{d}x/\mathrm{d}t$) and up ($\mathrm{d}y/\mathrm{d}t$). Thousands of particles ride it, each drawn as a streak that is brightest at its head and fades behind.");
        if self.time_dependent() {
            ui::note(ui, "This field changes with time (it uses $t$), so a click draws the path ahead of a particle released there now, and a white dot travels it. A particle that passes the same point later takes another path.");
        } else {
            ui::note(ui, "Click anywhere to draw the path through that point: every particle that passes there follows it.");
        }

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The field");
        let mut changed = self.p.ui(ui, "dx/dt =");
        changed |= self.q.ui(ui, "dy/dt =");
        if changed {
            self.formula_changed();
        }
        let labels: Vec<&str> = FIELD_PRESETS.iter().map(|p| p.0).collect();
        if let Some(i) = ui::buttons(ui, &labels) {
            self.load(i);
        }
        // say what is on show, for as long as the formulas are those of a preset
        if let Some(i) = FIELD_PRESETS.iter().position(|p| p.1 == self.p.text && p.2 == self.q.text) {
            ui::note(ui, FIELD_NOTES[i]);
        }
        ui::section(ui, "Particles");
        ui::slider_int_log(ui, "count", &mut self.wanted, 500..=40000);
        if self.wanted != self.count && !ui.input(|i| i.pointer.any_down()) {
            self.count = self.wanted;
            self.trails = Some(Self::make_trails(gfx, self.count));
            self.pos.clear();
        }
        ui::slider(ui, "speed", &mut self.speed, 0.1..=4.0);
        ui::note(ui, "How fast time runs: for the particles, and for a field that changes with time.");
        ui::section(ui, "Background");
        if ui::choice(ui, &["none", "speed", "divergence", "curl"], &mut self.bg_mode) {
            self.bg_view = [f32::NAN; 4];
        }
        match self.bg_mode {
            1 => ui::note(ui, "Speed: dark where the flow is slow (it stops altogether at a fixed point), bright where it is fast."),
            2 => ui::note(ui, "Divergence: red where the flow spreads out (a source), blue where it converges (a sink)."),
            3 => ui::note(ui, "Curl: red where the flow turns anticlockwise, blue where it turns clockwise."),
            _ => {}
        }
        ui.checkbox(&mut self.arrows, "direction arrows");
        if ui.button("clear traced paths").clicked() {
            self.probes.clear();
        }

        ui::section(ui, "Reading the picture");
        let aurora = Cmap::Aurora.table();
        let mut key = vec![
            (aurora[table_index(0.3)], "dark streaks: slow particles (each streak fades behind its head)"),
            (aurora[table_index(1.0)], "pale, bright streaks: fast particles"),
            (pal::WHITE, "white line: a path you clicked (with a dot on it: the particle travelling it)"),
        ];
        match self.bg_mode {
            1 => key.extend([(Cmap::Ocean.color(0.1), "dark background: slow flow"), (Cmap::Ocean.color(0.6), "bright background: fast flow")]),
            2 | 3 => key.extend([(Cmap::CoolWarm.color(1.0), if self.bg_mode == 2 { "red: spreading out" } else { "red: turning anticlockwise" }), (Cmap::CoolWarm.color(0.0), if self.bg_mode == 2 { "blue: gathering in" } else { "blue: turning clockwise" })]),
            _ => {}
        }
        if self.arrows {
            key.push((pal::WHITE.a(0.45), "arrows: which way the flow goes there"));
        }
        ui::key(ui, &key);
        ui::section(ui, "Words");
        ui::term(ui, "Fixed point:", "a place where the velocity is zero, so that a particle there stays put.");
        ui::term(ui, "Divergence:", "how fast the flow spreads out from a point (positive) or gathers into it (negative).");
        ui::term(ui, "Curl:", "twice the rate at which a tiny paddle wheel placed at a point would turn: positive anticlockwise.");
        ui::see_also(ui, &[("descent", "gradient descent follows the field that points downhill"), ("surface", "the downhill field of a surface"), ("attractor", "flows in three dimensions that never settle"), ("expm", "the fields that are a matrix times the position, and where they carry the plane")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let (lo, hi) = f.cam.flat_bounds();
        let size = hi - lo;
        if self.pos.len() != self.count {
            self.scatter(lo, hi);
        }

        if self.bg_mode > 0 {
            self.update_backdrop(f.dt, lo, hi);
            if let Some(bg) = &mut self.bg {
                f.surface(bg);
            }
        }

        if self.arrows {
            let cols = 26;
            let cell = size.x / cols as f32;
            let rows = (size.y / cell).ceil() as i32;
            for j in 0..rows {
                for i in 0..cols {
                    let c = lo + Vec2::new(i as f32 + 0.5, j as f32 + 0.5) * cell;
                    let d = self.field_at(c, self.clock).normalize_or_zero() * cell * 0.36;
                    if d != Vec2::ZERO {
                        f.arrow(c - d, c + d, 1.2, pal::WHITE.a(0.45));
                    }
                }
            }
        }

        // (before this frame's steps, so that it starts at the time the experiment set)
        if let Some(at) = self.release.take() {
            self.trace(at, lo, hi);
        }
        // by scene time owed, so that a streak is the same stretch of motion on any
        // display, and never more than the frame has time for
        let due = steps_due(&mut self.owed, f.dt);
        let budget = crate::clock::frame_budget(f.real_dt, 0.008);
        for k in 0..due {
            if k > 0 && budget.spent() {
                break;
            }
            self.step(lo, hi);
            if let Some(trails) = &mut self.trails {
                trails.commit(f.gfx);
            }
        }
        if let Some(trails) = &mut self.trails {
            f.trails(trails);
        }

        // (a click anywhere traces a path, so both clicks of a double-click are the
        // scene's: R still resets the view)
        f.handles.claim_clicks();
        if f.clicked_canvas() {
            if let Some(m) = f.mouse() {
                self.trace(m, lo, hi);
            }
        }
        // a released particle's path is only of interest until it has travelled it
        let now = self.clock;
        self.probes.retain(|p| p.times.last().is_none_or(|&end| end > now));
        for probe in &self.probes {
            f.dl.lines.polyline(probe.path.iter().copied(), 2.6, pal::WHITE);
            if let Some(at) = probe.at(now) {
                f.glow(at, 22.0, pal::WHITE.a(0.6));
                f.dot(at, 10.0, pal::WHITE);
            }
        }
        // the axis numbers on dark plates, and brighter: the streaks run straight through them otherwise
        let first = f.dl.labels.len();
        f.axes2();
        for label in &mut f.dl.labels[first..] {
            label.backed = true;
            label.color = pal::TEXT;
        }
        // what the background shows, where it can be read without the panel (at the top:
        // the application's hint about dragging sits at the bottom for a while)
        let says = match self.bg_mode {
            1 => "background: speed (dark where slow)",
            2 => "background: divergence (red spreading out, blue gathering)",
            3 => "background: curl (red anticlockwise, blue clockwise)",
            _ => "",
        };
        if !says.is_empty() && text_width(says, 12.0) + 36.0 <= f.view_size().x {
            f.caption_px(Vec2::new(18.0, f.top_inset + 14.0), says, 12.0, pal::DIM, Align2::LEFT_TOP);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every field's note is typeset as written (the lint and the scene check read
    /// only the fields the scene opens with and its things to try set up).
    #[test]
    fn every_note_is_typeset_as_written() {
        for (text, shown) in FIELD_NOTES.iter().zip(&FIELD_PRESETS) {
            let found = crate::typeset::inspect(text);
            assert!(found.error.is_none() && found.words.is_empty(), "{}: {found:?}", shown.0);
        }
    }

    /// The formulas of preset `i`, compiled.
    fn preset(i: usize) -> (Expr, Expr) {
        let vars = &["x", "y", "t"];
        (Expr::parse(FIELD_PRESETS[i].1, vars).unwrap(), Expr::parse(FIELD_PRESETS[i].2, vars).unwrap())
    }

    /// The view a preset opens with on a canvas `aspect` times as wide as it is high.
    fn view_of(scene: &VectorField, aspect: f32) -> (Vec2, Vec2) {
        let half = Vec2::new(scene.half_height * aspect, scene.half_height);
        (-half, half)
    }

    /// Points carried along by the scene's own step (`Points::midpoint`, steps of
    /// `ROW_EVERY` at speed 1) for `time` of the field's time from time `t`.
    fn carry(p: &Expr, q: &Expr, starts: &[Vec2], t: f32, time: f32) -> Vec<Vec2> {
        let mut pts = Points::default();
        let mut at: Vec<Vec2> = starts.to_vec();
        for v in [&mut pts.xs, &mut pts.ys, &mut pts.us, &mut pts.vs] {
            v.resize(at.len(), 0.0);
        }
        let steps = (time / ROW_EVERY).round() as usize;
        for k in 0..steps {
            for (i, a) in at.iter().enumerate() {
                (pts.xs[i], pts.ys[i]) = (a.x, a.y);
            }
            pts.midpoint(p, q, t + k as f32 * ROW_EVERY, ROW_EVERY);
            for (i, a) in at.iter_mut().enumerate() {
                *a += Vec2::new(pts.us[i], pts.vs[i]) * ROW_EVERY;
            }
        }
        at
    }

    /// A streak is the same stretch of scene time on any display: one step and one
    /// row per `ROW_EVERY`, the remainder kept for the next frame. (Rows used to be
    /// one a frame: a quarter of the length at 240 Hz, twice it at 30.)
    #[test]
    fn streaks_last_the_same_time_at_any_frame_rate() {
        for hz in [30.0f32, 60.0, 72.0, 144.0, 240.0, 15.0] {
            let (mut owed, mut steps) = (0.0, 0);
            let frames = (hz * 20.0) as usize;
            for _ in 0..frames {
                steps += steps_due(&mut owed, 1.0 / hz);
            }
            // twenty seconds: 1200 rows, give or take the one in hand
            assert!((steps as i64 - 1200).abs() <= 1, "{hz} Hz: {steps} steps in twenty seconds");
        }
        // sixty frames a second: exactly one a frame, as before
        let mut owed = 0.0;
        assert!((0..600).all(|_| steps_due(&mut owed, 1.0 / 60.0) == 1));
        // the app's time speed: four times as many
        let mut owed = 0.0;
        assert!((0..600).all(|_| steps_due(&mut owed, 4.0 / 60.0) == 4));
        // a stall is not chased: at most MOST_STEPS, and nothing more is owed
        let mut owed = 0.0;
        assert_eq!(steps_due(&mut owed, 5.0), MOST_STEPS);
        assert!(owed < ROW_EVERY);
        assert_eq!(steps_due(&mut owed, 0.0), 0);
    }

    /// The divergence and curl from differences agree with the exact ones (from dual
    /// numbers) for every preset, all over its view; and the speed is the speed.
    #[test]
    fn the_backdrop_matches_the_exact_derivatives() {
        for (i, shown) in FIELD_PRESETS.iter().enumerate() {
            let (p, q) = preset(i);
            let half = Vec2::new(shown.3 * 1.6, shown.3);
            let (lo, hi) = (-half, half);
            let mut b = Backdrop::default();
            b.grid(lo, hi, 37, 23);
            let t = 0.7;
            let h = difference_step(lo, hi);
            let mut worst = [0.0f32; 3];
            for mode in 1..=3 {
                b.work_out(&p, &q, mode, t, h);
                for k in 0..b.xs.len() {
                    let vars = [Dual::var_x(b.xs[k]), Dual::var_y(b.ys[k]), Dual::cst(t)];
                    let (a, c) = (p.eval_num::<Dual>(&vars), q.eval_num::<Dual>(&vars));
                    let exact = match mode {
                        1 => a.v.hypot(c.v),
                        2 => a.dx + c.dy,
                        _ => c.dx - a.dy,
                    };
                    let error = (b.values[k] - exact).abs() / (1.0 + exact.abs());
                    worst[mode - 1] = worst[mode - 1].max(error);
                }
            }
            println!("{:<14} worst relative error: speed {:.1e}, divergence {:.1e}, curl {:.1e}", shown.0, worst[0], worst[1], worst[2]);
            assert!(worst[0] < 1e-6 && worst[1] < 1e-3 && worst[2] < 1e-3, "{}: {worst:?}", shown.0);
        }
    }

    /// The pendulum's note: every particle comes to rest in one of the eyes at x = 0,
    /// ±2π, ±4π, ..., the faster ones after more turns over the top; none stays at a
    /// pinch point x = ±π. (It once said everything ends at the origin, but the eyes at
    /// ±2π are in a wide view too, and the fastest particles go further still.)
    #[test]
    fn the_pendulum_comes_to_rest_in_the_eyes() {
        let (p, q) = preset(3);
        let starts: Vec<Vec2> = (0..21).flat_map(|i| (0..13).map(move |j| Vec2::new(-6.6 + 0.66 * i as f32, -4.0 + 0.667 * j as f32))).collect();
        let ends = carry(&p, &q, &starts, 0.0, 90.0);
        let mut eyes = [0; 9];
        for (s, e) in starts.iter().zip(&ends) {
            let k = (e.x / TAU).round();
            let off = (*e - Vec2::new(k * TAU, 0.0)).length();
            assert!(off < 0.02 && k.abs() <= 4.0, "from {s:?} to {e:?}");
            eyes[(k + 4.0) as usize] += 1;
        }
        println!("ends at x = 2πk for k = -4..=4: {eyes:?}");
        // most stay within a turn; some are fast enough for two or more
        assert!(eyes[3] > 0 && eyes[5] > 0 && eyes[4] > eyes[3] && eyes[4] > eyes[5]);
        assert!(eyes[..3].iter().sum::<i32>() > 0 && eyes[6..].iter().sum::<i32>() > 0);
        // fast enough to swing over the top: from the bottom at speed 3 it comes to rest a turn on
        let e = carry(&p, &q, &[Vec2::new(0.0, 3.0), Vec2::new(0.0, 1.5)], 0.0, 90.0);
        assert!((e[0] - Vec2::new(TAU, 0.0)).length() < 0.02 && e[1].length() < 0.02, "{e:?}");
    }

    /// Where the field does not change with time, a click draws the path that
    /// particles really follow (carried by the scene's own step).
    #[test]
    fn a_click_draws_the_path_particles_follow() {
        for (i, start) in [(3usize, Vec2::new(1.0, 1.0)), (4, Vec2::new(-3.0, 3.0)), (5, Vec2::new(0.5, 0.5))] {
            let mut scene = VectorField::create();
            scene.load(i);
            let (lo, hi) = view_of(&scene, 1.6);
            let size = hi - lo;
            let path = scene.streamline(start, lo - size, hi + size);
            let (p, q) = preset(i);
            let mut worst = 0.0f32;
            for time in [0.5f32, 1.0, 2.0, 4.0, 6.0] {
                let at = carry(&p, &q, &[start], 0.0, time)[0];
                let gap = path.windows(2).map(|w| distance_to_segment(at, w[0].truncate(), w[1].truncate())).fold(f32::MAX, f32::min);
                worst = worst.max(gap);
            }
            println!("{}: a particle strays at most {worst} from the drawn path", FIELD_PRESETS[i].0);
            assert!(worst < 0.008, "{}: a particle strays {worst} from the drawn path", FIELD_PRESETS[i].0);
        }
    }

    fn distance_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
        let d = b - a;
        let u = ((p - a).dot(d) / d.length_squared().max(1e-12)).clamp(0.0, 1.0);
        (p - (a + d * u)).length()
    }

    /// Where it does change with time, the marker on a drawn path is where a particle
    /// released there is (carried by the scene's own step); and a particle released
    /// at the same point later goes elsewhere.
    #[test]
    fn a_released_particle_travels_its_path() {
        for (i, start) in [(7usize, Vec2::new(0.9, 0.5)), (8, Vec2::new(1.0, -0.5))] {
            let mut scene = VectorField::create();
            scene.load(i);
            assert!(scene.time_dependent());
            let (lo, hi) = view_of(&scene, 1.6);
            let size = hi - lo;
            let probe = scene.pathline(start, 0.3, lo - size, hi + size);
            let (p, q) = preset(i);
            let mut worst = 0.0f32;
            for time in [1.0f32, 3.0, 6.0] {
                let particle = carry(&p, &q, &[start], 0.3, time)[0];
                let marker = probe.at(0.3 + time).map(|m| m.truncate());
                worst = worst.max(marker.map_or(f32::MAX, |m| (m - particle).length()));
            }
            println!("{}: the marker is at most {worst} from the particle", FIELD_PRESETS[i].0);
            assert!(worst < 0.002, "{}: the marker is {worst} from the particle", FIELD_PRESETS[i].0);
            let later = scene.pathline(start, 1.3, lo - size, hi + size);
            let apart = (later.at(1.3 + 3.0).unwrap().truncate() - probe.at(0.3 + 3.0).unwrap().truncate()).length();
            println!("{}: released a second later, {apart} apart after three", FIELD_PRESETS[i].0);
            assert!(apart > 1.0, "{}: released a second later, only {apart} apart after three", FIELD_PRESETS[i].0);
        }
    }

    /// The experiments do what their captions say.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // a source and a sink: red at (-1, 0), blue at (1, 0), fading farther out
        let mut scene = VectorField::create();
        scene.set_up(0);
        assert_eq!(scene.bg_mode, 2);
        let (lo, hi) = view_of(&scene, 1.6);
        let h = difference_step(lo, hi);
        let mut b = Backdrop::default();
        b.grid(lo, hi, 144, 96);
        b.work_out(&scene.p.expr, &scene.q.expr, 2, 0.0, h);
        let scale = b.scale();
        let far = (0..b.xs.len())
            .filter(|&k| [-1.0f32, 1.0].iter().all(|&x| (Vec2::new(b.xs[k], b.ys[k]) - Vec2::new(x, 0.0)).length() > 1.0))
            .map(|k| b.values[k].abs())
            .fold(0.0, f32::max);
        let mut poles = Backdrop { xs: vec![-1.0, 1.0], ys: vec![0.0, 0.0], ..Default::default() };
        poles.work_out(&scene.p.expr, &scene.q.expr, 2, 0.0, h);
        assert!(poles.values[0] > scale && poles.values[1] < -scale, "{:?} against {scale}", poles.values);
        assert!(far < 0.15 * scale, "beyond 1 of the poles: {far} against {scale}");

        // the vortex: the same curl everywhere
        scene.set_up(1);
        assert_eq!(scene.bg_mode, 3);
        b.work_out(&scene.p.expr, &scene.q.expr, 3, 0.0, h);
        assert!(b.values.iter().all(|v| (v - 2.0).abs() < 1e-3), "{:?}", b.values.iter().fold((f32::MAX, f32::MIN), |m, v| (m.0.min(*v), m.1.max(*v))));

        // the loop: from near the centre and from outside, onto the same loop
        scene.set_up(2);
        let (lo, hi) = view_of(&scene, 1.6);
        let size = hi - lo;
        let (p, q) = preset(4);
        let on_loop = carry(&p, &q, &[Vec2::new(0.5, 0.0)], 0.0, 60.0)[0];
        let cycle = scene.streamline(on_loop, lo - size, hi + size);
        let reach = |path: &[Vec3]| path[path.len() - 150..].iter().map(|v| v.x.abs()).fold(0.0, f32::max);
        let widest = reach(&cycle);
        assert!(scene.release == Some(Vec2::new(0.2, 0.0)));
        for start in [Vec2::new(0.2, 0.0), Vec2::new(-3.0, 3.0)] {
            let path = scene.streamline(start, lo - size, hi + size);
            assert!((reach(&path) - widest).abs() < 0.02, "from {start:?}: {} against the loop's {widest}", reach(&path));
        }
        let inside = scene.streamline(Vec2::new(0.2, 0.0), lo - size, hi + size);
        let middle = inside.iter().position(|v| (v.truncate() - Vec2::new(0.2, 0.0)).length() < 1e-6).unwrap_or(0);
        assert!(inside[..middle].iter().all(|v| v.truncate().length() < 0.21), "backwards from near the centre it stays near the centre");

        // the gyres: the released particle leaves its cell for the next
        scene.set_up(3);
        assert!(scene.clock == 0.0 && scene.time_dependent());
        let (lo, hi) = view_of(&scene, 1.6);
        let size = hi - lo;
        let at = scene.release.unwrap();
        let probe = scene.pathline(at, scene.clock, lo - size, hi + size);
        let left = probe.path.iter().zip(&probe.times).find(|(v, _)| !(0.0..=1.0).contains(&v.x) || !(0.0..=1.0).contains(&v.y));
        assert!(left.is_some_and(|(_, &t)| t < 3.0), "{left:?}");
    }
}

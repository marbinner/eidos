//! Spreading directions apart: n lines through the origin, placed so that the closest
//! two are as far from each other as they can be.

use super::*;
use crate::clock::Pace;
use crate::math::packing::{best_known, coherence, orthoplex, welch, Spread};
use glam::DVec3;

pub const DEF: SceneDef = SceneDef {
    id: "packing",
    title: "Spreading directions apart",
    group: "Linear algebra: data and dimensions",
    blurb: r"Place $n$ directions in space so that the largest $|\text{dot product}|$ between any two is as small as it can be.",
    make: |_| Box::new(Packing::create()),
};

/// The most lines the scene offers.
const MOST_LINES: usize = 32;

/// Pairs whose angle is within this many degrees of the smallest count as "the closest".
const CLOSE: f64 = 0.05;

/// For 2 to 32 lines in space: a start (sharpness of the soft maximum, seed) whose run
/// ends at the arrangement of the table in `math/packing.rs`, or a better one. Found
/// by the search `find_starts` there (the softest start that works, then the lowest
/// seed); `every_start_ends_at_the_tables_arrangement` below holds them to it.
const STARTS: [(f64, u64); MOST_LINES - 1] = [
    (2.0, 0), (2.0, 0), (2.0, 0), (2.0, 0), (2.0, 0), (2.0, 0), (2.0, 0), (2.0, 0),
    (2.0, 0), (2.0, 0), (2.0, 0), (2.0, 0), (32.0, 278), (512.0, 3), (2.0, 0), (2.0, 0),
    (2.0, 1), (2.0, 8), (2.0, 0), (2.0, 1), (2.0, 0), (512.0, 47), (2.0, 0), (512.0, 0),
    (2.0, 0), (2.0, 0), (2.0, 0), (512.0, 2), (512.0, 137), (512.0, 9), (512.0, 31),
];

/// How closely the picture follows the run, in seconds of the scene's time. The run
/// moves in steps, and not the same number of them each frame (90 a second on a
/// display that shows 60 is one, two, one, two), nor are its steps equally long;
/// drawn as they come, the lines jerk. So the picture follows the run through two
/// delays of this length, one after the other, which evens the steps out and puts
/// it about a twelfth of a second behind.
const FOLLOW: f32 = 0.04;
/// The picture has caught up when none of its lines is further than this from where
/// the run has it (as a distance between ends of length 1: a hundredth of a pixel).
const CAUGHT_UP: f64 = 2e-5;

const LINE: Color = pal::CYAN;
const CLOSEST: Color = pal::ORANGE;
const TABLE: Color = pal::GREEN;
const FLOOR: Color = pal::PINK;

/// How a run begins.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Start {
    /// From the start in `STARTS` (in the plane any start will do).
    Best,
    /// From random directions, with every pair pushing on every other at first.
    Soft(u64),
}

/// Experiments: (button, what to look for, lines, dimensions, start).
const TRIALS: [(&str, &str, usize, usize, Start); 6] = [
    (
        "four lines: a cube",
        r"Four lines settle with every pair the same 70.53° apart: all six $|\text{dot products}|$ are $1/3$. Their eight ends are the corners of a cube, and the orange lines are its edges. The floor is reached: no four directions do better.",
        4,
        3,
        Start::Best,
    ),
    (
        "six lines: an icosahedron",
        r"Six lines fit with every pair 63.43° apart: all fifteen $|\text{dot products}|$ are $1/\sqrt{5} = 0.4472$. The twelve ends are the corners of an icosahedron: five orange edges meet at each. The floor is reached again.",
        6,
        3,
        Start::Best,
    ),
    (
        "five lines: no better than six",
        "Taking one line away gains nothing: five lines also end 63.43° apart, as five of the icosahedron's six. The floor for five lines, 0.4082, is out of reach.",
        5,
        3,
        Start::Best,
    ),
    (
        "seven lines",
        r"The seventh line costs a lot: 54.74°, a $|\text{dot product}|$ of $1/\sqrt{3} = 0.5774$. The arrangement is the four diagonals of a cube and the three lines through the middles of its faces. Each of those three is equally close to all four diagonals: twelve closest pairs.",
        7,
        3,
        Start::Best,
    ),
    (
        "fourteen lines: a dead end",
        "This start settles with the closest lines 38.51° apart, and no small change improves that. Yet it is not the best: the table's arrangement for fourteen lines manages 38.68°. Press \"best start\" to see that one. Dead ends like this are what make the problem hard, and \"scatter\" lands in this one nearly every time.",
        14,
        3,
        Start::Soft(0),
    ),
    (
        "in the plane",
        r"Five lines in a plane: the best is to share the half turn out evenly, 36° between neighbours, a $|\text{dot product}|$ of $\cos 36^\circ = 0.809$. In the plane the answer is always $180^\circ/n$. It is in space that the answers become surprising.",
        5,
        2,
        Start::Best,
    ),
];

/// What the picture and the read-outs say about an arrangement.
struct Reading {
    /// The largest |dot product|, and the angle it is the cosine of, in degrees.
    mu: f64,
    angle: f64,
    /// The closest pairs: (i, j, their dot product).
    close: Vec<(usize, usize, f64)>,
}

fn degrees(dot: f64) -> f64 {
    dot.abs().min(1.0).acos().to_degrees()
}

fn reading(v: &[DVec3]) -> Reading {
    let mu = coherence(v);
    let angle = degrees(mu);
    let mut close = Vec::new();
    for i in 0..v.len() {
        for j in i + 1..v.len() {
            let c = v[i].dot(v[j]);
            if degrees(c) - angle < CLOSE {
                close.push((i, j, c));
            }
        }
    }
    Reading { mu, angle, close }
}

/// The pairs of ends joined for a closest pair of lines: each end of the one with the
/// nearer end of the other (at a right angle both are as near, and all four are joined).
fn joined(v: &[DVec3], (i, j, c): (usize, usize, f64)) -> Vec<(DVec3, DVec3)> {
    let signs: &[f64] = if c.abs() < 1e-4 {
        &[1.0, -1.0]
    } else if c > 0.0 {
        &[1.0]
    } else {
        &[-1.0]
    };
    signs.iter().flat_map(|s| [(v[i], v[j] * *s), (-v[i], v[j] * -*s)]).collect()
}

/// Where the screen panels go on a canvas of size `view` (logical pixels): in a column
/// on the right, or, on a narrow canvas, below the picture. Each is its top-left
/// corner and its size.
struct Panels {
    narrow: bool,
    /// Where the part kept for the panels begins on a narrow canvas.
    below: f32,
    heat: (Vec2, f32),
    chart: (Vec2, Vec2),
}

impl Panels {
    fn of(view: Vec2, inset: f32) -> Panels {
        if view.x < view.y * 0.95 && view.y >= 560.0 {
            let below = (view.y * 0.56).max(300.0);
            let side = ((view.x - 96.0) * 0.46).min(view.y - below - 84.0).max(40.0);
            let top = below + 40.0;
            Panels { narrow: true, below, heat: (Vec2::new(24.0, top), side), chart: (Vec2::new(24.0 + side + 36.0, top), Vec2::new((view.x - side - 84.0).max(40.0), side)) }
        } else {
            // the table in the top right corner and the chart in the bottom right
            // one, which the round picture in the middle leaves free
            let side = (view.y * 0.27).min(view.x * 0.18).max(40.0).min(240.0);
            let top = 48.0f32.max(inset + 38.0);
            let x = view.x - side - 36.0;
            let chart = Vec2::new(side, (view.y * 0.17).max(30.0).min(150.0).min((view.y - top - side - 150.0).max(20.0)));
            Panels { narrow: false, below: view.y, heat: (Vec2::new(x, top), side), chart: (Vec2::new(x, view.y - chart.y - 30.0), chart) }
        }
    }
}

pub struct Packing {
    spread: Spread,
    /// The arrangement as it is drawn, which follows the run's (see `FOLLOW`), and
    /// the stage between the two.
    shown: Vec<DVec3>,
    between: Vec<DVec3>,
    n: usize,
    dim: usize,
    /// How many times "scatter" has been pressed (each press is a new start).
    scatters: u64,
    running: bool,
    /// Steps per second asked for.
    rate: usize,
    turning: bool,
    pace: Pace,
    /// The largest |dot product| after each step of the run.
    history: History,
    ball: Option<Surface>,
    /// The size of the canvas last drawn on, and how much of its top the
    /// application's buttons cover (they decide where the panels and the picture go).
    canvas: Vec2,
    inset: f32,
    new_view: bool,
}

impl Packing {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let mut scene = Packing {
            spread: Spread::new(6, 3, 0),
            shown: Vec::new(),
            between: Vec::new(),
            n: 6,
            dim: 3,
            scatters: 0,
            running: true,
            rate: 90,
            turning: true,
            pace: Pace::default(),
            history: History::new(600),
            ball: None,
            canvas: Vec2::new(980.0, 940.0),
            inset: 0.0,
            new_view: false,
        };
        scene.begin(6, 3, Start::Best);
        scene
    }

    /// Start a run of `n` lines in `dim` dimensions.
    fn begin(&mut self, n: usize, dim: usize, start: Start) {
        let n = n.clamp(2, MOST_LINES);
        self.new_view |= dim != self.dim;
        (self.n, self.dim) = (n, dim);
        self.spread = match start {
            Start::Best if dim == 3 => {
                let (sharpness, seed) = STARTS[n - 2];
                Spread::start(n, 3, seed, sharpness)
            }
            Start::Best => Spread::new(n, dim, 0),
            Start::Soft(seed) => Spread::new(n, dim, seed),
        };
        self.catch_up();
        self.history.clear();
        self.history.push(self.spread.coherence() as f32);
        self.pace.reset();
        self.running = true;
    }

    /// Draw the arrangement exactly where the run has it.
    fn catch_up(&mut self) {
        self.shown.clone_from(&self.spread.v);
        self.between.clone_from(&self.spread.v);
    }

    fn caught_up(&self) -> bool {
        self.shown == self.spread.v
    }

    /// Bring the picture `dt` seconds of the scene's time nearer to the run.
    fn follow(&mut self, dt: f32) {
        let v = &self.spread.v;
        if self.shown.len() != v.len() || self.between.len() != v.len() {
            return self.catch_up();
        }
        if dt <= 0.0 || self.caught_up() {
            return;
        }
        let k = 1.0 - (-(dt / FOLLOW) as f64).exp();
        let mut apart = 0.0f64;
        for i in 0..v.len() {
            // (the ends stay on the ball: a line never turns far enough in a frame
            // for the way between to pass near the centre, and if it did it would
            // be put where the run has it)
            self.between[i] = (self.between[i] + (v[i] - self.between[i]) * k).try_normalize().unwrap_or(v[i]);
            self.shown[i] = (self.shown[i] + (self.between[i] - self.shown[i]) * k).try_normalize().unwrap_or(v[i]);
            apart = apart.max((self.shown[i] - v[i]).length()).max((self.between[i] - v[i]).length());
        }
        // (exactly there in the end, not nearly: the picture then stops changing)
        if apart < CAUGHT_UP {
            self.catch_up();
        }
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let (_, _, n, dim, start) = TRIALS[k];
        self.begin(n, dim, start);
    }

    /// One step of the run; false once it has settled.
    fn step(&mut self) -> bool {
        let going = self.spread.step();
        if going {
            self.history.push(self.spread.coherence() as f32);
        }
        going
    }

    /// The table's value for this many lines (as a |dot product|), and whether it is
    /// proved to be the lowest possible.
    fn table(&self) -> Option<(f64, bool)> {
        best_known(self.n, self.dim)
    }

    /// The floor under every arrangement of this many lines: the higher of the two bounds.
    fn floor(&self) -> f64 {
        welch(self.n, self.dim).max(orthoplex(self.n, self.dim))
    }

    /// What to say about where the run stands, against the table.
    fn verdict(&self) -> String {
        if !self.spread.settled() {
            return if self.running { format!("Settling: step {}.", self.spread.steps()) } else { "Paused.".into() };
        }
        let Some((best, proved)) = self.table() else { return "At rest: no small change improves this arrangement.".into() };
        let (found, wanted) = (degrees(self.spread.coherence()), degrees(best));
        let source = if self.dim == 2 || self.n <= self.dim { "the best possible" } else { "the table's arrangement" };
        if found > wanted + 0.0005 {
            format!("At rest, {:.3}° wider than the table's arrangement: this one is better than the 1996 table's.", found - wanted)
        } else if found > wanted - 0.0005 {
            format!("At rest, at {source}{}.", if proved && self.dim == 3 && self.n > self.dim { ", which is proved to be the best possible" } else { "" })
        } else {
            format!("At rest in a dead end: no small change improves it, but it is {:.2}° short of {source}. Another start does better.", wanted - found)
        }
    }

    fn the_view(&self) -> View {
        let panels = Panels::of(self.canvas, self.inset);
        if panels.narrow {
            // the picture has the part of the canvas above the panels (and below the
            // application's buttons) to itself: the origin goes in the middle of it
            let (top, region) = (52.0, (panels.below - 52.0).max(120.0));
            let per_unit = (self.canvas.x / 2.7).min(region / 2.7);
            let lift = (self.canvas.y * 0.5 - (top + region * 0.5)) / per_unit;
            if self.dim == 2 {
                View::Fit { center: Vec2::new(0.0, -lift), half: Vec2::new(0.5, self.canvas.y * 0.5 / per_unit) }
            } else {
                // (0.32 is the tangent of half the camera's field of view)
                let dist = self.canvas.y * 0.5 / per_unit / 0.32;
                View::Orbit { target: Vec3::new(0.0, 0.0, -lift / 0.93), yaw: 0.7, pitch: 0.38, dist }
            }
        } else {
            // the circle (or the ball) in the middle, small enough to stay clear of
            // the two panels in the corners on the right
            let per_unit = (self.canvas.y * 0.3).min(self.canvas.x * 0.27).max(20.0);
            if self.dim == 2 {
                View::Fit { center: Vec2::ZERO, half: Vec2::new(0.5, self.canvas.y * 0.5 / per_unit) }
            } else {
                View::Orbit { target: Vec3::ZERO, yaw: 0.7, pitch: 0.38, dist: self.canvas.y * 0.5 / per_unit / 0.32 }
            }
        }
    }

    /// The see-through ball the ends lie on: it dims what is behind it, which is what
    /// tells the near side of the picture from the far side.
    fn ball(&mut self, f: &mut Frame) {
        let ball = self.ball.get_or_insert_with(|| {
            let mut s = Surface::new(f.gfx, 64, 33);
            s.fill(|u, v| {
                let (lon, lat) = (u * TAU, (v - 0.5) * PI);
                [0.985 * lat.cos() * lon.cos(), 0.985 * lat.cos() * lon.sin(), 0.985 * lat.sin(), 0.0]
            });
            s.style.cmap = None;
            s.style.color = Color::hex(0x1B2740);
            s.style.opacity = 0.4;
            s.style.specular = 0.1;
            s.style.rim = 0.5;
            s.style.wrap = [true, false];
            s
        });
        f.surface(ball);
    }

    /// The two screen panels: every pair's |dot product|, and the largest of them
    /// step by step.
    fn panels(&self, f: &mut Frame, at: &Panels, read: &Reading) {
        let (n, view) = (self.n, f.view_size());
        if at.narrow {
            hud_panel(f, Vec2::new(0.0, at.below), view);
        }
        let (o, side) = at.heat;
        if !at.narrow {
            hud_panel(f, o - Vec2::new(14.0, 34.0), o + Vec2::new(side + 14.0, side + 36.0));
        }
        let v = &self.shown;
        let close = |i: usize, j: usize| read.close.iter().any(|&(a, b, _)| (a, b) == (i.min(j), i.max(j)));
        hud_heat(f, o, Vec2::splat(side), n, n, |i, j| {
            if i == j {
                Color::hex(0x0E1320)
            } else if close(i, j) {
                CLOSEST
            } else {
                shade(v[i].dot(v[j]).abs() as f32)
            }
        });
        let title = if side > 150.0 { r"$|v_i \cdot v_j|$ for every pair of lines" } else { r"$|v_i \cdot v_j|$, every pair" };
        f.text_px(o + Vec2::new(0.0, -10.0), title, 11.5, pal::TEXT, Align2::LEFT_BOTTOM);
        // what the shades stand for
        let y = o.y + side + 8.0;
        let ramp = (side * 0.42).min(90.0);
        for k in 0..16 {
            let x = |t: f32| o.x + 12.0 + ramp * t;
            hud_rect(f, Vec2::new(x(k as f32 / 16.0), y), Vec2::new(x((k + 1) as f32 / 16.0) + 0.5, y + 7.0), shade((k as f32 + 0.5) / 16.0));
        }
        f.text_px(Vec2::new(o.x, y + 3.5), "0", 10.0, pal::DIM, Align2::LEFT_CENTER);
        f.text_px(Vec2::new(o.x + ramp + 17.0, y + 3.5), "1", 10.0, pal::DIM, Align2::LEFT_CENTER);
        if side > 120.0 {
            hud_rect(f, Vec2::new(o.x + side - 62.0, y), Vec2::new(o.x + side - 52.0, y + 7.0), CLOSEST);
            f.text_px(Vec2::new(o.x + side, y + 3.5), "largest", 10.0, pal::DIM, Align2::RIGHT_CENTER);
        }

        // the largest |dot product| against the steps of the run, with the table's
        // value and the floor ruled across
        let (co, size) = at.chart;
        let points = self.history.points();
        let (table, floor) = (self.table().map(|t| t.0 as f32), self.floor() as f32);
        let low = points.iter().copied().chain(table).fold(1.0f32, f32::min);
        // the floor is ruled across if it is near where the run ends; far below (as it
        // is for many lines) it would squeeze the run into the top of the chart
        let hi = 1.0f32;
        let ruled = floor > low - 0.6 * (hi - low).max(0.02);
        // (with room under the lowest rule for its name)
        let lo = if ruled { low.min(floor) } else { low } - (hi - low).max(0.02) * 0.24;
        hud_chart(f, co, size, &[(points, LINE)], false, Some([lo, hi]));
        let y_of = |v: f32| co.y + size.y * (1.0 - ((v - lo) / (hi - lo)).clamp(0.0, 1.0));
        let named = size.x > 90.0;
        let rule = |f: &mut Frame, v: f32, name: &str, c: Color, above: bool| {
            hud_line(f, Vec2::new(co.x, y_of(v)), Vec2::new(co.x + size.x, y_of(v)), 1.0, c.a(0.85));
            if named {
                f.text_px(Vec2::new(co.x + 2.0, y_of(v) + if above { -2.0 } else { 3.0 }), name, 10.0, c, if above { Align2::LEFT_BOTTOM } else { Align2::LEFT_TOP });
            }
        };
        let best = if self.dim == 2 || self.n <= 7 { "best possible" } else { "table" };
        match table {
            // (where the two rules fall together, one name is enough)
            Some(t) if ruled && t - floor > 0.02 * (hi - lo) => {
                rule(f, floor, "floor", FLOOR, false);
                rule(f, t, best, TABLE, true);
            }
            Some(t) => rule(f, t, best, TABLE, true),
            None => rule(f, floor, "floor", FLOOR, false),
        }
        if !ruled && named {
            f.text_px(Vec2::new(co.x + 2.0, co.y + size.y - 3.0), format!("floor: {floor:.3}, below the chart"), 10.0, FLOOR, Align2::LEFT_BOTTOM);
        }
        let title = if size.x > 170.0 { r"largest $|\text{dot product}|$, step by step" } else { r"largest $|\text{dot}|$" };
        f.text_px(co + Vec2::new(0.0, -12.0), title, 11.5, pal::TEXT, Align2::LEFT_BOTTOM);
    }
}

/// The vectors as lines of text, one each: its coordinates to six decimals.
fn coordinates(v: &[DVec3], dim: usize) -> String {
    // (no "-0.000000")
    let tidy = |x: f64| if x.abs() < 5e-7 { 0.0 } else { x };
    v.iter().map(|v| if dim == 2 { format!("{:9.6} {:9.6}", tidy(v.x), tidy(v.y)) } else { format!("{:9.6} {:9.6} {:9.6}", tidy(v.x), tidy(v.y), tidy(v.z)) }).collect::<Vec<_>>().join("\n")
}

/// The shade of a |dot product| in the table of pairs: dark at 0 (a right angle),
/// light at 1 (the same line).
fn shade(t: f32) -> Color {
    Color::hex(0x161D2E).mix(Color::hex(0xB9D4F5), t.max(0.0).min(1.0).powf(1.4))
}

impl Scene for Packing {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn animated(&self) -> bool {
        (self.running && !self.spread.settled()) || !self.caught_up() || (self.dim == 3 && self.turning)
    }

    /// In space the arrangement is a solid: turn it slowly so its shape can be read.
    fn auto_rotate(&self) -> f32 {
        if self.dim == 3 && self.turning {
            0.12
        } else {
            0.0
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"How far apart can $n$ directions be kept? Each line through the centre is one direction: a vector of length 1 and its opposite, which count as the same here, since their dot products with anything differ only in sign. For vectors of length 1 the dot product is the cosine of the angle between them. So making the largest $|\text{dot product}|$ small means keeping the closest two lines as far apart as possible. The picture starts from random directions and moves them a little at a time, until no small change helps.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Lines");
        let mut space = self.dim - 2;
        if ui::choice(ui, &["in the plane", "in space"], &mut space) {
            self.begin(self.n, space + 2, Start::Best);
        }
        let mut n = self.n;
        if ui::slider_int(ui, "number of lines", &mut n, 2..=MOST_LINES) {
            self.begin(n, self.dim, Start::Best);
        }
        ui.horizontal(|ui| {
            if ui.button("best start").clicked() {
                self.begin(self.n, self.dim, Start::Best);
            }
            if ui.button("scatter").clicked() {
                self.scatters += 1;
                self.begin(self.n, self.dim, Start::Soft(1000 + self.scatters));
            }
            if ui.button(if self.running { "pause" } else { "run" }).clicked() {
                self.running = !self.running;
            }
        });
        ui::note(ui, "\"scatter\" throws the lines down at random and lets them settle: where they end depends on where they start, and some starts end in a dead end. \"best start\" uses a start that is known to end at the best arrangement found. You can also drag any end and let go.");
        ui::slider_int_log(ui, "steps per second", &mut self.rate, 5..=2000);
        if self.dim == 3 {
            ui.checkbox(&mut self.turning, "turn slowly");
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (LINE, "each line through the centre: one direction. Its two ends are the same direction, and either can be dragged"),
                (CLOSEST, "orange: the closest pairs of lines (within 0.05° of the smallest angle), each joined from end to nearest end: the edges of the solid the ends make. Also their cells in the table of pairs"),
                (shade(0.75), r"table of pairs: lighter means a larger $|\text{dot product}|$, the two lines closer together"),
                (TABLE, "green rule in the chart: the best possible value where that is known, and otherwise the value in the table of 1996"),
                (FLOOR, "pink rule: the floor, which no arrangement can go below"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Coherence:", r"the largest $|\text{dot product}|$ between two of a set of vectors of length 1. It is the cosine of the smallest angle between two of their lines.");
        ui::term(ui, "The floor:", r"the Welch bound $\sqrt{(n - d) / (d(n - 1))}$ for $n \ge d$ vectors in $d$ dimensions, which no set can go below (for $n \le d$ the floor is $0$: the lines can all be at right angles); and, for more than $d(d + 1)/2$ vectors, $1/\sqrt{d}$.");
        ui::term(ui, "Equiangular:", "every pair of lines at the same angle. Only arrangements like that reach the Welch bound, and they are rare: in space, those of two and three lines (at right angles, on the floor of $0$), and of four and six.");

        ui::section(ui, "Numbers");
        // (of the arrangement as it is drawn, like everything said on the picture)
        let read = reading(&self.shown);
        ui::readout_colored(ui, r"largest $|\text{dot product}|$", format!("{:.4}", read.mu), LINE);
        ui::readout(ui, "smallest angle between lines", format!("{:.2}°", read.angle));
        ui::readout_colored(ui, "pairs that close", format!("{} of {}", read.close.len(), self.n * (self.n - 1) / 2), CLOSEST);
        if let Some((best, _)) = self.table() {
            let label = if self.dim == 2 || self.n <= 7 { "best possible" } else { "table (best known in 1996)" };
            ui::readout_colored(ui, label, format!("{:.4}  ({:.2}°)", best, degrees(best)), TABLE);
        }
        ui::readout_colored(ui, "floor", format!("{:.4}", self.floor()), FLOOR);
        ui::note(ui, &self.verdict());
        if self.dim == 3 && self.n > 7 {
            ui::note(ui, "The table is from Conway, Hardin and Sloane, \"Packing lines, planes, etc.\" (1996). Up to seven lines its arrangements are proved to be the best possible; beyond that they are the best its authors found, which need not be the best there are.");
        }
        egui::CollapsingHeader::new("the vectors as numbers").id_salt("packing vectors").show(ui, |ui| {
            let listed = coordinates(&self.spread.v, self.dim);
            if ui.button("copy").clicked() {
                ui.ctx().copy_text(listed.clone());
            }
            ui.label(egui::RichText::new(listed).monospace().size(11.0));
            ui::note(ui, "One vector of length 1 for each line, as it stands now (its opposite would do as well). Turning all of them together changes none of the dot products, so the same arrangement can come out facing any way.");
        });
        ui::see_also(
            ui,
            &[
                ("vectors", "dot products and angles"),
                ("sensing", "compressed sensing: why columns that are far apart are easy to tell apart"),
                ("superposition", "a network packing more features than it has dimensions"),
                ("highdim", "random directions in many dimensions: nearly at right angles already"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        // the run: so many steps for each second of the scene's time, and none while
        // an end is being dragged
        if self.running && !self.spread.settled() && f.dt > 0.0 && !f.handles.dragging() {
            let mut turn = self.pace.begin(self.rate as f32, f.dt, f.real_dt, 0.004);
            while turn.next() {
                if !self.step() {
                    break;
                }
            }
            self.pace.end(turn);
        }
        // the picture follows the run, evenly however unevenly the steps come
        self.follow(f.dt);

        // the panels' places follow the canvas, and the picture's place follows theirs
        let view = f.view_size();
        let panels = Panels::of(view, f.top_inset);
        if panels.narrow != Panels::of(self.canvas, self.inset).narrow || (view - self.canvas).abs().max_element() > 24.0 {
            self.canvas = view;
            self.new_view = true;
        }
        self.inset = f.top_inset;

        let flat = self.dim == 2;
        let read = reading(&self.shown);
        let towards = f.cam.eye.normalize_or_zero();
        // how bright something at `p` is drawn: the far side of the ball fainter than
        // the near side, which is what lets a still picture be read as a solid
        let depth = |p: Vec3| if flat { 1.0 } else { 0.24 + 0.76 * (0.5 + 0.5 * p.normalize_or_zero().dot(towards)).powf(1.5) };

        if flat {
            f.circle(Vec3::ZERO, Vec3::Z, 1.0, 1.3, pal::DIM.a(0.7));
        } else {
            self.ball(f);
            // the ball's outline as the camera sees it
            let d = f.cam.eye.length().max(1.01);
            f.circle(towards / d, towards, (1.0 - 1.0 / (d * d)).sqrt(), 1.2, pal::DIM.a(0.6));
        }

        // the closest pairs, joined end to end: the edges of the solid the ends make
        for &pair in &read.close {
            for (a, b) in joined(&self.shown, pair) {
                let (a, b) = (a.as_vec3(), b.as_vec3());
                let bright = depth(a + b);
                f.line(a, b, 1.2 + 1.2 * bright, CLOSEST.a(bright));
            }
        }

        // the lines, each with a handle at either end
        let reach = if flat { 1.12 } else { 1.1 };
        let mut placed = None;
        for (i, v) in self.shown.iter().enumerate() {
            let v = v.as_vec3();
            for (k, end) in [v, -v].into_iter().enumerate() {
                let bright = depth(end);
                f.line(Vec3::ZERO, end * reach, 1.0 + 1.2 * bright, LINE.a(0.25 + 0.7 * bright));
                let mut p = end;
                let dot = LINE.mix(Color::hex(0x1B2740), 0.7 * (1.0 - bright));
                if f.drag3_sized((2 * i + k) as u32, &mut p, flat.then_some(Vec3::Z), dot, if self.n > 16 { 7.0 } else { 9.0 }) {
                    placed = Some((i, if k == 0 { p } else { -p }));
                }
            }
        }
        if let Some((i, to)) = placed {
            self.spread.place(i, to.as_dvec3());
            // (the line in the hand is drawn where the hand has it, at once)
            (self.shown[i], self.between[i]) = (self.spread.v[i], self.spread.v[i]);
            self.history.clear();
            self.history.push(self.spread.coherence() as f32);
        }
        f.dot(Vec3::ZERO, 5.0, pal::DIM);

        if flat && self.n <= 12 {
            // the angle between two neighbours, written in one of the closest gaps
            if let Some(&(i, j, c)) = read.close.first() {
                let (a, b) = (self.shown[i].as_vec3(), (self.shown[j] * if c >= 0.0 { 1.0 } else { -1.0 }).as_vec3());
                let mid = (a + b).normalize_or_zero();
                f.text(mid * 0.62, format!("{:.1}°", read.angle), 13.0, CLOSEST, Align2::CENTER_CENTER);
            }
        }

        // what the picture is, said where it is drawn
        let top = 14.0f32.max(f.top_inset + 8.0);
        let room = if panels.narrow { view.x } else { panels.heat.0.x - 30.0 } - 28.0;
        let long = format!(r"{} lines: the closest two are {:.2}° apart, $|\text{{dot product}}|$ {:.4}", self.n, read.angle, read.mu);
        let short = format!(r"{} lines: closest {:.2}°, $|\text{{dot}}|$ {:.4}", self.n, read.angle, read.mu);
        // (each measured once: measuring a formula means laying it out)
        let fits = |text: &str| text_width(text, 13.0) <= room;
        let (said, fit) = if fits(&long) {
            (long, true)
        } else {
            let fit = fits(&short);
            (short, fit)
        };
        if fit && view.y > 200.0 {
            f.caption_px(Vec2::new(14.0, top), said, 13.0, pal::TEXT, Align2::LEFT_TOP);
        }
        self.panels(f, &panels, &read);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, run until it has settled.
    fn run(k: usize) -> Packing {
        let mut scene = Packing::create();
        scene.set_up(k);
        settle(&mut scene);
        scene
    }

    fn settle(scene: &mut Packing) {
        for _ in 0..6000 {
            if !scene.step() {
                break;
            }
        }
        assert!(scene.spread.settled(), "{} lines did not settle", scene.n);
    }

    /// How many edges meet at each end of each line (both ends of a line have as many).
    fn edges_at_each_end(scene: &Packing) -> Vec<usize> {
        let read = reading(&scene.spread.v);
        (0..scene.n).map(|i| read.close.iter().filter(|&&(a, b, _)| a == i || b == i).map(|&p| joined(&scene.spread.v, p).len() / 2).sum()).collect()
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let two = |x: f64| format!("{x:.2}");
        let four = |x: f64| format!("{x:.4}");

        // four lines: every pair 70.53 degrees apart, all six |dot products| 1/3; the
        // eight ends are a cube's corners (three edges at each, and the edges at a
        // corner at right angles to one another); the floor is reached
        let s = run(0);
        let r = reading(&s.spread.v);
        assert!(two(r.angle) == "70.53" && r.close.len() == 6 && r.close.iter().all(|p| (p.2.abs() - 1.0 / 3.0).abs() < 1e-6), "{} {}", r.angle, r.close.len());
        assert_eq!(edges_at_each_end(&s), vec![3; 4]);
        let corner = s.spread.v[0];
        let edges: Vec<DVec3> = r.close.iter().filter(|p| p.0 == 0).map(|&(_, j, c)| s.spread.v[j] * c.signum() - corner).collect();
        assert!(edges.len() == 3 && (0..3).all(|a| (a + 1..3).all(|b| edges[a].dot(edges[b]).abs() < 1e-5)), "{edges:?}");
        assert!((r.mu - s.floor()).abs() < 1e-6 && s.verdict().contains("proved"));

        // six lines: all fifteen pairs 63.43 degrees apart, |dot product| 1/sqrt 5 =
        // 0.4472; five edges at each of the twelve ends; the floor is reached
        let six = run(1);
        let r = reading(&six.spread.v);
        assert!(two(r.angle) == "63.43" && four(r.mu) == "0.4472" && r.close.len() == 15 && (r.mu - 0.2f64.sqrt()).abs() < 1e-6);
        assert_eq!(edges_at_each_end(&six), vec![5; 6]);
        assert!((r.mu - six.floor()).abs() < 1e-6);

        // five lines: 63.43 degrees again, and they are five of the icosahedron's six:
        // a sixth line can be added at the same angle from all of them (it is the sum
        // of the five, with signs). The floor, 0.4082, is not reached
        let five = run(2);
        let r = reading(&five.spread.v);
        assert!(two(r.angle) == "63.43" && r.close.len() == 10 && four(five.floor()) == "0.4082" && r.mu > five.floor() + 0.03);
        let sixth = (0..32u32).map(|signs| (0..5).map(|i| five.spread.v[i] * if signs >> i & 1 == 1 { 1.0 } else { -1.0 }).sum::<DVec3>().normalize_or_zero()).any(|w| five.spread.v.iter().all(|v| (v.dot(w).abs() - r.mu).abs() < 1e-5));
        assert!(sixth, "the five lines are not five of an icosahedron's six");

        // seven lines: 54.74 degrees, 1/sqrt 3 = 0.5774; three lines at right angles to
        // one another (the cube's face lines), each as close to all of the other four
        // (its diagonals, which are 70.53 degrees from one another): twelve closest pairs
        let s = run(3);
        let r = reading(&s.spread.v);
        assert!(two(r.angle) == "54.74" && four(r.mu) == "0.5774" && r.close.len() == 12 && (r.mu - (1.0f64 / 3.0).sqrt()).abs() < 1e-6);
        let partners = |i: usize| r.close.iter().filter(|p| p.0 == i || p.1 == i).count();
        let (faces, diagonals): (Vec<usize>, Vec<usize>) = (0..7).partition(|&i| partners(i) == 4);
        assert!(faces.len() == 3 && diagonals.iter().all(|&i| partners(i) == 3), "{faces:?} {diagonals:?}");
        let dot = |a: usize, b: usize| s.spread.v[a].dot(s.spread.v[b]).abs();
        assert!(faces.iter().all(|&a| faces.iter().all(|&b| a == b || dot(a, b) < 1e-5)));
        assert!(diagonals.iter().all(|&a| diagonals.iter().all(|&b| a == b || two(degrees(dot(a, b))) == "70.53")));
        assert!(s.verdict().contains("proved") && two(degrees(reading(&six.spread.v).mu) - r.angle) == "8.70", "a seventh line costs a lot");

        // fourteen lines: this start settles at 38.51 degrees, short of the table's
        // 38.68, which the best start does reach; and scatter lands there nearly always
        let dead = run(4);
        let r = reading(&dead.spread.v);
        let (table, _) = dead.table().unwrap();
        assert!(two(r.angle) == "38.51" && two(degrees(table)) == "38.68" && dead.verdict().contains("dead end") && dead.verdict().contains("0.17°"), "{} | {}", r.angle, dead.verdict());
        let mut best = Packing::create();
        best.begin(14, 3, Start::Best);
        settle(&mut best);
        assert!(two(reading(&best.spread.v).angle) == "38.68" && best.verdict().contains("at the table's arrangement"), "{}", best.verdict());
        let mut scattered = Packing::create();
        (scattered.n, scattered.dim) = (14, 3);
        let stuck = (1..=30)
            .filter(|&k| {
                // (as the button does it)
                scattered.begin(14, 3, Start::Soft(1000 + k));
                settle(&mut scattered);
                (scattered.spread.coherence() - r.mu).abs() < 1e-5
            })
            .count();
        assert!(stuck >= 27, "{stuck} of 30 scattered starts end in the dead end");

        // in the plane: five lines 36 degrees from their neighbours, cos 36 = 0.809
        let s = run(5);
        let r = reading(&s.spread.v);
        assert!(s.dim == 2 && two(r.angle) == "36.00" && format!("{:.3}", r.mu) == "0.809" && r.close.len() == 5 && s.spread.v.iter().all(|v| v.z == 0.0));
        assert_eq!(edges_at_each_end(&s), vec![2; 5]);
    }

    /// "best start" ends at the table's arrangement for every number of lines the scene
    /// offers (or at a wider one), and in the plane at 180°/n. Where the panel says the
    /// best is known, that is also as low as the floor allows or within what is proved.
    #[test]
    fn every_start_ends_at_the_tables_arrangement() {
        let mut scene = Packing::create();
        for n in 2..=MOST_LINES {
            scene.begin(n, 3, Start::Best);
            settle(&mut scene);
            let (table, _) = scene.table().unwrap();
            let found = scene.spread.coherence();
            assert!(found < table + 1e-6, "{n} lines: {found} against the table's {table}");
            assert!(found > scene.floor() - 1e-9, "{n} lines: below the floor");
            let verdict = scene.verdict();
            assert!(verdict.contains("at the table's") || verdict.contains("best possible") || verdict.contains("wider"), "{n}: {verdict}");

            scene.begin(n, 2, Start::Best);
            settle(&mut scene);
            let even = (std::f64::consts::PI / n as f64).cos();
            assert!((scene.spread.coherence() - if n == 2 { 0.0 } else { even }).abs() < 1e-6, "{n} lines in the plane");
        }
        // 22 lines: the run ends wider than the 1996 table's arrangement, and says so
        scene.begin(22, 3, Start::Best);
        settle(&mut scene);
        assert!(scene.verdict().contains("0.009° wider"), "{}", scene.verdict());
    }

    /// "Only equiangular arrangements reach the Welch bound: in space, those of two and
    /// three lines (at right angles, on the floor of 0), and of four and six"; the
    /// other floor is reached by seven. (For n ≤ d the formula would take the root of
    /// a negative number, or of 0: the floor there is 0.)
    #[test]
    fn the_floor_is_reached_where_the_panel_says() {
        let mut scene = Packing::create();
        for n in 2..=16 {
            scene.begin(n, 3, Start::Best);
            settle(&mut scene);
            let r = reading(&scene.spread.v);
            assert!(n > 3 || welch(n, 3) == 0.0);
            let at_welch = (r.mu - welch(n, 3)).abs() < 1e-6;
            assert_eq!(at_welch, [2, 3, 4, 6].contains(&n), "{n} lines: {} against {}", r.mu, welch(n, 3));
            if at_welch {
                assert_eq!(r.close.len(), n * (n - 1) / 2, "{n} lines at the Welch bound are not equiangular");
            }
            assert_eq!((r.mu - scene.floor()).abs() < 1e-6, [2, 3, 4, 6, 7].contains(&n), "{n} lines");
        }
    }

    /// How far the ends moved in each frame of a run shown at `hz` frames a second
    /// (the furthest any end went): in the run's own arrangement, and in the picture.
    fn moves(scene: &mut Packing, hz: f32) -> (Vec<f64>, Vec<f64>) {
        let dt = 1.0 / hz;
        let (mut owed, mut run, mut picture) = (0.0f32, Vec::new(), Vec::new());
        let furthest = |a: &[DVec3], b: &[DVec3]| a.iter().zip(b).map(|(p, q)| (*p - *q).length()).fold(0.0, f64::max);
        for _ in 0..(hz as usize * 120) {
            let (run_was, picture_was) = (scene.spread.v.clone(), scene.shown.clone());
            // (the steps as `Pace` deals them out: what is owed, whole steps only)
            owed += scene.rate as f32 * dt;
            while owed >= 1.0 {
                owed -= 1.0;
                scene.step();
            }
            scene.follow(dt);
            run.push(furthest(&scene.spread.v, &run_was));
            picture.push(furthest(&scene.shown, &picture_was));
            if scene.spread.settled() && scene.caught_up() {
                break;
            }
        }
        (run, picture)
    }

    /// How unevenly something moved: the change in its speed from one frame to the
    /// next, on average, as a share of its average speed (0 for a steady glide).
    fn unevenness(moves: &[f64]) -> f64 {
        let mean = moves.iter().sum::<f64>() / moves.len().max(1) as f64;
        let change = moves.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / (moves.len().max(2) - 1) as f64;
        change / mean.max(1e-300)
    }

    /// The run's steps come unevenly (one in a frame, then two; none in a frame, on a
    /// fast display), and the picture moves evenly all the same: from one frame to
    /// the next its speed changes by a small share of itself, where the run's changes
    /// by most of itself or more. The picture is never far behind, and it ends
    /// exactly where the run does, after which nothing moves.
    #[test]
    fn the_picture_moves_evenly_whatever_the_display() {
        for (n, start) in [(6, Start::Best), (14, Start::Soft(0)), (32, Start::Best)] {
            for hz in [60.0, 120.0, 144.0] {
                let mut scene = Packing::create();
                scene.begin(n, 3, start);
                scene.turning = false;
                assert!(scene.caught_up() && scene.animated());
                let (run, picture) = moves(&mut scene, hz);
                // (the part of the run in which the lines visibly move)
                let busy = run.iter().rposition(|m| *m > 2e-3).unwrap_or(0) + 1;
                let (rough, even) = (unevenness(&run[..busy]), unevenness(&picture[..busy]));
                assert!(busy > 60 && rough > 0.5 && even < 0.1, "{n} lines at {hz} Hz: the run {rough:.3}, the picture {even:.3}");
                // no end is drawn moving faster than the run moves it
                let fastest = |m: &[f64]| m.iter().copied().fold(0.0, f64::max);
                assert!(fastest(&picture) <= fastest(&run) + 1e-12);
                assert!(scene.spread.settled() && scene.shown == scene.spread.v && !scene.animated(), "{n} lines at {hz} Hz did not come to rest");
            }
        }
        // how far behind: at the run's fastest (every step turning a line as far as a
        // step may), the picture trails it by what the run does in a twelfth of a second
        let mut scene = Packing::create();
        scene.begin(14, 3, Start::Soft(0));
        let mut behind = 0.0f64;
        for frame in 0..90 {
            for _ in 0..1 + frame % 2 {
                scene.step();
            }
            scene.follow(1.0 / 60.0);
            behind = scene.shown.iter().zip(&scene.spread.v).map(|(p, q)| (*p - *q).length()).fold(behind, f64::max);
        }
        let most = crate::math::packing::REACH * scene.rate as f64 * 2.0 * FOLLOW as f64;
        assert!(behind > 0.3 * most && behind < 1.1 * most, "{behind} against {most}");
        // and while the scene's time stands still, so does the picture
        let was = scene.shown.clone();
        scene.follow(0.0);
        assert_eq!(scene.shown, was);
    }

    /// An end dragged out of place, even onto another line: the numbers stay finite,
    /// and the run takes up again and settles; a small disturbance is undone.
    #[test]
    fn a_dragged_end_is_taken_up_again() {
        let mut scene = run(1);
        let at_rest = scene.spread.coherence();
        let nudged = (scene.spread.v[0] + scene.spread.v[0].any_orthonormal_vector() * 0.2).normalize();
        scene.spread.place(0, nudged);
        assert!(!scene.spread.settled() && scene.spread.coherence() > at_rest + 0.05);
        settle(&mut scene);
        assert!((scene.spread.coherence() - at_rest).abs() < 1e-6, "{}", scene.spread.coherence());

        // on top of another line, on its own opposite end, at nothing at all
        for to in [scene.spread.v[1], -scene.spread.v[1], -scene.spread.v[0], DVec3::ZERO, DVec3::NAN] {
            scene.spread.place(0, to);
            let r = reading(&scene.spread.v);
            assert!(r.mu.is_finite() && r.angle.is_finite() && scene.spread.v.iter().all(|v| v.is_finite() && (v.length() - 1.0).abs() < 1e-9));
            settle(&mut scene);
            assert!(scene.spread.coherence() < 0.6, "left on top of another line: {}", scene.spread.coherence());
        }
    }

    /// The pairs joined in the picture: at a right angle both ends are as near, so three
    /// lines at right angles make an octahedron (four edges at each of its six ends).
    #[test]
    fn lines_at_right_angles_are_joined_at_both_ends() {
        let mut scene = Packing::create();
        scene.begin(3, 3, Start::Best);
        settle(&mut scene);
        let r = reading(&scene.spread.v);
        assert!(r.mu < 1e-6 && r.close.len() == 3 && format!("{:.2}", r.angle) == "90.00");
        assert_eq!(edges_at_each_end(&scene), vec![4; 3]);
        // every edge drawn joins two ends that are as close as the closest pair is
        for scene in [run(0), run(1), run(3), run(4)] {
            let r = reading(&scene.spread.v);
            for &pair in &r.close {
                for (a, b) in joined(&scene.spread.v, pair) {
                    assert!((degrees(a.dot(b)) - r.angle).abs() < CLOSE && a.dot(b) > -1e-4);
                }
            }
        }
    }

    /// The numbers listed are the arrangement: read back, they have its dot products.
    #[test]
    fn the_listed_numbers_are_the_arrangement() {
        for scene in [run(1), run(5)] {
            let listed = coordinates(&scene.spread.v, scene.dim);
            let read: Vec<DVec3> = listed
                .lines()
                .map(|line| {
                    let x: Vec<f64> = line.split_whitespace().map(|n| n.parse().unwrap()).collect();
                    assert_eq!(x.len(), scene.dim);
                    DVec3::new(x[0], x[1], if scene.dim == 3 { x[2] } else { 0.0 })
                })
                .collect();
            assert!(read.len() == scene.n && !listed.contains("-0.000000"));
            assert!(read.iter().all(|v| (v.length() - 1.0).abs() < 2e-6) && (coherence(&read) - scene.spread.coherence()).abs() < 2e-6);
        }
    }

    /// The panels stay on the canvas, whatever its size, and the view is one the camera
    /// can take.
    #[test]
    fn the_panels_fit_every_canvas() {
        for (w, h) in [(1080.0, 1000.0), (390.0, 800.0), (390.0, 560.0), (844.0, 390.0), (640.0, 480.0), (240.0, 180.0), (70.0, 800.0), (3000.0, 1600.0), (1.0, 1.0)] {
            let view = Vec2::new(w, h);
            for inset in [0.0, 44.0] {
                let at = Panels::of(view, inset);
                let (o, side) = at.heat;
                let (c, size) = at.chart;
                assert!(side > 0.0 && size.min_element() > 0.0 && o.is_finite() && c.is_finite(), "{view}");
                if w >= 390.0 && h >= 390.0 {
                    assert!(o.x >= 0.0 && o.x + side <= w && o.y + side <= h && c.x + size.x <= w && c.y + size.y <= h, "{view}: {o} {side} {c} {size}");
                    assert!(at.narrow || c.y >= o.y + side + 40.0, "{view}: the chart runs into the table");
                }
                let mut scene = Packing::create();
                (scene.canvas, scene.inset) = (view, inset);
                for dim in [2, 3] {
                    scene.dim = dim;
                    match scene.the_view() {
                        View::Orbit { target, dist, .. } => assert!(target.is_finite() && dist.is_finite() && dist > 0.0, "{view}"),
                        View::Fit { center, half } => assert!(center.is_finite() && half.is_finite() && half.min_element() > 0.0, "{view}"),
                        _ => panic!("a view of another kind"),
                    }
                }
            }
        }
    }
}

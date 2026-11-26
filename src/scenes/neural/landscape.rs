//! Loss landscape.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "landscape",
    title: "Loss landscape",
    group: "Neural networks",
    blurb: "A true 2D slice of a network's loss surface, with the shadow of the optimiser's path on it.",
    make: |g| Box::new(Landscape::new(g)),
};

const LAND_N: usize = 56;
const LAND_OPTS: [&str; 3] = ["gradient descent", "momentum", "Adam"];
/// The rules behind those names (the same that train every network here).
const LAND_OPTIM: [Optim; 3] = [Optim::Sgd, Optim::Momentum, Optim::Adam];

/// Experiments: (button, what to look for, optimiser as an index into `LAND_OPTS`,
/// learning rate, steps).
const TRIALS: [(&str, &str, usize, f32, usize); 5] = [
    ("Adam", "Four hundred steps of Adam. Watch the replay: a fast drop down the wall, then a slower run along the valley floor. Note the loss at the end.", 2, 0.05, 400),
    ("plain gradient descent", "The same number of steps, each simply downhill. It comes down the wall as fast, then creeps along the floor, and ends with a loss many times higher.", 0, 0.05, 400),
    ("momentum", "Momentum keeps part of each step for the next, which carries it along the floor: it ends close to Adam.", 1, 0.05, 400),
    ("a step too long", "Gradient descent with a step ten times as long. It leaps from wall to wall and ends far above where it started.", 0, 0.5, 400),
    ("a long run", "Plain gradient descent again, with 3000 steps instead of 400: patience gets it most of the way to where Adam was.", 0, 0.05, 3000),
];

/// A loss as one would write it by hand, typeset: 4.9 × 10⁻⁵ rather than 4.85e-5.
fn loss_text(v: f32) -> String {
    if v > 0.0 && v < 1e-3 {
        let mut e = v.log10().floor() as i32;
        let mut m = (v / 10f32.powi(e) * 10.0).round() / 10.0;
        if m >= 10.0 {
            m = 1.0;
            e += 1;
        }
        format!(r"${m:.1} \times 10^{{{e}}}$")
    } else {
        loss_number(v)
    }
}

/// At most this many points of the optimiser's path are kept, evenly spaced over
/// the run: enough to draw it and to find its plane, however many steps it took.
const PATH_POINTS: usize = 400;
/// The most training steps done in one frame when no clock is consulted (a
/// reproducible run): a long run is then spread over a few frames all the same.
const STEPS_PER_FRAME: usize = 600;
/// A run is made again only once a slider has rested for this many seconds, so that
/// dragging one does not start a run on every frame of the drag.
const SETTLE: f32 = 0.15;

/// One run of an optimiser on the scene's network (two hidden layers of 8 tanh
/// neurons) and task (the curve y = sin(1.7x) + 0.3x at 48 points), taken a step at
/// a time so that a long one can be spread over several frames.
struct Run {
    net: Mlp,
    rng: Rng,
    xs: Vec<f32>,
    ys: Vec<f32>,
    optimiser: usize,
    lr: f32,
    wanted: usize,
    taken: usize,
    theta: Vec<f32>,
    /// The weights at the start and after every `every`-th step.
    every: usize,
    traj: Vec<Vec<f32>>,
    diverged: bool,
}

impl Run {
    fn new(optimiser: usize, lr: f32, steps: usize, seed: u64) -> Run {
        let mut rng = Rng::new(seed);
        let n = 48;
        let xs: Vec<f32> = (0..n).map(|i| -2.5 + 5.0 * i as f32 / (n - 1) as f32).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| (1.7 * x).sin() + 0.3 * x).collect();
        let net = Mlp::new(&[1, 8, 8, 1], Act::Tanh, &mut rng);
        let theta = net.params();
        Run { traj: vec![theta.clone()], theta, net, rng, xs, ys, optimiser, lr, wanted: steps, taken: 0, every: steps.div_ceil(PATH_POINTS).max(1), diverged: false }
    }

    fn done(&self) -> bool {
        self.taken >= self.wanted || self.diverged
    }

    /// The loss at the weights `theta`.
    fn loss_at(&mut self, theta: &[f32]) -> f32 {
        self.net.set_params(theta);
        self.net.eval(&self.xs, &self.ys, self.xs.len(), Loss::Mse)
    }

    /// One step of the optimiser.
    fn step(&mut self) {
        let n = self.xs.len();
        self.net.forward(&self.xs, n);
        self.net.backward(&self.ys, Loss::Mse);
        self.net.step(LAND_OPTIM[self.optimiser.min(LAND_OPTIM.len() - 1)], self.lr, 0.9, 0.0);
        self.theta = self.net.params();
        if self.theta.iter().any(|v| !v.is_finite()) {
            self.diverged = true;
            return;
        }
        self.taken += 1;
        if self.taken.is_multiple_of(self.every) || self.taken == self.wanted {
            self.traj.push(self.theta.clone());
        }
    }
}

/// The plane that best contains a path: its top two principal directions about the
/// path's end, by power iteration (each stopped once it has settled), and the share
/// of the path's spread about the end that lies in that plane.
fn principal_plane(traj: &[Vec<f32>], rng: &mut Rng) -> ([Vec<f32>; 2], f32) {
    let Some(end) = traj.last() else {
        return ([Vec::new(), Vec::new()], 0.0);
    };
    let dim = end.len();
    let mut rows: Vec<Vec<f32>> = traj.iter().map(|t| t.iter().zip(end).map(|(a, b)| a - b).collect()).collect();
    let total: f32 = rows.iter().map(|r| r.iter().map(|v| v * v).sum::<f32>()).sum::<f32>().max(1e-12);
    let mut dirs = [Vec::new(), Vec::new()];
    let mut captured = 0.0;
    let mut next = vec![0.0f32; dim];
    for dir in &mut dirs {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.normal()).collect();
        for _ in 0..80 {
            next.fill(0.0);
            for r in &rows {
                let dot: f32 = r.iter().zip(&v).map(|(a, b)| a * b).sum();
                for (o, a) in next.iter_mut().zip(r) {
                    *o += dot * a;
                }
            }
            let norm = next.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-20);
            let mut moved = 0.0;
            for (v, n) in v.iter_mut().zip(&next) {
                moved += (n / norm - *v) * (n / norm - *v);
                *v = n / norm;
            }
            // (a path's spread is nearly always dominated by one or two directions,
            // and then this settles in a few rounds rather than eighty)
            if moved < 1e-10 {
                break;
            }
        }
        // remove this direction from every row before looking for the next one
        for r in &mut rows {
            let dot: f32 = r.iter().zip(&v).map(|(a, b)| a * b).sum();
            captured += dot * dot;
            for (o, a) in r.iter_mut().zip(&v) {
                *o -= dot * a;
            }
        }
        *dir = v;
    }
    (dirs, captured / total)
}

/// The picture of a finished run, made a few rows of the grid at a time: the loss is
/// worked out at 56 × 56 points of the plane and at every point of the path, which
/// takes longer than one frame should.
struct Picture {
    run: Run,
    traj: Vec<Vec<f32>>,
    dirs: [Vec<f32>; 2],
    explained: f32,
    /// Every point of the path in the plane's coordinates, and the part of the plane drawn.
    coords: Vec<Vec2>,
    lo: Vec2,
    hi: Vec2,
    /// The loss at the points of the grid, row by row, and then at the points of the
    /// path, as far as they have been worked out.
    vals: Vec<f32>,
    /// One point of the plane in weights (room for it, reused).
    point: Vec<f32>,
}

impl Picture {
    fn new(mut run: Run) -> Picture {
        let traj = std::mem::take(&mut run.traj);
        let (dirs, explained) = principal_plane(&traj, &mut run.rng);
        let end = traj.last().cloned().unwrap_or_default();
        let coords: Vec<Vec2> = traj
            .iter()
            .map(|t| {
                let along = |dir: &[f32]| t.iter().zip(&end).zip(dir).map(|((a, b), d)| (a - b) * d).sum::<f32>();
                Vec2::new(along(&dirs[0]), along(&dirs[1]))
            })
            .collect();
        let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for c in &coords {
            lo = lo.min(*c);
            hi = hi.max(*c);
        }
        let pad = (hi - lo).max_element().max(1e-3) * 0.45;
        let (lo, hi) = (lo - Vec2::splat(pad), hi + Vec2::splat(pad));
        let point = end.clone();
        Picture { run, vals: Vec::with_capacity(LAND_N * LAND_N + coords.len()), traj, dirs, explained, coords, lo, hi, point }
    }

    /// Point `i` of the grid, in the plane's coordinates.
    fn grid(&self, i: usize) -> Vec2 {
        let (lo, hi) = (self.lo, self.hi);
        let (x, y) = ((i % LAND_N) as f32 / (LAND_N - 1) as f32, (i / LAND_N) as f32 / (LAND_N - 1) as f32);
        Vec2::new(lo.x + (hi.x - lo.x) * x, lo.y + (hi.y - lo.y) * y)
    }

    /// Work out more of the losses, a row of the grid at a time, until `budget` is
    /// spent (one row at least). True once they are all known.
    fn work(&mut self, budget: &crate::clock::Budget) -> bool {
        let grid = LAND_N * LAND_N;
        let total = grid + self.coords.len();
        let Some(end) = self.traj.last() else {
            return true;
        };
        while self.vals.len() < total {
            let row_end = ((self.vals.len() / LAND_N + 1) * LAND_N).min(total);
            while self.vals.len() < row_end {
                let k = self.vals.len();
                let c = if k < grid { self.grid(k) } else { self.coords[k - grid] };
                for (((p, e), a), b) in self.point.iter_mut().zip(end).zip(&self.dirs[0]).zip(&self.dirs[1]) {
                    *p = e + c.x * a + c.y * b;
                }
                let loss = self.run.loss_at(&self.point);
                self.vals.push(loss);
            }
            if budget.spent() {
                break;
            }
        }
        self.vals.len() >= total
    }
}

pub struct Landscape {
    optimiser: usize,
    lr: f32,
    steps: usize,
    seed: u64,
    surf: Surface,
    path: Vec<Vec3>,
    replay: f32,
    explained: f32,
    final_loss: f32,
    start_loss: f32,
    /// The settings have changed since the picture was made, and the seconds still to
    /// wait before making it again (see `SETTLE`).
    stale: bool,
    wait: f32,
    /// The run the next picture will be of, while it is being made; and then the
    /// picture, while that is being made. The last picture stays up meanwhile.
    job: Option<Run>,
    picture: Option<Picture>,
}

impl Landscape {
    pub fn new(gfx: &Gfx) -> Self {
        let mut surf = Surface::new(gfx, LAND_N as u32, LAND_N as u32);
        surf.style.cmap = Some(Cmap::Inferno);
        surf.style.contour = 0.2;
        surf.style.contour_alpha = 0.4;
        surf.style.specular = 0.3;
        Landscape {
            optimiser: 2,
            lr: 0.05,
            steps: 400,
            seed: 2,
            surf,
            path: Vec::new(),
            replay: 0.0,
            explained: 0.0,
            final_loss: 0.0,
            start_loss: 0.0,
            stale: true,
            wait: 0.0,
            job: None,
            picture: None,
        }
    }

    /// Show a finished picture: the loss on the plane that best contains the path the
    /// optimiser took, and the path's shadow on it.
    fn show(&mut self, mut pic: Picture) {
        let grid = LAND_N * LAND_N;
        let (lo, hi) = (pic.lo, pic.hi);
        // a fractional power compresses five orders of magnitude without turning the
        // minimum into a needle, as a logarithm would
        let mut sorted = pic.vals[..grid].to_vec();
        sorted.sort_by(f32::total_cmp);
        let top = sorted[sorted.len() * 9 / 10].max(1e-9);
        let height = |v: f32| 3.3 * (v.max(0.0) / top).min(1.6).powf(0.3);
        let span = (hi - lo).max_element();
        let to_world = |c: Vec2| (c - (lo + hi) * 0.5) / span * 6.4;
        for (k, p) in self.surf.data_mut().iter_mut().enumerate() {
            let w = to_world(pic.grid(k));
            let h = height(pic.vals[k]);
            *p = [w.x, w.y, h, h];
        }
        self.surf.style.range = [-0.5, 4.3];
        self.path = pic.coords.iter().zip(&pic.vals[grid..]).map(|(&c, &v)| to_world(c).extend(height(v) + 0.04)).collect();
        self.explained = pic.explained;
        if let (Some(first), Some(last)) = (pic.traj.first(), pic.traj.last()) {
            self.start_loss = pic.run.loss_at(first);
            self.final_loss = pic.run.loss_at(last);
        }
        self.replay = 0.0;
    }
}

/// The line batch the path travelled so far is drawn in (above the surface, or over
/// everything), which gains points as the replay goes on: a batch of its own, which
/// differs from the usual one by a hair of its width and nothing else, so that from
/// one frame to the next what stays the same can still be compared point for point
/// (that is how `eidos motion` tells smooth motion from jumps).
fn travelling<'a>(f: &'a mut Frame, kind: LineKind) -> &'a mut crate::gfx::draw::LineBatch {
    // (the style of the batch it would otherwise be in: `lines` or `overlay`)
    let depth_bias = if kind == LineKind::Overlay { 0.0 } else { LineStyle::default().depth_bias };
    f.dl.lines_with(kind, LineStyle { width_scale: 1.0001, depth_bias, ..Default::default() })
}

impl Scene for Landscape {
    fn view(&self) -> View {
        View::Orbit { target: Vec3::new(0.0, 0.0, 1.5), yaw: -0.8, pitch: 0.62, dist: 15.5 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Training a network is a walk downhill on its loss: a surface with one direction for every weight. This small network (two hidden layers of 8 tanh neurons, fitting the curve $y = \sin(1.7x) + 0.3x$) has 97 weights, so its surface has 97 directions. What you see is a true flat slice through it: the plane that best contains the path the optimiser actually took, found from the path itself.");
        ui::term(ui, "Loss:", "how wrong the network is, here its mean squared error on the curve.");

        ui::section(ui, "Things to try");
        // (a button makes the new run at once; a slider, once it has rested)
        let (mut again, mut dragged) = (false, false);
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                (_, _, self.optimiser, self.lr, self.steps) = TRIALS[k];
                again = true;
            }
        }

        ui::section(ui, "Optimiser");
        again |= ui::choice(ui, &LAND_OPTS, &mut self.optimiser);
        dragged |= ui::slider_log(ui, "learning rate", &mut self.lr, 0.001..=0.5);
        ui::note(ui, "The learning rate is the length of a step.");
        dragged |= ui::slider_int_log(ui, "steps", &mut self.steps, 20..=3000);
        ui.horizontal(|ui| {
            if ui.button("new starting weights").clicked() {
                self.seed += 1;
                again = true;
            }
            if ui.button("replay").clicked() {
                self.replay = 0.0;
            }
        });
        if again || dragged {
            self.stale = true;
            self.wait = if again { 0.0 } else { SETTLE };
        }
        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (Cmap::Inferno.color(0.9), "height and colour: the loss, yellow high"),
                (Cmap::Inferno.color(0.3), "purple: low. The scale is compressed, so both walls and floor show"),
                (pal::WHITE, "white ball: the starting weights"),
                (pal::CYAN, "cyan line and ball: the optimiser's path, and where it is now; the ring marks its end"),
            ],
        );
        ui::note(ui, "The path leaves the plane now and then, so what is drawn is its shadow on the plane, at the plane's height there: where it has left the plane, a ball can sit above or below the loss the optimiser really had. The end lies on the plane, so there the two agree. The numbers below are the real losses.");
        ui::section(ui, "Numbers");
        ui::readout(ui, "loss at the start", loss_text(self.start_loss));
        ui::readout(ui, "loss at the end", loss_text(self.final_loss));
        ui::readout(ui, "path captured by this plane", format!("{:.1} %", self.explained * 100.0));
        ui::note(ui, "The last number says how honest the picture is: how much of the optimiser's movement really happened inside the plane you are looking at. The plane is worked out afresh for every run, so changing the optimiser also changes the slice you see.");
        ui::see_also(ui, &[("descent", "the same optimisers on surfaces you can choose"), ("race", "optimisers racing on one network"), ("surface", "surfaces and gradients")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // a new run, if one is called for: started once the sliders have rested, and
        // trained within the frame's allowance of time, over several frames if it is
        // long; then its picture, within the same allowance. The picture of the last
        // run stays up meanwhile.
        if self.stale {
            if self.wait > 0.0 {
                self.wait -= f.real_dt;
            } else {
                self.job = Some(Run::new(self.optimiser, self.lr, self.steps, self.seed));
                self.picture = None;
                self.stale = false;
            }
        }
        let budget = crate::clock::frame_budget(f.real_dt, 0.006);
        if let Some(mut run) = self.job.take() {
            let mut taken = 0;
            while !run.done() && taken < STEPS_PER_FRAME {
                run.step();
                taken += 1;
                if budget.spent() {
                    break;
                }
            }
            // (finding the plane takes a millisecond or so: on the next frame, if this
            // one has had its share)
            if run.done() && !budget.spent() {
                self.picture = Some(Picture::new(run));
            } else {
                self.job = Some(run);
            }
        }
        if let Some(mut pic) = self.picture.take() {
            if pic.work(&budget) {
                self.show(pic);
            } else {
                self.picture = Some(pic);
            }
        }
        f.surface(&mut self.surf);
        let n = self.path.len();
        if n < 2 {
            return;
        }
        self.replay = (self.replay + f.dt * 0.16).min(1.15);
        let t = self.replay.min(1.0);
        // ease in, so the fast early descent is visible. The path is a point every so
        // many steps, and the ball goes on between two of them as far as the replay
        // has gone, so that it moves in every frame: it used to jump from point to
        // point, and to stand still in between while the replay went slower than a
        // point a frame
        let at = (t * t * (n - 1) as f32).min((n - 1) as f32);
        let upto = (at as usize).min(n - 1);
        let past = at - upto as f32;
        let ball = match self.path.get(upto + 1) {
            Some(&next) if past > 0.0 => self.path[upto].lerp(next, past),
            _ => self.path[upto],
        };
        let travelled = || self.path[..=upto].iter().copied().chain((ball != self.path[upto]).then_some(ball));
        f.dl.lines.polyline(self.path.iter().copied(), 1.2, pal::WHITE.a(0.25));
        travelling(f, LineKind::Blend).polyline(travelled(), 3.0, pal::CYAN);
        // the same again, faintly, over the surface: where a wall of the bowl stands in
        // front of the path (usually at its end) it can still be followed
        f.dl.overlay.polyline(self.path.iter().copied(), 1.0, pal::WHITE.a(0.22));
        travelling(f, LineKind::Overlay).polyline(travelled(), 1.6, pal::CYAN.a(0.4));
        let end = self.path[n - 1];
        // (the ring in the same batch, after the path, as it always was)
        travelling(f, LineKind::Overlay).polyline((0..=24).map(|i| i as f32 / 24.0 * TAU).map(|a| end + Vec3::new(a.cos(), a.sin(), 0.0) * 0.13), 1.6, pal::CYAN);
        f.sphere(ball + Vec3::Z * 0.1, 0.24, pal::CYAN);
        f.sphere(self.path[0] + Vec3::Z * 0.06, 0.16, pal::WHITE);
        // names on plates: the surface behind them runs from pale yellow to black
        for (at, offset, name, color, align) in [
            (self.path[0], Vec2::new(12.0, -12.0), "start", pal::WHITE, Align2::LEFT_BOTTOM),
            (end, Vec2::new(14.0, 12.0), "end", pal::CYAN, Align2::LEFT_TOP),
        ] {
            if let Some(p) = on_screen(f, at) {
                f.caption_px(p + offset, name, 13.0, color, align);
            }
        }
        bottom_caption(f, &["height and colour: the loss, on a compressed scale", "yellow high, purple low", "cyan: the path's shadow on this plane"], 12.0);
    }

    fn animated(&self) -> bool {
        self.replay < 1.15 || self.stale || self.job.is_some() || self.picture.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One run as the scene makes it, without the pictures: the losses at the start
    /// and the end, and how much longer the path is than the straight line between
    /// its ends.
    fn run(optimiser: usize, lr: f32, steps: usize, seed: u64) -> (f32, f32, f32) {
        let mut run = Run::new(optimiser, lr, steps, seed);
        let start = run.theta.clone();
        let first = run.loss_at(&start);
        let mut length = 0.0;
        while !run.done() {
            let before = run.theta.clone();
            run.step();
            length += run.theta.iter().zip(&before).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
        }
        let end = run.theta.clone();
        let straight = end.iter().zip(&start).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
        (first, run.loss_at(&end), length / straight.max(1e-9))
    }

    /// However long the run, the path that is kept has its two ends and no more
    /// than a few hundred points; and a short run is kept whole.
    #[test]
    fn the_path_is_kept_at_a_bounded_length() {
        for steps in [20usize, 400, 401, 3000] {
            let mut run = Run::new(0, 0.05, steps, 2);
            let start = run.theta.clone();
            while !run.done() {
                run.step();
            }
            assert!(run.traj.len() <= PATH_POINTS + 2 && run.traj.len() >= steps.min(PATH_POINTS / 2), "{steps} steps kept as {} points", run.traj.len());
            assert_eq!(run.traj[0], start);
            assert_eq!(run.traj.last().unwrap(), &run.theta);
            assert!(steps > PATH_POINTS || run.traj.len() == steps + 1);
        }
    }

    /// A picture made a row at a time (a frame with no time to spare) is the picture
    /// made at once, and each frame's share of it is one row of the grid.
    #[test]
    fn a_picture_made_in_slices_is_the_same_picture() {
        let finished = || {
            let mut run = Run::new(0, 0.05, 3000, 3);
            while !run.done() {
                run.step();
            }
            Picture::new(run)
        };
        let mut whole = finished();
        assert!(whole.work(&crate::clock::Budget::new(1e9)));
        let mut sliced = finished();
        let mut frames = 0;
        while !sliced.work(&crate::clock::Budget::new(0.0)) {
            frames += 1;
            assert!(sliced.vals.len() == frames * LAND_N, "frame {frames}: {} values", sliced.vals.len());
        }
        assert_eq!(frames + 1, (LAND_N * LAND_N + sliced.coords.len()).div_ceil(LAND_N));
        assert_eq!(whole.vals, sliced.vals);
        assert!(whole.explained > 0.9, "{}", whole.explained);
    }

    /// The experiments' captions, held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        for seed in [2u64, 3, 4] {
            let [adam, plain, momentum, too_long, patient] = TRIALS.map(|(_, _, optimiser, lr, steps)| run(optimiser, lr, steps, seed));
            assert!(plain.1 > 5.0 * adam.1, "seed {seed}: plain {} against Adam {}", plain.1, adam.1);
            assert!(momentum.1 < 0.5 * plain.1 && momentum.1 < 4.0 * adam.1, "seed {seed}: momentum {}", momentum.1);
            assert!(too_long.1 > 10.0 * too_long.0 && too_long.2 > 20.0, "seed {seed}: too long a step ends at {}", too_long.1);
            assert!(patient.1 < 0.3 * plain.1 && patient.1 < 10.0 * adam.1, "seed {seed}: the long run ends at {}", patient.1);
        }
    }

    /// What the experiments lead to (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for seed in [2u64, 3, 4] {
            for (name, _, optimiser, lr, steps) in TRIALS {
                let (first, last, wander) = run(optimiser, lr, steps, seed);
                let mut whole = Run::new(optimiser, lr, steps, seed);
                while !whole.done() {
                    whole.step();
                }
                let start = std::time::Instant::now();
                let (_, captured) = principal_plane(&whole.traj, &mut whole.rng);
                let took = start.elapsed().as_secs_f64() * 1e3;
                println!("seed {seed} {name:<24} loss {first:.3} -> {last:.5}   path / straight line {wander:.2}   captured {:.2} % ({took:.2} ms)", captured * 100.0);
            }
        }
    }
}

//! Neurons as particles: one wide hidden layer, each neuron drawn as a point in the
//! plane of its input weights, so that learning becomes visible as motion.

use super::wide::Wide;
use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "particles",
    title: "Neurons as particles",
    group: "Neural networks",
    blurb: "Feature learning against the lazy regime: do the neurons move, or only their weights in the sum?",
    make: |g| Box::new(Particles::new(g)),
};

const FIELD_N: usize = 44;
const POINTS: usize = 200;
/// Points kept of each chart's curve.
const HISTORY: usize = 600;
/// Positions remembered along each neuron's path; when full, every other one is dropped.
const PATH: usize = 40;
/// At most this many paths and switch lines are drawn (evenly chosen among the neurons).
const DRAWN: usize = 240;
/// How quickly the left square's zoom follows the neurons as they spread (seconds:
/// the time to cover all but 1/e of the way).
const ZOOM: f32 = 0.16;
const REGIMES: [(&str, f32); 3] = [("feature learning", 0.05), ("in between", 1.0), ("lazy", 20.0)];

/// Experiments: (button, what to look for). `Particles::set_up` holds what each sets.
const TRIALS: [(&str, &str); 5] = [
    (r"feature learning ($\alpha = 0.05$)", "The dots travel far and gather along a few directions: the network is building features for this data. \"Weights moved\" passes 100 %."),
    (r"lazy ($\alpha = 20$)", "The dots hardly move (a few per cent), yet the loss falls about as fast. The fit comes from tiny changes spread over all the neurons."),
    (r"in between ($\alpha = 1$)", "Some travel, much less. Press the first two again to compare: all three start from the same weights."),
    ("lazy, with only 16 neurons", "It still fits, but each neuron must now move about three times as far (compare \"weights moved\"). Staying put is a luxury of wide layers."),
    ("feature learning on xor", r"This data is split along the two axes, so the useful directions are the diagonals, and the neurons find them: nearly all end up along one. Then slide $\alpha$ up to 20: the same fit, and no pattern at all."),
];

struct Particles {
    net: Wide,
    data: Data,
    data_kind: usize,
    noise: f32,
    neurons: usize,
    act: usize,
    alpha: f32,
    lr: f32,
    rate: usize,
    running: bool,
    steps: u64,
    pace: crate::clock::Pace,
    /// What `pace` owes, kept beside it in the same steps: its part of a step is how
    /// far the picture is from the weights before the last step to the weights now.
    owed: f32,
    /// The weights as they were before the last step, and as they are drawn (see
    /// `Particles::follow`). The network's own weights (`net`) are the true ones: the
    /// numbers, the charts and training go by them.
    before: Neurons,
    shown: Neurons,
    /// Counts the changes to `shown`, for `redraw` (the answer field is drawn for it).
    shown_changes: u64,
    loss: f32,
    accuracy: f32,
    loss_history: History,
    moved_history: History,
    paths: Vec<Vec<Vec2>>,
    path_every: u64,
    /// The step at which the paths' last points were taken.
    path_last: u64,
    field: Surface,
    grid: Vec<f32>,
    g0_grid: Vec<f32>,
    scratch: Vec<f32>,
    reach: f32,
    show_paths: bool,
    show_lines: bool,
    rng: Rng,
    stale: bool,
    redraw: Redraw,
}

impl Particles {
    fn new(gfx: &Gfx) -> Self {
        let mut rng = Rng::new(41);
        let data = make_data(1, POINTS, 0.2, &mut rng);
        let net = Wide::new(200, Act::Relu, &mut rng);
        let mut field = answer_field(gfx, FIELD_N);
        field.style.range = [-5.0, 5.0];
        // one line only: the boundary, where the output is zero
        field.style.contour = 1.0;
        let mut scene = Particles {
            net,
            data,
            data_kind: 1,
            noise: 0.2,
            neurons: 200,
            act: 1,
            alpha: REGIMES[0].1,
            lr: 1.0,
            rate: 60,
            running: true,
            steps: 0,
            pace: crate::clock::Pace::default(),
            owed: 0.0,
            before: Neurons::default(),
            shown: Neurons::default(),
            shown_changes: 0,
            loss: 0.0,
            accuracy: 0.0,
            loss_history: History::new(HISTORY),
            moved_history: History::new(HISTORY),
            paths: Vec::new(),
            path_every: 1,
            path_last: 0,
            field,
            grid: input_grid(FIELD_N, DATA_R),
            g0_grid: Vec::new(),
            scratch: Vec::new(),
            reach: 3.0,
            show_paths: true,
            show_lines: true,
            rng,
            stale: true,
            redraw: Redraw::default(),
        };
        scene.restart();
        scene
    }

    /// The same starting weights again (so that regimes can be compared like for like).
    fn restart(&mut self) {
        self.net.rewind();
        // (also where the run stands before its first step: the read-outs would
        // otherwise show the last run's final loss beside "steps 0")
        (self.loss, self.accuracy) = self.net.anchor(&self.data, self.alpha);
        let mut g0 = std::mem::take(&mut self.g0_grid);
        self.net.g(&self.grid, &mut g0);
        self.g0_grid = g0;
        self.steps = 0;
        self.loss_history.clear();
        self.moved_history.clear();
        self.paths = (0..self.net.neurons()).map(|j| vec![self.net.w[j]]).collect();
        (self.path_every, self.path_last) = (1, 0);
        // (drawn where the run starts: nothing to go between yet)
        self.before.take(&self.net);
        self.shown.take(&self.net);
        self.stale = true;
    }

    /// Set up one of the experiments in `TRIALS`, from the default settings.
    fn set_up(&mut self, k: usize) {
        let (alpha, neurons, kind) = match k {
            0 => (REGIMES[0].1, 200, 1),
            1 => (REGIMES[2].1, 200, 1),
            2 => (REGIMES[1].1, 200, 1),
            3 => (REGIMES[2].1, 16, 1),
            _ => (REGIMES[0].1, 200, 3),
        };
        let rebuild = neurons != self.neurons || self.act != 1;
        let new_data = kind != self.data_kind || self.noise != 0.2;
        (self.alpha, self.neurons, self.act, self.data_kind, self.noise, self.lr) = (alpha, neurons, 1, kind, 0.2, 1.0);
        if new_data {
            self.data = make_data(self.data_kind, POINTS, self.noise, &mut self.rng);
        }
        if rebuild {
            self.rebuild();
        } else {
            self.restart();
        }
        self.running = true;
    }

    /// New starting weights.
    fn rebuild(&mut self) {
        self.net = Wide::new(self.neurons, Act::ALL[self.act], &mut self.rng);
        self.restart();
    }

    fn train(&mut self) {
        let (loss, accuracy) = self.net.step(&self.data, self.alpha, self.lr);
        (self.loss, self.accuracy) = (loss, accuracy);
        self.steps += 1;
        if self.steps.is_multiple_of(self.path_every) {
            for (path, w) in self.paths.iter_mut().zip(&self.net.w) {
                path.push(*w);
            }
            self.path_last = self.steps;
            if self.paths.first().is_some_and(|p| p.len() > PATH) {
                for path in &mut self.paths {
                    let mut k = 0;
                    path.retain(|_| {
                        k += 1;
                        k % 2 == 1
                    });
                }
                self.path_every *= 2;
            }
        }
        if self.steps.is_multiple_of(2) {
            self.loss_history.push(loss);
            self.moved_history.push(self.net.moved());
        }
    }

    /// Bring the weights as drawn to the part of a step the pace owes, between the
    /// weights before the last step and the weights now: the steps come whole, a
    /// number of them in a frame and not the same number in every frame (60 a second
    /// on a display of 144 is one in some frames and none in the others), and drawn
    /// as they come the neurons would move in fits and starts. So the picture is drawn
    /// up to a step behind the run, at the point the steps have reached by now.
    fn follow(&mut self) {
        let t = self.owed.clamp(0.0, 1.0);
        let (before, now, shown) = (&self.before, &self.net, &mut self.shown);
        if shown.w.len() != now.w.len() || before.w.len() != now.w.len() {
            return self.shown.take(&self.net);
        }
        let mut changed = false;
        for j in 0..now.w.len() {
            let w = before.w[j] + (now.w[j] - before.w[j]) * t;
            let b = before.b[j] + (now.b[j] - before.b[j]) * t;
            let a = before.a[j] + (now.a[j] - before.a[j]) * t;
            changed |= w != shown.w[j] || b != shown.b[j] || a != shown.a[j];
            (shown.w[j], shown.b[j], shown.a[j]) = (w, b, a);
        }
        self.shown_changes += changed as u64;
    }

    /// The network's output over the plane of inputs, for the weights as drawn.
    fn refresh(&mut self) {
        let mut g = std::mem::take(&mut self.scratch);
        // (the network is lent the drawn weights for this, and given its own back)
        self.shown.swap(&mut self.net);
        self.net.g(&self.grid, &mut g);
        self.shown.swap(&mut self.net);
        for ((p, g), g0) in self.field.data_mut().iter_mut().zip(&g).zip(&self.g0_grid) {
            p[3] = self.alpha * (g - g0);
        }
        self.scratch = g;
    }
}

/// The neurons' weights: the input weights `w`, the biases `b` and the output
/// weights `a` (one each per neuron).
#[derive(Default)]
struct Neurons {
    w: Vec<Vec2>,
    b: Vec<f32>,
    a: Vec<f32>,
}

impl Neurons {
    /// The network's weights as they are now.
    fn take(&mut self, net: &Wide) {
        self.w.clone_from(&net.w);
        self.b.clone_from(&net.b);
        self.a.clone_from(&net.a);
    }

    /// Trade weights with the network.
    fn swap(&mut self, net: &mut Wide) {
        std::mem::swap(&mut self.w, &mut net.w);
        std::mem::swap(&mut self.b, &mut net.b);
        std::mem::swap(&mut self.a, &mut net.a);
    }
}

/// The line batch of a part of the picture whose number of points changes as
/// training goes on (`k` = 1: the paths, which gain a point now and then; 2: the
/// lines where a neuron's input is zero, which leave the square and come back into
/// it). Each has a batch of its own, different from the usual one by a hair of its
/// width and nothing else, so that from one frame to the next the parts that stay
/// the same can still be compared point for point: that is how `eidos motion` tells
/// smooth motion from jumps.
fn batch<'a>(f: &'a mut Frame, k: u8) -> &'a mut crate::gfx::draw::LineBatch {
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0 + 1e-4 * k as f32, ..Default::default() })
}

/// The part of the line `w·x + b = 0` inside the square of half-width `r`.
fn switch_line(w: Vec2, b: f32, r: f32) -> Option<(Vec2, Vec2)> {
    let len2 = w.length_squared();
    if len2 < 1e-8 {
        return None;
    }
    let (foot, along) = (w * (-b / len2), Vec2::new(-w.y, w.x) / len2.sqrt());
    let (mut lo, mut hi) = (f32::NEG_INFINITY, f32::INFINITY);
    for (p, d) in [(foot.x, along.x), (foot.y, along.y)] {
        if d.abs() < 1e-9 {
            if p.abs() > r {
                return None;
            }
        } else {
            let (t0, t1) = ((-r - p) / d, (r - p) / d);
            lo = lo.max(t0.min(t1));
            hi = hi.min(t0.max(t1));
        }
    }
    (lo < hi).then(|| (foot + along * lo, foot + along * hi))
}

impl Scene for Particles {
    fn view(&self) -> View {
        STAGE_VIEW
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn animated(&self) -> bool {
        self.running
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A layer of neurons can fit the same data in two quite different ways: by moving its neurons to where they are useful, or by leaving them where chance put them and only adjusting how they are added up. One number decides which. Here every neuron is a dot, so you can watch which is happening.");
        ui.horizontal(|ui| {
            if ui.button(if self.running { "pause" } else { "train" }).clicked() {
                self.running = !self.running;
            }
            if ui.button("same start again").clicked() {
                self.restart();
            }
            if ui.button("new start").clicked() {
                self.rebuild();
            }
        });

        ui::section(ui, "Things to try");
        for (k, (label, look_for)) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, label, look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Regime");
        if ui::slider_log(ui, r"output scale $\alpha$", &mut self.alpha, 0.02..=50.0) {
            self.restart();
        }
        ui::note(ui, r"The output is $\alpha$ times the change in the network since the start, and the step is scaled so that, at the start, the output learns at the same speed whatever $\alpha$ is. With a large $\alpha$ a tiny change in the weights is enough. With a small $\alpha$ the weights must travel far. Changing $\alpha$ starts again from the same weights.");
        ui::term(ui, "Lazy regime:", "the weights barely move, so the network behaves like a fixed set of random features with adjustable weights on them.");
        ui::term(ui, "Feature learning:", "the neurons themselves move to fit the data.");
        ui::section(ui, "Network");
        let mut rebuilt = ui::slider_int_log(ui, "neurons", &mut self.neurons, 8..=1024);
        let names: Vec<&str> = Act::ALL.iter().map(|a| activation_name(*a)).collect();
        rebuilt |= ui::choice(ui, &names, &mut self.act);
        if rebuilt {
            self.rebuild();
        }
        ui::section(ui, "Data");
        let mut data_changed = ui::choice(ui, &DATASETS, &mut self.data_kind);
        data_changed |= ui::slider(ui, "noise", &mut self.noise, 0.0..=1.0);
        if ui.button("new sample").clicked() || data_changed {
            self.data = make_data(self.data_kind, POINTS, self.noise, &mut self.rng);
            self.restart();
        }
        ui::section(ui, "Training");
        ui::slider_log(ui, "learning rate", &mut self.lr, 0.05..=4.0);
        ui::slider_int_log(ui, "steps per second", &mut self.rate, 5..=2000);
        ui::section(ui, "Show");
        ui.checkbox(&mut self.show_paths, "the path each neuron has taken");
        ui.checkbox(&mut self.show_lines, "where each neuron's input is zero");
        ui::section(ui, "Reading the picture");
        ui::note(ui, "Left: one dot per neuron, placed by its two input weights (how strongly it listens to each input). Right: the plane of inputs, with the data and the network's answer.");
        ui::key(
            ui,
            &[
                (CLASS[1], "orange dot or line: a neuron that adds to the output"),
                (CLASS[0], "blue: one that subtracts from it"),
                (pal::DIM, "small grey dot: where a neuron started; the thread from it is its path"),
                (pal::WHITE, "white line on the right: the boundary between the classes"),
            ],
        );
        ui::note(ui, "A bigger dot is a neuron with a stronger say in the output. Each faint line on the right is where one neuron's input is zero: for a ReLU, where it switches on.");

        ui::section(ui, "Numbers");
        ui::readout(ui, "steps", self.steps.to_string());
        ui::readout_colored(ui, "loss", loss_number(self.loss), pal::CYAN);
        ui::readout(ui, "accuracy", format!("{:.1} %", self.accuracy * 100.0));
        ui::readout_colored(ui, "weights moved", format!("{:.1} %", self.net.moved() * 100.0), pal::YELLOW);
        ui::note(ui, "\"Weights moved\" is the distance of all the weights from where they started, as a share of their starting size. The loss is how wrong the network is on the data: 0.69 is a coin toss.");
        ui::see_also(ui, &[("mlp", "a deeper network, neuron by neuron"), ("landscape", "the loss surface a network moves on"), ("fit1d", "neurons as building blocks of a curve")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        if self.running && f.dt > 0.0 {
            // (the rate is per second of the scene's time, which the app's time speed sets)
            let rate = self.rate as f32;
            let mut turn = self.pace.begin(rate, f.dt, f.real_dt, 0.004);
            // (what the pace owes, worked out as it works it out, and paid off step by step)
            self.owed = (self.owed + rate * f.dt).min((rate * 0.1).max(2.0));
            while turn.next() {
                self.owed -= 1.0;
                self.before.take(&self.net);
                self.train();
            }
            self.pace.end(turn);
            self.follow();
        }
        if self.redraw.due(self.shown_changes, self.stale) {
            let start = crate::clock::now();
            self.refresh();
            self.redraw.done(self.shown_changes, crate::clock::now() - start);
            self.stale = false;
        }

        // ----- the layout: the two squares side by side with a chart under each, or
        // (on a tall canvas) one above the other with the charts side by side below
        let (w, ppu) = (stage_width(f), pixels_per_unit(f));
        let (margin, gap, head, foot, chart_head, chart_least, bottom) = (0.14, 0.34, 0.5, 0.44, 0.3, 0.55, 0.14);
        let beside = ((w - 2.0 * margin - gap) / 2.0).min(STAGE_H - head - foot - chart_head - chart_least - bottom);
        let stacked = (w - 2.0 * margin).min((STAGE_H - head - 2.0 * foot - 0.26 - chart_head - chart_least - bottom) / 2.0);
        let tall = stacked > beside;
        let side = if tall { stacked } else { beside }.max(0.5);
        let half = side * 0.5;
        let squares_h = if tall { 2.0 * (side + foot) + 0.26 } else { side + foot };
        let chart_h = (STAGE_H - head - squares_h - chart_head - bottom).min(1.3);
        let top = 2.5 - head - 0.5 * (STAGE_H - head - squares_h - chart_head - chart_h - bottom);
        let (centre_w, centre_x) = if tall {
            (Vec2::new(0.0, top - half), Vec2::new(0.0, top - side - foot - 0.26 - half))
        } else {
            (Vec2::new(-(side + gap) * 0.5, top - half), Vec2::new((side + gap) * 0.5, top - half))
        };
        let chart_y = top - squares_h - chart_head - chart_h;
        let chart_w = if tall { (w - 2.0 * margin - gap) * 0.5 } else { side };
        let (o1, o2) = if tall { (Vec2::new(-w * 0.5 + margin, chart_y), Vec2::new(gap * 0.5, chart_y)) } else { (Vec2::new(centre_w.x - half, chart_y), Vec2::new(centre_x.x - half, chart_y)) };
        let side_px = side * ppu;
        let fits = |text: &str, size: f32| text_width(text, size) < side_px;

        // (everything below draws the neurons as `shown` has them; the numbers written
        // in the picture are the run's own)
        let neurons = self.shown.w.len();
        let stride = neurons.div_ceil(DRAWN).max(1);
        let sign = |a: f32| if a > 0.0 { CLASS[1] } else { CLASS[0] };

        // ----- the plane of input weights: one dot per neuron
        let centre = centre_w;
        let furthest = self.shown.w.iter().map(|w| w.abs().max_element()).fold(0.0, f32::max);
        let want = (furthest * 1.08).max(3.0);
        // eased over the scene's time while training runs, whatever the frame rate;
        // at once when it does not (after "same start again" while paused, say)
        let ease = if self.running && f.dt > 0.0 { 1.0 - (-f.dt / ZOOM).exp() } else { 1.0 };
        self.reach += (want - self.reach) * ease;
        let scale = half / self.reach;
        let place = |w: Vec2| centre + w * scale;
        inset(f, centre - Vec2::splat(half), centre + Vec2::splat(half));
        frame_square(f, centre, half, 1.2, pal::DIM);
        f.line(centre - Vec2::new(half, 0.0), centre + Vec2::new(half, 0.0), 1.0, pal::FAINT.a(0.6));
        f.line(centre - Vec2::new(0.0, half), centre + Vec2::new(0.0, half), 1.0, pal::FAINT.a(0.6));
        // rings give the scale, which grows as the neurons spread
        let step = nice_step(self.reach / 2.5).max(1.0);
        let mut ring = step;
        while ring < self.reach {
            f.circle(centre.extend(0.0), Vec3::Z, ring * scale, 1.0, pal::FAINT.a(0.5));
            f.text(centre + Vec2::new(ring * scale, 0.0) + Vec2::new(0.03, 0.03), fmt_tick(ring, step), 10.5, pal::DIM, Align2::LEFT_BOTTOM);
            ring += step;
        }
        let largest = self.shown.a.iter().fold(1e-6f32, |m, a| m.max(a.abs()));
        if self.show_paths {
            // (a point taken at the last step lies ahead of the dot, which is drawn up
            // to a step behind: the path goes as far as the dot)
            let ahead = (self.steps > 0 && self.path_last == self.steps) as usize;
            let paths = batch(f, 1);
            for j in (0..neurons).step_by(stride) {
                let taken = &self.paths[j][..self.paths[j].len().saturating_sub(ahead)];
                let path = taken.iter().copied().chain(std::iter::once(self.shown.w[j])).map(|w| place(w).extend(0.0));
                paths.polyline(path, 1.1, sign(self.shown.a[j]).a(0.35));
            }
        }
        f.dl.discs.style.outline_px = 0.8;
        let grow = (side_px / 340.0).clamp(0.6, 1.3);
        for j in 0..neurons {
            if self.show_paths && j % stride == 0 {
                f.dot(place(self.net.start_of(j)), 2.5 * grow, pal::DIM.a(0.7));
            }
            let size = (3.0 + 6.0 * (self.shown.a[j].abs() / largest).sqrt()) * grow;
            f.dot(place(self.shown.w[j]), size, sign(self.shown.a[j]));
        }
        let title = if fits("the neurons: each dot placed by its two input weights", 12.5) { "the neurons: each dot placed by its two input weights" } else { "the neurons, by their input weights" };
        f.text(centre + Vec2::new(-half, half + 0.07), title, 12.5, pal::TEXT, Align2::LEFT_BOTTOM);
        f.text(centre + Vec2::new(half - 0.05, -half + 0.04), "weight on $x_1$ →", 10.5, pal::DIM, Align2::RIGHT_BOTTOM);
        f.text(centre + Vec2::new(-half + 0.05, half - 0.04), "↑ weight on $x_2$", 10.5, pal::DIM, Align2::LEFT_TOP);
        let under = centre + Vec2::new(-half, -half - 0.06);
        let (adds, subtracts) = ("orange: adds to the output", "blue: subtracts from it");
        let mut lines = 1.0;
        f.text(under, adds, 11.0, CLASS[1], Align2::LEFT_TOP);
        if text_width(adds, 11.0) + text_width(subtracts, 11.0) + 16.0 < side_px {
            f.text(centre + Vec2::new(half, -half - 0.06), subtracts, 11.0, CLASS[0], Align2::RIGHT_TOP);
        } else {
            f.text_off(under, Vec2::new(0.0, 14.0), subtracts, 11.0, CLASS[0], Align2::LEFT_TOP);
            lines = 2.0;
        }
        f.text_off(under, Vec2::new(0.0, 14.0 * lines), "bigger dot: a stronger say    grey dot: its start", 10.5, pal::DIM, Align2::LEFT_TOP);

        // ----- the plane of inputs: the data, the output, and every neuron's switch line
        let centre = centre_x;
        self.field.style.model = Mat4::from_translation(centre.extend(-0.2)) * Mat4::from_scale(Vec3::new(side, side, 1.0));
        f.surface(&mut self.field);
        frame_square(f, centre, half, 1.2, pal::DIM);
        let to_world = |p: Vec2| centre + p / DATA_R * half;
        if self.show_lines {
            let shown = &self.shown;
            let weight = |j: usize| shown.a[j].abs() * shown.w[j].length();
            let heaviest = (0..neurons).map(weight).fold(1e-6f32, f32::max);
            // (fewer lines than paths: they all cross the same small square)
            let lines = batch(f, 2);
            for j in (0..neurons).step_by(stride * 2) {
                if let Some((p, q)) = switch_line(shown.w[j], shown.b[j], DATA_R) {
                    lines.segment(to_world(p).extend(0.0), to_world(q).extend(0.0), 1.0, sign(shown.a[j]).mix(pal::WHITE, 0.35).a(0.05 + 0.3 * weight(j) / heaviest));
                }
            }
        }
        for i in 0..self.data.len() {
            let p = self.data.point(i);
            if p.abs().max_element() < DATA_R {
                f.dot(to_world(p), 5.5 * grow, CLASS[self.data.y[i] as usize]);
            }
        }
        let title = if fits("the plane of inputs: the data and the network's answer", 12.5) { "the plane of inputs: the data and the network's answer" } else { "the inputs: data and answer" };
        f.text(centre + Vec2::new(-half, half + 0.07), title, 12.5, pal::TEXT, Align2::LEFT_BOTTOM);
        f.text(centre + Vec2::new(half - 0.05, -half + 0.04), "$x_1$ →", 10.5, pal::WHITE.a(0.8), Align2::RIGHT_BOTTOM);
        f.text(centre + Vec2::new(-half + 0.05, half - 0.04), "↑ $x_2$", 10.5, pal::WHITE.a(0.8), Align2::LEFT_TOP);
        wrapped(f, centre + Vec2::new(-half, -half - 0.06), side, "Colour: the answer, from sure it is blue to sure it is orange. White line: the boundary. Faint lines: where one neuron's input is zero.", 10.5, pal::DIM);

        // ----- the loss, and how far the weights have gone
        let size = Vec2::new(chart_w, chart_h);
        chart(f, o1, size, &[(self.loss_history.points(), pal::CYAN)], true);
        let loss = format!("loss (log scale)  {}", loss_number(self.loss));
        let steps = format!("{} steps", self.steps);
        // (the loss is typeset, wider when small: measured as drawn, so that the count
        // of steps is left out rather than written over it)
        let clear = f.text_size(&loss, 12.0).x + text_width(&steps, 11.0) + 12.0 < chart_w * ppu;
        f.text(o1 + Vec2::new(0.0, size.y + 0.07), loss, 12.0, pal::CYAN, Align2::LEFT_BOTTOM);
        if chart_w * ppu > 190.0 && clear {
            f.text(o1 + Vec2::new(size.x, size.y + 0.07), steps, 11.0, pal::DIM, Align2::RIGHT_BOTTOM);
        }
        chart_in(f, o2, size, &[(self.moved_history.points(), pal::YELLOW)], false, Some([0.0, self.moved_history.points().iter().fold(0.05f32, |m, &v| m.max(v)) * 1.1]));
        let moved = self.net.moved() * 100.0;
        let long = format!("how far the weights have moved  {moved:.1} %");
        let label = if text_width(&long, 12.0) < chart_w * ppu { long } else { format!("weights moved  {moved:.0} %") };
        f.text(o2 + Vec2::new(0.0, size.y + 0.07), label, 12.0, pal::YELLOW, Align2::LEFT_BOTTOM);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scene's claim: the regimes fit the data equally fast, and differ in how far
    /// the weights go to do it.
    #[test]
    fn a_large_output_scale_is_lazy_and_a_small_one_moves_the_weights() {
        let run = |alpha: f32| {
            let mut rng = Rng::new(9);
            let data = make_data(1, 200, 0.2, &mut rng);
            let mut net = Wide::new(256, Act::Relu, &mut rng);
            net.anchor(&data, 1.0);
            let mut steps = 0;
            while net.step(&data, alpha, 1.0).0 > 0.3 {
                steps += 1;
                assert!(steps < 2000, "alpha {alpha} did not reach a loss of 0.3");
            }
            (steps, net.moved())
        };
        let (lazy_steps, lazy_moved) = run(20.0);
        let (rich_steps, rich_moved) = run(0.05);
        assert!(lazy_moved < 0.02, "lazy regime moved {lazy_moved}");
        assert!(rich_moved > 1.0, "feature-learning regime moved only {rich_moved}");
        assert!(rich_steps < 4 * lazy_steps && lazy_steps < 4 * rich_steps, "{lazy_steps} against {rich_steps} steps");
    }

    #[test]
    fn switch_lines_are_cut_to_the_square() {
        let (p, q) = switch_line(Vec2::new(1.0, 0.0), -0.5, 1.0).unwrap();
        assert!((p.x - 0.5).abs() < 1e-6 && (q.x - 0.5).abs() < 1e-6 && (p.y - q.y).abs() > 1.99);
        assert!(switch_line(Vec2::new(1.0, 0.0), 3.0, 1.0).is_none());
        assert!(switch_line(Vec2::ZERO, 1.0, 1.0).is_none());
        let (p, q) = switch_line(Vec2::new(1.0, 1.0), 0.0, 1.0).unwrap();
        assert!((p + q).length() < 1e-5 && (p - q).length() > 2.8);
    }

    /// How training goes across output scales (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for (kind, name) in [(1usize, "circles"), (0, "spirals"), (3, "xor")] {
            for act in [Act::Relu, Act::Tanh] {
                for alpha in [0.02f32, 0.05, 0.2, 1.0, 5.0, 20.0, 50.0] {
                    for lr in [1.0f32] {
                        let mut rng = Rng::new(9);
                        let data = make_data(kind, 200, 0.2, &mut rng);
                        let mut net = Wide::new(256, act, &mut rng);
                        net.anchor(&data, 1.0);
                        let (mut loss, mut acc, mut at) = (0.0, 0.0, None);
                        for step in 0..4000 {
                            (loss, acc) = net.step(&data, alpha, lr);
                            if at.is_none() && loss < 0.3 {
                                at = Some((step, net.moved()));
                            }
                            if !loss.is_finite() {
                                break;
                            }
                        }
                        println!(
                            "{name:<8} {:<5} alpha={alpha:<5} lr={lr}: loss {loss:.3} acc {acc:.2} moved {:.3}   loss<0.3 at {}",
                            act.name(),
                            net.moved(),
                            at.map_or("never".to_string(), |(s, m)| format!("step {s} (moved {m:.3})"))
                        );
                    }
                }
            }
        }
    }

    /// The share of the neurons' pull on the output held by neurons within 15 degrees
    /// of a diagonal (a third, if their directions are spread evenly).
    fn near_a_diagonal(net: &Wide) -> f32 {
        let pull = |j: usize| net.a[j].abs() * net.w[j].length();
        let all: f32 = (0..net.neurons()).map(pull).sum();
        let near: f32 = (0..net.neurons()).filter(|&j| (net.w[j].y.atan2(net.w[j].x).to_degrees().rem_euclid(90.0) - 45.0).abs() < 15.0).map(pull).sum();
        near / all.max(1e-9)
    }

    fn after(steps: usize, neurons: usize, alpha: f32, kind: usize, seed: u64) -> Wide {
        let mut rng = Rng::new(seed);
        let data = make_data(kind, 200, 0.2, &mut rng);
        let mut net = Wide::new(neurons, Act::Relu, &mut rng);
        net.anchor(&data, 1.0);
        for _ in 0..steps {
            net.step(&data, alpha, 1.0);
        }
        net
    }

    /// Two of the experiments' captions, held to account.
    #[test]
    fn a_narrow_lazy_layer_moves_further_and_xor_lines_the_neurons_up() {
        for seed in [9u64, 11] {
            let (narrow, wide) = (after(300, 16, 20.0, 1, seed).moved(), after(300, 200, 20.0, 1, seed).moved());
            assert!(narrow > 2.0 * wide && narrow < 6.0 * wide && wide < 0.03, "seed {seed}: {narrow} against {wide}");
            let (rich, lazy) = (near_a_diagonal(&after(600, 200, 0.05, 3, seed)), near_a_diagonal(&after(600, 200, 20.0, 3, seed)));
            assert!(rich > 0.9 && lazy < 0.6, "seed {seed}: {rich} against {lazy}");
        }
    }

    /// What the last two experiments lead to (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore_experiments() {
        for seed in [9u64, 10, 11] {
            for (neurons, alpha, kind) in [(16usize, 20.0f32, 1usize), (200, 20.0, 1), (16, 0.05, 1), (200, 0.05, 3), (200, 20.0, 3)] {
                let mut line = format!("seed {seed} neurons {neurons:<4} alpha {alpha:<5} data {kind}:");
                for steps in [300, 1000, 3000] {
                    let net = after(steps, neurons, alpha, kind, seed);
                    line += &format!("  step {steps}: moved {:.3} near a diagonal {:.0} %", net.moved(), 100.0 * near_a_diagonal(&net));
                }
                println!("{line}");
            }
        }
    }
}

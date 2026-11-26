//! Optimiser race: one network, one starting point, one stream of batches, and a
//! different optimiser in each lane.

use super::session::{Problem, Session, Settings};
use super::*;
use crate::clock;

pub const DEF: SceneDef = SceneDef {
    id: "race",
    title: "Optimiser race",
    group: "Neural networks",
    blurb: "The same network, start and batches: only the optimiser differs.",
    make: |g| Box::new(Race::new(g)),
};

const FIELD_N: usize = 72;
/// Points kept of each lane's loss curve.
const HISTORY: usize = 600;
/// The output at which the network gives nine to one for a class: 90 % (and, negated, 10 %).
const NINE_TO_ONE: f32 = 2.197_224_6;
/// The output is coloured over this range either side of zero (odds of about 400 to 1).
const SURE: f32 = 6.0;
const POINTS: usize = 260;

/// The usual race, which every experiment starts from: the lanes in it, the noise in
/// the data, the network (hidden layers, neurons per layer, activation) and the batch.
const USUAL: [Optim; 4] = [Optim::Sgd, Optim::Momentum, Optim::Adam, Optim::Muon];
const NOISE: f32 = 0.25;
const HIDDEN: usize = 2;
const WIDTH: usize = 8;
const ACT: Act = Act::Tanh;
const BATCH: usize = 32;

/// Weight decay at 1 shrinks every weight by this share at each step (one per cent),
/// in every lane alike.
const SHRINK: f32 = 0.01;

/// Experiments: (button, what to look for, data set, batch size, learning rates ×,
/// every rule in the race rather than the usual four). Everything else is as in the
/// usual race; the tests run the same table.
const TRIALS: [(&str, &str, usize, usize, f32, bool); 6] = [
    ("the usual race", "SGD, momentum, Adam and Muon on two spirals. Watch which lanes bend their boundary round the arms, and which stay stuck with a nearly straight one.", 0, BATCH, 1.0, false),
    ("all seven rules", "Every optimiser here, same start. The standings at the bottom of this panel rank them as they go.", 0, BATCH, 1.0, true),
    ("steps five times as long", "Each rule's usual step, times five. The curves turn jumpy as the steps overshoot, and the leaders are held back: a longer step is not a faster race.", 0, BATCH, 5.0, false),
    ("one point per step", "Each step now follows a single data point. Nobody gets far, and momentum is thrown about most: it keeps pushing the way the last stray points sent it.", 0, 1, 1.0, false),
    ("all the data every step", "No sampling noise: the curves are smooth, and every difference left is the rule's own.", 0, POINTS, 1.0, false),
    ("an easy problem", "Two separate blobs. Every rule gets there; they differ only in how fast the pale band narrows, which is how fast the weights grow.", 4, BATCH, 1.0, false),
];

/// The settings of the lane of rule `optim` (its weight decay is applied by
/// [`lane_step`], not by them).
fn lane_settings(optim: Optim, lr_scale: f32, batch: usize) -> Settings {
    Settings { optim, lr: optim.default_lr() * lr_scale, momentum: optim.default_momentum(), decay: 0.0, batch, rate: 0.0, running: false }
}

/// One step of one lane, with weight decay `decay`. The decay is applied here, after
/// the rule's step and the same in every lane: inside the optimiser it is scaled by
/// the learning rate, which differs from rule to rule (SGD's is fifteen times Adam's),
/// so the lanes would differ by more than their rule.
fn lane_step(job: &mut Session, settings: Settings, decay: f32) {
    job.settings = Settings { decay: 0.0, ..settings };
    let before = job.steps;
    job.request_steps(1);
    job.work(0.0, f64::MAX);
    if decay > 0.0 && job.steps > before {
        let keep = (1.0 - SHRINK * decay).max(0.0);
        for w in job.net.w.iter_mut().flatten() {
            let v = *w * keep;
            // (numbers below the normal range are very slow to compute with)
            *w = if v.abs() < 1e-30 { 0.0 } else { v };
        }
    }
}

/// Each rule's usual learning rate, for the panel: "SGD 0.3, momentum and Nesterov
/// 0.05, ...", rules with the same rate named together.
fn usual_rates() -> String {
    let mut groups: Vec<(Vec<&str>, f32)> = Vec::new();
    for optim in Optim::ALL {
        match groups.last_mut() {
            Some((names, lr)) if *lr == optim.default_lr() => names.push(optim.name()),
            _ => groups.push((vec![optim.name()], optim.default_lr())),
        }
    }
    let list = |names: &[&str]| match names.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
        _ => names.join(""),
    };
    groups.iter().map(|(names, lr)| format!("{} {lr}", list(names))).collect::<Vec<_>>().join(", ")
}

struct Lane {
    optim: Optim,
    on: bool,
    job: Session,
    field: Surface,
    history: History,
    loss: f32,
    accuracy: f32,
    /// Which data points this lane currently puts on the wrong side.
    wrong: Vec<bool>,
    /// The step its picture shows (none yet, after a restart).
    drawn: Option<u64>,
}

/// What the lanes are ranked by (lower is better): the loss, or, for a lane that
/// has diverged, something above every loss. (Its loss itself is not a number, and
/// in the usual ordering of floats a NaN with its sign bit set, which is what the
/// arithmetic here produces, comes before every number: the diverged lanes used to
/// head the list.)
fn standing(loss: f32, diverged: bool) -> f32 {
    if diverged || !loss.is_finite() {
        f32::INFINITY
    } else {
        loss
    }
}

impl Lane {
    fn diverged(&self) -> bool {
        self.job.diverged || !self.loss.is_finite()
    }

    fn standing(&self) -> f32 {
        standing(self.loss, self.diverged())
    }
}

pub struct Race {
    data_kind: usize,
    noise: f32,
    data: Data,
    hidden: usize,
    width: usize,
    act: usize,
    batch: usize,
    /// Every optimiser's usual learning rate is multiplied by this.
    lr_scale: f32,
    decay: f32,
    rate: usize,
    running: bool,
    steps: u64,
    pace: clock::Pace,
    lanes: Vec<Lane>,
    /// The weights every lane starts from, and the seed of the batches they all see.
    start: Mlp,
    seed: u64,
    rng: Rng,
    grid: Vec<f32>,
    stale: bool,
    redraw: Redraw,
    /// The step the loss curves have been brought up to (none yet, after a restart).
    recorded: Option<u64>,
    /// The lane whose picture is to be brought up to date first next time.
    next_field: usize,
}

impl Race {
    pub fn new(gfx: &Gfx) -> Self {
        let mut rng = Rng::new(31);
        let data = make_data(0, POINTS, NOISE, &mut rng);
        let start = Mlp::new(&layer_sizes(2, HIDDEN, WIDTH, 1), ACT, &mut rng);
        let lanes = Optim::ALL
            .iter()
            .map(|&optim| {
                let mut field = answer_field(gfx, FIELD_N);
                field.style.range = [-SURE, SURE];
                // three lines: the boundary (even odds), and 10 % and 90 % either side of it
                field.style.contour = NINE_TO_ONE;
                field.style.contour_reach = 1.5 * NINE_TO_ONE;
                field.style.contour_alpha = 0.95;
                Lane {
                    optim,
                    on: USUAL.contains(&optim),
                    job: Session::placeholder(),
                    field,
                    history: History::new(HISTORY),
                    loss: 0.0,
                    accuracy: 0.0,
                    wrong: Vec::new(),
                    drawn: None,
                }
            })
            .collect();
        let mut race = Race {
            data_kind: 0,
            noise: NOISE,
            data,
            hidden: HIDDEN,
            width: WIDTH,
            act: Act::ALL.iter().position(|&a| a == ACT).unwrap_or(0),
            batch: BATCH,
            lr_scale: 1.0,
            decay: 0.0,
            rate: 240,
            running: true,
            steps: 0,
            pace: clock::Pace::default(),
            lanes,
            start,
            seed: 1,
            rng,
            grid: input_grid(FIELD_N, DATA_R),
            stale: true,
            redraw: Redraw::default(),
            recorded: None,
            next_field: 0,
        };
        race.restart();
        race
    }

    /// Put every lane back on the starting weights, with the same batches to come.
    fn restart(&mut self) {
        for lane in &mut self.lanes {
            let net = Mlp::from_params(&self.start.sizes, self.start.act, &self.start.params());
            lane.job = Session::new(net, Problem::Plane(self.data.clone()), self.seed);
            lane.history.clear();
            lane.drawn = None;
        }
        self.steps = 0;
        self.recorded = None;
        self.stale = true;
    }

    /// New starting weights (and a new stream of batches) for everyone.
    fn reshuffle(&mut self) {
        self.start = Mlp::new(&layer_sizes(2, self.hidden, self.width, 1), Act::ALL[self.act], &mut self.rng);
        self.seed += 1;
        self.restart();
    }

    /// Set up one of the experiments in `TRIALS`: the usual race with one thing changed.
    fn set_up(&mut self, k: usize) {
        let (_, _, data, batch, lr_scale, every) = TRIALS[k];
        for lane in &mut self.lanes {
            lane.on = every || USUAL.contains(&lane.optim);
        }
        (self.data_kind, self.noise, self.hidden, self.width) = (data, NOISE, HIDDEN, WIDTH);
        self.act = Act::ALL.iter().position(|&a| a == ACT).unwrap_or(0);
        (self.batch, self.lr_scale, self.decay) = (batch, lr_scale, 0.0);
        self.data = make_data(self.data_kind, POINTS, self.noise, &mut self.rng);
        self.reshuffle();
        self.running = true;
    }

    fn color(optim: Optim) -> Color {
        pal::SERIES[Optim::ALL.iter().position(|&o| o == optim).unwrap_or(0) % pal::SERIES.len()]
    }

    fn settings(&self, optim: Optim) -> Settings {
        lane_settings(optim, self.lr_scale, self.batch)
    }

    /// One step in every lane that is still in the race.
    fn round(&mut self) {
        for i in 0..self.lanes.len() {
            if self.lanes[i].on {
                let settings = self.settings(self.lanes[i].optim);
                lane_step(&mut self.lanes[i].job, settings, self.decay);
            }
        }
        self.steps += 1;
    }

    /// Where every lane stands: its loss and its accuracy, and a point on its loss
    /// curve for every step taken since the last time.
    fn refresh(&mut self) {
        let n = self.data.len();
        // (one point per step, the loss just measured standing for the steps since the
        // last measurement: the curves' horizontal axis is then in steps, whatever the
        // frame rate and the steps per second)
        let new_points = match self.recorded {
            None => 1,
            Some(at) => self.steps.saturating_sub(at),
        };
        self.recorded = Some(self.steps);
        for lane in self.lanes.iter_mut().filter(|l| l.on) {
            let net = &mut lane.job.net;
            let last = net.layers() - 1;
            lane.loss = net.eval(&self.data.x, &self.data.y, n, Loss::Logistic);
            lane.wrong.clear();
            lane.wrong.extend(net.neuron(last, 0).iter().zip(&self.data.y).map(|(o, y)| (*o > 0.0) != (*y > 0.5)));
            lane.accuracy = share_right(net.neuron(last, 0), &self.data.y);
            for _ in 0..new_points {
                lane.history.push(lane.loss);
            }
        }
    }

    /// Does any lane's picture show an earlier step than the race has reached?
    fn fields_behind(&self) -> bool {
        self.lanes.iter().any(|l| l.on && l.drawn != Some(self.steps))
    }

    /// Bring the lanes' pictures of the plane up to date, one lane after another, for
    /// as long as `budget` lasts (and at least one). With many large networks a frame
    /// does not have time for all of them; the next frame goes on where this one
    /// stopped, so the cost is spread evenly instead of falling on one frame in five.
    fn draw_fields(&mut self, budget: &clock::Budget) {
        let count = self.lanes.len();
        for k in 0..count {
            let i = (self.next_field + k) % count;
            let lane = &mut self.lanes[i];
            if !lane.on || lane.drawn == Some(self.steps) {
                continue;
            }
            let out = lane.job.net.forward(&self.grid, FIELD_N * FIELD_N);
            for (p, &v) in lane.field.data_mut().iter_mut().zip(out) {
                p[3] = v;
            }
            lane.drawn = Some(self.steps);
            self.next_field = (i + 1) % count;
            if budget.spent() {
                break;
            }
        }
    }

    /// The lanes in the race, best first: by loss on all the data, a lane whose
    /// loss is no longer a number coming after every lane whose loss is.
    fn standings(&self) -> Vec<&Lane> {
        let mut order: Vec<&Lane> = self.lanes.iter().filter(|l| l.on).collect();
        order.sort_by(|a, b| a.standing().total_cmp(&b.standing()));
        order
    }
}

impl Scene for Race {
    fn view(&self) -> View {
        STAGE_VIEW
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn animated(&self) -> bool {
        self.running || self.fields_behind()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "An optimiser is the rule that turns a gradient (which way is downhill for the loss, and how steeply) into a step of the weights. Every lane here trains the same network from the same weights on the same sequence of batches. Only the rule differs, so whatever differs between the pictures is the optimiser's doing.");
        ui.horizontal(|ui| {
            if ui.button(if self.running { "pause" } else { "race" }).clicked() {
                self.running = !self.running;
            }
            if ui.button("same start again").clicked() {
                self.restart();
            }
            if ui.button("new start").clicked() {
                self.reshuffle();
            }
        });

        ui::section(ui, "Things to try");
        for (k, (label, look_for, ..)) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, label, look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "In the race");
        let mut changed = false;
        for lane in &mut self.lanes {
            let label = egui::RichText::new(lane.optim.name()).color(Self::color(lane.optim).egui());
            changed |= ui.checkbox(&mut lane.on, label).changed();
        }
        if changed {
            // a lane that joins late would not be comparable: everyone starts over
            self.restart();
        }
        ui::slider_log(ui, r"learning rates $\times$", &mut self.lr_scale, 0.05..=20.0);
        ui::note(ui, &format!("The learning rate is the length of a step. Each rule has its own usual one ({}); this multiplies all of them.", usual_rates()));
        ui::section(ui, "Data");
        let mut data_changed = ui::choice(ui, &DATASETS, &mut self.data_kind);
        data_changed |= ui::slider(ui, "noise", &mut self.noise, 0.0..=1.0);
        if ui.button("new sample").clicked() || data_changed {
            self.data = make_data(self.data_kind, POINTS, self.noise, &mut self.rng);
            self.restart();
        }
        ui::section(ui, "Network");
        let mut arch = ui::slider_int(ui, "hidden layers", &mut self.hidden, 1..=4);
        arch |= ui::slider_int(ui, "neurons per layer", &mut self.width, 1..=16);
        let names: Vec<&str> = Act::ALL.iter().map(|a| activation_name(*a)).collect();
        arch |= ui::choice(ui, &names, &mut self.act);
        if arch {
            self.reshuffle();
        }
        ui::section(ui, "Training");
        ui::slider_int_log(ui, "batch size", &mut self.batch, 1..=POINTS);
        ui::slider_log_or_zero(ui, "weight decay", &mut self.decay, 1e-4, 3.0);
        ui::note(ui, "Weight decay: at each step every weight shrinks by this many per cent, in every lane alike, whatever its learning rate. It keeps the boundary from becoming ever sharper.");
        ui::slider_int_log(ui, "steps per second", &mut self.rate, 10..=5000);
        ui::section(ui, "Reading the picture");
        ui::note(ui, "Each square is the plane of the network's two inputs, and each dot is one data point in the colour of its class. The colour behind the dots is the network's answer at that place.");
        let heat = Cmap::CoolWarm.table();
        ui::key(
            ui,
            &[
                (heat[table_index(0.03)], "sure it is blue"),
                (heat[table_index(0.5)], "pale: undecided"),
                (heat[table_index(0.97)], "sure it is orange"),
                (pal::WHITE, "bold white line: the boundary (even odds). Fainter lines: 10 % and 90 %"),
                (pal::WHITE, "white ring: a point on the wrong side"),
            ],
        );
        ui::note(ui, "A wide pale band means a hesitant, gradual boundary. A band too thin to see means the network has become very sure, which rules that grow the weights quickly reach first.");
        ui::term(ui, "Loss:", "how wrong the network is, averaged over the data. 0.69 is a coin toss; 0 is full confidence in every right answer.");
        ui::term(ui, "Batch:", "the handful of data points the gradient is worked out from at each step.");

        ui::section(ui, "Standings");
        ui::readout(ui, "steps", self.steps.to_string());
        for lane in self.standings() {
            let text = if lane.diverged() { "diverged".to_string() } else { format!("${}$   {:.0} %", fmt(lane.loss), lane.accuracy * 100.0) };
            ui::readout_series(ui, lane.optim.name(), text, Self::color(lane.optim));
        }
        ui::note(ui, "Best first: the loss on all the data, and the share of points on the right side of the boundary.");
        ui::see_also(ui, &[("descent", "the same rules on a surface you can see"), ("mlp", "one network, neuron by neuron"), ("landscape", "a network's own loss surface")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        if self.running && f.dt > 0.0 {
            // whole rounds only, so that every lane has always taken the same number of steps
            let mut turn = self.pace.begin(self.rate as f32, f.dt, f.real_dt, 0.005);
            while turn.next() {
                self.round();
            }
            self.pace.end(turn);
        }
        if self.redraw.due(self.steps, self.stale) {
            let start = clock::now();
            self.refresh();
            self.redraw.done(self.steps, clock::now() - start);
            self.stale = false;
        }
        if self.fields_behind() {
            self.draw_fields(&clock::frame_budget(f.real_dt, 0.002));
        }

        // ----- the layout: tiles in as many rows as makes them largest, then the key
        // to the colours, then the chart
        let w = stage_width(f);
        let ppu = pixels_per_unit(f);
        let (margin, narrow) = (0.14, w < 3.4);
        let active: Vec<usize> = (0..self.lanes.len()).filter(|&i| self.lanes[i].on).collect();
        let count = active.len().max(1);
        let (head, key_h, chart_extra) = (0.34, if narrow { 1.02 } else { 0.74 }, 0.42);
        let grid_h = STAGE_H - head - key_h - (0.9 + chart_extra);
        let tile = |rows: usize| ((w - 2.0 * margin) / count.div_ceil(rows) as f32 * 0.92).min(grid_h / rows as f32 - 0.26);
        let rows = (1..=3).max_by(|a, b| tile(*a).total_cmp(&tile(*b)).then(b.cmp(a))).unwrap_or(1);
        let (cols, side) = (count.div_ceil(rows), tile(rows).max(0.3));
        let pitch = ((w - 2.0 * margin) / cols as f32).min(side * 1.14);
        let row_h = side + 0.26;
        // spare height goes to the chart; what is still left over is shared top and bottom
        let chart_h = (0.9 + grid_h - rows as f32 * row_h).min(1.7);
        let top = 2.5 - head - 0.5 * (grid_h + 0.9 - rows as f32 * row_h - chart_h).max(0.0);

        let half = side * 0.5;
        let dot = (side * ppu / 56.0).clamp(2.0, 4.6);
        f.dl.discs.style.outline_px = 0.8;
        for (k, &i) in active.iter().enumerate() {
            let (row, col) = (k / cols, k % cols);
            let in_row = if row + 1 == rows { count - row * cols } else { cols };
            let centre = Vec2::new((col as f32 - (in_row as f32 - 1.0) * 0.5) * pitch, top - row as f32 * row_h - 0.22 - half);
            let color = Self::color(self.lanes[i].optim);
            let lane = &mut self.lanes[i];
            lane.field.style.model = Mat4::from_translation(centre.extend(-0.2)) * Mat4::from_scale(Vec3::new(side, side, 1.0));
            f.surface(&mut lane.field);
            frame_square(f, centre, half, 2.0, color);
            let lane = &self.lanes[i];
            for j in 0..self.data.len() {
                let p = self.data.point(j);
                if p.abs().max_element() < DATA_R {
                    let at = centre + p / DATA_R * half;
                    if lane.wrong.get(j).copied().unwrap_or(false) {
                        // a white ring around a point on the wrong side
                        f.dot(at, dot + 4.5, pal::WHITE);
                    }
                    f.dot(at, dot, CLASS[self.data.y[j] as usize]);
                }
            }
            let name = lane.optim.name();
            f.text(centre + Vec2::new(-half, half + 0.05), name, 14.0, color, Align2::LEFT_BOTTOM);
            // as much of the lane's standing as fits beside its name
            let long = if lane.diverged() { "diverged".to_string() } else { format!("{:.0} % right   loss ${}$", lane.accuracy * 100.0, fmt(lane.loss)) };
            let short = if lane.diverged() { "diverged".to_string() } else { format!("{:.0} % right", lane.accuracy * 100.0) };
            let room = side * ppu - text_width(name, 14.0) - 14.0;
            let status = if text_width(&long, 12.0) < room { long } else { short };
            if text_width(&status, 12.0) < room {
                f.text(centre + Vec2::new(half, half + 0.05), status, 12.0, pal::DIM, Align2::RIGHT_BOTTOM);
            }
        }

        // ----- the key to the colours and lines, under the pictures
        let below = top - rows as f32 * row_h;
        let key_y = below - if narrow { 0.36 } else { 0.14 };
        let key_height = 0.07;
        let key_half = if narrow { w * 0.5 - margin - 0.04 } else { (w * 0.5 - margin - 118.0 / ppu).clamp(0.6, 2.4) };
        let heat = Cmap::CoolWarm.table();
        let pieces = 48;
        for k in 0..pieces {
            let (t0, t1) = (k as f32 / pieces as f32, (k + 1) as f32 / pieces as f32);
            let x = |t: f32| -key_half + 2.0 * key_half * t;
            f.dl.tris_alpha.rect(Vec2::new(x(t0), key_y), Vec2::new(x(t1), key_y + key_height), 0.0, heat[table_index((t0 + t1) * 0.5)]);
        }
        for (value, label, bold) in [(-NINE_TO_ONE, "10 %", false), (0.0, "boundary", true), (NINE_TO_ONE, "90 %", false)] {
            let x = key_half * value / SURE;
            f.line((x, key_y - 0.02), (x, key_y + key_height + 0.02), if bold { 2.2 } else { 1.2 }, pal::WHITE.a(if bold { 1.0 } else { 0.6 }));
            f.text((x, key_y - 0.04), label, 11.0, if bold { pal::TEXT } else { pal::DIM }, Align2::CENTER_TOP);
        }
        if narrow {
            f.text((-key_half, key_y + key_height + 0.05), "sure it is blue", 11.0, CLASS[0], Align2::LEFT_BOTTOM);
            f.text((key_half, key_y + key_height + 0.05), "sure it is orange", 11.0, CLASS[1], Align2::RIGHT_BOTTOM);
        } else {
            f.text((-key_half - 0.08, key_y + key_height * 0.5), "sure it is blue", 11.0, CLASS[0], Align2::RIGHT_CENTER);
            f.text((key_half + 0.08, key_y + key_height * 0.5), "sure it is orange", 11.0, CLASS[1], Align2::LEFT_CENTER);
        }
        let caption = "Each square: the plane of the two inputs. Dots: the data, coloured by class. White ring: a point on the wrong side.";
        let caption_w = (w - 2.0 * margin).min(text_width(caption, 11.0) / ppu + 0.1);
        wrapped(f, Vec2::new(-caption_w * 0.5, key_y - 0.24), caption_w, caption, 11.0, pal::DIM);

        // ----- every lane's loss against the step, on one chart
        // (room is left at the right for each curve's name, written where the curve ends)
        let names_room = (88.0 / ppu).min(w * 0.3);
        let origin = Vec2::new(-w * 0.5 + margin + 0.04, below - key_h - 0.3 - chart_h);
        let size = Vec2::new(w - 2.0 * margin - 0.08 - names_room, chart_h);
        let series: Vec<(&[f32], Color)> = active.iter().map(|&i| (self.lanes[i].history.points(), Self::color(self.lanes[i].optim))).collect();
        let ends = chart(f, origin, size, &series, true);
        // round values of the loss, so that the heights can be read
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for (history, _) in &series {
            for v in history.iter().filter(|v| v.is_finite()) {
                lo = lo.min(v.max(1e-6).ln());
                hi = hi.max(v.max(1e-6).ln());
            }
        }
        if lo < hi {
            let mut decade = 10.0f32;
            while decade > 1e-5 {
                for value in [0.5 * decade, 0.2 * decade, 0.1 * decade] {
                    let t = (value.ln() - lo) / (hi - lo);
                    // (labels need room between them: where values ln 2 apart would be
                    // crowded, only the decades are written)
                    if t > 0.04 && t < 0.9 && ((0.5f32).ln().abs() / (hi - lo) * chart_h * ppu > 15.0 || value / decade < 0.15) {
                        let y = origin.y + size.y * t;
                        f.line((origin.x, y), (origin.x + size.x, y), 1.0, pal::FAINT.a(0.35));
                        let decimals = (-value.log10().floor()).max(0.0) as usize;
                        f.text((origin.x + 0.03, y + 0.01), format!("{value:.decimals$}"), 10.0, pal::DIM, Align2::LEFT_BOTTOM);
                    }
                }
                decade *= 0.1;
            }
        }
        let names: Vec<(Vec2, String, Color)> = active
            .iter()
            .zip(ends)
            .filter_map(|(&i, end)| {
                let at = on_screen(f, Vec3::new(origin.x + size.x, end.y, 0.0))?;
                Some((at + Vec2::new(8.0, 0.0), self.lanes[i].optim.name().to_string(), Self::color(self.lanes[i].optim)))
            })
            .collect();
        spread_labels(f, names, 12.5, Align2::LEFT_CENTER);
        let title = if size.x * ppu > 430.0 { "loss on all the data (log scale) as the race goes" } else { "loss (log scale)" };
        f.text(origin + Vec2::new(0.0, size.y + 0.07), title, 12.0, pal::DIM, Align2::LEFT_BOTTOM);
        f.text(origin + Vec2::new(size.x, size.y + 0.07), format!("{} steps so far", self.steps), 12.0, pal::DIM, Align2::RIGHT_BOTTOM);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The premise of the scene: two lanes given the same optimiser stay identical,
    /// step for step, because nothing else differs between lanes.
    #[test]
    fn lanes_differ_only_by_their_optimiser() {
        let mut rng = Rng::new(2);
        let data = make_data(2, 120, 0.2, &mut rng);
        let start = Mlp::new(&[2, 6, 1], Act::Tanh, &mut rng);
        let lane = |optim: Optim| {
            let mut job = Session::new(Mlp::from_params(&start.sizes, start.act, &start.params()), Problem::Plane(data.clone()), 5);
            job.settings = Settings { optim, lr: optim.default_lr(), momentum: optim.default_momentum(), decay: 0.0, batch: 16, rate: 0.0, running: false };
            job
        };
        let (mut a, mut b, mut c) = (lane(Optim::Adam), lane(Optim::Adam), lane(Optim::Sgd));
        for _ in 0..50 {
            for job in [&mut a, &mut b, &mut c] {
                job.request_steps(1);
                job.work(0.0, f64::MAX);
            }
        }
        assert_eq!((a.steps, b.steps, c.steps), (50, 50, 50));
        assert_eq!(a.net.params(), b.net.params());
        assert_ne!(a.net.params(), c.net.params());
    }

    #[test]
    fn a_diverged_lane_comes_last() {
        let mut lanes = [(f32::NAN, true), (-f32::NAN, false), (4.09e-5, false), (0.3, false), (f32::INFINITY, false)];
        lanes.sort_by(|a, b| standing(a.0, a.1).total_cmp(&standing(b.0, b.1)));
        assert_eq!((lanes[0].0, lanes[1].0), (4.09e-5, 0.3));
    }

    /// One lane of experiment `k` of `TRIALS`, stepped as the scene steps it, with
    /// weight decay `decay`: the loss on all the data at the end, how much it was still
    /// jumping about (its average change between looks, ten steps apart), and the job.
    fn lane_with(seed: u64, k: usize, optim: Optim, decay: f32, steps: usize) -> (f32, f32, Session) {
        let (_, _, kind, batch, scale, _) = TRIALS[k];
        let mut rng = Rng::new(seed);
        let data = make_data(kind, POINTS, NOISE, &mut rng);
        let start = Mlp::new(&layer_sizes(2, HIDDEN, WIDTH, 1), ACT, &mut rng);
        let mut job = Session::new(Mlp::from_params(&start.sizes, start.act, &start.params()), Problem::Plane(data.clone()), 5);
        let (mut last, mut jitter, mut looks) = (0.0f32, 0.0f32, 0);
        for step in 0..steps {
            lane_step(&mut job, lane_settings(optim, scale, batch), decay);
            if step + 500 >= steps && step % 10 == 0 {
                let loss = job.net.eval(&data.x, &data.y, POINTS, Loss::Logistic);
                if looks > 0 {
                    jitter += (loss - last).abs();
                }
                looks += 1;
                last = loss;
            }
        }
        (last, jitter / (looks.max(2) - 1) as f32, job)
    }

    fn lane(seed: u64, k: usize, optim: Optim, steps: usize) -> (f32, f32) {
        let (loss, jitter, _) = lane_with(seed, k, optim, 0.0, steps);
        (loss, jitter)
    }

    /// The lanes of experiment `k`, in the order of `Optim::ALL`.
    fn lanes(seed: u64, k: usize) -> Vec<(f32, f32)> {
        Optim::ALL.iter().filter(|o| TRIALS[k].5 || USUAL.contains(o)).map(|&o| lane(seed, k, o, 2000)).collect()
    }

    /// The captions of the experiments, held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        for seed in [31u64, 33] {
            // (the usual four lanes: SGD, momentum, Adam, Muon)
            let usual = lanes(seed, 0);
            // the adaptive rules lead on the spirals
            assert!(usual[2].0 < usual[0].0 && usual[3].0 < usual[0].0, "seed {seed}: {usual:?}");
            // five times the step: jumpier, and the best lane is worse off than before
            let long = lanes(seed, 2);
            let best = |lanes: &[(f32, f32)]| lanes.iter().map(|l| l.0).fold(f32::MAX, f32::min);
            assert!(best(&long) > best(&usual), "seed {seed}: {long:?} against {usual:?}");
            assert!(long[0].1 > 2.0 * usual[0].1 && long[1].1 > 2.0 * usual[1].1, "seed {seed}: {long:?} against {usual:?}");
            // one point per step: momentum is thrown about most
            let single = lanes(seed, 3);
            assert!([0, 2, 3].iter().all(|&k| single[1].1 > 2.0 * single[k].1), "seed {seed}: {single:?}");
            // all the data every step: smoother than with batches
            let whole = lanes(seed, 4);
            assert!((0..4).all(|k| whole[k].1 < usual[k].1), "seed {seed}: {whole:?} against {usual:?}");
            // two blobs: everyone gets there
            assert!(lanes(seed, 5).iter().all(|l| l.0 < 0.05), "seed {seed}");
        }
    }

    /// The race's weight decay is one shrink, the same in every lane: one step with
    /// it is the step without it with every weight then multiplied by the same
    /// factor, whatever the rule and its learning rate (here at the top of both
    /// sliders, where a decay scaled by the learning rate would wipe SGD's weights out).
    #[test]
    fn weight_decay_is_the_same_in_every_lane() {
        let mut rng = Rng::new(3);
        let data = make_data(0, POINTS, NOISE, &mut rng);
        let start = Mlp::new(&layer_sizes(2, HIDDEN, WIDTH, 1), ACT, &mut rng);
        let job = || Session::new(Mlp::from_params(&start.sizes, start.act, &start.params()), Problem::Plane(data.clone()), 5);
        let keep = 1.0 - SHRINK * 3.0;
        for optim in Optim::ALL {
            let (mut plain, mut decayed) = (job(), job());
            lane_step(&mut plain, lane_settings(optim, 20.0, BATCH), 0.0);
            lane_step(&mut decayed, lane_settings(optim, 20.0, BATCH), 3.0);
            for (w, v) in plain.net.w.iter().flatten().zip(decayed.net.w.iter().flatten()) {
                assert!(*v == *w * keep || (*v == 0.0 && (*w * keep).abs() < 1e-30), "{}: {w} became {v}", optim.name());
            }
            assert_eq!(plain.net.b, decayed.net.b, "{}: the biases are not decayed", optim.name());
        }
        assert!(keep > 0.95);
    }

    /// "It keeps the boundary from becoming ever sharper": with weight decay at 1 the
    /// network's answers on the data stay less sure than without it, in every lane.
    #[test]
    fn weight_decay_keeps_the_boundary_soft() {
        for k in [0, 5] {
            for optim in USUAL {
                let sureness = |decay: f32| {
                    let (_, _, mut job) = lane_with(31, k, optim, decay, 2000);
                    let data = match &job.problem {
                        Problem::Plane(d) => d.clone(),
                        _ => unreachable!(),
                    };
                    let out = job.net.forward(&data.x, data.len());
                    out.iter().map(|v| v.abs()).sum::<f32>() / out.len() as f32
                };
                let (free, held) = (sureness(0.0), sureness(1.0));
                println!("{} on {}: mean |output| {free:.2} without decay, {held:.2} with", optim.name(), DATASETS[TRIALS[k].2]);
                assert!(held < 0.55 * free, "{} on {}: {held} with decay, {free} without", optim.name(), DATASETS[TRIALS[k].2]);
            }
        }
    }

    /// The panel's list of the usual learning rates is made from the optimisers' own.
    #[test]
    fn the_usual_rates_are_the_optimisers_own() {
        let text = usual_rates();
        for optim in Optim::ALL {
            let name = optim.name();
            let after = &text[text.find(name).unwrap_or_else(|| panic!("{name} missing from {text}"))..];
            // the first number after the name is its rate
            let rate = after.split(|c: char| !(c.is_ascii_digit() || c == '.')).find(|w| !w.is_empty()).and_then(|w| w.parse::<f32>().ok());
            assert_eq!(rate, Some(optim.default_lr()), "{name} in {text}");
        }
    }

    /// What each experiment in `TRIALS` leads to (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for seed in [31u64, 32, 33] {
            for (k, trial) in TRIALS.iter().enumerate() {
                let mut line = format!("seed {seed} {:<26}", trial.0);
                for optim in Optim::ALL.into_iter().filter(|o| trial.5 || USUAL.contains(o)) {
                    let (loss, jitter) = lane(seed, k, optim, 2000);
                    line += &format!("  {} {loss:.3} ~{jitter:.4}", optim.name());
                }
                println!("{line}");
            }
        }
    }
}

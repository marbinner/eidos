//! How layers fold space.

use super::session::Problem;
use super::training::Training;
use super::*;
use crate::clock;
use crate::math::linalg::{eig_sym, svd3};

pub const DEF: SceneDef = SceneDef {
    id: "fold",
    title: "How layers fold space",
    group: "Neural networks",
    blurb: "The input plane, bent by each layer until a flat cut separates the classes.",
    make: |g| Box::new(Fold::new(g)),
};

const SHEET_N: usize = 44;
const MAX_HIDDEN: usize = 4;
const MAX_WIDTH: usize = 16;
/// The unit cube of a layer's activity is drawn this large.
const FOLD_SCALE: f32 = 2.4;
/// How fast the view travels between layers when a button sends it (layers per second).
const TRAVEL: f32 = 2.2;
/// How quickly the scale a wide layer is drawn at follows its activity while
/// training (seconds: the time to cover all but 1/e of the way).
const RESCALE: f32 = 0.1;

/// Experiments: (button, what to look for, data set, hidden layers, neurons per
/// layer, activation as an index into `Act::ALL`, stop once separated).
const TRIALS: [(&str, &str, usize, usize, usize, usize, bool); 6] = [
    ("a ring round a disc", "No straight line parts a ring from the disc inside it. Watch the sheet being bent into a bowl until one flat cut does. Step through with \"next layer\".", 1, 2, 3, 0, true),
    ("only two neurons per layer", "Two tanh neurons can stretch and squash the plane but never lift part of it out. The ring stays wrapped round the disc, so a straight cut always leaves some points on the wrong side: accuracy stops short of 100 %.", 1, 2, 2, 0, true),
    ("ReLU: creases, not curves", "ReLU keeps the sheet in flat pieces and folds it along straight creases, each where one neuron switches on. (Stuck short of 100 %? Some neurons never switch on: reset the weights.)", 1, 2, 3, 1, true),
    ("opposite corners (xor)", "Diagonally opposite corners share a class. The first layer twists the plane so that they end up on the same side of the cut.", 3, 2, 3, 0, true),
    ("two spirals, wider layers", "Three layers of eight neurons, too many to draw: each layer is shown along the three directions in which its activity varies most. Pulling two spiral arms apart takes longer.", 0, 3, 8, 0, true),
    ("keep training after separation", "Training no longer stops at the first clean cut. The tanh neurons saturate, and every point is squashed towards a corner of the cube.", 1, 2, 3, 0, false),
];

/// How one layer's activity (a number per neuron) becomes a point in space: three
/// directions among its neurons, measured from `origin`, scaled to fit the unit cube.
#[derive(Clone)]
struct Projection {
    axes: [Vec<f32>; 3],
    origin: Vec<f32>,
    scale: f32,
    /// The share of the layer's variation that these three directions carry: 1 for a
    /// layer of at most three neurons, where nothing is left out.
    captured: f32,
    /// Is every neuron confined to a fixed range, so that the unit cube is a true wall?
    boxed: bool,
}

impl Projection {
    /// For a layer of at most three neurons the axes are the neurons themselves.
    /// Wider layers are shown along their three principal directions (the directions
    /// of greatest variation over the sheet), turned to stay as close as possible to
    /// `previous`, so that the picture does not spin or flip as training moves them.
    /// Their scale moves the share `ease` of the way from the previous one to what
    /// fits the layer's activity now (1: all the way).
    fn of(net: &Mlp, layer: usize, act: Act, previous: Option<&Projection>, ease: f32) -> Projection {
        let width = net.sizes[layer + 1];
        let column = |j: usize| net.neuron(layer, j);
        let count = column(0).len().max(1) as f64;
        let mut p = if width <= 3 {
            let axes = [0, 1, 2].map(|k| (0..width).map(|j| if j == k { 1.0 } else { 0.0 }).collect());
            // neurons that live in a fixed range are drawn in that range
            let (centre, fixed) = match act {
                Act::Tanh | Act::Sin => (0.0, Some(1.0)),
                Act::Sigmoid => (0.5, Some(2.0)),
                _ => (0.0, None),
            };
            Projection { axes, origin: vec![centre; width], scale: fixed.unwrap_or(0.0), captured: 1.0, boxed: fixed.is_some() }
        } else {
            let mean: Vec<f64> = (0..width).map(|j| column(j).iter().map(|&v| v as f64).sum::<f64>() / count).collect();
            let mut cov = vec![0.0f64; width * width];
            for a in 0..width {
                for b in a..width {
                    let c = column(a).iter().zip(column(b)).map(|(&x, &y)| (x as f64 - mean[a]) * (y as f64 - mean[b])).sum::<f64>() / count;
                    cov[a * width + b] = c;
                    cov[b * width + a] = c;
                }
            }
            let (values, vectors) = eig_sym(&cov, width);
            let total: f64 = values.iter().map(|v| v.max(0.0)).sum();
            let captured = if total > 1e-12 { (values[..3].iter().map(|v| v.max(0.0)).sum::<f64>() / total) as f32 } else { 1.0 };
            let mut axes = [0, 1, 2].map(|k| vectors[k * width..(k + 1) * width].iter().map(|&v| v as f32).collect::<Vec<f32>>());
            if let Some(prev) = previous.filter(|p| !p.boxed && p.axes[0].len() == width && p.captured < 1.0) {
                // the rotation of the new three directions (within their own span) that
                // brings them nearest the old ones
                let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
                let m = Mat3::from_cols_array_2d(&[0, 1, 2].map(|b| [0, 1, 2].map(|a| dot(&axes[a], &prev.axes[b]))));
                let (u, _, v) = svd3(m);
                let turn = u * v.transpose();
                if turn.is_finite() && (turn.determinant().abs() - 1.0).abs() < 1e-2 {
                    let turned = [0, 1, 2].map(|b| (0..width).map(|j| (0..3).map(|a| axes[a][j] * turn.col(b)[a]).sum()).collect::<Vec<f32>>());
                    axes = turned;
                }
            }
            Projection { axes, origin: mean.iter().map(|&m| m as f32).collect(), scale: 0.0, captured, boxed: false }
        };
        if p.scale == 0.0 {
            // fit what the sheet does into the unit cube, changing size smoothly
            let mut reach = 1e-6f32;
            for k in 0..3 {
                for i in 0..column(0).len() {
                    let c: f32 = (0..width).map(|j| p.axes[k][j] * (column(j)[i] - p.origin[j])).sum();
                    reach = reach.max(c.abs());
                }
            }
            let fit = 1.0 / reach;
            p.scale = match previous.filter(|q| !q.boxed && q.axes[0].len() == width) {
                Some(q) => q.scale + (fit - q.scale) * ease,
                None => fit,
            };
        }
        p
    }

    /// Where every sample of the last forward pass lands.
    fn apply(&self, net: &Mlp, layer: usize, out: &mut Vec<Vec3>) {
        let n = net.neuron(layer, 0).len();
        out.clear();
        out.resize(n, Vec3::ZERO);
        for j in 0..net.sizes[layer + 1] {
            let direction = Vec3::new(self.axes[0][j], self.axes[1][j], self.axes[2][j]) * self.scale;
            for (p, &v) in out.iter_mut().zip(net.neuron(layer, j)) {
                *p += direction * (v - self.origin[j]);
            }
        }
    }
}

pub struct Fold {
    data_kind: usize,
    noise: f32,
    points: usize,
    data: Data,
    gain: f32,
    t: Training,
    rng: Rng,
    /// Where in the network the view is: 0 is the input plane, k is hidden layer k.
    stage: f32,
    goal: Option<f32>,
    sweep: bool,
    sweep_t: f32,
    stop_when_separated: bool,
    halted: bool,
    sheet: Surface,
    grid: Vec<f32>,
    /// One projection per hidden layer, and where it puts the sheet and the data
    /// (index 0 of the last two is the input plane itself).
    projections: Vec<Projection>,
    sheet_at: Vec<Vec<Vec3>>,
    data_at: Vec<Vec<Vec3>>,
    output: Vec<f32>,
    loss: f32,
    acc: f32,
    stale: bool,
    /// Scene time since the pictures were last worked out.
    since_refresh: f32,
    drawn_stage: f32,
    /// The canvas's width over its height, as last seen: on a tall canvas the camera
    /// has to stand further back for the cube to fit across it.
    shape: f32,
    new_view: bool,
}

impl Fold {
    pub fn new(gfx: &Gfx) -> Self {
        let mut rng = Rng::new(23);
        let data = make_data(1, 220, 0.2, &mut rng);
        let mut t = Training::new(2, 3);
        t.batch = usize::MAX;
        t.rate = 180;
        t.load(Mlp::new(&t.sizes(2, 1), t.act(), &mut rng), Problem::Plane(data.clone()));
        let mut sheet = Surface::new(gfx, SHEET_N as u32, SHEET_N as u32);
        sheet.style.cmap = Some(Cmap::CoolWarm);
        sheet.style.range = [-4.0, 4.0];
        sheet.style.grid = [14.0, 14.0];
        sheet.style.grid_alpha = 0.5;
        sheet.style.line_color = Color::hex(0x0A0D14);
        sheet.style.lit = 0.75;
        sheet.style.specular = 0.25;
        Fold {
            data_kind: 1,
            noise: 0.2,
            points: 220,
            data,
            gain: 1.0,
            t,
            rng,
            stage: 0.0,
            goal: None,
            sweep: true,
            sweep_t: 0.0,
            stop_when_separated: true,
            halted: false,
            sheet,
            grid: input_grid(SHEET_N, 1.1),
            projections: Vec::new(),
            sheet_at: Vec::new(),
            data_at: Vec::new(),
            output: Vec::new(),
            loss: 0.0,
            acc: 0.0,
            stale: true,
            since_refresh: 0.0,
            drawn_stage: -1.0,
            shape: 1.0,
            new_view: false,
        }
    }

    fn hidden(&self) -> usize {
        self.t.net().layers() - 1
    }

    fn rebuild(&mut self) {
        let mut net = Mlp::new(&self.t.sizes(2, 1), self.t.act(), &mut self.rng);
        net.scale_weights(self.gain);
        self.t.load(net, Problem::Plane(self.data.clone()));
        self.projections.clear();
        self.resume();
        self.stale = true;
        // (the network may have fewer layers now: nowhere to be, or to be going, beyond them)
        let last = self.hidden() as f32;
        self.stage = self.stage.min(last);
        self.goal = self.goal.map(|g| g.min(last));
    }

    fn resample(&mut self) {
        self.data = make_data(self.data_kind, self.points, self.noise, &mut self.rng);
        self.t.set_problem(Problem::Plane(self.data.clone()));
        self.resume();
        self.stale = true;
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let (_, _, kind, hidden, width, act, stop) = TRIALS[k];
        (self.data_kind, self.noise, self.points, self.gain) = (kind, 0.2, 220, 1.0);
        (self.t.hidden, self.t.width, self.t.act) = (hidden, width, act);
        // (the captions are about the optimiser and settings the scene opens with)
        self.t.usual_settings(Optim::Adam, 0.02, usize::MAX, 0.0);
        self.stop_when_separated = stop;
        self.data = make_data(self.data_kind, self.points, self.noise, &mut self.rng);
        self.rebuild();
        self.t.running = true;
        (self.stage, self.goal, self.sweep, self.sweep_t) = (0.0, None, true, 0.0);
    }

    /// Something was changed: if training had stopped only because the classes were
    /// separated, it starts again.
    fn resume(&mut self) {
        if self.halted {
            self.t.running = true;
        }
        self.halted = false;
    }

    /// Send the view to a whole layer, travelling there smoothly.
    fn go_to(&mut self, layer: f32) {
        self.goal = Some(layer.clamp(0.0, self.hidden() as f32));
        self.sweep = false;
    }

    /// Run the sheet and the data through the network as it now stands, and work out
    /// where each layer puts them.
    fn refresh(&mut self) {
        let hidden = self.hidden();
        let act = self.t.act();
        let flat = |x: &[f32]| x.chunks_exact(2).map(|p| Vec3::new(p[0], p[1], 0.0)).collect::<Vec<Vec3>>();
        self.sheet_at.resize(hidden + 1, Vec::new());
        self.data_at.resize(hidden + 1, Vec::new());
        self.sheet_at[0] = flat(&self.grid);
        self.data_at[0] = flat(&self.data.x);

        // the sheet decides how each layer is shown
        self.output = self.t.net_mut().forward(&self.grid, SHEET_N * SHEET_N).to_vec();
        let old = std::mem::take(&mut self.projections);
        // (a change of scale is eased over the scene's time while training goes on,
        // whatever the frame rate, and made at once when it is not)
        let ease = if self.t.running { 1.0 - (-self.since_refresh / RESCALE).exp() } else { 1.0 };
        self.since_refresh = 0.0;
        for l in 0..hidden {
            let p = Projection::of(self.t.net(), l, act, old.get(l), ease);
            p.apply(self.t.net(), l, &mut self.sheet_at[l + 1]);
            self.projections.push(p);
        }
        // the data points are carried the same way
        let n = self.data.len();
        self.t.net_mut().forward(&self.data.x, n);
        for l in 0..hidden {
            self.projections[l].apply(self.t.net(), l, &mut self.data_at[l + 1]);
        }
        self.loss = self.t.net_mut().loss(&self.data.y, Loss::Logistic);
        self.acc = share_right(self.t.net().neuron(hidden, 0), &self.data.y);
        self.t.record([self.loss; 2], [self.acc; 2]);
    }

    fn separated(&self) -> bool {
        separated(self.acc, self.loss)
    }
}

/// Are the classes apart: every point on the right side of the cut, and not only
/// just? (One point in 220 on the wrong side is 99.5 % right, and not separated.)
fn separated(accuracy: f32, loss: f32) -> bool {
    accuracy >= 1.0 && loss < 0.08
}

impl Scene for Fold {
    fn view(&self) -> View {
        View::Orbit { target: Vec3::ZERO, yaw: -0.9, pitch: 0.42, dist: 15.5 * (0.74 / self.shape).max(1.0) }
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn auto_rotate(&self) -> f32 {
        0.1
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "The last neuron of a network can only make one flat cut through whatever the layers before hand it. So the layers' job is to bend the input until a flat cut is enough. The sheet is the plane of inputs as one layer sees it: travel through the layers and watch it being bent.");
        if self.t.ui_run(ui) {
            self.rebuild();
        }
        ui.checkbox(&mut self.stop_when_separated, "stop once the classes are separated");
        if self.halted && !self.t.running {
            ui::note(ui, "Separated, so training has stopped. Press \"train\" to push on (saturating activations then squash everything into the corners), or reset the weights to watch it happen again.");
        }

        ui::section(ui, "Depth through the network");
        let hidden = self.hidden() as f32;
        ui.horizontal(|ui| {
            if ui.button("◀ previous layer").clicked() {
                self.go_to((self.goal.unwrap_or(self.stage) - 1.0).ceil());
            }
            if ui.button("next layer ▶").clicked() {
                self.go_to((self.goal.unwrap_or(self.stage) + 1.0).floor());
            }
        });
        ui.horizontal(|ui| {
            if ui.button("⇤ input").clicked() {
                self.go_to(0.0);
            }
            if ui.button("last layer ⇥").clicked() {
                self.go_to(hidden);
            }
        });
        if ui::slider(ui, "layer", &mut self.stage, 0.0..=hidden) {
            self.sweep = false;
            self.goal = None;
        }
        if ui.checkbox(&mut self.sweep, "sweep back and forth").changed() && self.sweep {
            self.goal = None;
        }
        ui::note(ui, "Layer 0 is the flat input plane. Layer $k$ is where hidden layer $k$ puts each point: one axis per neuron.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Data");
        let mut changed = ui::choice(ui, &DATASETS, &mut self.data_kind);
        changed |= ui::slider(ui, "noise", &mut self.noise, 0.0..=1.0);
        changed |= ui::slider_int(ui, "points", &mut self.points, 40..=400);
        if ui.button("new sample").clicked() || changed {
            self.resample();
        }

        ui::section(ui, "Network");
        let mut arch = self.t.ui_network(ui, MAX_HIDDEN, MAX_WIDTH);
        arch |= ui::slider_log(ui, "initial weight scale", &mut self.gain, 0.2..=4.0);
        arch |= self.t.ui_activation(ui);
        if arch {
            self.rebuild();
        }
        ui::note(ui, "With three neurons in a layer, each axis is one neuron. With two, the sheet stays flat; with more than three, it is shown along the three directions in which the layer's activity varies most.");
        self.t.ui_optimizer(ui);
        self.t.ui_training(ui, self.data.len());

        ui::section(ui, "Reading the picture");
        let heat = Cmap::CoolWarm.table();
        ui::key(
            ui,
            &[
                (heat[table_index(0.08)], "sheet, blue: inputs the network calls blue"),
                (heat[table_index(0.92)], "sheet, orange: inputs it calls orange"),
                (CLASS[0], "balls: the data points, in the colour of their true class"),
                (pal::DIM, "dark lines on the sheet: the square grid of the input plane, carried along"),
                (pal::WHITE, "white plane (last layer): the output neuron's cut"),
            ],
        );
        ui::note(ui, "The wire cube is the range of the neurons' values, −1 to 1 for tanh. Activations without a fixed range are drawn with three axes instead.");

        ui::section(ui, "Numbers");
        self.t.ui_numbers(ui);
        ui::readout(ui, "loss", loss_number(self.loss));
        ui::readout(ui, "accuracy", format!("{:.1} %", self.acc * 100.0));
        let layer = self.stage.round() as usize;
        if let Some(p) = layer.checked_sub(1).and_then(|l| self.projections.get(l)) {
            if p.captured < 1.0 {
                ui::readout(ui, &format!("layer {layer}: variation shown"), format!("{:.1} %", p.captured * 100.0));
            }
        }
        ui::note(ui, "The loss is how wrong the network is on the data (0.69 is a coin toss); the accuracy is the share of points on the right side of the cut.");
        ui::see_also(ui, &[("mlp", "the same networks, neuron by neuron"), ("signal", "what depth does before any training"), ("linmap3", "what one matrix does to space")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        self.t.advance(f.dt, f.real_dt);
        self.since_refresh += f.dt.max(0.0);
        let fresh = self.t.refresh_due(self.stale);
        if fresh {
            let start = clock::now();
            self.refresh();
            self.t.refreshed(clock::now() - start);
            self.stale = false;
            // by default stop at the first moment the classes are apart: pushed further,
            // saturating activations squash every point into a corner
            if self.stop_when_separated && !self.halted && self.t.running && self.separated() {
                self.t.running = false;
                self.halted = true;
            }
        }

        // where in the network the view is
        let hidden = self.hidden() as f32;
        if let Some(goal) = self.goal {
            let step = f.real_dt * TRAVEL;
            if (goal - self.stage).abs() <= step {
                self.stage = goal;
                self.goal = None;
            } else {
                self.stage += step * (goal - self.stage).signum();
            }
        } else if self.sweep {
            // dwell at both ends
            self.sweep_t += f.dt * 0.22;
            let tri = 1.0 - (2.0 * (self.sweep_t.fract()) - 1.0).abs();
            self.stage = hidden * ((tri * 1.5 - 0.25).clamp(0.0, 1.0));
        }
        self.stage = self.stage.clamp(0.0, hidden);
        let stage = self.stage;
        let k = FOLD_SCALE;
        // between two layers, every point glides from one picture to the next
        let below = (stage.floor() as usize).min(self.hidden().saturating_sub(1));
        let part = stage - below as f32;
        let part = part * part * (3.0 - 2.0 * part);

        if fresh || stage != self.drawn_stage {
            let (a, b) = (&self.sheet_at[below], &self.sheet_at[below + 1]);
            for (i, p) in self.sheet.data_mut().iter_mut().enumerate() {
                let r = a[i].lerp(b[i], part) * k;
                *p = [r.x, r.y, r.z, self.output[i]];
            }
            self.drawn_stage = stage;
        }
        f.surface(&mut self.sheet);
        let (a, b) = (&self.data_at[below], &self.data_at[below + 1]);
        for i in 0..self.data.len().min(a.len()) {
            f.sphere(a[i].lerp(b[i], part) * k, 0.13, CLASS[self.data.y[i] as usize]);
        }

        // what bounds the layer being looked at: a cube if its neurons live in a fixed
        // range, otherwise just the three axes
        let nearest = (stage.round() as usize).max(1) - 1;
        let fade = stage.min(1.0);
        if self.projections[nearest].boxed {
            for (a, b) in [
                ((-1., -1., -1.), (1., -1., -1.)), ((-1., 1., -1.), (1., 1., -1.)), ((-1., -1., 1.), (1., -1., 1.)), ((-1., 1., 1.), (1., 1., 1.)),
                ((-1., -1., -1.), (-1., 1., -1.)), ((1., -1., -1.), (1., 1., -1.)), ((-1., -1., 1.), (-1., 1., 1.)), ((1., -1., 1.), (1., 1., 1.)),
                ((-1., -1., -1.), (-1., -1., 1.)), ((1., -1., -1.), (1., -1., 1.)), ((-1., 1., -1.), (-1., 1., 1.)), ((1., 1., -1.), (1., 1., 1.)),
            ] {
                f.line(Vec3::new(a.0, a.1, a.2) * k, Vec3::new(b.0, b.1, b.2) * k, 1.0, pal::DIM.a(0.45 * fade));
            }
            // one axis per neuron (a layer of two has no third)
            let width = self.t.net().sizes[nearest + 1];
            if stage > 0.5 {
                // (named at the middle of an edge along it: of the four such edges, the
                // one furthest out in the picture, so that the name stays off the sheet)
                let centre = f.cam.project(Vec3::ZERO).unwrap_or_default();
                for n in 0..width.min(3) {
                    let mut best = (Vec3::ZERO, -1.0f32);
                    for (a, b) in [(-1.0f32, -1.0f32), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)] {
                        let mut at = Vec3::ZERO;
                        at[(n + 1) % 3] = a * 1.13;
                        at[(n + 2) % 3] = b * 1.13;
                        let out = f.cam.project(at * k).map_or(-1.0, |p| (p - centre).length());
                        if out > best.1 {
                            best = (at, out);
                        }
                    }
                    f.text(best.0 * k, format!("neuron {}", n + 1), 11.0, pal::DIM.a(fade), Align2::CENTER_CENTER);
                }
            }
        } else {
            for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                f.line(-axis * k * 1.15, axis * k * 1.15, 1.0, pal::DIM.a(0.4 * fade));
            }
        }

        // in the last hidden layer, the output neuron's decision surface is a flat plane
        let last = self.hidden();
        let reveal = (stage - (hidden - 1.0)).clamp(0.0, 1.0);
        if reveal > 0.02 {
            let p = &self.projections[last - 1];
            let net = self.t.net();
            let weights = &net.w[last];
            // the neuron's weights, seen along the three directions drawn
            let w = Vec3::from_array([0, 1, 2].map(|a| p.axes[a].iter().zip(weights).map(|(x, y)| x * y).sum::<f32>()));
            let offset = net.b[last][0] + weights.iter().zip(&p.origin).map(|(x, y)| x * y).sum::<f32>();
            if w.length() > 1e-4 {
                let nrm = w.normalize();
                let centre = nrm * (-offset * p.scale / w.length()) * k;
                if centre.length() < 3.0 * k {
                    let u = nrm.any_orthonormal_vector();
                    let v = nrm.cross(u);
                    let r = 1.2 * k;
                    let q = [centre - u * r - v * r, centre + u * r - v * r, centre + u * r + v * r, centre - u * r + v * r];
                    f.dl.tris_alpha.quad(q[0], q[1], q[2], q[3], pal::WHITE.a(0.13 * reveal));
                    f.dl.lines.closed(&q, 1.6, pal::WHITE.a(0.8 * reveal));
                    f.text(q[2], "the output neuron's cut", 13.0, pal::WHITE.a(reveal), Align2::LEFT_BOTTOM);
                }
            }
        }

        let label = if stage < 0.05 {
            "the input plane".to_string()
        } else if (stage - stage.round()).abs() < 0.05 {
            format!("as hidden layer {} sees it", stage.round() as usize)
        } else {
            format!("between layers {} and {}", stage.floor() as usize, stage.floor() as usize + 1)
        };
        let s = f.view_size();
        // a tall canvas (a phone held upright) needs the camera further back, and has
        // the application's own title where this one would go
        let shape = s.x / s.y.max(1.0);
        if (shape.min(0.74) / self.shape.min(0.74) - 1.0).abs() > 0.04 {
            self.shape = shape;
            self.new_view = true;
        }
        let head = if s.x < 620.0 { 50.0 } else { 20.0 };
        f.text_px(Vec2::new(s.x * 0.5, head), label, 18.0, pal::WHITE, Align2::CENTER_TOP);
        let p = &self.projections[nearest];
        if stage > 0.5 && p.captured < 1.0 {
            let width = self.t.net().sizes[nearest + 1];
            let note = format!("its {width} neurons along their three main directions ({:.0} % of the variation)", p.captured * 100.0);
            if text_width(&note, 12.5) < s.x - 20.0 {
                f.text_px(Vec2::new(s.x * 0.5, head + 26.0), note, 12.5, pal::DIM, Align2::CENTER_TOP);
            } else {
                f.text_px(Vec2::new(s.x * 0.5, head + 26.0), format!("along its three main directions ({:.0} %)", p.captured * 100.0), 12.5, pal::DIM, Align2::CENTER_TOP);
            }
        }

        // the loss so far
        let size = Vec2::new((s.x * 0.24).clamp(170.0, 300.0), (s.y * 0.14).clamp(60.0, 110.0));
        let origin = Vec2::new(26.0, s.y - size.y - 26.0);
        hud_chart(f, origin, size, &[(self.t.loss[0].points(), pal::CYAN)], true, None);
        f.text_px(origin + Vec2::new(0.0, -12.0), format!("loss (log scale)  {}    accuracy  {:.0} %", loss_number(self.loss), self.acc * 100.0), 11.5, pal::TEXT, Align2::LEFT_BOTTOM);

        // what is what, beside the chart if there is room and otherwise above it
        let lines = ["sheet: the plane of inputs, as this layer has bent it", "its colour: the network's answer, blue or orange", "balls: the data points, coloured by their true class"];
        for (i, line) in lines.iter().enumerate() {
            let up = (lines.len() - 1 - i) as f32 * 17.0;
            if s.x > 700.0 {
                f.caption_px(Vec2::new(s.x - 18.0, s.y - 18.0 - up), *line, 11.5, pal::DIM, Align2::RIGHT_BOTTOM);
            } else {
                f.caption_px(Vec2::new(18.0, origin.y - 36.0 - up), *line, 11.5, pal::DIM, Align2::LEFT_BOTTOM);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn network(width: usize, act: Act, seed: u64) -> (Mlp, Vec<f32>) {
        let mut rng = Rng::new(seed);
        let mut net = Mlp::new(&[2, width, 1], act, &mut rng);
        let grid = input_grid(12, 1.1);
        net.forward(&grid, 144);
        (net, grid)
    }

    #[test]
    fn three_neurons_are_drawn_as_themselves() {
        let (net, _) = network(3, Act::Tanh, 1);
        let p = Projection::of(&net, 0, Act::Tanh, None, 1.0);
        assert!(p.boxed && p.captured == 1.0 && p.scale == 1.0);
        let mut at = Vec::new();
        p.apply(&net, 0, &mut at);
        for (i, point) in at.iter().enumerate() {
            let direct = Vec3::new(net.neuron(0, 0)[i], net.neuron(0, 1)[i], net.neuron(0, 2)[i]);
            assert!((*point - direct).length() < 1e-6);
            assert!(point.abs().max_element() <= 1.0);
        }
        // a sigmoid's range is [0, 1]: recentred to fill the same cube
        let (net, _) = network(2, Act::Sigmoid, 1);
        let p = Projection::of(&net, 0, Act::Sigmoid, None, 1.0);
        p.apply(&net, 0, &mut at);
        assert!(at.iter().all(|q| q.abs().max_element() <= 1.0 && q.z == 0.0));
    }

    #[test]
    fn wide_layers_are_shown_along_their_main_directions() {
        let (net, _) = network(9, Act::Relu, 2);
        let p = Projection::of(&net, 0, Act::Relu, None, 1.0);
        // orthonormal directions, most of the variation, everything inside the cube
        for a in 0..3 {
            for b in 0..3 {
                let dot: f32 = p.axes[a].iter().zip(&p.axes[b]).map(|(x, y)| x * y).sum();
                assert!((dot - if a == b { 1.0 } else { 0.0 }).abs() < 1e-4);
            }
        }
        assert!(p.captured > 0.8 && p.captured <= 1.0 && !p.boxed);
        let mut at = Vec::new();
        p.apply(&net, 0, &mut at);
        let reach = at.iter().map(|q| q.abs().max_element()).fold(0.0, f32::max);
        assert!((reach - 1.0).abs() < 1e-3, "reach {reach}");
        // asked again with itself as the previous picture, the directions do not move
        let again = Projection::of(&net, 0, Act::Relu, Some(&p), 1.0);
        for a in 0..3 {
            let dot: f32 = p.axes[a].iter().zip(&again.axes[a]).map(|(x, y)| x * y).sum();
            assert!(dot > 0.999, "direction {a} turned: {dot}");
        }
        // and if the directions come back flipped and reordered, they are turned back
        let mut scrambled = p.clone();
        scrambled.axes = [p.axes[2].iter().map(|v| -v).collect(), p.axes[0].clone(), p.axes[1].iter().map(|v| -v).collect()];
        let restored = Projection::of(&net, 0, Act::Relu, Some(&scrambled), 1.0);
        for a in 0..3 {
            let dot: f32 = scrambled.axes[a].iter().zip(&restored.axes[a]).map(|(x, y)| x * y).sum();
            assert!(dot > 0.999, "direction {a} not matched to the previous picture: {dot}");
        }
    }

    /// The share of a data set that a network of the given shape gets right after
    /// training on all of it with Adam.
    fn accuracy_after(kind: usize, hidden: usize, width: usize, act: Act, seed: u64, steps: usize) -> f32 {
        let mut rng = Rng::new(seed);
        let data = make_data(kind, 220, 0.2, &mut rng);
        let mut net = Mlp::new(&layer_sizes(2, hidden, width, 1), act, &mut rng);
        for _ in 0..steps {
            net.train(&data.x, &data.y, data.len(), 0.02, Loss::Logistic);
        }
        net.forward(&data.x, data.len());
        let hits = net.neuron(hidden, 0).iter().zip(&data.y).filter(|(o, y)| (**o > 0.0) == (**y > 0.5)).count();
        hits as f32 / data.len() as f32
    }

    /// The experiments' captions, held to account.
    #[test]
    fn two_neurons_cannot_free_the_disc_and_three_can() {
        for seed in 1..=4 {
            let two = accuracy_after(1, 2, 2, Act::Tanh, seed, 3000);
            let three = accuracy_after(1, 2, 3, Act::Tanh, seed, 600);
            assert!(two < 0.99, "seed {seed}: two neurons per layer reached {two}");
            assert!(three > 0.995, "seed {seed}: three neurons per layer reached only {three}");
            assert!(accuracy_after(3, 2, 3, Act::Tanh, seed, 600) > 0.995, "seed {seed}: xor");
        }
        assert!(accuracy_after(0, 3, 8, Act::Tanh, 3, 3000) > 0.995, "spirals");
    }

    /// What the experiments lead to (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for (name, kind, hidden, width, act) in [
            ("circles 2x2 tanh", 1usize, 2usize, 2usize, Act::Tanh),
            ("circles 4x2 tanh", 1, 4, 2, Act::Tanh),
            ("circles 2x3 tanh", 1, 2, 3, Act::Tanh),
            ("circles 2x3 relu", 1, 2, 3, Act::Relu),
            ("circles 2x2 relu", 1, 2, 2, Act::Relu),
            ("xor 2x3 tanh", 3, 2, 3, Act::Tanh),
            ("xor 2x2 tanh", 3, 2, 2, Act::Tanh),
            ("moons 2x3 tanh", 2, 2, 3, Act::Tanh),
            ("spirals 3x8 tanh", 0, 3, 8, Act::Tanh),
        ] {
            let mut line = format!("{name:<18}");
            for steps in [600, 3000] {
                line += &format!("   {steps} steps:");
                for seed in 1..=6 {
                    line += &format!(" {:.2}", accuracy_after(kind, hidden, width, act, seed, steps));
                }
            }
            println!("{line}");
        }
    }

    /// "Stop once the classes are separated" stops only when every point is on the
    /// right side (it used to stop with one in 220 still wrong, as it did here for
    /// three of the eight seeds), and with three neurons a layer the ring and the
    /// xor corners do get there within a thousand steps, as the captions promise.
    #[test]
    fn training_stops_only_once_every_point_is_right() {
        for (kind, name) in [(1usize, "ring"), (3, "xor")] {
            for seed in 1..=8u64 {
                let mut rng = Rng::new(seed);
                let data = make_data(kind, 220, 0.2, &mut rng);
                let mut net = Mlp::new(&layer_sizes(2, 2, 3, 1), Act::Tanh, &mut rng);
                let mut stopped = None;
                for step in 1..=1000 {
                    net.forward(&data.x, data.len());
                    net.backward(&data.y, Loss::Logistic);
                    net.step(Optim::Adam, 0.02, 0.9, 0.0);
                    net.forward(&data.x, data.len());
                    let loss = net.loss(&data.y, Loss::Logistic);
                    if separated(share_right(net.neuron(2, 0), &data.y), loss) {
                        stopped = Some(step);
                        break;
                    }
                }
                assert!(stopped.is_some(), "{name}, seed {seed}: not separated after 1000 steps");
                let wrong = net.neuron(2, 0).iter().zip(&data.y).filter(|(o, y)| (**o > 0.0) != (**y > 0.5)).count();
                assert_eq!(wrong, 0, "{name}, seed {seed}: stopped at step {stopped:?} with {wrong} points wrong");
            }
        }
    }
}


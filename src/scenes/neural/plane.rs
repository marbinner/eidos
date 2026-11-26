//! The playground's first task: sorting points in the plane into two classes.

use super::session::Problem;
use super::training::Training;
use super::*;
use crate::clock;

/// The network's output is drawn over this many points per side.
const FIELD_N: usize = 88;
/// Each neuron's tile shows every fourth point of that grid, so one pass over the
/// grid gives every picture.
const TILE_STEP: usize = 4;
const TILE_N: usize = FIELD_N / TILE_STEP;
const MAX_WIDTH: usize = 8;
const MAX_HIDDEN: usize = 4;
/// The view that holds everything this task draws.
pub const VIEW: View = View::Fit { center: Vec2::new(0.2, 0.12), half: Vec2::new(4.25, 2.5) };
/// The tiles live side by side in one picture, this many per row.
const ATLAS_COLS: usize = 8;
const ATLAS_ROWS: usize = (2 + MAX_HIDDEN * MAX_WIDTH + 1).div_ceil(ATLAS_COLS);
/// How a knocked-out neuron is silenced.
const KNOCK: [&str; 2] = ["set to zero", "set to its average"];
/// How much each neuron matters is worked out afresh every this many redraws.
const IMPORTANCE_EVERY: u32 = 10;

/// The first layer of a sin network starts with weights this many times the usual
/// size. A sin neuron draws stripes, and at the usual size the stripes are broader
/// than the whole picture, which leaves the network a tanh in all but name; narrower
/// ones are something to build with from the first step.
const SIN_FIRST_LAYER: f32 = 3.0;

/// The optimiser and settings every experiment runs with, which are those the scene
/// opens with: optimiser, learning rate, batch size, weight decay.
const USUAL: (Optim, f32, usize, f32) = (Optim::Adam, 0.02, 32, 0.0);

/// One of the "things to try": a data set and a network that show one lesson.
#[derive(Clone, Copy)]
struct Trial {
    button: &'static str,
    look_for: &'static str,
    /// An index into `DATASETS`.
    data: usize,
    noise: f32,
    hidden: usize,
    width: usize,
    act: Act,
    show_test: bool,
}

/// The "things to try". The tests run the same table.
const TRIALS: [Trial; 5] = [
    Trial {
        button: "Two neurons cannot close a ring",
        look_for: "The boundary can only be a band or a wedge, so about one point in ten stays on the wrong side and the loss stalls. Then raise \"neurons per layer\" to 3: three edges can make a triangle round the inner class. (Now and then a start gets stuck with a wedge; reset the weights.)",
        data: 1,
        noise: 0.15,
        hidden: 1,
        width: 2,
        act: Act::Tanh,
        show_test: false,
    },
    Trial {
        button: "ReLU draws with straight lines",
        look_for: "Each ReLU neuron is one fold. Six of them close a polygon round the middle within a few hundred steps; watch its corners.",
        data: 1,
        noise: 0.15,
        hidden: 1,
        width: 6,
        act: Act::Relu,
        show_test: false,
    },
    Trial {
        button: "Depth for spirals",
        look_for: "First-layer tiles stay straight edges; the tiles of later layers are bent shapes built from them, and a few thousand steps on the boundary winds round both arms.",
        data: 0,
        noise: 0.25,
        hidden: 3,
        width: 8,
        act: Act::Tanh,
        show_test: false,
    },
    Trial {
        button: "Fitting the noise",
        look_for: "Noisy data, a big network. The orange loss (held-out points, drawn pale) soon parts company with the cyan one (training points) and ends well above it. That gap is what the network has learned about these particular points rather than about the pattern: it is fitting the noise.",
        data: 2,
        noise: 0.95,
        hidden: 4,
        width: 8,
        act: Act::Tanh,
        show_test: true,
    },
    Trial {
        button: "A wave for stripes",
        look_for: "With a periodic activation one layer has the stripes within a thousand steps. Switch the activation to tanh: it gets there in the end, but takes several times as long.",
        data: 6,
        noise: 0.1,
        hidden: 1,
        width: 8,
        act: Act::Sin,
        show_test: false,
    },
];

/// A fresh network as this task starts it.
fn fresh_net(sizes: &[usize], act: Act, rng: &mut Rng) -> Mlp {
    let mut net = Mlp::new(sizes, act, rng);
    if act == Act::Sin {
        net.w[0].iter_mut().for_each(|w| *w *= SIN_FIRST_LAYER);
    }
    net
}

/// Silence some hidden neurons. Each entry is ((column, index), value): the neuron's
/// output is replaced by `value` for everything downstream, by zeroing its outgoing
/// weights and adding `value * weight` to the biases they fed. (Column 0 is the
/// inputs.) Used inside [`Training::with_edited`], which puts the weights back.
fn knock(net: &mut Mlp, knocked: &[((usize, usize), f32)]) {
    for &((l, j), value) in knocked {
        let (n_in, n_out) = (net.sizes[l], net.sizes[l + 1]);
        for k in 0..n_out {
            net.b[l][k] += value * net.w[l][k * n_in + j];
            net.w[l][k * n_in + j] = 0.0;
        }
    }
}

pub struct Plane {
    data_kind: usize,
    noise: f32,
    train: Data,
    test: Data,
    t: Training,
    rng: Rng,
    field: Surface,
    tiles: Image,
    grid: Vec<f32>,
    hover: Option<(usize, usize)>,
    show_test: bool,
    /// Loss on the training data, on the held-out data, and held-out accuracy.
    stats: [f32; 3],
    /// Something other than training changed what the pictures should show.
    stale: bool,
    /// The node chosen by a click: (column, index), column 0 being the inputs.
    picked: Option<(usize, usize)>,
    /// Hidden neurons knocked out of the network as it is drawn and scored (training
    /// goes on with all of them, as in an ablation study).
    knocked: Vec<(usize, usize)>,
    knock_mode: usize,
    /// For every hidden neuron, column by column: its average output over the
    /// training data, and how much the training loss rises without it.
    means: Vec<f32>,
    importance: Vec<f32>,
    importance_age: u32,
    /// Training loss, held-out loss and held-out accuracy with the knock-outs applied.
    knocked_stats: [f32; 3],
}

impl Plane {
    pub fn new(gfx: &Gfx) -> Self {
        let mut rng = Rng::new(17);
        let train = make_data(0, 260, 0.25, &mut rng);
        let test = make_data(0, 160, 0.25, &mut rng);
        let mut t = Training::new(2, 8);
        let (optim, lr, batch, decay) = USUAL;
        t.usual_settings(optim, lr, batch, decay);
        t.load(fresh_net(&t.sizes(2, 1), t.act(), &mut rng), Problem::Plane(train.clone()));
        let field = answer_field(gfx, FIELD_N);
        let mut tiles = Image::new(gfx, (ATLAS_COLS * TILE_N) as u32, (ATLAS_ROWS * TILE_N) as u32);
        tiles.smooth = true;
        Plane {
            data_kind: 0,
            noise: 0.25,
            train,
            test,
            t,
            rng,
            field,
            tiles,
            grid: input_grid(FIELD_N, DATA_R),
            hover: None,
            show_test: false,
            stats: [0.0; 3],
            stale: true,
            picked: None,
            knocked: Vec::new(),
            knock_mode: 0,
            means: Vec::new(),
            importance: Vec::new(),
            importance_age: 0,
            knocked_stats: [0.0; 3],
        }
    }

    pub fn running(&self) -> bool {
        self.t.busy()
    }

    fn rebuild(&mut self) {
        let net = fresh_net(&self.t.sizes(2, 1), self.t.act(), &mut self.rng);
        self.t.load(net, Problem::Plane(self.train.clone()));
        // a new network: the old choices of neuron mean nothing in it
        self.picked = None;
        self.knocked.clear();
        self.importance.clear();
        self.importance_age = 0;
        self.stale = true;
    }

    /// One of the "things to try" (`TRIALS`): a data set and a network that show one lesson.
    fn set_up(&mut self, k: usize) {
        let trial = TRIALS[k];
        (self.data_kind, self.noise, self.show_test) = (trial.data, trial.noise, trial.show_test);
        (self.t.hidden, self.t.width) = (trial.hidden, trial.width);
        self.t.act = Act::ALL.iter().position(|&a| a == trial.act).unwrap_or(0);
        // (the captions are about the optimiser and settings the scene opens with)
        let (optim, lr, batch, decay) = USUAL;
        self.t.usual_settings(optim, lr, batch, decay);
        self.t.running = true;
        self.regenerate();
        self.rebuild();
    }

    /// Is (column, index) a hidden neuron (one that can be knocked out)?
    fn is_hidden(&self, node: (usize, usize)) -> bool {
        node.0 >= 1 && node.0 < self.t.net().layers()
    }

    /// Where hidden neuron (column, index) is in `means` and `importance`.
    fn hidden_index(&self, (l, j): (usize, usize)) -> usize {
        self.t.net().sizes[1..l].iter().sum::<usize>() + j
    }

    /// Every hidden neuron, column by column.
    fn hidden_nodes(&self) -> Vec<(usize, usize)> {
        let sizes = &self.t.net().sizes;
        (1..sizes.len() - 1).flat_map(|l| (0..sizes[l]).map(move |j| (l, j))).collect()
    }

    /// The knock-outs with the value each neuron is replaced by.
    fn knock_list(&self, nodes: &[(usize, usize)]) -> Vec<((usize, usize), f32)> {
        nodes.iter().map(|&n| (n, if self.knock_mode == 1 { self.means.get(self.hidden_index(n)).copied().unwrap_or(0.0) } else { 0.0 })).collect()
    }

    fn toggle_knocked(&mut self, node: (usize, usize)) {
        match self.knocked.iter().position(|&n| n == node) {
            Some(i) => {
                self.knocked.remove(i);
            }
            None => self.knocked.push(node),
        }
        self.importance_age = 0;
        self.stale = true;
    }

    fn regenerate(&mut self) {
        self.train = make_data(self.data_kind, 260, self.noise, &mut self.rng);
        self.test = make_data(self.data_kind, 160, self.noise, &mut self.rng);
        self.t.set_problem(Problem::Plane(self.train.clone()));
        // (how much each neuron matters is a statement about the data)
        self.importance_age = 0;
        self.stale = true;
    }

    /// Where node `j` of column `l` sits. Column 0 is the inputs; the last is the output.
    fn node_pos(&self, l: usize, j: usize) -> Vec2 {
        let sizes = &self.t.net().sizes;
        let (cols, count) = (sizes.len(), sizes[l]);
        let x = 1.15 + 2.95 * l as f32 / (cols - 1) as f32;
        let pitch = (2.72 / count.max(3) as f32).min(0.62);
        let y = 0.62 + (count as f32 - 1.0) * 0.5 * pitch - j as f32 * pitch;
        Vec2::new(x, y)
    }

    fn node_size(&self) -> f32 {
        let widest = self.t.net().sizes.iter().copied().max().unwrap_or(1);
        (2.72 / widest.max(3) as f32 * 0.78).min(0.44)
    }

    /// The node under the pointer, as (column, index).
    fn pick(&self, mouse: Option<Vec2>) -> Option<(usize, usize)> {
        let (m, h) = (mouse?, self.node_size() * 0.5);
        let sizes = &self.t.net().sizes;
        for l in 0..sizes.len() {
            for j in 0..sizes[l] {
                if (m - self.node_pos(l, j)).abs().max_element() < h {
                    return Some((l, j));
                }
            }
        }
        None
    }

    /// Recompute the losses and every picture from the current weights.
    fn refresh(&mut self) {
        let layers = self.t.net().layers();
        let hidden = self.hidden_nodes();
        // the network as it is trained: its scores are what the charts follow
        let lt = self.t.net_mut().eval(&self.train.x, &self.train.y, self.train.len(), Loss::Logistic);
        self.means.clear();
        self.means.extend(hidden.iter().map(|&(l, j)| {
            let out = self.t.net().neuron(l - 1, j);
            out.iter().sum::<f32>() / out.len().max(1) as f32
        }));
        let (test, n_test) = (&self.test, self.test.len());
        let held_out = |net: &mut Mlp| {
            let loss = net.eval(&test.x, &test.y, n_test, Loss::Logistic);
            (loss, share_right(net.neuron(layers - 1, 0), &test.y))
        };
        let (le, accuracy) = held_out(self.t.net_mut());
        self.stats = [lt, le, accuracy];
        self.t.record([lt, le], [0.0, accuracy]);

        // the same with the knocked-out neurons silenced
        self.knocked.retain(|&(l, j)| l >= 1 && l < layers && j < self.t.net().sizes[l]);
        let knocked = self.knock_list(&self.knocked);
        let (train, n_train) = (&self.train, self.train.len());
        self.knocked_stats = if knocked.is_empty() {
            self.stats
        } else {
            self.t.with_edited(
                |net| knock(net, &knocked),
                |net| {
                    let loss = net.eval(&train.x, &train.y, n_train, Loss::Logistic);
                    let (le, accuracy) = held_out(net);
                    [loss, le, accuracy]
                },
            )
        };

        // how much each neuron matters: the loss without it, less the loss with it
        // (the others staying as they are). Worked out only now and then.
        if self.importance_age == 0 || self.importance.len() != hidden.len() {
            let with_all = self.knocked_stats[0];
            let mut importance = Vec::with_capacity(hidden.len());
            for &node in &hidden {
                let was_out = self.knocked.contains(&node);
                let others: Vec<(usize, usize)> = self.knocked.iter().copied().filter(|&n| n != node).chain((!was_out).then_some(node)).collect();
                let list = self.knock_list(&others);
                let loss = self.t.with_edited(|net| knock(net, &list), |net| net.eval(&train.x, &train.y, n_train, Loss::Logistic));
                importance.push(if was_out { with_all - loss } else { loss - with_all });
            }
            self.importance = importance;
        }
        self.importance_age = (self.importance_age + 1) % IMPORTANCE_EVERY;

        // one pass over the grid gives the big picture and every tile
        let grid = &self.grid;
        self.t.with_edited(
            |net| knock(net, &knocked),
            |net| {
                net.forward(grid, FIELD_N * FIELD_N);
            },
        );
        let net = self.t.net();
        let act_range = self.t.act().display_range();
        let output_range = [-6.0, 6.0];

        // the big picture: the output, or the node under the pointer
        let (range, contour) = match self.hover {
            Some((0, j)) => {
                for (k, p) in self.field.data_mut().iter_mut().enumerate() {
                    p[3] = self.grid[2 * k + j];
                }
                ([-DATA_R, DATA_R], 0.0)
            }
            Some((l, j)) if l < layers => {
                for (p, &v) in self.field.data_mut().iter_mut().zip(net.neuron(l - 1, j)) {
                    p[3] = v;
                }
                (act_range, 0.0)
            }
            _ => {
                for (p, &v) in self.field.data_mut().iter_mut().zip(net.neuron(layers - 1, 0)) {
                    p[3] = v;
                }
                // a single contour at zero: the decision boundary
                (output_range, 1.0)
            }
        };
        self.field.style.range = range;
        self.field.style.contour = contour;

        // the tiles, coloured here and uploaded as one picture
        let table = Cmap::CoolWarm.table();
        let width = ATLAS_COLS * TILE_N;
        let pixels = self.tiles.pixels_mut();
        let mut tile = 0;
        for l in 0..=layers {
            let [lo, hi] = if l == 0 {
                [-DATA_R, DATA_R]
            } else if l == layers {
                output_range
            } else {
                act_range
            };
            let scale = 1.0 / (hi - lo);
            for j in 0..net.sizes[l] {
                let origin = (tile / ATLAS_COLS) * TILE_N * width + (tile % ATLAS_COLS) * TILE_N;
                let column = if l == 0 { None } else { Some(net.neuron(l - 1, j)) };
                for ty in 0..TILE_N {
                    let row = &mut pixels[origin + ty * width..][..TILE_N];
                    for (tx, px) in row.iter_mut().enumerate() {
                        let k = (TILE_STEP * ty + TILE_STEP / 2) * FIELD_N + TILE_STEP * tx + TILE_STEP / 2;
                        let v = match column {
                            Some(c) => c[k],
                            None => self.grid[2 * k + j],
                        };
                        *px = table[table_index((v - lo) * scale)];
                    }
                }
                tile += 1;
            }
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui::note(ui, "A small network learning, live, to tell blue points from orange ones. The big picture is its answer at every place in the plane. Each small tile is one neuron: what it responds to across the same plane. Early neurons draw straight edges; later ones combine them into curves. Hover a tile to enlarge it; click it to select it.");
        if self.t.ui_run(ui) {
            self.rebuild();
        }

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.look_for) {
                self.set_up(k);
            }
        }
        ui::note(ui, "After any of these: which neurons does the answer rest on? See \"Knock out neurons\" below.");

        ui::section(ui, "Data");
        let mut data_changed = ui::choice(ui, &DATASETS, &mut self.data_kind);
        data_changed |= ui::slider(ui, "noise", &mut self.noise, 0.0..=1.0);
        if ui.button("new sample").clicked() || data_changed {
            self.regenerate();
        }
        ui.checkbox(&mut self.show_test, "show held-out points");
        ui::section(ui, "Network");
        let mut arch = self.t.ui_network(ui, MAX_HIDDEN, MAX_WIDTH);
        arch |= self.t.ui_activation(ui);
        if self.t.act() == Act::Sin {
            ui::note(
                ui,
                r"Here the first layer of a $\sin$ network starts with weights three times the usual size, so that its stripes begin narrower than the picture.",
            );
        }
        if arch {
            self.rebuild();
        }
        self.t.ui_optimizer(ui);
        self.t.ui_training(ui, self.train.len());

        ui::section(ui, "Knock out neurons");
        ui::note(ui, "To find out what a neuron is for, remove it and see what breaks (an ablation). Click a tile to select it and click it again to knock it out. The yellow bar under each tile is how much the loss would rise without that neuron.");
        if ui::choice(ui, &KNOCK, &mut self.knock_mode) {
            self.importance_age = 0;
            self.stale = true;
        }
        ui::note(
            ui,
            if self.knock_mode == 0 {
                "Its output is replaced by zero."
            } else {
                "Its output is replaced by its average over the training data: this removes what the neuron says about each point but not its overall level, which is usually the fairer test."
            },
        );
        let hidden = self.hidden_nodes();
        ui.horizontal_wrapped(|ui| {
            if ui.button("knock out the least important").clicked() {
                let standing = hidden.iter().filter(|n| !self.knocked.contains(n));
                let least = standing.min_by(|a, b| {
                    let at = |n: &&(usize, usize)| self.importance.get(self.hidden_index(**n)).copied().unwrap_or(f32::MAX);
                    at(a).total_cmp(&at(b))
                });
                if let Some(&node) = least {
                    self.toggle_knocked(node);
                }
            }
            if ui.button("restore all").clicked() && !self.knocked.is_empty() {
                self.knocked.clear();
                self.importance_age = 0;
                self.stale = true;
            }
        });
        ui::readout(ui, "neurons knocked out", format!("{} of {}", self.knocked.len(), hidden.len()));
        if !self.knocked.is_empty() {
            ui::readout(ui, "loss with them out", loss_number(self.knocked_stats[0]));
            ui::readout(ui, "held-out accuracy with them out", format!("{:.1} %", self.knocked_stats[2] * 100.0));
            ui::note(ui, "Training goes on with every neuron in place; the knock-outs change only how the network is read here.");
        }
        if let Some(node) = self.picked.filter(|&n| self.is_hidden(n)) {
            let (l, j) = node;
            let i = self.hidden_index(node);
            let out = self.knocked.contains(&node);
            ui::section(ui, &format!("Selected: layer {l}, neuron {}", j + 1));
            if ui.button(if out { "restore it" } else { "knock it out" }).clicked() {
                self.toggle_knocked(node);
            }
            let net = self.t.net();
            let n_in = net.sizes[l - 1];
            let from = |k: usize| if l == 1 { format!("$x_{}$", k + 1) } else { format!("neuron {}", k + 1) };
            let mut inputs: Vec<(usize, f32)> = (0..n_in).map(|k| (k, net.w[l - 1][j * n_in + k])).collect();
            inputs.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
            let listed: Vec<String> = inputs.iter().take(3).map(|&(k, w)| format!("{} {}", from(k), signed(w, 2))).collect();
            ui::readout(ui, "strongest weights in", listed.join(",  "));
            ui::readout(ui, "bias", signed(net.b[l - 1][j], 2));
            ui::readout(ui, "average output", signed(self.means.get(i).copied().unwrap_or(0.0), 2));
            let rise = self.importance.get(i).copied().unwrap_or(0.0);
            ui::readout(ui, if out { "loss it would win back" } else { "rise in loss without it" }, signed(rise, 3));
        }

        ui::section(ui, "Colours");
        ui::key(
            ui,
            &[
                (CLASS[0], "blue points, and where the network answers blue"),
                (CLASS[1], "orange points, and where it answers orange"),
                (pal::WHITE, "the boundary: where it gives even odds"),
                (pal::YELLOW, "bar under a tile: how much that neuron matters"),
            ],
        );
        ui::note(ui, "In a tile, red is where the neuron's output is high and blue where it is low. A line between tiles is a weight: orange positive, blue negative, thicker for larger.");

        ui::section(ui, "Numbers");
        self.t.ui_numbers(ui);
        ui::readout_colored(ui, "loss, training data", loss_number(self.stats[0]), pal::CYAN);
        ui::readout_colored(ui, "loss, held-out data", loss_number(self.stats[1]), pal::ORANGE);
        ui::readout(ui, "held-out accuracy", format!("{:.1} %", self.stats[2] * 100.0));
        ui::term(ui, "Loss:", "how wrong the answers are, on average (lower is better).");
        ui::term(
            ui,
            "Held-out data:",
            "points from the same source that are never trained on. They show whether the network has learned the pattern or only its training points.",
        );
        ui::see_also(
            ui,
            &[("fold", "How layers fold space"), ("race", "Optimiser race"), ("particles", "Neurons as particles"), ("grokking", "Grokking a table")],
        );
    }

    pub fn frame(&mut self, f: &mut Frame) {
        self.t.advance(f.dt, f.real_dt);
        let hover = self.pick(f.mouse());
        if f.clicked_canvas() {
            match hover {
                // a second click on the selected neuron knocks it out (or restores it)
                Some(node) if self.picked == Some(node) && self.is_hidden(node) => self.toggle_knocked(node),
                other => self.picked = other,
            }
        }
        let force = self.stale || hover != self.hover;
        self.hover = hover;
        if self.t.refresh_due(force) {
            let start = clock::now();
            self.refresh();
            self.t.refreshed(clock::now() - start);
            self.stale = false;
        }

        // the big picture
        let layers = self.t.net().layers();
        let centre = Vec2::new(-1.75, 0.2);
        let scale = 4.3;
        let half = scale * 0.5;
        self.field.style.model = Mat4::from_translation(centre.extend(-0.2)) * Mat4::from_scale(Vec3::new(scale, scale, 1.0));
        f.surface(&mut self.field);
        frame_rect(f, centre - Vec2::splat(half), centre + Vec2::splat(half), 1.4, pal::DIM);
        let to_world = |p: Vec2| centre + p / DATA_R * half;
        f.dl.discs.style.outline_px = 1.1;
        for i in 0..self.train.len() {
            let p = self.train.point(i);
            if p.abs().max_element() < DATA_R {
                f.dot(to_world(p), 6.5, CLASS[self.train.y[i] as usize]);
            }
        }
        if self.show_test {
            for i in 0..self.test.len() {
                let p = self.test.point(i);
                if p.abs().max_element() < DATA_R {
                    f.dot(to_world(p), 4.0, CLASS[self.test.y[i] as usize].mix(pal::WHITE, 0.55));
                }
            }
        }
        let title = match self.hover {
            Some((0, j)) => format!("input $x_{}$", j + 1),
            Some((l, j)) if l < layers => format!("layer {l}, neuron {}", j + 1),
            _ if self.knocked.is_empty() => "the network's answer at every place".to_string(),
            _ => format!(
                "the network's answer with {} knocked out",
                if self.knocked.len() == 1 { "1 neuron".to_string() } else { format!("{} neurons", self.knocked.len()) }
            ),
        };
        f.text(centre + Vec2::new(-half, half + 0.06), title, 14.0, pal::TEXT, Align2::LEFT_BOTTOM);
        // what the colours of the big picture mean
        let (low, high, middle) = match self.hover {
            Some((0, _)) => ("small", "large", ""),
            Some((l, _)) if l < layers => ("low output", "high output", ""),
            _ => ("sure it is blue", "sure it is orange", "undecided"),
        };
        ramp_key(f, Vec2::new(centre.x, centre.y - half - 0.165), 0.72, 0.07, Cmap::CoolWarm, (low, CLASS[0]), (high, CLASS[1]), middle);

        // the network diagram: weights as lines, neurons as small response maps
        let size = self.node_size();
        let cols = self.t.net().sizes.len();
        for l in 0..layers {
            let (n_in, n_out) = (self.t.net().sizes[l], self.t.net().sizes[l + 1]);
            let wmax = self.t.net().w[l].iter().fold(1e-6f32, |m, w| m.max(w.abs()));
            for j in 0..n_out {
                for i in 0..n_in {
                    let w = self.t.net().w[l][j * n_in + i];
                    let s = (w.abs() / wmax).powf(0.7);
                    if s < 0.04 {
                        continue;
                    }
                    let (a, b) = (self.node_pos(l, i), self.node_pos(l + 1, j));
                    let c = if w > 0.0 { CLASS[1] } else { CLASS[0] };
                    f.line(a + Vec2::new(size * 0.5, 0.0), b - Vec2::new(size * 0.5, 0.0), 0.6 + 3.4 * s, c.a(0.15 + 0.7 * s));
                }
            }
        }
        let h = size * 0.5;
        let most = self.importance.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
        let bar_gap = (self.node_pos(1, 0).y - self.node_pos(1, 1.min(self.t.net().sizes[1] - 1)).y - size).clamp(0.02, 0.05) * 0.5;
        let mut tile = 0;
        for l in 0..cols {
            for j in 0..self.t.net().sizes[l] {
                let pos = self.node_pos(l, j);
                let part = [(tile % ATLAS_COLS) * TILE_N, (tile / ATLAS_COLS) * TILE_N, TILE_N, TILE_N].map(|v| v as u32);
                f.image_part(&mut self.tiles, part, pos - Vec2::splat(h), pos + Vec2::splat(h), 0.0);
                tile += 1;
                let (hot, chosen) = (self.hover == Some((l, j)), self.picked == Some((l, j)));
                let corners = [pos + Vec2::new(-h, -h), pos + Vec2::new(h, -h), pos + Vec2::new(h, h), pos + Vec2::new(-h, h)];
                if self.knocked.contains(&(l, j)) {
                    // silenced: dimmed and crossed out
                    f.dl.tris_alpha.rect(corners[0], corners[2], 0.005, Color::hex(0x05070D).a(0.72));
                    f.line(corners[0].extend(0.01), corners[2].extend(0.01), 2.0, pal::RED);
                    f.line(corners[1].extend(0.01), corners[3].extend(0.01), 2.0, pal::RED);
                }
                let (width, colour) = if chosen {
                    (2.6, pal::YELLOW)
                } else if hot {
                    (2.4, pal::WHITE)
                } else {
                    (1.2, pal::DIM)
                };
                f.dl.lines.closed(&corners.map(|p| p.extend(0.01)), width, colour);
                if l >= 1 && l < layers {
                    // how much the loss rises without this neuron, against the most
                    // that any neuron matters
                    let i = self.hidden_index((l, j));
                    if let Some(&rise) = self.importance.get(i) {
                        let share = (rise.abs() / most).clamp(0.0, 1.0);
                        let from = pos + Vec2::new(-h, -h - bar_gap);
                        f.line(from, from + Vec2::new(2.0 * h * share, 0.0), 2.6, if rise >= 0.0 { pal::YELLOW } else { pal::CYAN.a(0.8) });
                    }
                }
            }
        }
        // (on a small canvas there is no room for these; the panel says the same)
        if pixels_per_unit(f) > 98.0 {
            f.text((0.95, -0.86), "yellow bar under a neuron: how much the loss rises without it", 11.0, pal::DIM, Align2::LEFT_TOP);
            wrapped(f, Vec2::new(0.95, -1.83), 3.35, "hover a neuron to enlarge it  ·  click to select  ·  click again to knock it out", 11.0, pal::DIM);
        }
        for j in 0..2 {
            f.text(self.node_pos(0, j) - Vec2::new(h + 0.06, 0.0), ["$x_1$", "$x_2$"][j], 14.0, pal::TEXT, Align2::RIGHT_CENTER);
        }
        f.text(self.node_pos(cols - 1, 0) + Vec2::new(0.0, h + 0.05), "output", 12.5, pal::TEXT, Align2::CENTER_BOTTOM);

        // loss history
        let origin = Vec2::new(0.95, -1.78);
        let csize = Vec2::new(3.35, 0.58);
        chart(f, origin, csize, &[(self.t.loss[0].points(), pal::CYAN), (self.t.loss[1].points(), pal::ORANGE)], true);
        let values = format!("training {}   held-out {}", loss_number(self.stats[0]), loss_number(self.stats[1]));
        // (the losses are typeset, wider when small: measured as drawn, and the chart's
        // name shortened on a canvas too small for both)
        let room = csize.x * pixels_per_unit(f) - f.text_size(&values, 12.0).x - 12.0;
        let name = if f.text_size("loss (log scale)", 12.0).x <= room { "loss (log scale)" } else { "loss" };
        f.text(origin + Vec2::new(0.0, csize.y + 0.06), name, 12.0, pal::DIM, Align2::LEFT_BOTTOM);
        f.text(origin + Vec2::new(csize.x, csize.y + 0.06), values, 12.0, pal::DIM, Align2::RIGHT_BOTTOM);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenes::neural::session::{Session, Settings};

    /// One of the "things to try" as the scene runs it (the same starting network,
    /// optimiser, batches and data), without the pictures: the loss on the training
    /// points, the loss on the held-out points and the share of training points on
    /// the right side, after each of the given numbers of steps.
    fn run(trial: Trial, seed: u64, marks: &[usize]) -> Vec<[f32; 3]> {
        let mut rng = Rng::new(seed);
        let train = make_data(trial.data, 260, trial.noise, &mut rng);
        let test = make_data(trial.data, 160, trial.noise, &mut rng);
        let net = fresh_net(&layer_sizes(2, trial.hidden, trial.width, 1), trial.act, &mut rng);
        let mut job = Session::new(net, Problem::Plane(train.clone()), seed);
        let (optim, lr, batch, decay) = USUAL;
        job.settings = Settings { optim, lr, momentum: optim.default_momentum(), decay, batch, rate: 0.0, running: false };
        let mut out = Vec::new();
        let mut done = 0;
        for &mark in marks {
            job.request_steps((mark - done) as u32);
            job.work(0.0, f64::MAX);
            done = mark;
            let last = job.net.layers() - 1;
            let lt = job.net.eval(&train.x, &train.y, train.len(), Loss::Logistic);
            let right = share_right(job.net.neuron(last, 0), &train.y);
            let le = job.net.eval(&test.x, &test.y, test.len(), Loss::Logistic);
            out.push([lt, le, right]);
        }
        out
    }

    /// Each run over a few starts.
    fn each_start(trial: Trial, marks: &[usize]) -> Vec<Vec<[f32; 3]>> {
        (0..8u64).map(|seed| run(trial, 100 + seed, marks)).collect()
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // two neurons cannot close a ring: about one point in ten stays wrong (between
        // one in twelve and one in six, over these starts), and the loss stalls
        for r in each_start(TRIALS[0], &[10_000]) {
            assert!((0.83..0.93).contains(&r[0][2]) && r[0][0] > 0.2, "two neurons on the ring: {:?}", r[0]);
        }
        // three usually can
        let closed = each_start(Trial { width: 3, ..TRIALS[0] }, &[10_000]).iter().filter(|r| r[0][2] > 0.99).count();
        assert!(closed >= 6, "three neurons closed the ring in {closed} of 8 starts");
        // six ReLU folds close it within a few hundred steps
        for r in each_start(TRIALS[1], &[1000]) {
            assert!(r[0][2] > 0.99, "six ReLU neurons on the ring: {:?}", r[0]);
        }
        // three layers wind the boundary round the spirals
        for r in each_start(TRIALS[2], &[10_000]) {
            assert!(r[0][2] > 0.9, "three layers on the spirals: {:?}", r[0]);
        }
        // noisy moons: the held-out loss parts from the training loss and ends well above it
        let moons = each_start(TRIALS[3], &[300, 3000, 10_000]);
        let mean = |f: &dyn Fn(&Vec<[f32; 3]>) -> f32| moons.iter().map(f).sum::<f32>() / moons.len() as f32;
        let (early, later) = (mean(&|r| r[0][1] - r[0][0]), mean(&|r| r[1][1] - r[1][0]));
        assert!(later > 0.08 && later > 2.0 * early, "gap between held-out and training loss: {early} after 300 steps, {later} after 3000");
        assert!(mean(&|r| r[2][1]) > mean(&|r| r[0][1]) + 0.04, "the held-out loss ends above where it was early on");
        // stripes: sin has them within a thousand steps, tanh takes several times as long
        for r in each_start(TRIALS[4], &[1000]) {
            assert!(r[0][2] > 0.95, "sin on the stripes after 1000 steps: {:?}", r[0]);
        }
        let slow = each_start(Trial { act: Act::Tanh, ..TRIALS[4] }, &[1000, 10_000]);
        assert!(slow.iter().all(|r| r[0][2] < 0.88), "tanh on the stripes after 1000 steps: {slow:?}");
        assert!(slow.iter().filter(|r| r[1][2] > 0.95).count() >= 6, "tanh on the stripes after 10 000 steps: {slow:?}");
    }

    /// What the experiments lead to (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        let marks = [300, 1000, 3000, 10_000];
        let show = |name: &str, trial: Trial| {
            for seed in 0..8u64 {
                let r = run(trial, 100 + seed, &marks);
                let cells: Vec<String> = r.iter().map(|m| format!("{:.3}/{:.3}/{:.0}%", m[0], m[1], m[2] * 100.0)).collect();
                println!("{name:<32} seed {seed}: {}", cells.join("   "));
            }
        };
        for trial in TRIALS {
            show(trial.button, trial);
        }
        show("ring, 3 tanh", Trial { width: 3, ..TRIALS[0] });
        show("stripes, 8 tanh", Trial { act: Act::Tanh, ..TRIALS[4] });
    }
}

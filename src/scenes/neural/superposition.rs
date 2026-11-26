//! Superposition: a network asked to carry more features than it has dimensions.

use super::toy::{importance, Toy};
use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "superposition",
    title: "Superposition",
    group: crate::scenes::interp::GROUP,
    blurb: "More features than dimensions: how a network packs them in.",
    make: |_| Box::new(Superposition::new()),
};

/// Settings that show one arrangement each: (label, what to look for, features,
/// hidden dimensions, sparsity, importance decay). What four of them settle into is
/// checked by the tests; "opposite pairs" and "3D: twelve" depend on the starting
/// weights, and their captions say so.
const PRESETS: [(&str, &str, usize, usize, f32, f32); 6] = [
    ("always on: two survive", "Sparsity 0: every feature is present in every input. The network keeps the two most important, at right angles, and lets the other three arrows shrink to nothing.", 5, 2, 0.0, 0.9),
    ("opposite pairs", "Each feature is absent four times in five. Usually four of them settle as two opposite pairs, half a dimension each, and the fifth is dropped. Some starts find a pentagon instead (press \"new weights\").", 5, 2, 0.8, 0.9),
    ("pentagon", "Each feature is absent nine times in ten, so two are rarely on together. All five fit, as a regular pentagon: $2/5$ of a dimension each.", 5, 2, 0.9, 1.0),
    ("3D: octahedron", "Six features in three dimensions: three opposite pairs, the corners of an octahedron. Half a dimension each.", 6, 3, 0.9, 1.0),
    ("3D: eight", "Eight features share three dimensions equally: $3/8$ each.", 8, 3, 0.95, 1.0),
    ("3D: twelve", "More than fit comfortably. Often one or two are dropped, and the rest get about a quarter of a dimension each.", 12, 3, 0.97, 1.0),
];

/// The learning rate and batch size every preset trains with (and the scene opens
/// with): what the tests of the presets assume.
const LR: f32 = 0.02;
const BATCH: usize = 512;

/// A feature shorter than this is treated as not represented at all.
const PRESENT: f32 = 0.5;

/// How slowly the number of a feature that is hardly represented turns about its
/// arrow (seconds: the time to turn all but 1/e of the way), for an arrow of no length;
/// none at all for one of length `PRESENT` or more (see `Superposition::lean`).
const LEAN: f32 = 0.3;

/// Where the panels go on a canvas of size `view` (logical pixels), and whether the
/// canvas is narrow: then they lie in the lower part, below the picture, and
/// otherwise in a column on the right. Each is its top-left corner and its size.
struct Panels {
    narrow: bool,
    /// Where the part kept for the panels begins on a narrow canvas.
    below: f32,
    heat: (Vec2, f32),
    share: (Vec2, Vec2),
    alone: (Vec2, Vec2),
    loss: (Vec2, Vec2),
}

impl Panels {
    /// `inset`: how much of the top of the canvas the application's own buttons cover
    /// (the column on the right starts below them).
    fn of(view: Vec2, inset: f32) -> Panels {
        // (stacked under the picture only on a canvas tall enough for all of that:
        // a phone held sideways gets the column, scaled down to its height)
        if view.x < 620.0 && view.y >= 560.0 {
            let below = (view.y * 0.4).max(200.0);
            let side = ((view.x - 96.0) * 0.5).min(view.y * 0.19).clamp(70.0, 200.0);
            let alone_y = below + 30.0 + side + 66.0;
            let alone_h = (view.y * 0.1).clamp(56.0, 90.0);
            let loss_y = alone_y + alone_h + 44.0;
            Panels {
                narrow: true,
                below,
                heat: (Vec2::new(30.0, below + 30.0), side),
                share: (Vec2::new(30.0 + side + 44.0, below + 30.0), Vec2::new((view.x - side - 104.0).max(60.0), side)),
                alone: (Vec2::new(30.0, alone_y), Vec2::new(view.x - 60.0, alone_h)),
                loss: (Vec2::new(30.0, loss_y), Vec2::new(view.x - 60.0, (view.y - 20.0 - loss_y).max(24.0))),
            }
        } else {
            // the three panels of the column at the sizes they would like, and
            // then, on a short canvas, all scaled to the height there is (what
            // lies between and around them, their titles and labels, is fixed)
            let wanted = [(view.y * 0.24).clamp(110.0, 200.0), (view.y * 0.15).clamp(70.0, 130.0), (view.y * 0.17).clamp(80.0, 140.0)];
            let top = 44.0f32.max(inset + 34.0);
            let fixed = top + 62.0 + 76.0 + 12.0;
            let fit = ((view.y - fixed) / wanted.iter().sum::<f32>()).clamp(0.05, 1.0);
            let [side, share_h, alone_h] = wanted.map(|h| h * fit);
            let x = view.x - side - 44.0;
            let share_y = top + side + 62.0;
            // (the chart of the loss keeps clear of the column and of the canvas's top)
            let loss = Vec2::new((view.x * 0.26).clamp(190.0, 320.0).min((x - 60.0).max(40.0)), (view.y * 0.16).clamp(70.0, 120.0).min((view.y - 60.0).max(20.0)));
            Panels {
                narrow: false,
                below: view.y,
                heat: (Vec2::new(x, top), side),
                share: (Vec2::new(x, share_y), Vec2::new(side, share_h)),
                alone: (Vec2::new(x, share_y + share_h + 76.0), Vec2::new(side, alone_h)),
                loss: (Vec2::new(26.0, view.y - loss.y - 26.0), loss),
            }
        }
    }
}

struct Superposition {
    /// The model as it is trained: the numbers, the chart and training go by it.
    toy: Toy,
    /// Its weights and biases before the last step, and the model as it is drawn,
    /// between the two (see `Superposition::follow`).
    before: (Vec<f32>, Vec<f32>),
    drawn: Toy,
    rng: Rng,
    n: usize,
    m: usize,
    sparsity: f32,
    decay: f32,
    lr: f32,
    batch: usize,
    /// Steps per second asked for.
    rate: usize,
    running: bool,
    steps: u64,
    pace: crate::clock::Pace,
    /// What `pace` owes, kept beside it in the same steps: its part of a step is how
    /// far the picture is from the model before the last step to the model now.
    owed: f32,
    loss: f32,
    history: History,
    /// The feature whose lone effect is shown in the bars.
    alone: usize,
    /// The way each feature's number is set off from the tip of its arrow (see `LEAN`).
    lean: Vec<Vec3>,
    /// The size of the canvas last drawn on (it decides where the panels go, and
    /// with that where the picture has to be).
    canvas: Vec2,
    /// How much of its top the application's buttons cover, likewise.
    inset: f32,
    new_view: bool,
}

impl Superposition {
    fn new() -> Self {
        let (_, _, n, m, sparsity, decay) = PRESETS[2];
        let mut rng = Rng::new(21);
        let toy = Toy::new(n, m, &mut rng);
        Superposition {
            before: (toy.w.clone(), toy.b.clone()),
            drawn: copy(&toy),
            toy,
            rng,
            n,
            m,
            sparsity,
            decay,
            lr: LR,
            batch: BATCH,
            rate: 240,
            running: true,
            steps: 0,
            pace: crate::clock::Pace::default(),
            owed: 0.0,
            loss: 0.0,
            history: History::new(600),
            alone: 0,
            lean: Vec::new(),
            canvas: Vec2::new(980.0, 940.0),
            inset: 0.0,
            new_view: false,
        }
    }

    fn restart(&mut self) {
        self.new_view |= self.m != self.toy.m;
        self.toy = Toy::new(self.n, self.m, &mut self.rng);
        // (drawn where the run starts: nothing to go between yet)
        self.drawn = copy(&self.toy);
        self.keep_before();
        self.lean.clear();
        self.steps = 0;
        self.history.clear();
        // where the new weights stand before their first step (measured on a batch
        // of its own, so that the batches training will see stay as they were)
        self.loss = self.toy.loss(512, self.sparsity, &importance(self.n, self.decay), &mut Rng::new(7));
        self.alone = self.alone.min(self.n - 1);
    }

    /// Remember the model as it is, before a step.
    fn keep_before(&mut self) {
        self.before.0.clone_from(&self.toy.w);
        self.before.1.clone_from(&self.toy.b);
    }

    /// Bring the model as drawn to the part of a step the pace owes, between the model
    /// before the last step and the model now. The steps come whole, and not the same
    /// number in every frame (240 a second on a display of 144 is one in some frames
    /// and two in others): drawn as they come, the arrows would move in fits and
    /// starts, on top of the shaking that training on random batches gives them, which
    /// is what they should show. So they are drawn up to a step behind the run, at the
    /// point the steps have reached.
    fn follow(&mut self) {
        let t = self.owed.clamp(0.0, 1.0);
        let ((w0, b0), toy, drawn) = (&self.before, &self.toy, &mut self.drawn);
        if (drawn.n, drawn.m) != (toy.n, toy.m) || w0.len() != toy.w.len() || b0.len() != toy.b.len() {
            *drawn = copy(toy);
            return;
        }
        for (to, (a, b)) in drawn.w.iter_mut().chain(&mut drawn.b).zip(w0.iter().chain(b0).zip(toy.w.iter().chain(&toy.b))) {
            *to = a + (b - a) * t;
        }
    }

    /// Turn the way each feature's number is set off from its arrow `dt` seconds of
    /// the scene's time nearer to the arrow's own direction. A long arrow's number
    /// sits straight out beyond its tip. A feature that has been dropped keeps a stub
    /// that shakes about the centre with every batch, pointing any way at all from one
    /// frame to the next: its number, set off along it, would hop around the centre,
    /// so it turns only slowly towards the stub's direction instead. (The stub itself
    /// is drawn as it is.)
    fn lean(&mut self, tips: &[Vec3], dt: f32) {
        if self.lean.len() != tips.len() {
            self.lean = tips.iter().map(|t| t.normalize_or_zero()).collect();
        }
        for (lean, tip) in self.lean.iter_mut().zip(tips) {
            let Some(along) = tip.try_normalize() else { continue };
            let slowness = LEAN * (1.0 - tip.length() / PRESENT).max(0.0);
            let k = if slowness <= 0.0 || *lean == Vec3::ZERO { 1.0 } else { 1.0 - (-dt.max(0.0) / slowness).exp() };
            // (never as much as half the way in a frame, so this never passes near zero)
            *lean = (*lean + (along - *lean) * k).try_normalize().unwrap_or(along);
        }
    }

    fn the_view(&self) -> View {
        let panels = Panels::of(self.canvas, self.inset);
        if panels.narrow {
            // the picture has the part of the canvas above the panels (and below the
            // application's buttons) to itself: put the origin in the middle of that part
            let (top, region) = (52.0, (panels.below - 52.0).max(120.0));
            let per_unit = (self.canvas.x / 2.75).min(region / 2.75);
            let lift = (self.canvas.y * 0.5 - (top + region * 0.5)) / per_unit;
            if self.m == 2 {
                View::Fit { center: Vec2::new(0.0, -lift), half: Vec2::new(0.5, self.canvas.y * 0.5 / per_unit) }
            } else {
                // (0.32 is the tangent of half the camera's field of view)
                let dist = self.canvas.y * 0.5 / per_unit / 0.32;
                View::Orbit { target: Vec3::new(0.0, 0.0, -lift / 0.93), yaw: 0.7, pitch: 0.38, dist }
            }
        } else if self.m == 2 {
            // the origin sits left of centre, clear of the panels on the right
            View::Fit { center: Vec2::new(0.55, 0.0), half: Vec2::new(2.3, 1.55) }
        } else {
            View::Orbit { target: Vec3::ZERO, yaw: 0.7, pitch: 0.38, dist: 6.2 }
        }
    }

    fn color(&self, i: usize) -> Color {
        Cmap::Plasma.color(0.97 - 0.62 * i as f32 / (self.n.max(2) - 1) as f32)
    }

    /// The panels: W'W, each feature's share of a dimension, what one feature alone
    /// produces, and the loss.
    fn panels(&self, f: &mut Frame, at: &Panels) {
        let (n, view) = (self.n, f.view_size());
        if at.narrow {
            hud_panel(f, Vec2::new(0.0, at.below), view);
        }
        let backing = |f: &mut Frame, min: Vec2, max: Vec2| {
            if !at.narrow {
                hud_panel(f, min, max);
            }
        };
        let (o, side) = at.heat;
        let cell = side / n as f32;
        backing(f, o - Vec2::new(22.0, 30.0), o + Vec2::new(side + 34.0, side + 30.0));
        let heat = Cmap::CoolWarm.table();
        let shade = |v: f32| heat[table_index(0.5 + 0.5 * v.clamp(-1.0, 1.0))];
        for i in 0..n {
            for j in 0..n {
                let p = o + Vec2::new(j as f32, i as f32) * cell;
                hud_rect(f, p, p + Vec2::splat(cell * 0.94), shade(self.drawn.overlap(i, j)));
            }
            // which feature this row and column are, and its bias
            let c = self.color(i);
            hud_rect(f, o + Vec2::new(-9.0, i as f32 * cell), o + Vec2::new(-4.0, (i as f32 + 0.94) * cell), c);
            hud_rect(f, o + Vec2::new(i as f32 * cell, -9.0), o + Vec2::new((i as f32 + 0.94) * cell, -4.0), c);
            let b = o + Vec2::new(side + 8.0, i as f32 * cell);
            hud_rect(f, b, b + Vec2::new(12.0, cell * 0.94), shade(self.drawn.b[i]));
        }
        // the column of the feature shown alone is marked
        let x = o.x + self.alone as f32 * cell;
        for (a, b) in [(Vec2::new(x - 1.0, o.y - 1.0), Vec2::new(x - 1.0, o.y + side)), (Vec2::new(x + cell * 0.94 + 1.0, o.y - 1.0), Vec2::new(x + cell * 0.94 + 1.0, o.y + side))] {
            hud_line(f, a, b, 1.4, pal::WHITE);
        }
        let title = if side > 165.0 {
            r"$W^\top W$: how the features overlap"
        } else if side > 100.0 {
            r"$W^\top W$: the overlaps"
        } else {
            r"$W^\top W$"
        };
        f.text_px(o + Vec2::new(-9.0, -14.0), title, 11.5, pal::TEXT, Align2::LEFT_BOTTOM);
        f.text_px(o + Vec2::new(side + 14.0, -12.0), "bias", 10.5, pal::DIM, Align2::CENTER_BOTTOM);
        // what the colours stand for
        let scale_y = o.y + side + 8.0;
        for k in 0..24 {
            let x = |t: f32| o.x + side * 0.62 * t;
            hud_rect(f, Vec2::new(x(k as f32 / 24.0), scale_y), Vec2::new(x((k + 1) as f32 / 24.0) + 0.5, scale_y + 6.0), heat[table_index((k as f32 + 0.5) / 24.0)]);
        }
        f.text_px(Vec2::new(o.x, scale_y + 9.0), "−1", 10.0, pal::DIM, Align2::LEFT_TOP);
        f.text_px(Vec2::new(o.x + side * 0.31, scale_y + 9.0), "0", 10.0, pal::DIM, Align2::CENTER_TOP);
        f.text_px(Vec2::new(o.x + side * 0.62, scale_y + 9.0), "+1", 10.0, pal::DIM, Align2::RIGHT_TOP);

        // each feature's share of a dimension
        let (so, size) = at.share;
        let bar = size.x / n as f32;
        backing(f, so - Vec2::new(22.0, 28.0), so + Vec2::new(size.x + 34.0, size.y + 12.0));
        for (value, label) in [(1.0, "$1$"), (0.5, "$1/2$"), (1.0 / 3.0, "$1/3$"), (0.25, "$1/4$")] {
            let y = so.y + size.y * (1.0 - value);
            hud_line(f, Vec2::new(so.x, y), Vec2::new(so.x + size.x, y), 1.0, pal::FAINT.a(0.6));
            // (a third and a quarter are too close to name both on a low panel)
            if value != 1.0 / 3.0 || size.y > 160.0 {
                f.text_px(Vec2::new(so.x + size.x + 5.0, y), label, 10.5, pal::DIM, Align2::LEFT_CENTER);
            }
        }
        for i in 0..n {
            let share = self.drawn.dimensionality(i).clamp(0.0, 1.0);
            let x = so.x + i as f32 * bar;
            hud_rect(f, Vec2::new(x, so.y + size.y * (1.0 - share)), Vec2::new(x + bar * 0.94, so.y + size.y), self.color(i));
        }
        hud_line(f, Vec2::new(so.x, so.y + size.y), Vec2::new(so.x + size.x, so.y + size.y), 1.0, pal::DIM.a(0.8));
        let title = if size.x > 185.0 { "share of a dimension each feature has" } else { "share of a dimension" };
        f.text_px(Vec2::new(so.x - if at.narrow { 0.0 } else { 9.0 }, so.y - 12.0), title, 11.5, pal::TEXT, Align2::LEFT_BOTTOM);

        // one feature switched on alone: what arrives at every output, before and
        // after the bias and the ReLU
        let (ao, size) = at.alone;
        let bar = size.x / n as f32;
        backing(f, ao - Vec2::new(22.0, 44.0), ao + Vec2::new(size.x + 34.0, size.y + 14.0));
        // the bars run from -0.9 to 1.3, so that leaks either way stay in the panel
        let (low, high) = (-0.9, 1.3);
        let y_of = |v: f32| ao.y + size.y * (high - v.clamp(low, high)) / (high - low);
        for (value, label) in [(1.0, "1"), (0.0, "0")] {
            hud_line(f, Vec2::new(ao.x, y_of(value)), Vec2::new(ao.x + size.x, y_of(value)), 1.0, if value == 0.0 { pal::DIM.a(0.8) } else { pal::FAINT.a(0.6) });
            f.text_px(Vec2::new(ao.x + size.x + 5.0, y_of(value)), label, 10.5, pal::DIM, Align2::LEFT_CENTER);
        }
        for i in 0..n {
            let (before, after) = self.drawn.alone(self.alone, i);
            let x = ao.x + i as f32 * bar;
            let c = self.color(i);
            let (top, bottom) = (y_of(before.max(0.0)), y_of(before.min(0.0)));
            hud_rect(f, Vec2::new(x, top), Vec2::new(x + bar * 0.42, bottom.max(top + 1.0)), c.mix(pal::WHITE, 0.25).a(0.38));
            hud_rect(f, Vec2::new(x + bar * 0.5, y_of(after)), Vec2::new(x + bar * 0.92, y_of(0.0).max(y_of(after) + 1.0)), c);
        }
        // (as much of the two lines as the panel is wide)
        let room = size.x + 20.0;
        let long = format!("feature {} alone, switched on at 1:", self.alone + 1);
        let name = if text_width(&long, 11.5) <= room { long } else { format!("feature {} alone:", self.alone + 1) };
        let key = ["pale: what arrives   solid: what comes out", "pale: in   solid: out"].into_iter().find(|k| text_width(k, 10.5) <= room).unwrap_or("");
        f.text_px(Vec2::new(ao.x - if at.narrow { 0.0 } else { 9.0 }, ao.y - 27.0), name, 11.5, pal::TEXT, Align2::LEFT_BOTTOM);
        f.text_px(Vec2::new(ao.x - if at.narrow { 0.0 } else { 9.0 }, ao.y - 12.0), key, 10.5, pal::DIM, Align2::LEFT_BOTTOM);

        let (lo, size) = at.loss;
        hud_chart(f, lo, size, &[(self.history.points(), pal::CYAN)], true, None);
        f.text_px(lo + Vec2::new(0.0, -12.0), format!("loss  {}    (log scale, against training steps)", loss_number(self.loss)), 11.5, pal::TEXT, Align2::LEFT_BOTTOM);
    }
}

impl Scene for Superposition {
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
        self.running || self.m == 3
    }

    /// In space the arrangement is a solid: turn it slowly so its shape can be read.
    fn auto_rotate(&self) -> f32 {
        if self.m == 3 {
            0.14
        } else {
            0.0
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A network often has more things to keep track of than it has neurons. This small one must pass several features through fewer dimensions and rebuild them on the other side. Each arrow is the direction it has given one feature. Watch how it packs them in, and what that costs.");
        ui::term(ui, "Feature:", "one thing the input can contain, here a number that is zero in a share of the inputs (the sparsity) and otherwise random between 0 and 1.");
        ui::term(ui, "Superposition:", "storing more features than there are dimensions, by letting their directions overlap.");
        ui::term(ui, "Polysemantic neuron:", "a neuron that answers to more than one feature. The neurons here are the axes of the picture: once more arrows are packed in than there are axes, each axis has more than one arrow leaning along it.");
        ui.horizontal(|ui| {
            if ui.button(if self.running { "pause" } else { "train" }).clicked() {
                self.running = !self.running;
            }
            if ui.button("new weights").clicked() {
                self.restart();
            }
        });

        ui::section(ui, "Things to try");
        for (k, preset) in PRESETS.iter().enumerate() {
            if ui::experiment(ui, preset.0, preset.1) {
                (_, _, self.n, self.m, self.sparsity, self.decay) = PRESETS[k];
                // (and the training settings: a preset runs the same whatever was set before)
                (self.lr, self.batch) = (LR, BATCH);
                self.restart();
                self.running = true;
            }
        }

        ui::section(ui, "What overlap costs");
        let mut shown = self.alone + 1;
        if ui::slider_int(ui, "feature switched on alone", &mut shown, 1..=self.n) {
            self.alone = shown - 1;
        }
        ui::note(ui, "Or click an arrow. The lowest bars show what happens when that one feature is on, at 1, and all the others are off. Pale: what arrives at each output through the shared dimensions. Its own output gets about 1, or more where features overlap (the bias takes some of it away again); the others get a leak, wherever their arrows are not at right angles to it. Solid: what comes out after the bias and the ReLU (which turns anything below zero into zero): about 1 at its own output.");
        ui::note(ui, "A leak below zero is removed by the ReLU for free. A leak above zero is only partly cancelled by a negative bias, which lowers the feature's own output too: some of the leak still comes out. Either way the trick only works while features are rarely on together: two at once, and their leaks add up. That is why sparsity decides how many can be packed in.");

        ui::section(ui, "The task");
        let mut dims = self.m - 2;
        let mut rebuilt = ui::choice(ui, &["2 dimensions", "3 dimensions"], &mut dims);
        self.m = dims + 2;
        rebuilt |= ui::slider_int(ui, "features", &mut self.n, 2..=20);
        if rebuilt {
            self.restart();
        }
        ui::slider(ui, "sparsity", &mut self.sparsity, 0.0..=0.99);
        ui::note(ui, "The chance that a feature is absent (zero) in any one input. When present it is a random number between 0 and 1.");
        ui::slider(ui, "importance falls off by", &mut self.decay, 0.5..=1.0);
        ui::note(ui, "Each feature counts this many times as much as the one before it in the loss. At 1 they all count the same. The brightest arrow is the most important.");
        ui::section(ui, "Training");
        ui::slider_log(ui, "learning rate", &mut self.lr, 0.001..=0.1);
        ui::slider_int_log(ui, "batch size", &mut self.batch, 32..=2048);
        ui::slider_int_log(ui, "steps per second", &mut self.rate, 10..=5000);

        ui::section(ui, "Reading the picture");
        let heat = Cmap::CoolWarm.table();
        ui::key(
            ui,
            &[
                (self.color(0), "arrows: one per feature, numbered; the brightest matters most. A faint stub is a feature the network has dropped"),
                (pal::DIM, "circle: length 1. Grey lines join the features that are kept"),
                (heat[table_index(0.97)], r"red cell of $W^\top W$: two features point the same way ($+1$ on the diagonal: a feature with itself)"),
                (heat[table_index(0.03)], "blue cell: they point opposite ways"),
                (heat[table_index(0.5)], "pale cell: at right angles, no overlap"),
            ],
        );
        ui::section(ui, "Numbers");
        let kept = (0..self.n).filter(|&i| self.toy.direction(i).length() > PRESENT).count();
        ui::readout(ui, "training steps", self.steps.to_string());
        ui::readout(ui, "loss", loss_number(self.loss));
        ui::readout(ui, "features represented", format!("{kept} of {}", self.n));
        ui::readout(ui, "dimensions", self.m.to_string());
        ui::note(ui, "A share of 1 means a feature has a dimension to itself; $1/2$, that it shares one with its opposite; $2/5$ is a pentagon in the plane (five features on two dimensions). The loss is the squared error of the rebuilt features, weighted by importance.");
        ui::see_also(ui, &[("sae", "Sparse autoencoders: getting the features back out"), ("packing", "how far apart $n$ directions can be kept at all"), ("transformer", "a transformer, opened up"), ("highdim", "why many dimensions have room for so many directions")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // train, within a time allowance
        if self.running && f.dt > 0.0 {
            let importance = importance(self.n, self.decay);
            let rate = self.rate as f32;
            let mut turn = self.pace.begin(rate, f.dt, f.real_dt, 0.004);
            // (what the pace owes, worked out as it works it out, and paid off step by step)
            self.owed = (self.owed + rate * f.dt).min((rate * 0.1).max(2.0));
            while turn.next() {
                self.owed -= 1.0;
                self.keep_before();
                let loss = self.toy.step(self.batch, self.sparsity, &importance, self.lr, &mut self.rng);
                self.loss = if self.steps == 0 { loss } else { self.loss + (loss - self.loss) * 0.05 };
                self.steps += 1;
                if self.steps.is_multiple_of(8) {
                    self.history.push(self.loss);
                }
            }
            self.pace.end(turn);
            self.follow();
        }

        // the panels' places follow the canvas, and the picture's place follows theirs
        let view = f.view_size();
        let panels = Panels::of(view, f.top_inset);
        if panels.narrow != Panels::of(self.canvas, self.inset).narrow || (panels.narrow && (view - self.canvas).abs().max_element() > 24.0) {
            self.canvas = view;
            self.new_view = true;
        }
        self.inset = f.top_inset;

        let (n, flat) = (self.n, self.m == 2);
        let panels_from = if panels.narrow { view.x } else { panels.heat.0.x - 26.0 };
        // (the arrows, the shape and the panels are the model as drawn; the numbers in
        // the panels at the side are the run's own)
        let tips: Vec<Vec3> = (0..n).map(|i| self.drawn.direction(i)).collect();
        let present: Vec<usize> = (0..n).filter(|&i| tips[i].length() > PRESENT).collect();
        self.lean(&tips, f.dt);

        // a click on an arrow's tip picks the feature shown alone in the bars
        if f.clicked_canvas() {
            if let Some(pointer) = f.ptr.pos {
                let nearest = (0..n).filter_map(|i| f.cam.project(tips[i]).map(|p| (i, (p - pointer).length() / f.px))).min_by(|a, b| a.1.total_cmp(&b.1));
                if let Some((i, _)) = nearest.filter(|(_, d)| *d < 28.0) {
                    self.alone = i;
                }
            }
        }

        // the unit circle (or sphere) the directions are measured against
        f.circle(Vec3::ZERO, Vec3::Z, 1.0, 1.1, pal::FAINT);
        if flat {
            f.line((-1.25, 0.0), (1.25, 0.0), 1.0, pal::FAINT.a(0.5));
            f.line((0.0, -1.25), (0.0, 1.25), 1.0, pal::FAINT.a(0.5));
            f.text((1.25, -0.05), "hidden dimension 1", 10.5, pal::DIM, Align2::RIGHT_TOP);
            f.text((0.0, 1.29), "hidden dimension 2", 10.5, pal::DIM, Align2::CENTER_BOTTOM);
            // the shape the represented features make, taken in order of angle
            let mut ring = present.clone();
            ring.sort_by(|&a, &b| tips[a].y.atan2(tips[a].x).total_cmp(&tips[b].y.atan2(tips[b].x)));
            // (begun at the first feature, wherever it points: begun at an angle, the
            // same shape would be drawn from another corner whenever an arrow passed
            // that angle)
            if let Some(first) = ring.iter().position(|&i| Some(&i) == present.first()) {
                ring.rotate_left(first);
            }
            if ring.len() >= 3 {
                let pts: Vec<Vec3> = ring.iter().map(|&i| tips[i]).collect();
                f.dl.lines.closed(&pts, 1.3, pal::DIM.a(0.55));
            }
        } else {
            f.circle(Vec3::ZERO, Vec3::X, 1.0, 1.1, pal::FAINT);
            f.circle(Vec3::ZERO, Vec3::Y, 1.0, 1.1, pal::FAINT);
            // the solid: each feature joined to the ones about as near as its nearest
            let angle = |a: usize, b: usize| tips[a].normalize_or_zero().dot(tips[b].normalize_or_zero()).clamp(-1.0, 1.0).acos();
            for (k, &a) in present.iter().enumerate() {
                let nearest = present.iter().filter(|&&b| b != a).map(|&b| angle(a, b)).fold(f32::MAX, f32::min);
                for &b in &present[k + 1..] {
                    let nearest_b = present.iter().filter(|&&c| c != b).map(|&c| angle(b, c)).fold(f32::MAX, f32::min);
                    if angle(a, b) < 1.4 * nearest.max(nearest_b) && angle(a, b) < 2.6 {
                        f.line(tips[a], tips[b], 1.3, pal::DIM.a(0.6));
                    }
                }
            }
        }
        for i in 0..n {
            let (tip, c) = (tips[i], self.color(i));
            let faint = if tip.length() > PRESENT { 1.0 } else { 0.45 };
            f.arrow(Vec3::ZERO, tip, if i == self.alone { 4.4 } else { 3.2 }, c.a(faint));
            if i == self.alone {
                f.glow(tip, 22.0, c);
            }
            if flat {
                f.dot(tip, 9.0, c);
            } else {
                f.sphere(tip, 0.085, c);
            }
            // (a label that would land on the panels at the right is left out)
            let at = tip + self.lean[i] * 0.16;
            let clear = f.cam.project(at).is_some_and(|p| p.x / f.px < panels_from);
            if clear {
                f.text(at, (i + 1).to_string(), 12.5, pal::TEXT.a(faint), Align2::CENTER_CENTER);
            }
        }
        self.panels(f, &panels);
    }
}

/// The same model (its weights and biases; the optimiser's memory is not wanted for
/// drawing).
fn copy(toy: &Toy) -> Toy {
    let mut drawn = Toy::new(toy.n, toy.m, &mut Rng::new(0));
    drawn.w.clone_from(&toy.w);
    drawn.b.clone_from(&toy.b);
    drawn
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Preset `k` of `PRESETS` after 5000 steps, trained as the scene trains it.
    fn trained(k: usize, seed: u64) -> Toy {
        let (_, _, n, m, sparsity, decay) = PRESETS[k];
        let mut rng = Rng::new(seed);
        let mut toy = Toy::new(n, m, &mut rng);
        let imp = importance(n, decay);
        for _ in 0..5000 {
            toy.step(BATCH, sparsity, &imp, LR, &mut rng);
        }
        toy
    }

    fn cosine(toy: &Toy, i: usize, j: usize) -> f32 {
        toy.direction(i).normalize().dot(toy.direction(j).normalize())
    }

    /// The scene's text, held to account: what each arrangement really is.
    #[test]
    fn dense_inputs_keep_the_two_most_important_features_at_right_angles() {
        for seed in [40, 41] {
            let toy = trained(0, seed);
            let lengths: Vec<f32> = (0..5).map(|i| toy.direction(i).length()).collect();
            assert!(lengths[0] > 0.9 && lengths[1] > 0.9 && lengths[2..].iter().all(|&l| l < 0.2), "{lengths:?}");
            assert!(cosine(&toy, 0, 1).abs() < 0.1);
            assert!((toy.dimensionality(0) - 1.0).abs() < 0.05);
        }
    }

    #[test]
    fn sparse_inputs_pack_five_features_into_a_pentagon() {
        for seed in [40, 41] {
            let toy = trained(2, seed);
            let mut angles: Vec<f32> = (0..5).map(|i| toy.direction(i).y.atan2(toy.direction(i).x)).collect();
            angles.sort_by(f32::total_cmp);
            for k in 0..5 {
                assert!(toy.direction(k).length() > 0.9);
                assert!((toy.dimensionality(k) - 0.4).abs() < 0.04, "share {}", toy.dimensionality(k));
                let gap = (angles[(k + 1) % 5] - angles[k]).rem_euclid(std::f32::consts::TAU).to_degrees();
                assert!((gap - 72.0).abs() < 8.0, "gap of {gap} degrees");
            }
        }
    }

    #[test]
    fn six_features_in_space_make_an_octahedron() {
        for seed in [40, 41] {
            let toy = trained(3, seed);
            for i in 0..6 {
                assert!((toy.dimensionality(i) - 0.5).abs() < 0.04);
                // one opposite, four at right angles
                let cosines: Vec<f32> = (0..6).filter(|&j| j != i).map(|j| cosine(&toy, i, j)).collect();
                assert_eq!(cosines.iter().filter(|c| **c < -0.9).count(), 1, "{cosines:?}");
                assert_eq!(cosines.iter().filter(|c| c.abs() < 0.15).count(), 4, "{cosines:?}");
            }
        }
    }

    /// The panel's account of what overlap costs: a lone feature comes out near 1; what
    /// arrives at its own output is about 1 where nothing overlaps (opposite pairs) and
    /// more where features overlap (the pentagon: 1.2 to 1.3), the bias taking some of
    /// it away again; its leak into the others is removed by the ReLU alone where it is
    /// negative, and only partly cancelled by a negative bias where it is positive:
    /// some of it still comes out.
    #[test]
    fn a_lone_feature_leaks_and_the_leak_is_silenced() {
        let pentagon = trained(2, 40);
        for k in 0..5 {
            let arrives: Vec<f32> = (0..5).filter(|&i| i != k).map(|i| pentagon.alone(k, i).0).collect();
            // two neighbours get a leak above zero, the two far ones a leak below it
            assert_eq!(arrives.iter().filter(|a| **a > 0.2).count(), 2, "{arrives:?}");
            assert_eq!(arrives.iter().filter(|a| **a < -0.6).count(), 2, "{arrives:?}");
            let (own_in, own_out) = pentagon.alone(k, k);
            assert!(own_in > 1.1 && (own_out - 1.0).abs() < 0.12, "own output: {own_in} arrives, {own_out} comes out");
            assert!((0..5).filter(|&i| i != k).all(|i| pentagon.alone(k, i).1 < 0.2), "what comes out elsewhere");
            // the leaks above zero: cut by the bias, not to nothing
            for i in (0..5).filter(|&i| i != k && pentagon.alone(k, i).0 > 0.2) {
                let (leak_in, leak_out) = pentagon.alone(k, i);
                assert!(leak_out > 0.05 && leak_out < leak_in - 0.1, "leak from {k} into {i}: {leak_in} arrives, {leak_out} comes out");
            }
            assert!(pentagon.b[k] < -0.1 && pentagon.b[k] > -0.4, "bias {}", pentagon.b[k]);
        }
        let pairs = trained(3, 40);
        for k in 0..6 {
            // nothing arrives above zero except at the feature itself, and no bias is needed
            assert!((0..6).filter(|&i| i != k).all(|i| pairs.alone(k, i).0 < 0.12 && pairs.alone(k, i).1 < 0.12));
            assert!((pairs.alone(k, k).0 - 1.0).abs() < 0.08 && (pairs.alone(k, k).1 - 1.0).abs() < 0.08 && pairs.b[k].abs() < 0.08, "bias {}", pairs.b[k]);
        }
    }

    /// The captions of the remaining arrangements.
    #[test]
    fn eight_features_share_three_dimensions_equally() {
        let toy = trained(4, 41);
        for i in 0..8 {
            assert!((toy.dimensionality(i) - 0.375).abs() < 0.05, "share {}", toy.dimensionality(i));
        }
    }

    /// The panels keep to the canvas and clear of each other, whatever its shape.
    #[test]
    fn the_panels_fit_every_canvas() {
        // (among them a phone held sideways and a small window, where the panels
        // used to run off the bottom)
        for (x, y) in [(390.0f32, 800.0f32), (360.0, 640.0), (600.0, 900.0), (590.0, 700.0), (640.0, 480.0), (980.0, 940.0), (1700.0, 800.0), (844.0, 390.0), (600.0, 400.0), (700.0, 330.0)] {
            // (under 900 wide the application's buttons lie over the top of the canvas)
            let inset = if x < 900.0 { 46.0 } else { 0.0 };
            let p = Panels::of(Vec2::new(x, y), inset);
            assert!(inset == 0.0 || p.narrow || p.heat.0.y - 28.0 >= inset, "{x}x{y}: the column starts under the buttons");
            // each with the room its title and labels take: (left, top, right, bottom)
            let rects = [
                (p.heat.0.x - 10.0, p.heat.0.y - 28.0, p.heat.0.x + p.heat.1 + 22.0, p.heat.0.y + p.heat.1 + 22.0),
                (p.share.0.x, p.share.0.y - 26.0, p.share.0.x + p.share.1.x + 22.0, p.share.0.y + p.share.1.y + 4.0),
                (p.alone.0.x, p.alone.0.y - 40.0, p.alone.0.x + p.alone.1.x + 14.0, p.alone.0.y + p.alone.1.y + 4.0),
                (p.loss.0.x, p.loss.0.y - 26.0, p.loss.0.x + p.loss.1.x, p.loss.0.y + p.loss.1.y),
            ];
            for (i, a) in rects.iter().enumerate() {
                assert!(a.0 >= 0.0 && a.1 >= 0.0 && a.2 <= x && a.3 <= y, "{x}x{y}: panel {i} leaves the canvas: {a:?}");
                for (j, b) in rects.iter().enumerate().skip(i + 1) {
                    assert!(a.2 <= b.0 || b.2 <= a.0 || a.3 <= b.1 || b.3 <= a.1, "{x}x{y}: panels {i} and {j} overlap: {a:?} {b:?}");
                }
            }
        }
    }

    /// What the model settles into for a range of settings (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for &(n, m, sparsity, decay) in &[
            (5usize, 2usize, 0.0f32, 0.9f32),
            (5, 2, 0.0, 0.7),
            (5, 2, 0.5, 0.9),
            (5, 2, 0.8, 0.9),
            (5, 2, 0.9, 0.9),
            (5, 2, 0.9, 1.0),
            (5, 2, 0.95, 1.0),
            (5, 2, 0.97, 0.9),
            (4, 3, 0.9, 1.0),
            (6, 3, 0.9, 1.0),
            (6, 3, 0.95, 1.0),
            (8, 3, 0.95, 1.0),
            (12, 3, 0.97, 1.0),
            (20, 3, 0.98, 1.0),
        ] {
            for seed in 0..3 {
                let mut rng = Rng::new(40 + seed);
                let mut toy = Toy::new(n, m, &mut rng);
                let imp = importance(n, decay);
                let mut loss = 0.0;
                for _ in 0..6000 {
                    loss = toy.step(BATCH, sparsity, &imp, LR, &mut rng);
                }
                let norms: Vec<String> = (0..n).map(|i| format!("{:.2}", toy.direction(i).length())).collect();
                let dims: Vec<String> = (0..n).map(|i| format!("{:.2}", toy.dimensionality(i))).collect();
                println!("n={n} m={m} S={sparsity} decay={decay} seed={seed}: loss {loss:.4}  lengths [{}]  shares [{}]  bias {:.2}", norms.join(" "), dims.join(" "), toy.b.iter().sum::<f32>() / n as f32);
            }
        }
    }
}

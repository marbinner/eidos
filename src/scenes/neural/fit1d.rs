//! Building a function from neurons.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "fit1d",
    title: "Building a function from neurons",
    group: "Neural networks",
    blurb: "Universal approximation, one hinge at a time.",
    make: |g| Box::new(Fit1d::new(g)),
};

const FIT_N: usize = 160;

const TARGETS: [(&str, &str); 5] = [("wave", "sin(2x) + 0.3 x"), ("bump", "1.6 exp(-x^2) - 0.4"), ("step", "tanh(6x)"), ("zig-zag", "abs(sin(1.6 x)) * 2 - 1"), ("chirp", "sin(x^2)")];

/// Experiments: (button, what to look for, target as an index into `TARGETS`, hidden
/// neurons, activation as an index into `Act::ALL`).
const TRIALS: [(&str, &str, usize, usize, usize); 6] = [
    ("three hinges", "Three ReLU neurons give a line with at most three bends: too few pieces for this wave.", 0, 3, 1),
    ("sixty-four hinges", "The same wave with sixty-four neurons: enough short straight pieces to follow it closely.", 0, 64, 1),
    ("smooth steps instead", r"With $\tanh$ each neuron adds a smooth step rather than a hinge, and the sum is a smooth curve.", 1, 8, 0),
    ("a sudden step", r"Two neurons. With $\tanh$ that is plenty: one neuron with a large weight is a steep step. Switch the activation to ReLU: two hinges can only make a ramp with corners, and now and then they get stuck far from it.", 2, 2, 0),
    ("a chirp", "The wiggles get faster towards the edges. The slow ones in the middle are learned first, the fast ones last.", 4, 32, 0),
    ("parabolas only", r"Sixty-four neurons again, but each piece is now the parabola $(x^2 - 1)/\sqrt{2}$. Parabolas add up to a parabola, however many there are: within a few hundred steps the network has the best one for this wave, which is a straight line, and it gets no closer. To follow any curve at all, a network needs an activation that is not a polynomial.", 0, 64, 7),
];

pub struct Fit1d {
    target: ExprField,
    width: usize,
    act: usize,
    lr: f32,
    /// Training steps per second asked for.
    rate: usize,
    pace: crate::clock::Pace,
    /// What `pace` owes, kept beside it in the same steps: its part of a step is how
    /// far the picture is from the network before the last step to the network now.
    owed: f32,
    running: bool,
    show_units: bool,
    /// The network as it is trained: the error and the chart go by it.
    net: Mlp,
    /// Its weights and biases before the last step, and the network as it is drawn,
    /// between the two (see `Fit1d::follow`).
    before: (Vec<Vec<f32>>, Vec<Vec<f32>>),
    drawn: Mlp,
    rng: Rng,
    xs: Vec<f32>,
    ys: Vec<f32>,
    hist: History,
    steps: u64,
    loss: f32,
}

impl Fit1d {
    pub fn new(_: &Gfx) -> Self {
        Self::create()
    }

    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let mut rng = Rng::new(31);
        let net = Mlp::new(&[1, 12, 1], Act::Relu, &mut rng);
        let mut s = Fit1d {
            target: ExprField::new("sin(2x) + 0.3 x", &["x"]),
            width: 12,
            act: 1,
            lr: 0.02,
            rate: 720,
            pace: crate::clock::Pace::default(),
            owed: 0.0,
            running: true,
            show_units: true,
            drawn: net.clone(),
            before: (net.w.clone(), net.b.clone()),
            net,
            rng,
            xs: (0..FIT_N).map(|i| -3.0 + 6.0 * i as f32 / (FIT_N - 1) as f32).collect(),
            ys: vec![0.0; FIT_N],
            hist: History::new(500),
            steps: 0,
            loss: 0.0,
        };
        s.sample_target();
        s.rebuild();
        s
    }

    fn sample_target(&mut self) {
        for (y, &x) in self.ys.iter_mut().zip(&self.xs) {
            let v = self.target.expr.eval(&[x]);
            *y = if v.is_finite() { v.clamp(-50.0, 50.0) } else { 0.0 };
        }
        // the error is measured against the new curve from now on
        self.loss = self.net.eval(&self.xs, &self.ys, FIT_N, Loss::Mse);
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let (_, _, target, width, act) = TRIALS[k];
        (self.width, self.act, self.lr) = (width, act, 0.02);
        self.target.set(TARGETS[target].1);
        self.sample_target();
        self.rebuild();
        self.running = true;
    }

    fn step(&mut self) {
        self.loss = self.net.train(&self.xs, &self.ys, FIT_N, self.lr, Loss::Mse);
        self.steps += 1;
        self.hist.push(self.loss);
    }

    fn rebuild(&mut self) {
        self.net = Mlp::new(&[1, self.width, 1], Act::ALL[self.act], &mut self.rng);
        // place each neuron's hinge (or the centre of its bump) somewhere in the interval
        for j in 0..self.width {
            let w = self.rng.range(0.6, 2.2) * if self.rng.bool(0.5) { 1.0 } else { -1.0 };
            let centre = self.rng.range(-3.0, 3.0);
            self.net.w[0][j] = w;
            self.net.b[0][j] = -w * centre;
            self.net.w[1][j] *= 0.5;
        }
        self.steps = 0;
        self.hist.clear();
        // (where the new weights stand, not where the last run ended)
        self.loss = self.net.eval(&self.xs, &self.ys, FIT_N, Loss::Mse);
        // (drawn where the run starts: nothing to go between yet)
        self.drawn = self.net.clone();
        self.keep_before();
    }

    /// Remember the network as it is, before a step.
    fn keep_before(&mut self) {
        self.before.0.clone_from(&self.net.w);
        self.before.1.clone_from(&self.net.b);
    }

    /// Bring the network as drawn to the part of a step the pace owes, between the
    /// network before the last step and the network now. The steps come whole, and
    /// not the same number in every frame (720 a second on a display of 75 is nine in
    /// some frames and ten in others; at a low rate, one in some and none in the
    /// rest): drawn as they come, the curves would move in fits and starts. So they
    /// are drawn up to a step behind the run, at the point the steps have reached.
    fn follow(&mut self) {
        let t = self.owed.clamp(0.0, 1.0);
        let (net, (w0, b0), drawn) = (&self.net, &self.before, &mut self.drawn);
        let between = |to: &mut [f32], from: &[f32], now: &[f32]| {
            for ((d, a), b) in to.iter_mut().zip(from).zip(now) {
                *d = a + (b - a) * t;
            }
        };
        let fits = |a: &[Vec<f32>], b: &[Vec<f32>]| a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.len() == b.len());
        if !(fits(w0, &net.w) && fits(b0, &net.b) && fits(&drawn.w, &net.w) && fits(&drawn.b, &net.b)) {
            *drawn = net.clone();
            return;
        }
        for l in 0..net.w.len() {
            between(&mut drawn.w[l], &w0[l], &net.w[l]);
            between(&mut drawn.b[l], &b0[l], &net.b[l]);
        }
    }
}

/// Draw what `draw` draws with `Frame::line` and `Frame::path` into a line batch of
/// its own, which differs from the usual one by a hair of its width and nothing else.
/// For the chart, which gains a point every few steps: in the usual batch it would
/// leave the curves beside it, which keep their number of points, impossible to
/// compare point for point from one frame to the next (as `eidos motion` does).
fn on_its_own(f: &mut Frame, draw: impl FnOnce(&mut Frame)) {
    let usual = std::mem::replace(&mut f.dl.lines, crate::gfx::draw::LineBatch::new(LineStyle::default()));
    draw(f);
    let own = std::mem::replace(&mut f.dl.lines, usual);
    // (each batch starts with a break: the first is enough)
    f.dl.lines_with(LineKind::Blend, LineStyle { width_scale: 1.0001, ..Default::default() }).pts.extend(own.pts.into_iter().skip(1));
}

impl Scene for Fit1d {
    fn view(&self) -> View {
        View::Fit { center: Vec2::new(0.0, 0.0), half: Vec2::new(3.5, 2.5) }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.16;
    }

    /// While training is paused nothing moves, so nothing needs redrawing.
    fn animated(&self) -> bool {
        self.running
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "One hidden layer is enough to approximate any continuous curve, given enough neurons. Here is why: every hidden neuron contributes one simple piece, a hinge or a smooth step (the thin lines), and the output just adds them up. More neurons, more pieces to build with. With $n$ neurons, each with its weights $w_j$, $v_j$ and bias $b_j$, and the activation $\\sigma$:");
        ui::equation(ui, r"\hat y(x) = c + \sum_{j=1}^{n} v_j \, \sigma(w_j x + b_j)");
        ui.horizontal(|ui| {
            if ui.button(if self.running { "pause" } else { "train" }).clicked() {
                self.running = !self.running;
            }
            if ui.button("start again").clicked() {
                self.rebuild();
            }
        });

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The curve to fit");
        if self.target.ui(ui, "target =") {
            self.sample_target();
        }
        let names: Vec<&str> = TARGETS.iter().map(|t| t.0).collect();
        if let Some(i) = ui::buttons(ui, &names) {
            self.target.set(TARGETS[i].1);
            self.sample_target();
        }
        ui::note(ui, "Any formula in $x$ will do. The network keeps training towards whatever is written here.");
        ui::section(ui, "Network");
        let mut arch = ui::slider_int_log(ui, "hidden neurons", &mut self.width, 1..=64);
        let names: Vec<&str> = Act::ALL.iter().map(|a| activation_name(*a)).collect();
        arch |= ui::choice(ui, &names, &mut self.act);
        ui::note(ui, r"The activation is the shape of one neuron's piece. The last two are the Hermite polynomials $h_2$ and $h_3$, and the one exception to the sentence at the top: every piece is then a parabola or a cubic, and pieces like that add up to a single parabola or cubic, however many neurons there are.");
        if arch {
            self.rebuild();
        }
        ui.checkbox(&mut self.show_units, "show each neuron's contribution");
        ui::section(ui, "Training");
        ui::slider_log(ui, "learning rate", &mut self.lr, 0.0005..=0.2);
        ui::slider_int_log(ui, "steps per second", &mut self.rate, 10..=12_000);
        ui::note(ui, "As many as the frame has time for: with many neurons a high rate is not reached, and the picture stays smooth instead.");
        ui::section(ui, "Reading the picture");
        let relu = Act::ALL[self.act] == Act::Relu;
        let mut key = vec![
            (pal::WHITE.mix(pal::DIM, 0.5), "thick grey curve: the target"),
            (pal::CYAN, r"cyan: the network's output $\hat y$, the sum of the thin lines (plus a constant $c$)"),
            (Cmap::Twilight.color(0.2).mix(pal::WHITE, 0.16), r"thin lines: one neuron's contribution $v_j \, \sigma(w_j x + b_j)$ each"),
        ];
        if relu {
            key.push((Cmap::Twilight.color(0.8).mix(pal::WHITE, 0.16), r"dots on the axis: where each ReLU switches on, at $x = -b_j / w_j$ (its hinge)"));
        }
        ui::key(ui, &key);
        if relu {
            ui::term(ui, "ReLU:", r"$\sigma(z) = \max(0, z)$: a neuron that gives zero up to some point and a straight line after it: one hinge.");
        }
        ui::section(ui, "Numbers");
        ui::readout(ui, "training steps", self.steps.to_string());
        ui::readout(ui, "mean squared error", loss_number(self.loss));
        ui::note(ui, r"The error is the average squared distance between the cyan curve and the target, over the points $x_i$ it is measured at:");
        ui::equation(ui, r"\frac{1}{N} \sum_{i=1}^{N} \bigl(\hat y(x_i) - y_i\bigr)^2");
        ui::see_also(ui, &[("hermite_expand", "building a curve from Hermite polynomials instead"), ("mlp", "a deeper network on points in the plane"), ("particles", "do the neurons move, or only their weights?")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        if self.running && f.dt > 0.0 {
            let rate = self.rate as f32;
            let mut turn = self.pace.begin(rate, f.dt, f.real_dt, 0.004);
            // (what the pace owes, worked out as it works it out, and paid off step by step)
            self.owed = (self.owed + rate * f.dt).min((rate * 0.1).max(2.0));
            while turn.next() {
                self.owed -= 1.0;
                self.keep_before();
                self.step();
            }
            self.pace.end(turn);
            self.follow();
        }
        // (the curves and the hinges are those of the network as drawn; the error
        // written beside them is the run's own)
        let pred: Vec<f32> = self.drawn.forward(&self.xs, FIT_N).to_vec();
        let h = self.width;

        if self.show_units {
            let w2 = &self.drawn.w[1];
            // the middle of this colour map is nearly black: lighten it, or half the
            // lines vanish into the background
            let tint = |j: usize| Cmap::Twilight.color((j as f32 + 0.5) / h as f32).mix(pal::WHITE, 0.16);
            for j in 0..h {
                let unit = self.drawn.neuron(0, j);
                f.path((0..FIT_N).map(|i| Vec2::new(self.xs[i], w2[j] * unit[i])), 1.3, tint(j).a(0.75));
            }
            // where each rectifier switches on
            if Act::ALL[self.act] == Act::Relu {
                for j in 0..h {
                    let (w, b) = (self.drawn.w[0][j], self.drawn.b[0][j]);
                    if w.abs() > 1e-4 {
                        let x = -b / w;
                        if x.abs() < 3.0 {
                            f.dot((x, 0.0), 7.0, tint(j));
                        }
                    }
                }
            }
        }
        f.path((0..FIT_N).map(|i| Vec2::new(self.xs[i], self.ys[i])), 4.5, pal::WHITE.a(0.3));
        f.glow_path((0..FIT_N).map(|i| Vec2::new(self.xs[i], pred[i])), 2.6, pal::CYAN);
        f.path((0..FIT_N).map(|i| Vec2::new(self.xs[i], pred[i])), 2.2, pal::CYAN);

        let (lo, hi) = f.cam.flat_bounds();
        let size = hi - lo;
        let o = Vec2::new(hi.x - size.x * 0.30, lo.y + size.y * 0.07);
        let cs = Vec2::new(size.x * 0.27, size.y * 0.16);
        on_its_own(f, |f| {
            chart(f, o, cs, &[(self.hist.points(), pal::CYAN)], true);
        });
        // (as much of the title as the chart is wide: it is a quarter of the view)
        let ppu = pixels_per_unit(f);
        let title = ["error over training steps (log scale)", "error (log scale)", "error"].into_iter().find(|t| text_width(t, 12.0) <= cs.x * ppu).unwrap_or("");
        f.text(o + Vec2::new(0.0, cs.y + 0.05), title, 12.0, pal::DIM, Align2::LEFT_BOTTOM);
        // clear of the end of the thick curve: beside it where there is room at the
        // left of the view, otherwise under it
        if (-3.0 - lo.x) * ppu > text_width("target", 13.0) + 14.0 {
            f.text_off((-3.0, self.ys[0]), Vec2::new(-9.0, 0.0), "target", 13.0, pal::DIM, Align2::RIGHT_CENTER);
        } else {
            f.text_off((-3.0, self.ys[0]), Vec2::new(0.0, 12.0), "target", 13.0, pal::DIM, Align2::LEFT_TOP);
        }
        f.axes2();
        let mut key = vec!["grey: target", "cyan: the network's output"];
        if self.show_units {
            key.push("thin: one neuron each");
            if Act::ALL[self.act] == Act::Relu {
                key.push("dots: where each ReLU switches on");
            }
        }
        bottom_caption(f, &key, 12.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as the scene runs it, for `steps` steps: the error it has
    /// settled at (the lowest of the last 300 steps: Adam's error jumps now and then)
    /// and the scene itself for a closer look. `act` overrides the experiment's activation.
    fn run_with(k: usize, act: Option<usize>, seed: u64, steps: usize) -> (f32, Fit1d) {
        let mut scene = Fit1d::create();
        scene.rng = Rng::new(seed);
        scene.set_up(k);
        if let Some(act) = act {
            scene.act = act;
            scene.rebuild();
        }
        let mut settled = f32::MAX;
        for step in 0..steps {
            scene.step();
            if step + 300 >= steps {
                settled = settled.min(scene.loss);
            }
        }
        (settled, scene)
    }

    fn run(k: usize, seed: u64, steps: usize) -> (f32, Fit1d) {
        run_with(k, None, seed, steps)
    }

    /// The captions of the "things to try", held to account (three starts each).
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        for seed in [31u64, 32, 33] {
            // three hinges are too few for the wave; sixty-four follow it closely
            let (three, _) = run(0, seed, 6000);
            let (many, _) = run(1, seed, 6000);
            assert!(three > 0.05 && many < 0.02 && three > 10.0 * many, "seed {seed}: three hinges {three}, sixty-four {many}");
            // eight smooth steps fit the bump
            let (bump, _) = run(2, seed, 6000);
            assert!(bump < 0.01, "seed {seed}: eight tanh neurons on the bump: {bump}");
            // a sudden step: two tanh neurons have it, two ReLU hinges are left with corners
            let (smooth, _) = run(3, seed, 6000);
            let (hinged, _) = run_with(3, Some(1), seed, 6000);
            assert!(smooth < 1e-4 && hinged > 3e-4, "seed {seed}: the step with tanh {smooth}, with ReLU {hinged}");
            // the chirp: the slow middle is learned before the fast edges
            let (_, mut chirp) = run(4, seed, 1500);
            let fit = chirp.net.forward(&chirp.xs, FIT_N).to_vec();
            let error_where = |inside: bool| {
                let picked: Vec<f32> = (0..FIT_N).filter(|&i| (chirp.xs[i].abs() < 1.5) == inside).map(|i| (fit[i] - chirp.ys[i]).powi(2)).collect();
                picked.iter().sum::<f32>() / picked.len() as f32
            };
            assert!(error_where(true) < 0.3 * error_where(false), "seed {seed}: error in the middle {} against {} at the edges", error_where(true), error_where(false));
        }
    }

    /// A restart shows the new network's error, not the last run's.
    #[test]
    fn a_restart_starts_the_read_outs_afresh() {
        let (trained, mut scene) = run(1, 31, 2000);
        scene.rebuild();
        assert!(scene.steps == 0 && scene.hist.is_empty() && scene.loss > 10.0 * trained, "{} after {trained}", scene.loss);
        // and a new curve to fit is measured against at once
        let before = scene.loss;
        scene.target.set(TARGETS[2].1);
        scene.sample_target();
        assert!(scene.loss != before);
    }

    /// The mean squared error of the best polynomial of degree `degree` through the
    /// points (least squares, in double precision).
    fn best_polynomial(xs: &[f32], ys: &[f32], degree: usize) -> f32 {
        let k = degree + 1;
        // the normal equations, solved by elimination
        let mut a = vec![vec![0.0f64; k + 1]; k];
        for (&x, &y) in xs.iter().zip(ys) {
            let powers: Vec<f64> = (0..k).map(|i| (x as f64).powi(i as i32)).collect();
            for i in 0..k {
                for j in 0..k {
                    a[i][j] += powers[i] * powers[j];
                }
                a[i][k] += powers[i] * y as f64;
            }
        }
        for c in 0..k {
            for r in c + 1..k {
                let f = a[r][c] / a[c][c];
                for j in c..=k {
                    a[r][j] -= f * a[c][j];
                }
            }
        }
        let mut coef = vec![0.0f64; k];
        for c in (0..k).rev() {
            coef[c] = (a[c][k] - (c + 1..k).map(|j| a[c][j] * coef[j]).sum::<f64>()) / a[c][c];
        }
        let fit = |x: f32| (0..k).map(|i| coef[i] * (x as f64).powi(i as i32)).sum::<f64>();
        (xs.iter().zip(ys).map(|(&x, &y)| (fit(x) - y as f64).powi(2)).sum::<f64>() / xs.len() as f64) as f32
    }

    /// Polynomial pieces add up to a polynomial: sixty-four neurons of (x²−1)/√2 come to
    /// the best parabola for the wave within a few hundred steps and never do better
    /// (for this wave it is a straight line), and the same holds for (x³−3x)/√6 and
    /// the best cubic. Sixty-four ReLU hinges get twenty times closer (the test above).
    #[test]
    fn polynomial_pieces_add_up_to_one_polynomial() {
        let quadratic = Act::ALL.iter().position(|&a| a == Act::Hermite2);
        assert_eq!(quadratic, Some(TRIALS[5].4));
        for seed in [31u64, 32, 33] {
            for (act, degree) in [(Act::Hermite2, 2), (Act::Hermite3, 3)] {
                let mut scene = Fit1d::create();
                scene.rng = Rng::new(seed);
                scene.set_up(5);
                scene.act = Act::ALL.iter().position(|&a| a == act).unwrap_or(0);
                scene.rebuild();
                let best = best_polynomial(&scene.xs, &scene.ys, degree);
                let mut lowest = f32::MAX;
                for _ in 0..500 {
                    scene.step();
                    lowest = lowest.min(scene.loss);
                }
                let now = scene.net.eval(&scene.xs, &scene.ys, FIT_N, Loss::Mse);
                for _ in 0..2500 {
                    scene.step();
                    lowest = lowest.min(scene.loss);
                }
                let name = act.name();
                assert!(lowest > best * (1.0 - 1e-4), "seed {seed}, {name}: {lowest} below the best polynomial's {best}");
                assert!(now < best * 1.005, "seed {seed}, {name}: {now} after 500 steps, against the best polynomial's {best}");
                if degree == 2 {
                    // the best parabola for this wave is a line: what the network draws is one
                    let out = scene.net.forward(&scene.xs, FIT_N).to_vec();
                    let off_line = best_polynomial(&scene.xs, &out, 1);
                    assert!(best > 0.4 && off_line < 1e-3, "seed {seed}: error {best}, {off_line} away from a line");
                }
            }
        }
    }

    /// The errors behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for seed in [31u64, 32, 33, 34] {
            for (k, trial) in TRIALS.iter().enumerate() {
                let errors: Vec<String> = [500, 1500, 3000, 6000].iter().map(|&steps| format!("{:.4}", run(k, seed, steps).0)).collect();
                println!("seed {seed} {:<22} {}", trial.0, errors.join("  "));
            }
            // the step with each activation and a few widths
            for (act, width) in [(0usize, 12usize), (1, 12), (0, 2), (1, 2), (0, 4), (1, 4)] {
                let mut scene = Fit1d::create();
                scene.rng = Rng::new(seed);
                scene.set_up(3);
                (scene.act, scene.width) = (act, width);
                scene.rebuild();
                let mut line = format!("seed {seed} step, {} x{width:<3}", Act::ALL[act].name());
                let mut done = 0;
                for mark in [100, 300, 1000, 3000, 6000] {
                    (done..mark).for_each(|_| scene.step());
                    done = mark;
                    line += &format!("  {:.5}", scene.loss);
                }
                println!("{line}");
            }
        }
    }
}


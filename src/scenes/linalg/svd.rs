//! SVD, step by step.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "svd",
    title: "SVD, step by step",
    group: "Linear algebra: data and dimensions",
    blurb: "Any matrix is a rotation, a stretch, and another rotation.",
    make: |g| Box::new(SvdSteps::new(g)),
};

/// Experiments: (button, what to look for). The matrix each one sets is `trial_matrix`.
const TRIALS: [(&str, &str); 4] = [
    (
        "a shear",
        r"A shear only slides the plane sideways, yet it too is a turn, a stretch and a turn. It keeps areas ($\det A = 1$), so its two stretches multiply to 1: the ellipse has the same area as the circle.",
    ),
    ("squash flat", r"The second column is two thirds of the first, so the whole plane lands on a line: $\sigma_2 = 0$. In step 2 the circle is pressed flat into a segment, and the last turn cannot open it up again."),
    (
        "a flip",
        r"This matrix flips the plane over ($\det A < 0$). A turn cannot flip anything, so the flip goes into the stretch: $\sigma_2$ comes out negative. In step 2 the circle is pressed flat and opens out mirrored: afterwards the beads run round the other way.",
    ),
    (
        "stretch, then turn",
        r"This matrix was made by stretching along the axes (by 1.8 and 0.7), then turning by 45.8°. The decomposition finds exactly that: the first turn is 0°, $\Sigma$ is 1.8 and 0.7, and $U$ turns by 45.8°.",
    ),
];

/// The matrix of experiment `k`.
fn trial_matrix(k: usize) -> Mat2 {
    match k {
        0 => Mat2::from_cols(Vec2::X, Vec2::new(1.2, 1.0)),
        1 => Mat2::from_cols(Vec2::new(1.2, 0.6), Vec2::new(0.8, 0.4)),
        2 => Mat2::from_cols(Vec2::new(0.4, 1.3), Vec2::new(1.2, 0.3)),
        _ => Mat2::from_angle(0.8) * Mat2::from_diagonal(Vec2::new(1.8, 0.7)),
    }
}

/// A = U Σ Vᵀ as three separate motions.
pub struct SvdSteps {
    m: Mat2,
    stage: f32,
    playing: bool,
}

impl SvdSteps {
    pub fn new(_: &Gfx) -> Self {
        Self::create()
    }

    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        SvdSteps { m: Mat2::from_cols(Vec2::new(1.6, 0.9), Vec2::new(-0.3, 1.1)), stage: 0.0, playing: true }
    }

    /// Play the three steps from the start: at stage 0 the picture is the identity.
    fn replay(&mut self) {
        self.stage = 0.0;
        self.playing = true;
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        self.m = trial_matrix(k);
        self.replay();
    }

    /// The four numbers of the matrix. A number typed in is a new matrix, shown from
    /// the start; one being dragged is followed as it goes, at whatever stage the
    /// picture is (the whole of A, once it has played).
    fn edit_matrix(&mut self, ui: &mut egui::Ui) {
        if mat2_edit(ui, &mut self.m) {
            self.replay();
        }
    }

    /// Move the animation on by `dt` seconds of scene time.
    fn advance(&mut self, dt: f32) {
        if self.playing {
            self.stage += dt / 1.5;
            if self.stage >= 3.0 {
                self.stage = 3.0;
                self.playing = false;
            }
        }
    }

    /// The matrix as drawn at the current stage: Vᵀ by the end of stage 1, Σ Vᵀ by
    /// the end of stage 2, U Σ Vᵀ = A at the end.
    fn staged(&self) -> Mat2 {
        let (u, s, v) = proper(svd2(self.m));
        let (a_v, a_u) = (-angle_of(v), angle_of(u));
        let k = [smoothstep(self.stage), smoothstep(self.stage - 1.0), smoothstep(self.stage - 2.0)];
        let stretch = Mat2::from_diagonal(Vec2::ONE.lerp(s, k[1]));
        Mat2::from_angle(a_u * k[2]) * stretch * Mat2::from_angle(a_v * k[0])
    }
}

/// Rotation angle of a proper rotation matrix.
fn angle_of(r: Mat2) -> f32 {
    r.x_axis.y.atan2(r.x_axis.x)
}

/// An angle in degrees, to a tenth, with no "−0.0" for a turn that rounds to nothing.
fn degrees(radians: f32) -> String {
    let d = radians.to_degrees();
    format!("{:.1}°", if d.abs() < 0.05 { 0.0 } else { d })
}

impl Scene for SvdSteps {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 3.6 }
    }

    fn animated(&self) -> bool {
        self.playing
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Every $2 \times 2$ matrix $A$, however it looks, does just three things in a row: it turns the plane (the turn $V^\top$), stretches it along the $x$ and $y$ axes (by $\sigma_1$ and $\sigma_2$, together $\Sigma$), and turns it again ($U$). That is what this says, read from right to left:");
        ui::equation(ui, r"A = \textcolor{pink}{U} \, \textcolor{yellow}{\Sigma} \, \textcolor{cyan}{V^\top}");
        ui::note(ui, "Watch the unit circle go through the three steps.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Matrix $A$");
        self.edit_matrix(ui);
        ui::note(ui, "Type a number to see the new matrix go through the steps from the start; drag one sideways and the picture follows it, at whatever stage it is.");
        ui::section(ui, "Stage");
        if ui.button(if self.playing { "pause" } else { "play" }).clicked() {
            self.playing = !self.playing;
            if self.stage >= 3.0 {
                self.stage = 0.0;
            }
        }
        if ui::slider(ui, "stage", &mut self.stage, 0.0..=3.0) {
            self.playing = false;
        }
        ui::note(ui, "0 is the plane as it was; 1, 2 and 3 are the ends of the three steps. At 3 the circle is where $A$ sends it.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::WHITE, "white curve: the unit circle, carried along; it ends as an ellipse"),
                (pal::DIM, "faint circle: where it started"),
                (crate::gfx::color::Cmap::Twilight.color(0.3), "beads: marks painted on the circle, so that a turn shows"),
                (pal::BLUE, "blue grid: the square grid, carried along"),
                (pal::ORANGE, r"orange arrow: $v_1$, the direction $A$ stretches most; it ends as $\sigma_1 u_1$, the ellipse's long half-axis"),
                (pal::PURPLE, r"purple arrow: $v_2$, the direction $A$ stretches least; it ends as $\sigma_2 u_2$, the short half-axis"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Singular values:", r"$\sigma_1$ and $\sigma_2$, how much the circle is stretched along the two axes: the most and the least $A$ stretches any direction (the least is $|\sigma_2|$, since for a matrix that flips the plane this scene makes $\sigma_2$ negative).");
        ui::term(ui, "Singular vectors:", r"$v_1$ and $v_2$, at right angles, are the directions stretched most and least; $A$ sends them to $\sigma_1 u_1$ and $\sigma_2 u_2$, again at right angles.");
        ui::term(ui, r"$V^\top$:", r"the turn that undoes $V$. $V$ turns the $x$ and $y$ axes onto $v_1$ and $v_2$, so $V^\top$ turns $v_1$ and $v_2$ onto the axes.");

        let d = svd2(self.m);
        ui::section(ui, "Decomposition");
        ui::readout_colored(ui, r"1. rotate by $V^\top$", format!("${}$", degrees(-angle_of(proper(d).2))), pal::CYAN);
        ui::readout_colored(ui, r"2. stretch by $\Sigma$", format!("${}, {}$", fmt(proper(d).1.x), fmt(proper(d).1.y)), pal::YELLOW);
        ui::readout_colored(ui, "3. rotate by $U$", format!("${}$", degrees(angle_of(proper(d).0))), pal::PINK);
        if proper(d).1.y < 0.0 {
            ui::note(ui, r"A negative stretch: this matrix flips orientation, and the flip has to live in $\Sigma$ because rotations cannot do it.");
        }
        ui::see_also(ui, &[("linmap2", "drag a matrix's columns and watch the circle's ellipse"), ("lowrank", "a picture taken apart the same way, and rebuilt from its largest pieces"), ("condition", r"$\sigma_1 / \sigma_2$: how far solving spreads an error"), ("pca", "the same axes for a cloud of data")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt);
        let v = proper(svd2(self.m)).2;
        let m = self.staged();

        warped_grid(f, m, 8, pal::BLUE);
        f.circle(Vec2::ZERO, Vec3::Z, 1.0, 1.0, pal::DIM.a(0.5));

        // the unit circle, with beads so that pure rotation is visible
        let n = 160;
        let ring: Vec<Vec3> = (0..n).map(|i| (m * Vec2::from_angle(i as f32 / n as f32 * TAU)).extend(0.0)).collect();
        f.dl.lines.closed(&ring, 2.6, pal::WHITE);
        for i in 0..16 {
            let a = i as f32 / 16.0 * TAU;
            let c = crate::gfx::color::Cmap::Twilight.color(i as f32 / 16.0);
            f.dot(m * Vec2::from_angle(a), 9.0, c);
        }

        // right singular vectors ride along and end up as the ellipse's axes; each is
        // named for what it is once the last finished step has been applied
        let done = (self.stage.floor() as usize).min(3);
        for (i, c) in [pal::ORANGE, pal::PURPLE].into_iter().enumerate() {
            let tip = m * v.col(i);
            f.arrow(Vec2::ZERO, tip, 3.0, c);
            // beside the tip, on the side away from the centre: outside the ellipse
            let (right, up) = (tip.x >= -1e-3, tip.y >= -1e-3);
            let align = match (right, up) {
                (true, true) => Align2::LEFT_BOTTOM,
                (false, true) => Align2::RIGHT_BOTTOM,
                (true, false) => Align2::LEFT_TOP,
                (false, false) => Align2::RIGHT_TOP,
            };
            let side = Vec2::new(if right { 9.0 } else { -9.0 }, if up { -9.0 } else { 9.0 });
            f.text_off(tip, side, ARROW_NAMES[i][done], 15.0, c, align);
        }

        let (title, col) = if self.stage < 1.0 {
            (r"1 / 3   rotate:  $V^\top$", pal::CYAN)
        } else if self.stage < 2.0 {
            (r"2 / 3   stretch:  $\Sigma$", pal::YELLOW)
        } else if self.stage < 3.0 {
            ("3 / 3   rotate:  $U$", pal::PINK)
        } else {
            (r"$A = U \Sigma V^\top$", pal::WHITE)
        };
        let size = f.view_size();
        // on a plate: the y axis and the grid run straight through the title otherwise
        f.caption_px(Vec2::new(size.x * 0.5, f.top_inset + 22.0), title, 20.0, col, Align2::CENTER_TOP);
        f.axes2();
        bottom_caption(f, &["white: the unit circle, carried along", "faint: where it started", "blue grid: the square grid"], 12.0);
    }
}

/// What the two arrows are called once 0, 1, 2 or 3 steps have been applied to them.
const ARROW_NAMES: [[&str; 4]; 2] = [["$v_1$", "$e_1$", r"$\sigma_1 e_1$", r"$\sigma_1 u_1$"], ["$v_2$", "$e_2$", r"$\sigma_2 e_2$", r"$\sigma_2 u_2$"]];

/// Make U a proper rotation as well, moving any reflection into the sign of σ₂.
/// (V always is one: `svd2` builds it from an angle.)
fn proper(d: crate::math::linalg::Svd2) -> (Mat2, Vec2, Mat2) {
    let (mut u, mut s, v) = (d.u, d.s, d.v);
    if u.determinant() < 0.0 {
        u.y_axis = -u.y_axis;
        s.y = -s.y;
    }
    (u, s, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, at `stage`.
    fn at(k: usize, stage: f32) -> SvdSteps {
        let mut scene = SvdSteps::create();
        scene.set_up(k);
        (scene.stage, scene.playing) = (stage, false);
        scene
    }

    /// The unit circle as drawn at the scene's stage, with `n` points.
    fn ring(scene: &SvdSteps, n: usize) -> Vec<Vec2> {
        (0..n).map(|i| scene.staged() * Vec2::from_angle(i as f32 / n as f32 * TAU)).collect()
    }

    /// The signed area of a polygon, positive if its corners run anticlockwise.
    fn area(corners: &[Vec2]) -> f32 {
        (0..corners.len()).map(|i| corners[i].perp_dot(corners[(i + 1) % corners.len()])).sum::<f32>() * 0.5
    }

    /// The panel's matrix editor, driven as a visitor would: the pointer and the
    /// keyboard through egui, a sixtieth of a second a frame, the scene's clock
    /// moving on with it.
    struct Panel {
        ctx: egui::Context,
        frame: u32,
        /// The middle of the first number (row 1, column 1).
        first: egui::Pos2,
    }

    impl Panel {
        fn new(scene: &mut SvdSteps) -> Panel {
            let mut panel = Panel { ctx: egui::Context::default(), frame: 0, first: egui::Pos2::ZERO };
            let mut corner = egui::Pos2::ZERO;
            let raw = panel.input(Vec::new());
            let _ = panel.ctx.run_ui(raw, |ui| {
                corner = ui.next_widget_position();
                scene.edit_matrix(ui);
            });
            panel.first = corner + egui::vec2(14.0, 9.0);
            panel
        }

        fn input(&mut self, events: Vec<egui::Event>) -> egui::RawInput {
            self.frame += 1;
            let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 300.0));
            egui::RawInput { screen_rect: Some(screen), time: Some(self.frame as f64 / 60.0), predicted_dt: 1.0 / 60.0, focused: true, events, ..Default::default() }
        }

        fn step(&mut self, scene: &mut SvdSteps, events: Vec<egui::Event>) {
            let raw = self.input(events);
            let _ = self.ctx.run_ui(raw, |ui| scene.edit_matrix(ui));
            scene.advance(1.0 / 60.0);
        }

        fn button(pos: egui::Pos2, pressed: bool) -> egui::Event {
            egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() }
        }
    }

    /// Dragging one of the numbers changes the matrix and the picture follows it, at
    /// the stage it was at (it used to start again from the identity every frame of
    /// the drag, so that nothing but the identity was seen until the drag ended).
    /// Typing a number plays the new matrix from the start.
    #[test]
    fn dragging_an_entry_follows_it_and_typing_one_replays() {
        let mut scene = SvdSteps::create();
        (scene.stage, scene.playing) = (3.0, false);
        let mut panel = Panel::new(&mut scene);
        let p = panel.first;
        panel.step(&mut scene, vec![egui::Event::PointerMoved(p)]);
        panel.step(&mut scene, vec![Panel::button(p, true)]);
        let mut seen = Vec::new();
        for k in 1..=20 {
            panel.step(&mut scene, vec![egui::Event::PointerMoved(p + egui::vec2(4.0 * k as f32, 0.0))]);
            seen.push(scene.m.x_axis.x);
            assert!(scene.stage == 3.0 && !scene.playing, "frame {k} of the drag: stage {}", scene.stage);
        }
        panel.step(&mut scene, vec![Panel::button(p + egui::vec2(80.0, 0.0), false)]);
        assert!(scene.stage == 3.0 && !scene.playing);
        // the number really was dragged, a little at a time, and A on screen is the new A
        assert!(seen.windows(2).filter(|w| w[1] > w[0]).count() > 10 && scene.m.x_axis.x > 2.2, "{seen:?}");
        let shown = scene.staged();
        assert!((shown.x_axis - scene.m.x_axis).length() + (shown.y_axis - scene.m.y_axis).length() < 1e-5);

        // click the number and type 2: a new matrix, played from the start
        let p = panel.first;
        panel.step(&mut scene, vec![egui::Event::PointerMoved(p), Panel::button(p, true)]);
        panel.step(&mut scene, vec![Panel::button(p, false)]);
        panel.step(&mut scene, Vec::new());
        assert!(scene.stage == 3.0 && !scene.playing, "a click alone changes nothing");
        panel.step(&mut scene, vec![egui::Event::Text("2".into())]);
        assert!(scene.m.x_axis.x == 2.0 && scene.playing && scene.stage < 0.02, "{} at stage {}", scene.m.x_axis.x, scene.stage);
    }

    /// The three steps put together are the matrix, and the arrows are what their
    /// names say after each step: e at 1, σe at 2, σu at 3.
    #[test]
    fn the_three_steps_compose_to_the_matrix() {
        let mut rng = Rng::new(3);
        let mut matrices: Vec<Mat2> = (0..TRIALS.len()).map(trial_matrix).collect();
        matrices.push(SvdSteps::create().m);
        matrices.extend((0..200).map(|_| Mat2::from_cols(Vec2::new(rng.normal(), rng.normal()), Vec2::new(rng.normal(), rng.normal()))));
        for a in matrices {
            let mut scene = SvdSteps::create();
            (scene.m, scene.playing) = (a, false);
            let (u, s, v) = proper(svd2(a));
            let size = a.to_cols_array().iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1.0);
            scene.stage = 0.0;
            assert!((scene.staged() - Mat2::IDENTITY).abs().to_cols_array().iter().all(|x| *x < 1e-6));
            for (stage, want) in [(1.0, [Vec2::X, Vec2::Y]), (2.0, [Vec2::X * s.x, Vec2::Y * s.y]), (3.0, [u.col(0) * s.x, u.col(1) * s.y])] {
                scene.stage = stage;
                for i in 0..2 {
                    let tip = scene.staged() * v.col(i);
                    assert!((tip - want[i]).length() < 1e-5 * size, "{a}: arrow {i} at stage {stage} is {tip}, not {} ({})", want[i], ARROW_NAMES[i][stage as usize]);
                }
            }
            let back = scene.staged();
            assert!((back.x_axis - a.x_axis).length() + (back.y_axis - a.y_axis).length() < 2e-5 * size, "{a} came back as {back}");
            assert!((u.determinant() - 1.0).abs() < 1e-5 && (v.determinant() - 1.0).abs() < 1e-5);
            // the note on a negative stretch is there exactly when A flips the plane
            assert!((s.y < 0.0) == (a.determinant() < 0.0) || a.determinant().abs() < 1e-6, "{a}: σ₂ = {}", s.y);
        }
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // a shear: det 1, the stretches multiply to 1, the ellipse has the circle's area
        let (_, s, _) = proper(svd2(trial_matrix(0)));
        assert!(trial_matrix(0).determinant() == 1.0 && (s.x * s.y - 1.0).abs() < 1e-5, "{s}");
        let ratio = area(&ring(&at(0, 3.0), 160)) / area(&ring(&at(0, 0.0), 160));
        assert!((ratio - 1.0).abs() < 1e-5, "{ratio}");

        // squash flat: sigma2 = 0; the circle is round after step 1, flat from the
        // end of step 2 on, and the last turn leaves it flat
        let (_, s, _) = proper(svd2(trial_matrix(1)));
        assert!(s.y.abs() < 1e-6 && fmt(s.y) == "0" && s.x > 1.0, "{s}");
        assert!(ring(&at(1, 1.0), 64).iter().all(|p| (p.length() - 1.0).abs() < 1e-5));
        assert!(area(&ring(&at(1, 1.5), 64)).abs() > 1.0);
        for stage in [2.0, 2.5, 3.0] {
            let pts = ring(&at(1, stage), 64);
            let along = pts.iter().copied().max_by(|a, b| a.length().total_cmp(&b.length())).unwrap().normalize();
            assert!(pts.iter().all(|p| p.perp_dot(along).abs() < 1e-5), "not flat at stage {stage}");
        }

        // a flip: det < 0 and sigma2 < 0; the circle goes flat during step 2 and the
        // beads run round the other way after it
        let (_, s, _) = proper(svd2(trial_matrix(2)));
        assert!(trial_matrix(2).determinant() < 0.0 && s.y < 0.0, "{s}");
        let beads = |stage: f32| area(&ring(&at(2, stage), 16));
        assert!(beads(1.0) > 2.0 && beads(2.0) < -0.5 && beads(3.0) < -0.5, "{} {} {}", beads(1.0), beads(2.0), beads(3.0));
        let flattest = (0..=100).map(|i| beads(1.0 + i as f32 / 100.0).abs()).fold(f32::MAX, f32::min);
        assert!(flattest < 0.03, "{flattest}");

        // stretch, then turn: the readouts give back 0 degrees, 1.8 and 0.7, 45.8 degrees
        let (u, s, v) = proper(svd2(trial_matrix(3)));
        assert_eq!([degrees(-angle_of(v)), format!("${}, {}$", fmt(s.x), fmt(s.y)), degrees(angle_of(u))], ["0.0°", "$1.800, 0.700$", "45.8°"]);
    }

    /// A column at the origin, the columns on one line, the whole matrix zero: the
    /// picture and the read-outs stay finite at every stage.
    #[test]
    fn degenerate_matrices_leave_everything_finite() {
        for a in [
            Mat2::ZERO,
            Mat2::from_cols(Vec2::ZERO, Vec2::new(-0.3, 1.1)),
            Mat2::from_cols(Vec2::new(1.6, 0.9), Vec2::ZERO),
            Mat2::from_cols(Vec2::new(1.6, 0.9), Vec2::new(-0.8, -0.45)),
            Mat2::from_cols(Vec2::new(1e-8, 0.0), Vec2::new(0.0, 1e-8)),
        ] {
            let mut scene = SvdSteps::create();
            (scene.m, scene.playing) = (a, false);
            let (u, s, v) = proper(svd2(a));
            assert!(u.is_finite() && s.is_finite() && v.is_finite(), "{a}");
            for i in 0..=30 {
                scene.stage = i as f32 * 0.1;
                assert!(scene.staged().is_finite() && ring(&scene, 16).iter().all(|p| p.is_finite()), "{a} at stage {}", scene.stage);
            }
            let read = [degrees(-angle_of(v)), fmt(s.x), fmt(s.y), degrees(angle_of(u))].join(" ");
            assert!(!read.contains("NaN") && !read.contains("inf"), "{a}: {read}");
        }
    }
}

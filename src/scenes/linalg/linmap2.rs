//! Linear maps.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "linmap2",
    title: "Linear maps",
    group: "Linear algebra: vectors and maps",
    blurb: "Drag the basis vectors; see determinant, eigenvectors and singular values.",
    make: |g| Box::new(LinearMap2::new(g)),
};

/// Experiments: (button, what to look for, the matrix's columns (x₁, y₁, x₂, y₂),
/// whether the unit circle is shown).
const TRIALS: [(&str, &str, [f32; 4], bool); 4] = [
    (
        "a shear",
        r"The $x$ axis stays put and every line above it slides sideways, the more the higher it is. The square leans over but keeps its area: $\det A = 1$. Only one line stays on itself: the $x$ axis ($\lambda = 1$).",
        [1.0, 0.0, 1.0, 1.0],
        false,
    ),
    (
        "a rotation",
        r"Every arrow turns by 45° and keeps its length, so no line stays on itself: there are no eigenvector lines to draw, and the eigenvalues are complex numbers. Areas are kept: $\det A = 1$.",
        [std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2, -std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2],
        false,
    ),
    (
        "a reflection",
        r"$x$ and $y$ swap places: the plane is flipped over its diagonal. The square turns pink, as it does whenever the plane is flipped, and $\det A = -1$. The diagonal stays put ($\lambda = 1$); the other diagonal is reversed ($\lambda = -1$).",
        [0.0, 1.0, 1.0, 0.0],
        false,
    ),
    (
        "squash onto a line",
        r"Every point drops at right angles onto the line through $(2, 1)$. Points on that line stay put ($\lambda = 1$); points on the line across it land on $0$ ($\lambda = 0$). The square and the circle are pressed flat onto the line: $\det A = 0$, $\sigma_2 = 0$, $\rank A = 1$.",
        [0.8, 0.4, 0.4, 0.2],
        true,
    ),
];

pub struct LinearMap2 {
    m: Mat2,
    v: Vec2,
    morph: f32,
    playing: bool,
    show_eigen: bool,
    show_circle: bool,
    show_area: bool,
}

impl LinearMap2 {
    pub fn new(_: &Gfx) -> Self {
        Self::create()
    }

    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        LinearMap2 {
            m: Mat2::from_cols(Vec2::new(1.5, 0.5), Vec2::new(-0.5, 1.2)),
            v: Vec2::new(1.0, 1.5),
            morph: 1.0,
            playing: false,
            show_eigen: true,
            show_circle: false,
            show_area: true,
        }
    }

    /// Set up one of the experiments in `TRIALS`, and play it from the identity.
    fn set_up(&mut self, k: usize) {
        let (_, _, [a, b, c, d], circle) = TRIALS[k];
        self.m = Mat2::from_cols(Vec2::new(a, b), Vec2::new(c, d));
        self.v = Vec2::new(1.0, 1.5);
        (self.show_area, self.show_eigen, self.show_circle) = (true, true, circle);
        (self.morph, self.playing) = (0.0, true);
    }

    /// Move the animation on by `dt` seconds of scene time.
    fn advance(&mut self, dt: f32) {
        if self.playing {
            self.morph += dt / 1.6;
            if self.morph >= 1.0 {
                self.morph = 1.0;
                self.playing = false;
            }
        }
    }

    /// The matrix as drawn: on its way from the identity to `m` while the animation plays.
    fn shown(&self) -> Mat2 {
        let t = smoothstep(self.morph);
        Mat2::from_cols(Vec2::X.lerp(self.m.x_axis, t), Vec2::Y.lerp(self.m.y_axis, t))
    }

    /// The eigenvector lines of the matrix, as drawn: (λ, unit direction, tip of the
    /// arrow on it). Each line is mapped onto itself at every stage of the animation,
    /// since (1 − t) I + t A sends v to (1 − t + t λ) v: the arrow is where the
    /// shown matrix sends the unit vector along the line.
    fn eigen_arrows(&self) -> Vec<(f32, Vec2, Vec2)> {
        let t = smoothstep(self.morph);
        match eig2(self.m) {
            Eig2::Real { l1, l2, v1, v2 } => [(l1, v1), (l2, v2)].into_iter().map(|(l, v)| (l, v, v * (1.0 + (l - 1.0) * t))).collect(),
            Eig2::Complex { .. } => Vec::new(),
        }
    }
}

/// The unit square after the map `m`, corner by corner from the origin.
fn square_image(m: Mat2) -> [Vec2; 4] {
    [Vec2::ZERO, m.x_axis, m.x_axis + m.y_axis, m.y_axis]
}

/// The colour of the square's image: pink once the plane has been flipped over.
fn area_colour(det: f32) -> Color {
    if det >= 0.0 {
        pal::YELLOW
    } else {
        pal::PINK
    }
}

/// How many dimensions the image has, from the singular values (largest first).
fn rank(s: Vec2) -> usize {
    if s.x < 1e-4 {
        0
    } else if s.y < 1e-4 * s.x.max(1.0) {
        1
    } else {
        2
    }
}

/// Where the area is written: in the middle of the square's image, unless v, Av or
/// an edge runs through it there; then at the roomiest spot inside. `wpp` is world
/// units per pixel.
fn area_label_at(i_hat: Vec2, j_hat: Vec2, v: Vec2, av: Vec2, wpp: f32) -> Vec2 {
    let half = Vec2::X * (34.0 * wpp);
    let lines = [(Vec2::ZERO, v), (Vec2::ZERO, av), (Vec2::ZERO, i_hat), (i_hat, i_hat + j_hat), (i_hat + j_hat, j_hat), (j_hat, Vec2::ZERO)];
    let room = |p: Vec2| [p - half, p, p + half].iter().flat_map(|q| lines.iter().map(move |(s, e)| seg_dist(*q, *s, *e))).fold(f32::MAX, f32::min);
    let mut at = (i_hat + j_hat) * 0.5;
    if room(at) < 14.0 * wpp {
        for s in [0.2, 0.35, 0.5, 0.65, 0.8] {
            for t in [0.2, 0.35, 0.5, 0.65, 0.8] {
                let p = i_hat * s + j_hat * t;
                if room(p) > room(at) {
                    at = p;
                }
            }
        }
    }
    at
}

impl Scene for LinearMap2 {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 4.2 }
    }

    fn animated(&self) -> bool {
        self.playing
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A $2 \times 2$ matrix $A$ moves every point of the plane, and its two columns say where: the first is where the arrow $\textcolor{pink}{\hat\imath} = (1, 0)$ lands, the second where $\textcolor{green}{\hat\jmath} = (0, 1)$ lands. Every other point follows, since $(x, y) = x\hat\imath + y\hat\jmath$, and so lands on $x$ times the first column plus $y$ times the second:");
        ui::equation(ui, r"A \begin{bmatrix} x \\ y \end{bmatrix} = x \, \textcolor{pink}{A\hat\imath} + y \, \textcolor{green}{A\hat\jmath}");
        ui::note(ui, r"Drag the tips of $\hat\imath$ and $\hat\jmath$, or the white dot $v$.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Matrix");
        mat2_edit(ui, &mut self.m);
        ui::note(ui, r"First column: where $\hat\imath$ lands. Second column: where $\hat\jmath$ lands. Drag a number sideways, or click it and type.");
        ui::section(ui, "Animation");
        if ui.button("replay").clicked() {
            self.morph = 0.0;
            self.playing = true;
        }
        ui::slider(ui, "progress", &mut self.morph, 0.0..=1.0);
        ui::note(ui, r"From the plane as it was (0) to the plane after the map (1). The tips of $\hat\imath$ and $\hat\jmath$ can be dragged at 1.");
        ui::section(ui, "Show");
        ui.checkbox(&mut self.show_area, "determinant as area");
        ui.checkbox(&mut self.show_eigen, "eigenvectors");
        ui.checkbox(&mut self.show_circle, "unit circle and singular vectors");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::BLUE, "blue grid: the square grid, after the map"),
                (pal::PINK, r"$\hat\imath$: where $(1, 0)$ lands, the first column of $A$"),
                (pal::GREEN, r"$\hat\jmath$: where $(0, 1)$ lands, the second column"),
                (pal::YELLOW, "yellow: the unit square after the map; its area is the determinant"),
                (pal::PINK, "pink instead: the plane has been flipped over, and the determinant is negative"),
                (pal::WHITE, r"the white dot $v$, and the white arrow $Av$: where the matrix sends it"),
                (pal::ORANGE, r"orange and purple, dashed: the eigenvector lines; the arrow on each is stretched by its $\lambda$"),
                (pal::WHITE, "white ellipse: where the unit circle (faint) lands"),
                (pal::ORANGE, r"orange and purple, solid, with the ellipse: its longest and shortest half-axes, $\sigma_1$ and $\sigma_2$"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Determinant:", "the area the unit square becomes, counted negative if the plane is flipped over. Every area is scaled by it.");
        ui::term(ui, "Eigenvector:", r"a direction the map does not turn: a vector $v$ along it is only stretched, $Av = \lambda v$, by a factor $\lambda$, its eigenvalue (negative: reversed).");
        ui::term(ui, "Trace:", r"the sum of the diagonal entries. It is the sum of the eigenvalues, $\tr A = \lambda_1 + \lambda_2$, and the determinant is their product, $\det A = \lambda_1 \lambda_2$.");
        ui::term(ui, "Singular values:", r"the most and the least the map stretches any direction: the half-axes $\sigma_1$ and $\sigma_2$ of the ellipse the unit circle becomes.");
        ui::term(ui, "Rank:", "how many dimensions the plane keeps: 2, or 1 if it is squashed onto a line, 0 if onto a single point.");

        ui::section(ui, "Numbers");
        let m = self.m;
        let det = m.determinant();
        ui::readout(ui, r"$\det A$ (area scale)", format!("${}$", fmt(det)));
        ui::readout(ui, r"$\tr A$", format!("${}$", fmt(m.x_axis.x + m.y_axis.y)));
        match eig2(m) {
            Eig2::Real { l1, l2, .. } => ui::readout(ui, r"eigenvalues $\lambda_1, \lambda_2$", format!("${}, {}$", fmt(l1), fmt(l2))),
            Eig2::Complex { re, im } => ui::readout(ui, r"eigenvalues $\lambda_1, \lambda_2$", format!(r"${} \pm {}i$", fmt(re), fmt(im))),
        }
        let s = svd2(m).s;
        ui::readout(ui, r"singular values $\sigma_1, \sigma_2$", format!("${}, {}$", fmt(s.x), fmt(s.y)));
        ui::readout(ui, r"$\rank A$", format!("${}$", rank(s)));
        if det < 0.0 {
            ui::note(ui, "Negative determinant: the plane has been flipped over.");
        }
        if let Eig2::Complex { .. } = eig2(m) {
            ui::note(ui, "Complex eigenvalues: no line through $0$ stays on itself; the map turns every direction.");
        }
        ui::see_also(ui, &[("linmap3", "the same in three dimensions"), ("compose", "one map after another: products and inverses"), ("svd", "why the circle always becomes an ellipse"), ("fold", "what a network's layers do with maps like these")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt);
        let m = self.shown();

        warped_grid(f, m, 9, pal::BLUE);

        if self.show_area {
            let det = m.determinant();
            let c = area_colour(det);
            let [o, a, ab, b] = square_image(m).map(|p| p.extend(0.0));
            f.dl.tris_alpha.quad(o, a, ab, b, c.a(0.22));
            f.dl.lines.closed(&[o, a, ab, b], 1.5, c.a(0.8));
            // The number belongs in the middle, unless v or Av runs through it there:
            // then it moves to the roomiest spot inside the parallelogram.
            let wpp = f.cam.world_per_px(Vec3::ZERO) * f.px;
            let at = area_label_at(m.x_axis, m.y_axis, self.v, m * self.v, wpp);
            // (when the roomiest spot changes, the number glides to it from where it
            // was, measured from the middle of the parallelogram)
            let middle = (m.x_axis + m.y_axis) * 0.5;
            if let (Some(from), Some(to)) = (on_screen(f, middle.extend(0.0)), on_screen(f, at.extend(0.0))) {
                f.text_placed("area", 0, from, to - from, format!("area ${}$", fmt(det)), 13.0, c);
            }
        }

        if self.show_circle {
            let pts: Vec<Vec3> = (0..128).map(|i| Vec2::from_angle(i as f32 / 128.0 * TAU)).map(|p| (m * p).extend(0.0)).collect();
            f.dl.lines.closed(&pts, 2.4, pal::WHITE);
            f.circle(Vec2::ZERO, Vec3::Z, 1.0, 1.0, pal::DIM.a(0.6));
            let d = svd2(m);
            for (k, c) in [pal::ORANGE, pal::PURPLE].into_iter().enumerate() {
                let u = d.u.col(k) * d.s[k];
                f.line(-u, u, 2.0, c);
                f.text_off(u, Vec2::new(8.0, -8.0), format!(r"$\sigma_{{{}}} = {}$", k + 1, fmt(d.s[k])), 13.0, c, Align2::LEFT_BOTTOM);
            }
        }

        if self.show_eigen {
            for ((l, v, tip), c) in self.eigen_arrows().into_iter().zip([pal::ORANGE, pal::PURPLE]) {
                f.dashed(-v * 20.0, v * 20.0, 1.6, c.a(0.85));
                f.arrow(Vec2::ZERO, tip, 2.4, c);
                f.text_off(tip, Vec2::new(8.0, 10.0), format!(r"$\lambda = {}$", fmt(l)), 13.0, c, Align2::LEFT_TOP);
            }
        }

        // the test vector and its image
        let mut v = self.v;
        if f.drag2(3, &mut v, pal::WHITE) {
            self.v = v;
        }
        f.dashed(Vec2::ZERO, self.v, 1.5, pal::DIM);
        f.arrow(Vec2::ZERO, m * self.v, 2.6, pal::WHITE);
        f.text_off(self.v, Vec2::new(16.0, 0.0), "$v$", 15.0, pal::TEXT, Align2::LEFT_CENTER);
        f.text_off(m * self.v, Vec2::new(10.0, -8.0), "$Av$", 15.0, pal::WHITE, Align2::LEFT_BOTTOM);

        // basis vectors: the handles are the columns of the matrix
        let (mut i_hat, mut j_hat) = (m.x_axis, m.y_axis);
        f.arrow(Vec2::ZERO, i_hat, 3.4, pal::PINK);
        f.arrow(Vec2::ZERO, j_hat, 3.4, pal::GREEN);
        if self.morph >= 1.0 {
            if f.drag2(1, &mut i_hat, pal::PINK) {
                self.m.x_axis = i_hat;
            }
            if f.drag2(2, &mut j_hat, pal::GREEN) {
                self.m.y_axis = j_hat;
            }
        }
        f.text_off(m.x_axis, Vec2::new(10.0, 12.0), r"$\hat\imath$", 17.0, pal::PINK, Align2::LEFT_TOP);
        f.text_off(m.y_axis, Vec2::new(-10.0, -10.0), r"$\hat\jmath$", 17.0, pal::GREEN, Align2::RIGHT_BOTTOM);
        f.axes2();
        bottom_caption(f, &["blue grid: the square grid, after the map"], 12.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, played to the end at 60 frames a second.
    fn run(k: usize) -> LinearMap2 {
        let mut scene = LinearMap2::create();
        scene.set_up(k);
        for _ in 0..1000 {
            scene.advance(1.0 / 60.0);
        }
        assert!(!scene.playing && scene.morph == 1.0);
        scene
    }

    /// The signed area of a polygon, positive if its corners run anticlockwise.
    fn area(corners: &[Vec2]) -> f32 {
        (0..corners.len()).map(|i| corners[i].perp_dot(corners[(i + 1) % corners.len()])).sum::<f32>() * 0.5
    }

    /// Unit vectors at every degree of a half turn, but for the ends.
    fn directions() -> impl Iterator<Item = Vec2> {
        (1..180).map(|d| Vec2::from_angle((d as f32).to_radians()))
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // a shear: the x axis stays put, the line at height y slides by y, the area is
        // kept, and the x axis is the only line left on itself
        let shear = run(0);
        let m = shear.shown();
        assert_eq!(m * Vec2::X, Vec2::X);
        for (x, y) in [(0.0, 0.5), (-1.0, 1.0), (2.0, 2.0), (0.5, -1.5)] {
            assert!((m * Vec2::new(x, y) - Vec2::new(x + y, y)).length() < 1e-6);
        }
        assert!((area(&square_image(m)) - 1.0).abs() < 1e-6 && fmt(shear.m.determinant()) == "1.000");
        let lines = shear.eigen_arrows();
        assert!(!lines.is_empty() && lines.iter().all(|(l, v, _)| (l - 1.0).abs() < 1e-6 && v.y.abs() < 1e-6), "{lines:?}");
        assert!(directions().all(|d| d.perp_dot(m * d).abs() > 1e-4), "another line stays on itself");

        // a rotation: every arrow turns by 45 degrees and keeps its length; no
        // eigenvector lines, complex eigenvalues, and the area is kept
        let rotation = run(1);
        let m = rotation.shown();
        for d in directions() {
            assert!(((m * d).length() - 1.0).abs() < 1e-6 && (d.angle_to(m * d) - PI / 4.0).abs() < 1e-5);
        }
        assert!(rotation.eigen_arrows().is_empty() && matches!(eig2(rotation.m), Eig2::Complex { .. }));
        assert!((area(&square_image(m)) - 1.0).abs() < 1e-6 && fmt(rotation.m.determinant()) == "1.000");

        // a reflection: x and y swap, the square is pink with area -1, and the two
        // diagonals are the eigenvector lines, with 1 and -1
        let reflection = run(2);
        let m = reflection.shown();
        assert_eq!(m * Vec2::new(0.3, 1.7), Vec2::new(1.7, 0.3));
        let det = reflection.m.determinant();
        assert!((area(&square_image(m)) + 1.0).abs() < 1e-6 && det == -1.0 && area_colour(det) == pal::PINK);
        let lines = reflection.eigen_arrows();
        let along = |l: f32, d: Vec2| lines.iter().any(|(k, v, _)| (k - l).abs() < 1e-6 && v.perp_dot(d.normalize()).abs() < 1e-6);
        assert!(lines.len() == 2 && along(1.0, Vec2::new(1.0, 1.0)) && along(-1.0, Vec2::new(1.0, -1.0)), "{lines:?}");

        // squash onto a line: points drop at right angles onto the line through (2, 1);
        // those on it stay put, those across it go to 0; the square and the circle
        // go flat; det = 0, sigma2 = 0, rank 1
        let squash = run(3);
        let m = squash.shown();
        let (on, across) = (Vec2::new(2.0, 1.0), Vec2::new(-1.0, 2.0));
        assert!((m * on - on).length() < 1e-6 && (m * across).length() < 1e-6);
        for d in directions() {
            let p = d * 1.7;
            assert!((m * p - p).dot(on).abs() < 1e-5 && (m * p).perp_dot(on).abs() < 1e-5);
        }
        assert!(area(&square_image(m)).abs() < 1e-7 && fmt(squash.m.determinant()) == "0");
        let s = svd2(squash.m).s;
        assert!(squash.show_circle && s.y < 1e-6 && rank(s) == 1, "{s}");
        let lines = squash.eigen_arrows();
        let along = |l: f32, d: Vec2| lines.iter().any(|(k, v, _)| (k - l).abs() < 1e-6 && v.perp_dot(d.normalize()).abs() < 1e-6);
        assert!(along(1.0, on) && along(0.0, across), "{lines:?}");

        // the other experiments keep the full rank, and the notes under the numbers
        // appear where they should
        for k in 0..3 {
            assert_eq!(rank(svd2(run(k).m).s), 2);
        }
        assert!(run(2).m.determinant() < 0.0 && (0..4).filter(|&k| k != 2).all(|k| run(k).m.determinant() >= 0.0));
    }

    /// Every dashed line is mapped onto itself, at every stage of the animation, and
    /// the arrow on it is where the matrix on screen sends the unit vector along it.
    /// "Trace" and "determinant" are the sum and the product of the eigenvalues.
    #[test]
    fn the_eigenvector_lines_drawn_are_mapped_onto_themselves() {
        let mut rng = Rng::new(7);
        let mut matrices: Vec<Mat2> = TRIALS.iter().map(|t| Mat2::from_cols(Vec2::new(t.2[0], t.2[1]), Vec2::new(t.2[2], t.2[3]))).collect();
        matrices.push(LinearMap2::create().m);
        matrices.extend((0..200).map(|_| Mat2::from_cols(Vec2::new(rng.normal(), rng.normal()), Vec2::new(rng.normal(), rng.normal())) * 1.5));
        let (mut real, mut complex) = (0, 0);
        for m in matrices {
            let mut scene = LinearMap2::create();
            scene.m = m;
            let size = m.to_cols_array().iter().fold(0.0f32, |s, v| s.max(v.abs())).max(1.0);
            for morph in [0.0, 0.3, 0.7, 1.0] {
                scene.morph = morph;
                let shown = scene.shown();
                for (l, v, tip) in scene.eigen_arrows() {
                    assert!((v.length() - 1.0).abs() < 1e-5 && (shown * v - tip).length() < 2e-4 * size, "{m}: λ = {l} along {v} at {morph}");
                    assert!((m * v - v * l).length() < 2e-4 * size, "{m}: λ = {l} along {v}");
                }
            }
            let (trace, det) = (m.x_axis.x + m.y_axis.y, m.determinant());
            match eig2(m) {
                Eig2::Real { l1, l2, .. } => {
                    real += 1;
                    assert!((l1 + l2 - trace).abs() < 1e-4 * size && (l1 * l2 - det).abs() < 1e-3 * size * size, "{m}");
                }
                Eig2::Complex { re, im } => {
                    complex += 1;
                    assert!((2.0 * re - trace).abs() < 1e-4 * size && (re * re + im * im - det).abs() < 1e-3 * size * size, "{m}");
                    // "no line through 0 stays on itself; the map turns every direction"
                    let turns: Vec<f32> = directions().chain([Vec2::X]).map(|d| d.perp_dot(m * d)).collect();
                    assert!(turns.iter().all(|t| *t > 0.0) || turns.iter().all(|t| *t < 0.0), "{m}");
                }
            }
        }
        assert!(real > 50 && complex > 50, "{real} real, {complex} complex");
    }

    /// The columns on top of each other, a column at the origin, both there: every
    /// number drawn or read out is finite, and the rank says what happened.
    #[test]
    fn degenerate_handles_leave_everything_finite() {
        let cases = [
            (Mat2::from_cols(Vec2::new(1.0, 0.5), Vec2::new(2.0, 1.0)), 1),
            (Mat2::from_cols(Vec2::new(1.0, 0.5), Vec2::new(-1.0, -0.5)), 1),
            (Mat2::from_cols(Vec2::ZERO, Vec2::new(-0.5, 1.2)), 1),
            (Mat2::from_cols(Vec2::new(1.5, 0.5), Vec2::ZERO), 1),
            (Mat2::ZERO, 0),
            (Mat2::from_cols(Vec2::new(1e-7, 0.0), Vec2::new(0.0, -1e-7)), 0),
        ];
        for (m, want) in cases {
            for v in [Vec2::new(1.0, 1.5), Vec2::ZERO] {
                let mut scene = LinearMap2::create();
                (scene.m, scene.v) = (m, v);
                for morph in [0.0, 0.5, 1.0] {
                    scene.morph = morph;
                    let shown = scene.shown();
                    assert!(shown.is_finite() && square_image(shown).iter().all(|p| p.is_finite()));
                    assert!(scene.eigen_arrows().iter().all(|(l, v, tip)| l.is_finite() && v.is_finite() && tip.is_finite()), "{m}");
                    assert!(area_label_at(shown.x_axis, shown.y_axis, v, shown * v, 0.01).is_finite(), "{m}");
                    let d = svd2(shown);
                    assert!(d.u.is_finite() && d.s.is_finite());
                }
            }
            let s = svd2(m).s;
            assert_eq!(rank(s), want, "{m}: singular values {s}");
            assert!(!fmt(m.determinant()).contains("NaN") && !format!("{:?}", eig2(m)).contains("NaN"));
        }
    }
}

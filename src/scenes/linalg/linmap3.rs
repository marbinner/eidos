//! Linear maps in 3D.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "linmap3",
    title: "Linear maps in 3D",
    group: "Linear algebra: vectors and maps",
    blurb: r"A $3 \times 3$ matrix reshapes the unit cube; its volume is the determinant.",
    make: |g| Box::new(LinearMap3::new(g)),
};

/// Experiments: (button, what to look for). The matrix each one sets is `trial_matrix`.
const TRIALS: [(&str, &str); 4] = [
    (
        "a rotation",
        r"The cube turns by 63° about the line through $(1, 1, 1)$ and keeps its shape: the volume ($\det A$) is 1, and so are all three singular values. That line is the only one that stays on itself ($\lambda = 1$).",
    ),
    (
        "a reflection",
        r"Space is mirrored in the floor. The box turns pink, as it does whenever $\det A$ is negative; here $\det A = -1$. The upright line is reversed ($\lambda = -1$); lines in the floor stay as they are ($\lambda = 1$: one of them is drawn).",
    ),
    (
        "squash onto a plane",
        r"The third column is the sum of the other two, so all of space lands on one plane: the cube is flattened, and the volume ($\det A$) and the smallest singular value are 0. A whole line of points lands on $0$: it is the dashed line with $\lambda = 0$.",
    ),
    ("squash onto a line", "All three columns lie along one line, so all of space lands on it: the cube becomes a stick, and two of the three singular values are 0. The rank is 1."),
];

/// The matrix of experiment `k`.
fn trial_matrix(k: usize) -> Mat3 {
    match k {
        0 => Mat3::from_axis_angle(Vec3::new(1.0, 1.0, 1.0).normalize(), 1.1),
        1 => Mat3::from_diagonal(Vec3::new(1.0, 1.0, -1.0)),
        2 => Mat3::from_cols(Vec3::new(1.0, 0.0, 0.5), Vec3::new(0.0, 1.0, 0.5), Vec3::new(1.0, 1.0, 1.0)),
        _ => Mat3::from_cols(Vec3::new(1.0, 0.5, 0.8), Vec3::new(0.6, 0.3, 0.48), Vec3::new(-0.8, -0.4, -0.64)),
    }
}

/// The colour of the solid the unit cube becomes, and of it turned inside out.
const SOLID: Color = Color::hex(0x3C6BD6);
const FLIPPED: Color = Color::hex(0xC0407A);

pub struct LinearMap3 {
    m: Mat3,
    morph: f32,
    playing: bool,
    show_eigen: bool,
    show_lattice: bool,
}

impl LinearMap3 {
    pub fn new(_: &Gfx) -> Self {
        Self::create()
    }

    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        LinearMap3 {
            m: Mat3::from_cols(Vec3::new(1.2, 0.3, 0.0), Vec3::new(-0.4, 1.0, 0.3), Vec3::new(0.2, 0.2, 1.4)),
            morph: 0.0,
            playing: true,
            show_eigen: true,
            show_lattice: true,
        }
    }

    /// Set up one of the experiments in `TRIALS`, and play it from the identity.
    fn set_up(&mut self, k: usize) {
        self.m = trial_matrix(k);
        (self.show_eigen, self.show_lattice) = (true, true);
        (self.morph, self.playing) = (0.0, true);
    }

    /// Move the animation on by `dt` seconds of scene time.
    fn advance(&mut self, dt: f32) {
        if self.playing {
            self.morph += dt / 2.0;
            if self.morph >= 1.0 {
                self.morph = 1.0;
                self.playing = false;
            }
        }
    }

    /// The matrix as drawn: on its way from the identity to `m` while the animation plays.
    fn shown(&self) -> Mat3 {
        let (g, t) = (self.m, smoothstep(self.morph));
        Mat3::from_cols(Vec3::X.lerp(g.x_axis, t), Vec3::Y.lerp(g.y_axis, t), Vec3::Z.lerp(g.z_axis, t))
    }

    /// The real eigenvector lines, as drawn: (λ, unit direction, tip of the arrow on
    /// it, which is where the matrix on screen sends that unit vector).
    fn eigen_arrows(&self) -> Vec<(f32, Vec3, Vec3)> {
        let t = smoothstep(self.morph);
        eig3_real(self.m).into_iter().map(|(l, v)| (l, v, v * (1.0 + (l - 1.0) * t))).collect()
    }
}

/// How many dimensions space keeps, from the singular values (largest first).
fn rank(s: Vec3) -> usize {
    (0..3).filter(|&k| s[k] > 1e-3 * s.x.max(1e-6)).count()
}

/// The points carried along by the map: a 5×5×5 grid, each with the colour that
/// says where it started.
fn lattice() -> impl Iterator<Item = (Vec3, Color)> {
    (0..125).map(|n| {
        let (i, j, k) = (n / 25 % 5, n / 5 % 5, n % 5);
        let p = Vec3::new(i as f32 - 2.0, j as f32 - 2.0, k as f32 - 2.0);
        (p, Color::linear(0.25 + 0.18 * i as f32, 0.25 + 0.18 * j as f32, 0.3 + 0.17 * k as f32))
    })
}

impl Scene for LinearMap3 {
    fn view(&self) -> View {
        View::Orbit { target: Vec3::new(0.3, 0.3, 0.5), yaw: -0.35, pitch: 0.42, dist: 12.5 }
    }

    fn animated(&self) -> bool {
        self.playing
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        let [e1, e2, e3] = [0, 1, 2].map(|k| tint(pal::AXIS[k], &format!("e_{}", k + 1)));
        ui::note(ui, &format!(r"A $3 \times 3$ matrix $A$ moves every point of space. Its three columns say where: they are where the arrows ${e1}, {e2}, {e3}$ (one step along each axis) land, and every other point follows. The unit cube they span becomes a slanted box whose volume is the determinant. Drag the arrow tips."));

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Matrix");
        let mut a = self.m.transpose().to_cols_array_2d();
        let mut changed = false;
        egui::Grid::new("m3").spacing([6.0, 4.0]).show(ui, |ui| {
            for row in &mut a {
                for v in row.iter_mut() {
                    changed |= entry(ui, v).changed();
                }
                ui.end_row();
            }
        });
        if changed {
            self.m = Mat3::from_cols_array_2d(&a).transpose();
        }
        ui::note(ui, "Each column is where one arrow lands: the first is $e_1$, the second $e_2$, the third $e_3$. Drag a number sideways, or click it and type.");
        ui::section(ui, "Animation");
        if ui.button("replay").clicked() {
            self.morph = 0.0;
            self.playing = true;
        }
        ui::slider(ui, "progress", &mut self.morph, 0.0..=1.0);
        ui::note(ui, "From space as it was (0) to space after the map (1). The arrow tips can be dragged at 1.");
        ui::section(ui, "Show");
        ui.checkbox(&mut self.show_lattice, "lattice of points");
        ui.checkbox(&mut self.show_eigen, "real eigenvectors");

        ui::section(ui, "Reading the picture");
        let columns = format!("${e1}, {e2}, {e3}$ (red, green, blue arrows): where one step along each axis lands, the columns");
        ui::key(
            ui,
            &[
                (pal::AXIS[0], columns.as_str()),
                (SOLID, "blue box: the unit cube after the map; its volume is the determinant"),
                (FLIPPED, "pink box instead: space has been turned inside out, and the determinant is negative"),
                (pal::DIM, "faint wire cube: the unit cube before the map"),
                (Color::linear(0.61, 0.61, 0.64), r"dots: a $5 \times 5 \times 5$ grid of points carried along by the map; the colour says where each one started"),
                (pal::ORANGE, r"dashed lines (orange, purple, yellow): the real eigenvector lines; points on them stay on them, stretched by $\lambda$"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Determinant:", "the volume the unit cube becomes, counted negative if space is turned inside out. Every volume is scaled by it.");
        ui::term(ui, "Eigenvector:", r"a direction the map does not turn: a vector $v$ along it is only stretched, $Av = \lambda v$, by a factor $\lambda$, its eigenvalue.");
        ui::term(ui, "Singular values:", "the stretches along three directions at right angles that the map sends to three directions at right angles, largest to smallest: the half-axes of the stretched ball (an ellipsoid) that the unit ball becomes. The largest and the smallest are the most and the least the map stretches any direction.");
        ui::term(ui, "Rank:", "how many dimensions space keeps: 3, or 2 if it is squashed onto a plane, 1 if onto a line, 0 if onto a single point.");

        ui::section(ui, "Numbers");
        let det = self.m.determinant();
        ui::readout(ui, r"$\det A$ (volume scale)", format!("${}$", fmt(det)));
        let (_, s, _) = crate::math::linalg::svd3(self.m);
        ui::readout(ui, r"singular values $\sigma_1, \sigma_2, \sigma_3$", format!("${}, {}, {}$", fmt(s.x), fmt(s.y), fmt(s.z)));
        let rank = rank(s);
        ui::readout(ui, r"$\rank A$", format!("${rank}$"));
        let eig: Vec<String> = eig3_real(self.m).iter().map(|e| fmt(e.0)).collect();
        // (one at least, for a real 3 × 3 matrix; an empty formula would not parse)
        ui::readout(ui, "real eigenvalues", if eig.is_empty() { String::new() } else { format!("${}$", eig.join(", ")) });
        match rank {
            2 => ui::note(ui, "Rank 2: all of space is squashed onto a plane, so the volume is zero."),
            1 => ui::note(ui, "Rank 1: all of space is squashed onto a line."),
            _ => {}
        }
        ui::see_also(ui, &[("linmap2", "the same in the plane"), ("rotations", "the maps that only turn space"), ("subspaces", "what a matrix that squashes space ignores and reaches"), ("fold", "what a network's layers do to space")]);
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.10;
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt);
        let m = self.shown();
        let det = m.determinant();

        // reference unit cube and its image
        let cube = |f: &mut Frame, m: Mat3, w: f32, c: Color| {
            for (a, b) in CUBE_EDGES {
                f.line(m * Vec3::from(a), m * Vec3::from(b), w, c);
            }
        };
        cube(f, Mat3::IDENTITY, 1.0, pal::DIM.a(0.5));
        let body = if det >= 0.0 { SOLID } else { FLIPPED };
        shaded_solid(f, m.x_axis, m.y_axis, m.z_axis, body);
        cube(f, m, 1.8, pal::WHITE.a(0.9));

        if self.show_lattice {
            for (p, c) in lattice() {
                f.sphere(m * p, 0.085, c);
            }
        }

        // Everything a label must keep clear of, as segments on screen: the edges of
        // the solid (three of them are the basis arrows) and the eigenvector arrows.
        let mut busy: Vec<(Vec2, Vec2)> = Vec::new();
        let mut keep_clear = |f: &Frame, a: Vec3, b: Vec3| {
            if let (Some(a), Some(b)) = (on_screen(f, a), on_screen(f, b)) {
                busy.push((a, b));
            }
        };
        for (a, b) in CUBE_EDGES {
            keep_clear(f, m * Vec3::from(a), m * Vec3::from(b));
        }

        // (each with a name and a number of its own, by which its place is kept
        // from frame to frame: see `label_beside`)
        let mut names: Vec<(Name, Vec3, String, f32, Color)> = Vec::new();
        if self.show_eigen {
            for (i, (l, v, tip)) in self.eigen_arrows().into_iter().enumerate() {
                let c = [pal::ORANGE, pal::PURPLE, pal::YELLOW][i % 3];
                f.dashed(-v * 6.0, v * 6.0, 1.5, c.a(0.8));
                f.arrow(Vec3::ZERO, tip, 2.2, c);
                keep_clear(f, Vec3::ZERO, tip);
                names.push((("eigen", i as u32), tip * 1.02, format!(r"$\lambda = {}$", fmt(l)), 13.0, c));
            }
        }

        let mut cols = [m.x_axis, m.y_axis, m.z_axis];
        for k in 0..3 {
            f.arrow(Vec3::ZERO, cols[k], 3.4, pal::AXIS[k]);
            if self.morph >= 1.0 && f.drag3(10 + k as u32, &mut cols[k], None, pal::AXIS[k]) {
                *self.m.col_mut(k) = cols[k];
            }
            names.push((("column", k as u32), cols[k], ["$e_1$", "$e_2$", "$e_3$"][k].to_string(), 16.0, pal::AXIS[k]));
        }
        // the volume is written in the middle of the solid, and the names keep clear of it too
        let centre = m * Vec3::splat(0.5);
        let volume = format!("volume ${}$", fmt(det));
        if let Some(at) = on_screen(f, centre) {
            let half = Vec2::X * (width_of(f, &volume, 13.0) * 0.5);
            busy.push((at - half, at + half));
        }
        // the basis vectors are named first, so that they get the better places
        names.rotate_right(3);
        for (name, tip, text, size, color) in names {
            label_beside(f, name, tip, text, size, color, &mut busy);
        }
        f.text_off(centre, Vec2::new(0.0, 0.0), volume, 13.0, pal::WHITE, Align2::CENTER_CENTER);
        bottom_caption(f, &["dots: a grid of points, carried along by the map", "faint wire cube: the unit cube before it"], 12.0);
    }
}

/// The twelve edges of the unit cube, as pairs of corners.
const CUBE_EDGES: [([f32; 3], [f32; 3]); 12] = [
    ([0., 0., 0.], [1., 0., 0.]),
    ([0., 1., 0.], [1., 1., 0.]),
    ([0., 0., 1.], [1., 0., 1.]),
    ([0., 1., 1.], [1., 1., 1.]),
    ([0., 0., 0.], [0., 1., 0.]),
    ([1., 0., 0.], [1., 1., 0.]),
    ([0., 0., 1.], [0., 1., 1.]),
    ([1., 0., 1.], [1., 1., 1.]),
    ([0., 0., 0.], [0., 0., 1.]),
    ([1., 0., 0.], [1., 0., 1.]),
    ([0., 1., 0.], [0., 1., 1.]),
    ([1., 1., 0.], [1., 1., 1.]),
];

/// The solid spanned by `a`, `b`, `c` from the origin. Each pair of opposite faces has
/// its own tone, as if lit from above and to one side, so that the shape reads as a
/// solid; the faces are drawn back to front because they are translucent.
fn shaded_solid(f: &mut Frame, a: Vec3, b: Vec3, c: Vec3, body: Color) {
    let light = Vec3::new(0.55, -0.25, 0.8).normalize();
    let eye = f.cam.eye;
    let mut faces = [(Vec3::ZERO, a, b), (c, a, b), (Vec3::ZERO, a, c), (b, a, c), (Vec3::ZERO, b, c), (a, b, c)];
    let far = |(o, u, v): &(Vec3, Vec3, Vec3)| (*o + (*u + *v) * 0.5).distance(eye);
    faces.sort_by(|p, q| far(q).total_cmp(&far(p)));
    for (o, u, v) in faces {
        let tone = 0.3 + u.cross(v).normalize_or_zero().dot(light).abs().powf(1.5);
        f.dl.tris_alpha.quad(o, o + u, o + u + v, o + v, Color::from_linear(body.to_linear() * tone).a(0.55));
    }
}

/// Name the point `tip` on whichever of its four diagonal sides is clearest of the
/// segments in `busy` (screen pixels); up and to the right unless another side is
/// clearly better. The label then joins `busy`, so that later ones avoid it too.
///
/// The side it was on in the last frame counts as roomier than it is (`name` is what
/// its place is remembered by): while the picture turns, two sides are often about
/// as clear as each other, and a label choosing afresh each frame flickers between
/// them. When it does change sides it is drawn gliding across (`Frame::text_placed`).
fn label_beside(f: &mut Frame, name: Name, tip: Vec3, text: String, size: f32, color: Color, busy: &mut Vec<(Vec2, Vec2)>) {
    let Some(at) = on_screen(f, tip) else { return };
    let half = Vec2::new(width_of(f, &text, size) * 0.5, size * 0.5);
    let was = f.places.chosen(name.0, name.1).map(|offset| at + offset);
    let mut best = (f32::MIN, Vec2::ZERO);
    for side in [Vec2::new(1.0, -1.0), Vec2::new(-1.0, -1.0), Vec2::new(1.0, 1.0), Vec2::new(-1.0, 1.0)] {
        let centre = at + side * 10.0 + side * half;
        let ends = [centre - Vec2::X * half.x, centre, centre + Vec2::X * half.x];
        let mut room = ends.iter().flat_map(|q| busy.iter().map(move |(a, b)| seg_dist(*q, *a, *b))).fold(f32::MAX, f32::min);
        if was.is_some_and(|was| (was - centre).length() < 0.5 * size) {
            room += STAYS;
        }
        if room > best.0 + 4.0 {
            best = (room, centre);
        }
    }
    busy.push((best.1 - Vec2::X * half.x, best.1 + Vec2::X * half.x));
    f.text_placed(name.0, name.1, at, best.1 - at, text, size, color);
}

/// How wide `text` is set, measured by a stand-in with every digit a 0: digits are all
/// as wide as one another, and the stand-in stays the same while the numbers in the
/// label change as the map plays (the frame keeps what it has measured, but a text it
/// has not seen is set afresh).
fn width_of(f: &Frame, text: &str, size: f32) -> f32 {
    let stand_in: String = text.chars().map(|c| if c.is_ascii_digit() { '0' } else { c }).collect();
    f.text_size(&stand_in, size).x
}

/// What a label's place is remembered by: a kind and a number.
type Name = (&'static str, u32);

/// How much clearer than it is the side a label is already on counts as (pixels).
const STAYS: f32 = 10.0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::linalg::svd3;

    /// An experiment as its button sets it up, played to the end at 60 frames a second.
    fn run(k: usize) -> LinearMap3 {
        let mut scene = LinearMap3::create();
        scene.set_up(k);
        for _ in 0..1000 {
            scene.advance(1.0 / 60.0);
        }
        assert!(!scene.playing && scene.morph == 1.0);
        scene
    }

    /// The signed volume of the box spanned by the columns, as the solid is drawn.
    fn volume(m: Mat3) -> f32 {
        m.x_axis.dot(m.y_axis.cross(m.z_axis))
    }

    /// The captions of the "things to try", and the notes on rank, held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // a rotation by 63 degrees about (1, 1, 1): lengths kept, det and the singular
        // values 1, and that axis the only eigenvector line, with 1
        let rotation = run(0);
        let m = rotation.shown();
        assert!(lattice().all(|(p, _)| ((m * p).length() - p.length()).abs() < 1e-5));
        let degrees = ((m.x_axis.x + m.y_axis.y + m.z_axis.z - 1.0) * 0.5).acos().to_degrees();
        assert!((degrees - 63.0).abs() < 0.5, "{degrees}");
        let (_, s, _) = svd3(rotation.m);
        assert!((s - Vec3::ONE).abs().max_element() < 1e-5 && fmt(volume(m)) == "1.000" && rank(s) == 3, "{s}");
        let lines = rotation.eigen_arrows();
        assert!(lines.len() == 1 && (lines[0].0 - 1.0).abs() < 1e-5 && lines[0].1.cross(Vec3::ONE.normalize()).length() < 1e-5, "{lines:?}");

        // a reflection in the floor: pink, det -1; the upright line reversed, a line
        // in the floor kept, and those two lines drawn
        let reflection = run(1);
        let m = reflection.shown();
        assert_eq!(m * Vec3::new(0.3, -1.2, 0.7), Vec3::new(0.3, -1.2, -0.7));
        assert!(volume(m) == -1.0 && reflection.m.determinant() == -1.0);
        let lines = reflection.eigen_arrows();
        let upright = lines.iter().any(|(l, v, _)| (l + 1.0).abs() < 1e-6 && v.cross(Vec3::Z).length() < 1e-6);
        let floor = lines.iter().any(|(l, v, _)| (l - 1.0).abs() < 1e-6 && v.z.abs() < 1e-6);
        assert!(lines.len() == 2 && upright && floor, "{lines:?}");

        // squash onto a plane: the third column is the sum of the others; every point
        // lands on one plane; det and the smallest singular value 0; rank 2; a line
        // of points lands on 0, and it is an eigenvector line with 0
        let plane = run(2);
        let (g, m) = (plane.m, plane.shown());
        assert_eq!(g.z_axis, g.x_axis + g.y_axis);
        let normal = g.x_axis.cross(g.y_axis).normalize();
        assert!(lattice().all(|(p, _)| (m * p).dot(normal).abs() < 1e-5));
        let (_, s, _) = svd3(g);
        assert!(volume(m).abs() < 1e-6 && fmt(g.determinant()) == "0" && s.z < 1e-5 * s.x && rank(s) == 2, "{s}");
        let lines = plane.eigen_arrows();
        assert!(lines.iter().any(|(l, v, _)| l.abs() < 1e-5 && (g * *v).length() < 1e-5), "{lines:?}");

        // squash onto a line: the columns are in line, every point lands on it, two
        // singular values are 0, rank 1
        let line = run(3);
        let (g, m) = (line.m, line.shown());
        let along = g.x_axis.normalize();
        assert!(g.y_axis.cross(along).length() < 1e-6 && g.z_axis.cross(along).length() < 1e-6);
        assert!(lattice().all(|(p, _)| (m * p).cross(along).length() < 1e-5));
        let (_, s, _) = svd3(g);
        assert!(s.y < 1e-5 * s.x && s.z < 1e-5 * s.x && rank(s) == 1, "{s}");
    }

    /// Every dashed line is mapped onto itself at every stage of the animation, and
    /// the arrow on it is where the matrix on screen sends the unit vector along it;
    /// the volume written on the solid is the volume of the solid drawn.
    #[test]
    fn the_eigenvector_lines_drawn_are_mapped_onto_themselves() {
        let mut rng = Rng::new(11);
        let mut matrices: Vec<Mat3> = (0..TRIALS.len()).map(trial_matrix).collect();
        matrices.push(LinearMap3::create().m);
        matrices.extend((0..200).map(|_| Mat3::from_cols_array(&[0; 9].map(|_| rng.normal()))));
        let mut lines = 0;
        for g in matrices {
            let mut scene = LinearMap3::create();
            scene.m = g;
            let size = g.to_cols_array().iter().fold(0.0f32, |s, v| s.max(v.abs())).max(1.0);
            for morph in [0.0, 0.3, 0.7, 1.0] {
                scene.morph = morph;
                let shown = scene.shown();
                for (l, v, tip) in scene.eigen_arrows() {
                    lines += 1;
                    assert!((v.length() - 1.0).abs() < 1e-5 && (shown * v - tip).length() < 1e-3 * size, "{g}: λ = {l} along {v} at {morph}");
                    assert!((g * v - v * l).length() < 1e-3 * size, "{g}: λ = {l} along {v}");
                }
                assert!((volume(shown) - shown.determinant()).abs() < 1e-5 * size * size * size);
            }
            assert!((volume(scene.shown()) - g.determinant()).abs() < 1e-5 * size * size * size);
        }
        assert!(lines > 800, "{lines}");
    }

    /// A column dragged onto the origin, two columns on top of each other, all of
    /// them at the origin: every number drawn or read out is finite, and the rank
    /// says what happened.
    #[test]
    fn degenerate_handles_leave_everything_finite() {
        let (a, b, c) = (Vec3::new(1.2, 0.3, 0.0), Vec3::new(-0.4, 1.0, 0.3), Vec3::new(0.2, 0.2, 1.4));
        let cases = [
            (Mat3::from_cols(Vec3::ZERO, b, c), 2),
            (Mat3::from_cols(a, b, Vec3::ZERO), 2),
            (Mat3::from_cols(a, a, c), 2),
            (Mat3::from_cols(a, -a * 2.0, Vec3::ZERO), 1),
            (Mat3::from_cols(Vec3::ZERO, Vec3::ZERO, c), 1),
            (Mat3::ZERO, 0),
        ];
        for (g, want) in cases {
            let mut scene = LinearMap3::create();
            scene.m = g;
            for morph in [0.0, 0.5, 1.0] {
                scene.morph = morph;
                let shown = scene.shown();
                assert!(shown.is_finite() && lattice().all(|(p, _)| (shown * p).is_finite()));
                assert!(scene.eigen_arrows().iter().all(|(l, v, tip)| l.is_finite() && v.is_finite() && tip.is_finite()), "{g}");
            }
            let (u, s, v) = svd3(g);
            assert!(u.is_finite() && s.is_finite() && v.is_finite());
            assert_eq!(rank(s), want, "{g}: singular values {s}");
        }
    }
}

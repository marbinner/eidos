//! Projection and least squares.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "lsq",
    title: "Projection and least squares",
    group: "Linear algebra: equations",
    blurb: "The best approximate solution of $Ax = b$ is an orthogonal projection.",
    make: |g| Box::new(Projection::new(g)),
};

/// Experiments: (button, what to look for, [a₁, a₂, b]).
const TRIALS: [(&str, &str, [[f32; 3]; 3]); 3] = [
    (
        "$b$ in the plane",
        r"$b$ now lies in the plane of the columns (it is $0.75 a_1 + 0.5 a_2$), so $Ax = b$ has an exact solution: $p$ is $b$ itself, the error $e$ is $0$, and $x^* = (0.75, 0.5)$.",
        [[2.0, 0.5, 0.25], [-0.25, 1.75, 0.75], [1.375, 1.25, 0.5625]],
    ),
    (
        "columns at right angles",
        r"With the columns at right angles, each part of $x^*$ can be found on its own: $x_1 = (a_1 \cdot b) / (a_1 \cdot a_1)$, how far $b$ reaches along $a_1$, and $x_2$ likewise along $a_2$. The dashed path turns a right angle.",
        [[1.6, 0.4, 0.4], [-0.4, 1.4, 0.2], [1.0, 1.6, 2.2]],
    ),
    (
        "columns nearly in line",
        r"$a_2$ is only 10° from $a_1$. They still span a plane, and $e$ still meets it at a right angle, but $x^*$ is large, its two parts of opposite sign: the dashed path runs far back along $a_1$ and out again along $a_2$. Nudge $a_2$ and $x^*$ swings.",
        [[2.0, 0.4, 0.3], [1.8, 0.7, 0.4], [1.0, 1.6, 2.4]],
    ),
];

/// Least squares as an orthogonal projection onto a column space.
pub struct Projection {
    a1: Vec3,
    a2: Vec3,
    b: Vec3,
}

impl Projection {
    pub fn new(_: &Gfx) -> Self {
        Self::create()
    }

    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        Projection { a1: Vec3::new(2.0, 0.4, 0.3), a2: Vec3::new(-0.3, 1.8, 0.7), b: Vec3::new(1.0, 1.6, 2.4) }
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let (_, _, [a1, a2, b]) = TRIALS[k];
        (self.a1, self.a2, self.b) = (Vec3::from(a1), Vec3::from(a2), Vec3::from(b));
    }

    fn solve(&self) -> (Vec2, Vec3) {
        project(self.a1, self.a2, self.b)
    }
}

/// The two directions of the sheet drawn for the column space: along a₁ (along a₂
/// if a₁ is nothing), and across that in the plane of both. If the columns are in
/// line the second is zero, and the sheet is drawn as the line it then is.
fn sheet_axes(a1: Vec3, a2: Vec3) -> (Vec3, Vec3) {
    let first = if a1.length_squared() > 0.0 { a1 } else { a2 };
    let n = a1.cross(a2).normalize_or_zero();
    (first.normalize_or_zero(), n.cross(first).normalize_or_zero())
}

/// The least-squares solution of [a1 a2] x = b: x* and the projection p of b onto
/// the column space.
///
/// The normal equations AᵀA x = Aᵀb say what p is, but solving them as they stand
/// squares the condition of the problem: with the two columns a few degrees apart
/// the answer's error was already visible in the read-outs, and at a fraction of a
/// degree it was the whole answer. So p is found from an orthonormal basis of the
/// column space instead (e1 along a1, e2 the part of a2 across it), in f64, and x*
/// by substituting back. If the columns line up, the column space is a line and x*
/// is not determined: the one with x2 = 0 is given.
fn project(a1: Vec3, a2: Vec3, b: Vec3) -> (Vec2, Vec3) {
    let (a1, a2, b) = (a1.as_dvec3(), a2.as_dvec3(), b.as_dvec3());
    let (l1, whole) = (a1.length(), a2.length());
    if l1 <= 1e-12 * whole.max(1e-300) {
        // no first column to speak of: the line of the second, if there is one
        let x2 = if whole > 0.0 { a2.dot(b) / (whole * whole) } else { 0.0 };
        return (Vec2::new(0.0, x2 as f32), (a2 * x2).as_vec3());
    }
    let e1 = a1 / l1;
    let c1 = e1.dot(b);
    let across = a2 - e1 * e1.dot(a2);
    let l2 = across.length();
    if l2 <= 1e-10 * whole.max(l1) {
        return (Vec2::new((c1 / l1) as f32, 0.0), (e1 * c1).as_vec3());
    }
    let e2 = across / l2;
    let c2 = e2.dot(b);
    // p = c1 e1 + c2 e2, where a1 = l1 e1 and a2 = (e1 . a2) e1 + l2 e2
    let x2 = c2 / l2;
    let x1 = (c1 - x2 * e1.dot(a2)) / l1;
    (Vec2::new(x1 as f32, x2 as f32), (e1 * c1 + e2 * c2).as_vec3())
}

impl Scene for Projection {
    fn view(&self) -> View {
        View::Orbit { target: Vec3::new(0.4, 0.6, 0.7), yaw: -0.5, pitch: 0.36, dist: 9.0 }
    }

    fn animated(&self) -> bool {
        false
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.07;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"$A$ has two columns, $\textcolor{green}{a_1}$ and $\textcolor{orange}{a_2}$, and $Ax = x_1 a_1 + x_2 a_2$: walk $x_1$ times along $a_1$, then $x_2$ times along $a_2$. In space, the points $Ax$ can reach make up a plane, so $Ax = b$ usually has no solution: $b$ is off the plane. Least squares settles for the point $p$ of the plane nearest to $b$:");
        ui::equation(ui, r"p = A x^*, \quad \lVert b - A x^* \rVert = \min_x \lVert b - A x \rVert");
        ui::note(ui, r"The error $\textcolor{pink}{e} = b - p$ then meets the plane at a right angle. Drag the tips of $a_1$, $a_2$ and $b$.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::GREEN, r"$a_1$: the first column of $A$"),
                (pal::ORANGE, r"$a_2$: the second column"),
                (pal::WHITE, r"$b$: the point to reach"),
                (pal::BLUE, r"blue sheet: the column space, every point $Ax$ can reach"),
                (pal::CYAN, r"$p = Ax^*$: the nearest point of the sheet to $b$; the dashed path gets there by $x_1$ steps along $a_1$, then $x_2$ along $a_2$"),
                (pal::PINK, r"$e = b - p$: the error, at a right angle to the sheet (the white corner)"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Column space:", r"every point $Ax$ can reach, for some $x$: here the plane through $a_1$ and $a_2$.");
        ui::term(ui, "Normal equations:", r"$A^\top A x^* = A^\top b$. They say $a_1 \cdot e = 0$ and $a_2 \cdot e = 0$: the error is at right angles to both columns.");

        let (x, p) = self.solve();
        let e = self.b - p;
        ui::section(ui, "Numbers");
        ui::readout_colored(ui, r"$x^* = (x_1, x_2)$", format!("$({}, {})$", fmt(x.x), fmt(x.y)), pal::CYAN);
        ui::readout_colored(ui, r"$\lVert e \rVert$ (the least error)", format!("${}$", fmt(e.length())), pal::PINK);
        ui::readout(ui, r"$a_1 \cdot e$", format!("${}$", signed(self.a1.dot(e), 4)));
        ui::readout(ui, r"$a_2 \cdot e$", format!("${}$", signed(self.a2.dot(e), 4)));
        ui::note(ui, r"Both dot products stay zero however you drag $a_1$, $a_2$ and $b$.");
        ui::see_also(ui, &[("overdet", "the same problem seen from the equations' side: lines that do not meet"), ("regress", "least squares fitting a curve to points"), ("gram", "taking shadows away, one vector at a time"), ("linmap3", "a matrix that squashes all of space onto a plane")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let (a1, a2) = (self.a1, self.a2);

        // the column space, as a translucent sheet with a square grid on it (one side
        // of the squares along a1)
        let (e1, e2) = sheet_axes(a1, a2);
        let r = 3.6;
        sheet(f, Vec3::ZERO, e1, e2, r, 4, pal::BLUE);
        f.text(e1 * (-0.5 * r) + e2 * (0.82 * r), "column space of $A$", 13.0, pal::BLUE.mix(pal::WHITE, 0.45), Align2::CENTER_CENTER);

        let (x, p) = self.solve();
        // p is reached by walking x1 along a1, then x2 along a2
        f.dashed(Vec3::ZERO, a1 * x.x, 1.6, pal::CYAN.a(0.8));
        f.dashed(a1 * x.x, p, 1.6, pal::CYAN.a(0.8));
        f.dot(a1 * x.x, 7.0, pal::CYAN);
        f.text_off(a1 * (x.x * 0.5), Vec2::new(0.0, 10.0), format!("${} a_1$", fmt(x.x)), 13.0, pal::CYAN, Align2::CENTER_TOP);
        f.text_off(a1 * x.x + a2 * (x.y * 0.5), Vec2::new(10.0, 6.0), format!("${} a_2$", fmt(x.y)), 13.0, pal::CYAN, Align2::LEFT_TOP);
        f.arrow(Vec3::ZERO, p, 2.8, pal::CYAN);
        f.arrow(p, self.b, 2.6, pal::PINK);
        // right-angle marker at p
        right_angle(f, p, -p, self.b - p, 0.22, pal::WHITE.a(0.8));

        f.arrow(Vec3::ZERO, a1, 3.0, pal::GREEN);
        f.arrow(Vec3::ZERO, a2, 3.0, pal::ORANGE);
        f.arrow(Vec3::ZERO, self.b, 3.0, pal::WHITE);
        f.drag3(1, &mut self.a1, None, pal::GREEN);
        f.drag3(2, &mut self.a2, None, pal::ORANGE);
        f.drag3(3, &mut self.b, None, pal::WHITE);
        let up = Vec2::new(10.0, -10.0);
        f.text_off(a1, up, "$a_1$", 15.0, pal::GREEN, Align2::LEFT_BOTTOM);
        f.text_off(a2, up, "$a_2$", 15.0, pal::ORANGE, Align2::LEFT_BOTTOM);
        f.text_off(self.b, up, "$b$", 16.0, pal::WHITE, Align2::LEFT_BOTTOM);
        f.text_off(p, Vec2::new(10.0, 12.0), "$p = Ax^*$", 15.0, pal::CYAN, Align2::LEFT_TOP);
        f.text_off((p + self.b) * 0.5, Vec2::new(10.0, 0.0), "$e$", 15.0, pal::PINK, Align2::LEFT_CENTER);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "Both dot products stay zero however you drag": also with the two columns
    /// nearly on top of each other, where solving the normal equations in f32 gave
    /// a1 . e = -2 at an angle of 0.03 degrees.
    #[test]
    fn the_error_is_perpendicular_to_both_columns() {
        let (a1, b) = (Vec3::new(2.0, 0.4, 0.3), Vec3::new(1.0, 1.6, 2.4));
        let turn = Vec3::new(-0.3, 1.8, 0.7).normalize();
        for degrees in [77.0f32, 30.0, 3.0, 0.3, 0.03, 0.003, 0.0] {
            // a second column of length 1.9 at that angle from the first
            let a2 = (a1.normalize() * degrees.to_radians().cos() + (turn - a1.normalize() * turn.dot(a1.normalize())).normalize() * degrees.to_radians().sin()) * 1.9;
            let (x, p) = project(a1, a2, b);
            let e = b - p;
            assert!(a1.dot(e).abs() < 5e-5 && a2.dot(e).abs() < 5e-5, "{degrees} degrees: a1.e = {}, a2.e = {}", a1.dot(e), a2.dot(e));
            assert!(x.is_finite() && p.is_finite());
            // p is in the column space, and no point of it is nearer to b
            assert!(e.length() <= (b - a1 * 0.5).length() + 1e-5);
            if degrees >= 3.0 {
                assert!((a1 * x.x + a2 * x.y - p).length() < 1e-4, "{degrees} degrees: x* does not rebuild p");
            }
        }
        // degenerate columns give an answer, not a NaN
        for (a1, a2) in [(Vec3::ZERO, Vec3::Y), (Vec3::X, Vec3::ZERO), (Vec3::ZERO, Vec3::ZERO), (Vec3::X, Vec3::X * -3.0)] {
            let (x, p) = project(a1, a2, b);
            assert!(x.is_finite() && p.is_finite(), "{a1} {a2}");
            assert!(a1.dot(b - p).abs() < 1e-5 && a2.dot(b - p).abs() < 1e-5);
        }
    }

    /// An experiment as its button sets it up.
    fn trial(k: usize) -> Projection {
        let mut scene = Projection::create();
        scene.set_up(k);
        scene
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // b in the plane, as 0.75 a1 + 0.5 a2: p is b, e is 0 (and so is its
        // read-out, not rounding error), x* = (0.75, 0.5)
        let s = trial(0);
        assert_eq!(s.b, s.a1 * 0.75 + s.a2 * 0.5);
        let (x, p) = s.solve();
        let read = [fmt((s.b - p).length()), fmt(x.x), fmt(x.y)];
        assert!((p - s.b).length() < 1e-6 && read == ["0", "0.750", "0.500"], "{read:?}");

        // columns at right angles: each part of x* on its own, and the dashed path
        // turns a right angle
        let s = trial(1);
        assert!(s.a1.dot(s.a2).abs() < 1e-6);
        let (x, p) = s.solve();
        assert!((x.x - s.a1.dot(s.b) / s.a1.dot(s.a1)).abs() < 1e-5 && (x.y - s.a2.dot(s.b) / s.a2.dot(s.a2)).abs() < 1e-5, "{x}");
        let corner = s.a1 * x.x;
        assert!(corner.normalize().dot((p - corner).normalize()).abs() < 1e-5);

        // columns nearly in line: 10 degrees apart, e still at right angles, x* large
        // with parts of opposite sign, and a nudge of a2 swings it far more than it
        // does with the columns the scene starts with
        let s = trial(2);
        let degrees = s.a1.angle_between(s.a2).to_degrees();
        let (x, p) = s.solve();
        let e = s.b - p;
        assert!((degrees - 10.0).abs() < 0.5 && s.a1.dot(e).abs() < 1e-5 && s.a2.dot(e).abs() < 1e-5, "{degrees}");
        assert!(x.x < -3.0 && x.y > 3.0, "{x}");
        let swing = |s: &Projection| [Vec3::X, Vec3::Y, Vec3::Z].iter().map(|d| (project(s.a1, s.a2 + *d * 0.05, s.b).0 - s.solve().0).length()).fold(0.0, f32::max);
        let start = Projection::create();
        assert!(swing(&s) > 0.5 && swing(&s) > 5.0 * swing(&start), "{} against {}", swing(&s), swing(&start));
    }

    /// Wherever the handles are, degenerate places included: the sheet drawn holds
    /// both columns, the error drawn meets it at a right angle, the dashed path ends
    /// at p, and nothing is not a finite number.
    #[test]
    fn the_error_drawn_meets_the_sheet_drawn_at_a_right_angle() {
        let mut rng = Rng::new(5);
        let mut point = || Vec3::new(rng.normal(), rng.normal(), rng.normal()) * 1.5;
        let mut cases: Vec<(Vec3, Vec3, Vec3)> = (0..300).map(|_| (point(), point(), point())).collect();
        let (a1, a2, b) = (Vec3::new(2.0, 0.4, 0.3), Vec3::new(-0.3, 1.8, 0.7), Vec3::new(1.0, 1.6, 2.4));
        cases.extend([(Vec3::ZERO, a2, b), (a1, Vec3::ZERO, b), (Vec3::ZERO, Vec3::ZERO, b), (a1, a1 * -1.5, b), (a1, a2, Vec3::ZERO), (a1, a2, a1.cross(a2))]);
        for (a1, a2, b) in cases {
            let (x, p) = project(a1, a2, b);
            let (e1, e2) = sheet_axes(a1, a2);
            let e = b - p;
            assert!(x.is_finite() && p.is_finite() && e1.is_finite() && e2.is_finite(), "{a1} {a2} {b}");
            let size = b.length().max(1.0);
            assert!(e.dot(e1).abs() < 1e-5 * size && e.dot(e2).abs() < 1e-5 * size, "{a1} {a2} {b}: e . sheet = {}, {}", e.dot(e1), e.dot(e2));
            for a in [a1, a2] {
                assert!((a - e1 * a.dot(e1) - e2 * a.dot(e2)).length() < 1e-5 * a.length().max(1.0), "{a1} {a2}: {a} is off the sheet");
            }
            if a1.cross(a2).length() > 1e-3 * a1.length() * a2.length() {
                assert!((a1 * x.x + a2 * x.y - p).length() < 1e-4 * size, "{a1} {a2} {b}: the path ends away from p");
            }
        }
    }
}

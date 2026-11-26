//! Two directions in Gaussian space.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "hermite_dirs",
    title: "Two directions in Gaussian space",
    group: "Hermite polynomials",
    blurb: r"Functions along two directions overlap by $(\cos\theta)^n$.",
    make: |g| Box::new(HermiteDirections::new(g)),
};

const DIR_N: usize = 150;
const DIR_EXTENT: f32 = 4.2;

/// The angle between two directions given by their angles: which way round `v` lies
/// from `u` (+1 anticlockwise, -1 clockwise) and how far, from 0 to π. Then
/// `v = u + sign * theta` up to whole turns, however the two angles were arrived at.
fn separation(angle_u: f32, angle_v: f32) -> (f32, f32) {
    let d = (angle_v - angle_u).rem_euclid(TAU);
    if d > PI {
        (-1.0, TAU - d)
    } else {
        (1.0, d)
    }
}

/// E[f(u·X) g(v·X)] for a standard normal X in the plane, by a product Gauss–Hermite
/// rule: exact when f and g are polynomials whose degrees add up to less than twice
/// the number of nodes.
fn overlap(rule: &(Vec<f64>, Vec<f64>), f: impl Fn(f64) -> f64, g: impl Fn(f64) -> f64, u: Vec2, v: Vec2) -> f64 {
    let (z, w) = (&rule.0, &rule.1);
    let (ux, uy, vx, vy) = (u.x as f64, u.y as f64, v.x as f64, v.y as f64);
    let mut total = 0.0;
    for (zi, wi) in z.iter().zip(w) {
        for (zj, wj) in z.iter().zip(w) {
            total += wi * wj * f(ux * zi + uy * zj) * g(vx * zi + vy * zj);
        }
    }
    total
}

/// Experiments: (button, what to look for, degree along u, degree along v, angle
/// between them in degrees).
const TRIALS: [(&str, &str, usize, usize, f32); 3] = [
    ("at right angles", r"Degree 3 along both, 90° apart: $\cos\theta = 0$, so $(\cos\theta)^3 = 0$. The warm volume above and the blue volume below are exactly equal.", 3, 3, 90.0),
    ("two different degrees", "Degrees 2 and 4: the volume is zero at this angle and at every other. Drag the arrow tips to check.", 2, 4, 30.0),
    (
        "degree one: the correlation",
        r"$h_1$ is just $x$, so the volume is the average of $(u \cdot X)(v \cdot X)$: $u \cdot v = \cos 60^\circ = 0.5$. It is the correlation of the two coordinates.",
        1,
        1,
        60.0,
    ),
];

/// The directions an experiment starts from: u where it is when the scene opens, v
/// the experiment's angle anticlockwise from it.
fn trial_angles(k: usize) -> (f32, f32) {
    (0.25, 0.25 + TRIALS[k].4.to_radians())
}

pub struct HermiteDirections {
    m: usize,
    n: usize,
    angle_u: f32,
    angle_v: f32,
    surf: Surface,
    key: [f32; 4],
    integral: f64,
    rule: (Vec<f64>, Vec<f64>),
}

impl HermiteDirections {
    pub fn new(gfx: &Gfx) -> Self {
        let mut surf = Surface::new(gfx, DIR_N as u32, DIR_N as u32);
        surf.style.cmap = Some(Cmap::IceFire);
        // barely glossy: on slopes this steep a stronger highlight turns into white smears
        surf.style.specular = 0.08;
        surf.style.contour_alpha = 0.35;
        surf.style.line_color = Color::hex(0xDDE6F5);
        HermiteDirections { m: 3, n: 3, angle_u: 0.25, angle_v: 1.05, surf, key: [f32::NAN; 4], integral: 0.0, rule: gauss_hermite(24) }
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        (self.m, self.n) = (TRIALS[k].2, TRIALS[k].3);
        (self.angle_u, self.angle_v) = trial_angles(k);
    }
}

impl Scene for HermiteDirections {
    fn view(&self) -> View {
        View::Orbit { target: Vec3::new(0.0, 0.0, 0.2), yaw: -1.05, pitch: 0.62, dist: 15.5 }
    }

    fn animated(&self) -> bool {
        false
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.08;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Now in the plane: $X$ is a random point whose two coordinates are independent standard normal numbers. Pick two directions $u$ and $v$. The function $h_m(u \cdot X)$ varies only along $u$, $h_n(v \cdot X)$ only along $v$. The surface is their product times the Gaussian density; its total signed volume is their inner product. That volume depends only on the angle between the directions: it is $(u \cdot v)^n$ when the degrees agree and zero when they differ:");
        ui::equation(ui, r"\mathbb{E}\bigl[h_m(u \cdot X)\, h_n(v \cdot X)\bigr] \\ = \begin{cases} (u \cdot v)^n & \text{if } m = n \\ 0 & \text{if } m \ne n \end{cases}");
        ui::note(ui, "The surface is always drawn at the same height, so read the volume from the numbers.");
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }
        ui::section(ui, "Degrees");
        ui::slider_int(ui, "degree along $u$", &mut self.m, 0..=8);
        ui::slider_int(ui, "degree along $v$", &mut self.n, 0..=8);
        if ui.button("make them equal").clicked() {
            self.n = self.m;
        }
        ui::section(ui, "Directions");
        // a tenth of a degree is all the slider can set and all anyone can see
        // (v may lie on either side of u: the slider keeps it on the side it is on)
        let (side, apart) = separation(self.angle_u, self.angle_v);
        let mut theta = (apart.to_degrees() * 10.0).round() / 10.0;
        if ui::slider(ui, "angle between them (degrees)", &mut theta, 0.0..=180.0) {
            self.angle_v = self.angle_u + side * theta.to_radians();
        }
        ui::note(ui, "Or drag the two arrow tips.");
        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (Cmap::IceFire.color(0.85), "warm surface: where the product is above zero"),
                (Cmap::IceFire.color(0.2), "blue: where it is below"),
                (Color::hex(0xDDE6F5), "light lines: contours; the straight ones are where the surface is zero"),
                (pal::PINK, "pink: the direction $u$, and the line along it (dashed)"),
                (pal::GREEN, "green: the direction $v$"),
                (pal::YELLOW, r"yellow arc: the angle $\theta$ between them"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, r"$u \cdot X$:", "the coordinate of $X$ along $u$ (the dot product). Like each coordinate, it is a standard normal number.");
        ui::term(ui, "Signed volume:", "the volume above zero minus the volume below it.");
        ui::section(ui, "Numbers");
        let c = (self.angle_v - self.angle_u).cos() as f64;
        let predicted = if self.m == self.n { c.powi(self.n as i32) } else { 0.0 };
        ui::readout(ui, r"$u \cdot v = \cos\theta$", format!("${}$", signed(c as f32, 4)));
        ui::readout_colored(ui, "signed volume (computed)", format!("${}$", signed(self.integral as f32, 4)), pal::YELLOW);
        ui::readout_colored(ui, if self.m == self.n { r"$(\cos\theta)^n$" } else { "different degrees" }, format!("${}$", signed(predicted as f32, 4)), pal::CYAN);
        ui::note(ui, r"This is the previous scene in disguise: $u \cdot X$ and $v \cdot X$ are two standard normals with correlation $\cos\theta$, so the average of any $f(u \cdot X)\, g(v \cdot X)$ depends on the angle alone. What only the Hermite polynomials give is so simple an answer: zero between different degrees, and $(\cos\theta)^n$ for the same degree. With plain powers different degrees overlap, $\mathbb{E}[(u \cdot X)(v \cdot X)^3] = 3\cos\theta$, and the answers are longer: $\mathbb{E}[(u \cdot X)^2 (v \cdot X)^2] = 1 + 2\cos^2\theta$. In high dimensions two random directions are nearly at right angles: $\cos\theta$ averages $0$, and its typical size is $1/\sqrt{d}$, so the overlap of degree-$n$ features of different neurons is typically of size only $d^{-n/2}$.");
        ui::see_also(ui, &[("hermite_noise", "the same overlap along one line"), ("hermite_orth", "why different degrees give zero"), ("gauss2", "two correlated normal variables")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let r = 3.6;
        let mut hu = Vec3::new(self.angle_u.cos() * r, self.angle_u.sin() * r, 0.0);
        let mut hv = Vec3::new(self.angle_v.cos() * r, self.angle_v.sin() * r, 0.0);
        if f.drag3(1, &mut hu, Some(Vec3::Z), pal::PINK) {
            self.angle_u = hu.y.atan2(hu.x);
        }
        if f.drag3(2, &mut hv, Some(Vec3::Z), pal::GREEN) {
            self.angle_v = hv.y.atan2(hv.x);
        }
        let key = [self.m as f32, self.n as f32, self.angle_u, self.angle_v];
        if key != self.key {
            self.key = key;
            let (m, n) = (self.m, self.n);
            let u = Vec2::from_angle(self.angle_u);
            let v = Vec2::from_angle(self.angle_v);
            let e = DIR_EXTENT;
            let nn = DIR_N;
            let mut vals = vec![0.0f64; nn * nn];
            let mut peak = 1e-12f64;
            for j in 0..nn {
                for i in 0..nn {
                    let x = -e + 2.0 * e * i as f32 / (nn - 1) as f32;
                    let y = -e + 2.0 * e * j as f32 / (nn - 1) as f32;
                    let p = Vec2::new(x, y);
                    let g = h(m, u.dot(p) as f64) * h(n, v.dot(p) as f64) * phi(x as f64) * phi(y as f64);
                    vals[j * nn + i] = g;
                    peak = peak.max(g.abs());
                }
            }
            // The volume itself by a 24 x 24 Gauss-Hermite rule, which is exact for this
            // integrand (a polynomial of degree at most 16 in each variable).
            self.integral = overlap(&self.rule, |t| h(m, t), |t| h(n, t), u, v);
            let scale = 2.4 / peak;
            for (k, q) in self.surf.data_mut().iter_mut().enumerate() {
                let (i, j) = (k % nn, k / nn);
                let x = -e + 2.0 * e * i as f32 / (nn - 1) as f32;
                let y = -e + 2.0 * e * j as f32 / (nn - 1) as f32;
                let z = (vals[k] * scale) as f32;
                *q = [x, y, z, z];
            }
            self.surf.style.range = [-2.4, 2.4];
            self.surf.style.contour = 0.3;
        }
        f.surface(&mut self.surf);

        let u3 = Vec3::new(self.angle_u.cos(), self.angle_u.sin(), 0.0);
        let v3 = Vec3::new(self.angle_v.cos(), self.angle_v.sin(), 0.0);
        let lift = Vec3::Z * 0.02;
        f.dashed(-u3 * 5.6 + lift, u3 * 5.6 + lift, 1.4, pal::PINK.a(0.7));
        f.dashed(-v3 * 5.6 + lift, v3 * 5.6 + lift, 1.4, pal::GREEN.a(0.7));
        // drawn over the surface, so the directions stay readable behind the bumps
        f.dl.overlay.segment(lift, u3 * r + lift, 2.6, pal::PINK.a(0.85));
        f.dl.overlay.segment(lift, v3 * r + lift, 2.6, pal::GREEN.a(0.85));
        // names on plates: the surface behind them is orange, red, blue or white by turns
        for (tip, name, c) in [(u3 * r, "$u$", pal::PINK), (v3 * r, "$v$", pal::GREEN)] {
            if let Some(p) = on_screen(f, tip) {
                f.caption_px(p + Vec2::new(12.0, -12.0), name, 17.0, c, Align2::LEFT_BOTTOM);
            }
        }
        // the angle, drawn over the surface like the two directions and in a colour of
        // its own, so that it is not taken for one of the contour lines
        // (the short way round from u to v, whatever the two angles happen to be)
        let (a0, a1) = (self.angle_u, self.angle_v);
        let (side, apart) = separation(a0, a1);
        f.dl.overlay.polyline((0..=40).map(|i| a0 + side * apart * i as f32 / 40.0).map(|a| Vec3::new(a.cos() * 1.3, a.sin() * 1.3, 0.03)), 2.2, pal::YELLOW);
        let mid = a0 + side * apart * 0.5;
        if let Some(p) = on_screen(f, Vec3::new(mid.cos() * 1.75, mid.sin() * 1.75, 0.03)) {
            f.caption_px(p, r"$\theta$", 15.0, pal::YELLOW, Align2::CENTER_CENTER);
        }

        let c = (a1 - a0).cos() as f64;
        let view = f.view_size();
        let line = if self.m == self.n {
            format!(r"$\text{{volume above}} - \text{{volume below}} = {}$        $(\cos\theta)^{{{}}} = {}$", signed(self.integral as f32, 4), self.n, signed(c.powi(self.n as i32) as f32, 4))
        } else {
            format!(r"$\text{{volume above}} - \text{{volume below}} = {}$        degrees {} and {} never overlap, at any angle", signed(self.integral as f32, 4), self.m, self.n)
        };
        // on two lines where one would run off the sides (a phone)
        for (k, part) in caption_lines(f, &line.split("        ").collect::<Vec<_>>(), "        ", 16.0, view.x - 36.0).into_iter().enumerate() {
            f.caption_px(Vec2::new(view.x * 0.5, f.top_inset + 18.0 + k as f32 * 24.0), part, 16.0, pal::YELLOW, Align2::CENTER_TOP);
        }
        let surface = format!(r"surface: $h_{{{}}}(u \cdot x)\, h_{{{}}}(v \cdot x)\, \varphi(x)$", self.m, self.n);
        bottom_caption(f, &[&surface, "warm above zero, blue below", "straight lines: where it is zero"], 12.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// v on either side of u, and angles that have run past a half turn (atan2
    /// jumps there): the separation is always the short way round, and putting it
    /// back gives the same direction.
    #[test]
    fn the_angle_between_is_the_short_way_round() {
        for (u, v, side, apart) in [(0.25f32, 1.05f32, 1.0f32, 0.8f32), (0.25, 0.2, -1.0, 0.05), (3.0, -3.0, 1.0, TAU - 6.0), (-3.0, 3.0, -1.0, TAU - 6.0), (1.0, 1.0, 1.0, 0.0), (0.0, 7.0, 1.0, 7.0 - TAU)] {
            let (s, a) = separation(u, v);
            assert!(s == side && (a - apart).abs() < 1e-5, "u {u}, v {v}: {s} {a}");
            assert!((0.0..=PI).contains(&a));
            let back = u + s * a;
            assert!((Vec2::from_angle(back) - Vec2::from_angle(v)).length() < 1e-5);
        }
    }

    /// The note: the average of f(u·X) g(v·X) depends on the angle alone for any f and
    /// g, Hermite or not; what is special about Hermite is the answer, zero or (cos θ)ⁿ.
    /// (It used to say that for any other family the answer would depend on more than
    /// the angle.) Measured with the scene's own rule, at several turns of the pair.
    #[test]
    fn only_the_answer_is_special_to_hermite() {
        let rule = gauss_hermite(24);
        // (the directions are single precision, so their lengths are 1 only to about 1e-7)
        for theta in [0.0f32, 0.4, 1.0, 1.9, 2.8] {
            let c = theta.cos() as f64;
            let at = |turn: f32| (Vec2::from_angle(turn), Vec2::from_angle(turn + theta));
            for turn in [0.0f32, 0.25, 1.3, 2.9, -2.2] {
                let (u, v) = at(turn);
                // plain powers: only the angle matters, but different degrees overlap and
                // the same degree gives a longer polynomial in cos θ
                let p = |k: i32| move |t: f64| t.powi(k);
                assert!((overlap(&rule, p(1), p(3), u, v) - 3.0 * c).abs() < 2e-6, "θ = {theta}");
                assert!((overlap(&rule, p(2), p(2), u, v) - (1.0 + 2.0 * c * c)).abs() < 2e-6, "θ = {theta}");
                // Hermite: (cos θ)ⁿ for equal degrees, zero otherwise
                for m in 0..=8 {
                    for n in 0..=8 {
                        let want = if m == n { c.powi(n as i32) } else { 0.0 };
                        let got = overlap(&rule, |t| h(m, t), |t| h(n, t), u, v);
                        assert!((got - want).abs() < 2e-6, "θ = {theta}, degrees {m} and {n}: {got} against {want}");
                    }
                }
            }
        }
        // (and the plain powers' overlap does change with the angle)
        let (u, v) = (Vec2::X, Vec2::from_angle(1.0));
        assert!((overlap(&rule, |t| t * t, |t| t * t, u, v) - overlap(&rule, |t| t * t, |t| t * t, u, u)).abs() > 1.0);
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let rule = gauss_hermite(24);
        let volume = |k: usize, angle_u: f32, angle_v: f32| {
            let (m, n) = (TRIALS[k].2, TRIALS[k].3);
            overlap(&rule, |t| h(m, t), |t| h(n, t), Vec2::from_angle(angle_u), Vec2::from_angle(angle_v))
        };
        // at right angles: (cos θ)³ = 0, the warm and blue volumes cancel
        let (u, v) = trial_angles(0);
        assert_eq!((TRIALS[0].2, TRIALS[0].3), (3, 3));
        assert!((separation(u, v).1.to_degrees() - 90.0).abs() < 1e-3);
        assert_eq!(signed(volume(0, u, v) as f32, 4), "0");
        assert!(volume(0, u, u + 0.3).abs() > 0.5, "not at a smaller angle");
        // two different degrees: zero here, and at every angle the arrows can be dragged to
        let (u, v) = trial_angles(1);
        assert!(TRIALS[1].2 != TRIALS[1].3 && volume(1, u, v).abs() < 3e-7);
        for i in 0..72 {
            let a = i as f32 * TAU / 72.0;
            assert!(volume(1, 0.7, 0.7 + a).abs() < 3e-7, "at {a}");
        }
        // degree one: the volume is cos 60° = 0.5
        let (u, v) = trial_angles(2);
        assert_eq!((TRIALS[2].2, TRIALS[2].3), (1, 1));
        assert!((volume(2, u, v) - 0.5).abs() < 1e-7, "{}", volume(2, u, v));
    }
}

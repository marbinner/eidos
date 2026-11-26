//! Correlated normals.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "gauss2",
    title: "Correlated normals",
    group: CHANCE,
    blurb: "Covariance, principal axes, and conditioning, on one bell.",
    make: |g| Box::new(Gauss2::new(g)),
};

/// Experiments: (button, what to look for, σx, σy, ρ, where the slice is).
const TRIALS: [(&str, &str, f32, f32, f32, f32); 3] = [
    (
        "almost a line",
        r"$\rho = 0.95$: the bell is squeezed into a narrow ridge along the diagonal. Knowing $x$ pins $y$ down: on the wall, $p(y \mid x)$ is a tall, narrow curve, its spread 0.31 against 1 for $p(y)$.",
        1.0,
        1.0,
        0.95,
        1.0,
    ),
    (
        "no correlation",
        r"$\rho = 0$: the yellow curve on the wall lies on $p(y)$ and stays there wherever you drag the slice. Knowing $x$ tells you nothing about $y$.",
        1.3,
        0.9,
        0.0,
        1.2,
    ),
    (
        "negative correlation",
        r"$\rho = -0.8$: the regression line slopes down. Drag the slice towards larger $x$ and the yellow curve on the wall moves towards smaller $y$: a large $x$ predicts a small $y$.",
        1.0,
        1.0,
        -0.8,
        1.0,
    ),
];

pub struct Gauss2 {
    sx: f32,
    sy: f32,
    rho: f32,
    x0: f32,
    base: Vec<Vec2>,
    /// (Made on the first frame: nothing else here needs the graphics device, so the
    /// tests can make a scene too.)
    surf: Option<Surface>,
    key: [f32; 3],
    show_samples: bool,
    show_conditional: bool,
}

impl Gauss2 {
    pub fn new(_: &Gfx) -> Self {
        Self::create()
    }

    fn create() -> Self {
        let mut rng = Rng::new(9);
        Gauss2 {
            sx: 1.3,
            sy: 0.9,
            rho: 0.7,
            x0: 1.2,
            base: (0..700).map(|_| Vec2::new(rng.normal(), rng.normal())).collect(),
            surf: None,
            key: [f32::NAN; 3],
            show_samples: true,
            show_conditional: true,
        }
    }

    /// Set up one of the experiments in `TRIALS`, with the slice and its curve shown.
    fn set_up(&mut self, k: usize) {
        let (_, _, sx, sy, rho, x0) = TRIALS[k];
        (self.sx, self.sy, self.rho, self.x0) = (sx, sy, rho, x0);
        self.show_conditional = true;
    }

    fn pdf(&self, x: f32, y: f32) -> f32 {
        let q = 1.0 - self.rho * self.rho;
        let (a, b) = (x / self.sx, y / self.sy);
        (-(a * a - 2.0 * self.rho * a * b + b * b) / (2.0 * q)).exp() / (TAU * self.sx * self.sy * q.sqrt())
    }

    /// The height of the surface: the density scaled so that its peak is always
    /// `G2_PEAK` high (a narrow bell is in truth far taller than a wide one).
    fn height(&self, x: f32, y: f32) -> f32 {
        self.pdf(x, y) / self.pdf(0.0, 0.0) * G2_PEAK
    }

    /// Lower-triangular factor: maps standard normals to this distribution.
    fn chol(&self, z: Vec2) -> Vec2 {
        Vec2::new(self.sx * z.x, self.sy * (self.rho * z.x + (1.0 - self.rho * self.rho).sqrt() * z.y))
    }

    /// The normal curve p(y | x = x0): its mean (on the regression line) and spread.
    fn conditional(&self) -> (f32, f32) {
        (self.rho * self.sy / self.sx * self.x0, self.sy * (1.0 - self.rho * self.rho).sqrt())
    }

    /// Height per unit of density for the curves on the walls: the same for all of
    /// them, so that they can be compared, and small enough that the tallest stays in view.
    fn wall_scale(&self) -> f32 {
        let narrowest = if self.show_conditional { self.conditional().1 } else { self.sy }.min(self.sx);
        G2_WALL.min(G2_WALL_TOP / normal_pdf(0.0, 0.0, narrowest))
    }
}

const G2_EXTENT: f32 = 3.6;
/// The height of the surface's peak, whatever the sliders say.
const G2_PEAK: f32 = 1.7;
/// Height per unit of density on the walls, and the most a curve there may rise.
const G2_WALL: f32 = 2.6;
const G2_WALL_TOP: f32 = 3.5;
/// The surface's grid points along each side.
const G2_N: usize = 128;
const SAMPLE_COLOUR: Color = Color::hex(0x9DB4FF);

impl Scene for Gauss2 {
    fn view(&self) -> View {
        // far enough back that the marginal on the left wall is not cut off
        View::Orbit { target: Vec3::new(0.0, 0.0, 0.5), yaw: -1.05, pitch: 0.5, dist: 17.0 }
    }

    fn animated(&self) -> bool {
        false
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.10;
        // the coloured arrows are the axes here: the floor's own two only add stray lines
        s.axis_alpha = 0.10;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Two quantities, $x$ and $y$, that vary together, each normally distributed: the surface shows how likely each pair $(x, y)$ is. The correlation $\rho$ says how closely they move together; it tilts and squeezes the bell. Fix $x$ (the yellow slice) and the values of $y$ that go with it again follow a normal curve, $p(y \mid x)$: centred on the regression line, and the narrower the stronger the correlation.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Distribution");
        ui::slider(ui, r"$\sigma_x$", &mut self.sx, 0.3..=2.0);
        ui::slider(ui, r"$\sigma_y$", &mut self.sy, 0.3..=2.0);
        ui::slider(ui, r"correlation $\rho$", &mut self.rho, -0.97..=0.97);
        ui::note(ui, r"$\sigma_x$ and $\sigma_y$: how widely $x$ and $y$ spread, each on its own. $\rho$: how closely they move together, from $-1$ to $1$ ($0$: not at all).");
        ui::section(ui, "Show");
        ui.checkbox(&mut self.show_samples, "samples");
        checkbox(ui, &mut self.show_conditional, r"the slice at one $x$, and $p(y \mid x)$ on the wall");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (Cmap::Magma.color(0.5), "the surface: how likely each pair $(x, y)$ is. It is always drawn the same height; in truth a narrow bell is far taller (see its peak below)"),
                (SAMPLE_COLOUR, "blue dots: 700 pairs drawn from it"),
                (pal::WHITE, "white rings: the $1\\sigma$ and $2\\sigma$ ellipses; about 39\u{a0}% and 86\u{a0}% of the pairs fall inside them"),
                (pal::ORANGE, "axis 1: the direction of most spread, as long as the spread along it"),
                (pal::PURPLE, "axis 2: the direction of least spread"),
                (pal::AXIS[0], "on the back wall: $p(x)$, how $x$ spreads on its own"),
                (pal::AXIS[1], "on the left wall: $p(y)$, how $y$ spreads on its own"),
                (pal::YELLOW, r"yellow: the slice of the surface at one $x$; the regression line, where the middle of each slice lies; and on the left wall $p(y \mid x)$, drawn to the same scale as $p(y)$"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Covariance matrix:", r"the spreads $\sigma_x^2$ and $\sigma_y^2$ on its diagonal, and $\rho\, \sigma_x \sigma_y$, the covariance, off it:");
        ui::equation(ui, r"\begin{bmatrix} \sigma_x^2 & \rho\, \sigma_x \sigma_y \\ \rho\, \sigma_x \sigma_y & \sigma_y^2 \end{bmatrix}");
        ui::term(ui, "Principal axes:", "the directions of most and least spread: the eigenvectors of the covariance matrix. The variances along them are its eigenvalues.");
        ui::term(ui, r"$p(y \mid x)$:", "how $y$ is spread among only those pairs with this $x$. It has the shape of the slice of the surface at $x$, scaled up to an area of 1.");

        ui::section(ui, "Numbers");
        let cov = self.rho * self.sx * self.sy;
        ui::readout(ui, "covariance matrix", format!("${}$", bmatrix(&[[self.sx * self.sx, cov], [cov, self.sy * self.sy]], 3)));
        let (l1, l2, _) = sym_eig(self.sx * self.sx, cov, self.sy * self.sy);
        ui::readout_colored(ui, "variance along axis 1", format!("${}$", fmt(l1)), pal::ORANGE);
        ui::readout_colored(ui, "variance along axis 2", format!("${}$", fmt(l2)), pal::PURPLE);
        if self.show_conditional {
            let (mean, sd) = self.conditional();
            ui::readout_colored(ui, &format!("mean of $y$ given $x = {}$", fmt(self.x0)), format!("${}$", fmt(mean)), pal::YELLOW);
            ui::readout_colored(ui, "spread (sd) of $y$ given $x$", format!("${}$", fmt(sd)), pal::YELLOW);
            ui::readout_colored(ui, "spread (sd) of $y$ overall", format!("${}$", fmt(self.sy)), pal::AXIS[1]);
        }
        ui::readout(ui, "peak density", format!("${}$", fmt(self.pdf(0.0, 0.0))));
        ui::see_also(ui, &[("lsq", "the regression line as a projection"), ("regress", "fitting curves to noisy points"), ("hermite_noise", "pairs of correlated normals, from another side"), ("pca", "the axes of a cloud of data: principal components")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let e = G2_EXTENT;
        if self.surf.is_none() {
            self.surf = Some(Surface::new(f.gfx, G2_N as u32, G2_N as u32));
            self.key = [f32::NAN; 3];
        }
        if self.key != [self.sx, self.sy, self.rho] {
            self.key = [self.sx, self.sy, self.rho];
            let n = G2_N;
            let data: Vec<[f32; 4]> = (0..n * n)
                .map(|k| {
                    let x = -e + 2.0 * e * (k % n) as f32 / (n - 1) as f32;
                    let y = -e + 2.0 * e * (k / n) as f32 / (n - 1) as f32;
                    let z = self.height(x, y);
                    [x, y, z, z]
                })
                .collect();
            if let Some(surf) = &mut self.surf {
                surf.data_mut().copy_from_slice(&data);
                let st = &mut surf.style;
                st.cmap = Some(Cmap::Magma);
                st.range = [-0.12 * G2_PEAK, G2_PEAK * 1.05];
                st.contour = G2_PEAK / 9.0;
                st.contour_alpha = 0.35;
                st.opacity = 0.88;
                st.specular = 0.35;
            }
        }
        if let Some(surf) = &mut self.surf {
            f.surface(surf);
        }
        f.axes3(Vec3::new(e + 0.5, e + 0.5, 2.2));

        if self.show_samples {
            for &z in &self.base {
                let p = self.chol(z);
                if p.x.abs() < e && p.y.abs() < e {
                    f.sphere(Vec3::new(p.x, p.y, 0.035), 0.07, SAMPLE_COLOUR);
                }
            }
        }

        // 1σ and 2σ ellipses, and the principal axes
        for (k, a) in [(1.0, 0.95), (2.0, 0.55)] {
            let ring: Vec<Vec3> = (0..120).map(|i| self.chol(Vec2::from_angle(i as f32 / 120.0 * TAU) * k).extend(0.012)).collect();
            f.dl.lines.closed(&ring, 2.0, pal::WHITE.a(a));
            // named on its near left side, clear of the regression line and the slice
            f.text_off(ring[80], Vec2::new(-3.0, 4.0), format!(r"${k:.0}\sigma$"), 12.0, pal::WHITE.a(a.max(0.8)), Align2::RIGHT_TOP);
        }
        let cov = self.rho * self.sx * self.sy;
        let (l1, l2, v1) = sym_eig(self.sx * self.sx, cov, self.sy * self.sy);
        let v2 = Vec2::new(-v1.y, v1.x);
        for (l, v, c, name) in [(l1, v1, pal::ORANGE, "axis 1"), (l2, v2, pal::PURPLE, "axis 2")] {
            let tip = (v * l.sqrt()).extend(0.02);
            f.arrow(Vec3::new(0.0, 0.0, 0.02), tip, 3.0, c);
            f.line(-tip, Vec3::new(0.0, 0.0, 0.02), 1.6, c.a(0.5));
            // on a plate: the bell behind is nearly the same colour
            if let Some(px) = on_screen(f, tip) {
                f.caption_px(px + Vec2::new(10.0, -8.0), name, 13.0, c, Align2::LEFT_BOTTOM);
            }
        }

        // marginals on the back walls, and beside p(y) the conditional, all to one scale
        let wall = e + 0.25;
        let mh = self.wall_scale();
        f.curve(-e, e, 160, 2.2, pal::AXIS[0], |x| Vec3::new(x, wall, normal_pdf(x, 0.0, self.sx) * mh));
        f.curve(-e, e, 160, 2.2, pal::AXIS[1], |y| Vec3::new(-wall, y, normal_pdf(y, 0.0, self.sy) * mh));
        f.line((-e, wall, 0.0), (e, wall, 0.0), 1.0, pal::AXIS[0].a(0.4));
        f.line((-wall, -e, 0.0), (-wall, e, 0.0), 1.0, pal::AXIS[1].a(0.4));
        f.text((0.0, wall, normal_pdf(0.0, 0.0, self.sx) * mh + 0.12), "$p(x)$", 13.0, pal::AXIS[0], Align2::CENTER_BOTTOM);
        if self.show_conditional {
            // on its near flank: the top is where p(y | x) rises past it
            let flank = Vec3::new(-wall, -1.3 * self.sy, normal_pdf(1.3, 0.0, 1.0) / self.sy * mh);
            f.text_off(flank, Vec2::new(-8.0, 0.0), "$p(y)$", 13.0, pal::AXIS[1], Align2::RIGHT_CENTER);
        } else {
            f.text((-wall, 0.0, normal_pdf(0.0, 0.0, self.sy) * mh + 0.12), "$p(y)$", 13.0, pal::AXIS[1], Align2::CENTER_BOTTOM);
        }

        if self.show_conditional {
            let mut h = Vec3::new(self.x0, -e - 0.35, 0.0);
            if f.drag3(1, &mut h, Some(Vec3::Z), pal::YELLOW) {
                self.x0 = h.x.clamp(-e + 0.1, e - 0.1);
            }
            let x0 = self.x0;
            let slope = self.rho * self.sy / self.sx;
            // the regression line: the conditional mean for every x
            f.dashed((-e, -e * slope, 0.015), (e, e * slope, 0.015), 1.8, pal::YELLOW.a(0.9));
            f.text_off((e, e * slope, 0.015), Vec2::new(8.0, 0.0), "regression line", 12.0, pal::YELLOW, Align2::LEFT_CENTER);
            // the cut through the surface at x0: the shape of p(y | x0), at the
            // height of the joint density (low, where x0 itself is unlikely)
            let cut: Vec<Vec3> = (0..=160).map(|i| -e + 2.0 * e * i as f32 / 160.0).map(|y| Vec3::new(x0, y, self.height(x0, y))).collect();
            for w in cut.windows(2) {
                let (a, b) = (w[0], w[1]);
                f.dl.tris_alpha.quad(Vec3::new(a.x, a.y, 0.0), Vec3::new(b.x, b.y, 0.0), b, a, pal::YELLOW.a(0.28));
            }
            f.path(cut.iter().copied(), 2.6, pal::YELLOW);
            let (cm, csd) = self.conditional();
            f.dashed((x0, -e, 0.0), (x0, e, 0.0), 1.2, pal::YELLOW.a(0.6));
            let mean = Vec3::new(x0, cm, 0.02);
            f.sphere(mean, 0.16, pal::YELLOW);
            f.line((x0, cm - csd, 0.02), (x0, cm + csd, 0.02), 4.0, pal::YELLOW);
            f.text_off(h, Vec2::new(0.0, 14.0), format!("drag to move the slice   ·   $x = {x0:.2}$"), 13.0, pal::YELLOW, Align2::CENTER_TOP);
            // the slice scaled to an area of 1 is p(y | x0) itself: beside p(y), to compare
            f.curve(-e, e, 200, 2.4, pal::YELLOW, |y| Vec3::new(-wall, y, normal_pdf(y, cm, csd) * mh));
            f.text((-wall, cm, normal_pdf(0.0, 0.0, csd) * mh + 0.12), format!(r"$p(y \mid x = {x0:.2})$"), 13.0, pal::YELLOW, Align2::CENTER_BOTTOM);
        }
    }
}

/// Eigen-decomposition of [[a, b], [b, d]]: (larger, smaller, unit vector of the
/// larger, the one of its two directions that points to the right, or straight up).
fn sym_eig(a: f32, b: f32, d: f32) -> (f32, f32, Vec2) {
    use crate::math::linalg::{eig2, Eig2};
    match eig2(Mat2::from_cols(Vec2::new(a, b), Vec2::new(b, d))) {
        Eig2::Real { l1, l2, v1, .. } => (l1, l2, if v1.x < 0.0 || (v1.x == 0.0 && v1.y < 0.0) { -v1 } else { v1 }),
        // (a symmetric matrix has real eigenvalues; rounding could only say otherwise
        // when the two are equal, and then every direction serves)
        Eig2::Complex { re, .. } => (re, re, Vec2::X),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_axes_of_the_ellipse() {
        // against the closed form, for covariances of every kind the sliders give
        for (a, b, d) in [(1.0f32, 0.6, 1.0), (4.0, -1.2, 0.25), (0.09, 0.0, 2.0), (2.0, 0.0, 0.5), (1.0, 0.0, 1.0), (0.04, 0.059, 0.09)] {
            let (l1, l2, v) = sym_eig(a, b, d);
            let s = (((a - d) * 0.5).powi(2) + b * b).sqrt();
            assert!((l1 - ((a + d) * 0.5 + s)).abs() < 1e-5 && (l2 - ((a + d) * 0.5 - s)).abs() < 1e-5, "({a}, {b}, {d}): {l1} {l2}");
            // v is a unit eigenvector of the larger value, pointing right (or up)
            let mv = Vec2::new(a * v.x + b * v.y, b * v.x + d * v.y);
            assert!((mv - v * l1).length() < 1e-4 && (v.length() - 1.0).abs() < 1e-5, "({a}, {b}, {d}): {v}");
            assert!(v.x > 0.0 || (v.x == 0.0 && v.y > 0.0) || (a == d && b == 0.0), "({a}, {b}, {d}): {v}");
        }
    }

    /// Settings from every corner of the sliders, and the slice anywhere it can go.
    fn corners() -> Vec<Gauss2> {
        let mut all = Vec::new();
        for (sx, sy) in [(0.3, 0.3), (0.3, 2.0), (2.0, 0.3), (2.0, 2.0), (1.3, 0.9)] {
            for rho in [-0.97, -0.5, 0.0, 0.7, 0.97] {
                for x0 in [-3.5, -1.2, 0.0, 1.2, 3.5] {
                    let mut g = Gauss2::create();
                    (g.sx, g.sy, g.rho, g.x0) = (sx, sy, rho, x0);
                    all.push(g);
                }
            }
        }
        all
    }

    /// The surface peaks at the same height whatever the sliders say (at σ = 0.3 and
    /// ρ = 0.97 the density itself peaks at 7.27, against 0.19 by default), and the
    /// curves on the walls stay in view, all three to one scale.
    #[test]
    fn the_picture_keeps_its_size() {
        for g in corners() {
            let e = G2_EXTENT;
            let top = (0..G2_N * G2_N).map(|k| g.height(-e + 2.0 * e * (k % G2_N) as f32 / (G2_N - 1) as f32, -e + 2.0 * e * (k / G2_N) as f32 / (G2_N - 1) as f32)).fold(0.0, f32::max);
            assert!((g.height(0.0, 0.0) - G2_PEAK).abs() < 1e-5 && top <= G2_PEAK * (1.0 + 1e-5), "{} {} {}: peak {top}", g.sx, g.sy, g.rho);
            let (_, csd) = g.conditional();
            let tallest = [g.sx, g.sy, csd].map(|sd| normal_pdf(0.0, 0.0, sd) * g.wall_scale()).into_iter().fold(0.0, f32::max);
            assert!(tallest <= G2_WALL_TOP * 1.0001, "{} {} {}: a curve {tallest} high on the wall", g.sx, g.sy, g.rho);
        }
        // the default keeps the old scale on the walls
        let g = Gauss2::create();
        assert!(g.wall_scale() == G2_WALL && (g.pdf(0.0, 0.0) - 0.1905).abs() < 1e-4);
        let mut narrow = Gauss2::create();
        (narrow.sx, narrow.sy, narrow.rho) = (0.3, 0.3, 0.97);
        assert!((narrow.pdf(0.0, 0.0) - 7.27).abs() < 0.01, "{}", narrow.pdf(0.0, 0.0));
    }

    /// p(y | x0), as drawn and read out, is how y is spread among the scene's own
    /// samples whose x is near x0: the right centre and spread, out in the tails too.
    #[test]
    fn the_conditional_is_what_the_samples_near_x_say() {
        let mut rng = Rng::new(5);
        let normals: Vec<Vec2> = (0..400_000).map(|_| Vec2::new(rng.normal(), rng.normal())).collect();
        for (sx, sy, rho, x0) in [(1.3, 0.9, 0.7, 1.2), (1.0, 1.0, 0.95, 1.0), (1.0, 1.0, -0.8, -0.6), (2.0, 0.5, 0.3, 2.5), (0.3, 0.3, 0.97, 0.45)] {
            let mut g = Gauss2::create();
            (g.sx, g.sy, g.rho, g.x0) = (sx, sy, rho, x0);
            let near: Vec<f32> = normals.iter().map(|&z| g.chol(z)).filter(|p| (p.x - x0).abs() < 0.02 * sx).map(|p| p.y).collect();
            let mean = near.iter().sum::<f32>() / near.len() as f32;
            let sd = (near.iter().map(|y| (y - mean).powi(2)).sum::<f32>() / near.len() as f32).sqrt();
            let (cm, csd) = g.conditional();
            assert!(near.len() > 2000 && (mean - cm).abs() < 0.03 * sy && (sd / csd - 1.0).abs() < 0.025, "{sx} {sy} {rho} at {x0}: {} samples, mean {mean} sd {sd} against {cm} {csd}", near.len());
        }
    }

    /// The rings: 39 % and 86 % of all pairs inside them (1 - e^(-k²/2) for k = 1, 2),
    /// and near enough as many of the 700 drawn.
    #[test]
    fn the_rings_hold_what_the_key_says() {
        let g = Gauss2::create();
        let inside = |k: f32| g.base.iter().filter(|z| z.length() < k).count() as f32 / g.base.len() as f32;
        assert!((inside(1.0) - 0.393).abs() < 0.01 && (inside(2.0) - 0.865).abs() < 0.025, "{} {}", inside(1.0), inside(2.0));
        assert!(((1.0 - (-0.5f32).exp()) - 0.393).abs() < 5e-4 && ((1.0 - (-2.0f32).exp()) - 0.865).abs() < 5e-4);
    }

    /// The experiments' captions, held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let mut g = Gauss2::create();
        g.show_conditional = false;
        // almost a line: a ridge along the diagonal; the spread of y given x is 0.31,
        // against 1 overall
        g.set_up(0);
        assert!(g.show_conditional && (g.conditional().1 - 0.31).abs() < 0.005 && g.sy == 1.0, "{:?}", g.conditional());
        let (l1, l2, v) = sym_eig(1.0, g.rho, 1.0);
        assert!((v - Vec2::splat(0.5f32.sqrt())).length() < 1e-5 && l2 / l1 < 0.03, "axis {v}, variances {l1} {l2}");
        // no correlation: p(y | x) is p(y), wherever the slice is
        g.set_up(1);
        for x0 in [-3.0, 0.0, 1.2, 3.0] {
            g.x0 = x0;
            assert!(g.conditional() == (0.0, g.sy), "{:?}", g.conditional());
        }
        // negative correlation: a larger x, a smaller y
        g.set_up(2);
        let at = |g: &mut Gauss2, x0: f32| {
            g.x0 = x0;
            g.conditional().0
        };
        assert!(at(&mut g, 2.0) < at(&mut g, 1.0) && at(&mut g, 1.0) < 0.0 && at(&mut g, -1.0) > 0.0);
    }
}

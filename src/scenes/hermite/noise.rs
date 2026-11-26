//! Correlation and the noise operator.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "hermite_noise",
    title: "Correlation and the noise operator",
    group: "Hermite polynomials",
    blurb: r"Averaging over a correlated Gaussian scales degree $n$ by $\rho^n$.",
    make: |g| Box::new(HermiteNoise::new(g)),
};

/// Experiments: (button, what to look for, function (0: one Hermite polynomial, 1: a
/// formula), its degree, the formula as an index into `EXPANSION_PRESETS`, ρ).
const TRIALS: [(&str, &str, usize, usize, usize, f32); 3] = [
    (
        "high degrees are forgotten first",
        r"Degree 8 at $\rho = 0.7$: $\rho^8 = 0.058$, so the cyan curve is the white one shrunk seventeen-fold, though $X$ and $Y$ are still strongly correlated. At degree 1 it would be 0.7 times the white one.",
        0,
        8,
        0,
        0.7,
    ),
    ("negative correlation", r"$\rho = -0.7$ and degree 3: $T h_3 = (-0.7)^3 h_3 = -0.343\, h_3$, the white curve turned upside down and shrunk. An even degree would keep its sign.", 0, 3, 0, -0.7),
    (
        "ReLU, smoothed",
        r"The kink is smoothed away: $T$ multiplies the $n$-th Hermite coefficient of $f$ by $\rho^n$. So $\mathbb{E}[f(X)\, f(Y)] = c_0^2 + c_1^2 \rho + c_2^2 \rho^2 + \cdots$ reads 0.3750 at $\rho = 0.7$: the kernel of one layer of random ReLU neurons, in the limit of infinite width (standard normal weights, no biases, and two inputs of length 1 at an angle whose cosine is $\rho$).",
        1,
        3,
        0,
        0.7,
    ),
];

pub struct HermiteNoise {
    single: usize,
    n: usize,
    f: ExprField,
    rho: f32,
    x0: f32,
    coef: Vec<f64>,
    /// E[f(X)²] and E[f(X) f(-X)]: the two ends of the kernel (the sum of the
    /// squared coefficients stops at degree 40, which for a function with a jump
    /// falls well short of them).
    square: f64,
    anti: f64,
    /// A Gauss-Hermite rule (exact for the polynomials) and a fine trapezoid rule
    /// (which copes with kinks and jumps), for averaging over a Gaussian.
    nodes: (Vec<f64>, Vec<f64>),
    fine: (Vec<f64>, Vec<f64>),
    dirty: bool,
    /// The smoothed curve across the view, and the (ρ, left edge, right edge) it was
    /// worked out for: it is a quarter of a million evaluations of a typed function,
    /// and nothing about it changes while only the handle moves.
    smooth: Vec<f32>,
    smooth_for: Option<[f32; 3]>,
}

/// Points and weights of the trapezoid rule for a standard normal average over
/// -6..6. Where the function averaged has a jump, a rule built on polynomials
/// (Gauss-Hermite) gives a staircase; this gives the smooth curve it should.
fn trapezoid_rule(points: usize) -> (Vec<f64>, Vec<f64>) {
    let step = 12.0 / (points - 1) as f64;
    let z: Vec<f64> = (0..points).map(|i| -6.0 + step * i as f64).collect();
    let mut w: Vec<f64> = z.iter().map(|&z| phi(z) * step).collect();
    w[0] *= 0.5;
    w[points - 1] *= 0.5;
    let total: f64 = w.iter().sum();
    w.iter_mut().for_each(|w| *w /= total);
    (z, w)
}

impl HermiteNoise {
    pub fn new(_: &Gfx) -> Self {
        Self::start()
    }

    fn start() -> Self {
        HermiteNoise {
            single: 0,
            n: 3,
            f: ExprField::new("relu(x)", &["x"]),
            rho: 0.7,
            x0: 1.6,
            coef: Vec::new(),
            square: 1.0,
            anti: 0.0,
            nodes: gauss_hermite(96),
            fine: trapezoid_rule(601),
            dirty: true,
            smooth: Vec::new(),
            smooth_for: None,
        }
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let (_, _, single, n, function, rho) = TRIALS[k];
        (self.single, self.n, self.rho, self.x0) = (single, n, rho, 1.6);
        if single == 1 {
            self.f.set(EXPANSION_PRESETS[function].1);
        }
        self.update();
    }

    fn eval(&self, x: f64) -> f64 {
        if self.single == 0 {
            h(self.n, x)
        } else {
            let v = self.f.expr.eval(&[x as f32]) as f64;
            if v.is_finite() {
                v
            } else {
                0.0
            }
        }
    }

    /// The noise operator: the average of f over a Gaussian centred at rho·x with
    /// variance 1 - rho², i.e. E[f(Y) | X = x] for rho-correlated standard normals.
    fn smoothed(&self, x: f64) -> f64 {
        let rho = self.rho as f64;
        let s = (1.0 - rho * rho).max(0.0).sqrt();
        // (a typed function may have kinks and jumps; a Hermite polynomial has none)
        let (z, w) = if self.single == 0 { &self.nodes } else { &self.fine };
        z.iter().zip(w).map(|(z, w)| w * self.eval(rho * x + s * z)).sum()
    }

    fn update(&mut self) {
        let this = &*self;
        let c = coefficients(|x| this.eval(x), EXP_NMAX);
        let square = gauss_mean(|x| this.eval(x).powi(2));
        let anti = gauss_mean(|x| this.eval(x) * this.eval(-x));
        (self.coef, self.square, self.anti) = (c, square, anti);
        self.dirty = false;
        self.smooth_for = None;
    }

    /// E[f(X) f(Y)] for correlation r: the sum of c_n² r^n, and at r = ±1 (where
    /// the sum of forty terms may still be short) the exact value.
    fn kernel(&self, r: f64) -> f64 {
        if r >= 1.0 {
            return self.square;
        }
        if r <= -1.0 {
            return self.anti;
        }
        let mut p = 1.0;
        let mut acc = 0.0;
        for c in &self.coef {
            acc += c * c * p;
            p *= r;
        }
        acc
    }
}

impl Scene for HermiteNoise {
    fn view(&self) -> View {
        View::Fit { center: Vec2::new(0.0, 0.4), half: Vec2::new(5.0, 3.0) }
    }

    fn animated(&self) -> bool {
        false
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.18;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Take two standard normal variables $X$ and $Y$ with correlation $\rho$. Knowing $X = x$, the other one is spread in a bump around $\rho x$ (its standard deviation is $\sqrt{1 - \rho^2}$). Averaging a function over that bump is the noise operator $T$. For a Hermite polynomial the average is the same polynomial, just scaled:");
        ui::equation(ui, r"T h_n = \rho^n \, h_n");
        ui::note(ui, "Nothing else is needed to see how any function decorrelates.");
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }
        ui::section(ui, "Function");
        if ui::choice(ui, &["one Hermite polynomial", "any function"], &mut self.single) {
            self.dirty = true;
        }
        if self.single == 0 {
            self.dirty |= ui::slider_int(ui, "degree $n$", &mut self.n, 0..=10);
        } else {
            self.dirty |= self.f.ui(ui, "f(x) =");
            let labels: Vec<&str> = EXPANSION_PRESETS.iter().map(|p| p.0).collect();
            if let Some(i) = ui::buttons(ui, &labels) {
                self.f.set(EXPANSION_PRESETS[i].1);
                self.dirty = true;
            }
        }
        if self.dirty {
            self.update();
        }
        ui::section(ui, "Correlation");
        ui::slider(ui, r"$\rho$", &mut self.rho, -1.0..=1.0);
        ui::readout(ui, r"angle $\theta$, where $\cos\theta = \rho$", format!("{:.1}°", (self.rho as f64).clamp(-1.0, 1.0).acos().to_degrees()));
        ui::note(ui, "Drag the yellow point along the axis to move $x$.");
        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::WHITE, "wide, faint white: $f$"),
                (pal::YELLOW, "yellow: the point $x$, the bump where $Y$ lands once $X = x$ is known, and the stretch of $f$ it averages"),
                (pal::CYAN, "cyan: $T f$, that average for every $x$; the dot is the one for the yellow point"),
                (pal::GREEN, r"green inset: $\mathbb{E}[f(X)\, f(Y)]$ for every $\rho$ from $-1$ to $1$"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, r"Correlation $\rho$:", "how closely $X$ and $Y$ move together: $1$, always equal; $0$, unrelated; $-1$, always opposite.");
        ui::term(ui, "Kernel:", r"$\mathbb{E}[f(X)\, f(Y)]$ as a function of $\rho$: how alike $f$'s values are at two points correlated that much.");
        ui::section(ui, "Numbers");
        if self.single == 0 {
            ui::readout_colored(ui, &format!(r"$\rho^{{{}}}$", self.n), format!("${}$", signed((self.rho as f64).powi(self.n as i32) as f32, 4)), pal::CYAN);
            ui::note(ui, "The cyan curve is the white one multiplied by that number.");
        }
        let k1 = self.square.max(1e-300);
        ui::readout(ui, r"$\mathbb{E}[f(X)\, f(Y)]$", format!("${}$", signed(self.kernel(self.rho as f64) as f32, 4)));
        ui::readout(ui, r"…as a fraction of $\mathbb{E}[f(X)^2]$", format!("${}$", signed((self.kernel(self.rho as f64) / k1) as f32, 4)));
        ui::note(ui, r"Since $T h_n = \rho^n h_n$ and the $h_n$ are orthonormal,");
        ui::equation(ui, r"\mathbb{E}[f(X)\, f(Y)] = c_0^2 + c_1^2 \rho + c_2^2 \rho^2 + \cdots");
        ui::note(ui, r"This is the kernel that one hidden layer of random neurons computes for two inputs of length 1 at the angle whose cosine is $\rho$, in the limit of infinite width (standard normal weights, no biases); for ReLU it is the arc-cosine kernel. High degrees are the first to be forgotten.");
        ui::see_also(ui, &[("hermite_expand", "the coefficients $c_n$ that $T$ scales"), ("hermite_dirs", "the same overlap along two directions in the plane"), ("gauss2", "two correlated normal variables, as a cloud of points")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        if self.dirty {
            self.update();
        }
        let (lo, hi) = f.cam.flat_bounds();
        let span = hi.y - lo.y;
        let rho = self.rho as f64;
        let s = (1.0 - rho * rho).max(0.0).sqrt();
        let visible = |y: f64| {
            let y = y as f32;
            if y.is_finite() && y > lo.y - span * 2.0 && y < hi.y + span * 2.0 {
                y
            } else {
                f32::NAN
            }
        };
        let samples = 420;
        let across = |i: usize| lo.x + (hi.x - lo.x) * i as f32 / samples as f32;
        f.curve(lo.x, hi.x, samples, 4.5, pal::WHITE.a(0.42), |x| Vec3::new(x, visible(self.eval(x as f64)), 0.0));
        if self.smooth_for != Some([self.rho, lo.x, hi.x]) {
            self.smooth_for = Some([self.rho, lo.x, hi.x]);
            self.smooth = (0..=samples).map(|i| self.smoothed(across(i) as f64) as f32).collect();
        }
        f.path(self.smooth.iter().enumerate().map(|(i, &y)| Vec3::new(across(i), visible(y as f64), 0.0)), 2.8, pal::CYAN);

        // the conditional distribution of Y given X = x0, and the stretch of f it averages
        let mut handle = Vec2::new(self.x0, 0.0);
        if f.drag2(1, &mut handle, pal::YELLOW) {
            self.x0 = handle.x.clamp(-6.0, 6.0);
        }
        let x0 = self.x0 as f64;
        let centre = rho * x0;
        let hump = span * 0.16;
        if s > 1e-3 {
            let bell = (0..=120).map(|i| centre - 4.0 * s + 8.0 * s * i as f64 / 120.0).map(|y| Vec2::new(y as f32, (-0.5 * ((y - centre) / s).powi(2)).exp() as f32 * hump));
            fill_under(f, bell, 0.0, pal::YELLOW.a(0.3));
            f.curve((centre - 2.0 * s) as f32, (centre + 2.0 * s) as f32, 80, 4.0, pal::YELLOW, |x| Vec3::new(x, visible(self.eval(x as f64)), 0.0));
        }
        let avg = self.smoothed(x0);
        let tip = Vec2::new(self.x0, visible(avg));
        if tip.y.is_finite() {
            f.dashed((self.x0, 0.0), tip, 1.4, pal::CYAN.a(0.9));
            f.glow(tip, 22.0, pal::CYAN);
            f.dot(tip, 10.0, pal::CYAN);
            // on a plate: this label usually lies over the yellow bump and the curves
            if let Some(p) = on_screen(f, tip.extend(0.0)) {
                let text = format!(r"$\text{{average}} = {}$", fmt(avg as f32));
                // (moved left, where the right-hand edge of a phone's view would cut it off)
                let x = (p.x + 14.0).min(f.view_size().x - f.text_size(&text, 13.0).x - 12.0);
                f.caption_px(Vec2::new(x, p.y - 12.0), text, 13.0, pal::CYAN, Align2::LEFT_BOTTOM);
            }
        }
        f.text_off((centre as f32, hump), Vec2::new(0.0, -6.0), format!("$Y$ given $X = {:.2}$", self.x0), 12.5, pal::YELLOW, Align2::CENTER_BOTTOM);
        // up and to the left of the handle: below it are the dashed line and the cyan dot
        f.text_off((self.x0, 0.0), Vec2::new(-11.0, -7.0), "$x$", 13.0, pal::YELLOW, Align2::RIGHT_BOTTOM);

        // inset: the kernel E[f(X) f(Y)] as a function of the correlation
        let size = hi - lo;
        let (w, hgt) = (size.x * 0.27, size.y * 0.24);
        let o = Vec2::new(hi.x - w - size.x * 0.04, lo.y + size.y * 0.09);
        inset(f, o - Vec2::splat(size.y * 0.02), o + Vec2::new(w, hgt) + Vec2::splat(size.y * 0.02));
        let k1 = self.square.max(1e-300);
        let at = |r: f64, v: f64| o + Vec2::new(w * (r as f32 + 1.0) * 0.5, hgt * (0.5 + 0.5 * (v / k1).clamp(-1.0, 1.0) as f32));
        // the inset runs from −E[f²] to E[f²] (k1), whatever size that is
        f.line(at(-1.0, 0.0), at(1.0, 0.0), 1.0, pal::DIM.a(0.8));
        f.line(at(0.0, -k1), at(0.0, k1), 1.0, pal::DIM.a(0.8));
        f.path((0..=120).map(|i| -1.0 + 2.0 * i as f64 / 120.0).map(|r| at(r, self.kernel(r))), 2.2, pal::GREEN);
        let here = at(rho, self.kernel(rho));
        f.line(at(rho, -k1), at(rho, k1), 1.0, pal::WHITE.a(0.4));
        f.dot(here, 9.0, pal::GREEN);
        // as much of the title as there is room for, from the inset's left edge to the
        // view's (on a phone it is narrow)
        let room = (hi.x - o.x) * pixels_per_unit(f) - 8.0;
        let title = [r"$\mathbb{E}[f(X)\, f(Y)]$ against $\rho$", r"$\mathbb{E}[f(X)\, f(Y)]$"].into_iter().find(|t| f.text_size(t, 12.0).x <= room).unwrap_or("");
        f.text(o + Vec2::new(0.0, hgt + size.y * 0.006), title, 12.0, pal::GREEN, Align2::LEFT_BOTTOM);
        f.text(at(-1.0, -k1), "−1", 11.0, pal::DIM, Align2::LEFT_TOP);
        f.text(at(1.0, -k1), "1", 11.0, pal::DIM, Align2::RIGHT_TOP);
        if rho.abs() < 0.8 {
            f.text(at(rho, -k1), r"$\rho$", 11.5, pal::TEXT, Align2::CENTER_TOP);
        }
        f.text_off(at(0.0, k1), Vec2::new(5.0, 2.0), r"$\mathbb{E}[f^2]$", 11.0, pal::DIM, Align2::LEFT_TOP);
        f.text_off(at(0.0, -k1), Vec2::new(5.0, -2.0), r"$-\mathbb{E}[f^2]$", 11.0, pal::DIM, Align2::LEFT_BOTTOM);

        // tick numbers first, so that the plates of the title and the caption lie over them
        f.axes2();
        let view = f.view_size();
        let title = if self.single == 0 {
            format!(r"$T h_{{{0}}} = \rho^{{{0}}} h_{{{0}}}$        $\rho = {1:.2}$,   $\rho^{{{0}}} = {2:+.3}$", self.n, self.rho, rho.powi(self.n as i32))
        } else {
            format!(r"$T f(x) = \mathbb{{E}}[f(Y) \mid X = x]$        $\rho = {:.2}$", self.rho)
        };
        f.caption_px(Vec2::new(view.x * 0.5, f.top_inset + 18.0), title, 16.0, pal::CYAN, Align2::CENTER_TOP);
        bottom_caption(f, &["white: $f$", "cyan: $T f$", "drag the yellow point"], 12.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The noise operator on a function with a jump: T sign is smooth (it is
    /// 2 Φ(ρ x / s) - 1), where the Gauss-Hermite rule used to draw a staircase with
    /// steps of a quarter.
    #[test]
    fn smoothing_a_jump_gives_a_smooth_curve() {
        let mut scene = HermiteNoise::start();
        scene.single = 1;
        scene.f.set("sign(x)");
        scene.update();
        let (rho, s) = (0.7f64, (1.0f64 - 0.49).sqrt());
        // the exact value, by brute force: the mass of the bump right of zero, less the mass left of it
        let exact = |x: f64| {
            let n = 200_000;
            let step = 16.0 / n as f64;
            (0..n).map(|i| -8.0 + step * (i as f64 + 0.5)).map(|z| phi(z) * step * if rho * x + s * z > 0.0 { 1.0 } else { -1.0 }).sum::<f64>()
        };
        let mut previous = scene.smoothed(-3.0);
        for i in 0..=120 {
            let x = -3.0 + 0.05 * i as f64;
            let t = scene.smoothed(x);
            assert!((t - exact(x)).abs() < 0.012, "T sign at {x}: {t} against {}", exact(x));
            assert!((t - previous).abs() < 0.06, "a step of {} at {x}", t - previous);
            previous = t;
        }
        // E[f²] is 1 for sign, and E[sign(X) sign(Y)] = (2 / π) asin ρ
        assert!((scene.square - 1.0).abs() < 1e-3 && (scene.anti + 1.0).abs() < 1e-3);
        let share = scene.kernel(rho) / scene.square;
        assert!((share - 2.0 / std::f64::consts::PI * rho.asin()).abs() < 2e-3, "{share}");
        assert_eq!(scene.kernel(1.0), scene.square);
    }

    /// For a Hermite polynomial nothing changed: T hₙ = ρⁿ hₙ, to rounding.
    #[test]
    fn a_hermite_polynomial_is_only_scaled() {
        let mut scene = HermiteNoise::start();
        scene.update();
        for x in [-2.0, -0.3, 0.0, 1.6] {
            assert!((scene.smoothed(x) - (scene.rho as f64).powi(3) * h(3, x)).abs() < 1e-9);
        }
        assert!((scene.square - 1.0).abs() < 1e-6);
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let mut scene = HermiteNoise::start();
        let xs = [-2.5, -1.0, 0.0, 0.4, 1.6, 3.0];

        // degree 8 at ρ = 0.7: the cyan curve is 0.058 times the white one, a seventeenth
        scene.set_up(0);
        assert!(scene.single == 0 && scene.n == 8 && scene.rho == 0.7);
        // (ρ as the slider holds it, in single precision)
        let r8 = (scene.rho as f64).powi(8);
        assert!((r8 - 0.058).abs() < 5e-4 && (1.0 / r8).round() == 17.0);
        assert!(xs.iter().all(|&x| (scene.smoothed(x) - r8 * h(8, x)).abs() < 1e-9));

        // ρ = −0.7, degree 3: upside down and shrunk by 0.343; an even degree keeps its sign
        scene.set_up(1);
        assert!(scene.single == 0 && scene.n == 3 && scene.rho == -0.7);
        let rho = scene.rho as f64;
        assert!((rho.powi(3) + 0.343).abs() < 1e-6);
        assert!(xs.iter().all(|&x| (scene.smoothed(x) - rho.powi(3) * h(3, x)).abs() < 1e-9));
        scene.n = 2;
        scene.update();
        assert!(rho.powi(2) > 0.0 && xs.iter().all(|&x| (scene.smoothed(x) - rho.powi(2) * h(2, x)).abs() < 1e-9));

        // ReLU: T scales the n-th coefficient by ρⁿ, and the kernel is the arc-cosine one
        scene.set_up(2);
        assert!(scene.single == 1 && scene.rho == 0.7 && scene.eval(1.5) == 1.5 && scene.eval(-1.0) == 0.0);
        let rho = scene.rho as f64;
        // the coefficients of the cyan curve, by the trapezoid rule over a fine grid
        let (points, reach) = (1601, 8.0);
        let dx = 2.0 * reach / (points - 1) as f64;
        let smooth: Vec<(f64, f64)> = (0..points).map(|i| -reach + dx * i as f64).map(|x| (x, scene.smoothed(x))).collect();
        for n in 0..=6 {
            let c = smooth.iter().map(|&(x, t)| t * h(n, x) * phi(x) * dx).sum::<f64>();
            assert!((c - rho.powi(n as i32) * scene.coef[n]).abs() < 1e-6, "degree {n}: {c} against {}", rho.powi(n as i32) * scene.coef[n]);
        }
        let theta = rho.acos();
        let arc_cosine = (theta.sin() + (std::f64::consts::PI - theta) * rho) / (2.0 * std::f64::consts::PI);
        assert!((scene.kernel(rho) - arc_cosine).abs() < 1e-6, "{} against {arc_cosine}", scene.kernel(rho));
        assert_eq!(signed(scene.kernel(rho) as f32, 4), "+0.3750");
        // and it is what a wide layer of random ReLU neurons computes, under the
        // caption's conditions: standard normal weights, no biases, two inputs of
        // length 1 at the angle whose cosine is ρ; the average over the neurons of the
        // product of their two outputs
        let (a, b) = ((1.0, 0.0), (rho, (1.0 - rho * rho).sqrt()));
        let (mut rng, width) = (Rng::new(5), 400_000);
        let wide = (0..width)
            .map(|_| {
                let w = (rng.normal() as f64, rng.normal() as f64);
                (w.0 * a.0 + w.1 * a.1).max(0.0) * (w.0 * b.0 + w.1 * b.1).max(0.0)
            })
            .sum::<f64>()
            / width as f64;
        assert!((wide - scene.kernel(rho)).abs() < 0.004, "{width} neurons: {wide}");
    }
}

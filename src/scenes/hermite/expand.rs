//! Hermite expansions.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "hermite_expand",
    title: "Hermite expansions",
    group: "Hermite polynomials",
    blurb: "Any function as a sum of Hermite polynomials, degree by degree.",
    make: |g| Box::new(HermiteExpand::new(g)),
};

/// Experiments: (button, what to look for, function as an index into
/// `EXPANSION_PRESETS`, highest degree kept).
const TRIALS: [(&str, &str, usize, usize); 3] = [
    (
        "ReLU: three numbers",
        r"$c_0 = 1/\sqrt{2\pi} = 0.399$, $c_1 = 1/2$, $c_2 = 1/(2\sqrt{\pi}) = 0.282$: with $N = 2$ these already capture 97.7 % of $\mathbb{E}[f^2] = 1/2$. Every odd degree above 1 is zero, because $\operatorname{ReLU}(x) = x/2 + |x|/2$.",
        0,
        2,
    ),
    ("$x^3$: only two bars", r"$x^3 = 3 h_1 + \sqrt{6}\, h_3$: two bars, and from $N = 3$ on the cyan curve is $x^3$ exactly (100 %).", 6, 3),
    (
        "sign: a long, slow tail",
        r"A jump needs every odd degree, and the bars shrink slowly: degrees up to 9 capture 84 % of $\mathbb{E}[f^2]$, all forty only 92 %. ($\tanh$, a smooth step, has over 99.9 % by degree 9.)",
        2,
        9,
    ),
];

pub struct HermiteExpand {
    f: ExprField,
    terms: usize,
    coef: Vec<f64>,
    norm2: f64,
    show_term: bool,
    dirty: bool,
}

impl HermiteExpand {
    pub fn new(_: &Gfx) -> Self {
        Self::create()
    }

    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        HermiteExpand { f: ExprField::new("relu(x)", &["x"]), terms: 6, coef: Vec::new(), norm2: 1.0, show_term: false, dirty: true }
    }

    /// The share of E[f²] that degrees 0 ..= `terms` capture (Parseval).
    fn captured(&self, terms: usize) -> f64 {
        let kept: f64 = self.coef.iter().take(terms + 1).map(|c| c * c).sum();
        kept / self.norm2.max(1e-300)
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let (_, _, function, terms) = TRIALS[k];
        self.f.set(EXPANSION_PRESETS[function].1);
        (self.terms, self.show_term) = (terms, false);
        self.update();
    }

    fn eval(&self, x: f64) -> f64 {
        self.f.expr.eval(&[x as f32]) as f64
    }

    fn update(&mut self) {
        let e = &self.f.expr;
        let f = |x: f64| {
            let v = e.eval(&[x as f32]) as f64;
            if v.is_finite() {
                v
            } else {
                0.0
            }
        };
        let (coef, norm2) = (coefficients(f, EXP_NMAX), gauss_mean(|x| f(x) * f(x)));
        (self.coef, self.norm2) = (coef, norm2);
        self.dirty = false;
    }

    /// The lowest degree n ≥ 1 present: the first coefficient that is not zero, up to
    /// the accuracy the coefficients are worked out to.
    fn lowest(&self) -> Option<usize> {
        (1..=EXP_NMAX).find(|&k| self.coef.get(k).is_some_and(|c| c.abs() > 1e-5 * self.norm2.sqrt().max(1e-9)))
    }

    fn partial(&self, x: f64, terms: usize, hs: &mut [f64]) -> f64 {
        h_all(x, hs);
        self.coef.iter().zip(hs.iter()).take(terms + 1).map(|(c, h)| c * h).sum()
    }
}

impl Scene for HermiteExpand {
    fn view(&self) -> View {
        View::Fit { center: Vec2::new(0.0, 0.7), half: Vec2::new(5.0, 2.9) }
    }

    fn animated(&self) -> bool {
        false
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.18;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "Because the $h_n$ are orthonormal, any function with finite Gaussian mean square is a sum of them, and each coefficient is just an average:");
        ui::equation(ui, r"f = \sum_{n=0}^{\infty} c_n h_n, \qquad c_n = \mathbb{E}[f(X)\, h_n(X)]");
        ui::note(ui, r#"Cutting the sum at degree $N$ gives the best degree-$N$ polynomial approximation of $f$, where "best" is measured by the Gaussian: errors count where a normal variable actually lands."#);
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }
        ui::section(ui, "The function");
        if self.f.ui(ui, "f(x) =") {
            self.dirty = true;
        }
        let labels: Vec<&str> = EXPANSION_PRESETS.iter().map(|p| p.0).collect();
        if let Some(i) = ui::buttons(ui, &labels) {
            self.f.set(EXPANSION_PRESETS[i].1);
            self.dirty = true;
        }
        if self.dirty {
            self.update();
        }
        ui::section(ui, "Truncation");
        ui::slider_int(ui, "highest degree $N$", &mut self.terms, 0..=EXP_NMAX);
        ui.checkbox(&mut self.show_term, "show the last term on its own");
        ui::section(ui, "Reading the picture");
        let mut key = vec![
            (pal::WHITE, "wide, faint white: $f$"),
            (pal::CYAN, "cyan: its expansion up to degree $N$; in the inset, the bars of the degrees kept"),
            (pal::ORANGE, "orange shading: the error, between the two"),
            (pal::PURPLE, r"purple: the Gaussian weight $\varphi(x)$, where the error counts"),
        ];
        if self.show_term {
            key.push((pal::YELLOW, "yellow: the last term, $c_N h_N$, on its own"));
        }
        key.push((pal::FAINT, "grey bars: the degrees left out"));
        ui::key(ui, &key);
        ui::section(ui, "Words");
        ui::term(ui, "Coefficient $c_n$:", r"$\mathbb{E}[f(X)\, h_n(X)]$, how much of $h_n$ there is in $f$.");
        ui::term(ui, "Parseval:", r"$\mathbb{E}[f^2] = c_0^2 + c_1^2 + c_2^2 + \cdots$, so the squares of the bars are how $\mathbb{E}[f^2]$ is shared out among degrees.");
        ui::term(ui, "Information exponent:", r"the lowest degree $n \ge 1$ with $c_n$ not zero. In work on learning single-index models it governs how many samples gradient descent needs.");
        ui::section(ui, "Numbers");
        let kept: f64 = self.coef.iter().take(self.terms + 1).map(|c| c * c).sum();
        ui::readout(ui, r"$\mathbb{E}[f(X)^2]$", format!("${}$", fmt(self.norm2 as f32)));
        ui::readout_colored(ui, r"captured by degrees $n \le N$", format!("{:.2} %", 100.0 * self.captured(self.terms)), pal::CYAN);
        ui::readout_colored(ui, "root-mean-square error", format!("${}$", fmt((self.norm2 - kept).max(0.0).sqrt() as f32)), pal::ORANGE);
        ui::readout(ui, r"lowest degree present ($n \ge 1$)", self.lowest().map_or("none".into(), |k| format!("${k}$")));
        for k in 0..=5.min(EXP_NMAX) {
            ui::readout(ui, &format!("$c_{{{k}}}$"), format!("${}$", signed(self.coef[k] as f32, 5)));
        }
        ui::note(ui, "For an $f$ with $n$ derivatives, none growing faster than a power of $x$, integrating by parts $n$ times gives");
        ui::equation(ui, r"c_n = \frac{\mathbb{E}[f^{(n)}(X)]}{\sqrt{n!}}");
        ui::see_also(ui, &[("hermite_noise", "what correlated noise does to each coefficient"), ("fit1d", "building a curve from neurons instead"), ("fourier", "another curve built from simple waves")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        if self.dirty {
            self.update();
        }
        let (lo, hi) = f.cam.flat_bounds();
        let span = hi.y - lo.y;
        weight_hump(f, 0.0, span * 0.2 / 0.3989, 1.0, 0.0, pal::PURPLE);
        // beside the axis rather than on it, where the tick numbers are
        if let Some(p) = on_screen(f, Vec3::new(0.0, span * 0.2, 0.0)) {
            weight_caption(f, p);
        }

        let samples = 600usize;
        let mut hs = vec![0.0f64; EXP_NMAX + 1];
        let xs: Vec<f32> = (0..=samples).map(|i| lo.x + (hi.x - lo.x) * i as f32 / samples as f32).collect();
        let fx: Vec<f32> = xs.iter().map(|&x| self.eval(x as f64) as f32).collect();
        let sx: Vec<f32> = xs.iter().map(|&x| self.partial(x as f64, self.terms, &mut hs) as f32).collect();
        let clamp = |y: f32| y.clamp(lo.y - span, hi.y + span);
        for i in 0..samples {
            if fx[i].is_finite() && fx[i + 1].is_finite() && sx[i].is_finite() && sx[i + 1].is_finite() {
                f.dl.tris_alpha.quad(
                    Vec3::new(xs[i], clamp(fx[i]), 0.0),
                    Vec3::new(xs[i + 1], clamp(fx[i + 1]), 0.0),
                    Vec3::new(xs[i + 1], clamp(sx[i + 1]), 0.0),
                    Vec3::new(xs[i], clamp(sx[i]), 0.0),
                    pal::ORANGE.a(0.16),
                );
            }
        }
        let visible = |y: f32| if y.is_finite() && y > lo.y - span * 2.0 && y < hi.y + span * 2.0 { y } else { f32::NAN };
        // (the two curves from the samples already taken for the shading)
        f.path(xs.iter().zip(&fx).map(|(&x, &y)| Vec3::new(x, visible(y), 0.0)), 5.0, pal::WHITE.a(0.4));
        if self.show_term {
            let n = self.terms;
            let c = self.coef[n];
            f.curve(lo.x, hi.x, samples, 1.6, pal::YELLOW, |x| Vec3::new(x, visible((c * h(n, x as f64)) as f32), 0.0));
        }
        f.path(xs.iter().zip(&sx).map(|(&x, &y)| Vec3::new(x, visible(y), 0.0)), 2.8, pal::CYAN);

        // the spectrum: how much of E[f²] sits in each degree
        let size = hi - lo;
        let (w, hgt) = (size.x * 0.36, size.y * 0.2);
        let o = Vec2::new(hi.x - w - size.x * 0.035, hi.y - hgt - size.y * 0.13);
        inset(f, o - Vec2::splat(size.y * 0.02), o + Vec2::new(w, hgt) + Vec2::splat(size.y * 0.02));
        let peak = self.coef.iter().map(|c| c * c).fold(1e-300, f64::max);
        let bw = w / (EXP_NMAX + 1) as f32;
        for (k, c) in self.coef.iter().enumerate() {
            // square-root scale, so the long tail of small terms stays visible
            let v = ((c * c / peak).sqrt() as f32) * hgt;
            let col = if k <= self.terms { pal::CYAN } else { pal::FAINT };
            if v > 0.0 {
                f.dl.tris_alpha.rect(o + Vec2::new(k as f32 * bw + bw * 0.12, 0.0), o + Vec2::new((k + 1) as f32 * bw - bw * 0.12, v), 0.0, col.a(0.9));
            }
        }
        f.line(o, o + Vec2::new(w, 0.0), 1.0, pal::DIM);
        for k in [0, 10, 20, 30, 40] {
            f.text(o + Vec2::new((k as f32 + 0.5) * bw, -size.y * 0.004), format!("{k}"), 11.0, pal::DIM, Align2::CENTER_TOP);
        }
        // as much of the title as the inset is wide (on a phone it is narrow)
        let inset_px = w * pixels_per_unit(f);
        let title = ["$|c_n|$ by degree $n$, relative to the largest", "$|c_n|$ by degree $n$"].into_iter().find(|t| text_width(t, 12.0) <= inset_px).unwrap_or("");
        f.text(o + Vec2::new(0.0, hgt + size.y * 0.006), title, 12.0, pal::TEXT, Align2::LEFT_BOTTOM);
        // the key sits to the right of the tall low-degree bars, on two lines if it must
        let room = inset_px * 0.76;
        let second = if 122.0 + text_width("grey: left out", 11.5) <= room { Vec2::new(122.0, 4.0) } else { Vec2::new(0.0, 20.0) };
        f.text_off(o + Vec2::new(w * 0.24, hgt), Vec2::new(0.0, 4.0), r"cyan: kept ($n \le N$)", 11.5, pal::CYAN, Align2::LEFT_TOP);
        f.text_off(o + Vec2::new(w * 0.24, hgt), second, "grey: left out", 11.5, pal::DIM, Align2::LEFT_TOP);

        // tick numbers first, so that the plates of the title and the caption lie over them
        f.axes2();
        f.caption_px(
            Vec2::new(18.0, f.top_inset + 18.0),
            format!(r"degrees 0 … {}   capture {:.2} % of $\mathbb{{E}}[f^2]$", self.terms, 100.0 * self.captured(self.terms)),
            16.0,
            pal::CYAN,
            Align2::LEFT_TOP,
        );
        let mut key = vec!["white: $f$", "cyan: its Hermite expansion", "shaded: the error, which only matters under the purple hump"];
        if self.show_term {
            key.push("yellow: the last term alone");
        }
        bottom_caption(f, &key, 12.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expanded(formula: &str) -> HermiteExpand {
        let mut scene = HermiteExpand::create();
        scene.f.set(formula);
        scene.update();
        scene
    }

    /// The numbers on show when the scene opens: ReLU's coefficients and E[f²].
    #[test]
    fn the_relu_coefficients_are_the_known_ones() {
        let mut scene = HermiteExpand::create();
        scene.update();
        let pi = std::f64::consts::PI;
        let c = &scene.coef;
        assert!((c[0] - 1.0 / (2.0 * pi).sqrt()).abs() < 1e-6 && (c[1] - 0.5).abs() < 1e-6 && (c[2] - 1.0 / (2.0 * pi.sqrt())).abs() < 1e-6, "{:?}", &c[..3]);
        assert!((scene.norm2 - 0.5).abs() < 1e-6);
        assert_eq!(scene.lowest(), Some(1));
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let mut scene = HermiteExpand::create();

        // ReLU: three numbers capture 97.7 %, and no odd degree above 1
        scene.set_up(0);
        assert!(scene.f.expr.eval(&[1.5]) == 1.5 && scene.f.expr.eval(&[-1.5]) == 0.0 && scene.terms == 2);
        assert!((scene.captured(2) - 0.977).abs() < 5e-4, "{}", scene.captured(2));
        assert!((3..=EXP_NMAX).step_by(2).all(|n| scene.coef[n].abs() < 1e-6));
        assert!(scene.coef[4].abs() > 0.05, "the even degrees go on: {}", scene.coef[4]);

        // x³: two bars, and exact from N = 3 on
        scene.set_up(1);
        assert!(scene.terms == 3 && scene.f.expr.eval(&[2.0]) == 8.0);
        assert!((scene.coef[1] - 3.0).abs() < 1e-7 && (scene.coef[3] - 6f64.sqrt()).abs() < 1e-7);
        let peak = scene.coef[1].abs();
        let bars = scene.coef.iter().filter(|c| c.abs() > 1e-5 * peak).count();
        assert_eq!(bars, 2);
        assert!((scene.captured(3) - 1.0).abs() < 1e-6 && (scene.captured(2) - 0.6).abs() < 1e-6);

        // sign: every odd degree, shrinking slowly; 84 % by degree 9, 92 % by forty
        scene.set_up(2);
        assert!(scene.terms == 9 && scene.f.expr.eval(&[-0.2]) == -1.0);
        // (the bars shrink like n^(-3/4): c₁ = 0.80, c₉ = 0.14, c₃₉ = 0.046)
        assert!((1..EXP_NMAX).step_by(2).all(|n| scene.coef[n].abs() > 0.04), "every odd degree");
        assert!((scene.coef[39].abs() - 0.0458).abs() < 1e-3 && (scene.coef[9].abs() - 0.1391).abs() < 1e-3);
        assert!((0..=EXP_NMAX).step_by(2).all(|n| scene.coef[n].abs() < 1e-9), "and no even one");
        assert!((scene.captured(9) - 0.84).abs() < 5e-3, "{}", scene.captured(9));
        assert!((scene.captured(EXP_NMAX) - 0.92).abs() < 5e-3, "{}", scene.captured(EXP_NMAX));
        // the comparison in brackets: a smooth step
        let tanh = expanded("tanh(x)");
        assert!(tanh.captured(9) > 0.999, "{}", tanh.captured(9));
    }

    /// "Lowest degree present": the first coefficient that is not zero.
    #[test]
    fn the_lowest_degree_present() {
        for (formula, lowest) in [("x^3", Some(1)), ("abs(x)", Some(2)), ("x^3 - 3x", Some(3)), ("x^4 - 6 x^2 + 3", Some(4)), ("1", None)] {
            assert_eq!(expanded(formula).lowest(), lowest, "{formula}");
        }
    }
}

//! Least-squares fitting.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "regress",
    title: "Least-squares fitting",
    group: MODELS,
    blurb: "Squares you can see, and what happens when a model is too flexible.",
    make: |g| Box::new(Regress::new(g)),
};

const MAX_DEG: usize = 12;

/// Experiments: (button, what to look for, degree, ridge penalty), all on the same
/// sample of fourteen points from the wave (the one the scene opens with).
const TRIALS: [(&str, &str, usize, f32); 3] = [
    ("too stiff", "Degree 1: a straight line cannot follow the wave. Its error is large both on the data and against the true curve (about 0.4 and 0.5, against 0.14 and 0.17 for degree 3).", 1, 0.0),
    ("too flexible", "Degree 12: the curve hugs the points (its error on the data, 0.05, is the smallest of any degree) but swings far from the true curve, most of all at the ends (error about 17).", 12, 0.0),
    ("ridge calms it", r"Degree 12 again, with $\lambda = 0.01$: the swings are gone, and the error against the true curve falls from about 17 to 0.15, as low as degree 3's.", 12, 0.01),
];

pub struct Regress {
    pts: Vec<Vec2>,
    degree: usize,
    ridge: f32,
    noise: f32,
    truth: usize,
    count: usize,
    show_squares: bool,
    show_truth: bool,
    rng: Rng,
    errs: Vec<(f32, f32)>,
    dirty: bool,
}

fn truth_fn(k: usize, x: f32) -> f32 {
    match k {
        0 => (1.3 * x).sin() + 0.25 * x,
        1 => 0.5 * x + 0.4,
        2 => 0.25 * x * x - 1.0,
        // (a square root both ways from zero; the 0.2 keeps its slope there finite,
        // and is taken off again so that the curve passes through zero without a jump)
        _ => ((x.abs() + 0.2).sqrt() - 0.2f32.sqrt()) * x.signum(),
    }
}

impl Regress {
    pub fn new(_: &Gfx) -> Self {
        Self::new_plain()
    }

    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn new_plain() -> Self {
        let mut r = Regress {
            pts: Vec::new(),
            degree: 3,
            ridge: 0.0,
            noise: 0.35,
            truth: 0,
            count: 14,
            show_squares: true,
            show_truth: true,
            rng: Rng::new(4),
            errs: Vec::new(),
            dirty: true,
        };
        r.resample();
        r
    }

    fn resample(&mut self) {
        self.pts = (0..self.count)
            .map(|i| {
                let x = -3.2 + 6.4 * (i as f32 + self.rng.f32()) / self.count as f32;
                Vec2::new(x, truth_fn(self.truth, x) + self.rng.normal() * self.noise)
            })
            .collect();
        self.dirty = true;
    }

    /// Set up one of the experiments in `TRIALS`, on the scene's first sample.
    fn set_up(&mut self, k: usize) {
        let (_, _, degree, ridge) = TRIALS[k];
        (self.degree, self.ridge, self.truth, self.count, self.noise, self.show_truth) = (degree, ridge, 0, 14, 0.35, true);
        self.rng = Rng::new(4);
        self.resample();
    }

    /// The degree fitted: the slider's, but never more than one less than the number
    /// of points (a polynomial of that degree already passes through all of them).
    fn used_degree(&self) -> usize {
        self.degree.min(self.pts.len().saturating_sub(1))
    }

    fn fit(&self, deg: usize) -> Option<(Vec<f64>, f64, f64)> {
        poly_fit(&self.pts, deg.min(self.pts.len().saturating_sub(1)), self.ridge as f64)
    }

    /// Training error and error against the true curve, for every degree that can be
    /// fitted (the others are not numbers, and are left out of the chart).
    fn sweep(&mut self) {
        self.errs.clear();
        for deg in 0..=MAX_DEG {
            let fitted = if deg < self.pts.len() { self.fit(deg) } else { None };
            let Some((c, cx, sx)) = fitted else {
                self.errs.push((f32::NAN, f32::NAN));
                continue;
            };
            let train = self.pts.iter().map(|p| (poly_eval(&c, cx, sx, p.x) - p.y).powi(2)).sum::<f32>() / self.pts.len().max(1) as f32;
            let m = 200;
            let test = (0..m)
                .map(|i| {
                    let x = -3.2 + 6.4 * i as f32 / (m - 1) as f32;
                    (poly_eval(&c, cx, sx, x) - truth_fn(self.truth, x)).powi(2)
                })
                .sum::<f32>()
                / m as f32;
            self.errs.push((train, test));
        }
    }
}

impl Scene for Regress {
    fn view(&self) -> View {
        // room for every sample (x from -3.2 to 3.2) and, to their right, for the small chart
        View::Fit { center: Vec2::new(0.35, 0.0), half: Vec2::new(3.85, 3.4) }
    }

    fn animated(&self) -> bool {
        false
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.18;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "Least squares picks the curve that makes the total area of the squares as small as possible. Give the curve more freedom and it hugs the points, then starts chasing their noise. Drag points; click empty space to add one.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Model");
        self.dirty |= ui::slider_int(ui, "polynomial degree", &mut self.degree, 0..=MAX_DEG);
        let used = self.used_degree();
        if used < self.degree {
            ui::note(ui, &format!("Only {} points: degree {used} already passes through every one, so that is the degree used.", self.pts.len()));
        }
        self.dirty |= ui::slider_log_or_zero(ui, r"ridge penalty $\lambda$", &mut self.ridge, 1e-6, 100.0);
        ui::note(ui, r"The ridge penalty adds $\lambda$ times the sum of the squared coefficients (all but the constant) to the total area of the squares:");
        ui::equation(ui, r"\sum_i \bigl(c_0 + c_1 u_i + \cdots + c_d u_i^d - y_i\bigr)^2 + \lambda \left(c_1^2 + \cdots + c_d^2\right)");
        ui::note(ui, r"$c_0, \ldots, c_d$ are the coefficients of the curve, of degree $d$, written in $u$, which runs from $-1$ at the leftmost point to $1$ at the rightmost. Raise $\lambda$ to calm a wiggly high-degree curve.");
        ui.checkbox(&mut self.show_squares, "show the squares");
        ui.checkbox(&mut self.show_truth, "show the true curve");
        ui::section(ui, "Data");
        let mut again = ui::choice(ui, &["wave", "line", "parabola", "root"], &mut self.truth);
        again |= ui::slider_int(ui, "points", &mut self.count, 3..=40);
        again |= ui::slider(ui, "noise", &mut self.noise, 0.0..=1.5);
        if ui.button("new sample").clicked() || again {
            self.resample();
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::WHITE, "white dots: the data, the true curve plus noise"),
                (pal::DIM, "grey: the true curve"),
                (pal::CYAN, "cyan: the fitted polynomial"),
                (pal::PINK, "pink: how far each point misses the curve, and the square on that miss"),
                (pal::ORANGE, "the small chart: the error on the data (cyan) and against the true curve (orange), for every degree"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Mean squared error:", "the average area of the squares: on the data, over the points; against the true curve, over $x$ from $-3.2$ to $3.2$, where the points are drawn.");
        ui::term(ui, "Overfitting:", "a curve so flexible that it follows the noise in the points rather than the curve they came from: a small error on the data, a large one against the truth.");

        if let Some(&(train, test)) = self.errs.get(used) {
            ui::section(ui, "Mean squared error");
            // (an error that is not a number, where the fit failed, is written as it always was)
            let number = |v: f32| if v.is_finite() { format!("${}$", fmt(v)) } else { fmt(v) };
            ui::readout_series(ui, "on the data", number(train), pal::CYAN);
            ui::readout_series(ui, "against the true curve", number(test), pal::ORANGE);
            ui::note(ui, "The small chart shows both for every degree, with the degree in use marked in white. Where orange climbs away from cyan, the curve has started to fit the noise.");
            if used + 1 == self.pts.len() && self.ridge < 1e-4 {
                ui::note(ui, "As many coefficients as points: the curve passes through every one, and tells you nothing about the points in between.");
            }
        }
        ui::see_also(ui, &[("lsq", "least squares as a projection"), ("overdet", "one equation per data point: too many equations"), ("fit1d", "a network fitting a curve instead"), ("confound", "when a fitted slope is not what one thing does to another")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let mut moved = false;
        for i in 0..self.pts.len() {
            let mut p = self.pts[i];
            if f.drag2(100 + i as u32, &mut p, pal::WHITE) {
                self.pts[i] = p;
                moved = true;
            }
        }
        // (a click on empty space adds a point, so both clicks of a double-click are
        // the scene's: R still resets the view)
        f.handles.claim_clicks();
        if f.clicked_canvas() && self.pts.len() < 60 {
            if let Some(m) = f.mouse() {
                self.pts.push(m);
                moved = true;
            }
        }
        if moved || self.dirty {
            self.sweep();
            self.dirty = false;
        }

        let (lo, hi) = f.cam.flat_bounds();
        if self.show_truth {
            f.curve(lo.x, hi.x, 300, 1.6, pal::DIM.a(0.8), |x| Vec3::new(x, truth_fn(self.truth, x), 0.0));
        }
        if let Some((c, cx, sx)) = self.fit(self.degree) {
            for p in &self.pts {
                let y = poly_eval(&c, cx, sx, p.x);
                let r = p.y - y;
                if self.show_squares && r.abs() < 20.0 {
                    let s = r.abs();
                    f.dl.tris_alpha.rect(Vec2::new(p.x, y.min(p.y)), Vec2::new(p.x + s, y.max(p.y)), 0.0, pal::PINK.a(0.2));
                }
                f.line((p.x, y), (p.x, p.y), 1.6, pal::PINK);
            }
            f.curve(lo.x, hi.x, 600, 3.0, pal::CYAN, |x| {
                let y = poly_eval(&c, cx, sx, x);
                Vec3::new(x, if y.abs() < 1e4 { y } else { f32::NAN }, 0.0)
            });
        }

        // inset: error against degree, on a logarithmic scale
        let size = hi - lo;
        // world units per pixel: the margins hold text, whose size is in pixels
        let wpp = size.y / f.view_size().y;
        let (w, h) = (size.x * 0.27, size.y * 0.24);
        let o = Vec2::new(hi.x - w - 34.0 * wpp, lo.y + 62.0 * wpp);
        // a row of names above the curves, the degrees and their title below
        inset(f, o - Vec2::new(18.0, 46.0) * wpp, o + Vec2::new(w, h) + Vec2::new(18.0, 30.0) * wpp);
        let finite: Vec<f32> = self.errs.iter().flat_map(|e| [e.0, e.1]).filter(|v| v.is_finite() && *v > 0.0).collect();
        if !finite.is_empty() {
            let (mn, mx) = finite.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
            let (l0, l1) = (mn.max(1e-5).ln(), mx.max(mn * 10.0).min(1e3).ln());
            let at = |deg: usize, v: f32| o + Vec2::new(w * deg as f32 / MAX_DEG as f32, h * ((v.clamp(1e-5, 1e3).ln() - l0) / (l1 - l0)).clamp(0.0, 1.0));
            let used = self.used_degree();
            let x = o.x + w * used as f32 / MAX_DEG as f32;
            f.line((x, o.y), (x, o.y + h), 1.4, pal::WHITE.a(0.5));
            for (k, col) in [(0, pal::CYAN), (1, pal::ORANGE)] {
                let pts: Vec<Vec2> = self
                    .errs
                    .iter()
                    .enumerate()
                    .filter(|(_, e)| if k == 0 { e.0.is_finite() } else { e.1.is_finite() })
                    .map(|(d, e)| at(d, if k == 0 { e.0 } else { e.1 }))
                    .collect();
                f.path(pts.iter().copied(), 2.0, col);
                for p in pts {
                    f.dot(p, 5.0, col);
                }
            }
            f.line(o, o + Vec2::new(w, 0.0), 1.0, pal::DIM);
            // the names stand clear of the highest point, which is at the very top
            let up = Vec2::new(0.0, -8.0);
            // (shorter where the three would not fit side by side: a phone)
            let long = ["error (log scale)", "on the data", "vs truth"];
            let room = w * pixels_per_unit(f);
            let names = if long.iter().map(|t| text_width(t, 11.5) + 14.0).sum::<f32>() <= room { long } else { ["error", "data", "truth"] };
            f.text_off(o + Vec2::new(0.0, h), up, names[0], 11.5, pal::DIM, Align2::LEFT_BOTTOM);
            f.text_off(o + Vec2::new(w * 0.57, h), up, names[1], 11.5, pal::CYAN, Align2::CENTER_BOTTOM);
            f.text_off(o + Vec2::new(w, h), up, names[2], 11.5, pal::ORANGE, Align2::RIGHT_BOTTOM);
            // the degrees along the bottom, the one in use in white under its line
            for deg in 0..=MAX_DEG {
                let at = o + Vec2::new(w * deg as f32 / MAX_DEG as f32, 0.0);
                if deg == used {
                    f.text_off(at, Vec2::new(0.0, 5.0), deg.to_string(), 11.5, pal::WHITE, Align2::CENTER_TOP);
                } else if deg % 2 == 0 {
                    f.text_off(at, Vec2::new(0.0, 5.0), deg.to_string(), 11.0, pal::DIM, Align2::CENTER_TOP);
                }
            }
            f.text_off(o + Vec2::new(w * 0.5, 0.0), Vec2::new(0.0, 22.0), "polynomial degree", 11.5, pal::DIM, Align2::CENTER_TOP);
        }
        f.axes2();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scene on sample `seed` of fourteen points from the wave, as it opens.
    fn scene(seed: u64) -> Regress {
        let mut r = Regress::new_plain();
        r.rng = Rng::new(seed);
        r.resample();
        r.sweep();
        r
    }

    /// The experiments' captions, held to account on the sample they set up, and
    /// what they show on five other samples too.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let mut r = Regress::new_plain();
        let errors = |r: &mut Regress, k: usize| {
            r.set_up(k);
            r.sweep();
            r.errs[r.used_degree()]
        };
        let (line, line_truth) = errors(&mut r, 0);
        let (cubic, cubic_truth) = r.errs[3];
        assert!((line - 0.43).abs() < 0.03 && (line_truth - 0.5).abs() < 0.03 && (cubic - 0.14).abs() < 0.005 && (cubic_truth - 0.17).abs() < 0.005, "{line} {line_truth} against {cubic} {cubic_truth}");
        let (wiggly, wiggly_truth) = errors(&mut r, 1);
        let smallest = r.errs.iter().all(|e| e.0 >= wiggly);
        assert!((wiggly - 0.05).abs() < 0.005 && (wiggly_truth - 17.0).abs() < 1.0 && smallest, "{wiggly} {wiggly_truth}");
        // the swings are widest at the ends
        let (c, cx, sx) = r.fit(12).unwrap();
        let miss = |x: f32| (poly_eval(&c, cx, sx, x) - truth_fn(0, x)).abs();
        let widest = (0..=640).map(|i| -3.2 + i as f32 * 0.01).fold((0.0, 0.0), |a, x| if miss(x) > a.1 { (x, miss(x)) } else { a });
        assert!(widest.0.abs() > 2.8, "the widest swing is at {}", widest.0);
        let (_, calm_truth) = errors(&mut r, 2);
        assert!((calm_truth - 0.15).abs() < 0.01 && calm_truth <= r.errs[3].1, "{calm_truth} against {} for degree 3", r.errs[3].1);

        for seed in 5..10 {
            let mut r = scene(seed);
            let (line, cubic, wiggly) = (r.errs[1], r.errs[3], r.errs[12]);
            assert!(line.0 > 2.0 * cubic.0 && line.1 > 2.0 * cubic.1, "seed {seed}: degree 1 {line:?}, degree 3 {cubic:?}");
            assert!(wiggly.0 < cubic.0 && wiggly.1 > 10.0 * cubic.1, "seed {seed}: degree 12 {wiggly:?}, degree 3 {cubic:?}");
            r.ridge = 0.01;
            r.sweep();
            assert!(r.errs[12].1 < wiggly.1 / 20.0, "seed {seed}: with the ridge {:?}, without {wiggly:?}", r.errs[12]);
        }
    }

    /// The ridge caption: the fit makes the total area of the squares plus λ times the
    /// sum of the squared coefficients, all but the constant, of the curve in u (from
    /// -1 at the leftmost point to 1 at the rightmost) as small as it can be. Where that
    /// is smallest, its slope along every coefficient is zero.
    #[test]
    fn the_ridge_penalises_what_its_caption_says() {
        let mut r = scene(4);
        r.degree = 12;
        for ridge in [0.01f32, 1.0] {
            r.ridge = ridge;
            let (c, cx, sx) = r.fit(12).unwrap();
            let (lo, hi) = r.pts.iter().fold((f32::MAX, f32::MIN), |a, p| (a.0.min(p.x), a.1.max(p.x)));
            assert!((((lo - cx as f32) / sx as f32) + 1.0).abs() < 1e-5 && (((hi - cx as f32) / sx as f32) - 1.0).abs() < 1e-5);
            let us: Vec<f64> = r.pts.iter().map(|p| (p.x as f64 - cx) / sx).collect();
            let fitted = |u: f64| c.iter().rev().fold(0.0, |a, &k| a * u + k);
            for k in 0..c.len() {
                let squares: f64 = us.iter().zip(&r.pts).map(|(&u, p)| 2.0 * (fitted(u) - p.y as f64) * u.powi(k as i32)).sum();
                let penalty = if k == 0 { 0.0 } else { 2.0 * ridge as f64 * c[k] };
                // (against the size of either part, which can be large)
                assert!((squares + penalty).abs() < 1e-9 * (1.0 + squares.abs()), "λ {ridge}, coefficient {k}: slope {}", squares + penalty);
            }
        }
    }

    /// Fewer points than the degree asks for: the degree used, the chart and the
    /// read-outs agree.
    #[test]
    fn with_few_points_the_degree_used_is_shown() {
        let mut r = Regress::new_plain();
        (r.count, r.degree) = (5, 9);
        r.resample();
        r.sweep();
        assert_eq!(r.used_degree(), 4);
        assert!(r.errs[..5].iter().all(|e| e.0.is_finite()) && r.errs[5..].iter().all(|e| e.0.is_nan() && e.1.is_nan()));
        // through every point
        assert!(r.errs[4].0 < 1e-6, "{:?}", r.errs[4]);
    }
}

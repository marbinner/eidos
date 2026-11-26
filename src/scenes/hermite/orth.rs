//! Orthogonality under the Gaussian.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "hermite_orth",
    title: "Orthogonality under the Gaussian",
    group: "Hermite polynomials",
    blurb: "Watch the inner product cancel to zero, and Gram–Schmidt build the family.",
    make: |g| Box::new(HermiteOrtho::new(g)),
};

const ORTH_K: usize = 10;

/// Experiments: (button, what to look for, first degree, second degree, orthogonalised
/// up to degree, steps down the ladder).
const TRIALS: [(&str, &str, usize, usize, usize, usize); 3] = [
    (
        "plain powers",
        r"Before Gram–Schmidt: $x^2/\sqrt{3}$ and $x^4/\sqrt{105}$ are both even and never negative, so their product has no area below the axis and the total climbs to $15/\sqrt{315} = 0.845$.",
        2,
        4,
        0,
        0,
    ),
    ("after Gram–Schmidt", "$h_2$ and $h_4$: their product has lobes above the axis (cyan) and below it (orange) of exactly equal area, and the running total comes back to 0.", 2, 4, ORTH_K - 1, 0),
    (
        "down the ladder",
        r"Integration by parts moves the pair $\langle h_3, h_5 \rangle$ down the diagonal (yellow arrow) to $\langle h_0, h_2 \rangle$: the constant $1$ against $h_2$, whose average is $0$. So $\langle h_3, h_5 \rangle$ is $0$ too.",
        3,
        5,
        ORTH_K - 1,
        3,
    ),
];

/// The twelve-point Gauss–Hermite rule, worked out once: finding its nodes is a scan
/// and a bisection, and the table asks for it 55 times whenever Gram–Schmidt moves on.
fn rule() -> &'static (Vec<f64>, Vec<f64>) {
    static RULE: std::sync::OnceLock<(Vec<f64>, Vec<f64>)> = std::sync::OnceLock::new();
    RULE.get_or_init(|| gauss_hermite(12))
}

pub struct HermiteOrtho {
    m: usize,
    n: usize,
    /// Functions 0..=step have been orthogonalised; the rest are still plain powers.
    step: usize,
    /// How many times integration by parts has lowered both degrees.
    ladder: usize,
    playing: bool,
    clock: f32,
    gram: Vec<f64>,
    gram_step: usize,
    wide: Option<bool>,
    view: Option<View>,
}

impl HermiteOrtho {
    pub fn new(_: &Gfx) -> Self {
        Self::create()
    }

    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        // the quadrature rule when the scene is made, rather than in its first frame
        rule();
        HermiteOrtho {
            m: 2,
            n: 4,
            step: ORTH_K - 1,
            ladder: 0,
            playing: false,
            clock: 0.0,
            gram: Vec::new(),
            gram_step: usize::MAX,
            wide: None,
            view: None,
        }
    }

    fn view_for(wide: bool) -> View {
        if wide {
            View::Fit { center: Vec2::new(0.55, 0.15), half: Vec2::new(6.0, 3.2) }
        } else {
            View::Fit { center: Vec2::new(0.0, 0.5), half: Vec2::new(3.75, 5.75) }
        }
    }

    /// The Gaussian average of the product of basis functions `i` and `j`: both are
    /// polynomials of degree under ten, for which a twelve-point Gauss-Hermite rule
    /// is exact (and several thousand times less work than a fine grid).
    fn inner(step: usize, i: usize, j: usize) -> f64 {
        let (z, w) = rule();
        z.iter().zip(w).map(|(&z, &w)| w * Self::basis(step, i, z) * Self::basis(step, j, z)).sum()
    }

    /// Basis function `k` at the current stage of Gram–Schmidt, normalised to mean
    /// square 1: a plain power xᵏ is divided by its root mean square √E[X²ᵏ] = √(2k − 1)!!.
    fn basis(step: usize, k: usize, x: f64) -> f64 {
        if k <= step {
            h(k, x)
        } else {
            x.powi(k as i32) / double_factorial_odd(k).sqrt()
        }
    }

    /// Basis function `k` as it is written beside its curve and in the numbers (markup,
    /// to go between dollars): a plain power says what it has been divided by, so that
    /// the numbers follow from the names.
    fn name(&self, k: usize) -> String {
        if k <= self.step || k <= 1 {
            self.short_name(k)
        } else {
            format!(r"x^{{{k}}}/\sqrt{{{}}}", double_factorial_odd(k))
        }
    }

    /// The same in the table's headings, where there is room only for the power (the
    /// caption under the table says how the powers are scaled).
    fn short_name(&self, k: usize) -> String {
        if k == 0 && self.step + 1 < ORTH_K {
            "1".into()
        } else if k <= self.step {
            format!("h_{{{k}}}")
        } else {
            match k {
                0 => "1".into(),
                1 => "x".into(),
                _ => format!("x^{{{k}}}"),
            }
        }
    }

    /// Every inner product at the current stage, worked out again when the stage changes.
    fn refresh_gram(&mut self) {
        if self.gram_step != self.step {
            self.gram_step = self.step;
            let step = self.step;
            self.gram = vec![0.0; ORTH_K * ORTH_K];
            for i in 0..ORTH_K {
                for j in i..ORTH_K {
                    let g = Self::inner(step, i, j);
                    self.gram[i * ORTH_K + j] = g;
                    self.gram[j * ORTH_K + i] = g;
                }
            }
        }
    }

    /// While "run it" plays, one more degree every 0.9 s of scene time (`dt`: it stands
    /// still while the app is paused and follows its speed control), carrying over
    /// what is left of a frame so that the pace is the same at any frame rate.
    fn advance(&mut self, dt: f32) {
        if !self.playing {
            return;
        }
        self.clock += dt;
        while self.playing && self.clock > 0.9 {
            self.clock -= 0.9;
            if self.step + 1 < ORTH_K {
                self.step += 1;
            } else {
                self.playing = false;
            }
        }
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let (_, _, m, n, step, ladder) = TRIALS[k];
        (self.m, self.n, self.step, self.ladder) = (m, n, step, ladder);
        (self.playing, self.clock) = (false, 0.0);
    }
}

impl Scene for HermiteOrtho {
    fn view(&self) -> View {
        Self::view_for(true)
    }

    fn take_view(&mut self) -> Option<View> {
        self.view.take()
    }

    fn animated(&self) -> bool {
        self.playing
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        // side by side on a wide window, stacked on a tall one: say where things really are
        let (one, all) = if self.wide == Some(false) { ("At the top", "Below it") } else { ("On the left", "On the right") };
        ui::note(ui, "Two functions are orthogonal under the Gaussian when the average of their product is zero:");
        ui::equation(ui, r"\langle f, g \rangle = \mathbb{E}[f(X)\, g(X)] = \int f g \, \varphi \, \mathrm{d}x");
        ui::note(ui, &format!("{one}, that integral for one pair: the lobes above and below the axis cancel exactly. {all}, all pairs at once. Click a cell to pick the pair."));
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }
        ui::section(ui, "The pair");
        ui::slider_int(ui, "first degree", &mut self.m, 0..=ORTH_K - 1);
        ui::slider_int(ui, "second degree", &mut self.n, 0..=ORTH_K - 1);
        ui::section(ui, "Gram–Schmidt");
        ui::note(ui, r"Start from the plain powers $1, x, x^2, \ldots$, each divided by its root mean square, and subtract from each one its components along all the earlier ones.");
        ui::slider_int(ui, "orthogonalised up to degree", &mut self.step, 0..=ORTH_K - 1);
        ui.horizontal(|ui| {
            if ui.button("powers").clicked() {
                self.step = 0;
                self.playing = false;
            }
            if ui.button("Hermite").clicked() {
                self.step = ORTH_K - 1;
                self.playing = false;
            }
            if ui.button(if self.playing { "stop" } else { "run it" }).clicked() {
                self.playing = !self.playing;
                if self.playing {
                    self.step = 0;
                    self.clock = 0.0;
                }
            }
        });
        ui::note(ui, "\"run it\" makes one more degree perpendicular every 0.9 seconds.");
        if self.step + 1 >= ORTH_K {
            ui::section(ui, "Why it is zero");
            let top = self.m.min(self.n);
            self.ladder = self.ladder.min(top);
            ui::note(ui, "Integrate by parts under the Gaussian: a derivative on one function becomes a raising operator on the other. Each step lowers both degrees by one and leaves the inner product unchanged up to a known factor.");
            ui::slider_int(ui, "steps down the ladder", &mut self.ladder, 0..=top.max(1));
            self.ladder = self.ladder.min(top);
            let (m, n, k) = (self.m, self.n, self.ladder);
            if k > 0 {
                let factor: f64 = (0..k).map(|j| ((n - j) as f64 / (m - j) as f64).sqrt()).product();
                ui::readout(ui, &format!(r"$\langle h_{{{m}}}, h_{{{n}}} \rangle =$"), format!(r"${} \times \langle h_{{{}}}, h_{{{}}} \rangle$", fmt(factor as f32), m - k, n - k));
            }
            if k == top && m != n {
                let rest = m.max(n) - top;
                ui::note(ui, &format!(r"At the bottom one function is the constant $1$, so what is left is just the average of $h_{{{rest}}}$ over the Gaussian, and every Hermite polynomial of degree $n \ge 1$ has average zero. That single fact is the whole of orthogonality."));
            } else if k == top && m == n {
                ui::note(ui, r"Equal degrees land on $\langle 1, 1 \rangle = 1$ with factor 1: the normalisation. For the monic $\mathrm{He}_n$ the factors multiply to $n \cdot (n - 1) \cdots 1 = n!$.");
            }
        } else {
            self.ladder = 0;
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::PINK, r"pink: the first function, times $\sqrt{\varphi}$"),
                (pal::GREEN, r"green: the second, times $\sqrt{\varphi}$ (so that their product carries $\varphi$ once)"),
                (pal::CYAN, "cyan: where their product is above the axis"),
                (pal::ORANGE, "orange: where it is below"),
                (pal::YELLOW, r"yellow: the area so far, counting orange as negative; it ends at $\langle f, g \rangle$"),
                (Cmap::CoolWarm.color(1.0), "table, red: 1; paler: smaller; dark: exactly 0. The white box is the pair plotted"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Root mean square:", r"$\sqrt{\mathbb{E}[f(X)^2]}$, the typical size of $f(X)$. Each function here is divided by its own, so that $\langle f, f \rangle = 1$ and the overlaps can be compared.");
        ui::term(ui, "Gram–Schmidt:", "take functions one at a time and subtract from each its components along the ones already done; what is left is perpendicular to all of them.");

        ui::section(ui, "Numbers");
        self.refresh_gram();
        let (m, n, step) = (self.m, self.n, self.step);
        let g = self.gram.get(m * ORTH_K + n).copied().unwrap_or(0.0);
        ui::readout(ui, &format!(r"$\langle {}, {} \rangle$", self.name(m), self.name(n)), format!("${}$", signed(g as f32, 4)));
        let (low, high) = (m.min(n), m.max(n));
        if m == n {
            // (whatever the parity test below would say: a function with itself is never zero)
            if m <= step {
                ui::note(ui, &format!("A function with itself: its mean square, 1 after normalising. For the monic ${}$ it is ${}! = {}$, the number of ways to pair up two sets of {} things.", he_name(m), m, factorial(m), m));
            } else {
                let mean_square = double_factorial_odd(m);
                let power = if m == 1 { "x".to_string() } else { format!("x^{{{m}}}") };
                ui::note(ui, &format!(r"A power with itself: $\mathbb{{E}}[X^{{{}}}] = {}!! = {}$, and dividing ${power}$ by $\sqrt{{{}}}$ is what makes this 1.", 2 * m, 2 * m - 1, mean_square, mean_square));
            }
        } else if low <= step && high > step {
            // one has been made perpendicular to everything below it; the other is
            // still a plain power of higher degree
            if (m + n) % 2 == 1 {
                ui::note(ui, "An odd function times an even one is an odd function, and the Gaussian is symmetric: zero for free.");
            } else {
                ui::note(ui, &format!("Not zero yet: ${}$ is perpendicular to every power below its own degree, but ${}$ is a plain power above it, and Gram–Schmidt has not reached it.", self.name(low), self.name(high)));
            }
        } else if low > step {
            if (m + n) % 2 == 1 {
                ui::note(ui, "An odd power times an even power is an odd function, and the Gaussian is symmetric: zero for free.");
            } else {
                // E[X^m X^n] = E[X^(m+n)] = (m+n-1)!!, over the two roots mean square
                let moment = double_factorial_odd((m + n) / 2);
                let power = |k: usize| if k == 1 { "X".to_string() } else { format!("X^{{{k}}}") };
                ui::note(
                    ui,
                    &format!(r"Plain powers of the same parity overlap heavily: $\mathbb{{E}}[{} {}] = \mathbb{{E}}[X^{{{}}}] = {}!! = {}$, so", power(low), power(high), m + n, m + n - 1, moment),
                );
                ui::equation(
                    ui,
                    &format!(
                        r"\langle {}, {} \rangle = \frac{{{}}}{{\sqrt{{{}}}\, \sqrt{{{}}}}} = {:.4}",
                        self.name(low),
                        self.name(high),
                        moment,
                        double_factorial_odd(low),
                        double_factorial_odd(high),
                        moment / (double_factorial_odd(low) * double_factorial_odd(high)).sqrt()
                    ),
                );
                ui::note(ui, "Nothing forces it to vanish.");
            }
        } else {
            ui::note(ui, "Zero. Not approximately: the positive and negative areas are equal.");
        }
        ui::see_also(ui, &[("hermite", "the polynomials this builds"), ("hermite_expand", "what a perpendicular family is good for"), ("lsq", "projection: the subtraction step, with arrows")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt);
        self.refresh_gram();
        // side by side on a wide window, stacked on a tall one
        let wide = f.cam.viewport.x > f.cam.viewport.y * 1.25;
        if self.wide != Some(wide) {
            self.wide = Some(wide);
            self.view = Some(Self::view_for(wide));
        }
        let step = self.step;
        let k = if step + 1 >= ORTH_K { self.ladder.min(self.m.min(self.n)) } else { 0 };
        let (m, n) = (self.m - k, self.n - k);
        let (cm, cn) = (pal::PINK, pal::GREEN);

        // the plot region maps x in [-W, W] onto 6.6 world units; degree-9 functions
        // are still far from negligible at |x| = 6, so the window has to be this wide
        const W: f64 = 7.5;
        let sx = (3.3 / W) as f32;
        let (ox, top, bot, origin) = if wide {
            (-2.0f32, 1.75f32, -1.9f32, Vec2::new(2.25, 1.75))
        } else {
            (0.0f32, 4.55f32, 0.95f32, Vec2::new(-2.0, -0.45))
        };
        let wx = |x: f32| x * sx + ox;
        let w32 = W as f32;

        // the two functions, each carrying half of the weight (so their product is the integrand)
        f.line((wx(-w32), top), (wx(w32), top), 1.0, pal::DIM.a(0.6));
        for (deg, c) in [(m, cm), (n, cn)] {
            f.curve(-w32, w32, 600, 2.2, c, |x| {
                let v = Self::basis(step, deg, x as f64) * phi(x as f64).sqrt();
                Vec3::new(wx(x), top + v as f32 * 1.35, 0.0)
            });
        }
        // the second name under the first by the first's height as typeset: on a phone
        // the two lines are closer together than a square root is tall
        let (first, second) = (format!(r"${} \cdot \sqrt{{\varphi}}$", self.name(m)), format!(r"${} \cdot \sqrt{{\varphi}}$", self.name(n)));
        let under = top + 1.25 - (f.text_size(&first, 14.0).y + 2.0) / pixels_per_unit(f);
        f.text((wx(-w32), top + 1.25), first, 14.0, cm, Align2::LEFT_BOTTOM);
        f.text((wx(-w32), under.min(top + 0.98)), second, 14.0, cn, Align2::LEFT_BOTTOM);

        // their product, and its integral accumulating from the left
        f.line((wx(-w32), bot), (wx(w32), bot), 1.0, pal::DIM.a(0.6));
        let unit = 1.9f32;
        f.dashed((wx(-w32), bot + unit), (wx(w32), bot + unit), 1.0, pal::DIM.a(0.7));
        // clear of the dot that ends the running total
        f.text_off((wx(w32), bot + unit), Vec2::new(10.0, 0.0), "1", 12.0, pal::DIM, Align2::LEFT_CENTER);
        f.text_off((wx(w32), bot), Vec2::new(10.0, 0.0), "0", 12.0, pal::DIM, Align2::LEFT_CENTER);
        f.text_off((wx(-w32), bot + unit), Vec2::new(0.0, 4.0), "1: the total for a function with itself", 11.5, pal::DIM, Align2::LEFT_TOP);
        let integrand = |x: f64| Self::basis(step, m, x) * Self::basis(step, n, x) * phi(x);
        // Simpson's rule over [-13, 13], which holds essentially all of the mass; only
        // the part inside the window is drawn
        let (span, steps) = (13.0f64, 2600usize);
        let dx = 2.0 * span / steps as f64;
        let g_scale = 2.6f32;
        let mut run = 0.0f64;
        let mut prev_g = integrand(-span);
        let mut path = Vec::with_capacity(1600);
        for i in 1..=steps {
            let x = -span + dx * i as f64;
            let g = integrand(x);
            run += (prev_g + 4.0 * integrand(x - 0.5 * dx) + g) * dx / 6.0;
            if x.abs() <= W + 1e-9 {
                if x - dx >= -W - 1e-9 {
                    let c = if g + prev_g >= 0.0 { pal::CYAN } else { pal::ORANGE };
                    f.dl.tris_alpha.quad(
                        Vec3::new(wx((x - dx) as f32), bot, 0.0),
                        Vec3::new(wx(x as f32), bot, 0.0),
                        Vec3::new(wx(x as f32), bot + g as f32 * g_scale, 0.0),
                        Vec3::new(wx((x - dx) as f32), bot + prev_g as f32 * g_scale, 0.0),
                        c.a(0.42),
                    );
                }
                path.push(Vec3::new(wx(x as f32), bot + run as f32 * unit, 0.0));
            }
            prev_g = g;
        }
        f.glow_path(path.iter().copied(), 2.4, pal::YELLOW);
        f.path(path.iter().copied(), 2.2, pal::YELLOW);
        let end = *path.last().unwrap();
        f.dot(end, 10.0, pal::YELLOW);
        let shown = if run.abs() < 5e-7 { 0.0 } else { run };
        f.text_off(end, Vec2::new(-8.0, -12.0), format!("total {}", signed(shown as f32, 4)), 15.0, pal::YELLOW, Align2::RIGHT_BOTTOM);
        // each curve above carries √φ, so the product carries the whole weight
        // as much of the caption as the plot is wide (the scaled powers have long names)
        let ppu = pixels_per_unit(f);
        let plot_px = 2.0 * w32 * sx * ppu;
        let full = format!(r"their product ${} \cdot {} \cdot \varphi$: what is being integrated", self.name(m), self.name(n));
        let caption = [full.as_str(), r"their product times $\varphi$: what is being integrated", r"their product times $\varphi$"].into_iter().find(|t| text_width(t, 13.0) <= plot_px).unwrap_or("");
        f.text((wx(-w32), bot + unit + 0.52), caption, 13.0, pal::TEXT, Align2::LEFT_BOTTOM);
        // the three names in a row, each moved right of the one before where the view is
        // too narrow for the places they have on a wide one
        let gap = 12.0 / ppu;
        let below_at = (wx(-w32) + 1.55).max(wx(-w32) + text_width("area above the axis", 12.0) / ppu + gap);
        let total_at = (wx(-w32) + 2.45).max(below_at + text_width("area below", 12.0) / ppu + gap);
        f.text((wx(-w32), bot + unit + 0.27), "area above the axis", 12.0, pal::CYAN, Align2::LEFT_BOTTOM);
        f.text((below_at, bot + unit + 0.27), "area below", 12.0, pal::ORANGE, Align2::LEFT_BOTTOM);
        f.text((total_at, bot + unit + 0.27), "running total", 12.0, pal::YELLOW, Align2::LEFT_BOTTOM);
        // the x scale has a rule of its own below the plot: just under the axis the numbers
        // would lie inside the lobes and under the running total
        let rule = bot - 0.5;
        f.line((wx(-w32), rule), (wx(w32), rule), 1.0, pal::DIM.a(0.45));
        for t in [-6, -4, -2, 0, 2, 4, 6] {
            f.line((wx(t as f32), rule), (wx(t as f32), rule + 0.06), 1.0, pal::DIM.a(0.8));
            f.text_off((wx(t as f32), rule), Vec2::new(0.0, 3.0), format!("{t}"), 11.5, pal::DIM, Align2::CENTER_TOP);
        }
        f.text_off((wx(w32), rule), Vec2::new(10.0, 0.0), "$x$", 12.0, pal::DIM, Align2::LEFT_CENTER);

        // every inner product at once
        let cell = 0.4f32;
        let mouse = f.mouse();
        let mut pick = None;
        for i in 0..ORTH_K {
            for j in 0..ORTH_K {
                let g = self.gram[i * ORTH_K + j] as f32;
                let lo = origin + Vec2::new(j as f32 * cell, -(i as f32 + 1.0) * cell);
                let hi = lo + Vec2::splat(cell);
                let c = Cmap::CoolWarm.color(0.5 + 0.5 * g.clamp(-1.0, 1.0));
                let shade = if g.abs() < 0.004 { Color::hex(0x141A28) } else { c };
                f.dl.tris_alpha.rect(lo + Vec2::splat(0.012), hi - Vec2::splat(0.012), 0.0, shade);
                if g.abs() >= 0.004 && g.abs() < 0.995 {
                    f.text((lo + hi) * 0.5, format!("{:.2}", g).trim_start_matches('0').replace("-0.", "-."), 10.5, pal::WHITE, Align2::CENTER_CENTER);
                } else if g.abs() >= 0.995 {
                    f.text((lo + hi) * 0.5, "1", 11.5, pal::WHITE, Align2::CENTER_CENTER);
                }
                if let Some(p) = mouse {
                    if p.x > lo.x && p.x < hi.x && p.y > lo.y && p.y < hi.y {
                        pick = Some((i, j));
                    }
                }
            }
        }
        if let Some((i, j)) = pick {
            if f.clicked_canvas() {
                self.m = i;
                self.n = j;
                self.ladder = 0;
            }
        }
        // the chosen pair, and the path integration by parts takes it along a diagonal
        let cell_lo = |i: usize, j: usize| origin + Vec2::new(j as f32 * cell, -(i as f32 + 1.0) * cell);
        for back in 0..=k {
            let lo = cell_lo(m + back, n + back);
            let ring = [lo, lo + Vec2::new(cell, 0.0), lo + Vec2::splat(cell), lo + Vec2::new(0.0, cell)];
            let (w, c) = if back == 0 { (2.4, pal::WHITE) } else { (1.3, pal::WHITE.a(0.45)) };
            f.dl.lines.closed(&ring.map(|p| p.extend(0.02)), w, c);
        }
        if k > 0 {
            let a = cell_lo(self.m, self.n) + Vec2::splat(cell * 0.5);
            let b = cell_lo(m, n) + Vec2::splat(cell * 0.5);
            f.arrow(a.extend(0.03), b.extend(0.03), 1.8, pal::YELLOW);
        }
        for k in 0..ORTH_K {
            let c = if k <= step { pal::TEXT } else { pal::DIM };
            let name = format!("${}$", self.short_name(k));
            f.text(origin + Vec2::new((k as f32 + 0.5) * cell, 0.05), name.as_str(), 11.0, c, Align2::CENTER_BOTTOM);
            f.text(origin + Vec2::new(-0.06, -(k as f32 + 0.5) * cell), name, 11.0, c, Align2::RIGHT_CENTER);
        }
        let off_diag: f64 = (0..ORTH_K).flat_map(|i| (0..ORTH_K).map(move |j| (i, j))).filter(|(i, j)| i != j).map(|(i, j)| self.gram[i * ORTH_K + j].abs()).fold(0.0, f64::max);
        let caption = if step + 1 >= ORTH_K {
            "every pair is orthogonal: the identity matrix".to_string()
        } else {
            format!("largest overlap between different functions: {:.2}", off_diag)
        };
        let under = origin + Vec2::new(ORTH_K as f32 * cell * 0.5, -(ORTH_K as f32) * cell - 0.12);
        f.text(under, caption, 12.5, pal::TEXT, Align2::CENTER_TOP);
        f.text_off(under, Vec2::new(0.0, 20.0), "red: 1      dark: 0      white box: the pair plotted", 11.5, pal::DIM, Align2::CENTER_TOP);
        if step + 1 < ORTH_K {
            // the headings are short; what the numbers in their cells are made of is not
            f.text_off(under, Vec2::new(0.0, 38.0), "the powers in the table are divided by their root mean square, so the diagonal is 1", 11.5, pal::DIM, Align2::CENTER_TOP);
        }
        f.text(origin + Vec2::new(ORTH_K as f32 * cell * 0.5, 0.42), r"$\langle f, g \rangle = \mathbb{E}[f(X)\, g(X)]$  for every pair", 13.5, pal::TEXT, Align2::CENTER_BOTTOM);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the note under the inner product says, for every pair at every stage of
    /// Gram–Schmidt: zero when both have been orthogonalised, or when one is odd and
    /// the other even; not zero for an orthogonalised function against a plain power
    /// of higher degree and the same parity (the note used to say "Zero" there, above
    /// a read-out of 0.83).
    #[test]
    fn the_inner_products_are_what_the_notes_say() {
        for step in 0..ORTH_K {
            for m in 0..ORTH_K {
                for n in 0..ORTH_K {
                    let g = HermiteOrtho::inner(step, m, n);
                    // the exact rule agrees with the fine grid it replaced
                    let fine = gauss_mean(|x| HermiteOrtho::basis(step, m, x) * HermiteOrtho::basis(step, n, x));
                    assert!((g - fine).abs() < 1e-6, "step {step}: <{m}, {n}> = {g} against {fine}");
                    let (low, high) = (m.min(n), m.max(n));
                    if m == n {
                        assert!((g - 1.0).abs() < 1e-9);
                    } else if (m + n) % 2 == 1 || high <= step {
                        assert!(g.abs() < 1e-9, "step {step}: <{m}, {n}> = {g}");
                    } else if low <= step {
                        assert!(g.abs() > 0.05, "step {step}: <{m}, {n}> = {g} should not vanish");
                    }
                }
            }
        }
        // the case on show by default once "orthogonalised up to degree" is lowered to 2:
        // E[h₂(X) X⁴] = (E[X⁶] - E[X⁴]) / √2 = 12 / √2, over √105
        assert!((HermiteOrtho::inner(2, 2, 4) - 6.0 * 2f64.sqrt() / 105f64.sqrt()).abs() < 1e-12);
        assert!((HermiteOrtho::inner(2, 2, 4) - 0.8281).abs() < 1e-4);
    }

    /// The plain powers are shown divided by their root mean square, and their names
    /// say so: every number beside them follows from what is written. (They used to be
    /// called x² and x⁴, above a read-out of 0.8452 and a note saying E[X⁶] = 15.)
    #[test]
    fn the_scaled_powers_are_named_for_what_they_are() {
        let mut scene = HermiteOrtho::create();
        scene.step = 0;
        assert_eq!((scene.name(0), scene.name(1), scene.name(2), scene.name(4)), ("1".to_string(), "x".to_string(), r"x^{2}/\sqrt{3}".to_string(), r"x^{4}/\sqrt{105}".to_string()));
        assert_eq!(scene.name(9), r"x^{9}/\sqrt{34459425}");
        // (the table keeps the short names, with a caption under it)
        assert_eq!(scene.short_name(4), "x^{4}");
        // the number under a name is the root mean square: E[X²ᵏ] = (2k - 1)!!
        for k in 0..ORTH_K {
            let mean_square = gauss_mean(|x| x.powi(2 * k as i32));
            assert!((mean_square / double_factorial_odd(k) - 1.0).abs() < 1e-9, "k = {k}");
        }
        // what the note works out, for every pair of plain powers it is shown for
        for m in 1..ORTH_K {
            for n in 1..ORTH_K {
                let g = HermiteOrtho::inner(0, m, n);
                if m == n {
                    assert!((g - 1.0).abs() < 1e-12, "a power with itself: {g}");
                } else if (m + n) % 2 == 0 {
                    let note = double_factorial_odd((m + n) / 2) / (double_factorial_odd(m) * double_factorial_odd(n)).sqrt();
                    assert!((g / note - 1.0).abs() < 1e-12, "<{m}, {n}> = {g} against {note}");
                }
            }
        }
        assert!((HermiteOrtho::inner(0, 2, 4) - 15.0 / 315f64.sqrt()).abs() < 1e-12);
        assert!((HermiteOrtho::inner(0, 2, 4) - 0.8452).abs() < 1e-4);
        // and once orthogonalised they are the Hermite polynomials again
        scene.step = ORTH_K - 1;
        assert_eq!(scene.name(4), "h_{4}");
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let mut scene = HermiteOrtho::create();
        let area = |step: usize, m: usize, n: usize, sign: f64| gauss_mean(|x| (sign * HermiteOrtho::basis(step, m, x) * HermiteOrtho::basis(step, n, x)).max(0.0));

        // plain powers: never below the axis, and a total of 15/√315
        scene.set_up(0);
        assert!((scene.m, scene.n, scene.step, scene.playing) == (2, 4, 0, false));
        assert_eq!((scene.name(2), scene.name(4)), (r"x^{2}/\sqrt{3}".to_string(), r"x^{4}/\sqrt{105}".to_string()));
        assert!(area(0, 2, 4, -1.0) == 0.0);
        assert!((HermiteOrtho::inner(0, 2, 4) - 0.845).abs() < 5e-4);

        // after Gram–Schmidt: real lobes either side, of equal area
        scene.set_up(1);
        assert!((scene.m, scene.n, scene.step) == (2, 4, ORTH_K - 1));
        let (above, below) = (area(ORTH_K - 1, 2, 4, 1.0), area(ORTH_K - 1, 2, 4, -1.0));
        assert!(above > 0.2 && (above - below).abs() < 1e-9, "{above} above, {below} below");

        // down the ladder: from <h₃, h₅> to <h₀, h₂>, the average of h₂, which is 0
        scene.set_up(2);
        assert!((scene.m, scene.n, scene.step, scene.ladder) == (3, 5, ORTH_K - 1, 3));
        assert_eq!((scene.m - scene.ladder, scene.n - scene.ladder), (0, 2));
        assert!(gauss_mean(|x| h(2, x)).abs() < 1e-9 && HermiteOrtho::inner(ORTH_K - 1, 3, 5).abs() < 1e-12);
    }

    /// "run it" moves on by scene time: the same at 30 and at 240 frames a second, and
    /// not at all while the app is paused (it used to run on real time).
    #[test]
    fn running_gram_schmidt_follows_scene_time() {
        for rate in [30.0f32, 60.0, 144.0, 240.0] {
            let mut scene = HermiteOrtho::create();
            (scene.playing, scene.step, scene.clock) = (true, 0, 0.0);
            for _ in 0..(2.75 * rate) as usize {
                scene.advance(1.0 / rate);
            }
            assert_eq!(scene.step, 3, "{rate} Hz");
            // paused: the app gives the scene no time
            for _ in 0..1000 {
                scene.advance(0.0);
            }
            assert!(scene.step == 3 && scene.playing);
            // and at the end it stops by itself, however large a step of time comes
            scene.advance(100.0);
            assert!(scene.step == ORTH_K - 1 && !scene.playing);
        }
    }
}

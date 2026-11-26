//! The Hermite family.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "hermite",
    title: "The Hermite family",
    group: "Hermite polynomials",
    blurb: "The polynomials, their zeros, and the ladder that connects them.",
    make: |g| Box::new(HermiteFamily::new(g)),
};

/// Experiments: (button, what to look for). `set_up` holds the settings of each.
const TRIALS: [(&str, &str); 3] = [
    (
        "the lowering rule",
        r"Dashed white: the slope of $h_4$, measured point by point. Wide yellow: $2 h_3$ (that is $\sqrt{4}\, h_3$). They lie on top of each other: the slope of a Hermite polynomial is the one a degree lower.",
    ),
    ("zeros interlace", "$h_5$ has five zeros (large dots), $h_6$ six (small): between two neighbouring zeros of either lies exactly one zero of the other."),
    (
        "Gauss–Hermite nodes",
        r"Six yellow stems at the zeros of $\mathrm{He}_6$, their tips on the purple bell. Weighted by the stems, the values of a polynomial at just these six points give its exact Gaussian average, up to degree 11. For $x^{12}$ they give 9675 instead of 10395.",
    ),
];

pub struct HermiteFamily {
    n: usize,
    mode: usize,
    show_lower: bool,
    show_weight: bool,
    show_zeros: bool,
    show_derivative: bool,
    show_nodes: bool,
    view: Option<View>,
}

impl HermiteFamily {
    pub fn new(_: &Gfx) -> Self {
        Self::create()
    }

    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        HermiteFamily {
            n: 4,
            mode: 0,
            show_lower: true,
            show_weight: true,
            show_zeros: true,
            show_derivative: false,
            show_nodes: false,
            view: None,
        }
    }

    fn view_for(mode: usize) -> View {
        match mode {
            1 => View::Fit { center: Vec2::new(0.0, 0.0), half: Vec2::new(4.2, 14.0) },
            // (these are at most 0.63 high: a closer view than the polynomials get)
            2 => View::Fit { center: Vec2::new(0.0, 0.0), half: Vec2::new(5.8, 0.8) },
            _ => View::Fit { center: Vec2::new(0.0, 0.0), half: Vec2::new(4.6, 3.3) },
        }
    }

    /// The displayed function of degree `k`.
    fn value(&self, k: usize, x: f64) -> f64 {
        Self::value_in(self.mode, k, x)
    }

    fn value_in(mode: usize, k: usize, x: f64) -> f64 {
        match mode {
            1 => he(k, x),
            // Hermite functions: orthonormal in plain L², h_k(x) sqrt(phi(x))
            2 => h(k, x) * phi(x).sqrt(),
            _ => h(k, x),
        }
    }

    /// The slope of the displayed function of degree `k`, measured (not from the rule
    /// it is drawn against).
    fn slope(&self, k: usize, x: f64) -> f64 {
        (self.value(k, x + 1e-4) - self.value(k, x - 1e-4)) / 2e-4
    }

    /// What the lowering rule says the slope of degree n is: this multiple of degree n - 1.
    fn lowering_factor(&self) -> f64 {
        if self.mode == 1 {
            self.n as f64
        } else {
            (self.n as f64).sqrt()
        }
    }

    /// The `n` Gauss–Hermite nodes, each with the height of its stem: its weight spread
    /// over its share of the axis, so that the stems trace the density they stand for.
    fn stems(n: usize) -> Vec<(f64, f64)> {
        let (x, w) = gauss_hermite(n);
        let k = x.len().min(w.len());
        (0..k)
            .map(|i| {
                let left = if i > 0 { x[i] - x[i - 1] } else { x.get(1).map_or(2.0, |b| b - x[0]) };
                let right = if i + 1 < k { x[i + 1] - x[i] } else { left };
                (x[i], w[i] / (0.5 * (left + right)))
            })
            .collect()
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let (n, derivative, zeros, nodes) = match k {
            0 => (4, true, false, false),
            1 => (5, false, true, false),
            _ => (6, false, false, true),
        };
        (self.n, self.mode, self.show_lower) = (n, 0, false);
        // the bell only where the stems are measured against it
        (self.show_weight, self.show_zeros, self.show_derivative, self.show_nodes) = (nodes, zeros, derivative, nodes);
        self.view = Some(Self::view_for(0));
    }
}

impl Scene for HermiteFamily {
    fn view(&self) -> View {
        Self::view_for(self.mode)
    }

    fn take_view(&mut self) -> Option<View> {
        self.view.take()
    }

    fn animated(&self) -> bool {
        false
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.2;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r#"The probabilists' Hermite polynomials $\mathrm{He}_n$ are what you get by taking $1, x, x^2, x^3, \ldots$ and making each one perpendicular to all the earlier ones, where "perpendicular" means the average of the product is zero when $X$ is a standard normal variable:"#);
        ui::equation(ui, r"\mathbb{E}\bigl[\mathrm{He}_m(X)\, \mathrm{He}_n(X)\bigr] = 0 \quad \text{for } m \ne n");
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }
        ui::section(ui, "Degree");
        ui::slider_int(ui, "$n$", &mut self.n, 0..=12);
        ui.checkbox(&mut self.show_lower, "show the lower degrees too");
        ui::section(ui, "Scaling");
        if ui::choice(ui, &[r"$h_n = \mathrm{He}_n / \sqrt{n!}$", r"$\mathrm{He}_n$ (monic)", r"$h_n \sqrt{\varphi}$"], &mut self.mode) {
            self.view = Some(Self::view_for(self.mode));
        }
        match self.mode {
            0 => ui::note(ui, "Normalised: each $h_n$ has mean square 1 under the Gaussian, so they are directly comparable."),
            1 => ui::note(ui, "Monic: leading coefficient 1, integer coefficients. They grow like $x^n$; zoom out to follow them."),
            _ => ui::note(ui, r"Multiplied by the square root of the weight these become the Hermite functions: orthonormal with no weight at all, and the stationary states of the quantum harmonic oscillator, stretched sideways by $\sqrt{2}$ (in its natural units)."),
        }
        ui::section(ui, "Show");
        checkbox(ui, &mut self.show_weight, r"the Gaussian weight $\varphi(x)$");
        checkbox(ui, &mut self.show_zeros, "zeros (and those of degree $n + 1$)");
        ui.checkbox(&mut self.show_derivative, "derivative (the lowering rule)");
        if self.show_derivative && self.mode == 2 {
            ui::note(ui, "The lowering rule is about the polynomials themselves: pick one of the first two scalings to see it.");
        }
        ui.checkbox(&mut self.show_nodes, "Gauss–Hermite nodes and weights");

        ui::section(ui, "Reading the picture");
        let n = self.n;
        let mut key = vec![(degree_color(n, 12), "thick curve: degree $n$, its formula at the top")];
        if self.show_lower {
            key.push((degree_color(2, 12).mix(pal::WHITE, 0.45), "thin curves: the lower degrees, each named beside it"));
        }
        if self.show_weight {
            key.push((pal::PURPLE, r"purple: the Gaussian weight $\varphi(x)$, the bell a standard normal $x$ is drawn from"));
        }
        if self.show_zeros {
            key.push((degree_color(n, 12), "large dots on the axis: the zeros of degree $n$"));
            key.push((pal::WHITE, "small white dots: the zeros of degree $n + 1$"));
        }
        if self.show_derivative && self.mode != 2 && n >= 1 {
            key.push((pal::WHITE, "dashed white: the slope of the thick curve, measured"));
            key.push((pal::YELLOW, "wide yellow: the slope the lowering rule predicts"));
        }
        if self.show_nodes && n >= 1 {
            key.push((pal::YELLOW, "yellow stems: the Gauss–Hermite nodes, each as tall as its weight divided by its share of the axis"));
        }
        ui::key(ui, &key);
        ui::section(ui, "Words");
        ui::term(ui, "Standard normal:", r"a random number drawn from the bell $\varphi$: mean 0, standard deviation 1. $\mathbb{E}[\ldots]$ is an average over it.");
        ui::term(ui, "Monic:", "with leading coefficient 1.");
        ui::term(ui, "Gauss–Hermite rule:", "$n$ points and weights whose weighted sum of a polynomial's values is its exact Gaussian average, up to degree $2n - 1$.");

        ui::section(ui, &format!("Identities at $n = {n}$"));
        // (a display: at the higher degrees, wider than the panel, it is broken after a sign)
        ui::equation(ui, &format!("{}(x) = {}", he_name(n), he_formula(n)));
        if n >= 1 {
            ui::readout(ui, "three-term rule", format!(r"${} = x\, {} - {}\, {}$", he_name(n + 1), he_name(n), n, he_name(n - 1)));
            ui::readout(ui, "lowering", format!(r"${}' = {}\, {}$", he_name(n), n, he_name(n - 1)));
        }
        ui::readout(ui, "raising", format!(r"$x\, {0} - {0}' = {1}$", he_name(n), he_name(n + 1)));
        ui::readout(ui, "eigenfunction", format!(r"${0}'' - x\, {0}' = -{1}\, {0}$", he_name(n), n));
        ui::readout(ui, &format!(r"$\mathbb{{E}}[{}(X)^2]$", he_name(n)), format!("${}! = {}$", n, factorial(n)));
        ui::note(ui, "Lowering and raising are adjoint under the Gaussian:");
        // (the right-hand side in growing brackets, which keep it whole in a narrow panel)
        ui::equation(ui, r"\mathbb{E}[f'(X)\, g(X)] = \mathbb{E}\left[f(X) \left(X g(X) - g'(X)\right)\right]");
        ui::note(ui, "That one line (Gaussian integration by parts) is where orthogonality, the eigenfunction equation and the $n!$ all come from.");
        ui::see_also(ui, &[("hermite_orth", "why they are perpendicular"), ("hermite_expand", "building other functions out of them"), ("fit1d", "$h_2$ and $h_3$ as a neuron's activation")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let (lo, hi) = f.cam.flat_bounds();
        let n = self.n;
        let span = hi.y - lo.y;
        // tick numbers first, so that the plates of the title and the captions lie over them
        f.axes2();
        if self.show_weight {
            weight_hump(f, 0.0, span * 0.28 / 0.3989, 1.0, 0.0, pal::PURPLE);
            // beside the axis rather than on it
            if let Some(p) = on_screen(f, Vec3::new(0.0, span * 0.28, 0.0)) {
                weight_caption(f, p);
            }
        }
        let samples = 700;
        let clip = |y: f64| -> f32 {
            let y = y as f32;
            if y.is_finite() && y > lo.y - span * 3.0 && y < hi.y + span * 3.0 {
                y
            } else {
                f32::NAN
            }
        };
        if self.show_lower {
            for k in 0..n {
                let c = degree_color(k, 12).a(0.55);
                f.curve(lo.x, hi.x, samples, 1.4, c, |x| Vec3::new(x, clip(self.value(k, x as f64)), 0.0));
            }
        }
        let col = degree_color(n, 12);
        let view = f.view_size();
        let title = match self.mode {
            1 => title_lines(f, &format!("{}(x) = ", he_name(n)), &he_formula(n), "", view.x - 36.0),
            2 => vec![format!(r"$h_{{{n}}}(x) \sqrt{{\varphi(x)}}$")],
            _ => title_lines(f, &format!("h_{{{n}}}(x) = ("), &he_formula(n), &format!(r") / \sqrt{{{n}!}}"), view.x - 36.0),
        };
        // the title's lines, and their plate, in logical pixels: what is drawn near the
        // top keeps out of it (typeset, it is taller and wider than it was, and on a
        // phone it is most of the width of the view)
        let mut lines = Vec::new();
        let (mut y, mut widest) = (f.top_inset + 18.0, 0.0f32);
        for line in title {
            let s = f.text_size(&line, 17.0);
            widest = widest.max(s.x);
            lines.push((y, line));
            y += s.y + 4.0;
        }
        let (title_min, title_max) = (Vec2::new((view.x - widest) * 0.5 - 7.0, f.top_inset + 15.0), Vec2::new((view.x + widest) * 0.5 + 7.0, y - 1.0));

        if self.show_derivative && self.mode != 2 && n >= 1 {
            // numerical derivative of the curve, and the lowering rule's prediction on top of it
            let factor = self.lowering_factor();
            f.curve(lo.x, hi.x, samples, 4.0, pal::YELLOW.a(0.35), |x| Vec3::new(x, clip(factor * self.value(n - 1, x as f64)), 0.0));
            // one dashed path, so that the dashes run on along the whole curve (dashed
            // segments each start their pattern afresh, and come out solid this short)
            let slope: Vec<Vec3> = (0..=samples)
                .map(|i| {
                    let x = lo.x as f64 + (hi.x - lo.x) as f64 * i as f64 / samples as f64;
                    Vec3::new(x as f32, clip(self.slope(n, x)), 0.0)
                })
                .collect();
            f.dashed_path(slope, 1.8, pal::WHITE);
            let rule = if self.mode == 1 { format!(r"${}\, {}$", n, he_name(n - 1)) } else { format!(r"$\sqrt{{{}}}\, h_{{{}}}$", n, n - 1) };
            let yellow = format!("yellow: {rule}.");
            // under the title, which on a phone starts lower down
            let below = (f.top_inset + 48.0).max(title_max.y + 6.0);
            for (k, line) in caption_lines(f, &["dashed: the derivative of the curve.", &yellow, "They are the same."], "   ", 13.0, view.x - 36.0).into_iter().enumerate() {
                f.caption_px(Vec2::new(view.x * 0.5, below + k as f32 * 20.0), line, 13.0, pal::YELLOW, Align2::CENTER_TOP);
            }
        }

        f.curve(lo.x, hi.x, samples, 3.2, col, |x| Vec3::new(x, clip(self.value(n, x as f64)), 0.0));

        if self.show_lower {
            // name each thin curve where it stands furthest from the axis in the left half
            // of the picture: where it leaves the view, or at its outermost lobe
            let pad = span * 0.09;
            let left = lo.x + (hi.x - lo.x) * 0.03;
            let middle = (lo.x + hi.x) * 0.5;
            let (mut above, mut below) = (Vec::new(), Vec::new());
            for k in 0..n {
                let mut best: Option<Vec2> = None;
                for i in 0..=240 {
                    let x = left + (middle - left) * i as f32 / 240.0;
                    let y = self.value(k, x as f64) as f32;
                    if y > lo.y + pad && y < hi.y - pad && best.is_none_or(|b| y.abs() > b.y.abs() + 1e-4) {
                        best = Some(Vec2::new(x, y));
                    }
                }
                let Some(at) = best else { continue };
                let Some(p) = on_screen(f, at.extend(0.0)) else { continue };
                let name = match self.mode {
                    1 => format!("${}$", he_name(k)),
                    2 => format!(r"$h_{{{k}}} \sqrt{{\varphi}}$"),
                    _ => format!("$h_{{{k}}}$"),
                };
                // the low degrees are a dark blue: lighten the names so they can be read
                let c = degree_color(k, 12).mix(pal::WHITE, 0.45);
                if at.y >= 0.0 {
                    let mut q = p + Vec2::new(6.0, -4.0);
                    let s = f.text_size(&name, 13.0);
                    if q.x < title_max.x && q.x + s.x > title_min.x && q.y > title_min.y && q.y - s.y < title_max.y {
                        q.y = title_max.y + 2.0 + s.y;
                    }
                    above.push((q, name, c));
                } else {
                    below.push((p + Vec2::new(6.0, 4.0), name, c));
                }
            }
            spread_labels(f, above, 13.0, Align2::LEFT_BOTTOM);
            spread_labels(f, below, 13.0, Align2::LEFT_TOP);
        }

        if self.show_zeros {
            for z in zeros(n + 1) {
                f.dot((z as f32, 0.0), 6.0, pal::WHITE.a(0.75));
            }
            for z in zeros(n) {
                f.dot((z as f32, 0.0), 10.0, col);
            }
            if n >= 1 && !self.show_nodes {
                bottom_caption(f, &[&format!("dots on the axis: zeros of degree {} (large) and of degree {} (small). They interlace.", n, n + 1)], 12.0);
            }
        }
        if self.show_nodes && n >= 1 {
            for (x, stem) in Self::stems(n) {
                let height = stem as f32 * span * 0.28 / 0.3989;
                f.line((x as f32, 0.0), (x as f32, height), 2.2, pal::YELLOW);
                f.dot((x as f32, height), 8.0, pal::YELLOW);
            }
            let size = f.view_size();
            let text = format!(r"{n} nodes at the zeros of ${}$: this weighted sum equals the Gaussian integral exactly for every polynomial of degree $\le {}$", he_name(n), 2 * n - 1);
            let lines = caption_lines(f, &[&text], "", 13.0, size.x - 36.0);
            let count = lines.len();
            for (k, line) in lines.into_iter().enumerate() {
                let up = (count - 1 - k) as f32 * 20.0;
                f.caption_px(Vec2::new(size.x * 0.5, size.y - 22.0 - up), line, 13.0, pal::YELLOW, Align2::CENTER_BOTTOM);
            }
        }
        for (y, line) in lines {
            f.caption_px(Vec2::new(view.x * 0.5, y), line, 17.0, col, Align2::CENTER_TOP);
        }
    }
}

/// A title made of `head`, the polynomial `terms` (as `he_formula` writes it) and
/// `tail`, on as many lines as `room` (logical pixels) needs: on a phone held upright
/// the higher degrees are wider than the view. A line is broken before a sign, as a
/// book breaks a long formula, and each line is a formula of its own.
fn title_lines(f: &Frame, head: &str, terms: &str, tail: &str, room: f32) -> Vec<String> {
    let whole = format!("${head}{terms}{tail}$");
    if f.text_size(&whole, 17.0).x <= room {
        return vec![whole];
    }
    // the first term with what comes before it, then each sign with its term
    let words: Vec<&str> = terms.split(' ').collect();
    let Some((first, rest)) = words.split_first() else { return vec![whole] };
    let mut pieces = vec![format!("{head}{first}")];
    pieces.extend(rest.chunks(2).map(|pair| pair.join(" ")));
    if let Some(last) = pieces.last_mut() {
        last.push_str(tail);
    }
    let mut lines = Vec::new();
    let mut line = String::new();
    for piece in pieces {
        let joined = if line.is_empty() { piece.clone() } else { format!("{line} {piece}") };
        if !line.is_empty() && f.text_size(&format!("${joined}$"), 17.0).x > room {
            lines.push(format!("${line}$"));
            // (an empty group before the sign, so that it is set as one between two terms)
            line = format!("{{}} {piece}");
        } else {
            line = joined;
        }
    }
    lines.push(format!("${line}$"));
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    /// "Orthonormal with no weight at all": the third scaling's functions have plain
    /// integral of the product 1 for a function with itself and 0 otherwise. (They
    /// used to be drawn 1.58 times too large for that, against the same tick labels.)
    #[test]
    fn the_hermite_functions_are_orthonormal_without_a_weight() {
        let integral = |m: usize, n: usize| {
            let (steps, reach) = (40_000, 14.0);
            let dx = 2.0 * reach / steps as f64;
            (0..steps).map(|i| -reach + dx * (i as f64 + 0.5)).map(|x| HermiteFamily::value_in(2, m, x) * HermiteFamily::value_in(2, n, x) * dx).sum::<f64>()
        };
        for m in 0..=8 {
            for n in 0..=8 {
                let want = if m == n { 1.0 } else { 0.0 };
                assert!((integral(m, n) - want).abs() < 1e-6, "<{m}, {n}> = {}", integral(m, n));
            }
        }
        assert!((HermiteFamily::value_in(2, 0, 0.0) - 0.6316).abs() < 1e-4);
    }

    /// The identities the panel states as facts, at every degree the slider reaches,
    /// with the derivatives measured numerically rather than taken from the rules.
    #[test]
    fn the_identities_in_the_panel_hold() {
        let d = |n: usize, x: f64| (he(n, x + 1e-5) - he(n, x - 1e-5)) / 2e-5;
        let d2 = |n: usize, x: f64| (he(n, x + 1e-3) - 2.0 * he(n, x) + he(n, x - 1e-3)) / 1e-6;
        for n in 0..=12 {
            for x in [-3.1, -1.7, -0.4, 0.0, 0.3, 1.1, 2.2, 3.6] {
                // the size of the terms involved, which the errors of measurement follow
                let size = 1.0 + he(n + 1, x).abs() + (x * he(n, x)).abs() + n as f64 * he(n.max(1) - 1, x).abs();
                if n >= 1 {
                    // three-term rule and lowering
                    assert!((he(n + 1, x) - (x * he(n, x) - n as f64 * he(n - 1, x))).abs() < 1e-12 * size, "three-term, n = {n}, x = {x}");
                    assert!((d(n, x) - n as f64 * he(n - 1, x)).abs() < 1e-9 * size, "lowering, n = {n}, x = {x}");
                }
                // raising: x·Heₙ − Heₙ′ = Heₙ₊₁
                assert!((x * he(n, x) - d(n, x) - he(n + 1, x)).abs() < 1e-9 * size, "raising, n = {n}, x = {x}");
                // eigenfunction: Heₙ″ − x·Heₙ′ = −n·Heₙ (a second difference also errs
                // by the size of the next terms down)
                let size = size + (n * n.saturating_sub(1)) as f64 * he(n.saturating_sub(2), x).abs();
                assert!((d2(n, x) - x * d(n, x) + n as f64 * he(n, x)).abs() < 1e-5 * size, "eigenfunction, n = {n}, x = {x}");
            }
            // E[Heₙ(X)²] = n!
            let mean_square = gauss_mean(|x| he(n, x).powi(2));
            assert!((mean_square / factorial(n) - 1.0).abs() < 1e-9, "n = {n}: {mean_square}");
        }
        // and a wrong rule fails: x·Heₙ + Heₙ′ is not Heₙ₊₁
        assert!((1.1 * he(4, 1.1) + d(4, 1.1) - he(5, 1.1)).abs() > 1.0);
    }

    /// "They interlace", at every degree the slider reaches.
    #[test]
    fn the_zeros_interlace() {
        for n in 1..=12 {
            let (a, b) = (zeros(n), zeros(n + 1));
            assert!(a.len() == n && b.len() == n + 1, "n = {n}");
            for i in 0..n {
                assert!(b[i] < a[i] && a[i] < b[i + 1], "n = {n}");
            }
        }
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let mut scene = HermiteFamily::create();
        let xs = (0..=400).map(|i| -4.6 + 9.2 * i as f64 / 400.0);

        // the lowering rule: the measured slope of h₄ lies on 2·h₃ all across the view
        scene.set_up(0);
        assert!(scene.n == 4 && scene.mode == 0 && scene.show_derivative && !scene.show_nodes);
        assert_eq!(scene.lowering_factor(), 2.0);
        for x in xs.clone() {
            assert!((scene.slope(4, x) - 2.0 * scene.value(3, x)).abs() < 2e-7, "x = {x}");
        }
        // and the yellow curve is right in both scalings it is drawn for, at every degree
        // (to the accuracy of the measured slope, far below a pixel)
        for mode in [0, 1] {
            for n in 1..=12 {
                (scene.mode, scene.n) = (mode, n);
                for x in xs.clone() {
                    let rule = scene.lowering_factor() * scene.value(n - 1, x);
                    assert!((scene.slope(n, x) - rule).abs() < 5e-5 * (1.0 + rule.abs()), "mode {mode}, n = {n}, x = {x}");
                }
            }
        }

        // zeros interlace: five large dots, six small ones, alternating
        scene.set_up(1);
        assert!(scene.n == 5 && scene.show_zeros && !scene.show_derivative && !scene.show_nodes);
        let (a, b) = (zeros(5), zeros(6));
        assert!(a.len() == 5 && b.len() == 6);
        assert!((0..5).all(|i| b[i] < a[i] && a[i] < b[i + 1]));

        // Gauss–Hermite: six stems with their tips on the bell (to within a hundredth of
        // its 0.4 height), exact up to degree 11, and 9675 for E[X¹²] = 10395
        scene.set_up(2);
        assert!(scene.n == 6 && scene.show_nodes && scene.show_weight);
        let stems = HermiteFamily::stems(6);
        assert_eq!(stems.len(), 6);
        for &(x, height) in &stems {
            assert!((height - phi(x)).abs() < 0.005, "stem at {x}: {height} against {}", phi(x));
        }
        let (x, w) = gauss_hermite(6);
        let moment = |k: i32| x.iter().zip(&w).map(|(x, w)| w * x.powi(k)).sum::<f64>();
        for (k, exact) in [(0, 1.0), (2, 1.0), (4, 3.0), (6, 15.0), (8, 105.0), (10, 945.0)] {
            assert!((moment(k) - exact).abs() < 1e-9 * exact, "E[X^{k}]: {}", moment(k));
            assert!(moment(k + 1).abs() < 1e-9 * exact);
        }
        assert!((moment(12) - 9675.0).abs() < 1e-6 && (double_factorial_odd(6) - 10395.0).abs() < 1e-9, "{}", moment(12));
    }
}

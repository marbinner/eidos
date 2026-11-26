//! Derivative, integral, Taylor.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "calc1",
    title: "Derivative, integral, Taylor",
    group: "Calculus",
    blurb: "Type any $f(x)$: tangent and secant lines, Riemann sums, Taylor polynomials.",
    make: |g| Box::new(Calc1::new(g)),
};

/// One of the "things to try": everything the lesson depends on.
struct Trial {
    label: &'static str,
    look_for: &'static str,
    mode: usize,
    formula: &'static str,
    x0: f32,
    h: f32,
    ab: [f32; 2],
    n: usize,
    rule: usize,
    order: usize,
}

const TRIALS: [Trial; 4] = [
    Trial {
        label: "the secant becomes the tangent",
        look_for: r#"Shrink $h$ from 1 towards 0.01: the pink secant swings onto the yellow tangent, and each tenfold cut in $h$ cuts "$\text{secant} - \text{exact}$" about tenfold: −0.44 at $h = 1$, −0.037 at 0.1, −0.004 at 0.01."#,
        mode: 0,
        formula: "sin(x) + 0.3x",
        x0: 0.8,
        h: 1.0,
        ab: [-2.0, 2.5],
        n: 12,
        rule: 1,
        order: 3,
    },
    Trial {
        label: "a point with no tangent",
        look_for: r"$\sqrt{|x|}$ at its sharp point. The secant's slope is $1/\sqrt{h}$: shrink $h$ and it grows without limit (3.2 at $h = 0.1$, 10 at 0.01, 32 at 0.001), so there is no tangent for it to settle on.",
        mode: 0,
        formula: "sqrt(abs(x))",
        x0: 0.0,
        h: 1.0,
        ab: [-2.0, 2.5],
        n: 12,
        rule: 1,
        order: 3,
    },
    Trial {
        label: "rectangles close in on the area",
        look_for: "Four rectangles under a bell, each as tall as the curve at its left side: their sum is 0.20 too big. Raise their number: each doubling about halves the error (0.11 at 8, 0.047 at 20, 0.024 at 40). Then pick midpoint: at 40 it is a hundred times smaller still.",
        mode: 1,
        formula: "exp(-x^2/2)",
        x0: 0.8,
        h: 1.2,
        ab: [-1.0, 2.5],
        n: 4,
        rule: 0,
        order: 3,
    },
    Trial {
        label: "where Taylor gives up",
        look_for: r"$1/(1 + x^2)$ is smooth everywhere, yet raising the order from 4 to 14 only stretches the agreement from about $|x| < 0.64$ to $|x| < 0.86$, and no order gets it past 1.",
        mode: 2,
        formula: "1/(1+x^2)",
        x0: 0.0,
        h: 1.2,
        ab: [-2.0, 2.5],
        n: 12,
        rule: 1,
        order: 4,
    },
];

/// What can be said about the integral of f over an interval.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Area {
    /// It exists, and this is its value.
    Value(f32),
    /// f has no value at some points between these two.
    Undefined(f32, f32),
    /// f grows past any size near this point.
    Unbounded(f32),
}

/// The integral of `f` from `a` to `b`, by Simpson's rule on 2000 intervals, or why
/// it does not exist. (A rectangle sum or a quadrature rule simply adds up what it
/// is given: across a pole or a gap in f's domain it still gives a number, and that
/// number means nothing.)
fn area(f: &impl Fn(f32) -> f32, a: f32, b: f32) -> Area {
    if b < a {
        return match area(f, b, a) {
            Area::Value(v) => Area::Value(-v),
            other => other,
        };
    }
    if b.is_nan() || a.is_nan() || a == b {
        return Area::Value(0.0);
    }
    const M: usize = 2000;
    let at = |i: usize| a + (b - a) * i as f32 / M as f32;
    let mut ys = [0.0f32; M + 1];
    let mut gap: Option<(f32, f32)> = None;
    let mut infinite = None;
    let mut peak = (0, -1.0f32);
    for (i, y) in ys.iter_mut().enumerate() {
        *y = f(at(i));
        if y.is_nan() {
            gap = Some(gap.map_or((at(i), at(i)), |g| (g.0, at(i))));
        } else if y.is_infinite() {
            infinite = infinite.or(Some(at(i)));
        } else if y.abs() > peak.1 {
            peak = (i, y.abs());
        }
    }
    if let Some((lo, hi)) = gap {
        return Area::Undefined(lo, hi);
    }
    if let Some(x) = infinite.or_else(|| blow_up(f, at(peak.0), a, b, (b - a) / M as f32)) {
        return Area::Unbounded(x);
    }
    let sum: f64 = ys.iter().enumerate().map(|(i, &y)| y as f64 * if i == 0 || i == M { 1.0 } else if i % 2 == 1 { 4.0 } else { 2.0 }).sum();
    Area::Value((sum * (b - a) as f64 / M as f64 / 3.0) as f32)
}

/// Does `f` grow without bound near `x`, the sample (`h` apart from the next) where
/// it is largest in size? First the largest value is looked for ever more closely,
/// as finely as the numbers allow; then f is read at distances from there that halve
/// each time. A bounded f soon stops growing (its steps shrink: a smooth top, a
/// corner, a narrow but finite peak); one that blows up keeps growing by at least as
/// much at each halving (a pole by more each time, a logarithm by the same amount).
fn blow_up(f: &impl Fn(f32) -> f32, x: f32, a: f32, b: f32, h: f32) -> Option<f32> {
    let (mut c, mut w, mut peak) = (x, h, f(x).abs());
    for _ in 0..4 {
        let (lo, hi) = ((c - w).max(a), (c + w).min(b));
        for k in 0..=32 {
            let x = lo + (hi - lo) * k as f32 / 32.0;
            let y = f(x).abs();
            if y.is_infinite() {
                return Some(x);
            }
            if y > peak {
                (peak, c) = (y, x);
            }
        }
        w /= 8.0;
    }
    // (the finest spacing looked at, or the spacing of the numbers themselves there)
    let res = (w / 2.0).max(c.abs() * f32::EPSILON);
    let mut sizes = Vec::new();
    let mut r = h;
    while r >= 32.0 * res && sizes.len() < 40 {
        let near = [c - r, c + r].into_iter().filter(|x| *x >= a && *x <= b).map(|x| f(x).abs()).filter(|y| !y.is_nan());
        sizes.push(near.fold(0.0f32, f32::max));
        r *= 0.5;
    }
    let n = sizes.len();
    if n < 6 {
        return None;
    }
    let (s4, s3, s2, s0) = (sizes[n - 5], sizes[n - 4], sizes[n - 3], sizes[n - 1]);
    let grows = s0 > 1.05 * s3 && s2 > s4 && s0 - s2 >= 0.9 * (s2 - s4);
    grows.then_some(c)
}

/// How far from x₀ a Taylor polynomial stays within 0.05 of f, among the points in view.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Reach {
    /// The polynomial has no value: f has none at x₀ (`true`), or no derivatives there.
    NoSeries(bool),
    /// x₀ itself is out of view, so the stretch round it cannot be looked at.
    OutOfView,
    /// It agrees for |x − x₀| below this, and not just beyond it: where the gap passes
    /// 0.05, between the nearest point sampled where they disagree and the one before
    /// it (not at that point, which would overstate the reach by up to a sample's width).
    Within(f32),
    /// It agrees at every point in view, out to this distance from x₀.
    WholeView(f32),
}

/// See [`Reach`]: `xs` are the points in view, from left to right, `ys` f there and
/// `poly` the polynomial. A point where f has no value is one where they disagree.
fn reach(xs: &[f32], ys: &[f32], poly: &[f32], x0: f32, series: Option<bool>) -> Reach {
    if let Some(no_value) = series {
        return Reach::NoSeries(no_value);
    }
    let (Some(&first), Some(&last)) = (xs.first(), xs.last()) else { return Reach::OutOfView };
    if !(x0 >= first && x0 <= last) {
        return Reach::OutOfView;
    }
    // how far the gap is over 0.05 (they agree where this is not above zero); where
    // either has no value, they do not agree
    let n = xs.len().min(ys.len()).min(poly.len());
    let over = |i: usize| if ys[i].is_finite() && poly[i].is_finite() { (ys[i] - poly[i]).abs() - 0.05 } else { f32::INFINITY };
    // where the gap passes 0.05 between the point `i`, where they disagree, and the
    // point `j` before it on the way from x₀, where they agree: on the straight line
    // between the gaps at the two (or, if f has no value at `i`, at `j` itself)
    let crossing = |i: usize, j: Option<usize>| match j.filter(|&j| over(j) <= 0.0) {
        Some(j) if over(i).is_finite() => {
            let t = over(j) / (over(j) - over(i));
            (xs[j] + t * (xs[i] - xs[j]) - x0).abs()
        }
        Some(j) => (xs[j] - x0).abs(),
        None => (xs[i] - x0).abs(),
    };
    // the nearest point where they disagree on each side of x₀
    let mut near = f32::INFINITY;
    if let Some(i) = (0..n).find(|&i| xs[i] >= x0 && over(i) > 0.0) {
        near = near.min(crossing(i, i.checked_sub(1)));
    }
    if let Some(i) = (0..n).rev().find(|&i| xs[i] <= x0 && over(i) > 0.0) {
        near = near.min(crossing(i, (i + 1 < n).then_some(i + 1)));
    }
    if near.is_finite() {
        Reach::Within(near)
    } else {
        Reach::WholeView((x0 - first).max(last - x0))
    }
}

/// A place on the x axis for a sentence: two decimals, and no "−0.00".
fn place(x: f32) -> String {
    let s = format!("{x:.2}");
    if s == "-0.00" {
        "0.00".into()
    } else {
        s.replace('-', "−")
    }
}

/// A read-out that may have no value (a number as a formula: `fmt` may write 1.5e-4).
fn number(v: f32) -> String {
    if v.is_finite() {
        format!("${}$", fmt(v))
    } else {
        "none".into()
    }
}

pub struct Calc1 {
    f: ExprField,
    mode: usize,
    x0: f32,
    h: f32,
    a: f32,
    b: f32,
    n: usize,
    rule: usize,
    order: usize,
    show_deriv: bool,
    xs: Vec<f32>,
    ys: Vec<f32>,
    batch: Batch,
    info: [f32; 4],
    area: Area,
    reach: Reach,
}

impl Calc1 {
    pub fn new(_: &Gfx) -> Self {
        Self::create()
    }

    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        Calc1 {
            f: ExprField::new("sin(x) + 0.3x", &["x"]),
            mode: 0,
            x0: 0.8,
            h: 1.2,
            a: -2.0,
            b: 2.5,
            n: 12,
            rule: 1,
            order: 3,
            show_deriv: false,
            xs: Vec::new(),
            ys: Vec::new(),
            batch: Batch::default(),
            info: [0.0; 4],
            area: Area::Value(0.0),
            reach: Reach::OutOfView,
        }
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.f.set(t.formula);
        (self.mode, self.x0, self.h, self.a, self.b, self.n, self.rule, self.order) = (t.mode, t.x0, t.h, t.ab[0], t.ab[1], t.n, t.rule, t.order);
        self.show_deriv = false;
    }

    fn val(&self, x: f32) -> f32 {
        self.f.expr.eval(&[x])
    }

    fn slope(&self, x: f32) -> f32 {
        self.f.expr.grad2(&[x], 0, 9).1
    }

    /// The `i`th of the `n` rectangles of width `w` from `lo`: its two sides, and the
    /// heights at them (the same but for the trapezoid rule).
    fn rectangle(&self, lo: f32, w: f32, i: usize) -> (f32, f32, f32, f32) {
        let (xl, xr) = (lo + w * i as f32, lo + w * (i + 1) as f32);
        let (yl, yr) = match self.rule {
            0 => (self.val(xl), self.val(xl)),
            1 => {
                let m = self.val((xl + xr) * 0.5);
                (m, m)
            }
            2 => (self.val(xr), self.val(xr)),
            _ => (self.val(xl), self.val(xr)),
        };
        (xl, xr, yl, yr)
    }

    /// The sum of the rectangles, taken from a to b (so it changes sign when b is left
    /// of a); not a number if a rectangle has no height.
    fn riemann(&self) -> f32 {
        let (lo, hi) = (self.a.min(self.b), self.a.max(self.b));
        let w = (hi - lo) / self.n as f32;
        let sum: f32 = (0..self.n)
            .map(|i| {
                let (_, _, yl, yr) = self.rectangle(lo, w, i);
                0.5 * (yl + yr) * w
            })
            .sum();
        if self.b < self.a {
            -sum
        } else {
            sum
        }
    }

    /// The integral from a to b (see [`area`]).
    fn integral(&self) -> Area {
        area(&|x| self.val(x), self.a, self.b)
    }

    /// The Taylor series at x₀, if it has a value there (`Err`: whether f itself has none).
    fn series(&self) -> Result<crate::math::expr::Jet, bool> {
        let jet = self.f.expr.taylor(&[self.x0], 0);
        if jet.c[..=self.order].iter().all(|c| c.is_finite()) {
            Ok(jet)
        } else {
            Err(!jet.c[0].is_finite())
        }
    }
}

/// Where the handle that moves x₀ is put: on the curve, or as near to it as the
/// view allows; and on the x axis where the function has no value (left of zero for
/// a square root or a logarithm). It is the only control for x₀, so it must never be
/// somewhere it cannot be seen and taken hold of.
fn handle_at(x0: f32, y0: f32, lo: Vec2, hi: Vec2) -> Vec2 {
    Vec2::new(x0, if y0.is_finite() { y0.max(lo.y).min(hi.y) } else { 0.0f32.max(lo.y).min(hi.y) })
}

impl Scene for Calc1 {
    fn view(&self) -> View {
        View::Flat { center: Vec2::new(0.0, 0.3), half_height: 3.4 }
    }

    fn animated(&self) -> bool {
        false
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.2;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "Three ideas of calculus, tried on any curve $y = f(x)$ you type: the slope of the curve at a point (the derivative), the area between the curve and the $x$ axis (the integral), and a polynomial that copies the curve near a point (a Taylor polynomial).");
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The function");
        self.f.ui(ui, "f(x) =");
        if let Some(i) = ui::buttons(ui, &["sin", "bell", "cubic", "$1/(1 + x^2)$", "wiggle", r"$\sqrt{|x|}$", "exp"]) {
            self.f.set(["sin(x) + 0.3x", "exp(-x^2/2)", "x^3/4 - x", "1/(1+x^2)", "sin(3x)/(1+x^2) + x/4", "sqrt(abs(x))", "exp(x/2)/2"][i]);
        }
        ui::section(ui, "Explore");
        ui::choice(ui, &["derivative", "integral", "Taylor series"], &mut self.mode);
        ui.add_space(4.0);
        match self.mode {
            0 => {
                ui::note(ui, "The derivative is the slope the secant line settles on as $h$ shrinks to zero:");
                ui::equation(ui, r"f'(x_0) = \lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h}");
                ui::note(ui, "Drag the yellow point along the curve; shrink $h$.");
                ui::slider_log(ui, "$h$", &mut self.h, 0.001..=3.0);
                ui::note(ui, "$h$: how far apart the secant's two points are.");
                // (egui's own checkbox sets its label as plain text)
                let plot = r"plot $f'(x)$";
                crate::typeset::seen(plot);
                let plot = ui::label_text(ui, plot, 12.5, None, false);
                ui.checkbox(&mut self.show_deriv, plot);
                ui::section(ui, "Reading the picture");
                let mut key = vec![(pal::CYAN, "the curve $y = f(x)$"), (pal::PINK, r"the secant through two points $h$ apart: its slope is $\text{rise}/h$"), (pal::YELLOW, "the tangent at $x_0$: the curve's own slope there")];
                if self.show_deriv {
                    key.push((pal::YELLOW.a(0.55), "faint: $f'(x)$, the slope at every $x$"));
                }
                ui::key(ui, &key);
                ui::section(ui, "Words");
                ui::term(ui, "Secant:", "a straight line through two points of the curve.");
                ui::term(ui, "Tangent:", "the line that grazes the curve at one point, with the curve's slope there.");
                ui::section(ui, "Numbers");
                ui::readout_colored(ui, "secant slope", number(self.info[0]), pal::PINK);
                ui::readout_colored(ui, "$f'(x_0)$ exact", number(self.info[1]), pal::YELLOW);
                ui::readout(ui, r"$\text{secant} - \text{exact}$", number(self.info[0] - self.info[1]));
                if !self.info[2].is_finite() {
                    ui::note(ui, "$f$ has no value at $x_0$, so there is no slope to find.");
                } else if !self.info[1].is_finite() {
                    ui::note(ui, r"The rule for $f'$ gives no number at $x_0$ (as for $\sqrt{|x|}$ at 0, where the slope grows past any size).");
                }
            }
            1 => {
                ui::note(ui, "The integral from $a$ to $b$ is the area between the curve and the $x$ axis, counted negative below the axis: the number that sums of ever thinner rectangles close in on. Drag $a$ and $b$; add rectangles.");
                ui::slider_int(ui, "rectangles", &mut self.n, 1..=200);
                ui::choice(ui, &["left", "midpoint", "right", "trapezoid"], &mut self.rule);
                ui::note(ui, "Where each rectangle takes its height: the curve at its left side, its middle or its right side; a trapezoid joins the two sides instead.");
                ui::section(ui, "Reading the picture");
                ui::key(ui, &[(pal::CYAN, "the curve $y = f(x)$"), (pal::BLUE, "rectangles above the axis: their area counts plus"), (pal::PINK, "rectangles below the axis: their area counts minus"), (pal::WHITE, "$a$ and $b$, the ends (drag them)")]);
                ui::section(ui, "Numbers");
                let sum = self.info[0];
                ui::readout(ui, "sum of the rectangles", number(sum));
                let exact = match self.area {
                    Area::Value(v) => v,
                    _ => f32::NAN,
                };
                ui::readout(ui, r"$\int_a^b f(x)\,\mathrm{d}x$", if exact.is_finite() { number(exact) } else { "does not exist".into() });
                if sum.is_finite() && exact.is_finite() {
                    ui::readout(ui, "error", number(sum - exact));
                }
                match self.area {
                    Area::Undefined(lo, hi) => ui::note(ui, &format!("$f$ is undefined on part of $[a, b]$: it has no value at some points between $x = {}$ and $x = {}$.", place(lo), place(hi))),
                    Area::Unbounded(x) => ui::note(ui, &format!(r"$f$ is unbounded near $x \approx {}$: it grows past any size there, and a rectangle whose height is taken close enough to that point can make the sum as large as you like.", place(x))),
                    Area::Value(_) if !sum.is_finite() => ui::note(ui, "A rectangle has no height: $f$ has no value where it is measured."),
                    Area::Value(_) => {}
                }
                if self.b < self.a {
                    ui::note(ui, "$b$ is left of $a$, so the integral from $a$ to $b$ is minus the area from $b$ to $a$: every rectangle counts with the opposite sign.");
                }
            }
            _ => {
                ui::note(ui, "A Taylor polynomial of order $n$ matches the function's value and first $n$ derivatives at one point, $x_0$:");
                ui::equation(ui, r"f(x_0 + h) \approx \sum_{k=0}^{n} \frac{f^{(k)}(x_0)}{k!}\, h^k");
                ui::note(ui, "Drag the orange point; raise the order and watch the stretch where the two agree grow (or refuse to).");
                ui::slider_int(ui, "order", &mut self.order, 0..=crate::math::expr::JET_ORDER);
                ui::note(ui, "order: the highest power of $h$ in the polynomial ($h = x - x_0$).");
                ui::section(ui, "Reading the picture");
                ui::key(ui, &[(pal::CYAN, "the curve $y = f(x)$"), (pal::ORANGE, "the Taylor polynomial at $x_0$, written out at the top"), (pal::ORANGE.a(0.35), "shaded: the gap between the two")]);
                ui::section(ui, "Numbers");
                ui::readout(ui, "$x_0$", number(self.x0));
                let reach = match self.reach {
                    Reach::NoSeries(_) => "none".to_string(),
                    Reach::OutOfView => "$x_0$ is out of view".to_string(),
                    Reach::Within(r) => format!("$|x - x_0| < {}$", fmt(r)),
                    Reach::WholeView(d) => format!(r"$|x - x_0| \le {}$ (all in view)", fmt(d)),
                };
                ui::readout(ui, "agrees within 0.05 for", reach);
                match self.reach {
                    Reach::NoSeries(true) => ui::note(ui, "No Taylor series at $x_0$: $f$ is not defined there."),
                    Reach::NoSeries(false) => ui::note(ui, "No Taylor series at $x_0$ of this order: $f$'s derivatives have no value there."),
                    Reach::WholeView(_) => ui::note(ui, "They agree everywhere in view: zoom out to see how far it goes."),
                    _ => {}
                }
            }
        }
        ui::see_also(ui, &[("surface", "slopes of a surface: partial derivatives and the gradient"), ("fourier", "a curve built from circles instead of powers"), ("graph", "plot several functions at once")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let (lo, hi) = f.cam.flat_bounds();
        let n = 900;
        self.xs.clear();
        self.xs.extend((0..=n).map(|i| lo.x + (hi.x - lo.x) * i as f32 / n as f32));
        self.ys.resize(self.xs.len(), 0.0);
        self.f.expr.eval_batch(&[&self.xs], &mut self.ys, &mut self.batch);

        match self.mode {
            0 => {
                if self.show_deriv {
                    let d: Vec<f32> = self.xs.iter().map(|&x| self.slope(x)).collect();
                    plot_samples(f, &self.xs, &d, 1.6, pal::YELLOW.a(0.55), false);
                }
                let (x0, y0) = (self.x0, self.val(self.x0));
                let (x1, y1) = (x0 + self.h, self.val(x0 + self.h));
                let m_sec = (y1 - y0) / self.h;
                let m_tan = self.slope(x0);
                self.info = [m_sec, m_tan, y0, 0.0];
                let w = hi.x - lo.x;
                // (where f has no value, at x₀ or a step h on, there is no secant to draw;
                // where its derivative has none, no tangent)
                if m_sec.is_finite() {
                    // rise over run
                    f.dl.tris_alpha.tri(Vec3::new(x0, y0, 0.0), Vec3::new(x1, y0, 0.0), Vec3::new(x1, y1, 0.0), pal::PINK.a(0.13));
                    f.dashed((x0, y0), (x1, y0), 1.4, pal::DIM);
                    f.dashed((x1, y0), (x1, y1), 1.4, pal::DIM);
                    // the run and the secant's slope are written on the far side of the base
                    // from the triangle; the rise beside its own edge, when there is room
                    let (side, align) = if y1 >= y0 { (1.0, Align2::CENTER_TOP) } else { (-1.0, Align2::CENTER_BOTTOM) };
                    f.text_off(((x0 + x1) * 0.5, y0), Vec2::new(0.0, 6.0 * side), format!("$h = {}$", fmt(self.h)), 12.0, pal::DIM, align);
                    f.text_off(((x0 + x1) * 0.5, y0), Vec2::new(0.0, 23.0 * side), format!("secant slope ${}$", fmt(m_sec)), 13.0, pal::PINK, align);
                    let px_per_unit = f.view_size().y / (hi.y - lo.y);
                    if (y1 - y0).abs() * px_per_unit > 22.0 {
                        f.text_off((x1, (y0 + y1) * 0.5), Vec2::new(8.0, 0.0), format!("rise ${}$", fmt(y1 - y0)), 12.0, pal::DIM, Align2::LEFT_CENTER);
                    }
                    f.line((x0 - w, y0 - m_sec * w), (x0 + w, y0 + m_sec * w), 1.8, pal::PINK);
                }
                if y0.is_finite() && m_tan.is_finite() {
                    f.line((x0 - w, y0 - m_tan * w), (x0 + w, y0 + m_tan * w), 2.2, pal::YELLOW);
                }
                plot_samples(f, &self.xs, &self.ys, 3.0, pal::CYAN, false);
                if m_sec.is_finite() {
                    f.dot((x1, y1), 9.0, pal::PINK);
                }
                let mut p = handle_at(x0, y0, lo, hi);
                if f.drag2(1, &mut p, pal::YELLOW) {
                    self.x0 = p.x;
                }
                // two short lines rather than one long one, which would reach across the y axis
                if !y0.is_finite() {
                    f.text_off(p, Vec2::new(-12.0, -14.0), "$f$ has no value here", 14.0, pal::YELLOW, Align2::RIGHT_BOTTOM);
                } else if !m_tan.is_finite() {
                    f.text_off(p, Vec2::new(-12.0, -14.0), "no tangent here", 14.0, pal::YELLOW, Align2::RIGHT_BOTTOM);
                } else {
                    f.text_off(p, Vec2::new(-12.0, -31.0), "tangent", 14.0, pal::YELLOW, Align2::RIGHT_BOTTOM);
                    f.text_off(p, Vec2::new(-12.0, -14.0), format!("slope ${}$", fmt(m_tan)), 14.0, pal::YELLOW, Align2::RIGHT_BOTTOM);
                }
            }
            1 => {
                let (a, b) = (self.a.min(self.b), self.a.max(self.b));
                let n = self.n;
                let w = (b - a) / n as f32;
                for i in 0..n {
                    let (xl, xr, yl, yr) = self.rectangle(a, w, i);
                    if !(yl.is_finite() && yr.is_finite()) {
                        continue;
                    }
                    let c = if yl + yr >= 0.0 { pal::BLUE } else { pal::PINK };
                    let q = [Vec3::new(xl, 0.0, 0.0), Vec3::new(xr, 0.0, 0.0), Vec3::new(xr, yr, 0.0), Vec3::new(xl, yl, 0.0)];
                    f.dl.tris_alpha.quad(q[0], q[1], q[2], q[3], c.a(0.34));
                    if n <= 80 {
                        f.dl.lines.closed(&q, 1.0, c.a(0.9));
                    }
                }
                let sum = self.riemann();
                self.area = self.integral();
                self.info = [sum, 0.0, 0.0, 0.0];
                plot_samples(f, &self.xs, &self.ys, 3.0, pal::CYAN, false);
                let (mut pa, mut pb) = (Vec2::new(self.a, 0.0), Vec2::new(self.b, 0.0));
                if f.drag2(1, &mut pa, pal::WHITE) {
                    self.a = pa.x;
                }
                if f.drag2(2, &mut pb, pal::WHITE) {
                    self.b = pb.x;
                }
                f.text_off((self.a, 0.0), Vec2::new(0.0, 12.0), "$a$", 14.0, pal::WHITE, Align2::CENTER_TOP);
                f.text_off((self.b, 0.0), Vec2::new(0.0, 12.0), "$b$", 14.0, pal::WHITE, Align2::CENTER_TOP);
                if sum.is_finite() {
                    f.text(((a + b) * 0.5, 0.0), format!(r"$\approx {}$", fmt(sum)), 15.0, pal::WHITE, Align2::CENTER_BOTTOM);
                }
            }
            _ => {
                let x0 = self.x0;
                let series = self.series();
                let order = self.order;
                let poly: Vec<f32> = match &series {
                    Ok(jet) => self.xs.iter().map(|&x| jet.eval(x - x0, order)).collect(),
                    Err(_) => vec![f32::NAN; self.xs.len()],
                };
                // shade the gap between function and polynomial
                let clampy = |v: f32| v.max(lo.y - 5.0).min(hi.y + 5.0);
                for i in 0..self.xs.len() - 1 {
                    let (y, p, y2, p2) = (self.ys[i], poly[i], self.ys[i + 1], poly[i + 1]);
                    if !(y.is_finite() && p.is_finite() && y2.is_finite() && p2.is_finite()) {
                        continue;
                    }
                    let (xa, xb) = (self.xs[i], self.xs[i + 1]);
                    f.dl.tris_alpha.quad(
                        Vec3::new(xa, clampy(y), 0.0),
                        Vec3::new(xb, clampy(y2), 0.0),
                        Vec3::new(xb, clampy(p2), 0.0),
                        Vec3::new(xa, clampy(p), 0.0),
                        pal::ORANGE.a(0.12),
                    );
                }
                self.reach = reach(&self.xs, &self.ys, &poly, x0, series.as_ref().err().copied());
                plot_samples(f, &self.xs, &self.ys, 3.0, pal::CYAN, false);
                plot_samples(f, &self.xs, &poly, 2.6, pal::ORANGE, false);
                let mut p = handle_at(x0, self.val(x0), lo, hi);
                if f.drag2(1, &mut p, pal::ORANGE) {
                    self.x0 = p.x;
                }
                let size = f.view_size();
                let title = match &series {
                    Ok(jet) => {
                        let mut terms = String::new();
                        for k in 0..=order.min(4) {
                            let c = jet.c[k];
                            let sign = if c < 0.0 { "-" } else if k > 0 { "+" } else { "" };
                            let body = match k {
                                0 => format!("{:.3}", c.abs()),
                                1 => format!("{:.3} h", c.abs()),
                                _ => format!("{:.3} h^{k}", c.abs()),
                            };
                            terms.push_str(&format!(" {sign} {body}"));
                        }
                        if order > 4 {
                            terms.push_str(r" + \cdots");
                        }
                        format!(r"$f(x_0 + h) \approx{terms}$")
                    }
                    Err(true) => "no Taylor polynomial at $x_0$: $f$ has no value there".to_string(),
                    Err(false) => "no Taylor polynomial of this order at $x_0$: $f$'s derivatives have no value there".to_string(),
                };
                // (smaller where it would not fit across the view: a phone held upright)
                let room = size.x - 36.0;
                let wide = f.text_size(&title, 16.0).x;
                let px = if wide > room { (16.0 * room / wide).max(10.0) } else { 16.0 };
                f.text_px(Vec2::new(size.x * 0.5, f.top_inset + 20.0), title, px, pal::ORANGE, Align2::CENTER_TOP);
            }
        }
        // name the curve near the right-hand edge, above it on the side where it falls away
        let w = hi.x - lo.x;
        let xl = hi.x - 0.06 * w;
        let yl = self.val(xl);
        if yl.is_finite() && yl > lo.y && yl < hi.y {
            // (where the view is too narrow for it on the right, a phone held upright, it
            // goes on the left and below, which the falling curve leaves clear)
            let name = "$f(x)$";
            let room = 0.06 * f.view_size().x - 8.0;
            let (off, align) = if self.slope(xl) <= 0.0 {
                if f.text_size(name, 14.0).x <= room {
                    (Vec2::new(8.0, -8.0), Align2::LEFT_BOTTOM)
                } else {
                    (Vec2::new(-8.0, 8.0), Align2::RIGHT_TOP)
                }
            } else {
                (Vec2::new(-8.0, -8.0), Align2::RIGHT_BOTTOM)
            };
            f.text_off((xl, yl), off, name, 14.0, pal::CYAN, align);
        }
        f.axes2();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_handle_can_always_be_reached() {
        let (lo, hi) = (Vec2::new(-4.0, -2.0), Vec2::new(4.0, 3.0));
        assert_eq!(handle_at(1.0, 0.5, lo, hi), Vec2::new(1.0, 0.5));
        // sqrt(-1), ln(-1), 1/0: on the axis
        for y in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(handle_at(-1.0, y, lo, hi), Vec2::new(-1.0, 0.0));
        }
        // a value far outside the view: at its edge
        assert_eq!(handle_at(2.0, 1e9, lo, hi), Vec2::new(2.0, 3.0));
        assert_eq!(handle_at(2.0, -50.0, lo, hi), Vec2::new(2.0, -2.0));
        // the axis itself out of view: still inside
        assert_eq!(handle_at(0.0, f32::NAN, Vec2::new(-1.0, 2.0), Vec2::new(1.0, 5.0)).y, 2.0);
    }

    fn integral_of(formula: &str, a: f32, b: f32) -> Area {
        let e = Expr::parse(formula, &["x"]).unwrap();
        area(&|x| e.eval(&[x]), a, b)
    }

    /// How far the scene's Taylor polynomial agrees with f, over a view from `lo` to
    /// `hi` sampled as the frame samples it.
    fn reach_in(s: &Calc1, lo: f32, hi: f32) -> Reach {
        let xs: Vec<f32> = (0..=900).map(|i| lo + (hi - lo) * i as f32 / 900.0).collect();
        let ys: Vec<f32> = xs.iter().map(|&x| s.val(x)).collect();
        let series = s.series();
        let poly: Vec<f32> = xs.iter().map(|&x| series.as_ref().map_or(f32::NAN, |jet| jet.eval(x - s.x0, s.order))).collect();
        reach(&xs, &ys, &poly, s.x0, series.err())
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let mut s = Calc1::create();
        // the secant settles on the tangent, its error shrinking with h
        s.set_up(0);
        let error = |s: &Calc1, h: f32| (s.val(s.x0 + h) - s.val(s.x0)) / h - s.slope(s.x0);
        let errors = [1.0, 0.1, 0.01].map(|h| error(&s, h));
        assert!((errors[0] + 0.44).abs() < 0.005 && (errors[1] + 0.037).abs() < 0.0005 && fmt(errors[2]) == "-0.004", "{errors:?}");
        assert!(errors.windows(2).all(|w| (8.0..14.0).contains(&(w[0] / w[1]))), "{errors:?}");
        // a cusp: the secants steepen without limit, and no tangent is drawn
        s.set_up(1);
        assert!(!s.slope(s.x0).is_finite());
        for (h, slope) in [(0.1, 3.16), (0.01, 10.0), (0.001, 31.6)] {
            let m = (s.val(s.x0 + h) - s.val(s.x0)) / h;
            assert!((m / slope - 1.0).abs() < 0.01, "h {h}: {m}");
        }
        // rectangles: doubling them about halves the error; midpoints do far better
        s.set_up(2);
        let Area::Value(exact) = s.integral() else { panic!("the bell has an area") };
        let mut error_at = |n: usize, rule: usize| {
            (s.n, s.rule) = (n, rule);
            s.riemann() - exact
        };
        let first = error_at(4, 0);
        assert!((first - 0.20).abs() < 0.005, "{first}");
        for (n, e) in [(8, 0.11), (20, 0.047), (40, 0.024)] {
            assert!((error_at(n, 0) / e - 1.0).abs() < 0.03, "{n}: {}", error_at(n, 0));
        }
        assert!(error_at(40, 0) / error_at(40, 1) > 90.0, "midpoint {}", error_at(40, 1));
        // Taylor: stuck short of 1, however high the order (in the view as it opens,
        // and in a wider one)
        s.set_up(3);
        for (lo, hi) in [(-3.7, 3.7), (-6.0, 6.0)] {
            let at = |s: &mut Calc1, order: usize| {
                s.order = order;
                match reach_in(s, lo, hi) {
                    Reach::Within(r) => r,
                    other => panic!("order {order}: {other:?}"),
                }
            };
            // (exactly where the gap x^(n+2)/(1 + x²) passes 0.05: 0.64300 at order 4,
            // 0.85836 at order 14; the read-out is no longer that of the first point
            // sampled beyond, which said 0.649)
            let (four, fourteen) = (at(&mut s, 4), at(&mut s, 14));
            assert!((four - 0.642996).abs() < 0.0005 && (fourteen - 0.858360).abs() < 0.0005, "{four} {fourteen}");
            assert_eq!(fmt(four), "0.643");
            assert!((0..=crate::math::expr::JET_ORDER).all(|order| at(&mut s, order) < 1.0));
        }
    }

    /// An integral is given only when there is one: not across a pole, nor where f
    /// has no value (they used to be given as numbers: 11.92 for 1/x on [−2, 2.5]).
    #[test]
    fn an_integral_that_does_not_exist_says_so() {
        for formula in ["1/x", "1/x^2", "ln(abs(x))", "1/sqrt(abs(x))"] {
            match integral_of(formula, -2.0, 2.5) {
                Area::Unbounded(x) => assert!(x.abs() < 1e-3, "{formula}: {x}"),
                other => panic!("{formula}: {other:?}"),
            }
        }
        match integral_of("tan(x)", -2.0, 2.5) {
            Area::Unbounded(x) => assert!((x.abs() - PI / 2.0).abs() < 1e-3, "{x}"),
            other => panic!("tan: {other:?}"),
        }
        // (a sample right on the pole, and one beside a pole at an end)
        assert!(matches!(integral_of("1/x", -1.0, 1.0), Area::Unbounded(_)));
        assert!(matches!(integral_of("1/(x-2)", 1.0, 3.0), Area::Unbounded(_)));
        match integral_of("sqrt(x)", -2.0, 2.5) {
            Area::Undefined(lo, hi) => assert!(lo == -2.0 && hi < 0.0 && hi > -0.01, "{lo} {hi}"),
            other => panic!("sqrt: {other:?}"),
        }
        assert!(matches!(integral_of("sqrt(-1-x^2)", -2.0, 2.5), Area::Undefined(..)));
        // ordinary functions keep their value, bounded ones with a jump or a sharp
        // top among them
        for (formula, a, b, value) in [
            ("sin(x) + 0.3x", -2.0, 2.5, (-2.0f32).cos() - 2.5f32.cos() + 0.15 * 6.25 - 0.6),
            ("sqrt(x)", 0.0, 2.5, 2.0 / 3.0 * 2.5f32.powf(1.5)),
            ("3", -2.0, 2.5, 13.5),
            ("1/x", 0.5, 3.0, 6.0f32.ln()),
            ("sign(x)", -2.0, 2.5, 0.5),
            ("1 - abs(x)^0.1", -2.0, 2.5, 4.5 - (2.0f32.powf(1.1) + 2.5f32.powf(1.1)) / 1.1),
        ] {
            match integral_of(formula, a, b) {
                Area::Value(v) => assert!((v - value).abs() < 5e-3, "{formula}: {v} against {value}"),
                other => panic!("{formula}: {other:?}"),
            }
        }
    }

    /// The integral from a to b with b left of a is minus the area between them, and
    /// the panel says so.
    #[test]
    fn the_integral_runs_from_a_to_b() {
        let mut s = Calc1::create();
        s.set_up(2);
        let (forward, sum) = (s.integral(), s.riemann());
        (s.a, s.b) = (s.b, s.a);
        let (Area::Value(there), Area::Value(back)) = (forward, s.integral()) else { panic!() };
        assert!(there > 2.0 && back == -there && s.riemann() == -sum && sum > 0.0);
    }

    /// "Agrees within 0.05 for |x − x₀| < r" is said only of what was compared: not
    /// "< inf" where there is no polynomial, nor where the view ran out first.
    #[test]
    fn the_taylor_reach_is_what_was_seen() {
        let mut s = Calc1::create();
        s.set_up(3);
        // no value at x₀: no series
        for (formula, x0) in [("ln(x)", -1.0), ("sqrt(x)", -0.5)] {
            s.f.set(formula);
            s.x0 = x0;
            assert_eq!(reach_in(&s, -4.0, 4.0), Reach::NoSeries(true), "{formula}");
        }
        // a value but no derivatives: none either (order 0 is the value alone)
        (s.x0, s.order) = (0.0, 3);
        assert_eq!(reach_in(&s, -4.0, 4.0), Reach::NoSeries(false));
        s.order = 0;
        assert!(matches!(reach_in(&s, -4.0, 4.0), Reach::Within(_)));
        // where f has no value it does not agree: sqrt at 1 agrees at most down to 0
        (s.x0, s.order) = (1.0, 14);
        assert!(matches!(reach_in(&s, -4.0, 4.0), Reach::Within(r) if r <= 1.0));
        // 1/(1+x²) at 0.8 agrees to within about 1.28 of it (its radius of convergence):
        // in a narrow view that is the whole view, not "infinity"
        s.f.set("1/(1+x^2)");
        (s.x0, s.order) = (0.8, 14);
        assert_eq!(reach_in(&s, 0.3, 1.5), Reach::WholeView(0.7));
        assert!(matches!(reach_in(&s, -4.0, 4.0), Reach::Within(r) if r > 0.6 && r < 1.28), "{:?}", reach_in(&s, -4.0, 4.0));
        // x₀ out of view: nothing is claimed
        assert_eq!(reach_in(&s, 2.0, 4.0), Reach::OutOfView);
    }

    /// The numbers behind the captions and the integral's verdicts (run with
    /// --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        let mut s = Calc1::create();
        s.set_up(0);
        for h in [1.0, 0.1, 0.01, 0.001] {
            s.h = h;
            let m = (s.val(s.x0 + h) - s.val(s.x0)) / h;
            println!("secant at h {h}: {m}, exact {}, error {}", s.slope(s.x0), m - s.slope(s.x0));
        }
        s.set_up(1);
        for h in [1.0, 0.1, 0.01, 0.001] {
            println!("cusp: secant at h {h}: {}, slope there {}", (s.val(s.x0 + h) - s.val(s.x0)) / h, s.slope(s.x0));
        }
        s.set_up(2);
        for n in [4, 8, 10, 20, 40, 80, 200] {
            s.n = n;
            let exact = s.integral();
            for rule in 0..4 {
                s.rule = rule;
                println!("bell n {n} rule {rule}: sum {} against {exact:?}: error {:?}", s.riemann(), if let Area::Value(v) = exact { s.riemann() - v } else { f32::NAN });
            }
        }
        s.set_up(3);
        for order in [0, 2, 4, 6, 8, 10, 12, 14] {
            s.order = order;
            let xs: Vec<f32> = (0..=900).map(|i| -5.0 + 10.0 * i as f32 / 900.0).collect();
            let ys: Vec<f32> = xs.iter().map(|&x| s.val(x)).collect();
            let jet = s.series().unwrap();
            let poly: Vec<f32> = xs.iter().map(|&x| jet.eval(x - s.x0, order)).collect();
            println!("1/(1+x^2) order {order}: {:?}", reach(&xs, &ys, &poly, s.x0, None));
        }
        for (formula, a, b) in [
            ("1/x", -2.0, 2.5),
            ("tan(x)", -2.0, 2.5),
            ("1/x^2", -2.0, 2.5),
            ("sqrt(x)", -2.0, 2.5),
            ("sqrt(x)", 0.0, 2.5),
            ("ln(abs(x))", -2.0, 2.5),
            ("1/sqrt(abs(x))", -2.0, 2.5),
            ("1/x", -1.0, 1.0),
            ("1/x", -1.0, 2.0),
            ("1/(x-2)", 1.0, 3.0),
            ("1/(x^2+1e-8)", -2.0, 2.5),
            ("1 - abs(x)^0.1", -2.0, 2.5),
            ("sign(x)", -2.0, 2.5),
            ("floor(x)", -2.0, 2.5),
            ("x^x", -2.0, 2.5),
            ("x^x", 0.0, 2.5),
            ("sin(x) + 0.3x", -2.0, 2.5),
            ("exp(x/2)/2", -2.0, 2.5),
            ("3", -2.0, 2.5),
            ("sqrt(-1-x^2)", -2.0, 2.5),
            ("sin(1/x)", -2.0, 2.5),
            ("exp(-x^2/2)", -1.0, 2.5),
            ("1/(1+x^2)", -2.0, 2.5),
            ("1/x", 0.5, 3.0),
        ] {
            println!("{formula:>16} on [{a}, {b}]: {:?}", integral_of(formula, a, b));
        }
    }
}

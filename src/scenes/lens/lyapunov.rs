//! Lyapunov exponents: how fast a small nudge grows or dies on its way through the
//! layers of a network. This is the idea the paper leans on, with depth playing the
//! part that time plays for a weather system.

use super::deep::{newman, quick, Agreement, Exponents, Averaging, Job, Net, Quick, Spec, EDGE, INPUTS};
use super::page::{width, world, write_foot, Page, Panel, Plot, Rect};
use super::*;
use crate::math::linalg;

pub const DEF: SceneDef = SceneDef {
    id: "lyapunov",
    title: "Lyapunov exponents",
    group: super::GROUP,
    blurb: "Nudge a deep network's input: how fast the nudge grows or dies, layer by layer.",
    make: |_| Box::new(Lyapunov::new()),
};

/// How closely the circle of nudges as drawn follows the layer shown, in seconds (see
/// `follow`).
const GLIDE: f32 = 0.05;

const START: Spec = Spec { width: 24, depth: 30, gain: 3.0, residual: false, beta: 0.3, linear: false, seed: 4 };

/// Settings worth seeing: (button, what to look for, the settings). What each note
/// says is checked by the tests.
const EXPERIMENTS: [(&str, &str, Spec); 5] = [
    (
        "Chaos: gain 3",
        "The lines climb: a nudge is multiplied by about the same factor at every layer. The dots are a real nudge of size 0.1, not an infinitely small one, along one of the cyan directions. They keep to its line at first; then they may run ahead of it or fall behind, and once the nudge is about as big as the neurons' activity they level off. At the bottom right the thick line is nothing like the thin ones: the Jacobians at different inputs have nothing in common, and their average resembles none of them.",
        START,
    ),
    (
        "Order: gain 0.7",
        "Every line slopes down and every bar is below zero: whichever way you nudge the input, the nudge dies out. At the bottom right the thick line lies along the thin ones: the network responds in much the same way at every input, and averaging loses little.",
        Spec { gain: 0.7, ..START },
    ),
    (
        "The edge: gain 1",
        "The leading exponent is close to zero: the best-placed nudge neither blows up nor dies quickly. This is where deep networks are easiest to train.",
        Spec { gain: 1.0, ..START },
    ),
    (
        "A residual stream",
        "Each layer only adds a little to what it passes on, as the blocks of a transformer do. The exponents crowd round zero, some just above and some just below, and more than half of each Jacobian survives the averaging.",
        Spec { gain: 1.5, residual: true, ..START },
    ),
    (
        "No activation function",
        "The layers are now plain matrices, so the Jacobian is the same at every input: all the lines at the bottom right fall on one another and averaging changes nothing. The white dots on the bars are the exact answer for an endlessly deep network of this kind.",
        Spec { gain: 1.3, linear: true, ..START },
    ),
];

struct Lyapunov {
    spec: Spec,
    net: Net,
    quick: Quick,
    quick_for: Spec,
    job: Option<Job>,
    exponents: Option<Exponents>,
    averaging: Option<Averaging>,
    /// The settings the exponents and the averaging shown were worked out for.
    solved: Option<Spec>,
    /// The layer the circle of nudges is shown at, and whether it moves by itself.
    layer: f32,
    sweep: bool,
    /// The room the circle's caption takes under it, as drawn and as the caption now
    /// needs it (0 until it is first drawn): see [`Lyapunov::circle`].
    foot: (f32, f32),
    /// The circle of nudges as drawn, and on the way there to the shape of the layer
    /// shown (see [`Lyapunov::shape_at`] and `follow`): None until there is one to draw.
    shape: Option<([f32; 4], [f32; 4])>,
    page: Page,
}

impl Lyapunov {
    fn new() -> Self {
        let net = Net::new(START);
        let quick = quick(&net);
        Lyapunov { spec: START, net, quick, quick_for: START, job: None, exponents: None, averaging: None, solved: None, layer: 0.0, sweep: true, foot: (0.0, 0.0), shape: None, page: Page::new() }
    }

    /// Bring what is shown up to date with the settings, within this frame's allowance
    /// (`real_dt`: the seconds since the last frame).
    fn work(&mut self, real_dt: f32) {
        if self.quick_for != self.spec {
            self.net = self.net.with(self.spec);
            self.quick = quick(&self.net);
            self.quick_for = self.spec;
        }
        // a job that has started is allowed to finish, even if the settings have moved
        // on: otherwise nothing would ever be shown while a slider is being dragged. But
        // not once the network has another shape, since nothing it found could be shown
        // (see `current`): the job for the network wanted would only wait behind it.
        if self.job.as_ref().is_some_and(|job| (job.spec().width, job.spec().depth) != (self.spec.width, self.spec.depth)) {
            self.job = None;
        }
        if self.job.is_none() && self.solved != Some(self.spec) {
            self.job = Some(Job::new(&self.net, &self.quick.jac));
        }
        let Some(job) = &mut self.job else { return };
        let budget = crate::clock::frame_budget(real_dt, 0.004);
        // (the count keeps headless renders, where the clock is not consulted, the same every time)
        let mut allowance = 5.0e6;
        while !job.done() && allowance > 0.0 && !budget.spent() {
            allowance -= job.step();
        }
        if let Some(e) = job.exponents.take() {
            self.exponents = Some(e);
        }
        if job.done() {
            self.averaging = job.averaging.take();
            self.solved = Some(job.spec());
            self.job = None;
        }
    }

    /// The exponents to draw, if those in hand belong to a network of the present
    /// shape. While a slider is being dragged they are those of a moment ago (see
    /// [`Lyapunov::fresh`]): better drawn faintly than not at all.
    fn current(&self) -> Option<&Exponents> {
        self.exponents.as_ref().filter(|e| e.width == self.spec.width && e.depth == self.spec.depth)
    }

    /// Are the exponents in hand those of the network as it is set now?
    fn fresh(&self) -> bool {
        self.exponents.as_ref().is_some_and(|e| e.spec == self.spec)
    }

    /// The exponents, once they are those of the present settings (for numbers,
    /// which cannot be shown faintly).
    fn settled(&self) -> Option<&Exponents> {
        self.exponents.as_ref().filter(|e| e.spec == self.spec)
    }

    /// How strongly to draw what comes from the exponents: fully, or faintly while
    /// they are still those of the settings of a moment ago.
    fn ink(&self) -> f32 {
        if self.fresh() {
            1.0
        } else {
            0.35
        }
    }

    fn regime(&self) -> &'static str {
        match self.settled().map(|e| e.chi(0)) {
            None => "working it out…",
            Some(c) if c > EDGE => "Chaos: some nudges grow at every layer.",
            Some(c) if c < -EDGE => "Order: every nudge dies out.",
            Some(_) => "Close to the edge between order and chaos.",
        }
    }

    // ----- the four charts
    //
    // Each says the same thing in fewer words when it is given little room.

    fn growth(&self, f: &mut Frame, r: Rect) {
        let (depth, small) = (self.spec.depth, r.w() < 430.0);
        let mut head = Panel::open(f, r, 12.0);
        let title = if small {
            r"how much a small nudge to the input has grown, layer by layer ($\ln$ of its size)"
        } else {
            r"how much a small nudge to the input has grown, layer by layer ($\ln$ of its size: a straight line is growth by the same factor at every layer)"
        };
        head.text(f, title, 12.0, pal::TEXT, 3.0);
        head.key(f, &[(pal::YELLOW, "the fastest-growing direction $v_1$"), (pal::CYAN, "six random directions"), (pal::PINK, "the fastest-dying direction"), (pal::GREEN, "dots: a real nudge, of size 0.1")], 4.0);
        let (inner, y) = (head.inner, head.y);
        let area = Rect::new(inner.min.x + 34.0, y, inner.max.x - 4.0, inner.max.y - 32.0);

        let exps = self.current();
        let n = self.spec.width;
        let top = exps.map(|e| (0..=depth).map(|l| e.grown[l * n] as f32).collect::<Vec<_>>());
        let last = exps.map(|e| e.smallest.iter().map(|v| *v as f32).collect::<Vec<_>>());
        let (mut lo, mut hi) = (0.0f32, 0.0f32);
        let seen = self.quick.random.iter().flatten().chain(&self.quick.real).map(|v| *v as f32).chain(top.iter().flatten().copied());
        for v in seen.filter(|v| v.is_finite()) {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        // the fastest-dying direction is shown whole if that does not squash the rest
        if let Some(least) = last.as_ref().map(|l| l.iter().copied().fold(0.0, f32::min)) {
            if lo - least < 2.0 * (hi - lo).max(1.0) {
                lo = lo.min(least);
            }
        }
        let span = (hi - lo).max(1.0);
        let plot = Plot { area, x: [0.0, depth as f32], y: [lo - 0.12 * span, hi + 0.08 * span] };
        plot.rules(f);
        for l in (0..=depth).step_by((depth / 6).max(1)) {
            f.text(plot.at(l as f32, plot.y[0]) - Vec2::new(0.0, 4.0), l.to_string(), 10.5, pal::DIM, Align2::CENTER_TOP);
        }
        f.text(world(Vec2::new(area.center().x, area.max.y + 18.0)), "layers passed through", 10.5, pal::DIM, Align2::CENTER_TOP);
        // the layer the circle of nudges is shown at
        f.line(plot.at(self.layer, plot.y[0]), plot.at(self.layer, plot.y[1]), 1.0, pal::WHITE.a(0.3));

        let line = |values: &[f32]| values.iter().enumerate().map(|(l, &v)| (l as f32, v)).collect::<Vec<_>>();
        let ink = self.ink();
        if let Some(last) = &last {
            plot.curve(f, line(last), 2.2, pal::PINK.a(ink));
        }
        for trace in &self.quick.random {
            let values: Vec<f32> = trace.iter().map(|v| *v as f32).collect();
            plot.curve(f, line(&values), 1.4, pal::CYAN.a(0.75));
        }
        if let Some(top) = &top {
            plot.curve(f, line(top), 2.8, pal::YELLOW.a(ink));
        }
        for (l, v) in self.quick.real.iter().enumerate() {
            let v = *v as f32;
            if v.is_finite() && v >= plot.y[0] && v <= plot.y[1] {
                f.dot(plot.at(l as f32, v), if small { 5.0 } else { 6.5 }, pal::GREEN);
            }
        }
        if exps.is_some() && !self.fresh() {
            // the yellow and pink lines are still those of the settings of a moment ago
            f.text(world(Vec2::new(area.min.x + 6.0, area.min.y + 2.0)), "yellow and pink: catching up with the new settings…", 11.0, pal::DIM, Align2::LEFT_TOP);
        } else if let Some(e) = exps {
            let slope = signed(e.chi(0) as f32, 3);
            let note = if small { format!(r"yellow's slope: $\chi_1 = {slope}$") } else { format!(r"slope of the yellow line: $\chi_1 = {slope}$ per layer") };
            // (where the lines are not: they start at the left and climb, or fall)
            let (at, align) = if e.chi(0) >= 0.0 { (Vec2::new(area.min.x + 6.0, area.min.y + 2.0), Align2::LEFT_TOP) } else { (Vec2::new(area.max.x - 4.0, area.min.y + 2.0), Align2::RIGHT_TOP) };
            f.text(world(at), note, 11.5, pal::YELLOW, align);
            if e.smallest.last().is_some_and(|v| (*v as f32) < plot.y[0]) {
                let rate = signed(e.chi(n - 1) as f32, 2);
                let note = if small { format!(r"pink: $\chi_{{{n}}} = {rate}$") } else { format!(r"pink runs off the chart: $\chi_{{{n}}} = {rate}$ per layer") };
                let (at, align) = if e.chi(0) >= 0.0 { (Vec2::new(area.max.x - 4.0, area.max.y - 3.0), Align2::RIGHT_BOTTOM) } else { (Vec2::new(area.min.x + 30.0, area.max.y - 3.0), Align2::LEFT_BOTTOM) };
                f.text(world(at), note, 11.0, pal::PINK, align);
            }
        }
    }

    fn spectrum(&self, f: &mut Frame, r: Rect) {
        let (n, small) = (self.spec.width, r.w() < 430.0);
        let mut head = Panel::open(f, r, 12.0);
        let title = if small {
            format!(r"all {n} Lyapunov exponents $\chi_i$: the growth rate per layer of each direction")
        } else {
            format!(r"all {n} Lyapunov exponents $\chi_i$: the growth rate per layer of each direction a nudge can take, over the whole depth")
        };
        head.text(f, &title, 12.0, pal::TEXT, 3.0);
        let theory = self.spec.linear && !self.spec.residual;
        let mut entries = vec![(pal::ORANGE, "above zero: grows"), (pal::BLUE, "below zero: dies")];
        if theory {
            entries.push((pal::WHITE, "dots: exact, for endless depth"));
        }
        head.key(f, &entries, 4.0);
        let (inner, y) = (head.inner, head.y);
        let area = Rect::new(inner.min.x + 34.0, y, inner.max.x - 4.0, inner.max.y - 20.0);
        let Some(e) = self.current() else {
            f.text(world(area.center()), "working it out…", 12.0, pal::DIM, Align2::CENTER_CENTER);
            return;
        };
        let chi: Vec<f32> = (0..n).map(|i| e.chi(i) as f32).collect();
        let (lo, hi) = (chi[n - 1].min(0.0), chi[0].max(0.0));
        let span = (hi - lo).max(0.2);
        let plot = Plot { area, x: [0.5, n as f32 + 0.5], y: [lo - 0.1 * span, hi + 0.14 * span] };
        plot.rules(f);
        let fresh = self.fresh();
        for (i, &c) in chi.iter().enumerate() {
            let (a, b) = (plot.at(i as f32 + 0.6, 0.0), plot.at(i as f32 + 1.4, c));
            let color = if c >= 0.0 { pal::ORANGE } else { pal::BLUE };
            f.dl.tris_alpha.rect(Vec2::new(a.x, a.y.min(b.y)), Vec2::new(b.x, a.y.max(b.y)), 0.0, color.a(0.85 * self.ink()));
        }
        if !fresh {
            // (the bars are those of the settings of a moment ago: no numbers, and
            // none of the exact dots, which are for the new settings)
            f.text(world(Vec2::new(area.center().x, area.min.y + 2.0)), "catching up with the new settings…", 11.0, pal::DIM, Align2::CENTER_TOP);
            return;
        }
        if theory {
            for i in 0..n {
                f.dot(plot.at(i as f32 + 1.0, newman(n, self.spec.gain as f64, i) as f32), 4.5, pal::WHITE);
            }
        }
        f.text(plot.at(1.0, chi[0].max(0.0)) + Vec2::new(0.0, 3.0), format!(r"$\chi_1 = {}$", signed(chi[0], 3)), 11.0, pal::YELLOW, Align2::LEFT_BOTTOM);
        let (last, last_at) = (format!(r"$\chi_{{{}}} = {}$", n, signed(chi[n - 1], 2)), plot.at(n as f32, chi[n - 1].min(0.0)) - Vec2::new(0.0, 3.0));
        // the axis's name, in fewer words if the last bar's label would run into it
        let mut name = "direction $i$, fastest-growing first";
        // (the label's top is at `last_at`, and it stands about 16 pixels tall)
        if -last_at.y + 16.0 > area.max.y + 4.0 && last_at.x - width(&last, 11.0) < area.center().x + 0.5 * width(name, 10.5) + 6.0 {
            name = "direction $i$";
        }
        f.text(last_at, last, 11.0, pal::PINK, Align2::RIGHT_TOP);
        f.text(world(Vec2::new(area.min.x, area.max.y + 4.0)), "1", 10.5, pal::DIM, Align2::LEFT_TOP);
        f.text(world(Vec2::new(area.center().x, area.max.y + 4.0)), name, 10.5, pal::DIM, Align2::CENTER_TOP);
    }

    /// The circle of nudges after `layer` layers as it is drawn: the matrix that makes
    /// it, scaled to largest stretch 1, `[a, b, d]` for `[[a, b], [0, d]]`, and the
    /// natural logarithm of its largest half-axis (what sets the dashed circle's size
    /// beside it).
    fn shape_at(e: &Exponents, layer: usize) -> [f32; 4] {
        let [_, a, b, d] = e.ellipse[layer.min(e.depth)];
        let own = linalg::svd2(Mat2::from_cols(Vec2::new(a as f32, 0.0), Vec2::new(b as f32, d as f32))).s.x.max(1e-20);
        [a as f32 / own, b as f32 / own, d as f32 / own, e.half_axes(layer).0 as f32]
    }

    fn circle(&mut self, f: &mut Frame, r: Rect) {
        let depth = self.spec.depth;
        let layer = (self.layer.round() as usize).min(depth);
        let mut head = Panel::open(f, r, 12.0);
        head.text(f, &format!("a small circle of nudges, after {layer} of the {depth} layers"), 12.0, pal::TEXT, 3.0);
        head.key(f, &[(pal::DIM, "dashed: the circle at the input"), (pal::YELLOW, "the nudge along $v_1$"), (pal::ORANGE, "the nudge along $v_2$")], 2.0);
        let (inner, y) = (head.inner, head.y);
        let Some(e) = self.current() else {
            self.shape = None;
            f.text(world(Vec2::new(inner.center().x, (y + inner.max.y) * 0.5)), "working it out…", 12.0, pal::DIM, Align2::CENTER_CENTER);
            return;
        };
        let (big, small) = e.half_axes(layer);
        let target = Self::shape_at(e, layer);
        let times = |ln: f64| if ln.abs() < 80.0 { fmt(ln.exp() as f32) } else { format!("e^{{{ln:.0}}}") };
        let mut note = format!(r"half-axes: $\times {}$ and $\times {}$", times(big), times(small));
        if layer == depth {
            note.push_str(", the singular values $s_1$ and $s_2$ of the whole Jacobian");
        }
        if !self.fresh() {
            note = "catching up with the new settings…".to_string();
        }
        // The circle is given the room the caption leaves it, and eases to its new size
        // when the caption gains or loses a line (as the numbers in it change from layer
        // to layer, a formula like 5.98 × 10⁻⁴ among them) rather than jumping.
        let foot = write_foot(f, inner, 0.0, &note, 11.0, pal::TEXT) + 4.0;
        let drawn = if self.foot.0 <= 0.0 { foot } else { self.foot.0 + (foot - self.foot.0) * (1.0 - (-f.real_dt / 0.08).exp()) };
        self.foot = (if (drawn - foot).abs() < 0.2 { foot } else { drawn }, foot);
        let area = Rect::new(inner.min.x, y, inner.max.x, inner.max.y - self.foot.0);

        // The layer is a whole number: swept through the layers, the circle took the
        // next layer's shape at once, a jump every few frames (every thirteen at 30
        // layers and 60 frames a second), and stood still in between. So the circle
        // drawn follows the shape of the layer shown (`follow`); the layer named and the
        // numbers above are those of the layer shown.
        let shape = follow(self.shape, target, f.real_dt, 1e-4);
        self.shape = Some(shape);
        let [a, b, d, big] = shape.0;

        let (c, radius) = (world(area.center()), 0.5 * area.w().min(area.h()) - 6.0);
        // drawn so that the longer half-axis always just fits; `unit` is then what a
        // nudge of the original size comes to on the page
        let m = Mat2::from_cols(Vec2::new(a, 0.0), Vec2::new(b, d));
        let shape = |t: f32| c + (m * Vec2::from_angle(t)) * radius;
        let unit = radius * (-big).exp();
        if unit > radius / 80.0 && unit < radius * 1.02 {
            let turn = TAU / 64.0;
            for k in (0..64).step_by(2) {
                f.line(c + Vec2::from_angle(k as f32 * turn) * unit, c + Vec2::from_angle((k + 1) as f32 * turn) * unit, 1.2, pal::DIM);
            }
        } else if unit <= radius / 80.0 {
            f.dot(c, 3.0, pal::DIM);
        } else {
            f.text(world(Vec2::new(area.center().x, area.min.y + 2.0)), "(the circle it started as is off the page)", 10.5, pal::DIM, Align2::CENTER_TOP);
        }
        let ink = self.ink();
        let outline: Vec<Vec3> = (0..96).map(|k| shape(k as f32 / 96.0 * TAU).extend(0.0)).collect();
        f.dl.lines.closed(&outline, 2.2, pal::TEXT.a(ink));
        f.arrow(c, shape(0.0), 2.4, pal::YELLOW.a(ink));
        f.arrow(c, shape(PI / 2.0), 2.4, pal::ORANGE.a(ink));
    }

    fn averaging(&self, f: &mut Frame, r: Rect) {
        let small = r.w() < 330.0;
        let mut head = Panel::open(f, r, 12.0);
        let title = if small { format!("at {INPUTS} other inputs: do the network's Jacobians agree?") } else { format!("the same network at {INPUTS} other inputs: do its Jacobians agree with one another?") };
        head.text(f, &title, 12.0, pal::TEXT, 3.0);
        let entries = if small {
            [(pal::CYAN, "each one's singular values"), (pal::PINK, "their average's")]
        } else {
            [(pal::CYAN, "singular values of each Jacobian"), (pal::PINK, "of the average of the Jacobians")]
        };
        head.key(f, &entries, 4.0);
        let (inner, y) = (head.inner, head.y);
        let shown = self.averaging.as_ref().filter(|_| self.solved.is_some_and(|s| s.width == self.spec.width && s.depth == self.spec.depth));
        let Some(avg) = shown else {
            f.text(world(Vec2::new(inner.center().x, (y + inner.max.y) * 0.5)), "working it out…", 12.0, pal::DIM, Align2::CENTER_CENTER);
            return;
        };
        let verdict = match (avg.verdict(), small) {
            (Agreement::Whole, _) => "they agree, and averaging loses little",
            (Agreement::Part, _) => "they agree in part",
            (Agreement::None, true) => "they have next to nothing in common",
            (Agreement::None, false) => "they have next to nothing in common: the average resembles none of them",
        };
        let (left, unrelated, overlap) = (avg.kept() * 100.0, avg.unrelated * 100.0, signed(avg.overlap as f32, 2));
        // (these Jacobians may still be those of the settings of a moment ago)
        let fresh = self.solved == Some(self.spec);
        let ink = if fresh { 1.0 } else { 0.35 };
        let note = if !fresh {
            "catching up with the new settings…".to_string()
        } else if small {
            format!("Left after averaging: {left:.0} % (unrelated ones would leave {unrelated:.0} %). Overlap {overlap}: {verdict}.")
        } else {
            format!("Left of the leading singular value after averaging: {left:.0} % (the same Jacobians turned every which way, so that they share nothing, would leave {unrelated:.0} %). Their overlap is {overlap} (1: the same, 0: unrelated). So {verdict}.")
        };
        let foot = write_foot(f, inner, 2.0, &note, 11.0, pal::TEXT) + 6.0;
        let area = Rect::new(inner.min.x + 34.0, y, inner.max.x - 4.0, inner.max.y - foot - 16.0);

        let kept = avg.mean.len();
        let lowest = avg.each.iter().flatten().chain(&avg.mean).fold(f64::MAX, |m, v| m.min(*v)) as f32;
        // (singular values this far below the largest are rounding, not information)
        let lo = lowest.max(-28.0).min(-1.0);
        let plot = Plot { area, x: [0.6, kept as f32 + 0.4], y: [lo * 1.06, -lo * 0.1] };
        plot.rules(f);
        for s in &avg.each {
            plot.curve(f, s.iter().enumerate().map(|(i, v)| (i as f32 + 1.0, *v as f32)), 1.2, pal::CYAN.a(0.6 * ink));
        }
        plot.curve(f, avg.mean.iter().enumerate().map(|(i, v)| (i as f32 + 1.0, *v as f32)), 2.8, pal::PINK.a(ink));
        for (i, v) in avg.mean.iter().enumerate() {
            if (*v as f32) >= plot.y[0] {
                f.dot(plot.at(i as f32 + 1.0, *v as f32), 5.5, pal::PINK.a(ink));
            }
        }
        f.text(world(Vec2::new(area.min.x, area.max.y + 4.0)), "largest", 10.5, pal::DIM, Align2::LEFT_TOP);
        f.text(world(Vec2::new(area.max.x, area.max.y + 4.0)), format!("{kept}th largest"), 10.5, pal::DIM, Align2::RIGHT_TOP);
        let caption = if small { r"$\ln$ (0: a Jacobian's largest)" } else { r"$\ln$ of the singular value (0: a Jacobian's largest)" };
        f.text(plot.at(plot.x[1], 0.0) + Vec2::new(-4.0, 3.0), caption, 10.5, pal::DIM, Align2::RIGHT_BOTTOM);
    }
}

/// Numbers as drawn, and on the way there, brought `dt` seconds nearer to `to` through
/// two eases of `GLIDE` one after the other, and exactly there once none is more than
/// `close` away (the picture then stops changing). Kept nowhere yet: there at once.
fn follow<const N: usize>(kept: Option<([f32; N], [f32; N])>, to: [f32; N], dt: f32, close: f32) -> ([f32; N], [f32; N]) {
    let Some((mut shown, mut between)) = kept.filter(|(s, b)| s.iter().chain(b).all(|v| v.is_finite())) else { return (to, to) };
    let k = 1.0 - (-dt.max(0.0) / GLIDE).exp();
    for i in 0..N {
        between[i] += (to[i] - between[i]) * k;
        shown[i] += (between[i] - shown[i]) * k;
    }
    let off = (0..N).map(|i| (to[i] - shown[i]).abs().max((to[i] - between[i]).abs())).fold(0.0, f32::max);
    if off < close {
        (to, to)
    } else {
        (shown, between)
    }
}

impl Scene for Lyapunov {
    fn view(&self) -> View {
        self.page.view()
    }

    fn take_view(&mut self) -> Option<View> {
        self.page.take_view()
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn animated(&self) -> bool {
        self.sweep || self.job.is_some() || self.solved != Some(self.spec) || self.foot.0 != self.foot.1 || self.shape.is_some_and(|(shown, between)| shown != between)
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "Nudge the input of a deep network a little and follow the nudge through the layers. Each layer stretches it in some directions and squashes it in others. The rates at which it grows or dies, per layer, are the Lyapunov exponents. Depth plays the part here that time plays for the weather: a positive exponent is the butterfly effect.");
        ui::section(ui, "Things to try");
        for (label, look_for, spec) in EXPERIMENTS {
            if ui::experiment(ui, label, look_for) {
                self.spec = Spec { seed: self.spec.seed, ..spec };
            }
        }
        ui::section(ui, "The network");
        let mut kind = self.spec.residual as usize;
        ui::choice(ui, &["each layer replaces", "each layer adds"], &mut kind);
        self.spec.residual = kind == 1;
        if self.spec.residual {
            ui::equation(ui, r"h' = h + \beta \varphi(W h)");
            ui::note(ui, "The state is passed on with a little added, as in a transformer's residual stream.");
        } else {
            ui::equation(ui, r"h' = \varphi(W h)");
            ui::note(ui, "Each layer makes the state anew from the last.");
        }
        ui::slider(ui, "gain $g$ (size of the weights)", &mut self.spec.gain, 0.3..=4.0);
        ui::note(ui, r"A weight is a random number of typical size $g / \sqrt{\text{width}}$.");
        if self.spec.residual {
            ui::slider(ui, r"how much each layer adds, $\beta$", &mut self.spec.beta, 0.02..=1.5);
        }
        let mut act = self.spec.linear as usize;
        ui::choice(ui, &[r"$\varphi = \tanh$", "no activation"], &mut act);
        self.spec.linear = act == 1;
        ui::slider_int(ui, "layers", &mut self.spec.depth, 4..=60);
        ui::slider_int(ui, "width (neurons per layer)", &mut self.spec.width, 4..=40);
        if ui.button("another random network").clicked() {
            self.spec.seed += 1;
        }
        ui::section(ui, "The circle of nudges");
        ui.checkbox(&mut self.sweep, "sweep through the layers");
        // (the layers may just have been cut back: a value beyond the slider's end would
        // be pulled in by the slider itself, which would count as moving it)
        let mut layer = (self.layer.round() as usize).min(self.spec.depth);
        if ui::slider_int(ui, "layer shown", &mut layer, 0..=self.spec.depth) {
            self.layer = layer as f32;
            self.sweep = false;
        }
        ui::section(ui, "What the words mean");
        ui::term(ui, "Jacobian $J$", r"the table of how far each number of the output moves when each number of the input is nudged. A small nudge $\delta$ to the input moves the output by $J \delta$.");
        ui::term(ui, r"Singular values $s_1 \ge s_2 \ge \cdots$", "the Jacobian's stretch factors: a small ball of nudges comes out as an ellipsoid with these half-axes. $v_1$ is the nudge that is stretched the most.");
        ui::term(ui, r"Lyapunov exponent $\chi_i$ (chi)", r"the same as a rate per layer: $\ln s_i$ divided by the number of layers (the paper's Eq. 4). Above zero the nudge grows, below zero it dies.");
        ui::section(ui, "Colours");
        ui::key(ui, &[
            (pal::YELLOW, "the nudge that grows fastest (along $v_1$)"),
            (pal::CYAN, "nudges in random directions; and, at the bottom right, the Jacobian at one input"),
            (pal::PINK, "the nudge that dies fastest; and, at the bottom right, the average of the Jacobians"),
            (pal::GREEN, "a real nudge, of size 0.1"),
            (pal::ORANGE, "an exponent above zero; and the nudge along $v_2$"),
            (pal::BLUE, "an exponent below zero"),
        ]);
        ui::section(ui, "Numbers");
        let depth = self.spec.depth;
        if let Some(e) = self.settled() {
            let n = e.width;
            ui::readout(ui, r"leading exponent $\chi_1$, per layer", format!("${}$", signed(e.chi(0) as f32, 3)));
            let total = e.chi(0) * depth as f64;
            ui::readout(ui, format!("the best nudge, over {depth} layers").as_str(), if total.abs() < 80.0 { format!(r"$\times {}$", fmt(total.exp() as f32)) } else { format!(r"$\times e^{{{total:.0}}}$") });
            ui::readout(ui, "directions that grow", format!("{} of {n}", (0..n).filter(|&i| e.chi(i) > 0.0).count()));
            if !self.spec.residual {
                let estimate = 0.5 * ((self.spec.gain as f64).powi(2) * self.quick.slope_square).ln();
                ui::readout(ui, r"rule of thumb, $\tfrac{1}{2} \ln(g^2 \cdot \operatorname{mean} \varphi'^2)$", format!("${}$", signed(estimate as f32, 3)));
            }
        }
        if let Some(a) = self.averaging.as_ref().filter(|_| self.solved == Some(self.spec)) {
            ui::readout(ui, "left after averaging the Jacobians", format!("{:.0} %", a.kept() * 100.0));
            ui::readout(ui, "left if they shared nothing", format!("{:.0} %", a.unrelated * 100.0));
            ui::readout(ui, "overlap of two Jacobians (0 to 1)", format!("${}$", signed(a.overlap as f32, 2)));
            let spread = a.sizes.iter().fold(f64::MIN, |m, v| m.max(*v)) - a.sizes.iter().fold(f64::MAX, |m, v| m.min(*v));
            ui::readout(ui, "the biggest Jacobian ÷ the smallest", format!("${}$", fmt(spread.exp() as f32)));
            ui::note(ui, "The Jacobians are scaled to the same size before they are averaged. Unscaled, the average would be little more than the biggest of them. \"If they shared nothing\" is measured, not assumed: the same Jacobians are each turned at random, on the input side and on the output side, and averaged again.");
        }
        ui.label(egui::RichText::new(self.regime()).size(12.5).color(pal::YELLOW.egui()));
        ui::note(ui, "The Signal propagation scene works the same thing out for an endlessly wide network: its slope at $c = 1$ is the factor by which the square of a nudge's size grows per layer, so the leading exponent is half its logarithm (the rule of thumb above). Order, chaos and the edge between them are the same three there.");
        ui::note(ui, "The bottom right chart is where the next scene starts, and the paper too (its Fig. 1 and appendix A): the average of many Jacobians need not look like any one of them. A lens that averages Jacobians over many inputs keeps only what they have in common, and how much they have in common is what the paper calls context coherence.");
        ui::section(ui, "In the paper");
        ui::note(ui, "Ostrow, Qian, Liu and Fiete, \"A Mathematical Theory of the Jacobian Lens\" (2026) treat a transformer as a system that evolves through its layers, and read its Jacobians with these exponents (their Eq. 3 and 4). The numbers here are found as they describe: the layers' Jacobians are chained, with the directions made orthonormal again after every layer.");
        ui::see_also(ui, &[("signal", "Signal propagation"), ("coherence", "What averaging Jacobians keeps"), ("belief", "The Jacobian lens on a belief filter"), ("powers", "Powers of a matrix: one map applied again and again")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        self.page.fit(f);
        self.work(f.real_dt);
        let depth = self.spec.depth as f32;
        if self.sweep {
            // forward through the layers, then a rest at the far end
            let phase = (f.t / 9.0).fract();
            self.layer = (phase / 0.72).min(1.0) * depth;
        }
        self.layer = self.layer.clamp(0.0, depth);

        let full = self.page.usable();
        let gap = 5.0;
        if self.page.size.x >= 560.0 {
            let split = full.min.x + full.w() * 0.58;
            let (left, right) = (full.min.y + full.h() * 0.55, full.min.y + full.h() * 0.5);
            self.growth(f, Rect::new(full.min.x, full.min.y, split - gap, left - gap));
            self.spectrum(f, Rect::new(full.min.x, left + gap, split - gap, full.max.y));
            self.circle(f, Rect::new(split + gap, full.min.y, full.max.x, right - gap));
            self.averaging(f, Rect::new(split + gap, right + gap, full.max.x, full.max.y));
        } else {
            let (a, b) = (full.min.y + full.h() * 0.31, full.min.y + full.h() * 0.54);
            let mid = full.center().x;
            self.growth(f, Rect::new(full.min.x, full.min.y, full.max.x, a - gap));
            self.spectrum(f, Rect::new(full.min.x, a + gap, full.max.x, b - gap));
            self.circle(f, Rect::new(full.min.x, b + gap, mid - gap, full.max.y));
            self.averaging(f, Rect::new(mid + gap, b + gap, full.max.x, full.max.y));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::deep::{solve, test_spec as spec, REAL};
    use super::*;

    /// What the notes of the "things to try" say, held to account.
    #[test]
    fn the_experiments_show_what_their_notes_say() {
        for seed in [4, 5, 6] {
            let run = |k: usize| solve(&Net::new(Spec { seed, ..EXPERIMENTS[k].2 }));
            let fifth = |a: &Averaging| a.each.iter().map(|s| s[4]).sum::<f64>() / INPUTS as f64;
            // chaos: growth, Jacobians with nothing in common, and an average whose
            // spectrum (nearly flat) is unlike any of theirs (for the dots, see below)
            let (_, e, a) = run(0);
            assert!(e.chi(0) > 0.15, "chaos: {}", e.chi(0));
            assert!(a.kept() < 0.3 && a.overlap.abs() < 0.1 && a.verdict() == Agreement::None, "chaos: kept {} overlap {}", a.kept(), a.overlap);
            // "nothing in common" is no figure of speech: scrambled Jacobians keep about as much
            assert!(a.kept() < 1.5 * a.unrelated && a.kept() > 0.7 * a.unrelated, "chaos: kept {} against {} unrelated", a.kept(), a.unrelated);
            // (which is well above one in INPUTS: random directions are not at right angles)
            assert!(a.unrelated > 1.3 / INPUTS as f64, "unrelated {}", a.unrelated);
            assert!(fifth(&a) < -5.0 && a.mean[4] - a.mean[0] > -2.0, "chaos: fifth value {} against {}", fifth(&a), a.mean[4] - a.mean[0]);
            // order: everything dies, and the Jacobians agree
            let (_, e, a) = run(1);
            assert!(e.chi(0) < 0.0);
            assert!(a.kept() > 0.85 && a.verdict() == Agreement::Whole && a.kept() > 3.0 * a.unrelated, "order: kept {} overlap {}", a.kept(), a.overlap);
            // the edge
            let (_, e, _) = run(2);
            assert!(e.chi(0).abs() < EDGE, "edge: {}", e.chi(0));
            // a residual stream: exponents round zero, on both sides, and agreement
            let (_, e, a) = run(3);
            assert!(e.chi(0) > 0.0 && e.chi(23) < 0.0 && e.chi(0) < 0.2 && e.chi(23) > -0.2, "residual: {} .. {}", e.chi(23), e.chi(0));
            assert!(a.kept() > 0.5 && a.verdict() == Agreement::Part, "residual: kept {} overlap {}", a.kept(), a.overlap);
            // no activation: one Jacobian for every input
            let (_, e, a) = run(4);
            assert!((a.kept() - 1.0).abs() < 1e-6 && (a.overlap - 1.0).abs() < 1e-6 && a.verdict() == Agreement::Whole);
            assert!((e.chi(0) - newman(24, 1.3f32 as f64, 0)).abs() < 0.1);
        }
    }

    /// The dots of "Chaos: gain 3", for one random network after another: on their
    /// line at first, then ahead of it or behind, and level once the nudge is as big as
    /// the activity. (Not "until the nudge is a few per cent of the activity": in some
    /// of these networks they part at 2 %, in others they keep together to 60 %.)
    #[test]
    fn the_real_nudge_keeps_to_its_line_at_first_and_levels_off_at_the_activity() {
        let (mut reached, mut climbed, mut ahead, mut behind, mut parted) = (0, 0, 0, 0, Vec::new());
        for seed in 4..44 {
            let net = Net::new(Spec { seed, ..EXPERIMENTS[0].2 });
            let (q, (n, depth)) = (quick(&net), (net.spec.width, net.spec.depth));
            let states = net.run(&net.input(0)).0;
            let activity: Vec<f64> = (0..=depth).map(|l| linalg::norm(&states[l * n..(l + 1) * n])).collect();
            // the real nudge as a share of the activity, and how far the dots are from their line
            let share: Vec<f64> = (0..=depth).map(|l| REAL * q.real[l].exp() / activity[l]).collect();
            let off: Vec<f64> = (0..=depth).map(|l| q.real[l] - q.random[0][l]).collect();
            // on the line at first: through the first layer always, and typically for five
            assert!(off[1].abs() < 0.1, "seed {seed}: {} off after one layer", off[1]);
            parted.push(off.iter().position(|o| o.abs() > 0.1).unwrap_or(depth + 1));
            let most = off.iter().fold(0.0f64, |m, o| m.max(o.abs()));
            assert!(most > 0.5, "seed {seed}: never more than {most} from its line");
            ahead += off.iter().any(|o| *o > 0.5) as usize;
            behind += off.iter().any(|o| *o < -0.5) as usize;
            // never much bigger than the activity, and no smaller than half of it once there
            let top = share.iter().fold(0.0f64, |m, s| m.max(*s));
            assert!(top < 2.0, "seed {seed}: the nudge grew to {top} times the activity");
            if let Some(first) = share.iter().position(|s| *s >= 1.0) {
                reached += 1;
                let low = share[first..].iter().fold(f64::MAX, |m, s| m.min(*s));
                assert!(low > 0.45, "seed {seed}: down to {low} of the activity after reaching it");
            }
            // while the line, an infinitely small nudge, has climbed far beyond it
            climbed += (REAL * q.random[0][depth].exp() / activity[depth] > 3.0) as usize;
        }
        parted.sort();
        assert!(parted[parted.len() / 2] >= 4, "they part from their lines after {parted:?} layers");
        assert!(reached >= 36 && climbed >= 30, "{reached} reached the activity, {climbed} lines climbed past it");
        assert!(ahead >= 12 && behind >= 34, "{ahead} ran ahead, {behind} fell behind");
        // what is seen first: on the line for six layers, then more than one unit ahead of
        // it, and in the end more than one behind
        let q = quick(&Net::new(START));
        let off: Vec<f64> = q.real.iter().zip(&q.random[0]).map(|(r, l)| r - l).collect();
        assert!(off[..7].iter().all(|o| o.abs() < 0.1) && off.iter().any(|o| *o > 1.0) && off[30] < -1.0, "{off:?}");
    }

    /// While a slider is dragged the exponents in hand are those of a moment ago:
    /// they may be drawn (faintly), but no number or verdict is given for them.
    #[test]
    fn results_for_other_settings_are_not_passed_off_as_current() {
        let mut scene = Lyapunov::new();
        let catch_up = |scene: &mut Lyapunov| {
            for _ in 0..100_000 {
                if scene.solved == Some(scene.spec) {
                    return;
                }
                scene.work(1.0 / 60.0);
            }
            panic!("never caught up");
        };
        catch_up(&mut scene);
        assert!(scene.fresh() && scene.settled().is_some() && scene.ink() == 1.0);
        let chaos = scene.regime();
        assert!(chaos.starts_with("Chaos"));
        // the gain is moved: the old exponents fit the new network's shape, but are not its own
        scene.spec.gain = 0.7;
        assert!(scene.current().is_some() && !scene.fresh() && scene.settled().is_none() && scene.ink() < 1.0);
        assert_eq!(scene.regime(), "working it out…");
        catch_up(&mut scene);
        assert!(scene.fresh() && scene.regime().starts_with("Order"));
        // a change of shape leaves nothing to draw until the new exponents are in
        scene.spec.width = 12;
        assert!(scene.current().is_none());
        // and a job half done for the old shape is dropped, not finished first
        let mut scene = Lyapunov::new();
        scene.work(1.0 / 60.0);
        assert!(scene.job.is_some());
        scene.spec.depth = 40;
        scene.work(1.0 / 60.0);
        assert_eq!(scene.job.as_ref().map(|job| job.spec()), Some(scene.spec));
    }

    /// "The edge: gain 1" stays the edge when "another random network" is pressed:
    /// the regime line under the numbers and the experiment's name must not disagree.
    #[test]
    fn the_edge_is_the_edge_for_every_random_network() {
        for seed in 4..28 {
            let chi = solve(&Net::new(Spec { seed, ..EXPERIMENTS[2].2 })).1.chi(0);
            assert!(chi.abs() < EDGE, "seed {seed}: χ₁ = {chi}");
        }
        // while chaos and order are well clear of it
        for seed in 4..10 {
            assert!(solve(&Net::new(Spec { seed, ..EXPERIMENTS[0].2 })).1.chi(0) > 1.5 * EDGE);
            assert!(solve(&Net::new(Spec { seed, ..EXPERIMENTS[1].2 })).1.chi(0) < -1.5 * EDGE);
        }
    }

    /// The circle of nudges glides from layer to layer: swept through the 30 layers at
    /// 60 and at 144 frames a second, the circle drawn changes in no frame by as much
    /// as half of what the layer's shape changes from one layer to the next (it took
    /// the whole of that at once), and left at a layer it is exactly that layer's.
    #[test]
    fn the_circle_glides_from_layer_to_layer() {
        let (_, e, _) = solve(&Net::new(START));
        let apart = |p: [f32; 4], q: [f32; 4]| (0..4).map(|i| (p[i] - q[i]).abs()).fold(0.0, f32::max);
        for hz in [60.0f32, 144.0] {
            let (dt, mut kept, mut was) = (1.0 / hz, None, None);
            let (mut layers, mut frames) = (0.0f32, 0.0f32);
            for frame in 0..(6.48 * hz) as usize {
                // (the sweep: through the 30 layers in 6.48 seconds)
                let to = Lyapunov::shape_at(&e, (frame as f32 * dt / 6.48 * 30.0).round() as usize);
                let next = follow(kept, to, dt, 1e-4);
                if let (Some((shown, _)), Some(before)) = (kept, was) {
                    frames = frames.max(apart(shown, next.0));
                    layers = layers.max(apart(before, to));
                }
                (kept, was) = (Some(next), Some(to));
            }
            assert!(layers > 0.05 && frames < 0.5 * layers, "{hz} Hz: {frames} in a frame, {layers} from one layer to the next");
            let last = Lyapunov::shape_at(&e, 30);
            for _ in 0..(hz as usize) {
                kept = Some(follow(kept, last, dt, 1e-4));
            }
            assert_eq!(kept, Some((last, last)));
        }
    }

    /// The numbers behind the notes (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for (k, (label, _, s)) in EXPERIMENTS.iter().enumerate() {
            for seed in [4, 5, 6] {
                let (q, e, a) = solve(&Net::new(Spec { seed, ..*s }));
                let spread = a.sizes.iter().fold(f64::MIN, |m, v| m.max(*v)) - a.sizes.iter().fold(f64::MAX, |m, v| m.min(*v));
                let fifth = a.each.iter().map(|s| s[4]).sum::<f64>() / INPUTS as f64;
                println!("{k} {label} seed {seed}: chi1 {:+.3} chi_n {:+.3} growing {} kept {:.3} overlap {:.3} sizes spread {spread:.1} 5th: each {fifth:.2} mean {:.2}  real-vs-linear at 3: {:.3} at 30: {:.2}", e.chi(0), e.chi(23), (0..24).filter(|&i| e.chi(i) > 0.0).count(), a.kept(), a.overlap, a.mean[4] - a.mean[0], q.random[0][3] - q.real[3], q.random[0][30] - q.real[30]);
            }
        }
        for gain in [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] {
            let (q, e, a) = solve(&Net::new(spec(24, 30, gain)));
            let estimate = 0.5 * ((gain as f64).powi(2) * q.slope_square).ln();
            println!("gain {gain}: chi1 {:+.3} (rule {estimate:+.3}) chi_n {:+.3} kept {:.3} overlap {:.3} real-vs-linear at 3: {:.3} at 30: {:.2}  random rates {:?}", e.chi(0), e.chi(23), a.kept(), a.overlap, q.random[0][3] - q.real[3], q.random[0][30] - q.real[30], q.random.iter().map(|t| (t[30] / 30.0 * 1000.0).round() / 1000.0).collect::<Vec<_>>());
        }
        // what unrelated Jacobians would leave, and the verdicts
        for (k, (label, _, s)) in EXPERIMENTS.iter().enumerate() {
            for seed in [4, 5, 6, 7] {
                let a = solve(&Net::new(Spec { seed, ..*s })).2;
                println!("{k} {label} seed {seed}: kept {:.3} unrelated {:.3} overlap {:+.3} -> {:?}", a.kept(), a.unrelated, a.overlap, a.verdict());
            }
        }
        for (width, gain) in [(4usize, 4.0f32), (8, 3.0), (24, 3.0), (40, 3.0), (4, 1.0), (40, 1.0), (4, 0.5), (40, 0.5)] {
            for seed in [5, 6] {
                let a = solve(&Net::new(Spec { seed, ..spec(width, 30, gain) })).2;
                println!("width {width} gain {gain} seed {seed}: kept {:.3} unrelated {:.3} overlap {:+.3} -> {:?}", a.kept(), a.unrelated, a.overlap, a.verdict());
            }
        }
        // the edge experiment with one random network after another
        let edge: Vec<String> = (4..28).map(|seed| format!("{:+.3}", solve(&Net::new(Spec { seed, ..EXPERIMENTS[2].2 })).1.chi(0))).collect();
        println!("gain 1, seeds 4..28: chi1 {}", edge.join(" "));
        let start = std::time::Instant::now();
        let net = Net::new(spec(40, 60, 1.5));
        let quick = quick(&net);
        let t_quick = start.elapsed();
        let mut job = Job::new(&net, &quick.jac);
        let (mut steps, mut cost) = (0, 0.0);
        while !job.done() {
            cost += job.step();
            steps += 1;
        }
        println!("largest network: quick {:?}, job {:?} in {steps} steps, {cost:.2e} multiplications counted", t_quick, start.elapsed() - t_quick);
    }
}

//! Surfaces and gradients.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "surface",
    title: "Surfaces and gradients",
    group: "Calculus",
    blurb: "Type any $z = f(x, y)$: tangent plane, partial derivatives, gradient field.",
    make: |g| Box::new(SurfaceGrad::new(g)),
};

const SURF_N: u32 = 144;

pub struct SurfaceGrad {
    f: ExprField,
    surf: Surface,
    map: Surface,
    p: Vec2,
    extent: f32,
    show_plane: bool,
    show_field: bool,
    show_slices: bool,
    cmap: usize,
    dirty: bool,
    xs: Vec<f32>,
    ys: Vec<f32>,
    zs: Vec<f32>,
    batch: Batch,
    zscale: f32,
    range: [f32; 2],
    /// Whether the low and the high end of the range cut off a pole.
    cut: [bool; 2],
    info: [f32; 3],
    /// Is the point's height within the heights drawn (not cut off, nor a hole)?
    on_surface: bool,
    /// The time the surface was last worked out for (an animated formula stands
    /// still while the scene is paused, and need not be worked out again).
    shown_t: f32,
}

/// The colour range for an animated formula: it follows the values on show, widening
/// at once when they go beyond it and closing in slowly when they fall short (most of
/// the way in about two seconds of scene time, whatever the frame rate), so that it
/// neither flickers nor stays stretched for good by one passing spike.
fn breathe(range: [f32; 2], lo: f32, hi: f32, dt: f32) -> [f32; 2] {
    let k = 1.0 - (-dt.max(0.0) / 0.8).exp();
    let low = lo.min(range[0] + (lo - range[0]) * k);
    let high = hi.max(range[1] + (hi - range[1]) * k);
    [low, high.max(low + 1e-6)]
}

/// The heights to draw: from the lowest value to the highest, unless an end is set by
/// a pole, where f grows past any size. Such an end is cut back to the bulk of the
/// values (the 5th or 95th percentile, and half their spread beyond), or one sample
/// beside the pole would squash the rest of the surface flat. Whether an end comes
/// from a pole is told by looking closer: round a pole f is very different within
/// half a cell (much larger, or, if the sample all but hit it, much smaller); round a
/// top that the grid resolves, a narrow one or a steep polynomial's at the edge, it is
/// nearly the same. `zs` are f at (`xs`, `ys`), `cell` apart; holes are left out.
/// Returns the range and which ends were cut.
fn height_range(zs: &[f32], xs: &[f32], ys: &[f32], cell: f32, f: &impl Fn(f32, f32) -> f32) -> ([f32; 2], [bool; 2]) {
    let mut finite: Vec<f32> = zs.iter().copied().filter(|z| z.is_finite()).collect();
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut at_lo, mut at_hi) = (0, 0);
    for (i, &z) in zs.iter().enumerate() {
        if z.is_finite() {
            if z < lo {
                (lo, at_lo) = (z, i);
            }
            if z > hi {
                (hi, at_hi) = (z, i);
            }
        }
    }
    if finite.len() < 50 {
        return if lo <= hi { ([lo, hi.max(lo + 1e-6)], [false; 2]) } else { ([0.0, 1.0], [false; 2]) };
    }
    let n = finite.len();
    let p_lo = *finite.select_nth_unstable_by(n * 5 / 100, f32::total_cmp).1;
    let p_hi = *finite.select_nth_unstable_by(n * 95 / 100, f32::total_cmp).1;
    let (reach, middle) = (0.5 * (p_hi - p_lo).max(1e-6), 0.5 * (p_lo + p_hi));
    // has the grid not caught up with the value at sample `i`? (Measured from the
    // middle of the bulk: within half a cell of it, f is half as far again from there,
    // or a third nearer; or infinite.)
    let pole = |i: usize| {
        let (x, y, v) = (xs[i], ys[i], zs[i]);
        (-2..=2).flat_map(|j| (-2..=2).map(move |i| (i, j))).any(|(dx, dy)| {
            let z = f(x + cell * 0.25 * dx as f32, y + cell * 0.25 * dy as f32);
            z.is_infinite() || (z.is_finite() && !(0.67..=1.5).contains(&((z - middle) / (v - middle))))
        })
    };
    let mut cut = [false; 2];
    if lo < p_lo - reach && pole(at_lo) {
        (lo, cut[0]) = (p_lo - reach, true);
    }
    if hi > p_hi + reach && pole(at_hi) {
        (hi, cut[1]) = (p_hi + reach, true);
    }
    ([lo, hi.max(lo + 1e-6)], cut)
}

/// A height as drawn on the surface: none where f has none, nor beyond a cut end of
/// the range. (The surface is cut open there. Held at the end of the range instead,
/// a pole that f jumps across, as tan does, would stand as a wall from one end to the
/// other.)
fn shown(z: f32, range: [f32; 2]) -> f32 {
    if z >= range[0] && z <= range[1] {
        z
    } else {
        f32::NAN
    }
}

/// A height as coloured on the map: beyond an end of the range, the colour of that end.
fn clamped(z: f32, range: [f32; 2]) -> f32 {
    if z.is_finite() {
        z.max(range[0]).min(range[1])
    } else {
        f32::NAN
    }
}

/// One of the "things to try": the formula, where the point is put, and the domain.
struct Trial {
    label: &'static str,
    look_for: &'static str,
    formula: &'static str,
    point: [f32; 2],
    extent: f32,
}

const TRIALS: [Trial; 4] = [
    Trial {
        label: "a saddle point",
        look_for: "At the centre the gradient is zero and the tangent plane lies flat, yet this is neither a top nor a bottom: the pink slice along $x$ curves up, the green one along $y$ curves down.",
        formula: "0.3 (x^2 - y^2)",
        point: [0.0, 0.0],
        extent: 3.0,
    },
    Trial {
        label: "steeper further out",
        look_for: r"A bowl. The gradient points straight away from its bottom, and $|\nabla f|$ is half the point's distance from the centre: here 1.5 away, so 0.75. Drag the point twice as far out and it doubles.",
        formula: "0.25 (x^2 + y^2)",
        point: [1.2, 0.9],
        extent: 3.0,
    },
    Trial {
        label: "where $f$ has no value",
        look_for: r"Half a ball. Outside the circle of radius 2 the square root has no value, and the surface has a hole there. Drag the point towards the rim: the slope $|\nabla f|$ grows without limit (3 at 1.9 from the centre, 10 at 1.99); past it the read-outs say none.",
        formula: "sqrt(4 - x^2 - y^2)",
        point: [0.6, 0.4],
        extent: 3.0,
    },
    Trial {
        label: "a pole",
        look_for: r"$1/(x^2 + y^2)$ grows past any size at the centre, where it has no value. Where it rises past the top of the colour scale (marked $\ge$ there) the surface is cut open, so the rest keeps its shape; the read-outs give the true values.",
        formula: "1/(x^2 + y^2)",
        point: [1.5, 1.0],
        extent: 3.0,
    },
];

impl SurfaceGrad {
    pub fn new(gfx: &Gfx) -> Self {
        SurfaceGrad {
            f: ExprField::new("sin(x) cos(y) + 0.1(x^2+y^2)", &["x", "y", "t"]),
            surf: Surface::new(gfx, SURF_N, SURF_N),
            map: Surface::new(gfx, SURF_N, SURF_N),
            p: Vec2::new(0.9, -0.7),
            extent: 3.0,
            show_plane: true,
            show_field: true,
            show_slices: true,
            cmap: 0,
            dirty: true,
            xs: Vec::new(),
            ys: Vec::new(),
            zs: Vec::new(),
            batch: Batch::default(),
            zscale: 1.0,
            range: [0.0, 1.0],
            cut: [false; 2],
            info: [0.0; 3],
            on_surface: true,
            shown_t: f32::NAN,
        }
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.f.set(t.formula);
        (self.p, self.extent) = (Vec2::from(t.point), t.extent);
        (self.show_plane, self.show_slices, self.show_field) = (true, true, true);
        self.dirty = true;
    }
}

/// Where the samples of the surface are: `n` by `n` over the square ±`e`.
fn grid(e: f32, n: usize, xs: &mut [f32], ys: &mut [f32]) {
    for j in 0..n {
        for i in 0..n {
            xs[j * n + i] = -e + 2.0 * e * i as f32 / (n - 1) as f32;
            ys[j * n + i] = -e + 2.0 * e * j as f32 / (n - 1) as f32;
        }
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

const SURF_CMAPS: [Cmap; 6] = [Cmap::Aurora, Cmap::Viridis, Cmap::Magma, Cmap::Turbo, Cmap::IceFire, Cmap::Ocean];

impl Scene for SurfaceGrad {
    fn view(&self) -> View {
        View::Orbit { target: Vec3::new(0.0, 0.0, -0.7), yaw: -0.95, pitch: 0.5, dist: 16.0 }
    }

    fn animated(&self) -> bool {
        self.f.expr.uses(2)
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A function of two numbers, $z = f(x, y)$, drawn as a landscape: $x$ and $y$ say where on the ground, $z$ how high. Below it the same function as a flat map, with contour lines joining points of equal height. At the white point you see how steep the landscape is, and which way is uphill.");
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }
        ui::section(ui, "The function");
        if self.f.ui(ui, "z =") {
            self.dirty = true;
        }
        let presets = [
            ("hills", "sin(x) cos(y) + 0.1(x^2+y^2)"),
            ("bowl", "0.25 (x^2 + y^2)"),
            ("saddle", "0.3 (x^2 - y^2)"),
            ("ripple", "cos(2 sqrt(x^2 + y^2) - 2t) exp(-0.15 (x^2 + y^2))"),
            ("peaks", "2 exp(-(x-1)^2 - (y+0.5)^2) - 1.5 exp(-(x+1)^2 - (y-1)^2) + exp(-x^2 - (y+2)^2)"),
            ("monkey", "(x^3 - 3 x y^2) / 8"),
            ("egg crate", "0.6 sin(2x) sin(2y)"),
        ];
        let labels: Vec<&str> = presets.iter().map(|p| p.0).collect();
        if let Some(i) = ui::buttons(ui, &labels) {
            self.f.set(presets[i].1);
            self.dirty = true;
        }
        ui::note(ui, "Use $x$, $y$ and $t$ (time, in seconds of the scene's clock) in the formula.");
        ui::section(ui, "Show");
        self.dirty |= ui::slider(ui, r"domain $\pm$", &mut self.extent, 0.5..=8.0);
        ui::note(ui, "How far $x$ and $y$ run each way from 0.");
        ui.checkbox(&mut self.show_plane, "tangent plane");
        ui.checkbox(&mut self.show_slices, "partial-derivative slices");
        ui.checkbox(&mut self.show_field, "gradient field on the contour map");
        let names: Vec<&str> = SURF_CMAPS.iter().map(|c| c.name()).collect();
        ui::choice(ui, &names, &mut self.cmap);
        ui::note(ui, "The colours that stand for height.");
        ui::section(ui, "Reading the picture");
        let cmap = SURF_CMAPS[self.cmap];
        ui::key(
            ui,
            &[
                (cmap.color(0.8), "colour: the height $z$, on the scale at the bottom left; thin lines are contour lines"),
                (pal::WHITE, "the point: drag it on the map"),
                (pal::PINK, r"the surface cut along $x$ through the point: its slope there is $\partial f/\partial x$"),
                (pal::GREEN, r"the same along $y$: its slope there is $\partial f/\partial y$"),
                (pal::WHITE.a(0.5), "the tangent plane: the flat sheet that touches the surface at the point"),
                (pal::YELLOW, r"the gradient $\nabla f$ on the map, and the way uphill on the surface"),
                (pal::WHITE.a(0.8), "white arrows on the map: the gradient everywhere; the longer, the steeper"),
            ],
        );
        ui::note(ui, r"Where $f$ has no value the surface and the map have a hole. Where it goes past an end of the colour scale marked $\ge$ or $\le$ (near a pole, where $f$ grows past any size), the surface is cut open.");
        ui::section(ui, "Words");
        ui::term(ui, r"Partial derivative $\partial f/\partial x$:", "the slope of the surface going in the $x$ direction only, $y$ held fixed.");
        ui::term(ui, r"Gradient $\nabla f$:", r"the arrow $(\partial f/\partial x, \partial f/\partial y)$. It points straight uphill, crosses the contour line through the point at a right angle, and its length $|\nabla f|$ is the steepest slope there.");
        ui::section(ui, "At the point");
        ui::readout(ui, "$f$", number(self.info[0]));
        ui::readout_colored(ui, r"$\partial f/\partial x$", number(self.info[1]), pal::PINK);
        ui::readout_colored(ui, r"$\partial f/\partial y$", number(self.info[2]), pal::GREEN);
        ui::readout_colored(ui, r"$|\nabla f|$", number(self.info[1].hypot(self.info[2])), pal::YELLOW);
        if !self.info[0].is_finite() {
            ui::note(ui, "$f$ has no value at this point.");
        } else if !self.on_surface {
            ui::note(ui, "The point is where the surface is cut open: $f$ there is beyond the colour scale.");
        }
        ui::see_also(ui, &[("calc1", "the slope of a curve, in one dimension"), ("descent", "walking downhill along the gradient"), ("field", "arrows all over the plane: vector fields"), ("parametric", "surfaces that fold back over themselves")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let e = self.extent;
        let n = SURF_N as usize;
        // (the domain may have shrunk since the point was put down)
        self.p = self.p.clamp(Vec2::splat(-e), Vec2::splat(e));
        if self.dirty || (self.f.expr.uses(2) && f.t != self.shown_t) {
            self.shown_t = f.t;
            if self.xs.len() != n * n {
                self.xs.resize(n * n, 0.0);
                self.ys.resize(n * n, 0.0);
                self.zs.resize(n * n, 0.0);
                self.dirty = true;
            }
            if self.dirty {
                grid(e, n, &mut self.xs, &mut self.ys);
            }
            self.f.expr.eval_batch(&[&self.xs, &self.ys, &[f.t]], &mut self.zs, &mut self.batch);
            // (where f has no value the surface has a hole: the range is that of the
            // values there are)
            let (expr, t) = (&self.f.expr, f.t);
            let ([lo, hi], cut) = height_range(&self.zs, &self.xs, &self.ys, 2.0 * e / (n - 1) as f32, &|x, y| expr.eval(&[x, y, t]));
            self.range = if self.dirty { [lo, hi] } else { breathe(self.range, lo, hi, f.dt) };
            self.cut = cut;
            // world: xy scaled so the domain is 6 wide; z scaled to stay readable
            let span = self.range[1] - self.range[0];
            self.zscale = if span * 3.0 / e > 3.6 { 3.6 / span } else { 3.0 / e };
            let (k, zs, zmid, range) = (3.0 / e, self.zscale, 0.5 * (self.range[0] + self.range[1]), self.range);
            let (xs, ys, zv) = (&self.xs, &self.ys, &self.zs);
            for (i, p) in self.surf.data_mut().iter_mut().enumerate() {
                let z = shown(zv[i], range);
                *p = [xs[i] * k, ys[i] * k, (z - zmid) * zs, z];
            }
            for (i, p) in self.map.data_mut().iter_mut().enumerate() {
                *p = [xs[i] * k, ys[i] * k, 0.0, clamped(zv[i], range)];
            }
            self.dirty = false;
        }
        let (k, zs, zmid) = (3.0 / e, self.zscale, 0.5 * (self.range[0] + self.range[1]));
        let floor_z = (self.range[0] - zmid) * zs - 1.5;
        let cmap = SURF_CMAPS[self.cmap];
        let contour = nice_step((self.range[1] - self.range[0]) / 14.0);
        let st = &mut self.surf.style;
        st.cmap = Some(cmap);
        st.range = self.range;
        st.contour = contour;
        st.contour_alpha = 0.45;
        st.specular = 0.4;
        let st = &mut self.map.style;
        st.cmap = Some(cmap);
        st.range = self.range;
        st.contour = contour;
        st.contour_alpha = 0.7;
        st.lit = 0.0;
        st.model = Mat4::from_translation(Vec3::new(0.0, 0.0, floor_z));
        f.surface(&mut self.surf);
        f.surface(&mut self.map);

        let t = f.t;
        let expr = &self.f.expr;
        let range = self.range;
        let val = |x: f32, y: f32| expr.eval(&[x, y, t]);
        // (the slices have a gap where the surface does)
        let lift = |x: f32, y: f32, z: f32| Vec3::new(x * k, y * k, (shown(z, range) - zmid) * zs);

        // gradient field on the map: arrows cross contour lines at right angles
        if self.show_field {
            let m = 15;
            let cell = 2.0 * e / m as f32;
            let mut grads = Vec::with_capacity(m * m);
            for j in 0..m {
                for i in 0..m {
                    let (x, y) = (-e + cell * (i as f32 + 0.5), -e + cell * (j as f32 + 0.5));
                    let (_, gx, gy) = expr.grad2(&[x, y, t], 0, 1);
                    let g = Vec2::new(gx, gy);
                    // (no arrow where f has no slope)
                    if g.is_finite() {
                        grads.push((x, y, g));
                    }
                }
            }
            // the longest arrow is the steepest slope, unless that is a pole's: then it
            // is three times the slope most of the map has, and steeper is drawn as long
            let mut lengths: Vec<f32> = grads.iter().map(|g| g.2.length()).collect();
            let at = lengths.len() * 9 / 10;
            let typical = if lengths.is_empty() { 0.0 } else { *lengths.select_nth_unstable_by(at, f32::total_cmp).1 };
            let gmax = grads.iter().map(|g| g.2.length()).fold(1e-9f32, f32::max).min(3.0 * typical).max(1e-9);
            for (x, y, g) in grads {
                let len = g.length();
                if len < 1e-6 * gmax {
                    continue;
                }
                let d = g / len * (0.25 + 0.6 * (len / gmax).sqrt().min(1.0)) * cell * 0.5;
                let c = Vec3::new(x * k, y * k, floor_z + 0.01);
                let d3 = Vec3::new(d.x * k, d.y * k, 0.0);
                f.arrow(c - d3, c + d3, 1.4, pal::WHITE.a(0.8));
            }
        }

        // the draggable point lives on the map
        let mut q = Vec3::new(self.p.x * k, self.p.y * k, floor_z);
        if f.drag3(1, &mut q, Some(Vec3::Z), pal::WHITE) {
            self.p = Vec2::new((q.x / k).clamp(-e, e), (q.y / k).clamp(-e, e));
        }
        let (x0, y0) = (self.p.x, self.p.y);
        let (z0, gx, gy) = expr.grad2(&[x0, y0, t], 0, 1);
        self.info = [z0, gx, gy];
        // on the surface only where it is drawn: not over a hole, nor beyond a cut (a
        // hair beyond an end that is not cut is the grid missing the very top, and
        // counts as on it)
        let slack = self.cut.map(|cut| if cut { 0.0 } else { 0.05 * (range[1] - range[0]) });
        self.on_surface = z0 >= range[0] - slack[0] && z0 <= range[1] + slack[1];
        let on = self.on_surface;
        // (a hair beyond the range is drawn at its end)
        let p3 = Vec3::new(x0 * k, y0 * k, (clamped(z0, range) - zmid) * zs);
        if on {
            f.dashed(Vec3::new(p3.x, p3.y, floor_z), p3, 1.4, pal::WHITE.a(0.7));
            f.sphere(p3, 0.16, pal::WHITE);
        }

        if self.show_slices {
            let m = 160;
            f.path((0..=m).map(|i| -e + 2.0 * e * i as f32 / m as f32).map(|x| lift(x, y0, val(x, y0))), 2.6, pal::PINK);
            f.path((0..=m).map(|i| -e + 2.0 * e * i as f32 / m as f32).map(|y| lift(x0, y, val(x0, y))), 2.6, pal::GREEN);
            // named at their far ends, where the surface has them (and kept in view: on
            // a phone the ends are often past its edge)
            let view = f.view_size();
            for (end, name, color) in [(lift(e, y0, val(e, y0)), "cut along $x$", pal::PINK), (lift(x0, e, val(x0, e)), "cut along $y$", pal::GREEN)] {
                if let Some(at) = on_screen(f, end).filter(|_| end.is_finite()) {
                    let at = Vec2::new((at.x + 6.0).min(view.x - f.text_size(name, 12.0).x - 14.0), at.y - 6.0);
                    f.caption_px(at, name, 12.0, color, Align2::LEFT_BOTTOM);
                }
            }
        }
        // tangent vectors in world space
        let tx = Vec3::new(k, 0.0, gx * zs);
        let ty = Vec3::new(0.0, k, gy * zs);
        let g = Vec2::new(gx, gy);
        if self.show_plane && on && g.is_finite() {
            let (u, v) = (tx.normalize() * 1.1, ty.normalize() * 1.1);
            let c = [p3 - u - v, p3 + u - v, p3 + u + v, p3 - u + v];
            f.dl.tris_alpha.quad(c[0], c[1], c[2], c[3], pal::WHITE.a(0.22));
            f.dl.lines.closed(&c, 1.4, pal::WHITE.a(0.85));
        }
        // gradient: on the map, and as the direction of steepest ascent on the surface
        if g.is_finite() && g.length() > 1e-6 {
            let len = (g.length() * zs / k).clamp(0.25, 1.0) * 1.6;
            let d = g.normalize();
            let base = Vec3::new(p3.x, p3.y, floor_z + 0.02);
            f.arrow(base, base + Vec3::new(d.x, d.y, 0.0) * len, 3.2, pal::YELLOW);
            let mut tips = vec![(base + Vec3::new(d.x, d.y, 0.0) * len, r"$\nabla f$")];
            if on {
                let up = (tx * d.x + ty * d.y).normalize() * len;
                f.arrow(p3, p3 + up, 3.0, pal::YELLOW);
                tips.push((p3 + up, "uphill"));
            }
            // on plates: yellow text is lost against the bright end of most colour maps
            for (tip, name) in tips {
                if let Some(at) = on_screen(f, tip) {
                    f.caption_px(at + Vec2::new(8.0, -8.0), name, 14.0, pal::YELLOW, Align2::LEFT_BOTTOM);
                }
            }
        }
        // colour key: the surface and the map share one height scale
        let s = f.view_size();
        let (x0, y0, w, h) = (18.0, s.y - 36.0, 150.0, 8.0);
        let steps = 30;
        for i in 0..steps {
            let (a, b) = (i as f32 / steps as f32, (i + 1) as f32 / steps as f32);
            hud_rect(f, Vec2::new(x0 + w * a, y0), Vec2::new(x0 + w * b + 0.5, y0 + h), cmap.color((a + b) * 0.5));
        }
        // a thin frame, because the low end of most colour maps is as dark as the background
        let corners = [Vec2::new(x0, y0), Vec2::new(x0 + w, y0), Vec2::new(x0 + w, y0 + h), Vec2::new(x0, y0 + h)];
        for i in 0..4 {
            hud_line(f, corners[i], corners[(i + 1) % 4], 1.0, pal::DIM.a(0.7));
        }
        f.text_px(Vec2::new(x0, y0 - 5.0), "colour shows the height $z$", 12.0, pal::DIM, Align2::LEFT_BOTTOM);
        // (an end that cuts off a pole says so: the surface goes on beyond it)
        let ends = [0, 1].map(|i| if self.cut[i] { format!("${} {}$", [r"\le", r"\ge"][i], fmt(self.range[i])) } else { format!("${}$", fmt(self.range[i])) });
        f.text_px(Vec2::new(x0, y0 + h + 4.0), &ends[0], 12.0, pal::DIM, Align2::LEFT_TOP);
        f.text_px(Vec2::new(x0 + w, y0 + h + 4.0), &ends[1], 12.0, pal::DIM, Align2::RIGHT_TOP);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_colour_range_breathes() {
        // one spike widens it at once
        let spiked = breathe([-1.0, 1.0], -1.0, 40.0, 1.0 / 60.0);
        assert_eq!(spiked, [-1.0, 40.0]);
        // and it closes in again once the spike has passed (it used to stay for good)
        let mut range = spiked;
        for _ in 0..600 {
            range = breathe(range, -1.0, 1.0, 1.0 / 60.0);
        }
        assert!(range[1] < 1.01 && range[0] == -1.0, "{range:?}");
        // never narrower than the values on show
        let r = breathe([-0.2, 0.2], -3.0, 5.0, 1.0 / 60.0);
        assert!(r[0] <= -3.0 && r[1] >= 5.0);
        // and nothing moves while the scene is paused
        assert_eq!(breathe([-1.0, 40.0], -1.0, 1.0, 0.0), [-1.0, 40.0]);
    }

    /// The range closes in at the same pace on a 30 Hz and a 240 Hz display (it used
    /// to move by a fixed share each frame, so eight times as fast on the second).
    #[test]
    fn the_colour_range_closes_in_by_time() {
        let after = |hz: f32, seconds: f32| {
            let mut range = [-1.0, 40.0];
            for _ in 0..(hz * seconds).round() as usize {
                range = breathe(range, -1.0, 1.0, 1.0 / hz);
            }
            range[1]
        };
        for seconds in [0.5, 1.0, 2.0] {
            let (slow, fast) = (after(30.0, seconds), after(240.0, seconds));
            assert!((slow - fast).abs() < 1e-3 * slow, "{seconds} s: {slow} at 30 Hz, {fast} at 240 Hz");
        }
        // most of the way in two seconds
        assert!(after(60.0, 2.0) < 1.0 + 0.1 * 39.0);
    }

    /// The scene's samples of `formula` over ±`e`, and its height range.
    fn heights(formula: &str, e: f32) -> (Vec<f32>, [f32; 2], [bool; 2]) {
        let expr = Expr::parse(formula, &["x", "y", "t"]).unwrap();
        let n = SURF_N as usize;
        let (mut xs, mut ys, mut zs) = (vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]);
        grid(e, n, &mut xs, &mut ys);
        expr.eval_batch(&[&xs, &ys, &[0.0]], &mut zs, &mut Batch::default());
        let (range, cut) = height_range(&zs, &xs, &ys, 2.0 * e / (n - 1) as f32, &|x, y| expr.eval(&[x, y, 0.0]));
        (zs, range, cut)
    }

    /// Where f has no value there is a hole, not a floor at 0; and one sample beside a
    /// pole no longer sets the scale for the whole surface. Smooth surfaces, peaks and
    /// all, keep their full range.
    #[test]
    fn holes_and_poles_leave_the_rest_of_the_surface_alone() {
        // sqrt(x) + 5: the left half is a hole, and the range is that of the right half
        let (zs, range, cut) = heights("sqrt(x) + 5", 3.0);
        let holes = zs.iter().filter(|z| !z.is_finite()).count();
        assert!(holes == zs.len() / 2 && cut == [false; 2], "{holes} holes");
        assert!((range[0] - 5.0).abs() < 0.2 && (range[1] - 5.0 - 3f32.sqrt()).abs() < 1e-4, "{range:?}");
        assert!(zs.iter().all(|&z| !z.is_finite() || shown(z, range) == z));
        assert!(shown(f32::NAN, range).is_nan() && shown(f32::INFINITY, range).is_nan() && shown(range[1] + 0.1, range).is_nan());
        assert!(clamped(range[1] + 0.1, range) == range[1] && clamped(f32::NAN, range).is_nan());
        // poles: the surface is no longer squashed flat. (The share of it within 1 % of
        // the range from the bottom, or for an odd function within 1 % of the middle,
        // with the old range of the lowest to the highest value and with the new.)
        for (formula, ends, before, after) in [("1/(x^2 + y^2)", [false, true], 0.98, 0.1), ("1/x", [true, true], 0.6, 0.05), ("tan(x)", [true, true], 0.6, 0.2)] {
            let (zs, range, cut) = heights(formula, 3.0);
            let finite: Vec<f32> = zs.iter().copied().filter(|z| z.is_finite()).collect();
            let raw = [finite.iter().copied().fold(f32::INFINITY, f32::min), finite.iter().copied().fold(f32::NEG_INFINITY, f32::max)];
            let flat = |range: [f32; 2]| {
                let level = if cut[0] { 0.5 * (range[0] + range[1]) } else { range[0] };
                finite.iter().filter(|&&z| (shown(z, range) - level).abs() < 0.01 * (range[1] - range[0])).count() as f32 / finite.len() as f32
            };
            assert!(cut == ends && flat(raw) > before && flat(range) < after, "{formula}: {range:?} {cut:?}: {} flat, {} with the old range", flat(range), flat(raw));
            assert!(range[1] - range[0] < 40.0, "{formula}: {range:?}");
        }
        // smooth surfaces keep the whole of their range, a narrow peak among them
        for formula in ["sin(x) cos(y) + 0.1(x^2+y^2)", "(x^3 - 3 x y^2) / 8", "5 exp(-20 (x^2 + y^2))", "x^9", "exp(3x) - exp(-4y)", "2 exp(-(x-1)^2 - (y+0.5)^2) - 1.5 exp(-(x+1)^2 - (y-1)^2) + exp(-x^2 - (y+2)^2)"] {
            let (zs, range, cut) = heights(formula, 3.0);
            let finite = zs.iter().copied().filter(|z| z.is_finite());
            let (lo, hi) = (finite.clone().fold(f32::INFINITY, f32::min), finite.fold(f32::NEG_INFINITY, f32::max));
            assert!(range == [lo, hi] && cut == [false; 2], "{formula}: {range:?} against {lo}..{hi}");
        }
        // nothing at all: a range all the same
        let (_, range, _) = heights("sqrt(-1 - x^2)", 3.0);
        assert!(range[0].is_finite() && range[1] > range[0]);
    }

    /// The captions of the "things to try", held to account, with each experiment's
    /// formula, point and domain as its button sets them.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let parse = |k: usize| Expr::parse(TRIALS[k].formula, &["x", "y", "t"]).unwrap();
        let at = |e: &Expr, p: Vec2| e.grad2(&[p.x, p.y, 0.0], 0, 1);
        // a saddle: no slope at the centre, yet up along x and down along y
        let (saddle, centre) = (parse(0), Vec2::from(TRIALS[0].point));
        assert_eq!(at(&saddle, centre), (0.0, 0.0, 0.0));
        for d in [0.5, 1.0, 2.0] {
            assert!(saddle.eval(&[d, 0.0, 0.0]) > 0.0 && saddle.eval(&[-d, 0.0, 0.0]) > 0.0 && saddle.eval(&[0.0, d, 0.0]) < 0.0 && saddle.eval(&[0.0, -d, 0.0]) < 0.0);
        }
        // the bowl: the gradient points away from the bottom, half as long as the distance
        let (bowl, p) = (parse(1), Vec2::from(TRIALS[1].point));
        for q in [p, 2.0 * p] {
            let (_, gx, gy) = at(&bowl, q);
            let g = Vec2::new(gx, gy);
            assert!((g.length() - 0.5 * q.length()).abs() < 1e-5 && g.normalize().dot(q.normalize()) > 0.99999, "{q}: {g}");
        }
        assert!((p.length() - 1.5).abs() < 1e-6);
        // half a ball: a hole outside the circle, a slope that grows without limit at its rim
        let (ball, n) = (parse(2), SURF_N as usize);
        let (zs, _, _) = heights(TRIALS[2].formula, TRIALS[2].extent);
        let holes = zs.iter().filter(|z| z.is_nan()).count() as f32 / (n * n) as f32;
        assert!((holes - (1.0 - PI * 4.0 / 36.0)).abs() < 0.01, "{holes}");
        let slope = |r: f32| {
            let (_, gx, gy) = at(&ball, Vec2::new(r, 0.0));
            gx.hypot(gy)
        };
        assert!((slope(1.9) - 3.0).abs() < 0.1 && (slope(1.99) - 10.0).abs() < 0.1 && slope(1.999) > 30.0, "{} {}", slope(1.9), slope(1.99));
        assert!(at(&ball, Vec2::new(2.01, 0.0)).0.is_nan() && number(f32::NAN) == "none");
        // a pole: the top of the range is cut (and the key says so), and the point is on
        // the surface that is drawn
        let (pole, p) = (parse(3), Vec2::from(TRIALS[3].point));
        let (_, range, cut) = heights(TRIALS[3].formula, TRIALS[3].extent);
        let z = pole.eval(&[p.x, p.y, 0.0]);
        assert!(cut == [false, true] && z > range[0] && z < range[1], "{range:?} {cut:?} {z}");
    }

    /// The numbers behind the captions and the gradient field's scale (run with
    /// --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for formula in ["sin(x) cos(y) + 0.1(x^2+y^2)", "0.25 (x^2 + y^2)", "0.3 (x^2 - y^2)", "2 exp(-(x-1)^2 - (y+0.5)^2) - 1.5 exp(-(x+1)^2 - (y-1)^2) + exp(-x^2 - (y+2)^2)", "(x^3 - 3 x y^2) / 8", "0.6 sin(2x) sin(2y)", "1/(x^2 + y^2)", "sqrt(4 - x^2 - y^2)"] {
            let expr = Expr::parse(formula, &["x", "y", "t"]).unwrap();
            let (e, m) = (3.0f32, 15);
            let cell = 2.0 * e / m as f32;
            let mut lengths = Vec::new();
            for j in 0..m {
                for i in 0..m {
                    let (_, gx, gy) = expr.grad2(&[-e + cell * (i as f32 + 0.5), -e + cell * (j as f32 + 0.5), 0.0], 0, 1);
                    if gx.is_finite() && gy.is_finite() {
                        lengths.push(gx.hypot(gy));
                    }
                }
            }
            lengths.sort_by(f32::total_cmp);
            let (_, range, cut) = heights(formula, 3.0);
            println!("{formula:>40}: largest slope {} against 3 x the 90th percentile {}; range {range:?} cut {cut:?}", lengths.last().unwrap(), 3.0 * lengths[lengths.len() * 9 / 10]);
        }
        let expr = Expr::parse(TRIALS[2].formula, &["x", "y", "t"]).unwrap();
        for r in [1.0, 1.5, 1.9, 1.99, 1.999, 2.01] {
            let (z, gx, gy) = expr.grad2(&[r, 0.0, 0.0], 0, 1);
            println!("half ball at {r}: {z} slope {}", gx.hypot(gy));
        }
    }
}


//! Parametric surfaces.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "parametric",
    title: "Parametric surfaces",
    group: "Plotting",
    blurb: "Torus, Möbius band, Klein bottle, minimal surfaces, or your own formula.",
    make: |g| Box::new(Parametric::new(g)),
};

struct Shape {
    name: &'static str,
    x: &'static str,
    y: &'static str,
    z: &'static str,
    u: [f32; 2],
    v: [f32; 2],
    wrap: [bool; 2],
    /// What the slider a does to this shape ("" when the shape does not use it).
    a: &'static str,
}

const SHAPES: [Shape; 10] = [
    Shape {
        name: "torus",
        x: "(2 + a cos(v)) cos(u)",
        y: "(2 + a cos(v)) sin(u)",
        z: "a sin(v)",
        u: [0.0, TAU],
        v: [0.0, TAU],
        wrap: [true, true],
        a: "tube radius",
    },
    Shape {
        name: "Möbius band",
        x: "(2 + v cos(u/2)) cos(u)",
        y: "(2 + v cos(u/2)) sin(u)",
        z: "v sin(u/2)",
        u: [0.0, TAU],
        v: [-0.8, 0.8],
        wrap: [false, false],
        a: "",
    },
    Shape {
        name: "Klein bottle",
        x: "(2.4 + cos(u/2) sin(v) - sin(u/2) sin(2v)) cos(u)",
        y: "(2.4 + cos(u/2) sin(v) - sin(u/2) sin(2v)) sin(u)",
        z: "sin(u/2) sin(v) + cos(u/2) sin(2v)",
        u: [0.0, TAU],
        v: [0.0, TAU],
        wrap: [false, true],
        a: "",
    },
    Shape {
        name: "trefoil knot",
        x: "(2 + cos(1.5 u)) cos(u) + 0.45 a cos(v) cos(u)",
        y: "(2 + cos(1.5 u)) sin(u) + 0.45 a cos(v) sin(u)",
        z: "sin(1.5 u) + 0.45 a sin(v)",
        u: [0.0, 2.0 * TAU],
        v: [0.0, TAU],
        wrap: [true, true],
        a: "tube thickness",
    },
    Shape {
        name: "helicoid ⇄ catenoid",
        x: "cos(t/2) sinh(v) sin(u) + sin(t/2) cosh(v) cos(u)",
        y: "-cos(t/2) sinh(v) cos(u) + sin(t/2) cosh(v) sin(u)",
        z: "u cos(t/2) + v sin(t/2)",
        u: [-PI, PI],
        v: [-1.4, 1.4],
        wrap: [false, false],
        a: "",
    },
    Shape {
        name: "seashell",
        x: "1.2^u (1 + cos(v)) cos(u) a / 2",
        y: "1.2^u (1 + cos(v)) sin(u) a / 2",
        z: "1.2^u sin(v) a / 2 - 1.5 1.2^u + 2",
        u: [-6.0 * PI, 0.0],
        v: [0.0, TAU],
        wrap: [false, true],
        a: "tube width",
    },
    Shape {
        name: "breathing sphere",
        x: "(2 + 0.35 a sin(5u) sin(4v + t)) sin(v) cos(u)",
        y: "(2 + 0.35 a sin(5u) sin(4v + t)) sin(v) sin(u)",
        z: "(2 + 0.35 a sin(5u) sin(4v + t)) cos(v)",
        u: [0.0, TAU],
        v: [0.0, PI],
        wrap: [true, false],
        a: "depth of the ripples",
    },
    Shape {
        name: "Enneper",
        x: "u - u^3/3 + u v^2",
        y: "v - v^3/3 + v u^2",
        z: "u^2 - v^2",
        u: [-1.9, 1.9],
        v: [-1.9, 1.9],
        wrap: [false, false],
        a: "",
    },
    Shape {
        name: "Dini",
        x: "2 cos(u) sin(v)",
        y: "2 sin(u) sin(v)",
        z: "2 (cos(v) + ln(tan(v/2))) + 0.3 a u",
        u: [0.0, 2.0 * TAU],
        v: [0.08, 2.0],
        wrap: [false, false],
        a: "rise per turn",
    },
    Shape {
        name: "wave sheet",
        x: "u",
        y: "v",
        z: "a sin(u + t) cos(v) + 0.5 sin(2u - 2v + t)",
        u: [-4.0, 4.0],
        v: [-4.0, 4.0],
        wrap: [false, false],
        a: "height of the first wave",
    },
];

/// One of the "things to try": a shape (an index into `SHAPES`), the slider a, and
/// what the surface is coloured by (as in the "Colour the surface" choice).
struct Trial {
    label: &'static str,
    look_for: &'static str,
    shape: usize,
    a: f32,
    color_by: usize,
}

const TRIALS: [Trial; 3] = [
    Trial {
        label: "a band with one side",
        look_for: "The Möbius band, coloured by $v$, which runs across it. Follow it once round: where the two ends meet, each colour meets its opposite ($v$ meets $-v$), because the band has turned over on the way. It has one side and one edge.",
        shape: 1,
        a: 1.0,
        color_by: 2,
    },
    Trial {
        label: "a torus with no hole",
        look_for: "The tube's radius $a$ is now $2$, as far as the middle of the tube is from the centre: the hole has closed to a single point, where the inside of the tube touches itself. Lower $a$ and it opens again.",
        shape: 0,
        a: 2.0,
        color_by: 0,
    },
    Trial {
        label: "the same surface, bent",
        look_for: "As time runs, a spiral staircase (a helicoid) bends into a soap film between two rings (a catenoid) and back. Nothing is stretched on the way: every small piece keeps its lengths and angles.",
        shape: 4,
        a: 1.0,
        color_by: 0,
    },
];

/// A parameter bound as one would write it by hand: in multiples of π where it is one
/// (as markup, to go between dollars).
fn bound(v: f32) -> String {
    let k = (v / PI).round();
    let text = if v == 0.0 {
        "0".to_string()
    } else if (v / PI - k).abs() > 1e-4 {
        format!("{v}")
    } else if k.abs() == 1.0 {
        format!("{}π", if k < 0.0 { "-" } else { "" })
    } else {
        format!("{k}π")
    };
    text.replace('-', "−")
}

/// The surface's vertices from the formulas' values `out` (x, y and z) at the
/// parameters (`us`, `vs`), fitted into a fixed box whatever the shape's natural size,
/// with the number each is coloured by. Where a formula has no value the vertex is a
/// hole: it is not moved anywhere, which would draw long triangles out to wherever
/// that was. Returns the range of the colouring numbers.
fn vertices(out: &[Vec<f32>; 3], us: &[f32], vs: &[f32], color_by: usize, data: &mut [[f32; 4]]) -> [f32; 2] {
    let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for k in 0..data.len() {
        let p = Vec3::new(out[0][k], out[1][k], out[2][k]);
        if p.is_finite() {
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    if lo.x > hi.x {
        lo = Vec3::splat(-1.0);
        hi = Vec3::splat(1.0);
    }
    let centre = (lo + hi) * 0.5;
    let scale = 5.6 / (hi - lo).max_element().max(1e-6);
    let (mut slo, mut shi) = (f32::MAX, f32::MIN);
    for (k, q) in data.iter_mut().enumerate() {
        let raw = Vec3::new(out[0][k], out[1][k], out[2][k]);
        if !raw.is_finite() {
            *q = Surface::HOLE;
            continue;
        }
        let p = (raw - centre) * scale;
        let s = match color_by {
            0 => p.z,
            1 => us[k],
            2 => vs[k],
            _ => p.length(),
        };
        slo = slo.min(s);
        shi = shi.max(s);
        *q = [p.x, p.y, p.z, s];
    }
    if slo > shi {
        [0.0, 1.0]
    } else {
        [slo, shi.max(slo + 1e-6)]
    }
}

const PARAM_VARS: &[&str] = &["u", "v", "t", "a"];
const PARAM_N: u32 = 192;
const PARAM_CMAPS: [Cmap; 7] = [Cmap::Aurora, Cmap::Viridis, Cmap::Magma, Cmap::Plasma, Cmap::Turbo, Cmap::Twilight, Cmap::IceFire];

pub struct Parametric {
    shape: usize,
    fx: ExprField,
    fy: ExprField,
    fz: ExprField,
    urange: [f32; 2],
    vrange: [f32; 2],
    a: f32,
    color_by: usize,
    cmap: usize,
    lines: bool,
    spin: bool,
    surf: Surface,
    us: Vec<f32>,
    vs: Vec<f32>,
    out: [Vec<f32>; 3],
    batch: Batch,
    dirty: bool,
    /// The time the surface was last worked out for (it stands still while paused).
    shown_t: f32,
}

impl Parametric {
    pub fn new(gfx: &Gfx) -> Self {
        let s = &SHAPES[0];
        Parametric {
            shape: 0,
            fx: ExprField::new(s.x, PARAM_VARS),
            fy: ExprField::new(s.y, PARAM_VARS),
            fz: ExprField::new(s.z, PARAM_VARS),
            urange: s.u,
            vrange: s.v,
            a: 1.0,
            color_by: 0,
            cmap: 0,
            lines: true,
            spin: true,
            surf: Surface::new(gfx, PARAM_N, PARAM_N),
            us: Vec::new(),
            vs: Vec::new(),
            out: [Vec::new(), Vec::new(), Vec::new()],
            batch: Batch::default(),
            dirty: true,
            shown_t: f32::NAN,
        }
    }

    fn load(&mut self, i: usize) {
        let s = &SHAPES[i];
        self.shape = i;
        self.fx.set(s.x);
        self.fy.set(s.y);
        self.fz.set(s.z);
        self.urange = s.u;
        self.vrange = s.v;
        self.a = 1.0;
        self.dirty = true;
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        self.load(t.shape);
        (self.a, self.color_by, self.lines) = (t.a, t.color_by, true);
    }

    fn time_dependent(&self) -> bool {
        self.fx.expr.uses(2) || self.fy.expr.uses(2) || self.fz.expr.uses(2)
    }

    /// Are the three formulas still those of the chosen shape (not edited)?
    fn own(&self) -> bool {
        let shape = &SHAPES[self.shape];
        self.fx.text == shape.x && self.fy.text == shape.y && self.fz.text == shape.z
    }
}

impl Scene for Parametric {
    fn view(&self) -> View {
        View::Orbit { target: Vec3::ZERO, yaw: -0.8, pitch: 0.5, dist: 11.5 }
    }

    fn animated(&self) -> bool {
        self.spin || self.time_dependent()
    }

    fn auto_rotate(&self) -> f32 {
        if self.spin {
            0.16
        } else {
            0.0
        }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A surface can be made by a rule that sends two numbers $(u, v)$ to a point $(x, y, z)$ in space. As $u$ and $v$ run over their ranges, the points sweep out the surface: a sheet, a tube, a knot, or a band with only one side. Pick one, or write your own in terms of $u$, $v$, the slider $a$, and $t$ for time.");
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }
        ui::section(ui, "Shape");
        let names: Vec<&str> = SHAPES.iter().map(|s| s.name).collect();
        if let Some(i) = ui::buttons(ui, &names) {
            self.load(i);
        }
        ui::section(ui, "Formula");
        self.dirty |= self.fx.ui(ui, "x =");
        self.dirty |= self.fy.ui(ui, "y =");
        self.dirty |= self.fz.ui(ui, "z =");
        ui::readout(ui, "$u$ runs from", format!("${}$ to ${}$", bound(self.urange[0]), bound(self.urange[1])));
        ui::readout(ui, "$v$ runs from", format!("${}$ to ${}$", bound(self.vrange[0]), bound(self.vrange[1])));
        // say what the slider does, as long as the formula is still the shape's own
        let shape = &SHAPES[self.shape];
        let own = self.own();
        let used = self.fx.expr.uses(3) || self.fy.expr.uses(3) || self.fz.expr.uses(3);
        let label = if !used {
            "$a$ (this formula does not use it)".to_string()
        } else if own && !shape.a.is_empty() {
            format!("$a$: {}", shape.a)
        } else {
            "$a$".to_string()
        };
        self.dirty |= ui::slider(ui, &label, &mut self.a, 0.0..=2.0);
        ui::note(ui, "$a$: a number the formulas may use, to change the shape while you watch.");
        ui::section(ui, "Look");
        ui::note(ui, "Colour the surface");
        self.dirty |= ui::choice(ui, &["by height", "by $u$", "by $v$", "by distance"], &mut self.color_by);
        let names: Vec<&str> = PARAM_CMAPS.iter().map(|c| c.name()).collect();
        ui::choice(ui, &names, &mut self.cmap);
        checkbox(ui, &mut self.lines, "parameter lines ($u$ or $v$ held fixed)");
        ui.checkbox(&mut self.spin, "slowly rotate");
        ui::section(ui, "Reading the picture");
        let cmap = PARAM_CMAPS[self.cmap];
        let by = ["colour: the height $z$", "colour: $u$, from one end of its range to the other", "colour: $v$, from one end of its range to the other", "colour: the distance from the centre"][self.color_by.min(3)];
        ui::key(ui, &[(cmap.color(0.6), by)]);
        ui::note(ui, "The dark lines are parameter lines, 36 each way. Where a formula has no value (a square root of a negative number, say) the surface has a hole.");
        ui::section(ui, "Words");
        ui::term(ui, "Parameters $u$, $v$:", "the two numbers the rule takes; each pair gives one point of the surface.");
        ui::term(ui, "Parameter line:", "the curve traced when one parameter is held fixed and the other runs through its range.");
        ui::section(ui, "Numbers");
        ui::readout(ui, "triangles drawn", count((2 * (PARAM_N - 1) * (PARAM_N - 1)) as usize));
        ui::see_also(ui, &[("surface", "a surface as a graph $z = f(x, y)$, with slopes"), ("graph", "curves $y = f(x)$ from formulas"), ("fourier", "a closed curve drawn by circles")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let n = PARAM_N as usize;
        if self.us.len() != n * n {
            self.us.resize(n * n, 0.0);
            self.vs.resize(n * n, 0.0);
            for o in &mut self.out {
                o.resize(n * n, 0.0);
            }
            self.dirty = true;
        }
        if self.dirty || (self.time_dependent() && f.t != self.shown_t) {
            self.shown_t = f.t;
            if self.dirty {
                for j in 0..n {
                    for i in 0..n {
                        self.us[j * n + i] = self.urange[0] + (self.urange[1] - self.urange[0]) * i as f32 / (n - 1) as f32;
                        self.vs[j * n + i] = self.vrange[0] + (self.vrange[1] - self.vrange[0]) * j as f32 / (n - 1) as f32;
                    }
                }
            }
            let vars: [&[f32]; 4] = [&self.us, &self.vs, &[f.t], &[self.a]];
            self.fx.expr.eval_batch(&vars, &mut self.out[0], &mut self.batch);
            self.fy.expr.eval_batch(&vars, &mut self.out[1], &mut self.batch);
            self.fz.expr.eval_batch(&vars, &mut self.out[2], &mut self.batch);

            let range = vertices(&self.out, &self.us, &self.vs, self.color_by, self.surf.data_mut());
            // (the seams are joined only for the shape's own formulas: an edited one
            // need not close up, and joining it would draw a sheet across the gap)
            let wrap = if self.own() { SHAPES[self.shape].wrap } else { [false, false] };
            let st = &mut self.surf.style;
            st.wrap = wrap;
            st.range = range;
            self.dirty = false;
        }
        let st = &mut self.surf.style;
        st.cmap = Some(PARAM_CMAPS[self.cmap]);
        st.grid = if self.lines { [36.0, 36.0] } else { [0.0, 0.0] };
        st.grid_alpha = 0.32;
        st.contour = 0.0;
        st.specular = 0.5;
        st.rim = 0.7;
        f.surface(&mut self.surf);
        let s = f.view_size();
        let name = if self.own() { SHAPES[self.shape].name } else { "a surface of your own" };
        let by = ["height", "$u$", "$v$", "distance from the centre"][self.color_by.min(3)];
        f.caption_px(Vec2::new(18.0, s.y - 18.0), format!("{name}, coloured by {by}"), 14.0, pal::DIM, Align2::LEFT_BOTTOM);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shape's three formulas, as the scene parses them.
    fn formulas(shape: &Shape) -> [Expr; 3] {
        [shape.x, shape.y, shape.z].map(|f| Expr::parse(f, PARAM_VARS).unwrap())
    }

    /// The point a shape's formulas give at (u, v), at time t with the slider at a.
    fn point(f: &[Expr; 3], u: f32, v: f32, t: f32, a: f32) -> Vec3 {
        Vec3::from(f.each_ref().map(|e| e.eval(&[u, v, t, a])))
    }

    /// Where a formula has no value the surface has a hole, rather than a vertex at
    /// the centre of the box with long triangles reaching to it.
    #[test]
    fn a_formula_with_no_value_leaves_a_hole() {
        let n = PARAM_N as usize;
        let mobius = &SHAPES[1];
        for (x, u, v, no_value) in [("sqrt(v)", mobius.u, mobius.v, n * (n / 2)), ("ln(u) cos(v)", [0.0, TAU], [0.0, TAU], n)] {
            let (mut us, mut vs) = (vec![0.0; n * n], vec![0.0; n * n]);
            for j in 0..n {
                for i in 0..n {
                    us[j * n + i] = u[0] + (u[1] - u[0]) * i as f32 / (n - 1) as f32;
                    vs[j * n + i] = v[0] + (v[1] - v[0]) * j as f32 / (n - 1) as f32;
                }
            }
            let f = [x, mobius.y, mobius.z].map(|f| Expr::parse(f, PARAM_VARS).unwrap());
            let out = f.each_ref().map(|e| {
                let mut o = vec![0.0; n * n];
                e.eval_batch(&[&us, &vs, &[0.0], &[1.0]], &mut o, &mut Batch::default());
                o
            });
            let undefined = (0..n * n).filter(|&k| !(out[0][k].is_finite() && out[1][k].is_finite() && out[2][k].is_finite())).count();
            let mut data = vec![[0.0f32; 4]; n * n];
            let range = vertices(&out, &us, &vs, 0, &mut data);
            let holes = data.iter().filter(|q| **q == Surface::HOLE).count();
            let at_centre = data.iter().filter(|q| **q != Surface::HOLE && q[..3] == [0.0, 0.0, 0.0]).count();
            assert!(undefined == no_value && holes == undefined && at_centre == 0, "{x}: {undefined} undefined, {holes} holes, {at_centre} at the centre");
            assert!(range[0].is_finite() && range[1] > range[0]);
        }
        // nothing defined at all: all holes, and a range all the same
        let out = [vec![f32::NAN; 4], vec![0.0; 4], vec![0.0; 4]];
        let mut data = vec![[0.0f32; 4]; 4];
        assert_eq!(vertices(&out, &[0.0; 4], &[0.0; 4], 0, &mut data), [0.0, 1.0]);
        assert!(data.iter().all(|q| *q == Surface::HOLE));
    }

    /// The captions of the "things to try", held to account with each experiment's
    /// shape and slider as its button sets them.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // the Möbius band: once round, v comes back as −v (the colours meet their opposites)
        let trial = &TRIALS[0];
        let (shape, f) = (&SHAPES[trial.shape], formulas(&SHAPES[trial.shape]));
        assert!(shape.name.starts_with("Möbius") && trial.color_by == 2);
        for k in 0..=8 {
            let v = shape.v[0] + (shape.v[1] - shape.v[0]) * k as f32 / 8.0;
            let (end, start) = (point(&f, shape.u[1], v, 0.0, trial.a), point(&f, shape.u[0], -v, 0.0, trial.a));
            assert!(end.distance(start) < 1e-4, "v {v}: {end} against {start}");
            // (and not v itself, unless at the middle of the band)
            assert!(v.abs() < 1e-6 || end.distance(point(&f, shape.u[0], v, 0.0, trial.a)) > 0.1);
        }
        // the torus with a = 2: the tube reaches the axis (its hole is a point); with a = 1 it does not
        let trial = &TRIALS[1];
        let f = formulas(&SHAPES[trial.shape]);
        let nearest = |a: f32| {
            let mut d = f32::MAX;
            for i in 0..=72 {
                for j in 0..=72 {
                    let p = point(&f, i as f32 / 72.0 * TAU, j as f32 / 72.0 * TAU, 0.0, a);
                    d = d.min(p.truncate().length());
                }
            }
            d
        };
        assert!(SHAPES[trial.shape].name == "torus" && nearest(trial.a) < 1e-5 && (nearest(1.0) - 1.0).abs() < 1e-5);
        // helicoid and catenoid: the same lengths and angles all the way through (the
        // first fundamental form, by small differences, at several points and times)
        let trial = &TRIALS[2];
        let f = formulas(&SHAPES[trial.shape]);
        assert!(f.iter().any(|e| e.uses(2)), "it is meant to move with time");
        let form = |u: f32, v: f32, t: f32| {
            let h = 1e-3;
            let du = (point(&f, u + h, v, t, 1.0) - point(&f, u - h, v, t, 1.0)) / (2.0 * h);
            let dv = (point(&f, u, v + h, t, 1.0) - point(&f, u, v - h, t, 1.0)) / (2.0 * h);
            Vec3::new(du.dot(du), du.dot(dv), dv.dot(dv))
        };
        for (u, v) in [(0.3, -0.9), (-2.0, 0.4), (1.1, 1.2)] {
            let start = form(u, v, 0.0);
            for t in [1.0, PI, 5.0] {
                assert!(form(u, v, t).distance(start) < 1e-2 * start.length(), "({u}, {v}) at t {t}: {} against {start}", form(u, v, t));
            }
        }
        // (and it is a different shape in space: a helicoid at first, a catenoid at t = π)
        assert!(point(&f, 1.0, 0.0, 0.0, 1.0).z == 1.0 && point(&f, 1.0, 0.0, PI, 1.0).z.abs() < 1e-5);
    }
}

//! Renderer test card.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "showcase",
    title: "Renderer test card",
    group: "Extras",
    blurb: "Every primitive the renderer has, in one view.",
    make: |g| Box::new(Showcase::new(g)),
};

pub struct Showcase {
    surf: Surface,
    trails: TrailSet,
    pos: Vec<Vec3>,
    /// Scene time not yet turned into rows of the trails.
    owed: f32,
    /// Colour bars, one row per colour map: the test card's picture.
    bars: Image,
    handle: Vec3,
    /// The time the surface was last worked out for (it stands still while paused).
    shown_t: f32,
}

/// Scene time between two rows of the trails (three rows a frame at 60 Hz, as there
/// were when a frame always took three); each row moves the flow on by 0.9 of this.
const ROW_DT: f32 = 1.0 / 180.0;

// (a row's step must be one the method can take: see FLOW_STEP_MAX)
const _: () = assert!(0.9 * ROW_DT <= FLOW_STEP_MAX);

/// The most rows one frame takes: at a very low frame rate or a high time speed the
/// flow falls behind instead.
const ROWS_MAX: usize = 8;

/// Every construct of the mathematics markup, as it is set on the picture (see
/// `typeset`; the panel has its own): rows of items, an item to a construct, wrapped
/// to the width of the view. (The first row is a sentence: its items are words.)
const SPECIMEN: [&[&str]; 12] = [
    &["Two", "arrows,", "$a$", "and", "$b$,", "and", "their", "dot", "product", r"$a \cdot b = |a|\,|b| \cos\theta$."],
    &[r"$x_1^2 + x_2^2 = r^2$", r"$A^{-1}$", r"$A^\top$", r"$\sigma^2_{k+1}$", r"$e^{At}$", r"$x''$", r"$2^{2^n}$"],
    &[r"typed: $x₁² + σ²ₖ₊₁ ≤ Aᵀ A⁻¹$,", r"$î ≠ ĵ$,", r"$λ ∈ ℝ$,", r"$√n$"],
    &[r"$\frac{a}{b}$", r"$\frac{1}{\sqrt{n}}$", r"$\sqrt{x^2 + y^2}$", r"$\sqrt[3]{x}$", r"$\binom{n}{k}$", r"$\tfrac{1}{2}$"],
    &[r"$\left( \frac{n-3}{2} \right)^2$", r"$(1 - c^2)^{(n-3)/2}$", r"$\left\| x \right\|$", r"$\left\{ x \middle| x > 0 \right\}$"],
    &[r"$\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$", r"$\begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc$"],
    &[r"$\det A$", r"$\tr A$", r"$\rank A$", r"$\dim \ker A$", r"$\ln x$", r"$\max_k$", r"$\lim_{n \to \infty}$", r"$\sum_{i=1}^n$", r"$\int_0^1 f\,dx$"],
    &[r"$\hat x$", r"$\bar x$", r"$\vec v$", r"$\tilde x$", r"$\dot x$", r"$\overline{AB}$", r"$\mathbf{x} \in \mathbb{R}^n$", r"$\mathrm{d}t$", r"$\text{in words}$", r"$\text{length}^2$", r"$|\text{dot product}|$"],
    &[r"$\textcolor{pink}{a_1} + \textcolor{green}{a_2} \ne b$", r"$x \notin S$", r"$2.5e-3$", r"$384\,000$", r"$1, \ldots, n$", r"$a + \cdots + z$", r"$\$5$"],
    &[r"$\mathrm{ReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$"],
    &[r"$\to \mapsto \Rightarrow \le \ge \approx \propto \subseteq$", r"$\pm \times \circ \otimes$", r"$\nabla \partial \infty \forall \exists \perp$", r"$\langle x, y \rangle \lfloor x \rfloor$"],
    &[r"$\big( \Big( \bigg( \Bigg($", r"$\left[ \frac{\frac{a}{b}}{c} \right]$", r"$\left\langle \frac{x}{2} \right\rangle$", r"$\left| \frac{1}{2} \right|$"],
];

/// The colour maps in the colour bars, top row first.
const BARS: [Cmap; 5] = [Cmap::Aurora, Cmap::Viridis, Cmap::Magma, Cmap::Turbo, Cmap::Twilight];

/// How many rows of the trails are due after `dt` more seconds of scene time; what is
/// left over stays `owed` for the next frame.
fn rows_due(owed: &mut f32, dt: f32) -> usize {
    *owed = (*owed + dt.max(0.0)).min(ROWS_MAX as f32 * ROW_DT);
    let rows = ((*owed / ROW_DT) as usize).min(ROWS_MAX);
    *owed = (*owed - rows as f32 * ROW_DT).max(0.0);
    rows
}

/// Where particle `i` of the flow starts.
fn flow_start(i: usize) -> Vec3 {
    let a = i as f32 * 0.618 * std::f32::consts::TAU;
    Vec3::new(0.3 * a.cos(), 0.3 * a.sin(), 2.0 + i as f32 * 0.002)
}

/// The longest step the flow is advanced by: it is integrated by Euler's method,
/// which loses its footing (and sends every particle to infinity within a few
/// frames) from a step of about 0.025.
const FLOW_STEP_MAX: f32 = 0.02;

/// One Euler step of the flow (the Lorenz equations, scaled to sit above the surface).
fn flow_step(p: Vec3, h: f32) -> Vec3 {
    let q = Vec3::new(p.x * 9.0, p.y * 9.0, (p.z - 1.2) * 16.0);
    let d = Vec3::new(10.0 * (q.y - q.x), q.x * (28.0 - q.z) - q.y, q.x * q.y - 8.0 / 3.0 * q.z);
    p + Vec3::new(d.x / 9.0, d.y / 9.0, d.z / 16.0) * h
}

impl Showcase {
    pub fn new(gfx: &Gfx) -> Self {
        let mut surf = Surface::new(gfx, 160, 160);
        surf.style.cmap = Some(Cmap::Aurora);
        surf.style.contour = 0.1;
        surf.style.contour_alpha = 0.55;
        let n = 400;
        let pos = (0..n).map(flow_start).collect();
        let cols = 24;
        let mut bars = Image::new(gfx, cols, BARS.len() as u32);
        for (i, c) in bars.pixels_mut().iter_mut().enumerate() {
            // (rows run from the bottom of the picture)
            let (row, col) = (BARS.len() - 1 - i / cols as usize, i % cols as usize);
            *c = BARS[row].color((col as f32 + 0.5) / cols as f32);
        }
        Showcase { surf, trails: TrailSet::new(gfx, n as u32, 140, true), pos, owed: 0.0, bars, handle: Vec3::new(1.6, -1.6, 0.0), shown_t: f32::NAN }
    }
}

/// The surface's height at (x, y) at time t.
fn ripple(x: f32, y: f32, t: f32) -> f32 {
    let r = (x * x + y * y).sqrt();
    0.55 * (r * 2.4 - t * 1.4).cos() * (-0.22 * r * r).exp() + 0.25
}

impl Scene for Showcase {
    fn view(&self) -> View {
        View::Orbit { target: Vec3::new(0.0, 0.0, 0.6), yaw: -0.9, pitch: 0.45, dist: 8.5 }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A test card rather than a lesson: everything the renderer can draw, in one view. A rippling surface with contour lines and a glowing ring floating on it; particles following the Lorenz equations above it, as trails; a lit six-sided block; lit spheres with dashed drop lines; a ring dashed evenly all round, with ringed discs and soft glows on it; the square under the surface in a long dash of its own; colour bars as a picture on the floor; the three axes; and an arrow whose tip you can drag.");
        ui::section(ui, "Mathematics");
        ui::note(ui, r"Every construct of the markup, here in the panel and on the picture. In running text: the vectors $a$ and $b$ make the angle $\theta$, and $a \cdot b = |a|\,|b| \cos\theta$; a matrix $A \in \mathbb{R}^{m \times n}$ with $A^\top A$ invertible has $x^* = (A^\top A)^{-1} A^\top b$; fractions in a line are small, $\frac{1}{\sqrt{n}}$; a long formula breaks after a relation or an operator: $\sigma^2_{k+1} + \sigma^2_{k+2} + \cdots + \sigma^2_n = \lVert A - A_k \rVert_F^2$.");
        ui::equation(ui, r"\sum_{i=1}^{n} x_i^2 = \lVert x \rVert^2");
        ui::equation(ui, r"\frac{\partial L}{\partial w} = \frac{1}{n} \sum_{i=1}^n (\hat y_i - y_i)\, x_i");
        ui::equation(ui, r"A = \begin{bmatrix} 1.50 & -0.50 \\ 0.50 & 1.20 \end{bmatrix}, \quad \det A = 2.05");
        ui::equation(ui, r"\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1} = 1 - \sigma(-x) = \frac{1}{2} + \frac{1}{2} \tanh\frac{x}{2}");
        ui::note(ui, r"A display too wide for the panel is broken between two formulas side by side, or before a relation or an operator, every line centred; and where its writer starts a new line:");
        ui::equation(ui, r"X = aZ + \sigma_X \varepsilon_X, \qquad Y = bX + cZ + \sigma_Y \varepsilon_Y");
        ui::equation(ui, r"\lVert A - A_k \rVert_F^2 = \sigma_{k+1}^2 + \sigma_{k+2}^2 + \sigma_{k+3}^2 + \sigma_{k+4}^2 + \cdots + \sigma_n^2");
        ui::equation(ui, r"(a + b)^2 = a^2 + 2ab + b^2 \\ (a - b)^2 = a^2 - 2ab + b^2");
        ui::note(ui, r"Words in a formula are set as the words round it are: $\text{length}^2 = |\text{dot product}|$, $\text{pixel} = u \times v^\top \times \sigma$; in a read-out too.");
        ui::readout(ui, r"$\text{pieces kept}$", r"$k = 3$");
        ui::term(ui, "Determinant:", r"$\det A = ad - bc$ for $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, the factor by which $A$ scales areas.");
        ui::term(ui, r"Partial derivative $\partial f / \partial x$:", r"the slope of the surface going in the $x$ direction only, $y$ held fixed.");
        ui::key(ui, &[(pal::PINK, r"$\textcolor{pink}{\hat\imath}$: where $(1, 0)$ lands"), (pal::GREEN, r"$\hat\jmath$: where $(0, 1)$ lands")]);
        ui::readout(ui, r"$\lambda_1, \lambda_2$", r"$1.35 \pm 0.48i$");
        ui::readout(ui, r"$\sigma_1 / \sigma_2$", "3.162");
        ui::readout_colored(ui, r"$x^* = (x_1, x_2)$", "(0.750, 0.500)", pal::CYAN);
        let _ = ui::experiment(ui, r"set $\lambda = 1$", r"An experiment's caption with $\lambda$ in it.");
    }

    fn frame(&mut self, f: &mut Frame) {
        let t = f.t;
        if t != self.shown_t {
            self.shown_t = t;
            self.surf.fill_height(Vec2::splat(-2.5), Vec2::splat(2.5), |x, y| ripple(x, y, t));
        }
        self.surf.style.range = [-0.3, 0.8];
        f.surface(&mut self.surf);
        f.axes3(Vec3::new(3.2, 3.2, 2.4));
        // a glowing ring that rides on the surface
        let crest = |a: f32| Vec3::new(1.25 * a.cos(), 1.25 * a.sin(), ripple(1.25, 0.0, t) + 0.03);
        f.glow_path((0..=96).map(|i| crest(i as f32 / 96.0 * TAU)), 2.0, pal::CYAN);
        // a ring of many short segments, dashed evenly all the way round
        f.dashed_path((0..=72).map(|i| Vec3::new(3.0 * (i as f32 * TAU / 72.0).cos(), 3.0 * (i as f32 * TAU / 72.0).sin(), 0.0)), 1.4, pal::DIM);
        // a line style and a point style of the scene's own (a fine long dash on the
        // straight sides of the square under the surface, ringed discs): the test card
        // is where the renderer's custom batches get exercised
        let square = f.dl.lines_with(LineKind::Blend, LineStyle { dash: 14.0, duty: 0.3, ..Default::default() });
        square.closed(&[Vec3::new(-2.5, -2.5, 0.0), Vec3::new(2.5, -2.5, 0.0), Vec3::new(2.5, 2.5, 0.0), Vec3::new(-2.5, 2.5, 0.0)], 1.4, pal::DIM);
        let marks = f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 2.5, outline: pal::WHITE, ..Default::default() });
        for i in 0..4 {
            marks.push(Vec3::new(3.0 * (i as f32 * TAU / 4.0).cos(), 3.0 * (i as f32 * TAU / 4.0).sin(), 0.0), 9.0, pal::PURPLE);
        }
        // soft glows halfway between them
        for i in 0..4 {
            let a = (i as f32 + 0.5) * TAU / 4.0;
            f.glow(Vec3::new(3.0 * a.cos(), 3.0 * a.sin(), 0.0), 56.0, pal::SERIES[i]);
        }
        // a picture: colour bars on the floor in front of the surface
        f.image(&mut self.bars, Vec2::new(-0.7, -2.95), Vec2::new(0.7, -2.6), 0.0);
        // filled polygons, lit and depth-tested: a six-sided block over the centre,
        // with flat discs at its top corners
        let corner = |k: usize, z: f32| Vec3::new(0.45 * (k as f32 * TAU / 6.0).cos(), 0.45 * (k as f32 * TAU / 6.0).sin(), z);
        let (bottom, top) = (1.0, 1.1);
        let tris = &mut f.dl.tris;
        tris.quad_lit(corner(0, top), corner(1, top), corner(2, top), corner(3, top), pal::PURPLE);
        tris.quad_lit(corner(0, top), corner(3, top), corner(4, top), corner(5, top), pal::PURPLE);
        tris.quad_lit(corner(0, bottom), corner(5, bottom), corner(4, bottom), corner(3, bottom), pal::PURPLE);
        tris.quad_lit(corner(0, bottom), corner(3, bottom), corner(2, bottom), corner(1, bottom), pal::PURPLE);
        for k in 0..6 {
            tris.quad_lit(corner(k, bottom), corner(k + 1, bottom), corner(k + 1, top), corner(k, top), pal::PURPLE);
        }
        for k in 0..6 {
            f.dot(corner(k, top), 7.0, pal::WHITE);
        }

        // a small chaotic flow above the surface: a row of the trails for every
        // ROW_DT of scene time, so they are as long on a 30 Hz display as on a 240 Hz one
        let colours = Cmap::Plasma.table();
        // (capped: a step too long for the method sends the particles away)
        let h = (0.9 * ROW_DT).min(FLOW_STEP_MAX);
        for _ in 0..rows_due(&mut self.owed, f.dt) {
            let row = self.trails.row_mut();
            for (i, (r, p)) in row.iter_mut().zip(&mut self.pos).enumerate() {
                let next = flow_step(*p, h);
                // (one that got away all the same starts again, on a new trail)
                let lost = !next.is_finite() || next.length() > 50.0;
                *p = if lost { flow_start(i) } else { next };
                r.pos = p.to_array();
                r.width = if lost { -1.0 } else { 1.6 };
                r.color = colours[table_index(((p.z - 1.2) / 2.6).clamp(0.0, 1.0))].a(0.5);
            }
            self.trails.commit(f.gfx);
        }
        f.trails(&mut self.trails);

        for i in 0..7 {
            let a = i as f32 / 7.0 * std::f32::consts::TAU + t * 0.3;
            let p = Vec3::new(2.9 * a.cos(), 2.9 * a.sin(), 0.35);
            f.sphere(p, 0.42, pal::SERIES[i % 8]);
            f.dashed(p, Vec3::new(p.x, p.y, 0.0), 1.2, pal::DIM);
        }
        f.drag3(1, &mut self.handle, Some(Vec3::Z), pal::YELLOW);
        f.arrow(Vec3::ZERO, self.handle, 3.0, pal::YELLOW);
        // the arrow lies in the plane z = 0, under the surface: show its line through it
        f.dl.overlay.segment(Vec3::ZERO, self.handle, 1.6, pal::YELLOW.a(0.6));
        // the mathematics specimen, on a panel of its own at the top left: each row's
        // items side by side for as long as they fit across the view
        let (size, gap, left, top) = (14.0, 22.0, 18.0, 16.0 + f.top_inset);
        let room = (f.view_size().x - 2.0 * left).max(40.0);
        let mut placed: Vec<(Vec2, &str)> = Vec::new();
        let (mut y, mut widest) = (top, 0.0f32);
        for (k, row) in SPECIMEN.iter().enumerate() {
            let gap = if k == 0 { f.text_size(" ", size).x } else { gap };
            let (mut x, mut tallest) = (0.0f32, 0.0f32);
            for item in row.iter() {
                let s = f.text_size(item, size);
                if x > 0.0 && x + s.x > room {
                    (x, y) = (0.0, y + tallest + 6.0);
                    tallest = 0.0;
                }
                placed.push((Vec2::new(left + x, y), item));
                widest = widest.max(x + s.x);
                tallest = tallest.max(s.y);
                x += s.x + gap;
            }
            y += tallest + 8.0;
        }
        hud_panel(f, Vec2::new(left - 10.0, top - 8.0), Vec2::new(left + widest + 10.0, y));
        for (at, item) in placed {
            f.text_px(at, item, size, pal::TEXT, Align2::LEFT_TOP);
        }
        // on a plate: yellow on the bright surface is hard to read
        if let Some(p) = on_screen(f, self.handle) {
            f.caption_px(p + Vec2::new(14.0, -12.0), "drag me", 14.0, pal::YELLOW, Align2::LEFT_BOTTOM);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// At the longest step allowed the particles stay on the attractor for good; at
    /// the step that time speed 4 asked for before the cap, they are gone in a moment.
    #[test]
    fn the_flow_keeps_its_particles() {
        let run = |h: f32, steps: usize| {
            let mut pos: Vec<Vec3> = (0..400).map(flow_start).collect();
            for _ in 0..steps {
                pos.iter_mut().for_each(|p| *p = flow_step(*p, h));
            }
            pos.iter().filter(|p| p.is_finite() && p.length() < 50.0).count()
        };
        assert_eq!(run(FLOW_STEP_MAX, 30_000), 400);
        assert!(run(0.03, 400) < 400, "a step of 0.03 was meant to show why there is a cap");
    }

    /// The trails get the same number of rows for a second of scene time at any frame
    /// rate (they used to get three a frame: four times as many at 240 Hz as at 60, and
    /// so a quarter as long); none while paused; and never more than ROWS_MAX a frame.
    #[test]
    fn the_trails_are_stepped_by_time() {
        let rows = |hz: f32, speed: f32| {
            let mut owed = 0.0;
            (0..hz as usize).map(|_| rows_due(&mut owed, speed / hz)).collect::<Vec<_>>()
        };
        for hz in [30.0, 60.0, 144.0, 240.0] {
            let total: usize = rows(hz, 1.0).iter().sum();
            assert!((179..=180).contains(&total), "{hz} Hz: {total} rows in a second");
        }
        assert!(rows(60.0, 0.0).iter().all(|&r| r == 0));
        // time speed 4 at 30 Hz asks for 24 a frame: it takes the most it may
        assert!(rows(30.0, 4.0).iter().all(|&r| r == ROWS_MAX));
        let mut owed = 0.0;
        assert_eq!(rows_due(&mut owed, 10.0), ROWS_MAX);
        assert!(owed < ROW_DT);
    }
}


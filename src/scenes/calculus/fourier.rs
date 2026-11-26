//! Fourier epicycles.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "fourier",
    title: "Fourier epicycles",
    group: "Calculus",
    blurb: "Any unbroken closed curve, as closely as you like, as a sum of rotating circles.",
    make: |g| Box::new(Fourier::new(g)),
};

const FOURIER_M: usize = 512;
const FOURIER_K: i32 = 96;

pub struct Fourier {
    shape: usize,
    terms: usize,
    speed: f32,
    phase: f32,
    pts: Vec<Vec2>,
    coef: Vec<(i32, Vec2)>,
    drawing: bool,
    stroke: Vec<Vec2>,
    show_circles: bool,
    /// The closed curve drawn by the circles in use, and how many that was for.
    curve: Vec<Vec3>,
    curve_terms: usize,
}

/// Points along the curve the circles draw.
const CURVE_M: usize = 720;

/// One of the "things to try": a shape (an index into the shape choice) and how many
/// circles draw it.
struct Trial {
    label: &'static str,
    look_for: &'static str,
    shape: usize,
    terms: usize,
}

const TRIALS: [Trial; 3] = [
    Trial {
        label: "a heart from eight circles",
        look_for: "Four circles give a lumpy heart, 0.31 off the grey outline at worst. Step up to 8: the curve lands on the outline exactly, because this heart's formula is made of eight circles and no more.",
        shape: 1,
        terms: 4,
    },
    Trial {
        label: "a square's corners",
        look_for: "Straight sides and sharp corners need far more circles. Eight leave the corners rounded, 0.057 off; each doubling of the circles about halves that (0.028 at 16, 0.011 at 40), and still a corner is never quite sharp.",
        shape: 0,
        terms: 8,
    },
    Trial {
        label: "two circles make a trefoil",
        look_for: "One circle draws only a circle (80 % of the shape). Step to 2 and the three-looped curve is drawn exactly: every circle after these two has radius zero.",
        shape: 3,
        terms: 1,
    },
];

fn shape_points(shape: usize) -> Vec<Vec2> {
    let m = FOURIER_M;
    let poly = |verts: &[Vec2]| -> Vec<Vec2> { resample(verts, m) };
    match shape {
        0 => poly(&[Vec2::new(-1.6, -1.6), Vec2::new(1.6, -1.6), Vec2::new(1.6, 1.6), Vec2::new(-1.6, 1.6)]),
        1 => (0..m)
            .map(|i| {
                let t = i as f32 / m as f32 * TAU;
                Vec2::new(16.0 * t.sin().powi(3), 13.0 * t.cos() - 5.0 * (2.0 * t).cos() - 2.0 * (3.0 * t).cos() - (4.0 * t).cos()) / 8.0
                    + Vec2::new(0.0, 0.25)
            })
            .collect(),
        2 => {
            let v: Vec<Vec2> = (0..10)
                .map(|i| Vec2::from_angle(i as f32 / 10.0 * TAU + TAU / 4.0) * if i % 2 == 0 { 2.1 } else { 0.85 })
                .collect();
            poly(&v)
        }
        3 => (0..m)
            .map(|i| {
                let t = i as f32 / m as f32 * TAU;
                Vec2::new((2.0 * t).sin() + 2.0 * t.sin(), (2.0 * t).cos() - 2.0 * t.cos()) * 0.72
            })
            .collect(),
        _ => {
            // the letter-like outline of a lightning bolt: sharp corners are the hard case
            poly(&[
                Vec2::new(0.3, 2.2),
                Vec2::new(-1.2, -0.2),
                Vec2::new(-0.1, -0.2),
                Vec2::new(-0.6, -2.2),
                Vec2::new(1.2, 0.5),
                Vec2::new(0.1, 0.5),
                Vec2::new(0.9, 2.2),
            ])
        }
    }
}

/// Resample a closed polyline to `m` points equally spaced along its length.
fn resample(verts: &[Vec2], m: usize) -> Vec<Vec2> {
    let n = verts.len();
    let mut cum = vec![0.0f32; n + 1];
    for i in 0..n {
        cum[i + 1] = cum[i] + (verts[(i + 1) % n] - verts[i]).length();
    }
    let total = cum[n].max(1e-9);
    let mut out = Vec::with_capacity(m);
    let mut seg = 0;
    for k in 0..m {
        let s = total * k as f32 / m as f32;
        while seg + 1 < n && cum[seg + 1] < s {
            seg += 1;
        }
        let t = (s - cum[seg]) / (cum[seg + 1] - cum[seg]).max(1e-9);
        out.push(verts[seg].lerp(verts[(seg + 1) % n], t));
    }
    out
}

fn cmul(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::new(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)
}

impl Fourier {
    pub fn new(_: &Gfx) -> Self {
        Self::create()
    }

    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let mut s = Fourier {
            shape: 1,
            terms: 24,
            speed: 0.12,
            phase: 0.0,
            pts: shape_points(1),
            coef: Vec::new(),
            drawing: false,
            stroke: Vec::new(),
            show_circles: true,
            curve: Vec::new(),
            curve_terms: 0,
        };
        s.transform();
        s
    }

    /// Discrete Fourier transform of the path, strongest terms first.
    fn transform(&mut self) {
        let m = self.pts.len();
        let mut coef = Vec::with_capacity((2 * FOURIER_K + 1) as usize);
        for k in -FOURIER_K..=FOURIER_K {
            let mut acc = Vec2::ZERO;
            for (j, &z) in self.pts.iter().enumerate() {
                let a = -TAU * k as f32 * j as f32 / m as f32;
                acc += cmul(z, Vec2::from_angle(a));
            }
            coef.push((k, acc / m as f32));
        }
        // the constant term first (it is just the centre), then by size
        coef.sort_by(|a, b| (b.0 == 0).cmp(&(a.0 == 0)).then(b.1.length_squared().total_cmp(&a.1.length_squared())));
        self.coef = coef;
        self.curve.clear();
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let t = &TRIALS[k];
        (self.shape, self.terms, self.drawing, self.show_circles) = (t.shape, t.terms, false, true);
        self.stroke.clear();
        self.pts = shape_points(self.shape);
        self.transform();
    }

    fn partial(&self, tau: f32, terms: usize) -> Vec2 {
        let mut p = Vec2::ZERO;
        for &(k, c) in self.coef.iter().take(terms + 1) {
            p += cmul(c, Vec2::from_angle(TAU * k as f32 * tau));
        }
        p
    }
}

impl Scene for Fourier {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 3.1 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.08;
        s.axis_alpha = 0.15;
    }

    /// While drawing, a drag draws (the other buttons still move the picture).
    fn pointer_hint(&self) -> Option<&'static str> {
        self.drawing.then_some("drag: draw a loop    right-drag: move    scroll: zoom")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "Any unbroken closed curve can be drawn, as closely as you like, by a chain of enough circles, each turning at its own whole-number speed while the next rides on its rim: the tip of the last one is the pen. The biggest circles give the rough shape; the small fast ones add the detail.");
        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.label, trial.look_for) {
                self.set_up(k);
            }
        }
        ui::section(ui, "Shape");
        if ui::choice(ui, &["square", "heart", "star", "trefoil", "bolt"], &mut self.shape) {
            self.pts = shape_points(self.shape);
            self.transform();
            self.drawing = false;
        }
        if ui.checkbox(&mut self.drawing, "draw my own (drag on the canvas)").changed() {
            self.stroke.clear();
        }
        ui::note(ui, "Drag one closed loop; when you let go, its circles are worked out.");
        ui::section(ui, "Series");
        ui::slider_int(ui, "circles", &mut self.terms, 1..=(2 * FOURIER_K as usize));
        ui::note(ui, "How many circles draw the curve, the biggest first.");
        ui::slider(ui, "speed", &mut self.speed, 0.0..=0.6);
        ui::note(ui, "Trips of the pen round the shape per second.");
        ui.checkbox(&mut self.show_circles, "show the circles");
        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::WHITE, "the arms, one for each circle"),
                (pal::PURPLE, "the circle each arm sweeps"),
                (pal::YELLOW, "the pen: the tip of the last arm"),
                (pal::CYAN, "the curve these circles draw, bright where the pen has been"),
                (pal::DIM, "grey: the shape itself"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Frequency:", "how many turns a circle makes while the pen goes once round (negative: the other way round).");
        let total: f32 = self.coef.iter().skip(1).map(|c| c.1.length_squared()).sum();
        let kept: f32 = self.coef.iter().skip(1).take(self.terms).map(|c| c.1.length_squared()).sum();
        ui::section(ui, "Numbers");
        // one decimal, and never a "100 %" that hides the circles still left out
        let share = 100.0 * kept / total.max(1e-12);
        let all = self.terms + 1 >= self.coef.len();
        ui::readout(ui, "share of the shape captured", if share > 99.9 && !all { "> 99.9 %".to_string() } else { format!("{share:.1} %") });
        ui::note(ui, "The share is the sum of the squared radii of the circles in use, out of all of them.");
        ui::see_also(ui, &[("calc1", "Taylor polynomials: a curve from powers of $x$ instead"), ("hermite_expand", "a curve built from Hermite polynomials"), ("waves", "waves adding up"), ("dft", "the same idea for a list of numbers: waves as a basis")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        if self.drawing {
            f.handles.capture();
            if f.ptr.down {
                if let Some(m) = f.mouse() {
                    if self.stroke.last().is_none_or(|l| (*l - m).length() > 0.01) {
                        self.stroke.push(m);
                    }
                }
            } else if self.stroke.len() > 8 {
                self.pts = resample(&self.stroke, FOURIER_M);
                self.transform();
                self.stroke.clear();
                self.phase = 0.0;
                // none of the named shapes is on show now, so none is marked as
                // chosen (and a click on any of them, the last one included, brings it back)
                self.shape = usize::MAX;
            } else if !self.stroke.is_empty() {
                self.stroke.clear();
            }
            if !self.stroke.is_empty() {
                f.path(self.stroke.iter().copied(), 2.5, pal::WHITE);
                return;
            }
        }
        self.phase = (self.phase + f.dt * self.speed).fract();
        let terms = self.terms.min(self.coef.len() - 1);

        // the target, faintly, and the series' version of it
        let target: Vec<Vec3> = self.pts.iter().map(|p| p.extend(0.0)).collect();
        f.dl.lines.closed(&target, 1.2, pal::DIM.a(0.55));
        // (the curve the circles draw changes only with the shape and their number:
        // it is worked out then, not every frame)
        let m = CURVE_M;
        if self.curve_terms != terms || self.curve.len() != m {
            self.curve = (0..m).map(|i| self.partial(i as f32 / m as f32, terms).extend(0.0)).collect();
            self.curve_terms = terms;
        }
        let approx = &self.curve;
        f.dl.lines.closed(approx, 1.6, pal::CYAN.a(0.35));
        let upto = (self.phase * m as f32) as usize;
        f.dl.glow_lines.polyline(approx.iter().take(upto + 1).copied().chain([self.partial(self.phase, terms).extend(0.0)]), 3.0, pal::CYAN);

        // the chain of arms
        let mut p = Vec2::ZERO;
        let mut joints = Vec::with_capacity(terms + 2);
        for (i, &(k, c)) in self.coef.iter().take(terms + 1).enumerate() {
            let next = p + cmul(c, Vec2::from_angle(TAU * k as f32 * self.phase));
            if i > 0 {
                let r = c.length();
                if self.show_circles && r > 0.004 {
                    f.circle(p, Vec3::Z, r, 1.0, pal::PURPLE.a((0.25 + r * 0.5).min(0.8)));
                }
            }
            joints.push(p.extend(0.0));
            p = next;
        }
        joints.push(p.extend(0.0));
        // arm by arm: as one path, an arm whose successor folds back on it is not
        // drawn (the join in lines.wgsl cuts it away)
        for arm in joints[1..].windows(2) {
            f.line(arm[0], arm[1], 1.5, pal::WHITE.a(0.9));
        }
        f.glow(p, 26.0, pal::YELLOW);
        f.dot(p, 7.0, pal::YELLOW);

        // what is what, in the corner on a plate
        let mut key = vec![(pal::WHITE, "the arms, one per circle")];
        if self.show_circles {
            key.push((pal::PURPLE, "the circles they turn on"));
        }
        key.extend([(pal::YELLOW, "the pen"), (pal::CYAN, "the curve the circles draw"), (pal::DIM, "the shape")]);
        let size = 12.0;
        let (h, w) = (key.len() as f32 * (size + 6.0), key.iter().map(|e| text_width(e.1, size)).fold(0.0, f32::max) + size + 6.0);
        let at = Vec2::new(18.0, f.view_size().y - 18.0 - h);
        hud_panel(f, at - Vec2::splat(8.0), at + Vec2::new(w, h) + Vec2::splat(4.0));
        hud_key(f, at, &key, size);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The farthest the circles' curve strays from the shape, over the shape's points.
    fn stray(s: &Fourier, terms: usize) -> f32 {
        let m = s.pts.len();
        s.pts.iter().enumerate().map(|(j, p)| (s.partial(j as f32 / m as f32, terms) - *p).length()).fold(0.0, f32::max)
    }

    fn share(s: &Fourier, terms: usize) -> f32 {
        let total: f32 = s.coef.iter().skip(1).map(|c| c.1.length_squared()).sum();
        let kept: f32 = s.coef.iter().skip(1).take(terms).map(|c| c.1.length_squared()).sum();
        100.0 * kept / total
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let mut s = Fourier::create();
        // the heart: lumpy with four circles, exactly on the outline with eight
        s.set_up(0);
        assert!((stray(&s, 4) - 0.31).abs() < 0.01 && stray(&s, 7) > 0.05 && stray(&s, 8) < 1e-4, "{} {}", stray(&s, 4), stray(&s, 8));
        // the square: rounded corners; each doubling about halves the miss, which never reaches zero
        s.set_up(1);
        let misses = [8, 16, 40].map(|n| stray(&s, n));
        for (miss, said) in misses.iter().zip([0.057, 0.028, 0.011]) {
            assert!((miss / said - 1.0).abs() < 0.1, "{misses:?}");
        }
        assert!([4, 8, 16].iter().all(|&n| (1.7..2.3).contains(&(stray(&s, n) / stray(&s, 2 * n)))));
        assert!(stray(&s, 2 * FOURIER_K as usize) > 1e-3);
        // the trefoil: one circle is 80 %, two are all of it
        s.set_up(2);
        assert!((share(&s, 1) - 80.0).abs() < 0.1 && stray(&s, 2) < 1e-4);
        assert!(s.coef.iter().skip(3).all(|c| c.1.length() < 1e-5));
        assert!(s.terms == 1 && s.shape == 3 && !s.drawing);
    }

    /// While a drag draws, the footer says so; otherwise it says what a drag usually does.
    #[test]
    fn the_pointer_hint_follows_the_mode() {
        let mut s = Fourier::create();
        assert_eq!(s.pointer_hint(), None);
        s.drawing = true;
        assert!(s.pointer_hint().is_some_and(|h| h.starts_with("drag: draw")));
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        let mut s = Fourier::create();
        for shape in 0..5 {
            s.pts = shape_points(shape);
            s.transform();
            let size = s.pts.iter().map(|p| p.length()).fold(0.0, f32::max);
            let line: Vec<String> = [1, 2, 3, 4, 6, 8, 12, 16, 24, 40, 60, 100, 150, 192].iter().map(|&n| format!("{n}: {:.3} ({:.2}%)", stray(&s, n), share(&s, n))).collect();
            println!("shape {shape} (size {size:.2}): {}", line.join("  "));
        }
    }
}

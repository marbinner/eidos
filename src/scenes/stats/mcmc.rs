//! MCMC sampling.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "mcmc",
    title: "MCMC sampling",
    group: ESTIMATION,
    blurb: "Metropolis, Hamiltonian and Langevin samplers exploring a density.",
    make: |g| Box::new(Mcmc::new(g)),
};

const MCMC_TARGETS: [(&str, &str); 5] = [
    ("banana", "-x^2/6 - 2 (y - 0.3 x^2 + 1)^2"),
    ("two peaks", "ln(exp(-2 ((x+1.6)^2 + y^2)) + 0.7 exp(-2 ((x-1.6)^2 + (y-0.6)^2)))"),
    ("correlated", "-(x^2 - 1.8 x y + y^2) / 0.5"),
    ("ring", "-6 (sqrt(x^2 + y^2) - 2)^2"),
    ("squiggle", "-(y - sin(2 x))^2 / 0.08 - x^2 / 6"),
];

pub struct Mcmc {
    logp: ExprField,
    method: usize,
    step: f32,
    leaps: usize,
    rate: f32,
    pace: Pace,
    /// How far the newest step is shown to have got (see `Phase`). The walker `x`, the
    /// proposals and the counts are the sampler's own, as its steps leave them, and
    /// every number says those; what is drawn is each step taking the time it has: the
    /// newest proposal reaching out from the walker over it, and the walker, if the
    /// proposal is taken, moving along it (`newest_shown`).
    phase: Phase,
    x: Vec2,
    /// The last `RECENT` places visited (a ring: `tried` says where the next goes).
    recent: Vec<Vec2>,
    proposals: Vec<(Vec<Vec2>, bool, f32)>,
    accepted: u64,
    tried: u64,
    rng: Rng,
    /// The target density behind the walker. (Made on the first frame: nothing else
    /// here needs the graphics device, so the tests can make a scene too.)
    bg: Option<Surface>,
    bg_key: [f32; 4],
    /// The backdrop's grid, and log p over it: kept, so that panning reuses them.
    bg_xs: Vec<f32>,
    bg_ys: Vec<f32>,
    bg_logp: Vec<f32>,
    batch: Batch,
    /// How often each x and each y was visited, over the whole run, in bins fixed
    /// in the plane (so the bars stay under the cloud however the view is moved).
    hist_x: Vec<u32>,
    hist_y: Vec<u32>,
    /// The sum of the places visited, for their mean (in double precision: a long
    /// run adds up millions of them).
    sum: glam::DVec2,
}

/// The histograms cover -HIST_REACH..HIST_REACH in this many bins.
const MCMC_BINS: usize = 128;
const HIST_REACH: f32 = 8.0;
/// How many visited places are drawn: the latest, so the picture stays readable.
const RECENT: usize = 5000;
/// Where every walk starts: off to one side, so that the walk in can be seen.
const START: Vec2 = Vec2::new(-2.5, 2.0);
/// The first steps of a walk, left out of the bars and the mean: they still remember
/// where it began. (From the start above, at the default step size, every sampler
/// reaches the high ground of every target well within this, but for Langevin on the
/// squiggle: see the tests.)
const BURN_IN: u64 = 50;
const VISIT_COLOUR: Color = Color::hex(0xBFE8FF);

/// Experiments: (button, what to look for, target and sampler as indices into
/// `MCMC_TARGETS` and the sampler choice, step size, steps per second).
const TRIALS: [(&str, &str, usize, usize, f32, f32); 4] = [
    ("steps too small", "Step size 0.05: over 90 % of the moves are accepted, but they are tiny. The walker crawls, and the bars fill in slowly.", 0, 0, 0.05, 60.0),
    ("steps too large", "Step size 4: only about 8 % are accepted. The walker stands still for a dozen steps at a time while red refusals pile up, then leaps.", 0, 0, 4.0, 30.0),
    ("one ridge, three samplers", "Metropolis creeps along the narrow ridge. Switch the sampler to Hamiltonian: each step rolls a long way along it, nearly every one is accepted, and the bars fill in many times faster. Langevin lies in between.", 2, 0, 0.5, 30.0),
    ("two peaks", "Metropolis stays on one peak for hundreds of steps before it finds the other. Switch to Hamiltonian: it crosses about five times as often.", 1, 0, 0.5, 200.0),
];
/// The backdrop's grid points across and up.
const BG_N: (usize, usize) = (160, 110);

/// log p at `p`, minus infinity where the formula gives no number.
fn log_density(expr: &Expr, p: Vec2) -> f32 {
    let v = expr.eval(&[p.x, p.y]);
    if v.is_finite() {
        v
    } else {
        f32::NEG_INFINITY
    }
}

/// The gradient of log p at `p`, capped in length; zero where it is not a number.
fn log_gradient(expr: &Expr, p: Vec2) -> Vec2 {
    let (_, gx, gy) = expr.grad2(&[p.x, p.y], 0, 1);
    let g = Vec2::new(gx, gy);
    if g.is_finite() {
        g.clamp_length_max(200.0)
    } else {
        Vec2::ZERO
    }
}

/// One step of sampler `method` (0 Metropolis, 1 Hamiltonian, 2 Langevin) from `cur`:
/// the path of the proposal, and whether it is taken.
fn sampler_step(method: usize, step: f32, leaps: usize, cur: Vec2, expr: &Expr, rng: &mut Rng) -> (Vec<Vec2>, bool) {
    let (lp, grad) = (|p: Vec2| log_density(expr, p), |p: Vec2| log_gradient(expr, p));
    let lp_cur = lp(cur);
    let (path, log_accept) = match method {
        0 => {
            // random-walk Metropolis: propose a nearby point, symmetric in both directions
            let prop = cur + Vec2::new(rng.normal(), rng.normal()) * step;
            (vec![cur, prop], lp(prop) - lp_cur)
        }
        1 => {
            // Hamiltonian Monte Carlo: give the point a random kick and let it roll
            // along the landscape of -log p for a while (leapfrog integration)
            let eps = step * HMC_STEP;
            let p0 = Vec2::new(rng.normal(), rng.normal());
            let (mut q, mut p) = (cur, p0);
            let mut path = vec![q];
            p += grad(q) * (eps * 0.5);
            for i in 0..leaps {
                q += p * eps;
                path.push(q);
                if i + 1 < leaps {
                    p += grad(q) * eps;
                }
            }
            p += grad(q) * (eps * 0.5);
            let h0 = -lp_cur + 0.5 * p0.length_squared();
            let h1 = -lp(q) + 0.5 * p.length_squared();
            (path, h0 - h1)
        }
        _ => {
            // Langevin (MALA): a random step that leans uphill, with the correction
            // for the proposal not being symmetric
            let tau = step * step * 0.5;
            let mean_f = cur + grad(cur) * tau;
            let prop = mean_f + Vec2::new(rng.normal(), rng.normal()) * (2.0 * tau).sqrt();
            let mean_b = prop + grad(prop) * tau;
            let q_fwd = -(prop - mean_f).length_squared() / (4.0 * tau);
            let q_back = -(cur - mean_b).length_squared() / (4.0 * tau);
            (vec![cur, prop], lp(prop) - lp_cur + q_back - q_fwd)
        }
    };
    let end = *path.last().unwrap();
    let accept = if lp_cur == f32::NEG_INFINITY {
        // the walker stands where p is zero (the start, for some formulas), where it
        // should never be: any move is taken until it finds the distribution
        end.is_finite()
    } else {
        log_accept.is_finite() && rng.f32().ln() < log_accept
    };
    (path, accept)
}

/// The Hamiltonian sampler's leapfrog step, as a share of the "step size" control.
const HMC_STEP: f32 = 0.3;

impl Mcmc {
    pub fn new(_: &Gfx) -> Self {
        Self::create()
    }

    fn create() -> Self {
        Mcmc {
            logp: ExprField::new(MCMC_TARGETS[0].1, &["x", "y"]),
            method: 0,
            step: 0.5,
            leaps: 18,
            rate: 12.0,
            pace: Pace::default(),
            phase: Phase::default(),
            x: START,
            recent: Vec::new(),
            proposals: Vec::new(),
            accepted: 0,
            tried: 0,
            rng: Rng::new(8),
            bg: None,
            bg_key: [f32::NAN; 4],
            bg_xs: Vec::new(),
            bg_ys: Vec::new(),
            bg_logp: Vec::new(),
            batch: Batch::default(),
            hist_x: vec![0; MCMC_BINS],
            hist_y: vec![0; MCMC_BINS],
            sum: glam::DVec2::ZERO,
        }
    }

    fn restart(&mut self) {
        self.recent.clear();
        self.proposals.clear();
        self.accepted = 0;
        self.tried = 0;
        self.hist_x.fill(0);
        self.hist_y.fill(0);
        self.sum = glam::DVec2::ZERO;
        self.x = START;
        self.pace.reset();
        self.phase.reset();
    }

    /// Set up one of the experiments in `TRIALS` (from the same random numbers every
    /// time).
    fn set_up(&mut self, k: usize) {
        let (_, _, target, method, step, rate) = TRIALS[k];
        self.logp.set(MCMC_TARGETS[target].1);
        self.bg_key = [f32::NAN; 4];
        (self.method, self.step, self.leaps, self.rate) = (method, step, 18, rate);
        self.rng = Rng::new(8);
        self.restart();
    }

    /// Is entry `i` of `recent` a place visited during the burn-in? (Entry i holds
    /// step i until the ring comes round to it again.)
    fn burn_in_visit(&self, i: usize) -> bool {
        (i as u64) < BURN_IN && self.tried <= (i + RECENT) as u64
    }

    fn step_once(&mut self) {
        let (path, accept) = sampler_step(self.method, self.step, self.leaps, self.x, &self.logp.expr, &mut self.rng);
        if accept {
            self.accepted += 1;
            self.x = *path.last().unwrap();
        }
        if self.proposals.len() >= 48 {
            self.proposals.remove(0);
        }
        self.proposals.push((path, accept, 0.0));
        // where the walker is after this step: one more visit
        if self.recent.len() < RECENT {
            self.recent.push(self.x);
        } else {
            self.recent[(self.tried % RECENT as u64) as usize] = self.x;
        }
        // counted once the burn-in is over
        if self.tried >= BURN_IN {
            let bin = |v: f32| ((v + HIST_REACH) / (2.0 * HIST_REACH) * MCMC_BINS as f32).floor();
            for (hist, b) in [(&mut self.hist_x, bin(self.x.x)), (&mut self.hist_y, bin(self.x.y))] {
                if b >= 0.0 && b < MCMC_BINS as f32 {
                    hist[b as usize] += 1;
                }
            }
            self.sum += self.x.as_dvec2();
        }
        self.tried += 1;
    }

    /// Take the steps that `dt` seconds of the scene's time ask for, within the time a
    /// frame `real_dt` long allows them; returns how many were taken.
    fn advance(&mut self, dt: f32, real_dt: f32) -> usize {
        let mut turn = scene_turn(&mut self.pace, self.rate, dt, real_dt);
        while turn.next() {
            self.step_once();
        }
        let taken = self.pace.end(turn);
        self.phase.after(self.rate, dt, taken);
        taken
    }

    /// The newest proposal as it is drawn, a share `phase` of the way through its
    /// step: its path as far as it has reached, and whether it is taken. (None before
    /// the first step.) An accepted one carries the walker at its tip.
    fn newest_shown(&self) -> Option<(Vec<Vec2>, bool)> {
        let (path, ok, _) = self.proposals.last()?;
        // (at an even pace along the path: a Hamiltonian one is a roll in even steps
        // of time)
        let along = self.phase.share() * (path.len() - 1) as f32;
        let i = (along.floor() as usize).min(path.len().saturating_sub(2));
        let tip = path[i].lerp(path[(i + 1).min(path.len() - 1)], along - i as f32);
        Some((path[..=i].iter().copied().chain([tip]).collect(), *ok))
    }

    /// Where the walker is drawn: at the tip of the newest proposal if that is taken,
    /// else where it stands.
    fn walker_shown(&self) -> Vec2 {
        match self.newest_shown() {
            Some((path, true)) => *path.last().unwrap_or(&self.x),
            _ => self.x,
        }
    }
}

impl Scene for Mcmc {
    fn view(&self) -> View {
        View::Flat { center: Vec2::new(0.0, 0.3), half_height: 3.6 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.08;
        s.axis_alpha = 0.2;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "How do you draw samples from a distribution you can only evaluate? Wander: propose a move, and accept it with a probability that depends on how much more likely the new place is. The places visited end up spread like the target.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Target");
        // the name on a line of its own, so that the box is wide enough for the formula
        ui::words(ui, r"$\log p(x, y) =$", 13.0, pal::DIM);
        if self.logp.ui(ui, "") {
            self.bg_key = [f32::NAN; 4];
            self.restart();
        }
        // the target whose formula is in the box is the one marked as chosen
        let labels: Vec<&str> = MCMC_TARGETS.iter().map(|t| t.0).collect();
        let mut target = MCMC_TARGETS.iter().position(|t| t.1 == self.logp.text).unwrap_or(usize::MAX);
        if ui::choice(ui, &labels, &mut target) {
            self.logp.set(MCMC_TARGETS[target].1);
            self.bg_key = [f32::NAN; 4];
            self.restart();
        }
        ui::note(ui, "Any formula in $x$ and $y$: the logarithm of the density, up to a constant (the sampler never needs the constant).");
        ui::section(ui, "Sampler");
        let mut again = ui::choice(ui, &["Metropolis", "Hamiltonian", "Langevin"], &mut self.method);
        match self.method {
            0 => ui::note(ui, "Proposes a random nearby point. Too small a step and nearly every move is accepted but the walker crawls; too large and nearly every one is refused, and it stands still. On these targets it covers the most ground with a surprisingly long step, one that is accepted only about 5–20\u{a0}% of the time."),
            1 => ui::note(ui, "Gives the walker a random push and lets it roll over the landscape of $-\\log p$ for a while, so a proposal can travel far and still be accepted: at the default step, about 99\u{a0}% of the time on all these targets but the squiggle (about 80\u{a0}% there). (Each leapfrog step is 0.3 of the step size below.)"),
            _ => ui::note(ui, "Leans each random proposal uphill, along the gradient. From a poor start it reaches the high ground within a few steps (on the squiggle, whose slopes are steep, it can take hundreds); once there the lean helps less, and it explores about as fast as Metropolis, give or take a factor of two."),
        }
        again |= ui::slider_log(ui, "step size", &mut self.step, 0.02..=4.0);
        if self.method == 1 {
            again |= ui::slider_int(ui, "leapfrog steps", &mut self.leaps, 2..=60);
        }
        ui::slider_log(ui, "steps per second", &mut self.rate, 1.0..=20000.0);
        if ui.button("restart").clicked() || again {
            self.restart();
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (Cmap::Magma.color(0.45), "the glow: the target density, brightest where $p$ is largest"),
                (pal::WHITE, "white dot: the walker"),
                (pal::GREEN, "green: a move it accepted (for Hamiltonian, the whole path it rolled)"),
                (pal::RED, "red: a proposal it refused"),
                (VISIT_COLOUR, "pale dots: where it has been (the latest 5,000 places)"),
                (pal::ORANGE, "orange dots: the first 50 steps, the burn-in, which the bars and the mean leave out"),
                (pal::CYAN, "cyan bars, along two edges: how often each $x$ and each $y$ was visited"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Target:", r"the distribution the samples should come from: the glow, its density $p(x, y)$ written into the box as $\log p$.");
        ui::term(ui, "Proposal:", "the move a sampler suggests at each step, from where the walker stands. It is accepted (the walker moves there) or refused (it stays), with chances set so that the places visited end up spread like the target.");
        ui::term(ui, "Step size:", "how far a proposal reaches. For Metropolis and Langevin it is the spread (the standard deviation) of the random part of the move, along each axis; for Hamiltonian, each leapfrog step is 0.3 of it.");
        ui::term(ui, "Markov chain:", "a sequence of random places in which the next depends only on the place now, not on the way there. Each sampler here makes one, and run long enough, the share of its steps spent in any region approaches the target's probability of that region.");
        ui::term(ui, "Burn-in:", "the first stretch of a walk, thrown away because it still remembers where the walk began, here $(-2.5, 2)$, rather than the target.");
        ui::term(ui, "Accepted:", "the share of proposals taken. A refused proposal still counts: the walker stays where it is, and that place is visited once more.");

        ui::section(ui, "Numbers");
        ui::readout(ui, "steps", self.tried.to_string());
        let rate = if self.tried > 0 { self.accepted as f32 / self.tried as f32 } else { 0.0 };
        ui::readout(ui, "accepted", format!("{:.1} %", rate * 100.0));
        if self.tried > BURN_IN {
            let m = self.sum / (self.tried - BURN_IN) as f64;
            ui::readout(ui, "mean of the places visited", format!("$({}, {})$", fmt(m.x as f32), fmt(m.y as f32)));
        }
        ui::see_also(ui, &[("clt", "averages of independent draws"), ("gauss2", "the correlated target as a bell"), ("descent", "following the gradient downhill, without the randomness")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let (lo, hi) = f.cam.flat_bounds();
        let size = hi - lo;
        let key = [lo.x, lo.y, hi.x, hi.y];
        if self.bg.is_none() {
            self.bg = Some(Surface::new(f.gfx, BG_N.0 as u32, BG_N.1 as u32));
            self.bg_key = [f32::NAN; 4];
        }
        // the backdrop follows the view, so it is worked out afresh whenever the view
        // moves: over the whole grid at once, which is many times faster than point by point
        if key != self.bg_key {
            self.bg_key = key;
            let (nx, ny) = BG_N;
            self.bg_xs.resize(nx * ny, 0.0);
            self.bg_ys.resize(nx * ny, 0.0);
            self.bg_logp.resize(nx * ny, 0.0);
            for j in 0..ny {
                for i in 0..nx {
                    self.bg_xs[j * nx + i] = lo.x + size.x * i as f32 / (nx - 1) as f32;
                    self.bg_ys[j * nx + i] = lo.y + size.y * j as f32 / (ny - 1) as f32;
                }
            }
            self.logp.expr.eval_batch(&[&self.bg_xs, &self.bg_ys], &mut self.bg_logp, &mut self.batch);
            let peak = self.bg_logp.iter().copied().filter(|l| l.is_finite()).fold(f32::NEG_INFINITY, f32::max);
            if let Some(bg) = &mut self.bg {
                for (k, p) in bg.data_mut().iter_mut().enumerate() {
                    // (where the formula gives no number, p is taken to be zero)
                    let l = self.bg_logp[k];
                    *p = [self.bg_xs[k], self.bg_ys[k], -0.5, if l.is_finite() { (l - peak).exp() } else { 0.0 }];
                }
                let st = &mut bg.style;
                st.lit = 0.0;
                st.cmap = Some(Cmap::Magma);
                st.range = [-0.02, 1.25];
                st.contour = 0.125;
                st.contour_alpha = 0.3;
                st.line_color = Color::hex(0xFFE9C9);
            }
        }
        if let Some(bg) = &mut self.bg {
            f.surface(bg);
        }

        // as many steps as scene time asks for and the frame has time for
        if f.dt > 0.0 {
            self.advance(f.dt, f.real_dt);
        }
        let newest = self.newest_shown();

        // visited points: the most recent ones, so the picture stays readable (the
        // newest not before the walker is drawn there)
        f.dl.discs.style.outline_px = 0.0;
        let ahead = match &newest {
            Some((_, true)) if self.phase.share() < 1.0 => self.tried.checked_sub(1).map(|t| (t % RECENT as u64) as usize),
            _ => None,
        };
        for (i, p) in self.recent.iter().enumerate() {
            if Some(i) == ahead {
                continue;
            }
            let c = if self.burn_in_visit(i) { pal::ORANGE.a(0.8) } else { VISIT_COLOUR.a(0.55) };
            f.dot(*p, 3.6, c);
        }

        // recent proposals: green if taken, red if refused
        for pr in &mut self.proposals {
            pr.2 += f.dt * (0.7 + self.rate * 0.02).min(6.0);
        }
        let count = self.proposals.len();
        for (k, (path, ok, age)) in self.proposals.iter().enumerate() {
            let a = (1.0 - age).clamp(0.0, 1.0);
            if a <= 0.0 {
                continue;
            }
            // (the newest as far as it has reached)
            let path = match &newest {
                Some((reached, _)) if k + 1 == count => reached,
                _ => path,
            };
            let c = if *ok { pal::GREEN } else { pal::RED };
            f.path(path.iter().copied(), if *ok { 2.2 } else { 1.4 }, c.a(a * if *ok { 0.95 } else { 0.7 }));
            if !*ok {
                f.dot(*path.last().unwrap(), 5.0, c.a(a));
            }
        }
        let walker = self.walker_shown();
        f.glow(walker, 30.0, pal::WHITE.a(0.9));
        f.dot(walker, 9.0, pal::WHITE);

        // marginal histograms along the bottom and left edges
        let peak_x = *self.hist_x.iter().max().unwrap_or(&1).max(&1) as f32;
        let peak_y = *self.hist_y.iter().max().unwrap_or(&1).max(&1) as f32;
        // (each bar stands over the stretch of the plane it counts, wherever the view is)
        let bw = 2.0 * HIST_REACH / MCMC_BINS as f32;
        let hx: Vec<f32> = self.hist_x.iter().map(|&c| c as f32 / peak_x * size.y * 0.13).collect();
        bars(f, -HIST_REACH, bw, &hx, lo.y, pal::CYAN);
        for (i, &c) in self.hist_y.iter().enumerate() {
            let w = c as f32 / peak_y * size.y * 0.13;
            if w > 0.0 {
                let y = -HIST_REACH + bw * i as f32;
                let p = [Vec3::new(lo.x, y + bw * 0.06, 0.0), Vec3::new(lo.x + w, y + bw * 0.06, 0.0), Vec3::new(lo.x + w, y + bw * 0.94, 0.0), Vec3::new(lo.x, y + bw * 0.94, 0.0)];
                f.dl.tris_alpha.quad_grad(p, [pal::CYAN.a(0.25), pal::CYAN.a(0.8), pal::CYAN.a(0.8), pal::CYAN.a(0.25)]);
            }
        }
        // name the two rows of bars, each at the end of its own edge (the top one
        // below where the app puts its buttons when the panels are not docked)
        let view = f.view_size();
        let strip = view.y * 0.13;
        f.caption_px(Vec2::new(12.0, f.top_inset + 12.0), "bars: how often each $y$ was visited", 12.0, pal::CYAN, Align2::LEFT_TOP);
        f.caption_px(Vec2::new(view.x - 12.0, view.y - strip - 8.0), "bars: how often each $x$ was visited", 12.0, pal::CYAN, Align2::RIGHT_BOTTOM);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(src: &str) -> Expr {
        Expr::parse(src, &["x", "y"]).unwrap()
    }

    /// A density that is zero where the walker starts (x <= -2.4): every sampler must
    /// leave. Before, a move from p = 0 to p > 0 was refused for having an infinite
    /// ratio, and the walker never moved at all.
    #[test]
    fn a_walker_starting_where_p_is_zero_gets_out() {
        let expr = target("ln(x + 2.4) - x^2 - y^2");
        assert_eq!(log_density(&expr, Vec2::new(-2.5, 2.0)), f32::NEG_INFINITY);
        for method in 0..3 {
            let (mut x, mut rng, mut moved) = (Vec2::new(-2.5, 2.0), Rng::new(8), 0);
            for _ in 0..200 {
                let (path, accept) = sampler_step(method, 0.5, 18, x, &expr, &mut rng);
                if accept {
                    x = *path.last().unwrap();
                    moved += 1;
                }
            }
            assert!(moved > 20 && log_density(&expr, x).is_finite(), "method {method}: {moved} moves, at {x}");
        }
    }

    /// A scene running `method` on target `t` (an index into `MCMC_TARGETS`).
    fn walker(t: usize, method: usize, step: f32, seed: u64) -> Mcmc {
        let mut m = Mcmc::create();
        m.logp.set(MCMC_TARGETS[t].1);
        (m.method, m.step, m.rng) = (method, step, Rng::new(seed));
        m.restart();
        m
    }

    /// Integrated autocorrelation time of a series, by batch means: how many steps
    /// make one independent sample.
    fn act(xs: &[f32]) -> f32 {
        let n = xs.len();
        let mean = xs.iter().sum::<f32>() / n as f32;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
        let b = 1000;
        let batches: Vec<f32> = xs.chunks_exact(b).map(|c| c.iter().sum::<f32>() / b as f32).collect();
        let bvar = batches.iter().map(|m| (m - mean).powi(2)).sum::<f32>() / batches.len() as f32;
        b as f32 * bvar / var
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for t in 0..MCMC_TARGETS.len() {
            for method in 0..3 {
                let walks: Vec<String> = (1..=8).map(|seed| walk_in(t, method, seed).to_string()).collect();
                println!("{:<11} sampler {method}: walk in {}", MCMC_TARGETS[t].0, walks.join(" "));
                for step in [0.05f32, 0.25, 0.5, 1.0, 2.0, 4.0] {
                    let (accepted, steps_per_sample) = run(&mut walker(t, method, step, 1), 100_000);
                    println!("    step {step:<5} {:>5.1} % accepted, {steps_per_sample:>7.1} steps per independent sample", accepted * 100.0);
                }
            }
        }
    }

    /// Steps until the walker first stands where log p is within 3 of its top (inside
    /// the region that holds 95 % of a two-dimensional bell; every target here peaks at
    /// log p = 0, or within a hair of it).
    fn walk_in(t: usize, method: usize, seed: u64) -> usize {
        let mut m = walker(t, method, 0.5, seed);
        let mut k = 0;
        while k < 2000 && log_density(&m.logp.expr, m.x) < -3.0 {
            m.step_once();
            k += 1;
        }
        k
    }

    /// The walk in from the start fits in the burn-in, at the default step size, for
    /// every target and sampler but Langevin on the squiggle; Langevin, as its note
    /// says, gets in within a few steps elsewhere and can take hundreds there.
    #[test]
    fn the_walk_in_fits_in_the_burn_in() {
        for t in 0..MCMC_TARGETS.len() {
            for method in 0..3 {
                let longest = (1..=8).map(|seed| walk_in(t, method, seed)).max().unwrap_or(0);
                let name = MCMC_TARGETS[t].0;
                match (method, name) {
                    (2, "squiggle") => assert!(longest >= 100, "Langevin on the squiggle: {longest} steps at most"),
                    (2, _) => assert!(longest <= 5, "Langevin on the {name}: {longest} steps"),
                    _ => assert!(longest <= 30 && (longest as u64) < BURN_IN, "sampler {method} on the {name}: {longest} steps"),
                }
            }
        }
        // and the burn-in is left out of the counts, but drawn
        let mut m = walker(0, 0, 0.5, 1);
        for _ in 0..BURN_IN + 10 {
            m.step_once();
        }
        assert_eq!(m.hist_x.iter().sum::<u32>(), 10);
        assert!(m.burn_in_visit(0) && m.burn_in_visit(BURN_IN as usize - 1) && !m.burn_in_visit(BURN_IN as usize));
    }

    /// Runs experiment `k` for `n` steps with sampler `method`: the share accepted, the
    /// mean squared jump, steps per independent sample of x and of y (the larger), and
    /// how often x crossed from below -0.8 to above 0.8 or back.
    fn trial(k: usize, method: usize, seed: u64, n: usize) -> (f32, f32, f32, usize) {
        let mut m = Mcmc::create();
        m.set_up(k);
        (m.method, m.rng) = (method, Rng::new(seed));
        m.restart();
        for _ in 0..BURN_IN {
            m.step_once();
        }
        let (a0, t0) = (m.accepted, m.tried);
        let (mut xs, mut ys) = (Vec::with_capacity(n), Vec::with_capacity(n));
        let (mut jump, mut prev, mut crossings, mut right) = (0.0f32, m.x, 0, m.x.x > 0.0);
        for _ in 0..n {
            m.step_once();
            jump += (m.x - prev).length_squared();
            prev = m.x;
            xs.push(m.x.x);
            ys.push(m.x.y);
            if (m.x.x > 0.8 && !right) || (m.x.x < -0.8 && right) {
                right = !right;
                crossings += 1;
            }
        }
        ((m.accepted - a0) as f32 / (m.tried - t0) as f32, jump / n as f32, act(&xs).max(act(&ys)), crossings)
    }

    /// The experiments' captions, held to account (two runs each).
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        for seed in [1u64, 2] {
            // too small: over 90 % accepted, tiny moves
            let (accepted, jump, _, _) = trial(0, 0, seed, 20_000);
            assert!((0.9..=0.97).contains(&accepted) && jump < 0.006, "too small: {accepted} accepted, mean squared jump {jump}");
            // too large: about 8 % accepted, a dozen steps standing still
            let (accepted, _, _, _) = trial(1, 0, seed, 20_000);
            assert!((0.065..=0.095).contains(&accepted) && 1.0 / accepted > 10.5, "too large: {accepted} accepted");
            // one ridge: Hamiltonian many times faster, nearly always accepted; Langevin in between
            let (_, _, metropolis, _) = trial(2, 0, seed, 100_000);
            let (accepted, _, hamiltonian, _) = trial(2, 1, seed, 100_000);
            let (_, _, langevin, _) = trial(2, 2, seed, 100_000);
            assert!(accepted > 0.97 && hamiltonian * 50.0 < metropolis && hamiltonian < langevin && langevin < metropolis, "the ridge: {metropolis} {hamiltonian} {langevin} steps per sample, {accepted} accepted");
            // two peaks: hundreds of steps on one; Hamiltonian crosses about five times as often
            let (_, _, _, metropolis) = trial(3, 0, seed, 100_000);
            let (_, _, _, hamiltonian) = trial(3, 1, seed, 100_000);
            let ratio = hamiltonian as f32 / metropolis as f32;
            assert!(100_000 / metropolis.max(1) > 400 && (4.0..=6.5).contains(&ratio), "two peaks: {metropolis} crossings, {hamiltonian} with Hamiltonian");
        }
    }

    /// The notes on the samplers: Metropolis covers the most ground at about 5–20 %
    /// accepted on every target here; Hamiltonian is accepted about 99 % of the time,
    /// but for the squiggle (80 %); and Langevin explores as fast as Metropolis within
    /// a factor of two. (Slow: several million steps.)
    #[test]
    #[ignore]
    fn the_sampler_notes_hold() {
        for t in 0..MCMC_TARGETS.len() {
            let name = MCMC_TARGETS[t].0;
            let mut best = (f32::MAX, 0.0);
            for step in [0.25f32, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0] {
                let mut m = walker(t, 0, step, 1);
                let (accepted, steps_per_sample) = run(&mut m, 200_000);
                if steps_per_sample < best.0 {
                    best = (steps_per_sample, accepted);
                }
            }
            assert!((0.04..=0.23).contains(&best.1), "{name}: Metropolis is best at {} accepted", best.1);
            let (accepted, _) = run(&mut walker(t, 1, 0.5, 1), 20_000);
            let expected = if name == "squiggle" { 0.7..=0.86 } else { 0.98..=1.0 };
            assert!(expected.contains(&accepted), "{name}: Hamiltonian accepted {accepted}");
            let (_, metropolis) = run(&mut walker(t, 0, 0.5, 1), 200_000);
            let (_, langevin) = run(&mut walker(t, 2, 0.5, 1), 200_000);
            assert!((0.5..=2.2).contains(&(langevin / metropolis)), "{name}: {metropolis} steps per sample with Metropolis, {langevin} with Langevin");
        }
    }

    /// `n` steps after the burn-in: the share accepted and steps per independent sample.
    fn run(m: &mut Mcmc, n: usize) -> (f32, f32) {
        for _ in 0..BURN_IN {
            m.step_once();
        }
        let (a0, t0) = (m.accepted, m.tried);
        let (mut xs, mut ys) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            m.step_once();
            xs.push(m.x.x);
            ys.push(m.x.y);
        }
        ((m.accepted - a0) as f32 / (m.tried - t0) as f32, act(&xs).max(act(&ys)))
    }

    /// "Step size": for Metropolis and Langevin, the standard deviation of the random
    /// part of a proposal along each axis (on a flat target there is no other part);
    /// for Hamiltonian, each leapfrog step is 0.3 of it.
    #[test]
    fn the_step_size_is_what_the_words_say() {
        let flat = target("0 * x");
        for method in [0, 2] {
            for step in [0.1f32, 0.7, 2.0] {
                let (mut rng, n) = (Rng::new(4), 40_000);
                let mut squares = Vec2::ZERO;
                for _ in 0..n {
                    let (path, _) = sampler_step(method, step, 18, Vec2::ZERO, &flat, &mut rng);
                    squares += path[1] * path[1];
                }
                let sd = (squares / n as f32).powf(0.5) / step;
                assert!((sd - Vec2::ONE).abs().max_element() < 0.02, "method {method}, step {step}: spread {sd} steps");
            }
        }
        assert_eq!(HMC_STEP, 0.3);
    }

    /// Each sampler, left to run on a standard Gaussian, visits it in the right
    /// proportions: mean 0 and variance 1 in both directions.
    #[test]
    fn the_samplers_sample_the_target() {
        let expr = target("-(x^2 + y^2) / 2");
        for method in 0..3 {
            let (mut x, mut rng) = (Vec2::new(1.0, -1.0), Rng::new(3));
            let (mut sum, mut squares, mut taken, n) = (Vec2::ZERO, Vec2::ZERO, 0, 40_000);
            for _ in 0..n {
                let (path, accept) = sampler_step(method, 1.0, 12, x, &expr, &mut rng);
                if accept {
                    x = *path.last().unwrap();
                    taken += 1;
                }
                sum += x;
                squares += x * x;
            }
            let (mean, variance) = (sum / n as f32, squares / n as f32);
            assert!(mean.abs().max_element() < 0.08, "method {method}: mean {mean}");
            assert!((variance - Vec2::ONE).abs().max_element() < 0.1, "method {method}: variance {variance}");
            let rate = taken as f32 / n as f32;
            assert!(rate > 0.2 && rate <= 1.0, "method {method}: {rate} accepted");
        }
    }

    /// The walker as drawn takes each step in the time the step has: on displays of
    /// 60 to 144 frames a second, at the rates of the opening and of the experiments
    /// slower than the display, it goes no further in a frame than that frame's share
    /// of the step it is on (it no longer stands still and then leaps), and it is
    /// where the sampler put it by the time the next step is taken. The sampler's own
    /// walk is the same as it would be without the drawing.
    #[test]
    fn the_walker_is_drawn_taking_each_step_in_its_time() {
        for (rate, k) in [(12.0f32, None), (30.0, Some(1)), (30.0, Some(2)), (60.0, Some(0))] {
            for hz in [60.0f32, 75.0, 120.0, 144.0] {
                if rate >= hz {
                    continue;
                }
                let mut m = Mcmc::create();
                if let Some(k) = k {
                    m.set_up(k);
                }
                let mut alone = Mcmc::create();
                if let Some(k) = k {
                    alone.set_up(k);
                }
                let (dt, mut drawn) = (1.0 / hz, m.walker_shown());
                for frame in 0..(3.0 * hz) as usize {
                    let was = m.x;
                    // (frames with time to spare: what a busy machine leaves of the
                    // frame's allowance is not what is tested here)
                    let taken = m.advance(dt, 1.0);
                    let now = m.walker_shown();
                    // (the frame's share of one step, of the longest path it may be on)
                    let longest = m.proposals.iter().rev().take(2).map(|(p, _, _)| p.windows(2).map(|w| (w[1] - w[0]).length()).sum::<f32>()).fold(0.0, f32::max);
                    let share = rate * dt;
                    assert!((now - drawn).length() <= share * longest * 1.0001 + 1e-5, "{rate} a second at {hz} Hz, frame {frame}: went {} of a step of {longest}", (now - drawn).length());
                    if taken > 0 {
                        // the step before the newest is shown whole: the walker is where
                        // that step left it
                        assert!((m.walker_shown() - was).length() <= share * longest * 1.0001 + 1e-5);
                    }
                    drawn = now;
                }
                for _ in 0..m.tried {
                    alone.step_once();
                }
                assert!(alone.x == m.x && alone.accepted == m.accepted, "{rate} a second at {hz} Hz");
            }
        }
    }
}


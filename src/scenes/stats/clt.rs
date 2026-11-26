//! Central limit theorem.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "clt",
    title: "Central limit theorem",
    group: CHANCE,
    blurb: "Average enough values of almost anything, and the averages pile up into a bell curve.",
    make: |g| Box::new(Clt::new(g)),
};

struct Dist {
    name: &'static str,
    mean: f32,
    sd: f32,
    /// Range of values shown in the top panel.
    lo: f32,
    hi: f32,
}

const DISTS: [Dist; 6] = [
    Dist { name: "uniform", mean: 0.5, sd: 0.288_675, lo: -0.25, hi: 1.25 },
    Dist { name: "exponential", mean: 1.0, sd: 1.0, lo: -0.3, hi: 6.0 },
    Dist { name: "two humps", mean: 0.0, sd: 2.061_553, lo: -4.5, hi: 4.5 },
    Dist { name: "coin ($p = 0.2$)", mean: 0.2, sd: 0.4, lo: -0.5, hi: 1.5 },
    Dist { name: "skewed", mean: 3.0, sd: 2.449_49, lo: -0.5, hi: 14.0 },
    Dist { name: "Cauchy", mean: 0.0, sd: 1.0, lo: -8.0, hi: 8.0 },
];

fn dist_pdf(d: usize, x: f32) -> f32 {
    match d {
        0 => {
            if (0.0..=1.0).contains(&x) {
                1.0
            } else {
                0.0
            }
        }
        1 => {
            if x >= 0.0 {
                (-x).exp()
            } else {
                0.0
            }
        }
        2 => 0.5 * normal_pdf(x, -2.0, 0.5) + 0.5 * normal_pdf(x, 2.0, 0.5),
        3 => 0.0,
        4 => {
            // chi-square with 3 degrees of freedom
            if x > 0.0 {
                (x / TAU).sqrt() * (-x * 0.5).exp()
            } else {
                0.0
            }
        }
        _ => 1.0 / (PI * (1.0 + x * x)),
    }
}

fn dist_sample(d: usize, r: &mut Rng) -> f32 {
    match d {
        0 => r.f32(),
        1 => r.exponential(),
        2 => r.normal() * 0.5 + if r.bool(0.5) { 2.0 } else { -2.0 },
        3 => {
            if r.bool(0.2) {
                1.0
            } else {
                0.0
            }
        }
        4 => {
            let (a, b, c) = (r.normal(), r.normal(), r.normal());
            a * a + b * b + c * c
        }
        _ => (PI * (r.f32() - 0.5)).tan(),
    }
}

/// Where the coin and the Cauchy distribution are in `DISTS`.
const COIN: usize = 3;
const CAUCHY: usize = 5;

/// What is true of each source in `DISTS`, shown under the choice (each checked by a
/// test at the end of this file).
const NOTES: [&str; 6] = [
    "Flat and even: at $n = 2$ the bars already make a tent, and from $n = 5$ on they are hard to tell from the bell.",
    "Lopsided, with a long tail to the right. For small $n$ the averages lean the same way: their peak lies left of $0$, their right tail is the longer. Still visibly off the bell at $n = 20$; close to it by $n = 100$.",
    r"Two humps, far apart. Averages of a few values keep humps of their own (at $n = 5$, two tall ones at about $\pm 0.4$ and smaller ones further out, at about $\pm 1.3$ and $\pm 2.2$): by $n = 20$ they have merged into the bell.",
    "Each value is $0$ or $1$, so an average of $n$ can only be one of $n + 1$ values: one bar for each, as wide as the gap between them. At small $n$ there are few bars and they are lopsided; as $n$ grows they narrow and close in on the bell.",
    "Lopsided, like the exponential but a little less: for small $n$ the averages' peak lies left of $0$, and the bars close in on the bell as $n$ grows.",
    "The Cauchy distribution has no mean and no finite spread, so the theorem does not apply: an average of $n$ values is spread exactly as widely as a single value, for every $n$. So the lower plot is not rescaled here, and its curve is the shape of a single value.",
];

/// Experiments: (button, what to look for, source as an index into `DISTS`, n).
const TRIALS: [(&str, &str, usize, usize); 4] = [
    ("lopsided, $n = 5$", "Averages of five exponential values. The bars settle onto a shape that still leans like the source: its peak is left of $0$, at about $-0.4$, and higher than the bell's.", 1, 5),
    ("lopsided, $n = 100$", "The same source, a hundred values to an average: now the bars lie close to the bell.", 1, 100),
    ("two humps, $n = 5$", r"Averages of five values from two humps have humps of their own: two tall ones at about $\pm 0.4$, and smaller ones further out, at about $\pm 1.3$ and $\pm 2.2$. Raise $n$ to $20$: they merge into the bell.", 2, 5),
    ("no mean at all", "Cauchy values, a hundred to an average. The bars settle onto the curve of a single value: averaging has not narrowed anything.", CAUCHY, 100),
];

const CLT_BINS: usize = 81;
const CLT_Z: f32 = 4.5;
/// How fast an average falls from the top plot into the bars: the share of the way
/// it goes in a second of the scene's time.
const FALL: f32 = 2.2;

pub struct Clt {
    dist: usize,
    n: usize,
    rate: f32,
    pace: Pace,
    /// How many averages fell in each bin: `CLT_BINS` bins across -CLT_Z..CLT_Z in
    /// standard units, except for the coin, which has one per possible number of heads.
    bins: Vec<u32>,
    total: u64,
    sum: f64,
    sumsq: f64,
    rng: Rng,
    last: Vec<f32>,
    /// Averages falling into the bars: from, to, and how far they are (0 to 1, eased).
    drops: Vec<(Vec2, Vec2, f32)>,
    /// How far the scene's time has gone towards the next average (see `Phase`): what
    /// sets when the newest of them began to fall.
    phase: Phase,
}

impl Clt {
    pub fn new(_: &Gfx) -> Self {
        Self::create()
    }

    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        Clt {
            dist: 1,
            n: 5,
            rate: 60.0,
            pace: Pace::default(),
            bins: vec![0; CLT_BINS],
            total: 0,
            sum: 0.0,
            sumsq: 0.0,
            rng: Rng::new(42),
            last: Vec::new(),
            drops: Vec::new(),
            phase: Phase::default(),
        }
    }

    /// Set up one of the experiments in `TRIALS`: the fastest rate, so that the bars
    /// settle within seconds.
    fn set_up(&mut self, k: usize) {
        let (_, _, dist, n) = TRIALS[k];
        (self.dist, self.n, self.rate) = (dist, n, 20000.0);
        self.reset();
    }

    fn reset(&mut self) {
        self.bins.clear();
        self.bins.resize(if self.dist == COIN { self.n + 1 } else { CLT_BINS }, 0);
        self.total = 0;
        self.sum = 0.0;
        self.sumsq = 0.0;
        self.drops.clear();
        self.pace.reset();
        self.phase.reset();
    }

    /// Draw n values, average them, and count the average. With `keep` the values are
    /// also kept, to be shown. Returns the average and its value in standard units.
    fn draw(&mut self, keep: bool) -> (f32, f32) {
        if keep {
            self.last.clear();
        }
        let mut s = 0.0;
        for _ in 0..self.n {
            let v = dist_sample(self.dist, &mut self.rng);
            s += v;
            if keep && self.last.len() < 80 {
                self.last.push(v);
            }
        }
        let mean = s / self.n as f32;
        let z = self.standardise(mean);
        self.total += 1;
        self.sum += mean as f64;
        self.sumsq += (mean as f64) * (mean as f64);
        if self.dist == COIN {
            // (the sum is the number of heads, a whole number: its own bar)
            if let Some(c) = self.bins.get_mut(s as usize) {
                *c += 1;
            }
        } else {
            let b = ((z + CLT_Z) / (2.0 * CLT_Z) * CLT_BINS as f32).floor();
            if b >= 0.0 && (b as usize) < CLT_BINS {
                self.bins[b as usize] += 1;
            }
        }
        (mean, z)
    }

    /// Draw the averages that `dt` seconds of the scene's time ask for, within the time
    /// a frame `real_dt` long allows them, and let those falling fall. The first of the
    /// frame's averages is the one shown: its values as the white dots, and its average
    /// falling, from the moment it was due. (Not from the frame it was drawn in: at 60
    /// averages a second on a display of 144 frames a second, one is due every 2.4
    /// frames, so each comes two or three frames after the last, and falling from
    /// their frames they were bunched and spread by turns.)
    fn advance(&mut self, dt: f32, real_dt: f32) {
        let d = &DISTS[self.dist];
        let mut turn = scene_turn(&mut self.pace, self.rate, dt, real_dt);
        let mut first = None;
        while turn.next() {
            let drawn = self.draw(first.is_none());
            first.get_or_insert(drawn);
        }
        let taken = self.pace.end(turn);
        let share = self.phase.after(self.rate, dt, taken);
        if let Some((mean, z)) = first.filter(|(mean, _)| self.drops.len() < 60 && *mean > d.lo && *mean < d.hi) {
            // (the steps owed are counted at the frame: the first of this frame's was
            // due when what was owed reached one)
            let since = ((taken as f32 - 1.0 + share) / self.rate).clamp(0.0, dt);
            self.drops.push((Vec2::new(self.to_top(mean), TOP), Vec2::new(to_bot(z), 0.02), since * FALL));
        }
        for drop in &mut self.drops {
            drop.2 += dt * FALL;
        }
        self.drops.retain(|d| d.2 < 1.0);
    }

    /// Where a value goes across the top plot.
    fn to_top(&self, v: f32) -> f32 {
        let d = &DISTS[self.dist];
        X_LO + (v - d.lo) / (d.hi - d.lo) * (X_HI - X_LO)
    }

    /// The coin's bars: for each number of heads k, where its average lies in standard
    /// units, and the height of a bar as wide as the step from one possible average to
    /// the next (so that its area is the share of averages with k heads).
    fn coin_bars(&self) -> impl Iterator<Item = (f32, f32)> + '_ {
        let step = self.coin_step();
        let total = self.total.max(1) as f32;
        self.bins.iter().enumerate().map(move |(k, &c)| (self.standardise(k as f32 / self.n as f32), c as f32 / (total * step)))
    }

    /// The step between two possible averages of n coins, in standard units: 1/n
    /// divided by σ/√n.
    fn coin_step(&self) -> f32 {
        1.0 / (DISTS[COIN].sd * (self.n as f32).sqrt())
    }

    /// Standardised mean: how many standard errors the sample mean is from the truth.
    fn standardise(&self, mean: f32) -> f32 {
        let d = &DISTS[self.dist];
        if self.dist == CAUCHY {
            mean
        } else {
            (mean - d.mean) / (d.sd / (self.n as f32).sqrt())
        }
    }
}

/// The two plots across (they share no scale), and the top one's axis.
const X_LO: f32 = -3.0;
const X_HI: f32 = 3.0;
const TOP: f32 = 2.0;

/// Where an average in standard units goes across the bars.
fn to_bot(z: f32) -> f32 {
    z / CLT_Z * 3.0
}

impl Scene for Clt {
    fn view(&self) -> View {
        View::Fit { center: Vec2::new(0.0, 1.38), half: Vec2::new(3.25, 1.78) }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "Take $n$ values from a distribution and average them. Do that again and again, and count where the averages land (the bars). The central limit theorem: as $n$ grows, the bars approach one bell curve, whatever the shape the values come from, as long as that shape has a mean and a finite spread.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Draw from");
        let names: Vec<&str> = DISTS.iter().map(|d| d.name).collect();
        let mut again = ui::choice(ui, &names, &mut self.dist);
        ui::note(ui, NOTES[self.dist.min(NOTES.len() - 1)]);
        again |= ui::slider_int_log(ui, "values per average ($n$)", &mut self.n, 1..=200);
        ui::slider_log(ui, "averages per second", &mut self.rate, 1.0..=20000.0);
        ui::note(ui, "The bars are ragged at first. The more averages, the closer they come to the true shape of an average of $n$ values: the bell only once $n$ is large enough, and how large depends on the source.");
        if ui.button("start over").clicked() || again {
            self.reset();
        }

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (pal::PURPLE, "purple, at the top: the shape each value is drawn from"),
                (pal::WHITE, "white dots: the $n$ values just drawn (up to 80 of them)"),
                (pal::YELLOW, "yellow: their average falling into the bars, and the bell curve the bars approach as $n$ grows"),
                (pal::CYAN, "cyan bars: how often the averages landed there, scaled so that their total area is 1, like the curve's"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, r"$\mu$ and $\sigma$:", r"the mean and the spread (standard deviation) of a single value. An average of $n$ values has the same mean and a spread of $\sigma/\sqrt{n}$.");
        ui::term(ui, "Standard units:", r"the lower plot counts $(\text{average} - \mu) / (\sigma/\sqrt{n})$: how many of those spreads an average lies from $\mu$. In these units the bell stays put while $n$ changes.");

        ui::section(ui, "Numbers");
        let d = &DISTS[self.dist];
        ui::readout(ui, "averages so far", self.total.to_string());
        if self.total > 1 && self.dist != CAUCHY {
            let m = self.sum / self.total as f64;
            let var = (self.sumsq / self.total as f64 - m * m).max(0.0);
            ui::readout(ui, "mean of the averages", format!("${}$", fmt(m as f32)));
            ui::readout(ui, r"true mean $\mu$", format!("${}$", fmt(d.mean)));
            ui::readout_colored(ui, "spread of the averages", format!("${}$", fmt(var.sqrt() as f32)), pal::CYAN);
            ui::readout_colored(ui, r"predicted $\sigma/\sqrt{n}$", format!("${}$", fmt(d.sd / (self.n as f32).sqrt())), pal::YELLOW);
        }
        ui::see_also(ui, &[("gauss2", "two normal variables at once"), ("mcmc", "drawing samples from a shape you choose"), ("hermite_orth", "the bell curve as a weight")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // draw new averages, as many as scene time asks for and the frame has time for;
        // the first of them is the one shown, falling
        if f.dt > 0.0 {
            self.advance(f.dt, f.real_dt);
        }
        let d = &DISTS[self.dist];
        let (x_lo, x_hi, top) = (X_LO, X_HI, TOP);
        let to_top = |v: f32| self.to_top(v);

        // top: the source distribution, the latest sample, and its average
        f.line((x_lo, top), (x_hi, top), 1.2, pal::DIM.a(0.7));
        let peak = if self.dist == COIN { 0.8 } else { (0..200).map(|i| dist_pdf(self.dist, d.lo + (d.hi - d.lo) * i as f32 / 199.0)).fold(0.0, f32::max) };
        let ys = 0.72 / peak;
        if self.dist == COIN {
            for (v, p) in [(0.0, 0.8), (1.0, 0.2)] {
                let x = to_top(v);
                f.dl.tris_alpha.rect(Vec2::new(x - 0.12, top), Vec2::new(x + 0.12, top + p * ys), 0.0, pal::PURPLE.a(0.7));
                // below the row of white dots, like the scale of the other distributions
                f.text((x, top - 0.1), format!("{v}"), 12.0, pal::DIM, Align2::CENTER_TOP);
            }
        } else {
            // a scale for the values: the two plots do not share one
            let step = nice_step((d.hi - d.lo) / 7.0);
            let mut v = (d.lo / step).ceil() * step;
            while v <= d.hi + step * 0.01 {
                // (adding zero turns the "-0" of a rounded-up negative start into "0")
                f.text((to_top(v), top - 0.1), fmt_tick(v + 0.0, step), 12.0, pal::DIM, Align2::CENTER_TOP);
                v += step;
            }
            let m = 400;
            let density = (0..=m).map(|i| d.lo + (d.hi - d.lo) * i as f32 / m as f32).map(|v| Vec2::new(to_top(v), top + dist_pdf(self.dist, v) * ys));
            fill_under(f, density, top, pal::PURPLE.a(0.22));
            f.curve(d.lo, d.hi, m, 2.4, pal::PURPLE, |v| Vec3::new(to_top(v), top + dist_pdf(self.dist, v) * ys, 0.0));
        }
        for &v in &self.last {
            if v > d.lo && v < d.hi {
                f.dot((to_top(v), top - 0.045), 7.0, pal::WHITE.a(0.8));
            }
        }
        f.text((x_lo, top + 0.86), format!("each value is drawn from:  {}", d.name), 14.0, pal::PURPLE, Align2::LEFT_BOTTOM);

        // falling averages
        for &(a, b, t) in &self.drops {
            let e = t * t;
            f.glow(a.lerp(b, e), 16.0, pal::YELLOW.a(0.8 * (1.0 - t * 0.5)));
        }

        // bottom: histogram of standardised averages against the predicted curve
        let scale = 1.25 / 0.3989;
        if self.dist == COIN {
            // an average of coins can only be one of n + 1 values: a bar for each, as
            // wide as the gap between them (narrower bins would leave some empty and
            // pile the rest up too high), and only those that fit on the axis whole
            let step = self.coin_step();
            for (z, h) in self.coin_bars() {
                if (z - step * 0.5).abs().max((z + step * 0.5).abs()) <= CLT_Z + 1e-3 {
                    bars(f, to_bot(z - step * 0.5), to_bot(step), &[(h * scale).min(1.72)], 0.0, pal::CYAN);
                }
            }
        } else {
            let binw = 6.0 / CLT_BINS as f32;
            let zw = 2.0 * CLT_Z / CLT_BINS as f32;
            let heights: Vec<f32> = self
                .bins
                .iter()
                .map(|&c| if self.total > 0 { (c as f32 / (self.total as f32 * zw) * scale).min(1.72) } else { 0.0 })
                .collect();
            bars(f, x_lo, binw, &heights, 0.0, pal::CYAN);
        }
        f.line((x_lo, 0.0), (x_hi, 0.0), 1.2, pal::DIM.a(0.7));
        let target = |z: f32| if self.dist == CAUCHY { 1.0 / (PI * (1.0 + z * z)) } else { normal_pdf(z, 0.0, 1.0) };
        f.dl.glow_lines.brk();
        f.glow_path((0..=300).map(|i| -CLT_Z + 2.0 * CLT_Z * i as f32 / 300.0).map(|z| Vec2::new(to_bot(z), target(z) * scale)), 2.6, pal::YELLOW);
        for z in [-4, -3, -2, -1, 0, 1, 2, 3, 4] {
            f.text((to_bot(z as f32), -0.03), format!("{z}"), 12.0, pal::DIM, Align2::CENTER_TOP);
        }
        let curve = if self.dist == CAUCHY { "the curve of a single value" } else { "the bell the bars approach as $n$ grows" };
        f.text((x_hi, 0.45), curve, 13.0, pal::YELLOW, Align2::RIGHT_BOTTOM);
        // on plates: the falling averages pass right through these two
        let label = if self.dist == CAUCHY { "averages of $n$ values  (as spread out as one value)" } else { "averages of $n$ values, in standard units" };
        if let Some(px) = on_screen(f, Vec3::new(x_lo, 1.36, 0.0)) {
            f.caption_px(px, label, 14.0, pal::CYAN, Align2::LEFT_BOTTOM);
        }
        if let Some(px) = on_screen(f, Vec3::new(x_hi, 1.36, 0.0)) {
            f.caption_px(px, format!("$n = {}$", self.n), 14.0, pal::WHITE, Align2::RIGHT_BOTTOM);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh scene drawing from `dist` with `n` values per average, as the panel
    /// sets it up.
    fn scene(dist: usize, n: usize, seed: u64) -> Clt {
        let mut s = Clt::create();
        (s.dist, s.n, s.rng) = (dist, n, Rng::new(seed));
        s.reset();
        s
    }

    /// The bars after `count` averages, as densities in standard units: (centre, height).
    fn bars_after(s: &mut Clt, count: usize) -> Vec<(f32, f32)> {
        for _ in 0..count {
            s.draw(false);
        }
        if s.dist == COIN {
            return s.coin_bars().collect();
        }
        let zw = 2.0 * CLT_Z / CLT_BINS as f32;
        s.bins.iter().enumerate().map(|(i, &c)| (-CLT_Z + zw * (i as f32 + 0.5), c as f32 / (s.total as f32 * zw))).collect()
    }

    /// The largest gap between the bars and the bell, within `reach` of the middle,
    /// and where the tallest bar is.
    fn gap_and_peak(bars: &[(f32, f32)], reach: f32) -> (f32, f32, f32) {
        let gap = bars.iter().filter(|b| b.0.abs() <= reach).map(|&(z, h)| (h - normal_pdf(z, 0.0, 1.0)).abs()).fold(0.0, f32::max);
        let peak = bars.iter().fold((0.0, 0.0), |a, &(z, h)| if h > a.1 { (z, h) } else { a });
        (gap, peak.0, peak.1)
    }

    /// The largest gap between wide bars and the bell, anywhere across each bar within
    /// `reach` of the middle (the coin's bars can be wider than the bell's peak, so
    /// comparing at their centres alone would flatter them).
    fn gap_across(bars: &[(f32, f32)], width: f32, reach: f32) -> f32 {
        let across = |&(z, h): &(f32, f32)| (0..=8).map(move |i| (h - normal_pdf(z + width * (i as f32 / 8.0 - 0.5), 0.0, 1.0)).abs()).fold(0.0, f32::max);
        bars.iter().filter(|b| b.0.abs() <= reach).map(across).fold(0.0, f32::max)
    }

    /// The bar nearest to `z`.
    fn bar_at(bars: &[(f32, f32)], z: f32) -> f32 {
        bars.iter().fold((f32::MAX, 0.0), |a, &(c, h)| if (c - z).abs() < a.0 { ((c - z).abs(), h) } else { a }).1
    }

    /// Averages of five from the two humps: two tall humps of their own at about ±0.43
    /// (a dip at 0 between them), and smaller ones further out on each side, at about
    /// ±1.30 and ±2.17, each with a dip on its inner side. (Exactly: the average is
    /// 2k/5 − 2 plus a little bell, k being how many of the five came from the right
    /// hump, so the humps stand at (4k − 10)/(5 · 0.922) in standard units, with
    /// weights 1, 5, 10, 10, 5, 1 in 32; the bars over them settle near 0.51, 0.25 and
    /// 0.05, the dips between them near 0.21, 0.15 and 0.04.)
    fn assert_humps_at_five(bars: &[(f32, f32)], what: &str) {
        let (hump, middle) = (bar_at(bars, -0.43).min(bar_at(bars, 0.43)), bar_at(bars, 0.0));
        assert!(hump > 0.48 && middle < 0.24, "{what}: {hump} at ±0.43, {middle} at 0");
        for side in [-1.0f32, 1.0] {
            let (second, dip) = (bar_at(bars, 1.3 * side), bar_at(bars, 0.94 * side));
            assert!(second > 0.22 && dip < 0.18, "{what}: {second} at {}, {dip} at {}", 1.3 * side, 0.94 * side);
            let (third, dip) = (bar_at(bars, 2.17 * side), bar_at(bars, 1.9 * side));
            assert!(third > 0.04 && third > dip + 0.004, "{what}: {third} at {}, {dip} at {}", 2.17 * side, 1.9 * side);
        }
    }

    /// The notes on the sources and their names are markup that sets: each is shown
    /// only once its source is chosen.
    #[test]
    fn the_notes_and_names_are_markup_that_sets() {
        for text in NOTES.iter().chain(DISTS.iter().map(|d| &d.name)) {
            let found = crate::typeset::inspect(text);
            assert!(found.error.is_none() && found.words.is_empty(), "{text}: {found:?}");
        }
    }

    /// The notes on the sources, held to account (the exact shapes, worked out on
    /// paper, are within 0.01 of these measurements).
    #[test]
    fn the_sources_close_in_on_the_bell_as_their_notes_say() {
        // flat: a tent at n = 2 (the exact shape is a triangle), the bell from n = 5
        let tent = bars_after(&mut scene(0, 2, 1), 200_000);
        let off_tent = tent.iter().map(|&(z, h)| (h - (1.0 - z.abs() / 6f32.sqrt()).max(0.0) / 6f32.sqrt()).abs()).fold(0.0, f32::max);
        let (bell_gap, _, _) = gap_and_peak(&bars_after(&mut scene(0, 5, 1), 200_000), 3.0);
        assert!(off_tent < 0.012 && bell_gap < 0.02, "uniform: {off_tent} from the tent at n = 2, {bell_gap} from the bell at n = 5");

        // lopsided: the peak left of 0 and a longer right tail at small n; still off
        // the bell at n = 20, close to it at n = 100 (exactly: 0.110, 0.047, 0.019)
        for (dist, small) in [(1, 0.1), (4, 0.075)] {
            let five = bars_after(&mut scene(dist, 5, 2), 200_000);
            let (gap5, at, _) = gap_and_peak(&five, 3.0);
            let tail = |side: f32| five.iter().filter(|b| b.0 * side > 2.0).map(|b| b.1).sum::<f32>();
            assert!(gap5 > small && (-0.6..=-0.15).contains(&at) && tail(1.0) > 10.0 * tail(-1.0), "{}: gap {gap5} at n = 5, peak at {at}, tails {} and {}", DISTS[dist].name, tail(-1.0), tail(1.0));
            let (gap100, _, _) = gap_and_peak(&bars_after(&mut scene(dist, 100, 2), 100_000), 3.0);
            assert!(gap100 < 0.03 && gap5 > 3.0 * gap100, "{}: gap {gap5} at n = 5 against {gap100} at n = 100", DISTS[dist].name);
        }
        let (gap20, _, _) = gap_and_peak(&bars_after(&mut scene(1, 20, 2), 200_000), 3.0);
        assert!(gap20 > 0.04, "exponential at n = 20: {gap20} from the bell");

        // two humps: humps of their own at n = 5, two tall ones near ±0.43 and smaller
        // ones further out; none at n = 20
        assert_humps_at_five(&bars_after(&mut scene(2, 5, 3), 200_000), "two humps at n = 5");
        let (gap20, _, _) = gap_and_peak(&bars_after(&mut scene(2, 20, 3), 200_000), 3.0);
        assert!(gap20 < 0.012, "two humps at n = 20: {gap20} from the bell");

        // the coin: n + 1 bars, narrowing and closing in on the bell
        let mut coin = scene(COIN, 5, 4);
        let few = bars_after(&mut coin, 200_000);
        let gap5 = gap_across(&few, coin.coin_step(), 3.0);
        let mut coin = scene(COIN, 200, 4);
        let many = bars_after(&mut coin, 100_000);
        let gap200 = gap_across(&many, coin.coin_step(), 3.0);
        assert!(few.len() == 6 && gap5 > 0.17 && gap200 < 0.04, "coin: {gap5} from the bell at n = 5, {gap200} at n = 200");

        // Cauchy: the same spread for every n, the shape of a single value
        for n in [1, 100] {
            let mut s = scene(CAUCHY, n, 5);
            let bars = bars_after(&mut s, 100_000);
            let off = bars.iter().filter(|b| b.0.abs() <= 3.0).map(|&(z, h)| (h - 1.0 / (PI * (1.0 + z * z))).abs()).fold(0.0, f32::max);
            // (a single value lies beyond ±4.5 with probability 1 - 2 atan(4.5) / π = 0.137)
            let beyond = 1.0 - s.bins.iter().sum::<u32>() as f32 / s.total as f32;
            assert!(off < 0.015 && (beyond - 0.137).abs() < 0.005, "Cauchy at n = {n}: {off} from one value's curve, {beyond} beyond ±4.5");
        }
    }

    /// The coin's bars have area one between them, and each settles at the chance of
    /// its number of heads divided by its width.
    #[test]
    fn the_coin_bars_are_the_chances_of_each_count() {
        let mut s = scene(COIN, 8, 6);
        let bars = bars_after(&mut s, 100_000);
        let step = s.coin_step();
        assert!((bars.iter().map(|b| b.1 * step).sum::<f32>() - 1.0).abs() < 1e-4);
        let mut chance = 0.8f32.powi(8);
        for (k, &(_, h)) in bars.iter().enumerate() {
            assert!((h * step - chance).abs() < 0.005, "{k} heads: {} against {chance}", h * step);
            chance *= (8 - k) as f32 / (k + 1) as f32 * 0.25;
        }
    }

    /// The bars settle onto the true shape of an average of n values, which is not the
    /// bell: for five exponential values it is a gamma density.
    #[test]
    fn the_bars_settle_onto_the_shape_of_the_average() {
        // the sum of five has density x⁴ e⁻ˣ / 24; in standard units x = 5 + √5 z
        let exact = |z: f32| {
            let x = 5.0 + 5f32.sqrt() * z;
            if x > 0.0 {
                5f32.sqrt() * x.powi(4) * (-x).exp() / 24.0
            } else {
                0.0
            }
        };
        let off = |bars: &[(f32, f32)]| bars.iter().filter(|b| b.0.abs() <= 3.0).map(|&(z, h)| (h - exact(z)).abs()).fold(0.0, f32::max);
        let mut s = scene(1, 5, 7);
        let early = off(&bars_after(&mut s, 2_000));
        let late = off(&bars_after(&mut s, 198_000));
        assert!(early > 0.08 && late < 0.01, "{early} after 2,000 averages, {late} after 200,000");
    }

    /// The experiments' captions, held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let run = |k: usize, count: usize| {
            let mut s = Clt::create();
            s.set_up(k);
            bars_after(&mut s, count)
        };
        // five exponential values: a peak left of 0, near -0.4, higher than the bell's
        let (_, at, peak) = gap_and_peak(&run(0, 200_000), 3.0);
        assert!((-0.6..=-0.2).contains(&at) && peak > 0.42, "n = 5: peak {peak} at {at}");
        // a hundred: close to the bell
        let (gap, _, _) = gap_and_peak(&run(1, 100_000), 3.0);
        assert!(gap < 0.03, "n = 100: {gap} from the bell");
        // two tall humps at ±0.4, smaller ones at ±1.3 and ±2.2 (see above for n = 20)
        assert_humps_at_five(&run(2, 200_000), "the two-humps experiment");
        // Cauchy: on the curve of a single value
        let cauchy = run(3, 100_000);
        let off = cauchy.iter().filter(|b| b.0.abs() <= 3.0).map(|&(z, h)| (h - 1.0 / (PI * (1.0 + z * z))).abs()).fold(0.0, f32::max);
        assert!(off < 0.015, "Cauchy: {off} from one value's curve");
    }

    /// The averages fall from the moment each was due, not from the frame it came in:
    /// at 60 a second, on displays of 60, 75, 120 and 144 frames a second, those falling
    /// are always a sixtieth of a second's fall apart. (Drawn from their frames, they
    /// were one or two frames' fall apart at 75, two or three at 144.) At 60 frames a
    /// second, where every frame has one, they fall exactly as they did.
    #[test]
    fn the_averages_fall_evenly_spaced_at_every_display_rate() {
        for hz in [60.0f32, 75.0, 120.0, 144.0] {
            // (the opening: five exponential values, whose average always lies on the top plot)
            let mut s = Clt::create();
            let dt = 1.0 / hz;
            for frame in 0..(2.0 * hz) as usize {
                // (frames with time to spare: what a busy machine leaves of the frame's
                // allowance is not what is tested here)
                s.advance(dt, 1.0);
                let ages: Vec<f32> = s.drops.iter().map(|d| d.2).collect();
                assert!(ages.len() > 20 || frame < hz as usize / 2, "{hz} Hz, frame {frame}: {} falling", ages.len());
                for pair in ages.windows(2) {
                    assert!((pair[0] - pair[1] - FALL / 60.0).abs() < 1e-4, "{hz} Hz, frame {frame}: {ages:?}");
                }
                if hz == 60.0 {
                    // as before: each a frame further on, the newest already a frame's fall down
                    assert!(ages.iter().rev().enumerate().all(|(k, a)| (a - (k + 1) as f32 * FALL / 60.0).abs() < 1e-4), "{ages:?}");
                }
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for dist in 0..6 {
            for n in [1usize, 2, 3, 5, 10, 20, 50, 100, 200] {
                let mut line = format!("{:<16} n {n:<4}", DISTS[dist].name);
                for seed in [1u64, 2, 3] {
                    let count = if n <= 20 { 400_000 } else { 100_000 };
                    let mut s = scene(dist, n, seed);
                    let bars = bars_after(&mut s, count);
                    let (gap, at, peak) = gap_and_peak(&bars, 3.0);
                    line += &format!("  gap {gap:.3} peak {peak:.3} at {at:+.2}");
                }
                println!("{line}");
            }
        }
    }
}

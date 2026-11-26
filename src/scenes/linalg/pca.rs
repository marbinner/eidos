//! Principal components: the directions a cloud of data spreads along most, and the
//! plane or line that flattens it with the least loss.

use super::*;
use crate::math::linalg::{eig_sym, svd};
use glam::{DMat3, DQuat, DVec3};

pub const DEF: SceneDef = SceneDef {
    id: "pca",
    title: "Principal components",
    group: "Linear algebra: data and dimensions",
    blurb: "The directions a cloud of data spreads along most, and the best way to flatten it.",
    make: |_| Box::new(Principal::create()),
};

/// Points in every cloud.
const N: usize = 300;
/// The draw of random numbers the scene opens with, and every experiment uses.
const SEED: u64 = 7;
/// Where the clouds are made: their middle, before the mean is moved to 0.
const OFFSET: DVec3 = DVec3::new(1.5, 1.0, 1.0);

#[derive(Clone, Copy, Debug, PartialEq)]
enum Cloud {
    Pancake,
    Cigar,
    Ball,
    Clusters,
    Far,
}

const CLOUDS: [(Cloud, &str); 5] = [(Cloud::Pancake, "pancake"), (Cloud::Cigar, "cigar"), (Cloud::Ball, "ball"), (Cloud::Clusters, "two clusters"), (Cloud::Far, "pancake and a far point")];

/// The spreads a cloud is made with along its own three directions, widest first (for
/// the two clusters, the spread of each about its middle).
fn spreads(cloud: Cloud) -> [f64; 3] {
    match cloud {
        Cloud::Pancake | Cloud::Far => [2.0, 1.2, 0.25],
        Cloud::Cigar => [2.0, 0.4, 0.35],
        Cloud::Ball => [1.1; 3],
        Cloud::Clusters => [0.55; 3],
    }
}

/// How far each cluster's middle is from the middle of both.
const GAP: f64 = 1.9;

/// The clouds' own directions in space, turned so that none lies along x, y or z.
fn turn3() -> DMat3 {
    DMat3::from_rotation_z(0.95) * DMat3::from_rotation_y(-0.45) * DMat3::from_rotation_x(0.45)
}

/// The same in the plane: the widest direction 35° up from x.
const PLANE_TURN: f64 = 0.61;

fn turn(plane: bool) -> DMat3 {
    if plane {
        DMat3::from_rotation_z(PLANE_TURN)
    } else {
        turn3()
    }
}

fn offset(plane: bool) -> DVec3 {
    if plane {
        DVec3::new(OFFSET.x, OFFSET.y, 0.0)
    } else {
        OFFSET
    }
}

/// Where the far point starts, in the pancake's own directions: a little along its
/// second direction and far out along its thinnest.
const FAR_OWN: DVec3 = DVec3::new(0.5, 7.0, 11.5);
const FAR_OWN_PLANE: DVec3 = DVec3::new(-2.0, 8.0, 0.0);

fn far_start(plane: bool) -> DVec3 {
    offset(plane) + turn(plane) * if plane { FAR_OWN_PLANE } else { FAR_OWN }
}

/// How far from the cloud's middle the far point may be dragged.
const FAR_REACH: f64 = 16.0;

/// Three standard normal numbers per point, from the seed. Every cloud is made from the
/// same draws, so that a change of cloud moves each point to its new place.
fn draws(seed: u64) -> Vec<DVec3> {
    let mut rng = Rng::new(seed);
    (0..N).map(|_| DVec3::new(rng.normal() as f64, rng.normal() as f64, rng.normal() as f64)).collect()
}

/// The points of a cloud, in its own units (before x is measured in others). In the
/// plane they have no z.
fn make(cloud: Cloud, z: &[DVec3], plane: bool, far: DVec3) -> Vec<DVec3> {
    let s = spreads(cloud);
    let (turn, offset) = (turn(plane), offset(plane));
    let mut points: Vec<DVec3> = z
        .iter()
        .enumerate()
        .map(|(i, z)| {
            let own = match cloud {
                Cloud::Clusters => DVec3::new(if i % 2 == 0 { GAP } else { -GAP }, 0.0, 0.0) + *z * s[0],
                _ => DVec3::new(z.x * s[0], z.y * s[1], z.z * s[2]),
            };
            offset + turn * if plane { DVec3::new(own.x, own.y, 0.0) } else { own }
        })
        .collect();
    if cloud == Cloud::Far {
        if let Some(last) = points.last_mut() {
            *last = far;
        }
    }
    points
}

/// x measured in other units: every x times `units`.
fn in_units(p: DVec3, units: f64) -> DVec3 {
    DVec3::new(p.x * units, p.y, p.z)
}

/// What a cloud's principal components are.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Stats {
    n: usize,
    mean: DVec3,
    /// The covariance matrix, row by row: the sum of (p − mean)(p − mean)ᵀ over the
    /// points, divided by n − 1.
    cov: [[f64; 3]; 3],
    /// The variances along the components, largest first: the eigenvalues of `cov`.
    var: [f64; 3],
    /// The components: unit eigenvectors of `cov`, as a right-handed frame, PC1 first.
    axes: [DVec3; 3],
}

impl Stats {
    /// The statistics of `points`. In the plane only x and y count, and PC3 is z. The
    /// sign of an eigenvector means nothing, so each is turned the way that keeps the
    /// frame nearest to `like` (the one drawn so far): the axes do not flip over when
    /// the data change a little.
    fn of(points: &[DVec3], plane: bool, like: [DVec3; 3]) -> Stats {
        let n = points.len();
        let mean = points.iter().copied().sum::<DVec3>() / n.max(1) as f64;
        let mut cov = [[0.0; 3]; 3];
        for p in points {
            let d = (*p - mean).to_array();
            for i in 0..3 {
                for j in 0..3 {
                    cov[i][j] += d[i] * d[j];
                }
            }
        }
        let div = (n.max(2) - 1) as f64;
        cov.iter_mut().flatten().for_each(|c| *c /= div);
        let (var, e1, e2) = if plane {
            let (values, vectors) = eig_sym(&[cov[0][0], cov[0][1], cov[1][0], cov[1][1]], 2);
            ([values[0], values[1], 0.0], DVec3::new(vectors[0], vectors[1], 0.0), DVec3::new(vectors[2], vectors[3], 0.0))
        } else {
            let flat: Vec<f64> = cov.iter().flatten().copied().collect();
            let (values, vectors) = eig_sym(&flat, 3);
            ([values[0], values[1], values[2]], DVec3::from_slice(&vectors[0..3]), DVec3::from_slice(&vectors[3..6]))
        };
        // (rounding leaves a variance of nothing a hair either side of 0)
        let top = var[0].abs();
        let var = var.map(|v| if v <= 1e-12 * top || !v.is_finite() { 0.0 } else { v });
        Stats { n, mean, cov, var, axes: orient(e1, e2, like) }
    }

    fn total(&self) -> f64 {
        self.var.iter().sum()
    }

    /// Is there any spread at all? (Not if every point is in the same place.)
    fn spread_out(&self) -> bool {
        self.total() > 1e-12 * (1.0 + self.mean.length_squared())
    }

    /// Each component's share of the variance (all 0 when there is no spread at all).
    fn shares(&self) -> [f64; 3] {
        if self.spread_out() {
            self.var.map(|v| v / self.total())
        } else {
            [0.0; 3]
        }
    }

    /// The share kept by the first `k` components.
    fn kept(&self, k: usize) -> f64 {
        self.shares().iter().take(k).sum()
    }

    /// The sum of the squared distances from the points to the plane (or line) of the
    /// first `k` components through the mean: (n − 1) times the variances left out.
    fn lost(&self, k: usize) -> f64 {
        (self.n.max(2) - 1) as f64 * self.var.iter().skip(k).sum::<f64>()
    }

    /// The sum of the squared distances of the points from the mean.
    fn whole(&self) -> f64 {
        (self.n.max(2) - 1) as f64 * self.total()
    }

    fn frame(&self) -> DQuat {
        DQuat::from_mat3(&DMat3::from_cols(self.axes[0], self.axes[1], self.axes[2]))
    }
}

/// A right-handed frame from the first two eigenvectors, each turned the way (of the
/// two) that keeps the frame nearest to `like`.
fn orient(e1: DVec3, e2: DVec3, like: [DVec3; 3]) -> [DVec3; 3] {
    let e1 = e1.try_normalize().unwrap_or(DVec3::X);
    // (exactly at right angles to the first, whatever rounding left)
    let e2 = (e2 - e1 * e1.dot(e2)).try_normalize().unwrap_or_else(|| e1.any_orthonormal_vector());
    let mut best = ([e1, e2, e1.cross(e2)], f64::MIN);
    for (s1, s2) in [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)] {
        let (v1, v2) = (e1 * s1, e2 * s2);
        let v3 = v1.cross(v2);
        let near = v1.dot(like[0]) + v2.dot(like[1]) + v3.dot(like[2]);
        if near > best.1 + 1e-12 {
            best = ([v1, v2, v3], near);
        }
    }
    best.0
}

/// The right singular vectors and singular values of the centred data (one row per
/// point): the same components, and σ² / (n − 1) the same variances.
fn singular(points: &[DVec3], mean: DVec3) -> (Vec<f64>, [DVec3; 3]) {
    let x: Vec<f64> = points.iter().flat_map(|p| (*p - mean).to_array()).collect();
    if points.is_empty() {
        return (vec![0.0; 3], [DVec3::X, DVec3::Y, DVec3::Z]);
    }
    let (_, s, v) = svd(&x, points.len(), 3);
    let col = |k: usize| DVec3::new(v[k], v[3 + k], v[6 + k]);
    (s, [col(0), col(1), col(2)])
}

/// A line through `mean` in the plane, along `dir` (a unit vector): the sum of the
/// squared misses measured at right angles to it, and measured up and down (none if
/// the line stands upright, where a miss up and down never reaches it).
fn misses(points: &[DVec3], mean: DVec3, dir: DVec3) -> (f64, Option<f64>) {
    let across = DVec3::new(-dir.y, dir.x, 0.0);
    let square = |v: f64| v * v;
    let right = points.iter().map(|p| square((*p - mean).dot(across))).sum();
    let upright = dir.x.abs() < 1e-9;
    let up = (!upright).then(|| points.iter().map(|p| square((p.y - mean.y) - dir.y / dir.x * (p.x - mean.x))).sum());
    (right, up)
}

/// The regression line of y on x, as a unit direction (none if every x is the same).
fn regression(s: &Stats) -> Option<DVec3> {
    let (cxx, cxy) = (s.cov[0][0], s.cov[0][1]);
    (cxx > 1e-12 * s.total().max(1e-300)).then(|| DVec3::new(1.0, cxy / cxx, 0.0).normalize())
}

/// The angle of a line in the plane from the x axis, in degrees from 0 up to 180.
fn angle_of(dir: DVec3) -> f64 {
    let a = dir.y.atan2(dir.x).to_degrees();
    a.rem_euclid(180.0)
}

/// The sum of the squared right-angle misses of the line through the mean at `degrees`.
fn misses_at(s: &Stats, degrees: f64) -> f64 {
    let (sn, cs) = degrees.to_radians().sin_cos();
    let c = s.cov;
    // across = (−sin, cos): acrossᵀ C across, times n − 1
    (s.n.max(2) - 1) as f64 * (sn * sn * c[0][0] - 2.0 * sn * cs * c[0][1] + cs * cs * c[1][1])
}

// ----- the scene ----------------------------------------------------------------------

/// Experiments: (button, what to look for).
const TRIALS: [(&str, &str); 5] = [
    ("flatten a pancake", "A cloud spread mostly in two directions. Each point drops straight onto the blue plane of PC1 and PC2, along its pink miss. The misses are short: the two components keep 98.7% of the variance, and the misses lose only 1.3%. The points in the PCA plot do not move: that plot is the flattened cloud, laid flat."),
    ("a ball", "Spread about as much in every direction: each component's share is near a third (37%, 33%, 30%), and the axes point wherever chance has put them. Press \"new points\" for another draw of the same ball: PC1 turns by 57°. (Another draw of the pancake turns its PC1 by 5°.)"),
    ("change the units", "The pancake again, with $x$ measured in units three times smaller, so that every $x$ is three times larger. Only the units have changed, yet PC1 swings round by 50°, from 61° off the $x$ axis to 11°. Principal components depend on the units: that is why each coordinate is often first divided by its own spread."),
    ("one far point", "The pancake, and one point (red) far off it. Misses count by their squares, so the best plane tilts towards that one point: by 12° from the plane of the other 299 (dashed), and the share of the variance lost grows from 1.3% to 7.4%. Even so, the red point's miss, squared, is more than twice all the others' put together. Drag it."),
    ("not the regression line", "In the plane. The regression line (cyan) is the line for predicting $y$ from $x$: it has the least sum of squared misses measured up and down. PC1 (orange) has the least measured at right angles, and is 17° steeper. Each line wins at its own kind of miss (the table), and the chart shows that no line through the mean has fewer right-angle misses than PC1."),
];

/// How long the changes take, in seconds of the scene's time.
const MORPH_TIME: f32 = 1.0;
const UNITS_TIME: f32 = 1.8;
const FLAT_TIME: f32 = 1.8;
const KEEP_TIME: f32 = 0.6;
const CENTRE_TIME: f32 = 0.9;
/// How quickly the drawn axes follow the computed ones: the time in which they cover
/// about two thirds of the way, in seconds.
const FOLLOW: f64 = 0.08;

/// Colours: the three components, what is kept, the misses, the points, the far
/// point, the regression line.
const PC: [Color; 3] = [pal::ORANGE, pal::PURPLE, pal::GREEN];
const KEPT: Color = pal::BLUE;
const MISS: Color = pal::PINK;
const DOT: Color = Color::hex(0xB9C6DC);
const FAR: Color = pal::RED;
const REG: Color = pal::CYAN;
const NAMES: [&str; 3] = ["PC1", "PC2", "PC3"];

pub struct Principal {
    cloud: Cloud,
    seed: u64,
    plane: bool,
    /// How many components are kept.
    keep: usize,
    /// The units of x: every x is multiplied by this.
    units: f32,
    /// A change of units being played: from, to, and how far (1 at the end; below 0
    /// while it waits for the cloud to arrive).
    units_change: Option<(f32, f32, f32)>,
    centred: bool,
    /// How far the mean has been moved towards 0 (0 to 1).
    centring: f32,
    /// How far the points are on their way to their shadows (0 to 1), and where they
    /// are going while that is being played.
    flat: f32,
    playing: Option<f32>,
    /// Seconds to wait before playing (while a new cloud arrives).
    wait: f32,
    /// How much of each component is kept, as drawn: 1 for the kept ones and 0 for the
    /// rest, in between while the number kept changes.
    kept: [f32; 3],
    /// The far point, in the cloud's own units.
    far: DVec3,
    draws: Vec<DVec3>,
    /// The points the cloud is changing from (in its own units), and how far (1: done).
    from: Vec<DVec3>,
    morph: f32,
    /// The points as they are now, in the cloud's own units and in the units of x.
    raw: Vec<DVec3>,
    data: Vec<DVec3>,
    stats: Stats,
    /// The frame and the spreads drawn: they follow the computed ones.
    frame: DQuat,
    spread: [f64; 3],
    /// The point whose two misses are picked out in the plane.
    picked: usize,
    /// The canvas last drawn on and how much of its top the application's buttons
    /// cover: they decide where the panels go, and with them where the picture goes.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
}

impl Principal {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let mut s = Principal {
            cloud: Cloud::Pancake,
            seed: SEED,
            plane: false,
            keep: 2,
            units: 1.0,
            units_change: None,
            centred: false,
            centring: 0.0,
            flat: 0.0,
            playing: None,
            wait: 0.0,
            kept: [1.0, 1.0, 0.0],
            far: far_start(false),
            draws: draws(SEED),
            from: Vec::new(),
            morph: 1.0,
            raw: Vec::new(),
            data: Vec::new(),
            stats: Stats::of(&[], false, [DVec3::X, DVec3::Y, DVec3::Z]),
            frame: DQuat::IDENTITY,
            spread: [0.0; 3],
            picked: 0,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
        };
        s.refresh();
        s.catch_up();
        s.pick();
        s
    }

    /// The units of x as drawn now (on their way, while a change is played).
    fn units_now(&self) -> f32 {
        match self.units_change {
            Some((a, b, t)) => a + (b - a) * smoothstep(t.max(0.0)),
            None => self.units,
        }
    }

    /// The points as the cloud and the far point are now, before any change of units.
    fn target(&self) -> Vec<DVec3> {
        make(self.cloud, &self.draws, self.plane, self.far)
    }

    /// Work out the points as they are now, and their statistics.
    fn refresh(&mut self) {
        let target = self.target();
        let t = smoothstep(self.morph) as f64;
        self.raw = if self.morph < 1.0 && self.from.len() == target.len() { self.from.iter().zip(&target).map(|(a, b)| a.lerp(*b, t)).collect() } else { target };
        let u = self.units_now() as f64;
        self.data = self.raw.iter().map(|p| in_units(*p, u)).collect();
        self.stats = Stats::of(&self.data, self.plane, self.axes());
    }

    /// The axes as drawn.
    fn axes(&self) -> [DVec3; 3] {
        [self.frame * DVec3::X, self.frame * DVec3::Y, self.frame * DVec3::Z]
    }

    /// Draw the axes and spreads exactly as computed.
    fn catch_up(&mut self) {
        self.frame = self.stats.frame();
        self.spread = self.stats.var.map(f64::sqrt);
    }

    fn caught_up(&self) -> bool {
        self.frame == self.stats.frame() && self.spread == self.stats.var.map(f64::sqrt)
    }

    /// Bring the drawn axes and spreads `dt` seconds nearer to the computed ones.
    fn follow(&mut self, dt: f32) {
        let goal = self.stats.frame();
        let spread = self.stats.var.map(f64::sqrt);
        let k = 1.0 - (-(dt.max(0.0) as f64) / FOLLOW).exp();
        if k > 0.0 {
            self.frame = self.frame.slerp(goal, k).normalize();
            for (s, g) in self.spread.iter_mut().zip(spread) {
                *s += (g - *s) * k;
            }
        }
        let size = spread[0].max(1e-9);
        // (exactly there in the end, not nearly: the picture then stops changing. A
        // turn of a hundred-thousandth of a radian or less: the cosine of half of it
        // is within a hundred-billionth of 1, and may round to just above 1)
        if !self.frame.is_finite() || self.frame.dot(goal).abs() >= 1.0 - 1e-11 {
            self.frame = goal;
        }
        if !self.spread.iter().all(|s| s.is_finite()) || self.spread.iter().zip(spread).all(|(s, g)| (s - g).abs() < 1e-5 * size) {
            self.spread = spread;
        }
    }

    /// Change the cloud (or the draw, or between space and plane): the points glide to
    /// their new places, but between space and plane, where the view changes, they
    /// are put there at once.
    fn set_cloud(&mut self, cloud: Cloud, seed: u64, plane: bool) {
        self.from = self.raw.clone();
        if (cloud, plane) != (self.cloud, self.plane) || seed != self.seed {
            self.far = far_start(plane);
        }
        let glide = plane == self.plane;
        (self.cloud, self.plane) = (cloud, plane);
        if seed != self.seed {
            self.seed = seed;
            self.draws = draws(seed);
        }
        // (no glide where nothing is to move)
        let same = self.target().iter().zip(&self.from).all(|(a, b)| a == b) && self.from.len() == N;
        self.morph = if glide && !same { 0.0 } else { 1.0 };
        self.refresh();
        if !glide {
            self.catch_up();
        }
        self.pick();
        self.new_view = true;
    }

    /// The point whose two misses are picked out in the plane: one well off PC1's line
    /// and some way along it, where the two kinds of miss differ plainly (not the far
    /// point, which has its own colour). Chosen when the cloud changes, so that it
    /// stays the same point while anything is dragged.
    fn pick(&mut self) {
        let target: Vec<DVec3> = self.target().iter().map(|p| in_units(*p, self.units as f64)).collect();
        let s = Stats::of(&target, true, [DVec3::X, DVec3::Y, DVec3::Z]);
        let last = if self.cloud == Cloud::Far { N - 1 } else { N };
        let sd = s.var.map(|v| v.sqrt().max(1e-9));
        let off = |p: DVec3| {
            let c = p - s.mean;
            // (the side of PC1 that points the same way as x, whichever way PC1 was given)
            let along = c.dot(s.axes[0]) * s.axes[0].x.signum();
            (along / sd[0] + 1.2).powi(2) + (c.dot(s.axes[1]).abs() / sd[1] - 1.7).powi(2)
        };
        self.picked = (0..last.min(target.len())).min_by(|&a, &b| off(target[a]).total_cmp(&off(target[b]))).unwrap_or(0);
    }

    /// Set the units of x at once (a slider being dragged).
    fn set_units(&mut self, units: f32) {
        self.units = units.max(0.0).min(10.0);
        self.units_change = None;
        self.refresh();
        self.new_view = true;
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let (cloud, plane, keep, play) = match k {
            0 => (Cloud::Pancake, false, 2, true),
            1 => (Cloud::Ball, false, 2, false),
            2 => (Cloud::Pancake, false, 2, false),
            3 => (Cloud::Far, false, 2, false),
            _ => (Cloud::Pancake, true, 2, false),
        };
        // (every setting the lesson depends on, whatever was set before; the points
        // glide from where they are drawn, in whatever units, to the lesson's cloud)
        if self.units_now() != 1.0 {
            self.raw = self.data.clone();
        }
        (self.units, self.units_change, self.centred) = (1.0, None, false);
        self.far = far_start(plane);
        self.set_cloud(cloud, SEED, plane);
        self.set_keep(keep);
        (self.flat, self.playing) = (0.0, play.then_some(1.0));
        self.wait = if self.morph < 1.0 { MORPH_TIME } else { 0.3 };
        if k == 2 {
            // played from the units it had to the new ones, once the cloud is there
            self.units = 3.0;
            self.units_change = Some((1.0, 3.0, -self.wait / UNITS_TIME));
        }
        self.new_view = true;
    }

    fn set_keep(&mut self, keep: usize) {
        self.keep = keep.max(1).min(3);
    }

    /// Play the flattening, or play it back.
    fn play(&mut self) {
        let goal = if self.flat >= 1.0 || self.playing == Some(1.0) { 0.0 } else { 1.0 };
        (self.playing, self.wait) = (Some(goal), 0.0);
    }

    fn moving(&self) -> bool {
        self.morph < 1.0
            || self.units_change.is_some()
            || self.playing.is_some()
            || self.centring != if self.centred { 1.0 } else { 0.0 }
            || (0..3).any(|j| self.kept[j] != if j < self.keep { 1.0 } else { 0.0 })
            || !self.caught_up()
    }

    /// Move everything that plays on by `dt` seconds of the scene's time.
    fn advance(&mut self, dt: f32) {
        let dt = dt.max(0.0);
        let mut changed = false;
        if self.morph < 1.0 {
            self.morph = (self.morph + dt / MORPH_TIME).min(1.0);
            changed = true;
        }
        let waited = dt.min(self.wait);
        self.wait -= waited;
        let left = dt - waited;
        if let Some((a, b, t)) = self.units_change {
            let t = t + dt / UNITS_TIME;
            self.units_change = (t < 1.0).then_some((a, b, t));
            changed = true;
        }
        if let Some(goal) = self.playing {
            let step = left / FLAT_TIME;
            self.flat = if goal > self.flat { (self.flat + step).min(goal) } else { (self.flat - step).max(goal) };
            if self.flat == goal {
                self.playing = None;
            }
        }
        let centre = if self.centred { 1.0 } else { 0.0 };
        if self.centring != centre {
            let step = dt / CENTRE_TIME;
            self.centring = if centre > self.centring { (self.centring + step).min(1.0) } else { (self.centring - step).max(0.0) };
        }
        for j in 0..3 {
            let goal = if j < self.keep { 1.0 } else { 0.0 };
            let step = dt / KEEP_TIME;
            self.kept[j] = if goal > self.kept[j] { (self.kept[j] + step).min(goal) } else { (self.kept[j] - step).max(goal) };
        }
        if changed {
            self.refresh();
        }
    }

    /// How far the drawing is moved so as to bring the mean towards 0.
    fn shift(&self) -> DVec3 {
        self.stats.mean * smoothstep(self.centring) as f64
    }

    /// The share of each component kept as drawn, for the points as they are on their way
    /// to their shadows: the coordinate along component j is multiplied by this.
    fn factors(&self) -> [f64; 3] {
        let t = if self.plane { 0.0 } else { smoothstep(self.flat) as f64 };
        self.kept.map(|w| 1.0 - t * (1.0 - smoothstep(w) as f64))
    }

    /// Each point as drawn, its shadow on what is kept (as drawn), and where it was
    /// before the flattening began.
    fn drawn(&self) -> Vec<(DVec3, DVec3, DVec3)> {
        let (m, shift, axes) = (self.stats.mean, self.shift(), self.axes());
        let f = self.factors();
        let w = self.kept.map(|w| smoothstep(w) as f64);
        self.data
            .iter()
            .map(|p| {
                let c = [0, 1, 2].map(|j| (*p - m).dot(axes[j]));
                let at = *p - shift;
                let now = at + (0..3).map(|j| axes[j] * (c[j] * (f[j] - 1.0))).sum::<DVec3>();
                let shadow = at - (0..3).map(|j| axes[j] * (c[j] * (1.0 - w[j]))).sum::<DVec3>();
                (now, shadow, at)
            })
            .collect()
    }

    /// What the camera should take in: the points where they are going, as they will be
    /// drawn, and the corners of the plane of PC1 and PC2 drawn through them.
    fn home(&self) -> Vec<DVec3> {
        let u = self.units as f64;
        let points: Vec<DVec3> = self.target().iter().map(|p| in_units(*p, u)).collect();
        let s = Stats::of(&points, self.plane, self.axes());
        let shift = if self.centred { s.mean } else { DVec3::ZERO };
        let (hu, hv) = sheet_size(s.var.map(f64::sqrt));
        let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(a, b)| s.mean + s.axes[0] * (a * hu) + s.axes[1] * (b * hv));
        let corners = if self.plane { &[][..] } else { &corners[..] };
        points.iter().chain(corners).map(|p| *p - shift).collect()
    }
}

// ----- where things go on the screen --------------------------------------------------

type Rect = (Vec2, Vec2);

/// Where the panels go on a canvas of `view` logical pixels, `inset` of whose top lies
/// under the application's buttons: in a column at the right of the picture, or, on a
/// canvas narrow and tall enough (a phone held upright), in a band below it. A panel
/// is left out where the canvas cannot spare the room.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    narrow: bool,
    /// The part of the canvas the picture has to itself.
    picture: Rect,
    /// The column or band the panels are in.
    area: Option<Rect>,
    buttons: Option<Rect>,
    bars: Option<Rect>,
    plot: Option<Rect>,
    matrix: Option<Rect>,
    table: Option<Rect>,
    chart: Option<Rect>,
}

const COLUMN: f32 = 250.0;
const BUTTONS_H: f32 = 28.0;
const BARS_H: f32 = 132.0;
const MATRIX_H: f32 = 118.0;
const TABLE_H: f32 = 104.0;
const CHART_H: f32 = 178.0;
const GAP_Y: f32 = 10.0;
/// The size of the caption's text, and the height of each of its lines.
const CAPTION_TEXT: f32 = 12.0;
const CAPTION_LINE: f32 = 19.0;

/// The caption's parts laid out in lines no wider than `width` (logical pixels): side
/// by side as long as they fit, on further lines when they do not. None at all in a
/// view too short to spare the room.
fn caption_lines(parts: &[&str], width: f32, height: f32) -> Vec<String> {
    if height < 300.0 {
        return Vec::new();
    }
    let (room, size, gap) = (width - 36.0, CAPTION_TEXT, "      ");
    let mut lines: Vec<String> = Vec::new();
    for part in parts {
        match lines.last_mut() {
            Some(last) if text_width(last, size) + text_width(gap, size) + text_width(part, size) <= room => {
                last.push_str(gap);
                last.push_str(part);
            }
            _ => lines.push(part.to_string()),
        }
    }
    lines
}

/// The room the caption takes at the bottom of a picture `width` wide.
fn caption_room(parts: &[&str], width: f32, height: f32) -> f32 {
    let lines = caption_lines(parts, width, height).len();
    if lines == 0 {
        0.0
    } else {
        lines as f32 * CAPTION_LINE + 12.0
    }
}

impl Layout {
    /// `caption`: the parts of the caption along the bottom of the picture, which is
    /// kept clear of it.
    fn of(view: Vec2, inset: f32, plane: bool, caption: &[&str]) -> Layout {
        let none = Layout { narrow: false, picture: (Vec2::new(0.0, inset), view), area: None, buttons: None, bars: None, plot: None, matrix: None, table: None, chart: None };
        if view.x < 620.0 && view.y >= 560.0 {
            let caption = caption_room(caption, view.x, view.y);
            // a band below the picture, the panels side by side where they fit
            let (x, w) = (12.0, view.x - 24.0);
            let half = ((w - 12.0) * 0.5).min(210.0);
            let (rows, height) = if plane {
                let chart = (view.y * 0.2).max(120.0).min(170.0);
                (vec![("table", TABLE_H), ("chart", chart)], TABLE_H + chart + GAP_Y)
            } else {
                let plot = half + 22.0;
                let matrix = view.y >= 900.0;
                (vec![("buttons", BUTTONS_H), ("pair", plot)], BUTTONS_H + plot + GAP_Y + if matrix { MATRIX_H + GAP_Y } else { 0.0 })
            };
            let top = view.y - height - 12.0;
            let mut lay = Layout { narrow: true, picture: (Vec2::new(0.0, inset), Vec2::new(view.x, top - 6.0 - caption)), area: Some((Vec2::new(0.0, top - 6.0), view)), ..none };
            let mut y = top;
            for (name, h) in rows {
                let rect = (Vec2::new(x, y), Vec2::new(x + w, y + h));
                match name {
                    "buttons" => lay.buttons = Some(rect),
                    "table" => lay.table = Some(rect),
                    "chart" => lay.chart = Some(rect),
                    _ => {
                        lay.bars = Some((Vec2::new(x, y), Vec2::new(x + w - half - 12.0, y + h)));
                        lay.plot = Some((Vec2::new(x + w - half, y), Vec2::new(x + w, y + h)));
                    }
                }
                y += h + GAP_Y;
            }
            if !plane && view.y >= 900.0 {
                lay.matrix = Some((Vec2::new(x, y), Vec2::new(x + w, y + MATRIX_H)));
            }
            return lay;
        }
        // a column at the right, as wide as can be spared
        let w = COLUMN.min(view.x * 0.4);
        if w < 170.0 || view.y < 260.0 {
            return Layout { picture: (Vec2::new(0.0, inset), Vec2::new(view.x, view.y - caption_room(caption, view.x, view.y))), ..none };
        }
        let x = view.x - w - 14.0;
        let caption = caption_room(caption, x - 10.0, view.y);
        let top = 14.0f32.max(inset + 10.0);
        let room = view.y - top - 12.0;
        let mut lay = Layout { picture: (Vec2::new(0.0, inset), Vec2::new(x - 10.0, view.y - caption)), area: Some((Vec2::new(x - 8.0, top - 6.0), Vec2::new(view.x, view.y))), ..none };
        let mut y = top;
        let mut put = |h: f32| {
            let rect = (Vec2::new(x, y), Vec2::new(x + w, y + h));
            y += h + GAP_Y;
            rect
        };
        if plane {
            if room >= TABLE_H {
                lay.table = Some(put(TABLE_H));
            }
            let chart = (room - TABLE_H - GAP_Y).min(CHART_H);
            if chart >= 110.0 {
                lay.chart = Some(put(chart));
            }
        } else {
            lay.buttons = Some(put(BUTTONS_H));
            if room >= BUTTONS_H + BARS_H + GAP_Y {
                lay.bars = Some(put(BARS_H));
            }
            let plot = (room - BUTTONS_H - BARS_H - 2.0 * GAP_Y).min(w + 22.0);
            let matrix = room - BUTTONS_H - BARS_H - plot - 3.0 * GAP_Y >= MATRIX_H;
            if plot >= 130.0 {
                lay.plot = Some(put(plot));
            }
            if matrix {
                lay.matrix = Some(put(MATRIX_H));
            }
        }
        // the column reaches down only as far as its panels do
        let bottom = [lay.buttons, lay.bars, lay.plot, lay.matrix, lay.table, lay.chart].iter().flatten().map(|r| r.1.y).fold(top, f32::max);
        lay.area = lay.area.map(|(min, max)| (min, Vec2::new(max.x, bottom + 6.0)));
        lay
    }

    fn covers(&self, p: Vec2) -> bool {
        self.area.is_some_and(|(min, max)| p.cmpge(min).all() && p.cmple(max).all())
    }
}

/// The direction the space is seen from at first: in front, to the right and above.
const YAW: f32 = -0.62;
const PITCH: f32 = 0.38;
/// (The tangent of half the camera's field of view.)
const TAN_HALF_FOV: f32 = 0.3203;

/// The camera's right and up directions in the view the scene opens with.
fn screen_axes() -> (Vec3, Vec3) {
    let right = Vec3::new(-YAW.sin(), YAW.cos(), 0.0);
    let up = Vec3::new(-PITCH.sin() * YAW.cos(), -PITCH.sin() * YAW.sin(), PITCH.cos());
    (right, up)
}

impl Principal {
    /// The view that fits the cloud (as it will be) into the part of the canvas the
    /// panels leave free.
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset, self.plane, &self.caption_parts());
        let (min, max) = lay.picture;
        let region = (max - min - Vec2::new(70.0, 50.0)).max(Vec2::splat(60.0));
        let points = self.home();
        if self.plane {
            let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
            for p in points.iter().map(|p| p.truncate().as_vec2()).filter(|p| p.is_finite()) {
                (lo, hi) = (lo.min(p), hi.max(p));
            }
            if lo.x > hi.x {
                (lo, hi) = (Vec2::splat(-1.0), Vec2::splat(1.0));
            }
            let size = (hi - lo).max(Vec2::splat(0.5)) + Vec2::splat(0.6);
            let per_unit = (region.x / size.x).min(region.y / size.y).max(1e-3);
            // the middle of the cloud in the middle of its part of the canvas
            let shift = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
            let centre = (lo + hi) * 0.5 - Vec2::new(shift.x, -shift.y);
            return View::Fit { center: centre, half: self.canvas * 0.5 / per_unit };
        }
        let (right, up) = screen_axes();
        let eye = right.cross(up);
        let points: Vec<Vec3> = points.iter().map(|p| p.as_vec3()).chain([Vec3::ZERO]).filter(|p| p.is_finite()).collect();
        let goal = (min + max) * 0.5;
        // pixels on the screen per unit across the line of sight, at unit distance
        let focal = self.canvas.y * 0.5 / TAN_HALF_FOV;
        let mut target = points.iter().copied().sum::<Vec3>() / points.len().max(1) as f32;
        let mut dist = 20.0f32;
        // seen in perspective, the near side looks larger than the far: fitted by
        // looking at the picture, moving the camera, and looking again
        for _ in 0..16 {
            let from = target + eye * dist;
            let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
            for p in &points {
                let d = *p - from;
                let depth = (-d.dot(eye)).max(1e-3);
                let s = self.canvas * 0.5 + Vec2::new(d.dot(right), -d.dot(up)) * (focal / depth);
                (lo, hi) = (lo.min(s), hi.max(s));
            }
            let size = (hi - lo).max(Vec2::ONE);
            let scale = (region.x / size.x).min(region.y / size.y);
            // the middle of the picture to the middle of its part of the canvas, and
            // nearer or further by the scale wanted
            let shift = ((lo + hi) * 0.5 - goal) * (dist / focal);
            target += right * shift.x - up * shift.y;
            dist = (dist / scale).clamp(2.0, 400.0);
        }
        View::Orbit { target, yaw: YAW, pitch: PITCH, dist }
    }
}

// ----- drawing --------------------------------------------------------------------------

/// A button drawn over the picture (logical pixels): true when it is clicked. One that
/// is chosen is drawn lit.
fn hud_button(f: &mut Frame, (min, max): Rect, text: &str, chosen: bool, pointer: Option<Vec2>) -> bool {
    let over = pointer.is_some_and(|p| p.cmpge(min).all() && p.cmple(max).all());
    let fill = match (chosen, over) {
        (true, _) => Color::hex(0x3A4C78),
        (false, true) => Color::hex(0x33405E),
        (false, false) => Color::hex(0x222B40),
    };
    hud_rect(f, min, max, fill);
    f.text_px((min + max) * 0.5, text, 12.5, pal::TEXT, Align2::CENTER_CENTER);
    over && f.ptr.clicked
}

/// The plane kept: a rectangle in space through `centre`, along the unit directions `u`
/// and `v`, out to `hu` and `hv` either way, see-through, with a few grid lines, and
/// faded by `alpha` (as it comes and goes).
fn sheet_rect(f: &mut Frame, centre: Vec3, u: Vec3, v: Vec3, (hu, hv): (f32, f32), alpha: f32) {
    let (q, color) = (|s: f32, t: f32| centre + u * s + v * t, KEPT);
    f.dl.tris_alpha.quad(q(-hu, -hv), q(hu, -hv), q(hu, hv), q(-hu, hv), color.a(0.11 * alpha));
    for k in -3..=3 {
        let s = k as f32 / 3.0;
        f.line(q(s * hu, -hv), q(s * hu, hv), 1.0, color.a(0.3 * alpha));
        f.line(q(-hu, s * hv), q(hu, s * hv), 1.0, color.a(0.3 * alpha));
    }
}

/// A caption along the bottom of the picture, on a dark plate, like `bottom_caption`
/// in the toolkit but kept to the picture's width and ending at `bottom`.
fn caption(f: &mut Frame, parts: &[&str], left: f32, width: f32, bottom: f32) {
    let lines = caption_lines(parts, width, f.view_size().y);
    let count = lines.len();
    for (k, line) in lines.into_iter().enumerate() {
        let up = (count - 1 - k) as f32 * CAPTION_LINE;
        f.caption_px(Vec2::new(left + 18.0, bottom - up), line, CAPTION_TEXT, pal::DIM, Align2::LEFT_BOTTOM);
    }
}

/// How far the plane of PC1 and PC2 is drawn out either way along each, for these
/// spreads along the components.
fn sheet_size<T: Into<f64> + Copy>(spread: [T; 3]) -> (f64, f64) {
    let (a, b) = (spread[0].into(), spread[1].into());
    (2.5 * a, (2.5 * b).max(0.35 * a))
}

/// The numbers along the axes of a flat view, as `Frame::axes2` writes them, but only
/// within the part `(min, max)` of the canvas (logical pixels) that the picture has to
/// itself: on a phone the panels lie below it.
fn ticks(f: &mut Frame, (min, max): Rect) {
    let (lo, hi) = f.cam.flat_bounds();
    let view = f.view_size();
    let wpp = (hi.y - lo.y) / view.y.max(1.0);
    if !(wpp.is_finite() && wpp > 0.0 && max.x - min.x > 40.0 && max.y - min.y > 40.0) {
        return;
    }
    // the picture's part of the view, in the scene's units
    let (lo, hi) = (Vec2::new(lo.x + min.x * wpp, hi.y - max.y * wpp), Vec2::new(lo.x + max.x * wpp, hi.y - min.y * wpp));
    let step = nice_step(wpp * 95.0);
    let margin = 16.0 * wpp;
    // (on the axis where it is in view, otherwise along the edge; max then min, which
    // cannot panic where the bounds cross)
    let y_at = 0.0f32.max(lo.y + margin * 1.6).min(hi.y - margin * 0.4);
    let x_at = 0.0f32.max(lo.x + margin * 0.4).min(hi.x - margin * 2.4);
    let edge = 20.0 * wpp;
    let values = |a: f32, b: f32| {
        let first = (a / step).ceil();
        let count = (((b - a) / step).floor().max(0.0) as usize).min(400);
        (0..=count).map(move |i| (first + i as f32) * step).filter(move |v| *v <= b)
    };
    for x in values(lo.x, hi.x) {
        if x.abs() > step * 0.01 && x > lo.x + edge && x < hi.x - edge {
            f.text_off(Vec2::new(x, y_at), Vec2::new(0.0, 5.0), fmt_tick(x, step), 12.0, pal::DIM, Align2::CENTER_TOP);
        }
    }
    for y in values(lo.y, hi.y) {
        if y.abs() > step * 0.01 && y > lo.y + edge * 0.6 && y < hi.y - edge * 0.6 {
            f.text_off(Vec2::new(x_at, y), Vec2::new(6.0, 0.0), fmt_tick(y, step), 12.0, pal::DIM, Align2::LEFT_CENTER);
        }
    }
}

/// The line under the bars: how much of the variance the components kept keep, in as
/// many words as fit in `room` logical pixels.
fn kept_line(s: &Stats, k: usize, room: f32) -> String {
    if !s.spread_out() {
        return "no spread at all".to_string();
    }
    let (kept, lost) = (percent(s.kept(k)), percent(1.0 - s.kept(k)));
    let by = if k == 1 { "PC1".to_string() } else { format!("PC1–{k}") };
    [format!("kept by {by}: {kept}   lost: {lost}"), format!("kept: {kept}   lost: {lost}"), format!("kept: {kept}")].into_iter().find(|t| text_width(t, 11.5) <= room).unwrap_or(kept)
}

/// A percentage, to one decimal.
fn percent(share: f64) -> String {
    format!("{}%", num(100.0 * share, 1))
}

impl Principal {
    /// The cloud in space: the points, what is kept, the misses, the axes.
    fn space(&mut self, f: &mut Frame, lay: &Layout, room: &mut Room) {
        let drawn = self.drawn();
        let axes = self.axes().map(|a| a.as_vec3());
        let mean = (self.stats.mean - self.shift()).as_vec3();
        let spread = self.spread.map(|s| s as f32);
        let w = self.kept.map(smoothstep);
        let line_alpha = w[0] * (1.0 - w[1]);
        let sheet_alpha = w[1] * (1.0 - w[2]);

        // the axes of x, y and z, faint, through 0 and out to beyond the cloud, where
        // they are named
        let reach = drawn.iter().map(|d| d.2.as_vec3().length()).fold(2.0, f32::max) * 1.05;
        for d in Vec3::AXES {
            f.line(-d * reach, d * reach, 1.0, pal::FAINT.a(0.8));
            room.line(f, -d * reach, d * reach);
        }

        // what is kept: the plane of the first two, or the line of the first
        let (hu, hv) = sheet_size(spread);
        let (hu, hv) = (hu as f32, hv as f32);
        if sheet_alpha > 0.0 && spread[0] > 0.0 {
            sheet_rect(f, mean, axes[0], axes[1], (hu, hv), sheet_alpha);
            room.shape(f, &[mean - axes[0] * hu - axes[1] * hv, mean + axes[0] * hu - axes[1] * hv, mean + axes[0] * hu + axes[1] * hv, mean - axes[0] * hu + axes[1] * hv]);
        }
        if line_alpha > 0.0 && spread[0] > 0.0 {
            let end = axes[0] * (3.0 * spread[0]);
            f.line(mean - end, mean + end, 3.0, KEPT.a(0.85 * line_alpha));
            room.line(f, mean - end, mean + end);
        }
        // with a far point, the plane the other points would have without it, outlined
        let ghost = self.without_far().filter(|_| sheet_alpha > 0.5);
        if let Some(g) = ghost {
            let (hu, hv) = sheet_size(g.var.map(f64::sqrt));
            let c = (g.mean - self.shift()).as_vec3();
            let (u, v) = (g.axes[0].as_vec3() * hu as f32, g.axes[1].as_vec3() * hv as f32);
            let corners = [c - u - v, c + u - v, c + u + v, c - u + v];
            for k in 0..4 {
                f.dashed(corners[k], corners[(k + 1) % 4], 1.6, pal::WHITE.a(0.7));
                room.line(f, corners[k], corners[(k + 1) % 4]);
            }
        }

        // the misses, for one point in three (and the far point), and the points
        let far = (self.cloud == Cloud::Far).then_some(N - 1);
        let mut dots = Vec::with_capacity(drawn.len());
        for (i, (now, shadow, at)) in drawn.iter().enumerate() {
            let (at, shadow) = (at.as_vec3(), shadow.as_vec3());
            // (every one of them, even where it is next to nothing: a line that came and
            // went as a point crossed the plane would be a flicker)
            if (i % 3 == 0 || Some(i) == far) && w[2] < 1.0 {
                let width = if Some(i) == far { 2.2 } else { 1.1 };
                f.line(at, shadow, width, MISS.a(0.6));
            }
            if Some(i) != far {
                dots.push(now.as_vec3());
            }
        }
        // (without the dark rim the built-in discs have: at this size it would be half the dot)
        let size = if lay.narrow { 4.2 } else { 5.5 };
        let batch = f.dl.points_with(PointKind::Disc, PointStyle::default());
        for p in dots {
            batch.push(p, size, DOT);
        }

        // the components from the mean, each as long as the spread along it
        for i in 0..3 {
            let tip = mean + axes[i] * spread[i];
            if spread[i] > 1e-4 {
                f.line(mean, mean - axes[i] * spread[i], 2.4, PC[i].a(0.55));
                f.arrow(mean, tip, 3.4, PC[i]);
            }
            room.line(f, mean - axes[i] * spread[i], tip);
        }
        f.dot(mean, 9.0, pal::WHITE);

        // the far point, to drag (while it is where it was put: not on its way to a
        // shadow, nor gliding to a new cloud)
        if let Some(i) = far {
            let mut p = drawn[i].0.as_vec3();
            let hidden = on_screen(f, p).is_some_and(|s| lay.covers(s));
            if self.can_drag() && (!hidden || f.handles.dragging()) {
                if f.drag3(1, &mut p, None, FAR) {
                    self.drag_far(p.as_dvec3());
                }
            } else {
                f.dot(p, 9.0, FAR);
            }
        }

        // the names: the components first, so that they get the better places. (Every
        // one is written whenever what it names is there, room or no room: a name that
        // came and went as the picture moved would be a flicker.)
        let outwards = |f: &Frame, from: Vec3, to: Vec3| match (on_screen(f, from), on_screen(f, to)) {
            (Some(a), Some(b)) if (b - a).length() > 1.0 => b - a,
            _ => Vec2::new(1.0, -1.0),
        };
        for i in 0..3 {
            if spread[i] > 1e-4 {
                let tip = mean + axes[i] * spread[i];
                let dir = outwards(f, mean, tip);
                room.label(f, ("pc", i), tip, NAMES[i], 15.0, PC[i], dir, 7.0, false);
            }
        }
        if !self.stats.spread_out() {
            room.label(f, "none", mean, "every point is the same: no spread at all", 13.0, pal::TEXT, Vec2::new(0.0, 1.0), 14.0, false);
        } else {
            room.label(f, "mean", mean, "mean", 12.5, pal::WHITE, Vec2::new(-1.0, 1.0), 8.0, false);
        }
        if let Some(i) = far {
            let p = drawn[i].0.as_vec3();
            room.label(f, "far", p, "far point", 13.0, FAR, Vec2::new(1.0, -0.4), 10.0, false);
        }
        // (the plane's name inside it, along its edge: beside a corner it would swap
        // sides of the corner as the view turns. On a phone the picture is too small
        // for it, and the caption names the plane.)
        if sheet_alpha > 0.5 && !lay.narrow {
            let at = mean + axes[0] * (0.6 * hu) - axes[1] * (0.85 * hv);
            room.label(f, "sheet", at, "plane of PC1 and PC2", 12.5, KEPT.mix(pal::WHITE, 0.45), Vec2::ZERO, 0.0, false);
        }
        // (an axis whose end is off the picture, or under a panel, goes unnamed)
        let (lo, hi) = lay.picture;
        for (k, (name, text)) in [("x", "$x$"), ("y", "$y$"), ("z", "$z$")].into_iter().enumerate() {
            let d = Vec3::AXES[k] * reach;
            if on_screen(f, d).is_some_and(|p| p.cmpge(lo + Vec2::splat(12.0)).all() && p.cmple(hi - Vec2::splat(12.0)).all()) {
                room.label(f, name, d, text, 12.0, pal::DIM, outwards(f, Vec3::ZERO, d), 4.0, false);
            }
        }
    }

    /// Can the far point be dragged? Only while it is drawn where it is.
    fn can_drag(&self) -> bool {
        self.morph >= 1.0 && self.units_change.is_none() && (self.plane || (self.flat <= 0.0 && self.playing.is_none()))
    }

    /// With a far point, the statistics of the other points alone.
    fn without_far(&self) -> Option<Stats> {
        (self.cloud == Cloud::Far && self.data.len() == N).then(|| Stats::of(&self.data[..N - 1], self.plane, self.stats.axes))
    }

    /// Move the far point to `p` (as drawn), no further than its reach.
    fn drag_far(&mut self, p: DVec3) {
        let u = self.units_now() as f64;
        let data = p + self.shift();
        // (with x in units of nothing, every x is 0, and where the point is along x
        // cannot be told: it stays where it was)
        let x = if u.abs() > 1e-6 { data.x / u } else { self.far.x };
        let raw = DVec3::new(x, data.y, if self.plane { 0.0 } else { data.z });
        let from = offset(self.plane);
        let d = raw - from;
        let raw = if d.length() > FAR_REACH { from + d * (FAR_REACH / d.length()) } else { raw };
        if raw.is_finite() {
            self.far = raw;
            self.refresh();
        }
    }

    /// The cloud in the plane, with the first component's line and the regression line.
    fn flat_picture(&mut self, f: &mut Frame, lay: &Layout, room: &mut Room) {
        let drawn = self.drawn();
        let axes = self.axes().map(|a| a.as_vec3());
        let mean = (self.stats.mean - self.shift()).as_vec3();
        let spread = self.spread.map(|s| s as f32);
        let reg = regression(&self.stats).map(|d| d.as_vec3());
        let (lo, hi) = f.cam.flat_bounds();
        let long = (hi - lo).length() * 2.0 + (mean.truncate() - (lo + hi) * 0.5).length();
        let far = (self.cloud == Cloud::Far).then_some(N - 1);

        // the two lines, across the whole view
        if spread[0] > 0.0 {
            f.line(mean - axes[0] * long, mean + axes[0] * long, 2.6, PC[0]);
            room.line(f, mean - axes[0] * long, mean + axes[0] * long);
        }
        if let Some(r) = reg {
            f.line(mean - r * long, mean + r * long, 2.6, REG);
            room.line(f, mean - r * long, mean + r * long);
        }

        // the misses: at right angles to PC1, and up and down to the regression line
        let foot = |p: Vec3| mean + axes[0] * (p - mean).dot(axes[0]);
        let below = |p: Vec3| reg.map(|r| Vec3::new(p.x, mean.y + r.y / r.x * (p.x - mean.x), 0.0));
        let mut dots = Vec::with_capacity(drawn.len());
        for (i, (now, _, _)) in drawn.iter().enumerate() {
            let p = now.as_vec3();
            // (both kinds for the picked point and the far point; for the rest, one
            // kind each, for one point in five)
            let both = i == self.picked || Some(i) == far;
            let bold = if both { 2.4 } else { 1.0 };
            if spread[0] > 0.0 && (both || i % 10 == 0) {
                f.line(p, foot(p), bold, MISS.a(if both { 1.0 } else { 0.6 }));
            }
            if let Some(q) = below(p).filter(|_| both || i % 10 == 5) {
                f.line(p, q, bold, REG.a(if both { 1.0 } else { 0.5 }));
            }
            if i == self.picked {
                f.dot(p, 8.0, pal::WHITE);
            } else if Some(i) != far {
                dots.push(p);
            }
        }
        let size = if lay.narrow { 4.2 } else { 5.0 };
        let batch = f.dl.points_with(PointKind::Disc, PointStyle::default());
        for p in dots {
            batch.push(p, size, DOT);
        }
        // the picked point's miss meets PC1's line at a right angle
        if let Some((p, _, _)) = drawn.get(self.picked) {
            let p = p.as_vec3();
            let q = foot(p);
            let size = 12.0 * f.cam.world_per_px(q) * f.px;
            if (p - q).length() > 2.5 * size && spread[0] > 0.0 {
                // (the corner on the side of the foot towards the mean)
                let along = if (mean - q).length() > size { mean - q } else { axes[0] };
                right_angle(f, q, p - q, along, size, pal::WHITE.a(0.9));
            }
            let across = if (p - q).dot(axes[1]) >= 0.0 { axes[1] } else { -axes[1] };
            let dir = Vec2::new(across.x, -across.y);
            room.label(f, "right", (p + q) * 0.5, "at right angles", 12.5, MISS.mix(pal::WHITE, 0.2), dir.perp(), 8.0, false);
            if let Some(b) = below(p) {
                let side = if (p.x - mean.x) >= 0.0 { Vec2::new(1.0, 0.0) } else { Vec2::new(-1.0, 0.0) };
                room.label(f, "up", (p + b) * 0.5, "up and down", 12.5, REG, side, 8.0, false);
            }
        }
        f.dot(mean, 9.0, pal::WHITE);
        room.label(f, "mean", mean, "mean", 12.5, pal::WHITE, Vec2::new(-1.0, 1.0), 8.0, false);

        // the far point, to drag
        if let Some(i) = far {
            let p = drawn[i].0.as_vec3();
            let mut q = p.truncate();
            let hidden = on_screen(f, p).is_some_and(|s| lay.covers(s));
            if self.can_drag() && (!hidden || f.handles.dragging()) {
                if f.drag2(1, &mut q, FAR) {
                    self.drag_far(q.extend(0.0).as_dvec3());
                }
            } else {
                f.dot(p, 9.0, FAR);
            }
            room.label(f, "far", p, "far point", 13.0, FAR, Vec2::new(1.0, -0.4), 10.0, false);
        }

        // the names of the lines, beyond the cloud's right-hand end (or its upper end)
        let reach = 3.2 * spread[0].max(0.3);
        let name_at = |d: Vec3| if d.x > 1e-3 || (d.x.abs() <= 1e-3 && d.y > 0.0) { mean + d * reach } else { mean - d * reach };
        if spread[0] > 0.0 {
            let at = name_at(axes[0]);
            room.label(f, "pc1", at, "PC1", 15.0, PC[0], Vec2::new(-1.0, -1.0), 8.0, false);
        }
        if let Some(r) = reg {
            let at = name_at(r);
            room.label(f, "reg", at, "regression line ($y$ from $x$)", 13.0, REG, Vec2::new(1.0, 1.0), 8.0, false);
        } else {
            room.label(f, "reg", mean, "no regression line: every $x$ is the same", 13.0, REG, Vec2::new(0.0, 1.0), 16.0, false);
        }
        if !self.stats.spread_out() {
            room.label(f, "none", mean, "every point is the same: no spread at all", 13.0, pal::TEXT, Vec2::new(0.0, -1.0), 14.0, false);
        }
    }
}

impl Principal {
    /// The buttons over the picture: how many components to keep, and the flattening.
    fn buttons(&mut self, f: &mut Frame, (min, max): Rect, pointer: Option<Vec2>) {
        hud_panel(f, min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        f.text_px(Vec2::new(min.x + 4.0, (min.y + max.y) * 0.5), "keep", 12.0, pal::DIM, Align2::LEFT_CENTER);
        let x = min.x + 40.0;
        for k in 1..=3 {
            let at = x + (k - 1) as f32 * 34.0;
            if hud_button(f, (Vec2::new(at, min.y), Vec2::new(at + 30.0, max.y)), &k.to_string(), self.keep == k, pointer) {
                self.set_keep(k);
            }
        }
        let back = self.flat >= 1.0 || self.playing == Some(1.0);
        let text = if back { "◀ unflatten" } else { "▶ flatten" };
        if hud_button(f, (Vec2::new(x + 3.0 * 34.0 + 6.0, min.y), max), text, false, pointer) {
            self.play();
        }
    }

    /// Each component's share of the variance, as bars.
    fn bars(&self, f: &mut Frame, (min, max): Rect) {
        hud_panel(f, min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        let w = max.x - min.x;
        let title = ["each component's share of the variance", "share of the variance"].into_iter().find(|t| text_width(t, 11.5) <= w).unwrap_or("shares");
        f.text_px(min + Vec2::new(4.0, 2.0), title, 11.5, pal::TEXT, Align2::LEFT_TOP);
        let shares = self.stats.shares();
        let (top, bottom) = (min.y + 34.0, max.y - 36.0);
        let bar = (w - 16.0) / 3.0;
        hud_line(f, Vec2::new(min.x + 4.0, bottom), Vec2::new(max.x - 4.0, bottom), 1.0, pal::DIM.a(0.8));
        for i in 0..3 {
            let x = min.x + 8.0 + bar * i as f32;
            let h = (bottom - top) * shares[i].max(0.0).min(1.0) as f32;
            let kept = i < self.keep;
            let c = if kept { PC[i] } else { PC[i].a(0.3) };
            hud_rect(f, Vec2::new(x + bar * 0.15, bottom - h), Vec2::new(x + bar * 0.85, bottom), c);
            f.text_px(Vec2::new(x + bar * 0.5, bottom - h - 3.0), percent(shares[i]), 11.5, if kept { pal::TEXT } else { pal::DIM }, Align2::CENTER_BOTTOM);
            f.text_px(Vec2::new(x + bar * 0.5, bottom + 3.0), NAMES[i], 11.5, PC[i], Align2::CENTER_TOP);
        }
        f.text_px(Vec2::new(min.x + 4.0, max.y - 2.0), kept_line(&self.stats, self.keep, w - 4.0), 11.5, pal::TEXT, Align2::LEFT_BOTTOM);
    }

    /// The points in the coordinates of PC1 and PC2: a "PCA plot".
    fn plot(&self, f: &mut Frame, (min, max): Rect, drawn: &[(DVec3, DVec3, DVec3)]) {
        hud_panel(f, min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        let w = max.x - min.x;
        let title = ["PCA plot: the points in PC1, PC2 coordinates", "PCA plot: in PC1, PC2 coordinates", "PCA plot"].into_iter().find(|t| text_width(t, 11.5) <= w).unwrap_or("");
        f.text_px(min + Vec2::new(4.0, 2.0), title, 11.5, pal::TEXT, Align2::LEFT_TOP);
        let side = (w - 8.0).min(max.y - min.y - 24.0).max(10.0);
        let centre = Vec2::new((min.x + max.x) * 0.5, max.y - 4.0 - side * 0.5);
        let (m, axes, shift) = (self.stats.mean, self.axes(), self.shift());
        // (the same scale across and up, and the same while the points are flattened)
        let reach = drawn.iter().map(|d| {
            let c = d.2 + shift - m;
            c.dot(axes[0]).abs().max(c.dot(axes[1]).abs())
        });
        let reach = reach.fold(1e-9, f64::max) * 1.08;
        let scale = side as f64 * 0.5 / reach;
        hud_line(f, centre - Vec2::X * side * 0.5, centre + Vec2::X * side * 0.5, 1.2, PC[0].a(0.8));
        hud_line(f, centre + Vec2::Y * side * 0.5, centre - Vec2::Y * side * 0.5, 1.2, PC[1].a(0.8));
        f.text_px(centre + Vec2::new(side * 0.5, 3.0), "PC1", 11.0, PC[0], Align2::RIGHT_TOP);
        f.text_px(centre + Vec2::new(4.0, -side * 0.5), "PC2", 11.0, PC[1], Align2::LEFT_TOP);
        let far = (self.cloud == Cloud::Far).then_some(N - 1);
        for (i, d) in drawn.iter().enumerate() {
            let c = d.0 + shift - m;
            let at = centre + Vec2::new((c.dot(axes[0]) * scale) as f32, -(c.dot(axes[1]) * scale) as f32);
            if at.is_finite() {
                let (size, color) = if Some(i) == far { (7.0, FAR) } else { (3.6, DOT) };
                f.dot(f.hud_at(2, at), size, color);
            }
        }
    }

    /// The covariance matrix, and its eigenvalues.
    fn matrix(&self, f: &mut Frame, (min, max): Rect) {
        hud_panel(f, min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        let c = self.stats.cov;
        let size = 12.5;
        // (the entries set as formulas, as "C =" is; the cells as wide as their numbers)
        let widest = c.iter().flatten().map(|v| num(*v, 2).chars().count()).max().unwrap_or(4).max(4);
        let rows: Vec<Vec<(String, Color)>> = c.iter().map(|row| row.iter().map(|v| (format!("${}$", num(*v, 2)), pal::TEXT)).collect()).collect();
        let msize = matrix_size(3, 3, widest, size);
        f.text_px(min + Vec2::new(4.0, 2.0), "covariance matrix of $x$, $y$, $z$", 11.5, pal::TEXT, Align2::LEFT_TOP);
        let top = Vec2::new(min.x + 34.0, min.y + 22.0);
        // (ending just before the bracket: set as a formula, it is wider than it was)
        f.text_px(Vec2::new(top.x - 5.0, top.y + msize.y * 0.5), "$C =$", 13.0, pal::TEXT, Align2::RIGHT_CENTER);
        hud_matrix(f, top, &rows, widest, size, pal::DIM);
        let values: Vec<String> = self.stats.var.iter().map(|v| num(*v, 2)).collect();
        let line = format!("eigenvalues: {}", values.join(", "));
        f.text_px(Vec2::new(min.x + 4.0, max.y - 18.0), line, 11.5, pal::TEXT, Align2::LEFT_BOTTOM);
        f.text_px(Vec2::new(min.x + 4.0, max.y - 2.0), "the variances along PC1, PC2, PC3", 11.0, pal::DIM, Align2::LEFT_BOTTOM);
    }

    /// The two lines' sums of squared misses, measured both ways.
    fn table(&self, f: &mut Frame, (min, max): Rect) {
        hud_panel(f, min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        let s = &self.stats;
        let pc = misses(&self.data, s.mean, s.axes[0]);
        let reg = regression(s).map(|r| misses(&self.data, s.mean, r));
        f.text_px(min + Vec2::new(4.0, 2.0), "sums of squared misses", 11.5, pal::TEXT, Align2::LEFT_TOP);
        let w = max.x - min.x;
        let cols = [min.x + w * 0.62, min.x + w * 0.88];
        let y0 = min.y + 30.0;
        f.text_px(Vec2::new(cols[0], y0), "PC1", 12.0, PC[0], Align2::CENTER_CENTER);
        f.text_px(Vec2::new(cols[1], y0), "regression", 12.0, REG, Align2::CENTER_CENTER);
        let rows = [("at right angles", MISS, Some(pc.0), reg.map(|r| r.0)), ("up and down", REG, pc.1, reg.and_then(|r| r.1))];
        for (k, (name, color, a, b)) in rows.into_iter().enumerate() {
            let y = y0 + 24.0 + k as f32 * 22.0;
            f.text_px(Vec2::new(min.x + 4.0, y), name, 12.0, color, Align2::LEFT_CENTER);
            let least = match (a, b) {
                (Some(a), Some(b)) => Some(if a <= b { 0 } else { 1 }),
                (Some(_), None) => Some(0),
                (None, Some(_)) => Some(1),
                _ => None,
            };
            for (j, v) in [a, b].into_iter().enumerate() {
                let text = v.map_or("—".to_string(), |v| num(v, 1));
                let best = least == Some(j);
                if best {
                    hud_rect(f, Vec2::new(cols[j] - 30.0, y - 9.0), Vec2::new(cols[j] + 30.0, y + 9.0), color.a(0.22));
                }
                f.text_px(Vec2::new(cols[j], y), text, 12.0, if best { pal::WHITE } else { pal::DIM }, Align2::CENTER_CENTER);
            }
        }
        f.text_px(Vec2::new(min.x + 4.0, max.y - 2.0), "lit: the smaller of the two", 11.0, pal::DIM, Align2::LEFT_BOTTOM);
    }

    /// The sum of squared right-angle misses of every line through the mean, by its angle.
    fn chart(&self, f: &mut Frame, (min, max): Rect) {
        hud_panel(f, min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        let s = &self.stats;
        let w = max.x - min.x;
        let title = ["every line through the mean, by its angle:", "every line through the mean:"].into_iter().find(|t| text_width(t, 11.5) <= w).unwrap_or("");
        f.text_px(min + Vec2::new(4.0, 2.0), title, 11.5, pal::TEXT, Align2::LEFT_TOP);
        f.text_px(min + Vec2::new(4.0, 18.0), "its sum of squared right-angle misses", 11.0, MISS, Align2::LEFT_TOP);
        let (o, size) = (Vec2::new(min.x + 10.0, min.y + 40.0), Vec2::new(w - 20.0, max.y - min.y - 58.0));
        let values: Vec<f64> = (0..=180).map(|d| misses_at(s, d as f64)).collect();
        let top = values.iter().copied().fold(1e-12, f64::max) * 1.1;
        let at = |deg: f64, v: f64| o + Vec2::new(size.x * (deg / 180.0) as f32, size.y * (1.0 - (v / top).max(0.0).min(1.0) as f32));
        hud_line(f, o + Vec2::new(0.0, size.y), o + size, 1.0, pal::DIM.a(0.8));
        hud_path(f, values.iter().enumerate().map(|(d, v)| at(d as f64, *v)), 1.8, MISS);
        for (deg, label) in [(0.0, "0°"), (45.0, "45°"), (90.0, "90°"), (135.0, "135°"), (180.0, "180°")] {
            let p = at(deg, 0.0);
            hud_line(f, p, p + Vec2::Y * 4.0, 1.0, pal::DIM);
            f.text_px(p + Vec2::Y * 5.0, label, 10.5, pal::DIM, Align2::CENTER_TOP);
        }
        // where the two lines are on it
        let mut marks = vec![(angle_of(s.axes[0]), PC[0])];
        if let Some(r) = regression(s) {
            marks.push((angle_of(r), REG));
        }
        for (deg, color) in marks {
            let p = at(deg, misses_at(s, deg));
            hud_line(f, Vec2::new(p.x, o.y + size.y), p, 1.0, color.a(0.6));
            f.dot(f.hud_at(2, p), 8.0, color);
        }
    }

    /// The names the caption gives, for what has no name of its own in the picture.
    fn caption_parts(&self) -> Vec<&'static str> {
        if self.plane {
            let mut parts = vec!["dots: the data", "pink: misses at right angles to PC1", "cyan: misses up and down to the regression line"];
            if self.cloud == Cloud::Far {
                parts.push("red: the far point (drag it)");
            }
            parts
        } else {
            let mut parts = vec!["dots: the data", "arrows: the components, as long as the spread along each"];
            if self.keep < 3 {
                parts.push(if self.keep == 2 { "blue: the plane kept" } else { "blue: the line kept" });
                parts.push("pink: misses");
            }
            if self.cloud == Cloud::Far {
                parts.push("red: the far point (drag it)");
                if self.keep == 2 {
                    parts.push("dashed: the plane without it");
                }
            }
            parts
        }
    }
}

impl Scene for Principal {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.07;
    }

    fn animated(&self) -> bool {
        self.moving()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "Each dot is one data point: three numbers, $x$, $y$ and $z$. Principal components describe the cloud's shape, measured from its mean (the white dot). The first, PC1, is the direction in which the cloud is spread most; PC2 is the direction of most spread at right angles to PC1; PC3 is at right angles to both. Each arrow is as long as the spread along it. Keeping the first two and dropping each point straight onto their plane loses no more than flattening the cloud onto any other plane.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The data");
        let mut space = usize::from(self.plane);
        if ui::choice(ui, &["in space", "in the plane"], &mut space) {
            self.set_cloud(self.cloud, self.seed, space == 1);
        }
        ui::note(ui, "In the plane each point has two numbers, $x$ and $y$, and the regression line is drawn beside PC1.");
        let names: Vec<&str> = CLOUDS.iter().map(|c| c.1).collect();
        let mut chosen = CLOUDS.iter().position(|c| c.0 == self.cloud).unwrap_or(0);
        if ui::choice(ui, &names, &mut chosen) {
            self.set_cloud(CLOUDS[chosen].0, self.seed, self.plane);
        }
        if ui.button("new points").clicked() {
            self.set_cloud(self.cloud, self.seed.wrapping_add(1), self.plane);
        }
        ui::note(ui, "Another draw of 300 points from the same shape.");
        let mut units = self.units;
        if ui::slider_log_or_zero(ui, "units of $x$", &mut units, 0.1, 10.0) {
            self.set_units(units);
        }
        ui::note(ui, "Every $x$ is multiplied by this: $x$ measured in other units. At 0 every $x$ is $0$, and the cloud is squashed flat.");
        if ui.checkbox(&mut self.centred, "move the mean to 0").changed() {
            self.new_view = true;
        }
        ui::note(ui, "Nothing about the components changes: they are measured from the mean wherever it is.");

        if !self.plane {
            ui::section(ui, "Flattening");
            let mut keep = self.keep - 1;
            if ui::choice(ui, &["keep PC1 (a line)", "PC1 and PC2 (a plane)", "all three"], &mut keep) {
                self.set_keep(keep + 1);
            }
            if ui::slider(ui, "flatten", &mut self.flat, 0.0..=1.0) {
                self.playing = None;
            }
            if ui.button(if self.flat >= 1.0 { "◀ unflatten" } else { "▶ flatten" }).clicked() {
                self.play();
            }
            ui::note(ui, "Moves each point straight to its shadow on the line or plane kept, along its miss.");
        }

        ui::section(ui, "Reading the picture");
        if self.plane {
            ui::key(
                ui,
                &[
                    (DOT, "dots: the data points"),
                    (PC[0], "orange line: PC1, the line through the mean along which the points are spread most"),
                    (MISS, "pink: misses measured at right angles to PC1 (for one point in ten)"),
                    (REG, "cyan line: the regression line of $y$ on $x$; thin cyan: its misses, measured up and down (for one point in ten)"),
                    (pal::WHITE, "white dot: the mean; the larger white point: one point with both its misses drawn thick"),
                    (MISS, "the chart: for each angle, the line through the mean at that angle, and its sum of squared right-angle misses; the dots: where PC1 and the regression line are on it"),
                    (FAR, "red: the far point (drag it)"),
                ],
            );
        } else {
            ui::key(
                ui,
                &[
                    (DOT, "dots: the data points; in the PCA plot, the same points in the coordinates of PC1 and PC2"),
                    (PC[0], "orange arrow: PC1, from the mean, as long as the spread along it (and a line as long the other way)"),
                    (PC[1], "purple: PC2"),
                    (PC[2], "green: PC3"),
                    (KEPT, "blue: what is kept, the plane of PC1 and PC2 (or the line of PC1)"),
                    (MISS, "pink: misses, each from a point to its shadow on what is kept (for one point in three)"),
                    (pal::WHITE, "white dot: the mean; dashed white (with a far point): the plane the other points would have without it"),
                    (FAR, "red: the far point (drag it)"),
                    (pal::FAINT, "thin grey lines: the $x$, $y$ and $z$ axes"),
                ],
            );
        }

        ui::section(ui, "Words");
        ui::term(ui, "Mean:", "the average of the points, their centre. PCA measures every point from it.");
        ui::term(ui, "Variance along a direction:", "the squared distances of the points from the mean, measured along that direction, added up and divided by $n - 1$ ($n$ is the number of points). Its square root is the spread, or standard deviation.");
        ui::term(ui, "Covariance matrix $C$:", r"the variances of $x$, $y$ and $z$ on its diagonal; off it, how two coordinates vary together. The variance along a unit direction $u$ is $u^\top C u$.");
        ui::term(ui, "Principal components:", "the eigenvectors of $C$, largest eigenvalue first. Each eigenvalue is the variance along its component, and its share of the total is how much of the variance that component carries.");
        ui::term(ui, "PCA plot:", "each point drawn at its coordinates along PC1 and PC2: the cloud flattened onto their plane, and laid flat.");
        ui::term(ui, "Shadow and miss:", "a point's shadow is the nearest point to it on the line or plane kept; its miss is the step from it to its shadow, at right angles to what is kept.");
        ui::term(ui, "Why both definitions agree:", "each point's squared distance from the mean is its shadow's squared distance plus its miss squared (Pythagoras). Their total is fixed, so the plane that keeps the most variance is the plane with the least sum of squared misses.");
        ui::term(ui, "Singular values:", r"write the points, less the mean, as the rows of a matrix $X$. Its right singular vectors are the components, and its singular values $\sigma$ satisfy $\sigma^2 / (n - 1) = \lambda$, the eigenvalues.");
        if self.plane {
            ui::term(ui, "Regression line:", "the line that predicts $y$ from $x$ with the least sum of squared misses measured up and down, along $y$.");
        }

        ui::section(ui, "Numbers");
        let s = self.stats;
        ui::readout(ui, "points", s.n.to_string());
        ui::readout(ui, "mean", format!("${}$", tuple(&[s.mean.x, s.mean.y, s.mean.z], 2)));
        let dims = if self.plane { 2 } else { 3 };
        let shares = s.shares();
        for i in 0..dims {
            let a = s.axes[i];
            let dir = format!("${}$", if self.plane { tuple(&[a.x, a.y], 2) } else { tuple(&[a.x, a.y, a.z], 2) });
            ui::readout_colored(ui, &format!("{} along", NAMES[i]), dir, PC[i]);
            ui::readout_colored(ui, &format!(r"$\lambda_{{{}}}$, its share", i + 1), format!("{}, {}", num(s.var[i], 3), percent(shares[i])), PC[i]);
        }
        if !s.spread_out() {
            ui::note(ui, "Every point is in the same place: there is no spread to share out, and no direction is a component more than any other.");
        }
        let (sv, _) = singular(&self.data, s.mean);
        let div = (s.n.max(2) - 1) as f64;
        ui::readout(ui, r"$\sigma_i^2 / (n - 1)$", sv.iter().take(dims).map(|v| num(v * v / div, 3)).collect::<Vec<_>>().join(", "));
        ui::note(ui, r"From the singular values $\sigma$ of the centred data: the same numbers as the $\lambda$'s.");
        if self.plane {
            let pc = misses(&self.data, s.mean, s.axes[0]);
            ui::readout_colored(ui, "PC1's angle", format!("{}°", num(angle_of(s.axes[0]), 1)), PC[0]);
            match regression(&s) {
                Some(r) => {
                    ui::readout_colored(ui, "regression line's angle", format!("{}°", num(angle_of(r), 1)), REG);
                    let (right, up) = misses(&self.data, s.mean, r);
                    ui::readout(ui, "right-angle misses$^2$, PC1 / regression", format!("{} / {}", num(pc.0, 1), num(right, 1)));
                    ui::readout(ui, "up-and-down misses$^2$, PC1 / regression", format!("{} / {}", pc.1.map_or("—".into(), |v| num(v, 1)), up.map_or("—".into(), |v| num(v, 1))));
                }
                None => ui::note(ui, "Every $x$ is the same: there is no regression line of $y$ on $x$."),
            }
        } else {
            let k = self.keep;
            ui::readout(ui, "share kept", percent(s.kept(k)));
            ui::readout_colored(ui, "sum of squared misses", num(s.lost(k), 2), MISS);
            ui::readout(ui, "sum of squared distances from the mean", num(s.whole(), 2));
            if let Some(g) = self.without_far() {
                ui::readout_colored(ui, "plane's tilt, from the one without the far point", format!("{}°", num(g.axes[2].dot(s.axes[2]).abs().min(1.0).acos().to_degrees(), 1)), FAR);
            }
            ui::note(ui, "The share lost is the sum of squared misses over the sum of squared distances from the mean.");
        }
        ui::see_also(ui, &[("lowrank", "keeping the largest pieces of a matrix"), ("gauss2", "principal axes of a bell curve"), ("regress", "least-squares fitting"), ("sae", "directions that need not be at right angles")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt);
        // (the axes follow by the clock: they answer the hand too)
        self.follow(f.real_dt);

        // the panels' places follow the canvas, and the picture's place follows theirs
        let view = f.view_size();
        let parts = self.caption_parts();
        let lay = Layout::of(view, f.top_inset, self.plane, &parts);
        let before = Layout::of(self.canvas, self.inset, self.plane, &parts);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        // (and the application's buttons, which come and go over the top of the picture)
        let changed = !covered
            && (lay.narrow != before.narrow || lay.area.is_some() != before.area.is_some() || (view - self.canvas).abs().max_element() > 24.0 || f.top_inset != self.inset);
        let first = !covered && !self.measured;
        self.new_view |= changed || (first && (view - self.canvas).abs().max_element() > 24.0);
        if !covered && (changed || first) {
            (self.canvas, self.inset) = (view, f.top_inset);
            self.measured = true;
        }
        let pointer = f.ptr.pos.map(|p| p / f.px);
        if pointer.is_some_and(|p| lay.covers(p)) {
            // the panels are for their buttons: no turning the view from there, and two
            // quick clicks are two presses, not a reset of the view
            f.handles.capture();
            f.handles.claim_clicks();
        }

        let mut room = Room::new(f);
        if let Some((min, max)) = lay.area {
            room.wall(min, max);
        }
        if self.plane {
            self.flat_picture(f, &lay, &mut room);
            ticks(f, lay.picture);
        } else {
            self.space(f, &lay, &mut room);
        }

        // the panels
        if lay.narrow {
            if let Some((min, max)) = lay.area {
                hud_panel(f, min, max);
            }
        }
        if let Some(rect) = lay.buttons {
            self.buttons(f, rect, pointer);
        }
        if let Some(rect) = lay.bars {
            self.bars(f, rect);
        }
        if let Some(rect) = lay.plot {
            let drawn = self.drawn();
            self.plot(f, rect, &drawn);
        }
        if let Some(rect) = lay.matrix {
            self.matrix(f, rect);
        }
        if let Some(rect) = lay.table {
            self.table(f, rect);
        }
        if let Some(rect) = lay.chart {
            self.chart(f, rect);
        }
        let (min, max) = lay.picture;
        caption(f, &parts, min.x, max.x - min.x, max.y + caption_room(&parts, max.x - min.x, view.y) - 12.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Play the scene to rest, at `hz` frames a second; false if it never got there.
    fn settle_at(s: &mut Principal, hz: f32) -> bool {
        for _ in 0..(hz * 20.0) as usize {
            if !s.moving() {
                return true;
            }
            s.advance(1.0 / hz);
            s.follow(1.0 / hz);
        }
        !s.moving()
    }

    fn settle(s: &mut Principal) {
        assert!(settle_at(s, 60.0), "still moving after twenty seconds");
    }

    /// An experiment as its button sets it up (from the scene as it opens), played to rest.
    fn run(k: usize) -> Principal {
        let mut s = Principal::create();
        s.set_up(k);
        settle(&mut s);
        s
    }

    /// A cloud, played to rest.
    fn cloud(cloud: Cloud, seed: u64, plane: bool) -> Principal {
        let mut s = Principal::create();
        s.set_cloud(cloud, seed, plane);
        settle(&mut s);
        s
    }

    /// The angle in degrees between two lines (the sign of a direction means nothing).
    fn degrees(a: DVec3, b: DVec3) -> f64 {
        a.normalize().dot(b.normalize()).abs().min(1.0).acos().to_degrees()
    }

    fn times(c: [[f64; 3]; 3], v: DVec3) -> DVec3 {
        DVec3::new(c[0][0] * v.x + c[0][1] * v.y + c[0][2] * v.z, c[1][0] * v.x + c[1][1] * v.y + c[1][2] * v.z, c[2][0] * v.x + c[2][1] * v.y + c[2][2] * v.z)
    }

    /// "They are the eigenvectors of the covariance matrix, and the variances along
    /// them its eigenvalues; the right singular vectors of the centred data, and σ² /
    /// (n − 1)": for every cloud, in space and in the plane, and several draws.
    #[test]
    fn the_components_are_eigenvectors_and_singular_vectors() {
        for plane in [false, true] {
            for (kind, _) in CLOUDS {
                for seed in [SEED, 8, 31] {
                    let s = cloud(kind, seed, plane);
                    let st = s.stats;
                    let scale = st.var[0];
                    let (sv, right) = singular(&s.data, st.mean);
                    let div = (N - 1) as f64;
                    for i in 0..3 {
                        let v = st.axes[i];
                        assert!((v.length() - 1.0).abs() < 1e-12 && (times(st.cov, v) - v * st.var[i]).length() < 1e-9 * scale, "{kind:?} {seed} {plane}: PC{}", i + 1);
                        assert!(st.var[i] >= 0.0 && (i == 0 || st.var[i] <= st.var[i - 1]));
                        // the variance along it, measured from the points themselves
                        let along: f64 = s.data.iter().map(|p| (*p - st.mean).dot(v).powi(2)).sum::<f64>() / div;
                        assert!((along - st.var[i]).abs() < 1e-9 * scale);
                        assert!((sv[i] * sv[i] / div - st.var[i]).abs() < 1e-9 * scale, "{kind:?} {seed} {plane}: σ{}² / (n − 1) = {} against λ = {}", i + 1, sv[i] * sv[i] / div, st.var[i]);
                        // (a direction is only fixed where its variance is not shared)
                        let apart = [i.checked_sub(1), Some(i + 1).filter(|&j| j < 3)].iter().flatten().all(|&j| (st.var[j] - st.var[i]).abs() > 1e-3 * scale);
                        if apart {
                            assert!(right[i].dot(v).abs() > 1.0 - 1e-9, "{kind:?} {seed} {plane}: right singular vector {} is {:?}, not {v:?}", i + 1, right[i]);
                        }
                    }
                    assert!((st.axes[0].cross(st.axes[1]) - st.axes[2]).length() < 1e-12 && st.axes[0].dot(st.axes[1]).abs() < 1e-12);
                    assert!((st.shares().iter().sum::<f64>() - 1.0).abs() < 1e-12);
                    if plane {
                        assert!(st.var[2] == 0.0 && (st.axes[2].z.abs() - 1.0).abs() < 1e-12 && st.axes[0].z == 0.0 && st.axes[1].z == 0.0);
                    }
                }
            }
        }
    }

    /// "Keeping the first two (or the first) is the best flattening": the misses drawn
    /// are at right angles to what is kept; each point's squared distance from the mean
    /// is its shadow's plus its miss squared (Pythagoras), so what is kept and what is
    /// lost add up to the whole; and no plane or line, through the mean or anywhere
    /// else, has a smaller sum of squared distances to the points than the one kept.
    #[test]
    fn no_plane_or_line_loses_less() {
        let mut rng = Rng::new(3);
        let unit = |rng: &mut Rng| DVec3::new(rng.normal() as f64, rng.normal() as f64, rng.normal() as f64).normalize();
        for (kind, _) in CLOUDS {
            let mut s = cloud(kind, SEED, false);
            for k in 1..=3 {
                s.set_keep(k);
                s.flat = 1.0;
                settle(&mut s);
                let (st, drawn) = (s.stats, s.drawn());
                let m = st.mean - s.shift();
                let whole: f64 = drawn.iter().map(|d| (d.2 - m).length_squared()).sum();
                let kept: f64 = drawn.iter().map(|d| (d.0 - m).length_squared()).sum();
                let lost: f64 = drawn.iter().map(|d| (d.2 - d.0).length_squared()).sum();
                assert!((whole - st.whole()).abs() < 1e-9 * whole && (lost - st.lost(k)).abs() < 1e-9 * whole, "{kind:?}, keeping {k}: lost {lost} against {}", st.lost(k));
                assert!((kept + lost - whole).abs() < 1e-9 * whole, "{kind:?}, keeping {k}: {kept} + {lost} is not {whole}");
                assert!((kept / whole - st.kept(k)).abs() < 1e-12, "{kind:?}: the share kept");
                for d in &drawn {
                    assert!(d.0 == d.1, "flattened, every point is at its shadow");
                    for j in 0..k {
                        assert!((d.2 - d.1).dot(st.axes[j]).abs() < 1e-9 * whole.sqrt(), "{kind:?}: a miss is not at right angles to PC{}", j + 1);
                    }
                }
                if k == 3 {
                    assert!(lost < 1e-20 * whole);
                    continue;
                }
                // planes and lines anywhere near the cloud, in any direction
                for trial in 0..2000 {
                    let centre = st.mean + unit(&mut rng) * (st.var[0].sqrt() * rng.f32() as f64 * 2.0);
                    // (half of them only a little turned from the best)
                    let near = trial % 2 == 0;
                    let tilt = |rng: &mut Rng, v: DVec3| if near { (v + unit(rng) * 0.05).normalize() } else { unit(rng) };
                    let u1 = tilt(&mut rng, st.axes[0]);
                    let t2 = tilt(&mut rng, st.axes[1]);
                    let u2 = (t2 - u1 * u1.dot(t2)).normalize_or_zero();
                    let u2 = if u2 == DVec3::ZERO { u1.any_orthonormal_vector() } else { u2 };
                    let other: f64 = s.data.iter().map(|p| {
                        let r = *p - centre;
                        r.length_squared() - r.dot(u1).powi(2) - if k == 2 { r.dot(u2).powi(2) } else { 0.0 }
                    }).sum();
                    assert!(other >= st.lost(k) * (1.0 - 1e-12), "{kind:?}, keeping {k}: another loses {other}, less than {}", st.lost(k));
                }
            }
        }
    }

    /// In the plane: the first component's line has the least sum of squared misses at
    /// right angles of all lines through the mean, and the regression line the least
    /// measured up and down; the chart's curve is that sum; for every cloud, and for
    /// random clouds stretched and turned every way.
    #[test]
    fn each_line_has_the_least_of_its_own_kind_of_miss() {
        let mut rng = Rng::new(11);
        let mut sets: Vec<Vec<DVec3>> = CLOUDS.iter().map(|(kind, _)| cloud(*kind, SEED, true).data).collect();
        for _ in 0..40 {
            let (a, b, turn) = (rng.range(0.2, 3.0) as f64, rng.range(0.05, 2.0) as f64, rng.range(0.0, 6.3) as f64);
            let (sn, cs) = turn.sin_cos();
            sets.push((0..80).map(|_| {
                let (x, y) = (rng.normal() as f64 * a, rng.normal() as f64 * b);
                DVec3::new(cs * x - sn * y + 1.0, sn * x + cs * y - 2.0, 0.0)
            }).collect());
        }
        for data in sets {
            let st = Stats::of(&data, true, [DVec3::X, DVec3::Y, DVec3::Z]);
            let reg = regression(&st).expect("x varies");
            let (pc_right, _) = misses(&data, st.mean, st.axes[0]);
            let (_, reg_up) = misses(&data, st.mean, reg);
            let reg_up = reg_up.expect("the regression line is never upright");
            assert!((pc_right - st.lost(1)).abs() < 1e-9 * st.whole() && (misses_at(&st, angle_of(st.axes[0])) - pc_right).abs() < 1e-9 * st.whole());
            for tenth in 0..1800 {
                let deg = tenth as f64 * 0.1;
                let dir = DVec3::new(deg.to_radians().cos(), deg.to_radians().sin(), 0.0);
                let (right, up) = misses(&data, st.mean, dir);
                assert!((misses_at(&st, deg) - right).abs() < 1e-9 * st.whole(), "the chart at {deg}°");
                assert!(right >= pc_right * (1.0 - 1e-12), "a line at {deg}° has fewer right-angle misses than PC1");
                if let Some(up) = up {
                    assert!(up >= reg_up * (1.0 - 1e-12), "a line at {deg}° has fewer up-and-down misses than the regression line");
                }
            }
        }
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // flatten a pancake: every point ends at its shadow on the plane, the two
        // components keep 98.7% (the squared misses come to 1.3%), and the points in
        // the PCA plot have not moved
        let s = run(0);
        let st = s.stats;
        assert!(s.cloud == Cloud::Pancake && !s.plane && s.keep == 2 && s.flat == 1.0 && s.units == 1.0);
        assert_eq!((percent(st.kept(2)), percent(st.lost(2) / st.whole())), ("98.7%".to_string(), "1.3%".to_string()));
        let m = st.mean - s.shift();
        for (now, shadow, at) in s.drawn() {
            assert!(now == shadow && (now - at).dot(st.axes[0]).abs() < 1e-12 && (now - at).dot(st.axes[1]).abs() < 1e-12);
            assert!(((now - m).dot(st.axes[0]) - (at - m).dot(st.axes[0])).abs() < 1e-12 && ((now - m).dot(st.axes[1]) - (at - m).dot(st.axes[1])).abs() < 1e-12);
        }

        // a ball: each share near a third (37%, 33%, 30%); another draw of the same
        // ball turns PC1 by 57°, where a pancake's turns by 5°
        let mut s = run(1);
        let whole: Vec<String> = s.stats.shares().iter().map(|v| num(100.0 * v, 0)).collect();
        assert_eq!(whole, ["37", "33", "30"]);
        let before = s.stats.axes[0];
        s.set_cloud(Cloud::Ball, SEED + 1, false);
        settle(&mut s);
        let ball = degrees(before, s.stats.axes[0]);
        let pancake = degrees(cloud(Cloud::Pancake, SEED, false).stats.axes[0], cloud(Cloud::Pancake, SEED + 1, false).stats.axes[0]);
        assert!(num(ball, 0) == "57" && num(pancake, 0) == "5", "{ball}° and {pancake}°");

        // change the units: x three times larger; PC1 swings by 50°, from 61° off the x
        // axis to 11°
        let (start, s) = (run(0).stats, run(2));
        assert!(s.units == 3.0 && s.units_now() == 3.0 && s.cloud == Cloud::Pancake && s.keep == 2);
        let (swing, from, to) = (degrees(start.axes[0], s.stats.axes[0]), degrees(start.axes[0], DVec3::X), degrees(s.stats.axes[0], DVec3::X));
        assert_eq!([swing, from, to].map(|d| num(d, 0)), ["50", "61", "11"]);

        // one far point: the plane tilts by 12° from the plane of the other 299; the share
        // lost grows from 1.3% to 7.4%; the far point's miss, squared, is more than
        // twice all the others' put together
        let s = run(3);
        let (st, without) = (s.stats, s.without_far().expect("a far point"));
        assert!(s.cloud == Cloud::Far && s.keep == 2);
        assert_eq!(num(degrees(st.axes[2], without.axes[2]), 0), "12");
        assert_eq!((percent(without.lost(2) / without.whole()), percent(st.lost(2) / st.whole())), ("1.3%".to_string(), "7.4%".to_string()));
        let miss = |p: DVec3| (p - st.mean).dot(st.axes[2]).powi(2);
        let ratio = miss(s.data[N - 1]) / s.data[..N - 1].iter().map(|p| miss(*p)).sum::<f64>();
        assert!(ratio > 2.0 && ratio < 2.5, "{ratio}");
        // (and without the far point the other 299 are the pancake as it was, near enough)
        assert!(degrees(without.axes[2], run(0).stats.axes[2]) < 0.1);

        // not the regression line: PC1 17° steeper; each line has the smaller sum of
        // its own kind of miss
        let s = run(4);
        let st = s.stats;
        let reg = regression(&st).expect("x varies");
        assert!(s.plane && s.cloud == Cloud::Pancake);
        assert_eq!(num(angle_of(st.axes[0]) - angle_of(reg), 0), "17");
        let (pc, rg) = (misses(&s.data, st.mean, st.axes[0]), misses(&s.data, st.mean, reg));
        assert!(pc.0 < rg.0 && rg.1.unwrap_or(f64::MAX) < pc.1.unwrap_or(0.0), "{pc:?} {rg:?}");
    }

    /// Every experiment shows the same thing whatever was done before it.
    #[test]
    fn an_experiment_does_not_depend_on_what_came_before() {
        for k in 0..TRIALS.len() {
            let alone = run(k);
            let mut s = Principal::create();
            s.set_up(4);
            s.set_cloud(Cloud::Clusters, 99, false);
            s.set_units(7.0);
            s.centred = true;
            s.set_keep(1);
            s.flat = 0.5;
            s.drag_far(DVec3::new(9.0, 9.0, 9.0));
            settle(&mut s);
            s.set_up(k);
            settle(&mut s);
            assert!(s.stats == alone.stats && s.data == alone.data && s.keep == alone.keep && s.flat == alone.flat && s.centring == alone.centring, "{}", TRIALS[k].0);
        }
    }

    /// Moving the mean to 0 moves the picture and nothing else.
    #[test]
    fn centring_changes_nothing_but_where_the_cloud_is() {
        let mut s = run(0);
        let before = s.stats;
        s.centred = true;
        settle(&mut s);
        assert!(s.stats == before && (s.stats.mean - s.shift()).length() == 0.0);
        let mean: DVec3 = s.drawn().iter().map(|d| d.0).sum::<DVec3>() / N as f64;
        assert!(mean.length() < 1e-12, "{mean}");
    }

    /// The drawn axes follow a change at any frame rate, stay a frame at right angles
    /// on the way, and come to rest exactly where they are computed to be.
    #[test]
    fn the_axes_glide_and_come_to_rest() {
        for hz in [30.0f32, 60.0, 144.0, 240.0] {
            for k in 0..TRIALS.len() {
                let mut s = Principal::create();
                s.set_up(k);
                for _ in 0..(hz * 0.4) as usize {
                    s.advance(1.0 / hz);
                    s.follow(1.0 / hz);
                    let a = s.axes();
                    assert!((a[0].cross(a[1]) - a[2]).length() < 1e-9 && s.spread.iter().all(|v| v.is_finite()));
                }
                assert!(settle_at(&mut s, hz), "{hz} Hz, {}: not at rest", TRIALS[k].0);
                assert!(s.frame == s.stats.frame() && s.spread == s.stats.var.map(f64::sqrt));
                // and at rest, nothing changes, frame after frame
                let (frame, kept, data) = (s.frame, s.kept, s.data.clone());
                for _ in 0..10 {
                    s.advance(1.0 / hz);
                    s.follow(1.0 / hz);
                    assert!(!s.moving() && s.frame == frame && s.kept == kept && s.data == data, "{hz} Hz, {}: it moves at rest", TRIALS[k].0);
                }
                s.refresh();
                s.follow(1.0 / hz);
                assert!(s.frame == frame && !s.moving());
            }
        }
    }

    /// The sign of an eigenvector means nothing: each is turned the way that keeps the
    /// frame nearest to the one drawn, so a small change of the data never flips an
    /// axis over.
    #[test]
    fn small_changes_never_flip_an_axis() {
        for (kind, _) in CLOUDS {
            let mut s = cloud(kind, SEED, false);
            for step in 0..200 {
                let before = s.stats.axes;
                s.set_units(1.0 + 0.002 * step as f32);
                s.catch_up();
                for i in 0..3 {
                    // (where two spreads are nearly the same their axes may well turn,
                    // but not over: the frame as a whole turns the least it can)
                    assert!(s.stats.axes[i].dot(before[i]) > -0.5, "{kind:?}: PC{} flipped at step {step}", i + 1);
                }
                let turn: f64 = (0..3).map(|i| s.stats.axes[i].dot(before[i])).sum();
                assert!(turn > 0.9, "{kind:?}: the frame turned over at step {step}");
            }
        }
    }

    /// Every point the same, every point on a line, every point in a plane, and a cloud
    /// whose three spreads are exactly equal: every number drawn or read out is finite,
    /// the shares say what happened, and the choice of axes stays put.
    #[test]
    fn degenerate_clouds_leave_everything_finite() {
        let ties: Vec<DVec3> = (0..N).map(|i| [DVec3::X, -DVec3::X, DVec3::Y, -DVec3::Y, DVec3::Z, -DVec3::Z][i % 6]).collect();
        let line: Vec<DVec3> = (0..N).map(|i| DVec3::new(i as f64 / N as f64 - 0.5, 0.0, 0.0)).collect();
        let cases: [(&str, Vec<DVec3>, f32, [f64; 3]); 5] = [
            ("the same point", vec![DVec3::ZERO; N], 1.0, [0.0; 3]),
            ("a line", line, 1.0, [1.0, 0.0, 0.0]),
            ("a plane (x in units of nothing)", draws(SEED), 0.0, [0.0, 0.0, 0.0]),
            ("three equal spreads", ties.clone(), 1.0, [1.0 / 3.0; 3]),
            ("a ball", draws(SEED), 1.0, [0.0; 3]),
        ];
        for (name, z, units, want) in cases {
            for plane in [false, true] {
                let mut s = Principal::create();
                s.draws = z.clone();
                s.set_cloud(Cloud::Ball, SEED, plane);
                s.set_units(units);
                settle(&mut s);
                let st = s.stats;
                let shares = st.shares();
                assert!(shares.iter().all(|v| v.is_finite() && *v >= 0.0) && st.axes.iter().all(|a| a.is_finite()), "{name}, {plane}");
                let total: f64 = shares.iter().sum();
                assert!(total == 0.0 || (total - 1.0).abs() < 1e-12, "{name}: the shares add up to {total}");
                if name == "the same point" {
                    assert!(!st.spread_out() && total == 0.0 && st.lost(2) == 0.0);
                } else if name == "a line" && !plane {
                    assert!((shares[0] - 1.0).abs() < 1e-12 && st.var[1] == 0.0 && st.var[2] == 0.0);
                } else if name.starts_with("a plane") && !plane {
                    assert!(st.var[2] == 0.0 && (st.axes[2].x.abs() - 1.0).abs() < 1e-12, "{name}: {st:?}");
                } else if name == "three equal spreads" && !plane {
                    assert!(shares.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-12), "{shares:?}");
                }
                // in the plane, x in units of nothing leaves no regression line, and PC1
                // upright, with no misses up and down
                if plane && units == 0.0 && st.spread_out() {
                    assert!(regression(&st).is_none() && misses(&s.data, st.mean, st.axes[0]).1.is_none());
                }
                for d in s.drawn() {
                    assert!(d.0.is_finite() && d.1.is_finite() && d.2.is_finite(), "{name}");
                }
                for (canvas, inset) in [(Vec2::new(1078.0, 1000.0), 0.0), (Vec2::new(390.0, 800.0), 46.0), (Vec2::new(240.0, 180.0), 46.0)] {
                    (s.canvas, s.inset) = (canvas, inset);
                    match s.the_view() {
                        View::Orbit { target, dist, .. } => assert!(target.is_finite() && dist.is_finite() && dist > 0.0, "{name}"),
                        View::Fit { center, half } => assert!(center.is_finite() && half.is_finite() && half.min_element() > 0.0, "{name}"),
                        View::Flat { .. } => panic!("not a view this scene makes"),
                    }
                }
                // the same data again: the same axes, exactly (no jitter from frame to frame)
                let axes = st.axes;
                for _ in 0..3 {
                    s.refresh();
                    assert!(s.stats.axes == axes, "{name}: the axes moved");
                }
            }
        }
    }

    /// The far point stays within its reach wherever it is dragged, and a drag with x
    /// in units of nothing (where where it is along x cannot be told) leaves x alone.
    #[test]
    fn the_far_point_stays_in_reach() {
        let mut s = run(3);
        for p in [DVec3::new(1e6, -1e6, 1e6), DVec3::new(0.0, 0.0, 0.0), DVec3::new(f64::NAN, 0.0, 0.0), DVec3::new(-3.0, 4.0, 20.0)] {
            s.drag_far(p);
            assert!(s.far.is_finite() && (s.far - OFFSET).length() <= FAR_REACH + 1e-9 && s.stats.var.iter().all(|v| v.is_finite()), "{p}: {:?}", s.far);
        }
        let x = s.far.x;
        s.set_units(0.0);
        s.drag_far(DVec3::new(5.0, 1.0, 2.0));
        assert!(s.far.x == x && s.far.is_finite());
    }

    /// Dragging the far point, a little in each frame, moves the axes and the plane a
    /// little in each frame: they follow the hand without a jump or a flip, in space and
    /// in the plane, and the tilt grows as the point goes further out.
    #[test]
    fn the_axes_follow_the_far_point_smoothly() {
        for plane in [false, true] {
            let mut s = run(3);
            if plane {
                s.set_cloud(Cloud::Far, SEED, true);
                settle(&mut s);
            }
            let start = s.far;
            let goal = offset(plane) + (start - offset(plane)) * 0.3;
            let without = s.without_far().expect("a far point").axes;
            let mut tilts = Vec::new();
            for step in 0..=240 {
                let before = s.axes();
                s.drag_far(start.lerp(goal, step as f64 / 240.0));
                s.advance(1.0 / 60.0);
                s.follow(1.0 / 60.0);
                let after = s.axes();
                for i in 0..3 {
                    assert!(after[i].dot(before[i]) > 0.999, "{plane}: PC{} turned by {}° in one frame at step {step}", i + 1, degrees(after[i], before[i]));
                }
                tilts.push(degrees(s.stats.axes[if plane { 0 } else { 2 }], without[if plane { 0 } else { 2 }]));
            }
            settle(&mut s);
            // nearer the cloud, the far point pulls less
            assert!(tilts.windows(2).all(|w| w[1] <= w[0] + 1e-9) && tilts[tilts.len() - 1] < 0.5 * tilts[0], "{plane}: {} then {}", tilts[0], tilts[tilts.len() - 1]);
        }
    }

    /// The panels keep to every canvas, clear of the application's buttons and of each
    /// other, and leave the picture some room; their lines of text fit across them.
    #[test]
    fn the_panels_fit_every_canvas() {
        let parts = Principal::create().caption_parts();
        for (x, y) in [(1078.0f32, 1000.0f32), (1600.0, 1000.0), (580.0, 700.0), (390.0, 800.0), (360.0, 640.0), (844.0, 390.0), (600.0, 400.0), (240.0, 180.0)] {
            let inset = if x < 900.0 { 46.0 } else { 0.0 };
            for plane in [false, true] {
                let lay = Layout::of(Vec2::new(x, y), inset, plane, &parts);
                let rects: Vec<Rect> = [lay.buttons, lay.bars, lay.plot, lay.matrix, lay.table, lay.chart].into_iter().flatten().collect();
                for (i, (min, max)) in rects.iter().enumerate() {
                    assert!(min.x >= 0.0 && min.y >= inset && max.x <= x && max.y <= y && max.x > min.x && max.y > min.y, "{x}x{y} {plane}: {min} {max}");
                    for (other_min, other_max) in &rects[i + 1..] {
                        assert!(max.x <= other_min.x || other_max.x <= min.x || max.y <= other_min.y || other_max.y <= min.y, "{x}x{y} {plane}: panels overlap");
                    }
                    assert!(lay.area.is_some_and(|(a, b)| min.cmpge(a).all() && max.cmple(b).all()), "{x}x{y}: a panel outside its area");
                }
                let (pmin, pmax) = lay.picture;
                assert!(pmax.x - pmin.x >= 200.0 && pmax.y - pmin.y >= 100.0, "{x}x{y} {plane}: {pmin} {pmax}");
                if (x, y) == (1078.0, 1000.0) {
                    assert!(if plane { lay.table.is_some() && lay.chart.is_some() } else { rects.len() == 4 });
                }
                if (x, y) == (390.0, 800.0) {
                    assert!(lay.narrow && if plane { lay.table.is_some() && lay.chart.is_some() } else { lay.buttons.is_some() && lay.bars.is_some() && lay.plot.is_some() });
                }
                // the widest lines the panels write, with long numbers in them
                if let Some((min, max)) = lay.bars {
                    let s = run(0).stats;
                    for k in 1..=3 {
                        let line = kept_line(&s, k, max.x - min.x - 4.0);
                        assert!(text_width(&line, 11.5) <= max.x - min.x - 4.0 && line.contains("kept"), "{x}x{y}: {line}");
                    }
                    assert!(text_width("kept: 100.0%   lost: 100.0%", 11.5) <= max.x - min.x - 4.0, "{x}x{y}");
                }
                if let Some((min, max)) = lay.matrix {
                    assert!(text_width("eigenvalues: 1234.57, 1234.57, 1234.57", 11.5) <= max.x - min.x && matrix_size(3, 3, 8, 12.5).x + 34.0 <= max.x - min.x, "{x}x{y}");
                }
                if let Some((min, max)) = lay.buttons {
                    assert!(40.0 + 3.0 * 34.0 + 6.0 + text_width("◀ unflatten", 12.5) + 10.0 <= max.x - min.x, "{x}x{y}");
                }
                if let Some((min, max)) = lay.table {
                    assert!(text_width("at right angles", 12.0) + 4.0 < 0.62 * (max.x - min.x) - 30.0, "{x}x{y}");
                }
            }
        }
    }

    /// Where every point the view is fitted to lands on the screen, for a scene laid
    /// out on `canvas` with `inset` of its top under the application's buttons: the
    /// smallest and largest screen positions.
    fn seen(s: &Principal, canvas: Vec2, inset: f32) -> (Vec2, Vec2) {
        let mut s = Principal { canvas, inset, ..Principal::create() }.with(s);
        s.catch_up();
        let mut cam = crate::gfx::camera::Camera::new(s.the_view());
        cam.update(0.0, canvas);
        let seen: Vec<Vec2> = s.home().iter().filter_map(|p| cam.project(p.as_vec3())).collect();
        (seen.iter().fold(Vec2::MAX, |a, b| a.min(*b)), seen.iter().fold(Vec2::MIN, |a, b| a.max(*b)))
    }

    impl Principal {
        /// This scene with the data of another.
        fn with(mut self, other: &Principal) -> Principal {
            (self.cloud, self.plane, self.units, self.centred, self.far, self.seed, self.draws) = (other.cloud, other.plane, other.units, other.centred, other.far, other.seed, other.draws.clone());
            self.refresh();
            self
        }
    }

    /// Every experiment's picture is in the part of the canvas the panels leave free,
    /// on a desktop, on a phone, and in windows between the two.
    #[test]
    fn the_picture_fits_its_part_of_the_canvas() {
        for (canvas, inset) in [(Vec2::new(1078.0, 1000.0), 0.0), (Vec2::new(390.0, 800.0), 46.0), (Vec2::new(580.0, 700.0), 0.0), (Vec2::new(844.0, 390.0), 46.0), (Vec2::new(1600.0, 1000.0), 46.0)] {
            for k in 0..TRIALS.len() {
                let s = run(k);
                let lay = Layout::of(canvas, inset, s.plane, &s.caption_parts());
                let (lo, hi) = seen(&s, canvas, inset);
                let (min, max) = lay.picture;
                assert!(lo.x >= min.x - 1.0 && lo.y >= min.y - 1.0 && hi.x <= max.x + 1.0 && hi.y <= max.y + 1.0, "{canvas}, {}: seen {lo} .. {hi}, picture {min} .. {max}", TRIALS[k].0);
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            let st = s.stats;
            println!("{}: shares {:?} kept {} lost {} var {:?}", TRIALS[k].0, st.shares(), st.kept(s.keep), st.lost(s.keep), st.var);
        }
        let start = run(0).stats;
        let units = run(2).stats;
        println!("units: PC1 swing {}°; off x before {} after {}; shares after {:?}", degrees(start.axes[0], units.axes[0]), degrees(start.axes[0], DVec3::X), degrees(units.axes[0], DVec3::X), units.shares());
        let far = run(3);
        let (st, without) = (far.stats, far.without_far().unwrap());
        let miss = |p: DVec3| (p - st.mean).dot(st.axes[2]).powi(2);
        let others: f64 = far.data[..N - 1].iter().map(|p| miss(*p)).sum();
        println!("far: tilt {}°, far miss² {} others {}; lost {} without {}", degrees(st.axes[2], without.axes[2]), miss(far.data[N - 1]), others, st.lost(2) / st.whole(), without.lost(2) / without.whole());
        for (kind, name) in [(Cloud::Ball, "ball"), (Cloud::Pancake, "pancake"), (Cloud::Cigar, "cigar")] {
            let turns: Vec<String> = (SEED..SEED + 5).map(|seed| format!("{:.1}", degrees(cloud(kind, seed, false).stats.axes[0], cloud(kind, seed + 1, false).stats.axes[0]))).collect();
            println!("{name}: PC1 turns by {} between draws", turns.join(", "));
        }
        let p = run(4);
        let st = p.stats;
        let r = regression(&st).unwrap();
        println!("plane: PC1 at {}°, regression at {}°; PC1's misses {:?}, the regression line's {:?}", angle_of(st.axes[0]), angle_of(r), misses(&p.data, st.mean, st.axes[0]), misses(&p.data, st.mean, r));
    }
}

//! Optimisers on a loss surface.

use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "descent",
    title: "Optimisers on a loss surface",
    group: "Calculus",
    blurb: "Seven optimisers, nineteen losses, each chosen to show one thing about how they behave.",
    make: |g| Box::new(Descent::new(g)),
};

pub const NAMES: [&str; 7] = ["gradient descent", "momentum", "Nesterov", "AdaGrad", "RMSProp", "Adam", "Newton"];
/// The first six as in the race of optimisers (`neural/race.rs` takes them from
/// `pal::SERIES` in the same order), so that a rule keeps its colour between the two
/// scenes; Newton's method, which only this scene has, in white.
const COLORS: [Color; 7] = [pal::SERIES[0], pal::SERIES[1], pal::SERIES[2], pal::SERIES[3], pal::SERIES[4], pal::SERIES[5], pal::WHITE];

const GD: u8 = 1;
const MOM: u8 = 2;
const NAG: u8 = 4;
const ADAGRAD: u8 = 8;
const RMS: u8 = 16;
const ADAM: u8 = 32;
const NEWTON: u8 = 64;

pub struct Preset {
    pub name: &'static str,
    pub group: &'static str,
    pub formula: &'static str,
    pub center: [f32; 2],
    pub extent: f32,
    pub start: [f32; 2],
    pub lr: f32,
    pub lr_adaptive: f32,
    pub momentum: f32,
    /// How strongly tall walls are compressed for display (0 = true height).
    pub squash: f32,
    /// Which optimisers are switched on by default.
    pub mask: u8,
    /// What to look for.
    pub note: &'static str,
    /// Variations to try while this loss is on show.
    pub tries: &'static [Try],
}

/// A variation on a loss: one of the "things to try" while it is on show. Its
/// button sets the loss up afresh and then makes this one change.
pub struct Try {
    pub label: &'static str,
    pub look_for: &'static str,
    pub set: fn(&mut Sim),
}

impl Preset {
    /// The optimisers as this loss sets them up (`expr` is its formula, compiled),
    /// with variation `k` of its `tries` made if one is given: what its buttons do,
    /// and what the tests check.
    pub fn sim(&self, expr: Expr, k: Option<usize>) -> Sim {
        let mut sim = Sim::new(expr, Vec2::from(self.center), self.extent, Vec2::from(self.start), settings_of(self), self.mask);
        if let Some(t) = k.and_then(|k| self.tries.get(k)) {
            (t.set)(&mut sim);
            sim.restart();
        }
        sim
    }
}

pub const PRESETS: [Preset; 19] = [
    Preset {
        name: "round bowl",
        group: "Quadratic bowls",
        formula: "0.5 (x^2 + y^2)",
        center: [0.0, 0.0],
        extent: 3.0,
        start: [-2.4, 1.8],
        lr: 0.15,
        lr_adaptive: 0.15,
        momentum: 0.9,
        squash: 2.0,
        mask: GD | MOM | ADAM | NEWTON,
        note: "The friendliest loss there is. Gradient descent walks straight to the minimum, each step a fixed fraction of the way. Momentum overshoots and swings back, like a real ball in a real bowl. Newton's method arrives in one step, because a quadratic is exactly what it assumes. The learning rate is limited by the curvature (how fast the slope changes), which is 1 here: above 2 divided by the curvature, each step overshoots further than the last.",
        tries: &[
            Try { label: "learning rate 1", look_for: "One divided by the curvature: gradient descent lands on the minimum in a single step, as Newton does.", set: |s| s.settings.lr = 1.0 },
            Try { label: "learning rate 1.9", look_for: "Just under the limit: every step jumps across the minimum to the other side, a little closer each time.", set: |s| s.settings.lr = 1.9 },
            Try { label: "learning rate 2.1", look_for: "Just over the limit: every jump lands further out than the last, and gradient descent leaves the map within three steps.", set: |s| s.settings.lr = 2.1 },
        ],
    },
    Preset {
        name: "ravine",
        group: "Quadratic bowls",
        formula: "0.05 x^2 + y^2",
        center: [0.0, 0.0],
        extent: 3.0,
        start: [-2.6, 1.6],
        lr: 0.9,
        lr_adaptive: 0.25,
        momentum: 0.5,
        squash: 4.0,
        mask: GD | MOM | ADAGRAD | ADAM,
        note: "Steep across, shallow along. The step size is limited by the steep direction, so gradient descent bounces from wall to wall while creeping along the floor. Momentum cancels the bouncing and builds speed along the floor: here it needs about a third of the steps. AdaGrad and Adam divide each coordinate by its own typical gradient, so they set off diagonally as if the valley were round; Adam then takes a while to come to rest, because its step does not shrink as the gradient does.",
        tries: &[Try {
            label: "learning rate 0.5",
            look_for: "A single step now lands on the valley floor and the bouncing is gone, but the crawl along the floor is slower still: gradient descent needs about 130 steps instead of 72.",
            set: |s| s.settings.lr = 0.5,
        }],
    },
    Preset {
        name: "rotated ravine",
        group: "Quadratic bowls",
        formula: "0.025 (x + y)^2 + 0.5 (x - y)^2",
        center: [0.0, 0.0],
        extent: 3.4,
        start: [-0.71, -2.97],
        lr: 0.9,
        lr_adaptive: 0.25,
        momentum: 0.5,
        squash: 4.0,
        mask: GD | MOM | ADAGRAD | ADAM,
        note: "The same ravine turned by 45 degrees. Gradient descent and momentum take exactly as many steps as before: they do not care how the axes are drawn. The coordinate-wise methods do. Scaling each coordinate separately only straightens a valley that is lined up with the coordinates; watch AdaGrad, quick on the previous loss, need several times as many steps here.",
        tries: &[],
    },
    Preset {
        name: "narrow ravine",
        group: "Quadratic bowls",
        formula: "0.01 x^2 + y^2",
        center: [0.0, 0.0],
        extent: 3.0,
        start: [-2.7, 1.2],
        lr: 0.95,
        lr_adaptive: 0.25,
        momentum: 0.8,
        squash: 4.0,
        mask: GD | MOM | ADAM | NEWTON,
        note: "A hundred times more curved across than along. That ratio is the condition number, and it is what makes optimisation slow: the steep direction caps the step size, and at that step size the shallow direction needs hundreds of iterations. Momentum needs about a quarter as many. Newton ignores the ratio entirely.",
        tries: &[],
    },
    Preset {
        name: "Rosenbrock",
        group: "Curved valleys",
        formula: "(1 - x)^2 + 10 (y - x^2)^2",
        center: [0.0, 1.0],
        extent: 2.2,
        start: [-1.6, 2.6],
        lr: 0.006,
        lr_adaptive: 0.08,
        momentum: 0.9,
        squash: 40.0,
        mask: GD | MOM | ADAM | NEWTON,
        note: "The classic banana valley. Falling into it is easy; following it round the bend to the minimum at $(1, 1)$ is slow, because the direction of the floor keeps changing and the walls punish any step that does not turn with it. Momentum carries speed round the bend. Newton reads the curvature and needs only a handful of steps.",
        tries: &[],
    },
    Preset {
        name: "winding valley",
        group: "Curved valleys",
        formula: "3 (y - sin(1.6 x))^2 + 0.04 (x - 2.5)^2",
        center: [0.0, 0.0],
        extent: 3.2,
        start: [-2.9, 1.5],
        lr: 0.06,
        lr_adaptive: 0.1,
        momentum: 0.9,
        squash: 6.0,
        mask: GD | MOM | NAG | ADAM,
        note: "A river that winds. The floor slopes only gently, so progress depends on keeping speed, but every bend punishes speed. Gradient descent is still creeping along long after the others have arrived. Watch momentum bank up the outside wall of each turn. Nesterov differs from it in one thing: it measures the slope at the point its momentum is about to carry it to.",
        tries: &[],
    },
    Preset {
        name: "Beale",
        group: "Curved valleys",
        formula: "(1.5 - x + x y)^2 + (2.25 - x + x y^2)^2 + (2.625 - x + x y^3)^2",
        center: [1.5, 0.0],
        extent: 3.0,
        start: [0.5, -1.6],
        lr: 0.006,
        lr_adaptive: 0.1,
        momentum: 0.9,
        squash: 300.0,
        mask: GD | MOM | ADAGRAD | ADAM,
        note: "Steep walls, a flat plain and a curved canyon in one function. The canyon floor leading to the minimum at $(3, 0.5)$ is so flat that fixed-step gradient descent appears to stop long before arriving: the loss chart shows it still falling, just absurdly slowly (it needs thousands of steps). Momentum and Adam keep a usable pace the whole way.",
        tries: &[],
    },
    Preset {
        name: "two weights multiplied",
        group: "Curved valleys",
        formula: "(x y - 1)^2",
        center: [0.0, 0.0],
        extent: 3.0,
        start: [0.15, 2.6],
        lr: 0.2,
        lr_adaptive: 0.1,
        momentum: 0.5,
        squash: 3.0,
        mask: GD | ADAM,
        note: "The smallest possible deep network: two weights multiplied together and asked to produce 1. Every point on the curve $xy = 1$ is a perfect solution, but the valley is sharp where one weight is large and flat where the two are balanced. With learning rate 0.2 gradient descent cannot stay in the sharp part: it bounces along until the valley is flat enough for its step. Large steps choose flat solutions.",
        tries: &[Try {
            label: "learning rate 0.05",
            look_for: "A smaller step can stay in the sharp part: gradient descent stops where it first reaches the valley, near $(0.4, 2.6)$, instead of near $(0.8, 1.3)$.",
            set: |s| s.settings.lr = 0.05,
        }],
    },
    Preset {
        name: "saddle",
        group: "Saddles and flats",
        formula: "x^2 - y^2 + 0.25 y^4 + 1",
        center: [0.0, 0.0],
        extent: 2.6,
        start: [2.4, 0.00001],
        lr: 0.08,
        lr_adaptive: 0.1,
        momentum: 0.9,
        squash: 6.0,
        mask: GD | MOM | ADAM | NEWTON,
        note: "A mountain pass. Coming down the ridge the gradient shrinks towards zero at the pass, though it is not a minimum, and gradient descent nearly stalls before it finds the way down the side. Adam's normalised steps hardly notice. Newton heads straight for the pass and stays: it looks for zero gradient, not for minima.",
        tries: &[Try {
            label: "gradient noise 0.1",
            look_for: "Random kicks knock gradient descent off the ridge long before it would have found its own way down, and momentum gets away sooner too. Adam, quick already, gains nothing, and Newton still stays on the pass.",
            set: |s| s.settings.noise = 0.1,
        }],
    },
    Preset {
        name: "plateau",
        group: "Saddles and flats",
        formula: "1 - exp(-0.5 (x^2 + 3 y^2))",
        center: [0.0, 0.0],
        extent: 3.0,
        start: [2.7, 1.35],
        lr: 0.5,
        lr_adaptive: 0.12,
        momentum: 0.9,
        squash: 0.0,
        mask: GD | MOM | RMS | ADAM,
        note: "Nearly flat far from the minimum. Out there the gradient is tiny, so anything whose step is proportional to the gradient barely moves, while Adam and RMSProp, which respond to the gradient's direction and recent size rather than its absolute size, stride across at full speed. This is why adaptive methods forgive a bad starting point.",
        tries: &[],
    },
    Preset {
        name: "cliff",
        group: "Saddles and flats",
        formula: "0.05 (x^2 + y^2) + 4 sigmoid(12 x)",
        center: [0.0, 0.0],
        extent: 3.0,
        start: [2.5, 1.2],
        lr: 1.0,
        lr_adaptive: 0.15,
        momentum: 0.5,
        squash: 0.0,
        mask: GD | MOM | RMS | ADAM,
        note: "A gentle slope that ends in a cliff. A step size that suits the slope is far too large on the cliff face, where the gradient is a hundred times bigger, and plain gradient descent is thrown clean off the map. This is the exploding-gradient problem. The methods that normalise their step never had the problem.",
        tries: &[Try {
            label: "gradient clipping 0.5",
            look_for: "Every gradient longer than 0.5 is shortened to 0.5: gradient descent now walks down the cliff instead of being thrown off it, and so does momentum.",
            set: |s| s.settings.clip = 0.5,
        }],
    },
    Preset {
        name: "crease",
        group: "Saddles and flats",
        formula: "abs(x) + 0.15 y^2",
        center: [0.0, 0.0],
        extent: 2.5,
        start: [2.05, 1.8],
        lr: 0.3,
        lr_adaptive: 0.15,
        momentum: 0.5,
        squash: 0.0,
        mask: GD | MOM | ADAGRAD | ADAM,
        note: "A fold: the slope across it is the same size however close you get. A fixed step overshoots the crease for ever, hopping from side to side without settling, and the hop never shrinks because the gradient never does. AdaGrad, whose steps shrink on their own as it accumulates gradient, quietens down without help. ReLU networks have creases everywhere, which is one reason learning-rate schedules matter.",
        tries: &[Try {
            label: "step-size decay 0.05",
            look_for: "Each step is a little shorter than the one before, and now gradient descent, momentum and Adam close in on the crease too.",
            set: |s| s.settings.decay = 0.05,
        }],
    },
    Preset {
        name: "two wells",
        group: "Many minima",
        formula: "(x^2 - 1)^2 + 0.3 x + 0.5 y^2 + 0.4",
        center: [0.0, 0.0],
        extent: 2.3,
        start: [2.1, 1.0],
        lr: 0.02,
        lr_adaptive: 0.06,
        momentum: 0.95,
        squash: 5.0,
        mask: GD | MOM | ADAM,
        note: "Two valleys, the far one deeper. Gradient descent settles in the first one it meets and has no way of knowing a better one exists. Momentum arrives with speed to spare: it crosses the ridge, comes back, and crosses again before it settles in the deeper valley.",
        tries: &[
            Try { label: "momentum 0.6", look_for: "Too little speed to climb the ridge: momentum settles in the near valley, beside gradient descent.", set: |s| s.settings.momentum = 0.6 },
            Try {
                label: "momentum 0.9",
                look_for: "Over the ridge and back again, and this time it settles in the near valley. Where it ends depends on how many times it crosses, not simply on how much momentum it has.",
                set: |s| s.settings.momentum = 0.9,
            },
            Try { label: "momentum 0.99", look_for: "It crosses the ridge back and forth more than a dozen times and takes about 1,500 steps to settle, in the near valley.", set: |s| s.settings.momentum = 0.99 },
        ],
    },
    Preset {
        name: "Himmelblau",
        group: "Many minima",
        formula: "(x^2 + y - 11)^2 + (x + y^2 - 7)^2",
        center: [0.0, 0.0],
        extent: 5.0,
        start: [0.0, 0.0],
        lr: 0.008,
        lr_adaptive: 0.25,
        momentum: 0.9,
        squash: 60.0,
        mask: GD | MOM | ADAM | NEWTON,
        note: "Four minima, all equally deep. Everyone starts from the same point beside the hill in the middle, yet momentum ends in a different minimum from gradient descent and Adam: where you finish depends on your dynamics, not only on the loss. Drag the start around the hill: within a distance of 1 are places that lead to each of the four. Newton climbs to the top of the hill instead. A problem with several minima does not have \"the\" answer.",
        tries: &[],
    },
    Preset {
        name: "three-hump camel",
        group: "Many minima",
        formula: "2 x^2 - 1.05 x^4 + x^6 / 6 + x y + y^2",
        center: [0.0, 0.0],
        extent: 2.3,
        start: [2.05, -1.6],
        lr: 0.1,
        lr_adaptive: 0.08,
        momentum: 0.9,
        squash: 10.0,
        mask: GD | MOM | ADAM,
        note: "Three minima: the best one in the middle and a slightly worse one on each side, separated from it by low ridges. Gradient descent and Adam take the first minimum they reach. Momentum arrives at the same place with speed in hand and rolls over the ridge into the better one.",
        tries: &[],
    },
    Preset {
        name: "Styblinski-Tang",
        group: "Many minima",
        formula: "0.5 (x^4 - 16 x^2 + 5 x + y^4 - 16 y^2 + 5 y) + 80",
        center: [0.0, 0.0],
        extent: 4.6,
        start: [0.3, 0.1],
        lr: 0.012,
        lr_adaptive: 0.2,
        momentum: 0.9,
        squash: 12.0,
        mask: GD | MOM | ADAM | NEWTON,
        note: "Four basins of different depths; the deepest is at the bottom left. The start is beside the hilltop at about $(0.16, 0.16)$, and which basin you fall into is settled in the first few steps by which side of the top you are on, long before depth has any say. Newton, meanwhile, climbs to the hilltop and stops: the gradient is zero there too.",
        tries: &[Try {
            label: "start at $(0, 0)$",
            look_for: "Just across the top from the usual start: now gradient descent, momentum and Adam all fall into the deepest basin. Newton still climbs to the top.",
            set: |s| s.start = Vec2::ZERO,
        }],
    },
    Preset {
        name: "bumpy bowl",
        group: "Many minima",
        formula: "0.4 (x^2 + y^2) + 0.6 sin(3 x) sin(3 y) + 0.6",
        center: [0.0, 0.0],
        extent: 3.0,
        start: [2.5, 2.3],
        lr: 0.08,
        lr_adaptive: 0.08,
        momentum: 0.9,
        squash: 0.0,
        mask: GD | MOM | ADAM,
        note: "A bowl covered in dimples. Gradient descent is caught by the first dimple on its way down. Momentum and Adam roll through a few more before being caught.",
        tries: &[Try {
            label: "gradient noise 1",
            look_for: "Shaken out of its dimple, gradient descent works its way down to the deepest ones, near the middle, where momentum and Adam got without help. Momentum, which turns the kicks into speed, is shaken out of them again.",
            set: |s| s.settings.noise = 1.0,
        }],
    },
    Preset {
        name: "Ackley",
        group: "Many minima",
        formula: "-20 exp(-0.2 sqrt(0.5 (x^2 + y^2))) - exp(0.5 (cos(2 pi x) + cos(2 pi y))) + e + 20",
        center: [0.0, 0.0],
        extent: 3.5,
        start: [2.6, 2.2],
        lr: 0.02,
        lr_adaptive: 0.08,
        momentum: 0.9,
        squash: 0.0,
        mask: GD | MOM | ADAM,
        note: "A funnel with a rippled surface: one deep minimum in the centre, ringed by many shallow traps. Following the local slope is nearly hopeless here: everyone is caught by the first trap they meet.",
        tries: &[Try {
            label: "momentum 0.95, noise 1",
            look_for: "Shaken out of trap after trap, momentum works its way in to the centre. Gradient descent and Adam, given the same kicks, get a ripple further in at most.",
            set: |s| {
                s.settings.momentum = 0.95;
                s.settings.noise = 1.0;
            },
        }],
    },
    Preset {
        name: "sharp and flat",
        group: "Many minima",
        formula: "1 - exp(-10 ((x + 1.3)^2 + y^2)) - exp(-0.7 ((x - 1.2)^2 + y^2)) + 0.02 (x^2 + y^2)",
        center: [0.0, 0.0],
        extent: 2.6,
        start: [-1.15, 0.3],
        lr: 0.06,
        lr_adaptive: 0.05,
        momentum: 0.8,
        squash: 0.0,
        mask: GD | MOM | ADAM,
        note: "Two minima of nearly the same depth: a narrow pit and a wide basin. Everyone starts on the lip of the pit and falls in.",
        tries: &[Try {
            label: "learning rate 0.3",
            look_for: "The pit is too sharply curved for that step: gradient descent and momentum are thrown out and come to rest in the wide basin, while AdaGrad and Adam, whose steps follow the adaptive rate, stay put. Large steps cannot stay in sharp minima: one proposed reason why networks trained with large steps often do better on data they have not seen.",
            set: |s| s.settings.lr = 0.3,
        }],
    },
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub lr: f32,
    pub lr_adaptive: f32,
    pub momentum: f32,
    pub noise: f32,
    /// Step sizes are divided by (1 + decay * step).
    pub decay: f32,
    /// Gradients longer than this are shortened to it (0 = off).
    pub clip: f32,
}

#[derive(Clone)]
pub struct Runner {
    pub on: bool,
    pub pos: Vec2,
    pub prev: Vec2,
    vel: Vec2,
    m: Vec2,
    v: Vec2,
    sum: Vec2,
    pub steps: u32,
    pub dead: bool,
    /// The step after which it stopped: the gradient had all but vanished, and the
    /// next step was negligible too. (None again if it sets off once more.)
    pub settled: Option<u32>,
    /// A step at which the gradient was negligible, while the next one is awaited.
    calm: Option<u32>,
    grad0: f32,
    /// The loss as the run went: at every `every`-th step (entry `k` is step
    /// `k * every`), and last of all at the latest step, whichever that is.
    pub losses: Vec<f32>,
    pub every: u32,
    /// Its own source of gradient noise, so that what one optimiser draws (or
    /// whether another is switched on at all) never changes the noise it gets.
    rng: Rng,
}

impl Runner {
    fn new(start: Vec2, on: bool, rng: Rng) -> Self {
        Runner {
            on,
            pos: start,
            prev: start,
            vel: Vec2::ZERO,
            m: Vec2::ZERO,
            v: Vec2::ZERO,
            sum: Vec2::ZERO,
            steps: 0,
            dead: false,
            settled: None,
            calm: None,
            grad0: 0.0,
            losses: Vec::new(),
            every: 1,
            rng,
        }
    }
}

/// Most entries a loss history holds before it is thinned.
const HISTORY: usize = 6000;

/// AdaGrad's step, as a multiple of the "adaptive rate" control: its running sum of
/// squares only grows, so at the rate that suits RMSProp and Adam it would crawl.
const ADAGRAD_RATE: f32 = 3.0;

/// The optimisers and the loss they share.
pub struct Sim {
    pub expr: Expr,
    pub center: Vec2,
    pub extent: f32,
    pub start: Vec2,
    pub settings: Settings,
    pub runners: Vec<Runner>,
    /// Where the runners' noise comes from: runner `i` draws from `Rng::new(8 seed + i)`.
    seed: u64,
}

impl Sim {
    pub fn new(expr: Expr, center: Vec2, extent: f32, start: Vec2, settings: Settings, mask: u8) -> Self {
        let mut s = Sim { expr, center, extent, start, settings, runners: Vec::new(), seed: 11 };
        s.runners = (0..7).map(|i| Runner::new(start, mask & (1 << i) != 0, Rng::new(0))).collect();
        s.restart();
        s
    }

    pub fn loss(&self, p: Vec2) -> f32 {
        self.expr.eval(&[p.x, p.y])
    }

    fn exact_grad(&self, p: Vec2) -> Vec2 {
        let (_, gx, gy) = self.expr.grad2(&[p.x, p.y], 0, 1);
        Vec2::new(gx, gy)
    }

    /// The gradient optimiser `i` gets to see: noisy and clipped, as configured.
    fn grad(&mut self, i: usize, p: Vec2) -> Vec2 {
        let mut g = self.exact_grad(p);
        if self.settings.noise > 0.0 {
            let rng = &mut self.runners[i].rng;
            g += Vec2::new(rng.normal(), rng.normal()) * self.settings.noise;
        }
        let c = self.settings.clip;
        if c > 0.0 && g.length() > c {
            g *= c / g.length();
        }
        g
    }

    pub fn restart(&mut self) {
        let start = self.start;
        let l0 = self.loss(start);
        let g0 = self.exact_grad(start).length();
        for (i, r) in self.runners.iter_mut().enumerate() {
            let on = r.on;
            *r = Runner::new(start, on, Rng::new(self.seed * 8 + i as u64));
            r.grad0 = g0;
            r.losses.push(l0);
        }
    }

    fn inside(&self, p: Vec2) -> bool {
        p.is_finite() && (p - self.center).abs().max_element() <= self.extent * 1.02
    }

    /// Advance optimiser `i` by one iteration.
    pub fn step(&mut self, i: usize) {
        if !self.runners[i].on || self.runners[i].dead {
            return;
        }
        let s = self.settings;
        let t = self.runners[i].steps as f32;
        let shrink = 1.0 / (1.0 + s.decay * t);
        let (lr, lra, mu) = (s.lr * shrink, s.lr_adaptive * shrink, s.momentum);
        let (pos, vel) = (self.runners[i].pos, self.runners[i].vel);
        let eps = 1e-8;
        let root = |v: Vec2| Vec2::new(v.x.sqrt(), v.y.sqrt());
        let new_pos = match i {
            0 => pos - self.grad(i, pos) * lr,
            1 => {
                let v = vel * mu - self.grad(i, pos) * lr;
                self.runners[i].vel = v;
                pos + v
            }
            2 => {
                // measure the slope where the momentum is about to carry us
                let v = vel * mu - self.grad(i, pos + vel * mu) * lr;
                self.runners[i].vel = v;
                pos + v
            }
            3 => {
                let g = self.grad(i, pos);
                let sum = self.runners[i].sum + g * g;
                self.runners[i].sum = sum;
                pos - g / (root(sum) + eps) * (lra * ADAGRAD_RATE)
            }
            4 => {
                let g = self.grad(i, pos);
                let v = self.runners[i].v * 0.9 + g * g * 0.1;
                self.runners[i].v = v;
                pos - g / (root(v) + eps) * lra
            }
            5 => {
                let g = self.grad(i, pos);
                let k = self.runners[i].steps as i32 + 1;
                let m = self.runners[i].m * 0.9 + g * 0.1;
                let v = self.runners[i].v * 0.999 + g * g * 0.001;
                self.runners[i].m = m;
                self.runners[i].v = v;
                let mh = m / (1.0 - 0.9f32.powi(k));
                let vh = v / (1.0 - 0.999f32.powi(k));
                pos - mh / (root(vh) + eps) * lra
            }
            _ => {
                // Newton: solve H d = -g, with the Hessian from differences of the gradient
                let g = self.grad(i, pos);
                let h = 1e-3 * self.extent;
                let gx = (self.exact_grad(pos + Vec2::X * h) - self.exact_grad(pos - Vec2::X * h)) / (2.0 * h);
                let gy = (self.exact_grad(pos + Vec2::Y * h) - self.exact_grad(pos - Vec2::Y * h)) / (2.0 * h);
                let (a, b, d) = (gx.x, 0.5 * (gx.y + gy.x), gy.y);
                let det = a * d - b * b;
                let mut step = if det.abs() > 1e-9 { Vec2::new(-(d * g.x - b * g.y) / det, -(a * g.y - b * g.x) / det) } else { -g * lr };
                // a trust region: the quadratic model is only believed nearby
                let limit = 1.5 * self.extent;
                if !step.is_finite() {
                    step = -g * lr;
                }
                if step.length() > limit {
                    step *= limit / step.length();
                }
                pos + step
            }
        };
        let ok = self.inside(new_pos);
        let (g_len, value) = if ok { (self.exact_grad(new_pos).length(), self.loss(new_pos)) } else { (f32::NAN, f32::NAN) };
        let extent = self.extent;
        let r = &mut self.runners[i];
        r.steps += 1;
        r.prev = r.pos;
        if !ok || !value.is_finite() {
            r.dead = true;
            return;
        }
        let moved = (new_pos - r.pos).length();
        r.pos = new_pos;
        // (set off again, by more than the jitter of one that has stopped: RMSProp does,
        // when its record of recent gradients has shrunk with the gradient)
        if r.settled.is_some() && moved >= 1e-3 * extent {
            r.settled = None;
        }
        if r.settled.is_none() {
            // a vanishing gradient is only where the optimiser stops if it then stays:
            // momentum passing through the minimum, say, is carried on by what it remembers
            if let Some(k) = r.calm.take() {
                if moved < 1e-4 * extent {
                    r.settled = Some(k);
                }
            }
            if r.settled.is_none() && g_len < (1e-4 * r.grad0).max(1e-6) {
                r.calm = Some(r.steps);
            }
        }
        // the last entry is the step before this one: it stays if it falls on the
        // grid of recorded steps, and otherwise makes way for this one
        if r.losses.len() >= 2 && !(r.steps - 1).is_multiple_of(r.every) {
            r.losses.pop();
        }
        r.losses.push(value);
        if r.losses.len() > HISTORY {
            // keep the whole history at half the resolution: every other recorded
            // step, and (still last) the latest
            let mut half: Vec<f32> = r.losses[..r.losses.len() - 1].iter().step_by(2).copied().collect();
            half.push(value);
            r.losses = half;
            r.every *= 2;
        }
    }

    pub fn step_all(&mut self) {
        for i in 0..self.runners.len() {
            self.step(i);
        }
    }
}

// ---------------------------------------------------------------------------------

const GRID: usize = 176;
const HEIGHT: f32 = 2.8;
const MAP_Z: f32 = -1.75;

/// Most points of an optimiser's path on the map (one per step) before the older half
/// is thinned. (The newest are never thinned: every one of the last 260 is drawn as a dot.)
const FLAT_POINTS: usize = 4000;

/// Keep every other point of the older half of a path; returns how many points went.
fn thin_old_half(path: &mut Vec<Vec3>) -> usize {
    let keep_from = path.len() / 2;
    let mut thin: Vec<Vec3> = path[..keep_from].iter().step_by(2).copied().collect();
    thin.extend_from_slice(&path[keep_from..]);
    let gone = path.len() - thin.len();
    *path = thin;
    gone
}

/// What is drawn for one optimiser: its path lifted onto the surface, finely enough
/// subdivided to follow the terrain.
#[derive(Default, Clone)]
struct Trail {
    lifted: Vec<Vec3>,
    flat: Vec<Vec3>,
    /// Index in `lifted` where the newest step's segment begins.
    seg: usize,
}

pub struct Descent {
    f: ExprField,
    preset: usize,
    sim: Sim,
    trails: Vec<Trail>,
    squash: f32,
    rate: f32,
    phase: f32,
    running: bool,
    surf: Surface,
    map: Surface,
    dirty: bool,
    fmin: f32,
    fmax: f32,
    show_map: bool,
    show_grad: bool,
    show_chart: bool,
    xs: Vec<f32>,
    ys: Vec<f32>,
    vals: Vec<f32>,
    batch: Batch,
    seed: u64,
    /// Where each optimiser's name is drawn, from its marker on the screen, whether one
    /// is still on its way to where it is pushed, and the order they were stacked in
    /// from the top (see `Descent::name_markers`).
    names: [Option<Vec2>; 7],
    names_moving: bool,
    stacked: Vec<usize>,
}

/// How quickly an optimiser's name glides to a new place beside its marker, in
/// seconds of real time (as `Frame::text_placed` glides a label).
const NAME_GLIDE: f32 = 0.07;

fn settings_of(p: &Preset) -> Settings {
    Settings { lr: p.lr, lr_adaptive: p.lr_adaptive, momentum: p.momentum, noise: 0.0, decay: 0.0, clip: 0.0 }
}

impl Descent {
    pub fn new(gfx: &Gfx) -> Self {
        let p = &PRESETS[1];
        let f = ExprField::new(p.formula, &["x", "y"]);
        let sim = Sim::new(f.expr.clone(), Vec2::from(p.center), p.extent, Vec2::from(p.start), settings_of(p), p.mask);
        let mut d = Descent {
            f,
            preset: 1,
            sim,
            trails: vec![Trail::default(); 7],
            squash: p.squash,
            rate: 10.0,
            phase: 0.0,
            running: true,
            surf: Surface::new(gfx, GRID as u32, GRID as u32),
            map: Surface::new(gfx, GRID as u32, GRID as u32),
            dirty: true,
            fmin: 0.0,
            fmax: 1.0,
            show_map: true,
            show_grad: true,
            show_chart: true,
            xs: Vec::new(),
            ys: Vec::new(),
            vals: Vec::new(),
            batch: Batch::default(),
            seed: 1,
            names: [None; 7],
            names_moving: false,
            stacked: Vec::new(),
        };
        d.restart();
        d
    }

    /// Show loss `i` as it is set up, with one of its variations if `k` is given.
    fn load(&mut self, i: usize, k: Option<usize>) {
        let p = &PRESETS[i];
        self.preset = i;
        self.f.set(p.formula);
        self.squash = p.squash;
        self.sim = p.sim(self.f.expr.clone(), k);
        self.dirty = true;
        self.restart();
    }

    fn restart(&mut self) {
        self.sim.expr = self.f.expr.clone();
        self.sim.restart();
        self.trails = vec![Trail::default(); 7];
        self.phase = 1.0;
        self.running = true;
    }

    /// Displayed height of a loss value.
    fn height(&self, v: f32) -> f32 {
        let u = ((v - self.fmin) / (self.fmax - self.fmin).max(1e-12)).clamp(0.0, 1.3);
        if self.squash > 0.01 {
            HEIGHT * (1.0 + self.squash * u).ln() / (1.0 + self.squash).ln()
        } else {
            HEIGHT * u
        }
    }

    fn scale(&self) -> f32 {
        3.0 / self.sim.extent
    }

    fn lift(&self, p: Vec2) -> Vec3 {
        let v = self.sim.loss(p);
        let q = (p - self.sim.center) * self.scale();
        Vec3::new(q.x, q.y, self.height(if v.is_finite() { v } else { self.fmax }) + 0.035)
    }

    fn on_map(&self, p: Vec2) -> Vec3 {
        let q = (p - self.sim.center) * self.scale();
        Vec3::new(q.x, q.y, MAP_Z + 0.02)
    }

    /// Append the newest step of optimiser `i` to its drawn trail.
    fn extend_trail(&mut self, i: usize) {
        let (a, b) = (self.sim.runners[i].prev, self.sim.runners[i].pos);
        if self.trails[i].lifted.is_empty() {
            let (l, m) = (self.lift(a), self.on_map(a));
            self.trails[i].lifted.push(l);
            self.trails[i].flat.push(m);
        }
        self.trails[i].seg = self.trails[i].lifted.len() - 1;
        if self.sim.runners[i].dead || a == b {
            return;
        }
        // enough pieces for the segment to hug the terrain
        let pieces = (((b - a).length() / (self.sim.extent * 0.02)).ceil() as usize).clamp(1, 28);
        for k in 1..=pieces {
            let p = a.lerp(b, k as f32 / pieces as f32);
            let l = self.lift(p);
            self.trails[i].lifted.push(l);
        }
        let m = self.on_map(b);
        self.trails[i].flat.push(m);
        // very long runs (on the crease gradient descent never settles): thin out the
        // old half rather than grow without bound, since all of it is drawn every frame
        let t = &mut self.trails[i];
        if t.lifted.len() > 12_000 {
            let gone = thin_old_half(&mut t.lifted);
            t.seg = t.seg.saturating_sub(gone);
        }
        if t.flat.len() > FLAT_POINTS {
            thin_old_half(&mut t.flat);
        }
    }

    fn advance(&mut self) {
        self.sim.step_all();
        for i in 0..7 {
            if self.sim.runners[i].on {
                self.extend_trail(i);
            }
        }
    }

    fn rebuild_surface(&mut self) {
        let n = GRID;
        let (c, e) = (self.sim.center, self.sim.extent);
        if self.xs.len() != n * n {
            self.xs.resize(n * n, 0.0);
            self.ys.resize(n * n, 0.0);
            self.vals.resize(n * n, 0.0);
        }
        for j in 0..n {
            for i in 0..n {
                self.xs[j * n + i] = c.x - e + 2.0 * e * i as f32 / (n - 1) as f32;
                self.ys[j * n + i] = c.y - e + 2.0 * e * j as f32 / (n - 1) as f32;
            }
        }
        self.f.expr.eval_batch(&[&self.xs, &self.ys], &mut self.vals, &mut self.batch);
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for v in &mut self.vals {
            if !v.is_finite() {
                *v = f32::NAN;
            } else {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
        }
        if !(lo.is_finite() && hi.is_finite()) {
            lo = 0.0;
            hi = 1.0;
        }
        self.fmin = lo;
        self.fmax = hi.max(lo + 1e-6);
        let k = self.scale();
        let heights: Vec<f32> = self.vals.iter().map(|&v| self.height(if v.is_finite() { v } else { hi })).collect();
        let (xs, ys) = (&self.xs, &self.ys);
        for (idx, p) in self.surf.data_mut().iter_mut().enumerate() {
            *p = [(xs[idx] - c.x) * k, (ys[idx] - c.y) * k, heights[idx], heights[idx]];
        }
        for (idx, p) in self.map.data_mut().iter_mut().enumerate() {
            *p = [(xs[idx] - c.x) * k, (ys[idx] - c.y) * k, 0.0, heights[idx]];
        }
        for (s, lit) in [(&mut self.surf, 1.0), (&mut self.map, 0.0)] {
            let st = &mut s.style;
            st.cmap = Some(Cmap::Slate);
            // only the darker half of the map: the coloured paths have to stand out
            st.range = [-0.45, HEIGHT * 1.9];
            st.contour = HEIGHT / 18.0;
            st.contour_alpha = if lit > 0.5 { 0.38 } else { 0.6 };
            st.line_color = Color::hex(0xE4F1FF);
            st.lit = lit;
            st.specular = 0.22;
            st.rim = 0.35;
        }
        self.map.style.model = Mat4::from_translation(Vec3::new(0.0, 0.0, MAP_Z));
        self.dirty = false;
    }

    /// The optimisers' names, each beside its marker (`markers`: which optimiser, and
    /// where its marker is on the screen), pushed down a line where it would overlap
    /// one higher up, as `spread_labels` does. Pushed apart afresh in every frame, the
    /// names jumped a line whenever two markers changed which was the higher, which
    /// markers that have come to rest side by side do back and forth by a fraction of
    /// a pixel. So while they run the names are stacked in the order they had, two
    /// changing places only once one marker is higher than the other by half a line;
    /// each is drawn gliding to where it is pushed, carried along by its marker; and
    /// they are always drawn in the same order. At rest they are stacked as before.
    fn name_markers(&mut self, f: &mut Frame, markers: &[(usize, Vec2)], size: f32) {
        let line = size + 3.0;
        let width = |i: usize| (NAMES[i].chars().count().max(4) as f32) * size * 0.62;
        let at = |i: usize| markers.iter().find(|m| m.0 == i).map(|m| m.1 + Vec2::new(14.0, -8.0));
        // the order to stack them in, from the top (newcomers by height, after the rest;
        // and once the optimisers have come to rest, or are paused, simply by height,
        // as spread_labels has it)
        let kept = if self.running { self.stacked.as_slice() } else { &[] };
        let mut order: Vec<usize> = kept.iter().copied().filter(|&i| at(i).is_some()).collect();
        let mut new: Vec<usize> = markers.iter().map(|m| m.0).filter(|i| !order.contains(i)).collect();
        let height = |i: usize| at(i).map_or(0.0, |p| p.y);
        new.sort_by(|&a, &b| height(a).total_cmp(&height(b)));
        order.extend(new);
        let mut swapped = true;
        while swapped {
            swapped = false;
            for k in 1..order.len() {
                if let (Some(up), Some(down)) = (at(order[k - 1]), at(order[k])) {
                    if down.y < up.y - 0.5 * line {
                        order.swap(k - 1, k);
                        swapped = true;
                    }
                }
            }
        }
        // where each is pushed (spread_labels' rule, in that order)
        let mut placed: Vec<(usize, Vec2)> = order.iter().filter_map(|&i| at(i).map(|p| (i, p))).collect();
        self.stacked = order;
        for k in 1..placed.len() {
            for j in 0..k {
                let (above, here) = (placed[j], placed[k]);
                let reach = if above.1.x <= here.1.x { width(above.0) } else { width(here.0) };
                if (above.1.x - here.1.x).abs() < reach && here.1.y < above.1.y + line {
                    placed[k].1.y = above.1.y + line;
                }
            }
        }
        let ease = 1.0 - (-f.real_dt.max(0.0) / NAME_GLIDE).exp();
        let mut kept = [None; 7];
        self.names_moving = false;
        for &(i, at) in markers {
            let Some(&(_, to)) = placed.iter().find(|p| p.0 == i) else { continue };
            let to = to - at;
            let mut drawn = self.names[i].map_or(to, |was| was + (to - was) * ease);
            // (exactly there in the end: half a pixel is nothing to glide)
            if !drawn.is_finite() || (to - drawn).length() < 0.5 {
                drawn = to;
            }
            self.names_moving |= drawn != to;
            kept[i] = Some(drawn);
            f.text_px(at + drawn, NAMES[i], size, COLORS[i], Align2::LEFT_CENTER);
        }
        self.names = kept;
    }
}

impl Scene for Descent {
    fn view(&self) -> View {
        View::Orbit { target: Vec3::new(0.0, 0.0, 0.35), yaw: -0.8, pitch: 0.66, dist: 19.5 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    fn animated(&self) -> bool {
        self.running || self.names_moving
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, "A loss is a number to be made as small as possible. This one depends on two numbers, $x$ and $y$, so it can be drawn as a landscape whose height is the loss. Each coloured ball is an optimiser: a rule for turning the slope where it stands into a step. All of them start at the white marker, and the map underneath shows their steps from above.");

        ui::section(ui, "Things to try");
        let mut pick = None;
        let p = &PRESETS[self.preset];
        if self.f.text == p.formula {
            ui.label(egui::RichText::new(p.name).size(15.0).strong().color(pal::WHITE.egui()));
            ui::words(ui, p.note, 12.5, pal::TEXT);
            ui.add_space(4.0);
            for (k, t) in p.tries.iter().enumerate() {
                if ui::experiment(ui, t.label, t.look_for) {
                    pick = Some((self.preset, Some(k)));
                }
            }
        } else {
            ui::note(ui, "Your own loss (below). Pick one of these to go back to a loss chosen to show something.");
        }
        let mut group = "";
        for (i, q) in PRESETS.iter().enumerate() {
            if q.group != group {
                group = q.group;
                ui.add_space(2.0);
                ui.label(egui::RichText::new(group).size(11.5).color(pal::MUTED.egui()));
                let names: Vec<&str> = PRESETS.iter().filter(|r| r.group == group).map(|r| r.name).collect();
                if let Some(k) = ui::buttons(ui, &names) {
                    pick = Some((i + k, None));
                }
            }
        }
        if let Some((i, k)) = pick {
            self.load(i, k);
        }

        ui::section(ui, "Your own loss");
        if self.f.ui(ui, "loss =") {
            self.sim.expr = self.f.expr.clone();
            self.dirty = true;
            self.restart();
        }
        ui::note(ui, "Any formula in $x$ and $y$. To start the optimisers somewhere else, drag the white marker.");

        ui::section(ui, "Run");
        ui.horizontal(|ui| {
            if ui.button("restart").clicked() {
                self.restart();
            }
            if ui.button("random start").clicked() {
                self.seed += 1;
                let mut r = Rng::new(self.seed * 7919);
                let e = self.sim.extent * 0.88;
                self.sim.start = self.sim.center + Vec2::new(r.range(-e, e), r.range(-e, e));
                self.restart();
            }
            if ui.button(if self.running { "pause" } else { "resume" }).clicked() {
                self.running = !self.running;
            }
        });
        ui::slider_log(ui, "steps per second", &mut self.rate, 0.5..=400.0);

        ui::section(ui, "Step sizes");
        let before = self.sim.settings;
        let s = &mut self.sim.settings;
        ui::slider_log(ui, "learning rate", &mut s.lr, 0.0005..=3.0);
        ui::note(ui, "For gradient descent, momentum and Nesterov: how far a step goes for a given slope.");
        ui::slider_log(ui, "adaptive rate (Adam family)", &mut s.lr_adaptive, 0.001..=1.0);
        ui::note(ui, "For AdaGrad, RMSProp and Adam. AdaGrad uses three times this rate: its record of past gradients only grows, so at the others' rate it would crawl.");
        ui::slider(ui, "momentum", &mut s.momentum, 0.0..=0.99);
        ui::note(ui, "For momentum and Nesterov: how much of the last step's velocity is kept.");
        ui::slider(ui, "gradient noise", &mut s.noise, 0.0..=3.0);
        ui::note(ui, "Random kicks added to every gradient, as a gradient worked out from a few examples at a time would have. Each optimiser gets kicks of its own.");
        ui::slider(ui, "step-size decay", &mut s.decay, 0.0..=0.2);
        ui::note(ui, r"Step $n$ is divided by $1 + \text{decay} \times n$.");
        ui::slider(ui, "gradient clipping (0 = off)", &mut s.clip, 0.0..=5.0);
        ui::note(ui, "Gradients longer than this are shortened to it.");
        if before != self.sim.settings {
            self.restart();
        }

        ui::section(ui, "Optimisers");
        let mut toggled = false;
        for (i, r) in self.sim.runners.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                toggled |= ui.checkbox(&mut r.on, "").changed();
                let status = if !r.on {
                    String::new()
                } else if r.dead {
                    format!("left the map at step {}", r.steps)
                } else if let Some(k) = r.settled {
                    format!("${}$  in {} steps", fmt(*r.losses.last().unwrap_or(&0.0)), k)
                } else {
                    format!("${}$  step {}", fmt(*r.losses.last().unwrap_or(&0.0)), r.steps)
                };
                // (in a column of its own: a status too long to share the name's line goes
                // on the next line down, not over the name)
                ui.vertical(|ui| ui::readout_series(ui, NAMES[i], status, COLORS[i]));
            });
        }
        if toggled {
            self.restart();
        }
        ui::note(ui, "Each line shows the current loss, and \"in $N$ steps\" once the optimiser has stopped: its gradient has all but vanished and it stays put. If it sets off again, the count goes on.");

        ui::section(ui, "Show");
        ui.checkbox(&mut self.show_map, "contour map underneath");
        ui.checkbox(&mut self.show_grad, "downhill direction at each optimiser");
        ui.checkbox(&mut self.show_chart, "loss against step");
        if ui::slider(ui, "compress tall walls", &mut self.squash, 0.0..=300.0) {
            self.dirty = true;
            self.restart();
        }
        ui::note(ui, "Squeezes the heights of the surface (not the loss itself), so that a deep valley and the tall walls round it both fit in the picture.");

        ui::section(ui, "Reading the picture");
        ui::key(
            ui,
            &[
                (COLORS[1], "a coloured ball and line: one optimiser and the way it came, in the colour of its name under Optimisers"),
                (pal::WHITE, "white ball: the start (drag it)"),
                (Cmap::Slate.color(0.55), "the surface: its height is the loss; the fine lines join points of equal loss"),
                (COLORS[5], "dots on the map below: one per step, so their spacing shows the step sizes"),
                (pal::WHITE, "white arrows on the map: downhill from each optimiser, longer where the slope is steeper"),
                (COLORS[0], "chart at the bottom left: each optimiser's loss against the step number"),
            ],
        );
        ui::section(ui, "Words");
        ui::term(ui, "Gradient:", "the direction in which the loss climbs fastest, and how fast. Every optimiser steps against it, downhill.");
        ui::term(ui, "Curvature:", "how fast the slope changes. A sharply curved valley needs short steps, or each one overshoots.");
        ui::term(ui, "Momentum:", "keeping part of the previous step's velocity, as a rolling ball does: it carries an optimiser across flat parts and over small bumps.");
        ui::see_also(ui, &[("race", "the same optimisers training a real network"), ("landscape", "the loss surface of a real network"), ("surface", "gradients and level curves"), ("quadform", "the bowl every minimum looks like, and why a narrow one is slow")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        if self.dirty {
            self.rebuild_surface();
            self.restart();
        }
        f.surface(&mut self.surf);
        if self.show_map {
            f.surface(&mut self.map);
        }

        // time: `phase` runs from 0 to 1 across the newest step, so motion is continuous
        if self.running && f.dt > 0.0 {
            // an optimiser is still worth watching until a little after it has settled
            let busy = self.sim.runners.iter().any(|r| r.on && !r.dead && r.steps < 20_000 && r.settled.is_none_or(|k| r.steps < k + 25));
            self.phase += f.dt * self.rate;
            if busy {
                let mut budget = 600;
                while self.phase >= 1.0 && budget > 0 {
                    self.advance();
                    self.phase -= 1.0;
                    budget -= 1;
                }
                if budget == 0 {
                    self.phase = 0.0;
                }
            } else if self.phase >= 1.0 {
                // finish the last glide, then rest (and stop redrawing)
                self.phase = 1.0;
                self.running = false;
            }
        }
        let phase = self.phase.clamp(0.0, 1.0);
        // Slow playback eases each step in and out, so single steps read clearly; faster
        // playback moves at constant speed along each step, which is what looks fluid.
        let smooth = phase * phase * (3.0 - 2.0 * phase);
        let eased = smooth + (phase - smooth) * ((self.rate - 2.0) / 4.0).clamp(0.0, 1.0);

        let gref = self.sim.runners.iter().map(|r| r.grad0).fold(1e-9f32, f32::max);
        // each optimiser is named beside its marker, in its colour
        let mut names = Vec::new();
        for i in 0..7 {
            let r = &self.sim.runners[i];
            if !r.on {
                continue;
            }
            let color = COLORS[i];
            let t = &self.trails[i];
            if t.lifted.is_empty() {
                let p = self.lift(r.pos);
                f.sphere(p + Vec3::Z * 0.1, 0.22, color);
                names.extend(on_screen(f, p).map(|at| (i, at)));
                continue;
            }
            // settled history, then the part of the newest segment travelled so far
            let seg = &t.lifted[t.seg..];
            let pieces = seg.len().saturating_sub(1);
            let along = eased * pieces as f32;
            let whole = (along.floor() as usize).min(pieces);
            let head = if pieces == 0 || whole >= pieces { *seg.last().unwrap() } else { seg[whole].lerp(seg[whole + 1], along - whole as f32) };
            let drawn = t.lifted[..=t.seg + whole].iter().copied().chain(std::iter::once(head));
            f.dl.glow_lines.polyline(drawn.clone(), 5.0, color.a(0.10));
            f.dl.lines.polyline(drawn, 2.6, color);
            f.glow(head, 26.0, color.a(0.5));
            f.sphere(head + Vec3::Z * 0.09, 0.2, color);
            names.extend(on_screen(f, head).map(|at| (i, at)));

            if self.show_map && !t.flat.is_empty() {
                let n = t.flat.len();
                let (a, b) = if n >= 2 { (t.flat[n - 2], t.flat[n - 1]) } else { (t.flat[0], t.flat[0]) };
                // (one that has left the map, or has stopped moving, added no new
                // point: its marker must not travel its last step over and over)
                let tip = if r.dead || r.prev == r.pos { b } else { a.lerp(b, eased) };
                f.dl.lines.polyline(t.flat[..n - 1].iter().copied().chain(std::iter::once(tip)), 2.0, color);
                // every iterate as a dot: the spacing shows the step sizes
                for p in t.flat[..n - 1].iter().rev().take(260) {
                    f.dot(*p, 4.2, color);
                }
                f.dot(tip, 8.0, color);
                f.dashed(tip, head, 1.0, color.a(0.35));
                if self.show_grad && !r.dead {
                    // the exact downhill direction here; its length shows how strong the pull is
                    let here = r.prev.lerp(r.pos, eased);
                    let (_, gx, gy) = self.sim.expr.grad2(&[here.x, here.y], 0, 1);
                    let g = Vec2::new(gx, gy);
                    if g.length() > 1e-9 && g.is_finite() {
                        let len = 0.12 + 0.62 * (g.length() / gref).min(1.0).sqrt();
                        let d = -g.normalize() * len;
                        f.arrow(tip + Vec3::Z * 0.01, tip + Vec3::new(d.x, d.y, 0.01), 1.6, pal::WHITE.a(0.85));
                    }
                }
            }
        }

        self.name_markers(f, &names, 12.5);

        // the starting point, draggable on the surface
        let mut s = self.lift(self.sim.start);
        if f.drag3(1, &mut s, Some(Vec3::Z), pal::WHITE) {
            let e = self.sim.extent;
            let p = self.sim.center + Vec2::new(s.x, s.y) / self.scale();
            self.sim.start = p.clamp(self.sim.center - Vec2::splat(e), self.sim.center + Vec2::splat(e));
            self.restart();
        }
        f.text_off(self.lift(self.sim.start), Vec2::new(-12.0, -12.0), "start (drag me)", 12.5, pal::WHITE, Align2::RIGHT_BOTTOM);

        if self.show_map {
            // a few words beside the map: next to the middle of whichever edge is lowest and
            // furthest right on screen, where there is room
            let corners = [(-3.0, -3.0), (3.0, -3.0), (3.0, 3.0), (-3.0, 3.0)].map(|(x, y)| on_screen(f, Vec3::new(x, y, MAP_Z)));
            let edge = (0..4).filter_map(|k| Some((corners[k]? + corners[(k + 1) % 4]?) * 0.5)).max_by(|a, b| (a.x + a.y).total_cmp(&(b.x + b.y)));
            if let Some(mid) = edge {
                let mut lines = vec!["The same loss, seen from above.", "Each dot is one step."];
                if self.show_grad {
                    lines.push("The white arrows point downhill.");
                }
                // (only where they fit: on a phone they would run off the edge; the key says it)
                let room = f.view_size().x - mid.x - 18.0;
                let fits = lines.iter().all(|t| text_width(t, 12.5) + 6.0 <= room);
                for (row, text) in lines.into_iter().enumerate().filter(|_| fits) {
                    f.text_px(mid + Vec2::new(18.0, 14.0 + 16.0 * row as f32), text, 12.5, pal::TEXT, Align2::LEFT_TOP);
                }
            }
        }

        if self.show_chart {
            self.chart(f);
        }
    }
}

impl Descent {
    /// Loss above the lowest point on the map (or the lowest loss reached, if that is
    /// lower), on a logarithmic scale, against the step number.
    fn chart(&self, f: &mut Frame) {
        let view = f.view_size();
        let size = Vec2::new((view.x * 0.34).clamp(220.0, 400.0), (view.y * 0.22).clamp(110.0, 200.0));
        let o = Vec2::new(18.0, view.y - 30.0 - size.y);
        let pad = 10.0;
        hud_panel(f, o - Vec2::splat(pad), o + size + Vec2::splat(pad));

        let runners: Vec<(usize, &Runner)> = self.sim.runners.iter().enumerate().filter(|(_, r)| r.on && !r.losses.is_empty()).collect();
        let best = runners.iter().flat_map(|(_, r)| r.losses.iter().copied()).fold(self.fmin, f32::min);
        let worst = runners.iter().flat_map(|(_, r)| r.losses.iter().copied()).fold(f32::MIN, f32::max);
        let span = (worst - best).max(1e-12);
        let floor = span * 1e-7;
        let (top, bottom) = (span.log10() + 0.15, (span * 1e-7).log10());
        let steps = runners.iter().map(|(_, r)| r.steps).max().unwrap_or(1).max(30) as f32;
        let at = |step: f32, v: f32| {
            let y = ((v - best).max(floor).log10() - bottom) / (top - bottom);
            o + Vec2::new(size.x * step / steps, size.y * (1.0 - y.clamp(0.0, 1.0)))
        };
        // decade lines
        for d in (bottom.ceil() as i32)..=(top.floor() as i32) {
            if (d - bottom.ceil() as i32) % 2 != 0 {
                continue;
            }
            let y = o.y + size.y * (1.0 - (d as f32 - bottom) / (top - bottom));
            hud_line(f, Vec2::new(o.x, y), Vec2::new(o.x + size.x, y), 1.0, pal::FAINT.a(0.5));
        }
        hud_line(f, o + Vec2::new(0.0, size.y), o + size, 1.0, pal::DIM.a(0.8));
        for (i, r) in &runners {
            let n = r.losses.len();
            // each entry at the step it was recorded at (see `Runner::losses`); one
            // that left the map recorded nothing at the step it left on
            let latest = if r.dead { r.steps.saturating_sub(1) } else { r.steps };
            let step_of = |k: usize| if k + 1 == n { latest as f32 } else { (k as u32 * r.every) as f32 };
            let stride = (n / 400).max(1);
            let pts: Vec<Vec3> = (0..n).step_by(stride).chain(std::iter::once(n - 1)).map(|k| f.hud(at(step_of(k), r.losses[k]))).collect();
            f.dl.overlay.polyline(pts, 1.8, COLORS[*i]);
        }
        for (row, (i, _)) in runners.iter().enumerate() {
            f.text_px(o + Vec2::new(size.x - 2.0, 2.0 + 14.0 * row as f32), NAMES[*i], 11.5, COLORS[*i], Align2::RIGHT_TOP);
        }
        f.text_px(o + Vec2::new(0.0, -pad - 19.0), "loss above the lowest point on the map", 12.0, pal::TEXT, Align2::LEFT_BOTTOM);
        f.text_px(o + Vec2::new(0.0, -pad - 4.0), "log scale: each grid line is 100× lower than the one above", 11.5, pal::DIM, Align2::LEFT_BOTTOM);
        f.text_px(o + Vec2::new(size.x, size.y + 2.0), format!("{}", steps as u32), 11.0, pal::DIM, Align2::RIGHT_TOP);
        f.text_px(o + Vec2::new(size.x * 0.5, size.y + 2.0), "step", 11.0, pal::DIM, Align2::CENTER_TOP);
        f.text_px(o + Vec2::new(0.0, size.y + 2.0), "0", 11.0, pal::DIM, Align2::LEFT_TOP);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every loss's note and things to try are typeset as written (the lint and the
    /// scene check read only the loss the scene opens with).
    #[test]
    fn every_loss_is_typeset_as_written() {
        for p in &PRESETS {
            for text in std::iter::once(p.note).chain(p.tries.iter().flat_map(|t| [t.label, t.look_for])) {
                let found = crate::typeset::inspect(text);
                assert!(found.error.is_none() && found.words.is_empty(), "{}: {found:?} in {text}", p.name);
            }
        }
    }

    fn preset(name: &str) -> &'static Preset {
        PRESETS.iter().find(|p| p.name == name).unwrap_or_else(|| panic!("no preset {name}"))
    }

    /// A loss as its button sets it up (with variation `k` of its "things to try" if
    /// given), and every optimiser switched on, so that any of them can be run. (Each
    /// has noise of its own, so the others being on changes nothing for it.)
    fn tried(name: &str, k: Option<usize>) -> Sim {
        let p = preset(name);
        let expr = Expr::parse(p.formula, &["x", "y"]).unwrap_or_else(|e| panic!("{}: {e}", p.name));
        let mut s = p.sim(expr, k);
        for r in &mut s.runners {
            r.on = true;
        }
        s
    }

    fn sim_of(p: &Preset) -> Sim {
        tried(p.name, None)
    }

    /// The same, with the noise drawn from seed `seed` (the scene's is 11).
    fn with_seed(name: &str, k: Option<usize>, seed: u64) -> Sim {
        let mut s = tried(name, k);
        s.seed = seed;
        s.restart();
        s
    }

    /// Run optimiser `i` until its loss drops below `target`; the number of steps taken.
    fn steps_to(s: &mut Sim, i: usize, target: f32) -> usize {
        for k in 0..6000 {
            if s.runners[i].dead {
                return usize::MAX;
            }
            if *s.runners[i].losses.last().unwrap() < target {
                return k;
            }
            s.step(i);
        }
        6000
    }

    fn run(s: &mut Sim, i: usize, steps: usize) -> Vec2 {
        for _ in 0..steps {
            s.step(i);
        }
        s.runners[i].pos
    }

    /// The step after which optimiser `i` stopped (see `Runner::settled`), running it
    /// for up to `most` steps.
    fn settles(s: &mut Sim, i: usize, most: usize) -> u32 {
        for _ in 0..most {
            if let Some(k) = s.runners[i].settled {
                return k;
            }
            s.step(i);
        }
        s.runners[i].settled.unwrap_or(u32::MAX)
    }

    /// The variation of a loss with this button label.
    fn variation(name: &str, label: &str) -> Option<usize> {
        let k = preset(name).tries.iter().position(|t| t.label == label);
        assert!(k.is_some(), "{name} has no \"{label}\"");
        k
    }

    const GD_: usize = 0;
    const MOM_: usize = 1;
    const NAG_: usize = 2;
    const ADAGRAD_: usize = 3;
    const RMS_: usize = 4;
    const ADAM_: usize = 5;
    const NEWTON_: usize = 6;

    /// Every preset parses, starts inside its map at a finite loss, and with its
    /// default settings the optimisers it switches on make progress and stay on the map.
    #[test]
    fn presets_behave() {
        println!("\n{:<24} {:>9}  {}", "preset", "start", NAMES.map(|n| format!("{:>13}", &n[..n.len().min(13)])).join(" "));
        let mut failures: Vec<String> = Vec::new();
        for p in &PRESETS {
            let mut sim = sim_of(p);
            assert!(sim.inside(sim.start), "{}: start outside the map", p.name);
            let l0 = sim.loss(sim.start);
            assert!(l0.is_finite(), "{}: loss at start", p.name);
            for _ in 0..800 {
                sim.step_all();
            }
            let shown = |i: usize| p.mask & (1 << i) != 0;
            let cells: Vec<String> = sim
                .runners
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let (open, close) = if shown(i) { ("", "") } else { ("(", ")") };
                    let body = if r.dead {
                        format!("out@{}", r.steps)
                    } else {
                        format!("{:.3}{}", r.losses.last().unwrap(), r.settled.map_or(String::new(), |k| format!("/{k}")))
                    };
                    format!("{:>13}", format!("{open}{body}{close}"))
                })
                .collect();
            println!("{:<24} {:>9.3}  {}", p.name, l0, cells.join(" "));
            let alive = sim.runners.iter().enumerate().filter(|(i, r)| shown(*i) && !r.dead).count();
            let improved = sim.runners.iter().enumerate().any(|(i, r)| shown(i) && !r.dead && *r.losses.last().unwrap() < l0);
            if !improved {
                failures.push(format!("{}: nothing improved", p.name));
            }
            // the cliff is the one loss that is meant to throw optimisers off the map
            if p.name != "cliff" && alive < p.mask.count_ones() as usize {
                failures.push(format!("{}: an optimiser shown by default leaves the map", p.name));
            }
        }
        assert!(failures.is_empty(), "{failures:?}");
    }

    /// The round bowl, its learning rates, and the ravines.
    #[test]
    fn quadratics() {
        // Newton solves a quadratic in one step.
        let mut s = sim_of(preset("round bowl"));
        s.step(NEWTON_);
        assert!(s.runners[NEWTON_].pos.length() < 1e-3, "Newton on a bowl: {:?}", s.runners[NEWTON_].pos);
        // Gradient descent walks straight in, each step the same fraction of the way.
        let start = s.start;
        let mut r = start.length();
        for _ in 0..20 {
            let p = run(&mut s, GD_, 1);
            assert!((p.length() / r - 0.85).abs() < 1e-4 && p.perp_dot(start).abs() < 1e-5);
            r = p.length();
        }
        // Momentum overshoots and swings back.
        let mut s = sim_of(preset("round bowl"));
        let sides: Vec<f32> = (0..60).map(|_| run(&mut s, MOM_, 1).dot(start)).collect();
        assert!(sides.iter().any(|&d| d < -0.1) && sides.windows(2).filter(|w| w[0] * w[1] < 0.0).count() >= 2, "momentum never swings across");

        // learning rate 1: one step, as Newton
        let mut s = tried("round bowl", variation("round bowl", "learning rate 1"));
        assert!(run(&mut s, GD_, 1).length() < 1e-6 && settles(&mut s, GD_, 10) == 1);
        // 1.9: across to the other side at every step, a little closer each time
        let mut s = tried("round bowl", variation("round bowl", "learning rate 1.9"));
        let mut before = s.start;
        for _ in 0..30 {
            let p = run(&mut s, GD_, 1);
            assert!(p.dot(before) < 0.0 && p.length() < before.length(), "{p:?} after {before:?}");
            before = p;
        }
        // 2.1: further out each time, and off the map within three steps
        let mut s = tried("round bowl", variation("round bowl", "learning rate 2.1"));
        run(&mut s, GD_, 3);
        assert!(s.runners[GD_].dead && s.runners[GD_].steps == 3, "a learning rate above 2/L must diverge");

        // The ravine: momentum settles in about a third of gradient descent's steps.
        let mut s = sim_of(preset("ravine"));
        let (gd, mom) = (settles(&mut s, GD_, 1000), settles(&mut s, MOM_, 1000));
        assert!((70..=74).contains(&gd) && (25..=30).contains(&mom), "gradient descent {gd}, momentum {mom}");
        let mut s = sim_of(preset("ravine"));
        let (gd, mom) = (steps_to(&mut s, GD_, 1e-4), steps_to(&mut s, MOM_, 1e-4));
        assert!(mom * 2 < gd, "momentum {mom} vs gradient descent {gd}");
        // AdaGrad and Adam set off diagonally, as if the valley were round ...
        for i in [ADAGRAD_, ADAM_] {
            let mut s = sim_of(preset("ravine"));
            let d = run(&mut s, i, 1) - s.start;
            assert!((d.x.abs() / d.y.abs() - 1.0).abs() < 1e-3, "{}: first step {d:?}", NAMES[i]);
        }
        // ... and Adam takes a while to come to rest: twice as long as gradient descent.
        let mut s = sim_of(preset("ravine"));
        let adam = settles(&mut s, ADAM_, 1000);
        assert!(adam > 2 * 72, "Adam settles at {adam}");
        // learning rate 0.5: one step to the floor, then a slower crawl (about 130 steps, not 72)
        let mut s = tried("ravine", variation("ravine", "learning rate 0.5"));
        assert!(run(&mut s, GD_, 1).y.abs() < 1e-6);
        let gd = settles(&mut s, GD_, 1000);
        assert!((125..=137).contains(&gd), "at 0.5 gradient descent settles at {gd}");

        // Rotating the ravine changes nothing for gradient descent and momentum ...
        for i in [GD_, MOM_] {
            let (a, b) = (settles(&mut sim_of(preset("ravine")), i, 1000), settles(&mut sim_of(preset("rotated ravine")), i, 1000));
            assert!((a as i64 - b as i64).abs() <= 1, "{}: aligned {a}, rotated {b}", NAMES[i]);
        }
        // ... and costs AdaGrad, which scales each coordinate separately, several times as many.
        let (a, b) = (settles(&mut sim_of(preset("ravine")), ADAGRAD_, 1000), settles(&mut sim_of(preset("rotated ravine")), ADAGRAD_, 1000));
        assert!(b > 5 * a, "AdaGrad: aligned {a}, rotated {b}");

        // The narrow ravine: hundreds of steps; momentum about a quarter as many; Newton one.
        let mut s = sim_of(preset("narrow ravine"));
        let (gd, mom, newton) = (settles(&mut s, GD_, 2000), settles(&mut s, MOM_, 2000), settles(&mut s, NEWTON_, 10));
        assert!(gd > 250 && mom * 3 < gd && mom * 5 > gd && newton == 1, "gradient descent {gd}, momentum {mom}, Newton {newton}");
    }

    /// The curved valleys: how long each takes.
    #[test]
    fn curved_valleys() {
        // Rosenbrock: Newton in a handful of steps, momentum and Adam an order faster than gradient descent
        let mut s = sim_of(preset("Rosenbrock"));
        let newton = settles(&mut s, NEWTON_, 100);
        assert!(newton <= 6 && (s.runners[NEWTON_].pos - Vec2::ONE).length() < 1e-3, "Newton {newton}");
        let (gd, mom) = (settles(&mut s, GD_, 6000), settles(&mut s, MOM_, 6000));
        let adam = steps_to(&mut s, ADAM_, 1e-4) as u32;
        assert!(gd > 3000 && mom * 10 < gd && adam * 10 < gd, "gradient descent {gd}, momentum {mom}, Adam {adam}");
        // the winding valley: gradient descent long after the others
        let mut s = sim_of(preset("winding valley"));
        let gd = settles(&mut s, GD_, 6000);
        for i in [MOM_, NAG_, ADAM_] {
            let k = settles(&mut s, i, 6000);
            assert!(k * 5 < gd, "{} {k}, gradient descent {gd}", NAMES[i]);
        }
        // Beale: thousands of steps for gradient descent, a few hundred for momentum and Adam
        let mut s = sim_of(preset("Beale"));
        let gd = settles(&mut s, GD_, 6000);
        let (mom, adam) = (settles(&mut s, MOM_, 6000), settles(&mut s, ADAM_, 6000));
        assert!((2500..=2700).contains(&gd) && mom < 400 && adam < 400, "gradient descent {gd}, momentum {mom}, Adam {adam}");
        assert!((s.runners[GD_].pos - Vec2::new(3.0, 0.5)).length() < 1e-2);
    }

    #[test]
    fn saddles_plateaus_and_cliffs() {
        // Newton is attracted to the saddle point; gradient descent leaves it, eventually.
        let mut s = sim_of(preset("saddle"));
        assert!(run(&mut s, NEWTON_, 50).length() < 1e-2, "Newton should sit on the saddle");
        assert!(run(&mut s, GD_, 400).y.abs() > 1.0, "gradient descent should have left");
        // Adam gets below the level of the pass well before gradient descent does.
        let mut s = sim_of(preset("saddle"));
        let (gd, mom, adam) = (steps_to(&mut s, GD_, 0.9), steps_to(&mut s, MOM_, 0.9), steps_to(&mut s, ADAM_, 0.9));
        assert!(adam * 2 < gd, "leaving the saddle: Adam {adam}, gradient descent {gd}");
        // With a little noise, gradient descent and momentum get away sooner (every seed);
        // Adam gains nothing, and Newton stays on the pass.
        let k = variation("saddle", "gradient noise 0.1");
        for seed in 11..21 {
            let mut s = with_seed("saddle", k, seed);
            let (gd2, mom2, adam2) = (steps_to(&mut s, GD_, 0.9), steps_to(&mut s, MOM_, 0.9), steps_to(&mut s, ADAM_, 0.9));
            assert!(gd2 * 3 < gd * 2 && mom2 < mom && adam2 >= adam, "seed {seed}: gradient descent {gd2} (was {gd}), momentum {mom2} (was {mom}), Adam {adam2} (was {adam})");
            let below = steps_to(&mut s, NEWTON_, 0.9);
            assert!(below == 6000 && s.runners[NEWTON_].pos.length() < 0.2, "seed {seed}: Newton left the pass for {:?}", s.runners[NEWTON_].pos);
        }

        // On the plateau the adaptive methods halve the loss several times sooner.
        let mut s = sim_of(preset("plateau"));
        let (gd, adam, rms) = (steps_to(&mut s, GD_, 0.5), steps_to(&mut s, ADAM_, 0.5), steps_to(&mut s, RMS_, 0.5));
        assert!(adam * 3 < gd && rms * 3 < gd, "plateau: gradient descent {gd}, Adam {adam}, RMSProp {rms}");

        // The cliff face is about a hundred times steeper than the slope just above it.
        let s = sim_of(preset("cliff"));
        let (face, above) = (s.exact_grad(Vec2::new(0.0, 0.2)).length(), s.exact_grad(Vec2::new(1.0, 0.2)).length());
        assert!((80.0..150.0).contains(&(face / above)), "the face is {} times steeper", face / above);
        // The cliff throws unclipped gradient descent off the map, momentum too; the
        // methods that normalise their step stay on. Clipping keeps the first two on.
        let mut s = sim_of(preset("cliff"));
        for i in [GD_, MOM_, ADAGRAD_, RMS_, ADAM_] {
            run(&mut s, i, 300);
        }
        assert!(s.runners[GD_].dead && s.runners[MOM_].dead, "unclipped gradient descent and momentum should be thrown off the cliff");
        assert!(!s.runners[RMS_].dead && !s.runners[ADAM_].dead && !s.runners[ADAGRAD_].dead);
        let mut s = tried("cliff", variation("cliff", "gradient clipping 0.5"));
        for i in [GD_, MOM_] {
            run(&mut s, i, 300);
            assert!(!s.runners[i].dead && s.runners[i].settled.is_some() && s.runners[i].pos.x < -0.5, "clipped {} should walk down", NAMES[i]);
        }

        // On the crease a fixed step never settles; AdaGrad's shrinking step gets much closer.
        let mut s = sim_of(preset("crease"));
        for i in [GD_, MOM_, ADAGRAD_, ADAM_] {
            run(&mut s, i, 800);
        }
        let loss = |s: &Sim, i: usize| *s.runners[i].losses.last().unwrap();
        assert!(s.runners[GD_].settled.is_none() && loss(&s, GD_) > 0.1 && loss(&s, MOM_) > 0.1);
        // (0.014 against 0.25 at step 800)
        assert!(loss(&s, ADAGRAD_) * 15.0 < loss(&s, GD_), "AdaGrad {}, gradient descent {}", loss(&s, ADAGRAD_), loss(&s, GD_));
        // ... and with step-size decay the others close in too.
        // (the highest loss over each stretch of steps shrinks: hops ever smaller,
        // where without decay gradient descent keeps hopping at 0.25 and momentum at 0.15)
        let mut s = tried("crease", variation("crease", "step-size decay 0.05"));
        for i in [GD_, MOM_, ADAM_] {
            let mut highest = Vec::new();
            for n in [100, 100, 200, 600] {
                let mut most = 0.0f32;
                for _ in 0..n {
                    s.step(i);
                    most = most.max(loss(&s, i));
                }
                highest.push(most);
            }
            assert!(highest.windows(2).all(|w| w[1] < w[0]) && highest[3] < 0.02, "{} with decay: {highest:?}", NAMES[i]);
        }
    }

    /// Two wells: how many times momentum crosses the ridge, and so where it ends,
    /// with each momentum the panel offers. (The note once said that less momentum
    /// keeps it in the near valley; 0.9 does, 0.8 does not.)
    #[test]
    fn two_wells_and_momentum() {
        let ridge = 0.075; // where the slope along x is zero between the wells
        let crossings = |k: Option<usize>| {
            let mut s = tried("two wells", k);
            let (mut side, mut count) = (s.runners[MOM_].pos.x > ridge, 0);
            for _ in 0..3000 {
                let x = run(&mut s, MOM_, 1).x;
                if (x > ridge) != side {
                    (side, count) = (!side, count + 1);
                }
            }
            let r = &s.runners[MOM_];
            (count, r.pos.x < 0.0, r.settled.unwrap_or(u32::MAX))
        };
        // as set up (0.95): over, back, and over again, into the deep valley
        assert_eq!(crossings(None).0, 3);
        assert!(crossings(None).1);
        let (n, far, _) = crossings(variation("two wells", "momentum 0.6"));
        assert!(n == 0 && !far, "0.6: {n} crossings");
        let (n, far, _) = crossings(variation("two wells", "momentum 0.9"));
        assert!(n == 2 && !far, "0.9: {n} crossings");
        let (n, far, settled) = crossings(variation("two wells", "momentum 0.99"));
        assert!(n > 12 && !far && (1400..1600).contains(&settled), "0.99: {n} crossings, settled at {settled}");
        // gradient descent stays in the near one
        let mut s = sim_of(preset("two wells"));
        assert!(run(&mut s, GD_, 800).x > 0.5);
    }

    #[test]
    fn which_minimum() {
        // Two weights multiplied: a larger step ends in a flatter part of the valley.
        let sharpness = |p: Vec2| 2.0 * (p.x * p.x + p.y * p.y);
        let end = |k: Option<usize>| {
            let mut s = tried("two weights multiplied", k);
            let p = run(&mut s, GD_, 600);
            assert!(!s.runners[GD_].dead && (p.x * p.y - 1.0).abs() < 1e-3, "{k:?}: ended at {p:?}");
            p
        };
        let (small, large) = (end(variation("two weights multiplied", "learning rate 0.05")), end(None));
        assert!(sharpness(large) < 0.5 * sharpness(small), "sharpness {} vs {}", sharpness(large), sharpness(small));
        assert!((small - Vec2::new(0.4, 2.6)).length() < 0.05 && (large - Vec2::new(0.8, 1.3)).length() < 0.05, "{small:?}, {large:?}");
        // and no step size can stay where the curvature exceeds 2 / lr
        assert!(sharpness(large) < 2.0 / 0.2 + 1e-3);

        // Three-hump camel: gradient descent and Adam take the side minimum, momentum the central one.
        let mut s = sim_of(preset("three-hump camel"));
        assert!(run(&mut s, GD_, 800).x > 1.0 && run(&mut s, ADAM_, 800).x > 1.0);
        assert!(run(&mut s, MOM_, 800).length() < 0.05);

        // Himmelblau: same start, different minima; Newton climbs the hill instead.
        let mut s = sim_of(preset("Himmelblau"));
        let (gd, mom, adam) = (run(&mut s, GD_, 800), run(&mut s, MOM_, 800), run(&mut s, ADAM_, 800));
        assert!([GD_, MOM_, ADAM_].iter().all(|&i| s.runners[i].losses.last().unwrap().abs() < 1e-3));
        assert!((gd - mom).length() > 1.0 && (gd - adam).length() < 1e-2, "gradient descent {gd:?}, momentum {mom:?}, Adam {adam:?}");
        let top = run(&mut s, NEWTON_, 50);
        assert!((top - Vec2::new(-0.271, -0.923)).length() < 0.01, "Newton at {top:?}");
        // within a distance of 1 of the start are places that lead to each of the four
        let minima = [Vec2::new(3.0, 2.0), Vec2::new(-2.805, 3.131), Vec2::new(-3.779, -3.283), Vec2::new(3.584, -1.848)];
        let mut reached = [false; 4];
        for k in 0..72 {
            let a = k as f32 * TAU / 72.0;
            let mut s = sim_of(preset("Himmelblau"));
            s.start = Vec2::new(a.cos(), a.sin());
            s.restart();
            let p = run(&mut s, GD_, 2000);
            if let Some(m) = minima.iter().position(|m| (*m - p).length() < 0.01) {
                reached[m] = true;
            }
        }
        assert_eq!(reached, [true; 4]);

        // Styblinski-Tang: which side of the top decides the basin; Newton climbs to the top.
        let deepest = Vec2::new(-2.904, -2.904);
        let mut s = sim_of(preset("Styblinski-Tang"));
        for i in [GD_, MOM_, ADAM_] {
            assert!((run(&mut s, i, 2000) - Vec2::new(2.747, -2.904)).length() < 0.01, "{}", NAMES[i]);
        }
        assert!((run(&mut s, NEWTON_, 50) - Vec2::splat(0.157)).length() < 0.01);
        let mut s = tried("Styblinski-Tang", variation("Styblinski-Tang", "start at $(0, 0)$"));
        for i in [GD_, MOM_, ADAM_] {
            assert!((run(&mut s, i, 2000) - deepest).length() < 0.01, "{} from (0, 0)", NAMES[i]);
        }
        assert!((run(&mut s, NEWTON_, 50) - Vec2::splat(0.157)).length() < 0.01);

        // Sharp and flat: a small step stays in the pit, a large one ends in the basin;
        // AdaGrad and Adam, which follow the adaptive rate, stay in the pit.
        let mut s = sim_of(preset("sharp and flat"));
        assert!([GD_, MOM_, ADAGRAD_, ADAM_].iter().all(|&i| run(&mut s, i, 2000).x < -1.0), "everyone falls into the pit");
        let mut s = tried("sharp and flat", variation("sharp and flat", "learning rate 0.3"));
        assert!(run(&mut s, GD_, 2000).x > 0.5 && run(&mut s, MOM_, 2000).x > 0.5);
        assert!(run(&mut s, ADAM_, 2000).x < -1.0 && run(&mut s, ADAGRAD_, 2000).x < -1.0);
    }

    /// The bumpy bowl and Ackley, without noise and with the noise their buttons add
    /// (ten seeds, the scene's own among them).
    #[test]
    fn noise_in_the_dimples() {
        // the average distance from the centre and loss over steps 2000 to 3000
        let late = |s: &mut Sim, i: usize| {
            let (mut far, mut loss) = (0.0, 0.0);
            for k in 0..3000 {
                s.step(i);
                if k >= 2000 {
                    far += s.runners[i].pos.length() / 1000.0;
                    loss += s.runners[i].losses.last().unwrap() / 1000.0;
                }
            }
            (far, loss, s.runners[i].dead)
        };
        // bumpy bowl: gradient descent is caught early, momentum and Adam go deeper
        let mut s = sim_of(preset("bumpy bowl"));
        let (gd, _, _) = late(&mut s, GD_);
        let (_, mom_loss, _) = late(&mut s, MOM_);
        let (adam, _, _) = late(&mut s, ADAM_);
        assert!(gd > 1.3 && adam < 0.7, "gradient descent at {gd}, Adam at {adam}");
        let k = variation("bumpy bowl", "gradient noise 1");
        for seed in 11..21 {
            let mut s = with_seed("bumpy bowl", k, seed);
            let (gd, _, _) = late(&mut s, GD_);
            let (mom, loss, dead) = late(&mut s, MOM_);
            assert!(gd < 0.7, "seed {seed}: gradient descent at {gd} from the centre");
            assert!(dead || (mom > 0.7 && loss > 2.0 * mom_loss), "seed {seed}: momentum at {mom}, loss {loss}");
        }

        // Ackley: everyone caught by the first trap; with momentum 0.95 and noise 1
        // momentum works its way in, the others a ripple (0.78) further at most
        let mut s = sim_of(preset("Ackley"));
        let first = run(&mut s, GD_, 2000);
        for i in [MOM_, ADAM_] {
            assert!((run(&mut s, i, 2000) - first).length() < 1e-3);
        }
        let k = variation("Ackley", "momentum 0.95, noise 1");
        for seed in 11..21 {
            let mut s = with_seed("Ackley", k, seed);
            let (mom, _, _) = late(&mut s, MOM_);
            let (gd, _, _) = late(&mut s, GD_);
            let (adam, _, _) = late(&mut s, ADAM_);
            assert!(mom < 0.8 && gd > first.length() - 0.8 && adam > first.length() - 0.8, "seed {seed}: momentum {mom}, gradient descent {gd}, Adam {adam}");
        }
    }

    /// With noise on, what an optimiser draws does not depend on which others are on
    /// (it used to: one generator served them all, in step order).
    #[test]
    fn each_optimiser_has_noise_of_its_own() {
        let path = |mask: u8| {
            let p = preset("bumpy bowl");
            let mut s = p.sim(Expr::parse(p.formula, &["x", "y"]).unwrap(), variation("bumpy bowl", "gradient noise 1"));
            for (i, r) in s.runners.iter_mut().enumerate() {
                r.on = mask & (1 << i) != 0;
            }
            s.restart();
            (0..300).map(|_| {
                s.step_all();
                s.runners[MOM_].pos
            }).collect::<Vec<Vec2>>()
        };
        assert_eq!(path(MOM), path(GD | MOM | ADAM));
        assert_eq!(path(MOM), path(127));
    }

    /// "in N steps" means stopped: momentum passing through the minimum at speed has
    /// not, and RMSProp hopping about again is no longer counted as stopped.
    #[test]
    fn settled_means_stopped() {
        let mut s = tried("round bowl", variation("round bowl", "learning rate 1"));
        assert_eq!(settles(&mut s, GD_, 10), 1);
        assert_eq!(settles(&mut s, NEWTON_, 10), 1);
        // momentum lands on the minimum after one step too, with all its speed
        assert!(run(&mut s, MOM_, 1).length() < 1e-6);
        let k = settles(&mut s, MOM_, 1000);
        assert!(k > 50, "momentum settled at {k}");
        // once stopped, they stay put for as long as the scene goes on showing them:
        // within 0.3 % of the map's width (which is twice the extent), two pixels on a
        // map 600 wide. (RMSProp on the bumpy bowl creeps that far before it sets off
        // again; the rest stay within 0.08 %.)
        let mut most = (0.0f32, "");
        for p in &PRESETS {
            let mut s = sim_of(p);
            for i in 0..7 {
                let k = settles(&mut s, i, 4000);
                if k != u32::MAX {
                    let here = s.runners[i].pos;
                    let drift = (0..25).map(|_| (run(&mut s, i, 1) - here).length()).fold(0.0, f32::max) / (2.0 * p.extent);
                    if drift > most.0 {
                        most = (drift, NAMES[i]);
                    }
                }
            }
        }
        println!("the most anything moved after stopping: {:.1e} of the map's width ({})", most.0, most.1);
        assert!(most.0 < 3e-3, "{most:?}");
        // the cliff: RMSProp stops, then sets off again for good
        let mut s = sim_of(preset("cliff"));
        assert_eq!(settles(&mut s, RMS_, 100), 38);
        run(&mut s, RMS_, 100);
        assert!(s.runners[RMS_].settled.is_none());
    }

    /// The map's path is thinned like the one on the surface: long runs keep a bounded
    /// number of points, the start, and every one of the latest steps.
    #[test]
    fn long_paths_are_thinned() {
        let mut path: Vec<Vec3> = Vec::new();
        for k in 0..20_000 {
            path.push(Vec3::new(k as f32, 0.0, 0.0));
            if path.len() > FLAT_POINTS {
                thin_old_half(&mut path);
            }
        }
        assert!(path.len() <= FLAT_POINTS && path[0].x == 0.0);
        let last: Vec<f32> = path[path.len() - 300..].iter().map(|p| p.x).collect();
        assert!(last.iter().enumerate().all(|(j, &x)| x == (19_700 + j) as f32));
    }

    /// The loss history after it has been thinned: entry k is still the loss at step
    /// k * every, and the last entry the loss at the latest step. (New entries used
    /// to go on arriving every step after a thinning, so a chart spread evenly over
    /// the entries put step 6000 at step 4500.)
    #[test]
    fn the_loss_history_keeps_its_steps() {
        let mut s = sim_of(preset("round bowl"));
        // a slow crawl, so that the loss is different at every step
        s.settings.lr = 1e-4;
        s.restart();
        let mut full = vec![*s.runners[0].losses.last().unwrap()];
        for _ in 0..15_000 {
            s.step(0);
            full.push(*s.runners[0].losses.last().unwrap());
        }
        let r = &s.runners[0];
        assert!(r.every >= 2 && r.losses.len() <= HISTORY + 1, "every {} with {} entries", r.every, r.losses.len());
        assert_eq!(r.steps as usize, full.len() - 1);
        let n = r.losses.len();
        for k in 0..n - 1 {
            assert_eq!(r.losses[k], full[k * r.every as usize], "entry {k}");
        }
        assert_eq!(r.losses[n - 1], full[r.steps as usize]);
        assert!((n - 2) * r.every as usize <= r.steps as usize);
    }
}

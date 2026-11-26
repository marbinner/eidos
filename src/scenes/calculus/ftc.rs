//! The fundamental theorem of calculus: the area under a curve from a fixed start to a
//! moving end, drawn as a function of the end. Its slope is the curve's height, and
//! that is why areas are found by undoing derivatives.

use super::*;
use crate::math::expr::Func;
use crate::scenes::stats::plot::{self, Plot, Rect, AXES, OWN};

pub const DEF: SceneDef = SceneDef {
    id: "ftc",
    title: "The fundamental theorem of calculus",
    group: "Calculus",
    blurb: "The area so far, as a function: its slope is the curve you started with, and that is why integrals are found by undoing derivatives.",
    make: |_| Box::new(Ftc::create()),
};

// ----- the formula, in double precision -------------------------------------------------

/// A number to run the visitor's formula on in double precision. (`Expr` works in
/// whatever `Num` it is given; only its constants are read as f32.) Areas are sums of
/// many values, and their digits are written on the screen.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Wide(f64);

impl Num for Wide {
    fn cst(v: f32) -> Self {
        Wide(v as f64)
    }
    fn value(self) -> f32 {
        self.0 as f32
    }
    fn add(self, o: Self) -> Self {
        Wide(self.0 + o.0)
    }
    fn sub(self, o: Self) -> Self {
        Wide(self.0 - o.0)
    }
    fn mul(self, o: Self) -> Self {
        Wide(self.0 * o.0)
    }
    fn div(self, o: Self) -> Self {
        Wide(self.0 / o.0)
    }
    fn neg(self) -> Self {
        Wide(-self.0)
    }
    fn call(self, f: Func) -> Self {
        let x = self.0;
        Wide(match f {
            Func::Sin => x.sin(),
            Func::Cos => x.cos(),
            Func::Tan => x.tan(),
            Func::Asin => x.asin(),
            Func::Acos => x.acos(),
            Func::Atan => x.atan(),
            Func::Sinh => x.sinh(),
            Func::Cosh => x.cosh(),
            Func::Tanh => x.tanh(),
            Func::Exp => x.exp(),
            Func::Ln => x.ln(),
            Func::Log10 => x.log10(),
            Func::Log2 => x.log2(),
            Func::Sqrt => x.sqrt(),
            Func::Abs => x.abs(),
            // (as the formula's own arithmetic has it: 0 at 0)
            Func::Sign => {
                if x > 0.0 {
                    1.0
                } else if x < 0.0 {
                    -1.0
                } else if x == 0.0 {
                    0.0
                } else {
                    f64::NAN
                }
            }
            Func::Floor => x.floor(),
            Func::Ceil => x.ceil(),
            Func::Relu => x.max(0.0),
            Func::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Func::Gauss => (-x * x).exp(),
        })
    }
    fn powi(self, n: i32) -> Self {
        Wide(self.0.powi(n))
    }
    fn pow(self, o: Self) -> Self {
        Wide(self.0.powf(o.0))
    }
    fn atan2(self, o: Self) -> Self {
        Wide(self.0.atan2(o.0))
    }
    fn min(self, o: Self) -> Self {
        Wide(self.0.min(o.0))
    }
    fn max(self, o: Self) -> Self {
        Wide(self.0.max(o.0))
    }
}

/// The visitor's f at t.
fn value(e: &Expr, t: f64) -> f64 {
    e.eval_num(&[Wide(t)]).0
}

// ----- areas ----------------------------------------------------------------------------

/// Gauss–Legendre points and weights on [−1, 1], eight of them, found once by Newton's
/// method on the Legendre polynomial: a rule exact for polynomials up to degree 15.
fn gauss() -> &'static [(f64, f64); GL] {
    static NODES: std::sync::OnceLock<[(f64, f64); GL]> = std::sync::OnceLock::new();
    NODES.get_or_init(|| {
        let n = GL;
        let mut out = [(0.0, 0.0); GL];
        for (i, node) in out.iter_mut().enumerate() {
            let mut x = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
            let legendre = |x: f64| {
                let (mut p0, mut p1) = (1.0, x);
                for k in 2..=n {
                    let k = k as f64;
                    (p0, p1) = (p1, ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k);
                }
                // the value, and the slope
                (p1, n as f64 * (x * p1 - p0) / (x * x - 1.0))
            };
            for _ in 0..100 {
                let (p, dp) = legendre(x);
                let step = p / dp;
                x -= step;
                if step.abs() < 1e-16 {
                    break;
                }
            }
            let dp = legendre(x).1;
            *node = (x, 2.0 / ((1.0 - x * x) * dp * dp));
        }
        out
    })
}

const GL: usize = 8;

/// The narrowest stretch the areas are cut into near t: below this, the steps between
/// numbers are too coarse for halving to mean anything.
fn finest(t: f64) -> f64 {
    1e-10 * t.abs().max(1.0)
}

/// Why an area could not be taken.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Fault {
    /// f has no value somewhere between these two places.
    NoValue(f64, f64),
    /// f grows past any size near this place: no sum of thin strips settles there.
    Unbounded(f64),
}

impl Fault {
    /// Where it is, for a sentence.
    fn at(self) -> f64 {
        match self {
            Fault::NoValue(lo, hi) => 0.5 * (lo + hi),
            Fault::Unbounded(t) => t,
        }
    }
}

/// One rule's sum over a stretch, and what it met there.
#[derive(Clone, Copy, Debug)]
struct Est {
    sum: f64,
    /// How many of its points had no value; where it met an infinite one.
    gaps: usize,
    infinite: Option<f64>,
}

/// An area worked out by halving the stretches where the rule's two estimates (one
/// stretch, or its two halves) disagree, until they agree to a hair, with what
/// cannot be summed told apart: a gap in f, and f growing past any size.
struct Quad<'a> {
    e: &'a Expr,
    /// The disagreement accepted for a piece.
    tol: f64,
    /// The size of f's values: what "it grows" is measured against.
    scale: f64,
    evals: usize,
    budget: usize,
    /// The sum of the disagreements of the pieces accepted: a bound on the error.
    err: f64,
    /// Whether the budget ran out (pieces were then accepted as they stood).
    rough: bool,
}

impl<'a> Quad<'a> {
    /// `scale` is the typical size of f, `width` the width of what is being summed.
    fn new(e: &'a Expr, scale: f64, width: f64, budget: usize) -> Quad<'a> {
        Quad { e, tol: 1e-13 * (1.0 + scale * width), scale, evals: 0, budget, err: 0.0, rough: false }
    }

    fn est(&mut self, lo: f64, hi: f64) -> Est {
        let (half, mid) = (0.5 * (hi - lo), 0.5 * (hi + lo));
        let mut out = Est { sum: 0.0, gaps: 0, infinite: None };
        for &(x, w) in gauss() {
            let t = mid + half * x;
            let y = value(self.e, t);
            if y.is_nan() {
                out.gaps += 1;
            } else if y.is_infinite() {
                out.infinite = out.infinite.or(Some(t));
            } else {
                out.sum += w * y;
            }
        }
        out.sum *= half;
        self.evals += GL;
        out
    }

    /// The integral of f from `lo` to `hi` (either way round), or why there is none.
    fn area(&mut self, lo: f64, hi: f64) -> Result<f64, Fault> {
        if !(lo.is_finite() && hi.is_finite()) {
            return Err(Fault::NoValue(lo, hi));
        }
        if lo == hi {
            return Ok(0.0);
        }
        if hi < lo {
            return self.area(hi, lo).map(|v| -v);
        }
        let whole = self.est(lo, hi);
        self.piece(lo, hi, whole, hi - lo, false)
    }

    /// The area of one piece, halved where its estimates disagree. `reach` is the
    /// width the halving started from; `probed`, whether f has been looked at for
    /// growth on the way down to this piece.
    fn piece(&mut self, lo: f64, hi: f64, whole: Est, reach: f64, probed: bool) -> Result<f64, Fault> {
        let mid = 0.5 * (lo + hi);
        let (l, r) = (self.est(lo, mid), self.est(mid, hi));
        if let Some(t) = whole.infinite.or(l.infinite).or(r.infinite) {
            return Err(Fault::Unbounded(t));
        }
        let narrow = hi - lo <= finest(mid);
        let spent = self.evals >= self.budget;
        if whole.gaps + l.gaps + r.gaps > 0 {
            if (l.gaps == GL && r.gaps == GL) || narrow || spent {
                return Err(Fault::NoValue(lo, hi));
            }
            return Ok(self.piece(lo, mid, l, reach, probed)? + self.piece(mid, hi, r, reach, probed)?);
        }
        let two = l.sum + r.sum;
        let err = (two - whole.sum).abs();
        if err <= self.tol {
            self.err += err;
            return Ok(two);
        }
        if spent {
            self.rough = true;
            self.err += err;
            return Ok(two);
        }
        // Halved this far towards one point and still not agreed: f may grow past any
        // size there. Some such curves, ln|x| among them, have pieces that would agree
        // all the same (their area has a limit, as an improper integral); but no sum of
        // thin strips settles there, and that is what is asked (as in "Derivative,
        // integral, Taylor"). So f is looked at for growth here, once on each way down.
        let deep = hi - lo < 1e-5 * reach;
        if (deep && !probed) || narrow {
            if self.grows(mid, reach, hi - lo) {
                return Err(Fault::Unbounded(mid));
            }
            if narrow {
                // bounded, and the rest is negligible: a jump, a corner
                self.err += err;
                return Ok(two);
            }
        }
        let probed = probed || deep;
        Ok(self.piece(lo, mid, l, reach, probed)? + self.piece(mid, hi, r, reach, probed)?)
    }

    /// Does f grow without bound near `c`, which is within `within` of where it would?
    /// It is read at distances from `c` that halve each time, from `reach` down to a
    /// good deal more than `within` (nearer, the distance to `c` is not the distance
    /// to the point), and not below what the numbers can tell apart. A bounded f soon
    /// stops growing (a jump, a corner, a narrow but finite peak; a wild one like
    /// sin(1/x) never passes its bound); one that blows up keeps growing by at least
    /// as much at each halving (a pole by more each time, a logarithm by the same
    /// amount), and ends far above the size of f's other values.
    fn grows(&mut self, c: f64, reach: f64, within: f64) -> bool {
        let nearest = (32.0 * finest(c)).max(16.0 * within);
        let mut sizes: Vec<f64> = Vec::new();
        let mut r = reach;
        while r >= nearest && sizes.len() < 80 {
            let near = [c - r, c + r].map(|t| value(self.e, t).abs());
            sizes.push(near.into_iter().filter(|y| !y.is_nan()).fold(0.0, f64::max));
            r *= 0.5;
        }
        self.evals += 2 * sizes.len();
        let n = sizes.len();
        if n < 6 {
            return false;
        }
        let (s4, s3, s2, s0) = (sizes[n - 5], sizes[n - 4], sizes[n - 3], sizes[n - 1]);
        let accelerating = s0 > 1.05 * s3 && s2 > s4 && s0 - s2 >= 0.9 * (s2 - s4);
        accelerating && s0 > 3.0 * sizes[0].max(self.scale)
    }
}

/// A jump of f: its limits from the left and from the right at `at` differ.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Jump {
    at: f64,
    left: f64,
    right: f64,
}

impl Jump {
    /// Whether the two limits differ (not only f's value at the point itself).
    fn real(&self, scale: f64) -> bool {
        (self.left - self.right).abs() > 1e-9 * (1.0 + scale)
    }
}

/// Is there a jump of f between `lo` and `hi` (values `flo` and `fhi`): a step that
/// does not shrink as the stretch round it does? Found by halving towards the larger
/// difference until the stretch is as narrow as the numbers allow.
fn jump_between(e: &Expr, lo: f64, hi: f64, flo: f64, fhi: f64, scale: f64) -> Option<Jump> {
    let first = (fhi - flo).abs();
    let (mut l, mut r, mut fl, mut fr) = (lo, hi, flo, fhi);
    for _ in 0..90 {
        let m = 0.5 * (l + r);
        if r - l <= finest(m) {
            break;
        }
        let fm = value(e, m);
        if !fm.is_finite() {
            return None;
        }
        if (fm - fl).abs() >= (fr - fm).abs() {
            (r, fr) = (m, fm);
        } else {
            (l, fl) = (m, fm);
        }
    }
    let step = (fr - fl).abs();
    (step >= 0.5 * first && step > 1e-9 * (1.0 + scale) && fl.is_finite() && fr.is_finite()).then_some(Jump { at: 0.5 * (l + r), left: fl, right: fr })
}

/// The cells the view is cut into: the curve is drawn through their ends, and the
/// areas are built from theirs.
const CELLS: usize = 600;
/// The most values of f a grid may read, and an area between two places.
const GRID_BUDGET: usize = 400_000;
const PART_BUDGET: usize = 40_000;

/// What the areas over the view are built from: f at the ends of the cells, each
/// cell's area (or why it has none), and the jumps of f.
struct Grid {
    lo: f64,
    hi: f64,
    xs: Vec<f64>,
    fx: Vec<f64>,
    cells: Vec<Result<f64, Fault>>,
    /// `faulty[k]`: how many of the cells left of end `k` have no area.
    faulty: Vec<usize>,
    jumps: Vec<Jump>,
    /// Where f grows past any size.
    poles: Vec<f64>,
    /// The typical size of f's values, a bound on the error of the cells' areas,
    /// and whether some were left rougher than asked.
    scale: f64,
    err: f64,
    rough: bool,
}

impl Grid {
    fn build(e: &Expr, lo: f64, hi: f64) -> Grid {
        let xs: Vec<f64> = (0..=CELLS).map(|k| lo + (hi - lo) * k as f64 / CELLS as f64).collect();
        let fx: Vec<f64> = xs.iter().map(|&t| value(e, t)).collect();
        let sizes: Vec<f64> = fx.iter().filter(|v| v.is_finite()).map(|v| v.abs()).collect();
        let scale = if sizes.is_empty() { 1.0 } else { sizes.iter().sum::<f64>() / sizes.len() as f64 };
        let mut q = Quad::new(e, scale, hi - lo, GRID_BUDGET);
        let cells: Vec<Result<f64, Fault>> = xs.windows(2).map(|p| q.area(p[0], p[1])).collect();
        let mut faulty = vec![0usize; CELLS + 1];
        for k in 0..CELLS {
            faulty[k + 1] = faulty[k] + cells[k].is_err() as usize;
        }
        // jumps: a step between two ends much larger than the steps round it (for a
        // jump that falls on an end, with f there between the two sides, the steps
        // either side of the end are both large: so also measured against the next
        // ones out), confirmed by closing in on it
        let step = |k: usize| fx.get(k + 1).zip(fx.get(k)).map_or(0.0, |(b, a)| (b - a).abs()).min(f64::MAX);
        let mut jumps: Vec<Jump> = Vec::new();
        for k in 0..CELLS {
            if matches!(cells[k], Err(Fault::Unbounded(_))) {
                continue;
            }
            let d = step(k);
            let near = step(k.wrapping_sub(1)).max(step(k + 1));
            let next = step(k.wrapping_sub(2)).max(step(k + 2));
            if !(d.is_finite() && d > 4.0 * near.min(next) && d > 1e-9 * (1.0 + scale)) {
                continue;
            }
            let Some(mut j) = jump_between(e, xs[k], xs[k + 1], fx[k], fx[k + 1], scale) else { continue };
            // (one that falls on an end is found from both sides of it: one jump)
            for &end in &xs[k..=k + 1] {
                if (j.at - end).abs() <= 100.0 * finest(end) {
                    j.at = end;
                }
            }
            match jumps.last_mut() {
                Some(last) if (last.at - j.at).abs() <= 100.0 * finest(j.at) => last.right = j.right,
                _ => jumps.push(j),
            }
        }
        let poles = cells.iter().filter_map(|c| if let Err(Fault::Unbounded(t)) = c { Some(*t) } else { None }).collect();
        Grid { lo, hi, xs, fx, cells, faulty, jumps, poles, scale, err: q.err, rough: q.rough }
    }

    fn width(&self) -> f64 {
        (self.hi - self.lo) / CELLS as f64
    }

    /// The cell `t` is in (the nearest one, if it is outside the view).
    fn cell_of(&self, t: f64) -> usize {
        let k = ((t - self.lo) / self.width()).floor();
        if k >= 0.0 {
            (k as usize).min(CELLS - 1)
        } else {
            0
        }
    }

    fn quad<'a>(&self, e: &'a Expr) -> Quad<'a> {
        Quad::new(e, self.scale, self.hi - self.lo, PART_BUDGET)
    }

    /// The integral of f from `p` to `q` (either way round), or the fault nearest `p`
    /// that stops it; with a bound on its error.
    fn area(&self, e: &Expr, p: f64, q: f64) -> (Result<f64, Fault>, f64) {
        let mut quad = self.quad(e);
        let (lo, hi, sign) = if q >= p { (p, q, 1.0) } else { (q, p, -1.0) };
        let (i, j) = (self.cell_of(lo), self.cell_of(hi));
        if j <= i + 1 {
            let v = quad.area(lo, hi).map(|v| sign * v);
            return (v, quad.err);
        }
        // whole cells from end i + 1 to end j, and the parts at either end
        let between = self.faulty[j] - self.faulty[i + 1];
        if between > 0 {
            let mut bad = (i + 1..j).filter_map(|k| self.cells[k].err());
            let fault = if sign > 0.0 { bad.next() } else { bad.next_back() };
            return (Err(fault.unwrap_or(Fault::NoValue(lo, hi))), 0.0);
        }
        let whole: f64 = self.cells[i + 1..j].iter().map(|c| *c.as_ref().unwrap_or(&0.0)).sum();
        let (head, tail) = if sign > 0.0 {
            let head = quad.area(lo, self.xs[i + 1]);
            (head, head.and_then(|_| quad.area(self.xs[j], hi)))
        } else {
            let tail = quad.area(self.xs[j], hi);
            (tail.and_then(|_| quad.area(lo, self.xs[i + 1])), tail)
        };
        let v = head.and_then(|h| tail.map(|t| sign * (h + whole + t)));
        (v, quad.err + self.err)
    }

    /// The area from `a` to each end of the cells, as far each way as it can be taken
    /// (NaN beyond), with a bound on its error.
    fn curve(&self, e: &Expr, a: f64) -> (Vec<f64>, f64) {
        let mut out = vec![f64::NAN; CELLS + 1];
        let i = self.cell_of(a);
        let mut quad = self.quad(e);
        if let Ok(v) = quad.area(a, self.xs[i + 1]) {
            out[i + 1] = v;
            for k in i + 2..=CELLS {
                match self.cells[k - 1] {
                    Ok(c) => out[k] = out[k - 1] + c,
                    Err(_) => break,
                }
            }
        }
        if let Ok(v) = quad.area(self.xs[i], a) {
            out[i] = -v;
            for k in (0..i).rev() {
                match self.cells[k] {
                    Ok(c) => out[k] = out[k + 1] - c,
                    Err(_) => break,
                }
            }
        }
        (out, quad.err + self.err)
    }

    /// The jump of f at `t`, if `t` is on one.
    fn jump_at(&self, t: f64) -> Option<Jump> {
        self.jumps.iter().copied().find(|j| (j.at - t).abs() <= 100.0 * finest(t))
    }

    /// The faults in the view, one for each run of cells without an area.
    fn faults(&self) -> Vec<Fault> {
        let mut out: Vec<Fault> = Vec::new();
        for (k, c) in self.cells.iter().enumerate() {
            if let Err(fault) = c {
                let joined = k > 0 && self.cells[k - 1].is_err();
                match (joined, out.last_mut(), fault) {
                    (true, Some(Fault::NoValue(_, hi)), Fault::NoValue(_, h)) => *hi = h.max(*hi),
                    (true, Some(Fault::Unbounded(_)), Fault::Unbounded(_)) => {}
                    _ => out.push(*fault),
                }
            }
        }
        out
    }
}

// ----- the curves on the buttons, and the things to try ---------------------------------

/// An antiderivative: as it is written on the screen, and as it is computed.
type Anti = (&'static str, fn(f64) -> f64);

/// A curve on the buttons, with an antiderivative where there is one in closed form.
struct Curve {
    button: &'static str,
    formula: &'static str,
    /// F(x) as written on the screen, and as computed.
    anti: Option<Anti>,
}

const PRESETS: [Curve; 8] = [
    Curve { button: "sin", formula: "sin(x)", anti: Some((r"-\cos x", |x| -x.cos())) },
    Curve { button: "cos", formula: "cos(x)", anti: Some((r"\sin x", |x| x.sin())) },
    Curve { button: "$x^2 - 1$", formula: "x^2 - 1", anti: Some((r"x^3/3 - x", |x| x * x * x / 3.0 - x)) },
    Curve { button: "cubic", formula: "(x^3 - 9x)/8", anti: Some((r"x^4/32 - 9x^2/16", |x| x.powi(4) / 32.0 - 9.0 * x * x / 16.0)) },
    Curve { button: "bell", formula: "exp(-x^2/2)", anti: None },
    Curve { button: "step", formula: "sign(x)", anti: Some((r"|x|", |x| x.abs())) },
    Curve { button: "$1/x$", formula: "1/x", anti: Some((r"\ln|x|", |x| x.abs().ln())) },
    Curve { button: "$e^x$", formula: "exp(x)", anti: Some((r"e^x", |x| x.exp())) },
];

const COS: usize = 1;
const SIN: usize = 0;
const CUBIC: usize = 3;
const BELL: usize = 4;
const STEP: usize = 5;

/// The preset whose formula this is (spaces aside), if any.
fn preset_of(text: &str) -> Option<usize> {
    let bare = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
    let typed = bare(text);
    PRESETS.iter().position(|c| bare(c.formula) == typed)
}

/// What a film moves.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Which {
    Start,
    End,
    /// The strip's width, by its logarithm.
    LogWidth,
}

/// One of the "things to try": everything the lesson depends on.
struct Trial {
    button: &'static str,
    caption: &'static str,
    preset: usize,
    a: f64,
    x: f64,
    h: f64,
    anti: bool,
    /// What plays once the picture has arrived: what moves, from, to, in seconds.
    film: Option<(Which, f64, f64, f64)>,
}

/// The view every thing to try is seen in: x from −4 to 4.
const TRIAL_HALF: f64 = 4.0;

const TRIALS: [Trial; 5] = [
    Trial {
        button: "the area's slope is the height",
        caption: r"The orange strip from $x$ to $x + h$ is what $A$ gains as $x$ moves on by $h$: below, the secant's rise. Its area over its width is its mean height, the secant's slope. As $h$ shrinks from 1 to 0.01 that goes 0.435, 0.653 (at 0.5), 0.796 (at 0.1), 0.822, closing in on the curve's height $f(0.6) = 0.825$: the slope of the yellow tangent.",
        preset: COS,
        a: -3.0,
        x: 0.6,
        h: 1.0,
        anti: false,
        film: Some((Which::LogWidth, 0.0, -2.0, 5.0)),
    },
    Trial {
        button: "below the axis, the area falls",
        caption: r"$x$ sweeps from $-3.8$ to 3.8 under the cubic $(x^3 - 9x)/8$. Where the shading is pink, $f < 0$ and $A$ falls; where it is blue, $A$ climbs. $A$ has its bottoms where $f$ crosses the axis going up ($x = -3$ and 3, $A = -0.925$) and its top where $f$ crosses going down ($x = 0$, $A = 1.606$). It is steepest at the ends, where $|f|$ is largest (2.58), and comes back to 0: the pink and blue areas cancel.",
        preset: CUBIC,
        a: -3.8,
        x: -3.8,
        h: 0.25,
        anti: false,
        film: Some((Which::End, -3.8, 3.8, 7.0)),
    },
    Trial {
        button: "move the start: the curve only shifts",
        caption: r"The start $a$ slides from $-3$ to 0.5 under the bell $e^{-x^2/2}$. The whole curve $A$ sinks by 1.730, the area between the two starts, and keeps its shape: at $x = 1$ it goes from 2.106 to 0.376, and the tangent there keeps its slope $f(1) = 0.607$. Whatever $a$ is, $A(a) = 0$.",
        preset: BELL,
        a: -3.0,
        x: 1.0,
        h: 0.25,
        anti: false,
        film: Some((Which::Start, -3.0, 0.5, 4.0)),
    },
    Trial {
        button: "a jump in $f$ is a corner in $A$",
        caption: r"The step $\operatorname{sign}(x)$ jumps from $-1$ to 1 at 0 (its value at 0 itself is 0). $A$ does not jump: a strip's area is at most its width times 1, so it shrinks with the width. $A$ turns a corner instead, falling at slope $-1$ to $A(0) = -2.5$ and climbing at slope 1 after. There is no tangent at the corner; the secant, which reaches to the right, has slope 1 for every $h$, not $f(0) = 0$.",
        preset: STEP,
        a: -2.5,
        x: 0.0,
        h: 0.5,
        anti: false,
        film: None,
    },
    Trial {
        button: "an area by undoing a derivative",
        caption: r"$F(x) = -\cos x$ has slope $\sin x$, which is $f$, so it climbs exactly as $A$ does: the dashed purple curve is $A$ moved down by 0.878 (that is $F(a) = -\cos 0.5$) at every $x$. So the area from $a = 0.5$ to $x = 2.5$ is $F(2.5) - F(0.5) = -\cos 2.5 + \cos 0.5 = 1.6787$, and the sum of thin strips gives the same digits.",
        preset: SIN,
        a: 0.5,
        x: 2.5,
        h: 0.25,
        anti: true,
        film: None,
    },
];

/// The scene as it opens.
const OPENING: (usize, f64, f64, f64) = (COS, -3.0, 0.6, 0.4);

const H_MIN: f64 = 0.001;
const H_MAX: f64 = 2.0;
const HALF_MIN: f64 = 1.0;
const HALF_MAX: f64 = 20.0;

// ----- what glides ----------------------------------------------------------------------

/// How fast what is drawn follows what is asked for: each of two eases one after the
/// other covers two thirds of what is left in this many seconds.
const FOLLOW: f64 = 0.09;

/// Numbers that follow where they are asked to be through two eases one after the
/// other, so that they set off gently and arrive exactly.
#[derive(Clone, Copy, Debug)]
struct Follow<const N: usize> {
    to: [f64; N],
    between: [f64; N],
    shown: [f64; N],
}

impl<const N: usize> Follow<N> {
    fn at(v: [f64; N]) -> Self {
        Follow { to: v, between: v, shown: v }
    }

    fn arrived(&self) -> bool {
        self.shown == self.to && self.between == self.to
    }

    /// Number `i` set at once, in all three: what a hand moves.
    fn now(&mut self, i: usize, v: f64) {
        if i < N && v.is_finite() {
            (self.to[i], self.between[i], self.shown[i]) = (v, v, v);
        }
    }

    fn step(&mut self, dt: f64) {
        if self.arrived() || dt <= 0.0 {
            return;
        }
        let k = 1.0 - (-dt / FOLLOW).exp();
        let mut apart = 0.0f64;
        for i in 0..N {
            self.between[i] += (self.to[i] - self.between[i]) * k;
            self.shown[i] += (self.between[i] - self.shown[i]) * k;
            let near = 1e-5 * self.to[i].abs().max(1.0);
            apart = apart.max((self.to[i] - self.between[i]).abs() / near).max((self.to[i] - self.shown[i]).abs() / near);
        }
        // (exactly there in the end: the picture then stops changing)
        if apart <= 1.0 || !apart.is_finite() {
            *self = Follow::at(self.to);
        }
    }
}

/// A film: one setting moved from one value to another, easing in and out.
#[derive(Clone, Copy, Debug)]
struct Film {
    which: Which,
    from: f64,
    to: f64,
    seconds: f64,
    /// Scene time since it started; it starts once the picture has arrived.
    t: f64,
    waiting: bool,
}

impl Film {
    fn at(&self) -> f64 {
        let s = (self.t / self.seconds.max(1e-9)).clamp(0.0, 1.0);
        self.from + (self.to - self.from) * s * s * (3.0 - 2.0 * s)
    }
}

/// The ranges a chart shows, gliding: kept while what is drawn fills enough of them,
/// so that a curve that moves is seen to move.
fn keep_range(goal: [f64; 2], data: [f64; 2]) -> [f64; 2] {
    let (lo, hi) = (data[0].min(0.0), data[1].max(0.0));
    if !(lo.is_finite() && hi.is_finite()) {
        return goal;
    }
    // (kept while the data stay a little way inside it, and fill enough of it)
    let inset = 0.04 * (goal[1] - goal[0]);
    let fits = lo >= goal[0] + inset && hi <= goal[1] - inset;
    let used = (hi - lo) / (goal[1] - goal[0]).max(1e-300);
    if fits && used >= 0.55 {
        return goal;
    }
    // (nothing but zero to show: a range round it)
    if hi - lo < 1e-12 {
        return [-1.0, 1.0];
    }
    let span = hi - lo;
    [lo - 0.15 * span, hi + 0.15 * span]
}

// ----- what is worked out for the picture and the numbers -------------------------------

/// The numbers at one setting of the ends and the strip.
#[derive(Clone, Copy, Debug)]
struct Look {
    a: f64,
    x: f64,
    h: f64,
    /// f at x.
    fx: f64,
    /// A(x), and the strip's area A(x + h) − A(x).
    area: Result<f64, Fault>,
    strip: Result<f64, Fault>,
    /// The slope of A at x: one, or one from each side at a corner (None where A has
    /// no value, or f none near x).
    slope: Option<(f64, f64)>,
    /// F(x) − F(a), where the formula is one whose F is known.
    anti: Option<f64>,
    /// A bound on the error of the areas.
    err: f64,
}

impl Look {
    /// The secant's slope, (A(x + h) − A(x))/h: none where A has no value at x.
    fn secant(&self) -> Option<f64> {
        self.area.and(self.strip).ok().map(|s| s / self.h)
    }

    /// Whether x is on a corner of A.
    fn corner(&self) -> bool {
        self.slope.is_some_and(|(l, r)| l != r)
    }
}

/// The decimals an area is written to: as many as its error leaves right.
fn decimals(err: f64) -> usize {
    if err < 2e-5 {
        4
    } else if err < 2e-4 {
        3
    } else if err < 2e-3 {
        2
    } else {
        1
    }
}

/// A number to `d` decimals as a formula (large ones in powers of ten).
fn number(v: f64, d: usize) -> String {
    if !v.is_finite() {
        return "?".into();
    }
    if v.abs() >= 1e5 {
        return format!("{v:.3e}");
    }
    num(v, d)
}

/// A place on the x axis, for a sentence.
fn place(t: f64) -> String {
    num(t, 2)
}

// ----- colours, and kinds of line -------------------------------------------------------

/// The curve f, as in "Derivative, integral, Taylor".
const CURVE: Color = pal::CYAN;
/// Area that adds to A, and area that takes from it (as that scene's rectangles).
const PLUS: Color = pal::BLUE;
const MINUS: Color = pal::PINK;
/// The start a.
const START: Color = pal::WHITE;
/// The moving end x, and the tangent to A there (the tangent's colour there too).
const END: Color = pal::YELLOW;
/// The strip from x to x + h, and the secant over it.
const STRIP: Color = pal::ORANGE;
/// A, the area so far.
const AREA: Color = pal::GREEN;
/// An antiderivative F.
const ANTI: Color = pal::PURPLE;
/// Where no area can be taken across.
const CUT: Color = pal::RED;

// Lines whose number of points changes from frame to frame, each in a batch of its own
// (see `plot::apart`).
const F_LINE: u8 = OWN;
const A_LINE: u8 = OWN + 1;
const ANTI_LINE: u8 = OWN + 2;
const TANGENT: u8 = OWN + 3;
const SECANT: u8 = OWN + 4;
const STRIP_LINE: u8 = OWN + 5;
const CUT_LINE: u8 = OWN + 6;
const ZOOM_LINE: u8 = OWN + 7;
const RINGS: u8 = OWN + 8;
const ENDS: u8 = OWN + 9;
const GAP: u8 = OWN + 10;
/// Each chart's axes and grid, apart: as one chart's range glides, its lines come and
/// go, and in a batch shared with the other's they would renumber those.
const TOP_AXES: u8 = OWN + 11;
const LOW_AXES: u8 = OWN + 12;

/// Every kind of line and point this scene draws, asked for in the same order at the
/// start of every frame, each with one unseen point: a batch not drawn in a frame would
/// drop out and move every later one up a place (and `eidos motion` would take one for
/// another).
fn hold_batches(f: &mut Frame) {
    let corner = f.hud(Vec2::ZERO);
    for kind in [AXES, plot::CURVES, plot::MARKS, plot::BUTTONS, F_LINE, A_LINE, ANTI_LINE, TANGENT, SECANT, STRIP_LINE, CUT_LINE, ZOOM_LINE, RINGS, ENDS, GAP, TOP_AXES, LOW_AXES] {
        plot::apart(f, kind, false).segment(corner, corner, 1.0, Color::CLEAR);
    }
    for kind in [STRIP_LINE, CUT_LINE, ZOOM_LINE, ENDS, GAP] {
        plot::apart(f, kind, true).segment(corner, corner, 1.0, Color::CLEAR);
    }
    plot::disc(f, Vec2::ZERO, 1.0, Color::CLEAR);
}

/// A curve in screen space (its points from left to right) drawn in dashes that keep
/// their places across the screen: each a short chord over a fixed stretch of columns,
/// so that a curve that changes shape keeps as many dashes, each where it was. (A
/// dashed batch starts its pattern afresh at every segment, which leaves a curve of
/// short segments solid; dashes measured along the curve slide along it as its length
/// changes, and one gained at an end moves every other up a place.) In the batch of
/// its kind; a point that is not a number breaks it.
fn dashes(f: &mut Frame, kind: u8, pts: &[Vec2], width: f32, color: Color) {
    let (period, on) = (10.0f32, 5.5f32);
    let pieces: Vec<(Vec2, Vec2)> = pts.windows(2).map(|w| (w[0], w[1])).filter(|(a, b)| a.is_finite() && b.is_finite() && b.x >= a.x).collect();
    let (Some(first), Some(last)) = (pieces.first(), pieces.last()) else { return };
    let (from, to) = (first.0.x, last.1.x);
    // which unbroken run of the curve each piece is in
    let mut runs = vec![0usize; pieces.len()];
    for j in 1..pieces.len() {
        runs[j] = runs[j - 1] + (pieces[j].0 != pieces[j - 1].1) as usize;
    }
    // (the columns grow from left to right, and so does the piece they are in)
    let mut k = 0usize;
    let mut height = |x: f32| -> Option<(f32, usize)> {
        while k < pieces.len() && pieces[k].1.x < x {
            k += 1;
        }
        let (a, b) = *pieces.get(k)?;
        if a.x > x {
            return None;
        }
        let t = if b.x > a.x { (x - a.x) / (b.x - a.x) } else { 0.0 };
        Some((a.y + (b.y - a.y) * t, runs[k]))
    };
    let mut x = (from / period).ceil() * period;
    while x + on <= to {
        if let (Some((y0, r0)), Some((y1, r1))) = (height(x), height(x + on)) {
            if r0 == r1 {
                plot::line_of(f, kind, Vec2::new(x, y0), Vec2::new(x + on, y1), width, color);
            }
        }
        x += period;
    }
}

/// A chart's axes and grid, in a batch of their own. The round values up the side are
/// chosen from where its range is going (`goal`), so that they keep their places while
/// it glides there.
fn axes(f: &mut Frame, p: &Plot, goal: [f64; 2], kind: u8, x_name: &str, size: f32) {
    if p.tiny() {
        return;
    }
    let (min, max) = p.rect;
    let dim = pal::DIM.a(0.8);
    plot::line_of(f, kind, min, Vec2::new(min.x, max.y), 1.0, dim);
    let ticks = plot::Ticks::of(goal[0], goal[1], p.height() * 1.6);
    for v in ticks.values().filter(|v| *v >= p.y[0] - 1e-9 && *v <= p.y[1] + 1e-9) {
        let y = p.px_y(v);
        plot::line_of(f, kind, Vec2::new(min.x, y), Vec2::new(max.x, y), 1.0, pal::DIM.a(0.14));
        plot::line_of(f, kind, Vec2::new(min.x - 4.0, y), Vec2::new(min.x, y), 1.0, dim);
        f.text_px(Vec2::new(min.x - 6.0, y), ticks.name(v), size, pal::DIM, Align2::RIGHT_CENTER);
    }
    plot::line_of(f, kind, Vec2::new(min.x, max.y), max, 1.0, dim);
    let ticks = plot::Ticks::of(p.x[0], p.x[1], p.width());
    for v in ticks.values() {
        let x = p.px_x(v);
        plot::line_of(f, kind, Vec2::new(x, max.y), Vec2::new(x, max.y + 4.0), 1.0, dim);
        f.text_px(Vec2::new(x, max.y + 6.0), ticks.name(v), size, pal::DIM, Align2::CENTER_TOP);
    }
    if !x_name.is_empty() {
        f.text_px(Vec2::new((min.x + max.x) * 0.5, max.y + 8.0 + size), x_name, size, pal::DIM, Align2::CENTER_TOP);
    }
}

/// A translucent quadrilateral in screen space (corners in order), over a chart's backing.
fn shade(f: &mut Frame, c: [Vec2; 4], color: Color) {
    if c.iter().all(|p| p.is_finite()) {
        let c = c.map(|p| f.hud_at(1, p));
        f.dl.tris_alpha.quad(c[0], c[1], c[2], c[3], color);
    }
}

// ----- where everything goes -------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Layout {
    view: Vec2,
    narrow: bool,
    /// Only the two charts: the window is too small for the rest.
    bare: bool,
    /// Text sizes: headings, words, small print.
    big: f32,
    text: f32,
    small: f32,
    /// The two charts' rectangles, and how much room is above each for its title.
    top: Rect,
    bottom: Rect,
    title: f32,
    /// The knob for h (its track: at least 18 pixels tall, or `Plot` offers no
    /// handle on it), the numbers, the strip magnified, and a note on
    /// how to read the picture (empty where there is no room for them).
    knob: Rect,
    numbers: Rect,
    zoom: Rect,
    note: Rect,
}

const NONE: Rect = (Vec2::ZERO, Vec2::ZERO);

fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Rect {
    (Vec2::new(x0, y0), Vec2::new(x1.max(x0 + 1.0), y1.max(y0 + 1.0)))
}

fn empty(r: Rect) -> bool {
    !(r.1.x - r.0.x > 8.0 && r.1.y - r.0.y > 4.0)
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let view = view.max(Vec2::splat(1.0));
        let narrow = view.x < 700.0;
        let (big, text, small) = if narrow { (13.0, 11.5, 10.5) } else { (15.0, 12.5, 11.5) };
        let inset = inset.max(0.0);
        let mut lay = Layout { view, narrow, bare: false, big, text, small, top: NONE, bottom: NONE, title: 0.0, knob: NONE, numbers: NONE, zoom: NONE, note: NONE };
        let bare = |lay: &mut Layout| {
            // the charts alone, one above the other, with just room for their numbers
            let top0 = inset + 6.0;
            let (x0, x1) = (34.0f32.min(view.x * 0.2), view.x - 8.0);
            let h = ((view.y - top0 - 8.0 - 20.0 - 22.0) * 0.5).max(1.0);
            lay.bare = true;
            lay.narrow = true;
            lay.top = rect(x0, top0, x1, top0 + h);
            lay.bottom = rect(x0, top0 + h + 20.0, x1, top0 + 2.0 * h + 20.0);
        };
        if narrow {
            let side = 10.0;
            let top0 = inset + 8.0;
            let (x0, x1) = (side + 38.0, view.x - side - 10.0);
            let title = 24.0;
            let lines = 3.0 * (text + 6.0);
            let block = lines + 12.0 + 60.0;
            let avail = view.y - 8.0 - block - top0 - 2.0 * title - 22.0 - 34.0;
            if avail < 150.0 || view.x < 300.0 {
                bare(&mut lay);
                return lay;
            }
            let ch = avail * 0.5;
            lay.title = title;
            lay.top = rect(x0, top0 + title, x1, top0 + title + ch);
            let y = lay.top.1.y + 22.0 + title;
            lay.bottom = rect(x0, y, x1, y + ch);
            let y = lay.bottom.1.y + 36.0;
            lay.numbers = rect(side, y, view.x - side, y + lines);
            let y = lay.numbers.1.y + 30.0;
            lay.knob = rect(side + 30.0, y, view.x - side - 24.0, y + 18.0);
        } else {
            let side = 22.0;
            let top0 = inset + 14.0;
            let column = (view.x * 0.3).max(270.0).min(350.0);
            let (c0, c1) = (view.x - side - column, view.x - side);
            let (x0, x1) = (side + 48.0, c0 - 44.0);
            let title = 30.0;
            let avail = view.y - 14.0 - top0 - 2.0 * title - 26.0 - 44.0;
            if avail < 200.0 {
                bare(&mut lay);
                return lay;
            }
            let ch = avail * 0.5;
            lay.title = title;
            lay.top = rect(x0, top0 + title, x1, top0 + title + ch);
            let y = lay.top.1.y + 26.0 + title;
            lay.bottom = rect(x0, y, x1, y + ch);
            lay.numbers = rect(c0, top0, c1, top0 + 236.0);
            let y = lay.numbers.1.y + 44.0;
            lay.knob = rect(c0 + 16.0, y, c1 - 20.0, y + 18.0);
            let y = lay.knob.1.y + 70.0;
            if view.y - 20.0 - y > 200.0 {
                lay.zoom = rect(c0 + 34.0, y, c1 - 12.0, y + 120.0);
                let y = lay.zoom.1.y + 76.0;
                if view.y - 16.0 - y > 90.0 {
                    lay.note = rect(c0, y, c1, view.y - 16.0);
                }
            }
        }
        lay
    }
}

// ----- the scene ------------------------------------------------------------------------

pub struct Ftc {
    f: ExprField,
    /// The formula and the view the grid was built for, and how many grids have been
    /// built (what the numbers kept below were worked out for).
    built: Option<(String, f64)>,
    grid: Grid,
    version: u64,
    /// The numbers last worked out, by the ends and the strip they are for: the
    /// panel's and the picture's, which at rest are the same (the sums can be dear).
    kept: std::cell::RefCell<Vec<([u64; 4], Look)>>,
    /// The ends and the strip's width as set (the panel's numbers are theirs), and the
    /// view: x from −half to half.
    a: f64,
    x: f64,
    h: f64,
    half: f64,
    show_anti: bool,
    /// The ends and the logarithm of the strip's width as drawn, following the set ones.
    drawn: Follow<3>,
    film: Option<Film>,
    /// How far the picture has gone from the curves drawn when the formula last
    /// changed (`old_f`, `old_a`) to the new ones: 0 to 1.
    morph: Follow<1>,
    old_f: Vec<f64>,
    old_a: Vec<f64>,
    /// The curves as drawn in the last frame, for a morph to start from.
    last_f: Vec<f64>,
    last_a: Vec<f64>,
    /// The two charts' ranges of y as drawn (lower and upper, for f and then for A),
    /// and where they are going.
    ranges: Follow<4>,
    goals: [[f64; 2]; 2],
    /// While a film moves the start, the lower chart's range only grows: the whole
    /// curve is seen to move.
    held: bool,
    /// The slopes the tangent is drawn with (from the left and from the right): they
    /// follow A's at once while that changes smoothly, and glide where it leaps (the
    /// end arriving on a jump of f, a new formula), so that the line swings round.
    slopes: Follow<2>,
    /// What the two charts draw goes from and to (where f grows past any size aside).
    extent: [[f64; 2]; 2],
    /// A at x and at x + h as drawn (NaN where there is none): on the drawn curve,
    /// also while it is on its way to a new formula's.
    anchor: [f64; 2],
    canvas: Option<Vec2>,
    dragging: bool,
}

impl Ftc {
    fn create() -> Ftc {
        let (preset, a, x, h) = OPENING;
        let f = ExprField::new(PRESETS[preset].formula, &["x"]);
        let grid = Grid::build(&f.expr, -TRIAL_HALF, TRIAL_HALF);
        let mut s = Ftc {
            built: Some((f.text.clone(), TRIAL_HALF)),
            f,
            grid,
            version: 0,
            kept: std::cell::RefCell::new(Vec::new()),
            a,
            x,
            h,
            half: TRIAL_HALF,
            show_anti: false,
            drawn: Follow::at([a, x, h.log10()]),
            film: None,
            morph: Follow::at([1.0]),
            old_f: Vec::new(),
            old_a: Vec::new(),
            last_f: Vec::new(),
            last_a: Vec::new(),
            // (empty: the first frame fits them to what is drawn)
            ranges: Follow::at([-1.0, 1.0, -1.0, 1.0]),
            goals: [[0.0, 0.0], [0.0, 0.0]],
            held: false,
            slopes: Follow::at([0.0, 0.0]),
            extent: [[1.0, 0.0], [1.0, 0.0]],
            anchor: [f64::NAN; 2],
            canvas: None,
            dragging: false,
        };
        s.ensure_grid();
        s
    }

    /// The grid for the formula and the view as they are now.
    fn ensure_grid(&mut self) {
        let key = (self.f.text.clone(), self.half);
        if self.built.as_ref() != Some(&key) {
            self.grid = Grid::build(&self.f.expr, -self.half, self.half);
            self.built = Some(key);
            self.version += 1;
        }
    }

    /// The antiderivative the scene knows for the formula as typed, if any.
    fn known(&self) -> Option<usize> {
        preset_of(&self.f.text).filter(|&i| PRESETS[i].anti.is_some())
    }

    /// The formula has changed: the picture glides from the curves drawn now.
    fn formula_changed(&mut self) {
        self.old_f = self.last_f.clone();
        self.old_a = self.last_a.clone();
        self.morph = Follow::at([0.0]);
        self.morph.to = [1.0];
        self.aim_ahead();
    }

    fn choose(&mut self, preset: usize) {
        if let Some(c) = PRESETS.get(preset) {
            if self.f.text != c.formula {
                self.f.set(c.formula);
                self.formula_changed();
            }
        }
    }

    /// Set up experiment `k` of `TRIALS`: glide to it, then play its film.
    fn set_up(&mut self, k: usize) {
        let Some(t) = TRIALS.get(k) else { return };
        self.choose(t.preset);
        if self.half != TRIAL_HALF {
            self.half = TRIAL_HALF;
            self.ensure_grid();
        }
        (self.a, self.x, self.h, self.show_anti) = (t.a, t.x, t.h, t.anti);
        self.drawn.to = [t.a, t.x, t.h.log10()];
        self.film = t.film.map(|(which, from, to, seconds)| Film { which, from, to, seconds, t: 0.0, waiting: true });
        self.held = false;
        self.aim_ahead();
    }

    /// Where a film leaves things (for the tests: the state its caption speaks of).
    #[cfg(test)]
    fn finish_film(&mut self) {
        if let Some(film) = self.film.take() {
            self.set(film.which, film.to);
        }
        self.drawn = Follow::at(self.drawn.to);
    }

    /// One setting, set by the hand or a film: at once, as set and as drawn.
    fn set(&mut self, which: Which, v: f64) {
        match which {
            Which::Start => {
                self.a = v.max(-self.half).min(self.half);
                self.drawn.now(0, self.a);
            }
            Which::End => {
                self.x = v.max(-self.half).min(self.half);
                self.drawn.now(1, self.x);
            }
            Which::LogWidth => {
                self.h = 10f64.powf(v).max(H_MIN).min(H_MAX);
                self.drawn.now(2, self.h.log10());
            }
        }
    }

    /// The numbers at the ends `a`, `x` and the strip's width `h`.
    fn measure(&self, a: f64, x: f64, h: f64) -> Look {
        let key = [a.to_bits(), x.to_bits(), h.to_bits(), self.version];
        if let Some((_, look)) = self.kept.borrow().iter().find(|(k, _)| *k == key) {
            return *look;
        }
        let look = self.work_out(a, x, h);
        let mut kept = self.kept.borrow_mut();
        if kept.len() >= 4 {
            kept.remove(0);
        }
        kept.push((key, look));
        look
    }

    fn work_out(&self, a: f64, x: f64, h: f64) -> Look {
        let e = &self.f.expr;
        let g = &self.grid;
        let fx = value(e, x);
        let (area, e1) = g.area(e, a, x);
        let (strip, e2) = g.area(e, x, x + h);
        // the slope of A at x: f(x) where f is continuous there; one from each side on
        // a jump of f; the limit at a single point where f's value is off its curve
        let slope = match g.jump_at(x) {
            Some(j) if j.real(g.scale) => Some((j.left, j.right)),
            Some(j) => Some((j.left, j.left)),
            None if fx.is_finite() => Some((fx, fx)),
            None => {
                let d = 1e-7 * x.abs().max(1.0);
                let (l, r) = (value(e, x - d), value(e, x + d));
                (l.is_finite() && r.is_finite() && (l - r).abs() <= 1e-4 * (1.0 + l.abs())).then_some((0.5 * (l + r), 0.5 * (l + r)))
            }
        };
        let slope = slope.filter(|_| area.is_ok());
        let anti = self.known().and_then(|i| PRESETS[i].anti).and_then(|(_, big_f)| area.ok().map(|_| big_f(x) - big_f(a))).filter(|v| v.is_finite());
        Look { a, x, h, fx, area, strip, slope, anti, err: e1.max(e2) }
    }

    /// The numbers as set (what the panel says).
    fn set_look(&self) -> Look {
        self.measure(self.a, self.x, self.h)
    }

    /// The ends and the strip as drawn.
    fn drawn_look(&self) -> Look {
        let [a, x, lh] = self.drawn.shown;
        self.measure(a, x, 10f64.powf(lh))
    }

    /// The curves f and A at the ends of the cells, for a start `a`, as drawn: on their
    /// way from the old ones while the formula changes.
    fn curves(&self, a: f64) -> (Vec<f64>, Vec<f64>) {
        let (area, _) = self.grid.curve(&self.f.expr, a);
        let m = self.morph.shown[0];
        let mix = |old: &[f64], new: &[f64]| -> Vec<f64> {
            if m >= 1.0 || old.len() != new.len() {
                return new.to_vec();
            }
            new.iter().zip(old).map(|(n, o)| if n.is_finite() && o.is_finite() { o + (n - o) * m } else { *n }).collect()
        };
        (mix(&self.old_f, &self.grid.fx), mix(&self.old_a, &area))
    }

    /// Whether a place is near one where f grows past any size (its values there say
    /// nothing about the range a chart should show).
    fn near_pole(&self, t: f64) -> bool {
        let w = 0.04 * (self.grid.hi - self.grid.lo);
        self.grid.poles.iter().any(|p| (p - t).abs() < w)
    }

    /// Move what follows and what plays: glides by the clock, films by the scene's time.
    fn advance(&mut self, dt: f64, real_dt: f64) {
        self.drawn.step(real_dt);
        self.morph.step(real_dt);
        let ready = self.drawn.arrived() && self.morph.arrived();
        let Some(mut film) = self.film else { return };
        if film.waiting {
            if !ready {
                return;
            }
            film.waiting = false;
            if film.which == Which::Start {
                // the lower chart shows the curve at both ends of the film throughout
                let e = &self.f.expr;
                let mut span = [f64::INFINITY, f64::NEG_INFINITY];
                for a in [film.from, film.to] {
                    for (t, v) in self.grid.xs.iter().zip(self.grid.curve(e, a).0) {
                        if v.is_finite() && !self.near_pole(*t) {
                            span = [span[0].min(v), span[1].max(v)];
                        }
                    }
                }
                self.goals[1] = keep_range([0.0, 0.0], span);
                self.held = true;
            }
        }
        film.t += dt;
        let done = film.t >= film.seconds;
        self.set(film.which, if done { film.to } else { film.at() });
        self.film = if done { None } else { Some(film) };
        if done {
            self.held = false;
        }
    }

    /// Where the charts' ranges are going, from what they show now: but while the
    /// picture glides to a new setting, they go where it is going, set when it was asked
    /// for (`aim_ahead`), and stay there (re-aimed at curves on their way, they would
    /// change their round values more than once); and while a handle is held they wait
    /// for it to be let go (what is drawn is taken in all the same: see `plots`).
    fn aim_ranges(&mut self, fs: &[f64], areas: &[f64]) {
        let [top, bottom] = self.spans(fs, areas, self.anchor);
        self.extent = [top, bottom];
        let gliding = !self.drawn.arrived() || !self.morph.arrived();
        if (gliding || self.dragging) && self.goals != [[0.0, 0.0], [0.0, 0.0]] {
            return;
        }
        self.aim(top, bottom);
    }

    /// Aim the ranges at what the setting asked for will show, once arrived.
    fn aim_ahead(&mut self) {
        self.ensure_grid();
        let (areas, _) = self.grid.curve(&self.f.expr, self.a);
        let look = self.set_look();
        let anchor = [look.area.ok(), look.area.and_then(|a| look.strip.map(|s| a + s)).ok()].map(|v| v.unwrap_or(f64::NAN));
        let fs = self.grid.fx.clone();
        let [top, bottom] = self.spans(&fs, &areas, anchor);
        self.aim(top, bottom);
    }

    fn aim(&mut self, top: [f64; 2], bottom: [f64; 2]) {
        self.goals[0] = keep_range(self.goals[0], top);
        let held = self.held && bottom[0] >= self.goals[1][0] && bottom[1] <= self.goals[1][1];
        if !held {
            self.goals[1] = keep_range(self.goals[1], bottom);
        }
        self.ranges.to = [self.goals[0][0], self.goals[0][1], self.goals[1][0], self.goals[1][1]];
    }

    /// From where to where the two charts' curves go (where f grows past any size
    /// aside): f; and A, with F where it is drawn and A at x and at x + h.
    fn spans(&self, fs: &[f64], areas: &[f64], anchor: [f64; 2]) -> [[f64; 2]; 2] {
        let xs = &self.grid.xs;
        let span = |values: &mut dyn Iterator<Item = f64>| values.fold([f64::INFINITY, f64::NEG_INFINITY], |s, v| [s[0].min(v), s[1].max(v)]);
        let top = span(&mut xs.iter().zip(fs).filter(|(t, v)| v.is_finite() && !self.near_pole(**t)).map(|(_, v)| *v));
        let anti = self.show_anti.then(|| self.known()).flatten().and_then(|i| PRESETS[i].anti).map(|(_, big_f)| big_f);
        let mut lower: Vec<f64> = xs.iter().zip(areas).filter(|(t, v)| v.is_finite() && !self.near_pole(**t)).map(|(_, v)| *v).collect();
        if let Some(big_f) = anti {
            lower.extend(xs.iter().zip(areas).filter(|(t, v)| v.is_finite() && !self.near_pole(**t)).map(|(t, _)| big_f(*t)).filter(|v| v.is_finite()));
        }
        lower.extend(anchor.iter().filter(|v| v.is_finite()));
        let bottom = span(&mut lower.into_iter());
        [top, bottom]
    }

    /// The two charts as they are drawn now.
    /// (Each takes in all it draws, even while its range glides more slowly than the
    /// curves change: a curve that crossed the edge would be cut there, and its points
    /// renumbered.)
    fn plots(&self, lay: &Layout) -> (Plot, Plot) {
        let r = self.ranges.shown;
        let x = [self.grid.lo, self.grid.hi];
        // (a tenth of what is drawn clear of the edges: see `inside`)
        let take_in = |lo: f64, hi: f64, e: [f64; 2]| {
            let pad = 0.1 * (e[1] - e[0]);
            if e[0] <= e[1] {
                [lo.min(e[0] - pad), hi.max(e[1] + pad)]
            } else {
                [lo, hi]
            }
        };
        (Plot::new(lay.top, x, take_in(r[0], r[1], self.extent[0])), Plot::new(lay.bottom, x, take_in(r[2], r[3], self.extent[1])))
    }

    /// The handles: the ends on the upper chart, the point on A, the knob for h.
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        let (top, bottom) = self.plots(lay);
        let [a, x, lh] = self.drawn.shown;
        if top.tiny() {
            return;
        }
        // (the end first: when the two are on top of each other, it is the one taken)
        if let Some((t, _)) = top.handle(f, 2, x, 0.0, END) {
            let t = self.snapped(&top, t);
            self.moved(Which::End, t);
        }
        if let Some((t, _)) = top.handle(f, 1, a, 0.0, START) {
            self.moved(Which::Start, t);
        }
        let ax = Some(self.anchor[0]).filter(|v| v.is_finite());
        if let (false, Some(ax)) = (bottom.tiny(), ax) {
            let y = ax.max(bottom.y[0]).min(bottom.y[1]);
            if let Some((t, _)) = bottom.handle(f, 4, x, y, END) {
                let t = self.snapped(&bottom, t);
                self.moved(Which::End, t);
            }
        }
        if !empty(lay.knob) {
            let knob = Plot::new(lay.knob, [H_MIN.log10(), H_MAX.log10()], [0.0, 1.0]);
            if let Some((v, _)) = knob.handle(f, 3, lh, 0.5, STRIP) {
                self.moved(Which::LogWidth, v);
            }
        }
    }

    /// A setting moved by the hand: it stops any film.
    fn moved(&mut self, which: Which, v: f64) {
        if v.is_finite() {
            self.film = None;
            self.held = false;
            self.set(which, v);
            self.dragging = true;
        }
    }

    /// The end, dragged to `t`: onto a jump of f if it is within a few pixels of one,
    /// so that the corner of A there can be reached by hand.
    fn snapped(&self, p: &Plot, t: f64) -> f64 {
        let px = p.px_x(t);
        self.grid.jumps.iter().filter(|j| j.real(self.grid.scale)).map(|j| j.at).find(|&at| (p.px_x(at) - px).abs() < 6.0).unwrap_or(t)
    }
}

impl Scene for Ftc {
    fn view(&self) -> View {
        View::Flat { center: Vec2::ZERO, half_height: 1.0 }
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
        s.bloom = s.bloom.min(0.2);
    }

    fn animated(&self) -> bool {
        self.film.is_some() || self.dragging || !self.drawn.arrived() || !self.morph.arrived() || !self.ranges.arrived() || !self.slopes.arrived()
    }

    fn pointer_hint(&self) -> Option<&'static str> {
        Some("white and yellow dots: drag the start $a$ and the end $x$    orange dot: drag the strip's width $h$")
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        self.ensure_grid();
        ui::note(
            ui,
            r"Take the area between a curve $f$ and the axis, from a fixed start $a$ to a moving end $x$, and call it $A(x)$ (area below the axis counts negative). Move $x$ on by a small step $h$ and $A$ gains a thin strip, nearly a rectangle $f(x)$ tall and $h$ wide. So $A$ grows at the rate $f(x)$: its slope is the curve's height, wherever $f$ is continuous at $x$. That is the fundamental theorem of calculus. It is why areas are found by undoing derivatives: on a stretch where $f$ is continuous, any $F$ whose slope is $f$ differs from $A$ by a constant, so the area is $F(x) - F(a)$.",
        );
        ui::equation(ui, r"A(x) = \int_a^x f(t)\,\mathrm{d}t, \qquad A'(x) = f(x), \qquad \int_a^x f(t)\,\mathrm{d}t = F(x) - F(a)");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.button, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "The curve");
        if self.f.ui(ui, "f(x) =") {
            self.formula_changed();
        }
        let names: Vec<&str> = PRESETS.iter().map(|c| c.button).collect();
        if let Some(i) = ui::buttons(ui, &names) {
            self.choose(i);
        }
        ui::note(ui, "Any formula in $x$. For the curves on the buttons but the bell, the scene also knows a formula $F$ whose slope is $f$ (for the step, everywhere but at 0); the bell has none made of the usual functions.");

        ui::section(ui, "The ends and the strip");
        let (lo, hi) = (-self.half as f32, self.half as f32);
        let mut a = self.a as f32;
        if ui::slider(ui, "start $a$", &mut a, lo..=hi) {
            self.moved(Which::Start, a as f64);
        }
        let mut x = self.x as f32;
        if ui::slider(ui, "end $x$", &mut x, lo..=hi) {
            self.moved(Which::End, x as f64);
        }
        let mut h = self.h as f32;
        if ui::slider_log(ui, "strip width $h$", &mut h, H_MIN as f32..=H_MAX as f32) {
            self.moved(Which::LogWidth, (h as f64).log10());
        }
        ui::note(ui, "$h$: the width of the orange strip from $x$ to $x + h$, which is also the step of the secant over it.");
        let mut half = self.half as f32;
        if ui::slider(ui, r"view: $x$ from $-L$ to $L$, with $L =$", &mut half, HALF_MIN as f32..=HALF_MAX as f32) {
            self.half = (half as f64).max(HALF_MIN).min(HALF_MAX);
            self.film = None;
            self.held = false;
            for (which, v) in [(Which::Start, self.a), (Which::End, self.x)] {
                self.set(which, v);
            }
            self.ensure_grid();
        }
        if self.known().is_some() {
            ui::checkbox(ui, &mut self.show_anti, r"draw $F$, an antiderivative");
        }

        ui::section(ui, "Reading the picture");
        let mut key = vec![
            (CURVE, "the curve $f$"),
            (PLUS, r"area that adds to $A$: above the axis when $x$ is right of $a$"),
            (MINUS, "area that takes from $A$: below the axis when $x$ is right of $a$"),
            (START, "$a$, the start (drag it)"),
            (END, "$x$, the moving end (drag it), and the tangent to $A$ there"),
            (STRIP, r"the strip from $x$ to $x + h$, and the secant over it below"),
            (AREA, "$A(x)$, the area from $a$ to $x$, for every $x$"),
        ];
        if self.show_anti && self.known().is_some() {
            key.push((ANTI, "dashed: an antiderivative $F$"));
        }
        if !self.grid.faults().is_empty() {
            key.push((CUT, "where $f$ has no value or grows past any size: no area across"));
        }
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, "Integral:", r"$\int_a^x f(t)\,\mathrm{d}t$, the area between the curve and the axis from $a$ to $x$, counted negative below the axis, and with the opposite sign when $x$ is left of $a$.");
        ui::term(ui, "Continuous at $x$:", "no jump or gap there: $f(t)$ comes as close to $f(x)$ as you like once $t$ is close enough to $x$.");
        ui::term(ui, "Antiderivative:", "a function $F$ whose slope is $f$: $F'(x) = f(x)$. Two of them differ by a constant.");
        ui::term(ui, "Secant:", "a straight line through two points of a curve: here of $A$, at $x$ and at $x + h$.");
        ui::term(ui, "Tangent:", "the line that grazes a curve at one point, with the curve's slope there.");

        ui::section(ui, "Numbers");
        let look = self.set_look();
        let d = decimals(look.err);
        let value = |r: Result<f64, Fault>| match r {
            Ok(v) => format!("${}$", number(v, d)),
            Err(_) => "none".to_string(),
        };
        ui::readout_colored(ui, r"$A(x)$, by a sum", value(look.area), AREA);
        if let Some(v) = look.anti {
            ui::readout_colored(ui, r"$F(x) - F(a)$", format!("${}$", number(v, d)), ANTI);
        }
        ui::readout_colored(ui, "$f(x)$", if look.fx.is_finite() { format!("${}$", number(look.fx, 4)) } else { "none".into() }, CURVE);
        let slope = match look.slope {
            Some((l, r)) if l == r => format!("${}$", number(l, 4)),
            Some((l, r)) => format!("${}$ left, ${}$ right", number(l, 4), number(r, 4)),
            None => "none".into(),
        };
        ui::readout_colored(ui, "slope of $A$ at $x$", slope, END);
        ui::readout_colored(ui, "strip's area", value(look.strip), STRIP);
        ui::readout_colored(ui, "secant's slope", look.secant().map_or("none".into(), |m| format!("${}$", number(m, d))), STRIP);
        for line in self.notes(&look) {
            ui::note(ui, &line);
        }
        ui::see_also(
            ui,
            &[
                ("calc1", "slopes, and areas as sums of rectangles"),
                ("distributions", "a density's area so far: its cumulative curve"),
                ("changevar", "changing variables in an integral"),
                ("surface", "slopes of a surface"),
            ],
        );
    }

    fn frame(&mut self, f: &mut Frame) {
        hold_batches(f);
        self.ensure_grid();
        let lay = Layout::of(f.view_size(), f.top_inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a phone)
        let squeezed = lay.view.x < 160.0 || lay.view.y < 160.0;
        self.dragging = false;
        if !squeezed {
            self.handles(f, &lay);
        }
        self.advance(f.dt as f64, f.real_dt as f64);
        let look = self.drawn_look();
        let (fs, areas) = self.curves(look.a);
        self.anchor = self.anchor_of(&look, &areas);
        self.aim_ranges(&fs, &areas);
        let first = self.canvas.is_none();
        if first {
            self.ranges = Follow::at(self.ranges.to);
        } else {
            self.ranges.step(f.real_dt as f64);
        }
        if !squeezed {
            self.canvas = Some(lay.view);
        }
        self.last_f.clone_from(&fs);
        self.last_a.clone_from(&areas);
        if squeezed {
            return;
        }
        let (top, bottom) = self.plots(&lay);
        self.ease_slopes(&look, &bottom, if first { f64::INFINITY } else { f.real_dt as f64 });
        self.draw_top(f, &lay, &top, &fs, &look);
        self.draw_bottom(f, &lay, &bottom, &areas, &look);
        if !lay.bare {
            self.draw_numbers(f, &lay, &look);
            self.draw_knob(f, &lay);
            self.draw_zoom(f, &lay, &look);
            self.draw_note(f, &lay);
        }
    }
}

/// A point of the plane under a place on the screen (for `kit::wrapped`, which takes
/// the plane's own units).
fn world(f: &Frame, px: Vec2) -> Option<Vec3> {
    px.is_finite().then(|| f.cam.pick_ground(px * f.px)).flatten()
}

/// A label beside the point `at` (on the screen), `gap` pixels off it on the side
/// `towards`, if that side is clear; nothing otherwise. (Never on another side: a
/// label that moved round its point when crowded would leap across the picture, and
/// what each label says is also written beside the charts.)
#[allow(clippy::too_many_arguments)]
fn put(f: &mut Frame, room: &mut Room, at: Vec2, text: &str, size: f32, color: Color, towards: Vec2, gap: f32) -> bool {
    let half = f.text_size(text, size) * 0.5 + Vec2::new(4.0, 1.0);
    let d = towards.normalize_or_zero();
    let centre = at + d * gap + Vec2::new(d.x * half.x, d.y * half.y);
    if !(centre.is_finite() && room.room_for(at, centre, half) >= 3.0) {
        return false;
    }
    room.boxed(centre - half, centre + half);
    f.text_px(centre, text, size, color, Align2::CENTER_CENTER);
    true
}

/// Room for labels inside a chart only.
fn room_in(f: &Frame, (min, max): Rect) -> Room {
    let mut room = Room::new(f);
    let v = f.view_size().max(max + Vec2::splat(1.0));
    room.wall(Vec2::ZERO, Vec2::new(v.x, min.y));
    room.wall(Vec2::new(0.0, max.y), v);
    room.wall(Vec2::ZERO, Vec2::new(min.x, v.y));
    room.wall(Vec2::new(max.x, 0.0), v);
    room
}

/// How far across (in the chart's units, at most `most`) a line of slope `m` from
/// `from` can go in the direction `dir` (1 right, −1 left) and stay inside the chart,
/// 14 pixels clear of its top and bottom (room for a label beyond its end) and 4 of
/// its sides. (A line from within that band would be cut to nothing heading out and
/// left whole heading in, and leap as it turned: but what is drawn is always further
/// inside than that, see `plots`.)
fn inside(p: &Plot, from: (f64, f64), m: f64, dir: f64, most: f64) -> f64 {
    let per_px = (p.y[1] - p.y[0]) / p.height().max(1.0) as f64;
    let (lo, hi) = (p.y[0] + 14.0 * per_px, p.y[1] - 14.0 * per_px);
    let side = 4.0 * (p.x[1] - p.x[0]) / p.width().max(1.0) as f64;
    let mut s = most.max(0.0);
    if dir > 0.0 {
        s = s.min(p.x[1] - side - from.0);
    } else {
        s = s.min(from.0 - p.x[0] - side);
    }
    let rise = m * dir;
    if rise > 0.0 {
        s = s.min((hi - from.1) / rise);
    } else if rise < 0.0 {
        s = s.min((lo - from.1) / rise);
    }
    if s.is_finite() {
        s.max(0.0)
    } else {
        0.0
    }
}

/// A step in from an end of a stretch at `t`, too small to see, so that f is read just
/// inside it.
fn inward(t: f64) -> f64 {
    1e3 * finest(t)
}

/// Where the `i`th of the `n + 1` points across a strip, at `t`, reads f: just inside
/// the strip at its two ends.
fn inside_strip(t: f64, i: usize, n: usize) -> f64 {
    if i == 0 {
        t + inward(t)
    } else if i == n {
        t - inward(t)
    } else {
        t
    }
}

/// A small ring in screen space (the open end of a jump), in the batch of its kind.
fn ring(f: &mut Frame, kind: u8, at: Vec2, radius: f32, color: Color) {
    let pts: Vec<Vec2> = (0..=16).map(|i| at + Vec2::from_angle(i as f32 * TAU / 16.0) * radius).collect();
    plot::path_of(f, kind, &pts, 1.4, color);
}

/// Text broken into lines no wider than `width`, from `at` down, in logical pixels of
/// the screen (`kit::wrapped` takes the plane's own units). Returns the height taken.
fn wrapped_px(f: &mut Frame, at: Vec2, width: f32, text: &str, size: f32, color: Color) -> f32 {
    let ppu = pixels_per_unit(f).max(1e-6);
    let Some(w) = world(f, at) else { return 0.0 };
    let width = width / ppu;
    wrapped(f, w.truncate(), width, text, size, color);
    wrapped_height(f, width, text, size)
}

impl Ftc {
    /// f at t as drawn: on its way from the old curve while the formula changes.
    fn curve_at(&self, t: f64) -> f64 {
        let now = value(&self.f.expr, t);
        let m = self.morph.shown[0];
        if m >= 1.0 || self.old_f.len() != CELLS + 1 {
            return now;
        }
        let g = &self.grid;
        let s = ((t - g.lo) / g.width()).max(0.0).min(CELLS as f64 - 1e-9);
        let k = (s.floor() as usize).min(CELLS - 1);
        let old = self.old_f[k] + (self.old_f[k + 1] - self.old_f[k]) * (s - k as f64);
        if old.is_finite() && now.is_finite() {
            old + (now - old) * m
        } else {
            now
        }
    }

    /// A at x and at x + h as drawn: where the formula is changing, on the drawn curve
    /// (between its points), and otherwise A's own values.
    fn anchor_of(&self, look: &Look, areas: &[f64]) -> [f64; 2] {
        let own = [look.area.ok(), look.area.and_then(|a| look.strip.map(|s| a + s)).ok()];
        if self.morph.shown[0] >= 1.0 {
            return own.map(|v| v.unwrap_or(f64::NAN));
        }
        let g = &self.grid;
        let at = |t: f64| {
            let s = (t - g.lo) / g.width();
            if !(s >= 0.0 && s <= CELLS as f64) {
                return f64::NAN;
            }
            let k = (s.floor() as usize).min(CELLS - 1);
            areas[k] + (areas[k + 1] - areas[k]) * (s - k as f64)
        };
        [own[0].map_or(f64::NAN, |_| at(look.x)), own[1].map_or(f64::NAN, |_| at(look.x + look.h))]
    }

    /// The tangent's drawn slopes brought towards A's own: at once where its far ends
    /// would move a few pixels, gliding where they would leap.
    fn ease_slopes(&mut self, look: &Look, q: &Plot, dt: f64) {
        let Some((l, r)) = look.slope else { return };
        let span = 0.15 * (self.grid.hi - self.grid.lo);
        let px = q.height() as f64 / (q.y[1] - q.y[0]).max(1e-300);
        let [sl, sr] = self.slopes.shown;
        let leap = ((l - sl).abs().max((r - sr).abs()) * span * px).min(f64::MAX);
        // (and at once in the first frame, where there is nothing to glide from; a glide
        // once begun is seen through to the end)
        let small = leap <= 10.0 || leap.is_nan();
        if (self.slopes.arrived() && small) || !(sl.is_finite() && sr.is_finite()) || dt.is_infinite() {
            self.slopes = Follow::at([l, r]);
        } else {
            self.slopes.to = [l, r];
            self.slopes.step(dt);
        }
    }

    /// What the panel says beyond the numbers: why there is no area, a corner, an end
    /// left of the start.
    fn notes(&self, look: &Look) -> Vec<String> {
        let mut out = Vec::new();
        match look.area {
            Err(Fault::Unbounded(t)) => out.push(format!(
                r"No area from $a$ to $x$: $f$ grows past any size near $t = {}$, and sums of thin strips across such a point settle on no number. (For some such curves, $1/\sqrt{{|x|}}$ among them, the area does have a limit, an improper integral; for $1/x$ it has none.)",
                place(t)
            )),
            Err(Fault::NoValue(lo, hi)) => out.push(format!(r"No area from $a$ to $x$: $f$ has no value between $t = {}$ and $t = {}$.", place(lo), place(hi))),
            Ok(_) => {}
        }
        if look.area.is_ok() && look.corner() {
            if let Some((l, r)) = look.slope {
                out.push(format!(
                    r"$f$ jumps at $x$, from ${}$ to ${}$: $A$ has a corner there and no tangent. Its slope from the left is ${}$ and from the right ${}$; the secant, on the right, settles on ${}$ as $h$ shrinks.",
                    number(l, 2),
                    number(r, 2),
                    number(l, 2),
                    number(r, 2),
                    number(r, 2)
                ));
            }
        } else if let (Ok(_), Some((s, _))) = (look.area, look.slope) {
            if look.fx.is_finite() && (s - look.fx).abs() > 1e-9 * (1.0 + s.abs()) {
                out.push(format!(r"$f(x) = {}$ is off the curve's line: on both sides $f$ comes to ${}$, and that is $A$'s slope at $x$.", number(look.fx, 4), number(s, 4)));
            }
        }
        if look.x < look.a {
            out.push(r"$x$ is left of $a$: going from $a$ to $x$ runs backwards, so each strip counts with the opposite sign, and area above the axis takes from $A$.".to_string());
        }
        if self.grid.rough {
            out.push("Some sums did not settle in the time allowed: their digits are written only as far as the estimate of their error allows.".to_string());
        }
        out
    }

    /// The area from a to x on the upper chart, and the strip.
    fn draw_top(&self, f: &mut Frame, lay: &Layout, p: &Plot, fs: &[f64], look: &Look) {
        if p.tiny() {
            return;
        }
        let (min, max) = p.rect;
        p.backing(f, if lay.bare { 30.0 } else { 46.0 }, 24.0, lay.title + 4.0, 12.0);
        if !lay.bare {
            let title = if lay.narrow { r"the curve $f$, and the area from $a$ to $x$" } else { r"the curve $f$, and the area under it from $a$ to $x$: blue adds to $A$, pink takes from it" };
            f.text_px(Vec2::new(min.x - 38.0, min.y - 10.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        }
        axes(f, p, self.goals[0], TOP_AXES, "", lay.small);
        p.hline(f, AXES, 0.0, 1.2, pal::DIM.a(0.9), false);
        let g = &self.grid;
        let (a, x) = (look.a, look.x);
        let zero = |p: &Plot, t: f64| Vec2::new(p.px_x(t), p.px_y(0.0).max(min.y).min(max.y));
        let at = |p: &Plot, t: f64, y: f64| Vec2::new(p.px_x(t), p.px_y(y).max(min.y).min(max.y));
        // the area from a to x, if there is one
        let dir = if x >= a { 1.0 } else { -1.0 };
        let (lo, hi) = (a.min(x).max(g.lo), a.max(x).min(g.hi));
        if hi > lo && look.area.is_ok() {
            // (at the ends, f just inside: its value at the very end point, off its
            // curve at a jump, has no part in the area)
            let mut points: Vec<(f64, f64)> = vec![(lo, self.curve_at(lo + inward(lo)))];
            points.extend(g.xs.iter().zip(fs).filter(|(t, _)| **t > lo && **t < hi).map(|(t, y)| (*t, *y)));
            points.push((hi, self.curve_at(hi - inward(hi))));
            for pair in points.windows(2) {
                let ((t0, y0), (t1, y1)) = (pair[0], pair[1]);
                if !(y0.is_finite() && y1.is_finite()) {
                    continue;
                }
                let mut column = |t0: f64, y0: f64, t1: f64, y1: f64| {
                    let color = if (y0 + y1) * dir >= 0.0 { PLUS } else { MINUS };
                    shade(f, [zero(p, t0), at(p, t0, y0), at(p, t1, y1), zero(p, t1)], color.a(0.34));
                };
                if y0 * y1 < 0.0 {
                    let tc = t0 + (t1 - t0) * y0 / (y0 - y1);
                    column(t0, y0, tc, 0.0);
                    column(tc, 0.0, t1, y1);
                } else {
                    column(t0, y0, t1, y1);
                }
            }
        }
        // the strip from x to x + h, with the rectangle f(x) tall drawn over it
        let (s0, s1) = (x.max(g.lo), (x + look.h).min(g.hi));
        if s1 > s0 && look.strip.is_ok() {
            let n = 24;
            let ts: Vec<f64> = (0..=n).map(|i| s0 + (s1 - s0) * i as f64 / n as f64).collect();
            let ys: Vec<f64> = ts.iter().enumerate().map(|(i, &t)| self.curve_at(inside_strip(t, i, n))).collect();
            for i in 0..n {
                if ys[i].is_finite() && ys[i + 1].is_finite() {
                    shade(f, [zero(p, ts[i]), at(p, ts[i], ys[i]), at(p, ts[i + 1], ys[i + 1]), zero(p, ts[i + 1])], STRIP.a(0.6));
                }
            }
            let top: Vec<Vec2> = ts.iter().zip(&ys).map(|(t, y)| if y.is_finite() { at(p, *t, *y) } else { Vec2::NAN }).collect();
            plot::path_of(f, STRIP_LINE, &top, 1.6, STRIP);
            for (t, y) in [(s0, ys[0]), (s1, ys[n])] {
                if y.is_finite() {
                    plot::line_of(f, STRIP_LINE, zero(p, t), at(p, t, y), 1.4, STRIP);
                }
            }
            // (where A has a tangent: at a jump of f the rectangle stands for nothing)
            if let (Some((l, r)), true) = (look.slope, p.px_x(s1) - p.px_x(s0) > 10.0) {
                if l == r {
                    plot::dashed_of(f, STRIP_LINE, at(p, s0, l), at(p, s1, l), 1.4, END);
                }
            }
        }
        // where f has no value, or grows past any size
        for fault in g.faults() {
            match fault {
                Fault::Unbounded(t) => p.vline(f, CUT_LINE, t, 1.4, CUT.a(0.8), true),
                Fault::NoValue(l, h) => {
                    let (l, h) = (l.max(g.lo), h.min(g.hi));
                    if h > l {
                        shade(f, [Vec2::new(p.px_x(l), min.y), Vec2::new(p.px_x(h), min.y), Vec2::new(p.px_x(h), max.y), Vec2::new(p.px_x(l), max.y)], CUT.a(0.05));
                    }
                }
            }
        }
        // the curve, broken at its jumps and where it grows past any size
        let mut pts: Vec<(f64, f64)> = Vec::with_capacity(CELLS + 16);
        for k in 0..=CELLS {
            pts.push((g.xs[k], fs[k]));
            if k < CELLS {
                let jump = g.jumps.iter().any(|j| j.at > g.xs[k] - 1e-12 && j.at < g.xs[k + 1] + 1e-12 && j.real(g.scale));
                if jump || matches!(g.cells[k], Err(Fault::Unbounded(_))) {
                    pts.push((f64::NAN, f64::NAN));
                }
            }
        }
        p.path(f, F_LINE, pts, 2.6, CURVE);
        // a jump's two sides, open, and f's value at it, filled, where that is neither
        for j in g.jumps.iter().filter(|j| j.real(g.scale) && j.at >= g.lo && j.at <= g.hi) {
            for y in [j.left, j.right] {
                let c = p.at(j.at, y);
                if plot::within(c, p.rect, 0.5) {
                    ring(f, RINGS, c, 3.6, CURVE);
                }
            }
            let v = value(&self.f.expr, j.at);
            if v.is_finite() && (v - j.left).abs() > 1e-9 && (v - j.right).abs() > 1e-9 {
                p.dot(f, j.at, v, 7.0, CURVE);
            }
        }
        // the ends, marked up the chart and named at its top
        for (t, color) in [(a, START), (x, END)] {
            p.vline(f, ENDS, t, 1.0, color.a(0.45), true);
        }
        let (pa, px) = (p.px_x(a), p.px_x(x));
        let close = (pa - px).abs() < 22.0;
        for (t, name, color, left) in [(a, "$a$", START, a <= x), (x, "$x$", END, x < a)] {
            if !(t >= g.lo && t <= g.hi) {
                continue;
            }
            let align = match (close, left) {
                (false, _) => Align2::CENTER_TOP,
                (true, true) => Align2::RIGHT_TOP,
                (true, false) => Align2::LEFT_TOP,
            };
            let nudge = if !close {
                0.0
            } else if left {
                -4.0
            } else {
                4.0
            };
            f.text_px(Vec2::new(p.px_x(t) + nudge, min.y + 4.0), name, lay.text + 1.0, color, align);
        }
        // the curve's name, near its right end
        let mut room = room_in(f, p.rect);
        let busy: Vec<Vec2> = g.xs.iter().zip(fs).step_by(4).map(|(t, y)| p.at(*t, *y)).collect();
        for pair in busy.windows(2) {
            if pair[0].is_finite() && pair[1].is_finite() {
                room.segment_px(pair[0], pair[1]);
            }
        }
        let k = CELLS * 15 / 16;
        let end = p.at(g.xs[k], fs[k]);
        if plot::within(end, p.rect, -6.0) {
            put(f, &mut room, end, "$f$", lay.big, CURVE, Vec2::new(0.3, -1.0), 8.0);
        }
    }
}

impl Ftc {
    /// A on the lower chart, with its tangent and the secant over the strip.
    fn draw_bottom(&self, f: &mut Frame, lay: &Layout, q: &Plot, areas: &[f64], look: &Look) {
        if q.tiny() {
            return;
        }
        let (min, max) = q.rect;
        q.backing(f, if lay.bare { 30.0 } else { 46.0 }, if lay.bare { 22.0 } else { 40.0 }, lay.title + 4.0, 12.0);
        if !lay.bare {
            let title = if lay.narrow { r"$A(x) = \int_a^x f(t)\,\mathrm{d}t$, the area so far" } else { r"$A(x) = \int_a^x f(t)\,\mathrm{d}t$: the area from $a$ to $x$, for every $x$" };
            f.text_px(Vec2::new(min.x - 38.0, min.y - 10.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        }
        axes(f, q, self.goals[1], LOW_AXES, if lay.bare { "" } else { "$x$" }, lay.small);
        q.hline(f, AXES, 0.0, 1.2, pal::DIM.a(0.9), false);
        let g = &self.grid;
        let (a, x, h) = (look.a, look.x, look.h);
        for (t, color) in [(a, START), (x, END)] {
            q.vline(f, ENDS, t, 1.0, color.a(0.45), true);
        }
        for fault in g.faults() {
            match fault {
                Fault::Unbounded(t) => q.vline(f, CUT_LINE, t, 1.4, CUT.a(0.8), true),
                Fault::NoValue(l, hi) => {
                    let (l, hi) = (l.max(g.lo), hi.min(g.hi));
                    if hi > l {
                        shade(f, [Vec2::new(q.px_x(l), min.y), Vec2::new(q.px_x(hi), min.y), Vec2::new(q.px_x(hi), max.y), Vec2::new(q.px_x(l), max.y)], CUT.a(0.05));
                    }
                }
            }
        }
        let mut room = room_in(f, q.rect);
        // an antiderivative F, from its own formula, where A is drawn
        let anti = self.show_anti.then(|| self.known()).flatten().and_then(|i| PRESETS[i].anti);
        if let Some((_, big_f)) = anti {
            let pts: Vec<Vec2> = g.xs.iter().zip(areas).map(|(t, v)| if v.is_finite() { q.at(*t, big_f(*t)) } else { Vec2::NAN }).collect();
            let pts = plot::clip(&pts, q.rect);
            dashes(f, ANTI_LINE, &pts, 2.2, ANTI);
            for pair in pts.chunks(4).filter_map(|c| c.first()).collect::<Vec<_>>().windows(2) {
                if pair[0].is_finite() && pair[1].is_finite() {
                    room.segment_px(*pair[0], *pair[1]);
                }
            }
        }
        // A, through its values at the ends of the cells (it passes through (a, 0)
        // between two of them: a point of its own there would renumber all the points
        // after it whenever a passes one)
        let pts: Vec<(f64, f64)> = g.xs.iter().copied().zip(areas.iter().copied()).collect();
        let busy: Vec<Vec2> = pts.iter().step_by(4).map(|(t, v)| q.at(*t, *v)).collect();
        q.path(f, A_LINE, pts, 2.6, AREA);
        for pair in busy.windows(2) {
            if pair[0].is_finite() && pair[1].is_finite() {
                room.segment_px(pair[0], pair[1]);
            }
        }
        q.dot(f, a, 0.0, 8.0, START);
        let ax = self.anchor[0];
        if !ax.is_finite() {
            self.cut_label(f, lay, q, look, &mut room);
            return;
        }
        let span = 0.15 * (g.hi - g.lo);
        let p0 = q.at(x, ax);
        // The tangent reaches mostly left of x and the secant mostly right of it, each
        // stopping short of the chart's edges, so that each has a free end to be named
        // beyond: near x the curve and the two lines run close together.
        let reach = |from: (f64, f64), m: f64, dir: f64, most: f64| inside(q, from, m, dir, most);
        // (where the label goes from, and which way: beyond the tangent's far end, or
        // from a corner away from both its sides)
        let mut tangent_end: Option<(Vec2, Vec2)> = None;
        // (how much of a corner the drawn tangent is, 0 to 1, by how far apart its two
        // sides are on the screen: the right side grows from a stub to a full side as
        // they part, and the secant's stub behind x shrinks away)
        let px = q.height() as f64 / (q.y[1] - q.y[0]).max(1e-300);
        let [l, r] = self.slopes.shown;
        let corner = ((l - r).abs() * span * px / 150.0).min(1.0);
        let corner = if corner.is_finite() { corner } else { 0.0 };
        if let Some((true_l, true_r)) = look.slope {
            let (sl, sr) = (reach((x, ax), l, -1.0, span), reach((x, ax), r, 1.0, span * (0.3 + 0.7 * corner)));
            let pts = [(x - sl, ax - l * sl), (x, ax), (x + sr, ax + r * sr)];
            let screen: Vec<Vec2> = pts.iter().map(|(t, v)| q.at(*t, *v)).collect();
            q.path(f, TANGENT, pts, 2.4, END);
            room.segment_px(screen[0], screen[1]);
            room.segment_px(screen[1], screen[2]);
            tangent_end = if true_l == true_r {
                Some((screen[0], Vec2::new(-1.0, 0.0)))
            } else {
                let away = -((screen[0] - screen[1]).normalize_or_zero() + (screen[2] - screen[1]).normalize_or_zero());
                Some((screen[1], if away.length() > 1e-3 { away } else { Vec2::new(0.0, 1.0) }))
            };
        }
        // the secant over the strip, and its rise: the strip's area
        let mut secant_end = None;
        if let Some(strip) = Some(self.anchor[1] - ax).filter(|v| v.is_finite()) {
            let m = strip / h;
            // (from a corner, only on: back from it, it would run under the corner)
            let back = 0.3 * span * (1.0 - corner);
            let (sl, sr) = (reach((x, ax), m, -1.0, back), reach((x + h, ax + strip), m, 1.0, 0.7 * span));
            let ends = [(x - sl, ax - m * sl), (x + h + sr, ax + m * (h + sr))];
            q.path(f, SECANT, ends, 1.8, STRIP);
            let screen = [q.at(ends[0].0, ends[0].1), q.at(ends[1].0, ends[1].1)];
            room.segment_px(screen[0], screen[1]);
            secant_end = Some(screen[1]);
            let p1 = q.at(x + h, ax + strip);
            let run = q.at(x + h, ax);
            plot::dashed_of(f, GAP, p0, run, 1.2, pal::DIM);
            if plot::within(run, q.rect, 0.5) && plot::within(p1, q.rect, 0.5) {
                plot::line_of(f, STRIP_LINE, run, p1, 2.2, STRIP);
                plot::disc(f, p1, 6.0, STRIP);
            }
        }
        // F's height above A: at a, where A is 0, it is F(a); and the same at every x
        if let (Some((_, big_f)), true) = (anti, !lay.bare) {
            let fa = big_f(a);
            let left = q.px_x(a) - 10.0 >= min.x + 4.0;
            let px = q.px_x(a) + if left { -9.0 } else { 9.0 };
            let (y0, y1) = (q.px_y(0.0), q.px_y(fa));
            if fa.is_finite() && y0.is_finite() && y1.is_finite() {
                let (y0, y1) = (y0.max(min.y).min(max.y), y1.max(min.y).min(max.y));
                let tick = if left { 5.0 } else { -5.0 };
                let bracket = [Vec2::new(px + tick, y0), Vec2::new(px, y0), Vec2::new(px, y1), Vec2::new(px + tick, y1)];
                plot::path_of(f, GAP, &bracket, 1.6, ANTI);
                for pair in bracket.windows(2) {
                    room.segment_px(pair[0], pair[1]);
                }
                // (the longer name where there is room for it)
                let texts = [format!(r"$F - A = F(a) = {}$", number(fa, 3)), format!(r"$F(a) = {}$", number(fa, 3))];
                let (at, towards) = (Vec2::new(px, 0.5 * (y0 + y1)), Vec2::new(if left { -1.0 } else { 1.0 }, 0.0));
                let color = ANTI.mix(pal::WHITE, 0.2);
                let _ = (lay.narrow || !put(f, &mut room, at, &texts[0], lay.small, color, towards, 6.0)) && put(f, &mut room, at, &texts[1], lay.small, color, towards, 6.0);
            }
        }
        // the names of the tangent, the secant and the curve
        let size = lay.small;
        if let Some((l, r)) = look.slope {
            let text = if l == r { format!("tangent: slope ${}$", number(l, 3)) } else { format!(r"slopes ${}$ and ${}$: no tangent", number(l, 2), number(r, 2)) };
            if let Some((at, towards)) = tangent_end {
                put(f, &mut room, at, &text, size, END, towards, 8.0);
            }
        }
        if let Some(m) = look.secant() {
            let text = format!("secant: slope ${}$", number(m, 3));
            if let Some(at) = secant_end {
                put(f, &mut room, at, &text, size, STRIP, Vec2::new(1.0, 0.0), 6.0);
            }
        }
        let k = CELLS * 15 / 16;
        let end = q.at(g.xs[k], areas[k]);
        if plot::within(end, q.rect, -6.0) {
            put(f, &mut room, end, "$A$", lay.big, AREA, Vec2::new(0.3, -1.0), 8.0);
        }
        self.cut_label(f, lay, q, look, &mut room);
    }

    /// Where A stops, said where it stops.
    fn cut_label(&self, f: &mut Frame, lay: &Layout, q: &Plot, look: &Look, room: &mut Room) {
        let Err(fault) = look.area else { return };
        if lay.bare {
            return;
        }
        let t = fault.at().max(q.x[0]).min(q.x[1]);
        let why = match fault {
            Fault::Unbounded(_) => r"no area across: $f$ grows past any size",
            Fault::NoValue(..) => r"no area across: $f$ has no value",
        };
        let from = Vec2::new(q.px_x(t), q.rect.0.y + 14.0);
        put(f, room, from, why, lay.small, CUT.mix(pal::WHITE, 0.3), Vec2::new(if look.x > look.a { 1.0 } else { -1.0 }, 0.0), 6.0);
    }

    /// The numbers at x, written beside the charts (or under them, on a phone).
    fn draw_numbers(&self, f: &mut Frame, lay: &Layout, look: &Look) {
        let (min, max) = lay.numbers;
        if empty(lay.numbers) {
            return;
        }
        let d = decimals(look.err);
        let area = match look.area {
            Ok(v) => format!("$A(x) = {}$", number(v, d)),
            Err(_) => r"$A(x)$: no area".to_string(),
        };
        let fx = if look.fx.is_finite() { format!("$f(x) = {}$", number(look.fx, 4)) } else { r"$f(x)$: no value".to_string() };
        let slope = match look.slope {
            Some((l, r)) if l == r => format!("tangent to $A$: slope ${}$", number(l, 4)),
            Some((l, r)) => format!(r"slopes of $A$: ${}$ left, ${}$ right", number(l, 2), number(r, 2)),
            None => "no tangent to $A$ at $x$".to_string(),
        };
        let secant = match (look.secant(), look.area) {
            (Some(m), _) => format!(r"$(A(x + h) - A(x))/h = {}$", number(m, d)),
            (None, Err(_)) => r"secant: none, $A$ has no value at $x$".to_string(),
            (None, Ok(_)) => "secant: none, no area in the strip".to_string(),
        };
        let anti = match (self.known(), look.anti) {
            (Some(i), Some(v)) => {
                // (|x| has no slope at 0; it is the area all the same, having no jump there)
                let but = if i == STEP { " but at 0" } else { "" };
                (format!(r"$F(x) - F(a) = {}$", number(v, d)), PRESETS[i].anti.map_or(String::new(), |(tex, _)| format!(r"with $F(x) = {tex}$, whose slope is $f${but}")))
            }
            (Some(_), None) => (r"$F(x) - F(a)$: no area to compare".to_string(), String::new()),
            (None, _) if preset_of(&self.f.text) == Some(BELL) => (r"$F$: no formula".to_string(), "none made of the usual functions has slope $f$".to_string()),
            (None, _) => (r"$F$: none known here".to_string(), "the scene knows $F$ for the curves on the buttons".to_string()),
        };
        if lay.narrow {
            let row = lay.text + 6.0;
            let mid = min.x + 0.5 * (max.x - min.x) + 4.0;
            let put = |f: &mut Frame, x: f32, k: f32, text: &str, color: Color| f.text_px(Vec2::new(x, min.y + k * row), text, lay.text, color, Align2::LEFT_TOP);
            put(f, min.x, 0.0, &area, AREA);
            put(f, mid, 0.0, &anti.0, ANTI);
            put(f, min.x, 1.0, &fx, CURVE);
            let slope = match look.slope {
                Some((l, r)) if l == r => format!("tangent slope ${}$", number(l, 3)),
                Some((l, r)) => format!(r"slopes ${}$, ${}$", number(l, 2), number(r, 2)),
                None => "no tangent".to_string(),
            };
            put(f, mid, 1.0, &slope, END);
            let secant = look.secant().map_or_else(|| "secant: none".to_string(), |m| format!("secant slope ${}$", number(m, 3)));
            put(f, min.x, 2.0, &secant, STRIP);
            put(f, mid, 2.0, &format!("$h = {}$", number(look.h, 3)), pal::DIM);
            return;
        }
        let mut y = min.y;
        let head = format!(r"at $x = {}$, from $a = {}$, with $h = {}$", place(look.x), place(look.a), number(look.h, 3));
        f.text_px(Vec2::new(min.x, y), head, lay.text, pal::DIM, Align2::LEFT_TOP);
        y += lay.text + 14.0;
        let rows: [(String, Color, String); 5] = [
            (area, AREA, "the area from $a$ to $x$, as a sum of thin strips".to_string()),
            (anti.0, ANTI, anti.1),
            (fx, CURVE, "the curve's height at $x$".to_string()),
            (slope, END, String::new()),
            (secant, STRIP, "the secant's slope: the strip's area over its width".to_string()),
        ];
        for (text, color, gloss) in rows {
            f.text_px(Vec2::new(min.x, y), text, lay.big, color, Align2::LEFT_TOP);
            y += lay.big + 7.0;
            if !gloss.is_empty() {
                f.text_px(Vec2::new(min.x + 10.0, y - 3.0), gloss, lay.small, pal::DIM, Align2::LEFT_TOP);
                y += lay.small + 8.0;
            } else {
                y += 4.0;
            }
        }
    }

    /// The knob for h: a track from 0.001 to 2, on a logarithmic scale.
    fn draw_knob(&self, f: &mut Frame, lay: &Layout) {
        let r = lay.knob;
        if empty(r) {
            return;
        }
        let k = Plot::new(r, [H_MIN.log10(), H_MAX.log10()], [0.0, 1.0]);
        let mid = 0.5 * (r.0.y + r.1.y);
        let title = if lay.narrow { "$h$, the strip's width" } else { "$h$, the strip's width: drag the orange dot" };
        f.text_px(Vec2::new(r.0.x - (if lay.narrow { 20.0 } else { 16.0 }), r.0.y - 6.0), title, lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        plot::line_of(f, AXES, Vec2::new(r.0.x, mid), Vec2::new(r.1.x, mid), 2.0, pal::DIM.a(0.6));
        let lh = self.drawn.shown[2];
        plot::line_of(f, STRIP_LINE, Vec2::new(r.0.x, mid), Vec2::new(k.px_x(lh), mid), 3.0, STRIP.a(0.8));
        for (v, name) in [(-3.0, "0.001"), (-2.0, "0.01"), (-1.0, "0.1"), (0.0, "1")] {
            let x = k.px_x(v);
            plot::line_of(f, AXES, Vec2::new(x, mid - 5.0), Vec2::new(x, mid + 5.0), 1.0, pal::DIM.a(0.8));
            f.text_px(Vec2::new(x, mid + 8.0), name, lay.small, pal::DIM, Align2::CENTER_TOP);
        }
    }

    /// The strip stretched to the width of a box: its top flattens onto f(x) as h
    /// shrinks, and its mean height is the secant's slope.
    fn draw_zoom(&self, f: &mut Frame, lay: &Layout, look: &Look) {
        let r = lay.zoom;
        if empty(r) {
            return;
        }
        let (x, h) = (look.x, look.h);
        f.text_px(Vec2::new(r.0.x - 34.0, r.0.y - 12.0), "the strip, stretched to the width of the box", lay.text, pal::TEXT, Align2::LEFT_BOTTOM);
        hud_panel(f, r.0 - Vec2::new(34.0, 6.0), r.1 + Vec2::new(10.0, 2.0 * lay.small + 34.0));
        let Ok(strip) = look.strip else {
            let why = "no area in the strip: $f$ has no value or grows past any size in it";
            wrapped_px(f, r.0 + Vec2::new(0.0, 8.0), r.1.x - r.0.x, why, lay.small, CUT.mix(pal::WHITE, 0.3));
            return;
        };
        let n = 48;
        let ts: Vec<f64> = (0..=n).map(|i| x + h * i as f64 / n as f64).collect();
        let ys: Vec<f64> = ts.iter().enumerate().map(|(i, &t)| self.curve_at(inside_strip(t, i, n))).collect();
        let mean = strip / h;
        let fx = self.curve_at(x);
        let (mut lo, mut hi) = (0.0f64, 0.0f64);
        for v in ys.iter().chain([&mean, &fx]).filter(|v| v.is_finite()) {
            (lo, hi) = (lo.min(*v), hi.max(*v));
        }
        let pad = 0.15 * (hi - lo).max(1e-9);
        let z = Plot::new(r, [x, x + h], [lo - pad, hi + pad]);
        let (min, max) = z.rect;
        let at = |t: f64, y: f64| Vec2::new(z.px_x(t), z.px_y(y).max(min.y).min(max.y));
        for i in 0..n {
            if ys[i].is_finite() && ys[i + 1].is_finite() {
                shade(f, [at(ts[i], 0.0), at(ts[i], ys[i]), at(ts[i + 1], ys[i + 1]), at(ts[i + 1], 0.0)], STRIP.a(0.45));
            }
        }
        z.hline(f, AXES, 0.0, 1.0, pal::DIM.a(0.8), false);
        let top: Vec<Vec2> = ts.iter().zip(&ys).map(|(t, y)| if y.is_finite() { at(*t, *y) } else { Vec2::NAN }).collect();
        plot::path_of(f, ZOOM_LINE, &top, 2.2, CURVE);
        if fx.is_finite() {
            plot::dashed_of(f, ZOOM_LINE, at(x, fx), at(x + h, fx), 1.6, END);
        }
        plot::line_of(f, ZOOM_LINE, at(x, mean), at(x + h, mean), 1.6, STRIP);
        let size = lay.small;
        f.text_px(Vec2::new(min.x, max.y + 3.0), "$x$", size, pal::DIM, Align2::CENTER_TOP);
        f.text_px(Vec2::new(max.x, max.y + 3.0), "$x + h$", size, pal::DIM, Align2::CENTER_TOP);
        // what the two lines are, under the box
        let y = max.y + size + 12.0;
        let rows = [
            (END, true, if fx.is_finite() { format!("$f(x) = {}$, the height at $x$", number(fx, 3)) } else { r"$f(x)$: no value".to_string() }),
            (STRIP, false, format!("mean height ${}${}", number(mean, 3), if look.area.is_ok() { ": the secant's slope" } else { "" })),
        ];
        for (k, (color, dashed, text)) in rows.into_iter().enumerate() {
            let at = Vec2::new(min.x - 30.0, y + k as f32 * (size + 7.0));
            let (a, b) = (at + Vec2::new(0.0, 0.5 * size + 1.0), at + Vec2::new(18.0, 0.5 * size + 1.0));
            if dashed {
                plot::dashed_of(f, ZOOM_LINE, a, b, 1.6, color);
            } else {
                plot::line_of(f, ZOOM_LINE, a, b, 1.6, color);
            }
            f.text_px(at + Vec2::new(24.0, 0.0), text, size, color.mix(pal::WHITE, 0.15), Align2::LEFT_TOP);
        }
    }

    /// How to read the picture, for a visitor who has not opened the panel.
    fn draw_note(&self, f: &mut Frame, lay: &Layout) {
        let (min, max) = lay.note;
        if empty(lay.note) {
            return;
        }
        let note = r"How to read it. The upper chart shades the area under $f$ from $a$ to $x$; the lower one draws that area, $A(x)$, for every $x$. Moving $x$ on by $h$ adds the orange strip to $A$. Where $f$ has no jump at $x$, a thin strip is nearly a rectangle $f(x)$ tall, so $A$ climbs at the rate $f(x)$: the slope of its tangent at $x$ is the height of $f$ there.";
        wrapped_px(f, min, max.x - min.x, note, lay.small, pal::DIM);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expr(formula: &str) -> Expr {
        Expr::parse(formula, &["x"]).unwrap()
    }

    /// The area of a formula from p to q, as the scene takes it over a view of ±half.
    fn area_of(formula: &str, half: f64, p: f64, q: f64) -> Result<f64, Fault> {
        let e = expr(formula);
        Grid::build(&e, -half, half).area(&e, p, q).0
    }

    /// The scene set up for experiment `k`, its film played to the end.
    fn after(k: usize) -> Ftc {
        let mut s = Ftc::create();
        s.set_up(k);
        s.finish_film();
        s.morph = Follow::at([1.0]);
        s
    }

    #[test]
    fn the_gauss_rule_is_exact_for_polynomials_to_degree_fifteen() {
        let nodes = gauss();
        assert!((nodes.iter().map(|n| n.1).sum::<f64>() - 2.0).abs() < 1e-14);
        for k in 0..=15 {
            let exact = if k % 2 == 1 { 0.0 } else { 2.0 / (k as f64 + 1.0) };
            let sum: f64 = nodes.iter().map(|(x, w)| w * x.powi(k)).sum();
            assert!((sum - exact).abs() < 1e-13, "x^{k}: {sum} against {exact}");
        }
        // and not for 16 (it is an eight-point rule)
        let sum: f64 = nodes.iter().map(|(x, w)| w * x.powi(16)).sum();
        assert!((sum - 2.0 / 17.0).abs() > 1e-6);
    }

    /// The formula in double precision gives what the formula's own numbers give.
    #[test]
    fn the_wide_formula_agrees_with_the_narrow_one() {
        for formula in ["sin(x) + 0.3x", "exp(-x^2/2)", "sign(x)", "1/x", "sqrt(x)", "abs(x)^1.5 - floor(x)", "max(x, 0.5) * atan2(x, 1)", "x^3/4 - x", "relu(x) + sigmoid(x) + gauss(x)"] {
            let e = expr(formula);
            for t in [-2.3f32, -1.0, -0.25, 0.0, 0.4, 1.0, 2.7] {
                let (narrow, wide) = (e.eval(&[t]), value(&e, t as f64));
                assert!(narrow.is_nan() == wide.is_nan(), "{formula} at {t}: {narrow} against {wide}");
                if narrow.is_finite() {
                    assert!((narrow as f64 - wide).abs() <= 1e-5 * (1.0 + wide.abs()), "{formula} at {t}: {narrow} against {wide}");
                }
            }
        }
    }

    /// Every curve on the buttons with a closed form: the sum of thin strips and
    /// F(b) − F(a) agree to far more than the digits written, either way round, from
    /// inside the view and beyond it.
    #[test]
    fn the_sums_agree_with_the_closed_forms() {
        for c in PRESETS.iter() {
            let Some((_, big_f)) = c.anti else { continue };
            for (p, q) in [(-3.0, 0.6), (0.5, 2.5), (2.5, 0.5), (-3.9, 3.7), (0.013, 0.013 + 1e-3), (1.0, 4.9), (-0.7, -0.7)] {
                // (1/x across 0 has no area: below)
                if c.formula == "1/x" && p * q <= 0.0 {
                    continue;
                }
                let got = area_of(c.formula, 4.0, p, q).unwrap_or_else(|e| panic!("{}: {p} to {q}: {e:?}", c.formula));
                let want = big_f(q) - big_f(p);
                assert!((got - want).abs() <= 1e-9 * (1.0 + want.abs()), "{}: {p} to {q}: {got} against {want}", c.formula);
            }
        }
        // the bell has no formula, but its area is known: from -3 to 0.5 it is
        // sqrt(2 pi) (Phi(0.5) - Phi(-3))
        let bell = area_of("exp(-x^2/2)", 4.0, -3.0, 0.5).unwrap();
        assert!((bell - 1.729_855_66).abs() < 1e-8, "{bell}");
    }

    /// No area is given across a point where f grows past any size, nor where it has
    /// no value; bounded curves with jumps, corners and wild wiggles keep theirs.
    #[test]
    fn an_area_that_does_not_exist_says_so() {
        for formula in ["1/x", "1/x^2", "ln(abs(x))", "1/sqrt(abs(x))", "-1/x^3"] {
            match area_of(formula, 4.0, -2.0, 2.5) {
                Err(Fault::Unbounded(t)) => assert!(t.abs() < 1e-3, "{formula}: {t}"),
                other => panic!("{formula}: {other:?}"),
            }
        }
        // poles between the ends of the cells, and the far way round
        match area_of("tan(x)", 4.0, -1.0, 2.5) {
            Err(Fault::Unbounded(t)) => assert!((t - std::f64::consts::FRAC_PI_2).abs() < 1e-3, "{t}"),
            other => panic!("tan: {other:?}"),
        }
        match area_of("1/(x-1.2345)", 4.0, 3.0, -1.0) {
            Err(Fault::Unbounded(t)) => assert!((t - 1.2345).abs() < 1e-3, "{t}"),
            other => panic!("1/(x - 1.2345): {other:?}"),
        }
        match area_of("sqrt(x)", 4.0, -2.0, 2.5) {
            Err(Fault::NoValue(lo, hi)) => assert!(lo >= -2.0 && hi <= 0.02, "{lo} {hi}"),
            other => panic!("sqrt: {other:?}"),
        }
        assert!(matches!(area_of("sqrt(-1-x^2)", 4.0, -2.0, 2.5), Err(Fault::NoValue(..))));
        // on one side of the pole, it is there
        let v = area_of("1/x", 4.0, 0.5, 3.0).unwrap();
        assert!((v - 6f64.ln()).abs() < 1e-9, "{v}");
        for (formula, a, b, want) in [
            ("sign(x)", -2.0, 2.5, 0.5),
            ("sign(x - 0.3)", -2.0, 2.5, (2.5 - 0.3f32 as f64) - (0.3f32 as f64 + 2.0)),
            // (−2 on [−1.5, −1), −1, 0, 1, and 2 on [2, 2.25])
            ("floor(x)", -1.5, 2.25, -2.0 * 0.5 - 1.0 + 0.0 + 1.0 + 2.0 * 0.25),
            ("sqrt(abs(x))", -2.0, 2.5, (2.0f64.powf(1.5) + 2.5f64.powf(1.5)) * 2.0 / 3.0),
            ("1 - abs(x)^0.1", -2.0, 2.5, 4.5 - (2.0f64.powf(1.1) + 2.5f64.powf(1.1)) / 1.1),
            ("1/(x^2 + 1e-6)", -2.0, 2.5, 1e3 * ((2e3f64).atan() + (2.5e3f64).atan())),
        ] {
            let got = area_of(formula, 4.0, a, b).unwrap_or_else(|e| panic!("{formula}: {e:?}"));
            assert!((got - want).abs() < 1e-6 * (1.0 + want.abs()), "{formula}: {got} against {want}");
        }
        // bounded and wild: a value, close to the true one (by a fine sum away from 0
        // and the bound 0.01 near it)
        let wild = area_of("sin(1/x)", 4.0, -0.5, 1.0);
        assert!(wild.is_ok(), "sin(1/x): {wild:?}");
    }

    /// Jumps of f are found where they are, and only there.
    #[test]
    fn the_jumps_are_found() {
        let jumps = |formula: &str| Grid::build(&expr(formula), -4.0, 4.0).jumps.into_iter().filter(|j| j.real(1.0)).collect::<Vec<_>>();
        let sign = jumps("sign(x)");
        assert_eq!(sign.len(), 1, "{sign:?}");
        assert!(sign[0].at == 0.0 && sign[0].left == -1.0 && sign[0].right == 1.0, "{sign:?}");
        let floor = jumps("floor(x)");
        // (and at 4, the view's right end, where it steps up from 3)
        assert_eq!(floor.iter().map(|j| j.at).collect::<Vec<_>>(), vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]);
        // (at 0.3 as the formula reads it, in single precision: 0.30000001)
        let off = jumps("sign(x - 0.3)");
        assert!(off.len() == 1 && (off[0].at - 0.3f32 as f64).abs() < 1e-9, "{off:?}");
        for smooth in ["sin(x)", "x^3/4 - x", "exp(x)", "sqrt(abs(x))", "abs(x)", "1/(1 + 100x^2)", "tanh(50x)"] {
            assert!(jumps(smooth).is_empty(), "{smooth}: {:?}", jumps(smooth));
        }
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let three = |v: f64| num(v, 3);
        // the secant's slope, the strip's mean height, closes in on f(x) as h shrinks
        let mut s = Ftc::create();
        s.set_up(0);
        assert_eq!(s.film.map(|f| (f.which, f.from, f.to)), Some((Which::LogWidth, 0.0, -2.0)));
        assert_eq!((s.h, s.f.text.as_str(), s.x), (1.0, "cos(x)", 0.6));
        let fx = s.set_look().fx;
        assert_eq!(three(fx), "0.825");
        let mut gaps = Vec::new();
        for (h, want) in [(1.0, "0.435"), (0.5, "0.653"), (0.1, "0.796"), (0.01, "0.822")] {
            let l = s.measure(s.a, s.x, h);
            let m = l.secant().unwrap();
            assert_eq!(three(m), want, "h {h}");
            // the strip's area is the secant's rise: A(x + h) − A(x)
            let rise = s.measure(s.a, s.x + h, h).area.unwrap() - l.area.unwrap();
            assert!((rise - l.strip.unwrap()).abs() < 1e-12, "h {h}");
            assert_eq!(l.slope, Some((fx, fx)));
            gaps.push(fx - m);
        }
        assert!(gaps.windows(2).all(|w| w[1] < w[0] && w[1] > 0.0), "{gaps:?}");
        assert!((after(0).h - 0.01).abs() < 1e-12);

        // below the axis A falls; its bottoms and top where f crosses the axis
        let mut s = Ftc::create();
        s.set_up(1);
        assert_eq!(s.film.map(|f| (f.which, f.from, f.to)), Some((Which::End, -3.8, 3.8)));
        let xs: Vec<f64> = (0..=760).map(|i| -3.8 + i as f64 * 0.01).collect();
        let areas: Vec<f64> = xs.iter().map(|&x| s.measure(s.a, x, s.h).area.unwrap()).collect();
        for i in 1..xs.len() {
            let f_mid = value(&s.f.expr, 0.5 * (xs[i - 1] + xs[i]));
            if f_mid.abs() > 1e-3 {
                assert_eq!(areas[i] > areas[i - 1], f_mid > 0.0, "at {}", xs[i]);
            }
        }
        let turns: Vec<(f64, f64, bool)> = (1..xs.len() - 1)
            .filter(|&i| (areas[i] - areas[i - 1]) * (areas[i + 1] - areas[i]) < 0.0)
            .map(|i| (xs[i], areas[i], areas[i] > areas[i - 1]))
            .collect();
        assert_eq!(turns.len(), 3, "{turns:?}");
        for ((x, area, top), (want_x, want_area, want_top)) in turns.iter().zip([(-3.0, "−0.925", false), (0.0, "1.606", true), (3.0, "−0.925", false)]) {
            assert!((x - want_x).abs() < 0.011 && three(*area) == want_area && *top == want_top, "{turns:?}");
        }
        let f_at = |x: f64| value(&s.f.expr, x);
        let steepest = xs.iter().copied().fold(0.0f64, |m, x| m.max(f_at(x).abs()));
        assert_eq!((num(steepest, 2), num(f_at(-3.8).abs(), 2), num(f_at(3.8), 2)), ("2.58".into(), "2.58".into(), "2.58".into()));
        assert!(areas[760].abs() < 1e-9 && after(1).x == 3.8);

        // moving the start shifts A by the area between the starts, and no slope
        let mut s = Ftc::create();
        s.set_up(2);
        assert_eq!(s.film.map(|f| (f.which, f.from, f.to)), Some((Which::Start, -3.0, 0.5)));
        let (before, later) = (s.measure(-3.0, 1.0, s.h), s.measure(0.5, 1.0, s.h));
        assert_eq!((three(before.area.unwrap()), three(later.area.unwrap())), ("2.106".into(), "0.376".into()));
        let between = s.measure(-3.0, 0.5, s.h).area.unwrap();
        assert_eq!(three(between), "1.730");
        assert!((before.area.unwrap() - later.area.unwrap() - between).abs() < 1e-12);
        assert_eq!(three(before.fx), "0.607");
        assert!(before.slope == later.slope && before.slope == Some((before.fx, before.fx)));
        // the whole curve, not one point: every end of a cell, by the same amount
        let (c0, c1) = (s.grid.curve(&s.f.expr, -3.0).0, s.grid.curve(&s.f.expr, 0.5).0);
        assert!(c0.iter().zip(&c1).all(|(p, q)| (p - q - between).abs() < 1e-12));
        for a in [-3.0, -1.234, 0.5] {
            assert_eq!(s.measure(a, a, s.h).area, Ok(0.0));
        }
        assert_eq!(after(2).a, 0.5);

        // a jump in f is a corner in A
        let s = after(3);
        let l = s.set_look();
        assert_eq!((s.f.text.as_str(), l.x, l.fx), ("sign(x)", 0.0, 0.0));
        assert_eq!(l.slope, Some((-1.0, 1.0)));
        assert!(l.corner());
        assert_eq!(num(l.area.unwrap(), 4), "−2.5000");
        for h in [0.5, 0.1, 0.01, 0.001] {
            assert!((s.measure(s.a, s.x, h).secant().unwrap() - 1.0).abs() < 1e-9, "h {h}");
        }
        // no jump in A: just either side of the corner it is as near A(0) as you like
        for d in [1e-2, 1e-4, 1e-6] {
            for side in [-1.0, 1.0] {
                let near = s.measure(s.a, side * d, s.h).area.unwrap();
                assert!((near - l.area.unwrap() - d).abs() < 1e-9, "{d} {side}: {near}");
            }
        }

        // an area by undoing a derivative
        let s = after(4);
        let l = s.set_look();
        assert!(s.show_anti && s.known() == Some(SIN));
        let (area, anti) = (l.area.unwrap(), l.anti.unwrap());
        let closed = -(2.5f64).cos() + (0.5f64).cos();
        assert!((anti - closed).abs() < 1e-15 && (area - closed).abs() < 1e-12);
        assert_eq!((num(area, 4), num(anti, 4)), ("1.6787".into(), "1.6787".into()));
        let shift = -(0.5f64).cos();
        assert_eq!(num(-shift, 3), "0.878");
        let curve = s.grid.curve(&s.f.expr, s.a).0;
        assert!(s.grid.xs.iter().zip(&curve).all(|(t, v)| (-t.cos() - v - shift).abs() < 1e-12));
    }

    /// The opening picture: the strip, the tangent and the secant all to be seen.
    #[test]
    fn the_scene_opens_on_a_curve_with_both_signs() {
        let s = Ftc::create();
        let l = s.set_look();
        assert_eq!((s.f.text.as_str(), s.a, s.x, s.h), ("cos(x)", -3.0, 0.6, 0.4));
        assert!(l.area.is_ok() && l.fx > 0.0 && value(&s.f.expr, s.a) < 0.0);
        let gap = l.fx - l.secant().unwrap();
        assert!(gap > 0.05 && gap < 0.3, "{gap}");
    }

    /// The scene with a formula typed in, its ends and strip set.
    fn typed(formula: &str, a: f64, x: f64, h: f64) -> Ftc {
        let mut s = Ftc::create();
        s.f.set(formula);
        s.formula_changed();
        s.morph = Follow::at([1.0]);
        for (which, v) in [(Which::Start, a), (Which::End, x), (Which::LogWidth, h.log10())] {
            s.set(which, v);
        }
        s
    }

    /// Whatever the formula and wherever the ends, every number is finite or says it
    /// has none, and the panel says why.
    #[test]
    fn degenerate_settings_are_told_honestly() {
        // the two ends on top of each other: no area, and the slope is still f(x)
        let s = typed("cos(x)", 0.7, 0.7, H_MIN);
        let l = s.set_look();
        assert_eq!(l.area, Ok(0.0));
        assert_eq!(l.slope, Some((0.7f64.cos(), 0.7f64.cos())));
        assert!((l.secant().unwrap() - 0.7f64.cos()).abs() < 1e-3);
        // the end at the view's edge, the strip reaching beyond it
        let s = typed("cos(x)", -4.0, 4.0, H_MAX);
        let l = s.set_look();
        assert!((l.area.unwrap() - (4f64.sin() + 4f64.sin())).abs() < 1e-9);
        assert!((l.strip.unwrap() - (6f64.sin() - 4f64.sin())).abs() < 1e-9);
        // nothing at all to sum, and no value anywhere
        let s = typed("0", -1.0, 2.0, 0.1);
        assert_eq!(s.set_look().area, Ok(0.0));
        assert_eq!(keep_range([0.0, 0.0], [0.0, 0.0]), [-1.0, 1.0]);
        let s = typed("sqrt(-1-x^2)", -1.0, 2.0, 0.1);
        let l = s.set_look();
        assert!(matches!(l.area, Err(Fault::NoValue(..))) && l.slope.is_none() && l.secant().is_none() && l.fx.is_nan());
        assert!(s.notes(&l).iter().any(|n| n.contains("has no value")));
        // the start where f has none: no area either way, not even the near side
        let s = typed("sqrt(x)", -1.0, 2.0, 0.1);
        assert!(s.set_look().area.is_err());
        // (but at a itself, an end of a cell, where A(a) = 0: one point, which draws nothing)
        let curve = s.grid.curve(&s.f.expr, -1.0).0;
        assert!(curve.iter().zip(&s.grid.xs).all(|(v, t)| v.is_nan() || (*t == -1.0 && *v == 0.0)), "{curve:?}");
        // across a pole: no area, no secant, and the panel says why; F is not compared
        let s = typed("1/x", 0.5, -1.5, 0.4);
        let l = s.set_look();
        assert!(matches!(l.area, Err(Fault::Unbounded(t)) if t.abs() < 1e-3));
        assert!(l.secant().is_none() && l.anti.is_none() && l.strip.is_ok());
        let notes = s.notes(&l);
        assert!(notes.iter().any(|n| n.contains("grows past any size")) && notes.iter().any(|n| n.contains("left of")), "{notes:?}");
        // on the near side of it, A is there, and F(x) - F(a) is ln|x| - ln|a|
        let l = s.measure(0.5, 3.0, 0.4);
        assert!((l.area.unwrap() - 6f64.ln()).abs() < 1e-9 && (l.anti.unwrap() - 6f64.ln()).abs() < 1e-12);
        // a corner says what its two slopes are, and that there is no tangent
        let s = after(3);
        let notes = s.notes(&s.set_look());
        assert!(notes.iter().any(|n| n.contains("corner") && n.contains("−1.00") && n.contains("1.00")), "{notes:?}");
        // a single point off the curve (sign(x)^2 at 0): the slope is the curve's
        let s = typed("sign(x)^2", -1.0, 0.0, 0.1);
        let l = s.set_look();
        assert_eq!((l.fx, l.slope), (0.0, Some((1.0, 1.0))));
        assert!(s.notes(&l).iter().any(|n| n.contains("off the curve")));
        // every window, down to a speck: rectangles that are rectangles
        for (w, h) in [(1600.0, 1000.0), (1100.0, 700.0), (390.0, 800.0), (800.0, 390.0), (240.0, 180.0), (120.0, 90.0), (1.0, 1.0)] {
            for inset in [0.0, 44.0] {
                let lay = Layout::of(Vec2::new(w, h), inset);
                for r in [lay.top, lay.bottom, lay.knob, lay.numbers, lay.zoom, lay.note] {
                    assert!(r.0.is_finite() && r.1.is_finite() && r.1.x >= r.0.x && r.1.y >= r.0.y, "{w}x{h}: {r:?}");
                }
                assert!(lay.top.1.y <= lay.bottom.0.y, "{w}x{h}");
            }
        }
        // every curve on the buttons, with the ends anywhere: finite or none
        for (i, c) in PRESETS.iter().enumerate() {
            for (a, x) in [(-4.0, 4.0), (0.0, 0.0), (3.9, -3.9), (-0.001, 0.001), (2.0, -1e-9)] {
                let s = typed(c.formula, a, x, 0.3);
                let l = s.set_look();
                for v in [l.area.ok(), l.strip.ok(), l.secant(), l.anti].into_iter().flatten() {
                    assert!(v.is_finite(), "{i} {a} {x}: {l:?}");
                }
                let (fs, areas) = s.curves(l.a);
                assert!(fs.len() == CELLS + 1 && areas.len() == CELLS + 1);
                assert!(s.anchor_of(&l, &areas).iter().all(|v| v.is_finite() || v.is_nan()));
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        let mut s = Ftc::create();
        let l = s.set_look();
        println!("opening: {l:?}");
        s.set_up(0);
        for h in [1.0, 0.5, 0.1, 0.01, 0.001] {
            let l = s.measure(s.a, s.x, h);
            println!("trial 0, h {h}: secant {:?}, f(x) {}, A(x) {:?}", l.secant(), l.fx, l.area);
        }
        s.set_up(1);
        for x in [-3.5, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.5] {
            let l = s.measure(s.a, x, s.h);
            println!("trial 1, x {x}: A {:?}, f {}", l.area, l.fx);
        }
        s.set_up(2);
        for a in [-3.0, 0.5] {
            let l = s.measure(a, s.x, s.h);
            println!("trial 2, a {a}: A {:?}, f(x) {}", l.area, l.fx);
        }
        let s = after(3);
        let l = s.set_look();
        println!("trial 3: {l:?}");
        for h in [0.5, 0.1, 0.01] {
            println!("  h {h}: secant {:?}", s.measure(s.a, s.x, h).secant());
        }
        let s = after(4);
        let l = s.set_look();
        println!("trial 4: {l:?}; F(a) {}", -s.a.cos());
    }
}

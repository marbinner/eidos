//! Span and independence: walks along two or three arrows, the points such walks can
//! end at, and when an arrow adds nothing.

use super::subspaces::{keeps, Inputs, Still};
use super::*;
use glam::DVec3;

pub const DEF: SceneDef = SceneDef {
    id: "span",
    title: "Span and independence",
    group: "Linear algebra: vectors and maps",
    blurb: "Walk along two or three arrows: the line, plane or space they can reach, and when one of them adds nothing.",
    make: |_| Box::new(Span::create()),
};

/// The arrows' colours (v₁, v₂, v₃): each for that arrow, its leg of a walk and its number.
const COLS: [Color; 3] = [pal::GREEN, pal::ORANGE, pal::PINK];
/// The arrows' names: as markup, to go inside a formula, and as labels of their own.
const NAMES: [&str; 3] = ["v_1", "v_2", "v_3"];
const LABELS: [&str; 3] = ["$v_1$", "$v_2$", "$v_3$"];
/// The point to reach.
const B_COL: Color = pal::WHITE;
/// The span: a line, a see-through sheet, or a slanted lattice.
const SPAN_COL: Color = pal::BLUE;
/// Where a walk ends: the dots a sweep leaves, and the nearest point to a b out of reach.
const END_COL: Color = pal::CYAN;
/// The other walks that end at b too, as their key shows them (they are drawn in the
/// arrows' colours, faint).
const GHOST: Color = Color::hex(0x6F8C7E);
/// The warning that the arrows are nearly dependent.
const WARN: Color = pal::YELLOW;

/// What counts as nothing. An arrow lies in the span of the ones before it when what is
/// left of it, once its shadows on them are taken away, is shorter than `TOL` times the
/// longest arrow; b is in the span when it is that close to it. The coordinates are kept
/// in single precision, about seven digits: an arrow put exactly into a plane (by an
/// experiment, or by the magnet below) ends up a ten-millionth or so of its length off
/// it, and must still count as in it. Arithmetic in f64 adds far less.
const TOL: f64 = 1e-5;
/// A length below this is no length at all, however short the other arrows are.
const TINY: f64 = 1e-6;
/// The magnet: an arrow dragged to within this many degrees of the line of another, or
/// of the plane of the other two, is put exactly into it, and so is b dragged that near
/// to the span. Without it dependence could only be come near by hand, never reached.
const SNAP_DEG: f64 = 3.0;
/// Independent arrows one of which is within this many degrees of the span of the
/// others are called nearly dependent: walks along them need large numbers.
const NEAR_DEG: f64 = 10.0;
/// How far a tip can be taken from the origin.
const REACH: f32 = 4.0;

/// How quickly an arrow (or b) is drawn gliding to where the magnet has put it (the
/// time in which it covers two thirds of the way, in seconds), and when it is there.
const GLIDE: f32 = 0.045;
const ARRIVED: f32 = 1e-4;
/// How quickly a new picture of the span fades in and the old one out (seconds).
const FADE: f32 = 0.12;

/// The sweep, in seconds. From the walk shown it blends onto a path of its own by
/// `S_IN`, speeds up for `RAMP` seconds, slows down over the `RAMP` seconds before
/// `S_PAINT`, and blends back onto the walk to b by `S_END`. It leaves its dots between
/// `S_IN` and `S_PAINT`.
const S_IN: f32 = 1.2;
const RAMP: f32 = 1.5;
const S_PAINT: f32 = 9.7;
const S_END: f32 = 11.2;
/// How fast the path is followed, slow and fast (path units per second of the scene):
/// slowly enough at first for the numbers to be read, then fast enough for the dots to
/// spread over what the walks can reach in a few seconds, but no faster: the walk is to
/// move evenly, frame after frame (see `path`).
const SLOW: f64 = 0.6;
const FAST: f64 = 5.0;
/// A dot every so much of the path, so that the dots lie evenly along it whatever the
/// speed and whatever the display's frame rate.
const DOT_STEP: f64 = 0.012;

/// How quickly the dots dim once a sweep is over (seconds).
const DIMMING: f32 = 0.5;

/// The slide along the walks to b: seconds it lasts, seconds of one swing to and fro,
/// how far it goes, and the seconds over which it eases in and out.
const SLIDE_TIME: f32 = 10.0;
const SWING: f32 = 5.0;
const SWING_AMP: f32 = 1.0;
const SLIDE_EASE: f32 = 0.8;
/// The other walks to b drawn faint: how far each is along the walk that comes back to 0.
const GHOSTS: [f32; 4] = [-1.0, -0.5, 0.5, 1.0];

/// The arrows the experiments are made of. `V3_IN` is v₂ − v₁/2, in the plane of the
/// first two; `V3_UP` is that lifted by 1.25 straight up. `B_IN` is v₁ + v₂/2, in the
/// plane; `B_OFF` is v₁ + v₂/2 + `V3_UP`. (Every coordinate is a multiple of 1/16, so
/// these are exactly what they say.)
const V1: [f32; 3] = [1.75, 0.25, 0.5];
const V2: [f32; 3] = [0.25, 1.5, 0.75];
const V3_IN: [f32; 3] = [-0.625, 1.375, 0.5];
const V3_UP: [f32; 3] = [-0.625, 1.375, 1.75];
const V2_BACK: [f32; 3] = [-1.3125, -0.1875, -0.375];
const B_IN: [f32; 3] = [1.875, 1.0, 0.875];
const B_OFF: [f32; 3] = [1.25, 2.375, 2.625];

/// An experiment: its button, what to look for, how many arrows and which, b, and
/// whether it sweeps or slides along the walks to b.
struct Trial {
    name: &'static str,
    caption: &'static str,
    count: usize,
    v: [[f32; 3]; 3],
    b: [f32; 3],
    sweep: bool,
    slide: bool,
    /// The direction the picture is seen from (yaw, pitch), if not the usual one.
    seen: Option<(f32, f32)>,
}

const TRIALS: [Trial; 5] = [
    Trial {
        name: "two arrows paint a plane",
        caption: r"The numbers $c_1$ and $c_2$ run between $-1$ and $1$ by themselves, slowly and then fast, and each walk $c_1 v_1 + c_2 v_2$ leaves a dot where it ends. Every dot lies in one flat sheet through $0$: two independent arrows span a plane. Then the walk settles on $b$, which is in that plane: $b = 1.00v_1 + 0.50v_2$, and no other walk ends there.",
        count: 2,
        v: [V1, V2, V3_UP],
        b: B_IN,
        sweep: true,
        slide: false,
        seen: None,
    },
    Trial {
        name: "a third arrow in the plane",
        caption: r"$v_3 = -0.50v_1 + 1.00v_2$ lies in the plane of $v_1$ and $v_2$. Three arrows now, but the dots of the sweep still stay in that one sheet: $v_3$ adds nothing, and the rank stays 2. $b$ is off the sheet, so no walk reaches it: the nearest point of the plane misses it by 1.11 (dashed, at a right angle).",
        count: 3,
        v: [V1, V2, V3_IN],
        b: B_OFF,
        sweep: true,
        slide: false,
        seen: Some(ALONG),
    },
    Trial {
        name: "lift it off the plane",
        caption: r"$v_3$ is lifted 1.25 straight up, out of the plane. Now the dots do not stay in a sheet: they spread through a slanted box on both sides of it, and walks with larger numbers reach all of space (rank 3). The same $b$ is reached, by exactly one walk: $b = 1.00v_1 + 0.50v_2 + 1.00v_3$. Those three numbers are $b$'s coordinates in the basis $v_1, v_2, v_3$.",
        count: 3,
        v: [V1, V2, V3_UP],
        b: B_OFF,
        sweep: true,
        slide: false,
        seen: Some(ALONG),
    },
    Trial {
        name: "many walks to one point",
        caption: r"$v_3$ is back in the plane, and so is $b$. Now many walks reach $b$: the walk slides from one to the next while its end stays on $b$ (four more are drawn faint). As $v_3 = -0.50v_1 + 1.00v_2$, a step $t$ along $v_3$ is undone by $0.5t$ along $v_1$ and $-t$ along $v_2$: $b = (1 + 0.5t)v_1 + (0.5 - t)v_2 + tv_3$ for every $t$.",
        count: 3,
        v: [V1, V2, V3_IN],
        b: B_IN,
        sweep: false,
        slide: true,
        seen: None,
    },
    Trial {
        name: "two arrows in one line",
        caption: r"$v_2 = -0.75v_1$ points back along the line of $v_1$. The sweep's dots stay on that line: two arrows, but the span is only a line (rank 1). $b$ is where it was in the first experiment, but off this line: no walk reaches it now, and the nearest point of the line misses it by 0.78.",
        count: 2,
        v: [V1, V2_BACK, V3_UP],
        b: B_IN,
        sweep: true,
        slide: false,
        seen: None,
    },
];

// ----- the arithmetic --------------------------------------------------------------------

/// What is left of `p` once its shadows on the orthonormal arrows `q` are taken away
/// (twice over, so that it is at right angles to them to the last digit even when it is a
/// small difference of large numbers).
fn leftover(p: DVec3, q: &[DVec3]) -> DVec3 {
    let mut w = p;
    for _ in 0..2 {
        for qi in q {
            w -= *qi * qi.dot(w);
        }
    }
    w
}

/// An orthonormal basis of the span of `arrows`: each made perpendicular to the ones kept
/// before it, and kept if what is left of it is longer than `small`.
fn orthonormal(arrows: &[DVec3], small: f64) -> Vec<DVec3> {
    let mut q = Vec::with_capacity(3);
    for a in arrows {
        let w = leftover(*a, &q);
        let len = w.length();
        if len > small {
            q.push(w / len);
        }
    }
    q
}

/// The angle, in degrees, between `p` and the span with orthonormal basis `q` (0 for a
/// point with no length, which is in every span).
fn degrees_off(p: DVec3, q: &[DVec3]) -> f64 {
    let len = p.length();
    if len <= TINY {
        return 0.0;
    }
    (leftover(p, q).length() / len).min(1.0).asin().to_degrees()
}

/// Everything the picture and the words are made of, worked out in double precision
/// from the arrows and b.
#[derive(Clone, Debug)]
struct Analysis {
    count: usize,
    v: [DVec3; 3],
    /// A leftover shorter than this is nothing (see `TOL`).
    small: f64,
    /// The arrows that add a direction to the ones before them.
    kept: [bool; 3],
    rank: usize,
    /// An orthonormal basis of the span (the first `rank` of these).
    q: [DVec3; 3],
    /// Each arrow as a walk along the kept ones: vⱼ = Σᵢ along[j][i] vᵢ.
    along: [[f64; 3]; 3],
    /// The point of the span nearest to b, and the walk to it along the kept arrows alone.
    p: DVec3,
    base: [f64; 3],
    /// How far b is from the span, and whether that is nothing.
    miss: f64,
    reach: bool,
    /// One walk that ends where it starts for each arrow left out: eⱼ − along[j].
    null: [[f64; 3]; 3],
    nulls: usize,
    /// Of independent arrows (two or three), the one nearest to the span of the others,
    /// and how many degrees it is off it.
    tightest: Option<(usize, f64)>,
}

impl Analysis {
    fn of(v: [Vec3; 3], count: usize, b: Vec3) -> Analysis {
        let count = count.clamp(1, 3);
        let v = v.map(|a| a.as_dvec3());
        let b = b.as_dvec3();
        let scale = v[..count].iter().map(|a| a.length()).fold(0.0, f64::max);
        let small = (TOL * scale).max(TINY);
        // each arrow made perpendicular to the ones kept before it, with its numbers on
        // them: r[m] are the kept arrow m's numbers on q₀ … q_m
        let mut q: Vec<DVec3> = Vec::with_capacity(3);
        let mut r: Vec<Vec<f64>> = Vec::with_capacity(3);
        let mut order: Vec<usize> = Vec::with_capacity(3);
        let mut kept = [false; 3];
        let mut along = [[0.0; 3]; 3];
        for j in 0..count {
            let w = leftover(v[j], &q);
            let y: Vec<f64> = q.iter().map(|qi| qi.dot(v[j] - w)).collect();
            let len = w.length();
            if len > small {
                q.push(w / len);
                let mut y = y;
                y.push(len);
                r.push(y);
                order.push(j);
                kept[j] = true;
                along[j][j] = 1.0;
            } else {
                for (m, a) in solve_upper(&r, &y).into_iter().enumerate() {
                    along[j][order[m]] = a;
                }
            }
        }
        let rank = q.len();
        // the nearest point of the span, and the walk to it along the kept arrows
        let yb: Vec<f64> = q.iter().map(|qi| qi.dot(b)).collect();
        let p = q.iter().zip(&yb).fold(DVec3::ZERO, |p, (qi, y)| p + *qi * *y);
        let mut base = [0.0; 3];
        for (m, c) in solve_upper(&r, &yb).into_iter().enumerate() {
            base[order[m]] = c;
        }
        let miss = (b - p).length();
        let reach = miss <= (TOL * scale.max(b.length())).max(TINY);
        let mut null = [[0.0; 3]; 3];
        let mut nulls = 0;
        for j in (0..count).filter(|&j| !kept[j]) {
            for i in 0..3 {
                null[nulls][i] = -along[j][i];
            }
            null[nulls][j] = 1.0;
            nulls += 1;
        }
        let tightest = (rank == count && count >= 2)
            .then(|| {
                (0..count)
                    .map(|j| {
                        let others: Vec<DVec3> = (0..count).filter(|&i| i != j).map(|i| v[i]).collect();
                        (j, degrees_off(v[j], &orthonormal(&others, small)))
                    })
                    // (of arrows as near as each other to the nearest tenth of a degree, the last:
                    // "v₃ is 4.0° out of the plane of v₁ and v₂" rather than the same of v₂)
                    .min_by(|x, y| (x.1 * 10.0).round().total_cmp(&(y.1 * 10.0).round()).then(y.0.cmp(&x.0)))
            })
            .flatten();
        let mut qs = [DVec3::ZERO; 3];
        qs[..rank].copy_from_slice(&q);
        Analysis { count, v, small, kept, rank, q: qs, along, p, base, miss, reach, null, nulls, tightest }
    }

    /// The point a walk with these numbers ends at.
    #[cfg(test)]
    fn end(&self, c: &[f64; 3]) -> DVec3 {
        (0..self.count).fold(DVec3::ZERO, |p, i| p + self.v[i] * c[i])
    }

    fn independent(&self) -> bool {
        self.rank == self.count
    }

    /// Are the arrows independent, but one of them within `NEAR_DEG` of the span of the others?
    fn nearly(&self) -> Option<(usize, f64)> {
        self.tightest.filter(|t| t.1 < NEAR_DEG)
    }

    /// Is b reached by more than one walk?
    fn many(&self) -> bool {
        self.reach && self.nulls > 0
    }
}

/// Solve R x = y for the upper triangular R whose column m is `r[m]` (its numbers from
/// the top down to the diagonal). The diagonal is a length longer than `small`, so
/// nothing is divided by zero.
fn solve_upper(r: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = r.len().min(y.len());
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for (m, xm) in x.iter().enumerate().skip(i + 1) {
            s -= r[m][i] * xm;
        }
        x[i] = s / r[i][i];
    }
    x
}

/// The sweep's path: three numbers between −1 and 1, each going to and fro at its own
/// rate (none a whole multiple of another, so that together they come near every
/// combination). Each turns a little more sharply than a sine does, so that the dots are
/// spread more evenly, but not much more: at the sweep's full speed a sharp turn takes
/// less than a frame, and the walk is seen to jerk there. (The rates and starting points
/// were picked from many: at that speed and in that time, their dots reach every one of
/// 4 × 4 × 4 parts of the box and 8 × 8 parts of the square, and the walk's speed rises
/// and falls without a jerk at 60 frames a second.)
fn path(u: f64) -> [f64; 3] {
    let k = 0.6f64;
    let wave = |x: f64| (k * x.sin()).asin() / k.asin();
    [wave(u + 5.95), wave(1.414 * u + 1.45), wave(1.618 * u + 3.83)]
}

/// ∫₀ˣ smoothstep(s / w) ds: how far a speed that rises smoothly from 0 to 1 over `w`
/// seconds, starting at 0, has carried something by time `x`.
fn eased_integral(x: f64, w: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else if x < w {
        let u = x / w;
        w * (u * u * u - 0.5 * u * u * u * u)
    } else {
        x - 0.5 * w
    }
}

/// How far along its path the sweep is `tau` seconds in: slow, then fast, then slow.
fn progress(tau: f64) -> f64 {
    let (ramp, up, down) = (RAMP as f64, S_IN as f64, (S_PAINT - RAMP) as f64);
    SLOW * tau + (FAST - SLOW) * (eased_integral(tau - up, ramp) - eased_integral(tau - down, ramp))
}

/// How much of the walk shown is the sweep's path, `tau` seconds in (the rest is the walk
/// to b): it blends in, holds, and blends out.
fn sweep_share(tau: f32) -> f64 {
    if tau < S_IN {
        smoothstep(tau / S_IN) as f64
    } else if tau < S_PAINT {
        1.0
    } else {
        1.0 - smoothstep((tau - S_PAINT) / (S_END - S_PAINT)) as f64
    }
}

/// How many dots a sweep has left by `tau` seconds in, and the path's place for dot k.
fn dots_by(tau: f32) -> usize {
    if tau < S_IN {
        return 0;
    }
    let (start, now) = (progress(S_IN as f64), progress(tau.min(S_PAINT) as f64));
    ((now - start) / DOT_STEP + 1e-9).floor() as usize + 1
}

fn dot_place(k: usize) -> f64 {
    progress(S_IN as f64) + k as f64 * DOT_STEP
}

/// How far along the walks that come back to 0 the walk to b has been taken, `sigma`
/// seconds into a slide that started at `from`: it eases away from there, swings to and
/// fro, and comes to rest at 0.
fn slide_at(sigma: f32, from: [f32; 2]) -> [f32; 2] {
    let leave = 1.0 - smoothstep(sigma / SLIDE_EASE);
    let swell = smoothstep(sigma / SLIDE_EASE) * smoothstep((SLIDE_TIME - sigma) / SLIDE_EASE);
    let turn = TAU * sigma / SWING;
    [from[0] * leave + SWING_AMP * turn.sin() * swell, from[1] * leave + SWING_AMP * 0.5 * (1.0 - turn.cos()) * swell]
}

// ----- words -----------------------------------------------------------------------------

/// A number of steps as written in a walk: two decimals, or none once it is in the
/// thousands (it is then far too large to be read to the hundredth anyway).
fn amount(c: f64) -> String {
    if c.abs() < 999.995 {
        num(c, 2)
    } else {
        num(c, 0)
    }
}

/// A length as written: two decimals, or two significant digits if it is smaller than
/// that (a miss of 0.004 is not "0.00").
fn length_text(x: f64) -> String {
    if !(x.is_finite() && x > 0.0) || x >= 0.005 {
        return num(x, 2);
    }
    let decimals = (1.0 - x.log10().floor()).clamp(3.0, 12.0) as usize;
    num(x, decimals)
}

fn degrees_text(d: f64) -> String {
    format!("{}°", num(d, 1))
}

/// A walk written out, $1.00v_1 - 0.50v_2 + 0.25v_3$, as pieces, each a formula: each
/// term in its arrow's colour, the signs between them in the colour of text.
fn terms(c: &[f64; 3], count: usize) -> Vec<(String, Color)> {
    let mut pieces = Vec::new();
    for i in 0..count {
        let text = amount(c[i]);
        let negative = text.starts_with('−');
        let size = text.trim_start_matches('−');
        if i > 0 {
            pieces.push((if negative { "$-$" } else { "$+$" }.to_string(), pal::TEXT));
            pieces.push((format!("${size}{}$", NAMES[i]), COLS[i]));
        } else {
            pieces.push((format!("${text}{}$", NAMES[i]), COLS[i]));
        }
    }
    pieces
}

/// The same as one formula.
#[cfg(test)]
fn terms_text(c: &[f64; 3], count: usize) -> String {
    format!("${}$", terms(c, count).into_iter().map(|p| p.0.trim_matches('$').to_string()).collect::<Vec<_>>().join(" "))
}

/// An arrow left out as a walk along the kept ones, as markup to go in a formula:
/// `−0.50v_1 + 1.00v_2` (the terms that round to nothing are left out).
fn as_walk(an: &Analysis, j: usize) -> String {
    let mut out = String::new();
    for i in (0..an.count).filter(|&i| an.kept[i]) {
        let text = amount(an.along[j][i]);
        if text.trim_start_matches('−') == "0.00" {
            continue;
        }
        if out.is_empty() {
            out = format!("{text}{}", NAMES[i]);
        } else if let Some(size) = text.strip_prefix('−') {
            out += &format!(" − {size}{}", NAMES[i]);
        } else {
            out += &format!(" + {text}{}", NAMES[i]);
        }
    }
    if out.is_empty() {
        "0".to_string()
    } else {
        out
    }
}

/// What the span is, as words after "only" or "every point of".
fn span_name(rank: usize) -> &'static str {
    ["the point 0", "a line", "a plane", "space"][rank.min(3)]
}

fn count_name(n: usize) -> &'static str {
    ["No", "One", "Two", "Three"][n.min(3)]
}

/// The verdict on the arrows, in one sentence.
fn headline(an: &Analysis) -> String {
    match an.rank {
        0 => format!("{} arrow has any length: the span is the point $0$ alone.", if an.count == 2 { "Neither" } else { "No" }),
        3 => "Three independent arrows: every point of space, one walk each.".to_string(),
        r if an.independent() => format!("{} independent arrows: every point of {}, one walk each.", count_name(an.count), span_name(r)),
        r => format!("{} dependent arrows: only {}, and many walks to each point of it.", count_name(an.count), span_name(r)),
    }
}

/// The rank, and why it is what it is.
fn rank_line(an: &Analysis) -> String {
    let why = if an.rank == 0 {
        "no arrow has any length, so none adds a direction.".to_string()
    } else if an.independent() {
        if an.count == 2 { "neither arrow is a walk along the other.".to_string() } else { "no arrow is a walk along the other two.".to_string() }
    } else {
        let left_out: Vec<String> = (0..an.count)
            .filter(|&j| !an.kept[j])
            .map(|j| if an.v[j].length() <= an.small { format!("{} has no length", LABELS[j]) } else { format!("${} = {}$", NAMES[j], as_walk(an, j)) })
            .collect();
        format!("{}, so {} nothing.", left_out.join(" and "), if left_out.len() > 1 { "they add" } else { "it adds" })
    };
    format!("Rank {}: {why}", an.rank)
}

/// What the walk shown is: its left side, as written before its terms.
fn walk_name(an: &Analysis, sweeping: bool) -> &'static str {
    if sweeping {
        "walk:"
    } else if an.reach {
        "$b =$"
    } else {
        r"$\text{nearest point} =$"
    }
}

/// What becomes of b: reached once, many times, or not at all.
fn about_b(an: &Analysis, walk: &[f64; 3], sweeping: bool, sliding: bool) -> String {
    if sweeping {
        return "Each walk leaves a dot where it ends: with the numbers between $-1$ and $1$, the dots paint part of the span.".to_string();
    }
    if !an.reach {
        return if an.rank == 0 {
            format!("$b$ is not $0$: no walk reaches it. It is {} away.", length_text(an.miss))
        } else {
            format!("$b$ is off the {}: no walk reaches it. The nearest point misses it by {}.", ["", "line", "plane"][an.rank.min(2)], length_text(an.miss))
        };
    }
    if an.many() {
        return if an.rank == 0 {
            "$b$ is $0$, and every walk ends there.".to_string()
        } else if sliding {
            "$b$ is in the span, and many walks reach it: this one slides from one to the next while its end stays on $b$.".to_string()
        } else {
            "$b$ is in the span, and many walks reach it: this one, the faint ones, and more. Slide along them with the button below.".to_string()
        };
    }
    let numbers: Vec<String> = walk[..an.count].iter().map(|c| amount(*c)).collect();
    let names: Vec<&str> = NAMES[..an.count].to_vec();
    let of = if an.count == 2 { " of this plane" } else { "" };
    format!("One walk reaches $b$: $({})$ are its coordinates in the basis ${}${of}.", numbers.join(", "), names.join(", "))
}

/// The warning that walks need large numbers, if the arrows are nearly dependent.
fn warning(an: &Analysis, walk: &[f64; 3], sweeping: bool) -> Option<String> {
    let (j, deg) = an.nearly()?;
    let which = if an.count == 2 {
        format!("The lines of $v_1$ and $v_2$ are only {} apart", degrees_text(deg))
    } else {
        let others: Vec<&str> = (0..3).filter(|&i| i != j).map(|i| LABELS[i]).collect();
        format!("{} is only {} out of the plane of {} and {}", LABELS[j], degrees_text(deg), others[0], others[1])
    };
    let largest = walk[..an.count].iter().fold(0.0f64, |m, c| m.max(c.abs()));
    let here = if sweeping { String::new() } else { format!(" (here up to {})", amount(largest)) };
    Some(format!("{which}: nearly dependent, so walks along them need large numbers{here}."))
}

/// What the picture shows that has no label of its own.
fn unlabelled(dots: bool, ghosts: bool, missed: bool) -> String {
    let mut parts = Vec::new();
    if dots {
        parts.push("cyan dots: where the sweep's walks ended");
    }
    if ghosts {
        parts.push("faint walks: other walks to $b$");
    }
    if missed {
        parts.push("dashed white: the miss, at a right angle to the span");
    }
    parts.join("   ")
}

// ----- the panel of words over the picture ---------------------------------------------------

/// How many logical pixels a line of plain words takes in the picture's type (a closer
/// estimate than `text_width`, which takes every character as the same width: digits
/// are wider than that, and spaces and subscripts narrower).
fn measure(text: &str, size: f32) -> f32 {
    text.chars()
        .map(|c| match c {
            '0'..='9' | '−' | '+' | '=' | '≠' | '_' => 0.564,
            '.' | ',' | ':' | ';' | '\'' | '’' | '!' | 'i' | 'l' | 'j' | 'I' => 0.25,
            ' ' => 0.23,
            '(' | ')' | '°' => 0.32,
            '₀'..='₉' => 0.36,
            't' | 'r' | 'f' => 0.39,
            'm' | 'w' | 'M' | 'W' => 0.8,
            'A'..='Z' => 0.64,
            _ => 0.55,
        })
        .sum::<f32>()
        * size
}

/// How a text's width is found, in logical pixels at a size: plain words as `measure`
/// estimates them, and a text with mathematics in it as it is set (`set_width`, in a
/// frame) or from the fonts' metrics (`estimate`, in a test, which has no frame).
type Width<'a> = &'a dyn Fn(&str, f32) -> f32;

/// How wide a text is drawn in the picture's type: plain words as `measure` estimates
/// them, a text with mathematics in it as it is set. (Until the fonts are in place, in
/// the first frame, it can only be estimated: `guessed` is set then, and what is laid
/// out from it is not kept.)
fn set_width(f: &Frame, guessed: &std::cell::Cell<bool>, text: &str, size: f32) -> f32 {
    if !crate::typeset::has_math(text) {
        return measure(text, size);
    }
    match crate::typeset::measure(f.ctx, text, size) {
        Some(set) => set.x,
        None => {
            guessed.set(true);
            text_width(text, size)
        }
    }
}

/// The same with no frame at hand: a formula from the fonts' metrics.
#[cfg(test)]
fn estimate(text: &str, size: f32) -> f32 {
    if crate::typeset::has_math(text) {
        text_width(text, size)
    } else {
        measure(text, size)
    }
}

/// The words of a text, as split at its spaces, each formula kept whole (with the
/// words written against it: "$b$'s").
fn words_of(text: &str) -> Vec<&str> {
    let mut words = Vec::new();
    let (mut start, mut inside) = (0, false);
    for (i, c) in text.char_indices() {
        match c {
            '$' => inside = !inside,
            ' ' if !inside => {
                words.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    words.push(&text[start..]);
    words
}

/// Words broken into lines no wider than `width` pixels, each word measured once by
/// `width_of`: each line, and its width.
fn wrap(text: &str, size: f32, width: f32, width_of: Width) -> Vec<(String, f32)> {
    let space = measure(" ", size);
    let mut lines: Vec<(String, f32)> = Vec::new();
    for word in words_of(text) {
        let wide = width_of(word, size);
        match lines.last_mut() {
            Some((last, so_far)) if *so_far + space + wide <= width => {
                last.push(' ');
                last.push_str(word);
                *so_far += space + wide;
            }
            _ => lines.push((word.to_string(), wide)),
        }
    }
    lines
}

/// One line of the panel: its pieces, each in its colour and with its width as drawn,
/// and the size of its type. A line of a walk written out has its first piece (what
/// the walk is: "b =") set right against a column of its own, so that the terms after
/// it do not move when it changes.
#[derive(Clone, Debug)]
struct Line {
    pieces: Vec<(String, Color, f32)>,
    size: f32,
    column: bool,
}

/// The panel's lines, and the width of the walk's first column (as wide as the widest
/// thing it can say, and a gap).
#[derive(Clone, Debug)]
struct Said {
    lines: Vec<Line>,
    column: f32,
}

/// The size of a walk written out in the panel.
const WALK_SIZE: f32 = 15.0;

/// How wide a line of the panel is, the walk's first column `column` wide.
fn line_width(line: &Line, column: f32) -> f32 {
    let widths = line.pieces.iter().map(|p| p.2);
    if line.column {
        column + widths.skip(1).map(|w| w + GAP).sum::<f32>() - GAP
    } else {
        widths.map(|w| w + GAP).sum::<f32>() - GAP
    }
}

/// Room between the pieces of a line, and the panel's padding.
const GAP: f32 = 5.0;
const PAD: f32 = 12.0;
/// The row of buttons at the panel's foot.
const BUTTON_H: f32 = 26.0;
/// Their words and widths, and shorter ones for a narrow panel.
const BUTTONS: [(&str, f32); 2] = [("▶ sweep: paint the span", 172.0), ("↔ slide along the walks", 168.0)];
const SHORT_BUTTONS: [(&str, f32); 2] = [("▶ sweep", 84.0), ("↔ slide the walks", 130.0)];
/// The panel's width on a wide canvas, and the height it is taken to have there when
/// deciding whether it fits and where the picture goes (about the most it takes); and the
/// height it keeps on a narrow one. Not the height it has: that changes with what it
/// says, and the picture is not to move when the words change.
const PANEL_W: f32 = 440.0;
const WIDE_H: f32 = 210.0;
const NARROW_H: f32 = 262.0;

fn line_height(size: f32) -> f32 {
    size + 6.0
}

/// The panel's lines at a width of `width` pixels of text, measured by `width_of`.
#[allow(clippy::too_many_arguments)]
fn panel_lines(an: &Analysis, walk: &[f64; 3], sweeping: bool, sliding: bool, dots: bool, width: f32, width_of: Width) -> Said {
    let mut lines = Vec::new();
    let words = |text: &str, size: f32, color: Color, lines: &mut Vec<Line>| {
        for (l, wide) in wrap(text, size, width, width_of) {
            lines.push(Line { pieces: vec![(l, color, wide)], size, column: false });
        }
    };
    words(&headline(an), 14.0, pal::WHITE, &mut lines);
    words(&rank_line(an), 12.5, pal::TEXT, &mut lines);
    // the walk, its terms in the arrows' colours (broken between terms if it is too long,
    // the rest starting at the same column)
    let size = WALK_SIZE;
    let column = width_of(r"$\text{nearest point} =$", size) + GAP;
    let name = walk_name(an, sweeping);
    let mut row = Line { pieces: vec![(name.to_string(), pal::TEXT, width_of(name, size))], size, column: true };
    for (text, color) in terms(walk, an.count) {
        let piece = (text.clone(), color, width_of(&text, size));
        let mut longer = row.clone();
        longer.pieces.push(piece.clone());
        if line_width(&longer, column) > width && row.pieces.len() > 1 {
            lines.push(std::mem::replace(&mut row, Line { pieces: vec![(String::new(), pal::TEXT, 0.0), piece], size, column: true }));
        } else {
            row = longer;
        }
    }
    lines.push(row);
    words(&about_b(an, walk, sweeping, sliding), 12.5, pal::TEXT, &mut lines);
    if let Some(w) = warning(an, walk, sweeping) {
        words(&w, 12.5, WARN, &mut lines);
    }
    let key = unlabelled(dots, an.many() && !sweeping, !an.reach && !sweeping);
    if !key.is_empty() {
        words(&key, 11.5, pal::DIM, &mut lines);
    }
    Said { lines, column }
}

/// The height the lines and the buttons take, padding and all.
fn panel_height(lines: &[Line]) -> f32 {
    lines.iter().map(|l| line_height(l.size)).sum::<f32>() + 2.0 * PAD + 8.0 + BUTTON_H
}

/// Is the canvas narrow and tall (a phone held upright)? Then the panel goes below the
/// picture instead of over its top-left corner.
fn narrow_canvas(view: Vec2) -> bool {
    view.x < 620.0 && view.y >= 560.0
}

/// The width of the panel's text on a canvas of `view` logical pixels.
fn text_room(view: Vec2) -> f32 {
    let panel = if narrow_canvas(view) { view.x - 16.0 } else { PANEL_W.min(view.x - 28.0) };
    (panel - 2.0 * PAD).max(40.0)
}

/// How a canvas of size `view`, whose top `inset` pixels the application's buttons cover,
/// is shared: the panel of words `height` pixels tall (None if the canvas cannot spare
/// the room), and the part left for the picture.
fn layout(view: Vec2, inset: f32, height: f32) -> (Option<(Vec2, Vec2)>, (Vec2, Vec2)) {
    let whole = (Vec2::new(0.0, inset.min(view.y * 0.5)), view.max(Vec2::ONE));
    if narrow_canvas(view) {
        // (a sliver of canvas, beside controls opened over a phone's screen, has no room
        // for words at all)
        if view.x < 300.0 || NARROW_H + inset > view.y * 0.5 {
            return (None, whole);
        }
        // (the panel's top stays put as its lines come and go; in the rare case of more
        // lines than the height kept for them it reaches further up)
        let top = view.y - 8.0 - NARROW_H.max(height);
        let panel = (Vec2::new(8.0, top), Vec2::new(view.x - 8.0, top + height));
        (Some(panel), (Vec2::new(0.0, inset), Vec2::new(view.x, view.y - 8.0 - NARROW_H)))
    } else {
        let top = inset + 14.0;
        if view.x < 480.0 || top + WIDE_H > view.y * 0.6 {
            return (None, whole);
        }
        let panel = (Vec2::new(14.0, top), Vec2::new(14.0 + PANEL_W.min(view.x - 28.0), top + height));
        // (the panel takes only the left of the top: on a tall canvas the picture may reach
        // up beside it; on a short one it keeps below it)
        let below = if view.y >= 760.0 { top + WIDE_H * 0.5 } else { top + WIDE_H + 6.0 };
        (Some(panel), (Vec2::new(0.0, below), view))
    }
}

/// A button drawn over the picture (logical pixels): true when it is clicked. One that
/// cannot be used now is drawn dim and ignores clicks.
fn hud_button(f: &mut Frame, min: Vec2, max: Vec2, text: &str, usable: bool, pointer: Option<Vec2>) -> bool {
    let over = pointer.is_some_and(|p| p.cmpge(min).all() && p.cmple(max).all());
    let fill = match (usable, over) {
        (false, _) => Color::hex(0x141925),
        (true, true) => Color::hex(0x33405E),
        (true, false) => Color::hex(0x222B40),
    };
    hud_rect(f, min, max, fill);
    f.text_px((min + max) * 0.5, text, 12.5, if usable { pal::TEXT } else { pal::FAINT }, Align2::CENTER_CENTER);
    usable && over && f.ptr.clicked
}

/// Where the buttons go in the panel from `min` to `max`, below lines `lines_height`
/// pixels tall, and what they say (in fewer words if the panel is narrow).
fn button_boxes(min: Vec2, max: Vec2, lines_height: f32) -> [(Vec2, Vec2, &'static str); 2] {
    let y = min.y + PAD + lines_height + 8.0;
    let room = max.x - min.x - 2.0 * PAD;
    let words = if BUTTONS[0].1 + BUTTONS[1].1 + 8.0 <= room { BUTTONS } else { SHORT_BUTTONS };
    let mut x = min.x + PAD;
    words.map(|(text, w)| {
        let b = (Vec2::new(x, y), Vec2::new(x + w, y + BUTTON_H), text);
        x += w + 8.0;
        b
    })
}

// ----- the scene ------------------------------------------------------------------------

/// The span as drawn: the point 0, a line (its direction), a plane (two directions at
/// right angles in it), or all of space (the lattice of the three arrows).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Shape {
    Point,
    Line(Vec3),
    /// Two directions in it, and how far out the sheet goes (it takes in every walk with
    /// numbers between −1 and 1, and then some).
    Plane(Vec3, Vec3, f32),
    Space([Vec3; 3]),
}

fn shape_of(an: &Analysis, arrows: [Vec3; 3]) -> Shape {
    match an.rank {
        0 => Shape::Point,
        1 => Shape::Line(an.q[0].as_vec3()),
        2 => {
            // as far out as the furthest walk with numbers between −1 and 1
            let reach = (0..1usize << an.count).map(|signs| (0..an.count).fold(DVec3::ZERO, |p, i| p + if signs & (1 << i) == 0 { an.v[i] } else { -an.v[i] }).length()).fold(0.0, f64::max);
            Shape::Plane(an.q[0].as_vec3(), an.q[1].as_vec3(), (1.15 * reach as f32).max(3.2))
        }
        _ => Shape::Space(arrows),
    }
}

/// The direction the scene is seen from at first.
const YAW: f32 = -0.95;
const PITCH: f32 = 0.42;
/// Seen from here, the plane of v₁ and v₂ of the experiments is nearly edge on (about
/// 12° above it), so that a cloud of dots in it is seen to be flat, and one that is not is
/// seen to be thick; v₁, v₂ and v₃ still stand apart on the screen.
const ALONG: (f32, f32) = (-0.65, 0.12);

pub struct Span {
    v: [Vec3; 3],
    b: Vec3,
    /// How many of the arrows are in use (two or three).
    count: usize,
    /// The magnet (see `SNAP_DEG`).
    snap: bool,
    /// Where each handle is while it is dragged (v₁, v₂, v₃, b): the tip is that point,
    /// kept within `REACH` and put into the span of the others when the magnet takes it.
    grip: [Vec3; 4],
    /// Where the arrows and b are drawn: gliding to where they are when the magnet has
    /// moved them (the numbers are those of where they are, from the first moment).
    drawn: [Vec3; 4],
    /// A sweep under way: seconds into it.
    sweep: Option<f32>,
    /// The sweep's dots, as the numbers of the walks that left them (so that they move
    /// with the arrows, and stay in their span): the path's places `dot_place(k)`.
    dots: Vec<[f32; 3]>,
    /// How brightly the dots are drawn: fully while they are being left, dimmer once the
    /// sweep is over, so that the arrows and the walk among them stand out again.
    shine: f32,
    /// A slide along the walks to b under way: seconds into it, and where it started.
    slide: Option<(f32, [f32; 2])>,
    /// How far along each walk that comes back to 0 the walk to b is taken.
    t: [f32; 2],
    /// How much of each picture of the span is shown (by its dimension), and the last
    /// one of each that was drawn (what fades out is drawn as it was).
    look: [f32; 4],
    shapes: [Shape; 4],
    /// The points the view is framed on, and the direction they are seen from.
    home: Vec<Vec3>,
    seen: (f32, f32),
    /// The canvas last drawn on, how much of its top the application's buttons cover,
    /// and whether it has been measured yet (until then it is a guess).
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
    /// Where the panel of words was drawn last (handles under it are left alone).
    panel: Option<(Vec2, Vec2)>,
    /// The names in the picture as they were written last, kept while nothing they are
    /// placed from changes (see `Still`).
    kept: Still,
    /// The panel's lines, measured, with the bits of what they were written from (see
    /// `Span::said_for`): written afresh only when one of those changes.
    said: Option<([u32; 23], Said)>,
    /// The sheet of a plane, as last drawn (see `Sheet`).
    sheet: Sheet,
}

impl Span {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let mut s = Span {
            v: [Vec3::from(V1), Vec3::from(V2), Vec3::from(V3_UP)],
            b: Vec3::from(B_IN),
            count: 2,
            snap: true,
            grip: [Vec3::ZERO; 4],
            drawn: [Vec3::ZERO; 4],
            sweep: None,
            dots: Vec::new(),
            shine: 1.0,
            slide: None,
            t: [0.0; 2],
            look: [0.0; 4],
            shapes: [Shape::Point; 4],
            home: Vec::new(),
            seen: (YAW, PITCH),
            canvas: Vec2::new(1070.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
            panel: None,
            kept: Still::default(),
            said: None,
            sheet: Sheet::default(),
        };
        s.set_up(0);
        s.new_view = false;
        s
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        let trial = &TRIALS[k];
        self.count = trial.count;
        self.v = trial.v.map(Vec3::from);
        self.b = Vec3::from(trial.b);
        self.snap = true;
        self.seen = trial.seen.unwrap_or((YAW, PITCH));
        self.restart();
        if trial.sweep {
            self.start_sweep();
        }
        if trial.slide {
            self.start_slide();
        }
        self.frame_on_picture(trial.sweep);
    }

    /// Start afresh from the arrows and b as they are: no dots, no sweep or slide, the
    /// walk to b the first one, and everything drawn where it is.
    fn restart(&mut self) {
        (self.sweep, self.slide, self.t) = (None, None, [0.0; 2]);
        self.dots.clear();
        self.grip = self.places();
        self.drawn = self.places();
        let an = self.analysis();
        self.look = [0.0; 4];
        self.look[an.rank] = 1.0;
        self.shapes[an.rank] = shape_of(&an, self.v);
    }

    /// Where v₁, v₂, v₃ and b are.
    fn places(&self) -> [Vec3; 4] {
        [self.v[0], self.v[1], self.v[2], self.b]
    }

    fn analysis(&self) -> Analysis {
        Analysis::of(self.v, self.count, self.b)
    }

    fn start_sweep(&mut self) {
        self.sweep = Some(0.0);
        self.shine = 1.0;
        self.slide = None;
        self.dots.clear();
    }

    fn start_slide(&mut self) {
        self.slide = Some((0.0, self.t));
    }

    /// The walk to b (or to the point nearest it) as it is now: the one along the kept
    /// arrows, taken along the walks that come back to 0 by `t`.
    fn walk_to_b(&self, an: &Analysis) -> [f64; 3] {
        let mut c = an.base;
        for (n, t) in an.null[..an.nulls.min(2)].iter().zip(self.t) {
            for j in 0..3 {
                c[j] += t as f64 * n[j];
            }
        }
        c
    }

    /// The walk shown: during a sweep, partly or wholly the sweep's own.
    fn walk(&self, an: &Analysis) -> [f64; 3] {
        let to_b = self.walk_to_b(an);
        let Some(tau) = self.sweep else { return to_b };
        let (share, on) = (sweep_share(tau), path(progress(tau as f64)));
        let mut c = [0.0; 3];
        for i in 0..an.count {
            c[i] = to_b[i] * (1.0 - share) + on[i] * share;
        }
        c
    }

    /// Move a sweep and a slide on by `dt` seconds of the scene's time.
    fn advance(&mut self, dt: f32) {
        let dt = dt.max(0.0);
        if let Some(tau) = self.sweep {
            let tau = tau + dt;
            // the dots owed by now, each at its own place on the path
            for k in self.dots.len()..dots_by(tau) {
                let c = path(dot_place(k));
                self.dots.push([c[0] as f32, c[1] as f32, if self.count == 3 { c[2] as f32 } else { 0.0 }]);
            }
            self.sweep = (tau < S_END).then_some(tau);
        } else if self.shine > 0.0 {
            self.shine *= (-dt / DIMMING).exp();
            // (exactly dim in the end: the picture then stops changing)
            if self.shine < 0.004 {
                self.shine = 0.0;
            }
        }
        if let Some((sigma, from)) = self.slide {
            let sigma = sigma + dt;
            if sigma >= SLIDE_TIME {
                // (exactly the first walk at the end, not nearly)
                (self.t, self.slide) = ([0.0; 2], None);
            } else {
                (self.t, self.slide) = (slide_at(sigma, from), Some((sigma, from)));
            }
        }
    }

    /// Where a dragged tip goes when its handle is at `p`: no further out than `REACH`,
    /// and with the magnet on, onto the line of another arrow or into the plane of the
    /// other two if it is within `SNAP_DEG` of it (the line first: a tip near the line
    /// of another is near their plane too).
    fn settle_arrow(&self, k: usize, p: Vec3) -> Vec3 {
        let p = if p.length() > REACH { p * (REACH / p.length()) } else { p };
        if !self.snap {
            return p;
        }
        let pd = p.as_dvec3();
        let others: Vec<DVec3> = (0..self.count).filter(|&j| j != k).map(|j| self.v[j].as_dvec3()).filter(|o| o.length() > TINY).collect();
        let mut best: Option<(f64, DVec3)> = None;
        for o in &others {
            let line = [o.normalize()];
            let off = degrees_off(pd, &line);
            if off < SNAP_DEG && best.is_none_or(|b| off < b.0) {
                best = Some((off, pd - leftover(pd, &line)));
            }
        }
        if best.is_none() && others.len() == 2 {
            let plane = orthonormal(&others, TINY);
            if plane.len() == 2 && degrees_off(pd, &plane) < SNAP_DEG {
                best = Some((0.0, pd - leftover(pd, &plane)));
            }
        }
        best.map_or(p, |b| b.1.as_vec3())
    }

    /// Where b goes when its handle is at `p`: likewise, into the span when it is within
    /// `SNAP_DEG` of it (if the span is not all of space).
    fn settle_b(&self, p: Vec3) -> Vec3 {
        let p = if p.length() > REACH { p * (REACH / p.length()) } else { p };
        let an = self.analysis();
        if !self.snap || an.rank == 0 || an.rank == 3 {
            return p;
        }
        let (pd, q) = (p.as_dvec3(), &an.q[..an.rank]);
        if degrees_off(pd, q) < SNAP_DEG {
            (pd - leftover(pd, q)).as_vec3()
        } else {
            p
        }
    }

    /// Are the arrows and b drawn where they are?
    fn arrived(&self) -> bool {
        self.drawn == self.places()
    }

    /// Bring what is drawn `dt` seconds nearer to where it is; with `at_once`, all the way.
    fn glide(&mut self, dt: f32, at_once: bool) {
        let k = 1.0 - (-dt.max(0.0) / GLIDE).exp();
        let places = self.places();
        for (drawn, to) in self.drawn.iter_mut().zip(places) {
            *drawn += (to - *drawn) * k;
            // (exactly there in the end, not nearly: the picture then stops changing)
            if at_once || !drawn.is_finite() || (to - *drawn).length() < ARRIVED {
                *drawn = to;
            }
        }
    }

    /// Fade the picture of the span `dt` seconds towards the one of this rank.
    fn fade(&mut self, an: &Analysis, dt: f32) {
        self.shapes[an.rank] = shape_of(an, [self.drawn[0], self.drawn[1], self.drawn[2]]);
        let k = 1.0 - (-dt.max(0.0) / FADE).exp();
        for (r, look) in self.look.iter_mut().enumerate() {
            let goal = if r == an.rank { 1.0 } else { 0.0 };
            *look += (goal - *look) * k;
            if (goal - *look).abs() < 0.004 {
                *look = goal;
            }
        }
    }

    fn faded(&self) -> bool {
        self.look.iter().all(|l| *l == 0.0 || *l == 1.0)
    }

    /// Have the view take in the picture as it now is (and, for a sweep, the box of walks
    /// with numbers between −1 and 1 that it paints).
    fn frame_on_picture(&mut self, sweep: bool) {
        let an = self.analysis();
        let mut points = vec![Vec3::ZERO, self.b, an.p.as_vec3()];
        points.extend(&self.v[..self.count]);
        let walk = self.walk_to_b(&an);
        let mut at = DVec3::ZERO;
        for i in 0..an.count {
            at += an.v[i] * walk[i];
            // (a walk that goes far out, along nearly dependent arrows, is not framed)
            if at.length() < 8.0 {
                points.push(at.as_vec3());
            }
        }
        if an.many() {
            for t in GHOSTS {
                let c = ghost_walk(self, &an, t);
                let mut at = DVec3::ZERO;
                for i in 0..an.count {
                    at += an.v[i] * c[i];
                    if at.length() < 8.0 {
                        points.push(at.as_vec3());
                    }
                }
            }
        }
        if sweep {
            for corner in 0..(1 << self.count) {
                let c: Vec<f32> = (0..self.count).map(|i| if corner & (1 << i) == 0 { -1.0 } else { 1.0 }).collect();
                points.push((0..self.count).fold(Vec3::ZERO, |p, i| p + self.v[i] * c[i]));
            }
        }
        self.home = points.into_iter().filter(|p| p.is_finite()).collect();
        self.new_view = true;
    }

    /// The view that fits the picture (`home`) into the part of the canvas the panel
    /// leaves free, and no larger than reads well.
    fn the_view(&self) -> View {
        let (yaw, pitch) = self.seen;
        let right = Vec3::new(-yaw.sin(), yaw.cos(), 0.0);
        let up = Vec3::new(-pitch.sin() * yaw.cos(), -pitch.sin() * yaw.sin(), pitch.cos());
        let eye = right.cross(up);
        // (the picture's part of the canvas does not depend on the panel's height)
        let (_, (min, max)) = layout(self.canvas, self.inset, 0.0);
        // (room round the edge for the names: less on a phone, where every pixel counts)
        let margin = if narrow_canvas(self.canvas) { Vec2::new(50.0, 60.0) } else { Vec2::new(110.0, 80.0) };
        let room = (max - min - margin).max(Vec2::splat(40.0));
        let goal = (min + max) * 0.5;
        // pixels on the screen per unit of length across the line of sight, at unit distance
        let focal = self.canvas.y.max(1.0) * 0.5 / (0.31f32).tan();
        let points: &[Vec3] = if self.home.is_empty() { &[Vec3::ZERO] } else { &self.home };
        let reach = points.iter().map(|p| p.length()).fold(0.5, f32::max);
        let mut target = points.iter().copied().sum::<Vec3>() / points.len() as f32;
        let mut dist = 8.0f32;
        // seen in perspective, the near side looks larger than the far: fitted by looking
        // at the picture, moving the camera, and looking again
        for _ in 0..12 {
            let from = target + eye * dist;
            let seen: Vec<Vec2> = points
                .iter()
                .map(|p| {
                    let d = *p - from;
                    let depth = (-d.dot(eye)).max(1e-3);
                    self.canvas * 0.5 + Vec2::new(d.dot(right), -d.dot(up)) * (focal / depth)
                })
                .collect();
            let lo = seen.iter().fold(Vec2::MAX, |m, p| m.min(*p));
            let hi = seen.iter().fold(Vec2::MIN, |m, p| m.max(*p));
            let size = (hi - lo).max(Vec2::ONE);
            let scale = (room.x / size.x).min(room.y / size.y).min(560.0 / size.max_element());
            // move the middle of the picture to the middle of the free part, then come
            // nearer or go further by the scale wanted
            let shift = ((lo + hi) * 0.5 - goal) * (dist / focal);
            target += right * shift.x - up * shift.y;
            // (max then min, not clamp: a reach beyond 40 would put the bounds the wrong
            // way round, and clamp would panic)
            dist = (dist / scale).max(2.0 * reach).min(80.0);
        }
        if !(target.is_finite() && dist.is_finite()) {
            return View::Orbit { target: Vec3::new(0.6, 0.7, 0.6), yaw, pitch, dist: 9.0 };
        }
        View::Orbit { target, yaw, pitch, dist }
    }

    /// The bits of everything the panel's lines are written from, `panel_lines`'
    /// arguments: the arrows and b (which the analysis is made from), the walk shown,
    /// what is under way and the width of the text.
    fn said_for(&self, walk: &[f64; 3], room: f32) -> [u32; 23] {
        let mut out = [0; 23];
        let words = self.v.iter().chain([&self.b]).flat_map(|v| v.to_array()).map(f32::to_bits);
        let words = words.chain(walk.iter().flat_map(|c| [c.to_bits() as u32, (c.to_bits() >> 32) as u32]));
        let words = words.chain([self.count as u32, self.sweep.is_some() as u32, self.slide.is_some() as u32, self.dots.is_empty() as u32, room.to_bits()]);
        for (o, w) in out.iter_mut().zip(words) {
            *o = w;
        }
        out
    }

    /// Everything the picture and its names are drawn from (see `Still`). (Not how
    /// brightly the dots shine, which no name hangs on: the names keep still while they dim.)
    fn inputs(&self, k: &mut Inputs) {
        for v in self.v.iter().chain([&self.b]).chain(&self.drawn).chain(&self.grip) {
            k.f32s(&v.to_array());
        }
        k.int(self.count as u64).int(self.snap as u64).int(self.dots.len() as u64).f32s(&self.t).f32s(&self.look);
        match self.sweep {
            None => k.int(0),
            Some(tau) => k.int(1).f32s(&[tau]),
        };
        match self.slide {
            None => k.int(0),
            Some((sigma, from)) => k.int(1).f32s(&[sigma, from[0], from[1]]),
        };
        for shape in &self.shapes {
            match *shape {
                Shape::Point => k.int(0),
                Shape::Line(d) => k.int(1).f32s(&d.to_array()),
                Shape::Plane(e1, e2, half) => k.int(2).f32s(&e1.to_array()).f32s(&e2.to_array()).f32s(&[half]),
                Shape::Space(a) => k.int(3).f32s(&a[0].to_array()).f32s(&a[1].to_array()).f32s(&a[2].to_array()),
            };
        }
        match self.panel {
            None => k.int(0),
            Some((lo, hi)) => k.int(1).f32s(&[lo.x, lo.y, hi.x, hi.y]),
        };
        k.f32s(&[self.canvas.x, self.canvas.y, self.inset]).int(self.measured as u64);
    }

    /// The number of arrows changed: start afresh with that many.
    fn set_count(&mut self, count: usize) {
        self.count = count.clamp(2, 3);
        self.restart();
        self.frame_on_picture(false);
    }
}

impl Scene for Span {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.07;
    }

    /// A sweep and a slide move by themselves, and so does what is gliding or fading.
    fn animated(&self) -> bool {
        self.sweep.is_some() || self.slide.is_some() || !self.arrived() || !self.faded() || (self.shine > 0.0 && !self.dots.is_empty())
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"A linear combination $c_1 v_1 + c_2 v_2 + c_3 v_3$ is a walk: $c_1$ times along $v_1$, then $c_2$ times along $v_2$, then $c_3$ times along $v_3$ (a negative number walks backwards). The span of the arrows is every point such walks can end at: a line, a plane through $0$, or all of space. Drag the tips of the arrows and the point $b$; the words over the picture say what the arrows span and how $b$ is reached.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.name, trial.caption) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Arrows");
        let mut count = self.count - 2;
        if ui::choice(ui, &["two arrows", "three arrows"], &mut count) {
            self.set_count(count + 2);
        }
        ui.horizontal_wrapped(|ui| {
            if ui.button("▶ sweep: paint the span").clicked() {
                self.start_sweep();
                self.frame_on_picture(true);
            }
            let an = self.analysis();
            if ui.add_enabled(an.many() && self.sweep.is_none(), egui::Button::new("↔ slide along the walks")).clicked() {
                self.start_slide();
            }
        });
        ui::note(ui, "The sweep runs the numbers between $-1$ and $1$ and leaves a dot where each walk ends. The slide is for arrows that are dependent, with $b$ in their span: it goes through the many walks that reach $b$.");
        let an = self.analysis();
        ui.add_enabled_ui(an.nulls > 0, |ui| {
            if ui::slider(ui, "which walk ($t$)", &mut self.t[0], -1.5..=1.5) {
                self.slide = None;
            }
        });
        ui::note(ui, "For dependent arrows: $t$ steps along the walk that comes back to $0$, added to the walk to $b$. Its end does not move.");
        ui.checkbox(&mut self.snap, "magnet");
        ui::note(ui, "An arrow dragged to within 3° of the line of another, or of the plane of the other two, is put exactly into it; and $b$, dragged that near to the span, into the span. Dependence can then be reached by hand, not only come near.");

        ui::section(ui, "Reading the picture");
        let mut key: Vec<(Color, &str)> =
            vec![(COLS[0], r"$\textcolor{green}{v_1}$ (green), $\textcolor{orange}{v_2}$ (orange), $\textcolor{pink}{v_3}$ (pink): the arrows; drag their tips"), (B_COL, "$b$: the point to reach; drag it")];
        key.push((SPAN_COL, "the span, in blue: a line, a see-through sheet for a plane, or a slanted lattice of the arrows' steps for all of space"));
        key.push((COLS[1], "the walk: legs along $v_1$, $v_2$, $v_3$ in their colours, tip to tail, each with its number of steps"));
        key.push((END_COL, "cyan: where walks end. The dots a sweep leaves; and the nearest point, when $b$ cannot be reached"));
        key.push((pal::WHITE, "dashed white, with a corner: from the nearest point to $b$, at a right angle to the span"));
        key.push((GHOST, "faint walks: other walks that end at $b$ too"));
        key.push((WARN, "yellow words: the arrows are nearly dependent"));
        key.push((pal::FAINT, "thin grey lines: the $x$, $y$ and $z$ axes"));
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, "Linear combination:", r"$c_1 v_1 + c_2 v_2 + c_3 v_3$ for some numbers $c_1, c_2, c_3$: the end of the walk along the arrows by those numbers.");
        ui::term(ui, "Span:", "every point a linear combination of the arrows can reach. It always contains $0$ (all the numbers $0$).");
        ui::term(ui, "Independent:", "no arrow is a combination of the others. Then every point of the span is reached by exactly one walk.");
        ui::term(ui, "Dependent:", "one arrow is a combination of the others (an arrow of no length counts). It adds nothing to the span, and every point of the span is reached by many walks.");
        ui::term(ui, "Basis:", "independent arrows that span the space in question: two for a plane, three for all of space.");
        ui::term(ui, "Coordinates:", "the numbers of the one walk to a point along a basis.");
        ui::term(ui, "Rank:", "how many independent directions the arrows give: the dimension of the span (1 for a line, 2 for a plane, 3 for space).");

        ui::section(ui, "Numbers");
        let walk = self.walk(&an);
        for i in 0..self.count {
            let v = self.v[i];
            ui::readout_colored(ui, LABELS[i], format!("${}$", tuple(&[v.x, v.y, v.z], 2)), COLS[i]);
        }
        ui::readout_colored(ui, "$b$", format!("${}$", tuple(&[self.b.x, self.b.y, self.b.z], 2)), B_COL);
        ui::readout(ui, "rank", an.rank.to_string());
        let numbers: Vec<String> = walk[..self.count].iter().map(|c| amount(*c)).collect();
        ui::readout(ui, if self.sweep.is_some() { "the sweep's walk" } else { "the walk shown" }, format!("$({})$", numbers.join(", ")));
        ui::readout(ui, "$b$'s distance from the span", if an.reach { "0: $b$ is in it".to_string() } else { length_text(an.miss) });
        if let Some((j, deg)) = an.tightest {
            let what = if self.count == 2 { "angle between the lines of $v_1$ and $v_2$".to_string() } else { format!("{} out of the plane of the other two", LABELS[j]) };
            ui::readout(ui, &what, degrees_text(deg));
        }
        ui::note(ui, "An arrow counts as in the span of the others when it is less than a hundred-thousandth of the longest arrow's length away from it; $b$, when it is that close, or a hundred-thousandth of its own length if $b$ is longer (and never less than a millionth). The coordinates are kept to about seven digits, so an arrow put exactly into a plane can end up a ten-millionth of its length off it by rounding alone.");
        ui::see_also(ui, &[("vectors", "arrows, their lengths and shadows"), ("basis", "coordinates in another basis"), ("subspaces", "the span of a matrix's columns, and what it sends to 0"), ("lsq", "the nearest point of a plane: least squares")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the canvas decides where the panel goes, and with that where the picture is
        let view = f.view_size();
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        if !covered && (!self.measured || (view - self.canvas).abs().max_element() > 24.0 || f.top_inset != self.inset) {
            (self.canvas, self.inset, self.measured) = (view, f.top_inset, true);
            self.new_view = true;
        }
        self.advance(f.dt);

        let pointer = f.ptr.pos.map(|p| p / f.px);
        let under_panel = |p: Vec2, panel: Option<(Vec2, Vec2)>| panel.is_some_and(|(lo, hi)| p.cmpge(lo).all() && p.cmple(hi).all());
        if pointer.is_some_and(|p| under_panel(p, self.panel)) {
            // the panel is for its buttons: no turning the view from there, and two
            // quick clicks are two presses, not a reset of the view
            f.handles.capture();
            f.handles.claim_clicks();
        }

        // the handles first, so that everything below is drawn where they now are (one
        // under the panel is left alone, unless it is being dragged already)
        for k in (0..4).filter(|&k| k == 3 || k < self.count) {
            let hidden = on_screen(f, self.grip[k]).is_some_and(|p| under_panel(p, self.panel));
            if hidden && !f.handles.dragging() {
                continue;
            }
            let color = if k == 3 { B_COL } else { COLS[k] };
            let mut grip = self.grip[k];
            if f.drag3(1 + k as u32, &mut grip, None, color) {
                self.grip[k] = grip;
                if k == 3 {
                    self.b = self.settle_b(grip);
                } else {
                    self.v[k] = self.settle_arrow(k, grip);
                }
            }
        }
        if !f.handles.dragging() {
            self.grip = self.places();
        }
        // What has been moved by the magnet glides there. Not while the scene's time
        // stands still (nothing glides then, and a drag must still be seen). By the
        // clock, as the view's own glide is: it answers the hand.
        self.glide(f.real_dt, f.dt <= 0.0);
        let an = self.analysis();
        self.fade(&an, if f.dt <= 0.0 { 1e3 } else { f.real_dt });
        let walk = self.walk(&an);

        // the panel's words, and with them its place
        let room = text_room(view);
        let said_for = self.said_for(&walk, room);
        let guessed = std::cell::Cell::new(false);
        let said = self.said.take().filter(|(was, _)| *was == said_for && keeps());
        let said = said.unwrap_or_else(|| {
            let width_of = |text: &str, size: f32| set_width(f, &guessed, text, size);
            (said_for, panel_lines(&an, &walk, self.sweep.is_some(), self.slide.is_some(), !self.dots.is_empty(), room, &width_of))
        });
        let (lines, column) = (&said.1.lines, said.1.column);
        let lines_height: f32 = lines.iter().map(|l| line_height(l.size)).sum();
        let (panel, _) = layout(view, f.top_inset, panel_height(lines));
        self.panel = panel;

        let (mut kept, mut sheet) = (std::mem::take(&mut self.kept), std::mem::take(&mut self.sheet));
        draw(f, self, (&an, &walk), panel, (&mut kept, &mut sheet));
        (self.kept, self.sheet) = (kept, sheet);

        // the panel, over the picture (its labels last: as its lines come and go, those
        // over the picture keep their places in the list of labels, which is how `eidos
        // motion` tells one from another)
        if let Some((min, max)) = panel {
            hud_panel(f, min, max);
            let mut y = min.y + PAD;
            for line in lines {
                let h = line_height(line.size);
                let mut x = min.x + PAD;
                for (k, (text, color, wide)) in line.pieces.iter().enumerate() {
                    if line.column && k == 0 {
                        f.text_px(Vec2::new(x + column - GAP, y + h * 0.5), text.as_str(), line.size, *color, Align2::RIGHT_CENTER);
                        x += column;
                    } else {
                        f.text_px(Vec2::new(x, y + h * 0.5), text.as_str(), line.size, *color, Align2::LEFT_CENTER);
                        x += wide + GAP;
                    }
                }
                y += h;
            }
            let [sweep, slide] = button_boxes(min, max, lines_height);
            if hud_button(f, sweep.0, sweep.1, sweep.2, true, pointer) {
                self.start_sweep();
                self.frame_on_picture(true);
            }
            if hud_button(f, slide.0, slide.1, slide.2, an.many() && self.sweep.is_none(), pointer) {
                self.start_slide();
            }
        } else if view.x >= 300.0 && view.y >= 240.0 {
            // no room for the panel (a phone held sideways): the verdict alone, at the foot
            let short = format!("Rank {}: the span is {}.", an.rank, ["the point $0$ alone", "a line", "a plane", "all of space"][an.rank.min(3)]);
            let text = [headline(&an), short].into_iter().find(|t| set_width(f, &guessed, t, 13.0) <= view.x - 40.0).unwrap_or_default();
            f.caption_px(Vec2::new(18.0, view.y - 14.0), text, 13.0, pal::TEXT, Align2::LEFT_BOTTOM);
        }
        // (nothing measured before the fonts were in place is kept)
        self.said = (!guessed.get()).then_some(said);
    }
}

/// The whole picture in space.
fn draw(f: &mut Frame, s: &Span, (an, walk): (&Analysis, &[f64; 3]), panel: Option<(Vec2, Vec2)>, (kept, sheet): (&mut Still, &mut Sheet)) {
    let mut room = Room::new(f);
    if let Some((min, max)) = panel {
        room.wall(min, max);
    }
    let wpp = |f: &Frame, at: Vec3| f.cam.world_per_px(at) * f.px;
    let arrows = [s.drawn[0], s.drawn[1], s.drawn[2]];
    let b = s.drawn[3];
    let count = s.count;

    // the axes, faint, for the coordinates the panel gives
    let axis = 3.0;
    for d in Vec3::AXES {
        f.line(-d * axis, d * axis, 1.0, pal::FAINT.a(0.8));
        room.line(f, -d * axis, d * axis);
    }

    // the span: every picture of it that is still (or already) shown
    for (r, &look) in s.look.iter().enumerate() {
        if look <= 0.0 {
            continue;
        }
        match s.shapes[r] {
            Shape::Point => {
                f.dot(Vec3::ZERO, 11.0, SPAN_COL.a(look));
            }
            Shape::Line(d) => {
                let long = 6.0;
                f.line(-d * long, d * long, 3.0, SPAN_COL.a(0.8 * look));
                room.line(f, -d * long, d * long);
            }
            Shape::Plane(e1, e2, half) => soft_sheet(f, sheet, e1, e2, half, look),
            Shape::Space(a) => lattice(f, a, look),
        }
    }

    // the dots a sweep has left: the ends of walks, carried along by the arrows
    // (in a batch of their own, without the dark rim a disc has: they are too small for one)
    let end_of = |c: [f32; 3]| (0..count).fold(Vec3::ZERO, |p, i| p + arrows[i] * c[i]);
    let dots = f.dl.points_with(PointKind::Disc, PointStyle { outline_px: 0.0, intensity: 1.2, ..Default::default() });
    for c in &s.dots {
        dots.push(end_of(*c), 3.4, END_COL.a(0.32 + 0.4 * s.shine));
    }

    // other walks to b, faint, each leg in its arrow's colour
    let ghosts = an.many() && s.sweep.is_none();
    if ghosts {
        for t in GHOSTS {
            let c = ghost_walk(s, an, t);
            let mut at = Vec3::ZERO;
            for i in 0..count {
                let next = at + arrows[i] * c[i] as f32;
                if (next - at).length() > 1e-6 {
                    f.line(at, next, 1.6, COLS[i].a(0.4));
                    room.line(f, at, next);
                }
                if i + 1 < count {
                    f.dot(next, 5.0, COLS[i].mix(GHOST, 0.5).a(0.7));
                }
                at = next;
            }
        }
    }

    // the walk itself, tip to tail, each leg in its arrow's colour
    let mut corners = [Vec3::ZERO; 4];
    for i in 0..count {
        corners[i + 1] = corners[i] + arrows[i] * walk[i] as f32;
    }
    for i in 0..count {
        let (from, to) = (corners[i], corners[i + 1]);
        if (to - from).length() > 1e-6 {
            f.line(from, to, 8.0, COLS[i].a(0.22));
            f.arrow(from, to, 2.4, COLS[i].mix(pal::WHITE, 0.2));
            room.line(f, from, to);
        }
    }
    let end = corners[count];
    for p in &corners[1..count] {
        f.dot(*p, 6.0, pal::WHITE.a(0.8));
    }

    // a b out of reach: the nearest point, and the miss at a right angle
    let missed = !an.reach && s.sweep.is_none();
    if missed {
        f.dashed(end, b, 1.8, pal::WHITE.a(0.85));
        room.line(f, end, b);
        let size = 13.0 * wpp(f, end);
        if (b - end).length() > 3.0 * size {
            let along = match an.rank {
                0 => Vec3::X,
                _ if end.length() > 2.0 * size => -end,
                _ => an.q[0].as_vec3(),
            };
            right_angle(f, end, along, b - end, size, pal::WHITE.a(0.9));
        }
        f.dot(end, 10.0, END_COL);
    } else if s.sweep.is_some() {
        f.dot(end, 9.0, END_COL);
    }

    // the arrows themselves
    for i in 0..count {
        f.arrow(Vec3::ZERO, arrows[i], 3.4, COLS[i]);
        room.line(f, Vec3::ZERO, arrows[i]);
    }

    // Names and numbers, the most important first. Every frame writes the same labels in
    // the same order: one with nothing to say now is written invisibly where it was last
    // put, so that each keeps its place in the list of labels and on the screen. They are
    // written as they were in the last frame if nothing they are placed from has changed
    // since, and they had come to rest (see `Still`).
    s.inputs(kept.begin(f));
    if kept.settle(f) {
        kept.replay(f, &mut room);
        return;
    }
    let outwards = |f: &Frame, v: Vec3| match (on_screen(f, Vec3::ZERO), on_screen(f, v)) {
        (Some(o), Some(p)) if v.length() > 1e-6 => p - o,
        _ => Vec2::new(1.0, -1.0),
    };
    // (a name is left out where it has no room: under the panel, say; and `label_any`
    // writes nothing when none of its points is on the screen)
    #[allow(clippy::too_many_arguments)]
    fn named((room, kept): (&mut Room, &mut Still), f: &mut Frame, name: (&'static str, usize), points: &[Vec3], text: &str, size: f32, color: Color, towards: Vec2, gap: f32) {
        if !kept.label_any(room, f, name, points, text, size, color, towards, gap, true) {
            hidden(f, kept, name, points.first().copied().unwrap_or(Vec3::ZERO));
        }
    }
    // (and the arrows' names and b's, which matter most, wherever they can go, unless
    // what they name is under the panel)
    let under = |f: &Frame, p: Vec3| on_screen(f, p).is_some_and(|p| panel.is_some_and(|(lo, hi)| p.cmpge(lo).all() && p.cmple(hi).all()));
    #[allow(clippy::too_many_arguments)]
    fn must((room, kept): (&mut Room, &mut Still), f: &mut Frame, name: (&'static str, usize), at: Vec3, hide: bool, text: &str, size: f32, color: Color, towards: Vec2, gap: f32) {
        if hide || !kept.label_any(room, f, name, &[at], text, size, color, towards, gap, false) {
            hidden(f, kept, name, at);
        }
    }
    for i in 0..count {
        let dir = outwards(f, arrows[i]);
        let hide = under(f, arrows[i]);
        must((&mut room, kept), f, ("arrow", i), arrows[i], hide, LABELS[i], 17.0, COLS[i], dir, 9.0);
    }
    let hide = under(f, b);
    must((&mut room, kept), f, ("b", 0), b, hide, "$b$", 17.0, B_COL, Vec2::new(1.0, -1.0), 10.0);
    // each leg's number, beside its middle
    for i in 0..count {
        let (from, to) = (corners[i], corners[i + 1]);
        let ends = (on_screen(f, from), on_screen(f, to));
        let (Some(p), Some(q)) = ends else {
            hidden(f, kept, ("leg", i), (from + to) * 0.5);
            continue;
        };
        if (to - from).length() < 1e-6 {
            hidden(f, kept, ("leg", i), (from + to) * 0.5);
            continue;
        }
        let across = (q - p).perp();
        let side = if across.y > 0.0 { across } else { -across };
        let text = format!("${}{}$", amount(walk[i]), NAMES[i]);
        let along = [0.5, 0.35, 0.65, 0.2, 0.8].map(|t| from + (to - from) * t);
        let colour = COLS[i].mix(pal::WHITE, 0.35);
        if under(f, along[0]) || !kept.label_any(&mut room, f, ("leg", i), &along, &text, 13.0, colour, side, 8.0, true) {
            // (where it is crowded, beside the middle of a long leg all the same: the
            // numbers matter; a short one's would only cover another's)
            let long = (q - p).length() > 60.0;
            must((&mut room, kept), f, ("leg", i), along[0], !long || under(f, along[0]), &text, 13.0, colour, side, 8.0);
        }
    }
    if missed {
        let dir = outwards(f, end);
        named((&mut room, kept), f, ("nearest", 0), &[end], "nearest point", 13.0, END_COL, if dir == Vec2::ZERO { Vec2::new(0.0, 1.0) } else { -dir.perp() }, 9.0);
        let along = match (on_screen(f, end), on_screen(f, b)) {
            (Some(p), Some(q)) => (q - p).perp(),
            _ => Vec2::X,
        };
        named((&mut room, kept), f, ("miss", 0), &[(end + b) * 0.5], &format!("misses by {}", length_text(an.miss)), 13.0, pal::TEXT, along, 8.0);
    } else {
        hidden(f, kept, ("nearest", 0), end);
        hidden(f, kept, ("miss", 0), (end + b) * 0.5);
    }
    // the span's name: on the sheet or among the lattice, or beside the line
    let (at, text, towards, gap) = match s.shapes[an.rank] {
        Shape::Point => (Vec3::ZERO, "span: the point $0$ alone", Vec2::new(0.0, 1.0), 12.0),
        Shape::Line(d) => {
            let at = if on_screen(f, d * 3.4).is_some() { d * 3.4 } else { -d * 3.4 };
            let across = outwards(f, at).perp();
            (at, "span: a line", if across.y > 0.0 { across } else { -across }, 10.0)
        }
        Shape::Plane(e1, e2, half) => ((e2 * 0.85 - e1 * 0.6) * half * 0.8, "span: a plane", Vec2::ZERO, 0.0),
        Shape::Space(a) => (a[0] + a[1] + a[2], "span: all of space", Vec2::new(1.0, -1.0), 8.0),
    };
    named((&mut room, kept), f, ("span", 0), &[at], text, 13.0, SPAN_COL.mix(pal::WHITE, 0.45), towards, gap);
    // (the axes' names at their ends, not placed by `room`: always there, and dim; none
    // where it would fall on the panel)
    for (k, name) in ["$x$", "$y$", "$z$"].into_iter().enumerate() {
        let d = Vec3::AXES[k] * axis;
        let seen = on_screen(f, d);
        let p = seen.unwrap_or(Vec2::ZERO);
        let off = outwards(f, d).normalize_or_zero() * 9.0;
        let shown = seen.is_some() && !panel.is_some_and(|(lo, hi)| p.cmpge(lo - Vec2::splat(12.0)).all() && p.cmple(hi + Vec2::splat(12.0)).all());
        kept.text_px(f, p + off, name, 12.0, pal::DIM.a(if shown { 1.0 } else { 0.0 }), Align2::CENTER_CENTER);
    }
    kept.end(f, false);
}

/// A label with nothing to show now, written invisibly where it was last put beside the
/// point `at` (or on it): it keeps its place in the list of labels, and on the screen for
/// when it has something to show again.
fn hidden(f: &mut Frame, kept: &mut Still, name: (&'static str, usize), at: Vec3) {
    let number = name.1 as u32;
    match on_screen(f, at) {
        Some(from) => {
            let offset = f.places.chosen(name.0, number).unwrap_or(Vec2::ZERO);
            kept.text_placed(f, name.0, number, from, offset, "", 1.0, pal::TEXT.a(0.0));
        }
        None => kept.text_px(f, Vec2::ZERO, "", 1.0, pal::TEXT.a(0.0), Align2::LEFT_TOP),
    }
}

/// The see-through sheet of a plane as last drawn (its pieces and their colours), with
/// the bits of what it was drawn from: the plane's directions, how far it reaches and
/// how strongly it is shown. Worked out again only when one of those changes, which a
/// sweep or a slide does not do. (It is taken down only once the plane has stayed the
/// same for a frame: while an arrow is dragged it changes in every frame, and taking it
/// down then would only cost time.)
#[derive(Default)]
struct Sheet {
    made_for: Option<[u32; 8]>,
    last: Option<[u32; 8]>,
    quads: Vec<([Vec3; 4], [Color; 4])>,
    lines: Vec<(Vec3, Vec3, Color)>,
}

/// A plane through 0, as a see-through sheet with a square grid on it (its sides along
/// the unit directions `e1` and `e2`) that fades away towards its edge: it goes on.
fn soft_sheet(f: &mut Frame, sheet: &mut Sheet, e1: Vec3, e2: Vec3, half: f32, look: f32) {
    let made_for = [e1.x, e1.y, e1.z, e2.x, e2.y, e2.z, half, look].map(f32::to_bits);
    if sheet.made_for == Some(made_for) && keeps() {
        for (p, colors) in &sheet.quads {
            f.dl.tris_alpha.quad_grad(*p, *colors);
        }
        for &(a, b, color) in &sheet.lines {
            f.line(a, b, 1.0, color);
        }
        return;
    }
    let take_down = sheet.last == Some(made_for);
    sheet.last = Some(made_for);
    sheet.made_for = take_down.then_some(made_for);
    sheet.quads.clear();
    sheet.lines.clear();
    let cells = 16;
    // fully there out to `clear`, gone at `half`
    let clear = 0.6 * half;
    let fade = |a: f32, c: f32| 1.0 - smoothstep(((a * a + c * c).sqrt() - clear) / (half - clear));
    let at = |a: f32, c: f32| e1 * a + e2 * c;
    let step = 2.0 * half / cells as f32;
    for i in 0..cells {
        for j in 0..cells {
            let (a0, c0) = (-half + i as f32 * step, -half + j as f32 * step);
            let corners = [(a0, c0), (a0 + step, c0), (a0 + step, c0 + step), (a0, c0 + step)];
            let colors = corners.map(|(a, c)| SPAN_COL.a(0.08 * look * fade(a, c)));
            let p = corners.map(|(a, c)| at(a, c));
            f.dl.tris_alpha.quad_grad(p, colors);
            if take_down {
                sheet.quads.push((p, colors));
            }
        }
    }
    // the grid lines, a unit apart, each in short pieces that fade with the sheet
    let pieces = 24;
    let lines = half.floor() as i32 - 1;
    for k in -lines..=lines {
        let k = k as f32;
        for n in 0..pieces {
            let (s0, s1) = (-half + 2.0 * half * n as f32 / pieces as f32, -half + 2.0 * half * (n + 1) as f32 / pieces as f32);
            let alpha = 0.45 * look * fade(k, (s0 + s1) * 0.5);
            if alpha > 0.003 {
                for (a, b) in [(at(k, s0), at(k, s1)), (at(s0, k), at(s1, k))] {
                    f.line(a, b, 1.0, SPAN_COL.a(alpha));
                    if take_down {
                        sheet.lines.push((a, b, SPAN_COL.a(alpha)));
                    }
                }
            }
        }
    }
}

/// The other walk to b drawn faint at `t` along the first walk that comes back to 0
/// (whatever the walk shown is).
fn ghost_walk(s: &Span, an: &Analysis, t: f32) -> [f64; 3] {
    let mut c = s.walk_to_b(an);
    for j in 0..3 {
        c[j] += (t - s.t[0]) as f64 * an.null[0][j];
    }
    c
}

/// All of space, drawn as the lattice of the three arrows: the walks of whole numbers of
/// steps from −1 to 1 along each, and the lines between them, faint.
fn lattice(f: &mut Frame, a: [Vec3; 3], look: f32) {
    let steps = [-1.0f32, 0.0, 1.0];
    for i in 0..3 {
        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
        for sj in steps {
            for sk in steps {
                let base = a[j] * sj + a[k] * sk;
                let middle = sj == 0.0 && sk == 0.0;
                f.line(base - a[i], base + a[i], if middle { 1.4 } else { 1.0 }, SPAN_COL.a((if middle { 0.5 } else { 0.28 }) * look));
            }
        }
    }
    for x in steps {
        for y in steps {
            for z in steps {
                f.dot(a[0] * x + a[1] * y + a[2] * z, 5.0, SPAN_COL.a(0.6 * look));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, played to rest at 60 frames a second.
    fn run(k: usize) -> Span {
        let mut s = Span::create();
        s.set_up(k);
        for _ in 0..60 * 20 {
            s.advance(1.0 / 60.0);
        }
        assert!(s.sweep.is_none() && s.slide.is_none() && s.shine == 0.0);
        s
    }

    fn d(v: [f32; 3]) -> DVec3 {
        Vec3::from(v).as_dvec3()
    }

    /// The unit normal of the plane of the experiments' v₁ and v₂.
    fn normal() -> DVec3 {
        d(V1).cross(d(V2)).normalize()
    }

    /// Where the scene's dots are, as it draws them (at rest, the arrows drawn are the arrows).
    fn dot_points(s: &Span) -> Vec<DVec3> {
        s.dots.iter().map(|c| (0..s.count).fold(DVec3::ZERO, |p, i| p + s.v[i].as_dvec3() * c[i] as f64)).collect()
    }

    /// The numbers of every dot left, one list per arrow.
    fn dot_numbers(s: &Span, i: usize) -> Vec<f32> {
        s.dots.iter().map(|c| c[i]).collect()
    }

    fn spread(xs: &[f32]) -> (f32, f32) {
        (xs.iter().copied().fold(f32::MAX, f32::min), xs.iter().copied().fold(f32::MIN, f32::max))
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        let n = normal();

        // two arrows paint a plane: the numbers run between -1 and 1, slowly and then
        // fast, every dot lies in one flat sheet through 0, and the walk settles on b
        // = 1.00 v1 + 0.50 v2, the only walk there
        let s = run(0);
        let an = s.analysis();
        assert!(s.count == 2 && an.rank == 2 && an.independent() && an.reach && an.nulls == 0 && !an.many());
        let dots = dot_points(&s);
        assert!(dots.len() > 2000 && dots.iter().all(|p| p.dot(n).abs() < 1e-6), "{} dots", dots.len());
        for i in 0..2 {
            let (lo, hi) = spread(&dot_numbers(&s, i));
            assert!(lo >= -1.0 && hi <= 1.0 && lo < -0.99 && hi > 0.99, "c{i} from {lo} to {hi}");
        }
        let speed = |tau: f64| (progress(tau + 1e-4) - progress(tau)) / 1e-4;
        // (eight times as fast in the middle as at the ends)
        assert!(speed(0.5) * 8.0 < speed(5.0) && speed(S_PAINT as f64 + 0.5) * 8.0 < speed(5.0));
        let walk = s.walk(&an);
        assert_eq!(terms_text(&walk, 2), "$1.00v_1 + 0.50v_2$");
        assert_eq!(walk_name(&an, false), "$b =$");
        assert_eq!(headline(&an), "Two independent arrows: every point of a plane, one walk each.");
        assert_eq!(about_b(&an, &walk, false, false), "One walk reaches $b$: $(1.00, 0.50)$ are its coordinates in the basis $v_1, v_2$ of this plane.");

        // a third arrow in the plane: v3 = -0.50 v1 + 1.00 v2; the dots stay in the
        // sheet; rank 2; b off it, the nearest point missing it by 1.11 at a right angle
        let s = run(1);
        let an = s.analysis();
        assert!(s.count == 3 && an.rank == 2 && !an.independent() && !an.reach && !an.kept[2]);
        assert_eq!(d(V3_IN), d(V2) - d(V1) * 0.5);
        assert_eq!(as_walk(&an, 2), "−0.50v_1 + 1.00v_2");
        assert_eq!(rank_line(&an), "Rank 2: $v_3 = −0.50v_1 + 1.00v_2$, so it adds nothing.");
        let dots = dot_points(&s);
        let (lo, hi) = spread(&dot_numbers(&s, 2));
        assert!(lo < -0.99 && hi > 0.99 && dots.iter().all(|p| p.dot(n).abs() < 1e-6), "the third number from {lo} to {hi}");
        assert_eq!(length_text(an.miss), "1.11");
        let miss = s.b.as_dvec3() - an.p;
        assert!(miss.dot(d(V1)).abs() < 1e-9 && miss.dot(d(V2)).abs() < 1e-9 && (an.end(&s.walk(&an)) - an.p).length() < 1e-9);
        assert_eq!(about_b(&an, &s.walk(&an), false, false), "$b$ is off the plane: no walk reaches it. The nearest point misses it by 1.11.");

        // lift it off the plane: 1.25 straight up; the dots leave the sheet on both
        // sides and spread through a slanted box (every one of 4 x 4 x 4 parts of it
        // holds dots);
        // rank 3; the same b, by exactly one walk, 1.00 v1 + 0.50 v2 + 1.00 v3
        let s = run(2);
        let an = s.analysis();
        assert_eq!(d(V3_UP) - d(V3_IN), DVec3::new(0.0, 0.0, 1.25));
        assert!(an.rank == 3 && an.independent() && an.reach && !an.many() && s.b == Vec3::from(TRIALS[1].b));
        let heights: Vec<f32> = dot_points(&s).iter().map(|p| p.dot(n) as f32).collect();
        let (lo, hi) = spread(&heights);
        assert!(lo < -1.0 && hi > 1.0, "heights from {lo} to {hi}");
        let mut parts = std::collections::HashSet::new();
        for c in &s.dots {
            let part = |x: f32| ((x + 1.0) * 2.0).floor().clamp(0.0, 3.0) as usize;
            parts.insert((part(c[0]), part(c[1]), part(c[2])));
        }
        assert_eq!(parts.len(), 64);
        // (and the plane's dots cover the square of their numbers as evenly: 8 x 8 parts)
        let s0 = run(0);
        let square: std::collections::HashSet<(usize, usize)> = s0.dots.iter().map(|c| (((c[0] + 1.0) * 4.0).floor().clamp(0.0, 7.0) as usize, ((c[1] + 1.0) * 4.0).floor().clamp(0.0, 7.0) as usize)).collect();
        assert_eq!(square.len(), 64);
        let walk = s.walk(&an);
        assert_eq!(terms_text(&walk, 3), "$1.00v_1 + 0.50v_2 + 1.00v_3$");
        assert_eq!(about_b(&an, &walk, false, false), "One walk reaches $b$: $(1.00, 0.50, 1.00)$ are its coordinates in the basis $v_1, v_2, v_3$.");
        assert_eq!(headline(&an), "Three independent arrows: every point of space, one walk each.");
        // (and walks with larger numbers reach anywhere)
        let mut rng = Rng::new(4);
        for _ in 0..50 {
            let far = Vec3::new(rng.normal(), rng.normal(), rng.normal()) * 3.0;
            let there = Analysis::of(s.v, 3, far);
            assert!(there.reach && (there.end(&there.base) - far.as_dvec3()).length() < 1e-5);
        }

        // many walks to one point: v3 and b in the plane; the walk slides while its end
        // stays on b, through b = (1 + 0.5t) v1 + (0.5 - t) v2 + t v3; four more drawn
        let mut s = Span::create();
        s.set_up(3);
        let an = s.analysis();
        assert!(s.slide.is_some() && an.many() && an.nulls == 1 && an.rank == 2);
        assert!(s.b.as_dvec3().dot(n).abs() < 1e-9 && s.v[2].as_dvec3().dot(n).abs() < 1e-9);
        let (mut least, mut most) = (0.0f64, 0.0f64);
        while s.slide.is_some() {
            s.advance(1.0 / 60.0);
            let c = s.walk(&an);
            let t = c[2];
            assert!((an.end(&c) - s.b.as_dvec3()).length() < 1e-9, "{c:?}");
            assert!((c[0] - (1.0 + 0.5 * t)).abs() < 1e-9 && (c[1] - (0.5 - t)).abs() < 1e-9, "{c:?}");
            (least, most) = (least.min(t), most.max(t));
        }
        assert!(least < -0.95 && most > 0.95, "t from {least} to {most}");
        let ends: Vec<[f64; 3]> = GHOSTS.iter().map(|t| ghost_walk(&s, &an, *t)).collect();
        for (k, c) in ends.iter().enumerate() {
            assert!((an.end(c) - s.b.as_dvec3()).length() < 1e-9 && ends[..k].iter().all(|o| o != c));
        }
        assert!(about_b(&an, &s.walk(&an), false, true).contains("many walks reach it: this one slides"));
        assert!(about_b(&an, &s.walk(&an), false, false).contains("many walks reach it: this one, the faint ones"));
        assert_eq!(headline(&an), "Three dependent arrows: only a plane, and many walks to each point of it.");

        // two arrows in one line: v2 = -0.75 v1; the dots stay on the line; rank 1; b
        // (which the opening's two arrows reached) is off it, missed by 0.78
        let s = run(4);
        let an = s.analysis();
        assert_eq!(d(V2_BACK), d(V1) * -0.75);
        assert!(s.count == 2 && an.rank == 1 && !an.reach && s.b == Vec3::from(TRIALS[0].b) && Analysis::of(run(0).v, 2, s.b).reach);
        assert_eq!(as_walk(&an, 1), "−0.75v_1");
        let along = d(V1).normalize();
        assert!(dot_points(&s).iter().all(|p| p.cross(along).length() < 1e-6));
        assert_eq!(length_text(an.miss), "0.78");
        assert_eq!(headline(&an), "Two dependent arrows: only a line, and many walks to each point of it.");
        assert_eq!(about_b(&an, &s.walk(&an), false, false), "$b$ is off the line: no walk reaches it. The nearest point misses it by 0.78.");
    }

    /// The scene opens on the first experiment.
    #[test]
    fn the_scene_opens_on_the_first_experiment() {
        let (a, mut b) = (Span::create(), Span::create());
        b.set_up(0);
        assert!(a.v == b.v && a.b == b.b && a.count == b.count && a.sweep == Some(0.0) && a.seen == (YAW, PITCH));
    }

    /// A sweep leaves the same dots at any frame rate, each where the path is, and
    /// ends exactly on the walk to b; the walk shown moves on smoothly all the while.
    #[test]
    fn a_sweep_is_the_same_at_every_frame_rate() {
        let sweep = |hz: f32| {
            let mut s = Span::create();
            s.set_up(2);
            let an = s.analysis();
            let mut before = s.walk(&an);
            let mut most = 0.0f64;
            while s.sweep.is_some() {
                s.advance(1.0 / hz);
                let now = s.walk(&an);
                most = most.max((0..3).map(|i| (now[i] - before[i]).abs()).fold(0.0, f64::max));
                before = now;
            }
            assert_eq!(s.walk(&an), s.walk_to_b(&an));
            (s.dots, most * hz as f64)
        };
        let (slow, speed60) = sweep(60.0);
        let (fast, speed144) = sweep(144.0);
        assert!(slow == fast && slow.len() == dots_by(S_END), "{} and {} dots", slow.len(), fast.len());
        for (k, c) in slow.iter().enumerate() {
            let on = path(dot_place(k));
            assert!((0..3).all(|i| (c[i] as f64 - on[i]).abs() < 1e-6));
        }
        // (no number moves by more than so much a second, at either rate: no jumps)
        assert!(speed60 < 25.0 && speed144 < 25.0, "{speed60} {speed144} a second");
    }

    /// The magnet: a tip within 3 degrees of the line of another arrow, or the plane of
    /// the other two, is put exactly into it (and the arrows are then dependent); a tip
    /// further off is left where it is; b likewise, into the span. Off, nothing is moved.
    #[test]
    fn the_magnet_takes_a_tip_within_three_degrees() {
        let n = normal().as_vec3();
        let within = d(V1).normalize().as_vec3();
        let across = n.cross(within);
        let mut s = run(2);
        for (deg, taken) in [(2.9f32, true), (3.1, false)] {
            let at = (across * deg.to_radians().cos() + n * deg.to_radians().sin()) * 1.6;
            let tip = s.settle_arrow(2, at);
            assert_eq!(tip != at, taken, "{deg} degrees");
            let mut t = run(2);
            t.v[2] = tip;
            assert_eq!(t.analysis().rank, if taken { 2 } else { 3 }, "{deg} degrees");
        }
        // onto the line of another, before the plane
        let near_v1 = (within * 2.5f32.to_radians().cos() + across * 2.5f32.to_radians().sin()) * 1.2;
        s.v[1] = s.settle_arrow(1, near_v1);
        let an = s.analysis();
        assert!(an.rank == 2 && !an.kept[1] && as_walk(&an, 1) == format!("{}v_1", amount(1.2 / d(V1).length())), "{}", as_walk(&an, 1));
        // b into a plane
        let mut s = run(1);
        let off = (across * 2.0f32.to_radians().cos() + n * 2.0f32.to_radians().sin()) * 2.0;
        s.b = s.settle_b(off);
        assert!(s.analysis().reach && s.b != off);
        s.snap = false;
        assert_eq!(s.settle_b(off), off);
        assert_eq!(s.settle_arrow(2, near_v1), near_v1);
        // and no tip goes further out than REACH
        assert!((s.settle_arrow(0, Vec3::new(30.0, 40.0, 0.0)).length() - REACH).abs() < 1e-5);
    }

    /// Wherever the handles are: the nearest point is in the span and the miss is at
    /// right angles to it; the walk to b ends at the nearest point; the walks that come
    /// back to 0 do; and each arrow left out is the walk along the kept ones it says.
    #[test]
    fn the_walks_end_where_they_say() {
        let mut rng = Rng::new(7);
        let mut point = || Vec3::new(rng.normal(), rng.normal(), rng.normal()) * 1.5;
        for k in 0..600 {
            let mut v = [point(), point(), point()];
            // (dependent sets too: some arrow a mix of the others, or on the line of one)
            match k % 4 {
                1 => v[2] = v[0] * 0.75 - v[1] * 1.25,
                2 => v[1] = v[0] * -0.5,
                _ => {}
            }
            let b = if k % 3 == 0 { v[0] * 0.5 + v[1] } else { point() };
            for count in [2, 3] {
                let an = Analysis::of(v, count, b);
                let scale = v[..count].iter().map(|a| a.length() as f64).fold(1.0, f64::max);
                let miss = b.as_dvec3() - an.p;
                // (an arrow left out is in the span to within `TOL` of the longest: only so
                // far is the miss at right angles to it, and do the walks along the kept
                // arrows rebuild it)
                let loose = |i: usize| if an.kept[i] { 1e-9 } else { 2.0 * TOL };
                for i in 0..count {
                    assert!(miss.dot(an.v[i]).abs() < loose(i) * scale * scale.max(b.length() as f64), "{k}");
                }
                assert!((an.end(&an.base) - an.p).length() < 1e-9 * scale, "{k}");
                for nv in &an.null[..an.nulls] {
                    assert!(an.end(nv).length() < 2.0 * TOL * scale, "{k}: {nv:?}");
                }
                for j in (0..count).filter(|&j| !an.kept[j]) {
                    let rebuilt = (0..count).fold(DVec3::ZERO, |p, i| p + an.v[i] * an.along[j][i]);
                    assert!((rebuilt - an.v[j]).length() < 2.0 * TOL * scale, "{k}");
                }
                assert_eq!(an.rank + an.nulls, count);
                if k % 4 == 1 && count == 3 {
                    assert!(an.rank == 2 && !an.kept[2], "{k}");
                }
                if k % 4 == 2 {
                    assert!(an.rank == count - 1 && !an.kept[1], "{k}");
                }
            }
        }
    }

    /// Arrows of no length, all in one line, two on top of each other, b at 0, nearly
    /// dependent arrows: every number drawn or written is finite, nothing says "?", and
    /// what the words say has happened has.
    #[test]
    fn degenerate_arrows_leave_everything_finite() {
        let (a, b, c) = (Vec3::from(V1), Vec3::from(V2), Vec3::from(V3_UP));
        let tilted = a + b.normalize() * 0.02;
        let far = ([a, tilted, a + tilted], a * 0.5 + b.normalize());
        let tilt = |deg: f32| (b.normalize() * deg.to_radians().cos() + normal().as_vec3() * deg.to_radians().sin()) * 1.6;
        let cases: Vec<([Vec3; 3], Vec3)> = vec![
            ([Vec3::ZERO, b, c], Vec3::from(B_OFF)),
            ([a, Vec3::ZERO, c], Vec3::from(B_OFF)),
            ([Vec3::ZERO; 3], Vec3::from(B_OFF)),
            ([Vec3::ZERO; 3], Vec3::ZERO),
            ([a, a, a], Vec3::from(B_IN)),
            ([a, a * 2.0, a * -0.5], a * 0.3),
            ([a, b, c], Vec3::ZERO),
            ([a, b, Vec3::ZERO], Vec3::ZERO),
            ([a, b, tilt(4.0)], Vec3::from(B_OFF)),
            ([a, b, tilt(0.01)], Vec3::from(B_OFF)),
            ([a, a + Vec3::splat(1e-7), c], Vec3::from(B_OFF)),
            ([a * 1e-7, b, c], Vec3::from(B_OFF)),
            ([Vec3::splat(REACH), Vec3::splat(-REACH), c], Vec3::splat(REACH)),
            // (two arrows half a degree apart, a third in their plane, and b in it too: the
            // walks to b, and the faint ones, go a hundred units out and more)
            far,
        ];
        let an = Analysis::of(far.0, 3, far.1);
        assert!(an.many() && an.rank == 2 && an.base.iter().any(|c| c.abs() > 40.0), "{an:?}");
        for (v, target) in cases {
            for count in [2, 3] {
                let mut s = Span::create();
                (s.v, s.b, s.count) = (v, target, count);
                s.restart();
                s.start_slide();
                for _ in 0..30 {
                    s.advance(0.1);
                }
                let an = s.analysis();
                let walk = s.walk(&an);
                assert!(walk.iter().all(|x| x.is_finite()) && an.p.is_finite() && an.miss.is_finite(), "{v:?} {count}");
                for nv in &an.null[..an.nulls] {
                    assert!(nv.iter().all(|x| x.is_finite()));
                }
                for sweeping in [false, true] {
                    for line in panel_lines(&an, &walk, sweeping, !sweeping, true, 300.0, &estimate).lines {
                        for (text, _, _) in line.pieces {
                            assert!(!text.contains('?') && !text.contains("NaN") && !text.contains("inf"), "{v:?} {count}: {text}");
                        }
                    }
                }
                s.frame_on_picture(true);
                assert!(matches!(s.the_view(), View::Orbit { target, dist, .. } if target.is_finite() && dist.is_finite() && dist > 0.0));
                // no length: it adds nothing, and is named so
                for j in 0..count {
                    if v[j].length() == 0.0 {
                        assert!(!an.kept[j] && (an.rank == 0 || rank_line(&an).contains(&format!("{} has no length", LABELS[j]))), "{}", rank_line(&an));
                    }
                }
                // b at 0 is always reached
                if target == Vec3::ZERO {
                    assert!(an.reach, "{v:?}");
                }
            }
        }
        // nearly dependent arrows are named so, with the angle, and the large numbers
        let an = Analysis::of([a, b, tilt(4.0)], 3, Vec3::from(B_OFF));
        let warned = warning(&an, &an.base, false).unwrap_or_default();
        let largest = an.base.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        assert!(warned.starts_with("$v_3$ is only 4.0° out of the plane of $v_1$ and $v_2$") && warned.contains(&amount(largest)) && largest > 5.0, "{warned}");
        assert!(warning(&Analysis::of([a, b, c], 3, Vec3::from(B_OFF)), &an.base, false).is_none());
        let flat = Analysis::of([a, b, tilt(0.01)], 3, Vec3::from(B_OFF));
        assert!(flat.rank == 3 && flat.base.iter().any(|x| x.abs() > 1000.0) && terms(&flat.base, 3).iter().all(|(t, _)| t.trim_matches('$').chars().count() < 12));
    }

    /// A tip the magnet has moved is drawn gliding there, nearer each frame, exactly
    /// there within 0.4 seconds at any frame rate; what is not moved is drawn
    /// where it is, and nothing glides while no time passes.
    #[test]
    fn a_moved_tip_glides_there() {
        for hz in [60.0f32, 144.0] {
            let mut s = run(0);
            assert!(s.arrived() && !s.animated());
            s.v[1] += Vec3::new(0.2, 0.0, -0.1);
            assert!(s.animated());
            let mut left = (s.v[1] - s.drawn[1]).length();
            let mut frames = 0;
            while !s.arrived() {
                s.glide(1.0 / hz, false);
                let now = (s.v[1] - s.drawn[1]).length();
                assert!(now < left && s.drawn[0] == s.v[0], "frame {frames}");
                left = now;
                frames += 1;
                assert!(frames as f32 / hz < 0.4, "{frames} frames at {hz} Hz");
            }
        }
        let mut s = run(0);
        s.b = Vec3::ONE;
        let was = s.drawn;
        s.glide(0.0, false);
        assert_eq!(s.drawn, was);
        s.glide(0.0, true);
        assert!(s.arrived());
    }

    /// The span's picture fades from one to the next and then comes to rest.
    #[test]
    fn the_span_fades_to_its_new_picture() {
        let mut s = run(2);
        assert!(s.faded() && s.look == [0.0, 0.0, 0.0, 1.0]);
        s.v[2] = Vec3::from(V3_IN);
        let an = s.analysis();
        s.fade(&an, 1.0 / 60.0);
        let mut frames = 0;
        while !s.faded() {
            s.fade(&an, 1.0 / 60.0);
            frames += 1;
            assert!(s.look[2] > 0.0 && frames < 60);
        }
        assert!(s.look == [0.0, 0.0, 1.0, 0.0] && matches!(s.shapes[2], Shape::Plane(..)));
    }

    /// The words and the picture share the canvas without leaving it, whatever its
    /// shape; the panel's lines fit in it; and the view fitted to what is left is a view.
    #[test]
    fn the_canvas_is_shared_on_every_screen() {
        for (x, y) in [(1070.0f32, 1000.0f32), (1100.0, 700.0), (390.0, 800.0), (360.0, 640.0), (600.0, 900.0), (844.0, 390.0), (640.0, 480.0), (240.0, 180.0), (2000.0, 1300.0), (8.0, 8.0)] {
            let view = Vec2::new(x, y);
            let inset = if x < 900.0 { 46.0 } else { 0.0 };
            for k in 0..TRIALS.len() {
                let mut s = run(k);
                (s.canvas, s.inset) = (view, inset);
                let an = s.analysis();
                for sweeping in [false, true] {
                    let room = text_room(view);
                    let said = panel_lines(&an, &s.walk(&an), sweeping, false, true, room, &estimate);
                    let (lines, column) = (said.lines, said.column);
                    let height = panel_height(&lines);
                    let (panel, (min, max)) = layout(view, inset, height);
                    if let Some((lo, hi)) = panel {
                        assert!(lo.x >= 0.0 && lo.y >= inset && hi.x <= x && hi.y <= y && hi.y - lo.y >= height - 0.01, "{x}x{y}: {lo} {hi} for {height}");
                        assert!(hi.x - lo.x >= room + 2.0 * PAD - 0.01);
                        for line in &lines {
                            let wide = line_width(line, column);
                            assert!(wide <= room + 1.0, "{x}x{y}: {line:?}");
                        }
                        // the buttons are in the panel
                        let lines_height: f32 = lines.iter().map(|l| line_height(l.size)).sum();
                        for (b0, b1, _) in button_boxes(lo, hi, lines_height) {
                            assert!(b0.cmpge(lo).all() && b1.cmple(hi).all(), "{x}x{y}: a button outside the panel");
                        }
                    }
                    assert!(min.x >= 0.0 && min.y >= 0.0 && max.x <= x.max(1.0) && max.y <= y.max(1.0) && max.x > min.x && max.y > min.y, "{x}x{y}");
                }
                s.frame_on_picture(true);
                assert!(matches!(s.the_view(), View::Orbit { target, dist, .. } if target.is_finite() && dist.is_finite() && dist > 0.0), "{x}x{y}");
            }
        }
    }

    /// Numbers as written: walks with true minus signs and no "−0.00", lengths too small
    /// for two decimals given in more.
    #[test]
    fn numbers_are_written_plainly() {
        assert_eq!(terms_text(&[1.0, -0.5, 0.25], 3), "$1.00v_1 - 0.50v_2 + 0.25v_3$");
        assert_eq!(terms_text(&[-0.001, -0.004, 1234.6], 3), "$0.00v_1 + 0.00v_2 + 1235v_3$");
        assert_eq!(terms_text(&[-2.0, 0.0, 0.0], 2), "$−2.00v_1 + 0.00v_2$");
        assert_eq!(length_text(0.0042), "0.0042");
        assert_eq!(length_text(1.114), "1.11");
        assert_eq!(length_text(0.0), "0.00");
        assert_eq!(degrees_text(4.04), "4.0°");
    }

    /// What is kept from frame to frame (the names, `Still`; the panel's lines; the sheet
    /// of a plane) follows
    /// every change: run side by side with a copy that works everything out afresh in
    /// every frame, the scene draws exactly the same in every frame, through the opening
    /// sweep, drags of each arrow, of b and of the view, the magnet, a third arrow, every
    /// experiment, the sweep and the slide over the picture, a zoom, a pause, a phone's
    /// window, and the scene opened again.
    #[test]
    fn what_is_kept_follows_every_change() {
        use super::super::subspaces::tried::{labels_follow, Step::*};
        let steps = [
            Rest(720),
            Drag([942.0, 552.0], [900.0, 610.0]),
            Rest(30),
            Drag([998.0, 488.0], [960.0, 530.0]),
            Rest(30),
            Drag([875.0, 444.0], [960.0, 550.0]),
            Rest(30),
            Drag([1000.0, 820.0], [1100.0, 780.0]),
            Rest(40),
            Scroll([800.0, 500.0], -200.0),
            Rest(30),
            Click([1450.0, 915.0]),
            Rest(40),
            Try(3),
            Rest(60),
            Click([520.0, 160.0]),
            Rest(660),
            Try(1),
            Rest(200),
            Try(2),
            Rest(720),
            Try(4),
            Rest(60),
            Try(0),
            Rest(60),
            Click([343.0, 160.0]),
            Rest(60),
            Pause,
            Rest(20),
            Pause,
            Rest(30),
            Size([390.0, 800.0]),
            Rest(30),
            Drag([298.0, 274.0], [280.0, 320.0]),
            Rest(30),
            Size([844.0, 390.0]),
            Rest(30),
            Size([1600.0, 1000.0]),
            Rest(30),
            Away,
            Rest(30),
        ];
        if let Some(again) = labels_follow("span", &steps) {
            assert!(again > 150, "the names were written again as they were only {again} times");
        }
    }
}

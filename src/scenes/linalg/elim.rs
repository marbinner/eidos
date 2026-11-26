//! Elimination: equations seen as planes (and, to begin with, as lines), changed one at
//! a time, without changing what solves them, until the answer can be read off.

use super::subspaces::{keeps, Inputs, Still};
use super::*;
use glam::{DVec2, DVec3};
use std::rc::Rc;

pub const DEF: SceneDef = SceneDef {
    id: "elim",
    title: "Elimination",
    group: "Linear algebra: equations",
    blurb: "Solve equations by turning planes about the line where they meet, until the answer can be read off.",
    make: |_| Box::new(Elim::create()),
};

/// The colours of the equations, their planes (or lines) and their rows, by the order
/// they were first given in; and what the words call them.
const EQ: [Color; 3] = [pal::PINK, pal::GREEN, pal::BLUE];
const EQ_NAMES: [&str; 3] = ["pink", "green", "blue"];
/// The line (or, in the plane, the point) that a plane turns about in a step.
const HINGE: Color = pal::YELLOW;
/// The solution: the point on every plane, or the line of such points.
const SOLUTION: Color = pal::WHITE;
/// The box, and the axes inside it.
const FRAME: Color = Color::hex(0x5A6582);

/// The unknowns, as markup to go in a formula and as formulas of their own.
const VAR: [&str; 3] = ["x", "y", "z"];
const VAR_LABELS: [&str; 3] = ["$x$", "$y$", "$z$"];

/// A number counts as 0 when it is less than this share of the largest number in A.
/// Rounding leaves crumbs where there should be nothing, of up to about one part in
/// 10¹⁴ (measured over thousands of singular systems typed with two decimals), far
/// below it; the numbers typed into the panel (two decimals, at most 100) are far above.
const ZERO: f64 = 1e-9;

/// Seconds a step takes to play forwards and backwards, and the pause after each one
/// when they are played through.
const STEP_TIME: f32 = 1.6;
const BACK_TIME: f32 = 0.5;
const HOLD: f32 = 1.0;
/// The pause before the first step of an experiment: time to see the planes as given.
const FIRST_HOLD: f32 = 1.5;

/// The equations as rows of numbers: in row i, the n numbers of A, then b (so n + 1 of
/// the four places are used).
type Rows = [[f64; 4]; 3];

/// n equations in n unknowns, A x = b, as typed (rows of A, and b).
#[derive(Clone, Copy, Debug, PartialEq)]
struct System {
    n: usize,
    a: [[f64; 3]; 3],
    b: [f64; 3],
}

impl System {
    const fn three(a: [[f64; 3]; 3], b: [f64; 3]) -> System {
        System { n: 3, a, b }
    }

    const fn two(a: [[f64; 2]; 2], b: [f64; 2]) -> System {
        System { n: 2, a: [[a[0][0], a[0][1], 0.0], [a[1][0], a[1][1], 0.0], [0.0; 3]], b: [b[0], b[1], 0.0] }
    }

    fn rows(&self) -> Rows {
        let n = self.n.clamp(2, 3);
        let mut rows = [[0.0; 4]; 3];
        for i in 0..n {
            for j in 0..n {
                rows[i][j] = self.a[i][j];
            }
            rows[i][n] = self.b[i];
        }
        rows
    }
}

// ----- the arithmetic -------------------------------------------------------------------

/// One step.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    /// Row `row` take away `m` times row `from`, which takes unknown `col` out of it.
    Sub { row: usize, from: usize, m: f64, col: usize },
    /// Rows `a` and `b` change places, to bring a number into the pivot's place in
    /// row `a`, column `col`.
    Swap { a: usize, b: usize, col: usize },
    /// Row `row` divided by its pivot `p`, in column `col`.
    Scale { row: usize, p: f64, col: usize },
}

/// What the equations come to.
#[derive(Clone, Debug, PartialEq)]
enum Outcome {
    /// One solution.
    Point([f64; 3]),
    /// No solution: row `row` reads 0 = `value`.
    Impossible { row: usize, value: f64 },
    /// A solution for every value of the free unknowns: `base` is the one with them
    /// all 0, and each of `dirs` the way it moves as one of them grows by 1.
    Many { free: Vec<usize>, base: [f64; 3], dirs: Vec<[f64; 3]> },
}

/// Elimination worked through: every step, and the rows before the first and after
/// each one.
#[derive(Clone, Debug)]
struct Plan {
    n: usize,
    states: Vec<Rows>,
    /// Which equation (as first given: 0, 1, 2) is in each row, before the first step
    /// and after each one.
    order: Vec<[usize; 3]>,
    ops: Vec<Op>,
    /// How many of the steps go down (elimination); the rest go back up.
    down: usize,
    /// The pivots, in the order they were found: (row, column, value).
    pivots: Vec<(usize, usize, f64)>,
    /// The multipliers: l[i][k] for the k-th pivot's row taken from row i, rows in
    /// their order once going down is done. With 1 on the diagonal this is L, and
    /// P A = L U, U being the rows then and P the order they are in.
    l: [[f64; 3]; 3],
    swaps: usize,
    /// What counts as 0 among the numbers of A.
    tol: f64,
    outcome: Outcome,
    /// The places after the point the numbers on their way between steps are written
    /// with (see `decimals`), worked out once.
    places: usize,
    /// The most characters a number of the rows takes, at any step or on its way
    /// between two (at least 2), and a multiplier of L (at least 1): the widths of the
    /// matrices' cells over the picture (see `Rowboard`), worked out once.
    widest: usize,
    l_widest: usize,
}

/// The largest size among the first n numbers of each of the first n rows (the n + 1,
/// b included, if `with_b`).
fn largest(m: &Rows, n: usize, with_b: bool) -> f64 {
    let cols = if with_b { n + 1 } else { n };
    m.iter().take(n).flat_map(|r| r.iter().take(cols)).map(|v| v.abs()).filter(|v| v.is_finite()).fold(0.0, f64::max)
}

impl Plan {
    fn of(sys: &System) -> Plan {
        let n = sys.n.clamp(2, 3);
        let mut m = sys.rows();
        let tol = ZERO * largest(&m, n, false);
        // (what is left of a zero on the right-hand side is measured against everything,
        // b included, at its largest along the way)
        let mut big = largest(&m, n, true);
        // (what counts as 0 is 0 from the start: nothing typed into the panel does, but
        // the rule is the same for every number)
        for row in m.iter_mut().take(n) {
            for v in row.iter_mut().take(n) {
                if v.abs() <= tol || !v.is_finite() {
                    *v = 0.0;
                }
            }
            if row[n].abs() <= ZERO * big || !row[n].is_finite() {
                row[n] = 0.0;
            }
        }
        let mut plan = Plan {
            n,
            states: vec![m],
            order: vec![[0, 1, 2]],
            ops: Vec::new(),
            down: 0,
            pivots: Vec::new(),
            l: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            swaps: 0,
            tol,
            outcome: Outcome::Point([0.0; 3]),
            places: 1,
            widest: 2,
            l_widest: 1,
        };
        let mut order = [0, 1, 2];
        // a row just changed, with what counts as 0 made exactly 0
        let snap = |row: &mut [f64; 4], big: f64| {
            for v in row.iter_mut().take(n) {
                if v.abs() <= tol {
                    *v = 0.0;
                }
            }
            if row[n].abs() <= ZERO * big {
                row[n] = 0.0;
            }
        };

        // going down: below each pivot, every number in its column taken out
        let mut r = 0;
        for c in 0..n {
            if r == n {
                break;
            }
            if m[r][c] == 0.0 {
                // no pivot here: a later row with a number in this column comes up
                match (r + 1..n).find(|&k| m[k][c] != 0.0) {
                    Some(k) => {
                        m.swap(r, k);
                        order.swap(r, k);
                        for j in 0..r {
                            let t = plan.l[r][j];
                            plan.l[r][j] = plan.l[k][j];
                            plan.l[k][j] = t;
                        }
                        plan.swaps += 1;
                        plan.push(Op::Swap { a: r, b: k, col: c }, m, order);
                    }
                    // none: this unknown will be free
                    None => continue,
                }
            }
            let p = m[r][c];
            for k in r + 1..n {
                if m[k][c] == 0.0 {
                    continue;
                }
                let mult = m[k][c] / p;
                for j in 0..=n {
                    m[k][j] -= mult * m[r][j];
                }
                m[k][c] = 0.0;
                big = big.max(largest(&m, n, true));
                snap(&mut m[k], big);
                plan.l[k][r] = mult;
                plan.push(Op::Sub { row: k, from: r, m: mult, col: c }, m, order);
            }
            plan.pivots.push((r, c, p));
            r += 1;
        }
        plan.down = plan.ops.len();

        // a row with nothing left of A but something on the right is true of no point
        let rank = plan.pivots.len();
        if let Some(k) = (rank..n).find(|&k| m[k][n] != 0.0) {
            plan.outcome = Outcome::Impossible { row: k, value: m[k][n] };
            plan.measure();
            return plan;
        }

        // going back up: from the last pivot's row to the first, put in the unknowns
        // already found (take them out of the row), then divide by the pivot
        for idx in (0..rank).rev() {
            let (r, c, _) = plan.pivots[idx];
            for idx2 in (idx + 1..rank).rev() {
                let (r2, c2, _) = plan.pivots[idx2];
                if m[r][c2] != 0.0 {
                    let mult = m[r][c2] / m[r2][c2];
                    for j in 0..=n {
                        m[r][j] -= mult * m[r2][j];
                    }
                    m[r][c2] = 0.0;
                    big = big.max(largest(&m, n, true));
                    snap(&mut m[r], big);
                    plan.push(Op::Sub { row: r, from: r2, m: mult, col: c2 }, m, order);
                }
            }
            let p = m[r][c];
            if p != 1.0 {
                for v in m[r].iter_mut().take(n + 1) {
                    *v /= p;
                }
                m[r][c] = 1.0;
                snap(&mut m[r], big);
                plan.push(Op::Scale { row: r, p, col: c }, m, order);
            }
        }

        // the answer, read off the rows
        let mut base = [0.0; 3];
        for &(r, c, _) in &plan.pivots {
            base[c] = m[r][n];
        }
        let free: Vec<usize> = (0..n).filter(|c| !plan.pivots.iter().any(|p| p.1 == *c)).collect();
        plan.outcome = if free.is_empty() {
            Outcome::Point(base)
        } else {
            let dirs = free
                .iter()
                .map(|&f| {
                    let mut d = [0.0; 3];
                    d[f] = 1.0;
                    for &(r, c, _) in &plan.pivots {
                        d[c] = -m[r][f];
                    }
                    d
                })
                .collect();
            Outcome::Many { free, base, dirs }
        };
        plan.measure();
        plan
    }

    /// Work out how the numbers are written: their places after the point, and the
    /// widths of the matrices' cells.
    fn measure(&mut self) {
        self.places = count_places(self);
        let (n, places) = (self.n, self.places);
        // (wide enough for the numbers of every step, and for those on their way
        // between two, which lie between them)
        let width = |v: f64| number(v).chars().count().max(passing(v, places).chars().count());
        self.widest = self.states.iter().flat_map(|m| m.iter().take(n).flat_map(|r| r.iter().take(n + 1))).map(|v| width(*v)).max().unwrap_or(1).max(2);
        // (and for a multiplier on its way from 0 to what it is, in the step that finds it)
        self.l_widest = (0..n).flat_map(|i| (0..i).map(move |j| (i, j))).map(|(i, j)| width(self.l[i][j])).max().unwrap_or(1).max(1);
    }

    fn push(&mut self, op: Op, m: Rows, order: [usize; 3]) {
        self.ops.push(op);
        self.states.push(m);
        self.order.push(order);
    }

    /// The steps, counting the last one, which reads off the answer.
    fn steps(&self) -> usize {
        self.ops.len() + 1
    }

    /// det A: the product of the pivots, negated for each swap (0 if a column had none).
    fn det(&self) -> f64 {
        if self.pivots.len() < self.n {
            return 0.0;
        }
        let product: f64 = self.pivots.iter().map(|p| p.2).product();
        if self.swaps % 2 == 1 {
            -product
        } else {
            product
        }
    }

    /// U: the rows once going down is done (the left-hand sides).
    fn u(&self) -> &Rows {
        &self.states[self.down]
    }
}

// ----- the rows as drawn ----------------------------------------------------------------

/// The rows at a stage of the steps: during a step, on their way from what they were
/// before it to what they are after it.
#[derive(Clone, Copy, Debug)]
struct Shown {
    /// The numbers as written.
    rows: Rows,
    /// The equations whose planes are drawn. (During a division, the row before it:
    /// the plane does not move, and the numbers on their way could pass through 0.)
    geo: Rows,
    /// Which equation is in each row, and where each row is written (counted in rows
    /// from the top: whole numbers but during a swap).
    order: [usize; 3],
    slot: [f32; 3],
    /// How strongly each row's numbers are written: two rows changing places fade as
    /// they pass through each other.
    ink: [f32; 3],
    /// The step playing (its index among the operations), and how far through it.
    playing: Option<(usize, f64)>,
    /// The row whose numbers are changing (written to a fixed number of places on
    /// their way, so that they keep their width).
    changing: Option<usize>,
    /// How much of each row's plane is drawn: 1, but for one fading away as its row
    /// comes to nothing.
    fade: [f32; 3],
}

/// Each row's sign after the first `k` steps, as its plane is drawn: a row divided by
/// a negative pivot is the same plane, but its numbers have all changed sign, and the
/// plane is drawn from them as they were (so that it keeps the way it faces, and the
/// pieces of it drawn keep their ends the same way round).
fn faces(plan: &Plan, k: usize) -> [f64; 3] {
    let mut sign = [1.0; 3];
    for op in plan.ops.iter().take(k) {
        match *op {
            Op::Swap { a, b, .. } => sign.swap(a, b),
            Op::Scale { row, p, .. } if p < 0.0 => sign[row] = -sign[row],
            _ => {}
        }
    }
    sign
}

/// Rows with each multiplied by its sign.
fn signed_rows(rows: &Rows, sign: [f64; 3]) -> Rows {
    std::array::from_fn(|i| rows[i].map(|v| v * sign[i]))
}

impl Shown {
    /// The row each equation is in (the inverse of `order`): the planes are drawn
    /// equation by equation, so that each keeps its place in what is drawn when rows
    /// change places.
    fn row_of(&self) -> [usize; 3] {
        let mut row = [0, 1, 2];
        for (i, &e) in self.order.iter().enumerate() {
            row[e.min(2)] = i;
        }
        row
    }
}

fn shown(plan: &Plan, stage: f32) -> Shown {
    let ops = plan.ops.len();
    let s = stage.clamp(0.0, ops as f32);
    let k = (s.floor() as usize).min(ops);
    let t = s - k as f32;
    let rest = |k: usize| Shown { rows: plan.states[k], geo: signed_rows(&plan.states[k], faces(plan, k)), order: plan.order[k], slot: [0.0, 1.0, 2.0], ink: [1.0; 3], playing: None, changing: None, fade: [1.0; 3] };
    if t <= 0.0 || k >= ops {
        return rest(k);
    }
    let mut out = rest(k);
    let before = plan.states[k];
    let e = smoothstep(t) as f64;
    out.playing = Some((k, e));
    let n = plan.n;
    match plan.ops[k] {
        Op::Sub { row, from, m, .. } => {
            out.changing = Some(row);
            for j in 0..=n {
                out.rows[row][j] = before[row][j] - e * m * before[from][j];
            }
            out.geo = signed_rows(&out.rows, faces(plan, k));
            // a row coming to nothing: its plane dissolves (0 = 0) or slides away
            // (0 = something), and fades as it goes
            if (0..n).all(|j| plan.states[k + 1][row][j] == 0.0) {
                out.fade[row] = (1.0 - e) as f32;
            }
        }
        Op::Swap { a, b, .. } => {
            let e = e as f32;
            out.slot[a] = a as f32 + (b as f32 - a as f32) * e;
            out.slot[b] = b as f32 + (a as f32 - b as f32) * e;
            let ink = 1.0 - 0.85 * (PI * e).sin();
            (out.ink[a], out.ink[b]) = (ink, ink);
        }
        Op::Scale { row, p, .. } => {
            out.changing = Some(row);
            let factor = 1.0 + e * (1.0 / p - 1.0);
            for j in 0..=n {
                out.rows[row][j] = before[row][j] * factor;
            }
        }
    }
    out
}

// ----- numbers and words ----------------------------------------------------------------

/// A number as the rows write it: to three significant digits, no more than it needs
/// (2, 0.5, −1.25, 0.333), with a true minus sign; 0 for what counts as 0.
fn number(v: f64) -> String {
    if !v.is_finite() {
        return "?".into();
    }
    let a = v.abs();
    if a == 0.0 {
        return "0".into();
    }
    if a >= 1e5 {
        let e = a.log10().floor() as i32;
        return format!("{}e{}", number(v / 10f64.powi(e)), e);
    }
    let digits = (2 - a.log10().floor() as i32).clamp(0, 7) as usize;
    let text = format!("{a:.digits$}");
    let text = if text.contains('.') { text.trim_end_matches('0').trim_end_matches('.').to_string() } else { text };
    if text == "0" {
        "0".into()
    } else if v < 0.0 {
        format!("−{text}")
    } else {
        text
    }
}

/// The places after the point that the numbers on their way between two steps are
/// written with: as many as the numbers of the steps need, at least one and at most
/// three. (So that they keep their width as they change.)
fn decimals(plan: &Plan) -> usize {
    plan.places
}

fn count_places(plan: &Plan) -> usize {
    let n = plan.n;
    plan.states.iter().flat_map(|m| m.iter().take(n).flat_map(|r| r.iter().take(n + 1))).map(|v| number(*v).split('.').nth(1).map_or(0, |d| d.len())).max().unwrap_or(0).clamp(1, 3)
}

/// A number on its way between two steps.
fn passing(v: f64, decimals: usize) -> String {
    num(v, decimals)
}

/// A multiplier as written in a step: in brackets when it is negative.
fn factor(m: f64) -> String {
    let t = number(m);
    if t.starts_with('−') {
        format!("({t})")
    } else {
        t
    }
}

/// A row as an equation, as markup to go in a formula: 2x + y − z = 5, z = 2, 0 = 1.
fn equation(row: &[f64; 4], n: usize) -> String {
    equation_with(row, n, &number)
}

/// The same with the numbers written by `write` (a term written as 0 is left out).
fn equation_with(row: &[f64; 4], n: usize, write: &dyn Fn(f64) -> String) -> String {
    let mut text = String::new();
    for (j, name) in VAR.iter().enumerate().take(n) {
        let c = row[j];
        let size = write(c.abs());
        if c == 0.0 || size.chars().all(|d| d == '0' || d == '.') {
            continue;
        }
        // (a number with a power of ten, in brackets before its unknown)
        let coef = if size == "1" {
            String::new()
        } else if size.contains('e') {
            format!("({size})")
        } else {
            size
        };
        if text.is_empty() {
            text = format!("{}{coef}{name}", if c < 0.0 { "−" } else { "" });
        } else {
            text += &format!(" {} {coef}{name}", if c < 0.0 { "−" } else { "+" });
        }
    }
    if text.is_empty() {
        text = "0".into();
    }
    format!("{text} = {}", write(row[n]))
}

fn is_empty(row: &[f64; 4], n: usize) -> bool {
    row.iter().take(n).all(|v| *v == 0.0)
}

/// What the unknown in column `c` is, read off the last rows, as markup to go in a
/// formula: "y = 3 − 2z".
fn solved_for(plan: &Plan, c: usize) -> String {
    let n = plan.n;
    let last = plan.states.last().copied().unwrap_or([[0.0; 4]; 3]);
    let Some(&(r, _, _)) = plan.pivots.iter().find(|p| p.1 == c) else { return format!(r"{} \text{{ is free}}", VAR[c]) };
    let mut text = String::new();
    let b = last[r][n];
    if b != 0.0 {
        text = number(b);
    }
    for f in 0..n {
        let coef = -last[r][f];
        if f == c || coef == 0.0 {
            continue;
        }
        let size = number(coef.abs());
        let size = if size == "1" {
            String::new()
        } else if size.contains('e') {
            format!("({size})")
        } else {
            size
        };
        if text.is_empty() {
            text = format!("{}{size}{}", if coef < 0.0 { "−" } else { "" }, VAR[f]);
        } else {
            text += &format!(" {} {size}{}", if coef < 0.0 { "−" } else { "+" }, VAR[f]);
        }
    }
    if text.is_empty() {
        text = "0".into();
    }
    format!("{} = {text}", VAR[c])
}

/// A point, as markup to go in a formula: (1, 1, 2).
fn point_text(p: &[f64; 3], n: usize) -> String {
    let parts: Vec<String> = p.iter().take(n).map(|v| number(*v)).collect();
    format!("({})", parts.join(", "))
}

/// Which way a row's plane (or line) stands, from the unknowns missing from it.
fn standing(row: &[f64; 4], n: usize) -> String {
    let missing: Vec<&str> = (0..n).filter(|&j| row[j] == 0.0).map(|j| VAR_LABELS[j]).collect();
    if n == 2 {
        return match missing.as_slice() {
            ["$x$"] => "level (parallel to the $x$ axis)".into(),
            ["$y$"] => "upright (parallel to the $y$ axis)".into(),
            _ => "as it was".into(),
        };
    }
    match missing.as_slice() {
        [one] => format!("parallel to the {one} axis"),
        [one, two] => format!("parallel to the {one} and {two} axes"),
        _ => "as it was".into(),
    }
}

/// The words for a part of the picture: plane or line, and where two meet.
struct Words {
    thing: &'static str,
    things: &'static str,
}

fn words(n: usize) -> Words {
    if n == 3 {
        Words { thing: "plane", things: "planes" }
    } else {
        Words { thing: "line", things: "lines" }
    }
}

/// Are two rows' planes parallel (or the same)? (In the plane: lines.)
fn parallel(p: &[f64; 4], q: &[f64; 4], n: usize) -> bool {
    let (a, b) = (DVec3::new(p[0], p[1], if n == 3 { p[2] } else { 0.0 }), DVec3::new(q[0], q[1], if n == 3 { q[2] } else { 0.0 }));
    a.cross(b).length() <= 1e-9 * a.length() * b.length()
}

/// The title of a step: what is done to the rows.
fn step_title(plan: &Plan, k: usize) -> String {
    match plan.ops.get(k) {
        Some(Op::Sub { row, from, m, .. }) => format!(r"$\text{{row }} {} - {} \times \text{{row }} {}$", row + 1, factor(*m), from + 1),
        Some(Op::Swap { a, b, .. }) => format!("swap rows {} and {}", a + 1, b + 1),
        Some(Op::Scale { row, p, .. }) => format!(r"$\text{{row }} {} \div {}$", row + 1, factor(*p)),
        None => answer_title(plan),
    }
}

/// The title of the last step: the answer.
fn answer_title(plan: &Plan) -> String {
    let n = plan.n;
    match &plan.outcome {
        Outcome::Point(p) => format!("${}$", (0..n).map(|c| format!("{} = {}", VAR[c], number(p[c]))).collect::<Vec<_>>().join(r",\quad ")),
        Outcome::Impossible { .. } => "No solution".into(),
        Outcome::Many { free, .. } => match free.len() {
            1 if n == 3 => "A line of solutions".into(),
            1 => "A line of solutions".into(),
            2 if n == 3 => "A plane of solutions".into(),
            _ => "Every point is a solution".into(),
        },
    }
}

/// What a step does, in a few sentences. `hinge_seen`: whether the line the plane
/// turns about is in the box; `answer_seen`: whether the answer is (for the last step).
fn step_words(plan: &Plan, k: usize, hinge_seen: bool, answer_seen: bool) -> String {
    let n = plan.n;
    let w = words(n);
    let Some(op) = plan.ops.get(k).copied() else { return answer_words(plan, answer_seen) };
    let (before, after, order) = (&plan.states[k], &plan.states[k + 1], plan.order[k]);
    let colour = |row: usize| EQ_NAMES[order[row]];
    match op {
        Op::Sub { row, from, m, col } => {
            if is_empty(&after[row], n) {
                let b = after[row][n];
                return if b == 0.0 {
                    format!(
                        r"Row {} is ${} \times \text{{row }} {}$: the {} {} is the {} one. Taking it away leaves $0 = 0$, true of every point. The row says nothing more, and its {} is gone.",
                        row + 1,
                        number(m),
                        from + 1,
                        colour(row),
                        w.thing,
                        colour(from),
                        w.thing
                    )
                } else {
                    format!(
                        r"The {} {} is parallel to the {} one: they never meet. Taking ${} \times \text{{row }} {}$ away leaves $0 = {}$, true of no point: the {} slides away and is gone.",
                        colour(row),
                        w.thing,
                        colour(from),
                        factor(m),
                        from + 1,
                        number(b),
                        w.thing
                    )
                };
            }
            let about = match (n, hinge_seen) {
                (2, true) => "the crossing (yellow)".to_string(),
                (2, false) => "the crossing (outside the view)".to_string(),
                (_, true) => format!("its line with the {} one (yellow)", colour(from)),
                _ => format!("its line with the {} one (outside the box)", colour(from)),
            };
            let turn = format!("the {} {} turns about {about} until it is {}.", colour(row), w.thing, standing(&after[row], n));
            if k < plan.down {
                let small = before[from][col].abs() < 0.05 * plan.tol / ZERO;
                let still = match (&plan.outcome, n) {
                    (Outcome::Point(_), 3) => " It still holds that line, and the solution on it.",
                    (Outcome::Many { .. }, 3) => " It still holds that line, and the solutions on it.",
                    _ => "",
                };
                let note = if small { format!(" The pivot {} is small but not 0, so there is no swap.", number(before[from][col])) } else { still.to_string() };
                format!(
                    r"Multiplier ${} = {} \div {}$: the number to take out, divided by the pivot. {} drops out of row {}: {turn}{note}",
                    number(m),
                    number(before[row][col]),
                    factor(before[from][col]),
                    VAR_LABELS[col],
                    row + 1
                )
            } else {
                let alone = (0..n).all(|j| j == col || before[from][j] == 0.0);
                let put = if alone { format!(" (the same as putting ${} = {}$ in)", VAR[col], number(before[from][n])) } else { String::new() };
                format!(
                    r"Row {} reads ${}$. Taking ${} \times \text{{row }} {}$ from row {} takes {} out of it{put}: {turn}",
                    from + 1,
                    equation(&before[from], n),
                    factor(m),
                    from + 1,
                    row + 1,
                    VAR_LABELS[col]
                )
            }
        }
        Op::Swap { a, b, col } => format!(
            "Row {} has 0 where the pivot should be, so no multiple of it can take {} out of the rows below. Row {} has {} there: the rows change places. The {} do not move; only the order of the equations does.",
            a + 1,
            VAR_LABELS[col],
            b + 1,
            number(before[b][col]),
            w.things
        ),
        Op::Scale { row, p, .. } => {
            let alone = (0..n).filter(|&j| after[row][j] != 0.0).count() == 1;
            let lead = if k == plan.down && alone { "The last row holds one unknown alone. " } else { "" };
            format!(
                "{lead}Divide row {} by its pivot, {}: it reads ${}$. The same {}: dividing both sides of an equation by a number changes none of its points.",
                row + 1,
                number(p),
                equation(&after[row], n),
                w.thing
            )
        }
    }
}

/// How the first planes stand, when no point is on them all.
fn arrangement(plan: &Plan) -> String {
    let n = plan.n;
    let rows = plan.states[0];
    let w = words(n);
    for (i, row) in rows.iter().enumerate().take(n) {
        if is_empty(row, n) && row[n] != 0.0 {
            return format!("Row {} reads $0 = {}$ from the start: no point satisfies it.", i + 1, number(row[n]));
        }
    }
    for i in 0..n {
        for j in i + 1..n {
            if !is_empty(&rows[i], n) && !is_empty(&rows[j], n) && parallel(&rows[i], &rows[j], n) {
                return format!("The {} and {} {} are parallel: they never meet.", EQ_NAMES[i], EQ_NAMES[j], w.things);
            }
        }
    }
    if n == 3 {
        "The planes meet two at a time, in three parallel lines, but never all three at once.".into()
    } else {
        String::new()
    }
}

/// The words of the last step.
fn answer_words(plan: &Plan, seen: bool) -> String {
    let n = plan.n;
    match &plan.outcome {
        Outcome::Point(p) => {
            if n == 3 {
                format!("Each row holds one unknown, so each plane is parallel to two axes. They meet at ${}$, where the first three met: no step moved it.", point_text(p, n))
            } else {
                format!("Each row holds one unknown: one line is upright, the other level, and they cross at ${}$, where the first two crossed.", point_text(p, n))
            }
        }
        Outcome::Impossible { row, value } => format!(
            "Row {} reads $0 = {}$, which no point satisfies, so no point satisfies all {} equations: there is no solution. {}",
            row + 1,
            number(*value),
            if n == 3 { "three" } else { "two" },
            arrangement(plan)
        )
        .trim_end()
        .to_string(),
        Outcome::Many { free, .. } => {
            let zero_rows: Vec<String> = (plan.pivots.len()..n).map(|r| (r + 1).to_string()).collect();
            let says = match zero_rows.len() {
                0 => String::new(),
                1 => format!("Row {} reads $0 = 0$: it said nothing new. ", zero_rows[0]),
                _ => format!("Rows {} read $0 = 0$. ", zero_rows.join(" and ")),
            };
            let names: Vec<&str> = free.iter().map(|&f| VAR_LABELS[f]).collect();
            if free.len() == n {
                return format!("{says}Every point satisfies every row: every point is a solution.");
            }
            let solved: Vec<String> = plan.pivots.iter().map(|p| format!("${}$", solved_for(plan, p.1))).collect();
            let is = if free.len() == 1 { "is" } else { "are" };
            let set = match (n, free.len()) {
                (2, _) => "the one line both equations are".to_string(),
                (_, 1) if seen => "the white line: each of them satisfies all three equations".to_string(),
                (_, 1) => "a line, outside the box: each of its points satisfies all three equations".to_string(),
                _ => "the plane of row 1, the one row left".into(),
            };
            format!("{says}{} {is} free: for any {}, {}. Those points make {set}.", names.join(" and "), names.join(" and "), solved.join(" and "))
        }
    }
}

/// The words before the first step. `seen`: whether the answer is in the picture.
fn intro_words(plan: &Plan, seen: bool) -> String {
    let n = plan.n;
    let (each, all) = if n == 3 { ("Each equation is a plane: the points $(x, y, z)$ that satisfy it.", "all three") } else { ("Each equation is a line: the points $(x, y)$ that satisfy it.", "both") };
    let w = words(n);
    let here = match &plan.outcome {
        Outcome::Point(p) => format!("A point on {all} satisfies {all} equations: ${}${}.", point_text(p, n), if seen { ", the white dot" } else { " (outside the picture)" }),
        Outcome::Impossible { .. } => format!("Here no point is on {all} {}.", w.things),
        Outcome::Many { free, .. } if free.len() == 1 => format!("Here a whole line of points satisfies {all}{}.", if seen { " (white)" } else { " (outside the picture)" }),
        Outcome::Many { .. } => format!("Here the {} have a whole plane of points in common.", w.things),
    };
    format!("{each} {here} Each step changes one equation, but not what solves them.")
}

/// The words before the first step, with what that step will be and what the ring and
/// the box on the rows mark, as much of that as fits in the lines there are for it at
/// `width` pixels.
fn opening_words(plan: &Plan, width: f32, seen: bool) -> String {
    let intro = intro_words(plan, seen);
    let Some(first) = plan.ops.first() else { return intro };
    let marks = match first {
        Op::Sub { .. } => " (ringed: the pivot; boxed: the number it takes out)",
        Op::Swap { .. } => " (ringed: a 0 where a pivot should be)",
        Op::Scale { .. } => " (ringed: the pivot)",
    };
    let title = step_title(plan, 0);
    [format!("{intro} First: {title}{marks}."), format!("{intro} First: {title}."), intro.clone()].into_iter().find(|t| wrap(t, SMALL, width).len() <= LINES).unwrap_or(intro)
}

/// The words of a text, as split at its spaces, each formula kept whole (with the
/// words written against it).
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

/// The words of the step shown, broken into lines no wider than `width` pixels.
fn wrap(text: &str, size: f32, width: f32) -> Vec<String> {
    // (a formula is kept whole, and so is a short bracket of words)
    let operator = |t: &str| matches!(t, "=" | "+" | "−" | "×" | "÷");
    let mut groups: Vec<String> = Vec::new();
    for token in words_of(text) {
        let glue = match groups.last() {
            Some(last) => {
                let open = last.matches('(').count() > last.matches(')').count() && last.chars().count() < 20;
                operator(token) || last.split(' ').next_back().is_some_and(operator) || open
            }
            None => false,
        };
        match groups.last_mut() {
            Some(last) if glue => {
                last.push(' ');
                last.push_str(token);
            }
            _ => groups.push(token.to_string()),
        }
    }
    // (each group measured once, its formulas from the fonts' metrics)
    let mut lines: Vec<(String, f32)> = vec![(String::new(), 0.0)];
    for word in &groups {
        let wide = text_width(word, size);
        let Some((last, so_far)) = lines.last_mut() else { break };
        if last.is_empty() {
            last.push_str(word);
            *so_far = wide;
        } else if *so_far + wide + size * 0.47 > width {
            lines.push((word.clone(), wide));
        } else {
            last.push(' ');
            last.push_str(word);
            *so_far += wide + size * 0.47;
        }
    }
    lines.into_iter().map(|(line, _)| line).collect()
}

// ----- the picture's geometry -----------------------------------------------------------

/// A plane n · p = d with n of length 1 (in the plane: a line, with n's last part 0).
#[derive(Clone, Copy, Debug)]
struct Plane {
    n: DVec3,
    d: f64,
}

impl Plane {
    /// The plane of a row, if it has one (a row of zeros on the left has none).
    fn of(row: &[f64; 4], n: usize) -> Option<Plane> {
        let normal = DVec3::new(row[0], row[1], if n == 3 { row[2] } else { 0.0 });
        let len = normal.length();
        let d = row[n];
        (len > 0.0 && len.is_finite() && d.is_finite()).then(|| Plane { n: normal / len, d: d / len })
    }

    /// The part of the plane inside the box |x|, |y|, |z| ≤ r: its corners in order
    /// round it (none if it misses the box).
    fn in_box(&self, r: f64) -> Vec<DVec3> {
        if !self.d.is_finite() || self.d.abs() > r * 3f64.sqrt() {
            return Vec::new();
        }
        let u = self.n.any_orthonormal_vector();
        let v = self.n.cross(u);
        let (c, big) = (self.n * self.d, r * 4.0);
        let mut poly = vec![c - u * big - v * big, c + u * big - v * big, c + u * big + v * big, c - u * big + v * big];
        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                poly = clip(&poly, |p| sign * p[axis] - r);
            }
        }
        // (a plane that only touches the box at an edge or a corner has nothing in it)
        poly.dedup_by(|a, b| a.distance(*b) < 1e-9 * r);
        if poly.len() >= 2 && poly[0].distance(poly[poly.len() - 1]) < 1e-9 * r {
            poly.pop();
        }
        if poly.len() < 3 {
            poly.clear();
        }
        poly
    }
}

/// Where a plane crosses each face of the box |x|, |y|, |z| ≤ r, face by face in a
/// fixed order (−x, +x, −y, +y, −z, +z): the outline of its part in the box, drawn as
/// pieces that each move smoothly as the plane turns. (The corners of the polygon,
/// counted round it, would start from a different one now and then.)
fn face_edges(p: &Plane, r: f64) -> Vec<(usize, DVec3, DVec3)> {
    let mut out = Vec::new();
    for k in 0..3 {
        let (u, v) = ((k + 1) % 3, (k + 2) % 3);
        for s in [-1.0, 1.0] {
            // on the face x_k = s r: n_u a + n_v b = d − n_k s r, within the square
            let line = Plane { n: DVec3::new(p.n[u], p.n[v], 0.0), d: p.d - p.n[k] * s * r };
            let len = DVec2::new(line.n.x, line.n.y).length();
            if len < 1e-12 {
                continue;
            }
            let line = Plane { n: line.n / len, d: line.d / len };
            if let Some((a, b)) = line_in_rect(&line, DVec2::splat(-r), DVec2::splat(r)) {
                let lift = |q: DVec2| {
                    let mut c = DVec3::ZERO;
                    (c[k], c[u], c[v]) = (s * r, q.x, q.y);
                    c
                };
                if a.distance(b) > 1e-9 * r {
                    out.push((2 * k + usize::from(s > 0.0), lift(a), lift(b)));
                }
            }
        }
    }
    out
}

/// What is left of a polygon where `side` is at most 0.
fn clip(poly: &[DVec3], side: impl Fn(DVec3) -> f64) -> Vec<DVec3> {
    let mut out = Vec::with_capacity(poly.len() + 2);
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        let (sa, sb) = (side(a), side(b));
        if sa <= 0.0 {
            out.push(a);
        }
        if (sa <= 0.0) != (sb <= 0.0) {
            out.push(a + (b - a) * (sa / (sa - sb)));
        }
    }
    out
}

/// The part of the line through `p` along `d` inside the box |x|, |y|, |z| ≤ r.
fn line_in_box(p: DVec3, d: DVec3, r: f64) -> Option<(DVec3, DVec3)> {
    let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
    for k in 0..3 {
        if d[k].abs() < 1e-12 {
            if p[k].abs() > r {
                return None;
            }
            continue;
        }
        let (a, b) = ((-r - p[k]) / d[k], (r - p[k]) / d[k]);
        t0 = t0.max(a.min(b));
        t1 = t1.min(a.max(b));
    }
    (t0 < t1 && t0.is_finite() && t1.is_finite()).then(|| (p + d * t0, p + d * t1))
}

/// Where two planes meet: a point of the line and its direction (length 1), if they
/// are not parallel.
fn meeting(p: &Plane, q: &Plane) -> Option<(DVec3, DVec3)> {
    let dir = p.n.cross(q.n);
    let s = dir.length();
    if s <= 1e-9 {
        return None;
    }
    let at = (q.n.cross(dir) * p.d + dir.cross(p.n) * q.d) / (s * s);
    at.is_finite().then(|| (at, dir / s))
}

/// Where two lines in the plane cross, if they are not parallel.
fn crossing(p: &Plane, q: &Plane) -> Option<DVec2> {
    let det = p.n.x * q.n.y - p.n.y * q.n.x;
    if det.abs() <= 1e-9 {
        return None;
    }
    let at = DVec2::new(p.d * q.n.y - q.d * p.n.y, p.n.x * q.d - q.n.x * p.d) / det;
    at.is_finite().then_some(at)
}

/// The part of a line in the plane inside the rectangle `lo`..`hi`.
fn line_in_rect(pl: &Plane, lo: DVec2, hi: DVec2) -> Option<(DVec2, DVec2)> {
    let (n, d) = (DVec2::new(pl.n.x, pl.n.y), pl.d);
    let (p, dir) = (n * d, DVec2::new(-n.y, n.x));
    let (mut t0, mut t1) = (f64::NEG_INFINITY, f64::INFINITY);
    for k in 0..2 {
        if dir[k].abs() < 1e-12 {
            if p[k] < lo[k] || p[k] > hi[k] {
                return None;
            }
            continue;
        }
        let (a, b) = ((lo[k] - p[k]) / dir[k], (hi[k] - p[k]) / dir[k]);
        t0 = t0.max(a.min(b));
        t1 = t1.min(a.max(b));
    }
    (t0 < t1 && t0.is_finite() && t1.is_finite()).then(|| (p + dir * t0, p + dir * t1))
}

// ----- the layout -----------------------------------------------------------------------

/// The size of the panel over the picture, and of its text.
const PANEL_W: f32 = 372.0;
const PANEL_H: f32 = 262.0;
const HEAD: f32 = 11.0;
const TITLE: f32 = 15.0;
const MAT: f32 = 13.0;
const SMALL: f32 = 11.5;
const LINE_H: f32 = 15.0;
/// The most lines the words of a step may take.
const LINES: usize = 5;

/// Where the panel goes on a canvas of `view` logical pixels, `inset` of whose top lies
/// under the application's buttons: over the picture's top-left corner, or below the
/// picture on a canvas narrow and tall enough (a phone held upright). Left out where
/// the canvas cannot spare the room.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    narrow: bool,
    panel: Option<(Vec2, Vec2)>,
    /// The part of the canvas the picture has to itself.
    picture: (Vec2, Vec2),
}

/// How many lines the caption along the bottom of the picture may take: one across a
/// wide canvas, three across a narrow one (none on a canvas too low for it).
fn caption_most(view: Vec2) -> usize {
    match (view.y >= 300.0 && view.x >= 200.0, view.x < 620.0) {
        (false, _) => 0,
        (true, false) => 2,
        (true, true) => 3,
    }
}

/// The height kept for it.
fn caption_height(view: Vec2) -> f32 {
    let lines = caption_most(view);
    if lines == 0 {
        0.0
    } else {
        lines as f32 * CAPTION_LINE + 12.0
    }
}

impl Layout {
    fn of(view: Vec2, inset: f32) -> Layout {
        let caption = caption_height(view);
        let whole = (Vec2::new(0.0, inset.min(view.y * 0.5)), (view - Vec2::Y * caption).max(Vec2::ONE));
        if view.x < 620.0 && view.y >= 600.0 {
            if view.x < 340.0 {
                return Layout { narrow: true, panel: None, picture: whole };
            }
            let top = view.y - PANEL_H - 10.0;
            let panel = (Vec2::new(8.0, top), Vec2::new(view.x - 8.0, view.y - 10.0));
            Layout { narrow: true, panel: Some(panel), picture: (Vec2::new(0.0, inset + 4.0), Vec2::new(view.x, top - 6.0 - caption)) }
        } else {
            let top = 14.0f32.max(inset + 8.0);
            let fits = view.x >= PANEL_W + 220.0 && view.y >= top + PANEL_H + 40.0;
            if !fits {
                return Layout { narrow: false, panel: None, picture: whole };
            }
            let panel = (Vec2::new(14.0, top), Vec2::new(14.0 + PANEL_W, top + PANEL_H));
            Layout { narrow: false, panel: Some(panel), picture: (Vec2::new(panel.1.x + 10.0, inset), view - Vec2::Y * caption) }
        }
    }

    fn covers(&self, p: Vec2) -> bool {
        self.panel.is_some_and(|(min, max)| p.cmpge(min).all() && p.cmple(max).all())
    }
}

/// The direction the planes are seen from at first (yaw and pitch, as the camera
/// counts them): x towards the viewer and to the right, y to the right and away, z up.
/// Each experiment has its own, from which its planes are seen well.
const LOOK: (f32, f32) = (-0.85, 0.42);
/// (The tangent of half the camera's field of view.)
const TAN_HALF_FOV: f32 = 0.3203;

// ----- marks that glide -----------------------------------------------------------------

/// A mark on the rows (the ring round the pivot, the box round the number to take out)
/// as drawn: it follows where it belongs through two eases one after the other, so
/// that it sets off and arrives gently, and fades in and out where it comes and goes.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Mark {
    between: Vec2,
    at: Vec2,
    alpha: f32,
}

/// How long a mark takes to follow (the time of each ease, in seconds).
const MARK_TIME: f32 = 0.06;

impl Mark {
    /// `dt` seconds nearer to `to` (or to nothing). Returns whether it is still on its way.
    fn follow(&mut self, to: Option<Vec2>, dt: f32) -> bool {
        self.follow_in(to, dt, MARK_TIME)
    }

    /// The same with eases of `time` seconds.
    fn follow_in(&mut self, to: Option<Vec2>, dt: f32, time: f32) -> bool {
        let k = 1.0 - (-dt.max(0.0) / time.max(1e-3)).exp();
        match to {
            Some(p) if p.is_finite() => {
                if self.alpha <= 0.0 {
                    // (coming from nothing: it appears where it belongs)
                    (self.between, self.at) = (p, p);
                }
                self.between += (p - self.between) * k;
                self.at += (self.between - self.at) * k;
                self.alpha += (1.0 - self.alpha) * k;
                if self.at.distance(p) < 0.3 && self.between.distance(p) < 0.3 && self.alpha > 0.99 {
                    (self.between, self.at, self.alpha) = (p, p, 1.0);
                }
            }
            _ => {
                self.alpha -= self.alpha * k;
                if self.alpha < 0.01 {
                    self.alpha = 0.0;
                }
            }
        }
        let settled = match to {
            Some(p) if p.is_finite() => self.at == p && self.alpha == 1.0,
            _ => self.alpha == 0.0,
        };
        !settled
    }
}

// ----- the experiments ------------------------------------------------------------------

/// How far the box reaches at first: |x|, |y|, |z| ≤ this. (Not a whole number: planes
/// with whole numbers in their equations so often go exactly through an edge of a box
/// of whole size, where the piece of the plane's outline on a face jumps.)
const BOX: f64 = 3.2;

/// Two lines crossing at (1, 2).
const TWO: System = System::two([[1.0, 1.0], [2.0, -1.0]], [3.0, 0.0]);
/// Three planes meeting at (−1, 2, 1). (Chosen, among systems whose steps go in
/// whole numbers, for planes at least 45° apart at every step, none of them seen
/// nearly edge on from the opening direction.)
const THREE: System = System::three([[1.0, -1.0, 0.0], [3.0, 2.0, 3.0], [-2.0, -3.0, 2.0]], [-3.0, 4.0, -2.0]);
/// Once x is out, a 0 is where the second pivot should be: (−1, 1, 2).
const SWAP: System = System::three([[2.0, -1.0, -1.0], [-2.0, 1.0, -2.0], [2.0, 3.0, -1.0]], [-5.0, -1.0, -1.0]);
/// Row 3 of A is row 1 plus row 2, but b₃ is not b₁ + b₂. (The three rows of A are
/// 60° apart, and all at right angles to (1, −1, −1), the way the lines run where
/// the planes meet.)
const NONE: System = System::three([[1.0, 2.0, -1.0], [1.0, -1.0, 2.0], [2.0, 1.0, 1.0]], [3.0, 0.0, 7.0]);
/// The same, with b₃ = b₁ + b₂: the planes share the line through (1, 1, 0) that way.
const LINE: System = System::three([[1.0, 2.0, -1.0], [1.0, -1.0, 2.0], [2.0, 1.0, 1.0]], [3.0, 0.0, 3.0]);

/// Experiments: (button, what to look for, the equations, how far the picture reaches,
/// the direction it is seen from).
type Trial = (&'static str, &'static str, System, f64, (f32, f32));
const TRIALS: [Trial; 5] = [
    (
        "two lines",
        r"Two equations in $x$ and $y$ are two lines, and the solution is where they cross: $(1, 2)$. $\text{Row } 2 - 2 \times \text{row } 1$ takes $x$ out of row 2: the green line turns about the crossing until it is level, $-3y = -6$. Dividing by the pivot −3 gives $y = 2$; going back up, $\text{row } 1 - \text{row } 2$ leaves $x = 1$. The crossing never moves.",
        TWO,
        3.0,
        LOOK,
    ),
    (
        "three planes, one point",
        "Three planes meeting at $(-1, 2, 1)$. Going down, each step turns one plane about its line with another (yellow) until an unknown drops out of its equation; after three steps the blue plane is level, $5z = 5$. Going back up gives $z = 1$, then $y = 2$, then $x = -1$. The pivots are 1, 5 and 5; the multipliers 3, −2 and −1 make up $L$.",
        THREE,
        BOX,
        LOOK,
    ),
    (
        "a zero pivot: swap",
        "Once $x$ is out, row 2 reads $-3z = -6$: 0 where the second pivot should be, so no multiple of it can take $y$ out of row 3. Row 3 has a 4 there, so the two rows swap places. The planes do not move; only the order of the equations does (the green one is now row 3). Then $z = 2$, $y = 1$ and $x = -1$.",
        SWAP,
        BOX,
        LOOK,
    ),
    (
        "no solution: $0 = 4$",
        r"These planes meet two at a time, in three parallel lines, but no point is on all three. Once $x$ is out of rows 2 and 3, their planes are parallel; $\text{row } 3 - \text{row } 2$ then leaves $0 = 4$, true of no point, and the blue plane slides away. There is no solution.",
        NONE,
        BOX,
        LOOK,
    ),
    (
        "a line of solutions: $0 = 0$",
        r"The same left-hand sides, with 3 in place of 7 on the right of row 3. Now the three planes share a line, like the pages of a book, and $\text{row } 3 - \text{row } 2$ leaves $0 = 0$, true of every point: the blue plane has said nothing new, and is gone. $z$ is free: for any $z$, $x = 1 - z$ and $y = 1 + z$, the white line.",
        LINE,
        BOX,
        LOOK,
    ),
];

/// Systems to start from, besides those of the experiments: (button, the equations).
const PRESETS3: [(&str, System); 4] = [
    ("a small pivot", System::three([[0.01, 1.0, 1.0], [1.0, 1.0, 2.0], [1.0, 2.0, 1.0]], [2.01, 4.0, 4.0])),
    ("two planes the same", System::three([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [1.0, -1.0, 2.0]], [2.0, 4.0, 1.0])),
    ("two planes parallel", System::three([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, -1.0, 0.0]], [1.0, 3.0, 0.0])),
    ("a row of zeros", System::three([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]], [1.0, 0.0, 2.0])),
];
const PRESETS2: [(&str, System); 3] = [
    ("a zero pivot", System::two([[0.0, 1.0], [1.0, 1.0]], [2.0, 3.0])),
    ("parallel lines", System::two([[1.0, 2.0], [2.0, 4.0]], [2.0, 2.0])),
    ("the same line twice", System::two([[1.0, 2.0], [2.0, 4.0]], [2.0, 4.0])),
];

// ----- the scene ------------------------------------------------------------------------

pub struct Elim {
    /// The equations in two unknowns and in three (each kept while the other is shown).
    two: System,
    three: System,
    n: usize,
    /// How far the picture reaches: the box is |x|, |y|, |z| ≤ this, and the view of the
    /// plane this far round (0, 0). One for each number of unknowns.
    reach: [f64; 2],
    /// The direction space is seen from (yaw, pitch).
    look: (f32, f32),
    /// How far through the steps the picture is (0 to the number of steps), and where
    /// it is going.
    stage: f32,
    goal: f32,
    /// Playing through with a pause after each step, and what is left of the pause.
    through: bool,
    hold: f32,
    /// The ring round the pivot and the box round the number to take out, as drawn.
    ring: Mark,
    target: Mark,
    /// The names of the equations' planes (or lines) and of the solution.
    names: [Name; 4],
    /// Whether any of these was still on its way in the last frame.
    gliding: bool,
    rng: Rng,
    /// The canvas last drawn on, and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and with it where the picture goes.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
    /// Elimination worked through for the equations shown, with the bits of the
    /// equations it was worked out for: it is worked out again only when they change.
    worked: Option<([u64; 13], Rc<Plan>)>,
    /// The panel's words on the step shown, with what they were written for.
    told: Option<(ToldFor, Vec<String>)>,
    /// How wide the name of the equation a step changes is at its widest through the
    /// step, with what it was measured for.
    widest: Option<(WidestFor, f32)>,
    /// The caption's lines, with what they were made from.
    captioned: Option<(CaptionFor, Vec<String>)>,
    /// The names in the picture as they were written last, kept while nothing they are
    /// placed from changes (see `Still`).
    kept: Still,
}

/// What the panel's words on a step were written for: the bits of the equations, the
/// step, whether its line and the answer are in the picture, and the bits of the
/// panel's width.
type ToldFor = ([u64; 13], usize, bool, bool, u32);

/// What the widest name of an equation in a step was measured for: the bits of the
/// equations, the step and the row.
type WidestFor = ([u64; 13], usize, usize);

/// What the caption's lines were made from: its parts, the bits of the width they
/// have, and how many lines they may take.
type CaptionFor = (Vec<String>, u32, usize);

/// The bits of a system's numbers, which tell it from every other.
fn bits(sys: &System) -> [u64; 13] {
    let mut out = [sys.n as u64; 13];
    for (k, v) in sys.a.iter().flatten().chain(&sys.b).enumerate() {
        out[k + 1] = v.to_bits();
    }
    out
}

impl Elim {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        Elim {
            two: TWO,
            three: THREE,
            n: 3,
            reach: [3.0, BOX],
            look: LOOK,
            stage: 0.0,
            goal: 0.0,
            through: false,
            hold: 0.0,
            ring: Mark::default(),
            target: Mark::default(),
            names: [Name::default(); 4],
            gliding: false,
            rng: Rng::new(11),
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
            worked: None,
            told: None,
            widest: None,
            captioned: None,
            kept: Still::default(),
        }
    }

    fn system(&self) -> &System {
        if self.n == 2 {
            &self.two
        } else {
            &self.three
        }
    }

    fn system_mut(&mut self) -> &mut System {
        if self.n == 2 {
            &mut self.two
        } else {
            &mut self.three
        }
    }

    fn plan(&self) -> Plan {
        Plan::of(self.system())
    }

    /// Everything the picture and its names are drawn from (see `Still`): not what is
    /// remembered of the names themselves, nor where the steps are going and the pause
    /// before the next (the picture is the same throughout a pause: its names are written
    /// again as they were while it lasts).
    fn inputs(&self, k: &mut Inputs) {
        for word in bits(&self.two).into_iter().chain(bits(&self.three)) {
            k.int(word);
        }
        k.int(self.n as u64).f64s(&self.reach).f32s(&[self.look.0, self.look.1, self.stage]);
        k.f32s(&[self.canvas.x, self.canvas.y, self.inset]).int(self.measured as u64);
    }

    /// The same, kept: worked out again only when the equations have changed.
    fn worked(&mut self) -> Rc<Plan> {
        let made_for = bits(self.system());
        match &self.worked {
            Some((was, plan)) if *was == made_for && keeps() => plan.clone(),
            _ => {
                let plan = Rc::new(self.plan());
                self.worked = Some((made_for, plan.clone()));
                plan
            }
        }
    }

    fn reach(&self) -> f64 {
        self.reach[if self.n == 2 { 0 } else { 1 }]
    }

    /// Use these equations, from the first step, with the picture reaching `reach`.
    fn start(&mut self, sys: System, reach: f64) {
        self.n = sys.n.clamp(2, 3);
        *self.system_mut() = sys;
        self.reach[if self.n == 2 { 0 } else { 1 }] = reach;
        self.look = LOOK;
        (self.stage, self.goal, self.through, self.hold) = (0.0, 0.0, false, 0.0);
        self.new_view = true;
        // (and the names appear where they belong, not gliding there)
        self.names = [Name::default(); 4];
    }

    /// Set up one of the experiments in `TRIALS`: its equations, played from the start
    /// to the end (or the stage it names), with a pause after each step.
    fn set_up(&mut self, k: usize) {
        let (_, _, sys, reach, look) = TRIALS[k];
        self.start(sys, reach);
        self.look = look;
        self.goal = self.plan().steps() as f32;
        (self.through, self.hold) = (true, FIRST_HOLD);
    }

    /// A system with whole numbers all the way: A = LU with small whole L and U, and
    /// an answer of whole numbers within the picture.
    fn random(&mut self) {
        let n = self.n;
        let rng = &mut self.rng;
        let mut pick = |lo: i32, hi: i32| lo + rng.below((hi - lo + 1) as usize) as i32;
        for _ in 0..200 {
            let mut l = [[0.0f64; 3]; 3];
            let mut u = [[0.0f64; 3]; 3];
            for i in 0..n {
                l[i][i] = 1.0;
                for j in 0..i {
                    l[i][j] = pick(-2, 2) as f64;
                }
                let p = pick(1, 3) * if pick(0, 1) == 0 { -1 } else { 1 };
                u[i][i] = p as f64;
                for j in i + 1..n {
                    u[i][j] = pick(-3, 3) as f64;
                }
            }
            let mut a = [[0.0f64; 3]; 3];
            for i in 0..n {
                for j in 0..n {
                    a[i][j] = (0..n).map(|k| l[i][k] * u[k][j]).sum();
                }
            }
            let x: Vec<f64> = (0..n).map(|_| pick(-2, 2) as f64).collect();
            let mut b = [0.0; 3];
            for i in 0..n {
                b[i] = (0..n).map(|j| a[i][j] * x[j]).sum();
            }
            let biggest = a.iter().flatten().fold(0.0f64, |m, v| m.max(v.abs()));
            if biggest <= 9.0 && b.iter().all(|v| v.abs() <= 30.0) {
                self.start(System { n, a, b }, if n == 3 { BOX } else { 3.0 });
                return;
            }
        }
    }

    /// One step on: after the steps already asked for, or, while they are being played
    /// through, after the one playing (and there it stops).
    fn next(&mut self, steps: usize) {
        let from = if self.through { self.stage } else { self.goal.max(self.stage) };
        self.goal = (from.floor() + 1.0).min(steps as f32);
        (self.through, self.hold) = (false, 0.0);
    }

    fn can_go_on(&self, steps: usize) -> bool {
        (if self.through { self.stage } else { self.goal.max(self.stage) }) < steps as f32
    }

    fn back(&mut self) {
        self.goal = (self.goal.min(self.stage).ceil() - 1.0).max(0.0);
        (self.through, self.hold) = (false, 0.0);
    }

    fn replay(&mut self, steps: usize) {
        (self.stage, self.goal, self.through, self.hold) = (0.0, steps as f32, true, HOLD * 0.6);
    }

    fn moving(&self) -> bool {
        self.stage != self.goal || self.hold > 0.0
    }

    /// Keep the steps within those there are (after the equations change).
    fn keep_within(&mut self, steps: usize) {
        let most = steps as f32;
        if self.stage > most || self.goal > most {
            self.stage = self.stage.min(most);
            self.goal = self.goal.min(most);
        }
    }

    /// Move the steps on by `dt` seconds of scene time: towards the goal, a step at a
    /// time, pausing after each when playing through. (All the time a frame owes is
    /// spent, across as many steps and pauses as it reaches.)
    fn advance(&mut self, dt: f32) {
        let mut dt = dt.max(0.0);
        while dt > 0.0 && self.moving() {
            if self.hold > 0.0 {
                let wait = self.hold.min(dt);
                self.hold -= wait;
                dt -= wait;
                if self.hold <= 0.0 {
                    self.hold = 0.0;
                }
                continue;
            }
            let forward = self.goal > self.stage;
            let (next, time) = if forward { ((self.stage.floor() + 1.0).min(self.goal), STEP_TIME) } else { ((self.stage.ceil() - 1.0).max(self.goal), BACK_TIME) };
            let need = (next - self.stage).abs() * time;
            if dt >= need {
                self.stage = next;
                dt -= need;
                if self.through && forward && self.stage < self.goal {
                    self.hold = HOLD;
                }
            } else {
                self.stage += if forward { dt / time } else { -dt / time };
                dt = 0.0;
            }
        }
        if !self.moving() {
            self.through = false;
        }
    }

    /// The step whose words and marks are shown: the one playing, or the last one
    /// done (0 before the first: then the marks are those of step 1, to come).
    fn step(&self) -> usize {
        self.stage.max(0.0).ceil() as usize
    }

    /// The view that takes in the box (three unknowns) or the square the picture
    /// reaches over (two), in the part of the canvas the panel leaves the picture.
    fn the_view(&self) -> View {
        let lay = Layout::of(self.canvas, self.inset);
        let (min, max) = lay.picture;
        let r = self.reach() as f32;
        if self.n == 2 {
            let region = (max - min).max(Vec2::splat(60.0));
            let per_unit = (region.x / (2.25 * r)).min(region.y / (2.25 * r)).max(1e-3);
            let shift = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
            return View::Fit { center: Vec2::new(-shift.x, shift.y), half: self.canvas * 0.5 / per_unit };
        }
        // seen in perspective, the near side looks larger than the far: fitted by
        // looking at the box, moving the camera, and looking again
        let (yaw, pitch) = self.look;
        let right = Vec3::new(-yaw.sin(), yaw.cos(), 0.0);
        let up = Vec3::new(-pitch.sin() * yaw.cos(), -pitch.sin() * yaw.sin(), pitch.cos());
        let eye = right.cross(up);
        let narrow = lay.narrow;
        let margin = if narrow { Vec2::new(24.0, 30.0) } else { Vec2::new(90.0, 70.0) };
        let room = (max - min - margin).max(Vec2::splat(40.0));
        let goal = (min + max) * 0.5;
        let focal = self.canvas.y * 0.5 / TAN_HALF_FOV;
        let mut points: Vec<Vec3> = Vec::new();
        for i in 0..8 {
            points.push(Vec3::new(if i & 1 == 0 { -r } else { r }, if i & 2 == 0 { -r } else { r }, if i & 4 == 0 { -r } else { r }));
        }
        let mut target = Vec3::ZERO;
        let mut dist = 8.0 * r;
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
            let scale = (room.x / size.x).min(room.y / size.y);
            let shift = ((lo + hi) * 0.5 - goal) * (dist / focal);
            target += right * shift.x - up * shift.y;
            dist = (dist / scale).clamp(2.0 * r, 400.0 * r);
        }
        if !(target.is_finite() && dist.is_finite()) {
            return View::Orbit { target: Vec3::ZERO, yaw, pitch, dist: 8.0 * r };
        }
        View::Orbit { target, yaw, pitch, dist }
    }
}

/// What drawing the picture found out: whether the line (or point) a plane turns about
/// in the step shown, and the answer, are in it; and what the caption should say of
/// what is not.
struct Drawn {
    hinge_seen: bool,
    answer_seen: bool,
    notes: Vec<String>,
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

/// A ring round a number over the picture: a rectangle with round ends, `half` its
/// half-width and half-height (logical pixels).
fn hud_pill(f: &mut Frame, centre: Vec2, half: Vec2, width: f32, color: Color) {
    let r = half.y.min(half.x).max(1.0);
    let ends = [centre + Vec2::new(half.x - r, 0.0), centre - Vec2::new(half.x - r, 0.0)];
    let mut points: Vec<Vec2> = Vec::with_capacity(34);
    for (k, end) in ends.iter().enumerate() {
        for i in 0..=16 {
            let a = -PI * 0.5 + PI * (k as f32 + i as f32 / 16.0);
            points.push(*end + Vec2::from_angle(a) * r);
        }
    }
    points.push(points[0]);
    hud_path(f, points.into_iter(), width, color);
}

/// The size of the caption's type, and the height of one of its lines.
const CAPTION_TEXT: f32 = 12.0;
const CAPTION_LINE: f32 = CAPTION_TEXT + 7.0;

/// The lines of a caption along the bottom of the picture `room` pixels wide: `parts`
/// side by side as long as they fit across, a part too long for a line broken between
/// words; no more than `most` lines.
fn caption_lines(parts: &[String], room: f32, most: usize) -> Vec<String> {
    let size = CAPTION_TEXT;
    let gap = "      ";
    let mut lines: Vec<String> = Vec::new();
    for part in parts {
        let pieces = if text_width(part, size) > room { wrap(part, size, room) } else { vec![part.clone()] };
        for (k, piece) in pieces.into_iter().enumerate() {
            match lines.last_mut() {
                Some(last) if k == 0 && text_width(last, size) + text_width(gap, size) + text_width(&piece, size) <= room => {
                    last.push_str(gap);
                    last.push_str(&piece);
                }
                _ => lines.push(piece),
            }
        }
    }
    lines.truncate(most);
    lines
}

/// Draw a caption's lines on a dark plate, the last ending at `bottom` (above the
/// panel, on a phone), like `bottom_caption` in the toolkit; and empty ones after them
/// up to `most`, so that there are as many labels in every frame.
fn caption(f: &mut Frame, lines: &[String], bottom: f32, most: usize) {
    let count = lines.len();
    for (k, line) in lines.iter().enumerate() {
        let up = (count - 1 - k) as f32 * CAPTION_LINE;
        f.caption_px(Vec2::new(18.0, bottom - up), line.as_str(), CAPTION_TEXT, pal::DIM, Align2::LEFT_BOTTOM);
    }
    for _ in count..most {
        f.text_px(Vec2::new(18.0, bottom), "", CAPTION_TEXT, pal::DIM, Align2::LEFT_BOTTOM);
    }
}

/// The places of the rows' numbers over the picture: the matrix [A | b], and L beside
/// it when there is room.
struct Rowboard {
    at: Vec2,
    widest: usize,
    /// The size of the numbers' type: smaller where large numbers would not fit across.
    text: f32,
    size: Vec2,
    l_at: Option<Vec2>,
    l_widest: usize,
}

impl Rowboard {
    fn of(plan: &Plan, panel: (Vec2, Vec2), top: f32) -> Rowboard {
        let (n, widest, l_widest) = (plan.n, plan.widest, plan.l_widest);
        let at = Vec2::new(panel.0.x + 30.0, top);
        let text = [MAT, 12.0, 11.0, 10.0].into_iter().find(|&t| at.x + matrix_size(n, n + 1, widest, t).x <= panel.1.x - 10.0).unwrap_or(9.0);
        let size = matrix_size(n, n + 1, widest, text);
        let l_size = matrix_size(n, n, l_widest, 12.0);
        let l_x = at.x + size.x + 46.0;
        let l_at = (l_x + l_size.x <= panel.1.x - 12.0).then(|| Vec2::new(l_x, top + (size.y - l_size.y) * 0.5));
        Rowboard { at, widest, text, size, l_at, l_widest }
    }

    /// The middle of the number in row slot `slot` (counted from the top), column `j`.
    fn cell(&self, slot: f32, j: usize) -> Vec2 {
        let (lo, hi) = matrix_entry(self.at, self.widest, self.text, 0, j);
        (lo + hi) * 0.5 + Vec2::new(0.0, slot * (hi.y - lo.y))
    }

    fn cell_size(&self) -> Vec2 {
        matrix_cell(self.widest, self.text)
    }
}

impl Elim {
    /// The panel over the picture: the step, its buttons, the rows with the pivot ringed
    /// and the number to take out boxed, L, and what the step does.
    #[allow(clippy::too_many_arguments)]
    fn panel(&mut self, f: &mut Frame, plan: &Plan, sh: &Shown, lay: &Layout, pointer: Option<Vec2>, drawn: &Drawn) {
        let (hinge_seen, answer_seen) = (drawn.hinge_seen, drawn.answer_seen);
        let Some((min, max)) = lay.panel else {
            self.gliding |= self.ring.follow(None, f.real_dt);
            self.gliding |= self.target.follow(None, f.real_dt);
            return;
        };
        hud_panel(f, min, max);
        let n = plan.n;
        let steps = plan.steps();
        let step = self.step().min(steps);
        let x = min.x + 14.0;
        let w = words(n);
        let (head, title) = if step == 0 {
            (format!("Step 0 of {steps} · the equations as given"), format!("{} equations, {} {}", if n == 3 { "Three" } else { "Two" }, if n == 3 { "three" } else { "two" }, w.things))
        } else {
            let phase = if step > plan.ops.len() {
                "the answer"
            } else if step <= plan.down {
                "going down: elimination"
            } else {
                "going back up: back substitution"
            };
            (format!("Step {step} of {steps} · {phase}"), step_title(plan, step - 1))
        };
        f.text_px(Vec2::new(x, min.y + 10.0), head, HEAD, pal::DIM, Align2::LEFT_TOP);
        f.text_px(Vec2::new(x, min.y + 25.0), title, TITLE, pal::TEXT, Align2::LEFT_TOP);

        // the buttons
        let y = min.y + 50.0;
        let at_start = self.goal.min(self.stage) <= 0.0;
        if hud_button(f, Vec2::new(x, y), Vec2::new(x + 72.0, y + 26.0), "◀ back", !at_start, pointer) {
            self.back();
        }
        if hud_button(f, Vec2::new(x + 80.0, y), Vec2::new(x + 152.0, y + 26.0), "next ▶", self.can_go_on(steps), pointer) {
            self.next(steps);
        }
        if hud_button(f, Vec2::new(x + 160.0, y), Vec2::new(x + 250.0, y + 26.0), "↺ replay", true, pointer) {
            self.replay(steps);
        }

        // the rows, under the names of their columns
        let board = Rowboard::of(plan, (min, max), y + 50.0);
        let blank: Vec<Vec<(String, Color)>> = (0..n).map(|_| vec![(String::new(), pal::TEXT); n + 1]).collect();
        hud_matrix(f, board.at, &blank, board.widest, board.text, pal::DIM);
        for j in 0..=n {
            let name = if j < n { VAR_LABELS[j] } else { "$b$" };
            f.text_px(board.cell(-1.0, j) + Vec2::new(0.0, 4.0), name, SMALL, pal::DIM, Align2::CENTER_CENTER);
        }
        let cell = board.cell_size();
        // (the bar between A and b)
        let bar = board.cell(0.0, n).x - cell.x * 0.5;
        hud_line(f, Vec2::new(bar, board.at.y + 3.0), Vec2::new(bar, board.at.y + board.size.y - 3.0), 1.0, pal::DIM.a(0.7));
        let places = decimals(plan);
        for i in 0..n {
            f.text_px(Vec2::new(board.at.x - 8.0, board.cell(i as f32, 0).y), (i + 1).to_string(), SMALL, pal::DIM, Align2::RIGHT_CENTER);
        }
        // (equation by equation, each one's numbers sliding to its new row in a swap)
        let row_of = sh.row_of();
        for i in row_of.into_iter().take(n) {
            let colour = EQ[sh.order[i]];
            for j in 0..=n {
                let v = sh.rows[i][j];
                let digits = if sh.changing == Some(i) { passing(v, places) } else { number(v) };
                let faded = j < n && digits.chars().all(|c| c == '0' || c == '.');
                let colour = if faded { colour.mix(pal::DIM, 0.6).a(0.6 * sh.ink[i]) } else { colour.a(sh.ink[i]) };
                let (text, size) = cell_text(f, &digits, board.text, cell.x);
                f.text_px(board.cell(sh.slot[i], j), text, size, colour, Align2::CENTER_CENTER);
            }
        }

        // L, beside: 1 on the diagonal, the multipliers below it as they are found
        if let Some(l_at) = board.l_at {
            let done = self.stage.floor().max(0.0) as usize;
            let l = multipliers_after(plan, done.min(plan.down));
            let blank: Vec<Vec<(String, Color)>> = (0..n).map(|_| vec![(String::new(), pal::TEXT); n]).collect();
            let lsize = hud_matrix(f, l_at, &blank, board.l_widest, 12.0, pal::DIM);
            f.text_px(Vec2::new(l_at.x - 8.0, l_at.y + lsize.y * 0.5), "$L =$", 13.0, pal::TEXT, Align2::RIGHT_CENTER);
            f.text_px(Vec2::new(l_at.x + lsize.x * 0.5, board.cell(-1.0, 0).y + 4.0), "multipliers", SMALL, pal::DIM, Align2::CENTER_CENTER);
            let lcell = matrix_cell(board.l_widest, 12.0);
            let at = |slot: f32, j: usize| {
                let (lo, hi) = matrix_entry(l_at, board.l_widest, 12.0, 0, j);
                (lo + hi) * 0.5 + Vec2::new(0.0, slot * lcell.y)
            };
            // the diagonal and above it, in their places
            for i in 0..n {
                for j in i..n {
                    f.text_px(at(i as f32, j), if i == j { "$1$" } else { "$0$" }, 12.0, if i == j { pal::DIM } else { pal::FAINT }, Align2::CENTER_CENTER);
                }
            }
            // below it, equation by equation: the multipliers used on each, riding with
            // its row when rows change places (and nothing written where its row has no
            // place below the diagonal)
            // (the one being found in the step playing grows from 0 as the plane turns)
            let growing = match (sh.playing, sh.playing.and_then(|p| plan.ops.get(p.0))) {
                (Some((k, e)), Some(Op::Sub { row, from, m, .. })) if k < plan.down => Some((*row, *from, e * m)),
                _ => None,
            };
            for i in row_of.into_iter().take(n) {
                for j in 0..n.saturating_sub(1) {
                    let (text, colour, size) = match (j < i, l[i][j]) {
                        (true, _) if growing.is_some_and(|g| g.0 == i && g.1 == j) => {
                            let (text, size) = cell_text(f, &passing(growing.map_or(0.0, |g| g.2), places), 12.0, lcell.x);
                            (text, EQ[sh.order[i]], size)
                        }
                        (true, Some(v)) => {
                            let (text, size) = cell_text(f, &number(v), 12.0, lcell.x);
                            (text, EQ[sh.order[i]], size)
                        }
                        (true, None) => ("·".to_string(), pal::DIM, 16.0),
                        (false, _) => (String::new(), pal::DIM, 12.0),
                    };
                    f.text_px(at(sh.slot[i], j), text, size, colour.a(sh.ink[i]), Align2::CENTER_CENTER);
                }
            }
        }

        // the marks: the pivot ringed, the number to take out (or bring up) boxed
        let shown_op = if step == 0 { plan.ops.first() } else { plan.ops.get(step - 1) };
        let (ring_at, box_at) = match shown_op {
            Some(Op::Sub { row, from, col, .. }) => (Some(board.cell(sh.slot[*from], *col)), Some(board.cell(sh.slot[*row], *col))),
            // (a swap rings the place where the pivot should be, which stays put while
            // the rows change places, and boxes the number coming up into it)
            Some(Op::Swap { a, b, col }) => (Some(board.cell(*a as f32, *col)), Some(board.cell(sh.slot[*b], *col))),
            Some(Op::Scale { row, col, .. }) => (Some(board.cell(sh.slot[*row], *col)), None),
            None => (None, None),
        };
        self.gliding |= self.ring.follow(ring_at, f.real_dt);
        self.gliding |= self.target.follow(box_at, f.real_dt);
        // (drawn in every frame, if need be where they last were and not to be seen:
        // what is drawn does not come and go, only how strongly)
        // (one faded away entirely is put off the picture: it reappears where it next
        // belongs, not gliding there from where it was)
        let off = Vec2::splat(-1e4);
        let shown_at = |m: &Mark| if m.alpha > 0.0 { m.at } else { off };
        hud_pill(f, shown_at(&self.ring), Vec2::new(cell.x * 0.5 - 1.0, cell.y * 0.5 - 1.0), 1.6, pal::WHITE.a(0.9 * self.ring.alpha));
        let half = Vec2::new(cell.x * 0.48, cell.y * 0.45);
        hud_outline(f, shown_at(&self.target) - half, shown_at(&self.target) + half, 1.3, pal::WHITE.a(0.75 * self.target.alpha));

        // what the step does (written afresh only when something it says has changed)
        let width = max.x - x - 12.0;
        let told_for = (bits(self.system()), step, hinge_seen, answer_seen, width.to_bits());
        if self.told.as_ref().is_none_or(|(was, _)| *was != told_for) || !keeps() {
            let text = if step == 0 { opening_words(plan, width, answer_seen) } else { step_words(plan, step - 1, hinge_seen, answer_seen) };
            self.told = Some((told_for, wrap(&text, SMALL, width)));
        }
        let lines = self.told.as_ref().map_or(&[][..], |(_, lines)| &lines[..]);
        let mut ty = board.at.y + board.size.y + 14.0;
        // (as many lines every time, some of them empty: what is written changes, not
        // how much there is of it)
        for k in 0..LINES {
            f.text_px(Vec2::new(x, ty), lines.get(k).map_or("", String::as_str), SMALL, pal::TEXT, Align2::LEFT_TOP);
            ty += LINE_H;
        }
    }
}

/// A number in a cell of the rows or of L, as a formula, at `size`: one written with a
/// power of ten set smaller where it would not fit across the cell, `room` wide. (Only
/// such a number is measured.)
fn cell_text(f: &Frame, digits: &str, size: f32, room: f32) -> (String, f32) {
    let text = format!("${digits}$");
    if !digits.contains('e') {
        return (text, size);
    }
    let wide = label_width(f, &text, size) / (room - 6.0).max(1.0);
    (text, if wide > 1.0 { size / wide.min(1.5) } else { size })
}

/// How wide a label is drawn: one with mathematics in it as it is set (which costs a
/// look-up when it is drawn in this frame as well), or from the fonts' metrics before
/// the fonts are in place; plain words estimated.
fn label_width(f: &Frame, text: &str, size: f32) -> f32 {
    if !crate::typeset::has_math(text) {
        return text_width(text, size);
    }
    crate::typeset::measure(f.ctx, text, size).map_or_else(|| text_width(text, size), |set| set.x)
}

/// Row `i`'s equation as written now: with the numbers of a step on their way written
/// to a fixed number of places, as over the picture.
fn shown_equation(plan: &Plan, sh: &Shown, i: usize) -> String {
    let places = decimals(plan);
    if sh.changing == Some(i) {
        equation_with(&sh.rows[i], plan.n, &|v| passing(v, places))
    } else {
        equation(&sh.rows[i], plan.n)
    }
}

/// A name the scene places itself in the picture (an equation's, the solution's):
/// which of its candidate places it has, and where it is drawn from its point, gliding
/// to a new place when it must move.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Name {
    place: usize,
    at: Mark,
}

/// How long a name takes to glide to a new place (each of its two eases, in seconds):
/// longer than a mark on the rows, so that a name moved across the picture never goes
/// more than about a twenty-fifth of the way in one frame at 60 frames a second.
const NAME_TIME: f32 = 0.15;

/// The names' places in `Elim::names`: one for each equation, and the solution's.
const SOLUTION_NAME: usize = 3;

impl Elim {
    /// Write `text`, `width` pixels wide, beside the first of the scene's `points` that
    /// has room for it, on the side `towards` if it can be (or one of eight others): its
    /// preferred place whenever that is clear, otherwise kept where it was while it has
    /// a little room there, and otherwise the first place with room (or the roomiest);
    /// glided to when it moves. On a dark plate if `plate`, and then never far from its
    /// point: the plate keeps it legible over the lines there.
    #[allow(clippy::too_many_arguments)]
    fn place_name(&mut self, which: usize, f: &mut Frame, (room, still): (&mut Room, &mut Still), points: &[Vec3], towards: Vec2, gap: f32, (text, width): (&str, f32), size: f32, colour: Color, plate: bool) {
        let anchors: Vec<Vec2> = points.iter().filter_map(|p| on_screen(f, *p)).filter(|p| p.is_finite()).collect();
        let Some(&from) = anchors.first() else { return };
        let half = Vec2::new(width * 0.5 + if plate { 7.0 } else { 4.0 }, size * 0.5 + if plate { 4.0 } else { 1.0 });
        let want = towards.normalize_or(Vec2::X);
        let mut sides: Vec<Vec2> = (0..8).map(|k| Vec2::from_angle(k as f32 * TAU / 8.0)).collect();
        sides.sort_by(|p, q| q.dot(want).total_cmp(&p.dot(want)));
        sides.insert(0, want);
        // (beside each point, on each side, and further out that way where it is crowded
        // close by: the places in the order they are preferred)
        // (a place to one side, the box wholly that side: off a diagonal, wholly in
        // its quarter, clear of lines running across and up through the point)
        let push = |c: f32, h: f32| if c.abs() > 0.3 { c.signum() * h } else { c * h };
        let mut places: Vec<(Vec2, Vec2)> = Vec::new();
        for &at in &anchors {
            for &side in &sides {
                for out in if plate { &[0.0, 10.0][..] } else { &[0.0, 22.0, 48.0, 80.0][..] } {
                    places.push((at, at + side * (gap + *out) + Vec2::new(push(side.x, half.x), push(side.y, half.y))));
                }
            }
        }
        // (each place's room is measured once, and only as far as it is needed)
        let mut measured: Vec<Option<f32>> = vec![None; places.len()];
        let mut clear = |k: usize| *measured[k].get_or_insert_with(|| room.room_for(places[k].0, places[k].1, half));
        let name = &mut self.names[which];
        let kept = name.place;
        if clear(0) >= 8.0 {
            name.place = 0;
        } else if !(kept < places.len() && clear(kept) >= 3.0) {
            // the first with room, or else the roomiest (the first of those)
            name.place = match (0..places.len()).find(|&k| clear(k) >= 5.0) {
                Some(k) => k,
                None => (0..places.len()).map(|k| (k, clear(k))).fold((0, f32::MIN), |best, c| if c.1 > best.1 { c } else { best }).0,
            };
        }
        let goal = places[name.place].1;
        self.gliding |= name.at.follow_in(Some(goal - from), f.real_dt, NAME_TIME);
        let drawn = from + name.at.at;
        // (where it is going: other names keep clear of that, not of every place on its way)
        room.boxed(goal - half, goal + half);
        if plate {
            still.caption_px(f, drawn, text, size, colour, Align2::CENTER_CENTER);
        } else {
            still.text_px(f, drawn, text, size, colour, Align2::CENTER_CENTER);
        }
    }
}

/// The line step `k` turns a plane about, if it does (a step that takes a multiple of
/// one row from another), within the box |x|, |y|, |z| ≤ r: where the two rows'
/// planes meet.
fn hinge(plan: &Plan, k: usize, r: f64) -> Option<(DVec3, DVec3)> {
    let Some(Op::Sub { row, from, .. }) = plan.ops.get(k) else { return None };
    let before = plan.states.get(k)?;
    let (p, q) = (Plane::of(&before[*row], plan.n)?, Plane::of(&before[*from], plan.n)?);
    meeting(&p, &q).and_then(|(at, dir)| line_in_box(at, dir, r))
}

/// The share of a step at either end over which one step's line gives way to the
/// next one's.
const HINGE_FADE: f32 = 0.15;

/// Which steps' lines are shown at a stage, and how strongly: the step playing, and
/// near its start the one before it, the one fading as the other comes; at rest, the
/// step last done.
fn hinges_shown(stage: f32) -> Vec<(usize, f32)> {
    let s = stage.max(0.0);
    let k = s.floor() as usize;
    let frac = s - k as f32;
    let mut out = Vec::new();
    if frac > 0.0 {
        out.push((k, (frac / HINGE_FADE).min(1.0)));
        let before = 1.0 - frac / HINGE_FADE;
        if k >= 1 && before > 0.0 {
            out.push((k - 1, before));
        }
    } else if k >= 1 {
        out.push((k - 1, 1.0));
    }
    out
}

/// The way, on the screen, that each equation's plane is named from (radians, y down):
/// to the left, up and to the right, down and to the right.
const LABEL_WAY: [f32; 3] = [PI * 0.92, -PI * 0.3, PI * 0.15];

/// A point of the polygon `poly` (in the scene) near its edge furthest out the way
/// `way` on the screen: a mean of the points all round its outline, each weighted by
/// how far out it is and how long a piece of the outline it stands for. It moves
/// smoothly where the furthest corner would jump from one to another, and where a
/// corner splits in two as the plane sweeps over a corner of the box.
fn reach_towards(f: &Frame, poly: &[Vec3], way: Vec2) -> Option<Vec3> {
    let seen: Vec<(Vec3, Vec2)> = poly.iter().filter_map(|c| on_screen(f, *c).map(|s| (*c, s))).collect();
    if seen.len() < 2 {
        return None;
    }
    let most = seen.iter().map(|c| c.1.dot(way)).fold(f32::MIN, f32::max);
    let (mut sum, mut weight) = (Vec3::ZERO, 0.0);
    for k in 0..seen.len() {
        let ((a, sa), (b, sb)) = (seen[k], seen[(k + 1) % seen.len()]);
        let length = sa.distance(sb);
        for i in 0..8 {
            let t = (i as f32 + 0.5) / 8.0;
            let w = length / 8.0 * ((sa.lerp(sb, t).dot(way) - most) / 60.0).exp();
            sum += a.lerp(b, t) * w;
            weight += w;
        }
    }
    (weight > 0.0).then(|| sum / weight).filter(|p| p.is_finite())
}

/// The size of the names of the equations in the picture.
const NAME_SIZE: f32 = 13.0;

impl Elim {
    /// Row `i`'s equation as a name in the picture, as written now (a formula), and how
    /// wide a place it is given: while a step changes its numbers, as wide as it is at
    /// its widest through that step, so that its place need not change as its numbers
    /// do. (That widest is measured once a step, and kept.)
    fn equation_label(&mut self, f: &Frame, plan: &Plan, sh: &Shown, i: usize) -> (String, f32) {
        let text = format!("${}$", shown_equation(plan, sh, i));
        let width = label_width(f, &text, NAME_SIZE);
        if sh.playing.is_none() {
            return (text, width);
        }
        let k = self.stage.max(0.0).floor() as usize;
        let Some(op) = plan.ops.get(k) else { return (text, width) };
        let row = match *op {
            Op::Sub { row, .. } | Op::Scale { row, .. } => row,
            Op::Swap { .. } => return (text, width),
        };
        if row != i {
            return (text, width);
        }
        let made_for = (bits(self.system()), k, i);
        let widest = match self.widest {
            Some((was, widest)) if was == made_for && keeps() => widest,
            _ => {
                // (measured as set, or from the fonts' metrics before they are in place:
                // then it is not kept)
                let mut exact = true;
                let widest = (0..=8)
                    .map(|j| {
                        let sh = shown(plan, k as f32 + j as f32 / 8.0);
                        let sample = format!("${}$", if j == 0 || j == 8 { equation(&sh.rows[i], plan.n) } else { shown_equation(plan, &sh, i) });
                        crate::typeset::measure(f.ctx, &sample, NAME_SIZE).map_or_else(
                            || {
                                exact = false;
                                text_width(&sample, NAME_SIZE)
                            },
                            |set| set.x,
                        )
                    })
                    .fold(0.0, f32::max);
                if exact {
                    self.widest = Some((made_for, widest));
                }
                widest
            }
        };
        (text, widest.max(width))
    }
}

/// The multipliers found in the first `done` steps, in L's places, with the rows in
/// their order after those steps: None where it is not known yet, and 0 below a pivot
/// whose column is done where no step was needed.
fn multipliers_after(plan: &Plan, done: usize) -> [[Option<f64>; 3]; 3] {
    let mut l = [[None; 3]; 3];
    // (the pivots before this one have had their columns cleared)
    let mut reached = 0;
    for op in plan.ops.iter().take(done) {
        match *op {
            Op::Sub { row, from, m, .. } => {
                l[row][from] = Some(m);
                reached = reached.max(from);
            }
            Op::Swap { a, b, .. } => {
                for j in 0..a {
                    let t = l[a][j];
                    l[a][j] = l[b][j];
                    l[b][j] = t;
                }
                reached = reached.max(a);
            }
            Op::Scale { .. } => {}
        }
    }
    if done >= plan.down {
        reached = plan.n;
    }
    for (i, row) in l.iter_mut().enumerate() {
        for (j, v) in row.iter_mut().enumerate().take(i) {
            if j < reached && v.is_none() {
                *v = Some(0.0);
            }
        }
    }
    l
}

impl Elim {
    /// Three unknowns: the box, the planes in it, where they meet, and the answer.
    fn space(&mut self, f: &mut Frame, plan: &Plan, sh: &Shown, room: &mut Room, kept: &mut Still) -> Drawn {
        let n = 3;
        let r = self.reach();
        let rf = r as f32;
        let to3 = |p: DVec3| p.as_vec3();
        let mut notes: Vec<String> = Vec::new();

        // the box, and the axes through it
        let corner = |i: usize| Vec3::new(if i & 1 == 0 { -rf } else { rf }, if i & 2 == 0 { -rf } else { rf }, if i & 4 == 0 { -rf } else { rf });
        for i in 0..8 {
            for bit in [1, 2, 4] {
                if i & bit == 0 {
                    let (a, b) = (corner(i), corner(i | bit));
                    f.line(a, b, 1.0, FRAME.a(0.9));
                    room.line(f, a, b);
                }
            }
        }
        for k in 0..3 {
            let d = Vec3::AXES[k];
            f.line(-d * rf, d * rf, 1.2, FRAME.mix(pal::DIM, 0.4));
            room.line(f, -d * rf, d * rf);
            // (and out of the box, to where it is named)
            f.arrow(d * rf, d * rf * 1.22, 1.6, pal::DIM);
            room.line(f, d * rf, d * rf * 1.22);
            // a tick at each whole number, while they are not too close together
            let step = nice_step(rf / 3.5).max(1.0);
            let mut t = step;
            while t < rf * 0.999 {
                for s in [-1.0f32, 1.0] {
                    f.dot(d * t * s, 4.0, FRAME.mix(pal::DIM, 0.5));
                }
                t += step;
            }
        }

        // The planes, and their outlines. Each outline is drawn as its pieces on the
        // faces of the box, every plane's piece on every face in a place of its own in
        // the list of lines, whether it is there or not (if not, it is put behind the
        // viewer, where it is nowhere on the picture): as a plane sweeps over a corner
        // of the box one piece goes as another comes, and the rest should stay what
        // they were, for whoever compares one frame with the next.
        let nowhere = f.cam.eye - f.cam.fwd;
        let planes: Vec<Option<Plane>> = (0..n).map(|i| Plane::of(&sh.geo[i], n)).collect();
        let row_of = sh.row_of();
        for i in row_of.into_iter().take(n) {
            let mut pieces: [Option<(Vec3, Vec3)>; 6] = [None; 6];
            let colour = EQ[sh.order[i]];
            let fade = sh.fade[i];
            if let Some(p) = planes[i] {
                let poly: Vec<Vec3> = p.in_box(r).into_iter().map(to3).collect();
                if poly.len() >= 3 {
                    for w in poly[1..].windows(2) {
                        f.dl.tris_alpha.tri(poly[0], w[0], w[1], colour.a(0.17 * fade));
                    }
                    for (face, a, b) in face_edges(&p, r) {
                        pieces[face] = Some((to3(a), to3(b)));
                        room.line(f, to3(a), to3(b));
                    }
                } else if fade > 0.5 {
                    notes.push(format!("the {} plane is outside the box", EQ_NAMES[sh.order[i]]));
                }
            }
            for piece in pieces {
                let (a, b) = piece.unwrap_or((nowhere, nowhere));
                f.line(a, b, 1.8, colour.a(0.85 * fade));
            }
        }

        // where two planes meet (each pair in its place, likewise)
        for (i, j) in [(0, 1), (0, 2), (1, 2)].map(|(a, b): (usize, usize)| (row_of[a], row_of[b])) {
            let seg = planes[i].zip(planes[j]).and_then(|(p, q)| meeting(&p, &q)).and_then(|(at, dir)| line_in_box(at, dir, r));
            let fade = sh.fade[i].min(sh.fade[j]);
            match seg {
                Some((a, b)) => {
                    f.line(to3(a), to3(b), 1.5, pal::TEXT.a(0.6 * fade));
                    room.line(f, to3(a), to3(b));
                }
                None => f.line(nowhere, nowhere, 1.5, pal::TEXT.a(0.0)),
            }
        }

        // the line the plane of the step shown turns about, highlighted: as a step
        // begins, the last one's fades as this one's comes (in a moment, not at once)
        let hinge_seen = (1..=plan.ops.len()).contains(&self.step()) && hinge(plan, self.step() - 1, r).is_some();
        let mut strongest = (0.0f32, None);
        // (in two places taken in turn by the steps, odd and even, each drawn in every
        // frame and put out of sight when it has nothing to show: so that a step's line
        // is only ever compared with itself from one frame to the next)
        let mut slots: [Option<(Vec3, Vec3, f32)>; 2] = [None; 2];
        for (k, alpha) in hinges_shown(self.stage) {
            let (Some((a, b)), Some(Op::Sub { row, .. })) = (hinge(plan, k, r), plan.ops.get(k)) else { continue };
            let (a, b) = (to3(a), to3(b));
            slots[k % 2] = Some((a, b, alpha));
            room.line(f, a, b);
            if alpha > strongest.0 {
                strongest = (alpha, Some(EQ_NAMES[plan.order[k][*row]]));
            }
        }
        for slot in slots {
            let (a, b, alpha) = slot.unwrap_or((nowhere, nowhere, 0.0));
            f.glow_path([a, b], 4.0, HINGE.a(0.7 * alpha));
            f.line(a, b, 3.0, HINGE.a(alpha));
        }
        if let (_, Some(name)) = strongest {
            notes.insert(0, format!("yellow line: the {name} plane turns about it"));
        }

        // the answer: the point on all three, or the line of them (each named below)
        let mut answer_seen = true;
        let (mut point_at, mut line_at) = (None, None);
        match &plan.outcome {
            Outcome::Point(x) => {
                let p = DVec3::from(*x);
                answer_seen = p.abs().max_element() <= r * 1.0001;
                if answer_seen {
                    let p = to3(p);
                    f.glow(p, 26.0, SOLUTION.a(0.55));
                    f.dot(p, 10.0, SOLUTION);
                    point_at = Some((p, x));
                } else {
                    notes.push(format!("the solution ${}$ is outside the box", point_text(x, n)));
                }
            }
            Outcome::Many { base, dirs, .. } if dirs.len() == 1 => {
                if let Some((a, b)) = line_in_box(DVec3::from(*base), DVec3::from(dirs[0]).normalize_or_zero(), r) {
                    let (a, b) = (to3(a), to3(b));
                    f.glow_path([a, b], 5.0, SOLUTION.a(0.5));
                    f.line(a, b, 3.2, SOLUTION);
                    room.line(f, a, b);
                    line_at = Some((a, b));
                } else {
                    answer_seen = false;
                    notes.push("the line of solutions is outside the box".into());
                }
            }
            _ => {}
        }

        // The names: written as they were in the last frame if nothing they are placed
        // from has changed since, and they had come to rest (see `Still`).
        self.inputs(kept.begin(f));
        if kept.settle(f) {
            kept.replay(f, room);
            return Drawn { hinge_seen, answer_seen, notes };
        }
        let were = self.names;
        if let Some((p, x)) = point_at {
            let text = format!("${}$", point_text(x, n));
            let width = label_width(f, &text, 14.0);
            self.place_name(SOLUTION_NAME, f, (room, kept), &[p], Vec2::new(1.0, -1.0), 14.0, (&text, width), 14.0, SOLUTION, true);
        }
        if let Some((a, b)) = line_at {
            kept.label_any(room, f, "solution", &[a * 0.85 + b * 0.15, a * 0.15 + b * 0.85, (a + b) * 0.5], "every point of it is a solution", 12.5, SOLUTION, Vec2::new(1.0, 0.0), 10.0, false);
        }

        // each plane named by its equation, beside it where there is room (equation by
        // equation, whatever row each is in; not on a canvas too small for the names to
        // be anything but in the way)
        let centre = on_screen(f, Vec3::ZERO).unwrap_or(f.view_size() * 0.5);
        let roomy = f.view_size().min_element() >= 260.0;
        for i in row_of.into_iter().take(n).filter(|_| roomy) {
            let Some(p) = planes[i] else { continue };
            if sh.fade[i] < 0.5 {
                continue;
            }
            let poly: Vec<Vec3> = p.in_box(r).into_iter().map(to3).collect();
            if poly.len() < 3 {
                continue;
            }
            // from the edge of the plane furthest out in its equation's own direction on
            // the screen, or either side of that (points that move smoothly as the plane
            // turns and the view moves, so the name keeps its place)
            let e = sh.order[i];
            let points: Vec<Vec3> = [0.0f32, 0.9, -0.9].iter().filter_map(|turn| reach_towards(f, &poly, Vec2::from_angle(LABEL_WAY[e] + turn))).collect();
            let Some(first) = points.first() else { continue };
            let away = on_screen(f, *first).map_or(Vec2::from_angle(LABEL_WAY[e]), |s| (s - centre).normalize_or(Vec2::from_angle(LABEL_WAY[e])));
            let (text, width) = self.equation_label(f, plan, sh, i);
            self.place_name(e, f, (room, kept), &points, away, 6.0, (&text, width), NAME_SIZE, EQ[e].mix(pal::WHITE, 0.15), false);
        }
        // and the axes by their names, where they leave the box
        for k in 0..3 {
            let d = Vec3::AXES[k] * rf * 1.22;
            let away = on_screen(f, d).zip(on_screen(f, Vec3::ZERO)).map_or(Vec2::X, |(e, o)| (e - o).normalize_or(Vec2::X));
            kept.label(room, f, ("axis", k), d, VAR_LABELS[k], 15.0, pal::TEXT, away, 4.0, false);
        }
        // (they have come to rest when none of them is on its way to a new place, and
        // none has moved since the last frame)
        kept.end(f, self.gliding || self.names != were);
        Drawn { hinge_seen, answer_seen, notes }
    }

    /// Two unknowns: the lines, where they cross, and the answer.
    fn flat(&mut self, f: &mut Frame, plan: &Plan, sh: &Shown, (room, kept): (&mut Room, &mut Still), picture: (Vec2, Vec2)) -> Drawn {
        let n = 2;
        let (lo, hi) = f.cam.flat_bounds();
        let (lo, hi) = (lo.as_dvec2(), hi.as_dvec2());
        // (the lines end just inside the edge of the view, where they can be seen to go)
        let margin = (hi - lo) * 0.001;
        let (blo, bhi) = (lo + margin, hi - margin);
        let to2 = |p: DVec2| p.as_vec2();
        let mut notes = Vec::new();
        // the grid, and the axes: at whole steps of a size that suits how far the picture
        // reaches, the same lines in every frame (however the view is zoomed or moved:
        // those out of it are simply not seen)
        // (each line ends just inside the view, where it can be seen to move; one out of
        // it is drawn as nothing, far off)
        let reach = self.reach();
        let step = nice_step(reach as f32 / 3.0) as f64;
        let most = (reach * 8.0 / step).ceil().min(100.0) as i32;
        let far = to2(hi + (hi - lo) * 10.0);
        for k in 0..2 {
            for i in -most..=most {
                let v = i as f64 * step;
                let inside = v > blo[k] && v < bhi[k];
                let (a, b) = match (inside, k) {
                    (false, _) => (far, far),
                    (true, 0) => (to2(DVec2::new(v, blo.y)), to2(DVec2::new(v, bhi.y))),
                    (true, _) => (to2(DVec2::new(blo.x, v)), to2(DVec2::new(bhi.x, v))),
                };
                f.line(a, b, if i == 0 { 1.4 } else { 1.0 }, if i == 0 { FRAME.mix(pal::DIM, 0.5) } else { FRAME.a(0.35) });
            }
        }
        let lines: Vec<Option<Plane>> = (0..n).map(|i| Plane::of(&sh.geo[i], n)).collect();
        let mut ends: Vec<Option<(DVec2, DVec2)>> = vec![None; n];
        // (each equation's line drawn in every frame, far out of sight when it has none)
        let far = to2(hi + (hi - lo) * 10.0);
        for i in sh.row_of().into_iter().take(n) {
            let colour = EQ[sh.order[i]];
            match lines[i].and_then(|l| line_in_rect(&l, blo, bhi)) {
                Some((a, b)) => {
                    f.line(to2(a), to2(b), 2.6, colour.a(sh.fade[i]));
                    room.line(f, to2(a).extend(0.0), to2(b).extend(0.0));
                    ends[i] = Some((a, b));
                }
                None => {
                    f.line(far, far, 2.6, colour.a(0.0));
                    if lines[i].is_some() && sh.fade[i] > 0.5 {
                        notes.push(format!("the {} line is outside the view", EQ_NAMES[sh.order[i]]));
                    }
                }
            }
        }

        // the point the line of the step shown turns about: the crossing, ringed (in
        // every frame, out of sight when there is none to ring), coming and going as the
        // steps do
        let crossing_of = |k: usize| {
            let Some(Op::Sub { row, from, .. }) = plan.ops.get(k) else { return None };
            let before = &plan.states[k];
            let (p, q) = (Plane::of(&before[*row], n)?, Plane::of(&before[*from], n)?);
            crossing(&p, &q).filter(|c| c.cmpge(lo).all() && c.cmple(hi).all()).map(|c| (c, EQ_NAMES[plan.order[k][*row]]))
        };
        let hinge_seen = (1..=plan.ops.len()).contains(&self.step()) && crossing_of(self.step() - 1).is_some();
        let mut ring: (Option<DVec2>, f32, Option<&str>) = (None, 0.0, None);
        for (k, alpha) in hinges_shown(self.stage) {
            if let Some((c, name)) = crossing_of(k) {
                ring.0 = ring.0.or(Some(c));
                ring.1 += alpha;
                if alpha >= 0.5 {
                    ring.2 = Some(name);
                }
            }
        }
        if let Some(name) = ring.2 {
            notes.insert(0, format!("yellow ring: the crossing the {name} line turns about"));
        }
        let far = to2(hi + (hi - lo) * 10.0);
        let (at, alpha) = (ring.0.map_or(far, to2), ring.1.min(1.0));
        f.glow(at, 40.0, HINGE.a(0.6 * alpha));
        f.circle(at, Vec3::Z, 14.0 * (hi.y - lo.y) as f32 / f.view_size().y, 2.4, HINGE.a(alpha));

        let mut answer_seen = true;
        let mut point_at = None;
        match &plan.outcome {
            Outcome::Point(x) => {
                let p = DVec2::new(x[0], x[1]);
                answer_seen = p.cmpge(lo).all() && p.cmple(hi).all();
                if answer_seen {
                    f.glow(to2(p), 26.0, SOLUTION.a(0.55));
                    f.dot(to2(p), 10.0, SOLUTION);
                    point_at = Some((p, x));
                } else {
                    notes.push(format!("the solution ${}$ is outside the view", point_text(x, n)));
                }
            }
            Outcome::Many { base, dirs, .. } if dirs.len() == 1 => {
                let d = DVec2::new(dirs[0][0], dirs[0][1]);
                let normal = DVec2::new(-d.y, d.x).normalize_or_zero();
                let line = Plane { n: DVec3::new(normal.x, normal.y, 0.0), d: normal.dot(DVec2::new(base[0], base[1])) };
                if let Some((a, b)) = line_in_rect(&line, blo, bhi) {
                    f.glow_path([to2(a), to2(b)], 6.0, SOLUTION.a(0.35));
                    f.line(to2(a), to2(b), 1.2, SOLUTION);
                } else {
                    answer_seen = false;
                }
            }
            _ => {}
        }

        // The names: written as they were in the last frame if nothing they are placed
        // from has changed since, and they had come to rest (see `Still`).
        self.inputs(kept.begin(f));
        if kept.settle(f) {
            kept.replay(f, room);
            f.axes2();
            return Drawn { hinge_seen, answer_seen, notes };
        }
        let were = self.names;
        if let Some((p, x)) = point_at {
            let text = format!("${}$", point_text(x, n));
            let width = label_width(f, &text, 14.0);
            self.place_name(SOLUTION_NAME, f, (room, kept), &[to2(p).extend(0.0)], Vec2::new(1.0, -1.0), 14.0, (&text, width), 14.0, SOLUTION, true);
        }
        // each line named by its equation, beside it near its far end in the picture
        // (from the answer, or the middle): a place that slides smoothly as the line turns
        let (plo, phi) = picture;
        let world = |px: Vec2| DVec2::new(lo.x + (hi.x - lo.x) * (px.x / f.view_size().x.max(1.0)) as f64, hi.y - (hi.y - lo.y) * (px.y / f.view_size().y.max(1.0)) as f64);
        let (wa, wb) = (world(plo + Vec2::new(70.0, 30.0)), world(phi - Vec2::new(70.0, 30.0)));
        let (inner_lo, inner_hi) = (wa.min(wb), wa.max(wb));
        let from = match &plan.outcome {
            Outcome::Point(x) => DVec2::new(x[0], x[1]),
            _ => (inner_lo + inner_hi) * 0.5,
        };
        let roomy = f.view_size().min_element() >= 260.0;
        for i in sh.row_of().into_iter().take(n).filter(|_| roomy) {
            let Some(l) = lines[i] else { continue };
            if sh.fade[i] < 0.5 {
                continue;
            }
            let Some((a, b)) = line_in_rect(&l, inner_lo, inner_hi).or(ends[i]) else { continue };
            let (far, near) = if a.distance(from) >= b.distance(from) { (a, b) } else { (b, a) };
            let points: Vec<Vec3> = [0.0, 0.12, 0.24, 0.36].iter().map(|t| to2(far.lerp(near, *t)).extend(0.0)).collect();
            // (across the line, on its upper side: the label turns with it)
            let across = Vec2::new(l.n.x as f32, -l.n.y as f32);
            let towards = if across.y < 0.0 || (across.y == 0.0 && across.x > 0.0) { across } else { -across };
            let (text, width) = self.equation_label(f, plan, sh, i);
            self.place_name(sh.order[i], f, (room, kept), &points, towards, 8.0, (&text, width), NAME_SIZE, EQ[sh.order[i]].mix(pal::WHITE, 0.15), false);
        }
        // (they have come to rest when none of them is on its way to a new place, and
        // none has moved since the last frame)
        kept.end(f, self.gliding || self.names != were);
        f.axes2();
        Drawn { hinge_seen, answer_seen, notes }
    }

    /// The names the caption gives, for what the picture shows unnamed: what the step
    /// turns about and what is out of view first, since a phone may have room for
    /// only some of them.
    fn caption_parts(&self, plan: &Plan, sh: &Shown, drawn: &Drawn) -> Vec<String> {
        let mut parts = drawn.notes.clone();
        // an equation with nothing on the left has no plane (or line) to show
        let w = words(plan.n);
        for i in 0..plan.n {
            if sh.playing.is_none() && is_empty(&sh.rows[i], plan.n) {
                let b = sh.rows[i][plan.n];
                let says = if b == 0.0 { "true of every point" } else { "true of no point" };
                parts.push(format!("{} row: $0 = {}$, {says}, so no {}", EQ_NAMES[sh.order[i]], number(b), w.thing));
            }
        }
        match &plan.outcome {
            _ if !drawn.answer_seen => {}
            Outcome::Point(_) => parts.push(format!("white dot: the point on {} {}, the solution", if plan.n == 3 { "all three" } else { "both" }, w.things)),
            Outcome::Many { dirs, .. } if dirs.len() == 1 && plan.n == 3 => parts.push("white line: the points that satisfy all three equations".into()),
            Outcome::Many { dirs, .. } if dirs.len() == 1 => parts.push("white line: every point of it satisfies both equations".into()),
            _ => {}
        }
        if plan.n == 3 {
            parts.push("thin white lines: where two planes meet".into());
        }
        parts
    }
}

impl Scene for Elim {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    /// (The ground grid would run on out of the box; the plane draws its own.)
    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    /// The steps as they play, and the marks and the solution's name gliding to their
    /// places.
    fn animated(&self) -> bool {
        self.moving() || self.gliding
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"An equation in $x$, $y$ and $z$, like $2x + y + z = 5$, holds on a plane: the points that satisfy it. Three planes usually meet at one point, and that point satisfies all three equations: the solution. Elimination finds it by changing one equation at a time, each time into itself minus a multiple of another. The point stays on every new plane, so it never moves; but the planes turn, until they read $x = \ldots$, $y = \ldots$, $z = \ldots$. (In two unknowns the equations are lines, and the solution is where they cross: the first experiment.)");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        let plan = self.worked();
        let steps = plan.steps();
        ui::section(ui, "Steps");
        ui.horizontal_wrapped(|ui| {
            if ui.button("◀ back").clicked() {
                self.back();
            }
            if ui.button("next ▶").clicked() {
                self.next(steps);
            }
            if ui.button("↺ replay").clicked() {
                self.replay(steps);
            }
            if ui.button("the end").clicked() {
                (self.stage, self.goal, self.through, self.hold) = (steps as f32, steps as f32, false, 0.0);
            }
        });
        if ui::slider(ui, "step", &mut self.stage, 0.0..=steps as f32) {
            (self.goal, self.through, self.hold) = (self.stage, false, 0.0);
        }
        ui::note(ui, "Next plays one step; replay plays them all, pausing after each. The same buttons are over the picture.");

        ui::section(ui, "The equations");
        let mut k = self.n - 2;
        if ui::choice(ui, &["two unknowns: lines", "three unknowns: planes"], &mut k) {
            self.n = k + 2;
            (self.stage, self.goal, self.through, self.hold) = (0.0, 0.0, false, 0.0);
            self.new_view = true;
        }
        let n = self.n;
        let mut changed = false;
        let sys = self.system_mut();
        egui::Grid::new("elim_numbers").spacing([4.0, 4.0]).show(ui, |ui| {
            for name in VAR_LABELS.iter().take(n) {
                ui.label(ui::label_text(ui, name, 12.0, Some(pal::DIM), false));
            }
            ui.label("");
            ui.label(ui::label_text(ui, "$b$", 12.0, Some(pal::DIM), false));
            ui.end_row();
            for i in 0..n {
                for j in 0..n {
                    changed |= number_box(ui, &mut sys.a[i][j], EQ[i]);
                }
                ui.label(egui::RichText::new("|").size(12.0).color(pal::DIM.egui()));
                changed |= number_box(ui, &mut sys.b[i], EQ[i]);
                ui.end_row();
            }
        });
        if changed {
            let steps = self.worked().steps();
            self.keep_within(steps);
        }
        ui::note(ui, "Row $i$ holds the numbers of equation $i$: those of $A$ on the left, $b$ on the right. Drag a number sideways, or click it and type (two decimals, up to 100).");
        let names: Vec<&str> = if n == 3 { PRESETS3.iter().map(|p| p.0).collect() } else { PRESETS2.iter().map(|p| p.0).collect() };
        let mut all = vec!["new whole numbers"];
        all.extend(names);
        if let Some(i) = ui::buttons(ui, &all) {
            if i == 0 {
                self.random();
            } else if n == 3 {
                self.start(PRESETS3[i - 1].1, BOX);
            } else {
                self.start(PRESETS2[i - 1].1, 3.0);
            }
        }
        let slot = if self.n == 2 { 0 } else { 1 };
        let mut reach = self.reach[slot] as f32;
        if ui::slider_log(ui, if n == 3 { "box: $|x|, |y|, |z|$ up to" } else { "view: $|x|, |y|$ up to" }, &mut reach, 1.0..=100.0) {
            self.reach[slot] = reach as f64;
            self.new_view = true;
        }
        ui::note(ui, "How much of space the picture shows: make it larger for an answer outside it.");
        ui::note(ui, "A number counts as 0 when it is less than $10^{-9}$ times the largest number in $A$: rounding can leave crumbs of up to about $10^{-14}$ times it where there should be nothing. A swap is made only where a pivot is 0. (Programs also swap whenever a row below has a number of larger size there, to keep rounding small.)");

        ui::section(ui, "Reading the picture");
        let w = words(n);
        let first = format!("pink, green, blue: the {} of equations 1, 2, 3 (each keeps its colour when rows change places), labelled with their equations", w.things);
        let mut key: Vec<(Color, &str)> = vec![(EQ[0], &first)];
        let hinge = if n == 3 { "yellow: the line a plane turns about in this step, where it meets the plane of the pivot's row" } else { "yellow: the crossing the line turns about in this step" };
        key.push((HINGE, hinge));
        let sol = if n == 3 { "white dot: the solution, on all three planes; white line: every point of it is a solution" } else { "white dot: the solution, where the lines cross" };
        key.push((SOLUTION, sol));
        if n == 3 {
            key.push((pal::TEXT.a(0.6), "thin white lines: where two planes meet"));
            key.push((FRAME, "grey box: the part of space shown; the axes run through its middle"));
        }
        key.push((pal::WHITE, "over the picture, a white ring round the pivot (or the 0 where one should be), and a white box round the number the step takes out (or, in a swap, brings up); grey numbers are 0; $L$, beside, holds the multipliers as they are used"));
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, "Row operation:", "replacing an equation by itself minus a multiple of another. Every point that satisfied both still does, so the solutions do not change.");
        ui::term(ui, "Pivot:", "the first number of a row that is not 0, used to take its unknown out of the rows below it. Going back up, each row is divided by its pivot.");
        ui::term(ui, "Multiplier:", "how many times the pivot's row is taken away: the number to take out divided by the pivot.");
        ui::term(ui, "Triangular:", "with zeros below the diagonal: the last row holds one unknown, the row above two, and so on.");
        ui::term(ui, "Back substitution:", "going back up a triangular system: the last row gives the last unknown, which is then put into the row above, and so on.");
        ui::term(ui, "$A = LU$:", "$A$ is $L$ (1s on the diagonal, the multipliers below it) times $U$ (the triangular rows). With swaps, $PA = LU$, where $P$ puts the rows of $A$ in their new order.");
        ui::term(ui, "Singular:", r"a matrix with a column that has no pivot. Then there is no solution (a row reads $0 = \text{a number that is not } 0$), or there are infinitely many (every such row reads $0 = 0$), and that column's unknown is free: it can be anything.");

        ui::section(ui, "Numbers");
        let n = plan.n;
        // (the numbers as formulas, every one: a large one is written with a power of ten)
        let pivots: Vec<String> = plan.pivots.iter().map(|p| number(p.2)).collect();
        ui::readout(ui, "pivots", if pivots.is_empty() { "none".into() } else { format!("${}$", pivots.join(", ")) });
        let mut ls = Vec::new();
        for i in 0..n {
            for j in 0..i {
                ls.push(number(plan.l[i][j]));
            }
        }
        ui::readout(ui, if n == 3 { r"multipliers $\ell_{21}, \ell_{31}, \ell_{32}$" } else { r"multiplier $\ell_{21}$" }, format!("${}$", ls.join(", ")));
        // U, the rows once going down is done
        let u = plan.u();
        let rows: Vec<String> = (0..n).map(|i| (0..n).map(|j| number(u[i][j])).collect::<Vec<_>>().join(" & ")).collect();
        ui::equation(ui, &format!(r"U = \begin{{bmatrix}} {} \end{{bmatrix}}", rows.join(r" \\ ")));
        let order: Vec<String> = plan.order[plan.down].iter().take(n).map(|o| (o + 1).to_string()).collect();
        if plan.swaps > 0 {
            ui::readout(ui, "rows of $A$ in the order", order.join(", "));
        }
        ui::readout(ui, r"$\det A = \pm$ the pivots multiplied", format!("${}$", number(plan.det())));
        match &plan.outcome {
            Outcome::Point(x) => ui::readout_colored(ui, if n == 3 { "$(x, y, z)$" } else { "$(x, y)$" }, format!("${}$", point_text(x, n)), SOLUTION),
            Outcome::Impossible { .. } => ui::readout(ui, "solution", "none"),
            Outcome::Many { free, .. } => ui::readout(ui, "free unknowns", format!("${}$", free.iter().map(|f| VAR[*f]).collect::<Vec<_>>().join(", "))),
        }
        ui::note(ui, r"$\det A$ is the product of the pivots, with its sign changed for each swap; 0 when a column has no pivot.");
        ui::see_also(ui, &[("subspaces", "the null space: what a free unknown can do"), ("det", "the determinant, a product of the pivots"), ("overdet", "no solution: settling for the nearest point"), ("condition", "nearly parallel planes: answers that move a lot")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        let plan = self.worked();
        self.keep_within(plan.steps());
        self.advance(f.dt);
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let lay = Layout::of(view, f.top_inset);
        let before = Layout::of(self.canvas, self.inset);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.panel.is_some() != before.panel.is_some() || (lay.narrow && (view - self.canvas).abs().max_element() > 24.0));
        let first = !covered && !self.measured && (view - self.canvas).abs().max_element() > 24.0;
        self.new_view |= changed || first;
        if !covered && (changed || !self.measured) {
            self.canvas = view;
            self.measured = true;
        }
        self.inset = f.top_inset;
        let pointer = f.ptr.pos.map(|p| p / f.px);
        if pointer.is_some_and(|p| lay.covers(p)) {
            // the panel is for its buttons: no turning the view from there, and two
            // quick clicks are two steps, not a reset of the view
            f.handles.capture();
            f.handles.claim_clicks();
        }

        let sh = shown(&plan, self.stage);
        self.gliding = false;
        let mut room = Room::new(f);
        if let Some((min, max)) = lay.panel {
            room.wall(min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        }
        // (the caption's lines are known only once the picture is drawn: the room for
        // as many as it may take is kept clear of labels)
        let bottom = lay.picture.1.y + caption_height(view) - 10.0;
        room.wall(Vec2::new(0.0, lay.picture.1.y), Vec2::new(view.x, bottom + 4.0));
        let mut kept = std::mem::take(&mut self.kept);
        let drawn = if plan.n == 3 { self.space(f, &plan, &sh, &mut room, &mut kept) } else { self.flat(f, &plan, &sh, (&mut room, &mut kept), lay.picture) };
        self.kept = kept;
        self.panel(f, &plan, &sh, &lay, pointer, &drawn);
        // (the caption's lines worked out again only when what it says, or the room for it,
        // changes: its parts are measured, formulas and all)
        let most = caption_most(view);
        let made_for = (self.caption_parts(&plan, &sh, &drawn), (view.x - 36.0).to_bits(), most);
        let lines = match self.captioned.take() {
            Some((was, lines)) if was == made_for && keeps() => lines,
            _ => caption_lines(&made_for.0, view.x - 36.0, most),
        };
        caption(f, &lines, bottom, most);
        self.captioned = Some((made_for, lines));
    }
}

/// One number of the equations, to drag or type: two decimals, within ±100, in its
/// equation's colour. True when it changed.
fn number_box(ui: &mut egui::Ui, v: &mut f64, colour: Color) -> bool {
    let mut x = *v;
    let r = ui.scope(|ui| {
        ui.visuals_mut().override_text_color = Some(colour.mix(pal::WHITE, 0.3).egui());
        ui.add(egui::DragValue::new(&mut x).speed(0.05).fixed_decimals(2).range(-100.0..=100.0))
    });
    let x = (x * 100.0).round() / 100.0;
    if r.inner.changed() && x.is_finite() && x != *v {
        *v = x;
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every system the scene offers: the experiments' and the presets'.
    fn offered() -> Vec<(&'static str, System)> {
        let mut all: Vec<(&str, System)> = TRIALS.iter().map(|t| (t.0, t.2)).collect();
        all.extend(PRESETS3.iter().chain(PRESETS2.iter()).copied());
        all
    }

    /// Random systems with numbers of two decimals, as typed into the panel, and among
    /// them many that are singular on purpose (a row a whole-number mixture of the
    /// others, the right-hand side too or not), with zeros sprinkled in.
    fn typed(rng: &mut Rng, count: usize) -> Vec<System> {
        let mut out = Vec::new();
        let pick = |rng: &mut Rng| ((rng.normal() as f64 * 300.0).round() / 100.0).clamp(-100.0, 100.0);
        for k in 0..count {
            let n = 2 + k % 2;
            let mut sys = System { n, a: [[0.0; 3]; 3], b: [0.0; 3] };
            for i in 0..n {
                for j in 0..n {
                    sys.a[i][j] = if rng.bool(0.15) { 0.0 } else { pick(rng) };
                }
                sys.b[i] = pick(rng);
            }
            if k % 3 == 0 {
                // the last row a mixture of the others
                let (c1, c2) = (rng.below(5) as f64 - 2.0, rng.below(5) as f64 - 2.0);
                let last = n - 1;
                for j in 0..n {
                    sys.a[last][j] = c1 * sys.a[0][j] + if n == 3 { c2 * sys.a[1][j] } else { 0.0 };
                }
                sys.b[last] = c1 * sys.b[0] + if n == 3 { c2 * sys.b[1] } else { 0.0 } + if rng.bool(0.5) { 0.0 } else { 1.0 };
            }
            out.push(sys);
        }
        out
    }

    fn residual(row: &[f64; 4], n: usize, x: &[f64; 3]) -> f64 {
        (0..n).map(|j| row[j] * x[j]).sum::<f64>() - row[n]
    }

    fn det3(a: &[[f64; 3]; 3], n: usize) -> f64 {
        if n == 2 {
            return a[0][0] * a[1][1] - a[0][1] * a[1][0];
        }
        DVec3::from(a[0]).dot(DVec3::from(a[1]).cross(DVec3::from(a[2])))
    }

    /// The arithmetic, for every system offered and thousands typed: no step changes
    /// what solves the equations (the answer satisfies every row at every step); a swap
    /// is made only where the pivot's place holds 0 and a row below does not; PA = LU;
    /// det A is the product of the pivots with a sign for each swap; and the last rows
    /// are the answer, read off.
    #[test]
    fn no_step_changes_what_solves_the_equations() {
        let mut rng = Rng::new(3);
        let mut systems: Vec<System> = offered().into_iter().map(|o| o.1).collect();
        systems.extend(typed(&mut rng, 3000));
        let mut seen = [0usize; 3];
        for sys in systems {
            let plan = Plan::of(&sys);
            let n = plan.n;
            let scale = largest(&plan.states[0], n, true).max(1e-300);
            assert!(plan.states.iter().flatten().flatten().all(|v| v.is_finite()), "{sys:?}");
            // the points that solve the last rows solve every row at every step
            let points: Vec<[f64; 3]> = match &plan.outcome {
                Outcome::Point(x) => vec![*x],
                Outcome::Many { base, dirs, .. } => {
                    let mut p = vec![*base];
                    for d in dirs {
                        p.push(std::array::from_fn(|j| base[j] + 1.5 * d[j]));
                    }
                    p
                }
                Outcome::Impossible { .. } => vec![],
            };
            for x in &points {
                let size = x.iter().fold(1.0f64, |m, v| m.max(v.abs()));
                for (k, state) in plan.states.iter().enumerate() {
                    for i in 0..n {
                        let r = residual(&state[i], n, x);
                        assert!(r.abs() <= 1e-9 * scale * size * 100.0, "{sys:?}: step {k}, row {i} misses {x:?} by {r}");
                    }
                }
            }
            // an impossible row really is: 0 = something, after the steps going down
            if let Outcome::Impossible { row, value } = plan.outcome {
                let last = plan.states.last().unwrap();
                assert!(is_empty(&last[row], n) && last[row][n] == value && value != 0.0 && plan.ops.len() == plan.down);
            }
            // swaps only where needed
            for (k, op) in plan.ops.iter().enumerate() {
                if let Op::Swap { a, b, col } = *op {
                    let before = &plan.states[k];
                    assert!(before[a][col] == 0.0 && before[b][col] != 0.0 && (a + 1..b).all(|r| before[r][col] == 0.0), "{sys:?}: swap {k}");
                }
            }
            // P A = L U, with U the rows once going down is done
            let u = plan.u();
            let order = plan.order[plan.down];
            for i in 0..n {
                for j in 0..n {
                    let lu: f64 = (0..n).map(|k| plan.l[i][k] * u[k][j]).sum();
                    let pa = sys.a[order[i]][j];
                    assert!((lu - pa).abs() <= 1e-9 * scale, "{sys:?}: (PA)[{i}][{j}] = {pa}, (LU) = {lu}");
                }
                assert!((0..i).all(|k| u[i][k] == 0.0) && plan.l[i][i] == 1.0 && (i + 1..n).all(|k| plan.l[i][k] == 0.0));
            }
            // the determinant
            let det = det3(&sys.a, n);
            assert!((plan.det() - det).abs() <= 1e-9 * scale.powi(n as i32), "{sys:?}: {} against {det}", plan.det());
            // the last rows: 1 at each pivot, 0 above and below it
            let last = plan.states.last().unwrap();
            if !matches!(plan.outcome, Outcome::Impossible { .. }) {
                for &(r, c, _) in &plan.pivots {
                    assert!(last[r][c] == 1.0 && (0..n).all(|i| i == r || last[i][c] == 0.0), "{sys:?}: column {c} is not cleared");
                }
            }
            seen[match plan.outcome {
                Outcome::Point(_) => 0,
                Outcome::Impossible { .. } => 1,
                Outcome::Many { .. } => 2,
            }] += 1;
        }
        // (the singular ones, both kinds, are well represented)
        assert!(seen.iter().all(|&s| s > 300), "{seen:?}");
    }

    /// Each step that takes a multiple of one row from another turns that row's plane
    /// about the line where it meets the other: every point of that line (and the
    /// answer) is on the plane all the way through the step. Going down, the unknown
    /// taken out leaves the plane parallel to its axis. A swap and a division move no
    /// plane at all.
    #[test]
    fn a_plane_turns_about_the_line_it_shares() {
        let mut rng = Rng::new(5);
        let mut systems: Vec<System> = offered().into_iter().map(|o| o.1).collect();
        systems.extend(typed(&mut rng, 600));
        let mut turned = 0;
        for sys in systems {
            let plan = Plan::of(&sys);
            let n = plan.n;
            for (k, op) in plan.ops.iter().enumerate() {
                let before = &plan.states[k];
                for t in [0.2f32, 0.5, 0.8] {
                    let sh = shown(&plan, k as f32 + t);
                    assert_eq!(sh.playing.map(|p| p.0), Some(k));
                    match *op {
                        Op::Sub { row, from, col, .. } => {
                            let (Some(p), Some(q)) = (Plane::of(&before[row], n), Plane::of(&before[from], n)) else { continue };
                            let Some(moving) = Plane::of(&sh.geo[row], n) else { continue };
                            if n == 3 {
                                if let Some((at, dir)) = meeting(&p, &q) {
                                    for s in [-4.0, 0.0, 3.0] {
                                        let x = at + dir * s;
                                        assert!((moving.n.dot(x) - moving.d).abs() < 1e-6 * (1.0 + at.length()), "{sys:?}: step {k} at {t}");
                                    }
                                    turned += 1;
                                }
                            } else if let Some(c) = crossing(&p, &q) {
                                assert!((moving.n.x * c.x + moving.n.y * c.y - moving.d).abs() < 1e-6 * (1.0 + c.length()), "{sys:?}: step {k} at {t}");
                                turned += 1;
                            }
                            if k < plan.down {
                                assert!(plan.states[k + 1][row][col] == 0.0);
                            }
                        }
                        Op::Swap { .. } | Op::Scale { .. } => {
                            assert_eq!(sh.geo, signed_rows(before, faces(&plan, k)), "{sys:?}: step {k} moves a plane");
                        }
                    }
                }
            }
        }
        assert!(turned > 1000, "{turned}");
    }

    /// The planes as drawn change smoothly at every step's start and end, and keep the
    /// way they face: divided by a negative pivot, a row's numbers all change sign, but
    /// its plane is still drawn facing the same way (or its pieces would be drawn with
    /// their ends the other way round, from one frame to the next).
    #[test]
    fn the_planes_drawn_keep_the_way_they_face() {
        let mut rng = Rng::new(15);
        let mut systems: Vec<System> = offered().into_iter().map(|o| o.1).collect();
        systems.extend(typed(&mut rng, 300));
        let mut flips = 0;
        for sys in systems {
            let plan = Plan::of(&sys);
            let n = plan.n;
            for k in 0..=plan.ops.len() {
                for (a, b) in [(k as f32 - 1e-4, k as f32), (k as f32, k as f32 + 1e-4)] {
                    if a < 0.0 || b > plan.ops.len() as f32 {
                        continue;
                    }
                    let (sa, sb) = (shown(&plan, a), shown(&plan, b));
                    for e in 0..n {
                        let (i, j) = (sa.row_of()[e], sb.row_of()[e]);
                        if let (Some(p), Some(q)) = (Plane::of(&sa.geo[i], n), Plane::of(&sb.geo[j], n)) {
                            assert!(p.n.dot(q.n) > 0.999 && (p.d - q.d).abs() < 1e-2 * (1.0 + p.d.abs()), "{sys:?}: equation {e} at {a} and {b}");
                        }
                    }
                }
            }
            flips += plan.ops.iter().filter(|op| matches!(op, Op::Scale { p, .. } if *p < 0.0)).count();
        }
        assert!(flips > 100, "{flips}");
    }

    /// The part of a plane inside the box: its corners are on the plane and in the box,
    /// there are 3 to 6 of them, and the polygon is convex; a plane that misses the box
    /// has none. Where two planes meet, inside the box, is on both.
    #[test]
    fn planes_are_cut_to_the_box() {
        let mut rng = Rng::new(9);
        let r = 3.0;
        let mut empty = 0;
        for _ in 0..4000 {
            let n = DVec3::new(rng.normal() as f64, rng.normal() as f64, rng.normal() as f64);
            let d = rng.normal() as f64 * 4.0;
            let Some(p) = Plane::of(&[n.x, n.y, n.z, d], 3) else { continue };
            let poly = p.in_box(r);
            if poly.is_empty() {
                // (it misses the box: every corner of the box is on one side of it)
                let sides: Vec<bool> = (0..8).map(|i| p.n.dot(DVec3::new(if i & 1 == 0 { -r } else { r }, if i & 2 == 0 { -r } else { r }, if i & 4 == 0 { -r } else { r })) > p.d).collect();
                assert!(sides.iter().all(|s| *s == sides[0]), "{p:?}");
                empty += 1;
                continue;
            }
            assert!((3..=6).contains(&poly.len()), "{} corners", poly.len());
            for (i, c) in poly.iter().enumerate() {
                assert!((p.n.dot(*c) - p.d).abs() < 1e-9 && c.abs().max_element() <= r * (1.0 + 1e-9), "{c}");
                let (a, b) = (poly[(i + 1) % poly.len()], poly[(i + 2) % poly.len()]);
                assert!((a - *c).cross(b - a).dot(p.n) * (poly[1] - poly[0]).cross(poly[2] - poly[1]).dot(p.n) >= -1e-9, "not convex");
            }
            let q = Plane::of(&[rng.normal() as f64, rng.normal() as f64, rng.normal() as f64, rng.normal() as f64], 3).unwrap();
            if let Some((a, b)) = meeting(&p, &q).and_then(|(at, dir)| line_in_box(at, dir, r)) {
                for x in [a, b, (a + b) * 0.5] {
                    assert!((p.n.dot(x) - p.d).abs() < 1e-9 && (q.n.dot(x) - q.d).abs() < 1e-9 && x.abs().max_element() <= r * (1.0 + 1e-9));
                }
            }
        }
        assert!(empty > 100, "{empty}");
        // the outline drawn face by face is the polygon's: as many pieces as corners,
        // each from one corner to the next
        for _ in 0..500 {
            let Some(p) = Plane::of(&[rng.normal() as f64, rng.normal() as f64, rng.normal() as f64, rng.normal() as f64 * 2.0], 3) else { continue };
            let (poly, edges) = (p.in_box(r), face_edges(&p, r));
            assert_eq!(poly.len(), edges.len(), "{p:?}");
            for (_, a, b) in edges {
                let near = |q: DVec3| poly.iter().any(|c| c.distance(q) < 1e-9);
                assert!(near(a) && near(b) && (p.n.dot(a) - p.d).abs() < 1e-9, "{p:?}");
            }
        }
        // a line, for two unknowns (whose b is the third number of the row)
        let line = Plane::of(&[1.0, 1.0, 3.0, 0.0], 2).unwrap();
        let (a, b) = line_in_rect(&line, DVec2::splat(-4.0), DVec2::splat(4.0)).unwrap();
        assert!((a.x + a.y - 3.0).abs() < 1e-12 && (b.x + b.y - 3.0).abs() < 1e-12 && (a.distance(b) - 50f64.sqrt()).abs() < 1e-9, "{a} {b}");
        assert!(line_in_rect(&Plane::of(&[1.0, 0.0, 9.0, 0.0], 2).unwrap(), DVec2::splat(-4.0), DVec2::splat(4.0)).is_none());
        assert!(Plane::of(&[0.0, 0.0, 0.0, 1.0], 3).is_none());
    }

    /// An experiment as its button sets it up, played to rest at 60 frames a second.
    fn run(k: usize) -> Elim {
        let mut s = Elim::create();
        s.set_up(k);
        for _ in 0..60 * 60 {
            s.advance(1.0 / 60.0);
        }
        assert!(!s.moving() && s.stage == s.plan().steps() as f32, "{}", TRIALS[k].0);
        s
    }

    fn rows_read(plan: &Plan, k: usize) -> Vec<String> {
        (0..plan.n).map(|i| equation(&plan.states[k][i], plan.n)).collect()
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // two lines crossing at (1, 2); row 2 − 2 × row 1 takes x out of the green row,
        // which turns about the crossing until it is level, −3y = −6; the pivot −3 gives
        // y = 2; row 1 − row 2 leaves x = 1; the crossing never moves
        let s = run(0);
        let plan = s.plan();
        assert!(plan.n == 2 && s.n == 2 && plan.outcome == Outcome::Point([1.0, 2.0, 0.0]));
        assert_eq!(plan.ops, [Op::Sub { row: 1, from: 0, m: 2.0, col: 0 }, Op::Scale { row: 1, p: -3.0, col: 1 }, Op::Sub { row: 0, from: 1, m: 1.0, col: 1 }]);
        assert_eq!(EQ_NAMES[plan.order[0][1]], "green");
        assert_eq!(rows_read(&plan, 1)[1], "−3y = −6");
        assert_eq!(standing(&plan.states[1][1], 2), "level (parallel to the $x$ axis)");
        assert_eq!(rows_read(&plan, 2)[1], "y = 2");
        assert_eq!(rows_read(&plan, 3), ["x = 1", "y = 2"]);
        assert_eq!(step_title(&plan, 0), r"$\text{row } 2 - 2 \times \text{row } 1$");

        // three planes meeting at (−1, 2, 1): after three steps the blue plane is level,
        // 5z = 5; going back up gives z = 1, then y = 2, then x = −1; pivots 1, 5, 5;
        // multipliers 3, −2, −1 make up L
        let s = run(1);
        let plan = s.plan();
        assert!(plan.outcome == Outcome::Point([-1.0, 2.0, 1.0]) && plan.down == 3);
        assert_eq!((rows_read(&plan, 3)[2].as_str(), EQ_NAMES[plan.order[3][2]]), ("5z = 5", "blue"));
        assert_eq!(standing(&plan.states[3][2], 3), "parallel to the $x$ and $y$ axes");
        // (each unknown is found, alone in its row with 1 before it, in that order)
        let found = |c: usize| plan.states.iter().position(|m| (0..3).any(|r| m[r][c] == 1.0 && (0..3).all(|j| j == c || m[r][j] == 0.0))).unwrap_or(usize::MAX);
        assert!(found(2) < found(1) && found(1) < found(0) && found(0) < usize::MAX, "{} {} {}", found(2), found(1), found(0));
        assert_eq!(rows_read(&plan, plan.states.len() - 1), ["x = −1", "y = 2", "z = 1"]);
        assert_eq!(plan.pivots.iter().map(|p| p.2).collect::<Vec<_>>(), [1.0, 5.0, 5.0]);
        assert_eq!(plan.l, [[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [-2.0, -1.0, 1.0]]);
        assert_eq!(answer_title(&plan), r"$x = −1,\quad y = 2,\quad z = 1$");

        // a zero pivot: once x is out row 2 reads −3z = −6, with 0 in the second
        // pivot's place; row 3 has 4 there; they swap, the planes do not move, and the
        // green equation is now row 3; then z = 2, y = 1, x = −1
        let s = run(2);
        let plan = s.plan();
        let swap = plan.ops.iter().position(|op| matches!(op, Op::Swap { .. })).unwrap_or(usize::MAX);
        assert_eq!((swap, plan.ops[2]), (2, Op::Swap { a: 1, b: 2, col: 1 }));
        assert_eq!(rows_read(&plan, 2)[1], "−3z = −6");
        assert!(plan.states[2][1][1] == 0.0 && plan.states[2][2][1] == 4.0);
        assert_eq!(EQ_NAMES[plan.order[3][2]], "green");
        assert!((0..3).all(|i| Plane::of(&plan.states[2][i], 3).map(|p| (p.n, p.d)) == Plane::of(&plan.states[3][plan.order[3].iter().position(|&e| e == plan.order[2][i]).unwrap()], 3).map(|p| (p.n, p.d))));
        assert_eq!(plan.outcome, Outcome::Point([-1.0, 1.0, 2.0]));
        assert_eq!(plan.swaps, 1);

        // no solution: the planes meet two at a time in three parallel lines, apart from
        // each other; once x is out of rows 2 and 3 their planes are parallel; row 3 −
        // row 2 leaves 0 = 4, and the blue plane slides away out of the box
        let s = run(3);
        let plan = s.plan();
        let first: Vec<Plane> = (0..3).filter_map(|i| Plane::of(&plan.states[0][i], 3)).collect();
        let lines: Vec<(DVec3, DVec3)> = [(0, 1), (0, 2), (1, 2)].iter().filter_map(|&(i, j)| meeting(&first[i], &first[j])).collect();
        assert_eq!(lines.len(), 3);
        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let ((p, d), (q, e)) = (lines[i], lines[j]);
            let apart = (q - p - d * d.dot(q - p)).length();
            assert!(d.cross(e).length() < 1e-12 && apart > 1.0, "lines {i} and {j}: {apart} apart");
        }
        assert!(matches!(plan.outcome, Outcome::Impossible { row: 2, value } if value == 4.0));
        assert!(parallel(&plan.states[2][1], &plan.states[2][2], 3) && plan.states[2][1][3] != plan.states[2][2][3]);
        assert_eq!(plan.ops[2], Op::Sub { row: 2, from: 1, m: 1.0, col: 1 });
        assert_eq!(rows_read(&plan, 3)[2], "0 = 4");
        assert_eq!(EQ_NAMES[plan.order[2][2]], "blue");
        let sliding = shown(&plan, 2.9);
        assert!(Plane::of(&sliding.geo[2], 3).is_some_and(|p| p.in_box(s.reach()).is_empty()));
        assert!(answer_words(&plan, true).ends_with("The planes meet two at a time, in three parallel lines, but never all three at once."));

        // a line of solutions: the same left-hand sides with 3 in place of 7; all three
        // planes hold one line; row 3 − row 2 leaves 0 = 0, and z is free with x = 1 − z
        // and y = 1 + z
        let s = run(4);
        let plan = s.plan();
        assert!(LINE.a == NONE.a && LINE.b[..2] == NONE.b[..2] && (NONE.b[2], LINE.b[2]) == (7.0, 3.0));
        let Outcome::Many { free, base, dirs } = &plan.outcome else { panic!("{:?}", plan.outcome) };
        assert!(free == &[2] && dirs.len() == 1);
        for t in [-2.0, 0.0, 1.5] {
            let x: [f64; 3] = std::array::from_fn(|j| base[j] + t * dirs[0][j]);
            assert!((0..3).all(|i| residual(&plan.states[0][i], 3, &x).abs() < 1e-12));
        }
        assert_eq!(plan.ops[2], Op::Sub { row: 2, from: 1, m: 1.0, col: 1 });
        assert_eq!(rows_read(&plan, 3)[2], "0 = 0");
        assert_eq!((solved_for(&plan, 0), solved_for(&plan, 1)), ("x = 1 − z".to_string(), "y = 1 + z".to_string()));
        assert!(answer_words(&plan, true).contains("$z$ is free: for any $z$, $x = 1 − z$ and $y = 1 + z$."));
        // (and the line is in the box, to be seen)
        assert!(line_in_box(DVec3::from(*base), DVec3::from(dirs[0]).normalize(), s.reach()).is_some_and(|(a, b)| a.distance(b) > 4.0));
    }

    /// The opening picture: the second experiment's planes, before the first step; and
    /// the planes of every experiment well apart at every step but where they are
    /// meant to be parallel (at least 45° apart for the experiments with one answer,
    /// whose numbers were chosen for it, and 30° for the singular ones, whose rows of
    /// A start 60° apart), none seen nearly edge on from where it is looked at.
    #[test]
    fn the_planes_are_seen_well() {
        let s = Elim::create();
        assert!(s.three == THREE && s.n == 3 && s.stage == 0.0);
        for (k, (name, _, sys, _, look)) in TRIALS.iter().enumerate().filter(|(_, t)| t.2.n == 3) {
            let plan = Plan::of(sys);
            let apart = if matches!(plan.outcome, Outcome::Point(_)) { 45.0 } else { 29.9 };
            if apart < 45.0 {
                let a = &plan.states[0];
                for (i, j) in [(0, 1), (0, 2), (1, 2)] {
                    let (p, q) = (Plane::of(&a[i], 3).unwrap(), Plane::of(&a[j], 3).unwrap());
                    assert!((p.n.dot(q.n).abs().acos().to_degrees() - 60.0).abs() < 1e-9, "{name}");
                }
            }
            let (yaw, pitch) = *look;
            let eye = DVec3::new((pitch.cos() * yaw.cos()) as f64, (pitch.cos() * yaw.sin()) as f64, pitch.sin() as f64);
            let mut least = f64::MAX;
            for state in &plan.states {
                let planes: Vec<Plane> = (0..3).filter_map(|i| Plane::of(&state[i], 3)).collect();
                for (i, p) in planes.iter().enumerate() {
                    least = least.min(p.n.dot(eye).abs());
                    for q in &planes[i + 1..] {
                        let angle = p.n.dot(q.n).abs().min(1.0).acos().to_degrees();
                        assert!(angle > apart || angle < 1e-6, "{name}: two planes {angle:.1}° apart");
                    }
                }
            }
            assert!(least > 0.3, "experiment {k}, {name}: a plane is seen nearly edge on ({least})");
        }
    }

    /// The words: every sentence of every step, for every system offered and many
    /// typed, fits the panel over the picture (on the narrowest canvas it is shown on),
    /// says no "NaN" or "inf", and the titles fit too.
    #[test]
    fn the_words_fit_the_panel() {
        let width = Layout::of(Vec2::new(300.0, 640.0), 46.0).panel.map_or(f32::MAX, |(min, max)| max.x - min.x - 26.0).min(PANEL_W - 26.0);
        let mut rng = Rng::new(13);
        let mut systems: Vec<System> = offered().into_iter().map(|o| o.1).collect();
        systems.extend(typed(&mut rng, 400));
        for sys in systems {
            let plan = Plan::of(&sys);
            let mut texts = vec![opening_words(&plan, width, true), opening_words(&plan, width, false)];
            for k in 0..plan.steps() {
                for seen in [true, false] {
                    texts.push(step_words(&plan, k, seen, seen));
                }
            }
            for text in texts {
                let lines = wrap(&text, SMALL, width);
                assert!(lines.len() <= LINES && lines.iter().all(|l| text_width(l, SMALL) <= width + 30.0), "{sys:?}: {} lines: {text}", lines.len());
                assert!(!text.contains("NaN") && !text.contains("inf") && !text.contains('?'), "{text}");
            }
            for k in 0..plan.steps() {
                let title = step_title(&plan, k);
                assert!(text_width(&title, TITLE) <= width, "{title}");
            }
        }
    }

    /// The pivot that counts as 0 is the stated one: under 10⁻⁹ of the largest number
    /// in A (and then a row below with a number there comes up), while a pivot of 0.01
    /// among numbers near 1 is used, with the words saying it is small but not 0.
    #[test]
    fn a_small_pivot_is_not_zero_and_crumbs_are() {
        let plan = Plan::of(&PRESETS3[0].1);
        assert_eq!(plan.pivots[0].2, 0.01);
        assert!(plan.ops.iter().all(|op| !matches!(op, Op::Swap { .. })));
        assert!(step_words(&plan, 0, true, true).contains("The pivot 0.01 is small but not 0, so there is no swap."));
        assert_eq!(plan.ops[0], Op::Sub { row: 1, from: 0, m: 100.0, col: 0 });
        assert!(matches!(plan.outcome, Outcome::Point(x) if x.iter().all(|v| (v - 1.0).abs() < 1e-12)));
        // a pivot of a hundredth of the threshold is 0, and is swapped away; one of a
        // hundred times it is used
        for (size, swapped) in [(1e-11, true), (1e-7, false)] {
            let sys = System::three([[size, 1.0, 1.0], [1.0, 1.0, 2.0], [1.0, 2.0, 1.0]], [1.0, 2.0, 3.0]);
            let plan = Plan::of(&sys);
            assert_eq!(matches!(plan.ops.first(), Some(Op::Swap { .. })), swapped || plan.states[0][0][0] == 0.0, "{size}");
        }
        // rounding leaves crumbs far below the threshold: whatever is set to 0 for being
        // under it, in singular systems typed with two decimals, is up to about 10⁻¹⁴ of
        // the largest number
        let mut rng = Rng::new(21);
        let mut worst: f64 = 0.0;
        for sys in typed(&mut rng, 3000).into_iter().step_by(3) {
            let n = sys.n;
            let mut m = sys.rows();
            let scale = largest(&m, n, false);
            // elimination as the plan does it, without making anything 0, to see the crumbs
            let plan = Plan::of(&sys);
            for op in plan.ops.iter().take(plan.down) {
                match *op {
                    Op::Sub { row, from, m: mult, col } => {
                        for j in 0..=n {
                            if j != col {
                                m[row][j] -= mult * m[from][j];
                            }
                        }
                        m[row][col] = 0.0;
                    }
                    Op::Swap { a, b, .. } => m.swap(a, b),
                    Op::Scale { .. } => {}
                }
            }
            for i in plan.pivots.len()..n {
                for j in 0..n {
                    worst = worst.max(m[i][j].abs() / scale.max(1e-300));
                }
            }
        }
        println!("the largest crumb: {worst:e} of the largest number");
        assert!(worst > 1e-16 && worst < 2e-14, "{worst}");
        // (and a number typed into the panel is at least a ten-thousandth of the largest)
        const { assert!(0.01 / 100.0 > ZERO * 1e4) };
    }

    /// The steps: next, back and replay go where they say, at any frame rate, with a
    /// pause after each step only when playing through; an experiment waits a moment
    /// before its first step; and the scene is still exactly when nothing moves.
    #[test]
    fn the_steps_play_and_come_to_rest() {
        let s = Elim::create();
        let steps = s.plan().steps();
        assert!(s.stage == 0.0 && !s.moving() && s.step() == 0 && steps == 8);
        for hz in [30.0f32, 60.0, 144.0, 240.0] {
            let mut s = Elim::create();
            s.replay(steps);
            let whole = HOLD * 0.6 + steps as f32 * STEP_TIME + (steps - 1) as f32 * HOLD;
            let (mut t, dt) = (0.0, 1.0 / hz);
            let mut seen = Vec::new();
            while s.moving() && t < 60.0 {
                s.advance(dt);
                t += dt;
                if (t - (HOLD * 0.6 + 2.0 * STEP_TIME + HOLD * 1.5)).abs() < 0.5 * dt {
                    seen.push(s.stage);
                }
            }
            // (a frame or two either way: the time is counted in f32)
            assert!((t - whole).abs() <= dt * 2.01 && s.stage == steps as f32 && !s.through, "{hz} Hz: {t} s, not {whole}");
            assert_eq!(seen, [2.0], "{hz} Hz: not waiting on step 2");
        }
        // an experiment first shows the equations as given
        let mut s = Elim::create();
        s.set_up(3);
        s.advance(FIRST_HOLD - 0.05);
        assert!(s.stage == 0.0 && s.moving());
        // one step on, one back; at the ends nothing more
        let mut s = Elim::create();
        s.next(steps);
        s.advance(STEP_TIME * 0.5);
        assert!(s.stage > 0.4 && s.stage < 0.6 && s.moving() && s.step() == 1);
        s.advance(STEP_TIME);
        assert!(s.stage == 1.0 && !s.moving());
        s.back();
        s.advance(BACK_TIME + 0.01);
        assert!(s.stage == 0.0 && !s.moving());
        s.back();
        assert!(!s.moving());
        (s.stage, s.goal) = (steps as f32, steps as f32);
        assert!(!s.can_go_on(steps));
        // paused, nothing moves
        s.back();
        s.advance(0.0);
        assert!(s.stage == steps as f32 && s.moving());
        // equations with fewer steps bring the stage within them
        s.keep_within(3);
        assert!(s.stage == 3.0 && s.goal <= 3.0);
    }

    /// The marks on the rows set off and arrive gently, and are exactly there in the
    /// end, whatever the display's rate; one with nowhere to be fades away.
    #[test]
    fn the_marks_glide_and_settle() {
        for hz in [60.0f32, 144.0] {
            let mut m = Mark::default();
            assert!(!m.follow(Some(Vec2::new(10.0, 10.0)), 1.0 / hz) || m.at == Vec2::new(10.0, 10.0));
            let to = Vec2::new(10.0, 52.0);
            let mut steps = Vec::new();
            let mut was = m.at;
            for _ in 0..(hz as usize) {
                m.follow(Some(to), 1.0 / hz);
                steps.push(m.at.distance(was));
                was = m.at;
            }
            assert_eq!(m.at, to, "{hz} Hz");
            // (no step much bigger than the ones beside it: it eases in and out)
            assert!(steps.windows(3).all(|w| w[1] <= 2.5 * w[0].max(w[2]) + 0.5), "{hz} Hz: {steps:?}");
            for _ in 0..(hz as usize) {
                m.follow(None, 1.0 / hz);
            }
            assert_eq!(m.alpha, 0.0);
        }
    }

    /// The panel keeps to every canvas and clear of the application's buttons, with
    /// room left for the picture, and the rows fit in it; and the view is finite.
    #[test]
    fn the_panel_and_the_picture_fit_every_canvas() {
        for (x, y) in [(1080.0f32, 1000.0f32), (580.0, 700.0), (390.0, 800.0), (360.0, 640.0), (844.0, 390.0), (1700.0, 800.0), (600.0, 400.0), (240.0, 180.0), (300.0, 600.0)] {
            let inset = if x < 900.0 { 46.0 } else { 0.0 };
            let lay = Layout::of(Vec2::new(x, y), inset);
            let (pmin, pmax) = lay.picture;
            assert!(pmin.is_finite() && pmax.is_finite() && pmax.x > pmin.x && pmax.y > pmin.y, "{x}x{y}: {lay:?}");
            if let Some((min, max)) = lay.panel {
                assert!(min.x >= 0.0 && min.y >= inset && max.x <= x && max.y <= y && max.y - min.y >= PANEL_H - 1.0, "{x}x{y}: {lay:?}");
                assert!(pmax.y - pmin.y >= 200.0 && pmax.x - pmin.x >= 200.0, "{x}x{y}: too little left for the picture: {lay:?}");
                for (_, sys) in offered() {
                    let board = Rowboard::of(&Plan::of(&sys), (min, max), min.y + 100.0);
                    let right = board.l_at.map_or(board.at.x + board.size.x, |l| l.x + matrix_size(sys.n, sys.n, board.l_widest, 12.0).x);
                    assert!(right <= max.x - 8.0 && board.at.y + board.size.y + 14.0 + LINES as f32 * LINE_H <= max.y + 2.0, "{x}x{y}");
                }
            }
            assert!((x, y) != (1080.0, 1000.0) || lay.panel.is_some());
            assert!((x, y) != (390.0, 800.0) || lay.narrow && lay.panel.is_some());
            for n in [2, 3] {
                let mut s = Elim::create();
                s.n = n;
                (s.canvas, s.inset) = (Vec2::new(x, y), inset);
                match s.the_view() {
                    View::Orbit { target, dist, .. } => assert!(target.is_finite() && dist.is_finite() && dist > 0.0),
                    View::Fit { center, half } => assert!(center.is_finite() && half.is_finite() && half.min_element() > 0.0),
                    View::Flat { .. } => panic!("not a view the scene uses"),
                }
            }
        }
    }

    /// Degenerate equations are normal: a row of zeros (0 = 0 and 0 = 1), every
    /// number 0, two planes the same, parallel ones, huge and tiny numbers, all as
    /// both two and three unknowns. Everything drawn or said is finite, and what is
    /// said is what happened.
    #[test]
    fn degenerate_equations_leave_everything_finite() {
        let mut cases: Vec<System> = offered().into_iter().map(|o| o.1).collect();
        cases.extend([
            System::three([[0.0; 3]; 3], [0.0; 3]),
            System::three([[0.0; 3]; 3], [0.0, 1.0, 0.0]),
            System::three([[1.0, 2.0, 3.0], [0.0; 3], [0.0; 3]], [1.0, 0.0, 0.0]),
            System::three([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [-1.0, -2.0, -3.0]], [1.0, 2.0, -1.0]),
            System::three([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], [1.0, 2.0, 3.0]),
            System::three([[100.0, -100.0, 100.0], [-100.0, 100.0, 99.99], [100.0, 99.99, -100.0]], [100.0, -100.0, 100.0]),
            System::three([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]], [100.0, -100.0, 100.0]),
            System::two([[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0]),
            System::two([[0.0, 0.0], [0.0, 0.0]], [1.0, 0.0]),
            System::two([[0.0, 1.0], [0.0, 2.0]], [1.0, 2.0]),
            System::two([[100.0, 99.99], [99.99, 99.98]], [1.0, 1.0]),
        ]);
        for sys in cases {
            let plan = Plan::of(&sys);
            let n = plan.n;
            let mut texts = vec![intro_words(&plan, true), intro_words(&plan, false), answer_title(&plan)];
            for k in 0..plan.steps() {
                texts.push(step_words(&plan, k, true, true));
                texts.push(step_title(&plan, k));
            }
            for c in 0..n {
                texts.push(solved_for(&plan, c));
            }
            for text in &texts {
                assert!(!text.contains("NaN") && !text.contains("inf") && !text.contains('?') && !text.is_empty(), "{sys:?}: {text}");
            }
            for k in 0..=plan.ops.len() * 4 {
                let sh = shown(&plan, k as f32 * 0.25);
                assert!(sh.rows.iter().flatten().all(|v| v.is_finite()) && sh.geo.iter().flatten().all(|v| v.is_finite()) && sh.fade.iter().all(|v| (0.0..=1.0).contains(v)));
                for i in 0..n {
                    if let Some(p) = Plane::of(&sh.geo[i], n) {
                        assert!(p.n.is_finite() && p.d.is_finite() && p.in_box(3.0).iter().all(|c| c.is_finite()));
                    }
                    assert!(!shown_equation(&plan, &sh, i).contains("NaN"));
                }
            }
            // what is said is what happened
            let says = answer_words(&plan, true);
            match &plan.outcome {
                Outcome::Point(_) => assert!(says.contains("meet at") || says.contains("cross at"), "{says}"),
                Outcome::Impossible { .. } => assert!(says.contains("there is no solution"), "{says}"),
                Outcome::Many { free, .. } => assert!(says.contains(if free.len() == n { "every point is a solution" } else { "free" }), "{says}"),
            }
            let mut s = Elim::create();
            s.start(sys, 3.0);
            for canvas in [Vec2::new(1080.0, 1000.0), Vec2::new(390.0, 800.0), Vec2::new(240.0, 180.0)] {
                s.canvas = canvas;
                let ok = match s.the_view() {
                    View::Orbit { target, dist, .. } => target.is_finite() && dist.is_finite(),
                    View::Fit { center, half } => center.is_finite() && half.is_finite(),
                    View::Flat { .. } => false,
                };
                assert!(ok, "{sys:?}");
            }
        }
        // the rows of zeros from the start say so
        let plan = Plan::of(&System::three([[0.0; 3]; 3], [0.0, 1.0, 0.0]));
        assert!(matches!(plan.outcome, Outcome::Impossible { row: 1, value } if value == 1.0) && arrangement(&plan) == "Row 2 reads $0 = 1$ from the start: no point satisfies it.");
        let plan = Plan::of(&System::three([[0.0; 3]; 3], [0.0; 3]));
        assert!(matches!(&plan.outcome, Outcome::Many { free, .. } if free.len() == 3) && plan.ops.is_empty());
    }

    /// Numbers as the rows write them: no "-0", a true minus sign, three significant
    /// digits and no trailing zeros; equations without coefficients of 1 or terms of 0.
    #[test]
    fn numbers_and_equations_are_written_plainly() {
        let cases = [(0.0, "0"), (-0.0, "0"), (2.0, "2"), (-1.5, "−1.5"), (1.0 / 3.0, "0.333"), (-2.0 / 3.0, "−0.667"), (0.01, "0.01"), (100.0, "100"), (12.345, "12.3"), (1e-4, "0.0001"), (-123456.0, "−1.23e5"), (f64::NAN, "?")];
        for (v, want) in cases {
            assert_eq!(number(v), want, "{v}");
        }
        assert_eq!(factor(-1.0), "(−1)");
        assert_eq!(equation(&[1.0, -1.0, 0.0, -3.0], 3), "x − y = −3");
        assert_eq!(equation(&[0.0, 0.0, 5.0, 5.0], 3), "5z = 5");
        assert_eq!(equation(&[-2.0, 1.0, 0.0, 0.0], 2), "−2x + y = 0");
        assert_eq!(equation(&[0.0, 0.0, 0.0, 1.0], 3), "0 = 1");
        assert_eq!(equation_with(&[0.04, 1.5, -1.0, 2.0], 3, &|v| passing(v, 1)), "1.5y − 1.0z = 2.0");
        assert_eq!(point_text(&[-1.0, 2.0, 1.0], 3), "(−1, 2, 1)");
        // the words keep an equation, and a point, on one line
        let lines = wrap("aaaa bbbb cccc dd $(−1, 2, 1)$ and $5y + 3z = 13$ then", SMALL, 120.0);
        assert!(lines.iter().any(|l| l.contains("$(−1, 2, 1)$")) && lines.iter().any(|l| l.contains("$5y + 3z = 13$")), "{lines:?}");
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for (name, sys) in offered() {
            let plan = Plan::of(&sys);
            println!("== {name}: {:?}", plan.outcome);
            for k in 0..plan.states.len() {
                let title = if k == 0 { "start".to_string() } else { step_title(&plan, k - 1) };
                println!("  {k:>2} {title:<24} {}", rows_read(&plan, k).join("  |  "));
            }
            println!("  L = {:?}, pivots {:?}, det {}", plan.l, plan.pivots, plan.det());
            for k in 0..plan.steps() {
                println!("  words {}: {}", k + 1, step_words(&plan, k, true, true));
            }
            println!("  intro: {}", intro_words(&plan, true));
        }
    }

    /// What is kept from frame to frame (the names, `Still`; the plan; the panel's words)
    /// follows every change: run side by side with a copy that works everything out
    /// afresh in every frame, the scene draws exactly the same in every frame, through
    /// the buttons over the picture and in the panel, the step slider, every experiment
    /// (two unknowns among them), drags of the view, a zoom, a pause, a phone's window
    /// and its buttons, and the scene opened again.
    #[test]
    fn what_is_kept_follows_every_change() {
        use super::super::subspaces::tried::{labels_follow, Step::*};
        let steps = [
            Rest(30),
            Click([376.0, 76.0]),
            Rest(130),
            Click([376.0, 76.0]),
            Click([295.0, 76.0]),
            Rest(60),
            Click([465.0, 76.0]),
            Rest(300),
            Click([1542.0, 918.0]),
            Rest(30),
            Drag([1325.0, 975.0], [1450.0, 975.0]),
            Rest(150),
            Drag([900.0, 800.0], [1000.0, 760.0]),
            Rest(150),
            Scroll([800.0, 500.0], -200.0),
            Rest(150),
            Try(0),
            Rest(150),
            Click([1542.0, 918.0]),
            Rest(60),
            Drag([900.0, 700.0], [1000.0, 600.0]),
            Rest(60),
            Try(1),
            Rest(150),
            Click([1542.0, 918.0]),
            Rest(60),
            Try(2),
            Rest(300),
            Click([1542.0, 918.0]),
            Rest(60),
            Try(3),
            Rest(250),
            Click([1542.0, 918.0]),
            Rest(60),
            Try(4),
            Rest(250),
            Click([1542.0, 918.0]),
            Rest(60),
            Pause,
            Click([1405.0, 918.0]),
            Rest(20),
            Pause,
            Rest(120),
            Size([390.0, 800.0]),
            Rest(30),
            Click([227.0, 591.0]),
            Rest(200),
            Size([1600.0, 1000.0]),
            Rest(30),
            Away,
            Rest(30),
        ];
        if let Some(again) = labels_follow("elim", &steps) {
            assert!(again > 150, "the names were written again as they were only {again} times");
        }
    }
}

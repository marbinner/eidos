//! Change of basis: coordinates are directions for a walk along two arrows, so the
//! same arrow has other numbers in another basis, and the same map another matrix.

use super::*;
use glam::{DMat2, DVec2};

pub const DEF: SceneDef = SceneDef {
    id: "basis",
    title: "Change of basis",
    group: "Linear algebra: vectors and maps",
    blurb: "The same arrow and the same map, written in another basis; the eigenvector basis makes a matrix diagonal.",
    make: |_| Box::new(ChangeOfBasis::create()),
};

/// What the picture is about: one arrow's two sets of numbers, or one map's two matrices.
const ARROW: usize = 0;
const MAP: usize = 1;
const MODES: [&str; 2] = ["an arrow", "a map"];

/// The colours, each with one meaning.
const B1_COL: Color = pal::PINK;
const B2_COL: Color = pal::GREEN;
/// The grid of b₁ and b₂.
const GRID_COL: Color = pal::BLUE;
/// The square grid of î and ĵ.
const SQUARE_COL: Color = pal::DIM;
const V_COL: Color = pal::WHITE;
/// The column of B⁻¹AB whose walk is drawn.
const PICK_COL: Color = pal::CYAN;

/// Where A sends b₁ and b₂: their colours, paler.
fn pale(c: Color) -> Color {
    c.mix(pal::WHITE, 0.5)
}

/// The two grids' colours for words on the dark panel, a little lighter so they read.
fn square_text() -> Color {
    SQUARE_COL.mix(pal::WHITE, 0.3)
}

fn grid_text() -> Color {
    GRID_COL.mix(pal::WHITE, 0.4)
}

/// How far a tip can be taken from the origin.
const REACH: f32 = 5.0;
/// The tips move in steps of this much.
const STEP: f32 = 0.1;
/// How quickly a tip is drawn gliding to where it has stepped: the time constant of
/// each of the two eases it goes through, in seconds (about two thirds of the way in
/// twice this).
const GLIDE: f32 = 0.03;
/// A tip drawn within this of its place is there.
const ARRIVED: f32 = 1e-4;
/// Seconds an experiment, or the glide onto the eigenvectors, takes.
const TWEEN_SECONDS: f32 = 1.6;
/// A length below this is no length at all.
const TINY: f64 = 1e-6;
/// How far out a walk is drawn (a nearly flat basis can send it millions of units out).
const CLIP: f64 = 1000.0;

/// The tips the scene opens with: b₁, b₂ and v, where v = 2 b₁ + 1 b₂.
const OPENING: [Vec2; 3] = [Vec2::new(1.2, 0.4), Vec2::new(0.5, 1.0), Vec2::new(2.9, 1.8)];

/// The map of the last two experiments, by its columns: it stretches by 2 along (1, 1)
/// and by 1/2 along (−1, 2).
const STRETCHES: [f32; 4] = [1.5, 1.0, 0.5, 1.0];

/// Quick choices for A: (button, the matrix by its columns).
const PRESETS: [(&str, [f32; 4]); 5] =
    [("stretches", STRETCHES), ("symmetric", [2.0, 1.0, 1.0, 2.0]), ("shear", [1.0, 0.0, 1.0, 1.0]), ("turn", [0.0, 1.0, -1.0, 0.0]), ("$2I$", [2.0, 0.0, 0.0, 2.0])];

/// Experiments: (button, what to look for). What each sets is `trial`.
const TRIALS: [(&str, &str); 5] = [
    (
        "the same arrow, new numbers",
        r"The square grid turns into the grid of $b_1$ and $b_2$, and $v$ stays where it is. In the square grid $v$ is $(2.9, 1.8)$: 2.9 steps of $\hat\imath$, then 1.8 of $\hat\jmath$. In the grid of $b$ it is 2 steps of $b_1$, then 1 of $b_2$, so its coordinates are $c = (2, 1)$. One arrow, two sets of numbers: $v = Bc$, and $c = B^{-1}v$.",
    ),
    (
        "a basis at right angles",
        r"$b_1 = (0.6, 0.8)$ and $b_2 = (-0.8, 0.6)$ have length 1 and meet at a right angle. Each coordinate is now $v$'s shadow on that arrow's line (dashed, with its right angle), so it is a dot product: $c_1 = b_1 \cdot v = 3.18$ and $c_2 = b_2 \cdot v = -1.24$, negative because that shadow falls behind the start. Undoing $B$ is then only swapping its rows and columns: $B^{-1} = B^\top$.",
    ),
    (
        "a nearly flat basis",
        r"$b_1$ and $b_2$ are now only 4.4° apart, and $\det B = 0.1$. To reach $v = (1, 0)$, an arrow of length 1, the walk goes 10 steps out along $b_1$ and 10 back along $b_2$: $c = (10, -10)$. The flatter the basis, the larger $B^{-1}$, and the more the coordinates change when $v$ moves a little.",
    ),
    (
        "the same map, another matrix",
        r"$A$ stretches the plane to twice along one line and to half along another. Its matrix in the square grid is $A$; in the grid of $b$ the same map has the matrix $B^{-1}AB$. Its first column $(1.2, 1.12)$ says where $b_1$ goes: $Ab_1 = 1.2 b_1 + 1.12 b_2$ (follow the walk). The numbers differ, but the trace 2.5 and the determinant 1 are the same in both.",
    ),
    (
        "the eigenvector basis",
        r"$b_1$ and $b_2$ turn onto the two lines that $A$ does not turn: its eigenvectors. Each is then only stretched, $Ab_1 = 2 b_1$ and $Ab_2 = 0.5 b_2$, with no step along the other, and $B^{-1}AB$ is diagonal, with the eigenvalues 2 and 0.5 on it: $A = BDB^{-1}$. (A turn has no such lines: choose “turn” and the button says why.)",
    ),
];
const EIGEN_TRIAL: usize = 4;

/// What experiment `k` sets: the picture, the tips it ends at (before the glide onto
/// the eigenvectors, in the last), A, and whether the shadows are shown.
fn trial(k: usize) -> (usize, [Vec2; 3], Mat2, bool) {
    let a = Mat2::from_cols_array(&STRETCHES);
    match k {
        0 => (ARROW, OPENING, a, false),
        1 => (ARROW, [Vec2::new(0.6, 0.8), Vec2::new(-0.8, 0.6), OPENING[2]], a, true),
        2 => (ARROW, [Vec2::new(0.6, 1.0), Vec2::new(0.5, 1.0), Vec2::new(1.0, 0.0)], a, false),
        _ => (MAP, [OPENING[0], OPENING[1], Vec2::new(-1.0, 0.5)], a, false),
    }
}

// ----- the arithmetic, in double precision ---------------------------------------------

/// A tip in double precision. One on the grid of tenths is taken at its exact decimal
/// value, so that the numbers come out as they would by hand.
fn exact(v: Vec2) -> DVec2 {
    let tenths = v.as_dvec2() * 10.0;
    if (tenths - tenths.round()).abs().max_element() < 1e-4 {
        tenths.round() / 10.0
    } else {
        v.as_dvec2()
    }
}

/// A's entries as typed: the editor holds hundredths, taken at their exact value.
fn exact_map(a: Mat2) -> DMat2 {
    let e = |x: f32| {
        let h = x as f64 * 100.0;
        if (h - h.round()).abs() < 1e-3 {
            h.round() / 100.0
        } else {
            x as f64
        }
    };
    DMat2::from_cols(DVec2::new(e(a.x_axis.x), e(a.x_axis.y)), DVec2::new(e(a.y_axis.x), e(a.y_axis.y)))
}

/// What b₁ and b₂ make.
#[derive(Clone, Copy, Debug)]
struct Grid {
    /// B, with b₁ and b₂ as its columns.
    b: DMat2,
    /// B⁻¹ if b₁ and b₂ are a basis; if not, why not.
    inv: Result<DMat2, &'static str>,
}

impl Grid {
    fn of(b1: Vec2, b2: Vec2) -> Grid {
        let (p, q) = (exact(b1), exact(b2));
        let b = DMat2::from_cols(p, q);
        let (l1, l2) = (p.length(), q.length());
        let inv = if l1 < TINY && l2 < TINY {
            Err("$b_1$ and $b_2$ have no length")
        } else if l1 < TINY {
            Err("$b_1$ has no length")
        } else if l2 < TINY {
            Err("$b_2$ has no length")
        } else if b.determinant().abs() <= 1e-6 * l1 * l2 {
            // (sin of the angle between them below a millionth: in line, as far as
            // the numbers they were made from can tell)
            Err("$b_1$ and $b_2$ lie on one line")
        } else {
            Ok(b.inverse())
        };
        Grid { b, inv }
    }

    fn det(&self) -> f64 {
        if self.inv.is_ok() {
            self.b.determinant()
        } else {
            0.0
        }
    }

    /// v's coordinates c = B⁻¹v: v = c₁ b₁ + c₂ b₂.
    fn coords(&self, v: DVec2) -> Option<DVec2> {
        self.inv.ok().map(|i| i * v)
    }

    /// The matrix of the map `a` in this basis, B⁻¹AB.
    fn of_map(&self, a: DMat2) -> Option<DMat2> {
        self.inv.ok().map(|i| i * a * self.b)
    }

    /// At right angles and of length 1 (then B⁻¹ = Bᵀ).
    fn orthonormal(&self) -> bool {
        let (p, q) = (self.b.x_axis, self.b.y_axis);
        (p.length() - 1.0).abs() < 1e-6 && (q.length() - 1.0).abs() < 1e-6 && p.dot(q).abs() < 1e-6
    }

    /// A basis turning the way î and ĵ do (b₂ anticlockwise of b₁).
    fn right_handed(&self) -> bool {
        self.inv.is_ok() && self.b.determinant() > 0.0
    }
}

/// Is the matrix diagonal, as far as its size lets one tell?
fn diagonal(m: DMat2) -> bool {
    let size = m.to_cols_array().iter().fold(0.0f64, |s, v| s.max(v.abs()));
    m.x_axis.y.abs() <= 5e-6 * size.max(1e-9) && m.y_axis.x.abs() <= 5e-6 * size.max(1e-9)
}

/// A's eigenvectors, as a pair of unit arrows with their eigenvalues, if there are two
/// to make a basis of; if not, why not, in words.
fn eigen_basis(a: Mat2) -> Result<[(f32, Vec2); 2], String> {
    let e = exact_map(a);
    let size = e.to_cols_array().iter().fold(0.0f64, |s, v| s.max(v.abs()));
    if size == 0.0 {
        return Err(r"$A = 0$ sends every arrow to $0$: every arrow is an eigenvector, and $B^{-1}AB = 0$ in every basis.".into());
    }
    if e.x_axis.y.abs() <= 1e-9 * size && e.y_axis.x.abs() <= 1e-9 * size && (e.x_axis.x - e.y_axis.y).abs() <= 1e-9 * size {
        let l = say(e.x_axis.x);
        return Err(format!(r"$A = {l}I$ stretches every arrow alike: every arrow is an eigenvector, and $B^{{-1}}AB = A$ in every basis."));
    }
    match eig2(a) {
        Eig2::Complex { .. } => Err("$A$ turns every direction: it has no real eigenvectors, so no basis makes its matrix diagonal.".into()),
        Eig2::Real { l1, l2, v1, v2 } => {
            if v1.perp_dot(v2).abs() < 1e-4 {
                Err(format!(r"$A$ has only one line of eigenvectors ($\lambda = {}$ twice): a basis needs two, so none makes it diagonal.", say(((l1 + l2) * 0.5) as f64)))
            } else {
                Ok([(l1, v1), (l2, v2)])
            }
        }
    }
}

/// Where b₁ and b₂ go to lie along the eigenvectors `e`: each keeps its length (an
/// arrow too short to see is given length 1), and of the ways to pair the two lines
/// with b₁ and b₂, and to point along them, the one that turns them least and keeps
/// the basis turning the way it did.
fn eigen_targets(b1: Vec2, b2: Vec2, e: [(f32, Vec2); 2]) -> [Vec2; 2] {
    let len = |b: Vec2| if b.length() > 0.3 { b.length() } else { 1.0 };
    let (r1, r2) = (len(b1), len(b2));
    let right = b1.perp_dot(b2) >= 0.0;
    let turn = |from: Vec2, to: Vec2| if from.length() > 1e-6 { from.angle_to(to).abs() } else { 0.0 };
    let mut best: Option<(f32, [Vec2; 2])> = None;
    for (p, q) in [(0, 1), (1, 0)] {
        for s1 in [1.0f32, -1.0] {
            for s2 in [1.0f32, -1.0] {
                let (t1, t2) = (e[p].1 * (s1 * r1), e[q].1 * (s2 * r2));
                if (t1.perp_dot(t2) > 0.0) != right {
                    continue;
                }
                let cost = turn(b1, t1) + turn(b2, t2);
                if best.is_none_or(|(c, _)| cost < c) {
                    best = Some((cost, [t1, t2]));
                }
            }
        }
    }
    best.map_or([e[0].1, e[1].1], |(_, t)| t)
}

/// A number for the screen: two decimals at most and as few as say it, a true minus
/// sign and no "−0"; a large one shorter.
fn say(v: f64) -> String {
    if !v.is_finite() {
        return "?".to_string();
    }
    let a = v.abs();
    let digits = if a >= 9.95e9 {
        // (a two-figure power of ten: one figure before it, so that no number is
        // longer than six characters)
        format!("{a:.0e}")
    } else if a >= 99_999.5 {
        format!("{a:.1e}")
    } else if a >= 999.5 {
        format!("{a:.0}")
    } else if a >= 99.995 {
        format!("{a:.1}")
    } else {
        format!("{a:.2}")
    };
    let digits = if digits.contains('.') && !digits.contains('e') { digits.trim_end_matches('0').trim_end_matches('.').to_string() } else { digits };
    if v < 0.0 && digits != "0" {
        format!("−{digits}")
    } else {
        digits
    }
}

/// "2 b₁ + 1 b₂": a walk of `c1` steps along the first arrow, then `c2` along the
/// second (markup, the arrows' names too, to go between dollars).
fn walk(c1: f64, n1: &str, c2: f64, n2: &str) -> String {
    let second = say(c2);
    match second.strip_prefix('−') {
        Some(rest) => format!("{} {n1} - {rest} {n2}", say(c1)),
        None => format!("{} {n1} + {second} {n2}", say(c1)),
    }
}

/// "2.9 î": so many steps of an arrow (`name` as markup), as a formula.
fn steps(c: f64, name: &str) -> String {
    format!("${} {name}$", say(c))
}

// ----- moving the tips -------------------------------------------------------------------

/// Where a dragged tip goes when its handle is at `p`: no further out than `REACH`,
/// and then to the nearest tenths.
fn settle(p: Vec2) -> Vec2 {
    let p = if p.length() > REACH { p * (REACH / p.length()) } else { p };
    (p / STEP).round() * STEP
}

/// The angle `a` brought into (−π, π].
fn wrap(a: f32) -> f32 {
    let a = a % TAU;
    if a > PI {
        a - TAU
    } else if a <= -PI {
        a + TAU
    } else {
        a
    }
}

/// The pair of arrows a share `e` of the way from (p1, p2) to (q1, q2). b₁ turns the
/// short way round to its new direction, the angle from b₁ to b₂ changes evenly from
/// its old value to its new one, and the lengths change evenly. So while both ends are
/// bases turning the same way round, every pair on the way is a basis too: the angle
/// between the two never passes 0° or 180°, as it could if each tip slid straight to
/// its new place.
fn turn_pair(p1: Vec2, p2: Vec2, q1: Vec2, q2: Vec2, e: f32) -> (Vec2, Vec2) {
    let tiny = 1e-6;
    let angle = |v: Vec2| v.y.atan2(v.x);
    let (r1, r2, s1, s2) = (p1.length(), p2.length(), q1.length(), q2.length());
    // (an arrow of no length has no direction: it takes the one it is heading for)
    let end1 = if s1 > tiny { angle(q1) } else if r1 > tiny { angle(p1) } else { 0.0 };
    let start1 = if r1 > tiny { angle(p1) } else { end1 };
    let first = start1 + wrap(end1 - start1) * e;
    let between = |a: Vec2, b: Vec2, ra: f32, rb: f32| (ra > tiny && rb > tiny).then(|| wrap(angle(b) - angle(a)));
    let (d0, d1) = (between(p1, p2, r1, r2), between(q1, q2, s1, s2));
    let d1 = d1.or(d0).unwrap_or(PI * 0.5);
    let mut d0 = d0.unwrap_or(d1);
    // (the shorter way between the two, if they differ in which way they turn)
    if d1 - d0 > PI {
        d0 += TAU;
    } else if d0 - d1 > PI {
        d0 -= TAU;
    }
    let second = first + d0 + (d1 - d0) * e;
    let (l1, l2) = (r1 + (s1 - r1) * e, r2 + (s2 - r2) * e);
    (Vec2::from_angle(first) * l1, Vec2::from_angle(second) * l2)
}

/// An animation of the tips (b₁, b₂, v) from one place to another.
#[derive(Clone, Copy, Debug)]
struct Tween {
    from: [Vec2; 3],
    to: [Vec2; 3],
    /// How far it has got, 0 to 1.
    t: f32,
}

impl Tween {
    fn at(&self, s: f32) -> [Vec2; 3] {
        if s >= 1.0 {
            // (exactly there at the end, not nearly)
            return self.to;
        }
        let e = smoothstep(s);
        let (b1, b2) = turn_pair(self.from[0], self.from[1], self.to[0], self.to[1], e);
        // (a v that is not to move stays exactly where it is)
        [b1, b2, self.from[2] + (self.to[2] - self.from[2]) * e]
    }
}

// ----- where things go on the canvas -----------------------------------------------------

/// Is the canvas narrow and tall (a phone held upright)? The panel of numbers then goes
/// below the picture instead of over its corner.
fn narrow(view: Vec2) -> bool {
    view.x < 620.0 && view.y >= 560.0
}

/// Text sizes in the panel: the numbers, the words, the small print.
const NUM: f32 = 14.0;
const WORDS: f32 = 12.5;
const SMALL: f32 = 11.0;
/// The panel's inner margin, and the height of a line of the key on a phone.
const PAD: f32 = 12.0;
const KEY_LINE: f32 = 17.0;
/// Room kept round the picture for the names written beside things (logical pixels).
const NAME_ROOM: Vec2 = Vec2::new(90.0, 60.0);
/// The closest the view comes, in logical pixels per unit.
const MOST_PER_UNIT: f32 = 140.0;

/// One line of the panel of numbers.
enum Line {
    Words(String, Color),
    /// Matrices side by side, with signs between.
    Matrices(Vec<Part>),
    /// The button "use the eigenvectors", and why it cannot be used, if it cannot.
    Button(Option<String>),
}

enum Part {
    /// A matrix: its name over it, a smaller line under the name, its rows of numbers,
    /// the column ringed (if one is), words under it, and whether a click on one of
    /// its columns picks it.
    Matrix { name: String, sub: String, rows: Vec<Vec<(String, Color)>>, ring: Option<usize>, under: String, picks: bool },
    Sign(&'static str),
    Gap(f32),
}

/// What a click on the panel asks for.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Click {
    Column(usize),
    Eigen,
}

/// The longest number in the panel's matrices (characters): every cell is that wide.
/// (Never less than the six `say` can write, so that the panel keeps its size while
/// the numbers change.) A number is counted as wide as it is set, in characters of
/// the width a cell keeps for each, less the cell's margin: a minus sign, or a power
/// of ten, is wider than a digit.
fn widest(lines: &[Line]) -> usize {
    let per = NUM * 0.56;
    let chars = |t: &str| ((width_of(t, NUM) - 6.0) / per).ceil().max(0.0) as usize;
    let mut w = 6;
    for line in lines {
        if let Line::Matrices(parts) = line {
            for part in parts {
                if let Part::Matrix { rows, .. } = part {
                    w = rows.iter().flatten().map(|(t, _)| chars(t)).fold(w, usize::max);
                }
            }
        }
    }
    w
}

fn part_width(part: &Part, widest: usize) -> f32 {
    match part {
        Part::Matrix { name, sub, rows, under, .. } => {
            let cols = rows.iter().map(Vec::len).max().unwrap_or(0);
            matrix_size(rows.len(), cols, widest, NUM).x.max(width_of(name, NUM)).max(width_of(sub, SMALL)).max(width_of(under, SMALL))
        }
        Part::Sign(_) => NUM + 8.0,
        Part::Gap(w) => *w,
    }
}

/// The heights of a row of matrices: (names, the line under the names, the matrices, the words under them).
fn matrices_heights(parts: &[Part]) -> (f32, f32, f32, f32) {
    let any = |f: &dyn Fn(&Part) -> bool| parts.iter().any(f);
    let sub = if any(&|p| matches!(p, Part::Matrix { sub, .. } if !sub.is_empty())) { SMALL + 4.0 } else { 0.0 };
    let under = if any(&|p| matches!(p, Part::Matrix { under, .. } if !under.is_empty())) { SMALL + 6.0 } else { 0.0 };
    let rows = parts.iter().map(|p| if let Part::Matrix { rows, .. } = p { rows.len() } else { 0 }).max().unwrap_or(0);
    (NUM + 5.0, sub, matrix_size(rows, 1, 4, NUM).y, under)
}

/// The button's size.
const BUTTON: Vec2 = Vec2::new(176.0, 26.0);

/// `text_width`, kept from the frames before: the same few texts are measured again
/// and again, every frame, and one with mathematics in it is measured by setting its
/// formulas, which is not cheap.
fn width_of(text: &str, size: f32) -> f32 {
    thread_local! {
        // (for each text size, the texts measured at it)
        static KEPT: std::cell::RefCell<Vec<(f32, std::collections::HashMap<String, f32>)>> = Default::default();
    }
    if !text.contains('$') {
        return text_width(text, size);
    }
    KEPT.with(|kept| {
        let mut kept = kept.borrow_mut();
        let at = match kept.iter().position(|(s, _)| *s == size) {
            Some(at) => at,
            None => {
                kept.push((size, Default::default()));
                kept.len() - 1
            }
        };
        let texts = &mut kept[at].1;
        if let Some(w) = texts.get(text) {
            return *w;
        }
        // (texts that change every frame, the numbers of a step under way, are not
        // kept for long)
        if texts.len() > 4096 {
            texts.clear();
        }
        let w = text_width(text, size);
        texts.insert(text.to_string(), w);
        w
    })
}

/// The words of a text, each formula whole (with what is glued to it, as in "$v$'s").
fn words_of(text: &str) -> Vec<&str> {
    let (bytes, mut words) = (text.as_bytes(), Vec::new());
    let (mut start, mut inside, mut i) = (0, false, 0);
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 1,
            b'$' => inside = !inside,
            b' ' if !inside => {
                words.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    words.push(&text[start.min(text.len())..]);
    words
}

/// Words broken into lines no wider than `width` pixels at text size `size` (a
/// formula is not broken).
fn wrap_words(text: &str, width: f32, size: f32) -> Vec<String> {
    let mut lines = vec![String::new()];
    for word in words_of(text) {
        let last = lines.last_mut().map_or(0.0, |l| width_of(l, size));
        match lines.last_mut() {
            Some(l) if l.is_empty() => l.push_str(word),
            Some(l) if last + width_of(word, size) + size * 0.47 <= width => {
                l.push(' ');
                l.push_str(word);
            }
            _ => lines.push(word.to_string()),
        }
    }
    lines
}

/// The width the panel's lines need (the button's reasons are wrapped to the rest).
fn content_width(lines: &[Line]) -> f32 {
    let w = widest(lines);
    lines
        .iter()
        .map(|line| match line {
            Line::Words(t, _) => width_of(t, WORDS),
            Line::Matrices(parts) => parts.iter().map(|p| part_width(p, w)).sum::<f32>() + 6.0 * parts.len().saturating_sub(1) as f32,
            Line::Button(_) => BUTTON.x,
        })
        .fold(330.0, f32::max)
}

fn line_height(line: &Line, width: f32) -> f32 {
    match line {
        Line::Words(..) => WORDS + 8.0,
        Line::Matrices(parts) => {
            let (a, b, c, d) = matrices_heights(parts);
            a + b + c + d + 6.0
        }
        // (room for two lines of reasons whether there are any or not, so that the
        // panel keeps its size when the button comes into use or goes out of it)
        Line::Button(why) => BUTTON.y + 10.0 + why.as_ref().map_or(0, |w| wrap_words(w, width, SMALL).len()).max(2) as f32 * (SMALL + 4.0),
    }
}

/// The size of the panel's contents, margins included.
fn content_size(lines: &[Line]) -> Vec2 {
    let width = content_width(lines);
    Vec2::new(width + 2.0 * PAD, lines.iter().map(|l| line_height(l, width)).sum::<f32>() + 2.0 * PAD - 4.0)
}

/// How the canvas is shared between the panel of numbers and the picture.
#[derive(Clone, Copy, Debug)]
struct Layout {
    view: Vec2,
    inset: f32,
    narrow: bool,
    /// The panel's corners (logical pixels), if the canvas can spare the room.
    panel: Option<(Vec2, Vec2)>,
    /// Whether the key is written inside the panel (on a phone) rather than under the picture.
    key_inside: bool,
}

impl Layout {
    /// For a canvas of `view` logical pixels, `inset` of whose top lies under the
    /// application's buttons, a panel holding `content` and, on a phone, `key` lines of key.
    fn of(view: Vec2, inset: f32, content: Vec2, key: usize) -> Layout {
        let narrow = narrow(view);
        let mut lay = Layout { view, inset, narrow, panel: None, key_inside: false };
        if narrow {
            // along the bottom, with the key under the numbers if there is room for it
            for with_key in [true, false] {
                let h = content.y + if with_key { key as f32 * KEY_LINE + 6.0 } else { 0.0 };
                let top = view.y - h;
                if content.x <= view.x && top >= inset + 0.4 * view.y {
                    lay.panel = Some((Vec2::new(0.0, top), view));
                    lay.key_inside = with_key;
                    break;
                }
            }
        } else {
            // over the top right corner, while it leaves most of the picture clear
            let min = Vec2::new(view.x - content.x - 16.0, inset.max(16.0));
            let max = min + content;
            if min.x >= 0.3 * view.x && max.y <= view.y - 60.0 && content.x * content.y <= 0.4 * view.x * view.y {
                lay.panel = Some((min, max));
            }
        }
        lay
    }

    fn covers(&self, p: Vec2) -> bool {
        self.panel.is_some_and(|(lo, hi)| p.cmpge(lo).all() && p.cmple(hi).all())
    }

    /// Where the caption along the bottom begins (the canvas's foot, if there is none).
    fn caption_top(&self) -> f32 {
        if self.key_inside || self.view.y < 300.0 {
            self.view.y
        } else {
            self.view.y - 34.0
        }
    }

    /// The part of the canvas the picture has to itself, and the scale (pixels per
    /// unit) a picture `size` world units across can be shown at there: beside the
    /// panel or below it, whichever shows it larger.
    fn region(&self, size: Vec2) -> ((Vec2, Vec2), f32) {
        let scale = |(lo, hi): (Vec2, Vec2)| {
            let room = (hi - lo - NAME_ROOM).max(Vec2::splat(60.0));
            (room.x / size.x.max(1e-6)).min(room.y / size.y.max(1e-6)).min(MOST_PER_UNIT).max(1e-3)
        };
        let with = |r: (Vec2, Vec2)| (r, scale(r));
        let (top, bottom) = (self.inset + 4.0, self.caption_top());
        match self.panel {
            None => with((Vec2::new(0.0, top), Vec2::new(self.view.x, bottom.max(top + 1.0)))),
            Some((lo, _)) if self.narrow => with((Vec2::new(0.0, top), Vec2::new(self.view.x, lo.y.max(top + 1.0)))),
            Some((lo, hi)) => {
                let beside = (Vec2::new(0.0, top), Vec2::new(lo.x.max(1.0), bottom.max(top + 1.0)));
                let below = (Vec2::new(0.0, hi.y), Vec2::new(self.view.x, bottom.max(hi.y + 1.0)));
                if scale(beside) >= scale(below) {
                    with(beside)
                } else {
                    with(below)
                }
            }
        }
    }
}

// ----- drawing helpers -------------------------------------------------------------------

/// The part of the segment from `a` to `b` within `CLIP` of the origin (in double
/// precision: the ends may be far out), if any.
fn clip_disc(a: DVec2, b: DVec2) -> Option<(Vec2, Vec2)> {
    let d = b - a;
    let (qa, qb, qc) = (d.dot(d), 2.0 * a.dot(d), a.dot(a) - CLIP * CLIP);
    let (t0, t1) = if qa < 1e-18 {
        if qc <= 0.0 {
            (0.0, 1.0)
        } else {
            return None;
        }
    } else {
        let disc = qb * qb - 4.0 * qa * qc;
        if disc < 0.0 {
            return None;
        }
        let r = disc.sqrt();
        (((-qb - r) / (2.0 * qa)).max(0.0), ((-qb + r) / (2.0 * qa)).min(1.0))
    };
    (t0 <= t1).then(|| ((a + d * t0).as_vec2(), (a + d * t1).as_vec2()))
}

/// The middle of the part of the segment from `a` to `b` that is in the rectangle
/// `lo`..`hi` (where a label for it can be seen), if any of it is.
fn visible_middle(a: Vec2, b: Vec2, lo: Vec2, hi: Vec2) -> Option<Vec2> {
    let d = b - a;
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (p, q) in [(-d.x, a.x - lo.x), (d.x, hi.x - a.x), (-d.y, a.y - lo.y), (d.y, hi.y - a.y)] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let r = q / p;
            if p < 0.0 {
                t0 = t0.max(r);
            } else {
                t1 = t1.min(r);
            }
        }
    }
    (t0 <= t1).then(|| a + d * ((t0 + t1) * 0.5))
}

/// How many grid lines either side of the origin cover `reach` grid steps: one of a
/// few counts, so that the number drawn changes seldom.
fn lines_for(reach: f64) -> i32 {
    [16, 32, 64, 128, 256, 512].into_iter().find(|&n| n as f64 >= reach).unwrap_or(512)
}

/// The grid of the basis `b` (the square grid for B = I): the lines through the whole
/// numbers of steps, out to `n` either side, drawn every `every`th, the two through the
/// origin bolder. `alpha` is that of the thin lines.
fn grid_lines(f: &mut Frame, b: DMat2, n: i32, every: i32, color: Color, alpha: f32) {
    let l = n as f64;
    let at = |c: DVec2| (b * c).as_vec2();
    for k in (-n..=n).filter(|k| k % every == 0) {
        let kf = k as f64;
        let (w, a) = if k == 0 { (1.8, (alpha * 2.2).clamp(0.55, 0.95)) } else { (1.1, alpha) };
        f.line(at(DVec2::new(kf, -l)), at(DVec2::new(kf, l)), w, color.a(a));
        f.line(at(DVec2::new(-l, kf)), at(DVec2::new(l, kf)), w, color.a(a));
    }
}

/// Where the number of a leg of a walk may be written beside it: by its middle `mid`
/// (of the part in view) first, then a little further along or back, where a name at
/// an arrow's tip may not be in the way. (Only a little: a label that moved between
/// places far apart would be seen to hop.)
fn along_leg(a: Vec2, b: Vec2, mid: Vec2) -> [Vec3; 3] {
    let d = (b - a) * 0.5;
    [0.0, 0.2, -0.2].map(|t| (mid + d * t).extend(0.0))
}

/// Screen direction from one point of the plane to another (zero if either is not in front).
fn screen_dir(f: &Frame, from: Vec2, to: Vec2) -> Vec2 {
    match (on_screen(f, from.extend(0.0)), on_screen(f, to.extend(0.0))) {
        (Some(p), Some(q)) => q - p,
        _ => Vec2::ZERO,
    }
}

/// The screen direction from the origin out to `v` (up and to the right for no length).
fn outwards(f: &Frame, v: Vec2) -> Vec2 {
    let d = screen_dir(f, Vec2::ZERO, v);
    if d.length() > 1e-3 {
        d
    } else {
        Vec2::new(1.0, -1.0)
    }
}

/// The side of the segment `a`–`b` away from `other`, as a screen direction.
fn away_from(f: &Frame, a: Vec2, b: Vec2, other: Vec2) -> Vec2 {
    let along = screen_dir(f, a, b);
    let to_other = screen_dir(f, (a + b) * 0.5, other);
    let across = along.perp();
    if across.length() < 1e-6 {
        return Vec2::new(0.0, 1.0);
    }
    if across.dot(to_other) > 0.0 {
        -across
    } else {
        across
    }
}

/// A walk drawn as two wide translucent bands: `c.x` steps of `b1` from the origin,
/// then `c.y` of `b2` to the end, each with a small arrowhead in its middle (as much
/// of it as is in the view `lo`..`hi`) for the way it goes: backwards along an arrow
/// is a negative number of steps. Returns those middles, for the numbers.
fn draw_walk(f: &mut Frame, room: &mut Room, b1: DVec2, b2: DVec2, c: DVec2, (lo, hi): (Vec2, Vec2)) -> [Option<(Vec2, Vec2, Vec2)>; 2] {
    let corner = b1 * c.x;
    let end = corner + b2 * c.y;
    let wpp = f.cam.world_per_px(Vec3::ZERO) * f.px;
    let mut legs = [None, None];
    for (k, (a, b, col)) in [(DVec2::ZERO, corner, B1_COL), (corner, end, B2_COL)].into_iter().enumerate() {
        if (b - a).length() < 1e-9 {
            continue;
        }
        let Some((p, q)) = clip_disc(a, b) else { continue };
        f.line(p, q, 9.0, col.a(0.38));
        room.line(f, p.extend(0.0), q.extend(0.0));
        if let Some(mid) = visible_middle(p, q, lo, hi) {
            // (only on a leg long enough to carry one)
            let d = (q - p).normalize_or_zero();
            if (q - p).length() > 40.0 * wpp {
                let (tip, back, side) = (mid + d * (6.0 * wpp), mid - d * (5.0 * wpp), d.perp() * (6.0 * wpp));
                f.dl.tris_alpha.tri(tip.extend(0.0), (back + side).extend(0.0), (back - side).extend(0.0), col.mix(pal::WHITE, 0.35));
            }
            legs[k] = Some((p, q, mid));
        }
    }
    legs
}

/// A 2 by 2 matrix for the panel: its numbers as `say` writes them (typeset), each in
/// the colour `colour` gives its row and column.
fn matrix_part(name: &str, sub: &str, m: DMat2, colour: impl Fn(usize, usize) -> Color, ring: Option<usize>, under: String, picks: bool) -> Part {
    let rows = (0..2).map(|i| (0..2).map(|j| (format!("${}$", say(m.col(j)[i])), colour(i, j))).collect()).collect();
    Part::Matrix { name: name.into(), sub: sub.into(), rows, ring, under, picks }
}

/// The same for a column of two numbers.
fn vector_part(name: &str, sub: &str, v: DVec2, colour: impl Fn(usize) -> Color) -> Part {
    let rows = (0..2).map(|i| vec![(format!("${}$", say(v[i])), colour(i))]).collect();
    Part::Matrix { name: name.into(), sub: sub.into(), rows, ring: None, under: String::new(), picks: false }
}

/// "tr = 2.5   det = 1".
fn trace_det(m: DMat2) -> String {
    format!(r"$\tr = {}$   $\det = {}$", say(m.x_axis.x + m.y_axis.y), say(m.determinant()))
}

/// The key to what has no name of its own in the picture.
fn key_parts(mode: usize) -> [&'static str; 3] {
    if mode == ARROW {
        [r"grey grid: the square grid of $\hat\imath$ and $\hat\jmath$", "blue grid: the grid of $b_1$ and $b_2$", "wide bands: $v$'s walk in $b_1$ and $b_2$"]
    } else {
        [r"grey grid: the square grid of $\hat\imath$ and $\hat\jmath$", "blue grid: the grid of $b_1$ and $b_2$", "pale: where $A$ sends $b_1$ and $b_2$"]
    }
}

// ----- the scene ------------------------------------------------------------------------

pub struct ChangeOfBasis {
    /// b₁, b₂ and v: what every number is worked out from.
    tips: [Vec2; 3],
    /// Where each handle is while it is dragged: the tip is that point taken to the
    /// nearest tenths, so the handle follows the pointer smoothly while the tip steps.
    grip: [Vec2; 3],
    /// Where the tips are drawn: gliding to `tips` when those have stepped, through
    /// `between`, so that a glide sets off gently instead of at its fastest (two
    /// eases one after the other, as `follow` in `packing.rs`).
    drawn: [Vec2; 3],
    between: [Vec2; 3],
    a: Mat2,
    mode: usize,
    /// v's shadows on the lines of b₁ and b₂ (the dot products).
    shadows: bool,
    /// In a map: v and Av.
    show_v: bool,
    /// Which column of B⁻¹AB has its walk drawn.
    pick: usize,
    tween: Option<Tween>,
    /// The canvas last drawn on, and how much of its top the application's buttons
    /// cover: the panel's place follows them, and the picture's follows the panel's.
    canvas: Vec2,
    inset: f32,
    new_view: bool,
    /// The size of the panel for the opening tips, and what it was worked out from
    /// (see `panel_size`).
    opening: std::cell::RefCell<Option<(OpeningFrom, Vec2)>>,
}

/// What the panel for the opening tips depends on: the picture, the shadows, the
/// column followed, A's entries, and why the eigenvector button cannot be used.
type OpeningFrom = (usize, bool, usize, [u32; 4], Option<String>);

impl ChangeOfBasis {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        ChangeOfBasis {
            tips: OPENING,
            grip: OPENING,
            drawn: OPENING,
            between: OPENING,
            a: Mat2::from_cols_array(&STRETCHES),
            mode: ARROW,
            shadows: false,
            show_v: true,
            pick: 0,
            tween: None,
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            new_view: false,
            opening: Default::default(),
        }
    }

    /// Set up one of the experiments in `TRIALS`, and play it.
    fn set_up(&mut self, k: usize) {
        let (mode, tips, a, shadows) = trial(k);
        (self.mode, self.a, self.shadows, self.show_v, self.pick) = (mode, a, shadows, true, 0);
        let here = Grid::of(self.tips[0], self.tips[1]).right_handed();
        let from = match k {
            // from the square grid: the lesson is the grid changing under v
            0 => [Vec2::X, Vec2::Y, tips[2]],
            // from the basis of the experiment before it, which it carries on
            EIGEN_TRIAL => tips,
            // from where the tips are, unless the way there would pass through a
            // flat basis (they turn the other way round, or are no basis now)
            _ if here => self.tips,
            _ => [Vec2::X, Vec2::Y, self.tips[2]],
        };
        let to = match (k, eigen_basis(a)) {
            (EIGEN_TRIAL, Ok(e)) => {
                let [t1, t2] = eigen_targets(tips[0], tips[1], e);
                [t1, t2, tips[2]]
            }
            _ => tips,
        };
        self.start(from, to);
    }

    fn start(&mut self, from: [Vec2; 3], to: [Vec2; 3]) {
        (self.tips, self.grip, self.drawn, self.between) = (from, from, from, from);
        self.tween = Some(Tween { from, to, t: 0.0 });
        self.new_view = true;
    }

    /// Turn b₁ and b₂ onto A's eigenvectors, if it has two.
    fn use_eigenvectors(&mut self) {
        if let Ok(e) = eigen_basis(self.a) {
            let [t1, t2] = eigen_targets(self.tips[0], self.tips[1], e);
            self.start(self.tips, [t1, t2, self.tips[2]]);
        }
    }

    /// Move an animation on by `dt` seconds of scene time.
    fn advance(&mut self, dt: f32) {
        let Some(tween) = self.tween.as_mut() else { return };
        tween.t += dt.max(0.0) / TWEEN_SECONDS;
        let (t, tween) = (tween.t, *tween);
        self.tips = tween.at(t);
        if t >= 1.0 {
            self.tween = None;
        }
        // (an animation is a glide already: it is drawn as it stands)
        (self.grip, self.drawn, self.between) = (self.tips, self.tips, self.tips);
    }

    /// Are the tips drawn where they are?
    fn arrived(&self) -> bool {
        self.drawn == self.tips && self.between == self.tips
    }

    /// Bring the tips as drawn `dt` seconds nearer to where they are; with `at_once`, all the way.
    fn glide(&mut self, dt: f32, at_once: bool) {
        let k = 1.0 - (-dt.max(0.0) / GLIDE).exp();
        for i in 0..3 {
            let to = self.tips[i];
            self.between[i] += (to - self.between[i]) * k;
            self.drawn[i] += (self.between[i] - self.drawn[i]) * k;
            // (exactly there in the end, not nearly: the picture then stops changing)
            let near = (to - self.drawn[i]).length().max((to - self.between[i]).length()) < ARRIVED;
            if at_once || near || !(self.drawn[i].is_finite() && self.between[i].is_finite()) {
                (self.drawn[i], self.between[i]) = (to, to);
            }
        }
    }

    /// Is the v handle in use? (In a map, only while v and Av are shown.)
    fn v_shown(&self) -> bool {
        self.mode == ARROW || self.show_v
    }

    /// What the view must show: the tips and what is drawn for them, where an animation
    /// under way is taking them.
    fn home(&self) -> Vec<Vec2> {
        let [b1, b2, v] = self.tween.map_or(self.tips, |t| t.to);
        let g = Grid::of(b1, b2);
        let mut points = vec![Vec2::ZERO, b1, b2];
        if self.mode == ARROW {
            points.extend([v, Vec2::new(v.x, 0.0)]);
            if let Some(c) = g.coords(exact(v)) {
                points.push((exact(b1) * c.x).as_vec2());
                if self.shadows {
                    for b in [exact(b1), exact(b2)] {
                        points.push((b * (b.dot(exact(v)) / b.length_squared())).as_vec2());
                    }
                }
            }
        } else {
            let a = self.a;
            points.extend([a * b1, a * b2]);
            if let Some(m) = g.of_map(exact_map(a)) {
                points.extend([(exact(b1) * m.x_axis.x).as_vec2(), (exact(b1) * m.y_axis.x).as_vec2()]);
            }
            if self.show_v {
                points.extend([v, a * v]);
            }
        }
        // (a walk far out along a nearly flat basis is not followed to its end)
        points.into_iter().filter(|p| p.is_finite()).map(|p| if p.length() > 14.0 { p * (14.0 / p.length()) } else { p }).collect()
    }

    /// The view that fits `home` into the part of the canvas the panel leaves free.
    fn the_view(&self) -> View {
        let points = self.home();
        let (lo, hi) = points.iter().fold((Vec2::ZERO, Vec2::ZERO), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
        let size = (hi - lo).max(Vec2::splat(2.0));
        // (the panel as it will be when the tips get there)
        let lines = self.lines_at(self.tween.map_or(self.tips, |t| t.to));
        let lay = Layout::of(self.canvas, self.inset, self.panel_size(&lines), key_parts(self.mode).len());
        let ((min, max), per_unit) = lay.region(size);
        // the middle of the free part, which is not the canvas's, shows the middle of
        // the picture (screen y runs down, world y up)
        let off = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        let centre = (lo + hi) * 0.5;
        View::Fit { center: Vec2::new(centre.x - off.x, centre.y + off.y), half: self.canvas.max(Vec2::ONE) * (0.5 / per_unit) }
    }

    /// The size of the panel holding `lines`: never smaller than it is for a basis, so
    /// that it keeps its size when b₁ and b₂ stop being one.
    fn panel_size(&self, lines: &[Line]) -> Vec2 {
        // (the panel for the opening tips is asked for in every frame, and changes
        // only with what it is worked out from: kept until that changes)
        let why = if self.mode == MAP { self.eigen_blocked() } else { None };
        let from = (self.mode, self.shadows, self.pick, self.a.to_cols_array().map(f32::to_bits), why);
        let mut kept = self.opening.borrow_mut();
        let opening = match kept.as_ref() {
            Some((was, size)) if *was == from => *size,
            _ => {
                let size = content_size(&self.lines_at(OPENING));
                *kept = Some((from, size));
                size
            }
        };
        content_size(lines).max(opening)
    }

    /// Why the eigenvector button cannot be used, if it cannot.
    fn eigen_blocked(&self) -> Option<String> {
        let g = Grid::of(self.tips[0], self.tips[1]);
        match eigen_basis(self.a) {
            Err(why) => Some(why),
            Ok(_) if g.of_map(exact_map(self.a)).is_some_and(diagonal) => Some(r"$b_1$ and $b_2$ lie along $A$'s eigenvectors already: $B^{-1}AB$ is diagonal.".into()),
            Ok(_) => None,
        }
    }

    /// The panel of numbers for the tips `tips`.
    fn lines_at(&self, tips: [Vec2; 3]) -> Vec<Line> {
        let [b1, b2, v] = tips;
        let g = Grid::of(b1, b2);
        let v = exact(v);
        let mut lines = Vec::new();
        let b_colour = |_: usize, j: usize| [B1_COL, B2_COL][j];
        let c_colour = |i: usize| [B1_COL, B2_COL][i];
        if self.mode == ARROW {
            lines.push(Line::Words(format!(r"in the square grid:   $v = {}$", walk(v.x, r"\hat\imath", v.y, r"\hat\jmath")), square_text()));
            match (g.coords(v), g.inv) {
                (Some(c), Ok(inv)) => {
                    lines.push(Line::Words(format!("in the grid of $b_1, b_2$:   $v = {}$", walk(c.x, "b_1", c.y, "b_2")), grid_text()));
                    let white = |_: usize| V_COL;
                    // (each named, and said what it is, under its name)
                    let (v_part, c_part) = (|| vector_part("$v$", r"in $\hat\imath, \hat\jmath$", v, white), || vector_part("$c$", "in $b_1, b_2$", c, c_colour));
                    lines.push(Line::Matrices(vec![v_part(), Part::Sign("$=$"), matrix_part("$B$", "columns $b_1, b_2$", g.b, b_colour, None, String::new(), false), c_part()]));
                    lines.push(Line::Matrices(vec![c_part(), Part::Sign("$=$"), matrix_part("$B^{-1}$", "undoes $B$", inv, |_, _| pal::TEXT, None, String::new(), false), v_part()]));
                    if g.orthonormal() {
                        lines.push(Line::Words(r"orthonormal: $B^{-1} = B^\top$, so $c_1 = b_1 \cdot v$ and $c_2 = b_2 \cdot v$".into(), pal::TEXT));
                    } else {
                        lines.push(Line::Words(format!(r"$\det B = {}$", say(g.det())), pal::DIM.mix(pal::WHITE, 0.25)));
                    }
                    if self.shadows {
                        let (d1, d2) = (g.b.x_axis.dot(v), g.b.y_axis.dot(v));
                        lines.push(Line::Words(format!(r"dot products:  $b_1 \cdot v = {}$,  $b_2 \cdot v = {}$", say(d1), say(d2)), pal::TEXT));
                    }
                }
                (_, Err(why)) => {
                    lines.push(Line::Words("in the grid of $b_1, b_2$:   none".into(), grid_text()));
                    lines.push(Line::Words(format!("{why}, so they are not a basis:"), pal::TEXT));
                    lines.push(Line::Words("walks along them reach only a line (or one point),".into(), pal::TEXT));
                    lines.push(Line::Words(r"and $v$ has no coordinates. $\det B = 0$: no $B^{-1}$.".into(), pal::TEXT));
                }
                (None, Ok(_)) => {}
            }
        } else {
            let a = exact_map(self.a);
            let a_part = matrix_part("$A$", "in the square grid", a, |_, _| pal::TEXT, None, trace_det(a), false);
            match g.of_map(a) {
                Some(m) => {
                    let m_colour = |i: usize, _: usize| pale([B1_COL, B2_COL][i]);
                    lines.push(Line::Matrices(vec![a_part, Part::Gap(22.0), matrix_part("$B^{-1}AB$", "in the grid of $b_1, b_2$", m, m_colour, Some(self.pick), trace_det(m), true)]));
                    let k = self.pick.min(1);
                    let names = ["Ab_1", "Ab_2"];
                    lines.push(Line::Words(format!("column {}:  ${} = {}$", k + 1, names[k], walk(m.col(k).x, "b_1", m.col(k).y, "b_2")), pale([B1_COL, B2_COL][k])));
                    // (two lines either way, so that the panel keeps its size)
                    if diagonal(m) {
                        lines.push(Line::Words(format!(r"diagonal: $b_1, b_2$ are eigenvectors, $\lambda = {}$ and ${}$", say(m.x_axis.x), say(m.y_axis.y)), pal::TEXT));
                        lines.push(Line::Words(r"so $A = BDB^{-1}$, with $D = B^{-1}AB$ diagonal".into(), pal::TEXT));
                    } else {
                        lines.push(Line::Words(r"click a column of $B^{-1}AB$ to follow its walk".into(), pal::DIM));
                        lines.push(Line::Words(String::new(), pal::DIM));
                    }
                }
                None => {
                    lines.push(Line::Matrices(vec![a_part]));
                    lines.push(Line::Words(format!(r"$B^{{-1}}AB$: none, since {}", g.inv.err().unwrap_or("")), grid_text()));
                    lines.push(Line::Words("they are not a basis, and $B$ has no inverse".into(), pal::TEXT));
                    lines.push(Line::Words(String::new(), pal::DIM));
                }
            }
            lines.push(Line::Button(self.eigen_blocked()));
        }
        lines
    }

    /// The panel, its top left corner at `min`, `width` wide; what a click on it asks for.
    fn draw_panel(&self, f: &mut Frame, lines: &[Line], (min, max): (Vec2, Vec2), lay: &Layout, pointer: Option<Vec2>) -> Option<Click> {
        hud_panel(f, min, max);
        let width = content_width(lines);
        let w = widest(lines);
        let (x, mut y) = (min.x + PAD, min.y + PAD);
        let mut asked = None;
        let over = |lo: Vec2, hi: Vec2| pointer.is_some_and(|p| p.cmpge(lo).all() && p.cmple(hi).all());
        for line in lines {
            match line {
                Line::Words(text, colour) => {
                    f.text_px(Vec2::new(x, y + 1.0), text.as_str(), WORDS, *colour, Align2::LEFT_TOP);
                }
                Line::Matrices(parts) => {
                    let (head, sub, rows, _) = matrices_heights(parts);
                    let top = y + head + sub;
                    let mut left = x;
                    for part in parts {
                        let pw = part_width(part, w);
                        let mid = left + pw * 0.5;
                        match part {
                            Part::Matrix { name, sub: words, rows: numbers, ring, under, picks } => {
                                let cols = numbers.iter().map(Vec::len).max().unwrap_or(0);
                                let size = matrix_size(numbers.len(), cols, w, NUM);
                                let at = Vec2::new(mid - size.x * 0.5, top);
                                f.text_px(Vec2::new(mid, y), name.as_str(), NUM, pal::TEXT, Align2::CENTER_TOP);
                                if !words.is_empty() {
                                    f.text_px(Vec2::new(mid, y + head), words.as_str(), SMALL, pal::DIM, Align2::CENTER_TOP);
                                }
                                hud_matrix(f, at, numbers, w, NUM, pal::DIM);
                                if !under.is_empty() {
                                    f.text_px(Vec2::new(mid, top + size.y + 4.0), under.as_str(), SMALL, pal::DIM.mix(pal::WHITE, 0.25), Align2::CENTER_TOP);
                                }
                                for j in 0..cols {
                                    let (lo, _) = matrix_entry(at, w, NUM, 0, j);
                                    let (_, hi) = matrix_entry(at, w, NUM, numbers.len().saturating_sub(1), j);
                                    if *ring == Some(j) {
                                        hud_outline(f, lo + Vec2::splat(2.0), hi - Vec2::splat(2.0), 1.4, PICK_COL);
                                    }
                                    if *picks && over(lo, hi) && f.ptr.clicked {
                                        asked = Some(Click::Column(j));
                                    }
                                }
                            }
                            Part::Sign(s) => {
                                f.text_px(Vec2::new(mid, top + rows * 0.5), *s, NUM, pal::TEXT, Align2::CENTER_CENTER);
                            }
                            Part::Gap(_) => {}
                        }
                        left += pw + 6.0;
                    }
                }
                Line::Button(why) => {
                    let (lo, hi) = (Vec2::new(x, y + 2.0), Vec2::new(x, y + 2.0) + BUTTON);
                    let usable = why.is_none();
                    let fill = match (usable, over(lo, hi)) {
                        (false, _) => Color::hex(0x141925),
                        (true, true) => Color::hex(0x33405E),
                        (true, false) => Color::hex(0x222B40),
                    };
                    hud_rect(f, lo, hi, fill);
                    f.text_px((lo + hi) * 0.5, "use the eigenvectors", 12.5, if usable { pal::TEXT } else { pal::FAINT }, Align2::CENTER_CENTER);
                    if usable && over(lo, hi) && f.ptr.clicked {
                        asked = Some(Click::Eigen);
                    }
                    if let Some(why) = why {
                        for (i, l) in wrap_words(why, width, SMALL).into_iter().enumerate() {
                            f.text_px(Vec2::new(x, hi.y + 5.0 + i as f32 * (SMALL + 4.0)), l, SMALL, pal::DIM, Align2::LEFT_TOP);
                        }
                    }
                }
            }
            y += line_height(line, width);
        }
        if lay.key_inside {
            y += 2.0;
            for part in key_parts(self.mode) {
                f.text_px(Vec2::new(x, y), part, 12.0, pal::DIM, Align2::LEFT_TOP);
                y += KEY_LINE;
            }
        }
        asked
    }

    /// The picture: the two grids, the arrows, the walks, and their names.
    fn picture(&self, f: &mut Frame, room: &mut Room) {
        // drawn where the tips are drawn; every number written is that of the tips themselves
        let [b1, b2, v] = self.drawn;
        let seen = Grid::of(b1, b2);
        let said = Grid::of(self.tips[0], self.tips[1]);
        let (lo, hi) = f.cam.flat_bounds();
        let wpp = f.cam.world_per_px(Vec3::ZERO) * f.px;

        // the square grid, as far as the view reaches (the number of lines changes seldom)
        let reach = lo.abs().max(hi.abs()).max_element() as f64;
        let n = lines_for(reach + 1.0);
        let every = if 1.0 / wpp < 4.0 { 10 } else { 1 };
        grid_lines(f, DMat2::IDENTITY, n, every, SQUARE_COL, 0.4);
        // (names keep off the two axes, where the axes' own numbers are)
        let far = n as f32;
        room.line(f, Vec3::new(-far, 0.0, 0.0), Vec3::new(far, 0.0, 0.0));
        room.line(f, Vec3::new(0.0, -far, 0.0), Vec3::new(0.0, far, 0.0));

        // the grid of b₁ and b₂: the whole numbers of steps the view's corners need
        // (none while the tips are no basis, whatever the tips gliding there are)
        let basis = said.inv.is_ok() && seen.inv.is_ok();
        match seen.inv {
            Ok(inv) if basis => {
                let corners = [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)].map(|p| inv * p.as_dvec2());
                let reach = corners.iter().fold(0.0f64, |m, c| m.max(c.abs().max_element()));
                // (lines closer together on the screen than a few pixels are drawn
                // a tenth as often, and fainter where they are still close)
                let spacing = |k: usize| (seen.det().abs() / seen.b.col(1 - k).length() / wpp as f64) as f32;
                // (a view squeezed to nothing has no spacing to speak of)
                let gap = Some(spacing(0).min(spacing(1))).filter(|g| g.is_finite()).unwrap_or(100.0);
                let every = if gap < 4.0 { 10 } else { 1 };
                let alpha = 0.42 * ((gap * every as f32 - 4.0) / 30.0).clamp(0.12, 1.0);
                grid_lines(f, seen.b, lines_for(reach + 1.0), every, GRID_COL, alpha);
            }
            _ => {
                // no grid: the walks reach only the line the two lie on (or the origin)
                let along = if b1.length() >= b2.length() { b1 } else { b2 };
                if along.length() > 1e-6 {
                    let d = along.normalize() * (reach as f32 * 2.0 + 10.0);
                    f.line(-d, d, 2.0, GRID_COL.a(0.9));
                }
            }
        }

        let (p1, p2, pv) = (exact(b1), exact(b2), exact(v));
        if self.mode == ARROW {
            // v's walk in the square grid, faint: so far along x, then up
            let corner = Vec2::new(v.x, 0.0);
            f.dashed(Vec2::ZERO, corner, 1.6, SQUARE_COL.a(0.9));
            f.dashed(corner, v, 1.6, SQUARE_COL.a(0.9));
            room.line(f, Vec3::ZERO, corner.extend(0.0));
            room.line(f, corner.extend(0.0), v.extend(0.0));
            // and in the grid of b
            let legs = seen.coords(pv).filter(|_| basis).map(|c| draw_walk(f, room, p1, p2, c, (lo, hi)));
            // v's shadows on the lines of b₁ and b₂
            if self.shadows && basis {
                for b in [b1, b2] {
                    let foot = b * (b.dot(v) / b.length_squared());
                    if (v - foot).length() > 3.0 * wpp {
                        f.dashed(v, foot, 1.5, pal::WHITE.a(0.85));
                        room.line(f, v.extend(0.0), foot.extend(0.0));
                        let size = 11.0 * wpp;
                        if (v - foot).length() > 2.0 * size && foot.length() > 1.5 * size {
                            right_angle(f, foot.extend(0.0), -foot.extend(0.0), (v - foot).extend(0.0), size, pal::WHITE.a(0.85));
                        }
                    }
                    f.dot(foot, 6.0, pal::WHITE);
                }
            }
            self.arrows(f, room, b1, b2);
            f.arrow(Vec2::ZERO, v, 3.0, V_COL);
            room.line(f, Vec3::ZERO, v.extend(0.0));
            self.handles_and_names(f, room, b1, b2);
            room.label(f, "v", v.extend(0.0), "$v$", 17.0, V_COL, outwards(f, v), 9.0, false);

            // the numbers of each walk beside its legs
            if let (Some(legs), Some(c)) = (legs, said.coords(exact(self.tips[2]))) {
                let names = [steps(c.x, "b_1"), steps(c.y, "b_2")];
                for (k, leg) in legs.iter().enumerate() {
                    if let Some((a, b, mid)) = leg {
                        let other = if k == 0 { v } else { Vec2::ZERO };
                        let along = along_leg(*a, *b, *mid);
                        room.label_any(f, ("leg", k), &along, &names[k], 14.0, [B1_COL, B2_COL][k].mix(pal::WHITE, 0.2), away_from(f, *a, *b, other), 10.0, false);
                    }
                }
            }
            let tv = exact(self.tips[2]);
            if v.x.abs() > 20.0 * wpp {
                room.label(f, "x", (corner * 0.5).extend(0.0), &steps(tv.x, r"\hat\imath"), 12.5, square_text(), away_from(f, Vec2::ZERO, corner, v), 6.0, true);
            }
            if v.y.abs() > 20.0 * wpp {
                room.label(f, "y", ((corner + v) * 0.5).extend(0.0), &steps(tv.y, r"\hat\jmath"), 12.5, square_text(), away_from(f, corner, v, Vec2::ZERO), 6.0, true);
            }
        } else {
            let a = self.a;
            let (ab1, ab2) = (a * b1, a * b2);
            // the walk of the column followed: where b₁ (or b₂) goes, in steps of b₁ and b₂
            let k = self.pick.min(1);
            let ea = exact_map(a);
            let legs = seen.of_map(ea).filter(|_| basis).map(|m| draw_walk(f, room, p1, p2, m.col(k), (lo, hi)));
            self.arrows(f, room, b1, b2);
            for (j, (ab, c)) in [(ab1, B1_COL), (ab2, B2_COL)].into_iter().enumerate() {
                f.arrow(Vec2::ZERO, ab, if j == k { 3.0 } else { 2.2 }, pale(c).a(if j == k { 1.0 } else { 0.7 }));
                room.line(f, Vec3::ZERO, ab.extend(0.0));
            }
            if self.show_v {
                f.dashed(Vec2::ZERO, v, 1.6, V_COL.a(0.85));
                f.arrow(Vec2::ZERO, a * v, 2.6, V_COL);
                room.line(f, Vec3::ZERO, v.extend(0.0));
                room.line(f, Vec3::ZERO, (a * v).extend(0.0));
            }
            self.handles_and_names(f, room, b1, b2);
            for (j, (ab, c, name)) in [(ab1, B1_COL, "$Ab_1$"), (ab2, B2_COL, "$Ab_2$")].into_iter().enumerate() {
                room.label(f, ("ab", j), ab.extend(0.0), name, 15.0, pale(c), outwards(f, ab), 8.0, false);
            }
            if let (Some(legs), Some(m)) = (legs, said.of_map(ea)) {
                let names = [steps(m.col(k).x, "b_1"), steps(m.col(k).y, "b_2")];
                let end = [ab1, ab2][k];
                for (j, leg) in legs.iter().enumerate() {
                    if let Some((p, q, mid)) = leg {
                        let other = if j == 0 { end } else { Vec2::ZERO };
                        let along = along_leg(*p, *q, *mid);
                        room.label_any(f, ("leg", j), &along, &names[j], 14.0, [B1_COL, B2_COL][j].mix(pal::WHITE, 0.2), away_from(f, *p, *q, other), 10.0, false);
                    }
                }
            }
            if self.show_v {
                room.label(f, "v", v.extend(0.0), "$v$", 16.0, V_COL, outwards(f, v), 9.0, true);
                room.label(f, "Av", (a * v).extend(0.0), "$Av$", 16.0, V_COL, outwards(f, a * v), 8.0, true);
            }
        }
        if let Err(why) = said.inv {
            room.label(f, "none", Vec3::ZERO, &format!("{why}: not a basis"), 13.5, pal::TEXT, Vec2::new(0.0, 1.0), 16.0, false);
        }
    }

    fn arrows(&self, f: &mut Frame, room: &mut Room, b1: Vec2, b2: Vec2) {
        for (b, c) in [(b1, B1_COL), (b2, B2_COL)] {
            f.arrow(Vec2::ZERO, b, 3.4, c);
            room.line(f, Vec3::ZERO, b.extend(0.0));
        }
    }

    fn handles_and_names(&self, f: &mut Frame, room: &mut Room, b1: Vec2, b2: Vec2) {
        for (k, (b, other, c, name)) in [(b1, b2, B1_COL, "$b_1$"), (b2, b1, B2_COL, "$b_2$")].into_iter().enumerate() {
            // beyond the tip and on the side away from the other arrow, where the walks
            // and v (which lie between the two, as often as not) are not
            let out = outwards(f, b).normalize_or_zero();
            let away = away_from(f, Vec2::ZERO, b, other).normalize_or_zero();
            let towards = if other.length() > 1e-6 && b.perp_dot(other).abs() > 1e-6 * b.length() * other.length() { out + away } else { out };
            room.label(f, ("b", k), b.extend(0.0), name, 17.0, c, towards, 9.0, false);
        }
    }
}

impl Scene for ChangeOfBasis {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        // (the square grid is drawn by the scene: it is the grid of î and ĵ, one line
        // for every whole step, whatever the zoom)
        s.grid = false;
    }

    /// Only an animation moves by itself, and a tip gliding to where it has stepped.
    fn animated(&self) -> bool {
        self.tween.is_some() || !self.arrived()
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"Coordinates are directions for a walk. In the grid of two arrows $\textcolor{pink}{b_1}$ and $\textcolor{green}{b_2}$, the numbers $(2, 1)$ mean: 2 steps of $b_1$, then 1 step of $b_2$. The same arrow has other numbers in another grid, and the same map another matrix. Drag the tips of $b_1$, $b_2$ and $v$.");

        ui::section(ui, "Things to try");
        for (k, (name, look_for)) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, name, look_for) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Show");
        if ui::choice(ui, &MODES, &mut self.mode) {
            self.new_view = true;
        }
        if self.mode == ARROW {
            if ui.checkbox(&mut self.shadows, ui::label_text(ui, "$v$'s shadows on the lines of $b_1$ and $b_2$", 12.5, None, false)).changed() {
                self.new_view = true;
            }
            ui::note(ui, r"Where $v$ lands when dropped at a right angle onto each line. When $b_1$ and $b_2$ are at right angles the walk turns exactly at the shadow on $b_1$'s line, and when they also have length 1 the coordinates are the dot products $b_1 \cdot v$ and $b_2 \cdot v$.");
        } else {
            ui::section(ui, "Matrix $A$");
            let names: Vec<&str> = PRESETS.iter().map(|p| p.0).collect();
            if let Some(i) = ui::buttons(ui, &names) {
                self.a = Mat2::from_cols_array(&PRESETS[i].1);
                self.new_view = true;
            }
            mat2_edit(ui, &mut self.a);
            ui::note(ui, r"stretches: by 2 and by ½ along two slanted lines; symmetric: by 3 and 1 along two lines at right angles; shear: one line stays, none other; turn: a quarter turn, no line stays; $2I$: every arrow doubles. $A$ is written in the square grid: its columns are where $\hat\imath$ and $\hat\jmath$ go.");
            let blocked = self.eigen_blocked();
            if ui.add_enabled(blocked.is_none() && self.tween.is_none(), egui::Button::new("use the eigenvectors")).clicked() {
                self.use_eigenvectors();
            }
            match &blocked {
                Some(why) => ui::note(ui, why),
                None => ui::note(ui, "Turns $b_1$ and $b_2$ onto the lines $A$ does not turn (the same button is on the picture)."),
            }
            let mut pick = self.pick;
            if ui::choice(ui, &["follow column 1: $Ab_1$", "column 2: $Ab_2$"], &mut pick) {
                self.pick = pick;
            }
            ui.checkbox(&mut self.show_v, ui::label_text(ui, "$v$ and $Av$", 12.5, None, false));
        }
        ui::note(ui, "The tips move in steps of 0.1, so that the numbers can be checked by hand.");

        ui::section(ui, "Reading the picture");
        let mut key: Vec<(Color, &str)> = vec![
            (SQUARE_COL, r"grey grid: the square grid of $\hat\imath = (1, 0)$ and $\hat\jmath = (0, 1)$; grey dashes: $v$'s walk along it"),
            (GRID_COL, "blue grid: the grid of $b_1$ and $b_2$, one step of each to a cell"),
            (B1_COL, "pink: $b_1$, and the steps taken along it"),
            (B2_COL, "green: $b_2$, and the steps taken along it"),
        ];
        if self.mode == ARROW {
            key.push((V_COL, "white: the arrow $v$"));
            if self.shadows {
                key.push((pal::WHITE, "white dashes with corners: $v$ dropped at a right angle onto the lines of $b_1$ and $b_2$"));
            }
        } else {
            key.push((pale(B1_COL), "pale pink: $Ab_1$, where $A$ sends $b_1$"));
            key.push((pale(B2_COL), "pale green: $Ab_2$, where $A$ sends $b_2$"));
            key.push((PICK_COL, "cyan box: the column of $B^{-1}AB$ whose walk is drawn"));
            if self.show_v {
                key.push((V_COL, "white: $v$ (dashed) and $Av$, where $A$ sends it"));
            }
        }
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, "Basis:", "two arrows $b_1, b_2$, neither of length 0 and not on one line. Every arrow of the plane is then exactly one walk $c_1 b_1 + c_2 b_2$.");
        ui::term(ui, "Coordinates:", r"the numbers of steps $c = (c_1, c_2)$ in that walk. In the square grid of $\hat\imath$ and $\hat\jmath$ they are an arrow's own $x$ and $y$.");
        ui::term(ui, "$B$:", r"the matrix with $b_1$ and $b_2$ as its columns. It turns coordinates into the arrow, $v = Bc$; its inverse turns the arrow into coordinates, $c = B^{-1}v$.");
        ui::term(ui, "Orthonormal:", r"at right angles and of length 1. Then $B^{-1} = B^\top$ ($B$ with rows and columns swapped), and each coordinate is a dot product.");
        if self.mode == MAP {
            ui::term(ui, r"$B^{-1}AB$:", r"the map $A$ written in the basis $b$: $B$ turns coordinates into an arrow, $A$ moves it, $B^{-1}$ turns the result back into coordinates. Its columns are $Ab_1$ and $Ab_2$ in steps of $b_1$ and $b_2$.");
            ui::term(ui, "Eigenvector:", r"an arrow that $A$ only stretches, by its eigenvalue $\lambda$, without turning it off its line.");
            ui::term(ui, "Diagonal:", r"with zeros off the diagonal: the map only stretches along $b_1$ and $b_2$. $A = BDB^{-1}$, with $D$ diagonal, is $A$ diagonalised.");
            ui::term(ui, "Trace, determinant:", "the sum of the diagonal entries, and the factor by which the map scales areas. They belong to the map, not to the grid: they are the same in every basis.");
        }

        ui::section(ui, "Numbers");
        let [b1, b2, v] = self.tips;
        let g = Grid::of(b1, b2);
        // (as formulas, every one: `say` writes a large number with a power of ten)
        let pair = |p: DVec2| format!("$({}, {})$", say(p.x), say(p.y));
        let two = |p: f64, q: f64| format!("${}, {}$", say(p), say(q));
        ui::readout_colored(ui, "$b_1$", pair(exact(b1)), B1_COL);
        ui::readout_colored(ui, "$b_2$", pair(exact(b2)), B2_COL);
        ui::readout(ui, r"$\det B$", format!("${}$", say(g.det())));
        if self.mode == ARROW {
            ui::readout_colored(ui, "$v$", pair(exact(v)), V_COL);
            ui::readout(ui, r"$c = B^{-1}v$", g.coords(exact(v)).map_or("none: not a basis".to_string(), pair));
            if g.inv.is_ok() {
                ui::readout(ui, r"$b_1 \cdot v$, $b_2 \cdot v$", two(g.b.x_axis.dot(exact(v)), g.b.y_axis.dot(exact(v))));
            }
        } else {
            let a = exact_map(self.a);
            ui::readout(ui, "$A$: trace, det", two(a.x_axis.x + a.y_axis.y, a.determinant()));
            match g.of_map(a) {
                Some(m) => ui::readout(ui, "$B^{-1}AB$: trace, det", two(m.x_axis.x + m.y_axis.y, m.determinant())),
                None => ui::readout(ui, "$B^{-1}AB$", "none: not a basis"),
            }
            match eig2(self.a) {
                Eig2::Real { l1, l2, .. } => ui::readout(ui, "eigenvalues of $A$", two(l1 as f64, l2 as f64)),
                Eig2::Complex { re, im } => ui::readout(ui, "eigenvalues of $A$", format!(r"${} \pm {}i$", say(re as f64), say(im as f64))),
            }
        }
        if let Err(why) = g.inv {
            ui::note(ui, &format!("{why}: they are not a basis, so there are no coordinates in them."));
        }
        ui::see_also(ui, &[("span", "walks along arrows: span and independence"), ("compose", "products and inverses of maps"), ("eigen", "finding eigenvalues"), ("svd", "two right-angled bases that make any matrix diagonal")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        // the canvas decides where the panel goes, and with that where the picture is
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let view = f.view_size();
        let covered = view.x < 160.0;
        if !covered && ((view - self.canvas).abs().max_element() > 24.0 || narrow(view) != narrow(self.canvas) || (f.top_inset - self.inset).abs() > 1.0) {
            (self.canvas, self.inset) = (view, f.top_inset);
            self.new_view = true;
        }
        self.advance(f.dt);
        let labels_before = f.dl.labels.len();

        let lines = self.lines_at(self.tips);
        let lay = Layout::of(view, f.top_inset, self.panel_size(&lines), key_parts(self.mode).len());
        let pointer = f.ptr.pos.map(|p| p / f.px);
        if pointer.is_some_and(|p| lay.covers(p)) {
            // the panel is for its buttons and columns: no panning from there, and two
            // quick clicks are two clicks, not a reset of the view
            f.handles.capture();
            f.handles.claim_clicks();
        }

        // the handles first, so that everything is drawn where they now are (one under
        // the panel is left alone, unless it is being dragged already)
        let colours = [B1_COL, B2_COL, V_COL];
        if self.tween.is_none() {
            for k in 0..3 {
                if k == 2 && !self.v_shown() {
                    continue;
                }
                let hidden = on_screen(f, self.grip[k].extend(0.0)).is_some_and(|p| lay.covers(p));
                if hidden && !f.handles.dragging() {
                    continue;
                }
                let mut grip = self.grip[k];
                if f.drag2(1 + k as u32, &mut grip, colours[k]) {
                    self.grip[k] = grip;
                    self.tips[k] = settle(grip);
                }
            }
            if !f.handles.dragging() {
                self.grip = self.tips;
            }
        } else {
            for k in 0..3 {
                if k < 2 || self.v_shown() {
                    f.dot(self.tips[k], 12.0, colours[k]);
                }
            }
        }
        // The tips are drawn gliding to where they have stepped; not while the scene's
        // time stands still (nothing glides then, and a drag must still be seen).
        // (By the clock, as the view's own glide is: it answers the hand.)
        self.glide(f.real_dt, f.dt <= 0.0);

        let mut room = Room::new(f);
        if let Some((lo, hi)) = lay.panel {
            room.wall(lo - Vec2::splat(4.0), hi + Vec2::splat(4.0));
        }
        if lay.caption_top() < view.y {
            room.wall(Vec2::new(0.0, lay.caption_top()), view);
        }
        self.picture(f, &mut room);
        f.axes2();

        // nothing written into the picture may show through the panel, or under the
        // application's buttons
        if let Some((lo, hi)) = lay.panel {
            let (lo, hi) = (lo - Vec2::splat(4.0), hi + Vec2::splat(4.0));
            let (mut k, inset) = (0, f.top_inset);
            f.dl.labels.retain(|l| {
                k += 1;
                k <= labels_before || !(l.pos.y < inset || l.pos.cmpge(lo).all() && l.pos.cmple(hi).all())
            });
        }
        if let Some(rect) = lay.panel {
            match self.draw_panel(f, &lines, rect, &lay, pointer) {
                Some(Click::Column(j)) => self.pick = j,
                Some(Click::Eigen) if self.tween.is_none() => self.use_eigenvectors(),
                _ => {}
            }
        }
        if !lay.key_inside {
            bottom_caption(f, &key_parts(self.mode), 12.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, played to the end at 60 frames a second.
    fn run(k: usize) -> ChangeOfBasis {
        let mut scene = ChangeOfBasis::create();
        scene.set_up(k);
        for _ in 0..200 {
            scene.advance(1.0 / 60.0);
        }
        assert!(scene.tween.is_none() && scene.arrived());
        scene
    }

    fn grid(s: &ChangeOfBasis) -> Grid {
        Grid::of(s.tips[0], s.tips[1])
    }

    fn coords(s: &ChangeOfBasis) -> DVec2 {
        grid(s).coords(exact(s.tips[2])).expect("a basis")
    }

    fn pair(p: DVec2) -> (String, String) {
        (say(p.x), say(p.y))
    }

    /// The words of the panel, line by line.
    fn words(s: &ChangeOfBasis) -> Vec<String> {
        s.lines_at(s.tips).iter().filter_map(|l| if let Line::Words(t, _) = l { Some(t.clone()) } else { None }).collect()
    }

    /// The angle between two arrows, in degrees.
    fn degrees_between(a: Vec2, b: Vec2) -> f64 {
        a.as_dvec2().angle_to(b.as_dvec2()).abs().to_degrees()
    }

    /// The most the coordinates can change when v moves by `by`: the largest
    /// stretch of B⁻¹ (its largest singular value), times `by`.
    fn sensitivity(g: &Grid, by: f64) -> f64 {
        let inv = g.inv.expect("a basis");
        let (a, b, c, d) = (inv.x_axis.x, inv.y_axis.x, inv.x_axis.y, inv.y_axis.y);
        let (sum, det) = (a * a + b * b + c * c + d * d, (a * d - b * c).abs());
        ((sum + (sum * sum - 4.0 * det * det).max(0.0).sqrt()) * 0.5).sqrt() * by
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // the same arrow, new numbers: the square grid turns into the grid of b while
        // v stays put; (2.9, 1.8) in the square grid is (2, 1) in the grid of b, and
        // v = Bc, c = B⁻¹v
        let mut s = ChangeOfBasis::create();
        s.set_up(0);
        assert_eq!(s.tips, [Vec2::X, Vec2::Y, OPENING[2]], "it starts from the square grid");
        let v = s.tips[2];
        let mut frames = 0;
        while s.tween.is_some() {
            s.advance(1.0 / 60.0);
            assert_eq!(s.tips[2], v, "v moves");
            assert!(grid(&s).right_handed());
            frames += 1;
        }
        assert!(frames > 60 && s.tips == OPENING && s.mode == ARROW, "{frames} frames");
        let (g, c) = (grid(&s), coords(&s));
        assert_eq!(pair(exact(s.tips[2])), ("2.9".into(), "1.8".into()));
        assert_eq!(pair(c), ("2".into(), "1".into()));
        assert!((g.b * c - exact(v)).length() < 1e-12 && (g.inv.unwrap() * exact(v) - c).length() < 1e-12);
        let w = words(&s);
        assert!(w[0].ends_with(r"$v = 2.9 \hat\imath + 1.8 \hat\jmath$") && w[1].ends_with("$v = 2 b_1 + 1 b_2$"), "{w:?}");

        // a basis at right angles: length 1, at right angles; each coordinate is the
        // shadow on that arrow's line, a dot product, 3.18 and -1.24 (behind the
        // start); B⁻¹ = Bᵀ
        let s = run(1);
        let (g, c, v) = (grid(&s), coords(&s), exact(s.tips[2]));
        assert_eq!(pair(g.b.x_axis), ("0.6".into(), "0.8".into()));
        assert_eq!(pair(g.b.y_axis), ("−0.8".into(), "0.6".into()));
        assert!(g.orthonormal() && s.shadows && degrees_between(s.tips[0], s.tips[1]) - 90.0 < 1e-5);
        for (k, b) in [g.b.x_axis, g.b.y_axis].into_iter().enumerate() {
            assert!((b.length() - 1.0).abs() < 1e-12);
            let foot = b * (b.dot(v) / b.length_squared());
            assert!((foot - b * c[k]).length() < 1e-12 && (b.dot(v) - c[k]).abs() < 1e-12, "the shadow on line {k}");
        }
        assert_eq!(pair(c), ("3.18".into(), "−1.24".into()));
        assert!(c.y < 0.0 && (g.b.y_axis * c.y).dot(g.b.y_axis) < 0.0);
        assert!((g.inv.unwrap() - g.b.transpose()).to_cols_array().iter().all(|e| e.abs() < 1e-12));
        assert!(words(&s).iter().any(|l| l.starts_with(r"orthonormal: $B^{-1} = B^\top$")));

        // a nearly flat basis: 4.4 degrees apart, det B = 0.1; v = (1, 0) of length 1
        // is c = (10, -10); B⁻¹ is larger, and a small move of v changes c more
        let s = run(2);
        let (g, c, v) = (grid(&s), coords(&s), exact(s.tips[2]));
        assert_eq!(format!("{:.1}", degrees_between(s.tips[0], s.tips[1])), "4.4");
        assert!(say(g.det()) == "0.1" && v == DVec2::new(1.0, 0.0) && pair(c) == ("10".into(), "−10".into()), "{c}");
        let opening = Grid::of(OPENING[0], OPENING[1]);
        let biggest = |m: DMat2| m.to_cols_array().iter().fold(0.0f64, |s, e| s.max(e.abs()));
        assert!(biggest(g.inv.unwrap()) > 5.0 * biggest(opening.inv.unwrap()));
        // (a move of v by a tenth can change the coordinates by more than 1, against
        // less than a fifth in the opening basis)
        assert!(sensitivity(&g, 0.1) > 1.0 && sensitivity(&opening, 0.1) < 0.2, "{} {}", sensitivity(&g, 0.1), sensitivity(&opening, 0.1));
        let mut rng = Rng::new(5);
        let worst = (0..400).map(|_| DVec2::new(rng.normal() as f64, rng.normal() as f64).normalize() * 0.1).map(|d| (g.inv.unwrap() * d).length()).fold(0.0, f64::max);
        assert!(worst > 1.0 && worst <= sensitivity(&g, 0.1) + 1e-9, "{worst}");

        // the same map, another matrix: A stretches to twice along one line and to
        // half along another; B⁻¹AB has first column (1.2, 1.12), which is where b₁
        // goes; the numbers differ but trace 2.5 and det 1 are the same
        let s = run(3);
        let (g, a) = (grid(&s), exact_map(s.a));
        assert!(s.mode == MAP && s.tips[..2] == OPENING[..2]);
        assert!((a * DVec2::new(1.0, 1.0) - DVec2::new(2.0, 2.0)).length() < 1e-12 && (a * DVec2::new(-1.0, 2.0) - DVec2::new(-0.5, 1.0)).length() < 1e-12);
        let m = g.of_map(a).unwrap();
        assert_eq!(pair(m.x_axis), ("1.2".into(), "1.12".into()));
        assert!((a * g.b.x_axis - (g.b.x_axis * m.x_axis.x + g.b.y_axis * m.x_axis.y)).length() < 1e-12, "Ab₁ is the walk");
        assert!((m - a).to_cols_array().iter().any(|e| e.abs() > 0.1));
        for x in [a, m] {
            assert_eq!((say(x.x_axis.x + x.y_axis.y), say(x.determinant())), ("2.5".into(), "1".into()));
        }
        assert!(words(&s).iter().any(|l| l == "column 1:  $Ab_1 = 1.2 b_1 + 1.12 b_2$"), "{:?}", words(&s));

        // the eigenvector basis: b₁ and b₂ turn (keeping their lengths) onto A's
        // eigenvectors, a basis every frame of the way; then Ab₁ = 2 b₁, Ab₂ = 0.5 b₂,
        // B⁻¹AB is diagonal with 2 and 0.5, and A = BDB⁻¹
        let mut s = ChangeOfBasis::create();
        s.set_up(EIGEN_TRIAL);
        let lengths = [s.tips[0].length(), s.tips[1].length()];
        assert_eq!(s.tips, trial(3).1, "it starts from the experiment before");
        while s.tween.is_some() {
            s.advance(1.0 / 60.0);
            assert!(grid(&s).right_handed() && grid(&s).det() > 0.5, "{:?}", s.tips);
        }
        let (g, a) = (grid(&s), exact_map(s.a));
        let m = g.of_map(a).unwrap();
        assert!(diagonal(m) && pair(DVec2::new(m.x_axis.x, m.y_axis.y)) == ("2".into(), "0.5".into()), "{m}");
        for (k, l) in [(0, 2.0), (1, 0.5)] {
            let b = g.b.col(k);
            assert!((a * b - b * l).length() < 1e-6 * b.length(), "b{k} is not an eigenvector");
            assert!((b.length() - lengths[k] as f64).abs() < 1e-6);
        }
        let d = DMat2::from_diagonal(DVec2::new(m.x_axis.x, m.y_axis.y));
        assert!((g.b * d * g.inv.unwrap() - a).to_cols_array().iter().all(|e| e.abs() < 1e-6));
        let w = words(&s);
        assert!(w.iter().any(|l| l == "column 1:  $Ab_1 = 2 b_1 + 0 b_2$") && w.iter().any(|l| l.starts_with("diagonal")), "{w:?}");
        // and a turn has no such lines: the button says why
        s.a = Mat2::from_cols_array(&PRESETS[3].1);
        assert!(s.eigen_blocked().is_some_and(|why| why.contains("no real eigenvectors")));
    }

    /// The note under the shadows: with b₁ and b₂ at right angles (of any lengths) the
    /// walk turns exactly at v's shadow on the line of b₁; with length 1 as well, the
    /// coordinates are the dot products; at another angle the walk turns elsewhere
    /// (unless v lies on the line of b₁).
    #[test]
    fn the_walk_turns_at_the_shadow_only_at_right_angles() {
        let mut rng = Rng::new(29);
        let (mut elsewhere, mut checked) = (0, 0);
        for i in 0..300 {
            let mut r = || (rng.normal() * 1.5 / STEP).round() * STEP;
            let (b1, v) = (Vec2::new(r(), r()), Vec2::new(r(), r()));
            // (b₁ turned a quarter turn, times a whole number, so b₂ stays on the tenths)
            let b2 = b1.perp() * [1.0, 2.0, -1.0, -3.0][i % 4];
            let g = Grid::of(b1, b2);
            let Some(c) = g.coords(exact(v)) else { continue };
            let p = g.b.x_axis;
            let foot = p * (p.dot(exact(v)) / p.length_squared());
            assert!((p * c.x - foot).length() < 1e-9 * (1.0 + foot.length()), "{b1} {b2} {v}");
            let turn = Vec2::from_angle(r());
            let u = Grid::of(turn, turn.perp());
            let c = u.coords(exact(v)).unwrap();
            assert!(u.orthonormal() && (c - DVec2::new(u.b.x_axis.dot(exact(v)), u.b.y_axis.dot(exact(v)))).length() < 1e-5, "{turn}");
            // and the same arrows a little off a right angle
            let g = Grid::of(b1, b2 + b1 * 0.3);
            let c = g.coords(exact(v)).unwrap();
            if c.y.abs() > 1e-6 {
                assert!((p * c.x - foot).length() > 1e-6);
                elsewhere += 1;
            }
            checked += 1;
        }
        assert!(checked > 250 && elsewhere > 250, "{checked} {elsewhere}");
    }

    /// For any map and any basis: the columns of B⁻¹AB are where b₁ and b₂ go, in
    /// steps of b₁ and b₂, and the trace and the determinant are those of A. And
    /// v = Bc for v's coordinates c.
    #[test]
    fn one_map_has_one_trace_and_determinant_in_every_basis() {
        let mut rng = Rng::new(11);
        let mut checked = 0;
        for _ in 0..500 {
            let mut r = || (rng.normal() * 1.5 / STEP).round() * STEP;
            let (b1, b2, v) = (Vec2::new(r(), r()), Vec2::new(r(), r()), Vec2::new(r(), r()));
            let a = Mat2::from_cols(Vec2::new(r(), r()), Vec2::new(r(), r()));
            let g = Grid::of(b1, b2);
            let Some(m) = g.of_map(exact_map(a)) else { continue };
            let ea = exact_map(a);
            let size = ea.to_cols_array().iter().fold(1.0f64, |s, e| s.max(e.abs()));
            let scale = size * g.inv.unwrap().to_cols_array().iter().fold(1.0f64, |s, e| s.max(e.abs())) * g.b.to_cols_array().iter().fold(1.0f64, |s, e| s.max(e.abs()));
            assert!(((m.x_axis.x + m.y_axis.y) - (ea.x_axis.x + ea.y_axis.y)).abs() < 1e-9 * scale, "{a} {b1} {b2}");
            assert!((m.determinant() - ea.determinant()).abs() < 1e-9 * scale * scale, "{a} {b1} {b2}");
            for k in 0..2 {
                assert!((g.b * m.col(k) - ea * g.b.col(k)).length() < 1e-9 * scale * g.b.col(k).length().max(1.0));
            }
            let c = g.coords(exact(v)).unwrap();
            assert!((g.b * c - exact(v)).length() < 1e-9 * (1.0 + c.length() * g.b.x_axis.length().max(g.b.y_axis.length())));
            checked += 1;
        }
        assert!(checked > 400, "{checked}");
    }

    /// The button: for a map with two lines of eigenvectors, from any basis, b₁ and b₂
    /// end on them (B⁻¹AB diagonal, with A's eigenvalues), turning the way they did and
    /// a basis all the way; for the others it is not offered, and says why.
    #[test]
    fn the_eigenvector_basis_makes_the_matrix_diagonal() {
        let mut rng = Rng::new(17);
        let mut done = 0;
        for _ in 0..300 {
            let mut r = || (rng.normal() * 1.4 / STEP).round() * STEP;
            let mut s = ChangeOfBasis::create();
            s.mode = MAP;
            s.a = Mat2::from_cols(Vec2::new(r(), r()), Vec2::new(r(), r()));
            s.tips = [Vec2::new(r(), r()), Vec2::new(r(), r()), Vec2::new(r(), r())];
            let before = grid(&s);
            let Eig2::Real { l1, l2, v1, v2 } = eig2(s.a) else {
                assert!(s.eigen_blocked().is_some());
                continue;
            };
            if v1.perp_dot(v2).abs() < 1e-2 || before.inv.is_err() {
                continue;
            }
            s.use_eigenvectors();
            while s.tween.is_some() {
                s.advance(1.0 / 60.0);
                let g = grid(&s);
                assert!(g.inv.is_ok() && (g.det() > 0.0) == (before.det() > 0.0), "{:?} from {:?}", s.tips, before);
            }
            let m = grid(&s).of_map(exact_map(s.a)).unwrap();
            let size = l1.abs().max(l2.abs()).max(1.0) as f64;
            assert!(m.x_axis.y.abs() < 1e-4 * size / v1.perp_dot(v2).abs() as f64 && m.y_axis.x.abs() < 1e-4 * size / v1.perp_dot(v2).abs() as f64, "{m} for {}", s.a);
            let mut diag = [m.x_axis.x, m.y_axis.y];
            diag.sort_by(f64::total_cmp);
            let mut want = [l1 as f64, l2 as f64];
            want.sort_by(f64::total_cmp);
            assert!((diag[0] - want[0]).abs() < 1e-4 * size && (diag[1] - want[1]).abs() < 1e-4 * size, "{diag:?} {want:?}");
            assert!(s.eigen_blocked().is_some(), "offered again once there");
            done += 1;
        }
        assert!(done > 100, "{done}");
        // the presets that have no such basis say why, and the symmetric one's is at right angles
        let blocked = |c: [f32; 4]| {
            let mut s = ChangeOfBasis::create();
            s.a = Mat2::from_cols_array(&c);
            s.eigen_blocked()
        };
        assert!(blocked(PRESETS[2].1).is_some_and(|w| w.contains("only one line") && w.contains(r"$\lambda = 1$ twice")), "shear");
        assert!(blocked(PRESETS[3].1).is_some_and(|w| w.contains("turns every direction")), "turn");
        assert!(blocked(PRESETS[4].1).is_some_and(|w| w.starts_with("$A = 2I$")), "2I");
        assert!(blocked([0.0; 4]).is_some_and(|w| w.starts_with("$A = 0$")), "zero");
        assert!(blocked(PRESETS[0].1).is_none() && blocked(PRESETS[1].1).is_none());
        let mut s = ChangeOfBasis::create();
        s.a = Mat2::from_cols_array(&PRESETS[1].1);
        s.use_eigenvectors();
        s.advance(10.0);
        let g = grid(&s);
        assert!(g.b.x_axis.dot(g.b.y_axis).abs() < 1e-6 && diagonal(g.of_map(exact_map(s.a)).unwrap()));
        // and with the scaled identity every basis already makes it diagonal
        let s2 = ChangeOfBasis { a: Mat2::from_cols_array(&PRESETS[4].1), ..ChangeOfBasis::create() };
        assert!(diagonal(grid(&s2).of_map(exact_map(s2.a)).unwrap()));
    }

    /// While both ends are bases turning the same way, every pair on the way is one;
    /// the ends are exact; nothing is ever not a number, whatever the ends.
    #[test]
    fn the_arrows_turn_without_passing_through_flat() {
        let mut rng = Rng::new(23);
        let mut point = || Vec2::new(rng.normal(), rng.normal()) * 2.0;
        for _ in 0..500 {
            let (p1, p2, q1, q2) = (point(), point(), point(), point());
            let tween = Tween { from: [p1, p2, Vec2::ZERO], to: [q1, q2, Vec2::ONE], t: 0.0 };
            let same = (p1.perp_dot(p2) > 0.0) == (q1.perp_dot(q2) > 0.0);
            for i in 0..=50 {
                let [b1, b2, v] = tween.at(i as f32 / 50.0);
                assert!(b1.is_finite() && b2.is_finite() && v.is_finite());
                if same {
                    let sin = b1.perp_dot(b2) / (b1.length() * b2.length());
                    let ends = (p1.perp_dot(p2) / (p1.length() * p2.length())).abs().min((q1.perp_dot(q2) / (q1.length() * q2.length())).abs());
                    // (the sine of the angle between them never falls below the smaller of
                    // its values at the two ends: the angle changes evenly, and the sine
                    // is concave between 0 and 180 degrees)
                    assert!(sin * p1.perp_dot(p2).signum() >= ends - 1e-5, "{p1} {p2} -> {q1} {q2} at {i}: {sin} against {ends}");
                }
            }
            assert_eq!(tween.at(1.0), [q1, q2, Vec2::ONE]);
            assert!((tween.at(0.0)[0] - p1).length() < 1e-5 && (tween.at(0.0)[1] - p2).length() < 1e-5);
        }
        // arrows of no length, and opposite ones, still give numbers
        for (p1, p2) in [(Vec2::ZERO, Vec2::Y), (Vec2::X, Vec2::ZERO), (Vec2::ZERO, Vec2::ZERO), (Vec2::X, -Vec2::X), (Vec2::X, Vec2::X)] {
            for e in [0.0, 0.3, 1.0] {
                let (a, b) = turn_pair(p1, p2, Vec2::new(1.0, 1.0), Vec2::new(-1.0, 2.0), e);
                assert!(a.is_finite() && b.is_finite());
                let (a, b) = turn_pair(Vec2::new(1.0, 1.0), Vec2::new(-1.0, 2.0), p1, p2, e);
                assert!(a.is_finite() && b.is_finite());
            }
        }
    }

    /// Tips on one line, of no length, very nearly in line, far out: every number
    /// drawn or written is finite, the panel says what has happened, and the view is a view.
    #[test]
    fn degenerate_tips_leave_everything_finite() {
        let b = OPENING[0];
        let cases = [
            ([b, b * 2.0], Some("lie on one line")),
            ([b, -b], Some("lie on one line")),
            ([b, b], Some("lie on one line")),
            ([Vec2::ZERO, b], Some("$b_1$ has no length")),
            ([b, Vec2::ZERO], Some("$b_2$ has no length")),
            ([Vec2::ZERO, Vec2::ZERO], Some("have no length")),
            ([Vec2::new(4.0, 3.9), Vec2::new(3.9, 3.8)], None),
            ([Vec2::new(1e-7, 0.0), Vec2::new(0.0, 1e-7)], Some("have no length")),
            ([b, b + Vec2::new(0.0, 1e-6)], Some("lie on one line")),
            ([b, b + Vec2::new(0.0, 1e-4)], None),
            ([Vec2::splat(REACH), Vec2::new(-REACH, REACH)], None),
        ];
        for (tips, why) in cases {
            for v in [OPENING[2], Vec2::ZERO, b * 3.0] {
                for mode in [ARROW, MAP] {
                    let mut s = ChangeOfBasis::create();
                    let here = [tips[0], tips[1], v];
                    (s.tips, s.drawn, s.between, s.grip, s.mode, s.shadows) = (here, here, here, here, mode, true);
                    let g = grid(&s);
                    assert_eq!(g.inv.err().map(|w| why.is_some_and(|y| w.contains(y))), why.map(|_| true), "{tips:?}: {:?}", g.inv);
                    for line in s.lines_at(s.tips) {
                        let text = match line {
                            Line::Words(t, _) => t,
                            Line::Matrices(parts) => parts.iter().map(|p| if let Part::Matrix { rows, .. } = p { rows.iter().flatten().map(|(t, _)| t.as_str()).collect::<Vec<_>>().join(" ") } else { String::new() }).collect(),
                            Line::Button(w) => w.unwrap_or_default(),
                        };
                        assert!(!text.contains('?') && !text.contains("NaN") && !text.contains("inf"), "{tips:?}: {text}");
                    }
                    assert!(s.home().iter().all(|p| p.is_finite() && p.length() <= 14.0 + 1e-3));
                    assert!(matches!(s.the_view(), View::Fit { center, half } if center.is_finite() && half.is_finite() && half.min_element() > 0.0));
                    if why.is_none() {
                        let c = g.coords(exact(v)).unwrap();
                        assert!(c.is_finite());
                        for (p, q) in [(DVec2::ZERO, g.b.x_axis * c.x), (g.b.x_axis * c.x, exact(v))] {
                            if let Some((p, q)) = clip_disc(p, q) {
                                assert!(p.is_finite() && q.is_finite() && p.length() <= CLIP as f32 * 1.001 && q.length() <= CLIP as f32 * 1.001);
                            }
                        }
                    }
                    // the eigenvector button from here, played out
                    s.mode = MAP;
                    s.use_eigenvectors();
                    s.advance(0.5);
                    assert!(s.tips.iter().all(|p| p.is_finite()));
                    s.advance(5.0);
                    assert!(s.tips.iter().all(|p| p.is_finite()) && grid(&s).inv.is_ok(), "{tips:?} -> {:?}", s.tips);
                }
            }
        }
        // the nearest to flat the tips can be put on the grid of tenths: huge, but numbers
        let g = Grid::of(Vec2::new(4.0, 3.9), Vec2::new(3.9, 3.8));
        let c = g.coords(DVec2::new(0.1, -0.1)).unwrap();
        assert!(c.is_finite() && c.length() > 50.0, "{c}");
        // a walk clipped where it leaves the disc keeps its line
        let (p, q) = clip_disc(DVec2::new(-5000.0, 1.0), DVec2::new(5000.0, 1.0)).unwrap();
        assert!((p.x + 1000.0).abs() < 1.0 && (q.x - 1000.0).abs() < 1.0 && p.y == 1.0);
        assert!(clip_disc(DVec2::new(-5000.0, 2000.0), DVec2::new(5000.0, 2000.0)).is_none());
    }

    #[test]
    fn numbers_are_written_plainly() {
        assert_eq!([say(2.0), say(1.0000000000000002), say(-0.004), say(1.12), say(-1.2399999999999998), say(f64::NAN)], ["2", "1", "0", "1.12", "−1.24", "?"]);
        assert_eq!([say(10.0), say(-10.000000000000002), say(123.456), say(12345.0), say(-1.5e6)], ["10", "−10", "123.5", "12345", "−1.5e6"]);
        assert_eq!([say(-7.3e12), say(1e30), say(9.96e9)], ["−7e12", "1e30", "1e10"]);
        assert!([1e-9, 0.123, -9.999, 99.99, -100.0, 999.4, -99_999.0, 1e7, -9.9e9, -1e30, f64::MAX].iter().all(|v| say(*v).chars().count() <= 6), "a number wider than the panel's cells");
        assert_eq!(walk(2.0, "b_1", 1.0, "b_2"), "2 b_1 + 1 b_2");
        assert_eq!(walk(3.18, "b_1", -1.24, "b_2"), "3.18 b_1 - 1.24 b_2");
        assert_eq!(walk(-0.001, r"\hat\imath", 0.0, r"\hat\jmath"), r"0 \hat\imath + 0 \hat\jmath");
        // the tips move in tenths, no further out than REACH
        assert_eq!(settle(Vec2::new(1.234, -0.051)), Vec2::new(1.2, -0.1));
        assert!((settle(Vec2::new(30.0, 40.0)) - Vec2::new(3.0, 4.0)).length() < 1e-6);
        assert_eq!(exact(Vec2::new(1.2, 0.4)), DVec2::new(1.2, 0.4));
    }

    /// A tip that has stepped is drawn gliding there, exactly there soon after, then
    /// at rest; an experiment is drawn as it stands.
    #[test]
    fn a_stepped_tip_is_drawn_gliding_to_its_place() {
        for hz in [60.0f32, 144.0] {
            let mut s = ChangeOfBasis::create();
            assert!(s.arrived() && !s.animated());
            let from = s.tips[2];
            s.tips[2] += Vec2::new(STEP, -STEP);
            assert!(s.animated());
            let (mut left, mut frames) = ((s.tips[2] - s.drawn[2]).length(), 0);
            while !s.arrived() {
                s.glide(1.0 / hz, false);
                let now = (s.tips[2] - s.drawn[2]).length();
                assert!(now < left && (s.drawn[2] - from).dot(s.tips[2] - from) > 0.0 && s.drawn[..2] == s.tips[..2]);
                left = now;
                frames += 1;
                assert!((frames as f32) / hz < 0.4, "still gliding after {frames} frames at {hz} Hz");
            }
            assert!(!s.animated());
        }
        let mut s = ChangeOfBasis::create();
        s.set_up(2);
        for _ in 0..30 {
            s.advance(1.0 / 60.0);
            assert!(s.arrived() && s.animated());
        }
    }

    /// The panel and the picture share every canvas without leaving it; the panel
    /// keeps its size while the numbers change in every experiment (its brackets
    /// would otherwise jump), and the view fitted to what is left is a view.
    #[test]
    fn the_canvas_is_shared_on_every_screen() {
        for (x, y) in [(1080.0f32, 1000.0f32), (1370.0, 1000.0), (1100.0, 700.0), (390.0, 800.0), (360.0, 640.0), (600.0, 900.0), (844.0, 390.0), (640.0, 480.0), (240.0, 180.0), (2000.0, 1300.0), (8.0, 8.0)] {
            let view = Vec2::new(x, y);
            let inset = if x < 900.0 { 46.0 } else { 0.0 };
            for k in 0..TRIALS.len() {
                let mut s = ChangeOfBasis::create();
                (s.canvas, s.inset) = (view, inset);
                s.set_up(k);
                let mut size = None;
                while s.tween.is_some() {
                    s.advance(1.0 / 60.0);
                    let now = s.panel_size(&s.lines_at(s.tips));
                    assert!(size.is_none_or(|was| was == now), "{x}x{y}, experiment {k}: the panel went from {size:?} to {now}");
                    size = Some(now);
                }
                let lay = Layout::of(view, inset, s.panel_size(&s.lines_at(s.tips)), 3);
                if let Some((lo, hi)) = lay.panel {
                    assert!(lo.x >= 0.0 && lo.y >= inset && hi.x <= x + 1e-3 && hi.y <= y + 1e-3, "{x}x{y}: {lo} {hi}");
                }
                if x >= 1000.0 && y >= 700.0 || narrow(view) && x >= 360.0 {
                    assert!(lay.panel.is_some(), "{x}x{y}: no room for the panel");
                }
                let ((lo, hi), scale) = lay.region(Vec2::new(4.0, 3.0));
                assert!(hi.x > lo.x && hi.y > lo.y && scale > 0.0 && scale.is_finite(), "{x}x{y}");
                assert!(matches!(s.the_view(), View::Fit { center, half } if center.is_finite() && half.is_finite() && half.min_element() > 0.0), "{x}x{y}");
            }
        }
    }
}

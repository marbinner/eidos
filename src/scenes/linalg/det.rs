//! Determinants: the signed area of the parallelogram on a matrix's columns, the signed
//! volume of the box on them, and where their rules and their formula come from.

use super::*;
use glam::DVec3;

pub const DEF: SceneDef = SceneDef {
    id: "det",
    title: "Determinants",
    group: "Linear algebra: vectors and maps",
    blurb: "Signed area and volume: why sliding a side changes nothing, and swapping two sides flips the sign.",
    make: |_| Box::new(Det::create()),
};

/// Things to try: (button, what to look for). What each one sets is in `Det::set_up`.
const TRIALS: [(&str, &str); 5] = [
    (
        "slide a side",
        r"$a_2$'s tip runs along the dashed line parallel to $a_1$, from leaning left to leaning far over to the right. The base $a_1$ and the height over it (dashed white, at a right angle) stay as they are, and so does the area: 3.2 the whole way. Adding a multiple of one column to another never changes the determinant. Drag the white dot to slide it yourself.",
    ),
    (
        "swap the columns",
        r"$a_1$ and $a_2$ trade places. Halfway they lie on top of each other and the area is 0. After, the parallelogram is the same shape, but the turn from $a_1$ to $a_2$ goes clockwise and the area is −2.7 instead of 2.7: swapping two columns flips the sign.",
    ),
    (
        "slide it square",
        r"$a_2$ slides along $a_1$ until it stands upright, then $a_1$ slides along $a_2$ until it lies flat, each slide keeping the area at 2.5. This is elimination done on the columns instead of the rows: watch the matrix over the picture. What is left is a rectangle along the axes, 2 by 1.25. Its sides are the pivots elimination finds, and $2 \times 1.25 = 2.5 = ad - bc$.",
    ),
    (
        "columns in one line",
        r"$a_2 = 2.5 a_1$: the parallelogram is squashed into a line and its area is 0. Sliding it square, $a_2$ slides all the way down to $0$, and there is no second pivot. The determinant is 0 exactly when the columns lie in one line (in space, in one plane): when they are dependent.",
    ),
    (
        r"in space: $\text{base} \times \text{height}$",
        r#"Three columns span a slanted box. Its volume is the area of its base (the parallelogram on $a_1$ and $a_2$, here on the floor) times the height of $a_3$ over it (dashed white, at a right angle): $2.5 \times 1.6 = 4$. The white arrow $a_1 \times a_2$ stands at a right angle to the base and is as long as its area, so $(a_1 \times a_2) \cdot a_3$ is that same product. Press "slide it square" over the picture: the same slides make a brick, 2 by 1.25 by 1.6."#,
    ),
];

/// The colours of the columns a₁, a₂, a₃.
const COLS: [Color; 3] = [pal::PINK, pal::GREEN, pal::ORANGE];
/// The colour of an area or a volume counted positive, and of one counted negative.
const PLUS: Color = pal::YELLOW;
const MINUS: Color = pal::BLUE;
/// The names of the columns, of the pivots and of the axes, as markup (to go
/// between dollars, alone or in a formula).
const NAMES: [&str; 3] = ["a_1", "a_2", "a_3"];
const PIVOTS: [&str; 3] = ["p_1", "p_2", "p_3"];
const AXES: [&str; 3] = ["x", "y", "z"];

/// How far from 0 a tip can be dragged, in the plane and in space, and the step its
/// numbers move in: tenths, so that the sums over the picture can be checked by hand.
const REACH: [f64; 2] = [4.0, 3.0];
const STEP: f64 = 0.1;
/// Whole tenths: `v` to the nearest (as the double nearest it: 12 / 10, not 12 × 0.1).
fn tenth(v: f64) -> f64 {
    (v / STEP).round() / 10.0
}
/// A number no larger than this share of the largest entry counts as 0 when a pivot
/// is looked for (and a determinant no larger than it times the largest entry to the
/// nth power). The entries are tenths, or hundredths once slid or typed, up to 10: a
/// pivot that is not 0 is then at least 5 × 10⁻¹⁰ of the largest entry (a minor of
/// the cube of a hundredth over one of 200), while what rounding leaves of a pivot
/// that is 0 is near 10⁻¹⁶ of it.
const ZERO: f64 = 1e-12;

/// Seconds a step takes forwards and backwards, and the pause after each while the
/// steps are played through.
const STEP_TIME: f32 = 1.6;
const BACK_TIME: f32 = 0.5;
const HOLD: f32 = 0.8;
/// How quickly an arrow catches up with a tip that has stepped: the time in which it
/// covers two thirds of the way (seconds).
const GLIDE: f32 = 0.08;
/// The same for the arrows' way into an experiment, from wherever they were.
const SETTLE: f32 = 0.14;
const ARRIVED: f32 = 1e-4;

/// Which columns the picture measures against: the one that moves, the base it
/// moves over (a column in the plane, given twice; two in space), and the column it
/// slides along, if it slides.
type Roles = (usize, [usize; 2], Option<usize>);

/// The share of a step over which the marks of the step before give way to its own.
const FADE: f32 = 0.18;

/// Are the marks drawn now named? Only while one set of them is drawn in full: names
/// that went at the moment others came would be taken for those others, moved.
fn named(marks: &[(Roles, f32)]) -> bool {
    matches!(marks, [(_, alpha)] if *alpha >= 1.0)
}

/// One thing done to the columns.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    /// aⱼ + m aᵢ: column j slides along column i. A slide of "slide it square" makes
    /// the entry in row `clears` of column j 0, and leaves it exactly 0.
    Slide { j: usize, i: usize, m: f64, clears: Option<usize> },
    /// Columns i and j trade places.
    Swap { i: usize, j: usize },
    /// Column j times k.
    Stretch { j: usize, k: f64 },
}

impl Op {
    /// The columns a share `t` (0 to 1) of the way through.
    fn at(self, c: [DVec3; 3], t: f64) -> [DVec3; 3] {
        let mut out = c;
        match self {
            Op::Slide { j, i, m, clears } => {
                out[j] = c[j] + c[i] * (m * t);
                if let (Some(r), true) = (clears, t >= 1.0) {
                    out[j][r] = 0.0;
                }
            }
            // (along the straight line between them: the determinant on the way is
            // (1 − 2t) times what it was, 0 halfway)
            Op::Swap { i, j } => {
                if t >= 1.0 {
                    (out[i], out[j]) = (c[j], c[i]);
                } else {
                    (out[i], out[j]) = (c[i] + (c[j] - c[i]) * t, c[j] + (c[i] - c[j]) * t);
                }
            }
            Op::Stretch { j, k } => out[j] = c[j] * (1.0 + (k - 1.0) * t),
        }
        out
    }
}

/// How "slide it square" ends.
#[derive(Clone, Copy, Debug, PartialEq)]
enum End {
    /// The columns lie along the axes: the numbers on the diagonal are the pivots.
    Brick { pivots: [f64; 3], swaps: usize },
    /// No pivot in row `row`: from column `row` on, none has a part along that axis
    /// left, and the determinant is 0.
    Flat { row: usize, swaps: usize },
}

/// The determinant: the signed volume of the box on the three columns. (In the plane
/// the third column is e₃, and this is the plane's ad − bc.)
fn det(c: &[DVec3; 3]) -> f64 {
    c[0].cross(c[1]).dot(c[2])
}

/// The largest entry of the first `n` columns, in size.
fn largest(c: &[DVec3; 3], n: usize) -> f64 {
    c[..n].iter().map(|v| v.abs().max_element()).fold(0.0, f64::max)
}

/// Is the determinant 0, as far as the numbers can tell?
fn is_flat(c: &[DVec3; 3], n: usize) -> bool {
    let size = largest(c, n);
    // (a determinant that is not a number counts as none)
    let d = det(c).abs();
    !d.is_finite() || d <= ZERO * size.powi(n as i32)
}

fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The slides (and swaps) that turn the columns into a brick along the axes: elimination
/// done on the columns. Row by row: if the column on the diagonal has nothing in this
/// row, a later one that has is swapped in; then each later column slides along it
/// until its entry in this row is 0. That leaves the columns lower triangular, with the
/// pivots on the diagonal; sliding them back up, from the last, clears what is below
/// the diagonal and leaves the pivots alone.
fn square_ops(start: [DVec3; 3], n: usize) -> (Vec<Op>, End) {
    let mut c = start;
    let size = largest(&c, n);
    let zero = |v: f64| !v.is_finite() || v.abs() <= ZERO * size;
    let mut ops = Vec::new();
    let mut swaps = 0;
    let mut push = |op: Op, c: &mut [DVec3; 3]| {
        *c = op.at(*c, 1.0);
        ops.push(op);
    };
    for r in 0..n {
        if zero(c[r][r]) {
            match (r + 1..n).find(|&k| !zero(c[k][r])) {
                Some(k) => {
                    push(Op::Swap { i: r, j: k }, &mut c);
                    swaps += 1;
                }
                None => return (ops, End::Flat { row: r, swaps }),
            }
        }
        for k in r + 1..n {
            if !zero(c[k][r]) {
                push(Op::Slide { j: k, i: r, m: -c[k][r] / c[r][r], clears: Some(r) }, &mut c);
            }
        }
    }
    for r in (1..n).rev() {
        for k in (0..r).rev() {
            if !zero(c[k][r]) {
                push(Op::Slide { j: k, i: r, m: -c[k][r] / c[r][r], clears: Some(r) }, &mut c);
            }
        }
    }
    let pivots = [0, 1, 2].map(|r| if r < n { c[r][r] } else { 1.0 });
    (ops, End::Brick { pivots, swaps })
}

// ----- numbers as they are written ---------------------------------------------------

/// `v` with the fewest decimals, up to `most`, that write it exactly (a true minus
/// sign, no "−0"); None if `most` are not enough.
fn exact(v: f64, most: usize) -> Option<String> {
    if !v.is_finite() {
        return None;
    }
    (0..=most).find(|&d| {
        let scaled = v * 10f64.powi(d as i32);
        (scaled - scaled.round()).abs() <= 1e-6
    })
    .map(|d| num(v, d))
}

/// `v` exactly, if `most` decimals can, or else cut to `most` with "…" after it.
fn written(v: f64, most: usize) -> String {
    exact(v, most).unwrap_or_else(|| format!("{}…", num(v, most)))
}

/// `v` to two decimals at most, without trailing zeros: an entry of a matrix on
/// the screen, or a number in a sum that says "≈".
fn short(v: f64) -> String {
    exact(v, 2).unwrap_or_else(|| exact((v * 100.0).round() / 100.0, 2).unwrap_or_else(|| num(v, 2)))
}

/// An area or a volume written on the picture: exactly, to three decimals at most, or
/// else to two (on its way from one step to the next).
fn measured(v: f64) -> String {
    exact(v, 3).unwrap_or_else(|| short(v))
}

/// A number after an operation sign: in brackets if it is negative.
fn bracketed(w: String) -> String {
    if w.starts_with('−') {
        format!("({w})")
    } else {
        w
    }
}

fn factor(v: f64, most: usize) -> String {
    bracketed(written(v, most))
}

/// "a₂ − 0.5 a₁", "a₂ + a₁", "swap a₁ and a₂", "a₂ × 2" (as markup, dollars and all).
fn op_name(op: Op) -> String {
    match op {
        Op::Slide { j, i, m, .. } => {
            let size = written(m.abs(), 3);
            let times = if size == "1" { String::new() } else { format!("{size} ") };
            format!("${} {} {times}{}$", NAMES[j], if m < 0.0 { "-" } else { "+" }, NAMES[i])
        }
        Op::Swap { i, j } => format!("swap ${}$ and ${}$", NAMES[i], NAMES[j]),
        Op::Stretch { j, k } => format!(r"${} \times {}$", NAMES[j], factor(k, 3)),
    }
}

/// What a step does, in a sentence or two: `before` is the determinant as it starts,
/// and `square` says whether it is a step of "slide it square".
fn op_says(op: Op, n: usize, before: f64, square: bool) -> String {
    let word = if n == 2 { "area" } else { "volume" };
    let (was, flipped) = (written(before, 3), written(-before, 3));
    match op {
        Op::Slide { j, i, clears, .. } => {
            let until = match clears {
                Some(r) if square => format!(" until it has no ${}$ part", AXES[r]),
                _ => String::new(),
            };
            let base = if n == 2 { format!("the base ${}$", NAMES[i]) } else { "the base".to_string() };
            format!("${}$ slides along ${}${until}. Its tip keeps its height over {base}, so the {word} stays {was}.", NAMES[j], NAMES[i])
        }
        Op::Swap { i, j } if square => {
            format!("${}$ has no ${}$ part to give a pivot, so it trades places with ${}$. A swap flips the sign: {was} becomes {flipped}.", NAMES[i], AXES[i], NAMES[j])
        }
        Op::Swap { i, j } => {
            let turn = if n == 2 { "the turn from $a_1$ to $a_2$ goes the other way" } else { "the columns turn the other way round" };
            format!("${}$ and ${}$ trade places. Halfway the {word} is 0; after, {turn}, and {was} has become {flipped}.", NAMES[i], NAMES[j])
        }
        Op::Stretch { j, k } => {
            let after = written(k * before, 3);
            if k < 0.0 {
                format!("${}$ turns round to point the other way: the {word} passes through 0 and {was} becomes {after}, {} times as much.", NAMES[j], factor(k, 3))
            } else {
                format!("${}$ is stretched {} times, and so is its height: the {word} goes from {was} to {after}.", NAMES[j], written(k, 3))
            }
        }
    }
}

/// A worked sum: its parts in order, exactly if every number in it can be written
/// exactly (to four decimals), each joined by "="; otherwise every number to two
/// decimals, joined by "≈". `parts` are (text before the number, the number, whether
/// it goes in brackets when negative); the texts are markup, and put the dollars
/// round the formulas themselves, as `tail` closes the last.
fn worked<S: AsRef<str>>(parts: &[(S, f64, bool)], tail: &str) -> String {
    let exactly = parts.iter().all(|(_, v, _)| exact(*v, 4).is_some());
    let mut out = String::new();
    for (before, v, brackets) in parts {
        out.push_str(&before.as_ref().replace('=', if exactly { "=" } else { r"\approx" }));
        let w = if exactly { written(*v, 4) } else { short(*v) };
        out.push_str(&if *brackets { bracketed(w) } else { w });
    }
    out + tail
}

/// ad − bc worked out with the numbers of the columns (the plane).
fn plane_sum(c: &[DVec3; 3]) -> String {
    let (a, b, cc, d) = (c[0].x, c[1].x, c[0].y, c[1].y);
    let times = r" \times ";
    worked(&[("$ad - bc = ", a, false), (times, d, true), (" - ", b, true), (times, cc, true), (" = ", a * d, false), (" - ", b * cc, true), (" = ", a * d - b * cc, false)], "$")
}

/// (a₁ × a₂) · a₃ worked out with the numbers of the columns (space): the cross
/// product, then its dot product with a₃.
fn space_sum(c: &[DVec3; 3]) -> [String; 2] {
    let (x, z) = (c[0].cross(c[1]), c[2]);
    let times = r" \times ";
    [
        worked(&[(r"$a_1 \times a_2 = (", x.x, false), (", ", x.y, false), (", ", x.z, false)], ")$"),
        worked(&[(r"$(a_1 \times a_2) \cdot a_3 = ", x.x, false), (times, z.x, true), (" + ", x.y, true), (times, z.y, true), (" + ", x.z, true), (times, z.z, true), (" = ", det(c), false)], "$"),
    ]
}

/// What the end of "slide it square" comes to: the pivots, their product, and why
/// that is the determinant, in a line or two.
fn end_says(end: End, start: &[DVec3; 3], n: usize) -> Vec<String> {
    match end {
        End::Brick { pivots, swaps } => {
            let p = &pivots[..n];
            let product: f64 = p.iter().product();
            let sign = if swaps % 2 == 0 { 1.0 } else { -1.0 };
            let (a, b, c, d) = (start[0].x, start[1].x, start[0].y, start[1].y);
            if n == 2 && swaps == 0 {
                // the pivots are a and d − bc/a, whose product is ad − bc
                let bca = b * c / a;
                return vec![
                    worked(&[("pivots: $a = ", a, false), ("$ and $d - bc/a = ", d, false), (" - ", bca, true), (" = ", d - bca, false)], "$"),
                    worked(&[("$", a, false), (r" \times ", d - bca, true), (" = ", product, false)], r"$, and $a(d - bc/a)$ is $ad - bc$"),
                ];
            }
            let opening = match swaps % 2 {
                0 => "pivots: $".to_string(),
                _ => format!("pivots, and {}: $-(", if swaps == 1 { "one swap" } else { "three swaps" }),
            };
            let mut parts: Vec<(String, f64, bool)> = p.iter().enumerate().map(|(k, v)| (if k == 0 { opening.clone() } else { r" \times ".to_string() }, *v, k > 0)).collect();
            parts.push((if sign > 0.0 { " = " } else { ") = " }.to_string(), sign * product, false));
            let mut lines = vec![worked(&parts, "$")];
            if n == 2 && a == 0.0 {
                lines.push(worked(&[("with $a = 0$, $ad - bc$ is just $-bc = -", b, true), (r" \times ", c, true)], "$"));
            }
            lines
        }
        End::Flat { row, .. } => vec![format!(r"no pivot in row {}: $\det A = 0$", row + 1)],
    }
}

/// Why "slide it square" found no pivot in `row`.
fn flat_says(n: usize, row: usize) -> String {
    match (n, row) {
        (2, 0) => "$a_1$ and $a_2$ both lie on the $y$ axis, in one line: no pivot in the top row, and no area.".into(),
        (2, _) => "$a_2$ is now $0$: it lay on the line of $a_1$. No second pivot, and no area.".into(),
        (_, 0) => "$a_1$, $a_2$ and $a_3$ all lie in the wall $x = 0$, one plane: no pivot in the top row, and no volume.".into(),
        (_, 1) => "$a_2$ and $a_3$ now both lie on the $z$ axis, in one line, so all three lay in one plane: no second pivot, and no volume.".into(),
        _ => "$a_3$ is now $0$: it lay in the plane of $a_1$ and $a_2$. No third pivot, and no volume.".into(),
    }
}

// ----- the panel over the picture ----------------------------------------------------

/// Where the panel with the steps and the matrix goes on a canvas of `view` logical
/// pixels, `inset` of whose top lies under the application's buttons: over the
/// picture's top-left corner, or below the picture on a canvas narrow and tall enough
/// (a phone held upright). Left out where the canvas cannot spare the room.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Layout {
    narrow: bool,
    panel: Option<(Vec2, Vec2)>,
    /// Where the picture's part of the canvas ends at the bottom.
    below: f32,
}

const PANEL_W: [f32; 2] = [350.0, 410.0];
/// Lines of words the panel keeps room for under its heading, and under the matrices.
const SAYS_LINES: usize = 3;
const SUM_LINES: [usize; 2] = [3, 4];
const SAYS_SIZE: f32 = 11.5;
const SUM_SIZE: f32 = 12.0;

/// The panel's height with `rows` rows in its matrices.
fn panel_height(rows: usize) -> f32 {
    10.0 + 19.0 + SAYS_LINES as f32 * 15.0 + 6.0 + 26.0 + 12.0 + 15.0 + rows as f32 * 22.0 + 10.0 + SUM_LINES[(rows == 3) as usize] as f32 * 16.0 + 6.0
}

impl Layout {
    fn of(view: Vec2, inset: f32, rows: usize) -> Layout {
        let h = panel_height(rows);
        if view.x < 620.0 && view.y >= 560.0 {
            let top = view.y - h - 10.0;
            let panel = (view.x >= 300.0).then(|| (Vec2::new(8.0, top), Vec2::new(view.x - 8.0, view.y - 10.0)));
            Layout { narrow: true, panel, below: if panel.is_some() { top - 4.0 } else { view.y } }
        } else {
            let w = PANEL_W[(rows == 3) as usize];
            let top = 14.0f32.max(inset + 8.0);
            let fits = view.x >= w + 220.0 && view.y >= top + h + 60.0;
            Layout { narrow: false, panel: fits.then(|| (Vec2::new(14.0, top), Vec2::new(14.0 + w, top + h))), below: view.y }
        }
    }

    fn covers(&self, p: Vec2) -> bool {
        self.panel.is_some_and(|(min, max)| p.cmpge(min).all() && p.cmple(max).all())
    }
}

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

/// The words of a text, each formula whole (with what is glued to it, as in "$a_2$'s").
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

/// A word too wide for a line, if it is one formula (with nothing but punctuation
/// glued to it), cut where a long sum is broken in print: before each relation (= or
/// ≈) outside any braces. Each piece after the first is marked as carrying on the
/// formula, to be set in one with the piece before it when they share a line.
fn cut(word: &str, width: f32, size: f32) -> Vec<(String, bool)> {
    let whole = vec![(word.to_string(), false)];
    let (Some(open), Some(close)) = (word.find('$'), word.rfind('$')) else { return whole };
    if open != 0 || close == 0 || word[1..close].contains('$') || width_of(word, size) <= width {
        return whole;
    }
    let (body, tail) = (&word[1..close], &word[close + 1..]);
    let mut cuts = Vec::new();
    let (mut depth, mut i) = (0i32, 0);
    while let Some(c) = body[i..].chars().next() {
        let rest = &body[i..];
        match c {
            // (a command's backslash, or an escaped brace's: no relation starts there)
            '\\' => i += rest.chars().nth(1).map_or(0, char::len_utf8),
            '{' => depth += 1,
            '}' => depth -= 1,
            ' ' if depth == 0 && i > 0 && (rest.starts_with(" = ") || rest.starts_with(r" \approx ")) => cuts.push(i + 1),
            _ => {}
        }
        i += c.len_utf8();
    }
    let mut pieces = Vec::new();
    let mut from = 0;
    for at in cuts.into_iter().chain([body.len()]) {
        let piece = body[from..at].trim();
        if !piece.is_empty() {
            pieces.push((format!("${piece}$"), from > 0));
        }
        from = at;
    }
    if let Some(last) = pieces.last_mut() {
        last.0.push_str(tail);
    }
    pieces
}

/// Text broken into lines no wider than `width` logical pixels: between words, each
/// formula whole, unless it is too wide for a line by itself (see `cut`).
fn wrap(text: &str, width: f32, size: f32) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for word in words_of(text) {
        for (piece, carries_on) in cut(word, width, size) {
            if let Some(last) = lines.last_mut() {
                // (a piece of the formula before is set in one with it)
                let joined = match last.strip_suffix('$') {
                    Some(start) if carries_on => format!("{start} {}", &piece[1..]),
                    _ => format!("{last} {piece}"),
                };
                if width_of(&joined, size) <= width {
                    *last = joined;
                    continue;
                }
            }
            lines.push(piece);
        }
    }
    lines
}

/// The sums as written in a panel `width` pixels across: each broken into lines, the
/// lines after the first of each set in.
fn sum_lines(lines: &[String], width: f32) -> Vec<String> {
    let mut out = Vec::new();
    for line in lines {
        let mut parts = wrap(line, width - 16.0, SUM_SIZE);
        for part in parts.iter_mut().skip(1) {
            *part = format!("    {part}");
        }
        out.extend(parts);
    }
    out
}

/// Lines broken by `breaking` to `width`, as they were in the frame before if what they
/// are broken from is the same (breaking them measures every word with all before it).
fn kept_lines(kept: &mut Broken, from: Vec<String>, width: f32, breaking: impl Fn(&[String], f32) -> Vec<String>) -> Vec<String> {
    if kept.0 != from || kept.1 != width {
        let lines = breaking(&from, width);
        *kept = (from, width, lines);
    }
    kept.2.clone()
}

/// Lines of words as they were broken: (what from, how wide, the lines).
type Broken = (Vec<String>, f32, Vec<String>);

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

/// A caption along the bottom of the picture, on a dark plate, like `bottom_caption`
/// in the toolkit but ending at `bottom` (above the panel, on a phone).
fn caption(f: &mut Frame, parts: &[&str], bottom: f32) {
    let (view, size) = (f.view_size(), 12.0);
    if view.y < 300.0 {
        return;
    }
    let room = view.x - 36.0;
    let gap = "      ";
    let width = |text: &str| width_of(text, size);
    let mut lines: Vec<String> = Vec::new();
    for part in parts {
        match lines.last_mut() {
            Some(last) if width(last) + width(gap) + width(part) <= room => {
                last.push_str(gap);
                last.push_str(part);
            }
            _ => lines.push(part.to_string()),
        }
    }
    let count = lines.len();
    for (k, line) in lines.into_iter().enumerate() {
        let up = (count - 1 - k) as f32 * (size + 7.0);
        f.caption_px(Vec2::new(18.0, bottom - up), line, size, pal::DIM, Align2::LEFT_BOTTOM);
    }
}

/// The direction space is seen from at first: in front, to the right and above.
const YAW: f32 = -0.95;
const PITCH: f32 = 0.42;
/// (The tangent of half the camera's field of view.)
const TAN_HALF_FOV: f32 = 0.3203;

/// The camera's right and up directions in the view space opens with.
fn screen_axes() -> (Vec3, Vec3) {
    let right = Vec3::new(-YAW.sin(), YAW.cos(), 0.0);
    let up = Vec3::new(-PITCH.sin() * YAW.cos(), -PITCH.sin() * YAW.sin(), PITCH.cos());
    (right, up)
}

/// The closest a flat picture comes, in logical pixels per unit, and the room kept
/// round it for the names written beside things.
const MOST_PER_UNIT: f32 = 170.0;
const NAME_ROOM: Vec2 = Vec2::new(90.0, 70.0);
/// (Less on a narrow canvas, where every pixel counts.)
const NAME_ROOM_NARROW: Vec2 = Vec2::new(50.0, 50.0);

/// The columns the scene opens with: in the plane, and in space.
const START_PLANE: [[f64; 2]; 2] = [[2.0, 0.7], [1.0, 1.6]];
/// (In space the base lies on the floor, the parallelogram the plane opens with, and
/// a₃ leans over it: base and height can be read off at once.)
const START_SPACE: [[f64; 3]; 3] = [[2.0, 0.7, 0.0], [1.0, 1.6, 0.0], [0.6, 0.5, 1.6]];

fn plane_cols(c: [[f64; 2]; 2]) -> [DVec3; 3] {
    [DVec3::new(c[0][0], c[0][1], 0.0), DVec3::new(c[1][0], c[1][1], 0.0), DVec3::Z]
}

fn space_cols(c: [[f64; 3]; 3]) -> [DVec3; 3] {
    c.map(DVec3::from)
}

/// Where a dragged tip goes when its handle is at `p`: no further out than `reach`,
/// and then to the nearest tenths (in the plane, z stays 0).
fn snap(p: Vec3, reach: f64, flat: bool) -> DVec3 {
    let mut p = p.as_dvec3();
    if flat {
        p.z = 0.0;
    }
    if !p.is_finite() {
        return DVec3::ZERO;
    }
    if p.length() > reach {
        p *= reach / p.length();
    }
    p.map(tenth)
}

pub struct Det {
    /// Three columns in space (true), or two in the plane.
    space: bool,
    /// The columns the steps start from: those of the plane (z = 0, with e₃ as a third
    /// that never moves) and those of space.
    start: [[DVec3; 3]; 2],
    /// What is done to them, step by step, and how "slide it square" ends if that is
    /// what the steps are.
    ops: Vec<Op>,
    end: Option<End>,
    /// How far through the steps the picture is (0 to the number of steps), and where
    /// it is going.
    stage: f32,
    goal: f32,
    /// Playing through with a pause after each step, and what is left of the pause.
    through: bool,
    hold: f32,
    /// Where the arrows are drawn: gliding to the columns when a tip has stepped, and
    /// how quickly (`GLIDE`, or `SETTLE` on the way into an experiment).
    drawn: [Vec3; 3],
    pace: f32,
    /// Where each handle is while it is dragged (the three tips, and the white corner).
    grip: [Vec3; 4],
    /// While the white corner is dragged: the column it slides, as it was when the drag began.
    slide_from: Option<DVec3>,
    /// What the labels placed by hand keep out of, this frame.
    keep: Keep,
    /// The canvas last drawn on, and how much of its top the application's buttons
    /// cover: they decide where the panel goes, and with it where the picture goes.
    canvas: Vec2,
    inset: f32,
    measured: bool,
    new_view: bool,
    /// The panel's words and sums as they were broken into lines in the frame before.
    broken: [Broken; 2],
}

impl Det {
    /// (Nothing here needs the graphics device, so the tests can make one too.)
    fn create() -> Self {
        let mut s = Det {
            space: false,
            start: [plane_cols(START_PLANE), space_cols(START_SPACE)],
            ops: Vec::new(),
            end: None,
            stage: 0.0,
            goal: 0.0,
            through: false,
            hold: 0.0,
            drawn: [Vec3::ZERO; 3],
            pace: GLIDE,
            grip: [Vec3::ZERO; 4],
            slide_from: None,
            keep: (None, 0.0, Vec2::ONE),
            canvas: Vec2::new(1080.0, 1000.0),
            inset: 0.0,
            measured: false,
            new_view: false,
            broken: Default::default(),
        };
        s.arrive();
        s
    }

    /// Set up one of the experiments in `TRIALS`.
    fn set_up(&mut self, k: usize) {
        self.space = k == 4;
        let (cols, ops, square) = match k {
            0 => ([[2.0, 0.5], [-0.8, 1.4]], vec![Op::Slide { j: 1, i: 0, m: 1.5, clears: None }], false),
            1 => ([[2.0, 0.5], [0.6, 1.5]], vec![Op::Swap { i: 0, j: 1 }], false),
            2 => ([[2.0, 0.7], [1.0, 1.6]], Vec::new(), true),
            3 => ([[1.0, 0.4], [2.5, 1.0]], Vec::new(), true),
            _ => (START_PLANE, Vec::new(), false),
        };
        if self.space {
            self.start[1] = space_cols(START_SPACE);
        } else {
            self.start[0] = plane_cols(cols);
        }
        (self.ops, self.end) = (Vec::new(), None);
        (self.stage, self.goal, self.through, self.hold) = (0.0, 0.0, false, 0.0);
        if square {
            self.slide_square();
        } else if !ops.is_empty() {
            self.play(ops, None);
        }
        // (the arrows glide there from wherever they were: a picture that jumped
        // would not show what has changed)
        self.pace = SETTLE;
        self.new_view = true;
    }

    fn n(&self) -> usize {
        if self.space {
            3
        } else {
            2
        }
    }

    /// The columns the steps start from.
    fn cols(&self) -> [DVec3; 3] {
        self.start[self.space as usize]
    }

    /// The columns after the first `k` steps.
    fn after(&self, k: usize) -> [DVec3; 3] {
        self.ops.iter().take(k).fold(self.cols(), |c, op| op.at(c, 1.0))
    }

    /// The columns as they are now, part of the way through a step.
    fn shown(&self) -> [DVec3; 3] {
        let mut c = self.cols();
        for (k, op) in self.ops.iter().enumerate() {
            let t = (self.stage as f64 - k as f64).clamp(0.0, 1.0);
            if t <= 0.0 {
                break;
            }
            c = op.at(c, smooth(t));
        }
        c
    }

    /// Start the steps `ops` from the picture as it is now.
    fn play(&mut self, ops: Vec<Op>, end: Option<End>) {
        let now = self.shown();
        self.start[self.space as usize] = now;
        (self.ops, self.end) = (ops, end);
        (self.stage, self.goal, self.through, self.hold) = (0.0, self.ops.len() as f32, true, HOLD * 0.5);
        self.new_view = true;
    }

    /// "Slide it square", from the picture as it is now.
    fn slide_square(&mut self) {
        let (ops, end) = square_ops(self.shown(), self.n());
        self.play(ops, Some(end));
    }

    /// Leave the steps where they are and start afresh from the picture as it is (a
    /// tip is being dragged, or a number typed).
    fn commit(&mut self) {
        if !self.ops.is_empty() {
            let now = self.shown();
            self.start[self.space as usize] = now;
            (self.ops, self.end) = (Vec::new(), None);
            (self.stage, self.goal, self.through, self.hold) = (0.0, 0.0, false, 0.0);
        }
    }

    fn set_cols(&mut self, c: [DVec3; 3]) {
        self.commit();
        self.start[self.space as usize] = c;
    }

    // ----- stepping (as in the Gram–Schmidt scene) -----

    /// One step on: after the steps already asked for, or, while they are being played
    /// through, after the one playing (and there it stops).
    fn next(&mut self) {
        let from = if self.through { self.stage } else { self.goal.max(self.stage) };
        self.goal = (from.floor() + 1.0).min(self.ops.len() as f32);
        (self.through, self.hold) = (false, 0.0);
    }

    fn can_go_on(&self) -> bool {
        (if self.through { self.stage } else { self.goal.max(self.stage) }) < self.ops.len() as f32
    }

    fn back(&mut self) {
        self.goal = (self.goal.min(self.stage).ceil() - 1.0).max(0.0);
        (self.through, self.hold) = (false, 0.0);
    }

    fn replay(&mut self) {
        (self.stage, self.goal, self.through, self.hold) = (0.0, self.ops.len() as f32, true, HOLD * 0.6);
    }

    fn moving(&self) -> bool {
        self.stage != self.goal || self.hold > 0.0
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

    /// The step being shown: the one playing, or the last one finished (0 before the first).
    fn step(&self) -> usize {
        (self.stage.max(0.0).ceil() as usize).min(self.ops.len())
    }

    /// Have the steps come to their end?
    /// Has "slide it square" come to its end? (Which may be at once: columns that lie
    /// along the axes already, or that have no pivot in the top row, need no step.)
    fn ended(&self) -> bool {
        self.end.is_some() && self.stage >= self.ops.len() as f32
    }

    // ----- the arrows as drawn -----

    /// Are the arrows drawn where the columns are?
    fn arrived(&self) -> bool {
        let c = self.shown();
        (0..3).all(|k| self.drawn[k] == c[k].as_vec3())
    }

    /// The arrows drawn where the columns are, at once.
    fn arrive(&mut self) {
        self.glide(0.0, true);
        for k in 0..3 {
            self.grip[k] = self.drawn[k];
        }
        self.grip[3] = self.corner();
    }

    /// Bring the arrows as drawn `dt` seconds nearer to the columns; with `at_once`,
    /// all the way.
    fn glide(&mut self, dt: f32, at_once: bool) {
        let k = 1.0 - (-dt.max(0.0) / self.pace).exp();
        let c = self.shown();
        for (drawn, to) in self.drawn.iter_mut().zip(c.map(|v| v.as_vec3())) {
            *drawn += (to - *drawn) * k;
            // (exactly there in the end, not nearly: the picture then stops changing)
            if at_once || !drawn.is_finite() || (to - *drawn).length() < ARRIVED {
                *drawn = to;
            }
        }
        if self.arrived() {
            self.pace = GLIDE;
        }
    }

    /// The far corner of the parallelogram (or the box), as drawn: the white handle.
    fn corner(&self) -> Vec3 {
        self.drawn[..self.n()].iter().copied().sum()
    }

    /// Which columns the picture measures against at this step: the one that moves,
    /// the base it moves over (a column in the plane, two in space), and the column
    /// it slides along, if it slides. None while two columns trade places, and at the
    /// end of "slide it square", where the sides are named instead.
    fn roles(&self) -> Option<Roles> {
        if self.ended() {
            None
        } else if self.stage > 0.0 && !self.ops.is_empty() {
            self.roles_at(self.step().max(1))
        } else {
            self.roles_at(0)
        }
    }

    /// The same during step `k` (counted from 1; 0 is before the first).
    fn roles_at(&self, k: usize) -> Option<Roles> {
        let n = self.n();
        let other = |j: usize, i: usize| 3 - i - j;
        let base = |j: usize, i: usize| if n == 2 { [i, i] } else { [i, other(j, i)] };
        match k.checked_sub(1).and_then(|k| self.ops.get(k)) {
            Some(Op::Swap { .. }) => None,
            Some(&Op::Slide { j, i, .. }) => Some((j, base(j, i), Some(i))),
            Some(&Op::Stretch { j, .. }) => {
                let i = if j == 0 { 1 } else { 0 };
                Some((j, base(j, i), None))
            }
            None => Some(if n == 2 { (1, [0, 0], Some(0)) } else { (2, [0, 1], None) }),
        }
    }

    /// What is measured against now, and how strongly it is drawn: at the start of a
    /// step the marks of the step before fade out while its own fade in, and at the
    /// end of "slide it square" they fade out (marks that changed places at once would
    /// jump).
    fn marks(&self) -> Vec<(Roles, f32)> {
        let now = self.roles();
        if self.stage > 0.0 && !self.ops.is_empty() && !self.ended() {
            let k = self.step().max(1);
            let into = self.stage - (k - 1) as f32;
            if into < FADE {
                let before = self.roles_at(k - 1);
                if before != now {
                    let f = into / FADE;
                    return before.map(|r| (r, 1.0 - f)).into_iter().chain(now.map(|r| (r, f))).collect();
                }
            }
            if self.end.is_some() && k == self.ops.len() && into > 1.0 - FADE {
                return now.map(|r| vec![(r, (1.0 - into) / FADE)]).unwrap_or_default();
            }
        }
        now.map(|r| vec![(r, 1.0)]).unwrap_or_default()
    }

    // ----- the view -----

    /// The columns at the moments the picture passes through: the start, the end of
    /// each step, and points along each (a swap swings wide, a stretch reaches far).
    fn moments(&self) -> Vec<[DVec3; 3]> {
        let mut c = self.cols();
        let mut out = vec![c];
        for op in &self.ops {
            for t in [0.25, 0.5, 0.75, 1.0] {
                out.push(op.at(c, t));
            }
            c = op.at(c, 1.0);
        }
        out
    }

    /// The part of the canvas the picture has to itself (logical pixels), for a
    /// picture of `size` world units, and the scale it can be shown at there: above
    /// the panel on a phone; otherwise beside it or below it, whichever is larger.
    fn region(&self, size: Vec2) -> ((Vec2, Vec2), f32) {
        let lay = Layout::of(self.canvas, self.inset, self.n());
        let bottom = if self.canvas.y >= 300.0 { lay.below - 34.0 } else { lay.below };
        let top = self.inset + 4.0;
        let margin = if lay.narrow { NAME_ROOM_NARROW } else { NAME_ROOM };
        let scale = |(lo, hi): (Vec2, Vec2)| {
            let room = (hi - lo - margin).max(Vec2::splat(40.0));
            (room.x / size.x.max(1e-6)).min(room.y / size.y.max(1e-6)).min(MOST_PER_UNIT).max(1e-3)
        };
        let whole = (Vec2::new(0.0, top), Vec2::new(self.canvas.x, bottom.max(top + 1.0)));
        let options = match (lay.narrow, lay.panel) {
            (false, Some((_, pmax))) => vec![(Vec2::new(pmax.x + 6.0, top), whole.1), (Vec2::new(0.0, (pmax.y + 6.0).min(bottom - 1.0)), whole.1), whole],
            _ => vec![whole],
        };
        // (the last, the whole canvas, only where neither part beside the panel is any use)
        let best = options[..options.len().saturating_sub(1).max(1)].iter().copied().max_by(|a, b| scale(*a).total_cmp(&scale(*b))).unwrap_or(whole);
        (best, scale(best))
    }

    fn the_view(&self) -> View {
        if self.space {
            return self.space_view();
        }
        let points: Vec<Vec2> = self.moments().iter().flat_map(|c| {
            let (a, b) = (c[0].truncate().as_vec2(), c[1].truncate().as_vec2());
            [Vec2::ZERO, a, b, a + b]
        }).filter(|p| p.is_finite()).collect();
        let lo = points.iter().fold(Vec2::splat(-0.3), |m, p| m.min(*p));
        let hi = points.iter().fold(Vec2::splat(0.3), |m, p| m.max(*p));
        let mid = (lo + hi) * 0.5;
        let size = (hi - lo).max(Vec2::splat(2.6));
        let ((min, max), per_unit) = self.region(size);
        // the middle of the region, which is not the canvas's, shows the middle of the
        // picture (screen y runs down, world y up)
        let off = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        View::Fit { center: Vec2::new(mid.x - off.x, mid.y + off.y), half: self.canvas.max(Vec2::ONE) * (0.5 / per_unit) }
    }

    /// What the camera takes in, in space: the boxes the steps pass through, seen
    /// from the opening direction, placed in the picture's part of the canvas.
    fn space_view(&self) -> View {
        let (right, up) = screen_axes();
        let mut points = vec![Vec3::ZERO];
        for c in self.moments() {
            let [a, b, d] = c.map(|v| v.as_vec3());
            points.extend([a, b, d, a + b, a + d, b + d, a + b + d]);
            points.push(c[0].cross(c[1]).as_vec3());
        }
        let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for p in points.iter().filter(|p| p.is_finite()) {
            let s = Vec2::new(p.dot(right), p.dot(up));
            (lo, hi) = (lo.min(s), hi.max(s));
        }
        let mid = (lo + hi) * 0.5;
        // (more round it than in the plane: seen in perspective, the near corners come
        // out larger than this measures them)
        let size = (hi - lo).max(Vec2::splat(1.5)) * 1.25;
        let ((min, max), _) = self.region(size);
        let margin = if Layout::of(self.canvas, self.inset, 3).narrow { NAME_ROOM_NARROW } else { NAME_ROOM };
        let region = (max - min - margin).max(Vec2::splat(60.0));
        let per_unit = (region.x / size.x).min(region.y / size.y).max(1.0);
        let dist = (self.canvas.y * 0.5 / per_unit / TAN_HALF_FOV).clamp(2.0, 80.0);
        let per_unit = self.canvas.y * 0.5 / (dist * TAN_HALF_FOV);
        let shift = ((min + max) * 0.5 - self.canvas * 0.5) / per_unit;
        let middle = right * mid.x + up * mid.y;
        View::Orbit { target: middle - right * shift.x + up * shift.y, yaw: YAW, pitch: PITCH, dist }
    }
}

impl Scene for Det {
    fn view(&self) -> View {
        self.the_view()
    }

    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.the_view())
    }

    /// Only the steps move by themselves, and an arrow gliding to where its tip was put.
    fn animated(&self) -> bool {
        self.moving() || !self.arrived()
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid_alpha = 0.12;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
        ui::note(ui, r"The columns of a $2 \times 2$ matrix are two arrows, $\textcolor{pink}{a_1}$ and $\textcolor{green}{a_2}$. Its determinant is the area of the parallelogram they span, counted positive if turning from $a_1$ to $a_2$ goes anticlockwise (as from the $x$ axis to the $y$ axis) and negative if it goes clockwise. In space, three columns span a slanted box, and the determinant is its volume. Drag the tips; drag the white dot to slide one side along another. Over the picture, the matrix and $ad - bc$ with its numbers.");

        ui::section(ui, "Things to try");
        for (k, trial) in TRIALS.iter().enumerate() {
            if ui::experiment(ui, trial.0, trial.1) {
                self.set_up(k);
            }
        }

        ui::section(ui, "Columns");
        let mut mode = self.space as usize;
        if ui::choice(ui, &["two, in the plane", "three, in space"], &mut mode) {
            self.space = mode == 1;
            (self.ops, self.end) = (Vec::new(), None);
            (self.stage, self.goal, self.through, self.hold) = (0.0, 0.0, false, 0.0);
            self.arrive();
            self.new_view = true;
        }
        let n = self.n();
        let mut c = self.shown();
        let mut typed = false;
        egui::Grid::new("det_columns").spacing([6.0, 4.0]).show(ui, |ui| {
            for k in 0..n {
                ui.label(ui::label_text(ui, &format!("${}$", NAMES[k]), 12.5, Some(COLS[k]), false));
            }
            ui.end_row();
            for r in 0..n {
                for col in c.iter_mut().take(n) {
                    let mut v = col[r];
                    if ui.add(egui::DragValue::new(&mut v).speed(0.1).max_decimals(2).range(-10.0..=10.0)).changed() {
                        col[r] = (v * 100.0).round() / 100.0;
                        typed = true;
                    }
                }
                ui.end_row();
            }
        });
        if typed {
            self.set_cols(c);
        }
        ui::note(ui, "Each column is an arrow, its numbers going down: $x$, then $y$ (then $z$). Dragging a tip moves it in tenths.");

        ui::section(ui, "Do one thing to the columns");
        let last = n - 1;
        let tools: [(String, Vec<Op>); 4] = [
            (format!("swap $a_1$ and ${}$", NAMES[last]), vec![Op::Swap { i: 0, j: last }]),
            (format!(r"${} \times 2$", NAMES[last]), vec![Op::Stretch { j: last, k: 2.0 }]),
            (format!(r"${} \times (-1)$", NAMES[last]), vec![Op::Stretch { j: last, k: -1.0 }]),
            (format!("${} + a_1$", NAMES[last]), vec![Op::Slide { j: last, i: 0, m: 1.0, clears: None }]),
        ];
        ui.horizontal_wrapped(|ui| {
            if ui.button(egui::RichText::new("slide it square").size(12.5)).clicked() {
                self.slide_square();
            }
            for (name, ops) in &tools {
                if ui.button(ui::label_text(ui, name, 12.5, None, false)).clicked() {
                    self.play(ops.clone(), None);
                }
            }
        });
        ui::note(ui, "Each plays as steps: ◀ back, next ▶ and ↺ replay over the picture go through them again. Dragging a tip, or typing a number, starts afresh from the picture as it is.");

        ui::section(ui, "Reading the picture");
        let mut key = vec![(COLS[0], "$a_1$, the first column (drag its tip)"), (COLS[1], "$a_2$, the second column")];
        if self.space {
            key.push((COLS[2], "$a_3$, the third column"));
            key.push((PLUS, "yellow box: positive volume ($a_1, a_2, a_3$ right-handed, like $x, y, z$); the base on $a_1$ and $a_2$ is filled more strongly"));
            key.push((MINUS, "blue box: negative volume (left-handed)"));
            key.push((pal::WHITE, r"white arrow $a_1 \times a_2$: at a right angle to the base, as long as the base's area"));
        } else {
            key.push((PLUS, "yellow: positive area (the turn from $a_1$ to $a_2$ is anticlockwise, shown by the small curved arrow)"));
            key.push((MINUS, "blue: negative area (the turn is clockwise)"));
        }
        key.push((pal::WHITE, "dashed white, with a white corner: the height, at a right angle to the base"));
        key.push((COLS[n - 1], "dashed in a column's colour: a line its tip can slide along without changing the height; the white dot slides it"));
        key.push((pal::FAINT, "faint dashed outline: the columns before the steps"));
        ui::key(ui, &key);

        ui::section(ui, "Words");
        ui::term(ui, "Determinant:", r"$\det A$, the signed area ($2 \times 2$) or volume ($3 \times 3$) spanned by the columns of $A$. For a $2 \times 2$ matrix it is $ad - bc$:");
        ui::equation(ui, r"A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \quad \det A = ad - bc");
        ui::term(ui, "Signed:", "counted negative when the columns turn the other way round from the axes: clockwise in the plane, left-handed in space.");
        ui::term(ui, "Pivot:", "the number elimination divides by in each row; the determinant is their product, with a minus sign for each swap.");
        ui::term(ui, "Elimination:", r"adding multiples of one row to another until the matrix is triangular. Sliding columns does the same to the columns, finds the same pivots when no swap is needed, and also leaves $\det A$ as it was.");
        ui::term(ui, "Linear in each column:", r#"$\det(a, b + c) = \det(a, b) + \det(a, c)$: over the same base the heights add, and so do the areas; and $\det(a, kb) = k \det(a, b)$ (the "$\times 2$" button above). Together they give the rule of slides: $\det(a, b + ma) = \det(a, b) + m \det(a, a)$, and $\det(a, a) = 0$, a flat parallelogram."#);
        ui::term(ui, "Dependent:", r"one column is made of the others: in one line (plane), in one plane (space). Exactly then $\det A = 0$.");
        if self.space {
            ui::term(ui, "Cross product:", r"$a_1 \times a_2$, the arrow at a right angle to $a_1$ and $a_2$, as long as the area of their parallelogram, pointing the way the right thumb does when the fingers curl from $a_1$ to $a_2$.");
            ui::term(ui, "Right-handed:", "$a_1, a_2, a_3$ turn like $x, y, z$: curl the right hand's fingers from $a_1$ to $a_2$, and the thumb points to the side of their plane that $a_3$ is on.");
        }

        ui::section(ui, "Numbers");
        let c = self.shown();
        let d = det(&c);
        let word = if n == 2 { "area" } else { "volume" };
        ui::readout(ui, &format!(r"$\det A$, the signed {word}"), written(d, 4));
        if let Some((base, height)) = base_and_height(&c, n, [0, 1]) {
            ui::readout(ui, if n == 2 { r"base $\lVert a_1 \rVert$" } else { r"base area $\lVert a_1 \times a_2 \rVert$" }, num(base, 3));
            ui::readout(ui, if n == 2 { "height of $a_2$ over it" } else { "height of $a_3$ over it" }, num(height, 3));
        }
        let turn = if is_flat(&c, n) {
            "none: flat"
        } else if d > 0.0 {
            if n == 2 {
                "anticlockwise: +"
            } else {
                "right-handed: +"
            }
        } else if n == 2 {
            "clockwise: −"
        } else {
            "left-handed: −"
        };
        ui::readout(ui, if n == 2 { "turn from $a_1$ to $a_2$" } else { "$a_1, a_2, a_3$" }, turn);
        match square_ops(c, n).1 {
            End::Brick { pivots, swaps } => {
                let p: Vec<String> = pivots[..n].iter().map(|v| num(*v, 3)).collect();
                ui::readout(ui, "pivots", p.join(", "));
                ui::readout(ui, "swaps on the way", swaps.to_string());
            }
            End::Flat { row, .. } => ui::readout(ui, "pivots", format!(r"none in row {}: $\det A = 0$", row + 1)),
        }
        ui::see_also(ui, &[("linmap2", "the determinant as how much a map scales areas"), ("compose", r"$\det AB = \det A \times \det B$"), ("elim", "elimination and its pivots, on the rows"), ("vectors", "the cross product: an area as an arrow")]);
    }

    fn frame(&mut self, f: &mut Frame) {
        self.advance(f.dt);
        // the panel's place follows the canvas, and the picture's place follows the panel's
        let view = f.view_size();
        let n = self.n();
        let lay = Layout::of(view, f.top_inset, n);
        let before = Layout::of(self.canvas, self.inset, n);
        // (a canvas squeezed to a sliver is one the controls have opened over, on a
        // phone: the picture is not seen then, and is left as it was for when they close)
        let covered = view.x < 160.0;
        let changed = !covered && (lay.narrow != before.narrow || lay.panel.is_some() != before.panel.is_some() || (view - self.canvas).abs().max_element() > 24.0);
        let first = !covered && !self.measured && (view - self.canvas).abs().max_element() > 1.0;
        self.new_view |= changed || first;
        if !covered && (changed || !self.measured) {
            self.canvas = view;
            self.measured = true;
        }
        self.inset = f.top_inset;
        let pointer = f.ptr.pos.map(|p| p / f.px);
        if pointer.is_some_and(|p| lay.covers(p)) {
            // the panel is for its buttons: no moving the view from there, and two
            // quick clicks are two steps, not a reset of the view
            f.handles.capture();
            f.handles.claim_clicks();
        }

        // The arrows glide to where the tips have stepped (by the clock, as the view's
        // own glide is: it answers the hand), but not while the scene's time stands
        // still: nothing glides then, and a drag must still be seen. (Before the
        // handles, so that each is drawn where its arrow now ends.)
        self.glide(f.real_dt, f.dt <= 0.0);
        self.handles(f, &lay);
        let labels_before = f.dl.labels.len();
        self.keep = (lay.panel, f.top_inset, view);
        let mut room = Room::new(f);
        if let Some((min, max)) = lay.panel {
            room.wall(min - Vec2::splat(4.0), max + Vec2::splat(4.0));
        }
        let bottom = if lay.narrow { lay.below - 10.0 } else { view.y - 18.0 };
        if view.y >= 300.0 {
            room.wall(Vec2::new(0.0, bottom - 22.0), Vec2::new(view.x, bottom + 4.0));
        }
        if self.space {
            self.draw_space(f, &mut room);
        } else {
            self.draw_plane(f, &mut room);
        }
        // nothing written into the picture (the numbers along the axes too) may show
        // through the panel, or under the application's buttons; and in a sliver of a
        // canvas (beside a phone's open controls) nothing at all: there is no room
        let (lo, hi) = lay.panel.map_or((Vec2::splat(-1.0), Vec2::splat(-1.0)), |(a, b)| (a - Vec2::splat(4.0), b + Vec2::splat(4.0)));
        let (mut k, inset) = (0, f.top_inset);
        f.dl.labels.retain(|l| {
            k += 1;
            k <= labels_before || !(covered || l.pos.y < inset || l.pos.cmpge(lo).all() && l.pos.cmple(hi).all())
        });
        self.panel(f, &lay, pointer);
        let parts: &[&str] = if self.space { &["drag the coloured tips", "the white dot slides the top of the box over its base"] } else { &["drag the coloured tips", "the white dot slides $a_2$ along $a_1$"] };
        if !covered {
            caption(f, parts, bottom);
        }
    }
}

/// Where the white corner puts the last column when it is dragged to `grip`: `from`
/// (that column as the drag began) moved by whole tenths of the base's sides, of a₁
/// in the plane and of a₁ and a₂ in space, so that its height over the base, and the
/// determinant, stay as they were. Also where the dot is drawn: at `grip`, kept to the
/// line (or the plane) it can move in. None if the base has no size.
fn slid(c: &[DVec3; 3], n: usize, from: DVec3, grip: DVec3) -> Option<(DVec3, DVec3)> {
    let rest = if n == 2 { c[0] } else { c[0] + c[1] };
    let d = grip - rest - from;
    let snapped = tenth;
    let (by, on) = if n == 2 {
        // along a₁ only: the corner keeps to its line
        let a = c[0];
        let s = d.dot(a) / a.length_squared();
        (a * snapped(s), rest + from + a * s)
    } else {
        // across the base: d = s a₁ + t a₂, as near as can be
        let (a, b) = (c[0], c[1]);
        let (aa, ab, bb) = (a.dot(a), a.dot(b), b.dot(b));
        let g = aa * bb - ab * ab;
        let (s, t) = ((d.dot(a) * bb - d.dot(b) * ab) / g, (d.dot(b) * aa - d.dot(a) * ab) / g);
        (a * snapped(s) + b * snapped(t), rest + from + a * s + b * t)
    };
    let column = from + by;
    (column.is_finite() && on.is_finite()).then_some((column, on))
}

/// The base and the height over it: in the plane the length of column `base[0]` and
/// the height of the other over its line; in space the area of the parallelogram on
/// columns `base[0]` and `base[1]`, and the height of the third over its plane. None
/// where the base has no size.
fn base_and_height(c: &[DVec3; 3], n: usize, base: [usize; 2]) -> Option<(f64, f64)> {
    let size = if n == 2 { c[base[0]].length() } else { c[base[0]].cross(c[base[1]]).length() };
    let largest = largest(c, n).max(1e-300);
    (size > ZERO * largest.powi(n as i32 - 1)).then(|| (size, det(c).abs() / size))
}

impl Det {
    /// The handles: the tips, and the white corner that slides one side along another.
    /// Dragging any of them starts afresh from the picture as it is.
    fn handles(&mut self, f: &mut Frame, lay: &Layout) {
        let (n, flat) = (self.n(), !self.space);
        let reach = REACH[self.space as usize];
        if !f.handles.dragging() {
            // (each handle at the tip of its arrow, and the corner at the corner)
            self.slide_from = None;
            for k in 0..3 {
                self.grip[k] = self.drawn[k];
            }
            self.grip[3] = self.corner();
        }
        for k in 0..n {
            let hidden = on_screen(f, self.drawn[k]).is_some_and(|p| lay.covers(p));
            if hidden && !f.handles.dragging() {
                continue;
            }
            let mut grip = self.grip[k];
            let moved = if flat {
                let mut g = grip.truncate();
                let moved = f.drag2(1 + k as u32, &mut g, COLS[k]);
                grip = g.extend(0.0);
                moved
            } else {
                f.drag3(1 + k as u32, &mut grip, None, COLS[k])
            };
            if moved {
                self.grip[k] = grip;
                let mut c = self.shown();
                c[k] = snap(grip, reach, flat);
                self.set_cols(c);
            }
        }

        // the white corner: in the plane it moves a₂ along a₁, in space the top of the
        // box (a₃) across its base
        let c = self.shown();
        let mover = n - 1;
        let usable = if flat { c[0].length() > 1e-6 } else { c[0].cross(c[1]).length() > 1e-6 };
        let at = self.corner();
        let hidden = on_screen(f, at).is_some_and(|p| lay.covers(p));
        if usable && (!hidden || f.handles.dragging()) {
            let mut grip = if self.slide_from.is_some() { self.grip[3] } else { at };
            let moved = if flat {
                let mut g = grip.truncate();
                let moved = f.drag2(4, &mut g, pal::WHITE);
                grip = g.extend(0.0);
                moved
            } else {
                let normal = c[0].cross(c[1]).normalize().as_vec3();
                f.drag3(4, &mut grip, Some(normal), pal::WHITE)
            };
            if moved {
                if self.slide_from.is_none() {
                    self.commit();
                    self.slide_from = Some(self.cols()[mover]);
                }
                if let Some((column, on)) = self.slide_from.and_then(|from| slid(&self.cols(), n, from, grip.as_dvec3())) {
                    self.start[self.space as usize][mover] = column;
                    self.grip[3] = on.as_vec3();
                }
            }
        }
    }

    /// The colour of the area or volume now: yellow positive, blue negative, grey none.
    fn tone(&self) -> Color {
        let c = self.shown();
        if is_flat(&c, self.n()) {
            pal::DIM
        } else if det(&c) > 0.0 {
            PLUS
        } else {
            MINUS
        }
    }

    /// The marks of one set of roles in the plane, `alpha` strong: the base and the
    /// line it lies on, the line the moving tip runs along, the height with its right
    /// angle, and, while a column is stretched, the copies of the parallelogram it now
    /// holds. Returns where the height and that line are, for their names.
    #[allow(clippy::type_complexity)]
    fn plane_marks(&self, f: &mut Frame, room: &mut Room, (j, [i, _], along): Roles, alpha: f32, tone: Color) -> (Option<(Vec3, Vec3)>, Option<(Vec3, Vec3, usize)>) {
        let wpp = f.cam.world_per_px(Vec3::ZERO) * f.px;
        let (b, m) = (self.drawn[i], self.drawn[j]);
        let (mut height_at, mut rail_at) = (None, None);
        if b.length() <= 4.0 * wpp {
            return (None, None);
        }
        let u = b.normalize();
        let long = 40.0;
        // the line of the base, both ways (the height may land beyond it)
        f.dashed(-u * long, u * long, 1.0, COLS[i].a(0.35 * alpha));
        // the base itself, as a band under its arrow
        f.line(Vec3::ZERO, b, 9.0, COLS[i].a(0.28 * alpha));
        // the line the moving tip runs along, parallel to the base
        if let Some(along) = along {
            let v = self.drawn[along].normalize_or_zero();
            if v != Vec3::ZERO {
                f.dashed(m - v * long, m + v * long, 1.3, COLS[j].a(0.55 * alpha));
                room.line(f, m - v * long, m + v * long);
                rail_at = Some((m, v, j));
            }
        }
        let foot = u * u.dot(m);
        let h = (m - foot).length();
        if h > 6.0 * wpp {
            f.dashed(m, foot, 1.6, pal::WHITE.a(0.85 * alpha));
            room.line(f, m, foot);
            let size = (12.0 * wpp).min(0.45 * h);
            // (always on the same side of the foot: a mark that changed sides as the
            // foot went by would jump)
            right_angle(f, foot, u, m - foot, size, pal::WHITE.a(0.9 * alpha));
            height_at = Some((m, foot));
        }
        // stretched: the copies of the parallelogram it now holds
        if let Some(Op::Stretch { .. }) = self.ops.get(self.step().max(1) - 1).filter(|_| self.stage > 0.0) {
            let original = self.after(self.step() - 1)[j].as_vec3();
            let len = original.length();
            if len > 1e-6 {
                let times = (m.dot(original) / (len * len)).abs();
                let dir = original * m.dot(original).signum();
                for k in 1..(times.floor() as i32).min(20) + 1 {
                    if (times - k as f32).abs() > 0.02 {
                        f.line(dir * k as f32, dir * k as f32 + b, 1.2, tone.a(0.7 * alpha));
                    }
                }
            }
        }
        (height_at, rail_at)
    }

    /// The plane: the parallelogram, what it is measured by, and the arrows.
    fn draw_plane(&self, f: &mut Frame, room: &mut Room) {
        let [a1, a2, _] = self.drawn;
        let said = self.shown();
        let d = det(&said);
        let flat = is_flat(&said, 2);
        let tone = self.tone();
        let wpp = f.cam.world_per_px(Vec3::ZERO) * f.px;

        // where the steps started, faintly, once they have moved away from it
        if self.stage > 0.0 && !self.ops.is_empty() {
            let s = self.cols().map(|v| v.as_vec3());
            f.dashed_path([Vec3::ZERO, s[0], s[0] + s[1], s[1], Vec3::ZERO], 1.2, pal::FAINT);
        }

        // the parallelogram
        if !flat {
            f.dl.tris_alpha.quad(Vec3::ZERO, a1, a1 + a2, a2, tone.a(0.24));
            f.dl.lines.closed(&[Vec3::ZERO, a1, a1 + a2, a2], 1.4, tone.a(0.8));
            room.shape(f, &[Vec3::ZERO, a1, a1 + a2, a2]);
        }
        room.line(f, a1, a1 + a2);
        room.line(f, a2, a1 + a2);

        // the base, the height over it, and the line the moving tip keeps to (named
        // only while one set of them is drawn)
        let marks = self.marks();
        let mut height_at: Option<(Vec3, Vec3)> = None;
        let mut rail_at: Option<(Vec3, Vec3, usize)> = None;
        for &(roles, alpha) in &marks {
            let (h, r) = self.plane_marks(f, room, roles, alpha, tone);
            if named(&marks) {
                (height_at, rail_at) = (h, r);
            }
        }

        // the turn from a₁ to a₂, as a small curved arrow at the start
        let mut arc_label: Option<Vec3> = None;
        if !flat && a1.length() > 1e-4 && a2.length() > 1e-4 {
            let (t1, t2) = (a1.y.atan2(a1.x), a2.y.atan2(a2.x));
            let mut turn = t2 - t1;
            while turn > PI {
                turn -= TAU;
            }
            while turn <= -PI {
                turn += TAU;
            }
            let r = (42.0 * wpp).min(0.4 * a1.length().min(a2.length()));
            if r > 10.0 * wpp && turn.abs() > 0.15 {
                let at = |t: f32| Vec3::new((t1 + turn * t).cos(), (t1 + turn * t).sin(), 0.0) * r;
                let pts: Vec<Vec3> = (0..=32).map(|k| at(k as f32 / 32.0)).collect();
                f.path(pts.iter().copied(), 1.8, tone);
                let back = ((11.0 * wpp) / (r * turn.abs())).min(0.5);
                f.arrow(at(1.0 - back), at(1.0), 1.8, tone);
                for k in (0..32).step_by(4) {
                    room.line(f, pts[k], pts[k + 4]);
                }
                arc_label = Some(at(0.5));
            }
        }

        // the brick at the end of "slide it square": its sides are the pivots
        let brick = match self.end {
            Some(End::Brick { pivots, .. }) if self.ended() => Some(pivots),
            _ => None,
        };

        // the arrows
        for k in 0..2 {
            f.arrow(Vec3::ZERO, self.drawn[k], 3.4, COLS[k]);
            room.line(f, Vec3::ZERO, self.drawn[k]);
        }

        // names and numbers, the most important first
        let outwards = |f: &Frame, v: Vec3| match (on_screen(f, Vec3::ZERO), on_screen(f, v)) {
            (Some(p), Some(q)) if (q - p).length() > 1.0 => q - p,
            _ => Vec2::new(1.0, -1.0),
        };
        for k in 0..2 {
            let dir = outwards(f, self.drawn[k]);
            room.label(f, ("name", k), self.drawn[k], &format!("${}$", NAMES[k]), 17.0, COLS[k], dir, 10.0, false);
        }
        // the area in the middle, on a plate (lines may run under it there): it then
        // moves only as the parallelogram does, through a swap and a squash too
        let area_text = if flat { "area $0$: in one line".to_string() } else { format!("area ${}$", measured(d)) };
        plate(f, room, self.keep, (a1 + a2) * 0.5, &area_text, 14.0, tone.mix(pal::WHITE, 0.35));
        if let Some(pivots) = brick {
            // beside each side, on the side away from the numbers along its axis (they
            // are under the x axis and to the right of the y axis): above a₁, to the
            // left of a₂, inside the rectangle or outside it as that falls out (named
            // from just inside it when inside)
            // (and on the other side where that one has no room: on a phone, the strip
            // left of the y axis may be too narrow)
            let middle = (a1 + a2) * 0.5;
            for (k, towards) in [(0, Vec2::new(0.0, -1.0)), (1, Vec2::new(-1.0, 0.0))] {
                let mid = self.drawn[k] * 0.5;
                let text = format!("${} = {}$", PIVOTS[k], written(pivots[k], 3));
                let colour = COLS[k].mix(pal::WHITE, 0.3);
                let anchor = |f: &Frame, towards: Vec2| {
                    let inward = on_screen(f, middle).zip(on_screen(f, mid)).is_some_and(|(c, m)| (c - m).dot(towards) > 0.0);
                    if inward {
                        mid + (middle - mid) * 0.08
                    } else {
                        mid
                    }
                };
                // the middle of the side, then further along it, away from the start
                let places = [(0.5, towards), (0.5, -towards), (0.8, -towards), (0.8, towards)];
                let placed = places.iter().any(|&(along, side)| {
                    let at = anchor(f, side) + self.drawn[k] * (along - 0.5);
                    room.label(f, ("pivot", k), at, &text, 14.0, colour, side, 8.0, true)
                });
                if !placed {
                    room.label(f, ("pivot", k), anchor(f, towards), &text, 14.0, colour, towards, 8.0, false);
                }
            }
        }
        let roles = named(&marks).then(|| marks[0].0);
        if let (Some((m, foot)), Some((_, base, _))) = (height_at, roles) {
            if let Some((_, height)) = base_and_height(&said, 2, base) {
                let away = outwards(f, (m + foot) * 0.5 - (a1 + a2) * 0.5);
                room.label(f, "height", (m + foot) * 0.5, &format!("height ${}$", short(height)), 12.5, pal::TEXT, away, 8.0, false);
            }
        }
        if let Some((_, [i, _], _)) = roles {
            let b = self.drawn[i];
            if b.length() > 1e-3 {
                let other = self.drawn[1 - i];
                let away = -outwards(f, other);
                room.label(f, "base", b * 0.5, &format!("base ${}$", short(said[i].length())), 12.5, COLS[i].mix(pal::WHITE, 0.4), away, 9.0, false);
            }
        }
        if let Some((m, v, j)) = rail_at {
            // behind the moving tip (the corner's end of the line runs off the screen
            // first), set off the line on the side away from the base, far enough that
            // a label as wide as this clears a line that slopes; in one place only, and
            // left out where it has no room there
            let text = format!("${}$'s tip slides along here", NAMES[j]);
            let base = self.drawn[roles.map_or(0, |r| r.1[0])];
            let along = if base.dot(v) >= 0.0 { -v } else { v };
            let up = m - along * along.dot(m);
            if up.length() > 12.0 * wpp {
                let normal = up.normalize();
                let (half_w, half_h) = (f.text_size(&text, 12.0).x * 0.5 + 4.0, 7.0);
                let off = 8.0 + along.y.abs() * half_w + along.x.abs() * half_h;
                let at = m + along * (100.0 * wpp) + normal * (off * wpp);
                if let Some(p) = on_screen(f, at) {
                    let half = Vec2::new(half_w, half_h);
                    if room.room_for(p, p, half) >= 2.0 {
                        f.text_px(p, text.as_str(), 12.0, COLS[j].mix(pal::WHITE, 0.25), Align2::CENTER_CENTER);
                        room.boxed(p - half, p + half);
                    }
                }
            }
        }
        if let (Some(at), Some(origin)) = (arc_label, on_screen(f, Vec3::ZERO)) {
            // one place, so that it never hops: just beyond the start, on the side away
            // from the arc (between the arrows there is seldom room), on a plate
            let text = if d > 0.0 { "↺ anticlockwise: +" } else { "↻ clockwise: −" };
            let away = -outwards(f, at).normalize_or_zero();
            let half = Vec2::new(text_width(text, 12.0) * 0.5 + 4.0, 8.0);
            let centre = keep_out(origin + away * (10.0 + away.x.abs() * half.x + away.y.abs() * half.y), half, self.keep);
            f.caption_px(centre, text, 12.0, tone.mix(pal::WHITE, 0.3), Align2::CENTER_CENTER);
            room.boxed(centre - half, centre + half);
        }
        f.axes2();
    }

    /// The marks of one set of roles in space, `alpha` strong: the base filled, the
    /// height with its right angle, and the line (or, while the white dot slides the
    /// top, the sheet) the moving tip keeps to. Returns where the height is.
    #[allow(clippy::too_many_arguments)]
    fn space_marks(&self, f: &mut Frame, room: &mut Room, (j, [p, q], along): Roles, alpha: f32, tone: Color, reach: f32) -> Option<(Vec3, Vec3)> {
        let wpp = |f: &Frame, at: Vec3| f.cam.world_per_px(at) * f.px;
        let (u, v, m) = (self.drawn[p], self.drawn[q], self.drawn[j]);
        let normal = u.cross(v);
        if normal.length() <= 1e-6 {
            return None;
        }
        let nn = normal.normalize();
        f.dl.tris_alpha.quad(Vec3::ZERO, u, u + v, v, tone.a(0.3 * alpha));
        f.dl.lines.closed(&[Vec3::ZERO, u, u + v, v], 1.6, tone.a(0.9 * alpha));
        let foot = m - nn * nn.dot(m);
        let h = (m - foot).length();
        let mut height_at = None;
        if h > 6.0 * wpp(f, foot) {
            f.dashed(m, foot, 1.6, pal::WHITE.a(0.9 * alpha));
            room.line(f, m, foot);
            let size = (13.0 * wpp(f, foot)).min(0.45 * h);
            // (always on the same side of the foot, as in the plane)
            right_angle(f, foot, u, m - foot, size, pal::WHITE.a(0.9 * alpha));
            height_at = Some((m, foot));
            // where the foot is in the base's plane: dashed back to the base's corner
            f.dashed(Vec3::ZERO, foot, 1.0, pal::WHITE.a(0.3 * alpha));
        }
        if let Some(along) = along {
            let w = self.drawn[along].normalize_or_zero();
            let long = 3.0 * reach;
            f.dashed(m - w * long, m + w * long, 1.3, COLS[j].a(0.55 * alpha));
            room.line(f, m - w * long, m + w * long);
        } else if self.slide_from.is_some() {
            // the top of the box, while the white dot slides it: anywhere across this
            // sheet keeps the height
            let (e1, e2) = (u.normalize(), nn.cross(u.normalize()));
            let half = reach * 0.9;
            let centre = m + (u + v) * 0.5;
            let at = |s: f32, t: f32| centre + e1 * s + e2 * t;
            f.dl.tris_alpha.quad(at(-half, -half), at(half, -half), at(half, half), at(-half, half), COLS[j].a(0.05 * alpha));
            for k in -2..=2 {
                let s = k as f32 * half / 2.0;
                f.dashed(at(s, -half), at(s, half), 1.0, COLS[j].a(0.25 * alpha));
                f.dashed(at(-half, s), at(half, s), 1.0, COLS[j].a(0.25 * alpha));
            }
        }
        height_at
    }

    /// Space: the box, its base and height, the arrows.
    fn draw_space(&self, f: &mut Frame, room: &mut Room) {
        let [a, b, c] = self.drawn;
        let said = self.shown();
        let flat = is_flat(&said, 3);
        let tone = self.tone();
        let wpp = |f: &Frame, at: Vec3| f.cam.world_per_px(at) * f.px;

        // the axes, faint, for the coordinates the sums are made of
        let reach = [a, b, c, a + b + c].iter().map(|v| v.abs().max_element()).fold(2.0f32, f32::max) * 1.15;
        for d in Vec3::AXES {
            f.line(-d * reach, d * reach, 1.0, pal::FAINT.a(0.8));
            room.line(f, -d * reach, d * reach);
        }

        // where the steps started, faintly
        if self.stage > 0.0 && !self.ops.is_empty() {
            let s = self.cols().map(|v| v.as_vec3());
            for (o, d) in box_edges(s[0], s[1], s[2]) {
                f.dashed(o, o + d, 1.0, pal::FAINT);
            }
        }

        // the box
        slanted_box(f, a, b, c, tone, if flat { 0.0 } else { 1.0 });
        for (o, d) in box_edges(a, b, c) {
            room.line(f, o, o + d);
        }

        // the base, the height over it, and the line or sheet the moving tip keeps to
        // (named only while one set of them is drawn)
        let marks = self.marks();
        let mut height_at: Option<(Vec3, Vec3)> = None;
        for &(roles, alpha) in &marks {
            let h = self.space_marks(f, room, roles, alpha, tone, reach);
            if named(&marks) {
                height_at = h;
            }
        }
        let roles = named(&marks).then(|| marks[0].0);

        // the cross product of the base's sides, before any steps
        let cross = a.cross(b);
        let show_cross = (self.stage <= 0.0 || self.ops.is_empty()) && cross.length() > 1e-4;
        if show_cross {
            f.arrow(Vec3::ZERO, cross, 2.0, pal::WHITE.a(0.75));
            let size = (14.0 * wpp(f, Vec3::ZERO)).min(0.3 * a.length().min(b.length()).min(cross.length()));
            right_angle(f, Vec3::ZERO, cross, a, size, pal::WHITE.a(0.6));
            right_angle(f, Vec3::ZERO, cross, b, size, pal::WHITE.a(0.6));
            room.line(f, Vec3::ZERO, cross);
        }

        // the arrows
        for k in 0..3 {
            f.arrow(Vec3::ZERO, self.drawn[k], 3.4, COLS[k]);
            room.line(f, Vec3::ZERO, self.drawn[k]);
        }

        // names and numbers
        let outwards = |f: &Frame, v: Vec3| match (on_screen(f, Vec3::ZERO), on_screen(f, v)) {
            (Some(p), Some(q)) if (q - p).length() > 1.0 => q - p,
            _ => Vec2::new(1.0, -1.0),
        };
        for k in 0..3 {
            let dir = outwards(f, self.drawn[k]);
            room.label(f, ("name", k), self.drawn[k], &format!("${}$", NAMES[k]), 17.0, COLS[k], dir, 10.0, false);
        }
        if show_cross {
            let dir = outwards(f, cross);
            room.label(f, "cross", cross, r"$a_1 \times a_2$", 14.0, pal::WHITE, dir, 8.0, true);
        }
        let d = det(&said);
        let text = if flat { "volume $0$: the columns lie in one plane".to_string() } else { format!("volume ${}$", measured(d)) };
        let centre = (a + b + c) * 0.5;
        plate(f, room, self.keep, centre, &text, 14.0, tone.mix(pal::WHITE, 0.35));
        if let Some(Some(End::Brick { pivots, .. })) = self.ended().then_some(self.end) {
            for k in 0..3 {
                let mid = self.drawn[k] * 0.5;
                let dir = outwards(f, self.drawn[k]).perp();
                room.label(f, ("pivot", k), mid, &format!("${} = {}$", PIVOTS[k], written(pivots[k], 3)), 13.5, COLS[k].mix(pal::WHITE, 0.3), dir, 8.0, false);
            }
        }
        if let (Some((m, foot)), Some((_, base, _))) = (height_at, roles) {
            if let Some((area, height)) = base_and_height(&said, 3, base) {
                let away = outwards(f, (m + foot) * 0.5 - centre);
                room.label_any(f, "height", &[(m + foot) * 0.5, m * 0.3 + foot * 0.7, m * 0.7 + foot * 0.3], &format!("height ${}$", short(height)), 12.5, pal::TEXT, away, 8.0, false);
                let base_mid = (self.drawn[base[0]] + self.drawn[base[1]]) * 0.5;
                room.label_any(f, "base", &[base_mid, base_mid * 0.6, base_mid * 1.3], &format!("base area ${}$", short(area)), 12.5, tone.mix(pal::WHITE, 0.5), Vec2::ZERO, 0.0, true);
            }
        }
        // the axes' names, where they are not lost in the box (seen through it, the end
        // of an axis behind it would look as if it were inside)
        let outline = hull(box_corners(a, b, c).iter().filter_map(|p| on_screen(f, *p)).collect());
        for (k, name) in AXES.into_iter().enumerate() {
            let d = Vec3::AXES[k] * reach;
            if on_screen(f, d).is_some_and(|p| !inside(p, &outline)) {
                room.label(f, ("axis", k), d, &format!("${name}$"), 12.0, pal::DIM, outwards(f, d), 4.0, true);
            }
        }
    }

    /// The panel over the picture: the step and its buttons, the matrix, and the sums.
    fn panel(&mut self, f: &mut Frame, lay: &Layout, pointer: Option<Vec2>) {
        let Some((min, max)) = lay.panel else { return };
        hud_panel(f, min, max);
        let n = self.n();
        let x = min.x + 12.0;
        let width = max.x - min.x - 24.0;
        let mut y = min.y + 10.0;
        let (head, says) = self.heading();
        f.text_px(Vec2::new(x, y), head, 13.0, pal::TEXT, Align2::LEFT_TOP);
        y += 19.0;
        for line in kept_lines(&mut self.broken[0], vec![says], width, |t, w| t.iter().flat_map(|t| wrap(t, w, SAYS_SIZE)).collect()).into_iter().take(SAYS_LINES) {
            f.text_px(Vec2::new(x, y), line, SAYS_SIZE, pal::DIM, Align2::LEFT_TOP);
            y += 15.0;
        }
        y = min.y + 10.0 + 19.0 + SAYS_LINES as f32 * 15.0 + 6.0;

        // the buttons
        let have = !self.ops.is_empty();
        let at_start = self.goal.min(self.stage) <= 0.0;
        let gap = 6.0;
        let widths = [64.0, 64.0, 76.0, 112.0];
        let scale = ((width - 3.0 * gap) / widths.iter().sum::<f32>()).min(1.0);
        let mut bx = x;
        let mut clicked = [false; 4];
        for (k, (text, usable)) in [("◀ back", have && !at_start), ("next ▶", have && self.can_go_on()), ("↺ replay", have), ("slide it square", true)].into_iter().enumerate() {
            let w = widths[k] * scale;
            clicked[k] = hud_button(f, Vec2::new(bx, y), Vec2::new(bx + w, y + 26.0), text, usable, pointer);
            bx += w + gap;
        }
        if clicked[0] {
            self.back();
        }
        if clicked[1] {
            self.next();
        }
        if clicked[2] {
            self.replay();
        }
        if clicked[3] {
            self.slide_square();
        }
        y += 26.0 + 12.0;

        // the matrix: as it was, and as it is now
        let shown = self.shown();
        let start = self.cols();
        let moved = have && self.stage > 0.0;
        let entry = |c: &[DVec3; 3], r: usize, k: usize| format!("${}$", short(c[k][r]));
        let widest = self.widest();
        let mut size = 13.0;
        // (the matrix after its name, "A =", as wide as it is set; and between the two
        // matrices, room for the arrow from one to the other)
        let lead = f.text_size("$A =$", 13.0).x.max(23.0) + 7.0;
        let between = 26.0;
        let room_for = if moved { (width + 6.0 - lead - between) * 0.5 } else { width - 10.0 - lead };
        while matrix_size(n, n, widest, size).x > room_for && size > 9.0 {
            size -= 0.5;
        }
        let rows = |c: &[DVec3; 3], dim: bool| -> Vec<Vec<(String, Color)>> {
            (0..n).map(|r| (0..n).map(|k| (entry(c, r, k), if dim { COLS[k].mix(pal::DIM, 0.55) } else { COLS[k] })).collect()).collect()
        };
        let msize = matrix_size(n, n, widest, size);
        let top = y + 15.0;
        if moved {
            f.text_px(Vec2::new(x + lead, y), "at the start", 11.0, pal::DIM, Align2::LEFT_TOP);
        }
        f.text_px(Vec2::new(x, top + msize.y * 0.5), "$A =$", 13.0, pal::TEXT, Align2::LEFT_CENTER);
        hud_matrix(f, Vec2::new(x + lead, top), &rows(&start, moved), widest, size, if moved { pal::DIM } else { pal::TEXT });
        if moved {
            let x2 = x + lead + msize.x + between;
            let k = self.step();
            let after = if self.stage >= k as f32 { format!("after step {k}") } else { format!("during step {k}") };
            f.text_px(Vec2::new(x2, y), after, 11.0, pal::DIM, Align2::LEFT_TOP);
            f.text_px(Vec2::new(x2 - between * 0.5, top + msize.y * 0.5), r"$\to$", 12.0, pal::DIM, Align2::CENTER_CENTER);
            hud_matrix(f, Vec2::new(x2, top), &rows(&shown, false), widest, size, pal::TEXT);
        }
        y = top + n as f32 * 22.0 + 10.0;

        let sums = self.sums();
        for line in kept_lines(&mut self.broken[1], sums, width, sum_lines).into_iter().take(SUM_LINES[self.space as usize]) {
            f.text_px(Vec2::new(x, y), line, SUM_SIZE, pal::TEXT, Align2::LEFT_TOP);
            y += 16.0;
        }
    }

    /// How many characters the matrix's cells keep room for: as many as its numbers
    /// take at any moment of the steps, so that the matrix keeps its size while they
    /// play (a number gaining a digit would otherwise throw its brackets out). Each
    /// entry moves in a straight line during a step, so the largest it gets is at one
    /// end of it; to two decimals, "−9.99" is the longest below 10.
    fn widest(&self) -> usize {
        let n = self.n();
        let largest = self.moments().iter().map(|c| largest(c, n)).fold(0.0, f64::max);
        let whole = if largest.is_finite() && largest >= 9.995 { (largest + 0.005).log10().floor() as usize + 1 } else { 1 };
        whole + 4
    }

    /// The sums under the matrix: with the numbers as they are now, or, once "slide it
    /// square" is done, with A's own, beside what the pivots make of them.
    fn sums(&self) -> Vec<String> {
        let (n, shown, start) = (self.n(), self.shown(), self.cols());
        let ended = self.end.filter(|_| self.ended());
        let summed = if ended.is_some() { &start } else { &shown };
        let mut lines: Vec<String> = if n == 2 { vec![plane_sum(summed)] } else { space_sum(summed).to_vec() };
        match ended {
            Some(end) => lines.extend(end_says(end, &start, n)),
            None => {
                if let Some((base, height)) = self.roles().and_then(|(_, base, _)| base_and_height(&shown, n, base)) {
                    let what = if n == 2 { r"|\text{area}| = \text{base} \times \text{height}" } else { r"|\text{volume}| = \text{base area} \times \text{height}" };
                    lines.push(format!(r"${what} = {} \times {}$", written(base, 2), written(height, 2)));
                }
            }
        }
        lines
    }

    /// What the panel says at its top: a heading, and a few lines under it.
    fn heading(&self) -> (String, String) {
        let n = self.n();
        let count = self.ops.len();
        let k = self.step();
        if let Some(end) = self.end.filter(|_| self.ended()) {
            return match end {
                End::Brick { swaps, .. } => (
                    format!("The end: a {} along the axes", if n == 2 { "rectangle" } else { "brick" }),
                    format!(
                        "Its sides are the pivots, the numbers left on the diagonal, and its {} is their product{}.",
                        if n == 2 { "area" } else { "volume" },
                        if swaps % 2 == 1 { ", with its sign flipped by the swap" } else { "" }
                    ),
                ),
                End::Flat { row, .. } => (r"No pivot: $\det A = 0$".to_string(), flat_says(n, row)),
            };
        }
        if count > 0 && k > 0 {
            let op = self.ops[k - 1];
            let before = det(&self.after(k - 1));
            return (format!("Step {k} of {count}: {}", op_name(op)), op_says(op, n, before, self.end.is_some()));
        }
        if n == 2 {
            (
                r"$\det A$: the signed area on the columns".to_string(),
                "The area of the parallelogram on $a_1$ and $a_2$, counted + when the turn from $a_1$ to $a_2$ is anticlockwise, as from the $x$ axis to the $y$ axis, and − when it is clockwise.".to_string(),
            )
        } else {
            (
                r"$\det A$: the signed volume on the columns".to_string(),
                r"The volume of the box on $a_1, a_2, a_3$, counted + when they turn like $x, y, z$ (right-handed: $a_3$ on the side $a_1 \times a_2$ points to) and − when they do not.".to_string(),
            )
        }
    }
}

/// A label on a dark plate, centred on the scene's point `at` (lines may run under
/// it), kept clear by the labels written after it.
fn plate(f: &mut Frame, room: &mut Room, keep: Keep, at: Vec3, text: &str, size: f32, colour: Color) {
    if let Some(p) = on_screen(f, at) {
        let half = Vec2::new(f.text_size(text, size).x * 0.5 + 6.0, size * 0.5 + 3.0);
        let p = keep_out(p, half, keep);
        f.caption_px(p, text, size, colour, Align2::CENTER_CENTER);
        room.boxed(p - half, p + half);
    }
}

/// What labels keep out of: the panel over the picture, if there is one, and the strip
/// under the application's buttons (its height); and the size of the canvas they keep to.
type Keep = (Option<(Vec2, Vec2)>, f32, Vec2);

/// Where a label centred at `p`, of half-size `half`, goes to keep out of the panel
/// and from under the application's buttons: pushed across the nearest edge, so that
/// it moves on as `p` does. (Removed there instead, and put back as it came out, it
/// would come and go with every move of the view.)
fn keep_out(p: Vec2, half: Vec2, (panel, top, view): Keep) -> Vec2 {
    let x = p.x.min(view.x - half.x - 4.0).max(half.x + 4.0);
    let mut p = Vec2::new(x, p.y.max(top + half.y + 2.0));
    if let Some((min, max)) = panel {
        let (lo, hi) = (min - half - Vec2::splat(6.0), max + half + Vec2::splat(6.0));
        if p.cmpgt(lo).all() && p.cmplt(hi).all() {
            let ways = [(p.x - lo.x, Vec2::new(lo.x, p.y)), (hi.x - p.x, Vec2::new(hi.x, p.y)), (p.y - lo.y, Vec2::new(p.x, lo.y)), (hi.y - p.y, Vec2::new(p.x, hi.y))];
            p = ways.iter().min_by(|a, b| a.0.total_cmp(&b.0)).map_or(p, |w| w.1);
        }
    }
    p
}

/// The eight corners of the box on `a`, `b`, `c` from 0.
fn box_corners(a: Vec3, b: Vec3, c: Vec3) -> [Vec3; 8] {
    [Vec3::ZERO, a, b, c, a + b, a + c, b + c, a + b + c]
}

/// The corners of the smallest convex shape round the points, in order (empty for
/// fewer than three).
fn hull(mut points: Vec<Vec2>) -> Vec<Vec2> {
    points.retain(|p| p.is_finite());
    points.sort_by(|p, q| p.x.total_cmp(&q.x).then(p.y.total_cmp(&q.y)));
    if points.len() < 3 {
        return Vec::new();
    }
    let turn = |o: Vec2, a: Vec2, b: Vec2| (a - o).perp_dot(b - o);
    let mut out: Vec<Vec2> = Vec::new();
    for pass in 0..2 {
        let start = out.len();
        let ordered: Vec<Vec2> = if pass == 0 { points.clone() } else { points.iter().rev().copied().collect() };
        for p in ordered {
            while out.len() >= start + 2 && turn(out[out.len() - 2], out[out.len() - 1], p) <= 0.0 {
                out.pop();
            }
            out.push(p);
        }
        out.pop();
    }
    out
}

/// The twelve edges of the box on `a`, `b`, `c` from 0, as (start, direction).
fn box_edges(a: Vec3, b: Vec3, c: Vec3) -> [(Vec3, Vec3); 12] {
    [(Vec3::ZERO, a), (b, a), (c, a), (b + c, a), (Vec3::ZERO, b), (a, b), (c, b), (a + c, b), (Vec3::ZERO, c), (a, c), (b, c), (a + b, c)]
}

/// The box spanned by `a`, `b`, `c` from the origin: see-through faces in `tone`, each
/// pair of opposite ones shaded by its own slant so that the shape reads as a solid,
/// drawn back to front, and its twelve edges. (As in the Gram–Schmidt scene.)
fn slanted_box(f: &mut Frame, a: Vec3, b: Vec3, c: Vec3, tone: Color, fill: f32) {
    let light = Vec3::new(0.55, -0.25, 0.8).normalize();
    let eye = f.cam.eye;
    let mut faces = [(Vec3::ZERO, a, b), (c, a, b), (Vec3::ZERO, a, c), (b, a, c), (Vec3::ZERO, b, c), (a, b, c)];
    let far = |(o, u, v): &(Vec3, Vec3, Vec3)| (*o + (*u + *v) * 0.5).distance(eye);
    faces.sort_by(|p, q| far(q).total_cmp(&far(p)));
    if fill > 0.0 {
        for (o, u, v) in faces {
            let shade = 0.35 + u.cross(v).normalize_or_zero().dot(light).abs().powf(1.5) * 0.65;
            f.dl.tris_alpha.quad(o, o + u, o + u + v, o + v, Color::from_linear(tone.to_linear() * shade).a(0.16 * fill));
        }
    }
    for (o, d) in box_edges(a, b, c) {
        f.line(o, o + d, 1.2, tone.a(0.7));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An experiment as its button sets it up, played to rest at 60 frames a second.
    fn run(k: usize) -> Det {
        let mut s = Det::create();
        s.set_up(k);
        for _ in 0..5000 {
            if !s.moving() {
                break;
            }
            s.advance(1.0 / 60.0);
        }
        assert!(!s.moving(), "experiment {k} is still moving");
        s
    }

    /// The same experiment, shown `stage` of the way through its steps.
    fn stopped(k: usize, stage: f32) -> Det {
        let mut s = run(k);
        (s.stage, s.goal) = (stage, stage);
        s
    }

    /// Columns of whole tenths, as the handles leave them, up to `reach` either way.
    fn tenths(rng: &mut Rng, n: usize, reach: f64) -> [DVec3; 3] {
        let mut c = [DVec3::X, DVec3::Y, DVec3::Z];
        let mut one = || tenth(rng.range(-reach as f32, reach as f32) as f64);
        for col in c.iter_mut().take(n) {
            *col = DVec3::new(one(), one(), if n == 3 { one() } else { 0.0 });
        }
        c
    }

    /// The pivots of elimination done on the rows, as the scene on elimination does it
    /// (None where one is 0 and a swap would be needed).
    fn row_pivots(c: &[DVec3; 3], n: usize) -> Option<Vec<f64>> {
        let mut a = [[0.0f64; 3]; 3];
        for (r, row) in a.iter_mut().enumerate().take(n) {
            for (k, v) in row.iter_mut().enumerate().take(n) {
                *v = c[k][r];
            }
        }
        let mut pivots = Vec::new();
        for r in 0..n {
            if a[r][r].abs() < 1e-9 {
                return None;
            }
            pivots.push(a[r][r]);
            for i in r + 1..n {
                let m = a[i][r] / a[r][r];
                for k in r..n {
                    a[i][k] -= m * a[r][k];
                }
            }
        }
        Some(pivots)
    }

    /// The determinant of columns of whole hundredths, exactly (in hundredths to the nth).
    fn exact_det(c: &[DVec3; 3], n: usize) -> i128 {
        let h = |v: f64| (v * 100.0).round() as i128;
        let m: Vec<Vec<i128>> = (0..3).map(|k| (0..3).map(|r| if k < n { h(c[k][r]) } else if r == k { 1 } else { 0 }).collect()).collect();
        let at = |k: usize, r: usize| m[k][r];
        at(0, 0) * (at(1, 1) * at(2, 2) - at(1, 2) * at(2, 1)) - at(1, 0) * (at(0, 1) * at(2, 2) - at(0, 2) * at(2, 1)) + at(2, 0) * (at(0, 1) * at(1, 2) - at(0, 2) * at(1, 1))
    }

    /// The numbers in a piece of a written sum, in order (brackets, true minus signs and
    /// the dollars round the formula read).
    fn numbers(text: &str) -> Vec<f64> {
        text.split([' ', '(', ')', ',', '$']).filter_map(|w| w.replace('−', "-").parse::<f64>().ok()).collect()
    }

    /// The captions of the "things to try", held to account.
    #[test]
    fn the_experiments_show_what_their_captions_say() {
        // slide a side: a₂ goes from leaning left to past a₁'s tip on the right, along
        // the line through its tip parallel to a₁; a₁, the height and the area 3.2 stay
        let s = run(0);
        let (start, end) = (s.cols(), s.shown());
        assert!(start[1].x < 0.0 && end[1].x > end[0].x, "{start:?} {end:?}");
        let height = det(&start).abs() / start[0].length();
        for i in 0..=60 {
            let c = stopped(0, i as f32 / 60.0).shown();
            assert_eq!(c[0], start[0]);
            assert!((det(&c) - 3.2).abs() < 1e-12, "{}", det(&c));
            assert!((c[1] - start[1]).truncate().perp_dot(start[0].truncate()).abs() < 1e-12, "a₂ left the line parallel to a₁");
            assert!((base_and_height(&c, 2, [0, 0]).map_or(0.0, |b| b.1) - height).abs() < 1e-12);
        }
        assert_eq!(measured(det(&end)), "3.2");
        assert!(s.heading().1.ends_with("so the area stays 3.2."), "{:?}", s.heading());
        assert!(TRIALS[0].1.contains("3.2 the whole way"));

        // swap the columns: on top of each other with area 0 halfway; after, each in
        // the other's place, the same parallelogram, turning clockwise, −2.7 for 2.7
        let s = run(1);
        let (start, end) = (s.cols(), s.shown());
        let half = stopped(1, 0.5).shown();
        assert!((half[0] - half[1]).length() < 1e-12 && det(&half).abs() < 1e-12 && is_flat(&half, 2), "{half:?}");
        assert!(end[0] == start[1] && end[1] == start[0]);
        let corners = |c: &[DVec3; 3]| {
            let mut v = [DVec3::ZERO, c[0], c[1], c[0] + c[1]].iter().map(|p| (p.x * 100.0).round() as i64 * 1000 + (p.y * 100.0).round() as i64).collect::<Vec<_>>();
            v.sort();
            v
        };
        assert_eq!(corners(&start), corners(&end), "the same parallelogram");
        assert!(end[0].truncate().perp_dot(end[1].truncate()) < 0.0, "the turn from a₁ to a₂ is clockwise");
        assert_eq!((measured(det(&start)), measured(det(&end))), ("2.7".to_string(), "−2.7".to_string()));
        assert!(s.tone() == MINUS && Det { stage: 0.0, ..run(1) }.tone() == PLUS);

        // slide it square: a₂ along a₁ until upright, then a₁ along a₂ until flat, the
        // area 2.5 throughout; a rectangle 2 by 1.25 along the axes, whose sides are
        // the pivots elimination (on the rows) finds, and 2 × 1.25 = 2.5 = ad − bc
        let s = run(2);
        let start = s.cols();
        assert!(matches!(s.ops[..], [Op::Slide { j: 1, i: 0, .. }, Op::Slide { j: 0, i: 1, .. }]), "{:?}", s.ops);
        let first = s.after(1);
        assert!(first[1].x == 0.0 && first[1].y > 0.0 && first[0] == start[0], "a₂ upright: {first:?}");
        let end = s.after(2);
        assert!(end[0].y == 0.0 && end[1] == first[1], "a₁ flat: {end:?}");
        assert_eq!(end, s.shown());
        for i in 0..=80 {
            assert!((det(&stopped(2, i as f32 / 40.0).shown()) - 2.5).abs() < 1e-12);
        }
        assert_eq!(s.end, Some(End::Brick { pivots: [2.0, 1.25, 1.0], swaps: 0 }));
        assert_eq!((end[0].x, end[1].y), (2.0, 1.25));
        assert_eq!(row_pivots(&start, 2), Some(vec![2.0, 1.25]));
        let (a, b, c, d) = (start[0].x, start[1].x, start[0].y, start[1].y);
        assert!((a * d - b * c - 2.5).abs() < 1e-12);
        let said = s.sums();
        assert_eq!(said, [r"$ad - bc = 2 \times 1.6 - 1 \times 0.7 = 3.2 - 0.7 = 2.5$", "pivots: $a = 2$ and $d - bc/a = 1.6 - 0.35 = 1.25$", r"$2 \times 1.25 = 2.5$, and $a(d - bc/a)$ is $ad - bc$"]);
        assert_eq!(s.heading().0, "The end: a rectangle along the axes");

        // columns in one line: a₂ = 2.5 a₁, no area; a₂ slides down to 0 and there is
        // no second pivot; and in general, 0 exactly when the columns are dependent
        let s = run(3);
        let start = s.cols();
        assert!(start[1] == start[0] * 2.5 && det(&start) == 0.0 && exact_det(&start, 2) == 0);
        assert_eq!(s.end, Some(End::Flat { row: 1, swaps: 0 }));
        assert!(matches!(s.ops[..], [Op::Slide { j: 1, i: 0, m, .. }] if m == -2.5));
        assert_eq!(s.shown()[1], DVec3::ZERO);
        assert!(s.heading().1.starts_with("$a_2$ is now $0$"));
        let mut rng = Rng::new(3);
        for _ in 0..500 {
            for n in [2, 3] {
                let c = tenths(&mut rng, n, 3.0);
                // made dependent: the last column from the others (tenths of tenths)
                let mut d = c;
                let (p, q) = ((rng.range(-2.0, 2.0) * 10.0).round() as f64 / 10.0, (rng.range(-2.0, 2.0) * 10.0).round() as f64 / 10.0);
                d[n - 1] = if n == 2 { d[0] * p } else { d[0] * p + d[1] * q };
                assert_eq!(exact_det(&d, n), 0);
                assert!(is_flat(&d, n) && matches!(square_ops(d, n).1, End::Flat { .. }), "{d:?}");
                // and the given ones are dependent exactly when their determinant is 0
                let flat = exact_det(&c, n) == 0;
                assert_eq!(is_flat(&c, n), flat, "{c:?}");
                assert_eq!(matches!(square_ops(c, n).1, End::Flat { .. }), flat, "{c:?}");
            }
        }

        // in space: the base on the floor, 2.5, times the height of a₃ over it, 1.6,
        // is 4; a₁ × a₂ stands upright, 2.5 long, and (a₁ × a₂) · a₃ = 4; sliding it
        // square makes a brick 2 by 1.25 by 1.6
        let s = run(4);
        let c = s.shown();
        assert!(s.space && s.ops.is_empty() && c[0].z == 0.0 && c[1].z == 0.0);
        let (area, height) = base_and_height(&c, 3, [0, 1]).expect("a base");
        assert!((area - 2.5).abs() < 1e-12 && (height - 1.6).abs() < 1e-12 && (det(&c) - 4.0).abs() < 1e-12);
        let cross = c[0].cross(c[1]);
        assert!(cross.x == 0.0 && cross.y == 0.0 && (cross.z - 2.5).abs() < 1e-12);
        assert_eq!(space_sum(&c), [r"$a_1 \times a_2 = (0, 0, 2.5)$", r"$(a_1 \times a_2) \cdot a_3 = 0 \times 0.6 + 0 \times 0.5 + 2.5 \times 1.6 = 4$"]);
        let (ops, end) = square_ops(c, 3);
        assert_eq!(end, End::Brick { pivots: [2.0, 1.25, 1.6], swaps: 0 });
        let brick = ops.iter().fold(c, |c, op| op.at(c, 1.0));
        assert_eq!(brick, [DVec3::new(2.0, 0.0, 0.0), DVec3::new(0.0, 1.25, 0.0), DVec3::new(0.0, 0.0, 1.6)]);
        assert!(TRIALS[4].1.contains(r"2.5 \times 1.6 = 4") && TRIALS[4].1.contains("2 by 1.25 by 1.6"));
    }

    /// The three rules, along the whole way of each step: a slide keeps the
    /// determinant, a stretch by k multiplies it by k, a swap flips it (passing through
    /// 0 halfway); and it adds up in each column.
    #[test]
    fn the_rules_hold_all_the_way() {
        let mut rng = Rng::new(7);
        for _ in 0..400 {
            for n in [2, 3] {
                let c = tenths(&mut rng, n, 3.0);
                let d = det(&c);
                let size = largest(&c, n).max(1.0).powi(n as i32);
                let (j, i) = (rng.below(n), rng.below(n));
                let m = rng.range(-3.0, 3.0) as f64;
                let k = rng.range(-3.0, 3.0) as f64;
                for step in 0..=20 {
                    let t = step as f64 / 20.0;
                    if i != j {
                        let slid = Op::Slide { j, i, m, clears: None }.at(c, t);
                        assert!((det(&slid) - d).abs() < 1e-12 * size * (1.0 + m.abs()), "slide {c:?}");
                        let swapped = Op::Swap { i, j }.at(c, t);
                        assert!((det(&swapped) - (1.0 - 2.0 * t) * d).abs() < 1e-12 * size, "swap {c:?} at {t}");
                    }
                    let stretched = Op::Stretch { j, k }.at(c, t);
                    assert!((det(&stretched) - (1.0 + (k - 1.0) * t) * d).abs() < 1e-12 * size * (1.0 + k.abs()), "stretch {c:?}");
                }
                // det(a, b + e) = det(a, b) + det(a, e), in the last column
                let e = tenths(&mut rng, n, 3.0)[0];
                let (mut with_b, mut with_e, mut with_sum) = (c, c, c);
                with_e[n - 1] = e;
                with_sum[n - 1] = c[n - 1] + e;
                with_b[n - 1] = c[n - 1];
                assert!((det(&with_sum) - det(&with_b) - det(&with_e)).abs() < 1e-12 * size);
            }
        }
    }

    /// "Slide it square", for any columns: the slides end in a brick along the axes
    /// whose sides are the pivots, the product of the pivots (with a sign per swap) is
    /// the determinant, the pivots are those of elimination on the rows whenever no
    /// swap is needed, and a first column with nothing on top is swapped out.
    #[test]
    fn sliding_square_finds_the_pivots() {
        let mut rng = Rng::new(11);
        let (mut swapped, mut flat, mut straight) = (0, 0, 0);
        for trial in 0..3000 {
            let n = 2 + trial % 2;
            let mut c = tenths(&mut rng, n, 2.0);
            // (zeros, often, so that swaps and missing pivots come up)
            for col in c.iter_mut().take(n) {
                for r in 0..n {
                    if rng.bool(0.25) {
                        col[r] = 0.0;
                    }
                }
            }
            let (ops, end) = square_ops(c, n);
            let after = ops.iter().fold(c, |c, op| op.at(c, 1.0));
            let size = largest(&c, n).max(1e-9);
            // no step changes the size of the determinant
            let mut now = c;
            for op in &ops {
                let next = op.at(now, 1.0);
                assert!((det(&next).abs() - det(&now).abs()).abs() < 1e-9 * size.powi(n as i32), "{op:?} on {now:?}");
                now = next;
            }
            match end {
                End::Brick { pivots, swaps } => {
                    for k in 0..n {
                        for r in 0..n {
                            if k != r {
                                assert!(after[k][r].abs() <= 1e-9 * size, "{c:?}: off the axes {after:?}");
                            }
                        }
                        assert_eq!(after[k][k], pivots[k]);
                    }
                    let product: f64 = pivots[..n].iter().product();
                    let sign = if swaps % 2 == 0 { 1.0 } else { -1.0 };
                    assert!((sign * product - det(&c)).abs() < 1e-9 * size.powi(n as i32), "{c:?}");
                    assert!(exact_det(&c, n) != 0);
                    assert_eq!(swaps, ops.iter().filter(|op| matches!(op, Op::Swap { .. })).count());
                    if swaps == 0 {
                        straight += 1;
                        let rows = row_pivots(&c, n).expect("no swap is needed on the rows either");
                        for k in 0..n {
                            assert!((rows[k] - pivots[k]).abs() < 1e-9 * size, "{c:?}: {rows:?} against {pivots:?}");
                        }
                    } else {
                        swapped += 1;
                    }
                    if c[0][0] == 0.0 {
                        assert!(matches!(ops.first(), Some(Op::Swap { i: 0, .. })), "{c:?}");
                    }
                }
                End::Flat { .. } => {
                    flat += 1;
                    assert_eq!(exact_det(&c, n), 0, "{c:?}");
                }
            }
        }
        assert!(swapped > 100 && flat > 100 && straight > 1000, "{swapped} with swaps, {flat} flat, {straight} without swaps");

        // a first pivot of 0: a₁ = (0, 1.5), a₂ = (2, 0.5): one swap, pivots 2 and 1.5,
        // det = −3, and the panel says so
        let mut s = Det::create();
        s.start[0] = plane_cols([[0.0, 1.5], [2.0, 0.5]]);
        s.slide_square();
        assert!(matches!(s.ops[0], Op::Swap { i: 0, j: 1 }));
        assert_eq!(s.end, Some(End::Brick { pivots: [2.0, 1.5, 1.0], swaps: 1 }));
        (s.stage, s.goal) = (s.ops.len() as f32, s.ops.len() as f32);
        assert_eq!(s.sums()[1..], [r"pivots, and one swap: $-(2 \times 1.5) = −3$", r"with $a = 0$, $ad - bc$ is just $-bc = -2 \times 1.5$"]);
        (s.stage, s.goal) = (1.0, 1.0);
        assert!(s.heading().1.starts_with("$a_1$ has no $x$ part to give a pivot, so it trades places with $a_2$. A swap flips the sign: −3 becomes 3."), "{:?}", s.heading());
        // nothing at all: no pivot in the top row
        s.start[0] = plane_cols([[0.0, 0.0], [0.0, 0.0]]);
        s.slide_square();
        assert!(s.ops.is_empty() && s.end == Some(End::Flat { row: 0, swaps: 0 }));
    }

    /// Every sum written over the picture adds up as written: the numbers in it are
    /// the matrix's, the products and the total are what those numbers make, and "="
    /// stands only where that is exact.
    #[test]
    fn the_sums_on_screen_add_up_as_written() {
        let mut rng = Rng::new(5);
        for trial in 0..2000 {
            let n = 2 + trial % 2;
            let mut c = tenths(&mut rng, n, 3.0);
            // and some in hundredths, as the white corner leaves them
            if trial % 3 == 0 {
                c[n - 1] += c[0] * ((rng.range(-2.0, 2.0) * 10.0).round() as f64 / 10.0);
            }
            if n == 2 {
                let line = plane_sum(&c);
                let parts: Vec<&str> = line.split(" = ").collect();
                assert_eq!(parts.len(), 4, "{line}");
                let (factors, products, total) = (numbers(parts[1]), numbers(parts[2]), numbers(parts[3]));
                assert_eq!(factors, [c[0].x, c[1].y, c[1].x, c[0].y].map(|v| (v * 1e4).round() / 1e4), "{line}");
                assert!((factors[0] * factors[1] - products[0]).abs() < 1e-9 && (factors[2] * factors[3] - products[1]).abs() < 1e-9, "{line}");
                assert!((products[0] - products[1] - total[0]).abs() < 1e-9 && (total[0] - det(&c)).abs() < 1e-9, "{line}");
            } else {
                let [cross, dot] = space_sum(&c);
                let x = c[0].cross(c[1]);
                assert_eq!(numbers(cross.split(" = ").nth(1).unwrap_or("")), [x.x, x.y, x.z].map(|v| (v * 1e4).round() / 1e4), "{cross}");
                let parts: Vec<&str> = dot.split(" = ").collect();
                let factors = numbers(parts[1]);
                let total = numbers(parts[2]);
                assert_eq!(factors.len(), 6, "{dot}");
                let sum = factors[0] * factors[1] + factors[2] * factors[3] + factors[4] * factors[5];
                assert!((sum - total[0]).abs() < 1e-9 && (total[0] - det(&c)).abs() < 1e-9, "{dot}");
            }
        }
        // a step half done is written "≈", to two decimals
        let s = stopped(0, 0.37);
        let line = plane_sum(&s.shown());
        assert!(line.contains(r" \approx ") && !line.contains(" = ") && line.ends_with(r"\approx 3.2$"), "{line}");
        // numbers on the screen: no "−0", a true minus sign, no digits they do not need
        assert_eq!([written(-0.0, 3), written(-1.5, 3), written(1.0 / 3.0, 3), short(2.0 / 3.0), short(-1.256), measured(2.5), measured(1.0 / 3.0)], ["0", "−1.5", "0.333…", "0.67", "−1.26", "2.5", "0.33"]);
        assert_eq!([op_name(Op::Slide { j: 1, i: 0, m: -0.5, clears: None }), op_name(Op::Slide { j: 2, i: 0, m: 1.0, clears: None }), op_name(Op::Stretch { j: 1, k: -1.0 }), op_name(Op::Slide { j: 0, i: 1, m: -0.5 / 1.4, clears: None })], ["$a_2 - 0.5 a_1$", "$a_3 + a_1$", r"$a_2 \times (−1)$", "$a_1 - 0.357… a_2$"]);
    }

    /// The white corner moves the last column by tenths of the base's sides only: the
    /// determinant stays as it was wherever it is dragged, and the dot keeps to its
    /// line in the plane (its plane in space).
    #[test]
    fn the_white_corner_slides_without_changing_the_determinant() {
        let mut rng = Rng::new(9);
        for trial in 0..1000 {
            let n = 2 + trial % 2;
            let c = tenths(&mut rng, n, 3.0);
            if is_flat(&c, n) || (n == 2 && c[0].length() < 1e-6) || (n == 3 && c[0].cross(c[1]).length() < 1e-6) {
                continue;
            }
            let from = c[n - 1];
            let mut grip = DVec3::new(rng.normal() as f64, rng.normal() as f64, if n == 3 { rng.normal() as f64 } else { 0.0 }) * 3.0;
            if n == 2 {
                grip.z = 0.0;
            }
            let (column, on) = slid(&c, n, from, grip).expect("a base");
            let mut moved = c;
            moved[n - 1] = column;
            let size = largest(&moved, n).max(1.0).powi(n as i32);
            assert!((det(&moved) - det(&c)).abs() < 1e-9 * size, "{c:?} to {moved:?}");
            // the dot: on the line through the corner parallel to a₁, or the plane
            // through it parallel to the base
            let corner: DVec3 = c[..n].iter().copied().sum();
            if n == 2 {
                assert!((on - corner).truncate().perp_dot(c[0].truncate()).abs() < 1e-9 * (1.0 + grip.length()) * c[0].length());
            } else {
                assert!((on - corner).dot(c[0].cross(c[1]).normalize()).abs() < 1e-9 * (1.0 + grip.length()));
            }
        }
    }

    /// The steps play forward and back, a pause after each when played through, end
    /// exactly where the last one leaves the columns, and then the picture stops.
    #[test]
    fn the_steps_play_and_come_to_rest() {
        let mut s = Det::create();
        s.set_up(2);
        assert!(s.moving() && s.stage == 0.0 && s.ops.len() == 2);
        // at 60 and at 144 frames a second, the same end
        for hz in [60.0f32, 144.0] {
            s.replay();
            let mut frames = 0;
            while s.moving() {
                s.advance(1.0 / hz);
                frames += 1;
                assert!(frames < 10_000);
            }
            let seconds = frames as f32 / hz;
            assert!((seconds - (HOLD * 0.6 + 2.0 * STEP_TIME + HOLD)).abs() < 2.0 / hz, "{seconds} s at {hz} Hz");
            assert!(s.stage == 2.0 && s.shown() == s.after(2) && s.ended());
        }
        // back one step, and next again
        s.back();
        while s.moving() {
            s.advance(1.0 / 60.0);
        }
        assert!(s.stage == 1.0 && s.shown() == s.after(1) && !s.ended() && s.can_go_on());
        s.next();
        while s.moving() {
            s.advance(1.0 / 60.0);
        }
        assert!(s.stage == 2.0 && !s.can_go_on());
        // dragging (or typing) from the middle of a step starts afresh from there
        (s.stage, s.goal) = (1.5, 1.5);
        let now = s.shown();
        s.commit();
        assert!(s.ops.is_empty() && s.end.is_none() && s.cols() == now && s.stage == 0.0 && !s.moving());
        // the arrows glide to a new place, within a second, and then stop exactly there
        // (after an experiment's slower glide, too)
        for pace in [GLIDE, SETTLE] {
            s.glide(0.0, true);
            s.pace = pace;
            s.set_cols(plane_cols([[1.0, 0.0], [0.0, 1.0 + pace as f64]]));
            assert!(!s.arrived());
            let mut frames = 0;
            while !s.arrived() {
                s.glide(1.0 / 60.0, false);
                frames += 1;
                assert!(frames < 60, "still gliding after a second at {pace}");
            }
            assert!(s.pace == GLIDE && s.drawn[1] == Vec3::new(0.0, 1.0 + pace, 0.0));
        }
    }

    /// Tips move in tenths, no further out than they may go.
    #[test]
    fn dragged_tips_move_in_tenths() {
        assert_eq!(snap(Vec3::new(1.234, -0.66, 0.4), 4.0, true), DVec3::new(1.2, -0.7, 0.0));
        assert_eq!(snap(Vec3::new(1.234, -0.66, 0.44), 3.0, false), DVec3::new(1.2, -0.7, 0.4));
        let far = snap(Vec3::new(30.0, 40.0, 0.0), 4.0, true);
        assert!((far - DVec3::new(2.4, 3.2, 0.0)).length() < 1e-12, "{far}");
        assert_eq!(snap(Vec3::new(f32::NAN, 1.0, 0.0), 4.0, true), DVec3::ZERO);
    }

    /// Every line the panel over the picture writes fits it, on a phone too, for every
    /// experiment, at every step and at the end.
    #[test]
    fn the_words_fit_the_panel() {
        let narrowest = [PANEL_W[0].min(390.0 - 16.0) - 24.0, PANEL_W[1].min(390.0 - 16.0) - 24.0];
        let mut checked = 0;
        let mut look = |s: &Det| {
            let width = narrowest[s.space as usize];
            let (head, says) = s.heading();
            assert!(text_width(&head, 13.0) <= width, "{head}");
            assert!(wrap(&says, width, SAYS_SIZE).len() <= SAYS_LINES, "{says}");
            let sums = sum_lines(&s.sums(), width);
            assert!(sums.len() <= SUM_LINES[s.space as usize], "{sums:?}");
            checked += 1;
        };
        for k in 0..TRIALS.len() {
            let s = run(k);
            for i in 0..=4 * s.ops.len() {
                let mut t = run(k);
                (t.stage, t.goal) = (i as f32 / 4.0, i as f32 / 4.0);
                look(&t);
            }
            let mut t = run(k);
            t.slide_square();
            for i in 0..=t.ops.len() {
                (t.stage, t.goal) = (i as f32, i as f32);
                look(&t);
            }
            for tool in [Op::Swap { i: 0, j: s.n() - 1 }, Op::Stretch { j: s.n() - 1, k: 2.0 }, Op::Stretch { j: s.n() - 1, k: -1.0 }, Op::Slide { j: s.n() - 1, i: 0, m: 1.0, clears: None }] {
                let mut t = run(k);
                t.play(vec![tool], None);
                for stage in [0.0, 0.5, 1.0] {
                    (t.stage, t.goal) = (stage, stage);
                    look(&t);
                }
            }
        }
        assert!(checked > 60, "{checked}");
    }

    /// The panel keeps to every canvas, below the application's buttons, and the
    /// picture's part of the canvas keeps clear of it.
    #[test]
    fn the_panel_fits_every_canvas() {
        let canvases = [(1080.0, 1000.0, 0.0), (1600.0, 1000.0, 46.0), (390.0, 800.0, 46.0), (360.0, 640.0, 46.0), (600.0, 900.0, 46.0), (844.0, 390.0, 46.0), (640.0, 480.0, 46.0), (240.0, 180.0, 46.0), (100.0, 800.0, 46.0)];
        for (x, y, inset) in canvases {
            let view = Vec2::new(x, y);
            for rows in [2, 3] {
                let lay = Layout::of(view, inset, rows);
                if let Some((min, max)) = lay.panel {
                    assert!(min.x >= 0.0 && min.y >= inset && max.x <= x && max.y <= y, "{view}: {lay:?}");
                    assert!(max.y - min.y >= panel_height(rows) - 0.01);
                    // the buttons fit across (they shrink to the panel's width, not below
                    // nine tenths of their size)
                    assert!(max.x - min.x - 24.0 >= 0.9 * (316.0 + 18.0), "{view}: {lay:?}");
                }
                let mut s = Det::create();
                s.space = rows == 3;
                (s.canvas, s.inset) = (view, inset);
                let ((lo, hi), per_unit) = s.region(Vec2::new(4.0, 3.0));
                assert!(hi.cmpgt(lo).all() && per_unit > 0.0 && per_unit.is_finite(), "{view}");
                if let Some((min, max)) = lay.panel {
                    assert!(hi.x <= min.x + 0.01 || max.x <= lo.x + 0.01 || hi.y <= min.y + 0.01 || max.y <= lo.y + 0.01, "{view}: the picture's part {lo}..{hi} runs under the panel {min}..{max}");
                }
                match s.the_view() {
                    View::Fit { center, half } => assert!(center.is_finite() && half.is_finite() && half.min_element() > 0.0),
                    View::Orbit { target, dist, .. } => assert!(target.is_finite() && dist.is_finite() && dist > 0.0),
                    View::Flat { .. } => panic!("a fitted view"),
                }
            }
            if x >= 390.0 && y >= 640.0 {
                assert!(Layout::of(view, inset, 2).panel.is_some() && Layout::of(view, inset, 3).panel.is_some(), "{view}");
            }
        }
    }

    /// Zero columns, equal ones, opposite ones, all in one plane, tiny and large: every
    /// number drawn or written stays finite, and the panel says what has happened.
    #[test]
    fn degenerate_columns_leave_everything_finite() {
        let plane = [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.5]], [[1.0, 0.5], [0.0, 0.0]], [[1.2, 0.4], [1.2, 0.4]], [[1.2, 0.4], [-1.2, -0.4]], [[0.0, 1.0], [0.0, -2.0]], [[0.0, 1.5], [2.0, 0.5]], [[0.01, 0.0], [0.0, 0.01]], [[10.0, -10.0], [10.0, 10.0]]];
        let space = [
            [[0.0; 3]; 3],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            [[1.0, 0.5, 0.2], [1.0, 0.5, 0.2], [0.0, 0.0, 1.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.5, 0.1, 0.0]],
            [[10.0, 0.0, 0.0], [0.0, -10.0, 0.0], [0.0, 0.0, 10.0]],
        ];
        let cases: Vec<(bool, [DVec3; 3])> = plane.iter().map(|c| (false, plane_cols(*c))).chain(space.iter().map(|c| (true, space_cols(*c)))).collect();
        for (in_space, c) in cases {
            let mut s = Det::create();
            s.space = in_space;
            s.start[in_space as usize] = c;
            let n = s.n();
            for play in 0..6 {
                match play {
                    0 => {}
                    1 => s.slide_square(),
                    2 => s.play(vec![Op::Swap { i: 0, j: n - 1 }], None),
                    3 => s.play(vec![Op::Stretch { j: n - 1, k: -1.0 }], None),
                    4 => s.play(vec![Op::Slide { j: n - 1, i: 0, m: 1.0, clears: None }], None),
                    _ => s.start[in_space as usize] = c,
                }
                for i in 0..=4 * s.ops.len() {
                    (s.stage, s.goal) = (i as f32 / 4.0, i as f32 / 4.0);
                    let shown = s.shown();
                    assert!(shown.iter().all(|v| v.is_finite()), "{c:?}");
                    s.glide(0.0, true);
                    assert!(s.drawn.iter().all(|v| v.is_finite()) && s.corner().is_finite());
                    let (head, says) = s.heading();
                    let text = [head, says, s.sums().join(" ")].join(" ");
                    assert!(!text.contains("NaN") && !text.contains("inf") && !text.contains('?'), "{c:?}: {text}");
                    if let Some((_, base, _)) = s.roles() {
                        if let Some((b, h)) = base_and_height(&shown, n, base) {
                            assert!(b.is_finite() && h.is_finite() && b > 0.0);
                        }
                    }
                    for canvas in [Vec2::new(1080.0, 1000.0), Vec2::new(390.0, 800.0), Vec2::new(240.0, 180.0)] {
                        s.canvas = canvas;
                        match s.the_view() {
                            View::Fit { center, half } => assert!(center.is_finite() && half.is_finite() && half.min_element() > 0.0, "{c:?}"),
                            View::Orbit { target, dist, .. } => assert!(target.is_finite() && dist.is_finite() && dist > 0.0, "{c:?}"),
                            View::Flat { .. } => panic!("a fitted view"),
                        }
                    }
                }
                (s.stage, s.goal) = (s.ops.len() as f32, s.ops.len() as f32);
            }
            // a flat set says so, and a white corner over no base is not offered
            if is_flat(&c, n) {
                s.start[in_space as usize] = c;
                (s.ops, s.end) = (Vec::new(), None);
                s.slide_square();
                (s.stage, s.goal) = (s.ops.len() as f32, s.ops.len() as f32);
                assert!(s.heading().0 == r"No pivot: $\det A = 0$", "{c:?}: {:?}", s.heading());
            }
            if n == 2 && c[0].length() == 0.0 {
                assert!(slid(&c, 2, c[1], DVec3::new(1.0, 1.0, 0.0)).is_none());
            }
        }
    }

    /// The numbers behind the captions (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        for k in 0..TRIALS.len() {
            let s = run(k);
            println!("{}: det {} -> {}  ops {:?}  end {:?}", TRIALS[k].0, det(&s.cols()), det(&s.shown()), s.ops, s.end);
            println!("    heading {:?}", s.heading());
            println!("    sums {:?}", s.sums());
            if let Some(bh) = base_and_height(&s.shown(), s.n(), [0, 1]) {
                println!("    base and height {bh:?}");
            }
        }
    }
}

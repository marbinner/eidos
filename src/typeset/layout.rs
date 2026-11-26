//! Formulas laid out as TeX lays them out (The TeXbook, appendix G, by way of KaTeX):
//! boxes with a width, a height above the baseline and a depth below it, holding
//! glyphs, rules and drawn strokes. Lengths are in ems of the formula's size, and y
//! is upwards from the baseline.

use super::parse::{Delim, Node};
use super::symbols::{self, Class, Face, Metrics};
use egui::{vec2, Color32, Vec2};

/// Something drawn. A colour of `None` is the colour of the text the formula is in.
#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    /// A glyph, `size` times the formula's size, its baseline's left end at `at`.
    Glyph { face: Face, ch: char, size: f32, at: Vec2, color: Option<Color32> },
    /// A filled rectangle: fraction bars, the bar of a root, the lines of a bracket.
    Rule { min: Vec2, max: Vec2, color: Option<Color32> },
    /// A stroke along a line of points, with its width at each: curved brackets and
    /// the sign of a root, drawn at whatever height they must have.
    Stroke { points: Vec<(Vec2, f32)>, color: Option<Color32> },
    /// Words (`\text`) in the font of the text round the formula, `size` times that
    /// text's size (less in a script), their baseline's left end at `at`; in bold, drawn
    /// twice side by side, a hair apart ([`embolden`]).
    Words { text: String, size: f32, at: Vec2, color: Option<Color32>, bold: bool },
}

/// How far apart the two impressions of bold words are, in ems of their font: the
/// interface's font has no bold face, so bold is made as a browser makes it when a font
/// lacks one.
pub fn embolden(size: f32) -> f32 {
    size * 0.05
}

/// What the layout knows of words set in the interface's font (`\text`), which it
/// cannot measure itself: their width and how far their ink reaches above and below
/// the baseline, in ems of the formula, set at the size of the text round it (and,
/// given `true`, in bold).
pub type Measure<'a> = &'a dyn Fn(&str, bool) -> [f32; 3];

/// A [`Measure`] for when there are no fonts at hand (an estimate, a unit test): each
/// character `per_char` ems of the text wide, the ink as tall as a capital and as deep
/// as a descender. `scale` is how many of the text's ems the formula's em is
/// (`TEXT_SCALE`).
pub fn estimate(per_char: f32, scale: f32) -> impl Fn(&str, bool) -> [f32; 3] {
    move |text: &str, _| [text.chars().count() as f32 * per_char / scale, 0.7 / scale, 0.2 / scale]
}

impl Item {
    fn moved(mut self, by: Vec2) -> Item {
        match &mut self {
            Item::Glyph { at, .. } => *at += by,
            Item::Rule { min, max, .. } => {
                *min += by;
                *max += by;
            }
            Item::Stroke { points, .. } => points.iter_mut().for_each(|(p, _)| *p += by),
            Item::Words { at, .. } => *at += by,
        }
        self
    }
}

/// A laid-out piece of a formula.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Bx {
    pub width: f32,
    pub height: f32,
    pub depth: f32,
    pub items: Vec<Item>,
    /// For a single character: its italic correction (part of `width`), and where over
    /// it an accent is centred. Scripts and accents are placed by these.
    italic: f32,
    accent_at: Option<f32>,
}

impl Bx {
    fn empty(width: f32) -> Bx {
        Bx { width, ..Bx::default() }
    }

    /// Put `child` into this box with its origin at (x, y); the width is the caller's.
    fn place(&mut self, child: Bx, x: f32, y: f32) {
        self.height = self.height.max(child.height + y);
        self.depth = self.depth.max(child.depth - y);
        let by = vec2(x, y);
        self.items.extend(child.items.into_iter().map(|item| item.moved(by)));
    }

    fn rule(&mut self, min: Vec2, max: Vec2, color: Option<Color32>) {
        self.height = self.height.max(max.y);
        self.depth = self.depth.max(-min.y);
        self.items.push(Item::Rule { min, max, color });
    }

    fn stroke(&mut self, points: Vec<(Vec2, f32)>, color: Option<Color32>) {
        for (p, w) in &points {
            self.height = self.height.max(p.y + w * 0.5);
            self.depth = self.depth.max(-p.y + w * 0.5);
        }
        self.items.push(Item::Stroke { points, color });
    }
}

/// TeX's styles: display, text, script and scriptscript, each cramped or not (a
/// cramped style raises superscripts less: under a bar, in a denominator).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
    level: u8,
    cramped: bool,
}

impl Style {
    pub const DISPLAY: Style = Style { level: 0, cramped: false };
    pub const TEXT: Style = Style { level: 1, cramped: false };

    fn of(level: u8, cramped: bool) -> Style {
        Style { level: level.min(3), cramped }
    }

    /// The size, relative to the formula's.
    fn size(self) -> f32 {
        [1.0, 1.0, 0.7, 0.5][self.level as usize]
    }

    /// Which column of the font parameters applies.
    fn index(self) -> usize {
        [0, 0, 1, 2][self.level as usize]
    }

    fn script(self) -> bool {
        self.level >= 2
    }

    fn sup(self) -> Style {
        Style::of(if self.level < 2 { 2 } else { 3 }, self.cramped)
    }

    fn sub(self) -> Style {
        Style::of(if self.level < 2 { 2 } else { 3 }, true)
    }

    fn num(self) -> Style {
        Style::of([1, 2, 3, 3][self.level as usize], self.cramped)
    }

    fn den(self) -> Style {
        Style::of([1, 2, 3, 3][self.level as usize], true)
    }

    fn cramp(self) -> Style {
        Style::of(self.level, true)
    }

    /// A parameter of the fonts, for this style, in ems of the formula.
    fn p(self, table: [f32; 3]) -> f32 {
        table[self.index()] * self.size()
    }
}

// The fonts' parameters, for text, script and scriptscript sizes, each in ems of
// its own size (TeX's sigma and xi, as KaTeX has them).
const X_HEIGHT: [f32; 3] = [0.431, 0.431, 0.431];
const QUAD: [f32; 3] = [1.0, 1.171, 1.472];
const NUM1: [f32; 3] = [0.677, 0.732, 0.925];
const NUM2: [f32; 3] = [0.394, 0.384, 0.387];
const NUM3: [f32; 3] = [0.444, 0.471, 0.504];
const DENOM1: [f32; 3] = [0.686, 0.752, 1.025];
const DENOM2: [f32; 3] = [0.345, 0.344, 0.532];
const SUP1: [f32; 3] = [0.413, 0.503, 0.504];
const SUP2: [f32; 3] = [0.363, 0.431, 0.404];
const SUP3: [f32; 3] = [0.289, 0.286, 0.294];
const SUB1: [f32; 3] = [0.15, 0.143, 0.2];
const SUB2: [f32; 3] = [0.247, 0.286, 0.4];
const SUP_DROP: [f32; 3] = [0.386, 0.353, 0.494];
const SUB_DROP: [f32; 3] = [0.05, 0.071, 0.1];
const DELIM1: [f32; 3] = [2.39, 1.7, 1.98];
const DELIM2: [f32; 3] = [1.01, 1.157, 1.42];
const AXIS: [f32; 3] = [0.25, 0.25, 0.25];
const RULE: [f32; 3] = [0.04, 0.049, 0.049];
const BIG_OP: [[f32; 3]; 5] = [[0.111, 0.111, 0.111], [0.166, 0.166, 0.166], [0.2, 0.2, 0.2], [0.6, 0.611, 0.611], [0.1, 0.143, 0.143]];
/// The space either side of a fraction (TeX's \nulldelimiterspace).
const NULL_DELIMITER: f32 = 0.12;
/// The space after scripts (TeX's \scriptspace).
const SCRIPT_SPACE: f32 = 0.05;
/// The width of a space in `\text`: Main's own.
const WORD_SPACE: f32 = 0.25;

/// What the boxes of a list are laid out with: the style, a colour set by `\color`,
/// and how to measure words.
#[derive(Clone, Copy)]
struct Cx<'a> {
    style: Style,
    color: Option<Color32>,
    measure: Measure<'a>,
}

impl<'a> Cx<'a> {
    fn with(self, style: Style) -> Cx<'a> {
        Cx { style, ..self }
    }
}

/// A piece of a formula's top level, up to a place where a line may be broken, and the
/// space after it if the line goes on.
#[derive(Debug)]
pub struct Chunk {
    pub bx: Bx,
    pub glue: f32,
    pub breakable: bool,
    /// What kind of place the break after it is (`Break::None` if there is none).
    pub kind: Break,
}

/// The places a formula's line may be broken, in the order a display prefers them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Break {
    None,
    /// At a binary operator.
    Operator,
    /// At a relation.
    Relation,
    /// Between two formulas set side by side: after a comma and a quad or more.
    Between,
}

/// Lay out a formula, in display style or in the style of running text, with its words
/// (`\text`) measured by `measure`. Its top level is cut into chunks where a line may
/// be broken: in running text after a relation or a binary operator, as TeX allows; in a
/// display before one, the relation or operator carried to the new line, as a display is
/// broken in print; and in both after a comma followed by a quad or more.
pub fn layout(nodes: &[Node], display: bool, measure: Measure) -> Vec<Chunk> {
    let cx = Cx { style: if display { Style::DISPLAY } else { Style::TEXT }, color: None, measure };
    let atoms = spaced(nodes, cx);
    let quad = cx.style.p(QUAD);
    let mut chunks = Vec::new();
    let mut bx = Bx::default();
    let mut x = 0.0;
    for (i, atom) in atoms.iter().enumerate() {
        let before = if i == 0 { None } else { Some(atoms[i - 1].class) };
        let kind = if before.is_none() || atom.bx.width < 0.0 {
            Break::None
        } else if before == Some(Class::Punct) && atom.before >= 0.9 * quad {
            Break::Between
        } else if display {
            match (before, atom.class) {
                (Some(Class::Rel | Class::Open | Class::Punct | Class::Bin), _) => Break::None,
                (_, Class::Rel) => Break::Relation,
                (_, Class::Bin) => Break::Operator,
                _ => Break::None,
            }
        } else if matches!(atom.class, Class::Rel | Class::Punct | Class::Close) {
            Break::None
        } else {
            match before {
                Some(Class::Rel) => Break::Relation,
                Some(Class::Bin) => Break::Operator,
                _ => Break::None,
            }
        };
        if kind != Break::None && !bx.items.is_empty() {
            bx.width = x;
            chunks.push(Chunk { bx: std::mem::take(&mut bx), glue: atom.before, breakable: true, kind });
            x = 0.0;
        } else {
            x += atom.before;
        }
        bx.place(atom.bx.clone(), x, 0.0);
        x += atom.bx.width;
    }
    bx.width = x + atoms.last().map_or(0.0, |a| a.after);
    chunks.push(Chunk { bx, glue: 0.0, breakable: false, kind: Break::None });
    chunks
}

/// An atom in a list, with the space before it (and, for the last, any space after).
struct Atom {
    class: Class,
    bx: Bx,
    before: f32,
    after: f32,
}

/// The atoms of a list, with TeX's spaces between them.
fn spaced(nodes: &[Node], cx: Cx<'_>) -> Vec<Atom> {
    let mut atoms = Vec::new();
    let mut space = 0.0;
    collect(nodes, cx, &mut atoms, &mut space);
    if let Some(last) = atoms.last_mut() {
        last.after += space;
    }
    // A binary operator with nothing to operate on is an ordinary symbol (a minus at the
    // start, or after a relation or an opening bracket, takes no space).
    for i in 0..atoms.len() {
        let before = if i == 0 { None } else { Some(atoms[i - 1].class) };
        if atoms[i].class == Class::Bin && matches!(before, None | Some(Class::Bin | Class::Op | Class::Rel | Class::Open | Class::Punct)) {
            atoms[i].class = Class::Ord;
        }
        if i > 0 && atoms[i - 1].class == Class::Bin && matches!(atoms[i].class, Class::Rel | Class::Close | Class::Punct) {
            atoms[i - 1].class = Class::Ord;
        }
    }
    if let Some(last) = atoms.last_mut() {
        if last.class == Class::Bin {
            last.class = Class::Ord;
        }
    }
    let mu = cx.style.p(QUAD) / 18.0;
    for i in 1..atoms.len() {
        let gap = between(atoms[i - 1].class, atoms[i].class, cx.style.script());
        atoms[i].before += gap * mu;
    }
    atoms
}

/// Gather the atoms of a list, flattening colours and styles (which do not change the
/// kind of atom anything is), with explicit spaces added to the next atom's.
fn collect(nodes: &[Node], cx: Cx<'_>, atoms: &mut Vec<Atom>, space: &mut f32) {
    for node in nodes {
        match node {
            Node::Color { color, body } => collect(body, Cx { color: Some(*color), ..cx }, atoms, space),
            Node::Style { level, body } => collect(body, cx.with(Style::of(*level, cx.style.cramped)), atoms, space),
            Node::Space(mu) => *space += mu * cx.style.p(QUAD) / 18.0,
            _ => {
                let (class, bx) = atom(node, cx);
                atoms.push(Atom { class, bx, before: std::mem::take(space), after: 0.0 });
            }
        }
    }
}

/// TeX's space between atoms, in eighteenths of an em: thin (3), medium (4), thick (5);
/// in scripts only the thin space between an operator and its neighbours.
fn between(left: Class, right: Class, script: bool) -> f32 {
    use Class::*;
    // 1: thin; 2: thin, not in scripts; 3: medium, not in scripts; 4: thick, not in scripts
    const TABLE: [[u8; 8]; 8] = [
        // Ord Op Bin Rel Open Close Punct Inner
        [0, 1, 3, 4, 0, 0, 0, 2], // Ord
        [1, 1, 0, 4, 0, 0, 0, 2], // Op
        [3, 3, 0, 0, 3, 0, 0, 3], // Bin
        [4, 4, 0, 0, 4, 0, 0, 4], // Rel
        [0, 0, 0, 0, 0, 0, 0, 0], // Open
        [0, 1, 3, 4, 0, 0, 0, 2], // Close
        [2, 2, 0, 2, 2, 2, 2, 2], // Punct
        [2, 1, 3, 4, 2, 0, 2, 2], // Inner
    ];
    let index = |c: Class| match c {
        Ord => 0,
        Op => 1,
        Bin => 2,
        Rel => 3,
        Open => 4,
        Close => 5,
        Punct => 6,
        Inner => 7,
    };
    match TABLE[index(left)][index(right)] {
        1 => 3.0,
        2 if !script => 3.0,
        3 if !script => 4.0,
        4 if !script => 5.0,
        _ => 0.0,
    }
}

/// A list laid out side by side.
fn hlist(nodes: &[Node], cx: Cx<'_>) -> Bx {
    let atoms = spaced(nodes, cx);
    let mut bx = Bx::default();
    let mut x = 0.0;
    for atom in &atoms {
        x += atom.before;
        bx.place(atom.bx.clone(), x, 0.0);
        x += atom.bx.width;
    }
    bx.width = x + atoms.last().map_or(0.0, |a| a.after);
    // (a list of one character keeps what scripts and accents need to know of it)
    if let [only] = atoms.as_slice() {
        if only.before == 0.0 && only.after == 0.0 {
            bx.italic = only.bx.italic;
            bx.accent_at = only.bx.accent_at;
        }
    }
    bx
}

fn glyph_metrics(face: Face, ch: char) -> Metrics {
    symbols::metrics(face, ch).unwrap_or(Metrics { width: 0.5, height: 0.7, depth: 0.0, italic: 0.0, skew: 0.0 })
}

/// One glyph, `size` times the formula's size. An italic letter's box takes in its
/// italic correction, as in TeX.
fn glyph(face: Face, ch: char, size: f32, color: Option<Color32>) -> Bx {
    let m = glyph_metrics(face, ch);
    let italic = if face == Face::MathItalic { m.italic * size } else { 0.0 };
    Bx {
        width: m.width * size + italic,
        height: m.height * size,
        depth: m.depth * size,
        items: vec![Item::Glyph { face, ch, size, at: Vec2::ZERO, color }],
        italic,
        accent_at: Some((m.width * 0.5 + m.skew) * size),
    }
}

/// Upright words in the formula's own font (a function's name): glyph after glyph.
fn word(text: &str, face: Face, size: f32, color: Option<Color32>) -> Bx {
    let mut bx = Bx::default();
    let mut x = 0.0;
    for ch in text.chars() {
        if ch == ' ' {
            x += WORD_SPACE * size;
            continue;
        }
        let g = glyph(face, ch, size, color);
        let width = g.width;
        bx.place(g, x, 0.0);
        x += width;
    }
    bx.width = x;
    bx
}

/// The box of one atom, and its class.
fn atom(node: &Node, cx: Cx<'_>) -> (Class, Bx) {
    let s = cx.style.size();
    match node {
        Node::Sym { ch, face, class } => (*class, glyph(*face, *ch, s, cx.color)),
        Node::Neg(ch) => {
            let mut bx = glyph(Face::Main, *ch, s, cx.color);
            let stroke = glyph(Face::Main, '\u{E020}', s, cx.color);
            let x = (bx.width - stroke.width) * 0.5;
            bx.place(stroke, x, 0.0);
            (Class::Rel, bx)
        }
        Node::BigOp { ch, .. } => {
            let face = if cx.style.level == 0 { Face::Size2 } else { Face::Size1 };
            let mut g = glyph(face, *ch, s, cx.color);
            // centred on the axis
            let shift = (g.height - g.depth) * 0.5 - cx.style.p(AXIS);
            let italic = glyph_metrics(face, *ch).italic * s;
            g.width += italic;
            let mut bx = Bx::default();
            bx.place(g.clone(), 0.0, -shift);
            bx.width = g.width;
            bx.italic = italic;
            (Class::Op, bx)
        }
        Node::Func { name, .. } => (Class::Op, word(name, Face::Main, s, cx.color)),
        Node::Text { text, bold } => {
            let [width, height, depth] = (cx.measure)(text, *bold);
            let items = vec![Item::Words { text: text.clone(), size: s, at: Vec2::ZERO, color: cx.color, bold: *bold }];
            (Class::Ord, Bx { width: width * s, height: height * s, depth: depth * s, items, ..Bx::default() })
        }
        Node::Group(list) => (Class::Ord, hlist(list, cx)),
        Node::Scripts { base, sup, sub } => {
            let (class, base_box) = atom(base, cx);
            let limits = matches!(**base, Node::BigOp { limits: true, .. } | Node::Func { limits: true, .. }) && cx.style.level == 0;
            let bx = if limits { limits_on(base_box, sup.as_deref(), sub.as_deref(), cx) } else { scripts_on(base_box, sup.as_deref(), sub.as_deref(), cx) };
            (class, bx)
        }
        Node::Frac { num, den, style, binom } => (Class::Inner, fraction(num, den, *style, *binom, cx)),
        Node::Sqrt { index, body } => (Class::Ord, root(index.as_deref(), body, cx)),
        Node::Delimited { left, right, body } => (Class::Inner, delimited(*left, *right, body, cx)),
        Node::Middle(d) => (Class::Rel, delimiter(*d, 1.0 * s, cx)),
        Node::Big { delim, size, class } => (*class, delimiter(*delim, size * s, cx)),
        Node::Matrix { left, right, cases, rows } => {
            let inner = matrix(rows, *cases, cx);
            if left.is_none() && right.is_none() {
                (Class::Ord, inner)
            } else {
                (Class::Inner, around(*left, *right, inner, cx))
            }
        }
        Node::Accent { glyph: accent, body } => (Class::Ord, accented(*accent, body, cx)),
        Node::Line { over, body } => (Class::Ord, lined(*over, body, cx)),
        // (spaces, colours and styles are taken apart by `collect`, and reach here only
        // as the base of scripts)
        Node::Space(_) | Node::Color { .. } | Node::Style { .. } => (Class::Ord, hlist(std::slice::from_ref(node), cx)),
    }
}

/// Superscript and subscript beside a base (TeX's rule 18).
fn scripts_on(base: Bx, sup: Option<&[Node]>, sub: Option<&[Node]>, cx: Cx<'_>) -> Bx {
    let style = cx.style;
    let char_base = base.accent_at.is_some();
    let sup_style = style.sup();
    let sub_style = style.sub();
    let sup_box = sup.map(|list| hlist(list, cx.with(sup_style)));
    let sub_box = sub.map(|list| hlist(list, cx.with(sub_style)));
    let (mut up, mut down) = if char_base { (0.0, 0.0) } else { (base.height - sup_style.p(SUP_DROP), base.depth + sub_style.p(SUB_DROP)) };
    let x_height = style.p(X_HEIGHT);
    let min_up = if style.level == 0 {
        style.p(SUP1)
    } else if style.cramped {
        style.p(SUP3)
    } else {
        style.p(SUP2)
    };
    match (&sup_box, &sub_box) {
        (Some(p), Some(b)) => {
            up = up.max(min_up).max(p.depth + 0.25 * x_height);
            down = down.max(style.p(SUB2));
            let rule = style.p(RULE);
            let gap = (up - p.depth) - (b.height - down);
            if gap < 4.0 * rule {
                down = 4.0 * rule - (up - p.depth) + b.height;
                let psi = 0.8 * x_height - (up - p.depth);
                if psi > 0.0 {
                    up += psi;
                    down -= psi;
                }
            }
        }
        (None, Some(b)) => down = down.max(style.p(SUB1)).max(b.height - 0.8 * x_height),
        (Some(p), None) => up = up.max(min_up).max(p.depth + 0.25 * x_height),
        (None, None) => {}
    }
    let italic = base.italic;
    let base_width = base.width;
    let mut bx = Bx::default();
    bx.place(base, 0.0, 0.0);
    let mut width = base_width;
    if let Some(p) = sup_box {
        width = width.max(base_width + p.width);
        bx.place(p, base_width, up);
    }
    if let Some(b) = sub_box {
        // (tucked under a leaning letter)
        let x = base_width - italic;
        width = width.max(x + b.width);
        bx.place(b, x, -down);
    }
    bx.width = width + SCRIPT_SPACE * style.size();
    bx
}

/// Limits above and below a large operator in a display (`\sum_{i=1}^n`).
fn limits_on(base: Bx, sup: Option<&[Node]>, sub: Option<&[Node]>, cx: Cx<'_>) -> Bx {
    let style = cx.style;
    let sup_box = sup.map(|list| hlist(list, cx.with(style.sup())));
    let sub_box = sub.map(|list| hlist(list, cx.with(style.sub())));
    let slant = base.italic;
    let width = [Some(base.width), sup_box.as_ref().map(|b| b.width), sub_box.as_ref().map(|b| b.width)].into_iter().flatten().fold(0.0f32, f32::max);
    let (base_height, base_depth, base_width) = (base.height, base.depth, base.width);
    let mut bx = Bx::default();
    bx.place(base, (width - base_width) * 0.5, 0.0);
    if let Some(p) = sup_box {
        let kern = style.p(BIG_OP[0]).max(style.p(BIG_OP[2]) - p.depth);
        let y = base_height + kern + p.depth;
        let top = y + p.height + style.p(BIG_OP[4]);
        bx.place(p.clone(), (width - p.width) * 0.5 + slant * 0.5, y);
        bx.height = bx.height.max(top);
    }
    if let Some(b) = sub_box {
        let kern = style.p(BIG_OP[1]).max(style.p(BIG_OP[3]) - b.height);
        let y = -(base_depth + kern + b.height);
        let bottom = -y + b.depth + style.p(BIG_OP[4]);
        bx.place(b.clone(), (width - b.width) * 0.5 - slant * 0.5, y);
        bx.depth = bx.depth.max(bottom);
    }
    bx.width = width;
    bx
}

/// A fraction (TeX's rule 15), or a binomial coefficient: the same without its bar,
/// in brackets.
fn fraction(num: &[Node], den: &[Node], forced: Option<u8>, binom: bool, cx: Cx<'_>) -> Bx {
    let style = forced.map_or(cx.style, |level| Style::of(level, cx.style.cramped));
    let s = style.size();
    let num_box = hlist(num, cx.with(style.num()));
    let den_box = hlist(den, cx.with(style.den()));
    let rule = if binom { 0.0 } else { style.p(RULE) };
    let display = style.level == 0;
    let (mut up, mut down) = if display {
        (style.p(NUM1), style.p(DENOM1))
    } else if !binom {
        (style.p(NUM2), style.p(DENOM2))
    } else {
        (style.p(NUM3), style.p(DENOM2))
    };
    let axis = style.p(AXIS);
    if binom {
        let clearance = if display { 7.0 } else { 3.0 } * style.p(RULE);
        let gap = (up - num_box.depth) - (den_box.height - down);
        if gap < clearance {
            up += (clearance - gap) * 0.5;
            down += (clearance - gap) * 0.5;
        }
    } else {
        let clearance = if display { 3.0 * rule } else { rule };
        let over = (up - num_box.depth) - (axis + rule * 0.5);
        if over < clearance {
            up += clearance - over;
        }
        let under = (axis - rule * 0.5) - (den_box.height - down);
        if under < clearance {
            down += clearance - under;
        }
    }
    let inner = num_box.width.max(den_box.width);
    let pad = NULL_DELIMITER * s;
    let mut bx = Bx::default();
    let (num_width, den_width) = (num_box.width, den_box.width);
    bx.place(num_box, pad + (inner - num_width) * 0.5, up);
    bx.place(den_box, pad + (inner - den_width) * 0.5, -down);
    if !binom {
        bx.rule(vec2(pad, axis - rule * 0.5), vec2(pad + inner, axis + rule * 0.5), cx.color);
    }
    bx.width = inner + 2.0 * pad;
    if binom {
        let size = if display { style.p(DELIM1) } else { style.p(DELIM2) };
        return around_sized(Some('('), Some(')'), bx, size, cx.with(style));
    }
    bx
}

/// A root (TeX's rule 11): the sign, drawn as tall as it must be, and the bar over
/// exactly what is under it; with a small index if one is given.
fn root(index: Option<&[Node]>, body: &[Node], cx: Cx<'_>) -> Bx {
    let style = cx.style;
    let s = style.size();
    let inner = hlist(body, cx.with(style.cramp()));
    let rule = 0.04 * s;
    let phi = if style.level == 0 { style.p(X_HEIGHT) } else { rule };
    let mut clearance = rule + phi * 0.25;
    let needed = inner.height + inner.depth + clearance + rule;
    let total = needed.max(1.0 * s);
    if total - rule > inner.height + inner.depth + clearance {
        clearance = (clearance + total - rule - inner.height - inner.depth) * 0.5;
    }
    let top = inner.height + clearance + rule;
    let bottom = top - total;
    // the sign: a short tick up, a heavy stroke down, a long light one up to the bar
    let sign_width = (0.833 + 0.15 * (total / s - 1.0).max(0.0)).min(1.05) * s;
    let mut sign = Bx::default();
    let tick = vec2(0.075, bottom / s + 0.39) * s;
    let knee = vec2(0.205, bottom / s + 0.505) * s;
    let foot = vec2(0.375, bottom / s + 0.012) * s;
    let end = vec2(sign_width + 0.006 * s, top - rule * 0.5);
    sign.stroke(vec![(tick, 0.034 * s), (knee + vec2(0.012, -0.004) * s, 0.04 * s)], cx.color);
    sign.stroke(vec![(knee, 0.05 * s), (knee + (foot - knee) * 0.25, 0.086 * s), (foot + vec2(0.004, 0.0) * s, 0.082 * s)], cx.color);
    sign.stroke(vec![(foot + vec2(-0.012, -0.004) * s, rule * 1.05), (end, rule)], cx.color);
    let mut bx = Bx::default();
    bx.place(sign, 0.0, 0.0);
    let inner_width = inner.width;
    bx.place(inner, sign_width, 0.0);
    bx.rule(vec2(sign_width, top - rule), vec2(sign_width + inner_width, top), cx.color);
    bx.width = sign_width + inner_width;
    bx.height = bx.height.max(top + rule);
    bx.depth = bx.depth.max(-bottom);
    let Some(index) = index else { return bx };
    // the index, raised and tucked into the sign's crook (TeX's \root: 5mu before it,
    // 10mu back after it)
    let index_box = hlist(index, cx.with(Style::of(3, false)));
    let raise = 0.6 * (bx.height - bx.depth);
    let mu = style.p(QUAD) / 18.0;
    let sign_x = 5.0 * mu + index_box.width - 10.0 * mu;
    let shift = (-sign_x).max(0.0);
    let width = bx.width;
    let mut out = Bx::default();
    out.place(index_box, 5.0 * mu + shift, raise);
    out.place(bx, sign_x + shift, 0.0);
    out.width = sign_x + shift + width;
    out
}

/// `\left( … \right)`: brackets as tall as their contents (TeX's rule 19).
fn delimited(left: Delim, right: Delim, body: &[Node], cx: Cx<'_>) -> Bx {
    let inner = hlist(body, cx);
    let axis = cx.style.p(AXIS);
    let reach = (inner.height - axis).max(inner.depth + axis);
    let size = (reach / 500.0 * 901.0).max(2.0 * reach - 0.5 * cx.style.size());
    // (the bars of \middle were made at one size, not knowing the rest: again, at this one)
    if body.iter().any(|n| matches!(n, Node::Middle(_))) {
        let mut bx = Bx::default();
        let mut x = 0.0;
        let mut pieces: Vec<Bx> = Vec::new();
        let mut run: Vec<Node> = Vec::new();
        let mut middles = Vec::new();
        for node in body {
            if let Node::Middle(d) = node {
                pieces.push(hlist(&run, cx));
                run.clear();
                middles.push(*d);
            } else {
                run.push(node.clone());
            }
        }
        pieces.push(hlist(&run, cx));
        let thick = 5.0 * cx.style.p(QUAD) / 18.0;
        let gap = if cx.style.script() { 0.0 } else { thick };
        for (k, piece) in pieces.into_iter().enumerate() {
            let w = piece.width;
            bx.place(piece, x, 0.0);
            x += w;
            if let Some(d) = middles.get(k) {
                let bar = delimiter(*d, size, cx);
                let bw = bar.width;
                bx.place(bar, x + gap, 0.0);
                x += bw + 2.0 * gap;
            }
        }
        bx.width = x;
        return around_sized(left, right, bx, size, cx);
    }
    around_sized(left, right, inner, size, cx)
}

/// Brackets round a box, as tall as it needs (for a matrix).
fn around(left: Delim, right: Delim, inner: Bx, cx: Cx<'_>) -> Bx {
    let axis = cx.style.p(AXIS);
    let reach = (inner.height - axis).max(inner.depth + axis);
    let size = (reach / 500.0 * 901.0).max(2.0 * reach - 0.5 * cx.style.size());
    around_sized(left, right, inner, size, cx)
}

fn around_sized(left: Delim, right: Delim, inner: Bx, size: f32, cx: Cx<'_>) -> Bx {
    let l = delimiter(left, size, cx);
    let r = delimiter(right, size, cx);
    let mut bx = Bx::default();
    let (lw, iw, rw) = (l.width, inner.width, r.width);
    bx.place(l, 0.0, 0.0);
    bx.place(inner, lw, 0.0);
    bx.place(r, lw + iw, 0.0);
    bx.width = lw + iw + rw;
    bx
}

/// A bracket of at least `size` (top to bottom, ems of the formula), centred on the
/// axis: the font's own glyph in the three sizes it comes in, and drawn when taller.
fn delimiter(delim: Delim, size: f32, cx: Cx<'_>) -> Bx {
    let s = cx.style.size();
    let axis = cx.style.p(AXIS);
    let Some(c) = delim else { return Bx::empty(NULL_DELIMITER * s) };
    let ch = match c {
        '|' => '∣',
        other => other,
    };
    for (face, scale) in [(Face::Main, s), (Face::Size1, s), (Face::Size2, s)] {
        let Some(m) = symbols::metrics(face, ch) else { continue };
        if (m.height + m.depth) * scale >= size - 0.01 || (face == Face::Size2 && !drawable(c)) {
            let mut g = glyph(face, ch, scale, cx.color);
            let shift = axis - (g.height - g.depth) * 0.5;
            let width = g.width;
            g.italic = 0.0;
            let mut bx = Bx::default();
            bx.place(g, 0.0, shift);
            bx.width = width;
            return bx;
        }
    }
    drawn_delimiter(c, size, axis, s, cx.color)
}

/// Brackets that can be drawn at any height.
fn drawable(c: char) -> bool {
    matches!(c, '(' | ')' | '[' | ']' | '{' | '}' | '|' | '∥' | '⟨' | '⟩' | '⌊' | '⌋' | '⌈' | '⌉' | '/' | '\\')
}

/// A bracket drawn as tall as `size`, centred at `axis`, in the manner of Computer
/// Modern's: curved ones heavier in the middle than at the ends.
fn drawn_delimiter(c: char, size: f32, axis: f32, s: f32, color: Option<Color32>) -> Bx {
    let (top, bottom) = (axis + size * 0.5, axis - size * 0.5);
    let tall = (size / s - 1.8).max(0.0);
    let mut bx = Bx::default();
    let line = 0.045 * s;
    let mirror = matches!(c, ')' | ']' | '}' | '⟩' | '⌋' | '⌉' | '\\');
    let width;
    match c {
        '(' | ')' => {
            width = (0.6 + 0.06 * tall).min(0.8) * s;
            let (outer, inner) = (0.3 * width, 0.9 * width);
            let power = (2.2 + 0.6 * tall).min(4.0);
            let thick = (0.09 + 0.01 * tall).min(0.11) * s;
            let n = 40;
            let points = (0..=n)
                .map(|k| {
                    let u = k as f32 / n as f32 * 2.0 - 1.0;
                    let x = outer + (inner - outer) * u.abs().powf(power);
                    let w = 0.022 * s + (thick - 0.022 * s) * (1.0 - u.abs().powf(power.max(2.0) * 1.3));
                    (vec2(x, axis + u * (size * 0.5 - 0.012 * s)), w)
                })
                .collect();
            bx.stroke(points, color);
        }
        '[' | ']' | '⌊' | '⌋' | '⌈' | '⌉' => {
            width = 0.47 * s;
            let x0 = 0.16 * s;
            bx.rule(vec2(x0, bottom), vec2(x0 + line, top), color);
            if matches!(c, '[' | ']' | '⌈' | '⌉') {
                bx.rule(vec2(x0, top - line), vec2(width - 0.06 * s, top), color);
            }
            if matches!(c, '[' | ']' | '⌊' | '⌋') {
                bx.rule(vec2(x0, bottom), vec2(width - 0.06 * s, bottom + line), color);
            }
        }
        '{' | '}' => {
            width = (0.62 + 0.04 * tall).min(0.8) * s;
            let (tip, mid, end) = (0.14 * width, 0.5 * width, 0.86 * width);
            let r = (0.22 * s).min(size * 0.2);
            let thick = 0.08 * s;
            let thin = 0.03 * s;
            let mut points = Vec::new();
            let n = 12;
            // the hook at the top, the straight run down, the turn to the tip; and back
            let upper: Vec<(Vec2, f32)> = (0..=n)
                .map(|k| {
                    let a = k as f32 / n as f32 * std::f32::consts::FRAC_PI_2;
                    (vec2(mid + (end - mid) * (1.0 - a.cos()), top - r + r * a.sin()), thin + (thick - thin) * a.cos())
                })
                .rev()
                .collect();
            points.extend(upper);
            let turn: Vec<(Vec2, f32)> = (0..=n)
                .map(|k| {
                    let a = k as f32 / n as f32 * std::f32::consts::FRAC_PI_2;
                    (vec2(mid - (mid - tip) * (1.0 - a.cos()), axis + r * (1.0 - a.sin())), thick + (thin - thick) * (1.0 - a.cos()))
                })
                .collect();
            points.extend(turn.iter().copied());
            let lower: Vec<(Vec2, f32)> = points.iter().rev().skip(1).map(|(p, w)| (vec2(p.x, 2.0 * axis - p.y), *w)).collect();
            points.extend(lower);
            bx.stroke(points, color);
        }
        '⟨' | '⟩' => {
            width = 0.45 * s + 0.08 * tall * s;
            let (point, ends) = (0.12 * s, width - 0.1 * s);
            bx.stroke(vec![(vec2(ends, top - 0.01 * s), line), (vec2(point, axis), line * 1.1), (vec2(ends, bottom + 0.01 * s), line)], color);
        }
        '/' | '\\' => {
            width = 0.5 * s + 0.15 * tall * s;
            bx.stroke(vec![(vec2(0.08 * s, bottom), line), (vec2(width - 0.08 * s, top), line)], color);
        }
        '∥' => {
            width = 0.5 * s;
            for x in [0.14, 0.32] {
                bx.rule(vec2(x * s, bottom), vec2(x * s + 0.04 * s, top), color);
            }
        }
        _ => {
            width = 0.28 * s;
            bx.rule(vec2(0.12 * s, bottom), vec2(0.16 * s, top), color);
        }
    }
    bx.width = width;
    if mirror {
        bx.items = bx
            .items
            .into_iter()
            .map(|item| match item {
                Item::Rule { min, max, color } => Item::Rule { min: vec2(width - max.x, min.y), max: vec2(width - min.x, max.y), color },
                Item::Stroke { points, color } => Item::Stroke { points: points.into_iter().map(|(p, w)| (vec2(width - p.x, p.y), w)).collect(), color },
                other => other,
            })
            .collect();
    }
    bx.height = bx.height.max(top);
    bx.depth = bx.depth.max(-bottom);
    bx
}

/// The rows and columns of a matrix, centred on the axis (or, for `cases`, set to the
/// left with a quad between the columns, and rows a little further apart).
fn matrix(rows: &[Vec<Vec<Node>>], cases: bool, cx: Cx<'_>) -> Bx {
    let style = if cx.style.level <= 1 { Style::TEXT } else { cx.style };
    let s = style.size();
    let cells: Vec<Vec<Bx>> = rows.iter().map(|row| row.iter().map(|cell| hlist(cell, cx.with(style))).collect()).collect();
    let columns = cells.iter().map(Vec::len).max().unwrap_or(0);
    let mut widths = vec![0.0f32; columns];
    for row in &cells {
        for (j, cell) in row.iter().enumerate() {
            widths[j] = widths[j].max(cell.width);
        }
    }
    let skip = if cases { 1.2 * 1.2 } else { 1.2 } * s;
    let (strut_height, strut_depth) = (0.7 * skip, 0.3 * skip);
    // (a quad between columns: twice \arraycolsep in a matrix, and \quad in cases)
    let gap = s;
    let heights: Vec<(f32, f32)> =
        cells.iter().map(|row| row.iter().fold((strut_height, strut_depth), |(h, d), cell| (h.max(cell.height), d.max(cell.depth)))).collect();
    let total: f32 = heights.iter().map(|(h, d)| h + d).sum();
    let mut bx = Bx::default();
    let mut y = total * 0.5 + style.p(AXIS);
    let top = y;
    for (row, (h, d)) in cells.into_iter().zip(&heights) {
        y -= h;
        let mut x = 0.0;
        for (j, cell) in row.into_iter().enumerate() {
            let w = cell.width;
            let at = if cases { x } else { x + (widths[j] - w) * 0.5 };
            bx.place(cell, at, y);
            x += widths[j] + gap;
        }
        y -= d;
    }
    bx.width = widths.iter().sum::<f32>() + gap * columns.saturating_sub(1) as f32;
    bx.height = bx.height.max(top);
    bx.depth = bx.depth.max(-y);
    bx
}

/// An accent over its base (TeX's rule 12): over the top of the letter, lowered onto a
/// short letter, as tall a letter as it has to clear.
fn accented(accent: char, body: &[Node], cx: Cx<'_>) -> Bx {
    let s = cx.style.size();
    let base = hlist(body, cx.with(cx.style.cramp()));
    let centre = base.accent_at.unwrap_or(base.width * 0.5);
    let m = glyph_metrics(Face::Main, accent);
    let ink_centre = (m.width * 0.5 + m.skew) * s;
    let clearance = base.height.min(cx.style.p(X_HEIGHT));
    let lift = base.height - clearance;
    let g = glyph(Face::Main, accent, s, cx.color);
    let width = base.width;
    let mut bx = Bx::default();
    bx.place(base, 0.0, 0.0);
    bx.place(g, centre - ink_centre, lift);
    bx.width = width;
    bx
}

/// A line over (or under) its argument.
fn lined(over: bool, body: &[Node], cx: Cx<'_>) -> Bx {
    let rule = cx.style.p(RULE);
    let inner = hlist(body, cx.with(if over { cx.style.cramp() } else { cx.style }));
    let (h, d, w) = (inner.height, inner.depth, inner.width);
    let mut bx = Bx::default();
    bx.place(inner, 0.0, 0.0);
    if over {
        bx.rule(vec2(0.0, h + 3.0 * rule), vec2(w, h + 4.0 * rule), cx.color);
        bx.height = h + 5.0 * rule;
    } else {
        bx.rule(vec2(0.0, -d - 4.0 * rule), vec2(w, -d - 3.0 * rule), cx.color);
        bx.depth = d + 5.0 * rule;
    }
    bx.width = w;
    bx
}

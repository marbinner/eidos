//! Formulas read into a tree: a subset of LaTeX, which also takes the Unicode that
//! text written without it is full of (x₁², Aᵀ, ≤, √, î).

use super::symbols::{self, Class, Cmd, Face, Font};
use egui::Color32;

/// A bracket: its character, or nothing (`\left.`).
pub type Delim = Option<char>;

/// One piece of a formula.
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    /// A symbol: letter, digit, operator, relation, bracket.
    Sym {
        ch: char,
        face: Face,
        class: Class,
    },
    /// A relation with a stroke through it (≠).
    Neg(char),
    /// A large operator (∑), and whether its scripts go above and below in a display.
    BigOp {
        ch: char,
        limits: bool,
    },
    /// A function's name set upright (`\sin`), and whether its scripts go above and
    /// below in a display (`\max`).
    Func {
        name: String,
        limits: bool,
    },
    /// Words (`\text`): set in the font of the text round the formula, not in the
    /// formula's own, and in bold for `\textbf`.
    Text {
        text: String,
        bold: bool,
    },
    /// `{…}`.
    Group(Vec<Node>),
    Scripts {
        base: Box<Node>,
        sup: Option<Vec<Node>>,
        sub: Option<Vec<Node>>,
    },
    /// A fraction (or, without its bar and in brackets, a binomial coefficient), in the
    /// style given or the usual one.
    Frac {
        num: Vec<Node>,
        den: Vec<Node>,
        style: Option<u8>,
        binom: bool,
    },
    Sqrt {
        index: Option<Vec<Node>>,
        body: Vec<Node>,
    },
    /// `\left( … \right)`: brackets as tall as what they hold.
    Delimited {
        left: Delim,
        right: Delim,
        body: Vec<Node>,
    },
    /// `\middle|` inside `\left … \right`.
    Middle(Delim),
    /// A bracket of a fixed size (`\big(`), in ems.
    Big {
        delim: Delim,
        size: f32,
        class: Class,
    },
    Matrix {
        left: Delim,
        right: Delim,
        cases: bool,
        rows: Vec<Vec<Vec<Node>>>,
    },
    /// An accent over what follows it (a glyph of Main).
    Accent {
        glyph: char,
        body: Vec<Node>,
    },
    /// `\overline` (or `\underline`).
    Line {
        over: bool,
        body: Vec<Node>,
    },
    /// Space, in eighteenths of an em.
    Space(f32),
    Color {
        color: Color32,
        body: Vec<Node>,
    },
    /// `\displaystyle` (0) to `\scriptscriptstyle` (3), for the rest of the group.
    Style {
        level: u8,
        body: Vec<Node>,
    },
}

/// Every node of a formula, outer ones before those inside them.
pub fn walk<'n>(nodes: &'n [Node], f: &mut dyn FnMut(&'n Node)) {
    for node in nodes {
        f(node);
        match node {
            Node::Group(list) | Node::Color { body: list, .. } | Node::Style { body: list, .. } | Node::Line { body: list, .. } => walk(list, f),
            Node::Accent { body, .. } | Node::Delimited { body, .. } => walk(body, f),
            Node::Scripts { base, sup, sub } => {
                walk(std::slice::from_ref(base), f);
                for list in [sup, sub].into_iter().flatten() {
                    walk(list, f);
                }
            }
            Node::Frac { num, den, .. } => {
                walk(num, f);
                walk(den, f);
            }
            Node::Sqrt { index, body } => {
                if let Some(index) = index {
                    walk(index, f);
                }
                walk(body, f);
            }
            Node::Matrix { rows, .. } => rows.iter().flatten().for_each(|cell| walk(cell, f)),
            Node::Sym { .. }
            | Node::Neg(_)
            | Node::BigOp { .. }
            | Node::Func { .. }
            | Node::Text { .. }
            | Node::Middle(_)
            | Node::Big { .. }
            | Node::Space(_) => {}
        }
    }
}

/// A formula read, with the runs of four or more italic letters found in it (words
/// set as if they were products of variables, which is almost always a mistake).
#[derive(Debug)]
pub struct Parsed {
    pub nodes: Vec<Node>,
    pub words: Vec<String>,
}

/// Read a formula (what stands between the dollar signs).
pub fn parse(src: &str) -> Result<Parsed, String> {
    let mut parser = Parser::new(src, 0);
    let nodes = parser.list(Stop::End)?;
    if let Some(c) = parser.peek() {
        return Err(format!("unexpected {c}"));
    }
    Ok(Parsed { nodes, words: parser.words })
}

/// What a superscript or subscript character stands for ([`symbols::superscript`]).
type Mapping = fn(char) -> Option<&'static str>;

/// Deeper than this, a formula is taken to be a mistake (and a stack is not overrun).
const DEEPEST: u32 = 40;
/// The most rows or columns a matrix may have.
const LARGEST: usize = 64;

/// Where a list of atoms ends.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Stop {
    /// At the end of the formula.
    End,
    /// At a `}`.
    Brace,
    /// At a `]` (the index of `\sqrt[3]{x}`).
    Bracket,
    /// At `\right`.
    Right,
    /// At `&`, `\\` or `\end` (a cell of a matrix).
    Cell,
}

struct Parser {
    chars: Vec<char>,
    pos: usize,
    depth: u32,
    font: Font,
    words: Vec<String>,
}

impl Parser {
    fn new(src: &str, depth: u32) -> Self {
        Parser { chars: src.chars().collect(), pos: 0, depth, font: Font::Normal, words: Vec::new() }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn peek_at(&self, ahead: usize) -> Option<char> {
        self.chars.get(self.pos + ahead).copied()
    }

    fn skip_space(&mut self) {
        while self.peek().is_some_and(|c| c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            self.pos += 1;
        }
    }

    /// The name of the command starting here (at a backslash), without reading it.
    fn command_here(&self) -> Option<String> {
        if self.peek() != Some('\\') {
            return None;
        }
        let first = self.peek_at(1)?;
        if !first.is_ascii_alphabetic() {
            return Some(first.to_string());
        }
        Some(self.chars[self.pos + 1..].iter().take_while(|c| c.is_ascii_alphabetic()).collect())
    }

    /// Read the command starting here (at a backslash) and the spaces after a named one.
    fn read_command(&mut self) -> Result<String, String> {
        let name = self.command_here().ok_or("a lone \\ at the end")?;
        self.pos += 1 + name.chars().count();
        if name.chars().all(|c| c.is_ascii_alphabetic()) {
            self.skip_space();
        }
        Ok(name)
    }

    fn deeper(&mut self) -> Result<(), String> {
        self.depth += 1;
        if self.depth > DEEPEST {
            return Err("nested too deeply".into());
        }
        Ok(())
    }

    /// A list of atoms, up to (not including) what ends it.
    fn list(&mut self, stop: Stop) -> Result<Vec<Node>, String> {
        self.deeper()?;
        let mut nodes = Vec::new();
        let mut word = String::new();
        loop {
            self.skip_space();
            let Some(c) = self.peek() else {
                match stop {
                    Stop::End => break,
                    Stop::Brace => return Err("a { is not closed".into()),
                    Stop::Bracket => return Err("a [ is not closed".into()),
                    Stop::Right => return Err("\\left without \\right".into()),
                    Stop::Cell => return Err("\\begin without \\end".into()),
                }
            };
            match c {
                '}' if stop == Stop::Brace => break,
                '}' => return Err("a } that closes nothing".into()),
                ']' if stop == Stop::Bracket => break,
                '&' if stop == Stop::Cell => break,
                '&' => return Err("& outside a matrix".into()),
                _ => {}
            }
            if let Some(name) = self.command_here() {
                match name.as_str() {
                    "\\" | "end" if stop == Stop::Cell => break,
                    "\\" => return Err("\\\\ outside a matrix".into()),
                    "right" if stop == Stop::Right => break,
                    "right" => return Err("\\right without \\left".into()),
                    "middle" if stop == Stop::Right => {
                        self.read_command()?;
                        nodes.push(Node::Middle(self.delimiter()?));
                        continue;
                    }
                    "color" => {
                        self.read_command()?;
                        let color = self.color()?;
                        let body = self.list(stop)?;
                        nodes.push(Node::Color { color, body });
                        break;
                    }
                    _ => {}
                }
                if let Some(Cmd::Style(level)) = symbols::command(&name) {
                    self.read_command()?;
                    let body = self.list(stop)?;
                    nodes.push(Node::Style { level, body });
                    break;
                }
            }
            if c.is_ascii_digit() && stop != Stop::Bracket {
                if let Some(number) = self.scientific()? {
                    self.note_word(&mut word);
                    nodes.extend(number);
                    continue;
                }
            }
            let atom = self.atom()?;
            let atom = self.scripts(atom)?;
            // (a run of italic letters: a word set as a product, if it is long)
            let letter = |n: &Node| matches!(n, Node::Sym { ch, face: Face::MathItalic, .. } if ch.is_ascii_alphabetic());
            match &atom {
                n if letter(n) && self.font == Font::Normal => {
                    if let Node::Sym { ch, .. } = n {
                        word.push(*ch);
                    }
                }
                Node::Scripts { base, .. } if letter(base) && self.font == Font::Normal => {
                    if let Node::Sym { ch, .. } = **base {
                        word.push(ch);
                    }
                    self.note_word(&mut word);
                }
                _ => self.note_word(&mut word),
            }
            nodes.push(atom);
        }
        self.note_word(&mut word);
        self.depth -= 1;
        Ok(nodes)
    }

    fn note_word(&mut self, word: &mut String) {
        if word.chars().count() >= 4 {
            self.words.push(word.clone());
        }
        word.clear();
    }

    /// A number in the form computers write (2.5e-3), set as 2.5 × 10⁻³.
    fn scientific(&mut self) -> Result<Option<Vec<Node>>, String> {
        let start = self.pos;
        let mut at = start;
        let digits = |at: &mut usize, chars: &[char]| {
            let from = *at;
            while chars.get(*at).is_some_and(|c| c.is_ascii_digit()) {
                *at += 1;
            }
            *at > from
        };
        digits(&mut at, &self.chars);
        if self.chars.get(at) == Some(&'.') && self.chars.get(at + 1).is_some_and(|c| c.is_ascii_digit()) {
            at += 1;
            digits(&mut at, &self.chars);
        }
        let mantissa_end = at;
        if !matches!(self.chars.get(at), Some('e' | 'E')) {
            return Ok(None);
        }
        at += 1;
        let negative = matches!(self.chars.get(at), Some('-' | '−'));
        if matches!(self.chars.get(at), Some('-' | '−' | '+')) {
            at += 1;
        }
        let exponent_start = at;
        if !digits(&mut at, &self.chars) || self.chars.get(at).is_some_and(|c| c.is_alphanumeric()) {
            return Ok(None);
        }
        let mantissa: String = self.chars[start..mantissa_end].iter().collect();
        let mut exponent: String = self.chars[exponent_start..at].iter().collect();
        exponent = exponent.trim_start_matches('0').to_string();
        if exponent.is_empty() {
            exponent.push('0');
        }
        self.pos = at;
        let digit = |c: char| Node::Sym { ch: c, face: Face::Main, class: Class::Ord };
        let mut nodes: Vec<Node> = mantissa.chars().map(digit).collect();
        nodes.push(Node::Sym { ch: '×', face: Face::Main, class: Class::Bin });
        let mut sup = Vec::new();
        if negative {
            sup.push(Node::Sym { ch: '−', face: Face::Main, class: Class::Bin });
        }
        sup.extend(exponent.chars().map(digit));
        nodes.push(Node::Scripts { base: Box::new(Node::Group(vec![digit('1'), digit('0')])), sup: Some(sup), sub: None });
        Ok(Some(nodes))
    }

    /// One atom, without its scripts.
    fn atom(&mut self) -> Result<Node, String> {
        self.skip_space();
        let Some(c) = self.peek() else { return Err("something is missing at the end".into()) };
        // (scripts on nothing, as TeX allows: ^2 alone)
        if matches!(c, '^' | '_' | '\'' | '′' | '″') || symbols::superscript(c).is_some() || symbols::subscript(c).is_some() {
            return Ok(Node::Group(Vec::new()));
        }
        let node = match c {
            '{' => {
                self.pos += 1;
                let inner = self.list(Stop::Brace)?;
                self.expect('}')?;
                Node::Group(inner)
            }
            '\\' => return self.command(),
            '√' => {
                self.pos += 1;
                Node::Sqrt { index: None, body: self.argument()? }
            }
            '~' | '\u{A0}' => {
                self.pos += 1;
                Node::Space(4.5)
            }
            '\u{2009}' | '\u{202F}' | '\u{2006}' => {
                self.pos += 1;
                Node::Space(3.0)
            }
            '\u{2002}' => {
                self.pos += 1;
                Node::Space(9.0)
            }
            '\u{2003}' => {
                self.pos += 1;
                Node::Space(18.0)
            }
            _ => {
                self.pos += 1;
                if let Some((base, accent)) = symbols::accented(c) {
                    let base = self.symbol(base)?;
                    Node::Accent { glyph: accent, body: vec![base] }
                } else {
                    self.symbol(c)?
                }
            }
        };
        // accents written after their letter (x̂)
        let mut node = node;
        while let Some(glyph) = self.peek().and_then(symbols::combining) {
            self.pos += 1;
            node = match node {
                // (a hat goes on a dotless i)
                Node::Sym { ch: 'i', face: Face::MathItalic, class } => Node::Sym { ch: '\u{E131}', face: Face::MathItalic, class },
                Node::Sym { ch: 'j', face: Face::MathItalic, class } => Node::Sym { ch: '\u{E237}', face: Face::MathItalic, class },
                other => other,
            };
            node = Node::Accent { glyph, body: vec![node] };
        }
        Ok(node)
    }

    /// A character typed into the formula.
    fn symbol(&mut self, c: char) -> Result<Node, String> {
        if let Some((ch, face)) = symbols::letter(c, self.font) {
            return sym(ch, face, Class::Ord);
        }
        match symbols::typed(c) {
            Some(Cmd::Sym(ch, face, class)) => sym(ch, face, class),
            Some(Cmd::Neg(ch)) => negated(ch),
            Some(Cmd::BigOp(ch, limits)) => {
                large(ch)?;
                self.limits(Node::BigOp { ch, limits })
            }
            _ if c.is_alphanumeric() && self.font != Font::Normal => Err(format!("{c} cannot be set in this alphabet")),
            _ => Err(format!("no glyph for {c}")),
        }
    }

    /// `\limits` or `\nolimits` after a large operator or a function.
    fn limits(&mut self, mut node: Node) -> Result<Node, String> {
        loop {
            self.skip_space();
            let Some(name) = self.command_here() else { return Ok(node) };
            let Some(Cmd::Limits(on)) = symbols::command(&name) else { return Ok(node) };
            self.read_command()?;
            match &mut node {
                Node::BigOp { limits, .. } | Node::Func { limits, .. } => *limits = on,
                _ => return Err(format!("\\{name} after something that is not an operator")),
            }
        }
    }

    fn expect(&mut self, c: char) -> Result<(), String> {
        self.skip_space();
        if self.peek() == Some(c) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("{c} expected"))
        }
    }

    /// What a command or a script applies to: a `{group}` or a single atom.
    fn argument(&mut self) -> Result<Vec<Node>, String> {
        self.skip_space();
        match self.peek() {
            Some('{') => {
                self.pos += 1;
                let inner = self.list(Stop::Brace)?;
                self.expect('}')?;
                Ok(inner)
            }
            Some('}') | None => Err("an argument is missing".into()),
            Some('^' | '_') => Err("an argument is missing".into()),
            _ => {
                self.deeper()?;
                let atom = self.atom()?;
                self.depth -= 1;
                Ok(vec![atom])
            }
        }
    }

    /// The raw text of a `{…}` argument (for `\text`, colours and environment names).
    fn raw(&mut self) -> Result<String, String> {
        self.skip_space();
        if self.peek() != Some('{') {
            return Err("{ expected".into());
        }
        self.pos += 1;
        let mut out = String::new();
        let mut level = 0;
        loop {
            let Some(c) = self.peek() else { return Err("a { is not closed".into()) };
            self.pos += 1;
            match c {
                '{' => {
                    level += 1;
                    out.push(c);
                }
                '}' if level == 0 => return Ok(out),
                '}' => {
                    level -= 1;
                    out.push(c);
                }
                '\\' => match self.peek() {
                    Some(e @ ('$' | '{' | '}' | '%' | '&' | '_' | '#')) => {
                        self.pos += 1;
                        out.push(e);
                    }
                    Some(' ') => {
                        self.pos += 1;
                        out.push(' ');
                    }
                    _ => return Err("commands inside \\text are not supported".into()),
                },
                _ => out.push(c),
            }
        }
    }

    fn color(&mut self) -> Result<Color32, String> {
        let name = self.raw()?;
        symbols::color(name.trim()).ok_or_else(|| format!("unknown colour {name}"))
    }

    /// A bracket after `\left`, `\right`, `\middle` or `\big`.
    fn delimiter(&mut self) -> Result<Delim, String> {
        self.skip_space();
        let Some(c) = self.peek() else { return Err("a bracket is missing".into()) };
        if c == '\\' {
            let name = self.read_command()?;
            return Ok(Some(match name.as_str() {
                "{" | "lbrace" => '{',
                "}" | "rbrace" => '}',
                "|" | "Vert" | "lVert" | "rVert" => '∥',
                "vert" | "lvert" | "rvert" => '|',
                "langle" => '⟨',
                "rangle" => '⟩',
                "lfloor" => '⌊',
                "rfloor" => '⌋',
                "lceil" => '⌈',
                "rceil" => '⌉',
                "lbrack" => '[',
                "rbrack" => ']',
                "backslash" => '\\',
                _ => return Err(format!("\\{name} is not a bracket")),
            }));
        }
        self.pos += 1;
        Ok(match c {
            '.' => None,
            '(' | ')' | '[' | ']' | '|' | '/' | '⟨' | '⟩' | '⌊' | '⌋' | '⌈' | '⌉' => Some(c),
            '‖' | '∥' => Some('∥'),
            '<' => Some('⟨'),
            '>' => Some('⟩'),
            _ => return Err(format!("{c} is not a bracket")),
        })
    }

    fn command(&mut self) -> Result<Node, String> {
        let name = self.read_command()?;
        let Some(cmd) = symbols::command(&name) else { return Err(format!("unknown command \\{name}")) };
        Ok(match cmd {
            Cmd::Sym(ch, face, class) => {
                // (an upright capital Greek letter follows \mathbf and \mathit)
                let (ch, face) = symbols::letter(ch, self.font).unwrap_or((ch, face));
                sym(ch, face, class)?
            }
            Cmd::Neg(ch) => negated(ch)?,
            Cmd::BigOp(ch, limits) => {
                large(ch)?;
                self.limits(Node::BigOp { ch, limits })?
            }
            Cmd::Func(name, limits) => self.limits(Node::Func { name: name.to_string(), limits })?,
            Cmd::Accent(glyph) => {
                let mut body = self.argument()?;
                // (a hat goes on a dotless i)
                if let [Node::Sym { ch: ch @ ('i' | 'j'), face: Face::MathItalic, .. }] = body.as_mut_slice() {
                    *ch = if *ch == 'i' { '\u{E131}' } else { '\u{E237}' };
                }
                Node::Accent { glyph, body }
            }
            Cmd::Overline => Node::Line { over: true, body: self.argument()? },
            Cmd::Underline => Node::Line { over: false, body: self.argument()? },
            Cmd::Frac(style) | Cmd::Binom(style) => {
                let num = self.argument()?;
                let den = self.argument()?;
                Node::Frac { num, den, style, binom: matches!(cmd, Cmd::Binom(_)) }
            }
            Cmd::Sqrt => {
                self.skip_space();
                let index = if self.peek() == Some('[') {
                    self.pos += 1;
                    let index = self.list(Stop::Bracket)?;
                    self.expect(']')?;
                    Some(index)
                } else {
                    None
                };
                Node::Sqrt { index, body: self.argument()? }
            }
            Cmd::Left => {
                let left = self.delimiter()?;
                let body = self.list(Stop::Right)?;
                self.read_command()?;
                let right = self.delimiter()?;
                Node::Delimited { left, right, body }
            }
            Cmd::Right => return Err("\\right without \\left".into()),
            Cmd::Middle => return Err("\\middle outside \\left … \\right".into()),
            Cmd::Big(size, class) => Node::Big { delim: self.delimiter()?, size, class },
            Cmd::Begin => self.environment()?,
            Cmd::End => return Err("\\end without \\begin".into()),
            // (any character goes: the words are set in the interface's fonts, which
            // have what plain text has)
            Cmd::Text(bold) => Node::Text { text: self.raw()?, bold },
            Cmd::OperatorName => {
                self.skip_space();
                let star = self.peek() == Some('*');
                if star {
                    self.pos += 1;
                }
                let name = self.raw()?.trim().to_string();
                if name.is_empty() || name.chars().any(|c| symbols::metrics(Face::Main, c).is_none() && c != ' ') {
                    return Err(format!("\\operatorname{{{name}}} cannot be set"));
                }
                self.limits(Node::Func { name, limits: star })?
            }
            Cmd::Font(font) => {
                let outer = std::mem::replace(&mut self.font, font);
                let body = self.argument();
                self.font = outer;
                // (one letter is that letter, as typed: \mathbb{R} is ℝ)
                match body? {
                    mut one if one.len() == 1 => one.remove(0),
                    many => Node::Group(many),
                }
            }
            Cmd::Space(mu) => Node::Space(mu),
            Cmd::TextColor => {
                let color = self.color()?;
                Node::Color { color, body: self.argument()? }
            }
            Cmd::Not => {
                let next = self.atom()?;
                match next {
                    Node::Sym { ch, face: Face::Main, class: Class::Rel } => negated(ch)?,
                    _ => return Err("\\not before something that is not a relation".into()),
                }
            }
            Cmd::Color | Cmd::Style(_) => return Err(format!("\\{name} cannot stand here")),
            Cmd::Limits(_) => return Err(format!("\\{name} after something that is not an operator")),
        })
    }

    /// `\begin{pmatrix} a & b \\ c & d \end{pmatrix}` and its kin.
    fn environment(&mut self) -> Result<Node, String> {
        let name = self.raw()?;
        let (left, right, cases) = match name.as_str() {
            "matrix" => (None, None, false),
            "pmatrix" => (Some('('), Some(')'), false),
            "bmatrix" => (Some('['), Some(']'), false),
            "Bmatrix" => (Some('{'), Some('}'), false),
            "vmatrix" => (Some('|'), Some('|'), false),
            "Vmatrix" => (Some('∥'), Some('∥'), false),
            "cases" => (Some('{'), None, true),
            _ => return Err(format!("unknown environment {name}")),
        };
        let mut rows: Vec<Vec<Vec<Node>>> = Vec::new();
        let mut row = Vec::new();
        loop {
            let cell = self.list(Stop::Cell)?;
            row.push(cell);
            if self.peek() == Some('&') {
                self.pos += 1;
            } else {
                let command = self.read_command()?;
                let last = command == "end";
                // (a \\ just before \end makes no row)
                if !(last && row.len() == 1 && row[0].is_empty() && !rows.is_empty()) {
                    rows.push(std::mem::take(&mut row));
                }
                if last {
                    let end = self.raw()?;
                    if end != name {
                        return Err(format!("\\begin{{{name}}} ended by \\end{{{end}}}"));
                    }
                    break;
                }
            }
            if rows.len() > LARGEST || row.len() > LARGEST {
                return Err("a matrix too large".into());
            }
        }
        Ok(Node::Matrix { left, right, cases, rows })
    }

    /// The scripts after an atom: `^`, `_`, primes, and the superscript and subscript
    /// characters (x₁², Aᵀ).
    fn scripts(&mut self, base: Node) -> Result<Node, String> {
        let (mut sup, mut sub): (Option<Vec<Node>>, Option<Vec<Node>>) = (None, None);
        let mut primes = 0;
        let twice = |which: &str| Err(format!("two {which}scripts on one atom"));
        loop {
            self.skip_space();
            let Some(c) = self.peek() else { break };
            match c {
                '^' => {
                    self.pos += 1;
                    if sup.is_some() {
                        return twice("super");
                    }
                    sup = Some(self.argument()?);
                }
                '_' => {
                    self.pos += 1;
                    if sub.is_some() {
                        return twice("sub");
                    }
                    sub = Some(self.argument()?);
                }
                '\'' | '′' | '″' if sup.is_none() => {
                    self.pos += 1;
                    primes += if c == '″' { 2 } else { 1 };
                }
                _ => {
                    let (map, slot, which): (Mapping, &mut Option<Vec<Node>>, &str) = if symbols::superscript(c).is_some() {
                        (symbols::superscript, &mut sup, "super")
                    } else if symbols::subscript(c).is_some() {
                        (symbols::subscript, &mut sub, "sub")
                    } else {
                        break;
                    };
                    let mut text = String::new();
                    while let Some(part) = self.peek().and_then(map) {
                        text.push_str(part);
                        self.pos += 1;
                    }
                    if slot.is_some() {
                        return twice(which);
                    }
                    let mut inner = Parser::new(&text, self.depth + 1);
                    *slot = Some(inner.list(Stop::End)?);
                }
            }
        }
        if primes > 0 {
            let mut marks: Vec<Node> = (0..primes).map(|_| Node::Sym { ch: '′', face: Face::Main, class: Class::Ord }).collect();
            marks.extend(sup.take().unwrap_or_default());
            sup = Some(marks);
        }
        if sup.is_none() && sub.is_none() {
            return Ok(base);
        }
        Ok(Node::Scripts { base: Box::new(base), sup, sub })
    }
}

/// A symbol, if the font has its glyph (a missing one would be drawn as a box).
fn sym(ch: char, face: Face, class: Class) -> Result<Node, String> {
    match symbols::metrics(face, ch) {
        Some(_) => Ok(Node::Sym { ch, face, class }),
        None => Err(format!("no glyph for {ch}")),
    }
}

/// A relation with a stroke through it.
fn negated(ch: char) -> Result<Node, String> {
    sym(ch, Face::Main, Class::Rel).map(|_| Node::Neg(ch))
}

/// A large operator: there must be both sizes of it.
fn large(ch: char) -> Result<(), String> {
    match (symbols::metrics(Face::Size1, ch), symbols::metrics(Face::Size2, ch)) {
        (Some(_), Some(_)) => Ok(()),
        _ => Err(format!("no large {ch}")),
    }
}

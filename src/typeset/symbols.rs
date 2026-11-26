//! What formulas are made of: the fonts, the size of each glyph in them, and for every
//! command and every character that may be typed, which glyph it is and what kind of
//! atom (which decides the space around it, as in TeX).

use super::metrics;

/// One of the mathematics fonts (assets/math): Computer Modern as KaTeX ships it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Face {
    /// Variables: Latin and Greek letters, italic.
    MathItalic,
    /// Digits, upright letters, operators, relations, arrows, brackets, upright capital Greek.
    Main,
    /// `\mathbf`.
    MainBold,
    /// Blackboard capitals (`\mathbb`) and the less common relations.
    Ams,
    /// Large operators in running text, and the first size up of the brackets.
    Size1,
    /// Large operators in a display, and the second size up of the brackets.
    Size2,
}

impl Face {
    pub const ALL: [Face; 6] = [Face::MathItalic, Face::Main, Face::MainBold, Face::Ams, Face::Size1, Face::Size2];

    /// The name of the font family the face is installed under (see `app::style`).
    pub fn family(self) -> &'static str {
        match self {
            Face::MathItalic => "math-italic",
            Face::Main => "math-main",
            Face::MainBold => "math-bold",
            Face::Ams => "math-ams",
            Face::Size1 => "math-size1",
            Face::Size2 => "math-size2",
        }
    }

    /// The font file.
    pub fn data(self) -> &'static [u8] {
        match self {
            Face::MathItalic => include_bytes!("../../assets/math/KaTeX_Math-Italic.ttf"),
            Face::Main => include_bytes!("../../assets/math/KaTeX_Main-Regular.ttf"),
            Face::MainBold => include_bytes!("../../assets/math/KaTeX_Main-Bold.ttf"),
            Face::Ams => include_bytes!("../../assets/math/KaTeX_AMS-Regular.ttf"),
            Face::Size1 => include_bytes!("../../assets/math/KaTeX_Size1-Regular.ttf"),
            Face::Size2 => include_bytes!("../../assets/math/KaTeX_Size2-Regular.ttf"),
        }
    }

    fn table(self) -> &'static [metrics::Glyph] {
        match self {
            Face::MathItalic => &metrics::MATH_ITALIC,
            Face::Main => &metrics::MAIN,
            Face::MainBold => &metrics::MAIN_BOLD,
            Face::Ams => &metrics::AMS,
            Face::Size1 => &metrics::SIZE1,
            Face::Size2 => &metrics::SIZE2,
        }
    }
}

/// The size of a glyph, in ems: its advance, its ink above and below the baseline, how
/// far an italic letter leans out past its advance, and how far right of the middle
/// of its advance the top of the letter is (where an accent goes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Metrics {
    pub width: f32,
    pub height: f32,
    pub depth: f32,
    pub italic: f32,
    pub skew: f32,
}

/// The size of `c` in `face`, if the face has it.
pub fn metrics(face: Face, c: char) -> Option<Metrics> {
    let table = face.table();
    let i = table.binary_search_by_key(&(c as u32), |g| g.0).ok()?;
    let (_, w, h, d, italic, skew) = table[i];
    let em = |v: i16| v as f32 / 1000.0;
    Some(Metrics { width: em(w), height: em(h), depth: em(d), italic: em(italic), skew: em(skew) })
}

/// What kind of atom something is, which decides the space between neighbours (TeX's
/// classes: ordinary, large operator, binary operator, relation, opening, closing,
/// punctuation, and inner, which fractions and bracketed groups are).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Ord,
    Op,
    Bin,
    Rel,
    Open,
    Close,
    Punct,
    Inner,
}

/// The alphabet a letter is set in (`\mathrm`, `\mathbf`, `\mathbb`, `\mathit`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Font {
    /// Letters italic, digits upright: the usual.
    Normal,
    Roman,
    Bold,
    Blackboard,
    Italic,
}

/// A letter or digit as it is set in `font`: the glyph and its face.
pub fn letter(c: char, font: Font) -> Option<(char, Face)> {
    let latin = c.is_ascii_alphabetic();
    let digit = c.is_ascii_digit();
    let upper_greek = UPPER_GREEK.contains(&c);
    let lower_greek = ('α'..='ω').contains(&c) || "ϑϕϖϱϵ".contains(c);
    let face = match font {
        _ if lower_greek => Face::MathItalic,
        Font::Normal if latin => Face::MathItalic,
        Font::Normal | Font::Roman if digit || upper_greek || latin => Face::Main,
        Font::Bold if latin || digit || upper_greek => Face::MainBold,
        Font::Italic if latin || upper_greek => Face::MathItalic,
        Font::Italic if digit => Face::Main,
        Font::Blackboard if c.is_ascii_uppercase() || c == 'k' => Face::Ams,
        _ => return None,
    };
    // (capital Greek letters other than these look like Latin ones, and TeX has none)
    if ('Α'..='Ω').contains(&c) && !upper_greek {
        return None;
    }
    Some((c, face))
}

const UPPER_GREEK: [char; 11] = ['Γ', 'Δ', 'Θ', 'Λ', 'Ξ', 'Π', 'Σ', 'Υ', 'Φ', 'Ψ', 'Ω'];

/// What a command stands for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cmd {
    /// A symbol: its glyph, face and class.
    Sym(char, Face, Class),
    /// A relation (from Main) with a stroke through it: ≠, ∉.
    Neg(char),
    /// A large operator, and whether its scripts go above and below it in a display.
    BigOp(char, bool),
    /// A function named in upright letters, and whether its scripts go above and below
    /// it in a display (`\lim`, `\max`).
    Func(&'static str, bool),
    /// An accent over its argument (a glyph of Main).
    Accent(char),
    Overline,
    Underline,
    /// A fraction, in the style given (`\dfrac`, `\tfrac`) or the usual one.
    Frac(Option<u8>),
    Binom(Option<u8>),
    Sqrt,
    Left,
    Right,
    Middle,
    /// A bracket of a fixed size (`\big(`): its height in ems and its class.
    Big(f32, Class),
    Begin,
    End,
    /// Words, `\text`, set in the interface's own font (in bold for `\textbf`).
    Text(bool),
    OperatorName,
    Font(Font),
    /// Space, in eighteenths of an em.
    Space(f32),
    TextColor,
    Color,
    Not,
    Limits(bool),
    /// `\displaystyle` (0) to `\scriptscriptstyle` (3).
    Style(u8),
}

/// The command `\name`.
pub fn command(name: &str) -> Option<Cmd> {
    use Class::*;
    use Cmd::*;
    use Face::*;
    let sym = |c: char, face: Face, class: Class| Some(Sym(c, face, class));
    match name {
        // Greek
        "alpha" => sym('α', MathItalic, Ord),
        "beta" => sym('β', MathItalic, Ord),
        "gamma" => sym('γ', MathItalic, Ord),
        "delta" => sym('δ', MathItalic, Ord),
        "epsilon" => sym('ϵ', MathItalic, Ord),
        "varepsilon" => sym('ε', MathItalic, Ord),
        "zeta" => sym('ζ', MathItalic, Ord),
        "eta" => sym('η', MathItalic, Ord),
        "theta" => sym('θ', MathItalic, Ord),
        "vartheta" => sym('ϑ', MathItalic, Ord),
        "iota" => sym('ι', MathItalic, Ord),
        "kappa" => sym('κ', MathItalic, Ord),
        "lambda" => sym('λ', MathItalic, Ord),
        "mu" => sym('μ', MathItalic, Ord),
        "nu" => sym('ν', MathItalic, Ord),
        "xi" => sym('ξ', MathItalic, Ord),
        "omicron" => sym('ο', MathItalic, Ord),
        "pi" => sym('π', MathItalic, Ord),
        "varpi" => sym('ϖ', MathItalic, Ord),
        "rho" => sym('ρ', MathItalic, Ord),
        "varrho" => sym('ϱ', MathItalic, Ord),
        "sigma" => sym('σ', MathItalic, Ord),
        "varsigma" => sym('ς', MathItalic, Ord),
        "tau" => sym('τ', MathItalic, Ord),
        "upsilon" => sym('υ', MathItalic, Ord),
        "phi" => sym('ϕ', MathItalic, Ord),
        "varphi" => sym('φ', MathItalic, Ord),
        "chi" => sym('χ', MathItalic, Ord),
        "psi" => sym('ψ', MathItalic, Ord),
        "omega" => sym('ω', MathItalic, Ord),
        "Gamma" => sym('Γ', Main, Ord),
        "Delta" => sym('Δ', Main, Ord),
        "Theta" => sym('Θ', Main, Ord),
        "Lambda" => sym('Λ', Main, Ord),
        "Xi" => sym('Ξ', Main, Ord),
        "Pi" => sym('Π', Main, Ord),
        "Sigma" => sym('Σ', Main, Ord),
        "Upsilon" => sym('Υ', Main, Ord),
        "Phi" => sym('Φ', Main, Ord),
        "Psi" => sym('Ψ', Main, Ord),
        "Omega" => sym('Ω', Main, Ord),
        // letter-like symbols
        "imath" => sym('\u{E131}', MathItalic, Ord),
        "jmath" => sym('\u{E237}', MathItalic, Ord),
        "ell" => sym('ℓ', Main, Ord),
        "hbar" => sym('ℏ', Main, Ord),
        "partial" => sym('∂', Main, Ord),
        "nabla" => sym('∇', Main, Ord),
        "infty" => sym('∞', Main, Ord),
        "emptyset" => sym('∅', Main, Ord),
        "varnothing" => sym('∅', Ams, Ord),
        "forall" => sym('∀', Main, Ord),
        "exists" => sym('∃', Main, Ord),
        "nexists" => sym('∄', Ams, Ord),
        "neg" | "lnot" => sym('¬', Main, Ord),
        "aleph" => sym('ℵ', Main, Ord),
        "Re" => sym('ℜ', Main, Ord),
        "Im" => sym('ℑ', Main, Ord),
        "wp" => sym('℘', Main, Ord),
        "angle" => sym('∠', Main, Ord),
        "triangle" => sym('△', Main, Ord),
        "prime" => sym('′', Main, Ord),
        "top" => sym('⊤', Main, Ord),
        "bot" => sym('⊥', Main, Ord),
        "degree" => sym('°', Main, Ord),
        "checkmark" => sym('✓', Ams, Ord),
        "square" | "Box" => sym('□', Ams, Ord),
        "ldots" | "dots" | "dotsc" | "dotsb" => sym('…', Main, Inner),
        "cdots" => sym('⋯', Main, Inner),
        "vdots" => sym('⋮', Main, Ord),
        "ddots" => sym('⋱', Main, Inner),
        "backslash" => sym('\\', Main, Ord),
        "vert" => sym('∣', Main, Ord),
        "Vert" => sym('∥', Main, Ord),
        "lvert" => sym('∣', Main, Open),
        "rvert" => sym('∣', Main, Close),
        "lVert" => sym('∥', Main, Open),
        "rVert" => sym('∥', Main, Close),
        "langle" => sym('⟨', Main, Open),
        "rangle" => sym('⟩', Main, Close),
        "lfloor" => sym('⌊', Main, Open),
        "rfloor" => sym('⌋', Main, Close),
        "lceil" => sym('⌈', Main, Open),
        "rceil" => sym('⌉', Main, Close),
        "lbrace" => sym('{', Main, Open),
        "rbrace" => sym('}', Main, Close),
        "lbrack" => sym('[', Main, Open),
        "rbrack" => sym(']', Main, Close),
        // binary operators
        "pm" => sym('±', Main, Bin),
        "mp" => sym('∓', Main, Bin),
        "times" => sym('×', Main, Bin),
        "div" => sym('÷', Main, Bin),
        "cdot" => sym('⋅', Main, Bin),
        "ast" => sym('∗', Main, Bin),
        "star" => sym('⋆', Main, Bin),
        "circ" => sym('∘', Main, Bin),
        "bullet" => sym('∙', Main, Bin),
        "oplus" => sym('⊕', Main, Bin),
        "ominus" => sym('⊖', Main, Bin),
        "otimes" => sym('⊗', Main, Bin),
        "oslash" => sym('⊘', Main, Bin),
        "odot" => sym('⊙', Main, Bin),
        "wedge" | "land" => sym('∧', Main, Bin),
        "vee" | "lor" => sym('∨', Main, Bin),
        "cap" => sym('∩', Main, Bin),
        "cup" => sym('∪', Main, Bin),
        "sqcap" => sym('⊓', Main, Bin),
        "sqcup" => sym('⊔', Main, Bin),
        "uplus" => sym('⊎', Main, Bin),
        "setminus" => sym('∖', Main, Bin),
        "smallsetminus" => sym('∖', Ams, Bin),
        "dagger" => sym('†', Main, Bin),
        "ddagger" => sym('‡', Main, Bin),
        "diamond" => sym('⋄', Main, Bin),
        "wr" => sym('≀', Main, Bin),
        // relations
        "le" | "leq" => sym('≤', Main, Rel),
        "ge" | "geq" => sym('≥', Main, Rel),
        "leqslant" => sym('⩽', Ams, Rel),
        "geqslant" => sym('⩾', Ams, Rel),
        "ne" | "neq" => Some(Neg('=')),
        "notin" => Some(Neg('∈')),
        "approx" => sym('≈', Main, Rel),
        "approxeq" => sym('≊', Ams, Rel),
        "equiv" => sym('≡', Main, Rel),
        "sim" => sym('∼', Main, Rel),
        "simeq" => sym('≃', Main, Rel),
        "cong" => sym('≅', Main, Rel),
        "propto" => sym('∝', Main, Rel),
        "in" => sym('∈', Main, Rel),
        "ni" | "owns" => sym('∋', Main, Rel),
        "subset" => sym('⊂', Main, Rel),
        "supset" => sym('⊃', Main, Rel),
        "subseteq" => sym('⊆', Main, Rel),
        "supseteq" => sym('⊇', Main, Rel),
        "subsetneq" => sym('⊊', Ams, Rel),
        "supsetneq" => sym('⊋', Ams, Rel),
        "sqsubseteq" => sym('⊑', Main, Rel),
        "sqsupseteq" => sym('⊒', Main, Rel),
        "to" | "rightarrow" => sym('→', Main, Rel),
        "gets" | "leftarrow" => sym('←', Main, Rel),
        "leftrightarrow" => sym('↔', Main, Rel),
        "Rightarrow" => sym('⇒', Main, Rel),
        "Leftarrow" => sym('⇐', Main, Rel),
        "Leftrightarrow" => sym('⇔', Main, Rel),
        "longrightarrow" => sym('⟶', Main, Rel),
        "longleftarrow" => sym('⟵', Main, Rel),
        "longleftrightarrow" => sym('⟷', Main, Rel),
        "Longrightarrow" | "implies" => sym('⟹', Main, Rel),
        "Longleftarrow" | "impliedby" => sym('⟸', Main, Rel),
        "Longleftrightarrow" | "iff" => sym('⟺', Main, Rel),
        "mapsto" => sym('↦', Main, Rel),
        "longmapsto" => sym('⟼', Main, Rel),
        "uparrow" => sym('↑', Main, Rel),
        "downarrow" => sym('↓', Main, Rel),
        "updownarrow" => sym('↕', Main, Rel),
        "Uparrow" => sym('⇑', Main, Rel),
        "Downarrow" => sym('⇓', Main, Rel),
        "nearrow" => sym('↗', Main, Rel),
        "searrow" => sym('↘', Main, Rel),
        "swarrow" => sym('↙', Main, Rel),
        "nwarrow" => sym('↖', Main, Rel),
        "hookrightarrow" => sym('↪', Main, Rel),
        "hookleftarrow" => sym('↩', Main, Rel),
        "rightleftharpoons" => sym('⇌', Main, Rel),
        "ll" => sym('≪', Main, Rel),
        "gg" => sym('≫', Main, Rel),
        "perp" => sym('⊥', Main, Rel),
        "parallel" => sym('∥', Main, Rel),
        "mid" => sym('∣', Main, Rel),
        "prec" => sym('≺', Main, Rel),
        "succ" => sym('≻', Main, Rel),
        "preceq" => sym('⪯', Main, Rel),
        "succeq" => sym('⪰', Main, Rel),
        "vdash" => sym('⊢', Main, Rel),
        "dashv" => sym('⊣', Main, Rel),
        "models" => sym('⊨', Main, Rel),
        "asymp" => sym('≍', Main, Rel),
        "doteq" => sym('≐', Main, Rel),
        "lesssim" => sym('≲', Ams, Rel),
        "gtrsim" => sym('≳', Ams, Rel),
        "triangleq" => sym('≜', Ams, Rel),
        "therefore" => sym('∴', Ams, Rel),
        "because" => sym('∵', Ams, Rel),
        "nleq" => sym('≰', Ams, Rel),
        "ngeq" => sym('≱', Ams, Rel),
        "nless" => sym('≮', Ams, Rel),
        "ngtr" => sym('≯', Ams, Rel),
        "nmid" => sym('∤', Ams, Rel),
        "nparallel" => sym('∦', Ams, Rel),
        "nsim" => sym('≁', Ams, Rel),
        // large operators
        "sum" => Some(BigOp('∑', true)),
        "prod" => Some(BigOp('∏', true)),
        "coprod" => Some(BigOp('∐', true)),
        "bigcup" => Some(BigOp('⋃', true)),
        "bigcap" => Some(BigOp('⋂', true)),
        "bigvee" => Some(BigOp('⋁', true)),
        "bigwedge" => Some(BigOp('⋀', true)),
        "bigoplus" => Some(BigOp('⨁', true)),
        "bigotimes" => Some(BigOp('⨂', true)),
        "bigodot" => Some(BigOp('⨀', true)),
        "bigsqcup" => Some(BigOp('⨆', true)),
        "int" => Some(BigOp('∫', false)),
        "iint" => Some(BigOp('∬', false)),
        "iiint" => Some(BigOp('∭', false)),
        "oint" => Some(BigOp('∮', false)),
        // functions
        "sin" | "cos" | "tan" | "cot" | "sec" | "csc" | "arcsin" | "arccos" | "arctan" | "sinh" | "cosh" | "tanh" | "coth" | "exp" | "ln" | "log" | "lg"
        | "arg" | "deg" | "dim" | "ker" | "hom" | "tr" | "rank" | "span" | "diag" | "sgn" | "sign" | "erf" | "Var" | "Cov" => {
            FUNCTIONS.iter().find(|f| **f == name).map(|f| Func(f, false))
        }
        "lim" | "max" | "min" | "sup" | "inf" | "det" | "gcd" | "Pr" => FUNCTIONS.iter().find(|f| **f == name).map(|f| Func(f, true)),
        "liminf" => Some(Func("lim inf", true)),
        "limsup" => Some(Func("lim sup", true)),
        "argmax" => Some(Func("arg max", true)),
        "argmin" => Some(Func("arg min", true)),
        // accents
        "hat" => Some(Accent('\u{2C6}')),
        "check" => Some(Accent('\u{2C7}')),
        "tilde" => Some(Accent('\u{2DC}')),
        "bar" => Some(Accent('\u{2C9}')),
        "vec" => Some(Accent('\u{20D7}')),
        "dot" => Some(Accent('\u{2D9}')),
        "ddot" => Some(Accent('\u{A8}')),
        "breve" => Some(Accent('\u{2D8}')),
        "acute" => Some(Accent('\u{2CA}')),
        "grave" => Some(Accent('\u{2CB}')),
        "mathring" => Some(Accent('\u{2DA}')),
        "overline" => Some(Overline),
        "underline" => Some(Underline),
        // structures
        "frac" => Some(Frac(None)),
        "dfrac" => Some(Frac(Some(0))),
        "tfrac" => Some(Frac(Some(1))),
        "binom" => Some(Binom(None)),
        "dbinom" => Some(Binom(Some(0))),
        "tbinom" => Some(Binom(Some(1))),
        "sqrt" => Some(Sqrt),
        "left" => Some(Left),
        "right" => Some(Right),
        "middle" => Some(Middle),
        "big" => Some(Big(1.2, Ord)),
        "Big" => Some(Big(1.8, Ord)),
        "bigg" => Some(Big(2.4, Ord)),
        "Bigg" => Some(Big(3.0, Ord)),
        "bigl" => Some(Big(1.2, Open)),
        "Bigl" => Some(Big(1.8, Open)),
        "biggl" => Some(Big(2.4, Open)),
        "Biggl" => Some(Big(3.0, Open)),
        "bigr" => Some(Big(1.2, Close)),
        "Bigr" => Some(Big(1.8, Close)),
        "biggr" => Some(Big(2.4, Close)),
        "Biggr" => Some(Big(3.0, Close)),
        "bigm" => Some(Big(1.2, Rel)),
        "Bigm" => Some(Big(1.8, Rel)),
        "biggm" => Some(Big(2.4, Rel)),
        "Biggm" => Some(Big(3.0, Rel)),
        "begin" => Some(Begin),
        "end" => Some(End),
        "text" | "textrm" | "mbox" => Some(Text(false)),
        "textbf" => Some(Text(true)),
        "operatorname" => Some(OperatorName),
        "mathrm" => Some(Font(self::Font::Roman)),
        "mathbf" => Some(Font(self::Font::Bold)),
        "mathbb" => Some(Font(self::Font::Blackboard)),
        "mathit" | "mathnormal" => Some(Font(self::Font::Italic)),
        // space
        "," | "thinspace" => Some(Space(3.0)),
        ":" | ">" | "medspace" => Some(Space(4.0)),
        ";" | "thickspace" => Some(Space(5.0)),
        "!" | "negthinspace" => Some(Space(-3.0)),
        " " => Some(Space(4.5)),
        "enspace" => Some(Space(9.0)),
        "quad" => Some(Space(18.0)),
        "qquad" => Some(Space(36.0)),
        // the rest
        "textcolor" => Some(TextColor),
        "color" => Some(Color),
        "not" => Some(Not),
        "limits" => Some(Limits(true)),
        "nolimits" => Some(Limits(false)),
        "displaystyle" => Some(Style(0)),
        "textstyle" => Some(Style(1)),
        "scriptstyle" => Some(Style(2)),
        "scriptscriptstyle" => Some(Style(3)),
        "$" => sym('$', Main, Ord),
        "%" => sym('%', Main, Ord),
        "#" => sym('#', Main, Ord),
        "&" => sym('&', Main, Ord),
        "_" => sym('_', Main, Ord),
        "{" => sym('{', Main, Open),
        "}" => sym('}', Main, Close),
        "|" => sym('∥', Main, Ord),
        _ => None,
    }
}

/// The names of the functions written in upright letters (as `\name`).
const FUNCTIONS: [&str; 39] = [
    "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "coth", "exp", "ln", "log", "lg", "arg", "deg", "dim",
    "ker", "hom", "tr", "rank", "span", "diag", "sgn", "sign", "erf", "Var", "Cov", "lim", "max", "min", "sup", "inf", "det", "gcd", "Pr",
];

/// Is `word` the name of a function that has a command of its own (`\sin`)?
pub fn is_function(word: &str) -> bool {
    FUNCTIONS.contains(&word)
}

/// What a character typed straight into a formula is (letters and digits apart, which
/// [`letter`] sets): its glyph, face and class. ≠ and ∉ are relations with a stroke
/// through them ([`Cmd::Neg`]).
pub fn typed(c: char) -> Option<Cmd> {
    use Class::*;
    use Face::*;
    let sym = |c: char, face: Face, class: Class| Some(Cmd::Sym(c, face, class));
    match c {
        '+' | '±' | '∓' | '×' | '÷' | '∘' | '∗' | '⊕' | '⊖' | '⊗' | '⊘' | '⊙' | '∧' | '∨' | '∩' | '∪' | '⊓' | '⊔' | '⊎' | '∖' | '⋆' | '∙' | '†' | '‡' | '⋄' => {
            sym(c, Main, Bin)
        }
        '-' | '−' => sym('−', Main, Bin),
        '*' => sym('∗', Main, Bin),
        '·' | '⋅' => sym('⋅', Main, Bin),
        '•' => sym('∙', Main, Bin),
        '=' | '<' | '>' | '≤' | '≥' | '≈' | '≡' | '∼' | '≃' | '≅' | '∝' | '∈' | '∋' | '⊂' | '⊃' | '⊆' | '⊇' | '⊑' | '⊒' | '→' | '←' | '↔' | '⇒' | '⇐' | '⇔'
        | '⟶' | '⟵' | '⟷' | '⟹' | '⟸' | '⟺' | '↦' | '⟼' | '↑' | '↓' | '↕' | '⇑' | '⇓' | '↗' | '↘' | '↙' | '↖' | '↪' | '↩' | '⇌' | '≪' | '≫' | '⊥' | '∥'
        | '∣' | '≺' | '≻' | '⪯' | '⪰' | '⊢' | '⊣' | '⊨' | '≍' | '≐' | ':' => sym(c, Main, Rel),
        '⩽' | '⩾' | '≊' | '⊊' | '⊋' | '≲' | '≳' | '≜' | '∴' | '∵' | '≰' | '≱' | '≮' | '≯' | '∤' | '∦' | '≁' => {
            sym(c, Ams, Rel)
        }
        '≠' => Some(Cmd::Neg('=')),
        '∉' => Some(Cmd::Neg('∈')),
        '≢' => Some(Cmd::Neg('≡')),
        '⊄' => Some(Cmd::Neg('⊂')),
        '(' | '[' | '⟨' | '⌊' | '⌈' => sym(c, Main, Open),
        ')' | ']' | '⟩' | '⌋' | '⌉' | '!' | '?' => sym(c, Main, Close),
        ',' | ';' => sym(c, Main, Punct),
        '.' | '/' | '|' | '∞' | '∂' | '∇' | '∅' | '∀' | '∃' | '¬' | 'ℓ' | 'ℏ' | '∠' | '△' | '⊤' | '°' | '%' | '#' | '@' | 'ℵ' | 'ℜ' | 'ℑ' | '℘' | '′' => {
            sym(c, Main, Ord)
        }
        '‖' => sym('∥', Main, Ord),
        '…' => sym('…', Main, Inner),
        '⋯' => sym('⋯', Main, Inner),
        '⋮' => sym('⋮', Main, Ord),
        '⋱' => sym('⋱', Main, Inner),
        'ℝ' => sym('R', Ams, Ord),
        'ℕ' => sym('N', Ams, Ord),
        'ℤ' => sym('Z', Ams, Ord),
        'ℚ' => sym('Q', Ams, Ord),
        'ℂ' => sym('C', Ams, Ord),
        'ℙ' => sym('P', Ams, Ord),
        '𝔼' => sym('E', Ams, Ord),
        'ı' => sym('\u{E131}', MathItalic, Ord),
        'ȷ' => sym('\u{E237}', MathItalic, Ord),
        '∑' | '∏' | '∐' | '⋃' | '⋂' | '⋁' | '⋀' | '⨁' | '⨂' | '⨀' | '⨆' => Some(Cmd::BigOp(c, true)),
        '∫' | '∬' | '∭' | '∮' => Some(Cmd::BigOp(c, false)),
        _ => None,
    }
}

/// A superscript character (as typed: x², A⁻¹, Aᵀ) and what it stands for.
pub fn superscript(c: char) -> Option<&'static str> {
    Some(match c {
        '⁰' => "0",
        '¹' => "1",
        '²' => "2",
        '³' => "3",
        '⁴' => "4",
        '⁵' => "5",
        '⁶' => "6",
        '⁷' => "7",
        '⁸' => "8",
        '⁹' => "9",
        '⁺' => "+",
        '⁻' => "-",
        '⁼' => "=",
        '⁽' => "(",
        '⁾' => ")",
        'ⁱ' => "i",
        'ⁿ' => "n",
        'ᵃ' => "a",
        'ᵇ' => "b",
        'ᶜ' => "c",
        'ᵈ' => "d",
        'ᵉ' => "e",
        'ᶠ' => "f",
        'ᵍ' => "g",
        'ʰ' => "h",
        'ʲ' => "j",
        'ᵏ' => "k",
        'ˡ' => "l",
        'ᵐ' => "m",
        'ᵒ' => "o",
        'ᵖ' => "p",
        'ʳ' => "r",
        'ˢ' => "s",
        'ᵗ' => "t",
        'ᵘ' => "u",
        'ᵛ' => "v",
        'ʷ' => "w",
        'ˣ' => "x",
        'ʸ' => "y",
        'ᶻ' => "z",
        'ᴬ' => "A",
        'ᴮ' => "B",
        'ᴰ' => "D",
        'ᴱ' => "E",
        'ᴳ' => "G",
        'ᴴ' => "H",
        'ᴵ' => "I",
        'ᴶ' => "J",
        'ᴷ' => "K",
        'ᴸ' => "L",
        'ᴹ' => "M",
        'ᴺ' => "N",
        'ᴼ' => "O",
        'ᴾ' => "P",
        'ᴿ' => "R",
        'ᵀ' => "\\top ",
        'ᵁ' => "U",
        'ⱽ' => "V",
        'ᵂ' => "W",
        'ᵅ' => "α",
        'ᵝ' => "β",
        'ᵞ' => "γ",
        'ᵟ' => "δ",
        'ᵋ' => "ε",
        'ᶿ' => "θ",
        'ᶥ' => "ι",
        'ᵠ' => "φ",
        'ᵡ' => "χ",
        _ => return None,
    })
}

/// A subscript character (x₁, σₖ₊₁) and what it stands for.
pub fn subscript(c: char) -> Option<&'static str> {
    Some(match c {
        '₀' => "0",
        '₁' => "1",
        '₂' => "2",
        '₃' => "3",
        '₄' => "4",
        '₅' => "5",
        '₆' => "6",
        '₇' => "7",
        '₈' => "8",
        '₉' => "9",
        '₊' => "+",
        '₋' => "-",
        '₌' => "=",
        '₍' => "(",
        '₎' => ")",
        'ₐ' => "a",
        'ₑ' => "e",
        'ₒ' => "o",
        'ₓ' => "x",
        'ₕ' => "h",
        'ₖ' => "k",
        'ₗ' => "l",
        'ₘ' => "m",
        'ₙ' => "n",
        'ₚ' => "p",
        'ₛ' => "s",
        'ₜ' => "t",
        'ᵢ' => "i",
        'ⱼ' => "j",
        'ᵣ' => "r",
        'ᵤ' => "u",
        'ᵥ' => "v",
        'ᵦ' => "β",
        'ᵧ' => "γ",
        'ᵨ' => "ρ",
        'ᵩ' => "φ",
        'ᵪ' => "χ",
        _ => return None,
    })
}

/// An accent written as a combining character after its letter (x̂, x̄, v⃗), as the
/// accent command's glyph.
pub fn combining(c: char) -> Option<char> {
    Some(match c {
        '\u{302}' => '\u{2C6}',
        '\u{303}' => '\u{2DC}',
        '\u{304}' | '\u{305}' => '\u{2C9}',
        '\u{307}' => '\u{2D9}',
        '\u{308}' => '\u{A8}',
        '\u{30C}' => '\u{2C7}',
        '\u{20D7}' => '\u{20D7}',
        _ => return None,
    })
}

/// A letter with its accent in one character (î, ĵ, ā, ẋ): the letter (dotless, for i
/// and j) and the accent.
pub fn accented(c: char) -> Option<(char, char)> {
    let (base, accent) = match c {
        'î' => ('ı', '\u{302}'),
        'ĵ' => ('ȷ', '\u{302}'),
        'â' | 'ê' | 'ô' | 'û' | 'ŷ' | 'ẑ' | 'ĉ' | 'ĝ' | 'ĥ' | 'ŝ' | 'ŵ' | 'Â' | 'Ê' | 'Ô' | 'Û' | 'Ŷ' | 'Ẑ' => (unaccent(c)?, '\u{302}'),
        'ā' | 'ē' | 'ō' | 'ū' | 'ȳ' | 'Ā' | 'Ē' | 'Ō' | 'Ū' | 'Ȳ' => (unaccent(c)?, '\u{304}'),
        'ī' => ('ı', '\u{304}'),
        'ã' | 'ẽ' | 'õ' | 'ũ' | 'ỹ' | 'ñ' | 'Ã' | 'Õ' | 'Ñ' => (unaccent(c)?, '\u{303}'),
        'ȧ' | 'ḃ' | 'ċ' | 'ḋ' | 'ė' | 'ḟ' | 'ġ' | 'ḣ' | 'ṁ' | 'ṅ' | 'ȯ' | 'ṗ' | 'ṙ' | 'ṡ' | 'ṫ' | 'ẇ' | 'ẋ' | 'ẏ' | 'ż' => {
            (unaccent(c)?, '\u{307}')
        }
        'ä' | 'ë' | 'ö' | 'ü' | 'ÿ' | 'ẍ' | 'Ä' | 'Ë' | 'Ö' | 'Ü' => (unaccent(c)?, '\u{308}'),
        'ï' => ('ı', '\u{308}'),
        _ => return None,
    };
    Some((base, combining(accent)?))
}

/// The letter under an accent, for the letters [`accented`] knows.
fn unaccent(c: char) -> Option<char> {
    let lower = "âêôûŷẑĉĝĥŝŵāēōūȳãẽõũỹñȧḃċḋėḟġḣṁṅȯṗṙṡṫẇẋẏżäëöüÿẍ";
    let plain = "aeouyzcghswaeouyaeouynabcdefghmnoprstwxyzaeouyx";
    let upper = "ÂÊÔÛŶẐĀĒŌŪȲÃÕÑÄËÖÜ";
    let upper_plain = "AEOUYZAEOUYAONAEOU";
    if let Some(i) = lower.chars().position(|l| l == c) {
        return plain.chars().nth(i);
    }
    upper.chars().position(|l| l == c).and_then(|i| upper_plain.chars().nth(i))
}

/// A colour by name, for `\textcolor{name}{…}`: the palette's names (the colours the
/// scenes draw in), or `#rrggbb`.
pub fn color(name: &str) -> Option<egui::Color32> {
    use crate::gfx::color::pal;
    let named = match name {
        "text" => pal::TEXT,
        "dim" => pal::DIM,
        "faint" => pal::FAINT,
        "muted" => pal::MUTED,
        "link" => pal::LINK,
        "cyan" => pal::CYAN,
        "blue" => pal::BLUE,
        "pink" => pal::PINK,
        "yellow" => pal::YELLOW,
        "green" => pal::GREEN,
        "orange" => pal::ORANGE,
        "purple" => pal::PURPLE,
        "red" => pal::RED,
        "white" => pal::WHITE,
        _ => {
            let hex = name.strip_prefix('#')?;
            if !(hex.len() == 6 || hex.len() == 8) || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
                return None;
            }
            let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).ok();
            let alpha = if hex.len() == 8 { byte(6)? } else { 255 };
            return Some(egui::Color32::from_rgba_unmultiplied(byte(0)?, byte(2)?, byte(4)?, alpha));
        }
    };
    Some(named.egui())
}

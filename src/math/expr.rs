//! A small expression compiler: text such as `sin(x)^2 + a*x*y` becomes stack bytecode.
//!
//! The same bytecode can be run three ways:
//! * `eval` on plain numbers,
//! * `eval_batch` on whole arrays at once (one tight loop per operation, so nothing
//!   is decided per number; the loops of plain arithmetic, sqrt and abs vectorise,
//!   while sin, exp and the other functions call the library for each number: this
//!   is what fills surfaces and fields every frame),
//! * `eval_num` on any [`Num`], which gives exact derivatives: [`Dual`] carries two
//!   partial derivatives, [`Jet`] carries a whole truncated Taylor series.

use std::f32::consts::{E, PI, TAU};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Func {
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Tanh,
    Exp,
    Ln,
    Log10,
    Log2,
    Sqrt,
    Abs,
    Sign,
    Floor,
    Ceil,
    Relu,
    Sigmoid,
    Gauss,
}

const FUNCS: &[(&str, Func)] = &[
    ("sin", Func::Sin),
    ("cos", Func::Cos),
    ("tan", Func::Tan),
    ("asin", Func::Asin),
    ("acos", Func::Acos),
    ("atan", Func::Atan),
    ("sinh", Func::Sinh),
    ("cosh", Func::Cosh),
    ("tanh", Func::Tanh),
    ("exp", Func::Exp),
    ("ln", Func::Ln),
    ("log", Func::Ln),
    ("log10", Func::Log10),
    ("log2", Func::Log2),
    ("sqrt", Func::Sqrt),
    ("abs", Func::Abs),
    ("sign", Func::Sign),
    ("floor", Func::Floor),
    ("ceil", Func::Ceil),
    ("relu", Func::Relu),
    ("sigmoid", Func::Sigmoid),
    ("gauss", Func::Gauss),
];

/// The functions of two arguments.
const FUNCS2: &[(&str, Op)] = &[("min", Op::Min), ("max", Op::Max), ("pow", Op::Pow), ("atan2", Op::Atan2)];

const CONSTANTS: &[(&str, f32)] = &[("pi", PI), ("tau", TAU), ("e", E)];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Const(f32),
    Var(u8),
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Powi(i32),
    Neg,
    Min,
    Max,
    Atan2,
    Call(Func),
}

/// The arithmetic an expression needs. Implemented for numbers and for the
/// derivative-carrying types below.
pub trait Num: Copy {
    fn cst(v: f32) -> Self;
    fn value(self) -> f32;
    fn add(self, o: Self) -> Self;
    fn sub(self, o: Self) -> Self;
    fn mul(self, o: Self) -> Self;
    fn div(self, o: Self) -> Self;
    fn neg(self) -> Self;
    fn call(self, f: Func) -> Self;

    fn powi(self, n: i32) -> Self {
        let mut base = self;
        let mut e = n.unsigned_abs();
        let mut acc = Self::cst(1.0);
        while e > 0 {
            if e & 1 == 1 {
                acc = acc.mul(base);
            }
            base = base.mul(base);
            e >>= 1;
        }
        if n < 0 {
            Self::cst(1.0).div(acc)
        } else {
            acc
        }
    }

    /// `self ^ o`. The general rule, for a positive base; the types override it to
    /// cover what else has a meaning (negative bases with whole exponents, zero).
    fn pow(self, o: Self) -> Self {
        o.mul(self.call(Func::Ln)).call(Func::Exp)
    }

    /// The angle of the point (o, self) from the positive x axis: `atan2(self, o)`.
    fn atan2(self, o: Self) -> Self;

    /// The smaller of two, by value. As with numbers, a NaN loses to anything.
    fn min(self, o: Self) -> Self {
        if self.value() <= o.value() || o.value().is_nan() {
            self
        } else {
            o
        }
    }

    /// The larger of two, likewise.
    fn max(self, o: Self) -> Self {
        if self.value() >= o.value() || o.value().is_nan() {
            self
        } else {
            o
        }
    }
}

#[inline]
fn call_f32(f: Func, x: f32) -> f32 {
    match f {
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
        Func::Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        Func::Floor => x.floor(),
        Func::Ceil => x.ceil(),
        Func::Relu => x.max(0.0),
        Func::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Func::Gauss => (-x * x).exp(),
    }
}

/// Derivative of `f` at `x`, given `y = f(x)`.
#[inline]
fn deriv_f32(f: Func, x: f32, y: f32) -> f32 {
    match f {
        Func::Sin => x.cos(),
        Func::Cos => -x.sin(),
        Func::Tan => 1.0 + y * y,
        Func::Asin => 1.0 / (1.0 - x * x).sqrt(),
        Func::Acos => -1.0 / (1.0 - x * x).sqrt(),
        Func::Atan => 1.0 / (1.0 + x * x),
        Func::Sinh => x.cosh(),
        Func::Cosh => x.sinh(),
        Func::Tanh => 1.0 - y * y,
        Func::Exp => y,
        Func::Ln => 1.0 / x,
        Func::Log10 => 1.0 / (x * std::f32::consts::LN_10),
        Func::Log2 => 1.0 / (x * std::f32::consts::LN_2),
        Func::Sqrt => 0.5 / y,
        Func::Abs => {
            if x >= 0.0 {
                1.0
            } else {
                -1.0
            }
        }
        Func::Sign | Func::Floor | Func::Ceil => 0.0,
        Func::Relu => {
            if x > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        Func::Sigmoid => y * (1.0 - y),
        Func::Gauss => -2.0 * x * y,
    }
}

impl Num for f32 {
    #[inline]
    fn cst(v: f32) -> Self {
        v
    }
    #[inline]
    fn value(self) -> f32 {
        self
    }
    #[inline]
    fn add(self, o: Self) -> Self {
        self + o
    }
    #[inline]
    fn sub(self, o: Self) -> Self {
        self - o
    }
    #[inline]
    fn mul(self, o: Self) -> Self {
        self * o
    }
    #[inline]
    fn div(self, o: Self) -> Self {
        self / o
    }
    #[inline]
    fn neg(self) -> Self {
        -self
    }
    #[inline]
    fn call(self, f: Func) -> Self {
        call_f32(f, self)
    }
    #[inline]
    fn powi(self, n: i32) -> Self {
        f32::powi(self, n)
    }
    #[inline]
    fn pow(self, o: Self) -> Self {
        self.powf(o)
    }
    #[inline]
    fn atan2(self, o: Self) -> Self {
        f32::atan2(self, o)
    }
    #[inline]
    fn min(self, o: Self) -> Self {
        f32::min(self, o)
    }
    #[inline]
    fn max(self, o: Self) -> Self {
        f32::max(self, o)
    }
}

/// A value with two partial derivatives (forward-mode automatic differentiation).
#[derive(Clone, Copy, Debug, Default)]
pub struct Dual {
    pub v: f32,
    pub dx: f32,
    pub dy: f32,
}

impl Dual {
    pub fn var_x(v: f32) -> Dual {
        Dual { v, dx: 1.0, dy: 0.0 }
    }
    pub fn var_y(v: f32) -> Dual {
        Dual { v, dx: 0.0, dy: 1.0 }
    }
}

impl Dual {
    /// The slope `k` along a direction in which the argument changes at rate `d`:
    /// nothing, where it does not change at all, however steep `k` is.
    #[inline]
    fn along(k: f32, d: f32) -> f32 {
        if d == 0.0 {
            0.0
        } else {
            k * d
        }
    }
}

impl Num for Dual {
    fn cst(v: f32) -> Self {
        Dual { v, dx: 0.0, dy: 0.0 }
    }
    fn value(self) -> f32 {
        self.v
    }
    fn add(self, o: Self) -> Self {
        Dual { v: self.v + o.v, dx: self.dx + o.dx, dy: self.dy + o.dy }
    }
    fn sub(self, o: Self) -> Self {
        Dual { v: self.v - o.v, dx: self.dx - o.dx, dy: self.dy - o.dy }
    }
    fn mul(self, o: Self) -> Self {
        Dual { v: self.v * o.v, dx: self.dx * o.v + self.v * o.dx, dy: self.dy * o.v + self.v * o.dy }
    }
    fn div(self, o: Self) -> Self {
        let inv = 1.0 / o.v;
        let q = self.v * inv;
        Dual { v: q, dx: (self.dx - q * o.dx) * inv, dy: (self.dy - q * o.dy) * inv }
    }
    fn neg(self) -> Self {
        Dual { v: -self.v, dx: -self.dx, dy: -self.dy }
    }
    /// (The slope is taken only along what varies: where the function is infinitely
    /// steep, as sqrt at 0, a direction it does not depend on still has slope 0.)
    fn call(self, f: Func) -> Self {
        let y = call_f32(f, self.v);
        let d = deriv_f32(f, self.v, y);
        Dual { v: y, dx: Dual::along(d, self.dx), dy: Dual::along(d, self.dy) }
    }
    /// d(a^b) = b a^(b-1) da + a^b ln(a) db. Each term is taken only where its
    /// factor varies, so that a constant exponent works for any base the power is
    /// defined for (negative bases with whole exponents; zero), and an infinite
    /// slope never meets a zero to make a NaN. (Nor does a zero exponent meet the
    /// infinite 0^(-1): the slope of x⁰ is 0 everywhere.)
    fn pow(self, o: Self) -> Self {
        let v = self.v.powf(o.v);
        let by_base = if (self.dx != 0.0 || self.dy != 0.0) && o.v != 0.0 { o.v * self.v.powf(o.v - 1.0) } else { 0.0 };
        let by_exponent = if (o.dx != 0.0 || o.dy != 0.0) && v != 0.0 { v * self.v.ln() } else { 0.0 };
        Dual { v, dx: Dual::along(by_base, self.dx) + Dual::along(by_exponent, o.dx), dy: Dual::along(by_base, self.dy) + Dual::along(by_exponent, o.dy) }
    }
    /// d atan2(a, b) = (b da - a db) / (a² + b²): finite everywhere but the origin,
    /// where the angle has no slope to speak of (zero is returned).
    fn atan2(self, o: Self) -> Self {
        let r2 = self.v * self.v + o.v * o.v;
        let (ka, kb) = if r2 > 0.0 { (o.v / r2, -self.v / r2) } else { (0.0, 0.0) };
        Dual { v: self.v.atan2(o.v), dx: ka * self.dx + kb * o.dx, dy: ka * self.dy + kb * o.dy }
    }
}

/// Highest Taylor order a [`Jet`] carries.
pub const JET_ORDER: usize = 14;
const JN: usize = JET_ORDER + 1;

/// A truncated Taylor series `c[0] + c[1] h + c[2] h^2 + ...`. Evaluating an
/// expression on `Jet::var(x0)` yields every Taylor coefficient of the function at
/// `x0` in one pass.
#[derive(Clone, Copy, Debug)]
pub struct Jet {
    pub c: [f32; JN],
}

impl Jet {
    pub fn var(x0: f32) -> Jet {
        let mut c = [0.0; JN];
        c[0] = x0;
        c[1] = 1.0;
        Jet { c }
    }

    /// Evaluate the series truncated after order `n` at offset `h`.
    pub fn eval(&self, h: f32, n: usize) -> f32 {
        let n = n.min(JET_ORDER);
        let mut acc = 0.0;
        for k in (0..=n).rev() {
            acc = acc * h + self.c[k];
        }
        acc
    }

    /// (sin, cos) of a series, by the coupled recurrence.
    fn sin_cos(self, hyperbolic: bool) -> (Jet, Jet) {
        let a = &self.c;
        let mut s = [0.0; JN];
        let mut c = [0.0; JN];
        if hyperbolic {
            s[0] = a[0].sinh();
            c[0] = a[0].cosh();
        } else {
            s[0] = a[0].sin();
            c[0] = a[0].cos();
        }
        let sign = if hyperbolic { 1.0 } else { -1.0 };
        for k in 1..JN {
            let (mut ss, mut cc) = (0.0, 0.0);
            for j in 1..=k {
                ss += j as f32 * a[j] * c[k - j];
                cc += j as f32 * a[j] * s[k - j];
            }
            s[k] = ss / k as f32;
            c[k] = sign * cc / k as f32;
        }
        (Jet { c: s }, Jet { c })
    }

    /// Integrate `self' * g` term by term: the series whose derivative is `a' g`.
    fn integrate_times(self, g: &Jet, c0: f32) -> Jet {
        let a = &self.c;
        let mut out = [0.0; JN];
        out[0] = c0;
        for k in 1..JN {
            let mut acc = 0.0;
            for j in 1..=k {
                acc += j as f32 * a[j] * g.c[k - j];
            }
            out[k] = acc / k as f32;
        }
        Jet { c: out }
    }
}

impl Jet {
    /// The series `y` with `y' = (p + q y + r y²) a'` and `y = y0` at the centre,
    /// where `a` is this series: tanh is (1, 0, -1) and the sigmoid (0, 1, -1).
    fn riccati(self, y0: f32, p: f32, q: f32, r: f32) -> Jet {
        let a = &self.c;
        let (mut y, mut w) = ([0.0; JN], [0.0; JN]);
        y[0] = y0;
        w[0] = p + q * y0 + r * y0 * y0;
        for k in 1..JN {
            let mut acc = 0.0;
            for j in 1..=k {
                acc += j as f32 * a[j] * w[k - j];
            }
            y[k] = acc / k as f32;
            let mut square = 0.0;
            for j in 0..=k {
                square += y[j] * y[k - j];
            }
            w[k] = q * y[k] + r * square;
        }
        Jet { c: y }
    }
}

impl Num for Jet {
    fn cst(v: f32) -> Self {
        let mut c = [0.0; JN];
        c[0] = v;
        Jet { c }
    }
    fn value(self) -> f32 {
        self.c[0]
    }
    fn add(self, o: Self) -> Self {
        let mut c = self.c;
        for k in 0..JN {
            c[k] += o.c[k];
        }
        Jet { c }
    }
    fn sub(self, o: Self) -> Self {
        let mut c = self.c;
        for k in 0..JN {
            c[k] -= o.c[k];
        }
        Jet { c }
    }
    fn mul(self, o: Self) -> Self {
        let mut c = [0.0; JN];
        for k in 0..JN {
            let mut acc = 0.0;
            for j in 0..=k {
                acc += self.c[j] * o.c[k - j];
            }
            c[k] = acc;
        }
        Jet { c }
    }
    fn div(self, o: Self) -> Self {
        let mut c = [0.0; JN];
        let inv = 1.0 / o.c[0];
        for k in 0..JN {
            let mut acc = self.c[k];
            for j in 1..=k {
                acc -= o.c[j] * c[k - j];
            }
            c[k] = acc * inv;
        }
        Jet { c }
    }
    fn neg(self) -> Self {
        let mut c = self.c;
        for v in &mut c {
            *v = -*v;
        }
        Jet { c }
    }
    fn call(self, f: Func) -> Self {
        let a = &self.c;
        let one = Jet::cst(1.0);
        match f {
            Func::Exp => {
                let mut e = [0.0; JN];
                e[0] = a[0].exp();
                for k in 1..JN {
                    let mut acc = 0.0;
                    for j in 1..=k {
                        acc += j as f32 * a[j] * e[k - j];
                    }
                    e[k] = acc / k as f32;
                }
                Jet { c: e }
            }
            Func::Ln => {
                // (ln a)' = a' / a
                self.integrate_times(&one.div(self), a[0].ln())
            }
            Func::Log10 => self.call(Func::Ln).mul(Jet::cst(1.0 / std::f32::consts::LN_10)),
            Func::Log2 => self.call(Func::Ln).mul(Jet::cst(1.0 / std::f32::consts::LN_2)),
            Func::Sin => self.sin_cos(false).0,
            Func::Cos => self.sin_cos(false).1,
            Func::Tan => {
                let (s, c) = self.sin_cos(false);
                s.div(c)
            }
            Func::Sinh => self.sin_cos(true).0,
            Func::Cosh => self.sin_cos(true).1,
            // t' = (1 - t²) a', order by order (sinh / cosh would overflow to
            // infinity over infinity for a large argument)
            Func::Tanh => self.riccati(a[0].tanh(), 1.0, 0.0, -1.0),
            Func::Sqrt => {
                // s*s = a  =>  s_k = (a_k - sum_{j=1}^{k-1} s_j s_{k-j}) / (2 s_0)
                let mut s = [0.0; JN];
                s[0] = a[0].sqrt();
                for k in 1..JN {
                    let mut acc = a[k];
                    for j in 1..k {
                        acc -= s[j] * s[k - j];
                    }
                    s[k] = acc / (2.0 * s[0]);
                }
                Jet { c: s }
            }
            Func::Atan => self.integrate_times(&one.div(one.add(self.mul(self))), a[0].atan()),
            Func::Asin => {
                let g = one.div(one.sub(self.mul(self)).call(Func::Sqrt));
                self.integrate_times(&g, a[0].asin())
            }
            Func::Acos => {
                let g = one.div(one.sub(self.mul(self)).call(Func::Sqrt)).neg();
                self.integrate_times(&g, a[0].acos())
            }
            // s' = (s - s²) a', likewise
            Func::Sigmoid => self.riccati(call_f32(Func::Sigmoid, a[0]), 0.0, 1.0, -1.0),
            Func::Gauss => self.mul(self).neg().call(Func::Exp),
            Func::Abs => {
                if a[0] >= 0.0 {
                    self
                } else {
                    self.neg()
                }
            }
            Func::Relu => {
                if a[0] > 0.0 {
                    self
                } else {
                    Jet::cst(0.0)
                }
            }
            Func::Sign | Func::Floor | Func::Ceil => Jet::cst(call_f32(f, a[0])),
        }
    }
    fn pow(self, o: Self) -> Self {
        let r = o.c[0];
        if o.c[1..].iter().all(|&c| c == 0.0) {
            // a constant exponent: a whole one by multiplication (exact, any base)
            if r.fract() == 0.0 && r.abs() < 1e6 {
                return self.powi(r as i32);
            }
            // otherwise from p' a = r a' p, which also covers a negative base
            // wherever the power itself is a number
            let a = &self.c;
            if a[0] != 0.0 {
                let mut p = [0.0; JN];
                p[0] = a[0].powf(r);
                for k in 1..JN {
                    let mut acc = 0.0;
                    for j in 1..=k {
                        acc += (r * j as f32 - (k - j) as f32) * a[j] * p[k - j];
                    }
                    p[k] = acc / (k as f32 * a[0]);
                }
                return Jet { c: p };
            }
        }
        o.mul(self.call(Func::Ln)).call(Func::Exp)
    }
    /// From (atan2(a, b))' = (b a' - a b') / (a² + b²).
    fn atan2(self, o: Self) -> Self {
        let r2 = self.mul(self).add(o.mul(o));
        let from_a = self.integrate_times(&o.div(r2), self.c[0].atan2(o.c[0]));
        let from_b = o.integrate_times(&self.div(r2), 0.0);
        from_a.sub(from_b)
    }
}

/// A compiled expression.
#[derive(Clone, Debug)]
pub struct Expr {
    ops: Vec<Op>,
    depth: usize,
    uses: u32,
}

/// Scratch registers for [`Expr::eval_batch`], reused between calls.
#[derive(Default)]
pub struct Batch {
    regs: Vec<Vec<f32>>,
}

const MAX_STACK: usize = 24;

impl Expr {
    /// Compile `src`. `vars` lists the variable names; their order is the slot order
    /// used when evaluating.
    pub fn parse(src: &str, vars: &[&str]) -> Result<Expr, String> {
        let tokens = split_names(lex(src)?, vars);
        let mut p = Parser { t: &tokens, i: 0, vars, ops: Vec::new(), depth: 0 };
        p.expr(0)?;
        if p.i < tokens.len() {
            return Err(format!("unexpected '{}'", tokens[p.i].text()));
        }
        let ops = fold(p.ops);
        // required stack depth, and which variables are read
        let (mut d, mut depth, mut uses) = (0usize, 0usize, 0u32);
        for op in &ops {
            match op {
                Op::Const(_) => d += 1,
                Op::Var(i) => {
                    d += 1;
                    uses |= 1 << i;
                }
                Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Pow | Op::Min | Op::Max | Op::Atan2 => d -= 1,
                Op::Neg | Op::Powi(_) | Op::Call(_) => {}
            }
            depth = depth.max(d);
        }
        if depth > MAX_STACK {
            return Err("expression is too deeply nested".into());
        }
        Ok(Expr { ops, depth, uses })
    }

    /// Does the expression read variable slot `i`?
    pub fn uses(&self, i: usize) -> bool {
        self.uses & (1 << i) != 0
    }

    #[inline]
    pub fn eval(&self, vars: &[f32]) -> f32 {
        self.eval_num::<f32>(vars)
    }

    /// Evaluate on any numeric type (see [`Dual`], [`Jet`]).
    pub fn eval_num<T: Num>(&self, vars: &[T]) -> T {
        let mut st = [T::cst(0.0); MAX_STACK];
        let mut n = 0usize;
        for op in &self.ops {
            match *op {
                Op::Const(c) => {
                    st[n] = T::cst(c);
                    n += 1;
                }
                Op::Var(i) => {
                    st[n] = vars.get(i as usize).copied().unwrap_or(T::cst(0.0));
                    n += 1;
                }
                Op::Neg => st[n - 1] = st[n - 1].neg(),
                Op::Powi(e) => st[n - 1] = st[n - 1].powi(e),
                Op::Call(f) => st[n - 1] = st[n - 1].call(f),
                bin => {
                    let b = st[n - 1];
                    let a = st[n - 2];
                    n -= 1;
                    st[n - 1] = match bin {
                        Op::Add => a.add(b),
                        Op::Sub => a.sub(b),
                        Op::Mul => a.mul(b),
                        Op::Div => a.div(b),
                        Op::Pow => a.pow(b),
                        Op::Min => a.min(b),
                        Op::Max => a.max(b),
                        Op::Atan2 => a.atan2(b),
                        _ => unreachable!(),
                    };
                }
            }
        }
        st[0]
    }

    /// Value and gradient with respect to variable slots `ix` and `iy`.
    pub fn grad2(&self, vars: &[f32], ix: usize, iy: usize) -> (f32, f32, f32) {
        let mut d = [Dual::default(); 8];
        let n = vars.len().min(8);
        for i in 0..n {
            d[i] = Dual::cst(vars[i]);
        }
        if ix < n {
            d[ix].dx = 1.0;
        }
        if iy < n {
            d[iy].dy = 1.0;
        }
        let r = self.eval_num::<Dual>(&d[..n]);
        (r.v, r.dx, r.dy)
    }

    /// Taylor coefficients in variable slot `ix` around `vars[ix]`.
    pub fn taylor(&self, vars: &[f32], ix: usize) -> Jet {
        let mut j = [Jet::cst(0.0); 8];
        let n = vars.len().min(8);
        for i in 0..n {
            j[i] = Jet::cst(vars[i]);
        }
        if ix < n {
            j[ix] = Jet::var(vars[ix]);
        }
        self.eval_num::<Jet>(&j[..n])
    }

    /// Evaluate for `out.len()` points at once. `vars[i]` is either one value per
    /// point or a single value shared by all of them.
    pub fn eval_batch(&self, vars: &[&[f32]], out: &mut [f32], batch: &mut Batch) {
        let len = out.len();
        if batch.regs.len() < self.depth {
            batch.regs.resize_with(self.depth, Vec::new);
        }
        for r in &mut batch.regs[..self.depth] {
            if r.len() != len {
                r.resize(len, 0.0);
            }
        }
        let regs = &mut batch.regs;
        let mut n = 0usize;
        for op in &self.ops {
            match *op {
                Op::Const(c) => {
                    regs[n].fill(c);
                    n += 1;
                }
                Op::Var(i) => {
                    match vars.get(i as usize) {
                        Some(v) if v.len() == len => regs[n].copy_from_slice(v),
                        Some(v) if !v.is_empty() => regs[n].fill(v[0]),
                        _ => regs[n].fill(0.0),
                    }
                    n += 1;
                }
                Op::Neg => regs[n - 1].iter_mut().for_each(|x| *x = -*x),
                Op::Powi(e) => match e {
                    2 => regs[n - 1].iter_mut().for_each(|x| *x *= *x),
                    3 => regs[n - 1].iter_mut().for_each(|x| *x = *x * *x * *x),
                    _ => regs[n - 1].iter_mut().for_each(|x| *x = x.powi(e)),
                },
                Op::Call(f) => {
                    let r = &mut regs[n - 1];
                    // one loop per function, so that the function is chosen once per
                    // array rather than once per number (only the sqrt and abs loops
                    // vectorise: the others call the library for each number)
                    macro_rules! map {
                        ($e:expr) => {
                            r.iter_mut().for_each(|x| *x = $e(*x))
                        };
                    }
                    match f {
                        Func::Sin => map!(f32::sin),
                        Func::Cos => map!(f32::cos),
                        Func::Exp => map!(f32::exp),
                        Func::Sqrt => map!(f32::sqrt),
                        Func::Abs => map!(f32::abs),
                        Func::Tanh => map!(f32::tanh),
                        Func::Ln => map!(f32::ln),
                        other => map!(|x| call_f32(other, x)),
                    }
                }
                bin => {
                    let (lo, hi) = regs.split_at_mut(n - 1);
                    let a = &mut lo[n - 2];
                    let b = &hi[0];
                    macro_rules! zip {
                        ($e:expr) => {
                            a.iter_mut().zip(b.iter()).for_each(|(x, y)| *x = $e(*x, *y))
                        };
                    }
                    match bin {
                        Op::Add => zip!(|x, y| x + y),
                        Op::Sub => zip!(|x, y| x - y),
                        Op::Mul => zip!(|x, y| x * y),
                        Op::Div => zip!(|x, y| x / y),
                        Op::Pow => zip!(f32::powf),
                        Op::Min => zip!(f32::min),
                        Op::Max => zip!(f32::max),
                        Op::Atan2 => zip!(f32::atan2),
                        _ => unreachable!(),
                    }
                    n -= 1;
                }
            }
        }
        out.copy_from_slice(&regs[0]);
    }
}

/// Evaluate constant sub-expressions at compile time.
fn fold(ops: Vec<Op>) -> Vec<Op> {
    let mut out: Vec<Op> = Vec::with_capacity(ops.len());
    for op in ops {
        let n = out.len();
        match op {
            Op::Neg | Op::Powi(_) | Op::Call(_) => {
                if let Some(Op::Const(a)) = out.last().copied() {
                    let v = match op {
                        Op::Neg => -a,
                        Op::Powi(e) => a.powi(e),
                        Op::Call(f) => call_f32(f, a),
                        _ => unreachable!(),
                    };
                    out[n - 1] = Op::Const(v);
                    continue;
                }
            }
            Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Pow | Op::Min | Op::Max | Op::Atan2 if n >= 2 => {
                if let (Op::Const(a), Op::Const(b)) = (out[n - 2], out[n - 1]) {
                    let v = match op {
                        Op::Add => a + b,
                        Op::Sub => a - b,
                        Op::Mul => a * b,
                        Op::Div => a / b,
                        Op::Pow => a.powf(b),
                        Op::Min => a.min(b),
                        Op::Max => a.max(b),
                        Op::Atan2 => a.atan2(b),
                        _ => unreachable!(),
                    };
                    out.truncate(n - 2);
                    out.push(Op::Const(v));
                    continue;
                }
                // x ^ small integer: exact, fast, and valid for negative bases
                if op == Op::Pow {
                    if let Op::Const(b) = out[n - 1] {
                        if b.fract() == 0.0 && b.abs() <= 64.0 {
                            out[n - 1] = Op::Powi(b as i32);
                            continue;
                        }
                        if b == 0.5 {
                            out[n - 1] = Op::Call(Func::Sqrt);
                            continue;
                        }
                    }
                }
            }
            _ => {}
        }
        out.push(op);
    }
    out
}

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f32),
    Ident(String),
    Sym(char),
}

impl Tok {
    fn text(&self) -> String {
        match self {
            Tok::Num(v) => v.to_string(),
            Tok::Ident(s) => s.clone(),
            Tok::Sym(c) => c.to_string(),
        }
    }
}

/// The digit a superscript stands for: x² is x^2, as on paper.
fn superscript(c: char) -> Option<u32> {
    "⁰¹²³⁴⁵⁶⁷⁸⁹".chars().position(|s| s == c).map(|d| d as u32)
}

/// Signs as they are printed (the Hermite scenes write their formulas so) read as the
/// keys of a keyboard: the minus sign as -, the multiplication sign and dots as *.
fn typed(c: char) -> char {
    match c {
        '\u{2212}' => '-',
        '×' | '·' | '⋅' => '*',
        c => c,
    }
}

fn lex(src: &str) -> Result<Vec<Tok>, String> {
    let chars: Vec<char> = src.chars().map(typed).collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            // An exponent only if a digit follows the e, straight away or after a sign:
            // 2e-1 is 0.2 and 2e+3x is 2000x, as on every calculator (2e-1 could also
            // be read as 2e minus 1; this is a choice, made on purpose and tested),
            // while in 2e, 2e-x and 2e - 1 (spaced) the e is the constant.
            if i < chars.len() && (chars[i] == 'e' || chars[i] == 'E') {
                let mut j = i + 1;
                if j < chars.len() && (chars[j] == '+' || chars[j] == '-') {
                    j += 1;
                }
                if j < chars.len() && chars[j].is_ascii_digit() {
                    while j < chars.len() && chars[j].is_ascii_digit() {
                        j += 1;
                    }
                    i = j;
                }
            }
            let s: String = chars[start..i].iter().collect();
            out.push(Tok::Num(s.parse().map_err(|_| format!("bad number '{s}'"))?));
        } else if superscript(c).is_some() || (c == '⁻' && chars.get(i + 1).is_some_and(|&d| superscript(d).is_some())) {
            // x² is x^2, and x⁻¹ is x^-1
            out.push(Tok::Sym('^'));
            if c == '⁻' {
                out.push(Tok::Sym('-'));
                i += 1;
            }
            let mut n = 0.0f32;
            while let Some(d) = chars.get(i).and_then(|&c| superscript(c)) {
                n = n * 10.0 + d as f32;
                i += 1;
            }
            out.push(Tok::Num(n));
        } else if c == 'π' {
            // (before the test for letters, which π is)
            out.push(Tok::Ident("pi".into()));
            i += 1;
        } else if c.is_alphabetic() || c == '_' {
            let start = i;
            // (π is a letter to the test for letters, and ² a digit, but each stands on
            // its own: xπ is x times π, and x² is x squared)
            while i < chars.len() && ((chars[i].is_alphanumeric() && chars[i] != 'π' && superscript(chars[i]).is_none()) || chars[i] == '_') {
                i += 1;
            }
            out.push(Tok::Ident(chars[start..i].iter().collect()));
        } else if "+-*/^(),".contains(c) {
            out.push(Tok::Sym(c));
            i += 1;
        } else {
            return Err(format!("unexpected character '{c}'"));
        }
    }
    Ok(out)
}

/// Names written side by side, as on paper: `ax` is a x, `2xy` is 2 x y, `xsin(x)` is
/// x sin(x). A name that is not known as it stands is read as known names one after
/// another, at each step the longest that fits: the formula's variables, the
/// constants, and a function's name if it ends the name and a bracket follows. A known
/// name is never split (`exp` is not e x p, nor `asin` a sin), and a name that cannot
/// be read this way is left whole, for the parser to report.
fn split_names(tokens: Vec<Tok>, vars: &[&str]) -> Vec<Tok> {
    let plain = || vars.iter().copied().chain(CONSTANTS.iter().map(|c| c.0)).filter(|n| !n.is_empty());
    let functions = || FUNCS.iter().map(|f| f.0).chain(FUNCS2.iter().map(|f| f.0));
    let mut out = Vec::with_capacity(tokens.len());
    for (k, tok) in tokens.iter().enumerate() {
        let name = match tok {
            Tok::Ident(name) if !plain().chain(functions()).any(|n| n == name.as_str()) => name,
            _ => {
                out.push(tok.clone());
                continue;
            }
        };
        let call = tokens.get(k + 1) == Some(&Tok::Sym('('));
        let (mut rest, mut parts) = (name.as_str(), Vec::new());
        while !rest.is_empty() {
            let last = |n: &str| call && n.len() == rest.len();
            let Some(n) = plain().chain(functions().filter(|&n| last(n))).filter(|n| rest.starts_with(n)).max_by_key(|n| n.len()) else {
                break;
            };
            parts.push(Tok::Ident(n.to_string()));
            rest = &rest[n.len()..];
        }
        if rest.is_empty() {
            out.extend(parts);
        } else {
            out.push(tok.clone());
        }
    }
    out
}

struct Parser<'a> {
    t: &'a [Tok],
    i: usize,
    vars: &'a [&'a str],
    ops: Vec<Op>,
    /// How many `expr` calls are open: brackets, minus signs, powers and function
    /// calls each nest one deeper.
    depth: usize,
}

/// Deeper nesting than this is refused, so that no text typed into a formula box
/// can overflow the stack (which, in a browser, would end the program).
const MAX_NESTING: usize = 200;

impl Parser<'_> {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(&Tok::Sym(c)) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    /// Precedence climbing. Binding powers: + - (1), * / and implicit product (2),
    /// unary minus (3), ^ (4, right associative).
    fn expr(&mut self, min_bp: u8) -> Result<(), String> {
        self.depth += 1;
        if self.depth > MAX_NESTING {
            return Err("expression is too deeply nested".into());
        }
        let result = self.expr_at(min_bp);
        self.depth -= 1;
        result
    }

    fn expr_at(&mut self, min_bp: u8) -> Result<(), String> {
        self.prefix()?;
        loop {
            let (op, bp, right) = match self.peek() {
                Some(Tok::Sym('+')) => (Some(Op::Add), 1, false),
                Some(Tok::Sym('-')) => (Some(Op::Sub), 1, false),
                Some(Tok::Sym('*')) => (Some(Op::Mul), 2, false),
                Some(Tok::Sym('/')) => (Some(Op::Div), 2, false),
                Some(Tok::Sym('^')) => (Some(Op::Pow), 4, true),
                // implicit multiplication: 2x, 3(x+1), (a)(b), x y
                Some(Tok::Num(_)) | Some(Tok::Ident(_)) | Some(Tok::Sym('(')) => (None, 2, false),
                _ => break,
            };
            if bp < min_bp {
                break;
            }
            if op.is_some() {
                self.i += 1;
            }
            self.expr(if right { bp } else { bp + 1 })?;
            self.ops.push(op.unwrap_or(Op::Mul));
        }
        Ok(())
    }

    fn prefix(&mut self) -> Result<(), String> {
        match self.peek().cloned() {
            Some(Tok::Sym('-')) => {
                self.i += 1;
                self.expr(3)?;
                self.ops.push(Op::Neg);
            }
            Some(Tok::Sym('+')) => {
                self.i += 1;
                self.expr(3)?;
            }
            Some(Tok::Num(v)) => {
                self.i += 1;
                self.ops.push(Op::Const(v));
            }
            Some(Tok::Sym('(')) => {
                self.i += 1;
                self.expr(0)?;
                if !self.eat(')') {
                    return Err("missing ')'".into());
                }
            }
            Some(Tok::Ident(name)) => {
                self.i += 1;
                self.ident(&name)?;
            }
            Some(t) => return Err(format!("unexpected '{}'", t.text())),
            None => return Err("unexpected end of expression".into()),
        }
        Ok(())
    }

    fn ident(&mut self, name: &str) -> Result<(), String> {
        if let Some(i) = self.vars.iter().position(|v| *v == name) {
            self.ops.push(Op::Var(i as u8));
            return Ok(());
        }
        if let Some(&(_, v)) = CONSTANTS.iter().find(|(n, _)| *n == name) {
            self.ops.push(Op::Const(v));
            return Ok(());
        }
        let two = FUNCS2.iter().find(|(n, _)| *n == name).map(|(_, op)| *op);
        let one = FUNCS.iter().find(|(n, _)| *n == name).map(|(_, f)| *f);
        if two.is_none() && one.is_none() {
            // `sinx` is most likely a function written without its brackets
            let bare = FUNCS.iter().map(|f| f.0).filter(|f| name.len() > f.len() && name.starts_with(f)).max_by_key(|f| f.len());
            return Err(match bare {
                Some(f) => format!("unknown name '{name}': did you mean {f}({})?", &name[f.len()..]),
                None => format!("unknown name '{name}'"),
            });
        }
        if !self.eat('(') {
            return Err(format!("'{name}' needs parentheses"));
        }
        self.expr(0)?;
        if let Some(op) = two {
            if !self.eat(',') {
                return Err(format!("'{name}' takes two arguments"));
            }
            self.expr(0)?;
            self.ops.push(op);
        } else if let Some(f) = one {
            self.ops.push(Op::Call(f));
        }
        if !self.eat(')') {
            return Err("missing ')'".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(src: &str, x: f32, y: f32) -> f32 {
        Expr::parse(src, &["x", "y"]).unwrap().eval(&[x, y])
    }

    #[test]
    fn arithmetic_and_precedence() {
        assert_eq!(ev("1 + 2 * 3", 0.0, 0.0), 7.0);
        assert_eq!(ev("(1 + 2) * 3", 0.0, 0.0), 9.0);
        assert_eq!(ev("2 ^ 3 ^ 2", 0.0, 0.0), 512.0);
        assert_eq!(ev("-x^2", 3.0, 0.0), -9.0);
        assert_eq!(ev("2x + 3y", 1.0, 2.0), 8.0);
        assert_eq!(ev("2(x+1)(y-1)", 1.0, 3.0), 8.0);
        assert_eq!(ev("x^-1", 4.0, 0.0), 0.25);
        assert!((ev("sin(pi/2) + cos(0)", 0.0, 0.0) - 2.0).abs() < 1e-6);
        assert!((ev("max(x, y) - min(x, y)", 2.0, 5.0) - 3.0).abs() < 1e-6);
        assert!((ev("2e", 0.0, 0.0) - 2.0 * E).abs() < 1e-5);
        assert!((ev("1.5e2", 0.0, 0.0) - 150.0).abs() < 1e-4);
    }

    #[test]
    fn errors() {
        assert!(Expr::parse("sin(", &["x"]).is_err());
        assert!(Expr::parse("x +", &["x"]).is_err());
        assert!(Expr::parse("foo(x)", &["x"]).is_err());
        assert!(Expr::parse("x $ 2", &["x"]).is_err());
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let e = Expr::parse("sin(x*y) + exp(-x^2) * tanh(y) + sqrt(x^2 + y^2 + 1) / (2 + cos(y))", &["x", "y"]).unwrap();
        let (x, y) = (0.7f32, -1.3f32);
        let (v, dx, dy) = e.grad2(&[x, y], 0, 1);
        let h = 1e-3;
        let fx = (e.eval(&[x + h, y]) - e.eval(&[x - h, y])) / (2.0 * h);
        let fy = (e.eval(&[x, y + h]) - e.eval(&[x, y - h])) / (2.0 * h);
        assert!((v - e.eval(&[x, y])).abs() < 1e-6);
        assert!((dx - fx).abs() < 2e-3, "{dx} vs {fx}");
        assert!((dy - fy).abs() < 2e-3, "{dy} vs {fy}");
    }

    #[test]
    fn taylor_series_reproduce_known_expansions() {
        // exp(x) at 0: 1/k!
        let j = Expr::parse("exp(x)", &["x"]).unwrap().taylor(&[0.0], 0);
        let mut fact = 1.0;
        for k in 0..8 {
            if k > 0 {
                fact *= k as f32;
            }
            assert!((j.c[k] - 1.0 / fact).abs() < 1e-5);
        }
        // sin(x) at 0: x - x^3/6 + x^5/120
        let j = Expr::parse("sin(x)", &["x"]).unwrap().taylor(&[0.0], 0);
        assert!((j.c[1] - 1.0).abs() < 1e-6 && (j.c[3] + 1.0 / 6.0).abs() < 1e-6 && (j.c[5] - 1.0 / 120.0).abs() < 1e-6);
        // 1/(1-x) at 0: all ones
        let j = Expr::parse("1/(1-x)", &["x"]).unwrap().taylor(&[0.0], 0);
        assert!(j.c.iter().all(|c| (c - 1.0).abs() < 1e-5));
        // ln(1+x): (-1)^(k+1)/k ; atan(x): x - x^3/3 + x^5/5
        let j = Expr::parse("ln(1+x)", &["x"]).unwrap().taylor(&[0.0], 0);
        assert!((j.c[4] + 0.25).abs() < 1e-6);
        let j = Expr::parse("atan(x)", &["x"]).unwrap().taylor(&[0.0], 0);
        assert!((j.c[3] + 1.0 / 3.0).abs() < 1e-6 && (j.c[5] - 0.2).abs() < 1e-6);
        // a composite function away from zero agrees with direct evaluation nearby
        let e = Expr::parse("sqrt(1 + x^2) * cos(2x) + tanh(x)", &["x"]).unwrap();
        let j = e.taylor(&[0.4], 0);
        for h in [-0.2f32, 0.1, 0.25] {
            assert!((j.eval(h, JET_ORDER) - e.eval(&[0.4 + h])).abs() < 1e-4);
        }
    }

    #[test]
    fn batch_matches_scalar() {
        let e = Expr::parse("sin(x) * cos(y*a) + x^3 - abs(y) / (1 + x^2)", &["x", "y", "a"]).unwrap();
        let xs: Vec<f32> = (0..64).map(|i| i as f32 * 0.13 - 4.0).collect();
        let ys: Vec<f32> = (0..64).map(|i| 3.0 - i as f32 * 0.09).collect();
        let mut out = vec![0.0; 64];
        e.eval_batch(&[&xs, &ys, &[1.7]], &mut out, &mut Batch::default());
        for i in 0..64 {
            let s = e.eval(&[xs[i], ys[i], 1.7]);
            assert!((out[i] - s).abs() < 1e-4, "{} vs {}", out[i], s);
        }
    }

    /// Hostile input: nesting far deeper than any formula, in each of the ways the
    /// grammar can nest. It must be refused, not overflow the stack.
    #[test]
    fn deep_nesting_is_refused() {
        for n in [20usize, 100_000] {
            let cases = [
                format!("{}x{}", "(".repeat(n), ")".repeat(n)),
                format!("{}x", "-".repeat(n)),
                vec!["x"; n].join("^"),
                format!("{}x{}", "sin(".repeat(n), ")".repeat(n)),
            ];
            for src in cases {
                let parsed = Expr::parse(&src, &["x"]);
                assert_eq!(parsed.is_ok(), n == 20, "{} levels of {:.12}...", n, src);
            }
        }
    }

    #[test]
    fn pi_can_be_written_as_a_letter() {
        assert!((ev("2π", 0.0, 0.0) - std::f32::consts::TAU).abs() < 1e-6);
        assert!(ev("sin(π)", 0.0, 0.0).abs() < 1e-6);
        // also straight after a name, where it used to be read as part of the name
        assert!((ev("xπ", 2.0, 0.0) - std::f32::consts::TAU).abs() < 1e-6);
        assert!((ev("2xπ + πy", 1.0, 1.0) - 3.0 * std::f32::consts::PI).abs() < 1e-5);
    }

    /// Derivatives and Taylor coefficients where the old shortcuts gave NaN: powers
    /// of negative numbers and of zero, atan2 on the axes, min and max with a NaN,
    /// tanh and the sigmoid of large arguments.
    #[test]
    fn derivatives_at_awkward_points() {
        let grad = |src: &str, vars: &[f32]| Expr::parse(src, &["x", "y", "a"]).unwrap().grad2(vars, 0, 1);
        let close =
            |got: (f32, f32, f32), want: (f32, f32, f32)| (got.0 - want.0).abs() < 1e-4 && (got.1 - want.1).abs() < 1e-4 && (got.2 - want.2).abs() < 1e-4;
        // x^a with the exponent a slider: d/dx x^3 = 3 x² at x = -2
        assert!(close(grad("x^a", &[-2.0, 0.0, 3.0]), (-8.0, 12.0, 0.0)), "{:?}", grad("x^a", &[-2.0, 0.0, 3.0]));
        assert!(close(grad("x^1.5", &[0.0, 0.0, 0.0]), (0.0, 0.0, 0.0)));
        assert!(close(grad("(x^2+y^2)^1.5", &[0.0, 0.0, 0.0]), (0.0, 0.0, 0.0)));
        assert!(close(grad("x^65", &[-1.0, 0.0, 0.0]), (-1.0, 65.0, 0.0)));
        assert!(close(grad("2^x", &[3.0, 0.0, 0.0]), (8.0, 8.0 * 2f32.ln(), 0.0)));
        // x^a with the slider at 0 is the constant 1, also at x = 0
        assert!(close(grad("x^a", &[0.0, 0.0, 0.0]), (1.0, 0.0, 0.0)), "{:?}", grad("x^a", &[0.0, 0.0, 0.0]));
        // where a function is infinitely steep, what it does not depend on still has
        // slope 0: d/dy of sqrt(x) + y at x = 0 is 1 (sqrt, ln and asin alike)
        for (src, at) in [("sqrt(x) + y", 0.0), ("x^0.5 + y", 0.0), ("ln(x) + y", 0.0), ("asin(x) + y", 1.0)] {
            let (_, _, dy) = grad(src, &[at, 0.0, 0.0]);
            assert_eq!(dy, 1.0, "{src}");
        }
        // atan2(y, x) at (0, 1): the angle is a quarter turn and falls as x grows
        assert!(close(grad("atan2(y, x)", &[0.0, 1.0, 0.0]), (std::f32::consts::FRAC_PI_2, -1.0, 0.0)));
        assert!(close(grad("atan2(y, x)", &[-1.0, 0.0, 0.0]), (std::f32::consts::PI, 0.0, -1.0)));
        assert_eq!(ev("atan2(y, x)", 0.0, 0.0), 0.0);
        // a NaN loses to a number, in every way of evaluating
        let e = Expr::parse("min(1, sqrt(x)) + max(2, sqrt(x))", &["x"]).unwrap();
        assert_eq!(e.eval(&[-1.0]), 3.0);
        assert_eq!(e.grad2(&[-1.0], 0, 0).0, 3.0);

        let taylor = |src: &str, x: f32| Expr::parse(src, &["x", "a"]).unwrap().taylor(&[x, 3.0], 0).c;
        let t = taylor("tanh(100x)", 1.0);
        assert!((t[0] - 1.0).abs() < 1e-6 && t[1..].iter().all(|c| c.abs() < 1e-6), "{t:?}");
        let s = taylor("sigmoid(-100x)", 1.0);
        assert!(s.iter().all(|c| c.abs() < 1e-6), "{s:?}");
        // and where nothing overflows they still agree with the textbook series:
        // tanh h = h - h³/3 + 2 h⁵/15, sigmoid h = 1/2 + h/4 - h³/48
        let t = taylor("tanh(x)", 0.0);
        assert!((t[1] - 1.0).abs() < 1e-6 && (t[3] + 1.0 / 3.0).abs() < 1e-6 && (t[5] - 2.0 / 15.0).abs() < 1e-6, "{t:?}");
        let s = taylor("sigmoid(x)", 0.0);
        assert!((s[0] - 0.5).abs() < 1e-6 && (s[1] - 0.25).abs() < 1e-6 && (s[3] + 1.0 / 48.0).abs() < 1e-6, "{s:?}");
        // (1 + h)^2.5 = 1 + 2.5 h + 1.875 h² + ...; (-2 + h)^3 by multiplication
        let p = taylor("x^2.5", 1.0);
        assert!((p[1] - 2.5).abs() < 1e-5 && (p[2] - 1.875).abs() < 1e-5, "{p:?}");
        // (h - 2)³ = -8 + 12 h - 6 h² + h³, the exponent being a variable set to 3
        let p = taylor("x^a", -2.0);
        assert!((p[0] + 8.0).abs() < 1e-4 && (p[1] - 12.0).abs() < 1e-4 && (p[2] + 6.0).abs() < 1e-4 && (p[3] - 1.0).abs() < 1e-4, "{p:?}");
    }

    /// `2e-1` is 0.2, as on every calculator: an exponent is digits straight after the
    /// e, with at most a sign between. With a space, or no digit, e is the constant.
    #[test]
    fn an_exponent_is_digits_straight_after_the_e() {
        let close = |src: &str, x: f32, want: f32| assert!((ev(src, x, 0.0) - want).abs() < 1e-5, "{src}: {} vs {want}", ev(src, x, 0.0));
        close("2e-1", 0.0, 0.2);
        close("2E+2", 0.0, 200.0);
        close("2e+3x", 0.5, 1000.0);
        close("2e - 1", 0.0, 2.0 * E - 1.0);
        close("2e -1", 0.0, 2.0 * E - 1.0);
        close("2e-x", 0.5, 2.0 * E - 0.5);
        close("2e", 0.0, 2.0 * E);
    }

    /// Names side by side are products, as on paper, with the variables of the graph
    /// scene (x, t, a, b); a name that is known as it stands is never split.
    #[test]
    fn names_written_side_by_side() {
        let graph = ["x", "t", "a", "b"];
        let (x, t, a, b) = (0.7f32, 0.3f32, 1.9f32, -1.3f32);
        let at = |src: &str, vars: &[&str], values: &[f32]| Expr::parse(src, vars).map(|e| e.eval(values)).unwrap_or_else(|e| panic!("{src}: {e}"));
        let close = |src: &str, want: f32| {
            let got = at(src, &graph, &[x, t, a, b]);
            assert!((got - want).abs() < 1e-5 * (1.0 + want.abs()), "{src}: {got} vs {want}");
        };
        close("sin(ax)", (a * x).sin());
        close("2ax", 2.0 * a * x);
        close("ax^2", a * x * x);
        close("2ax^2 - bx + t", 2.0 * a * x * x - b * x + t);
        close("xsin(x)", x * x.sin());
        close("bexp(-x)", b * (-x).exp());
        close("atx", a * t * x);
        close("ex", E * x);
        close("xpi", x * PI);
        close("2pix", 2.0 * PI * x);
        close("taux", TAU * x);
        close("xmax(a, b)", x * a);
        // known names are not split, even where every letter is a variable
        let letters = ["s", "i", "n", "x"];
        assert_eq!(at("sin(x)", &letters, &[2.0, 3.0, 5.0, 0.5]), 0.5f32.sin());
        assert_eq!(at("sinx", &letters, &[2.0, 3.0, 5.0, 0.5]), 15.0);
        assert_eq!(Expr::parse("exp", &["x", "p"]).unwrap_err(), "'exp' needs parentheses");
        assert_eq!(at("exp(x)", &["x", "p"], &[0.5, 3.0]), 0.5f32.exp());
        // a variable named like the start of a function: a, and asin, abs
        assert_eq!(at("asin(x)", &graph, &[0.5, t, a, b]), 0.5f32.asin());
        assert_eq!(at("abs(a)", &graph, &[x, t, -2.0, b]), 2.0);
        assert_eq!(at("a", &graph, &[x, t, a, b]), a);
        let longer = ["co", "x"];
        assert_eq!(at("cos(x)", &longer, &[3.0, 0.5]), 0.5f32.cos());
        assert_eq!(at("cox", &longer, &[3.0, 0.5]), 1.5);
        // what cannot be read is named
        for (src, error) in [
            ("foo", "unknown name 'foo'"),
            ("ay", "unknown name 'ay'"),
            ("sinx", "unknown name 'sinx': did you mean sin(x)?"),
            ("2sqrtx", "unknown name 'sqrtx': did you mean sqrt(x)?"),
            ("foo(x)", "unknown name 'foo'"),
        ] {
            assert_eq!(Expr::parse(src, &graph).unwrap_err(), error);
        }
        // and a name thousands of letters long is read in a moment
        let long = "ab".repeat(50_000);
        assert!(Expr::parse(&long, &graph).is_ok() && Expr::parse(&format!("{long}y"), &graph).is_err());
    }

    /// Signs as they are printed: x², x⁻¹, the minus sign and the multiplication signs.
    /// Every Hermite polynomial as the Hermite scenes write it can be typed back in.
    #[test]
    fn signs_as_printed() {
        let close =
            |src: &str, x: f32, y: f32, want: f32| assert!((ev(src, x, y) - want).abs() < 1e-5 * (1.0 + want.abs()), "{src}: {} vs {want}", ev(src, x, y));
        close("x²", 3.0, 0.0, 9.0);
        close("x²y", 3.0, 2.0, 18.0);
        close("-x²", 3.0, 0.0, -9.0);
        close("2x³ − x", 2.0, 0.0, 14.0);
        close("x¹⁰", 2.0, 0.0, 1024.0);
        close("x⁻¹ + y⁻²", 4.0, 2.0, 0.5);
        close("2·x × y ⋅ 3", 2.0, 5.0, 60.0);
        close("e⁻²", 0.0, 0.0, (-2.0f32).exp());
        for n in 0..=10 {
            let formula = crate::math::hermite::he_formula(n);
            let e = Expr::parse(&formula, &["x"]).unwrap_or_else(|e| panic!("{formula}: {e}"));
            for x in [-1.3f32, 0.4, 2.1] {
                let want = crate::math::hermite::he(n, x as f64) as f32;
                assert!((e.eval(&[x]) - want).abs() < 1e-5 * (1.0 + want.abs()), "{formula} at {x}");
            }
        }
        for bad in ["²", "x⁻", "x^²", "⁻¹"] {
            assert!(Expr::parse(bad, &["x"]).is_err(), "{bad}");
        }
    }
}

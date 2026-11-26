//! Complex numbers, for the scenes that add up rotating quantities.

use std::ops::{Add, Div, Mul, Neg, Sub};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C32 {
    pub re: f32,
    pub im: f32,
}

impl C32 {
    pub const ZERO: C32 = C32 { re: 0.0, im: 0.0 };
    pub const ONE: C32 = C32 { re: 1.0, im: 0.0 };

    pub const fn new(re: f32, im: f32) -> Self {
        C32 { re, im }
    }

    /// `r e^{i theta}`.
    pub fn polar(r: f32, theta: f32) -> Self {
        C32 { re: r * theta.cos(), im: r * theta.sin() }
    }

    pub fn abs(self) -> f32 {
        self.re.hypot(self.im)
    }

    pub fn arg(self) -> f32 {
        self.im.atan2(self.re)
    }

    pub fn conj(self) -> Self {
        C32 { re: self.re, im: -self.im }
    }

    pub fn powi(self, n: i32) -> Self {
        C32::polar(self.abs().powi(n), self.arg() * n as f32)
    }

    pub fn scale(self, k: f32) -> Self {
        C32 { re: self.re * k, im: self.im * k }
    }

    /// As a point of the plane.
    pub fn vec2(self) -> glam::Vec2 {
        glam::Vec2::new(self.re, self.im)
    }
}

impl Add for C32 {
    type Output = C32;
    fn add(self, o: C32) -> C32 {
        C32 { re: self.re + o.re, im: self.im + o.im }
    }
}

impl Sub for C32 {
    type Output = C32;
    fn sub(self, o: C32) -> C32 {
        C32 { re: self.re - o.re, im: self.im - o.im }
    }
}

impl Mul for C32 {
    type Output = C32;
    fn mul(self, o: C32) -> C32 {
        C32 { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
}

impl Div for C32 {
    type Output = C32;
    fn div(self, o: C32) -> C32 {
        let d = o.re * o.re + o.im * o.im;
        C32 { re: (self.re * o.re + self.im * o.im) / d, im: (self.im * o.re - self.re * o.im) / d }
    }
}

impl Neg for C32 {
    type Output = C32;
    fn neg(self) -> C32 {
        C32 { re: -self.re, im: -self.im }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic() {
        let (a, b) = (C32::new(1.0, 2.0), C32::new(-0.5, 0.25));
        let q = a * b / b;
        assert!((q - a).abs() < 1e-6);
        let p = C32::polar(0.9, 0.7);
        assert!((p.powi(3) - p * p * p).abs() < 1e-6);
        assert!(((C32::ONE - p.powi(5)) / (C32::ONE - p) - (0..5).fold(C32::ZERO, |s, k| s + p.powi(k))).abs() < 1e-5);
    }
}

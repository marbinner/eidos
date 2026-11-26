//! A small, fast, seedable random number generator (xoshiro256++), so simulations are
//! reproducible. The integers and uniform numbers it draws are identical on native and
//! web. Normal and exponential draws go through `ln`, `sin` and `cos`, which come from
//! the C library natively and from Rust's `libm` in the browser; neither is correctly
//! rounded, so these can differ in the last bit.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
    spare: Option<f32>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // expand the seed with splitmix64
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()], spare: None }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
    }

    /// Uniform in [lo, hi).
    #[inline]
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f32()
    }

    /// Uniform integer in 0..n.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        (((self.next_u64() >> 32) * n as u64) >> 32) as usize
    }

    /// Standard normal (Box-Muller, both values used).
    pub fn normal(&mut self) -> f32 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let u = (1.0 - self.f32()).max(1e-12);
        let v = self.f32();
        let r = (-2.0 * u.ln()).sqrt();
        let (s, c) = (std::f32::consts::TAU * v).sin_cos();
        self.spare = Some(r * s);
        r * c
    }

    pub fn exponential(&mut self) -> f32 {
        -(1.0 - self.f32()).max(1e-12).ln()
    }

    pub fn bool(&mut self, p: f32) -> bool {
        self.f32() < p
    }

    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i + 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn moments() {
        let mut r = Rng::new(7);
        let n = 200_000;
        let (mut su, mut sn, mut sn2) = (0.0f64, 0.0f64, 0.0f64);
        for _ in 0..n {
            let u = r.f32();
            assert!((0.0..1.0).contains(&u));
            su += u as f64;
            let z = r.normal() as f64;
            sn += z;
            sn2 += z * z;
        }
        assert!((su / n as f64 - 0.5).abs() < 5e-3);
        assert!((sn / n as f64).abs() < 1e-2);
        assert!((sn2 / n as f64 - 1.0).abs() < 2e-2);
    }

    /// `below(7)` is a fair seven-sided die: every face turns up, and about equally
    /// often. The chi-squared statistic of 70 000 throws exceeds 22.5 one time in a
    /// thousand for a fair die (six degrees of freedom); with six seeds it came out
    /// between 3 and 12. A die that never shows a face scores over ten thousand; one
    /// that shows a face a tenth too often, over a hundred.
    #[test]
    fn below_is_a_fair_die() {
        let mut r = Rng::new(7);
        let mut counts = [0u32; 7];
        for _ in 0..70_000 {
            let face = r.below(7);
            assert!(face < 7, "below(7) gave {face}");
            counts[face.min(6)] += 1;
        }
        let chi2: f64 = counts.iter().map(|&c| (c as f64 - 10_000.0).powi(2) / 10_000.0).sum();
        assert!(chi2 < 22.5, "{counts:?}: chi-squared {chi2}");
    }
}

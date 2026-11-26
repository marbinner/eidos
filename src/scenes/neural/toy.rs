//! The toy model of superposition: more features than dimensions, squeezed through
//! and read back. The "Superposition" scene (`superposition.rs`) draws it.

use super::*;
use crate::math::nn::{adam_step_size, adam_update};

/// The toy model of Elhage et al., "Toy Models of Superposition" (2022): `n` features
/// are squeezed through `m < n` hidden dimensions and reconstructed,
///
/// ```text
///     h = W x            (m numbers)
///     x' = ReLU(W' h + b)    (n numbers; W' is the transpose: the same weights read back)
/// ```
///
/// trained to minimise `sum_i importance_i (x_i - x'_i)^2`. Each feature is zero with
/// probability `sparsity` and otherwise uniform in [0, 1]. Column `i` of `W` is the
/// direction feature `i` is given in the hidden space, which is what the scene draws.
pub struct Toy {
    pub n: usize,
    pub m: usize,
    /// `m` rows of `n`: `w[k * n + i]` is component `k` of feature `i`'s direction.
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    gw: Vec<f32>,
    gb: Vec<f32>,
    /// Adam's running averages for `w` then `b`, and its step count.
    mean: Vec<f32>,
    square: Vec<f32>,
    t: u32,
    x: Vec<f32>,
}

impl Toy {
    pub fn new(n: usize, m: usize, rng: &mut Rng) -> Self {
        let scale = 0.6 / (m as f32).sqrt();
        Toy {
            n,
            m,
            w: (0..m * n).map(|_| rng.normal() * scale).collect(),
            b: vec![0.0; n],
            gw: vec![0.0; m * n],
            gb: vec![0.0; n],
            mean: vec![0.0; m * n + n],
            square: vec![0.0; m * n + n],
            t: 0,
            x: Vec::new(),
        }
    }

    /// Feature `i`'s direction in the hidden space (z = 0 when there are two
    /// dimensions; with more than three, the first three of its components).
    pub fn direction(&self, i: usize) -> Vec3 {
        let at = |k: usize| if k < self.m { self.w[k * self.n + i] } else { 0.0 };
        Vec3::new(at(0), at(1), at(2))
    }

    /// Entry (i, j) of W'W: how much of feature j arrives at output i before the bias
    /// and the ReLU. The diagonal is each feature's squared length.
    pub fn overlap(&self, i: usize, j: usize) -> f32 {
        (0..self.m).map(|k| self.w[k * self.n + i] * self.w[k * self.n + j]).sum()
    }

    /// The share of a hidden dimension that feature `i` has to itself: its squared
    /// length divided by everything that lies along its direction. 1 for a feature
    /// with a dimension of its own, 1/2 for one of an opposite pair, 2/5 for a corner
    /// of a pentagon in the plane; 0 for a feature that is not represented.
    pub fn dimensionality(&self, i: usize) -> f32 {
        let own = self.overlap(i, i);
        if own < 1e-8 {
            return 0.0;
        }
        let along: f32 = (0..self.n).map(|j| self.overlap(i, j).powi(2) / own).sum();
        own / along
    }

    /// Feature `k` switched on alone, at 1: what arrives at output `i` through the
    /// hidden dimensions, and what comes out of it after the bias and the ReLU.
    pub fn alone(&self, k: usize, i: usize) -> (f32, f32) {
        let arrives = self.overlap(i, k);
        (arrives, (arrives + self.b[i]).max(0.0))
    }

    /// The loss on `count` samples laid end to end in `x`, and its gradient (left in
    /// `gw` and `gb`).
    fn loss_and_gradient(&mut self, x: &[f32], count: usize, importance: &[f32]) -> f32 {
        let (n, m) = (self.n, self.m);
        self.gw.fill(0.0);
        self.gb.fill(0.0);
        let mut total = 0.0;
        // (as many hidden dimensions as there are: the scene asks for two or three)
        let mut h = vec![0.0f32; m];
        let mut back = vec![0.0f32; m];
        for sample in x.chunks_exact(n).take(count) {
            for k in 0..m {
                h[k] = self.w[k * n..(k + 1) * n].iter().zip(sample).map(|(w, x)| w * x).sum();
            }
            back.fill(0.0);
            for i in 0..n {
                let z = self.b[i] + (0..m).map(|k| self.w[k * n + i] * h[k]).sum::<f32>();
                let error = z.max(0.0) - sample[i];
                total += importance[i] * error * error;
                if z > 0.0 {
                    // derivative with respect to z_i; it flows to the bias, to the
                    // weights that read feature i back out, and (through h) to the
                    // weights that wrote the sample in
                    let g = 2.0 * importance[i] * error;
                    self.gb[i] += g;
                    for k in 0..m {
                        self.gw[k * n + i] += g * h[k];
                        back[k] += g * self.w[k * n + i];
                    }
                }
            }
            for k in 0..m {
                for (gw, &xi) in self.gw[k * n..(k + 1) * n].iter_mut().zip(sample) {
                    *gw += back[k] * xi;
                }
            }
        }
        let scale = 1.0 / count.max(1) as f32;
        self.gw.iter_mut().chain(&mut self.gb).for_each(|g| *g *= scale);
        total * scale
    }

    /// The loss on a fresh random batch, as [`Toy::step`] would find it, without
    /// taking the step.
    pub fn loss(&mut self, batch: usize, sparsity: f32, importance: &[f32], rng: &mut Rng) -> f32 {
        let x: Vec<f32> = (0..batch * self.n).map(|_| if rng.f32() < sparsity { 0.0 } else { rng.f32() }).collect();
        self.loss_and_gradient(&x, batch, importance)
    }

    /// One Adam step on a fresh random batch; returns the loss before the step.
    pub fn step(&mut self, batch: usize, sparsity: f32, importance: &[f32], lr: f32, rng: &mut Rng) -> f32 {
        let n = self.n;
        let mut x = std::mem::take(&mut self.x);
        x.clear();
        x.extend((0..batch * n).map(|_| if rng.f32() < sparsity { 0.0 } else { rng.f32() }));
        let loss = self.loss_and_gradient(&x, batch, importance);
        self.x = x;
        self.t = self.t.saturating_add(1);
        let step = adam_step_size(self.t, lr);
        // (the running averages are kept for the weights, then the biases)
        let (mean_w, mean_b) = self.mean.split_at_mut(self.w.len());
        let (square_w, square_b) = self.square.split_at_mut(self.w.len());
        adam_update(step, &mut self.w, &self.gw, mean_w, square_w);
        adam_update(step, &mut self.b, &self.gb, mean_b, square_b);
        loss
    }
}

/// Feature `i` matters `decay^i` times as much as the first.
pub fn importance(n: usize, decay: f32) -> Vec<f32> {
    (0..n).map(|i| decay.powi(i as i32)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_finite_differences() {
        let mut rng = Rng::new(3);
        for m in [2, 3, 5] {
            let mut toy = Toy::new(6, m, &mut rng);
            toy.b.iter_mut().for_each(|b| *b = rng.normal() * 0.2);
            let x: Vec<f32> = (0..9 * 6).map(|_| if rng.f32() < 0.4 { 0.0 } else { rng.f32() }).collect();
            let imp = importance(6, 0.8);
            toy.loss_and_gradient(&x, 9, &imp);
            let (gw, gb) = (toy.gw.clone(), toy.gb.clone());
            let h = 1e-3;
            for i in 0..toy.w.len() + toy.b.len() {
                let (analytic, mut nudge) = if i < gw.len() { (gw[i], 0.0) } else { (gb[i - gw.len()], 0.0) };
                let mut value = |toy: &mut Toy, d: f32| {
                    if i < gw.len() {
                        toy.w[i] += d - nudge;
                    } else {
                        toy.b[i - gw.len()] += d - nudge;
                    }
                    nudge = d;
                    toy.loss_and_gradient(&x, 9, &imp)
                };
                let (up, down) = (value(&mut toy, h), value(&mut toy, -h));
                value(&mut toy, 0.0);
                let numeric = (up - down) / (2.0 * h);
                assert!((numeric - analytic).abs() < 2e-3 + 0.03 * numeric.abs(), "m={m} parameter {i}: {numeric} vs {analytic}");
            }
        }
    }
}

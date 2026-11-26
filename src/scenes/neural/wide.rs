//! A wide two-layer network whose output scale decides whether it learns by moving
//! its neurons or by leaving them where they are: the model behind the "Neurons as
//! particles" scene (`particles.rs`), which draws it.

use super::*;

/// One hidden layer of `N` neurons on two inputs,
///
/// ```text
///     g(x) = (1/√N) Σ aⱼ φ(wⱼ·x + bⱼ)
///     f(x) = α (g(x) − g₀(x))
/// ```
///
/// where `g₀` is `g` at the starting weights, so that the output starts at zero
/// whatever the scale `α`. It is trained by gradient descent on the logistic loss
/// with step `lr / α²` (Chizat and Bach's scaling): the output then moves at a speed
/// that does not depend on `α`, while the weights move at a speed proportional to
/// `1 / α`. A large `α` is the lazy regime: the output changes although the weights
/// hardly move. A small `α` forces the weights to travel. (The step also shrinks
/// where the tangent kernel has grown beyond its starting size: see [`Wide::step`].)
pub struct Wide {
    pub act: Act,
    pub w: Vec<Vec2>,
    pub b: Vec<f32>,
    pub a: Vec<f32>,
    start: (Vec<Vec2>, Vec<f32>, Vec<f32>),
    /// `g₀` on the training points.
    g0: Vec<f32>,
    /// The size of the tangent kernel at the start (see `step`).
    kernel0: f32,
    gw: Vec<Vec2>,
    gb: Vec<f32>,
    ga: Vec<f32>,
    z: Vec<f32>,
    fz: Vec<f32>,
    /// Every neuron's output at every training point, neuron after neuron, from the
    /// forward pass of [`Wide::loss_and_gradient`], which its backward pass reuses.
    phi: Vec<f32>,
    out: Vec<f32>,
    r: Vec<f32>,
}

impl Wide {
    pub fn new(n: usize, act: Act, rng: &mut Rng) -> Self {
        let w: Vec<Vec2> = (0..n).map(|_| Vec2::new(rng.normal(), rng.normal()) * 1.5).collect();
        let b: Vec<f32> = (0..n).map(|_| rng.normal() * 0.6).collect();
        let a: Vec<f32> = (0..n).map(|_| rng.normal()).collect();
        Wide {
            act,
            start: (w.clone(), b.clone(), a.clone()),
            w,
            b,
            a,
            g0: Vec::new(),
            kernel0: 1.0,
            gw: vec![Vec2::ZERO; n],
            gb: vec![0.0; n],
            ga: vec![0.0; n],
            z: Vec::new(),
            fz: Vec::new(),
            phi: Vec::new(),
            out: Vec::new(),
            r: Vec::new(),
        }
    }

    pub fn neurons(&self) -> usize {
        self.a.len()
    }

    /// Back to the starting weights.
    pub fn rewind(&mut self) {
        self.w.clone_from(&self.start.0);
        self.b.clone_from(&self.start.1);
        self.a.clone_from(&self.start.2);
    }

    pub fn start_of(&self, j: usize) -> Vec2 {
        self.start.0[j]
    }

    /// `g` at each of the points laid end to end in `xs`.
    pub fn g(&mut self, xs: &[f32], out: &mut Vec<f32>) {
        self.forward(xs, out, false);
    }

    /// `g` at the points, keeping every neuron's output in `phi` if `keep`.
    fn forward(&mut self, xs: &[f32], out: &mut Vec<f32>, keep: bool) {
        let n = xs.len() / 2;
        out.clear();
        out.resize(n, 0.0);
        self.z.resize(n, 0.0);
        self.fz.resize(n, 0.0);
        if keep {
            self.phi.resize(n * self.neurons(), 0.0);
        }
        for j in 0..self.neurons() {
            let (w, b) = (self.w[j], self.b[j]);
            for (z, x) in self.z.iter_mut().zip(xs.chunks_exact(2)) {
                *z = w.x * x[0] + w.y * x[1] + b;
            }
            let fz = if keep { &mut self.phi[j * n..(j + 1) * n] } else { &mut self.fz[..] };
            self.act.apply(&self.z, fz);
            let aj = self.a[j];
            for (o, f) in out.iter_mut().zip(fz.iter()) {
                *o += aj * f;
            }
        }
        let scale = 1.0 / (self.neurons() as f32).sqrt();
        out.iter_mut().for_each(|o| *o *= scale);
    }

    /// Remember `g₀` on the data (call at the starting weights, before training).
    /// Returns the loss and accuracy there, with the output scaled by `alpha`.
    pub fn anchor(&mut self, data: &Data, alpha: f32) -> (f32, f32) {
        let mut g0 = std::mem::take(&mut self.g0);
        self.g(&data.x, &mut g0);
        self.g0 = g0;
        self.kernel0 = 0.0;
        let (_, _, kernel) = self.loss_and_gradient(data, 1.0);
        self.kernel0 = kernel.max(1e-9);
        let (loss, accuracy, _) = self.loss_and_gradient(data, alpha);
        (loss, accuracy)
    }

    /// The loss and accuracy at the present weights, the gradient of the loss with
    /// respect to every weight (left in `gw`, `gb`, `ga`), and the average size of
    /// the tangent kernel `|∇g(x)|²` over the data.
    fn loss_and_gradient(&mut self, data: &Data, alpha: f32) -> (f32, f32, f32) {
        let (n, neurons) = (data.len(), self.neurons());
        let scale = 1.0 / (neurons as f32).sqrt();
        let mut out = std::mem::take(&mut self.out);
        self.forward(&data.x, &mut out, true);
        // the output, the loss, and how the loss changes with each output
        self.r.resize(n, 0.0);
        let (mut loss, mut hits) = (0.0, 0);
        for i in 0..n {
            let f = alpha * (out[i] - self.g0.get(i).copied().unwrap_or(0.0));
            let y = data.y[i];
            loss += f.max(0.0) - f * y + (1.0 + (-f.abs()).exp()).ln();
            hits += ((f > 0.0) == (y > 0.5)) as usize;
            self.r[i] = (1.0 / (1.0 + (-f).exp()) - y) / n as f32;
        }
        self.out = out;
        let mut kernel = 0.0;
        for j in 0..neurons {
            let (w, b, aj) = (self.w[j], self.b[j], self.a[j]);
            let (mut ga, mut gb, mut gw) = (0.0, 0.0, Vec2::ZERO);
            // (each neuron's outputs from the forward pass; its inputs again, which
            // costs two multiplications where the activation would cost many more)
            let phi = &self.phi[j * n..(j + 1) * n];
            for ((i, x), &fz) in data.x.chunks_exact(2).enumerate().zip(phi) {
                let z = w.x * x[0] + w.y * x[1] + b;
                let slope = self.act.df(z, fz);
                ga += self.r[i] * fz;
                gb += self.r[i] * slope;
                gw += Vec2::new(x[0], x[1]) * (self.r[i] * slope);
                kernel += fz * fz + aj * aj * slope * slope * (1.0 + x[0] * x[0] + x[1] * x[1]);
            }
            self.ga[j] = alpha * scale * ga;
            self.gb[j] = alpha * scale * aj * gb;
            self.gw[j] = gw * (alpha * scale * aj);
        }
        (loss / n as f32, hits as f32 / n.max(1) as f32, kernel * scale * scale / n.max(1) as f32)
    }

    /// One step of gradient descent; returns the loss and accuracy before it. The
    /// step shrinks as the tangent kernel grows beyond its starting size (as it does
    /// once the weights have grown), which is what keeps a fixed learning rate stable
    /// in the regime where the weights travel far.
    pub fn step(&mut self, data: &Data, alpha: f32, lr: f32) -> (f32, f32) {
        let (loss, accuracy, kernel) = self.loss_and_gradient(data, alpha);
        let step = lr / (alpha * alpha) * (self.kernel0 / kernel.max(1e-9)).min(1.0);
        for j in 0..self.neurons() {
            self.w[j] -= self.gw[j] * step;
            self.b[j] -= self.gb[j] * step;
            self.a[j] -= self.ga[j] * step;
        }
        (loss, accuracy)
    }

    /// How far the weights are from where they started, relative to their starting size.
    pub fn moved(&self) -> f32 {
        let (mut gone, mut size) = (0.0, 0.0);
        for j in 0..self.neurons() {
            gone += (self.w[j] - self.start.0[j]).length_squared() + (self.b[j] - self.start.1[j]).powi(2) + (self.a[j] - self.start.2[j]).powi(2);
            size += self.start.0[j].length_squared() + self.start.1[j].powi(2) + self.start.2[j].powi(2);
        }
        (gone / size.max(1e-12)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_finite_differences() {
        let mut rng = Rng::new(4);
        let data = make_data(2, 30, 0.2, &mut rng);
        for act in [Act::Tanh, Act::Gelu] {
            let mut net = Wide::new(7, act, &mut rng);
            net.anchor(&data, 1.0);
            // away from the start, so that the output is not zero
            for _ in 0..5 {
                net.step(&data, 0.7, 0.5);
            }
            net.loss_and_gradient(&data, 0.7);
            let (gw, gb, ga) = (net.gw.clone(), net.gb.clone(), net.ga.clone());
            let h = 1e-3;
            for j in 0..7 {
                let check = |net: &mut Wide, analytic: f32, nudge: &dyn Fn(&mut Wide, f32)| {
                    nudge(net, h);
                    let up = net.loss_and_gradient(&data, 0.7).0;
                    nudge(net, -2.0 * h);
                    let down = net.loss_and_gradient(&data, 0.7).0;
                    nudge(net, h);
                    let numeric = (up - down) / (2.0 * h);
                    assert!((numeric - analytic).abs() < 2e-4 + 0.03 * numeric.abs(), "{act:?} neuron {j}: {numeric} vs {analytic}");
                };
                check(&mut net, gw[j].x, &|n, d| n.w[j].x += d);
                check(&mut net, gw[j].y, &|n, d| n.w[j].y += d);
                check(&mut net, gb[j], &|n, d| n.b[j] += d);
                check(&mut net, ga[j], &|n, d| n.a[j] += d);
            }
        }
    }
}

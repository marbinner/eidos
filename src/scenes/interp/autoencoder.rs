//! The sparse autoencoder itself, apart from the scene that shows it (`sae.rs`): the
//! toy activations it is trained on (a few true features packed into too few
//! dimensions), the autoencoder and its training step, and the report that says how
//! well it found the features.

use crate::math::nn::{adam_step_size, adam_update, axpy, dot};
use crate::math::rng::Rng;
use std::f32::consts::{PI, TAU};

// ----- the data: features in superposition ------------------------------------------------

/// `n` "true features", each a fixed unit direction in `m` dimensions. An activation
/// is a sum of a few of them, `h = sum_i x_i d_i` (plus noise), where each `x_i` is
/// zero with probability `sparsity` and otherwise uniform in (0, 1]. This is what the
/// hidden layer of the Superposition scene's model looks like once it has settled.
pub struct Features {
    pub n: usize,
    pub m: usize,
    /// `n` rows of `m`.
    pub dirs: Vec<f32>,
}

impl Features {
    /// Evenly spread round the circle for `m = 2`, evenly over the sphere for `m = 3`,
    /// and otherwise random directions pushed apart a little so that no two nearly
    /// coincide.
    pub fn new(n: usize, m: usize, rng: &mut Rng) -> Self {
        let mut dirs = vec![0.0; n * m];
        match m {
            2 => {
                for i in 0..n {
                    let a = 0.5 * PI + TAU * i as f32 / n as f32;
                    dirs[i * 2] = a.cos();
                    dirs[i * 2 + 1] = a.sin();
                }
            }
            3 => {
                // a spiral over the sphere (the "Fibonacci" lattice): close to even
                let golden = PI * (3.0 - 5f32.sqrt());
                for i in 0..n {
                    let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
                    let r = (1.0 - z * z).sqrt();
                    let a = golden * i as f32;
                    dirs[i * 3..i * 3 + 3].copy_from_slice(&[r * a.cos(), r * a.sin(), z]);
                }
            }
            _ => {
                dirs.iter_mut().for_each(|d| *d = rng.normal());
                normalise_rows(&mut dirs, m);
            }
        }
        if m >= 3 {
            // each direction is pushed away from the others it is nearly parallel to
            for _ in 0..60 {
                let before = dirs.clone();
                for i in 0..n {
                    for j in 0..n {
                        let c = dot(&before[i * m..(i + 1) * m], &before[j * m..(j + 1) * m]);
                        if i != j && c > 0.0 {
                            let (row, other) = (&mut dirs[i * m..(i + 1) * m], &before[j * m..(j + 1) * m]);
                            axpy(row, -0.25 * c * c * c, other);
                        }
                    }
                }
                normalise_rows(&mut dirs, m);
            }
        }
        Features { n, m, dirs }
    }

    pub fn dir(&self, i: usize) -> &[f32] {
        &self.dirs[i * self.m..(i + 1) * self.m]
    }

    /// `count` samples: the feature values go to `x` (`count` rows of `n`) and the
    /// activations to `h` (`count` rows of `m`).
    pub fn sample(&self, count: usize, sparsity: f32, noise: f32, rng: &mut Rng, x: &mut Vec<f32>, h: &mut Vec<f32>) {
        let (n, m) = (self.n, self.m);
        x.clear();
        x.extend((0..count * n).map(|_| if rng.f32() < sparsity { 0.0 } else { 1.0 - rng.f32() }));
        h.clear();
        h.resize(count * m, 0.0);
        for (x, h) in x.chunks_exact(n).zip(h.chunks_exact_mut(m)) {
            for (i, &xi) in x.iter().enumerate() {
                if xi != 0.0 {
                    axpy(h, xi, self.dir(i));
                }
            }
            if noise > 0.0 {
                h.iter_mut().for_each(|v| *v += noise * rng.normal());
            }
        }
    }
}

fn normalise_rows(a: &mut [f32], m: usize) {
    for row in a.chunks_exact_mut(m) {
        let len = dot(row, row).sqrt();
        if len > 1e-12 {
            row.iter_mut().for_each(|v| *v /= len);
        }
    }
}

// ----- the autoencoder --------------------------------------------------------------------

/// A sparse autoencoder with `k` latents on `m`-dimensional activations:
///
/// ```text
///     f  = ReLU(E (h - b_d) + b_e)        (k numbers, most of them zero)
///     h' = D f + b_d                      (m numbers: the reconstruction)
/// ```
///
/// trained to minimise `|h - h'|^2 + lambda * sum_j f_j` averaged over samples, with
/// every column of `D` (a latent's "decoder direction") held at length 1, so that the
/// penalty cannot be dodged by making the directions long and the latents small.
pub struct Sae {
    pub m: usize,
    pub k: usize,
    /// `k` rows of `m`: row `j` is what latent `j` listens for.
    pub enc: Vec<f32>,
    pub be: Vec<f32>,
    /// `k` rows of `m`: row `j` is the unit direction latent `j` writes back.
    pub dec: Vec<f32>,
    pub bd: Vec<f32>,
    g_enc: Vec<f32>,
    g_be: Vec<f32>,
    g_dec: Vec<f32>,
    g_bd: Vec<f32>,
    /// Adam's running averages (in the order enc, be, dec, bd), and its step counts:
    /// one for each latent (its row of `enc`, its `be`, its row of `dec`), which starts
    /// again from nothing when the latent is re-seeded, and one for `bd`.
    mean: Vec<f32>,
    square: Vec<f32>,
    steps: Vec<u32>,
    t: u32,
    // scratch
    c: Vec<f32>,
    r: Vec<f32>,
    back: Vec<f32>,
    f: Vec<f32>,
}

/// The most dimensions an autoencoder can have here (the room `run` keeps on the
/// stack, since it is called for every sample measured and may not touch `self`).
pub const MAX_DIM: usize = 16;

/// What a batch came to: squared error and number of latents firing, per sample.
#[derive(Clone, Copy, Default)]
pub struct BatchLoss {
    pub error: f32,
    pub firing: f32,
    /// The whole objective (error plus penalty): what the gradient is the gradient of.
    #[cfg_attr(not(test), allow(dead_code))]
    pub total: f32,
}

impl Sae {
    /// Random unit decoder directions, and an encoder that starts as their mirror.
    pub fn new(m: usize, k: usize, rng: &mut Rng) -> Self {
        assert!(m <= MAX_DIM, "an autoencoder on {m} dimensions: at most {MAX_DIM}");
        let mut dec: Vec<f32> = (0..k * m).map(|_| rng.normal()).collect();
        normalise_rows(&mut dec, m);
        let count = 2 * k * m + k + m;
        Sae {
            m,
            k,
            enc: dec.clone(),
            be: vec![0.0; k],
            dec,
            bd: vec![0.0; m],
            g_enc: vec![0.0; k * m],
            g_be: vec![0.0; k],
            g_dec: vec![0.0; k * m],
            g_bd: vec![0.0; m],
            mean: vec![0.0; count],
            square: vec![0.0; count],
            steps: vec![0; k],
            t: 0,
            c: vec![0.0; m],
            r: vec![0.0; m],
            back: vec![0.0; m],
            f: vec![0.0; k],
        }
    }

    pub fn decoder(&self, j: usize) -> &[f32] {
        &self.dec[j * self.m..(j + 1) * self.m]
    }

    pub fn encoder(&self, j: usize) -> &[f32] {
        &self.enc[j * self.m..(j + 1) * self.m]
    }

    /// The latents for one activation (into `f`, `k` numbers) and its reconstruction
    /// (into `out`, `m` numbers).
    pub fn run(&self, h: &[f32], f: &mut [f32], out: &mut [f32]) {
        let m = self.m;
        let mut c = [0.0f32; MAX_DIM];
        for i in 0..m {
            c[i] = h[i] - self.bd[i];
        }
        out.copy_from_slice(&self.bd);
        for j in 0..self.k {
            let z = dot(self.encoder(j), &c[..m]) + self.be[j];
            f[j] = z.max(0.0);
            if z > 0.0 {
                axpy(out, z, self.decoder(j));
            }
        }
    }

    /// The loss on `count` activations laid end to end in `h`, and its gradient (left
    /// in `g_*`).
    fn loss_and_gradient(&mut self, h: &[f32], count: usize, lambda: f32) -> BatchLoss {
        let (m, k) = (self.m, self.k);
        self.g_enc.fill(0.0);
        self.g_be.fill(0.0);
        self.g_dec.fill(0.0);
        self.g_bd.fill(0.0);
        let (mut error, mut firing, mut penalty) = (0.0f32, 0usize, 0.0f32);
        let mut f = std::mem::take(&mut self.f);
        for h in h.chunks_exact(m).take(count) {
            for i in 0..m {
                self.c[i] = h[i] - self.bd[i];
                self.r[i] = -self.c[i];
            }
            for j in 0..k {
                let z = dot(&self.enc[j * m..(j + 1) * m], &self.c) + self.be[j];
                f[j] = z.max(0.0);
                if z > 0.0 {
                    axpy(&mut self.r, z, &self.dec[j * m..(j + 1) * m]);
                    firing += 1;
                    penalty += z;
                }
            }
            // r = h' - h
            error += dot(&self.r, &self.r);
            self.back.fill(0.0);
            for j in 0..k {
                if f[j] > 0.0 {
                    let g = 2.0 * dot(&self.r, &self.dec[j * m..(j + 1) * m]) + lambda;
                    axpy(&mut self.g_dec[j * m..(j + 1) * m], 2.0 * f[j], &self.r);
                    axpy(&mut self.g_enc[j * m..(j + 1) * m], g, &self.c);
                    self.g_be[j] += g;
                    axpy(&mut self.back, g, &self.enc[j * m..(j + 1) * m]);
                }
            }
            // b_d is added to the output and taken from the input
            for i in 0..m {
                self.g_bd[i] += 2.0 * self.r[i] - self.back[i];
            }
        }
        self.f = f;
        let scale = 1.0 / count.max(1) as f32;
        for g in self.g_enc.iter_mut().chain(&mut self.g_be).chain(&mut self.g_dec).chain(&mut self.g_bd) {
            *g *= scale;
        }
        BatchLoss { error: error * scale, firing: firing as f32 * scale, total: (error + lambda * penalty) * scale }
    }

    /// One Adam step on the activations in `h`; returns the loss before the step.
    pub fn step(&mut self, h: &[f32], count: usize, lambda: f32, lr: f32) -> BatchLoss {
        let (m, k) = (self.m, self.k);
        let loss = self.loss_and_gradient(h, count, lambda);
        // a decoder direction may turn but not grow: drop the part of its gradient
        // that only changes its length
        for j in 0..k {
            let along = dot(&self.g_dec[j * m..(j + 1) * m], &self.dec[j * m..(j + 1) * m]);
            axpy(&mut self.g_dec[j * m..(j + 1) * m], -along, &self.dec[j * m..(j + 1) * m]);
        }
        // (the running averages are kept in this order: encoder, its biases, decoder, its
        // biases; each latent's parts are corrected by its own step count, so that one
        // just re-seeded takes the careful first steps of a fresh start)
        let (dec_at, bd_at) = (k * m + k, 2 * k * m + k);
        for j in 0..k {
            self.steps[j] = self.steps[j].saturating_add(1);
            let step = adam_step_size(self.steps[j], lr);
            for (p, g, at, n) in [(&mut self.enc, &self.g_enc, 0, m), (&mut self.be, &self.g_be, k * m, 1), (&mut self.dec, &self.g_dec, dec_at, m)] {
                let (part, at) = (j * n..(j + 1) * n, at + j * n);
                adam_update(step, &mut p[part.clone()], &g[part], &mut self.mean[at..at + n], &mut self.square[at..at + n]);
            }
        }
        self.t = self.t.saturating_add(1);
        adam_update(adam_step_size(self.t, lr), &mut self.bd, &self.g_bd, &mut self.mean[bd_at..bd_at + m], &mut self.square[bd_at..bd_at + m]);
        normalise_rows(&mut self.dec, m);
        loss
    }

    /// Give latent `j` a fresh start pointing along `direction` (an activation the
    /// autoencoder reconstructs badly is a good choice).
    pub fn reseed(&mut self, j: usize, direction: &[f32]) {
        let (m, k) = (self.m, self.k);
        let len = dot(direction, direction).sqrt().max(1e-6);
        for i in 0..m {
            self.dec[j * m + i] = direction[i] / len;
            self.enc[j * m + i] = direction[i] / len;
        }
        self.be[j] = 0.0;
        // forget what the optimiser remembers about it, its step count included (with
        // the averages emptied but the count kept, the bias correction would be about 1
        // and the first steps several times the learning rate)
        let spans = [(j * m, m), (k * m + j, 1), (k * m + k + j * m, m)];
        for (start, len) in spans {
            self.mean[start..start + len].fill(0.0);
            self.square[start..start + len].fill(0.0);
        }
        self.steps[j] = 0;
    }
}

// ----- measuring a dictionary against the truth -------------------------------------------

/// How an autoencoder is doing, measured on a fixed set of samples whose true feature
/// values are known.
#[derive(Clone, Default)]
pub struct Report {
    /// Squared error left, as a share of the activations' variance.
    pub unexplained: f32,
    /// Latents firing per sample, on average.
    pub l0: f32,
    /// How often each latent fires (share of samples).
    pub rate: Vec<f32>,
    /// `k` rows of `n`: cosine between decoder direction `j` and true feature `i`.
    pub cosine: Vec<f32>,
    /// For each true feature, the latent whose direction is nearest and that cosine.
    pub best: Vec<(usize, f32)>,
    /// For each latent, the true feature it is nearest to and that cosine.
    #[cfg_attr(not(test), allow(dead_code))]
    pub nearest: Vec<(usize, f32)>,
    /// For each latent, the feature it stands for, if its direction is close to one.
    pub owner: Vec<Option<usize>>,
    #[cfg_attr(not(test), allow(dead_code))]
    /// For each latent: shown every feature alone and at full strength, the share of
    /// its response that goes to the feature it is nearest to. 1 for a latent that
    /// answers to its own feature and to no other; a fresh one, which answers to
    /// whatever lies on its side of the plane, scores far lower. (0 if it answers
    /// to none.)
    pub telling: Vec<f32>,
    /// The latents in display order: by the feature they match, the dead ones last.
    pub order: Vec<usize>,
    pub dead: usize,
    pub recovered: usize,
    pub mean_best: f32,
    /// Over the features some latent stands for: what their latents report in total,
    /// divided by what was truly there. 1 is a faithful reading; less is shrinkage.
    pub read: f32,
    /// A sample of (true value of a feature, what its latents report, the feature),
    /// taken wherever a feature that has a latent is present.
    pub pairs: Vec<(f32, f32, u16)>,
}

/// A cosine above this counts as "the same direction".
pub const RECOVERED: f32 = 0.95;
/// A feature is found when a latent points along it and answers to that feature
/// rather than to anything nearby: at least this share of its response to the
/// features, shown one at a time, is to its own. Without this, directions that
/// merely happen to point the right way before any training would count as finds
/// (and with a dozen features in a plane, every direction points at one). A fresh
/// latent that happens to point at one of five features in a plane scores 0.62, and
/// about a half in space; a trained one 0.85 and more, whatever the dimensions. (With
/// four or fewer in a plane, or six or fewer in space, the others are so far off that a
/// fresh latent pointing at one answers to it alone and does count: nothing that
/// looks only at what a latent answers to could tell it from a trained one.)
pub const TELLING: f32 = 0.75;
/// A latent this close to a feature is taken to stand for it (when measuring what it reports).
pub const MATCHED: f32 = 0.9;
/// A latent that fires in fewer than this share of samples is dead.
pub const DEAD: f32 = 0.002;
/// How many points the scatter of true against reported values keeps.
const PAIRS: usize = 360;

impl Report {
    /// `variance`: the activations' spread, as [`spread`] gives it (it does not
    /// change with the autoencoder, so the caller keeps it).
    pub fn of(sae: &Sae, features: &Features, x: &[f32], h: &[f32], variance: f32) -> Report {
        let (n, m, k) = (features.n, features.m, sae.k);
        let count = h.len() / m;
        let cosine: Vec<f32> = (0..k * n).map(|e| dot(sae.decoder(e / n), features.dir(e % n))).collect();
        let argmax = |it: &mut dyn Iterator<Item = (usize, f32)>| it.fold((0, -2.0), |a, b| if b.1 > a.1 { b } else { a });
        let best: Vec<(usize, f32)> = (0..n).map(|i| argmax(&mut (0..k).map(|j| (j, cosine[j * n + i])))).collect();
        let nearest: Vec<(usize, f32)> = (0..k).map(|j| argmax(&mut (0..n).map(|i| (i, cosine[j * n + i])))).collect();
        let owner: Vec<Option<usize>> = nearest.iter().map(|&(i, c)| (c > MATCHED).then_some(i)).collect();
        let has_latent: Vec<bool> = (0..n).map(|i| owner.contains(&Some(i))).collect();

        // each latent's answer to every feature on its own
        let mut telling = vec![0.0f32; k];
        let (mut alone, mut rebuilt) = (vec![0.0f32; k], vec![0.0f32; m]);
        let mut total = vec![0.0f32; k];
        for i in 0..n {
            sae.run(features.dir(i), &mut alone, &mut rebuilt);
            for j in 0..k {
                total[j] += alone[j];
                if nearest[j].0 == i {
                    telling[j] = alone[j];
                }
            }
        }
        for j in 0..k {
            telling[j] = if total[j] > 0.0 { telling[j] / total[j] } else { 0.0 };
        }

        let mut rate = vec![0.0f32; k];
        let (mut f, mut out, mut reported) = (vec![0.0f32; k], vec![0.0f32; m], vec![0.0f32; n]);
        let (mut error, mut l0, mut got, mut want) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut pairs = Vec::new();
        for (s, h) in h.chunks_exact(m).enumerate() {
            sae.run(h, &mut f, &mut out);
            error += h.iter().zip(&out).map(|(a, b)| (a - b) * (a - b)).sum::<f32>();
            reported.fill(0.0);
            for j in 0..k {
                if f[j] > 0.0 {
                    rate[j] += 1.0;
                    l0 += 1.0;
                    if let Some(i) = owner[j] {
                        reported[i] += f[j];
                    }
                }
            }
            for i in 0..n {
                let xi = x[s * n + i];
                if xi > 0.0 && has_latent[i] {
                    got += reported[i];
                    want += xi;
                    if pairs.len() < PAIRS {
                        pairs.push((xi, reported[i], i as u16));
                    }
                }
            }
        }
        let scale = 1.0 / count.max(1) as f32;
        rate.iter_mut().for_each(|r| *r *= scale);
        let dead = rate.iter().filter(|r| **r < DEAD).count();
        let finds = |j: usize, i: usize| cosine[j * n + i] > RECOVERED && rate[j] >= DEAD && telling[j] >= TELLING;
        let recovered = (0..n).filter(|&i| (0..k).any(|j| finds(j, i))).count();
        let mut order: Vec<usize> = (0..k).collect();
        let rank = |j: usize| (rate[j] < DEAD, nearest[j].0, -nearest[j].1);
        order.sort_by(|&a, &b| rank(a).partial_cmp(&rank(b)).unwrap_or(std::cmp::Ordering::Equal));
        Report {
            unexplained: error * scale / variance.max(1e-12),
            l0: l0 * scale,
            mean_best: best.iter().map(|b| b.1).sum::<f32>() / n.max(1) as f32,
            read: if want > 0.0 { got / want } else { 0.0 },
            rate,
            cosine,
            best,
            nearest,
            owner,
            telling,
            order,
            dead,
            recovered,
            pairs,
        }
    }
}

/// How far the activations in `h` (rows of `m`) lie from their mean, squared and
/// averaged: the variance an autoencoder is asked to explain.
pub fn spread(h: &[f32], m: usize) -> f32 {
    let count = (h.len() / m).max(1) as f32;
    let mut mean = vec![0.0f32; m];
    for row in h.chunks_exact(m) {
        for i in 0..m {
            mean[i] += row[i] / count;
        }
    }
    h.chunks_exact(m).map(|row| row.iter().zip(&mean).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()).sum::<f32>() / count
}

/// The mean of the activations in `h` (rows of `m`), their principal components (`m`
/// rows of `m`, widest first) and the standard deviation along each.
pub fn principal_components(h: &[f32], m: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let count = (h.len() / m).max(1) as f64;
    let mut mean = vec![0.0f64; m];
    for row in h.chunks_exact(m) {
        for i in 0..m {
            mean[i] += row[i] as f64 / count;
        }
    }
    let mut cov = vec![0.0f64; m * m];
    for row in h.chunks_exact(m) {
        for a in 0..m {
            for b in 0..m {
                cov[a * m + b] += (row[a] as f64 - mean[a]) * (row[b] as f64 - mean[b]) / count;
            }
        }
    }
    let (values, vectors) = crate::math::linalg::eig_sym(&cov, m);
    (mean.iter().map(|&v| v as f32).collect(), vectors.iter().map(|&v| v as f32).collect(), values.iter().map(|&v| v.max(0.0).sqrt() as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_matches_finite_differences() {
        let mut rng = Rng::new(3);
        for (n, m, k) in [(5, 2, 6), (7, 4, 5)] {
            let features = Features::new(n, m, &mut rng);
            let mut sae = Sae::new(m, k, &mut rng);
            sae.enc.iter_mut().for_each(|v| *v += 0.3 * rng.normal());
            sae.be.iter_mut().for_each(|v| *v = 0.1 * rng.normal());
            sae.bd.iter_mut().for_each(|v| *v = 0.1 * rng.normal());
            let (mut x, mut h) = (Vec::new(), Vec::new());
            features.sample(11, 0.5, 0.02, &mut rng, &mut x, &mut h);
            let lambda = 0.13;
            sae.loss_and_gradient(&h, 11, lambda);
            let analytic: Vec<f32> = sae.g_enc.iter().chain(&sae.g_be).chain(&sae.g_dec).chain(&sae.g_bd).copied().collect();
            let d = 2e-3;
            for i in 0..analytic.len() {
                let value = |sae: &mut Sae, delta: f32| {
                    let (a, b, c) = (k * m, k * m + k, 2 * k * m + k);
                    let p = if i < a {
                        &mut sae.enc[i]
                    } else if i < b {
                        &mut sae.be[i - a]
                    } else if i < c {
                        &mut sae.dec[i - b]
                    } else {
                        &mut sae.bd[i - c]
                    };
                    let keep = *p;
                    *p += delta;
                    let total = sae.loss_and_gradient(&h, 11, lambda).total;
                    let p = if i < a {
                        &mut sae.enc[i]
                    } else if i < b {
                        &mut sae.be[i - a]
                    } else if i < c {
                        &mut sae.dec[i - b]
                    } else {
                        &mut sae.bd[i - c]
                    };
                    *p = keep;
                    total
                };
                let numeric = (value(&mut sae, d) - value(&mut sae, -d)) / (2.0 * d);
                assert!((numeric - analytic[i]).abs() < 3e-3 + 0.03 * numeric.abs(), "n={n} m={m} parameter {i}: {numeric} vs {}", analytic[i]);
            }
        }
    }

    #[test]
    fn true_features_are_unit_directions_kept_apart() {
        let mut rng = Rng::new(9);
        for (n, m) in [(5usize, 2usize), (8, 3), (16, 8), (32, 16)] {
            let f = Features::new(n, m, &mut rng);
            for i in 0..n {
                assert!((dot(f.dir(i), f.dir(i)) - 1.0).abs() < 1e-4);
                for j in 0..i {
                    assert!(dot(f.dir(i), f.dir(j)) < 0.9, "n={n} m={m}: features {i} and {j} nearly coincide");
                }
            }
        }
    }
}

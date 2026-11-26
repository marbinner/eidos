//! Arithmetic tables as a learning task: every pair (a, b) with a, b < p is one
//! example, the answer is one of p classes, and some pairs are held out. This is the
//! setting in which networks are seen to memorise first and generalise much later.
//!
//! This file is the task itself (no drawing): the playground shows it.

use crate::math::nn::{fast_exp, Act, Loss, Mlp};
use crate::math::rng::Rng;

/// The tables, as formulas (markup, to go between dollars).
pub const OPS: [&str; 8] = ["a + b", "a - b", r"a \cdot b", "a^2 + b^2", "a^2 + ab + b^2", r"\max(a, b)", "|a - b|", r"a \;\mathrm{xor}\; b"];
pub const MODULI: [usize; 6] = [5, 7, 11, 13, 17, 23];

/// The answer for the pair (a, b). Everything is reduced mod p, so there are always
/// exactly p possible answers.
pub fn answer(kind: usize, a: usize, b: usize, p: usize) -> usize {
    match kind {
        0 => (a + b) % p,
        1 => (a + p - b) % p,
        2 => (a * b) % p,
        3 => (a * a + b * b) % p,
        4 => (a * a + a * b + b * b) % p,
        5 => a.max(b),
        6 => a.abs_diff(b),
        _ => (a ^ b) % p,
    }
}

/// Is the operation built on addition mod p, so that answers p - 1 and 0 are neighbours?
pub fn cyclic(kind: usize) -> bool {
    kind < 5
}

#[derive(Clone)]
pub struct Task {
    pub p: usize,
    pub kind: usize,
    /// The fraction and the random split it was made with (enough to make it again).
    pub fraction: f32,
    pub split: u64,
    /// The right answer for cell `b * p + a`.
    pub answer: Vec<u8>,
    pub held_out: Vec<bool>,
    /// Cells trained on, and cells held out.
    pub train: Vec<u32>,
    pub test: Vec<u32>,
    /// The network's input for every cell: the two active input units, `a` and `p + b`
    /// (a one-hot code for each number, side by side).
    pub active: Vec<u32>,
    /// The same for the training cells only, with their answers: a ready-made full batch.
    pub train_active: Vec<u32>,
    pub train_y: Vec<f32>,
}

impl Task {
    /// `fraction` of the table (chosen at random from `seed`) is trained on.
    pub fn new(kind: usize, p: usize, fraction: f32, seed: u64) -> Task {
        let cells = p * p;
        let answer: Vec<u8> = (0..cells).map(|s| answer(kind, s % p, s / p, p) as u8).collect();
        let active: Vec<u32> = (0..cells).flat_map(|s| [(s % p) as u32, (p + s / p) as u32]).collect();
        let mut order: Vec<u32> = (0..cells as u32).collect();
        let mut rng = Rng::new(seed);
        for i in (1..cells).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let n_train = ((fraction * cells as f32).round() as usize).clamp(p.min(cells), cells);
        let mut train = order[..n_train].to_vec();
        let mut test = order[n_train..].to_vec();
        train.sort_unstable();
        test.sort_unstable();
        let mut held_out = vec![false; cells];
        for &s in &test {
            held_out[s as usize] = true;
        }
        let train_active = train.iter().flat_map(|&s| [active[2 * s as usize], active[2 * s as usize + 1]]).collect();
        let train_y = train.iter().map(|&s| answer[s as usize] as f32).collect();
        Task { p, kind, fraction, split: seed, answer, held_out, train, test, active, train_active, train_y }
    }

    pub fn cells(&self) -> usize {
        self.p * self.p
    }

    /// Layer widths for a network on this task.
    pub fn sizes(&self, hidden: usize, width: usize) -> Vec<usize> {
        super::layer_sizes(2 * self.p, hidden, width, self.p)
    }

    /// Fill `x` and `y` with a batch of `n` training cells drawn at random.
    pub fn sample(&self, n: usize, rng: &mut Rng, x: &mut Vec<u32>, y: &mut Vec<f32>) {
        x.clear();
        y.clear();
        for _ in 0..n {
            let s = self.train[rng.below(self.train.len())] as usize;
            x.extend_from_slice(&self.active[2 * s..2 * s + 2]);
            y.push(self.answer[s] as f32);
        }
    }
}

/// The settings the playground opens with: chosen (by the exploration in the tests)
/// so that memorising and generalising show up as two separate phases.
pub struct Defaults {
    pub kind: usize,
    pub p: usize,
    pub fraction: f32,
    pub split: u64,
    pub seed: u64,
    pub hidden: usize,
    pub width: usize,
    pub act: Act,
    pub lr: f32,
    pub decay: f32,
}

impl Default for Defaults {
    fn default() -> Self {
        Defaults { kind: 0, p: 17, fraction: 0.8, split: 1, seed: 29, hidden: 1, width: 128, act: Act::Relu, lr: 0.01, decay: 1.0 }
    }
}

/// How the network is doing on the whole table.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Score {
    /// Cross-entropy on the training cells and on the held-out cells.
    pub loss: [f32; 2],
    /// Fraction answered correctly, likewise.
    pub accuracy: [f32; 2],
}

/// Run the whole table through `net` (its activations stay available for drawing),
/// record its answer for every cell in `pred`, and score it.
pub fn evaluate(net: &mut Mlp, task: &Task, pred: &mut Vec<u8>) -> Score {
    let (p, n) = (task.p, task.cells());
    let logits = net.forward_hot(&task.active, 2, n);
    pred.clear();
    pred.resize(n, 0);
    let (mut loss, mut hits, mut count) = ([0.0f64; 2], [0usize; 2], [0usize; 2]);
    for s in 0..n {
        let at = |c: usize| logits[c * n + s];
        let mut best = 0;
        for c in 1..p {
            if at(c) > at(best) {
                best = c;
            }
        }
        let top = at(best);
        let total: f32 = (0..p).map(|c| fast_exp((at(c) - top).max(-60.0))).sum();
        let right = task.answer[s] as usize;
        let group = task.held_out[s] as usize;
        loss[group] += (total.ln() + top - at(right)) as f64;
        hits[group] += (best == right) as usize;
        count[group] += 1;
        pred[s] = best as u8;
    }
    let mean = |v: f64, n: usize| if n == 0 { 0.0 } else { (v / n as f64) as f32 };
    Score { loss: [mean(loss[0], count[0]), mean(loss[1], count[1])], accuracy: [mean(hits[0] as f64, count[0]), mean(hits[1] as f64, count[1])] }
}

/// One full-batch training step's worth of forward and backward; returns the loss.
pub fn full_batch(net: &mut Mlp, task: &Task) -> f32 {
    net.forward_hot(&task.train_active, 2, task.train.len());
    net.backward(&task.train_y, Loss::Softmax)
}

/// For each first-layer neuron, how its input weights divide among the frequencies
/// k = 1..p/2: `out[j * (p / 2) + k - 1]` is the share of neuron j's oscillating power
/// at frequency k, the power being the squared size of the discrete Fourier transform
/// of its weights along `a` plus that of its weights along `b`. Each neuron's shares
/// sum to one. A network that has found the periodic solution to a + b has neurons
/// that each put nearly everything on one frequency.
pub fn spectra(net: &Mlp, p: usize, out: &mut Vec<f32>) {
    let (n_in, k_max, neurons) = (2 * p, p / 2, net.sizes[1]);
    out.clear();
    out.resize(neurons * k_max, 0.0);
    // cos and sin of 2 pi k a / p for k = 1..=k_max and every a
    let waves: Vec<(f32, f32)> = (0..k_max * p)
        .map(|i| {
            let angle = std::f32::consts::TAU * (((i / p + 1) * (i % p)) % p) as f32 / p as f32;
            (angle.cos(), angle.sin())
        })
        .collect();
    for (j, share) in out.chunks_exact_mut(k_max.max(1)).enumerate().take(if k_max == 0 { 0 } else { neurons }) {
        let row = &net.w[0][j * n_in..(j + 1) * n_in];
        for (k, power) in share.iter_mut().enumerate() {
            for half in row.chunks_exact(p) {
                let (mut re, mut im) = (0.0, 0.0);
                for (&v, &(c, s)) in half.iter().zip(&waves[k * p..(k + 1) * p]) {
                    re += v * c;
                    im += v * s;
                }
                *power += re * re + im * im;
            }
        }
        let total: f32 = share.iter().sum();
        if total > 1e-20 {
            share.iter_mut().for_each(|s| *s /= total);
        }
    }
}

/// The share of its power a first-layer neuron must give one frequency to count as
/// devoted to it: more than a neuron with random weights gives any one frequency,
/// bar once in a hundred; and at least half, so that a neuron is devoted to one
/// frequency at most. (With random weights the shares of the `k_max` frequencies are
/// Dirichlet(2, ..., 2), so a given one of them exceeds `t` with probability
/// `(1 - t)^n + n t (1 - t)^(n - 1)`, `n = 2 k_max - 1`. That is 0.94 for p = 5,
/// 0.78 for p = 7, 0.55 for p = 11, and from p = 13 on the half decides.)
pub fn devotion(k_max: usize) -> f32 {
    let n = (2 * k_max.max(1) - 1) as i32;
    let chance = |t: f32| (1.0 - t).powi(n) + n as f32 * t * (1.0 - t).powi(n - 1);
    let (mut lo, mut hi) = (0.0f32, 1.0f32);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if chance(mid) > 0.01 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    hi.max(0.5)
}

/// How much the first layer uses each frequency: `out[k - 1]` is the share of its
/// neurons devoted to frequency k (see [`devotion`]), from their `spectra`. Unlike the
/// rings of [`ring`], this tells a layer that has learned a frequency from one that
/// has not: random weights give about 0.01 or less to a frequency, the default run's
/// trained network 0.09 to 0.16 to each of the eight (every neuron devoted to one),
/// and a network trained on a · b nothing at all.
pub fn devoted(spectra: &[f32], k_max: usize, out: &mut Vec<f32>) {
    out.clear();
    out.resize(k_max, 0.0);
    if k_max == 0 || spectra.is_empty() {
        return;
    }
    let (t, rows) = (devotion(k_max), spectra.chunks_exact(k_max));
    let n = rows.len() as f32;
    for row in rows {
        for (o, &share) in out.iter_mut().zip(row) {
            if share > t {
                *o += 1.0 / n;
            }
        }
    }
}

/// cos and sin of `2 pi k a / p` for a = 0..p.
fn wave(k: usize, p: usize) -> Vec<(f32, f32)> {
    (0..p)
        .map(|a| {
            let angle = std::f32::consts::TAU * ((k * a) % p) as f32 / p as f32;
            (angle.cos(), angle.sin())
        })
        .collect()
}

/// How the first layer lays out the numbers 0..p along frequency `k`: frequency k's
/// part of the embedding.
///
/// Number `a` reaches the first layer as a vector of `width` weights (one per neuron):
/// its embedding. If the layer has learned the wave of frequency `k`, the embeddings
/// are `cos(2 pi k a / p) A + sin(2 pi k a / p) B` for two fixed vectors A and B, plus
/// whatever other frequencies add. This returns each number's coordinates (x, y) in
/// the plane spanned by A and B, in units in which a pure wave gives exactly the
/// point at angle `2 pi k a / p` on the unit circle; and the strength of the wave,
/// `|A|^2 + |B|^2`. (A and B are read off by Fourier analysis of the embeddings; the
/// coordinates are the least-squares fit of each centred embedding by A and B.)
///
/// The picture alone does not say whether the layer has learned the frequency. In a
/// layer of many neurons the other frequencies' parts are nearly at right angles to A
/// and B, so nearly any layer comes out close to a regular star, random weights
/// included (the playground's untrained first layer puts every ring within 0.25 to
/// 0.4, root mean square, of the star). What tells them apart is [`devoted`].
pub fn ring(net: &Mlp, p: usize, k: usize) -> (Vec<glam::Vec2>, f32) {
    let (n_in, neurons) = (2 * p, net.sizes[1]);
    let waves = wave(k, p);
    let embedding = |a: usize, j: usize| net.w[0][j * n_in + a];
    let (mut va, mut vb, mut mean) = (vec![0.0f32; neurons], vec![0.0f32; neurons], vec![0.0f32; neurons]);
    for j in 0..neurons {
        for (a, &(c, s)) in waves.iter().enumerate() {
            let e = embedding(a, j);
            va[j] += e * c;
            vb[j] += e * s;
            mean[j] += e;
        }
        va[j] *= 2.0 / p as f32;
        vb[j] *= 2.0 / p as f32;
        mean[j] /= p as f32;
    }
    let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(a, b)| a * b).sum::<f32>();
    let (aa, ab, bb) = (dot(&va, &va), dot(&va, &vb), dot(&vb, &vb));
    let det = aa * bb - ab * ab;
    let points = (0..p)
        .map(|a| {
            if det < 1e-12 * (aa + bb).powi(2) || det <= 0.0 {
                return glam::Vec2::ZERO;
            }
            let (mut ea, mut eb) = (0.0, 0.0);
            for j in 0..neurons {
                let e = embedding(a, j) - mean[j];
                ea += e * va[j];
                eb += e * vb[j];
            }
            glam::Vec2::new((bb * ea - ab * eb) / det, (aa * eb - ab * ea) / det)
        })
        .collect();
    (points, aa + bb)
}

/// Take frequency `k` out of the first layer: every neuron's weights along `a` and
/// along `b` lose their component at that frequency (the mean and every other
/// frequency are untouched). Keep a copy of `net.w[0]` to put it back.
pub fn remove_frequency(net: &mut Mlp, p: usize, k: usize) {
    let (n_in, waves) = (2 * p, wave(k, p));
    for row in net.w[0].chunks_exact_mut(n_in) {
        for half in row.chunks_exact_mut(p) {
            let (mut re, mut im) = (0.0, 0.0);
            for (&v, &(c, s)) in half.iter().zip(&waves) {
                re += v * c;
                im += v * s;
            }
            let scale = 2.0 / p as f32;
            for (v, &(c, s)) in half.iter_mut().zip(&waves) {
                *v -= scale * (re * c + im * s);
            }
        }
    }
}

/// What [`concentration`] comes to for random weights, for each modulus in `MODULI`:
/// measured, over twenty random first layers of 128 neurons each.
const RANDOM_CONCENTRATION: [f32; 6] = [0.685, 0.534, 0.378, 0.335, 0.272, 0.214];

/// The concentration of a random first layer for modulus `p` (see
/// `RANDOM_CONCENTRATION`), if `p` is one of `MODULI`.
pub fn random_concentration(p: usize) -> Option<f32> {
    MODULI.iter().position(|&m| m == p).map(|i| RANDOM_CONCENTRATION[i])
}

/// The largest share of each neuron, averaged over neurons: 1 when every neuron uses
/// a single frequency. For random weights it is well above 1/(p/2), the share of an
/// average frequency, because the largest of a neuron's shares is larger than their
/// average: 0.27 for p = 17 and 0.69 for p = 5 (see `RANDOM_CONCENTRATION`).
pub fn concentration(spectra: &[f32], k_max: usize) -> f32 {
    if k_max == 0 || spectra.is_empty() {
        return 0.0;
    }
    let rows = spectra.chunks_exact(k_max);
    let n = rows.len() as f32;
    rows.map(|r| r.iter().copied().fold(0.0, f32::max)).sum::<f32>() / n
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::nn::Optim;

    #[test]
    fn tables_are_what_they_say() {
        for (kind, name) in OPS.iter().enumerate() {
            for p in MODULI {
                let t = Task::new(kind, p, 0.6, 1);
                assert_eq!(t.train.len() + t.test.len(), p * p);
                assert!(t.answer.iter().all(|&c| (c as usize) < p), "{name} mod {p}");
                assert_eq!(t.train.len(), (0.6 * (p * p) as f32).round() as usize);
                // every cell is in exactly one of the two sets
                let mut seen = vec![0; p * p];
                t.train.iter().chain(&t.test).for_each(|&s| seen[s as usize] += 1);
                assert!(seen.iter().all(|&c| c == 1));
            }
        }
        let t = Task::new(0, 7, 0.5, 3);
        assert_eq!(t.answer[3 * 7 + 6], 2); // 6 + 3 = 9 = 2 mod 7
        assert_eq!(&t.active[2 * (3 * 7 + 6)..][..2], &[6, 7 + 3]);
        assert_eq!(answer(1, 2, 5, 7), 4); // 2 - 5 = -3 = 4 mod 7
        assert_eq!(answer(2, 4, 5, 7), 6);
    }

    #[test]
    fn spectra_find_planted_frequencies() {
        let p = 11;
        let mut rng = Rng::new(1);
        let mut net = Mlp::new(&[2 * p, 3, p], Act::Relu, &mut rng);
        let mut s = Vec::new();
        spectra(&net, p, &mut s);
        assert_eq!(s.len(), 3 * 5);
        let random = concentration(&s, 5);
        // neuron j gets a pure wave of frequency j + 2 (with a phase and an offset)
        for j in 0..3 {
            for i in 0..2 * p {
                net.w[0][j * 2 * p + i] = 0.7 + (std::f32::consts::TAU * (j + 2) as f32 * (i % p) as f32 / p as f32 + j as f32).cos();
            }
        }
        spectra(&net, p, &mut s);
        for j in 0..3 {
            assert!((s[j * 5..(j + 1) * 5].iter().sum::<f32>() - 1.0).abs() < 1e-4);
            assert!(s[j * 5 + j + 1] > 0.999, "{s:?}");
        }
        assert!(concentration(&s, 5) > 0.999 && random < 0.8);
    }

    /// How many of the first layer's neurons are devoted to some frequency (see
    /// `devoted`), and to the frequency fewest are devoted to: (all, fewest).
    fn devotion_of(net: &Mlp, p: usize) -> (f32, f32) {
        let (mut s, mut d) = (Vec::new(), Vec::new());
        spectra(net, p, &mut s);
        devoted(&s, p / 2, &mut d);
        (d.iter().sum(), d.iter().copied().fold(f32::MAX, f32::min))
    }

    /// The playground's default settings do what its text says: the pairs trained on
    /// are all right within two hundred steps while the held-out pairs are still all
    /// wrong, the held-out pairs follow over the next few thousand steps, and the
    /// first layer's neurons each settle on a frequency (half of them within the
    /// first hundred steps, all of them by the end, and every frequency used).
    #[test]
    fn default_run_memorises_then_generalises() {
        let d = Defaults::default();
        let task = Task::new(d.kind, d.p, d.fraction, d.split);
        let mut rng = Rng::new(d.seed);
        let mut net = Mlp::new(&task.sizes(d.hidden, d.width), d.act, &mut rng);
        let (mut pred, mut s) = (Vec::new(), Vec::new());
        spectra(&net, d.p, &mut s);
        let before = concentration(&s, d.p / 2);
        let train = |net: &mut Mlp, steps: usize| {
            for _ in 0..steps {
                full_batch(net, &task);
                net.step(Optim::Adam, d.lr, 0.9, d.decay);
            }
        };
        assert!(devotion_of(&net, d.p).0 < 0.02, "random weights: {:?}", devotion_of(&net, d.p));
        train(&mut net, 100);
        let (half, _) = devotion_of(&net, d.p);
        assert!((0.4..0.6).contains(&half), "devoted to a frequency after 100 steps: {half}");
        train(&mut net, 100);
        let early = evaluate(&mut net, &task, &mut pred);
        assert!(early.accuracy[0] > 0.999 && early.accuracy[1] < 0.1, "after 200 steps: {early:?}");
        train(&mut net, 3000);
        let late = evaluate(&mut net, &task, &mut pred);
        assert!(late.accuracy[0] > 0.999 && late.accuracy[1] > 0.6, "after 3200 steps: {late:?}");
        spectra(&net, d.p, &mut s);
        let after = concentration(&s, d.p / 2);
        println!("held-out accuracy {:.2} -> {:.2}; frequency concentration {before:.2} -> {after:.2}", early.accuracy[1], late.accuracy[1]);
        assert!(after > before + 0.2, "concentration {before} -> {after}");
        let (all, fewest) = devotion_of(&net, d.p);
        assert!(all > 0.99 && fewest > 0.08, "devoted after 3200 steps: {all} in all, {fewest} to the least used frequency");

        // what the scene says about removing frequencies from the first layer: one
        // alone costs little (the answer is a vote of all of them), and with all of
        // them gone the network is guessing
        let overall = |s: &Score| (s.accuracy[0] * task.train.len() as f32 + s.accuracy[1] * task.test.len() as f32) / task.cells() as f32;
        let (base, saved) = (overall(&late), net.w[0].clone());
        for k in 1..=d.p / 2 {
            remove_frequency(&mut net, d.p, k);
            let alone = overall(&evaluate(&mut net, &task, &mut pred));
            assert!(alone > base - 0.15, "without frequency {k}: {alone} (with it: {base})");
            net.w[0].copy_from_slice(&saved);
        }
        (1..=d.p / 2).for_each(|k| remove_frequency(&mut net, d.p, k));
        let none = overall(&evaluate(&mut net, &task, &mut pred));
        assert!(none < 2.0 / d.p as f32, "with every frequency removed: {none}");
    }

    /// Random first layers: their concentration is what `RANDOM_CONCENTRATION` says,
    /// for every modulus, and next to none of their neurons count as devoted to a
    /// frequency. Yet their rings come out close to regular stars, which is why the
    /// rings are not to be read on their own (see `ring`).
    #[test]
    fn random_layers_have_no_frequencies_but_look_like_stars() {
        for p in MODULI {
            let (mut mean, mut most) = (0.0, 0.0f32);
            let (mut s, mut d) = (Vec::new(), Vec::new());
            for seed in 0..20 {
                let net = Mlp::new(&Task::new(0, p, 0.8, 1).sizes(1, 128), Act::Relu, &mut Rng::new(seed));
                spectra(&net, p, &mut s);
                mean += concentration(&s, p / 2) / 20.0;
                devoted(&s, p / 2, &mut d);
                most = d.iter().copied().fold(most, f32::max);
            }
            let tabled = random_concentration(p).unwrap_or(0.0);
            assert!((mean - tabled).abs() < 0.015, "p = {p}: random concentration {mean}, tabled {tabled}");
            assert!(most < 0.05, "p = {p}: up to {most} of a random layer devoted to one frequency");
        }
        let d = Defaults::default();
        for seed in 0..3 {
            let net = Mlp::new(&Task::new(0, d.p, d.fraction, d.split).sizes(d.hidden, d.width), d.act, &mut Rng::new(seed));
            for k in 1..=d.p / 2 {
                let (points, _) = ring(&net, d.p, k);
                let star = |a: usize| {
                    let angle = std::f32::consts::TAU * ((k * a) % d.p) as f32 / d.p as f32;
                    glam::Vec2::new(angle.cos(), angle.sin())
                };
                let rms = (points.iter().enumerate().map(|(a, q)| (*q - star(a)).length_squared()).sum::<f32>() / d.p as f32).sqrt();
                assert!(rms < 0.45, "seed {seed}, frequency {k}: {rms} from the star");
            }
        }
    }

    /// Held-out accuracy at a series of step counts, for choosing defaults that show
    /// memorising and generalising as two separate phases (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn explore() {
        const MARKS: [usize; 11] = [50, 100, 200, 400, 800, 1600, 3200, 6400, 9600, 12800, 19200];
        let trajectory = |p: usize, frac: f32, act: Act, width: usize, lr: f32, decay: f32, gain: f32, seed: u64| -> String {
            let task = Task::new(0, p, frac, 7 + seed);
            let mut rng = Rng::new(11 + seed);
            let mut net = Mlp::new(&task.sizes(1, width), act, &mut rng);
            net.scale_weights(gain);
            let mut pred = Vec::new();
            let mut line = format!("p={p} f={frac} {:<9} w={width:<3} lr={lr} wd={decay:<4} gain={gain} seed={seed}:", act.name());
            let start = std::time::Instant::now();
            for step in 1..=*MARKS.last().unwrap() {
                full_batch(&mut net, &task);
                net.step(Optim::Adam, lr, 0.9, decay);
                if MARKS.contains(&step) {
                    let s = evaluate(&mut net, &task, &mut pred);
                    line += &format!(" {:>3.0}/{:<3.0}", s.accuracy[0] * 100.0, s.accuracy[1] * 100.0);
                }
            }
            line + &format!("  {:.0} us/step", start.elapsed().as_secs_f64() * 1e6 / *MARKS.last().unwrap() as f64)
        };
        let mut jobs = Vec::new();
        for seed in 0..2 {
            for &(p, frac, act, width, lr, decay, gain) in &[
                (23usize, 0.8f32, Act::Relu, 128usize, 0.01f32, 1.0f32, 1.0f32),
                (23, 0.7, Act::Relu, 128, 0.01, 1.0, 1.0),
                (23, 0.6, Act::Relu, 128, 0.01, 1.0, 1.0),
                (23, 0.8, Act::Relu, 128, 0.01, 0.3, 1.0),
                (23, 0.8, Act::Relu, 128, 0.01, 1.0, 2.0),
                (23, 0.8, Act::Relu, 128, 0.01, 0.3, 2.0),
                (23, 0.7, Act::Relu, 128, 0.01, 0.3, 2.0),
                (23, 0.8, Act::Relu, 64, 0.01, 1.0, 1.0),
                (23, 0.8, Act::Relu, 64, 0.01, 1.0, 2.0),
                (17, 0.8, Act::Relu, 128, 0.01, 1.0, 1.0),
                (17, 0.8, Act::Relu, 128, 0.01, 1.0, 2.0),
                (17, 0.8, Act::Relu, 64, 0.01, 1.0, 2.0),
                (23, 0.8, Act::Hermite2, 128, 0.01, 0.3, 1.0),
                (23, 0.8, Act::Hermite2, 128, 0.03, 0.3, 1.0),
            ] {
                jobs.push((p, frac, act, width, lr, decay, gain, seed));
            }
        }
        let lines: Vec<String> = std::thread::scope(|scope| {
            let handles: Vec<_> = jobs
                .iter()
                .map(|&(p, frac, act, width, lr, decay, gain, seed)| scope.spawn(move || trajectory(p, frac, act, width, lr, decay, gain, seed)))
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        println!("accuracy (trained-on / held-out, %) after {MARKS:?} steps");
        for l in lines {
            println!("{l}");
        }
    }

    /// The pictures of the first layer, checked on weights built by hand: a layer made
    /// of pure waves puts the numbers on the unit circle, and removing that frequency
    /// removes the wave and nothing else.
    #[test]
    fn rings_and_frequency_removal() {
        let (p, width) = (11usize, 6usize);
        let task = Task::new(0, p, 0.8, 1);
        let mut rng = Rng::new(4);
        let mut net = Mlp::new(&task.sizes(1, width), Act::Relu, &mut rng);
        // neurons 0..3 carry frequency 2, neurons 3..6 frequency 5, with random phases
        let phases: Vec<f32> = (0..width).map(|_| rng.range(0.0, std::f32::consts::TAU)).collect();
        for j in 0..width {
            let k = if j < 3 { 2 } else { 5 };
            for a in 0..p {
                let angle = std::f32::consts::TAU * (k * a) as f32 / p as f32;
                net.w[0][j * 2 * p + a] = 0.3 + 1.5 * (angle + phases[j]).cos();
                net.w[0][j * 2 * p + p + a] = (angle - phases[j]).sin();
            }
        }
        for k in [2, 5] {
            let (points, strength) = ring(&net, p, k);
            assert!(strength > 1.0);
            for (a, q) in points.iter().enumerate() {
                let angle = std::f32::consts::TAU * ((k * a) % p) as f32 / p as f32;
                assert!((q.x - angle.cos()).abs() < 1e-3 && (q.y - angle.sin()).abs() < 1e-3, "k={k} a={a}: {q:?}");
            }
        }
        // a frequency nobody uses has no strength
        assert!(ring(&net, p, 3).1 < 1e-6);
        let mut s = Vec::new();
        spectra(&net, p, &mut s);
        assert!((s[1] - 1.0).abs() < 1e-4, "neuron 0 is all frequency 2: {:?}", &s[..5]);
        let before = net.w[0].clone();
        remove_frequency(&mut net, p, 2);
        // neurons of frequency 2 are left with their mean only; the others are untouched
        for j in 0..width {
            for a in 0..2 * p {
                let (was, is) = (before[j * 2 * p + a], net.w[0][j * 2 * p + a]);
                if j < 3 {
                    let mean = if a < p { 0.3 } else { 0.0 };
                    assert!((is - mean).abs() < 1e-4, "neuron {j} input {a}: {is}");
                } else {
                    assert!((is - was).abs() < 1e-4);
                }
            }
        }
    }

    /// The other "things to try" of the arithmetic view, held to account (the first
    /// is the test above). Each is the default run with one thing changed.
    #[test]
    fn the_other_tables_do_what_their_captions_say() {
        let d = Defaults::default();
        let run = |kind: usize, fraction: f32, decay: f32, steps: usize| {
            let task = Task::new(kind, d.p, fraction, d.split);
            let mut net = Mlp::new(&task.sizes(d.hidden, d.width), d.act, &mut Rng::new(d.seed));
            for _ in 0..steps {
                full_batch(&mut net, &task);
                net.step(Optim::Adam, d.lr, 0.9, decay);
            }
            let (mut pred, mut s) = (Vec::new(), Vec::new());
            let score = evaluate(&mut net, &task, &mut pred);
            spectra(&net, d.p, &mut s);
            (score, concentration(&s, d.p / 2), devotion_of(&net, d.p).0)
        };
        // without weight decay it memorises and stays there, and only about half the
        // neurons settle on a frequency (with weight decay all of them have by now);
        // the same after much longer in the slow test below
        let (score, sharpness, devoted) = run(0, 0.8, 0.0, 3200);
        let random = random_concentration(d.p).unwrap_or(0.0);
        assert!(score.accuracy[0] > 0.999 && score.accuracy[1] < 0.05, "no weight decay: {score:?}");
        assert!(
            sharpness > random + 0.15 && sharpness < 0.6 && (0.35..0.6).contains(&devoted),
            "no weight decay: concentration {sharpness}, devoted {devoted}"
        );
        // half the table is too little to pin the rule down
        let (score, _, _) = run(0, 0.5, d.decay, 3200);
        assert!(score.accuracy[1] < 0.1 && score.accuracy[0] > 0.999, "half the table: {score:?}");
        // multiplication comes right too, but not on single frequencies: less
        // concentrated than random weights, and no neuron devoted to one
        let (score, sharpness, devoted) = run(2, 0.8, d.decay, 3200);
        assert!(score.accuracy[1] > 0.85 && sharpness < random && devoted < 0.02, "a * b: {score:?}, concentration {sharpness}, devoted {devoted}");
        // max(a, b) has no surprise: the held-out cells are right almost at once
        let (score, _, _) = run(5, 0.8, d.decay, 200);
        assert!(score.accuracy[1] > 0.85, "max(a, b) after 200 steps: {score:?}");
    }

    /// "Without weight decay", as a visitor sees it after a while: twenty thousand
    /// steps (seventeen seconds at the playground's rate) on and the held-out cells
    /// are still wrong, and still only about half of the first layer's neurons are
    /// devoted to a frequency; well above random weights, well short of the run with
    /// weight decay. Three starts. (Slow: run with --ignored.)
    #[test]
    #[ignore]
    fn without_weight_decay_the_rings_only_partly_order() {
        let d = Defaults::default();
        let task = Task::new(d.kind, d.p, d.fraction, d.split);
        let random = random_concentration(d.p).unwrap_or(0.0);
        std::thread::scope(|scope| {
            for seed in [d.seed, 1, 2] {
                let task = &task;
                scope.spawn(move || {
                    let mut net = Mlp::new(&task.sizes(d.hidden, d.width), d.act, &mut Rng::new(seed));
                    for _ in 0..20_000 {
                        full_batch(&mut net, task);
                        net.step(Optim::Adam, d.lr, 0.9, 0.0);
                    }
                    let score = evaluate(&mut net, task, &mut Vec::new());
                    let mut s = Vec::new();
                    spectra(&net, d.p, &mut s);
                    let (sharpness, (devoted, _)) = (concentration(&s, d.p / 2), devotion_of(&net, d.p));
                    println!("seed {seed}: {score:?}, concentration {sharpness:.3}, devoted {devoted:.2}");
                    assert!(score.accuracy[0] > 0.999 && score.accuracy[1] < 0.05, "seed {seed}: {score:?}");
                    assert!(sharpness > random + 0.2 && sharpness < 0.65, "seed {seed}: concentration {sharpness} (random weights {random})");
                    assert!((0.4..0.7).contains(&devoted), "seed {seed}: {devoted} of the neurons devoted to a frequency");
                });
            }
        });
    }

    /// The numbers behind those captions, at a series of step counts and for a few
    /// more variations (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn experiments() {
        let d = Defaults::default();
        let run = |name: &str, kind: usize, fraction: f32, decay: f32, gain: f32| {
            let task = Task::new(kind, d.p, fraction, d.split);
            let mut rng = Rng::new(d.seed);
            let mut net = Mlp::new(&task.sizes(d.hidden, d.width), d.act, &mut rng);
            net.scale_weights(gain);
            let mut pred = Vec::new();
            let mut line = format!("{name:<28}");
            let mut done = 0;
            for mark in [100usize, 200, 400, 800, 1600, 3200, 6400, 12800] {
                for _ in done..mark {
                    full_batch(&mut net, &task);
                    net.step(Optim::Adam, d.lr, 0.9, decay);
                }
                done = mark;
                let s = evaluate(&mut net, &task, &mut pred);
                line += &format!("  {mark}: {:.0}/{:.0}", s.accuracy[0] * 100.0, s.accuracy[1] * 100.0);
            }
            let mut s = Vec::new();
            spectra(&net, d.p, &mut s);
            let base = evaluate(&mut net, &task, &mut pred);
            let saved = net.w[0].clone();
            let cuts: Vec<String> = (1..=d.p / 2)
                .map(|k| {
                    remove_frequency(&mut net, d.p, k);
                    let sc = evaluate(&mut net, &task, &mut pred);
                    net.w[0].copy_from_slice(&saved);
                    format!("{:.0}", (sc.accuracy[0] * task.train.len() as f32 + sc.accuracy[1] * task.test.len() as f32) / task.cells() as f32 * 100.0)
                })
                .collect();
            let strengths: Vec<String> = (1..=d.p / 2).map(|k| format!("{:.1}", ring(&net, d.p, k).1)).collect();
            // frequencies removed one after another, weakest effect first is not known: in order
            let mut running = Vec::new();
            for k in 1..=d.p / 2 {
                remove_frequency(&mut net, d.p, k);
                let sc = evaluate(&mut net, &task, &mut pred);
                running
                    .push(format!("{:.0}", (sc.accuracy[0] * task.train.len() as f32 + sc.accuracy[1] * task.test.len() as f32) / task.cells() as f32 * 100.0));
            }
            net.w[0].copy_from_slice(&saved);
            println!("    right with frequencies 1..k all removed [{}]", running.join(" "));
            println!(
                "{line}\n    concentration {:.2}; devoted {:.2}; right {:.0}/{:.0}; right with each frequency removed [{}]; strengths [{}]",
                concentration(&s, d.p / 2),
                devotion_of(&net, d.p).0,
                base.accuracy[0] * 100.0,
                base.accuracy[1] * 100.0,
                cuts.join(" "),
                strengths.join(" ")
            );
        };
        println!("(steps: % right on trained-on pairs / held-out pairs)");
        run("default", 0, 0.8, d.decay, 1.0);
        run("no weight decay", 0, 0.8, 0.0, 1.0);
        run("weight decay 0.1", 0, 0.8, 0.1, 1.0);
        run("half the table", 0, 0.5, d.decay, 1.0);
        run("60 % of the table", 0, 0.6, d.decay, 1.0);
        run("small initial weights", 0, 0.8, d.decay, 0.3);
        run("large initial weights", 0, 0.8, d.decay, 3.0);
        run("a - b", 1, 0.8, d.decay, 1.0);
        run("a * b", 2, 0.8, d.decay, 1.0);
        run("max(a, b)", 5, 0.8, d.decay, 1.0);
    }
}

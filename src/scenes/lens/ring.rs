//! The toy model of Ostrow, Qian, Liu and Fiete, "A Mathematical Theory of the Jacobian
//! Lens" (2026): hidden states moving on rings, the exact Bayes filter that follows
//! them from occasional glimpses, and the Jacobian lens of that filter, in closed form
//! and measured.
//!
//! **The process.** Each latent is a position `X` on a ring of `n` sites. From one
//! token to the next it carries on with probability λ (the persistence), moving `v`
//! sites and slipping one site either way with probability `q` each; otherwise it
//! jumps to a site chosen uniformly. A token shows every latent at once with
//! probability ε and is the null token otherwise. A *keyed* latent also shows a key
//! `W = ±1` in every token (`+1` with probability `(1 + ψ)/2`), and is itself shown
//! shifted half-way round its ring whenever the previous token's key was `−1`.
//!
//! **The filter.** Because a reveal is exact and a null token says nothing, the best
//! forecast of the next state is the last revealed state pushed forward by the
//! dynamics. Fourier mode `j` of that forecast, `m = E[φ_j(X)]` with
//! `φ_j(x) = e^{2πi j x / n}`, is multiplied at every token by
//! `α_j = λ c_j e^{i j θ}`, where `θ = 2π v / n` and `c_j = 1 − 2q + 2q cos(2π j / n)`.
//!
//! **The lens.** The response of the forecast `τ` tokens after a revealed token to a
//! nudge of that token's evidence is `K(c, τ) = ±α^{τ+1}` until the next reveal and
//! zero after it (the sign is the key). Summing over the lags `τ = 1..H` and averaging
//! over contexts `c` (how soon the next reveal comes, and the key) gives the lens; its
//! singular value for this mode is the product of three factors (see [`Factors`]).
//!
//! Conventions: lags run from 1 to `H`, and the factors use the sum over lags rather
//! than the average (as in the paper's section 3; the common factor `1/H` changes no
//! ordering). Tokens number `(1 + Π n) · 2^k` for `k` keyed latents: the null token or
//! the sites as shown, times the key bits. (The paper counts `(1 + 2 Π n) · 2^k`; the
//! information carried is the same.)

use crate::math::complex::C32;
use crate::math::rng::Rng;
use std::f32::consts::{PI, TAU};

/// The largest ring, and the most latents a process may have.
pub const MAX_SITES: usize = 32;
pub const MAX_LATENTS: usize = 16;

/// An angle brought into (−π, π].
pub fn wrap(angle: f32) -> f32 {
    let a = angle.rem_euclid(TAU);
    if a > PI {
        a - TAU
    } else {
        a
    }
}

/// One hidden variable: a position on a ring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Latent {
    /// Sites on the ring (2 to [`MAX_SITES`]; even if the latent is keyed).
    pub n: usize,
    /// Persistence λ: the chance per token of carrying on rather than jumping anywhere.
    pub lambda: f32,
    /// Velocity: sites moved per token while it carries on.
    pub v: usize,
    /// Key bias ψ if the latent is keyed: the key is +1 with probability (1 + ψ)/2.
    pub psi: Option<f32>,
}

impl Latent {
    /// The turn per token, θ = 2π v / n, in (−π, π].
    pub fn theta(&self) -> f32 {
        wrap(TAU * (self.v % self.n) as f32 / self.n as f32)
    }

    /// How far round the ring the state is shown from where it is.
    fn shift(&self, minus: bool) -> usize {
        if minus && self.psi.is_some() {
            self.n / 2
        } else {
            0
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Process {
    pub latents: Vec<Latent>,
    /// Diffusion q: the chance of slipping one site forward, and the same back (0 to 1/2).
    pub q: f32,
    /// Reveal probability ε.
    pub epsilon: f32,
}

impl Process {
    /// The diffusion factor c_j of latent `i`: what slipping does to Fourier mode `j`.
    pub fn diffusion_factor(&self, i: usize, j: usize) -> f32 {
        1.0 - 2.0 * self.q + 2.0 * self.q * (TAU * j as f32 / self.latents[i].n as f32).cos()
    }

    /// α_j of latent `i`: what one token without a reveal does to mode `j` of the forecast.
    pub fn alpha(&self, i: usize, j: usize) -> C32 {
        let l = &self.latents[i];
        C32::polar(l.lambda * self.diffusion_factor(i, j), wrap(j as f32 * l.theta()))
    }

    /// The turn per token of mode `j` of latent `i`'s forecast: the angle of α_j.
    pub fn turn(&self, i: usize, j: usize) -> f32 {
        self.alpha(i, j).arg()
    }

    fn keyed(&self) -> usize {
        self.latents.iter().filter(|l| l.psi.is_some()).count()
    }

    /// The number of different tokens.
    pub fn vocab(&self) -> usize {
        (1 + self.latents.iter().map(|l| l.n).product::<usize>()) << self.keyed()
    }

    /// The token that shows the sites `shown` (or nothing) and the keys `minus`
    /// (true for a key of −1; ignored for latents without a key).
    pub fn encode(&self, shown: Option<&[usize]>, minus: &[bool]) -> u32 {
        let mut part = 0;
        if let Some(sites) = shown {
            let mut stride = 1;
            part = 1;
            for (l, &y) in self.latents.iter().zip(sites) {
                part += y * stride;
                stride *= l.n;
            }
        }
        let (mut bits, mut k) = (0, 0);
        for (l, &m) in self.latents.iter().zip(minus) {
            if l.psi.is_some() {
                bits |= (m as usize) << k;
                k += 1;
            }
        }
        ((part << self.keyed()) | bits) as u32
    }

    /// Whether `token` shows the latents, and if so the site each is shown at (in `out`).
    pub fn shown(&self, token: u32, out: &mut [usize]) -> bool {
        let part = token as usize >> self.keyed();
        if part == 0 {
            return false;
        }
        let mut rest = part - 1;
        for (l, y) in self.latents.iter().zip(out) {
            *y = rest % l.n;
            rest /= l.n;
        }
        true
    }

    /// Whether `token` shows a key of −1 for latent `i` (false for a latent without a key).
    pub fn key_minus(&self, token: u32, i: usize) -> bool {
        if self.latents[i].psi.is_none() {
            return false;
        }
        let k = self.latents[..i].iter().filter(|l| l.psi.is_some()).count();
        (token >> k) & 1 == 1
    }

    /// The probability of `token` for a predictor that expects latent `i` to be
    /// shown at site `y` with probability `site(i, y)`: a reveal or not by ε, the
    /// sites if it is one, and the keys by their bias.
    fn chance(&self, token: u32, site: impl Fn(usize, usize) -> f32) -> f32 {
        let mut shown = [0usize; MAX_LATENTS];
        let revealed = self.shown(token, &mut shown);
        let mut chance = if revealed { self.epsilon } else { 1.0 - self.epsilon };
        for (i, l) in self.latents.iter().enumerate() {
            if revealed {
                chance *= site(i, shown[i]);
            }
            if let Some(psi) = l.psi {
                chance *= 0.5 * if self.key_minus(token, i) { 1.0 - psi } else { 1.0 + psi };
            }
        }
        chance
    }

    /// The probability a predictor with no memory gives `token`: every site of a
    /// revealed latent equally likely.
    pub fn memoryless_chance(&self, token: u32) -> f32 {
        self.chance(token, |i, _| 1.0 / self.latents[i].n as f32)
    }

    /// The tuning ω as mode `j` of latent `i` feels it. A mode and its mirror image
    /// (j and n − j) turn opposite ways and the lens takes whichever is nearer its
    /// tuning; for a mode that turns backwards that is the mirror image, which
    /// amounts to reversing ω.
    pub fn tuned(&self, i: usize, j: usize, omega: f32) -> f32 {
        if self.turn(i, j) < 0.0 {
            -omega
        } else {
            omega
        }
    }

    /// The three factors of the lens for mode `j` of latent `i`, over lags 1 to
    /// `window` (None: the long-window limit), for the lens tuned to frequency `omega`
    /// (0 for the plain lens; 0 to π; see [`Process::tuned`] for a mode that turns
    /// backwards).
    pub fn lens(&self, i: usize, j: usize, window: Option<usize>, omega: f32) -> Factors {
        // the key's half-turn of the ring reverses the odd modes and leaves the even ones alone
        let psi = if j % 2 == 1 { self.latents[i].psi.unwrap_or(1.0) } else { 1.0 };
        factors(self.alpha(i, j), self.epsilon, psi, window, self.tuned(i, j, omega))
    }

    /// Every direction the lens finds, strongest first: one per latent and Fourier
    /// mode. Modes come in mirror-image pairs (j and n − j, which turn opposite ways),
    /// and each pair is listed once, as mode j: under the plain lens the two have the
    /// same singular value; under a tuned one (ω ≠ 0) they do not, and the larger is
    /// the one listed (see [`Process::tuned`]). The mode half-way round an even ring
    /// is its own mirror image.
    pub fn spectrum(&self, window: Option<usize>, omega: f32) -> Vec<Direction> {
        let mut all = Vec::new();
        for (i, l) in self.latents.iter().enumerate() {
            for j in 1..=l.n / 2 {
                all.push(Direction { latent: i, mode: j, sigma: self.lens(i, j, window, omega).sigma() });
            }
        }
        all.sort_by(|a, b| b.sigma.total_cmp(&a.sigma));
        all
    }

    /// The singular value of mode 1 of latent `i` against its own persistence,
    /// everything else staying as it is: at λ = 0, 1/steps, 2/steps, ..., 1.
    pub fn sweep(&self, i: usize, steps: usize, window: Option<usize>, omega: f32) -> Vec<f32> {
        let mut trial = self.clone();
        (0..=steps)
            .map(|s| {
                trial.latents[i].lambda = s as f32 / steps as f32;
                trial.lens(i, 1, window, omega).sigma()
            })
            .collect()
    }

    /// Every persistence at which mode 1 of latent `i` would have singular value
    /// `sigma`: where the lens's order switches between that latent and one whose
    /// singular value is `sigma`. `sweep` is [`Process::sweep`] for the same latent
    /// and lens; each place where it passes `sigma` is then pinned down by bisection.
    /// Usually there is one such place, since a longer memory mostly means a larger
    /// singular value. But with a short window and a response that swings to and fro
    /// the singular value rises and falls again, and there are two; and there is none
    /// if `sigma` is out of reach, or zero (nothing is overtaken at λ = 0). With each
    /// place comes its direction: true where the singular value rises through
    /// `sigma` as λ grows (latent `i` overtakes), false where it falls back below.
    pub fn crossings(&self, i: usize, sweep: &[f32], sigma: f32, window: Option<usize>, omega: f32) -> Vec<(f32, bool)> {
        if sigma.is_nan() || sigma <= 0.0 || sweep.len() < 2 {
            return Vec::new();
        }
        let steps = (sweep.len() - 1) as f32;
        let mut trial = self.clone();
        let mut below = |lambda: f32| {
            trial.latents[i].lambda = lambda;
            trial.lens(i, 1, window, omega).sigma() < sigma
        };
        let mut found = Vec::new();
        for (s, pair) in sweep.windows(2).enumerate() {
            let rising = pair[0] < sigma;
            if rising == (pair[1] < sigma) {
                continue;
            }
            let (mut lo, mut hi) = (s as f32 / steps, (s + 1) as f32 / steps);
            for _ in 0..30 {
                let mid = 0.5 * (lo + hi);
                if below(mid) == rising {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            found.push((0.5 * (lo + hi), rising));
        }
        found
    }
}

/// The paper's closed form for the same switch (its Eq. for λ₀*): the persistence at
/// which a latent that does not turn and has no key, with diffusion factor `c` > 0,
/// reaches singular value `sigma` in the long-window limit. (Not for a finite window,
/// a tuned lens, or a mode that slipping reverses, c < 0.)
pub fn crossing_static(c: f32, epsilon: f32, sigma: f32) -> f32 {
    let keep = 1.0 - epsilon;
    let kappa = sigma / keep;
    (((keep * kappa).powi(2) + 4.0 * kappa).sqrt() - keep * kappa) / (2.0 * c)
}

/// One direction of the lens.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Direction {
    pub latent: usize,
    /// Fourier mode: how many waves fit round the ring.
    pub mode: usize,
    pub sigma: f32,
}

/// The three numbers whose product is a direction's singular value in the lens.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Factors {
    /// Magnitude a: the size of the response to a nudge, added over the lags and
    /// averaged over contexts.
    pub a: f32,
    /// Lag coherence ρ_τ: how much of that survives adding the lags' responses as
    /// arrows rather than as lengths (1 if they all point the same way).
    pub rho_tau: f32,
    /// Context coherence ρ_c: how much of that survives averaging the contexts' sums
    /// as arrows rather than as lengths (1 if every context gives the same arrow).
    pub rho_c: f32,
}

impl Factors {
    /// The singular value σ = a ρ_τ ρ_c.
    pub fn sigma(&self) -> f32 {
        self.a * self.rho_tau * self.rho_c
    }
}

/// The three factors for one Fourier mode whose per-token multiplier is `alpha`, with
/// reveal probability `epsilon` and key bias `psi` (1 without a key), over lags 1 to
/// `window` (None: the long-window limit), for the lens tuned to frequency `omega`:
/// lag τ's response is weighted by e^{−iωτ} before the lags are added (0: the plain lens).
pub fn factors(alpha: C32, epsilon: f32, psi: f32, window: Option<usize>, omega: f32) -> Factors {
    let keep = 1.0 - epsilon;
    let rho = alpha.abs();
    let beta = alpha * C32::polar(1.0, -omega);
    let limit = window.unwrap_or(usize::MAX);
    // S_k: the response gathered when the next reveal comes k tokens on, the sum of
    // beta^tau over tau = 1..k-1. Walk k upwards, carrying keep^(k-1) as `weight`.
    let (mut s, mut power, mut weight, mut reach) = (C32::ZERO, beta, 1.0f32, 1.0f32);
    let (mut path, mut mean_length, mut mean) = (0.0f32, 0.0f32, C32::ZERO);
    let mut k = 1;
    loop {
        if k > limit || weight < 1e-10 || power.abs() < 1e-10 || k > 20_000 {
            // no reveal inside the window (or none before the response has died away):
            // the sum is whatever has been gathered
            mean_length += weight * s.abs();
            break;
        }
        mean_length += weight * epsilon * s.abs();
        // lag k is reached without a reveal with probability keep^k
        weight *= keep;
        reach *= keep * rho;
        path += reach;
        mean = mean + power.scale(weight);
        s = s + power;
        power = power * beta;
        k += 1;
    }
    Factors {
        a: rho * path,
        rho_tau: if path > 0.0 { mean_length / path } else { 0.0 },
        rho_c: if mean_length > 0.0 { psi.abs() * mean.abs() / mean_length } else { 0.0 },
    }
}

/// The singular value in the long-window limit, in closed form:
/// σ = |ψ| |α| r / |1 − r e^{i(θ − ω)}| with r = |α| (1 − ε).
pub fn lens_weight(alpha: C32, epsilon: f32, psi: f32, omega: f32) -> f32 {
    let r = alpha.abs() * (1.0 - epsilon);
    psi.abs() * alpha.abs() * r / (C32::ONE - C32::polar(r, alpha.arg() - omega)).abs()
}

/// Fourier mode `j` of a function on a ring: the sum of f(x) e^{2πi j x / n}.
pub fn fourier(f: &[f32], j: usize) -> C32 {
    // (the wave is built by turning a unit arrow one site at a time: one sine and
    // cosine for the whole ring; rings are short, so nothing accumulates)
    let step = C32::polar(1.0, TAU * (j % f.len()) as f32 / f.len() as f32);
    let (mut wave, mut sum) = (C32::ONE, C32::ZERO);
    for &v in f {
        sum = sum + wave.scale(v);
        wave = wave * step;
    }
    sum
}

/// One token's worth of the dynamics applied to a distribution over the ring (or to
/// any function on it: the map is linear).
pub fn propagate(l: &Latent, q: f32, from: &[f32], out: &mut [f32]) {
    let n = l.n;
    let mass: f32 = from.iter().sum();
    let floor = (1.0 - l.lambda) * mass / n as f32;
    let (stay, slip) = (l.lambda * (1.0 - 2.0 * q), l.lambda * q);
    // site x is reached from x − v, or by a slip from the sites either side of that
    // (indices are stepped rather than taken modulo n: this loop runs millions of times)
    let mut src = (n - l.v % n) % n;
    for o in out[..n].iter_mut() {
        let before = if src == 0 { n - 1 } else { src - 1 };
        let after = if src + 1 == n { 0 } else { src + 1 };
        *o = floor + stay * from[src] + slip * (from[before] + from[after]);
        src = after;
    }
}

/// One step of the filter for one latent: the forecast of the next state, from the
/// forecast so far and what the token showed. `evidence` is None for a null token;
/// otherwise it is a distribution over the sites as shown (all on one site for a real
/// reveal), and `minus` says whether the previous token's key was −1.
pub fn filter_step(l: &Latent, q: f32, forecast: &[f32], evidence: Option<&[f32]>, minus: bool, out: &mut [f32]) {
    match evidence {
        None => propagate(l, q, forecast, out),
        Some(shown) => {
            // a reveal replaces all that was believed; undo the key's shift first
            let (n, shift) = (l.n, l.shift(minus));
            let mut state = [0.0f32; MAX_SITES];
            for (x, s) in state[..n].iter_mut().enumerate() {
                *s = shown[(x + shift) % n];
            }
            propagate(l, q, &state[..n], out);
        }
    }
}

/// The hidden states, and the tokens they give rise to.
#[derive(Clone, Debug)]
pub struct Chain {
    /// Where each latent is now: the state the next token will show or hide.
    pub x: Vec<usize>,
    /// For each latent, whether the key shown by the last token was −1.
    pub minus: Vec<bool>,
}

impl Chain {
    pub fn new(p: &Process, rng: &mut Rng) -> Self {
        Chain { x: p.latents.iter().map(|l| rng.below(l.n)).collect(), minus: vec![false; p.latents.len()] }
    }

    /// Emit the next token, then move every latent on. Returns the token and whether
    /// it showed the latents.
    pub fn emit(&mut self, p: &Process, rng: &mut Rng) -> (u32, bool) {
        let revealed = rng.bool(p.epsilon);
        let mut shown = [0usize; MAX_LATENTS];
        for (i, l) in p.latents.iter().enumerate() {
            shown[i] = (self.x[i] + l.shift(self.minus[i])) % l.n;
        }
        for (l, m) in p.latents.iter().zip(&mut self.minus) {
            if let Some(psi) = l.psi {
                *m = !rng.bool(0.5 * (1.0 + psi));
            }
        }
        let token = p.encode(revealed.then_some(&shown[..p.latents.len()]), &self.minus);
        for (l, x) in p.latents.iter().zip(&mut self.x) {
            *x = if rng.bool(l.lambda) {
                let u = rng.f32();
                let slip = if u < p.q {
                    1
                } else if u < 2.0 * p.q {
                    l.n - 1
                } else {
                    0
                };
                (*x + l.v + slip) % l.n
            } else {
                rng.below(l.n)
            };
        }
        (token, revealed)
    }
}

/// The exact Bayes filter: for each latent, the probability of every site for the
/// state the next token will show or hide, given all tokens so far.
#[derive(Clone, Debug)]
pub struct Filter {
    forecast: Vec<Vec<f32>>,
    minus: Vec<bool>,
}

impl Filter {
    /// Before any token: every site equally likely.
    pub fn new(p: &Process) -> Self {
        Filter { forecast: p.latents.iter().map(|l| vec![1.0 / l.n as f32; l.n]).collect(), minus: vec![false; p.latents.len()] }
    }

    pub fn observe(&mut self, p: &Process, token: u32) {
        let mut shown = [0usize; MAX_LATENTS];
        let revealed = p.shown(token, &mut shown);
        let mut next = [0.0f32; MAX_SITES];
        for (i, l) in p.latents.iter().enumerate() {
            let mut seen = [0.0f32; MAX_SITES];
            seen[shown[i]] = 1.0;
            filter_step(l, p.q, &self.forecast[i], revealed.then_some(&seen[..l.n]), self.minus[i], &mut next);
            self.forecast[i].copy_from_slice(&next[..l.n]);
            self.minus[i] = p.key_minus(token, i);
        }
    }

    /// The forecast for latent `i`.
    pub fn forecast(&self, i: usize) -> &[f32] {
        &self.forecast[i]
    }

    /// Fourier mode `j` of the forecast for latent `i`: the paper's belief state m.
    pub fn mode(&self, i: usize, j: usize) -> C32 {
        fourier(&self.forecast[i], j)
    }

    /// The probability the filter gives to `token` being the next one.
    pub fn chance_of(&self, p: &Process, token: u32) -> f32 {
        // (a latent shown at site y is at y less the shift the last key calls for)
        p.chance(token, |i, y| {
            let l = &p.latents[i];
            self.forecast[i][(y + l.n - l.shift(self.minus[i])) % l.n]
        })
    }

    /// The probability of each possible next token: the best prediction there is.
    /// (So far for the tests, and for a model yet to be trained.)
    #[cfg(test)]
    pub fn predict(&self, p: &Process, out: &mut Vec<f32>) {
        out.clear();
        out.extend((0..p.vocab() as u32).map(|token| self.chance_of(p, token)));
    }
}

/// A nudge to the evidence at one revealed token, carried along by the filter: the
/// derivative of the forecast with respect to Fourier mode `j` of what that token
/// showed. (The filter is linear in its evidence, so this is its own code applied to
/// the nudge: forward-mode differentiation.)
#[derive(Clone, Debug)]
pub struct Nudge {
    j: usize,
    n: usize,
    re: [f32; MAX_SITES],
    im: [f32; MAX_SITES],
}

impl Nudge {
    /// The nudge as the forecast first feels it, at the revealed token itself (lag 0).
    pub fn start(l: &Latent, q: f32, j: usize, minus: bool) -> Self {
        // a change to the evidence whose mode j is one and whose other modes are zero
        let n = l.n;
        let (mut er, mut ei) = ([0.0f32; MAX_SITES], [0.0f32; MAX_SITES]);
        let step = C32::polar(1.0, -TAU * (j % n) as f32 / n as f32);
        let mut wave = C32::new(1.0 / n as f32, 0.0);
        for y in 0..n {
            (er[y], ei[y]) = (wave.re, wave.im);
            wave = wave * step;
        }
        let mut nudge = Nudge { j, n, re: [0.0; MAX_SITES], im: [0.0; MAX_SITES] };
        filter_step(l, q, &[], Some(&er[..n]), minus, &mut nudge.re);
        filter_step(l, q, &[], Some(&ei[..n]), minus, &mut nudge.im);
        nudge
    }

    /// One more token without a reveal.
    pub fn advance(&mut self, l: &Latent, q: f32) {
        let (re, im) = (self.re, self.im);
        filter_step(l, q, &re[..self.n], None, false, &mut self.re);
        filter_step(l, q, &im[..self.n], None, false, &mut self.im);
    }

    /// The Jacobian at this lag: the change in mode `j` of the forecast per unit nudge.
    pub fn response(&self) -> C32 {
        fourier(&self.re[..self.n], self.j) + C32::new(0.0, 1.0) * fourier(&self.im[..self.n], self.j)
    }
}

/// A running estimate of the three factors, straight from their definitions: over
/// the contexts seen so far, the average path length Σ|K|, the average length of the
/// sum |ΣK|, and the average of the sum itself.
#[derive(Clone, Copy, Debug, Default)]
pub struct Measured {
    pub contexts: u64,
    path: f64,
    length: f64,
    sum: [f64; 2],
}

impl Measured {
    /// One context: the lengths of its responses added up, and their sum as arrows.
    pub fn add(&mut self, path: f32, sum: C32) {
        self.contexts += 1;
        self.path += path as f64;
        self.length += sum.abs() as f64;
        self.sum[0] += sum.re as f64;
        self.sum[1] += sum.im as f64;
    }

    pub fn factors(&self) -> Factors {
        if self.contexts == 0 {
            return Factors::default();
        }
        let n = self.contexts as f64;
        let (path, length, mean) = (self.path / n, self.length / n, self.sum[0].hypot(self.sum[1]) / n);
        // (a response that is rounding error, 1e-15 say, has no coherence to speak of)
        Factors {
            a: path as f32,
            rho_tau: if path > 1e-9 { (length / path) as f32 } else { 0.0 },
            rho_c: if length > 1e-9 { (mean / length) as f32 } else { 0.0 },
        }
    }
}

/// Measures the lens on a stream of tokens, for one mode of one latent: at every
/// reveal it nudges the evidence and follows the filter's response until the next
/// reveal, which closes that context.
#[derive(Clone, Debug, Default)]
pub struct Probe {
    nudge: Option<Nudge>,
    lag: usize,
    path: f32,
    sum: C32,
}

impl Probe {
    /// Feed the next token: whether it was a reveal, and whether the token before it
    /// showed a key of −1. Returns the response at this token to the nudge in flight,
    /// if there is one (lag 1 onwards). Closed contexts are added to `into`. The lag
    /// sum stops at `window` and weights lag τ by e^{−iωτ}.
    #[allow(clippy::too_many_arguments)]
    pub fn token(&mut self, l: &Latent, q: f32, j: usize, revealed: bool, minus: bool, window: usize, omega: f32, into: &mut Measured) -> Option<C32> {
        if revealed {
            if self.nudge.is_some() {
                into.add(self.path, self.sum);
            }
            *self = Probe { nudge: Some(Nudge::start(l, q, j, minus)), lag: 0, path: 0.0, sum: C32::ZERO };
            return None;
        }
        let nudge = self.nudge.as_mut()?;
        nudge.advance(l, q);
        self.lag += 1;
        let k = nudge.response();
        if self.lag <= window {
            self.path += k.abs();
            self.sum = self.sum + if omega == 0.0 { k } else { k * C32::polar(1.0, -omega * self.lag as f32) };
        }
        Some(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(n: usize, lambda: f32, v: usize) -> Latent {
        Latent { n, lambda, v, psi: None }
    }

    /// The paper's settings: rings of 8, q = 1/4, reveals one token in five.
    fn paper(latents: Vec<Latent>) -> Process {
        Process { latents, q: 0.25, epsilon: 0.2 }
    }

    fn close(a: f32, b: f32, tolerance: f32) -> bool {
        (a - b).abs() <= tolerance
    }

    #[test]
    fn tokens_round_trip() {
        let three = paper(vec![ring(8, 0.9, 0), ring(8, 0.9, 2), ring(8, 0.9, 4)]);
        assert_eq!(three.vocab(), 513);
        assert_eq!(paper(vec![ring(8, 0.9, 0), ring(8, 0.9, 2)]).vocab(), 65);
        let keyed = Process { latents: vec![Latent { n: 2, lambda: 0.95, v: 0, psi: Some(0.5) }; 3], q: 0.0, epsilon: 0.2 };
        assert_eq!(keyed.vocab(), 9 * 8);
        let mixed = Process { latents: vec![ring(8, 0.9, 1), Latent { n: 4, lambda: 0.9, v: 1, psi: Some(0.2) }, ring(2, 0.9, 0)], q: 0.1, epsilon: 0.3 };
        let mut seen = vec![false; mixed.vocab()];
        for sites in [None, Some([7usize, 3, 1]), Some([0, 0, 0]), Some([5, 2, 0])] {
            for minus in [[false, false, false], [false, true, false]] {
                let token = mixed.encode(sites.as_ref().map(|s| &s[..]), &minus);
                assert!(!std::mem::replace(&mut seen[token as usize], true), "two meanings for token {token}");
                let mut out = [9usize; 3];
                assert_eq!(mixed.shown(token, &mut out), sites.is_some());
                if let Some(s) = sites {
                    assert_eq!(out, s);
                }
                assert_eq!([0, 1, 2].map(|i| mixed.key_minus(token, i)), minus);
            }
        }
    }

    /// The lemma everything rests on: a step of the dynamics multiplies Fourier mode j
    /// of a distribution by α_j.
    #[test]
    fn a_step_multiplies_each_mode_by_alpha() {
        for (n, v, q, lambda) in
            [(8usize, 0usize, 0.25f32, 0.9f32), (8, 2, 0.25, 0.8), (8, 4, 0.1, 0.95), (2, 0, 0.0, 0.95), (2, 1, 0.2, 0.7), (6, 5, 0.3, 0.6)]
        {
            let p = Process { latents: vec![ring(n, lambda, v)], q, epsilon: 0.2 };
            for x in 0..n {
                let mut at = vec![0.0; n];
                at[x] = 1.0;
                let mut next = vec![0.0; n];
                propagate(&p.latents[0], q, &at, &mut next);
                assert!(close(next.iter().sum::<f32>(), 1.0, 1e-5));
                for j in 1..n {
                    let (got, want) = (fourier(&next, j), p.alpha(0, j) * fourier(&at, j));
                    assert!((got - want).abs() < 1e-5, "n={n} v={v} mode {j}: {got:?} vs {want:?}");
                }
            }
        }
    }

    /// The simulated chain moves as `propagate` says it does.
    #[test]
    fn the_chain_follows_the_same_dynamics() {
        let p = Process { latents: vec![ring(8, 0.8, 3)], q: 0.2, epsilon: 0.2 };
        let mut rng = Rng::new(11);
        let mut chain = Chain::new(&p, &mut rng);
        let mut counts = [0.0f32; 8];
        let total = 200_000;
        for _ in 0..total {
            let before = chain.x[0];
            chain.emit(&p, &mut rng);
            counts[(chain.x[0] + 8 - before) % 8] += 1.0 / total as f32;
        }
        let mut at = vec![0.0; 8];
        at[0] = 1.0;
        let mut want = vec![0.0; 8];
        propagate(&p.latents[0], p.q, &at, &mut want);
        for d in 0..8 {
            assert!(close(counts[d], want[d], 0.005), "a move of {d}: {} against {}", counts[d], want[d]);
        }
    }

    #[test]
    fn magnitudes_match_the_paper() {
        // "giving magnitude parameters of 0.82, 1.23, 1.50 for λ = 0.8, 0.9, 0.95"
        for (lambda, want) in [(0.8, 0.82), (0.9, 1.23), (0.95, 1.50)] {
            let p = paper(vec![ring(8, lambda, 0)]);
            for window in [None, Some(128)] {
                let f = p.lens(0, 1, window, 0.0);
                assert!(close(f.a, want, 0.006), "λ = {lambda}: a = {}", f.a);
                assert!(close(f.rho_tau, 1.0, 1e-4) && close(f.rho_c, 1.0, 1e-4), "{f:?}");
            }
            // and the closed form a = |α| r / (1 − r)
            let alpha = p.alpha(0, 1).abs();
            let r = alpha * 0.8;
            assert!(close(p.lens(0, 1, None, 0.0).a, alpha * r / (1.0 - r), 1e-4));
        }
    }

    #[test]
    fn lag_coherences_match_the_paper() {
        // "ρ_τ = 1, 0.35, 0.24 for θ = 0, π/2, π ... For π/2, the value is slightly less, at ρ_c = 0.945"
        let p = paper(vec![ring(8, 0.9, 0), ring(8, 0.9, 2), ring(8, 0.9, 4)]);
        let f: Vec<Factors> = (0..3).map(|i| p.lens(i, 1, None, 0.0)).collect();
        assert!(close(p.latents[1].theta(), PI / 2.0, 1e-6) && close(p.latents[2].theta(), PI, 1e-6));
        for (got, want) in f.iter().zip([1.0, 0.35, 0.24]) {
            assert!(close(got.rho_tau, want, 0.006), "{got:?}");
        }
        for (got, want) in f.iter().zip([1.0, 0.945, 1.0]) {
            assert!(close(got.rho_c, want, 0.001), "{got:?}");
        }
        // the magnitude is the same for all three: only the turn differs
        assert!(close(f[0].a, f[1].a, 1e-5) && close(f[0].a, f[2].a, 1e-5));
        // an alternating mode: ρ_τ = (1 − r)/(1 + r)
        let r = p.alpha(2, 1).abs() * 0.8;
        assert!(close(f[2].rho_tau, (1.0 - r) / (1.0 + r), 1e-4));
    }

    #[test]
    fn context_coherence_is_the_key_bias() {
        let keyed = |psi| Latent { n: 2, lambda: 0.95, v: 0, psi: Some(psi) };
        let p = Process { latents: vec![keyed(0.0), keyed(0.75), keyed(1.0)], q: 0.0, epsilon: 0.2 };
        let f: Vec<Factors> = (0..3).map(|i| p.lens(i, 1, None, 0.0)).collect();
        for (got, psi) in f.iter().zip([0.0, 0.75, 1.0]) {
            assert!(close(got.rho_c, psi, 1e-5) && close(got.rho_tau, 1.0, 1e-4) && close(got.a, f[0].a, 1e-5), "{got:?}");
        }
    }

    #[test]
    fn the_coherences_multiply_to_the_closed_form() {
        for v in 0..8 {
            for (lambda, epsilon, q) in [(0.9f32, 0.2f32, 0.25f32), (0.6, 0.5, 0.1), (0.97, 0.1, 0.0)] {
                let p = Process { latents: vec![ring(8, lambda, v)], q, epsilon };
                let (alpha, f) = (p.alpha(0, 1), p.lens(0, 1, None, 0.0));
                let r = alpha.abs() * (1.0 - epsilon);
                let turned = (C32::ONE - C32::polar(r, alpha.arg())).abs();
                assert!(close(f.rho_tau * f.rho_c, (1.0 - r) / turned, 2e-4), "v={v}: {f:?}");
                assert!(close(f.sigma(), alpha.abs() * r / turned, 2e-4));
                assert!(close(f.sigma(), lens_weight(alpha, epsilon, 1.0, 0.0), 2e-4));
                assert!(f.rho_tau <= 1.0 + 1e-5 && f.rho_c <= 1.0 + 1e-5);
            }
        }
    }

    #[test]
    fn a_tuned_lens_peaks_at_the_latents_own_turn() {
        let p = paper(vec![ring(8, 0.9, 0), ring(8, 0.9, 2), ring(8, 0.9, 4), ring(8, 0.9, 6)]);
        for i in 0..4 {
            let turn = p.latents[i].theta().abs();
            let mut best = (0.0f32, 0.0f32);
            for step in 0..=64 {
                let omega = PI * step as f32 / 64.0;
                let (long, finite) = (p.lens(i, 1, None, omega).sigma(), p.lens(i, 1, Some(128), omega).sigma());
                // the derived closed form, and the finite window agreeing with the limit
                assert!(close(long, lens_weight(C32::polar(p.alpha(i, 1).abs(), turn), 0.2, 1.0, omega), 2e-4));
                assert!(close(long, finite, 1e-4));
                if long > best.0 {
                    best = (long, omega);
                }
            }
            assert!(close(best.1, turn, 1e-3), "ring {i} peaks at {} rather than {turn}", best.1);
            // tuned to the turn, the latent is as coherent as one that stands still
            let f = p.lens(i, 1, None, turn);
            assert!(close(f.rho_tau, 1.0, 1e-4) && close(f.rho_c, 1.0, 1e-4));
        }
    }

    /// Run the process, nudge the real filter at every reveal, and compare what comes
    /// out with the closed forms.
    fn measure(p: &Process, tokens: usize, window: usize, omega: f32, seed: u64) -> Vec<Factors> {
        let mut rng = Rng::new(seed);
        let mut chain = Chain::new(p, &mut rng);
        let mut probes = vec![Probe::default(); p.latents.len()];
        let mut measured = vec![Measured::default(); p.latents.len()];
        for _ in 0..tokens {
            let before = chain.minus.clone();
            let (_, revealed) = chain.emit(p, &mut rng);
            for (i, l) in p.latents.iter().enumerate() {
                // (the mirror-image mode, if that is the one the tuned lens follows)
                let tuned = p.tuned(i, 1, omega);
                probes[i].token(l, p.q, 1, revealed, before[i], window, tuned, &mut measured[i]);
            }
        }
        measured.iter().map(Measured::factors).collect()
    }

    #[test]
    fn measured_factors_agree_with_the_closed_forms() {
        let keyed = |psi| Latent { n: 2, lambda: 0.95, v: 0, psi: Some(psi) };
        let cases = [
            (paper(vec![ring(8, 0.8, 0), ring(8, 0.9, 0), ring(8, 0.95, 0)]), 128, 0.0),
            (paper(vec![ring(8, 0.9, 4), ring(8, 0.9, 2), ring(8, 0.9, 0)]), 128, 0.0),
            (paper(vec![ring(8, 0.9, 4), ring(8, 0.9, 6), ring(8, 0.9, 1)]), 128, 1.3),
            (paper(vec![ring(8, 0.9, 3), ring(8, 0.95, 1), ring(8, 0.7, 5)]), 6, 0.0),
            (Process { latents: vec![keyed(0.0), keyed(0.75), keyed(1.0)], q: 0.0, epsilon: 0.2 }, 128, 0.0),
            (Process { latents: vec![Latent { n: 8, lambda: 0.9, v: 1, psi: Some(0.5) }, ring(4, 0.85, 1)], q: 0.1, epsilon: 0.35 }, 20, 0.4),
            // so much slipping on a ring of two that a still dial's forecast flips sign every token
            (Process { latents: vec![ring(2, 0.9, 0), ring(2, 0.9, 1)], q: 0.4, epsilon: 0.2 }, 128, 0.0),
        ];
        for (case, (p, window, omega)) in cases.iter().enumerate() {
            let got = measure(p, 600_000, *window, *omega, 7 + case as u64);
            for (i, m) in got.iter().enumerate() {
                let want = p.lens(i, 1, Some(*window), *omega);
                assert!(close(m.a, want.a, 0.015 * want.a.max(0.2)), "case {case} ring {i}: a {} against {}", m.a, want.a);
                assert!(close(m.rho_tau, want.rho_tau, 0.01), "case {case} ring {i}: lag coherence {} against {}", m.rho_tau, want.rho_tau);
                assert!(close(m.rho_c, want.rho_c, 0.02), "case {case} ring {i}: context coherence {} against {}", m.rho_c, want.rho_c);
            }
        }
    }

    /// The Jacobian of the filter itself, by finite differences on its evidence: it is
    /// α^(τ+1) until the next reveal and zero from then on (times the key).
    #[test]
    fn the_filters_jacobian_is_alpha_to_the_lag() {
        for (l, q, minus) in [
            (ring(8, 0.9, 2), 0.25, false),
            (ring(8, 0.8, 5), 0.1, false),
            (Latent { n: 2, lambda: 0.95, v: 0, psi: Some(0.5) }, 0.0, true),
            (Latent { n: 6, lambda: 0.7, v: 1, psi: Some(0.5) }, 0.2, true),
        ] {
            let p = Process { latents: vec![l], q, epsilon: 0.2 };
            let n = l.n;
            for j in 1..=n / 2 {
                // what the token shows, made soft so that it can be nudged along mode j
                let run = |nudge: f32, gap: usize, lags: usize| -> Vec<C32> {
                    let mut shown = vec![0.0f32; n];
                    shown[3 % n] = 1.0;
                    for (y, s) in shown.iter_mut().enumerate() {
                        *s += nudge * (TAU * ((j * y) % n) as f32 / n as f32).cos();
                    }
                    let mut forecast = vec![1.0 / n as f32; n];
                    let mut next = vec![0.0; n];
                    let mut out = Vec::new();
                    for step in 0..=lags {
                        let reveal_again = [0.0f32, 1.0].repeat(n / 2);
                        let evidence = if step == 0 {
                            Some(&shown[..])
                        } else if step == gap {
                            Some(&reveal_again[..])
                        } else {
                            None
                        };
                        filter_step(&l, q, &forecast, evidence, minus && step == 0, &mut next);
                        forecast.copy_from_slice(&next);
                        out.push(fourier(&forecast, j));
                    }
                    out
                };
                let (h, gap, lags) = (1e-2, 5, 8);
                let (up, down) = (run(h, gap, lags), run(-h, gap, lags));
                // the nudge's own mode j: the sum of cos · e^{...}, which is n/2 (n when 2j = n)
                let size = if 2 * j == n { n as f32 } else { n as f32 / 2.0 };
                let sign = if minus && j % 2 == 1 { -1.0 } else { 1.0 };
                let mut nudge = Nudge::start(&l, q, j, minus);
                for tau in 0..=lags {
                    let measured = (up[tau] - down[tau]).scale(1.0 / (2.0 * h * size));
                    let want = if tau < gap { p.alpha(0, j).powi(tau as i32 + 1).scale(sign) } else { C32::ZERO };
                    assert!((measured - want).abs() < 2e-4, "n={n} mode {j} lag {tau}: {measured:?} against {want:?}");
                    if tau < gap {
                        // and the nudge carried along by the filter says the same
                        assert!((nudge.response() - want).abs() < 1e-5, "nudge at lag {tau}");
                        nudge.advance(&l, q);
                    }
                }
            }
        }
    }

    #[test]
    fn the_order_switches_where_the_theory_says() {
        // two rings as in the paper's crossing experiment: one static whose persistence
        // is swept, one turning with λ = 0.95
        for v in 1..=4 {
            let mut p = paper(vec![ring(8, 0.5, 0), ring(8, 0.95, v)]);
            let other = p.lens(1, 1, None, 0.0).sigma();
            let star = crossing_static(p.diffusion_factor(0, 1), p.epsilon, other);
            let found = p.crossings(0, &p.sweep(0, 96, None, 0.0), other, None, 0.0);
            assert!(found.len() == 1 && close(found[0].0, star, 1e-4) && found[0].1, "{found:?} against {star}");
            p.latents[0].lambda = star;
            assert!(close(p.lens(0, 1, None, 0.0).sigma(), other, 1e-4));
            for (nudge, first) in [(-0.03, 1), (0.03, 0)] {
                p.latents[0].lambda = star + nudge;
                assert_eq!(p.spectrum(None, 0.0)[0].latent, first);
            }
            // a ring that turns faster is overtaken sooner
            if v > 1 {
                let slower = paper(vec![ring(8, 0.5, 0), ring(8, 0.95, v - 1)]).lens(1, 1, None, 0.0).sigma();
                assert!(other < slower);
            }
            if v == 2 {
                assert!(close(star, 0.649, 0.002), "λ₀* = {star}");
            }
        }
        // out of reach, or nothing there to be overtaken: no crossing
        let alone = paper(vec![ring(8, 0.5, 0)]);
        let sweep = alone.sweep(0, 96, None, 0.0);
        assert!(alone.crossings(0, &sweep, 100.0, None, 0.0).is_empty());
        assert!(alone.crossings(0, &sweep, 0.0, None, 0.0).is_empty());
    }

    /// With a short window and a lens tuned to what flips, a longer memory is not
    /// always a larger singular value: it rises, then falls, and passes a level twice.
    #[test]
    fn a_singular_value_that_rises_and_falls_crosses_twice() {
        let p = Process { latents: vec![ring(8, 0.8, 0), ring(8, 0.9, 0), ring(8, 0.95, 0)], q: 0.0, epsilon: 0.2 };
        let (window, omega) = (Some(2), PI);
        let sweep = p.sweep(2, 96, window, omega);
        let peak = sweep.iter().copied().fold(0.0, f32::max);
        assert!(peak > sweep[96] + 0.01, "it falls again: peak {peak}, at λ = 1 {}", sweep[96]);
        for other in [0, 1] {
            let level = p.lens(other, 1, window, omega).sigma();
            let found = p.crossings(2, &sweep, level, window, omega);
            assert_eq!(found.len(), 2, "against latent {other}: {found:?}");
            assert!(found[0].1 && !found[1].1, "up through the level, then down through it: {found:?}");
            for (lambda, _) in found {
                let mut at = p.clone();
                at.latents[2].lambda = lambda;
                assert!(close(at.lens(2, 1, window, omega).sigma(), level, 1e-4));
            }
        }
    }

    /// A response that is nothing but rounding error has no coherence to report.
    #[test]
    fn a_measured_nothing_is_nothing() {
        let mut m = Measured::default();
        for k in 0..100 {
            m.add(2e-15, C32::polar(2e-15, k as f32));
        }
        let f = m.factors();
        assert!(f.a < 1e-12 && f.rho_tau == 0.0 && f.rho_c == 0.0, "{f:?}");
        // (two sites and q = 1/4: slipping cancels the first mode outright, α = 0)
        let p = Process { latents: vec![ring(2, 0.9, 0)], q: 0.25, epsilon: 0.2 };
        assert_eq!(p.alpha(0, 1).abs(), 0.0);
        assert_eq!(p.lens(0, 1, Some(128), 0.0).sigma(), 0.0);
    }

    /// The textbook forward algorithm for one latent, written out with an explicit
    /// transition matrix, must give the forecasts the filter gives.
    #[test]
    fn the_filter_is_the_forward_algorithm() {
        let cases = [
            Process { latents: vec![ring(8, 0.9, 2)], q: 0.25, epsilon: 0.2 },
            Process { latents: vec![Latent { n: 6, lambda: 0.8, v: 1, psi: Some(0.3) }], q: 0.1, epsilon: 0.4 },
            Process { latents: vec![Latent { n: 2, lambda: 0.95, v: 0, psi: Some(0.75) }], q: 0.0, epsilon: 0.2 },
        ];
        for (case, p) in cases.iter().enumerate() {
            let l = p.latents[0];
            let n = l.n;
            // transition[x][y]: the chance of going from x to y
            let mut transition = vec![vec![(1.0 - l.lambda) / n as f32; n]; n];
            for (x, row) in transition.iter_mut().enumerate() {
                for (slip, chance) in [(0, 1.0 - 2.0 * p.q), (1, p.q), (n - 1, p.q)] {
                    row[(x + l.v + slip) % n] += l.lambda * chance;
                }
            }
            let mut rng = Rng::new(3 + case as u64);
            let mut chain = Chain::new(p, &mut rng);
            let mut filter = Filter::new(p);
            let mut prior = vec![1.0 / n as f32; n];
            let mut minus = false;
            for _ in 0..3000 {
                let (token, revealed) = chain.emit(p, &mut rng);
                let mut shown = [0usize; 1];
                assert_eq!(p.shown(token, &mut shown), revealed);
                // weigh each state by the chance that it would have produced this token
                let mut posterior: Vec<f32> = (0..n)
                    .map(|x| {
                        let would_show = (x + if minus && l.psi.is_some() { n / 2 } else { 0 }) % n;
                        prior[x]
                            * if !revealed {
                                1.0 - p.epsilon
                            } else if would_show == shown[0] {
                                p.epsilon
                            } else {
                                0.0
                            }
                    })
                    .collect();
                let total: f32 = posterior.iter().sum();
                posterior.iter_mut().for_each(|v| *v /= total);
                prior = (0..n).map(|y| (0..n).map(|x| posterior[x] * transition[x][y]).sum()).collect();
                minus = p.key_minus(token, 0);
                filter.observe(p, token);
                for y in 0..n {
                    assert!(close(filter.forecast(0)[y], prior[y], 1e-4), "case {case}: {:?} against {prior:?}", filter.forecast(0));
                }
                assert!((filter.mode(0, 1) - fourier(&prior, 1)).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn the_filter_predicts_tokens_better_than_no_memory() {
        let keyed = |psi| Latent { n: 2, lambda: 0.95, v: 0, psi: Some(psi) };
        for (case, p) in [
            paper(vec![ring(8, 0.8, 0), ring(8, 0.9, 2), ring(8, 0.95, 4)]),
            Process { latents: vec![keyed(0.0), keyed(0.75), keyed(1.0)], q: 0.0, epsilon: 0.2 },
        ]
        .iter()
        .enumerate()
        {
            let mut rng = Rng::new(5 + case as u64);
            let mut chain = Chain::new(p, &mut rng);
            let mut filter = Filter::new(p);
            let mut predicted = Vec::new();
            let flat: Vec<f32> = (0..p.vocab() as u32).map(|token| p.memoryless_chance(token)).collect();
            assert!(close(flat.iter().sum::<f32>(), 1.0, 1e-4));
            let (mut with_memory, mut without) = (0.0f64, 0.0f64);
            let total = 40_000;
            for _ in 0..total {
                filter.predict(p, &mut predicted);
                assert!(close(predicted.iter().sum::<f32>(), 1.0, 1e-4));
                let (token, _) = chain.emit(p, &mut rng);
                assert!(predicted[token as usize] > 0.0, "the filter ruled out what happened");
                with_memory -= (predicted[token as usize] as f64).ln() / total as f64;
                without -= (flat[token as usize] as f64).ln() / total as f64;
                filter.observe(p, token);
            }
            // (each reveal is worth a good fraction of a nat once the rings are remembered)
            assert!(with_memory < without - 0.1, "case {case}: {with_memory} nats per token with the filter, {without} without");
        }
    }

    /// The paper's numbers as computed here (run with --ignored --nocapture).
    #[test]
    #[ignore]
    fn numbers() {
        for lambda in [0.8, 0.9, 0.95] {
            println!("magnitude sweep λ = {lambda}: {:?}", paper(vec![ring(8, lambda, 0)]).lens(0, 1, Some(128), 0.0));
        }
        for v in [0, 2, 4] {
            let p = paper(vec![ring(8, 0.9, v)]);
            println!("lag sweep v = {v}: {:?}  σ = {:.4}", p.lens(0, 1, None, 0.0), p.lens(0, 1, None, 0.0).sigma());
        }
        for psi in [0.0, 0.75, 1.0] {
            let p = Process { latents: vec![Latent { n: 2, lambda: 0.95, v: 0, psi: Some(psi) }], q: 0.0, epsilon: 0.2 };
            println!("context sweep ψ = {psi}: {:?}", p.lens(0, 1, None, 0.0));
        }
        for v in 1..=4 {
            let p = paper(vec![ring(8, 0.5, 0), ring(8, 0.95, v)]);
            let other = p.lens(1, 1, None, 0.0).sigma();
            println!("crossing, θ₁ = {v}π/4: σ₁ = {other:.4}, λ₀* = {:.4}", crossing_static(p.diffusion_factor(0, 1), p.epsilon, other));
        }
    }

    #[test]
    fn the_spectrum_lists_every_mode_once() {
        let p = paper(vec![ring(8, 0.9, 4), ring(2, 0.9, 0)]);
        let s = p.spectrum(None, 0.0);
        assert_eq!(s.len(), 4 + 1);
        assert!(s.windows(2).all(|w| w[0].sigma >= w[1].sigma));
        // under the plain lens a mode and its mirror image weigh the same ...
        for (i, l) in p.latents.iter().enumerate() {
            for j in 1..l.n {
                assert!(close(p.lens(i, j, None, 0.0).sigma(), p.lens(i, l.n - j, None, 0.0).sigma(), 1e-5), "latent {i} mode {j}");
            }
        }
        // ... under a tuned one they do not, and the one listed is the larger
        let tuned = paper(vec![ring(8, 0.9, 2)]);
        let own = |j: usize| factors(tuned.alpha(0, j), 0.2, 1.0, None, PI / 2.0).sigma();
        let listed = tuned.spectrum(None, PI / 2.0).iter().find(|d| d.mode == 1).map(|d| d.sigma);
        assert!(close(own(1), 1.225, 1e-3) && close(own(7), 0.292, 1e-3), "{} {}", own(1), own(7));
        assert_eq!(listed, Some(own(1)));
        // with little slipping, a ring that flips every token is found by its second
        // mode (two waves round the ring stand still) before its first
        let fast = Process { latents: vec![ring(8, 0.9, 4)], q: 0.05, epsilon: 0.2 };
        assert_eq!(fast.spectrum(None, 0.0)[0].mode, 2);
        // with the paper's diffusion the first mode still leads
        assert_eq!(paper(vec![ring(8, 0.9, 4)]).spectrum(None, 0.0)[0].mode, 1);
    }
}

//! Training as a self-contained job: a network, the data it learns from and the
//! optimiser's settings, advanced a few steps at a time. The same code runs in the
//! frame (when there is nowhere else to run it), on a background thread, or in a Web
//! Worker. This file also defines the messages that drive it from the foreground and
//! the service that answers them.

use super::arith::{self, Task};
use super::*;
use crate::background::{Next, Service};
use crate::clock::{self, Budget};

/// This service's address: its position in `scenes::services()`.
pub const SERVICE: u8 = 0;
/// A rate at or above this many steps per second means "no limit".
pub const UNLIMITED: f32 = 1e5;
/// A job nobody has asked after for this long (seconds) rests.
const PATIENCE: f64 = 0.4;

/// What a network is being trained to do.
#[derive(Clone)]
pub enum Problem {
    /// Sort points in the plane into two classes (one logit).
    Plane(Data),
    /// Name the answer for a pair of numbers (one logit per answer).
    Table(Task),
}

impl Problem {
    /// Does a network of these layer widths fit the problem?
    fn fits(&self, sizes: &[usize]) -> bool {
        let (inputs, outputs) = match self {
            Problem::Plane(_) => (2, 1),
            Problem::Table(task) => (2 * task.p, task.p),
        };
        sizes.len() >= 2 && sizes[0] == inputs && sizes[sizes.len() - 1] == outputs
    }
}

/// Everything about a run that can change while it runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    pub optim: Optim,
    pub lr: f32,
    pub momentum: f32,
    pub decay: f32,
    pub batch: usize,
    /// Steps per second to aim for ([`UNLIMITED`] or more: as many as possible).
    pub rate: f32,
    pub running: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { optim: Optim::Adam, lr: 0.02, momentum: 0.9, decay: 0.0, batch: 32, rate: 360.0, running: false }
    }
}

pub struct Session {
    pub net: Mlp,
    pub problem: Problem,
    pub settings: Settings,
    pub steps: u64,
    pub diverged: bool,
    seed: u64,
    rng: Rng,
    /// Steps asked for one at a time (the "step" button), done whether running or not.
    pending: u32,
    /// Steps the rate has allowed but that have not been taken yet.
    allowance: f64,
    xb: Vec<f32>,
    hot: Vec<u32>,
    yb: Vec<f32>,
}

impl Session {
    pub fn new(net: Mlp, problem: Problem, seed: u64) -> Self {
        Session {
            net,
            problem,
            settings: Settings::default(),
            steps: 0,
            diverged: false,
            seed,
            rng: Rng::new(seed),
            pending: 0,
            allowance: 0.0,
            xb: Vec::new(),
            hot: Vec::new(),
            yb: Vec::new(),
        }
    }

    /// Something to hold before the first real job is loaded.
    pub fn placeholder() -> Self {
        Session::new(Mlp::from_params(&[2, 1], Act::Tanh, &[0.0; 3]), Problem::Plane(Data { x: Vec::new(), y: Vec::new() }), 0)
    }

    pub fn request_steps(&mut self, n: u32) {
        self.pending = self.pending.saturating_add(n);
    }

    /// One training step.
    fn step(&mut self) {
        let s = self.settings;
        let loss = match &self.problem {
            Problem::Plane(data) => {
                let n = data.len();
                if n == 0 {
                    return;
                }
                if s.batch >= n {
                    // the full batch is every point once, not n points drawn at random
                    self.net.forward(&data.x, n);
                    self.net.backward(&data.y, Loss::Logistic)
                } else {
                    let b = s.batch.max(1);
                    self.xb.clear();
                    self.yb.clear();
                    for _ in 0..b {
                        let i = self.rng.below(n);
                        self.xb.extend_from_slice(&data.x[2 * i..2 * i + 2]);
                        self.yb.push(data.y[i]);
                    }
                    self.net.forward(&self.xb, b);
                    self.net.backward(&self.yb, Loss::Logistic)
                }
            }
            Problem::Table(task) => {
                if s.batch >= task.train.len() {
                    arith::full_batch(&mut self.net, task)
                } else {
                    let b = s.batch.max(1);
                    task.sample(b, &mut self.rng, &mut self.hot, &mut self.yb);
                    self.net.forward_hot(&self.hot, 2, b);
                    self.net.backward(&self.yb, Loss::Softmax)
                }
            }
        };
        if !loss.is_finite() {
            // the weights have run away; stop here rather than train on NaN
            self.diverged = true;
            return;
        }
        self.net.step(s.optim, s.lr, s.momentum, s.decay);
        self.steps += 1;
    }

    /// Train. `elapsed` seconds have passed since the last call, which decides how
    /// many steps the chosen rate allows; whatever the rate, stop after `seconds`.
    pub fn work(&mut self, elapsed: f64, seconds: f64) -> Next {
        let s = self.settings;
        if s.running && !self.diverged {
            // (in a reproducible run no time allowance ends the loop below, so even
            // "no limit" must come to a definite number of steps)
            self.allowance = if s.rate >= UNLIMITED && !clock::reproducible() {
                f64::MAX
            } else {
                // a slow moment may be made up for, but not more than a quarter second of it
                (self.allowance + s.rate as f64 * elapsed).min((s.rate as f64 * 0.25).max(1.0))
            };
        } else {
            self.allowance = 0.0;
        }
        let budget = Budget::new(seconds);
        loop {
            if self.diverged {
                self.pending = 0;
                break;
            }
            if self.pending > 0 {
                self.pending -= 1;
            } else if self.allowance >= 1.0 {
                self.allowance -= 1.0;
            } else {
                break;
            }
            self.step();
            if budget.spent() {
                break;
            }
        }
        if self.diverged || !(s.running || self.pending > 0) {
            Next::Idle
        } else if self.pending > 0 || self.allowance >= 1.0 {
            Next::Now
        } else {
            Next::Soon
        }
    }
}

/// How fast training goes on this machine, in this build: steps per second for the
/// playground's two default jobs and a few variations. The same function runs
/// natively (`eidos bench-training`) and in the WebAssembly build
/// (`tools/wasm-bench.mjs`), which is how the two are compared.
pub fn benchmark() -> String {
    let mut out = String::from("training speed, steps per second (one thread)\n");
    if !crate::math::nn::kernels_agree() {
        out += "  WARNING: the SIMD loops disagree with the plain ones\n";
    }
    let mut time = |label: &str, mut job: Session, settings: Settings| {
        // (not running: only the steps asked for are taken, twenty to warm up and
        // then fifty at a time for as long as the clock allows)
        job.settings = Settings { running: false, ..settings };
        job.request_steps(20);
        job.work(0.0, 10.0);
        let (start, before) = (clock::now(), job.steps);
        while clock::now() - start < 0.4 {
            job.request_steps(50);
            job.work(0.0, 10.0);
        }
        let rate = (job.steps - before) as f64 / (clock::now() - start);
        out += &format!("  {label:<52} {rate:>9.0}\n");
    };
    let plane = |act: Act| {
        let mut rng = Rng::new(1);
        let data = make_data(0, 260, 0.25, &mut rng);
        Session::new(Mlp::new(&[2, 8, 8, 1], act, &mut rng), Problem::Plane(data), 1)
    };
    let table = |p: usize, width: usize| {
        let task = Task::new(0, p, 0.8, 1);
        let net = Mlp::new(&task.sizes(1, width), Act::Relu, &mut Rng::new(1));
        Session::new(net, Problem::Table(task), 1)
    };
    let adam = Settings::default();
    let full = Settings { batch: usize::MAX, lr: 0.01, decay: 1.0, ..adam };
    time("points, 2 x 8 tanh, batch 32, Adam", plane(Act::Tanh), adam);
    time("points, 2 x 8 GELU, batch 32, Adam", plane(Act::Gelu), adam);
    time("points, 2 x 8 tanh, batch 260, Adam", plane(Act::Tanh), Settings { batch: 260, ..adam });
    time("points, 2 x 8 tanh, batch 32, Muon", plane(Act::Tanh), Settings { optim: Optim::Muon, ..adam });
    time("a + b mod 17, 128 ReLU, full batch, Adam (default)", table(17, 128), full);
    time("a + b mod 17, 128 ReLU, full batch, Muon", table(17, 128), Settings { optim: Optim::Muon, ..full });
    time("a + b mod 23, 128 ReLU, full batch, Adam", table(23, 128), full);
    out
}

// ----- messages --------------------------------------------------------------------

mod cmd {
    pub const LOAD: u8 = 0;
    pub const PROBLEM: u8 = 1;
    pub const SETTINGS: u8 = 2;
    pub const STEP: u8 = 3;
    pub const DROP: u8 = 4;
    pub const ALIVE: u8 = 5;
}

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32s(&mut self, v: &[f32]) {
        self.u32(v.len() as u32);
        self.0.reserve(v.len() * 4);
        for x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
    fn settings(&mut self, s: Settings) {
        self.u8(Optim::ALL.iter().position(|&o| o == s.optim).unwrap_or(0) as u8);
        self.f32(s.lr);
        self.f32(s.momentum);
        self.f32(s.decay);
        self.u32(s.batch.min(u32::MAX as usize) as u32);
        self.f32(s.rate);
        self.u8(s.running as u8);
    }
    fn problem(&mut self, p: &Problem) {
        match p {
            Problem::Plane(data) => {
                self.u8(0);
                self.f32s(&data.x);
                self.f32s(&data.y);
            }
            Problem::Table(task) => {
                self.u8(1);
                self.u32(task.kind as u32);
                self.u32(task.p as u32);
                self.f32(task.fraction);
                self.u64(task.split);
            }
        }
    }
}

struct Reader<'a>(&'a [u8]);

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> Option<[u8; N]> {
        let (head, rest) = self.0.split_first_chunk::<N>()?;
        self.0 = rest;
        Some(*head)
    }
    fn u8(&mut self) -> Option<u8> {
        self.take::<1>().map(|b| b[0])
    }
    fn u32(&mut self) -> Option<u32> {
        self.take().map(u32::from_le_bytes)
    }
    fn u64(&mut self) -> Option<u64> {
        self.take().map(u64::from_le_bytes)
    }
    fn f32(&mut self) -> Option<f32> {
        self.take().map(f32::from_le_bytes)
    }
    fn f32s(&mut self) -> Option<Vec<f32>> {
        let n = self.u32()? as usize;
        if self.0.len() < n.checked_mul(4)? {
            return None;
        }
        let (head, rest) = self.0.split_at(n * 4);
        self.0 = rest;
        Some(head.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
    fn settings(&mut self) -> Option<Settings> {
        Some(Settings {
            optim: *Optim::ALL.get(self.u8()? as usize)?,
            lr: self.f32()?,
            momentum: self.f32()?,
            decay: self.f32()?,
            batch: self.u32()? as usize,
            rate: self.f32()?,
            running: self.u8()? != 0,
        })
    }
    fn problem(&mut self) -> Option<Problem> {
        match self.u8()? {
            0 => {
                let (x, y) = (self.f32s()?, self.f32s()?);
                (x.len() == 2 * y.len()).then_some(Problem::Plane(Data { x, y }))
            }
            1 => {
                let (kind, p, fraction, split) = (self.u32()? as usize, self.u32()? as usize, self.f32()?, self.u64()?);
                (kind < arith::OPS.len() && (2..=64).contains(&p) && fraction.is_finite()).then(|| Problem::Table(Task::new(kind, p, fraction, split)))
            }
            _ => None,
        }
    }
}

fn header(command: u8, slot: u32) -> Writer {
    let mut w = Writer::default();
    w.u8(command);
    w.u32(slot);
    w
}

/// "Here is a job": the network, its problem, the settings to train it with, where
/// its step count stands and any steps asked for one at a time. `epoch` comes back
/// with every snapshot, so that weights from before a reset are recognised and
/// ignored. (The settings travel with the job so that its first steps are taken
/// with them, and not with whatever the job before it had.)
pub fn load(slot: u32, epoch: u32, session: &Session) -> Vec<u8> {
    let mut w = header(cmd::LOAD, slot);
    w.u32(epoch);
    w.u64(session.steps);
    w.u64(session.seed);
    w.u32(session.pending);
    w.settings(session.settings);
    w.u8(Act::ALL.iter().position(|&a| a == session.net.act).unwrap_or(0) as u8);
    w.u32(session.net.sizes.len() as u32);
    for &s in &session.net.sizes {
        w.u32(s as u32);
    }
    w.f32s(&session.net.params());
    w.problem(&session.problem);
    w.0
}

/// "The data changed" (the weights and the optimiser's memory stay).
pub fn problem(slot: u32, problem: &Problem) -> Vec<u8> {
    let mut w = header(cmd::PROBLEM, slot);
    w.problem(problem);
    w.0
}

pub fn settings(slot: u32, s: Settings) -> Vec<u8> {
    let mut w = header(cmd::SETTINGS, slot);
    w.settings(s);
    w.0
}

pub fn step(slot: u32, n: u32) -> Vec<u8> {
    let mut w = header(cmd::STEP, slot);
    w.u32(n);
    w.0
}

pub fn drop(slot: u32) -> Vec<u8> {
    header(cmd::DROP, slot).0
}

/// "Someone is still watching this job": keep it going, and send where it stands.
/// (Any message about a job says as much; this is the one sent in a frame that has
/// nothing else to say.)
pub fn alive(slot: u32) -> Vec<u8> {
    header(cmd::ALIVE, slot).0
}

/// What the background sends back: where a job stands.
pub struct Snapshot {
    pub slot: u32,
    pub epoch: u32,
    pub steps: u64,
    pub diverged: bool,
    pub params: Vec<f32>,
}

impl Snapshot {
    fn encode(&self) -> Vec<u8> {
        let mut w = Writer::default();
        w.u32(self.slot);
        w.u32(self.epoch);
        w.u64(self.steps);
        w.u8(self.diverged as u8);
        w.f32s(&self.params);
        w.0
    }

    /// Whose snapshot is this? (Cheap: reads only the first field.)
    pub fn slot_of(msg: &[u8]) -> Option<u32> {
        Reader(msg).u32()
    }

    pub fn decode(msg: &[u8]) -> Option<Snapshot> {
        let mut r = Reader(msg);
        Some(Snapshot { slot: r.u32()?, epoch: r.u32()?, steps: r.u64()?, diverged: r.u8()? != 0, params: r.f32s()? })
    }
}

// ----- the service -----------------------------------------------------------------

struct Slot {
    id: u32,
    epoch: u32,
    session: Session,
    heard: f64,
    worked: f64,
    sent: Option<(u64, bool)>,
    /// The foreground has spoken about this job since the last snapshot. It does in
    /// every frame it draws, and each time it is sent one snapshot, of the job as it
    /// stands then (once it has anything new): one in every frame, with the steps of
    /// one frame's time in it. (Sent every 8 ms instead, as they were, the snapshots
    /// fell into the frames unevenly: one in some frames and none in others at 144
    /// frames a second, one or two at 75, so that the picture stood still in some
    /// frames and took a double step in others.)
    asked: bool,
}

/// The background's trainer: any number of jobs, each advanced while its owner in
/// the foreground keeps asking after it.
#[derive(Default)]
pub struct Trainer {
    slots: Vec<Slot>,
}

impl Trainer {
    fn apply(&mut self, msg: &[u8]) -> Option<()> {
        let mut r = Reader(msg);
        let (command, id) = (r.u8()?, r.u32()?);
        let now = clock::now();
        if command == cmd::LOAD {
            let (epoch, steps, seed, pending, settings) = (r.u32()?, r.u64()?, r.u64()?, r.u32()?, r.settings()?);
            let act = *Act::ALL.get(r.u8()? as usize)?;
            let count = r.u32()? as usize;
            let sizes: Vec<usize> = (0..count.min(16)).map(|_| r.u32().map(|s| s as usize)).collect::<Option<_>>()?;
            let params = r.f32s()?;
            let problem = r.problem()?;
            let expected: usize = sizes.windows(2).map(|w| w[0] * w[1] + w[1]).sum();
            if count > 16 || sizes.iter().any(|&s| s == 0 || s > 4096) || !problem.fits(&sizes) || params.len() != expected {
                return None;
            }
            let mut session = Session::new(Mlp::from_params(&sizes, act, &params), problem, seed);
            (session.steps, session.pending, session.settings) = (steps, pending, settings);
            self.slots.retain(|s| s.id != id);
            self.slots.push(Slot { id, epoch, session, heard: now, worked: now, sent: None, asked: true });
            return Some(());
        }
        let i = self.slots.iter().position(|s| s.id == id)?;
        if command == cmd::DROP {
            self.slots.remove(i);
            return Some(());
        }
        let slot = &mut self.slots[i];
        (slot.heard, slot.asked) = (now, true);
        match command {
            cmd::PROBLEM => {
                let problem = r.problem()?;
                if problem.fits(&slot.session.net.sizes) {
                    slot.session.problem = problem;
                }
            }
            cmd::SETTINGS => slot.session.settings = r.settings()?,
            cmd::STEP => slot.session.request_steps(r.u32()?),
            _ => {}
        }
        Some(())
    }
}

impl Service for Trainer {
    fn handle(&mut self, msg: &[u8]) {
        // a message that does not parse is one we did not send; it changes nothing
        let _ = self.apply(msg);
    }

    fn work(&mut self, seconds: f64, send: &mut dyn FnMut(Vec<u8>)) -> Next {
        let mut next = Next::Idle;
        for slot in &mut self.slots {
            let now = clock::now();
            if now - slot.heard > PATIENCE {
                slot.worked = now;
                continue;
            }
            let elapsed = (now - slot.worked).clamp(0.0, 0.1);
            slot.worked = now;
            next = next.max(slot.session.work(elapsed, seconds));
            let state = (slot.session.steps, slot.session.diverged);
            if slot.asked && slot.sent != Some(state) {
                send(Snapshot { slot: slot.id, epoch: slot.epoch, steps: state.0, diverged: state.1, params: slot.session.net.params() }.encode());
                (slot.sent, slot.asked) = (Some(state), false);
            }
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane_job(seed: u64) -> Session {
        let mut rng = Rng::new(seed);
        let data = make_data(0, 120, 0.2, &mut rng);
        Session::new(Mlp::new(&[2, 6, 6, 1], Act::Tanh, &mut rng), Problem::Plane(data), 99)
    }

    /// A job sent through the messages and trained by the service arrives at exactly
    /// the weights it reaches when trained in place.
    #[test]
    fn the_service_trains_what_the_session_trains() {
        for table in [false, true] {
            let make = || {
                if table {
                    let task = Task::new(0, 7, 0.7, 3);
                    let net = Mlp::new(&task.sizes(1, 16), Act::Relu, &mut Rng::new(5));
                    Session::new(net, Problem::Table(task), 42)
                } else {
                    plane_job(4)
                }
            };
            let wanted = Settings { optim: Optim::Muon, lr: 0.02, momentum: 0.95, decay: 0.01, batch: 16, rate: 600.0, running: true };

            let mut here = make();
            here.settings = wanted;
            here.request_steps(3);
            for _ in 0..5 {
                here.work(0.01, 1.0);
            }
            assert_eq!(here.steps, 3 + 5 * 6);

            let mut service = Trainer::default();
            service.handle(&load(7, 2, &make()));
            service.handle(&settings(7, wanted));
            service.handle(&step(7, 3));
            let mut last = None;
            for _ in 0..5 {
                // the service measures elapsed time itself; drive its job directly
                service.slots[0].session.work(0.01, 1.0);
                service.slots[0].worked = clock::now();
                // (asked after, as the foreground does in every frame)
                service.handle(&alive(7));
                service.work(0.0, &mut |msg| last = Snapshot::decode(&msg));
            }
            let snap = last.expect("no snapshot");
            assert_eq!((snap.slot, snap.epoch, snap.steps, snap.diverged), (7, 2, here.steps, false));
            assert_eq!(snap.params, here.net.params(), "table = {table}");
        }
    }

    #[test]
    fn messages_that_make_no_sense_change_nothing() {
        let mut service = Trainer::default();
        let good = load(1, 1, &plane_job(2));
        for cut in 0..good.len() {
            service.handle(&good[..cut]);
        }
        assert!(service.slots.is_empty());
        // a network that does not fit its problem
        let mut wrong = plane_job(2);
        wrong.net = Mlp::new(&[3, 4, 1], Act::Tanh, &mut Rng::new(1));
        service.handle(&load(1, 1, &wrong));
        assert!(service.slots.is_empty());
        service.handle(&good);
        assert_eq!(service.slots.len(), 1);
        // commands for a job that does not exist, and a truncated settings message
        service.handle(&step(9, 5));
        service.handle(&settings(1, Settings::default())[..8]);
        assert_eq!(service.slots[0].session.settings, Settings::default());
        service.handle(&drop(1));
        assert!(service.slots.is_empty());
    }

    /// The weights go back once each time the foreground asks after the job (it does
    /// in every frame), with whatever steps have been taken since the last time: not
    /// on a clock of their own, which fell into the frames unevenly. Nothing goes back
    /// while nothing has changed.
    #[test]
    fn a_snapshot_answers_each_asking() {
        let mut service = Trainer::default();
        service.handle(&load(2, 1, &plane_job(5)));
        let mut sent = Vec::new();
        let turn = |service: &mut Trainer, steps: u32| {
            // (steps taken as the step button takes them: the rate's clock decides nothing)
            service.slots[0].session.request_steps(steps);
            service.slots[0].session.work(0.0, 1.0);
            let mut got = Vec::new();
            service.work(0.0, &mut |msg| got.extend(Snapshot::decode(&msg).map(|s| s.steps)));
            got
        };
        // the job as it arrives, at once
        sent.push(turn(&mut service, 0));
        // steps taken, and nobody has asked: nothing; then asked, all of them in one
        sent.push(turn(&mut service, 3));
        sent.push(turn(&mut service, 2));
        service.handle(&alive(2));
        sent.push(turn(&mut service, 1));
        // asked, and nothing new: nothing, until there is
        service.handle(&alive(2));
        sent.push(turn(&mut service, 0));
        sent.push(turn(&mut service, 4));
        assert_eq!(sent, [vec![0], vec![], vec![], vec![6], vec![], vec![10]]);
    }

    /// A job arrives with the settings it is to be trained with and the steps asked
    /// for one at a time, and its first steps use them, not the settings of the job
    /// it replaces (whose weight decay, say, used to apply until the next message).
    #[test]
    fn a_job_brings_its_settings() {
        let mut service = Trainer::default();
        let mut first = plane_job(1);
        first.settings = Settings { decay: 1.0, lr: 0.2, ..Settings::default() };
        service.handle(&load(4, 1, &first));
        let mut second = plane_job(2);
        second.settings = Settings { optim: Optim::Sgd, lr: 0.05, decay: 0.0, batch: 16, ..Settings::default() };
        second.request_steps(2);
        service.handle(&load(4, 2, &second));
        assert_eq!(service.slots.len(), 1);
        let job = &mut service.slots[0].session;
        assert_eq!((job.settings, job.pending), (second.settings, 2));
        job.work(0.0, 1.0);
        second.work(0.0, 1.0);
        assert_eq!((job.steps, job.net.params()), (2, second.net.params()));
    }

    /// "Full batch" means every point once in every step: no sampling noise, so the
    /// seed of the batches does not matter, and under plain gradient descent with a
    /// small step the loss on all the data only ever falls.
    #[test]
    fn a_full_batch_is_all_the_data() {
        let run = |seed: u64, batch: usize| {
            let mut rng = Rng::new(3);
            let data = make_data(0, 120, 0.2, &mut rng);
            let mut job = Session::new(Mlp::new(&[2, 6, 6, 1], Act::Tanh, &mut rng), Problem::Plane(data.clone()), seed);
            job.settings = Settings { optim: Optim::Sgd, lr: 0.05, batch, ..Settings::default() };
            let mut rises = 0;
            let mut last = f32::MAX;
            for _ in 0..200 {
                job.request_steps(1);
                job.work(0.0, f64::MAX);
                let loss = job.net.eval(&data.x, &data.y, data.len(), Loss::Logistic);
                rises += (loss > last) as usize;
                last = loss;
            }
            (job.net.params(), rises)
        };
        for batch in [120, 500, usize::MAX] {
            let (a, b) = (run(1, batch), run(2, batch));
            assert_eq!(a.0, b.0, "batch {batch}: the seed of the batches changed the weights");
            assert_eq!(a.1, 0, "batch {batch}: the loss rose on {} of 200 steps", a.1);
        }
        // one point short of everything is sampled, and so is noisy
        let (a, b) = (run(1, 119), run(2, 119));
        assert!(a.0 != b.0 && a.1 > 0);
    }

    /// The benchmark takes the steps it asks for and no more: it used to train for
    /// the whole ten seconds of its allowance before timing anything.
    #[test]
    fn the_benchmark_is_quick() {
        let start = clock::now();
        let report = benchmark();
        assert!(clock::now() - start < 8.0, "the benchmark took {:.1} s", clock::now() - start);
        assert_eq!(report.lines().count(), 8, "{report}");
    }

    /// The pacing: a rate gives that many steps per second, however the time is cut up.
    #[test]
    fn the_rate_is_kept() {
        let mut s = plane_job(3);
        s.settings = Settings { rate: 240.0, running: true, ..Settings::default() };
        for _ in 0..120 {
            s.work(1.0 / 120.0, 1.0);
        }
        assert!((239..=241).contains(&s.steps), "{} steps in one second at 240 per second", s.steps);
        s.settings.running = false;
        assert_eq!(s.work(1.0, 1.0), Next::Idle);
        assert!((239..=241).contains(&s.steps));
    }

    /// End to end over the real thread: load, run, and watch the steps arrive.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_background_thread_trains() {
        // (in a thread of its own, because the link belongs to the thread that made it)
        std::thread::spawn(|| {
            let link = crate::background::link().expect("no background");
            link.send(SERVICE, &load(3, 1, &plane_job(8)));
            link.send(SERVICE, &settings(3, Settings { rate: UNLIMITED, running: true, ..Settings::default() }));
            let mut steps = 0;
            let start = std::time::Instant::now();
            while steps < 500 && start.elapsed().as_secs_f64() < 5.0 {
                std::thread::sleep(std::time::Duration::from_millis(5));
                link.send(SERVICE, &alive(3));
                link.take(SERVICE, |m| Snapshot::slot_of(m) == Some(3), |m| steps = Snapshot::decode(m).map_or(steps, |s| s.steps));
            }
            link.send(SERVICE, &drop(3));
            assert!(steps >= 500, "only {steps} steps arrived");
        })
        .join()
        .unwrap();
    }
}

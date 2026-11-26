//! What the trainable scenes share: the network's shape, the optimiser's settings,
//! the job that does the training, and the history of the loss.
//!
//! The job ([`Session`]) runs in the background when there is one (a thread natively,
//! a Web Worker in the browser) and the frame only displays its progress. Otherwise
//! it runs in the frame within a time allowance. A Web Worker takes a moment to
//! start: training waits for it (see [`HOLD`]) rather than start in the frame and
//! move house mid-run.

use super::session::{self, Problem, Session, Settings, Snapshot, SERVICE, UNLIMITED};
use super::*;
use crate::background::{self, Link};
use crate::clock;
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};

/// When training has to share the frame: the share of each frame it may take, and
/// the most it may take in seconds. A share rather than a fixed time, so that a
/// 120 Hz display keeps its 120 frames a second. A step cannot be cut short, so a
/// step that takes longer than this is followed by as many frames without one as
/// it overran by: however large the batch, the network or the rate asked for,
/// training slows down and the display does not (only a single step longer than a
/// whole frame makes that one frame late).
const TRAIN_SHARE: f64 = 0.4;
const TRAIN_BUDGET_MAX: f64 = 0.006;

/// How long training waits for the background to be up before it starts in the
/// frame instead (seconds). A Web Worker usually takes a fraction of a second;
/// training in the frame meanwhile and then handing over would restart the
/// optimiser's memory and the order of the batches partway through the run.
const HOLD: f64 = 1.0;

/// Points kept in each history curve (see [`History`]: the whole run stays visible,
/// at a resolution that halves each time the curve fills up).
const HISTORY: usize = 600;

/// The top of the "steps per second" control, which means "no limit".
pub const RATE_MAX: usize = 100_000;

/// After asking the background for something while paused, keep drawing for this
/// many frames so that its answer is seen.
const AWAIT_FRAMES: u32 = 45;

static NEXT_SLOT: AtomicU32 = AtomicU32::new(0);

/// Number the next job as the first again. (For the scene check, which opens every
/// scene in one process and wants each to start as it does when opened first; only
/// to be called while no training scene is open.)
pub fn start_afresh() {
    NEXT_SLOT.store(0, Ordering::Relaxed);
}

pub struct Training {
    pub hidden: usize,
    pub width: usize,
    pub act: usize,
    pub optim: usize,
    pub momentum: f32,
    pub lr: f32,
    pub batch: usize,
    pub decay: f32,
    /// Steps per second asked for ([`RATE_MAX`]: as many as possible).
    pub rate: usize,
    pub running: bool,
    pub diverged: bool,
    pub steps: u64,
    /// Steps per second actually achieved.
    pub achieved: f32,
    /// Loss on the training data and on the held-out data, as training went.
    pub loss: [History; 2],
    pub accuracy: [History; 2],
    /// The job. While the background has it, this copy follows the background's.
    session: Session,
    link: Option<Rc<Link>>,
    remote: bool,
    /// This job's name in the background, and which network it is on (so that
    /// weights from before a reset are recognised and ignored).
    slot: u32,
    epoch: u32,
    sent: Option<Settings>,
    awaiting: u32,
    redraw: Redraw,
    /// The step count and the values of the last reading in the history curves.
    recorded_steps: u64,
    recorded: ([f32; 2], [f32; 2]),
    /// When this was made (the background is waited for until [`HOLD`] after it),
    /// and whether training is being held back for it now.
    born: f64,
    held: bool,
    /// Is the scene's clock running, and how fast against real time (the app's
    /// time speed)? The rate counts steps per second of the scene's time.
    live: bool,
    speed: f32,
    /// Seconds of training in the frame still to be made up for (see [`TRAIN_SHARE`]).
    overrun: f64,
    /// Scene time so far, and where it, the clock and the step count stood when
    /// the achieved rate was last measured.
    scene_time: f64,
    rate_scene: f64,
    rate_time: f64,
    rate_steps: u64,
    /// The display's frame interval: the shortest interval seen lately.
    frame_interval: f64,
    /// The weights and biases set aside by [`Training::with_edited`].
    kept: (Vec<Vec<f32>>, Vec<Vec<f32>>),
}

impl Training {
    pub fn new(hidden: usize, width: usize) -> Self {
        Training {
            hidden,
            width,
            act: 0,
            optim: 5,
            momentum: 0.9,
            lr: 0.02,
            batch: 32,
            decay: 0.0,
            rate: 360,
            running: true,
            diverged: false,
            steps: 0,
            achieved: 0.0,
            loss: [History::new(HISTORY), History::new(HISTORY)],
            accuracy: [History::new(HISTORY), History::new(HISTORY)],
            kept: (Vec::new(), Vec::new()),
            session: Session::placeholder(),
            link: background::link(),
            remote: false,
            slot: NEXT_SLOT.fetch_add(1, Ordering::Relaxed),
            epoch: 0,
            sent: None,
            awaiting: 0,
            redraw: Redraw::default(),
            recorded_steps: u64::MAX,
            recorded: ([0.0; 2], [0.0; 2]),
            born: clock::now(),
            held: false,
            live: false,
            speed: 1.0,
            overrun: 0.0,
            scene_time: 0.0,
            rate_scene: 0.0,
            rate_time: 0.0,
            rate_steps: 0,
            frame_interval: 1.0 / 60.0,
        }
    }

    pub fn act(&self) -> Act {
        Act::ALL[self.act]
    }

    /// The network as it stands (for drawing).
    pub fn net(&self) -> &Mlp {
        &self.session.net
    }

    /// The same, for running data through it (a forward pass leaves its activations
    /// in the network). Its weights are the job's: read them, change them for a moment
    /// with [`Training::with_edited`], replace the whole network with
    /// [`Training::load`], but do not edit them here.
    pub fn net_mut(&mut self) -> &mut Mlp {
        &mut self.session.net
    }

    /// Run `body` on the network as `edit` leaves it (a neuron silenced, a frequency
    /// taken out of a layer), then put every weight and bias back as it was. The
    /// activations of `body`'s last forward pass stay, for drawing. This is how the
    /// scenes look at an altered network while the job goes on training the real one.
    pub fn with_edited<R>(&mut self, edit: impl FnOnce(&mut Mlp), body: impl FnOnce(&mut Mlp) -> R) -> R {
        let net = &mut self.session.net;
        self.kept.0.clone_from(&net.w);
        self.kept.1.clone_from(&net.b);
        edit(net);
        let out = body(net);
        net.w.clone_from(&self.kept.0);
        net.b.clone_from(&self.kept.1);
        out
    }

    /// Put the optimiser and its settings back to what a scene opens with: the
    /// "things to try" start from here, so that their captions hold whatever was
    /// set before. (The network's shape and the rate are the caller's to choose.)
    pub fn usual_settings(&mut self, optim: Optim, lr: f32, batch: usize, decay: f32) {
        self.optim = Optim::ALL.iter().position(|&o| o == optim).unwrap_or(0);
        (self.lr, self.momentum, self.batch, self.decay) = (lr, optim.default_momentum(), batch, decay);
    }

    /// Is anything still changing (training, or an answer on its way)? (A run that
    /// has diverged changes no more, whatever `running` says.)
    pub fn busy(&self) -> bool {
        (self.running && !self.diverged) || self.awaiting > 0
    }

    /// Layer widths for a network with the chosen hidden layers.
    pub fn sizes(&self, inputs: usize, outputs: usize) -> Vec<usize> {
        layer_sizes(inputs, self.hidden, self.width, outputs)
    }

    /// The settings to train with now. The rate is in steps per second of the
    /// scene's time: the job counts real seconds, so it is told the rate the app's
    /// time speed makes of it.
    fn settings(&self) -> Settings {
        Settings {
            optim: Optim::ALL[self.optim],
            lr: self.lr,
            momentum: self.momentum,
            decay: self.decay,
            batch: self.batch,
            rate: if self.rate >= RATE_MAX { UNLIMITED } else { self.rate as f32 * self.speed },
            running: self.running && self.live && !self.diverged,
        }
    }

    fn send(&self, msg: Vec<u8>) {
        if let (true, Some(link)) = (self.remote, &self.link) {
            link.send(SERVICE, &msg);
        }
    }

    /// Start afresh on a new network (and the problem it is to learn).
    pub fn load(&mut self, net: Mlp, problem: Problem) {
        self.epoch += 1;
        self.session = Session::new(net, problem, 7919 * self.slot as u64 + self.epoch as u64);
        self.steps = 0;
        self.rate_steps = 0;
        self.diverged = false;
        self.redraw.reset();
        self.recorded_steps = u64::MAX;
        for h in self.loss.iter_mut().chain(&mut self.accuracy) {
            h.clear();
        }
        self.session.settings = self.settings();
        self.send(session::load(self.slot, self.epoch, &self.session));
        self.sent = Some(self.session.settings);
        self.awaiting = if self.remote { AWAIT_FRAMES } else { 0 };
    }

    /// The data changed; the weights and the run so far stay.
    pub fn set_problem(&mut self, problem: Problem) {
        self.send(session::problem(self.slot, &problem));
        self.session.problem = problem;
    }

    /// Take one step, whether running or paused.
    pub fn step_once(&mut self) {
        if self.remote {
            self.send(session::step(self.slot, 1));
            self.awaiting = AWAIT_FRAMES;
        } else {
            self.session.request_steps(1);
        }
    }

    /// Once a frame: move training along, here or in the background, and bring the
    /// network up to date. `dt` is the scene's time step (`Frame::dt`: zero while
    /// the scene is paused, and following the app's time speed), which the rate
    /// counts in; `real_dt` is the time since the previous frame (`Frame::real_dt`),
    /// which sizes the time allowance.
    pub fn advance(&mut self, dt: f32, real_dt: f32) {
        self.live = dt > 0.0;
        // (rounded, so that the settings sent stay the same from frame to frame)
        self.speed = if self.live && real_dt > 0.0 { ((dt / real_dt) * 1000.0).round() / 1000.0 } else { 1.0 };
        self.scene_time += dt.max(0.0) as f64;
        if real_dt > 0.0 {
            // Frames can only be late, never early, so the shortest recent interval is
            // the display's own. It may drift up slowly, to follow a change of display.
            self.frame_interval = (self.frame_interval * 1.002).min(real_dt as f64).clamp(1.0 / 360.0, 1.0 / 30.0);
        }
        let settings = self.settings();

        // hand the job over as soon as the background is up
        if !self.remote {
            match self.link.as_ref().map(|l| (l.ready(), l.failed())) {
                Some((true, _)) => {
                    self.remote = true;
                    self.session.settings = settings;
                    self.send(session::load(self.slot, self.epoch, &self.session));
                    self.sent = Some(settings);
                }
                Some((_, true)) => self.link = None,
                _ => {}
            }
        } else if self.link.as_ref().is_some_and(|l| l.failed()) {
            self.fall_back();
        }
        // (until the background is up, or has had its chance)
        self.held = !self.remote && self.link.is_some() && clock::now() - self.born < HOLD;

        if let (true, Some(link)) = (self.remote, self.link.clone()) {
            if self.sent != Some(settings) {
                if self.sent.is_some_and(|s| s.running) && !settings.running {
                    // the background finishes the slice it is in and sends where that
                    // left the job: keep looking until it has been seen
                    self.awaiting = AWAIT_FRAMES;
                }
                link.send(SERVICE, &session::settings(self.slot, settings));
                self.sent = Some(settings);
            } else {
                link.send(SERVICE, &session::alive(self.slot));
            }
            let (slot, epoch) = (self.slot, self.epoch);
            let job = &mut self.session;
            link.take(
                SERVICE,
                |msg| Snapshot::slot_of(msg) == Some(slot),
                |msg| {
                    if let Some(s) = Snapshot::decode(msg).filter(|s| s.epoch == epoch && s.params.len() == job.net.n_params()) {
                        job.net.set_params(&s.params);
                        job.steps = s.steps;
                        job.diverged = s.diverged;
                    }
                },
            );
            // (counted down, not ended by the first snapshot: more may follow it)
            self.awaiting = self.awaiting.saturating_sub(1);
        } else {
            self.session.settings = settings;
            let budget = (TRAIN_SHARE * self.frame_interval).min(TRAIN_BUDGET_MAX);
            if self.held {
                // nothing yet: the background is about to take the job
            } else if self.overrun > 0.0 {
                // the last frame's training overran its allowance: this frame's is
                // spent making up for it
                self.overrun = (self.overrun - budget).max(0.0);
            } else {
                // (the rate in the settings is per real second, as the background has it)
                let start = clock::now();
                self.session.work(if self.live { real_dt.max(0.0) as f64 } else { 0.0 }, budget);
                // (in a reproducible run the time taken must decide nothing)
                self.overrun = if clock::reproducible() { 0.0 } else { (clock::now() - start - budget).max(0.0) };
            }
            self.awaiting = 0;
        }

        self.steps = self.session.steps;
        if self.session.diverged && !self.diverged {
            self.diverged = true;
            self.running = false;
        }
        let now = clock::now();
        if now - self.rate_time >= 0.5 {
            // in steps per second of the scene's time, like the rate asked for (and
            // not shown in reproducible renders: it depends on the machine)
            let scene = self.scene_time - self.rate_scene;
            self.achieved = if clock::reproducible() || scene <= 0.0 { 0.0 } else { (self.steps.saturating_sub(self.rate_steps) as f64 / scene) as f32 };
            self.rate_time = now;
            self.rate_scene = self.scene_time;
            self.rate_steps = self.steps;
        }
    }

    /// The background has failed after taking the job: carry on here, from the
    /// weights and the step count of its last snapshot. Its optimiser's memory went
    /// with it, so the optimiser starts afresh (the copy here is from before the
    /// hand-over, and would be further from the truth).
    fn fall_back(&mut self) {
        (self.remote, self.link, self.sent) = (false, None, None);
        self.session.net.reset_optimizer();
    }

    /// Do the pictures show the weights as they stand (nothing has been trained
    /// since they were last recomputed)?
    pub fn drawn_is_current(&self) -> bool {
        self.redraw.shown == Some(self.steps)
    }

    /// Should the pictures be recomputed this frame? Yes if the weights have changed
    /// since they were drawn, or `force` says something else has. Nothing is
    /// recomputed while nothing changes, and a slow recomputation is spread out.
    pub fn refresh_due(&mut self, force: bool) -> bool {
        self.redraw.due(self.steps, force)
    }

    /// The pictures are up to date again; recomputing them took `seconds`.
    pub fn refreshed(&mut self, seconds: f64) {
        self.redraw.done(self.steps, seconds);
    }

    /// Add a reading to the history curves (once per step count). It stands for
    /// every step since the last reading, so that the charts' horizontal axis is in
    /// steps however often the pictures happen to be recomputed. (Those steps are
    /// given values along the straight line from the last reading to this one, as
    /// a chart joining the readings would draw them.)
    pub fn record(&mut self, loss: [f32; 2], accuracy: [f32; 2]) {
        if self.steps == self.recorded_steps {
            return;
        }
        // (the first reading of a run is of where it starts)
        let first = self.recorded_steps > self.steps;
        let covered = if first { 1 } else { self.steps - self.recorded_steps };
        let (before, now) = (self.recorded, (loss, accuracy));
        for i in 0..2 {
            fill(&mut self.loss[i], if first { loss[i] } else { before.0[i] }, loss[i], covered);
            fill(&mut self.accuracy[i], if first { accuracy[i] } else { before.1[i] }, accuracy[i], covered);
        }
        (self.recorded_steps, self.recorded) = (self.steps, now);
    }

    // ----- controls --------------------------------------------------------------

    /// The pause / step / reset row. True if the weights are to be reset.
    pub fn ui_run(&mut self, ui: &mut egui::Ui) -> bool {
        let mut reset = false;
        // (a run that has diverged cannot go on: only a reset helps)
        let can = !self.diverged;
        ui.horizontal(|ui| {
            let label = if self.running && can { "pause" } else { "train" };
            if ui.add_enabled(can, egui::Button::new(label)).on_disabled_hover_text("Training diverged: reset the weights.").clicked() {
                self.running = !self.running;
            }
            if ui.add_enabled(can, egui::Button::new("step")).clicked() {
                self.step_once();
            }
            reset = ui.button("reset weights").clicked();
        });
        if self.held {
            ui::note(ui, "Waiting for the training thread to start (a moment).");
        }
        reset
    }

    /// True if the network must be rebuilt.
    pub fn ui_network(&mut self, ui: &mut egui::Ui, max_hidden: usize, max_width: usize) -> bool {
        let mut changed = ui::slider_int(ui, "hidden layers", &mut self.hidden, 1..=max_hidden);
        changed |= if max_width > 16 {
            ui::slider_int_log(ui, "neurons per layer", &mut self.width, 2..=max_width)
        } else {
            ui::slider_int(ui, "neurons per layer", &mut self.width, 1..=max_width)
        };
        changed
    }

    /// True if the network must be rebuilt.
    pub fn ui_activation(&mut self, ui: &mut egui::Ui) -> bool {
        ui::section(ui, "Activation");
        let names: Vec<&str> = Act::ALL.iter().map(|a| activation_name(*a)).collect();
        let changed = ui::choice(ui, &names, &mut self.act);
        match self.act() {
            Act::Hermite2 | Act::Hermite3 => ui::note(ui, "A Hermite polynomial as activation: under Gaussian inputs it has a single Hermite component. Polynomials grow without bound, so training can run away; lower the learning rate if it does."),
            Act::Sin => ui::note(ui, "Periodic: each neuron draws stripes rather than one edge."),
            Act::ReluSquared => ui::note(ui, "Smooth where ReLU has a kink, and quadratic beyond it."),
            _ => {}
        }
        changed
    }

    pub fn ui_optimizer(&mut self, ui: &mut egui::Ui) {
        ui::section(ui, "Optimiser");
        let names: Vec<&str> = Optim::ALL.iter().map(|o| o.name()).collect();
        if ui::choice(ui, &names, &mut self.optim) {
            // each optimiser has its own natural step size (and its own memory, which
            // the network starts afresh by itself at the first step of another one)
            self.lr = Optim::ALL[self.optim].default_lr();
            self.momentum = Optim::ALL[self.optim].default_momentum();
        }
        if Optim::ALL[self.optim].uses_momentum() {
            ui::slider(ui, "momentum", &mut self.momentum, 0.0..=0.99);
        }
        if Optim::ALL[self.optim] == Optim::Muon {
            ui::note(ui, "Muon replaces each weight matrix's update by an orthogonal matrix: all its singular values are pushed to about one, so every direction of the layer moves at the same rate, however lopsided the gradient. Applied here to every weight matrix (its authors recommend hidden layers only); biases use Adam.");
        }
        ui::note(ui, "Switching optimiser keeps the weights and continues from where training is, so you can compare from the same point: press \"reset weights\" for a fresh race.");
        if self.diverged {
            ui.label(
                egui::RichText::new("Training diverged (the loss is no longer a number). Lower the learning rate and reset the weights.")
                    .size(12.5)
                    .color(pal::RED.egui()),
            );
        }
    }

    pub fn ui_training(&mut self, ui: &mut egui::Ui, max_batch: usize) {
        ui::section(ui, "Training");
        ui::slider_log(ui, "learning rate", &mut self.lr, 0.0005..=2.0);
        // The slider shows the batch as it will be used, which is at most all the
        // data; what was asked for is kept, so that a batch of 100 is 100 again when
        // the data grows back, and "all of it" (the top of the slider) stays all of it.
        let max_batch = max_batch.max(1);
        let mut shown = self.batch.clamp(1, max_batch);
        if ui::slider_int_log(ui, "batch size", &mut shown, 1..=max_batch) {
            self.batch = if shown >= max_batch { usize::MAX } else { shown };
        }
        if self.batch >= max_batch {
            ui::note(ui, "Full batch: every step uses all the training data.");
        }
        ui::slider_log_or_zero(ui, "weight decay", &mut self.decay, 1e-4, 3.0);
        ui::slider_int_log(ui, "steps per second", &mut self.rate, 10..=RATE_MAX);
        if self.rate >= RATE_MAX {
            ui::note(ui, "No limit: as many steps as the machine manages.");
        }
        ui::note(
            ui,
            if self.remote {
                "Training runs on a thread of its own; the display only shows its progress, so no setting here can slow it."
            } else if self.held {
                "Training will run on a thread of its own, which is still starting."
            } else {
                "Training shares each frame with the display and is given a fixed part of it, so heavy settings train more slowly instead of slowing the display."
            },
        );
    }

    pub fn ui_numbers(&self, ui: &mut egui::Ui) {
        ui::readout(ui, "training steps", self.steps.to_string());
        let rate = if self.running && self.achieved > 0.0 { format!("{:.0}", self.achieved) } else { "-".to_string() };
        ui::readout(ui, "steps per second, achieved", rate);
        ui::readout(ui, "parameters", self.session.net.n_params().to_string());
    }
}

/// `steps` values into a history, going in a straight line from just after `from`
/// to `to`: in as many pieces as the points they will make, so that a long stretch
/// is a slope and not a stair.
fn fill(history: &mut History, from: f32, to: f32, steps: u64) {
    let pieces = steps.div_ceil(history.every() as u64).max(1);
    let mut done = 0;
    for k in 1..=pieces {
        let upto = steps * k / pieces;
        history.push_n(from + (to - from) * (k as f32 / pieces as f32), upto - done);
        done = upto;
    }
}

impl Drop for Training {
    fn drop(&mut self) {
        self.send(session::drop(self.slot));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A job trained here in the frame, whatever background there is.
    fn local(net: Mlp, problem: Problem) -> Training {
        let mut t = Training::new(2, 8);
        t.link = None;
        t.load(net, problem);
        t
    }

    fn job(seed: u64, sizes: &[usize]) -> (Mlp, Problem) {
        let mut rng = Rng::new(seed);
        let data = make_data(1, 260, 0.2, &mut rng);
        (Mlp::new(sizes, Act::Tanh, &mut rng), Problem::Plane(data))
    }

    /// The charts' horizontal axis is in steps, however unevenly the readings come:
    /// fifty readings twenty steps apart, a hundred one step apart (the rate lowered
    /// mid-run), then one 5000 steps on ("no limit"). Reading the step count itself,
    /// the points must lie evenly from start to finish; the hundred steps take a
    /// sixtieth of the width, not (as they did) two thirds of it.
    #[test]
    fn a_history_point_stands_for_steps_not_for_readings() {
        let (net, problem) = job(1, &[2, 4, 1]);
        let mut t = local(net, problem);
        let read = |t: &mut Training, steps: u64| {
            t.steps = steps;
            t.record([steps as f32; 2], [0.0; 2]);
        };
        read(&mut t, 0);
        for k in 1..=50 {
            read(&mut t, 20 * k);
        }
        for k in 1..=100 {
            read(&mut t, 1000 + k);
        }
        read(&mut t, 6100);
        let (points, every) = (t.loss[0].points(), t.loss[0].every() as f32);
        assert!((points.len() as f32 * every - 6101.0).abs() <= every, "{} points of {every}", points.len());
        for (k, &p) in points.iter().enumerate() {
            // the mean of steps k * every .. (k + 1) * every, to within a point
            assert!((p - (k as f32 * every + (every - 1.0) * 0.5)).abs() <= every, "point {k} of {every} steps: {p}");
        }
        let late = points.iter().filter(|&&v| v > 1000.0 && v <= 1100.0).count();
        assert!(late as f32 <= 100.0 / every + 1.0, "the hundred steps take {late} of {} points", points.len());
        // and a reading after a reset stands for the start of the new run
        let (net, problem) = job(2, &[2, 4, 1]);
        t.load(net, problem);
        read(&mut t, 0);
        assert_eq!((t.loss[0].points(), t.loss[0].every()), (&[0.0][..], 1));
    }

    /// The rate is in steps per second of the scene's time: at half the app's time
    /// speed, a rate of 240 is 120 steps in a second of real time.
    #[test]
    fn the_rate_follows_the_time_speed() {
        let (net, problem) = job(3, &[2, 4, 1]);
        let mut t = local(net, problem);
        t.rate = 240;
        t.running = true;
        for _ in 0..120 {
            t.advance(0.5 / 120.0, 1.0 / 120.0);
        }
        assert!((119..=121).contains(&t.steps), "{} steps", t.steps);
        // what the background would be told to do in real seconds
        assert_eq!(t.settings().rate, 120.0);
        // and nothing at all while the scene is paused
        for _ in 0..30 {
            t.advance(0.0, 1.0 / 120.0);
        }
        assert!((119..=121).contains(&t.steps));
    }

    /// A step that takes longer than the frame's allowance is followed by frames
    /// without one: training slows down, the frame does not. (It used to take at
    /// least one step every frame, however long.)
    #[test]
    fn a_slow_step_slows_training_and_not_the_frame() {
        let (net, problem) = job(4, &[2, 1024, 1024, 1]);
        let mut t = local(net, problem);
        (t.rate, t.batch, t.running) = (RATE_MAX, usize::MAX, true);
        let start = clock::now();
        t.advance(1.0 / 60.0, 1.0 / 60.0);
        let step = clock::now() - start;
        assert!(t.steps == 1 && step > 0.012, "a step took {:.1} ms: too quick for this test", step * 1e3);
        let (start, steps, frames) = (clock::now(), t.steps, 40);
        for _ in 0..frames {
            t.advance(1.0 / 60.0, 1.0 / 60.0);
        }
        let per_frame = (clock::now() - start) / frames as f64;
        let budget = (TRAIN_SHARE / 60.0).min(TRAIN_BUDGET_MAX);
        assert!(t.steps > steps, "no training at all");
        assert!(per_frame < 2.5 * budget, "{:.1} ms a frame against an allowance of {:.1} ms ({} steps)", per_frame * 1e3, budget * 1e3, t.steps - steps);
    }

    /// A run that has diverged is not running, whatever the button was last set to,
    /// so the scene stops redrawing an unchanging network.
    #[test]
    fn a_diverged_run_is_not_busy() {
        let (net, problem) = job(5, &[2, 4, 1]);
        let mut t = local(net, problem);
        t.running = true;
        t.session.diverged = true;
        t.advance(1.0 / 60.0, 1.0 / 60.0);
        assert!(t.diverged && !t.running && !t.busy());
        t.running = true;
        assert!(!t.busy());
        assert!(!t.settings().running);
    }

    /// Over the real background thread: pausing keeps the scene looking until the
    /// job's last steps have arrived, so that what is drawn is where the job stopped
    /// (the first snapshot after the pause used to end the wait, and steps the job
    /// took after it were found only at the next "step", as a jump of thousands).
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_pause_waits_for_the_last_steps() {
        std::thread::spawn(|| {
            let (net, problem) = job(6, &[2, 8, 8, 1]);
            let mut t = Training::new(2, 8);
            t.load(net, problem);
            (t.rate, t.running) = (RATE_MAX, true);
            let frame = || std::thread::sleep(std::time::Duration::from_millis(4));
            for _ in 0..40 {
                t.advance(1.0 / 60.0, 1.0 / 60.0);
                frame();
            }
            assert!(t.remote && t.steps > 0, "the background never took the job ({} steps)", t.steps);
            // (as in a scene: the button, then frames for as long as it is busy)
            t.running = false;
            loop {
                t.advance(1.0 / 60.0, 1.0 / 60.0);
                if !t.busy() {
                    break;
                }
                frame();
            }
            let shown = t.steps;
            std::thread::sleep(std::time::Duration::from_millis(100));
            t.advance(1.0 / 60.0, 1.0 / 60.0);
            assert_eq!(t.steps, shown, "steps arrived after the scene stopped looking");

            // if the background then fails, training goes on here from where it was
            t.fall_back();
            t.running = true;
            t.advance(1.0 / 60.0, 1.0 / 60.0);
            assert!(!t.remote && t.steps > shown, "{} steps after {shown}", t.steps);
        })
        .join()
        .unwrap();
    }
}

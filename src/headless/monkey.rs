//! `eidos monkey`: random input, to find what crashes a scene.

use super::{Options, Res, Stage};
use crate::math::rng::Rng;
use egui::{Event, Modifiers, PointerButton, Pos2, TouchDeviceId, TouchId, TouchPhase};

/// The scenes with a box that takes a formula. A formula typed there may fairly be
/// infinite or undefined everywhere (`1/0`, `sqrt(-1)`), so under random input these
/// are not held to drawing only finite things; every other scene is, whatever is done
/// to it. A new scene with a formula box belongs here. (`tools/monkey.sh` has a copy of
/// this list; a test below keeps the two the same.)
pub const FORMULA_SCENES: [&str; 10] = ["calc1", "surface", "descent", "field", "mcmc", "hermite_expand", "hermite_noise", "fit1d", "graph", "parametric"];

/// Random input, to find what crashes a scene: see [`Input`]. The sequence follows
/// from the seed, so a crash can be run again; `--verbose` prints each action as it
/// happens (the last one printed is the one that did it).
pub fn monkey(args: &[String]) -> Res<()> {
    let scene = args.first().ok_or("monkey: missing scene id")?.clone();
    let mut options = Options::new(&args[1..]);
    let frames: u32 = options.parsed("--frames", 900)?;
    let seed: u64 = options.parsed("--seed", 1)?;
    let verbose = options.flag("--verbose");
    // (without drawing: many times faster where there is no GPU, and it then tests
    // everything but the renderer)
    let dry = options.flag("--dry");
    // stop at the first frame that places something where the numbers are not finite
    // (which input that reaches a formula box may fairly do: this is for the scenes
    // that have none)
    let finite = options.flag("--finite");
    options.done("monkey")?;
    crate::clock::set_reproducible(true);
    let mut stage = Stage::showing(&scene)?;
    stage.app.set_dry(dry);
    if let Some((frame, count)) = run(&mut stage, &scene, seed, frames, verbose, finite) {
        return Err(format!("{scene}: frame {frame} (seed {seed}) places {count} things where the numbers are not finite").into());
    }
    let _ = stage.device.poll(wgpu::PollType::wait_indefinitely());
    println!("{scene}: survived {frames} frames of random input (seed {seed})");
    Ok(())
}

/// Play `frames` frames of the input that follows from `seed` to `scene` on `stage`
/// (what `eidos monkey` does, and the scene check too). With `finite`, stop at the
/// first frame that places something where the numbers are not finite, and say which
/// frame that was and how many things it placed there.
pub(crate) fn run(stage: &mut Stage, scene: &str, seed: u64, frames: u32, verbose: bool, finite: bool) -> Option<(u32, usize)> {
    let mut input = Input::new(seed);
    for frame in 0..frames {
        let (raw, did) = input.next(frame);
        if verbose && !did.is_empty() {
            println!("frame {frame}: {did}");
        }
        let began = std::time::Instant::now();
        let out = stage.frame(raw);
        stage.done(&out.textures_delta);
        if verbose && began.elapsed().as_secs_f32() > 1.0 {
            println!("frame {frame}: took {:.1} s (scene code {:.1} s of it)", began.elapsed().as_secs_f32(), stage.app.last_scene_seconds());
        }
        if finite && stage.app.left_out() > 0 {
            return Some((frame, stage.app.left_out()));
        }
        // a key or a click in a narrow window may have changed the scene: come back
        if stage.app.current_id() != scene {
            stage.app.select(scene);
        }
    }
    None
}

/// Window sizes, mostly a small one (quick to draw), now and then another shape, down
/// to the smallest the contract asks for.
const SIZES: [[f32; 2]; 8] =
    [[900.0, 520.0], [1500.0, 940.0], [390.0, 800.0], [640.0, 360.0], [1100.0, 700.0], [300.0, 240.0], [240.0, 180.0], [2000.0, 1100.0]];
/// Seconds between frames: a fast display's, the usual, a slow machine's, and the
/// longest the application lets a frame step its scene by.
const STEPS: [f64; 4] = [1.0 / 240.0, 1.0 / 60.0, 1.0 / 30.0, 1.0 / 15.0];
/// Pixels per point: an ordinary screen, and a phone's or a high-resolution laptop's.
const RATIOS: [f32; 3] = [1.0, 2.0, 3.0];
const KEYS: [egui::Key; 12] = [
    egui::Key::Space,
    egui::Key::R,
    egui::Key::H,
    egui::Key::G,
    egui::Key::F1,
    egui::Key::Escape,
    egui::Key::Enter,
    egui::Key::Tab,
    egui::Key::Backspace,
    egui::Key::ArrowLeft,
    egui::Key::ArrowUp,
    egui::Key::Delete,
];
const TEXT: [&str; 18] = ["sin(", "x", "y", "^", "1e30", "/0", ")", "-", "abc", "0", "99999", ".", "*", " ", "(((", "sqrt(-1)", "t", ","];

/// The random input itself, a frame at a time: clicks and taps (mostly on the
/// controls); drags with the mouse (with either button, or with shift held) and with
/// a finger; two-finger pinches; scrolling and zooming; key presses; text typed into
/// whatever has the focus; and now and then another window, with another size,
/// pixel ratio and time between frames.
struct Input {
    rng: Rng,
    size: [f32; 2],
    ratio: f32,
    step: f64,
    time: f64,
    held: Option<Held>,
}

/// A press in progress: what is pressing, where it started and where it is going,
/// and how many frames are left of it and in all.
struct Held {
    how: Press,
    from: Pos2,
    to: Pos2,
    left: u32,
    total: u32,
}

#[derive(Clone, Copy)]
enum Press {
    /// a mouse button, with these keys held
    Mouse(PointerButton, Modifiers),
    Finger,
    /// two fingers: the first goes from `from` to `to`, the second opposite it about
    /// this centre
    Pinch(Pos2),
}

impl Input {
    fn new(seed: u64) -> Self {
        Input { rng: Rng::new(seed), size: SIZES[0], ratio: RATIOS[0], step: STEPS[1], time: 0.0, held: None }
    }

    /// The input of frame number `frame`, and what it does, in words (empty if nothing new).
    fn next(&mut self, frame: u32) -> (egui::RawInput, String) {
        let mut events = Vec::new();
        let mut did = String::new();
        let rng = &mut self.rng;
        if self.held.is_none() && frame % 150 == 149 {
            self.size = SIZES[rng.below(SIZES.len())];
            self.ratio = RATIOS[rng.below(RATIOS.len())];
            self.step = STEPS[rng.below(STEPS.len())];
            did = format!("window {}x{} at {} pixels per point, 1/{:.0} s between frames", self.size[0], self.size[1], self.ratio, 1.0 / self.step);
        }
        let (w, h) = (self.size[0], self.size[1]);
        // where on the window: mostly the controls on the right, otherwise anywhere
        // but the scene list (which would change the scene)
        let place = |rng: &mut Rng| {
            let list = if w >= 900.0 { 236.0 } else { 0.0 };
            let x = if rng.bool(0.6) { rng.range((w - 285.0).max(0.0), w - 4.0) } else { rng.range(list, w - 1.0) };
            egui::pos2(x, rng.range(0.0, h - 1.0))
        };
        let finger = |id: u64, phase, pos| Event::Touch { device_id: TouchDeviceId(0), id: TouchId(id), phase, pos, force: None };
        let button = |pos, button, pressed, modifiers| Event::PointerButton { pos, button, pressed, modifiers };
        match self.held.take() {
            Some(Held { how, from, to, left, total }) => {
                let at = from + (to - from) * (1.0 - left as f32 / total as f32);
                let up = left == 0;
                match how {
                    Press::Mouse(which, keys) => {
                        events.push(Event::PointerMoved(at));
                        if up {
                            events.push(button(to, which, false, keys));
                            if !keys.is_none() {
                                events.push(Event::ModifiersChanged(Modifiers::NONE));
                            }
                        }
                    }
                    Press::Finger | Press::Pinch(_) => {
                        // (as a touch screen reports it: each finger, and the first as the pointer too)
                        let phase = if up { TouchPhase::End } else { TouchPhase::Move };
                        events.push(finger(0, phase, at));
                        if let Press::Pinch(centre) = how {
                            events.push(finger(1, phase, centre + (centre - at)));
                        }
                        events.push(Event::PointerMoved(at));
                        if up {
                            events.extend([button(to, PointerButton::Primary, false, Modifiers::NONE), Event::PointerGone]);
                        }
                    }
                }
                if !up {
                    self.held = Some(Held { how, from, to, left: left - 1, total });
                }
            }
            None if did.is_empty() && frame.is_multiple_of(2) => {
                let at = place(rng);
                match rng.below(100) {
                    0..=39 => {
                        let how = if rng.bool(0.8) { Press::Mouse(PointerButton::Primary, Modifiers::NONE) } else { Press::Finger };
                        did = format!("{} {:.0},{:.0}", if matches!(how, Press::Finger) { "tap" } else { "click" }, at.x, at.y);
                        self.held = Some(Held { how, from: at, to: at, left: 0, total: 1 });
                    }
                    40..=64 => {
                        // along a slider as often as not
                        let to = if rng.bool(0.5) { egui::pos2(at.x + rng.range(-260.0, 260.0), at.y) } else { place(rng) };
                        let total = 2 + rng.below(8) as u32;
                        let (how, name) = match rng.below(20) {
                            0..=10 => (Press::Mouse(PointerButton::Primary, Modifiers::NONE), "drag"),
                            11..=13 => (Press::Mouse(PointerButton::Secondary, Modifiers::NONE), "right-drag"),
                            14..=16 => (Press::Mouse(PointerButton::Primary, Modifiers::SHIFT), "shift-drag"),
                            _ => (Press::Finger, "finger drag"),
                        };
                        did = format!("{name} {:.0},{:.0} to {:.0},{:.0}", at.x, at.y, to.x, to.y);
                        self.held = Some(Held { how, from: at, to, left: total, total });
                    }
                    65..=71 => {
                        // the first finger from one distance from the centre to another
                        // (never quite onto it: real fingertips do not meet at a point)
                        let angle = rng.range(0.0, std::f32::consts::TAU);
                        let dir = egui::vec2(angle.cos(), angle.sin());
                        let (near, far) = (rng.range(4.0, 200.0), rng.range(4.0, 200.0));
                        let total = 2 + rng.below(10) as u32;
                        did = format!("pinch at {:.0},{:.0}, fingers {:.0} to {:.0} apart", at.x, at.y, 2.0 * near, 2.0 * far);
                        self.held = Some(Held { how: Press::Pinch(at), from: at + dir * near, to: at + dir * far, left: total, total });
                    }
                    72..=84 => {
                        events.push(Event::PointerMoved(at));
                        if rng.bool(0.75) {
                            let amount = rng.range(-500.0, 500.0);
                            did = format!("scroll {amount:.0} at {:.0},{:.0}", at.x, at.y);
                            events.push(Event::MouseWheel {
                                unit: egui::MouseWheelUnit::Point,
                                delta: egui::vec2(0.0, amount),
                                phase: TouchPhase::Move,
                                source: egui::MouseWheelSource::Wheel,
                                modifiers: Modifiers::NONE,
                            });
                        } else {
                            // a trackpad's pinch
                            let factor = rng.range(-0.7, 0.7).exp();
                            did = format!("zoom by {factor:.2} at {:.0},{:.0}", at.x, at.y);
                            events.push(Event::Zoom(factor));
                        }
                    }
                    85..=92 => {
                        let key = KEYS[rng.below(KEYS.len())];
                        did = format!("key {key:?}");
                        for pressed in [true, false] {
                            events.push(Event::Key { key, physical_key: None, pressed, repeat: false, modifiers: Modifiers::NONE });
                        }
                    }
                    _ => {
                        let text = TEXT[rng.below(TEXT.len())];
                        did = format!("type {text:?}");
                        events.push(Event::Text(text.to_string()));
                    }
                }
                // the press itself
                if let Some(Held { how, from, .. }) = &self.held {
                    match *how {
                        Press::Mouse(which, keys) => {
                            if !keys.is_none() {
                                events.push(Event::ModifiersChanged(keys));
                            }
                            events.extend([Event::PointerMoved(*from), button(*from, which, true, keys)]);
                        }
                        Press::Finger | Press::Pinch(_) => {
                            events.push(finger(0, TouchPhase::Start, *from));
                            if let Press::Pinch(centre) = *how {
                                events.push(finger(1, TouchPhase::Start, centre + (centre - *from)));
                            }
                            events.extend([Event::PointerMoved(*from), button(*from, PointerButton::Primary, true, Modifiers::NONE)]);
                        }
                    }
                }
            }
            None => {}
        }
        let mut raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(Pos2::ZERO, egui::vec2(w, h))),
            time: Some(self.time),
            predicted_dt: self.step as f32,
            focused: true,
            events,
            ..Default::default()
        };
        raw.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(self.ratio);
        self.time += self.step;
        (raw, did)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_script_exempts_the_same_scenes() {
        // tools/monkey.sh does not pass --finite for the scenes with a formula box:
        // its list must be this one
        let script = include_str!("../../tools/monkey.sh");
        let line = script.lines().find(|l| l.starts_with("formula_scenes=")).expect("tools/monkey.sh names the scenes with a formula box");
        let named: Vec<&str> = line.trim_start_matches("formula_scenes=").trim_matches('"').split_whitespace().collect();
        assert_eq!(named, FORMULA_SCENES, "tools/monkey.sh and FORMULA_SCENES name different scenes");
        let ids: Vec<&str> = crate::scenes::registry().iter().map(|d| d.id).collect();
        for id in FORMULA_SCENES {
            assert!(ids.contains(&id), "FORMULA_SCENES names \"{id}\", which is not a scene");
        }
    }

    #[test]
    fn the_input_follows_from_the_seed_and_tries_everything() {
        let play = |seed| {
            let mut input = Input::new(seed);
            (0..3000).map(|frame| input.next(frame)).map(|(raw, did)| (format!("{:?} {:?} {:?}", raw.events, raw.time, raw.viewports), did)).collect::<Vec<_>>()
        };
        assert!(play(5) == play(5), "the same seed gave different input");
        assert!(play(5) != play(6), "another seed gave the same input");
        // (the window changes only now and then, so it takes a few runs to see them all)
        let said: String = (1..=8).flat_map(play).map(|(_, did)| did).collect::<Vec<_>>().join("\n");
        for kind in [
            "click",
            "tap",
            "drag",
            "right-drag",
            "shift-drag",
            "finger drag",
            "pinch",
            "scroll",
            "zoom",
            "key",
            "type",
            "window 240x180",
            "at 3 pixels",
            "1/240 s",
            "1/15 s",
        ] {
            assert!(said.contains(kind), "eight runs of 3000 frames never had a {kind}");
        }
    }
}

//! `eidos motion`: how evenly each scene moves.
//!
//! A picture can cost next to nothing to draw and still not look smooth. Whatever
//! takes steps (a solver, a simulation) takes a whole number of them in a frame, and
//! not the same number in every frame; whatever snaps moves in jumps; a label that
//! looks for room hops from one side of its point to the other. None of that shows in
//! `eidos bench`. This runs a scene as a display of a given rate would show it, as it
//! opens and then after each of its "things to try", and measures how far the
//! picture's furthest-moving point went in each frame.
//!
//! What it prints for each run:
//!
//! - `moving`: the frames in which anything moved;
//! - `px`: how far, on average, in those frames (pixels);
//! - `jitter`: the share of those frames, in percent, in which the speed went one
//!   way and then straight back (faster, slower, faster) by a fifth of itself or
//!   more. A glide, an ease in and out, a movement that dies away: 0. Steps taken one
//!   in a frame and two in the next: 100;
//! - `stalls`: frames in which nothing moved between two in which something did;
//! - `jumps`: frames in which something went at least eight pixels, and more than
//!   five times as far as in the frame before and the frame after;
//! - `hops`: frames in which a label went twenty pixels further than twice what the
//!   picture moved (`--verbose` says which label, and from where to where);
//! - `redrawn`: frames in which nothing could be compared with the frame before.
//!
//! With `--drag x0,y0,x1,y1` the pointer drags from the one place to the other over
//! the whole run (in a window of `--size`, so that a handle can be found on a
//! screenshot of that size), and only the scene as it opens is run: what a visitor
//! sees while moving a handle.
//!
//! Only lines, points and labels are followed (not surfaces, fills or pictures), and
//! of the lines and points only those on the picture, not the ones off its edge. A
//! point is taken to be the same point as the one drawn in its place in its batch the
//! frame before, so a batch that has gained or lost points since then is left out of
//! that frame (a chart that grows, say: what moves beside it is still measured). A
//! run is the same every time: fixed amounts of work per frame instead of time
//! budgets, as in `eidos check`.

use super::{known_scene, Options, Res, Stage};
use glam::Vec2;

/// The window the scenes are shown in (a desktop one, with the panels beside the picture).
const WINDOW: [f32; 2] = [1100.0, 700.0];
/// Less than this in a frame is not movement (pixels).
const STILL: f32 = 0.05;
/// A label that goes further than this in a frame, beyond what the picture under it
/// moved, has hopped (pixels).
const HOP: f32 = 20.0;

/// What one run came to.
#[derive(Debug, PartialEq)]
pub struct Motion {
    pub moving: usize,
    pub px: f32,
    /// In percent of the moving frames.
    pub jitter: f32,
    pub stalls: usize,
    pub jumps: usize,
    pub hops: usize,
    pub redrawn: usize,
}

/// The furthest any place went between two frames, among the batches that have as
/// many places in both (None if there is no such batch with a place in it).
fn furthest(before: &[Vec<Vec2>], after: &[Vec<Vec2>]) -> Option<f32> {
    let mut most = None;
    for (b, a) in before.iter().zip(after).filter(|(b, a)| b.len() == a.len() && !a.is_empty()) {
        // (a place that could be seen in neither frame went nowhere)
        let went = b.iter().zip(a).map(|(p, q)| (*q - *p).length()).filter(|d| d.is_finite()).fold(0.0, f32::max);
        most = Some(most.map_or(went, |m: f32| m.max(went)));
    }
    most
}

/// Whether two frames' labels are the same labels, one for one: as many, each saying
/// what the one in its place said before, but for its numbers (which change as the
/// picture does).
fn same_labels<S: AsRef<str>, T: AsRef<str>>(before: &[S], after: &[T]) -> bool {
    let words = |text: &str| -> String { text.chars().filter(|c| !c.is_ascii_digit() && !matches!(c, '.' | '-' | '−' | '+' | ' ')).collect() };
    before.len() == after.len() && before.iter().zip(after).all(|(b, a)| words(b.as_ref()) == words(a.as_ref()))
}

/// Sum up a run from how far the picture moved in each frame (None for a frame that
/// could not be compared with the one before) and how far its labels did.
fn summed(moves: &[Option<f32>], labels: &[Option<f32>]) -> Motion {
    let moved: Vec<f32> = moves.iter().flatten().copied().filter(|m| *m > STILL).collect();
    let px = moved.iter().sum::<f32>() / moved.len().max(1) as f32;
    // the speed going one way and then straight back, by a fifth of itself each time
    let back_and_forth = |w: &&[Option<f32>]| match w {
        [Some(a), Some(b), Some(c)] => {
            let fifth = 0.2 * a.max(*b).max(*c);
            fifth > STILL && (b - a) * (c - b) < 0.0 && (b - a).abs().min((c - b).abs()) > fifth
        }
        _ => false,
    };
    Motion {
        moving: moved.len(),
        px,
        jitter: 100.0 * moves.windows(3).filter(back_and_forth).count() as f32 / moved.len().max(1) as f32,
        stalls: moves.windows(3).filter(|w| matches!(w, [Some(a), Some(b), Some(c)] if *a > 5.0 * STILL && *b < STILL && *c > 5.0 * STILL)).count(),
        jumps: moves.windows(3).filter(|w| matches!(w, [Some(a), Some(b), Some(c)] if *b > (5.0 * a.max(*c)).max(8.0))).count(),
        hops: labels.iter().flatten().filter(|m| **m > HOP).count(),
        redrawn: moves.iter().filter(|m| m.is_none()).count(),
    }
}

/// Run `scene` for `frames` frames at `hz` frames a second, pressing its `press`-th
/// "thing to try" at the start (None: as it opens). Returns the summary and how many
/// things to try its panel has.
///
/// In a window of `size` logical pixels and, with `drag`, while the pointer drags
/// from one place to another (x0, y0, x1, y1) over the whole run. `verbose` prints
/// each hop of a label.
pub fn run(scene: &str, press: Option<usize>, hz: f32, frames: u32, verbose: bool, size: [f32; 2], drag: Option<[f32; 4]>) -> Res<(Motion, usize)> {
    crate::clock::set_reproducible(true);
    crate::scenes::start_afresh();
    let mut stage = Stage::showing(scene)?;
    stage.app.set_dry(true);
    let (mut moves, mut labels) = (Vec::new(), Vec::new());
    let (mut before, mut experiments): (Option<Vec<Vec<Vec2>>>, usize) = (None, 0);
    let mut said_before: Vec<String> = Vec::new();
    for frame in 0..frames {
        // (the panel is collected in every frame: that is what counts its experiments,
        // and so what lets one of them be pressed)
        crate::ui::collect_panel(true);
        if frame == 2 {
            if let Some(k) = press {
                crate::ui::press_experiment(k);
            }
        }
        let mut events = Vec::new();
        if let Some([x0, y0, x1, y1]) = drag {
            // down on the fourth frame, up on the last, and evenly along the way between
            let (from, to) = (egui::pos2(x0, y0), egui::pos2(x1, y1));
            let button = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
            let at = from + (to - from) * ((frame as f32 - 4.0) / (frames as f32 - 6.0).max(1.0)).max(0.0).min(1.0);
            events.push(egui::Event::PointerMoved(at));
            if frame == 3 {
                events.push(button(from, true));
            } else if frame + 1 == frames {
                events.push(button(to, false));
            }
        }
        let mut raw = super::input(size, 0, events);
        (raw.time, raw.predicted_dt) = (Some(frame as f64 / hz as f64), 1.0 / hz);
        let out = stage.frame(raw);
        stage.done(&out.textures_delta);
        experiments = crate::ui::collect_panel(false).experiments;
        let now = stage.app.drawn_points();
        // (the frames before the press, and the one it lands in, are not the run)
        if let (Some(was), true) = (&before, frame > 4) {
            // the labels are the last batch
            let (geometry, names) = (now.len().saturating_sub(1), now.len().saturating_sub(1)..);
            let went = furthest(&was[..geometry.min(was.len())], &now[..geometry]);
            moves.push(went);
            // a label is carried along by what it names: only what it goes beyond
            // that (twice that, to be safe) is its own doing
            let carried = 2.0 * went.unwrap_or(0.0);
            // (and only the same labels are compared: where one has come as another
            // went, the labels after it are each in the other's place in the list)
            let same = same_labels(&said_before, &stage.app.drawn_labels());
            labels.push(furthest(was.get(names.clone()).unwrap_or_default(), &now[names]).filter(|_| same).map(|m| m - carried));
            if verbose && same {
                // which labels hopped, and how far
                let (old, new, said) = (was.last().map_or(&[][..], Vec::as_slice), now.last().map_or(&[][..], Vec::as_slice), stage.app.drawn_labels());
                for ((p, q), text) in old.iter().zip(new).zip(said).filter(|_| old.len() == new.len()) {
                    if (*q - *p).length() - carried > HOP {
                        println!(
                            "    frame {frame}: \"{text}\" went {:.0} px, from ({:.0}, {:.0}) to ({:.0}, {:.0}), while the picture moved {:.0}",
                            (*q - *p).length(),
                            p.x,
                            p.y,
                            q.x,
                            q.y,
                            carried * 0.5
                        );
                    }
                }
            }
        }
        before = Some(now);
        said_before = stage.app.drawn_labels().iter().map(|l| l.to_string()).collect();
    }
    drop(stage);
    crate::scenes::start_afresh();
    Ok((summed(&moves, &labels), experiments))
}

/// `eidos motion [scene...] [--hz 60] [--seconds 6] [--size 1100x700] [--drag x0,y0,x1,y1] [--verbose]`.
pub fn motion(args: &[String]) -> Res<()> {
    let named = args.iter().take_while(|a| !a.starts_with("--")).count();
    let mut options = Options::new(&args[named..]);
    let hz: f32 = options.parsed("--hz", 60.0)?;
    let seconds: f32 = options.parsed("--seconds", 6.0)?;
    // say which label hopped in which frame
    let verbose = options.flag("--verbose");
    let size: [f32; 2] = options.numbers("--size")?.unwrap_or(WINDOW);
    let drag: Option<[f32; 4]> = options.numbers("--drag")?;
    options.done("motion")?;
    if !((1.0..=1000.0).contains(&hz) && seconds > 0.0 && seconds <= 600.0) {
        return Err(format!("motion: --hz {hz} should be from 1 to 1000, and --seconds {seconds} above 0 and no more than 600").into());
    }
    for id in &args[..named] {
        known_scene(id)?;
    }
    let all: Vec<&str> = crate::scenes::registry().iter().map(|d| d.id).collect();
    let scenes: Vec<&str> = if named == 0 { all } else { args[..named].iter().map(String::as_str).collect() };
    let frames = (hz * seconds).ceil() as u32 + 4;
    println!("How evenly each scene moves at {hz} frames a second, for {seconds} seconds after it opens and after each of its");
    println!("\"things to try\" (numbered from 1). See src/headless/motion.rs for what the columns mean.\n");
    println!("{:<15} {:>5} {:>7} {:>7} {:>7} {:>7} {:>6} {:>5} {:>8}", "scene", "try", "moving", "px", "jitter", "stalls", "jumps", "hops", "redrawn");
    for id in scenes {
        let (mut press, mut experiments) = (None, 0);
        loop {
            let (m, count) = run(id, press, hz, frames, verbose, size, drag)?;
            // (a drag is measured on the scene as it opens, and on that alone)
            experiments = if drag.is_some() { 0 } else { experiments.max(count) };
            let which = press.map_or("opens".to_string(), |k: usize| (k + 1).to_string());
            println!("{:<15} {:>5} {:>7} {:>7.2} {:>6.0}% {:>7} {:>6} {:>5} {:>8}", id, which, m.moving, m.px, m.jitter, m.stalls, m.jumps, m.hops, m.redrawn);
            let next = press.map_or(0, |k| k + 1);
            if next >= experiments {
                break;
            }
            press = Some(next);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A steady glide and a movement that dies away are even; steps taken one in a
    /// frame and two in the next are not, and a frame without any between two with
    /// some is a stall.
    #[test]
    fn evenness_is_told_from_steps() {
        let glide: Vec<Option<f32>> = (0..60).map(|_| Some(3.0)).collect();
        // labels are the same labels if only their numbers differ
        assert!(same_labels(&["area $1.00$", "$x_0$", "-4"], &["area $0.98$", "$x_0$", "2"]));
        assert!(!same_labels(&["area $1.00$", "$x_0$"], &["velocity $Ax$", "area $1.00$"]));
        assert!(!same_labels(&["$x_0$"], &["$x_0$", "$x_0$"]));
        assert_eq!(summed(&glide, &[]), Motion { moving: 60, px: 3.0, jitter: 0.0, stalls: 0, jumps: 0, hops: 0, redrawn: 0 });
        let dying: Vec<Option<f32>> = (0..40).map(|i| Some(30.0 * 0.77f32.powi(i))).collect();
        let m = summed(&dying, &[]);
        assert!(m.jitter == 0.0 && m.stalls == 0, "{m:?}");
        let ease: Vec<Option<f32>> = (0..60).map(|i| Some(8.0 * (i as f32 / 59.0 * std::f32::consts::PI).sin())).collect();
        assert_eq!(summed(&ease, &[]).jitter, 0.0);
        let steps: Vec<Option<f32>> = (0..60).map(|i| Some(if i % 2 == 0 { 2.0 } else { 4.0 })).collect();
        let m = summed(&steps, &[]);
        assert!(m.jitter > 95.0 && m.stalls == 0, "{m:?}");
        let fast_display: Vec<Option<f32>> = (0..60).map(|i| Some(if i % 4 == 3 { 0.0 } else { 2.0 })).collect();
        let m = summed(&fast_display, &[]);
        assert!(m.stalls == 14 && m.jitter > 25.0, "{m:?}");
        // a frame that cannot be compared is counted, and breaks nothing
        let mut broken = glide.clone();
        broken[10] = None;
        let m = summed(&broken, &[Some(0.0), Some(30.0)]);
        assert!(m.redrawn == 1 && m.jitter == 0.0 && m.hops == 1 && m.moving == 59, "{m:?}");
        // one frame far beyond its neighbours is a jump; the fast middle of an ease is not
        let mut jolt = glide;
        jolt[30] = Some(40.0);
        assert_eq!(summed(&jolt, &[]).jumps, 1);
        let burst: Vec<Option<f32>> = (0..60).map(|i| Some(0.2 + 40.0 * (-((i as f32 - 30.0) / 6.0).powi(2)).exp())).collect();
        assert_eq!(summed(&burst, &[]).jumps, 0);
    }

    /// Batches are compared one with one: a batch that has grown is left out, and
    /// what moved beside it still counts.
    #[test]
    fn a_batch_that_grew_is_left_out() {
        let chart = |n: usize| (0..n).map(|i| Vec2::new(i as f32, 0.0)).collect::<Vec<_>>();
        let before = vec![chart(5), vec![Vec2::ZERO, Vec2::splat(f32::NAN)]];
        let after = vec![chart(6), vec![Vec2::new(3.0, 4.0), Vec2::ONE]];
        assert_eq!(furthest(&before, &after), Some(5.0));
        assert_eq!(furthest(&before[..1], &after[..1]), None);
        assert_eq!(furthest(&[], &[]), None);
        assert_eq!(furthest(&[Vec::new()], &[Vec::new()]), None);
    }
}

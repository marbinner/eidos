//! `eidos profile`: what a visit to each scene costs from the moment it is opened,
//! frame by frame, where `eidos bench` gives one average of a settled stretch.

use super::{known_scene, Options, Res, Stage};

/// One scene's visit.
pub struct Visit {
    /// Milliseconds from choosing the scene to the end of its first frame, without
    /// the encoding of that frame's drawing.
    pub open: f64,
    /// Of that: making the scene, the scene's code in its first frame, and the rest
    /// of the first frame (the interface: the panels, and typesetting their text).
    pub made: f64,
    pub first_scene: f64,
    pub first_panels: f64,
    /// Bytes of interface textures sent in the first frame (letter shapes not met before).
    pub first_letters: f64,
    /// The longest later frame, in milliseconds, and which frame it was.
    pub worst: (f64, usize),
    /// Frames after the first that took more than `HITCH` milliseconds.
    pub hitches: usize,
    /// Mean milliseconds per frame early in the visit and at its end.
    pub early: f64,
    pub late: f64,
    /// Of `late`: the scene's own code, and encoding the drawing for the GPU.
    pub scene: f64,
    pub drawing: f64,
    /// Bytes sent to the GPU per frame at the end: the scene's buffers and pixels, and
    /// the interface's textures (the letter shapes, mostly).
    pub sent: f64,
    pub letters: f64,
    /// Frames drawn and frames found unchanged at the end.
    pub drawn: u64,
    pub reused: u64,
}

/// A frame slower than this many milliseconds of our own code is a hitch.
const HITCH: f64 = 4.0;

/// Each frame's cost as `[all, scene, drawing, sent, letters]`, with the opening time.
fn run(stage: &mut Stage, id: &str, frames: usize, size: [f32; 2], frame_no: &mut u64) -> (f64, Vec<[f64; 5]>, (u64, u64)) {
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(size[0], size[1]));
    let began = std::time::Instant::now();
    stage.app.select(id);
    let made = began.elapsed().as_secs_f64() * 1e3;
    let mut all = Vec::with_capacity(frames);
    let mut counts = (0, 0);
    for i in 0..frames {
        if i + 120 == frames || (frames < 120 && i == 0) {
            let (d, r) = stage.app.frame_counts();
            counts = (d, r);
        }
        let before = (stage.app.total_cpu, stage.app.scene_cpu, stage.app.render_cpu);
        crate::gfx::take_sent();
        let raw =
            egui::RawInput { screen_rect: Some(screen), time: Some(*frame_no as f64 / 60.0), predicted_dt: 1.0 / 60.0, focused: true, ..Default::default() };
        *frame_no += 1;
        let out = stage.frame(raw);
        let letters: usize = out.textures_delta.set.values().flat_map(|deltas| deltas.iter()).map(|delta| delta.image.width() * delta.image.height() * 4).sum();
        stage.done(&out.textures_delta);
        let app = &stage.app;
        all.push([
            (app.total_cpu - before.0) * 1e3,
            (app.scene_cpu - before.1) * 1e3,
            (app.render_cpu - before.2) * 1e3,
            crate::gfx::take_sent() as f64,
            letters as f64,
        ]);
    }
    let _ = stage.device.poll(wgpu::PollType::wait_indefinitely());
    let (d, r) = stage.app.frame_counts();
    (made, all, (d - counts.0, r - counts.1))
}

/// Visit `id` for `frames` frames, `runs` times over, and keep of each frame the
/// quickest it was: a hitch that is the scene's own comes back every time, one that
/// is this machine's (another program taking the processor) does not.
pub fn visit(stage: &mut Stage, id: &str, frames: usize, runs: usize, size: [f32; 2], frame_no: &mut u64) -> Visit {
    let frames = frames.max(300);
    let other = if id == "surface" { "graph" } else { "surface" };
    let (mut made, mut least, mut counts) = (0.0, Vec::new(), (0, 0));
    let mut first: Option<[f64; 5]> = None;
    for _ in 0..runs.max(1) {
        // (from another scene each time, so that this one is made afresh; and that one
        // shown for a moment first, so that what the program does once, whatever it
        // opens on, is not counted as this scene's)
        run(stage, other, 20, size, frame_no);
        let (m, all, c) = run(stage, id, frames, size, frame_no);
        // (the first visit's first frame is the one that meets every text for the first
        // time, as a visitor's does: that one is kept, not the quickest)
        if first.is_none() {
            (first, made) = (Some(all[0]), m);
        }
        counts = c;
        if least.is_empty() {
            least = all;
        } else {
            for (kept, new) in least.iter_mut().zip(&all) {
                if new[0] - new[2] < kept[0] - kept[2] {
                    *kept = *new;
                }
            }
        }
    }
    // our own code: everything but the encoding of the drawing, which on a machine
    // without a GPU waits on the software rasteriser
    let code = |f: &[f64; 5]| f[0] - f[2];
    let mean = |from: usize, to: usize, of: &dyn Fn(&[f64; 5]) -> f64| least[from..to].iter().map(of).sum::<f64>() / (to - from) as f64;
    let worst = least.iter().enumerate().skip(1).map(|(i, f)| (code(f), i)).fold((0.0, 0), |a, b| if b.0 > a.0 { b } else { a });
    let end = frames - 120;
    let first = first.unwrap_or([0.0; 5]);
    Visit {
        open: made + first[0] - first[2],
        made,
        first_scene: first[1],
        first_panels: first[0] - first[1] - first[2],
        first_letters: first[4],
        worst,
        hitches: least.iter().skip(1).filter(|f| code(f) > HITCH).count(),
        early: mean(30, 150, &code),
        late: mean(end, frames, &code),
        scene: mean(end, frames, &|f| f[1]),
        drawing: mean(end, frames, &|f| f[2]),
        sent: mean(end, frames, &|f| f[3]),
        letters: mean(end, frames, &|f| f[4]),
        drawn: counts.0,
        reused: counts.1,
    }
}

pub fn profile(args: &[String]) -> Res<()> {
    let named = args.iter().take_while(|a| !a.starts_with("--")).count();
    let mut options = Options::new(&args[named..]);
    let frames: usize = options.parsed("--frames", 600)?;
    let runs: usize = options.parsed("--runs", 2)?;
    let size: [f32; 2] = options.numbers("--size")?.unwrap_or([1100.0, 620.0]);
    // the first frames of the first visit, one by one
    let trace: usize = options.parsed("--trace", 0)?;
    options.done("profile")?;
    for id in &args[..named] {
        known_scene(id)?;
    }
    let all: Vec<&str> = crate::scenes::registry().iter().map(|d| d.id).collect();
    let scenes: Vec<&str> = if named == 0 { all } else { args[..named].iter().map(String::as_str).collect() };
    let mut stage = Stage::new()?;
    // (as a visitor's program does)
    crate::ui::defer_unseen_text(true);
    println!("A visit to each scene, {frames} frames at 60 a second, the quickest of {runs} visits frame by frame.");
    println!("Milliseconds of this thread's time, without the drawing itself (see `eidos bench` for what that leaves out).");
    println!("  open     from choosing the scene to the end of its first frame, the first time it is opened here: making it,");
    println!("           its own code in that frame (scene1), the panels and their text (panels1); new1: bytes of letter");
    println!("           shapes met for the first time. For a visitor's first sight of one scene, give that scene alone.");
    println!("  worst    the longest frame after the first, and which one it was");
    println!("  >{HITCH:.0}ms    how many frames after the first took longer than that");
    println!("  early    mean frame, frames 30 to 150;  late: the last 120 frames, of which scene is the scene's own code");
    println!("  encode   handing the last 120 frames' drawing to the GPU (on a machine without one, the rasteriser's time too)");
    println!("  sent     bytes of buffers and pixels sent to the GPU per frame at the end;  letters: of interface textures");
    println!("  drawn    of the last 120 frames, how many were drawn afresh\n");
    println!(
        "{:<15} {:>6} {:>6} {:>6} {:>7} {:>8} {:>7} {:>5} {:>5} {:>7} {:>7} {:>7} {:>7} {:>9} {:>8} {:>6}",
        "scene", "open", "made", "scene1", "panels1", "new1", "worst", "at", ">4ms", "early", "late", "scene", "encode", "sent", "letters", "drawn"
    );
    let mut frame_no = 0u64;
    for id in scenes {
        // (twice: the second visit finds every letter shape already drawn)
        for _ in 0..if trace > 0 { 2 } else { 0 } {
            let other = if id == "surface" { "graph" } else { "surface" };
            run(&mut stage, other, 20, size, &mut frame_no);
            // (frame by frame here, to read what the typesetter did in each)
            let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(size[0], size[1]));
            let began = std::time::Instant::now();
            stage.app.select(id);
            let made = began.elapsed().as_secs_f64() * 1e3;
            let (mut all, mut stage_sets) = (Vec::new(), Vec::new());
            crate::typeset::take_built();
            for _ in 0..trace.max(2) {
                let before = (stage.app.total_cpu, stage.app.scene_cpu, stage.app.render_cpu);
                let raw = egui::RawInput {
                    screen_rect: Some(screen),
                    time: Some(frame_no as f64 / 60.0),
                    predicted_dt: 1.0 / 60.0,
                    focused: true,
                    ..Default::default()
                };
                frame_no += 1;
                let out = stage.frame(raw);
                let letters: usize =
                    out.textures_delta.set.values().flat_map(|deltas| deltas.iter()).map(|delta| delta.image.width() * delta.image.height() * 4).sum();
                stage.done(&out.textures_delta);
                let app = &stage.app;
                all.push([(app.total_cpu - before.0) * 1e3, (app.scene_cpu - before.1) * 1e3, (app.render_cpu - before.2) * 1e3, 0.0, letters as f64]);
                stage_sets.push(crate::typeset::take_built());
            }
            println!("{id}: made in {made:.2} ms; then frame by frame: panels, scene, letters sent, paragraphs set afresh and the ms they took");
            for (i, (f, set)) in all.iter().zip(&stage_sets).enumerate() {
                println!("  {i:>3} {:>7.2} {:>7.2} {:>9.0} {:>5} {:>7.2}", f[0] - f[1] - f[2], f[1], f[4], set.0, set.1);
            }
        }
        if trace > 0 {
            let (fill, size) = stage.ctx.fonts(|f| (f.font_atlas_fill_ratio(), f.image().size));
            println!("  (egui's font atlas is now {} by {}, {:.0} % of the largest it may be)", size[0], size[1], fill * 100.0);
            continue;
        }
        let v = visit(&mut stage, id, frames, runs, size, &mut frame_no);
        println!(
            "{:<15} {:>6.1} {:>6.1} {:>6.1} {:>7.1} {:>8.0} {:>7.2} {:>5} {:>5} {:>7.3} {:>7.3} {:>7.3} {:>7.3} {:>9.0} {:>8.0} {:>6}",
            id,
            v.open,
            v.made,
            v.first_scene,
            v.first_panels,
            v.first_letters,
            v.worst.0,
            v.worst.1,
            v.hitches,
            v.early,
            v.late,
            v.scene,
            v.drawing,
            v.sent,
            v.letters,
            v.drawn
        );
        let _ = v.reused;
    }
    Ok(())
}

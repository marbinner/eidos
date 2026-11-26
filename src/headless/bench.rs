//! `eidos bench`: what each scene costs per frame.

use super::{Res, Stage};

/// The time this thread spends per frame on every scene, spent as in a visitor's
/// frame: time allowances apply and the background thread is running, so a scene
/// that trains there is charged only with what its frame does (drawing the latest
/// weights), not with the training. The GPU's own time is not included (and on a
/// machine without a GPU it would only measure the software rasteriser).
pub fn bench() -> Res<()> {
    let mut stage = Stage::new()?;
    // a desktop window, with the picture in it small: this measures our code, not the rasteriser
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1100.0, 620.0));
    println!("Milliseconds of this thread's time per frame, spent as in a visitor's frame: a desktop window with");
    println!("the panels shown, 60 frames a second of scene time, time allowances applying, the background running.");
    println!("  scene   the scene's own code: its simulation, whatever work it does in the frame, the draw list");
    println!("  panels  the interface: the panels, the scene's own controls among them");
    println!("Not included: work a scene hands to the background thread (training, in the scenes that train:");
    println!("it runs beside the frame, in a Web Worker in the browser), and drawing, which without a GPU would");
    println!("only measure the software rasteriser. Timings vary from run to run: compare runs, not single numbers.");
    println!("\"drawn\" and \"reused\" count frames rendered on the GPU and frames found unchanged.\n");
    println!("{:<15} {:>7} {:>7} {:>9} {:>8} {:>10} {:>7} {:>7}", "scene", "scene", "panels", "segments", "points", "triangles", "drawn", "reused");
    let mut frame_no = 0u64;
    for id in stage.app.scene_ids() {
        stage.app.select(id);
        for i in 0..150 {
            if i == 30 {
                stage.app.reset_timers();
            }
            let raw =
                egui::RawInput { screen_rect: Some(screen), time: Some(frame_no as f64 / 60.0), predicted_dt: 1.0 / 60.0, focused: true, ..Default::default() };
            frame_no += 1;
            let out = stage.frame(raw);
            stage.done(&out.textures_delta);
        }
        let _ = stage.device.poll(wgpu::PollType::wait_indefinitely());
        let app = &stage.app;
        let n = app.frames.max(1) as f64;
        let (segs, pts, tris) = app.stats();
        let (drawn, reused) = app.frame_counts();
        let panels = (app.total_cpu - app.scene_cpu - app.render_cpu) / n * 1e3;
        println!("{:<15} {:>7.3} {:>7.3} {:>9} {:>8} {:>10} {:>7} {:>7}", id, app.scene_cpu / n * 1e3, panels, segs, pts, tris, drawn, reused);
    }
    Ok(())
}

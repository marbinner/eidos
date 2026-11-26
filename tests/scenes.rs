//! Every scene in the catalogue is held to the scene contract (src/headless/check.rs):
//! it opens and runs in every kind of window, places nothing where the numbers are not finite
//! (left alone, or under a short run of random input), comes to rest if it says it does,
//! and links only to scenes that exist. A scene added to the catalogue is covered by
//! this without a line written for it.
//!
//! The application needs something to draw with: a GPU, or a software renderer (on
//! Linux, Mesa's: see [`HOW`]). Where there is neither, the test fails and says how to
//! give it one, unless `EIDOS_ALLOW_NO_GPU=1` says that is known and to be let pass
//! without checking anything. `eidos check` does the same work from the command line
//! and prints a line per scene.

#![cfg(not(target_arch = "wasm32"))]

const HOW: &str = "Give it a GPU, or Mesa's software renderer:\n\n    \
    WGPU_BACKEND=gl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json cargo test\n\n\
    or, where there is no way to draw at all, set EIDOS_ALLOW_NO_GPU=1 to let this test pass without checking any scene.";

/// The tests here each run the whole application, which has state of its own (the
/// background thread that trains, the reproducible clock): one at a time.
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn every_scene_keeps_the_contract() {
    let _alone = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let reports = match eidos::headless::check_all() {
        Ok(reports) => reports,
        Err(e) if e.is::<eidos::headless::NoAdapter>() => {
            if std::env::var("EIDOS_ALLOW_NO_GPU").is_ok_and(|v| v == "1") {
                eprintln!("\nSKIPPED (EIDOS_ALLOW_NO_GPU=1): {e}, so no scene was checked.\n");
                return;
            }
            panic!("the scene contract was not checked: {e}.\n{HOW}");
        }
        Err(e) => panic!("the scene contract could not be checked: {e}"),
    };
    assert!(reports.len() > 30, "the catalogue has shrunk to {} scenes", reports.len());
    let broken: Vec<String> = reports.iter().filter(|r| !r.problems.is_empty()).map(|r| format!("{}: {}", r.scene, r.problems.join("; "))).collect();
    assert!(broken.is_empty(), "scenes breaking the contract:\n{}", broken.join("\n"));
}

/// Movement that is cheap to draw can still look rough (src/headless/motion.rs). Two
/// things that once did, measured as a display of 60 and of 120 frames a second shows
/// them: the lines of "Spreading directions apart", whose solver takes one step in a
/// frame and two in the next, and the labels of "Dot and cross products", which
/// jumped to and fro while an arrow was dragged (21 times in these three seconds).
#[test]
fn what_moves_moves_evenly() {
    use eidos::headless::{motion_of, NoAdapter};
    let _alone = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for hz in [60.0f32, 120.0] {
        let frames = (3.0 * hz) as u32;
        let runs = [("packing", None), ("vectors", Some([897.0, 780.0, 1050.0, 560.0]))]
            .map(|(scene, drag)| motion_of(scene, None, hz, frames, false, [1600.0, 1000.0], drag));
        let [packing, vectors] = match runs {
            [Ok(packing), Ok(vectors)] => [packing.0, vectors.0],
            [Err(e), _] | [_, Err(e)] if e.is::<NoAdapter>() && std::env::var("EIDOS_ALLOW_NO_GPU").is_ok_and(|v| v == "1") => return,
            [Err(e), _] | [_, Err(e)] => panic!("motion could not be measured: {e}.\n{HOW}"),
        };
        assert!(packing.moving as f32 > 2.5 * hz && packing.jitter < 2.0 && packing.stalls == 0 && packing.jumps == 0, "packing at {hz} Hz: {packing:?}");
        assert!(vectors.moving as f32 > 2.0 * hz && vectors.hops == 0 && vectors.stalls == 0 && vectors.jumps == 0, "vectors at {hz} Hz: {vectors:?}");
    }
}

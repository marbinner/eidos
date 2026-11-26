//! `eidos check`: the scene contract.
//!
//! What every scene must do, whatever it shows, checked the same way for all of them,
//! so that a new scene is held to it from the moment it is listed:
//!
//! - open, and get through frames in every kind of window: a desktop one with the
//!   panels beside the picture, a phone's (with the controls closed and open), one
//!   far too small, and one with the panels hidden;
//! - place nothing where the numbers are not finite: no point, triangle, picture or
//!   label at NaN or infinity (a gap in a curve is another matter, and allowed), when
//!   left alone and under a short run of the monkey's random input (where a scene
//!   with a formula box is excused: a typed formula may fairly be infinite everywhere);
//! - if it says it only changes in answer to input (`Scene::animated` is false),
//!   really come to rest: its picture stops changing, and it stops asking for new
//!   frames;
//! - have a panel with the parts every panel has: things to try, a colour key, and
//!   links to related scenes, which lead only to scenes that exist;
//! - show no text whose mathematics does not parse (which would be drawn as it is
//!   written, dollars and all), in its panel or on its picture, as it opens.
//!
//! Each scene is tried on an application of its own, as if it had just been opened,
//! so that what one leaves behind cannot change another's verdict. Frames are built
//! but not drawn, except a still scene's in the desktop window (the renderer is what
//! tells a changed picture from an unchanged one), so the whole catalogue takes
//! seconds with or without a GPU. What a scene does under much more random input is
//! the business of `tools/monkey.sh`; whether what it says is true, of its own tests.

use super::{input, monkey, Res, Stage};

/// What the check found for one scene.
pub struct Report {
    pub scene: &'static str,
    /// Each way it breaks the contract, in words; empty if it keeps it.
    pub problems: Vec<String>,
    /// Milliseconds of scene code per frame in the desktop window.
    pub ms: f64,
}

/// The windows a scene is tried in: logical size, number of frames, whether the
/// panels are shown, and (on a narrow window) whether the controls are open.
const WINDOWS: [([f32; 2], u32, bool, bool); 5] = [
    ([1100.0, 700.0], 120, true, false),
    ([390.0, 800.0], 30, true, false),
    ([390.0, 800.0], 20, true, true),
    ([240.0, 180.0], 20, true, false),
    ([1100.0, 700.0], 20, false, false),
];

/// The run of random input every scene gets: its seed, and how many frames (enough
/// for three windows, and the whole check still takes under half a minute on a
/// software renderer).
const MONKEY_SEED: u64 = 1;
/// The scenes whose panels need not have the standard parts: the renderer's test card
/// teaches nothing.
const NOT_A_LESSON: [&str; 1] = ["showcase"];
const MONKEY_FRAMES: u32 = 400;

/// Hold the scene `id` to the contract. (An error means it could not be tried at
/// all; what it does wrong is in the report.) Every run is the same: fixed amounts of
/// work per frame instead of time budgets, and no passing hint over the picture.
pub fn check_scene(id: &str) -> Res<Report> {
    crate::clock::set_reproducible(true);
    // (so that a scene that trains starts as it does when it is the first one opened)
    crate::scenes::start_afresh();
    let mut stage = Stage::showing(id)?;
    let scene = stage.app.current_id();
    stage.app.set_dry(true);
    crate::ui::collect_panel(true);
    // every text it shows as it opens (in the first window), to be read for markup that does not parse
    crate::typeset::collect(true);
    let mut texts: Vec<String> = crate::scenes::registry().iter().filter(|d| d.id == scene).flat_map(|d| [d.title.to_string(), d.blurb.to_string()]).collect();
    let mut problems: Vec<String> = Vec::new();
    let mut note = |problem: String| {
        if !problems.contains(&problem) {
            problems.push(problem);
        }
    };
    let (mut frame_no, mut ms) = (0u64, 0.0);
    for (window, (size, frames, panels, controls)) in WINDOWS.into_iter().enumerate() {
        stage.app.set_panels(panels);
        stage.app.show_controls(controls);
        // how many of the last ten frames asked for another straight away, and how
        // many of them were a changed picture (in the desktop window, while the scene
        // said all along that it was still)
        let (mut restless, mut changed, mut still) = (0, 0, true);
        for i in 0..frames {
            if window == 0 && i == 30 {
                stage.app.reset_timers();
            }
            let quiet = !stage.app.scene_animated();
            if window == 0 {
                stage.app.set_dry(!quiet);
                // (the frame before the last ten as well: the one they are compared with)
                still &= quiet || i + 11 < frames;
            }
            let drawn = stage.app.frame_counts().0;
            let out = stage.frame(input(size, frame_no, Vec::new()));
            stage.done(&out.textures_delta);
            frame_no += 1;
            if window == 0 {
                texts.extend(stage.app.drawn_labels().into_iter().filter(|t| crate::typeset::has_math(t)).map(str::to_string));
            }
            if stage.app.left_out() > 0 {
                note(format!("places things where the numbers are not finite (in a window of {} by {})", size[0], size[1]));
            }
            let again = out.viewport_output.get(&egui::ViewportId::ROOT).is_some_and(|v| v.repaint_delay.is_zero());
            if i + 10 >= frames {
                restless += again as u32;
                changed += stage.app.frame_counts().0 - drawn;
            }
        }
        if window == 0 {
            texts.extend(crate::typeset::collect(false));
            texts.sort();
            texts.dedup();
            for text in &texts {
                if let Some(error) = crate::typeset::inspect(text).error {
                    note(format!("shows text whose mathematics does not parse ({error}): \"{text}\""));
                }
            }
            stage.app.set_dry(true);
            ms = stage.app.scene_cpu / stage.app.frames.max(1) as f64 * 1e3;
            if !stage.app.scene_animated() && restless == 10 {
                note("says it changes only in answer to input, yet keeps asking for new frames when left alone".to_string());
            }
            if still && changed > 0 {
                note(format!("says it changes only in answer to input, yet its picture still changed in {changed} of the last 10 frames when left alone"));
            }
        }
    }
    let panel = crate::ui::collect_panel(false);
    for link in &panel.links {
        // (a scene that is planned and not yet written may be linked to: the link is
        // shown once that scene is there)
        if !crate::scenes::planned(link) {
            note(format!("links to \"{link}\", which is not a scene"));
        }
    }
    if !NOT_A_LESSON.contains(&scene) {
        let missing: Vec<&str> = [
            (panel.experiments == 0, "things to try (ui::experiment)"),
            (!panel.key, "colour key (ui::key)"),
            (panel.links.is_empty(), "links to related scenes (ui::see_also)"),
        ]
        .into_iter()
        .filter_map(|(lacks, what)| lacks.then_some(what))
        .collect();
        if !missing.is_empty() {
            note(format!("its panel has no {}", missing.join(", no ")));
        }
    }
    drop(stage);
    crate::scenes::start_afresh();

    // Random input, from the start on a new application, exactly as `eidos monkey`
    // gives it (so the command in the report shows it again).
    let mut stage = Stage::showing(id)?;
    stage.app.set_dry(true);
    let finite = !monkey::FORMULA_SCENES.contains(&scene);
    if let Some((frame, count)) = monkey::run(&mut stage, scene, MONKEY_SEED, MONKEY_FRAMES, false, finite) {
        note(format!(
            "places {count} things where the numbers are not finite after random input (eidos monkey {scene} --seed {MONKEY_SEED} --frames {} --dry --finite)",
            frame + 1
        ));
    }
    Ok(Report { scene, problems, ms })
}

/// Every scene in the catalogue, in order.
pub fn check_all() -> Res<Vec<Report>> {
    crate::scenes::registry().iter().map(|d| check_scene(d.id)).collect()
}

/// `eidos check [scene...]`: print what was found; an error if anything was.
pub fn check(args: &[String]) -> Res<()> {
    // (every name first, so that a mistyped one is said at once)
    for id in args {
        super::known_scene(id)?;
    }
    let reports = if args.is_empty() { check_all()? } else { args.iter().map(|id| check_scene(id)).collect::<Res<Vec<_>>>()? };
    println!("{:<15} {:>9}   contract", "scene", "ms/frame");
    for r in &reports {
        println!("{:<15} {:>9.3}   {}", r.scene, r.ms, if r.problems.is_empty() { "kept".to_string() } else { format!("BROKEN: {}", r.problems.join("; ")) });
    }
    let broken = reports.iter().filter(|r| !r.problems.is_empty()).count();
    println!("\n{} scenes checked, {} of them breaking the contract", reports.len(), broken);
    if broken > 0 {
        return Err(format!("{broken} scenes break the scene contract").into());
    }
    Ok(())
}

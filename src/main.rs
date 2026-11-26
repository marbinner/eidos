//! Native entry point.
//!
//!   eidos [--scene <id>]        open the interactive window
//!   eidos list                  list scene ids
//!   eidos bench                 CPU cost per frame of every scene (no window)
//!   eidos motion [scene...] [--hz 60] [--seconds 6] [--verbose]
//!                [--size 1100x700] [--drag x0,y0,x1,y1]  (measure while dragging)
//!                               how evenly each scene moves from frame to frame, as it
//!                               opens and after each of its "things to try"
//!                               (see src/headless/motion.rs)
//!   eidos bench-training        how many training steps per second this machine does
//!   eidos glyphs <text>         which of these characters the fonts can draw
//!   eidos shot <scene> <out.png> [--size 1600x1000] [--frames 90] [--scale 1]
//!              [--no-panels] [--bare]
//!              [--click x,y]... (ten frames apart from the start of the run)
//!              [--scroll x,y,amount]  (once, after the clicks)
//!              [--drag x0,y0,x1,y1]...  (over the last two thirds of the run, one after another)
//!              [--hover x,y]    (where the pointer rests when nothing else has it)
//!              [--trace N]      (print timings every N frames)
//!              [--realtime]     (let time budgets apply, as in the window)
//!                               render a scene without a window and save a PNG
//!   eidos monkey <scene> [--frames 900] [--seed 1] [--verbose] [--dry] [--finite]
//!                               throw random clicks, drags, touches, scrolls, typing and
//!                               windows at a scene, to find what crashes it
//!                               (tools/monkey.sh runs every scene with several seeds)
//!   eidos mathlint [scene...]  what is left to typeset: markup that does not parse,
//!                               mathematics outside dollars, words in italic inside them
//!                               (see src/headless/mathlint.rs)
//!   eidos check [scene...]     hold every scene (or the ones named) to the scene contract:
//!                               finite drawing at every window size and under random
//!                               input, rest when still, links that lead somewhere
//!                               (see src/headless/check.rs)

#[cfg(not(target_arch = "wasm32"))]
use eidos::headless;

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rest = args.get(1..).unwrap_or_default();
    let result = match args.first().map(String::as_str) {
        Some("shot") => headless::shot(rest),
        Some("monkey") => headless::monkey(rest),
        Some("check") => headless::check(rest),
        Some("mathlint") => headless::mathlint(rest),
        Some("texts") => headless::print_texts(rest),
        Some("motion") => headless::motion(rest),
        Some("profile") => headless::profile(rest),
        Some("primers") => headless::primers(rest),
        Some("list") => nothing_more("list", rest).and_then(|_| headless::list()),
        Some("bench") => nothing_more("bench", rest).and_then(|_| headless::bench()),
        Some("bench-training") => nothing_more("bench-training", rest).map(|_| print!("{}", eidos::scenes::training_benchmark())),
        Some("glyphs") => headless::glyphs(&rest.join(" ")),
        Some("-h") | Some("--help") => nothing_more("--help", rest).map(|_| usage()),
        None => window(None),
        Some("--scene") => match rest {
            [id] => headless::known_scene(id).and_then(|_| window(Some(id.clone()))),
            [] => Err("--scene needs a scene id (eidos list names them)".into()),
            [_, more, ..] => Err(format!("--scene: unknown argument {more}").into()),
        },
        Some(other) => Err(format!(
            "unknown command {other}; the commands are: shot, monkey, check, mathlint, texts, primers, list, bench, profile, motion, bench-training, glyphs, --scene <id> (or nothing) to open the window, and --help"
        )
        .into()),
    };
    if let Err(e) = result {
        eprintln!("eidos: {e}");
        std::process::exit(1);
    }
}

/// An error if a command that takes no arguments was given some.
#[cfg(not(target_arch = "wasm32"))]
fn nothing_more(command: &str, rest: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    match rest.first() {
        Some(unknown) => Err(format!("{command} takes no arguments, and was given {unknown}").into()),
        None => Ok(()),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn usage() {
    println!("usage: eidos [--scene <id>]         open the window");
    println!("       eidos list                   list the scene ids");
    println!("       eidos shot <scene> <out.png> [--size WxH] [--frames N] [--scale F] [--no-panels] [--bare]");
    println!("                  [--click x,y]... [--scroll x,y,amount] [--drag x0,y0,x1,y1]... [--hover x,y]");
    println!("                  [--trace N] [--realtime]       render a scene without a window");
    println!("       eidos monkey <scene> [--frames N] [--seed S] [--verbose] [--dry] [--finite]   random input, to find crashes");
    println!("       eidos check [scene...]       hold scenes to the scene contract (all of them, if none is named)");
    println!("       eidos mathlint [scene...]    what is left to typeset (all scenes and the glossary, if none is named)");
    println!("       eidos bench                  CPU cost per frame of every scene");
    println!("       eidos motion [scene...] [--hz N] [--seconds S] [--verbose] [--size WxH] [--drag x0,y0,x1,y1]\n                                    how evenly each scene moves from frame to frame");
    println!("       eidos bench-training         training steps per second");
    println!("       eidos glyphs <text>          which of these characters the fonts can draw");
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn window(scene: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    let options =
        eframe::NativeOptions { viewport: egui::ViewportBuilder::default().with_inner_size([1480.0, 920.0]).with_title("eidos"), ..Default::default() };
    eframe::run_native(
        "eidos",
        options,
        Box::new(move |cc| {
            let mut app = eidos::EframeApp::new(cc)?;
            if let Some(id) = &scene {
                app = app.with_scene(id);
            }
            Ok(Box::new(app) as Box<dyn eframe::App>)
        }),
    )
    .map_err(|e| e.to_string().into())
}

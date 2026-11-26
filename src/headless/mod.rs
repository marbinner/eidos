//! The application without a window: what the command-line tools drive (`eidos shot`,
//! `bench`, `motion`, `monkey`, `check`) and what the scene-contract test runs (`tests/scenes.rs`).
//! Not part of the browser build.

mod bench;
mod check;
mod mathlint;
mod monkey;
mod motion;
mod primers;
mod profile;
mod shot;

pub use bench::bench;
pub use check::{check, check_all, check_scene, Report};
pub use mathlint::{mathlint, print_texts};
pub use monkey::monkey;
pub use motion::{motion, run as motion_of, Motion};
pub use primers::primers;
pub use profile::{profile, visit, Visit};
pub use shot::shot;

use crate::app::App;
use std::sync::Arc;

pub type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// Options on a command line, after the positional arguments: `--name value` pairs
/// and bare `--flags`. Each is taken out as it is asked for; whatever is left at the
/// end is something the command does not know, and [`Options::done`] says so.
pub(crate) struct Options {
    rest: Vec<String>,
}

impl Options {
    pub fn new(args: &[String]) -> Self {
        Options { rest: args.to_vec() }
    }

    /// Whether `--name` is there.
    pub fn flag(&mut self, name: &str) -> bool {
        let at = self.rest.iter().position(|a| a == name);
        at.map(|i| self.rest.remove(i)).is_some()
    }

    /// The value after the first `--name`, if it is there.
    pub fn value(&mut self, name: &str) -> Res<Option<String>> {
        let Some(i) = self.rest.iter().position(|a| a == name) else { return Ok(None) };
        if i + 1 >= self.rest.len() {
            return Err(format!("{name} needs a value").into());
        }
        let value = self.rest.remove(i + 1);
        self.rest.remove(i);
        Ok(Some(value))
    }

    /// The same, read as a number (or anything else that can be read from text).
    pub fn parsed<T: std::str::FromStr>(&mut self, name: &str, otherwise: T) -> Res<T> {
        match self.value(name)? {
            Some(text) => text.parse().map_err(|_| format!("{name}: cannot read \"{text}\"").into()),
            None => Ok(otherwise),
        }
    }

    /// The same, read as exactly `N` numbers separated by commas (or an `x`, as in 800x600).
    pub fn numbers<const N: usize>(&mut self, name: &str) -> Res<Option<[f32; N]>> {
        let Some(text) = self.value(name)? else { return Ok(None) };
        let mut read = Vec::new();
        for part in text.split([',', 'x']) {
            match part.trim().parse::<f32>() {
                Ok(v) if v.is_finite() => read.push(v),
                _ => return Err(format!("{name}: \"{part}\" in \"{text}\" is not a number").into()),
            }
        }
        match <[f32; N]>::try_from(read) {
            Ok(numbers) => Ok(Some(numbers)),
            Err(_) => Err(format!("{name}: \"{text}\" should be {N} numbers").into()),
        }
    }

    /// Every use of an option that may be given several times.
    pub fn every<const N: usize>(&mut self, name: &str) -> Res<Vec<[f32; N]>> {
        let mut all = Vec::new();
        while let Some(one) = self.numbers(name)? {
            all.push(one);
        }
        Ok(all)
    }

    /// Whatever is left was not understood.
    pub fn done(self, command: &str) -> Res<()> {
        match self.rest.first() {
            Some(unknown) => Err(format!("{command}: unknown option {unknown}").into()),
            None => Ok(()),
        }
    }
}

/// There is nothing here to draw with: no GPU, and no software renderer either.
#[derive(Debug)]
pub struct NoAdapter(String);

impl std::fmt::Display for NoAdapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "nothing here can draw ({})", self.0)
    }
}

impl std::error::Error for NoAdapter {}

#[derive(Clone)]
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter: wgpu::Adapter,
}

thread_local! {
    /// The device, once made. Every stage on a thread shares it: making one is most of
    /// the cost of a stage, and the scene check makes a stage for every run.
    static GPU: std::cell::RefCell<Option<Gpu>> = const { std::cell::RefCell::new(None) };
}

fn gpu() -> Res<Gpu> {
    if let Some(gpu) = GPU.with_borrow(Clone::clone) {
        return Ok(gpu);
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle().with_env());
    let adapter = pollster::block_on(
        instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }),
    )
    .map_err(|e| NoAdapter(e.to_string()))?;
    let info = adapter.get_info();
    eprintln!("adapter: {} ({:?})", info.name, info.backend);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("eidos"),
        required_limits: wgpu::Limits::downlevel_defaults().using_resolution(adapter.limits()),
        ..Default::default()
    }))?;
    let gpu = Gpu { device, queue, adapter };
    GPU.set(Some(gpu.clone()));
    Ok(gpu)
}

/// The application without a window: what `bench`, `monkey`, `shot` and `check` drive.
/// Each is a new application (scene, panels, egui's memory), on the one device.
pub struct Stage {
    pub app: App,
    pub ctx: egui::Context,
    pub renderer: Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Stage {
    pub fn new() -> Res<Stage> {
        let Gpu { device, queue, adapter } = gpu()?;
        let options = egui_wgpu::RendererOptions { msaa_samples: 1, depth_stencil_format: None, dithering: false, predictable_texture_filtering: false };
        let renderer = Arc::new(egui::mutex::RwLock::new(egui_wgpu::Renderer::new(&device, FORMAT, options)));
        let mut app = App::new(device.clone(), queue.clone(), &adapter, renderer.clone());
        // (the pictures made here are the ones a visitor with a GPU sees)
        app.draw_fully();
        Ok(Stage { app, ctx: egui::Context::default(), renderer, device, queue })
    }

    /// The same, already showing `scene`.
    pub fn showing(scene: &str) -> Res<Stage> {
        known_scene(scene)?;
        let mut stage = Stage::new()?;
        stage.app.select(scene);
        Ok(stage)
    }

    /// Run one frame of the interface on `input` and give the renderer the
    /// textures egui made in it. Returns what egui wants drawn; hand it back to
    /// [`Stage::done`] once that has been drawn (or is not going to be).
    pub fn frame(&mut self, input: egui::RawInput) -> egui::FullOutput {
        let app = &mut self.app;
        let out = self.ctx.run_ui(input, |ui| app.ui(ui));
        let mut renderer = self.renderer.write();
        for (id, deltas) in &out.textures_delta.set {
            for delta in deltas {
                renderer.update_texture(&self.device, &self.queue, *id, delta);
            }
        }
        out
    }

    /// Drop the textures egui no longer needs after this frame.
    pub fn done(&mut self, textures: &egui::TexturesDelta) {
        let mut renderer = self.renderer.write();
        for id in &textures.free {
            renderer.free_texture(id);
        }
    }
}

/// The pixel format the headless picture is drawn in.
pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// An error, naming every scene there is, unless `id` is one of them.
pub fn known_scene(id: &str) -> Res<()> {
    let ids: Vec<&str> = crate::scenes::registry().iter().map(|d| d.id).collect();
    if ids.contains(&id) {
        return Ok(());
    }
    Err(format!("unknown scene \"{id}\"; the scenes are: {}", ids.join(" ")).into())
}

pub fn list() -> Res<()> {
    for d in crate::scenes::registry() {
        println!("{:<14} {:<12} {}", d.id, d.group, d.title);
    }
    Ok(())
}

/// Report which characters the UI fonts can draw.
pub fn glyphs(text: &str) -> Res<()> {
    let ctx = egui::Context::default();
    crate::app::apply_style(&ctx);
    let _ = ctx.run_ui(egui::RawInput::default(), |_| {});
    let _ = ctx.run_ui(egui::RawInput::default(), |_| {});
    let id = egui::FontId::proportional(14.0);
    let (mut ok, mut missing) = (String::new(), String::new());
    for c in text.chars().filter(|c| !c.is_whitespace()) {
        if ctx.fonts_mut(|f| f.has_glyph(&id, c)) {
            ok.push(c);
        } else {
            missing.push(c);
        }
    }
    println!("ok:      {ok}\nmissing: {missing}");
    Ok(())
}

/// One frame's worth of input for a window of `size` logical pixels at frame number
/// `frame` (sixty to the second), at one pixel per point.
pub(crate) fn input(size: [f32; 2], frame: u64, events: Vec<egui::Event>) -> egui::RawInput {
    let mut raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(size[0], size[1]))),
        time: Some(frame as f64 / 60.0),
        predicted_dt: 1.0 / 60.0,
        focused: true,
        events,
        ..Default::default()
    };
    raw.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(1.0);
    raw
}

#[cfg(test)]
mod tests {
    use crate::typeset::{self, Piece};

    /// The string literals of a Rust source file, as the program has them (escapes
    /// read), leaving out comments and character literals.
    fn string_literals(source: &str) -> Vec<String> {
        let chars: Vec<char> = source.chars().collect();
        let at = |i: usize| chars.get(i).copied().unwrap_or('\0');
        let mut out = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            let word_before = i > 0 && (chars[i - 1].is_alphanumeric() || chars[i - 1] == '_');
            if c == '/' && at(i + 1) == '/' {
                while i < chars.len() && chars[i] != '\n' {
                    i += 1;
                }
            } else if c == '/' && at(i + 1) == '*' {
                i += 2;
                while i < chars.len() && !(chars[i] == '*' && at(i + 1) == '/') {
                    i += 1;
                }
                i += 2;
            } else if c == '\'' {
                // a character literal ('x', '\n', '\u{2009}'), or a lifetime
                if at(i + 1) == '\\' {
                    i += 2;
                    while i < chars.len() && chars[i] != '\'' {
                        i += 1;
                    }
                    i += 1;
                } else if at(i + 2) == '\'' {
                    i += 3;
                } else {
                    i += 1;
                }
            } else if c == 'r' && !word_before && (at(i + 1) == '"' || (at(i + 1) == '#' && matches!(at(i + 2), '"' | '#'))) {
                let mut j = i + 1;
                let mut hashes = 0;
                while at(j) == '#' {
                    hashes += 1;
                    j += 1;
                }
                j += 1;
                let start = j;
                while j < chars.len() && !(chars[j] == '"' && (1..=hashes).all(|k| at(j + k) == '#')) {
                    j += 1;
                }
                out.push(chars[start..j.min(chars.len())].iter().collect());
                i = j + 1 + hashes;
            } else if c == '"' {
                let mut text = String::new();
                let mut j = i + 1;
                while j < chars.len() && chars[j] != '"' {
                    if chars[j] == '\\' {
                        j += 1;
                        match at(j) {
                            'n' => text.push('\n'),
                            'u' => {
                                let end = (j..chars.len()).find(|&k| chars[k] == '}').unwrap_or(j);
                                let hex: String = chars[(j + 2).min(end)..end].iter().collect();
                                text.extend(u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32));
                                j = end;
                            }
                            other => text.push(other),
                        }
                    } else {
                        text.push(chars[j]);
                    }
                    j += 1;
                }
                out.push(text);
                i = j + 1;
            } else {
                i += 1;
            }
        }
        out
    }

    /// Every character the scenes write outside formulas is one the interface's fonts
    /// have: one they lack is drawn as an empty box (the low-rank scene's pause button
    /// was, with ⏸). Read from the scenes' sources, so that a button's label, which no
    /// tool sees drawn, is read too. (In a formula a missing glyph is an error of the
    /// markup, which the lint and the scene check find.)
    #[test]
    fn every_character_the_scenes_write_has_a_glyph() {
        let ctx = egui::Context::default();
        crate::app::apply_style(&ctx);
        for _ in 0..2 {
            let _ = ctx.run_ui(egui::RawInput::default(), |_| {});
        }
        let font = egui::FontId::proportional(14.0);
        let mut files = vec![std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/src/scenes"))];
        let mut missing = Vec::new();
        let mut read = 0;
        while let Some(path) = files.pop() {
            if path.is_dir() {
                files.extend(std::fs::read_dir(&path).expect("the scenes' directory").flatten().map(|e| e.path()));
                continue;
            }
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            let source = std::fs::read_to_string(&path).expect("a scene's source");
            read += 1;
            for literal in string_literals(&source) {
                let words: Vec<String> = match typeset::pieces(&literal) {
                    Ok(parts) => parts.into_iter().filter_map(|p| if let Piece::Text(t) = p { Some(t.into_owned()) } else { None }).collect(),
                    Err(_) => vec![literal.clone()],
                };
                for c in words.iter().flat_map(|w| w.chars()).filter(|c| !c.is_ascii() && !c.is_whitespace() && !typeset::combining_char(*c)) {
                    if !ctx.fonts_mut(|f| f.has_glyph(&font, c)) {
                        missing.push(format!("{c} (U+{:04X}) in {}: {literal:?}", c as u32, path.display()));
                    }
                }
            }
        }
        assert!(read > 50, "only {read} scene files read");
        assert!(missing.is_empty(), "{}", missing.join("\n"));
    }

    #[test]
    fn literals_are_read_as_the_program_has_them() {
        let source = "let a = \"x² \\u{23F8}\"; // \"not ⏸\"\nlet b = r#\"a \"quoted\" ⏸\"#; let c = '\"'; let d = 'a; /* \"⏸\" */ f(r\"\\$\")";
        assert_eq!(string_literals(source), ["x² \u{23F8}", "a \"quoted\" ⏸", "\\$"]);
    }
}

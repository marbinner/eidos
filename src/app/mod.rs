//! The application shell: scene browser, control panel, viewport, input and timing.
//! It only depends on egui + wgpu, so the same code runs in a native window, in the
//! browser, and headless for screenshots.
//!
//! Beside this file: `governor` lowers the resolution when frames are slow, `fullscreen`
//! is the full-screen button, and `style` the fonts and colours.

use crate::frame::{Frame, Handles, Places, Pointer};
use crate::gfx::camera::Camera;
use crate::gfx::color::pal;
use crate::gfx::draw::DrawList;
use crate::gfx::{EguiRenderer, Gfx, ViewSettings};
use crate::scenes::{registry, Scene, SceneDef};
use egui::{Color32, RichText};
use glam::Vec2;

mod fullscreen;
mod governor;
mod style;

use governor::Governor;
pub use style::apply_style;

/// Narrower than this (logical pixels) and the panels no longer fit beside the picture.
const NARROW: f32 = 900.0;
/// The picture is drawn at no more than this many pixels per logical pixel. Phones
/// report three or more, which costs several times the drawing for no visible gain.
const MAX_PIXEL_RATIO: f32 = 2.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Scenes,
    Controls,
}

pub struct App {
    gfx: Gfx,
    defs: Vec<SceneDef>,
    current: usize,
    scene: Box<dyn Scene>,
    cam: Camera,
    dl: DrawList,
    handles: Handles,
    places: Places,
    /// The look in force: the visitor's own choices with the scene's look set on top
    /// when it opened, then whatever the visitor changes (what the View controls show).
    settings: ViewSettings,
    /// What the visitor has chosen in the View controls, carried from scene to scene.
    chosen: ViewSettings,
    /// Scene time. (In double precision: added to frame by frame for hours, a single
    /// precision clock would run fast, and at a low speed stop.)
    time: f64,
    speed: f32,
    paused: bool,
    show_panels: bool,
    /// On a screen too narrow for both panels beside the picture, the picture gets the
    /// whole screen, and at most one panel is open at a time (it takes its width from
    /// the picture, which is left as a strip beside it).
    narrow: bool,
    open: Option<Side>,
    show_help: bool,
    show_glossary: bool,
    /// The scene's primer (the mathematics behind it, written out) is open over the
    /// application; and the primer last read into blocks, with its scene.
    show_primer: bool,
    primer_of: Option<(&'static str, Vec<crate::primer::Block>)>,
    glossary_filter: String,
    /// What the searches of the scene list and of the glossary look in, an entry for
    /// each scene and each glossary entry in order (made when first searched).
    scene_index: Vec<String>,
    glossary_index: Vec<String>,
    bare: bool,
    /// Build every frame but draw none (for the tools that check scenes: on a machine
    /// without a GPU, drawing is most of the time and none of the point).
    dry: bool,
    /// Seconds for which the "how to move the picture" hint is still shown, and the
    /// clock reading of the last frame that counted it down (frames can be seconds
    /// apart while nothing moves).
    hint: f32,
    hint_clock: Option<f64>,
    show_stats: bool,
    dragging_cam: bool,
    frame_ms: f32,
    styled: bool,
    governor: Governor,
    /// The display's frame interval in seconds, once it has been seen (0 before).
    beat: f32,
    was_animating: bool,
    filter: String,
    /// The scene list's rows as last laid out: the width they were laid out for, and
    /// the size of each scene's button (see `scene_list`).
    rows: (f32, Vec<Option<egui::Vec2>>),
    /// Seconds spent in the whole frame function (interface included), accumulated.
    pub total_cpu: f64,
    /// Seconds spent in scene code (simulation + building the draw list), accumulated.
    pub scene_cpu: f64,
    /// Seconds spent encoding and submitting GPU work, accumulated.
    pub render_cpu: f64,
    pub frames: u64,
    /// This frame's scene and render seconds, and smoothed milliseconds per frame of
    /// interface, scene and render work (for the statistics overlay).
    last: [f64; 2],
    cpu_ms: [f32; 3],
    visit: Visit,
    /// Never draw plainly, however slow the frames (see [`App::draw_fully`]).
    fully: bool,
}

/// This visit to the scene being shown, for the statistics overlay and the tools
/// that time the program in a browser. All in milliseconds.
#[derive(Clone, Copy, Default)]
struct Visit {
    /// Making the scene.
    made: f32,
    /// Its first frame, and the longest since: this thread's work in the frame.
    first: f32,
    worst: f32,
    /// The longest wait from one frame to the next since the visit settled (drawing
    /// and the display included), while the scene was moving.
    gap: f32,
    frames: u32,
}

/// Whether a tool has asked for [`stats`], and what it gets.
static STATS_WANTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static STATS: std::sync::Mutex<String> = std::sync::Mutex::new(String::new());

/// The frame timings of the program as it is running, as a line of JSON: empty until
/// the frame after it was first asked for (keeping it costs a little in every frame).
pub fn stats() -> String {
    STATS_WANTED.store(true, std::sync::atomic::Ordering::Relaxed);
    STATS.lock().map(|s| s.clone()).unwrap_or_default()
}

impl App {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, adapter: &wgpu::Adapter, egui_renderer: EguiRenderer) -> Self {
        let gfx = Gfx::new(device, queue, adapter, egui_renderer);
        // (with no GPU, plain drawing from the first frame; the tools that make
        // pictures ask for the full drawing, see `App::draw_fully`)
        let governor = if gfx.software() { Governor::plainly() } else { Governor::new() };
        let defs = registry();
        let current = defs.iter().position(|d| d.id == "surface").unwrap_or(0);
        let scene = (defs[current].make)(&gfx);
        let cam = Camera::new(scene.view());
        let mut settings = ViewSettings::default();
        scene.settings(&mut settings);
        App {
            gfx,
            defs,
            current,
            scene,
            cam,
            dl: DrawList::default(),
            handles: Handles::default(),
            places: Places::default(),
            settings,
            chosen: ViewSettings::default(),
            time: 0.0,
            speed: 1.0,
            paused: false,
            show_panels: true,
            narrow: false,
            open: None,
            show_help: false,
            show_glossary: false,
            show_primer: false,
            primer_of: None,
            glossary_filter: String::new(),
            scene_index: Vec::new(),
            glossary_index: Vec::new(),
            bare: false,
            dry: false,
            hint: if crate::clock::reproducible() { 0.0 } else { 9.0 },
            hint_clock: None,
            show_stats: false,
            dragging_cam: false,
            frame_ms: 16.7,
            styled: false,
            governor,
            fully: false,
            beat: 0.0,
            was_animating: false,
            filter: String::new(),
            rows: (0.0, Vec::new()),
            total_cpu: 0.0,
            scene_cpu: 0.0,
            render_cpu: 0.0,
            frames: 0,
            last: [0.0; 2],
            cpu_ms: [0.0; 3],
            visit: Visit::default(),
        }
    }

    /// Seconds the last frame spent in scene code (simulation and building the
    /// draw list), for the tools that look for slow frames.
    pub fn last_scene_seconds(&self) -> f32 {
        self.last[0] as f32
    }

    /// Identifier of the scene being shown.
    pub fn current_id(&self) -> &'static str {
        self.defs[self.current].id
    }

    /// Frames drawn on the GPU, and frames found unchanged and reused, since the last reset.
    pub fn frame_counts(&self) -> (u64, u64) {
        (self.gfx.drawn, self.gfx.reused)
    }

    pub fn scene_ids(&self) -> Vec<&'static str> {
        self.defs.iter().map(|d| d.id).collect()
    }

    /// Switch scene by id. Returns false if there is no such scene.
    pub fn select(&mut self, id: &str) -> bool {
        match self.defs.iter().position(|d| d.id == id) {
            Some(i) => {
                if i != self.current {
                    self.switch(i);
                }
                true
            }
            None => false,
        }
    }

    fn switch(&mut self, i: usize) {
        // a new scene is judged afresh, starting at full resolution
        self.governor = self.governor.afresh(self.beat);
        self.current = i;
        self.show_primer = false;
        let began = now();
        self.scene = (self.defs[i].make)(&self.gfx);
        self.visit = Visit { made: ((now() - began) * 1e3) as f32, ..Visit::default() };
        // its own look, on top of what the visitor has chosen
        self.settings = self.chosen;
        self.scene.settings(&mut self.settings);
        self.cam.set_view(self.scene.view(), true);
        self.handles = Handles::default();
        self.places = Places::default();
        self.time = 0.0;
        self.paused = false;
    }

    /// The governor as a visit begins with it: drawing plainly where no GPU is doing
    /// the drawing.
    fn governor_anew(&self) -> Governor {
        if self.gfx.software() && !self.fully {
            Governor::plainly()
        } else {
            Governor::new()
        }
    }

    /// Draw every picture as well as the device can, however long it takes: for the
    /// tools that make pictures to be looked at or compared, on machines with no GPU.
    pub fn draw_fully(&mut self) {
        self.fully = true;
        self.governor = Governor::new();
        self.gfx.set_plain(false);
    }

    /// Start the scene being shown afresh, as if it had just been opened.
    pub fn restart(&mut self) {
        self.switch(self.current);
    }

    /// Whether the scene being shown moves by itself (see [`Scene::animated`]).
    pub fn scene_animated(&self) -> bool {
        self.scene.animated()
    }

    /// Build frames without drawing them (see the `dry` field).
    pub fn set_dry(&mut self, on: bool) {
        self.dry = on;
    }

    /// On a narrow screen, open the controls over the picture (as its button does).
    pub fn show_controls(&mut self, on: bool) {
        self.open = on.then_some(Side::Controls);
    }

    /// How many things the last frame asked for at places that are not finite
    /// numbers (see [`DrawList::left_out`]). For the tools that check scenes.
    pub fn left_out(&self) -> usize {
        self.dl.left_out()
    }

    /// Nothing but the picture: no panels, and no buttons or title over it (for
    /// rendering images of a scene).
    pub fn set_bare(&mut self) {
        self.show_panels = false;
        self.bare = true;
        self.hint = 0.0;
    }

    pub fn set_panels(&mut self, on: bool) {
        self.show_panels = on;
    }

    /// Primitive counts of the last frame: (line segments, points, triangles).
    pub fn stats(&self) -> (usize, usize, usize) {
        self.dl.stats()
    }

    /// Where the last frame drew things, in pixels of the picture and in the order
    /// they were drawn in: the points of each batch of lines and of each batch of
    /// discs, spheres and glows, and then the labels (for `eidos motion`, which
    /// measures how evenly a scene moves).
    pub fn drawn_points(&self) -> Vec<Vec<Vec2>> {
        let dl = &self.dl;
        // (what is behind the viewer, or off the picture, is nowhere on it)
        let size = self.cam.viewport;
        let place = |p: [f32; 3]| self.cam.project(glam::Vec3::from(p)).filter(|at| at.cmpge(Vec2::ZERO).all() && at.cmple(size).all()).unwrap_or(Vec2::NAN);
        let lines = dl.built_in_lines().into_iter().chain(dl.extra_lines.iter().map(|(_, b)| b));
        let points = dl.built_in_points().into_iter().chain(dl.extra_points.iter().map(|(_, b)| b));
        // (a break in a line is no point of it)
        let mut batches: Vec<Vec<Vec2>> = lines.map(|b| b.pts.iter().filter(|p| p.width >= 0.0).map(|p| place(p.pos)).collect()).collect();
        batches.extend(points.map(|b| b.pts.iter().map(|p| place(p.pos)).collect()));
        batches.push(dl.labels.iter().map(|l| l.pos).collect());
        batches
    }

    /// What the last frame's labels say, in the order of the last batch of
    /// [`App::drawn_points`].
    pub fn drawn_labels(&self) -> Vec<&str> {
        self.dl.labels.iter().map(|l| l.text.as_str()).collect()
    }

    pub fn reset_timers(&mut self) {
        self.gfx.drawn = 0;
        self.gfx.reused = 0;
        self.total_cpu = 0.0;
        self.scene_cpu = 0.0;
        self.render_cpu = 0.0;
        self.frames = 0;
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let start = now();
        self.last = [0.0; 2];
        self.ui_inner(ui);
        let total = now() - start;
        self.total_cpu += total;
        let parts = [total - self.last[0] - self.last[1], self.last[0], self.last[1]];
        for (smooth, part) in self.cpu_ms.iter_mut().zip(parts) {
            *smooth += (part as f32 * 1e3 - *smooth) * 0.08;
        }
        let ms = (total * 1e3) as f32;
        let visit = &mut self.visit;
        if visit.frames == 0 {
            visit.first = ms;
        } else {
            visit.worst = visit.worst.max(ms);
        }
        if visit.frames >= 20 && self.was_animating {
            visit.gap = visit.gap.max(ui.input(|i| i.unstable_dt) * 1e3);
        }
        visit.frames += 1;
        if STATS_WANTED.load(std::sync::atomic::Ordering::Relaxed) {
            if let Ok(mut stats) = STATS.lock() {
                *stats = self.stats_json();
            }
        }
    }

    fn stats_json(&self) -> String {
        let (segments, points, triangles) = self.dl.stats();
        let v = &self.visit;
        format!(
            concat!(
                r#"{{"scene":"{}","frame_ms":{:.2},"interface":{:.3},"scene_ms":{:.3},"drawing":{:.3},"made":{:.2},"first":{:.2},"worst":{:.2},"gap":{:.1},"frames":{},"#,
                r#""scale":{:.2},"moving":{},"width":{},"height":{},"msaa":{},"hdr":{},"segments":{},"points":{},"triangles":{},"drawn":{},"reused":{},"adapter":"{}"}}"#
            ),
            self.defs[self.current].id,
            self.frame_ms,
            self.cpu_ms[0],
            self.cpu_ms[1],
            self.cpu_ms[2],
            v.made,
            v.first,
            v.worst,
            v.gap,
            v.frames,
            self.settings.render_scale * if self.was_animating { self.governor.scale } else { 1.0 },
            self.was_animating,
            self.cam.viewport.x as u32,
            self.cam.viewport.y as u32,
            self.gfx.msaa_samples(),
            self.gfx.is_hdr(),
            segments,
            points,
            triangles,
            self.gfx.drawn,
            self.gfx.reused,
            self.gfx.adapter.chars().filter(|c| c.is_ascii_alphanumeric() || " .-_()/".contains(*c)).take(60).collect::<String>(),
        )
    }

    fn ui_inner(&mut self, ui: &mut egui::Ui) {
        if !self.styled {
            apply_style(ui.ctx());
            self.styled = true;
        }
        // (what the scenes' toolkit keeps of the texts it measured lasts while they are shown)
        crate::scenes::kit::next_frame();
        crate::ui::frame_begins();
        // Letters are placed to a quarter of a pixel (each drawn up to four times over,
        // once for each part of a pixel it may start at) only where pixels are large
        // enough for that to show. On a dense display a whole pixel is as fine as a
        // quarter of one elsewhere, a letter's picture is two to four times the work
        // to make, and making them is most of what showing a text for the first time costs.
        let fine = ui.ctx().pixels_per_point() < 1.5;
        if ui.visuals().text_options.subpixel_binning != fine {
            ui.ctx().all_styles_mut(|s| s.visuals.text_options.subpixel_binning = fine);
        }
        self.shortcuts(ui);

        let width = ui.available_width();
        self.narrow = width < NARROW;
        if self.narrow {
            // one panel at a time, over most of the screen
            let size = (width - 44.0).clamp(160.0, 320.0);
            match self.open {
                Some(Side::Scenes) => {
                    egui::Panel::left("scenes").resizable(false).exact_size(size).show(ui, |ui| exactly(ui, |ui| self.scene_list(ui)));
                }
                Some(Side::Controls) => {
                    egui::Panel::right("controls").resizable(false).exact_size(size).show(ui, |ui| exactly(ui, |ui| self.controls(ui)));
                }
                None => {}
            }
        } else if self.show_panels {
            egui::Panel::left("scenes").resizable(false).exact_size(232.0).show(ui, |ui| exactly(ui, |ui| self.scene_list(ui)));
            egui::Panel::right("controls").resizable(false).exact_size(290.0).show(ui, |ui| exactly(ui, |ui| self.controls(ui)));
        }
        self.help(ui.ctx());
        self.glossary(ui.ctx());
        self.primer(ui.ctx());
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| self.viewport(ui));
    }

    /// What this is and how to use it.
    fn help(&mut self, ctx: &egui::Context) {
        if !self.show_help {
            return;
        }
        let mut open = true;
        // within the screen whatever its shape, title bar and close button included
        // (a phone held sideways is not 400 pixels tall): what does not fit scrolls
        let screen = ctx.content_rect();
        let width = (screen.width() - 16.0).min(420.0).max(120.0);
        let height = (screen.height() - 90.0).min(640.0).max(80.0);
        egui::Window::new("About eidos").open(&mut open).collapsible(false).resizable(false).default_width(width).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            ui.set_width(width);
            egui::ScrollArea::vertical().max_height(height).auto_shrink([false, true]).show(ui, |ui| {
                let body = |ui: &mut egui::Ui, text: &str| {
                    ui.label(RichText::new(text).size(13.0).color(pal::TEXT.egui()));
                };
                body(ui, "Pictures of mathematics that you can take hold of: linear algebra, calculus, dynamics, statistics, Hermite polynomials, neural networks and how to look inside them. Everything is computed and drawn live on your own machine; nothing is a recording. Each scene's panel has things to try, and the glossary (G) explains the words.");
                ui.add_space(8.0);
                ui.label(RichText::new("Moving the picture").size(12.0).strong().color(pal::DIM.egui()));
                egui::Grid::new("help keys").num_columns(2).spacing([14.0, 4.0]).show(ui, |ui| {
                    for (what, how) in [
                        ("drag", "turn a 3D picture, move a flat one"),
                        ("right-drag, shift-drag", "move"),
                        ("scroll, pinch", "zoom"),
                        ("double-click, R", "back to the starting view"),
                        ("coloured dots", "can be dragged: they are the scene's handles"),
                        ("space", "pause and resume"),
                        ("H", "hide or show the panels"),
                        ("G", "the glossary: what the words mean"),
                        ("F", "full screen (also the button with four corner marks)"),
                        ("M", "the mathematics behind the scene, written out"),
                        ("F1", "frame timings, and whether a GPU is in use"),
                        ("PageUp, PageDown", "previous and next scene"),
                    ] {
                        ui.label(RichText::new(what).size(12.5).color(pal::CYAN.egui()));
                        // (wrapped to what is left of the width: a grid does not wrap by itself)
                        ui.add(egui::Label::new(RichText::new(how).size(12.5).color(pal::TEXT.egui())).wrap());
                        ui.end_row();
                    }
                });
                ui.add_space(8.0);
                ui.label(RichText::new("Worth knowing").size(12.0).strong().color(pal::DIM.egui()));
                body(ui, "Every scene has its own address (the part after # in the address bar), so you can link straight to one.");
                body(ui, "Text boxes take formulas: sin(x)^2 + a x y, exp(-x^2/2), sqrt, abs, tanh, min(a, b), pi and so on.");
                body(ui, "Sliders also take typed numbers: click the number beside one.");
                body(ui, "On a phone: the button with four corner marks, beside \"scenes\", fills the screen. An iPhone's browser cannot do that for a page; there, use Share, then \"Add to Home Screen\", and open eidos from its icon.");
            });
        });
        self.show_help &= open;
    }

    /// Open or close the primer of the scene being shown (it opens only if one is written).
    pub fn show_primer(&mut self, on: bool) {
        self.show_primer = on && crate::primer::of(self.defs[self.current].id).is_some();
    }

    /// The mathematics behind the scene being shown, written out: its primer, over the
    /// application, to be read before the picture.
    fn primer(&mut self, ctx: &egui::Context) {
        if !self.show_primer {
            return;
        }
        let d = self.defs[self.current];
        let Some(text) = crate::primer::of(d.id) else {
            self.show_primer = false;
            return;
        };
        if self.primer_of.as_ref().is_none_or(|(id, _)| *id != d.id) {
            self.primer_of = Some((d.id, crate::primer::parse(text)));
        }
        let mut open = true;
        let mut go = None;
        let screen = ctx.content_rect();
        let width = (screen.width() - 24.0).min(640.0).max(120.0);
        let height = (screen.height() - 110.0).min(860.0).max(60.0);
        let title = format!("{}: the mathematics", crate::typeset::plain(d.title));
        egui::Window::new(RichText::new(title).size(15.0))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(width)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_width(width);
                // (its text is as wide whether or not the scroll bar is showing: see the
                // controls panel)
                let room = (width - ui.spacing().scroll.allocated_width() - 10.0).max(40.0);
                egui::ScrollArea::vertical().id_salt(("primer", d.id)).max_height(height).min_scrolled_height(height).auto_shrink([false, true]).show(
                    ui,
                    |ui| {
                        ui.horizontal(|ui| {
                            ui.add_space(6.0);
                            ui.vertical(|ui| {
                                ui.set_width(room);
                                if let Some((_, blocks)) = &self.primer_of {
                                    go = crate::primer::show(ui, blocks);
                                }
                                ui.add_space(12.0);
                            });
                        });
                    },
                );
            });
        self.show_primer &= open;
        if let Some(id) = go {
            // (a related scene, with its own primer open if it has one)
            if self.select(&id) {
                self.show_primer = crate::primer::of(&id).is_some();
            }
        }
    }

    /// The words the network scenes use, each with the scene that shows it.
    fn glossary(&mut self, ctx: &egui::Context) {
        if !self.show_glossary {
            return;
        }
        let mut open = true;
        let mut go = None;
        let height = (ctx.content_rect().height() - 150.0).min(620.0).max(60.0);
        let width = (ctx.content_rect().width() - 16.0).min(470.0).max(120.0);
        egui::Window::new("Glossary")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .default_width(width)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ctx, |ui| {
                ui.set_width(width);
                ui.add(egui::TextEdit::singleline(&mut self.glossary_filter).hint_text("find a word").desired_width(f32::INFINITY));
                let needle = squeeze(&self.glossary_filter.to_lowercase());
                if !needle.is_empty() && self.glossary_index.is_empty() {
                    self.glossary_index = crate::glossary::SECTIONS.iter().flat_map(|(_, entries)| entries.iter()).map(glossary_search_text).collect();
                }
                let index = &self.glossary_index;
                egui::ScrollArea::vertical().max_height(height).min_scrolled_height(height).auto_shrink([false, true]).show(ui, |ui| {
                    let mut n = 0;
                    for (heading, entries) in crate::glossary::SECTIONS {
                        let first = n;
                        n += entries.len();
                        let found: Vec<_> = entries
                            .iter()
                            .enumerate()
                            .filter(|(k, _)| needle.is_empty() || index.get(first + k).is_some_and(|t| t.contains(&needle)))
                            .map(|(_, e)| e)
                            .collect();
                        if found.is_empty() {
                            continue;
                        }
                        ui.add_space(8.0);
                        ui.label(RichText::new(heading.to_uppercase()).size(10.5).strong().color(pal::MUTED.egui()));
                        for entry in found {
                            ui.add_space(4.0);
                            let def = entry.scene.and_then(|id| self.defs.iter().find(|d| d.id == id));
                            let link = def.map(|d| format!("→ {}", d.title));
                            // Every entry is one paragraph, the link its last part, set by
                            // the typesetter whether it has mathematics in it or not: so
                            // that every entry's lines are spaced alike. (Laid out by egui
                            // as words beside a term, an entry's second line came lower
                            // than its third.)
                            {
                                let mut spans = vec![
                                    crate::typeset::Span::new(entry.term, 13.0, pal::CYAN.egui()),
                                    crate::typeset::Span { gap: 6.0, ..crate::typeset::Span::new(entry.meaning, 12.5, pal::TEXT.egui()) },
                                ];
                                if let Some(link) = &link {
                                    spans.push(crate::typeset::Span { gap: 6.0, ..crate::typeset::Span::new(link, 12.5, pal::LINK.egui()) });
                                }
                                if let Some(clicked) = crate::ui::linked_words(ui, &spans, link.is_some().then_some(2)) {
                                    if clicked {
                                        go = def.map(|d| d.id);
                                    }
                                    continue;
                                }
                            }
                            ui.horizontal_wrapped(|ui| {
                                ui.spacing_mut().item_spacing.x = 6.0;
                                ui.label(RichText::new(entry.term).size(13.0).strong().color(pal::CYAN.egui()));
                                ui.label(RichText::new(entry.meaning).size(12.5).color(pal::TEXT.egui()));
                                // (only scenes that exist are offered)
                                if let Some(def) = entry.scene.and_then(|id| self.defs.iter().find(|d| d.id == id)) {
                                    if ui.link(RichText::new(format!("→ {}", def.title)).size(12.5).color(pal::LINK.egui())).clicked() {
                                        go = Some(def.id);
                                    }
                                }
                            });
                        }
                    }
                    ui.add_space(6.0);
                });
            });
        self.show_glossary &= open;
        if let Some(id) = go {
            self.select(id);
            self.show_glossary = false;
            if self.narrow {
                self.open = None;
            }
        }
    }

    fn shortcuts(&mut self, ui: &mut egui::Ui) {
        // (while a text box has the focus, not merely a button or a slider)
        if ui.ctx().text_edit_focused() {
            return;
        }
        // Keys pressed on their own (or with shift) only: with Ctrl, Cmd or Alt held
        // they are the browser's or the system's (Ctrl+F finds, Cmd+R reloads), and
        // the page is told of them all the same.
        let pressed = |key: egui::Key| {
            ui.input(|i| {
                i.events.iter().any(|e| {
                    matches!(e, egui::Event::Key { key: k, pressed: true, modifiers: m, .. }
                        if *k == key && !(m.ctrl || m.alt || m.command || m.mac_cmd))
                })
            })
        };
        let (space, r, h, stats, next, prev) = (
            pressed(egui::Key::Space),
            pressed(egui::Key::R),
            pressed(egui::Key::H),
            pressed(egui::Key::F1),
            pressed(egui::Key::PageDown),
            pressed(egui::Key::PageUp),
        );
        if ui.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Text(t) if t == "?"))) {
            self.show_help = !self.show_help;
        }
        if pressed(egui::Key::G) {
            self.show_glossary = !self.show_glossary;
        }
        if pressed(egui::Key::M) {
            self.show_primer = !self.show_primer && crate::primer::of(self.defs[self.current].id).is_some();
        }
        if pressed(egui::Key::Escape) {
            self.show_primer = false;
        }
        if pressed(egui::Key::F) && fullscreen::available() {
            fullscreen::toggle(ui.ctx());
        }
        if space {
            // (time speed 0 stops the scene as a pause does, and space resumes it too)
            if self.paused || self.speed <= 0.0 {
                self.paused = false;
                if self.speed <= 0.0 {
                    self.speed = 1.0;
                }
            } else {
                self.paused = true;
            }
        }
        if r {
            self.cam.reset();
        }
        if h {
            self.show_panels = !self.show_panels;
        }
        if stats {
            self.show_stats = !self.show_stats;
        }
        if next {
            self.switch((self.current + 1) % self.defs.len());
        }
        if prev {
            self.switch((self.current + self.defs.len() - 1) % self.defs.len());
        }
    }

    fn scene_list(&mut self, ui: &mut egui::Ui) {
        ui.add_space(14.0);
        ui.horizontal(|ui| {
            ui.add_space(6.0);
            ui.label(RichText::new("eidos").size(26.0).strong().color(pal::WHITE.egui()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(8.0);
                if self.narrow && ui.button("◀ back").clicked() {
                    self.open = None;
                }
                if ui.button("?").on_hover_text("what this is and how to use it").clicked() {
                    self.show_help = !self.show_help;
                }
                fullscreen::button(ui, None);
            });
        });
        ui.horizontal(|ui| {
            ui.add_space(7.0);
            ui.label(RichText::new("mathematics you can touch").size(11.5).color(pal::DIM.egui()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(8.0);
                if ui
                    .link(RichText::new("glossary").size(11.5).color(pal::LINK.egui()))
                    .on_hover_text("the words the network scenes use, each explained (G)")
                    .clicked()
                {
                    self.show_glossary = !self.show_glossary;
                }
            });
        });
        ui.add_space(10.0);
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            ui.add(egui::TextEdit::singleline(&mut self.filter).hint_text("search").desired_width(ui.available_width() - 14.0));
        });
        ui.add_space(4.0);
        let needle = squeeze(&self.filter.to_lowercase());
        // (what is searched is what the visitor sees, not the markup: see `scene_search_text`)
        if !needle.is_empty() && self.scene_index.is_empty() {
            self.scene_index = self.defs.iter().map(scene_search_text).collect();
        }
        let index = &self.scene_index;
        let matches = |i: usize| needle.is_empty() || index.get(i).is_some_and(|t| t.contains(&needle));
        let mut pick = None;
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            let mut group = "";
            // (the width the rows are kept for is the list's own, taken before any of
            // them: a title too long for its row widens what is left for the rows after it)
            let width = ui.available_width();
            if self.rows.0 != width {
                self.rows = (width, vec![None; self.defs.len()]);
            }
            if !(0..self.defs.len()).any(matches) {
                ui.label(RichText::new("  nothing matches").size(12.5).color(pal::DIM.egui()));
            }
            for (i, d) in self.defs.iter().enumerate() {
                if !matches(i) {
                    continue;
                }
                if d.group != group {
                    group = d.group;
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        ui.add_space(7.0);
                        ui.label(RichText::new(group.to_uppercase()).size(10.5).strong().color(pal::MUTED.egui()));
                    });
                    ui.add_space(2.0);
                }
                // A row scrolled out of sight is only given its room: the list is some
                // seventy buttons long, a dozen or two of them in view, and laying all
                // of them out in every frame cost as much as a scene's whole panel.
                // (its room is what its button took when it was last laid out at this
                // width: a long title takes two lines)
                let row = egui::vec2(ui.available_width(), ROW);
                if let Some(button) = self.rows.1.get(i).copied().flatten() {
                    if !ui.is_rect_visible(egui::Rect::from_min_size(ui.cursor().min, egui::vec2(row.x, button.y.max(ROW)))) {
                        // (laid out as the button's row is, with empty room the size
                        // of the button in it, so that the list is where it would be
                        // to the last fraction of a pixel)
                        let layout = egui::Layout::centered_and_justified(ui.layout().main_dir());
                        ui.allocate_ui_with_layout(row, layout, |ui| ui.allocate_space(button));
                        continue;
                    }
                }
                let selected = i == self.current;
                let text = crate::ui::widget_text(ui, d.title, 14.0, Some(if selected { pal::WHITE } else { pal::TEXT }), false);
                let resp = crate::ui::hover(ui.add_sized(row, egui::Button::selectable(selected, text)), d.blurb);
                if let Some(kept) = self.rows.1.get_mut(i) {
                    *kept = Some(resp.rect.size()).filter(|size| size.is_finite());
                }
                if resp.clicked() {
                    if !selected {
                        pick = Some(i);
                    }
                    // on a narrow screen the list has done its job: back to the picture
                    self.open = None;
                }
            }
            ui.add_space(12.0);
        });
        if let Some(i) = pick {
            self.switch(i);
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        let d = &self.defs[self.current];
        ui.add_space(14.0);
        if self.narrow && ui.button("back ▶").clicked() {
            self.open = None;
        }
        if crate::typeset::has_math(d.title) {
            crate::ui::words(ui, d.title, 19.0, pal::WHITE);
        } else {
            ui.label(RichText::new(d.title).size(19.0).strong().color(pal::WHITE.egui()));
        }
        ui.add_space(2.0);
        crate::ui::words(ui, d.blurb, 12.5, pal::DIM);
        if crate::primer::of(d.id).is_some() {
            ui.add_space(6.0);
            let label = RichText::new("the mathematics behind this scene").size(12.5).color(pal::LINK.egui());
            if ui.add(egui::Button::new(label)).on_hover_text("A short walk through the mathematics, to read before the picture (M).").clicked() {
                self.show_primer = true;
            }
        }
        ui.add_space(8.0);
        ui.separator();
        // (one scroll position per scene: a panel left scrolled down does not open
        // the next scene in its middle)
        // The text is wrapped to a width that does not depend on whether the scroll bar
        // is showing: its room is kept free either way. egui slides the bar in over
        // half a dozen frames, the panel narrowing a pixel at a time, and every
        // paragraph of the panel was set afresh in each of them, at every opening of
        // every scene (3 to 6 ms a frame instead of half a millisecond).
        let room = (ui.available_width() - ui.spacing().scroll.allocated_width()).max(40.0);
        egui::ScrollArea::vertical().id_salt(("controls", self.current)).auto_shrink([false, false]).show(ui, |ui| {
            ui.set_width(room);
            ui.add_space(6.0);
            self.scene.ui(ui, &self.gfx);
            // (a "see also" link in the panel asks for another scene; on a narrow
            // screen the panel then makes way for it, as the scene list does)
            if let Some(id) = crate::ui::take_wanted_scene() {
                self.select(id);
                if self.narrow {
                    self.open = None;
                }
            }
            // the way on: the catalogue is written to be read in order
            if let Some(i) = self.way_on(ui) {
                self.switch(i);
                if self.narrow {
                    self.open = None;
                }
            }
            ui.add_space(14.0);
            ui.separator();
            egui::CollapsingHeader::new(RichText::new("View").color(pal::DIM.egui())).default_open(false).show(ui, |ui| {
                crate::ui::slider(ui, "time speed", &mut self.speed, 0.0..=4.0);
                // (the look in force, the scene's own included; what is changed here wins)
                let before = self.settings;
                crate::ui::slider(ui, "glow", &mut self.settings.bloom, 0.0..=1.5);
                crate::ui::slider(ui, "exposure", &mut self.settings.exposure, 0.3..=2.5);
                crate::ui::slider(ui, "render scale", &mut self.settings.render_scale, 0.5..=2.0);
                ui.checkbox(&mut self.settings.adaptive, "lower resolution if frames are slow");
                if self.settings.adaptive {
                    crate::ui::note(ui, &self.governor.status());
                }
                ui.checkbox(&mut self.settings.grid, "grid");
                self.keep_choices(before);
                ui.checkbox(&mut self.show_stats, "statistics (F1)");
            });
            ui.add_space(8.0);
            let usual = if self.cam.is_flat() { "drag: move    scroll: zoom" } else { "drag: turn    right-drag: move    scroll: zoom" };
            // (a scene's own hint may have mathematics in it, and is the scene's text)
            let own = self.scene.pointer_hint();
            if let Some(own) = own {
                crate::typeset::seen(own);
            }
            let moving = own.unwrap_or(usual);
            crate::ui::words_unnoted(ui, &format!("{moving}\nspace: pause    R: reset view    H: hide panels    ?: more"), 11.0, pal::MUTED);
        });
    }

    /// Where the scene being shown stands in its section, with links to the scene
    /// before it and the one after it in the catalogue (the sections are courses, to be
    /// read from the top). Returns the scene a link asked for.
    fn way_on(&self, ui: &mut egui::Ui) -> Option<usize> {
        let (at, before, after) = neighbours(&self.defs, self.current);
        let mut wanted = None;
        ui.add_space(12.0);
        ui.label(RichText::new(at).size(11.0).color(pal::MUTED.egui()));
        ui.horizontal_wrapped(|ui| {
            for (to, forwards) in [(before, false), (after, true)] {
                let Some(i) = to else { continue };
                let title = self.defs[i].title;
                let text = if forwards { format!("next: {title} →") } else { format!("← {title}") };
                let link = crate::ui::link_label(ui, crate::ui::widget_text(ui, &text, 12.5, Some(pal::LINK), false));
                if crate::ui::hover(link, self.defs[i].blurb).clicked() {
                    wanted = Some(i);
                }
            }
        });
        wanted
    }

    /// Whatever the visitor has changed in the look in force since `before` is their
    /// own choice: it holds for as long as they stay in this scene, and the next
    /// scene starts from it (before setting its own look on top).
    fn keep_choices(&mut self, before: ViewSettings) {
        let (now, chosen) = (self.settings, &mut self.chosen);
        if now.bloom != before.bloom {
            chosen.bloom = now.bloom;
        }
        if now.exposure != before.exposure {
            chosen.exposure = now.exposure;
        }
        if now.render_scale != before.render_scale {
            chosen.render_scale = now.render_scale;
        }
        if now.adaptive != before.adaptive {
            chosen.adaptive = now.adaptive;
        }
        if now.grid != before.grid {
            chosen.grid = now.grid;
        }
    }

    fn viewport(&mut self, ui: &mut egui::Ui) {
        let rect = ui.available_rect_before_wrap();
        let resp = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let ppp = ui.ctx().pixels_per_point().min(MAX_PIXEL_RATIO);
        let frame_dt = ui.input(|i| i.unstable_dt);
        if self.settings.adaptive {
            // frames that follow one in which nothing moved cost next to nothing to
            // draw: how soon they come is the display's own pace (it creeps up, so
            // that a change of display is noticed)
            if !self.was_animating && (0.002..0.25).contains(&frame_dt) {
                self.beat = if self.beat == 0.0 { frame_dt } else { (self.beat * 1.002).min(frame_dt) };
            }
            // only consecutive animation frames say anything about how fast we can draw
            // (frames of up to a second and a half count: a device that needs a
            // quarter of a second for each is the one with most to gain, and a single
            // long wait, a tab coming back, is told from slowness by not being repeated)
            if self.was_animating && frame_dt < 1.5 {
                self.governor.observe(frame_dt, self.beat);
            }
        } else {
            self.governor = self.governor_anew();
        }
        if self.gfx.set_plain(!self.fully && self.settings.adaptive && self.governor.plain) {
            // (the changeover itself is not a slow frame)
            self.governor.pause(0.6);
        }
        // only moving pictures are drawn at the reduced resolution: one at rest costs
        // nothing to draw once, so it is drawn sharp
        let adaptive = if self.was_animating { self.governor.scale } else { 1.0 };
        let px = ppp * self.settings.render_scale * adaptive;
        let want = [(rect.width() * px).round() as u32, (rect.height() * px).round() as u32];
        let size = self.gfx.clamp_size(want[0], want[1]);
        // the device may cap the target size; keep widths in logical pixels consistent
        let px = px * size[0] as f32 / want[0].max(1) as f32;
        let viewport = Vec2::new(size[0] as f32, size[1] as f32);

        let real_dt = ui.input(|i| i.stable_dt).clamp(0.0, 1.0 / 15.0);
        if self.hint > 0.0 {
            // counted down by the clock, not by frames: a still picture is not redrawn
            // while the hint stays as it is, only for its last second, as it fades
            let clock = ui.input(|i| i.time);
            if let Some(last) = self.hint_clock {
                self.hint -= (clock - last).max(0.0) as f32;
            }
            self.hint_clock = Some(clock);
            if resp.dragged() || resp.clicked() {
                self.hint = self.hint.min(0.6);
            }
            if self.hint > 1.0 {
                ui.ctx().request_repaint_after(std::time::Duration::from_secs_f32(self.hint - 1.0));
            } else if self.hint > 0.0 {
                ui.ctx().request_repaint();
            }
        }
        self.frame_ms += (ui.input(|i| i.unstable_dt) * 1000.0 - self.frame_ms) * 0.08;
        // (a time speed of 0 stops the scene as a pause does)
        let still = self.paused || self.speed <= 0.0;
        // Scenes step by `dt` with their own limits on a step (sub-steps, caps on what
        // is owed), so it may exceed the 1/15 s of `real_dt`: at a high time speed, and
        // on a device that shows fewer than 15 frames a second, where the scene is
        // given the frame's whole time (up to the tenth of a second egui reports), or
        // everything on it would run slow as well as in jerks: at 9 frames a second,
        // at six tenths of its speed. No step is longer than the longest there has
        // always been, the 4/15 s of the highest time speed.
        let whole = ui.input(|i| i.stable_dt).clamp(0.0, 0.1);
        let dt = if still { 0.0 } else { (whole * self.speed).min(4.0 / 15.0) };
        self.time += dt as f64;

        // pointer in target pixels
        let to_px = |p: egui::Pos2| Vec2::new(p.x - rect.min.x, p.y - rect.min.y) * px;
        let hover = resp.hover_pos().or_else(|| resp.interact_pointer_pos());
        let (pressed, down, released) = ui.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_down(), i.pointer.primary_released()));
        let over = resp.hovered() || resp.dragged() || resp.is_pointer_button_down_on();
        let mut ptr =
            Pointer { pos: hover.filter(|_| over).map(to_px), down: down && over, pressed: pressed && resp.hovered(), released, clicked: resp.clicked() };
        self.handles.begin_frame(&ptr);
        self.places.begin_frame();
        // A double-click away from the handles puts the view back where it started,
        // and its second click is for that alone: it does not reach the scene. Unless
        // the scene does something with clicks on empty canvas and has claimed them:
        // then both are its clicks, and the view stays (R still resets it).
        let reset = resp.double_clicked() && !self.handles.on_handle() && !self.handles.clicks_claimed();
        if reset {
            ptr.clicked = false;
        }

        // camera input, unless a handle has the pointer
        // A scene that captures the pointer (drawing, poking) keeps primary drags for
        // itself; the other mouse buttons still move the camera.
        let secondary = resp.dragged_by(egui::PointerButton::Secondary) || resp.dragged_by(egui::PointerButton::Middle);
        let captured = self.handles.captured();
        if resp.drag_started() {
            self.dragging_cam = !self.handles.on_handle() && (!captured || secondary);
        }
        if !resp.dragged() {
            self.dragging_cam = false;
        }
        if self.handles.dragging() {
            self.dragging_cam = false;
        }
        if self.dragging_cam {
            let d = resp.drag_delta();
            let d = Vec2::new(d.x, d.y);
            let shift = ui.input(|i| i.modifiers.shift);
            if self.cam.is_flat() || shift || (secondary && !captured) {
                self.cam.pan(d * px);
            } else if d != Vec2::ZERO {
                self.cam.orbit(d, real_dt);
            } else {
                self.cam.hold();
            }
        }
        if resp.hovered() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = (scroll * 0.0045).exp() * pinch;
            if (factor - 1.0).abs() > 1e-5 {
                self.cam.zoom(factor, hover.map(to_px));
            }
        }
        if reset {
            self.cam.reset();
        }
        if let Some(v) = self.scene.take_view() {
            self.cam.set_view(v, false);
        }
        self.cam.auto_rotate = if still || self.dragging_cam { 0.0 } else { self.scene.auto_rotate() };
        self.cam.update(real_dt, viewport);

        // build and render the frame
        self.dl.clear();
        let t0 = now();
        {
            let top_inset = if self.buttons_over_picture(rect) { 46.0 } else { 0.0 };
            let mut f = Frame {
                ctx: ui.ctx(),
                gfx: &self.gfx,
                dl: &mut self.dl,
                cam: &self.cam,
                handles: &mut self.handles,
                places: &mut self.places,
                t: self.time as f32,
                dt,
                real_dt,
                ptr,
                px,
                top_inset,
                hud: Frame::hud_planes(&self.cam, px),
            };
            self.scene.frame(&mut f);
        }
        self.dl.finish();
        let t1 = now();
        if !self.dry {
            self.gfx.render(&mut self.dl, &self.cam, &self.settings, size, px);
        }
        self.last = [t1 - t0, now() - t1];
        self.scene_cpu += self.last[0];
        self.render_cpu += self.last[1];
        self.frames += 1;

        let painter = ui.painter_at(rect);
        if let Some(id) = self.gfx.texture_id() {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            painter.image(id, rect, uv, Color32::WHITE);
        }
        for l in &self.dl.labels {
            let pos = rect.min + egui::vec2(l.pos.x, l.pos.y);
            // (a label's dark outline, and the plate under a caption, fade with it: a label
            // fading out left them behind)
            let (shadow, plate) = backing(l.color.0[3]);
            // (a label with mathematics in it is typeset; one without is drawn as it always was)
            if crate::typeset::has_math(&l.text) {
                if let Some(set) = crate::typeset::label(ui.ctx(), &l.text, l.size) {
                    let galley = set.galley.clone();
                    let at = l.align.anchor_size(pos, galley.size());
                    if l.backed {
                        painter.rect_filled(at.expand2(egui::vec2(7.0, 3.0)), 6.0, plate);
                    } else {
                        painter.add(egui::epaint::TextShape::new(at.min + egui::vec2(0.0, 1.2), galley.clone(), shadow).with_override_text_color(shadow));
                    }
                    painter.add(egui::epaint::TextShape::new(at.min, galley, l.color.egui()));
                    continue;
                }
            }
            let font = egui::FontId::proportional(l.size);
            if l.backed {
                // a dark plate, so the text stays readable over whatever is drawn there
                let galley = painter.layout_no_wrap(l.text.clone(), font, l.color.egui());
                let at = l.align.anchor_size(pos, galley.size());
                painter.rect_filled(at.expand2(egui::vec2(7.0, 3.0)), 6.0, plate);
                painter.galley(at.min, galley, Color32::WHITE);
                continue;
            }
            painter.text(pos + egui::vec2(0.0, 1.2), l.align, &l.text, font.clone(), shadow);
            painter.text(pos, l.align, &l.text, font, l.color.egui());
        }

        self.overlay(ui, rect);
        if self.handles.on_handle() {
            ui.ctx().set_cursor_icon(if self.handles.dragging() { egui::CursorIcon::Grabbing } else { egui::CursorIcon::Grab });
        }

        // redraw continuously only while something is moving
        let animating = (self.scene.animated() && !still) || self.cam.moving() || self.handles.dragging();
        if animating {
            ui.ctx().request_repaint();
            if !self.was_animating {
                // the first frames after a rest say nothing about speed
                self.governor.pause(0.3);
            }
        } else if self.was_animating && self.governor.scale < 1.0 {
            // movement has just stopped at reduced resolution: draw the picture once
            // more, sharp
            ui.ctx().request_repaint();
        } else if self.places.gliding() {
            // a label is on its way to a new place: frames until it is there, though
            // nothing else moves (and the picture under it is not drawn again)
            ui.ctx().request_repaint();
        }
        self.was_animating = animating;
    }

    /// When the panels are not beside the picture, buttons on it open them (and the
    /// scene's title is written between the buttons). With a panel open on a narrow
    /// screen only a strip of the picture may be left: nothing is put on a narrow one.
    fn buttons_over_picture(&self, rect: egui::Rect) -> bool {
        let strip = self.narrow && self.open.is_some() && rect.width() < 260.0;
        (self.narrow || !self.show_panels) && !strip && !self.bare
    }

    /// What is said over the picture about the passing of time, if it is not as usual.
    /// A time speed of 0 stops the scene as a pause does, and space resumes it too;
    /// any other speed is said as well, since it lasts from scene to scene.
    fn time_note(&self) -> Option<String> {
        if self.paused || self.speed <= 0.0 {
            Some("paused  -  space to resume".to_string())
        } else if self.speed != 1.0 {
            let speed = if self.speed < 0.1 { format!("{:.3}", self.speed) } else { format!("{:.2}", self.speed) };
            Some(format!("time speed {}  -  under View, in the controls", speed.trim_end_matches('0').trim_end_matches('.')))
        } else {
            None
        }
    }

    /// The hint over the picture, in lines no wider than `room` (as `width` measures
    /// them): the scene's own if it has one, otherwise what dragging, zooming and the
    /// handles do here. `brief` for a phone, which has no wheel nor a right button.
    fn hint_lines(&self, brief: bool, room: f32, width: impl Fn(&str) -> f32) -> String {
        let parts: Vec<&str> = match self.scene.pointer_hint() {
            // (its parts are set apart by four spaces)
            Some(own) => own.split("    ").map(str::trim).filter(|p| !p.is_empty()).collect(),
            None => {
                let flat = self.cam.is_flat();
                let mut parts = Vec::new();
                // a scene that has the pointer for itself (drawing, poking) keeps a
                // plain drag; the right button still moves the camera, turning a 3D one
                if !self.handles.captured() {
                    parts.push(if flat { "drag to move" } else { "drag to turn" });
                }
                parts.push(if brief { "pinch to zoom" } else { "scroll or pinch to zoom" });
                if !brief && (self.handles.captured() || !flat) {
                    parts.push(if flat || !self.handles.captured() { "right-drag to move" } else { "right-drag to turn" });
                }
                if self.handles.offered() {
                    parts.push("coloured dots can be dragged");
                }
                parts
            }
        };
        let gap = "   ·   ";
        let mut lines: Vec<String> = Vec::new();
        for part in parts {
            match lines.last_mut() {
                Some(line) if width(line) + width(gap) + width(part) <= room => {
                    line.push_str(gap);
                    line.push_str(part);
                }
                _ => lines.push(part.to_string()),
            }
        }
        lines.join("\n")
    }

    fn overlay(&mut self, ui: &mut egui::Ui, rect: egui::Rect) {
        let roomy = self.buttons_over_picture(rect);
        // (on a picture too narrow for the worded buttons, signs alone)
        let compact = rect.width() < 260.0;
        let size = if compact { egui::vec2(28.0, 26.0) } else { egui::vec2(78.0, 26.0) };
        if roomy {
            let left = egui::Rect::from_min_size(rect.left_top() + egui::vec2(10.0, 10.0), size);
            let right = egui::Rect::from_min_size(rect.right_top() + egui::vec2(-10.0 - size.x, 10.0), size);
            if self.narrow {
                let mut button = |at: egui::Rect, words: &str, sign: &str| {
                    let response = ui.put(at, egui::Button::new(if compact { sign } else { words }));
                    if compact {
                        response.on_hover_text(words.trim_matches([' ', '≡'])).clicked()
                    } else {
                        response.clicked()
                    }
                };
                if self.open != Some(Side::Scenes) && button(left, "≡ scenes", "≡") {
                    self.open = Some(Side::Scenes);
                }
                if self.open != Some(Side::Controls) && button(right, "controls ≡", "⚙") {
                    self.open = Some(Side::Controls);
                }
            } else if ui.put(left, egui::Button::new("≡ panels")).on_hover_text("show the panels (H)").clicked() {
                self.show_panels = true;
            }
            // full screen: one tap away where it matters most, on a phone
            fullscreen::button(ui, Some(egui::Rect::from_min_size(left.right_top() + egui::vec2(6.0, 0.0), egui::vec2(28.0, 26.0))));
        }
        let painter = ui.painter_at(rect);
        if roomy {
            let d = &self.defs[self.current];
            // between the buttons at the two ends (on the left, the full-screen
            // button too, where there is one): in the largest size that fits there,
            // and cut short if even the smallest does not
            let (from, to) = (rect.left() + 10.0 + size.x + if fullscreen::available() { 6.0 + 28.0 } else { 0.0 } + 8.0, rect.right() - 10.0 - size.x - 8.0);
            let between = (to - from).max(40.0);
            let centre = egui::pos2((from + to) * 0.5, rect.top() + 23.0);
            let sizes = [16.0, 14.5, 13.0, 12.0, 11.0];
            // (a title with mathematics in it is typeset if it fits, and cut short as plain text if not)
            let typeset = crate::typeset::has_math(d.title)
                .then(|| sizes.iter().find_map(|&size| crate::typeset::label(ui.ctx(), d.title, size).filter(|set| set.galley.size().x <= between)))
                .flatten();
            if let Some(set) = &typeset {
                let at = egui::Align2::CENTER_CENTER.anchor_size(centre, set.galley.size());
                painter.galley(at.min, set.galley.clone(), pal::WHITE.egui());
            }
            let measure = |text: &str, size: f32| painter.layout_no_wrap(text.to_string(), egui::FontId::proportional(size), pal::WHITE.egui()).size().x;
            let full = crate::typeset::plain(d.title);
            let size = sizes.into_iter().find(|&size| measure(&full, size) <= between).unwrap_or(11.0);
            let mut title = full.clone();
            while measure(&title, size) > between && title.chars().count() > 4 {
                title = title.trim_end_matches('…').chars().take(title.chars().count().saturating_sub(3)).collect::<String>().trim_end().to_string() + "…";
            }
            if typeset.is_none() {
                painter.text(centre, egui::Align2::CENTER_CENTER, title, egui::FontId::proportional(size), pal::WHITE.egui());
            }
        }
        if self.hint > 0.0 {
            // how to move the picture, for the first few seconds of a visit
            let fade = self.hint.min(1.0);
            let font = egui::FontId::proportional(13.0);
            let room = (rect.width() - 40.0).max(60.0);
            let ctx = ui.ctx().clone();
            let width = |text: &str| match crate::typeset::has_math(text).then(|| crate::typeset::measure(&ctx, text, 13.0)).flatten() {
                Some(size) => size.x,
                None => painter.layout_no_wrap(text.to_string(), font.clone(), Color32::WHITE).size().x,
            };
            let lines = self.hint_lines(rect.width() < 560.0, room, width);
            let color = pal::TEXT.egui().gamma_multiply(fade);
            let pill_color = Color32::from_rgba_unmultiplied(8, 11, 20, (220.0 * fade) as u8);
            // (a scene's own hint with mathematics in it: each line typeset, and centred)
            let typeset: Option<Vec<_>> = crate::typeset::has_math(&lines)
                .then(|| lines.split('\n').map(|line| crate::typeset::label(&ctx, line, 13.0).map(|set| set.galley.clone())).collect())
                .flatten();
            if let Some(galleys) = typeset {
                let size = galleys.iter().fold(egui::Vec2::ZERO, |s, g| egui::vec2(s.x.max(g.size().x), s.y + g.size().y));
                // (clear of a line of caption along the bottom of the picture)
                let mut y = rect.bottom() - 48.0 - size.y;
                let pill = egui::Rect::from_center_size(egui::pos2(rect.center().x, y + size.y * 0.5), size).expand2(egui::vec2(12.0, 6.0));
                painter.rect_filled(pill, 12.0, pill_color);
                for galley in galleys {
                    let h = galley.size().y;
                    painter.galley(egui::pos2(rect.center().x - galley.size().x * 0.5, y), galley, color);
                    y += h;
                }
            } else {
                let mut job = egui::text::LayoutJob::simple(lines, font.clone(), color, room);
                job.halign = egui::Align::Center;
                let galley = painter.layout_job(job);
                // (clear of a line of caption along the bottom of the picture)
                let at = rect.center_bottom() - egui::vec2(0.0, 48.0 + galley.size().y);
                let pill = galley.rect.translate(at.to_vec2()).expand2(egui::vec2(12.0, 6.0));
                painter.rect_filled(pill, 12.0, pill_color);
                painter.galley(at, galley, Color32::WHITE);
            }
        }
        // (below the row of buttons and title, when that row is shown)
        let top = if roomy { 44.0 } else { 0.0 };
        let note = self.time_note();
        if let Some(note) = &note {
            painter.text(rect.center_top() + egui::vec2(0.0, top + 14.0), egui::Align2::CENTER_TOP, note, egui::FontId::proportional(13.0), pal::YELLOW.egui());
        }
        if self.show_stats {
            let (segs, pts, tris) = self.dl.stats();
            let text = format!(
                "{:.1} ms  ({:.0} fps)\nof which on this thread:\ninterface {:.1}  scene {:.1}  drawing {:.1} ms\n{}\n{} x {}  {}x MSAA{}\nrender scale {:.0} %\n{} line segments\n{} points\n{} triangles\nframes drawn {}, reused {}\nthis scene: opened in {:.0} ms\nlongest frame since {:.1} ms\nlongest wait between frames {:.0} ms",
                self.frame_ms,
                1000.0 / self.frame_ms.max(0.01),
                self.cpu_ms[0],
                self.cpu_ms[1],
                self.cpu_ms[2],
                // (a browser's adapter name can run to sixty characters)
                self.gfx.adapter.chars().take(40).collect::<String>(),
                self.cam.viewport.x as u32,
                self.cam.viewport.y as u32,
                self.gfx.msaa_samples(),
                if self.gfx.is_hdr() { "  HDR" } else { "" },
                100.0 * self.settings.render_scale * if self.was_animating { self.governor.scale } else { 1.0 },
                segs,
                pts,
                tris,
                self.gfx.drawn,
                self.gfx.reused,
                self.visit.made + self.visit.first,
                self.visit.worst,
                self.visit.gap
            );
            // (under the line about time, if that is showing)
            let down = top + if note.is_some() { 34.0 } else { 12.0 };
            painter.text(rect.right_top() + egui::vec2(-14.0, down), egui::Align2::RIGHT_TOP, text, egui::FontId::monospace(11.5), pal::DIM.egui());
        }
    }
}

/// What the scene list's search looks in for a scene: what the list and its panel show
/// of it (title, heading and blurb, formulas as they are drawn and Greek letters by name
/// too: see `typeset::searchable`), and its address.
fn scene_search_text(d: &SceneDef) -> String {
    squeeze(&format!("{} {} {} {}", crate::typeset::searchable(d.title), crate::typeset::searchable(d.group), crate::typeset::searchable(d.blurb), d.id))
}

/// What the glossary's search looks in for an entry: its term and its meaning as shown.
fn glossary_search_text(entry: &crate::glossary::Entry) -> String {
    squeeze(&format!("{} {}", crate::typeset::searchable(entry.term), crate::typeset::searchable(entry.meaning)))
}

/// A text without its spaces, for a search: a formula is shown with spaces round its
/// relations and operators that its written-out form does not have, and "Ax = b" typed
/// into the search should find it all the same.
fn squeeze(text: &str) -> String {
    text.chars().filter(|c| !c.is_whitespace()).collect()
}

/// The dark outline drawn under a label, and the plate under a caption, for a label of
/// opacity `alpha` (out of 255): as opaque as the label is.
fn backing(alpha: u8) -> (Color32, Color32) {
    let of = |most: f32| (most * alpha as f32 / 255.0).round() as u8;
    (Color32::from_rgba_unmultiplied(5, 7, 12, of(200.0)), Color32::from_rgba_unmultiplied(7, 10, 18, of(205.0)))
}

/// Lay out a panel's contents in exactly the room the panel gives them, and take
/// exactly that much of it, whatever they are.
///
/// egui sizes a panel by what it holds: anything a fraction of a pixel wider than the
/// panel (a typeset line set a hair past its width, a long link that cannot wrap)
/// makes it wider, and since a panel on the right keeps its width by moving its left
/// edge, the picture beside it widened by as much, shifting the whole picture for as
/// long as it lasted (a frame, or for good). Here the contents are laid out in a child
/// that the panel is not told the size of: what is too wide is cut off at the panel's
/// edge instead, and the panel is told it holds exactly its own width.
fn exactly<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let room = ui.available_rect_before_wrap();
    let mut inner = ui.new_child(egui::UiBuilder::new().max_rect(room).layout(*ui.layout()));
    let out = add(&mut inner);
    let height = inner.min_rect().height();
    ui.advance_cursor_after_rect(egui::Rect::from_min_size(room.min, egui::vec2(room.width(), height)));
    out
}

/// The height of one row of the scene list.
const ROW: f32 = 26.0;

/// The sections that are not part of the course, at the end of the catalogue: the case
/// studies and the test card. A scene there is reached from the list (or by a link in
/// another scene's panel), never by reading on from a scene outside its section.
const NOT_ON_THE_WAY: [&str; 2] = [crate::scenes::CASE_STUDIES, "Extras"];

/// Where scene `i` stands in its section ("Calculus · 2 of 5"), and the scenes
/// before and after it in the catalogue, if there are any to read on to.
fn neighbours(defs: &[SceneDef], i: usize) -> (String, Option<usize>, Option<usize>) {
    let group = defs[i].group;
    let first = defs.iter().position(|d| d.group == group).unwrap_or(i);
    let count = defs.iter().filter(|d| d.group == group).count();
    // (a section's own scenes lead to each other; the way back from a section that is
    // not part of the course passes over the other such sections, to the course's end)
    let offered = |k: &usize| defs[*k].group == group || !NOT_ON_THE_WAY.contains(&defs[*k].group);
    let before = (0..i).rev().find(offered);
    let after = Some(i + 1).filter(|k| *k < defs.len()).filter(offered);
    (format!("{group} · {} of {count}", i - first + 1), before, after)
}

use crate::clock::now;

#[cfg(test)]
mod tests {
    /// The way on from a scene: the one after it in the catalogue, across headings,
    /// but never onto the test card; and each scene knows its place under its heading.
    #[test]
    fn every_scene_knows_its_neighbours() {
        let defs = registry();
        let last = defs.len() - 1;
        assert_eq!(neighbours(&defs, 0).1, None);
        assert_eq!(neighbours(&defs, 0).2, Some(1));
        assert!(neighbours(&defs, 0).0.ends_with("· 1 of 8"), "{}", neighbours(&defs, 0).0);
        // the test card is the last entry: nothing leads to it, and it leads back to
        // the end of the course, past the case studies
        assert_eq!(defs[last].group, "Extras");
        assert_eq!(neighbours(&defs, last - 1).2, None);
        let course_end = defs.iter().rposition(|d| !NOT_ON_THE_WAY.contains(&d.group)).unwrap();
        assert_eq!(neighbours(&defs, last).1, Some(course_end));
        // the case studies: reached from no other section by reading on, leading to
        // each other, and back to the course
        let cases: Vec<usize> = (0..defs.len()).filter(|i| defs[*i].group == crate::scenes::CASE_STUDIES).collect();
        assert_eq!(cases, vec![course_end + 1, course_end + 2]);
        assert_eq!(neighbours(&defs, course_end).2, None);
        assert_eq!((neighbours(&defs, cases[0]).1, neighbours(&defs, cases[0]).2), (Some(course_end), Some(cases[1])));
        assert_eq!((neighbours(&defs, cases[1]).1, neighbours(&defs, cases[1]).2), (Some(cases[0]), None));
        for i in 0..defs.len() {
            let (at, before, after) = neighbours(&defs, i);
            assert!(at.starts_with(defs[i].group) && !at.contains(" 0 of "), "{at}");
            assert!(after.is_none_or(|k| k == i + 1));
            // (back: the scene before, or the nearest before it that is not in another
            // section off the course)
            assert!(before.is_none_or(|k| k < i && (k + 1..i).all(|j| NOT_ON_THE_WAY.contains(&defs[j].group))));
        }
    }

    /// The searches of the glossary and of the scene list find what the visitor sees:
    /// "λ" and "lambda" both find an entry written with `$\lambda$`, and the markup's
    /// commands and braces find nothing.
    #[test]
    fn the_searches_find_what_is_shown() {
        let entries: Vec<&crate::glossary::Entry> = crate::glossary::SECTIONS.iter().flat_map(|(_, entries)| entries.iter()).collect();
        let index: Vec<String> = entries.iter().map(|e| glossary_search_text(e)).collect();
        let find =
            |needle: &str| entries.iter().zip(&index).filter(|(_, t)| t.contains(&squeeze(&needle.to_lowercase()))).map(|(e, _)| e.term).collect::<Vec<_>>();
        for needle in ["λ", "lambda", "eigenvalue", "Ax = λx"] {
            assert!(find(needle).contains(&"eigenvector and eigenvalue"), "{needle}: {:?}", find(needle));
        }
        for needle in ["\\", "{", "}", "$", "mathbb", "textcolor", "cdot"] {
            assert!(find(needle).is_empty(), "{needle}: {:?}", find(needle));
        }
        let defs = registry();
        let scenes = |needle: &str| defs.iter().filter(|d| scene_search_text(d).contains(&squeeze(&needle.to_lowercase()))).map(|d| d.id).collect::<Vec<_>>();
        for needle in ["ρ", "rho", "hermite_noise", "noise operator"] {
            assert!(scenes(needle).contains(&"hermite_noise"), "{needle}: {:?}", scenes(needle));
        }
        for needle in ["\\", "{", "$", "cos\\theta"] {
            assert!(scenes(needle).is_empty(), "{needle}: {:?}", scenes(needle));
        }
    }

    use super::*;
    use crate::gfx::camera::View;
    use crate::headless::{input, Stage};
    use std::cell::Cell;
    use std::rc::Rc;
    use std::sync::Mutex;
    use std::time::Duration;

    /// One drawing device at a time (the software renderer is shared).
    static GPU: Mutex<()> = Mutex::new(());

    /// The application without a window, and the number of its next frame.
    struct Bench {
        stage: Stage,
        frame: u64,
    }

    /// `None` (and the test passes, saying so) where nothing can draw.
    fn bench() -> Option<Bench> {
        match Stage::new() {
            Ok(stage) => Some(Bench { stage, frame: 0 }),
            Err(e) => {
                eprintln!("SKIPPED: nothing here can draw ({e})");
                None
            }
        }
    }

    impl Bench {
        fn app(&mut self) -> &mut App {
            &mut self.stage.app
        }

        /// `n` frames of a window of `size`, sixty to the second, the first with
        /// `events`; what the last of them asked for.
        fn run(&mut self, size: [f32; 2], n: u32, events: Vec<egui::Event>) -> egui::FullOutput {
            let mut events = Some(events);
            let mut last = None;
            for _ in 0..n {
                let out = self.stage.frame(input(size, self.frame, events.take().unwrap_or_default()));
                self.stage.done(&out.textures_delta);
                self.frame += 1;
                last = Some(out);
            }
            last.expect("at least one frame")
        }

        fn key(&mut self, size: [f32; 2], key: egui::Key, modifiers: egui::Modifiers) -> egui::FullOutput {
            self.run(size, 1, vec![egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers }])
        }

        /// Press and let go at `at`, `times` times in a row.
        fn click(&mut self, size: [f32; 2], at: [f32; 2], times: u32) {
            let pos = egui::pos2(at[0], at[1]);
            let button = |pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
            self.run(size, 1, vec![egui::Event::PointerMoved(pos)]);
            for _ in 0..times {
                self.run(size, 1, vec![button(true)]);
                self.run(size, 1, vec![button(false)]);
            }
            self.run(size, 2, Vec::new());
        }

        fn drag(&mut self, size: [f32; 2], from: [f32; 2], to: [f32; 2]) {
            let (a, b) = (egui::pos2(from[0], from[1]), egui::pos2(to[0], to[1]));
            let button = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
            self.run(size, 1, vec![egui::Event::PointerMoved(a)]);
            self.run(size, 1, vec![button(a, true)]);
            for i in 1..=10 {
                self.run(size, 1, vec![egui::Event::PointerMoved(a + (b - a) * i as f32 / 10.0)]);
            }
            self.run(size, 1, vec![button(b, false)]);
        }

        /// Show `scene` in place of the one in the catalogue.
        fn show(&mut self, scene: Probe) {
            let app = self.app();
            app.scene = Box::new(scene);
            app.cam = Camera::new(app.scene.view());
            app.handles = Handles::default();
        }
    }

    /// A scene made to order: flat or solid, with a handle or not, keeping the
    /// pointer or the clicks for itself or not, and counting the clicks it is given.
    #[derive(Default)]
    struct Probe {
        solid: bool,
        handle: bool,
        capture: bool,
        claim: bool,
        moving: bool,
        hint: Option<&'static str>,
        clicks: Rc<Cell<u32>>,
        dot: Vec2,
        /// A note written on the picture with `kit::wrapped`, a word to a line.
        note: Option<&'static str>,
        /// A label, and its opacity.
        label: Option<(&'static str, f32)>,
        /// How tall the note says it is.
        height: Rc<Cell<f32>>,
        /// A line in the panel far wider than the panel, that does not wrap.
        wide: bool,
    }

    impl Scene for Probe {
        fn frame(&mut self, f: &mut Frame) {
            if self.capture {
                f.handles.capture();
            }
            if self.claim {
                f.handles.claim_clicks();
            }
            if self.handle {
                f.drag2(0, &mut self.dot, pal::CYAN);
            }
            if f.ptr.clicked {
                self.clicks.set(self.clicks.get() + 1);
            }
            if let Some(note) = self.note {
                crate::scenes::kit::wrapped(f, Vec2::new(-1.5, 1.5), 0.01, note, 13.0, pal::TEXT);
                self.height.set(crate::scenes::kit::wrapped_height(f, 0.01, note, 13.0));
            }
            if let Some((text, alpha)) = self.label {
                f.text_px(Vec2::new(300.0, 300.0), text, 14.0, pal::YELLOW.a(alpha), egui::Align2::LEFT_TOP);
                f.caption_px(Vec2::new(300.0, 400.0), text, 14.0, pal::YELLOW.a(alpha), egui::Align2::LEFT_TOP);
            }
        }

        fn view(&self) -> View {
            if self.solid {
                View::Orbit { target: glam::Vec3::ZERO, yaw: 0.5, pitch: 0.4, dist: 8.0 }
            } else {
                View::Flat { center: Vec2::ZERO, half_height: 2.0 }
            }
        }

        fn animated(&self) -> bool {
            self.moving
        }

        fn pointer_hint(&self) -> Option<&'static str> {
            self.hint
        }

        fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
            if self.wide {
                ui.add(egui::Label::new("a line far wider than the panel, that will not wrap: it runs on and on and on").extend());
            }
        }
    }

    const DESK: [f32; 2] = [1600.0, 1000.0];
    const PHONE: [f32; 2] = [390.0, 800.0];

    /// Ctrl+F is the browser's find bar, Cmd+R its reload, Ctrl+PageDown its next
    /// tab: with Ctrl, Cmd or Alt held, a key is not also taken as one of ours.
    #[test]
    fn keys_held_with_ctrl_or_cmd_are_left_to_the_browser() {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut b) = bench() else { return };
        b.run(DESK, 3, Vec::new());
        let fullscreen = |out: &egui::FullOutput| {
            out.viewport_output.get(&egui::ViewportId::ROOT).is_some_and(|v| v.commands.iter().any(|c| matches!(c, egui::ViewportCommand::Fullscreen(_))))
        };
        let first = b.app().current;
        for held in [egui::Modifiers::CTRL, egui::Modifiers::COMMAND, egui::Modifiers::MAC_CMD, egui::Modifiers::ALT] {
            for k in [egui::Key::G, egui::Key::H, egui::Key::Space, egui::Key::PageDown] {
                b.key(DESK, k, held);
            }
            let out = b.key(DESK, egui::Key::F, held);
            let app = b.app();
            assert!(!app.show_glossary && app.show_panels && !app.paused && app.current == first && !fullscreen(&out), "with {held:?} held");
        }
        // on their own (or with shift) they are ours
        b.key(DESK, egui::Key::G, egui::Modifiers::NONE);
        b.key(DESK, egui::Key::H, egui::Modifiers::SHIFT);
        b.key(DESK, egui::Key::Space, egui::Modifiers::NONE);
        let out = b.key(DESK, egui::Key::F, egui::Modifiers::NONE);
        let app = b.app();
        assert!(app.show_glossary && !app.show_panels && app.paused && fullscreen(&out));
    }

    /// The help fits on the screen, close button and all, on a phone held upright or
    /// sideways and in a very small window.
    #[test]
    fn the_help_fits_on_any_screen() {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut b) = bench() else { return };
        for size in [DESK, PHONE, [360.0, 640.0], [800.0, 390.0], [240.0, 180.0]] {
            b.app().show_help = true;
            b.run(size, 4, Vec::new());
            // (the only window open: the help)
            let window =
                b.stage.ctx.memory(|m| m.layer_ids().filter(|l| l.order == egui::Order::Middle).find_map(|l| m.area_rect(l.id))).expect("the help is open");
            let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(size[0], size[1]));
            assert!(screen.contains_rect(window), "in a window of {size:?} the help is at {window:?}");
        }
    }

    /// The hint over the picture says what dragging does here: coloured dots only
    /// where there are some, no "drag to move" where the scene has the drag, and a
    /// scene's own words on a phone too, in lines that fit.
    #[test]
    fn the_hint_says_what_the_pointer_does_here() {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut b) = bench() else { return };
        let width = |text: &str| text.chars().count() as f32 * 7.0;
        let mut hint = |scene: Probe, brief: bool, room: f32| {
            b.show(scene);
            b.run(DESK, 2, Vec::new());
            b.app().hint_lines(brief, room, width)
        };
        assert_eq!(hint(Probe::default(), false, 900.0), "drag to move   ·   scroll or pinch to zoom");
        assert_eq!(
            hint(Probe { handle: true, ..Default::default() }, false, 900.0),
            "drag to move   ·   scroll or pinch to zoom   ·   coloured dots can be dragged"
        );
        assert_eq!(hint(Probe { solid: true, ..Default::default() }, true, 350.0), "drag to turn   ·   pinch to zoom");
        assert_eq!(hint(Probe { capture: true, ..Default::default() }, false, 900.0), "scroll or pinch to zoom   ·   right-drag to move");
        assert_eq!(hint(Probe { capture: true, solid: true, ..Default::default() }, false, 900.0), "scroll or pinch to zoom   ·   right-drag to turn");
        assert_eq!(hint(Probe { capture: true, ..Default::default() }, true, 350.0), "pinch to zoom");
        let own = hint(Probe { hint: Some("drag: make waves    right-drag: turn    scroll: zoom"), ..Default::default() }, true, 350.0);
        assert_eq!(own, "drag: make waves   ·   right-drag: turn\nscroll: zoom");
        assert!(own.lines().all(|line| width(line) <= 350.0));
    }

    /// A double-click puts the view back and is one click for the scene, not two;
    /// a scene that has claimed the clicks gets both, and the view stays.
    #[test]
    fn a_double_click_is_the_views_unless_the_scene_claims_clicks() {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut b) = bench() else { return };
        for claim in [false, true] {
            let clicks = Rc::new(Cell::new(0));
            b.show(Probe { claim, clicks: clicks.clone(), ..Default::default() });
            b.run(DESK, 2, Vec::new());
            let centre = |app: &mut App| {
                let (lo, hi) = app.cam.flat_bounds();
                (lo + hi) * 0.5
            };
            let home = centre(b.app());
            b.drag(DESK, [800.0, 500.0], [900.0, 560.0]);
            b.run(DESK, 60, Vec::new());
            let moved = centre(b.app());
            assert!((moved - home).length() > 0.1 && clicks.get() == 0, "the drag moved the view from {home} to {moved}, {} clicks", clicks.get());
            b.click(DESK, [800.0, 500.0], 2);
            b.run(DESK, 90, Vec::new());
            let after = centre(b.app());
            if claim {
                assert_eq!(clicks.get(), 2);
                assert!((after - moved).length() < 1e-3, "the view moved from {moved} to {after}");
            } else {
                assert_eq!(clicks.get(), 1);
                assert!((after - home).length() < 1e-2, "the view is at {after}, not back at {home}");
            }
        }
    }

    /// However small the window, with no panel open there are buttons to open one.
    #[test]
    fn a_small_window_has_buttons_for_the_panels() {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut b) = bench() else { return };
        let size = [250.0, 300.0];
        b.run(size, 3, Vec::new());
        assert_eq!(b.app().open, None);
        b.click(size, [24.0, 23.0], 1);
        assert_eq!(b.app().open, Some(Side::Scenes));
        b.app().open = None;
        b.click(size, [226.0, 23.0], 1);
        assert_eq!(b.app().open, Some(Side::Controls));
        // (and on the strip left beside an open panel, none)
        let strip = egui::Rect::from_min_size(egui::pos2(320.0, 0.0), egui::vec2(70.0, 800.0));
        b.run(PHONE, 2, Vec::new());
        assert!(!b.app().buttons_over_picture(strip));
    }

    /// A time speed of 0 stops the scene as a pause does: nothing is redrawn, the
    /// picture says so, and space starts it again. Any other speed than the usual
    /// one is said over the picture.
    #[test]
    fn time_speed_zero_is_a_pause() {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut b) = bench() else { return };
        b.show(Probe { moving: true, ..Default::default() });
        b.app().speed = 0.0;
        let out = b.run(DESK, 20, Vec::new());
        assert!(out.viewport_output[&egui::ViewportId::ROOT].repaint_delay > Duration::from_millis(500));
        assert_eq!(b.app().time, 0.0);
        assert_eq!(b.app().time_note().as_deref(), Some("paused  -  space to resume"));
        b.key(DESK, egui::Key::Space, egui::Modifiers::NONE);
        let out = b.run(DESK, 20, Vec::new());
        assert!(b.app().speed == 1.0 && !b.app().paused && b.app().time > 0.2);
        assert!(out.viewport_output[&egui::ViewportId::ROOT].repaint_delay.is_zero());
        assert_eq!(b.app().time_note(), None);
        b.app().speed = 2.5;
        assert_eq!(b.app().time_note().as_deref(), Some("time speed 2.5  -  under View, in the controls"));
    }

    /// Hours into a visit, the scene clock still keeps time, also slowed down (in
    /// single precision, 2.3 hours in at a tenth of the speed, it ran 17 % fast at
    /// 60 frames a second and stood still at 240).
    #[test]
    fn the_scene_clock_keeps_time_for_hours() {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut b) = bench() else { return };
        b.show(Probe { moving: true, ..Default::default() });
        b.run(DESK, 5, Vec::new());
        b.app().speed = 0.1;
        b.app().time = 2.3 * 3600.0;
        b.run(DESK, 60, Vec::new());
        let passed = b.app().time - 2.3 * 3600.0;
        assert!((passed - 0.1).abs() < 1e-5, "a second at a tenth of the speed moved the clock {passed} s");
    }

    /// The scene sets the look it starts with; what the visitor changes in the View
    /// controls wins while they stay, and is where the next scene starts from.
    #[test]
    fn the_scene_sets_the_look_and_the_visitor_has_the_last_word() {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut b) = bench() else { return };
        let change = |b: &mut Bench, to: &dyn Fn(&mut ViewSettings)| {
            let app = b.app();
            let before = app.settings;
            to(&mut app.settings);
            app.keep_choices(before);
            b.run(DESK, 3, Vec::new());
        };
        // surfaces start without the grid; ticking "grid" puts it there, and it stays
        b.app().select("surface");
        b.run(DESK, 3, Vec::new());
        assert!(!b.app().settings.grid);
        change(&mut b, &|s| s.grid = true);
        assert!(b.app().settings.grid);
        // away and back: the scene's own look again
        b.app().select("graph");
        assert!(b.app().settings.grid);
        b.app().select("surface");
        assert!(!b.app().settings.grid);
        // the attractors start with more glow; less, if the visitor wants it, and
        // the next scene starts from their choice
        b.app().select("attractor");
        assert_eq!(b.app().settings.bloom, 0.75);
        change(&mut b, &|s| s.bloom = 0.3);
        assert_eq!(b.app().settings.bloom, 0.3);
        b.app().select("graph");
        assert_eq!(b.app().settings.bloom, 0.3);
        b.app().select("attractor");
        assert_eq!(b.app().settings.bloom, 0.75);
        // a grid turned off by the visitor stays off where the scene says nothing about it
        b.app().select("graph");
        change(&mut b, &|s| s.grid = false);
        b.app().select("calc1");
        assert!(!b.app().settings.grid);
    }

    /// A note on the picture (`kit::wrapped`) has its lines `size + 3` apart, and a line
    /// that holds a formula taller than the words further by the room the formula needs
    /// (its root no longer reaching into the line above); and it is broken into lines
    /// once, not in every frame.
    #[test]
    fn a_note_on_the_picture_makes_room_for_tall_formulas() {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut b) = bench() else { return };
        let ys = |b: &mut Bench| b.app().dl.labels.iter().map(|l| l.pos.y).collect::<Vec<f32>>();
        // without formulas, as it always was
        b.show(Probe { note: Some("aaaa bbbb cccc"), ..Default::default() });
        b.run(DESK, 3, Vec::new());
        let plain = ys(&mut b);
        assert!(plain.len() == 3 && plain.windows(2).all(|w| (w[1] - w[0] - 16.0).abs() < 1e-3), "{plain:?}");
        b.show(Probe { note: Some(r"aaaa bbbb $\sqrt{x^2}$ cccc $x$ dddd"), ..Default::default() });
        b.run(DESK, 3, Vec::new());
        let at = ys(&mut b);
        let root = crate::typeset::label(&b.stage.ctx, r"$\sqrt{x^2}$", 13.0).expect("fonts in place");
        assert!(root.rise > 0.5, "a root needs room above it: {}", root.rise);
        let pitch = |a: f32, b: f32| b - a;
        assert!(at.len() == 6, "{at:?}");
        assert!((pitch(at[0], at[1]) - 16.0).abs() < 1e-3);
        // the root's line: its galley's top is `over` above where a plain line's would be
        let top = at[2] + root.over;
        assert!((pitch(at[1], top) - 16.0 - root.rise).abs() < 1e-3, "{at:?}, over {} rise {}", root.over, root.rise);
        assert!((pitch(top, at[3]) - 16.0 - root.drop).abs() < 1e-3 && (pitch(at[3], at[4]) - 16.0).abs() < 1e-3 && (pitch(at[4], at[5]) - 16.0).abs() < 1e-3);
        // kept: no frame after the first breaks it into lines again
        let wraps = crate::scenes::kit::WRAPS.with(Cell::get);
        b.run(DESK, 20, Vec::new());
        assert_eq!(crate::scenes::kit::WRAPS.with(Cell::get), wraps);
        assert_eq!(ys(&mut b), at);
        // and how tall it is, said beforehand: six lines and the root's room
        let height = Rc::new(Cell::new(0.0));
        b.show(Probe { note: Some(r"aaaa bbbb $\sqrt{x^2}$ cccc $x$ dddd"), height: height.clone(), ..Default::default() });
        b.run(DESK, 3, Vec::new());
        assert!((height.get() - (6.0 * 16.0 + root.rise + root.drop)).abs() < 1e-3, "{} tall", height.get());
    }

    /// A label's dark outline, and a caption's plate, are as opaque as the label: a label
    /// fading out takes them with it, plain or typeset.
    #[test]
    fn a_fading_label_takes_its_outline_with_it() {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut b) = bench() else { return };
        for text in ["plain", "$x^2$"] {
            for alpha in [1.0f32, 0.5, 0.0] {
                b.show(Probe { label: Some((text, alpha)), ..Default::default() });
                let out = b.run(DESK, 3, Vec::new());
                // (the outlines: dark text; the plates: dark rectangles with round corners)
                let dark = |c: Color32| c.r() < 16 && c.g() < 16 && c.b() < 24;
                let (mut outlines, mut plates) = (Vec::new(), Vec::new());
                for clipped in &out.shapes {
                    match &clipped.shape {
                        egui::Shape::Text(t) if dark(t.fallback_color) => outlines.push(t.fallback_color.a()),
                        egui::Shape::Rect(r) if dark(r.fill) && r.corner_radius.nw == 6 => plates.push(r.fill.a()),
                        _ => {}
                    }
                }
                let expect = |most: f32| (most * alpha).round() as u8;
                assert!(outlines.contains(&expect(200.0)) && outlines.iter().all(|a| *a == expect(200.0)), "{text} at {alpha}: outlines {outlines:?}");
                assert!(plates.contains(&expect(205.0)), "{text} at {alpha}: plates {plates:?}");
            }
        }
    }

    /// The side panels are exactly as wide as they are set to be, whatever is in them:
    /// a line in the controls wider than the panel, which egui gave to the picture beside
    /// it (widening the picture, and shifting all of it), is cut off at the panel's edge.
    #[test]
    fn the_panels_keep_their_width() {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut b) = bench() else { return };
        for (size, controls, width) in [(DESK, false, DESK[0] - 232.0 - 290.0), (PHONE, true, PHONE[0] - 320.0)] {
            b.app().show_controls(controls);
            b.show(Probe { wide: true, ..Default::default() });
            for frame in 0..6 {
                b.run(size, 1, Vec::new());
                assert_eq!(b.app().cam.viewport.x, width, "in a window of {size:?}, frame {frame}");
            }
        }
    }

    /// The same in every frame of every scene as it opens (a link too long for the
    /// controls once widened that panel a few frames after a scene opened, and the
    /// picture was a pixel wider for one frame early in a run). Slow: every scene.
    #[test]
    #[ignore]
    fn the_panels_keep_their_width_in_every_scene() {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut b) = bench() else { return };
        b.app().set_dry(true);
        let mut wrong = Vec::new();
        for id in b.app().scene_ids() {
            b.app().select(id);
            for frame in 0..24 {
                b.run(DESK, 1, Vec::new());
                let width = b.app().cam.viewport.x;
                if width != DESK[0] - 232.0 - 290.0 {
                    wrong.push(format!("{id} frame {frame}: the picture {width} wide"));
                }
            }
        }
        // and on a phone, with the controls open over most of the screen
        b.app().show_controls(true);
        for id in b.app().scene_ids() {
            b.app().select(id);
            for frame in 0..12 {
                b.run(PHONE, 1, Vec::new());
                let width = b.app().cam.viewport.x;
                if width != PHONE[0] - 320.0 {
                    wrong.push(format!("{id} on a phone, frame {frame}: the picture {width} wide"));
                }
            }
        }
        assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    }

    /// While the hint is up over a still picture, the picture is not redrawn for it
    /// until the hint's last second, when it fades.
    #[test]
    fn a_still_picture_is_not_redrawn_for_the_hint() {
        let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
        let Some(mut b) = bench() else { return };
        b.show(Probe::default());
        let delay = |out: &egui::FullOutput| out.viewport_output[&egui::ViewportId::ROOT].repaint_delay;
        let out = b.run(DESK, 30, Vec::new());
        assert!(b.app().hint > 8.0 && delay(&out) > Duration::from_secs(7), "hint {} delay {:?}", b.app().hint, delay(&out));
        // the next frame comes when it is asked for, eight seconds on
        b.frame = 8 * 60 + 18;
        let out = b.run(DESK, 1, Vec::new());
        assert!(b.app().hint < 1.0 && b.app().hint > 0.0 && delay(&out).is_zero(), "hint {}", b.app().hint);
        // and once it is gone, nothing more
        b.frame = 10 * 60;
        let out = b.run(DESK, 2, Vec::new());
        assert!(b.app().hint <= 0.0 && delay(&out) > Duration::from_secs(1), "hint {} delay {:?}", b.app().hint, delay(&out));
    }
}

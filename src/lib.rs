//! eidos: interactive, GPU-rendered explorations of mathematics.

pub mod app;
pub mod background;
pub mod clock;
pub mod frame;
pub mod gfx;
pub mod glossary;
#[cfg(not(target_arch = "wasm32"))]
pub mod headless;
pub mod math;
pub mod primer;
pub mod scenes;
pub mod typeset;
pub mod ui;

/// Adapter between the application and eframe (window / browser canvas).
pub struct EframeApp {
    app: app::App,
    /// The scene the address bar currently names (browser only).
    linked: String,
}

impl EframeApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Result<Self, String> {
        let rs = cc.wgpu_render_state.as_ref().ok_or_else(|| "eidos needs the wgpu renderer (WebGPU or WebGL2)".to_string())?;
        // (the fonts in place before the first frame, so that its formulas are typeset:
        // in the first frame itself they would be shown as written)
        app::apply_style(&cc.egui_ctx);
        // (a visitor is watching: text below what is in sight may be set a frame late)
        ui::defer_unseen_text(true);
        let app = app::App::new(rs.device.clone(), rs.queue.clone(), &rs.adapter, rs.renderer.clone());
        Ok(EframeApp { app, linked: String::new() })
    }

    /// Start on a particular scene (`eidos --scene <id>`).
    pub fn with_scene(mut self, id: &str) -> Self {
        self.app.select(id);
        self
    }

    /// Keep the page address and the current scene in step: `.../#hermite_orth` opens
    /// that scene, and switching scenes updates the address, so links can be shared
    /// and Back and Forward walk through the scenes visited.
    #[cfg(target_arch = "wasm32")]
    fn sync_link(&mut self) {
        let Some(location) = web_sys::window().map(|w| w.location()) else { return };
        let hash = location.hash().unwrap_or_default();
        let named = hash.trim_start_matches('#');
        if named != self.linked {
            // the address changed (first load, a link, or back / forward)
            self.linked = named.to_string();
            if !named.is_empty() {
                self.app.select(named);
            }
        }
        let current = self.app.current_id();
        if current != self.linked {
            // An address that named no scene (the bare address, a mistyped one) is
            // replaced, not added to: otherwise Back would lead to it, and from it
            // straight back here, and the visitor could never leave.
            let named_a_scene = self.app.scene_ids().contains(&self.linked.as_str());
            self.linked = current.to_string();
            let _ = if named_a_scene { location.set_hash(current) } else { location.replace(&format!("#{current}")) };
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn sync_link(&mut self) {
        let _ = &self.linked;
    }
}

impl eframe::App for EframeApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.sync_link();
        self.app.ui(ui);
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        [0.03, 0.035, 0.05, 1.0]
    }
}

#[cfg(target_arch = "wasm32")]
mod web {
    use wasm_bindgen::prelude::*;

    /// Training speed in this build, as text (called by `tools/wasm-bench.mjs`).
    #[wasm_bindgen]
    pub fn bench_training() -> String {
        crate::scenes::training_benchmark()
    }

    /// The frame timings of the running program, as JSON (called by `tools/webshot.mjs
    /// --stats`; empty the first time it is asked).
    #[wasm_bindgen]
    pub fn stats() -> String {
        crate::app::stats()
    }

    #[wasm_bindgen(start)]
    pub fn start() {
        // In a Web Worker there is no page to draw on: the same program is loaded
        // there to do background work, and `worker_start` takes over.
        if web_sys::window().is_none() {
            return;
        }
        eframe::WebLogger::init(log::LevelFilter::Info).ok();
        wasm_bindgen_futures::spawn_local(async {
            let document = web_sys::window().expect("no window").document().expect("no document");
            let canvas = document
                .get_element_by_id("eidos")
                .expect("missing <canvas id=\"eidos\">")
                .dyn_into::<web_sys::HtmlCanvasElement>()
                .expect("#eidos is not a canvas");
            let result = eframe::WebRunner::new()
                .start(
                    canvas,
                    eframe::WebOptions::default(),
                    Box::new(|cc| match crate::EframeApp::new(cc) {
                        Ok(app) => Ok(Box::new(app) as Box<dyn eframe::App>),
                        Err(e) => Err(e.into()),
                    }),
                )
                .await;
            if let Some(el) = document.get_element_by_id("loading") {
                match result {
                    Ok(()) => el.remove(),
                    Err(e) => el.set_inner_html(&format!("<p>eidos could not start.</p><p>{e:?}</p><p>It needs a browser with WebGPU or WebGL2.</p>")),
                }
            }
        });
    }
}

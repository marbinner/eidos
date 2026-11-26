//! Full screen. In the browser this is the page's Fullscreen API (by the prefixed
//! names too, which older Safari has); natively it is the window's own mode. A
//! phone's browser gives the picture the whole screen this way; an iPhone's has no
//! such mode for pages, and there `available` is false (the page can instead be
//! added to the home screen, from where it opens without the browser around it).

#[cfg(target_arch = "wasm32")]
mod imp {
    use wasm_bindgen::{JsCast, JsValue};

    fn document() -> Option<JsValue> {
        web_sys::window().and_then(|w| w.document()).map(JsValue::from)
    }

    /// The first of `names` that `target` has a value for.
    fn get(target: &JsValue, names: &[&str]) -> Option<JsValue> {
        names.iter().filter_map(|n| js_sys::Reflect::get(target, &JsValue::from_str(n)).ok()).find(|v| !v.is_undefined() && !v.is_null())
    }

    fn call(target: &JsValue, names: &[&str]) {
        if let Some(f) = get(target, names).and_then(|f| f.dyn_into::<js_sys::Function>().ok()) {
            let _ = f.call0(target);
        }
    }

    pub fn available() -> bool {
        document().and_then(|d| get(&d, &["fullscreenEnabled", "webkitFullscreenEnabled"])).is_some_and(|v| v.is_truthy())
    }

    pub fn active(_: &egui::Context) -> bool {
        document().and_then(|d| get(&d, &["fullscreenElement", "webkitFullscreenElement"])).is_some()
    }

    /// (Browsers allow this only in answer to a tap, a click or a key: it is
    /// called in the frame that sees one, which is soon enough.)
    pub fn toggle(ctx: &egui::Context) {
        let Some(d) = document() else { return };
        if active(ctx) {
            call(&d, &["exitFullscreen", "webkitExitFullscreen"]);
        } else if let Some(page) = get(&d, &["documentElement"]) {
            call(&page, &["requestFullscreen", "webkitRequestFullscreen"]);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    pub fn available() -> bool {
        true
    }

    pub fn active(ctx: &egui::Context) -> bool {
        ctx.input(|i| i.viewport().fullscreen.unwrap_or(false))
    }

    pub fn toggle(ctx: &egui::Context) {
        ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(!active(ctx)));
    }
}

pub use imp::{active, available, toggle};

/// The button: four corner marks, pointing outwards to enter full screen and
/// inwards to leave it (drawn, since the fonts have no such sign). Give it a
/// place with `at` to put it over the picture; otherwise it goes in the layout.
pub fn button(ui: &mut egui::Ui, at: Option<egui::Rect>) {
    if !available() {
        return;
    }
    let on = active(ui.ctx());
    let button = egui::Button::new("").min_size(egui::vec2(28.0, 26.0));
    let response = match at {
        Some(rect) => ui.put(rect, button),
        None => ui.add(button),
    };
    let response = response.on_hover_text(if on { "leave full screen (F)" } else { "full screen (F)" });
    let stroke = egui::Stroke::new(1.6, ui.style().interact(&response).fg_stroke.color);
    let (marks, arm) = (egui::Rect::from_center_size(response.rect.center(), egui::vec2(13.0, 11.0)), 4.0);
    for (corner, sx, sy) in [(marks.left_top(), 1.0, 1.0), (marks.right_top(), -1.0, 1.0), (marks.right_bottom(), -1.0, -1.0), (marks.left_bottom(), 1.0, -1.0)]
    {
        // outwards: the angle sits in the corner; inwards: it sits a little way in, opening towards the corner
        let (elbow, dx, dy) = if on { (corner + egui::vec2(sx * arm, sy * arm), -sx, -sy) } else { (corner, sx, sy) };
        ui.painter().line_segment([elbow, elbow + egui::vec2(dx * arm, 0.0)], stroke);
        ui.painter().line_segment([elbow, elbow + egui::vec2(0.0, dy * arm)], stroke);
    }
    if response.clicked() {
        toggle(ui.ctx());
    }
}

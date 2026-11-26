//! General-purpose plotting: any curves, any parametric surface.

use crate::scenes::prelude::*;

// The scenes of this topic, in the order they appear in the catalogue.
crate::scenes::topic!(graph, parametric);

/// A checkbox whose label may have mathematics in it (egui's own sets its label as
/// plain text).
fn checkbox(ui: &mut egui::Ui, on: &mut bool, label: &str) -> egui::Response {
    crate::typeset::seen(label);
    let text = crate::ui::label_text(ui, label, 12.5, None, false);
    ui.checkbox(on, text)
}

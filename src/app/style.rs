//! How the interface looks: fonts, colours, spacing.

use crate::gfx::color::pal;
use egui::Color32;

pub fn apply_style(ctx: &egui::Context) {
    // the monospace face covers many mathematical symbols the text face lacks
    let mut fonts = egui::FontDefinitions::default();
    fonts.families.entry(egui::FontFamily::Proportional).or_default().push("Hack".to_owned());
    // and a last resort for what both lack: superscript and subscript letters,
    // blackboard letters, ticks and the like (see assets/README.md)
    fonts.font_data.insert("symbols".to_owned(), std::sync::Arc::new(egui::FontData::from_static(include_bytes!("../../assets/symbols.ttf"))));
    for family in [egui::FontFamily::Proportional, egui::FontFamily::Monospace] {
        fonts.families.entry(family).or_default().push("symbols".to_owned());
    }
    // the fonts formulas are set in, each a family of its own (see `typeset`)
    for face in crate::typeset::Face::ALL {
        fonts.font_data.insert(face.family().to_owned(), std::sync::Arc::new(egui::FontData::from_static(face.data())));
        fonts.families.insert(egui::FontFamily::Name(face.family().into()), vec![face.family().to_owned()]);
    }
    ctx.set_fonts(fonts);
    let mut v = egui::Visuals::dark();
    let panel = Color32::from_rgb(13, 16, 24);
    v.panel_fill = panel;
    v.window_fill = Color32::from_rgb(17, 21, 31);
    v.extreme_bg_color = Color32::from_rgb(8, 10, 16);
    v.faint_bg_color = Color32::from_rgb(20, 24, 35);
    v.override_text_color = Some(pal::TEXT.egui());
    // links (to other scenes, to the glossary) in a blue of their own
    v.hyperlink_color = Color32::from_rgb(130, 178, 255);
    v.selection.bg_fill = Color32::from_rgb(38, 78, 150);
    v.selection.stroke = egui::Stroke::new(1.0, pal::WHITE.egui());
    v.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, Color32::from_rgb(30, 36, 52));
    v.widgets.inactive.bg_fill = Color32::from_rgb(26, 32, 47);
    v.widgets.inactive.weak_bg_fill = Color32::from_rgb(22, 27, 40);
    v.widgets.hovered.bg_fill = Color32::from_rgb(40, 50, 74);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(32, 40, 60);
    v.widgets.active.bg_fill = Color32::from_rgb(52, 92, 170);
    v.widgets.active.weak_bg_fill = Color32::from_rgb(44, 78, 146);
    v.slider_trailing_fill = true;
    // Always these colours: left to follow the system, the interface would switch to
    // egui's light look when the visitor's system turns light, and the text coloured
    // here for a dark background would be unreadable on it.
    ctx.set_theme(egui::ThemePreference::Dark);
    ctx.set_visuals_of(egui::Theme::Dark, v);
    ctx.all_styles_mut(|s| {
        s.spacing.item_spacing = egui::vec2(8.0, 6.0);
        s.spacing.slider_width = 128.0;
        s.spacing.button_padding = egui::vec2(8.0, 4.0);
        s.interaction.selectable_labels = false;
        // scroll bars that take their own space, so they never cover right-aligned numbers
        s.spacing.scroll = egui::style::ScrollStyle::solid();
        s.spacing.scroll.bar_width = 7.0;
    });
}

#[cfg(test)]
mod tests {
    /// A visitor whose system turns from dark to light (or starts light) still sees
    /// the interface in its own colours.
    #[test]
    fn the_look_does_not_follow_the_system_into_light() {
        let ctx = egui::Context::default();
        let panel = |ctx: &egui::Context| ctx.global_style().visuals.panel_fill;
        let dark = egui::RawInput { system_theme: Some(egui::Theme::Dark), ..Default::default() };
        let _ = ctx.run_ui(dark.clone(), |ui| super::apply_style(ui.ctx()));
        let ours = panel(&ctx);
        assert_ne!(ours, egui::Visuals::light().panel_fill);
        for theme in [egui::Theme::Light, egui::Theme::Dark, egui::Theme::Light] {
            let _ = ctx.run_ui(egui::RawInput { system_theme: Some(theme), ..Default::default() }, |_| {});
            assert_eq!(panel(&ctx), ours, "after the system turned {theme:?}");
        }
        // and one that is light from the start
        let ctx = egui::Context::default();
        let light = egui::RawInput { system_theme: Some(egui::Theme::Light), ..Default::default() };
        let _ = ctx.run_ui(light.clone(), |ui| super::apply_style(ui.ctx()));
        let _ = ctx.run_ui(light, |_| {});
        assert_eq!(panel(&ctx), ours);
    }
}

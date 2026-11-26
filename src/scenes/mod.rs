//! Scenes: self-contained interactive explorations.
//!
//! Layout: one directory per topic, one file per scene. A scene file holds the scene's
//! state, its controls, its per-frame update and drawing, and a `DEF` constant that
//! describes it to the catalogue. A topic's `mod.rs` lists its scenes with `topic!`
//! and holds whatever those scenes share.
//!
//! To add a scene: create `<topic>/<name>.rs` (start from any existing one), give it a
//! `DEF`, and add `<name>` to the `topic!` list in `<topic>/mod.rs`. To add a topic:
//! create the directory with a `mod.rs`, and add it to `TOPICS` below.

use crate::frame::Frame;
use crate::gfx::camera::View;
use crate::gfx::{Gfx, ViewSettings};

pub mod kit;
pub mod prelude;

pub trait Scene {
    /// Controls for the side panel.
    fn ui(&mut self, _ui: &mut egui::Ui, _gfx: &Gfx) {}

    /// Advance the simulation and draw one frame.
    fn frame(&mut self, f: &mut Frame);

    /// The camera the scene starts with.
    fn view(&self) -> View;

    /// The look the scene starts with (grid on or off, bloom, ...), set when it is
    /// opened (not every frame) on top of what the visitor has chosen in the View
    /// controls so far. Whatever the visitor then changes there wins for as long as
    /// they stay in the scene, and is what the next scene starts from, before it sets
    /// its own look on top in turn.
    fn settings(&self, _s: &mut ViewSettings) {}

    /// False for scenes that only change in response to input; they are then
    /// redrawn on demand instead of every display refresh.
    fn animated(&self) -> bool {
        true
    }

    /// Radians per second of slow camera rotation (3D scenes that want it).
    fn auto_rotate(&self) -> f32 {
        0.0
    }

    /// How the pointer moves the picture here, if not in the usual way (a scene that
    /// uses plain dragging or clicking for something of its own). Shown in place of
    /// the usual hint: at the foot of the controls, and over the picture for the first
    /// seconds of a visit, on a phone too, where it is wrapped to the width (parts set
    /// apart by four spaces are kept whole on a line).
    fn pointer_hint(&self) -> Option<&'static str> {
        None
    }

    /// A new default view the scene wants the camera to glide to (after a change of
    /// scale, for instance). Polled every frame.
    fn take_view(&mut self) -> Option<View> {
        None
    }
}

/// A scene's entry in the catalogue. Scenes are constructed only when first shown.
#[derive(Clone, Copy)]
pub struct SceneDef {
    /// Stable identifier (command line, links).
    pub id: &'static str,
    pub title: &'static str,
    /// Heading it is listed under.
    pub group: &'static str,
    pub blurb: &'static str,
    pub make: fn(&Gfx) -> Box<dyn Scene>,
}

/// Declares a topic's scene modules and its `scenes()` list in one place.
macro_rules! topic {
    ($($scene:ident),* $(,)?) => {
        $(mod $scene;)*

        /// This topic's scenes, in catalogue order.
        pub fn scenes() -> Vec<SceneDef> {
            vec![$($scene::DEF),*]
        }
    };
}
pub(crate) use topic;

mod backprop;
mod calculus;
mod causal;
mod dynamics;
mod extras;
mod hermite;
mod interp;
mod lens;
mod linalg;
mod neural;
mod plots;
mod stats;

/// Every topic, in the order the catalogue shows them.
const TOPICS: &[fn() -> Vec<SceneDef>] = &[
    linalg::scenes,
    calculus::scenes,
    dynamics::scenes,
    stats::scenes,
    causal::scenes,
    hermite::scenes,
    neural::scenes,
    backprop::scenes,
    neural::opening_interpretability,
    interp::scenes,
    lens::scenes,
    plots::scenes,
    causal::case_studies,
    extras::scenes,
];

/// The group of the scenes about particular studies: what one body of published work
/// can and cannot show, not a piece of mathematics. Listed near the end of the
/// catalogue, off the way that leads from each scene of the course to the next.
pub const CASE_STUDIES: &str = "Case studies";

/// The group of a scene that is planned and not yet written: its file holds its name
/// and a stand-in. Such a scene is in none of what a visitor sees (the catalogue, the
/// search, the way on from one scene to the next, "see also" links), and none of what
/// the tools go through; it is known only so that a finished scene may link to it
/// before it exists. It joins the catalogue when its file is written and names the
/// group it belongs to.
pub const UNWRITTEN: &str = "(being written)";

/// The stand-in for a scene that is planned and not yet written (see [`UNWRITTEN`]).
pub struct Unwritten;

impl Scene for Unwritten {
    fn view(&self) -> View {
        View::Flat { center: glam::Vec2::ZERO, half_height: 3.0 }
    }

    fn animated(&self) -> bool {
        false
    }

    fn frame(&mut self, _: &mut Frame) {}
}

/// Every scene there is, the planned ones too.
fn everything() -> Vec<SceneDef> {
    TOPICS.iter().flat_map(|scenes| scenes()).collect()
}

/// The catalogue: every written scene, in order.
pub fn registry() -> Vec<SceneDef> {
    everything().into_iter().filter(|d| d.group != UNWRITTEN).collect()
}

/// Whether `id` is a scene of the catalogue.
pub fn written(id: &str) -> bool {
    static IDS: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    IDS.get_or_init(|| registry().iter().map(|d| d.id).collect()).contains(&id)
}

/// Whether `id` is a scene of the catalogue or one that is planned.
pub fn planned(id: &str) -> bool {
    static IDS: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
    IDS.get_or_init(|| everything().iter().map(|d| d.id).collect()).contains(&id)
}

/// Forget what earlier scenes of this run have left in the process, so that the next
/// one opened starts as the first would (for the scene check).
pub fn start_afresh() {
    neural::start_afresh();
}

/// How fast networks train in this build (see `neural/session.rs`).
pub fn training_benchmark() -> String {
    neural::training_benchmark() + &interp::training_benchmark()
}

/// The services that run in the background (see [`crate::background`]). A service's
/// position in this list is its address, so add new ones at the end.
pub fn services() -> Vec<Box<dyn crate::background::Service>> {
    vec![neural::trainer()]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What the catalogue promises about every entry, whatever the scene: an id that
    /// can stand in a page address and names one scene only, words to list it by,
    /// and a place among its group's other scenes. (What a scene must do when it
    /// runs is checked in `tests/scenes.rs`.)
    /// The headings of the catalogue's sections, in order: what a scene's `group` may be.
    /// (A heading mistyped in one scene would otherwise pass for a section of its own.)
    const HEADINGS: [&str; 22] = [
        "Linear algebra: vectors and maps",
        "Linear algebra: equations",
        "Linear algebra: eigenvalues",
        "Linear algebra: data and dimensions",
        linalg::SPECIAL,
        linalg::FACTOR,
        linalg::EXPONENTIAL,
        "Calculus",
        "Dynamics",
        stats::CHANCE,
        stats::ESTIMATION,
        stats::TESTING,
        stats::MODELS,
        causal::GROUP,
        "Hermite polynomials",
        "Neural networks",
        backprop::GROUP,
        interp::GROUP,
        lens::GROUP,
        "Plotting",
        CASE_STUDIES,
        "Extras",
    ];

    #[test]
    fn the_catalogue_is_well_formed() {
        let all = registry();
        assert!(all.len() >= 39);
        for (i, d) in all.iter().enumerate() {
            assert!(
                !d.id.is_empty() && d.id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_'),
                "id {:?} cannot stand in an address",
                d.id
            );
            assert!(all[..i].iter().all(|other| other.id != d.id), "two scenes are called {:?}", d.id);
            assert!(all[..i].iter().all(|other| other.title != d.title), "two scenes are titled {:?}", d.title);
            assert!(!d.title.trim().is_empty() && !d.group.trim().is_empty(), "{} has no title or no group", d.id);
            assert!(d.blurb.trim().len() >= 20 && d.blurb.trim_end().ends_with(['.', '?']), "{}: the blurb should be a sentence: {:?}", d.id, d.blurb);
            // a group's scenes are listed together: its heading appears once
            let earlier = all[..i].iter().rposition(|other| other.group == d.group);
            assert!(earlier.is_none_or(|e| e + 1 == i), "{}: its group, {:?}, is split in two in the list", d.id, d.group);
        }
        // the sections are known ones, in their order (`backprop::GROUP` is not there
        // until the first of its scenes is written)
        let mut seen: Vec<&str> = all.iter().map(|d| d.group).collect();
        seen.dedup();
        let known: Vec<&str> = HEADINGS.iter().copied().filter(|h| seen.contains(h) || *h != backprop::GROUP).collect();
        assert_eq!(seen, known, "the sections of the catalogue");
        // every planned scene has a name of its own too, and a written one links nowhere else
        let every = everything();
        for (i, d) in every.iter().enumerate() {
            assert!(every[..i].iter().all(|other| other.id != d.id && other.title != d.title), "two scenes are called {:?}", d.id);
            assert_eq!(written(d.id), d.group != UNWRITTEN);
            assert!(planned(d.id));
        }
        assert!(!planned("no such scene"));
        // the scene the application opens on
        assert!(all.iter().any(|d| d.id == "surface"));
    }
}

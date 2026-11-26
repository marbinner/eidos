//! Network playground: watch a network learn, neuron by neuron, on a choice of tasks.

use super::plane::Plane;
use super::table::Table;
use super::*;

pub const DEF: SceneDef = SceneDef {
    id: "mlp",
    title: "Network playground",
    group: "Neural networks",
    blurb: "Watch a network learn, neuron by neuron.",
    make: |g| Box::new(Playground::open(g, 0, false)),
};

/// The arithmetic-table task as a scene of its own (it is listed under
/// interpretability as the grokking scene).
pub fn arithmetic(gfx: &Gfx) -> Box<dyn Scene> {
    Box::new(Playground::open(gfx, 1, true))
}

const TASKS: [&str; 2] = ["points in the plane", "arithmetic tables"];

/// Each task keeps its own network and settings, so switching between them loses
/// nothing; and each is made only when it is first shown, so a task that is never
/// opened costs nothing (no network, no training job).
struct Playground {
    task: usize,
    plane: Option<Plane>,
    table: Option<Table>,
    new_view: bool,
    /// Opened as one task only: no switch between tasks is offered.
    solo: bool,
    /// Whether the view is phone-sized (not known until the first frame).
    narrow: Option<bool>,
}

impl Playground {
    fn open(gfx: &Gfx, task: usize, solo: bool) -> Self {
        let mut playground = Playground { task, plane: None, table: None, new_view: false, solo, narrow: None };
        playground.make_ready(gfx);
        playground
    }

    /// Make the task on show, if this is the first time it is shown.
    fn make_ready(&mut self, gfx: &Gfx) {
        match self.task {
            0 if self.plane.is_none() => self.plane = Some(Plane::new(gfx)),
            1 if self.table.is_none() => self.table = Some(Table::new(gfx)),
            _ => {}
        }
    }
}

impl Scene for Playground {
    fn view(&self) -> View {
        // on a phone: start on the main picture, the rest is a drag away
        match (self.task, self.narrow == Some(true)) {
            (0, false) => super::plane::VIEW,
            (0, true) => View::Fit { center: Vec2::new(-1.75, 0.15), half: Vec2::new(2.3, 2.75) },
            (_, false) => super::table::VIEW,
            (_, true) => View::Fit { center: Vec2::new(-2.22, 0.4), half: Vec2::new(1.75, 2.2) },
        }
    }

    /// Each task is laid out for a view of its own.
    fn take_view(&mut self) -> Option<View> {
        std::mem::take(&mut self.new_view).then(|| self.view())
    }

    fn settings(&self, s: &mut ViewSettings) {
        s.grid = false;
    }

    /// While training is paused nothing moves, so nothing needs redrawing.
    fn animated(&self) -> bool {
        match self.task {
            0 => self.plane.as_ref().is_some_and(Plane::running),
            _ => self.table.as_ref().is_some_and(Table::running),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, gfx: &Gfx) {
        if !self.solo {
            self.new_view |= ui::choice(ui, &TASKS, &mut self.task);
            ui.add_space(4.0);
        }
        self.make_ready(gfx);
        match (self.task, &mut self.plane, &mut self.table) {
            (0, Some(plane), _) => plane.ui(ui),
            (1, _, Some(table)) => table.ui(ui),
            _ => {}
        }
    }

    fn frame(&mut self, f: &mut Frame) {
        let narrow = narrow_view(f);
        if self.narrow != Some(narrow) {
            self.new_view |= self.narrow.is_some() || narrow;
            self.narrow = Some(narrow);
        }
        self.make_ready(f.gfx);
        match (self.task, &mut self.plane, &mut self.table) {
            (0, Some(plane), _) => plane.frame(f),
            (1, _, Some(table)) => table.frame(f),
            _ => {}
        }
    }
}

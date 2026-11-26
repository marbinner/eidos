//! What nearly every scene needs, in one import: `use crate::scenes::prelude::*;`.

pub use super::kit::*;
pub use super::{Scene, SceneDef};
pub use crate::frame::{count, fmt, fmt_tick, nice_step, signed, Frame};
pub use crate::gfx::camera::View;
pub use crate::gfx::color::{pal, table_index, Cmap, Color};
pub use crate::gfx::draw::{LineKind, LinePoint, LineStyle, PointKind, PointStyle};
pub use crate::gfx::{Gfx, Image, Surface, SurfaceStyle, TrailSet, ViewSettings};
pub use crate::math::expr::{Batch, Dual, Expr, Num};
pub use crate::math::rng::Rng;
pub use crate::ui::{self, ExprField};
pub use egui::Align2;
pub use glam::{Mat2, Mat3, Mat4, Vec2, Vec3};
pub use std::f32::consts::{PI, TAU};

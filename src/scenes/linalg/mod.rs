//! Linear algebra: what matrices do to space.

use crate::math::linalg::{eig2, eig3_real, svd2, Eig2};
use crate::scenes::prelude::*;
use crate::ui;

// The scenes of this topic, in the order they appear in the catalogue: seven sections,
// each a course of its own, to be read from the top.
crate::scenes::topic!(
    // vectors and maps
    vectors, span, linmap2, linmap3, compose, det, basis, rotations, // equations
    elim, subspaces, gram, lsq, overdet, underdet, condition, // eigenvalues
    eigen, powers, quadform, // data and dimensions
    svd, lowrank, pca, dft, highdim, sensing, packing, // special matrices
    special, patterns, symskew, project, commute, // factorisations
    factor, spectral, cholesky, polar, pinv, schur, jordan, // exponentials
    expm, expskew, expsum, matfun, modes,
);

/// The headings of this topic's three later sections (the scenes of each name it as
/// their group): the kinds of matrix there are, the ways of taking one apart, and the
/// exponential of a matrix with what follows from it.
pub const SPECIAL: &str = "Linear algebra: special matrices";
pub const FACTOR: &str = "Linear algebra: factorisations";
pub const EXPONENTIAL: &str = "Linear algebra: exponentials";

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One entry of a matrix, to drag or type. (Within limits: text can say "1e30" or
/// "inf", and the picture of a matrix like that is not made of finite numbers.)
fn entry(ui: &mut egui::Ui, v: &mut f32) -> egui::Response {
    ui.add(egui::DragValue::new(v).speed(0.01).fixed_decimals(2).range(-100.0..=100.0))
}

/// Edit a 2×2 matrix as its four numbers. Returns true when a number was typed in
/// (or stepped with the arrow keys). A number being dragged changes the matrix too,
/// a little every frame, but does not count: a scene that replays its animation for
/// a new matrix should let the picture follow a drag as it goes instead.
fn mat2_edit(ui: &mut egui::Ui, m: &mut Mat2) -> bool {
    let (mut changed, mut typed) = (false, false);
    let mut a = [[m.x_axis.x, m.y_axis.x], [m.x_axis.y, m.y_axis.y]];
    egui::Grid::new("m2").spacing([6.0, 4.0]).show(ui, |ui| {
        for row in &mut a {
            for v in row.iter_mut() {
                let r = entry(ui, v);
                changed |= r.changed();
                typed |= r.changed() && !r.dragged();
            }
            ui.end_row();
        }
    });
    if changed {
        *m = Mat2::from_cols(Vec2::new(a[0][0], a[1][0]), Vec2::new(a[0][1], a[1][1]));
    }
    typed
}

/// Draw the image of the integer grid under `m`.
fn warped_grid(f: &mut Frame, m: Mat2, n: i32, color: Color) {
    let l = n as f32;
    for k in -n..=n {
        let kf = k as f32;
        let (w, a) = if k == 0 { (2.0, 0.95) } else { (1.1, 0.42) };
        f.line(m * Vec2::new(kf, -l), m * Vec2::new(kf, l), w, color.a(a));
        f.line(m * Vec2::new(-l, kf), m * Vec2::new(l, kf), w, color.a(a));
    }
}

/// A flat sheet in space: the part of the plane through `centre` spanned by the unit
/// directions `u` and `v`, out to `half` either way, see-through, with `lines` grid
/// lines each side of the middle one. (If `v` is zero the plane has come down to a
/// line, and the sheet is drawn as the line it then is.)
fn sheet(f: &mut Frame, centre: Vec3, u: Vec3, v: Vec3, half: f32, lines: i32, color: Color) {
    let q = |s: f32, t: f32| centre + u * s + v * t;
    f.dl.tris_alpha.quad(q(-half, -half), q(half, -half), q(half, half), q(-half, half), color.a(0.16));
    for k in -lines..=lines {
        let s = k as f32 * half / lines.max(1) as f32;
        f.line(q(s, -half), q(s, half), 1.0, color.a(0.35));
        f.line(q(-half, s), q(half, s), 1.0, color.a(0.35));
    }
}

/// The small square corner that marks a right angle at `at`, between the directions
/// `d1` and `d2` (any lengths), with sides `size` long.
fn right_angle(f: &mut Frame, at: Vec3, d1: Vec3, d2: Vec3, size: f32, color: Color) {
    let (a, b) = (d1.normalize_or_zero() * size, d2.normalize_or_zero() * size);
    f.path([at + a, at + a + b, at + b], 1.4, color);
}

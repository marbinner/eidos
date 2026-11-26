//! Cameras: a z-up orbit camera for 3D and a pan/zoom camera for 2D.
//!
//! Every input moves a *goal*; the visible state eases towards it with frame-rate
//! independent exponential smoothing, and orbiting keeps a little inertia after release.

use glam::camera::rh::{proj::directx as proj, view::look_at_mat4};
use glam::{Mat4, Vec2, Vec3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CamMode {
    Orbit,
    Flat,
}

/// Where a scene wants the camera to start.
#[derive(Clone, Copy, Debug)]
pub enum View {
    /// 2D view centred on `center`, showing `half_height` world units above and below.
    Flat { center: Vec2, half_height: f32 },
    /// 2D view that keeps the whole rectangle `center ± half` visible whatever the
    /// shape of the window.
    Fit { center: Vec2, half: Vec2 },
    /// 3D orbit around `target`; angles in radians.
    Orbit { target: Vec3, yaw: f32, pitch: f32, dist: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Pose {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    dist: f32,
    center: Vec2,
    half_h: f32,
}

#[derive(Clone, Debug)]
pub struct Camera {
    pub mode: CamMode,
    cur: Pose,
    goal: Pose,
    home: Pose,
    spin: Vec2,
    grip: Grip,
    /// A `View::Fit` rectangle: its half extents, whether the view was to snap to
    /// it, and whether it has been fitted to the viewport once already.
    fit: Option<(Vec2, bool, bool)>,
    pub fov_y: f32,
    pub auto_rotate: f32,

    // derived each frame by `update`
    pub viewport: Vec2,
    pub eye: Vec3,
    pub right: Vec3,
    pub up: Vec3,
    pub fwd: Vec3,
    pub view: Mat4,
    pub proj: Mat4,
    pub view_proj: Mat4,
    pub inv_view_proj: Mat4,
    pub near: f32,
}

const PITCH_LIMIT: f32 = 1.553;
/// Radians the view turns per logical pixel of drag.
const TURN: f32 = 0.0125;
/// Radians a second of glide per pixel a second of drag at release: about half of
/// `TURN`, so letting go reads as friction taking hold rather than as a push.
const GLIDE: f32 = 0.0065;

/// What the pointer is doing to an orbiting view this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Grip {
    Free,
    Dragging,
    Still,
}

impl Camera {
    pub fn new(view: View) -> Self {
        let mut cam = Camera {
            mode: CamMode::Flat,
            cur: Pose { target: Vec3::ZERO, yaw: 0.9, pitch: 0.5, dist: 8.0, center: Vec2::ZERO, half_h: 3.0 },
            goal: Pose { target: Vec3::ZERO, yaw: 0.9, pitch: 0.5, dist: 8.0, center: Vec2::ZERO, half_h: 3.0 },
            home: Pose { target: Vec3::ZERO, yaw: 0.9, pitch: 0.5, dist: 8.0, center: Vec2::ZERO, half_h: 3.0 },
            spin: Vec2::ZERO,
            grip: Grip::Free,
            fit: None,
            fov_y: 0.62,
            auto_rotate: 0.0,
            viewport: Vec2::new(1280.0, 720.0),
            eye: Vec3::ZERO,
            right: Vec3::X,
            up: Vec3::Y,
            fwd: -Vec3::Z,
            view: Mat4::IDENTITY,
            proj: Mat4::IDENTITY,
            view_proj: Mat4::IDENTITY,
            inv_view_proj: Mat4::IDENTITY,
            near: 0.01,
        };
        cam.set_view(view, true);
        cam.update(0.0, cam.viewport);
        cam
    }

    /// Point the camera at a new default view. With `snap` it jumps, otherwise it glides.
    pub fn set_view(&mut self, view: View, snap: bool) {
        match view {
            View::Flat { center, half_height } => {
                self.mode = CamMode::Flat;
                self.goal.center = center;
                self.goal.half_h = half_height;
                self.fit = None;
            }
            View::Fit { center, half } => {
                self.mode = CamMode::Flat;
                self.goal.center = center;
                self.goal.half_h = half.y;
                self.fit = Some((half, snap, false));
            }
            View::Orbit { target, yaw, pitch, dist } => {
                self.mode = CamMode::Orbit;
                self.goal.target = target;
                self.goal.yaw = yaw;
                self.goal.pitch = pitch;
                self.goal.dist = dist;
                self.fit = None;
            }
        }
        self.home = self.goal;
        self.spin = Vec2::ZERO;
        if snap {
            self.cur = self.goal;
        }
    }

    pub fn reset(&mut self) {
        // take the short way round
        let turns = ((self.cur.yaw - self.home.yaw) / std::f32::consts::TAU).round();
        self.cur.yaw -= turns * std::f32::consts::TAU;
        self.goal = self.home;
        self.spin = Vec2::ZERO;
    }

    /// True while the view is still gliding or spinning.
    pub fn moving(&self) -> bool {
        self.spin.length() > 1e-4 || (self.auto_rotate != 0.0 && self.mode == CamMode::Orbit) || self.cur != self.goal
    }

    /// Drag to orbit (3D). `d` in logical pixels: the view turns by `TURN` radians for
    /// each, and the speed of the drag is kept for the glide after release.
    pub fn orbit(&mut self, d: Vec2, dt: f32) {
        self.goal.yaw -= d.x * TURN;
        self.goal.pitch = (self.goal.pitch + d.y * TURN).clamp(-PITCH_LIMIT, PITCH_LIMIT);
        if dt > 0.0 {
            let v = Vec2::new(-d.x, d.y) * GLIDE / dt;
            self.spin = self.spin.lerp(v, 0.5);
        }
        self.grip = Grip::Dragging;
    }

    /// Called every frame the pointer is held but did not move, so a flick that stops
    /// before release does not keep spinning.
    pub fn hold(&mut self) {
        self.grip = Grip::Still;
    }

    /// Drag to pan. `d` in target pixels.
    pub fn pan(&mut self, d: Vec2) {
        match self.mode {
            CamMode::Flat => {
                let k = 2.0 * self.cur.half_h / self.viewport.y.max(1.0);
                self.goal.center += Vec2::new(-d.x, d.y) * k;
                self.cur.center += Vec2::new(-d.x, d.y) * k;
            }
            CamMode::Orbit => {
                let k = 2.0 * self.cur.dist * (self.fov_y * 0.5).tan() / self.viewport.y.max(1.0);
                let delta = (-self.right * d.x + self.up * d.y) * k;
                self.goal.target += delta;
                self.cur.target += delta;
                self.goal.dist = self.goal.dist.max(self.closest());
            }
        }
    }

    /// The nearest an orbiting view may come to its target: a fiftieth of the
    /// target's distance from the origin. Closer than that, the near plane lies
    /// where neighbouring numbers are no longer different places, and what is drawn
    /// just in front of the camera (charts and legends in screen space) falls apart.
    fn closest(&self) -> f32 {
        1e-4f32.max(0.02 * self.goal.target.abs().max_element())
    }

    /// Zoom by `factor` (> 1 zooms in), keeping the world point under `px` fixed in 2D.
    pub fn zoom(&mut self, factor: f32, px: Option<Vec2>) {
        let factor = factor.clamp(0.2, 5.0);
        match self.mode {
            CamMode::Flat => {
                // (no further in than a ten-thousandth of the distance from the
                // origin: beyond that, neighbouring pixels are the same number)
                let closest = 1e-6f32.max(1e-4 * self.goal.center.abs().max_element());
                let new_h = (self.goal.half_h / factor).max(closest).min(1e9);
                let f = new_h / self.goal.half_h;
                if let Some(px) = px {
                    let w = self.flat_unproject(px, self.goal.center, self.goal.half_h);
                    self.goal.center = w + (self.goal.center - w) * f;
                }
                self.goal.half_h = new_h;
            }
            CamMode::Orbit => {
                self.goal.dist = (self.goal.dist / factor).max(self.closest()).min(1e7);
            }
        }
    }

    fn flat_unproject(&self, px: Vec2, center: Vec2, half_h: f32) -> Vec2 {
        let ndc = Vec2::new(px.x / self.viewport.x * 2.0 - 1.0, 1.0 - px.y / self.viewport.y * 2.0);
        let aspect = self.viewport.x / self.viewport.y.max(1.0);
        center + Vec2::new(ndc.x * half_h * aspect, ndc.y * half_h)
    }

    /// Advance smoothing and rebuild matrices. `viewport` is the render target size in pixels.
    pub fn update(&mut self, dt: f32, viewport: Vec2) {
        self.viewport = viewport.max(Vec2::ONE);
        if let Some((half, snap, fitted)) = self.fit {
            // re-fit while the window changes shape, until the user takes over the zoom
            let h = half.y.max(half.x * self.viewport.y / self.viewport.x);
            if (self.home.half_h - h).abs() > 1e-6 * h {
                let untouched = (self.goal.half_h - self.home.half_h).abs() <= 1e-6 * self.home.half_h;
                if untouched {
                    if fitted || snap {
                        // the viewport changed shape under a view already in place:
                        // follow it at once. (Gliding would count as movement, and
                        // movement lowers the render scale, which changes the
                        // viewport's rounded shape again: round and round.)
                        self.cur.half_h *= h / self.goal.half_h;
                    }
                    self.goal.half_h = h;
                }
                self.home.half_h = h;
            }
            self.fit = Some((half, snap, true));
        }
        if dt > 0.0 {
            if self.mode == CamMode::Orbit {
                // While the pointer has hold of the view, the pointer alone turns it:
                // the glide is for after it lets go. (It used to be added during the
                // drag as well, so the view turned twice as far as the pointer went,
                // by an amount that depended on the frame rate.)
                let glide = match self.grip {
                    Grip::Free => self.spin,
                    Grip::Dragging => Vec2::ZERO,
                    Grip::Still => {
                        // held without moving: the speed of the flick dies away fast
                        self.spin *= (-dt * 40.0).exp();
                        Vec2::ZERO
                    }
                };
                self.goal.yaw += (glide.x + self.auto_rotate) * dt;
                self.goal.pitch = (self.goal.pitch + glide.y * dt).clamp(-PITCH_LIMIT, PITCH_LIMIT);
                if self.grip == Grip::Free {
                    self.spin *= (-dt * 3.5).exp();
                }
                if self.spin.length() < 2e-3 {
                    self.spin = Vec2::ZERO;
                }
            }
            self.grip = Grip::Free;
            // (whole turns the view has made are dropped, so the angle stays a small
            // number however long the picture has been turning)
            let turns = (self.goal.yaw / std::f32::consts::TAU).trunc() * std::f32::consts::TAU;
            if turns != 0.0 {
                self.goal.yaw -= turns;
                self.cur.yaw -= turns;
            }
            let k = 1.0 - (-dt * 16.0).exp();
            let before = self.cur;
            let c = &mut self.cur;
            let g = &self.goal;
            c.target = c.target.lerp(g.target, k);
            c.yaw += (g.yaw - c.yaw) * k;
            c.pitch += (g.pitch - c.pitch) * k;
            c.dist *= (g.dist / c.dist).powf(k);
            c.center = c.center.lerp(g.center, k);
            c.half_h *= (g.half_h / c.half_h).powf(k);
            // Easing only ever approaches its goal. Arrive exactly once the difference
            // is far below a pixel, so a camera at rest is bit-for-bit at rest and the
            // renderer can tell that nothing changed.
            let tol = 1e-5;
            if (c.target - g.target).length() <= tol * g.dist {
                c.target = g.target;
            }
            if (c.yaw - g.yaw).abs() <= tol {
                c.yaw = g.yaw;
            }
            if (c.pitch - g.pitch).abs() <= tol {
                c.pitch = g.pitch;
            }
            if (c.dist / g.dist - 1.0).abs() <= tol {
                c.dist = g.dist;
            }
            if (c.center - g.center).length() <= tol * g.half_h {
                c.center = g.center;
            }
            if (c.half_h / g.half_h - 1.0).abs() <= tol {
                c.half_h = g.half_h;
            }
            // Far from the origin or zoomed far in, those tolerances are finer than
            // the numbers themselves: a step that changed nothing has arrived too.
            if c.target == before.target {
                c.target = g.target;
            }
            if c.yaw == before.yaw {
                c.yaw = g.yaw;
            }
            if c.pitch == before.pitch {
                c.pitch = g.pitch;
            }
            if c.dist == before.dist {
                c.dist = g.dist;
            }
            if c.center == before.center {
                c.center = g.center;
            }
            if c.half_h == before.half_h {
                c.half_h = g.half_h;
            }
        }
        let aspect = self.viewport.x / self.viewport.y;
        match self.mode {
            CamMode::Orbit => {
                let c = &self.cur;
                let dir = Vec3::new(c.pitch.cos() * c.yaw.cos(), c.pitch.cos() * c.yaw.sin(), c.pitch.sin());
                self.eye = c.target + dir * c.dist;
                self.view = look_at_mat4(self.eye, c.target, Vec3::Z);
                self.near = (c.dist * 0.02).max(1e-5);
                self.proj = proj::perspective(self.fov_y, aspect, self.near, c.dist * 200.0);
            }
            CamMode::Flat => {
                let c = &self.cur;
                self.eye = Vec3::new(c.center.x, c.center.y, 50.0);
                self.view = look_at_mat4(self.eye, Vec3::new(c.center.x, c.center.y, 0.0), Vec3::Y);
                let hw = c.half_h * aspect;
                self.near = 0.1;
                self.proj = proj::orthographic(-hw, hw, -c.half_h, c.half_h, 0.1, 100.0);
            }
        }
        self.view_proj = self.proj * self.view;
        self.inv_view_proj = self.view_proj.inverse();
        let inv_view = self.view.inverse();
        self.right = inv_view.x_axis.truncate();
        self.up = inv_view.y_axis.truncate();
        self.fwd = -inv_view.z_axis.truncate();
    }

    pub fn is_flat(&self) -> bool {
        self.mode == CamMode::Flat
    }

    pub fn target(&self) -> Vec3 {
        match self.mode {
            CamMode::Orbit => self.cur.target,
            CamMode::Flat => self.cur.center.extend(0.0),
        }
    }

    pub fn dist(&self) -> f32 {
        self.cur.dist
    }

    /// Visible world rectangle in 2D: (min, max).
    pub fn flat_bounds(&self) -> (Vec2, Vec2) {
        let hw = self.cur.half_h * self.viewport.x / self.viewport.y;
        let h = Vec2::new(hw, self.cur.half_h);
        (self.cur.center - h, self.cur.center + h)
    }

    /// World point to target pixels (y down). None if behind the camera.
    pub fn project(&self, p: Vec3) -> Option<Vec2> {
        let c = self.view_proj * p.extend(1.0);
        if c.w <= 1e-6 {
            return None;
        }
        let ndc = c.truncate() / c.w;
        Some(Vec2::new((ndc.x * 0.5 + 0.5) * self.viewport.x, (0.5 - ndc.y * 0.5) * self.viewport.y))
    }

    /// Ray through a pixel: (origin, unit direction).
    pub fn ray(&self, px: Vec2) -> (Vec3, Vec3) {
        let ndc = Vec2::new(px.x / self.viewport.x * 2.0 - 1.0, 1.0 - px.y / self.viewport.y * 2.0);
        let a = self.inv_view_proj.project_point3(ndc.extend(0.0));
        let b = self.inv_view_proj.project_point3(ndc.extend(1.0));
        (a, (b - a).normalize_or_zero())
    }

    /// Where the ray through `px` meets the plane through `point` with `normal`.
    pub fn pick_plane(&self, px: Vec2, point: Vec3, normal: Vec3) -> Option<Vec3> {
        let (o, d) = self.ray(px);
        let denom = d.dot(normal);
        if denom.abs() < 1e-6 {
            return None;
        }
        let t = (point - o).dot(normal) / denom;
        (t > 0.0).then(|| o + d * t)
    }

    /// World point under a pixel on the plane z = 0.
    pub fn pick_ground(&self, px: Vec2) -> Option<Vec3> {
        self.pick_plane(px, Vec3::ZERO, Vec3::Z)
    }

    /// World units covered by one target pixel at `at`.
    pub fn world_per_px(&self, at: Vec3) -> f32 {
        match self.mode {
            CamMode::Flat => 2.0 * self.cur.half_h / self.viewport.y,
            CamMode::Orbit => {
                let depth = (at - self.eye).dot(self.fwd).max(self.near);
                2.0 * depth * (self.fov_y * 0.5).tan() / self.viewport.y
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WINDOW: Vec2 = Vec2::new(1200.0, 800.0);

    /// Frames until the camera is at rest (none of these cases should need many).
    fn settle(cam: &mut Camera, viewport: Vec2) -> usize {
        let mut frames = 0;
        while cam.moving() {
            cam.update(1.0 / 60.0, viewport);
            frames += 1;
            assert!(frames < 2000, "still moving after {frames} frames: {cam:?}");
        }
        assert!(cam.view_proj.is_finite() && cam.inv_view_proj.is_finite() && cam.eye.is_finite(), "{cam:?}");
        frames
    }

    /// A view that is being eased towards its goal arrives, exactly: zoomed as far in
    /// as it goes, as far out, and far from the origin, where the steps become too
    /// small for the numbers to hold (it once went on for ever there, and the page
    /// with it).
    #[test]
    fn a_flat_view_left_alone_comes_to_rest() {
        for centre in [Vec2::ZERO, Vec2::new(3.0, -2.0), Vec2::new(1e6, -3e5), Vec2::new(1e-6, 1e-6)] {
            let mut cam = Camera::new(View::Flat { center: centre, half_height: 4.0 });
            cam.update(1.0 / 60.0, WINDOW);
            for factor in [5.0, 0.2] {
                for _ in 0..60 {
                    cam.zoom(factor, Some(Vec2::new(900.0, 150.0)));
                    cam.update(1.0 / 60.0, WINDOW);
                }
                settle(&mut cam, WINDOW);
                let (lo, hi) = cam.flat_bounds();
                assert!(lo.is_finite() && hi.is_finite() && hi.x > lo.x && hi.y > lo.y, "centre {centre}, zoom {factor}: {lo} to {hi}");
            }
            cam.pan(Vec2::new(-300.0, 120.0));
            cam.reset();
            settle(&mut cam, WINDOW);
            let (lo, hi) = cam.flat_bounds();
            assert!(
                ((lo + hi) * 0.5 - centre).length() <= 1e-3 * centre.length().max(1.0) && ((hi.y - lo.y) * 0.5 - 4.0).abs() < 1e-3,
                "reset leads home: {lo} to {hi}"
            );
        }
    }

    /// Zooming keeps the place under the pointer where it is.
    #[test]
    fn zooming_holds_the_point_under_the_pointer() {
        let mut cam = Camera::new(View::Flat { center: Vec2::new(1.0, 2.0), half_height: 3.0 });
        cam.update(1.0 / 60.0, WINDOW);
        let pointer = Vec2::new(950.0, 220.0);
        let under = |cam: &Camera| cam.pick_ground(pointer).expect("the pointer is over the plane").truncate();
        let before = under(&cam);
        for _ in 0..12 {
            cam.zoom(1.3, Some(pointer));
        }
        settle(&mut cam, WINDOW);
        assert!((under(&cam) - before).length() < 1e-3, "{before} became {}", under(&cam));
    }

    /// A fitted view shows its whole rectangle in a window of any shape, and follows
    /// a change of shape at once, without gliding (gliding counts as movement, and
    /// movement lowers the resolution, which changes the shape again).
    #[test]
    fn a_fitted_view_follows_the_window() {
        let (centre, half) = (Vec2::new(1.0, 0.5), Vec2::new(2.0, 1.0));
        let mut cam = Camera::new(View::Fit { center: centre, half });
        cam.set_view(View::Fit { center: centre, half }, true);
        for window in [WINDOW, Vec2::new(400.0, 800.0), Vec2::new(2000.0, 300.0), Vec2::new(801.0, 799.0)] {
            cam.update(1.0 / 60.0, window);
            assert!(!cam.moving(), "the view glides after the window became {window}");
            let (lo, hi) = cam.flat_bounds();
            let slack = 1e-4;
            assert!(
                lo.x <= centre.x - half.x + slack
                    && hi.x >= centre.x + half.x - slack
                    && lo.y <= centre.y - half.y + slack
                    && hi.y >= centre.y + half.y - slack,
                "{window}: {lo} to {hi}"
            );
            // and no more than it has to: one of the two directions is filled exactly
            let spare = ((hi.x - lo.x) * 0.5 - half.x).min((hi.y - lo.y) * 0.5 - half.y);
            assert!(spare.abs() < 1e-3, "{window}: spare room {spare}");
        }
        // once the visitor has zoomed, the window's shape no longer overrides them
        cam.zoom(2.0, None);
        settle(&mut cam, WINDOW);
        let zoomed = cam.flat_bounds();
        cam.update(1.0 / 60.0, Vec2::new(400.0, 800.0));
        assert_eq!(cam.flat_bounds().1.y - cam.flat_bounds().0.y, zoomed.1.y - zoomed.0.y);
    }

    /// A 3D view: a flick keeps turning for a moment and then stops; a view that
    /// turns by itself keeps a small angle however long it has been turning, and
    /// "back to the start" takes the short way round.
    /// Dragging turns the view by what the pointer moved, the same at any frame
    /// rate, and nothing more while the pointer keeps hold; let go in mid-drag and
    /// it glides on.
    #[test]
    fn a_drag_turns_the_view_by_what_the_pointer_moved() {
        let home = View::Orbit { target: Vec3::ZERO, yaw: 0.9, pitch: 0.5, dist: 8.0 };
        let mut ends = Vec::new();
        for hz in [30.0f32, 60.0, 240.0] {
            let dt = 1.0 / hz;
            let mut cam = Camera::new(home);
            cam.update(dt, WINDOW);
            // 150 pixels to the left in half a second, then held still for as long
            let frames = (0.5 * hz) as usize;
            for _ in 0..frames {
                cam.orbit(Vec2::new(-150.0 / frames as f32, 0.0), dt);
                cam.update(dt, WINDOW);
            }
            for _ in 0..frames {
                cam.hold();
                cam.update(dt, WINDOW);
            }
            assert!((cam.goal.yaw - (0.9 + 150.0 * TURN)).abs() < 1e-3, "{hz} Hz: yaw {}", cam.goal.yaw);
            // released after standing still, it stays where it was put
            let held = cam.goal.yaw;
            settle(&mut cam, WINDOW);
            assert!((cam.goal.yaw - held).abs() < 0.02, "{hz} Hz: drifted {}", cam.goal.yaw - held);
            // released in mid-flick, it glides on the same way, and comes to rest
            for _ in 0..frames {
                cam.orbit(Vec2::new(-150.0 / frames as f32, 0.0), dt);
                cam.update(dt, WINDOW);
            }
            let let_go = cam.goal.yaw;
            settle(&mut cam, WINDOW);
            assert!(cam.goal.yaw - let_go > 0.2, "{hz} Hz: glided {}", cam.goal.yaw - let_go);
            ends.push(cam.goal.yaw);
        }
        assert!(ends.iter().all(|e| (e - ends[1]).abs() < 0.15), "the glide depends on the frame rate: {ends:?}");
    }

    #[test]
    fn an_orbiting_view_stops_and_keeps_its_angle_small() {
        let home = View::Orbit { target: Vec3::new(0.0, 0.0, 0.5), yaw: 0.9, pitch: 0.5, dist: 8.0 };
        let mut cam = Camera::new(home);
        cam.update(1.0 / 60.0, WINDOW);
        for _ in 0..5 {
            cam.orbit(Vec2::new(40.0, -12.0), 1.0 / 60.0);
            cam.update(1.0 / 60.0, WINDOW);
        }
        assert!(cam.moving());
        assert!(settle(&mut cam, WINDOW) < 600);
        // an hour of turning by itself
        cam.auto_rotate = 0.4;
        for _ in 0..216_000 {
            cam.update(1.0 / 60.0, WINDOW);
        }
        assert!(cam.cur.yaw.abs() <= std::f32::consts::TAU + 0.1 && cam.goal.yaw.abs() <= std::f32::consts::TAU + 0.1, "yaw {}", cam.cur.yaw);
        cam.auto_rotate = 0.0;
        cam.reset();
        assert!((cam.cur.yaw - 0.9).abs() <= std::f32::consts::PI + 1e-3, "the short way round from {}", cam.cur.yaw);
        settle(&mut cam, WINDOW);
        assert!((cam.cur.yaw - 0.9).abs() < 1e-4 && (cam.dist() - 8.0).abs() < 1e-4);
        // zoomed all the way in and all the way out, it still arrives
        for factor in [5.0, 0.2] {
            for _ in 0..40 {
                cam.zoom(factor, None);
            }
            settle(&mut cam, WINDOW);
        }
        // a point in front of the camera lands inside the window; one behind it nowhere
        cam.reset();
        settle(&mut cam, WINDOW);
        let centre = cam.project(Vec3::new(0.0, 0.0, 0.5)).expect("the target is in view");
        assert!((centre - WINDOW * 0.5).length() < 1.0, "{centre}");
        assert!(cam.project(cam.eye - cam.fwd).is_none());
    }
}

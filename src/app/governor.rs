//! Resolution and finish traded for frame rate on slow GPUs, and where there is none.

/// Keeps animation fluid on slow GPUs by trading resolution for frame rate, and
/// gives the resolution back as soon as it can.
///
/// Frame intervals are watched only while something is animating. The rules:
/// - only sustained slowness counts (not a stall while a scene starts, nor one hitch);
/// - each reduction is sized by how slow the frames are;
/// - at a quarter of the pixels and still slow, the picture is drawn plainly as well
///   (see `Gfx::set_plain`: no multisampling, 8 bits a channel, no glow), and that
///   stays for the visit, since changing over rebuilds the pipelines; where no GPU is
///   doing the drawing it is plain from the start (see [`Governor::plainly`]);
/// - if even a quarter of the pixels makes frames no faster, resolution was never the
///   problem (the time goes elsewhere), so full resolution is restored and left alone
///   (after the plain drawing has been given its chance in the same way);
/// - after a few comfortable seconds a higher resolution is tried; if that turns out
///   too much it is taken back, and the next try waits longer, so the picture does
///   not pulse;
/// - a new scene starts at the resolution the last one ended at, and tries the full
///   one after half a comfortable second: on a slow device that is a second of slow
///   frames where coming down from the full resolution again took three or four, and
///   on a fast one that a heavy scene had slowed, half a second of a coarser picture;
///   a picture at rest is always drawn at full resolution (see the viewport code);
/// - but while the display's own pace is unknown, a verdict of futility is kept from
///   scene to scene (see [`Governor::afresh`]).
pub(super) struct Governor {
    pub(super) scale: f32,
    /// Frames are ignored for this long after a change (allocation, compilation).
    settle: f32,
    /// How long frames have been slow without a break, and how slow.
    slow_for: f32,
    recent: f32,
    quick_run: u32,
    /// How long frames have been comfortable since the last change.
    calm: f32,
    /// The frame interval when slowness was first met at full resolution.
    baseline: Option<f32>,
    /// While a higher resolution is on trial: the scale to return to if it fails.
    trial: Option<f32>,
    /// Comfortable seconds required before the next trial.
    patience: f32,
    /// Lowering the resolution did not help here.
    futile: bool,
    /// Slow frames in a row: one long frame, or two, is a hitch (a tab coming back,
    /// a shader being compiled), whatever its length.
    slow_run: u32,
    /// The picture is to be drawn plainly.
    pub(super) plain: bool,
    /// The scene began at a resolution left by the one before it: the first try at a
    /// higher one comes soon, and goes straight to the full one.
    inherited: bool,
}

impl Governor {
    /// A frame slower than this (about 42 per second) counts against the current scale.
    const SLOW: f32 = 0.024;
    /// Slow frames must go on for this long before anything is done about them.
    const SUSTAINED: f32 = 0.4;
    const FLOOR: f32 = 0.5;
    const PATIENCE: f32 = 2.5;
    /// Comfortable seconds before a scene that began at a lowered resolution tries the full one.
    const SOON: f32 = 0.5;

    pub(super) fn new() -> Self {
        Governor {
            scale: 1.0,
            settle: 1.0,
            slow_for: 0.0,
            recent: 0.0,
            quick_run: 0,
            calm: 0.0,
            baseline: None,
            trial: None,
            patience: Self::PATIENCE,
            futile: false,
            slow_run: 0,
            plain: false,
            inherited: false,
        }
    }

    /// The same, drawing plainly from the start: for where a software rasteriser is
    /// doing the drawing, which has the most to gain from it and whose every frame
    /// before it was found out would take a quarter of a second or more.
    pub(super) fn plainly() -> Self {
        Governor { plain: true, ..Governor::new() }
    }

    /// The governor for a new scene, which is judged afresh at full resolution. Except
    /// when lowering the resolution was found futile while the display's own pace
    /// (`beat`) was still unknown: then the display itself may be what is slow (30 or
    /// 40 frames a second, as a phone saving power shows), and every moving scene
    /// would find that out again only after seconds of blur. That verdict stands
    /// until the pace has been seen.
    pub(super) fn afresh(&self, beat: f32) -> Governor {
        // (drawing plainly is kept from scene to scene: it says how fast the device is,
        // not how heavy one scene is, and changing over is not free)
        let futile = self.futile && beat == 0.0;
        // (a resolution on trial when the scene was left is not one that was found to do)
        let scale = if futile { 1.0 } else { self.trial.unwrap_or(self.scale) };
        Governor {
            futile,
            plain: self.plain,
            scale,
            inherited: scale < 1.0,
            patience: if scale < 1.0 { Self::SOON } else { Self::PATIENCE },
            ..Governor::new()
        }
    }

    fn set(&mut self, scale: f32) {
        self.scale = scale;
        self.pause(0.5);
        self.calm = 0.0;
    }

    /// Disregard the next `seconds` of frames (something is settling).
    pub(super) fn pause(&mut self, seconds: f32) {
        self.settle = self.settle.max(seconds);
        self.slow_for = 0.0;
        self.slow_run = 0;
        self.quick_run = 0;
    }

    /// `beat` is the display's own frame interval if it is known (0 otherwise): the
    /// shortest interval seen between frames in which nothing had to be drawn. On a
    /// display that only shows 30 or 40 frames a second, every frame is longer
    /// than `SLOW`, and no resolution could make it shorter.
    pub(super) fn observe(&mut self, dt: f32, beat: f32) {
        if self.settle > 0.0 {
            self.settle -= dt;
            return;
        }
        let slow = Self::SLOW.max(1.5 * beat);
        if dt > slow {
            self.recent = if self.slow_for == 0.0 { dt } else { self.recent + (dt - self.recent) * 0.25 };
            self.slow_for += dt;
            self.slow_run += 1;
            self.quick_run = 0;
        } else {
            self.quick_run += 1;
            if self.quick_run >= 3 {
                self.slow_for = 0.0;
                self.slow_run = 0;
            }
            self.calm += dt;
        }
        if self.futile {
            return;
        }
        if self.slow_for >= Self::SUSTAINED && self.slow_run >= 3 {
            if let Some(fallback) = self.trial.take() {
                // the higher resolution on trial is too much: back, and wait longer next time
                self.patience = (self.patience * 2.0).clamp(Self::PATIENCE, 60.0);
                self.set(fallback);
            } else if self.scale > Self::FLOOR {
                self.baseline.get_or_insert(self.recent);
                // drawing time goes roughly with the number of pixels
                let factor = (0.85 * slow / self.recent).sqrt().clamp(0.7, 0.9);
                self.set((self.scale * factor).max(Self::FLOOR));
            } else if self.baseline.is_some_and(|before| self.recent > 0.9 * before) {
                if self.plain {
                    self.futile = true;
                } else {
                    // resolution is not what is slow: plain drawing is tried before
                    // giving up, and judged as the resolution was, from the start
                    self.plain = true;
                    self.baseline = None;
                }
                self.set(1.0);
            } else {
                // at the floor, and it does help, but not enough: the plain drawing
                // too, if it is not that already; then stay
                self.plain = true;
                self.slow_for = 0.0;
                self.slow_run = 0;
                self.calm = 0.0;
            }
        } else if self.scale < 1.0 {
            if self.trial.is_some() {
                if self.calm >= 1.5 {
                    // the higher resolution holds
                    self.trial = None;
                    self.patience = Self::PATIENCE;
                    self.calm = 0.0;
                }
            } else if self.calm >= self.patience {
                self.trial = Some(self.scale);
                let higher = if self.inherited { 1.0 } else { (self.scale * 1.25).min(1.0) };
                self.inherited = false;
                self.set(higher);
            }
        } else if self.trial.is_some() && self.calm >= 1.5 {
            // back at full resolution, and it holds
            self.trial = None;
            self.patience = Self::PATIENCE;
            self.baseline = None;
        }
    }

    /// What it is doing, for the View panel.
    pub(super) fn status(&self) -> String {
        let plain = if self.plain { " Drawn plainly here (no glow, edges less smooth), which is several times less work for a slow device." } else { "" };
        if self.futile {
            format!("Frames come no faster here at a lower resolution, so the resolution is left at full.{plain}")
        } else if self.scale < 1.0 {
            format!("Drawing moving pictures at {:.0} % resolution to keep them fluid; it tries a higher one every few seconds.{plain}", self.scale * 100.0)
        } else {
            format!("Full resolution.{plain}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Governor;

    /// A fast machine in front of a display that shows 30 frames a second: every
    /// frame takes 33 ms, and once the display's pace is known the resolution is
    /// left alone.
    #[test]
    fn a_slow_display_is_not_a_slow_machine() {
        let mut g = Governor::new();
        for _ in 0..600 {
            g.observe(1.0 / 30.0, 1.0 / 30.0);
        }
        assert!(g.scale == 1.0 && !g.futile, "scale {}", g.scale);
        // while a machine that needs three such intervals for a frame still has a
        // lower resolution tried on it
        let mut lowest = 1.0f32;
        for _ in 0..20 {
            g.observe(0.1, 1.0 / 30.0);
            lowest = lowest.min(g.scale);
        }
        assert!(lowest < 1.0);
    }

    /// The same display before its pace has been seen (a visit that opens on a moving
    /// scene, from a shared link): a lower resolution is tried and found futile once
    /// in the visit, not again in every moving scene after it.
    #[test]
    fn a_slow_display_not_yet_measured_costs_one_trial_a_visit() {
        let mut g = Governor::new();
        let mut lowest = 1.0f32;
        for _ in 0..600 {
            g.observe(1.0 / 30.0, 0.0);
            lowest = lowest.min(g.scale);
        }
        assert!(lowest < 1.0 && g.futile && g.scale == 1.0, "lowest {lowest} scale {} futile {}", g.scale, g.futile);
        // the next scene, and the one after
        for _ in 0..2 {
            g = g.afresh(0.0);
            for _ in 0..600 {
                g.observe(1.0 / 30.0, 0.0);
                assert_eq!(g.scale, 1.0);
            }
        }
        // once the pace is known, a new scene is judged afresh (and the display's
        // slowness is then no longer mistaken for the machine's)
        let mut g = g.afresh(1.0 / 30.0);
        assert!(!g.futile);
        for _ in 0..600 {
            g.observe(1.0 / 30.0, 1.0 / 30.0);
            assert_eq!(g.scale, 1.0);
        }
        // nor does a verdict reached on a display whose pace was known carry over
        let mut g = Governor::new();
        run(&mut g, 0.030, 0.001, 20.0);
        assert!(g.futile);
        assert!(!g.afresh(1.0 / 60.0).futile);
    }

    /// A display at 60 Hz in front of a machine whose frame takes `cpu` seconds plus
    /// `gpu` seconds at full resolution (the GPU part going with the pixel count, and
    /// drawn plainly taking two fifths of the time: between what was measured for a
    /// software rasteriser, a half to a fifth). Frames arrive on the display's beat,
    /// as in a browser.
    fn run(g: &mut Governor, cpu: f32, gpu: f32, seconds: f32) -> f32 {
        let (mut t, mut dt) = (0.0, 1.0 / 60.0);
        while t < seconds {
            let need = cpu + gpu * g.scale * g.scale * if g.plain { 0.4 } else { 1.0 };
            dt = (need * 60.0).ceil().max(1.0) / 60.0;
            g.observe(dt, 0.0);
            t += dt;
        }
        dt
    }

    #[test]
    fn a_slow_gpu_gets_a_lower_resolution_and_a_fast_one_gets_it_back() {
        let mut g = Governor::new();
        // 30 ms of drawing at full resolution: every other display frame is missed
        let dt = run(&mut g, 0.002, 0.030, 8.0);
        assert!(dt < 0.02 && g.scale < 0.75 && g.scale >= Governor::FLOOR, "scale {} interval {dt}", g.scale);
        assert!(!g.futile);
        // it does not creep back up into slowness and stay there
        let dt = run(&mut g, 0.002, 0.030, 30.0);
        assert!(g.scale < 0.8, "scale {}", g.scale);
        let _ = dt;
        // the load goes away (another scene, a lighter view): full resolution within half a minute
        run(&mut g, 0.002, 0.004, 30.0);
        assert_eq!(g.scale, 1.0);
    }

    #[test]
    fn a_stall_at_the_start_or_a_single_hitch_changes_nothing() {
        let mut g = Governor::new();
        for _ in 0..12 {
            g.observe(0.08, 0.0); // a second of start-up stutter
        }
        run(&mut g, 0.002, 0.004, 3.0);
        g.observe(0.2, 0.0); // one long frame
        run(&mut g, 0.002, 0.004, 3.0);
        assert_eq!(g.scale, 1.0);
    }

    #[test]
    fn when_resolution_is_not_the_problem_it_is_left_alone() {
        let mut g = Governor::new();
        // 30 ms of work that has nothing to do with the number of pixels
        run(&mut g, 0.030, 0.001, 20.0);
        assert!(g.futile && g.scale == 1.0, "scale {} futile {}", g.scale, g.futile);
        // (not before the plain drawing had been tried as well)
        assert!(g.plain);
        // a very slow GPU, on the other hand, keeps the lowest resolution, which does help
        let mut g = Governor::new();
        run(&mut g, 0.002, 0.200, 20.0);
        assert!(!g.futile && g.scale == Governor::FLOOR, "scale {}", g.scale);
    }

    /// A browser with no GPU behind it, or a GPU far too slow for the picture: a
    /// quarter of a second and more for every frame. (Frames that long were once
    /// taken for hitches, every one of them, and such a device was left at the full
    /// drawing and three frames a second.)
    #[test]
    fn a_device_that_needs_a_quarter_of_a_second_a_frame_is_helped() {
        let mut g = Governor::new();
        let dt = run(&mut g, 0.004, 0.300, 20.0);
        assert!(g.scale == Governor::FLOOR && g.plain && !g.futile, "scale {} plain {} futile {}", g.scale, g.plain, g.futile);
        assert!(dt <= 0.05 + 1e-6, "interval {dt}");
        // the next scene starts where this one ended, and stays plain: that says how
        // fast the device is, not how heavy one scene was
        let next = g.afresh(0.0);
        assert!(next.plain && next.scale == Governor::FLOOR);
        // a fast device is never drawn for plainly
        let mut g = Governor::new();
        run(&mut g, 0.002, 0.030, 30.0);
        assert!(!g.plain);
    }

    /// Seconds of slow frames (slower than 30 a second) in the first `seconds` of `g`
    /// on the device of [`run`].
    fn slow_seconds(g: &mut Governor, cpu: f32, gpu: f32, seconds: f32) -> f32 {
        let (mut t, mut slow) = (0.0, 0.0);
        while t < seconds {
            let need = cpu + gpu * g.scale * g.scale * if g.plain { 0.4 } else { 1.0 };
            let dt = (need * 60.0).ceil().max(1.0) / 60.0;
            g.observe(dt, 0.0);
            t += dt;
            if dt > 1.0 / 30.0 + 1e-4 {
                slow += dt;
            }
        }
        slow
    }

    /// A slow device does not begin every scene with seconds of slow frames while
    /// the resolution comes down again: the next scene starts where the last ended,
    /// tries the full resolution once, soon, and is back within about a second.
    #[test]
    fn the_next_scene_starts_where_the_last_one_ended() {
        let mut first = Governor::new();
        let began_slow = slow_seconds(&mut first, 0.002, 0.060, 8.0);
        assert!(first.scale < 0.75);
        let mut next = first.afresh(1.0 / 60.0);
        // (where it had settled: a higher resolution it was only trying does not count)
        assert_eq!(next.scale, first.trial.unwrap_or(first.scale));
        let slow = slow_seconds(&mut next, 0.002, 0.060, 8.0);
        assert!(slow < 1.3 && slow < 0.6 * began_slow, "{slow} s of slow frames in the next scene, {began_slow} in the first");
        assert!(next.scale < 0.75);
        // and a fast device that one heavy scene had slowed is back at the full
        // resolution in the next, lighter, one within two seconds
        let mut light = first.afresh(1.0 / 60.0);
        run(&mut light, 0.002, 0.004, 2.0);
        assert_eq!(light.scale, 1.0);
    }

    #[test]
    fn two_long_frames_are_a_hitch_and_not_slowness() {
        let mut g = Governor::new();
        run(&mut g, 0.002, 0.004, 3.0);
        // (a tab coming back to the front, a shader being compiled)
        g.observe(1.2, 0.0);
        g.observe(0.9, 0.0);
        run(&mut g, 0.002, 0.004, 3.0);
        assert!(g.scale == 1.0 && !g.plain, "scale {} plain {}", g.scale, g.plain);
    }

    #[test]
    fn where_no_gpu_draws_it_is_plain_from_the_first_frame() {
        let g = Governor::plainly();
        assert!(g.plain && g.scale == 1.0 && !g.futile);
    }
}

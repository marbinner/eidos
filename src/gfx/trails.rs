//! Trails: the paths of many particles, kept on the GPU as a ring of rows.

use super::color::Color;
use super::draw::{LinePoint, LineStyle, TrailDraw};
use super::gpu::next_id;
use super::{Gfx, LineU};

/// Motion trails for many particles at once.
///
/// The GPU buffer is a ring of rows: one row per time step, one column per particle.
/// Advancing the simulation uploads a single row, however long the trails are, and
/// the whole set is one draw call. Age-based fading happens in the vertex shader.
///
/// A row need not be a frame. A scene whose trails are short may lay a row only every
/// few frames and have the newest one follow its particles in between
/// ([`TrailSet::update`], [`TrailSet::follow`]): the trails then move in every frame,
/// whatever the display's rate, with a fraction of the rows (and of the segments to
/// draw) that a row a frame would take.
pub struct TrailSet {
    cols: u32,
    rows: u32,
    head: u32,
    filled: u32,
    buf: wgpu::Buffer,
    ubo: wgpu::Buffer,
    bind: wgpu::BindGroup,
    row: Vec<LinePoint>,
    pub style: LineStyle,
    pub additive: bool,
    /// How far towards the next row the time since the newest whole row was laid has
    /// got (0 to 1), if the newest row follows its particles between rows.
    follow: Option<f32>,
    last_u: Option<LineU>,
    id: u64,
    version: u64,
}

impl TrailSet {
    pub fn new(gfx: &Gfx, particles: u32, length: u32, additive: bool) -> Self {
        // (no more than fits in one buffer on this device: asked for more, the trails
        // come out shorter rather than the program stopping)
        let most = (gfx.device.limits().max_buffer_size / 20).min(u32::MAX as u64);
        let cols = particles.clamp(1, (most / 7).max(1) as u32);
        let rows = length.clamp(4, (most / cols as u64).saturating_sub(3).max(4) as u32);
        let buf = gfx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trails"),
            size: (rows as u64 + 3) * cols as u64 * 20,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ubo = gfx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("trails"),
            size: std::mem::size_of::<LineU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = gfx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("trails"),
            layout: &gfx.layout_line,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() }],
        });
        TrailSet {
            cols,
            rows,
            head: 0,
            filled: 0,
            buf,
            ubo,
            bind,
            row: vec![LinePoint { pos: [0.0; 3], width: 1.0, color: Color::WHITE }; cols as usize],
            style: LineStyle { intensity: if additive { 1.6 } else { 1.0 }, taper: 0.6, ..Default::default() },
            additive,
            follow: None,
            last_u: None,
            id: next_id(),
            version: 0,
        }
    }

    /// Forget all history (after a reset or a jump).
    pub fn clear(&mut self) {
        self.filled = 0;
        self.version += 1;
    }

    /// Whether a row has been laid since the trails were made or cleared.
    pub fn started(&self) -> bool {
        self.filled > 0
    }

    /// The newest row, one entry per particle. Fill it, then call [`TrailSet::commit`].
    pub fn row_mut(&mut self) -> &mut [LinePoint] {
        &mut self.row
    }

    /// Append the row as the newest time step. (A particle that has flown off to
    /// infinity, or to no number at all, leaves a gap in its trail.)
    pub fn commit(&mut self, gfx: &Gfx) {
        if self.filled > 0 {
            self.head = (self.head + 1) % self.rows;
        }
        self.filled = (self.filled + 1).min(self.rows);
        self.write(gfx);
    }

    /// Put the row in place of the newest time step: where the particles are now,
    /// between one row laid and the next (see [`TrailSet::follow`]). The first row of
    /// all is laid.
    pub fn update(&mut self, gfx: &Gfx) {
        if self.filled == 0 {
            return self.commit(gfx);
        }
        self.write(gfx);
    }

    /// Say that the newest row follows its particles between rows, and how far
    /// towards the next row the time since the last whole one was laid has got (from
    /// 0, just laid, to 1, the next is due): the trails then fade by that time and
    /// not by whole rows, so that nothing in them steps as a row is laid.
    pub fn follow(&mut self, towards_next: f32) {
        self.follow = Some(if towards_next.is_finite() { towards_next.clamp(0.0, 1.0) } else { 1.0 });
    }

    /// Send the row to the GPU as the newest time step.
    fn write(&mut self, gfx: &Gfx) {
        for p in &mut self.row {
            if !(p.pos.iter().all(|x| x.is_finite()) && p.width.is_finite()) {
                *p = super::draw::BREAK;
            }
        }
        self.version += 1;
        let bytes: &[u8] = bytemuck::cast_slice(&self.row);
        let stride = self.cols as u64 * 20;
        super::gpu::sent(bytes.len());
        gfx.queue.write_buffer(&self.buf, self.head as u64 * stride, bytes);
        if self.head < 3 {
            // the first three rows are mirrored past the end so windows can run off the ring
            gfx.queue.write_buffer(&self.buf, (self.head + self.rows) as u64 * stride, bytes);
        }
    }

    pub(crate) fn prepare(&mut self, gfx: &Gfx) -> Option<TrailDraw> {
        if self.filled < 2 {
            return None;
        }
        let mut u = LineU::new(&self.style, [self.cols as f32, self.head as f32, self.rows as f32, self.filled as f32]);
        if let Some(towards_next) = self.follow {
            // (1 + how much of a row's time the rows behind the newest are short of)
            u.dash[3] = 2.0 - towards_next;
        }
        if self.last_u != Some(u) {
            gfx.queue.write_buffer(&self.ubo, 0, bytemuck::bytes_of(&u));
            self.last_u = Some(u);
            self.version += 1;
        }
        Some(TrailDraw {
            buf: self.buf.clone(),
            bind: self.bind.clone(),
            cols: self.cols,
            rows: self.rows,
            additive: self.additive,
            stamp: [self.id, self.version],
        })
    }
}

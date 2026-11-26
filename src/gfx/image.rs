//! Pictures: grids of colours kept on the GPU as a texture.

use super::color::Color;
use super::draw::{DrawList, ImageDraw, ImageRect};
use super::gpu::next_id;
use super::Gfx;

/// A picture: a grid of colours kept on the GPU and drawn as rectangles in world
/// space. Use it for anything that is naturally a grid of values (a table, a matrix,
/// a heat map with visible cells) or for many small pictures at once: lay them out
/// side by side in one image, and every rectangle that shows a part of it is drawn in
/// the same draw call, with a single texture upload when the contents change.
///
/// Pixel (0, 0) is the bottom-left corner of the picture, to match the y-up world.
pub struct Image {
    w: u32,
    h: u32,
    pixels: Vec<Color>,
    /// Blend between neighbouring pixels (a smooth field) instead of showing each
    /// one as a crisp cell.
    pub smooth: bool,
    tex: wgpu::Texture,
    bind: [wgpu::BindGroup; 2],
    dirty: Dirty,
    id: u64,
    version: u64,
}

/// What the GPU's copy of a picture lacks.
enum Dirty {
    Nothing,
    /// These blocks of pixels, each `[x, y, width, height]`.
    Parts(Vec<[u32; 4]>),
    Everything,
}

/// More blocks than this waiting to be sent, and the whole picture is sent instead.
const MOST_PARTS: usize = 16;

impl Image {
    pub fn new(gfx: &Gfx, w: u32, h: u32) -> Self {
        let w = w.clamp(1, gfx.max_dim);
        let h = h.clamp(1, gfx.max_dim);
        let tex = gfx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("image"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let bind = [&gfx.sampler_nearest, &gfx.sampler].map(|sampler| {
            gfx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("image"),
                layout: &gfx.layout_image,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                ],
            })
        });
        Image { w, h, pixels: vec![Color::CLEAR; (w * h) as usize], smooth: false, tex, bind, dirty: Dirty::Everything, id: next_id(), version: 0 }
    }

    pub fn size(&self) -> [u32; 2] {
        [self.w, self.h]
    }

    /// Make sure the picture is exactly `w` by `h`, replacing it if not (its
    /// contents are then cleared). Cheap when the size already matches.
    pub fn ensure_size(&mut self, gfx: &Gfx, w: u32, h: u32) {
        if [w.clamp(1, gfx.max_dim), h.clamp(1, gfx.max_dim)] != [self.w, self.h] {
            let smooth = self.smooth;
            *self = Image::new(gfx, w, h);
            self.smooth = smooth;
        }
    }

    /// The pixels, row by row from the bottom; marks the picture for upload.
    pub fn pixels_mut(&mut self) -> &mut [Color] {
        self.dirty = Dirty::Everything;
        &mut self.pixels
    }

    /// The pixels, to be changed only inside the blocks `parts` (each `[x, y, width,
    /// height]` in pixels, row 0 at the bottom as in the slice): only those blocks are
    /// sent to the GPU. For a large picture that changes a little at a time, sending
    /// all of it in every frame is most of what it costs (a thousand by a thousand is
    /// four megabytes). A pixel changed outside the blocks stays as it was on the
    /// screen until the whole picture is next sent.
    pub fn pixels_mut_in(&mut self, parts: &[[u32; 4]]) -> &mut [Color] {
        match &mut self.dirty {
            Dirty::Everything => {}
            Dirty::Parts(waiting) if waiting.len() + parts.len() <= MOST_PARTS => waiting.extend_from_slice(parts),
            Dirty::Nothing if parts.len() <= MOST_PARTS => self.dirty = Dirty::Parts(parts.to_vec()),
            _ => self.dirty = Dirty::Everything,
        }
        &mut self.pixels
    }

    /// Send one block of the pixels to the GPU (the part of it that is in the picture).
    fn upload(&self, gfx: &Gfx, [x, y, w, h]: [u32; 4]) {
        let (x, y) = (x.min(self.w), y.min(self.h));
        let (w, h) = (w.min(self.w - x), h.min(self.h - y));
        if w == 0 || h == 0 {
            return;
        }
        let to = wgpu::TexelCopyTextureInfo { texture: &self.tex, mip_level: 0, origin: wgpu::Origin3d { x, y, z: 0 }, aspect: wgpu::TextureAspect::All };
        let extent = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
        let rows = wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: None };
        super::gpu::sent((w * h * 4) as usize);
        if w == self.w {
            // whole rows lie together in the slice
            let from = (y * self.w) as usize;
            gfx.queue.write_texture(to, bytemuck::cast_slice(&self.pixels[from..from + (w * h) as usize]), rows, extent);
        } else {
            // a narrower block is gathered first, so that only its own pixels are sent
            let mut block = Vec::with_capacity((w * h) as usize);
            for row in y..y + h {
                let from = (row * self.w + x) as usize;
                block.extend_from_slice(&self.pixels[from..from + w as usize]);
            }
            gfx.queue.write_texture(to, bytemuck::cast_slice(&block), rows, extent);
        }
    }

    /// Queue one rectangle showing the block of pixels `part` = [x, y, width, height].
    pub(crate) fn prepare(&mut self, gfx: &Gfx, dl: &mut DrawList, min: glam::Vec2, max: glam::Vec2, part: [u32; 4], z: f32) {
        match std::mem::replace(&mut self.dirty, Dirty::Nothing) {
            Dirty::Nothing => {}
            Dirty::Everything => {
                self.upload(gfx, [0, 0, self.w, self.h]);
                self.version += 1;
            }
            Dirty::Parts(parts) => {
                for part in parts {
                    self.upload(gfx, part);
                }
                self.version += 1;
            }
        }
        // a smooth picture spans pixel centres, so nothing from outside the block blends in
        let inset = if self.smooth { 0.5 } else { 0.0 };
        let (sx, sy) = (1.0 / self.w as f32, 1.0 / self.h as f32);
        let [x, y, w, h] = part.map(|v| v as f32);
        let uv = [(x + inset) * sx, (y + inset) * sy, (x + w - inset) * sx, (y + h - inset) * sy];
        if !(min.is_finite() && max.is_finite() && z.is_finite()) {
            dl.left_out += 1;
            return;
        }
        dl.image_rects.push(ImageRect { rect: [min.x, min.y, max.x, max.y], uv, z, alpha: 1.0 });
        let stamp = [self.id, self.version * 2 + self.smooth as u64];
        match dl.images.last_mut() {
            Some(last) if last.stamp == stamp && last.first + last.count == dl.image_rects.len() as u32 - 1 => last.count += 1,
            _ => dl.images.push(ImageDraw { bind: self.bind[self.smooth as usize].clone(), stamp, first: dl.image_rects.len() as u32 - 1, count: 1 }),
        }
    }
}

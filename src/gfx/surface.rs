//! Surfaces: grids of points kept on the GPU as a texture of positions.

use super::color::{self, Color};
use super::draw::SurfaceDraw;
use super::gpu::next_id;
use super::Gfx;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;

#[derive(Clone, Copy, Pod, Zeroable, PartialEq)]
#[repr(C)]
struct SurfU {
    model: [[f32; 4]; 4],
    dims: [f32; 4],
    color: [f32; 4],
    cmap: [f32; 4],
    iso: [f32; 4],
    shade: [f32; 4],
    line_color: [f32; 4],
}

/// How a [`Surface`] is coloured and lit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceStyle {
    pub model: Mat4,
    pub color: Color,
    pub opacity: f32,
    /// Colour by the per-vertex scalar through this map.
    pub cmap: Option<color::Cmap>,
    pub range: [f32; 2],
    /// Spacing between contour lines of the scalar; 0 turns them off.
    pub contour: f32,
    pub contour_alpha: f32,
    /// Draw contour lines only where the scalar is within this distance of zero (0:
    /// everywhere). With a reach of half the spacing the only line is the one at zero:
    /// a boundary. With one and a half spacings there is one more on each side, fainter.
    pub contour_reach: f32,
    /// Number of parameter lines along u and v; 0 turns them off.
    pub grid: [f32; 2],
    pub grid_alpha: f32,
    pub line_color: Color,
    /// 0 = flat colour (heat maps), 1 = fully lit.
    pub lit: f32,
    pub specular: f32,
    pub rim: f32,
    pub wrap: [bool; 2],
}

impl Default for SurfaceStyle {
    fn default() -> Self {
        SurfaceStyle {
            model: Mat4::IDENTITY,
            color: Color::hex(0x4F8BFF),
            opacity: 1.0,
            cmap: Some(color::Cmap::Viridis),
            range: [0.0, 1.0],
            contour: 0.0,
            contour_alpha: 0.5,
            contour_reach: 0.0,
            grid: [0.0, 0.0],
            grid_alpha: 0.35,
            line_color: Color::hex(0x0A0D14),
            lit: 1.0,
            specular: 0.35,
            rim: 0.5,
            wrap: [false, false],
        }
    }
}

/// A grid surface stored as a texture of positions. Fill `data` (xyz + scalar per
/// vertex, row-major, `nx` wide) and draw it; only changed data is uploaded.
///
/// A vertex with a number that is not finite is a hole: the cells round it are not
/// drawn. That is how to show a function where it has no value (`sqrt(x)` left of
/// zero, a pole): give it NaN, or [`Surface::HOLE`], and take the colour range over
/// the vertices that do have one. (`model` should turn and scale evenly, or stretch a
/// flat sheet: the shader lights the surface with normals carried through it as they
/// are, which a lopsided stretch of a curved surface would tilt.)
pub struct Surface {
    nx: u32,
    ny: u32,
    data: Vec<[f32; 4]>,
    pub style: SurfaceStyle,
    tex: wgpu::Texture,
    ubo: wgpu::Buffer,
    bind: wgpu::BindGroup,
    dirty: bool,
    last_u: Option<SurfU>,
    id: u64,
    version: u64,
}

impl Surface {
    /// A vertex that is not there (what a vertex with a non-finite number becomes when
    /// the surface is drawn; the shader knows it by its last number).
    pub const HOLE: [f32; 4] = [0.0, 0.0, 0.0, 3.0e38];

    pub fn new(gfx: &Gfx, nx: u32, ny: u32) -> Self {
        let nx = nx.clamp(2, gfx.max_dim);
        let ny = ny.clamp(2, gfx.max_dim);
        let tex = gfx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("surface positions"),
            size: wgpu::Extent3d { width: nx, height: ny, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
        let ubo = gfx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("surface"),
            size: std::mem::size_of::<SurfU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = gfx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("surface"),
            layout: &gfx.layout_surf,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ubo.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
            ],
        });
        Surface {
            nx,
            ny,
            data: vec![[0.0; 4]; (nx * ny) as usize],
            style: SurfaceStyle::default(),
            tex,
            ubo,
            bind,
            dirty: true,
            last_u: None,
            id: next_id(),
            version: 0,
        }
    }

    /// How many vertices across and down.
    pub fn size(&self) -> (usize, usize) {
        (self.nx as usize, self.ny as usize)
    }

    /// Mutable vertex data; marks the surface for upload.
    pub fn data_mut(&mut self) -> &mut [[f32; 4]] {
        self.dirty = true;
        &mut self.data
    }

    pub fn data(&self) -> &[[f32; 4]] {
        &self.data
    }

    /// Fill every vertex from its parameters `(u, v)` in 0..1.
    pub fn fill(&mut self, mut f: impl FnMut(f32, f32) -> [f32; 4]) {
        let (nx, ny) = (self.nx as usize, self.ny as usize);
        let (du, dv) = (1.0 / (nx - 1) as f32, 1.0 / (ny - 1) as f32);
        for j in 0..ny {
            let v = j as f32 * dv;
            let row = &mut self.data[j * nx..(j + 1) * nx];
            for (i, out) in row.iter_mut().enumerate() {
                *out = f(i as f32 * du, v);
            }
        }
        self.dirty = true;
    }

    /// Graph of `z = f(x, y)` over a rectangle, coloured by height.
    pub fn fill_height(&mut self, min: glam::Vec2, max: glam::Vec2, mut f: impl FnMut(f32, f32) -> f32) {
        self.fill(|u, v| {
            let x = min.x + (max.x - min.x) * u;
            let y = min.y + (max.y - min.y) * v;
            let z = f(x, y);
            [x, y, z, z]
        });
    }

    pub(crate) fn prepare(&mut self, gfx: &Gfx) -> SurfaceDraw {
        if self.dirty {
            // nothing that is not finite goes to the GPU: such a vertex becomes a hole
            for v in self.data.iter_mut().filter(|v| !v.iter().all(|x| x.is_finite())) {
                *v = Self::HOLE;
            }
            super::gpu::sent(self.data.len() * 16);
            gfx.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &self.tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                bytemuck::cast_slice(&self.data),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.nx * 16), rows_per_image: None },
                wgpu::Extent3d { width: self.nx, height: self.ny, depth_or_array_layers: 1 },
            );
            self.dirty = false;
            self.version += 1;
        }
        let s = &self.style;
        let base = s.color.to_linear_rgba();
        let b = |x: bool| if x { 1.0 } else { 0.0 };
        let u = SurfU {
            model: s.model.to_cols_array_2d(),
            dims: [self.nx as f32, self.ny as f32, b(s.wrap[0]), b(s.wrap[1])],
            color: [base[0], base[1], base[2], s.opacity],
            cmap: [s.cmap.map_or(-1.0, |c| c as u8 as f32), s.range[0], s.range[1], s.contour_reach],
            iso: [s.contour, s.contour_alpha, s.grid[0], s.grid[1]],
            shade: [s.lit, s.specular, s.rim, s.grid_alpha],
            line_color: s.line_color.to_linear_rgba(),
        };
        if self.last_u != Some(u) {
            gfx.queue.write_buffer(&self.ubo, 0, bytemuck::bytes_of(&u));
            self.last_u = Some(u);
            self.version += 1;
        }
        SurfaceDraw { bind: self.bind.clone(), verts: 6 * (self.nx - 1) * (self.ny - 1), opaque: s.opacity >= 0.999, stamp: [self.id, self.version] }
    }
}

#[cfg(test)]
mod tests {
    use super::Surface;

    /// The shader tells a hole by `w > 1e37` (surface.wgsl, `hole`): the marker has to
    /// be that, and finite, and no height or colour value a scene would use.
    #[test]
    fn the_hole_marker_is_what_the_shader_looks_for() {
        assert!(Surface::HOLE.iter().all(|x| x.is_finite()) && Surface::HOLE[3] > 1.0e37);
        assert!(include_str!("shaders/surface.wgsl").contains("p.w > 1.0e37"));
    }
}

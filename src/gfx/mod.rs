//! The renderer.
//!
//! A frame is one multisampled HDR pass (background, surfaces, fills, spheres, grid,
//! lines, points) followed by a bloom chain and a tone-mapping composite into an 8-bit
//! texture that egui displays. Everything a scene draws is either a retained GPU object
//! (surfaces, trails, images) or an instanced batch uploaded with a single buffer write.
//!
//! Where things are: this file holds the renderer's state and draws the frame;
//! [`draw`] is the list of things to draw that scenes fill in; `surface`, `image` and
//! `trails` are the retained objects; `pipelines` builds the shaders' pipelines,
//! `targets` the textures a frame is drawn into, and `gpu` the small buffer helpers.

pub mod camera;
pub mod color;
pub mod draw;
mod gpu;
mod image;
mod pipelines;
mod surface;
mod targets;
mod trails;

pub use image::Image;
pub use surface::{Surface, SurfaceStyle};
pub use trails::TrailSet;

use bytemuck::{Pod, Zeroable};
use camera::Camera;
use color::Color;
use draw::*;
use glam::Vec3;
pub use gpu::take_sent;
use gpu::{Batch, DynBuf, Fingerprint};
use pipelines::{build_pipelines, sampler_entry, texture_entry, uniform_entry, Pipelines};
use std::sync::Arc;
use targets::Targets;

pub type EguiRenderer = Arc<egui::mutex::RwLock<egui_wgpu::Renderer>>;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const LDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// What the scene is drawn into where 16-bit floats are not to be had, or not to be
/// afforded: 8 bits, stored gamma-encoded (the GPU converts on the way in and out), so
/// that the dark end of the range, where the backgrounds are, keeps its steps fine.
/// Stored as linear light, a dark gradient shows as rings.
const PLAIN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const MAX_BLOOM_LEVELS: usize = 6;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    cam_right: [f32; 4],
    cam_up: [f32; 4],
    cam_fwd: [f32; 4],
    viewport: [f32; 4],
    misc: [f32; 4],
    light: [f32; 4],
    bg0: [f32; 4],
    bg1: [f32; 4],
    grid: [f32; 4],
    grid_color: [f32; 4],
    focus: [f32; 4],
}

impl Globals {
    fn new(cam: &Camera, settings: &ViewSettings, size: [u32; 2], px_scale: f32) -> Self {
        let (w, h) = (size[0] as f32, size[1] as f32);
        // key light: over the viewer's left shoulder, so shading reads well from any angle
        let light = if cam.is_flat() {
            Vec3::new(-0.4, 0.5, 0.75).normalize()
        } else {
            (-cam.fwd * 0.62 - cam.right * 0.45 + cam.up * 0.55 + Vec3::Z * 0.35).normalize()
        };
        let fade = if cam.is_flat() { 1e30 } else { cam.dist() * 7.0 };
        let grid_on = if settings.grid { 1.0 } else { 0.0 };
        Globals {
            view_proj: cam.view_proj.to_cols_array_2d(),
            inv_view_proj: cam.inv_view_proj.to_cols_array_2d(),
            view: cam.view.to_cols_array_2d(),
            proj: cam.proj.to_cols_array_2d(),
            cam_pos: cam.eye.extend(if cam.is_flat() { 1.0 } else { 0.0 }).to_array(),
            cam_right: cam.right.extend(0.0).to_array(),
            cam_up: cam.up.extend(0.0).to_array(),
            cam_fwd: cam.fwd.extend(0.0).to_array(),
            viewport: [w, h, 1.0 / w, 1.0 / h],
            misc: [0.0, px_scale, cam.near, 0.0],
            light: light.extend(1.0).to_array(),
            bg0: settings.bg_center.to_linear_rgba(),
            bg1: settings.bg_edge.to_linear_rgba(),
            grid: [fade, settings.grid_alpha * grid_on, settings.axis_alpha * grid_on, 13.0 * px_scale],
            grid_color: settings.grid_color.to_linear_rgba(),
            focus: cam.target().extend(0.0).to_array(),
        }
    }
}

#[derive(Clone, Copy, Pod, Zeroable, PartialEq)]
#[repr(C)]
struct LineU {
    ring: [f32; 4],
    style: [f32; 4],
    dash: [f32; 4],
    tint: [f32; 4],
}

impl LineU {
    fn new(s: &LineStyle, ring: [f32; 4]) -> Self {
        LineU { ring, style: [s.width_scale, s.intensity, s.depth_bias, s.fade_power], dash: [s.dash, s.duty, s.taper, 0.0], tint: s.tint }
    }
}

#[derive(Clone, Copy, Pod, Zeroable, PartialEq)]
#[repr(C)]
struct PointU {
    params: [f32; 4],
    outline: [f32; 4],
}

impl PointU {
    fn new(s: &PointStyle) -> Self {
        PointU { params: [s.size_scale, s.intensity, s.outline_px, if s.world_size { 1.0 } else { 0.0 }], outline: s.outline.to_linear_rgba() }
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
struct PostU {
    a: [f32; 4],
    b: [f32; 4],
}

/// Look and quality settings shared by all scenes.
#[derive(Clone, Copy, Debug)]
pub struct ViewSettings {
    pub bloom: f32,
    pub bloom_threshold: f32,
    pub exposure: f32,
    pub vignette: f32,
    pub grid: bool,
    pub grid_alpha: f32,
    pub axis_alpha: f32,
    pub bg_center: Color,
    pub bg_edge: Color,
    pub grid_color: Color,
    /// Render resolution relative to the display (1 = native).
    pub render_scale: f32,
    /// Lower the render resolution automatically when frames take too long.
    pub adaptive: bool,
}

impl Default for ViewSettings {
    fn default() -> Self {
        ViewSettings {
            bloom: 0.55,
            bloom_threshold: 1.0,
            exposure: 1.0,
            vignette: 0.28,
            grid: true,
            grid_alpha: 0.16,
            axis_alpha: 0.42,
            bg_center: Color::hex(0x161B2A),
            bg_edge: Color::hex(0x07090F),
            grid_color: Color::hex(0xA9B8D8),
            render_scale: 1.0,
            adaptive: true,
        }
    }
}

pub struct Gfx {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    /// What is doing the drawing, for the statistics overlay.
    pub adapter: String,
    egui_renderer: EguiRenderer,
    hdr_format: wgpu::TextureFormat,
    samples: u32,
    /// The best this device can do: what `hdr_format` and `samples` are unless the
    /// picture is being drawn plainly (see [`Gfx::set_plain`]).
    best: (wgpu::TextureFormat, u32),
    plain: bool,
    /// No GPU is doing the drawing: a software rasteriser is.
    software: bool,
    max_dim: u32,

    layout_globals: wgpu::BindGroupLayout,
    layout_line: wgpu::BindGroupLayout,
    layout_surf: wgpu::BindGroupLayout,
    layout_image: wgpu::BindGroupLayout,
    layout_post: wgpu::BindGroupLayout,
    pipes: Pipelines,

    globals: wgpu::Buffer,
    bind_globals: wgpu::BindGroup,
    post_u: wgpu::Buffer,
    sampler: wgpu::Sampler,
    sampler_nearest: wgpu::Sampler,

    image_rects: DynBuf,
    lines: [Batch<LineU>; 4],
    points: [Batch<PointU>; 4],
    tris: [DynBuf; 2],
    /// GPU state for the draw list's custom batches, grown on demand and reused.
    extra_lines: Vec<Batch<LineU>>,
    extra_points: Vec<Batch<PointU>>,

    targets: Option<Targets>,
    tex_id: Option<egui::TextureId>,
    last_frame: u64,
    /// Frames actually drawn, and frames found identical to the previous one and reused.
    pub drawn: u64,
    pub reused: u64,
}

impl Gfx {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, adapter: &wgpu::Adapter, egui_renderer: EguiRenderer) -> Self {
        // Prefer 16-bit float HDR with 4x MSAA; degrade gracefully where unsupported.
        let hdr = adapter.get_texture_format_features(wgpu::TextureFormat::Rgba16Float);
        let hdr_ok = hdr.allowed_usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) && hdr.flags.contains(wgpu::TextureFormatFeatureFlags::FILTERABLE);
        let (hdr_format, samples) = if hdr_ok {
            let s = if hdr.flags.sample_count_supported(4) { 4 } else { 1 };
            (wgpu::TextureFormat::Rgba16Float, s)
        } else {
            (PLAIN_FORMAT, 4)
        };
        log::info!("render format {hdr_format:?}, {samples}x MSAA");
        // Say plainly whether a GPU is doing the drawing. Browsers do not name the
        // adapter under WebGPU, but they do say when it is their software fallback.
        let info = adapter.get_info();
        let software = info.device_type == wgpu::DeviceType::Cpu;
        let kind = match info.device_type {
            wgpu::DeviceType::Cpu => "SOFTWARE rendering (no GPU)",
            wgpu::DeviceType::IntegratedGpu => "integrated GPU",
            wgpu::DeviceType::DiscreteGpu => "discrete GPU",
            wgpu::DeviceType::VirtualGpu => "virtual GPU",
            wgpu::DeviceType::Other => "GPU",
        };
        let name: String = info.name.chars().take(60).collect();
        let adapter = format!("{kind}, {:?}{}{name}", info.backend, if name.is_empty() { "" } else { "\n" });
        let max_dim = device.limits().max_texture_dimension_2d;

        let vf = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let layout_globals = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[uniform_entry(0, vf), texture_entry(1, vf, true), sampler_entry(2)],
        });
        let layout_line = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("per-batch"), entries: &[uniform_entry(0, vf)] });
        let layout_surf = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surface"),
            entries: &[uniform_entry(0, vf), texture_entry(1, wgpu::ShaderStages::VERTEX, false)],
        });
        let layout_image = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("image"),
            entries: &[texture_entry(0, wgpu::ShaderStages::FRAGMENT, true), sampler_entry(1)],
        });
        let layout_post = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("post"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                texture_entry(1, wgpu::ShaderStages::FRAGMENT, true),
                sampler_entry(2),
                texture_entry(3, wgpu::ShaderStages::FRAGMENT, true),
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let sampler_nearest = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("nearest clamp"), ..Default::default() });

        // colour-map lookup texture
        let lut_data = color::build_lut();
        let lut_size = wgpu::Extent3d { width: color::LUT_WIDTH, height: color::Cmap::ALL.len() as u32, depth_or_array_layers: 1 };
        let lut = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("colormaps"),
            size: lut_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &lut, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &lut_data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(color::LUT_WIDTH * 4), rows_per_image: None },
            lut_size,
        );
        let lut_view = lut.create_view(&wgpu::TextureViewDescriptor::default());

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_globals = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &layout_globals,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&lut_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let post_u = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("post"),
            size: std::mem::size_of::<PostU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pipes = build_pipelines(&device, hdr_format, samples, &layout_globals, &layout_line, &layout_surf, &layout_image, &layout_post);

        let lines = ["lines", "dashed", "glow lines", "overlay lines"].map(|label| Batch::new(&device, &layout_line, label));
        let points = ["discs", "spheres", "glows", "overlay discs"].map(|label| Batch::new(&device, &layout_line, label));

        Gfx {
            device,
            queue,
            adapter,
            egui_renderer,
            hdr_format,
            samples,
            best: (hdr_format, samples),
            plain: false,
            software,
            max_dim,
            layout_globals,
            layout_line,
            layout_surf,
            layout_image,
            layout_post,
            pipes,
            globals,
            bind_globals,
            post_u,
            sampler,
            sampler_nearest,
            image_rects: DynBuf::new("image rectangles"),
            lines,
            points,
            tris: [DynBuf::new("tris"), DynBuf::new("tris alpha")],
            extra_lines: Vec::new(),
            extra_points: Vec::new(),
            targets: None,
            tex_id: None,
            last_frame: 0,
            drawn: 0,
            reused: 0,
        }
    }

    pub fn msaa_samples(&self) -> u32 {
        self.samples
    }

    /// Whether a software rasteriser is doing the drawing (no GPU).
    pub fn software(&self) -> bool {
        self.software
    }

    pub fn is_plain(&self) -> bool {
        self.plain
    }

    /// Draw plainly, or as well as the device can. Plainly: one sample a pixel where
    /// there were four, 8 bits a channel where there were 16, and no glow, which is
    /// between a half and a fifth of the work for whatever fills the pixels (measured
    /// with a software rasteriser: 260 to 490 ms a frame became 80 to 140). Shapes
    /// still have smooth edges where the shaders compute them. Changing over rebuilds
    /// the pipelines and the targets, once. Returns whether anything changed.
    pub fn set_plain(&mut self, plain: bool) -> bool {
        if plain == self.plain {
            return false;
        }
        self.plain = plain;
        let (format, samples) = if plain { (PLAIN_FORMAT, 1) } else { self.best };
        if (format, samples) != (self.hdr_format, self.samples) {
            (self.hdr_format, self.samples) = (format, samples);
            self.pipes = build_pipelines(
                &self.device,
                format,
                samples,
                &self.layout_globals,
                &self.layout_line,
                &self.layout_surf,
                &self.layout_image,
                &self.layout_post,
            );
        }
        self.targets = None;
        true
    }

    pub fn is_hdr(&self) -> bool {
        self.hdr_format == wgpu::TextureFormat::Rgba16Float
    }

    /// The egui texture holding the last rendered frame.
    pub fn texture_id(&self) -> Option<egui::TextureId> {
        self.tex_id
    }

    /// Clamp a requested target size to what the device supports.
    pub fn clamp_size(&self, w: u32, h: u32) -> [u32; 2] {
        let m = self.max_dim.min(8192);
        let mut size = [w.max(8), h.max(8)];
        let biggest = size[0].max(size[1]);
        if biggest > m {
            let k = m as f32 / biggest as f32;
            size = [((size[0] as f32 * k) as u32).max(8), ((size[1] as f32 * k) as u32).max(8)];
        }
        size
    }

    /// Render the draw list into the display texture. `px_scale` is target pixels per
    /// logical pixel, so line widths and point sizes stay constant on any display.
    ///
    /// Returns false, having done no GPU work at all, when the frame is identical to
    /// the previous one: a still scene costs nothing while the pointer moves over it.
    pub fn render(&mut self, dl: &mut DrawList, cam: &Camera, settings: &ViewSettings, size: [u32; 2], px_scale: f32) -> bool {
        let fresh_targets = self.ensure_targets(size);
        let plainly = ViewSettings { bloom: 0.0, ..*settings };
        let settings = if self.plain { &plainly } else { settings };
        let globals = Globals::new(cam, settings, size, px_scale);
        let threshold = if self.is_hdr() { settings.bloom_threshold } else { 0.82 };
        let post = PostU { a: [threshold, 0.35, settings.bloom, settings.exposure], b: [settings.vignette, 1.0, 0.0, 0.0] };

        // Has anything that affects the image changed since the last frame?
        dl.finish();
        let print = fingerprint(dl, &globals, &post, settings);
        if !fresh_targets && print == self.last_frame {
            self.reused += 1;
            return false;
        }
        self.last_frame = print;
        self.drawn += 1;

        self.queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
        self.queue.write_buffer(&self.post_u, 0, bytemuck::bytes_of(&post));
        self.upload(dl);
        let mut enc = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        self.scene_pass(&mut enc, dl, settings);
        self.post_passes(&mut enc, settings);
        self.queue.submit([enc.finish()]);
        true
    }

    /// Send this frame's batches to the GPU: one buffer write each.
    fn upload(&mut self, dl: &DrawList) {
        let (device, queue) = (&self.device, &self.queue);
        for (gpu, b) in self.lines.iter_mut().zip(dl.built_in_lines()) {
            gpu.upload(device, queue, b.instances(), bytemuck::cast_slice(&b.pts), LineU::new(&b.style, [0.0; 4]));
        }
        for (gpu, b) in self.points.iter_mut().zip(dl.built_in_points()) {
            gpu.upload(device, queue, b.pts.len() as u32, bytemuck::cast_slice(&b.pts), PointU::new(&b.style));
        }
        self.tris[0].upload(device, queue, bytemuck::cast_slice(&dl.tris.v));
        self.tris[1].upload(device, queue, bytemuck::cast_slice(&dl.tris_alpha.v));
        self.image_rects.upload(device, queue, bytemuck::cast_slice(&dl.image_rects));

        // custom batches: one pooled buffer and uniform block each
        while self.extra_lines.len() < dl.extra_lines.len() {
            self.extra_lines.push(Batch::new(device, &self.layout_line, "custom lines"));
        }
        for (gpu, (_, b)) in self.extra_lines.iter_mut().zip(&dl.extra_lines) {
            gpu.upload(device, queue, b.instances(), bytemuck::cast_slice(&b.pts), LineU::new(&b.style, [0.0; 4]));
        }
        while self.extra_points.len() < dl.extra_points.len() {
            self.extra_points.push(Batch::new(device, &self.layout_line, "custom points"));
        }
        for (gpu, (_, b)) in self.extra_points.iter_mut().zip(&dl.extra_points) {
            gpu.upload(device, queue, b.pts.len() as u32, bytemuck::cast_slice(&b.pts), PointU::new(&b.style));
        }
    }

    /// The one pass that draws the scene. In order: the background; opaque surfaces,
    /// fills and spheres; the grid; pictures; translucent surfaces and fills; lines
    /// and trails; discs; whatever glows; and last the overlays, which ignore depth.
    fn scene_pass(&self, enc: &mut wgpu::CommandEncoder, dl: &DrawList, settings: &ViewSettings) {
        let t = self.targets.as_ref().expect("a frame is drawn into targets that exist");
        let pipes = &self.pipes;
        let (view, resolve) = match &t.msaa {
            Some(m) => (m, Some(&t.hdr)),
            None => (&t.hdr, None),
        };
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: resolve,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: if resolve.is_some() { wgpu::StoreOp::Discard } else { wgpu::StoreOp::Store },
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &t.depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        let rp = &mut rp;
        rp.set_bind_group(0, &self.bind_globals, &[]);
        let [lines, dashed, glow_lines, overlay] = dl.built_in_lines();
        let [discs, spheres, glows, overlay_discs] = dl.built_in_points();
        // the batches a scene asked for with a style of its own, of one kind
        let extra_points = |rp: &mut wgpu::RenderPass<'_>, pipe: &wgpu::RenderPipeline, kind: PointKind| {
            for (gpu, (_, b)) in self.extra_points.iter().zip(&dl.extra_points).filter(|(_, (k, _))| *k == kind) {
                draw_points(rp, pipe, gpu, b.pts.len() as u32);
            }
        };
        let extra_lines = |rp: &mut wgpu::RenderPass<'_>, pipe: &wgpu::RenderPipeline, kind: LineKind| {
            for (gpu, (_, b)) in self.extra_lines.iter().zip(&dl.extra_lines).filter(|(_, (k, _))| *k == kind) {
                draw_lines(rp, pipe, gpu, b.instances());
            }
        };
        let surfaces = |rp: &mut wgpu::RenderPass<'_>, pipe: &wgpu::RenderPipeline, opaque: bool| {
            if dl.surfaces.iter().any(|s| s.opaque == opaque) {
                rp.set_pipeline(pipe);
                for s in dl.surfaces.iter().filter(|s| s.opaque == opaque) {
                    rp.set_bind_group(1, &s.bind, &[]);
                    rp.draw(0..s.verts, 0..1);
                }
            }
        };
        let fills = |rp: &mut wgpu::RenderPass<'_>, pipe: &wgpu::RenderPipeline, buf: &DynBuf, batch: &TriBatch| {
            if let (Some(buf), false) = (&buf.buf, batch.v.is_empty()) {
                rp.set_pipeline(pipe);
                rp.set_vertex_buffer(0, buf.slice(..));
                rp.draw(0..batch.v.len() as u32, 0..1);
            }
        };

        rp.set_pipeline(&pipes.bg);
        rp.draw(0..3, 0..1);

        // opaque geometry
        surfaces(rp, &pipes.surface, true);
        fills(rp, &pipes.tris, &self.tris[0], &dl.tris);
        draw_points(rp, &pipes.spheres, &self.points[1], spheres.pts.len() as u32);
        extra_points(rp, &pipes.spheres, PointKind::Sphere);

        if settings.grid {
            rp.set_pipeline(&pipes.grid);
            rp.draw(0..3, 0..1);
        }

        // pictures: over the grid, under everything drawn with lines and points
        if let (Some(rects), false) = (&self.image_rects.buf, dl.images.is_empty()) {
            rp.set_pipeline(&pipes.images);
            rp.set_vertex_buffer(0, rects.slice(..));
            for i in &dl.images {
                rp.set_bind_group(1, &i.bind, &[]);
                rp.draw(0..4, i.first..i.first + i.count);
            }
        }

        // translucent geometry
        surfaces(rp, &pipes.surface_alpha, false);
        fills(rp, &pipes.tris_alpha, &self.tris[1], &dl.tris_alpha);

        draw_lines(rp, &pipes.lines, &self.lines[0], lines.instances());
        draw_lines(rp, &pipes.lines, &self.lines[1], dashed.instances());
        extra_lines(rp, &pipes.lines, LineKind::Blend);

        // trails: same shader, windows one row (= `cols` points) apart
        for additive in [false, true] {
            for tr in dl.trails.iter().filter(|t| t.additive == additive) {
                rp.set_pipeline(if additive { &pipes.lines_add } else { &pipes.lines });
                rp.set_bind_group(1, &tr.bind, &[]);
                let stride = tr.cols as u64 * 20;
                for k in 0..4u64 {
                    rp.set_vertex_buffer(k as u32, tr.buf.slice(k * stride..));
                }
                rp.draw(0..4, 0..tr.cols * tr.rows);
            }
        }

        draw_points(rp, &pipes.discs, &self.points[0], discs.pts.len() as u32);
        extra_points(rp, &pipes.discs, PointKind::Disc);
        draw_lines(rp, &pipes.lines_add, &self.lines[2], glow_lines.instances());
        extra_lines(rp, &pipes.lines_add, LineKind::Additive);
        draw_points(rp, &pipes.glows, &self.points[2], glows.pts.len() as u32);
        extra_points(rp, &pipes.glows, PointKind::Glow);
        draw_lines(rp, &pipes.lines_overlay, &self.lines[3], overlay.instances());
        extra_lines(rp, &pipes.lines_overlay, LineKind::Overlay);
        draw_points(rp, &pipes.discs_overlay, &self.points[3], overlay_discs.pts.len() as u32);
        extra_points(rp, &pipes.discs_overlay, PointKind::Overlay);
    }

    /// What follows the scene: the bloom (a prefilter and a chain of ever smaller
    /// copies, added back together on the way up), then tone mapping into the 8-bit
    /// picture that is shown.
    fn post_passes(&self, enc: &mut wgpu::CommandEncoder, settings: &ViewSettings) {
        let t = self.targets.as_ref().expect("a frame is drawn into targets that exist");
        let clear = wgpu::LoadOp::Clear(wgpu::Color::BLACK);
        // with the glow turned off the composite adds nothing from these, so they are
        // not drawn at all: about a dozen full-screen passes saved
        if settings.bloom > 0.0 {
            full_screen(enc, &t.bloom[0], &self.pipes.prefilter, &t.bind_prefilter, clear);
            for i in 0..t.bind_down.len() {
                full_screen(enc, &t.bloom[i + 1], &self.pipes.down, &t.bind_down[i], clear);
            }
            for i in (0..t.bind_up.len()).rev() {
                full_screen(enc, &t.bloom[i], &self.pipes.up, &t.bind_up[i], wgpu::LoadOp::Load);
            }
        }
        full_screen(enc, &t.ldr, &self.pipes.composite, &t.bind_composite, clear);
    }
}

/// A fingerprint of everything that determines the picture: the camera and settings,
/// every batch's contents and style, and the identity and version of every retained
/// object drawn.
fn fingerprint(dl: &DrawList, globals: &Globals, post: &PostU, settings: &ViewSettings) -> u64 {
    let mut fp = Fingerprint::new();
    fp.bytes(bytemuck::bytes_of(globals));
    fp.bytes(bytemuck::bytes_of(post));
    fp.word(settings.grid as u64);
    for b in dl.built_in_lines().into_iter().chain(dl.extra_lines.iter().map(|(_, b)| b)) {
        fp.bytes(bytemuck::cast_slice(&b.pts));
        fp.bytes(bytemuck::bytes_of(&LineU::new(&b.style, [0.0; 4])));
    }
    for (k, _) in &dl.extra_lines {
        fp.word(*k as u64);
    }
    for b in dl.built_in_points().into_iter().chain(dl.extra_points.iter().map(|(_, b)| b)) {
        fp.bytes(bytemuck::cast_slice(&b.pts));
        fp.bytes(bytemuck::bytes_of(&PointU::new(&b.style)));
    }
    for (k, _) in &dl.extra_points {
        fp.word(*k as u64);
    }
    fp.bytes(bytemuck::cast_slice(&dl.tris.v));
    fp.bytes(bytemuck::cast_slice(&dl.tris_alpha.v));
    for s in &dl.surfaces {
        fp.word(s.stamp[0]);
        fp.word(s.stamp[1]);
        fp.word(s.opaque as u64);
    }
    fp.word(u64::MAX);
    fp.bytes(bytemuck::cast_slice(&dl.image_rects));
    for i in &dl.images {
        fp.word(i.stamp[0]);
        fp.word(i.stamp[1]);
        fp.word(i.first as u64 | (i.count as u64) << 32);
    }
    fp.word(u64::MAX);
    for t in &dl.trails {
        fp.word(t.stamp[0]);
        fp.word(t.stamp[1]);
        fp.word(t.additive as u64);
    }
    fp.0
}

/// Points are instanced quads: `count` of them from the batch's buffer.
fn draw_points(rp: &mut wgpu::RenderPass<'_>, pipe: &wgpu::RenderPipeline, batch: &Batch<PointU>, count: u32) {
    if let (Some(buf), true) = (&batch.buf.buf, count > 0) {
        rp.set_pipeline(pipe);
        rp.set_bind_group(1, &batch.uni.bind, &[]);
        rp.set_vertex_buffer(0, buf.slice(..));
        rp.draw(0..4, 0..count);
    }
}

/// So are lines. A line batch binds its buffer of points four times, one point apart,
/// so that every instance sees a segment and its two neighbours.
fn draw_lines(rp: &mut wgpu::RenderPass<'_>, pipe: &wgpu::RenderPipeline, batch: &Batch<LineU>, count: u32) {
    if let (Some(buf), true) = (&batch.buf.buf, count > 0) {
        rp.set_pipeline(pipe);
        rp.set_bind_group(1, &batch.uni.bind, &[]);
        for slot in 0..4u64 {
            rp.set_vertex_buffer(slot as u32, buf.slice(slot * 20..));
        }
        rp.draw(0..4, 0..count);
    }
}

/// One triangle over the whole of `target`, through a post-processing shader.
fn full_screen(
    enc: &mut wgpu::CommandEncoder,
    target: &wgpu::TextureView,
    pipe: &wgpu::RenderPipeline,
    bind: &wgpu::BindGroup,
    load: wgpu::LoadOp<wgpu::Color>,
) {
    let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("post"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    rp.set_pipeline(pipe);
    rp.set_bind_group(0, bind, &[]);
    rp.draw(0..3, 0..1);
}

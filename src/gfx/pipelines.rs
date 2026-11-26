//! The render pipelines: one per kind of thing drawn, all built once at start-up.

use super::{DEPTH_FORMAT, LDR_FORMAT};

pub(super) struct Pipelines {
    pub(super) bg: wgpu::RenderPipeline,
    pub(super) grid: wgpu::RenderPipeline,
    pub(super) surface: wgpu::RenderPipeline,
    pub(super) surface_alpha: wgpu::RenderPipeline,
    pub(super) tris: wgpu::RenderPipeline,
    pub(super) tris_alpha: wgpu::RenderPipeline,
    pub(super) images: wgpu::RenderPipeline,
    pub(super) lines: wgpu::RenderPipeline,
    pub(super) lines_add: wgpu::RenderPipeline,
    pub(super) lines_overlay: wgpu::RenderPipeline,
    pub(super) discs: wgpu::RenderPipeline,
    pub(super) discs_overlay: wgpu::RenderPipeline,
    pub(super) spheres: wgpu::RenderPipeline,
    pub(super) glows: wgpu::RenderPipeline,
    pub(super) prefilter: wgpu::RenderPipeline,
    pub(super) down: wgpu::RenderPipeline,
    pub(super) up: wgpu::RenderPipeline,
    pub(super) composite: wgpu::RenderPipeline,
}

const PREMUL: wgpu::BlendState = wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING;
const ADDITIVE: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
    alpha: wgpu::BlendComponent { src_factor: wgpu::BlendFactor::Zero, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add },
};

#[derive(Clone, Copy)]
enum Depth {
    None,
    Test,
    Write,
}

pub(super) fn uniform_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

pub(super) fn texture_entry(binding: u32, vis: wgpu::ShaderStages, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

pub(super) fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_pipelines(
    device: &wgpu::Device,
    hdr: wgpu::TextureFormat,
    samples: u32,
    layout_globals: &wgpu::BindGroupLayout,
    layout_batch: &wgpu::BindGroupLayout,
    layout_surf: &wgpu::BindGroupLayout,
    layout_image: &wgpu::BindGroupLayout,
    layout_post: &wgpu::BindGroupLayout,
) -> Pipelines {
    let module =
        |label: &str, src: &str| device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) });
    const COMMON: &str = include_str!("shaders/common.wgsl");
    let with_common = |src: &str| format!("{COMMON}\n{src}");
    let m_grid = module("grid", &with_common(include_str!("shaders/grid.wgsl")));
    let m_surf = module("surface", &with_common(include_str!("shaders/surface.wgsl")));
    let m_tris = module("tris", &with_common(include_str!("shaders/tris.wgsl")));
    let m_image = module("image", &with_common(include_str!("shaders/image.wgsl")));
    let m_lines = module("lines", &with_common(include_str!("shaders/lines.wgsl")));
    let m_points = module("points", &with_common(include_str!("shaders/points.wgsl")));
    let m_post = module("post", include_str!("shaders/post.wgsl"));

    let pl = |label: &str, groups: &[Option<&wgpu::BindGroupLayout>]| {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: groups, immediate_size: 0 })
    };
    let pl_globals = pl("globals", &[Some(layout_globals)]);
    let pl_batch = pl("batch", &[Some(layout_globals), Some(layout_batch)]);
    let pl_surf = pl("surface", &[Some(layout_globals), Some(layout_surf)]);
    let pl_image = pl("image", &[Some(layout_globals), Some(layout_image)]);
    let pl_post = pl("post", &[Some(layout_post)]);

    struct Spec<'a> {
        label: &'a str,
        layout: &'a wgpu::PipelineLayout,
        module: &'a wgpu::ShaderModule,
        vs: &'a str,
        fs: &'a str,
        buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
        topology: wgpu::PrimitiveTopology,
        blend: Option<wgpu::BlendState>,
        depth: Depth,
        format: wgpu::TextureFormat,
        samples: u32,
        a2c: bool,
        has_depth: bool,
    }
    let make = |s: Spec| {
        let depth_stencil = s.has_depth.then(|| {
            let (write, compare) = match s.depth {
                Depth::None => (false, wgpu::CompareFunction::Always),
                Depth::Test => (false, wgpu::CompareFunction::LessEqual),
                Depth::Write => (true, wgpu::CompareFunction::LessEqual),
            };
            wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(write),
                depth_compare: Some(compare),
                stencil: Default::default(),
                bias: Default::default(),
            }
        });
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(s.label),
            layout: Some(s.layout),
            vertex: wgpu::VertexState { module: s.module, entry_point: Some(s.vs), compilation_options: Default::default(), buffers: s.buffers },
            primitive: wgpu::PrimitiveState { topology: s.topology, cull_mode: None, ..Default::default() },
            depth_stencil,
            multisample: wgpu::MultisampleState { count: s.samples, mask: !0, alpha_to_coverage_enabled: s.a2c },
            fragment: Some(wgpu::FragmentState {
                module: s.module,
                entry_point: Some(s.fs),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState { format: s.format, blend: s.blend, write_mask: wgpu::ColorWrites::ALL })],
            }),
            multiview_mask: None,
            cache: None,
        })
    };

    use wgpu::PrimitiveTopology::{TriangleList, TriangleStrip};
    use wgpu::VertexFormat::{Float32x2, Float32x3, Float32x4, Snorm8x4, Unorm8x4};
    let inst = wgpu::VertexStepMode::Instance;
    let attr = |format, offset, shader_location| wgpu::VertexAttribute { format, offset, shader_location };

    // line points: the same 20-byte layout bound in four slots
    let la = [attr(Float32x4, 0, 0)];
    let lb = [attr(Float32x4, 0, 1), attr(Unorm8x4, 16, 2)];
    let lc = [attr(Float32x4, 0, 3), attr(Unorm8x4, 16, 4)];
    let ld = [attr(Float32x4, 0, 5)];
    let line_buffers = [
        Some(wgpu::VertexBufferLayout { array_stride: 20, step_mode: inst, attributes: &la }),
        Some(wgpu::VertexBufferLayout { array_stride: 20, step_mode: inst, attributes: &lb }),
        Some(wgpu::VertexBufferLayout { array_stride: 20, step_mode: inst, attributes: &lc }),
        Some(wgpu::VertexBufferLayout { array_stride: 20, step_mode: inst, attributes: &ld }),
    ];
    let pa = [attr(Float32x4, 0, 0), attr(Unorm8x4, 16, 1)];
    let point_buffers = [Some(wgpu::VertexBufferLayout { array_stride: 20, step_mode: inst, attributes: &pa })];
    let ta = [attr(Float32x3, 0, 0), attr(Unorm8x4, 12, 1), attr(Snorm8x4, 16, 2)];
    let tri_buffers = [Some(wgpu::VertexBufferLayout { array_stride: 20, step_mode: wgpu::VertexStepMode::Vertex, attributes: &ta })];

    let ia = [attr(Float32x4, 0, 0), attr(Float32x4, 16, 1), attr(Float32x2, 32, 2)];
    let image_buffers = [Some(wgpu::VertexBufferLayout { array_stride: 40, step_mode: inst, attributes: &ia })];

    let scene = |label, layout, module, vs, fs, buffers, topology, blend, depth| Spec {
        label,
        layout,
        module,
        vs,
        fs,
        buffers,
        topology,
        blend,
        depth,
        format: hdr,
        samples,
        a2c: false,
        has_depth: true,
    };
    let post = |label, fs, blend, format| Spec {
        label,
        layout: &pl_post,
        module: &m_post,
        vs: "vs",
        fs,
        buffers: &[],
        topology: TriangleList,
        blend,
        depth: Depth::None,
        format,
        samples: 1,
        a2c: false,
        has_depth: false,
    };

    let a2c = samples > 1;
    Pipelines {
        bg: make(scene("bg", &pl_globals, &m_grid, "vs", "fs_bg", &[], TriangleList, None, Depth::None)),
        grid: make(scene("grid", &pl_globals, &m_grid, "vs", "fs_grid", &[], TriangleList, Some(PREMUL), Depth::Test)),
        surface: make(scene("surface", &pl_surf, &m_surf, "vs", "fs", &[], TriangleList, None, Depth::Write)),
        surface_alpha: make(scene("surface alpha", &pl_surf, &m_surf, "vs", "fs", &[], TriangleList, Some(PREMUL), Depth::Test)),
        tris: make(scene("tris", &pl_globals, &m_tris, "vs", "fs", &tri_buffers, TriangleList, Some(PREMUL), Depth::Write)),
        tris_alpha: make(scene("tris alpha", &pl_globals, &m_tris, "vs", "fs", &tri_buffers, TriangleList, Some(PREMUL), Depth::Test)),
        images: make(scene("images", &pl_image, &m_image, "vs", "fs", &image_buffers, TriangleStrip, Some(PREMUL), Depth::Test)),
        lines: make(scene("lines", &pl_batch, &m_lines, "vs", "fs", &line_buffers, TriangleStrip, Some(PREMUL), Depth::Test)),
        lines_add: make(scene("lines add", &pl_batch, &m_lines, "vs", "fs", &line_buffers, TriangleStrip, Some(ADDITIVE), Depth::Test)),
        lines_overlay: make(scene("lines overlay", &pl_batch, &m_lines, "vs", "fs", &line_buffers, TriangleStrip, Some(PREMUL), Depth::None)),
        discs: make(scene("discs", &pl_batch, &m_points, "vs", "fs_disc", &point_buffers, TriangleStrip, Some(PREMUL), Depth::Test)),
        discs_overlay: make(scene("discs overlay", &pl_batch, &m_points, "vs", "fs_disc", &point_buffers, TriangleStrip, Some(PREMUL), Depth::None)),
        spheres: make(Spec {
            a2c,
            ..scene(
                "spheres",
                &pl_batch,
                &m_points,
                "vs",
                "fs_sphere",
                &point_buffers,
                TriangleStrip,
                if a2c { None } else { Some(wgpu::BlendState::ALPHA_BLENDING) },
                Depth::Write,
            )
        }),
        glows: make(scene("glows", &pl_batch, &m_points, "vs", "fs_glow", &point_buffers, TriangleStrip, Some(ADDITIVE), Depth::Test)),
        prefilter: make(post("bloom prefilter", "fs_prefilter", None, hdr)),
        down: make(post("bloom down", "fs_down", None, hdr)),
        up: make(post("bloom up", "fs_up", Some(ADDITIVE), hdr)),
        composite: make(post("composite", "fs_composite", None, LDR_FORMAT)),
    }
}

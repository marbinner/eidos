//! The textures a frame is drawn into: the multisampled HDR picture and its depth,
//! the bloom chain, and the 8-bit result that egui shows. Remade when the size changes.

use super::{Gfx, DEPTH_FORMAT, LDR_FORMAT, MAX_BLOOM_LEVELS};

pub(super) struct Targets {
    pub(super) size: [u32; 2],
    pub(super) msaa: Option<wgpu::TextureView>,
    pub(super) depth: wgpu::TextureView,
    pub(super) hdr: wgpu::TextureView,
    pub(super) bloom: Vec<wgpu::TextureView>,
    pub(super) ldr: wgpu::TextureView,
    pub(super) bind_prefilter: wgpu::BindGroup,
    pub(super) bind_down: Vec<wgpu::BindGroup>,
    pub(super) bind_up: Vec<wgpu::BindGroup>,
    pub(super) bind_composite: wgpu::BindGroup,
}

impl Gfx {
    fn make_view(&self, label: &str, size: [u32; 2], format: wgpu::TextureFormat, samples: u32, sampled: bool) -> wgpu::TextureView {
        let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        if sampled {
            usage |= wgpu::TextureUsages::TEXTURE_BINDING;
        }
        self.device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn post_bind(&self, src: &wgpu::TextureView, bloom: &wgpu::TextureView) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("post"),
            layout: &self.layout_post,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.post_u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(bloom) },
            ],
        })
    }

    /// Returns true if the targets had to be (re)created.
    pub(super) fn ensure_targets(&mut self, size: [u32; 2]) -> bool {
        if self.targets.as_ref().is_some_and(|t| t.size == size) {
            return false;
        }
        let msaa = (self.samples > 1).then(|| self.make_view("msaa colour", size, self.hdr_format, self.samples, false));
        let depth = self.make_view("depth", size, DEPTH_FORMAT, self.samples, false);
        let hdr = self.make_view("hdr", size, self.hdr_format, 1, true);
        let ldr = self.make_view("ldr", size, LDR_FORMAT, 1, true);
        let mut bloom = Vec::new();
        let mut s = [size[0] / 2, size[1] / 2];
        while bloom.len() < MAX_BLOOM_LEVELS && s[0] >= 8 && s[1] >= 8 {
            bloom.push(self.make_view("bloom", s, self.hdr_format, 1, true));
            s = [s[0] / 2, s[1] / 2];
        }
        if bloom.is_empty() {
            bloom.push(self.make_view("bloom", [size[0].max(2) / 2, size[1].max(2) / 2], self.hdr_format, 1, true));
        }
        let bind_prefilter = self.post_bind(&hdr, &hdr);
        let bind_down = (0..bloom.len() - 1).map(|i| self.post_bind(&bloom[i], &bloom[i])).collect();
        let bind_up = (1..bloom.len()).map(|i| self.post_bind(&bloom[i], &bloom[i])).collect();
        let bind_composite = self.post_bind(&hdr, &bloom[0]);

        let mut r = self.egui_renderer.write();
        match self.tex_id {
            Some(id) => r.update_egui_texture_from_wgpu_texture(&self.device, &ldr, wgpu::FilterMode::Linear, id),
            None => self.tex_id = Some(r.register_native_texture(&self.device, &ldr, wgpu::FilterMode::Linear)),
        }
        drop(r);

        self.targets = Some(Targets { size, msaa, depth, hdr, bloom, ldr, bind_prefilter, bind_down, bind_up, bind_composite });
        true
    }
}

//! Small pieces the renderer is built from: identities for retained objects, the
//! fingerprint that tells a frame from the one before, and buffers that are only
//! written when there is something new to write.

use bytemuck::Pod;

/// Identity for retained GPU objects, so the renderer can recognise them between frames.
pub(super) fn next_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// A fast 64-bit fingerprint of everything that determines a frame. Not cryptographic;
/// it only has to tell "same as last frame" from "different".
pub(super) struct Fingerprint(pub(super) u64);

impl Fingerprint {
    pub(super) fn new() -> Self {
        Fingerprint(0x243F_6A88_85A3_08D3)
    }

    #[inline]
    pub(super) fn word(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    pub(super) fn bytes(&mut self, b: &[u8]) {
        self.word(b.len() as u64);
        let mut chunks = b.chunks_exact(8);
        for c in &mut chunks {
            self.word(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut tail = [0u8; 8];
            tail[..rest.len()].copy_from_slice(rest);
            self.word(u64::from_le_bytes(tail));
        }
    }
}

/// A vertex buffer that grows geometrically and is rewritten each frame.
pub(super) struct DynBuf {
    pub(super) buf: Option<wgpu::Buffer>,
    pub(super) cap: u64,
    pub(super) label: &'static str,
}

impl DynBuf {
    pub(super) fn new(label: &'static str) -> Self {
        DynBuf { buf: None, cap: 0, label }
    }

    pub(super) fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) {
        let need = bytes.len() as u64;
        if need == 0 {
            return;
        }
        if self.buf.is_none() || need > self.cap {
            self.cap = need.next_power_of_two().max(4096);
            self.buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: self.cap,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        sent(bytes.len());
        queue.write_buffer(self.buf.as_ref().unwrap(), 0, bytes);
    }
}

/// Bytes of buffers and pixels handed to the GPU since [`take_sent`] was last called
/// (for `eidos profile`: a scene that sends megabytes a frame costs a visitor's GPU
/// and memory bus time that no timing of our own code shows).
static SENT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn sent(bytes: usize) {
    SENT.fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed);
}

pub fn take_sent() -> u64 {
    SENT.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// A small uniform block that is only re-uploaded when its contents change.
pub(super) struct Uniform<T: Pod + PartialEq> {
    pub(super) buf: wgpu::Buffer,
    pub(super) bind: wgpu::BindGroup,
    pub(super) last: Option<T>,
}

impl<T: Pod + PartialEq> Uniform<T> {
    pub(super) fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, label: &str) -> Self {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<T>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
        });
        Uniform { buf, bind, last: None }
    }

    pub(super) fn set(&mut self, queue: &wgpu::Queue, value: T) {
        if self.last != Some(value) {
            queue.write_buffer(&self.buf, 0, bytemuck::bytes_of(&value));
            self.last = Some(value);
        }
    }
}

/// One batch of lines or points on the GPU: its instances and its style.
pub(super) struct Batch<U: Pod + PartialEq> {
    pub(super) buf: DynBuf,
    pub(super) uni: Uniform<U>,
}

impl<U: Pod + PartialEq> Batch<U> {
    pub(super) fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, label: &'static str) -> Self {
        Batch { buf: DynBuf::new(label), uni: Uniform::new(device, layout, label) }
    }

    /// Send this frame's instances and style (nothing at all, if there are none to draw).
    pub(super) fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, count: u32, bytes: &[u8], style: U) {
        if count > 0 {
            self.buf.upload(device, queue, bytes);
            self.uni.set(queue, style);
        }
    }
}

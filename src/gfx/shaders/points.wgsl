// Instanced point sprites: crisp discs, lit sphere impostors with correct depth,
// and soft additive glows.

struct PointU {
    params: vec4<f32>,   // size scale, intensity, outline px, 1 = sizes are world units
    outline: vec4<f32>,  // outline colour (linear) and alpha
};

@group(1) @binding(0) var<uniform> U: PointU;

struct VIn {
    @location(0) ps: vec4<f32>,
    @location(1) col: vec4<f32>,
    @builtin(vertex_index) vi: u32,
};

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) col: vec4<f32>,
    @location(2) @interpolate(flat) geo: vec4<f32>,     // radius px, radius world, view depth, unused
    @location(3) @interpolate(flat) center: vec3<f32>,
};

@vertex
fn vs(in: VIn) -> VOut {
    var out: VOut;
    let clip = G.view_proj * vec4<f32>(in.ps.xyz, 1.0);
    out.col = vec4<f32>(srgb_to_linear(in.col.rgb), in.col.a);
    out.center = in.ps.xyz;
    out.uv = vec2<f32>(0.0);
    out.geo = vec4<f32>(0.0);
    if (clip.w < G.misc.z || in.ps.w <= 0.0) {
        out.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }
    let ortho = G.cam_pos.w > 0.5;
    // pixels per world unit at this depth
    var ppw = G.proj[1][1] * G.viewport.y * 0.5;
    if (!ortho) {
        ppw = ppw / clip.w;
    }
    var r_px = 0.5 * in.ps.w * U.params.x * G.misc.y;
    if (U.params.w > 0.5) {
        r_px = 0.5 * in.ps.w * U.params.x * ppw;
    }
    let r_world = r_px / ppw;
    let ext = r_px + 1.5;
    let corner = vec2<f32>(f32(in.vi & 1u), f32(in.vi >> 1u)) * 2.0 - 1.0;
    out.uv = corner * ext / max(r_px, 1e-4);
    out.geo = vec4<f32>(r_px, r_world, clip.w, 0.0);
    let off = corner * ext * 2.0 * G.viewport.zw * clip.w;
    out.pos = vec4<f32>(clip.xy + off, clip.z, clip.w);
    return out;
}

@fragment
fn fs_disc(in: VOut) -> @location(0) vec4<f32> {
    let r_px = in.geo.x;
    let r = length(in.uv) * r_px;
    let a = clamp(r_px - r + 0.5, 0.0, 1.0);
    if (a <= 0.0) {
        discard;
    }
    let ow = U.params.z * G.misc.y;
    var col = in.col;
    if (ow > 0.0) {
        let t = clamp(r - (r_px - ow) + 0.5, 0.0, 1.0) * U.outline.a;
        col = vec4<f32>(mix(col.rgb, U.outline.rgb, t), mix(col.a, 1.0, t));
    }
    let alpha = col.a * a;
    return vec4<f32>(col.rgb * alpha * U.params.y, alpha);
}

struct SphereOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_sphere(in: VOut) -> SphereOut {
    var out: SphereOut;
    let r_px = in.geo.x;
    let rl = length(in.uv);
    let a = clamp((1.0 - rl) * r_px + 0.5, 0.0, 1.0);
    if (a <= 0.0) {
        discard;
    }
    let z = sqrt(max(1.0 - rl * rl, 0.0));
    let n = normalize(G.cam_right.xyz * in.uv.x + G.cam_up.xyz * in.uv.y - G.cam_fwd.xyz * z);
    let wpos = in.center + n * in.geo.y;
    let col = shade(in.col.rgb, n, wpos, 0.55, 0.9) * U.params.y;
    // depth of the sphere surface, not of the billboard
    var d = in.geo.z - z * in.geo.y;
    if (G.cam_pos.w > 0.5) {
        d = dot(wpos - G.cam_pos.xyz, G.cam_fwd.xyz);
    }
    let cz = G.proj[2][2] * (-d) + G.proj[3][2];
    let cw = G.proj[2][3] * (-d) + G.proj[3][3];
    out.depth = clamp(cz / cw, 0.0, 1.0);
    out.color = vec4<f32>(col, a * in.col.a);
    return out;
}

@fragment
fn fs_glow(in: VOut) -> @location(0) vec4<f32> {
    let r2 = dot(in.uv, in.uv);
    let g = max(exp(-r2 * 4.5) - 0.0111, 0.0);
    let core = exp(-r2 * 40.0);
    let v = (g + core * 0.6) * in.col.a * U.params.y;
    return vec4<f32>(in.col.rgb * v, v);
}

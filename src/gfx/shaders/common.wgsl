// Shared by every scene shader: per-frame globals, colour maps, lighting.

struct Globals {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    cam_pos: vec4<f32>,   // xyz eye position, w = 1 for orthographic
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    cam_fwd: vec4<f32>,
    viewport: vec4<f32>,  // width, height, 1/width, 1/height (target pixels)
    misc: vec4<f32>,      // unused, px_scale, near plane, unused
    light: vec4<f32>,     // direction towards the key light
    bg0: vec4<f32>,       // background centre colour (linear)
    bg1: vec4<f32>,       // background edge colour (linear)
    grid: vec4<f32>,      // fade radius, line alpha, axis alpha, min px between lines
    grid_color: vec4<f32>,
    focus: vec4<f32>,     // xyz point the camera looks at
};

@group(0) @binding(0) var<uniform> G: Globals;
@group(0) @binding(1) var lut_tex: texture_2d<f32>;
@group(0) @binding(2) var lut_smp: sampler;

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    let lo = c / 12.92;
    let hi = pow((c + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4));
    return select(hi, lo, c <= vec3<f32>(0.04045));
}

fn colormap(row: f32, t: f32) -> vec3<f32> {
    let n = f32(textureDimensions(lut_tex).y);
    let uv = vec2<f32>(clamp(t, 0.002, 0.998), (row + 0.5) / n);
    return textureSampleLevel(lut_tex, lut_smp, uv, 0.0).rgb;
}

// Pixel coordinates matching @builtin(position) in the fragment stage.
fn to_screen(clip: vec4<f32>) -> vec2<f32> {
    let ndc = clip.xy / clip.w;
    return vec2<f32>((ndc.x * 0.5 + 0.5) * G.viewport.x, (0.5 - ndc.y * 0.5) * G.viewport.y);
}

fn view_dir(wpos: vec3<f32>) -> vec3<f32> {
    if (G.cam_pos.w > 0.5) {
        return -G.cam_fwd.xyz;
    }
    return normalize(G.cam_pos.xyz - wpos);
}

// Soft studio lighting: wrapped key light, hemisphere ambient, specular, Fresnel rim.
// Two-sided: the normal is flipped to face the viewer.
fn shade(base: vec3<f32>, n_in: vec3<f32>, wpos: vec3<f32>, spec: f32, rim: f32) -> vec3<f32> {
    let v = view_dir(wpos);
    var n = normalize(n_in);
    let back = dot(n, v) < 0.0;
    if (back) {
        n = -n;
    }
    let l = normalize(G.light.xyz);
    let ndl = dot(n, l);
    let diff = clamp((ndl + 0.4) / 1.4, 0.0, 1.0);
    let sky = vec3<f32>(0.62, 0.70, 0.84);
    let ground = vec3<f32>(0.20, 0.17, 0.22);
    let amb = mix(ground, sky, n.z * 0.5 + 0.5) * 0.50;
    let h = normalize(l + v);
    let s = pow(max(dot(n, h), 0.0), 56.0) * spec * step(0.0, ndl);
    let fres = pow(1.0 - max(dot(n, v), 0.0), 3.0);
    var col = base * (amb + diff * 0.82) + vec3<f32>(s) + (base * 0.55 + vec3<f32>(0.07)) * fres * rim;
    if (back) {
        col = col * 0.70;
    }
    return col;
}

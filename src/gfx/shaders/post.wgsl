// Post-processing: HDR bloom (progressive down/up-sampling), tone mapping, vignette
// and dithering. Output is gamma-encoded for display.

struct PostU {
    a: vec4<f32>,  // bloom threshold, knee, bloom strength, exposure
    b: vec4<f32>,  // vignette, dither amplitude, unused, unused
};

@group(0) @binding(0) var<uniform> P: PostU;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var smp: sampler;
@group(0) @binding(3) var bloom: texture_2d<f32>;

struct FOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> FOut {
    var out: FOut;
    let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
    out.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn tap(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(src, smp, uv, 0.0).rgb;
}

// 13-tap downsample (Jimenez, "Next Generation Post Processing in Call of Duty").
fn down13(uv: vec2<f32>, karis: bool) -> vec3<f32> {
    let ts = 1.0 / vec2<f32>(textureDimensions(src));
    let a = tap(uv + ts * vec2<f32>(-2.0, -2.0));
    let b = tap(uv + ts * vec2<f32>(0.0, -2.0));
    let c = tap(uv + ts * vec2<f32>(2.0, -2.0));
    let d = tap(uv + ts * vec2<f32>(-2.0, 0.0));
    let e = tap(uv);
    let f = tap(uv + ts * vec2<f32>(2.0, 0.0));
    let g = tap(uv + ts * vec2<f32>(-2.0, 2.0));
    let h = tap(uv + ts * vec2<f32>(0.0, 2.0));
    let i = tap(uv + ts * vec2<f32>(2.0, 2.0));
    let j = tap(uv + ts * vec2<f32>(-1.0, -1.0));
    let k = tap(uv + ts * vec2<f32>(1.0, -1.0));
    let l = tap(uv + ts * vec2<f32>(-1.0, 1.0));
    let m = tap(uv + ts * vec2<f32>(1.0, 1.0));
    if (karis) {
        // weight each group by 1/(1+luma) to stop single bright pixels from flickering
        let g0 = (j + k + l + m) * 0.25;
        let g1 = (a + b + d + e) * 0.25;
        let g2 = (b + c + e + f) * 0.25;
        let g3 = (d + e + g + h) * 0.25;
        let g4 = (e + f + h + i) * 0.25;
        let w0 = 0.5 / (1.0 + luma(g0));
        let w1 = 0.125 / (1.0 + luma(g1));
        let w2 = 0.125 / (1.0 + luma(g2));
        let w3 = 0.125 / (1.0 + luma(g3));
        let w4 = 0.125 / (1.0 + luma(g4));
        return (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
    }
    return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
}

@fragment
fn fs_prefilter(in: FOut) -> @location(0) vec4<f32> {
    let c = down13(in.uv, true);
    // soft-knee threshold: only light brighter than the threshold blooms
    let br = max(c.r, max(c.g, c.b));
    let knee = P.a.y;
    let soft = clamp(br - P.a.x + knee, 0.0, 2.0 * knee);
    let contrib = max(soft * soft / (4.0 * knee + 1e-5), br - P.a.x) / max(br, 1e-5);
    return vec4<f32>(c * max(contrib, 0.0), 1.0);
}

@fragment
fn fs_down(in: FOut) -> @location(0) vec4<f32> {
    return vec4<f32>(down13(in.uv, false), 1.0);
}

// 3x3 tent upsample, blended additively onto the next larger level.
@fragment
fn fs_up(in: FOut) -> @location(0) vec4<f32> {
    let ts = 1.0 / vec2<f32>(textureDimensions(src));
    var c = tap(in.uv) * 4.0;
    c = c + (tap(in.uv + ts * vec2<f32>(-1.0, 0.0)) + tap(in.uv + ts * vec2<f32>(1.0, 0.0))
        + tap(in.uv + ts * vec2<f32>(0.0, -1.0)) + tap(in.uv + ts * vec2<f32>(0.0, 1.0))) * 2.0;
    c = c + tap(in.uv + ts * vec2<f32>(-1.0, -1.0)) + tap(in.uv + ts * vec2<f32>(1.0, -1.0))
        + tap(in.uv + ts * vec2<f32>(-1.0, 1.0)) + tap(in.uv + ts * vec2<f32>(1.0, 1.0));
    return vec4<f32>(c / 16.0, 1.0);
}

// Khronos PBR Neutral: keeps hues as they are and rolls highlights off smoothly
// towards white. Its toe takes up to 0.04 (linear) off every channel, which is what
// deepens the background and the dark ends of the colour maps. (The swatches in the
// panels are drawn without it, and so are a shade lighter than the same colours in
// the picture.)
fn tonemap(color_in: vec3<f32>) -> vec3<f32> {
    let start = 0.8 - 0.04;
    let desat = 0.15;
    var color = color_in;
    let x = min(color.r, min(color.g, color.b));
    var offset = 0.04;
    if (x < 0.08) {
        offset = x - 6.25 * x * x;
    }
    color = color - vec3<f32>(offset);
    let peak = max(color.r, max(color.g, color.b));
    if (peak < start) {
        return color;
    }
    let d = 1.0 - start;
    let new_peak = 1.0 - d * d / (peak + d - start);
    color = color * (new_peak / peak);
    let g = 1.0 - 1.0 / (desat * (peak - new_peak) + 1.0);
    return mix(color, vec3<f32>(new_peak), g);
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(hi, lo, c <= vec3<f32>(0.0031308));
}

fn hash(p: vec2<f32>) -> f32 {
    var q = fract(vec3<f32>(p.xyx) * vec3<f32>(0.1031, 0.1030, 0.0973));
    q = q + dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

@fragment
fn fs_composite(in: FOut) -> @location(0) vec4<f32> {
    var c = tap(in.uv) + textureSampleLevel(bloom, smp, in.uv, 0.0).rgb * P.a.z;
    c = tonemap(max(c * P.a.w, vec3<f32>(0.0)));
    let q = in.uv - vec2<f32>(0.5);
    c = c * (1.0 - P.b.x * smoothstep(0.25, 1.0, dot(q, q) * 2.0));
    var g = linear_to_srgb(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)));
    // triangular dither removes banding in dark gradients
    let n = hash(in.pos.xy) + hash(in.pos.xy + vec2<f32>(17.0, 59.0)) - 1.0;
    g = g + vec3<f32>(n * P.b.y / 255.0);
    return vec4<f32>(clamp(g, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}

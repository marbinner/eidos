// Pictures: one texture drawn as any number of rectangles, each showing a part of it
// (a heat map, a grid of small multiples, a matrix). One instance per rectangle.

@group(1) @binding(0) var img: texture_2d<f32>;
@group(1) @binding(1) var img_smp: sampler;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) alpha: f32,
};

@vertex
fn vs(
    @builtin(vertex_index) vi: u32,
    @location(0) rect: vec4<f32>,   // min.xy, max.xy in world units
    @location(1) uv: vec4<f32>,     // the texture coordinates of those two corners
    @location(2) misc: vec2<f32>,   // z, opacity
) -> VOut {
    let c = vec2<f32>(f32(vi & 1u), f32(vi >> 1u));
    var out: VOut;
    out.pos = G.view_proj * vec4<f32>(mix(rect.xy, rect.zw, c), misc.x, 1.0);
    out.uv = mix(uv.xy, uv.zw, c);
    out.alpha = misc.y;
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let c = textureSampleLevel(img, img_smp, in.uv, 0.0);
    let a = c.a * in.alpha;
    return vec4<f32>(c.rgb * a, a);
}

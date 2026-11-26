// Immediate-mode filled triangles: areas, polygons, translucent planes, boxes.
// A zero normal means "unlit".

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) col: vec4<f32>,
    @location(2) nrm: vec4<f32>,
};

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) wpos: vec3<f32>,
    @location(1) col: vec4<f32>,
    @location(2) nrm: vec3<f32>,
};

@vertex
fn vs(in: VIn) -> VOut {
    var out: VOut;
    out.wpos = in.pos;
    out.col = vec4<f32>(srgb_to_linear(in.col.rgb), in.col.a);
    out.nrm = in.nrm.xyz;
    out.pos = G.view_proj * vec4<f32>(in.pos, 1.0);
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    var col = in.col.rgb;
    if (dot(in.nrm, in.nrm) > 0.01) {
        col = shade(col, in.nrm, in.wpos, 0.25, 0.6);
    }
    return vec4<f32>(col * in.col.a, in.col.a);
}

// Grid surfaces with no vertex or index buffers. Positions live in an RGBA32F texture
// (xyz + a scalar used for colouring); the vertex stage derives the grid topology from
// the vertex index and the normals from neighbouring texels. Updating a surface is a
// single texture upload. Contour and parameter lines are drawn analytically per pixel.
//
// A texel whose scalar is beyond 1e37 is a hole (Surface::HOLE in surface.rs): the four
// cells round it are not drawn, and its neighbours take their normals from the side that
// is there.

struct SurfU {
    model: mat4x4<f32>,
    dims: vec4<f32>,       // nx, ny, wrap u, wrap v
    color: vec4<f32>,      // base colour (linear) and opacity
    cmap: vec4<f32>,       // colour-map row (< 0 = none), scalar min, scalar max, contour reach (0 = everywhere)
    iso: vec4<f32>,        // contour spacing (0 = off), contour alpha, u lines, v lines
    shade: vec4<f32>,      // lit amount, specular, rim, parameter-line alpha
    line_color: vec4<f32>,
};

@group(1) @binding(0) var<uniform> S: SurfU;
@group(1) @binding(1) var ptex: texture_2d<f32>;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) wpos: vec3<f32>,
    @location(1) nrm: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) s: f32,
};

fn wrap_index(i: i32, n: i32, wrap: bool) -> i32 {
    if (wrap) {
        let m = n - 1;
        return ((i % m) + m) % m;
    }
    return clamp(i, 0, n - 1);
}

fn fetch(i: i32, j: i32) -> vec4<f32> {
    let ii = wrap_index(i, i32(S.dims.x), S.dims.z > 0.5);
    let jj = wrap_index(j, i32(S.dims.y), S.dims.w > 0.5);
    return textureLoad(ptex, vec2<i32>(ii, jj), 0);
}

fn hole(p: vec4<f32>) -> bool {
    return p.w > 1.0e37;
}

// A neighbour for the normal: the texel itself where the neighbour is a hole.
fn beside(i: i32, j: i32, p: vec4<f32>) -> vec3<f32> {
    let q = fetch(i, j);
    if (hole(q)) {
        return p.xyz;
    }
    return q.xyz;
}

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    let nx = u32(S.dims.x);
    let ny = u32(S.dims.y);
    let cell = vi / 6u;
    let k = vi % 6u;
    let cx = cell % (nx - 1u);
    let cy = cell / (nx - 1u);
    var ox = array<u32, 6>(0u, 1u, 1u, 0u, 1u, 0u);
    var oy = array<u32, 6>(0u, 0u, 1u, 0u, 1u, 1u);
    let i = i32(cx + ox[k]);
    let j = i32(cy + oy[k]);
    let p = fetch(i, j);
    let x0 = i32(cx);
    let y0 = i32(cy);
    if (hole(fetch(x0, y0)) || hole(fetch(x0 + 1, y0)) || hole(fetch(x0, y0 + 1)) || hole(fetch(x0 + 1, y0 + 1))) {
        // a cell with a corner missing: all six of its vertices go to one point
        // outside the view, so it has no area
        var gone: VOut;
        gone.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        gone.wpos = vec3<f32>(0.0);
        gone.nrm = vec3<f32>(0.0, 0.0, 1.0);
        gone.uv = vec2<f32>(0.0);
        gone.s = 0.0;
        return gone;
    }
    let du = beside(i + 1, j, p) - beside(i - 1, j, p);
    let dv = beside(i, j + 1, p) - beside(i, j - 1, p);
    var n = cross(du, dv);
    let nl = length(n);
    if (nl > 1e-20) {
        n = n / nl;
    } else {
        n = vec3<f32>(0.0, 0.0, 1.0);
    }
    var out: VOut;
    let w = S.model * vec4<f32>(p.xyz, 1.0);
    out.wpos = w.xyz;
    out.nrm = (S.model * vec4<f32>(n, 0.0)).xyz;
    out.uv = vec2<f32>(f32(i) / f32(nx - 1u), f32(j) / f32(ny - 1u));
    out.s = p.w;
    out.pos = G.view_proj * w;
    return out;
}

// Coverage of an anti-aliased line at integer values of x.
fn iso_line(x: f32, width_px: f32) -> f32 {
    let raw = fwidth(x);
    let fw = max(raw, 1e-7);
    let d = abs(fract(x - 0.5) - 0.5) / fw;
    let l = 1.0 - clamp(d - (width_px - 1.0) * 0.5, 0.0, 1.0);
    // no line where the value is perfectly flat (a plateau sitting exactly on a level),
    // and fade out before lines get denser than a few pixels, which would alias
    return l * clamp(raw * 1e5, 0.0, 1.0) * clamp(1.0 - fw * 2.2, 0.0, 1.0);
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    var base = S.color.rgb;
    if (S.cmap.x >= 0.0) {
        let t = (in.s - S.cmap.y) / max(S.cmap.z - S.cmap.y, 1e-12);
        base = colormap(S.cmap.x, t);
    }
    var col = base;
    if (S.shade.x > 0.0) {
        col = mix(base, shade(base, in.nrm, in.wpos, S.shade.y, S.shade.z), S.shade.x);
    }
    var line = 0.0;
    if (S.iso.x > 0.0) {
        var l = iso_line(in.s / S.iso.x, 1.2) * S.iso.y;
        // with a reach set, only values within it get lines, and the lines either side
        // of the one at zero are fainter than it
        if (S.cmap.w > 0.0) {
            if (abs(in.s) > S.cmap.w) {
                l = 0.0;
            } else if (abs(in.s) > 0.5 * S.iso.x) {
                l = l * 0.5;
            }
        }
        line = max(line, l);
    }
    if (S.iso.z > 0.0) {
        line = max(line, iso_line(in.uv.x * S.iso.z, 1.0) * S.shade.w);
    }
    if (S.iso.w > 0.0) {
        line = max(line, iso_line(in.uv.y * S.iso.w, 1.0) * S.shade.w);
    }
    col = mix(col, S.line_color.rgb, line * S.line_color.a);
    let a = S.color.a;
    return vec4<f32>(col * a, a);
}

// Background gradient and the infinite reference grid on the plane z = 0.
// The grid is evaluated per pixel by intersecting the view ray with the plane, so it
// never runs out, and line spacing cross-fades between decades as you zoom.

struct FOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> FOut {
    var out: FOut;
    let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
    out.pos = vec4<f32>(p, 1.0, 1.0);
    out.ndc = p;
    return out;
}

@fragment
fn fs_bg(in: FOut) -> @location(0) vec4<f32> {
    let aspect = G.viewport.x * G.viewport.w;
    let q = vec2<f32>(in.ndc.x * aspect, in.ndc.y - 0.15);
    let r = clamp(length(q) / (1.0 + aspect), 0.0, 1.0);
    let t = smoothstep(0.0, 1.0, r);
    return vec4<f32>(mix(G.bg0.rgb, G.bg1.rgb, t), 1.0);
}

fn grid_cov(coord: vec2<f32>, spacing: f32) -> f32 {
    let c = coord / spacing;
    let fw = max(fwidth(c), vec2<f32>(1e-7));
    let g = abs(fract(c - 0.5) - 0.5) / fw;
    let l = 1.0 - clamp(min(g.x, g.y), 0.0, 1.0);
    return l * clamp(1.0 - max(fw.x, fw.y) * 1.6, 0.0, 1.0);
}

struct GridOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_grid(in: FOut) -> GridOut {
    var out: GridOut;
    let n4 = G.inv_view_proj * vec4<f32>(in.ndc, 0.0, 1.0);
    let f4 = G.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let o = n4.xyz / n4.w;
    let dir = f4.xyz / f4.w - o;
    if (abs(dir.z) < 1e-9) {
        discard;
    }
    let t = -o.z / dir.z;
    if (t <= 0.0 || t >= 1.0) {
        discard;
    }
    let wp = o + dir * t;
    let clip = G.view_proj * vec4<f32>(wp, 1.0);
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);

    // world units covered by one pixel
    let dx = dpdx(wp.xy);
    let dy = dpdy(wp.xy);
    let ppx = max(max(length(dx), length(dy)), 1e-12);
    let lod = log(ppx * G.grid.w) / 2.302585;
    let k = ceil(lod);
    let f = k - lod;
    let s0 = pow(10.0, k);
    let minor = grid_cov(wp.xy, s0) * smoothstep(0.0, 1.0, f);
    let major = grid_cov(wp.xy, s0 * 10.0);
    var a = max(minor * 0.55, major) * G.grid.y;

    // the two coordinate axes
    let fw = max(fwidth(wp.xy), vec2<f32>(1e-12));
    let ax = 1.0 - clamp(abs(wp.y) / fw.y - 0.25, 0.0, 1.0);
    let ay = 1.0 - clamp(abs(wp.x) / fw.x - 0.25, 0.0, 1.0);
    a = max(a, max(ax, ay) * G.grid.z);

    let r = length(wp.xy - G.focus.xy);
    let fade = 1.0 - smoothstep(G.grid.x * 0.3, G.grid.x, r);
    a = a * fade;
    if (a <= 0.002) {
        discard;
    }
    out.color = vec4<f32>(G.grid_color.rgb * a, a);
    return out;
}

// Anti-aliased polylines with exact round joins.
//
// One point buffer is bound four times at offsets 0, 1, 2 and 3 "strides", so every
// instance sees a sliding window (A, B, C, D) and draws the segment B-C. A and D are
// only used to build the joins. Adjacent segments are split along the bisector of
// their joint, so translucent and additive lines do not double-blend at vertices
// (except at a sharp turn between short segments, which is left whole: see `joint`).
//
// Polyline mode: stride is one point. width > 0 is a vertex, width == 0 is a ghost
// (contributes to joins, draws nothing), width < 0 is a break.
// Trail mode (ring.x > 0): the buffer is a ring of rows, one row per time step and
// one column per particle, so a whole frame of history is a single contiguous upload.

struct LineU {
    ring: vec4<f32>,   // columns (0 = polyline mode), head row, row count, rows filled
    style: vec4<f32>,  // width scale, intensity, depth bias, fade power
    dash: vec4<f32>,   // period px (0 = solid), duty, width taper, trails: 0 or 1 + lag (see vs)
    tint: vec4<f32>,
};

@group(1) @binding(0) var<uniform> U: LineU;

struct VIn {
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
    @location(2) b_col: vec4<f32>,
    @location(3) c: vec4<f32>,
    @location(4) c_col: vec4<f32>,
    @location(5) d: vec4<f32>,
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
};

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) seg: vec4<f32>,  // screen px of B (xy) and C (zw)
    @location(1) @interpolate(flat) jb: vec3<f32>,   // joint tangent at B, enabled flag
    @location(2) @interpolate(flat) jc: vec3<f32>,
    @location(3) @interpolate(flat) c0: vec4<f32>,
    @location(4) @interpolate(flat) c1: vec4<f32>,
    @location(5) @interpolate(flat) hw: vec2<f32>,   // half widths at B and C, px
};

fn clip_near(p: vec4<f32>, q: vec4<f32>, near: f32) -> vec4<f32> {
    if (p.w >= near || q.w <= near) {
        return p;
    }
    let t = (near - p.w) / (q.w - p.w);
    return mix(p, q, t);
}

// The joint between two segments meeting at a point: the direction along which they
// are split (so that each pixel belongs to one of them), and whether to split at all.
// `shorter` is the length in pixels of the shorter of the two segments and `hw` the
// half width there. At a sharp turn the split line runs nearly along the segments
// and cuts a long sliver off each; if the segments are too short to survive that (a
// jagged chart curve, for instance), they are left whole with round ends instead,
// which draws the corner twice but never thins the line to dots.
fn joint(d_in: vec2<f32>, d_out: vec2<f32>, shorter: f32, hw: f32) -> vec3<f32> {
    let t = d_in + d_out;
    let tl = length(t);
    if (tl > 0.15) {
        // how far along a segment the split line reaches before leaving its width
        let cut = hw * sqrt(max(4.0 - tl * tl, 0.0)) / tl;
        if (cut <= 0.5 * shorter) {
            return vec3<f32>(t / tl, 1.0);
        }
    }
    return vec3<f32>(0.0);
}

@vertex
fn vs(in: VIn) -> VOut {
    var out: VOut;
    out.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
    out.seg = vec4<f32>(0.0);
    out.jb = vec3<f32>(0.0);
    out.jc = vec3<f32>(0.0);
    out.c0 = vec4<f32>(0.0);
    out.c1 = vec4<f32>(0.0);
    out.hw = vec2<f32>(0.0);

    var valid = in.b.w > 0.0 && in.c.w > 0.0;
    var join_b = in.a.w >= 0.0;
    var join_c = in.d.w >= 0.0;
    var fade = vec2<f32>(1.0);

    if (U.ring.x > 0.5) {
        let cols = u32(U.ring.x);
        let head = u32(U.ring.y);
        let rows = u32(U.ring.z);
        let filled = u32(U.ring.w);
        let row_b = (in.ii / cols + 1u) % rows;
        let age_b = (head + rows - row_b) % rows;
        valid = valid && row_b != head && age_b < filled;
        join_b = join_b && (age_b + 1u) < filled;
        join_c = join_c && ((row_b + 1u) % rows) != head;
        var a0 = 0.0;
        var a1 = 0.0;
        if (U.dash.w >= 1.0) {
            // The newest row follows its particle between rows (`TrailSet::follow`): the
            // rows behind it are younger than whole steps by `lag`, and the oldest has
            // faded out by the time it is dropped.
            let lag = U.dash.w - 1.0;
            let denom = f32(max(rows, 3u) - 2u);
            a0 = max(f32(age_b) - lag, 0.0) / denom;
            a1 = max(f32(age_b) - 1.0 - lag, 0.0) / denom;
        } else {
            let denom = f32(max(rows, 2u) - 1u);
            a0 = f32(age_b) / denom;
            a1 = max(f32(age_b) - 1.0, 0.0) / denom;
        }
        fade = pow(max(vec2<f32>(1.0 - a0, 1.0 - a1), vec2<f32>(0.0)), vec2<f32>(U.style.w));
    }

    let near = G.misc.z;
    let pb = G.view_proj * vec4<f32>(in.b.xyz, 1.0);
    let pc = G.view_proj * vec4<f32>(in.c.xyz, 1.0);
    if (!valid || (pb.w < near && pc.w < near)) {
        return out;
    }
    let cb = clip_near(pb, pc, near);
    let cc = clip_near(pc, pb, near);
    let sb = to_screen(cb);
    let sc = to_screen(cc);
    var d1 = sc - sb;
    let l1 = length(d1);
    if (l1 > 1e-5) {
        d1 = d1 / l1;
    } else {
        d1 = vec2<f32>(1.0, 0.0);
    }

    let taper = mix(vec2<f32>(1.0), fade, U.dash.z);
    let hw = 0.5 * vec2<f32>(abs(in.b.w), abs(in.c.w)) * U.style.x * G.misc.y * taper;

    if (join_b) {
        let pa = G.view_proj * vec4<f32>(in.a.xyz, 1.0);
        if (pa.w > near) {
            let d0 = sb - to_screen(pa);
            let l0 = length(d0);
            if (l0 > 1e-5) {
                out.jb = joint(d0 / l0, d1, min(l0, l1), hw.x);
            }
        }
    }
    if (join_c) {
        let pd = G.view_proj * vec4<f32>(in.d.xyz, 1.0);
        if (pd.w > near) {
            let d2 = to_screen(pd) - sc;
            let l2 = length(d2);
            if (l2 > 1e-5) {
                // (the direction must be a unit vector here, as it is at the other end:
                // for a long time it was not, and segments before a sharp turn vanished)
                out.jc = joint(d1, d2 / l2, min(l1, l2), hw.y);
            }
        }
    }

    let ext = max(hw.x, hw.y) + 1.0;
    let nrm = vec2<f32>(-d1.y, d1.x);
    let cx = f32(in.vi & 1u);
    let cy = f32(in.vi >> 1u) * 2.0 - 1.0;
    let spx = mix(sb, sc, cx) + d1 * ((cx * 2.0 - 1.0) * ext) + nrm * (cy * ext);
    let cl = mix(cb, cc, cx);
    let ndc = vec2<f32>(spx.x * G.viewport.z * 2.0 - 1.0, 1.0 - spx.y * G.viewport.w * 2.0);
    out.pos = vec4<f32>(ndc * cl.w, cl.z - U.style.z * cl.w, cl.w);
    out.seg = vec4<f32>(sb, sc);
    out.hw = hw;
    let c0 = vec4<f32>(srgb_to_linear(in.b_col.rgb), in.b_col.a) * U.tint;
    let c1 = vec4<f32>(srgb_to_linear(in.c_col.rgb), in.c_col.a) * U.tint;
    out.c0 = vec4<f32>(c0.rgb, c0.a * fade.x);
    out.c1 = vec4<f32>(c1.rgb, c1.a * fade.y);
    return out;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let p = in.pos.xy;
    let sb = in.seg.xy;
    let sc = in.seg.zw;
    if (in.jb.z > 0.5 && dot(p - sb, in.jb.xy) < 0.0) {
        discard;
    }
    if (in.jc.z > 0.5 && dot(p - sc, in.jc.xy) >= 0.0) {
        discard;
    }
    let ba = sc - sb;
    let pa = p - sb;
    let l2 = max(dot(ba, ba), 1e-8);
    let h = clamp(dot(pa, ba) / l2, 0.0, 1.0);
    let dist = length(pa - ba * h);
    let hw = mix(in.hw.x, in.hw.y, h);
    var a = clamp(hw - dist + 0.5, 0.0, 1.0);
    if (U.dash.x > 0.0) {
        // (the period is in logical pixels, like every width; the pattern starts
        // afresh at each segment, so it suits straight lines: Frame::dashed_path
        // dashes a curve)
        let period = U.dash.x * G.misc.y;
        let s = dot(pa, ba) / sqrt(l2);
        let ph = fract(s / period) * period;
        let on = period * U.dash.y;
        a = a * clamp(min(ph, on - ph) + 0.5, 0.0, 1.0);
    }
    let c = mix(in.c0, in.c1, h);
    let alpha = c.a * a;
    if (alpha <= 0.0) {
        discard;
    }
    return vec4<f32>(c.rgb * alpha * U.style.y, alpha);
}

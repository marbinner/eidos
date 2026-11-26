//! Colours and perceptual colour maps.
//!
//! Vertex colours travel to the GPU as 4 sRGB bytes. Colour maps are interpolated in
//! Oklab from a handful of stops and baked once into a lookup texture.

use glam::Vec3;

/// An sRGB colour with straight alpha, 4 bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(transparent)]
pub struct Color(pub [u8; 4]);

fn srgb_encode(x: f32) -> f32 {
    if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

fn srgb_decode(x: f32) -> f32 {
    if x <= 0.040_45 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

fn to_u8(x: f32) -> u8 {
    (x.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

impl Color {
    pub const WHITE: Color = Color([255, 255, 255, 255]);
    pub const BLACK: Color = Color([0, 0, 0, 255]);
    pub const CLEAR: Color = Color([0, 0, 0, 0]);

    /// From a 0xRRGGBB literal.
    pub const fn hex(v: u32) -> Color {
        Color([(v >> 16) as u8, (v >> 8) as u8, v as u8, 255])
    }

    /// From linear-light components.
    pub fn linear(r: f32, g: f32, b: f32) -> Color {
        Color([to_u8(srgb_encode(r)), to_u8(srgb_encode(g)), to_u8(srgb_encode(b)), 255])
    }

    pub fn from_linear(v: Vec3) -> Color {
        Color::linear(v.x, v.y, v.z)
    }

    pub fn to_linear(self) -> Vec3 {
        Vec3::new(srgb_decode(self.0[0] as f32 / 255.0), srgb_decode(self.0[1] as f32 / 255.0), srgb_decode(self.0[2] as f32 / 255.0))
    }

    /// Linear RGBA as needed by shader uniforms.
    pub fn to_linear_rgba(self) -> [f32; 4] {
        let l = self.to_linear();
        [l.x, l.y, l.z, self.0[3] as f32 / 255.0]
    }

    /// Replace alpha (0..1).
    pub fn a(self, alpha: f32) -> Color {
        Color([self.0[0], self.0[1], self.0[2], to_u8(alpha)])
    }

    /// Interpolate in linear light.
    pub fn mix(self, other: Color, t: f32) -> Color {
        let c = self.to_linear().lerp(other.to_linear(), t.clamp(0.0, 1.0));
        let a = self.0[3] as f32 + (other.0[3] as f32 - self.0[3] as f32) * t.clamp(0.0, 1.0);
        let mut out = Color::from_linear(c);
        out.0[3] = a as u8;
        out
    }

    pub fn egui(self) -> egui::Color32 {
        egui::Color32::from_rgba_unmultiplied(self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

/// The shared palette. Chosen to stay distinct on a dark background and under bloom.
pub mod pal {
    use super::Color;
    pub const TEXT: Color = Color::hex(0xE6EAF2);
    pub const DIM: Color = Color::hex(0x8792A8);
    pub const FAINT: Color = Color::hex(0x4A5468);
    /// Small headings and footnotes: quiet, but still readable on the dark panels
    /// (FAINT is for lines, not for text).
    pub const MUTED: Color = Color::hex(0x7F8AA3);
    /// Links: to another scene, to the glossary.
    pub const LINK: Color = Color::hex(0x82B2FF);
    pub const CYAN: Color = Color::hex(0x4CC9F0);
    pub const BLUE: Color = Color::hex(0x4F8BFF);
    pub const PINK: Color = Color::hex(0xFF4D8D);
    pub const YELLOW: Color = Color::hex(0xFFD166);
    pub const GREEN: Color = Color::hex(0x2EE6A6);
    pub const ORANGE: Color = Color::hex(0xFF9F43);
    pub const PURPLE: Color = Color::hex(0xA78BFA);
    pub const RED: Color = Color::hex(0xFF5D5D);
    pub const WHITE: Color = Color::hex(0xF5F7FB);
    /// Axis colours (x, y, z).
    pub const AXIS: [Color; 3] = [Color::hex(0xFF6B81), Color::hex(0x5BE39C), Color::hex(0x5AA9FF)];
    /// A cycle for "series 0, 1, 2, ...".
    pub const SERIES: [Color; 8] = [CYAN, PINK, YELLOW, GREEN, PURPLE, ORANGE, BLUE, RED];
}

fn linear_to_oklab(c: Vec3) -> Vec3 {
    let l = 0.412_221_47 * c.x + 0.536_332_55 * c.y + 0.051_445_995 * c.z;
    let m = 0.211_903_5 * c.x + 0.680_699_5 * c.y + 0.107_396_96 * c.z;
    let s = 0.088_302_46 * c.x + 0.281_718_85 * c.y + 0.629_978_7 * c.z;
    let (l, m, s) = (l.cbrt(), m.cbrt(), s.cbrt());
    Vec3::new(
        0.210_454_26 * l + 0.793_617_8 * m - 0.004_072_047 * s,
        1.977_998_5 * l - 2.428_592_2 * m + 0.450_593_7 * s,
        0.025_904_037 * l + 0.782_771_77 * m - 0.808_675_77 * s,
    )
}

fn oklab_to_linear(c: Vec3) -> Vec3 {
    let l = c.x + 0.396_337_78 * c.y + 0.215_803_76 * c.z;
    let m = c.x - 0.105_561_346 * c.y - 0.063_854_17 * c.z;
    let s = c.x - 0.089_484_18 * c.y - 1.291_485_5 * c.z;
    let (l, m, s) = (l * l * l, m * m * m, s * s * s);
    Vec3::new(
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086_3 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    )
}

/// Colour maps, in the order they are stored in the lookup texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Cmap {
    Viridis,
    Magma,
    Inferno,
    Plasma,
    Turbo,
    Cividis,
    Aurora,
    CoolWarm,
    IceFire,
    Twilight,
    Ocean,
    Grays,
    Slate,
}

impl Cmap {
    pub const ALL: [Cmap; 13] = [
        Cmap::Viridis,
        Cmap::Magma,
        Cmap::Inferno,
        Cmap::Plasma,
        Cmap::Turbo,
        Cmap::Cividis,
        Cmap::Aurora,
        Cmap::CoolWarm,
        Cmap::IceFire,
        Cmap::Twilight,
        Cmap::Ocean,
        Cmap::Grays,
        Cmap::Slate,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Cmap::Viridis => "Viridis",
            Cmap::Magma => "Magma",
            Cmap::Inferno => "Inferno",
            Cmap::Plasma => "Plasma",
            Cmap::Turbo => "Turbo",
            Cmap::Cividis => "Cividis",
            Cmap::Aurora => "Aurora",
            Cmap::CoolWarm => "Cool-warm",
            Cmap::IceFire => "Ice-fire",
            Cmap::Twilight => "Twilight",
            Cmap::Ocean => "Ocean",
            Cmap::Grays => "Grays",
            Cmap::Slate => "Slate",
        }
    }

    fn stops(self) -> &'static [u32] {
        match self {
            Cmap::Viridis => &[0x440154, 0x482878, 0x3E4989, 0x31688E, 0x26828E, 0x1F9E89, 0x35B779, 0x6ECE58, 0xB5DE2B, 0xFDE725],
            Cmap::Magma => &[0x000004, 0x180F3D, 0x440F76, 0x721F81, 0x9E2F7F, 0xCD4071, 0xF1605D, 0xFD9668, 0xFECA8D, 0xFCFDBF],
            Cmap::Inferno => &[0x000004, 0x1B0C41, 0x4A0C6B, 0x781C6D, 0xA52C60, 0xCF4446, 0xED6925, 0xFB9B06, 0xF7D13D, 0xFCFFA4],
            Cmap::Plasma => &[0x0D0887, 0x47039F, 0x7301A8, 0x9C179E, 0xBD3786, 0xD8576B, 0xED7953, 0xFA9E3B, 0xFDC926, 0xF0F921],
            Cmap::Turbo => &[0x30123B, 0x4662D7, 0x36AAF9, 0x1AE4B6, 0x72FE5E, 0xC7EF34, 0xFABA39, 0xF66B19, 0xCB2A04, 0x7A0403],
            Cmap::Cividis => &[0x00204D, 0x00336F, 0x39486B, 0x575C6D, 0x707173, 0x8A8779, 0xA69D75, 0xC4B56C, 0xE4CF5B, 0xFFEA46],
            Cmap::Aurora => &[0x120B3A, 0x2A2B8F, 0x2462D0, 0x18A3D8, 0x17CFC0, 0x57E89A, 0xB4F58A, 0xF9F5A6],
            Cmap::CoolWarm => &[0x2B5BD7, 0x5E8DF0, 0x9DBBF7, 0xDCE2EE, 0xF4C6AE, 0xF08A6C, 0xD9412F, 0xB0102A],
            Cmap::IceFire => &[0xBDE7F5, 0x5FB5E8, 0x3A6FD8, 0x3A3F9C, 0x241F3A, 0x1F1A1C, 0x4D1F28, 0xA32B33, 0xE0562B, 0xF99F4A, 0xFDDCA0],
            Cmap::Twilight => &[0xE2D9E2, 0x9FBBD0, 0x5F86C0, 0x5548A8, 0x4A1F6E, 0x2F1436, 0x561B40, 0x8E2A4F, 0xBA5A52, 0xCF9C87, 0xE2D9E2],
            Cmap::Ocean => &[0x061A2E, 0x0B3A5B, 0x0F6A8C, 0x17A2A8, 0x55D3B4, 0xB9F2D0, 0xF4FFF4],
            Cmap::Grays => &[0x0A0C10, 0xF4F6FA],
            Cmap::Slate => &[0x0B1222, 0x16284A, 0x24406B, 0x35608A, 0x5383A6, 0x86AEC6, 0xC6DDE8],
        }
    }

    /// Sample at `t` in 0..1, returning linear RGB.
    pub fn sample(self, t: f32) -> Vec3 {
        let stops = self.stops();
        let n = stops.len() - 1;
        let x = t.clamp(0.0, 1.0) * n as f32;
        let i = (x.floor() as usize).min(n - 1);
        let f = x - i as f32;
        let a = linear_to_oklab(Color::hex(stops[i]).to_linear());
        let b = linear_to_oklab(Color::hex(stops[i + 1]).to_linear());
        oklab_to_linear(a.lerp(b, f)).max(Vec3::ZERO)
    }

    pub fn color(self, t: f32) -> Color {
        Color::from_linear(self.sample(t))
    }

    /// The map as 256 ready-made colours, for colouring many values on the CPU
    /// (`color` interpolates exactly and is far slower).
    pub fn table(self) -> &'static [Color] {
        static TABLES: std::sync::OnceLock<Vec<Color>> = std::sync::OnceLock::new();
        let all = TABLES.get_or_init(|| bytemuck::cast_slice::<u8, Color>(&build_lut()).to_vec());
        let w = LUT_WIDTH as usize;
        &all[self as usize * w..(self as usize + 1) * w]
    }
}

/// Index into a [`Cmap::table`] for `t` in 0..1 (clamped; NaN maps to the middle).
#[inline]
pub fn table_index(t: f32) -> usize {
    let t = if t.is_nan() { 0.5 } else { t };
    (t.clamp(0.0, 1.0) * 255.0 + 0.5) as usize
}

pub const LUT_WIDTH: u32 = 256;

/// sRGB bytes for the lookup texture: one row per colour map.
pub fn build_lut() -> Vec<u8> {
    let mut out = Vec::with_capacity((LUT_WIDTH as usize) * Cmap::ALL.len() * 4);
    for map in Cmap::ALL {
        for i in 0..LUT_WIDTH {
            let c = map.color(i as f32 / (LUT_WIDTH - 1) as f32);
            out.extend_from_slice(&c.0);
        }
    }
    out
}

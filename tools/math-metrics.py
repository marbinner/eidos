#!/usr/bin/env python3
"""Write src/typeset/metrics.rs: the size of every glyph of the mathematics fonts
(assets/math), which is what formulas are laid out by.

    python3 tools/math-metrics.py      (needs fontTools: pip install fonttools)

For each character: its advance, how far its ink reaches above and below the
baseline, its italic correction (how far an italic letter leans out past its
advance, which keeps a superscript off it) and its skew (how far right of the middle
of its advance the top of the letter is, which is where an accent goes over it), in
thousandths of an em. The last two are measured from the outlines: TeX's own
figures for them are not in these files.
"""

import os
import sys

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.ttLib import TTFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACES = [
    ("MATH_ITALIC", "KaTeX_Math-Italic.ttf"),
    ("MAIN", "KaTeX_Main-Regular.ttf"),
    ("MAIN_BOLD", "KaTeX_Main-Bold.ttf"),
    ("AMS", "KaTeX_AMS-Regular.ttf"),
    ("SIZE1", "KaTeX_Size1-Regular.ttf"),
    ("SIZE2", "KaTeX_Size2-Regular.ttf"),
]


def glyphs(path):
    font = TTFont(path)
    assert font["head"].unitsPerEm == 1000, path
    glyph_set, cmap, hmtx = font.getGlyphSet(), font.getBestCmap(), font["hmtx"]
    for code in sorted(cmap):
        if code <= 32:
            continue
        name = cmap[code]
        advance = hmtx[name][0]
        bounds = BoundsPen(glyph_set)
        glyph_set[name].draw(bounds)
        if bounds.bounds is None:
            yield code, advance, 0, 0, 0, 0
            continue
        x0, y0, x1, y1 = bounds.bounds
        # the top of the letter: the outline's points in the top third of its ink
        pen = DecomposingRecordingPen(glyph_set)
        glyph_set[name].draw(pen)
        points = [p for _, args in pen.value for p in args if p is not None]
        top = [x for x, y in points if y > y1 - 0.3 * (y1 - max(y0, 0.0))]
        skew = (min(top) + max(top)) / 2 - advance / 2 if top else 0.0
        italic = max(0.0, x1 - advance)
        yield code, advance, round(y1), round(-y0), round(italic), round(skew)


def main():
    out = [
        "//! The size of every glyph of the mathematics fonts, in thousandths of an em:",
        "//! (character, advance, height, depth, italic correction, skew). Written by",
        "//! `tools/math-metrics.py` from the fonts in `assets/math`; do not edit by hand.",
        "",
        "pub(super) type Glyph = (u32, i16, i16, i16, i16, i16);",
        "",
    ]
    for const, file in FACES:
        rows = list(glyphs(os.path.join(ROOT, "assets", "math", file)))
        out.append(f"pub(super) static {const}: [Glyph; {len(rows)}] = [")
        line = "   "
        for row in rows:
            item = " (0x%X, %d, %d, %d, %d, %d)," % row
            if len(line) + len(item) > 150:
                out.append(line)
                line = "   "
            line += item
        out.append(line)
        out.append("];")
        out.append("")
    path = os.path.join(ROOT, "src", "typeset", "metrics.rs")
    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {path}", file=sys.stderr)


if __name__ == "__main__":
    main()

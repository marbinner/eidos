# Assets

`symbols.ttf` is a subset of DejaVu Sans Mono (mathematical symbols, superscript and
subscript letters, arrows, geometric shapes), used as a fallback for characters the
interface's main font lacks. DejaVu fonts are (c) 2003 Bitstream, Inc. (Bitstream Vera)
with changes in the public domain; they may be embedded and redistributed freely.
Licence: https://dejavu-fonts.github.io/License.html

Made with:

    pyftsubset DejaVuSansMono.ttf --output-file=symbols.ttf --no-hinting --layout-features='' \
        --unicodes="U+00A0-017F,U+02B0-02FF,U+1D00-1DBF,U+1E00-1EFF,U+2010-2027,U+2030-205E,U+2070-209F,U+2100-214F,U+2190-21FF,U+2200-22FF,U+2300-23FF,U+25A0-25FF,U+2600-26FF,U+2700-27BF,U+27C0-27EF,U+2C60-2C7F"

`math/` holds the fonts formulas are set in (see `src/typeset`): six of the KaTeX fonts,
Computer Modern as KaTeX ships it, unmodified: `KaTeX_Math-Italic.ttf` (variables, Latin
and Greek), `KaTeX_Main-Regular.ttf` (digits, operators, relations, arrows, brackets,
upright letters and capital Greek), `KaTeX_Main-Bold.ttf` (`\mathbf`),
`KaTeX_AMS-Regular.ttf` (blackboard capitals, `\mathbb`, and the less common relations),
`KaTeX_Size1-Regular.ttf` and `KaTeX_Size2-Regular.ttf` (large operators, and brackets
in their two larger sizes). Copyright (c) 2009-2010 Design Science, Inc.
(<www.mathjax.org>); Copyright (c) 2014-2018 Khan Academy (<www.khanacademy.org>), with
Reserved Font Names KaTeX_Main, KaTeX_Math, KaTeX_AMS, KaTeX_Size1 and KaTeX_Size2.
Licensed under the SIL Open Font License, Version 1.1 (stated in each file):
https://openfontlicense.org. They may be embedded and redistributed as they are; being
under Reserved Font Names, they are not to be altered (subset, renamed) here.

`src/typeset/metrics.rs` (each glyph's advance, height, depth, italic correction and
where an accent goes over it) is written from these files by `tools/math-metrics.py`.

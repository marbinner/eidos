#!/bin/bash
# Build the browser version into web/pkg.   ./build-web.sh [fast]
# "fast" skips the slow whole-program optimisation, for quick iteration.
set -e
cd "$(dirname "$0")"
# (this machine's own places, if it has a file of them: EIDOS_TOOLS, EIDOS_PAGES)
[ -f ./local.env ] && . ./local.env
TOOLS=${EIDOS_TOOLS:-../eidos-tools}
PROFILE=${1:-release}
TARGET=$(cargo metadata --format-version 1 --no-deps | sed -E 's/.*"target_directory":"([^"]+)".*/\1/')
# Keep this machine's directory names out of the file that gets published: error
# messages would otherwise carry the full paths of the sources, home directory and
# all. (Setting RUSTFLAGS replaces the flags in .cargo/config.toml, so the SIMD flag
# is repeated here.)
# web/pkg/BUILD says what the files beside it were built from. It is written last, so
# a build that stops half-way leaves none, and publish.sh refuses to upload.
rm -f web/pkg/BUILD
export RUSTFLAGS="-C target-feature=+simd128 --remap-path-prefix=$HOME=~ --remap-path-prefix=$PWD=eidos --remap-path-prefix=$TARGET=target"
cargo build --lib --target wasm32-unknown-unknown --profile "$PROFILE"
"$TOOLS/wasm-bindgen/wasm-bindgen" --target web --no-typescript --out-dir web/pkg --out-name eidos \
    "$TARGET/wasm32-unknown-unknown/$PROFILE/eidos.wasm"
if [ "$PROFILE" = release ]; then
    # (the newest binaryen there, if there are several)
    "$(ls -d "$TOOLS"/binaryen-*/ | sort -V | tail -1)bin/wasm-opt" -O3 --enable-simd --enable-bulk-memory --enable-nontrapping-float-to-int \
        --enable-sign-ext --enable-mutable-globals -o web/pkg/eidos_bg.wasm web/pkg/eidos_bg.wasm
fi
# pre-compressed copies: serve.sh sends these to browsers that accept gzip (all of them)
gzip -9 -k -f web/pkg/eidos_bg.wasm web/pkg/eidos.js
REV=$(git rev-parse HEAD 2>/dev/null || echo none)
[ -z "$(git status --porcelain 2>/dev/null)" ] || REV="$REV+changes"
echo "$PROFILE $REV" > web/pkg/BUILD
ls -la web/pkg/eidos_bg.wasm web/pkg/eidos_bg.wasm.gz

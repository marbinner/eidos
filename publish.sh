#!/bin/bash
# Publish the current browser build to https://marbinner.github.io/eidos/
# (the gh-pages branch of github.com/marbinner/eidos).
#
#   ./build-web.sh && ./publish.sh ["what changed"]
#
# The local copy of that branch lives in $EIDOS_PAGES; it pushes with a deploy key
# that can write to that one repository only (set in the copy's own git config).
set -e
cd "$(dirname "$0")"
# (this machine's own places, if it has a file of them: EIDOS_TOOLS, EIDOS_PAGES)
[ -f ./local.env ] && . ./local.env
PAGES=${EIDOS_PAGES:-../eidos-pages}
[ -d "$PAGES/.git" ] || { echo "publish: no local copy of the site at $PAGES" >&2; exit 1; }
# only a finished, optimised build of exactly what is committed: not a quick "fast"
# build, not one that stopped half-way, not one older than the source
read -r BUILT REV < web/pkg/BUILD 2>/dev/null || { echo "publish: web/pkg has no finished build; run ./build-web.sh first" >&2; exit 1; }
[ "$BUILT" = release ] || { echo "publish: web/pkg is a '$BUILT' build; run ./build-web.sh (no argument) first" >&2; exit 1; }
if [ -n "$(git status --porcelain)" ] || [ "$REV" != "$(git rev-parse HEAD)" ]; then
    echo "publish: web/pkg was built from '$REV', which is not the source as committed now;" >&2
    echo "         commit, then run ./build-web.sh again" >&2
    exit 1
fi
# never publish a build that still carries this machine's directory names
TARGET=$(cargo metadata --format-version 1 --no-deps 2>/dev/null | sed 's/.*"target_directory":"\([^"]*\)".*/\1/')
if strings -n 6 web/pkg/eidos_bg.wasm web/pkg/eidos.js | grep -qF -e "$HOME" -e "$(dirname "$PAGES")" -e "${TARGET:-$HOME}"; then
    echo "publish: the build contains local paths; rebuild with ./build-web.sh first" >&2
    exit 1
fi
cp web/index.html web/diag.html web/worker.js web/preview.png web/manifest.webmanifest web/icon-*.png "$PAGES/"
cp web/pkg/eidos.js web/pkg/eidos_bg.wasm "$PAGES/pkg/"
cd "$PAGES"
git add -A
if git diff --cached --quiet; then
    echo "publish: nothing has changed since the last upload"
else
    git commit -q -m "${1:-Update the build}"
fi
git push -u origin gh-pages
# (in the source repository, the tag "published" marks what is live)
git -C "$OLDPWD" tag -f published "$REV" > /dev/null
echo "published; live in a minute or two at https://marbinner.github.io/eidos/"

#!/bin/bash
# Publish the source as it is committed now to the "main" branch of the repository the
# site is published from (github.com/marbinner/eidos).
#
#   ./publish-source.sh ["what changed"]
#
# What goes up is the tree of HEAD as one new commit on top of the last one published:
# the files, not this repository's own history (which stays here). It is pushed from
# the local copy of the site's repository ($EIDOS_PAGES), with that copy's deploy key.
set -e
cd "$(dirname "$0")"
[ -f ./local.env ] && . ./local.env
PAGES=${EIDOS_PAGES:-../eidos-pages}
[ -d "$PAGES/.git" ] || { echo "publish-source: no local copy of the site's repository at $PAGES (set EIDOS_PAGES)" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "publish-source: there are uncommitted changes; commit first" >&2; exit 1; }
# never publish a tree that carries this machine's directory names
TARGET=$(cargo metadata --format-version 1 --no-deps 2>/dev/null | sed 's/.*"target_directory":"\([^"]*\)".*/\1/')
if git grep -I -l -F -e "$HOME" -e "$(cd "$PAGES/.." && pwd)" -e "${TARGET:-$HOME}" HEAD -- . > /dev/null; then
    echo "publish-source: these files name this machine's directories:" >&2
    git grep -I -l -F -e "$HOME" -e "$(cd "$PAGES/.." && pwd)" -e "${TARGET:-$HOME}" HEAD -- . >&2
    exit 1
fi
SOURCE=$(pwd)
REV=$(git rev-parse HEAD)
MESSAGE=${1:-Update the source}
cd "$PAGES"
git fetch -q "$SOURCE" HEAD
TREE=$(git rev-parse "FETCH_HEAD^{tree}")
PARENT=$(git rev-parse -q --verify refs/heads/main || true)
if [ -n "$PARENT" ] && [ "$(git rev-parse "$PARENT^{tree}")" = "$TREE" ]; then
    echo "publish-source: nothing has changed since the last upload"
else
    COMMIT=$(git commit-tree "$TREE" ${PARENT:+-p "$PARENT"} -m "$MESSAGE")
    git update-ref refs/heads/main "$COMMIT"
fi
git push -q origin main
# (in the source repository, the tag "published-source" marks what is up)
git -C "$SOURCE" tag -f published-source "$REV" > /dev/null
echo "source published: the main branch of github.com/marbinner/eidos"

#!/bin/bash
# Time every scene in the browser build, in a headless Chrome: what the program itself
# measures of its frames (see `stats` in src/app/mod.rs), one scene after another, each
# opened from another scene as a click on its name opens it.
#
#   tools/webbench.sh [-m webgl|webgpu] [-s seconds] [-j jobs] [scene...]   > table
#
# Needs ./build-web.sh to have run, the page served (tools/serve.py web 8765) and
# EIDOS_CHROME naming a Chrome (or chrome-headless-shell). On a machine without a GPU
# the frame interval measures the software rasteriser: compare scenes, not machines.
set -e
cd "$(dirname "$0")/.."
mode=webgl seconds=7 jobs=2
while getopts m:s:j: o; do
    case $o in m) mode=$OPTARG ;; s) seconds=$OPTARG ;; j) jobs=$OPTARG ;; *) exit 2 ;; esac
done
shift $((OPTIND - 1))
chrome=${EIDOS_CHROME:?set EIDOS_CHROME to a Chrome or chrome-headless-shell}
url=${EIDOS_URL:-http://127.0.0.1:8765/}
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
scenes=("$@")
[ ${#scenes[@]} -gt 0 ] || mapfile -t scenes < <(cargo run -q --profile fast -- list 2>/dev/null | awk '{print $1}')
printf '%s\n' "${scenes[@]}" | xargs -P "$jobs" -I{} sh -c \
    "node tools/webshot.mjs --chrome '$chrome' --url '$url#'\$([ {} = graph ] && echo surface || echo graph) --out '$out/{}.png' --mode $mode --wait 2500 --then {} --stats ${seconds}000 2>/dev/null | grep '^stats:' | sed 's/^stats: //' > '$out/{}.json'"
python3 - "$out" "${scenes[@]}" <<'PY'
import json, sys
out, scenes = sys.argv[1], sys.argv[2:]
print(f"{'scene':<15} {'frame':>7} {'iface':>6} {'scene':>6} {'draw':>6} {'made':>6} {'first':>6} {'worst':>6} {'gap':>6} {'frames':>6} {'scale':>5} {'moving':>6}")
for s in scenes:
    try:
        d = json.loads(open(f"{out}/{s}.json").read() or "{}")
    except Exception:
        d = {}
    if d.get("scene") != s:
        print(f"{s:<15} (no reading)")
        continue
    print(f"{s:<15} {d['frame_ms']:>7.1f} {d['interface']:>6.2f} {d['scene_ms']:>6.2f} {d['drawing']:>6.2f} {d['made']:>6.1f} {d['first']:>6.1f} {d['worst']:>6.1f} {d['gap']:>6.0f} {d['frames']:>6} {d['scale']:>5.2f} {str(d['moving']).lower():>6}")
PY

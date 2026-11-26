#!/bin/bash
# Throw random input at every scene (or the ones named) and report what breaks.
#
#   tools/monkey.sh [-d] [-s SEEDS] [-f FRAMES] [-j JOBS] [-t SECONDS] [scene ...]
#
# Each run is `eidos monkey <scene> --seed N --frames FRAMES --finite`: random clicks
# and taps, drags (with either button, with shift held, with a finger), pinches,
# scrolling and zooming, key presses, typed text, and windows of many sizes, pixel
# ratios and frame rates. What it reports, each with the command that plays the same
# run again (add --verbose to see every action: the last one printed is the one that
# did it):
#   CRASH       a panic, and its message;
#   NOT FINITE  a frame that placed something where the numbers are not finite. The
#               scenes with a formula box are not asked this (a typed formula may
#               fairly be infinite everywhere): they are `formula_scenes` below, which
#               must name the same scenes as FORMULA_SCENES in src/headless/monkey.rs
#               (a test sees to it);
#   FAILED      the run ended in some other error;
#   SLOW        still going after SECONDS, which without a GPU usually means a scene
#               set to draw a great deal, not a hang.
# The script fails if anything but SLOW was reported.
# With -d nothing is drawn: every frame is still built, so everything but the renderer
# is tested, and where there is no GPU it is some fifty times faster (use it for many
# seeds, and a run without it for the renderer).
# Uses the binary in the cargo target directory (profile "fast"; or $EIDOS_BIN) and, where there is
# no GPU, Mesa's software renderer.
formula_scenes="calc1 surface descent field mcmc hermite_expand hermite_noise fit1d graph parametric"
seeds=4; frames=900; jobs=16; limit=300; dry=
while getopts "ds:f:j:t:" o; do case $o in d) dry=--dry;; s) seeds=$OPTARG;; f) frames=$OPTARG;; j) jobs=$OPTARG;; t) limit=$OPTARG;; *) exit 2;; esac; done
shift $((OPTIND - 1))
target=$(cargo metadata --format-version 1 --no-deps 2>/dev/null | sed 's/.*"target_directory":"\([^"]*\)".*/\1/')
bin=${EIDOS_BIN:-"$target/fast/eidos"}
[ -x "$bin" ] || { echo "build first: cargo build --profile fast" >&2; exit 2; }
if [ -z "$WGPU_BACKEND" ] && [ -f /usr/share/glvnd/egl_vendor.d/50_mesa.json ]; then
  export WGPU_BACKEND=gl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json
fi
scenes=${*:-$("$bin" list | awk '{print $1}')}
out=$(mktemp -d)
export bin frames limit out dry formula_scenes
one() {
  local scene=$1 seed=$2 finite=--finite
  case " $formula_scenes " in *" $scene "*) finite=;; esac
  local log="$out/$scene-$seed.log" again="$bin monkey $scene --seed $seed --frames $frames${dry:+ $dry}${finite:+ $finite}"
  timeout "$limit" "$bin" monkey "$scene" --seed "$seed" --frames "$frames" $dry $finite > "$log" 2>&1
  case $? in
    0) ;;
    124) echo "SLOW        $scene  seed $seed:  not finished after $limit s;  again: $again";;
    *) if grep -q panicked "$log"; then
         echo "CRASH       $scene  seed $seed:  $(grep -m1 -A2 panicked "$log" | tr '\n' ' ' | cut -c1-300);  again: $again"
       elif grep -q "not finite" "$log"; then
         echo "NOT FINITE  $scene  seed $seed:  $(grep -m1 "not finite" "$log" | sed 's/^eidos: [^:]*: //');  again: $again"
       else
         echo "FAILED      $scene  seed $seed:  $(grep -v libEGL "$log" | grep -v '^$' | tail -1 | cut -c1-300);  again: $again"
       fi;;
  esac
}
export -f one
for scene in $scenes; do for seed in $(seq 1 "$seeds"); do echo "$scene $seed"; done; done \
  | xargs -P "$jobs" -L 1 bash -c 'one "$0" "$1"' 2>/dev/null \
  | sed "s/^/  /" | tee "$out/summary.txt"
echo "$(ls "$out"/*.log | wc -l) runs of $frames frames${dry:+ (nothing drawn)}; logs in $out"
! grep -qE '^  (CRASH|NOT FINITE|FAILED)' "$out/summary.txt"

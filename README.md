# eidos

Interactive, GPU-rendered explorations of linear algebra, calculus, statistics, dynamics
and neural networks. Written in Rust on `wgpu`; the same code runs in a native window and
in the browser (WebGPU, falling back to WebGL2).

## Run it

**In the browser** (the way to use it on a machine without a display, such as a server
reached over SSH):

    ./build-web.sh        # once, and after changing the code
    ./serve.sh            # serves http://localhost:8765

Forward port 8765 to your own machine (VS Code: *Ports* panel → *Forward a Port*; plain
SSH: `ssh -L 8765:localhost:8765 <host>`) and open <http://localhost:8765>. All rendering
then happens on your local GPU. Each scene has its own address (for example
`http://localhost:8765/#hermite_orth`), so scenes can be bookmarked. `diag.html` on the
same address reports which graphics backend the browser offers and mirrors any console
errors on screen.

**Natively** (on a machine with a display):

    cargo run --release

## Controls

| | |
|---|---|
| drag | orbit (3D) or pan (2D); coloured dots are handles you can drag |
| right-drag, shift-drag | pan |
| scroll, pinch | zoom (towards the pointer in 2D) |
| double-click | reset the view (also `R`; in scenes where a click on empty space does something, only `R`) |
| `space` | pause / resume |
| `H` | hide the side panels |
| `F`, the button with four corner marks | full screen |
| `F1` | frame time, where it goes, and primitive counts |
| `PageUp` / `PageDown` | previous / next scene |
| search box | filter the scene list by any word |
| `?` button | what this is and how to use it |
| `G`, "glossary" | the words the scenes use, with links to the scenes |
| `M`, "the mathematics behind this scene" | the scene's mathematics written out, to read before the picture |

On a narrow screen (a phone, a small window) the picture takes the whole screen and the
scene list and the controls open from buttons at its top corners; the full-screen button
sits beside the first of them. An iPhone's browser has no full-screen mode for pages:
there, "Add to Home Screen" (the page carries a manifest and icons for it) gives an icon
that opens eidos without the browser around it.

Text boxes accept formulas as written on paper: `sin(x)^2 + axy`, `2x`, `x²`,
`exp(-x^2/2)`, `ln`, `sqrt`, `abs`, `tanh`, `sigmoid`, `relu`, `min(a, b)`, `atan2(y, x)`,
`pi`, `e`, and so on. Names side by side are multiplied (`ax` is a times x); a function
needs its brackets (`sin(x)`, not `sin x`).

## What is in it

- **Linear algebra**, as a course in seven parts.
  *Vectors and maps*: dot and cross products as a shadow and an area; span and
  independence; linear maps in 2D and 3D with draggable basis vectors; one map after
  another (products, order, inverses); determinants as area and volume; change of basis;
  rotations in 3D.
  *Equations*: elimination step by step, as planes meeting; the four subspaces; Gram–Schmidt
  and QR; least squares as a projection; too many equations (least squares, least absolute
  values and least maximum side by side); too few equations (the 1-, 2- and max-norm balls
  growing until they touch the answers: why the 1-norm finds sparse ones); sensitive
  equations and the condition number.
  *Eigenvalues*: finding eigenvalues (sweep a vector round the circle, slide λ until
  A − λI collapses the plane); powers of a matrix; quadratic forms as bowls, saddles
  and ridges.
  *Data and dimensions*: SVD as rotate–stretch–rotate; low-rank approximation of a
  picture; principal components; the Fourier basis; what high dimensions do to lengths
  and angles; compressed sensing; and spreading n directions apart so that the largest
  |dot product| is as small as it can be (line packings: the cube, the icosahedron, and
  the table up to 32 lines).
  *Special matrices*: a zoo of thirty kinds of 2 × 2 matrix (symmetric, orthogonal,
  projections, nilpotent, Markov, ...), each with its defining equation worked out, what
  it does to the plane and the region of the complex plane its eigenvalues must lie in,
  and a map of which kinds contain which; a second zoo of nineteen kinds of larger matrix
  known by their pattern (banded, circulant, companion, Toeplitz, Vandermonde, graph
  Laplacians, ...), with what each pattern buys; the symmetric and skew-symmetric parts
  of a matrix as a stretch plus a spin; projections and reflections; matrices that
  commute, and what the commutator measures when they do not.
  *Factorisations*: LU, QR, eigenvectors, the SVD and the polar form of one matrix as
  five films side by side; the spectral theorem as a sum of projections; Cholesky; polar
  form and the nearest rotation; the pseudoinverse; Schur form and the QR algorithm,
  with and without shifts; Jordan blocks and why defective matrices are delicate.
  *Exponentials*: the matrix exponential as a flow (saddles, spirals, centres, the map
  of every flow by trace and determinant, the series a term at a time); spin into
  rotation (Rodrigues' formula, the logarithm of a rotation); the exponential of a sum
  against the product of the exponentials, and splitting; powers, roots and logarithms
  of a matrix; coupled springs and their normal modes.
- **Calculus**: derivative / Riemann sums / Taylor polynomials of any `f(x)`; the
  fundamental theorem (the area so far, its tangent, and a strip shrinking to the curve's
  height); surfaces `z = f(x, y)` with tangent plane and gradient field; the Jacobian as
  a curved map seen up close (a patch, its true image and its linear image, zoomed until
  they agree); the chain rule in three planes, a small step pushed forwards by the
  Jacobians and a gradient pulled back by their transposes; curvature, the Hessian and
  Newton's method against gradient descent; Lagrange multipliers (the path touching a
  contour line, and what the multiplier measures); changing variables in an integral
  (why polar coordinates bring an extra r); seven optimisers on nineteen
  loss surfaces, each chosen to show one behaviour (ravines, saddles, plateaus, cliffs,
  sharp and flat minima, ...), with a contour map and a loss-against-step chart;
  vector-field flow with thousands of particles; Fourier epicycles (draw your own).
- **Dynamics**: strange attractors, double-pendulum chaos, the wave equation with
  interference and slits.
- **Statistics**: a course of twenty-four scenes, each built on the same few charts.
  *Chance and distributions*: probability as areas of a square of a thousand people
  (conditional probability, independence, Bayes' rule and the base rate); fourteen
  distributions with density, cumulative curve and quantile tied together; the mean as a
  balance point, with variance, skewness and tails; the law of large numbers (and the
  Cauchy distribution, where it fails); the central limit theorem; correlated normals and
  conditioning.
  *From data to estimates*: looking at a sample (histogram and bin width, kernel
  density, box plot, quantile plot); estimators with their bias, variance and mean squared
  error (why n − 1); likelihood and its peak; confidence intervals and what "95 %" is a
  promise about; the bootstrap; Bayesian updating from prior to posterior; MCMC samplers.
  *Tests*: one test taken apart (null, statistic, p-value, level, the two kinds of error,
  power); what p-values do over many studies (multiple comparisons, peeking, the winner's
  curse); comparing two groups four ways (Welch's and the pooled t test, shuffled labels,
  ranks) and pairs; analysis of variance; chi-square tests of fit and of independence.
  *Models*: correlation and regression to the mean; least-squares fitting with visible
  squares and a chart of its error on the data and against the true curve, degree by
  degree; regression with several inputs (holding fixed,
  collinearity, leverage); logistic regression and the ROC curve; choosing a model by
  cross-validation; entropy, cross-entropy and the Kullback–Leibler divergence.
- **Causal inference**: confounding and Simpson's paradox; seeing against doing (an
  intervention cuts the arrows into what is set); chains, forks and colliders; which
  variables to adjust for (the back-door rule); counterfactuals for one unit. Then finding
  the direction of an arrow from data alone: fit each variable from the other and see in
  which direction what is left over is independent of the input (additive noise models,
  with a dependence test against shuffled pairings); LiNGAM; noise whose size depends on
  the cause (location-scale models: four ways of fitting level and spread together,
  counted against each other, and what a hidden common cause does to the check); finding
  a whole causal order; and the PC algorithm.
- **Hermite polynomials**: the family with its zeros, ladder operators and Gauss–Hermite
  nodes; orthogonality under the Gaussian shown as cancelling areas, as a Gram matrix
  built up by Gram–Schmidt, and as integration by parts down the ladder; Hermite
  expansions of any function with their spectrum; the noise operator and Mehler's
  ρⁿ scaling with the resulting kernel; and two directions in Gaussian space overlapping
  by cosⁿθ.
- **Neural networks**: a live training playground with nine activations (including SiLU,
  GELU, sin and Hermite polynomials) and seven optimisers (SGD, momentum, Nesterov,
  AdaGrad, RMSProp, Adam, Muon), on two kinds of task. *Points in
  the plane* (thirteen datasets) shows what every neuron responds to and the decision
  boundary forming; click a neuron to select it and again to knock it out, with a bar
  under each one for how much the loss would rise without it. *Arithmetic tables* is
  described under interpretability. *Optimiser race*: the same network, starting
  weights and batches in every lane, with only the optimiser differing. *Neurons as
  particles*: a wide layer drawn as points in weight space, fitting the same data by
  moving its neurons (feature learning) or by leaving them in place (the lazy regime),
  depending on one scale. *Signal propagation*: what depth does to the correlation
  between two inputs in a random network, as a map whose power series is the
  activation's Hermite spectrum, with the order and chaos phases and a real network as
  a check. *How layers fold space*: the input plane carried through the layers, with
  every activation, optimiser and training setting adjustable. Also: building a
  function out of hinges; a true 2D slice of a loss landscape with the optimiser's path.
- **Backpropagation**: a course of eleven scenes on how the gradient of a loss is handed
  back through a network, and what each design choice does to it. Every gradient drawn is
  worked out by reverse-mode differentiation on small matrices (`src/math/autograd.rs`,
  with networks built from a specification in `src/math/arch.rs`) and tested against
  finite differences of the loss. *Derivatives by machine*: a formula as a computation
  graph, one sweep forwards and one back, against forward mode and finite differences.
  *The smallest network*: two inputs, two hidden units and one output with every number
  shown, a step at a time. *A layer at a time*: the same sweep in matrices for a batch,
  a weight's gradient as a sum of outer products. *Where the gradient starts*: outputs
  and losses in four pairings, and which of them stall. *Activation functions as gates*:
  saturating, dying and leaking, with a live map of every unit's gate while a network
  trains. *Gradients through depth*: vanishing and exploding across fifty layers, the
  starting weights that sit on the edge, and the singular values of the whole stack's
  Jacobian. *Normalisation*: what batch, layer and RMS normalisation drop from the
  gradient and what they do to it over depth. *Skip connections*: fifty plain layers
  against fifty with skips, and the gradient multiplied out into paths (three blocks are
  eight), with which lengths carry it. *Attention*: the two roads back through a head, by its
  values and by its pattern. *A transformer*: one loss's gradient at every place of the
  residual stream, pre-norm against post-norm, with and without skips, before and after
  training. *The architecture playground*: build an MLP or a transformer (depth, width,
  activation, normalisation and where it goes, skips, starting weights, optimiser), keep
  a second design beside it, and compare signals, gradients, the Jacobian's stretch and
  training itself.
- **Interpretability**: opening a trained network up. *Superposition*: a toy model
  packing more features than it has dimensions (pairs, a pentagon, polyhedra), and what
  the overlap costs when one feature is switched on alone. *Sparse autoencoders*:
  getting those features back from the activations alone, with the trade-off set by
  the sparsity penalty (dense latents, shrinkage, dead latents), dictionaries too small
  and too large, and the contrast with PCA. *Probing*: what a straight-line read-out
  can tell from each layer, and a causal test (push the activations across) of whether
  the network uses it. *Grokking a table*: a network memorising a + b mod p and then
  finding the rule, with every neuron's activity, each neuron settling on a frequency,
  the numbers laid out on one circle per frequency, and frequencies that can be
  removed. *A transformer, opened up*: a small transformer trained in the browser, with
  its attention patterns, the logit lens, direct logit attribution, head knock-outs
  and activation patching.
- **The Jacobian lens**: the ideas of Ostrow, Qian, Liu and Fiete, "A Mathematical
  Theory of the Jacobian Lens" (2026). *Lyapunov exponents*: how a nudge grows or dies
  through the layers of a random network, the whole spectrum, the circle of nudges
  becoming an ellipse, and what averaging Jacobians over inputs does. *What averaging
  Jacobians keeps*: a two-dimensional residual stream in which the three factors
  (magnitude, lag coherence, context coherence) can be seen as nested curves, with the
  paper's variance corollary and the ω-lens. *The Jacobian lens on a belief filter*:
  the paper's ring hidden Markov model and its Bayes filter, the three factors in
  closed form and measured, the paper's four experiments as presets, and the predicted
  crossing point.
- **Case studies** (a section of its own near the end of the list, apart from the
  course: these are about particular published studies, not about a piece of
  mathematics): does vaccination against shingles lower the risk of dementia? A map of
  the published studies as one causal graph, showing which paths each kind of design
  closes, leaves open or must assume absent, and a close look at one natural experiment
  (Taquet and others, 2024) with a toy model of which mechanisms could and could not
  give its pattern. They teach how to read the evidence; they give no verdict on the
  vaccine and are not medical advice.
- **Glossary** (G): the words the scenes use, each in a sentence with a link to
  the scene that shows it.
- **Plotting**: graph any set of functions with parameters and time; parametric surfaces.

## How it is built

```
src/app/            panels, input, timing; independent of windowing, so it also runs headless
                    (beside it: the full-screen button, the fonts and colours, and the
                    governor that trades resolution for frame rate on a slow GPU)
src/headless/       the application without a window: the command-line tools (shot,
                    bench, monkey, check) and what the scene-contract test runs
src/background.rs   work off the main thread: a thread natively, a Web Worker in the browser
src/clock.rs        a clock that works in the browser, time budgets, and pacing for work
                    that shares the frame
src/frame.rs        what a scene sees each frame: drawing API, handles, axes, text
src/ui.rs           the widgets the control panels are built from
src/typeset/        mathematics between dollar signs, typeset as TeX does: the markup's
                    parser, the layout (boxes, spaces, scripts, fractions, brackets), and
                    the painter that makes text and formulas one galley and keeps it
src/gfx/            renderer: one multisampled HDR pass, bloom, tone mapping
  mod.rs            its state, and the frame: fingerprint, upload, scene pass, post passes
  draw.rs           the list of things to draw that scenes fill in each frame
  surface.rs, image.rs, trails.rs   the objects scenes keep on the GPU
  pipelines.rs, targets.rs, gpu.rs  pipelines, render targets, buffer helpers
  camera.rs, color.rs               what their names say
src/gfx/shaders/    WGSL
assets/             a fallback font for mathematical symbols, and the fonts formulas are
                    set in (see its README)
src/math/           formula compiler with automatic differentiation, linear algebra
                    (small matrices, general SVD, orthonormalisation), complex numbers,
                    Hermite polynomials and quadrature, a dense neural network with
                    nine activations and seven optimisers, a small transformer with
                    hand-written gradients and interventions, a seedable RNG
src/glossary.rs     the words the network scenes use, each with the scene that shows it
src/scenes/         the catalogue
  mod.rs            the Scene trait, SceneDef, the list of topics
  prelude.rs        one import that gives a scene everything it normally needs
  kit.rs            what more than one topic needs: charts, screen-space panels, legends,
                    layouts that follow the canvas, histories for long runs
  <topic>/mod.rs    the topic's scene list and whatever its scenes share
  <topic>/<scene>.rs  one scene: state, controls, per-frame update and drawing
tests/scenes.rs     the scene contract, held against every scene in the catalogue
```

## Adding a scene

1. Create `src/scenes/<topic>/<name>.rs`. The smallest complete scene:

   ```rust
   //! Lissajous figures.

   use super::*;

   pub const DEF: SceneDef = SceneDef {
       id: "lissajous",
       title: "Lissajous figures",
       group: "Plotting",
       blurb: "Two oscillations at right angles.",
       make: |_gfx| Box::new(Lissajous { a: 3.0, b: 2.0 }),
   };

   struct Lissajous {
       a: f32,
       b: f32,
   }

   impl Scene for Lissajous {
       fn view(&self) -> View {
           View::Fit { center: Vec2::ZERO, half: Vec2::splat(1.3) }
       }

       fn ui(&mut self, ui: &mut egui::Ui, _: &Gfx) {
           ui::slider(ui, "a", &mut self.a, 1.0..=9.0);
           ui::slider(ui, "b", &mut self.b, 1.0..=9.0);
       }

       fn frame(&mut self, f: &mut Frame) {
           let (a, b, t) = (self.a, self.b, f.t);
           f.curve(0.0, TAU, 800, 2.5, pal::CYAN, |s| Vec3::new((a * s + t).sin(), (b * s).sin(), 0.0));
           f.axes2();
       }
   }
   ```

2. Add `lissajous` to the `topic!(...)` list in `src/scenes/<topic>/mod.rs`.

That is all: the catalogue, search, links (`#lissajous`), `--scene lissajous`, `shot` and
`bench` pick it up. A new topic is a new directory with a `mod.rs` like the others, plus
its name in `TOPICS` in `src/scenes/mod.rs`.

**The scene contract.** From the moment it is listed, a scene is also held to what
every scene must do, without a line of test written for it (`cargo test` runs
`tests/scenes.rs`; `eidos check` prints a line per scene):

- it opens, and gets through frames in every kind of window: a desktop one with the
  panels beside the picture, a phone's (controls closed and open), one far too small,
  and one with the panels hidden;
- it places nothing where the numbers are not finite: no point, triangle, picture or
  label at NaN or infinity, whether left alone or under a short run of random input.
  (A curve is another matter: give it NaN where it has no value and it comes out with
  a gap there; a surface likewise gets a hole. Nothing that is not a finite number is
  ever sent to the GPU, whatever a scene asks for. A scene with a formula box is
  excused under random input, since a typed formula may fairly be infinite everywhere:
  see `FORMULA_SCENES` in `src/headless/monkey.rs`.)
- if `animated()` says it only changes in answer to input, it really comes to rest: its
  picture stops changing, and it stops asking for new frames;
- its panel has the parts every panel has (below): things to try, a colour key, and
  links to related scenes, which lead to scenes that exist; its id can stand in a page
  address, and its catalogue entry is complete.

What a scene does under much more random input (clicks, drags with either button,
touches and pinches, typing, every window size, pixel ratio and frame rate) is for
`tools/monkey.sh`, which fails on a crash or on drawing that is not finite; whether
what its captions say is true is for its own tests.

What a scene can draw (all through `Frame`): lines, arrows, dashed lines (`dashed`)
and dashed curves (`dashed_path`), glowing curves;
discs, lit spheres and glows; filled polygons; text anchored to world positions;
draggable handles; and three retained GPU objects: `Surface` (any grid surface or smooth
heat map; a vertex given NaN is a hole), `TrailSet` (motion trails for many particles) and `Image` (a grid of colours:
tables, matrices, or many small pictures laid out in one image and drawn together).
Coordinates are world units; line widths and point sizes are pixels. Panels that must
stay put in a 3D view (a chart, a matrix) are drawn in screen pixels with the `hud_*`
helpers in `scenes/kit.rs`. For a line or point style the built-in batches do not
offer, ask for one: `f.dl.lines_with(LineKind::Blend, LineStyle { dash: 4.0, ..Default::default() })`.

Return `false` from `animated()` if the scene only changes in response to input: it is
then redrawn on demand rather than every display refresh.

**The panel.** Every scene's panel follows one order, so that a visitor always knows
where to look: a short paragraph on what this is; *Things to try* (`ui::experiment`: a
button that sets the scene up for one lesson, with a line saying what to look for; it
sets every setting the lesson depends on, so it shows the same thing whatever was done
before); the controls, each non-obvious one with a one-line `ui::note`; *Reading the
picture*, a colour key (`ui::key`); *Words* (`ui::term`) for the technical terms the
panel uses; the numbers (`ui::readout`); and links to related scenes (`ui::see_also`).
Add a term to `src/glossary.rs` if other scenes use it too. Whatever an experiment's
caption says should happen is checked by a test in the scene's file, which sets the
scene up by the same function the button calls. The scene check holds every scene to
having things to try, a key and links. `src/scenes/neural/fit1d.rs` is a small scene
to copy the order and tone from.

**The picture.** A first-time visitor should be able to read it without the panel:
name every curve, colour, axis and number where it is drawn (`f.caption_px` for text
on a plate, `hud_key` for a legend, `hud_heat` for a matrix). A flat scene whose
layout should follow the shape of the canvas (wide on a desktop, tall on a phone) can
use the "stage" helpers in `kit.rs` (`STAGE_VIEW`, `stage_width`, `wrapped`); one laid
out in fixed world units can at least start zoomed on its main picture on a phone
(`narrow_view`).

A scene whose work per frame can grow (training, a long simulation) should not do a
fixed amount of it in the frame. Two tools, both used by the playground
(`neural/training.rs`, `neural/session.rs`):

- **A background service.** Implement `background::Service` (handle a byte message, do a
  slice of work, send results back) and add it to `scenes::services()`. It then runs on
  a thread natively and in a Web Worker in the browser (`web/worker.js` starts a second
  copy of the program for it from the page's own compiled module), and the scene talks
  to it through `background::link()`.
  It works only while its owner keeps asking after it, so a hidden tab costs nothing.
- **A time allowance**, for when there is no background (it has not started within a
  second, the browser refused it, it failed, or the run must be reproducible):
  `clock::Budget::new(seconds)`, and stop when `spent()`. For work asked for at a rate,
  `clock::Pace` does the counting: `pace.begin(rate, f.dt, f.real_dt, most)` owes
  `rate` steps for each second of the scene's time and allows a share of the real time
  between frames.

Two rules of time that every scene keeps. `f.dt` is the scene's time (it follows the
time-speed control and is zero while the app is paused) and is what anything that moves
or is counted "per second" uses; `f.real_dt` is real time and is only for sizing time
allowances. And nothing is done "once per frame" that should be done "once per so much
time": displays run at 30 to 240 frames a second, so ease with `1 - exp(-dt / tau)`,
and step and record by time owed.

Whole steps do not come evenly, though: 90 a second on a display that shows 60 is one
step in a frame, then two. Where a step moves the picture visibly, do not draw the
state as the steps leave it, or it jerks. Draw a copy that follows it (`follow` in
`linalg/packing.rs`: two eases one after the other, exactly there in the end, and
`animated()` until then). The same goes for a value that snaps while it is dragged
(`glide` in `linalg/vectors.rs`): the numbers step, the picture glides.

A label that looks for a clear place each frame, with nothing to remember the last
frame by, jumps to and fro between two places about as clear as each other, all the
while something is dragged. Draw such a label with `f.text_placed(name, number, from,
offset, …)`: the frame remembers where each one was (`f.places.chosen`), so that a
scene can leave it there while it still has room, and a label whose place does change
is drawn gliding to it (`Room::label_any` in `linalg/vectors.rs`, `label_beside` in
`linalg/linmap3.rs`).

`eidos motion [scene...]` measures all of this: how far the picture's furthest-moving
point goes in each frame, as it opens, after each of its "things to try" and (with
`--drag`) while a handle is dragged, and from that the jitter, the stalls, the jumps
and the labels that hop (`src/headless/motion.rs` says what each means).

Either way the frame rate does not depend on the settings, with one exception: a single
step longer than a frame's allowance cannot be split. `Pace` then takes one only every
few frames, so such work gets at most a third of the time; the frame that takes it is
still as long as the step. Headless renders use neither
(no background, budgets never run out), so they stay reproducible.

### Writing mathematics

Every formula on the screen is typeset, the way TeX sets it, in the fonts of KaTeX
(Computer Modern): never the interface font's letters, Unicode superscripts or ASCII.
It is set a tenth larger than the words round it (`typeset::TEXT_SCALE`, a compromise:
matched by x-height its digits would stand a size above the words', matched by the
digits its letters would look a size too small); in a monospaced read-out, its digits
exactly as tall as the read-out's. Text stays a plain string; mathematics goes between
dollar signs, as in LaTeX:

```rust
ui::note(ui, r"Two arrows, $a$ and $b$, and their dot product $a \cdot b = |a|\,|b| \cos\theta$.");
f.text_off(tip, Vec2::new(8.0, 10.0), format!(r"$\lambda = {}$", fmt(l)), 13.0, c, Align2::LEFT_TOP);
```

Every function that puts words on the screen takes it, with no new call to learn: in
the panel `ui::note`, `ui::term`, `ui::key`, `ui::experiment` (caption and button),
`ui::readout` and its kin (label and value), `ui::see_also`, `ui::section`, the labels
of sliders, `ui::buttons`, `ui::choice` and `ui::checkbox`, a scene's `title`, `blurb`
and `pointer_hint`, the glossary; on the picture `f.text`, `f.text_off`, `f.text_px`,
`f.text_placed`, `f.caption_px` and what `kit` builds on them (`bottom_caption`,
`hud_key`, `hud_matrix`, `wrapped`, `spread_labels`, `Room::label`). Not egui's own
widgets called directly (`ui.checkbox`, `ui.label`: give them `ui::label_text(…)`,
which also lets the lint read the text), and not what is typed in the formula boxes
(`ExprField`), which is something to type (their labels, "f(x) =", are typeset by the
box). Text without a dollar sign is drawn exactly as before. Use raw strings (`r"…"`), so that
a backslash is a backslash.

**What goes in dollars.** Every variable, even a lone one in a sentence (`the white dot
$v$`, `the $x$ axis`); every equation and formula (`$\det A = 1$`, `$Ax = b$`); and the
numbers that are part of one (`$\lambda = 1$`, the point `$(2, 1)$`, `$0.75 a_1 + 0.5 a_2$`,
a matrix). Not: counts and quantities in prose ("after 30 steps", "3 million people",
"turns by 45°", "from the plane as it was (0)"), names ("ReLU", "Adam") and words. A
read-out's value is a number, and stays plain unless it is a formula: a point, a
complex number, or anything `fmt` may write as `1.5e-4` (wrap it: `format!("${}$",
fmt(v))`, and then every value of that read-out list alike). From the pilot scenes
(`linalg/linmap2.rs`, `linalg/lsq.rs`, `neural/fit1d.rs`; read them before converting):

| before | after |
|---|---|
| `"… keeps its area: det = 1. Only one line stays on itself: the x axis (λ = 1)."` | `r"… keeps its area: $\det A = 1$. Only one line stays on itself: the $x$ axis ($\lambda = 1$)."` |
| `"the arrow î = (1, 0) lands"` | `r"the arrow $\textcolor{pink}{\hat\imath} = (1, 0)$ lands"` |
| `"AᵀA x* = Aᵀb. They say a₁ · e = 0 …"` | `r"$A^\top A x^* = A^\top b$. They say $a_1 \cdot e = 0$ …"` |
| `"the parabola (x²−1)/√2"` | `r"the parabola $(x^2 - 1)/\sqrt{2}$"` |
| `format!("σ{} = {}", k + 1, fmt(s))` | `format!(r"$\sigma_{{{}}} = {}$", k + 1, fmt(s))` |
| `format!("{} ± {}i", fmt(re), fmt(im))` | `format!(r"${} \pm {}i$", fmt(re), fmt(im))` |
| `"p = Ax*"`, `"a₁"`, `"î"` (labels) | `"$p = Ax^*$"`, `"$a_1$"`, `r"$\hat\imath$"` |

**The markup** is the LaTeX you know. Letters are variables, italic, each its own (a
word in italic is a mistake: `\text{…}` or `\operatorname{…}`, below); digits,
punctuation and brackets upright; spaces typed in a formula count for nothing, and TeX's
spacing does the rest (none round a minus with nothing before it).

| | |
|---|---|
| Greek | `\alpha` … `\omega`, `\varepsilon`, `\varphi`, `\Gamma` … `\Omega` (upright), or typed: `λ`, `Δ` |
| scripts | `x^2`, `x_1^2`, `A^{-1}`, `A^\top`, `x'`, `\sigma^2_{k+1}` (stacked); `^` and `_` take one token or a `{group}` |
| operators, relations | `+ - \pm \times \cdot \circ \otimes`, `= \ne \le \ge \approx \equiv \sim \propto \in \notin \subseteq \perp \mid`, `\to \mapsto \Rightarrow \iff`, `\ll \gg`, `\lesssim` |
| fractions, roots | `\frac{a}{b}` (`\dfrac` full size, `\tfrac` small), `\binom{n}{k}`, `\sqrt{x}`, `\sqrt[3]{x}` |
| brackets that grow | `\left( … \right)`, also `[ ] \{ \} | \| \langle \rangle \lfloor \rfloor \lceil \rceil .` and `\middle|`; fixed sizes `\big( \Big( \bigg( \Bigg(` |
| matrices | `\begin{bmatrix} a & b \\ c & d \end{bmatrix}`; `pmatrix`, `vmatrix`, `Vmatrix`, `Bmatrix`, `matrix`; a column vector is one column; `cases` |
| functions | `\det \tr \rank \dim \ker \span \diag \cos \sin \tan \tanh \exp \ln \log \arg \sgn` and, with limits under them in a display, `\max \min \sup \inf \lim \argmax \argmin`; anything else `\operatorname{softmax}` (`\operatorname*` for limits) |
| large operators | `\sum \prod \int \oint \bigcup \bigcap`, with `_{…}^{…}`; `\limits`, `\nolimits` |
| words | `\text{if } x > 0`, `\text{length}^2`, `\textbf{…}`: in the font, size and weight of the words round the formula (the monospaced one in a read-out), on its baseline, scripts and all |
| alphabets | `\mathbf{x}`, `\mathbb{R}` (capitals), `\mathrm{d}x`, `\mathit{…}` |
| accents | `\hat x \bar x \vec v \tilde x \dot x \ddot x`, `\hat\imath \hat\jmath` (dotless), `\overline{AB}`, `\underline{…}` |
| spaces, dots | `\, \: \; \! \quad \qquad`, `~`; `\ldots \cdots \vdots \ddots` |
| colour | `\textcolor{pink}{a_1}`; `{\color{green} …}` for the rest of a group |
| a dollar sign | `\$`, in a formula or out of one |

The Unicode the text was written in before means the same, so that converting a string is
mostly putting the dollars in: `$x₁² + σ²ₖ₊₁ ≤ Aᵀ A⁻¹$`, `$î ≠ ĵ$`, `$λ ∈ ℝ$`, `$√n$`
(`ᵀ` is `\top`). A number written the way computers write it, `2.5e-3`, is set as
2.5 × 10⁻³, so `fmt` and `kit::num` can go straight into a formula; so can a hyphen,
which is a minus sign there.

**Which form.** In running text, a fraction of more than single symbols is written with
a slash (`$(a_1 \cdot b) / (a_1 \cdot a_1)$`): `\frac` in a line is set small, as it is
in print. `\frac` is for displays and for labels with room. A matrix in running text is
rarely readable: put it in a display (`ui::equation`), or write the point `$(1, 0)$`.
Units stay in prose where they can ("45°", "in 10 ms"); inside a formula,
`$v = 3\,\text{m/s}$`. A word that is part of a formula goes in `\text`, which sets it
as the words round it are: `$\text{length}^2$`, `$|\text{dot product}|$`, not
`length$^2$` or `$|$dot product$|$`. A function's name is mathematics, not a word:
`\det`, `\cos`, `\operatorname{softmax}`, upright in Computer Modern.

**Displays.** `ui::equation(ui, r"\hat y(x) = c + \sum_{j=1}^{n} v_j \, \sigma(w_j x + b_j)")`
sets a formula on a line of its own in display style (full-size fractions, limits above
and below), centred in the panel. The text is the formula, with or without dollars. One
wider than the panel is broken where a mathematician would break it: between two
formulas set side by side (after `,\quad`), or else before a relation, or else before a
binary operator, the relation or operator carried to the new line, and every line
centred; never before its first relation, which would leave the left side alone. To
choose the breaks yourself, write `\\` where a new line starts:
`r"X = aZ + \sigma_X \varepsilon_X, \\ Y = bX + cZ + \sigma_Y \varepsilon_Y"`. (In running
text a formula is broken after a relation or an operator, as TeX breaks it, and only
if it does not fit on a line of its own.)

**Colour.** A formula's terms can carry the colours of what they name in the picture:
`\textcolor{name}{…}` with the palette's names (`text dim faint muted link cyan blue pink
yellow green orange purple red white`) or `#rrggbb`; `kit::tint(color, body)` writes it
for any `Color`. A label's own colour (the colour given to `f.text`) is the default.

**Building formulas.** In `format!` a literal brace is doubled: `format!(r"$\sigma_{{{}}}
= {}$", k, fmt(v))` gives `$\sigma_{2} = 1.000$`. Helpers that write the markup:
`kit::num(v, decimals)` (a true minus sign), `kit::tuple(&[x, y], 2)` (`(1.50, −0.50)`),
`kit::bmatrix(&[[a, b], [c, d]], 2)` and `kit::pmatrix`, `kit::column(&v, 2)`,
`kit::tint`, and `kit::activation_name(act)` for an activation's name in a panel
(`Act::name` is the plain one). They give the markup without the dollars, to go inside a formula:
`format!("$A = {}$", kit::bmatrix(&rows, 2))`.

**Measuring.** A label's size on the screen, formulas and all, exactly:
`f.text_size(text, size)` (kept from frame to frame, so cheap for a label drawn every
frame). `kit::text_width(text, size)` is the estimate for when there is no frame (a
unit test, a layout decided beforehand): it counts the characters of words but
measures formulas exactly, not their markup (and keeps what it measured while it is
asked again in every frame, so it is cheap too). The `kit` helpers that place labels
(`Room::label`, `bottom_caption`, `spread_labels`, `wrapped`) measure text with
mathematics in it exactly already; `wrapped` breaks a text into lines once and keeps
them while it is drawn at the same size and width, so a scene need not keep its own.
Its lines are `size + 3` apart, more where a formula is taller than the words:
`kit::wrapped_height` says how tall a note is, where counting its lines would not. `hud_matrix`'s `widest` counts the characters of
a cell's number, not of its markup.

**Mistakes** show: markup that does not parse (a lone `$`, an unclosed brace, an unknown
command) is drawn as written, dollars and all, fails `eidos check` and the scene
contract if the scene shows it as it opens, and is listed by the lint:

    cargo run --profile fast -- mathlint linmap2 lsq     # or all scenes and the glossary

which reads every text a scene shows (panel and picture, as it opens and after each of
its things to try) for (a) markup that does not parse, (b) mathematics left outside
dollars, and (c) words set in italic inside them. (b) has false alarms by nature: a text
looked at and found right as it is goes in `src/headless/mathlint-checked.txt`
(`scene: part of the text`). A scene is converted when its (a), (b) and (c) are 0.

### The mathematics behind a scene

Each scene can have a **primer**: its mathematics written out for someone who has not met
it yet, opened over the application by the button under the scene's blurb (or `M`). One
file a scene, `src/primers/<scene id>.md`, compiled in (`src/primer.rs` lists them; an
empty file means no primer yet and no button). The format is a small subset of Markdown
with the same `$…$` mathematics: `# The idea`, `# The mathematics` (with `## ` headings
inside) and `# In the picture`, in that order; paragraphs; formulas on a line of their own
between `$$ … $$`; lists (`- `, `1. `); one line set apart with `> ` saying what to try
first; and a last line `see: id, id` naming related scenes. `src/primers/vectors.md` is
the model.

    cargo run --profile fast -- primers [scene...]       # what is wrong with a primer: its parts,
                                                         #   length, formulas, loose mathematics, links
    cargo run --profile fast -- shot vectors out.png --primer     # the primer as a visitor sees it
    EIDOS_PRIMERS=some/dir eidos primers vectors         # the same for files not yet compiled in

## Why it is fast

- **Nothing is drawn twice.** Every frame the renderer fingerprints everything that
  determines the image (camera, settings, every batch, the versions of retained objects).
  If it matches the previous frame, no GPU work happens at all: a still scene is not
  drawn again while the pointer moves over it or over the panels. `F1` shows how many
  frames were drawn and how many reused.
- **Resolution follows the GPU.** While a scene animates, frame intervals are watched.
  Sustained slowness lowers the render resolution; after a few comfortable seconds a
  higher one is tried again. If even a quarter of the pixels makes frames no faster,
  resolution was not the problem and it is left at full. A picture at rest is always
  drawn at full resolution. A new scene starts at the resolution the last one ended
  at and tries the full one after half a second, so a slow device does not begin
  every scene with seconds of slow frames. (*View* shows what it is doing, and has
  the switch.)
- **Text is set once.** A paragraph with formulas in it is built from many small
  requests to egui (a word measured, a letter of a formula placed), each about ten
  microseconds when the answer is not at hand. What was built is kept, by text, size
  and width, also when it is not being drawn: the panel of a scene opened a second time
  costs nothing to set. Widths of words, the fonts' measures and formula letters laid
  out are remembered from one paragraph to the next. The panel's text is wrapped to a
  width that does not depend on its scroll bar (which egui slides in over half a dozen
  frames: every paragraph used to be set again in each). And the heights of letters,
  which were found by drawing each one 200 points tall, come from a table for the
  common ones: those drawings were most of what egui's font atlas held.
  A paragraph that starts below what is in sight is not set in the frame that first
  meets it if that frame is already 4 ms old: it is given room by a guess and set a
  frame or two later (what is in sight never waits, so nothing a visitor is looking at
  moves). On a dense display letters are placed on whole pixels, one drawing each,
  where on a coarse one they are placed to a quarter of a pixel, up to four each.
- **A slow device is drawn for plainly.** The full drawing is four samples a pixel in
  16-bit colour with a glow on top. Where a quarter of the pixels is still too slow,
  and from the first frame where no GPU is doing the drawing at all (a browser with
  hardware acceleration off, a remote desktop), the picture is drawn with one sample
  in 8 bits and no glow: lines and dots keep their smooth edges, which the shaders
  compute. In a browser without a GPU that took a moving scene from 260 to 490 ms a
  frame to 75 to 110. Frames of a quarter of a second and more count as slowness when
  they come three in a row; one or two are a hitch, whatever their length.
- **Lines** are instanced quads with exact round joins computed in the shader. One point
  buffer is bound four times at different offsets, so a polyline costs 20 bytes per
  vertex and one draw call per batch.
- **Trails** (attractors, flow fields, pendulums) keep their history in a ring buffer on
  the GPU laid out one row per time step. Advancing the simulation uploads a single row
  however long the trails are; fading with age is done in the vertex shader. Hundreds of
  thousands of segments are one draw call.
- **Surfaces** have no vertex or index buffers. Positions live in a float texture; the
  vertex shader derives topology from the vertex index and normals from neighbouring
  texels. Contour and grid lines are computed per pixel, so they stay sharp at any zoom.
- **Spheres** are billboards shaded as true spheres with correct depth.
- **Training happens off the main thread.** The playground's networks train on a
  thread of their own (a Web Worker in the browser, started from the page's own
  compiled program, so nothing is downloaded or compiled twice) at a chosen number of
  steps per second, or as fast as the machine goes; the frame only draws the latest
  weights, which costs about half a millisecond. Where no background is available, or
  the worker fails, training shares the frame within a time allowance instead.
- **Networks** store one column per neuron, so every inner loop runs along the batch
  and compiles to SIMD; `exp`, `sin` and `tanh` are short polynomial forms rather than
  library calls. One pass over the playground's 7,744-point grid through four layers
  takes 0.2 to 1 ms depending on the activation. Inputs that are one-hot codes are
  gathered, not multiplied. Numbers too small to matter are flushed to zero, because
  arithmetic below the normal floating-point range is very slow and a saturated network
  produces it constantly. For the browser the four innermost loops are also written
  out with WebAssembly SIMD instructions (bit for bit the same results), which the
  browser's compiler handles better than the automatic version.
- **Pictures** of many neurons are painted into one image and drawn with one upload and
  one draw call, however many there are.
- **Formulas** compile to bytecode that is evaluated one operation at a time over whole
  arrays, so the choice of operation is made once per array rather than once per number.
  The loops of plain arithmetic, square roots and absolute values are vectorised by the
  compiler (the browser build uses WebAssembly SIMD); sin, exp, ln and the other
  functions are a library call for each number. The same bytecode runs on dual numbers
  for exact gradients and on truncated power series for Taylor coefficients.
- Draw-list buffers are reused between frames, uniforms are uploaded only when they
  change, scenes are constructed only when first opened, the glow passes are skipped
  when glow is set to zero, and the browser build is served compressed (about 3 MB
  instead of 7.5).
- **The download is not larger than it has to be.** The code that runs once (shader
  translation and checking, drawing a glyph the first time it is needed) is a third of
  the program and is built for size; everything in an inner loop is built for speed.

Cheap is not the same as smooth: `eidos motion` measures how evenly each scene moves
from frame to frame (see "Adding a scene").

To see where time goes: `F1` shows the frame time and how much of it this thread spent
on the interface, on scene code and on issuing drawing commands (the rest is the GPU
and the browser), with the graphics adapter in use, how long the scene took to open and
its longest frame since. `cargo run --release -- profile` follows a visit to each scene
from the click that opens it (time to open, the longest frame and how many were long,
early frames against late ones, bytes sent to the GPU a frame); `tools/webbench.sh`
reads the same from the browser build in a headless Chrome. `cargo run --release -- bench`
prints, per scene, what a visitor's frame costs this thread (scene code and panels;
training on the background thread is not in it) and how many frames were actually drawn. `cargo run --release -- bench-training` prints training steps per
second; `node tools/wasm-bench.mjs` runs the same benchmark inside the browser build.

## Development

    cargo test --profile fast                   # the maths, the simulations, what the captions
                                                #   claim, and the scene contract for every scene
    cargo test --profile fast -- --ignored      # the slow ones: captions checked over many runs
    cargo clippy --all-targets                  # lints (two are allowed on purpose: Cargo.toml)
    cargo fmt                                   # formatting (rustfmt.toml)
    cargo run --profile fast -- check           # the scene contract, a line per scene
    cargo run --profile fast -- mathlint        # what is left to typeset (see "Writing mathematics")
    cargo run --profile fast -- primers         # what is wrong with the scenes' primers
    tools/monkey.sh                             # random input at every scene, to find crashes and
                                                #   drawing that is not finite (fails if it finds any)
    tools/monkey.sh -d -s 24 -f 2000            #   the same without drawing: many more runs
    cargo run --profile fast -- bench           # what a visitor's frame costs this thread, per scene
    cargo run --profile fast                    # optimised build with quick links
    cargo run --profile fast -- --scene mlp     # open on a particular scene
    cargo run --profile fast -- list            # scene ids
    cargo run --profile fast -- shot <scene> out.png [--size 1600x1000] [--frames 90]
                                 [--scale 2] [--no-panels] [--bare] [--primer]
                                 [--click x,y]... [--hover x,y]
                                 [--drag x0,y0,x1,y1]... [--scroll x,y,amount]
                                 [--trace N] [--realtime]

`shot` renders a scene without a window and saves a PNG (`--bare` gives the picture
alone; `--size 780x1560 --scale 2` is a phone); `--click`, `--hover`, `--drag`
and `--scroll` script the pointer, which is how interaction is tested. `--trace N` prints
the time spent in scene code and in rendering every N frames, which is how to check that
a long run does not slow down; add `--realtime` to let time budgets apply as they do in
the window. Rendering is
deterministic, so comparing the PNGs of every scene before and after a change shows
exactly what it affected. Without a GPU, select Mesa's software renderer (the
scene-contract test needs something to draw with too: where it finds nothing it fails
and says how to give it something; `EIDOS_ALLOW_NO_GPU=1` lets it pass there without
checking any scene):

    WGPU_BACKEND=gl __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \
        cargo run --profile fast -- shot surface out.png

`tools/webshot.mjs` drives a headless Chrome over the DevTools protocol to check the
browser build under both WebGPU and WebGL2 (console errors, screenshots, clicks, drags).
Chrome's shader compiler is stricter than the native one, so run it after shader changes.

Build output goes to cargo's target directory (`target/` here, unless `build.target-dir`
is set in a `.cargo/config.toml` above this directory, or `CARGO_TARGET_DIR` is).
`build-web.sh` expects `wasm-bindgen` (matching the crate version in `Cargo.lock`) and
`wasm-opt` under `$EIDOS_TOOLS`. `publish.sh` uploads only a finished release build of
the source exactly as committed (it refuses a quick build, a build that stopped
half-way, and one made before the last change) to the local copy of the site's branch,
`$EIDOS_PAGES`, and moves the tag `published` to that commit; `publish-source.sh`
uploads the committed source itself to the `main` branch of the same repository. A file
`local.env` beside the scripts (not in the repository) can set `EIDOS_TOOLS` and
`EIDOS_PAGES` for a machine.

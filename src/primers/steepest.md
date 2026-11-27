# The idea

Gradient descent steps against the gradient, often called the steepest way down. But steepest compares steps of one size, and what size means is a choice: a ruler, or norm. Each ruler has its own steepest step; several well-known optimisers are this rule under different rulers. You need the gradient, a bowl's Hessian $H$ (scene "hessian") and the singular value decomposition (scene "svd").

# The mathematics

## The lowest point of a shape

Near weights $w$ with gradient $g$, the loss is to first order a tilted plane:

$$ f(w + \Delta) \approx f(w) + g \cdot \Delta $$

The steepest step of size $\eta$ makes $g \cdot \Delta$ as low as it can:

$$ \Delta^* = \argmin_{\|\Delta\| \le \eta} \; g \cdot \Delta $$

The steps of size at most $\eta$ fill a shape round $w$. Slide a level line of the plane (at right angles to $g$) downhill until it last touches the shape: that point is $\Delta^*$.

- Length, $\sqrt{\Delta_1^2 + \Delta_2^2}$: a circle, touched along $-g$.
- The largest entry, $\max(|\Delta_1|, |\Delta_2|)$: a square, touched at a corner, $\Delta^* = -\eta \operatorname{sign} g$.
- The sum of sizes, $|\Delta_1| + |\Delta_2|$: a diamond, touched at a vertex; only the coordinate with the larger $|g_i|$ moves.
- Curvature, $\sqrt{\Delta^\top H \Delta}$: an ellipse shaped like the bowl's level lines, touched along $-H^{-1} g$, which is $-w$ on the bowl $f = \tfrac12 w^\top H w$ (where $g = Hw$): straight at the bottom.

If the level line lies along an edge, all of that edge is as low: for the square when an entry of $g$ is 0, for the diamond when $|g_1| = |g_2|$.

## How far down, and the second form

The lowest value is $-\eta \|g\|_*$; the dual size $\|g\|_*$ is the most $g \cdot t$ for a step $t$ of size 1: $|g|$ for length, $|g_1| + |g_2|$ for the largest entry, $\max(|g_1|, |g_2|)$ for the sum of sizes.

Example: $g = (2, -1)$, $\eta = 0.5$. The circle's step $(-0.447, 0.224)$ lowers the plane by $0.5 \times \sqrt{5} = 1.118$, the square's $(-0.5, 0.5)$ by $0.5 \times 3 = 1.5$, the diamond's $(-0.5, 0)$ by $0.5 \times 2 = 1$.

The first form's step has size $\eta$ however large $g$ is; the second's makes this least:

$$ g \cdot \Delta + \tfrac{L}{2} \|\Delta\|^2 $$

Its step points the same way, with size $\|g\|_* / L$. With $L$ the most $t^\top H t$ for $t$ of size 1, this sum is never below the bowl's true change, so the step never raises the loss. Under length it is $-g/L$, gradient descent; under curvature, $-H^{-1} g$, Newton's step, landing on the bottom at once.

## Turning, and matrices

Choosing the ruler is called preconditioning. Turn the bowl and the start together: gradient descent's path turns with them and Newton's still takes one step. The square, the diamond and an ellipse along the axes stay with the axes, so their paths change, unless it is a quarter turn, which takes each onto itself. A square turned by $45°$ is a diamond: those two trade places.

For a weight matrix $G = U S V^\top$, the root of its summed squared entries gives a step along $-G$; its largest entry gives $-\eta \operatorname{sign} G$; the spectral norm, the most it stretches a vector of length 1, gives $-\eta\, U V^\top$: every singular value set to 1. Adam with both running averages off and $\epsilon$ left out steps by $-\eta \operatorname{sign} g$; Muon with its momentum off, an exact $UV^\top$ and no factor for the matrix's shape steps by $-\eta\, UV^\top$. Both are first forms.

# In the picture

- White: $w$, the arrow $g$ (shortened), and the plane's level lines.
- Each coloured shape: the steps of size $\eta$ under one ruler, with its steepest step and the level it touches, $f - \eta \|g\|_*$.
- The paths chart counts each ruler's second-form steps to the bottom.
- The bars: singular values of a gradient matrix and its three steepest steps.

> Try first "turn the gradient": the circle's step turns smoothly, the square's and the diamond's jump.

see: hessian, svd, adam, muon

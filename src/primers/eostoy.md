# The idea

Gradient descent with step $\eta$ settles in a bowl only if the bowl's sharpest curvature is less than $2/\eta$ (scene "stepsize"). A real loss is not one bowl: its curvature changes from place to place. Here is a small loss where that happens, of two weights $a$ and $b$, and what gradient descent does above a minimum too sharp for its step: unless the step is far too long, it is moved along to flatter ground. You need the gradient, the Hessian and its eigenvalues.

# The mathematics

## A valley of minima

The first loss is $L = \tfrac12(ab - 1)^2$. Every point of the curve $ab = 1$ has $L = 0$: the minima form a valley floor. Write $g = ab - 1$; then $\nabla L = g\,(b, a)$, and the Hessian is

$$ H = \begin{bmatrix} b^2 & 2ab - 1 \\ 2ab - 1 & a^2 \end{bmatrix} $$

On the floor $2ab - 1 = 1$, so $\det H = a^2b^2 - 1 = 0$: one eigenvalue is 0 (along the floor nothing changes), and the other is the trace, the sharpness $S$. Since $(a^2 + b^2)^2 = (a^2 - b^2)^2 + 4a^2b^2$,

$$ S = a^2 + b^2 = \sqrt{(a^2 - b^2)^2 + 4} $$

It is 2 at $a = b = 1$ and grows without bound along each arm. Near a floor point the loss is nearly a bowl of curvature $S$ across the valley, so gradient descent can stay only where $S < 2/\eta$.

## Why a bounce moves it along

One step is $a \to a - \eta g b$ and $b \to b - \eta g a$. Square and subtract (the terms $-2\eta g ab$ cancel):

$$ (a - \eta g b)^2 - (b - \eta g a)^2 = (1 - \eta^2 g^2)(a^2 - b^2) $$

So gradient flow, with infinitely small steps, keeps $a^2 - b^2$ fixed (the change per step is of order $\eta^2$), and comes down to the floor point with the start's $a^2 - b^2$. A short step near the floor, where $g$ is small, does nearly the same. Where the floor is too sharp, every step throws the point further across, $g$ grows, and each step shrinks $a^2 - b^2$: flatter ground beneath. Once $S < 2/\eta$ there, the swings die away.

Example: from $(a, b) = (3, 0.35)$, $a^2 - b^2 = 8.8775$, and the floor beneath has $S = \sqrt{8.8775^2 + 4} = 9.1$, more than $2/\eta = 8$ for $\eta = 0.25$. The first step has $g = 0.05$ and multiplies $a^2 - b^2$ by $1 - 0.0625 \times 0.0025 = 0.99984$; at the top of the bounce the loss is 0.37, $g^2 = 0.75$, and the factor is 0.953. The run settles at $a^2 - b^2 = 6.41$, where $S = \sqrt{6.41^2 + 4} = 6.71$.

## How far it goes

Adding the squares instead gives $\lambda = a^2 + b^2$ the step $\lambda \to \lambda(1 + \eta^2 g^2) - 4\eta g\,ab$, a change of $\eta g^2 (\eta\lambda - 4) - 4\eta g$ when $ab = 1 + g$. The last term changes sign at every bounce and largely cancels over the run; so, roughly (this is no proof), bouncing lowers $\lambda$ while $\eta\lambda < 4$ and raises it beyond. The scene measures it: from the start above, every step tried from 0.01 to 2 settles up to 0.4395 and is thrown off beyond, and $4/9.1 = 0.4396$.

The second loss, $\tfrac14(1 - a^2b^2)^2$, has $g = ab(a^2b^2 - 1)$, the same factor on $a^2 - b^2$, and a floor twice as sharp. In the scene's runs its swing stays small, and it creeps along to a little under $2/\eta$, where the first loss goes well inside.

# In the picture

- Green: the floor, brighter where sharper. Yellow: where $S < 2/\eta$.
- Cyan: gradient descent from the start (drag it); its dots, where it is after an even and an odd number of steps. Dashed: gradient flow.
- The charts: the loss; the sharpness beneath the point (bold) and at it (faint), against $2/\eta$; where a run from here ends, for every step.

> Try first "a longer step": the loss rises for fifteen steps while the dots slide towards the middle. Then compare "the other loss".

see: stepsize, hessian, eos

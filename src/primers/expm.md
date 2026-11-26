# The idea

Instead of jumping in steps, let every point $x$ of the plane move continuously, with its velocity (which way, and how fast) equal to $Ax$. The whole plane then flows, and following the flow for a time $t$ is itself a linear map: a matrix, written $e^{At}$, the matrix exponential. Such flows model a spring with friction, two linked populations, and many systems near rest. You need eigenvalues and eigenvectors, determinants as areas, and $e^{\lambda t}$, the number that at every moment grows at $\lambda$ times its own size (and shrinks if $\lambda < 0$).

# The mathematics

## The rule and the series

$$ \frac{\mathrm{d}x}{\mathrm{d}t} = Ax, \qquad x(t) = e^{At} x_0 $$

Here $\mathrm{d}x/\mathrm{d}t$ is the velocity of $x$ and $x_0$ is where it starts. For a number $a$, the rule $\mathrm{d}x/\mathrm{d}t = ax$ is solved by $e^{at} x_0$, with $e^{at} = 1 + at + (at)^2/2! + \cdots$. The matrix exponential is the same series with the matrix $A$ in place of $a$:

$$ e^{At} = I + At + \frac{(At)^2}{2!} + \frac{(At)^3}{3!} + \cdots $$

Each term's rate of change in $t$ is $A$ times the term before it ($I$ does not change), so the whole sum changes at $A$ times itself, which is the rule; and at $t = 0$ it is $I$.

## Along an eigenline

If $Av = \lambda v$, a point on the line of $v$ (an eigenline) has its velocity along that line, $\lambda$ times its position. So it stays on the line and grows like a number:

$$ e^{At} v = e^{\lambda t} v $$

out if $\lambda > 0$, in if $\lambda < 0$. If $A$ has two eigenlines, split the start into its parts along them and let each grow by its own $e^{\lambda t}$. Areas are then multiplied by $e^{\lambda_1 t} e^{\lambda_2 t}$, and as $\lambda_1 + \lambda_2 = \tr A$,

$$ \det e^{At} = e^{t \tr A} $$

a rule that holds for every $A$.

Example, the scene's saddle:

$$ A = \begin{bmatrix} 1 & -1 \\ 0 & -1 \end{bmatrix} $$

It has $\lambda = 1$ along $(1, 0)$ and $\lambda = -1$ along $(1, 2)$ (check: $A$ sends $(1, 2)$ to $(-1, -2)$). So $(1, 0)$ flows to $e^t (1, 0)$, and $(0, 1) = -0.5 \cdot (1, 0) + 0.5 \cdot (1, 2)$ flows to $-0.5 e^t (1, 0) + 0.5 e^{-t} (1, 2)$. These are the columns of $e^{At}$. At $t = 1.5$, with $e^{1.5} = 4.48$ and $e^{-1.5} = 0.22$:

$$ e^{At} = \begin{bmatrix} 4.48 & -2.13 \\ 0 & 0.22 \end{bmatrix} $$

Its determinant is $e^{1.5} e^{-1.5} = 1$: since $\tr A = 0$, areas are kept.

## Complex eigenvalues, and the map of flows

If $\lambda = a \pm bi$, no line stays put and points go round $0$: at $t = 2\pi/b$ each is back on the ray it started on, $e^{at}$ times as far out. So the flow grows if $a > 0$ and shrinks if $a < 0$.

The eigenvalues are the roots of $\lambda^2 - (\tr A)\lambda + \det A$, so the trace and determinant decide the kind of flow:

- $\det A < 0$: one $\lambda$ of each sign, a saddle.
- $\det A > 0$: everything flows in (a sink) if $\tr A < 0$, out (a source) if $\tr A > 0$.
- Above the parabola $\det A = (\tr A)^2/4$ the eigenvalues are complex and the flow turns: a spiral, or, on $\tr A = 0$, a centre of closed loops.

# In the picture

- Faint streaks: the flow. White: the start $x_0$ (drag it), its path, and $e^{At} x_0$, with dots at $t = 1, 2, 3, \ldots$; the cyan arrow is the velocity $Ax$ there.
- Orange and purple dashed: the eigenlines, each with a unit arrow carried to $e^{\lambda t}$ times its length.
- Pink and green: the paths of $(1, 0)$ and $(0, 1)$, which end at the columns of $e^{At}$. Yellow: the unit square carried along, of area $\det e^{At}$.
- Red, when shown: the terms of the series, end to end. The map: $\tr A$ across, $\det A$ up; its dot is $A$.

> Try first "a saddle": $x_0$ is drawn in towards $0$, then swept out along the orange line, and the yellow square keeps its area.

see: eigen, powers, field

# The idea

Every square matrix splits, in exactly one way, into a symmetric part and a skew-symmetric part. Read the matrix as velocities, as in the scene on flows: the point $x$ moves with the velocity $Ax$. Then the symmetric part is a stretch along two lines at right angles, the skew-symmetric part a spin that turns every point at one rate, and every velocity is the sum of the two. You need the transpose, eigenvalues and eigenvectors, and flows.

# The mathematics

## The split

The transpose $A^\top$ is $A$ mirrored across its diagonal. Set

$$ S = \frac{A + A^\top}{2}, \qquad K = \frac{A - A^\top}{2}, \qquad A = S + K $$

Then $S^\top = S$ (symmetric) and $K^\top = -K$ (skew-symmetric). There is no other way: if $A = S' + K'$ with $S'$ symmetric and $K'$ skew, then $A^\top = S' - K'$, and adding and subtracting give $S' = S$, $K' = K$. Also $A^\top = S - K$: transposing keeps the stretch and reverses the spin.

## Stretch and spin

A symmetric $S$ has real eigenvalues $s_1 \ge s_2$, the stretch rates, with eigenvectors at right angles. On those axes a point moves straight out (rate above $0$) or in, at the rate times its distance from $0$.

In the plane a skew-symmetric $K$ has $0$ on its diagonal, so it is $wJ$, with $J$ the quarter turn and $w = (a_{21} - a_{12})/2$. The arrow $Kx$ is $x$ turned a right angle and multiplied by $w$: every point turns round $0$ at $w$ radians per unit of time, anticlockwise if $w > 0$.

Example, the shear:

$$ \begin{bmatrix} 0 & 1 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 0.5 \\ 0.5 & 0 \end{bmatrix} + \begin{bmatrix} 0 & 0.5 \\ -0.5 & 0 \end{bmatrix} $$

$S$ stretches along the 45° line at $s_1 = 0.5$ and squeezes across it at $s_2 = -0.5$; $K$ spins clockwise, $w = -0.5$. At $(1, 0)$ their velocities, $(0, 0.5)$ and $(0, -0.5)$, cancel: the $x$ axis stays put.

## What each part decides

- $Kx$ is at right angles to $x$, so $x \cdot Kx = 0$ and $x \cdot Ax = x \cdot Sx$: the quadratic form sees only $S$.
- $\tr K = 0$, so $\tr A = \tr S$. After a time $t$ the flow multiplies areas by $e^{t \tr A}$, all of it from $S$.
- As lists of four numbers $S$ and $K$ are at right angles, so the sums of the squares of their entries add up to that of $A$, and $S$ is the symmetric matrix nearest to $A$.
- $S$ has 3 numbers of its own and $K$ has 1: the 4 of $A$. In space it is 6 and 3, and $Kx = \omega \times x$ for a vector $\omega$.

## Stretch against spin

The spin turns every line through $0$ at the rate $w$. The stretch turns the line at the angle $\alpha$ from $s_1$'s axis at the rate $-((s_1 - s_2)/2) \sin 2\alpha$: towards that axis, at most $(s_1 - s_2)/2$. A line $A$ keeps, an eigenline, is one where the two cancel. Worked out,

$$ \lambda = \frac{s_1 + s_2}{2} \pm \sqrt{\left(\frac{s_1 - s_2}{2}\right)^2 - w^2} $$

so $A$'s eigenvalues are complex, and its flow goes round $0$, exactly when $|w| > (s_1 - s_2)/2$. The scene opens with $s_1 = 0.8$, $s_2 = -1.2$, $w = 1.2$: $\lambda = -0.2 \pm \sqrt{1 - 1.44} = -0.2 \pm 0.66i$.

The parts add as velocities, not as journeys: flowing by $A$ for every time $t$ is flowing by $S$ and then by $K$ only if $S$ and $K$ commute.

# In the picture

- The flows of $A$ (cyan), $S$ (green) and $K$ (pink).
- The white dot $x$ is the same point in all three; in $A$'s square its velocities make the parallelogram $Ax = Sx + Kx$.
- Orange and purple dashed: eigenlines. $S$'s are at right angles; $K$ has none, unless $K = 0$.
- Yellow: a small circle carried for a time $t$. $S$ makes it an ellipse along its axes; $K$ only turns it.
- The chart, on a wide screen: how fast each flow turns each line; $A$'s curve meets $0$ on its eigenlines.

> Try first "stretch against spin": as $w$ grows past $(s_1 - s_2)/2$, $A$'s eigenlines close up and vanish, and the flow begins to turn.

see: expm, quadform, special, expsum

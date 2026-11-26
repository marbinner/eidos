# The idea

A vector field attaches an arrow to every point of the plane. Here the arrow is a velocity: it says which way, and how fast, a particle at that point moves. Following the arrows from a starting point traces the particle's path through time, and finding such paths is what solving a differential equation means. Fields like these describe pendulums, populations, fluids, and gradient descent. You need coordinates in the plane and the derivative as a rate of change.

# The mathematics

## A velocity at every point

The two formulas of the scene are

$$ \frac{\mathrm{d}x}{\mathrm{d}t} = p(x, y), \qquad \frac{\mathrm{d}y}{\mathrm{d}t} = q(x, y) $$

where $t$ is time and $p$, $q$ are our names for the two right-hand sides: a particle at $(x, y)$ moves across at the rate $p$ and up at the rate $q$, with speed $\sqrt{p^2 + q^2}$. To follow it, move it by $(p, q)\,\Delta t$ over a short time $\Delta t$, then read the arrow again where it now is; shorter steps follow the path more closely. (The scene uses a more careful step, which reads the arrow half a step ahead.) For a smooth field that does not change with time, exactly one path runs through each point, so paths never cross.

## Fixed points

Where $p = 0$ and $q = 0$ a particle stays put: a fixed point. Particles nearby may be drawn in (a sink), pushed out (a source), sent round it, or drawn in along one line and pushed out along another (a saddle).

The pendulum the scene opens with has $p = y$ and $q = -\sin x - 0.15\,y$, where $x$ is the angle and $y$ how fast it changes. Both are 0 where $y = 0$ and $\sin x = 0$: at $x = 0, \pm\pi, \pm 2\pi, \ldots$ At even multiples of $\pi$ the pendulum hangs at rest, and the friction term $-0.15\,y$ makes particles spiral in. At odd multiples it is balanced upside down: saddles.

## Divergence and curl

Two numbers tell what the flow does close to a point. With $\partial p/\partial x$ the rate at which $p$ changes as $x$ alone changes, and so on,

$$ \text{divergence} = \frac{\partial p}{\partial x} + \frac{\partial q}{\partial y}, \qquad \text{curl} = \frac{\partial q}{\partial x} - \frac{\partial p}{\partial y} $$

If $p$ grows with $x$, particles on the right of a small patch move right faster than those on its left, and the patch widens: positive divergence is a flow spreading out, negative a flow gathering in. If $q$ grows with $x$, the right side of a patch moves up faster than its left; if $p$ falls as $y$ grows, its top moves left: both turn it anticlockwise, which is positive curl. A tiny paddle wheel placed there turns at half the curl.

Example: the vortex, $p = -y$, $q = x$. At $(1, 0)$ the velocity is $(0, 1)$, straight up; at $(0, 1)$ it is $(-1, 0)$, to the left. Particles go round anticlockwise, each with speed equal to its distance from the centre, so all take the same time, $2\pi$, for a turn. The divergence is $0 + 0 = 0$ and the curl $1 - (-1) = 2$, everywhere: a paddle wheel turns at 1, as fast as the whole flow. For the pendulum the divergence is $-0.15$ everywhere: friction slowly squeezes every patch.

# In the picture

- Each streak is a particle, brightest at its head: pale and bright when fast, dark when slow.
- The background shows the speed $\sqrt{p^2 + q^2}$, dark where slow; or the divergence or the curl, red above 0 and blue below.
- The white line is the path through a point you clicked. For a field that changes with time, it is the path ahead of a particle released there, and the white dot travels it.
- The two boxes in the panel are $p$ and $q$; the buttons below them load other fields.

> Try first "a vortex spins everywhere", then "a source and a sink".

see: surface, descent, expm, attractor

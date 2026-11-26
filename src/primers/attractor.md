# The idea

Some rules give a velocity at every point of space, and a particle put anywhere follows it for ever. The Lorenz system is three short rules of this kind. At the scene's starting values its paths, from almost any start, never settle down and never repeat: they are drawn onto a thin shape with two wings, the attractor, and wander over it, while two neighbours drift apart until their paths have nothing in common. That is chaos: exact rules whose future cannot in practice be foretold. You need to know a derivative as a rate of change.

# The mathematics

## A velocity at every point

A particle is at $(x, y, z)$. Write $\dot x$ for the rate at which $x$ changes with time, and $\dot y$, $\dot z$ likewise. The Lorenz system gives these rates at every point:

$$ \dot x = \sigma (y - x), \quad \dot y = x(\rho - z) - y, \quad \dot z = xy - \beta z $$

The numbers $\sigma$, $\rho$ and $\beta$ stay fixed while it runs; the scene starts with $\sigma = 10$, $\rho = 28$ and $\beta = 2.667$, about $8/3$. Where a particle is now fixes its whole future. The scene moves each particle on in small steps of time, each following a careful average of the velocities near it.

## Resting points

Where all three rates are zero, a particle stays put. With $\sigma$ not 0, $\dot x = 0$ needs $y = x$; then $\dot y = 0$ needs $x = 0$ or $z = \rho - 1$; and $\dot z = 0$ needs $x^2 = \beta z$. So when $\rho > 1$, besides the origin there are two resting points,

$$ x = y = \pm\sqrt{\beta(\rho - 1)}, \qquad z = \rho - 1 $$

Example: with $\rho = 10$ and $\beta = 8/3$ they are $(\sqrt{24}, \sqrt{24}, 9)$ and $(-\sqrt{24}, -\sqrt{24}, 9)$, with $\sqrt{24} \approx 4.90$. Check the first: $\dot x = 10(\sqrt{24} - \sqrt{24}) = 0$, $\dot y = \sqrt{24}\,(10 - 9) - \sqrt{24} = 0$ and $\dot z = 24 - \tfrac{8}{3} \cdot 9 = 0$. At $\rho = 10$, particles near these points spiral in to them. At $\rho = 28$ the points are still there, at $z = 27$, but particles near them spiral outwards: they are the centres of the two wings, circled for ever and never reached.

## Squeezed and stretched

A small cloud of particles is at once squeezed and stretched. Squeezed: $\dot x$ falls by $\sigma$ for each unit that $x$ grows, $\dot y$ by $1$ for each unit of $y$, and $\dot z$ by $\beta$ for each unit of $z$. Added up, these give the rate at which the cloud's volume shrinks in proportion to itself: $\sigma + 1 + \beta \approx 13.7$. Each unit of time multiplies the volume by $e^{-13.7}$, about a millionth, so the paths crowd onto a shape with no volume at all.

Stretched: on the attractor, a small gap $d$ between two particles grows on average like

$$ d(t) \approx d(0)\, e^{\lambda t} $$

while it is still small. No formula gives $\lambda$; measured at the starting values, it is about $0.9$, so a gap doubles about every $0.77$ units of time. Twenty doublings make a gap about a million times larger, so starting two particles a million times closer only buys twenty doublings more: their paths still part. The rate $\lambda$ is the system's largest Lyapunov exponent, and a positive one is the mark of chaos.

# In the picture

- Each glowing thread is the recent path of one particle through $(x, y, z)$, coloured at first by its speed, the length of $(\dot x, \dot y, \dot z)$.
- The two wings circle the two resting points, and a particle changes wings at irregular moments.
- The sliders set $\sigma$, $\rho$ and $\beta$. At speed 1 the scene runs 0.72 units of the system's time a second (when the computer keeps up), so a small gap doubles about once a second.

> Try first "two starts a hair apart": two particles trace a single path for about ten seconds, then part and go round the wings each in its own way.

see: field, pendulum, lyapunov

# The idea

A stretched membrane, like a drum skin, carries waves: push it somewhere and the disturbance spreads outwards at a steady speed. One equation, the wave equation, describes it while the motions are small. Because such waves add up, it also explains interference: crest on crest makes a bigger crest, and crest on trough cancels. You need a derivative as a rate of change, and the sine of an angle.

# The mathematics

## The wave equation

Let $u$ be the height of the membrane above its resting level at the point $(x, y)$ and the time $t$. The rate at which $u$ changes with time, the point held fixed, is written $\partial u / \partial t$: the curly $\partial$ says that only $t$ moves. That rate's own rate of change, $\partial^2 u / \partial t^2$, is the point's up-and-down acceleration. Likewise $\partial^2 u / \partial x^2$ is how fast the slope along $x$ changes: positive where the surface curves up like a bowl, negative where it humps like a hill. The wave equation is

$$ \frac{\partial^2 u}{\partial t^2} = c^2 \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) $$

The bracket is positive where a point lies below the average level of the points round it, and negative where it lies above. So the equation says: each point accelerates towards the level of its neighbours, the harder the further it is from it. The scene does this on a grid of 192 by 192 points: at each step, a point's velocity changes in proportion to

$$ u_\text{left} + u_\text{right} + u_\text{up} + u_\text{down} - 4u $$

which is four times the gap between its neighbours' average height and its own.

## Speed and wavelength

Take any shape $g$ and slide it along $x$, the same for every $y$: $u = g(x - ct)$. Both sides of the equation are then $c^2 g''(x - ct)$, so this is a solution: the shape moves at speed $c$ without changing. A source that swings $f$ times a second sends out $f$ crests a second, each travelling at $c$, so neighbouring crests are a wavelength

$$ \lambda = c / f $$

apart. Twice the frequency gives half the wavelength.

## Adding up

If $u_1$ and $u_2$ both satisfy the wave equation, so does $u_1 + u_2$: each side of the equation for the sum is the sum of the two sides. So waves pass through each other unchanged, and where they overlap the membrane is their sum. Damping, which takes a share of every point's velocity each step, leaves this true.

Take two sources swinging in step. Where the distances to them differ by a whole number of wavelengths, crests arrive together and add up; where they differ by half a wavelength, or one and a half, crest meets trough and the water hardly moves. Far from sources a distance $d$ apart, in the direction at the angle $\theta$ from the middle line, the distances differ by about $d \sin\theta$. So, far out, the still spokes are where

$$ d \sin\theta = \tfrac{1}{2}\lambda,\ \tfrac{3}{2}\lambda,\ \ldots $$

Example: the scene's sources are 36 grid points apart, and at wave speed 1 its waves travel 86.4 grid points a second. At frequency 6, $\lambda = 86.4 / 6 = 14.4$ and $d = 2.5\lambda$, so $\sin\theta = 0.5 / 2.5 = 0.2$ or $1.5 / 2.5 = 0.6$, and $\theta \approx 11.5^\circ$ or $36.9^\circ$: the spokes of "a quicker rhythm" (and $2.5 / 2.5 = 1$, along the line through the sources). At frequency 3, $d = 1.25\lambda$, only the half wavelength fits, and there is one spoke each side.

# In the picture

- The surface is the height $u$, exaggerated. In the first colour map red is above the resting level, blue below, and black level with it.
- The two patches marked "source" are pushed up and down $f$ times a second, both in step.
- "Wave speed" sets $c$ and "frequency" sets $f$. The edges soak waves up unless you turn that off.

> Try first "two sources in step": find the two still spokes either side of the middle line, where crest always meets trough.

see: surface, fourier, pendulum

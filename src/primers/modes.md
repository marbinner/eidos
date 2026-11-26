# The idea

Masses joined by springs in a row between two walls can move in tangled ways: pull one aside and soon they all swing. A few motions are simple: every mass swings in step, at one frequency, keeping one shape. These normal modes are the eigenvectors of a symmetric matrix, and every motion is a sum of them. You need eigenvectors, and dot products with perpendicular unit vectors.

# The mathematics

## The springs as a matrix

Let $x_i$ be how far mass $i$ is from where it rests. Each spring pulls with its stiffness times its stretch; together they push mass $i$ with a force $-(Kx)_i$. With two masses and three springs of stiffness $k$,

$$ K = \begin{bmatrix} 2k & -k \\ -k & 2k \end{bmatrix} $$

Moving mass 1 alone by $x_1$ stretches the left spring and squeezes the middle one: both drive it back, $2k x_1$ in all, and the middle spring pushes mass 2 by $k x_1$. $K$ is symmetric because a spring pulls the masses at its two ends equally hard. With every mass $1$, Newton's law says $x'' = -Kx$.

## The pure motions are eigenvectors

Try $x = v \cos(\omega t)$ for a fixed vector $v$. Then $x'' = -\omega^2 v \cos(\omega t)$, so it is a motion of the masses exactly when

$$ Kv = \omega^2 v $$

The shape $v$ is an eigenvector of $K$, and $\omega$ is the square root of its eigenvalue: a swing there and back takes $2\pi/\omega$ seconds.

Example, with $k = 9$: $K(1, 1) = (9, 9)$, so the masses swing together at $\omega = 3$, every 2.09 seconds; $K(1, -1) = (27, -27)$, so against each other they swing at $\sqrt{27} = 5.20$, every 1.21 seconds.

## Every motion is a sum of them

The eigenvectors of a symmetric matrix can be chosen perpendicular; take them of length 1. Let go from rest at $x_0$, the masses move as

$$ x(t) = \sum_m (v_m \cdot x_0) \cos(\omega_m t)\, v_m $$

Each term obeys the law, so their sum does; at $t = 0$ the terms add up to $x_0$, whose coordinates along the perpendicular $v_m$ are these dot products; and every velocity starts at $0$. Only one motion starts that way, so this is it: each mode keeps its amplitude $v_m \cdot x_0$ for ever.

Example: from $x_0 = (0.4, 0)$, with $v_1 = (1, 1)/\sqrt 2$ and $v_2 = (1, -1)/\sqrt 2$, both amplitudes are $0.4/\sqrt 2 = 0.28$. A mode's energy at amplitude $a$ is $\tfrac12 \omega^2 a^2$, so mode 2, with $\omega_2^2 = 3 \omega_1^2$, holds 75 % of it.

## Beats, sines and friction

- A weak middle spring $\kappa$ makes $\omega_2^2 = k + 2\kappa$ close to $\omega_1^2 = k$. Pull the left mass, and its swing passes to the right one and back every $2\pi/(\omega_2 - \omega_1)$ seconds: 13.54 for $\kappa = 1.5$.
- With $n$ equal masses and $n + 1$ equal springs, mode $j$ goes as $\sin(ij\pi/(n + 1))$ at mass $i$: a sine curve that is $0$ at both walls.
- A drag of $\gamma$ times each mass's velocity, the same for every mass, leaves the modes apart: while $\gamma < 2\omega$, each swings at $\sqrt{\omega^2 - \gamma^2/4}$ and dies away as $e^{-\gamma t/2}$.

As a flow, the state $(x, x')$ moves as

$$ \begin{bmatrix} x \\ x' \end{bmatrix}' = \begin{bmatrix} 0 & I \\ -K & -\gamma I \end{bmatrix} \begin{bmatrix} x \\ x' \end{bmatrix} $$

In the coordinates of the modes this matrix splits into one $2 \times 2$ block for each mode, acting on its position $q$ and velocity $q'$ alone, with eigenvalues $-\gamma/2 \pm i\sqrt{\omega^2 - \gamma^2/4}$: its exponential turns $q$ and $q'$ round together, spiralling in with friction.

# In the picture

- White: the masses (drag one) and below them each mass's position over time.
- Each row is a mode: its $\omega$, its shape $v$ as stems (up for a swing to the right), and small masses moving with its part of the motion alone.
- The circle's pointer is the mode's amplitude, turning at $\omega$; the bar is its share of the energy.

> Try first "pluck one mass": both pointers stay the same length while each turns at its own rate.

see: eigen, spectral, expm, dft

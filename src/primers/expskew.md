# The idea

In the plane, following the velocities $Jx$ of the quarter turn $J$ for a time $t$ turns the plane by the angle $t$. In space a skew-symmetric matrix $K$, one with $K^\top = -K$, plays that part: it is a spin, and $e^{tK}$ is a rotation. The scene shows why, how the series for $e^{tK}$ folds into three terms (Rodrigues' formula), and how to read the spin back from a rotation. You need the cross product, the matrix exponential and rotations as matrices.

# The mathematics

## A skew-symmetric matrix is a spin

A $3 \times 3$ matrix with $K^\top = -K$ has zeros down its diagonal and is made of three numbers:

$$ K = \begin{bmatrix} 0 & -c & b \\ c & 0 & -a \\ -b & a & 0 \end{bmatrix}, \qquad Kx = w \times x, \quad w = (a, b, c) $$

So $Kx$ is at right angles to $w$ and to $x$, and $|w|$ times as long as the distance of $x$ from the line of $w$. If every point moves with the velocity $Kx$, each keeps its height along $w$ and its distance from $0$: it goes round a circle about that line, at $|w|$ radians per unit of time, by the right-hand rule. That is a spin about the axis $n = w/|w|$.

## Why $e^{tK}$ is a rotation

Following the velocities for a time $t$ is $e^{tK} = I + tK + (tK)^2/2! + \cdots$. Transposing each term turns $K$ into $-K$, so

$$ (e^{tK})^\top = e^{-tK} = (e^{tK})^{-1} $$

So $R = e^{tK}$ has $R^\top R = I$, and $\det R = e^{t \tr K} = e^0 = 1$. It is a rotation, by the angle $\theta = t|w|$ about $n$.

## Rodrigues' formula

Write $K = |w|\,N$, with $N$ the spin of rate 1 about $n$. Then $N^2 x = n \times (n \times x)$ is minus the part of $x$ across the axis, and one more step gives $N^3 = -N$. The powers of $N$ run $N, N^2, -N, -N^2, \ldots$, and the series sorts into two known ones:

$$ e^{\theta N} = I + \left(\theta - \frac{\theta^3}{3!} + \cdots\right) N + \left(\frac{\theta^2}{2!} - \frac{\theta^4}{4!} + \cdots\right) N^2 = I + \sin\theta\, N + (1 - \cos\theta)\, N^2 $$

The part of $x$ along the axis stays; the part across it turns, $\cos\theta$ of itself plus $\sin\theta$ of $n \times x$. In the plane, with $J^2 = -I$, the same sorting gives $e^{tJ} = \cos t\, I + \sin t\, J$.

Example: $w$ along $(1, 1, 1)$ with $|w| = 2\pi/3$, a turn of 120°: $\sin\theta = 0.866$ and $1 - \cos\theta = 1.5$. For $x = (1, 0, 0)$, $Nx = (0, 0.577, -0.577)$ and $N^2 x = (-0.667, 0.333, 0.333)$, so

$$ e^K x = (1, 0, 0) + 0.866\,(0, 0.577, -0.577) + 1.5\,(-0.667, 0.333, 0.333) = (0, 1, 0) $$

Likewise $y$ goes to $z$ and $z$ to $x$.

## Back from a rotation to its spin

The diagonals of $I$, $N$ and $N^2$ add up to $3$, $0$ and $-2$, and $I$ and $N^2$ are symmetric while $N$ is not. So

$$ \tr R = 1 + 2\cos\theta, \qquad R - R^\top = 2\sin\theta\, N $$

give the angle and the axis. The second fails at a half turn, where $\sin\theta = 0$; there $R + I = 2nn^\top$ gives the axis but not its sign: $\pi n$ and $-\pi n$ give the same half turn. Below a half turn, the spin read back is the only one shorter than $\pi$; adding $2\pi$ to its angle gives longer ones.

## Spins add, turns do not

Velocities add, $w_1 \times x + w_2 \times x = (w_1 + w_2) \times x$, so $K_1 + K_2$ is one spin. But one turn and then the other, $e^{K_2} e^{K_1}$, is in general not $e^{K_1 + K_2}$; they agree when both axes are the same line.

# In the picture

- Orange: the spin $w$ (drag its tip); its dashed line is the axis.
- Yellow: the cube turned by $R = e^{tK}$; pink, green, blue: its columns $Re_1, Re_2, Re_3$.
- Cyan: the velocities $w \times x$ on a sphere, a sixth of their length.
- White: the point $p$ going round its circle to $e^{tK} p$. Red: the series' terms. Cyan and purple steps: $\sin\theta\, Np$ and $(1 - \cos\theta)\, N^2 p$.

> Try first "one spin, followed to the end": every point circles the orange axis, and at 120° the cube lands on itself with its arrows moved round.

see: vectors, expm, rotations, expsum

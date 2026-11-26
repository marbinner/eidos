# The idea

Now the input is a random point $X$ in the plane, whose two coordinates are independent standard normal numbers. A function that looks at $X$ only along one direction, like a neuron with one weight vector, is constant along every line at right angles to that direction. Take two such functions, along directions $u$ and $v$: the average of their product depends only on the angle between the directions, and for Hermite polynomials the answer is $(\cos\theta)^n$ or zero. You need the dot product, the normalised Hermite polynomials $h_n$ and the previous scene's noise operator.

# The mathematics

## A coordinate along a direction

Let $u = (u_1, u_2)$ have length 1. The coordinate of $X$ along it is the dot product $u \cdot X = u_1 X_1 + u_2 X_2$. It is a standard normal number: a sum of independent normals is normal, its mean is 0, and its variance is $u_1^2 + u_2^2 = 1$. For two such directions $u$ and $v$, at an angle $\theta$, since $\mathbb{E}[X_1 X_2] = 0$ and $\mathbb{E}[X_1^2] = \mathbb{E}[X_2^2] = 1$,

$$ \mathbb{E}[(u \cdot X)(v \cdot X)] = u_1 v_1 + u_2 v_2 = u \cdot v = \cos\theta $$

So $A = u \cdot X$ and $B = v \cdot X$ are two standard normals with correlation $\rho = \cos\theta$: exactly the pair $X$, $Y$ of the noise scene, under different names.

## The overlap

$$ \mathbb{E}\left[h_m(u \cdot X)\, h_n(v \cdot X)\right] = \begin{cases} (u \cdot v)^n & \text{if } m = n \\ 0 & \text{if } m \ne n \end{cases} $$

Why: average first over $B$ with $A$ held fixed. By the noise scene that turns $h_n(B)$ into $\rho^n h_n(A)$. What is left is $\rho^n\, \mathbb{E}[h_m(A)\, h_n(A)]$, which is $\rho^n$ when $m = n$ and 0 otherwise, by orthogonality.

The average is an integral over the plane, with $\varphi(x) = e^{-|x|^2/2} / (2\pi)$ the density of $X$:

$$ \mathbb{E}\left[h_m(u \cdot X)\, h_n(v \cdot X)\right] = \int h_m(u \cdot x)\, h_n(v \cdot x)\, \varphi(x)\, \mathrm{d}x $$

That is the signed volume under the surface the scene draws: the volume above zero minus the volume below.

## Worked examples

Degree 1 at $60^\circ$: $h_1(t) = t$, so the volume is $\mathbb{E}[(u \cdot X)(v \cdot X)] = \cos 60^\circ = 0.5$, the correlation itself.

Degree 2 at $60^\circ$: use $h_2(t) = (t^2 - 1)/\sqrt{2}$ and $\mathbb{E}[A^2 B^2] = 1 + 2\cos^2\theta$. Then

$$ \mathbb{E}[h_2(A)\, h_2(B)] = \frac{\mathbb{E}[A^2 B^2] - \mathbb{E}[A^2] - \mathbb{E}[B^2] + 1}{2} = \frac{1 + 2\cos^2\theta - 1}{2} = \cos^2\theta $$

which at $60^\circ$ is $0.25 = 0.5^2$. At right angles $\cos\theta = 0$, so every degree $n \ge 1$ gives volume 0.

Plain powers do not separate so cleanly: $\mathbb{E}[(u \cdot X)(v \cdot X)^3] = 3\cos\theta$, which is not zero although the degrees differ.

# In the picture

- The surface is $h_m(u \cdot x)\, h_n(v \cdot x)\, \varphi(x)$: warm where it is above zero, blue below. It is always drawn at the same height, so read the volume from the numbers.
- Light lines are contours; the straight ones are where the surface is zero, where $u \cdot x$ is a zero of $h_m$ or $v \cdot x$ a zero of $h_n$.
- Pink is the direction $u$ and green is $v$, each with a dashed line along it; the yellow arc is the angle $\theta$. Drag the arrow tips.
- The line at the top gives the computed volume and, when the degrees are equal, $(\cos\theta)^n$.

> Try first "at right angles", then "two different degrees" and drag the arrows round.

see: vectors, gauss2, hermite_noise

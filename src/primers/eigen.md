# The idea

A matrix $A$ moves the arrows of the plane, and most arrows come out turned as well as stretched. A few lines through $0$ may be kept: an arrow $x$ on one of them comes out as a multiple of itself, $Ax = \lambda x$. These arrows are the eigenvectors, and the multiples $\lambda$ are the eigenvalues. Along them a matrix is easiest to understand; powers, flows and quadratic forms rest on them. You need matrices as maps, and determinants as areas.

# The mathematics

## Arrows that are not turned

An eigenvector of $A$ is an arrow $x$, not $0$, with

$$ Ax = \lambda x $$

for some number $\lambda$, its eigenvalue. If $\lambda < 0$, $Ax$ points the other way; if $\lambda = 0$, $A$ crushes $x$ to $0$. Every arrow $cx$ on its line ($c \ne 0$) is one too, as $A(cx) = \lambda(cx)$.

## An equation for $\lambda$

The identity $I$ is the matrix that changes nothing, so $\lambda I$ stretches every arrow by $\lambda$, and $Ax = \lambda x$ says

$$ (A - \lambda I)x = 0 $$

So at an eigenvalue, $A - \lambda I$ crushes an arrow that is not $0$, and its whole line, to $0$. That flattens the plane, so its determinant, the area it gives the unit square, is 0. The other way round, a determinant of 0 means the plane is flattened, so some line is crushed. The eigenvalues are therefore where $\det(A - \lambda I) = 0$. Written out:

$$ A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \qquad \det(A - \lambda I) = (a - \lambda)(d - \lambda) - bc = \lambda^2 - (a + d)\lambda + (ad - bc) $$

This is the characteristic polynomial, a parabola in $\lambda$. If its roots are $\lambda_1$ and $\lambda_2$, it is also $(\lambda - \lambda_1)(\lambda - \lambda_2)$, and matching the two forms gives $\lambda_1 + \lambda_2 = a + d$, the trace $\tr A$ (the sum of the diagonal), and $\lambda_1 \lambda_2 = ad - bc = \det A$.

Example: the scene's first matrix has $a = 1$, $b = 2$, $c = 1$, $d = 0$, so

$$ \det(A - \lambda I) = (1 - \lambda)(0 - \lambda) - 2 \cdot 1 = \lambda^2 - \lambda - 2 = (\lambda - 2)(\lambda + 1) $$

and $\lambda_1 = 2$, $\lambda_2 = -1$. Check: $A$ sends $(2, 1)$ to $(2 + 2,\ 2 + 0) = (4, 2)$, twice itself, and $(1, -1)$ to $(-1, 1)$, itself reversed. The sum $2 - 1 = 1$ is $\tr A$; the product $-2$ is $\det A$.

## When there are not two real roots

- If the parabola stays above 0, no real $\lambda$ works and no line is kept. Its roots are complex numbers $p \pm qi$, with $i^2 = -1$. A turn by 60° is like this: $\lambda^2 - \lambda + 1$, with roots $0.5 \pm 0.87i$.
- If it only touches 0, the root is double. The shear keeps one line and the identity $I$ keeps every line, yet both have $(\lambda - 1)^2$: a double root does not say how many lines are kept.
- If $A$ is symmetric ($b = c$), the roots are always real, because the discriminant $(a + d)^2 - 4(ad - b^2) = (a - d)^2 + 4b^2$ is never below 0. When the two roots differ, their lines cross at a right angle.

# In the picture

- The white arrow $x$ goes round the unit circle; the cyan arrow $Ax$ traces an ellipse. The green arc and the chart below show the turn from $x$ to $Ax$.
- The dashed lines are the kept lines, where the turn is 0° or 180°: orange for the larger eigenvalue $\lambda_1$, purple for $\lambda_2$.
- The small plane shows $A - \lambda I$ for the $\lambda$ you choose: the blue grid and the yellow unit square after it, of area $\det(A - \lambda I)$ (pink below 0). Its orange arrow is where the unit arrow $v_1$ (the orange ring) goes: $(\lambda_1 - \lambda)v_1$.
- The graph is the parabola $\det(A - \lambda I)$; its roots, orange and purple, are $\lambda_1$ and $\lambda_2$.

> Try first "sweep $x$ round": four times a turn $x$ pauses on a dashed line, where $Ax$ lines up with it. Then "slide $\lambda$ to a root" and watch the plane fold flat.

see: linmap2, det, powers, quadform

# The idea

Three independent vectors $a_1, a_2, a_3$ (none a combination of the others) span all of space, but they may meet at any angles. Gram–Schmidt replaces them, one at a time, by three vectors $q_1, q_2, q_3$ at right angles to each other and of length 1 that span the same space: an orthonormal frame. In such a frame a vector's coordinates are its dot products with the $q$'s, which makes them easy to work with. The record of the work is the factorisation $A = QR$. You need the dot product and its shadow, and the determinant.

# The mathematics

## Taking a shadow away

Let $q$ have length 1. The shadow of $a$ on $q$ is $(q \cdot a)\, q$, the arrow from $0$ to the point of $q$'s line nearest to $a$'s tip. What is left is at right angles to $q$:

$$ q \cdot \big(a - (q \cdot a)\, q\big) = q \cdot a - (q \cdot a)(q \cdot q) = 0 $$

since $q \cdot q = 1$. That one fact is the whole method.

## The steps

Keep $a_1$'s direction and scale it to length 1. From $a_2$ take away its shadow on $q_1$ and scale what is left. From $a_3$ take away its shadows on $q_1$ and on $q_2$ and scale what is left:

$$ r_{11} = \lVert a_1 \rVert, \quad q_1 = a_1 / r_{11} \\ r_{12} = q_1 \cdot a_2, \quad q_2 = (a_2 - r_{12} q_1) / r_{22} \\ r_{13} = q_1 \cdot a_3, \quad r_{23} = q_2 \cdot a_3, \quad q_3 = (a_3 - r_{13} q_1 - r_{23} q_2) / r_{33} $$

where $r_{22}$ and $r_{33}$ are the lengths of what is left. What is left of $a_3$ is at right angles to both $q_1$ and $q_2$: taking away the shadow on $q_2$ does not change the dot product with $q_1$, because $q_1$ and $q_2$ are at right angles.

If nothing is left ($r_{22} = 0$ or $r_{33} = 0$), that vector lies in the span of the ones before it: it adds no new direction and there is no new $q$. The number of $q$'s made is the rank. When little is left (a small $r_{22}$, say), $q_2$ is that little scaled up by $1/r_{22}$, and a small change in $a_2$ can turn $q_2$ a long way.

## Put back together: $A = QR$

Read the steps backwards: $a_1 = r_{11} q_1$, $a_2 = r_{12} q_1 + r_{22} q_2$, and $a_3 = r_{13} q_1 + r_{23} q_2 + r_{33} q_3$. With the $a$'s as the columns of $A$ and the $q$'s as the columns of $Q$,

$$ A = QR, \qquad R = \begin{bmatrix} r_{11} & r_{12} & r_{13} \\ 0 & r_{22} & r_{23} \\ 0 & 0 & r_{33} \end{bmatrix} $$

$R$ is zero below its diagonal because each $a$ needs only the $q$'s made so far. Its diagonal gives the volume of the slanted box the $a$'s span:

$$ \lvert \det A \rvert = r_{11}\, r_{22}\, r_{33} $$

the length of $a_1$, times the height of $a_2$ above the line of $a_1$, times the height of $a_3$ above the floor that $a_1$ and $a_2$ span.

## An example

Take $a_1 = (3, 4, 0)$, $a_2 = (2, 1, 0)$, $a_3 = (1, 2, 3)$.

1. $r_{11} = 5$ and $q_1 = (0.6, 0.8, 0)$.
2. $r_{12} = 1.2 + 0.8 = 2$; what is left of $a_2$ is $(2, 1, 0) - 2 q_1 = (0.8, -0.6, 0)$, of length $r_{22} = 1$, so $q_2 = (0.8, -0.6, 0)$.
3. $r_{13} = 0.6 + 1.6 = 2.2$ and $r_{23} = 0.8 - 1.2 = -0.4$; what is left of $a_3$ is $a_3 - 2.2 q_1 + 0.4 q_2 = (0, 0, 3)$, so $r_{33} = 3$ and $q_3 = (0, 0, 1)$.

The box has volume $5 \times 1 \times 3 = 15$, which is $\lvert \det A \rvert$.

# In the picture

- The pale arrows are $a_1, a_2, a_3$; drag their tips.
- The bright arrows are $q_1$ (pink), $q_2$ (green) and $q_3$ (purple).
- Dashed lines in a $q$'s colour are shadows on that $q$; yellow is what is left once they are taken away.
- The blue sheet is the plane of $q_1$ and $q_2$; the grey box, when shown, is the one the $a$'s span.
- $R$ over the picture fills in step by step.

> Try first "nearly in line": $a_2$ is so close to $a_1$ that little is left of it, and a small nudge swings $q_2$ round.

see: vectors, det, lsq, svd

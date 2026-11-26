# The idea

The determinant of a $2 \times 2$ matrix is one number: the area of the parallelogram its two columns span, with a sign. For a $3 \times 3$ matrix it is the volume of the box its three columns span. It is the factor by which the matrix's map scales areas (volumes, in space): negative if the map flips the plane over, $0$ if it squashes it flat. This scene shows two rules the area obeys, sliding a side changes nothing and swapping two sides flips the sign, and how they give $ad - bc$. You need the linear maps primer, and for space the cross product of the first primer.

# The mathematics

## Signed area

Write the $2 \times 2$ matrix with its two columns, $a_1$ and $a_2$:

$$ A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \qquad a_1 = (a, c), \quad a_2 = (b, d) $$

$\det A$ is the area of the parallelogram with sides $a_1$ and $a_2$, counted positive if the turn from $a_1$ to $a_2$ goes anticlockwise (as from the $x$ axis to the $y$ axis) and negative if it goes clockwise. Swapping the columns reverses that turn, so it flips the sign.

## Sliding a side

The area is base times height. Replace $a_2$ by $a_2 + m\,a_1$, for any number $m$: its tip slides along a line parallel to $a_1$. The base $a_1$ stays, and so does the height of the tip above it, so the area does not change:

$$ \det(a_1,\ a_2 + m\,a_1) = \det(a_1,\ a_2) $$

Sliding $a_1$ along $a_2$ changes nothing either.

## Sliding it square

Slides can turn the parallelogram into a rectangle along the axes. If $a \ne 0$, slide $a_2$ by $m = -b/a$: it becomes $(0,\ d - bc/a)$, upright. Then slide $a_1$ along that upright arrow until it lies flat, at $(a, 0)$. (If the upright arrow has become $0$, the area is $0$ and so is $ad - bc$.) The rectangle has sides $a$ and $d - bc/a$, with their signs, so

$$ \det A = a \left( d - \frac{bc}{a} \right) = ad - bc $$

These two sides are the pivots that elimination finds: this is elimination done on the columns instead of the rows.

Example: the scene opens with $a_1 = (2, 0.7)$ and $a_2 = (1, 1.6)$. Sliding $a_2$ by $-0.5\,a_1$ gives $(0, 1.25)$; sliding $a_1$ by $-0.56$ times that gives $(2, 0)$. The rectangle is 2 by 1.25, and

$$ ad - bc = 2 \times 1.6 - 1 \times 0.7 = 2.5 = 2 \times 1.25 $$

## Zero, and space

$\det A = 0$ exactly when the parallelogram is flat: when the columns lie in one line, dependent in the words of the span primer.

In space, three columns $a_1, a_2, a_3$ span a slanted box, and $\det A$ is its volume, base area times height. The cross product $a_1 \times a_2$ stands at right angles to the base and is as long as its area, so its dot product with $a_3$ is that area times the height of $a_3$ above the base:

$$ \det A = (a_1 \times a_2) \cdot a_3 $$

It is positive when $a_3$ is on the side $a_1 \times a_2$ points to (the columns turn like $x, y, z$), negative on the other side, and $0$ exactly when the three columns lie in one plane. Sliding and swapping work as before. In the scene, $a_1 = (2, 0.7, 0)$ and $a_2 = (1, 1.6, 0)$ lie on the floor with $a_1 \times a_2 = (0, 0, 2.5)$, and $a_3 = (0.6, 0.5, 1.6)$ stands 1.6 above it: the volume is $2.5 \times 1.6 = 4$.

# In the picture

- The pink arrow is $a_1$ and the green one $a_2$; drag their tips.
- The parallelogram is yellow when its area counts positive, with a small curved arrow showing the anticlockwise turn, and blue when it counts negative.
- The dashed white line with a corner is the height over the base $a_1$.
- The dashed green line through $a_2$'s tip runs parallel to $a_1$; the white dot slides $a_2$ along it.
- With "three, in space": $a_3$, the box, and the white arrow $a_1 \times a_2$.

> Try first "slide a side": the parallelogram leans from one side to the other, and its area stays 3.2 the whole way.

see: vectors, linmap2, compose, elim

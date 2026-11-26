# The idea

Three equations in two unknowns usually have no solution: there is more to satisfy than there is freedom to satisfy it. Least squares settles for the next best thing, the $x$ that comes nearest, and finding it comes down to dropping a perpendicular onto a plane. This scene shows the problem from the side of the columns of the matrix, where it is a question of geometry: which point of a plane is nearest to a given point? You need vectors, the dot product, span, and matrices.

# The mathematics

## What $Ax$ can reach

$A$ has two columns $a_1$ and $a_2$, each an arrow in space, and

$$ Ax = x_1 a_1 + x_2 a_2 $$

walk $x_1$ times along $a_1$, then $x_2$ times along $a_2$. If $a_1$ and $a_2$ are not in line, the points $Ax$ fill a plane through $0$, the column space. $Ax = b$ has a solution only if $b$ is on that plane, and most points $b$ are not.

## The nearest point is straight below

Least squares looks for the point $p$ of the plane nearest to $b$, and the $x^*$ that reaches it:

$$ p = A x^*, \quad \lVert b - A x^* \rVert = \min_x \lVert b - A x \rVert $$

The name comes from $\lVert b - Ax \rVert^2$, the sum of the squares of how much $Ax$ misses each of the three numbers of $b$. The nearest point is the foot of the perpendicular from $b$, where the error $e = b - p$ meets the plane at a right angle. Why: for any other point $q$ of the plane, $b - q = e + (p - q)$, where $p - q$ lies in the plane and so is at right angles to $e$. By Pythagoras,

$$ \lVert b - q \rVert^2 = \lVert e \rVert^2 + \lVert p - q \rVert^2 $$

which is more than $\lVert e \rVert^2$ unless $q = p$.

## The normal equations

$e$ is at right angles to the plane when it is at right angles to both columns: $a_1 \cdot e = 0$ and $a_2 \cdot e = 0$. Write $A^\top$ for $A$ with its rows and columns swapped, so that the rows of $A^\top$ are $a_1$ and $a_2$, and $A^\top e$ is the pair of numbers $a_1 \cdot e$, $a_2 \cdot e$. The two conditions are then $A^\top (b - Ax^*) = 0$:

$$ A^\top A\, x^* = A^\top b $$

two equations in two unknowns, which have one solution when the columns are not in line. If the columns are at right angles, the two equations come apart: $x_1 = (a_1 \cdot b) / (a_1 \cdot a_1)$, how far $b$ reaches along $a_1$, and $x_2$ likewise. If the columns are nearly in line, $x^*$ can be large, with its two parts of opposite sign, and a small change in a column can swing it a long way.

## An example

Take $a_1 = (1, 1, 1)$, $a_2 = (0, 1, 2)$ and $b = (1, 1, 4)$. Then $a_1 \cdot a_1 = 3$, $a_1 \cdot a_2 = 3$, $a_2 \cdot a_2 = 5$, $a_1 \cdot b = 6$ and $a_2 \cdot b = 9$, so the normal equations are

$$ 3x_1 + 3x_2 = 6 \\ 3x_1 + 5x_2 = 9 $$

with solution $x^* = (0.5, 1.5)$. Then $p = 0.5 a_1 + 1.5 a_2 = (0.5, 2, 3.5)$ and $e = (0.5, -1, 0.5)$. Check: $a_1 \cdot e = 0.5 - 1 + 0.5 = 0$ and $a_2 \cdot e = -1 + 1 = 0$. The least error is $\lVert e \rVert = \sqrt{1.5} \approx 1.22$.

# In the picture

- $a_1$ is green and $a_2$ orange: the columns of $A$. $b$ is white. Drag all three tips.
- The blue sheet is the column space.
- The cyan arrow is $p = Ax^*$; the dashed cyan path gets there by $x_1$ steps along $a_1$, then $x_2$ along $a_2$.
- The pink arrow is the error $e = b - p$; the white corner marks its right angle with the sheet.

> Try first "columns nearly in line": $e$ still meets the sheet at a right angle, but the dashed path runs far back along $a_1$ and out again along $a_2$.

see: vectors, span, overdet, gram

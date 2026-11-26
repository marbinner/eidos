# The idea

Do one linear map and then another. The result is again a linear map, so it has a matrix of its own: the product of the two. This scene plays the two maps one after the other, shows how the product's numbers come from theirs, and shows that the order matters. It ends with the map that undoes another, its inverse, and the maps that nothing can undo. You need the linear maps primer: a matrix's columns are where $\hat\imath$ and $\hat\jmath$ land.

# The mathematics

## The product

Write $AB$ for "first $B$, then $A$": a point $v$ goes to $Bv$ and then to $A(Bv)$, so the map on the right acts first. $\hat\imath$ goes first to $B\hat\imath$, the first column of $B$, and then $A$ moves that: so the columns of $AB$ are $A$ applied to the columns of $B$. Working that out with the column rule gives

$$ \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} e & f \\ g & h \end{bmatrix} = \begin{bmatrix} ae + bg & af + bh \\ ce + dg & cf + dh \end{bmatrix} $$

Each entry is a row of $A$ times a column of $B$: multiply the pairs of numbers and add, as in the dot product. The entry in row $i$ and column $j$ uses row $i$ of $A$ and column $j$ of $B$.

Example: the scene opens with a shear $A$, which slides every point sideways by its height, and a quarter turn $B$:

$$ AB = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 1 & -1 \\ 1 & 0 \end{bmatrix}, \qquad BA = \begin{bmatrix} 0 & -1 \\ 1 & 1 \end{bmatrix} $$

The top left entry of $AB$ is $1 \times 0 + 1 \times 1 = 1$.

## Order matters

$AB$ and $BA$ are often different maps: above, $AB$ sends $\hat\imath$ to $(1, 1)$, and $BA$ sends it to $(0, 1)$. When $AB = BA$ the two maps commute. Two turns about $0$ do (the angles add, in either order), and so do two stretches along the $x$ and $y$ axes.

## Areas multiply

$B$ multiplies every area by $\det B$, and then $A$ multiplies every area by $\det A$. So

$$ \det(AB) = \det A \times \det B $$

with signs: two flips make no flip. The two orders always have the same determinant, even when they are different maps.

## Undoing a map

The identity $I$ leaves every point where it is; its columns are $\hat\imath$ and $\hat\jmath$. The inverse $A^{-1}$ is the map that undoes $A$:

$$ A^{-1}A = AA^{-1} = I $$

It exists exactly when $\det A \ne 0$. If $\det A = 0$, the plane is squashed onto a line or a point, so some two different points land on one spot. No map can part them again, since a map sends each spot to one place. When $\det A \ne 0$,

$$ \begin{bmatrix} a & b \\ c & d \end{bmatrix}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix} $$

as multiplying the two out checks. By the product rule, $\det A^{-1} = 1 / \det A$: the flatter $A$ squashes, the more its inverse must stretch. Example: the shear above has $\det A = 1$, and its inverse, with columns $(1, 0)$ and $(-1, 1)$, slides every point back by its height.

# In the picture

- The blue grid and the yellow unit square are carried along: by the first map as the progress goes from 0 to 1, by the second from 1 to 2. The area on the square is the determinant so far.
- The white letter painted on the square tells turns from flips: after a flip it reads mirrored.
- The pink and green arrows are $\hat\imath$ and $\hat\jmath$ carried along. Their dots halfway along the dashed paths are the first map's columns; their tips at the end are the product's columns.
- The purple outline is where the square and the letter end in the other order.
- Over the picture, cyan boxes mark the row of $A$ and the column of $B$ that make one entry of $AB$; click an entry to pick another.
- The red dots $p$ and $q$, shown when the plane is squashed, are two points that land on one spot.

> Try first "a turn, then a shear": the letter's long stroke ends lying flat, while in the other order (purple) it ends leaning at 45°.

see: linmap2, det, basis, rotations

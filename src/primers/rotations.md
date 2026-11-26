# The idea

A rotation turns all of space about the origin and keeps every length and every angle. It is a linear map, so it has a $3 \times 3$ matrix $R$, whose columns have length 1, are at right angles, and keep the order of $x$, $y$ and $z$. This scene builds rotations from three turns about the axes, shows that their order matters, and shows that every rotation, however it was made, is one turn about one line, its axis. You need the primers on linear maps in 3D, on products of maps and on change of basis.

# The mathematics

## What the columns must be

The columns of $R$ are where the steps $e_1, e_2, e_3$ along the axes go: $Re_1$, $Re_2$, $Re_3$. A rotation keeps lengths and angles, so these have length 1 and are at right angles: they are orthonormal. In one equation,

$$ R^\top R = I $$

where $R^\top$, the transpose, is $R$ with its rows and columns swapped. The entry of $R^\top R$ in row $i$ and column $j$ is the dot product of columns $i$ and $j$ of $R$: 1 when $i = j$ (length 1) and 0 otherwise (right angles). So $R^\top$ is the inverse of $R$, the turn back. A rotation also has $\det R = 1$: it keeps volumes and does not turn space inside out. A mirror also has orthonormal columns, but its determinant is $-1$.

## Turns about the axes

A turn by $\theta$ about the $z$ axis sends $e_1$ to $(\cos\theta, \sin\theta, 0)$ and $e_2$ to $(-\sin\theta, \cos\theta, 0)$, and leaves $e_3$ where it is:

$$ R_z = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix} $$

$R_x$ and $R_y$, about $x$ and $y$, are built the same way. The scene's three sliders make $R = R_z R_y R_x$: about $x$ first and $z$ last, since the matrix on the right acts first. As with any product of maps, the order matters. Quarter turns about $x$ and then $y$ give

$$ R_y R_x = \begin{bmatrix} 0 & 0 & 1 \\ 0 & 1 & 0 \\ -1 & 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 & 0 \\ 0 & 0 & -1 \\ 0 & 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & -1 \\ -1 & 0 & 0 \end{bmatrix} $$

which sends the nose, $e_1$, to $(0, 0, -1)$, straight down. In the other order, $R_x R_y$ sends it to $(0, 1, 0)$.

## One turn about one axis

However a rotation is made, it is one turn by some angle $\theta$ about one line, its axis. Arrows $v$ along the axis do not move: they are eigenvectors with eigenvalue 1, $Rv = v$. The angle can be read from the diagonal:

$$ \tr R = 1 + 2\cos\theta $$

Why: in a basis of two arrows of length 1 at right angles across the axis, and a third along it, the turn has a matrix like $R_z$ above, whose diagonal adds up to $1 + 2\cos\theta$; and the trace is the same in every basis.

Example: $R_y R_x$ above has trace $0$, so $\cos\theta = -1/2$ and $\theta = 120^\circ$. It sends $(1, 1, -1)$ to $(1, 1, -1)$, so that line is its axis: two quarter turns make one turn of 120°.

## Gimbal lock

Three angles about the fixed axes reach every rotation, but not always smoothly. When the 2nd turn is $-90^\circ$ about $y$, the nose points straight up, and the 1st and 3rd angles then turn the aeroplane about the same line: one way of turning cannot be reached by changing the angles a little.

# In the picture

- The red, green and blue arrows are $Re_1$, $Re_2$, $Re_3$, the columns of $R$, along the aeroplane's nose, left wing and fin.
- The yellow line is the axis of the single turn, with the angle $\theta$ curled round it; drag the yellow dot to point it anywhere. The dashed arcs are the paths of the arrow tips in that one turn.
- Grey lines are the axes of space; the faint outline is the aeroplane before any turn.
- Purple shows the same turns made in the other order.
- Over the picture: $R$, with $R^\top R = I$, $\det R$, and $\tr R = 1 + 2\cos\theta$ checked in numbers.

> Try first "order matters": two quarter turns in the two orders end 120° apart.

see: linmap3, compose, basis, gram

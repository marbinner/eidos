# The idea

This is the linear maps scene one dimension up. A $3 \times 3$ matrix moves every point of space, and its three columns say where: they are where the three steps along the axes land, and every other point follows. The unit cube becomes a slanted box, and the box's volume, the determinant, says how much the map scales volumes. You need the primer on linear maps in the plane: all of it carries over.

# The mathematics

## Three columns

Write $e_1 = (1, 0, 0)$, $e_2 = (0, 1, 0)$ and $e_3 = (0, 0, 1)$: one step along $x$, along $y$ and along $z$. Every point is the walk $x e_1 + y e_2 + z e_3$, and a linear map keeps walks as walks, so

$$ A \begin{bmatrix} x \\ y \\ z \end{bmatrix} = x\,Ae_1 + y\,Ae_2 + z\,Ae_3 $$

The matrix $A$ has $Ae_1$, $Ae_2$ and $Ae_3$ as its three columns. The scene opens with

$$ A = \begin{bmatrix} 1.2 & -0.4 & 0.2 \\ 0.3 & 1 & 0.2 \\ 0 & 0.3 & 1.4 \end{bmatrix} $$

so $e_1$ lands on $(1.2, 0.3, 0)$, and the far corner $(1, 1, 1)$ of the cube lands on the sum of the three columns, $(1, 1.5, 1.7)$.

## Volume: the determinant

The unit cube lands on the slanted box with the three columns as edges. Its volume is $\det A$; here $\det A = 1.794$. Every small cube of a fine grid lands on a scaled copy of that box, so every volume is multiplied by $|\det A|$. The volume counts as negative when the map turns space inside out, as a mirror does, so that a right hand would come out as a left hand.

## Squashed space and rank

If the three columns lie in one plane (they are dependent, as in the span primer), all of space lands on that plane: the box is flat and $\det A = 0$. The rank is the dimension of where space lands, the span of the columns: 3, 2, 1 or 0. Below 3, a whole line of points, or more, lands on $0$.

Example: in "squash onto a plane" the columns are $(1, 0, 0.5)$, $(0, 1, 0.5)$ and their sum $(1, 1, 1)$. So the point $(1, 1, -1)$ lands on

$$ Ae_1 + Ae_2 - Ae_3 = (0, 0, 0) $$

and so does every multiple of it: an eigenvector line with $\lambda = 0$.

## Lines that stay put, and stretches

An eigenvector is again an arrow $v$, not $0$, that the map only stretches: $Av = \lambda v$. In the plane a map can have no such line (a turn). In space every map has at least one: the eigenvalues are the solutions of an equation of degree 3 in $\lambda$, and such an equation always has a real solution. The matrix above has one, $\lambda = 1.528$; its other two eigenvalues are complex.

The unit ball lands on an ellipsoid, a stretched ball with three half-axes. They are the singular values $\sigma_1 \ge \sigma_2 \ge \sigma_3$: $\sigma_1$ is the most the map stretches any arrow, and $\sigma_3$ the least. As in the plane, comparing volumes gives

$$ \sigma_1 \sigma_2 \sigma_3 = |\det A| $$

since a ball of radius 1 has volume $\tfrac{4}{3}\pi$ and the ellipsoid $\tfrac{4}{3}\pi\sigma_1\sigma_2\sigma_3$. Here $1.536 \times 1.272 \times 0.918 \approx 1.794$. The number of singular values that are $0$ is 3 minus the rank.

# In the picture

- The red, green and blue arrows, labelled $e_1$, $e_2$, $e_3$, are where those steps land: the columns $Ae_1$, $Ae_2$, $Ae_3$. Drag their tips, or type in the matrix.
- The faint wire cube is the unit cube before the map. The blue box is where it lands, with its volume written on it; it turns pink when $\det A$ is negative.
- The dots are a $5 \times 5 \times 5$ grid of points carried along by the map, each coloured by where it started.
- The dashed orange, purple and yellow lines are the real eigenvector lines, each labelled with its $\lambda$.

> Try first "squash onto a plane": the box goes flat, and a dashed line with $\lambda = 0$ shows the points that land on $0$.

see: linmap2, det, rotations, subspaces

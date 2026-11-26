# The idea

Every $2 \times 2$ matrix $A$, however it looks, does three plain things one after another: it turns the plane, stretches it along the two axes, and turns it again. Written as matrices this is $A = U \Sigma V^\top$, the singular value decomposition (SVD). It says in which directions a matrix stretches most and least, and by how much, and it is the tool behind low-rank approximation and principal components. You need matrices as maps, one map after another, determinants, and the fact that a turn keeps lengths and angles.

# The mathematics

## Circle to ellipse

$A$ sends the unit circle to an ellipse, pressed flat if $\det A = 0$. Call the ellipse's long half-axis $\sigma_1 u_1$ and its short one $\sigma_2 u_2$: $u_1$ and $u_2$ are unit arrows at right angles, and the lengths $\sigma_1 \ge \sigma_2$ are the singular values. Let $v_1$ and $v_2$ be the arrows of the circle that land there:

$$ A v_1 = \sigma_1 u_1, \qquad A v_2 = \sigma_2 u_2 $$

So $v_1$ is the direction $A$ stretches most, and $v_2$ the one it stretches least. The fact that makes everything work: $v_1$ and $v_2$ are at right angles too. The reason is that the squared length $|Ax|^2 = x^\top (A^\top A) x$ is a quadratic form, with the symmetric matrix $A^\top A$, and on the unit circle such a form is largest and smallest along two eigenvectors at right angles.

## Three steps

Take $\det A > 0$ for now, and $v_2$ a quarter turn anticlockwise from $v_1$ (if not, reverse $v_2$ and $u_2$ together). Let $V$ be the turn that takes the $x$ and $y$ axes onto $v_1$ and $v_2$; its transpose $V^\top$ is the turn back. Let $\Sigma$ stretch along the axes by $\sigma_1$ and $\sigma_2$, and let $U$ turn the axes onto $u_1$ and $u_2$. Follow $v_1$: $V^\top$ turns it onto $e_1 = (1, 0)$, $\Sigma$ stretches that to $\sigma_1 e_1$, and $U$ turns it to $\sigma_1 u_1$, which is $Av_1$. The same happens to $v_2$. Two linear maps that agree on a basis agree everywhere, so

$$ A = U \Sigma V^\top $$

read from right to left: first $V^\top$, then $\Sigma$, then $U$.

In this scene $U$ and $V$ are always turns, never mirror images. A turn keeps areas, so all the change of area is in $\Sigma$: $\det A = \sigma_1 \sigma_2$. A matrix that flips the plane over ($\det A < 0$) therefore needs a negative $\sigma_2$: the scene reverses $u_2$ and the sign of $\sigma_2$, which leaves $\sigma_2 u_2$ as it was. (Books usually keep $\sigma_2 \ge 0$ and let $U$ or $V$ be a mirror image instead.)

## An example

$$ A = \begin{bmatrix} 2 & -1 \\ 2 & 1 \end{bmatrix} $$

$A$ sends $e_1$ and $e_2 = (0, 1)$ to its columns, $(2, 2)$ and $(-1, 1)$. These are already at right angles, with lengths $\sqrt{8} = 2.83$ and $\sqrt{2} = 1.41$. So $v_1$ and $v_2$ are the axes themselves and the first turn is 0°; $\sigma_1 = 2.83$, $\sigma_2 = 1.41$; and $U$ turns by 45°, taking $e_1$ to $(1, 1)/\sqrt{2}$. Check: $\sigma_1 \sigma_2 = \sqrt{16} = 4$, and $\det A = 2 \cdot 1 - (-1) \cdot 2 = 4$. For the scene's first matrix, too, $1.868 \times 1.087 = 2.03 = \det A$.

# In the picture

- White: the unit circle, carried through the three steps until it is the ellipse. The faint circle is where it started; the beads on the white curve make turns visible.
- Blue: the square grid, carried along.
- The orange arrow starts as $v_1$, becomes $e_1$ after $V^\top$, $\sigma_1 e_1$ after $\Sigma$, and $\sigma_1 u_1$ after $U$. The purple arrow does the same with $v_2$, $e_2$, $\sigma_2$ and $u_2$.
- The title names the step in its colour: cyan for $V^\top$, yellow for $\Sigma$, pink for $U$. The panel gives each turn's angle and the two stretches.

> Try first "stretch, then turn": the decomposition finds the very turn and stretches the matrix was made from.

see: linmap2, quadform, lowrank, pca

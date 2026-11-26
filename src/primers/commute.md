# The idea

Doing a map $B$ and then a map $A$ is the product $AB$; doing $A$ and then $B$ is $BA$. For numbers the order never matters, but for matrices it usually does. This scene measures by how much, with a matrix of its own, and asks which matrices commute and why. Much of the answer is in the eigenlines. You need the primers on one map after another (products) and on eigenvalues.

# The mathematics

## The commutator

The difference of the two orders is the commutator:

$$ [A, B] = AB - BA $$

It sends a point $x$ to $ABx - BAx$, the gap between where the two orders take it. So its two columns are the gaps for $(1, 0)$ and for $(0, 1)$. $A$ and $B$ commute when it is $0$. Its size is

$$ \|M\| = \sqrt{\text{sum of the squares of the entries of } M} $$

Example: $A$ stretches $x$ to 1.6 times and $y$ to 0.6, and $B$ is a quarter turn.

$$ AB = \begin{bmatrix} 0 & -1.6 \\ 0.6 & 0 \end{bmatrix}, \quad BA = \begin{bmatrix} 0 & -0.6 \\ 1.6 & 0 \end{bmatrix}, \quad AB - BA = \begin{bmatrix} 0 & -1 \\ -1 & 0 \end{bmatrix} $$

of size $\sqrt{2} = 1.41$. Whatever $A$ and $B$ are, the trace (the sum of the diagonal) of $AB - BA$ is $0$: $\tr AB$ and $\tr BA$ are both the sum of all the products $a_{ij} b_{ji}$. So $AB - BA$ is never $I$, whose trace is $2$.

## What commutes, and why

Some pairs always commute: any matrix with a multiple of $I$; $A$ with its own powers, with any polynomial in it such as $A^2 - 3A$, and with $A^{-1}$ if it has one; two turns of the plane, whose angles add in either order; two diagonal matrices.

The eigenlines say why. Suppose $A$ has two different real eigenvalues, so two eigenlines. Then

$$ AB = BA \iff B \text{ keeps both of } A\text{'s eigenlines} $$

If $AB = BA$ and $Av = \lambda v$, then $A(Bv) = BAv = \lambda\, Bv$: $Bv$ is again on the line of eigenvectors for $\lambda$. The other way round, if $B$ keeps both lines, then in a basis along them both matrices are diagonal, and diagonal matrices commute.

The condition matters. If $A = cI$, every line is an eigenline and every matrix commutes with it. A turn by anything but $0^\circ$ or $180^\circ$ keeps no line at all (its eigenvalues are complex). For any $2 \times 2$ matrix $A$ that is not a multiple of $I$, the matrices that commute with it are exactly

$$ aI + bA \qquad (a, b \text{ numbers}) $$

For such a turn these are the maps that turn and stretch evenly, so it commutes with no uneven stretch.

For two symmetric matrices, $(AB)^\top = B^\top A^\top = BA$, so their product is symmetric exactly when they commute.

## Small maps nearly commute

Take maps near the identity, $P = I + eA$ and $Q = I + eB$ for a small number $e$. Multiplied out, the terms with one $e$ are the same in both orders and cancel:

$$ PQ - QP = e^2 (AB - BA) $$

Halving $e$ divides the gap by 4. On logarithmic scales the gap against $e$ is a straight line of slope 2.

## In space

Turns of space about different axes usually do not commute: a quarter turn about $x$ and then about $y$ ends $120^\circ$ away from the same turns in the other order. Half turns about $x$ and about $y$ are an exception: both matrices are diagonal.

# In the picture

- Yellow: the letter and the unit square carried by $B$ and then $A$, ending at $AB$. White: by $A$ and then $B$, ending at $BA$. Faint outlines: where each is after its first step.
- Red arrows: from where $BA$ sends $(1, 0)$ and $(0, 1)$ to where $AB$ does, the columns of $AB - BA$, which the panel writes out.
- Orange and purple bands: $A$'s eigenlines, with eigenvalues $\lambda$. Dashed blue: $B$'s, with $\mu$. The orange and blue dots turn all of $A$ or of $B$.
- With small maps, the chart shows the gap against $e$. In space, the two books are the two orders.

> Try first "the same axes, then turned": the two orders agree while $B$'s blue lines lie on $A$'s, and part as soon as they turn off them.

see: compose, eigen, expsum, rotations

# The idea

The eigenvalues of a triangular matrix can be read off its diagonal. Most matrices are not triangular, but every one whose eigenvalues are real is triangular in a well-chosen perpendicular basis: its Schur form. The QR algorithm finds that basis by repeating one move: factor the matrix into a turn (or mirror) times a triangle, and multiply them back in the other order. Refined, it is how computers find eigenvalues. It builds on eigenvalues, the QR factorisation and the power method.

# The mathematics

## Triangular matrices and the Schur form

If $T$ is upper triangular, so is $T - \lambda I$, and its determinant is the product of its diagonal, $(t_{11} - \lambda)(t_{22} - \lambda) \cdots$. So the eigenvalues of $T$ are its diagonal entries.

A change of basis turns $A$ into $P^{-1} A P$ and keeps its eigenvalues. For an orthogonal $Q$ (perpendicular columns of length 1) $Q^{-1} = Q^\top$, and the Schur form is

$$ A = Q T Q^\top, \quad Q \text{ orthogonal}, \ T \text{ upper triangular} $$

It exists whenever the eigenvalues of $A$ are real. Take an eigenvector of length 1 and add perpendicular directions to make a basis; in it the first column of the matrix is $(\lambda_1, 0, \ldots, 0)$. Do the same to the block below and to the right of $\lambda_1$, which has the other eigenvalues, and so on. A complex pair $a \pm bi$ keeps a $2 \times 2$ block on the diagonal instead. For a symmetric $A$, $T$ is symmetric and triangular, so diagonal: the spectral theorem.

## The QR algorithm

From $A_0 = A$, repeat

$$ A_k = Q_k R_k, \qquad A_{k+1} = R_k Q_k $$

with $Q_k$ orthogonal and $R_k$ upper triangular. Since $R_k = Q_k^\top A_k$, the new matrix is $A_{k+1} = Q_k^\top A_k Q_k$: the same map in a turned basis, so the same eigenvalues.

An example:

$$ A = \begin{bmatrix} 3 & 1 \\ 4 & 2 \end{bmatrix} $$

Its first column $(3, 4)$ has length 5, so $Q_0$ has columns $(0.6, 0.8)$ and $(-0.8, 0.6)$, and $R_0 = Q_0^\top A$ has rows $(5, 2.2)$ and $(0, 0.4)$. Then

$$ A_1 = R_0 Q_0 = \begin{bmatrix} 4.76 & -2.68 \\ 0.32 & 0.24 \end{bmatrix} $$

The trace is still 5 and the determinant still 2, but the entry below the diagonal fell from 4 to 0.32. The eigenvalues are 4.56 and 0.44; from here that entry shrinks by about $0.44/4.56 = 0.096$ a step (0.028, then 0.0027).

## Why the lower part dies

Multiplying the steps together gives $A^k = (Q_0 Q_1 \cdots Q_{k-1})(R_{k-1} \cdots R_1 R_0)$. So the first column of $Q_0 \cdots Q_{k-1}$ is $A^k$ applied to the first axis, scaled to length 1: the power method. Its first two columns span what $A^k$ makes of the first two axes, and so on. With the eigenvalues in order of size, $|\lambda_1| > |\lambda_2| > \cdots$, the entry between the $i$-th and $(i+1)$-th diagonal entries typically shrinks by about $|\lambda_{i+1}| / |\lambda_i|$ a step: slowly when two are close in size. Two different ones of the same size, such as a complex pair, are not told apart: a $2 \times 2$ block stays, and its eigenvalues come to be the pair.

## Shifts

$A - sI$ has the eigenvalues $\lambda - s$. Factor $A_k - sI = Q_k R_k$ and take $A_{k+1} = R_k Q_k + sI$, still $Q_k^\top A_k Q_k$. With $s$ the bottom-right entry, the last row's entry below the diagonal can fall very fast: in "the same with shifts" its zeros after the point roughly double with each of the last steps. That row is then set aside and the work moves up.

# In the picture

- The grid is $A_k$, each cell coloured by its entry's size on a log scale; white outlines the diagonal; $a_{21}$, $a_{32}$, $a_{43}$ are red, orange and yellow.
- The chart: their sizes against the step $k$, the predicted rates dashed.
- The complex plane: green rings are the eigenvalues of $A$, white dots the diagonal entries (or a block's complex pair).
- With shifts, blue marks the part worked on and its corner $s$.

> Try first "eigenvalues well apart": the lower triangle darkens, each entry along its own dashed line.

see: gram, powers, spectral, jordan

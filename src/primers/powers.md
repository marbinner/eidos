# The idea

Take a start $x_0$ and apply a matrix $A$ to it again and again: $x_1 = Ax_0$, $x_2 = Ax_1 = A^2 x_0$, and after $k$ steps $x_k = A^k x_0$. Steps like this describe people moving between places year by year, or the Fibonacci numbers. Multiplying the matrices out hides what happens; eigenvectors show it: in the long run the eigenvalue largest in size usually takes over. You need eigenvalues and eigenvectors, $Av = \lambda v$, and writing a vector in a basis.

# The mathematics

## Each eigenvector on its own

On an eigenvector $v$, each step multiplies by $\lambda$: $A^k v = \lambda^k v$. Suppose $A$ has two eigenvector lines, along $v_1$ and $v_2$. They are a basis, so the start can be written $x_0 = c_1 v_1 + c_2 v_2$, and since $A$ is linear each part goes its own way:

$$ x_k = A^k x_0 = c_1 \lambda_1^k v_1 + c_2 \lambda_2^k v_2 $$

## The largest eigenvalue wins

If one eigenvalue is larger in size, call it $\lambda_1$: $|\lambda_1| > |\lambda_2|$. Every step multiplies the second part's length over the first's by $|\lambda_2| / |\lambda_1| < 1$. So if $c_1 \ne 0$ the second part dies away against the first, $x_k$ turns ever closer to $v_1$'s line, and its growth per step comes ever nearer $|\lambda_1|$. Then if $|\lambda_1| > 1$, $x_k$ grows without end; if both are below 1 in size it falls to $0$. Finding $v_1$ this way is the power method.

On the scene's chart each mark up is ten times the one below (a log scale), so a part multiplied by the same factor every step draws a straight line.

The scene's first matrix:

$$ A = \begin{bmatrix} 1.32 & -0.36 \\ 0.24 & 0.48 \end{bmatrix} $$

It sends $(3, 1)$ to $(3.6, 1.2) = 1.2 \cdot (3, 1)$ and $(1, 2)$ to $(0.6, 1.2) = 0.6 \cdot (1, 2)$. So $\lambda_1 = 1.2$ along $(3, 1)$ and $\lambda_2 = 0.6$ along $(1, 2)$. The start $x_0 = (2, 3)$ is $0.2 \cdot (3, 1) + 1.4 \cdot (1, 2)$, so

$$ x_k = 0.2 \cdot 1.2^k \, (3, 1) + 1.4 \cdot 0.6^k \, (1, 2) $$

One step gives $0.24 \cdot (3, 1) + 0.84 \cdot (1, 2) = (1.56, 1.92)$, the same as $Ax_0$. The second part starts nearly five times as long as the first and halves against it every step ($0.6 / 1.2 = 0.5$), so after 10 steps it is under 1% of it. (The scene takes $v_1$ and $v_2$ of length 1, so its $c_1$ and $c_2$ are $0.632$ and $3.13$.)

## Other kinds of matrix

- A Markov matrix has entries $\ge 0$ and columns that add to 1, so the total is kept. One eigenvalue is 1; when the other is smaller than 1 in size, $x_k$ settles on the steady state: the eigenvector of $\lambda = 1$ with the same total as $x_0$.
- With complex eigenvalues $r(\cos\theta \pm i \sin\theta)$ no line is kept and $x_k$ keeps going round, shrinking to $0$ if $r < 1$ and growing without end if $r > 1$. A turn by $\theta = 30^\circ$ with a shrink to $r = 0.9$ goes once round in twelve steps, leaving $0.9^{12} = 0.28$ of the length.
- The shear has $\lambda = 1$ twice but only one eigenvector line, so the formula above does not apply: from $(0, 1)$ it gives $x_k = (k, 1)$, growth by adding instead of multiplying.

# In the picture

- White dots: $x_0, x_1, x_2, \ldots$, joined in order. Drag $x_0$.
- Orange: $v_1$'s line (dashed) and the part $c_1 \lambda_1^k v_1$; purple: $v_2$'s line and $c_2 \lambda_2^k v_2$. The two arrows add up to $x_k$.
- The upper chart: the lengths of the parts and of $x_k$, on a log scale. The lower chart, yellow: the angle between $x_k$ and $v_1$'s line.
- When they appear: cyan, a ring of starts stepped together; green, a Markov matrix's line of equal totals; grey dashed, for complex eigenvalues, the ellipse $x$ goes round.

> Try first "the largest eigenvalue wins": $x$ falls in towards $0$, then turns and runs out along the orange line.

see: eigen, basis, expm

# The idea

A matrix $A$ takes an input $x$ to an output $Ax$. Four sets of vectors say what it does: among the inputs, those it sends to $0$ and those it answers to; among the outputs, those it can reach and the directions no output has any part along. Together they say when $Ax = b$ has a solution, and how many. You need vectors, the dot product, span, and matrices as maps.

# The mathematics

A subspace is a set of vectors through the origin that holds every combination of its members: in space, the origin alone, a line, a plane, or all of space. Its dimension is 0, 1, 2 or 3.

## Inputs: the row space and the null space

Each number of $Ax$ is the dot product of a row of $A$ with $x$. So an $x$ at right angles to every row gives $Ax = 0$. Those inputs make up the null space. The row space is every combination of the rows.

The two are at right angles, and between them they fill the space of inputs: every $x$ splits in one way into a row part and a null part,

$$ x = x_r + x_n, \qquad Ax = Ax_r + Ax_n = Ax_r $$

so only the row part matters to $A$. The inputs with the same output as $x$ are $x_r$ plus anything in the null space.

## Outputs: the column space and the left null space

With columns $a_1, a_2, a_3$,

$$ Ax = x_1 a_1 + x_2 a_2 + x_3 a_3 $$

so the outputs are the combinations of the columns: the column space. $Ax = b$ has a solution exactly when $b$ is in it. The left null space is every $y$ at right angles to every column; no output has any part along such a $y$. It is called left because $y$, written as a row to the left of $A$, gives $y^\top A = 0$.

## Rank, and why the counts add up

No two points of the row space have the same output: their difference would be in both the row space and the null space, which are at right angles, so it is at right angles to itself, and only $0$ is. And every output is $Ax_r$ for some row part. So $A$ matches the row space with the column space point for point, and the two have the same dimension, the rank $r$. Here inputs and outputs have 3 numbers each:

$$ \dim(\text{row space}) + \dim(\text{null space}) = r + (3 - r) = 3 \\ \dim(\text{column space}) + \dim(\text{left null space}) = r + (3 - r) = 3 $$

For other shapes, the first 3 is the number of columns, the second the number of rows.

The panel also gives the singular values $\sigma_1 \ge \sigma_2 \ge \sigma_3 \ge 0$, how much $A$ stretches its most and least stretched directions; the rank is how many are not 0.

## An example

The scene opens with

$$ A = \begin{bmatrix} 0.3 & 0.6 & 0.3 \\ 0.6 & 0 & 0 \\ -0.3 & 0.6 & 0.3 \end{bmatrix} $$

Row 3 is row 1 minus row 2, and rows 1 and 2 are not in line, so the rows span a plane: $r = 2$. Column 2 is twice column 3, so $A(0, 1, -2) = 0$: the null space is the line through $(0, 1, -2)$, of dimension $3 - 2 = 1$. And $(1, -1, -1)$ has dot product 0 with each column (for column 1, $0.3 - 0.6 + 0.3 = 0$), so the left null space is the line through it.

# In the picture

- The first ball holds the inputs, the second the outputs.
- Blue is the row space and $x_r$; pink is the null space and $x_n$.
- Green is the column space; orange is the left null space.
- White are $x$ and $Ax$ (drag either); white dashed, every input with the same output as $x$.
- A sheet is a plane, a thick line a line, a tinted ball all of space, a ringed dot the origin alone. The paler arrows are bases: $v$ for the row space, $n$ for the null space, $u$ for the column space, $w$ for the left null space.

> Try first "slide $x$ along the null space": $x_n$ changes, $x_r$ does not, and $Ax$ stays where it is.

see: span, elim, lsq, underdet

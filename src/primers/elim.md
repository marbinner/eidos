# The idea

An equation in $x$, $y$ and $z$, like $2x + y + z = 5$, holds on a plane: the points that satisfy it. Three equations are three planes, and a point on all three is a solution. Elimination finds it with one move, made again and again: an equation is replaced by itself minus a multiple of another. Each move takes an unknown out of an equation, until the answer can be read off. You need to know what a matrix and a determinant are.

# The mathematics

## One move that keeps the solutions

The matrix $A$ holds the numbers on the left of the equations, a row for each, and $b$ those on the right. The move is a row operation:

$$ \text{row } i \;\to\; \text{row } i - \ell \times \text{row } j $$

A point that satisfies rows $i$ and $j$ satisfies the new row too, and adding $\ell \times \text{row } j$ back undoes the move: no solution is gained or lost. In the picture, when the two planes meet in a line, the new plane still holds that line and turns about it; the solution, on that line, never moves.

## Going down, then back up

To take $x$ out of row 2, use row 1: its number in front of $x$ is the pivot, and the multiplier is

$$ \ell = \frac{\text{the number to take out}}{\text{the pivot}} $$

Clear $x$ from the rows below the first, then $y$ from those below the second. What is left is triangular, $U$: the last row holds $z$ alone, the one above it $y$ and $z$. Back substitution goes back up: $z$ from the last row, then $y$ from the row above, and so on.

An example, the scene's opening:

$$ x - y = -3 \\ 3x + 2y + 3z = 4 \\ -2x - 3y + 2z = -2 $$

Row 2 minus 3 times row 1 is $5y + 3z = 13$. Row 3 minus $-2$ times row 1 is $-5y + 2z = -8$. Then row 3 minus $-1$ times the new row 2 is $5z = 5$. So $z = 1$; then $5y + 3 = 13$ gives $y = 2$, and $x - 2 = -3$ gives $x = -1$. The pivots are 1, 5 and 5; the multipliers $\ell_{21}, \ell_{31}, \ell_{32}$ are 3, $-2$ and $-1$.

## What the record gives: $A = LU$ and $\det A$

Let $L$ hold the multipliers below its diagonal and ones on it. Then

$$ A = LU $$

since each row of $A$ is its row of $U$ plus the multiples of rows of $U$ that were taken from it. A row operation slants the box the rows span without changing its volume, so it leaves the determinant alone; and a triangular matrix's determinant is the product of its diagonal. So $\det A$ is the product of the pivots: $1 \times 5 \times 5 = 25$ here.

## When a pivot is 0

If a 0 stands where a pivot should be, a row below with a number there that is not 0 swaps places with it. That changes the determinant's sign; with swaps, $PA = LU$, where $P$ puts the rows in their new order. If there is no such row, the column has no pivot and $A$ is singular: some row ends as $0 = \text{a number}$. If that number is not 0, there is no solution. If every such row reads $0 = 0$, there are infinitely many, and the column's unknown is free: it can be anything.

# In the picture

- The pink, green and blue planes are equations 1, 2 and 3, labelled with the equations as they now stand.
- The yellow line is what a plane turns about in this step: where it meets the plane of the pivot's row.
- The white dot is the solution; a white line is a line of solutions.
- Over the picture, the pivot is ringed and the number the step takes out is boxed; $L$ fills with the multipliers.

> Try first "two lines": one move turns the green line about the crossing until it is level, and the crossing does not move.

see: det, subspaces, overdet, condition

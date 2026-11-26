# The idea

Read a matrix $A$ as a rule for velocities: the point $x$ moves with velocity $Ax$, and following that flow for a time $t$ is the matrix $e^{tA}$ (the scene on the matrix exponential). With two matrices $A$ and $B$ there are two ways to use both: give every point both velocities at once, or follow one flow and then the other. For numbers $e^{a+b} = e^a e^b$, so the two ways agree. For matrices they agree when $AB = BA$, and in general they do not. This scene measures the difference, and shows how alternating short steps of each flow closes it. You need the matrix exponential and the product of two matrices.

# The mathematics

## Two ways to combine two flows

Both at once, the velocity is $Ax + Bx = (A + B)x$: the flow of $A + B$. One after the other is a product, whose right-hand matrix acts first ($B$ for the time $t$, then $A$):

$$ e^{t(A+B)} x_0 \qquad \text{against} \qquad e^{tA} e^{tB} x_0 $$

If $AB = BA$, the two are equal for every $t$: the series of $e^{t(A+B)}$ can then be multiplied out as if $A$ and $B$ were numbers.

## Where they first differ

Multiply out the series as far as $t^2$:

$$ e^{tA} e^{tB} = I + t(A + B) + t^2 \left( \tfrac12 A^2 + AB + \tfrac12 B^2 \right) + \cdots $$

$$ e^{t(A+B)} = I + t(A + B) + \tfrac12 t^2 \left( A^2 + AB + BA + B^2 \right) + \cdots $$

The first two terms agree. The terms in $t^2$ differ by half of $AB - BA$, the commutator:

$$ e^{tA} e^{tB} - e^{t(A+B)} = \frac{t^2}{2} (AB - BA) + \text{terms in } t^3 \text{ and higher} $$

So for a short time the gap between the two matrices is about $(t^2/2) \lVert AB - BA \rVert$, where the size $\lVert M \rVert$ of a matrix is the square root of the sum of the squares of its entries. Halve $t$ and the gap falls to about a quarter. If the two agreed for every $t$, this $t^2$ term would have to be $0$: so they agree for every $t$ exactly when $AB = BA$.

Example: the stretch $A$ and the turn $B$,

$$ A = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}, \quad B = \begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}, \quad AB - BA = \begin{bmatrix} 0 & -2 \\ -2 & 0 \end{bmatrix} $$

of size $\sqrt{8} = 2.83$. At $t = 0.1$ the formula gives $0.005 \times \sqrt{8} = 0.01414$; the true gap is 0.01416. Here $(A + B)^2 = 0$, so the series stops after two terms and $e^{t(A+B)} = I + t(A + B)$: both at once, every point slides along a straight line.

## Small steps

Split the time into $n$ rounds, each $B$ for $t/n$ and then $A$ for $t/n$:

$$ \left( e^{tA/n} e^{tB/n} \right)^n \to e^{t(A+B)} \quad \text{as } n \to \infty $$

Each round is off by an amount of order $(t/n)^2$, and there are $n$ of them, which makes $n (t/n)^2 = t^2/n$: the gap falls like $1/n$. For the pair above at $t = 1$ it is 0.355, 0.177, 0.088 and 0.044 for $n = 4$, 8, 16 and 32.

The symmetric round puts half of $A$'s step on each side of $B$'s:

$$ \left( e^{tA/2n} e^{tB/n} e^{tA/2n} \right)^n $$

Write one round as $e^{h(A+B) + E}$, with $h = t/n$ and $E$ its error. Turning $h$ into $-h$ makes the symmetric round exactly its inverse, $e^{-h(A+B) - E}$, so $E$ must change sign with $h$. A term in $h^2$ does not, so there is none. What is left is of order $(t/n)^3$ a round and $t^3/n^2$ in all, so the gap falls like $1/n^2$: 0.042, 0.0105, 0.0026 and 0.00066 for the same $n$, for nearly the same work.

# In the picture

- White: the path of $x_0$ (drag it) with both velocities at once, ending at $e^{t(A+B)} x_0$.
- Blue, then orange: $B$ alone, then $A$ alone, in $n$ rounds: the zig-zag, ending at the rounds' product times $x_0$. Dots mark where each round ends (up to 16 rounds).
- Red: the gap between the two ends. On the chart, the gap between the matrices against $n$ (or against $t$) on logarithmic axes, where a gap like $1/n$ is a straight line of slope $-1$, and one like $1/n^2$ of slope $-2$.
- The panel writes out $A$, $B$, $AB - BA$ and both products.

> Try first "a stretch and a turn", then "small steps close the gap": the zig-zag closes in on the white path as $n$ doubles.

see: expm, commute, symskew

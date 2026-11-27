# The idea

A layer multiplies its input by a weight matrix $W$, and training moves $W$ along the gradient of the loss, a matrix too. Often a few of its directions are far stronger than the rest, so a gradient step changes the layer mostly along those. Muon takes the momentum of the gradient, keeps its directions, pushes their strengths towards 1, and steps along that. This scene takes one step apart on a real layer. You need the singular value decomposition (scene svd), the polar factor (scene polar) and momentum.

# The mathematics

## Singular values and the polar factor

Any matrix can be written $G = USV^\top$: $U$ and $V$ have orthonormal columns (the singular vectors), and $S$ is diagonal with the singular values $s_1 \ge s_2 \ge \dots \ge 0$. The polar factor $UV^\top$ is $G$ with every singular value set to 1; for $G$ of full rank it is the matrix with orthonormal rows (or columns) nearest to $G$ in the Frobenius norm, the root of the sum of the squared entries.

## The rule, and why a polynomial acts on each singular value

For a layer with gradient $G$, the reference code takes

$$ M \leftarrow \beta M + (1 - \beta)\,G, \qquad \tilde M = (1 - \beta)\,G + \beta M $$

with $\beta = 0.95$, divides $\tilde M$ by its Frobenius norm $\|\tilde M\|_F$, and applies five rounds of

$$ X \leftarrow aX + b\,(XX^\top)X + c\,(XX^\top)^2 X $$

before the step $W \leftarrow W - \eta\,\sqrt{\max(1, \text{fan-out}/\text{fan-in})}\,X$. If $X = USV^\top$, then $XX^\top = US^2U^\top$ and $(XX^\top)^k X = US^{2k+1}V^\top$. So a round gives $U\,p(S)\,V^\top$ with

$$ p(s) = as + bs^3 + cs^5 $$

applied to each singular value alone; the singular vectors never change, and a value of 0 stays 0. The division by $\|\tilde M\|_F$, at least $s_1$, puts every value in $[0, 1]$.

The reference's $(a, b, c) = (3.4445, -4.7750, 2.0315)$ has slope $p'(0) = 3.4445$: a small value grows about 3.44 times a round: $p(0.001) \approx 0.00344$. Near 1 it does not settle: from $s = 1$, $p(1) = 3.4445 - 4.775 + 2.0315 = 0.701$, and then

$$ 1 \to 0.701 \to 1.114 \to 0.721 \to 1.090 \to 0.696 $$

After five rounds every start of at least 0.01 lies in $[0.682, 1.134]$. The classical cubic $1.5s - 0.5s^3$ converges to 1 from every start in $(0, 1]$, but its slope at 0 is only 1.5. The quintic trades exactness for speed.

## Steepest descent under the spectral norm

The spectral norm $\|\Delta W\|$ is the largest singular value: the most $\Delta W$ stretches any vector. To first order a step changes the loss by $\langle G, \Delta W\rangle = \sum_{ij} G_{ij}\Delta W_{ij}$. Writing $G = \sum_i s_i u_i v_i^\top$,

$$ \langle G, \Delta W\rangle = \sum_i s_i\, u_i^\top \Delta W\, v_i \ge -\eta \sum_i s_i \quad \text{if } \|\Delta W\| \le \eta $$

since each $u_i^\top \Delta W v_i$ is at most $\|\Delta W\|$ in size, and $-\eta UV^\top$ reaches the bound: it is a steepest step of spectral norm $\eta$. For every input, $\|\Delta W x\| \le \eta\,\|x\|$. A gradient step moves the output for a unit input along $v_i$ in proportion to $s_i$: mostly along a few directions.

## One linear layer

For $y = Wx$ fitted to $W^* x$, with inputs of variance $\lambda_j$ along orthonormal directions $v_j$, the loss is $\tfrac12 \sum_j \lambda_j \|(W - W^*)v_j\|^2$; gradient descent multiplies part $j$ by $(1 - \eta\lambda_j)^2$ a step, near 1 where $\lambda_j$ is small.

# In the picture

- The bars, on a log scale: the singular values over the Frobenius norm of the gradient $G$ (cyan), of $\tilde M$ (pink, then kept as ticks), after each round (blue) and of the step.
- The curve chart: $p$ (blue), the cubic (grey, dashed), the diagonal $p(s) = s$, and one value's staircase, round by round.
- The inputs: for each principal direction of the layer's inputs, how far a unit input moves the output under a gradient step (cyan) and Muon's step (blue), both of the same spectral norm (dashed).
- The fourth chart: the linear layer's loss along each direction under three rules, or this network trained.

> Try first "the rounds one at a time", then "the classical cubic instead": small values climb 3.44 times a round with the quintic, 1.5 times with the cubic.

see: svd, polar, steepest, muonkin

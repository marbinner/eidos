# The idea

Each training step hands a layer of a network its gradient: a matrix $G$, a row per output unit, a column per input. An optimiser turns $G$, and the gradients before it, into the matrix the weights step along. Muon keeps the directions of a running average of $G$ and makes every one count about the same; its relatives answer the same question in other ways, and this scene sorts them by what they do to $G$. It helps to know the singular value decomposition (scene svd) and momentum.

# The mathematics

## Three answers for one gradient

Write the gradient by its singular value decomposition,

$$ G = U S V^\top, $$

where the columns of $V$ are input directions, $S$ holds how much each is stretched, $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$, and the columns of $U$ say where each goes. With a learning rate $\eta$ and one gradient, most of the scene's rules step along one of three things:

$$ -\eta\, U S V^\top, \qquad -\eta\, U V^\top, \qquad -\eta \sgn G. $$

Gradient descent keeps $S$, so most of its step falls on the directions with large $\sigma_i$. Muon, without its momentum, sets every $\sigma_i$ to 1 (computed exactly; Muon's own polynomial only pushes them towards 1, scene muon). Of all steps that stretch no vector by more than $\eta$, this one lowers the loss most to first order, when $G$ has full rank (scene steepest). Spectral descent takes that direction with a length growing with $\sum_i \sigma_i$. Sign descent works on entries, not singular values: each entry moves by $\eta$, up or down. That is the steepest step when a step's size is its largest entry, and in general its singular values are not all equal.

## Shampoo without memory is the orthogonal step

Shampoo keeps two sums, $L$ of every $G G^\top$ and $R$ of every $G^\top G$, and steps along

$$ -\eta\, L^{-1/4}\, G\, R^{-1/4}. $$

With one gradient and no damping, $G G^\top = U S^2 U^\top$, so $(G G^\top)^{-1/4} = U S^{-1/2} U^\top$, and likewise on the right. For a square $G$ of full rank the step is $U S^{-1/2} S S^{-1/2} V^\top = U V^\top$: Muon's step with the momentum switched off and orthogonalised exactly.

By hand: for $G = \operatorname{diag}(4, 1)$, $G G^\top = \operatorname{diag}(16, 1)$ and its $-1/4$ power is $\operatorname{diag}(1/2, 1)$. Then $\operatorname{diag}(1/2, 1)\operatorname{diag}(4, 1)\operatorname{diag}(1/2, 1) = \operatorname{diag}(1, 1)$, which is $U V^\top$ here ($U = V = I$).

With memory, if the gradient never changed (and nothing else were added), after $t$ steps $L = t\, G G^\top$ and the step would be $U V^\top / \sqrt{t}$: the same direction, shorter. When each gradient comes from another batch, the sums no longer match the present $G$, and the step is in general not flat.

## Momentum before or after

Muon averages the gradients (momentum), $M \leftarrow \beta M + (1 - \beta) G$, then orthogonalises. Orthogonal-SGDM orthogonalises each $G$, then averages. An average of matrices whose singular values are all 1 does not in general have them all 1, and orthogonalising a noisy gradient first gives its noise directions full size.

## Comparing two steps

Two steps are compared by their angle as lists of entries, $\cos\theta = \langle A, B\rangle / (\|A\|_F\, \|B\|_F)$, and a step's spread by its stable rank $\sum_i \sigma_i^2 / \sigma_1^2$: 16 for $U V^\top$ of a 16 by 16 matrix of full rank, 1 for a matrix of rank 1.

# In the picture

- Each box is a rule: blue ones keep $U$ and $V$, orange ones act on entries, green ones whiten with memory. An arrow says what turns one rule into the next; a box's number is its step's angle from Muon's.
- Grey bars: the gradient's singular values; coloured bars: those of the lit rule's step; each over its largest, on a logarithmic scale.
- The race: a small network's loss under each rule, at the rate a search chose.

> Try first "one gradient, three answers": most boxes read 0°, 72° or 63° from Muon, the three answers above.

see: svd, polar, steepest, muon

# The idea

A network that sorts things into classes ends with one raw score for each class, its logits. An output function makes a prediction of them, and a loss says how wrong it is. The backward sweep starts with the loss's slope with respect to each logit: every gradient further back is this one, carried back by the chain rule. Its size depends on how output function and loss are paired, and one pairing goes nearly silent just when the network is most wrong. You need derivatives, the chain rule, and the natural logarithm $\ln$.

# The mathematics

## Softmax with cross-entropy: probabilities minus the truth

Write the logits $s_1, \ldots, s_K$, and let class $t$ be the right answer. The truth $y$ is 1 for the target and 0 elsewhere. The softmax turns logits into probabilities, positive and adding up to 1:

$$ p_k = \frac{e^{s_k}}{\sum_j e^{s_j}} $$

The cross-entropy is $L = -\sum_k y_k \ln p_k$, here $-\ln p_t$, which is also $-s_t + \ln \sum_j e^{s_j}$. The slope of the second term with respect to $s_k$ is $e^{s_k} / \sum_j e^{s_j} = p_k$, and the first adds $-1$ when $k = t$. So

$$ \frac{\partial L}{\partial s_k} = p_k - y_k $$

These slopes add up to $1 - 1 = 0$, and none is larger than 1 in size. The target's is $p_t - 1$: nearly $-1$, the strongest pull this loss gives, when the target gets almost no probability.

Example: $s = (1, 2, 0, -1)$ with class 1 the target. Then $e^{s} = (2.718, 7.389, 1, 0.368)$, adding up to 11.475, so $p = (0.237, 0.644, 0.087, 0.032)$ and the slopes are $(-0.763, 0.644, 0.087, 0.032)$, which add up to 0.

## A sigmoid with the squared error

A sigmoid $\sigma(s) = 1/(1 + e^{-s})$ on each logit, scored by $L = \tfrac12 \sum_k (\sigma_k - y_k)^2$, gives by the chain rule

$$ \frac{\partial L}{\partial s_k} = (\sigma_k - y_k)\,\sigma_k(1 - \sigma_k) $$

The last factor is the sigmoid's own slope, at most $1/4$, and nearly 0 wherever the sigmoid is flat. At $s_t = -4$ the target's output is $\sigma = 0.018$, badly wrong, yet its slope is only $-0.017$: learning stalls. The slope is small, not 0: it gets going many steps later.

The same sigmoid scored by its own cross-entropy, $-[y \ln \sigma + (1 - y) \ln(1 - \sigma)]$, has slope $\sigma - y$: the flat factor cancels, and at $s_t = -4$ the slope is $-0.98$. The pairing matters, not the sigmoid. A linear output with the squared error, used for numbers, has slope $s - y$: the error.

## Temperature, a softened truth, and the Jacobian

With a temperature $T$, $p = \operatorname{softmax}(s/T)$ and the slope is $(p - y)/T$, never more than $1/T$ in size.

With label smoothing the truth is $1 - \varepsilon$ for the target and $\varepsilon/(K - 1)$ for the others. Then $p = y$ happens at finite logits, where the slope is 0: with the target's logit $\ln((1 - \varepsilon)(K - 1)/\varepsilon)$ above each other one, $\ln 27 = 3.30$ for $\varepsilon = 0.1$ and four classes. With the hard truth $p_t$ never reaches 1, and the slope never stops pushing the logits apart.

The softmax's own Jacobian is $\partial p/\partial s = \operatorname{diag}(p) - p\,p^\top$. Times the loss's slope with respect to each $p_k$, which is $-y_k/p_k$, it gives $p - y$ again.

# In the picture

- The cyan bars at the top are the logits $s_k$ (drag the dots); the orange column is the target.
- Each row is a pairing: output in cyan, truth $y_k$ as an orange tick, slope $\partial L/\partial s_k$ in pink, loss $L$ in orange.
- The charts follow the loss and the pull on the target's logit as that logit moves, a colour for each pairing (red: the sigmoid with the squared error).
- The race takes gradient steps under each pairing; above 0 its gap says the answer is right.

> Try first "confident and wrong: one shouts, one whispers", then "the race from a wrong start".

see: chain, entropy, logistic, backpropmat

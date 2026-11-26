# The idea

How much does learning an outcome tell you? An outcome you were sure of tells you nothing; a rare one tells you a lot. Information theory turns this into a number, in bits, and the entropy of a distribution is how much its outcomes tell you on average: how uncertain it is. Two more numbers measure what it costs to plan for the wrong distribution, and they turn out to be what maximum likelihood makes small. You need probabilities of a few outcomes (scene chance) and the logarithm; the last part uses scene likelihood.

# The mathematics

## Surprise and entropy

A distribution $p$ gives each outcome $x$ a probability $p(x)$, and these add to 1. An outcome of probability $p$ carries a surprise of $\log_2(1/p)$ bits. Why a logarithm: for independent outcomes the probabilities multiply, and the logarithm makes their surprises add. Why base 2: a fair coin's toss, the answer to one even yes/no question, is then 1 bit. The entropy is the average surprise:

$$ H(p) = \sum_x p(x) \log_2 \frac{1}{p(x)} $$

An outcome that never happens adds 0. $H$ is 0 when one outcome is certain, and for $n$ outcomes at most $\log_2 n$, reached when all are equally likely.

Example: a coin that lands heads 9 times in 10. Heads surprises little, $\log_2(10/9) = 0.15$ bits; tails a lot, $\log_2 10 = 3.32$; on average $0.9 \times 0.15 + 0.1 \times 3.32 = 0.47$ bits.

## Questions

Entropy also counts yes/no questions. The best questions for a distribution (a Huffman code) take on average $L$ questions to find the outcome, with

$$ H \le L < H + 1 $$

For $p = (\tfrac12, \tfrac14, \tfrac18, \tfrac18)$ over $\mathrm{A}$, $\mathrm{B}$, $\mathrm{C}$, $\mathrm{D}$, ask whether it is $\mathrm{A}$, then $\mathrm{B}$, then $\mathrm{C}$: the outcomes take 1, 2, 3 and 3 questions, exactly their surprises, so $L = H = \tfrac12 + \tfrac24 + \tfrac38 + \tfrac38 = 1.75$.

## Planning for the wrong distribution

If outcomes follow $p$ but the questions are planned for another distribution $q$, outcome $x$ costs about $\log_2(1/q(x))$ questions. The average is the cross-entropy, and its excess over $H(p)$ is the Kullback–Leibler divergence:

$$ H(p, q) = \sum_x p(x) \log_2 \frac{1}{q(x)}, \qquad D(p \,\|\, q) = H(p, q) - H(p) = \sum_x p(x) \log_2 \frac{p(x)}{q(x)} $$

$D$ is never below 0, and is 0 only when $q = p$. Why: $\ln t \le t - 1$, equal only at $t = 1$, so $-D$ in nats, $\sum p \ln(q/p)$, is at most $\sum (q - p) \le 0$, summing where $p(x) > 0$. Planning for $p$ itself is best.

With $p$ as above and $q$ even, every outcome takes 2 questions: $H(p, q) = 2$ and $D(p \,\|\, q) = 0.25$ bits. $D$ is in general not the same both ways round, and if $q$ calls impossible an outcome that $p$ allows, $D(p \,\|\, q) = \infty$.

## Two variables, and fitting

For two variables $X$ and $Y$, the mutual information $I(X; Y) = H(X) + H(Y) - H(X, Y)$ is how many bits one tells about the other, $H(X, Y)$ being the entropy of the pair. It equals $D$ from the table of pairs to the table they would have if independent, so it is 0 exactly when they are independent.

A sample of $N$ outcomes has shares $\hat p$. The average log-likelihood of a model $q$ per outcome is

$$ \frac{1}{N} \sum_{k=1}^{N} \log_2 q(x_k) = -H(\hat p, q) = -H(\hat p) - D(\hat p \,\|\, q) $$

since each outcome $x$ appears $N \hat p(x)$ times. So maximum likelihood picks the model closest to the sample in $D$; with $q$ free, $q = \hat p$. With $\ln$ in place of $\log_2$ the unit is the nat: 1 nat is $1.44$ bits.

# In the picture

- Green bars: $p$; drag the dots on top. Under each, its surprise; below, each outcome's part $p \log_2(1/p)$, which stack to $H(p)$.
- White: the tree of best questions, and the dot for their average $L$ between $H$ and $H + 1$.
- Purple: $q$. Pink: the extra bits of planning for $q$, which average to $D(p \,\|\, q)$.
- Cyan: a sample's shares $\hat p$; yellow: the average log-likelihood of $q$.

> Try first "the wrong model", then "ruled out".

see: chance, likelihood, logistic, chisquare

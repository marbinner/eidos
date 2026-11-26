# The idea

A coin's chance of heads, $p$, is not known. The likelihood says how well each value of $p$ explains the tosses; Bayesian statistics goes further and describes what is believed about $p$ by a probability distribution over 0 to 1, updated as tosses arrive. That belief can answer questions such as how probable it is that $p$ is above one half. The price is having to state a belief before the data, the prior. You need Bayes' rule, from the scene on chance, and the likelihood.

# The mathematics

## Bayes' rule for a parameter

Bayes' rule, $P(A \mid B) = P(B \mid A)\,P(A) / P(B)$, with $A$ a value of $p$ and $B$ the tosses seen (and densities for $p$), becomes

$$ \text{posterior}(p) \propto \text{prior}(p) \times \text{likelihood}(p) $$

The posterior is the belief after the data; the likelihood is the chance of the tosses seen if $p$ were the true rate. $P(B)$ does not depend on $p$: it only rescales the product so that its area is 1, which is what $\propto$ (in proportion to) leaves out.

## The Beta family

$\mathrm{Beta}(a, b)$ is the distribution on 0 to 1 with density in proportion to $p^{a-1}(1-p)^{b-1}$, and mean $a/(a + b)$; $\mathrm{Beta}(1, 1)$ is flat. For $h$ heads and $t$ tails the likelihood is $p^h (1-p)^t$. Multiplying adds the powers:

$$ \mathrm{Beta}(a, b) \text{ and } h \text{ heads}, t \text{ tails} \ \to\ \mathrm{Beta}(a + h, b + t) $$

So the posterior's mean, $(a + h)/(a + b + n)$ after $n = h + t$ tosses, weighs the prior's mean by $a + b$ and the share of heads $h/n$ by $n$: the prior counts as $a + b$ tosses made earlier. With more and more tosses the data outweigh any prior that leaves the truth some room. A range of values the prior gives no chance at all keeps none, whatever the data. The tosses are taken as independent, with one fixed rate, so only the counts enter, not their order.

Example: a flat prior and 7 heads in 10 make $\mathrm{Beta}(8, 4)$. Its peak is at $7/10 = 0.7$, where the likelihood peaks, since a flat prior leaves the likelihood's shape alone; its mean is $8/12 = 0.67$. It puts 95 % of its probability between 0.39 and 0.89, and 0.89 above one half. A prior as sure of a fair coin as 100 earlier tosses, $\mathrm{Beta}(50, 50)$, moves only to $(50 + 7)/110 = 0.52$.

## Credible and confidence intervals

A 95 % credible interval is a range to which the posterior gives probability 95 %; here the middle one. It says: given this prior and this model, $p$ is in here with probability 95 %. It is a statement about $p$, after these data, and only as good as the prior.

A 95 % confidence interval comes from a recipe made to cover the true $p$ in 95 % of samples, whatever $p$ is (for a share only roughly: with 100 tosses of this coin, Wilson's covers 94 %). It needs no prior, and gives no probability that $p$ is in this one interval.

With a flat prior and 70 heads in 100, both run from 0.60 to 0.78, yet they answer different questions. They can also part. A credible interval makes no promise to cover the truth in 95 % of samples: with the prior that rules out every rate above one half and a true rate of 0.7, after 1000 tosses it runs from 0.496 to 0.500, and no data can bring it to 0.7.

# In the picture

- Purple: the prior; under the chart its middle 95 % and its mean (drag its dots).
- Cyan: the likelihood, scaled to the posterior's height, since only its shape matters.
- Yellow: the posterior; shaded, its middle 95 %, the credible interval.
- Blue: the threshold; the striped area is the posterior's probability above it.
- Green: the true rate, which only decides what the tosses will be.
- Orange: a second observer's prior and posterior.

> Try first "seven heads in ten", then "credible and confidence".

see: chance, likelihood, intervals, mcmc

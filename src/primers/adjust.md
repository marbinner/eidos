# The idea

To measure how much $X$ changes $Y$ from data that were only observed, one compares units that are alike in other variables: one controls, or adjusts, for them. Here that means putting them in the least-squares fit of $Y$ beside $X$ and reading off the coefficient of $X$. Which variables matters: some remove a bias, some add one, some only change the precision. You need regression with several inputs, and chains, forks and colliders.

# The mathematics

## The true effect

Here every variable is its causes, each times the strength of its arrow, plus a noise of its own (mean 0, variance 1, independent of the rest). The total effect of $X$ on $Y$ is how much $Y$ changes when $X$ is set one higher and everything it causes follows. With straight-line effects it is a sum over the paths that follow the arrows from $X$ to $Y$, each the product of its strengths. As the scene opens these are $X \to Y$ and $X \to M \to Y$:

$$ 0.50 + 0.80 \cdot 0.50 = 0.90 $$

## Paths, open and blocked

A path is a way from $X$ to $Y$ along arrows, whichever way they point. A back door is a path that starts with an arrow into $X$, such as $X \gets Z \to Y$. A path is blocked at a held variable where its arrows do not meet, and at a collider (where two of its arrows meet) when neither it nor anything it causes is held. An open back door carries association that is not effect: a bias.

Example: hold nothing. As in the first scene, the open back door through $Z$ adds the product of its two strengths times $\operatorname{Var}(Z) / \operatorname{Var}(X)$, and here $\operatorname{Var}(Z) = 1$. $X$ gets variance $1.5^2 = 2.25$ from $I$, 1 from $Z$ and 1 of its own, so $\operatorname{Var}(X) = 4.25$, and with endless data the estimate is

$$ 0.90 + \frac{1 \cdot 1 \cdot 1}{4.25} \approx 0.90 + 0.24 = 1.14 $$

More data would not remove this error.

## The back-door rule

With straight-line effects, as here, the coefficient of $X$ is the total effect, given endless data, if the set held

1. blocks every back door, and
2. holds nothing that $X$ causes.

Holding $Z$ does both. Holding the mediator $M$ breaks the second part and removes the share of the effect that goes through $M$; holding the collider $C$ breaks it too, and opens $X \to C \gets Y$. The rule is enough whatever the strengths, but not always needed.

The rule takes the graph on trust: it is sure to give the effect only if every arrow is drawn and every common cause of any two variables is in the graph, measured or not. A back door that no set of measured variables blocks cannot be closed by any fit. And looking like a confounder is not enough: in "the ‘M’ shape", $B$ goes with both $X$ and $Y$, but it is a collider on $X \gets U_1 \to B \gets U_2 \to Y$, and holding it opens that path.

## Precision

Once the back doors are blocked, holding a cause of $Y$ only ($P$) takes away part of the spread of $Y$, so the estimate wanders less from sample to sample. Holding a cause of $X$ only ($I$, an instrument) takes away part of the spread of $X$, leaving less to learn from, so it wanders more. Neither moves the estimate on average.

# In the picture

- The graph: a boxed variable is held. Paths: green if they carry the effect, pink if bias, grey if blocked.
- What the fit sees: $Y$ against $X$, each less its least-squares fit on the held variables. The white line's slope is the estimate, the green line's the true effect.
- Below: the estimate from this sample (white) and from 40 others (faint), and a dot for every set that could be held, green if right.

> Try first "hold everything": the estimate falls far below the true effect, and the paths show why.

see: multireg, junctions, dosee

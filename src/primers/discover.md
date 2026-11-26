# The idea

With more than two variables, the question is the whole graph. For straight-line effects and noise that is not bell-shaped, the method called DirectLiNGAM can find it: it finds a variable nothing else causes, takes its effect away from the rest, and repeats. The order this gives says which arrows can exist, and least squares gives their strengths. It builds on "Fingerprints of the noise".

# The mathematics

## The model and its order

Each variable $x_i$ is a sum of straight-line effects of others plus a noise $e_i$ of its own:

$$ x_i = \sum_{j} b_{ij}\, x_j + e_i $$

where $b_{ij}$ is the strength of the arrow from $x_j$ to $x_i$ (0 if there is none). The method assumes no loop of cause and effect, noises independent of each other and not bell-shaped, and nothing unmeasured causing two of the variables. A causal order lists the variables so that every arrow points from earlier to later. Four variables can be ordered in $4! = 24$ ways.

## Finding a root

A root is a variable that nothing among those left causes. To test a candidate, fit every other variable from it by a straight line and measure how much the residuals depend on it (by HSIC, as in "Cause or effect?"). Its score adds these up, in units of what random pairing gives.

A root scores low: it is its own noise, the line takes away exactly its share of each other variable, and what is left is made of the other noises only, independent of it. A candidate that something causes contains a cause's noise as well as its own, and the residual of one of its causes, fitted from it, shares one of these. By the theorem of "Fingerprints of the noise", sums sharing a noise that is not bell-shaped are dependent, so it scores higher.

Example: the chain the scene opens with, every noise of variance 1. $C$ is a root and $A = 0.8\,C + e_A$, so $\operatorname{var} A = 0.64 + 1 = 1.64$.

- Fit $A$ from $C$: the slope is $0.8$, and what is left is $e_A$, independent of $C$.
- Fit $C$ from $A$: the slope is $0.8/1.64$, and what is left is $(C - 0.8\,e_A)/1.64$. Its covariance with $A$ is $(0.8 - 0.8)/1.64 = 0$, but both are made of $e_C$ and $e_A$, so with flat noise they are dependent.

## Taking the root away

Once a root is placed, every variable left is replaced by its residual fitted from the root (written $\tilde A$ and so on). These follow the same arrows among themselves as before: a smaller model of the same kind. Repeat until one is left. Then fit each variable by least squares from all those before it in the order; a fitted strength smaller than the threshold, either sign, counts as no arrow.

## When it goes wrong

- Bell-shaped noise: a residual uncorrelated with its input is then independent of it whichever way the arrow points. Every candidate scores like random pairing, and the order is right about as often as a random one (for the chain, 1 time in 24).
- A hidden common cause: to the method it looks like an arrow between its effects, and the fit can add one.
- Few samples: rough scores, and a wrong candidate can win.

# In the picture

- Left, the true graph: green arrows with their strengths; purple, an unmeasured cause.
- Right, the graph found: cyan roots, numbered in the order placed. Then green: an arrow in the true graph; pink: one not there, or a wrong order; yellow: one the wrong way round.
- The bars: each candidate's score, a piece for each other variable's residual; white dashed, random pairing (1). The shortest bar wins.
- The scatters: residuals against the root and against a candidate not chosen; orange bars span each slice's middle 80 %.

> Try first "a chain, link by link", stepping with next ▶; then "bell-shaped noise".

see: direction, lingam, pc

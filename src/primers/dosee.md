# The idea

There are two ways to ask what $X$ does to $Y$. Seeing: look at the units that happen to have $X = x$, and take the mean of their $Y$. Doing: set $X$ to $x$ for every unit, by hand, and take the mean of $Y$ then. A decision to treat needs the answer of doing, but data that were only observed give the answer of seeing. This scene shows why the two can differ: something else pushes both $X$ and $Y$, or $Y$ is what causes $X$. You need a mean, a least-squares line, and the confounder of the scene before.

# The mathematics

## The rules

Each unit has its own $Z$, $N_X$ and $N_Y$, drawn independently from a bell curve of mean $0$ and standard deviation $1$. From them

$$ X = aZ + 0.6\,N_X \\ Y = bX + cZ + 0.5\,N_Y $$

Each rule is a mechanism: it makes a variable from its own noise and its causes, the variables whose arrows point into it.

## Seeing

Seeing is the distribution of $Y$ among the units with $X = x$, written $P(Y \mid X = x)$. Write $E[\ldots]$ for a mean over endless units. By the rule for $Y$,

$$ E[Y \mid X = x] = b\,x + c\,E[Z \mid X = x] $$

Here the units seen with a high $X$ are mostly those with a high $Z$. Since every effect is a straight line and every draw a bell curve, the mean of $Z$ among units with $X = x$ lies on the least-squares line of $Z$ on $X$, of slope $\operatorname{Cov}(X, Z) / \operatorname{Var}(X) = a / (a^2 + 0.36)$. So the line of seeing has slope $b + c\,a / (a^2 + 0.36)$.

## Doing

Doing sets $X$ to $x$ for every unit, written $\operatorname{do}(X = x)$. It replaces the rule for $X$ by $X = x$: every arrow into $X$ is cut, and nothing else changes, each unit's $Z$ and noise included. Since the mean of $Z$ over all units is 0,

$$ E[Y \mid \operatorname{do}(X = x)] = b\,x + c\,E[Z] = b\,x $$

Its slope, $b$, is the average causal effect. Seeing and doing agree exactly when $c\,a = 0$: when $Z$ does not push on both. And if the arrow points the other way, $Y \to X$, setting $X$ leaves every $Y$ as it was: doing's line is flat, though seeing can still show a trend.

Example, the scene as it opens: $a = 1$, $b = 0.5$, $c = 1$, $x = 1.5$. The mean $Z$ among the units with $X = 1.5$ is $1.5 / 1.36 = 1.10$, so

$$ E[Y \mid X = 1.5] = 0.75 + 1.10 = 1.85, \qquad E[Y \mid \operatorname{do}(X = 1.5)] = 0.75 $$

The scene's 1000 units give $1.84$ (the 76 within $0.25$ of $1.5$) and $0.77$.

## A randomised experiment

Deal each unit's $X$ by chance, as with a die. Then nothing about the unit has a say in its $X$: the arrow $Z \to X$ is cut by the design, and seeing gives doing's answer, up to the luck of the draw. That is why trials randomise. Without one, doing's answer can sometimes still be had from seen data: for instance by holding the right variables fixed, which needs them measured and the arrows known (a later scene).

# In the picture

- Dots: one unit each, $X$ across and $Y$ up, paler the higher its $Z$.
- Pink is seeing: the window round $x$, the units in it (ringed), the mean $Y$ seen at each $x$, and the histogram of their $Y$.
- Green is doing: the mean $Y$ with $X$ set to each $x$ for every unit (slope $b$), and the histogram of every unit's $Y$ then.
- Cyan is $X$ and $x$; purple is $Z$ and its arrows. A cut arrow is dashed and struck through.
- In doing, each unit travels to $X = x$ with its own $Z$ and noise, along a line of slope $b$ (when $X$ causes $Y$).

> Try first "seeing is not doing": the pink and green means at $x = 1.5$ sit far apart, and the travelling units show why.

see: confound, adjust, counterfactual

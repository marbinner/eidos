# The idea

The other road to causal discovery uses only which variables are independent, alone or once others are held, and nothing about the shape of the effects or the noise. So it works where the residual methods cannot, as with straight lines and bell-shaped noise. But several graphs often give the same independences, and then it cannot choose. The scene runs the PC algorithm step by step. It builds on "Chains, forks and colliders".

# The mathematics

## What it assumes

The causes form a graph without loops; every common cause is measured; every independence in the data comes from the shape of the graph, none from strengths that happen to cancel (faithfulness); and the tests are right.

## Independence as a partial correlation

Here the effects are straight lines and the noise bell-shaped, so two variables are independent given held ones exactly when their partial correlation $r$ is 0: the correlation of what is left of each after a straight-line fit on the held ones. With one variable $B$ held,

$$ r_{AC \cdot B} = \frac{r_{AC} - r_{AB}\, r_{CB}}{\sqrt{(1 - r_{AB}^2)(1 - r_{CB}^2)}} $$

where $r_{AC}$ is the plain correlation of $A$ and $C$. A link goes when $r$ is nearer 0 than a threshold. Chance alone gives $|r|$ up to about $1.96/\sqrt{n}$ from $n$ samples, 95 times in 100: $0.06$ for 1000, $0.36$ for 30.

Example, with strengths $0.8$ and noises of variance 1. In the chain $A \to B \to C$, $\operatorname{var} B = 1.64$, $\operatorname{var} C = 0.64 \cdot 1.64 + 1 = 2.0496$ and $\operatorname{cov}(A, C) = 0.64$, so $r_{AC} = 0.64/\sqrt{2.0496} = 0.45$. Hold $B$, and what is left of $C$ is its own noise, unrelated to $A$: $r = 0$, and the link goes. In the collider $A \to B \leftarrow C$, $r_{AC} = 0$, but $r_{AB} = r_{CB} = 0.8/\sqrt{2.28}$, and holding $B$ gives $r = -0.64/1.64 = -0.39$.

## The four steps

1. Join every pair of variables.
2. Remove a link when its two ends are independent, alone or given some of the others; remember the set that parted them.
3. For every $X$ – $Z$ – $Y$ with no link between $X$ and $Y$: if $Z$ was not in the set that parted them, $Z$ is a collider, $X \to Z \leftarrow Y$.
4. Orient the links that the other way round would make a new collider or a loop.

Step 3 works because holding the middle of a chain or fork parts its ends, while holding a collider joins them. So if $Z$ were the middle of a chain or fork, every set that parts $X$ and $Y$ would contain it; if it is a collider, none does.

## How far it can go

Graphs with the same links and the same colliders give exactly the same independences (they are Markov equivalent): no test tells them apart. PC draws an arrow where all of them agree, a plain line where they do not. The chain $A \to B \to C$, the chain the other way and the fork $A \leftarrow B \to C$ are three such graphs, so no arrow can be drawn; the collider is alone with its independences. Without faithfulness it can fail: in "two paths that cancel", $A$ acts on $C$ by $-0.64$ directly and by $0.8 \times 0.8 = 0.64$ through $B$; the two add to 0, and a real link is removed.

# In the picture

- Left, green: the true graph with its strengths. Right, white: what the tests have found, a plain line where they cannot say which way.
- Yellow: the test being done, its link dashed when it goes. Purple rings: the variables held.
- At the end, green: an arrow found as in the truth; red: a wrong link or arrow, or (dashed) a link lost. Below, all the graphs with these independences.
- The strip of every test's $r$, with the threshold (drag its marks) and the grey band of chance.

> Try first "a chain", then "a collider": only the second can be oriented.

see: junctions, direction, discover

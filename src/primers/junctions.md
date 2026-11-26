# The idea

Three variables can be wired by two arrows in three ways, and every longer path through a causal graph is a string of these. This scene shows what becomes of the trend between the ends, $X$ and $Y$, when the middle variable is held fixed: holding can remove a trend, or make one up. You need the correlation $r$ and the confounder of the earlier scene.

# The mathematics

## The three junctions

An arrow $U \to V$ says that $U$ is a direct cause of $V$. Here every variable is its causes, each times the strength of its arrow, plus a noise $\varepsilon$ of its own: a bell curve of spread 1, independent of everything else. A variable with no arrow into it is its noise alone. With strengths $a$ and $b$:

$$ \text{chain } X \to M \to Y\text{:} \quad M = aX + \varepsilon_M, \quad Y = bM + \varepsilon_Y \\ \text{fork } X \gets Z \to Y\text{:} \quad X = aZ + \varepsilon_X, \quad Y = bZ + \varepsilon_Y \\ \text{collider } X \to C \gets Y\text{:} \quad C = aX + bY + \varepsilon_C $$

Holding a variable fixed means comparing only units with the same value of it (in the scene, a narrow band); selecting on it, keeping only the units above a threshold of it.

- Chain: $X$ acts on $Y$ only through $M$. Once $M$ is known, what is left of $Y$ is $\varepsilon_Y$, which has nothing to do with $X$. Holding $M$ takes the trend away, though $X$ really does cause $Y$.
- Fork: $X$ and $Y$ go together only because $Z$ moves both. With $Z$ held, $X$ varies only by $\varepsilon_X$ and $Y$ only by $\varepsilon_Y$, which are independent: the trend goes.
- Collider: $X$ and $Y$ are independent. But among units with the same $C$, $aX + bY$ is the same but for $\varepsilon_C$, so if $a$ and $b$ have the same sign, a high $X$ goes with a low $Y$. Holding or selecting on $C$ makes a trend that no cause produced. So does holding or selecting on anything $C$ causes, such as the interview mark $D$, though here more weakly.

## A worked example

Take the scene's strengths, $a = b = 1.5$. In the fork, $\operatorname{Cov}(X, Y) = ab = 2.25$ (as $\operatorname{Var}(Z) = 1$) and $\operatorname{Var}(X) = \operatorname{Var}(Y) = 1.5^2 + 1 = 3.25$, so over all units

$$ r = \frac{ab}{\sqrt{(a^2 + 1)(b^2 + 1)}} = \frac{2.25}{3.25} = 0.69 $$

and with $Z$ held exactly, $r = 0$. The collider is the other way round: $r = 0$ over all units, and with $C$ held exactly the same number with a minus sign, $r = -0.69$. The scene's band gives $-0.66$. In the chain, $r = 0.78$ over all units and $0$ with $M$ held, although setting $X$ one higher moves the mean of $Y$ by $ab = 2.25$.

## The rule

These add up to one rule, called $d$-separation. A path between $X$ and $Y$ is a route along arrows, whichever way they point. It carries dependence unless it is blocked: at a chain or fork variable that is held, or at a collider when neither it nor anything it causes is held. If every path between $X$ and $Y$ is blocked, they are independent given what is held, whatever the strengths. If one is open, they are dependent, unless the strengths happen to cancel. Both assume the graph is right: no arrow missing, and no hidden common cause left out.

So, to measure what $X$ does to $Y$, hold the middle of a fork, never that of a chain or a collider: what decides is which way the arrows point, not how strongly the middle goes with $X$ and $Y$.

# In the picture

- Each dot is a unit, $X$ across and $Y$ up, coloured by the held variable: blue low, pale pink high.
- Yellow: the band or threshold. The units in it are bright, and the yellow line is their trend.
- White: the trend among all units.
- Green: the arrows and (dashed, when shown) what $X$ really does to $Y$: $ab$ in the chain, nothing otherwise.
- On the right, all four wirings with the same band held.

> Try first "a collider: who gets in": two talents unrelated over all candidates become related among those the school takes.

see: correlation, confound, adjust, pc

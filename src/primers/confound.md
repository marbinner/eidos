# The idea

Two quantities can rise and fall together without either one causing the other. This scene makes data in which a third variable $Z$ pushes on both $X$ and $Y$, and shows how far the trend in a scatter of $Y$ against $X$ can be from the true effect of $X$ on $Y$: too large, hidden, or turned round. You need to know what a covariance and a least-squares line are.

# The mathematics

## Arrows and the rules

$X$ causes $Y$ if changing $X$, and nothing else, changes $Y$. A causal graph draws each such link as an arrow from cause to effect. Here the arrows are $Z \to X$, $X \to Y$ and $Z \to Y$, with strengths $a$, $b$ and $c$, and each unit (a day, a patient) is made by the rules

$$ X = aZ + \sigma_X \varepsilon_X, \qquad Y = bX + cZ + \sigma_Y \varepsilon_Y $$

$Z$ is one of $-2, -1, 0, 1, 2$, each as likely, so $\operatorname{Var}(Z) = 2$. The unit's own variation, $\varepsilon_X$ and $\varepsilon_Y$, is bell-shaped, of mean 0 and spread 1, independent of $Z$ and of each other. Raise $X$ by 1, leave everything else alone, and $Y$ rises by $b$: $b$ is the true effect. A common cause of $X$ and $Y$, like $Z$, is a confounder.

## The line through all units

The least-squares slope of $Y$ on $X$ is $\operatorname{Cov}(X, Y) / \operatorname{Var}(X)$. The rule for $Y$ gives $\operatorname{Cov}(X, Y) = b \operatorname{Var}(X) + c \operatorname{Cov}(X, Z)$, and the rule for $X$ gives $\operatorname{Cov}(X, Z) = a \operatorname{Var}(Z)$. So, with many units, the slope tends to

$$ b + \frac{c\,a \operatorname{Var}(Z)}{\operatorname{Var}(X)}, \qquad \operatorname{Var}(X) = a^2 \operatorname{Var}(Z) + \sigma_X^2 $$

The first term is the effect. The second is what the confounder adds: units with more $X$ tend to differ in $Z$, and $Z$ moves $Y$ by itself. It is 0 only if one of $Z$'s two arrows is missing ($a = 0$ or $c = 0$). When it outweighs $b$ and has the other sign, the trend turns round.

Example: as the scene opens, $a = c = 1$, $b = 0.5$, $\sigma_X = 0.6$. Then $\operatorname{Var}(X) = 2 + 0.36 = 2.36$, and the slope tends to $0.5 + 2 / 2.36 = 0.5 + 0.85 = 1.35$, 2.7 times the true effect.

## Holding $Z$ fixed

Take only the units with one value of $Z$, say $Z = 1$. Among them the rule for $Y$ reads $Y = bX + c + \sigma_Y \varepsilon_Y$: a line of slope $b$, plus variation unrelated to $X$. So the line fitted within each group has slope $b$ but for sampling noise (if $\sigma_X > 0$), and so does the coefficient of $X$ when $Y$ is fitted on $X$ and $Z$ together. This is adjusting for $Z$. It needs $Z$ to have been measured, and it gives the effect here because $Z$ is the only common cause.

Simpson's paradox is a trend that goes one way within every group and the other way when the groups are put together. In "Simpson's reversal", $a = 1$, $b = 0.6$, $c = -2$: the slope through all units tends to $0.6 - 4 / 2.36 = -1.09$, yet within every grade of illness it is $b = 0.6$. Which is right is not in the data but in the arrows: illness decides the dose, so compare like with like. A randomised trial deals the dose by lot: then $a = 0$, and the line through all units finds $b$, but for sampling noise.

# In the picture

- The graph: $Z$, $X$, $Y$ and the arrows' strengths $a$, $b$, $c$.
- Each dot is a unit, $X$ across and $Y$ up, coloured by its $Z$ from deep ($-2$) to pale ($2$).
- Pink: the line through all units. Green: slope $b$, the true effect.
- With "hold $Z$ fixed", a line in each colour, fitted within one value of $Z$.
- Above the scatter: the slope formula, worked out for the present settings.

> Try first "Simpson's reversal": the pink line falls while every coloured line rises.

see: correlation, regress, multireg, dosee

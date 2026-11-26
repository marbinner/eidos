# The idea

When an effect is a straight line of its cause plus noise, the shape of the noise can show which way the arrow points, unless the cause and the noise are both bell-shaped. The scene draws this fingerprint, shows how a bell hides it, and reads it from the points alone. It builds on "Cause or effect?".

# The mathematics

## The model is a shear

Two noises $N_X$ and $N_Y$ are drawn independently. If $X$ causes $Y$,

$$ X = N_X, \qquad Y = bX + N_Y $$

where $b$ is how much $Y$ moves for each unit of $X$. So the data are the noises' cloud after the shear $(x, y) \mapsto (x, y + bx)$, which moves every point straight up by $b$ times its $x$. An upright line's points share one $x$, so it stays upright. Flat noise fills a square, and the data fill a parallelogram with two sides still upright. Had $Y$ caused $X$, two sides would stay level.

## Why a bell hides the arrow

Bell-shaped noise fills a round blob, which the shear turns into an ellipse, with no sides. A bell-shaped cloud is fixed by its middle, its two variances and its covariance, and two models can share these.

Example. $X \to Y$ with $b = 1$ and both noises of variance 1 gives $\operatorname{var} X = 1$, $\operatorname{var} Y = 2$, $\operatorname{cov}(X, Y) = 1$. Now let $Y \to X$: $Y$ of variance 2, and $X = 0.5\,Y + N_X$ with $\operatorname{var} N_X = 0.5$. Then $\operatorname{var} X = 0.25 \cdot 2 + 0.5 = 1$ and $\operatorname{cov}(X, Y) = 0.5 \cdot 2 = 1$: the same numbers, so with bell-shaped noise the same cloud.

Behind this is a theorem: if two sums of the same independent noises are independent, every noise in both is bell-shaped. Fit $X$ from $Y$ by a line: what is left, $X - \hat c Y$, and $Y$ both contain both noises (if $b \ne 0$). The line makes them uncorrelated, but unless both noises are bell-shaped they are dependent; in the scene, the spread of what is left changes along $Y$, while that of $Y - \hat b X$ stays even along $X$.

## The turn

Rescale the cloud to the same spread every way, and the noises' own cloud comes back, each noise at spread 1, but turned. Turn a line through it, and measure how far the cloud's shadow $v$ on the line is from a bell by its kurtosis

$$ \kappa = \frac{\text{mean of } (v - \bar v)^4}{(\operatorname{var} v)^2} - 3 $$

0 for a bell, $-1.2$ for flat noise, 3 for peaked (Laplace) noise. Between the noises' directions the shadow mixes them, and a mix of two independent noises of one shape has $\kappa$ nearer 0 than either's: two flat noises, half and half, give $\kappa = -0.6$. So $\kappa$ is largest in size where the shadow is one noise alone, at two angles a right angle apart. Carried back to the data, these are the sides: an upright one says $X \to Y$, a level one $Y \to X$. If the best turn is no further from normal than bell-shaped noise gets 1 time in 40, it cannot tell.

All this assumes straight-line effects, noises independent of each other (so no hidden common cause), and noise that is not bell-shaped.

# In the picture

- Top left, the noises, $N_X$ across and $N_Y$ up. Along the orange sides only the effect's noise varies, along the cyan ones only the cause's.
- Top right, the data; drag the cyan dot to set $b$. White dashed: the noises' directions, found from the points alone.
- Next to the noises, the two models; a white frame marks the one the points fit. Green: how the data were made.
- Bottom: the fits both ways (purple: twice the spread of what is left); the round cloud and its yellow line (drag its dot); in yellow, the size of $\kappa$ at each angle.

> Try first "flat noise: $X$ causes $Y$"; then "blend in a bell".

see: direction, clt, discover

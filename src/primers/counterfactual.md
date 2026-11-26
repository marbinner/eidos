# The idea

A counterfactual asks about one unit, after the fact. This unit had $X = x$, and its $Y$ came out as $y$: what would its $Y$ have been had $X$ been $x'$? That is not the average $Y$ when $X$ is set to $x'$ for everyone (an intervention), whose answer is the same for every unit. Answering it needs a model of how $Y$ is made, and this scene shows the three steps that use one, and what data can and cannot settle. You need the idea of an intervention and of a curve through a scatter.

# The mathematics

## The model

As the scene opens, each unit's $Y$ is made by

$$ Y = f(X) + N $$

$f$ is a curve: the average $Y$ when $X$ is set to $x$ for everyone. $N$ is the unit's own noise, of mean 0: all its other causes, whatever sets it apart from the average. Nothing that sets $X$ also acts on $Y$ here (as when a trial deals out the doses), so the scatter is centred on the curve $f$. The model also says that a unit's $N$ would have been the same had its $X$ been different.

## The three steps

1. Abduction: read the noise off what was seen, $n = y - f(x)$.
2. Action: set $X$ to $x'$, and change nothing else.
3. Prediction: work out $Y$ again with the same noise, $y' = f(x') + n$.

So the unit stays as far above or below the curve as it was: its path, $y'$ for every $x'$, is the curve moved up by $n$, and it keeps its rank among all units given the same $X$.

Example, "the three steps": the unit was seen at $x = 1.20$ with $y = 3.72$, where the curve is at $f(1.20) = 3.22$. So $n = 3.72 - 3.22 = 0.50$. At $x' = 4.40$ the curve is at $4.33$, so

$$ y' = f(x') + n = 4.33 + 0.50 = 4.83 $$

Here the unit's own change, $y' - y$, equals the average change, $f(x') - f(x)$: with added noise, every unit responds alike.

## Noise that multiplies

If instead $Y = f(X)\,M$, with $M$ a positive factor of the unit's own and on average 1, steps 1 and 3 become

$$ m = \frac{y}{f(x)}, \qquad y' = m\,f(x') $$

Now a unit's own change, $y' - y = m\,(f(x') - f(x))$, is its factor $m$ times the average change. In "noise that multiplies", $m = 3.66 / 3.05 = 1.200$ and $y' = 1.200 \times 4.35 = 5.22$: a change of $+1.56$, against the average $4.35 - 3.05 = 1.30$. The average effect answers for the population; the counterfactual, for this unit.

## What the data cannot settle

Each unit is seen at one $X$ only, so a model that draws each unit's noise afresh at every value of $X$ gives exactly the same data, under every intervention, as one that keeps it. Yet their counterfactuals differ. With fresh noise, what was seen says nothing about the noise at $x'$, and $y'$ could be anything the population has there: in the first example, between 3.42 and 5.23 with 9 chances in 10. A counterfactual is only as sure as the model's account of how a unit's noise carries over to another $X$, and data that show each unit at one $X$ cannot check that account.

# In the picture

- White dots: the units, each seen at one $X$. Green: the curve $f$.
- Orange: this unit as seen, $(x, y)$, and where it would have been, $(x', y')$; dotted, its $y'$ for every $x'$.
- Purple: its noise $n$, from the curve to the unit, and the same $n$ at $x'$.
- Cyan: $x$, and the $x'$ asked about (drag it); the arrow is $X$ being set.
- The strips at the right: everyone's $Y$ with $X$ set to $x$, and to $x'$, with the average in green.

> Try first "the three steps", and watch the noise carried from $x$ to $x'$.

see: regress, dosee, direction

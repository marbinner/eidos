# The idea

An outcome often depends on several inputs that move together, and fitting one input at a time then mixes their effects. Least squares with two inputs fits a plane instead, and each of its slopes is how the plane rises along one input, the other held fixed. The scene shows what that means, and what inputs that move together do to it. You need the regression line (scenes correlation and regress).

# The mathematics

## The plane

Each case $i$ has inputs $x_{1i}$ and $x_{2i}$, on the floor, and an output $y_i$, upwards. The fitted plane is

$$ \hat y = b_0 + b_1 x_1 + b_2 x_2 $$

with $b_0, b_1, b_2$ chosen to make the sum of the squared residuals $y_i - \hat y_i$ as small as it can be. The coefficient $b_1$ is the slope of the plane along $x_1$: how much $\hat y$ rises for one more unit of $x_1$ with $x_2$ held fixed. The scene makes its data as

$$ y = \beta_1 x_1 + \beta_2 x_2 + \gamma\,(x_1^2 - 1) + \sigma\varepsilon $$

with $\varepsilon$ a new standard normal draw for each case; $b_1$ and $b_2$ estimate $\beta_1$ and $\beta_2$.

## Held fixed, and left out

The line of $y$ on $x_1$ alone has in general another slope, tied to $b_1$ exactly:

$$ \text{slope of } y \text{ on } x_1 \text{ alone} = b_1 + b_2 \times (\text{slope of } x_2 \text{ on } x_1) $$

Why: $y$ is the plane plus residuals, and slopes on $x_1$ add. The term $b_1 x_1$ gives $b_1$, the term $b_2 x_2$ gives $b_2$ times the slope of $x_2$ on $x_1$, and the residuals give 0, since least squares leaves them uncorrelated with each input. So the two slopes agree only when $x_2$ does not rise with $x_1$ in the sample, or $b_2 = 0$. Leaving out an input that matters, and moves with the one kept, changes the answer: confounding.

Example: four cases $(x_1, x_2, y)$: $(0, 0, 0)$, $(0, 1, 2)$, $(2, 1, 0)$, $(2, 2, 2)$. All lie on the plane $y = -x_1 + 2x_2$, so $b_1 = -1$ and $b_2 = 2$. Hold $x_2$ at 1: from the second case to the third, $x_1$ rises by 2 and $y$ falls by 2, a slope of $-1$. Alone, $x_1$ seems to do nothing: $y$ averages 1 at both $x_1 = 0$ and $x_1 = 2$. And $x_2$ rises $0.5$ per unit of $x_1$ (it averages $0.5$, then $1.5$): indeed $0 = -1 + 2 \times 0.5$.

Another picture of $b_1$, the added-variable chart: take from $y$, and from $x_1$, all that a straight line in $x_2$ predicts. The slope of what is left of $y$ against what is left of $x_1$ is exactly $b_1$.

## Inputs that move together

If the inputs have correlation $r$ in the sample, what is left of $x_1$ once $x_2$ is known has only $\sqrt{1 - r^2}$ of the spread of $x_1$. A slope read off a narrower spread is less sure, so the standard error of $b_1$ (how much it varies from sample to sample) is larger by the factor

$$ \frac{1}{\sqrt{1 - r^2}} $$

than with unrelated inputs of the same spreads and the same noise. At $r = 0.985$ it is $5.8$: the points lie near a line on the floor, and the data say little about how to share the effect between $b_1$ and $b_2$.

## How well it fits

$R^2$, the share of the spread of $y$ about its mean that the plane accounts for, never falls when an input is added: the fit could give the new one coefficient 0. Adjusted $R^2$ charges for each input, so it can fall.

# In the picture

- Cyan: the cases, and their shadows on the floor. Yellow: the fitted plane and its slopes; orange: the line on one input alone.
- Pink: the residuals. Plotted against $\hat y$ they show only noise when the plane is the right shape.
- White: two points to drag. Lift one by 1 and the plane under it rises by its leverage $h$, between $1/n$ and 1.
- Green: the truth. Grey ellipse: where new samples' coefficients are expected.

> Try first "inputs that move together": the slope alone and $b_1$ have opposite signs.

see: correlation, regress, confound, modelselect

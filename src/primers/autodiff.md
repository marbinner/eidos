# The idea

A computer finds a derivative neither by algebra on paper nor only by nudging the input. It writes the formula as a chain of small steps (a sum, a product, a sine) whose slopes it knows, and combines them by the chain rule, a step at a time. Drawn as a graph, each box is a step and each arrow a value handed on. Swept backwards from the output, this gives the slope with respect to every input at once: run on a network's loss, it is backpropagation. You need the derivative as a slope, and the chain rule.

# The mathematics

## Boxes, arrows and their slopes

Take $f = x\,y + \sin x$. Its steps are $u = x\,y$, $v = \sin x$ and $f = u + v$. Each arrow into a box has a slope of its own: the derivative of that box with respect to the box the arrow comes from, its other arguments held still.

$$ \frac{\partial u}{\partial x} = y, \quad \frac{\partial u}{\partial y} = x, \quad \frac{\partial v}{\partial x} = \cos x, \quad \frac{\partial f}{\partial u} = \frac{\partial f}{\partial v} = 1 $$

For a sum it is 1, for a product the other factor, for a sine the cosine. The forward sweep works out the values, and these slopes with them.

## The backward sweep

Write $\bar u$ for the slope of the output with respect to the box $u$: how much $f$ changes per unit change of $u$. At the output, $\bar f = 1$. A change in a box reaches $f$ only through the boxes it feeds, so by the chain rule

$$ \bar u = \sum_{\text{arrows } u \to w} \bar w \, \frac{\partial w}{\partial u} $$

Working back from $f$, every $\bar w$ on the right is known by the time it is needed. A box used twice, as $x$ is here, has two arrows leaving it: two roads to $f$, and their shares add.

Example, at $x = 2$ and $y = 1.5$: $\bar u = \bar v = 1$, then $\bar y = \bar u \, x = 2$, and

$$ \bar x = \bar u \, y + \bar v \cos x = 1.5 + \cos 2 \approx 1.5 - 0.416 = 1.084 $$

which is $\partial f/\partial x = y + \cos x$. For $f = x \sin x$ the two roads from $x$ give $x \cos x + \sin x$: the product rule.

## Forward mode

The other way round: choose one input, say $x$, and carry with every box how fast it changes as $x$ moves, written $\dot u$. Start with $\dot x = 1$ and 0 for every other input; then

$$ \dot w = \sum_{\text{arrows } u \to w} \frac{\partial w}{\partial u} \, \dot u $$

At the output, $\dot f = \partial f/\partial x$: one slope for each sweep. With $n$ inputs and one output, forward mode takes $n$ sweeps, the backward way two (one forwards for the values, one back). A network's loss is one output of millions of inputs, its weights, which is why networks are differentiated backwards.

## Differences

A slope can also be estimated from two values:

$$ \frac{f(x + h) - f(x - h)}{2h} = f'(x) + \frac{f'''(x)}{6}\,h^2 + \cdots $$

With too large an $h$, the $h^2$ term is the error. With too small an $h$, rounding is: a computer keeps about 16 significant digits, so each value of $f$ is off by about $10^{-16}$ of its size, which dividing by $2h$ makes roughly $10^{-16}/h$ when $f$ is near 1. Between the two lies a valley; at its floor about ten digits are right. The sweeps are off only by the rounding in their own few steps.

# In the picture

- Each box is a step, the inputs on the left (at the top on a tall screen); the dot in an input's box sets it.
- Cyan numbers are values, pink ones the slopes $\bar u$, blue ones the tangents $\dot u$ of forward mode; the number on an arrow is the arrow's slope.
- The lit box is the one being worked out, with its sum written under the graph.
- The chart is the difference's error against $h$, its two causes dashed.
- At the corner of $\max(0, z)$ there is no slope: the sweep puts 0 there, and the difference gives the average of the two sides.

> Try first "forwards for the value, backwards for every slope", then "a variable used twice: its roads add".

see: chain, calc1, backprop1

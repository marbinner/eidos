# The idea

This scene shows three ideas of calculus on one curve $y = f(x)$. The derivative is the slope of the curve at a point. The integral is the area between the curve and the $x$ axis. A Taylor polynomial copies the curve near a point, built from its slopes. Each is reached through approximations: secant lines, sums of thin rectangles, polynomials of rising order. You need only functions and their graphs.

# The mathematics

## The derivative

A straight line through two points of the curve, $h$ apart across, is a secant. Its slope is the rise divided by $h$:

$$ \frac{f(x_0 + h) - f(x_0)}{h} $$

Shrink $h$ and the second point slides towards the first. If the slope settles on a number as $h$ goes to 0, that number is the derivative at $x_0$, the slope of the tangent there:

$$ f'(x_0) = \lim_{h \to 0} \frac{f(x_0 + h) - f(x_0)}{h} $$

Example: $f(x) = x^2$ at $x_0 = 1$. The secant's slope is $((1 + h)^2 - 1)/h = (2h + h^2)/h = 2 + h$: $3$ at $h = 1$, $2.1$ at $h = 0.1$, $2.01$ at $h = 0.01$. It settles on $f'(1) = 2$, and the error, $h$, falls tenfold with each tenfold cut in $h$. At any $x_0$ the same steps give $2x_0 + h$: the slope of $x^2$ is $2x$.

Not every curve has a slope everywhere. At the sharp point of $\sqrt{|x|}$ the secant's slope is $\sqrt{h}/h = 1/\sqrt{h}$ (for $h > 0$), which grows without limit.

## The integral

The integral $\int_a^b f(x)\,\mathrm{d}x$ is the area between the curve and the $x$ axis from $x = a$ to $x = b$, with area below the axis counted negative. To find it, cut the stretch from $a$ to $b$ into strips of width $\Delta x$, and put on each a rectangle as tall as the curve at one place $x_i$ in the strip:

$$ \int_a^b f(x)\,\mathrm{d}x \approx \sum_i f(x_i)\,\Delta x $$

As the strips get thinner the sum closes in on the area. With the height taken at the strip's left side, the error typically shrinks in step with $\Delta x$: twice the rectangles, half the error. Taken at the middle, a rectangle overshoots on one half of its strip about as much as it falls short on the other, and for a smooth curve the error shrinks like $\Delta x^2$: twice the rectangles, a quarter of the error.

## Taylor polynomials

Near $x_0$ a smooth curve looks like its tangent line, $f(x_0) + f'(x_0)\,h$, with $h = x - x_0$. Higher powers of $h$ copy it better:

$$ f(x_0 + h) \approx \sum_{k=0}^{n} \frac{f^{(k)}(x_0)}{k!}\, h^k $$

Here $f^{(k)}$ is the derivative taken $k$ times (the slope of the slope, and so on; $f^{(0)}$ is $f$ itself), $k! = 1 \cdot 2 \cdots k$ with $0! = 1$, and $n$ is the order. Each coefficient is chosen so that the polynomial and $f$ have the same value and the same first $n$ derivatives at $x_0$.

A higher order need not reach further. For $1/(1 + x^2)$ at $x_0 = 0$ the polynomials are $1 - x^2 + x^4 - \cdots$, the sum $1 + r + r^2 + \cdots = 1/(1 - r)$ with $r = -x^2$, which adds up only while $|r| < 1$. So no order gets past $|x| = 1$. Check at $x = 0.5$: order 4 gives $1 - 0.25 + 0.0625 = 0.8125$, against $1/1.25 = 0.8$.

# In the picture

- The cyan curve is $y = f(x)$.
- Derivative: the yellow point is $x_0$ and the yellow line its tangent; the pink line is the secant through two points $h$ apart, its rise marked. "plot $f'(x)$" draws the slope at every $x$, faint yellow.
- Integral: blue rectangles stand above the axis and count plus, pink ones below and count minus; $a$ and $b$ are the white ends.
- Taylor: the orange curve is the polynomial at the orange point $x_0$; the shading is the gap.

> Try first "the secant becomes the tangent", then "where Taylor gives up".

see: graph, surface, fourier

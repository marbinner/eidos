# The idea

Take the area between a curve $f$ and the axis from a fixed start $a$ to an end $x$, and let the end move: the area becomes a function of $x$. The fundamental theorem of calculus says that its slope is the height of the curve at $x$, and that is why areas are found by undoing derivatives. You need a derivative as the slope a secant settles on, and an area as a sum of thin rectangles, both from "Derivative, integral, Taylor".

# The mathematics

## The area so far

Call the area from $a$ to $x$

$$ A(x) = \int_a^x f(t)\,\mathrm{d}t $$

Here $t$ runs along the axis to $x$. Area below the axis counts negative; when $x$ is left of $a$, every piece counts with the opposite sign. And $A(a) = 0$.

## Its slope is the height

Move the end from $x$ to $x + h$. The area gains a thin strip, the area from $x$ to $x + h$:

$$ A(x + h) - A(x) = \int_x^{x+h} f(t)\,\mathrm{d}t $$

Where $f$ is continuous, the strip lies between two rectangles $h$ wide, as tall as its lowest and its highest point. So the secant's slope, $(A(x + h) - A(x))/h$, is the strip's mean height, between the lowest and highest values of $f$ on it. As $h$ shrinks, both close in on $f(x)$, and so does the slope:

$$ A'(x) = \lim_{h \to 0} \frac{A(x + h) - A(x)}{h} = f(x) $$

Example: $f(t) = t^2$, $x = 1$, $h = 0.1$. The strip's area is $(1.1^3 - 1^3)/3 = 0.11033$, so its mean height is $1.1033$: between $f(1) = 1$ and $f(1.1) = 1.21$, and near $f(1)$.

So:

- $A$ climbs where $f > 0$ and falls where $f < 0$. Its tops and bottoms are where $f$ crosses the axis, and it is steepest where $f$ is largest.
- Moving the start $a$ adds the same amount to $A(x)$ for every $x$, the area from the new start to the old: the whole curve shifts up or down, and no slope changes.
- Where $f$ jumps at $x$, a thin strip to the right of $x$ has the height just right of the jump, and one to the left the height just left of it: $A$ has two slopes there, a corner, and no tangent. It does not jump, though: a strip's area is at most $h$ times the largest size of $f$ on it, which shrinks with $h$ when $f$ is bounded.

## Undoing a derivative

Let $F$ be any function with slope $f$ ($F' = f$) on a stretch where $f$ is continuous: an antiderivative. Then $F - A$ has slope $f - f = 0$ there, so it is a constant. At $x = a$, where $A$ is 0, that constant is $F(a)$. So

$$ \int_a^b f(t)\,\mathrm{d}t = F(b) - F(a) $$

Example: $F(x) = -\cos x$ has slope $\sin x$. The area under $\sin$ from $0.5$ to $2.5$ is $-\cos 2.5 + \cos 0.5 = 0.8011 + 0.8776 = 1.6787$, as a fine sum of thin strips confirms.

Two cautions. $F$ must have slope $f$ all the way from $a$ to $b$: $\ln|x|$ has slope $1/x$ on each side of 0, but $1/x$ has no area across 0. And the bell $e^{-x^2/2}$ has no $F$ made of the usual functions. Its $A$ is a new function: as $a$ moves far to the left, $A/\sqrt{2\pi}$ becomes the normal distribution's cumulative curve.

# In the picture

- Upper chart: the cyan curve is $f$, shaded from the start $a$ (white) to the end $x$ (yellow), blue where it adds to $A$ and pink where it takes away; the orange strip runs from $x$ to $x + h$.
- Lower chart: the green curve is $A$, with its yellow tangent at $x$ and the orange secant, whose rise is the strip's area. Purple dashes, when drawn, are $F$.
- Beside the charts: the numbers, $F(x) - F(a)$ among them where $F$ is known, and the strip stretched in a box.

> Try first "the area's slope is the height", then "a jump in $f$ is a corner in $A$".

see: calc1, distributions, changevar

# The idea

An integral over a region adds up $f$ times small pieces of area. Often the region is easier to describe in other coordinates: a disc of radius $R$ is a plain rectangle in polar coordinates, $0 \le r \le R$ and $0 \le t \le 2\pi$. Changing variables means integrating over that rectangle instead. The catch: the map from the rectangle to the region stretches some pieces and shrinks others, and a factor must make up for it. You need integrals as sums of small pieces, and the Jacobian of a map of the plane.

# The mathematics

## Cells and where they land

Write the map as $(x, y) = T(u, v)$. Cut the rectangle of $(u, v)$ into cells, each $\Delta u$ by $\Delta v$. $T$ carries each cell onto a small patch of the region; if $T$ is one to one, the patches cover the region without overlapping. So the integral of $f$ over the region is close to

$$ \sum f(x_i, y_i) \cdot (\text{area of patch } i) $$

where $(x_i, y_i)$ is a point of patch $i$; for $f$ continuous, it settles on the integral as the cells shrink.

## The factor

Up close, $T$ is nearly linear: $T(p + h) \approx T(p) + J h$, where $J$ is the Jacobian, the matrix of $T$'s partial derivatives. Its columns say where small steps along $u$ and along $v$ go. So the cell's sides land nearly as the arrows $J(\Delta u, 0)$ and $J(0, \Delta v)$, and the patch is nearly the parallelogram they span. A linear map scales every area by the size of its determinant, so

$$ \text{area of the patch} \approx |\det J|\,\Delta u\,\Delta v $$

The absolute value is there because an area is never negative: where $\det J < 0$ the map turns the cell over, and it still has an area. Put this into the sum and let the cells shrink:

$$ \iint_{\text{region}} f\,dx\,dy = \iint_{\text{rectangle}} f(T(u, v))\,|\det J|\,du\,dv $$

This holds for $f$ continuous and $T$ smooth and one to one inside the rectangle. On its edge $T$ may fail to be: in polar coordinates the whole edge $r = 0$ lands on the point $0$. But an edge has no area, so nothing is lost. Leave the factor out and the sum still settles down, but to $\iint f(T(u, v))\,du\,dv$, in general a different number. On a line the same rule reads $dx = |g'(u)|\,du$ for $x = g(u)$.

## Polar coordinates

For $T(r, t) = (r\cos t, r\sin t)$,

$$ J = \begin{bmatrix} \cos t & -r\sin t \\ \sin t & r\cos t \end{bmatrix}, \qquad \det J = r\cos^2 t + r\sin^2 t = r $$

A cell lands as a piece of a ring, $\Delta r$ wide and $r\,\Delta t$ long round at its middle: cells further out are larger, in proportion to $r$. The disc of radius $R$ has the area

$$ \int_0^{2\pi}\!\!\int_0^R r\,dr\,dt = 2\pi \cdot \frac{R^2}{2} = \pi R^2 $$

Example: with $R = 3$ and $8 \times 8$ cells, $\Delta r = 0.375$ and $\Delta t = \pi/4$, so $\Delta r\,\Delta t = 0.2945$. A cell of the outer ring, with $r = 2.8125$ at its middle, has the area $2.8125 \times 0.2945 = 0.8283$; one of the innermost, with $r = 0.1875$, has $0.0552$, a fifteenth as much.

The factor also lets the bell be integrated by hand. Since $-e^{-r^2}/2$ has the slope $e^{-r^2} r$,

$$ \iint e^{-(x^2 + y^2)}\,dx\,dy = \int_0^{2\pi}\!\!\int_0^\infty e^{-r^2}\,r\,dr\,dt = 2\pi \cdot \frac{1}{2} = \pi $$

over the whole plane. And since $e^{-(x^2 + y^2)} = e^{-x^2} e^{-y^2}$, the same integral is $(\int e^{-x^2}\,dx)^2$, so $\int e^{-x^2}\,dx = \sqrt{\pi}$ over the whole line.

# In the picture

- Left, the rectangle of $(r, t)$, or of $(u, v)$, cut into $n \times n$ cells; right, the region, each cell where $T$ puts it.
- A cell's tint is how much $T$ stretches it: the area of its image over $\Delta u\,\Delta v$.
- Yellow: the picked cell and its image. Pink and green: its sides and where $J$ takes them. Cyan, dashed: the parallelogram those span, of area $|\det J|\,\Delta u\,\Delta v$.
- Under the planes, the sum three ways, with the true areas (yellow), with the factor (cyan) and without it (red), against the exact integral (white), charted as $n$ grows.

> Try first "polar cells are wider further out", then "forget the factor, and get it wrong".

see: jacobian, det, ftc, gauss2

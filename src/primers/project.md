# The idea

A projection is a matrix that does nothing more the second time. The ones here move every point onto a line, like a shadow, and the shadow of a shadow is the shadow itself. From a projection comes a mirror, a matrix that undoes itself. Projections are how least squares finds nearest points, and mirrors are how computers usually take a matrix apart into $QR$. You need vectors, dot products, matrices as maps, and eigenvalues.

# The mathematics

## Nothing more the second time

A projection is a matrix $P$ with

$$ P^2 = P $$

Split any $x$ into its shadow $Px$ and the leftover $x - Px$:

$$ x = Px + (x - Px) $$

$P$ keeps the shadow, $P(Px) = P^2 x = Px$, and kills the leftover, since $P(x - Px) = Px - P^2 x = 0$. So its eigenvalues are $1$ on what it keeps and $0$ on what it kills, and its trace counts the independent directions it keeps. $I - P$ is a projection too, $(I - P)^2 = I - 2P + P^2 = I - P$: it keeps the leftover and kills the shadow.

## Onto a line, along a direction

Take the line through $0$ along $u$, and a direction $w$ not along it. To project $x$, slide it along $w$ to the line, at $cu$. With $n$ at right angles to $w$, the slide does not change the dot product with $n$, so $n \cdot x = c\, n \cdot u$, and

$$ Px = \frac{n \cdot x}{n \cdot u}\, u, \qquad P = \frac{u n^\top}{n \cdot u} $$

Then $Pu = u$ and $Pw = 0$. If $w$ lies along the line, $n \cdot u = 0$: a point off the line never reaches it, and there is no projection.

When $w$ is at right angles to the line, $n$ is along $u$ and $P = uu^\top / (u^\top u)$, the orthogonal projection. It is symmetric, and the leftover meets the line at a right angle, which makes $Px$ the nearest point of the line to $x$: for any point $q$ of the line, by Pythagoras,

$$ |x - q|^2 = |x - Px|^2 + |Px - q|^2 $$

Example: $u = (2, 1)$ gives

$$ P = \begin{bmatrix} 0.8 & 0.4 \\ 0.4 & 0.2 \end{bmatrix} $$

and $x = (1, 2.5)$ has shadow $(1.8, 0.9)$ and leftover $(-0.8, 1.6)$, whose dot product with $u$ is $-1.6 + 1.6 = 0$. Its length, $1.79$, is the distance from $x$ to the line.

When $w$ slants at the angle $\theta$ to the line, $P^2 = P$ still, but $P$ is not symmetric, and it can lengthen a vector up to $1/\sin\theta$ times.

## Mirrors

The mirror made from $P$ keeps the shadow and reverses the leftover:

$$ H = 2P - I, \qquad Hx = Px - (x - Px) $$

It undoes itself, $H^2 = 4P^2 - 4P + I = I$, and its eigenvalues are $1$ on the line and $-1$ along $w$. For the orthogonal projection, $H$ keeps every length, and with $n$ the unit vector at right angles to the line it is $H = I - 2nn^\top$, a Householder reflection. To send $a$ to a different $b$ of the same length, take $n$ along $a - b$. With $u = (2, 1)$, $H$ sends $(1.5, 2)$ to $(2.5, 0)$, onto the $x$ axis: the step $QR$ is usually built from.

## Two mirrors make a turn

The mirror in the line at the angle $\alpha_1$ sends the direction at the angle $\varphi$ to $2\alpha_1 - \varphi$. A mirror at $\alpha_2$ then sends that to $2\alpha_2 - (2\alpha_1 - \varphi) = \varphi + 2(\alpha_2 - \alpha_1)$: every direction turns by twice the angle between the mirrors. Mirrors at $0.2$ and $0.5$ radians make a turn by $0.6$.

# In the picture

- Orange: the line $P$ keeps, with the knob $u$. Purple, dashed: the line of $w$, sent to $0$.
- $x$ is white, its shadow $Px$ orange, the leftover red and $Hx$ blue.
- The yellow letter $F$ has its shadow $PF$ and its blue mirror image $HF$.
- With two mirrors: $H_1 F$ in blue, and $H_2 H_1 F$, the letter turned.

> Try first "the shadow of the shadow": $P$ acts, then acts again, and the second time nothing moves.

see: linmap2, eigen, lsq, gram

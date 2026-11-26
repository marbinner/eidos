# The idea

An arrow in space (a vector) is three numbers: how far it goes along $x$, along $y$ and along $z$. Two arrows $a$ and $b$ can be multiplied in two different ways. The dot product $a \cdot b$ is a number that says how far the two point the same way. The cross product $a \times b$ is a new arrow, at right angles to both, whose length is the area they span. Nothing more is needed than Pythagoras' theorem and the sine and cosine of an angle.

# The mathematics

## Length and angle

Write $a = (a_1, a_2, a_3)$. By Pythagoras' theorem, used twice, its length is

$$ |a| = \sqrt{a_1^2 + a_2^2 + a_3^2} $$

The angle $\theta$ between $a$ and $b$ is measured where they start, from $0^\circ$ (the same way) to $180^\circ$ (opposite ways).

## The dot product

Multiply the arrows' numbers pair by pair, and add:

$$ a \cdot b = a_1 b_1 + a_2 b_2 + a_3 b_3 $$

The same number comes out of the geometry:

$$ a \cdot b = |a|\,|b| \cos\theta $$

Why: drop $b$'s tip at a right angle onto the line of $a$. It lands $|b| \cos\theta$ along that line; call that $b$'s shadow. The dot product is the shadow's length times the length of $a$. So:

- At a right angle the shadow is a single point, and $a \cdot b = 0$.
- Past a right angle the shadow falls behind the start and counts as negative: $a \cdot b < 0$.
- Dividing by both lengths leaves only the angle: $\cos\theta = a \cdot b / (|a|\,|b|)$, the cosine similarity.

Example: $a = (2, 0, 0)$ and $b = (1, 1, 0)$ give $a \cdot b = 2 \cdot 1 + 0 \cdot 1 + 0 \cdot 0 = 2$. Check: $|a| = 2$, $|b| = \sqrt{2}$ and $\theta = 45^\circ$, and $2 \cdot \sqrt{2} \cdot \cos 45^\circ = 2$.

## The cross product

$$ a \times b = (a_2 b_3 - a_3 b_2,\ a_3 b_1 - a_1 b_3,\ a_1 b_2 - a_2 b_1) $$

It is built to stand at right angles to both arrows: take its dot product with $a$, or with $b$, and every term cancels. Its length is

$$ |a \times b| = |a|\,|b| \sin\theta $$

which is base times height of the parallelogram with sides $a$ and $b$: its area. Of the two directions at right angles to both, it takes the one the right-hand rule gives: curl the fingers of the right hand the way $a$ turns towards $b$, and the thumb points along $a \times b$. Swapping the arrows therefore turns the result round, $b \times a = -(a \times b)$, and arrows in line, which span no area, give the arrow of length 0.

Example: the same $a$ and $b$ give $a \times b = (0, 0, 2 \cdot 1 - 0 \cdot 1) = (0, 0, 2)$: along $z$, at right angles to both, of length 2, the area of a parallelogram of base 2 and height 1.

# In the picture

- The green arrow is $a$ and the orange one is $b$. Drag their tips.
- The yellow piece on the dashed line of $a$ is $b$'s shadow. It turns pink when it falls behind the start, where the dot product is negative.
- The purple parallelogram is the area, and the purple arrow standing on it is $a \times b$.
- The sums written over the picture work out each product both ways, from the three pairs of numbers and from the lengths and the angle, with the numbers as they are.

> Try first "at right angles": the shadow shrinks to a point and the area is as large as these lengths allow. Then "in line", where it is the other way round.

see: span, det, lsq

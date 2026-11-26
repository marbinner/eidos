# The idea

Take two or three arrows that start at $0$. Walks along them, some number of steps along each, reach some points and perhaps not others. The span of the arrows is everything such walks can reach: a line, a flat plane through $0$, or all of space (or $0$ alone, if every arrow has length 0). This scene shows which it is, and when an arrow adds nothing new. You need only arrows as lists of numbers, as in the primer on dot and cross products.

# The mathematics

## Walks along arrows

Call the arrows $v_1$, $v_2$, $v_3$. Multiplying an arrow by a number $c$ multiplies each of its numbers by $c$; a negative $c$ turns it backwards. Adding two arrows adds their numbers pair by pair, which puts the second on the tip of the first. A linear combination of the arrows is

$$ c_1 v_1 + c_2 v_2 + c_3 v_3 $$

for some numbers $c_1, c_2, c_3$: a walk of $c_1$ steps along $v_1$, then $c_2$ along $v_2$, then $c_3$ along $v_3$. The span is every point where such a walk can end. It always contains $0$ (all the numbers $0$).

Example: the scene opens with $v_1 = (1.75, 0.25, 0.5)$ and $v_2 = (0.25, 1.5, 0.75)$. The walk with $c_1 = 1$ and $c_2 = 0.5$ ends at

$$ b = v_1 + 0.5\,v_2 = (1.875,\ 1,\ 0.875) $$

## Line, plane or space

One arrow, not of length 0, spans a line: all its multiples. Two arrows not on one line span a plane through $0$. A third arrow that lies in that plane adds nothing: every walk still ends in the plane. One that points out of the plane lets walks leave it, and with large enough numbers they reach every point of space.

## Independent and dependent

The arrows are dependent if one of them is a linear combination of the others (an arrow of length 0 counts, being $0$ times any other); otherwise they are independent. For example, $v_3 = -0.5\,v_1 + v_2 = (-0.625, 1.375, 0.5)$ depends on $v_1$ and $v_2$. Then:

- $v_3$ adds nothing to the span: any step along it can be replaced by the walk it equals.
- Every point of the span is reached by many walks. A step $t$ along $v_3$ is undone by $0.5t$ along $v_1$ and $-t$ along $v_2$, so for every number $t$

$$ b = (1 + 0.5t)\,v_1 + (0.5 - t)\,v_2 + t\,v_3 $$

With independent arrows, each point of the span is reached by exactly one walk. If two different walks ended at the same point, going out along one and back along the other would be a walk to $0$ with some number not $0$, and that number's arrow would be a combination of the others.

The rank is how many independent directions the arrows give: the dimension of the span, 1 for a line, 2 for a plane, 3 for space. Independent arrows that span a line, plane or space are a basis of it, and the numbers of the one walk to a point are its coordinates in that basis. The usual $(x, y, z)$ are coordinates in the basis of one step along each axis.

# In the picture

- The green, orange and pink arrows are $v_1$, $v_2$, $v_3$, and the white point is $b$. Drag their tips.
- The blue shape is the span: a line, a see-through sheet for a plane, or a slanted lattice of the arrows' steps for all of space.
- The walk is drawn leg by leg in the arrows' colours, each leg labelled with its number of steps. Faint walks are other walks that also end at $b$.
- The cyan dots are where the walks of a sweep end. When no walk reaches $b$, a cyan point marks the nearest point of the span, and a dashed white line meets the span from $b$ at a right angle.

> Try first "two arrows paint a plane": every dot of the sweep lands in one flat sheet. Then "a third arrow in the plane", where the third adds nothing.

see: vectors, linmap2, basis, lsq

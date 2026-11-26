# The idea

How far apart can $n$ directions be kept? Here a direction is a line through the centre: a vector of length 1 together with its opposite. Between two such vectors the dot product is the cosine of the angle between them, so the aim is to make the largest $|\text{dot product}|$ as small as it can be. Directions far apart are hard to mistake for one another, which is what compressed sensing and networks that store many features in few dimensions need. You need vectors, the dot product, and for one step the eigenvalues of a symmetric matrix.

# The mathematics

## The number to make small

For $n$ vectors $v_1, \ldots, v_n$ of length 1 in $d$ dimensions, the coherence is

$$ \mu = \max_{i \ne j} |v_i \cdot v_j| $$

the cosine of the smallest angle between two of the lines. Turning a vector round to its opposite changes only signs, so it is lines, not arrows, that count. In the plane the best is to share the half turn out evenly: neighbours $180^\circ/n$ apart, and $\mu = \cos(180^\circ/n)$.

## The floor

With at least as many lines as dimensions, $n \ge d$, no arrangement can go below the Welch bound:

$$ \mu \ge \sqrt{\frac{n - d}{d(n - 1)}} $$

Why: put the vectors as the columns of a $d \times n$ matrix $V$. The table of all dot products, $V^\top V$, has 1 at every place of its diagonal, and at most $d$ eigenvalues that are not 0; they add up to the diagonal's sum, $n$. The squares of all its entries add up to the same total as the squares of its eigenvalues, and that is at least $n^2/d$ (the least when the $d$ eigenvalues are equal). Take away the $n$ ones on the diagonal: the other $n(n - 1)$ squared dot products add up to at least $n^2/d - n$. Their average is at least $(n - d)/(d(n - 1))$, and the largest, $\mu^2$, is at least the average.

The floor is reached only if every pair is at the same angle (equiangular lines), which is rare. For more than $d(d + 1)/2$ lines a second floor, $1/\sqrt{d}$, is higher still.

Example: the four diagonals of a cube,

$$ \tfrac{1}{\sqrt{3}}(1, 1, 1), \quad \tfrac{1}{\sqrt{3}}(1, -1, -1), \quad \tfrac{1}{\sqrt{3}}(-1, 1, -1), \quad \tfrac{1}{\sqrt{3}}(-1, -1, 1) $$

Every pair has dot product $-1/3$ (for the first two, $(1 - 1 - 1)/3$), so $\mu = 1/3$, an angle of $70.53^\circ$. The floor for $n = 4$, $d = 3$ is $\sqrt{1/9} = 1/3$: reached, so no four lines do better. For six lines the floor is $\sqrt{1/5} = 0.4472$, reached by the icosahedron; for seven or more the second floor, $1/\sqrt{3} = 0.5774$, takes over, and seven lines reach it.

## Finding arrangements

The scene starts from random directions and improves them a little at a time. The largest $|\text{dot product}|$ jumps from pair to pair as the lines move, so the scene lowers a smooth stand-in for it, a soft maximum of the squared dot products, and sharpens it as the lines settle. Each step mostly pushes the closest lines apart. A run ends where no small change helps: a local best, which need not be the best of all.

# In the picture

- Each blue line through the centre is one direction, $\pm v_i$; drag either end.
- Orange: the closest pairs, joined end to nearest end. They are the edges of the solid the ends make, and their cells in the table are orange too.
- The table of pairs shows $|v_i \cdot v_j|$, lighter for larger.
- The chart shows the largest $|\text{dot product}|$ step by step: the green rule is the best known value, the pink rule the floor.

> Try first "four lines: a cube", then "fourteen lines: a dead end", where a run stops short of the best.

see: vectors, highdim, sensing, superposition

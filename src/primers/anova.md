# The idea

Several groups, and one question: are their means further apart than the scatter within the groups would put them by chance? Testing every pair of groups on its own raises the chance of a false alarm; analysis of variance asks the question once. It splits the spread of all the points into a part between the groups and a part within, and compares them. It builds on the scenes on hypothesis tests and on comparing two groups.

# The mathematics

## Splitting the spread

There are $k$ groups, $N$ points in all. Point $j$ of group $i$ is $y_{ij}$; group $i$ has $n_i$ points and mean $\bar y_i$; the grand mean of all the points is $\bar y$. Each point's distance from the grand mean comes in two pieces, between and within:

$$ y_{ij} - \bar y = (\bar y_i - \bar y) + (y_{ij} - \bar y_i) $$

Summed over all the points, their squares add up too:

$$ \sum_{i,j} (y_{ij} - \bar y)^2 = \sum_i n_i (\bar y_i - \bar y)^2 + \sum_{i,j} (y_{ij} - \bar y_i)^2 $$

that is, $\text{total} = \text{between} + \text{within}$. Squaring a sum of two pieces also gives twice their product; but in each group the between piece is the same for every point and the within pieces add up to 0, so those products add up to 0 too. This holds for any data.

## Mean squares and $F$

Each part divided by its degrees of freedom is a mean square, and their ratio is

$$ F = \frac{\text{between} / (k - 1)}{\text{within} / (N - k)} $$

The degrees of freedom count the distances free to vary: $k - 1$ between, since the group means' distances, weighted by size, add up to 0; $N - k$ within, since in each group the points' distances from its mean add up to 0.

If the points are independent, the groups share one spread $\sigma$ and their true means are all equal, both mean squares estimate $\sigma^2$, and $F$ tends to be near 1. Unequal true means tend to make the between part, and $F$, larger. If each group is also bell-shaped, $F$ follows the $F$ distribution with $k - 1$ and $N - k$ degrees of freedom when the means are equal, and its tail beyond the observed $F$ is the $p$-value. Only a large $F$ counts against equal means. A small $p$ is evidence that the means are not all equal, not of which ones differ.

Example: groups $\{1, 3\}$, $\{4, 6\}$ and $\{7, 9\}$. The grand mean is 5, the group means 2, 5 and 8. Between: $2(3^2 + 0^2 + 3^2) = 36$. Within: every point is 1 from its group's mean, so 6. Total: $16 + 4 + 1 + 1 + 4 + 16 = 42 = 36 + 6$. Then $F = (36/2)/(6/3) = 9$, yet with so few points $p = 0.054$. The scene's opening data give $F = (10.55/3)/(26.98/36) = 3.52/0.75 = 4.69$, and $p = 0.0072$.

## Two groups, and the assumptions

With two groups, $F$ is the square of the pooled two-sample $t$, and the two tests give the same $p$. The share between, $\eta^2 = \text{between}/\text{total}$, measures how large the differences are; yet even when the true means are equal it averages $(k - 1)/(N - 1)$, for independent bell-shaped points of one spread. The test assumes one common spread. When two small groups are noisier than a large one, the within part is mostly the quiet group's, and $F$ passes the 5 % line too often: 19.9 % of the time in the scene's example.

# In the picture

- Cyan: the points, and each one's distance from its group's mean (within).
- Yellow: the group means (drag a bar), their distances from the grand mean (between), and $F$.
- White, dashed: the grand mean $\bar y$. Click a point to see its distance split in two.
- Purple: the $F$ distribution, and the $F$ of shuffled groups. Pink: the tail beyond $F$, of area $p$.
- Green: the true means, and the $F$ of new samples drawn from them.

> Try first "one group apart", then drag group 4's bar further away: the part between grows, and $F$ with it.

see: twogroups, pvalues, regress, multireg

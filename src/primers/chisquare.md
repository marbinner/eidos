# The idea

Counts in categories: the faces a die came up, or people sorted by where they live and what they drink. A hypothesis (the die is fair; place and drink have nothing to do with each other) says how many should fall in each cell on average. The chi-square test adds up, cell by cell, how far the counts are from that, and asks whether the total is more than chance would give. It builds on the scene on hypothesis tests.

# The mathematics

## The statistic

For each cell, $O$ is the observed count and $E$ the expected count: how many the hypothesis says fall there on average. Then

$$ \chi^2 = \sum_{\text{cells}} \frac{(O - E)^2}{E} $$

Why divide by $E$: a die face's count has variance $E \times \tfrac56$, a little under $E$ ($60 \times \tfrac16 \times \tfrac56 = 8.3$ in 60 rolls). So $(O - E)/\sqrt{E}$ is roughly the miss in units of the count's own wobble, and $\chi^2$ adds up their squares.

Example, the scene's fair die: 60 rolls, $E = 10$ for every face, counts 12, 8, 9, 7, 9, 15. The parts are $4/10$, $4/10$, $1/10$, $9/10$, $1/10$ and $25/10$, so $\chi^2 = 4.40$.

## Degrees of freedom and the curve

The chi-square distribution with $k$ degrees of freedom is how a sum of $k$ independent squared standard normal numbers is spread; its mean is $k$. The six parts of a die are not free: the counts must add up to 60, so once five are known the sixth is forced. That leaves $6 - 1 = 5$ degrees of freedom. If the die is fair, the rolls are independent and no expected count is small, $\chi^2$ follows nearly the chi-square curve with 5 degrees of freedom. The $p$-value is the curve's share beyond the observed $\chi^2$: for 4.40 it is 0.49, nothing unusual. Only a large $\chi^2$ counts against the hypothesis.

## Tables

A table sorts the same people two ways at once: rows (north, south) and columns (tea, coffee, juice). The hypothesis of independence says every row has its people in the same shares across the columns, so

$$ E = \frac{\text{row total} \times \text{column total}}{N} $$

with $N$ people in all. With the totals kept, only $(\text{rows} - 1)(\text{columns} - 1)$ counts can be chosen; the rest are forced. That is the degrees of freedom: 2, for two rows and three columns.

Example: north 20, 18, 12 and south 10, 24, 16. Each row has 50 people and the columns 30, 42 and 28 of 100, so each row expects 15, 21 and 14. Then $\chi^2 = 2\,(25/15 + 9/21 + 4/14) = 4.76$ and $p = 0.092$.

## Evidence is not strength

The same shares with ten times the people make every $O - E$ and every $E$ ten times as large, so every part grows tenfold, and $\chi^2$ with them: 47.62, and $p < 0.0001$. Strength is measured instead by Cramér's

$$ V = \sqrt{\chi^2 / (N(k - 1))} $$

with $k$ the smaller of the numbers of rows and columns: from 0 (none) to 1. Here it is 0.22 with 100 people or 1 000. With two rows and two columns, $\chi^2$ is the square of the $z$ statistic for the difference of two shares (its standard error from both rows together), and gives the same $p$. And the curve is an approximation: the usual advice asks for expected counts of at least 5.

# In the picture

- Cyan: the observed counts (bars for the die; for a table, tiles whose areas are the counts).
- Purple: the expected counts (dashed), and the chi-square curve.
- Yellow: each cell's part $(O - E)^2/E$ as the area of a square, and the parts laid end to end to make $\chi^2$.
- Pink: the curve beyond $\chi^2$, whose area is $p$.
- Green: the true die (drag the dot over the six to load it), and the $\chi^2$ of many new samples.

> Try first "a fair die": the green histogram of new sets of 60 rolls settles on the purple curve, and averages 5.

see: distributions, testing, pvalues, entropy

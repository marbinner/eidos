# The idea

Pick two directions at random in $n$ dimensions and measure the angle between them. In the plane every angle is equally likely; in a hundred dimensions the two are nearly always close to a right angle. That fact explains why $n$ dimensions have room for far more than $n$ directions that are almost at right angles; the same kind of averaging makes random vectors all nearly the same length, and lets multiplying by a random matrix nearly keep distances. You need vectors, the dot product, and the bell-shaped normal distribution.

# The mathematics

## A random direction

Take $n$ independent standard normal numbers $g_1, \ldots, g_n$ (bell-shaped, average 0, standard deviation 1). How likely each $g = (g_1, \ldots, g_n)$ is depends only on its length, so the cloud of draws looks the same from every side, and $g$ divided by its length is a direction $u$ in which no way is favoured. For two such directions $u$ and $v$, of length 1, the cosine of the angle between them is their dot product $u \cdot v$.

## Why the cosine is small

The squares of the $n$ coordinates of $u$ add up to 1, so by symmetry each is $1/n$ on average. Turn space so that $v$ lies along the first axis; then $u \cdot v$ is $u_1$, and so

$$ \text{average of } (u \cdot v)^2 = \frac{1}{n} $$

The cosine is spread round 0 with standard deviation exactly $1/\sqrt{n}$, and for large $n$ its spread is close to a bell.

Example: in $n = 100$ dimensions the typical cosine is $\pm 0.1$. A cosine of $0.2$, twice that, is an angle of $78.5^\circ$ (check: $\cos 78.5^\circ \approx 0.2$), and about 95 pairs in 100 fall between $78.5^\circ$ and $101.5^\circ$.

## Room for many directions

Exactly at right angles, $n$ dimensions hold only $n$ directions. Nearly at right angles they hold vastly more. Among $m$ random directions there are $m(m - 1)/2$ pairs, and for $n$ of 100 or more the largest $|\text{cosine}|$ among them is roughly

$$ \sqrt{2 \ln(m^2/2)/n} $$

which grows only very slowly with $m$, as the largest of many bell-shaped values with spread $1/\sqrt{n}$ does. In the scene's draw of a thousand directions in a hundred dimensions, every two are at least $62.7^\circ$ apart.

## Lengths and projections

The same averaging fixes lengths. The squared length $g_1^2 + \cdots + g_n^2$ averages $n$, with standard deviation $\sqrt{2n}$, so for large $n$

$$ |g| \approx \sqrt{n} \pm 1/\sqrt{2} $$

Nearly the same $\pm 1/\sqrt{2}$ whatever $n$ is: ever smaller in proportion to the length.

A random projection multiplies points in $N$ dimensions by a $k \times N$ matrix of standard normal numbers divided by $\sqrt{k}$. Each of the $k$ new coordinates of the difference $d$ between two points is then a normal number with standard deviation $|d|/\sqrt{k}$, so the new distance is $|d|$ times the length of $k$ standard normal numbers divided by $\sqrt{k}$:

$$ \frac{\text{distance after}}{\text{distance before}} \approx 1 \pm \frac{1}{\sqrt{2k}} $$

There is no $N$ in it. With $k = 50$ the typical change is 10 per cent, however many dimensions the points began in.

# In the picture

- "angles": the cyan bars are the cosines of random pairs, the white curve the exact law for this $n$, the orange marks $\pm 1/\sqrt{n}$. The small charts show $n = 2$, $3$, $10$ and $100$.
- The heat map gives the cosine of every pair of $m$ directions (red above 0, blue below); the green frame and green curve mark the largest $|\text{cosine}|$.
- "lengths": the cyan bars are lengths $|g|$; orange marks $\sqrt{n}$ and $1/\sqrt{2}$ either side.
- "projection": each cyan dot is a pair of points, distance before across and after up; the orange lines are $1 \pm 1/\sqrt{2k}$.

> Try first "the plane", then "a hundred dimensions": the same question, and the wide spread of cosines becomes a narrow bell round 0.

see: vectors, packing, sensing, superposition

# The idea

A grey picture of $64 \times 64$ pixels is a grid of 4096 numbers, 0 for black and 1 for white: a matrix $A$. The singular value decomposition writes it as a sum of simple pieces, the largest first. Keeping only the first few pieces often gives a picture close to the original, made from far fewer numbers, and no picture made of that many pieces is closer. The same idea compresses data, removes noise, and is at the heart of principal components. You need the SVD, $A = U \Sigma V^\top$, and the length of a vector.

# The mathematics

## Pieces

A piece is a column pattern $u$ times a row pattern $v^\top$, times a size $\sigma$: its pixel in row $i$ and column $j$ is $\sigma u_i v_j$, so every row of it is the row pattern, scaled. The SVD writes the picture as a sum of 64 such pieces:

$$ A = \sigma_1 u_1 v_1^\top + \sigma_2 u_2 v_2^\top + \cdots + \sigma_{64} u_{64} v_{64}^\top $$

The singular values come largest first, $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$. The $u$'s, the columns of $U$, have length 1 and are at right angles to each other, and so are the $v$'s, the columns of $V$. This is $A = U \Sigma V^\top$ written out: both sides send each $v_j$ to $\sigma_j u_j$. The rank of $A$ is the number of $\sigma$ that are not 0, the number of pieces it needs.

## Sizes add as in Pythagoras

The size of a picture is the square root of the sum of the squares of its pixels: its length, as a list of 4096 numbers. A piece $\sigma u v^\top$ has size $\sigma$, because $u$ and $v$ have length 1. Two different pieces $\sigma u v^\top$ and $\sigma' u' v'^\top$ are at right angles as lists of numbers: the sum of their pixels' products is $\sigma \sigma' (u \cdot u')(v \cdot v') = 0$. So, as in Pythagoras' theorem, their squared sizes add. Call the sum of the first $k$ pieces $A_k$; what it leaves out has size

$$ \text{size of } (A - A_k) = \sqrt{\sigma_{k+1}^2 + \sigma_{k+2}^2 + \cdots + \sigma_{64}^2} $$

That no picture of rank $k$ comes closer to $A$ than $A_k$ is a theorem, the Eckart–Young theorem. Storing $A_k$ takes $k(64 + 64 + 1)$ numbers: a $u$, a $v$ and a $\sigma$ for each piece.

Example: a $2 \times 2$ picture.

$$ A = \begin{bmatrix} 2 & 2 \\ 1 & -1 \end{bmatrix} = \begin{bmatrix} 2 & 2 \\ 0 & 0 \end{bmatrix} + \begin{bmatrix} 0 & 0 \\ 1 & -1 \end{bmatrix} $$

The first piece is the column pattern $(1, 0)$ times the row pattern $(2, 2)$; the second is $(0, 1)$ times $(1, -1)$. The column patterns are at right angles, and so are the row patterns, so these are the SVD's pieces. Their sizes are $\sigma_1 = \sqrt{2^2 + 2^2} = \sqrt{8} = 2.83$ and $\sigma_2 = \sqrt{2} = 1.41$, and $A$'s size is $\sqrt{4 + 4 + 1 + 1} = \sqrt{10} = 3.16$, where $10 = 8 + 2$. Keeping the first piece leaves an error of size $\sigma_2 = 1.41$, 45% of $A$'s size.

## What the sizes say

- If a few $\sigma$ are large and the rest small, a few pieces carry the picture.
- If all are equal, no piece matters more than another. The line (the identity matrix) has every $\sigma = 1$, so $k$ pieces leave an error of size $\sqrt{64 - k}$, out of 8.
- Noise added to every pixel spreads over all 64 pieces, as a floor of $\sigma$ that falls slowly. When the picture's own pieces stand well above that floor, cutting the floor off removes mostly noise.

# In the picture

- The grey pictures: $A$ and $A_k$. Beside them, what is left, $A - A_k$: blue below zero, red above.
- Below: one piece alone, $\sigma u v^\top$, with its column pattern $u$ and its row pattern $v^\top$ on the strips beside it.
- The chart: the sizes $\sigma_1 \ge \sigma_2 \ge \cdots$. Green bars are the pieces kept, purple the pieces dropped; the yellow dot is $k$.

> Try first "a few pieces carry a picture": watch each piece put back what the ones before it missed.

see: svd, pca, dft

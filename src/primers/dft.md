# The idea

A list of $N$ numbers, such as a sound sampled $N$ times, is a vector in $N$ dimensions, one axis per sample. This scene measures it along $N$ other axes, made of waves. Its coordinates along the waves are its discrete Fourier transform: how much of each frequency the signal holds. You need vectors, the dot product and the idea of a basis.

# The mathematics

## The waves

The signal is $x = (x_0, x_1, \ldots, x_{N-1})$, its samples. The wave $\cos k$ is the list of $N$ numbers

$$ \sqrt{2/N}\, \cos(2\pi k n/N), \qquad n = 0, 1, \ldots, N - 1 $$

and $\sin k$ is the same with $\sin$. The whole number $k$, the frequency, is how many times it fits into the $N$ samples. For even $N$ the basis is the constant, $\cos k$ and $\sin k$ for $k = 1$ to $N/2 - 1$, and the alternating wave $+1, -1, +1, \ldots$: $N$ waves in all.

## Why they are at right angles

The product of two waves is a sum of two others, of the difference and the sum of their frequencies:

$$ \cos a \cos b = \tfrac{1}{2} \cos(a - b) + \tfrac{1}{2} \cos(a + b) $$

and likewise for sines. A cosine or sine that fits between 1 and $N - 1$ whole times into the samples adds up to 0 over them. So two different waves have dot product 0. A cosine times itself is $\frac{1}{2} + \frac{1}{2} \cos 2a$, of which only the $\frac{1}{2}$ survives the adding up: $N/2$ in all, hence the $\sqrt{2/N}$. (The constant and the alternating wave take $1/\sqrt{N}$ instead.) The waves are orthonormal: of length 1 and at right angles.

## Coordinates are dot products

Along such axes, the coordinate of $x$ along a wave $w$ is the dot product $x \cdot w = \sum_n x_n w_n$, and the waves, each times its coordinate, add up to $x$ again. With the waves as the rows of a matrix $W$:

$$ \text{coordinates} = Wx, \qquad x = W^\top (Wx), \qquad \sum_n x_n^2 = \sum \text{coordinate}^2 $$

The last holds because $W W^\top = I$: the change of basis is rigid, like a turn, and keeps lengths.

Example, $N = 4$. The waves are the constant $\frac{1}{2}(1, 1, 1, 1)$, $\cos 1 = \frac{1}{\sqrt{2}}(1, 0, -1, 0)$, $\sin 1 = \frac{1}{\sqrt{2}}(0, 1, 0, -1)$ and the alternating wave $\frac{1}{2}(1, -1, 1, -1)$. For $x = (3, 1, 1, 1)$ the four dot products are $3$, $\sqrt{2}$, $0$ and $1$, and the lengths agree: $9 + 1 + 1 + 1 = 12 = 9 + 2 + 0 + 1$.

## Few waves or many

A smooth signal changes little from one sample to the next, the last to the first included, so a few slow waves rebuild it closely. A single spike is the opposite: every frequency from 1 to $N/2 - 1$ holds the same share of it.

A blur that replaces each sample by the average of itself and its neighbours treats every place alike: its matrix $C$ is circulant, each row the one above shifted along. Averaged over its neighbours, a wave comes back as itself times a number. For the average of 5, with $\theta = 2\pi k/N$, both $\cos k$ and $\sin k$ are multiplied by

$$ (1 + 2\cos\theta + 2\cos 2\theta)/5 $$

so in the wave basis the blur, $W C W^\top$, is diagonal.

# In the picture

- The white stems are the samples $x_n$ (drag them).
- Below them, the picked wave $w$ and the stack of products $x_n w_n$, yellow above 0 and purple below; where it ends is $x \cdot w$.
- The bars on the right are the $N$ coordinates, cyan for $\cos k$, pink for $\sin k$; the grey lid over a pair is that frequency's size.
- The green curve is $x$ rebuilt from the kept waves alone.
- The heat maps are $W$ and $W W^\top = I$; with a blur, $C$ and $W C W^\top$, its factors the orange dots.

> Try first "one wave, one coordinate": the signal is $1.5 \cos(2\pi \cdot 3n/32)$, which is $1.5 \times \sqrt{32/2} = 6$ times the wave $\cos 3$, so its coordinate there is 6, every other 0.

see: basis, fourier, lowrank

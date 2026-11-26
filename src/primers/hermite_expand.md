# The idea

An arrow can be written in an orthonormal basis, one coordinate per direction. The normalised Hermite polynomials $h_n$ are such a basis for the functions of finite Gaussian mean square, with the Gaussian average as the dot product. Writing a function $f$ in it says how much of $f$ is constant, how much is of degree 1, of degree 2, and so on, and cutting the sum short gives the best polynomial of each degree, judged where a normal variable actually lands. You need the previous scenes: $h_n = \mathrm{He}_n / \sqrt{n!}$ and $\langle f, g \rangle = \mathbb{E}[f(X)\, g(X)]$ for a standard normal $X$, whose density is $\varphi$.

# The mathematics

## Coordinates are averages

$$ f = \sum_{n=0}^{\infty} c_n h_n, \qquad c_n = \mathbb{E}[f(X)\, h_n(X)] $$

Take the average of both sides times $h_k$: since $\mathbb{E}[h_n h_k]$ is 1 for $n = k$ and 0 otherwise, only $c_k$ survives. Call $f_N = c_0 h_0 + \cdots + c_N h_N$ the sum cut at degree $N$. When $\mathbb{E}[f(X)^2]$ is finite, the mean square error $\mathbb{E}[(f(X) - f_N(X))^2]$ goes to 0 as $N$ grows. That is a statement about the average, not about every point: far out in the tails, where $\varphi$ is tiny, $f_N$ may stay far from $f$.

## Parseval, and the best approximation

Pythagoras with infinitely many perpendicular pieces:

$$ \mathbb{E}[f(X)^2] = c_0^2 + c_1^2 + c_2^2 + \cdots, \qquad \mathbb{E}[(f - f_N)^2] = \sum_{n > N} c_n^2 $$

So $c_n^2$ is the share of $\mathbb{E}[f^2]$ that degree $n$ carries. No polynomial $p$ of degree $N$ or less does better than $f_N$: the error $f - f_N$ is perpendicular to $h_0, \ldots, h_N$, hence to $f_N - p$, and by Pythagoras $\mathbb{E}[(f - p)^2] = \mathbb{E}[(f - f_N)^2] + \mathbb{E}[(f_N - p)^2]$. Best is measured by the Gaussian: errors count where $\varphi$ is large.

For a smooth $f$ whose first $n$ derivatives grow at most like a power of $x$, integrating by parts $n$ times gives a shortcut:

$$ c_n = \frac{\mathbb{E}[f^{(n)}(X)]}{\sqrt{n!}} $$

## Two worked examples

For $f(x) = x^3$: its derivatives are $3x^2$, $6x$ and $6$, so $c_1 = \mathbb{E}[3X^2] = 3$, $c_2 = \mathbb{E}[6X]/\sqrt{2} = 0$ and $c_3 = 6/\sqrt{6} = \sqrt{6}$. Indeed $x^3 = (x^3 - 3x) + 3x$, that is

$$ x^3 = 3 h_1 + \sqrt{6}\, h_3 $$

and Parseval checks: $9 + 6 = 15 = \mathbb{E}[X^6]$.

For $\operatorname{ReLU}(x) = \max(x, 0)$: $\mathbb{E}[f^2] = 1/2$, and

$$ c_0 = \frac{1}{\sqrt{2\pi}} = 0.399, \quad c_1 = \frac{1}{2}, \quad c_2 = \frac{1}{2\sqrt{\pi}} = 0.282 $$

Here $c_1 = \mathbb{E}[X \operatorname{ReLU}(X)]$ is half of $\mathbb{E}[X^2] = 1$, since $X \operatorname{ReLU}(X)$ is $X^2$ where $X > 0$ and 0 elsewhere. The three squares add to 0.489, which is 97.7 % of $1/2$. Every odd degree above 1 is zero, because $\operatorname{ReLU}(x) = x/2 + |x|/2$: the first half is all degree 1, and the second is even.

## How fast the bars shrink

For a smooth step such as $\tanh$ the bars fall fast: degrees up to 9 carry over 99.9 % of $\mathbb{E}[f^2]$. The jump of $\operatorname{sign}$ needs every odd degree, and its bars fall slowly: degrees up to 9 carry 84 %, degrees up to 40 only 92 %. The lowest degree $n \ge 1$ with $c_n \ne 0$ is called the information exponent: 1 for ReLU, 2 for $|x|$.

# In the picture

- Wide, faint white is $f$; cyan is $f_N$; the orange shading between them is the error.
- Purple is the Gaussian weight $\varphi$, where the error counts.
- The inset shows the bars $|c_n|$ by degree: cyan for the degrees kept ($n \le N$), grey for those left out.
- The line at the top gives the share of $\mathbb{E}[f^2]$ that degrees 0 to $N$ capture; the panel reads out $\mathbb{E}[f(X)^2]$, the root-mean-square error and the coefficients.

> Try first "ReLU: three numbers", then slide $N$ up and watch the orange shrink under the purple hump.

see: hermite_orth, hermite_noise, fourier

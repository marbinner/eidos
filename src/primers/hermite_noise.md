# The idea

Take two standard normal numbers $X$ and $Y$ that are correlated: knowing $X$ tells you something about $Y$, but not all of it. Averaging a function $f$ over what $Y$ might be, once $X$ is known, is the noise operator $T$. It leaves the shape of each Hermite polynomial alone and only rescales it, degree $n$ by $\rho^n$, and from that one fact follows how alike $f(X)$ and $f(Y)$ are. It is the kernel that a random neural network computes in the limit of infinite width, if it has one hidden layer of units applying $f$, standard Gaussian weights and no biases, and two inputs of length 1 whose dot product is $\rho$. You need the normalised Hermite polynomials $h_n$, the coefficients $c_n$ of the expansion scene, and what a correlation is.

# The mathematics

## Two correlated normals

Let $Z$ be a second standard normal, independent of $X$, and $-1 \le \rho \le 1$. Put

$$ Y = \rho X + \sqrt{1 - \rho^2}\, Z $$

Then $Y$ is standard normal too (its variance is $\rho^2 + 1 - \rho^2 = 1$), and its correlation with $X$ is $\mathbb{E}[XY] = \rho$. Once $X = x$ is known, $Y$ is spread in a bell centred at $\rho x$ with standard deviation $\sqrt{1 - \rho^2}$.

## The noise operator

$$ T f(x) = \mathbb{E}[f(Y) \mid X = x] = \mathbb{E}\left[f\left(\rho x + \sqrt{1 - \rho^2}\, Z\right)\right] $$

At $\rho = 1$ nothing changes, $T f = f$. At $\rho = 0$ every $x$ gives the same number, $\mathbb{E}[f(X)]$. A $\rho$ strictly between 0 and 1 smooths $f$ and, in mean square, pulls it towards its average.

## Hermite polynomials are only scaled

$$ T h_n = \rho^n \, h_n $$

By hand for degree 2, with $h_2 = (x^2 - 1)/\sqrt{2}$ and $s = \sqrt{1 - \rho^2}$: since $\mathbb{E}[Z] = 0$ and $\mathbb{E}[Z^2] = 1$,

$$ \mathbb{E}[(\rho x + s Z)^2 - 1] = \rho^2 x^2 + s^2 - 1 = \rho^2 (x^2 - 1) $$

so $T h_2 = \rho^2 h_2$. For every $n$: $T$ turns a polynomial of degree $n$ into one of degree at most $n$, with the coefficient of $x^n$ multiplied by $\rho^n$. And $T$ can be moved across an average, $\mathbb{E}[(Tf)(X)\, g(X)] = \mathbb{E}[f(Y)\, g(X)] = \mathbb{E}[f(X)\, (Tg)(X)]$, because the pair $(Y, X)$ has the same distribution as $(X, Y)$. So for $k < n$, $\mathbb{E}[(T h_n)\, h_k] = \mathbb{E}[h_n\, (T h_k)] = 0$, as $T h_k$ has degree at most $k$. A polynomial of degree at most $n$ perpendicular to all lower degrees is a multiple of $h_n$, and the leading coefficient fixes the multiple at $\rho^n$.

Example from the picture: at $\rho = 0.7$, degree 3 and $x = 1.6$, $h_3(1.6) = (1.6^3 - 3 \cdot 1.6)/\sqrt{6} = -0.287$, and $\rho^3 = 0.343$ makes the average $-0.099$. High degrees fade first: at the same $\rho$, degree 8 keeps only $0.7^8 = 0.058$. A negative $\rho$ turns the odd degrees upside down.

## Any function, and the kernel

Writing $f = \sum c_n h_n$ gives $T f = \sum \rho^n c_n h_n$, and then

$$ \mathbb{E}[f(X)\, f(Y)] = \mathbb{E}[f(X)\, T f(X)] = c_0^2 + c_1^2 \rho + c_2^2 \rho^2 + \cdots $$

As a function of $\rho$ this is the kernel: at $\rho = 1$ it is $\mathbb{E}[f^2]$, at $\rho = 0$ it is $c_0^2 = \mathbb{E}[f]^2$. For ReLU at $\rho = 0.7$ it is 0.3750. Packed into one line, all of this is Mehler's formula: for $|\rho| < 1$, with $\varphi$ the standard normal density, the density of the pair $(X, Y)$ is $\varphi(x)\, \varphi(y) \sum_n \rho^n h_n(x)\, h_n(y)$.

# In the picture

- Wide, faint white is $f$: one $h_n$, or a function you choose.
- The yellow point is $x$ (drag it along the axis); the yellow bell is where $Y$ lands once $X = x$ is known, and the yellow stretch of $f$ is what it averages.
- Cyan is $T f$; its dot over the yellow point is that average.
- The green inset is $\mathbb{E}[f(X)\, f(Y)]$ for every $\rho$ from $-1$ to $1$.
- The panel also gives the angle $\theta$ with $\cos\theta = \rho$, used in the next scene.

> Try first "high degrees are forgotten first": the cyan curve is the white one shrunk seventeen-fold.

see: gauss2, hermite_expand, hermite_dirs

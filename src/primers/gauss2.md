# The idea

Two quantities measured on the same thing, such as a person's height and weight, often vary together. A first model of this is a pair of correlated normals: each on its own follows a bell curve, and one number, the correlation $\rho$, says how closely they move together. The scene draws the bell over the plane, its tilted rings and axes, and what knowing $x$ says about $y$. It leans on the standard deviation, the normal distribution and conditional probability, from the earlier scenes.

# The mathematics

## Making a correlated pair

Start from two independent standard normal numbers $z_1$ and $z_2$ (mean 0, standard deviation 1), and make

$$ x = \sigma_x z_1, \qquad y = \sigma_y \left( \rho\, z_1 + \sqrt{1 - \rho^2}\, z_2 \right) $$

with $\rho$ between $-1$ and 1. Then $x$ is normal with mean 0 and standard deviation $\sigma_x$. So is $y$, with $\sigma_y$: a sum of independent normal numbers is normal, and the variances of independent parts add, $\rho^2 + (1 - \rho^2) = 1$. What ties $y$ to $x$ is the shared part $\rho\, z_1$. At $\rho = 0$ there is none, and $x$ and $y$ are independent; as $\rho$ nears 1 or $-1$, $y$ is nearly a fixed multiple of $x$.

## Covariance and correlation

The covariance is the mean of $(x - \mu_x)(y - \mu_y)$, where $\mu_x$ and $\mu_y$ are the means of $x$ and $y$; here both are 0, so it is the mean of $xy$. The mean of $z_1^2$ is 1 and that of $z_1 z_2$ is 0, so the covariance is $\rho\, \sigma_x \sigma_y$, and the correlation, the covariance over $\sigma_x \sigma_y$, is $\rho$. The covariance matrix holds all three numbers:

$$ \begin{bmatrix} \sigma_x^2 & \rho\, \sigma_x \sigma_y \\ \rho\, \sigma_x \sigma_y & \sigma_y^2 \end{bmatrix} $$

## Rings and axes

The recipe turns each circle $z_1^2 + z_2^2 = k^2$ into an ellipse, the $k\sigma$ ellipse, and the bell's height is the same all round it. A share $1 - e^{-k^2/2}$ of the pairs falls inside it, about 39 % for $k = 1$ and 86 % for $k = 2$, since $z_1^2 + z_2^2$ is a chi-square with 2 degrees of freedom, which is an exponential with rate $1/2$.

The principal axes are the directions of most and least spread: in linear algebra's words, the eigenvectors of the covariance matrix, and the variances along them are its eigenvalues. When $\sigma_x = \sigma_y = 1$ they are the two diagonals, with variances $1 + \rho$ and $1 - \rho$.

## Knowing $x$

Put $z_1 = x / \sigma_x$ into the recipe:

$$ y = \rho\, \frac{\sigma_y}{\sigma_x}\, x + \sigma_y \sqrt{1 - \rho^2}\, z_2 $$

Fix $x$, and the first part is fixed; the second is a normal number with mean 0 and standard deviation $\sigma_y \sqrt{1 - \rho^2}$, whatever $x$ is. So $p(y \mid x)$, how $y$ spreads among the pairs with this $x$, is normal: centred on the regression line $y = \rho (\sigma_y / \sigma_x)\, x$, and narrower than $p(y)$ by the factor $\sqrt{1 - \rho^2}$.

Example, as the scene opens: $\sigma_x = 1.3$, $\sigma_y = 0.9$, $\rho = 0.7$, slice at $x = 1.2$. The covariance is $0.7 \times 1.3 \times 0.9 = 0.819$. The mean of $y$ given $x$ is $0.7 \times (0.9 / 1.3) \times 1.2 \approx 0.58$, and its spread is $0.9 \times \sqrt{1 - 0.49} \approx 0.64$, against 0.9 for $y$ overall.

# In the picture

- The surface: how likely each pair $(x, y)$ is (its peak always drawn at the same height). Blue dots: 700 pairs drawn from it.
- White rings: the $1\sigma$ and $2\sigma$ ellipses. Orange and purple: axis 1 and axis 2, each as long as the standard deviation along it.
- On the back wall, in red, $p(x)$; on the left wall, in green, $p(y)$: how each spreads on its own.
- Yellow: the slice at one $x$ (drag it), the regression line, and $p(y \mid x)$ on the left wall.

> Try first "almost a line", and drag the slice: the yellow curve on the wall stays narrow and moves with $x$.

see: chance, moments, eigen, pca

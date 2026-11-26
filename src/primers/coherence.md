# The idea

A Jacobian lens reads a network's middle layers by asking how the output answers a small nudge there. It averages that answer over the tokens that follow the nudge (the lags) and over many texts (the contexts). Averaging arrows that point different ways loses length, so the lens keeps what is strong, steady and agreed on. This scene shows it in a residual stream of only two directions, the modes, where everything can be drawn. You need lengths of vectors, averages and singular values; the Lyapunov scene comes first.

# The mathematics

## The answers

Nudge the stream along a direction $v$ of length 1. In context $c$, $\tau$ tokens later, the output answers $K(c, \tau)\, v$, where $K(c, \tau)$ is the Jacobian there. In this network each mode, a direction $e_i$ (the two at right angles), answers on its own:

$$ K(c, \tau)\, v = \sum_{i = 1}^{2} \pm\, g_i \, \gamma_i^{\tau} \cos(\theta_i \tau) \, (e_i \cdot v) \, e_i $$

Mode $i$ has strength $g_i$, fade per token $\gamma_i$ (1 for no fading) and turn per token $\theta_i$; at $\theta_i = \pi$ its answer flips sign at every token. The sign $\pm$ is the context's: $+$ in a share $(1 + \kappa_i)/2$ of the contexts and $-$ in the rest, so $\kappa_i = 1$ means every context agrees and $\kappa_i = 0$ is a coin toss.

## The lens and its three factors

The lens averages over the lags $\tau = 0, \ldots, H - 1$, then over the contexts:

$$ G_c v = \frac{1}{H} \sum_{\tau = 0}^{H - 1} K(c, \tau)\, v, \qquad J v = \operatorname{mean}_c \, G_c v $$

Three numbers track what each averaging keeps, for each $v$:

- magnitude $a$: the length $|K(c, \tau)\, v|$, averaged over lags and contexts;
- lag coherence $\rho_\tau$: the average length $|G_c v|$, as a share of $a$;
- context coherence $\rho_c$: the length $|J v|$, as a share of the average $|G_c v|$.

The two coherences are the shares that each averaging leaves, so, by these definitions alone, the three multiply to what the lens sees:

$$ |J v| = a \cdot \rho_\tau \cdot \rho_c $$

The lens ranks directions by its singular values. The largest is the largest $|J v|$ over all directions $v$, and here, with the modes at right angles, the two are $|J e_1|$ and $|J e_2|$. How much the activations vary along a direction plays no part.

## Two examples

In "Only the contexts differ", mode 2 is steady and unfaded with $g_2 = 1$, and $\kappa_2 = 1/3$. Along $v = e_2$ every answer has length 1, so $a = 1$. Nothing turns, so averaging over lags loses nothing: $\rho_\tau = 1$. The sign is $+$ in $2/3$ of the contexts and $-$ in $1/3$, so $|J e_2| = 2/3 - 1/3 = 1/3$, and $\rho_c = 1/3$.

Let mode 2 flip instead ($\theta_2 = \pi$, $\kappa_2 = 1$) with $H = 16$. Its answers go $+1, -1, +1, \ldots$, eight of each, so $G_c e_2 = 0$ and $\rho_\tau = 0$, although $a = 1$.

## A tuned lens

Tuned to a frequency $\omega$, the lens turns the answer at lag $\tau$ back by $\omega \tau$ before it adds them up, which picks out answers that turn at that rate. At $\omega = 0$ it is the plain average.

# In the picture

- The big picture: for every direction $v$, three curves at distance $a$ (grey, outer), $a \rho_\tau$ (yellow) and $a \rho_\tau \rho_c = |J v|$ (pink, inner) from the centre. Drag $v$, the white dot.
- The blue and orange lines are modes 1 and 2; the pink line is the lens's first direction. The purple dots are the activations, with the direction of most variance (PC 1).
- Top right: the three factors along $v$, as bars.
- "one context, lag by lag": the answers $K(c, \tau)\, v$ laid tip to tail; how far the chain gets, over the length walked, is $\rho_\tau$.
- "context by context": each context's lag average $G_c v$ (yellow, thicker when likelier) and their mean $J v$ (pink).

> Try first "Strong but unsteady loses": mode 2 answers more strongly, yet the lens picks mode 1.

see: lyapunov, svd, belief

# The idea

The scene looks closely at one observational study of shingles vaccination and dementia, a natural experiment built on a change of vaccine. This primer explains the scene's methods; like the scene, it gives no verdict on any vaccine and is not medical advice. It assumes causal graphs, and that holding a common effect opens a road.

# The mathematics

## A natural experiment

A natural experiment compares groups formed by an event the people did not choose, here the date of a switch between vaccines. In the graph, $E$ is the period of the first dose, $T$ the vaccine, $D$ dementia, $D^*$ its recorded diagnosis. The comparison measures $E \to T \to D \to D^*$ only if every other road from $E$ to $D^*$ is closed. Matching balances recorded traits $X$, not unrecorded ones, $U$. Counting only those still in the records holds the common effect $C$ ($E \to C \gets D$): a period in which more people on their way to a diagnosis left the records first would also show fewer diagnoses.

## A Lexis diagram

Calendar time runs across, time since the first dose up. Someone first vaccinated at calendar time $v$ is at $v + t$ after $t$ years: a diagonal. An event at calendar time $c$ is a vertical line, met at follow-up time

$$ t = c - v $$

so cohorts vaccinated at different $v$ meet it at different $t$.

## The toy model

In the toy, someone vaccinated at $v$ is diagnosed $t$ years later, in follow-up year $k$, at the rate

$$ \lambda = b_k \, \theta \, (1 - u) \, p(v + t) \, e^{\delta (v + t - t_0)} $$

$b_k$ is the baseline; $\theta$ the newer vaccine's true hazard ratio and $u$ the share by which healthier adopters' risk is lower, both in the later cohort only; $p$ is $1 - s$ during a toy pandemic, else 1; $\delta$ a yearly drift in how readily dementia is diagnosed, from a date $t_0$. For a later dose at $v_L$ and an earlier at $v_E$, at the same $t$:

$$ \frac{\lambda_L}{\lambda_E} = \theta\,(1 - u)\,\frac{p(v_L + t)}{p(v_E + t)}\,e^{\delta (v_L - v_E)} $$

$\theta$ and $1 - u$ enter only as a product: this comparison cannot tell a true effect from healthier adopters. A drift gives a factor constant in $t$; a calendar event, one that changes with $t$.

With made-up numbers: $v_E = 1$, $v_L = 4$, and only a toy pandemic, from calendar time 5 to 7 with $s = 0.4$. The later person is in it for $t$ from 1 to 3, the earlier for $t$ from 4 to 6, so the ratio is 0.6 in follow-up years 2 and 3, $1 / 0.6 \approx 1.67$ in years 5 and 6, and 1 otherwise, with no effect. Spread-out doses blur these steps.

## Measures and checks

The cumulative incidence $F(t)$ is the share diagnosed by $t$. The restricted mean time lost over a window from $a$ to $\tau$ is

$$ \text{RMTL} = \int_a^\tau F(t)\,dt $$

the average time spent after a diagnosis within the window; the scene's ratio is later over earlier. With few diagnosed, a hazard ratio $\theta$ constant in $t$ gives a ratio close to $\theta$; and the same diagnoses made later give less time lost, so a delay alone lowers it.

- A negative-control outcome shares the suspected biases but should not be changed by the vaccine: a difference in it signals bias. In the toy, only adopters healthier in general move it.
- A placebo switch is the same comparison among people given a vaccine that did not change: whatever came with the period, and not with the new vaccine, shows in it.

# In the picture

- "the comparison": the Lexis diagram, earlier cohort grey, later cyan, with the switch, the pandemic and the end of the data as vertical lines.
- The graph: purple roads from $E$, dashed where the design needs them absent; a box is held.
- "toy model", in a dashed frame: pink, the toy's comparison; green, the vaccine alone; yellow, the paper's numbers.

> Try first "no effect, and a pandemic": the toy's ratio by year falls below 1 early and rises above 1 late.

see: junctions, adjust, shingles

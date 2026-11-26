# The idea

A network is trained on an arithmetic table: for every pair of numbers $a$ and $b$ from $0$ to $p - 1$ it must give $a + b \;\mathrm{mod}\; p$, and some cells are never shown to it. It first memorises the cells it trains on, and only much later gets the held-out cells right: that is grokking. The rule it finds can be read off its weights: it puts numbers on circles and adds angles. You need sines and cosines, and the Network playground.

# The mathematics

## The task and the network

$a + b \;\mathrm{mod}\; p$ is the remainder of $a + b$ on division by $p$: with $p = 17$, the sum $10 + 12 = 22$ gives $5$. Of the $17^2 = 289$ cells, by default it trains on 80 % (231) and never sees the other 58.

The input is two one-hot codes side by side: $2p$ numbers, all $0$ but a $1$ in place $a$ of the first half and a $1$ in place $b$ of the second. By default a layer of 128 neurons follows, then $p$ outputs, one per answer; the largest is the network's answer. The weights from input $a$ to those neurons are the number $a$'s own: its embedding. Weight decay pulls every weight towards zero at each step; without it, the scene shows, the held-out cells stay wrong.

## Numbers on circles

Frequency $k$ puts number $a$ at the angle

$$ \theta_a = 2\pi k a / p $$

on a circle, going $k$ times round as $a$ goes from $0$ to $p$. Angles add as the numbers do, and a whole turn changes nothing, as adding $p$ changes nothing mod $p$. Example, $p = 17$ and $k = 1$: $10$ and $12$ sit $10/17$ and $12/17$ of a turn round; together $22/17$, one whole turn and $5/17$: where $5$ sits. The cosine of a sum needs only the cosines and sines of the parts,

$$ \cos(\theta_a + \theta_b) = \cos\theta_a \cos\theta_b - \sin\theta_a \sin\theta_b $$

so adding two numbers becomes adding two angles, which a layer of neurons can do.

## A vote of frequencies

In an idealised vote, score each candidate answer $c$ by how well its angle matches, summed over the frequencies $k = 1$ to $(p - 1)/2$:

$$ \sum_{k = 1}^{(p - 1)/2} \cos\frac{2\pi k (a + b - c)}{p} = \begin{cases} (p - 1)/2 & \text{if } c = a + b \;\mathrm{mod}\; p \\ -1/2 & \text{otherwise} \end{cases} $$

For $p = 17$: 8 for the right answer, $-1/2$ for each wrong one. Check $p = 5$, $c$ one off: $\cos 72^\circ + \cos 144^\circ = 0.31 - 0.81 = -0.5$. Alone, each frequency gives some wrong answer $\cos(360^\circ / 17) = 0.93$ against the right one's 1; summed, they are sharp, and removing one costs little.

## Finding the waves in the weights

Take one first-layer neuron's weights from the inputs $a = 0, \ldots, p - 1$: a list $w_0, \ldots, w_{p - 1}$. Its power at frequency $k$ is

$$ \Big( \sum_a w_a \cos\theta_a \Big)^2 + \Big( \sum_a w_a \sin\theta_a \Big)^2 $$

plus the same for its weights from $b$. Frequencies $k$ and $p - k$ give the same power, so for $p = 17$ only $k = 1$ to $8$ are counted, and each neuron's power is shared out among them. The frequency concentration is a neuron's largest share, averaged over the neurons: 1 if every neuron is a single wave, about 0.27 for random weights.

# In the picture

- The big table: the network's answer for each cell, $a$ across and $b$ up, one colour per answer. White dots mark wrong answers, small dark dots the held-out cells.
- The small tiles: one per neuron, its value over the whole table.
- The strip under the table: a column per first-layer neuron, bright at the frequency $k$ it uses.
- The rings at the bottom: frequency $k$'s part of the embedding, each number near its angle $\theta_a$; the brighter, the more neurons use that wave. Click one to remove it.
- The curves: loss and share right, cyan for trained-on cells, orange for held-out ones.

> Try first "Memorise first, understand later": cyan reaches 100 % within about a hundred steps, and orange climbs over the next few thousand steps.

see: mlp, dft, fourier, transformer

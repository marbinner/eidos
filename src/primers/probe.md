# The idea

A common way to ask what a network knows is a probe: a straight-line read-out that tries to tell some property of the input from one layer's activations. This scene fits probes for five properties at every layer of a small network as it trains. Then it asks whether the network uses what a probe found, by moving the activations so that the probe says the opposite and watching whether the network's answers change. You need dot products, least squares, and the Network playground.

# The mathematics

## The network and its layers

The input is a point $(x_1, x_2)$ of the plane. Three hidden layers of eight $\tanh$ neurons follow, then one output, whose sign is the network's answer: one class or the other. At each layer the activations form a list $h$: 2 numbers at the input, 8 at each hidden layer, 1 at the output.

## The probe

A probe for one property and one layer is a weight vector $w$ and a threshold $t$. It says yes when

$$ r(h) = w \cdot h - t > 0 $$

and the size of $r(h)$ says how firmly. It is fitted by least squares on the training points: $r(h)$ is pulled towards $+1$ where the property holds and $-1$ where it does not, the two kinds weighted equally. Then it is scored on held-out points, by its hit rate on each kind, averaged:

$$ \text{score} = \tfrac{1}{2} \, (\text{hit rate on yes} + \text{hit rate on no}) $$

A probe that always says yes scores 50 % even where 90 of 100 points have the property, so chance is 50 % for every property. A score of 80 % or more counts as readable.

## Pushing across

A high score shows that a property can be read off a layer, not that the network uses it. To test use, let $s$ be the average $h$ of the points with the property minus the average of those without, the direction in which the two kinds differ, and move each point along it:

$$ h' = h - \frac{2 \, r(h)}{w \cdot s} \, s $$

Since $r$ is a straight-line function, $r(h') = r(h) - 2 r(h) = -r(h)$: the probe now says the opposite, as firmly as before. The rest of the network then runs on from $h'$, and the scene counts how many of its answers change. If none change, the network was not using where the points sat along $s$; if they change, it was.

Example: $w = (1, 0)$, $t = 0.5$ and $h = (1, 2)$ give $r = 0.5$. With $s = (2, 1)$, $w \cdot s = 2$, so $h' = (1, 2) - 0.5 \cdot (2, 1) = (0, 1.5)$, and $r(h') = 0 - 0.5 = -0.5$.

Only a readable probe is pushed, and none at the output, after which there is no network left to run.

## Readable is not used

Eight $\tanh$ neurons keep much about the input, including things the answer does not depend on: for a ring and its middle, whether a point lies right of centre. And a probe brings weights of its own: it can read the class off layer 2 of a network that has learned nothing. Only the push asks the network itself.

# In the picture

- The table: one row per property, one column per layer, each number a score in %. The bottom row, random labels, shows what failure looks like.
- A red bar in a cell: the share of the network's answers that change when the points are pushed across there. No bar: none change, or nothing is pushed there.
- On the left, the input plane for the chosen cell. The ground shows what that probe says at each place, the input run through the network to its layer: red for yes, blue for no, white on the boundary $r(h) = 0$. The dots are the training points: orange where the property holds, blue where not.
- Under the table, the network's own loss.

> Try first "Readable is not the same as used", then "What the network does use".

see: logistic, mlp, fold, sae

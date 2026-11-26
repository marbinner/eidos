# The idea

The last neuron of a network can only make one flat cut. A ring of points round a disc cannot be parted by any straight line, so the layers before it must bend the plane of inputs until a flat cut is enough. This scene follows that plane through the layers, as a sheet in space. You need the playground's neurons and layers, and linear maps.

# The mathematics

## A layer moves every point

A hidden layer of $m$ neurons sends each point $x$ of the space before it to

$$ h = \varphi(W x + b) $$

a point of $m$-dimensional space, one axis per neuron. First $x \mapsto W x + b$, a linear map and a shift, stretches, turns and shears, but keeps straight lines straight. Then $\varphi$ bends each coordinate on its own: $\tanh$ squeezes it into the range from $-1$ to $1$ (the cube), pressing large values towards the walls; ReLU sets negative values to 0, folding space flat against the walls of a corner. With three neurons the plane goes into three dimensions as a bent sheet, carrying its grid lines along.

## The output's cut

The output neuron computes $s = v \cdot h + c$ from the last hidden layer's $h$, and the answer is orange where $s > 0$, blue where $s < 0$. The boundary $s = 0$ is a flat plane in the space of $h$. So the network can be right about every point exactly when its last hidden layer has put the classes on opposite sides of some flat plane.

## Why two tanh neurons cannot free the disc

With two neurons a layer maps the plane to the plane. When $W$ is invertible, as it almost always is, and $\varphi = \tanh$, which only ever increases, this map is continuous and one-to-one: it can stretch and squash but never tear or fold the plane over. So the circle the ring lies on stays a closed loop round the disc, and a straight line with the whole loop on one side has everything inside it on that side too. With three neurons the sheet can leave the plane: the disc sinks into a bowl, the ring rises round its rim, and a flat cut passes between them.

ReLU is not one-to-one: a fold can send two points to one place. Example, a small version of "opposite corners (xor)": one class at $(1, 1)$ and $(-1, -1)$, the other at $(1, -1)$ and $(-1, 1)$, and two ReLU neurons

$$ h_1 = \max(0,\, x_1 + x_2), \qquad h_2 = \max(0,\, -x_1 - x_2) $$

They send $(1, 1)$ to $(2, 0)$ and $(-1, -1)$ to $(0, 2)$, but both $(1, -1)$ and $(-1, 1)$ to $(0, 0)$. The plane has been folded along the crease $x_1 + x_2 = 0$, and the straight cut $h_1 + h_2 = 1$ now separates the classes: 2 for the first, 0 for the second. Back in the input plane that cut is $|x_1 + x_2| = 1$, two parallel lines, which no single straight line could make.

## Wider layers, and after the cut

A layer of more than three neurons is drawn along the three directions in which its activity varies most. By default, training stops once the classes are separated. If it goes on, the loss can still fall by making the output larger, and the $\tanh$ neurons saturate: the points crowd into corners of the cube.

# In the picture

- The sheet is the plane of inputs as the chosen layer places it, coloured by the network's answer; its dark lines are the input plane's grid.
- The balls are the data points, in the colour of their true class.
- At the last hidden layer, the white plane is the output's cut, $s = 0$.
- The wire cube is the range of $\tanh$ neurons. Between two layers, each point glides straight from one picture to the next.

> Try first "a ring round a disc", stepping through with "next layer"; then "only two neurons per layer".

see: linmap3, mlp, signal

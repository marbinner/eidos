# The idea

An optimiser is the rule that turns a gradient into a step of the weights. Here one network is trained once per rule, from the same starting weights on the same batches, so whatever differs between the lanes is the rule's doing. You need the playground's loss and gradient, and gradient descent.

# The mathematics

## A gradient from a batch

Write $\theta$ for all the weights and biases and $L(\theta)$ for the loss on all the data. Each step uses the gradient $g$ of the loss on a batch of points (32 at first), not on all of them. It is right on average but noisy: very noisy with one point per step, exactly $\nabla L$ with all the data.

## The rules

$\eta$ is the learning rate and $k$ counts the steps; except in Muon, each number in $\theta$ is treated on its own. SGD (stochastic gradient descent) steps straight downhill:

$$ \theta_k = \theta_{k-1} - \eta\, g_k $$

Momentum keeps a running sum $m$ of gradients, with $\mu = 0.9$:

$$ m_k = \mu\, m_{k-1} + g_k, \qquad \theta_k = \theta_{k-1} - \eta\, m_k $$

A gradient that stays the same piles up to $1/(1 - \mu) = 10$ times its size; one that keeps changing sign largely cancels. Nesterov steps by $\eta\,(g_k + \mu\, m_k)$ instead, looking one step ahead.

AdaGrad and RMSProp divide each step by the size of the gradients so far, kept in $v$. AdaGrad adds up every $g^2$; RMSProp keeps an average that forgets old ones, $v_k = 0.9\, v_{k-1} + 0.1\, g_k^2$. Then (with $10^{-8}$ added to the root)

$$ \theta_k = \theta_{k-1} - \eta\, \frac{g_k}{\sqrt{v_k}} $$

Adam does both, with running averages of $g$ and of $g^2$:

$$ m_k = 0.9\, m_{k-1} + 0.1\, g_k, \qquad v_k = 0.999\, v_{k-1} + 0.001\, g_k^2 $$

Both start at 0 and so are too small at first; dividing by $1 - 0.9^k$ and $1 - 0.999^k$ corrects that, giving $\hat m_k$ and $\hat v_k$, and

$$ \theta_k = \theta_{k-1} - \eta\, \frac{\hat m_k}{\sqrt{\hat v_k}} $$

Muon treats each weight matrix as a whole. It writes a running average of the matrix's gradients as $M = U S V^\top$, its singular value decomposition: a turn or mirror, a stretch along the axes by the singular values in $S$, and another. It steps along $U V^\top$, the same with every stretch set to 1, so that every direction of the layer moves at the same rate. (The scene finds $U V^\top$ approximately; biases take Adam steps.)

Example: a weight whose gradient is always $g = 0.5$. SGD at its usual rate 0.3 steps $0.3 \cdot 0.5 = 0.15$ every time. Momentum at 0.05: $m$ runs $0.5, 0.95, 1.355, \ldots$ towards $0.5 / (1 - 0.9) = 5$, so its steps grow from 0.025 towards 0.25. Adam at 0.02, first step: $m_1 = 0.05$ and $v_1 = 0.00025$, corrected to $\hat m_1 = 0.5$ and $\hat v_1 = 0.25$, a step of $0.02 \cdot 0.5 / \sqrt{0.25} = 0.02$. With $g = 100$ or $g = 0.001$ it would still be 0.02: Adam's step depends on $\eta$, not on the gradient's overall scale.

## What the experiments turn on

A step too long overshoots the bottom of a valley and lands on the far wall, so a larger $\eta$ is not always faster. Once every point is on its right side, doubling the output neuron's weights and bias makes every answer surer and lowers every point's loss, so the loss keeps falling as the weights grow and the boundary sharpens; rules differ in how fast they grow them. Weight decay pulls against this: after each step every weight (not bias) is multiplied by $1 - d/100$, for the setting $d$.

# In the picture

- Each square is the input plane, coloured by the network's answer. The bold white line is the boundary, the fainter ones 10 % and 90 %; a white ring marks a point on the wrong side.
- Each lane has its rule's colour: SGD cyan, momentum pink, Nesterov yellow, AdaGrad green, RMSProp purple, Adam orange, Muon blue.
- The chart is the loss on all the data, on a log scale.

> Try first "the usual race", then "all the data every step".

see: mlp, descent, svd, landscape

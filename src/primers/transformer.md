# The idea

A transformer reads a row of tokens, here letters, and at every position guesses the next one. This one is small enough to read: two layers with one attention head each, 9 856 weights. It is trained on rows in which a run of letters repeats, and finds a two-step method, which the scene's tools take apart. You need matrices, dot products and the Network playground.

# The mathematics

## The residual stream

At position $t$ the model keeps a list of 32 numbers, the residual stream $x_t$. It starts as an embedding of the letter there plus one of the position, $E_{\text{letter}} + P_t$. Every head reads the stream and adds its result back, so after the last layer

$$ x_t = E_{\text{letter}} + P_t + y^{(1)}_t + y^{(2)}_t $$

where $y^{(1)}_t$ and $y^{(2)}_t$ are what the heads of layers 1 and 2 add.

A matrix $U$ turns this into one score per letter, $U_c \cdot x_t$ for letter $c$, and a softmax turns the scores into probabilities:

$$ p(c) = \frac{e^{U_c \cdot x_t}}{\sum_{c'} e^{U_{c'} \cdot x_t}} $$

Training lowers $-\ln p$ of the right letter in the repeats. A blind guess among 16 letters has loss $\ln 16 = 2.77$.

## An attention head

From position $t$ a head looks back at every position $s \le t$, scores each by a dot product, turns the scores into shares that add up to 1, and copies a weighted sum:

$$ A_{ts} = \operatorname{softmax}_{s \le t} \frac{(Q x_t) \cdot (K x_s)}{\sqrt{32}}, \qquad z_t = \sum_{s \le t} A_{ts} \, V x_s $$

$Q$, $K$, $V$ and $O$ are the head's matrices, and it adds $y_t = O z_t$ to the stream. Example: scores $2, 0, 0$ give shares $e^2 / (e^2 + 2) = 0.79$, $0.11$ and $0.11$.

## The induction head

Number positions from 1. In the row $\mathrm{C\,A\,M\,D\,H\,J\,C\,A\,M}$, the letter after position 9 must be $\mathrm{D}$. Two heads do it together:

1. In the first copy, the first-layer head looks at the letter before: at position 4 it writes "the letter before me was $\mathrm{M}$" into the stream.
2. The second-layer head, at the second $\mathrm{M}$, scores highest the position whose stream says "the letter before me was $\mathrm{M}$": position 4. It copies what is there, $\mathrm{D}$.

The second step needs what the first wrote, so one layer is not enough.

## Tools for reading it

- Logit lens: apply $U$ and the softmax to the stream after the embeddings, after layer 1 and after layer 2, to see at what depth the answer appears.
- Direct logit attribution: the stream is a plain sum and a score is a dot product, so each part's share of a letter's score adds up exactly to the whole. The scene shows each part's lead: its score for the right letter minus its average over all letters. Taking the same number off every score leaves the probabilities unchanged.
- Knocking out: replace a head's output by its average over many rows, or by nothing, and see how far the loss rises.

Patching: swap every earlier copy of the studied letter for one the row does not use, so that the model loses the answer. Put the undamaged stream back at one position and depth, and measure the share of the lost $\ln p$ that returns:

$$ \frac{\ln p_{\text{patched}} - \ln p_{\text{damaged}}}{\ln p_{\text{clean}} - \ln p_{\text{damaged}}} $$

kept between 0 and 1. Example: $p$ of $0.9$ clean, $0.15$ damaged and $0.6$ patched give $1.386 / 1.792 = 0.77$.

# In the picture

- The letters are the row; under them, the model's guess of each next letter (green right, red wrong) and how sure it is of the right one.
- The arcs: where each head looks from the ringed letter, one colour per head.
- The square maps: each head's $A_{ts}$, row $t$ looking from, column $s$ looked at. The red bar under one: how far the loss rises with it knocked out.
- The lens and patching rows: one square per position and depth, yellow near 1, dark purple near 0.
- The bars: what each part adds to the right letter's lead; they add up to the sum.
- The curves: loss and repeats right.

> Try first "watch it learn", then "see where the heads look".

see: mlp, superposition, probe, lyapunov

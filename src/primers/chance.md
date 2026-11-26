# The idea

A probability is a share: of 1 000 people, the share who have a medical condition, or who test positive for it. This scene draws each share as an area of a square, so that the rules of chance become rules of areas and counting. Its lesson: the share of positives who have the condition and the share of those with it who test positive are different numbers, and can be far apart. Fractions and percentages are all you need.

# The mathematics

## Events and their probabilities

An event is something that is true or not of each person: here $A$, "has the condition", and $B$, "tests positive". Its probability $P(A)$ is the share of everyone of whom it is true: if 10 of the 1 000 have the condition, $P(A) = 10/1000 = 1\%$. It is also the chance that a person picked at random, all equally likely, is in $A$. The rest are in the complement, $P(\text{not } A) = 1 - P(A)$. The joint probability $P(A \text{ and } B)$ is the share of whom both are true.

## A share of one group only

A conditional probability is a share within one group. $P(B \mid A)$, read "the probability of $B$ given $A$", is the share of the $A$'s who are also $B$'s (undefined if there are no $A$'s):

$$ P(B \mid A) = \frac{P(A \text{ and } B)}{P(A)} $$

The region of those in both is $P(A)$ wide and $P(B \mid A)$ high, so its area is

$$ P(A \text{ and } B) = P(A)\, P(B \mid A) $$

Two events are independent when $P(A \text{ and } B) = P(A)\,P(B)$. If some people are $A$'s and some not, that means $P(B \mid A) = P(B \mid \text{not } A)$: the level of $B$ is the same on both sides, and learning whether $A$ is true tells nothing about $B$.

## Bayes' rule is counting

$P(A \mid B)$ is the share of the positives who have the condition. The positives lie in two regions, with and without the condition, so

$$ P(A \mid B) = \frac{P(A)\, P(B \mid A)}{P(A)\, P(B \mid A) + P(\text{not } A)\, P(B \mid \text{not } A)} $$

This is Bayes' rule: the region in both, over the two regions of $B$. Three numbers go in: the base rate $P(A)$, the sensitivity $P(B \mid A)$ and the false alarm rate $P(B \mid \text{not } A)$.

Example, as the scene opens: $P(A) = 1\%$, $P(B \mid A) = 90\%$, $P(B \mid \text{not } A) = 9\%$. Of 1 000 people, 10 have it; 9 of them test positive. Of the other 990, 9 %, about 89, test positive anyway. So only 9 of about 98 positives have it: $P(A \mid B) = 9/98 \approx 9.2\%$. Most positives are false alarms: 99 times as many people lack the condition.

## Odds

The odds of a probability $p$ are $p : (1 - p)$; 1 % is $1 : 99$. Divide one region of $B$ by the other: the odds after a positive are the odds before times the likelihood ratio, $P(B \mid A) / P(B \mid \text{not } A)$.

$$ \frac{P(A \mid B)}{P(\text{not } A \mid B)} = \frac{P(A)}{P(\text{not } A)} \times \frac{P(B \mid A)}{P(B \mid \text{not } A)} $$

Here that ratio is $0.9 / 0.09 = 10$, so $1 : 99$ becomes $10 : 99$, which is $10/109 \approx 9.2\%$ again. A second test like the first, with mistakes independent of the first's, multiplies by 10 again: $100 : 99$, slightly over a half.

# In the picture

- The square is everyone, one dot a person. The left part, as wide as $P(A)$, holds those with the condition.
- In each part the bottom holds the positives, as high as that part's level: $P(B \mid A)$ on the left, $P(B \mid \text{not } A)$ on the right.
- Blue: found. Dim blue: missed. Pink: a false alarm. Grey: all clear.
- The green dots set $P(A)$ and the two levels: drag them.
- The bars on the right are $P(B \mid A)$ and $P(A \mid B)$; the white tick on each is the share among everyone, $P(B)$ and $P(A)$.

> Try first "a rare condition", and count how few of the positives are blue; then "a common condition", where the same test means something else.

see: distributions, bayes, pvalues

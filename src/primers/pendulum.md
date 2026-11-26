# The idea

A double pendulum is one pendulum hanging from the end of another. Its motion follows exact laws, with nothing left to chance, and yet two of them released from high up, a hair apart, soon swing in completely different ways. The scene releases hundreds at once to show this: chaos, in a machine you could build. You need the sine and cosine of an angle, and a derivative as a rate of change.

# The mathematics

## Four numbers say everything

The upper rod makes the angle $\theta_1$ with the straight-down direction, and the lower rod the angle $\theta_2$, both in radians. Write $\omega_1$ and $\omega_2$ for the rates at which they change, the angular velocities. Both rods have length $1$, and the lower bob's mass over the upper one's is $m_2/m_1$. With $m_1 = 1$ and the upper rod hanging from the pivot at $(0, 0)$, the two bobs are at

$$ (\sin\theta_1,\ -\cos\theta_1) \quad\text{and}\quad (\sin\theta_1 + \sin\theta_2,\ -\cos\theta_1 - \cos\theta_2) $$

Newton's laws turn the four numbers $\theta_1, \theta_2, \omega_1, \omega_2$ into the rates at which $\omega_1$ and $\omega_2$ change. The formulas are long, but the point is that the four numbers now fix every later moment. The scene moves each pendulum on in steps of a few thousandths of a second or less.

## Energy

The energy of a pendulum is its energy of motion plus $g$ times each mass times its height above the pivot (negative below it). Released at rest, it has only the second part:

$$ E = -(m_1 + m_2)\, g \cos\theta_1 - m_2\, g \cos\theta_2 $$

Without friction $E$ stays the same however wild the motion, because nothing takes energy out. Friction, in the scene, brakes each rod's turning in proportion to its angular velocity, and can only take energy away.

Example: the scene starts with $\theta_1 = 2.2$, $\theta_2 = 2.6$, $g = 9.81$ and equal masses, so

$$ E = -2 \cdot 9.81 \cdot \cos 2.2 - 9.81 \cdot \cos 2.6 \approx 11.546 + 8.406 \approx 19.95 $$

the energy shown in the panel. Hanging still, with both angles $0$, it would be $-2 \cdot 9.81 - 9.81 = -29.43$, the value that friction drains it down to.

## A difference that grows

The 600 pendulums differ only in $\theta_2$, by a thousandth of a radian from first to last. While two of them are close, their difference changes in proportion to itself: halve the starting difference and, for a while, the whole later difference halves too. In wild swinging it grows on average by the same factor every second, as money grows at compound interest: exponentially. So starting a thousand times closer only adds the time a small difference needs to grow a thousandfold, and then the pendulums part all the same. The scene shows this: they part after about four and a half seconds when a thousandth apart, and after about nine when a millionth apart.

Released low, with little energy, the motion stays regular, close to a sum of two steady swings, and a small difference grows only slowly, if at all.

# In the picture

- Each coloured trail is the path of one pendulum's lower bob: blue for the first, red for the last, the colours between for those between.
- The white rods are those of the first, middle and last pendulum; the grey dot is the pivot.
- "Energy of the first pendulum" is $E$; "tip separation" is the distance between the first and last lower bobs, in rod lengths.

> Try first "released together": the 600 swing as one for about four and a half seconds, then the trails fan out into a rainbow.

see: attractor, lyapunov, waves

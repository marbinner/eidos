# The idea

A closed curve without breaks, such as any loop you could draw by hand in one stroke, can be drawn as closely as you like by a pen riding on a chain of turning circles. Each circle turns at a steady speed, and the next one rides on the tip of its arm; the tip of the last arm is the pen. Big, slow circles usually give the rough shape and small, fast ones the detail. Taking a curve apart into circles like this is a Fourier series. You need the sine and cosine of an angle, and the idea of an average.

# The mathematics

## One circle

Let $t$ run from $0$ to $2\pi$ while the pen goes once round the shape. An arm of length $r$ that starts at the angle $\varphi$ and turns $k$ times during that trip has its tip at

$$ r\,\big(\cos(kt + \varphi),\ \sin(kt + \varphi)\big) $$

The number $k$ is the circle's frequency: positive turns anticlockwise, negative clockwise. It is a whole number, so that every arm is back where it began when the pen is, and the curve closes.

## A chain of circles

Put the arms end to end, starting from a centre point $c$. Then the pen is at

$$ p(t) = c + \sum_{k \ne 0} r_k \big(\cos(kt + \varphi_k),\ \sin(kt + \varphi_k)\big) $$

with one term for each circle: its radius $r_k$, its frequency $k$ and its starting angle $\varphi_k$.

## Finding the circles

Suppose a shape is drawn by circles. To find circle $k$, turn the drawing backwards, $k$ turns per trip, and average the pen's position over the trip. Circle $k$ now stands still, so it adds its arm as it was at the start. Every other circle $j$ still turns, $j - k$ whole times, and a point that goes steadily round a circle a whole number of times averages to the circle's centre: it adds nothing. So the average is the arm of circle $k$, which gives $r_k$ and $\varphi_k$ (for $k = 0$ it is $c$). The scene does this for any shape, with 512 points taken in order round it and every frequency from $-96$ to $96$, and puts the biggest circles first.

The same idea measures what some circles capture. Over a trip, the average squared distance of the pen from $c$ is the sum of the squared radii,

$$ \text{average of } |p(t) - c|^2 = \sum_{k \ne 0} r_k^2 $$

because the angle between two arms of different frequencies turns a whole number of times, so the cross terms, which carry its cosine, average to zero.

A sharp corner needs circles of every size: the square's have frequencies $1, -3, 5, -7, \ldots$ and radii that shrink only like $1/k^2$, so any number of them leaves the corners a little round.

Example: the scene's trefoil is the curve

$$ \big(0.72\,(\sin 2t + 2\sin t),\ 0.72\,(\cos 2t - 2\cos t)\big) $$

Its part $0.72\,(2\sin t, -2\cos t)$ is a circle of radius $1.44$ turned once anticlockwise: at $t = 0$ it points down, to $(0, -1.44)$, and at $t = \pi/2$ right, to $(1.44, 0)$. Its part $0.72\,(\sin 2t, \cos 2t)$ is a circle of radius $0.72$ turned twice clockwise: up at $t = 0$, right at $t = \pi/4$. So the trefoil is frequencies $1$ and $-2$ alone, and the bigger circle captures

$$ \frac{1.44^2}{1.44^2 + 0.72^2} = \frac{4}{4 + 1} = 80\% $$

# In the picture

- The white lines are the arms, one for each radius $r_k$, the biggest first, starting from the centre $c$. The purple circles are what each arm sweeps.
- The yellow dot is the pen $p(t)$. The cyan curve is what the circles in use draw in one trip, bright where the pen has been.
- The grey outline is the shape itself.
- The "circles" slider sets how many circles are kept, and "share of the shape captured" is the sum of their $r_k^2$ out of the sum over all of them.

> Try first "two circles make a trefoil": one circle draws a plain circle (80 % of the shape), and the second, turning twice the other way, completes it exactly.

see: parametric, dft, waves

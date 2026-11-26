# The idea

The scene maps the kinds of study that ask whether vaccination against shingles changes the risk of dementia, and which paths of a causal graph each kind closes. This primer explains their methods; like the scene, it gives no verdict on any vaccine and is not medical advice. It assumes causal graphs: back doors, and colliders, where two arrows meet.

# The mathematics

## What is counted

$V$ is vaccinated, $D$ dementia and $D^*$ its recorded diagnosis, a dated code. $D$ itself is never observed directly, so a difference in $D^*$ is a difference in $D$ only if diagnosis is as complete, and as early, in the groups compared. The hazard ratio is

$$ \text{HR} = \frac{h_1(t)}{h_0(t)} $$

where $h_1(t)$ and $h_0(t)$ are the rates at which people not yet diagnosed are diagnosed, $t$ years into follow-up, in each group: 1 means equal rates, 0.8 a rate a fifth lower.

## Who chooses vaccination

Comparing vaccinated with unvaccinated people can leave open a back door, $V \gets H \to D$, through health and engagement $H$. With made-up numbers: robust people are diagnosed at 1 in 100 a year, frail ones at 3, vaccination changes nothing, and the vaccinated are 75% robust, the unvaccinated 25%. Their rates are

$$ 0.75 \cdot 1 + 0.25 \cdot 3 = 1.5 \qquad 0.25 \cdot 1 + 0.75 \cdot 3 = 2.5 $$

a ratio of 0.6, made only by who chose it: the healthy-vaccinee effect. Adjusting for $H$ closes the door only as far as the records measure health. An active comparator, a group given another vaccine, narrows it: at 70% robust its rate is 1.6, and the ratio $1.5 / 1.6 = 15/16$, provided that vaccine has no effect on dementia of its own.

Immortal time is follow-up in which, by how exposure is defined, one group's outcome cannot be counted. If people count as vaccinated from the start of follow-up though their dose came later, their time before it holds no diagnoses (anyone diagnosed first left follow-up unvaccinated), so vaccination looks protective even if it does nothing. Target-trial emulation, one start of follow-up for everyone as in a trial, fixes such timing, not unmeasured confounding.

## A threshold that decides

Where eligibility $E$ depends on date of birth, a regression discontinuity compares people born just either side of the cut-off. Nobody chooses their birthday, so no arrow enters $E$, and on $E \to V \gets H \to D$ two arrows meet at $V$, which is not held, closing the path. If nothing else affecting recorded dementia changes at the cut-off, the jump in the share diagnosed is the effect of eligibility (intention to treat, ITT), and

$$ \text{effect per person vaccinated} = \frac{\text{jump in share diagnosed}}{\text{jump in share vaccinated}} $$

among those vaccinated because eligible, if eligibility acts only through vaccination, nobody is vaccinated because ineligible, and uptake is measured accurately. With made-up numbers: 10.0% are diagnosed just before the cut-off, 9.0% just after, and uptake jumps by 40 percentage points, so $1.0 / 0.4 = 2.5$ points fewer: a quarter of their risk, if that is 10%. Had uptake jumped by 5 points, the same 1.0 would mean 20 points, twice that risk: more than 100%, impossible if the assumptions hold.

A randomised trial cuts every arrow into $V$; a blinded placebo, giving both arms the same visits, also cuts $V \to K$ (health-care contact).

# In the picture

- A double ring: what a design compares; a dashed ring: not in the data. A box: held (matched, adjusted, or alike in everyone compared); a dashed box: only as far as the records go.
- Green: a path through dementia, what a design estimates. Pink: a path giving a difference in diagnoses without any change in dementia; faint, closed in part; dashed with ?, assumed absent. Grey: closed; a stroke across it: cut by randomisation.
- The table: designs in rows, explanations in columns. A full disc, a half, a ring: the design can tell that explanation apart, partly, or not.

> Try first "the healthy vaccinee": the faint pink back door is closed by adjustment only in part. Then "a birthday decides", where it is grey.

see: confound, junctions, adjust, switch

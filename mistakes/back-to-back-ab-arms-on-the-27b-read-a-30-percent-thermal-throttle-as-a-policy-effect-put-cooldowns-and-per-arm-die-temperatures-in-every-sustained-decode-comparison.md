# Back-to-back A/B arms on the 27B read a 30 percent thermal throttle as a policy effect. Put cooldowns and per-arm die temperatures in every sustained-decode comparison

**Symptom:** Two ABBA depth-policy pairs on the 27B (2026-09-06, C15d and
C15e, 19k context, 3k tokens per arm) decoded 35.6, 29.6, 24.9 and 25.2
tok/s in arm order. The first pair's half result read as "the policy wins
by 17 percent"; the full pair still read as +12 percent. Both were the
arm order, not the policy.

**Cause:** At max fans a single 75 s decode arm takes the package sensors
from about 45 C to 85 C (Tp0C) and 55 C to 95 C (TCMb). Arms run
back-to-back never let them fall, the chip throttles, and every later arm
is slower regardless of its setting. ABBA cancels a linear drift, not a
saturating one. Four identical arms with 120 s idle between them held at
41 to 42.5 tok/s with the sensors recovering in the pause (C15f).

**Fix / rule:** A sustained-decode comparison on this machine logs the die
temperatures before and after every arm and idles until they are back at
the starting value (two minutes here) before the next arm; a verdict is
valid only when adjacent arms of the same setting agree. The depth-policy
verdicts that stand are the ones where that held (Flash-Next 7 to 8
percent slower with the policy; 27B neutral). Twenty-five minutes of GPU
were spent before the sensors were read.

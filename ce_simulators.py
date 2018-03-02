#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Classes for C. elegans behavior simulations
Behavior is modeled after data in (Ryu, 2002)
Note that behavior is merely approximated as the goal is not to derive a realistic simulation of C elegans behavior
"""

import numpy as np
from core import FRAME_RATE, HIST_SECONDS, PRED_WINDOW
from core import RandCash, GradientData, GradientStandards, ZfGpNetworkModel, indexing_matrix

# Run-speed: 16 +/- 2 mm/min
# Model as appropriately per-step scaled gaussian: ~ N(16 mm/min, 2 mm/min)

# Run-durations: Exponentially distributed. avg=14s up gradient, avg=25s down gradient
# To roughly support these durations evaluate our model with a probability that matches 1/10 Hz (so that if reversal and
# sharp turn are the two top behaviors we will get a termination about once in 14 seconds. If they are bottom two
# runs will only be terminated once in 50 s however

# To introduce curvature into runs, add scaled angle at every step ~ N(0, 1deg/s)

# Runs terminated by undirectional sharp turns or reversals
# Model sharp turns as gaussian ~ N(45 degrees, 5 degrees)
# Model reversals as gaussian ~ N(180 degrees, 10 degrees)

# To potentially allow for isothermal tracking introduce two shallow *directional* turns
# Small-turn right modeled as N ~ (5 deg/s, 1 deg/s)
# Small-turn left modeled as N ~ (-5 deg/s, 1 deg/s)

# => Behaviors: Continue-run; Sharp turn (random dir); Reverse; Small turn left; Small turn right
# Rank evaluations: Top->50%; 2nd->20%; 3rd-5th->10%

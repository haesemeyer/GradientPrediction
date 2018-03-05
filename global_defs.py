#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Module to aggregate standard parameters used for simulations etc.
"""


class GlobalDefs:
    def __init__(self):
        raise NotImplementedError("Static class")

    tPreferred = 26

    n_steps = 2000000

    frame_rate = 100  # all simulations, training data models have a native framerate of 100 Hz

    hist_seconds = 4  # all inputs to the models are 4s into the past

    model_rate = 5  # model input rate is 5 Hz

    circle_sim_params = {"radius": 100, "t_min": 22, "t_max": 37, "t_preferred": tPreferred}

    rev_circle_sim_params = {"radius": 100, "t_min": 37, "t_max": 22, "t_preferred": tPreferred}

    lin_sim_params = {"xmax": 100, "ymax": 100, "t_min": 22, "t_max": 37, "t_preferred": tPreferred}

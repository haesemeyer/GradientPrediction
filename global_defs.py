#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Module to aggregate standard parameters used for simulations etc.
"""

import ce_simulators
import zf_simulators
from core import CeGpNetworkModel, ZfGpNetworkModel


class GlobalDefs:
    def __init__(self):
        raise NotImplementedError("Static class")

    tPreferred = 26

    n_steps = 2000000

    circle_sim_params = {"radius": 100, "t_min": 22, "t_max": 37, "t_preferred": tPreferred}

    rev_circle_sim_params = {"radius": 100, "t_min": 37, "t_max": 22, "t_preferred": tPreferred}

    lin_sim_params = {"xmax": 100, "ymax": 100, "t_min": 22, "t_max": 37, "t_preferred": tPreferred}


class MoTypes:
    """
    Class with properties aggregating model organism types
    """
    def __init__(self, c_elegans=False):
        """
        Creates a new MoTypes object
        :param c_elegans: Determines whether served types will be C elegans or zebrafish types
        """
        self.c_elegans = c_elegans

    @property
    def network_model(self):
        if self.c_elegans:
            return CeGpNetworkModel
        else:
            return ZfGpNetworkModel

    @property
    def rad_sim(self):
        if self.c_elegans:
            return ce_simulators.CircleGradSimulation
        else:
            return zf_simulators.CircleGradSimulation

    @property
    def lin_sim(self):
        if self.c_elegans:
            return ce_simulators.LinearGradientSimulation
        else:
            return zf_simulators.LinearGradientSimulation

    @property
    def wn_sim(self):
        if self.c_elegans:
            raise NotImplementedError("C elegans white noise simulation not implemented yet")
        else:
            return zf_simulators.WhiteNoiseSimulation

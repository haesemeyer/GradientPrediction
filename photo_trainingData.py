#  Copyright 2018 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Module to generate, save and load training and test data for phototaxis simulations
"""

import numpy as np
from core import GradientData
from zf_simulators import TrainingSimulation


class CircGradientTrainer(TrainingSimulation):
    """
    Class to run a gradient simulation, creating training data in the process
    The simulation will at each timepoint simulate what could have happened
    500 ms into the future for each selectable behavior but only one path
    will actually be chosen to advance the simulation to avoid massive branching
    """
    def __init__(self, radius):
        """
        Creates a new GradientSimulation
        :param radius: The arena radius in mm
        """
        super().__init__()
        self.radius = radius

    def temperature(self, x, y, a=0):
        """
        Returns the angle to the central light-source at the given positions
        in a coordinate system centered on the object, facing in the heading direction
        """
        # calculate vector *to* source which is at coordinate 0/0
        vec_x = -x
        vec_y = -y
        # calculate heading unit vector
        head_x = np.cos(a)
        head_y = np.sin(a)
        # calculate and return angular position of source
        return np.arctan2(vec_y, vec_x) - np.arctan2(head_y, head_x)

    def out_of_bounds(self, x, y):
        """
        Detects whether the given x-y position is out of the arena
        :param x: The x position
        :param y: The y position
        :return: True if the given position is outside the arena, false otherwise
        """
        # circular arena, compute radial position of point and compare to arena radius
        r = np.sqrt(x**2 + y**2)
        return r > self.radius


if __name__ == '__main__':
    import matplotlib.pyplot as pl
    import seaborn as sns
    response = ""
    while response not in ["y", "n"]:
        response = input("Run simulation with default arena? [y/n]:")
    if response == "y":
        n_steps = int(input("Number of steps to perform?"))
        gradsim = CircGradientTrainer(100)
        print("Running radial simulation")
        pos = gradsim.run_simulation(n_steps)
        pl.figure()
        pl.plot(pos[:, 0], pos[:, 1])
        pl.xlabel("X position [mm]")
        pl.ylabel("Y position [mm]")
        sns.despine()
        print("Generating gradient data")
        grad_data = gradsim.create_dataset(pos)
        print("Done")

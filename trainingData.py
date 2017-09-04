#  Copyright 2017 Martin Haesemeyer. All rights reserved.
#
# Licensed under the MIT license

"""
Module to generate, save and load training and test data and provide access to that data in convenient batches
"""

import numpy as np
from core import TrainingSimulation


class CircGradientTrainer(TrainingSimulation):
    """
    Class to run a gradient simulation, creating training data in the process
    The simulation will at each timepoint simulate what could have happened
    500 ms into the future for each selectable behavior but only one path
    will actually be chosen to advance the simulation to avoid massive branching
    """
    def __init__(self, radius, t_min, t_max):
        """
        Creates a new GradientSimulation
        :param radius: The arena radius in mm
        :param t_min: The center temperature
        :param t_max: The edge temperature
        """
        super().__init__()
        self.radius = radius
        self.t_min = t_min
        self.t_max = t_max

    def temperature(self, x, y):
        """
        Returns the temperature at the given positions
        """
        r = np.sqrt(x**2 + y**2)  # this is a circular arena so compute radius
        return (r / self.radius) * (self.t_max - self.t_min) + self.t_min

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
        gradsim = CircGradientTrainer(100, 22, 37)
        print("Running gradient simulation")
        pos = gradsim.run_simulation(n_steps)
        pl.figure()
        pl.plot(pos[:, 0], pos[:, 1])
        pl.xlabel("X position [mm]")
        pl.ylabel("Y position [mm]")
        sns.despine()
        print("Generating gradient data")
        grad_data = gradsim.create_dataset(pos)
        print("Done")

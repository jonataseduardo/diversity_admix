"""
  Helper functions and classes for simulation of admixed individuals
"""

import numpy as np
import msprime as msp


class Simulation(object):
    """
    Parameters for a simulation
    """

    def __init__(self):
        # Defining effective population sizes
        self.Ne = None

        # Defining samples in this case
        self.samples = None

        # Define demographic events to sample across
        self.demography = None

        # Define population configuration
        self.pop_config = None

    def simulate(self, **kwargs):
        """
        Conduct a simulation using msprime and the demographic paramters we
        have established
        """
        tree_seq = msp.simulate(
            Ne=self.Ne,
            samples=self.samples,
            demographic_events=self.demography,
            population_configurations=self.pop_config,
            **kwargs
        )
        return tree_seq


class DivergenceAdmixture(Simulation):
    """
    Runs the simulation
    """

    def __init__(self, Na, Nb, Nc, t_div=30, t_adm=0.0, n=50, alpha=0.5, eps=1e-8):
        """
            Class defining two-population coalescent model where we have
            admixture between
        """
        super().__init__()

        # Define effective population size
        self.Ne = Na

        # Setup samples here
        samples = [msp.Sample(population=0, time=0) for _ in range(n)]
        samples = samples + [msp.Sample(population=1, time=0) for _ in range(n)]
        samples = samples + [msp.Sample(population=2, time=0) for _ in range(n)]
        self.samples = samples

        # Define population configuration
        self.pop_config = [
            msp.PopulationConfiguration(initial_size=Na),
            msp.PopulationConfiguration(initial_size=Nc),
            msp.PopulationConfiguration(initial_size=Nb),
        ]

        # Define a mass migration of all the lineages post-split
        self.demography = [
            msp.MassMigration(time=t_adm, source=1, dest=2, proportion=1 - alpha),
            msp.MassMigration(time=t_adm + eps, source=1, dest=0),
            msp.MassMigration(time=t_div + t_adm, source=2, dest=0),
        ]

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


class OutOfAfrica(Simulation):
    """
    Runs Out Africa Simulation with Jouganous 2017 parameters
    """

    def __init__(
        self,
        N_A=7300,
        N_B=2100,
        N_AF=12300,
        N_EU0=1000,
        N_AS0=510,
        generation_time=25,
        T_AF_years=220e3,
        T_B_years=140e3,
        T_EU_AS_years=21.2e3,
        # We need to work out the starting (diploid) population sizes based on
        # the growth rates provided for these two populations
        r_EU=0.004,
        r_AS=0.0055,
        # Migration rates during the various epochs.
        m_AF_B=25e-5,
        m_AF_EU=3e-5,
        m_AF_AS=1.9e-5,
        m_EU_AS=9.6e-5,
    ):
        super().__init__()
        self.Ne = N_A
        self.T_AF = T_AF_years / generation_time
        self.T_B = T_B_years / generation_time
        self.T_EU_AS = T_EU_AS_years / generation_time
        self.N_EU = N_EU0 / np.exp(-r_EU * self.T_EU_AS)
        self.N_AS = N_AS0 / np.exp(-r_AS * self.T_EU_AS)

        self.pop_config = [
            msp.PopulationConfiguration(sample_size=0, initial_size=N_AF),
            msp.PopulationConfiguration(sample_size=1, initial_size=self.N_EU, growth_rate=r_EU),
            msp.PopulationConfiguration(sample_size=1, initial_size=self.N_AS, growth_rate=r_AS),
        ]

        self.migration_matrix = [
            [0, m_AF_EU, m_AF_AS],
            [m_AF_EU, 0, m_EU_AS],
            [m_AF_AS, m_EU_AS, 0],
        ]
        self.demography = [
            # CEU and CHB merge into B with rate changes at T_EU_AS
            msp.MassMigration(time=self.T_EU_AS, source=2, destination=1, proportion=1.0),
            msp.MigrationRateChange(time=self.T_EU_AS, rate=0),
            msp.MigrationRateChange(time=self.T_EU_AS, rate=m_AF_B, matrix_index=(0, 1)),
            msp.MigrationRateChange(time=self.T_EU_AS, rate=m_AF_B, matrix_index=(1, 0)),
            msp.PopulationParametersChange(
                time=self.T_EU_AS, initial_size=N_B, growth_rate=0, population_id=1
            ),
            # Population B merges into YRI at T_B
            msp.MassMigration(time=self.T_B, source=1, destination=0, proportion=1.0),
            # Size changes to N_A at T_AF
            msp.PopulationParametersChange(time=self.T_AF, initial_size=N_A, population_id=0),
        ]

        self.dd = msp.DemographyDebugger(
            population_configurations=self.pop_config,
            migration_matrix=self.migration_matrix,
            demographic_events=self.demography,
        )


class OutOfAfricaAdmixture(Simulation):
    """
    Runs Out Africa Simulation with Jouganous 2017 parameters
    """

    def __init__(
        self,
        N_AX=1000,
        N_A=7300,
        N_B=2100,
        N_AF=12300,
        N_EU0=1000,
        N_AS0=510,
        generation_time=25,
        T_AF_years=220e3,
        T_B_years=140e3,
        T_EU_AS_years=21.2e3,
        # We need to work out the starting (diploid) population sizes based on
        # the growth rates provided for these two populations
        r_EU=0.004,
        r_AS=0.0055,
        # Migration rates during the various epochs.
        m_AF_B=25e-5,
        m_AF_EU=3e-5,
        m_AF_AS=1.9e-5,
        m_EU_AS=9.6e-5,
        n=50,
        alpha1=0.5,
        alpha2=0.5,
        t_adm=0,
        eps=1e-8,
        debug=False,
    ):
        super().__init__()

        # alpha2 is the proportion of AS population
        self.Ne = N_A
        self.alpha2 = alpha2
        # alpha1 is the proportion of YRI population and self.alpha1 is the proportion of
        # CEU pop for the msprime 2 sources admixture process
        self.alpha1 = (1 - alpha1 - alpha2) / (1 - alpha2)

        self.T_AF = T_AF_years / generation_time
        self.T_B = T_B_years / generation_time
        self.T_EU_AS = T_EU_AS_years / generation_time
        self.N_EU = N_EU0 / np.exp(-r_EU * self.T_EU_AS)
        self.N_AS = N_AS0 / np.exp(-r_AS * self.T_EU_AS)

        samples = [msp.Sample(population=0, time=0) for _ in range(n)]
        samples = samples + [msp.Sample(population=1, time=0) for _ in range(n)]
        samples = samples + [msp.Sample(population=2, time=0) for _ in range(n)]
        samples = samples + [msp.Sample(population=3, time=0) for _ in range(n)]
        self.samples = samples

        self.pop_config = [
            msp.PopulationConfiguration(initial_size=N_AF),
            msp.PopulationConfiguration(initial_size=N_AF),
            msp.PopulationConfiguration(initial_size=self.N_EU, growth_rate=r_EU),
            msp.PopulationConfiguration(initial_size=self.N_AS, growth_rate=r_AS),
        ]

        self.migration_matrix = [
            [0, 0, m_AF_EU, m_AF_AS],
            [0, 0, 0, 0],
            [m_AF_EU, 0, 0, m_EU_AS],
            [m_AF_AS, 0, m_EU_AS, 0],
        ]
        self.demography = [
            # Admixture events
            msp.MassMigration(time=t_adm, source=1, dest=3, proportion=self.alpha2),
            msp.MassMigration(time=t_adm + eps, source=1, dest=2, proportion=self.alpha1),
            msp.MassMigration(time=t_adm + 1.1 * eps, source=1, dest=0),
            # CEU and CHB merge into B with rate changes at T_EU_AS
            msp.MassMigration(time=self.T_EU_AS, source=2, destination=1, proportion=1.0),
            msp.MigrationRateChange(time=self.T_EU_AS, rate=0),
            msp.MigrationRateChange(time=self.T_EU_AS, rate=m_AF_B, matrix_index=(0, 1)),
            msp.MigrationRateChange(time=self.T_EU_AS, rate=m_AF_B, matrix_index=(1, 0)),
            msp.PopulationParametersChange(
                time=self.T_EU_AS, initial_size=N_B, growth_rate=0, population_id=1
            ),
            # Population B merges into YRI at T_B
            msp.MassMigration(time=self.T_B, source=1, destination=0, proportion=1.0),
            # Size changes to N_A at T_AF
            msp.PopulationParametersChange(time=self.T_AF, initial_size=N_A, population_id=0),
        ]

        if debug:
            self.dd = msp.DemographyDebugger(
                population_configurations=self.pop_config,
                migration_matrix=self.migration_matrix,
                demographic_events=self.demography,
            )

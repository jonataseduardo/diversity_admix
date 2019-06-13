"""
  Helper functions and classes for simulation of admixed individuals
"""

import numpy as np
import msprime as msp


class Simulation():
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

    def _simulate(self, **kwargs):
        """
        Conduct a simulation using msprime and the demographic paramters we
        have established
        """
        tree_seq = msp.simulate(Ne=self.Ne,
                                samples=self.samples,
                                demographic_events=self.demography,
                                population_configurations=self.pop_config,
                                **kwargs)
        return tree_seq

    def _ploidy_genotype_matrix(self, tree_seq, ploidy=2):
        """
        Function to generate diploid genotype matrices from TreeSequences
        """
        geno = tree_seq.genotype_matrix().T
        num_haps, num_snps = geno.shape
        assert num_haps % ploidy == 0
        true_n = int(num_haps/2)
        true_geno = np.zeros(shape=(true_n, num_snps))
        for i in range(true_n):
            snp_x = ploidy*i
            snp_y = ploidy*i + ploidy
            true_geno[i] = np.sum(geno[snp_x:snp_y, :], axis=0)
        return true_geno


class DivergenceAdmixture(Simulation):
    """
    Runs the simulation
    """

    def __init__(self, Na, Nb, Nc, t_div=30, t_adm=0.0,
                 n=50, alpha=0.5, eps=1e-8):
        """
            Class defining two-population coalescent model where we have
            admixture between
        """
        super().__init__()
        assert t_div > t_adm

        # Define effective population size
        self.Ne = Na

        # Setup samples here
        samples = [msp.Sample(population=0, time=0) for _ in range(n)]
        samples = samples + [msp.Sample(population=1, time=0)
                             for _ in range(n)]
        samples = samples + [msp.Sample(population=2, time=0)
                             for _ in range(n)]
        self.samples = samples

        # Define population configuration
        self.pop_config = [msp.PopulationConfiguration(initial_size=Na),
                           msp.PopulationConfiguration(initial_size=Nc),
                           msp.PopulationConfiguration(initial_size=Nb)]

        # Define a mass migration of all the lineages post-split
        self.demography = [msp.MassMigration(time=t_adm, source=1,
                                             dest=2, proportion=1-alpha),
                           msp.MassMigration(time=t_adm+eps, source=1,
                                             dest=0),
                           msp.MassMigration(time=t_div, source=2, dest=0)]


class CoalSimUtils:
    """
    Functions to calculate statistics of interest
    """

    def seg_sites_pops(self, tree_seq):
        """
          Calculates segregating sites per population for a given tree sequence

          Arguments
          ---------
          tree_seq : msprime.TreeSequence
              tree sequence object

          Returns
          -------
          seg-sites: np.array
            vector of npop x 1 with segregating sites info

        """
        npops = tree_seq.num_populations
        seg_sites = np.zeros(npops, dtype=np.uint16)
        for j in range(npops):
            ts_simp_pop = tree_seq.simplify(tree_seq.samples(population=j))
            seg_sites[j] = ts_simp_pop.num_sites
        return seg_sites

    def nucleotide_div_pops(self, tree_seq):
        """
         Calculates nucleotide diversity within each population separately

         Arguments
         ---------
         tree_seq : msprime.TreeSequence
             tree sequence object

         Returns
         -------
         pair_div : np.array
           vector of npop x 1 with estimated pairwise diversity

        """
        npops = tree_seq.num_populations
        pair_div = np.zeros(npops, dtype=np.float32)
        for j in range(npops):
            pair_div[j] = tree_seq.pairwise_diversity(tree_seq.samples(population=j))
        return pair_div

    def total_branch_length_pops(self, tree_seq):
        """
         Calculates nucleotide diversity within each population separately

         Arguments
         ---------
         tree_seq : msprime.TreeSequence
             tree sequence object

         Returns
         -------
         branch_length : np.array
           vector of npop x 1 with estimated total branch length

        """
        npops = tree_seq.num_populations
        branch_length_pop = np.zeros(npops, dtype=np.float32)
        for j in range(npops):
            ts_simp = tree_seq.simplify(tree_seq.samples(population=j)).first()
            branch_length_pop[j] = ts_simp.get_total_branch_length()
        return branch_length_pop

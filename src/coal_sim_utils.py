"""
Statistics for msprime.TreeSequence
"""
import numpy as np


def seg_sites_pops(tree_seq):
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
        # ts_simp_pop = tree_seq.simplify(tree_seq.samples(population=j))
        # seg_sites[j] = ts_simp_pop.num_sites
        seg_sites[j] = tree_seq.segregating_sites(
            tree_seq.samples(population=j), span_normalise=False
        )
    return seg_sites


def nucleotide_div_pops(tree_seq):
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


def total_branch_length_pops(tree_seq):
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


def ploidy_genotype_matrix(tree_seq, ploidy=2):
    """
    Function to generate diploid genotype matrices from TreeSequences
    """
    geno = tree_seq.genotype_matrix().T
    num_haps, num_snps = geno.shape
    assert num_haps % ploidy == 0
    true_n = int(num_haps / 2)
    true_geno = np.zeros(shape=(true_n, num_snps))
    for i in range(true_n):
        snp_x = ploidy * i
        snp_y = ploidy * i + ploidy
        true_geno[i] = np.sum(geno[snp_x:snp_y, :], axis=0)
    return true_geno


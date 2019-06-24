import numpy as np
import msprime
import coal_theory_utils as ctu


def segregating_sites_example(n=100, Ne=1e4, mutation_rate=1e-3, length=1e4, num_replicates=10):

    S = np.zeros(num_replicates)
    L = np.zeros(num_replicates)
    theta = 2 * mutation_rate * length

    replicates = msprime.simulate(
        Ne=0.5,
        sample_size=n,
        mutation_rate=mutation_rate,
        length=length,
        num_replicates=num_replicates,
    )

    for j, tree_sequence in enumerate(replicates):
        S[j] = tree_sequence.num_sites
        L[j] = tree_sequence.first().get_total_branch_length()

    # Now, calculate the analytical predictions
    L_mean_a = 2 * np.sum(1 / np.arange(1, n))
    L_mean_ap = 2 * np.log(n - 1)

    LT1 = ctu.branch_length(n, Ne, Ne) / Ne
    nlin = ctu.nlinages(n, Ne, Ne)
    LT2 = 2 * np.log(nlin)
    LT = LT1 + LT2

    ST1 = theta * LT1 / 2
    ST2 = theta * LT2 / 2
    ST = theta * LT / 2

    S_mean_a = np.sum(1 / np.arange(1, n)) * theta
    S_mean_ap = np.log(n - 1) * theta

    L_var_a = np.sum(1 / np.arange(1, n)) + np.sum(1 / np.arange(1, n) ** 2)

    S_var_a = theta * np.sum(1 / np.arange(1, n)) + theta ** 2 * np.sum(1 / np.arange(1, n) ** 2)

    print(" Seg sites")
    print("              mean              variance")
    print("Observed      {:.2f}\t\t{:.2f}".format(np.mean(S), np.var(S)))
    print("Analytical    {:.2f}\t\t{:.2f}".format(S_mean_a, S_var_a))
    print("Aprox 1       {:.2f}".format(S_mean_ap))

    print("\n Branch Length")
    print("              mean              variance")
    print("Observed      {:.2f}\t\t{:.2f}".format(np.mean(L), np.var(L)))
    print("Analytical    {:.2f}\t\t{:.2f}".format(L_mean_a, L_var_a))
    print("Aprox 1       {:.2f}".format(L_mean_ap))

    print("\n Branch Length Partition Aprox")
    print("{:.2f}\t{:.2f}\t{:.2f}".format(LT, LT1, LT2))

    print("\n Seg Sites Partition Aprox")
    print("{:.2f}\t{:.2f}\t{:.2f}".format(ST, ST1, ST2))


segregating_sites_example(n=100, Ne=1e4, mutation_rate=1e-3, length=1e4, num_replicates=10)

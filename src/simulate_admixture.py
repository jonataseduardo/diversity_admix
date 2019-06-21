import numpy as np
from itertools import product
from sim_engine_admix import DivergenceAdmixture
import coal_sim_utils as csu
import datetime
import pandas as pd


def run_admix(
    Na=1e4,
    Nb=1e4,
    Nc=1e4,
    n=100,
    t_div=1e4,
    t_adm=0,
    alpha=0.2,
    length=1e4,
    mutation_rate=1e-8,
    num_replicates=500,
):

    admix_sim = DivergenceAdmixture(Na=Na, Nb=Nb, Nc=Nc, t_adm=t_adm, t_div=t_div, alpha=alpha, n=n)

    # 1b. Simulating 1000 replicates of 10 kb
    ts_test_ss = admix_sim.simulate(
        length=length, mutation_rate=mutation_rate, num_replicates=num_replicates
    )

    output = np.zeros([6, 3])

    # 1c. Calculate segregating sites across all the populations
    simul_trees = [ts for ts in ts_test_ss]
    seg_sites_admix = [csu.seg_sites_pops(ts) for ts in simul_trees]
    seg_sites_admix = np.vstack(seg_sites_admix).T
    output[0] = seg_sites_admix.mean(axis=1)
    output[1] = seg_sites_admix.var(axis=1)

    htz_admix = [csu.nucleotide_div_pops(ts) for ts in simul_trees]
    htz_admix = np.vstack(htz_admix).T

    output[2] = htz_admix.mean(axis=1)
    output[3] = htz_admix.var(axis=1)

    branch_length_admix = [csu.total_branch_length_pops(ts) for ts in simul_trees]
    branch_length_admix = np.vstack(branch_length_admix).T

    output[4] = branch_length_admix.mean(axis=1)
    output[5] = branch_length_admix.var(axis=1)

    return output.reshape(-1)


def main():
    """TODO: Docstring for main.
    :returns: TODO

    """
    Na = 1e4
    t_div_list = np.linspace(0, 2 * Na, 2)[1:]
    Nb_list = np.linspace(0, Na, 2)[1:]
    alpha_list = np.arange(1, 10, 5) / 10.0
    t_div_list / (2 * Na)
    num_samples = 100

    # Na = 1e4
    # t_div_list = np.linspace(0, 2 * Na, 41)[1:]
    # Nb_list = np.linspace(0, Na, 21)[1:]
    # alpha_list = np.arange(1, 10, 1) / 10.0
    # t_div_list / (2 * Na)
    # num_samples = 100

    simul_lenght = len(t_div_list) * len(Nb_list) * len(alpha_list)
    simul_params = np.zeros([simul_lenght, 4])
    simul_results = np.zeros([simul_lenght, 18])

    for (i, (t_div, Nb, alpha)) in enumerate(product(t_div_list, Nb_list, alpha_list)):
        simul_params[i] = np.array([t_div, Na, Nb, alpha])
        simul_results[i] = run_admix(
            t_div=t_div, Na=Na, Nb=Nb, Nc=Na, alpha=alpha, n=num_samples, num_replicates=2000
        )

    time_stamp = datetime.datetime.now().isoformat().split(".")[0]

    pop_labels = ["pop_a", "pop_c", "pop_b"]
    stats_lbels = [
        "mean_num_seg_sites_",
        "var_num_seg_sites_",
        "mean_nucleotide_div_",
        "var_nucleotide_div_",
        "mean_branch_length_",
        "var_branch_length_",
    ]
    columns = [i[0] + i[1] for i in product(stats_labels, pop_labels)]
    columns = ["t_div", "num_samples", "Na", "Nb", "alpha"] + columns

    output = pd.DataFrame(np.hstack((simul_params, simul_results)), columns=columns)
    output.to_csv("../data/msprime_admix_results_{}.csv.gz".format(time_stamp))

    pass

    # n = 100
    # x = np.loadtxt("/home/jonatas/admix_diversity-overleaf/scripts/simul_params_2.txt")
    # y = np.loadtxt("/home/jonatas/admix_diversity-overleaf/scripts/simul_results_2.txt")
    # nn = np.array([[n, Na]] * len(x[:, 2]))
    # xx = np.hstack((x, nn))
    # xxx = xx[:, [0,3,4,1,2]]
    # out = np.hstack((xxx, y))
    # output = pd.DataFrame(out, columns=columns)


if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
from joblib import Parallel, delayed
from itertools import product
import datetime
import pandas as pd
import importlib
from importlib import reload
import sim_engine_admix
from sim_engine_admix import DivergenceAdmixture
from sim_engine_admix import OutOfAfricaAdmixture
from sim_engine_admix import OutOfAfrica
import coal_sim_utils as csu

reload(sim_engine_admix)


def eval_statistics(simul_trees):

    seg_sites_admix = [csu.seg_sites_pops(ts) for ts in simul_trees]
    seg_sites_admix = np.vstack(seg_sites_admix).T

    htz_admix = [csu.nucleotide_div_pops(ts) for ts in simul_trees]
    htz_admix = np.vstack(htz_admix).T

    branch_length_admix = [csu.total_branch_length_pops(ts) for ts in simul_trees]
    branch_length_admix = np.vstack(branch_length_admix).T

    output = np.zeros([6, seg_sites_admix.shape[0]])
    output[0] = seg_sites_admix.mean(axis=1)
    output[1] = seg_sites_admix.var(axis=1)
    output[2] = htz_admix.mean(axis=1)
    output[3] = htz_admix.var(axis=1)
    output[4] = branch_length_admix.mean(axis=1)
    output[5] = branch_length_admix.var(axis=1)

    return output.reshape(-1)


def RunDA(
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
    ts_result = admix_sim.simulate(
        length=length, mutation_rate=mutation_rate, num_replicates=num_replicates
    )

    # Create a list from the simulated trees
    simul_trees = [ts for ts in ts_result]

    return eval_statistics(simul_trees)


def RunDivergenceAdmixture(simul_type="test", num_samples=1000, n_jobs=2):
    """TODO: Docstring for main.
    :returns: TODO

    """
    if simul_type is "test":
        Na = 1e4
        t_div_list = np.linspace(0, 2 * Na, 3)[1:]
        Nb_list = np.linspace(0, Na, 2)[1:]
        alpha_list = np.arange(1, 4, 5) / 10.0
        t_div_list / (2 * Na)
        num_samples = 20
    if simul_type is "long":
        Na = 1e4
        t_div_list = np.linspace(0, 2 * Na, 41)[1:]
        Nb_list = np.linspace(0, Na, 21)[1:]
        alpha_list = np.arange(1, 10, 1) / 10.0
        t_div_list / (2 * Na)
        num_samples = num_samples
    if simul_type is "fine_alpha":
        Na = 1e4
        t_div_list = np.linspace(0, 2 * Na, 41)[1:]
        # Nb_list = np.linspace(0, Na, 21)[1:]
        Nb_list = [0.7 * Na]
        alpha_list = np.linspace(0, 1, 41)[1:]
        t_div_list / (2 * Na)
        num_samples = num_samples

    def run_simul(i):
        (t_div, Nb, alpha) = i
        par = np.array([t_div, num_samples, Na, Nb, alpha])
        res = RunDA(
            t_div=t_div, Na=Na, Nb=Nb, Nc=Na, alpha=alpha, n=num_samples, num_replicates=2000
        )
        return np.hstack((par, res))

    pout = Parallel(n_jobs=n_jobs, prefer="processes", backend="loky")(
        delayed(run_simul)(i) for i in product(t_div_list, Nb_list, alpha_list)
    )

    time_stamp = datetime.datetime.now().isoformat().split(".")[0]

    pop_labels = ["pop_a", "pop_c", "pop_b"]
    stats_labels = [
        "mean_num_seg_sites_",
        "var_num_seg_sites_",
        "mean_nucleotide_div_",
        "var_nucleotide_div_",
        "mean_branch_length_",
        "var_branch_length_",
    ]
    columns = [i[0] + i[1] for i in product(stats_labels, pop_labels)]
    columns = ["t_div", "num_samples", "Na", "Nb", "alpha"] + columns

    output = pd.DataFrame(pout, columns=columns)
    output.to_csv("../data/results_{}_{}.csv.gz".format(simul_type, time_stamp))

    return output


def RunOOA(
    n=100, t_adm=0, alpha1=0.2, alpha2=0.05, length=1e4, mutation_rate=1e-8, num_replicates=500,
):

    admix_sim = OutOfAfricaAdmixture(t_adm=t_adm, alpha1=alpha1, alpha2=alpha2, n=n, debug=True)
    ts_result = admix_sim.simulate(
        length=length, mutation_rate=mutation_rate, num_replicates=num_replicates
    )

    # Create a list from the simulated trees
    simul_trees = [ts for ts in ts_result]

    return eval_statistics(simul_trees)


def RunOutOfAfricaAdmixture(simul_type="test", num_samples=1000, n_jobs=2):
    """TODO: Docstring for main.
    :returns: TODO

    """
    if simul_type is "test":
        alpha2 = 0.05
        alpha1_list = np.linspace(0, 1 - alpha2, 3)
        num_replicates = 10
        num_samples = 20
    if simul_type is "2sources":
        alpha2 = 0.00
        alpha1_list = np.linspace(0, 1 - alpha2, 41)[1:-1]
        num_replicates = 1000
        num_samples = 100
    if simul_type is "3sources":
        alpha2 = 0.05
        alpha1_list = np.linspace(0, 1 - alpha2, 40)[1:]
        num_replicates = 1000
        num_samples = 100

    def run_simul(i):
        (alpha1, alpha2, num_replicates) = i
        par = np.array([alpha1, alpha2])
        res = RunOOA(alpha1=alpha1, alpha2=alpha2, n=num_samples, num_replicates=num_replicates)
        return np.hstack((par, res))

    pout = Parallel(n_jobs=n_jobs, prefer="processes", backend="loky")(
        delayed(run_simul)(i) for i in product(alpha1_list, [alpha2], [num_replicates])
    )

    time_stamp = datetime.datetime.now().isoformat().split(".")[0]

    pop_labels = ["pop_a", "pop_c", "pop_b", "pop_d"]
    # pop_labels = ["pop_a", "pop_c", "pop_b"]
    stats_labels = [
        "mean_num_seg_sites_",
        "var_num_seg_sites_",
        "mean_nucleotide_div_",
        "var_nucleotide_div_",
        "mean_branch_length_",
        "var_branch_length_",
    ]
    columns = [i[0] + i[1] for i in product(stats_labels, pop_labels)]
    columns = ["alpha1", "alpha2"] + columns

    output = pd.DataFrame(pout, columns=columns)
    output.to_csv("../data/results_ooa_{}_{}.csv.gz".format(simul_type, time_stamp))

    return output


if __name__ == "__main__":
    output_ooa_test = RunDivergenceAdmixture(simul_type="test")
    output_ooa_2sources = RunOutOfAfricaAdmixture(simul_type="2sources", n_jobs=3)
    output_ooa_3sources = RunOutOfAfricaAdmixture(simul_type="3sources", n_jobs=3)
    # cols = ["alpha1", "alpha2"] + [i for i in cout if "mean_num_seg_sites_" in i]
    # output_ooa_2sources.loc[:,cols]
    output_1 = RunDivergenceAdmixture(simul_type="long", num_samples=100, n_jobs=120)
    output_2 = RunDivergenceAdmixture(simul_type="fine_alpha", num_samples=100, n_jobs=120)


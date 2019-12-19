import numpy as np
import pandas as pd
import coal_theory_utils as ctu
import admix_plot_utils as apu
from importlib import reload


def main(showfig=False):
    simul_data = pd.read_csv("../data/msprime_admix_results_2019-11-22T17:25:51.csv.gz")
    alpha_list = simul_data.alpha.unique()
    alpha_ref = alpha_list[-1]

    Na = simul_data.Na.unique()[0]
    simul_data["tajima_d_pop_c"] = (
        simul_data.mean_nucleotide_div_pop_c - simul_data.mean_num_seg_sites_pop_c
    )

    n = 2 * simul_data.num_samples.unique()[0]
    cte = sum(1.0 / np.arange(1, n))

    simul_data["tajima_d_pop_a"] = (
        simul_data.mean_nucleotide_div_pop_a * (n / (n - 1))
        - simul_data.mean_num_seg_sites_pop_a / cte
    )

    simul_data["tajima_d_pop_c"] = (
        simul_data.mean_nucleotide_div_pop_c * (n / (n - 1))
        - simul_data.mean_num_seg_sites_pop_c / cte
    )

    # simul_data.loc[:5, ["mean_nucleotide_div_pop_c", "mean_num_seg_sites_pop_c", "tajima_d_pop_c"]]

    for alpha_ref in alpha_list:
        try:
            contour_stats(simul_data, alpha_ref, stat="mean_nucleotide_div", showfig=showfig)
            lines_stats(simul_data, alpha_ref, stat="mean_nucleotide_div", showfig=showfig)
        except:
            None

        try:
            contour_stats(simul_data, alpha_ref, stat="mean_num_seg_sites", showfig=showfig)
            lines_stats(simul_data, alpha_ref, stat="mean_num_seg_sites", showfig=showfig)
        except:
            None

        try:
            contour_stats(simul_data, alpha_ref, stat="tajima_d", showfig=showfig)
            lines_stats(simul_data, alpha_ref, stat="tajima_d", showfig=showfig)
        except:
            None
    pass


if __name__ == "__main__":
    # main()
    showfig = True
    simul_data = pd.read_csv("../data/msprime_admix_results_2019-12-16T19:57:21.csv.gz")
    alpha_list = simul_data.alpha.unique()
    Na = simul_data.Na.unique()[0]
    n = simul_data.num_samples.unique()[0]
    alpha_ref = alpha_list[1]
    reload(ctu)
    reload(apu)
    apu.lines_stats(simul_data, alpha_ref, stat="mean_num_seg_sites", showfig=showfig)
    apu.contour_stats(simul_data, alpha_ref, stat="mean_num_seg_sites", showfig=showfig)



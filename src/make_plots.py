import numpy as np
import pandas as pd
import coal_theory_utils as ctu
import admix_plot_utils as apu
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors
from importlib import reload


def single_plots_simul(simul_data, showfig=False):
    alpha_list = simul_data.alpha.unique()
    alpha_ref = alpha_list[-1]

    for alpha_ref in alpha_list:
        try:
            apu.contour_stats(simul_data, alpha_ref, stat="mean_nucleotide_div", showfig=showfig)
            apu.lines_stats(simul_data, alpha_ref, stat="mean_nucleotide_div", showfig=showfig)
        except:
            None

        try:
            apu.contour_stats(simul_data, alpha_ref, stat="mean_num_seg_sites", showfig=showfig)
            apu.lines_stats(simul_data, alpha_ref, stat="mean_num_seg_sites", showfig=showfig)
        except:
            None

    pass


if __name__ == "__main__":

    apu.make_alpha_fst_plot(savefig=True, showfig=True)
    simul_data = pd.read_csv("../data/results_long_2020-02-26T16:45:24.csv.gz")
    single_plots_simul(simul_data)
    apu.multi_contour(simul_data)
    apu.multi_lines(simul_data)

    # apu.lines_stats(simul_data, alpha_ref, stat="mean_num_seg_sites", showfig=True)
    # apu.contour_stats(simul_data, alpha_ref, stat="mean_num_seg_sites", showfig=True)

    simul_data_alpha = pd.read_csv("../data/results_fine_alpha_2020-03-03T22:33:58.csv.gz")
    reload(apu)
    apu.multi_alpha(simul_data_alpha)
    apu.alpha_contour(simul_data_alpha)
    simul_data = simul_data_alpha


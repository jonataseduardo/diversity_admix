import numpy as np
import pandas as pd
import coal_theory_utils as ctu
import admix_plot_utils as apu
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from importlib import reload


def min_alpha(h_ratio, Fst):
    alpha = 0.5 * (1 - Fst) * (1 - h_ratio) / (1 + h_ratio) / Fst
    return np.ma.masked_outside(alpha, 0, 1)


def make_alpha_fst_plot(savefig=True, showfig=False):
    hr = fst = np.linspace(0.001, 0.999, 150)
    X, Y = np.meshgrid(hr, fst)
    Z = min_alpha(X, Y)

    fig, ax = plt.subplots()
    im = ax.imshow(
        Z, cmap="viridis_r", interpolation="bilinear", origin="lower", extent=[0, 1, 0, 1]
    )
    ax.set_xlabel(r"$F_{st}$", size=18)
    ax.set_ylabel(r"$\frac{h_1}{h_0}$", size=20, rotation=0, labelpad=12)
    axcb = fig.colorbar(im)
    axcb.set_label(r"$\alpha^*$", size=18, rotation=0, labelpad=12)
    fig.tight_layout()
    figname = "../figures/alpha_min_fst.pdf"
    if savefig:
        fig.savefig(figname)
    if showfig:
        fig.show()
    pass


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
    single_plots_simul(simul_data)
    make_alpha_fst_plot(savefig=True, showfig=True)

    showfig = True
    simul_data = pd.read_csv("../data/results_long_2020-02-26T16:45:24.csv.gz")

    alpha_ref = 0.2
    apu.lines_stats(simul_data, alpha_ref, stat="mean_num_seg_sites", showfig=showfig)
    apu.contour_stats(simul_data, alpha_ref, stat="mean_num_seg_sites", showfig=showfig)

    reload(apu)
    apu.multi_lines(simul_data)



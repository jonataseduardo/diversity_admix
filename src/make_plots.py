import numpy as np
import pandas as pd
import coal_theory_utils as ctu
import admix_plot_utils as apu
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

    reload(apu)
    apu.alpha_max_plot(savefig=True, showfig=True)
    apu.alpha_min_plot(savefig=True, showfig=True)

    simul_data = pd.read_csv("../data/results_long_2020-02-26T16:45:24.csv.gz")
    single_plots_simul(simul_data)
    apu.multi_contour(simul_data)
    apu.multi_lines(simul_data)

    data_ooa = pd.concat(
        [
            pd.read_csv("../data/results_ooa_2sources_2020-04-07T11:00:30.csv.gz"),
            pd.read_csv("../data/results_ooa_3sources_2020-04-07T11:04:11.csv.gz"),
        ]
    )

    # apu.lines_stats(simul_data, alpha_ref, stat="mean_num_seg_sites", showfig=True)
    # apu.contour_stats(simul_data, alpha_ref, stat="mean_num_seg_sites", showfig=True)

    simul_data_alpha = pd.read_csv("../data/results_fine_alpha_2020-03-03T22:33:58.csv.gz")
    apu.multi_alpha(simul_data_alpha)
    apu.alpha_contour(simul_data_alpha)

    def ratio_error(a, b, sigma_a, sigma_b, sigma_ab=0):
        return np.abs(a / b) * np.sqrt(
            (sigma_a / a) ** 2 + (sigma_b / b) ** 2 - 2 * sigma_ab / (a * b)
        )

    from scipy.interpolate import CubicHermiteSpline
    from scipy.interpolate import interp1d
    from scipy.interpolate import CubicSpline
    interp1d?


    k = 4
    x1 = data_ooa[data_ooa.alpha2 == 0.0].alpha1
    y1 = (
        data_ooa[data_ooa.alpha2 == 0.0].mean_nucleotide_div_pop_c
        / data_ooa[data_ooa.alpha2 == 0.0].mean_nucleotide_div_pop_a
    )
    x1_new = np.linspace(x1.min(), x1.max(),50)
    y1_new = np.poly1d(np.polyfit(x1, y1, k))(x1_new)
    
    z1 = (
        data_ooa[data_ooa.alpha2 == 0.0].mean_num_seg_sites_pop_c
        / data_ooa[data_ooa.alpha2 == 0.0].mean_num_seg_sites_pop_a
    )
    z1_new = np.poly1d(np.polyfit(x1, z1, k))(x1_new)

    x2 = data_ooa[data_ooa.alpha2 == 0.05].alpha1
    x2_new = np.linspace(x2.min(), x2.max(),50)
    y2_new = np.poly1d(np.polyfit(x2, y2, k))(x2_new)
    y2 = (
        data_ooa[data_ooa.alpha2 == 0.05].mean_nucleotide_div_pop_c
        / data_ooa[data_ooa.alpha2 == 0.0].mean_nucleotide_div_pop_a
    )
    z2 = (
        data_ooa[data_ooa.alpha2 == 0.05].mean_num_seg_sites_pop_c
        / data_ooa[data_ooa.alpha2 == 0.05].mean_num_seg_sites_pop_a
    )
    z2_new = np.poly1d(np.polyfit(x2, z2, k))(x2_new)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.scatter(x1, y1, color="darkorange", marker="o", s=22, alpha=0.5)
    ax1.plot(x1_new, y1_new, color="darkorange", linewidth = 2, ls="solid")
    ax1.scatter(x2, y2, color="seagreen", marker="D", s=22, alpha=0.5)
    ax1.plot(x2_new, y2_new, color="seagreen", linewidth = 2, ls="dashed")

    ax2 = fig.add_subplot(122, sharey=ax1)
    ax2.scatter(x1, z1, color="darkorange", marker="o", s=22, alpha=0.5)
    ax2.plot(x1_new, z1_new, color="darkorange", linewidth = 2, ls="solid")
    ax2.scatter(x2, z2, color="seagreen", marker="D", s=22, alpha=0.5)
    ax2.plot(x2_new, z2_new, color="seagreen", linewidth = 2, ls="dashed")
    fig.show()

    plt.plot?

    err_y1 = ratio_error(
        data_ooa[data_ooa.alpha2 == 0.0].mean_nucleotide_div_pop_c,
        data_ooa[data_ooa.alpha2 == 0.0].mean_nucleotide_div_pop_a,
        np.sqrt(data_ooa[data_ooa.alpha2 == 0.0].var_nucleotide_div_pop_c),
        np.sqrt(data_ooa[data_ooa.alpha2 == 0.0].mean_nucleotide_div_pop_a)
    )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import coal_theory_utils as ctu
from numpy import ma
from matplotlib import cbook
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from numpy import polyfit


def set_cmap_levels(max_value, min_value, midpoint=1, digits=1, nticks=15):
    if midpoint < min_value:
        midpoint = min_value

    if digits <= 2:
        wd = (5 ** ((digits + 1) % 2)) * (1 / 10 ** digits)
        min_value = min_value - min_value % wd
        max_value = max_value - max_value % wd + wd
        lower_levels = np.arange(round(min_value, digits), midpoint, wd)
        upper_levels = np.arange(midpoint, round(max_value, digits), wd)
    else:
        upper_nticks = round(nticks * (max_value - midpoint) / (max_value - min_value))
        lower_nticks = round(nticks * (midpoint - min_value) / (max_value - min_value))
        lower_levels = np.round(np.linspace(min_value, midpoint, lower_nticks), decimals=2)
        upper_levels = np.round(
            np.linspace(midpoint, max_value, upper_nticks, endpoint=True), decimals=2
        )

    lower_ticks = lower_levels[::-1][1::2][::-1]
    upper_ticks = upper_levels[0::2]
    levels = np.hstack((lower_levels, upper_levels))
    ticks = np.hstack((lower_ticks, upper_ticks))
    return (np.unique(levels), np.unique(ticks))


# Normalizing colormap solition was given in
# https://stackoverflow.com/a/7404116/
class MidPointNorm(Normalize):
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self, vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")
        elif vmin == vmax:
            # Or should it be all masked? Or 0.5?
            result.fill(0)
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax), mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            # First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint
            resdat[resdat > 0] /= abs(vmax - midpoint)
            resdat[resdat < 0] /= abs(vmin - midpoint)

            resdat /= 2.0
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)

        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val - 0.5)
            val[val > 0] *= abs(vmax - midpoint)
            val[val < 0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0:
                return val * abs(vmin - midpoint) + midpoint
            else:
                return val * abs(vmax - midpoint) + midpoint


def lines_htz(simul_params, simul_results, Na, alpha_ref):

    t_coal = np.sort(np.unique(simul_params[:, 0])) / (2 * Na)
    Nb_list = np.sort(np.unique(simul_params[:, 1]))

    idx_alpha = simul_params[:, 2] == alpha_ref
    ha = simul_results[:, 6]
    hc = simul_results[:, 7]
    h = hc / ha

    def get_h(h, Nb_ref):
        idx_Nb = simul_params[:, 1] == Nb_ref
        return h[idx_Nb & idx_alpha]

    h_simul_list = [list(zip(t_coal, get_h(h, Nb_ref))) for Nb_ref in Nb_list]
    h_theory_list = [
        list(zip(t_coal, ctu.admix_coal_time_ratio(t_coal, alpha_ref, Nb_ref / Na)))
        for Nb_ref in Nb_list
    ]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim((0, 1))
    ax.set_ylim((h.min(), h.max()))
    ax.set_title(r"$\alpha={{{}}}$".format(alpha_ref))
    ax.set_xlabel(r"$\mathrm{time} \, (2 N_a \, \mathrm{coalescent \, units})$")
    ax.set_ylabel(r"$H_c / H_a$")
    line_segments_simul = LineCollection(
        h_simul_list, linewidths=1.5, linestyles="solid", cmap="viridis_r"
    )
    line_segments_simul.set_array(Nb_list / Na)
    ax.add_collection(line_segments_simul)
    line_segments_theory = LineCollection(
        h_theory_list, linewidths=1.3, linestyles="dashed", cmap="viridis_r"
    )
    ax.add_collection(line_segments_theory)
    line_segments_theory.set_array(Nb_list / Na)
    axcb = fig.colorbar(line_segments_simul)
    axcb.set_label(r"$N_b / N_a$")
    fig.savefig("htz_time_alpha-{}.pdf".format(alpha_ref))
    fig.show()
    pass


def contour_stats(simul_data, alpha_ref, stat, digits=2, savefig=True, showfig=True):

    Nb_list = simul_data.Nb.unique()
    t_div_list = simul_data.t_div.unique()
    Na = simul_data.Na.unique()
    psize = len(t_div_list), len(Nb_list)
    data = simul_data[simul_data.alpha == alpha_ref]

    x = data.t_div.values.reshape(psize) / (2 * Na)
    y = data.Nb.values.reshape(psize) / Na

    if stat == "mean_nucleotide_div":
        H1 = data.loc[:, "mean_nucleotide_div_pop_a"].values
        H2 = data.loc[:, "mean_nucleotide_div_pop_c"].values
        res = H2 / H1
        z = res.reshape(psize)
        z_th = ctu.admix_coal_time_ratio(x, alpha_ref, y)
        s_label = r"$H_c / H_a$"
        figname = "../figures/contour_htz_alpha-{}.pdf".format(alpha_ref)
    elif stat == "mean_num_seg_sites":
        n = 100
        H1 = data.loc[:, "mean_num_seg_sites_pop_a"].values
        H2 = data.loc[:, "mean_num_seg_sites_pop_c"].values
        res = H2 / H1
        z = res.reshape(psize)
        z_th = ctu.s_admix_ratio(2 * Na * x / 4, n, Na, Na * y, alpha_ref)
        s_label = r"$S_c / S_a$"
        figname = "../figures/contour_num_seg_sites_alpha-{}.pdf".format(alpha_ref)

    cmap = plt.get_cmap("bwr")
    norm = MidPointNorm(midpoint=1)
    z_max = z.max()
    z_min = z.min() if z.min() < 1 else 0.95
    cmap_levels, cmap_ticks = set_cmap_levels(z_max, z_min, digits=digits)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(r"$\alpha={{{}}}$".format(alpha_ref))
    ax.set_xlabel(r"split time ($2 N_a$  coalescent units)")
    ax.set_ylabel(r"$N_b / N_a$")
    ax.set_xlim((x.min(), x.max()))
    ax.set_ylim((y.min(), y.max()))
    ct = ax.contourf(x, y, z, levels=cmap_levels, cmap=cmap, norm=norm)
    ct_th = ax.contour(x, y, z_th, levels=cmap_levels, colors="black", linestyles="dashed")
    ax.clabel(ct_th, cmap_ticks, inline=True, fmt=f"%.1f", fontsize=10)
    axcb = fig.colorbar(ct)
    axcb.set_label(s_label)
    axcb.set_ticks(cmap_ticks)

    if savefig:
        fig.savefig(figname)
    if showfig:
        fig.show()
    pass


def main():
    # simul_data = pd.read_csv("../data/msprime_admix_results_2019-06-18T14:31:39.csv.gz")
    simul_data = pd.read_csv("../data/msprime_admix_results_2019-06-20T16:56:08.csv")
    simul_data.columns
    alpha_list = simul_data.alpha.unique()
    alpha_ref = alpha_list[-2]

    for alpha_ref in alpha_list:
        try:
            contour_stats(simul_data, alpha_ref, stat="mean_nucleotide_div", showfig=True)
            contour_stats(simul_data, alpha_ref, stat="mean_num_seg_sites", showfig=True)
        except:
            None

        try:
            contour_num_seg_sites(simul_data, alpha_ref, showfig=True, savefig=False)
        except:
            None

    pass


if __name__ == "__main__":
    main()


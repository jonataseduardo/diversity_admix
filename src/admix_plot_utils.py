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
        upper_nticks = int(round(nticks * (max_value - midpoint) / (max_value - min_value)))
        lower_nticks = int(round(nticks * (midpoint - min_value) / (max_value - min_value)))
        lower_levels = np.round(np.linspace(min_value, midpoint, lower_nticks), decimals=2)
        upper_levels = np.round(
            np.linspace(midpoint, max_value, upper_nticks, endpoint=True), decimals=2
        )

    lower_ticks = lower_levels[::-1][1::2][::-1]
    upper_ticks = upper_levels[0::2]
    levels = np.hstack((lower_levels, upper_levels))
    ticks = np.hstack((lower_ticks, upper_ticks))
    return (np.unique(levels), np.unique(ticks))


class MidPointNorm(Normalize):
    # Normalizing colormap solition was given in
    # https://stackoverflow.com/a/7404116/
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


def lines_stats(simul_data, alpha_ref, stat, digits=2, savefig=True, showfig=True):

    data = simul_data[simul_data.alpha == alpha_ref]

    Nb_list = simul_data.Nb.unique()[::2]
    t_div_list = simul_data.t_div.unique()
    Na = simul_data.Na.unique()
    t_coal = t_div_list / (2 * Na)

    if stat == "mean_nucleotide_div":

        def get_val(Nb_ref):
            h1 = data[data.Nb == Nb_ref].loc[:, "mean_nucleotide_div_pop_a"].values
            h2 = data[data.Nb == Nb_ref].loc[:, "mean_nucleotide_div_pop_c"].values
            return h2 / h1

        h_simul_list = [list(zip(t_coal, get_val(Nb_ref))) for Nb_ref in Nb_list]
        h_theory_list = [
            list(zip(t_coal, ctu.admix_coal_time_ratio(t_coal, alpha_ref, Nb_ref / Na)))
            for Nb_ref in Nb_list
        ]
        y_label = r"$\frac{\pi_A}{\pi_0}$"
        lr = 0
        figname = "../figures/lines_htz_alpha-{}.pdf".format(alpha_ref)
    elif stat == "mean_num_seg_sites":
        n = data.num_samples.unique()

        def get_val(Nb_ref):
            # h1 = data[data.Nb == Nb_ref].loc[:, "mean_num_seg_sites_pop_b"].values
            # h2 = data[data.Nb == Nb_ref].loc[:, "mean_branch_length_pop_b"].values
            h1 = data[data.Nb == Nb_ref].loc[:, "mean_num_seg_sites_pop_c"].values
            h2 = data[data.Nb == Nb_ref].loc[:, "mean_num_seg_sites_pop_a"].values
            # return h2 / h1
            h = h1 / h2
            return h1

        h_simul_list = [list(zip(t_coal, get_val(Nb_ref))) for Nb_ref in Nb_list]
        h_theory_list = [
            list(
                zip(t_coal, ctu.s_admix_ratio((2 * Na) * t_coal, n, 2 * Na, 2 * Nb_ref, alpha_ref))
            )
            for Nb_ref in Nb_list
        ]
        y_label = r"$\frac{S_A}{S_0}$"
        lr = 0
        figname = "../figures/lines_num_seg_sites_alpha-{}.pdf".format(alpha_ref)
    elif stat == "tajima_d":
        n = 2 * data.num_samples.unique()

        def get_val(Nb_ref):
            h = data[data.Nb == Nb_ref].loc[:, "tajima_d_pop_c"].values
            return h

        h_simul_list = [list(zip(t_coal, get_val(Nb_ref))) for Nb_ref in Nb_list]
        h_theory_list = [
            list(zip(t_coal, ctu.tajima_d_admix((2 * Na) * t_coal, n, Na, Nb_ref, alpha_ref)))
            for Nb_ref in Nb_list
        ]
        y_label = r"$\hat{\theta}_{\pi_A} - \hat{\theta}_{S_A}}$"
        lr = 90
        figname = "../figures/lines_tajimas_d_admix_alpha-{}.png".format(alpha_ref)

    cmap = plt.get_cmap("Spectral")

    hs = np.array(h_simul_list)
    ht = np.array(h_theory_list)
    y_min, y_max = (
        min(hs[:, :, 1].min(), ht[:, :, 1].min()),
        max(hs[:, :, 1].max(), ht[:, :, 1].max()),
    )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim((0, 1))
    ax.set_ylim((0.9 * y_min, 1.1 * y_max))
    ax.set_title(r"$\alpha={{{}}}$".format(alpha_ref))
    ax.set_xlabel(r"$\mathrm{time} \, (2 N_0 \, \mathrm{coalescent \, units})$", size=16)
    ax.set_ylabel(y_label, rotation=lr, size=18, labelpad=10)
    line_segments_simul = LineCollection(
        h_simul_list, linewidths=1.5, linestyles="solid", cmap=cmap
    )
    line_segments_simul.set_array(Nb_list / Na)
    ax.add_collection(line_segments_simul)
    line_segments_theory = LineCollection(
        h_theory_list, linewidths=1.3, linestyles="dashed", cmap=cmap
    )
    ax.add_collection(line_segments_theory)
    line_segments_theory.set_array(Nb_list / Na)
    axcb = fig.colorbar(line_segments_simul)
    axcb.set_label(r"$\frac{N_1}{N_0}$", rotation=0, size=18, labelpad=10)

    fig.tight_layout()
    if savefig:
        fig.savefig(figname)
    if showfig:
        fig.show()


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
        s_label = r"$\frac{\pi_A}{\pi_0}$"
        figname = "../figures/contour_htz_alpha-{}.pdf".format(alpha_ref)
        lr = 0
        midpoint = 1
        nticks = 15
        digits = 1
    elif stat == "mean_num_seg_sites":
        n = 2 * data.num_samples.unique()
        H1 = data.loc[:, "mean_num_seg_sites_pop_a"].values
        H2 = data.loc[:, "mean_num_seg_sites_pop_c"].values
        # res = H2 / H1
        z = res.reshape(psize)
        z_th = ctu.s_admix_ratio((2 * Na) * x, n, 2 * Na, 2 * Na * y, alpha_ref)
        s_label = r"$\frac{S_A}{S_0}$"
        figname = "../figures/contour_num_seg_sites_alpha-{}.pdf".format(alpha_ref)
        lr = 0
        midpoint = 1
        nticks = 15
        digits = 1
    elif stat == "tajima_d":
        n = 2 * data.num_samples.unique()
        res = data.loc[:, "tajima_d_pop_c"].values
        z = res.reshape(psize)
        lr = 0
        # reload(ctu)
        z_th = ctu.tajima_d_admix((2 * Na) * x, n, Na, Na * y, alpha_ref)
        # z_th.max()
        # z_th.min()
        s_label = r""
        figname = "../figures/contour_tajimas_d_admix_alpha-{}.png".format(alpha_ref)
        midpoint = 0
        nticks = 15
        digits = 3

    cmap = plt.get_cmap("bwr")
    norm = MidPointNorm(midpoint=midpoint)
    z_max = z.max()
    z_min = z.min() if z.min() < midpoint else midpoint - 0.05
    cmap_levels, cmap_ticks = set_cmap_levels(
        z_max, z_min, midpoint=midpoint, digits=digits, nticks=nticks
    )

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(r"$\alpha={{{}}}$".format(alpha_ref))
    ax.set_xlabel(r"split time ($2 N_a$  coalescent units)", size=16)
    ax.set_ylabel(r"$\frac{N_1}{N_0}$", size=16, rotation=0, labelpad=10)
    ax.set_xlim((x.min(), x.max()))
    ax.set_ylim((y.min(), y.max()))
    ct = ax.contourf(x, y, z, levels=cmap_levels, cmap=cmap, norm=norm)
    if stat != "tajima_d":
        ct_th = ax.contour(x, y, z_th, levels=cmap_levels, colors="black", linestyles="dashed")
        ax.clabel(ct_th, cmap_ticks, inline=True, fmt=f"%.1f", fontsize=10)
    axcb = fig.colorbar(ct)
    axcb.set_label(s_label, size=16, rotation=lr, labelpad=10)
    axcb.set_ticks(cmap_ticks)

    fig.tight_layout()
    if savefig:
        fig.savefig(figname)
    if showfig:
        fig.show()
    pass


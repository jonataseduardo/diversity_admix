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
from itertools import product
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patheffects as PathEffects
import matplotlib.cm as cm
import matplotlib.colors


def my_ceil(a, precision=0):
    return np.round(a + 0.5 * 10 ** (-precision), precision)


def my_floor(a, precision=0):
    return np.round(a - 0.5 * 10 ** (-precision), precision)


def set_cmap_levels(max_value, min_value, midpoint=1, digits=1, nticks=15):
    if midpoint < min_value:
        midpoint = min_value

    if digits <= 2:
        wd = (5 ** ((digits + 1) % 2)) * (1 / 10 ** digits)
        min_value = min_value - min_value % wd
        max_value = max_value - max_value % wd + wd
        lower_levels = np.arange(round(min_value, digits), midpoint, wd)
        upper_levels = np.arange(midpoint, round(max_value, digits) + wd, wd)
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


def add_inner_title(ax, title, loc, size=None, **kwargs):
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.patheffects import withStroke

    if size is None:
        size = dict(size=plt.rcParams["legend.fontsize"])
    at = AnchoredText(title, loc=loc, prop=size, pad=0.0, borderpad=0.5, frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
    return at


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
            # h = data[data.Nb == Nb_ref].loc[:, "mean_num_seg_sites_pop_b"].values
            h2 = data[data.Nb == Nb_ref].loc[:, "mean_num_seg_sites_pop_a"].values
            h1 = data[data.Nb == Nb_ref].loc[:, "mean_num_seg_sites_pop_c"].values
            h = h1 / h2
            return h

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
    elif stat == "prop_diff":
        n = data.num_samples.unique()

        def get_val(Nb_ref):
            h1 = data[data.Nb == Nb_ref].loc[:, "mean_nucleotide_div_pop_a"].values
            h2 = data[data.Nb == Nb_ref].loc[:, "mean_nucleotide_div_pop_c"].values
            s1 = data[data.Nb == Nb_ref].loc[:, "mean_num_seg_sites_pop_a"].values
            s2 = data[data.Nb == Nb_ref].loc[:, "mean_num_seg_sites_pop_c"].values
            s = h2 / h1
            h = s2 / s1
            return s - h

        h_simul_list = [list(zip(t_coal, get_val(Nb_ref))) for Nb_ref in Nb_list]
        h_theory_list = [
            list(
                zip(
                    t_coal,
                    ctu.admix_coal_time_ratio(t_coal, alpha_ref, Nb_ref / Na)
                    - ctu.s_admix_ratio((2 * Na) * t_coal, n, 2 * Na, 2 * Nb_ref, alpha_ref),
                )
            )
            for Nb_ref in Nb_list
        ]
        y_label = r"$\frac{\pi_A}{\pi_0} - \frac{S_A}{S_0}$"
        lr = 90
        figname = "../figures/lines_prop_diff_alpha-{}.png".format(alpha_ref)

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
    ax.set_title(r"$\alpha={{{}}}$".format(alpha_ref), size=16)
    ax.set_xlabel(r"$\mathrm{time} \, (2 N_0 \, \mathrm{coalescent \, units})$", size=16)
    ax.set_ylabel(y_label, rotation=lr, size=20, labelpad=10)
    ax.plot(np.linspace(0, 1, 20), np.ones(20), color="black", linestyle="dotted", linewidth=1.5)
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
    axcb.set_label(r"$\frac{N_1}{N_0}$", rotation=0, size=20, labelpad=12)

    fig.tight_layout()
    if savefig:
        fig.savefig(figname)
    if showfig:
        fig.show()


def multi_lines(simul_data, alpha_list=[0.2, 0.5, 0.8], savefig=True, showfig=True):
    Nb_list = simul_data.Nb.unique()[::2]
    t_div_list = simul_data.t_div.unique()
    Na = simul_data.Na.unique()
    t_coal = t_div_list / (2 * Na)

    def get_stat(alpha_ref, stat):
        data = simul_data[simul_data.alpha == alpha_ref].copy()

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

        elif stat == "mean_num_seg_sites":
            n = data.num_samples.unique()

            def get_val(Nb_ref):
                # h = data[data.Nb == Nb_ref].loc[:, "mean_num_seg_sites_pop_b"].values
                h2 = data[data.Nb == Nb_ref].loc[:, "mean_num_seg_sites_pop_a"].values
                h1 = data[data.Nb == Nb_ref].loc[:, "mean_num_seg_sites_pop_c"].values
                h = h1 / h2
                return h

            h_simul_list = [list(zip(t_coal, get_val(Nb_ref))) for Nb_ref in Nb_list]
            h_theory_list = [
                list(
                    zip(
                        t_coal,
                        ctu.s_admix_ratio((2 * Na) * t_coal, n, 2 * Na, 2 * Nb_ref, alpha_ref),
                    )
                )
                for Nb_ref in Nb_list
            ]
            y_label = r"$\frac{S_A}{S_0}$"
        return (h_simul_list, h_theory_list, y_label)

    cmap = plt.get_cmap("Spectral")
    fig = plt.figure()
    grid = ImageGrid(
        fig,
        (0.1, 0.12, 0.8, 0.8),
        nrows_ncols=(2, 3),
        aspect=1,
        direction="row",
        axes_pad=0.05,
        add_all=True,
        label_mode="L",
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="5%",
        cbar_pad=0.05,
    )

    stat_list = ["mean_nucleotide_div", "mean_num_seg_sites"]
    for ax_id, (stat, alpha_ref) in enumerate(product(stat_list, alpha_list)):

        ax = grid[ax_id]
        ax.tick_params(labelsize=12)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])

        if ax_id <= 2:
            ax.set_title(r"$\alpha={{{}}}$".format(alpha_ref), size=16)

        ax.plot(
            np.linspace(0, 1, 20), np.ones(20), color="black", linestyle="dotted", linewidth=1.5
        )
        simul_data_list, theory_data_list, y_label = get_stat(alpha_ref, stat)
        line_segments_simul = LineCollection(
            simul_data_list, linewidths=1.5, linestyles="solid", cmap=cmap
        )
        line_segments_simul.set_array(Nb_list / Na)
        ax.add_collection(line_segments_simul)
        line_segments_theory = LineCollection(
            theory_data_list, linewidths=1.3, linestyles="dashed", cmap=cmap
        )
        im = ax.add_collection(line_segments_theory)
        line_segments_theory.set_array(Nb_list / Na)

        if ax_id == 0 or ax_id == 3:
            ax.set_ylabel(y_label, rotation=0, size=20, labelpad=10)

    grid[4].set_xlabel(r"$\mathrm{time} \, (2 N_0 \, \mathrm{coalescent \, units})$", size=16)
    cbar = ax.cax.colorbar(im)
    cbar.set_label_text(r"$\frac{N_1}{N_0}$", size=20, rotation=0)
    cbar.ax.xaxis.set_label_coords(-1, -1)
    cbar.ax.tick_params(labelsize=12, rotation=-30, length=3, pad=3)

    for ax, im_title in zip(grid, ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]):
        t = add_inner_title(ax, im_title, loc=2)
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)

    if savefig:
        figname = "../figures/multi_lines.pdf"
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
        n = data.num_samples.unique()
        H1 = data.loc[:, "mean_num_seg_sites_pop_a"].values
        H2 = data.loc[:, "mean_num_seg_sites_pop_c"].values
        res = H2 / H1
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
        z_th = ctu.tajima_d_admix((2 * Na) * x, n, Na, Na * y, alpha_ref)
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
    ax.set_title(r"$\alpha={{{}}}$".format(alpha_ref), size=16)
    ax.set_xlabel(r"split time ($2 N_0$  coalescent units)", size=16)
    ax.set_ylabel(r"$\frac{N_1}{N_0}$", size=20, rotation=0, labelpad=10)
    ax.set_xlim((x.min(), x.max()))
    ax.set_ylim((y.min(), y.max()))
    ct = ax.contourf(x, y, z, levels=cmap_levels, cmap=cmap, norm=norm)
    if stat != "tajima_d":
        ct_th = ax.contour(x, y, z_th, levels=cmap_levels, colors="black", linestyles="dashed")
        ax.clabel(ct_th, cmap_ticks, inline=True, fmt=f"%.1f", fontsize=10)
    axcb = fig.colorbar(ct)
    axcb.set_label(s_label, size=20, rotation=lr, labelpad=12)
    axcb.set_ticks(cmap_ticks)
    fig.tight_layout()
    if savefig:
        fig.savefig(figname)
    if showfig:
        fig.show()
    pass


def multi_contour(simul_data, alpha_list=[0.2, 0.5, 0.8], savefig=True, showfig=True):
    Nb_list = simul_data.Nb.unique()
    t_div_list = simul_data.t_div.unique()
    Na = simul_data.Na.unique()
    psize = len(t_div_list), len(Nb_list)

    def get_stat(alpha_ref, stat):
        data = simul_data[simul_data.alpha == alpha_ref]
        x = data.t_div.values.reshape(psize) / (2 * Na)
        y = data.Nb.values.reshape(psize) / Na

        if stat == "mean_nucleotide_div":
            h1 = data.loc[:, "mean_nucleotide_div_pop_a"].values
            h2 = data.loc[:, "mean_nucleotide_div_pop_c"].values
            z = (h2 / h1).reshape(psize)
            z_th = ctu.admix_coal_time_ratio(x, alpha_ref, y)
            s_label = r"   $\frac{\pi_A}{\pi_0}$"

        elif stat == "mean_num_seg_sites":
            h1 = data.loc[:, "mean_num_seg_sites_pop_a"].values
            h2 = data.loc[:, "mean_num_seg_sites_pop_c"].values
            z = (h2 / h1).reshape(psize)
            n = data.num_samples.unique()
            z_th = ctu.s_admix_ratio((2 * Na) * x, n, 2 * Na, 2 * Na * y, alpha_ref)
            s_label = r"   $\frac{S_A}{S_0}$"

        return (x, y, z, z_th, s_label)

    cmap = plt.get_cmap("bwr")
    fig = plt.figure()

    def make_grid(stat, rect, digits=1, nticks=4):
        midpoint = 1
        norm = MidPointNorm(midpoint=midpoint)
        data_aux = simul_data[simul_data.alpha.isin(alpha_list)]
        aux = data_aux.loc[:, stat + "_pop_c"] / data_aux.loc[:, stat + "_pop_a"]
        z_max = my_ceil(aux.max(), precision=digits)
        z_min = my_floor(aux.min(), precision=digits)
        cmap_levels, cmap_ticks = set_cmap_levels(
            z_max, z_min, midpoint=midpoint, digits=digits, nticks=nticks
        )

        grid = ImageGrid(
            fig,
            rect,
            nrows_ncols=(1, 3),
            aspect=1,
            direction="row",
            axes_pad=0.08,
            add_all=True,
            label_mode="L",
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=0.05,
        )

        for ax_id, alpha_ref in enumerate(alpha_list):
            x, y, z, z_th, cbar_label = get_stat(alpha_ref, stat)
            ax = grid[ax_id]
            ax.tick_params(labelsize=12)
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["  0", "0.5", "1  "])
            ax.set_xlim((0, 1))

            if stat is "mean_nucleotide_div":
                ax.set_title(r"$\alpha={{{}}}$".format(alpha_ref), size=16)
            if stat is "mean_num_seg_sites":
                if ax_id == 1:
                    ax.set_xlabel(r"split time ($2 N_0$  coalescent units)", size=16)

            ax.set_ylabel(r"$\frac{N_1}{N_0}$", size=20, rotation=0, labelpad=10)
            ct = ax.contourf(x, y, z, levels=cmap_levels, cmap=cmap, norm=norm)
            ct_th = ax.contour(x, y, z_th, levels=cmap_levels, colors="black", linestyles="dashed")
            ax.clabel(ct_th, cmap_ticks, inline=True, fmt=f"%.1f", fontsize=10)

        cbar = ax.cax.colorbar(ct, ticks=cmap_ticks)
        cbar.set_label_text(cbar_label, size=20, rotation=0)
        return grid

    grid1 = make_grid("mean_nucleotide_div", 211)
    grid2 = make_grid("mean_num_seg_sites", 212)

    for ax, im_title in zip(grid1, ["(a)", "(b)", "(c)"]):
        t = add_inner_title(ax, im_title, loc=2)
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)

    for ax, im_title in zip(grid2, ["(d)", "(e)", "(f)"]):
        t = add_inner_title(ax, im_title, loc=2)
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)

    if savefig:
        figname = "../figures/multi_contour.pdf"
        fig.savefig(figname)
    if showfig:
        fig.show()


def alpha_contour(simul_data, savefig=True, showfig=True):
    Nb = simul_data.Nb.unique()
    alpha_list = simul_data.alpha.unique()
    t_div_list = simul_data.t_div.unique()
    Na = simul_data.Na.unique()
    psize = len(t_div_list), len(alpha_list)

    def get_stat(stat):
        data = simul_data
        x = data.t_div.values.reshape(psize) / (2 * Na)
        y = data.alpha.values.reshape(psize)

        if stat == "mean_nucleotide_div":
            h1 = data.loc[:, "mean_nucleotide_div_pop_a"].values
            h2 = data.loc[:, "mean_nucleotide_div_pop_c"].values
            z = (h2 / h1).reshape(psize)
            z_th = ctu.admix_coal_time_ratio(x, y, Nb / Na)
            s_label = r"   $\frac{\pi_A}{\pi_0}$"

        elif stat == "mean_num_seg_sites":
            h1 = data.loc[:, "mean_num_seg_sites_pop_a"].values
            h2 = data.loc[:, "mean_num_seg_sites_pop_c"].values
            z = (h2 / h1).reshape(psize)
            n = data.num_samples.unique()
            z_th = ctu.s_admix_ratio((2 * Na) * x, n, 2 * Na, 2 * Nb, y)
            s_label = r"   $\frac{S_A}{S_0}$"

        return (x, y, z, z_th, s_label)

    cmap = plt.get_cmap("bwr")
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    def make_subplot(stat, ax, digits=2, nticks=5):
        midpoint = 1
        norm = MidPointNorm(midpoint=midpoint)
        aux = simul_data.loc[:, stat + "_pop_c"] / simul_data.loc[:, stat + "_pop_a"]
        z_max = my_ceil(aux.max(), precision=digits)
        z_min = my_floor(aux.min(), precision=digits)
        cmap_levels, cmap_ticks = set_cmap_levels(
            z_max, z_min, midpoint=midpoint, digits=digits, nticks=nticks
        )

        x, y, z, z_th, cbar_label = get_stat(stat)
        ax.tick_params(labelsize=12)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["  0", "0.5", "1  "])
        ax.set_xlim((0, 1))
        ax.set_xlabel(r"$T_s/(2 N_0)$ ", size=16)
        ax.set_ylabel(r"$\alpha$", size=20, rotation=0, labelpad=10)

        ct = ax.contourf(x, y, z, levels=cmap_levels, cmap=cmap, norm=norm)
        ct_th = ax.contour(x, y, z_th, levels=cmap_levels, colors="black", linestyles="dashed")
        ax.clabel(ct_th, cmap_ticks, inline=True, fmt=f"%.1f", fontsize=10)

        # cbar = ax.cax.colorbar(ct, ticks=cmap_ticks)
        # cbar.set_label_text(cbar_label, size=20, rotation=0)
        return ax

    make_subplot("mean_nucleotide_div", ax1)
    make_subplot("mean_num_seg_sites", ax2)

    # for ax, im_title in zip(grid1, ["(a)", "(b)", "(c)"]):
    #    t = add_inner_title(ax, im_title, loc=2)
    #    t.patch.set_ec("none")
    #    t.patch.set_alpha(0.5)

    # for ax, im_title in zip(grid2, ["(d)", "(e)", "(f)"]):
    #    t = add_inner_title(ax, im_title, loc=2)
    #    t.patch.set_ec("none")
    #    t.patch.set_alpha(0.5)

    fig.tight_layout()
    if savefig:
        figname = "../figures/multi_alpha_contour.pdf"
        fig.savefig(figname)
    if showfig:
        fig.show()


def multi_alpha(simul_data, Nb_list_prop=[0.3, 0.5, 0.7], savefig=True, showfig=True):
    alpha_list = simul_data.alpha.unique()
    t_div_list = simul_data.t_div.unique()
    Na = simul_data.Na.unique()
    Nb_list = np.array(Nb_list_prop) * Na
    psize = len(t_div_list), len(alpha_list)

    def get_stat(Nb_ref, stat):
        data = simul_data[simul_data.Nb == Nb_ref]
        x = data.t_div.values.reshape(psize) / (2 * Na)
        y = data.alpha.values.reshape(psize)

        if stat == "mean_nucleotide_div":
            h1 = data.loc[:, "mean_nucleotide_div_pop_a"].values
            h2 = data.loc[:, "mean_nucleotide_div_pop_c"].values
            z = (h2 / h1).reshape(psize)
            z_th = ctu.admix_coal_time_ratio(x, y, Nb_ref / Na)
            s_label = r"   $\frac{\pi_A}{\pi_0}$"

        elif stat == "mean_num_seg_sites":
            h1 = data.loc[:, "mean_num_seg_sites_pop_a"].values
            h2 = data.loc[:, "mean_num_seg_sites_pop_c"].values
            z = (h2 / h1).reshape(psize)
            n = data.num_samples.unique()
            z_th = ctu.s_admix_ratio((2 * Na) * x, n, 2 * Na, 2 * Nb_ref, y)
            s_label = r"   $\frac{S_A}{S_0}$"

        return (x, y, z, z_th, s_label)

    cmap = plt.get_cmap("bwr")
    fig = plt.figure()

    def make_grid(stat, rect, digits=1, nticks=3):
        midpoint = 1
        norm = MidPointNorm(midpoint=midpoint)
        data_aux = simul_data[simul_data.Nb.isin(Nb_list)]
        aux = data_aux.loc[:, stat + "_pop_c"] / data_aux.loc[:, stat + "_pop_a"]
        z_max = my_ceil(aux.max(), precision=digits)
        z_min = my_floor(aux.min(), precision=digits)
        cmap_levels, cmap_ticks = set_cmap_levels(
            z_max, z_min, midpoint=midpoint, digits=digits, nticks=nticks
        )

        grid = ImageGrid(
            fig,
            rect,
            nrows_ncols=(1, 3),
            aspect=1,
            direction="row",
            axes_pad=0.08,
            add_all=True,
            label_mode="L",
            share_all=True,
            cbar_location="right",
            cbar_mode="single",
            cbar_size="5%",
            cbar_pad=0.05,
        )

        for ax_id, Nb_ref in enumerate(Nb_list):
            x, y, z, z_th, cbar_label = get_stat(Nb_ref, stat)
            ax = grid[ax_id]
            ax.tick_params(labelsize=12)
            ax.set_ylabel(r"$\alpha$", size=20, rotation=0, labelpad=10)
            ax.set_xticks([0, 0.5, 1])
            ax.set_xticklabels(["  0", "0.5", "1  "])
            ax.set_xlim((0, 1))
            if stat is "mean_nucleotide_div":
                ax.set_title(r"$\kappa={{{}}}$".format(Nb_list_prop[ax_id]), size=16)
            if stat is "mean_num_seg_sites":
                if ax_id == 1:
                    ax.set_xlabel(r"split time ($2 N_0$  coalescent units)", size=16)

            ct = ax.contourf(x, y, z, levels=cmap_levels, cmap=cmap, norm=norm)
            ct_th = ax.contour(x, y, z_th, levels=cmap_levels, colors="black", linestyles="dashed")
            ax.clabel(ct_th, cmap_ticks, inline=True, fmt=f"%.1f", fontsize=10)
            if ax_id == 2:
                ax.plot(0.09, 0.85, "P", color="black", markersize="8")
                ax.plot(0.09, 0.23, "X", color="black", markersize="8")
                ax.text(
                    0.09,
                    0.85,
                    "  ASW",
                    path_effects=[PathEffects.withStroke(linewidth=4, foreground="w")],
                )
                ax.text(
                    0.09,
                    0.23,
                    "  BRL",
                    path_effects=[PathEffects.withStroke(linewidth=4, foreground="w")],
                )

        cbar = ax.cax.colorbar(ct, ticks=cmap_ticks)
        cbar.set_label_text(cbar_label, size=20, rotation=0)
        return grid

    grid1 = make_grid("mean_nucleotide_div", 211)
    grid2 = make_grid("mean_num_seg_sites", 212)

    for ax, im_title in zip(grid1, ["(a)", "(b)", "(c)"]):
        t = add_inner_title(ax, im_title, loc=1)
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)

    for ax, im_title in zip(grid2, ["(d)", "(e)", "(f)"]):
        t = add_inner_title(ax, im_title, loc=1)
        t.patch.set_ec("none")
        t.patch.set_alpha(0.5)

    if savefig:
        figname = "../figures/multi_alpha.pdf"
        fig.savefig(figname)
    if showfig:
        fig.show()


def min_alpha(h_ratio, Fst):
    alpha = ((1 - Fst) / (2 * Fst)) * ((1 - h_ratio) / (1 + h_ratio))
    return np.ma.masked_outside(alpha, 0, 1)


def alpha_pi_max(h_ratio, Fst):
    alpha = ((3 * Fst - 2) * h_ratio + Fst) / (2 * (2 * Fst - 1) * (1 + h_ratio))
    return np.ma.masked_outside(alpha, 0, 1)


def h_min(Fst):
    return (1 - 3 * Fst) / (1 + Fst)


def alpha_min_plot(savefig=True, showfig=False):
    hr = fst = np.linspace(0.001, 0.999, 150)
    X, Y = np.meshgrid(hr, fst, indexing="ij")
    Z = min_alpha(X, Y)

    # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap("viridis_r")
    fig, ax = plt.subplots()
    ax.plot(fst, h_min(fst), color=cmap(np.max(Z)), lw=2)
    im = ax.imshow(
        Z, cmap=cmap, vmax=1, vmin=0, interpolation="nearest", origin="lower", extent=[0, 1, 0, 1]
    )
    ax.tick_params(labelsize=12)
    ax.set_ylim((0, 1))
    ax.set_xlabel(r"$F_{st}$", size=22)
    ax.set_ylabel(r"$\frac{\pi_1}{\pi_0}$", size=26, rotation=0, labelpad=12)
    axcb = fig.colorbar(im)
    axcb.set_label(r"$\alpha^*_\pi$", size=22, rotation=0, labelpad=12)
    axcb.ax.tick_params(labelsize=12)
    fig.tight_layout()
    figname = "../figures/alpha_min.pdf"
    if savefig:
        fig.savefig(figname)
    if showfig:
        fig.show()
    pass


def alpha_max_plot(savefig=True, showfig=False):
    hr = fst = np.linspace(0.001, 0.999, 150)
    X, Y = np.meshgrid(hr, fst, indexing="ij")
    Z = alpha_pi_max(X, Y)

    # norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cmap = cm.get_cmap("viridis_r")
    fig, ax = plt.subplots()
    # ax.plot(fst, h_min(fst), color=cmap(np.max(Z)), lw=2)
    im = ax.imshow(
        Z, cmap=cmap, vmax=1, vmin=0, interpolation="nearest", origin="lower", extent=[0, 1, 0, 1]
    )
    ax.tick_params(labelsize=12)
    ax.set_ylim((0, 1))
    ax.set_xlabel(r"$F_{st}$", size=22)
    ax.set_ylabel(r"$\frac{\pi_1}{\pi_0}$", size=26, rotation=0, labelpad=12)
    axcb = fig.colorbar(im)
    axcb.set_label(r"$\alpha^{**}_\pi$", size=22, rotation=0, labelpad=12)
    axcb.ax.tick_params(labelsize=12)
    fig.tight_layout()
    figname = "../figures/alpha_max.pdf"
    if savefig:
        fig.savefig(figname)
    if showfig:
        fig.show()
    pass

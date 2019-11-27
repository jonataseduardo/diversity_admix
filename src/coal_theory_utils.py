import numpy as np
from scipy.special import comb


def ccomb(N, k):
    return np.exp(gammaln(N + 1) - gammaln(N - k + 1) - gammaln(k + 1))


def branch_length(n, N, T):
    """
    Expected Branch Length of a Single Population in time interval 
    and constant population size (Haploid Individuals)

    Arguments
    ---------
    n: initial (present) number of linages 
    N: effective population size 
    T: Time interval in generations

    Returns
    -------
    branch-length: np.array
    
    """
    return 2 * N * np.log(1 + n * np.exp(T / (2 * N)) - n)


def nlinages(n, N, T):
    """
    Expected number of linages of a Single Population of constant effective 
    size N at a past time T in generations 

    Arguments
    ---------
    n: initial (present) number of linages 
    N: effective population size (Haploid indviduals) 
    T: Time interval in generations

    Returns
    -------
    num_linages: np.array
    
    """
    return n / (n + (1 - n) * np.exp(-T / (2 * N)))


def sites_fixed_after(t_div, na, nb, Na, Nb):
    """
    Calculate the number of sites the are fixed in both sorce populatins after 
    
    Arguments
    ---------
    t_div: Time interval in generations

    n: initial (present) number of linages 

    Na: effective population size of the constant size source

    Nb: effective population size of the of the second source after the split

    Returns
    -------
    branch_length: np.array
    
    """

    na2 = na * (na - 1)
    nb2 = nb * (nb - 1)
    s_fixed_after = (
        t_div
        - 2 * Na * (1 - np.exp(-0.5 * na2 * t_div / Na)) / na2
        - 2 * Nb * (1 - np.exp(-0.5 * nb2 * t_div / Nb)) / nb2
    )
    return s_fixed_after


def nsites_pop_a(t_div, na, nb, Na, Nb, subtract_fixed=True):
    """
    number of seg sites in the source population of constant size in the demographic model.

    Arguments
    ---------
    t_div: Time interval in generations

    n: initial (present) number of linages 

    Na: effective population size of the constant size source

    Nb: effective population size of the of the second source after the split

    subtract_fixed: boolean (default True), subtract the fixed sites on both source populations

    Returns
    -------
    branch_length: np.array
    
    """
    ns_a = nlinages(na, Na, t_div)
    sa_before = 2 * Na * np.log(ns_a)
    sa_after = branch_length(na, Na, t_div)
    if subtract_fixed:
        sf_after = sites_fixed_after(t_div, na, nb, Na, Nb)
        sa = sa_before + sa_after - sf_after
    else:
        sa = sa_before + sa_after
    return sa


def nsites_pop_b(t_div, na, nb, Na, Nb, subtract_fixed=True):
    """
    number of seg sites in the source population of constant size in the demographic model.

    Arguments
    ---------
    t_div: Time interval in generations

    n: initial (present) number of linages 

    Na: effective population size of the constant size source

    Nb: effective population size of the of the second source after the split

    subtract_fixed: boolean (default True), subtract the fixed sites on both source populations

    Returns
    -------
    branch_length: np.array
    
    """
    ns_b = nlinages(nb, Nb, t_div)
    sb_before = 2 * Na * np.log(ns_b)
    sb_after = branch_length(nb, Nb, t_div)
    if subtract_fixed:
        sf_after = sites_fixed_after(t_div, na, nb, Na, Nb)
        sb = sb_before + sb_after - sf_after
    else:
        sb = sb_before + sb_after
    return sb


def nsites_2pop(t_div, na, nb, Na, Nb, subtract_fixed=True):
    """
    number of seg sites in the source population of constant size in the demographic model.

    Arguments
    ---------
    t_div: Time interval in generations

    na: initial (present) number of linages in pop a
    
    nb: initial (present) number of linages in pop b

    Na: effective population size of the constant size source

    Nb: effective population size of the of the second source after the split

    subtract_fixed: boolean (default True), subtract the fixed sites on both source populations

    Returns
    -------
    branch_length: np.array
    
    """
    ns_a = nlinages(na, Na, t_div)
    ns_b = nlinages(nb, Nb, t_div)
    s_before = 2 * Na * np.log(ns_a + ns_b)

    sa_after = branch_length(na, Na, t_div)
    sb_after = branch_length(nb, Nb, t_div)
    s_after = sa_after + sb_after
    if subtract_fixed:
        sf_after = sites_fixed_after(t_div, na, nb, Na, Nb)
        sf_before = (1 / ns_a + 1 / ns_b) / comb(ns_a + ns_b, ns_a)
        stotal = s_before + s_after - sf_before - sf_after
    else:
        stotal = s_before + s_after
    return stotal


def s_admix_ratio(t_div, n, Na, Nb, alpha):
    """
    Ratio between the number of segregating sites of an admixed population
    and one of its source populations immediately after the admixture event.

    Arguments
    ---------
    t_div: Time interval in generations

    n: initial (present) number of linages 

    Na: effective population size of the focal source

    Nb: effective population size of the non-focal source

    alpha: proportion of the focal source population in the admixed population

    Returns
    -------
    num_linages: np.array
    
    """
    Sadmix = nsites_2pop(t_div, alpha * n, (1 - alpha) * n, Na, Nb, subtract_fixed=True)
    Sa = nsites_pop_a(t_div, n, n, Na, Nb, subtract_fixed=True)
    return Sadmix / Sa


def admix_coal_time_ratio(t_div, alpha, kappa):
    """
    Ratio between the average coalescent time of two linages for the admixed
    population and one of its source populations immediately after the admixture
    event.
     

    Arguments
    ---------
    t_div: Time interval in generations coalescent units of the focal source
    population (generations / Ne for haploids or generations / 2 * Ne for
    diploids)

    alpha: proportion of the focal source population in the admixed population

    Nb: effective population size of the non-focal source

    Returns
    -------
    coal_time_ratio: np.array
    
    """
    r = kappa
    t = t_div
    p = alpha
    q = 1 - alpha
    return -np.exp(-t / r) * (r - 1.0) * q ** 2 + r * q ** 2 - p * (-2.0 * t * q + p - 2.0)


def tajima_d_admix(t_div, n, Na, Nb, alpha):
    """
    Calculate the Tajima's D statistics for an admixed population
    and one of its source populations immediately after the admixture event.
     

    Arguments
    ---------
    t_div: Time interval in generations

    n: initial (present) number of linages 

    Na: effective population size of the focal source

    Nb: effective population size of the non-focal source

    alpha: proportion of the focal source population in the admixed population

    Returns
    -------
    num_linages: np.array
    
    """
    S1 = branch_length(alpha * n, Na, t_div)
    S2 = branch_length((1 - alpha) * n, Nb, t_div)

    nlina = nlinages(alpha * n, Na, t_div)
    nlinb = nlinages((1 - alpha) * n, Nb, t_div)
    nlinsplit = nlina + nlinb
    S0 = 2 * Na * np.log(nlinsplit - 1.0)

    theta_s = (S1 + S2 + S0) / (sum(1.0 / np.arange(1, n))) / (2 * Na)

    t_coal = t_div / (2 * Na)
    theta_pi = (n / (n - 1)) * admix_coal_time_ratio(t_coal, alpha, Nb / Na)

    td = theta_pi - theta_s

    return td

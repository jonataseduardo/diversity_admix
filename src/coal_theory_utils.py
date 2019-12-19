import numpy as np
from scipy.special import comb

gamma_cte = 0.577


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


def nsites_pop_a(na, Na, exact=False):
    """
    number of seg sites in the source population of constant size in the demographic model.

    Arguments
    ---------
    t_div: Time interval in generations

    na: initial (present) number of linages 

    Na: effective population size of the constant size source

    Returns
    -------
    branch_length: np.array
    
    """
    if exact:
        sa = 2 * Na * np.sum(1 / np.arange(1, na))
    else:
        sa = 2 * Na * (np.log(na) + gamma_cte)
    return sa


def nsites_pop_b(t_div, nb, Na, Nb):
    """
    number of seg sites in the source population of constant size in the demographic model.

    Arguments
    ---------
    t_div: Time interval in generations

    nb: initial (present) number of linages 

    Nb: effective population size of pop a 

    Nb: effective population size of pop b 

    Returns
    -------
    branch_length: np.array
    
    """
    ns_b = nlinages(nb, Nb, t_div)
    sb_before = 2 * Na * (np.log(ns_b) + gamma_cte)
    sb_after = branch_length(nb, Nb, t_div) - t_div
    # sb_after = 2 * Nb * (np.log(nb) - np.log(ns_b))
    sb = sb_before + sb_after
    return sb


def s_admix(t_div, na, nb, Na, Nb):
    """
    number of seg sites in the source population of constant size in the demographic model.

    Arguments
    ---------
    t_div: Time interval in generations

    na: initial (present) number of linages in pop a
    
    nb: initial (present) number of linages in pop b

    Na: effective population size of the constant size source

    Nb: effective population size of the of the second source after the split

    Returns
    -------
    branch_length: np.array
    
    """
    ns_a = nlinages(na, Na, t_div)
    ns_b = nlinages(nb, Nb, t_div)
    s_before = 2 * Na * (np.log(ns_a + ns_b) + gamma_cte)

    sa_after = branch_length(na, Na, t_div)
    sb_after = branch_length(nb, Nb, t_div)
    s_after = sa_after + sb_after - t_div

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
    Sadmix = s_admix(t_div, alpha * n, (1 - alpha) * n, Na, Nb)
    Sa = nsites_pop_a(n, Na, exact=False)
    Sb = nsites_pop_b(t_div, n, Na, Nb)
    # return 1e-4 * Sb
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

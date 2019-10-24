import numpy as np


def branch_length(n, N, T):
    """
    Expected Branch Length of a Single Population in time interval 
    and constant population size

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
    N: effective population size 
    T: Time interval in generations

    Returns
    -------
    num_linages: np.array
    
    """
    return n / (n + (1 - n) * np.exp(-T / (2 * N)))


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
    # gamma_cte from Wakeley book (section 3.3)
    gamma_cte = 0.577216
    S1 = branch_length(alpha * n, Na, t_div)
    S2 = branch_length((1 - alpha) * n, Nb, t_div)

    nlina = nlinages(alpha * n, Na, t_div)
    nlinb = nlinages((1 - alpha) * n, Nb, t_div)
    nlinsplit = nlina + nlinb
    S0 = 2 * Na * np.log(nlinsplit + gamma_cte)
    Sa = 2 * Na * np.sum(1.0 / np.arange(1, n))
    ratio = (S1 + S2 + S0) / Sa
    return ratio


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


def tajima_d_admix(t_div, n, Na, Nb, alpha, k):
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

    t_coal = k * t_div / (2 * Na)
    theta_pi = (n / (n - 1)) * admix_coal_time_ratio(t_coal, alpha, Nb / Na)

    td = theta_pi - theta_s

    return td

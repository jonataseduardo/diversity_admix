import numpy as np


def branch_length(n, N, T):
    """
    Expected Branch Length of a Single Popupulation in time interval 
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
    Expected number of linages of a Single Popupulation of constant effective 
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
    Ratio beteween the number of segreating sites of an admixed population
    and one of its source populations imediately after the admixture event.
     

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
    S0 = 2 * Na * np.log(nlinsplit)
    ratio = (S1 + S2 + S0) / (2 * Na * np.log(n))
    return ratio


def admix_coal_time_ratio(t_div, alpha, kappa):
    """
    Ratio beteween the average coalesncet time of two linages for the admixed
    population and one of its source populations imediately after the admixture
    event.
     

    Arguments
    ---------
    t_div: Time interval in generations coalesncent units of the focal source
    popualtion (generations / Ne for haploids or genearatios / 2 * Ne for
    diplods)

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

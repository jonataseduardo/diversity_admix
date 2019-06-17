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


def s_admix_ratio(n, Na, Nb, alpha, t_div):
    """
    Ratio of the number of segreating sites of beteween an admixed population
    and one of its source populations imediately after the admixture event.
     

    Arguments
    ---------
    n: initial (present) number of linages 
    Na: effective population size of the focal source
    Nb: effective population size of the non-focal source
    t_div: Time interval in generations

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

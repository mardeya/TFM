import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from scipy import integrate


def get_Pk(G):
    r'''
    Calcula la distribució de graus del graf.
    '''
    Nk = Counter(dict(G.degree()).values())
    Pk = {x: Nk[x] / float(G.order()) for x in Nk.keys()}
    return Pk


def get_Pnk(G):
    r'''
    Calcula la distribució de graus dels veïnats condicionada al grau.
    '''
    Pnk = {k1: defaultdict(int) for k1 in dict(G.degree()).values()}
    Nk = Counter(dict(G.degree()).values())

    for node in G.nodes():
        k1 = G.degree(node)
        nbr_degrees = [G.degree(nbr) for nbr in G.neighbors(node)]
        for k2 in nbr_degrees:
            Pnk[k1][k2] += 1.0 / (k1 * Nk[k1])

    return Pnk


def _dEBCM_pref_mix_(X, t, rho, tau, gamma, Pk, Pnk):
    R = X[0]
    theta = {}
    phiR = {}

    for index, k in enumerate(sorted(Pk.keys())):
        theta[k] = X[1 + 2 * index]
        phiR[k] = X[2 + 2 * index]

    S = (1 - rho) * sum([Pk[k] * theta[k] ** k for k in Pk.keys()])
    I = 1 - S - R

    returnval = [gamma * I]

    phiS = {}
    phiI = {}

    for k1 in Pk.keys():
        phiS[k1] = (1 - rho) * sum(
            [Pnk[k1][k2] * theta[k2] ** (k2 - 1) for k2 in Pnk[k1].keys()]
        )
        phiI[k1] = theta[k1] - phiS[k1] - phiR[k1]

    for k in sorted(Pk.keys()):
        dthetak_dt = -tau * phiI[k]
        dphiRk_dt = gamma * phiI[k]
        returnval.extend([dthetak_dt, dphiRk_dt])

    return np.array(returnval)


def EBCM_pref_mix(
    N,
    Pk,
    Pnk,
    tau,
    gamma,
    rho=None,
    tmin=0,
    tmax=100,
    tcount=1001,
    return_full_data=False
):
    r'''
    Resol l'EBCM amb correlacions entre graus.

    '''
    if rho is None:
        rho = 1.0 / N

    ts = np.linspace(tmin, tmax, tcount)

    IC = [0]
    for k in sorted(Pk.keys()):
        IC.extend([1, 0])

    IC = np.array(IC)

    X = integrate.odeint(
        _dEBCM_pref_mix_,
        IC,
        ts,
        args=(rho, tau, gamma, Pk, Pnk)
    )

    R = X.T[0]
    theta = {}
    phiR = {}

    for index, k in enumerate(sorted(Pk.keys())):
        theta[k] = X.T[1 + 2 * index]
        phiR[k] = X.T[1 + 2 * index]

    S = (1 - rho) * sum([Pk[k] * theta[k] ** k for k in Pk.keys()])
    I = 1 - S - R

    if return_full_data:
        return ts, N * S, N * I, N * R, theta
    else:
        return ts, N * S, N * I, N * R


def EBCM_corr(
    G,
    tau,
    gamma,
    rho=None,
    tmin=0,
    tmax=100,
    tcount=1001,
    return_full_data=False
):
    r'''
    Calcula les correlacions de grau del graf i resol l'EBCM.

    '''
    N = G.order()
    Pk = get_Pk(G)
    Pnk = get_Pnk(G)

    return EBCM_pref_mix(
        N,
        Pk,
        Pnk,
        tau,
        gamma,
        rho=rho,
        tmin=tmin,
        tmax=tmax,
        tcount=tcount,
        return_full_data=return_full_data
    )
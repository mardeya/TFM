import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from scipy import integrate

def get_Pk(G):
    r'''
    Used in several places so that we can input a graph and then we 
    can call the methods that depend on the degree distribution

    :Arguments: 

    **G** networkx Graph
    
    :Returns: 

    **Pk** dict
        ``Pk[k]`` is the proportion of nodes with degree ``k``.
    '''

    Nk = Counter(dict(G.degree()).values())
    Pk = {x:Nk[x]/float(G.order()) for x in Nk.keys()}
    return Pk


def get_Pnk(G):

    r'''    
    :Arguments: 

    **G** networkx Graph

    :Returns: 

    **Pnk** dict
        ``Pnk[k1][k2]`` is the proportion of neighbors of degree ``k1`` nodes 
        that have degree ``k2``.
    '''
    Pnk = {k1:defaultdict(int)  for k1 in dict(G.degree()).values()}
    Nk = Counter(dict(G.degree()).values())

    for node in G.nodes():
        k1 = G.degree(node)
        nbr_degrees = [G.degree(nbr) for nbr in G.neighbors(node)]
        for k2 in nbr_degrees:
            Pnk[k1][k2] += 1./(k1*Nk[k1])
    return Pnk

    
def _dEBCM_pref_mix_(X, t, rho, tau, gamma, Pk, Pnk):
    #print t
    R= X[0]
    theta = {}
    phiR={}
    for index, k in enumerate(sorted(Pk.keys())):
        theta[k] = X[1+2*index]
        phiR[k] = X[2+2*index]
    S = (1-rho)*sum([Pk[k]*theta[k]**k for k in Pk.keys()])
    #print 'S', S
    I = 1 - S - R
    returnval = [gamma*I]#xidot, I2dot, Rdot
    phiS = {}
    phiI = {}
    for k1 in Pk.keys():
        phiS[k1] = (1-rho)*sum([Pnk[k1][k2]*theta[k2]**(k2-1) for k2 in Pnk[k1].keys()])
        phiI[k1] = theta[k1] - phiS[k1]  - phiR[k1]
    for k in sorted(Pk.keys()):
        dthetak_dt = -tau*phiI[k]
        dphiRk_dt = gamma*phiI[k]
        returnval.extend([dthetak_dt, dphiRk_dt])
    return np.array(returnval)

#N, psihat, psihatPrime, tau, gamma, phiS0, phiR0=0, R0=0, tmin=0, 
#            tmax=100, tcount=1001, return_full_data=False
def EBCM_pref_mix(N, Pk, Pnk, tau, gamma, rho = None, tmin = 0, tmax = 100, tcount = 1001, return_full_data=False):
    r'''
    
    Encodes the system derived in exercise 6.21 of Kiss, Miller, & Simon.  Please cite the
    book if using this algorithm.

    I anticipate eventually adding an option so that the initial condition is
    not uniformly distributed.  So could give rho_k
    
    :Arguments: 
    **N**  positive integer
        number of nodes.
    **Pk**  dict  (could also be an array or a list)
        Pk[k] is the probability a random node has degree k.
    **Pnk** dict of dicts (possibly array/list)
        Pnk[k1][k2] is the probability a neighbor of a degree k1 node has
        degree k2.
    **tau** positive float
        transmission rate
    **gamma** number
        recovery rate
    **rho**  number (optional)
        initial proportion infected.  Defaults to 1/N.
    **tmin**  number (default 0)
        minimum time
    **tmax**  number (default 100)
        maximum time
    **tcount**  integer (default 1001)
        number of time points for data (including end points)
    **return_full_data**  boolean (default False)
        whether to return theta or not
            

    :Returns: 

    if return_full_data == False:
        returns **t, S, I, R**, all numpy arrays
    if ...== True
        returns **t, S, I, R** and **theta**
        where theta[k] is a numpy array giving theta for degree k
            
    '''
    if rho is None:
        rho = 1./N
        
    ts = np.linspace(tmin, tmax, tcount)
    IC = [0] #R(0)
    for k in sorted(Pk.keys()):
        IC.extend([1,0]) #theta_k(0), phiR_k(0)
    IC = np.array(IC)
    X = integrate.odeint(_dEBCM_pref_mix_, IC, ts, args = (rho, tau, gamma, Pk, Pnk))
    R =  X.T[0]
    theta = {}
    phiR={}
    for index, k in enumerate(sorted(Pk.keys())):
        theta[k] = X.T[1+2*index]
        phiR[k] = X.T[1+2*index]
    
    S = (1-rho)*sum([Pk[k]*theta[k]**k for k in Pk.keys()])
    I = 1-S-R
    if return_full_data:
        return ts, N*S, N*I, N*R, theta
    else:
        return ts, N*S, N*I, N*R

def EBCM_corr(G, tau, gamma, rho = None, tmin = 0, tmax = 100, tcount = 1001, return_full_data=False):
    r'''
    Takes a given graph, finds degree correlations, and calls EBCM_pref_mix
    
    
    I anticipate eventually adding an option so that the initial condition is
    not uniformly distributed.  So could give rho_k
    
    :Arguments: 
    **G** networkx Graph
        The contact network
    **tau** positive float
        transmission rate
    **gamma** positive float
        recovery rate
    **rho**  positive float (default ``None``)
        initial proportion infected.  Defaults to 1/N.
    **tmin**  number (default 0)
        minimum time
    **tmax**  number (default 100)
        maximum time
    **tcount**  integer (default 1001)
        number of time points for data (including end points)
    **return_full_data**  boolean (default False)
        whether to return theta or not
            

    :Returns: 

    if return_full_data == False:
        returns **t, S, I, R**, all numpy arrays
    if ...== True
        returns **t, S, I, R** and **theta**, 
        where theta[k] is a numpy array giving theta for degree k
    '''

    N=G.order()
    Pk = get_Pk(G)
    Pnk = get_Pnk(G)
    return EBCM_pref_mix(N, Pk, Pnk, tau, gamma, rho = rho, tmin = tmin, tmax = tmax, tcount = tcount, return_full_data=return_full_data)
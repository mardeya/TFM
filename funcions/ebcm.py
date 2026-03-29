import numpy as np
import networkx as nx
from collections import Counter, defaultdict
from scipy import integrate


# ---------------------------------------------------------------------
# Distribució de graus
# ---------------------------------------------------------------------
def obtenir_Pk(graf: nx.Graph) -> dict[int, float]:
    r"""
    Calcula la distribució de graus empírica del graf.

    Paràmetres
    ----------
    graf : nx.Graph
        Graf de NetworkX.

    Retorna
    -------
    Pk : dict[int, float]
        Diccionari tal que Pk[k] és la proporció de nodes amb grau k.
    """
    Nk = Counter(dict(graf.degree()).values())
    N = float(graf.number_of_nodes())
    return {k: Nk[k] / N for k in Nk.keys()}


# ---------------------------------------------------------------------
# Inicialització d'estats (sense EoN)
# ---------------------------------------------------------------------
def inicialitzar_estat_nodes(
    graf: nx.Graph,
    infectats_inicials,
    recuperats_inicials=None,
) -> dict:
    """
    Assigna un estat inicial a cada node: 'S', 'I' o 'R'.

    - Per defecte, tots els nodes són susceptibles 'S'.
    - Els nodes a infectats_inicials passen a 'I'.
    - Els nodes a recuperats_inicials passen a 'R'.

    Valida que:
      (i) no hi hagi intersecció entre infectats i recuperats,
      (ii) tots els nodes indicats existeixin al graf.

    Retorna
    -------
    estat : dict
        Diccionari node -> 'S'/'I'/'R'.
    """
    if recuperats_inicials is None:
        recuperats_inicials = []

    infectats_inicials = list(infectats_inicials)
    recuperats_inicials = list(recuperats_inicials)

    interseccio = set(infectats_inicials).intersection(set(recuperats_inicials))
    if interseccio:
        raise ValueError(f"{sorted(interseccio)} apareixen tant a infectats com a recuperats inicials.")

    estat = defaultdict(lambda: "S")

    for node in infectats_inicials:
        if not graf.has_node(node):
            raise ValueError(f"El node {node} no existeix al graf.")
        estat[node] = "I"

    for node in recuperats_inicials:
        if not graf.has_node(node):
            raise ValueError(f"El node {node} no existeix al graf.")
        estat[node] = "R"

    return dict(estat)


# ---------------------------------------------------------------------
# EBCM: integració del sistema (Kiss, Miller & Simon, eq. (6.12))
# ---------------------------------------------------------------------
def EBCM(
    N: int,
    psihat,
    psihat_derivada,
    tau: float,
    gamma: float,
    phiS0: float,
    phiR0: float = 0.0,
    R0: float = 0.0,
    tmin: float = 0.0,
    tmax: float = 100.0,
    tcount: int = 1001,
    retornar_dades_completes: bool = False,
):
    """
    Integra el sistema EBCM (edge-based compartmental model) per a SIR Markovià.

    Variables d'estat integrades:
      - theta(t): probabilitat que una aresta "cap a un veí" NO hagi transmès cap a l'arrel
      - R(t): nombre acumulat de recuperats

    A partir d'això:
      - S(t) = N * psihat(theta(t))
      - I(t) = N - S(t) - R(t)

    Retorna
    -------
    Si retornar_dades_completes és False:
        times, S, I, R
    Si True:
        times, S, I, R, theta
    """
    temps = np.linspace(tmin, tmax, tcount)
    X0 = np.array([1.0, float(R0)], dtype=float)

    X = integrate.odeint(
        _dEBCM_,
        X0,
        temps,
        args=(N, tau, gamma, psihat, psihat_derivada, phiS0, phiR0),
    )

    theta = X[:, 0]
    R = X[:, 1]
    S = N * psihat(theta)
    I = N - S - R

    if not retornar_dades_completes:
        return temps, S, I, R
    return temps, S, I, R, theta


def _dEBCM_(X, t, N, tau, gamma, psihat, psihat_derivada, phiS0, phiR0):
    """
    RHS del sistema EBCM (eq. 6.12 en notació equivalent).

    dtheta = -tau*theta + tau*phiS0 * psihat'(theta)/psihat'(1) + gamma*(1-theta) + tau*phiR0
    dR     = gamma * I
    """
    theta = float(X[0])
    R = float(X[1])

    denom = psihat_derivada(1.0)
    if denom == 0:
        # cas degenerat (sense arestes efectives): theta no canvia per infecció
        dtheta = gamma * (1.0 - theta) + tau * phiR0
    else:
        dtheta = (
            -tau * theta
            + tau * phiS0 * psihat_derivada(theta) / denom
            + gamma * (1.0 - theta)
            + tau * phiR0
        )

    S = N * psihat(theta)
    I = N - S - R
    dR = gamma * I

    return np.array([dtheta, dR], dtype=float)


# ---------------------------------------------------------------------
# EBCM a partir d’un graf de NetworkX (funció "general")
# ---------------------------------------------------------------------
def EBCM_graf(
    graf: nx.Graph,
    tau: float,
    gamma: float,
    infectats_inicials=None,
    recuperats_inicials=None,
    rho: float | None = None,
    tmin: float = 0.0,
    tmax: float = 100.0,
    tcount: int = 1001,
    retornar_dades_completes: bool = False,
):
    r"""
    Donat un graf de NetworkX, construeix automàticament:
      - N
      - psihat(x) i psihat'(x)
      - phiS0 i phiR0
      - R0
    i crida EBCM.

    Inicialització:
    - Si passes infectats_inicials (llista/iterable de nodes), s'usa exactament aquesta condició inicial.
    - Alternativament, pots passar rho (fracció infectada inicial), assumint selecció uniforme.
      (En aquest cas no pots passar recuperats_inicials ni infectats_inicials.)

    Retorna
    -------
    times, S, I, R (i opcionalment theta)
    """
    if rho is not None and infectats_inicials is not None:
        raise ValueError("No pots definir alhora infectats_inicials i rho.")
    if rho is not None and recuperats_inicials is not None:
        raise ValueError("No pots definir alhora recuperats_inicials i rho.")

    Pk = obtenir_Pk(graf)
    N = graf.number_of_nodes()

    if N == 0:
        raise ValueError("El graf no té nodes.")

    # Cas 1: condició inicial explícita per nodes
    if infectats_inicials is not None:
        estat = inicialitzar_estat_nodes(
            graf,
            infectats_inicials=infectats_inicials,
            recuperats_inicials=recuperats_inicials,
        )

        Nk = Counter(dict(graf.degree()).values())
        maxk = max(Nk.keys()) if Nk else 0
        Nk_vec = np.array([Nk.get(k, 0) for k in range(maxk + 1)], dtype=float)

        # Comptatges per estimar phiS0 i phiR0:
        # SS = nombre d'arestes orientades (S->S) mirant des de susceptibles
        # SR = nombre d'arestes orientades (S->R)
        # SX = total d'arestes orientades sortints des de nodes susceptibles (sumatori graus de S)
        SS = 0.0
        SR = 0.0
        SX = 0.0
        R0 = 0.0

        # Sk0[k] ~ "pes" de susceptibles per grau: s'utilitza per definir psihat
        # La fórmula original feia servir 1/Nk[k] per node, i Pk[k] a psihat.
        Sk0 = np.zeros(maxk + 1, dtype=float)

        for node in graf.nodes():
            est = estat.get(node, "S")
            if est == "S":
                k = graf.degree(node)
                if Nk_vec[k] > 0:
                    Sk0[k] += 1.0 / Nk_vec[k]

                # comptem veïns segons estat
                ss_node = 0
                sr_node = 0
                for nbr in graf.neighbors(node):
                    if estat.get(nbr, "S") == "S":
                        ss_node += 1
                    elif estat.get(nbr, "S") == "R":
                        sr_node += 1

                SS += ss_node
                SR += sr_node
                SX += k

            elif est == "R":
                R0 += 1.0

        def psihat(x):
            x = np.asarray(x, dtype=float)
            return sum(Pk[k] * Sk0[k] * (x ** k) for k in Pk)

        def psihat_derivada(x):
            x = np.asarray(x, dtype=float)
            return sum(k * Pk[k] * Sk0[k] * (x ** (k - 1)) for k in Pk if k >= 1)

        if SX <= 0:
            phiS0 = 0.0
            phiR0 = 0.0
        else:
            phiS0 = SS / SX
            phiR0 = SR / SX

    # Cas 2: inicialització "homogènia" per fracció rho
    else:
        if rho is None:
            rho = 1.0 / N  # un infectat inicial de mitjana

        rho = float(rho)
        if not (0.0 <= rho <= 1.0):
            raise ValueError("rho ha d'estar entre 0 i 1.")

        def psihat(x):
            x = np.asarray(x, dtype=float)
            return (1.0 - rho) * sum(Pk[k] * (x ** k) for k in Pk)

        def psihat_derivada(x):
            x = np.asarray(x, dtype=float)
            return (1.0 - rho) * sum(k * Pk[k] * (x ** (k - 1)) for k in Pk if k >= 1)

        phiS0 = 1.0 - rho
        phiR0 = 0.0
        R0 = 0.0

    return EBCM(
        N=N,
        psihat=psihat,
        psihat_derivada=psihat_derivada,
        tau=tau,
        gamma=gamma,
        phiS0=phiS0,
        phiR0=phiR0,
        R0=R0,
        tmin=tmin,
        tmax=tmax,
        tcount=tcount,
        retornar_dades_completes=retornar_dades_completes,
    )
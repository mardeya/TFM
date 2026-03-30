import numpy as np
import networkx as nx

from funcions.percolacio_dirigida import percolacio_dirigida


# Funcions auxiliars (PGF + punt fix)
def distribucio_graus(graf):
    """Retorna pk (vector indexat per k) i <k> a partir dels graus empírics del graf."""
    graus = np.fromiter((d for _, d in graf.degree()), dtype=int)
    if graus.size == 0:
        return np.array([1.0], dtype=float), 0.0

    kmax = int(graus.max())
    comptatges = np.bincount(graus, minlength=kmax + 1).astype(float)
    pk = comptatges / comptatges.sum()
    k_mitja = float((np.arange(len(pk)) * pk).sum())
    return pk, k_mitja

def G0(x, pk):
    """G0(x) = sum_k p_k x^k."""
    k = np.arange(len(pk), dtype=float)
    return float(np.sum(pk * (x ** k)))

def G1(x, pk, k_mitja):
    """G1(x) = sum_{k>=1} (k p_k / <k>) x^{k-1}."""
    if k_mitja <= 0:
        return 0.0
    k = np.arange(len(pk), dtype=float)
    return float(np.sum((k[1:] * pk[1:] / k_mitja) * (x ** (k[1:] - 1.0))))


def punt_fix_chi(
    T,
    pk,
    k_mitja,
    tol=1e-12,
    max_iter=50000
):
    """
    Resol chi = 1 - T + T * G1(chi).

    Inicialitzem lleugerament per sota d'1 per
    evitar quedar-nos al punt fix trivial chi=1 quan el sistema és supercrític.
    """
    if T <= 0 or k_mitja <= 0:
        return 1.0

    chi = 1.0 - 1e-10
    for _ in range(max_iter):
        chi_nou = 1.0 - T + T * G1(chi, pk, k_mitja)
        chi_nou = float(np.clip(chi_nou, 0.0, 1.0))
        if abs(chi_nou - chi) < tol:
            return chi_nou
        chi = chi_nou

    return chi


def probabilitat_epidemia_teoria(
    tau,
    gamma,
    pk,
    k_mitja
):
    """
    Teoria (Configuration-Model / EBCM estàtica):
      T = tau/(tau+gamma)
      P_th = 1 - G0(chi), amb chi = 1 - T + T G1(chi)
    """
    if tau <= 0:
        return 0.0
    T = tau / (tau + gamma)
    chi = punt_fix_chi(T, pk, k_mitja)
    Pth = 1.0 - G0(chi, pk)
    return float(np.clip(Pth, 0.0, 1.0))


# Simulació EPN: P_sim 
def fraccio_HIN(H: nx.DiGraph) -> float:
    """
    Calcula |HIN + HSCC|/N a partir del graf dirigit H:
      - HIN + HSCC 
    """
    N = H.number_of_nodes()
    if N == 0 or H.number_of_edges() == 0:
        return 0.0

    sccs = list(nx.strongly_connected_components(H))
    if not sccs:
        return 0.0

    C = nx.condensation(H, sccs)
    hscc = max(C.nodes(), key=lambda c: len(C.nodes[c]["members"]))

    hin_scc = {hscc} | nx.ancestors(C, hscc)
    compta = sum(len(C.nodes[c]["members"]) for c in hin_scc)
    return compta / N



# Funció GENERAL: entra un graf ja generat i retorna Pth i Psim
def probabilitat_epidemia_en_graf(
    graf,
    tau_vals,
    gamma,
    percolacions_per_tau,
    seed,
    weights = True
):
    """
    Estima la probabilitat d'epidèmia per un graf fix de NetworkX.
    """
    # --- Teoria 
    pk, k_mitja = distribucio_graus(graf)
    Pth = np.array(
        [probabilitat_epidemia_teoria(float(tau), gamma, pk, k_mitja) for tau in tau_vals],
        dtype=float
    )

    # --- Simulació (EPN) amb mitjana sobre percolacions
    rng = np.random.default_rng(seed)
    Psim = np.zeros_like(tau_vals, dtype=float)

    for i, tau in enumerate(tau_vals):
        acum = 0.0
        for _ in range(percolacions_per_tau):
            _ = rng.integers(0, 2**31 - 1)  
            H = percolacio_dirigida(graf, tau=float(tau), gamma=gamma)
            acum += fraccio_HIN(H)
        Psim[i] = acum / percolacions_per_tau

    return Pth, Psim
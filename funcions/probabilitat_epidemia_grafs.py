import numpy as np
import networkx as nx

from funcions.percolacio_dirigida import percolacio_dirigida


# ---------------------------------------------------------------------
# Funcions auxiliars (PGF + punt fix)
# ---------------------------------------------------------------------
def distribucio_graus(graf: nx.Graph) -> tuple[np.ndarray, float]:
    """Retorna pk (vector indexat per k) i <k> a partir dels graus empírics del graf."""
    graus = np.fromiter((d for _, d in graf.degree()), dtype=int)
    if graus.size == 0:
        return np.array([1.0], dtype=float), 0.0

    kmax = int(graus.max())
    comptatges = np.bincount(graus, minlength=kmax + 1).astype(float)
    pk = comptatges / comptatges.sum()
    k_mitja = float((np.arange(len(pk)) * pk).sum())
    return pk, k_mitja


def G0(x: float, pk: np.ndarray) -> float:
    """G0(x) = sum_k p_k x^k."""
    k = np.arange(len(pk), dtype=float)
    return float(np.sum(pk * (x ** k)))


def G1(x: float, pk: np.ndarray, k_mitja: float) -> float:
    """G1(x) = sum_{k>=1} (k p_k / <k>) x^{k-1}."""
    if k_mitja <= 0:
        return 0.0
    k = np.arange(len(pk), dtype=float)
    return float(np.sum((k[1:] * pk[1:] / k_mitja) * (x ** (k[1:] - 1.0))))


def punt_fix_chi(
    T: float,
    pk: np.ndarray,
    k_mitja: float,
    tol: float = 1e-12,
    max_iter: int = 50_000
) -> float:
    """
    Resol chi = 1 - T + T * G1(chi).

    Detall numèric important: inicialitzem lleugerament per sota d'1 per
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
    tau: float,
    gamma: float,
    pk: np.ndarray,
    k_mitja: float
) -> float:
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


# ---------------------------------------------------------------------
# Simulació EPN: P_sim = |HIN|/N
# ---------------------------------------------------------------------
def fraccio_HIN(H: nx.DiGraph) -> float:
    """
    Calcula |HIN|/N a partir del graf dirigit H:
      - Condensem SCCs => DAG
      - Troben HSCC (SCC més gran)
      - HIN = HSCC + tots els ancestres al DAG de condensació
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


# ---------------------------------------------------------------------
# Funció GENERAL: entra un graf ja generat i retorna Pth i Psim
# ---------------------------------------------------------------------
def probabilitat_epidemia_en_graf(
    graf: nx.Graph,
    tau_vals: np.ndarray,
    gamma: float,
    percolacions_per_tau: int = 1,
    seed: int | None = None,
    weights: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estima la probabilitat d'epidèmia per un graf fix de NetworkX.

    Paràmetres
    ----------
    graf:
        Graf de NetworkX (no dirigit) ja generat.
    tau_vals:
        Vector de valors de tau.
    gamma:
        Taxa de recuperació.
    percolacions_per_tau:
        Nombre de realitzacions EPN per cada tau (mitjana Monte Carlo).
    seed:
        Seed per repetir (controla els seeds interns a directed_percolate_network).
    weights:
        Es passa tal qual a directed_percolate_network(..., weights=weights).

    Retorna
    -------
    (Pth, Psim):
        Pth: teoria punt fix (basada en la distribució de graus empírica del graf)
        Psim: simulació EPN, mitjana de |HIN|/N
    """
    # --- Teoria (un cop, perquè el graf és fix)
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
            # Seed diferent per cada percolació (si vols reproductibilitat)
            _ = rng.integers(0, 2**31 - 1)  # consumim RNG perquè variï internament
            H = percolacio_dirigida(graf, tau=float(tau), gamma=gamma)
            acum += fraccio_HIN(H)
        Psim[i] = acum / percolacions_per_tau

    return Pth, Psim
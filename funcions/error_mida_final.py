import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from funcions.percolacio_dirigida import percolacio_dirigida


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


def punt_fix_omega(T, pk, k_mitja, tol=1e-12, max_iter=100_000):
    """
    Resol omega = 1 - T + T*G1(omega) per iteració de punt fix.

    Detall numèric: comencem lleugerament per sota d'1 per evitar quedar-nos
    al punt fix trivial omega=1 quan el sistema és supercrític.
    """
    if T <= 0 or k_mitja <= 0:
        return 1.0

    omega = 1.0 - 1e-10
    for _ in range(max_iter):
        omega_nou = 1.0 - T + T * G1(omega, pk, k_mitja)
        omega_nou = float(np.clip(omega_nou, 0.0, 1.0))
        if abs(omega_nou - omega) < tol:
            return omega_nou
        omega = omega_nou

    return omega


def mida_final_teorica(tau, gamma, pk, k_mitja):
    """
    Mida final (attack rate) en xarxes localment arborescents:
      T = tau/(tau+gamma)
      A_th = 1 - G0(omega)
      omega resol: omega = 1 - T + T G1(omega)
    """
    if tau <= 0 or gamma <= 0:
        return 0.0

    T = tau / (tau + gamma)
    omega = punt_fix_omega(T, pk, k_mitja)
    Ath = 1.0 - G0(omega, pk)
    return float(np.clip(Ath, 0.0, 1.0))


def fraccio_HOUT(H):
    """
    Aproxima la mida final via EPN:
      A_sim ≈ |HOUT + HSCC|/N
    """
    N = H.number_of_nodes()
    if N == 0 or H.number_of_edges() == 0:
        return 0.0

    sccs = list(nx.strongly_connected_components(H))
    if not sccs:
        return 0.0

    C = nx.condensation(H, sccs)
    hscc = max(C.nodes(), key=lambda c: len(C.nodes[c]["members"]))

    hout_scc = {hscc} | nx.descendants(C, hscc)
    compta = sum(len(C.nodes[c]["members"]) for c in hout_scc)
    return compta / N


def mida_final_epidemia_en_graf(
    graf,
    tau_vals,
    gamma,
    percolacions_per_tau=1,
    seed=None,
    weights=True,
):
    """
    Calcula mida final (attack rate) teòrica vs simulada (EPN) per un graf fix.

    (Ath, Asim):
      Ath[i] = mida final teòrica per tau_vals[i]
      Asim[i] = estimació EPN: mitjana de |HOUT + HSCC|/N sobre percolacions_per_tau
    """
    pk, k_mitja = distribucio_graus(graf)
    Ath = np.array(
        [mida_final_teorica(float(tau), gamma, pk, k_mitja) for tau in tau_vals],
        dtype=float,
    )

    rng = np.random.default_rng(seed)
    Asim = np.zeros_like(tau_vals, dtype=float)

    for i, tau in enumerate(tau_vals):
        acum = 0.0
        for _ in range(percolacions_per_tau):
            _ = rng.integers(0, 2**31 - 1)
            H = percolacio_dirigida(graf, tau=float(tau), gamma=gamma)
            acum += fraccio_HOUT(H)
        Asim[i] = acum / percolacions_per_tau

    return Ath, Asim


def plot_error_mida_final(
    tau_vals,
    Ath,
    Asim,
    tau_c=None,
    fitxer_sortida=None,
    mostrar=True,
):
    """Plot de l'error Asim - Ath."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(0.0, linewidth=1.0, color="black")
    ax.plot(
        tau_vals,
        Asim - Ath,
        linewidth=1.6,
        label=r"$\mathcal{A}_{\mathrm{sim}}-\mathcal{A}_{\mathrm{teòric}}$",
        color="#6D597A",
    )

    if tau_c is not None:
        ax.axvline(tau_c, linestyle="--", linewidth=1.4, color="#2A9D8F")

    ax.set_xlabel(r"$\tau$", fontsize=16)
    ax.set_ylabel(r"Error en el càlcul $\mathcal{A}$", fontsize=16)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=True, loc="best", fontsize=16)
    plt.ylim(-0.6, 0.1)
    plt.tight_layout()

    if fitxer_sortida is not None:
        carpeta = os.path.dirname(fitxer_sortida)
        if carpeta:
            os.makedirs(carpeta, exist_ok=True)
        plt.savefig(fitxer_sortida, dpi=300)

    if mostrar:
        plt.show()

    return fig, ax


def plot_mida_final_teoria_vs_sim(
    tau_vals,
    Ath,
    Asim,
    gamma,
    titol=None,
    fitxer_sortida=None,
    mostrar=True,
):
    """Plot Ath vs Asim."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(tau_vals, Ath, linewidth=1.6, label=r"Teoria (punt fix)", color="#2A9D8F")
    ax.plot(tau_vals, Asim, linewidth=1.6, label=r"Simulació (EPN)", color="#6D597A")

    ax.set_xlabel(r"$\tau$", fontsize=16)
    ax.set_ylabel(r"$\mathcal{A}$", fontsize=16)
    ax.set_ylim(-0.02, 1.02)

    if titol is not None:
        ax.set_title(titol)
    else:
        ax.set_title(rf"Mida final de l'epidèmia, $\gamma={gamma}$")

    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=True, loc="best", fontsize=16)
    plt.tight_layout()

    if fitxer_sortida is not None:
        carpeta = os.path.dirname(fitxer_sortida)
        if carpeta:
            os.makedirs(carpeta, exist_ok=True)
        plt.savefig(fitxer_sortida, dpi=200)

    if mostrar:
        plt.show()

    return fig, ax
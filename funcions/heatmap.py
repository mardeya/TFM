import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from funcions.percolacio_dirigida import percolacio_dirigida


PALETA_TFM = ['#2A9D8F', "#A690B5", '#6D597A', "#1E414F"]
cmap_tfm = LinearSegmentedColormap.from_list("tfm", PALETA_TFM, N=256)


def moments_grau(graf: nx.Graph) -> tuple[float, float]:  
    """Retorna (<K>, <K^2>) del graf."""
    graus = np.fromiter((d for _, d in graf.degree()), dtype=float)
    if graus.size == 0:
        return 0.0, 0.0
    return float(graus.mean()), float((graus ** 2).mean())


def fraccio_epidemica_attack_rate(graf_dirigit: nx.DiGraph) -> float:
    """
    Aproximació de la mida epidèmica via percolació dirigida:
    |HSCC ∪ HOUT| / N, on HSCC és la SCC gegant del graf de condensació.
    """
    N = graf_dirigit.number_of_nodes()
    if N == 0 or graf_dirigit.number_of_edges() == 0:
        return 0.0

    sccs = list(nx.strongly_connected_components(graf_dirigit))
    if not sccs:
        return 0.0

    condensat = nx.condensation(graf_dirigit, sccs)
    hscc = max(condensat.nodes(), key=lambda c: len(condensat.nodes[c]["members"]))
    scc_assolibles = {hscc} | nx.descendants(condensat, hscc)

    compta = sum(len(condensat.nodes[c]["members"]) for c in scc_assolibles)
    return compta / N



# Heatmap

def heatmap_attack_rate(
    generador_graf,               # callable: generador_graf(seed=..., **params) -> nx.Graph
    params_graf: dict,            # paràmetres del generador (ex: {"N":1000, "k":4, "p":0.8})
    tau_vals: np.ndarray | None = None,
    gamma_vals: np.ndarray | None = None,
    realitzacions: int = 10,
    seed: int = 12345,
    funcio_llindar=None,         
    guardar_figura: str | None = None,
    mostrar: bool = True,
):
    """
    Calcula el heatmap Z[ig, it] = E[ attack_rate ] en una graella (gamma, tau),
    on attack_rate = |HSCC ∪ HOUT|/N del graf percolat dirigit.

    Retorna:
      tau_vals, gamma_vals, Z, k1_mitja, k2_mitja, (fig, ax) si es dibuixa.
    """
    if tau_vals is None:
        tau_vals = np.linspace(0.0, 2.0, 41)
    if gamma_vals is None:
        gamma_vals = np.linspace(0.5, 2.0, 31)

    rng = np.random.default_rng(seed)
    Z = np.zeros((len(gamma_vals), len(tau_vals)), dtype=float)

    k1_list, k2_list = [], []

    for r in range(realitzacions):
        print("realització:", r)
        seed_r = int(rng.integers(0, 2**31 - 1))

        # Generam un graf per realització
        graf = generador_graf(seed=seed_r, **params_graf)

        # Moments empírics (per fer una línia llindar "mitjana" si convé)
        k1, k2 = moments_grau(graf)
        k1_list.append(k1)
        k2_list.append(k2)

        # Graella (gamma, tau)
        for ig, gamma in enumerate(gamma_vals):
            for it, tau in enumerate(tau_vals):
                graf_dir = percolacio_dirigida(graf, tau=tau, gamma=gamma)
                Z[ig, it] += fraccio_epidemica_attack_rate(graf_dir)

    Z /= realitzacions
    k1_mitja = float(np.mean(k1_list)) if k1_list else 0.0
    k2_mitja = float(np.mean(k2_list)) if k2_list else 0.0

    
    fig, ax = None, None
    if guardar_figura is not None or mostrar:
        fig, ax = plt.subplots(figsize=(7.2, 4.8))

        im = ax.imshow(
            Z,
            origin="lower",
            aspect="auto",
            cmap=cmap_tfm,
            extent=[tau_vals.min(), tau_vals.max(), gamma_vals.min(), gamma_vals.max()],
            vmin=0.0,
            vmax=1.0,
        )
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label(r"Proporció de nodes infectats", fontsize=16)

        # Llindar opcional
        if funcio_llindar is not None:
            gamma_line = np.linspace(gamma_vals.min(), gamma_vals.max(), 400)
            tau_line = funcio_llindar(gamma_line, k1_mitja, k2_mitja, params_graf)
            if tau_line is not None:
                mask = (tau_line >= tau_vals.min()) & (tau_line <= tau_vals.max())
                ax.plot(tau_line[mask], gamma_line[mask], color="black", linewidth=1.0)

        ax.set_xlabel(r"$\tau$", fontsize=16)
        ax.set_ylabel(r"$\gamma$", fontsize=16)
        ax.tick_params(direction="in", top=True, right=True)

        if guardar_figura is not None:
            plt.savefig(guardar_figura, dpi=300, bbox_inches="tight")
        if mostrar:
            plt.show()
        else:
            plt.close(fig)

    return tau_vals, gamma_vals, Z, k1_mitja, k2_mitja, fig, ax


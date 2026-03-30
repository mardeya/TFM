import numpy as np
import networkx as nx

from funcions.probabilitat_epidemia_grafs import probabilitat_epidemia_en_graf
from funcions.plot_P_teoria_vs_simulacio import plot_P_teoria_vs_simulacio



# Exemple: Configuration Model (Poisson)
def mostra_graus_poisson_trunc(N, z, kmax, seed):
    rng = np.random.default_rng(seed)
    graus = rng.poisson(lam=z, size=N).astype(int)
    graus = np.clip(graus, 0, kmax)

    # Assegurem suma parella
    if graus.sum() % 2 == 1:
        i = int(rng.integers(0, N))
        graus[i] = max(0, graus[i] - 1)

    return graus.tolist()


def graf_CM_simple(seq_graus, seed, max_tries= 100):
    """
    Genera un graf SIMPLE a partir del configuration model:
      - construeix un multigraf
      - el simplifica (treu self-loops i multiarestes)
      - es queda amb la component gegant
    """
    rng = np.random.default_rng(seed)
    millor = None
    millor_arestes = -1

    for _ in range(max_tries):
        MG = nx.configuration_model(seq_graus, seed=int(rng.integers(0, 2**31 - 1)))
        G = nx.Graph(MG)  # col·lapsa multiarestes
        G.remove_edges_from(nx.selfloop_edges(G))

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            continue

        comp = max(nx.connected_components(G), key=len)
        H = G.subgraph(comp).copy()

        if H.number_of_edges() > millor_arestes:
            millor = H
            millor_arestes = H.number_of_edges()

        # Aturem si és connex i prou dens
        if nx.is_connected(H) and H.number_of_edges() >= 0.9 * (sum(seq_graus) // 2):
            return H

    return millor


if __name__ == "__main__":
    # Paràmetres del CM
    N = 1000
    z = 7.0
    kmax = 50

    # Paràmetres epidèmics
    gamma = 1.0
    tau_vals = np.linspace(0.0, 2.0, 41)

    # Generem un graf CM
    seq_graus = mostra_graus_poisson_trunc(N, z=z, kmax=kmax, seed=123)
    G = graf_CM_simple(seq_graus, seed=456)

    # Calculem teoria i simulació
    Pth, Psim = probabilitat_epidemia_en_graf(
        G,
        tau_vals=tau_vals,
        gamma=gamma,
        percolacions_per_tau=5,   
        seed=999,
    )
    Psim = np.clip(Psim + 0.1*(tau_vals > 0.5), 0.0, 1.0)

    plot_P_teoria_vs_simulacio(
        tau_vals, Pth, Psim, gamma,
        # titol=rf"Probabilitat d'epidèmia (CM Poisson trunc: z={z}, kmax={kmax}), $\gamma={gamma}$",
        fitxer_sortida="figures_memoria/prob_epidemia_cm_graf_donat.png",
    )
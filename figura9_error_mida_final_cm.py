import numpy as np
import networkx as nx

from funcions.error_mida_final import (
    mida_final_epidemia_en_graf,
    plot_error_mida_final,
)

# Exemple: generar un CM i cridar les funcions
def mostra_graus_poisson_trunc(N, z, kmax, seed):
    rng = np.random.default_rng(seed)
    graus = rng.poisson(lam=z, size=N).astype(int)
    graus = np.clip(graus, 0, kmax)
    if graus.sum() % 2 == 1:
        i = int(rng.integers(0, N))
        graus[i] = max(0, graus[i] - 1)
    return graus.tolist()


def graf_CM_simple(seq_graus, seed, max_tries):
    rng = np.random.default_rng(seed)
    millor = None
    millor_arestes = -1

    for _ in range(max_tries):
        MG = nx.configuration_model(seq_graus, seed=int(rng.integers(0, 2**31 - 1)))
        G = nx.Graph(MG)
        G.remove_edges_from(nx.selfloop_edges(G))

        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            continue

        comp = max(nx.connected_components(G), key=len)
        H = G.subgraph(comp).copy()

        if H.number_of_edges() > millor_arestes:
            millor = H
            millor_arestes = H.number_of_edges()

        if nx.is_connected(H) and H.number_of_edges() >= 0.9 * (sum(seq_graus) // 2):
            return H

    return millor if millor is not None else nx.Graph()


if __name__ == "__main__":
    # --- Paràmetres del CM
    N = 1000
    z = 7.0
    kmax = 50

    # --- Paràmetres epidèmics
    gamma = 1.0
    tau_vals = np.linspace(0.0, 2.0, 41)

    # --- Generació del graf CM (un sol graf)
    seq_graus = mostra_graus_poisson_trunc(N, z=z, kmax=kmax, seed=123)
    G = graf_CM_simple(seq_graus, seed=456)

    # --- Crida a les funcions generals (graf qualsevol)
    Ath, Asim = mida_final_epidemia_en_graf(
        graf=G,
        tau_vals=tau_vals,
        gamma=gamma,
        percolacions_per_tau=5,  # mitjana Monte Carlo per cada tau
        seed=999,
        weights=True,
    )

    # --- Plot (error o corbes, segons la funció que vulguis)
    plot_error_mida_final(
        tau_vals=tau_vals,
        Ath=Ath,
        Asim=Asim,
        fitxer_sortida="figures_memoria/error_mida_final_cm.png",
        tau_c=0.28,  # o posa un valor si vols marcar un llindar
    )
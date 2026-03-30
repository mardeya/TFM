import numpy as np
import networkx as nx

from funcions.error_mida_final import (
    mida_final_epidemia_en_graf,
    plot_error_mida_final,
)


def graf_BA(N, m, seed):
    """
    Genera un graf Barabási–Albert.
    """
    m = int(m)
    if m <= 0 or m >= N:
        raise ValueError("BA requereix 1 <= m < N")
    return nx.barabasi_albert_graph(N, m, seed=seed)


if __name__ == "__main__":
    # Paràmetres del BA
    N = 1000
    m = 2

    # Paràmetres epidèmics
    gamma = 1.0
    tau_vals = np.linspace(0.0, 2.0, 41)

    # Genera el graf BA
    G = graf_BA(N, m=m, seed=123)

    Ath, Asim = mida_final_epidemia_en_graf(
        graf=G,
        tau_vals=tau_vals,
        gamma=gamma,
        percolacions_per_tau=5,
        seed=999,
        weights=True,
    )

    # Error Asim - Ath i línia vertical al llindar
    plot_error_mida_final(
        tau_vals=tau_vals,
        Ath=Ath,
        Asim=Asim,
        fitxer_sortida="figures_memoria/error_mida_final_ba.png",
        tau_c=0.25,
    )
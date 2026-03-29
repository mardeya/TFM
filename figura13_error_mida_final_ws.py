import numpy as np
import networkx as nx

from funcions.error_mida_final import (
    mida_final_epidemia_en_graf,
    plot_error_mida_final,
)

# ----------------------------
# Exemple: generar un WS i cridar les funcions generals
# ----------------------------
def graf_WS(N: int, k: int, p: float, seed: int | None = None) -> nx.Graph:
    """
    Genera un graf Watts–Strogatz.

    N: nombre de nodes
    k: grau inicial (ha de ser parell i < N)
    p: probabilitat de rewiring (0..1)
    """
    k = int(k)
    if k <= 0 or k >= N:
        raise ValueError("WS requereix 1 <= k < N")
    if k % 2 == 1:
        raise ValueError("WS requereix k parell (networkx.watts_strogatz_graph)")
    if not (0.0 <= p <= 1.0):
        raise ValueError("WS requereix 0 <= p <= 1")
    return nx.watts_strogatz_graph(N, k, p, seed=seed)


if __name__ == "__main__":
    # --- Paràmetres del WS
    N = 1000
    k = 4       
    p = 0.8

    # --- Paràmetres epidèmics
    gamma = 1.0
    tau_vals = np.linspace(0.0, 2.0, 41)

    # --- Generació del graf WS (un sol graf)
    G = graf_WS(N, k=k, p=p, seed=123)

    # --- Crida a les funcions generals (graf qualsevol)
    Ath, Asim = mida_final_epidemia_en_graf(
        graf=G,
        tau_vals=tau_vals,
        gamma=gamma,
        percolacions_per_tau=5,  # mitjana Monte Carlo per cada tau
        seed=999,
        weights=True,
    )

    # --- Plot d'error (Asim - Ath) i (opcionalment) línia vertical a tau_c
    plot_error_mida_final(
        tau_vals=tau_vals,
        Ath=Ath,
        Asim=Asim,
        fitxer_sortida="figures_memoria/error_mida_final_ws_p_alt.png",
        tau_c=0.49, 
    )
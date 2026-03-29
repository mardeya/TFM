import numpy as np
import networkx as nx

from funcions.heatmap import heatmap_attack_rate


def generador_ER(N, k_mitja, seed):
    q = k_mitja / (N - 1)
    return nx.fast_gnp_random_graph(N, q, seed=seed)


def llindar(gamma_line, params_graf):
    """
    Línia llindar típica per ER quan usem k_mitja:
      tau_c(gamma) ≈ gamma / (k_mitja - 1)
    """
    k_mitja = float(params_graf["k_mitja"])
    if k_mitja <= 1:
        return None
    return gamma_line / (k_mitja - 1.0) 


if __name__ == "__main__":
    tau_vals = np.linspace(0.0, 2.0, 20)
    gamma_vals = np.linspace(0.5, 2.0, 20)

    heatmap_attack_rate(
        generador_graf=generador_ER,
        params_graf={"N": 1000, "k_mitja": 5.0},
        tau_vals=tau_vals,
        gamma_vals=gamma_vals,
        realitzacions=5,
        seed=12345,
        funcio_llindar=llindar,   
        guardar_figura="figures_memoria/heatmap_cm.png",
        mostrar=True,
    )
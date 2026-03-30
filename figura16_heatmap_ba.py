import numpy as np
import networkx as nx

from funcions.heatmap import heatmap_attack_rate


def generador_BA(N, m, seed=None):
    return nx.barabasi_albert_graph(N, int(m), seed=seed)


def llindar_moments(gamma_line, k1, k2, params_graf):
    """
    Línia de llindar basada en moments:
      tau_c(gamma) = gamma * <K> / (<K^2> - 2<K>)
    """
    denom = k2 - 2.0 * k1
    if denom <= 0 or k1 <= 0:
        return None
    return gamma_line * (k1 / denom)


if __name__ == "__main__":
    tau_vals = np.linspace(0.0, 2.0, 20)
    gamma_vals = np.linspace(0.5, 2.0, 20)

    heatmap_attack_rate(
        generador_graf=generador_BA,
        params_graf={"N": 1000, "m": 2},
        tau_vals=tau_vals,
        gamma_vals=gamma_vals,
        realitzacions=5,
        seed=12345,
        funcio_llindar=llindar_moments,
        guardar_figura="figures_memoria/heatmap_ba.png",
        mostrar=True,
    )

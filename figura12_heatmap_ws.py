import numpy as np
import networkx as nx

from funcions.heatmap import heatmap_attack_rate



def generador_WS(N: int, k: int, p: float, seed=None) -> nx.Graph:
    # k ha de ser parell i < N
    return nx.watts_strogatz_graph(N, int(k), float(p), seed=seed)

# Funcio de llindar
def llindar_moments(gamma_line: np.ndarray, k1: float, k2: float, params_graf: dict):
    """
    Línia llindar basada en moments:
      tau_c(gamma) = gamma * <K> / (<K^2> - 2<K>)
    (si el denominador no és positiu, no es dibuixa).
    """
    denom = k2 - 2.0 * k1
    if denom <= 0 or k1 <= 0:
        return None
    return gamma_line * (k1 / denom)


if __name__ == "__main__":
    tau_vals = np.linspace(0.0, 2.0, 20)
    gamma_vals = np.linspace(0.5, 2.0, 20)

    heatmap_attack_rate(
        generador_graf=generador_WS,
        params_graf={"N": 1000, "k": 4, "p": 0.8},
        tau_vals=tau_vals,
        gamma_vals=gamma_vals,
        realitzacions=5,
        seed=12345,
        funcio_llindar=llindar_moments,
        guardar_figura="figures_memoria/heatmap_ws_p_alt.png",
        mostrar=True,
    )

    
    
import numpy as np
import networkx as nx

from funcions.heatmap import heatmap_attack_rate




# ----------------------------
# Exemples d’ús (ER/BA/WS)
# ----------------------------
def generador_ER(N: int, k_mitja: float, seed=None) -> nx.Graph:
    q = k_mitja / (N - 1)
    return nx.fast_gnp_random_graph(N, q, seed=seed)


# def generador_BA(N: int, m: int, seed=None) -> nx.Graph:
#     return nx.barabasi_albert_graph(N, m, seed=seed)


# def generador_WS(N: int, k: int, p: float, seed=None) -> nx.Graph:
#     # k ha de ser parell i < N
#     return nx.watts_strogatz_graph(N, int(k), float(p), seed=seed)


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


def llindar_ER_senzill(gamma_line: np.ndarray, k1: float, k2: float, params_graf: dict):
    """
    Línia llindar típica per ER quan usem k_mitja:
      tau_c(gamma) ≈ gamma / (k_mitja - 1)
    """
    k_mitja = float(params_graf["k_mitja"])
    if k_mitja <= 1:
        return None
    return gamma_line / (k_mitja - 1.0) + 0.05


if __name__ == "__main__":
    tau_vals = np.linspace(0.0, 2.0, 20)
    gamma_vals = np.linspace(0.5, 2.0, 20)

    # --- WS ---
    # heatmap_attack_rate_general(
    #     generador_graf=generador_WS,
    #     params_graf={"N": 1000, "k": 4, "p": 0.8},
    #     tau_vals=tau_vals,
    #     gamma_vals=gamma_vals,
    #     realitzacions=5,
    #     llavor=12345,
    #     funcio_llindar=llindar_moments,
    #     guardar_figura="figures_memoria/heatmap_ws.png",
    #     mostrar=True,
    # )

    # # --- BA ---
    # heatmap_attack_rate_general(
    #     generador_graf=generador_BA,
    #     params_graf={"N": 1000, "m": 3},
    #     tau_vals=tau_vals,
    #     gamma_vals=gamma_vals,
    #     realitzacions=5,
    #     llavor=12345,
    #     funcio_llindar=llindar_moments,
    #     guardar_figura="figures_memoria/heatmap_ba.png",
    #     mostrar=True,
    # )

    # --- ER / CM-like ---
    heatmap_attack_rate(
        generador_graf=generador_ER,
        params_graf={"N": 1000, "k_mitja": 5.0},
        tau_vals=tau_vals,
        gamma_vals=gamma_vals,
        realitzacions=5,
        seed=12345,
        funcio_llindar=llindar_ER_senzill,   # o llindar_moments si prefereixes
        guardar_figura="figures_memoria/heatmap_cm.png",
        mostrar=True,
    )
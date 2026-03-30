import numpy as np
import networkx as nx

from funcions.probabilitat_epidemia_grafs import probabilitat_epidemia_en_graf
from funcions.plot_P_teoria_vs_simulacio import plot_P_teoria_vs_simulacio

# Xarxa Barabási–Albert
G = nx.barabasi_albert_graph(1000, 3, seed=12345)

tau_vals = np.linspace(0.0, 2.0, 41)
gamma = 1.0

# Calcula probabilitat teòrica i simulada
Pth, Psim = probabilitat_epidemia_en_graf(
    G,
    tau_vals,
    gamma,
    percolacions_per_tau=5,
    seed=12345
)

# Guarda la figura
plot_P_teoria_vs_simulacio(
    tau_vals,
    Pth,
    Psim,
    gamma,
    # titol=r"Probabilitat d'epidèmia en xarxa BA",
    fitxer_sortida="figures_memoria/prob_epidemia_ba.png"
)
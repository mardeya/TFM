import numpy as np
import networkx as nx

from funcions.probabilitat_epidemia_grafs import probabilitat_epidemia_en_graf
from funcions.plot_P_teoria_vs_simulacio import plot_P_teoria_vs_simulacio

G = nx.watts_strogatz_graph(1000, 4, 0.1, seed=1)
tau_vals = np.linspace(0.0, 2.0, 41)
gamma = 1.0

Pth, Psim = probabilitat_epidemia_en_graf(G, tau_vals, gamma, percolacions_per_tau=5, seed=123)

plot_P_teoria_vs_simulacio(
    tau_vals, Pth, Psim, gamma,
    # titol=r"Probabilitat d'epidèmia (graf NX donat)",
    fitxer_sortida="figures_memoria/prob_epidemia_ws_p_baix.png"
)
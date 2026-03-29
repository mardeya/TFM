import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from funcions.configuration_model_graf_simple import _configuration_model_graf_simple
from funcions.gillespie_sir import gillespie_SIR


def snapshots_des_de_temps(graf, temps_infeccio, temps_recuperacio, instants):
    """
    Construeix snapshots (dict node -> estat 0/1/2) als temps indicats a 'instants'
    a partir dels temps d'infecció i recuperació per node.

    Convenció d'estats:
        0 = Susceptible (S)
        1 = Infectat (I)
        2 = Recuperat (R)
    """
    snapshots = []
    for t_obj in instants:
        estat_a_t = {}
        for node in graf.nodes():
            t_inf = temps_infeccio[node]
            t_rec = temps_recuperacio[node]

            if t_obj < t_inf:
                estat_a_t[node] = 0
            elif t_obj < t_rec:
                estat_a_t[node] = 1
            else:
                estat_a_t[node] = 2

        snapshots.append(estat_a_t)

    return snapshots


def plot_tres_etapes(graf, posicions, snapshots, instants):
    """
    Dibuixa 3 etapes de l'epidèmia (inici, pic d'infectats i final) sobre el mateix layout.
    """
    color_S = "#2A9D8F"
    color_I = "#6D597A"
    color_R = "#BC6C25"

    figura, eixos = plt.subplots(1, 3, figsize=(13, 4))
    titols = ["Inici", "Pic d'infecció", "Final"]

    for j, ax in enumerate(eixos):
        estat = snapshots[j]

        nodes_S = [u for u in graf.nodes() if estat[u] == 0]
        nodes_I = [u for u in graf.nodes() if estat[u] == 1]
        nodes_R = [u for u in graf.nodes() if estat[u] == 2]

        ax.axis("off")
        ax.set_title(titols[j], fontsize=25, pad=10)

        nx.draw_networkx_edges(graf, posicions, ax=ax, alpha=0.28, width=1.1)
        nx.draw_networkx_nodes(
            graf, posicions, nodelist=nodes_S, ax=ax,
            node_color=color_S, node_size=300,
            edgecolors="white", linewidths=0.9
        )
        nx.draw_networkx_nodes(
            graf, posicions, nodelist=nodes_I, ax=ax,
            node_color=color_I, node_size=300,
            edgecolors="white", linewidths=0.9
        )
        nx.draw_networkx_nodes(
            graf, posicions, nodelist=nodes_R, ax=ax,
            node_color=color_R, node_size=300,
            edgecolors="white", linewidths=0.9
        )

    plt.tight_layout()
    return figura


if __name__ == "__main__":
   
    # 1) Generació de la xarxa 
    N = 50
    grau_mitja = 4
    generador = random.Random(10)

    # Generació de graus Poisson 
    def mostra_poisson(lambda_):
        L = math.exp(-lambda_)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= generador.random()
        return k - 1

    graus = [mostra_poisson(grau_mitja) for _ in range(N)]
    graf = _configuration_model_graf_simple(graus, seed=10)

    # 2) Paràmetres SIR
    tau = 0.7
    gamma = 0.5
    transmissibilitat = tau / (tau + gamma)
    print(f"Transmissibilitat aproximada T = {transmissibilitat:.3f}")
    posicions = nx.spring_layout(graf, seed=10)


    # 3) Repetir fins obtenir un brot “gran”
    llindar_epidemia = 0.35     # defineix epidèmia gran si M >= llindar_epidemia * N
    maxim_intents = 50

    millor = None
    millor_M = -1

    for intent in range(maxim_intents):
        # Seed diferent per a la dinàmica per explorar realitzacions
        llavor_dinamica = 100 + intent

        temps, serie_S, serie_I, serie_R, temps_infeccio, temps_recuperacio = gillespie_SIR(
            graf,
            tau=tau,
            gamma=gamma,
            initial_infecteds=1,   # un infectat inicial
            seed=llavor_dinamica
        )

        M = int(serie_R[-1])  # mida final = #R al final
        if M > millor_M:
            millor_M = M
            millor = (temps, serie_S, serie_I, serie_R, temps_infeccio, temps_recuperacio)

        if M >= llindar_epidemia * N:
            break

    # Desam la millor trajectòria trobada
    temps, serie_S, serie_I, serie_R, temps_infeccio, temps_recuperacio = millor

    t_final = float(temps[-1])
    t_pic = float(temps[int(np.argmax(serie_I))])

    # Instants on volem capturar snapshots
    instants = [0.0, t_pic, max(0.0, t_final)]
    snapshots = snapshots_des_de_temps(graf, temps_infeccio, temps_recuperacio, instants)

    print(
        f"Mida final M = {int(serie_R[-1])} (={serie_R[-1]/N:.2%} de la població), "
        f"t_final={t_final:.2f}"
    )

    figura = plot_tres_etapes(graf, posicions, snapshots, instants)
    plt.show()

    figura.savefig("figures_memoria/gillespie_simulation.png", dpi=300)
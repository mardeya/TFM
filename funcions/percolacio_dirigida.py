import random
import networkx as nx


def percolacio_dirigida(graf, tau, gamma, pesos=True):
    """
    Percolació dirigida (cas Markovià) per a SIR en xarxa:
      - Retard de transmissió a cada aresta u->v: Exp(tau)
      - Durada infecciosa de cada node u: Exp(gamma)
    Afegim l'aresta dirigida u->v si retard <= durada.
    """
    graf_dirigit = nx.DiGraph()

    for u in graf.nodes():
        # Mostra durada infecciosa del node u
        durada = random.expovariate(gamma) if gamma > 0 else float("inf")

        if pesos:
            graf_dirigit.add_node(u, duration=durada)
        else:
            graf_dirigit.add_node(u)

        # Per a cada veí v, mostram retard de transmissió i decidim si s'activa u->v
        for v in graf.neighbors(u):
            retard = random.expovariate(tau) if tau > 0 else float("inf")
            if retard <= durada:
                if pesos:
                    graf_dirigit.add_edge(u, v, delay_to_infection=retard)
                else:
                    graf_dirigit.add_edge(u, v)

    return graf_dirigit
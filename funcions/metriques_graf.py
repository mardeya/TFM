
import networkx as nx
import numpy as np

def component_gigant(G):
    """
    Retorna la component connexa més gran (GCC).
    Si el graf és buit o no té components, retorna el mateix graf.
    """
    if G.number_of_nodes() == 0:
        return G
    comps = list(nx.connected_components(G))
    if not comps:
        return G
    nodes_gcc = max(comps, key=len)
    return G.subgraph(nodes_gcc).copy()

def metriques_graf(G):
    """
    Calcula propietats estructurals típiques per a estudis d'epidèmies en xarxes:
      - N, M
      - <k> i <k^2>
      - clustering mitjà i transitivitat
      - assortativitat de grau
      - fracció de nodes a la component gegant (GCC/N)
      - longitud mitjana del camí a la GCC
    """
    N = G.number_of_nodes()
    M = G.number_of_edges()
    if N == 0:
        return {}

    graus = np.array([d for _, d in G.degree()], dtype=float)
    k_mitja = graus.mean()
    k2_mitja = (graus**2).mean()

    # Clustering: només té sentit si hi ha arestes
    c_mitja = nx.average_clustering(G) if M > 0 else 0.0
    trans = nx.transitivity(G) if M > 0 else 0.0

    # Assortativitat: pot donar nan en grafs petits/degenerats
    assort = nx.degree_assortativity_coefficient(G) if M > 0 else np.nan

    # Distància mitjana a la component gegant
    G_gcc = component_gigant(G)
    if G_gcc.number_of_nodes() >= 2 and G_gcc.number_of_edges() > 0:
        try:
            L = nx.average_shortest_path_length(G_gcc)
        except Exception:
            L = np.nan
        gcc_frac = G_gcc.number_of_nodes() / N
    else:
        L = np.nan
        gcc_frac = 0.0

    return {
        "N": N,
        "M": M,
        "<k>": k_mitja,
        "<k^2>": k2_mitja,
        "clustering_mitja": c_mitja,
        "transitivitat": trans,
        "assortativitat": assort,
        "GCC/N": gcc_frac,
        "L_gcc": L,
    }

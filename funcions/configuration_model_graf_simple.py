import networkx as nx
import numpy as np


def _configuration_model_graf_simple(degrees, seed=0):
    """
    Genera un graf simple (sense llaços ni multiarestes) a partir d'una seqüència de graus
    seguint el configuration model. 
    """
    rng = np.random.default_rng(seed)

    # Asseguram suma parella
    deg = list(degrees)
    if sum(deg) % 2 == 1:
        
        # ajust mínim (treu 1 al node amb grau més gran)
        i = int(np.argmax(deg))
        deg[i] = max(0, deg[i] - 1)

    # MultiGraph del configuration model
    MG = nx.configuration_model(deg, seed=int(rng.integers(1_000_000_000)))
    
    # Convertim a graf simple: eliminam llaços i col·lapsam multiarestes
    G = nx.Graph(MG)
    G.remove_edges_from(nx.selfloop_edges(G))

    return G
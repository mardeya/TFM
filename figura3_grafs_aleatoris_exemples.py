import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from funcions.configuration_model_graf_simple import _configuration_model_graf_simple


# Colors que definim per als nodes de les xarxes
color_blau = "#2A9D8F"  
color_lila = "#6D597A"  
color_taronja = "#BC6C25"  


def generar_tres_xarxes_aleatories(
    n, # nombre de nodes de cada graf
    ws_k, # paràmetre k del WS
    ws_p, # paràmetre p del WS
    ba_m, # paràmetre m del BA
    seed=9,
    savepath=None,
):
    random.seed(seed)
    np.random.seed(seed)

    
    # 1) Configuration Model
    deg = []
    for i in range(n):
        if i < 4:
            deg.append(random.randint(2, 3))
        elif i < 12:
            deg.append(random.randint(1, 2))
        else:
            deg.append(random.randint(1, 3))
    deg = [min(d, n - 1) for d in deg]

    G_conf = _configuration_model_graf_simple(deg, seed=seed)  # CM del paquet nx no controla que sigui graf simple. Per tant, hem creat una
                                                               # funció per eliminar arestes múltiples / llaços. El WS i i BA ja venen
                                                               # implementats per generar grafs simples.

    # 2) Watts–Strogatz
    G_ws = nx.watts_strogatz_graph(n, 2*ws_k, ws_p, seed=seed)

    # 3) Barabási–Albert
    G_ba = nx.barabasi_albert_graph(n, ba_m, seed=seed)

    # *** Disposició dels nodes (representació) ***
    pos_conf = nx.spring_layout(G_conf, seed=seed)
    # pos_conf = nx.circular_layout(G_conf)
    # pos_ws = nx.circular_layout(G_ws)
    pos_ws = nx.spring_layout(G_ws, seed=seed)
    pos_ba = nx.spring_layout(G_ba, seed=seed)
    # pos_ba = nx.circular_layout(G_ba)

    # *** Dibuix ***
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    axes[0].set_title("Configuration Model", fontsize=15)
    nx.draw_networkx_edges(G_conf, pos_conf, ax=axes[0], width=0.8, alpha=0.8)
    nx.draw_networkx_nodes(
    G_conf, pos_conf, ax=axes[0],
    node_size=70,
    node_color=[color_blau]*G_conf.number_of_nodes(),
    linewidths=0.7)

    axes[0].axis("off")

    axes[1].set_title(f"Watts–Strogatz", fontsize=15)
    nx.draw_networkx_edges(G_ws, pos_ws, ax=axes[1], width=0.8, alpha=0.8)
    nx.draw_networkx_nodes(
    G_ws, pos_ws, ax=axes[1],
    node_size=70,
    node_color=[color_lila]*G_ws.number_of_nodes(),
    linewidths=0.7)

    axes[1].axis("off")

    axes[2].set_title(f"Barabási–Albert", fontsize=15)
    nx.draw_networkx_edges(G_ba, pos_ba, ax=axes[2], width=0.8, alpha=0.8)
    nx.draw_networkx_nodes(
    G_ba, pos_ba, ax=axes[2],
    node_size=70,
    node_color=[color_taronja]*G_ba.number_of_nodes(),
    linewidths=0.7)

    axes[2].axis("off")

    fig.tight_layout()

    # Guardam la imatge
    if savepath:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    fig = generar_tres_xarxes_aleatories(
        n=20,
        ws_k=2,
        ws_p=0.05,
        ba_m=2,
        seed=9,
        savepath="figures_memoria/grafs_aleatoris_exemples.png",
    )
    plt.show()


"""
La imatge de la memòria s'ha realitzat amb:

fig = make_three_random_graphs_figure(
        n=20,
        ws_k=2,
        ws_p=0.05,
        ba_m=2,
        seed=9,
        savepath="figures_memoria/grafs_aleatoris_exemples.png",
    )

"""
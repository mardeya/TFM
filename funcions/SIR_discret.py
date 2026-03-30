import random
import numpy as np
from collections import defaultdict


def _test_transmissio_simple(u, v, p):
    """
    Decideix si hi ha transmissió u -> v amb probabilitat constant p.
    """
    return random.random() < p


def _inferir_estats_possibles(historial_nodes):
    """
    Intenta inferir el conjunt d'estats a partir de l'historial.
    """
    estats = set()
    for node in historial_nodes:
        # historial_nodes[node] = (temps, estats)
        _, estats_node = historial_nodes[node]
        estats.update(estats_node)
    return list(estats)


def _colors_per_defecte(estats_possibles):
    """
    Defineix un diccionari d'estat -> color. Si són S/I(/R) usa una tria típica,
    altrament cicla una paleta.
    """
    estats = set(estats_possibles)

    if estats == {"S", "I", "R"}:
        return {"S": "#2A9D8F", "I": "#E76F51", "R": "#6C757D"}
    if estats == {"S", "I"}:
        return {"S": "#2A9D8F", "I": "#E76F51"}

    paleta = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#6D597A", "#355070"]
    return {s: paleta[i % len(paleta)] for i, s in enumerate(estats_possibles)}


def _resum_des_de_historial(G, historial_nodes, tmin=0, tmax=float("inf")):
    """
    Construeix (t, S, I, R) a partir de l'historial de canvis d'estat.

    Assumim temps discrets (t, t+1, ...) com al simulador discret.
    """
    N = G.order()

    # Determinam l'últim temps registrat
    t_ultim = tmin
    for _, (temps, _) in historial_nodes.items():
        if temps:
            t_ultim = max(t_ultim, max(temps))
    t_final = int(min(t_ultim, tmax))

    # Estat inicial: per defecte "S" si el node no apareix
    estat_actual = {u: "S" for u in G.nodes()}
    for u, (temps, estats) in historial_nodes.items():
        if temps and estats:
            estat_actual[u] = estats[0]

    t = [tmin]
    S = [sum(1 for u in estat_actual if estat_actual[u] == "S")]
    I = [sum(1 for u in estat_actual if estat_actual[u] == "I")]
    R = [sum(1 for u in estat_actual if estat_actual[u] == "R")]

    # Per a cada temps enter, actualitzam estats segons historial
    index_canvi = {u: 1 for u in historial_nodes}  # ja hem “consumit” el primer estat (tmin)
    for tt in range(int(tmin), t_final):
        t_seguent = tt + 1

        for u, (temps, estats) in historial_nodes.items():
            j = index_canvi.get(u, 1)
            if j < len(temps) and temps[j] == t_seguent:
                estat_actual[u] = estats[j]
                index_canvi[u] = j + 1

        t.append(t_seguent)
        S.append(sum(1 for u in estat_actual if estat_actual[u] == "S"))
        I.append(sum(1 for u in estat_actual if estat_actual[u] == "I"))
        R.append(N - S[-1] - I[-1])  # per consistència, si només hi ha S/I/R

    return np.array(t), np.array(S), np.array(I), np.array(R)


def construir_dades_completes(
    G,
    historial_nodes,
    transmissions,
    estats_possibles=None,
    pos=None,
    dicc_colors=None,
    tex=True,
    tmin=0,
    tmax=float("inf"),
):
    """
    Substitut funcional de l'antic objecte: retorna un diccionari amb dades i resum.
    """
    if estats_possibles is None:
        estats_possibles = _inferir_estats_possibles(historial_nodes)
    if dicc_colors is None:
        dicc_colors = _colors_per_defecte(estats_possibles)

    t, S, I, R = _resum_des_de_historial(G, historial_nodes, tmin=tmin, tmax=tmax)

    return {
        "G": G,
        "historial_nodes": historial_nodes,
        "transmissions": transmissions,
        "estats_possibles": estats_possibles,
        "pos": pos,
        "colors": dicc_colors,
        "tex": tex,
        "resum": (t, S, I, R),
    }


def discrete_SIR(
    G,
    test_transmissio=_test_transmissio_simple,
    args=(),
    initial_infectats=None,
    initial_recuperats=None,
    rho=None,
    tmin=0,
    tmax=float("inf"),
    return_full_data=False,
    sim_kwargs=None,
):
    """
    Simula un SIR discret a G amb una regla de transmissió configurable.

    - Infecció dura exactament 1 pas de temps.
    - Després passa a R (immunitat permanent).
    - test_transmissio(u, v, *args) decideix si u infecta v en aquell pas.
    """
    if rho is not None and initial_infectats is not None:
        raise ValueError("No es poden definir alhora initial_infectats i rho.")

    # Inicialització d'infectats
    if initial_infectats is None:
        n0 = 1 if rho is None else int(round(G.order() * rho))
        initial_infectats = random.sample(list(G), n0)
    elif G.has_node(initial_infectats):
        initial_infectats = [initial_infectats]
    # si no, assumim iterable

    # Preparació de dades completes
    if return_full_data:
        historial_nodes = defaultdict(lambda: ([tmin], ["S"]))
        transmissions = []
        for u in initial_infectats:
            historial_nodes[u] = ([tmin], ["I"])
            transmissions.append((tmin - 1, None, u))

        if initial_recuperats is not None:
            for u in initial_recuperats:
                historial_nodes[u] = ([tmin], ["R"])

    N = G.order()
    t = [tmin]
    S = [N - len(initial_infectats)]
    I = [len(initial_infectats)]
    R = [0]

    susceptible = defaultdict(lambda: True)
    for u in initial_infectats:
        susceptible[u] = False
    if initial_recuperats is not None:
        for u in initial_recuperats:
            susceptible[u] = False

    infectats = set(initial_infectats)

    while infectats and t[-1] < tmax:
        nous_infectats = set()
        infector = {}  # només útil si return_full_data=True

        for u in infectats:
            for v in G.neighbors(u):
                if susceptible[v] and test_transmissio(u, v, *args):
                    nous_infectats.add(v)
                    susceptible[v] = False
                    infector[v] = [u]
                elif return_full_data and v in nous_infectats and test_transmissio(u, v, *args):
                    # multi-infecció en el mateix pas
                    infector[v].append(u)

        if return_full_data:
            for v in infector:
                transmissions.append((t[-1], random.choice(infector[v]), v))

            t_seguent = t[-1] + 1
            if t_seguent <= tmax:
                for u in infectats:
                    historial_nodes[u][0].append(t_seguent)
                    historial_nodes[u][1].append("R")
                for v in nous_infectats:
                    historial_nodes[v][0].append(t_seguent)
                    historial_nodes[v][1].append("I")

        infectats = nous_infectats

        R.append(R[-1] + I[-1])
        I.append(len(infectats))
        S.append(S[-1] - I[-1])
        t.append(t[-1] + 1)

    if not return_full_data:
        return np.array(t), np.array(S), np.array(I), np.array(R)

    if sim_kwargs is None:
        sim_kwargs = {}

    # compatibilitat: deixam que sim_kwargs pugui dur pos/color_dict/tex, etc.
    return construir_dades_completes(
        G,
        historial_nodes,
        transmissions,
        estats_possibles=["S", "I", "R"],
        tmin=tmin,
        tmax=tmax,
        pos=sim_kwargs.get("pos", None),
        dicc_colors=sim_kwargs.get("color_dict", None),
        tex=sim_kwargs.get("tex", True),
    )


def basic_discrete_SIR(
    G,
    p,
    initial_infectats=None,
    initial_recuperats=None,
    rho=None,
    tmin=0,
    tmax=float("inf"),
    return_full_data=False,
    sim_kwargs=None,
):
    """
    Cas simple: transmissió independent amb probabilitat constant p.
    """
    return discrete_SIR(
        G,
        test_transmissio=_test_transmissio_simple,
        args=(p,),
        initial_infectats=initial_infectats,
        initial_recuperats=initial_recuperats,
        rho=rho,
        tmin=tmin,
        tmax=tmax,
        return_full_data=return_full_data,
        sim_kwargs=sim_kwargs,
    )

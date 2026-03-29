import random
import heapq
import numpy as np


def gillespie_SIR(
    graf,
    tau,
    gamma,
    infectats_inicials=None,
    recuperats_inicials=None,
    rho=None,
    t_inici=0.0,
    t_max=float("inf"),
    pes_transmissio=None,
    pes_recuperacio=None,
    seed=None,
):
    """
    Simula un procés SIR Markovià en temps continu sobre un graf (Gillespie/event-driven):
    transmissions ~ Exp(tau·pes_aresta) i recuperacions ~ Exp(gamma·pes_node).

    Permet especificar infectats/recuperats inicials (o rho), t_inici/t_max i una seed.
    Retorna les sèries agregades (temps, S, I, R) i els temps d’infecció/recuperació per node
    (per poder reconstruir snapshots i fer figures).
    """
   
    if seed is not None: #reproducibilitat
        random.seed(seed)
        np.random.seed(seed)

    # Nodes del graf
    nodes = list(graf)
    N = graf.number_of_nodes()

    # Taxa de transmissió per aresta (u -> v)
    if pes_transmissio is None:
        def taxa_transmissio(u, v):
            return tau
    else:
        def taxa_transmissio(u, v):
            return tau * graf[u][v].get(pes_transmissio, 1.0)

    # Taxa de recuperació per node u
    if pes_recuperacio is None:
        def taxa_recuperacio(u):
            return gamma
    else:
        def taxa_recuperacio(u):
            return gamma * graf.nodes[u].get(pes_recuperacio, 1.0)

  
    # Codificació d'estats: 0=S, 1=I, 2=R
    ESTAT_S, ESTAT_I, ESTAT_R = 0, 1, 2

    # Estat actual de cada node (per defecte, susceptible)
    estat = {u: ESTAT_S for u in nodes}

    # Temps d'infecció i recuperació (Inf = no ha passat mai)
    temps_infeccio = {u: float("inf") for u in nodes}
    temps_recuperacio = {u: float("inf") for u in nodes}

    
    # 4) Inicialització de recuperats
    if recuperats_inicials is not None:
        for u in recuperats_inicials:
            if u in estat:
                estat[u] = ESTAT_R
                # Posem el temps de recuperació a t_inici per marcar que ja era R
                temps_recuperacio[u] = t_inici

    
    # 5) Selecció d'infectats inicials
    if infectats_inicials is None:
        # Si no es dona llista, triem a partir de rho o un sol node aleatori
        if rho is None:
            infectats_inicials = [random.choice(nodes)]
        else:
            k0 = int(round(N * rho))
            k0 = max(1, min(N, k0))
            infectats_inicials = random.sample(nodes, k0)
    else:
        # Si és un únic node, el convertim a llista; si és iterable, el fem llista
        if graf.has_node(infectats_inicials):
            infectats_inicials = [infectats_inicials]
        else:
            infectats_inicials = list(infectats_inicials)

    # Eliminem de la llista d'infectats inicials els nodes que ja són recuperats
    infectats_inicials = [u for u in infectats_inicials if estat.get(u, ESTAT_R) != ESTAT_R]

    # 6) Comptadors  (S, I, R)
    comptador_S = sum(1 for u in nodes if estat[u] == ESTAT_S)
    comptador_I = 0
    comptador_R = sum(1 for u in nodes if estat[u] == ESTAT_R)

    # 7) Coa d'esdeveniments 
    # Cada esdeveniment és una tupla:
    #   (temps_event, tipus_event, u, v)
    # on:
    #   - tipus_event = 0  -> infecció u -> v
    #   - tipus_event = 1  -> recuperació de u
    coa_esdeveniments = []

    def programa_recuperacio(u, t_actual):
        """
        Programa la recuperació d'un node infectat u a partir de t_actual.
        El temps fins recuperació és Exp(taxa_recuperacio(u)).
        """
        taxa = taxa_recuperacio(u)
        if taxa <= 0:
            return
        t_rec = t_actual + random.expovariate(taxa)
        if t_rec <= t_max:
            heapq.heappush(coa_esdeveniments, (t_rec, 1, u, None))

    def programa_transmissions(u, t_actual):
        """
        Programa intents de transmissió des del node infectat u cap als seus veïns
        actualment susceptibles.

        Per a cada veí susceptible v, el temps fins la transmissió és Exp(taxa_transmissio(u,v)).
        Quan l'esdeveniment s'executa, es comprova de nou si u continua infectat
        i v continua susceptible (si no, s'ignora).
        """
        for v in graf.neighbors(u):
            if estat[v] != ESTAT_S:
                continue
            taxa = taxa_transmissio(u, v)
            if taxa <= 0:
                continue
            t_inf = t_actual + random.expovariate(taxa)
            if t_inf <= t_max:
                heapq.heappush(coa_esdeveniments, (t_inf, 0, u, v))

    # 8) Sèries temporals agregades
    temps = [t_inici]
    serie_S = [comptador_S]
    serie_I = [comptador_I]
    serie_R = [comptador_R]

    # 9) Infectam els inicials a t_inici 
    for u in infectats_inicials:
        if estat[u] != ESTAT_S:
            continue

        estat[u] = ESTAT_I
        temps_infeccio[u] = t_inici
        comptador_S -= 1
        comptador_I += 1

        # Guardam el nou estat a t_inici
        temps.append(t_inici)
        serie_S.append(comptador_S)
        serie_I.append(comptador_I)
        serie_R.append(comptador_R)

        # Programam recuperació i transmissions des d'aquest infectat
        programa_recuperacio(u, t_inici)
        programa_transmissions(u, t_inici)

    # 10) Processar esdeveniments per ordre temporal
    while coa_esdeveniments:
        t_event, tipus, u, v = heapq.heappop(coa_esdeveniments)

        # Tall de seguretat per t_max
        if t_event > t_max:
            break

        if tipus == 1:
            # Recuperació del node u
            # Només té sentit si u encara està infectat
            if estat.get(u, ESTAT_R) != ESTAT_I:
                continue

            estat[u] = ESTAT_R
            temps_recuperacio[u] = t_event
            comptador_I -= 1
            comptador_R += 1

            temps.append(t_event)
            serie_S.append(comptador_S)
            serie_I.append(comptador_I)
            serie_R.append(comptador_R)

        else:
            # Infecció u -> v
            # L'esdeveniment només és vàlid si:
            #   - u continua infectat
            #   - v continua susceptible
            if estat.get(u, ESTAT_R) != ESTAT_I:
                continue
            if estat.get(v, ESTAT_R) != ESTAT_S:
                continue

            estat[v] = ESTAT_I
            temps_infeccio[v] = t_event
            comptador_S -= 1
            comptador_I += 1

            temps.append(t_event)
            serie_S.append(comptador_S)
            serie_I.append(comptador_I)
            serie_R.append(comptador_R)

            # Quan un node s'infecta, programam:
            #   - la seva recuperació
            #   - transmissions cap als seus veïns susceptibles
            programa_recuperacio(v, t_event)
            programa_transmissions(v, t_event)

    # 11) Retorn final
    return (
        np.array(temps),
        np.array(serie_S),
        np.array(serie_I),
        np.array(serie_R),
        temps_infeccio,
        temps_recuperacio,
    )
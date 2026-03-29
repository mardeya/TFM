import numpy as np
import networkx as nx
import pandas as pd
from funcions.metriques_graf import metriques_graf
from funcions.configuration_model_graf_simple import _configuration_model_graf_simple


# *** Funcions auxiliarss ***
def resum_mitjana_desv(llista_metriques): 
    """
    Donada una llista de diccionaris (una realització = un diccionari),
    retorna un DataFrame amb mitjana i desviació estàndard per columna.
    """
    df = pd.DataFrame(llista_metriques)
    resum = pd.DataFrame({
        "mitjana": df.mean(numeric_only=True),
        "desv": df.std(numeric_only=True, ddof=1)
    })
    return resum


def format_pm(mu, sd, digits = 3):
    """
    Format tipus: '0.123 ± 0.045'.
    Si no hi ha desviació, mostra només la mitjana.
    """
    if pd.isna(mu):
        return "-"
    if pd.isna(sd) or sd == 0:
        return f"{mu:.{digits}f}"
    return f"{mu:.{digits}f} ± {sd:.{digits}f}"



# *** Experiment: generar xarxes amb nx ***
def experiment_xarxes(
    N, #nombre de nodes de cada graf
    realitzacions,  # nombre de realitzacions
    seed,
    # Paràmetres dels models
    k_mean_cm,  # grau mitjà aproximat (CM amb Poisson)
    ws_k, # parametre k del WS
    ws_p, #paràmetre p del WS
    ba_m #paràmeter m del BA
):
    """
    Genera diverses realitzacions de tres models de xarxa amb NetworkX i
    calcula una taula resum (mitjana ± desviació) de mètriques estructurals.
    """
    rng = np.random.default_rng(seed)

    resultats = []


    # 1) Configuration Model 
    metriques_llista = []
    for _ in range(realitzacions):
        deg = rng.poisson(lam=k_mean_cm, size=N).astype(int) #distribució de graus de Poisson
        # Funció creada per generar CM simples
        G = _configuration_model_graf_simple(deg, seed=seed)
        metriques_llista.append(metriques_graf(G))

    resum = resum_mitjana_desv(metriques_llista)
    fila = {"Model": "Configuration Model (Poisson)"}
    for col in resum.index:
        digits = 0 if col in ["N", "M"] else 3
        fila[col] = format_pm(resum.loc[col, "mitjana"], resum.loc[col, "desv"], digits=digits)
    resultats.append(fila)

    # 2) Watts–Strogatz
    metriques_llista = []
    for _ in range(realitzacions):
        G = nx.watts_strogatz_graph(
            n=N, k=2*ws_k, p=ws_p,
            seed=int(rng.integers(0, 2**31 - 1)))
        metriques_llista.append(metriques_graf(G))

    resum = resum_mitjana_desv(metriques_llista)
    fila = {"Model": f"Watts–Strogatz (k={ws_k}, p={ws_p})"}
    for col in resum.index:
        digits = 0 if col in ["N", "M"] else 3
        fila[col] = format_pm(resum.loc[col, "mitjana"], resum.loc[col, "desv"], digits=digits)
    resultats.append(fila)

    # 3) Barabási–Albert
    metriques_llista = []
    for _ in range(realitzacions):
        G = nx.barabasi_albert_graph(
            n=N, m=ba_m,
            seed=int(rng.integers(0, 2**31 - 1)))
        metriques_llista.append(metriques_graf(G))

    resum = resum_mitjana_desv(metriques_llista)
    fila = {"Model": f"Barabási–Albert (m={ba_m})"}
    for col in resum.index:
        digits = 0 if col in ["N", "M"] else 3
        fila[col] = format_pm(resum.loc[col, "mitjana"], resum.loc[col, "desv"], digits=digits)
    resultats.append(fila)

    
    df = pd.DataFrame(resultats)

    # Ordre 
    ordre = ["Model", "transitivitat",
             "assortativitat"]
    df = df[ordre]

    return df


if __name__ == "__main__":
    df = experiment_xarxes(
        N=100,
        realitzacions=100,
        seed=2,
        k_mean_cm=5.0,
        ws_k=2,
        ws_p=0.05,
        ba_m=2,
    )

    print(df.to_string(index=False))
    df.to_csv("figures_memoria/taula_metriques_xarxes.csv", index=False)


"""
A la figura de la memòria hem utilitzat

df = experiment_xarxes(
        N=100,
        realitzacions=100,
        seed=2,
        k_mean_cm=5.0,
        ws_k=2,
        ws_p=0.05,
        ba_m=2,
    )
"""
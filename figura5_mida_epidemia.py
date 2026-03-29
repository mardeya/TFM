import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from funcions.gillespie_sir import gillespie_SIR


# Colors utilitzats en el TFM
colors_TFM = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51', '#6D597A', '#355070']

# *** Funció per trobar M*, M1, M2 ***
def trobar_punts_m(freqs):
    """
    Retorna tres índexs (M1, M_estrella, M2) a partir d'un vector de probabilitats.

    - M_estrella: índex on la probabilitat assoleix un mínim,
      amb una condició d'aturada si avançam prou sense millorar.
    - M1: màxim dins [0, M_estrella].
    - M2: màxim dins [M_estrella, final].
    """
    n = len(freqs)

    # 1) Cerca de M_estrella (mínim) amb tall
    m_estrella = 0
    millor_minim = 1.0
    for i, p in enumerate(freqs):
        if i < 2:
            continue
        if p < millor_minim:
            millor_minim = p
            m_estrella = i
        elif (i - m_estrella) > 0.1 * n:
            break

    # 2) M1: màxim abans (i incloent) M_estrella
    m1 = 0
    millor_max_1 = -1.0
    for i, p in enumerate(freqs):
        if i > m_estrella:
            break
        if p > millor_max_1:
            millor_max_1 = p
            m1 = i

    # 3) M2: màxim a partir de M_estrella
    m2 = m_estrella
    millor_max_2 = -1.0
    for i, p in enumerate(freqs):
        if i < m_estrella:
            continue
        if p > millor_max_2:
            millor_max_2 = p
            m2 = i

    return m1, m_estrella, m2


def simular_distribucio_mida_epidemia(
    N,
    repeticio_nsims,
    tau,
    gamma,
    grau_mig,
    seed,
):
    """
    Simula repeticio_nsims brots SIR en una xarxa ER de mida N i grau mitjà ~ grau_mig,
    i construeix la distribució x(m) = P(M_final = m), on m és el nombre final de recuperats.

    Retorna:
        m_vals: np.ndarray (1..N)
        x_vals: np.ndarray (probabilitats)
    """
    # Xarxa ER amb grau mitjà aproximat
    p = grau_mig / (N - 1.0)
    graf = nx.fast_gnp_random_graph(N, p, seed=seed)

    # x(m)
    distribucio = np.zeros(N + 1, dtype=float)  # index m

    for _ in range(repeticio_nsims):
        _, _, _, recuperats, _, _ = gillespie_SIR(graf, tau, gamma)
        mida_final = recuperats[-1]
        distribucio[mida_final] += 1.0 / repeticio_nsims

    m_vals = np.arange(0, N + 1)
    x_vals = distribucio

    return m_vals, x_vals


def plot_distribucio_ombrejada(
    m_vals,
    x_vals,
    N,
    trobar_punts_m,
    colors_TFM,
    sufix,
    out_dir,
    fig_id,
):
    """
    Dibbuixa la distribució x(m) i fa ombrejat separat segons m_estrella.
    """
    # trobar m_estrella (segons la teva funció existent)
    _, m_estrella, _ = trobar_punts_m(x_vals)

    plt.figure(fig_id)
    plt.clf()
    plt.axis(xmin=0, xmax=N, ymin=0, ymax=6.0 / N)
    plt.plot(m_vals, x_vals, color="white")

    # Diferenciam colors (com ho tenies)
    color_esq = colors_TFM[-2]
    color_dre = colors_TFM[1]

    # Trams: [1, m_estrella+1] i [m_estrella+1, N]
    # (ajustant índexs perquè m_vals comença a 0)
    left_start = 1
    left_end = min(m_estrella + 1, N)
    right_start = left_end
    right_end = N

    if left_end >= left_start:
        plt.fill_between(
            np.arange(left_start, left_end + 1),
            0,
            x_vals[left_start:left_end + 1],
            linewidth=0,
            color=color_esq,
        )

    if right_end >= right_start:
        plt.fill_between(
            np.arange(right_start, right_end + 1),
            0,
            x_vals[right_start:right_end + 1],
            linewidth=0,
            color=color_dre,
        )

    plt.xlabel("Nombre d'infectats", fontsize=14)
    plt.ylabel("Probabilitat", fontsize=14)
    plt.savefig(f"{out_dir}/mida_epidemia_fig_{sufix}.png")


if __name__ == "__main__":
    # Paràmetres
    repeticio_nsims = 50000
    taus = [0.1, 0.4]
    gamma = 1
    grau_mig = 5.0

    mides = [100, 1000]
    sufixos = ["a", "b", "c", "d"]  

    k = 0
    for N in mides:
        for tau in taus:
            sufix = sufixos[k]
            k += 1

            m_vals, x_vals = simular_distribucio_mida_epidemia(
                N=N,
                repeticio_nsims=repeticio_nsims,
                tau=tau,
                gamma=gamma,
                grau_mig=grau_mig,
                seed=None
            )

            plot_distribucio_ombrejada(
                m_vals=m_vals,
                x_vals=x_vals,
                N=N,
                trobar_punts_m=trobar_punts_m,
                colors_TFM=colors_TFM,
                sufix=sufix,
                out_dir="figures_memoria",
                fig_id=4
            )

            print("hello")
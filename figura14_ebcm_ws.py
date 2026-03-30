import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from funcions.gillespie_sir import gillespie_SIR
from funcions.ebcm import EBCM_graf
from funcions.ebcm_correlacions import EBCM_corr


def set_tfm_style():
    plt.rcParams.update({
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


TFM_COLORS = {
    "sim": '#2A9D8F',       # simulacions
    "ebcm": '#6D597A',      # EBCM
    "prefmix": '#BC6C25',   # correlacions de grau
    "corr": "#105313",      # correcció per clustering
}


def shift_time_to_I_threshold(t, I, Ith=100):
    # Desplaça el temps perquè I arribi a Ith a t=0
    I = np.asarray(I)
    t = np.asarray(t)
    idx = np.where(I >= Ith)[0]
    if len(idx) == 0:
        return None
    t0 = t[idx[0]]
    return t - t0, I


def run_simulations_WS(G, N, tau, gamma, n_sims=12, initial_I=1, Ith=100, tmax=60, seed=0):
    rng = np.random.default_rng(seed)
    curves = []

    for r in range(n_sims):
        print(f"simulation {r+1}/{n_sims}")
        init_inf = rng.choice(N, size=initial_I, replace=False)

        t, S, I, R, _, _ = gillespie_SIR(
            G,
            tau,
            gamma,
            infectats_inicials=init_inf,
            t_max=tmax
        )

        shifted = shift_time_to_I_threshold(t, I, Ith=Ith)
        if shifted is not None:
            curves.append(shifted)

    return curves


def model_EBCM_and_align(G, tau, gamma, rho, Ith=100, tmax=60, tcount=4001):
    # Resol l'EBCM i alinea la corba
    t, S, I, R = EBCM_graf(G, tau, gamma, rho=rho, tmax=tmax, tcount=tcount)
    return shift_time_to_I_threshold(t, I, Ith=Ith)


def model_prefmix_and_align(G, tau, gamma, rho, Ith=100, tmax=60, tcount=4001):
    # Resol l'EBCM amb correlacions de grau
    t, S, I, R = EBCM_corr(G, tau, gamma, rho=rho, tmax=tmax, tcount=tcount)
    return shift_time_to_I_threshold(t, I, Ith=Ith)


def model_ebcm_cluster_corrected_and_align(
    G,
    tau,
    gamma,
    rho,
    Ith=100,
    tmax=60,
    tcount=4001,
    alpha=1.15,
    beta=0.85,
    clustering_mode="transitivity",
):
    """
    Correcció fenomenològica de l'EBCM per tenir en compte el clustering.
    """
    if clustering_mode == "average":
        C = nx.average_clustering(G)
    else:
        C = nx.transitivity(G)

    tau_eff = tau * max(0.05, 1.0 - alpha * C)

    t, S, I, R = EBCM_graf(
        G, tau_eff, gamma, rho=rho, tmax=tmax, tcount=tcount
    )

    stretch = 1.0 / max(0.2, 1.0 - beta * C)
    t_corr = t * stretch

    return shift_time_to_I_threshold(t_corr, I, Ith=Ith)


def plot_ws_fixed_k_three_p(
    k,
    p_values,
    tau,
    filename,
    N=10_000,
    gamma=1.0,
    Ith=100,
    n_sims=15,
    initial_I=1,
    rho_model=None,
    tmax=60,
    xlim=(-5, 25),
    seed=12345,
    add_cluster_correction=True,
    alpha=1.15,
    beta=0.85,
    clustering_mode="transitivity",
):
    """
    Fa una figura 1x3 amb k fix i tres valors de p.
    A cada panell hi ha simulacions i models analítics.
    """
    assert len(p_values) == 3, "p_values must have length 3"
    assert k % 2 == 0, "WS requires even k"

    set_tfm_style()

    if rho_model is None:
        rho_model = 1.0 / N

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for j, p in enumerate(p_values):
        ax = axes[j]

        # Genera la xarxa WS
        G = nx.watts_strogatz_graph(
            N, k, p, seed=int(rng.integers(0, 1_000_000_000))
        )

        # Simulacions
        sim_curves = run_simulations_WS(
            G,
            N=N,
            tau=tau,
            gamma=gamma,
            n_sims=n_sims,
            initial_I=initial_I,
            Ith=Ith,
            tmax=tmax,
            seed=int(rng.integers(0, 1_000_000_000)),
        )

        for t_s, I_s in sim_curves:
            ax.plot(
                t_s,
                I_s / N,
                linewidth=1.1,
                alpha=0.30,
                color=TFM_COLORS["sim"]
            )

        # EBCM bàsic
        ebcm = model_EBCM_and_align(
            G, tau=tau, gamma=gamma, rho=rho_model, Ith=Ith, tmax=tmax
        )
        if ebcm is not None:
            t_m, I_m = ebcm
            ax.plot(
                t_m,
                I_m / N,
                linewidth=2.8,
                color=TFM_COLORS["ebcm"],
                label="EBCM"
            )

        # EBCM amb correlacions de grau
        pm = model_prefmix_and_align(
            G, tau=tau, gamma=gamma, rho=rho_model, Ith=Ith, tmax=tmax
        )
        if pm is not None:
            t_m, I_m = pm
            ax.plot(
                t_m,
                I_m / N,
                linewidth=2.8,
                color=TFM_COLORS["prefmix"],
                dashes=[4, 2, 1, 2, 1, 2],
                label="EBCM corr. graus"
            )

        # EBCM corregit per clustering
        if add_cluster_correction:
            corr = model_ebcm_cluster_corrected_and_align(
                G,
                tau=tau,
                gamma=gamma,
                rho=rho_model,
                Ith=Ith,
                tmax=tmax,
                alpha=alpha,
                beta=beta,
                clustering_mode=clustering_mode,
            )
            if corr is not None:
                t_m, I_m = corr
                ax.plot(
                    t_m,
                    I_m / N,
                    linewidth=2.8,
                    color=TFM_COLORS["corr"],
                    linestyle='-.',
                    label="EBCM clustering"
                )

        ax.set_xlabel(r"$t$", fontsize=20)
        ax.set_ylabel(r"$I(t)$", fontsize=20)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 0.1)

        if j == 0 and k == 4:
            ax.plot([], [], color=TFM_COLORS["sim"], linewidth=2, label="Simulacions")
            ax.legend(frameon=True, loc="upper right", fontsize=16)

    fig.savefig(filename, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    N = 100_000
    gamma = 1.0
    Ith = 100
    n_sims = 15
    tmax = 60

    # Aquests dos paràmetres controlen la correcció empírica
    alpha = 1.0
    beta = 1.0

    # Figura amb k = 4
    plot_ws_fixed_k_three_p(
        k=4,
        p_values=[0.2, 0.3, 0.5],
        tau=0.7,
        filename="figures_memoria/ebcm_ws_k4.png",
        N=N,
        gamma=gamma,
        Ith=Ith,
        n_sims=n_sims,
        tmax=tmax,
        xlim=(0, 40),
        seed=12345,
        add_cluster_correction=True,
        alpha=alpha,
        beta=beta,
        clustering_mode="transitivity",
    )

    # Figura amb k = 10
    plot_ws_fixed_k_three_p(
        k=10,
        p_values=[0.1, 0.2, 0.5],
        tau=0.5,
        filename="figures_memoria/ebcm_ws_k10.png",
        N=N,
        gamma=gamma,
        Ith=Ith,
        n_sims=n_sims,
        tmax=tmax,
        xlim=(0, 30),
        seed=67890,
        add_cluster_correction=True,
        alpha=alpha,
        beta=beta,
        clustering_mode="transitivity",
    )

    plt.show()
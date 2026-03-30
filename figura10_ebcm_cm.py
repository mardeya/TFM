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
    "sim": '#2A9D8F',      # simulacions
    "ebcm": '#6D597A',     # EBCM
    "prefmix": '#BC6C25',  # EBCM amb correlacions de grau
}


def shift_time_to_I_threshold(t, I, Ith=100):
    I = np.asarray(I)
    t = np.asarray(t)
    idx = np.where(I >= Ith)[0]

    if len(idx) == 0:
        return None

    t0 = t[idx[0]]
    return t - t0, I


# Simulacions Gillespie
def run_simulations(G, N, tau, gamma, n_sims=12, initial_I=1, Ith=100, tmax=60, seed=0):
    rng = np.random.default_rng(seed)
    curves = []

    for r in range(n_sims):
        print(r)
        init_inf = rng.choice(N, size=initial_I, replace=False)
        t, S, I, R, _, _ = gillespie_SIR(
            G, tau, gamma,
            infectats_inicials=init_inf,
            t_max=tmax
        )

        shifted = shift_time_to_I_threshold(t, I, Ith=Ith)
        if shifted is not None:
            curves.append(shifted)

    return curves


# Resolem l’EBCM sobre el graf i després desplaçam la corba
# amb el mateix criteri que a les simulacions.
def model_EBCM_and_align(G, tau, gamma, rho, Ith=100, tmax=60, tcount=4001):
    t, S, I, R = EBCM_graf(G, tau, gamma, rho=rho, tmax=tmax, tcount=tcount)
    return shift_time_to_I_threshold(t, I, Ith=Ith)


# Mateixa idea que abans, però fent servir la versió
# amb correlacions de grau.
def model_prefmix_and_align(G, tau, gamma, rho, Ith=100, tmax=60, tcount=4001):
    t, S, I, R = EBCM_corr(G, tau, gamma, rho=rho, tmax=tmax, tcount=tcount)
    return shift_time_to_I_threshold(t, I, Ith=Ith)


# Generam un Configuration Model amb graus aproximadament
# poisson i després el passam a graf simple.
def make_CM_poisson(N, c, seed=0):
    rng = np.random.default_rng(seed)
    deg = rng.poisson(lam=c, size=N)

    # Si la suma de graus és senar, la corregim perquè
    # el configuration model es pugui construir.
    if deg.sum() % 2 == 1:
        deg[rng.integers(0, N)] += 1

    Gm = nx.configuration_model(deg, seed=int(rng.integers(0, 1_000_000_000)))

    # Eliminam múltiples arestes i llaços per quedar-nos
    # amb un graf simple.
    G = nx.Graph(Gm)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


# Generam un Configuration Model amb distribució de graus
# de llei de potència truncada.
def make_CM_powerlaw_truncated(N, alpha=2.5, kmin=1, kmax=50, seed=0):
    rng = np.random.default_rng(seed)
    ks = np.arange(kmin, kmax + 1)
    probs = ks ** (-alpha)
    probs = probs / probs.sum()

    deg = rng.choice(ks, size=N, p=probs)

    # Tornam a forçar que la suma de graus sigui parella.
    if deg.sum() % 2 == 1:
        deg[rng.integers(0, N)] += 1

    Gm = nx.configuration_model(deg, seed=int(rng.integers(0, 1_000_000_000)))
    G = nx.Graph(Gm)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


# Aquesta funció dibuixa tres panells, cadascun amb un valor
# diferent del paràmetre del model.
def plot_cm_three_params(
    model_name,
    make_graph_fn,
    param_list,
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
):
    assert len(param_list) == 3, "param_list must have length 3"

    set_tfm_style()

    if rho_model is None:
        rho_model = 1.0 / N

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    for j, param in enumerate(param_list):
        ax = axes[j]

        G = make_graph_fn(N, param, seed=int(rng.integers(0, 1_000_000_000)))

        # Primer dibuixam les simulacions individuals,
        # més fines i transparents.
        sim_curves = run_simulations(
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
                t_s, I_s / N,
                linewidth=1.0,
                alpha=0.30,
                color=TFM_COLORS["sim"]
            )

        # Afegim la corba teòrica de l’EBCM.
        ebcm = model_EBCM_and_align(
            G, tau=tau, gamma=gamma,
            rho=rho_model, Ith=Ith, tmax=tmax
        )
        if ebcm is not None:
            t_m, I_m = ebcm
            ax.plot(
                t_m, I_m / N,
                linewidth=2.7,
                color=TFM_COLORS["ebcm"],
                label="EBCM"
            )

        # I també la versió amb correlacions de grau.
        pm = model_prefmix_and_align(
            G, tau=tau, gamma=gamma,
            rho=rho_model, Ith=Ith, tmax=tmax
        )
        if pm is not None:
            t_m, I_m = pm
            ax.plot(
                t_m, I_m / N,
                linewidth=2.7,
                color=TFM_COLORS["prefmix"],
                dashes=[4, 2, 1, 2, 1, 2],
                label="EBCM correlació graus"
            )

        # Aquí només preparam el text del paràmetre per si
        # més endavant el vols recuperar al títol.
        if model_name == "CM-Poisson":
            param_tex = rf"c={param}"
        else:
            param_tex = rf"\alpha={param}"

        ax.set_xlabel(r"$t$", fontsize=16)
        ax.set_ylabel(r"$I(t)$", fontsize=16)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 0.4)

        # La llegenda només la posam al primer panell
        # per no carregar massa la figura.
        if j == 0:
            ax.plot([], [], color=TFM_COLORS["sim"], linewidth=2, label="Simulacions")
            ax.legend(loc="upper right", frameon=True, fontsize=16)

    fig.savefig(filename, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    N = 30_000
    gamma = 1.0
    Ith = 100
    n_sims = 50
    tmax = 60

    # Figura per al cas poisson, comparant tres valors
    # diferents del grau mitjà.
    plot_cm_three_params(
        model_name="CM-Poisson",
        make_graph_fn=make_CM_poisson,
        param_list=[5, 7, 9],
        tau=0.4,
        filename="figures_memoria/ebcm_cm.png",
        N=N,
        gamma=gamma,
        Ith=Ith,
        n_sims=n_sims,
        tmax=tmax,
        xlim=(0, 16),
        seed=12345,
    )

    plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_P_teoria_vs_simulacio(
    tau_vals,
    Pth,
    Psim,
    gamma,
    fitxer_sortida,
    titol = None,
    mostrar=True,
):

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(tau_vals, Pth, linewidth=1.6, label=r"Teoria (punt fix)", color="#2A9D8F")
    ax.plot(tau_vals, Psim, linewidth=1.6, label=r"Simulació", color="#6D597A")

    ax.set_xlabel(r"$\tau$", fontsize=16)
    ax.set_ylabel(r"$\mathcal{P}$", fontsize=16)
    ax.set_ylim(-0.02, 1.02)

    if titol is not None:
        ax.set_title(titol)
    # else:
    #     ax.set_title(rf"Probabilitat d'epidèmia, $\gamma={gamma}$")

    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=True, loc="lower right", fontsize=16)
    plt.tight_layout()

    if fitxer_sortida is not None:
        carpeta = os.path.dirname(fitxer_sortida)
        if carpeta:
            os.makedirs(carpeta, exist_ok=True)
        plt.savefig(fitxer_sortida, dpi=200)

    if mostrar:
        plt.show()

    return fig, ax
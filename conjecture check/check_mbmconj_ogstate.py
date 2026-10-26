"""Numerical check of the paper's counterexample to the MBM conjecture .
For every d the separable state of eq 35 is measured with the observables of eq 36, and
|Cor{A1,B1}|+|Cor{A2,B2}| is compared with the conjectured bound 2-4/(2+d) of eq 38.
."""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from correlation import PCC
from results_io import run_checkpointed, load_results

RESULTS_FILE = "mbm_results.txt"
COMPUTE = True  # compute the missing results and save them in RESULTS_FILE
PLOT = True     # plot from the saved results in RESULTS_FILE (nothing is computed)
D_MIN, D_MAX = 2, 40
TOL = 1e-9  


HEADER = ["d", "lhs", "rhs", "diff", "violated"]


def get_bases(d: int):
    """Creates the two mutually unbiased bases (MUB) of eq 35: |e_k> is the standard basis and
    |h_l> = 1/sqrt(d) Σ_k w^{kl}|e_k>, w=e^{2πi/d}, its discrete Fourier transform."""
    ek = [qt.basis(d, k) for k in range(d)]
    hk = []
    for l in range(d):
        el_np = ek[l].full().flatten()
        hk_np = np.fft.ifft(el_np, norm="ortho")  # Fourier transform of |e_l>
        hk.append(qt.Qobj(hk_np))
    return ek, hk


def separable_dm(d: int):
    """Separable state of eq 35: ¼ Σ_{k=0,1} (Π_ek⊗Π_ek + Π_hk⊗Π_hk),
    an equal mixture of 4 product states."""
    ek, hk = get_bases(d)
    rho_sep = 0
    for k in (0, 1):
        P_ek = qt.ket2dm(ek[k])
        P_hk = qt.ket2dm(hk[k])
        term1 = qt.tensor(P_ek, P_ek)
        term2 = qt.tensor(P_hk, P_hk)
        rho_sep += ((term1 + term2)/4)
    return rho_sep


def counterexample_setup(d: int):
    """State of eq 35 and the complementary observables of eq 36 with ε=0:
    A1=B1=Π_e0-Π_e1, A2=B2=Π_h0-Π_h1."""
    e_k, h_k = get_bases(d)
    Pi_e = [qt.ket2dm(e) for e in e_k]
    Pi_h = [qt.ket2dm(h) for h in h_k]
    A1 = B1 = Pi_e[0] - Pi_e[1]  # eq 36a
    A2 = B2 = Pi_h[0] - Pi_h[1]  # eq 36b
    return separable_dm(d), A1, B1, A2, B2


def check_conjecture_bound(d: int):
    """Returns lhs = |Cor{A1,B1}|+|Cor{A2,B2}| of the counterexample, rhs = 2-4/(2+d) (eq 38)
    and whether lhs goes above rhs ."""
    rho, A1, B1, A2, B2 = counterexample_setup(d)
    lhs = abs(PCC(rho, A1, B1)) + abs(PCC(rho, A2, B2))  # left hand side of eq 38
    rhs = 2 - 4 / (2 + d)                                # right hand side of eq 38
    return lhs, rhs, lhs > rhs + TOL


def compute_row(i: int):
    """Row i of the results file: d = D_MIN + i"""
    d = D_MIN + i
    lhs, rhs, violated = check_conjecture_bound(d)
    return [d, lhs, rhs, lhs - rhs, int(violated)]


def print_table(data):
    """Saved results as a table, one row per d"""
    print(f"{'d':<3} | {'lhs':<12} | {'rhs':<12} | {'lhs-rhs':<10} | above bound")
    for row in data:
        print(f"{int(row['d']):<3} | {row['lhs']:<12.8f} | {row['rhs']:<12.8f} | "
              f"{row['diff']:<10.1e} | {bool(row['violated'])}")


def plot_results(dims, lhs_vals):
    """Fig 3 style: bound 2-4/(2+d) as a curve with a shaded region below it, one column per d,
    and the computed counterexample values as red dots on the column of their d."""
    blue = "#5b7fbf"
    d_cont = np.linspace(dims.min(), dims.max(), 400)
    bound = lambda d: 2 - 4 / (2 + d)
    _, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(d_cont, 1, bound(d_cont), color="#dfe6f1", zorder=0)
    ax.vlines(dims, 1, bound(dims), color="#c3d0e4", lw=3, zorder=1)  # one column per d
    ax.plot(d_cont, bound(d_cont), color=blue, lw=2, zorder=2)
    ax.plot(dims, bound(dims), "o", color=blue, ms=6, zorder=3)
    ax.plot(dims, lhs_vals, "o", color="red", ms=3, zorder=4, label="Separable state of eq 35")
    # ax.text(5, 1.8, "Entangled states", fontsize=12)
    # ax.text(12, 1.3, "States that cannot be\nconclusively categorized as\nseparable or entangled",
    #         fontsize=12, ha="center", va="center")
    ax.set_xlim(dims.min() - 1, dims.max())
    ax.set_ylim(1, 2)
    ax.set_xticks([2, 5, 10, 15, 20, 40])
    ax.set_xlabel(r"Dimensions $d$")
    ax.set_ylabel(r"Entanglement criterion bound  $2-4/(2+d)$")
    ax.legend(loc="lower right")
    plt.savefig("graph_mbm.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # delete mbm_results.txt to recompute, e.g. after changing D_MIN, D_MAX or HEADER
    if COMPUTE:
        run_checkpointed(RESULTS_FILE, HEADER, D_MAX - D_MIN + 1, compute_row)
    data = load_results(RESULTS_FILE)
    print_table(data)
    if PLOT:
        plot_results(data["d"], data["lhs"])

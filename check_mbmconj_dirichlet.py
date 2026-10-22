"""Random separable states with the structure of the counterexample of eq 35, tested against eq 38.
eq 35 is ¼ Σ_{k=0,1} (Π_ek⊗Π_ek + Π_hk⊗Π_hk). Here every sample is a random mixture
Σ_j p_j Π_xj⊗Π_xj with random weights p (Dirichlet) and random vectors x_j picked from the two
mutually unbiased bases {|e_k>} ∪ {|h_k>}. The observables are always those of eq 36, and
|Cor{A1,B1}|+|Cor{A2,B2}| is compared with the conjectured bound 2-4/(2+d) of eq 38.
"""


import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from correlation import PCC
from results_io import run_checkpointed, load_results

RESULTS_FILE = "countermbm_results.txt"
COMPUTE = True  # compute the missing results and save them in RESULTS_FILE
PLOT = True     # plot from the saved results in RESULTS_FILE (nothing is computed)
D_MIN, D_MAX = 2, 26
N_SAMPLES = 100  # random states per d
N_TERMS = 4     # product states in each mixture (eq 35 has 4)
SEED = 0        # sample i always uses the same random numbers, so a resumed run gives the same results
TOL = 1e-9      # ignore floating point differences of ~1e-16 when comparing with the bound

# d: dimension, lhs/rhs: two sides of eq 38, diff: lhs-rhs, violated: 1 if lhs > rhs+TOL else 0
# (diff and violated are computed before saving: the file keeps only 10 significant digits)
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


def observables(d: int):
    """Complementary observables of eq 36 with ε=0:
    A1=B1=Π_e0-Π_e1, A2=B2=Π_h0-Π_h1 (ε→0 limit gives the same PCC values)."""
    e_k, h_k = get_bases(d)
    A1 = qt.ket2dm(e_k[0]) - qt.ket2dm(e_k[1])  # eq 36a
    A2 = qt.ket2dm(h_k[0]) - qt.ket2dm(h_k[1])  # eq 36b
    return A1, A1, A2, A2


def random_separable_dm(d: int, rng):
    """Random state with the separable structure of eq 35: Σ_j p_j Π_xj⊗Π_xj, j=1..N_TERMS,
    weights p uniformly random on the simplex (Σp_j=1) and each |x_j> a random vector of {|e_k>} ∪ {|h_k>}."""
    ek, hk = get_bases(d)
    pool = ek + hk
    p = rng.dirichlet(np.ones(N_TERMS))
    picks = rng.integers(len(pool), size=N_TERMS)
    rho_sep = 0
    for p_j, j in zip(p, picks):
        P_x = qt.ket2dm(pool[j])
        rho_sep += p_j * qt.tensor(P_x, P_x)
    return rho_sep


def check_conjecture_bound(rho, d: int):
    """Returns lhs = |Cor{A1,B1}|+|Cor{A2,B2}| of rho, rhs = 2-4/(2+d) (eq 38)
    and whether lhs goes above rhs (beyond floating point error)."""
    A1, B1, A2, B2 = observables(d)
    lhs = abs(PCC(rho, A1, B1)) + abs(PCC(rho, A2, B2))  # left hand side of eq 38
    rhs = 2 - 4 / (2 + d)                                # right hand side of eq 38
    return lhs, rhs, lhs > rhs + TOL


def compute_row(i: int):
    """Row i of the results file: sample i % N_SAMPLES of d = D_MIN + i // N_SAMPLES"""
    d = D_MIN + i // N_SAMPLES
    rng = np.random.default_rng([SEED, i])
    lhs, rhs, violated = check_conjecture_bound(random_separable_dm(d, rng), d)
    return [d, lhs, rhs, lhs - rhs, int(violated)]


def print_table(data):
    """Largest lhs per d and how many samples go above the bound"""
    print(f"{'d':<3} | {'max lhs':<12} | {'rhs':<12} | above bound")
    for d in np.unique(data["d"]):
        rows = data[data["d"] == d]
        print(f"{int(d):<3} | {rows['lhs'].max():<12.8f} | {rows['rhs'][0]:<12.8f} | "
              f"{int(rows['violated'].sum())}/{len(rows)}")


def plot_results(data):
    """Fig 3 style: bound 2-4/(2+d) as a curve with a shaded region below it, one column per d,
    and every random sample as a red dot on the column of its d."""
    blue = "#5b7fbf"
    dims = np.unique(data["d"])
    d_cont = np.linspace(dims.min(), dims.max(), 400)
    bound = lambda d: 2 - 4 / (2 + d)
    _, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(d_cont, 0, bound(d_cont), color="#dfe6f1", zorder=0)
    ax.vlines(dims, 0, bound(dims), color="#c3d0e4", lw=3, zorder=1)  # one column per d
    ax.plot(d_cont, bound(d_cont), color=blue, lw=2, zorder=2)
    ax.plot(dims, bound(dims), "o", color=blue, ms=6, zorder=3)
    jitter = np.random.default_rng(1).uniform(-0.15, 0.15, len(data))  # so equal values don't hide each other
    ax.plot(data["d"] + jitter, data["lhs"], "o", color="red", ms=3, zorder=4,
            label=f"Random probability states ({N_SAMPLES} per d)")
    # ax.text(5, 1.85, "Entangled states", fontsize=12)
    # ax.text(12, 1.3, "States that cannot be\nconclusively categorized as\nseparable or entangled",
    #         fontsize=12, ha="center", va="center")
    ax.set_xlim(dims.min() - 1, dims.max() + 0.5)
    ax.set_ylim(0, 2)  # starts at 0 (paper: 1) because most samples give values below 1
    ax.set_xticks([2, 5, 10, 15, 20, 26])
    ax.set_xlabel(r"Dimensions $d$")
    ax.set_ylabel(r"Entanglement criterion bound  $2-4/(2+d)$")
    ax.legend(loc="lower right")
    plt.savefig("graph_counter.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # delete countermbm_results.txt to recompute, e.g. after changing D_MIN, D_MAX, N_SAMPLES or N_TERMS
    if COMPUTE:
        run_checkpointed(RESULTS_FILE, HEADER, (D_MAX - D_MIN + 1) * N_SAMPLES, compute_row)
    data = load_results(RESULTS_FILE)
    print_table(data)
    if PLOT:
        plot_results(data)

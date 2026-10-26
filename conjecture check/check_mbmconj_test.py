import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from correlation import PCC
from results_io import run_checkpointed, load_results

RESULTS_FILE = "randombasis_results.txt"
COMPUTE = True  
PLOT = True     
D_MIN, D_MAX = 2, 20
N_SAMPLES = 3  # random states per d
N_TERMS = 4     # product states in each mixture (eq 35 has 4)
SEED = 0        
TOL = 1e-9      
HEADER = ["d", "lhs", "rhs", "diff", "violated"]


def get_rand_bases(d: int, rng):
    """Creates two mutually unbiased bases (MUB): |e_k> = U|k> is a Haar random basis (U a Haar
    random unitary) and |h_l> = 1/sqrt(d) Σ_k w^{kl}|e_k>, w=e^{2πi/d}, its discrete Fourier transform."""
    U = qt.rand_unitary(d, distribution="haar", seed=rng).full()
    ek = [qt.Qobj(U[:, k]) for k in range(d)]  # columns of U
    hk = []
    for l in range(d):
        l_np = qt.basis(d, l).full().flatten()
        hk_np = U @ np.fft.ifft(l_np, norm="ortho")  # Fourier transform of |e_l>
        hk.append(qt.Qobj(hk_np))
    return ek, hk


def observables(d: int, rng):
    """Complementary observables of eq 36 with ε=0 in a random basis:
    A1=B1=Π_e0-Π_e1, A2=B2=Π_h0-Π_h1 ."""
    e_k, h_k = get_rand_bases(d, rng)
    A1 = qt.ket2dm(e_k[0]) - qt.ket2dm(e_k[1])  # eq 36a
    A2 = qt.ket2dm(h_k[0]) - qt.ket2dm(h_k[1])  # eq 36b
    return A1, A1, A2, A2


def random_separable_dm(d: int, rng):
    """Random state with the separable structure of eq 35: Σ_j p_j Π_xj⊗Π_xj, j=1..N_TERMS,
    fixed weights p_j = 1/N_TERMS = 1/4 and each |x_j> a Haar random ket of dimension d."""
    p = 1 / N_TERMS
    rho_sep = 0
    for _ in range(N_TERMS):
        P_x = qt.ket2dm(qt.rand_ket(d, seed=rng))
        rho_sep += p * qt.tensor(P_x, P_x)
    return rho_sep



def check_conjecture_bound(rho, d: int, rng):
    """Returns lhs = |Cor{A1,B1}|+|Cor{A2,B2}| of rho with random observables, rhs = 2-4/(2+d) (eq 38)
    and whether lhs goes above rhs (beyond floating point error)."""
    A1, B1, A2, B2 = observables(d, rng)
    lhs = abs(PCC(rho, A1, B1)) + abs(PCC(rho, A2, B2))  # left hand side of eq 38
    rhs = 2 - 4 / (2 + d)                                # right hand side of eq 38
    return lhs, rhs, lhs > rhs + TOL


def compute_row(i: int):
    """Row i of the results file: sample i % N_SAMPLES of d = D_MIN + i // N_SAMPLES"""
    d = D_MIN + i // N_SAMPLES
    rng = np.random.default_rng([SEED, i])
    lhs, rhs, violated = check_conjecture_bound(random_separable_dm(d, rng), d, rng)
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
            label=f"Haar random states ({N_SAMPLES} per d)")
    # ax.text(5, 1.85, "Entangled states", fontsize=12)
    # ax.text(12, 1.3, "States that cannot be\nconclusively categorized as\nseparable or entangled",
    #         fontsize=12, ha="center", va="center")
    ax.set_xlim(dims.min() - 1, dims.max() + 0.5)
    ax.set_ylim(0, 2)  # starts at 0 (paper: 1) because most samples give values below 1
    ax.set_xticks([2, 5, 10, 15, 20,26])
    ax.set_xlabel(r"Dimensions $d$")
    ax.set_ylabel(r"Entanglement criterion bound  $2-4/(2+d)$")
    ax.legend(loc="center right")  # the dots fill the bottom of the plot
    plt.savefig("graph_randombasis.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # delete counterhaar_results.txt to recompute, e.g. after changing D_MIN, D_MAX, N_SAMPLES or N_TERMS
    if COMPUTE:
        run_checkpointed(RESULTS_FILE, HEADER, (D_MAX - D_MIN + 1) * N_SAMPLES, compute_row)
    data = load_results(RESULTS_FILE)
    print_table(data)
    if PLOT:
        plot_results(data)
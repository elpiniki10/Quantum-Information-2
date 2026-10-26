import os
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from correlation import PCC
from results_io import run_checkpointed, load_results

RESULTS_FILE = "obs_2results.txt"
COMPUTE = True  
PLOT = True     
D_MIN, D_MAX = 4, 5
N_SAMPLES = 2000  # random states per d
N_TERMS = 4     # product states in each mixture (eq 35 has 4)
SEED = 0        
TOL = 1e-9      
DETAILS_DIR = "obs_details_eq36"  # observables of samples with lhs > 0.7*rhs
HEADER = ["d", "lhs", "rhs", "diff", "violated", "flagged"]


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
    details = None
    if lhs > 0.7*rhs + TOL:  # keep eigenstates and trace norm of each observable
        details = {}
        for name, obs in zip(["A1", "B1", "A2", "B2"], [A1, B1, A2, B2]):
            eigvals, eigvecs = obs.eigenstates()
            details[f"{name}_eigvals"] = eigvals
            details[f"{name}_eigvecs"] = np.column_stack([v.full().ravel() for v in eigvecs])#for each eigenvector v in the list eigvecs, convert it to a NumPy array and to 1d
            details[f"{name}_norm"] = obs.norm(norm="tr")
    return lhs, rhs, lhs > rhs + TOL, details


def save_details(path: str, d: int, lhs: float, rhs: float, details: dict):
    """Writes the eigenvalues, eigenvectors (as columns) and trace norm of each observable in a readable txt file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)#creates the directory
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"d = {d}\nlhs = {lhs:.10g}\nrhs = {rhs:.10g}\n")
        for name in ["A1", "B1", "A2", "B2"]:
            f.write(f"\n{name}: trace norm = {details[name + '_norm']:.10g}\n")
            f.write(f"eigvals = {np.array2string(details[name + '_eigvals'], precision=6, suppress_small=True)}\n")
            f.write("eigvecs (columns) =\n")
            f.write(np.array2string(details[name + "_eigvecs"], precision=6, suppress_small=True, max_line_width=200) + "\n")


def compute_row(i: int):
    """Row i of the results file: sample i % N_SAMPLES of d = D_MIN + i // N_SAMPLES.
    Arrays don't fit in the results file, so flagged samples are saved in DETAILS_DIR/row_i.txt"""
    d = D_MIN + i // N_SAMPLES
    rng = np.random.default_rng([SEED, i])
    lhs, rhs, violated, details = check_conjecture_bound(random_separable_dm(d, rng), d, rng)
    if details is not None:
        save_details(os.path.join(DETAILS_DIR, f"row_{i}.txt"), d, lhs, rhs, details)
    return [d, lhs, rhs, lhs - rhs, int(violated), int(details is not None)]


def print_table(data):
    """Largest lhs per d and how many samples go above the bound"""
    print(f"{'d':<3} | {'max lhs':<12} | {'rhs':<12} | above bound")
    for d in np.unique(data["d"]):
        rows = data[data["d"] == d]
        print(f"{int(d):<3} | {rows['lhs'].max():<12.8f} | {rows['rhs'][0]:<12.8f} | "
              f"{int(rows['violated'].sum())}/{len(rows)}")


def plot_histograms(data):
    """Histograms of the eigenvalues, trace norms and eigenvector components |<k|v>|^2 (eigenvalue ±1)
    of A1, A2 in the flagged samples, one row per d. The observables are rebuilt from the seed of their row."""
    ds = np.unique(data["d"]).astype(int)
    fig, axes = plt.subplots(len(ds), 3, figsize=(12, 3.5 * len(ds)), squeeze=False)
    for ax_row, d in zip(axes, ds):
        eigvals, norms, comps = [], [], []
        flagged = np.flatnonzero((data["d"] == d) & (data["flagged"] == 1))
        for i in flagged:
            rng = np.random.default_rng([SEED, i])
            random_separable_dm(d, rng)  # same random draws as compute_row, so we get the same observables
            A1, _, A2, _ = observables(d, rng)
            for obs in [A1, A2]:
                vals, vecs = obs.eigenstates()
                eigvals.extend(vals)
                norms.append(obs.norm(norm="tr"))
                for val, v in zip(vals, vecs):
                    if abs(val) > 0.5:
                        comps.extend(np.abs(v.full().ravel()) ** 2)
        ax_row[0].hist(eigvals, bins="auto", color="#3a6fc4")
        ax_row[0].set(title=f"d={d}: eigenvalues", ylabel="count")
        ax_row[1].hist(norms, bins="auto", color="#3a6fc4")
        ax_row[1].set(title=f"d={d}: trace norm")
        ax_row[2].hist(comps, bins="auto", color="#3a6fc4")
        ax_row[2].set(title=f"d={d}: |<k|v>|² of ±1 eigenvectors", xlabel="|<k|v>|²")
    fig.suptitle(f"Observables A1, A2 of samples with lhs > 0.7·rhs")
    fig.tight_layout()
    fig.savefig("graph_obs_histograms_eq36.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    # delete RESULTS_FILE to recompute, e.g. after changing D_MIN, D_MAX, N_SAMPLES or N_TERMS
    if COMPUTE:
        run_checkpointed(RESULTS_FILE, HEADER, (D_MAX - D_MIN + 1) * N_SAMPLES, compute_row)
    data = load_results(RESULTS_FILE)
    print_table(data)
    if PLOT:
        plot_histograms(data)
   
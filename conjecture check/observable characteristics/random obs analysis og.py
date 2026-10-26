import os
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from correlation import PCC
from results_io import run_checkpointed, load_results

RESULTS_FILE = "obs_4results.txt"
COMPUTE = True  
PLOT = True     
D_MIN, D_MAX = 4,4
N_SAMPLES = 2000  # random states per d
N_TERMS = 4     # product states in each mixture (eq 35 has 4)
SEED = 0        
TOL = 1e-9      
DETAILS_DIR = "obs_details_random2"  # observables of every sample, and ranked by lhs per d
N_TOP = 50      # samples with the highest lhs per d in the ranked figures
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

def random_nondegenerate_spectrum(d: int, rng, low=-1.0, high=1.0, min_gap=1e-3):
    """Draws d distinct real eigenvalues uniformly in [low, high], with consecutive gaps ≥ min_gap."""
    while True:
        lam = np.sort(rng.uniform(low, high, size=d))
        if d == 1 or np.min(np.diff(lam)) >= min_gap:
            return lam


def observables(d: int, rng, lam1=None, lam2=None):
    """Complementary observables in a random MUB pair:
        X1 = Σ_k λ1k |e_k><e_k|,   X2 = Σ_k λ2k |h_k><h_k|,
    with {e_k}, {h_k} mutually unbiased and λ1, λ2 non-degenerate.
    A1 = B1 = X1, A2 = B2 = X2."""
    e_k, h_k = get_rand_bases(d, rng)

    lam1 = random_nondegenerate_spectrum(d, rng) if lam1 is None else np.asarray(lam1, float)
    lam2 = random_nondegenerate_spectrum(d, rng) if lam2 is None else np.asarray(lam2, float)
    for name, lam in (("lam1", lam1), ("lam2", lam2)):
        if lam.shape != (d,):
            raise ValueError(f"{name} must have length d={d}")
        if len(np.unique(lam)) != d:
            raise ValueError(f"{name} must be non-degenerate")

    X1 = sum(l * qt.ket2dm(e) for l, e in zip(lam1, e_k))
    X2 = sum(l * qt.ket2dm(h) for l, h in zip(lam2, h_k))
    return X1, X1, X2, X2


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
    return lhs, rhs, lhs > rhs + TOL, obs_details(A1, B1, A2, B2)


def obs_details(A1, B1, A2, B2):
    """Eigenstates and trace norm of each observable"""
    details = {}
    for name, obs in zip(["A1", "B1", "A2", "B2"], [A1, B1, A2, B2]):
        eigvals, eigvecs = obs.eigenstates()
        details[f"{name}_eigvals"] = eigvals
        details[f"{name}_eigvecs"] = np.column_stack([v.full().ravel() for v in eigvecs])#for each eigenvector v in the list eigvecs, convert it to a NumPy array and to 1d
        details[f"{name}_norm"] = obs.norm(norm="tr")
    return details


def write_details(f, details: dict):
    """Writes the eigenvalues, eigenvectors (as columns) and trace norm of each observable to the open file f"""
    for name in ["A1", "B1", "A2", "B2"]:
        f.write(f"\n{name}: trace norm = {details[name + '_norm']:.10g}\n")
        f.write(f"eigvals = {np.array2string(details[name + '_eigvals'], precision=6, suppress_small=True)}\n")
        f.write("eigvecs (columns) =\n")
        f.write(np.array2string(details[name + "_eigvecs"], precision=6, suppress_small=True, max_line_width=200) + "\n")


def save_details(path: str, d: int, lhs: float, rhs: float, details: dict):
    """Writes the details of one sample in a readable txt file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)#creates the directory
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"d = {d}\nlhs = {lhs:.10g}\nrhs = {rhs:.10g}\n")
        write_details(f, details)


def save_ranked(data):
    """Writes DETAILS_DIR/ranked_d{d}.txt for each d: every sample of that d ordered from the highest lhs
    (rank 1) to the lowest, with its |Cor| terms and observables. These are rebuilt from the seed of their row."""
    os.makedirs(DETAILS_DIR, exist_ok=True)
    for d in np.unique(data["d"]).astype(int):
        rows = np.flatnonzero(data["d"] == d)
        ranked = rows[np.argsort(data["lhs"][rows])[::-1]]
        path = os.path.join(DETAILS_DIR, f"ranked_d{d}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"d = {d}\nrhs = {data['rhs'][ranked[0]]:.10g}\n")
            for rank, i in enumerate(ranked, start=1):
                rng = np.random.default_rng([SEED, i])
                rho = random_separable_dm(d, rng)  # same random draws as compute_row, so we get the same state and observables
                A1, B1, A2, B2 = observables(d, rng)
                f.write(f"\n{'=' * 60}\nrank {rank} (row {i})\nlhs = {data['lhs'][i]:.10g}\n"
                        f"|Cor{{A1,B1}}| = {abs(PCC(rho, A1, B1)):.10g}\n|Cor{{A2,B2}}| = {abs(PCC(rho, A2, B2)):.10g}\n")
                write_details(f, obs_details(A1, B1, A2, B2))
        print(f"saved {path}")


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


if __name__ == "__main__":
    # delete RESULTS_FILE to recompute, e.g. after changing D_MIN, D_MAX, N_SAMPLES or N_TERMS
    if COMPUTE:
        run_checkpointed(RESULTS_FILE, HEADER, (D_MAX - D_MIN + 1) * N_SAMPLES, compute_row)
    data = load_results(RESULTS_FILE)
    print_table(data)
    save_ranked(data)
    

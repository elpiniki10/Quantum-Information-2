"""Monte Carlo test of the conjecture of eq 38,
    |Cor{A1,B1}| + |Cor{A2,B2}| <= 2 - 4/(2+d)   for separable states,
with random separable states of the form of eq 35 and random complementary observables
    X1 = Σ_k λ1k |e_k><e_k|,   X2 = Σ_k λ2k |h_k><h_k|,   A1 = B1 = X1,  A2 = B2 = X2,
where {e_k}, {h_k} are mutually unbiased bases and λ1, λ2 are non-degenerate spectra.

The Pearson correlation is unchanged by X -> aX + b (a ≠ 0), so a spectrum only matters through
its shape (λ - λ_min)/(λ_max - λ_min). Drawing eigenvalues in [-1, 1] therefore loses nothing.
d=2: every non-degenerate spectrum is equivalent to {-1, 1}, so the spectrum has no effect.
d=3: the shape is one number, the position of the middle eigenvalue.
Each spectrum is summarised by its minimum normalised gap
    g = min_k (λ_{k+1} - λ_k) / (λ_max - λ_min)  in (0, 1/(d-1)],
g -> 0: two eigenvalues almost degenerate (X close to a projector), g = 1/(d-1): equally spaced.
g is also unchanged by X -> -X, which leaves |Cor{X,X}| unchanged. For d=3 it fixes the shape exactly.
"""
import os
import shutil
import numpy as np
import qutip as qt
import matplotlib.pyplot as plt

from correlation import PCC
from results_io import run_checkpointed, load_results

RESULTS_FILE = "obs_mub_results.txt"
DETAILS_DIR = "obs_mub_details"  # one txt file per flagged sample (lhs > FLAG_FRAC*rhs)
COMPUTE = True
PLOT = True
D_MIN, D_MAX = 2, 3
N_SAMPLES = 2000  # random samples (state + observables) per d
N_TERMS = 4       # product states in each mixture (eq 35 has 4)
SEED = 0
TOL = 1e-9        # lhs counts as above the bound only if lhs > rhs + TOL
FLAG_FRAC = 0.7   # save the observables of samples with lhs > FLAG_FRAC*rhs
N_TOP = 30        # samples with the highest lhs per d, highlighted in the plots
HEADER = ["d", "lhs", "rhs", "diff", "cor1", "cor2", "gA1", "gA2", "violated", "flagged"]


# ---------------------------------------------------------------- random states and observables

def get_rand_bases(d: int, rng):
    """Two mutually unbiased bases: |e_k> = U|k> with U a Haar random unitary, and
    |h_l> = 1/sqrt(d) Σ_k w^{kl}|e_k>, w = e^{2πi/d}, its discrete Fourier transform."""
    U = qt.rand_unitary(d, distribution="haar", seed=rng).full()
    F = np.fft.ifft(np.eye(d), axis=0, norm="ortho")  # columns: Fourier transform of |l>
    e_k = [qt.Qobj(U[:, k]) for k in range(d)]
    h_k = [qt.Qobj((U @ F)[:, l]) for l in range(d)]
    return e_k, h_k


def random_spectrum(d: int, rng, min_gap=1e-3):
    """d distinct eigenvalues, uniform in [-1, 1] and sorted, with consecutive gaps >= min_gap.
    [-1, 1] is no restriction: PCC only depends on the shape of the spectrum (see top of file)."""
    while True:
        lam = np.sort(rng.uniform(-1, 1, size=d))
        if np.min(np.diff(lam)) >= min_gap:
            return lam


def observables(d: int, rng):
    """Random complementary observables X1 (eigenbasis {e_k}) and X2 (eigenbasis {h_k}), each with
    a random non-degenerate spectrum. Returns A1, B1, A2, B2 = X1, X1, X2, X2 and the two spectra."""
    e_k, h_k = get_rand_bases(d, rng)
    lam1, lam2 = random_spectrum(d, rng), random_spectrum(d, rng)
    X1 = sum(l * qt.ket2dm(e) for l, e in zip(lam1, e_k))
    X2 = sum(l * qt.ket2dm(h) for l, h in zip(lam2, h_k))
    return (X1, X1, X2, X2), (lam1, lam2)


def random_separable_dm(d: int, rng):
    """Random separable state with the structure of eq 35: Σ_j p_j Π_xj⊗Π_xj, j=1..N_TERMS,
    with equal weights p_j = 1/N_TERMS and each |x_j> a Haar random ket of dimension d."""
    rho = 0
    for _ in range(N_TERMS):
        P_x = qt.ket2dm(qt.rand_ket(d, seed=rng))
        rho += qt.tensor(P_x, P_x) / N_TERMS
    return rho


def draw_sample(i: int):
    """State and observables of row i, always drawn in the same order from the seed [SEED, i],
    so any row can be rebuilt exactly (e.g. for plotting)."""
    d = D_MIN + i // N_SAMPLES
    rng = np.random.default_rng([SEED, i])
    rho = random_separable_dm(d, rng)
    obs, spectra = observables(d, rng)
    return d, rho, obs, spectra


# ---------------------------------------------------------------- spectrum shape

def normalised_spectrum(lam):
    """(λ - λ_min)/(λ_max - λ_min), sorted: runs from 0 to 1, the part of the spectrum PCC depends on."""
    lam = np.sort(np.asarray(lam, float))
    return (lam - lam[0]) / (lam[-1] - lam[0])


def min_gap(lam):
    """Minimum normalised gap g of a spectrum (see top of file)."""
    return np.min(np.diff(normalised_spectrum(lam)))


# ---------------------------------------------------------------- computation

def compute_row(i: int):
    """Row i of the results file: sample i % N_SAMPLES of d = D_MIN + i // N_SAMPLES.
    Columns: d, lhs and rhs of eq 38, lhs-rhs, the two |Cor| terms, the minimum gaps of the
    spectra of A1 and A2, whether lhs > rhs, whether the sample was flagged (details saved)."""
    d, rho, (A1, B1, A2, B2), (lam1, lam2) = draw_sample(i)
    cor1 = abs(PCC(rho, A1, B1))
    cor2 = abs(PCC(rho, A2, B2))
    lhs = cor1 + cor2               # left hand side of eq 38
    rhs = 2 - 4 / (2 + d)           # right hand side of eq 38
    violated = lhs > rhs + TOL
    flagged = lhs > FLAG_FRAC * rhs
    if flagged:
        save_details(os.path.join(DETAILS_DIR, f"row_{i}.txt"), i, d, lhs, rhs, cor1, cor2,
                     {"A1": A1, "A2": A2})
    return [d, lhs, rhs, lhs - rhs, cor1, cor2, min_gap(lam1), min_gap(lam2), int(violated), int(flagged)]


def save_details(path: str, i: int, d: int, lhs, rhs, cor1, cor2, obs: dict):
    """Readable txt file with the correlations and, for each observable, its eigenvalues,
    normalised spectrum, minimum gap and eigenvectors (as columns). B1 = A1 and B2 = A2."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fmt = dict(precision=6, suppress_small=True, max_line_width=200)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"row = {i}  (rebuild with draw_sample({i}))\nd = {d}\n")
        f.write(f"lhs = {lhs:.10g}\nrhs = {rhs:.10g}\n|Cor{{A1,B1}}| = {cor1:.10g}\n|Cor{{A2,B2}}| = {cor2:.10g}\n")
        for name, X in obs.items():
            eigvals, eigvecs = X.eigenstates()
            f.write(f"\n{name} (= B{name[1]}):\n")
            f.write(f"eigvals = {np.array2string(eigvals, **fmt)}\n")
            f.write(f"normalised spectrum = {np.array2string(normalised_spectrum(eigvals), **fmt)}\n")
            f.write(f"min gap = {min_gap(eigvals):.6g}\n")
            f.write("eigvecs (columns) =\n")
            f.write(np.array2string(np.column_stack([v.full().ravel() for v in eigvecs]), **fmt) + "\n")


# ---------------------------------------------------------------- output

def print_table(data):
    """Per d: largest lhs, the bound, samples above the bound and flagged samples."""
    print(f"{'d':<3} | {'max lhs':<11} | {'mean lhs':<11} | {'rhs':<11} | {'above bound':<12} | flagged")
    for d in np.unique(data["d"]):
        rows = data[data["d"] == d]
        n = len(rows)
        print(f"{int(d):<3} | {rows['lhs'].max():<11.6f} | {rows['lhs'].mean():<11.6f} | {rows['rhs'][0]:<11.6f} | "
              f"{int(rows['violated'].sum())}/{n:<10} | {int(rows['flagged'].sum())}/{n}")


def top_rows(data, d: int):
    """Row numbers of the N_TOP samples of dimension d with the highest lhs, highest first."""
    rows = np.flatnonzero(data["d"] == d)
    return rows[np.argsort(data["lhs"][rows])[::-1][:N_TOP]]


def plot_gap_map(data):
    """Which spectrum shapes give high lhs, using all samples. One row per d.
    d=2: histogram of lhs (the spectrum has no effect).
    d>=3: left, min gap of A1 vs min gap of A2 coloured by lhs, top N_TOP circled;
          right, lhs vs the min gap of each observable."""
    ds = np.unique(data["d"]).astype(int)
    fig, axes = plt.subplots(len(ds), 2, figsize=(12, 4.5 * len(ds)), squeeze=False)
    for (ax_map, ax_lhs), d in zip(axes, ds):
        sel = data["d"] == d
        rows, top = data[sel], top_rows(data, d)
        rhs = rows["rhs"][0]
        if d == 2:
            ax_map.hist(rows["lhs"], bins=50, color="0.6")
            ax_map.axvline(rhs, color="k", ls="--", label="rhs")
            ax_map.set(title="d=2: lhs of all samples (spectrum has no effect)", xlabel="lhs", ylabel="count")
            ax_map.legend()
            ax_lhs.axis("off")
            continue
        g_max = 1 / (d - 1)
        sc = ax_map.scatter(rows["gA1"], rows["gA2"], c=rows["lhs"], s=8, cmap="viridis")
        ax_map.scatter(data["gA1"][top], data["gA2"][top], s=45, facecolors="none",
                       edgecolors="red", label=f"top {N_TOP} lhs")
        fig.colorbar(sc, ax=ax_map, label="lhs")
        ax_map.set(title=f"d={d}: spectrum shape of A1 and A2", xlim=(0, g_max), ylim=(0, g_max),
                   xlabel="min gap of A1  (0 = near degenerate, max = equally spaced)", ylabel="min gap of A2")
        ax_map.legend(loc="upper left", fontsize=8)
        ax_lhs.scatter(rows["gA1"], rows["lhs"], s=5, alpha=0.4, label="vs min gap of A1")
        ax_lhs.scatter(rows["gA2"], rows["lhs"], s=5, alpha=0.4, label="vs min gap of A2")
        ax_lhs.axhline(rhs, color="k", ls="--", lw=1, label="rhs")
        ax_lhs.set(title=f"d={d}: lhs vs spectrum shape", xlabel="min gap", ylabel="lhs", xlim=(0, g_max))
        ax_lhs.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("graph_obs_mub_gapmap.png", dpi=150)


def plot_ranked(data):
    """The N_TOP samples with the highest lhs per d, from rank 1 (highest) down, one column per d.
    Row 1: lhs and the two |Cor| terms against the bound.
    Rows 2-3: normalised spectra of A1 and A2 (0 = λ_min, 1 = λ_max, interior points = the shape).
    The spectra are rebuilt from the seed of each row."""
    ds = np.unique(data["d"]).astype(int)
    fig, axes = plt.subplots(3, len(ds), figsize=(6 * len(ds), 10), squeeze=False, sharex=True)
    for col, d in zip(axes.T, ds):
        top = top_rows(data, d)
        rank = np.arange(1, len(top) + 1)
        spec1, spec2 = [], []
        for i in top:
            _, _, _, (lam1, lam2) = draw_sample(int(i))
            spec1.append(normalised_spectrum(lam1))
            spec2.append(normalised_spectrum(lam2))

        col[0].plot(rank, data["lhs"][top], "o-", ms=3, label="lhs")
        col[0].plot(rank, data["cor1"][top], "o", ms=3, label="|Cor{A1,B1}|")
        col[0].plot(rank, data["cor2"][top], "o", ms=3, label="|Cor{A2,B2}|")
        col[0].axhline(data["rhs"][top[0]], color="k", ls="--", lw=1, label="rhs")
        col[0].set(title=f"d={d}", ylabel="correlation")
        col[0].legend(fontsize=8)
        for ax, spec, name in [(col[1], np.array(spec1), "A1"), (col[2], np.array(spec2), "A2")]:
            for k in range(d):
                ax.plot(rank, spec[:, k], "o", ms=3, label=f"λ_{k + 1}")
            ax.set(ylabel=f"normalised spectrum of {name}", ylim=(-0.05, 1.05))
            ax.legend(fontsize=8, loc="center right")
        col[2].set(xlabel="rank (1 = highest lhs)")
    fig.suptitle(f"The {N_TOP} samples with the highest lhs")
    fig.tight_layout()
    fig.savefig("graph_obs_mub_ranked.png", dpi=150)


if __name__ == "__main__":
    # delete RESULTS_FILE to recompute, e.g. after changing D_MIN, D_MAX, N_SAMPLES, N_TERMS or SEED
    if COMPUTE:
        if not os.path.exists(RESULTS_FILE):  # fresh run: remove details left by an older run
            shutil.rmtree(DETAILS_DIR, ignore_errors=True)
        run_checkpointed(RESULTS_FILE, HEADER, (D_MAX - D_MIN + 1) * N_SAMPLES, compute_row)
    data = load_results(RESULTS_FILE)
    print_table(data)
    if PLOT:
        plot_gap_map(data)
        plot_ranked(data)
        plt.show()
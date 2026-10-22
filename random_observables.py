"""Random complementary observables (section 2.3 eq 5-6, and footnote 5 of the paper).
X1 = Σ_k λ1k |e_k><e_k| and X2 = Σ_k λ2k |h_k><h_k| are complementary when the bases {|e_k>}, {|h_k>}
are mutually unbiased, |<e_k|h_l>|^2 = 1/d (eq 6), and the eigenvalues are non-degenerate.
Footnote 5: random eigenvalues are assigned to the spectral decompositions of mutually unbiased bases.
Here the bases are also rotated by a Haar random unitary U, since |<U e_k|U h_l>| = |<e_k|h_l>|.
"""

import numpy as np
import qutip as qt

from correlation import sum_PCC, CHSH1_value, CHSH2_value, CHSH3_value


def mub_bases(d: int, n: int = 2):
    """n<=3 mutually unbiased bases of C^d, as d x d matrices with the basis vectors as columns:
    1. |e_k>: standard basis
    2. |h_l> = 1/sqrt(d) Σ_k w^{kl}|e_k>, w=e^{2πi/d}: discrete Fourier transform (as in eq 35)
    3. |g_l> = D|h_l>, D = diag(e^{iπk(k+c)/d}), c = d mod 2: chirp (Zadoff-Chu) phases on the Fourier basis.
    For d=2 these are the eigenbases of Λ3, Λ1, Λ2. Remark 1: there are at most d+1 MUBs, here n<=3."""
    if not 1 <= n <= 3:
        raise ValueError("n must be 1, 2 or 3")
    k = np.arange(d)
    E = np.eye(d, dtype=complex)
    F = np.fft.ifft(E, axis=0, norm="ortho")  # column l = |h_l>
    D = np.exp(1j * np.pi * k * (k + d % 2) / d)
    G = D[:, None] * F                        # column l = |g_l>
    return [E, F, G][:n]


def random_eigenvalues(d: int, rng, kind: str = "random"):
    """Eigenvalues for one observable:
    "random": d uniform values in [-1,1] (non-degenerate with probability 1)
    "dichotomic": outcomes ±1 in random order, only for d=2 (Lemma 2, needed in eq 39-43)"""
    if kind == "dichotomic":
        if d != 2:
            raise ValueError("dichotomic ±1 observables are non-degenerate only for d=2")
        return rng.permutation([1.0, -1.0])
    if kind == "random":
        return rng.uniform(-1, 1, d)
    raise ValueError("kind must be 'random' or 'dichotomic'")


def random_complementary_observables(d: int, n: int = 2, rng=None, kind: str = "random", haar: bool = True):
    """n pairwise complementary observables X_i = Σ_k λ_ik |b_ik><b_ik| (eq 5) in H_d.

    Args:
        d (int): dimension of the Hilbert space
        n (int): number of observables, 2 for MBM/CHSH (eq 33, 38, 42, 43, 46) or 3 for eq 15
        rng: np.random.Generator or seed
        kind (str): "random" eigenvalues (footnote 5) or "dichotomic" ±1 (d=2 only)
        haar (bool): rotate all the bases by the same Haar random unitary

    Returns:
        list[Qobj]: the n observables
    """
    rng = np.random.default_rng(rng)
    bases = mub_bases(d, n)
    if haar:
        U = qt.rand_unitary(d, distribution="haar", seed=rng).full()
        bases = [U @ B for B in bases]
    obs = []
    for B in bases:
        lam = random_eigenvalues(d, rng, kind)
        obs.append(qt.Qobj(B @ np.diag(lam) @ B.conj().T))
    return obs


def are_complementary(X1: qt.Qobj, X2: qt.Qobj, tol: float = 1e-9) -> bool:
    """Checks the definition of section 2.3: non-degenerate eigenvalues and |<e_k|h_l>|^2 = 1/d (eq 6)"""
    l1, v1 = np.linalg.eigh(X1.full())
    l2, v2 = np.linalg.eigh(X2.full())
    d = len(l1)
    nondegenerate = np.diff(l1).min() > tol and np.diff(l2).min() > tol
    overlaps = np.abs(v1.conj().T @ v2) ** 2
    return bool(nondegenerate and np.allclose(overlaps, 1 / d, atol=tol))


def bloch_vector(X: qt.Qobj):
    """x_k = tr{X Λ_k}/2 of eq 10"""
    return np.array([(X * L).tr().real / 2 for L in (qt.sigmax(), qt.sigmay(), qt.sigmaz())])


def random_separable_2qubit(rng, n_terms: int = 4):
    """Σ_j p_j S_A,j ⊗ S_B,j with Haar random pure product states (separable by eq 4)"""
    p = rng.dirichlet(np.ones(n_terms))
    return sum(p_j * qt.tensor(qt.ket2dm(qt.rand_ket(2, distribution="haar", seed=rng)),
                               qt.ket2dm(qt.rand_ket(2, distribution="haar", seed=rng))) for p_j in p)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # complementarity of the generated observables for several d
    for d in range(2, 11):
        X = random_complementary_observables(d, 3, rng)
        ok = all(are_complementary(X[i], X[j]) for i in range(3) for j in range(i + 1, 3))
        print(f"d={d:<2} 3 pairwise complementary observables: {ok}")

    # qubits: Lemma 4 (orthogonal Bloch vectors) and Remark 2 (±1 observables anticommute)
    A = random_complementary_observables(2, 3, rng)
    print("\nexample qubit observable A1 =\n", A[0])
    print("Lemma 4, x1.x2, x1.x3, x2.x3 =",
          [round(bloch_vector(A[i]) @ bloch_vector(A[j]), 12) for i, j in [(0, 1), (0, 2), (1, 2)]])
    A1, A2 = random_complementary_observables(2, 2, rng, kind="dichotomic")
    print("Remark 2, ||A1A2+A2A1|| =", (A1 * A2 + A2 * A1).norm())

    # Theorem 1 / Corollary 1 / Theorems 2, 3: no separable state may go above the bounds
    worst = {"eq15": 0, "eq33": 0, "eq42": 0, "eq43": 0, "eq46": 0}
    for _ in range(2000):
        rho = random_separable_2qubit(rng)
        A = random_complementary_observables(2, 3, rng)
        B = random_complementary_observables(2, 3, rng)
        Ad = random_complementary_observables(2, 2, rng, kind="dichotomic")
        Bd = random_complementary_observables(2, 2, rng, kind="dichotomic")
        worst["eq15"] = max(worst["eq15"], sum_PCC(rho, A, B))
        worst["eq33"] = max(worst["eq33"], sum_PCC(rho, A[:2], B[:2]))
        worst["eq42"] = max(worst["eq42"], CHSH1_value(rho, Ad, Bd))
        worst["eq43"] = max(worst["eq43"], CHSH2_value(rho, Ad, Bd))
        worst["eq46"] = max(worst["eq46"], CHSH3_value(rho, A[:2], B[:2]))
    print("\nlargest value over 2000 random separable states (bounds: 1, 1, √2, √2, √2):")
    for eq, v in worst.items():
        print(f"  {eq}: {v:.6f}")

"""
Self-Awareness Operator: T_SA
Core implementation of the fixed-point theory of human self-awareness.

Mathematical basis: formulations/fixed_point_theory.md
"""

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Core Types
# ---------------------------------------------------------------------------

Belief = NDArray[np.float64]       # shape (n_types,) — element of Δ^n
Likelihood = NDArray[np.float64]   # shape (n_obs, n_types) — L matrix


# ---------------------------------------------------------------------------
# Hilbert Projective Metric
# ---------------------------------------------------------------------------

def hilbert_metric(p: Belief, q: Belief) -> float:
    """
    Hilbert projective metric on int(Δ^n).

    d_H(p, q) = log(max_j p_j/q_j) - log(min_j p_j/q_j)

    Requires p, q ∈ int(Δ^n) (all entries strictly positive).
    Returns 0 iff p = q (as probability vectors).
    """
    ratio = p / q
    return float(np.log(ratio.max()) - np.log(ratio.min()))


def kl_divergence(p: Belief, q: Belief, eps: float = 1e-12) -> float:
    """
    KL divergence D_KL(p || q) = Σ_j p_j log(p_j / q_j).
    Numerically stable version.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def total_variation(p: Belief, q: Belief) -> float:
    """TV distance: 0.5 * Σ_j |p_j - q_j|"""
    return float(0.5 * np.sum(np.abs(p - q)))


# ---------------------------------------------------------------------------
# SA Operator
# ---------------------------------------------------------------------------

class SelfAwarenessOperator:
    """
    T_SA: the Bayesian self-belief update operator.

    T_SA^k(q)_j = L_{kj} * q_j / Z_k(q)

    where Z_k(q) = Σ_j L_{kj} * q_j  (marginal likelihood of observing k)

    Parameters
    ----------
    likelihood : array of shape (n_obs, n_types)
        L[k, j] = P(obs=k | type=j). Must be strictly positive; columns need
        not sum to 1 (rows are the proper distributions, but column-wise
        we need strict positivity for Birkhoff's theorem).
        Convention: each COLUMN sums to 1 (it is a conditional dist over obs).
    """

    def __init__(self, likelihood: Likelihood):
        L = np.asarray(likelihood, dtype=float)
        assert L.ndim == 2, "likelihood must be 2D: (n_obs, n_types)"
        assert (L > 0).all(), "All entries of likelihood must be strictly positive"
        assert np.allclose(L.sum(axis=0), 1.0, atol=1e-9), (
            "Each column of likelihood must sum to 1: P(obs|type=j) is a distribution"
        )
        self.L = L
        self.n_obs, self.n_types = L.shape

    # ------------------------------------------------------------------
    # Operator application
    # ------------------------------------------------------------------

    def update(self, q: Belief, obs: int) -> Belief:
        """
        Apply T_SA^obs to belief q.

        q_new_j = L[obs, j] * q_j / Z
        """
        lk = self.L[obs]
        unnorm = lk * q
        Z = unnorm.sum()
        if Z < 1e-300:
            raise ValueError(f"Zero marginal likelihood for obs={obs} — check L and q")
        return unnorm / Z

    def marginal_likelihood(self, q: Belief, obs: int) -> float:
        """P(obs | q) = Σ_j L[obs,j] * q_j"""
        return float(self.L[obs] @ q)

    def marginal_obs_dist(self, q: Belief) -> NDArray:
        """P(obs) = L @ q — marginal distribution over observations"""
        return self.L @ q  # shape (n_obs,)

    # ------------------------------------------------------------------
    # Contraction analysis
    # ------------------------------------------------------------------

    def kl_sa_measure(self, true_type: int) -> float:
        """
        SA measure based on KL divergence (Section 4.6 of formulations).

        SA = min_{j ≠ θ*} D_KL( P(obs|θ*) || P(obs|θ_j) )

        Interpretable as: how hard is it to confuse the true type with
        the nearest-neighbor type?  Higher = more self-aware.

        Parameters
        ----------
        true_type : int — index of the true self-type θ*
        """
        p_true = self.L[:, true_type]   # observation dist of true type
        kl_vals = []
        for j in range(self.n_types):
            if j == true_type:
                continue
            p_j = self.L[:, j]
            kl_vals.append(kl_divergence(p_true, p_j))
        return float(min(kl_vals))

    def all_kl_separations(self, true_type: int) -> NDArray:
        """
        D_KL(P(obs|θ*) || P(obs|θ_j)) for all j ≠ θ*.
        Shape: (n_types - 1,)
        """
        p_true = self.L[:, true_type]
        vals = []
        for j in range(self.n_types):
            if j == true_type:
                continue
            vals.append(kl_divergence(p_true, self.L[:, j]))
        return np.array(vals)

    def birkhoff_ratio(self, obs: int) -> float:
        """
        Birkhoff contraction ratio c(T_SA^obs) for a single observation.

        For the diagonal map Λ^k = diag(L[obs, :]):
            Δ(Λ^k) = log(max_j L[obs,j] / min_j L[obs,j])
            c = tanh(Δ/4)

        Note: this is a bound — valid when we treat the diagonal map
        on the positive cone (see formulations Section 4 discussion).
        c → 0: high self-awareness per step.
        c → 1: low self-awareness per step.

        SA_birkhoff(obs) = 1 - c
        """
        lk = self.L[obs]
        ratio = lk.max() / lk.min()
        delta = np.log(ratio)
        c = np.tanh(delta / 4)
        return float(c)

    def birkhoff_sa_measure(self, obs: int) -> float:
        """SA = 1 - Birkhoff ratio for observation obs."""
        return 1.0 - self.birkhoff_ratio(obs)

    def expected_birkhoff_sa(self, q: Belief) -> float:
        """
        Expected SA measure (Birkhoff) over current belief q:
        SA_birkhoff = Σ_k P(obs=k|q) * (1 - c(T_SA^k))
        """
        p_obs = self.marginal_obs_dist(q)
        return float(sum(
            p_obs[k] * self.birkhoff_sa_measure(k)
            for k in range(self.n_obs)
        ))

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iterate(
        self,
        q0: Belief,
        obs_sequence: list[int],
    ) -> list[Belief]:
        """
        Iterate T_SA over a sequence of observations.

        Returns list of belief states [q^0, q^1, ..., q^T].
        """
        trajectory = [np.array(q0, dtype=float)]
        q = np.array(q0, dtype=float)
        for obs in obs_sequence:
            q = self.update(q, obs)
            trajectory.append(q.copy())
        return trajectory

    def simulate(
        self,
        q0: Belief,
        true_type: int,
        n_steps: int,
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """
        Simulate self-awareness dynamics: draw observations from true_type,
        iterate T_SA, and record convergence metrics at each step.

        Returns dict with:
            'beliefs'    : array (n_steps+1, n_types)
            'kl_to_true' : array (n_steps+1,)     D_KL(δ_θ* || q^t)
            'tv_to_true' : array (n_steps+1,)     TV(δ_θ* || q^t)
            'hilbert'    : array (n_steps+1,)     d_H(q^t, q*_ε) — uses interior approx
            'obs_seq'    : array (n_steps,)       drawn observations
        """
        if rng is None:
            rng = np.random.default_rng()

        p_obs_true = self.L[:, true_type]  # observation dist of true type

        beliefs = np.zeros((n_steps + 1, self.n_types))
        kl_to_true = np.zeros(n_steps + 1)
        tv_to_true = np.zeros(n_steps + 1)
        hilbert_dist = np.zeros(n_steps + 1)
        obs_seq = np.zeros(n_steps, dtype=int)

        # Interior approximation of δ_{θ*}: concentrate eps mass on others
        eps = 1e-8
        q_star_int = np.full(self.n_types, eps / (self.n_types - 1))
        q_star_int[true_type] = 1.0 - eps
        q_star_int /= q_star_int.sum()

        q = np.array(q0, dtype=float)
        beliefs[0] = q
        kl_to_true[0] = kl_divergence(q_star_int, q)
        tv_to_true[0] = total_variation(q_star_int, q)
        hilbert_dist[0] = hilbert_metric(q, q_star_int) if (q > 0).all() else np.nan

        for t in range(n_steps):
            obs = rng.choice(self.n_obs, p=p_obs_true)
            obs_seq[t] = obs
            q = self.update(q, obs)
            beliefs[t + 1] = q
            kl_to_true[t + 1] = kl_divergence(q_star_int, q)
            tv_to_true[t + 1] = total_variation(q_star_int, q)
            hilbert_dist[t + 1] = (
                hilbert_metric(q, q_star_int)
                if (q > 1e-300).all() else np.nan
            )

        return {
            "beliefs": beliefs,
            "kl_to_true": kl_to_true,
            "tv_to_true": tv_to_true,
            "hilbert": hilbert_dist,
            "obs_seq": obs_seq,
        }


# ---------------------------------------------------------------------------
# Likelihood matrix constructors
# ---------------------------------------------------------------------------

def make_uniform_likelihood(n_obs: int, n_types: int) -> Likelihood:
    """
    Completely uninformative likelihood: P(obs|type) = 1/n_obs for all types.
    SA = 0: observations carry no self-information.
    """
    return np.full((n_obs, n_types), 1.0 / n_obs)


def make_identity_likelihood(n_types: int) -> Likelihood:
    """
    Perfectly discriminating: obs = type (n_obs = n_types).
    SA = max: a single observation perfectly identifies the type.
    L[k, j] = 1 if k == j else eps.
    """
    eps = 1e-6
    L = np.full((n_types, n_types), eps / (n_types - 1))
    np.fill_diagonal(L, 1.0 - eps)
    L /= L.sum(axis=0, keepdims=True)
    return L


def make_noisy_identity_likelihood(n_types: int, noise: float = 0.1) -> Likelihood:
    """
    Noisy version of identity: correct observation with prob (1-noise),
    random wrong observation with prob noise.
    noise=0 → perfect discrimination, noise=1 → uniform (no info).
    """
    assert 0 <= noise <= 1
    L = np.full((n_types, n_types), noise / (n_types - 1))
    np.fill_diagonal(L, 1.0 - noise)
    L /= L.sum(axis=0, keepdims=True)
    return L


def make_random_likelihood(
    n_obs: int,
    n_types: int,
    concentration: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> Likelihood:
    """
    Random likelihood matrix via Dirichlet sampling.
    Each column j ~ Dirichlet(concentration * 1_m).
    Higher concentration → more uniform → lower SA.
    Lower concentration → more peaked → higher SA.
    """
    if rng is None:
        rng = np.random.default_rng()
    alpha = np.full(n_obs, concentration)
    L = rng.dirichlet(alpha, size=n_types).T   # shape (n_obs, n_types)
    return L


def make_davidson_likelihood(
    n_types: int = 5,
    sa_score: float = 0.7,
    n_obs: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> Likelihood:
    """
    Construct a likelihood matrix parameterized by a Davidson-style
    self-awareness score in [0, 1].

    sa_score = 0 → uniform likelihood (no self-discrimination)
    sa_score = 1 → near-identity likelihood (perfect discrimination)

    Achieved by interpolating between uniform and noisy-identity:
    L = (1 - sa_score) * L_uniform + sa_score * L_identity

    If n_obs != n_types, the identity part is constructed by assigning
    each type to its closest observation bin.
    """
    if rng is None:
        rng = np.random.default_rng()

    L_uniform = make_uniform_likelihood(n_obs, n_types)

    # For the identity part: each type owns one observation bin
    if n_obs >= n_types:
        # Assign one "primary" observation per type
        L_identity = np.full((n_obs, n_types), 0.01 / (n_obs - 1))
        for j in range(n_types):
            primary_obs = int(j * n_obs / n_types)
            L_identity[:, j] = 0.01 / (n_obs - 1)
            L_identity[primary_obs, j] = 1.0 - 0.01
        L_identity /= L_identity.sum(axis=0, keepdims=True)
    else:
        L_identity = make_noisy_identity_likelihood(n_types, noise=0.05)[:n_obs, :]
        L_identity /= L_identity.sum(axis=0, keepdims=True)

    L = (1 - sa_score) * L_uniform + sa_score * L_identity
    L /= L.sum(axis=0, keepdims=True)
    return L


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def uniform_belief(n_types: int) -> Belief:
    """Uniform prior: q_j = 1/n for all j."""
    return np.full(n_types, 1.0 / n_types)


def point_belief(n_types: int, type_idx: int, eps: float = 1e-8) -> Belief:
    """Near-point-mass at type_idx (interior approximation)."""
    q = np.full(n_types, eps / (n_types - 1))
    q[type_idx] = 1.0 - eps
    q /= q.sum()
    return q

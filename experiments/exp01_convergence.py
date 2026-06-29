"""
Experiment 01: Fixed-Point Convergence of T_SA

Question: Does T_SA converge to the true self-model q* = δ_{θ*}?
          How fast? Does it depend on SA level and starting belief?

Demonstrates:
  - Convergence of KL distance to true type over update steps
  - Exponential rate (linear in log scale) as predicted by Theorem 4.2
  - Comparison across three SA levels (low / medium / high likelihood sharpness)
  - Convergence from three different starting beliefs (uniform / wrong / right)

Run:
    python experiments/exp01_convergence.py
    Saves: results/exp01_convergence.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from experiments.sa_operator import (
    SelfAwarenessOperator,
    make_noisy_identity_likelihood,
    uniform_belief,
    point_belief,
    kl_divergence,
    total_variation,
)

os.makedirs("results", exist_ok=True)

RNG = np.random.default_rng(42)
N_TYPES = 4          # θ_1, θ_2, θ_3, θ_4
TRUE_TYPE = 0        # θ* = θ_1
N_STEPS = 80

# ---------------------------------------------------------------------------
# 1. Three SA levels via noise parameter
# ---------------------------------------------------------------------------

sa_configs = [
    ("Low SA  (noise=0.40)", 0.40, "tab:red"),
    ("Med SA  (noise=0.15)", 0.15, "tab:orange"),
    ("High SA (noise=0.02)", 0.02, "tab:green"),
]

# ---------------------------------------------------------------------------
# 2. Three starting beliefs
# ---------------------------------------------------------------------------

q_starts = {
    "Uniform prior\n(max uncertainty)": uniform_belief(N_TYPES),
    "Wrong belief\n(q = δ_{θ_2})":     point_belief(N_TYPES, 1),
    "Near-correct\n(q ≈ δ_{θ*})":      point_belief(N_TYPES, TRUE_TYPE, eps=0.3),
}

# ---------------------------------------------------------------------------
# 3. Run simulations
# ---------------------------------------------------------------------------

results = {}   # (sa_label, start_label) → dict from simulate()

for sa_label, noise, color in sa_configs:
    L = make_noisy_identity_likelihood(N_TYPES, noise=noise)
    op = SelfAwarenessOperator(L)
    sa_kl = op.kl_sa_measure(TRUE_TYPE)
    birkhoff_vals = [op.birkhoff_sa_measure(k) for k in range(op.n_obs)]

    print(f"\n{sa_label}")
    print(f"  KL-based SA measure : {sa_kl:.4f}")
    print(f"  Birkhoff SA (mean)  : {np.mean(birkhoff_vals):.4f}")

    for start_label, q0 in q_starts.items():
        sim = op.simulate(q0.copy(), TRUE_TYPE, N_STEPS, rng=RNG)
        results[(sa_label, start_label)] = sim

# ---------------------------------------------------------------------------
# 4. Plot: convergence in KL divergence (log scale)
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 11))
fig.suptitle(
    "Experiment 01 — Fixed-Point Convergence of T_SA\n"
    r"$D_{KL}(\delta_{\theta^*} \| q^t)$ over Bayesian update steps",
    fontsize=14, fontweight="bold", y=0.98,
)

gs = gridspec.GridSpec(
    2, 3,
    figure=fig,
    hspace=0.42,
    wspace=0.32,
    top=0.90, bottom=0.08,
)

steps = np.arange(N_STEPS + 1)

for col_idx, (start_label, q0) in enumerate(q_starts.items()):
    ax_log = fig.add_subplot(gs[0, col_idx])
    ax_lin = fig.add_subplot(gs[1, col_idx])

    for sa_label, noise, color in sa_configs:
        sim = results[(sa_label, start_label)]
        kl = sim["kl_to_true"]
        tv = sim["tv_to_true"]

        ax_log.semilogy(steps, kl, color=color, lw=1.8, label=sa_label)
        ax_lin.plot(steps, tv, color=color, lw=1.8, label=sa_label)

    ax_log.set_title(start_label, fontsize=10)
    ax_log.set_xlabel("Update steps t", fontsize=9)
    ax_log.set_ylabel(r"$D_{KL}(\delta_{\theta^*} \| q^t)$ (log)", fontsize=9)
    ax_log.legend(fontsize=7.5, loc="upper right")
    ax_log.grid(True, which="both", alpha=0.3)
    ax_log.set_xlim(0, N_STEPS)

    ax_lin.set_title(start_label, fontsize=10)
    ax_lin.set_xlabel("Update steps t", fontsize=9)
    ax_lin.set_ylabel(r"Total Variation $\|q^t - \delta_{\theta^*}\|_{TV}$", fontsize=9)
    ax_lin.legend(fontsize=7.5, loc="upper right")
    ax_lin.grid(True, alpha=0.3)
    ax_lin.set_xlim(0, N_STEPS)
    ax_lin.set_ylim(bottom=0)

fig.text(
    0.5, 0.01,
    "Top row: log scale shows exponential convergence rate.\n"
    "Bottom row: linear scale shows absolute error. Higher SA → faster convergence to q*.",
    ha="center", fontsize=9, style="italic",
)

plt.savefig("results/exp01_convergence.png", dpi=150, bbox_inches="tight")
print("\nSaved: results/exp01_convergence.png")
plt.show()

# ---------------------------------------------------------------------------
# 5. Print convergence rate estimates
# ---------------------------------------------------------------------------

print("\n--- Estimated convergence rates (slope in log-KL space) ---")
for sa_label, noise, color in sa_configs:
    for start_label, q0 in q_starts.items():
        kl = results[(sa_label, start_label)]["kl_to_true"]
        # Fit linear regression to log(KL) vs t (after burn-in)
        t_fit = steps[5:]
        log_kl = np.log(kl[5:] + 1e-15)
        mask = np.isfinite(log_kl)
        if mask.sum() > 5:
            slope, intercept = np.polyfit(t_fit[mask], log_kl[mask], 1)
            print(f"  {sa_label[:12]} | {start_label[:20]} -> slope = {slope:.4f} (rate ≈ e^{slope:.3f} per step)")

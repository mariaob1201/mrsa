"""
Experiment 02: SA Measure vs Likelihood Sharpness

Questions:
  1. How does SA(system) vary as the likelihood matrix becomes more/less discriminating?
  2. Are the KL-based and Birkhoff-based SA measures consistent?
  3. How does n_types affect the SA measure at a given noise level?
  4. What is the phase structure — is there a sharp transition in convergence rate?

Demonstrates:
  - SA_KL(noise) and SA_Birkhoff(noise) as functions of observation noise
  - Convergence steps-to-threshold at each SA level (empirical vs theoretical)
  - Effect of n_types: more types → harder to discriminate self
  - Geometry of the belief simplex under different noise levels (2D simplex)

Run:
    python experiments/exp02_sa_measure.py
    Saves: results/exp02_sa_measure.png
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
    kl_divergence,
)

os.makedirs("results", exist_ok=True)

RNG = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# 1. SA measure vs noise — sweep noise ∈ [0.01, 0.49]
# ---------------------------------------------------------------------------

N_TYPES = 4
TRUE_TYPE = 0
noise_vals = np.linspace(0.01, 0.49, 60)

sa_kl_vals = []
sa_birkhoff_vals = []

for noise in noise_vals:
    L = make_noisy_identity_likelihood(N_TYPES, noise=noise)
    op = SelfAwarenessOperator(L)
    sa_kl_vals.append(op.kl_sa_measure(TRUE_TYPE))
    # Average Birkhoff SA across all observations (uniform obs prior)
    sa_b = np.mean([op.birkhoff_sa_measure(k) for k in range(N_TYPES)])
    sa_birkhoff_vals.append(sa_b)

sa_kl_vals = np.array(sa_kl_vals)
sa_birkhoff_vals = np.array(sa_birkhoff_vals)

# ---------------------------------------------------------------------------
# 2. Steps to reach KL < threshold vs noise
# ---------------------------------------------------------------------------

KL_THRESHOLD = 0.05
N_STEPS_MAX = 500
noise_coarse = np.linspace(0.02, 0.45, 20)
steps_to_threshold = []

for noise in noise_coarse:
    L = make_noisy_identity_likelihood(N_TYPES, noise=noise)
    op = SelfAwarenessOperator(L)
    q0 = uniform_belief(N_TYPES)
    sim = op.simulate(q0, TRUE_TYPE, N_STEPS_MAX, rng=RNG)
    kl = sim["kl_to_true"]
    crossings = np.where(kl < KL_THRESHOLD)[0]
    steps_to_threshold.append(crossings[0] if len(crossings) > 0 else N_STEPS_MAX)

steps_to_threshold = np.array(steps_to_threshold)

# ---------------------------------------------------------------------------
# 3. Effect of n_types on SA_KL at fixed noise
# ---------------------------------------------------------------------------

noise_fixed = 0.10
n_types_vals = [2, 3, 4, 6, 8, 12]
sa_kl_by_ntypes = []

for n in n_types_vals:
    L = make_noisy_identity_likelihood(n, noise=noise_fixed)
    op = SelfAwarenessOperator(L)
    sa_kl_by_ntypes.append(op.kl_sa_measure(0))

# ---------------------------------------------------------------------------
# 4. Belief simplex trajectory (3-type system, 2D simplex)
# ---------------------------------------------------------------------------

N_TYPES_3 = 3
TRUE_TYPE_3 = 0
N_STEPS_TRAJ = 40

def to_2d_simplex(q):
    """Project 3-simplex point to 2D equilateral triangle coordinates."""
    x = 0.5 * (2 * q[1] + q[2]) / (q[0] + q[1] + q[2])
    y = (np.sqrt(3) / 2) * q[2] / (q[0] + q[1] + q[2])
    return x, y

trajectories = {}
for noise, label, color in [
    (0.40, "Low SA", "tab:red"),
    (0.15, "Med SA", "tab:orange"),
    (0.03, "High SA", "tab:green"),
]:
    L = make_noisy_identity_likelihood(N_TYPES_3, noise=noise)
    op = SelfAwarenessOperator(L)
    q0 = uniform_belief(N_TYPES_3)
    sim = op.simulate(q0, TRUE_TYPE_3, N_STEPS_TRAJ, rng=RNG)
    trajectories[label] = (sim["beliefs"], color)

# Simplex corners
corners_3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
corner_2d = [to_2d_simplex(c) for c in corners_3]

# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 12))
fig.suptitle(
    "Experiment 02 — SA Measure vs Likelihood Sharpness",
    fontsize=14, fontweight="bold", y=0.98,
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35,
                       top=0.91, bottom=0.07)

# -- Panel A: SA measure vs noise --
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot(noise_vals, sa_kl_vals, color="steelblue", lw=2, label="SA (KL-based)")
ax_a.plot(noise_vals, sa_birkhoff_vals, color="darkorange", lw=2, ls="--",
          label="SA (Birkhoff-based)")
ax_a.set_xlabel("Observation noise", fontsize=10)
ax_a.set_ylabel("SA measure", fontsize=10)
ax_a.set_title("A: SA measures vs noise\n(n_types=4)", fontsize=10)
ax_a.legend(fontsize=9)
ax_a.grid(True, alpha=0.3)
ax_a.set_xlim(noise_vals[0], noise_vals[-1])
ax_a.set_ylim(bottom=0)

# -- Panel B: Steps to convergence vs noise --
ax_b = fig.add_subplot(gs[0, 1])
ax_b.plot(noise_coarse, steps_to_threshold, "o-", color="firebrick", lw=2, ms=5)
ax_b.axhline(N_STEPS_MAX, color="gray", ls=":", lw=1, label=f"max steps = {N_STEPS_MAX}")
ax_b.set_xlabel("Observation noise", fontsize=10)
ax_b.set_ylabel(f"Steps to KL < {KL_THRESHOLD}", fontsize=10)
ax_b.set_title(f"B: Convergence time vs noise\n(threshold = KL < {KL_THRESHOLD})", fontsize=10)
ax_b.legend(fontsize=9)
ax_b.grid(True, alpha=0.3)
ax_b.set_xlim(noise_coarse[0], noise_coarse[-1])

# -- Panel C: SA_KL vs n_types --
ax_c = fig.add_subplot(gs[0, 2])
ax_c.plot(n_types_vals, sa_kl_by_ntypes, "s-", color="mediumseagreen", lw=2, ms=7)
ax_c.set_xlabel("Number of self-types  |Θ|", fontsize=10)
ax_c.set_ylabel("SA (KL-based)", fontsize=10)
ax_c.set_title(f"C: SA vs type complexity\n(noise = {noise_fixed})", fontsize=10)
ax_c.grid(True, alpha=0.3)
ax_c.set_xticks(n_types_vals)

# -- Panel D: Log KL curves --
ax_d = fig.add_subplot(gs[1, 0])
steps_arr = np.arange(N_STEPS_TRAJ + 1)
for noise, label, color in [(0.40, "Low SA", "tab:red"),
                              (0.15, "Med SA", "tab:orange"),
                              (0.03, "High SA", "tab:green")]:
    L = make_noisy_identity_likelihood(N_TYPES, noise=noise)
    op = SelfAwarenessOperator(L)
    q0 = uniform_belief(N_TYPES)
    sim = op.simulate(q0, TRUE_TYPE, N_STEPS_TRAJ, rng=RNG)
    kl = sim["kl_to_true"]
    ax_d.semilogy(steps_arr, kl + 1e-15, lw=2, color=color, label=label)
ax_d.set_xlabel("Update steps t", fontsize=10)
ax_d.set_ylabel(r"$D_{KL}(\delta_{\theta^*} \| q^t)$ (log)", fontsize=10)
ax_d.set_title("D: Convergence curves (log)\n(n_types=4, from uniform prior)", fontsize=10)
ax_d.legend(fontsize=9)
ax_d.grid(True, which="both", alpha=0.3)
ax_d.set_xlim(0, N_STEPS_TRAJ)

# -- Panel E: Belief simplex trajectories --
ax_e = fig.add_subplot(gs[1, 1])

# Draw simplex edges
tri_pts = [to_2d_simplex(c) for c in corners_3]
for i in range(3):
    x0, y0 = tri_pts[i]
    x1, y1 = tri_pts[(i+1) % 3]
    ax_e.plot([x0, x1], [y0, y1], "k-", lw=1.5, zorder=1)

# Corner labels
labels_corner = [r"$\theta_1^*$ (true)", r"$\theta_2$", r"$\theta_3$"]
offsets = [(0.02, -0.06), (-0.12, -0.06), (-0.04, 0.03)]
for (cx, cy), lbl, off in zip(tri_pts, labels_corner, offsets):
    ax_e.text(cx + off[0], cy + off[1], lbl, fontsize=9, ha="center")

for label, (beliefs, color) in trajectories.items():
    xs, ys = zip(*[to_2d_simplex(b) for b in beliefs])
    ax_e.plot(xs, ys, "-o", color=color, lw=1.5, ms=3, label=label, zorder=2)
    ax_e.plot(xs[0], ys[0], "o", color=color, ms=8, zorder=3)
    ax_e.plot(xs[-1], ys[-1], "*", color=color, ms=12, zorder=3)

ax_e.set_aspect("equal")
ax_e.set_title("E: Belief simplex trajectories\n(3-type system, uniform start → θ*)", fontsize=10)
ax_e.legend(fontsize=9, loc="upper right")
ax_e.set_xlim(-0.15, 1.15)
ax_e.set_ylim(-0.12, 1.0)
ax_e.axis("off")

# -- Panel F: Theoretical vs empirical convergence rate --
ax_f = fig.add_subplot(gs[1, 2])
noise_rate = np.linspace(0.01, 0.48, 40)
theoretical_rate = []   # D_KL(true || nearest wrong)
empirical_slope = []    # measured from log-KL linear fit

for noise in noise_rate:
    L = make_noisy_identity_likelihood(N_TYPES, noise=noise)
    op = SelfAwarenessOperator(L)
    theoretical_rate.append(op.kl_sa_measure(TRUE_TYPE))

    q0 = uniform_belief(N_TYPES)
    sim = op.simulate(q0, TRUE_TYPE, 200, rng=RNG)
    kl = sim["kl_to_true"]
    log_kl = np.log(kl[10:] + 1e-15)
    t_fit = np.arange(len(log_kl))
    mask = np.isfinite(log_kl) & (log_kl > -30)
    if mask.sum() > 10:
        slope, _ = np.polyfit(t_fit[mask], log_kl[mask], 1)
        empirical_slope.append(abs(slope))
    else:
        empirical_slope.append(np.nan)

ax_f.plot(noise_rate, theoretical_rate, color="steelblue", lw=2, label="SA_KL (theoretical rate)")
ax_f2 = ax_f.twinx()
ax_f2.plot(noise_rate, empirical_slope, color="firebrick", lw=2, ls="--",
           label="|empirical slope|")
ax_f2.set_ylabel("|Empirical log-KL slope|", fontsize=9, color="firebrick")
ax_f2.tick_params(axis="y", colors="firebrick")

ax_f.set_xlabel("Observation noise", fontsize=10)
ax_f.set_ylabel("SA_KL measure", fontsize=10, color="steelblue")
ax_f.tick_params(axis="y", colors="steelblue")
ax_f.set_title("F: Theoretical vs empirical\nconvergence rate", fontsize=10)
ax_f.grid(True, alpha=0.3)

lines_a, labels_a = ax_f.get_legend_handles_labels()
lines_b, labels_b = ax_f2.get_legend_handles_labels()
ax_f.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc="upper left")

plt.savefig("results/exp02_sa_measure.png", dpi=150, bbox_inches="tight")
print("Saved: results/exp02_sa_measure.png")
plt.show()

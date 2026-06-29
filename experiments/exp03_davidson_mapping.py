"""
Experiment 03: Davidson's Six Dimensions as Operator Parameters

Davidson (UW–Madison, Center for Healthy Minds) identified six dimensions of
emotional style:
  1. Self-Awareness    — accuracy of emotional self-perception
  2. Resilience        — speed of recovery from adversity
  3. Attention         — ability to sustain focus
  4. Social Intuition  — reading social cues from others
  5. Sensitivity       — sensitivity to context signals
  6. Outlook           — tendency toward positive vs negative affect

This experiment:
  A. Maps each dimension to a specific parameter of the T_SA operator
  B. Generates synthetic "subjects" with varying dimension profiles
  C. Computes the SA_KL measure and convergence rate for each subject
  D. Shows how dimension interactions affect self-awareness
  E. Demonstrates bifurcation: what parameter values cause SA collapse

Run:
    python experiments/exp03_davidson_mapping.py
    Saves: results/exp03_davidson_mapping.png
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from experiments.sa_operator import (
    SelfAwarenessOperator,
    make_davidson_likelihood,
    make_noisy_identity_likelihood,
    uniform_belief,
    kl_divergence,
)

os.makedirs("results", exist_ok=True)

RNG = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# Parameter Mapping: Davidson Dimension → Operator Property
# ---------------------------------------------------------------------------
#
# Θ = emotional type space (e.g., 5 types: calm, anxious, excited, sad, content)
# O = interoceptive/behavioral observations
#
# D1 Self-Awareness  → SA_KL = min_j D_KL(L[:,θ*] || L[:,θ_j])
#                       parameterized by: noise level in L
#
# D2 Resilience      → Rate of return to q* after a perturbation
#                       = second eigenvalue of Jacobian of T_SA at q*
#                       approximated by: recovery KL after a "shock" obs
#
# D3 Attention       → Effective number of discriminating observations
#                       = rank / entropy of L (how many obs channels are used)
#
# D4 Social Intuition → Accuracy of the agent's model of OTHERS' L matrices
#                       (not directly in T_SA; leave as annotation)
#
# D5 Sensitivity     → Rate of change of SA_KL under context perturbation
#                       = |d(SA_KL)/d(context_param)|
#
# D6 Outlook         → Mean utility E_{q*}[u(θ)]: not about SA per se,
#                       but affects which fixed point the system rests at

N_TYPES = 5
TRUE_TYPE = 0
N_OBS = 10
N_STEPS_SIM = 120

# ---------------------------------------------------------------------------
# A. Mapping: SA measure vs each dimension parameter (univariate)
# ---------------------------------------------------------------------------

# D1: Self-Awareness dimension → noise in L
d1_range = np.linspace(0.01, 0.48, 40)
d1_sa_kl = []
for noise in d1_range:
    L = make_davidson_likelihood(N_TYPES, sa_score=1 - 2 * noise, n_obs=N_OBS, rng=RNG)
    op = SelfAwarenessOperator(L)
    d1_sa_kl.append(op.kl_sa_measure(TRUE_TYPE))
d1_sa_kl = np.array(d1_sa_kl)

# D2: Resilience → perturbation recovery rate
# Measure: after a "shocking" wrong observation, how fast does belief return to q*?
def resilience_measure(op: SelfAwarenessOperator, true_type: int,
                        n_steps_recover: int = 30,
                        rng=None) -> float:
    """
    Perturb the belief with a maximally wrong observation, then let it recover.
    Resilience = 1 / (steps to return within KL < 0.1 of q*).
    """
    if rng is None:
        rng = np.random.default_rng()
    # Start AT the fixed point (near-correct)
    q_star = np.zeros(op.n_types)
    q_star[true_type] = 1.0 - (op.n_types - 1) * 1e-6
    q_star[q_star < 0] = 1e-6
    q_star /= q_star.sum()

    # Apply worst-case perturbation: observe the observation least likely under true type
    p_true = op.L[:, true_type]
    worst_obs = int(np.argmin(p_true))
    q_perturbed = op.update(q_star, worst_obs)

    # Recover by iterating with true-type observations
    sim = op.simulate(q_perturbed, true_type, n_steps_recover, rng=rng)
    kl = sim["kl_to_true"]
    crossings = np.where(kl < 0.1)[0]
    steps = crossings[0] if len(crossings) > 0 else n_steps_recover
    return 1.0 / (steps + 1)

noise_d2 = np.linspace(0.02, 0.48, 25)
d2_resilience = []
for noise in noise_d2:
    L = make_davidson_likelihood(N_TYPES, sa_score=1 - 2 * noise, n_obs=N_OBS, rng=RNG)
    op = SelfAwarenessOperator(L)
    d2_resilience.append(resilience_measure(op, TRUE_TYPE, rng=RNG))
d2_resilience = np.array(d2_resilience)

# D3: Attention → effective rank of observation matrix
def effective_rank(L: np.ndarray) -> float:
    """
    Effective rank of L via entropy of singular value spectrum.
    High effective rank = many independent observation channels = high attention.
    """
    _, s, _ = np.linalg.svd(L, full_matrices=False)
    s = s / s.sum()
    s = s[s > 1e-12]
    return float(np.exp(-np.sum(s * np.log(s))))

d3_n_obs_vals = [2, 3, 5, 8, 12, 20]
d3_eff_rank = []
d3_sa_kl_by_nobs = []
for n_obs in d3_n_obs_vals:
    L = make_davidson_likelihood(N_TYPES, sa_score=0.7, n_obs=n_obs, rng=RNG)
    op = SelfAwarenessOperator(L)
    d3_eff_rank.append(effective_rank(L))
    d3_sa_kl_by_nobs.append(op.kl_sa_measure(TRUE_TYPE))
d3_eff_rank = np.array(d3_eff_rank)
d3_sa_kl_by_nobs = np.array(d3_sa_kl_by_nobs)

# D5: Sensitivity to context → how SA_KL changes when L is perturbed
def sensitivity_measure(op: SelfAwarenessOperator, true_type: int,
                         delta: float = 0.05) -> float:
    """
    Numerical gradient of SA_KL with respect to noise perturbation.
    High sensitivity = SA is fragile; small context change → large SA shift.
    """
    noise_base = 0.20
    L_base = make_noisy_identity_likelihood(op.n_types, noise=noise_base)
    L_plus = make_noisy_identity_likelihood(op.n_types, noise=noise_base + delta)
    L_minus = make_noisy_identity_likelihood(op.n_types, noise=noise_base - delta)
    sa_plus = SelfAwarenessOperator(L_plus).kl_sa_measure(true_type)
    sa_minus = SelfAwarenessOperator(L_minus).kl_sa_measure(true_type)
    return abs(sa_plus - sa_minus) / (2 * delta)

# ---------------------------------------------------------------------------
# B. Synthetic subjects: sample random profiles across 4 key dimensions
# ---------------------------------------------------------------------------

N_SUBJECTS = 50
dim_names = ["D1 Self-Awareness", "D2 Resilience", "D3 Attention", "D6 Outlook"]

subjects = RNG.uniform(0.1, 0.9, size=(N_SUBJECTS, 4))  # random profiles

subject_sa_kl = []
subject_conv_rate = []

for profile in subjects:
    sa_score, resil_score, attn_score, outlook_score = profile
    # SA score → noise level
    noise = 0.5 * (1 - sa_score)
    # Attention score → number of obs channels
    n_obs_subj = max(3, int(attn_score * 15))

    L = make_davidson_likelihood(N_TYPES, sa_score=sa_score, n_obs=n_obs_subj, rng=RNG)
    op = SelfAwarenessOperator(L)
    sa_kl = op.kl_sa_measure(TRUE_TYPE)
    subject_sa_kl.append(sa_kl)

    q0 = uniform_belief(N_TYPES)
    sim = op.simulate(q0, TRUE_TYPE, N_STEPS_SIM, rng=RNG)
    kl = sim["kl_to_true"]
    log_kl = np.log(kl[5:] + 1e-15)
    t_fit = np.arange(len(log_kl))
    mask = np.isfinite(log_kl) & (log_kl > -30)
    if mask.sum() > 10:
        slope, _ = np.polyfit(t_fit[mask], log_kl[mask], 1)
        subject_conv_rate.append(abs(slope))
    else:
        subject_conv_rate.append(0.0)

subjects_sa_kl = np.array(subject_sa_kl)
subjects_conv_rate = np.array(subject_conv_rate)
subject_d1 = subjects[:, 0]  # D1 self-awareness dimension score

# ---------------------------------------------------------------------------
# C. SA bifurcation: noise sweep showing SA collapse
# ---------------------------------------------------------------------------

noise_bif = np.linspace(0.001, 0.499, 200)
sa_bif = []
for noise in noise_bif:
    L = make_noisy_identity_likelihood(N_TYPES, noise=noise)
    op = SelfAwarenessOperator(L)
    sa_bif.append(op.kl_sa_measure(TRUE_TYPE))
sa_bif = np.array(sa_bif)

critical_noise = 1.0 / N_TYPES   # where types become indistinguishable
sa_at_critical = None
for i, n in enumerate(noise_bif):
    if n >= critical_noise:
        sa_at_critical = (n, sa_bif[i])
        break

# ---------------------------------------------------------------------------
# D. Plot
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(17, 13))
fig.suptitle(
    "Experiment 03 — Davidson's Six Dimensions as T_SA Operator Parameters\n"
    "(UW–Madison empirical anchors → mathematical operator properties)",
    fontsize=13, fontweight="bold", y=0.98,
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35,
                       top=0.91, bottom=0.05)

# Panel A: D1 → SA_KL
ax_a = fig.add_subplot(gs[0, 0])
ax_a.plot(1 - 2 * d1_range, d1_sa_kl, color="steelblue", lw=2)
ax_a.set_xlabel("D1 Self-Awareness score", fontsize=10)
ax_a.set_ylabel("SA_KL measure", fontsize=10)
ax_a.set_title("A: D1 (Self-Awareness)\n→ SA_KL (KL separation)", fontsize=10)
ax_a.grid(True, alpha=0.3)

# Panel B: D2 → resilience measure
ax_b = fig.add_subplot(gs[0, 1])
ax_b.plot(1 - 2 * noise_d2, d2_resilience, "o-", color="darkorange", lw=2, ms=4)
ax_b.set_xlabel("D1 score (proxy for D2 base)", fontsize=10)
ax_b.set_ylabel("Resilience (1/recovery steps)", fontsize=10)
ax_b.set_title("B: D2 (Resilience)\n→ perturbation recovery rate", fontsize=10)
ax_b.grid(True, alpha=0.3)

# Panel C: D3 → effective rank / observation channels
ax_c = fig.add_subplot(gs[0, 2])
color_rank = "mediumseagreen"
color_sa = "firebrick"
ax_c2 = ax_c.twinx()
ax_c.bar(range(len(d3_n_obs_vals)), d3_eff_rank,
         color=color_rank, alpha=0.6, label="Effective rank")
ax_c2.plot(range(len(d3_n_obs_vals)), d3_sa_kl_by_nobs, "D-",
           color=color_sa, lw=2, ms=7, label="SA_KL")
ax_c.set_xticks(range(len(d3_n_obs_vals)))
ax_c.set_xticklabels([str(n) for n in d3_n_obs_vals])
ax_c.set_xlabel("D3: #observation channels (n_obs)", fontsize=10)
ax_c.set_ylabel("Effective rank", fontsize=10, color=color_rank)
ax_c2.set_ylabel("SA_KL", fontsize=10, color=color_sa)
ax_c.set_title("C: D3 (Attention)\n→ effective rank of L", fontsize=10)
lines_a, labels_a = ax_c.get_legend_handles_labels()
lines_b, labels_b = ax_c2.get_legend_handles_labels()
ax_c.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8)

# Panel D: Subject scatter D1 × SA_KL
ax_d = fig.add_subplot(gs[1, 0])
sc = ax_d.scatter(subject_d1, subjects_sa_kl,
                  c=subjects_conv_rate, cmap="plasma", s=40, alpha=0.8)
plt.colorbar(sc, ax=ax_d, label="Convergence rate")
ax_d.set_xlabel("D1 Self-Awareness (dimensional score)", fontsize=10)
ax_d.set_ylabel("SA_KL (operator measure)", fontsize=10)
ax_d.set_title(f"D: {N_SUBJECTS} synthetic subjects\nD1 score vs SA_KL", fontsize=10)
# Correlation
r = np.corrcoef(subject_d1, subjects_sa_kl)[0, 1]
ax_d.text(0.05, 0.92, f"r = {r:.3f}", transform=ax_d.transAxes, fontsize=10)
ax_d.grid(True, alpha=0.3)

# Panel E: Subject convergence rate vs D1
ax_e = fig.add_subplot(gs[1, 1])
ax_e.scatter(subject_d1, subjects_conv_rate,
             c=subjects_sa_kl, cmap="viridis", s=40, alpha=0.8)
ax_e.set_xlabel("D1 Self-Awareness score", fontsize=10)
ax_e.set_ylabel("|log-KL slope| (convergence rate)", fontsize=10)
ax_e.set_title("E: Subjects — D1 score\nvs empirical convergence rate", fontsize=10)
r2 = np.corrcoef(subject_d1, subjects_conv_rate)[0, 1]
ax_e.text(0.05, 0.92, f"r = {r2:.3f}", transform=ax_e.transAxes, fontsize=10)
ax_e.grid(True, alpha=0.3)

# Panel F: SA bifurcation
ax_f = fig.add_subplot(gs[1, 2])
ax_f.plot(noise_bif, sa_bif, color="darkviolet", lw=2.5)
if sa_at_critical:
    ax_f.axvline(sa_at_critical[0], color="red", ls="--", lw=1.5,
                 label=f"critical noise = 1/n_types = {critical_noise:.3f}")
    ax_f.axhline(sa_at_critical[1], color="red", ls=":", lw=1)
ax_f.set_xlabel("Observation noise", fontsize=10)
ax_f.set_ylabel("SA_KL", fontsize=10)
ax_f.set_title("F: SA bifurcation\n(noise → 1/n_types → SA collapses)", fontsize=10)
ax_f.legend(fontsize=8)
ax_f.grid(True, alpha=0.3)
ax_f.fill_between(noise_bif,
                  sa_bif,
                  where=noise_bif >= critical_noise,
                  alpha=0.15, color="red",
                  label="SA = 0 region")

# Panel G: Sample belief trajectories for high vs low D1 subjects
ax_g = fig.add_subplot(gs[2, :2])
n_show = 6
idx_high = np.argsort(subject_d1)[-n_show // 2:]
idx_low  = np.argsort(subject_d1)[:n_show // 2]
steps_arr = np.arange(N_STEPS_SIM + 1)

for idx, ls, lbl_sfx in [(idx_high, "-", "High D1"), (idx_low, "--", "Low D1")]:
    for i, subj_idx in enumerate(idx):
        sa_s, _, attn_s, _ = subjects[subj_idx]
        noise_s = 0.5 * (1 - sa_s)
        n_obs_s = max(3, int(attn_s * 15))
        L_s = make_davidson_likelihood(N_TYPES, sa_score=sa_s, n_obs=n_obs_s, rng=RNG)
        op_s = SelfAwarenessOperator(L_s)
        q0 = uniform_belief(N_TYPES)
        sim_s = op_s.simulate(q0, TRUE_TYPE, N_STEPS_SIM, rng=RNG)
        color = "tab:green" if lbl_sfx == "High D1" else "tab:red"
        label = lbl_sfx if i == 0 else None
        ax_g.semilogy(steps_arr, sim_s["kl_to_true"] + 1e-15,
                      color=color, alpha=0.55, lw=1.4, ls=ls, label=label)

ax_g.set_xlabel("Update steps t", fontsize=10)
ax_g.set_ylabel(r"$D_{KL}(\delta_{\theta^*} \| q^t)$ (log)", fontsize=10)
ax_g.set_title(
    "G: Belief convergence — high D1 subjects (green) vs low D1 subjects (red)\n"
    "(each line = one synthetic subject; uniform starting belief)",
    fontsize=10
)
ax_g.legend(fontsize=10)
ax_g.grid(True, which="both", alpha=0.3)
ax_g.set_xlim(0, N_STEPS_SIM)

# Panel H: Davidson mapping summary table
ax_h = fig.add_subplot(gs[2, 2])
ax_h.axis("off")
table_data = [
    ["Dimension", "Operator Property", "Measure"],
    ["D1 Self-Awareness", "Likelihood sharpness", "SA_KL"],
    ["D2 Resilience", "Perturbation recovery", "1/steps_recover"],
    ["D3 Attention", "Effective rank of L", "exp(H(σ))"],
    ["D4 Social Intuition", "Accuracy of L_other", "(future work)"],
    ["D5 Sensitivity", "dSA_KL/d(context)", "gradient"],
    ["D6 Outlook", "E_{q*}[utility(θ)]", "mean payoff"],
]
col_widths = [0.36, 0.38, 0.26]
table = ax_h.table(
    cellText=table_data[1:],
    colLabels=table_data[0],
    cellLoc="left",
    loc="center",
    colWidths=col_widths,
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.55)
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#d0e8ff")
    cell.set_edgecolor("gray")
ax_h.set_title("H: Davidson → T_SA mapping", fontsize=10, pad=12)

plt.savefig("results/exp03_davidson_mapping.png", dpi=150, bbox_inches="tight")
print("Saved: results/exp03_davidson_mapping.png")
plt.show()

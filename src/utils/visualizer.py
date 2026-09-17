"""
Scientific Plotting & Clinical Oncology Visualizer for ChemoRL.
Generates publication-quality RECIST waterfall plots, patient trajectory curves,
CTCAE safety profiles, and CT progression montage images.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def plot_recist_waterfall(results_dict: Dict[str, Any], output_path: str = "results/recist_waterfall.png"):
    """
    Generates a standard clinical RECIST 1.1 Waterfall plot showing percentage
    change in tumor burden per patient, color-coded by RECIST response.
    """
    patient_details = results_dict["patient_details"]
    changes = [r["reduction_pct"] for r in patient_details]
    recist_codes = [r["recist"]["code"] for r in patient_details]

    # Sort descending (PD on left/top, CR on right/bottom)
    sorted_indices = np.argsort(changes)[::-1]
    sorted_changes = [changes[i] for i in sorted_indices]
    sorted_codes = [recist_codes[i] for i in sorted_indices]

    # Palette
    colors = []
    for c in sorted_codes:
        if c == "CR":
            colors.append("#10b981") # Emerald
        elif c == "PR":
            colors.append("#06b6d4") # Cyan
        elif c == "SD":
            colors.append("#f59e0b") # Amber
        else:
            colors.append("#f43f5e") # Rose

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    fig.patch.set_facecolor("#0a0d14")
    ax.set_facecolor("#0f1420")

    bars = ax.bar(range(len(sorted_changes)), sorted_changes, color=colors, width=0.8, alpha=0.9, edgecolor="none")

    # RECIST threshold lines
    ax.axhline(20.0, color="#f43f5e", linestyle="--", linewidth=1.2, label="PD threshold (+20%)")
    ax.axhline(-30.0, color="#06b6d4", linestyle="--", linewidth=1.2, label="PR threshold (-30%)")
    ax.axhline(-95.0, color="#10b981", linestyle=":", linewidth=1.2, label="CR threshold (-95% / Complete)")
    ax.axhline(0.0, color="#64748b", linewidth=0.8)

    ax.set_title(f"RECIST 1.1 Waterfall Plot — {results_dict['policy_name']} (ORR: {results_dict['ORR_pct']:.1f}%)", fontsize=14, color="#f8fafc", fontweight="bold", pad=14)
    ax.set_xlabel("Patients (Rank-Ordered by Response)", fontsize=11, color="#94a3b8")
    ax.set_ylabel("Max Tumor Diameter Change from Baseline (%)", fontsize=11, color="#94a3b8")
    
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # Legend
    custom_lines = [
        plt.Rectangle((0,0),1,1, color="#10b981", label=f"CR (Complete Response) [n={results_dict['CR_count']}]"),
        plt.Rectangle((0,0),1,1, color="#06b6d4", label=f"PR (Partial Response) [n={results_dict['PR_count']}]"),
        plt.Rectangle((0,0),1,1, color="#f59e0b", label=f"SD (Stable Disease) [n={results_dict['SD_count']}]"),
        plt.Rectangle((0,0),1,1, color="#f43f5e", label=f"PD (Progressive Disease) [n={results_dict['PD_count']}]"),
    ]
    ax.legend(handles=custom_lines, loc="lower left", facecolor="#0a0d14", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.25, color="#64748b")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), dpi=300)
    plt.close()
    print(f"[ChemoRL Plot] Saved RECIST Waterfall plot -> {output_path}")


def plot_policy_comparison_bar(benchmark_summaries: List[Dict[str, Any]], output_path: str = "results/policy_comparison.png"):
    """
    Generates side-by-side bar chart comparing ORR, DCR, and CTCAE Severe Toxicity
    across ChemoRL and standard baseline dosing strategies.
    """
    policies = [b["policy_name"] for b in benchmark_summaries]
    orr_scores = [b["ORR_pct"] for b in benchmark_summaries]
    dcr_scores = [b["DCR_pct"] for b in benchmark_summaries]
    tox_scores = [b["Severe_Toxicity_G3_G4_pct"] for b in benchmark_summaries]

    x = np.arange(len(policies))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    fig.patch.set_facecolor("#0a0d14")
    ax.set_facecolor("#0f1420")

    rects1 = ax.bar(x - width, orr_scores, width, label="ORR (Objective Response %)", color="#06b6d4", alpha=0.9, edgecolor="#38bdf8")
    rects2 = ax.bar(x, dcr_scores, width, label="DCR (Disease Control %)", color="#10b981", alpha=0.9, edgecolor="#34d399")
    rects3 = ax.bar(x + width, tox_scores, width, label="CTCAE Grade 3+ Toxicity (%)", color="#f43f5e", alpha=0.9, edgecolor="#fb7185")

    ax.set_ylabel("Cohort Percentage (%)", fontsize=11, color="#94a3b8")
    ax.set_title("Chemotherapy Policy Comparison: Efficacy vs Toxicity Profile", fontsize=14, color="#f8fafc", fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontsize=10, color="#cbd5e1", fontweight="semibold")
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    # Add data values on top of bars
    for rect in rects1 + rects2 + rects3:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color="#f1f5f9", fontweight="medium")

    ax.legend(facecolor="#0a0d14", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=9.5)
    ax.grid(axis="y", linestyle=":", alpha=0.25, color="#64748b")
    ax.set_ylim(0, 115)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), dpi=300)
    plt.close()
    print(f"[ChemoRL Plot] Saved Policy Comparison chart -> {output_path}")


def plot_generative_progression_montage(
    progression_slices: List[np.ndarray],
    cycle_diameters: List[float],
    cycle_doses: List[float],
    output_path: str = "results/generative_progression_ct.png"
):
    """
    Renders a clinical multi-panel montage of synthetic CT slices illustrating
    disease progression/regression across 6 chemotherapy cycles.
    """
    n_cycles = len(progression_slices)
    fig, axes = plt.subplots(1, n_cycles, figsize=(3.2 * n_cycles, 3.8), dpi=300)
    fig.patch.set_facecolor("#0a0d14")

    if n_cycles == 1:
        axes = [axes]

    for idx, (ax, slc) in enumerate(zip(axes, progression_slices)):
        ax.set_facecolor("#0a0d14")
        if isinstance(slc, np.ndarray) and slc.ndim == 3:
            slc_2d = slc[0]
        else:
            slc_2d = slc
        im = ax.imshow(slc_2d, cmap="bone", vmin=0.0, vmax=1.0)
        ax.set_title(f"Cycle {idx}\nDiam: {cycle_diameters[idx]:.1f}mm\nDose: {cycle_doses[idx]:.2f}x",
                     fontsize=10, color="#38bdf8", fontweight="bold", pad=8)
        ax.axis("off")

    fig.suptitle("Generative CT Disease Progression Visualization (ChemoRL Regimen)", fontsize=13, color="#f8fafc", fontweight="bold", y=1.02)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, facecolor=fig.get_facecolor(), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[ChemoRL Plot] Saved Generative CT Progression montage -> {output_path}")

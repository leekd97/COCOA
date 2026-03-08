#!/usr/bin/env python3
"""
PCGrad Ablation — Training Dynamics Analysis

Compares convergence speed and loss stability across gradient methods.

Outputs:
  1. Convergence summary table (best step, loss oscillation)
  2. Loss curve figure (representative cultures)
  3. CBS EMA curve figure

Usage:
    python analysis/pcgrad_dynamics.py \
        --exp_dir experiments \
        --output analysis/ablation
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 11
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False


CULTURES_ORDER = ["ko", "ja", "zh", "ar", "hi", "mr", "ml", "gu", "ur", "vi"]
METHODS_ORDER = ["weighted", "pcgrad", "goal_aware_pcgrad"]
METHOD_DISPLAY = {"weighted": "Weighted", "pcgrad": "PCGrad", "goal_aware_pcgrad": "GA-PCGrad"}
METHOD_SUFFIX = {"weighted": "_weighted", "pcgrad": "_pcgrad", "goal_aware_pcgrad": ""}
MODEL_FOLDER = {"llama": "llama3-8b", "qwen": "qwen3-8b"}

METHOD_COLORS = {
    "weighted": "#94a3b8",
    "pcgrad": "#f97316",
    "goal_aware_pcgrad": "#3b82f6",
}
METHOD_LINESTYLES = {
    "weighted": "--",
    "pcgrad": "-.",
    "goal_aware_pcgrad": "-",
}

BEST_SEEDS = {
    ("ar", "llama"): 45, ("ar", "qwen"): 45,
    ("gu", "llama"): 42, ("gu", "qwen"): 42,
    ("hi", "llama"): 45, ("hi", "qwen"): 45,
    ("ja", "llama"): 45, ("ja", "qwen"): 45,
    ("ko", "llama"): 45, ("ko", "qwen"): 45,
    ("ml", "llama"): 42, ("ml", "qwen"): 42,
    ("mr", "llama"): 42, ("mr", "qwen"): 42,
    ("ur", "llama"): 42, ("ur", "qwen"): 42,
    ("vi", "llama"): 48, ("vi", "qwen"): 48,
    ("zh", "llama"): 42, ("zh", "qwen"): 42,
}


def find_folder(exp_dir, culture, model, seed, method):
    mf = MODEL_FOLDER.get(model, model)
    suffix = METHOD_SUFFIX[method]
    pattern = f"{culture}_cu_{mf}_mse_wg1.0_wn2.0_tau1.0_r16{suffix}_seed{seed}"
    folder = exp_dir / pattern
    return folder if folder.exists() else None


def load_experiment(folder):
    rfile = folder / "results.json"
    if not rfile.exists():
        return None
    with open(rfile) as f:
        return json.load(f)


def compute_oscillation(values):
    """Compute oscillation: mean of absolute step-to-step changes."""
    if len(values) < 2:
        return 0.0
    diffs = [abs(values[i+1] - values[i]) for i in range(len(values)-1)]
    return float(np.mean(diffs))


def analyze_dynamics(exp_dir, cultures, model):
    """Analyze training dynamics for all methods across cultures."""
    results = []

    for culture in cultures:
        seed = BEST_SEEDS.get((culture, model))
        if seed is None:
            continue

        row = {"culture": culture}

        for method in METHODS_ORDER:
            folder = find_folder(exp_dir, culture, model, seed, method)
            if folder is None:
                continue

            data = load_experiment(folder)
            if data is None:
                continue

            history = data.get("history", [])
            best = data.get("best", {})
            final = data.get("final", {})

            if not history:
                continue

            steps = [h["step"] for h in history]
            loss_g = [h["loss_g"] for h in history]
            loss_n = [h["loss_n"] for h in history]
            cbs_g_ema = [h["cbs_g_ema"] for h in history]
            cbs_n_ema = [h["cbs_n_ema"] for h in history]

            row[method] = {
                "best_step": best.get("step", steps[-1]),
                "best_score": best.get("combined_score"),
                "final_score": final.get("combined_score"),
                "loss_g_osc": compute_oscillation(loss_g),
                "loss_n_osc": compute_oscillation(loss_n),
                "total_osc": compute_oscillation(loss_g) + compute_oscillation(loss_n),
                "cbs_g_osc": compute_oscillation(cbs_g_ema),
                "cbs_n_osc": compute_oscillation(cbs_n_ema),
                "steps": steps,
                "loss_g": loss_g,
                "loss_n": loss_n,
                "cbs_g_ema": cbs_g_ema,
                "cbs_n_ema": cbs_n_ema,
            }

        results.append(row)

    return results


def print_dynamics_table(results, model):
    """Print convergence speed and stability table."""
    print(f"\n{'='*95}")
    print(f"  Training Dynamics — {model.upper()}")
    print(f"{'='*95}")
    print(f"  {'':>5} │ {'Weighted':^26} │ {'PCGrad':^26} │ {'GA-PCGrad':^26}")
    print(f"  {'Cult':>5} │ {'BestStep':>8} {'LossOsc':>8} {'CBSOsc':>8} │ {'BestStep':>8} {'LossOsc':>8} {'CBSOsc':>8} │ {'BestStep':>8} {'LossOsc':>8} {'CBSOsc':>8}")
    print(f"  {'─'*90}")

    # Collect for averages
    avgs = {m: {"best_step": [], "total_osc": [], "cbs_osc": []} for m in METHODS_ORDER}

    for row in results:
        culture = row["culture"]
        parts = []
        for method in METHODS_ORDER:
            r = row.get(method)
            if r:
                cbs_osc = r["cbs_g_osc"] + r["cbs_n_osc"]
                parts.append(f" {r['best_step']:>8} {r['total_osc']:>8.3f} {cbs_osc:>8.2f}")
                avgs[method]["best_step"].append(r["best_step"])
                avgs[method]["total_osc"].append(r["total_osc"])
                avgs[method]["cbs_osc"].append(cbs_osc)
            else:
                parts.append(f" {'—':>8} {'—':>8} {'—':>8}")

        print(f"  {culture:>5} │{'│'.join(parts)}")

    # Averages
    print(f"  {'─'*90}")
    avg_parts = []
    for method in METHODS_ORDER:
        a = avgs[method]
        if a["best_step"]:
            avg_parts.append(
                f" {np.mean(a['best_step']):>8.0f} {np.mean(a['total_osc']):>8.3f} {np.mean(a['cbs_osc']):>8.2f}"
            )
        else:
            avg_parts.append(f" {'—':>8} {'—':>8} {'—':>8}")
    print(f"  {'Avg':>5} │{'│'.join(avg_parts)}")

    return avgs


def plot_loss_curves(results, output_path, model, cultures_to_plot=None):
    """Plot loss curves for representative cultures."""
    if cultures_to_plot is None:
        cultures_to_plot = ["ko", "ja", "hi"]

    n = len(cultures_to_plot)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 8), squeeze=False)

    for col, culture in enumerate(cultures_to_plot):
        row_data = next((r for r in results if r["culture"] == culture), None)
        if row_data is None:
            continue

        # Top: loss_g
        ax_g = axes[0, col]
        # Bottom: loss_n
        ax_n = axes[1, col]

        for method in METHODS_ORDER:
            r = row_data.get(method)
            if r is None:
                continue

            color = METHOD_COLORS[method]
            ls = METHOD_LINESTYLES[method]
            label = METHOD_DISPLAY[method]

            ax_g.plot(r["steps"], r["loss_g"], color=color, linestyle=ls,
                      linewidth=1.5, label=label, alpha=0.8)
            ax_n.plot(r["steps"], r["loss_n"], color=color, linestyle=ls,
                      linewidth=1.5, label=label, alpha=0.8)

        ax_g.set_title(f"{culture.upper()}", fontsize=13, fontweight="bold")
        ax_g.set_ylabel("Grounded Loss", fontsize=10)
        ax_g.legend(fontsize=8)
        ax_g.grid(alpha=0.2)

        ax_n.set_xlabel("Step", fontsize=10)
        ax_n.set_ylabel("Neutral Loss", fontsize=10)
        ax_n.legend(fontsize=8)
        ax_n.grid(alpha=0.2)

    plt.suptitle(f"Loss Curves by Gradient Method ({model.upper()})",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_cbs_curves(results, output_path, model, cultures_to_plot=None):
    """Plot CBS EMA curves for representative cultures."""
    if cultures_to_plot is None:
        cultures_to_plot = ["ko", "ja", "hi"]

    n = len(cultures_to_plot)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 8), squeeze=False)

    for col, culture in enumerate(cultures_to_plot):
        row_data = next((r for r in results if r["culture"] == culture), None)
        if row_data is None:
            continue

        ax_g = axes[0, col]
        ax_n = axes[1, col]

        for method in METHODS_ORDER:
            r = row_data.get(method)
            if r is None:
                continue

            color = METHOD_COLORS[method]
            ls = METHOD_LINESTYLES[method]
            label = METHOD_DISPLAY[method]

            ax_g.plot(r["steps"], r["cbs_g_ema"], color=color, linestyle=ls,
                      linewidth=1.5, label=label, alpha=0.8)
            ax_n.plot(r["steps"], r["cbs_n_ema"], color=color, linestyle=ls,
                      linewidth=1.5, label=label, alpha=0.8)

        # Target lines
        ax_g.axhline(y=0, color="red", linestyle=":", linewidth=0.8, alpha=0.5, label="Target (0%)")
        ax_n.axhline(y=50, color="red", linestyle=":", linewidth=0.8, alpha=0.5, label="Target (50%)")

        ax_g.set_title(f"{culture.upper()}", fontsize=13, fontweight="bold")
        ax_g.set_ylabel("CBS_g EMA (%)", fontsize=10)
        ax_g.legend(fontsize=8)
        ax_g.grid(alpha=0.2)

        ax_n.set_xlabel("Step", fontsize=10)
        ax_n.set_ylabel("CBS_n EMA (%)", fontsize=10)
        ax_n.legend(fontsize=8)
        ax_n.grid(alpha=0.2)

    plt.suptitle(f"CBS EMA Curves by Gradient Method ({model.upper()})",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_convergence_comparison(results, output_path, model):
    """Bar chart: average best step and oscillation per method."""
    avgs = {m: {"best_step": [], "total_osc": []} for m in METHODS_ORDER}

    for row in results:
        for method in METHODS_ORDER:
            r = row.get(method)
            if r:
                avgs[method]["best_step"].append(r["best_step"])
                avgs[method]["total_osc"].append(r["total_osc"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    methods = [METHOD_DISPLAY[m] for m in METHODS_ORDER]
    colors = [METHOD_COLORS[m] for m in METHODS_ORDER]

    # Left: Average best step (lower = faster convergence)
    mean_steps = [np.mean(avgs[m]["best_step"]) for m in METHODS_ORDER]
    std_steps = [np.std(avgs[m]["best_step"]) for m in METHODS_ORDER]
    bars1 = ax1.bar(methods, mean_steps, color=colors, edgecolor="white", linewidth=0.8)
    ax1.errorbar(range(len(methods)), mean_steps, yerr=std_steps,
                 fmt="none", ecolor="black", capsize=5, linewidth=1)
    for bar, val in zip(bars1, mean_steps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 f"{val:.0f}", ha="center", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Average Best Step", fontsize=11)
    ax1.set_title("Convergence Speed\n(lower = faster)", fontsize=12, fontweight="bold")

    # Right: Average loss oscillation (lower = more stable)
    mean_osc = [np.mean(avgs[m]["total_osc"]) for m in METHODS_ORDER]
    std_osc = [np.std(avgs[m]["total_osc"]) for m in METHODS_ORDER]
    bars2 = ax2.bar(methods, mean_osc, color=colors, edgecolor="white", linewidth=0.8)
    ax2.errorbar(range(len(methods)), mean_osc, yerr=std_osc,
                 fmt="none", ecolor="black", capsize=5, linewidth=1)
    for bar, val in zip(bars2, mean_osc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Average Loss Oscillation", fontsize=11)
    ax2.set_title("Training Stability\n(lower = more stable)", fontsize=12, fontweight="bold")

    plt.suptitle(f"Gradient Method Comparison ({model.upper()})",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", default="experiments")
    p.add_argument("--output", default="analysis/ablation")
    p.add_argument("--model", default="llama")
    p.add_argument("--plot_cultures", nargs="*", default=["ko", "ja", "hi"])
    args = p.parse_args()

    exp_dir = Path(args.exp_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = args.model

    print(f"Analyzing training dynamics for {model}...")
    results = analyze_dynamics(exp_dir, CULTURES_ORDER, model)

    # Table
    avgs = print_dynamics_table(results, model)

    # Figures
    tag = f"_{model}"

    print(f"\nPlotting loss curves...")
    plot_loss_curves(results, output_dir / f"fig_loss_curves{tag}.png", model, args.plot_cultures)
    plot_loss_curves(results, output_dir / f"fig_loss_curves{tag}.pdf", model, args.plot_cultures)

    print(f"Plotting CBS EMA curves...")
    plot_cbs_curves(results, output_dir / f"fig_cbs_curves{tag}.png", model, args.plot_cultures)
    plot_cbs_curves(results, output_dir / f"fig_cbs_curves{tag}.pdf", model, args.plot_cultures)

    print(f"Plotting convergence comparison...")
    plot_convergence_comparison(results, output_dir / f"fig_convergence{tag}.png", model)
    plot_convergence_comparison(results, output_dir / f"fig_convergence{tag}.pdf", model)

    # Save raw data
    save_data = []
    for row in results:
        save_row = {"culture": row["culture"]}
        for method in METHODS_ORDER:
            r = row.get(method)
            if r:
                save_row[method] = {
                    "best_step": r["best_step"],
                    "best_score": r["best_score"],
                    "final_score": r["final_score"],
                    "loss_g_osc": r["loss_g_osc"],
                    "loss_n_osc": r["loss_n_osc"],
                    "total_osc": r["total_osc"],
                    "cbs_g_osc": r["cbs_g_osc"],
                    "cbs_n_osc": r["cbs_n_osc"],
                }
        save_data.append(save_row)

    with open(output_dir / f"dynamics_{model}.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {output_dir}/dynamics_{model}.json")


if __name__ == "__main__":
    main()
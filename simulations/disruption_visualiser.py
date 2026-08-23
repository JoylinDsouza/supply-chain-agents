# simulations/disruption_visualiser.py
# Visualisation for Simulation 3: Supplier Disruption and Resilience

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulations.supplier_disruption_sim import (
    run_strategy_comparison,
    run_supplier_disruption_sim,
    STRATEGIES
)

os.makedirs("results", exist_ok=True)

# ── Colours per strategy ──────────────────────────────────────────────────────
COLOURS = {
    "no_backup":     "#D63A2E",
    "safety_stock":  "#7B3FD6",
    "dual_sourcing": "#2E9E5A",
    "air_freight":   "#2E6FD6"
}
LABELS = {
    "no_backup":    "No backup",
    "safety_stock": "Safety stock",
    "dual_sourcing":"Dual sourcing",
    "air_freight":  "Air freight"
}
PROBS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
PROB_LABELS = [f"{int(p*100)}%" for p in PROBS]


def plot_all():
    print("Running strategy comparison...")
    results = run_strategy_comparison(
        disruption_probabilities=PROBS,
        disruption_duration_days=42,
        daily_demand=200,
        trials=50,
        save_csv=True
    )

    # Organise by strategy
    by_strategy = {s: {"prob":[], "total":[], "service":[], "shortage":[], "strategy_cost":[]}
                   for s in STRATEGIES}
    for r in results:
        s = r["strategy"]
        by_strategy[s]["prob"].append(r["disruption_probability_pct"])
        by_strategy[s]["total"].append(r["avg_total_cost_gbp"])
        by_strategy[s]["service"].append(r["avg_service_level_pct"])
        by_strategy[s]["shortage"].append(r["avg_shortage_cost_gbp"])
        by_strategy[s]["strategy_cost"].append(r["avg_strategy_cost_gbp"])

    prob_x = [p * 100 for p in PROBS]

    # ── FIGURE 1: Four-panel strategy comparison ──────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig1.suptitle(
        "Simulation 3: Supplier Disruption — Strategy Comparison\n"
        "(Monte Carlo, 50 trials per scenario, 365-day horizon, 42-day avg disruption)",
        fontsize=13, fontweight="bold", y=1.01
    )

    # Panel 1: Total cost vs disruption probability
    ax = axes[0, 0]
    for s in STRATEGIES:
        ax.plot(prob_x, by_strategy[s]["total"],
                marker="o", linewidth=2.5, markersize=7,
                color=COLOURS[s], label=LABELS[s])
    ax.set_xlabel("Annual Disruption Probability (%)", fontsize=11)
    ax.set_ylabel("Average Annual Total Cost (£)", fontsize=11)
    ax.set_title("Total Cost vs Disruption Probability", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(prob_x)
    ax.set_xticklabels(PROB_LABELS)

    # Panel 2: Service level vs disruption probability
    ax = axes[0, 1]
    for s in STRATEGIES:
        ax.plot(prob_x, by_strategy[s]["service"],
                marker="s", linewidth=2.5, markersize=7,
                color=COLOURS[s], label=LABELS[s])
    ax.axhline(y=95, color="black", linestyle="--",
               linewidth=1.5, alpha=0.6, label="95% target")
    ax.set_xlabel("Annual Disruption Probability (%)", fontsize=11)
    ax.set_ylabel("Average Service Level (%)", fontsize=11)
    ax.set_title("Service Level vs Disruption Probability", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(prob_x)
    ax.set_xticklabels(PROB_LABELS)
    ax.set_ylim(75, 102)

    # Panel 3: Cost breakdown at 30% disruption probability
    ax = axes[1, 0]
    target_idx = PROBS.index(0.30)
    strategy_names = [LABELS[s] for s in STRATEGIES]
    holding_vals  = [by_strategy[s]["total"][target_idx]
                     - by_strategy[s]["shortage"][target_idx]
                     - by_strategy[s]["strategy_cost"][target_idx]
                     for s in STRATEGIES]
    shortage_vals = [by_strategy[s]["shortage"][target_idx]  for s in STRATEGIES]
    strategy_vals = [by_strategy[s]["strategy_cost"][target_idx] for s in STRATEGIES]

    x = np.arange(len(STRATEGIES))
    width = 0.5
    bars1 = ax.bar(x, holding_vals,  width, label="Holding cost",   color="#4A90D9", alpha=0.85)
    bars2 = ax.bar(x, shortage_vals, width, label="Shortage cost",  color="#D63A2E", alpha=0.85,
                   bottom=holding_vals)
    bars3 = ax.bar(x, strategy_vals, width, label="Strategy cost",  color="#2E9E5A", alpha=0.85,
                   bottom=[h+s for h,s in zip(holding_vals, shortage_vals)])
    ax.set_xticks(x)
    ax.set_xticklabels(strategy_names, fontsize=10)
    ax.set_ylabel("Annual Cost (£)", fontsize=11)
    ax.set_title("Cost Breakdown at 30% Disruption Probability", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: Strategy cost vs disruption probability
    ax = axes[1, 1]
    for s in ["dual_sourcing", "air_freight", "safety_stock"]:
        ax.plot(prob_x, by_strategy[s]["strategy_cost"],
                marker="D", linewidth=2.5, markersize=7,
                color=COLOURS[s], label=LABELS[s])
    ax.set_xlabel("Annual Disruption Probability (%)", fontsize=11)
    ax.set_ylabel("Strategy-specific Cost (£)", fontsize=11)
    ax.set_title("Cost of Each Backup Strategy vs Disruption Probability\n"
                 "(No backup excluded — zero strategy cost)",
                 fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(prob_x)
    ax.set_xticklabels(PROB_LABELS)

    plt.tight_layout()
    path1 = "results/sim3_strategy_comparison.png"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"Saved: {path1}")

    # ── FIGURE 2: Disruption timeline trace + crossover analysis ─────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle(
        "Simulation 3: Disruption Impact Detail\n"
        "(Left: cost crossover point | Right: service level degradation)",
        fontsize=12, fontweight="bold"
    )

    # Left: find crossover — where does dual sourcing beat air freight?
    ax1.plot(prob_x, by_strategy["no_backup"]["total"],
             color=COLOURS["no_backup"],    linewidth=2.5, marker="o", markersize=7, label="No backup")
    ax1.plot(prob_x, by_strategy["dual_sourcing"]["total"],
             color=COLOURS["dual_sourcing"], linewidth=2.5, marker="s", markersize=7, label="Dual sourcing")
    ax1.plot(prob_x, by_strategy["air_freight"]["total"],
             color=COLOURS["air_freight"],  linewidth=2.5, marker="^", markersize=7, label="Air freight")

    # Find crossover point between air freight and dual sourcing
    af_costs = by_strategy["air_freight"]["total"]
    ds_costs = by_strategy["dual_sourcing"]["total"]
    for i in range(len(af_costs)-1):
        if (af_costs[i] <= ds_costs[i]) and (af_costs[i+1] > ds_costs[i+1]):
            crossover_prob = (prob_x[i] + prob_x[i+1]) / 2
            ax1.axvline(x=crossover_prob, color="black", linestyle=":",
                        linewidth=2, alpha=0.7,
                        label=f"Crossover ~{crossover_prob:.0f}%")
            break

    ax1.set_xlabel("Annual Disruption Probability (%)", fontsize=11)
    ax1.set_ylabel("Average Annual Total Cost (£)", fontsize=11)
    ax1.set_title("Cost Crossover: When Does Dual Sourcing Beat Air Freight?",
                  fontsize=11, fontweight="bold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(prob_x)
    ax1.set_xticklabels(PROB_LABELS)

    # Right: service level degradation — how bad does no_backup get?
    ax2.fill_between(prob_x, by_strategy["no_backup"]["service"],
                     95, where=[s < 95 for s in by_strategy["no_backup"]["service"]],
                     alpha=0.2, color=COLOURS["no_backup"], label="Service level deficit")
    ax2.plot(prob_x, by_strategy["no_backup"]["service"],
             color=COLOURS["no_backup"],    linewidth=2.5, marker="o", markersize=7, label="No backup")
    ax2.plot(prob_x, by_strategy["dual_sourcing"]["service"],
             color=COLOURS["dual_sourcing"], linewidth=2.5, marker="s", markersize=7, label="Dual sourcing")
    ax2.plot(prob_x, by_strategy["air_freight"]["service"],
             color=COLOURS["air_freight"],  linewidth=2.5, marker="^", markersize=7, label="Air freight")
    ax2.axhline(y=95, color="black", linestyle="--", linewidth=1.5,
                alpha=0.6, label="95% service target")
    ax2.set_xlabel("Annual Disruption Probability (%)", fontsize=11)
    ax2.set_ylabel("Average Service Level (%)", fontsize=11)
    ax2.set_title("Service Level Degradation Without a Backup Strategy",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(prob_x)
    ax2.set_xticklabels(PROB_LABELS)
    ax2.set_ylim(78, 102)

    plt.tight_layout()
    path2 = "results/sim3_disruption_detail.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {path2}")

    plt.show()
    return path1, path2


if __name__ == "__main__":
    print("=" * 65)
    print("Simulation 3 Visualiser — generating all graphs")
    print("=" * 65)
    plot_all()
    print("\nAll done. Check results/ folder for:")
    print("  sim3_strategy_comparison.png — four-panel strategy analysis")
    print("  sim3_disruption_detail.png   — crossover and service degradation")
    print("  simulation3_results.csv      — full results table")
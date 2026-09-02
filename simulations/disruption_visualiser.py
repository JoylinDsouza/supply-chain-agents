# simulations/disruption_visualiser.py
# Visualisation for Simulation 3: Supplier Disruption and Resilience
#
# Uses the canonical Monte Carlo configuration from supplier_disruption_sim.py:
#   Trials = 1000
#   Seed   = 42
#
# Produces:
#   1. Four-panel strategy comparison
#   2. Cost comparison and service-level resilience analysis

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulations.supplier_disruption_sim import (
    run_strategy_comparison,
    STRATEGIES
)

os.makedirs("results", exist_ok=True)


# ── Canonical experiment configuration ────────────────────────────────────────
TRIALS = 1000
SEED = 42
SIMULATION_DAYS = 365
DISRUPTION_DURATION_DAYS = 42
DAILY_DEMAND = 200

# Annual disruption probabilities tested
PROBS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
PROB_LABELS = [f"{int(p * 100)}%" for p in PROBS]


# ── Colours per strategy ──────────────────────────────────────────────────────
COLOURS = {
    "no_backup": "#D63A2E",
    "safety_stock": "#7B3FD6",
    "dual_sourcing": "#2E9E5A",
    "air_freight": "#2E6FD6"
}

LABELS = {
    "no_backup": "No backup",
    "safety_stock": "Safety stock",
    "dual_sourcing": "Dual sourcing",
    "air_freight": "Air freight"
}


def plot_all():
    print("Running Simulation 3 visualisation...")
    print()
    print("Canonical Monte Carlo configuration:")
    print(f"  Trials: {TRIALS}")
    print(f"  Seed:   {SEED}")
    print(f"  Horizon: {SIMULATION_DAYS} days")
    print(f"  Average disruption duration: {DISRUPTION_DURATION_DAYS} days")
    print()

    # The simulation module controls the random seed internally.
    # The visualiser therefore uses the canonical experiment configuration.
    results = run_strategy_comparison(
        disruption_probabilities=PROBS,
        disruption_duration_days=DISRUPTION_DURATION_DAYS,
        daily_demand=DAILY_DEMAND,
        trials=TRIALS,
        save_csv=True,
        csv_path="results/simulation3_results.csv"
    )

    # ── Organise results by strategy ──────────────────────────────────────────
    by_strategy = {
        s: {
            "prob": [],
            "total": [],
            "service": [],
            "procurement": [],
            "shortage": [],
            "holding": [],
            "strategy_cost": []
        }
        for s in STRATEGIES
    }

    for r in results:
        s = r["strategy"]

        by_strategy[s]["prob"].append(
            r["disruption_probability_pct"]
        )

        by_strategy[s]["total"].append(
            r["avg_total_cost_gbp"]
        )

        by_strategy[s]["service"].append(
            r["avg_service_level_pct"]
        )

        # Procurement cost is now explicitly included in the corrected
        # Simulation 3 accounting model.
        by_strategy[s]["procurement"].append(
            r["avg_procurement_cost_gbp"]
        )

        by_strategy[s]["shortage"].append(
            r["avg_shortage_cost_gbp"]
        )

        by_strategy[s]["holding"].append(
            r["avg_holding_cost_gbp"]
        )

        by_strategy[s]["strategy_cost"].append(
            r["avg_strategy_cost_gbp"]
        )

    prob_x = [p * 100 for p in PROBS]

    # ── FIGURE 1: Four-panel strategy comparison ──────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(14, 9))

    fig1.suptitle(
        "Simulation 3: Supplier Disruption — Strategy Comparison\n"
        f"(Monte Carlo, {TRIALS:,} trials per scenario, "
        f"{SIMULATION_DAYS}-day horizon, "
        f"{DISRUPTION_DURATION_DAYS}-day average disruption)",
        fontsize=13,
        fontweight="bold",
        y=1.01
    )

    # ── Panel 1: Total cost vs disruption probability ────────────────────────
    ax = axes[0, 0]

    for s in STRATEGIES:
        ax.plot(
            prob_x,
            by_strategy[s]["total"],
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=COLOURS[s],
            label=LABELS[s]
        )

    ax.set_xlabel(
        "Annual Disruption Probability (%)",
        fontsize=11
    )

    ax.set_ylabel(
        "Average Total Cost (£)",
        fontsize=11
    )

    ax.set_title(
        "Total Cost vs Disruption Probability",
        fontsize=11,
        fontweight="bold"
    )

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"£{x:,.0f}")
    )

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.set_xticks(prob_x)
    ax.set_xticklabels(PROB_LABELS)

    # ── Panel 2: Service level vs disruption probability ──────────────────────
    ax = axes[0, 1]

    for s in STRATEGIES:
        ax.plot(
            prob_x,
            by_strategy[s]["service"],
            marker="s",
            linewidth=2.5,
            markersize=7,
            color=COLOURS[s],
            label=LABELS[s]
        )

    ax.axhline(
        y=95,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label="95% target"
    )

    ax.set_xlabel(
        "Annual Disruption Probability (%)",
        fontsize=11
    )

    ax.set_ylabel(
        "Average Service Level (%)",
        fontsize=11
    )

    ax.set_title(
        "Service Level vs Disruption Probability",
        fontsize=11,
        fontweight="bold"
    )

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.set_xticks(prob_x)
    ax.set_xticklabels(PROB_LABELS)

    ax.set_ylim(90, 102)

    # ── Panel 3: Cost breakdown at 30% disruption probability ────────────────
    ax = axes[1, 0]

    target_idx = PROBS.index(0.30)

    strategy_names = [
        LABELS[s]
        for s in STRATEGIES
    ]

    total_vals = [
        by_strategy[s]["total"][target_idx]
        for s in STRATEGIES
    ]

    procurement_vals = [
        by_strategy[s]["procurement"][target_idx]
        for s in STRATEGIES
    ]

    holding_vals = [
        by_strategy[s]["holding"][target_idx]
        for s in STRATEGIES
    ]

    shortage_vals = [
        by_strategy[s]["shortage"][target_idx]
        for s in STRATEGIES
    ]

    strategy_vals = [
        by_strategy[s]["strategy_cost"][target_idx]
        for s in STRATEGIES
    ]

    x = np.arange(len(STRATEGIES))
    width = 0.55

    # Procurement cost
    ax.bar(
        x,
        procurement_vals,
        width,
        label="Procurement cost",
        color="#5B8DB8",
        alpha=0.85
    )

    # Holding cost
    ax.bar(
        x,
        holding_vals,
        width,
        label="Holding cost",
        color="#4A90D9",
        alpha=0.85,
        bottom=procurement_vals
    )

    # Shortage cost
    procurement_holding = [
        p + h
        for p, h in zip(procurement_vals, holding_vals)
    ]

    ax.bar(
        x,
        shortage_vals,
        width,
        label="Shortage cost",
        color="#D63A2E",
        alpha=0.85,
        bottom=procurement_holding
    )

    # Strategy-specific cost
    procurement_holding_shortage = [
        p + h + sh
        for p, h, sh in zip(
            procurement_vals,
            holding_vals,
            shortage_vals
        )
    ]

    ax.bar(
        x,
        strategy_vals,
        width,
        label="Strategy cost",
        color="#2E9E5A",
        alpha=0.85,
        bottom=procurement_holding_shortage
    )

    # Check that the stacked components reconcile with total cost.
    # This is deliberately a warning rather than a hard failure so that
    # small floating-point differences do not stop graph generation.
    stacked_totals = [
        p + h + sh + st
        for p, h, sh, st in zip(
            procurement_vals,
            holding_vals,
            shortage_vals,
            strategy_vals
        )
    ]

    for strategy_name, stacked, reported in zip(
        strategy_names,
        stacked_totals,
        total_vals
    ):
        if not np.isclose(stacked, reported, rtol=1e-5, atol=1.0):
            print(
                f"WARNING: Cost breakdown does not reconcile for "
                f"{strategy_name}: "
                f"stacked=£{stacked:,.2f}, "
                f"reported total=£{reported:,.2f}"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        strategy_names,
        fontsize=10
    )

    ax.set_ylabel(
        "Average Cost (£)",
        fontsize=11
    )

    ax.set_title(
        "Cost Breakdown at 30% Disruption Probability",
        fontsize=11,
        fontweight="bold"
    )

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"£{x:,.0f}")
    )

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 4: Strategy-specific cost ───────────────────────────────────────
    ax = axes[1, 1]

    for s in [
        "dual_sourcing",
        "air_freight",
        "safety_stock"
    ]:
        ax.plot(
            prob_x,
            by_strategy[s]["strategy_cost"],
            marker="D",
            linewidth=2.5,
            markersize=7,
            color=COLOURS[s],
            label=LABELS[s]
        )

    ax.set_xlabel(
        "Annual Disruption Probability (%)",
        fontsize=11
    )

    ax.set_ylabel(
        "Average Strategy-specific Cost (£)",
        fontsize=11
    )

    ax.set_title(
        "Cost of Backup Strategies vs Disruption Probability",
        fontsize=11,
        fontweight="bold"
    )

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"£{x:,.0f}")
    )

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.set_xticks(prob_x)
    ax.set_xticklabels(PROB_LABELS)

    plt.tight_layout()

    path1 = "results/sim3_strategy_comparison.png"

    fig1.savefig(
        path1,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"Saved: {path1}")

    # ── FIGURE 2: Cost and service resilience analysis ────────────────────────
    fig2, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    fig2.suptitle(
        "Simulation 3: Resilience Impact Analysis\n"
        "(Monte Carlo strategy comparison)",
        fontsize=12,
        fontweight="bold"
    )

    # ── Left: cost comparison ─────────────────────────────────────────────────
    ax1.plot(
        prob_x,
        by_strategy["no_backup"]["total"],
        color=COLOURS["no_backup"],
        linewidth=2.5,
        marker="o",
        markersize=7,
        label="No backup"
    )

    ax1.plot(
        prob_x,
        by_strategy["dual_sourcing"]["total"],
        color=COLOURS["dual_sourcing"],
        linewidth=2.5,
        marker="s",
        markersize=7,
        label="Dual sourcing"
    )

    ax1.plot(
        prob_x,
        by_strategy["air_freight"]["total"],
        color=COLOURS["air_freight"],
        linewidth=2.5,
        marker="^",
        markersize=7,
        label="Air freight"
    )

    # Highlight the 20% baseline scenario used in the dissertation.
    ax1.axvline(
        x=20,
        color="black",
        linestyle=":",
        linewidth=1.5,
        alpha=0.6
    )

    ax1.text(
        20.5,
        ax1.get_ylim()[0],
        "20% baseline",
        rotation=90,
        va="bottom",
        fontsize=9
    )

    ax1.set_xlabel(
        "Annual Disruption Probability (%)",
        fontsize=11
    )

    ax1.set_ylabel(
        "Average Total Cost (£)",
        fontsize=11
    )

    ax1.set_title(
        "Cost Comparison Across Resilience Strategies",
        fontsize=11,
        fontweight="bold"
    )

    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"£{x:,.0f}")
    )

    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax1.set_xticks(prob_x)
    ax1.set_xticklabels(PROB_LABELS)

    # ── Right: service degradation ────────────────────────────────────────────
    ax2.plot(
        prob_x,
        by_strategy["no_backup"]["service"],
        color=COLOURS["no_backup"],
        linewidth=2.5,
        marker="o",
        markersize=7,
        label="No backup"
    )

    ax2.plot(
        prob_x,
        by_strategy["dual_sourcing"]["service"],
        color=COLOURS["dual_sourcing"],
        linewidth=2.5,
        marker="s",
        markersize=7,
        label="Dual sourcing"
    )

    ax2.plot(
        prob_x,
        by_strategy["air_freight"]["service"],
        color=COLOURS["air_freight"],
        linewidth=2.5,
        marker="^",
        markersize=7,
        label="Air freight"
    )

    ax2.plot(
        prob_x,
        by_strategy["safety_stock"]["service"],
        color=COLOURS["safety_stock"],
        linewidth=2.5,
        marker="D",
        markersize=7,
        label="Safety stock"
    )

    ax2.axhline(
        y=95,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label="95% service target"
    )

    ax2.set_xlabel(
        "Annual Disruption Probability (%)",
        fontsize=11
    )

    ax2.set_ylabel(
        "Average Service Level (%)",
        fontsize=11
    )

    ax2.set_title(
        "Service-Level Resilience Under Increasing Disruption",
        fontsize=11,
        fontweight="bold"
    )

    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax2.set_xticks(prob_x)
    ax2.set_xticklabels(PROB_LABELS)

    ax2.set_ylim(90, 102)

    plt.tight_layout()

    path2 = "results/sim3_disruption_detail.png"

    fig2.savefig(
        path2,
        dpi=150,
        bbox_inches="tight"
    )

    print(f"Saved: {path2}")

    plt.show()

    return path1, path2


if __name__ == "__main__":
    print("=" * 70)
    print("Simulation 3 Visualiser — generating all graphs")
    print("=" * 70)

    plot_all()

    print()
    print("All done. Check your results/ folder for:")
    print("  sim3_strategy_comparison.png")
    print("  sim3_disruption_detail.png")
    print("  simulation3_results.csv")
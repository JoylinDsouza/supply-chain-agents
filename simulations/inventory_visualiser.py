# simulations/inventory_visualiser.py
# Visualisation for Simulation 1: Inventory Replenishment
# Produces 5 publication-ready graphs.

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import sys

# Allow importing from parent folder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulations.inventory_sim import run_experiment, run_inventory_sim

os.makedirs("results", exist_ok=True)

# ── COLOUR PALETTE ────────────────────────────────────────────────────────────
C_BLUE    = "#2E6FD6"
C_ORANGE  = "#E07B2A"
C_RED     = "#D63A2E"
C_GREEN   = "#2E9E5A"
C_PURPLE  = "#7B3FD6"
C_GREY    = "#888888"
TARGET_SL = 95   # industry standard service level target


def plot_all(results, daily_trace, reorder_points, title_suffix=""):
    """
    Produces all five graphs for Simulation 1.
    """

    rp_values       = [r["reorder_point"] for r in results]
    service_levels  = [r["avg_service_level_pct"] for r in results]
    total_costs     = [r["avg_total_cost_gbp"] for r in results]
    holding_costs   = [r["avg_holding_cost_gbp"] for r in results]
    ordering_costs  = [r["avg_ordering_cost_gbp"] for r in results]
    shortage_costs  = [r["avg_shortage_cost_gbp"] for r in results]
    num_orders      = [r["avg_num_orders"] for r in results]

    # ── FIGURE 1: Four-panel policy comparison ────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig1.suptitle(
        f"Simulation 1: Inventory Replenishment — Policy Comparison\n"
        f"(Monte Carlo, 20 trials each, 90-day horizon with seasonal spike{title_suffix})",
        fontsize=13, fontweight="bold", y=1.01
    )

    # Panel 1: Service Level
    ax = axes[0, 0]
    ax.plot(rp_values, service_levels, marker="o", color=C_BLUE,
            linewidth=2.5, markersize=8, label="Service level")
    ax.axhline(y=TARGET_SL, color=C_RED, linestyle="--",
               linewidth=1.5, alpha=0.8, label=f"{TARGET_SL}% target")
    ax.fill_between(rp_values, service_levels, TARGET_SL,
                    where=[s < TARGET_SL for s in service_levels],
                    alpha=0.12, color=C_RED, label="Below target")
    ax.set_xlabel("Reorder Point (units)", fontsize=11)
    ax.set_ylabel("Average Service Level (%)", fontsize=11)
    ax.set_title("Service Level vs Reorder Point", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    # Annotate each point
    for rp, sl in zip(rp_values, service_levels):
        ax.annotate(f"{sl}%", (rp, sl), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, color=C_BLUE)

    # Panel 2: Total Cost (the key graph)
    ax = axes[0, 1]
    ax.plot(rp_values, total_costs, marker="D", color=C_PURPLE,
            linewidth=2.5, markersize=8, label="Total cost")
    min_cost = min(total_costs)
    min_rp   = rp_values[total_costs.index(min_cost)]
    ax.axvline(x=min_rp, color=C_GREEN, linestyle="--",
               linewidth=1.5, alpha=0.8, label=f"Optimal ROP = {min_rp}")
    ax.set_xlabel("Reorder Point (units)", fontsize=11)
    ax.set_ylabel("Average Total Cost (£)", fontsize=11)
    ax.set_title("Total Cost vs Reorder Point", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"£{x:,.0f}")
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    for rp, tc in zip(rp_values, total_costs):
        ax.annotate(f"£{tc:,.0f}", (rp, tc), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color=C_PURPLE)

    # Panel 3: Cost breakdown stacked bar
    ax = axes[1, 0]
    width = 150
    bars1 = ax.bar(rp_values, holding_costs,  width, label="Holding",  color=C_BLUE,   alpha=0.85)
    bars2 = ax.bar(rp_values, ordering_costs, width, label="Ordering", color=C_ORANGE, alpha=0.85,
                   bottom=holding_costs)
    shortage_bottoms = [h + o for h, o in zip(holding_costs, ordering_costs)]
    bars3 = ax.bar(rp_values, shortage_costs, width, label="Shortage", color=C_RED,    alpha=0.85,
                   bottom=shortage_bottoms)
    ax.set_xlabel("Reorder Point (units)", fontsize=11)
    ax.set_ylabel("Cost (£)", fontsize=11)
    ax.set_title("Cost Breakdown by Component", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: Number of orders placed
    ax = axes[1, 1]
    ax.bar(rp_values, num_orders, width=150, color=C_GREEN, alpha=0.85, label="Orders placed")
    ax.set_xlabel("Reorder Point (units)", fontsize=11)
    ax.set_ylabel("Average Orders Placed (90 days)", fontsize=11)
    ax.set_title("Number of Orders vs Reorder Point", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for rp, n in zip(rp_values, num_orders):
        ax.text(rp, n + 0.1, f"{n:.1f}", ha="center", fontsize=9, color=C_GREEN)

    plt.tight_layout()
    path1 = "results/sim1_policy_comparison.png"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"Saved: {path1}")

    # ── FIGURE 2: Daily inventory trace ──────────────────────────────────────
    fig2, (ax_inv, ax_demand) = plt.subplots(2, 1, figsize=(14, 8),
                                              sharex=True, height_ratios=[2, 1])
    fig2.suptitle(
        "Simulation 1: Daily Inventory Trace (Optimal Policy — ROP=1000, OQ=1500)\n"
        "One representative trial showing the sawtooth replenishment pattern",
        fontsize=12, fontweight="bold"
    )

    days      = [d["day"] for d in daily_trace]
    inventory = [d["inventory"] for d in daily_trace]
    demand    = [d["demand"] for d in daily_trace]
    weeks     = [d["week"] for d in daily_trace]

    # Shade the seasonal spike weeks
    spike_days = [d["day"] for d in daily_trace if d["week"] in (6, 7)]
    if spike_days:
        ax_inv.axvspan(min(spike_days), max(spike_days),
                       alpha=0.12, color=C_ORANGE, label="Seasonal spike (weeks 6-7)")
        ax_demand.axvspan(min(spike_days), max(spike_days),
                          alpha=0.12, color=C_ORANGE)

    # Inventory line
    ax_inv.plot(days, inventory, color=C_BLUE, linewidth=1.5, label="Inventory level")
    ax_inv.axhline(y=1000, color=C_RED, linestyle="--", linewidth=1.2,
                   alpha=0.7, label="Reorder point (1000 units)")
    ax_inv.fill_between(days, inventory, alpha=0.15, color=C_BLUE)
    ax_inv.set_ylabel("Inventory (units)", fontsize=11)
    ax_inv.set_title("Inventory Level Over 90 Days", fontsize=11)
    ax_inv.legend(fontsize=9, loc="upper right")
    ax_inv.grid(True, alpha=0.3)
    ax_inv.set_ylim(bottom=0)

    # Demand bars
    spike_mask  = [True if w in (6, 7) else False for w in weeks]
    normal_d    = [d if not s else 0 for d, s in zip(demand, spike_mask)]
    spike_d     = [d if s else 0 for d, s in zip(demand, spike_mask)]
    ax_demand.bar(days, normal_d, color=C_BLUE,   alpha=0.7, label="Normal demand", width=1)
    ax_demand.bar(days, spike_d,  color=C_ORANGE, alpha=0.8, label="Spike demand",  width=1)
    ax_demand.axhline(y=200, color=C_GREY, linestyle="--",
                      linewidth=1, alpha=0.6, label="Mean demand (200)")
    ax_demand.set_xlabel("Day", fontsize=11)
    ax_demand.set_ylabel("Daily Demand (units)", fontsize=11)
    ax_demand.set_title("Daily Demand (normal vs spike period)", fontsize=11)
    ax_demand.legend(fontsize=9)
    ax_demand.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = "results/sim1_daily_trace.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {path2}")

    plt.show()
    return path1, path2


# ── Run when executed directly ────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("Simulation 1 Visualiser — generating all graphs")
    print("=" * 65)

    reorder_points = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800]
    
    print("\nRunning experiment (this takes ~30 seconds)...")
    results = run_experiment(
        reorder_points=reorder_points,
        order_quantity=1500,
        demand_mean=200,
        demand_std=40,
        lead_time_days=7,
        seasonal_spike=True,
        trials=20,
        save_csv=True
    )

    # Get daily trace from optimal policy (ROP=1000)
    print("Getting daily trace for optimal policy...")
    optimal = run_inventory_sim(
        reorder_point=1000,
        order_quantity=1500,
        demand_mean=200,
        demand_std=40,
        lead_time_days=7,
        seasonal_spike=True,
        trials=20
    )
    daily_trace = optimal["daily_trace"]

    print("\nGenerating graphs...")
    plot_all(results, daily_trace, reorder_points)

    print("\nAll done. Check your results/ folder for:")
    print("  sim1_policy_comparison.png  — four-panel policy analysis")
    print("  sim1_daily_trace.png        — daily inventory and demand trace")
    print("  simulation1_results.csv     — full results table")

    # Print summary table
    print("\nResults summary:")
    print("-" * 75)
    print(f"{'ROP':>6} | {'Service':>8} | {'Total Cost':>11} | "
          f"{'Holding':>9} | {'Ordering':>9} | {'Shortage':>9}")
    print("-" * 75)
    for r in results:
        marker = " ← optimal" if r["avg_total_cost_gbp"] == min(x["avg_total_cost_gbp"] for x in results) else ""
        print(
            f"{r['reorder_point']:>6} | "
            f"{r['avg_service_level_pct']:>7}% | "
            f"£{r['avg_total_cost_gbp']:>10,.0f} | "
            f"£{r['avg_holding_cost_gbp']:>8,.0f} | "
            f"£{r['avg_ordering_cost_gbp']:>8,.0f} | "
            f"£{r['avg_shortage_cost_gbp']:>8,.0f}{marker}"
        )
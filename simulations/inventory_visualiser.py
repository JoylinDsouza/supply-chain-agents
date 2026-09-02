# simulations/inventory_visualiser.py
#
# Visualisation for Simulation 1: Inventory Replenishment
#
# Compatible with the current inventory_sim.py:
#   - reorder_points
#   - order_quantities
#   - joint ROP x OQ experiment
#   - 95% service-level feasibility constraint
#   - reproducible random seed
#
# Produces:
#   1. Service level heatmap
#   2. Total cost heatmap
#   3. Service level vs reorder point
#   4. Total cost vs reorder point
#   5. Cost breakdown
#   6. Daily inventory trace
#
# Run from project root:
#   python simulations/inventory_visualiser.py


import matplotlib.pyplot as plt
import pandas as pd
import os
import sys


# ---------------------------------------------------------------------------
# PROJECT PATHS AND IMPORTS
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from simulations.inventory_sim import (
    run_experiment,
    run_inventory_sim
)


# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

TARGET_SERVICE_LEVEL = 95.0

# Fixed seed for reproducible visualisation.
# The final quantitative evaluation can use a larger number of trials.
RANDOM_SEED = 42

# Candidate reorder points.
REORDER_POINTS = [
    200,
    400,
    600,
    800,
    1000,
    1200,
    1400,
    1600
]

# Candidate order quantities.
ORDER_QUANTITIES = [
    1000,
    1500,
    2000
]

# Number of Monte Carlo trials used for the visualisation run.
VISUALISATION_TRIALS = 20


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def find_feasible_policies(results):
    """
    Return all policies satisfying the service-level constraint.
    """

    return [
        r
        for r in results
        if r["avg_service_level_pct"]
        >= TARGET_SERVICE_LEVEL
    ]


def find_best_policy(results):
    """
    Find the lowest-cost policy satisfying the service-level target.

    If no policy is feasible, return None.

    This is deliberately different from simply selecting the cheapest
    policy. The optimisation objective is:

        minimise total cost
        subject to service level >= 95%
    """

    feasible = find_feasible_policies(results)

    if not feasible:
        return None

    return min(
        feasible,
        key=lambda r: (
            r["avg_total_cost_gbp"],
            -r["avg_service_level_pct"]
        )
    )


def find_cheapest_policy(results):
    """
    Find the overall cheapest policy without applying the service-level
    constraint.

    This is useful for showing the difference between unconstrained
    cost minimisation and constrained optimisation.
    """

    return min(
        results,
        key=lambda r: r["avg_total_cost_gbp"]
    )


def print_summary(results):
    """
    Print a clear summary of the joint ROP x OQ experiment.
    """

    print("\n" + "=" * 95)
    print("FINAL INVENTORY POLICY SUMMARY")
    print("=" * 95)

    feasible = find_feasible_policies(results)

    cheapest = find_cheapest_policy(results)

    print(
        f"\nTarget service level: "
        f"{TARGET_SERVICE_LEVEL:.1f}%"
    )

    print(
        f"Policies evaluated: "
        f"{len(results)}"
    )

    print(
        f"Feasible policies: "
        f"{len(feasible)}"
    )

    print("\nLowest-cost policy without service constraint:")

    print(
        f"  ROP:             "
        f"{cheapest['reorder_point']} units"
    )

    print(
        f"  Order Quantity:  "
        f"{cheapest['order_quantity']} units"
    )

    print(
        f"  Service Level:   "
        f"{cheapest['avg_service_level_pct']:.2f}%"
    )

    print(
        f"  Total Cost:      "
        f"£{cheapest['avg_total_cost_gbp']:,.2f}"
    )

    if feasible:

        best = find_best_policy(results)

        print(
            "\nLowest-cost FEASIBLE policy:"
        )

        print(
            f"  ROP:             "
            f"{best['reorder_point']} units"
        )

        print(
            f"  Order Quantity:  "
            f"{best['order_quantity']} units"
        )

        print(
            f"  Service Level:   "
            f"{best['avg_service_level_pct']:.2f}%"
        )

        if "avg_fill_rate_pct" in best:

            print(
                f"  Fill Rate:       "
                f"{best['avg_fill_rate_pct']:.2f}%"
            )

        print(
            f"  Total Cost:      "
            f"£{best['avg_total_cost_gbp']:,.2f}"
        )

        print(
            f"  Holding Cost:    "
            f"£{best['avg_holding_cost_gbp']:,.2f}"
        )

        print(
            f"  Ordering Cost:   "
            f"£{best['avg_ordering_cost_gbp']:,.2f}"
        )

        print(
            f"  Shortage Cost:   "
            f"£{best['avg_shortage_cost_gbp']:,.2f}"
        )

    else:

        print(
            "\nWARNING: No evaluated policy satisfies "
            f"the {TARGET_SERVICE_LEVEL:.1f}% service-level target."
        )

        print(
            "No policy will be labelled as the recommended "
            "feasible policy."
        )

    print("=" * 95)


# ---------------------------------------------------------------------------
# FIGURE 1: SERVICE LEVEL HEATMAP
# ---------------------------------------------------------------------------

def plot_service_heatmap(results):

    df = pd.DataFrame(results)

    pivot = df.pivot(
        index="order_quantity",
        columns="reorder_point",
        values="avg_service_level_pct"
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    image = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower"
    )

    ax.set_xticks(
        range(len(pivot.columns))
    )

    ax.set_xticklabels(
        pivot.columns
    )

    ax.set_yticks(
        range(len(pivot.index))
    )

    ax.set_yticklabels(
        pivot.index
    )

    ax.set_xlabel(
        "Reorder Point (units)"
    )

    ax.set_ylabel(
        "Order Quantity (units)"
    )

    ax.set_title(
        "Inventory Replenishment: "
        "Service Level by ROP and Order Quantity"
    )

    colourbar = fig.colorbar(
        image,
        ax=ax
    )

    colourbar.set_label(
        "Service Level (%)"
    )

    # Write values inside cells.
    for i in range(len(pivot.index)):

        for j in range(len(pivot.columns)):

            value = pivot.iloc[i, j]

            ax.text(
                j,
                i,
                f"{value:.1f}%",
                ha="center",
                va="center",
                fontsize=9
            )

    # 95% target reference.
    ax.text(
        1.02,
        0.5,
        f"Target = {TARGET_SERVICE_LEVEL:.0f}%",
        transform=ax.transAxes,
        rotation=90,
        va="center"
    )

    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "sim1_service_level_heatmap.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {path}"
    )


# ---------------------------------------------------------------------------
# FIGURE 2: TOTAL COST HEATMAP
# ---------------------------------------------------------------------------

def plot_cost_heatmap(results):

    df = pd.DataFrame(results)

    pivot = df.pivot(
        index="order_quantity",
        columns="reorder_point",
        values="avg_total_cost_gbp"
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    image = ax.imshow(
        pivot.values,
        aspect="auto",
        origin="lower"
    )

    ax.set_xticks(
        range(len(pivot.columns))
    )

    ax.set_xticklabels(
        pivot.columns
    )

    ax.set_yticks(
        range(len(pivot.index))
    )

    ax.set_yticklabels(
        pivot.index
    )

    ax.set_xlabel(
        "Reorder Point (units)"
    )

    ax.set_ylabel(
        "Order Quantity (units)"
    )

    ax.set_title(
        "Inventory Replenishment: "
        "Total Cost by ROP and Order Quantity"
    )

    colourbar = fig.colorbar(
        image,
        ax=ax
    )

    colourbar.set_label(
        "Total Cost (£)"
    )

    for i in range(len(pivot.index)):

        for j in range(len(pivot.columns)):

            value = pivot.iloc[i, j]

            ax.text(
                j,
                i,
                f"£{value:,.0f}",
                ha="center",
                va="center",
                fontsize=8
            )

    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "sim1_total_cost_heatmap.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {path}"
    )


# ---------------------------------------------------------------------------
# FIGURE 3: SERVICE LEVEL VS ROP
# ---------------------------------------------------------------------------

def plot_service_level(results):

    df = pd.DataFrame(results)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for oq in sorted(
        df["order_quantity"].unique()
    ):

        subset = df[
            df["order_quantity"] == oq
        ].sort_values(
            "reorder_point"
        )

        ax.plot(
            subset["reorder_point"],
            subset["avg_service_level_pct"],
            marker="o",
            linewidth=2,
            label=f"OQ = {oq}"
        )

    ax.axhline(
        TARGET_SERVICE_LEVEL,
        linestyle="--",
        linewidth=1.5,
        label=(
            f"{TARGET_SERVICE_LEVEL:.0f}% "
            "service target"
        )
    )

    ax.set_xlabel(
        "Reorder Point (units)"
    )

    ax.set_ylabel(
        "Average Service Level (%)"
    )

    ax.set_title(
        "Service Level vs Reorder Point"
    )

    ax.set_ylim(
        0,
        105
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "sim1_service_level_vs_rop.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {path}"
    )


# ---------------------------------------------------------------------------
# FIGURE 4: TOTAL COST VS ROP
# ---------------------------------------------------------------------------

def plot_total_cost(results):

    df = pd.DataFrame(results)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for oq in sorted(
        df["order_quantity"].unique()
    ):

        subset = df[
            df["order_quantity"] == oq
        ].sort_values(
            "reorder_point"
        )

        ax.plot(
            subset["reorder_point"],
            subset["avg_total_cost_gbp"],
            marker="o",
            linewidth=2,
            label=f"OQ = {oq}"
        )

    # Identify feasible policies.
    feasible = df[
        df["avg_service_level_pct"]
        >= TARGET_SERVICE_LEVEL
    ]

    if not feasible.empty:

        best = feasible.loc[
            feasible["avg_total_cost_gbp"].idxmin()
        ]

        ax.scatter(
            best["reorder_point"],
            best["avg_total_cost_gbp"],
            s=150,
            marker="*",
            label=(
                "Recommended: "
                f"ROP={int(best['reorder_point'])}, "
                f"OQ={int(best['order_quantity'])}"
            )
        )

    ax.set_xlabel(
        "Reorder Point (units)"
    )

    ax.set_ylabel(
        "Average Total Cost (£)"
    )

    ax.set_title(
        "Total Cost vs Reorder Point"
    )

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"£{x:,.0f}"
        )
    )

    ax.grid(
        True,
        alpha=0.3
    )

    ax.legend()

    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "sim1_total_cost_vs_rop.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {path}"
    )


# ---------------------------------------------------------------------------
# FIGURE 5: COST BREAKDOWN
# ---------------------------------------------------------------------------

def plot_cost_breakdown(results):

    best = find_best_policy(results)

    if best is None:

        print(
            "Skipped cost breakdown: "
            "no feasible policy was found."
        )

        return

    labels = [
        "Holding",
        "Ordering",
        "Shortage"
    ]

    values = [
        best["avg_holding_cost_gbp"],
        best["avg_ordering_cost_gbp"],
        best["avg_shortage_cost_gbp"]
    ]

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    bars = ax.bar(
        labels,
        values
    )

    ax.set_title(
        "Cost Breakdown of Recommended Inventory Policy"
    )

    ax.set_ylabel(
        "Average Cost (£)"
    )

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, _: f"£{x:,.0f}"
        )
    )

    ax.grid(
        True,
        alpha=0.3,
        axis="y"
    )

    for bar, value in zip(
        bars,
        values
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"£{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.text(
        0.5,
        0.98,
        (
            f"ROP = {best['reorder_point']} | "
            f"OQ = {best['order_quantity']} | "
            f"Service = "
            f"{best['avg_service_level_pct']:.1f}%"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10
    )

    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "sim1_cost_breakdown.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {path}"
    )


# ---------------------------------------------------------------------------
# FIGURE 6: DAILY INVENTORY TRACE
# ---------------------------------------------------------------------------

def plot_daily_trace(best):

    if best is None:

        print(
            "Skipped daily trace: "
            "no feasible policy was found."
        )

        return

    print(
        "\nGenerating daily trace using:"
    )

    print(
        f"  ROP = {best['reorder_point']}"
    )

    print(
        f"  OQ  = {best['order_quantity']}"
    )

    # IMPORTANT:
    # The trace uses the same selected policy as the main experiment.
    #
    # A fixed seed is used so that the trace is reproducible.
    # This avoids silently generating a different stochastic scenario.
    trace_result = run_inventory_sim(
        reorder_point=int(
            best["reorder_point"]
        ),
        order_quantity=int(
            best["order_quantity"]
        ),
        demand_mean=200,
        demand_std=40,
        lead_time_days=7,
        seasonal_spike=True,
        trials=VISUALISATION_TRIALS,
        seed=RANDOM_SEED
    )

    daily_trace = trace_result[
        "daily_trace"
    ]

    if not daily_trace:

        print(
            "WARNING: No daily trace was returned."
        )

        return

    days = [
        d["day"]
        for d in daily_trace
    ]

    inventory = [
        d["inventory"]
        for d in daily_trace
    ]

    demand = [
        d["demand"]
        for d in daily_trace
    ]

    weeks = [
        d["week"]
        for d in daily_trace
    ]

    fig, (ax_inv, ax_demand) = plt.subplots(
        2,
        1,
        figsize=(13, 8),
        sharex=True,
        height_ratios=[2, 1]
    )

    fig.suptitle(
        (
            "Simulation 1: Daily Inventory Trace\n"
            f"Recommended Policy: "
            f"ROP={best['reorder_point']}, "
            f"OQ={best['order_quantity']}"
        ),
        fontsize=13,
        fontweight="bold"
    )

    # --------------------------------------------------------------
    # Inventory level.
    # --------------------------------------------------------------
    ax_inv.plot(
        days,
        inventory,
        linewidth=1.5,
        label="Inventory level"
    )

    ax_inv.axhline(
        best["reorder_point"],
        linestyle="--",
        linewidth=1.2,
        label=(
            f"ROP = "
            f"{best['reorder_point']}"
        )
    )

    ax_inv.set_ylabel(
        "Inventory (units)"
    )

    ax_inv.set_title(
        "Inventory Level Over 90 Days"
    )

    ax_inv.grid(
        True,
        alpha=0.3
    )

    ax_inv.legend()

    ax_inv.set_ylim(
        bottom=0
    )

    # --------------------------------------------------------------
    # Daily demand.
    # --------------------------------------------------------------
    normal_demand = []
    spike_demand = []

    for d, week in zip(
        demand,
        weeks
    ):

        if week in (6, 7):

            normal_demand.append(0)
            spike_demand.append(d)

        else:

            normal_demand.append(d)
            spike_demand.append(0)

    ax_demand.bar(
        days,
        normal_demand,
        width=1,
        alpha=0.7,
        label="Normal demand"
    )

    ax_demand.bar(
        days,
        spike_demand,
        width=1,
        alpha=0.8,
        label="Seasonal spike"
    )

    ax_demand.set_xlabel(
        "Day"
    )

    ax_demand.set_ylabel(
        "Demand (units)"
    )

    ax_demand.set_title(
        "Daily Demand"
    )

    ax_demand.grid(
        True,
        alpha=0.3
    )

    ax_demand.legend()

    plt.tight_layout()

    path = os.path.join(
        RESULTS_DIR,
        "sim1_daily_inventory_trace.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        f"Saved: {path}"
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "Simulation 1 Visualiser"
    )

    print(
        "Joint ROP x Order Quantity Analysis"
    )

    print(
        "=" * 70
    )

    print(
        "\nRunning joint experiment..."
    )

    print(
        f"Monte Carlo trials per policy: "
        f"{VISUALISATION_TRIALS}"
    )

    print(
        f"Random seed: {RANDOM_SEED}"
    )

    results = run_experiment(
        reorder_points=REORDER_POINTS,
        order_quantities=ORDER_QUANTITIES,
        demand_mean=200,
        demand_std=40,
        lead_time_days=7,
        seasonal_spike=True,
        trials=VISUALISATION_TRIALS,
        save_csv=True,
        target_service_level=TARGET_SERVICE_LEVEL,
        seed=RANDOM_SEED
    )

    print(
        f"\nGenerated {len(results)} "
        "policy combinations."
    )

    # --------------------------------------------------------------
    # Determine the selected policy ONCE.
    #
    # The same object is then passed to the summary, cost breakdown,
    # total-cost plot and daily trace.
    # --------------------------------------------------------------
    best = find_best_policy(
        results
    )

    # Print summary.
    print_summary(
        results
    )

    print(
        "\nGenerating graphs..."
    )

    # --------------------------------------------------------------
    # Generate all visualisations from the same experiment results.
    # --------------------------------------------------------------
    plot_service_heatmap(
        results
    )

    plot_cost_heatmap(
        results
    )

    plot_service_level(
        results
    )

    plot_total_cost(
        results
    )

    plot_cost_breakdown(
        results
    )

    # IMPORTANT:
    # Pass the already-selected policy into the daily trace.
    # The trace therefore cannot accidentally select another policy.
    plot_daily_trace(
        best
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL GRAPHS GENERATED SUCCESSFULLY"
    )

    print(
        "=" * 70
    )

    print(
        "\nCheck your results/ folder for:"
    )

    print(
        "  sim1_service_level_heatmap.png"
    )

    print(
        "  sim1_total_cost_heatmap.png"
    )

    print(
        "  sim1_service_level_vs_rop.png"
    )

    print(
        "  sim1_total_cost_vs_rop.png"
    )

    print(
        "  sim1_cost_breakdown.png"
    )

    print(
        "  sim1_daily_inventory_trace.png"
    )

    print(
        "  simulation1_results.csv"
    )
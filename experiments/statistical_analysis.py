# experiments/statistical_analysis.py
#
# Statistical analysis of the simulation experiments.
#
# Uses repeated independent simulation runs and bootstrap resampling to
# estimate uncertainty around mean service level and cost.
#
# IMPORTANT:
# - Bootstrap intervals around a mean are confidence intervals for the
#   estimated mean, not outcome ranges.
# - Statistical comparison between two strategies is performed on the
#   paired difference in each repeated simulation run.
# - Overlap/non-overlap of two separate confidence intervals is NOT used
#   as a formal significance test.
#
# Canonical simulation configuration:
#   Inventory: 1000 trials per simulation, seed 42
#   Repeated statistical runs use different seeds to generate independent
#   replications.
#
# The statistical analysis therefore asks:
#   "How stable are the estimated simulation means across repeated runs?"
# and
#   "Is the difference between two strategies distinguishable from zero
#    under bootstrap resampling?"


import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from simulations.inventory_sim import run_inventory_sim
from simulations.supplier_disruption_sim import (
    run_supplier_disruption_sim,
)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "statistics",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Canonical configuration
# ---------------------------------------------------------------------------

INVENTORY_SIM_TRIALS = 1000
INVENTORY_BASE_SEED = 42

INVENTORY_SERVICE_TARGET = 95.0

DISRUPTION_DURATION_DAYS = 42
DISRUPTION_DAILY_DEMAND = 200
DISRUPTION_UNIT_COST = 10
DISRUPTION_SHORTAGE_COST_PER_UNIT = 50

RESILIENCE_SERVICE_TARGET = 95.0

DEFAULT_N_REPLICATIONS = 100
DEFAULT_N_BOOTSTRAP = 2000

CONFIDENCE_LEVEL = 0.95


# ===========================================================================
# Bootstrap utilities
# ===========================================================================

def bootstrap_ci(
    data,
    stat_fn=np.mean,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
    ci=CONFIDENCE_LEVEL,
    seed=12345,
):
    """
    Calculate a percentile bootstrap confidence interval.

    Returns:
        {
            "lower": lower confidence limit,
            "mean": original-sample statistic,
            "upper": upper confidence limit
        }

    The interval is a confidence interval for the estimated statistic.
    It should not be described as a prediction interval or as the range
    of individual simulation outcomes.
    """

    data = np.asarray(data, dtype=float)

    if data.size == 0:
        raise ValueError("Cannot bootstrap an empty dataset.")

    if data.size == 1:
        value = float(stat_fn(data))
        return {
            "lower": round(value, 2),
            "mean": round(value, 2),
            "upper": round(value, 2),
        }

    rng = np.random.default_rng(seed)

    bootstrap_stats = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(n_bootstrap):
        sample = rng.choice(
            data,
            size=data.size,
            replace=True,
        )
        bootstrap_stats[i] = stat_fn(sample)

    alpha = (1.0 - ci) / 2.0

    lower = np.percentile(
        bootstrap_stats,
        alpha * 100,
    )

    upper = np.percentile(
        bootstrap_stats,
        (1.0 - alpha) * 100,
    )

    estimate = stat_fn(data)

    return {
        "lower": round(float(lower), 2),
        "mean": round(float(estimate), 2),
        "upper": round(float(upper), 2),
    }


def bootstrap_difference_ci(
    data_a,
    data_b,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
    ci=CONFIDENCE_LEVEL,
    seed=54321,
):
    """
    Bootstrap confidence interval for the paired mean difference:

        difference = A - B

    The two arrays must represent matched replications.

    For example, if replication i uses the same random seed/scenario for
    strategy A and strategy B, then the difference for replication i is
    directly comparable.

    Returns:
        {
            "mean_difference": ...,
            "lower": ...,
            "upper": ...,
            "probability_difference_gt_zero": ...
        }
    """

    a = np.asarray(data_a, dtype=float)
    b = np.asarray(data_b, dtype=float)

    if len(a) != len(b):
        raise ValueError(
            "Paired bootstrap requires datasets of equal length."
        )

    if len(a) == 0:
        raise ValueError(
            "Cannot calculate a difference from empty datasets."
        )

    differences = a - b

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(n_bootstrap):
        indices = rng.integers(
            0,
            len(differences),
            size=len(differences),
        )

        bootstrap_means[i] = np.mean(
            differences[indices]
        )

    alpha = (1.0 - ci) / 2.0

    lower = np.percentile(
        bootstrap_means,
        alpha * 100,
    )

    upper = np.percentile(
        bootstrap_means,
        (1.0 - alpha) * 100,
    )

    return {
        "mean_difference": round(
            float(np.mean(differences)),
            2,
        ),
        "lower": round(
            float(lower),
            2,
        ),
        "upper": round(
            float(upper),
            2,
        ),
        "probability_difference_gt_zero": round(
            float(np.mean(bootstrap_means > 0)),
            4,
        ),
    }


def interpret_difference(ci_result):
    """
    Interpret a bootstrap confidence interval for a difference.

    If zero is outside the 95% interval, the result is described as
    statistically distinguishable from zero at the corresponding
    two-sided 5% level.

    This avoids the incorrect practice of comparing two separate CIs.
    """

    lower = ci_result["lower"]
    upper = ci_result["upper"]

    if lower > 0:
        return "difference_positive_and_excludes_zero"

    if upper < 0:
        return "difference_negative_and_excludes_zero"

    return "difference_ci_includes_zero"


# ===========================================================================
# Experiment 1: Inventory statistical analysis
# ===========================================================================

def run_statistical_inventory(
    reorder_point=1750,
    order_quantity=1500,
    n_replications=DEFAULT_N_REPLICATIONS,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
):
    """
    Estimate uncertainty around the inventory simulation metrics.

    The selected policy defaults to the policy found by the standalone
    optimiser:

        ROP = 1750
        OQ  = 1500

    Each replication performs a 1000-trial Monte Carlo simulation.

    Different seeds are used for the independent replications.
    """

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS — INVENTORY")
    print("=" * 70)

    print(
        f"Policy: ROP={reorder_point}, OQ={order_quantity}"
    )
    print(
        f"Replications: {n_replications}"
    )
    print(
        f"Simulation trials per replication: "
        f"{INVENTORY_SIM_TRIALS}"
    )
    print(
        f"Bootstrap resamples: {n_bootstrap}"
    )

    service_levels = []
    total_costs = []
    holding_costs = []
    shortage_costs = []

    for i in range(n_replications):

        # Different seed for each independent replication.
        seed = INVENTORY_BASE_SEED + i + 1

        result = run_inventory_sim(
            reorder_point=reorder_point,
            order_quantity=order_quantity,
            demand_mean=200,
            demand_std=40,
            lead_time_days=7,
            trials=INVENTORY_SIM_TRIALS,
            seed=seed,
        )

        r = result["results"]

        service_levels.append(
            float(r["avg_service_level_pct"])
        )

        total_costs.append(
            float(r["avg_total_cost_gbp"])
        )

        holding_costs.append(
            float(r["avg_holding_cost_gbp"])
        )

        shortage_costs.append(
            float(r["avg_shortage_cost_gbp"])
        )

    # -----------------------------------------------------------------------
    # Bootstrap confidence intervals
    # -----------------------------------------------------------------------

    service_ci = bootstrap_ci(
        service_levels,
        n_bootstrap=n_bootstrap,
        seed=1001,
    )

    cost_ci = bootstrap_ci(
        total_costs,
        n_bootstrap=n_bootstrap,
        seed=1002,
    )

    holding_ci = bootstrap_ci(
        holding_costs,
        n_bootstrap=n_bootstrap,
        seed=1003,
    )

    shortage_ci = bootstrap_ci(
        shortage_costs,
        n_bootstrap=n_bootstrap,
        seed=1004,
    )

    results = {
        "reorder_point": reorder_point,
        "order_quantity": order_quantity,
        "n_replications": n_replications,
        "simulation_trials_per_replication": (
            INVENTORY_SIM_TRIALS
        ),
        "service_target_pct": INVENTORY_SERVICE_TARGET,

        "service_level_pct": {
            "lower_95ci": service_ci["lower"],
            "mean": service_ci["mean"],
            "upper_95ci": service_ci["upper"],
        },

        "total_cost_gbp": {
            "lower_95ci": cost_ci["lower"],
            "mean": cost_ci["mean"],
            "upper_95ci": cost_ci["upper"],
        },

        "holding_cost_gbp": {
            "lower_95ci": holding_ci["lower"],
            "mean": holding_ci["mean"],
            "upper_95ci": holding_ci["upper"],
        },

        "shortage_cost_gbp": {
            "lower_95ci": shortage_ci["lower"],
            "mean": shortage_ci["mean"],
            "upper_95ci": shortage_ci["upper"],
        },
    }

    # -----------------------------------------------------------------------
    # Print
    # -----------------------------------------------------------------------

    print("\nInventory results")
    print("-" * 70)

    print(
        f"Service level: "
        f"{service_ci['mean']:.2f}% "
        f"(95% CI: "
        f"{service_ci['lower']:.2f}%–"
        f"{service_ci['upper']:.2f}%)"
    )

    print(
        f"Total cost: "
        f"£{cost_ci['mean']:,.2f} "
        f"(95% CI: "
        f"£{cost_ci['lower']:,.2f}–"
        f"£{cost_ci['upper']:,.2f})"
    )

    print(
        f"Holding cost: "
        f"£{holding_ci['mean']:,.2f} "
        f"(95% CI: "
        f"£{holding_ci['lower']:,.2f}–"
        f"£{holding_ci['upper']:,.2f})"
    )

    print(
        f"Shortage cost: "
        f"£{shortage_ci['mean']:,.2f} "
        f"(95% CI: "
        f"£{shortage_ci['lower']:,.2f}–"
        f"£{shortage_ci['upper']:,.2f})"
    )

    return results


# ===========================================================================
# Experiment 3: Disruption statistical comparison
# ===========================================================================

def run_statistical_disruption(
    disruption_probability=0.25,
    n_replications=DEFAULT_N_REPLICATIONS,
    n_bootstrap=DEFAULT_N_BOOTSTRAP,
):
    """
    Compare dual sourcing with no backup using matched simulation
    replications.

    The comparison uses the difference:

        no_backup cost - dual_sourcing cost

    A positive value therefore means dual sourcing is cheaper.

    The same random seed is used for both strategies in each replication,
    providing a paired comparison under the same simulated disruption
    scenario.
    """

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS — DISRUPTION STRATEGIES")
    print("=" * 70)

    print(
        f"Disruption probability: "
        f"{disruption_probability * 100:.0f}%"
    )
    print(
        f"Replications: {n_replications}"
    )
    print(
        f"Bootstrap resamples: {n_bootstrap}"
    )

    strategy_data = {
        "no_backup": {
            "costs": [],
            "service_levels": [],
        },
        "dual_sourcing": {
            "costs": [],
            "service_levels": [],
        },
    }

    # -----------------------------------------------------------------------
    # Matched replications
    # -----------------------------------------------------------------------

    for i in range(n_replications):

        seed = 1000 + i

        for strategy in [
            "no_backup",
            "dual_sourcing",
        ]:

            kwargs = {
                "strategy": strategy,
                "disruption_probability": (
                    disruption_probability
                ),
                "disruption_duration_days": (
                    DISRUPTION_DURATION_DAYS
                ),
                "daily_demand": (
                    DISRUPTION_DAILY_DEMAND
                ),
                "unit_cost": DISRUPTION_UNIT_COST,
                "shortage_cost_per_unit": (
                    DISRUPTION_SHORTAGE_COST_PER_UNIT
                ),
                "trials": 1,
                "seed": seed,
            }

            if strategy == "dual_sourcing":
                kwargs["dual_sourcing_premium"] = 0.10

            result = run_supplier_disruption_sim(
                **kwargs
            )

            r = result["results"]

            strategy_data[strategy]["costs"].append(
                float(r["avg_total_cost_gbp"])
            )

            strategy_data[strategy]["service_levels"].append(
                float(r["avg_service_level_pct"])
            )

    # -----------------------------------------------------------------------
    # Individual strategy confidence intervals
    # -----------------------------------------------------------------------

    output = {}

    for strategy, data in strategy_data.items():

        cost_ci = bootstrap_ci(
            data["costs"],
            n_bootstrap=n_bootstrap,
            seed=2000 + len(output),
        )

        service_ci = bootstrap_ci(
            data["service_levels"],
            n_bootstrap=n_bootstrap,
            seed=3000 + len(output),
        )

        output[strategy] = {
            "total_cost_gbp": {
                "lower_95ci": cost_ci["lower"],
                "mean": cost_ci["mean"],
                "upper_95ci": cost_ci["upper"],
            },
            "service_level_pct": {
                "lower_95ci": service_ci["lower"],
                "mean": service_ci["mean"],
                "upper_95ci": service_ci["upper"],
            },
        }

        print(
            f"\n{strategy}"
        )
        print(
            f"  Cost: £{cost_ci['mean']:,.2f} "
            f"(95% CI: "
            f"£{cost_ci['lower']:,.2f}–"
            f"£{cost_ci['upper']:,.2f})"
        )
        print(
            f"  Service: {service_ci['mean']:.2f}% "
            f"(95% CI: "
            f"{service_ci['lower']:.2f}%–"
            f"{service_ci['upper']:.2f}%)"
        )

    # -----------------------------------------------------------------------
    # Paired cost comparison
    # -----------------------------------------------------------------------

    no_backup_costs = np.asarray(
        strategy_data["no_backup"]["costs"],
        dtype=float,
    )

    dual_sourcing_costs = np.asarray(
        strategy_data["dual_sourcing"]["costs"],
        dtype=float,
    )

    # Positive difference means dual sourcing costs less.
    cost_difference = bootstrap_difference_ci(
        no_backup_costs,
        dual_sourcing_costs,
        n_bootstrap=n_bootstrap,
        seed=4001,
    )

    cost_difference_interpretation = interpret_difference(
        cost_difference
    )

    output["paired_cost_comparison"] = {
        "comparison": (
            "no_backup_minus_dual_sourcing"
        ),
        "mean_difference_gbp": (
            cost_difference["mean_difference"]
        ),
        "lower_95ci_gbp": (
            cost_difference["lower"]
        ),
        "upper_95ci_gbp": (
            cost_difference["upper"]
        ),
        "bootstrap_probability_difference_gt_zero": (
            cost_difference[
                "probability_difference_gt_zero"
            ]
        ),
        "interpretation": (
            cost_difference_interpretation
        ),
    }

    # -----------------------------------------------------------------------
    # Paired service-level comparison
    # -----------------------------------------------------------------------

    no_backup_service = np.asarray(
        strategy_data["no_backup"]["service_levels"],
        dtype=float,
    )

    dual_sourcing_service = np.asarray(
        strategy_data["dual_sourcing"]["service_levels"],
        dtype=float,
    )

    # Positive difference means no backup has higher service.
    service_difference = bootstrap_difference_ci(
        no_backup_service,
        dual_sourcing_service,
        n_bootstrap=n_bootstrap,
        seed=4002,
    )

    service_difference_interpretation = interpret_difference(
        service_difference
    )

    output["paired_service_comparison"] = {
        "comparison": (
            "no_backup_minus_dual_sourcing"
        ),
        "mean_difference_percentage_points": (
            service_difference["mean_difference"]
        ),
        "lower_95ci_percentage_points": (
            service_difference["lower"]
        ),
        "upper_95ci_percentage_points": (
            service_difference["upper"]
        ),
        "bootstrap_probability_difference_gt_zero": (
            service_difference[
                "probability_difference_gt_zero"
            ]
        ),
        "interpretation": (
            service_difference_interpretation
        ),
    }

    # -----------------------------------------------------------------------
    # Print formal comparison
    # -----------------------------------------------------------------------

    print("\nPaired comparison")
    print("-" * 70)

    print(
        "Cost difference = no_backup − dual_sourcing"
    )

    print(
        f"Mean difference: "
        f"£{cost_difference['mean_difference']:,.2f}"
    )

    print(
        f"95% bootstrap CI: "
        f"£{cost_difference['lower']:,.2f}–"
        f"£{cost_difference['upper']:,.2f}"
    )

    print(
        f"Bootstrap P(difference > 0): "
        f"{cost_difference['probability_difference_gt_zero']:.4f}"
    )

    print(
        f"Interpretation: "
        f"{cost_difference_interpretation}"
    )

    print(
        "\nService difference = "
        "no_backup − dual_sourcing"
    )

    print(
        f"Mean difference: "
        f"{service_difference['mean_difference']:.2f} "
        f"percentage points"
    )

    print(
        f"95% bootstrap CI: "
        f"{service_difference['lower']:.2f}–"
        f"{service_difference['upper']:.2f} "
        f"percentage points"
    )

    print(
        f"Interpretation: "
        f"{service_difference_interpretation}"
    )

    return output


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STATISTICAL ANALYSIS")
    print(
        "Bootstrap confidence intervals and paired comparisons"
    )
    print("=" * 70)

    print("\nConfiguration:")
    print(
        f"  Inventory simulation trials: "
        f"{INVENTORY_SIM_TRIALS}"
    )
    print(
        f"  Statistical replications: "
        f"{DEFAULT_N_REPLICATIONS}"
    )
    print(
        f"  Bootstrap resamples: "
        f"{DEFAULT_N_BOOTSTRAP}"
    )
    print(
        f"  Confidence level: "
        f"{CONFIDENCE_LEVEL:.0%}"
    )

    # -----------------------------------------------------------------------
    # Inventory
    # -----------------------------------------------------------------------

    inv_stats = run_statistical_inventory(
        reorder_point=1750,
        order_quantity=1500,
        n_replications=DEFAULT_N_REPLICATIONS,
        n_bootstrap=DEFAULT_N_BOOTSTRAP,
    )

    # -----------------------------------------------------------------------
    # Disruption
    # -----------------------------------------------------------------------

    dis_stats = run_statistical_disruption(
        disruption_probability=0.25,
        n_replications=DEFAULT_N_REPLICATIONS,
        n_bootstrap=DEFAULT_N_BOOTSTRAP,
    )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------

    all_stats = {
        "configuration": {
            "inventory_simulation_trials": (
                INVENTORY_SIM_TRIALS
            ),
            "inventory_base_seed": (
                INVENTORY_BASE_SEED
            ),
            "inventory_service_target_pct": (
                INVENTORY_SERVICE_TARGET
            ),
            "disruption_duration_days": (
                DISRUPTION_DURATION_DAYS
            ),
            "disruption_daily_demand": (
                DISRUPTION_DAILY_DEMAND
            ),
            "disruption_unit_cost": (
                DISRUPTION_UNIT_COST
            ),
            "disruption_shortage_cost_per_unit": (
                DISRUPTION_SHORTAGE_COST_PER_UNIT
            ),
            "resilience_service_target_pct": (
                RESILIENCE_SERVICE_TARGET
            ),
            "n_replications": (
                DEFAULT_N_REPLICATIONS
            ),
            "n_bootstrap": (
                DEFAULT_N_BOOTSTRAP
            ),
            "confidence_level": (
                CONFIDENCE_LEVEL
            ),
        },

        "inventory_selected_policy": inv_stats,

        "disruption_strategy_comparison": dis_stats,
    }

    path = os.path.join(
        OUTPUT_DIR,
        "confidence_intervals.json",
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            all_stats,
            f,
            indent=2,
        )

    print(
        f"\nFull results saved to: {path}"
    )

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("=" * 70)
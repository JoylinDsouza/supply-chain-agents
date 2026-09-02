# experiments/optimiser_baseline.py
#
# Standalone optimiser baseline — no LLM involved.
#
# This experiment provides a third comparison point:
#
#   1. Standalone LLM
#   2. Standalone simulation optimiser
#   3. LLM + Simulation multi-agent system
#
# The optimiser performs an exhaustive search over the defined simulation
# parameter spaces and selects the best feasible result according to the
# stated objective.
#
# Reproducibility:
#   Inventory simulation: 1000 Monte Carlo trials, seed 42
#   Hub simulation:       canonical simulator configuration
#   Disruption simulation: canonical simulator configuration
#
# Important:
# The optimiser identifies the best policy/location/strategy FOUND WITHIN
# THE EVALUATED SEARCH SPACE. It must not be described as a global optimum.


import csv
import json
import os
import sys
from datetime import datetime


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from simulations.inventory_sim import run_inventory_sim
from simulations.hub_location_sim import run_hub_location_sim
from simulations.supplier_disruption_sim import (
    run_supplier_disruption_sim,
    STRATEGIES,
)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "optimiser")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Canonical experiment configuration
# ---------------------------------------------------------------------------

INVENTORY_TRIALS = 1000
INVENTORY_SEED = 42

# Service-level constraint used throughout the inventory optimisation.
INVENTORY_SERVICE_TARGET = 95.0

# Resilience service-level constraint.
RESILIENCE_SERVICE_TARGET = 95.0

# Common disruption assumptions.
DISRUPTION_PROBABILITY = 0.25
DISRUPTION_DURATION_DAYS = 42

# Common supplier simulation assumptions.
SUPPLIER_DAILY_DEMAND = 200
SUPPLIER_UNIT_COST = 10
SUPPLIER_SHORTAGE_COST = 50

# Common hub investment assumptions.
HUB_DISCOUNT_RATE = 0.08
HUB_YEARS = 10


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def save_csv(path, rows):
    """Save a list of dictionaries to CSV."""
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ===========================================================================
# Experiment 1: Inventory policy optimiser
# ===========================================================================

def optimise_inventory(
    demand_mean=200,
    demand_std=40,
    lead_time_days=7,
    trials=INVENTORY_TRIALS,
    seed=INVENTORY_SEED,
):
    """
    Exhaustively search the canonical inventory policy space.

    Objective:
        Minimise average total cost subject to service level >= 95%.

    Search space:
        ROP = 500 to 3500 in increments of 250
        OQ  = 500, 750, 1000, 1250, 1500, 2000, 2500

    If no policy satisfies the 95% service-level constraint, the policy
    with the highest service level is returned as a fallback.

    The result represents the lowest-cost feasible policy found within
    this evaluated search space. It is not a claim of global optimality.
    """

    print("\n" + "=" * 70)
    print("INVENTORY POLICY OPTIMISER")
    print("=" * 70)

    reorder_points = list(range(500, 3501, 250))
    order_quantities = [
        500,
        750,
        1000,
        1250,
        1500,
        2000,
        2500,
    ]

    total_combinations = len(reorder_points) * len(order_quantities)

    print(f"Demand mean:       {demand_mean}")
    print(f"Demand std:        {demand_std}")
    print(f"Lead time:         {lead_time_days} days")
    print(f"Monte Carlo trials:{trials}")
    print(f"Seed:              {seed}")
    print(f"Service target:    {INVENTORY_SERVICE_TARGET}%")
    print(f"Search space:      {total_combinations} policies")
    print()

    best_feasible = None
    best_infeasible = None
    cheapest_overall = None

    all_results = []

    count = 0

    for reorder_point in reorder_points:
        for order_quantity in order_quantities:

            count += 1

            if count == 1 or count % 20 == 0 or count == total_combinations:
                print(
                    f"  Progress: {count}/{total_combinations} "
                    f"({100 * count / total_combinations:.1f}%)"
                )

            result = run_inventory_sim(
                reorder_point=reorder_point,
                order_quantity=order_quantity,
                demand_mean=demand_mean,
                demand_std=demand_std,
                lead_time_days=lead_time_days,
                trials=trials,
                seed=seed,
            )

            r = result["results"]

            cost = float(r["avg_total_cost_gbp"])
            service_level = float(r["avg_service_level_pct"])

            row = {
                "reorder_point": reorder_point,
                "order_quantity": order_quantity,
                "avg_total_cost_gbp": cost,
                "avg_service_level_pct": service_level,
                "avg_fill_rate_pct": r.get("avg_fill_rate_pct"),
                "avg_stockout_days": r.get("avg_stockout_days"),
                "avg_holding_cost_gbp": r.get("avg_holding_cost_gbp"),
                "avg_shortage_cost_gbp": r.get("avg_shortage_cost_gbp"),
                "avg_ordering_cost_gbp": r.get("avg_ordering_cost_gbp"),
                "service_constraint_met": service_level >= INVENTORY_SERVICE_TARGET,
            }

            all_results.append(row)

            # Cheapest policy overall.
            if (
                cheapest_overall is None
                or cost < cheapest_overall["cost"]
            ):
                cheapest_overall = {
                    "reorder_point": reorder_point,
                    "order_quantity": order_quantity,
                    "cost": cost,
                    "service_level": service_level,
                    "result": r,
                }

            # Feasible policies.
            if service_level >= INVENTORY_SERVICE_TARGET:

                candidate = {
                    "reorder_point": reorder_point,
                    "order_quantity": order_quantity,
                    "cost": cost,
                    "service_level": service_level,
                    "result": r,
                }

                if best_feasible is None:
                    best_feasible = candidate

                elif (
                    cost < best_feasible["cost"]
                    or (
                        cost == best_feasible["cost"]
                        and service_level > best_feasible["service_level"]
                    )
                ):
                    best_feasible = candidate

            # Infeasible fallback.
            else:

                candidate = {
                    "reorder_point": reorder_point,
                    "order_quantity": order_quantity,
                    "cost": cost,
                    "service_level": service_level,
                    "result": r,
                }

                if best_infeasible is None:
                    best_infeasible = candidate

                elif (
                    service_level > best_infeasible["service_level"]
                    or (
                        service_level == best_infeasible["service_level"]
                        and cost < best_infeasible["cost"]
                    )
                ):
                    best_infeasible = candidate

    # -----------------------------------------------------------------------
    # Select recommendation
    # -----------------------------------------------------------------------

    if best_feasible is not None:
        recommendation = best_feasible
        feasible = True
        selection_status = "lowest_cost_feasible_policy"
    else:
        recommendation = best_infeasible
        feasible = False
        selection_status = "highest_service_policy_no_feasible_policy"

    # -----------------------------------------------------------------------
    # Rank feasible policies
    # -----------------------------------------------------------------------

    feasible_results = [
        row
        for row in all_results
        if row["service_constraint_met"]
    ]

    feasible_results_sorted = sorted(
        feasible_results,
        key=lambda row: (
            row["avg_total_cost_gbp"],
            -row["avg_service_level_pct"],
        ),
    )

    top_feasible = feasible_results_sorted[:10]

    # -----------------------------------------------------------------------
    # Save complete search
    # -----------------------------------------------------------------------

    search_path = os.path.join(
        OUTPUT_DIR,
        "inventory_search.csv",
    )

    save_csv(search_path, all_results)

    # -----------------------------------------------------------------------
    # Print result
    # -----------------------------------------------------------------------

    print("\nInventory optimiser result")
    print("-" * 70)

    if recommendation is None:
        print("No result was produced.")
        return None, False

    print(
        f"Selected policy: ROP={recommendation['reorder_point']}, "
        f"OQ={recommendation['order_quantity']}"
    )
    print(
        f"Average total cost: "
        f"£{recommendation['cost']:,.2f}"
    )
    print(
        f"Average service level: "
        f"{recommendation['service_level']:.2f}%"
    )
    print(
        f"95% service constraint met: "
        f"{feasible}"
    )
    print(f"Selection status: {selection_status}")

    if cheapest_overall is not None:
        print(
            f"Cheapest overall policy: "
            f"ROP={cheapest_overall['reorder_point']}, "
            f"OQ={cheapest_overall['order_quantity']}, "
            f"cost=£{cheapest_overall['cost']:,.2f}, "
            f"service={cheapest_overall['service_level']:.2f}%"
        )

    print(f"Full search saved to: {search_path}")

    return recommendation, feasible


# ===========================================================================
# Experiment 2: Hub location optimiser
# ===========================================================================

def optimise_hub_location():
    """
    Exhaustively evaluate the defined hub-location and demand-growth
    scenarios.

    The optimiser selects the scenario with the highest average NPV within
    the evaluated search space.

    Hub parameters match the dissertation scenarios:

        Germany:
            build cost = £12M
            annual operating cost = £2.5M
            freight saving = 18%

        Poland:
            build cost = £7M
            annual operating cost = £1.6M
            freight saving = 13%

        Netherlands:
            build cost = £15M
            annual operating cost = £3M
            freight saving = 23%

    Cost of capital = 8%
    Evaluation horizon = 10 years
    """

    print("\n" + "=" * 70)
    print("HUB LOCATION OPTIMISER")
    print("=" * 70)

    locations = [
        {
            "name": "Germany",
            "build_cost": 12.0,
            "ops_cost": 2.5,
            "freight_cost": 12.0,
            "freight_saving": 0.18,
        },
        {
            "name": "Poland",
            "build_cost": 7.0,
            "ops_cost": 1.6,
            "freight_cost": 12.0,
            "freight_saving": 0.13,
        },
        {
            "name": "Netherlands",
            "build_cost": 15.0,
            "ops_cost": 3.0,
            "freight_cost": 12.0,
            "freight_saving": 0.23,
        },
    ]

    growth_rates = [
        0.05,
        0.08,
        0.10,
        0.12,
        0.15,
        0.18,
        0.20,
    ]

    total_scenarios = len(locations) * len(growth_rates)

    print(f"Locations:         {len(locations)}")
    print(f"Growth scenarios:  {len(growth_rates)}")
    print(f"Total scenarios:   {total_scenarios}")
    print(f"Discount rate:     {HUB_DISCOUNT_RATE:.0%}")
    print(f"Horizon:           {HUB_YEARS} years")
    print()

    best = None
    all_results = []

    count = 0

    for location in locations:
        for growth in growth_rates:

            count += 1

            print(
                f"  Evaluating {count}/{total_scenarios}: "
                f"{location['name']} at {growth:.0%} growth"
            )

            # IMPORTANT:
            # The current hub simulator controls its canonical Monte Carlo
            # configuration internally. Do not pass an unsupported `trials`
            # argument here.
            result = run_hub_location_sim(
                candidate_location=location["name"],
                build_cost_millions=location["build_cost"],
                annual_ops_cost_millions=location["ops_cost"],
                current_freight_cost_millions=location["freight_cost"],
                freight_saving_pct=location["freight_saving"],
                demand_growth_rate=growth,
                discount_rate=HUB_DISCOUNT_RATE,
                years=HUB_YEARS,
            )

            r = result["results"]

            avg_npv = float(r["avg_npv_millions"])

            row = {
                "location": location["name"],
                "growth_rate_pct": growth * 100,
                "avg_npv_millions": avg_npv,
                "npv_p10_millions": r.get("npv_p10_millions"),
                "npv_p90_millions": r.get("npv_p90_millions"),
                "avg_breakeven_year": r.get("avg_breakeven_year"),
                "probability_profitable_pct": r.get(
                    "probability_profitable_pct"
                ),
                "recommendation": r.get("recommendation"),
            }

            all_results.append(row)

            if (
                best is None
                or avg_npv > best["npv"]
            ):
                best = {
                    "location": location["name"],
                    "growth_rate": growth,
                    "npv": avg_npv,
                    "probability": r.get(
                        "probability_profitable_pct"
                    ),
                    "breakeven": r.get(
                        "avg_breakeven_year"
                    ),
                    "npv_p10": r.get(
                        "npv_p10_millions"
                    ),
                    "npv_p90": r.get(
                        "npv_p90_millions"
                    ),
                    "recommendation": r.get(
                        "recommendation"
                    ),
                }

    search_path = os.path.join(
        OUTPUT_DIR,
        "hub_search.csv",
    )

    save_csv(search_path, all_results)

    print("\nHub optimiser result")
    print("-" * 70)

    print(
        f"Highest-NPV scenario found: "
        f"{best['location']} at "
        f"{best['growth_rate'] * 100:.0f}% growth"
    )
    print(
        f"Average NPV: "
        f"£{best['npv']:.3f}M"
    )
    print(
        f"Probability profitable: "
        f"{best['probability']}%"
    )
    print(
        f"Average break-even year: "
        f"{best['breakeven']}"
    )
    print(
        f"P10 NPV: "
        f"£{best['npv_p10']}M"
    )
    print(
        f"P90 NPV: "
        f"£{best['npv_p90']}M"
    )
    print(f"Full search saved to: {search_path}")

    return best


# ===========================================================================
# Experiment 3: Supplier disruption optimiser
# ===========================================================================

def optimise_disruption(
    disruption_probability=DISRUPTION_PROBABILITY,
    disruption_duration_days=DISRUPTION_DURATION_DAYS,
    daily_demand=SUPPLIER_DAILY_DEMAND,
):
    """
    Evaluate all available disruption strategies under identical simulation
    assumptions.

    Objective:
        Minimise average total cost subject to service level >= 95%.

    The same daily demand, unit cost, shortage cost, disruption probability,
    and disruption duration are supplied to every strategy.

    This is important because the optimiser baseline must compare strategies
    under a common parameterisation.
    """

    print("\n" + "=" * 70)
    print("SUPPLIER DISRUPTION OPTIMISER")
    print("=" * 70)

    print(
        f"Disruption probability: "
        f"{disruption_probability * 100:.0f}%"
    )
    print(
        f"Disruption duration: "
        f"{disruption_duration_days} days"
    )
    print(
        f"Daily demand: "
        f"{daily_demand}"
    )
    print(
        f"Unit cost: "
        f"£{SUPPLIER_UNIT_COST}"
    )
    print(
        f"Shortage cost/unit: "
        f"£{SUPPLIER_SHORTAGE_COST}"
    )
    print(
        f"Service target: "
        f"{RESILIENCE_SERVICE_TARGET}%"
    )
    print()

    results = []

    for strategy in STRATEGIES:

        print(f"  Evaluating strategy: {strategy}")

        # Use exactly the same numerical assumptions for each strategy.
        #
        # Strategy-specific parameters are supplied only where applicable.
        kwargs = {
            "strategy": strategy,
            "disruption_probability": disruption_probability,
            "disruption_duration_days": disruption_duration_days,
            "daily_demand": daily_demand,
            "unit_cost": SUPPLIER_UNIT_COST,
            "shortage_cost_per_unit": SUPPLIER_SHORTAGE_COST,
        }

        if strategy == "safety_stock":
            kwargs["safety_stock_weeks"] = 4

        elif strategy == "dual_sourcing":
            kwargs["dual_sourcing_premium"] = 0.10

        elif strategy == "air_freight":
            kwargs["air_freight_premium"] = 2.0

        result = run_supplier_disruption_sim(**kwargs)

        r = result["results"]

        row = {
            "strategy": strategy,
            "avg_total_cost_gbp": r["avg_total_cost_gbp"],
            "avg_procurement_cost_gbp": r.get(
                "avg_procurement_cost_gbp"
            ),
            "avg_shortage_cost_gbp": r.get(
                "avg_shortage_cost_gbp"
            ),
            "avg_holding_cost_gbp": r.get(
                "avg_holding_cost_gbp"
            ),
            "avg_strategy_cost_gbp": r.get(
                "avg_strategy_cost_gbp"
            ),
            "avg_service_level_pct": r[
                "avg_service_level_pct"
            ],
            "avg_disruption_days": r.get(
                "avg_disruption_days"
            ),
            "avg_units_short": r.get(
                "avg_units_short"
            ),
            "service_constraint_met": (
                r["avg_service_level_pct"]
                >= RESILIENCE_SERVICE_TARGET
            ),
            "verdict": r.get("verdict"),
        }

        results.append(row)

        print(
            f"    Cost=£{r['avg_total_cost_gbp']:,.2f}  "
            f"SL={r['avg_service_level_pct']:.2f}%"
        )

    # -----------------------------------------------------------------------
    # Select lowest-cost feasible strategy
    # -----------------------------------------------------------------------

    feasible = [
        row
        for row in results
        if row["service_constraint_met"]
    ]

    if feasible:

        best = min(
            feasible,
            key=lambda row: (
                row["avg_total_cost_gbp"],
                -row["avg_service_level_pct"],
            ),
        )

        selection_status = (
            "lowest_cost_strategy_meeting_95_percent_service"
        )

    else:

        # If no strategy meets the constraint, select the highest-service
        # strategy and use cost as a tie-break.
        best = max(
            results,
            key=lambda row: (
                row["avg_service_level_pct"],
                -row["avg_total_cost_gbp"],
            ),
        )

        selection_status = (
            "highest_service_strategy_no_feasible_strategy"
        )

    search_path = os.path.join(
        OUTPUT_DIR,
        "disruption_search.csv",
    )

    save_csv(search_path, results)

    print("\nDisruption optimiser result")
    print("-" * 70)

    print(
        f"Selected strategy: "
        f"{best['strategy']}"
    )
    print(
        f"Average total cost: "
        f"£{best['avg_total_cost_gbp']:,.2f}"
    )
    print(
        f"Average service level: "
        f"{best['avg_service_level_pct']:.2f}%"
    )
    print(
        f"95% service constraint met: "
        f"{best['service_constraint_met']}"
    )
    print(
        f"Selection status: "
        f"{selection_status}"
    )
    print(f"Full search saved to: {search_path}")

    return best


# ===========================================================================
# Summary
# ===========================================================================

def print_comparison_summary(
    inv_opt,
    inv_feasible,
    hub_opt,
    dis_opt,
):
    """
    Print a factual three-way comparison.

    This function deliberately avoids hard-coded claims about the LLM
    baselines. Those claims should be calculated from the actual experiment
    traces/results rather than manually embedded in the optimiser.
    """

    print("\n" + "=" * 70)
    print("THREE-WAY COMPARISON SUMMARY")
    print("Standalone LLM | Standalone Optimiser | LLM + Simulation")
    print("=" * 70)

    print("\nExperiment 1 — Inventory Policy")
    print("-" * 70)

    print(
        "Standalone optimiser:"
    )
    print(
        f"  ROP={inv_opt['reorder_point']}, "
        f"OQ={inv_opt['order_quantity']}"
    )
    print(
        f"  Cost=£{inv_opt['cost']:,.2f}"
    )
    print(
        f"  Service={inv_opt['service_level']:.2f}%"
    )
    print(
        f"  95% constraint met={inv_feasible}"
    )
    print(
        "  Interpretation: lowest-cost feasible policy "
        "within the evaluated search space."
    )

    print("\nExperiment 2 — Hub Location")
    print("-" * 70)

    print(
        f"Standalone optimiser:"
    )
    print(
        f"  Location={hub_opt['location']}"
    )
    print(
        f"  Growth={hub_opt['growth_rate'] * 100:.0f}%"
    )
    print(
        f"  Average NPV=£{hub_opt['npv']:.3f}M"
    )
    print(
        f"  Probability profitable={hub_opt['probability']}%"
    )
    print(
        f"  Average break-even={hub_opt['breakeven']}"
    )
    print(
        "  Interpretation: highest-NPV scenario "
        "within the evaluated location/growth search space."
    )

    print("\nExperiment 3 — Disruption Strategy")
    print("-" * 70)

    print(
        f"Standalone optimiser:"
    )
    print(
        f"  Strategy={dis_opt['strategy']}"
    )
    print(
        f"  Cost=£{dis_opt['avg_total_cost_gbp']:,.2f}"
    )
    print(
        f"  Service={dis_opt['avg_service_level_pct']:.2f}%"
    )
    print(
        f"  95% constraint met={dis_opt['service_constraint_met']}"
    )
    print(
        "  Interpretation: lowest-cost feasible strategy "
        "within the evaluated strategy set."
    )

    print("\n" + "=" * 70)


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("STANDALONE OPTIMISER BASELINE")
    print("No LLM — exhaustive simulation search only")
    print("=" * 70)

    print("\nCanonical configuration:")
    print(f"  Inventory trials: {INVENTORY_TRIALS}")
    print(f"  Inventory seed:   {INVENTORY_SEED}")
    print(f"  Inventory target: {INVENTORY_SERVICE_TARGET}%")
    print(f"  Hub discount:     {HUB_DISCOUNT_RATE:.0%}")
    print(f"  Hub horizon:      {HUB_YEARS} years")
    print(
        f"  Disruption probability: "
        f"{DISRUPTION_PROBABILITY:.0%}"
    )
    print(
        f"  Disruption duration: "
        f"{DISRUPTION_DURATION_DAYS} days"
    )

    # -----------------------------------------------------------------------
    # Experiment 1
    # -----------------------------------------------------------------------

    inv_opt, inv_feasible = optimise_inventory(
        demand_mean=200,
        demand_std=40,
        lead_time_days=7,
        trials=INVENTORY_TRIALS,
        seed=INVENTORY_SEED,
    )

    # -----------------------------------------------------------------------
    # Experiment 2
    # -----------------------------------------------------------------------

    hub_opt = optimise_hub_location()

    # -----------------------------------------------------------------------
    # Experiment 3
    # -----------------------------------------------------------------------

    dis_opt = optimise_disruption(
        disruption_probability=DISRUPTION_PROBABILITY,
        disruption_duration_days=DISRUPTION_DURATION_DAYS,
        daily_demand=SUPPLIER_DAILY_DEMAND,
    )

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------

    print_comparison_summary(
        inv_opt,
        inv_feasible,
        hub_opt,
        dis_opt,
    )

    # -----------------------------------------------------------------------
    # Save machine-readable summary
    # -----------------------------------------------------------------------

    summary = {
        "timestamp": datetime.now().isoformat(),

        "configuration": {
            "inventory_trials": INVENTORY_TRIALS,
            "inventory_seed": INVENTORY_SEED,
            "inventory_service_target_pct": (
                INVENTORY_SERVICE_TARGET
            ),
            "hub_discount_rate": HUB_DISCOUNT_RATE,
            "hub_years": HUB_YEARS,
            "disruption_probability": DISRUPTION_PROBABILITY,
            "disruption_duration_days": (
                DISRUPTION_DURATION_DAYS
            ),
            "supplier_daily_demand": SUPPLIER_DAILY_DEMAND,
            "supplier_unit_cost": SUPPLIER_UNIT_COST,
            "supplier_shortage_cost_per_unit": (
                SUPPLIER_SHORTAGE_COST
            ),
            "resilience_service_target_pct": (
                RESILIENCE_SERVICE_TARGET
            ),
        },

        "inventory_optimiser": {
            "reorder_point": inv_opt["reorder_point"],
            "order_quantity": inv_opt["order_quantity"],
            "cost": inv_opt["cost"],
            "service_level": inv_opt["service_level"],
            "feasible": inv_feasible,
            "selection_status": (
                "lowest_cost_feasible_policy"
                if inv_feasible
                else "highest_service_policy_no_feasible_policy"
            ),
        },

        "hub_optimiser": {
            "location": hub_opt["location"],
            "growth_rate": hub_opt["growth_rate"],
            "npv": hub_opt["npv"],
            "probability_profitable": hub_opt["probability"],
            "breakeven": hub_opt["breakeven"],
            "npv_p10": hub_opt["npv_p10"],
            "npv_p90": hub_opt["npv_p90"],
            "recommendation": hub_opt["recommendation"],
        },

        "disruption_optimiser": {
            key: value
            for key, value in dis_opt.items()
        },
    }

    summary_path = os.path.join(
        OUTPUT_DIR,
        "optimiser_summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"\nSummary saved to: "
        f"{summary_path}"
    )

    print("\n" + "=" * 70)
    print("STANDALONE OPTIMISER COMPLETE")
    print("=" * 70)
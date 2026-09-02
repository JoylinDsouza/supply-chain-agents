# simulations/supplier_disruption_sim.py
# Simulation 3: Supplier Disruption and Resilience
# Fully independent module. All parameters exposed as function arguments.
# Tests four backup strategies under different disruption scenarios.
# Tracks: procurement cost, shortage cost, holding cost, strategy cost,
# total cost, service level.
#
# FIXES APPLIED:
# Fix 1 — Disruption probability:
#          daily hazard is derived from annual probability using
#          1-(1-p)^(1/365).
#
# Fix 2 — Safety stock double-charging removed:
#          safety-stock units are added to inventory and therefore incur
#          holding cost through the normal inventory calculation only.
#
# Fix 3 — Dual sourcing model clarified:
#          a daily retainer is charged for maintaining the backup supplier.
#          Emergency backup procurement is charged separately.
#
# Fix 4 — Procurement costs included:
#          normal primary-supplier purchases are now included in total cost.
#          Backup and air-freight procurement include their appropriate
#          procurement cost and premium.
#
# Fix 5 — Reproducibility:
#          canonical Monte Carlo configuration is 1,000 trials with seed 42.
#          A local NumPy random generator is used rather than the global
#          random-number state.


import numpy as np
import csv
import os


# ── Canonical experiment configuration ────────────────────────────────────────
DEFAULT_TRIALS = 1000
DEFAULT_SEED = 42

# ── Strategy definitions ──────────────────────────────────────────────────────
STRATEGIES = [
    "no_backup",
    "safety_stock",
    "dual_sourcing",
    "air_freight"
]


def run_supplier_disruption_sim(
    strategy="no_backup",
    daily_demand=200,
    demand_std=40,
    normal_lead_time=14,
    disruption_probability=0.20,
    disruption_duration_days=42,
    disruption_duration_std=7,
    unit_cost=12.0,
    shortage_cost_per_unit=8.0,
    holding_cost_per_unit_day=0.50,
    simulation_days=365,
    safety_stock_weeks=4,
    dual_sourcing_premium=0.15,
    dual_sourcing_capacity_pct=0.60,
    air_freight_premium=2.5,
    air_freight_lead_time=3,
    trials=DEFAULT_TRIALS,
    seed=DEFAULT_SEED
):
    """
    Simulates a supply chain facing probabilistic supplier disruption.

    Four resilience strategies are evaluated:

        no_backup
            No contingency supply during a disruption.

        safety_stock
            Additional inventory is held as a buffer.

        dual_sourcing
            A secondary supplier is maintained on contract.
            During disruption, the secondary supplier can fulfil a
            proportion of replenishment demand at a premium.

        air_freight
            During disruption, replenishment is switched to air freight
            with a shorter lead time and higher procurement cost.

    Cost accounting
    ---------------
    Total cost consists of:

        procurement cost
        + holding cost
        + shortage cost
        + strategy cost

    Normal primary-supplier purchases are charged at unit_cost.

    Dual-sourcing emergency purchases are charged at:
        unit_cost * (1 + dual_sourcing_premium)

    Air-freight purchases are charged at:
        unit_cost * air_freight_premium

    The dual-sourcing retainer is reported separately as a strategy cost.

    Parameters
    ----------
    strategy                  : backup strategy
    daily_demand              : average daily demand (units)
    demand_std                : daily demand variability
    normal_lead_time          : normal supplier lead time (days)
    disruption_probability    : annual disruption probability (0-1)
    disruption_duration_days  : average disruption duration (days)
    disruption_duration_std   : disruption duration standard deviation
    unit_cost                 : normal procurement cost per unit (GBP)
    shortage_cost_per_unit    : cost per unit of unmet demand (GBP)
    holding_cost_per_unit_day : inventory holding cost per unit per day (GBP)
    simulation_days           : simulation horizon
    safety_stock_weeks        : weeks of additional safety stock
    dual_sourcing_premium     : backup supplier procurement premium
    dual_sourcing_capacity_pct
                              : fraction of replenishment covered by backup
    air_freight_premium       : air-freight cost multiplier
    air_freight_lead_time     : air-freight lead time (days)
    trials                    : Monte Carlo trials
    seed                      : random seed for reproducibility

    Returns
    -------
    dict
        Parameters and averaged simulation KPIs.
    """

    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose one of: {', '.join(STRATEGIES)}"
        )

    if not 0 <= disruption_probability <= 1:
        raise ValueError("disruption_probability must be between 0 and 1.")

    if not 0 <= dual_sourcing_capacity_pct <= 1:
        raise ValueError(
            "dual_sourcing_capacity_pct must be between 0 and 1."
        )

    if trials <= 0:
        raise ValueError("trials must be greater than zero.")

    if simulation_days <= 0:
        raise ValueError("simulation_days must be greater than zero.")

    # ── Strategy-specific inventory parameters ────────────────────────────────

    # Safety stock is physically added to inventory.
    # Its holding cost is therefore captured by the normal inventory
    # holding-cost calculation. No additional safety-stock charge is applied.
    safety_stock_units = (
        daily_demand * 7 * safety_stock_weeks
        if strategy == "safety_stock"
        else 0
    )

    # Dual sourcing uses Model A:
    # a small daily retainer maintains access to the backup supplier.
    dual_sourcing_daily_retainer = (
        daily_demand
        * dual_sourcing_capacity_pct
        * unit_cost
        * dual_sourcing_premium
        * 0.10
        if strategy == "dual_sourcing"
        else 0
    )

    # ── Convert annual disruption probability to daily hazard ────────────────
    #
    # P(disruption during year) = 1 - (1 - daily_hazard)^365
    #
    # Therefore:
    #
    # daily_hazard = 1 - (1 - annual_probability)^(1/365)
    #
    daily_hazard = (
        1 - (1 - disruption_probability) ** (1 / 365)
    )

    # ── Accumulators ──────────────────────────────────────────────────────────

    all_total_costs = []
    all_procurement_costs = []
    all_shortage_costs = []
    all_holding_costs = []
    all_strategy_costs = []
    all_service_levels = []
    all_disruption_days = []
    all_units_short = []

    # Local generator makes the experiment reproducible without modifying
    # NumPy's global random-number state.
    rng = np.random.default_rng(seed)

    # ── Monte Carlo trials ─────────────────────────────────────────────────────

    for _ in range(trials):

        # Starting inventory:
        # one week's demand plus optional safety stock.
        inventory = daily_demand * 7

        if strategy == "safety_stock":
            inventory += safety_stock_units

        # Trial-level accumulators
        total_procurement_cost = 0.0
        total_shortage_cost = 0.0
        total_strategy_cost = 0.0
        total_holding_cost = 0.0

        stockout_days = 0
        units_short_total = 0

        orders_pending = []

        disruption_active = False
        disruption_days_remaining = 0
        total_disruption_days = 0

        # ── Daily simulation ──────────────────────────────────────────────────

        for day in range(simulation_days):

            # 1. Check whether a new disruption starts today.
            if not disruption_active and rng.random() < daily_hazard:
                disruption_active = True

                actual_duration = max(
                    7,
                    int(
                        rng.normal(
                            disruption_duration_days,
                            disruption_duration_std
                        )
                    )
                )

                disruption_days_remaining = actual_duration

            # 2. Count the active disruption day.
            if disruption_active:
                disruption_days_remaining -= 1
                total_disruption_days += 1

                if disruption_days_remaining <= 0:
                    disruption_active = False

            # 3. Receive arriving replenishment orders.
            arrived = [
                (arrival_day, quantity)
                for arrival_day, quantity in orders_pending
                if arrival_day <= day
            ]

            for _, quantity in arrived:
                inventory += quantity

            orders_pending = [
                (arrival_day, quantity)
                for arrival_day, quantity in orders_pending
                if arrival_day > day
            ]

            # 4. Generate daily demand.
            demand = max(
                0,
                int(rng.normal(daily_demand, demand_std))
            )

            # 5. Serve demand from inventory.
            if inventory >= demand:
                inventory -= demand

            else:
                short = demand - inventory

                units_short_total += short
                total_shortage_cost += (
                    short * shortage_cost_per_unit
                )

                stockout_days += 1
                inventory = 0

            # 6. Holding cost on end-of-day inventory.
            total_holding_cost += (
                inventory * holding_cost_per_unit_day
            )

            # 7. Strategy-specific daily retainer.
            if strategy == "dual_sourcing":
                total_strategy_cost += dual_sourcing_daily_retainer

            # 8. Determine whether replenishment is required.
            reorder_point = (
                daily_demand
                * normal_lead_time
                * 1.2
            )

            if strategy == "safety_stock":
                reorder_point += safety_stock_units

            if inventory <= reorder_point and not orders_pending:

                order_qty = (
                    daily_demand
                    * normal_lead_time
                    * 2
                )

                # ── Normal primary supplier ──────────────────────────────────
                if not disruption_active:

                    lead = max(
                        1,
                        int(
                            rng.normal(
                                normal_lead_time,
                                2
                            )
                        )
                    )

                    orders_pending.append(
                        (day + lead, order_qty)
                    )

                    # Normal procurement cost.
                    total_procurement_cost += (
                        order_qty * unit_cost
                    )

                # ── Dual sourcing ────────────────────────────────────────────
                elif strategy == "dual_sourcing":

                    backup_qty = (
                        order_qty
                        * dual_sourcing_capacity_pct
                    )

                    lead = max(
                        1,
                        int(
                            rng.normal(
                                normal_lead_time + 3,
                                3
                            )
                        )
                    )

                    orders_pending.append(
                        (day + lead, backup_qty)
                    )

                    # Full backup procurement cost.
                    # The base procurement component goes into procurement
                    # cost, while the incremental premium is reported as a
                    # strategy cost.
                    backup_base_cost = (
                        backup_qty * unit_cost
                    )

                    backup_premium_cost = (
                        backup_qty
                        * unit_cost
                        * dual_sourcing_premium
                    )

                    total_procurement_cost += backup_base_cost
                    total_strategy_cost += backup_premium_cost

                # ── Air freight ──────────────────────────────────────────────
                elif strategy == "air_freight":

                    lead = max(
                        1,
                        int(
                            rng.normal(
                                air_freight_lead_time,
                                1
                            )
                        )
                    )

                    orders_pending.append(
                        (day + lead, order_qty)
                    )

                    # Air freight procurement cost.
                    # The full cost is procurement cost because the order
                    # is actually purchased and transported by air.
                    air_procurement_cost = (
                        order_qty
                        * unit_cost
                        * air_freight_premium
                    )

                    total_procurement_cost += (
                        air_procurement_cost
                    )

                # ── No backup ─────────────────────────────────────────────────
                # During disruption no order can be placed.
                # Existing inventory must therefore absorb the disruption.
                elif strategy == "no_backup":
                    pass

        # ── Trial-level total cost and service ────────────────────────────────

        total_cost = (
            total_procurement_cost
            + total_shortage_cost
            + total_holding_cost
            + total_strategy_cost
        )

        service_level = (
            1 - stockout_days / simulation_days
        ) * 100

        # Store trial results
        all_total_costs.append(total_cost)
        all_procurement_costs.append(total_procurement_cost)
        all_shortage_costs.append(total_shortage_cost)
        all_holding_costs.append(total_holding_cost)
        all_strategy_costs.append(total_strategy_cost)
        all_service_levels.append(round(service_level, 1))
        all_disruption_days.append(total_disruption_days)
        all_units_short.append(units_short_total)

    # ── Averaging helper ──────────────────────────────────────────────────────

    def avg(values):
        return round(
            sum(values) / len(values),
            2
        )

    avg_total = avg(all_total_costs)
    avg_service = avg(all_service_levels)

    # ── Return structured result ──────────────────────────────────────────────

    return {
        "simulation": "supplier_disruption",

        "parameters": {
            "strategy": strategy,
            "disruption_probability": disruption_probability,
            "effective_daily_hazard": round(
                daily_hazard,
                6
            ),
            "disruption_duration_days": disruption_duration_days,
            "daily_demand": daily_demand,
            "unit_cost": unit_cost,
            "trials": trials,
            "seed": seed
        },

        "results": {
            "avg_total_cost_gbp": avg_total,
            "avg_procurement_cost_gbp": avg(
                all_procurement_costs
            ),
            "avg_shortage_cost_gbp": avg(
                all_shortage_costs
            ),
            "avg_holding_cost_gbp": avg(
                all_holding_costs
            ),
            "avg_strategy_cost_gbp": avg(
                all_strategy_costs
            ),
            "avg_service_level_pct": avg_service,
            "avg_disruption_days": avg(
                all_disruption_days
            ),
            "avg_units_short": avg(
                all_units_short
            ),
            "verdict": (
                "good"
                if avg_service >= 90
                else "needs improvement"
            )
        }
    }


def run_strategy_comparison(
    disruption_probabilities=None,
    disruption_duration_days=42,
    daily_demand=200,
    unit_cost=12.0,
    trials=DEFAULT_TRIALS,
    seed=DEFAULT_SEED,
    save_csv=True,
    csv_path="results/simulation3_results.csv"
):
    """
    Compares all four resilience strategies across a range of
    annual disruption probabilities.

    Returns a list of flattened result dictionaries suitable for
    CSV output and visualisation.
    """

    if disruption_probabilities is None:
        disruption_probabilities = [
            0.05,
            0.10,
            0.15,
            0.20,
            0.30,
            0.40,
            0.50
        ]

    os.makedirs(
        os.path.dirname(csv_path),
        exist_ok=True
    )

    all_results = []

    for prob in disruption_probabilities:

        for strategy in STRATEGIES:

            result = run_supplier_disruption_sim(
                strategy=strategy,
                disruption_probability=prob,
                disruption_duration_days=disruption_duration_days,
                daily_demand=daily_demand,
                unit_cost=unit_cost,
                trials=trials,
                seed=seed
            )

            flat = {
                "disruption_probability_pct": round(
                    prob * 100,
                    0
                ),
                "strategy": strategy,
                "avg_total_cost_gbp": result[
                    "results"
                ]["avg_total_cost_gbp"],
                "avg_procurement_cost_gbp": result[
                    "results"
                ]["avg_procurement_cost_gbp"],
                "avg_shortage_cost_gbp": result[
                    "results"
                ]["avg_shortage_cost_gbp"],
                "avg_holding_cost_gbp": result[
                    "results"
                ]["avg_holding_cost_gbp"],
                "avg_strategy_cost_gbp": result[
                    "results"
                ]["avg_strategy_cost_gbp"],
                "avg_service_level_pct": result[
                    "results"
                ]["avg_service_level_pct"],
                "avg_disruption_days": result[
                    "results"
                ]["avg_disruption_days"],
                "avg_units_short": result[
                    "results"
                ]["avg_units_short"],
                "verdict": result[
                    "results"
                ]["verdict"]
            }

            all_results.append(flat)

    if save_csv:

        with open(
            csv_path,
            "w",
            newline=""
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=all_results[0].keys()
            )

            writer.writeheader()
            writer.writerows(all_results)

        print(
            f"Results saved to {csv_path}"
        )

    return all_results


# ── Quick test when run directly ──────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 76)
    print("Simulation 3: Supplier Disruption and Resilience")
    print("=" * 76)

    print("\nCanonical Monte Carlo configuration:")
    print(f"  Trials: {DEFAULT_TRIALS}")
    print(f"  Seed:   {DEFAULT_SEED}")

    # ── Verify daily hazard calculation ───────────────────────────────────────

    print("\nEffective daily disruption hazard rates:")
    print("-" * 76)

    for p in [
        0.05,
        0.10,
        0.20,
        0.25,
        0.30,
        0.50
    ]:

        h = 1 - (1 - p) ** (1 / 365)

        effective = (
            1 - (1 - h) ** 365
        )

        print(
            f"  Annual p={p:.2f} "
            f"→ daily hazard={h:.6f} "
            f"→ mathematical annual probability={effective:.4f} ✓"
        )

    # ── Run main comparison ──────────────────────────────────────────────────

    disruption_probs = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50
    ]

    print(
        "\nComparing 4 resilience strategies "
        "across disruption probabilities..."
    )

    print("-" * 100)

    print(
        f"{'Prob':>6} | "
        f"{'Strategy':<16} | "
        f"{'Total Cost':>13} | "
        f"{'Procurement':>13} | "
        f"{'Shortage':>11} | "
        f"{'Holding':>11} | "
        f"{'Strategy':>11} | "
        f"{'Service':>8}"
    )

    print("-" * 100)

    results = run_strategy_comparison(
        disruption_probabilities=disruption_probs,
        disruption_duration_days=42,
        daily_demand=200,
        unit_cost=12.0,
        trials=DEFAULT_TRIALS,
        seed=DEFAULT_SEED,
        save_csv=True
    )

    current_prob = None

    for r in results:

        if (
            r["disruption_probability_pct"]
            != current_prob
        ):

            if current_prob is not None:
                print()

            current_prob = (
                r["disruption_probability_pct"]
            )

        print(
            f"{r['disruption_probability_pct']:>5.0f}% | "
            f"{r['strategy']:<16} | "
            f"£{r['avg_total_cost_gbp']:>12,.0f} | "
            f"£{r['avg_procurement_cost_gbp']:>12,.0f} | "
            f"£{r['avg_shortage_cost_gbp']:>10,.0f} | "
            f"£{r['avg_holding_cost_gbp']:>10,.0f} | "
            f"£{r['avg_strategy_cost_gbp']:>10,.0f} | "
            f"{r['avg_service_level_pct']:>7.1f}%"
        )

    print("\nDone.")
    print(
        "Results saved to results/simulation3_results.csv"
    )
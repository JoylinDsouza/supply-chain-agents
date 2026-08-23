# simulations/supplier_disruption_sim.py
# Simulation 3: Supplier Disruption and Resilience
# Fully independent module. All parameters exposed as function arguments.
# Tests three backup strategies under different disruption scenarios.
# Tracks: lost sales cost, emergency procurement cost, dual sourcing premium,
#         safety stock holding cost, total disruption cost.

import numpy as np
import csv
import os


# ── Strategy definitions ──────────────────────────────────────────────────────
STRATEGIES = ["no_backup", "safety_stock", "dual_sourcing", "air_freight"]


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
    trials=50
):
    """
    Simulates a supply chain facing a probabilistic supplier disruption.
    Tests one of four backup strategies and returns cost and performance metrics.

    Parameters
    ----------
    strategy                  : backup strategy — one of:
                                 'no_backup'     — no contingency plan
                                 'safety_stock'  — hold extra weeks of stock
                                 'dual_sourcing' — second supplier at premium
                                 'air_freight'   — switch to air freight during disruption
    daily_demand              : average daily demand (units)
    demand_std                : daily demand variability (std dev)
    normal_lead_time          : standard supplier lead time (days)
    disruption_probability    : probability of a disruption occurring per year (0-1)
    disruption_duration_days  : average length of a disruption (days)
    disruption_duration_std   : variability in disruption length (days)
    unit_cost                 : cost per unit from primary supplier (GBP)
    shortage_cost_per_unit    : cost per unit of unmet demand (GBP)
    holding_cost_per_unit_day : cost to hold one unit for one day (GBP)
    simulation_days           : number of days to simulate
    safety_stock_weeks        : weeks of extra stock held (safety_stock strategy)
    dual_sourcing_premium     : price premium for backup supplier (0.15 = 15% more)
    dual_sourcing_capacity_pct: fraction of demand backup supplier can cover (0-1)
    air_freight_premium       : cost multiplier for air freight vs sea (e.g. 2.5 = 2.5x)
    air_freight_lead_time     : lead time in days when using air freight
    trials                    : Monte Carlo trials

    Returns
    -------
    dict with parameters, averaged KPIs, and cost breakdown
    """

    # Pre-compute strategy-specific costs
    # Safety stock holding cost — paid every day regardless of disruption
    safety_stock_units = (
        daily_demand * 7 * safety_stock_weeks
        if strategy == "safety_stock" else 0
    )
    daily_safety_stock_cost = safety_stock_units * holding_cost_per_unit_day

    # Dual sourcing premium — paid on fraction of daily procurement
    dual_sourcing_daily_premium = (
        daily_demand * dual_sourcing_capacity_pct
        * unit_cost * dual_sourcing_premium
        if strategy == "dual_sourcing" else 0
    )

    # Accumulators
    all_total_costs = []
    all_shortage_costs = []
    all_disruption_costs = []
    all_strategy_costs = []
    all_service_levels = []
    all_disruption_days = []
    all_units_short = []

    for _ in range(trials):
        inventory = daily_demand * 7  # start with 7x daily demand stock
        if strategy == "safety_stock":
            inventory += safety_stock_units

        total_shortage_cost = 0.0
        total_strategy_cost = 0.0
        total_holding_cost = 0.0
        stockout_days = 0
        units_short_total = 0
        orders_pending = []
        disruption_active = False
        disruption_days_remaining = 0
        total_disruption_days = 0
        orders_placed = 0

        for day in range(simulation_days):

            # 1. Check if a new disruption starts today
            if not disruption_active and np.random.random() < disruption_probability / 52:
                disruption_active = True
                actual_duration = max(
                    7,
                    int(np.random.normal(disruption_duration_days, disruption_duration_std))
                )
                disruption_days_remaining = actual_duration

            # 2. Count down disruption
            if disruption_active:
                disruption_days_remaining -= 1
                total_disruption_days += 1
                if disruption_days_remaining <= 0:
                    disruption_active = False

            # 3. Receive arriving orders
            arrived = [(d, q) for (d, q) in orders_pending if d <= day]
            for (_, qty) in arrived:
                inventory += qty
            orders_pending = [(d, q) for (d, q) in orders_pending if d > day]

            # 4. Daily demand
            demand = max(0, int(np.random.normal(daily_demand, demand_std)))

            # 5. Serve demand
            if inventory >= demand:
                inventory -= demand
            else:
                short = demand - inventory
                units_short_total += short
                total_shortage_cost += short * shortage_cost_per_unit
                stockout_days += 1
                inventory = 0

            # 6. Holding cost
            total_holding_cost += inventory * holding_cost_per_unit_day

            # 7. Apply strategy-specific daily cost
            if strategy == "safety_stock":
                total_strategy_cost += daily_safety_stock_cost
            elif strategy == "dual_sourcing":
                total_strategy_cost += dual_sourcing_daily_premium

            # 8. Place orders based on strategy and disruption status
            reorder_point = daily_demand * normal_lead_time * 1.2
            if strategy == "safety_stock":
                reorder_point += safety_stock_units

            if inventory <= reorder_point and not orders_pending:
                order_qty = daily_demand * normal_lead_time * 2

                if not disruption_active:
                    # Normal order from primary supplier
                    lead = max(1, int(np.random.normal(normal_lead_time, 2)))
                    orders_pending.append((day + lead, order_qty))
                    orders_placed += 1

                elif strategy == "dual_sourcing":
                    # Backup supplier covers partial demand
                    backup_qty = order_qty * dual_sourcing_capacity_pct
                    lead = max(1, int(np.random.normal(normal_lead_time + 3, 3)))
                    orders_pending.append((day + lead, backup_qty))
                    total_strategy_cost += backup_qty * unit_cost * dual_sourcing_premium
                    orders_placed += 1

                elif strategy == "air_freight":
                    # Switch to air freight — faster but much more expensive
                    air_cost_premium = order_qty * unit_cost * (air_freight_premium - 1)
                    total_strategy_cost += air_cost_premium
                    lead = max(1, int(np.random.normal(air_freight_lead_time, 1)))
                    orders_pending.append((day + lead, order_qty))
                    orders_placed += 1
                # no_backup: cannot order during disruption — shortages accumulate

        total_cost = total_shortage_cost + total_holding_cost + total_strategy_cost
        service_level = round((1 - stockout_days / simulation_days) * 100, 1)

        all_total_costs.append(total_cost)
        all_shortage_costs.append(total_shortage_cost)
        all_disruption_costs.append(total_holding_cost)
        all_strategy_costs.append(total_strategy_cost)
        all_service_levels.append(service_level)
        all_disruption_days.append(total_disruption_days)
        all_units_short.append(units_short_total)

    def avg(lst):
        return round(sum(lst) / len(lst), 2)

    avg_total = avg(all_total_costs)
    no_backup_baseline = daily_demand * disruption_duration_days * shortage_cost_per_unit

    return {
        "simulation": "supplier_disruption",
        "parameters": {
            "strategy": strategy,
            "disruption_probability": disruption_probability,
            "disruption_duration_days": disruption_duration_days,
            "daily_demand": daily_demand,
            "unit_cost": unit_cost,
            "trials": trials
        },
        "results": {
            "avg_total_cost_gbp": avg_total,
            "avg_shortage_cost_gbp": avg(all_shortage_costs),
            "avg_holding_cost_gbp": avg(all_disruption_costs),
            "avg_strategy_cost_gbp": avg(all_strategy_costs),
            "avg_service_level_pct": avg(all_service_levels),
            "avg_disruption_days": avg(all_disruption_days),
            "avg_units_short": avg(all_units_short),
            "verdict": (
                "good" if avg(all_service_levels) >= 90 else "needs improvement"
            )
        }
    }


def run_strategy_comparison(
    disruption_probabilities=None,
    disruption_duration_days=42,
    daily_demand=200,
    unit_cost=12.0,
    trials=50,
    save_csv=True,
    csv_path="results/simulation3_results.csv"
):
    """
    Compares all four strategies across a range of disruption probabilities.
    Returns a list of result dicts.
    """
    if disruption_probabilities is None:
        disruption_probabilities = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    all_results = []

    for prob in disruption_probabilities:
        for strategy in STRATEGIES:
            result = run_supplier_disruption_sim(
                strategy=strategy,
                disruption_probability=prob,
                disruption_duration_days=disruption_duration_days,
                daily_demand=daily_demand,
                unit_cost=unit_cost,
                trials=trials
            )
            flat = {
                "disruption_probability_pct": round(prob * 100, 0),
                "strategy": strategy,
                "avg_total_cost_gbp": result["results"]["avg_total_cost_gbp"],
                "avg_shortage_cost_gbp": result["results"]["avg_shortage_cost_gbp"],
                "avg_holding_cost_gbp": result["results"]["avg_holding_cost_gbp"],
                "avg_strategy_cost_gbp": result["results"]["avg_strategy_cost_gbp"],
                "avg_service_level_pct": result["results"]["avg_service_level_pct"],
                "avg_disruption_days": result["results"]["avg_disruption_days"],
                "verdict": result["results"]["verdict"]
            }
            all_results.append(flat)

    if save_csv:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"Results saved to {csv_path}")

    return all_results


# ── Quick test when run directly ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 72)
    print("Simulation 3: Supplier Disruption and Resilience")
    print("=" * 72)

    disruption_probs = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

    print("\nComparing 4 strategies across disruption probabilities...")
    print("-" * 72)
    print(f"{'Prob':>6} | {'Strategy':<16} | {'Total Cost':>11} | "
          f"{'Service':>8} | {'Shortage':>10} | {'Strategy Cost':>13}")
    print("-" * 72)

    results = run_strategy_comparison(
        disruption_probabilities=disruption_probs,
        disruption_duration_days=42,
        daily_demand=200,
        trials=50,
        save_csv=True
    )

    current_prob = None
    for r in results:
        if r["disruption_probability_pct"] != current_prob:
            if current_prob is not None:
                print()
            current_prob = r["disruption_probability_pct"]

        print(
            f"{r['disruption_probability_pct']:>5.0f}% | "
            f"{r['strategy']:<16} | "
            f"£{r['avg_total_cost_gbp']:>10,.0f} | "
            f"{r['avg_service_level_pct']:>7.1f}% | "
            f"£{r['avg_shortage_cost_gbp']:>9,.0f} | "
            f"£{r['avg_strategy_cost_gbp']:>12,.0f}"
        )

    print("\nDone. Results saved to results/simulation3_results.csv")
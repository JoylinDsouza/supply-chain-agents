# simulations/inventory_sim.py
# Simulation 1: Inventory Replenishment — Complete Version
# Fully independent module. All parameters exposed as function arguments.
# Tracks: holding cost, ordering cost, shortage cost, service level.
# Includes: Monte Carlo averaging, seasonal spike, safety stock calculation.

import numpy as np
import csv
import os


def run_inventory_sim(
    demand_mean=200,
    demand_std=40,
    lead_time_days=7,
    lead_time_std=1.5,
    reorder_point=500,
    order_quantity=1000,
    simulation_days=90,
    holding_cost_per_unit=0.50,
    order_fixed_cost=100,
    shortage_cost_per_unit=5.0,
    seasonal_spike=True,
    spike_week=6,
    spike_multiplier=1.4,
    trials=20
):
    """
    Simulates a retail store's inventory replenishment policy.

    Parameters
    ----------
    demand_mean             : average units sold per day
    demand_std              : variability of daily demand (standard deviation)
    lead_time_days          : average days between placing and receiving an order
    lead_time_std           : variability in lead time (standard deviation in days)
    reorder_point           : place a new order when stock falls to this level
    order_quantity          : how many units to order each time
    simulation_days         : number of days to simulate (default 90)
    holding_cost_per_unit   : GBP cost to store one unit for one day
    order_fixed_cost        : GBP fixed cost per order placed (admin + logistics)
    shortage_cost_per_unit  : GBP cost per unit of unmet demand (lost sale penalty)
    seasonal_spike          : whether to include a demand spike period
    spike_week              : week number the spike begins (0-indexed)
    spike_multiplier        : demand multiplier during spike (1.4 = 40% increase)
    trials                  : number of Monte Carlo trials to average

    Returns
    -------
    dict containing parameters, averaged KPIs, cost breakdown, and daily trace
    """

    # --- Accumulators across all trials ---
    all_service_levels = []
    all_stockout_days = []
    all_holding_costs = []
    all_ordering_costs = []
    all_shortage_costs = []
    all_total_costs = []
    all_num_orders = []

    # Store one trial's daily trace for visualisation (the last trial)
    daily_trace = []

    for trial in range(trials):
        inventory = reorder_point * 2      # start with double reorder point
        holding_cost = 0.0
        ordering_cost = 0.0
        shortage_cost = 0.0
        stockout_days = 0
        units_short_total = 0
        num_orders = 0
        orders_pending = []                # list of (arrival_day, quantity)
        trial_trace = []                   # daily inventory levels this trial

        for day in range(simulation_days):

            # 1. Receive any orders due today
            arrived = [(d, q) for (d, q) in orders_pending if d <= day]
            for (_, qty) in arrived:
                inventory += qty
            orders_pending = [(d, q) for (d, q) in orders_pending if d > day]

            # 2. Calculate today's demand (with seasonal spike if enabled)
            week = day // 7
            if seasonal_spike and (week == spike_week or week == spike_week + 1):
                today_mean = demand_mean * spike_multiplier
            else:
                today_mean = demand_mean

            demand = max(0, int(np.random.normal(today_mean, demand_std)))

            # 3. Serve demand — track shortages
            if inventory >= demand:
                inventory -= demand
            else:
                units_short = demand - inventory
                units_short_total += units_short
                shortage_cost += units_short * shortage_cost_per_unit
                stockout_days += 1
                inventory = 0

            # 4. Charge holding cost on remaining inventory
            holding_cost += inventory * holding_cost_per_unit

            # 5. Place a reorder if stock is low and no order pending
            if inventory <= reorder_point and not orders_pending:
                # Variable lead time — more realistic than fixed
                actual_lead_time = max(
                    1,
                    int(np.random.normal(lead_time_days, lead_time_std))
                )
                arrival_day = day + actual_lead_time
                orders_pending.append((arrival_day, order_quantity))
                ordering_cost += order_fixed_cost
                num_orders += 1

            # 6. Record daily inventory for trace (last trial only)
            if trial == trials - 1:
                trial_trace.append({
                    "day": day,
                    "inventory": inventory,
                    "demand": demand,
                    "week": week
                })

        # End of trial — compute service level
        service_level = round((1 - stockout_days / simulation_days) * 100, 1)
        total_cost = holding_cost + ordering_cost + shortage_cost

        all_service_levels.append(service_level)
        all_stockout_days.append(stockout_days)
        all_holding_costs.append(holding_cost)
        all_ordering_costs.append(ordering_cost)
        all_shortage_costs.append(shortage_cost)
        all_total_costs.append(total_cost)
        all_num_orders.append(num_orders)
        daily_trace = trial_trace   # keep last trial's trace

    # --- Average across all trials ---
    def avg(lst):
        return round(sum(lst) / len(lst), 2)

    # --- Safety stock formula (theoretical) ---
    # Z = 1.65 for 95% service level target
    z_score = 1.65
    safety_stock_theoretical = round(
        z_score * demand_std * (lead_time_days ** 0.5), 0
    )
    theoretical_rop = round(demand_mean * lead_time_days + safety_stock_theoretical, 0)

    return {
        "simulation": "inventory_replenishment",
        "parameters": {
            "demand_mean": demand_mean,
            "demand_std": demand_std,
            "lead_time_days": lead_time_days,
            "reorder_point": reorder_point,
            "order_quantity": order_quantity,
            "holding_cost_per_unit": holding_cost_per_unit,
            "order_fixed_cost": order_fixed_cost,
            "shortage_cost_per_unit": shortage_cost_per_unit,
            "seasonal_spike": seasonal_spike,
            "spike_multiplier": spike_multiplier,
            "trials": trials
        },
        "results": {
            "avg_service_level_pct": avg(all_service_levels),
            "avg_stockout_days": avg(all_stockout_days),
            "avg_total_cost_gbp": avg(all_total_costs),
            "avg_holding_cost_gbp": avg(all_holding_costs),
            "avg_ordering_cost_gbp": avg(all_ordering_costs),
            "avg_shortage_cost_gbp": avg(all_shortage_costs),
            "avg_num_orders": avg(all_num_orders),
            "safety_stock_theoretical": safety_stock_theoretical,
            "theoretical_optimal_rop": theoretical_rop,
            "verdict": "good" if avg(all_service_levels) >= 95 else "needs improvement"
        },
        "daily_trace": daily_trace
    }


def run_experiment(
    reorder_points=None,
    order_quantity=1500,
    demand_mean=200,
    demand_std=40,
    lead_time_days=7,
    seasonal_spike=True,
    trials=20,
    save_csv=True,
    csv_path="results/simulation1_results.csv"
):
    """
    Runs the simulation across a range of reorder points and returns all results.
    Optionally saves to CSV.
    """
    if reorder_points is None:
        reorder_points = [200, 400, 600, 800, 1000, 1200]

    # Resolve path relative to project root, not current working directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_csv_path = os.path.join(project_root, csv_path)
    os.makedirs(os.path.dirname(abs_csv_path), exist_ok=True)
    csv_path = abs_csv_path
    all_results = []

    for rp in reorder_points:
        result = run_inventory_sim(
            reorder_point=rp,
            order_quantity=order_quantity,
            demand_mean=demand_mean,
            demand_std=demand_std,
            lead_time_days=lead_time_days,
            seasonal_spike=seasonal_spike,
            trials=trials
        )
        flat = {
            "reorder_point": rp,
            "order_quantity": order_quantity,
            "avg_service_level_pct": result["results"]["avg_service_level_pct"],
            "avg_stockout_days": result["results"]["avg_stockout_days"],
            "avg_total_cost_gbp": result["results"]["avg_total_cost_gbp"],
            "avg_holding_cost_gbp": result["results"]["avg_holding_cost_gbp"],
            "avg_ordering_cost_gbp": result["results"]["avg_ordering_cost_gbp"],
            "avg_shortage_cost_gbp": result["results"]["avg_shortage_cost_gbp"],
            "avg_num_orders": result["results"]["avg_num_orders"],
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


# ── Quick test when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("Simulation 1: Inventory Replenishment — Complete Version")
    print("=" * 65)

    # Print theoretical optimal reorder point
    test = run_inventory_sim(reorder_point=800, order_quantity=1500)
    rop_theory = test["results"]["theoretical_optimal_rop"]
    ss_theory = test["results"]["safety_stock_theoretical"]
    print(f"\nTheoretical safety stock (95% SL): {ss_theory} units")
    print(f"Theoretical optimal ROP:           {rop_theory} units")

    print("\nRunning experiment across 6 reorder points...")
    print("-" * 65)
    print(f"{'ROP':>6} | {'Service':>8} | {'Stockouts':>9} | "
          f"{'Total Cost':>11} | {'Holding':>9} | {'Ordering':>9} | {'Shortage':>9}")
    print("-" * 65)

    results = run_experiment(
        reorder_points=[200, 400, 600, 800, 1000, 1200],
        order_quantity=1500,
        demand_mean=200,
        demand_std=40,
        lead_time_days=7,
        seasonal_spike=True,
        trials=20,
        save_csv=True
    )

    for r in results:
        print(
            f"{r['reorder_point']:>6} | "
            f"{r['avg_service_level_pct']:>7}% | "
            f"{r['avg_stockout_days']:>9} | "
            f"£{r['avg_total_cost_gbp']:>10,.0f} | "
            f"£{r['avg_holding_cost_gbp']:>8,.0f} | "
            f"£{r['avg_ordering_cost_gbp']:>8,.0f} | "
            f"£{r['avg_shortage_cost_gbp']:>8,.0f}"
        )

    print("-" * 65)
    print("\nDone. Results saved to results/simulation1_results.csv") 
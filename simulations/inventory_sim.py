# simulations/inventory_sim.py
# Simulation 1: Inventory Replenishment
#
# Purpose:
#   Discrete-event / Monte Carlo inventory simulation for evaluating
#   reorder-point (ROP) and order-quantity (OQ) policies.
#
# Methodological note:
#   The theoretical ROP is an analytical benchmark.
#   The simulation optimiser selects the lowest-cost feasible policy
#   within the evaluated search space.
#
# Canonical dissertation configuration:
#   Monte Carlo trials = 1000
#   Random seed = 42
#   Service-level target = 95%


import numpy as np
import csv
import os


# ---------------------------------------------------------------------
# Canonical experiment defaults
# ---------------------------------------------------------------------

DEFAULT_TRIALS = 1000
DEFAULT_SEED = 42
DEFAULT_TARGET_SERVICE_LEVEL = 95.0


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
    trials=DEFAULT_TRIALS,
    seed=DEFAULT_SEED
):
    """
    Simulate a retail inventory replenishment policy using Monte Carlo trials.

    The simulation evaluates a fixed ROP/OQ policy over multiple stochastic
    trials and returns averaged operational and cost metrics.

    When seed is supplied, trial i uses seed + i. This provides reproducible
    Monte Carlo results while ensuring that individual trials use distinct
    random-number streams.

    Returns
    -------
    dict
        Simulation parameters, averaged KPIs, cost breakdown,
        theoretical benchmark values, and the daily trace from
        the final Monte Carlo trial.
    """

    if trials <= 0:
        raise ValueError("trials must be greater than zero.")

    if simulation_days <= 0:
        raise ValueError("simulation_days must be greater than zero.")

    if reorder_point < 0:
        raise ValueError("reorder_point must be non-negative.")

    if order_quantity <= 0:
        raise ValueError("order_quantity must be greater than zero.")

    all_service_levels = []
    all_stockout_days = []
    all_holding_costs = []
    all_ordering_costs = []
    all_shortage_costs = []
    all_total_costs = []
    all_num_orders = []
    all_units_short = []
    all_total_demand = []

    daily_trace = []

    # -----------------------------------------------------------------
    # Monte Carlo trials
    # -----------------------------------------------------------------

    for trial_idx in range(trials):

        if seed is not None:
            np.random.seed(seed + trial_idx)

        # Initially stocked system.
        inventory = reorder_point * 2

        holding_cost = 0.0
        ordering_cost = 0.0
        shortage_cost = 0.0

        stockout_days = 0
        units_short_total = 0
        total_demand_trial = 0
        num_orders = 0

        # Only one replenishment order can be outstanding at once.
        orders_pending = []

        trial_trace = []

        # -------------------------------------------------------------
        # Daily simulation
        # -------------------------------------------------------------

        for day in range(simulation_days):

            # ---------------------------------------------------------
            # 1. Receive replenishment orders due today.
            # ---------------------------------------------------------

            arrived = [
                (d, q)
                for (d, q) in orders_pending
                if d <= day
            ]

            for (_, qty) in arrived:
                inventory += qty

            orders_pending = [
                (d, q)
                for (d, q) in orders_pending
                if d > day
            ]

            # ---------------------------------------------------------
            # 2. Generate today's demand.
            # ---------------------------------------------------------

            week = day // 7

            if (
                seasonal_spike
                and (week == spike_week or week == spike_week + 1)
            ):
                today_mean = demand_mean * spike_multiplier
            else:
                today_mean = demand_mean

            demand = max(
                0,
                int(
                    np.random.normal(
                        today_mean,
                        demand_std
                    )
                )
            )

            total_demand_trial += demand

            # ---------------------------------------------------------
            # 3. Satisfy demand.
            # ---------------------------------------------------------

            if inventory >= demand:

                inventory -= demand

            else:

                units_short = demand - inventory

                units_short_total += units_short

                shortage_cost += (
                    units_short
                    * shortage_cost_per_unit
                )

                stockout_days += 1

                inventory = 0

            # ---------------------------------------------------------
            # 4. Apply holding cost.
            # ---------------------------------------------------------

            holding_cost += (
                inventory
                * holding_cost_per_unit
            )

            # ---------------------------------------------------------
            # 5. Replenishment decision.
            #
            # One outstanding order is permitted at a time.
            # ---------------------------------------------------------

            if (
                inventory <= reorder_point
                and not orders_pending
            ):

                actual_lead_time = max(
                    1,
                    int(
                        np.random.normal(
                            lead_time_days,
                            lead_time_std
                        )
                    )
                )

                orders_pending.append(
                    (
                        day + actual_lead_time,
                        order_quantity
                    )
                )

                ordering_cost += order_fixed_cost

                num_orders += 1

            # ---------------------------------------------------------
            # 6. Save trace for final Monte Carlo trial.
            # ---------------------------------------------------------

            if trial_idx == trials - 1:

                trial_trace.append(
                    {
                        "day": day,
                        "inventory": inventory,
                        "demand": demand,
                        "week": week
                    }
                )

        # -------------------------------------------------------------
        # Trial-level KPIs
        # -------------------------------------------------------------

        cycle_service_level = round(
            (
                1
                - stockout_days / simulation_days
            )
            * 100,
            1
        )

        fill_rate = round(
            (
                1
                - units_short_total
                / max(1, total_demand_trial)
            )
            * 100,
            1
        )

        total_cost = (
            holding_cost
            + ordering_cost
            + shortage_cost
        )

        all_service_levels.append(
            cycle_service_level
        )

        all_stockout_days.append(
            stockout_days
        )

        all_holding_costs.append(
            holding_cost
        )

        all_ordering_costs.append(
            ordering_cost
        )

        all_shortage_costs.append(
            shortage_cost
        )

        all_total_costs.append(
            total_cost
        )

        all_num_orders.append(
            num_orders
        )

        all_units_short.append(
            units_short_total
        )

        all_total_demand.append(
            total_demand_trial
        )

        daily_trace = trial_trace

    # -----------------------------------------------------------------
    # Averaging helper
    # -----------------------------------------------------------------

    def avg(values):

        return round(
            sum(values) / len(values),
            2
        )

    # -----------------------------------------------------------------
    # Analytical safety-stock benchmark
    #
    # This is retained as a benchmark only.
    # It is not treated as the simulation optimum.
    # -----------------------------------------------------------------

    z_score = 1.65

    safety_stock_theoretical = round(
        z_score
        * demand_std
        * (lead_time_days ** 0.5),
        0
    )

    theoretical_rop = round(
        demand_mean * lead_time_days
        + safety_stock_theoretical,
        0
    )

    # -----------------------------------------------------------------
    # Monte Carlo averaged KPIs
    # -----------------------------------------------------------------

    avg_sl = avg(
        all_service_levels
    )

    avg_fill = round(
        (
            1
            - avg(all_units_short)
            / max(
                1,
                avg(all_total_demand)
            )
        )
        * 100,
        1
    )

    return {
        "simulation":
            "inventory_replenishment",

        "parameters": {
            "demand_mean":
                demand_mean,

            "demand_std":
                demand_std,

            "lead_time_days":
                lead_time_days,

            "lead_time_std":
                lead_time_std,

            "reorder_point":
                reorder_point,

            "order_quantity":
                order_quantity,

            "simulation_days":
                simulation_days,

            "holding_cost_per_unit":
                holding_cost_per_unit,

            "order_fixed_cost":
                order_fixed_cost,

            "shortage_cost_per_unit":
                shortage_cost_per_unit,

            "seasonal_spike":
                seasonal_spike,

            "spike_week":
                spike_week,

            "spike_multiplier":
                spike_multiplier,

            "trials":
                trials,

            "seed":
                seed
        },

        "results": {

            "avg_service_level_pct":
                avg_sl,

            "avg_fill_rate_pct":
                avg_fill,

            "avg_stockout_days":
                avg(
                    all_stockout_days
                ),

            "avg_total_cost_gbp":
                avg(
                    all_total_costs
                ),

            "avg_holding_cost_gbp":
                avg(
                    all_holding_costs
                ),

            "avg_ordering_cost_gbp":
                avg(
                    all_ordering_costs
                ),

            "avg_shortage_cost_gbp":
                avg(
                    all_shortage_costs
                ),

            "avg_num_orders":
                avg(
                    all_num_orders
                ),

            "safety_stock_theoretical":
                safety_stock_theoretical,

            "theoretical_optimal_rop":
                theoretical_rop,

            "verdict":
                (
                    "good"
                    if avg_sl >= DEFAULT_TARGET_SERVICE_LEVEL
                    else "needs improvement"
                )
        },

        "daily_trace":
            daily_trace
    }


def run_experiment(
    reorder_points=None,
    order_quantities=None,
    demand_mean=200,
    demand_std=40,
    lead_time_days=7,
    seasonal_spike=True,
    trials=DEFAULT_TRIALS,
    save_csv=True,
    csv_path="results/simulation1_results.csv",
    target_service_level=DEFAULT_TARGET_SERVICE_LEVEL,
    seed=DEFAULT_SEED
):
    """
    Evaluate a grid of ROP x OQ policies.

    A policy is feasible when its simulated average service level
    is greater than or equal to target_service_level.

    The function does not claim global optimality.
    """

    if reorder_points is None:

        reorder_points = [
            200,
            400,
            600,
            800,
            1000,
            1200,
            1400,
            1600
        ]

    if order_quantities is None:

        order_quantities = [
            1000,
            1500,
            2000
        ]

    project_root = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    abs_csv_path = os.path.join(
        project_root,
        csv_path
    )

    os.makedirs(
        os.path.dirname(abs_csv_path),
        exist_ok=True
    )

    all_results = []

    # -----------------------------------------------------------------
    # Evaluate every ROP x OQ combination.
    # -----------------------------------------------------------------

    for rp in reorder_points:

        for oq in order_quantities:

            result = run_inventory_sim(
                reorder_point=rp,
                order_quantity=oq,
                demand_mean=demand_mean,
                demand_std=demand_std,
                lead_time_days=lead_time_days,
                seasonal_spike=seasonal_spike,
                trials=trials,
                seed=seed
            )

            service_level = (
                result["results"]
                ["avg_service_level_pct"]
            )

            flat = {

                "reorder_point":
                    rp,

                "order_quantity":
                    oq,

                "avg_service_level_pct":
                    service_level,

                "avg_fill_rate_pct":
                    result["results"]
                    ["avg_fill_rate_pct"],

                "avg_stockout_days":
                    result["results"]
                    ["avg_stockout_days"],

                "avg_total_cost_gbp":
                    result["results"]
                    ["avg_total_cost_gbp"],

                "avg_holding_cost_gbp":
                    result["results"]
                    ["avg_holding_cost_gbp"],

                "avg_ordering_cost_gbp":
                    result["results"]
                    ["avg_ordering_cost_gbp"],

                "avg_shortage_cost_gbp":
                    result["results"]
                    ["avg_shortage_cost_gbp"],

                "avg_num_orders":
                    result["results"]
                    ["avg_num_orders"],

                "feasible":
                    service_level
                    >= target_service_level,

                "verdict":
                    result["results"]
                    ["verdict"]
            }

            all_results.append(flat)

    # -----------------------------------------------------------------
    # Save results.
    # -----------------------------------------------------------------

    if save_csv and all_results:

        with open(
            abs_csv_path,
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
            f"Results saved to {abs_csv_path}"
        )

    return all_results


def optimise_inventory_policy(
    demand_mean=200,
    demand_std=40,
    lead_time_days=7,
    reorder_points=None,
    order_quantities=None,
    target_service_level=DEFAULT_TARGET_SERVICE_LEVEL,
    seasonal_spike=True,
    trials=DEFAULT_TRIALS,
    seed=DEFAULT_SEED,
    save_csv=True,
    csv_path="results/inventory_optimisation_results.csv"
):
    """
    Find the lowest-cost feasible inventory policy.

    The optimiser performs an exhaustive grid search over the supplied
    ROP and OQ candidate values.

    The selected policy is therefore:

        lowest-cost feasible policy
        within the evaluated search space.

    It must not be interpreted as a global optimum outside that
    evaluated search space.
    """

    if reorder_points is None:

        reorder_points = list(
            range(500, 3501, 250)
        )

    if order_quantities is None:

        order_quantities = [
            500,
            750,
            1000,
            1250,
            1500,
            2000,
            2500
        ]

    results = run_experiment(
        reorder_points=reorder_points,
        order_quantities=order_quantities,
        demand_mean=demand_mean,
        demand_std=demand_std,
        lead_time_days=lead_time_days,
        seasonal_spike=seasonal_spike,
        trials=trials,
        save_csv=save_csv,
        csv_path=csv_path,
        target_service_level=target_service_level,
        seed=seed
    )

    # -----------------------------------------------------------------
    # Identify feasible policies.
    # -----------------------------------------------------------------

    feasible_policies = [
        r
        for r in results
        if r["avg_service_level_pct"]
        >= target_service_level
    ]

    # -----------------------------------------------------------------
    # Select lowest-cost feasible policy.
    # -----------------------------------------------------------------

    if feasible_policies:

        best_policy = min(
            feasible_policies,
            key=lambda r: (
                r["avg_total_cost_gbp"],
                -r["avg_service_level_pct"]
            )
        )

        status = "feasible_policy_found"

    else:

        # No infeasible policy is presented as the constrained optimum.
        # This fallback identifies the highest-service policy only.

        best_policy = min(
            results,
            key=lambda r: (
                -r["avg_service_level_pct"],
                r["avg_total_cost_gbp"]
            )
        )

        status = "no_feasible_policy_found"

    # -----------------------------------------------------------------
    # Rank feasible policies by total cost.
    # -----------------------------------------------------------------

    ranked_feasible_policies = sorted(
        feasible_policies,
        key=lambda r: (
            r["avg_total_cost_gbp"],
            -r["avg_service_level_pct"]
        )
    )

    # -----------------------------------------------------------------
    # Identify unconstrained cheapest policy.
    # -----------------------------------------------------------------

    cheapest_overall = min(
        results,
        key=lambda r: r["avg_total_cost_gbp"]
    )

    return {

        "optimisation":
            "inventory_replenishment_policy",

        "target_service_level_pct":
            target_service_level,

        "trials":
            trials,

        "seed":
            seed,

        "search_space": {

            "reorder_points":
                reorder_points,

            "order_quantities":
                order_quantities
        },

        "number_of_policies_evaluated":
            len(results),

        "number_of_feasible_policies":
            len(feasible_policies),

        "status":
            status,

        "selected_policy":
            best_policy,

        "cheapest_overall_policy":
            cheapest_overall,

        "top_feasible_policies":
            ranked_feasible_policies[:10],

        "all_results":
            results
    }


# ---------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 75)
    print(
        "Simulation 1: Inventory Replenishment"
    )
    print("=" * 75)

    print(
        f"\nCanonical Monte Carlo configuration:"
        f"\n  Trials: {DEFAULT_TRIALS}"
        f"\n  Seed:   {DEFAULT_SEED}"
    )

    # -----------------------------------------------------------------
    # Analytical benchmark
    # -----------------------------------------------------------------

    test = run_inventory_sim(
        reorder_point=800,
        order_quantity=1500,
        seed=DEFAULT_SEED
    )

    rop_theory = (
        test["results"]
        ["theoretical_optimal_rop"]
    )

    ss_theory = (
        test["results"]
        ["safety_stock_theoretical"]
    )

    print(
        f"\nTheoretical safety stock (95% benchmark): "
        f"{ss_theory} units"
    )

    print(
        f"Theoretical ROP benchmark: "
        f"{rop_theory} units"
    )

    # -----------------------------------------------------------------
    # Joint ROP x OQ experiment
    #
    # This standalone demonstration uses the smaller visualisation
    # search space. The dissertation optimiser above remains the
    # canonical quantitative search.
    # -----------------------------------------------------------------

    print(
        "\nRunning joint experiment "
        "(ROP x OQ combinations)..."
    )

    print("-" * 85)

    print(
        f"{'ROP':>6} | "
        f"{'OQ':>6} | "
        f"{'Service':>8} | "
        f"{'Fill Rate':>10} | "
        f"{'Feasible':>9} | "
        f"{'Total Cost':>12} | "
        f"{'Holding':>10} | "
        f"{'Shortage':>10}"
    )

    print("-" * 85)

    results = run_experiment(
        reorder_points=[
            200,
            400,
            600,
            800,
            1000,
            1200,
            1400,
            1600
        ],

        order_quantities=[
            1000,
            1500,
            2000
        ],

        demand_mean=200,
        demand_std=40,
        lead_time_days=7,

        seasonal_spike=True,

        # Use canonical dissertation trial count.
        trials=DEFAULT_TRIALS,

        save_csv=True,

        target_service_level=DEFAULT_TARGET_SERVICE_LEVEL,

        seed=DEFAULT_SEED
    )

    # -----------------------------------------------------------------
    # Select lowest-cost feasible policy.
    # -----------------------------------------------------------------

    feasible_results = [
        r
        for r in results
        if r["feasible"]
    ]

    if feasible_results:

        best = min(
            feasible_results,
            key=lambda r: (
                r["avg_total_cost_gbp"],
                -r["avg_service_level_pct"]
            )
        )

    else:

        best = min(
            results,
            key=lambda r: (
                -r["avg_service_level_pct"],
                r["avg_total_cost_gbp"]
            )
        )

    # -----------------------------------------------------------------
    # Print results.
    # -----------------------------------------------------------------

    for r in results:

        marker = (
            " <- selected"
            if r == best
            else ""
        )

        print(
            f"{r['reorder_point']:>6} | "
            f"{r['order_quantity']:>6} | "
            f"{r['avg_service_level_pct']:>7.1f}% | "
            f"{r['avg_fill_rate_pct']:>9.1f}% | "
            f"{str(r['feasible']):>9} | "
            f"£{r['avg_total_cost_gbp']:>11,.0f} | "
            f"£{r['avg_holding_cost_gbp']:>9,.0f} | "
            f"£{r['avg_shortage_cost_gbp']:>9,.0f}"
            f"{marker}"
        )

    print("-" * 85)

    # -----------------------------------------------------------------
    # Final recommendation
    # -----------------------------------------------------------------

    if feasible_results:

        print(
            "\nLowest-cost feasible policy:"
        )

    else:

        print(
            "\nNo policy met the 95% service-level target."
        )

        print(
            "Best available policy by service level:"
        )

    print(
        f"ROP = {best['reorder_point']}"
    )

    print(
        f"OQ  = {best['order_quantity']}"
    )

    print(
        f"Service Level = "
        f"{best['avg_service_level_pct']}%"
    )

    print(
        f"Fill Rate = "
        f"{best['avg_fill_rate_pct']}%"
    )

    print(
        f"Total Cost = "
        f"£{best['avg_total_cost_gbp']:,.2f}"
    )

    print(
        f"Holding Cost = "
        f"£{best['avg_holding_cost_gbp']:,.2f}"
    )

    print(
        f"Ordering Cost = "
        f"£{best['avg_ordering_cost_gbp']:,.2f}"
    )

    print(
        f"Shortage Cost = "
        f"£{best['avg_shortage_cost_gbp']:,.2f}"
    )

    print(
        "\nDone."
    )
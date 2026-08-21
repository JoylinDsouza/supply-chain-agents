# simulations/hub_location_sim.py
# Simulation 2: Distribution Hub Location — Financial Analysis
# Fully independent module. All parameters exposed as function arguments.
# Tracks: NPV, break-even year, probability of profitability, sensitivity analysis.

import numpy as np
import csv
import os


def run_hub_location_sim(
    candidate_location="Unknown",
    build_cost_millions=10.0,
    annual_ops_cost_millions=2.0,
    current_freight_cost_millions=8.0,
    freight_saving_pct=0.15,
    demand_growth_rate=0.10,
    demand_uncertainty=0.05,
    discount_rate=0.08,
    years=10,
    trials=100
):
    """
    Models the financial case for opening a distribution hub in a given location.

    Parameters
    ----------
    candidate_location            : name of location being evaluated (e.g. "Germany")
    build_cost_millions           : one-off capital cost to build the hub (GBP millions)
    annual_ops_cost_millions      : ongoing annual operating cost (GBP millions)
    current_freight_cost_millions : current annual freight spend for this region
    freight_saving_pct            : fraction of freight cost saved by having a local hub
    demand_growth_rate            : expected annual demand growth rate (e.g. 0.10 = 10%)
    demand_uncertainty            : std deviation of growth rate (e.g. 0.05 = 5%)
    discount_rate                 : cost of capital for NPV calculation (e.g. 0.08 = 8%)
    years                         : number of years to model
    trials                        : Monte Carlo trials for demand uncertainty

    Returns
    -------
    dict with NPV, break-even, recommendation, confidence metrics, annual cash flows
    """
    all_npvs = []
    all_breakeven_years = []
    profitable_count = 0
    all_annual_cashflows = [[] for _ in range(years)]

    for _ in range(trials):
        npv = -build_cost_millions
        cumulative = -build_cost_millions
        breakeven_year = None

        for year in range(1, years + 1):
            # Demand growth with Monte Carlo uncertainty
            actual_growth = np.random.normal(demand_growth_rate, demand_uncertainty)
            actual_growth = max(-0.20, actual_growth)  # floor at -20%
            demand_multiplier = (1 + actual_growth) ** year

            # Annual freight saving grows as demand grows
            freight_saving = (
                current_freight_cost_millions
                * freight_saving_pct
                * demand_multiplier
            )

            # Net annual benefit = saving minus running cost
            net_benefit = freight_saving - annual_ops_cost_millions

            # Discounted present value
            pv = net_benefit / ((1 + discount_rate) ** year)
            npv += pv
            cumulative += net_benefit
            all_annual_cashflows[year - 1].append(net_benefit)

            if cumulative > 0 and breakeven_year is None:
                breakeven_year = year

        all_npvs.append(npv)
        if npv > 0:
            profitable_count += 1
        all_breakeven_years.append(
            breakeven_year if breakeven_year is not None else years + 1
        )

    avg_npv = sum(all_npvs) / trials
    avg_breakeven = sum(all_breakeven_years) / trials
    prob_profitable = round((profitable_count / trials) * 100, 1)

    # Percentiles for confidence interval
    sorted_npvs = sorted(all_npvs)
    npv_p10 = sorted_npvs[int(0.10 * trials)]
    npv_p90 = sorted_npvs[int(0.90 * trials)]

    # Average annual cash flows across trials (for cash flow chart)
    avg_cashflows = [
        round(sum(year_cf) / len(year_cf), 3)
        for year_cf in all_annual_cashflows
    ]

    # Recommendation logic
    if avg_npv > 0 and avg_breakeven <= 7 and prob_profitable >= 70:
        recommendation = "INVEST — financially justified"
    elif avg_npv > 0 and prob_profitable >= 50:
        recommendation = "CONDITIONAL — invest only if demand growth confirmed"
    else:
        recommendation = "DO NOT INVEST — insufficient return"

    return {
        "simulation": "hub_location",
        "parameters": {
            "candidate_location": candidate_location,
            "build_cost_millions": build_cost_millions,
            "annual_ops_cost_millions": annual_ops_cost_millions,
            "current_freight_cost_millions": current_freight_cost_millions,
            "freight_saving_pct": freight_saving_pct,
            "demand_growth_rate": demand_growth_rate,
            "discount_rate": discount_rate,
            "years": years,
            "trials": trials
        },
        "results": {
            "avg_npv_millions": round(avg_npv, 3),
            "npv_p10_millions": round(npv_p10, 3),
            "npv_p90_millions": round(npv_p90, 3),
            "avg_breakeven_year": round(avg_breakeven, 1),
            "probability_profitable_pct": prob_profitable,
            "recommendation": recommendation,
            "avg_annual_cashflows": avg_cashflows
        }
    }


def run_sensitivity_analysis(
    base_params,
    variable="demand_growth_rate",
    values=None
):
    """
    Runs the hub simulation across a range of values for one parameter,
    holding all others constant. Returns a list of results.

    Parameters
    ----------
    base_params : dict of base parameter values
    variable    : the parameter name to vary
    values      : list of values to test

    Returns
    -------
    list of (value, avg_npv, recommendation) tuples
    """
    if values is None:
        values = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

    results = []
    for v in values:
        params = base_params.copy()
        params[variable] = v
        result = run_hub_location_sim(**params)
        results.append({
            "variable": variable,
            "value": v,
            "avg_npv_millions": result["results"]["avg_npv_millions"],
            "probability_profitable_pct": result["results"]["probability_profitable_pct"],
            "avg_breakeven_year": result["results"]["avg_breakeven_year"],
            "recommendation": result["results"]["recommendation"]
        })
    return results


def run_location_comparison(
    locations,
    demand_growth_rates=None,
    save_csv=True,
    csv_path="results/simulation2_results.csv"
):
    """
    Compares multiple candidate hub locations across a range of demand growth rates.

    Parameters
    ----------
    locations          : list of dicts, each with location-specific parameters
    demand_growth_rates: list of growth rates to test for each location
    save_csv           : whether to save results to CSV
    csv_path           : where to save the CSV

    Returns
    -------
    list of result dicts (one per location × growth rate combination)
    """
    if demand_growth_rates is None:
        demand_growth_rates = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    all_results = []

    for loc in locations:
        for growth in demand_growth_rates:
            result = run_hub_location_sim(
                candidate_location=loc["name"],
                build_cost_millions=loc["build_cost"],
                annual_ops_cost_millions=loc["ops_cost"],
                current_freight_cost_millions=loc.get("freight_cost", 8.0),
                freight_saving_pct=loc["freight_saving"],
                demand_growth_rate=growth,
                trials=100
            )
            flat = {
                "location": loc["name"],
                "demand_growth_rate_pct": round(growth * 100, 0),
                "build_cost_millions": loc["build_cost"],
                "avg_npv_millions": result["results"]["avg_npv_millions"],
                "npv_p10_millions": result["results"]["npv_p10_millions"],
                "npv_p90_millions": result["results"]["npv_p90_millions"],
                "avg_breakeven_year": result["results"]["avg_breakeven_year"],
                "probability_profitable_pct": result["results"]["probability_profitable_pct"],
                "recommendation": result["results"]["recommendation"]
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
    print("=" * 70)
    print("Simulation 2: Hub Location Financial Analysis")
    print("=" * 70)

    # Three candidate locations with realistic parameters
    locations = [
        {
            "name": "Germany",
            "build_cost": 12.0,
            "ops_cost": 2.5,
            "freight_cost": 12.0,
            "freight_saving": 0.18
        },
        {
            "name": "Poland",
            "build_cost": 7.0,
            "ops_cost": 1.6,
            "freight_cost": 12.0,
            "freight_saving": 0.13
        },
        {
            "name": "Netherlands",
            "build_cost": 15.0,
            "ops_cost": 3.0,
            "freight_cost": 12.0,
            "freight_saving": 0.23
        }
    ]

    growth_rates = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    
    print("\nRunning location comparison across demand growth rates...")
    print("-" * 70)
    print(f"{'Location':<14} {'Growth':>8} {'NPV (£M)':>10} "
          f"{'Break-even':>11} {'Prob %':>8}  Recommendation")
    print("-" * 70)

    results = run_location_comparison(
        locations=locations,
        demand_growth_rates=growth_rates,
        save_csv=True
    )

    current_loc = ""
    for r in results:
        if r["location"] != current_loc:
            if current_loc:
                print()
            current_loc = r["location"]
        print(
            f"{r['location']:<14} "
            f"{r['demand_growth_rate_pct']:>7.0f}% "
            f"{r['avg_npv_millions']:>+10.2f}M "
            f"{r['avg_breakeven_year']:>10.1f}yr "
            f"{r['probability_profitable_pct']:>7.0f}%  "
            f"{r['recommendation']}"
        )

    print("\nDone. Results saved to results/simulation2_results.csv")
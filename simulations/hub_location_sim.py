# simulations/hub_location_sim.py
# Simulation 2: Distribution Hub Location — Financial Analysis
# Fully independent module — all parameters exposed as function arguments.

import numpy as np


def run_hub_location_sim(
    candidate_location="Unknown",
    annual_demand_units=500000,
    demand_growth_rate=0.10,
    build_cost_millions=10.0,
    annual_ops_cost_millions=2.0,
    current_freight_cost_millions=8.0,
    freight_saving_pct=0.15,
    discount_rate=0.08,
    years=10,
    demand_uncertainty=0.05,
    trials=50
):
    """
    Models the financial case for opening a distribution hub in a given location.

    Parameters
    ----------
    candidate_location          : name of the location being evaluated (e.g. "Germany")
    annual_demand_units         : current annual demand served through this region
    demand_growth_rate          : expected annual demand growth rate (e.g. 0.10 = 10%)
    build_cost_millions         : one-off capital cost to build the hub (GBP millions)
    annual_ops_cost_millions    : ongoing annual operating cost (GBP millions)
    current_freight_cost_millions: current annual freight spend for this region (GBP millions)
    freight_saving_pct          : fraction of freight cost saved by having a local hub
                                  (e.g. 0.15 = 15% saving)
    discount_rate               : cost of capital / discount rate for NPV (e.g. 0.08 = 8%)
    years                       : number of years to model
    demand_uncertainty          : standard deviation of annual demand growth as a fraction
                                  (e.g. 0.05 = 5% uncertainty around the growth rate)
    trials                      : Monte Carlo trials to account for demand uncertainty

    Returns
    -------
    dict with NPV, break-even year, recommendation, and confidence metrics
    """
    all_npvs = []
    all_breakeven_years = []
    profitable_trials = 0

    for _ in range(trials):
        npv = -build_cost_millions  # Start with the upfront cost as a negative
        cumulative_cash_flow = -build_cost_millions
        breakeven_year = None

        for year in range(1, years + 1):
            # Demand grows each year — with uncertainty added via Monte Carlo
            actual_growth = np.random.normal(demand_growth_rate, demand_uncertainty)
            actual_growth = max(-0.20, actual_growth)  # cap downside at -20%
            demand_multiplier = (1 + actual_growth) ** year

            # Freight saving grows as demand grows
            annual_freight_saving = (
                current_freight_cost_millions
                * freight_saving_pct
                * demand_multiplier
            )

            # Net annual benefit = freight saving minus running cost
            net_annual_benefit = annual_freight_saving - annual_ops_cost_millions

            # Discount to present value (money today is worth more than money later)
            present_value = net_annual_benefit / ((1 + discount_rate) ** year)
            npv += present_value
            cumulative_cash_flow += net_annual_benefit

            # Record the first year the cumulative cash flow turns positive
            if cumulative_cash_flow > 0 and breakeven_year is None:
                breakeven_year = year

        all_npvs.append(npv)
        if npv > 0:
            profitable_trials += 1
        all_breakeven_years.append(breakeven_year if breakeven_year else years + 1)

    avg_npv = sum(all_npvs) / trials
    avg_breakeven = sum(all_breakeven_years) / trials
    probability_profitable = round((profitable_trials / trials) * 100, 1)

    # Decide recommendation
    if avg_npv > 0 and avg_breakeven <= 7 and probability_profitable >= 70:
        recommendation = "INVEST — financially justified"
    elif avg_npv > 0 and probability_profitable >= 50:
        recommendation = "CONDITIONAL — invest only if demand growth confirmed"
    else:
        recommendation = "DO NOT INVEST — insufficient financial return"

    return {
        "simulation": "hub_location",
        "parameters": {
            "candidate_location": candidate_location,
            "demand_growth_rate": demand_growth_rate,
            "build_cost_millions": build_cost_millions,
            "annual_ops_cost_millions": annual_ops_cost_millions,
            "freight_saving_pct": freight_saving_pct,
            "years": years,
            "trials": trials
        },
        "results": {
            "avg_10yr_npv_millions": round(avg_npv, 2),
            "avg_breakeven_year": round(avg_breakeven, 1),
            "probability_profitable_pct": probability_profitable,
            "recommendation": recommendation
        }
    }


# ── Quick test when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Simulation 2: Hub Location Financial Analysis")
    print("=" * 60)

    locations = [
        {
            "name": "Germany",
            "build_cost": 12.0,
            "ops_cost": 2.5,
            "freight_saving": 0.18
        },
        {
            "name": "Poland",
            "build_cost": 7.0,
            "ops_cost": 1.6,
            "freight_saving": 0.12
        },
        {
            "name": "Netherlands",
            "build_cost": 14.0,
            "ops_cost": 3.0,
            "freight_saving": 0.20
        }
    ]

    growth_rates = [0.05, 0.10, 0.15, 0.20]

    for loc in locations:
        print(f"\n--- {loc['name']} ---")
        for growth in growth_rates:
            result = run_hub_location_sim(
                candidate_location=loc["name"],
                demand_growth_rate=growth,
                build_cost_millions=loc["build_cost"],
                annual_ops_cost_millions=loc["ops_cost"],
                freight_saving_pct=loc["freight_saving"],
                trials=50
            )
            r = result["results"]
            print(
                f"  Growth {growth*100:.0f}% | "
                f"NPV: £{r['avg_10yr_npv_millions']:>6.1f}M | "
                f"Break-even: yr {r['avg_breakeven_year']:>4.1f} | "
                f"Profitable: {r['probability_profitable_pct']:>5}% | "
                f"{r['recommendation']}"
            )
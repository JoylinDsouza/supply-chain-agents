# simulations/hub_location_sim.py
# Simulation 2: Distribution Hub Location — Financial Analysis
# Fully independent module.
# Canonical Monte Carlo configuration: 1000 trials, seed 42.
# Tracks: NPV, break-even year, probability of profitability,
# sensitivity analysis, and outcome percentiles.

import numpy as np
import csv
import os


CANONICAL_TRIALS = 1000
CANONICAL_SEED = 42


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
    trials=CANONICAL_TRIALS,
    seed=CANONICAL_SEED
):
    """
    Models the financial case for opening a distribution hub.

    Monte Carlo uncertainty is introduced through annual demand growth.
    Each trial receives an independent but reproducible random sequence.

    Parameters
    ----------
    candidate_location            : name of location being evaluated
    build_cost_millions            : one-off capital cost (£ millions)
    annual_ops_cost_millions       : annual operating cost (£ millions)
    current_freight_cost_millions  : current annual freight spend (£ millions)
    freight_saving_pct              : fraction of freight cost saved
    demand_growth_rate              : expected annual demand growth
    demand_uncertainty              : standard deviation of annual growth
    discount_rate                   : discount rate used for NPV
    years                           : number of years to model
    trials                          : number of Monte Carlo trials
    seed                            : random seed for reproducibility

    Returns
    -------
    dict
        Simulation parameters and financial results.
    """

    if trials < 1:
        raise ValueError("trials must be at least 1.")

    if years < 1:
        raise ValueError("years must be at least 1.")

    if discount_rate <= -1:
        raise ValueError("discount_rate must be greater than -1.")

    rng = np.random.default_rng(seed)

    all_npvs = []
    all_breakeven_years = []
    profitable_count = 0
    all_annual_cashflows = [[] for _ in range(years)]

    for _ in range(trials):
        npv = -build_cost_millions
        cumulative = -build_cost_millions
        breakeven_year = None

        # Current year demand is the baseline.
        demand_index = 1.0

        for year in range(1, years + 1):

            # Draw the realised annual demand growth.
            actual_growth = rng.normal(
                demand_growth_rate,
                demand_uncertainty
            )

            # Prevent extreme downside scenarios.
            actual_growth = max(-0.20, actual_growth)

            # Compound demand growth across years.
            demand_index *= (1 + actual_growth)

            # Freight saving increases with cumulative demand.
            freight_saving = (
                current_freight_cost_millions
                * freight_saving_pct
                * demand_index
            )

            # Net annual financial benefit.
            net_benefit = (
                freight_saving
                - annual_ops_cost_millions
            )

            # Discount annual benefit to present value.
            pv = net_benefit / (
                (1 + discount_rate) ** year
            )

            npv += pv
            cumulative += net_benefit

            all_annual_cashflows[year - 1].append(
                net_benefit
            )

            if cumulative > 0 and breakeven_year is None:
                breakeven_year = year

        all_npvs.append(npv)

        if npv > 0:
            profitable_count += 1

        # years + 1 represents no break-even within
        # the simulated horizon.
        all_breakeven_years.append(
            breakeven_year
            if breakeven_year is not None
            else years + 1
        )

    # Aggregate Monte Carlo results.
    avg_npv = float(np.mean(all_npvs))
    avg_breakeven = float(np.mean(all_breakeven_years))

    prob_profitable = round(
        (profitable_count / trials) * 100,
        1
    )

    # These are outcome percentiles, not confidence intervals.
    npv_p10 = float(np.percentile(all_npvs, 10))
    npv_p90 = float(np.percentile(all_npvs, 90))

    # Average annual cash flow across Monte Carlo trials.
    avg_cashflows = [
        round(float(np.mean(year_cf)), 3)
        for year_cf in all_annual_cashflows
    ]

    # Recommendation rule based on the model's financial criteria.
    if (
        avg_npv > 0
        and avg_breakeven <= 7
        and prob_profitable >= 70
    ):
        recommendation = "INVEST — financially justified"

    elif (
        avg_npv > 0
        and prob_profitable >= 50
    ):
        recommendation = (
            "CONDITIONAL — invest only if demand growth confirmed"
        )

    else:
        recommendation = (
            "DO NOT INVEST — insufficient return"
        )

    return {
        "simulation": "hub_location",

        "parameters": {
            "candidate_location": candidate_location,
            "build_cost_millions": build_cost_millions,
            "annual_ops_cost_millions": annual_ops_cost_millions,
            "current_freight_cost_millions": (
                current_freight_cost_millions
            ),
            "freight_saving_pct": freight_saving_pct,
            "demand_growth_rate": demand_growth_rate,
            "demand_uncertainty": demand_uncertainty,
            "discount_rate": discount_rate,
            "years": years,
            "trials": trials,
            "seed": seed
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
    holding all other parameters constant.

    Returns a list of result dictionaries.
    """

    if values is None:
        values = [
            0.03,
            0.05,
            0.08,
            0.10,
            0.12,
            0.15,
            0.18,
            0.20
        ]

    results = []

    for v in values:
        params = base_params.copy()
        params[variable] = v

        result = run_hub_location_sim(**params)

        results.append({
            "variable": variable,
            "value": v,
            "avg_npv_millions": (
                result["results"]["avg_npv_millions"]
            ),
            "probability_profitable_pct": (
                result["results"]["probability_profitable_pct"]
            ),
            "avg_breakeven_year": (
                result["results"]["avg_breakeven_year"]
            ),
            "recommendation": (
                result["results"]["recommendation"]
            )
        })

    return results


def run_location_comparison(
    locations,
    demand_growth_rates=None,
    save_csv=True,
    csv_path="results/simulation2_results.csv",
    trials=CANONICAL_TRIALS,
    seed=CANONICAL_SEED
):
    """
    Compares multiple candidate hub locations across a range
    of demand growth rates.

    Parameters
    ----------
    locations           : list of location-specific dictionaries
    demand_growth_rates : growth rates to evaluate
    save_csv            : whether to save results to CSV
    csv_path            : output CSV path
    trials              : Monte Carlo trials per scenario
    seed                : base random seed

    Returns
    -------
    list of result dictionaries.
    """

    if demand_growth_rates is None:
        demand_growth_rates = [
            0.05,
            0.08,
            0.10,
            0.12,
            0.15,
            0.18,
            0.20
        ]

    output_dir = os.path.dirname(csv_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_results = []

    scenario_index = 0

    for loc in locations:
        for growth in demand_growth_rates:

            # Give each scenario a reproducible but distinct seed.
            scenario_seed = seed + scenario_index
            scenario_index += 1

            result = run_hub_location_sim(
                candidate_location=loc["name"],
                build_cost_millions=loc["build_cost"],
                annual_ops_cost_millions=loc["ops_cost"],
                current_freight_cost_millions=loc.get(
                    "freight_cost",
                    8.0
                ),
                freight_saving_pct=loc["freight_saving"],
                demand_growth_rate=growth,
                trials=trials,
                seed=scenario_seed
            )

            flat = {
                "location": loc["name"],
                "demand_growth_rate_pct": round(
                    growth * 100,
                    0
                ),
                "build_cost_millions": loc["build_cost"],
                "avg_npv_millions": (
                    result["results"]["avg_npv_millions"]
                ),
                "npv_p10_millions": (
                    result["results"]["npv_p10_millions"]
                ),
                "npv_p90_millions": (
                    result["results"]["npv_p90_millions"]
                ),
                "avg_breakeven_year": (
                    result["results"]["avg_breakeven_year"]
                ),
                "probability_profitable_pct": (
                    result["results"][
                        "probability_profitable_pct"
                    ]
                ),
                "recommendation": (
                    result["results"]["recommendation"]
                )
            }

            all_results.append(flat)

    if save_csv:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=all_results[0].keys()
            )
            writer.writeheader()
            writer.writerows(all_results)

        print(f"Results saved to {csv_path}")

    return all_results


# ---------------------------------------------------------------------------
# Quick test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("Simulation 2: Hub Location Financial Analysis")
    print("=" * 70)

    print("\nCanonical Monte Carlo configuration:")
    print(f"  Trials: {CANONICAL_TRIALS}")
    print(f"  Seed:   {CANONICAL_SEED}")

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

    growth_rates = [
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
        0.15,
        0.18,
        0.20
    ]

    print("\nRunning location comparison across demand growth rates...")
    print("-" * 70)

    print(
        f"{'Location':<14} "
        f"{'Growth':>8} "
        f"{'NPV (£M)':>10} "
        f"{'Break-even':>11} "
        f"{'Prob %':>8}  "
        f"Recommendation"
    )

    print("-" * 70)

    results = run_location_comparison(
        locations=locations,
        demand_growth_rates=growth_rates,
        save_csv=True,
        trials=CANONICAL_TRIALS,
        seed=CANONICAL_SEED
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

    print("\nDone.")
    print(
        "Results saved to results/simulation2_results.csv"
    )
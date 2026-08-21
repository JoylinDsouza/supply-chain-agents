# simulations/hub_visualiser.py
# Visualisation for Simulation 2: Hub Location Financial Analysis
# Produces 4 publication-ready graphs including sensitivity analysis.

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulations.hub_location_sim import (
    run_hub_location_sim,
    run_location_comparison,
    run_sensitivity_analysis
)

os.makedirs("results", exist_ok=True)

# ── Colours and location definitions ─────────────────────────────────────────
C_GERMANY     = "#2E6FD6"
C_POLAND      = "#2E9E5A"
C_NETHERLANDS = "#E07B2A"
C_ZERO        = "#D63A2E"
C_GREY        = "#AAAAAA"

LOCATIONS = [
    {"name": "Germany",     "build_cost": 12.0, "ops_cost": 2.5,
     "freight_cost": 12.0, "freight_saving": 0.18},
    {"name": "Poland",      "build_cost": 7.0,  "ops_cost": 1.6,
     "freight_cost": 12.0, "freight_saving": 0.13},
    {"name": "Netherlands", "build_cost": 15.0, "ops_cost": 3.0,
     "freight_cost": 12.0, "freight_saving": 0.23},
]
LOC_COLOURS = {
    "Germany": C_GERMANY,
    "Poland": C_POLAND,
    "Netherlands": C_NETHERLANDS
}
GROWTH_RATES = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
GROWTH_LABELS = [f"{int(g*100)}%" for g in GROWTH_RATES]


def plot_all():
    print("Running location comparison...")
    results = run_location_comparison(
        locations=LOCATIONS,
        demand_growth_rates=GROWTH_RATES,
        save_csv=True
    )

    # Organise results by location
    by_loc = {}
    for r in results:
        loc = r["location"]
        if loc not in by_loc:
            by_loc[loc] = {"growth": [], "npv": [], "breakeven": [], "prob": []}
        by_loc[loc]["growth"].append(r["demand_growth_rate_pct"])
        by_loc[loc]["npv"].append(r["avg_npv_millions"])
        by_loc[loc]["breakeven"].append(r["avg_breakeven_year"])
        by_loc[loc]["prob"].append(r["probability_profitable_pct"])

    # ── FIGURE 1: Four-panel comparison ──────────────────────────────────────
    fig1, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig1.suptitle(
        "Simulation 2: Distribution Hub Location — Financial Analysis\n"
        "(Monte Carlo, 100 trials per scenario, 10-year horizon)",
        fontsize=13, fontweight="bold", y=1.01
    )

    growth_x = [g * 100 for g in GROWTH_RATES]

    # Panel 1: NPV vs demand growth for all three locations
    ax = axes[0, 0]
    for loc_name, data in by_loc.items():
        ax.plot(data["growth"], data["npv"],
                marker="o", linewidth=2.5, markersize=7,
                color=LOC_COLOURS[loc_name], label=loc_name)
    ax.axhline(y=0, color=C_ZERO, linestyle="--", linewidth=1.5,
               alpha=0.8, label="Break-even (NPV = 0)")
    ax.fill_between(growth_x,
                    [min(d["npv"][i] for d in by_loc.values())
                     for i in range(len(GROWTH_RATES))],
                    0,
                    where=[all(by_loc[l]["npv"][i] < 0
                               for l in by_loc) for i in range(len(GROWTH_RATES))],
                    alpha=0.08, color=C_ZERO)
    ax.set_xlabel("Annual Demand Growth Rate (%)", fontsize=11)
    ax.set_ylabel("Average 10-Year NPV (£M)", fontsize=11)
    ax.set_title("NPV vs Demand Growth Rate by Location", fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:+.1f}M"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(growth_x)
    ax.set_xticklabels(GROWTH_LABELS)

    # Panel 2: Break-even year vs growth rate
    ax = axes[0, 1]
    for loc_name, data in by_loc.items():
        be = [min(b, 11) for b in data["breakeven"]]
        ax.plot(data["growth"], be,
                marker="s", linewidth=2.5, markersize=7,
                color=LOC_COLOURS[loc_name], label=loc_name)
    ax.axhline(y=7, color=C_ZERO, linestyle="--", linewidth=1.5,
               alpha=0.8, label="7-year payback target")
    ax.set_xlabel("Annual Demand Growth Rate (%)", fontsize=11)
    ax.set_ylabel("Average Break-even Year", fontsize=11)
    ax.set_title("Break-even Year vs Demand Growth Rate", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(growth_x)
    ax.set_xticklabels(GROWTH_LABELS)
    ax.set_ylim(0, 12)
    ax.invert_yaxis()   # lower break-even = better, so put at top

    # Panel 3: Probability of profitability
    ax = axes[1, 0]
    width = 0.8
    x = np.arange(len(GROWTH_RATES))
    n = len(LOCATIONS)
    bar_width = width / n
    for i, (loc_name, data) in enumerate(by_loc.items()):
        offset = (i - n / 2 + 0.5) * bar_width
        ax.bar(x + offset, data["prob"], bar_width,
               label=loc_name, color=LOC_COLOURS[loc_name], alpha=0.85)
    ax.axhline(y=70, color=C_ZERO, linestyle="--", linewidth=1.5,
               alpha=0.8, label="70% confidence threshold")
    ax.set_xlabel("Annual Demand Growth Rate (%)", fontsize=11)
    ax.set_ylabel("Probability of Profitable (%)", fontsize=11)
    ax.set_title("Probability of Profitability by Location", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(GROWTH_LABELS)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 110)

    # Panel 4: Head-to-head NPV comparison at 10% growth
    ax = axes[1, 1]
    target_growth = 10
    npv_at_10 = {}
    p10_at_10 = {}
    p90_at_10 = {}
    for loc in LOCATIONS:
        r = run_hub_location_sim(
            candidate_location=loc["name"],
            build_cost_millions=loc["build_cost"],
            annual_ops_cost_millions=loc["ops_cost"],
            current_freight_cost_millions=loc["freight_cost"],
            freight_saving_pct=loc["freight_saving"],
            demand_growth_rate=target_growth / 100,
            trials=200
        )
        npv_at_10[loc["name"]] = r["results"]["avg_npv_millions"]
        p10_at_10[loc["name"]] = r["results"]["npv_p10_millions"]
        p90_at_10[loc["name"]] = r["results"]["npv_p90_millions"]

    loc_names = list(npv_at_10.keys())
    npv_vals  = [npv_at_10[l] for l in loc_names]
    colours   = [LOC_COLOURS[l] for l in loc_names]
    err_low   = [npv_at_10[l] - p10_at_10[l] for l in loc_names]
    err_high  = [p90_at_10[l] - npv_at_10[l] for l in loc_names]

    bars = ax.bar(loc_names, npv_vals, color=colours, alpha=0.85, width=0.5)
    ax.errorbar(loc_names, npv_vals,
                yerr=[err_low, err_high],
                fmt="none", color="black", capsize=8, linewidth=2)
    ax.axhline(y=0, color=C_ZERO, linestyle="--", linewidth=1.5, alpha=0.8)
    ax.set_ylabel("Average 10-Year NPV (£M)", fontsize=11)
    ax.set_title(f"NPV Comparison at {target_growth}% Demand Growth\n"
                 f"(bars = mean, error bars = P10–P90 range)",
                 fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:+.1f}M"))
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, npv_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + (0.3 if val >= 0 else -0.8),
                f"£{val:+.1f}M",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path1 = "results/sim2_location_comparison.png"
    fig1.savefig(path1, dpi=150, bbox_inches="tight")
    print(f"Saved: {path1}")

    # ── FIGURE 2: Sensitivity analysis for Poland ─────────────────────────────
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle(
        "Simulation 2: Sensitivity Analysis — Poland Hub\n"
        "(How each assumption affects the 10-Year NPV at 10% baseline growth)",
        fontsize=12, fontweight="bold"
    )

    poland_base = {
        "candidate_location": "Poland",
        "build_cost_millions": 7.0,
        "annual_ops_cost_millions": 1.6,
        "current_freight_cost_millions": 12.0,
        "freight_saving_pct": 0.13,
        "demand_growth_rate": 0.10,
        "trials": 100
    }

    # Sensitivity: vary each key parameter ±30%
    base_result = run_hub_location_sim(**poland_base)
    base_npv = base_result["results"]["avg_npv_millions"]

    sensitivity_vars = [
        ("demand_growth_rate",          0.07,  0.13,  "Demand growth rate"),
        ("freight_saving_pct",          0.091, 0.169, "Freight saving %"),
        ("build_cost_millions",         4.9,   9.1,   "Build cost"),
        ("annual_ops_cost_millions",    1.12,  2.08,  "Annual ops cost"),
        ("current_freight_cost_millions", 8.4, 15.6,  "Freight spend baseline"),
    ]

    tornado_labels = []
    tornado_low    = []
    tornado_high   = []

    for var, low_val, high_val, label in sensitivity_vars:
        params_low  = poland_base.copy(); params_low[var]  = low_val
        params_high = poland_base.copy(); params_high[var] = high_val
        npv_low  = run_hub_location_sim(**params_low)["results"]["avg_npv_millions"]
        npv_high = run_hub_location_sim(**params_high)["results"]["avg_npv_millions"]
        tornado_labels.append(label)
        tornado_low.append(min(npv_low, npv_high))
        tornado_high.append(max(npv_low, npv_high))

    # Sort by range width (widest at top = tornado shape)
    ranges = [(h - l) for l, h in zip(tornado_low, tornado_high)]
    order  = sorted(range(len(ranges)), key=lambda i: ranges[i], reverse=True)
    tornado_labels = [tornado_labels[i] for i in order]
    tornado_low    = [tornado_low[i]    for i in order]
    tornado_high   = [tornado_high[i]   for i in order]

    y_pos = range(len(tornado_labels))
    ax1.barh(list(y_pos), [h - l for l, h in zip(tornado_low, tornado_high)],
             left=tornado_low, color=C_POLAND, alpha=0.8, height=0.5)
    ax1.axvline(x=base_npv, color=C_ZERO, linestyle="--",
                linewidth=2, label=f"Base NPV £{base_npv:+.1f}M")
    ax1.set_yticks(list(y_pos))
    ax1.set_yticklabels(tornado_labels, fontsize=10)
    ax1.set_xlabel("10-Year NPV (£M)", fontsize=11)
    ax1.set_title("Tornado Chart — Sensitivity to Each Assumption\n"
                  "(±30% change in each parameter)", fontsize=11, fontweight="bold")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:+.1f}M"))
    ax1.axvline(x=0, color="black", linewidth=0.8, alpha=0.5)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="x")

    # Annual cash flow for Poland at 10% growth
    cashflows = base_result["results"]["avg_annual_cashflows"]
    years = list(range(1, len(cashflows) + 1))
    colours_cf = [C_POLAND if cf >= 0 else C_ZERO for cf in cashflows]
    ax2.bar(years, cashflows, color=colours_cf, alpha=0.85)
    cumulative = []
    running = -poland_base["build_cost_millions"]
    for cf in cashflows:
        running += cf
        cumulative.append(running)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(years, cumulative, color="black", linewidth=2.5,
                  marker="o", markersize=6, label="Cumulative cash flow")
    ax2_twin.axhline(y=0, color=C_ZERO, linestyle="--",
                     linewidth=1.5, alpha=0.7)
    ax2_twin.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"£{x:+.1f}M"))
    ax2_twin.set_ylabel("Cumulative Cash Flow (£M)", fontsize=11)
    ax2_twin.legend(fontsize=9, loc="lower right")
    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_ylabel("Annual Net Benefit (£M)", fontsize=11)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:+.1f}M"))
    ax2.set_title("Poland: Annual Cash Flow Profile\n"
                  "(10% demand growth, 10-year horizon)",
                  fontsize=11, fontweight="bold")
    ax2.set_xticks(years)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path2 = "results/sim2_sensitivity_analysis.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    print(f"Saved: {path2}")

    plt.show()
    return path1, path2


if __name__ == "__main__":
    print("=" * 65)
    print("Simulation 2 Visualiser — generating all graphs")
    print("=" * 65)
    plot_all()
    print("\nAll done. Check your results/ folder for:")
    print("  sim2_location_comparison.png  — four-panel location analysis")
    print("  sim2_sensitivity_analysis.png — tornado chart and cash flows")
    print("  simulation2_results.csv       — full results table")
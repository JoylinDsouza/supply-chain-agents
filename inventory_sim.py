import numpy as np
import json
import os
import csv
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── THE SIMULATION ───────────────────────────────────────────────────────────
def run_inventory_sim(
    reorder_point,
    order_quantity,
    lead_time_days,
    demand_mean,
    demand_std,
    simulation_days=90,
    seasonal_spike_week=6,
    seasonal_spike_multiplier=1.4
):
    """
    Simulates a retail store's inventory over N days.
    Includes seasonal demand spike (e.g. a sale period).
    Returns key performance indicators.
    """
    np.random.seed(None)  # Different random seed each run (for Monte Carlo)
    inventory = reorder_point * 2
    holding_cost = 0
    stockout_days = 0
    orders_pending = []

    for day in range(simulation_days):
        # Check if any pending orders arrive today
        arrived = [(d, q) for (d, q) in orders_pending if d <= day]
        for (_, qty) in arrived:
            inventory += qty
        orders_pending = [(d, q) for (d, q) in orders_pending if d > day]

        # Seasonal spike: weeks 6 and 7 (days 35-49) have higher demand
        week = day // 7
        if week == seasonal_spike_week or week == seasonal_spike_week + 1:
            today_mean = demand_mean * seasonal_spike_multiplier
        else:
            today_mean = demand_mean

        # Random daily demand
        demand = max(0, int(np.random.normal(today_mean, demand_std)))

        # Fill demand from inventory
        if inventory >= demand:
            inventory -= demand
        else:
            stockout_days += 1
            inventory = 0

        # Holding cost
        holding_cost += inventory * 0.5

        # Reorder if low
        if inventory <= reorder_point and not orders_pending:
            orders_pending.append((day + lead_time_days, order_quantity))

    service_level = round((1 - stockout_days / simulation_days) * 100, 1)
    return {
        "service_level_pct": service_level,
        "stockout_days": stockout_days,
        "total_holding_cost_gbp": round(holding_cost, 2),
        "avg_daily_holding_cost_gbp": round(holding_cost / simulation_days, 2),
        "verdict": "good" if service_level >= 95 else "needs improvement"
    }


# ── MONTE CARLO: RUN MULTIPLE TRIALS AND AVERAGE ────────────────────────────
def run_monte_carlo(reorder_point, order_quantity, lead_time_days,
                    demand_mean, demand_std, trials=20):
    """
    Runs the simulation multiple times and returns averaged results.
    This smooths out lucky or unlucky random runs.
    """
    all_service_levels = []
    all_stockout_days = []
    all_holding_costs = []

    for _ in range(trials):
        result = run_inventory_sim(
            reorder_point, order_quantity, lead_time_days,
            demand_mean, demand_std
        )
        all_service_levels.append(result["service_level_pct"])
        all_stockout_days.append(result["stockout_days"])
        all_holding_costs.append(result["total_holding_cost_gbp"])

    return {
        "reorder_point": reorder_point,
        "order_quantity": order_quantity,
        "avg_service_level_pct": round(sum(all_service_levels) / trials, 1),
        "avg_stockout_days": round(sum(all_stockout_days) / trials, 1),
        "avg_holding_cost_gbp": round(sum(all_holding_costs) / trials, 2),
        "trials_run": trials
    }


# ── EXPERIMENT: TEST A RANGE OF POLICIES ────────────────────────────────────
print("=" * 60)
print("EXPERIMENT: Testing 6 reorder point policies")
print("Each policy is run 20 times and averaged (Monte Carlo)")
print("=" * 60)

# Parameters that stay the same across all policies
LEAD_TIME = 7
DEMAND_MEAN = 200
DEMAND_STD = 40
ORDER_QTY = 1500

# The range of reorder points we are testing
reorder_points = [200, 400, 600, 800, 1000, 1200]

experiment_results = []

for rp in reorder_points:
    result = run_monte_carlo(
        reorder_point=rp,
        order_quantity=ORDER_QTY,
        lead_time_days=LEAD_TIME,
        demand_mean=DEMAND_MEAN,
        demand_std=DEMAND_STD,
        trials=20
    )
    experiment_results.append(result)
    print(f"Reorder point {rp:>5} | "
          f"Service level: {result['avg_service_level_pct']:>5}% | "
          f"Stockout days: {result['avg_stockout_days']:>4} | "
          f"Holding cost: £{result['avg_holding_cost_gbp']:>8,.0f}")

# ── SAVE RESULTS TO CSV ──────────────────────────────────────────────────────
csv_filename = "simulation1_results.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=experiment_results[0].keys())
    writer.writeheader()
    writer.writerows(experiment_results)

print(f"\nResults saved to {csv_filename}")

# ── PLOT THE GRAPH ───────────────────────────────────────────────────────────
rp_values = [r["reorder_point"] for r in experiment_results]
service_levels = [r["avg_service_level_pct"] for r in experiment_results]
holding_costs = [r["avg_holding_cost_gbp"] for r in experiment_results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Simulation 1: Inventory Replenishment Policy Comparison\n(Monte Carlo, 20 trials each, with seasonal demand spike)",
             fontsize=13, fontweight="bold")

# Graph 1: Service Level
ax1.plot(rp_values, service_levels, marker="o", color="steelblue",
         linewidth=2, markersize=8)
ax1.axhline(y=95, color="red", linestyle="--", alpha=0.7, label="95% target")
ax1.set_xlabel("Reorder Point (units)", fontsize=11)
ax1.set_ylabel("Average Service Level (%)", fontsize=11)
ax1.set_title("Service Level vs Reorder Point", fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 105)

# Graph 2: Holding Cost
ax2.plot(rp_values, holding_costs, marker="s", color="darkorange",
         linewidth=2, markersize=8)
ax2.set_xlabel("Reorder Point (units)", fontsize=11)
ax2.set_ylabel("Average Holding Cost (£)", fontsize=11)
ax2.set_title("Holding Cost vs Reorder Point", fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"£{x:,.0f}"))

plt.tight_layout()
graph_filename = "simulation1_graph.png"
plt.savefig(graph_filename, dpi=150, bbox_inches="tight")
plt.show()
print(f"Graph saved to {graph_filename}")

# ── ASK GPT TO ANALYSE THE RESULTS ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Asking GPT to analyse the full experiment results...")
print("=" * 60)

# Format results as a readable table for GPT
results_text = "Reorder Point | Service Level | Stockout Days | Holding Cost\n"
results_text += "-" * 60 + "\n"
for r in experiment_results:
    results_text += (f"{r['reorder_point']:>13} | "
                     f"{r['avg_service_level_pct']:>13}% | "
                     f"{r['avg_stockout_days']:>13} | "
                     f"£{r['avg_holding_cost_gbp']:>11,.0f}\n")

messages = [
    {
        "role": "system",
        "content": (
            "You are a supply chain analyst writing for an academic dissertation. "
            "Analyse results precisely. Identify the optimal policy. "
            "Explain the trade-off between service level and holding cost. "
            "Note that a seasonal demand spike was included in weeks 6 and 7."
        )
    },
    {
        "role": "user",
        "content": (
            f"Here are the results of a Monte Carlo inventory simulation "
            f"(20 trials per policy, 90 days each, seasonal demand spike at week 6-7):\n\n"
            f"{results_text}\n"
            f"Demand mean: {DEMAND_MEAN} units/day, std: {DEMAND_STD}, "
            f"lead time: {LEAD_TIME} days, order quantity: {ORDER_QTY}.\n\n"
            f"Which reorder point gives the best balance of service level and cost? "
            f"What does the seasonal spike reveal? "
            f"What would you recommend and why?"
        )
    }
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("\nGPT's analysis:")
print(response.choices[0].message.content)
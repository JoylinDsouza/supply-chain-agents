import numpy as np
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── THE SIMULATION ──────────────────────────────────────────
def run_inventory_sim(reorder_point, order_quantity, lead_time_days, demand_mean, demand_std, simulation_days=90):
    """
    Simulates a Landmark Group store managing inventory over N days.
    Models realistic random demand and calculates key performance metrics.
    """
    np.random.seed(42)
    inventory = reorder_point * 2
    holding_cost = 0
    stockout_days = 0
    orders_pending = []

    for day in range(simulation_days):

        # Check if any orders arrive today
        arrived = [(d, q) for (d, q) in orders_pending if d <= day]
        for (_, qty) in arrived:
            inventory += qty
        orders_pending = [(d, q) for (d, q) in orders_pending if d > day]

        # Simulate today's demand — random, based on mean and variability
        demand = max(0, int(np.random.normal(demand_mean, demand_std)))

        # Fill demand from inventory
        if inventory >= demand:
            inventory -= demand
        else:
            stockout_days += 1
            inventory = 0

        # Pay holding cost for stock sitting in warehouse
        holding_cost += inventory * 0.5  # £0.50 per unit per day

        # Place a new order if stock is running low
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

# ── THE TOOL DESCRIPTION ────────────────────────────────────
tools = [{
    "type": "function",
    "function": {
        "name": "run_inventory_sim",
        "description": "Simulates a retail store's inventory over 90 days. Models random daily demand, stock replenishment, and calculates service level and holding costs. Use this when asked about inventory policy, reorder points, or stock management.",
        "parameters": {
            "type": "object",
            "properties": {
                "reorder_point": {
                    "type": "integer",
                    "description": "Place a new order when stock falls below this number"
                },
                "order_quantity": {
                    "type": "integer",
                    "description": "How many units to order each time"
                },
                "lead_time_days": {
                    "type": "integer",
                    "description": "How many days until the order arrives"
                },
                "demand_mean": {
                    "type": "number",
                    "description": "Average units sold per day"
                },
                "demand_std": {
                    "type": "number",
                    "description": "How unpredictable daily demand is (standard deviation)"
                }
            },
            "required": ["reorder_point", "order_quantity", "lead_time_days", "demand_mean", "demand_std"]
        }
    }
}]

# ── ASK GPT TO USE THE SIMULATION ───────────────────────────
messages = [
    {
        "role": "system",
        "content": "You are a supply chain analyst for Landmark Group. You have access to an inventory simulation tool. Always use it to get real numbers before giving advice. Never guess."
    },
    {
        "role": "user",
        "content": "A UK retail store sells around 200 units per day on average, but demand varies a lot (std dev of 40 units). Supplier takes 7 days to deliver. We currently reorder 1000 units when stock hits 300. Is this a good policy? Run the simulation and tell me."
    }
]

print("Running inventory simulation via GPT...")
print("=" * 55)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

message = response.choices[0].message

if message.tool_calls:
    tool_call = message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    print(f"GPT chose to run: {tool_call.function.name}")
    print(f"Parameters GPT picked: {args}")
    print("=" * 55)

    # Run the actual simulation
    result = run_inventory_sim(**args)

    print("Simulation results:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print("=" * 55)

    # Send results back to GPT for analysis
    messages.append(message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    print("GPT's recommendation:")
    print(final.choices[0].message.content)


# ── NOW ASK GPT TO FIND A BETTER POLICY ─────────────────────
print("\n" + "=" * 55)
print("Asking GPT to find a better policy...")
print("=" * 55)

messages2 = [
    {
        "role": "system",
        "content": "You are a supply chain analyst. You have access to an inventory simulation tool. Use it multiple times to test different policies and find the best one. Always run the simulation before giving advice."
    },
    {
        "role": "user",
        "content": "The current policy gives only 46.7% service level which is terrible. Run the simulation with these improved parameters: reorder_point=1500, order_quantity=2000, lead_time_days=7, demand_mean=200, demand_std=40. Then tell me if it is better." 
    }
]

response2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages2,
    tools=tools,
    tool_choice="auto"
)

message2 = response2.choices[0].message

if message2.tool_calls:
    tool_call2 = message2.tool_calls[0]
    args2 = json.loads(tool_call2.function.arguments)

    print(f"GPT is testing new policy: {args2}")
    print("=" * 55)

    result2 = run_inventory_sim(**args2)

    print("New policy results:")
    for key, value in result2.items():
        print(f"  {key}: {value}")
    print("=" * 55)

    messages2.append(message2)
    messages2.append({
        "role": "tool",
        "tool_call_id": tool_call2.id,
        "content": json.dumps(result2)
    })

    final2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages2
    )

    print("GPT's comparison and final recommendation:")
    print(final2.choices[0].message.content)
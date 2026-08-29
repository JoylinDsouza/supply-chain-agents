# agents/cost_analyst.py
# The Cost Analyst Agent
# Uses Simulation 1 (inventory) and Simulation 2 (hub location) as tools.
# Answers cost and investment questions by running real simulations.

import json
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulations.inventory_sim import run_inventory_sim
from simulations.hub_location_sim import run_hub_location_sim

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Tool definitions ──────────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_inventory_sim",
            "description": (
                "Simulates a retail store's inventory replenishment policy over 90 days. "
                "Use this when asked about stock levels, reorder points, order quantities, "
                "holding costs, shortage costs, or optimal inventory policies. "
                "Returns service level, stockout days, and full cost breakdown."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "demand_mean":           {"type": "number",  "description": "Average daily demand in units"},
                    "demand_std":            {"type": "number",  "description": "Daily demand variability (std dev)"},
                    "lead_time_days":        {"type": "integer", "description": "Days between placing and receiving an order"},
                    "reorder_point":         {"type": "integer", "description": "Place a new order when stock falls to this level"},
                    "order_quantity":        {"type": "integer", "description": "Number of units to order each time"},
                    "holding_cost_per_unit": {"type": "number",  "description": "GBP cost to store one unit per day"},
                    "order_fixed_cost":      {"type": "number",  "description": "Fixed GBP cost per order placed"},
                    "shortage_cost_per_unit":{"type": "number",  "description": "GBP cost per unit of unmet demand"},
                    "seasonal_spike":        {"type": "boolean", "description": "Whether to include a seasonal demand spike"},
                    "trials":                {"type": "integer", "description": "Number of Monte Carlo trials (default 20)"}
                },
                "required": ["demand_mean", "reorder_point", "order_quantity"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_hub_location_sim",
            "description": (
                "Models the financial case for opening a distribution hub in a specific location. "
                "Use this when asked about hub investment decisions, NPV, break-even year, "
                "or whether a new distribution centre is financially justified. "
                "Returns NPV, break-even year, probability of profitability, and recommendation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_location":             {"type": "string", "description": "Name of the location e.g. Germany, Poland"},
                    "build_cost_millions":            {"type": "number", "description": "One-off capital cost in GBP millions"},
                    "annual_ops_cost_millions":       {"type": "number", "description": "Annual operating cost in GBP millions"},
                    "current_freight_cost_millions":  {"type": "number", "description": "Current annual freight spend in GBP millions"},
                    "freight_saving_pct":             {"type": "number", "description": "Fraction of freight cost saved by the hub e.g. 0.18"},
                    "demand_growth_rate":             {"type": "number", "description": "Expected annual demand growth rate e.g. 0.10"},
                    "discount_rate":                  {"type": "number", "description": "Cost of capital e.g. 0.08"},
                    "years":                          {"type": "integer","description": "Number of years to model"},
                    "trials":                         {"type": "integer","description": "Monte Carlo trials (default 100)"}
                },
                "required": ["candidate_location", "build_cost_millions",
                             "annual_ops_cost_millions", "freight_saving_pct",
                             "demand_growth_rate"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "run_inventory_sim":    run_inventory_sim,
    "run_hub_location_sim": run_hub_location_sim
}

SYSTEM_PROMPT = """You are a senior supply chain cost analyst with deep expertise in \
inventory management and distribution network investment.

When given a question:
1. Think carefully about which simulation(s) you need to run
2. Choose realistic parameters based on the context given
3. Call the simulation tool(s) with your chosen parameters
4. Read the results carefully
5. If the results suggest a better set of parameters to try, run the simulation again
6. Give a clear, justified recommendation backed by the simulation numbers

Always explain your reasoning before calling a tool, and always interpret \
the results after receiving them. Never guess numbers — always use the simulation."""


def run_cost_analyst(question: str, verbose: bool = True) -> str:
    """
    Runs the Cost Analyst agent on a question.
    Returns the final answer as a string.
    Logs every tool call if verbose=True.
    """
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": question}
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"COST ANALYST: {question}")
        print(f"{'='*60}")

    iteration = 0
    # Max 6 iterations: agent typically needs 2-3 simulation calls plus
    # a reasoning step before producing a final answer. 6 provides headroom
    # without risking runaway API costs.
    max_iterations = 6

    while iteration < max_iterations:
        iteration += 1
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2    # Low temperature (0.2) for deterministic, analytical outputs.
            # Higher values introduce variability unsuitable for quantitative reasoning.
        )
        message = response.choices[0].message
        messages.append(message)

        # No tool calls — agent is done
        if not message.tool_calls:
            if verbose:
                print(f"\nFINAL ANSWER:\n{message.content}")
            return message.content

        # Process each tool call
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            args    = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"\n  → Agent calling: {fn_name}")
                print(f"    Parameters: {json.dumps(args, indent=6)}")

            # Run the actual simulation
            result = TOOL_FUNCTIONS[fn_name](**args)

            if verbose:
                print(f"    Results: {json.dumps(result.get('results', result), indent=6)}")

            # Send result back to agent
            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      json.dumps(result)
            })

    return "Max iterations reached — agent could not complete the task."


# ── Test when run directly ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Cost Analyst Agent — Test Run")
    print("=" * 60)

    # Question 1: Inventory policy
    run_cost_analyst(
        "A retailer sells on average 250 units per day with high variability "
        "(std dev of 60 units). Their supplier takes 10 days to deliver. "
        "What reorder point and order quantity would you recommend to achieve "
        "at least 95% service level while keeping total costs reasonable? "
        "Run at least two scenarios and compare them."
    )

    print("\n" + "="*60 + "\n")

    # Question 2: Hub location
    run_cost_analyst(
        "We are considering opening a distribution hub in either Germany or Poland. "
        "Germany would cost £12M to build with £2.5M annual operating cost and "
        "would save 18% of our £12M annual freight spend. "
        "Poland would cost £7M to build with £1.6M annual operating cost and "
        "would save 13% of freight spend. "
        "We expect 12% annual demand growth. Which location is the better investment?"
    )
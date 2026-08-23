# agents/risk_analyst.py
# The Risk Analyst Agent
# Uses Simulation 3 (supplier disruption) as its primary tool.
# Answers risk and resilience questions by running real simulations.

import json
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulations.supplier_disruption_sim import run_supplier_disruption_sim

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_supplier_disruption_sim",
            "description": (
                "Simulates a retail supply chain facing a probabilistic supplier disruption "
                "over 365 days. Tests one of four backup strategies: "
                "'no_backup' (do nothing), 'safety_stock' (hold extra weeks of inventory), "
                "'dual_sourcing' (use a second supplier at a cost premium), or "
                "'air_freight' (switch to air freight during a disruption). "
                "Use this when asked about supply chain risk, supplier failure, "
                "resilience strategies, or disruption costs. "
                "Returns total cost, service level, and breakdown of shortage vs strategy costs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["no_backup", "safety_stock", "dual_sourcing", "air_freight"],
                        "description": "The backup strategy to evaluate"
                    },
                    "daily_demand": {
                        "type": "number",
                        "description": "Average daily demand in units"
                    },
                    "disruption_probability": {
                        "type": "number",
                        "description": "Annual probability of a disruption occurring (0-1, e.g. 0.20 = 20%)"
                    },
                    "disruption_duration_days": {
                        "type": "integer",
                        "description": "Average number of days a disruption lasts"
                    },
                    "unit_cost": {
                        "type": "number",
                        "description": "Cost per unit from the primary supplier (GBP)"
                    },
                    "shortage_cost_per_unit": {
                        "type": "number",
                        "description": "Cost per unit of unmet demand (GBP)"
                    },
                    "dual_sourcing_premium": {
                        "type": "number",
                        "description": "Price premium for the backup supplier e.g. 0.15 = 15% more expensive"
                    },
                    "air_freight_premium": {
                        "type": "number",
                        "description": "Cost multiplier for air freight vs sea e.g. 2.5 = 2.5x more expensive"
                    },
                    "safety_stock_weeks": {
                        "type": "integer",
                        "description": "Number of weeks of extra stock to hold (safety_stock strategy only)"
                    },
                    "trials": {
                        "type": "integer",
                        "description": "Number of Monte Carlo trials (default 50)"
                    }
                },
                "required": ["strategy", "disruption_probability"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "run_supplier_disruption_sim": run_supplier_disruption_sim
}

SYSTEM_PROMPT = """You are a senior supply chain risk analyst specialising in \
supplier resilience and disruption management.

When given a question about supply chain risk:
1. Identify the disruption scenario — what is the probability and severity?
2. Test ALL relevant backup strategies by calling the simulation for each one
3. Compare the strategies on total cost AND service level — never optimise for just one
4. Identify the crossover point: at what disruption probability does each strategy become optimal?
5. Give a clear recommendation with the financial justification

Always test at least two strategies before recommending one. \
Never guess costs — always use the simulation tool."""


def run_risk_analyst(question: str, verbose: bool = True) -> str:
    """
    Runs the Risk Analyst agent on a question.
    Returns the final answer as a string.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question}
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"RISK ANALYST: {question}")
        print(f"{'='*60}")

    iteration = 0
    max_iterations = 8

    while iteration < max_iterations:
        iteration += 1
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2
        )
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            if verbose:
                print(f"\nFINAL ANSWER:\n{message.content}")
            return message.content

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            args    = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"\n  → Agent calling: {fn_name}")
                print(f"    Strategy: {args.get('strategy', 'N/A')}")
                print(f"    Disruption prob: {args.get('disruption_probability', 'N/A')}")

            result = TOOL_FUNCTIONS[fn_name](**args)

            if verbose:
                r = result.get("results", {})
                print(f"    Service level: {r.get('avg_service_level_pct')}%")
                print(f"    Total cost: £{r.get('avg_total_cost_gbp'):,.0f}")

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      json.dumps(result)
            })

    return "Max iterations reached."


if __name__ == "__main__":
    print("=" * 60)
    print("Risk Analyst Agent — Test Run")
    print("=" * 60)

    run_risk_analyst(
        "Our main supplier in Vietnam has been increasingly unreliable. "
        "We estimate there is a 25% chance of a major disruption per year, "
        "with disruptions typically lasting 6 weeks (42 days). "
        "We sell 200 units per day at £12 per unit. "
        "Should we invest in dual sourcing, build up safety stock, "
        "or rely on air freight during disruptions? "
        "What is the cheapest strategy that maintains acceptable service levels?"
    )
# agents/risk_analyst.py
# The Risk Analyst Agent
# Uses Simulation 3 (supplier disruption) as its primary tool.
# Answers risk and resilience questions by running real simulations.

import json
import os
import sys

from openai import OpenAI
from dotenv import load_dotenv

# Make project root importable when this file is executed directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulations.supplier_disruption_sim import run_supplier_disruption_sim


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_ITERATIONS = 8


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_supplier_disruption_sim",
            "description": (
                "Runs the canonical Monte Carlo supplier-disruption simulation "
                "over a 365-day horizon. Evaluates one of four resilience "
                "strategies: 'no_backup', 'safety_stock', 'dual_sourcing', "
                "or 'air_freight'. Use this tool for supplier disruption, "
                "supply chain risk, resilience, backup strategy, service-level, "
                "or disruption-cost questions. The simulator internally controls "
                "its canonical Monte Carlo configuration of 1000 trials with "
                "seed 42. Do not attempt to provide trials or seed. The result "
                "includes total cost, procurement cost, shortage cost, holding "
                "cost, strategy cost, service level, and related metrics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": [
                            "no_backup",
                            "safety_stock",
                            "dual_sourcing",
                            "air_freight"
                        ],
                        "description": (
                            "Resilience strategy to evaluate. "
                            "'no_backup' means no additional resilience measure; "
                            "'safety_stock' holds extra inventory; "
                            "'dual_sourcing' uses a second supplier; "
                            "'air_freight' switches to air freight during disruption."
                        )
                    },
                    "daily_demand": {
                        "type": "number",
                        "description": "Average daily demand in units."
                    },
                    "disruption_probability": {
                        "type": "number",
                        "description": (
                            "Annual probability of a supplier disruption, "
                            "between 0 and 1. For example, 0.20 represents 20%."
                        )
                    },
                    "disruption_duration_days": {
                        "type": "integer",
                        "description": (
                            "Average duration of a supplier disruption in days."
                        )
                    },
                    "unit_cost": {
                        "type": "number",
                        "description": (
                            "Normal procurement cost per unit from the primary "
                            "supplier in GBP."
                        )
                    },
                    "shortage_cost_per_unit": {
                        "type": "number",
                        "description": (
                            "Cost per unit of unmet demand in GBP."
                        )
                    },
                    "dual_sourcing_premium": {
                        "type": "number",
                        "description": (
                            "Price premium for dual sourcing. "
                            "For example, 0.15 represents a 15% premium."
                        )
                    },
                    "air_freight_premium": {
                        "type": "number",
                        "description": (
                            "Air-freight cost multiplier relative to normal "
                            "procurement. For example, 2.5 represents 2.5 times "
                            "the normal procurement cost."
                        )
                    },
                    "safety_stock_weeks": {
                        "type": "integer",
                        "description": (
                            "Number of weeks of additional inventory for the "
                            "safety-stock strategy."
                        )
                    }
                },
                "required": [
                    "strategy",
                    "disruption_probability"
                ]
            }
        }
    }
]


TOOL_FUNCTIONS = {
    "run_supplier_disruption_sim": run_supplier_disruption_sim
}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a senior supply chain risk analyst specialising in supplier
resilience and disruption management.

Your responsibility is to use Simulation 3, the supplier-disruption
Monte Carlo simulator, to provide evidence-based risk recommendations.

CORE RULES
----------

1. ALWAYS use the simulation tool for supply-chain disruption and resilience
   questions. Do not invent numerical results.

2. Test ALL FOUR resilience strategies when comparing strategies:
   - no_backup
   - safety_stock
   - dual_sourcing
   - air_freight

3. Compare strategies using BOTH:
   - total cost
   - service level

   Never recommend a strategy based on cost alone when a service-level
   requirement has been stated.

4. If the user gives an explicit service-level target, treat it as a hard
   constraint. The preferred strategy should be the lowest-cost strategy
   that satisfies the stated service-level requirement, unless the user
   explicitly asks for a different objective.

5. If no explicit service-level target is given, compare the cost-service
   trade-off and explain which strategy provides the most appropriate
   balance under the simulated assumptions.

6. Use these defaults when the user does not provide specific values:
   - daily_demand: 200 units
   - unit_cost: 12.0 GBP
   - shortage_cost_per_unit: 8.0 GBP
   - disruption_duration_days: 42 days
   - dual_sourcing_premium: 0.15
   - air_freight_premium: 2.5
   - safety_stock_weeks: 4

7. The simulator controls its own canonical Monte Carlo configuration:
   - 1000 trials
   - random seed 42
   - 365-day simulation horizon

   Do NOT attempt to pass trials or seed to the simulation tool.

8. The simulation's cost breakdown includes:
   - procurement cost
   - shortage cost
   - holding cost
   - strategy-specific cost

   Consider total cost when making the overall financial comparison.

9. Be explicit about the distinction between:
   - simulated results
   - assumptions supplied by the user
   - the analyst's interpretation

10. Do not describe the simulation as proving that a strategy is universally
    optimal. Recommendations apply only under the simulated assumptions and
    evaluated scenario.

11. If asked for a disruption-probability crossover point, do not infer the
    crossover from a single probability. Run simulations at multiple relevant
    disruption probabilities and compare the resulting strategies.

12. If asked to compare several disruption probabilities, run the simulation
    for the relevant probabilities and strategies rather than estimating
    missing values manually.

13. If the simulation produces percentile or distribution information, describe
    it as an outcome range or percentile range. Do not call P10/P90 a
    confidence interval.

14. Low-temperature model generation is intended to encourage consistency,
    but do not claim that it guarantees deterministic LLM behaviour.

15. Never ask the user for information that can reasonably be handled using
    the defaults above. Proceed with the available information and clearly
    state important assumptions in the final answer.

RECOMMENDATION STYLE
--------------------

After running the required simulations:

- Give a concise comparison of the strategies.
- Report the important simulated cost and service metrics.
- Identify the cheapest feasible strategy when a service target exists.
- Explain important trade-offs.
- State the recommendation clearly.
- State that the recommendation is conditional on the simulated assumptions.

Never guess numerical results. Always obtain numerical evidence from the
simulation tool.
"""


# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------

def run_risk_analyst(question: str, verbose: bool = True) -> str:
    """
    Run the Risk Analyst agent on a natural-language question.

    The agent can call Simulation 3 multiple times so that it can compare
    resilience strategies or evaluate different disruption probabilities.
    """

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": question
        }
    ]

    if verbose:
        print("\n" + "=" * 60)
        print("RISK ANALYST")
        print("=" * 60)
        print(f"Question: {question}")
        print(f"Model: {MODEL}")

    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1

        if verbose:
            print(f"\n--- Agent iteration {iteration}/{MAX_ITERATIONS} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=TEMPERATURE
        )

        message = response.choices[0].message
        messages.append(message)

        # ---------------------------------------------------------------
        # No further tool calls: return the final answer.
        # ---------------------------------------------------------------

        if not message.tool_calls:
            if verbose:
                print("\nFINAL ANSWER")
                print("-" * 60)
                print(message.content)
                print("-" * 60)

            return message.content

        # ---------------------------------------------------------------
        # Execute requested simulation calls.
        # ---------------------------------------------------------------

        for tool_call in message.tool_calls:

            fn_name = tool_call.function.name

            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as exc:
                error_message = (
                    f"Invalid tool arguments generated by the model: {exc}"
                )

                if verbose:
                    print(f"\nERROR: {error_message}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            {
                                "error": error_message
                            }
                        )
                    }
                )

                continue

            if fn_name not in TOOL_FUNCTIONS:
                error_message = f"Unknown tool requested: {fn_name}"

                if verbose:
                    print(f"\nERROR: {error_message}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            {
                                "error": error_message
                            }
                        )
                    }
                )

                continue

            if verbose:
                print(f"\n→ Agent calling: {fn_name}")
                print(
                    f"  Strategy: "
                    f"{args.get('strategy', 'N/A')}"
                )
                print(
                    f"  Disruption probability: "
                    f"{args.get('disruption_probability', 'N/A')}"
                )

            try:
                result = TOOL_FUNCTIONS[fn_name](**args)
            except Exception as exc:
                error_message = (
                    f"Simulation tool failed: {type(exc).__name__}: {exc}"
                )

                if verbose:
                    print(f"  ERROR: {error_message}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(
                            {
                                "error": error_message
                            }
                        )
                    }
                )

                continue

            if verbose:
                result_data = result.get("results", {})

                service_level = result_data.get(
                    "avg_service_level_pct"
                )

                total_cost = result_data.get(
                    "avg_total_cost_gbp"
                )

                procurement_cost = result_data.get(
                    "avg_procurement_cost_gbp"
                )

                shortage_cost = result_data.get(
                    "avg_shortage_cost_gbp"
                )

                holding_cost = result_data.get(
                    "avg_holding_cost_gbp"
                )

                strategy_cost = result_data.get(
                    "avg_strategy_cost_gbp"
                )

                print(
                    f"  Service level: "
                    f"{service_level}%"
                )

                if total_cost is not None:
                    print(
                        f"  Total cost: "
                        f"£{total_cost:,.0f}"
                    )

                if procurement_cost is not None:
                    print(
                        f"  Procurement cost: "
                        f"£{procurement_cost:,.0f}"
                    )

                if shortage_cost is not None:
                    print(
                        f"  Shortage cost: "
                        f"£{shortage_cost:,.0f}"
                    )

                if holding_cost is not None:
                    print(
                        f"  Holding cost: "
                        f"£{holding_cost:,.0f}"
                    )

                if strategy_cost is not None:
                    print(
                        f"  Strategy cost: "
                        f"£{strategy_cost:,.0f}"
                    )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                }
            )

    # -----------------------------------------------------------------------
    # Safety fallback if the model keeps requesting tools.
    # -----------------------------------------------------------------------

    fallback = (
        "The Risk Analyst reached the maximum number of reasoning iterations "
        "before producing a final recommendation."
    )

    if verbose:
        print(f"\n{fallback}")

    return fallback


# ---------------------------------------------------------------------------
# Direct test
# ---------------------------------------------------------------------------

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
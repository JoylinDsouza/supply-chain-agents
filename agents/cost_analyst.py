# agents/cost_analyst.py
# The Cost Analyst Agent
#
# Uses:
#   Simulation 1: Inventory optimisation
#   Simulation 2: Distribution hub investment
#
# Experimental design:
#   - Inventory optimisation uses the canonical numerical optimiser.
#   - The inventory optimiser controls its own search space, trial count
#     and random seed.
#   - Hub investment analysis uses the canonical hub simulation.
#   - The LLM interprets simulation results rather than controlling the
#     numerical experiment.
#
# Reproducibility:
#   - OPENAI_MODEL controls the selected LLM.
#   - Temperature is low but does not guarantee determinism.
#   - Simulation 1 uses its canonical 1000-trial / seed-42 configuration.
#   - Simulation 2 uses its canonical simulation configuration.
#   - Tool calls and numerical results are printed as an experiment trace.


import json
import os
import sys

from openai import OpenAI
from dotenv import load_dotenv


# ─────────────────────────────────────────────────────────────────────────────
# Project imports
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from simulations.inventory_sim import (
    run_inventory_sim,
    optimise_inventory_policy
)

from simulations.hub_location_sim import (
    run_hub_location_sim
)


# ─────────────────────────────────────────────────────────────────────────────
# Environment and OpenAI client
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY was not found. "
        "Create a .env file containing OPENAI_API_KEY=your_key "
        "before running the Cost Analyst agent."
    )


MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini"
)


client = OpenAI(
    api_key=API_KEY
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool definitions
# ─────────────────────────────────────────────────────────────────────────────

TOOLS = [

    # =========================================================================
    # Simulation 1A: Evaluate one inventory policy
    # =========================================================================
    {
        "type": "function",
        "function": {
            "name": "run_inventory_sim",
            "description": (
                "Runs a Monte Carlo inventory simulation for one specified "
                "retail inventory policy. Use this tool when a particular "
                "reorder point and order quantity need to be evaluated or "
                "validated. It returns service level, fill rate, stockouts, "
                "total cost, holding cost, ordering cost and shortage cost."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "demand_mean": {
                        "type": "number",
                        "description":
                            "Average daily demand in units."
                    },

                    "demand_std": {
                        "type": "number",
                        "description":
                            "Standard deviation of daily demand in units."
                    },

                    "lead_time_days": {
                        "type": "integer",
                        "description":
                            "Supplier lead time in days."
                    },

                    "reorder_point": {
                        "type": "integer",
                        "description":
                            "Inventory level at which a replenishment "
                            "order is placed."
                    },

                    "order_quantity": {
                        "type": "integer",
                        "description":
                            "Number of units ordered whenever a "
                            "replenishment order is placed."
                    },

                    "holding_cost_per_unit": {
                        "type": "number",
                        "description":
                            "Holding cost in GBP per unit per day."
                    },

                    "order_fixed_cost": {
                        "type": "number",
                        "description":
                            "Fixed ordering cost in GBP for each "
                            "replenishment order."
                    },

                    "shortage_cost_per_unit": {
                        "type": "number",
                        "description":
                            "Penalty cost in GBP for each unit of "
                            "unmet demand."
                    },

                    "seasonal_spike": {
                        "type": "boolean",
                        "description":
                            "Whether to include the seasonal demand spike."
                    },

                    "trials": {
                        "type": "integer",
                        "description":
                            "Number of Monte Carlo trials. For final "
                            "quantitative evaluation use 1000."
                    },

                    "seed": {
                        "type": "integer",
                        "description":
                            "Random seed for reproducible evaluation."
                    }
                },

                "required": [
                    "demand_mean",
                    "reorder_point",
                    "order_quantity"
                ]
            }
        }
    },


    # =========================================================================
    # Simulation 1B: Canonical inventory optimisation
    # =========================================================================
    {
        "type": "function",
        "function": {
            "name": "optimise_inventory_policy",
            "description": (
                "Performs the canonical numerical inventory optimisation. "
                "The Python optimiser systematically evaluates its predefined "
                "ROP/OQ search space using Monte Carlo simulation. It uses "
                "1000 trials and seed 42 by default. The service-level target "
                "is treated as a hard constraint. The selected policy is the "
                "lowest-cost feasible policy within the evaluated search space. "
                "The LLM must not manually construct a competing search space "
                "or change the Monte Carlo configuration. Use this tool "
                "whenever the user asks to optimise or recommend an inventory "
                "policy under a service-level constraint."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "demand_mean": {
                        "type": "number",
                        "description":
                            "Average daily demand in units."
                    },

                    "demand_std": {
                        "type": "number",
                        "description":
                            "Standard deviation of daily demand in units."
                    },

                    "lead_time_days": {
                        "type": "integer",
                        "description":
                            "Supplier lead time in days."
                    },

                    "target_service_level": {
                        "type": "number",
                        "description":
                            "Minimum required service level in percent, "
                            "for example 95.0 for a 95% target."
                    },

                    "seasonal_spike": {
                        "type": "boolean",
                        "description":
                            "Whether to include the seasonal demand spike."
                    }
                },

                "required": [
                    "demand_mean",
                    "demand_std",
                    "lead_time_days",
                    "target_service_level"
                ]
            }
        }
    },


    # =========================================================================
    # Simulation 2: Distribution hub investment
    # =========================================================================
    {
        "type": "function",
        "function": {
            "name": "run_hub_location_sim",
            "description": (
                "Runs the canonical Monte Carlo financial simulation for a "
                "proposed distribution hub. Use this tool for questions "
                "involving hub location, capital investment, annual operating "
                "cost, freight savings, NPV, break-even, profitability, or "
                "whether a distribution centre is financially justified. "
                "The Python simulation controls its own canonical Monte Carlo "
                "configuration. Do not supply trial-count or random-seed "
                "arguments unless the tool explicitly exposes them. "
                "Compare candidate locations using simulated financial "
                "metrics rather than unsupported intuition."
            ),
            "parameters": {
                "type": "object",
                "properties": {

                    "candidate_location": {
                        "type": "string",
                        "description":
                            "Name of the candidate location."
                    },

                    "build_cost_millions": {
                        "type": "number",
                        "description":
                            "One-off construction or capital cost "
                            "in GBP millions."
                    },

                    "annual_ops_cost_millions": {
                        "type": "number",
                        "description":
                            "Annual operating cost in GBP millions."
                    },

                    "current_freight_cost_millions": {
                        "type": "number",
                        "description":
                            "Current annual freight expenditure "
                            "in GBP millions."
                    },

                    "freight_saving_pct": {
                        "type": "number",
                        "description":
                            "Fraction of freight cost expected to be saved. "
                            "For example, 0.18 means 18%."
                    },

                    "demand_growth_rate": {
                        "type": "number",
                        "description":
                            "Expected annual demand growth rate. "
                            "For example, 0.10 means 10%."
                    },

                    "discount_rate": {
                        "type": "number",
                        "description":
                            "Discount rate or cost of capital used for NPV."
                    },

                    "years": {
                        "type": "integer",
                        "description":
                            "Number of years over which the investment "
                            "is evaluated."
                    }
                },

                "required": [
                    "candidate_location",
                    "build_cost_millions",
                    "annual_ops_cost_millions",
                    "current_freight_cost_millions",
                    "freight_saving_pct",
                    "demand_growth_rate",
                    "discount_rate",
                    "years"
                ]
            }
        }
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# Mapping from tool name to Python function
# ─────────────────────────────────────────────────────────────────────────────

TOOL_FUNCTIONS = {

    "run_inventory_sim":
        run_inventory_sim,

    "optimise_inventory_policy":
        optimise_inventory_policy,

    "run_hub_location_sim":
        run_hub_location_sim
}


# ─────────────────────────────────────────────────────────────────────────────
# Agent system prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a senior supply chain cost analyst with expertise in inventory
management, operations research, Monte Carlo simulation, and distribution
network investment.

Your responsibility is to analyse the cost and financial dimensions of
supply-chain decisions using Simulation 1 and Simulation 2.

Follow this process:

1. Understand the decision being asked.

2. Identify the relevant simulation tool or tools.

3. Extract the numerical assumptions provided by the user.

4. Choose reasonable parameters only where the user has not specified them.
   Clearly state important assumptions in the final answer.

5. For inventory optimisation questions, ALWAYS use the
   `optimise_inventory_policy` tool.

6. Do not manually construct an alternative inventory search when the
   optimiser is applicable.

7. The inventory optimiser controls its canonical numerical configuration:
      - 1000 Monte Carlo trials
      - random seed 42
      - predefined ROP search space
      - predefined OQ search space

   Do not claim that these settings were selected by the LLM.

8. Treat the inventory service-level target as a genuine constraint:

       service level >= target service level

   A cheaper policy that violates the target is not a feasible recommendation.

9. The inventory optimiser returns the lowest-cost feasible policy within
   its evaluated search space. Do not describe it as a global optimum outside
   that search space.

10. If useful, validate a selected inventory policy with
    `run_inventory_sim`. If doing so, use the same assumptions and clearly
    state that it is a validation run.

11. For hub investment questions, use `run_hub_location_sim`.

12. Do not manually calculate or replace the hub simulation with unsupported
    intuition when the simulation is applicable.

13. The hub simulation controls its own numerical experiment configuration.
    Do not invent or pass unsupported trial-count or random-seed parameters.

14. For comparisons, use comparable assumptions wherever possible.

15. Base recommendations on numerical simulation results rather than
    unsupported intuition.

16. Distinguish clearly between:
      - assumptions,
      - simulated results,
      - recommendation.

17. Monte Carlo results are stochastic estimates produced under the
    implemented simulation model. Do not claim that simulation guarantees
    correctness.

18. Do not describe low LLM temperature as making the LLM deterministic.

19. For inventory recommendations, report:
      - reorder point,
      - order quantity,
      - service level,
      - fill rate,
      - total cost,
      - holding cost,
      - ordering cost,
      - shortage cost where available.

20. When the optimiser reports an unconstrained cheapest policy that violates
    the service target, explicitly distinguish it from the feasible policy.

21. If no feasible inventory policy exists within the evaluated search space,
    state that clearly. Do not silently relax the service-level constraint.

22. For hub investment questions, report relevant financial metrics such as:
      - average NPV,
      - outcome P10/P90 where available,
      - break-even year,
      - probability of profitability,
      - annual cash-flow information where useful.

23. P10 and P90 are outcome percentiles from the Monte Carlo simulation.
    Do not describe them as confidence intervals.

24. If comparing candidate hub locations, evaluate each candidate using
    comparable assumptions and then explain which candidate performs better
    under the simulated assumptions.

25. Never invent simulation results.

26. Keep final answers concise but sufficiently detailed to justify the
    decision.

27. Remember that the Cost Analyst is responsible for cost and financial
    analysis. Supplier-disruption resilience analysis is handled separately
    by the Risk Analyst.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Cost Analyst execution function
# ─────────────────────────────────────────────────────────────────────────────

def run_cost_analyst(
    question: str,
    verbose: bool = True
) -> str:
    """
    Run the Cost Analyst agent on a supply-chain question.
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

        print("\n" + "=" * 70)
        print("COST ANALYST")
        print("=" * 70)

        print(
            f"Question: {question}"
        )

        print(
            f"Model: {MODEL}"
        )

        print("=" * 70)

    iteration = 0
    max_iterations = 6

    while iteration < max_iterations:

        iteration += 1

        if verbose:

            print(
                f"\n--- Agent iteration "
                f"{iteration}/{max_iterations} ---"
            )

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2
        )

        message = response.choices[0].message

        messages.append(message)

        # ---------------------------------------------------------------------
        # Final answer
        # ---------------------------------------------------------------------

        if not message.tool_calls:

            final_answer = message.content or ""

            if verbose:

                print("\nFINAL ANSWER")
                print("-" * 70)
                print(final_answer)
                print("-" * 70)

            return final_answer

        # ---------------------------------------------------------------------
        # Process tool calls
        # ---------------------------------------------------------------------

        for tool_call in message.tool_calls:

            fn_name = tool_call.function.name

            try:

                args = json.loads(
                    tool_call.function.arguments
                )

            except json.JSONDecodeError as exc:

                error_message = (
                    f"Invalid JSON arguments generated for tool "
                    f"{fn_name}: {exc}"
                )

                if verbose:
                    print(
                        f"\nERROR: {error_message}"
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({
                        "error": error_message
                    })
                })

                continue

            if verbose:

                print(
                    f"\n→ Agent calling: {fn_name}"
                )

                print(
                    "  Parameters:\n"
                    + json.dumps(
                        args,
                        indent=4
                    )
                )

            # -----------------------------------------------------------------
            # Check tool exists
            # -----------------------------------------------------------------

            if fn_name not in TOOL_FUNCTIONS:

                error_message = (
                    f"Unknown simulation tool requested: "
                    f"{fn_name}"
                )

                if verbose:

                    print(
                        f"  ERROR: {error_message}"
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({
                        "error": error_message
                    })
                })

                continue

            # -----------------------------------------------------------------
            # Run numerical simulation
            # -----------------------------------------------------------------

            try:

                result = TOOL_FUNCTIONS[
                    fn_name
                ](**args)

            except Exception as exc:

                error_message = (
                    f"Simulation tool {fn_name} failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                if verbose:

                    print(
                        f"  ERROR: {error_message}"
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({
                        "error": error_message
                    })
                })

                continue

            # -----------------------------------------------------------------
            # Display results in experiment trace
            # -----------------------------------------------------------------

            if verbose:

                display_result = (
                    result.get(
                        "results",
                        result
                    )
                    if isinstance(result, dict)
                    else result
                )

                print(
                    "  Results:\n"
                    + json.dumps(
                        display_result,
                        indent=4,
                        default=str
                    )
                )

            # -----------------------------------------------------------------
            # Return simulation results to LLM
            # -----------------------------------------------------------------

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(
                    result,
                    default=str
                )
            })

    # -------------------------------------------------------------------------
    # Maximum iterations reached
    # -------------------------------------------------------------------------

    warning = (
        "Maximum agent iterations reached before a final recommendation "
        "could be produced."
    )

    if verbose:

        print(
            f"\nWARNING: {warning}"
        )

    return warning


# ─────────────────────────────────────────────────────────────────────────────
# Direct test execution
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 70)
    print("Cost Analyst Agent - Test Run")
    print("=" * 70)

    print(
        f"Model: {MODEL}"
    )

    print("=" * 70)

    # =========================================================================
    # Question 1: Inventory policy
    # =========================================================================

    run_cost_analyst(
        "A retailer sells on average 250 units per day with high variability "
        "(standard deviation of 60 units). Their supplier takes 10 days to "
        "deliver. What reorder point and order quantity would you recommend "
        "to achieve at least 95% service level while keeping total costs "
        "reasonable? Run at least two scenarios and compare them.",
        verbose=True
    )

    print(
        "\n" + "=" * 70 + "\n"
    )

    # =========================================================================
    # Question 2: Hub location
    # =========================================================================

    run_cost_analyst(
        "We are considering opening a distribution hub in either Germany or "
        "Poland. Germany would cost £12M to build with £2.5M annual operating "
        "cost and would save 18% of our £12M annual freight spend. Poland would "
        "cost £7M to build with £1.6M annual operating cost and would save 13% "
        "of freight spend. We expect 12% annual demand growth. Use a discount "
        "rate of 8% and a 10-year evaluation horizon. Which location is the "
        "better investment under the simulated assumptions?",
        verbose=True
    )
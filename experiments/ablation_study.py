# experiments/ablation_study.py
#
# Ablation study:
# Single agent with access to all simulation tools vs the full
# multi-agent system (Orchestrator + specialist agents).
#
# Purpose:
# Isolate whether explicit multi-agent coordination adds value over
# a single LLM agent that has direct access to the same simulation tools.

import json
import os
import sys
from datetime import datetime

from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulations.inventory_sim import run_inventory_sim
from simulations.hub_location_sim import run_hub_location_sim
from simulations.supplier_disruption_sim import run_supplier_disruption_sim
from agents.orchestrator import run_orchestrator


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY was not found. "
        "Please check the .env file in the project root."
    )

client = OpenAI(api_key=api_key)

MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2
MAX_ITERATIONS = 10

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "ablation")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------
#
# These schemas are deliberately aligned with the current simulation
# implementations.
#
# The hub and supplier simulations control their own canonical
# Monte Carlo configuration, so trials/seed are NOT exposed as LLM
# parameters for those tools.
# ---------------------------------------------------------------------

ALL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_inventory_sim",
            "description": (
                "Simulates an inventory replenishment policy over a "
                "90-day horizon using Monte Carlo trials. Use for "
                "inventory policy, reorder point, order quantity, "
                "service level, stockout and cost questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "demand_mean": {
                        "type": "number",
                        "description": "Average daily demand in units."
                    },
                    "demand_std": {
                        "type": "number",
                        "description": "Standard deviation of daily demand."
                    },
                    "reorder_point": {
                        "type": "integer",
                        "description": "Reorder point in units."
                    },
                    "order_quantity": {
                        "type": "integer",
                        "description": "Order quantity in units."
                    },
                    "lead_time_days": {
                        "type": "integer",
                        "description": "Supplier lead time in days."
                    },
                    "holding_cost_per_unit_per_day": {
                        "type": "number",
                        "description": (
                            "Holding cost per unit per day. "
                            "Use the value supplied by the user."
                        )
                    },
                    "ordering_cost": {
                        "type": "number",
                        "description": (
                            "Fixed ordering cost per replenishment order."
                        )
                    },
                    "shortage_cost_per_unit": {
                        "type": "number",
                        "description": (
                            "Cost incurred per unit of unmet demand."
                        )
                    },
                    "seasonal_spike_multiplier": {
                        "type": "number",
                        "description": (
                            "Multiplier applied during seasonal spike "
                            "periods. Use 1.0 if no spike is specified."
                        )
                    },
                    "trials": {
                        "type": "integer",
                        "description": (
                            "Number of Monte Carlo trials. Use 1000 "
                            "for the canonical experiment."
                        )
                    },
                    "seed": {
                        "type": "integer",
                        "description": (
                            "Random seed. Use 42 for the canonical experiment."
                        )
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
    {
        "type": "function",
        "function": {
            "name": "run_hub_location_sim",
            "description": (
                "Models the financial case for opening a distribution "
                "hub using the canonical Monte Carlo simulation. "
                "Use for hub investment, NPV, profitability and "
                "break-even questions. The simulator controls its "
                "own canonical trial count and random seed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "candidate_location": {
                        "type": "string",
                        "description": "Candidate hub location."
                    },
                    "build_cost_millions": {
                        "type": "number",
                        "description": "Initial build cost in millions of GBP."
                    },
                    "annual_ops_cost_millions": {
                        "type": "number",
                        "description": (
                            "Annual operating cost in millions of GBP."
                        )
                    },
                    "current_freight_cost_millions": {
                        "type": "number",
                        "description": (
                            "Current annual freight spend in millions of GBP."
                        )
                    },
                    "freight_saving_pct": {
                        "type": "number",
                        "description": (
                            "Expected freight saving as a decimal, "
                            "for example 0.13 for 13%."
                        )
                    },
                    "demand_growth_rate": {
                        "type": "number",
                        "description": (
                            "Expected annual demand growth as a decimal, "
                            "for example 0.12 for 12%."
                        )
                    },
                    "discount_rate": {
                        "type": "number",
                        "description": (
                            "Cost of capital / discount rate as a decimal."
                        )
                    },
                    "years": {
                        "type": "integer",
                        "description": "Evaluation horizon in years."
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
    },
    {
        "type": "function",
        "function": {
            "name": "run_supplier_disruption_sim",
            "description": (
                "Simulates supplier disruption and resilience "
                "strategies over a 365-day horizon. Use for supplier "
                "disruption, resilience, shortage and cost-service "
                "trade-off questions. The simulator controls its own "
                "canonical Monte Carlo trial count and random seed."
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
                        "description": "Resilience strategy to evaluate."
                    },
                    "disruption_probability": {
                        "type": "number",
                        "description": (
                            "Annual probability of supplier disruption "
                            "as a decimal."
                        )
                    },
                    "disruption_duration_days": {
                        "type": "integer",
                        "description": "Typical disruption duration in days."
                    },
                    "daily_demand": {
                        "type": "number",
                        "description": "Daily demand in units."
                    },
                    "unit_cost": {
                        "type": "number",
                        "description": "Normal procurement cost per unit."
                    },
                    "shortage_cost_per_unit": {
                        "type": "number",
                        "description": (
                            "Cost per unit of unmet demand."
                        )
                    },
                    "safety_stock_weeks": {
                        "type": "number",
                        "description": (
                            "Safety-stock coverage in weeks. "
                            "Use the user-specified value."
                        )
                    },
                    "dual_sourcing_premium": {
                        "type": "number",
                        "description": (
                            "Dual-sourcing price premium as a decimal."
                        )
                    },
                    "air_freight_premium": {
                        "type": "number",
                        "description": (
                            "Air-freight cost multiplier/premium "
                            "specified by the simulation."
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


# ---------------------------------------------------------------------
# Actual Python tool functions
# ---------------------------------------------------------------------

TOOL_FUNCTIONS = {
    "run_inventory_sim": run_inventory_sim,
    "run_hub_location_sim": run_hub_location_sim,
    "run_supplier_disruption_sim": run_supplier_disruption_sim,
}


# ---------------------------------------------------------------------
# Single-agent prompt
# ---------------------------------------------------------------------

SINGLE_AGENT_PROMPT = """
You are a senior supply chain analyst with direct access to three
simulation tools covering:

1. inventory policy optimisation,
2. distribution hub investment,
3. supplier disruption resilience.

You are being evaluated as a SINGLE AGENT. You have access to all
available simulation tools directly.

When given a question:

1. Identify which simulation or simulations are relevant.
2. Preserve every numerical parameter supplied by the user.
3. Call the appropriate simulation tools with those parameters.
4. If multiple candidate strategies or locations are mentioned,
   evaluate the relevant alternatives rather than assuming one.
5. Interpret the simulation results.
6. Give a specific recommendation supported by the simulation results.
7. State important assumptions and identify conditions under which
   the recommendation would change.
8. Do not invent simulation results.
9. Do not claim a global optimum unless the simulation actually
   establishes one. Use wording such as "lowest-cost feasible policy
   within the evaluated search space" where appropriate.
10. Distinguish simulated results from assumptions and from your own
    recommendation.

For financial questions, preserve build cost, operating cost,
freight spend, freight saving, demand growth, discount rate and
evaluation horizon.

For resilience questions, preserve disruption probability,
disruption duration, daily demand, unit cost, shortage cost and
strategy parameters.

Use the simulation tools as the primary quantitative evidence.
"""


# ---------------------------------------------------------------------
# Utility for saving traces
# ---------------------------------------------------------------------

def save_trace(trace: dict, prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(
        RESULTS_DIR,
        f"{prefix}_{timestamp}.json"
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2, ensure_ascii=False)

    return path


# ---------------------------------------------------------------------
# Single-agent execution
# ---------------------------------------------------------------------

def run_single_agent(question: str, verbose: bool = True) -> str:
    """
    Run a single LLM agent with direct access to all three simulation
    tools.

    This is the ablation condition used to compare against the
    full multi-agent Orchestrator.
    """

    messages = [
        {
            "role": "system",
            "content": SINGLE_AGENT_PROMPT
        },
        {
            "role": "user",
            "content": question
        }
    ]

    if verbose:
        print("\n" + "=" * 60)
        print("SINGLE AGENT (ABLATION)")
        print("=" * 60)
        print(question)

    trace = {
        "agent": "single_agent_ablation",
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "question": question,
        "iterations": [],
        "final_answer": None,
        "total_iterations": None,
        "status": "running"
    }

    for iteration in range(1, MAX_ITERATIONS + 1):

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=ALL_TOOLS,
                tool_choice="auto",
                temperature=TEMPERATURE
            )
        except Exception as exc:
            trace["status"] = "error"
            trace["error"] = str(exc)
            save_trace(trace, "single_agent_error")

            raise RuntimeError(
                f"Single-agent OpenAI call failed: {exc}"
            ) from exc

        message = response.choices[0].message

        # The OpenAI SDK message object can be appended to the
        # conversation in the current SDK interface.
        messages.append(message)

        # -------------------------------------------------------------
        # Final answer
        # -------------------------------------------------------------

        if not message.tool_calls:

            final_answer = message.content or ""

            trace["final_answer"] = final_answer
            trace["total_iterations"] = iteration
            trace["status"] = "completed"

            if verbose:
                print("\nSINGLE AGENT ANSWER:")
                print("-" * 60)
                print(final_answer)

            path = save_trace(
                trace,
                "single_agent"
            )

            if verbose:
                print(f"\nTrace saved to: {path}")

            return final_answer

        # -------------------------------------------------------------
        # Tool calls
        # -------------------------------------------------------------

        for tool_call in message.tool_calls:

            fn_name = tool_call.function.name

            try:
                args = json.loads(
                    tool_call.function.arguments
                )
            except json.JSONDecodeError as exc:

                error_message = (
                    f"Invalid JSON arguments for tool "
                    f"{fn_name}: {exc}"
                )

                trace["iterations"].append({
                    "iteration": iteration,
                    "tool_name": fn_name,
                    "parameters": tool_call.function.arguments,
                    "error": error_message
                })

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
                    f"\n  → Single agent calling: {fn_name}"
                )
                print(
                    f"    Parameters: {json.dumps(args)}"
                )

            # ---------------------------------------------------------
            # Unknown tool protection
            # ---------------------------------------------------------

            if fn_name not in TOOL_FUNCTIONS:

                error_message = (
                    f"Unknown tool requested: {fn_name}"
                )

                trace["iterations"].append({
                    "iteration": iteration,
                    "tool_name": fn_name,
                    "parameters": args,
                    "error": error_message
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({
                        "error": error_message
                    })
                })

                continue

            # ---------------------------------------------------------
            # Execute simulation
            # ---------------------------------------------------------

            try:

                result = TOOL_FUNCTIONS[fn_name](**args)

                result_for_trace = result.get(
                    "results",
                    result
                )

                trace["iterations"].append({
                    "iteration": iteration,
                    "tool_name": fn_name,
                    "parameters": args,
                    "result": result_for_trace
                })

                if verbose:

                    if isinstance(result, dict):
                        result_summary = result.get(
                            "results",
                            result
                        )

                        if isinstance(result_summary, dict):
                            print(
                                "    Result keys: "
                                f"{list(result_summary.keys())}"
                            )
                        else:
                            print(
                                f"    Result: {result_summary}"
                            )

            except Exception as exc:

                error_message = (
                    f"Tool execution failed for {fn_name}: "
                    f"{type(exc).__name__}: {exc}"
                )

                trace["iterations"].append({
                    "iteration": iteration,
                    "tool_name": fn_name,
                    "parameters": args,
                    "error": error_message
                })

                if verbose:
                    print(
                        f"    ERROR: {error_message}"
                    )

                result = {
                    "error": error_message
                }

            # ---------------------------------------------------------
            # Return tool result to the model
            # ---------------------------------------------------------

            try:
                tool_content = json.dumps(
                    result,
                    ensure_ascii=False
                )
            except TypeError:

                tool_content = json.dumps(
                    {
                        "error": (
                            "Simulation returned a non-serialisable "
                            "result."
                        )
                    }
                )

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": tool_content
            })

    # -----------------------------------------------------------------
    # Maximum iterations reached
    # -----------------------------------------------------------------

    final_answer = (
        "Maximum single-agent tool-use iterations reached "
        "without producing a final answer."
    )

    trace["final_answer"] = final_answer
    trace["total_iterations"] = MAX_ITERATIONS
    trace["status"] = "max_iterations"

    path = save_trace(
        trace,
        "single_agent_max_iterations"
    )

    if verbose:
        print(f"\nWARNING: {final_answer}")
        print(f"Trace saved to: {path}")

    return final_answer


# ---------------------------------------------------------------------
# Main ablation experiment
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("ABLATION STUDY")
    print("Single Agent with All Tools vs Multi-Agent System")
    print("=" * 70)

    # -------------------------------------------------------------
    # IMPORTANT:
    # The question deliberately contains the same numerical
    # information used in Experiment 4, including discount rate
    # and evaluation horizon.
    #
    # This makes the ablation comparison fair.
    # -------------------------------------------------------------

    question = (
        "We are considering opening a distribution hub in Poland "
        "(£7M build cost, £1.6M annual operating cost, 13% freight "
        "saving on £12M annual freight spend, 12% annual demand growth). "
        "Use an 8% cost of capital and a 10-year evaluation horizon. "
        "Our supplier in the region has a 20% annual disruption "
        "probability with disruptions lasting around 6 weeks. "
        "Should we proceed with the hub, and what resilience strategy "
        "should we implement alongside it? Give a complete strategic "
        "recommendation."
    )

    # -------------------------------------------------------------
    # SINGLE AGENT
    # -------------------------------------------------------------

    print("\n--- SINGLE AGENT (ALL TOOLS) ---")

    single_answer = run_single_agent(
        question,
        verbose=True
    )

    # -------------------------------------------------------------
    # MULTI-AGENT SYSTEM
    # -------------------------------------------------------------

    print("\n--- MULTI-AGENT SYSTEM ---")

    try:
        multi_answer = run_orchestrator(
            question,
            verbose=False
        )
    except Exception as exc:

        print(
            "\nERROR: Multi-agent orchestrator failed:"
            f" {type(exc).__name__}: {exc}"
        )

        multi_answer = (
            "Multi-agent execution failed: "
            f"{type(exc).__name__}: {exc}"
        )

    print("\nMULTI-AGENT ANSWER:")
    print("-" * 60)
    print(multi_answer)

    # -------------------------------------------------------------
    # Basic comparison
    # -------------------------------------------------------------

    print("\n" + "=" * 70)
    print("ABLATION COMPARISON SUMMARY")
    print("=" * 70)

    print(
        f"Single agent answer length: "
        f"{len(single_answer)} chars"
    )

    print(
        f"Multi-agent answer length:  "
        f"{len(multi_answer)} chars"
    )

    print("\nComparison question:")
    print(
        "Does explicit multi-agent coordination produce a more "
        "integrated and conditionally justified recommendation than "
        "a single agent with direct access to all simulations?"
    )

    # -------------------------------------------------------------
    # Save comparison
    # -------------------------------------------------------------

    result = {
        "experiment": "ablation_single_agent_vs_multi_agent",
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "question": question,
        "single_agent_answer": single_answer,
        "multi_agent_answer": multi_answer,
        "single_agent_length": len(single_answer),
        "multi_agent_length": len(multi_answer),
        "interpretation_note": (
            "This ablation compares architectural coordination "
            "under the same overall decision problem. Answer length "
            "is descriptive only and is not treated as a performance "
            "metric."
        )
    }

    comparison_path = os.path.join(
        RESULTS_DIR,
        "ablation_comparison.json"
    )

    with open(
        comparison_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nResults saved to: {comparison_path}"
    )

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)
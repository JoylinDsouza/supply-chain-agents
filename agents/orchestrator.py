# agents/orchestrator.py
# The Orchestrator Agent — centrepiece of the dissertation system.
# Receives a strategic supply-chain question, delegates to specialist agents,
# and synthesises their simulation-backed findings into a final recommendation.

import json
import os
import sys

from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.cost_analyst import run_cost_analyst
from agents.risk_analyst import run_risk_analyst


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.3
MAX_ITERATIONS = 6


# ---------------------------------------------------------------------------
# Specialist delegation tools
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ask_cost_analyst",
            "description": (
                "Delegates a cost, inventory, or investment question to the "
                "Cost Analyst agent. Use for inventory policy optimisation, "
                "reorder points, order quantities, holding costs, hub location "
                "NPV, break-even analysis, or distribution-centre investment "
                "comparisons. The Cost Analyst uses Simulation 1 and Simulation 2 "
                "and returns simulation-backed findings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "A self-contained cost or investment sub-question "
                            "for the Cost Analyst."
                        )
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_risk_analyst",
            "description": (
                "Delegates a supplier-risk or resilience question to the Risk "
                "Analyst agent. Use for supplier disruption probability, "
                "disruption duration, resilience strategies, dual sourcing, "
                "safety stock for disruption resilience, air freight contingency, "
                "and disruption-related costs. The Risk Analyst uses Simulation 3 "
                "and returns simulation-backed findings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "A self-contained risk or resilience question "
                            "for the Risk Analyst."
                        )
                    }
                },
                "required": ["question"]
            }
        }
    }
]


AGENT_FUNCTIONS = {
    "ask_cost_analyst": run_cost_analyst,
    "ask_risk_analyst": run_risk_analyst
}


# ---------------------------------------------------------------------------
# Orchestrator system prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are the Chief Supply Chain Strategist for a major retail group.

You coordinate specialist analysts whose numerical recommendations are backed
by discrete-event / Monte Carlo simulation. Your role is to decompose a
strategic supply-chain question, obtain the relevant specialist analyses,
and synthesise them into one coherent decision.

SPECIALIST RESPONSIBILITIES
---------------------------

COST ANALYST:
- Simulation 1: inventory policy optimisation
- Simulation 2: distribution-hub investment and location analysis
- Cost, service-level, NPV, break-even and investment questions

RISK ANALYST:
- Simulation 3: supplier disruption and resilience
- Disruption probability and duration
- No backup, safety stock, dual sourcing and air freight strategies
- Risk, resilience and disruption-cost questions

DELEGATION RULES
----------------

1. For a question involving BOTH investment/cost and supplier disruption or
   resilience, delegate to BOTH the Cost Analyst and Risk Analyst.

2. For an investment-only or inventory-only question, use the Cost Analyst.

3. For a disruption/resilience-only question, use the Risk Analyst.

4. Give each specialist a self-contained sub-question containing ALL
   decision-relevant parameters from the original user query.

   NEVER omit, alter, or replace numerical assumptions supplied by the user.

   This includes, where applicable:
   - investment/build cost
   - operating cost
   - freight cost
   - saving percentage
   - demand growth rate
   - discount rate / cost of capital
   - analysis horizon
   - disruption probability
   - disruption duration
   - demand
   - unit costs
   - service-level targets
   - resilience strategy assumptions

   Before calling a specialist, explicitly check that every numerical
   parameter relevant to that specialist's simulation has been preserved.

   If a parameter is relevant to the specialist but not specified by the user,
   allow the specialist to apply its documented default. Do not invent a
   replacement value for a parameter that WAS specified by the user.

5. Do not perform the specialists' simulation calculations yourself.

6. The specialist sub-question is a parameter-preserving translation of the
   original question, not a simplified summary. Preserve the original values
   exactly, including percentages, monetary amounts, time horizons, and
   probabilities.

7. Wait for specialist results before producing the final recommendation.

SYNTHESIS RULES
---------------

8. Treat specialist simulation outputs as the quantitative evidence.

9. Clearly distinguish:
   - assumptions supplied by the user,
   - simulated results,
   - your strategic interpretation.

10. Do not invent numerical results, probabilities, costs, thresholds, or
    crossover points that were not produced by a specialist simulation.

11. When the Cost Analyst reports an investment recommendation, preserve its
    simulated assumptions, parameter values, numerical results, and
    interpretation. Do not substitute different assumptions when synthesising
    the final answer.

12. When the Risk Analyst compares resilience strategies, preserve its
    simulated parameter values, service results, cost results, and
    trade-off interpretation.

13. Look for interactions between investment and resilience findings. For
    example, a hub investment may be financially attractive under one demand
    scenario while a supplier disruption strategy may change its operational
    risk profile.

14. Do not claim that combining two specialist findings proves causality
    between the simulated systems. The simulations address different parts
    of the supply-chain decision.

15. Recommendations are conditional on the simulated assumptions and
    evaluated scenarios.

16. Do not call a simulated policy a "global optimum". Where relevant, use
    wording such as "lowest-cost feasible policy within the evaluated search
    space".

17. Do not describe P10/P90 simulation outcomes as confidence intervals.
    They are outcome percentiles.

18. Do not claim that low-temperature LLM generation is deterministic.

19. Never ask the user for additional information. Specialists have their own
    documented defaults for unspecified parameters.

FINAL ANSWER STRUCTURE
----------------------

Produce a concise but substantive executive recommendation containing:

1. Overall decision
2. Cost / investment evidence
3. Risk / resilience evidence
4. Integrated trade-off
5. Recommended action
6. Conditions that would cause the recommendation to change

The final answer should make clear which conclusions are simulation-backed
and which are strategic interpretations.
"""


# ---------------------------------------------------------------------------
# Orchestrator execution
# ---------------------------------------------------------------------------

def run_orchestrator(question: str, verbose: bool = True) -> str:
    """
    Run the multi-agent supply-chain system.

    The Orchestrator delegates relevant sub-questions to the Cost Analyst
    and/or Risk Analyst and then synthesises their returned findings.
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
        print("\n" + "=" * 65)
        print("ORCHESTRATOR")
        print("=" * 65)
        print(f"Question: {question}")
        print(f"Model: {MODEL}")

    iteration = 0

    while iteration < MAX_ITERATIONS:
        iteration += 1

        if verbose:
            print(
                f"\n--- Orchestrator iteration "
                f"{iteration}/{MAX_ITERATIONS} ---"
            )

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
        # Final synthesis
        # ---------------------------------------------------------------

        if not message.tool_calls:
            if verbose:
                print("\n" + "=" * 65)
                print("ORCHESTRATOR FINAL RECOMMENDATION")
                print("=" * 65)
                print(message.content)
                print("=" * 65)

            return message.content

        # ---------------------------------------------------------------
        # Execute specialist calls
        # ---------------------------------------------------------------

        for tool_call in message.tool_calls:

            fn_name = tool_call.function.name

            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as exc:

                error_message = (
                    f"Invalid delegation arguments generated by the model: "
                    f"{exc}"
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

            if fn_name not in AGENT_FUNCTIONS:

                error_message = f"Unknown specialist requested: {fn_name}"

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

            sub_question = args["question"]

            if verbose:
                print(f"\n[Orchestrator → {fn_name}]")
                print(f"Sub-question: {sub_question}")
                print("-" * 65)

            try:
                result = AGENT_FUNCTIONS[fn_name](
                    sub_question,
                    verbose=False
                )
            except Exception as exc:

                error_message = (
                    f"Specialist agent failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                if verbose:
                    print(f"ERROR: {error_message}")

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
                preview = result[:500]

                if len(result) > 500:
                    preview += "..."

                print("Specialist returned:")
                print(preview)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                }
            )

    fallback = (
        "The Orchestrator reached the maximum number of reasoning iterations "
        "before producing a final integrated recommendation."
    )

    if verbose:
        print(f"\n{fallback}")

    return fallback


# ---------------------------------------------------------------------------
# Direct end-to-end test
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 65)
    print("MULTI-AGENT SUPPLY CHAIN SYSTEM")
    print("Orchestrator + Cost Analyst + Risk Analyst")
    print("=" * 65)

    run_orchestrator(
        "We are a large UK retailer evaluating whether to open a new "
        "distribution hub in Poland. The hub would cost £7M to build, "
        "£1.6M per year to operate, and would save 13% of our £12M annual "
        "freight spend. We expect 12% annual demand growth. "
        "Use an 8% cost of capital and a 10-year analysis horizon. "
        "However, our main supplier in the region has a 20% annual probability "
        "of disruption lasting around 6 weeks. "
        "Should we proceed with the Poland hub, and what resilience strategy "
        "should we adopt alongside it? "
        "Give us a complete strategic recommendation."
    )
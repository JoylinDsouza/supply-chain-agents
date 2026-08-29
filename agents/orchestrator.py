# agents/orchestrator.py
# The Orchestrator Agent — the centrepiece of the dissertation system.
# Receives a strategic supply chain question, delegates to specialist agents,
# synthesises their findings into a final recommendation.

import json
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.cost_analyst import run_cost_analyst
from agents.risk_analyst import run_risk_analyst

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ask_cost_analyst",
            "description": (
                "Delegates a cost or investment question to the Cost Analyst agent. "
                "Use for questions about: inventory policy optimisation, reorder points, "
                "order quantities, holding costs, hub location NPV, break-even analysis, "
                "or comparing distribution centre investment options. "
                "The Cost Analyst will run real simulations and return a justified answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The specific cost or investment question to investigate"
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
                "Delegates a risk or resilience question to the Risk Analyst agent. "
                "Use for questions about: supplier disruption risk, backup strategies, "
                "dual sourcing decisions, safety stock for resilience, air freight contingency, "
                "or the cost of supply chain vulnerabilities. "
                "The Risk Analyst will run real simulations and return a justified answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The specific risk or resilience question to investigate"
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

SYSTEM_PROMPT = """You are the Chief Supply Chain Strategist for a major retail group. \
You lead a team of specialist analysts and are responsible for making \
high-level strategic decisions backed by rigorous quantitative analysis.

When given a strategic question:
1. Break it into sub-questions — what cost analysis is needed? what risk analysis is needed?
2. Delegate each sub-question to the right specialist agent
3. The specialist agents will run real simulations — do not ask for more information,
   trust that they will use appropriate defaults for any unspecified parameters
4. Wait for their findings before drawing conclusions
5. Look for connections between the cost and risk findings
6. Synthesise everything into a final strategic recommendation
7. State clearly: what to do, why, and under what conditions the recommendation would change

IMPORTANT: Never ask the user for more information. Always synthesise the findings
from your specialist agents into a final recommendation, even if some parameters
were assumed. The most valuable recommendations are conditional — they identify
the specific thresholds and circumstances that change the decision.

Your job is to find insights that neither the Cost Analyst nor the Risk Analyst \
could find working alone."""


def run_orchestrator(question: str, verbose: bool = True) -> str:
    """
    Runs the full multi-agent system on a strategic question.
    Delegates to Cost Analyst and Risk Analyst as needed.
    Returns the Orchestrator's final synthesised recommendation.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question}
    ]

    if verbose:
        print(f"\n{'='*65}")
        print(f"ORCHESTRATOR RECEIVES: {question}")
        print(f"{'='*65}")

    iteration = 0
    max_iterations = 6

    while iteration < max_iterations:
        iteration += 1
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3    # Slightly higher than specialists (0.2) to allow more
            # flexible synthesis reasoning when combining cost and risk findings.
            # Still low enough to maintain analytical rigour.
        )
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            if verbose:
                print(f"\n{'='*65}")
                print("ORCHESTRATOR FINAL RECOMMENDATION:")
                print('='*65)
                print(message.content)
            return message.content

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            args    = json.loads(tool_call.function.arguments)
            sub_q   = args["question"]

            if verbose:
                print(f"\n  [Orchestrator → {fn_name}]")
                print(f"  Sub-question: {sub_q}")
                print(f"  {'-'*55}")

            # Run the specialist agent — suppress its verbose output
            result = AGENT_FUNCTIONS[fn_name](sub_q, verbose=False)

            if verbose:
                # Show a short preview of the agent's answer
                preview = result[:300] + "..." if len(result) > 300 else result
                print(f"  Agent returned: {preview}")

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      result
            })

    return "Max iterations reached."


if __name__ == "__main__":
    print("=" * 65)
    print("MULTI-AGENT SUPPLY CHAIN SYSTEM")
    print("Orchestrator + Cost Analyst + Risk Analyst")
    print("=" * 65)

    # The big strategic question — the kind your professor described
    run_orchestrator(
        "We are a large UK retailer evaluating whether to open a new "
        "distribution hub in Poland. The hub would cost £7M to build, "
        "£1.6M per year to operate, and would save 13% of our £12M annual "
        "freight spend. We expect 12% annual demand growth. "
        "However, our main supplier in the region has a 20% annual probability "
        "of disruption lasting around 6 weeks. "
        "Should we proceed with the Poland hub, and what resilience strategy "
        "should we adopt alongside it? "
        "Give us a complete strategic recommendation."
    )
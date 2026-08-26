# experiments/run_experiments.py
# Formal experiment runner for the dissertation.
# Compares the multi-agent system against a standalone LLM baseline.
# Saves all results, reasoning traces, and comparison tables to files.

import json
import os
import sys
import csv
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.orchestrator import run_orchestrator
from agents.cost_analyst import run_cost_analyst
from agents.risk_analyst import run_risk_analyst

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

os.makedirs("results/traces", exist_ok=True)
os.makedirs("results/experiments", exist_ok=True)


# ── Baseline: standalone LLM with no tools ───────────────────────────────────
def run_baseline(question: str) -> str:
    """Ask a plain LLM the same question with no tools and no simulations."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a supply chain consultant. Answer the question based on "
                    "your knowledge. Give specific numbers and a clear recommendation."
                )
            },
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


# ── Trace logger ─────────────────────────────────────────────────────────────
def save_trace(experiment_name: str, question: str,
               agent_answer: str, baseline_answer: str,
               metadata: dict = None):
    """Saves the full trace of a comparison experiment to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace = {
        "experiment": experiment_name,
        "timestamp": timestamp,
        "question": question,
        "agent_answer": agent_answer,
        "baseline_answer": baseline_answer,
        "metadata": metadata or {}
    }
    path = f"results/traces/{experiment_name}_{timestamp}.json"
    with open(path, "w") as f:
        json.dump(trace, f, indent=2)
    return path


# ── Experiment 1: Inventory policy optimisation ───────────────────────────────
def experiment_1():
    """
    Research question: Can the agent system find a better inventory policy
    than a standalone LLM when given the same scenario?
    """
    print("\n" + "="*65)
    print("EXPERIMENT 1: Inventory Policy Optimisation")
    print("="*65)

    question = (
        "A UK retailer sells on average 200 units per day with significant "
        "demand variability (standard deviation of 40 units). Their supplier "
        "takes 7 days to deliver. Each order costs £100 to place. Holding "
        "stock costs £0.50 per unit per day. Running out of stock costs "
        "£5 per unit short. They currently reorder 1000 units when stock "
        "hits 300 units. Is this a good policy? What would you recommend instead?"
    )

    print("\nRunning agent system...")
    agent_answer = run_cost_analyst(question, verbose=True)

    print("\n" + "-"*65)
    print("Running standalone LLM baseline (no tools, no simulation)...")
    baseline_answer = run_baseline(question)
    print("\nBASELINE ANSWER:")
    print(baseline_answer)

    path = save_trace(
        "exp1_inventory_policy",
        question, agent_answer, baseline_answer,
        {"domain": "inventory", "agent_type": "cost_analyst"}
    )
    print(f"\nTrace saved to: {path}")
    return agent_answer, baseline_answer


# ── Experiment 2: Hub location decision ──────────────────────────────────────
def experiment_2():
    """
    Research question: Does the agent system give a more quantitatively
    grounded hub location recommendation than a standalone LLM?
    """
    print("\n" + "="*65)
    print("EXPERIMENT 2: Distribution Hub Location Decision")
    print("="*65)

    question = (
        "We are a UK retailer considering opening a European distribution hub. "
        "We have two options: Germany (£12M to build, £2.5M annual operating cost, "
        "saves 18% of our £12M freight spend) or Poland (£7M to build, "
        "£1.6M annual operating cost, saves 13% of freight spend). "
        "We expect 12% annual demand growth. Our cost of capital is 8%. "
        "Which location is the better investment and why?"
    )

    print("\nRunning agent system...")
    agent_answer = run_cost_analyst(question, verbose=True)

    print("\n" + "-"*65)
    print("Running standalone LLM baseline...")
    baseline_answer = run_baseline(question)
    print("\nBASELINE ANSWER:")
    print(baseline_answer)

    path = save_trace(
        "exp2_hub_location",
        question, agent_answer, baseline_answer,
        {"domain": "hub_location", "agent_type": "cost_analyst"}
    )
    print(f"\nTrace saved to: {path}")
    return agent_answer, baseline_answer


# ── Experiment 3: Supplier resilience strategy ────────────────────────────────
def experiment_3():
    """
    Research question: Does the agent system correctly identify the crossover
    point where dual sourcing beats air freight?
    """
    print("\n" + "="*65)
    print("EXPERIMENT 3: Supplier Resilience Strategy")
    print("="*65)

    question = (
        "Our supply chain faces a 25% annual probability of supplier disruption, "
        "with disruptions typically lasting 6 weeks. We sell 200 units per day "
        "at £12 per unit. Unmet demand costs us £8 per unit short. "
        "We are evaluating three options: doing nothing and absorbing shortages, "
        "building a safety stock buffer of 4 weeks, or contracting a second "
        "supplier (dual sourcing) at a 15% price premium who can cover 60% of "
        "our demand during a disruption. Which strategy do you recommend and why?"
    )

    print("\nRunning agent system...")
    agent_answer = run_risk_analyst(question, verbose=True)

    print("\n" + "-"*65)
    print("Running standalone LLM baseline...")
    baseline_answer = run_baseline(question)
    print("\nBASELINE ANSWER:")
    print(baseline_answer)

    path = save_trace(
        "exp3_supplier_resilience",
        question, agent_answer, baseline_answer,
        {"domain": "supplier_disruption", "agent_type": "risk_analyst"}
    )
    print(f"\nTrace saved to: {path}")
    return agent_answer, baseline_answer


# ── Experiment 4: Full orchestrator — combined strategic question ──────────────
def experiment_4():
    """
    Research question: Can the Orchestrator synthesise cost and risk findings
    into a recommendation that neither agent could produce alone?
    """
    print("\n" + "="*65)
    print("EXPERIMENT 4: Strategic Question — Full Orchestrator")
    print("="*65)

    question = (
        "We are considering opening a distribution hub in Poland (£7M build cost, "
        "£1.6M annual operating cost, 13% freight saving on £12M spend, 12% demand growth). "
        "Our supplier in the region has a 20% annual disruption probability "
        "with disruptions lasting around 6 weeks. "
        "Should we proceed with the hub, and what resilience strategy should "
        "we implement alongside it? Give a complete strategic recommendation."
    )

    print("\nRunning full multi-agent system...")
    agent_answer = run_orchestrator(question, verbose=True)

    print("\n" + "-"*65)
    print("Running standalone LLM baseline...")
    baseline_answer = run_baseline(question)
    print("\nBASELINE ANSWER:")
    print(baseline_answer)

    path = save_trace(
        "exp4_strategic_orchestrator",
        question, agent_answer, baseline_answer,
        {"domain": "combined", "agent_type": "orchestrator"}
    )
    print(f"\nTrace saved to: {path}")
    return agent_answer, baseline_answer


# ── Comparison summary table ──────────────────────────────────────────────────
def save_comparison_table(results: list):
    """Saves a CSV summary of all experiments for the dissertation table."""
    path = "results/experiments/comparison_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment", "domain", "agent_answer_length",
            "baseline_answer_length", "agent_contains_numbers",
            "baseline_contains_numbers", "agent_gives_specific_recommendation",
            "baseline_gives_specific_recommendation"
        ])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"\nComparison table saved to: {path}")


# ── Main: run all experiments ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*65)
    print("DISSERTATION EXPERIMENT RUNNER")
    print("Multi-agent system vs Standalone LLM baseline")
    print("="*65)

    summary = []

    # Run each experiment
    exp1_agent, exp1_base = experiment_1()
    summary.append({
        "experiment": "Exp 1: Inventory policy",
        "domain": "Inventory replenishment",
        "agent_answer_length": len(exp1_agent),
        "baseline_answer_length": len(exp1_base),
        "agent_contains_numbers": any(c.isdigit() for c in exp1_agent),
        "baseline_contains_numbers": any(c.isdigit() for c in exp1_base),
        "agent_gives_specific_recommendation": "recommend" in exp1_agent.lower(),
        "baseline_gives_specific_recommendation": "recommend" in exp1_base.lower()
    })

    exp2_agent, exp2_base = experiment_2()
    summary.append({
        "experiment": "Exp 2: Hub location",
        "domain": "Hub location investment",
        "agent_answer_length": len(exp2_agent),
        "baseline_answer_length": len(exp2_base),
        "agent_contains_numbers": any(c.isdigit() for c in exp2_agent),
        "baseline_contains_numbers": any(c.isdigit() for c in exp2_base),
        "agent_gives_specific_recommendation": "recommend" in exp2_agent.lower(),
        "baseline_gives_specific_recommendation": "recommend" in exp2_base.lower()
    })

    exp3_agent, exp3_base = experiment_3()
    summary.append({
        "experiment": "Exp 3: Supplier resilience",
        "domain": "Supplier disruption",
        "agent_answer_length": len(exp3_agent),
        "baseline_answer_length": len(exp3_base),
        "agent_contains_numbers": any(c.isdigit() for c in exp3_agent),
        "baseline_contains_numbers": any(c.isdigit() for c in exp3_base),
        "agent_gives_specific_recommendation": "recommend" in exp3_agent.lower(),
        "baseline_gives_specific_recommendation": "recommend" in exp3_base.lower()
    })

    exp4_agent, exp4_base = experiment_4()
    summary.append({
        "experiment": "Exp 4: Strategic orchestrator",
        "domain": "Combined cost and risk",
        "agent_answer_length": len(exp4_agent),
        "baseline_answer_length": len(exp4_base),
        "agent_contains_numbers": any(c.isdigit() for c in exp4_agent),
        "baseline_contains_numbers": any(c.isdigit() for c in exp4_base),
        "agent_gives_specific_recommendation": "recommend" in exp4_agent.lower(),
        "baseline_gives_specific_recommendation": "recommend" in exp4_base.lower()
    })

    save_comparison_table(summary)

    print("\n" + "="*65)
    print("ALL EXPERIMENTS COMPLETE")
    print("Check results/traces/ for full reasoning traces")
    print("Check results/experiments/ for comparison summary")
    print("="*65)
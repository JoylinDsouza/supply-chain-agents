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

# Fix working directory so results always save to the project root
# regardless of where the script is called from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

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
    """
    Saves a structured comparison summary using the four evaluation dimensions
    defined in the dissertation:
    1. Quantitative grounding - specific numbers from computation
    2. Threshold identification - specific decision boundaries
    3. Conditionality - explicit conditions for the recommendation
    4. Recommendation correctness - aligns with simulation ground truth
    """
    path = "results/experiments/comparison_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "experiment",
            "domain",
            "agent_quantitative_grounding",
            "agent_threshold_identification",
            "agent_conditionality",
            "agent_correctness",
            "baseline_quantitative_grounding",
            "baseline_threshold_identification",
            "baseline_conditionality",
            "baseline_correctness",
            "agent_total_score",
            "baseline_total_score"
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

    # ── Evaluation helper functions ───────────────────────────────────────────
    def score_quantitative_grounding(text: str) -> str:
        """
        Checks whether the response contains specific computed figures
        (cost amounts, percentages, NPV values, years) rather than
        general estimates. Returns Met, Partially Met, or Not Met.
        """
        import re
        # Look for currency figures (£X,XXX or £X.XM), specific percentages,
        # and year references that indicate real computed outputs
        currency_pattern = r'£[\d,]+(\.\d+)?'
        pct_pattern = r'\b\d+\.?\d*\s*%'
        npv_pattern = r'NPV|net present value|break.?even'
        has_currency = bool(re.search(currency_pattern, text))
        has_percentage = bool(re.search(pct_pattern, text, re.IGNORECASE))
        has_financial = bool(re.search(npv_pattern, text, re.IGNORECASE))
        if has_currency and (has_percentage or has_financial):
            return "Met"
        elif has_currency or has_percentage:
            return "Partially Met"
        else:
            return "Not Met"

    def score_threshold_identification(text: str) -> str:
        """
        Checks whether the response identifies a specific numerical boundary
        at which the recommendation changes — e.g. 'below 8% growth' or
        'above 20% disruption probability'.
        """
        import re
        threshold_patterns = [
            r'below\s+\d+\.?\d*\s*%',
            r'above\s+\d+\.?\d*\s*%',
            r'less than\s+\d+\.?\d*\s*%',
            r'more than\s+\d+\.?\d*\s*%',
            r'exceed\w*\s+\d+\.?\d*\s*%',
            r'fall\w*\s+below\s+\d+',
            r'crossover',
            r'threshold',
            r'if.*growth.*falls',
            r'if.*probability.*exceed',
        ]
        matches = sum(1 for p in threshold_patterns
                      if re.search(p, text, re.IGNORECASE))
        if matches >= 2:
            return "Met"
        elif matches == 1:
            return "Partially Met"
        else:
            return "Not Met"

    def score_conditionality(text: str) -> str:
        """
        Checks whether the recommendation explicitly states conditions
        under which it applies or should be revisited.
        """
        import re
        conditional_patterns = [
            r'if.*demand.*growth',
            r'if.*disruption',
            r'provided.*confirm',
            r'assuming.*growth',
            r'should.*reassess',
            r'reevaluat',
            r'condition',
            r'only if',
            r'subject to',
            r'contingent',
        ]
        matches = sum(1 for p in conditional_patterns
                      if re.search(p, text, re.IGNORECASE))
        if matches >= 2:
            return "Met"
        elif matches == 1:
            return "Partially Met"
        else:
            return "Not Met"

    def score_correctness(text: str, correct_answer_keywords: list) -> str:
        """
        Checks whether the recommendation aligns with the simulation
        ground truth by looking for expected keywords in the response.
        """
        text_lower = text.lower()
        matches = sum(1 for kw in correct_answer_keywords if kw in text_lower)
        if matches >= len(correct_answer_keywords) // 2 + 1:
            return "Met"
        elif matches > 0:
            return "Partially Met"
        else:
            return "Not Met"

    def total_score(q, t, c, corr):
        """Converts Met/Partially Met/Not Met to numeric score out of 4."""
        def s(v):
            return {"Met": 1, "Partially Met": 0.5, "Not Met": 0}.get(v, 0)
        return s(q) + s(t) + s(c) + s(corr)

    # ── Run experiments and score ─────────────────────────────────────────────
    exp1_agent, exp1_base = experiment_1()
    e1_aq = score_quantitative_grounding(exp1_agent)
    e1_at = score_threshold_identification(exp1_agent)
    e1_ac = score_conditionality(exp1_agent)
    e1_ar = score_correctness(exp1_agent, ["reorder point", "increase", "higher"])
    e1_bq = score_quantitative_grounding(exp1_base)
    e1_bt = score_threshold_identification(exp1_base)
    e1_bc = score_conditionality(exp1_base)
    e1_br = score_correctness(exp1_base, ["reorder point", "increase", "higher"])
    summary.append({
        "experiment": "Exp 1: Inventory policy",
        "domain": "Inventory replenishment",
        "agent_quantitative_grounding": e1_aq,
        "agent_threshold_identification": e1_at,
        "agent_conditionality": e1_ac,
        "agent_correctness": e1_ar,
        "baseline_quantitative_grounding": e1_bq,
        "baseline_threshold_identification": e1_bt,
        "baseline_conditionality": e1_bc,
        "baseline_correctness": e1_br,
        "agent_total_score": total_score(e1_aq, e1_at, e1_ac, e1_ar),
        "baseline_total_score": total_score(e1_bq, e1_bt, e1_bc, e1_br),
    })

    exp2_agent, exp2_base = experiment_2()
    e2_aq = score_quantitative_grounding(exp2_agent)
    e2_at = score_threshold_identification(exp2_agent)
    e2_ac = score_conditionality(exp2_agent)
    e2_ar = score_correctness(exp2_agent, ["poland", "positive", "npv"])
    e2_bq = score_quantitative_grounding(exp2_base)
    e2_bt = score_threshold_identification(exp2_base)
    e2_bc = score_conditionality(exp2_base)
    e2_br = score_correctness(exp2_base, ["poland"])
    summary.append({
        "experiment": "Exp 2: Hub location",
        "domain": "Hub location investment",
        "agent_quantitative_grounding": e2_aq,
        "agent_threshold_identification": e2_at,
        "agent_conditionality": e2_ac,
        "agent_correctness": e2_ar,
        "baseline_quantitative_grounding": e2_bq,
        "baseline_threshold_identification": e2_bt,
        "baseline_conditionality": e2_bc,
        "baseline_correctness": e2_br,
        "agent_total_score": total_score(e2_aq, e2_at, e2_ac, e2_ar),
        "baseline_total_score": total_score(e2_bq, e2_bt, e2_bc, e2_br),
    })

    exp3_agent, exp3_base = experiment_3()
    e3_aq = score_quantitative_grounding(exp3_agent)
    e3_at = score_threshold_identification(exp3_agent)
    e3_ac = score_conditionality(exp3_agent)
    e3_ar = score_correctness(exp3_agent, ["dual sourcing", "dual_sourcing"])
    e3_bq = score_quantitative_grounding(exp3_base)
    e3_bt = score_threshold_identification(exp3_base)
    e3_bc = score_conditionality(exp3_base)
    e3_br = score_correctness(exp3_base, ["dual sourcing"])
    summary.append({
        "experiment": "Exp 3: Supplier resilience",
        "domain": "Supplier disruption",
        "agent_quantitative_grounding": e3_aq,
        "agent_threshold_identification": e3_at,
        "agent_conditionality": e3_ac,
        "agent_correctness": e3_ar,
        "baseline_quantitative_grounding": e3_bq,
        "baseline_threshold_identification": e3_bt,
        "baseline_conditionality": e3_bc,
        "baseline_correctness": e3_br,
        "agent_total_score": total_score(e3_aq, e3_at, e3_ac, e3_ar),
        "baseline_total_score": total_score(e3_bq, e3_bt, e3_bc, e3_br),
    })

    exp4_agent, exp4_base = experiment_4()
    e4_aq = score_quantitative_grounding(exp4_agent)
    e4_at = score_threshold_identification(exp4_agent)
    e4_ac = score_conditionality(exp4_agent)
    e4_ar = score_correctness(exp4_agent, ["poland", "dual sourcing", "proceed"])
    e4_bq = score_quantitative_grounding(exp4_base)
    e4_bt = score_threshold_identification(exp4_base)
    e4_bc = score_conditionality(exp4_base)
    e4_br = score_correctness(exp4_base, ["poland", "dual sourcing"])
    summary.append({
        "experiment": "Exp 4: Strategic orchestrator",
        "domain": "Combined cost and risk",
        "agent_quantitative_grounding": e4_aq,
        "agent_threshold_identification": e4_at,
        "agent_conditionality": e4_ac,
        "agent_correctness": e4_ar,
        "baseline_quantitative_grounding": e4_bq,
        "baseline_threshold_identification": e4_bt,
        "baseline_conditionality": e4_bc,
        "baseline_correctness": e4_br,
        "agent_total_score": total_score(e4_aq, e4_at, e4_ac, e4_ar),
        "baseline_total_score": total_score(e4_bq, e4_bt, e4_bc, e4_br),
    })

    save_comparison_table(summary)

    print("\n" + "="*65)
    print("ALL EXPERIMENTS COMPLETE")
    print("Check results/traces/ for full reasoning traces")
    print("Check results/experiments/ for comparison summary")
    print("="*65)
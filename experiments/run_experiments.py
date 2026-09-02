"""
Formal experiment runner for the dissertation.

Compares:
    1. Tool-enabled multi-agent system
    2. Standalone LLM baseline

The experiment runner evaluates both systems using:
    - qualitative rubric scores
    - quantitative grounding
    - conditionality
    - objective decision-level metrics
    - constraint satisfaction
    - inventory policy regret where independently measurable
    - Pareto efficiency for resilience recommendations

Canonical simulation configuration:
    - 1,000 Monte Carlo trials
    - seed = 42

Important:
    The simulator outputs are treated as the reference evidence for
    objective evaluation. The LLM's own reported numbers are not treated
    as ground truth.
"""

import csv
import inspect
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Project path setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from agents.orchestrator import run_orchestrator
from agents.cost_analyst import run_cost_analyst
from agents.risk_analyst import run_risk_analyst

from simulations.inventory_sim import (
    optimise_inventory_policy,
    run_inventory_sim,
)

from simulations.hub_location_sim import run_hub_location_sim
from simulations.supplier_disruption_sim import run_supplier_disruption_sim


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "gpt-4o-mini"

BASELINE_TEMPERATURE = 0.2

SIMULATION_TRIALS = 1000
SIMULATION_SEED = 42

SERVICE_LEVEL_TARGET = 95.0

RESULTS_DIR = PROJECT_ROOT / "results"
TRACE_DIR = RESULTS_DIR / "traces"
EXPERIMENT_DIR = RESULTS_DIR / "experiments"

TRACE_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def now_timestamp() -> str:
    """Return a filesystem-safe timestamp."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_json_value(value: Any) -> Any:
    """Convert values to JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(k): safe_json_value(v) for k, v in value.items()}

    if isinstance(value, list):
        return [safe_json_value(v) for v in value]

    if isinstance(value, tuple):
        return [safe_json_value(v) for v in value]

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def save_json(path: Path, data: Dict[str, Any]) -> None:
    """Save a JSON file."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            safe_json_value(data),
            f,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Robust numerical parsing
# ---------------------------------------------------------------------------

NUMBER_PATTERN = re.compile(
    r"""
    (?<![A-Za-z])
    [-+]?
    (?:
        \d{1,3}(?:,\d{3})+
        |
        \d+(?:\.\d+)?
    )
    (?:\s*(?:million|millions|billion|billions|thousand|thousands|m|mn|bn|k))?
    (?![A-Za-z])
    """,
    re.IGNORECASE | re.VERBOSE,
)


def parse_number_token(token: str) -> Optional[float]:
    """
    Convert a textual number into a numeric value.

    Examples:
        £57,747.03       -> 57747.03
        1.37 million     -> 1370000
        £1.37M           -> 1370000
        -£0.771 million  -> -771000
        77.0%             -> 77.0

    Percentages are deliberately NOT scaled.
    """

    if not token:
        return None

    text = token.lower().strip()

    is_percentage = "%" in text

    text = text.replace("£", "")
    text = text.replace("$", "")
    text = text.replace("€", "")
    text = text.replace(",", "")

    multiplier = 1.0

    if re.search(r"\b(?:million|millions|m|mn)\b", text):
        multiplier = 1_000_000.0

    elif re.search(r"\b(?:billion|billions|b|bn)\b", text):
        multiplier = 1_000_000_000.0

    elif re.search(r"\b(?:thousand|thousands|k)\b", text):
        multiplier = 1_000.0

    number_match = re.search(
        r"[-+]?(?:\d+(?:\.\d+)?)",
        text,
    )

    if not number_match:
        return None

    try:
        value = float(number_match.group(0))
    except ValueError:
        return None

    if not is_percentage:
        value *= multiplier

    return value


def extract_numbers(text: str) -> List[float]:
    """Extract normalised numerical values from text."""
    if not text:
        return []

    values = []

    for match in NUMBER_PATTERN.finditer(text):
        value = parse_number_token(match.group(0))

        if value is not None:
            values.append(value)

    return values


def normalise_text(text: str) -> str:
    """Normalise text for robust matching."""
    return re.sub(r"\s+", " ", text.lower()).strip()


# ---------------------------------------------------------------------------
# Quantitative evaluation
# ---------------------------------------------------------------------------

def values_close(
    actual: float,
    expected: float,
    relative_tolerance: float = 0.02,
    absolute_tolerance: float = 0.01,
) -> bool:
    """
    Determine whether two numerical values are sufficiently close.

    A 2% relative tolerance is used for most values because LLM responses
    frequently round simulation outputs.
    """

    if expected == 0:
        return abs(actual) <= absolute_tolerance

    return abs(actual - expected) <= max(
        abs(expected) * relative_tolerance,
        absolute_tolerance,
    )


def percentage_close(
    actual: float,
    expected: float,
    tolerance: float = 1.0,
) -> bool:
    """Compare percentage values using percentage-point tolerance."""
    return abs(actual - expected) <= tolerance


def contains_number_near(
    text: str,
    target: float,
    tolerance: float = 0.02,
) -> bool:
    """Check whether a target numerical value appears in text."""
    for value in extract_numbers(text):
        if values_close(value, target, tolerance):
            return True

    return False


def score_quantitative_grounding(
    answer: str,
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate whether the answer contains the important reference values.

    The function deliberately uses reference values from the simulation
    rather than treating all numbers in the LLM answer as equivalent.
    """

    checks = []

    for key, expected in ground_truth.items():

        if expected is None:
            continue

        if isinstance(expected, bool):
            continue

        if isinstance(expected, (int, float)):

            # Percentages
            if (
                "pct" in key.lower()
                or "percentage" in key.lower()
                or "service_level" in key.lower()
                or "probability" in key.lower()
            ):
                matched = any(
                    percentage_close(value, float(expected), tolerance=1.0)
                    for value in extract_numbers(answer)
                )

            else:
                matched = contains_number_near(
                    answer,
                    float(expected),
                    tolerance=0.02,
                )

            checks.append(
                {
                    "metric": key,
                    "expected": expected,
                    "matched": matched,
                }
            )

    if not checks:
        return {
            "status": "Not Available",
            "matched": 0,
            "total": 0,
            "details": [],
        }

    matched_count = sum(
        1 for item in checks if item["matched"]
    )

    total_count = len(checks)

    ratio = matched_count / total_count

    if ratio >= 0.75:
        status = "Met"
    elif ratio >= 0.40:
        status = "Partially Met"
    else:
        status = "Not Met"

    return {
        "status": status,
        "matched": matched_count,
        "total": total_count,
        "details": checks,
    }


# ---------------------------------------------------------------------------
# Conditionality evaluation
# ---------------------------------------------------------------------------

def score_conditionality(answer: str) -> str:
    """
    Determine whether the answer appropriately recognises assumptions,
    uncertainty, or conditions.

    This is a qualitative secondary measure.
    """

    text = normalise_text(answer)

    strong_patterns = [
        r"\bconditional\b",
        r"\bconditionally\b",
        r"\bdepends on\b",
        r"\bsubject to\b",
        r"\bprovided that\b",
        r"\bif demand\b",
        r"\bif growth\b",
        r"\bunder the simulated assumptions\b",
        r"\bbased on the simulated assumptions\b",
        r"\bwould need to be confirmed\b",
        r"\bshould be re-?evaluated\b",
        r"\bmay change\b",
        r"\buncertainty\b",
    ]

    weak_patterns = [
        r"\bassumption\b",
        r"\brisk\b",
        r"\bscenario\b",
        r"\bvariability\b",
        r"\bsensitivity\b",
    ]

    if any(re.search(pattern, text) for pattern in strong_patterns):
        return "Met"

    if any(re.search(pattern, text) for pattern in weak_patterns):
        return "Partially Met"

    return "Not Met"


# ---------------------------------------------------------------------------
# Threshold identification
# ---------------------------------------------------------------------------

def score_threshold_identification(answer: str) -> str:
    """
    Determine whether the answer identifies a decision threshold,
    service constraint, crossover, or comparable condition.
    """

    text = normalise_text(answer)

    patterns = [
        r"\bbelow\s+\d+(?:\.\d+)?\s*%",
        r"\babove\s+\d+(?:\.\d+)?\s*%",
        r"\bless than\s+\d+(?:\.\d+)?\s*%",
        r"\bmore than\s+\d+(?:\.\d+)?\s*%",
        r"\bminimum\b",
        r"\btarget\b",
        r"\bthreshold\b",
        r"\bcrossover\b",
        r"\bbreak[- ]even\b",
        r"\bbreakeven\b",
        r"\bservice level target\b",
        r"\bprobability of profitability\b",
        r"\bprobability\b.*\bchange\b",
        r"\bfalls below\b",
        r"\brises above\b",
    ]

    if any(re.search(pattern, text) for pattern in patterns):
        return "Met"

    return "Not Met"


# ---------------------------------------------------------------------------
# Qualitative correctness
# ---------------------------------------------------------------------------

def score_correctness(
    answer: str,
    ground_truth: Dict[str, Any],
) -> float:
    """
    Secondary qualitative score.

    This is intentionally not the primary evaluation measure.

    4 = strong numerical grounding and appropriate reasoning
    3 = mostly correct with minor omissions
    2 = partially correct
    1 = weakly grounded
    0 = substantially incorrect
    """

    quantitative = score_quantitative_grounding(
        answer,
        ground_truth,
    )

    quantitative_status = quantitative["status"]

    conditionality = score_conditionality(answer)
    threshold = score_threshold_identification(answer)

    score = 0.0

    if quantitative_status == "Met":
        score += 2.0
    elif quantitative_status == "Partially Met":
        score += 1.0
    elif quantitative_status == "Not Met":
        score += 0.0

    if conditionality == "Met":
        score += 1.0
    elif conditionality == "Partially Met":
        score += 0.5

    if threshold == "Met":
        score += 1.0

    return min(score, 4.0)


def total_score(
    answer: str,
    ground_truth: Dict[str, Any],
) -> float:
    """Backward-compatible wrapper."""
    return score_correctness(answer, ground_truth)


# ---------------------------------------------------------------------------
# Recommendation extraction
# ---------------------------------------------------------------------------

def extract_last_numeric_match(
    text: str,
    patterns: List[str],
) -> Optional[float]:
    """
    Return the final numeric value matching any supplied pattern.

    Taking the final occurrence is useful for Experiment 1 because the
    answer normally discusses the current policy first and recommended
    policy second.
    """

    matches = []

    for pattern in patterns:
        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            value = parse_number_token(match.group(1))

            if value is not None:
                matches.append(
                    (
                        match.start(),
                        value,
                    )
                )

    if not matches:
        return None

    matches.sort(key=lambda item: item[0])

    return matches[-1][1]


def extract_inventory_policy(
    answer: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract the recommended reorder point and order quantity.

    The parser intentionally looks for ROP/reorder point and OQ/order
    quantity terminology rather than arbitrary numbers.
    """

    reorder_patterns = [
        r"\breorder\s+point\b\s*(?:is|of|=|:|around|approximately)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"\bROP\b\s*(?:is|of|=|:|around|approximately)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    ]

    order_patterns = [
        r"\border\s+quantity\b\s*(?:is|of|=|:|around|approximately)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
        r"\bOQ\b\s*(?:is|of|=|:|around|approximately)?\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    ]

    rop = extract_last_numeric_match(
        answer,
        reorder_patterns,
    )

    oq = extract_last_numeric_match(
        answer,
        order_patterns,
    )

    return rop, oq


def extract_recommended_location(answer: str) -> Optional[str]:
    """
    Extract a hub location recommendation.

    The function first searches for explicit recommendation language.
    """

    text = normalise_text(answer)

    explicit_patterns = [
        r"\brecommend(?:ation)?\b.{0,100}\b(poland|germany)\b",
        r"\bchoose\b.{0,50}\b(poland|germany)\b",
        r"\bselect\b.{0,50}\b(poland|germany)\b",
        r"\bbetter investment\b.{0,100}\b(poland|germany)\b",
        r"\bproceed\b.{0,100}\b(poland|germany)\b",
        r"\binvest\b.{0,100}\b(poland|germany)\b",
    ]

    for pattern in explicit_patterns:
        matches = list(re.finditer(pattern, text))

        if matches:
            location = matches[-1].group(1).lower()
            return location.capitalize()

    # Reverse-order patterns
    reverse_patterns = [
        r"\b(poland|germany)\b.{0,80}\bbetter investment\b",
        r"\b(poland|germany)\b.{0,80}\brecommend",
        r"\b(poland|germany)\b.{0,80}\bshould\b",
    ]

    for pattern in reverse_patterns:
        matches = list(re.finditer(pattern, text))

        if matches:
            location = matches[-1].group(1).lower()
            return location.capitalize()

    return None


def extract_hub_decision(answer: str) -> Optional[str]:
    """
    Extract the investment decision.

    Returns:
        "proceed"
        "do_not_invest"
        None
    """

    text = normalise_text(answer)

    negative_patterns = [
        r"\bdo not invest\b",
        r"\bdon't invest\b",
        r"\bshould not proceed\b",
        r"\bdo not proceed\b",
        r"\bnot invest\b",
        r"\bnot proceed\b",
        r"\breject the investment\b",
    ]

    if any(re.search(pattern, text) for pattern in negative_patterns):
        return "do_not_invest"

    positive_patterns = [
        r"\bproceed with\b",
        r"\bproceed\b.*\binvest",
        r"\brecommend.*\binvest",
        r"\brecommend.*\bproceed",
        r"\bshould invest\b",
        r"\binvestment is viable\b",
        r"\binvestment is financially viable\b",
    ]

    if any(re.search(pattern, text) for pattern in positive_patterns):
        return "proceed"

    return None


def extract_resilience_strategy(answer: str) -> Optional[str]:
    """
    Extract the recommended resilience strategy.

    Preference is given to phrases close to recommendation language.
    """

    text = normalise_text(answer)

    strategies = [
        "dual_sourcing",
        "safety_stock",
        "air_freight",
        "no_backup",
    ]

    labels = {
        "dual_sourcing": [
            "dual sourcing",
            "dual-source",
            "dual sourcing strategy",
        ],
        "safety_stock": [
            "safety stock",
            "safety stock buffer",
        ],
        "air_freight": [
            "air freight",
            "air-freight",
        ],
        "no_backup": [
            "no backup",
            "no-backup",
        ],
    }

    recommendation_words = [
        "recommend",
        "recommended",
        "recommendation",
        "best",
        "choose",
        "select",
        "adopt",
        "implement",
    ]

    candidates = []

    for strategy in strategies:
        for label in labels[strategy]:
            for match in re.finditer(
                re.escape(label),
                text,
            ):
                start = max(0, match.start() - 120)
                end = min(len(text), match.end() + 120)

                context = text[start:end]

                score = sum(
                    1
                    for word in recommendation_words
                    if word in context
                )

                candidates.append(
                    (
                        score,
                        match.start(),
                        strategy,
                    )
                )

    if candidates:
        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        best_score = candidates[-1][0]

        best = [
            item
            for item in candidates
            if item[0] == best_score
        ]

        return best[-1][2]

    return None


# ---------------------------------------------------------------------------
# Inventory policy independent evaluation
# ---------------------------------------------------------------------------

def evaluate_inventory_policy(
    reorder_point: float,
    order_quantity: float,
) -> Optional[Dict[str, Any]]:
    """
    Independently simulate an LLM-recommended inventory policy.

    inspect.signature() is used so this evaluator remains compatible with
    the current simulator function signature without guessing unsupported
    arguments.
    """

    try:
        signature = inspect.signature(run_inventory_sim)
    except (TypeError, ValueError):
        return None

    available = set(signature.parameters.keys())

    candidate_kwargs = {
        "demand_mean": 200,
        "demand_std": 40,
        "lead_time_days": 7,
        "reorder_point": int(round(reorder_point)),
        "order_quantity": int(round(order_quantity)),
        "seasonal_spike": False,
        "trials": SIMULATION_TRIALS,
        "seed": SIMULATION_SEED,
    }

    kwargs = {
        key: value
        for key, value in candidate_kwargs.items()
        if key in available
    }

    try:
        result = run_inventory_sim(**kwargs)
    except Exception as exc:
        print(
            f"WARNING: Could not independently simulate inventory "
            f"policy ROP={reorder_point}, OQ={order_quantity}: {exc}"
        )
        return None

    if not isinstance(result, dict):
        return None

    return result


def calculate_inventory_objective_metrics(
    answer: str,
    reference: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate objective inventory metrics.

    Metrics:
        - policy extraction
        - exact policy match
        - independently simulated service level
        - service constraint violation
        - independently simulated cost
        - cost regret against reference policy
    """

    rop, oq = extract_inventory_policy(answer)

    metrics = {
        "recommended_rop": rop,
        "recommended_order_quantity": oq,
        "policy_extracted": rop is not None and oq is not None,
        "policy_exact_match": None,
        "independent_service_level_pct": None,
        "constraint_violation_pct_points": None,
        "constraint_satisfied": None,
        "independent_total_cost_gbp": None,
        "regret_gbp": None,
        "regret_pct": None,
    }

    if rop is None or oq is None:
        return metrics

    reference_rop = reference.get("reorder_point")
    reference_oq = reference.get("order_quantity")

    if reference_rop is not None and reference_oq is not None:
        metrics["policy_exact_match"] = (
            int(round(rop)) == int(round(reference_rop))
            and int(round(oq)) == int(round(reference_oq))
        )

    simulated = evaluate_inventory_policy(
        rop,
        oq,
    )

    if simulated is None:
        return metrics

    service = simulated.get("avg_service_level_pct")

    if service is None:
        service = simulated.get("service_level")

    if service is not None:
        try:
            service = float(service)
            metrics["independent_service_level_pct"] = service

            if service < SERVICE_LEVEL_TARGET:
                metrics["constraint_satisfied"] = False
                metrics["constraint_violation_pct_points"] = (
                    SERVICE_LEVEL_TARGET - service
                )
            else:
                metrics["constraint_satisfied"] = True
                metrics["constraint_violation_pct_points"] = 0.0
        except (TypeError, ValueError):
            pass

    total_cost = simulated.get("avg_total_cost_gbp")

    if total_cost is None:
        total_cost = simulated.get("total_cost")

    if total_cost is not None:
        try:
            total_cost = float(total_cost)

            metrics["independent_total_cost_gbp"] = total_cost

            reference_cost = reference.get("total_cost_gbp")

            if reference_cost is not None:
                reference_cost = float(reference_cost)

                regret = total_cost - reference_cost

                metrics["regret_gbp"] = regret

                if reference_cost != 0:
                    metrics["regret_pct"] = (
                        regret / reference_cost
                    ) * 100.0

        except (TypeError, ValueError):
            pass

    return metrics


# ---------------------------------------------------------------------------
# Hub-location objective evaluation
# ---------------------------------------------------------------------------

def calculate_hub_objective_metrics(
    answer: str,
    reference: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate hub-location recommendation against independent simulation.

    The expected decision is based on the simulation's explicit
    recommendation rather than merely selecting the highest NPV.
    """

    location = extract_recommended_location(answer)
    decision = extract_hub_decision(answer)

    expected_location = reference.get("expected_location")
    expected_decision = reference.get("expected_decision")

    location_correct = None
    decision_correct = None

    if location is not None and expected_location is not None:
        location_correct = (
            location.lower()
            == str(expected_location).lower()
        )

    if decision is not None and expected_decision is not None:
        decision_correct = (
            decision
            == expected_decision
        )

    return {
        "recommended_location": location,
        "investment_decision": decision,
        "expected_location": expected_location,
        "expected_decision": expected_decision,
        "location_correct": location_correct,
        "decision_correct": decision_correct,
    }


# ---------------------------------------------------------------------------
# Pareto analysis for resilience strategies
# ---------------------------------------------------------------------------

def pareto_efficient_strategies(
    results: Dict[str, Dict[str, float]],
) -> List[str]:
    """
    Identify strategies on the cost-service Pareto frontier.

    Lower cost is better.
    Higher service level is better.

    A strategy is dominated if another strategy has:
        - equal or lower cost
        - equal or higher service
        - and at least one strict improvement.
    """

    strategies = list(results.keys())

    efficient = []

    for candidate in strategies:

        candidate_cost = results[candidate]["total_cost_gbp"]
        candidate_service = results[candidate]["service_level_pct"]

        dominated = False

        for other in strategies:

            if other == candidate:
                continue

            other_cost = results[other]["total_cost_gbp"]
            other_service = results[other]["service_level_pct"]

            no_worse_cost = other_cost <= candidate_cost
            no_worse_service = other_service >= candidate_service

            strictly_better = (
                other_cost < candidate_cost
                or other_service > candidate_service
            )

            if (
                no_worse_cost
                and no_worse_service
                and strictly_better
            ):
                dominated = True
                break

        if not dominated:
            efficient.append(candidate)

    return efficient


def calculate_resilience_objective_metrics(
    answer: str,
    reference: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate a resilience recommendation using Pareto efficiency.

    This avoids falsely declaring one unique 'correct' strategy when the
    question asks for a cost-service trade-off and multiple strategies
    lie on the cost-service frontier.
    """

    strategy = extract_resilience_strategy(answer)

    results = reference.get("strategies", {})

    pareto_set = pareto_efficient_strategies(results)

    pareto_efficient = None

    if strategy is not None:
        pareto_efficient = strategy in pareto_set

    selected_result = None

    if strategy in results:
        selected_result = results[strategy]

    return {
        "recommended_strategy": strategy,
        "pareto_efficient": pareto_efficient,
        "pareto_frontier": pareto_set,
        "selected_reference_total_cost_gbp": (
            selected_result["total_cost_gbp"]
            if selected_result
            else None
        ),
        "selected_reference_service_level_pct": (
            selected_result["service_level_pct"]
            if selected_result
            else None
        ),
    }


# ---------------------------------------------------------------------------
# Full strategic objective evaluation
# ---------------------------------------------------------------------------

def calculate_strategic_objective_metrics(
    answer: str,
    reference: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate the integrated Experiment 4 recommendation.
    """

    hub_metrics = calculate_hub_objective_metrics(
        answer,
        reference["hub"],
    )

    resilience_metrics = calculate_resilience_objective_metrics(
        answer,
        reference["resilience"],
    )

    return {
        "hub": hub_metrics,
        "resilience": resilience_metrics,
    }


# ---------------------------------------------------------------------------
# Generic objective evaluation dispatcher
# ---------------------------------------------------------------------------

def evaluate_objective_metrics(
    experiment_id: int,
    answer: str,
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:

    if experiment_id == 1:
        return calculate_inventory_objective_metrics(
            answer,
            ground_truth,
        )

    if experiment_id == 2:
        return calculate_hub_objective_metrics(
            answer,
            ground_truth,
        )

    if experiment_id == 3:
        return calculate_resilience_objective_metrics(
            answer,
            ground_truth,
        )

    if experiment_id == 4:
        return calculate_strategic_objective_metrics(
            answer,
            ground_truth,
        )

    return {}


# ---------------------------------------------------------------------------
# Pair evaluation
# ---------------------------------------------------------------------------

def evaluate_pair(
    experiment_id: int,
    agent_answer: str,
    baseline_answer: str,
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:

    agent_quantitative = score_quantitative_grounding(
        agent_answer,
        ground_truth,
    )

    baseline_quantitative = score_quantitative_grounding(
        baseline_answer,
        ground_truth,
    )

    agent_conditionality = score_conditionality(
        agent_answer,
    )

    baseline_conditionality = score_conditionality(
        baseline_answer,
    )

    agent_threshold = score_threshold_identification(
        agent_answer,
    )

    baseline_threshold = score_threshold_identification(
        baseline_answer,
    )

    agent_score = total_score(
        agent_answer,
        ground_truth,
    )

    baseline_score = total_score(
        baseline_answer,
        ground_truth,
    )

    agent_objective = evaluate_objective_metrics(
        experiment_id,
        agent_answer,
        ground_truth,
    )

    baseline_objective = evaluate_objective_metrics(
        experiment_id,
        baseline_answer,
        ground_truth,
    )

    return {
        "agent_qualitative_score": agent_score,
        "baseline_qualitative_score": baseline_score,

        "agent_quantitative_grounding": agent_quantitative,
        "baseline_quantitative_grounding": baseline_quantitative,

        "agent_conditionality": agent_conditionality,
        "baseline_conditionality": baseline_conditionality,

        "agent_threshold_identification": agent_threshold,
        "baseline_threshold_identification": baseline_threshold,

        "agent_objective_metrics": agent_objective,
        "baseline_objective_metrics": baseline_objective,
    }


# ---------------------------------------------------------------------------
# Experiment 1 reference
# ---------------------------------------------------------------------------

def build_inventory_reference() -> Dict[str, Any]:
    """
    Independently run the canonical inventory optimisation.

    The current inventory optimiser returns the evaluated policy records
    under the 'results' key rather than necessarily exposing a top-level
    'best_policy'. This function therefore derives the reference policy
    directly from the returned simulation results.

    Selection rule:
        1. Keep policies satisfying the 95% service-level constraint.
        2. Select the lowest-cost feasible policy.
        3. If costs tie, prefer the higher-service policy.

    This is the same decision rule used by the inventory optimiser and
    avoids depending on a particular return-dictionary key.
    """

    result = optimise_inventory_policy(
        demand_mean=200,
        demand_std=40,
        lead_time_days=7,
        target_service_level=SERVICE_LEVEL_TARGET,
        seasonal_spike=False,
        trials=SIMULATION_TRIALS,
        seed=SIMULATION_SEED,
    )

    if not isinstance(result, dict):
        raise RuntimeError(
            "Inventory optimiser returned an unexpected result type: "
            f"{type(result).__name__}"
        )

    # ---------------------------------------------------------------
    # First preference: explicit best-policy keys, if available.
    # ---------------------------------------------------------------
    best = None

    for key in (
        "best_policy",
        "selected_policy",
        "recommended_policy",
        "best_feasible_policy",
    ):
        candidate = result.get(key)

        if isinstance(candidate, dict):
            best = candidate
            break

    # ---------------------------------------------------------------
    # Current optimiser format: a list of policy results under
    # result["results"].
    # ---------------------------------------------------------------
    if best is None:
        policy_results = result.get("results")

        if isinstance(policy_results, list):
            policies = [
                policy
                for policy in policy_results
                if isinstance(policy, dict)
            ]

            feasible = [
                policy
                for policy in policies
                if bool(policy.get("feasible"))
                and policy.get("avg_service_level_pct") is not None
                and policy.get("avg_total_cost_gbp") is not None
            ]

            if feasible:
                best = min(
                    feasible,
                    key=lambda policy: (
                        float(policy["avg_total_cost_gbp"]),
                        -float(policy["avg_service_level_pct"]),
                    ),
                )

            elif policies:
                # Defensive fallback. This should not normally be reached
                # for the canonical experiment because feasible policies
                # exist. If no policy satisfies the service constraint,
                # select the highest-service policy, breaking ties using
                # lower total cost.
                best = min(
                    policies,
                    key=lambda policy: (
                        -float(
                            policy.get(
                                "avg_service_level_pct",
                                float("-inf"),
                            )
                        ),
                        float(
                            policy.get(
                                "avg_total_cost_gbp",
                                float("inf"),
                            )
                        ),
                    ),
                )

    # ---------------------------------------------------------------
    # Defensive support for a nested results dictionary.
    # ---------------------------------------------------------------
    if best is None:
        nested_results = result.get("results")

        if isinstance(nested_results, dict):
            for key in (
                "policies",
                "policy_results",
                "evaluated_policies",
            ):
                candidate_list = nested_results.get(key)

                if isinstance(candidate_list, list):
                    policies = [
                        policy
                        for policy in candidate_list
                        if isinstance(policy, dict)
                    ]

                    feasible = [
                        policy
                        for policy in policies
                        if bool(policy.get("feasible"))
                        and policy.get("avg_service_level_pct") is not None
                        and policy.get("avg_total_cost_gbp") is not None
                    ]

                    if feasible:
                        best = min(
                            feasible,
                            key=lambda policy: (
                                float(policy["avg_total_cost_gbp"]),
                                -float(
                                    policy["avg_service_level_pct"]
                                ),
                            ),
                        )
                    elif policies:
                        best = min(
                            policies,
                            key=lambda policy: (
                                -float(
                                    policy.get(
                                        "avg_service_level_pct",
                                        float("-inf"),
                                    )
                                ),
                                float(
                                    policy.get(
                                        "avg_total_cost_gbp",
                                        float("inf"),
                                    )
                                ),
                            ),
                        )

                    if best is not None:
                        break

    if best is None:
        available_keys = list(result.keys())

        raise RuntimeError(
            "Could not identify the best inventory policy from the "
            f"optimiser output. Available top-level keys: "
            f"{available_keys}"
        )

    # ---------------------------------------------------------------
    # Build the compact reference record used by the evaluator.
    # ---------------------------------------------------------------
    return {
        "reorder_point": best.get("reorder_point"),
        "order_quantity": best.get("order_quantity"),
        "service_level_pct": best.get(
            "avg_service_level_pct"
        ),
        "fill_rate_pct": best.get(
            "avg_fill_rate_pct"
        ),
        "total_cost_gbp": best.get(
            "avg_total_cost_gbp"
        ),
        "stockout_days": best.get(
            "avg_stockout_days"
        ),
        "feasible": best.get("feasible"),
    }


# ---------------------------------------------------------------------------
# Experiment 2 reference
# ---------------------------------------------------------------------------

def run_hub_reference(
    candidate_location: str,
    build_cost: float,
    annual_ops: float,
    freight_saving_pct: float,
    demand_growth_rate: float,
    discount_rate: float,
    years: int,
) -> Dict[str, Any]:

    result = run_hub_location_sim(
        candidate_location=candidate_location,
        build_cost_millions=build_cost,
        annual_ops_cost_millions=annual_ops,
        current_freight_cost_millions=12,
        freight_saving_pct=freight_saving_pct,
        demand_growth_rate=demand_growth_rate,
        discount_rate=discount_rate,
        years=years,
    )

    return result


# ---------------------------------------------------------------------------
# Experiment 3 reference
# ---------------------------------------------------------------------------

def run_resilience_reference(
    disruption_probability: float,
    disruption_duration_days: int,
    daily_demand: int,
    unit_cost: float,
    shortage_cost: float,
) -> Dict[str, Any]:

    strategies = [
        "no_backup",
        "safety_stock",
        "dual_sourcing",
        "air_freight",
    ]

    results = {}

    for strategy in strategies:

        result = run_supplier_disruption_sim(
            strategy=strategy,
            disruption_probability=disruption_probability,
            disruption_duration_days=disruption_duration_days,
            daily_demand=daily_demand,
            unit_cost=unit_cost,
            shortage_cost_per_unit=shortage_cost,
            safety_stock_weeks=4,
            dual_sourcing_premium=0.15,
            air_freight_premium=2.5,
            trials=SIMULATION_TRIALS,
            seed=SIMULATION_SEED,
        )

        results[strategy] = {
            "total_cost_gbp": float(
                result.get("avg_total_cost_gbp", 0)
            ),
            "service_level_pct": float(
                result.get("avg_service_level_pct", 0)
            ),
        }

    return {
        "strategies": results,
    }


# ---------------------------------------------------------------------------
# Experiment 4 reference
# ---------------------------------------------------------------------------

def build_strategic_reference() -> Dict[str, Any]:

    hub = run_hub_reference(
        candidate_location="Poland",
        build_cost=7,
        annual_ops=1.6,
        freight_saving_pct=0.13,
        demand_growth_rate=0.12,
        discount_rate=0.08,
        years=10,
    )

    resilience = run_resilience_reference(
        disruption_probability=0.20,
        disruption_duration_days=42,
        daily_demand=200,
        unit_cost=12,
        shortage_cost=8,
    )

    return {
        "hub": {
            **hub,
            "expected_location": "Poland",
            "expected_decision": "proceed",
        },
        "resilience": resilience,
    }


# ---------------------------------------------------------------------------
# Standalone LLM baseline
# ---------------------------------------------------------------------------

def run_standalone_llm(
    question: str,
) -> str:
    """
    Run the standalone LLM baseline.

    No simulation tools, specialist agents, or optimisation tools are
    provided to the baseline.
    """

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "OpenAI package is not installed in the active virtual "
            "environment."
        ) from exc

    client = OpenAI()

    system_prompt = """
You are a standalone large language model baseline for a research
experiment on supply-chain decision making.

Answer the user's question directly using analytical reasoning.

Important:
- You do NOT have access to simulation tools.
- You do NOT have access to optimisation tools.
- You do NOT have access to specialist agents.
- Do not claim to have run a simulation.
- Clearly distinguish assumptions from calculated results.
- Do not invent empirical simulation outputs.
- If a numerical assumption is required, state it explicitly.
- Give a practical recommendation.
- Recognise uncertainty where appropriate.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=BASELINE_TEMPERATURE,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Experiment 1
# ---------------------------------------------------------------------------

def experiment_1() -> Dict[str, Any]:

    print("\n")
    print("=" * 65)
    print("EXPERIMENT 1: INVENTORY POLICY OPTIMISATION")
    print("=" * 65)

    question = """
We currently reorder 1000 units whenever inventory falls to 300 units.
Average daily demand is 200 units with standard deviation 40 units.
Supplier lead time is 7 days. Ordering costs are £100 per order,
holding cost is £0.50 per unit per day, and shortage cost is £5 per
unit short. The target service level is 95%.

Evaluate the current policy and recommend a better reorder point and
order quantity if appropriate. Explain the trade-off between cost and
service level.
""".strip()

    print("\nRunning tool-enabled Cost Analyst...")

    agent_answer = run_cost_analyst(
        question
    )

    print("\n")
    print("-" * 65)
    print("AGENT ANSWER")
    print("-" * 65)
    print(agent_answer)

    print("\nRunning standalone LLM baseline...")

    baseline_answer = run_standalone_llm(
        question
    )

    print("\nBASELINE ANSWER:")
    print(baseline_answer)

    print("\nRunning independent reference optimisation...")

    reference = build_inventory_reference()

    print(
        "\nReference policy:"
        f" ROP={reference['reorder_point']},"
        f" OQ={reference['order_quantity']},"
        f" Service={reference['service_level_pct']:.2f}%,"
        f" Cost=£{reference['total_cost_gbp']:,.2f}"
    )

    evaluation = evaluate_pair(
        1,
        agent_answer,
        baseline_answer,
        reference,
    )

    return {
        "experiment_id": 1,
        "experiment_name": "Inventory policy",
        "question": question,
        "agent_answer": agent_answer,
        "baseline_answer": baseline_answer,
        "ground_truth": reference,
        "evaluation": evaluation,
    }


# ---------------------------------------------------------------------------
# Experiment 2
# ---------------------------------------------------------------------------

def experiment_2() -> Dict[str, Any]:

    print("\n")
    print("=" * 65)
    print("EXPERIMENT 2: DISTRIBUTION HUB LOCATION")
    print("=" * 65)

    question = """
We are a UK retailer considering opening a European distribution hub.
We have two options: Germany (£12M to build, £2.5M annual operating
cost, saves 18% of our £12M freight spend) or Poland (£7M to build,
£1.6M annual operating cost, saves 13% of our £12M freight spend).
We expect 12% annual demand growth. Our cost of capital is 8%.
Evaluate both options over a 10-year horizon. Which location is the
better investment and why?
""".strip()

    print("\nRunning tool-enabled Cost Analyst...")

    agent_answer = run_cost_analyst(
        question
    )

    print("\n")
    print("-" * 65)
    print("AGENT ANSWER")
    print("-" * 65)
    print(agent_answer)

    print("\nRunning standalone LLM baseline...")

    baseline_answer = run_standalone_llm(
        question
    )

    print("\nBASELINE ANSWER:")
    print(baseline_answer)

    print("\nRunning independent hub simulations...")

    germany = run_hub_reference(
        candidate_location="Germany",
        build_cost=12,
        annual_ops=2.5,
        freight_saving_pct=0.18,
        demand_growth_rate=0.12,
        discount_rate=0.08,
        years=10,
    )

    poland = run_hub_reference(
        candidate_location="Poland",
        build_cost=7,
        annual_ops=1.6,
        freight_saving_pct=0.13,
        demand_growth_rate=0.12,
        discount_rate=0.08,
        years=10,
    )

    # The reference location is the candidate whose simulator recommendation
    # supports investment under the stated thresholds. Where one is
    # conditional and the other is "do not invest", the conditional
    # investment option is the relevant location.
    if (
        str(polander := poland.get("recommendation", "")).upper()
        .find("INVEST") >= 0
        and str(polander).upper().find("DO NOT") < 0
    ):
        expected_location = "Poland"
        expected_decision = "proceed"
    elif (
        str(germany.get("recommendation", "")).upper()
        .find("INVEST") >= 0
        and str(germany.get("recommendation", "")).upper()
        .find("DO NOT") < 0
    ):
        expected_location = "Germany"
        expected_decision = "proceed"
    else:
        # If neither passes the investment threshold, use the higher-NPV
        # location as the comparative location while preserving the fact
        # that the absolute investment decision may be negative.
        if (
            float(poland.get("avg_npv_millions", float("-inf")))
            >= float(germany.get("avg_npv_millions", float("-inf")))
        ):
            expected_location = "Poland"
        else:
            expected_location = "Germany"

        expected_decision = "do_not_invest"

    reference = {
        "Germany": germany,
        "Poland": poland,
        "expected_location": expected_location,
        "expected_decision": expected_decision,
        "avg_npv_millions": {
            "Germany": germany.get("avg_npv_millions"),
            "Poland": poland.get("avg_npv_millions"),
        },
    }

    evaluation_reference = {
        "expected_location": expected_location,
        "expected_decision": expected_decision,
    }

    # Add the complete simulation outputs to the reference record.
    evaluation_reference["Germany"] = germany
    evaluation_reference["Poland"] = poland

    evaluation = evaluate_pair(
        2,
        agent_answer,
        baseline_answer,
        evaluation_reference,
    )

    return {
        "experiment_id": 2,
        "experiment_name": "Hub location",
        "question": question,
        "agent_answer": agent_answer,
        "baseline_answer": baseline_answer,
        "ground_truth": reference,
        "evaluation": evaluation,
    }


# ---------------------------------------------------------------------------
# Experiment 3
# ---------------------------------------------------------------------------

def experiment_3() -> Dict[str, Any]:

    print("\n")
    print("=" * 65)
    print("EXPERIMENT 3: SUPPLIER RESILIENCE STRATEGY")
    print("=" * 65)

    question = """
Our supply chain faces a 25% annual probability of supplier disruption,
with disruptions typically lasting 6 weeks. We sell 200 units per day
at £12 per unit. Unmet demand costs us £8 per unit short. We are
evaluating no backup, a safety stock buffer of 4 weeks, dual sourcing
at a 15% price premium, and air freight contingency.

Which strategy provides the best cost-service trade-off?
""".strip()

    print("\nRunning tool-enabled Risk Analyst...")

    agent_answer = run_risk_analyst(
        question
    )

    print("\n")
    print("-" * 65)
    print("AGENT ANSWER")
    print("-" * 65)
    print(agent_answer)

    print("\nRunning standalone LLM baseline...")

    baseline_answer = run_standalone_llm(
        question
    )

    print("\nBASELINE ANSWER:")
    print(baseline_answer)

    print("\nRunning independent resilience simulations...")

    reference = run_resilience_reference(
        disruption_probability=0.25,
        disruption_duration_days=42,
        daily_demand=200,
        unit_cost=12,
        shortage_cost=8,
    )

    evaluation = evaluate_pair(
        3,
        agent_answer,
        baseline_answer,
        reference,
    )

    return {
        "experiment_id": 3,
        "experiment_name": "Supplier resilience",
        "question": question,
        "agent_answer": agent_answer,
        "baseline_answer": baseline_answer,
        "ground_truth": reference,
        "evaluation": evaluation,
    }


# ---------------------------------------------------------------------------
# Experiment 4
# ---------------------------------------------------------------------------

def experiment_4() -> Dict[str, Any]:

    print("\n")
    print("=" * 65)
    print("EXPERIMENT 4: FULL MULTI-AGENT ORCHESTRATOR")
    print("=" * 65)

    question = """
We are considering opening a distribution hub in Poland (£7M build
cost, £1.6M annual operating cost, 13% freight saving on £12M annual
freight spend, 12% annual demand growth). Use an 8% cost of capital and
a 10-year evaluation horizon. Our supplier in the region has a 20%
annual disruption probability with disruptions lasting around 6 weeks.

Should we proceed with the hub, and what resilience strategy should we
implement alongside it? Give a complete strategic recommendation.
""".strip()

    print("\nRunning full multi-agent system...")

    agent_answer = run_orchestrator(
        question
    )

    print("\n")
    print("-" * 65)
    print("ORCHESTRATOR ANSWER")
    print("-" * 65)
    print(agent_answer)

    print("\nRunning standalone LLM baseline...")

    baseline_answer = run_standalone_llm(
        question
    )

    print("\nBASELINE ANSWER:")
    print(baseline_answer)

    print("\nRunning independent integrated reference simulations...")

    reference = build_strategic_reference()

    evaluation = evaluate_pair(
        4,
        agent_answer,
        baseline_answer,
        reference,
    )

    return {
        "experiment_id": 4,
        "experiment_name": "Strategic orchestrator",
        "question": question,
        "agent_answer": agent_answer,
        "baseline_answer": baseline_answer,
        "ground_truth": reference,
        "evaluation": evaluation,
    }


# ---------------------------------------------------------------------------
# Trace saving
# ---------------------------------------------------------------------------

def save_experiment_trace(
    experiment: Dict[str, Any],
) -> Path:
    """Save the full experiment record."""

    experiment_id = experiment["experiment_id"]
    experiment_name = experiment["experiment_name"]

    safe_name = re.sub(
        r"[^a-z0-9]+",
        "_",
        experiment_name.lower(),
    ).strip("_")

    filename = (
        f"exp{experiment_id}_"
        f"{safe_name}_"
        f"{now_timestamp()}.json"
    )

    path = TRACE_DIR / filename

    record = {
        **experiment,
        "metadata": {
            "model": MODEL_NAME,
            "baseline_temperature": BASELINE_TEMPERATURE,
            "simulation_trials": SIMULATION_TRIALS,
            "simulation_seed": SIMULATION_SEED,
            "service_level_target": SERVICE_LEVEL_TARGET,
            "timestamp": datetime.now().isoformat(),
        },
    }

    save_json(
        path,
        record,
    )

    print(f"\nTrace saved to: {path}")

    return path


# ---------------------------------------------------------------------------
# Objective metric flattening
# ---------------------------------------------------------------------------

def flatten_objective_metrics(
    experiment_id: int,
    metrics: Dict[str, Any],
    prefix: str,
) -> Dict[str, Any]:
    """
    Flatten nested objective metrics for CSV output.
    """

    row = {}

    if experiment_id == 1:

        row[f"{prefix}_recommended_rop"] = metrics.get(
            "recommended_rop"
        )

        row[f"{prefix}_recommended_order_quantity"] = metrics.get(
            "recommended_order_quantity"
        )

        row[f"{prefix}_policy_extracted"] = metrics.get(
            "policy_extracted"
        )

        row[f"{prefix}_policy_exact_match"] = metrics.get(
            "policy_exact_match"
        )

        row[f"{prefix}_independent_service_level_pct"] = metrics.get(
            "independent_service_level_pct"
        )

        row[f"{prefix}_constraint_satisfied"] = metrics.get(
            "constraint_satisfied"
        )

        row[f"{prefix}_constraint_violation_pct_points"] = metrics.get(
            "constraint_violation_pct_points"
        )

        row[f"{prefix}_independent_total_cost_gbp"] = metrics.get(
            "independent_total_cost_gbp"
        )

        row[f"{prefix}_regret_gbp"] = metrics.get(
            "regret_gbp"
        )

        row[f"{prefix}_regret_pct"] = metrics.get(
            "regret_pct"
        )

    elif experiment_id == 2:

        row[f"{prefix}_recommended_location"] = metrics.get(
            "recommended_location"
        )

        row[f"{prefix}_investment_decision"] = metrics.get(
            "investment_decision"
        )

        row[f"{prefix}_location_correct"] = metrics.get(
            "location_correct"
        )

        row[f"{prefix}_decision_correct"] = metrics.get(
            "decision_correct"
        )

    elif experiment_id == 3:

        row[f"{prefix}_recommended_strategy"] = metrics.get(
            "recommended_strategy"
        )

        row[f"{prefix}_pareto_efficient"] = metrics.get(
            "pareto_efficient"
        )

        row[f"{prefix}_pareto_frontier"] = (
            ", ".join(metrics.get("pareto_frontier", []))
            if metrics.get("pareto_frontier")
            else None
        )

        row[f"{prefix}_selected_reference_total_cost_gbp"] = (
            metrics.get(
                "selected_reference_total_cost_gbp"
            )
        )

        row[f"{prefix}_selected_reference_service_level_pct"] = (
            metrics.get(
                "selected_reference_service_level_pct"
            )
        )

    elif experiment_id == 4:

        hub = metrics.get("hub", {})
        resilience = metrics.get("resilience", {})

        row[f"{prefix}_hub_location"] = hub.get(
            "recommended_location"
        )

        row[f"{prefix}_hub_decision"] = hub.get(
            "investment_decision"
        )

        row[f"{prefix}_hub_location_correct"] = hub.get(
            "location_correct"
        )

        row[f"{prefix}_hub_decision_correct"] = hub.get(
            "decision_correct"
        )

        row[f"{prefix}_resilience_strategy"] = resilience.get(
            "recommended_strategy"
        )

        row[f"{prefix}_resilience_pareto_efficient"] = (
            resilience.get("pareto_efficient")
        )

        row[f"{prefix}_resilience_pareto_frontier"] = (
            ", ".join(
                resilience.get("pareto_frontier", [])
            )
            if resilience.get("pareto_frontier")
            else None
        )

    return row


# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------

def save_comparison_table(
    experiments: List[Dict[str, Any]],
) -> Path:

    path = EXPERIMENT_DIR / "comparison_summary.csv"

    rows = []

    for experiment in experiments:

        evaluation = experiment["evaluation"]

        row = {
            "experiment_id": experiment["experiment_id"],
            "experiment_name": experiment["experiment_name"],

            "agent_qualitative_score": evaluation[
                "agent_qualitative_score"
            ],

            "baseline_qualitative_score": evaluation[
                "baseline_qualitative_score"
            ],

            "agent_quantitative_grounding": evaluation[
                "agent_quantitative_grounding"
            ]["status"],

            "baseline_quantitative_grounding": evaluation[
                "baseline_quantitative_grounding"
            ]["status"],

            "agent_conditionality": evaluation[
                "agent_conditionality"
            ],

            "baseline_conditionality": evaluation[
                "baseline_conditionality"
            ],

            "agent_threshold_identification": evaluation[
                "agent_threshold_identification"
            ],

            "baseline_threshold_identification": evaluation[
                "baseline_threshold_identification"
            ],
        }

        row.update(
            flatten_objective_metrics(
                experiment["experiment_id"],
                evaluation["agent_objective_metrics"],
                "agent",
            )
        )

        row.update(
            flatten_objective_metrics(
                experiment["experiment_id"],
                evaluation["baseline_objective_metrics"],
                "baseline",
            )
        )

        rows.append(row)

    fieldnames = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nComparison table saved to: {path}"
    )

    return path


# ---------------------------------------------------------------------------
# Experiment summary JSON
# ---------------------------------------------------------------------------

def build_summary(
    experiments: List[Dict[str, Any]],
) -> Dict[str, Any]:

    summary = {
        "metadata": {
            "model": MODEL_NAME,
            "baseline_temperature": BASELINE_TEMPERATURE,
            "simulation_trials": SIMULATION_TRIALS,
            "simulation_seed": SIMULATION_SEED,
            "service_level_target": SERVICE_LEVEL_TARGET,
            "evaluation_framework": {
                "qualitative_score": (
                    "Secondary descriptive measure"
                ),
                "quantitative_grounding": (
                    "Checks whether important independent "
                    "simulation values are reported"
                ),
                "objective_metrics": (
                    "Primary decision-level evaluation"
                ),
                "inventory_regret": (
                    "Independent simulation of extracted "
                    "LLM-recommended inventory policy"
                ),
                "resilience_evaluation": (
                    "Pareto efficiency rather than forced "
                    "unique optimum for cost-service trade-off"
                ),
            },
        },
        "experiments": [],
    }

    for experiment in experiments:

        evaluation = experiment["evaluation"]

        item = {
            "experiment_id": experiment["experiment_id"],
            "experiment_name": experiment["experiment_name"],
            "agent_qualitative_score": evaluation[
                "agent_qualitative_score"
            ],
            "baseline_qualitative_score": evaluation[
                "baseline_qualitative_score"
            ],
            "agent_quantitative_grounding": evaluation[
                "agent_quantitative_grounding"
            ],
            "baseline_quantitative_grounding": evaluation[
                "baseline_quantitative_grounding"
            ],
            "agent_conditionality": evaluation[
                "agent_conditionality"
            ],
            "baseline_conditionality": evaluation[
                "baseline_conditionality"
            ],
            "agent_threshold_identification": evaluation[
                "agent_threshold_identification"
            ],
            "baseline_threshold_identification": evaluation[
                "baseline_threshold_identification"
            ],
            "agent_objective_metrics": evaluation[
                "agent_objective_metrics"
            ],
            "baseline_objective_metrics": evaluation[
                "baseline_objective_metrics"
            ],
        }

        summary["experiments"].append(item)

    return summary


def save_experiment_summary(
    experiments: List[Dict[str, Any]],
) -> Path:

    path = EXPERIMENT_DIR / "experiment_summary.json"

    save_json(
        path,
        build_summary(experiments),
    )

    print(
        f"Experiment summary saved to: {path}"
    )

    return path


# ---------------------------------------------------------------------------
# Console summary helpers
# ---------------------------------------------------------------------------

def get_decision_correct(
    experiment_id: int,
    metrics: Dict[str, Any],
) -> Optional[bool]:

    if experiment_id == 1:
        # Inventory has no single binary correctness criterion because a
        # feasible near-optimal policy can differ from the reference policy.
        # Exact policy match and regret are reported instead.
        return None

    if experiment_id == 2:
        return metrics.get("location_correct")

    if experiment_id == 3:
        return metrics.get("pareto_efficient")

    if experiment_id == 4:
        hub = metrics.get("hub", {})
        resilience = metrics.get("resilience", {})

        hub_correct = hub.get("location_correct")
        resilience_correct = resilience.get("pareto_efficient")

        if (
            hub_correct is None
            and resilience_correct is None
        ):
            return None

        if hub_correct is None:
            return resilience_correct

        if resilience_correct is None:
            return hub_correct

        return bool(
            hub_correct
            and resilience_correct
        )

    return None


def get_regret(
    experiment_id: int,
    metrics: Dict[str, Any],
) -> Optional[float]:

    if experiment_id == 1:
        return metrics.get("regret_pct")

    return None


def get_constraint_violation(
    experiment_id: int,
    metrics: Dict[str, Any],
) -> Optional[float]:

    if experiment_id == 1:
        return metrics.get(
            "constraint_violation_pct_points"
        )

    return None


def print_experiment_summary(
    experiments: List[Dict[str, Any]],
) -> None:

    print("\n")
    print("=" * 65)
    print("EXPERIMENT SUMMARY")
    print("=" * 65)

    for experiment in experiments:

        experiment_id = experiment["experiment_id"]
        name = experiment["experiment_name"]
        evaluation = experiment["evaluation"]

        agent_metrics = evaluation[
            "agent_objective_metrics"
        ]

        baseline_metrics = evaluation[
            "baseline_objective_metrics"
        ]

        agent_decision = get_decision_correct(
            experiment_id,
            agent_metrics,
        )

        baseline_decision = get_decision_correct(
            experiment_id,
            baseline_metrics,
        )

        agent_regret = get_regret(
            experiment_id,
            agent_metrics,
        )

        baseline_regret = get_regret(
            experiment_id,
            baseline_metrics,
        )

        agent_constraint = get_constraint_violation(
            experiment_id,
            agent_metrics,
        )

        baseline_constraint = get_constraint_violation(
            experiment_id,
            baseline_metrics,
        )

        print(
            f"\nExp {experiment_id}: {name}"
        )

        print(
            "  Agent qualitative score:      "
            f"{evaluation['agent_qualitative_score']:.1f}/4"
        )

        print(
            "  Baseline qualitative score:   "
            f"{evaluation['baseline_qualitative_score']:.1f}/4"
        )

        print(
            "  Agent decision correct:        "
            f"{agent_decision}"
        )

        print(
            "  Baseline decision correct:     "
            f"{baseline_decision}"
        )

        if agent_regret is None:
            print(
                "  Agent regret:                 None"
            )
        else:
            print(
                "  Agent regret:                 "
                f"{agent_regret:.2f}%"
            )

        if baseline_regret is None:
            print(
                "  Baseline regret:              None"
            )
        else:
            print(
                "  Baseline regret:              "
                f"{baseline_regret:.2f}%"
            )

        if agent_constraint is None:
            print(
                "  Agent constraint violation:    None"
            )
        else:
            print(
                "  Agent constraint violation:    "
                f"{agent_constraint:.2f} pp"
            )

        if baseline_constraint is None:
            print(
                "  Baseline constraint violation: "
                "None"
            )
        else:
            print(
                "  Baseline constraint violation: "
                f"{baseline_constraint:.2f} pp"
            )

        print(
            "  Quantitative grounding:        "
            f"{evaluation['agent_quantitative_grounding']['status']}"
            " vs "
            f"{evaluation['baseline_quantitative_grounding']['status']}"
        )

        print(
            "  Conditionality:               "
            f"{evaluation['agent_conditionality']}"
            " vs "
            f"{evaluation['baseline_conditionality']}"
        )

        print(
            "  Threshold identification:     "
            f"{evaluation['agent_threshold_identification']}"
            " vs "
            f"{evaluation['baseline_threshold_identification']}"
        )


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 65)
    print("SUPPLY CHAIN AI DISSERTATION EXPERIMENT RUNNER")
    print("=" * 65)

    print("\nConfiguration:")
    print(f"  Model:              {MODEL_NAME}")
    print(f"  Baseline temp:      {BASELINE_TEMPERATURE}")
    print(f"  Simulation trials:  {SIMULATION_TRIALS}")
    print(f"  Simulation seed:     {SIMULATION_SEED}")
    print(f"  Service target:      {SERVICE_LEVEL_TARGET}%")

    experiments = []

    # ------------------------------------------------------------------
    # Experiment 1
    # ------------------------------------------------------------------

    try:
        exp1 = experiment_1()
        save_experiment_trace(exp1)
        experiments.append(exp1)

    except Exception as exc:
        print(
            "\nERROR IN EXPERIMENT 1:"
            f" {type(exc).__name__}: {exc}"
        )
        raise

    # ------------------------------------------------------------------
    # Experiment 2
    # ------------------------------------------------------------------

    try:
        exp2 = experiment_2()
        save_experiment_trace(exp2)
        experiments.append(exp2)

    except Exception as exc:
        print(
            "\nERROR IN EXPERIMENT 2:"
            f" {type(exc).__name__}: {exc}"
        )
        raise

    # ------------------------------------------------------------------
    # Experiment 3
    # ------------------------------------------------------------------

    try:
        exp3 = experiment_3()
        save_experiment_trace(exp3)
        experiments.append(exp3)

    except Exception as exc:
        print(
            "\nERROR IN EXPERIMENT 3:"
            f" {type(exc).__name__}: {exc}"
        )
        raise

    # ------------------------------------------------------------------
    # Experiment 4
    # ------------------------------------------------------------------

    try:
        exp4 = experiment_4()
        save_experiment_trace(exp4)
        experiments.append(exp4)

    except Exception as exc:
        print(
            "\nERROR IN EXPERIMENT 4:"
            f" {type(exc).__name__}: {exc}"
        )
        raise

    # ------------------------------------------------------------------
    # Save aggregate outputs
    # ------------------------------------------------------------------

    save_comparison_table(
        experiments
    )

    save_experiment_summary(
        experiments
    )

    print_experiment_summary(
        experiments
    )

    print("\n")
    print("=" * 65)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 65)

    print(
        "\nOutputs:"
        "\n  results/traces/"
        "\n      Full experiment records"
        "\n  results/experiments/comparison_summary.csv"
        "\n      Structured evaluation summary"
        "\n  results/experiments/experiment_summary.json"
        "\n      Machine-readable experiment metadata"
    )


if __name__ == "__main__":
    main()
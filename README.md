# Supply Chain Multi-Agent Decision System

**MSc Artificial Intelligence Dissertation**

Loughborough University | Project GV-13

*Simulation-in-the-loop Reasoning for Supply Chain Optimisation: A Multi-Agent LLM System with Discrete-Event Simulation Tools*

---

## Overview

This project implements a multi-agent large language model (LLM) decision-support system for supply chain optimisation.

The system combines:

- Natural-language reasoning using an LLM
- Specialist AI agents for cost and risk analysis
- Python-based supply chain simulations
- Numerical optimisation baselines
- Monte Carlo experimentation
- Statistical analysis using repeated simulation replications
- An interactive demonstration interface

The central research question is:

> **Can an LLM-based multi-agent system that uses simulation tools produce more quantitatively grounded supply chain decisions than a standalone LLM?**

Rather than requiring the LLM to estimate operational quantities directly, the system allows specialist agents to call computational simulation tools. The resulting numerical evidence is then incorporated into the agents' reasoning and recommendations.

The architecture follows a tool-using, iterative reasoning approach inspired by the ReAct framework (Yao et al., 2022), implemented within a multi-agent architecture.

---

## System Architecture

```text
                         USER QUESTION
                               |
                               v
                       +---------------+
                       | ORCHESTRATOR  |
                       +---------------+
                          /           \
                         /             \
                        v               v
               +---------------+ +---------------+
               | COST ANALYST  | | RISK ANALYST  |
               +---------------+ +---------------+
                       |                 |
                       v                 v
                Cost / investment   Risk / resilience
                   simulations          simulations
                       |                 |
                       +--------+--------+
                                |
                                v
                     Evidence synthesis
                                |
                                v
                    FINAL RECOMMENDATION

--

Agents

Cost Analyst

Handles:

Inventory replenishment policy
Reorder point (ROP) and order quantity (OQ)
Distribution hub investment
Net present value (NPV)
Cost and investment trade-offs

Risk Analyst

Handles:

Supplier disruption
Safety stock
Dual sourcing
Air freight
Service-level and resilience trade-offs

Orchestrator

Coordinates the specialist agents for combined strategic questions and synthesises their findings into an overall recommendation.

Simulation Tools
Simulation 1 — Inventory Replenishment

A stochastic inventory simulation evaluates replenishment policies using:

Reorder point (ROP)
Order quantity (OQ)
Stochastic daily demand
Variable lead times
Seasonal demand spikes
Holding costs
Ordering costs
Shortage costs
Service level
Fill rate

The numerical optimiser searches the evaluated policy space and identifies the lowest-cost feasible policy within that search space, subject to the 95% service-level target.

Canonical optimisation configuration:

1,000 Monte Carlo trials
Random seed: 42
ROP search: 500–3,500 units in increments of 250
OQ candidates: 500, 750, 1,000, 1,250, 1,500, 2,000, 2,500 units
Service-level target: 95%
Simulation 2 — Distribution Hub Investment

The hub simulation evaluates distribution-centre investment decisions using stochastic financial modelling.

Parameters include:

Construction/build cost
Annual operating cost
Freight expenditure
Freight saving percentage
Annual demand growth
Discount rate
Investment horizon

The simulation produces:

Average NPV
P10 outcome
P90 outcome
Probability of a profitable investment
Break-even year

P10 and P90 are reported as simulation outcome percentiles and should not be interpreted as confidence intervals.

Simulation 3 — Supplier Disruption and Resilience

The supplier disruption simulation models disruption risk and evaluates alternative resilience strategies:

No backup
Safety stock
Dual sourcing
Air freight

The simulation reports:

Total cost
Shortage cost
Units short
Service level

The strategy comparison is evaluated under a specified disruption probability and disruption duration.

Project Structure
supply-chain-ai/
│
├── agents/
│   ├── cost_analyst.py
│   ├── risk_analyst.py
│   └── orchestrator.py
│
├── simulations/
│   ├── inventory_sim.py
│   ├── inventory_visualiser.py
│   ├── hub_location_sim.py
│   ├── hub_visualiser.py
│   ├── supplier_disruption_sim.py
│   └── disruption_visualiser.py
│
├── experiments/
│   ├── run_experiments.py
│   ├── optimiser_baseline.py
│   ├── ablation_study.py
│   └── statistical_analysis.py
│
├── results/
│   ├── traces/
│   ├── experiments/
│   ├── optimiser/
│   ├── ablation/
│   ├── statistics/
│   └── *.png / *.csv / *.json
│
├── demo.py
├── requirements.txt
├── .gitignore
└── README.md
Results

The results/ directory contains outputs generated by the experiments.

Examples include:

Agent reasoning traces
Experiment summary files
Optimiser search results
Ablation results
Statistical analysis
Simulation visualisations
CSV datasets
JSON experiment records

These saved results provide a reproducible record of the experiments without requiring an evaluator to make new LLM API calls.

Experimental Evaluation

The project evaluates the system using several complementary approaches.

1. Multi-Agent vs Standalone Baseline

The main experiment compares the simulation-enabled agent system with a standalone numerical/analytical baseline.

The comparison considers:

Decision quality
Constraint satisfaction
Quantitative grounding
Conditional reasoning
Threshold identification
Objective performance where directly measurable

The baseline is deliberately separated from the LLM system so that simulation-enabled recommendations can be compared against an independently implemented numerical search.

2. Numerical Optimisation Baseline

A standalone optimiser provides an independent reference point for the simulation problems.

Inventory

The optimiser evaluates 91 ROP/OQ combinations and selects the lowest-cost policy satisfying the 95% service-level requirement.

Final result:

ROP = 1750 units
OQ  = 1500 units

Average total cost ≈ £57,747
Average service level ≈ 96.00%

The cheapest policy overall was not necessarily feasible:

ROP = 1250 units
OQ  = 1250 units

Average total cost ≈ £42,512
Average service level ≈ 84.29%

This illustrates why the optimisation problem must be treated as a constrained cost-minimisation problem rather than simply selecting the cheapest policy.

3. Single-Agent Ablation

A single-agent ablation was implemented to investigate whether the observed behaviour depends specifically on the multi-agent architecture.

The single agent was given direct access to the relevant simulation tools and was asked to reason about the same classes of supply chain problems.

The ablation showed that a single simulation-enabled agent could also use the computational tools and reach the same broad strategic resilience conclusion as the multi-agent system.

This means the evaluation does not assume that multi-agent coordination is automatically superior.

Instead, the multi-agent architecture provides explicit role specialisation:

Cost Analyst → financial and operational analysis
Risk Analyst → disruption and resilience analysis
Orchestrator → cross-domain synthesis

The ablation therefore helps isolate the contribution of specialist coordination from the broader benefit of giving an LLM access to simulation tools.

4. Statistical Analysis

Repeated simulation replications were used to assess the stability of selected results.

The statistical analysis uses:

100 independent replications
1,000 Monte Carlo trials per inventory replication
2,000 bootstrap resamples
95% bootstrap confidence intervals

The bootstrap intervals describe uncertainty in the estimated mean across repeated simulation replications. They are not ranges containing individual simulation outcomes.

Inventory

For ROP=1750 and OQ=1500:

Mean service level: 96.05%
95% CI:             96.05% – 96.06%

Mean total cost:    £57,793
95% CI:             £57,788 – £57,798

The result is therefore highly stable across repeated simulation replications under the specified model assumptions.

Supplier Disruption

At a 25% annual disruption probability:

Strategy          Mean Cost       Mean Service
------------------------------------------------
No backup         £1.483M         96.45%
Dual sourcing     £1.457M         97.75%

A paired bootstrap comparison was also performed.

For total cost:

No backup - Dual sourcing

Mean difference:   £25,962
95% CI:             £13,170 – £39,859

Because the confidence interval excludes zero, the cost difference was statistically distinguishable from zero under the specified bootstrap procedure.

For service level:

No backup - Dual sourcing

Mean difference:   -1.30 percentage points
95% CI:             -1.85 – -0.80 percentage points

The negative value indicates that dual sourcing achieved higher service in the simulated comparison.

Key Experimental Findings
Inventory Replenishment

The independent numerical optimiser selected:

ROP = 1750
OQ  = 1500

as the lowest-cost feasible policy within the evaluated search space.

The simulation-enabled agent also recommended:

ROP = 1750
OQ  = 1500

in the formal experiment.

This provides evidence that simulation-enabled LLM reasoning can reproduce a quantitatively supported policy identified by an independent numerical search.

However, this should not be interpreted as evidence that the LLM found a global optimum. The claim is restricted to the evaluated search space.

Hub Location

The hub experiment demonstrates that investment recommendations are sensitive to assumptions such as:

Demand growth
Construction cost
Operating cost
Freight savings
Discount rate
Investment horizon

For the specified Poland scenario used in the controlled experiment, the simulation produced:

Average NPV:        ≈ £1.37M
P10:                ≈ -£0.771M
P90:                ≈ £3.677M
Probability profit: ≈ 77%
Break-even:         ≈ 7.9 years

The resulting recommendation was conditional on the demand-growth assumption.

The independent optimiser also searched across multiple hub locations and growth assumptions. Its highest-NPV scenario was a different scenario from the controlled Poland case, so those two results should not be interpreted as a direct numerical comparison.

Supplier Disruption

At a 25% annual disruption probability, the multi-agent system evaluated four resilience strategies.

The simulation results indicated that:

No backup had the lowest cost among the evaluated strategies but lower service than the alternatives.
Safety stock provided the highest service level but at substantially higher cost.
Dual sourcing provided a cost-service trade-off that was selected by the agent.
Air freight provided an alternative high-service response at a higher cost than dual sourcing under the tested assumptions.

The standalone numerical optimiser independently selected dual sourcing as the lowest-cost strategy satisfying the 95% service-level constraint under its common parameterisation.

The paired statistical analysis also found a statistically distinguishable cost difference between dual sourcing and no backup under the specified simulation assumptions.

Reproducibility

The experiments are designed to be reproducible through fixed simulation configurations, explicit parameters, random seeds, and saved experiment outputs.

The project records:

Simulation parameters
Random seeds
Monte Carlo trial counts
Search spaces
Agent prompts and responses where applicable
Reasoning traces
Numerical results
Optimiser search results
Ablation results
Statistical analysis results

LLM outputs can vary between API calls. Therefore, saved traces and experiment outputs are used as the primary record of the reported experimental runs.

Re-running the LLM experiments may produce different natural-language responses even when the underlying simulation tools and parameters remain unchanged.

Installation
Requirements
Python 3.11 or later
OpenAI API access
An OpenAI API key
Internet access for LLM API calls

The simulation and analysis components use Python packages specified in requirements.txt.

1. Clone the Repository
git clone https://github.com/JoylinDsouza/supply-chain-agents.git

cd supply-chain-agents
2. Create a Virtual Environment
Windows
python -m venv venv

venv\Scripts\activate
macOS/Linux
python3 -m venv venv

source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt

If requirements.txt is unavailable, the principal dependencies are:

pip install openai python-dotenv numpy matplotlib
4. Configure the API Key

Create a .env file in the project root:

OPENAI_API_KEY=sk-your-key-here

Replace the placeholder with your own API key.

Never commit the .env file or expose an API key in the repository.

The evaluator should use their own OpenAI API credentials if they wish to reproduce live LLM calls.

The saved experiment traces and numerical results do not require access to the original API key.

Running the Project
Interactive Demonstration

Run:

python demo.py

The demonstration accepts natural-language supply chain questions and routes them through the multi-agent system.

Example:

We are opening a hub in Poland and our supplier is unreliable.
What resilience strategy should we adopt?
Formal Experiments

Run the main experiment suite:

python experiments/run_experiments.py

This produces the main agent and baseline comparison outputs.

Numerical Optimiser Baseline

Run:

python experiments/optimiser_baseline.py

This evaluates the standalone numerical search procedures for:

Inventory policy
Hub investment scenarios
Supplier resilience strategies

Results are written to:

results/optimiser/
Ablation Study

Run:

python experiments/ablation_study.py

This compares the multi-agent system with a single simulation-enabled agent.

Results are written to:

results/ablation/
Statistical Analysis

Run:

python experiments/statistical_analysis.py

This performs repeated simulation replications and bootstrap analysis.

Results are written to:

results/statistics/
Individual Simulations
Inventory
python simulations/inventory_sim.py
Hub Location
python simulations/hub_location_sim.py
Supplier Disruption
python simulations/supplier_disruption_sim.py
Generate Visualisations
Inventory
python simulations/inventory_visualiser.py
Hub Location
python simulations/hub_visualiser.py
Supplier Disruption
python simulations/disruption_visualiser.py
Important Modelling Assumptions

The simulations are computational models rather than direct representations of a real company's supply chain.

Results depend on assumptions including:

Demand distributions
Lead-time distributions
Cost parameters
Disruption probabilities
Disruption duration
Demand growth
Freight savings
Discount rates
Strategy-specific premiums
Service-level definitions

Consequently, numerical results should be interpreted as results within the evaluated model and parameter space.

Simulation provides computational evaluation of the specified assumptions. It does not guarantee that those assumptions accurately represent a real-world supply chain.

Limitations

Several limitations should be considered when interpreting the results.

LLM variability

LLM-generated reasoning is not guaranteed to be deterministic. Low-temperature generation can improve consistency but does not guarantee identical outputs.

Simulation validity

The simulations are only as realistic as their assumptions and implementation. Correct execution of a simulation does not establish that the model is a perfect representation of a real supply chain.

Search-space limitation

The numerical optimiser identifies the lowest-cost feasible policy within the evaluated search space. It does not establish a mathematical global optimum over all possible policies.

Ablation interpretation

The single-agent ablation demonstrates that direct simulation access itself can produce quantitatively grounded recommendations. It does not establish that the multi-agent architecture is universally superior to a single simulation-enabled agent.

Resilience cost model

The supplier disruption model contains modelling assumptions concerning procurement, shortages and disruption. In particular, cost outcomes for the no-backup strategy can be affected by the treatment of unserved demand and avoided procurement costs. Results should therefore be interpreted within the implemented cost model.

Live API reproduction

LLM responses may differ between API calls. Saved traces are therefore retained as the authoritative record of the reported experimental runs.

Technologies
Python 3.11+
OpenAI API
GPT-4o-mini
NumPy
Matplotlib
python-dotenv
Academic Context

The project investigates simulation-in-the-loop reasoning using LLM-based agents.

The architecture draws on:

ReAct for iterative reasoning and tool interaction (Yao et al., 2022)
Multi-agent LLM systems (Wu et al., 2023)
Chain-of-thought reasoning (Wei et al., 2022)
Tool-augmented language models and tool-use approaches

Full references are provided in the dissertation.

Dissertation

Title:

Simulation-in-the-loop Reasoning for Supply Chain Optimisation: A Multi-Agent LLM System with Discrete-Event Simulation Tools

Programme: MSc Artificial Intelligence (COP327)

Institution: Loughborough University

Supervisor: Prof. George Vogiatzis

Student: Joylin Dsouza

Project: GV-13

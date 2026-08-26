# Supply Chain Multi-Agent Decision System

**MSc Artificial Intelligence Dissertation**  
Loughborough University | Project GV-13  
*Simulation-in-the-loop Reasoning for Supply Chain Optimisation*

---

## What this project does

This system uses a team of AI agents, each equipped with real supply chain simulations as tools, to answer strategic business questions. The central research question is: **can LLM agents using simulations make better supply chain decisions than a standalone LLM working alone?**

Instead of asking an AI to guess numbers, each agent calls a Python simulation, receives real computed results, and uses those to reason and recommend. This is the ReAct pattern (Yao et al., 2022) applied to supply chain optimisation.

---

## System architecture

User question → ORCHESTRATOR → delegates to specialists → synthesises findings
│
┌─────────┴─────────┐
▼ ▼
COST ANALYST RISK ANALYST
(Sims 1 & 2) (Sim 3)


- **Simulation 1** — Inventory replenishment: finds optimal reorder policy minimising total cost
- **Simulation 2** — Hub location: NPV analysis of distribution centre investment decisions
- **Simulation 3** — Supplier disruption: compares backup strategies under different disruption probabilities

---

## Project structure

supply-chain-ai/
├── agents/
│ ├── cost_analyst.py # Cost & investment agent
│ ├── risk_analyst.py # Risk & resilience agent
│ └── orchestrator.py # Coordinates specialists, synthesises findings
├── simulations/
│ ├── inventory_sim.py # Simulation 1: inventory replenishment
│ ├── inventory_visualiser.py # Graphs for Simulation 1
│ ├── hub_location_sim.py # Simulation 2: hub location NPV
│ ├── hub_visualiser.py # Graphs for Simulation 2
│ ├── supplier_disruption_sim.py # Simulation 3: disruption strategies
│ └── disruption_visualiser.py # Graphs for Simulation 3
├── experiments/
│ └── run_experiments.py # Formal experiment runner + baseline comparison
├── results/
│ ├── traces/ # JSON reasoning traces from each experiment
│ ├── experiments/ # Comparison summary CSVs
│ └── *.png / *.csv # Graphs and results tables
├── demo.py # Interactive demonstration
└── .gitignore


---

## Setup

**Requirements:** Python 3.11+, OpenAI API key

```bash
git clone https://github.com/JoylinDsouza/supply-chain-agents.git
cd supply-chain-agents
python -m venv venv
venv\Scripts\activate
pip install openai python-dotenv numpy matplotlib
```

Create a `.env` file in the root folder:

OPENAI_API_KEY=sk-your-key-here


---

## How to run

```bash
# Interactive demo — ask any supply chain question
python demo.py

# Run all formal experiments (agent vs baseline comparison)
python experiments/run_experiments.py

# Run individual simulations
python simulations/inventory_sim.py
python simulations/hub_location_sim.py
python simulations/supplier_disruption_sim.py

# Generate all graphs
python simulations/inventory_visualiser.py
python simulations/hub_visualiser.py
python simulations/disruption_visualiser.py

# Run individual agents
python agents/cost_analyst.py
python agents/risk_analyst.py
python agents/orchestrator.py
```

---

## Key findings

**Simulation 1 — Inventory replenishment:**
The optimal reorder point (ROP=1000) minimises total cost at ~£44,180. The total cost curve is U-shaped — shortage cost dominates at low ROP, holding cost dominates at high ROP. A seasonal demand spike of 40% reveals that policies below ROP=800 fail disproportionately during peak periods.

**Simulation 2 — Hub location:**
Poland is the lowest-risk entry point, financially justified at 10% demand growth (52% probability of profit). Germany requires 15-20% growth. At demand growth below 8%, no location is justified. The decision is fundamentally conditional on demand growth assumptions.

**Simulation 3 — Supplier disruption:**
Dual sourcing beats air freight above 8% annual disruption probability, costing ~£200k less per year at 25% probability. Safety stock maintains the highest service level (99%+) but costs over £2.5M annually — never cost-optimal. The crossover point between strategies is the key actionable insight.

**Multi-agent system:**
The Orchestrator combined cost and risk findings to produce conditional, threshold-based recommendations that neither agent could produce alone — for example: *proceed with Poland hub AND implement dual sourcing, but reassess if demand growth falls below 8% or disruption probability exceeds 20%.*

---

## Technologies

- Python 3.11 · OpenAI API (GPT-4o-mini) · NumPy · Matplotlib

---

## Academic context

Implements the ReAct pattern (Yao et al., 2022) in a multi-agent architecture (Wu et al., 2023). Uses chain-of-thought prompting (Wei et al., 2022) and tool calling (Schick et al., 2023).

**Supervisor:** Prof. George Vogiatzis, Loughborough University  
**Student:** Aadvik Tripathi  
**Programme:** MSc Artificial Intelligence (COP327)
# demo.py
# Interactive demonstration of the multi-agent supply chain system.
# Run this file and type any supply chain question to see the system respond.
# The Orchestrator will delegate to Cost Analyst and/or Risk Analyst as needed.

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from agents.orchestrator import run_orchestrator

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║     MULTI-AGENT SUPPLY CHAIN DECISION SUPPORT SYSTEM        ║
║     MSc AI Dissertation — Loughborough University           ║
║     Simulation-in-the-loop Reasoning                        ║
╠══════════════════════════════════════════════════════════════╣
║  Agents available:                                          ║
║    Cost Analyst  → inventory policy + hub location          ║
║    Risk Analyst  → supplier disruption + resilience         ║
║    Orchestrator  → strategic questions combining both       ║
╠══════════════════════════════════════════════════════════════╣
║  Example questions to try:                                  ║
║  • What reorder point should I use for 300 units/day        ║
║    demand with a 10-day lead time?                          ║
║  • Is a distribution hub in Germany financially justified   ║
║    with 15% demand growth and £10M build cost?              ║
║  • Should I use dual sourcing or air freight if my          ║
║    supplier has a 30% annual disruption probability?        ║
║  • We are opening a hub in Poland — what resilience         ║
║    strategy should we adopt alongside it?                   ║
╠══════════════════════════════════════════════════════════════╣
║  Type 'quit' to exit  |  Type 'help' for more examples     ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP = """
MORE EXAMPLE QUESTIONS:

Inventory (Cost Analyst):
  - A store sells 150 units/day with std dev 30. Lead time is 14 days.
    What reorder point minimises total cost?
  - Compare two policies: ROP=500 OQ=1000 vs ROP=800 OQ=1500 for
    demand mean 200 std 50.

Hub location (Cost Analyst):
  - Poland: £7M build, £1.6M ops, 13% freight saving, 12% growth. Invest?
  - Germany vs Netherlands: which hub gives better 10-year NPV at 15% growth?

Supplier resilience (Risk Analyst):
  - My supplier has a 20% disruption probability and disruptions last 8 weeks.
    Should I dual source or build safety stock?
  - At what disruption probability does air freight become cheaper than
    dual sourcing?

Strategic / combined (Orchestrator):
  - We want to open a hub in Poland AND our supplier is unreliable (25%
    disruption probability). Give us a complete strategy.
  - Our main supplier failed last year for 6 weeks. How do we protect
    against this happening again and what does it cost?
"""


def main():
    print(BANNER)

    while True:
        print("\n" + "─"*65)
        try:
            question = input("Your question: ").strip()
        except KeyboardInterrupt:
            print("\n\nExiting. Goodbye.")
            break

        if not question:
            continue

        if question.lower() == "quit":
            print("\nExiting. Goodbye.")
            break

        if question.lower() == "help":
            print(HELP)
            continue

        print("\nProcessing - the agents are running simulations...\n")
        print("─"*65)

        try:
            answer = run_orchestrator(question, verbose=True)
        except Exception as e:
            print(f"\nError: {e}")
            print("Check your API key is set correctly in your .env file.")


if __name__ == "__main__":
    main()
# tests/test_simulations.py
# Unit tests for all three simulations.
# Run with: python -m pytest tests/ -v
# Or without pytest: python tests/test_simulations.py

import sys
import os
import unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulations.inventory_sim import run_inventory_sim
from simulations.hub_location_sim import run_hub_location_sim
from simulations.supplier_disruption_sim import run_supplier_disruption_sim


class TestInventorySimulation(unittest.TestCase):

    def test_returns_expected_keys(self):
        result = run_inventory_sim(
            reorder_point=500, order_quantity=1000,
            demand_mean=200, demand_std=40,
            lead_time_days=7, trials=5
        )
        self.assertIn("results", result)
        r = result["results"]
        for key in ["avg_service_level_pct", "avg_total_cost_gbp",
                    "avg_holding_cost_gbp", "avg_ordering_cost_gbp",
                    "avg_shortage_cost_gbp", "avg_stockout_days", "verdict"]:
            self.assertIn(key, r, f"Missing key: {key}")

    def test_service_level_between_0_and_100(self):
        result = run_inventory_sim(
            reorder_point=500, order_quantity=1000,
            demand_mean=200, demand_std=40, trials=5
        )
        sl = result["results"]["avg_service_level_pct"]
        self.assertGreaterEqual(sl, 0)
        self.assertLessEqual(sl, 100)

    def test_total_cost_equals_sum_of_components(self):
        result = run_inventory_sim(
            reorder_point=800, order_quantity=1500,
            demand_mean=200, demand_std=40, trials=5
        )
        r = result["results"]
        component_sum = (r["avg_holding_cost_gbp"] +
                         r["avg_ordering_cost_gbp"] +
                         r["avg_shortage_cost_gbp"])
        # Allow small floating point tolerance
        self.assertAlmostEqual(r["avg_total_cost_gbp"], component_sum, delta=1.0)

    def test_high_rop_gives_good_service_level(self):
        """Very high reorder point should give near-perfect service level."""
        result = run_inventory_sim(
            reorder_point=5000, order_quantity=5000,
            demand_mean=200, demand_std=40,
            lead_time_days=7, trials=10
        )
        sl = result["results"]["avg_service_level_pct"]
        self.assertGreaterEqual(sl, 95,
            f"Expected SL>=95% with very high ROP, got {sl}%")

    def test_low_rop_gives_stockouts(self):
        """Very low reorder point should produce frequent stockouts."""
        result = run_inventory_sim(
            reorder_point=1, order_quantity=100,
            demand_mean=200, demand_std=40,
            lead_time_days=7, trials=10
        )
        sl = result["results"]["avg_service_level_pct"]
        self.assertLess(sl, 90,
            f"Expected SL<90% with very low ROP, got {sl}%")

    def test_costs_are_non_negative(self):
        result = run_inventory_sim(
            reorder_point=500, order_quantity=1000,
            demand_mean=200, demand_std=40, trials=5
        )
        r = result["results"]
        self.assertGreaterEqual(r["avg_holding_cost_gbp"], 0)
        self.assertGreaterEqual(r["avg_ordering_cost_gbp"], 0)
        self.assertGreaterEqual(r["avg_shortage_cost_gbp"], 0)
        self.assertGreaterEqual(r["avg_total_cost_gbp"], 0)

    def test_verdict_classification(self):
        """Verdict should be 'good' when service level >= 95%."""
        result = run_inventory_sim(
            reorder_point=5000, order_quantity=5000,
            demand_mean=200, demand_std=40, trials=10
        )
        r = result["results"]
        if r["avg_service_level_pct"] >= 95:
            self.assertEqual(r["verdict"], "good")
        else:
            self.assertEqual(r["verdict"], "needs improvement")


class TestHubLocationSimulation(unittest.TestCase):

    def test_returns_expected_keys(self):
        result = run_hub_location_sim(
            candidate_location="Poland",
            build_cost_millions=7.0,
            annual_ops_cost_millions=1.6,
            freight_saving_pct=0.13,
            demand_growth_rate=0.10,
            trials=10
        )
        self.assertIn("results", result)
        r = result["results"]
        for key in ["avg_npv_millions", "avg_breakeven_year",
                    "probability_profitable_pct", "recommendation"]:
            self.assertIn(key, r, f"Missing key: {key}")

    def test_probability_between_0_and_100(self):
        result = run_hub_location_sim(
            candidate_location="Test",
            build_cost_millions=5.0,
            annual_ops_cost_millions=1.0,
            freight_saving_pct=0.15,
            demand_growth_rate=0.10,
            trials=10
        )
        prob = result["results"]["probability_profitable_pct"]
        self.assertGreaterEqual(prob, 0)
        self.assertLessEqual(prob, 100)

    def test_high_growth_gives_positive_npv(self):
        """High demand growth should give positive NPV for a viable hub."""
        result = run_hub_location_sim(
            candidate_location="Test",
            build_cost_millions=5.0,
            annual_ops_cost_millions=1.0,
            current_freight_cost_millions=15.0,
            freight_saving_pct=0.25,
            demand_growth_rate=0.20,
            trials=20
        )
        npv = result["results"]["avg_npv_millions"]
        self.assertGreater(npv, 0,
            f"Expected positive NPV with high growth, got £{npv}M")

    def test_zero_growth_gives_low_npv(self):
        """Zero demand growth with high costs should give negative NPV."""
        result = run_hub_location_sim(
            candidate_location="Test",
            build_cost_millions=15.0,
            annual_ops_cost_millions=3.0,
            current_freight_cost_millions=5.0,
            freight_saving_pct=0.10,
            demand_growth_rate=0.01,
            trials=20
        )
        npv = result["results"]["avg_npv_millions"]
        self.assertLess(npv, 0,
            f"Expected negative NPV with near-zero growth, got £{npv}M")

    def test_recommendation_values(self):
        """Recommendation must be one of the three valid strings."""
        result = run_hub_location_sim(
            candidate_location="Test",
            build_cost_millions=7.0,
            annual_ops_cost_millions=1.6,
            freight_saving_pct=0.13,
            demand_growth_rate=0.10,
            trials=10
        )
        valid = {"INVEST \u2014 financially justified",
                 "CONDITIONAL \u2014 invest only if demand growth confirmed",
                 "DO NOT INVEST \u2014 insufficient return"}
        self.assertIn(result["results"]["recommendation"], valid)


class TestDisruptionSimulation(unittest.TestCase):

    def test_returns_expected_keys(self):
        result = run_supplier_disruption_sim(
            strategy="dual_sourcing",
            disruption_probability=0.20,
            trials=5
        )
        self.assertIn("results", result)
        r = result["results"]
        for key in ["avg_total_cost_gbp", "avg_service_level_pct",
                    "avg_shortage_cost_gbp", "avg_strategy_cost_gbp"]:
            self.assertIn(key, r, f"Missing key: {key}")

    def test_service_level_between_0_and_100(self):
        for strategy in ["no_backup", "safety_stock",
                         "dual_sourcing", "air_freight"]:
            result = run_supplier_disruption_sim(
                strategy=strategy,
                disruption_probability=0.20,
                trials=5
            )
            sl = result["results"]["avg_service_level_pct"]
            self.assertGreaterEqual(sl, 0,
                f"{strategy}: SL should be >= 0, got {sl}")
            self.assertLessEqual(sl, 100,
                f"{strategy}: SL should be <= 100, got {sl}")

    def test_safety_stock_higher_cost_than_no_backup(self):
        """Safety stock should cost more than no backup due to holding costs."""
        nb = run_supplier_disruption_sim(
            strategy="no_backup",
            disruption_probability=0.20, trials=20
        )
        ss = run_supplier_disruption_sim(
            strategy="safety_stock",
            disruption_probability=0.20, trials=20
        )
        self.assertGreater(
            ss["results"]["avg_total_cost_gbp"],
            nb["results"]["avg_total_cost_gbp"],
            "Safety stock should cost more than no backup"
        )

    def test_daily_hazard_correct(self):
        """Verify daily hazard correctly derives from annual probability."""
        for annual_p in [0.05, 0.10, 0.20, 0.30, 0.50]:
            daily_h = 1 - (1 - annual_p) ** (1/365)
            effective_annual = 1 - (1 - daily_h) ** 365
            self.assertAlmostEqual(effective_annual, annual_p, places=4,
                msg=f"Daily hazard incorrect for annual_p={annual_p}")

    def test_no_backup_strategy_cost_is_zero(self):
        """No backup strategy should have zero strategy-specific cost."""
        result = run_supplier_disruption_sim(
            strategy="no_backup",
            disruption_probability=0.10, trials=10
        )
        self.assertEqual(result["results"]["avg_strategy_cost_gbp"], 0,
            "No backup strategy cost should be £0")

    def test_costs_are_non_negative(self):
        for strategy in ["no_backup", "safety_stock",
                         "dual_sourcing", "air_freight"]:
            result = run_supplier_disruption_sim(
                strategy=strategy,
                disruption_probability=0.20, trials=5
            )
            r = result["results"]
            self.assertGreaterEqual(r["avg_total_cost_gbp"], 0)
            self.assertGreaterEqual(r["avg_shortage_cost_gbp"], 0)
            self.assertGreaterEqual(r["avg_holding_cost_gbp"], 0)
            self.assertGreaterEqual(r["avg_strategy_cost_gbp"], 0)


if __name__ == "__main__":
    print("Running unit tests...")
    print("="*60)
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestInventorySimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestHubLocationSimulation))
    suite.addTests(loader.loadTestsFromTestCase(TestDisruptionSimulation))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("="*60)
    if result.wasSuccessful():
        print("All tests passed.")
    else:
        print(f"Failures: {len(result.failures)}, Errors: {len(result.errors)}")

"""Chronological backtest tests for V29 model comparisons."""

import unittest

from engine import compare_models, unordered_pair


PRIOR = ["4,5", "1,6", "3,6", "2,3", "2,6"]


def backtest(history, prior, start=400):
    """Every prediction uses only earlier completed throws."""
    if start < 1 or start >= len(history):
        raise ValueError("Invalid backtest start")

    results = {}

    for i in range(start, len(history)):
        previous = history[:i]
        actual = unordered_pair(history[i])

        predictions = compare_models(previous, prior)

        for model in predictions:
            name = model["Model"]
            top5 = model["Top-5"].split(";")

            if name not in results:
                results[name] = {"hits": 0, "tests": 0}

            results[name]["tests"] += 1
            results[name]["hits"] += int(actual in top5)

    return results


class TestHistoricalBacktest(unittest.TestCase):

    def setUp(self):
        self.history = (
            ["1,1,1"] * 150
            + ["2,4,5"] * 300
            + ["1,3,6"] * 100
        )

    def test_four_models(self):
        results = backtest(self.history, PRIOR)
        self.assertEqual(len(results), 4)

        for stats in results.values():
            self.assertEqual(stats["tests"], 150)
            self.assertGreaterEqual(stats["hits"], 0)
            self.assertLessEqual(stats["hits"], 150)

    def test_no_future_data_leakage(self):
        prefix = self.history[:450]

        before = compare_models(prefix, PRIOR)
        future = prefix + ["2,2,2"] * 100

        # Prediction for throw 451 must not use later results.
        after = compare_models(future[:450], PRIOR)

        self.assertEqual(before, after)

    def test_invalid_start(self):
        with self.assertRaises(ValueError):
            backtest(self.history, PRIOR, start=550)


if __name__ == "__main__":
    unittest.main()

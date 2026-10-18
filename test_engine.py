
import unittest

from engine import (
    PAIRS,
    score_v29,
    score_details,
    compare_models,
    validate_throw,
    unordered_pair,
)

PRIOR = ["4,5", "1,6", "3,6", "2,3", "2,6"]


class TestV29Engine(unittest.TestCase):

    def setUp(self):
        self.history = (
            ["1,1,1"] * 300
            + ["2,4,5"] * 400
        )

    def test_validate_throw(self):
        self.assertEqual(validate_throw("1, 3, 2"), "1,3,2")
        with self.assertRaises(ValueError):
            validate_throw("3,1,2")

    def test_unordered_pair(self):
        self.assertEqual(unordered_pair("1,3,2"), "2,3")

    def test_legacy_scoring(self):
        predictions = score_v29(self.history, PRIOR)
        self.assertEqual(len(predictions), 5)
        self.assertEqual(len(set(predictions)), 5)
        self.assertTrue(set(predictions).issubset(set(PAIRS)))
        self.assertEqual(predictions[0], "4,5")

    def test_score_details(self):
        details = score_details(self.history, PRIOR)
        self.assertEqual(len(details), 21)
        self.assertEqual(
            [row["Pair"] for row in details[:5]],
            score_v29(self.history, PRIOR),
        )
        self.assertEqual(
            sum(row["Last 400 Frequency"] for row in details),
            400,
        )

    def test_compare_models(self):
        results = compare_models(self.history, PRIOR)
        self.assertEqual(len(results), 4)

        by_name = {row["Model"]: row for row in results}

        for row in results:
            self.assertEqual(
                len(row["Top-5"].split(";")),
                5,
            )

        self.assertEqual(
            by_name["V29 legacy (400 + fixed V28)"]["History Used"],
            400,
        )
        self.assertEqual(
            by_name["Full history + fixed V28"]["History Used"],
            700,
        )
        self.assertEqual(
            by_name["Rolling frequency only (400)"]["History Used"],
            400,
        )
        self.assertEqual(
            by_name["Full history frequency only"]["History Used"],
            700,
        )


if __name__ == "__main__":
    unittest.main()

"""Unit tests for lexicographic model selection in train_optimal_overtaker.py.

Verifies:
1. Safety-first ordering: A 0%-crash / 70 km/h candidate strictly beats a 10%-crash / 95 km/h
   candidate, resolving the flaw in the legacy eval_score (speed - 2*crash_rate).
2. Equal crash counts: Higher mean speed wins among candidates with equal crashes (e.g., both 0).
3. Non-zero crashes: Fewer crashes wins when comparing candidates with >= 1 crash.
4. Sequence simulation: Across training iterations, zero-crash checkpoints are never displaced
   by faster but crashing checkpoints.
"""

import unittest
from train_optimal_overtaker import is_better_candidate


class TestLexicographicModelSelection(unittest.TestCase):
    def test_zero_crash_beats_fast_crashing_candidate(self) -> None:
        """Confirm a 0%-crash / 70 km/h candidate is selected over a 10%-crash / 95 km/h candidate."""
        cand_zero_crash = {
            "crashes": 0,
            "crash_rate": 0.0,
            "mean_speed_kmh": 70.0,
            "mean_overtakes": 4.5,
            "mean_lane_changes": 2.0,
        }
        cand_crashing_fast = {
            "crashes": 1,
            "crash_rate": 10.0,
            "mean_speed_kmh": 95.0,
            "mean_overtakes": 8.0,
            "mean_lane_changes": 4.0,
        }

        # 1. Under old eval_score (speed - crash_rate * 2):
        old_score_zero = cand_zero_crash["mean_speed_kmh"] - cand_zero_crash["crash_rate"] * 2.0
        old_score_crashing = cand_crashing_fast["mean_speed_kmh"] - cand_crashing_fast["crash_rate"] * 2.0
        # Notice: 70 - 0 = 70.0, while 95 - 20 = 75.0. Legacy score erroneously picked the crashing policy!
        self.assertGreater(
            old_score_crashing,
            old_score_zero,
            "Legacy eval_score should demonstrate the dangerous preference for the crashing candidate.",
        )

        # 2. Under lexicographic selection:
        # If cand_crashing_fast is current best, cand_zero_crash MUST replace it.
        self.assertTrue(
            is_better_candidate(cand_zero_crash, cand_crashing_fast),
            "Zero-crash candidate (70 km/h) must beat 10%-crash candidate (95 km/h).",
        )

        # If cand_zero_crash is current best, cand_crashing_fast MUST NOT replace it.
        self.assertFalse(
            is_better_candidate(cand_crashing_fast, cand_zero_crash),
            "10%-crash candidate (95 km/h) must not displace a zero-crash candidate.",
        )

    def test_zero_crash_speed_tiebreaker(self) -> None:
        """Among zero-crash candidates, higher mean speed wins."""
        cand_70 = {"crashes": 0, "crash_rate": 0.0, "mean_speed_kmh": 70.0}
        cand_78 = {"crashes": 0, "crash_rate": 0.0, "mean_speed_kmh": 78.5}
        cand_65 = {"crashes": 0, "crash_rate": 0.0, "mean_speed_kmh": 65.0}

        self.assertTrue(is_better_candidate(cand_78, cand_70))
        self.assertFalse(is_better_candidate(cand_65, cand_70))

    def test_crashing_candidates_fewer_crashes_wins(self) -> None:
        """Among candidates with crashes, fewer crashes wins."""
        cand_1_crash = {"crashes": 1, "crash_rate": 10.0, "mean_speed_kmh": 80.0}
        cand_2_crash = {"crashes": 2, "crash_rate": 20.0, "mean_speed_kmh": 95.0}

        self.assertTrue(is_better_candidate(cand_1_crash, cand_2_crash))
        self.assertFalse(is_better_candidate(cand_2_crash, cand_1_crash))

    def test_crashing_candidates_equal_crashes_higher_speed_wins(self) -> None:
        """Among candidates with equal non-zero crashes, higher speed wins."""
        cand_1_crash_slow = {"crashes": 1, "crash_rate": 10.0, "mean_speed_kmh": 75.0}
        cand_1_crash_fast = {"crashes": 1, "crash_rate": 10.0, "mean_speed_kmh": 85.0}

        self.assertTrue(is_better_candidate(cand_1_crash_fast, cand_1_crash_slow))
        self.assertFalse(is_better_candidate(cand_1_crash_slow, cand_1_crash_fast))

    def test_initial_candidate_accepted(self) -> None:
        """When best is None, any valid candidate is accepted as the initial best."""
        cand = {"crashes": 3, "crash_rate": 30.0, "mean_speed_kmh": 60.0}
        self.assertTrue(is_better_candidate(cand, None))

    def test_training_sequence_simulation(self) -> None:
        """Simulate candidate arrivals across iterations to verify model trajectory selection."""
        candidates = [
            {"iter": 20, "crashes": 3, "crash_rate": 30.0, "mean_speed_kmh": 80.0},
            {"iter": 40, "crashes": 1, "crash_rate": 10.0, "mean_speed_kmh": 85.0},
            {"iter": 60, "crashes": 1, "crash_rate": 10.0, "mean_speed_kmh": 92.0},
            {"iter": 80, "crashes": 0, "crash_rate": 0.0, "mean_speed_kmh": 70.0},
            {"iter": 100, "crashes": 1, "crash_rate": 10.0, "mean_speed_kmh": 99.0},  # Aggressive fast outlier
            {"iter": 120, "crashes": 0, "crash_rate": 0.0, "mean_speed_kmh": 74.2},  # Faster safe policy
        ]

        best = None
        best_history = []
        for cand in candidates:
            if is_better_candidate(cand, best):
                best = cand
            best_history.append(best["iter"])

        # Expected selection progression:
        # Iter 20: initial best (iter 20)
        # Iter 40: 1 crash beats 3 crashes (iter 40)
        # Iter 60: 1 crash @ 92 km/h beats 1 crash @ 85 km/h (iter 60)
        # Iter 80: 0 crash @ 70 km/h beats 1 crash @ 92 km/h (iter 80)
        # Iter 100: 1 crash @ 99 km/h rejected against 0 crash (remains iter 80)
        # Iter 120: 0 crash @ 74.2 km/h beats 0 crash @ 70 km/h (iter 120)
        expected_history = [20, 40, 60, 80, 80, 120]
        self.assertEqual(best_history, expected_history)
        self.assertEqual(best["iter"], 120)
        self.assertEqual(best["crashes"], 0)
        self.assertEqual(best["mean_speed_kmh"], 74.2)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for lexicographic model selection with behavioral sanity checks.

Verifies:
1. Safety-first ordering: A 0%-crash / 70 km/h candidate strictly beats a 10%-crash / 95 km/h candidate.
2. Equal crash counts: Higher mean speed wins among candidates with equal crashes and low chatter.
3. Non-zero crashes: Fewer crashes wins when comparing candidates with >= 1 crash.
4. Behavioral sanity check: A low-crash / high-chatter candidate (excessive rapid lane-boundary reversals)
   is deprioritized / rejected in favor of a low-crash / low-chatter candidate, even if the high-chatter
   candidate has higher speed.
5. Sequence simulation: Across training iterations, low-chatter safe checkpoints are never displaced
   by faster but violently chattering checkpoints.
6. Pipeline consistency: Both train_optimal_overtaker.py and scripts/train_deepset_v2.py enforce
   identical behavioral sanity criteria.
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from train_optimal_overtaker import is_better_candidate as is_better_candidate_optimal
from scripts.train_deepset_v2 import is_better_candidate as is_better_candidate_deepset


class TestLexicographicModelSelection(unittest.TestCase):
    def test_zero_crash_beats_fast_crashing_candidate(self) -> None:
        """Confirm a 0%-crash / 70 km/h candidate is selected over a 10%-crash / 95 km/h candidate."""
        cand_zero_crash = {
            "crashes": 0,
            "crash_rate": 0.0,
            "mean_speed_kmh": 70.0,
            "mean_overtakes": 4.5,
            "mean_lane_changes": 2.0,
            "action_reversal_rate": 0.0,
        }
        cand_crashing_fast = {
            "crashes": 1,
            "crash_rate": 10.0,
            "mean_speed_kmh": 95.0,
            "mean_overtakes": 8.0,
            "mean_lane_changes": 4.0,
            "action_reversal_rate": 0.0,
        }

        # Legacy eval_score (speed - 2*crash_rate) erroneously preferred crashing:
        old_score_zero = cand_zero_crash["mean_speed_kmh"] - cand_zero_crash["crash_rate"] * 2.0
        old_score_crashing = cand_crashing_fast["mean_speed_kmh"] - cand_crashing_fast["crash_rate"] * 2.0
        self.assertGreater(old_score_crashing, old_score_zero)

        # Under lexicographic selection:
        self.assertTrue(
            is_better_candidate_optimal(cand_zero_crash, cand_crashing_fast),
            "Zero-crash candidate (70 km/h) must beat 10%-crash candidate (95 km/h).",
        )
        self.assertFalse(
            is_better_candidate_optimal(cand_crashing_fast, cand_zero_crash),
            "10%-crash candidate (95 km/h) must not displace a zero-crash candidate.",
        )

    def test_zero_crash_speed_tiebreaker(self) -> None:
        """Among zero-crash candidates with acceptable chatter, higher mean speed wins."""
        cand_70 = {"crashes": 0, "crash_rate": 0.0, "mean_speed_kmh": 70.0, "action_reversal_rate": 0.2}
        cand_78 = {"crashes": 0, "crash_rate": 0.0, "mean_speed_kmh": 78.5, "action_reversal_rate": 0.3}
        cand_65 = {"crashes": 0, "crash_rate": 0.0, "mean_speed_kmh": 65.0, "action_reversal_rate": 0.1}

        self.assertTrue(is_better_candidate_optimal(cand_78, cand_70))
        self.assertFalse(is_better_candidate_optimal(cand_65, cand_70))

    def test_crashing_candidates_fewer_crashes_wins(self) -> None:
        """Among candidates with crashes, fewer crashes wins."""
        cand_1_crash = {"crashes": 1, "crash_rate": 10.0, "mean_speed_kmh": 80.0}
        cand_2_crash = {"crashes": 2, "crash_rate": 20.0, "mean_speed_kmh": 95.0}

        self.assertTrue(is_better_candidate_optimal(cand_1_crash, cand_2_crash))
        self.assertFalse(is_better_candidate_optimal(cand_2_crash, cand_1_crash))

    def test_crashing_candidates_equal_crashes_higher_speed_wins(self) -> None:
        """Among candidates with equal non-zero crashes, higher speed wins."""
        cand_1_crash_slow = {"crashes": 1, "crash_rate": 10.0, "mean_speed_kmh": 75.0}
        cand_1_crash_fast = {"crashes": 1, "crash_rate": 10.0, "mean_speed_kmh": 85.0}

        self.assertTrue(is_better_candidate_optimal(cand_1_crash_fast, cand_1_crash_slow))
        self.assertFalse(is_better_candidate_optimal(cand_1_crash_slow, cand_1_crash_fast))

    def test_initial_candidate_accepted(self) -> None:
        """When best is None, any valid candidate is accepted as the initial best."""
        cand = {"crashes": 3, "crash_rate": 30.0, "mean_speed_kmh": 60.0}
        self.assertTrue(is_better_candidate_optimal(cand, None))

    def test_low_crash_low_chatter_beats_low_crash_high_chatter(self) -> None:
        """Synthetic test: A zero-crash candidate with excessive rapid reversals (> threshold)

        must NOT be selected over a zero-crash low-chatter candidate, even if the
        chattering candidate is substantially faster.
        """
        # Candidate A: 0 crashes, fast (78.5 km/h), but severe chatter (8.5 reversals/ep)
        cand_high_chatter = {
            "crashes": 0,
            "crash_rate": 0.0,
            "mean_speed_kmh": 78.5,
            "action_reversal_rate": 8.5,
            "mean_lane_changes": 22.0,
        }
        # Candidate B: 0 crashes, calmer speed (71.0 km/h), clean driving (0.2 reversals/ep)
        cand_low_chatter = {
            "crashes": 0,
            "crash_rate": 0.0,
            "mean_speed_kmh": 71.0,
            "action_reversal_rate": 0.2,
            "mean_lane_changes": 2.5,
        }

        # 1. Low-chatter candidate strictly beats high-chatter candidate
        self.assertTrue(
            is_better_candidate_optimal(cand_low_chatter, cand_high_chatter, reversal_threshold=2.0),
            "Low-chatter candidate (71 km/h, 0.2 rev/ep) must beat high-chatter candidate (78.5 km/h, 8.5 rev/ep).",
        )

        # 2. High-chatter candidate CANNOT displace an established low-chatter candidate
        self.assertFalse(
            is_better_candidate_optimal(cand_high_chatter, cand_low_chatter, reversal_threshold=2.0),
            "High-chatter candidate (78.5 km/h, 8.5 rev/ep) must NOT displace low-chatter candidate (71 km/h, 0.2 rev/ep).",
        )

        # 3. Check alias keys ('reversal_rate', 'rapid_reversals_per_ep')
        cand_high_chatter_alt = {"crashes": 0, "mean_speed_kmh": 82.0, "reversal_rate": 6.0}
        cand_low_chatter_alt = {"crashes": 0, "mean_speed_kmh": 70.0, "rapid_reversals_per_ep": 0.5}
        self.assertTrue(is_better_candidate_optimal(cand_low_chatter_alt, cand_high_chatter_alt))
        self.assertFalse(is_better_candidate_optimal(cand_high_chatter_alt, cand_low_chatter_alt))

    def test_chatter_sequence_simulation(self) -> None:
        """Simulate training trajectory where exploratory iterations exhibit high chatter.

        Verifies that high-chatter policies do not lock in or displace safe, low-chatter policies.
        """
        candidates = [
            {"iter": 10, "crashes": 2, "mean_speed_kmh": 80.0, "action_reversal_rate": 1.0},
            {"iter": 20, "crashes": 0, "mean_speed_kmh": 78.0, "action_reversal_rate": 9.2},  # 0 crashes but chatter
            {"iter": 30, "crashes": 0, "mean_speed_kmh": 72.0, "action_reversal_rate": 0.4},  # Clean 0 crash
            {"iter": 40, "crashes": 0, "mean_speed_kmh": 84.0, "action_reversal_rate": 11.0}, # Very fast chatter outlier
            {"iter": 50, "crashes": 0, "mean_speed_kmh": 75.5, "action_reversal_rate": 0.2},  # Faster clean policy
        ]

        best = None
        best_history = []
        for cand in candidates:
            if is_better_candidate_optimal(cand, best, reversal_threshold=2.0):
                best = cand
            best_history.append(best["iter"])

        # Expected selection progression:
        # Iter 10: initial best (iter 10, 2 crashes)
        # Iter 20: 0 crash beats 2 crashes (iter 20)
        # Iter 30: clean 0 crash (0.4 rev) beats chattering 0 crash (9.2 rev) despite lower speed (iter 30)
        # Iter 40: 84 km/h chattering outlier rejected against clean iter 30 (remains iter 30)
        # Iter 50: clean 75.5 km/h beats clean 72.0 km/h (iter 50)
        expected_history = [10, 20, 30, 30, 50]
        self.assertEqual(best_history, expected_history)
        self.assertEqual(best["iter"], 50)
        self.assertEqual(best["crashes"], 0)
        self.assertEqual(best["mean_speed_kmh"], 75.5)
        self.assertLessEqual(best["action_reversal_rate"], 2.0)

    def test_deepset_v2_model_selection_chatter(self) -> None:
        """Verify scripts/train_deepset_v2.py's is_better_candidate has identical behavioral sanity."""
        cand_high = {"crashes": 0, "merge_successes": 50, "mean_speed_kmh": 85.0, "action_reversal_rate": 7.4}
        cand_low = {"crashes": 0, "merge_successes": 50, "mean_speed_kmh": 74.0, "action_reversal_rate": 0.5}

        self.assertTrue(is_better_candidate_deepset(cand_low, cand_high, reversal_threshold=2.0))
        self.assertFalse(is_better_candidate_deepset(cand_high, cand_low, reversal_threshold=2.0))


if __name__ == "__main__":
    unittest.main()

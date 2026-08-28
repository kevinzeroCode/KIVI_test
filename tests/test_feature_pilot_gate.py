"""CPU-only tests for analysis/feature_pilot_gate.py -- the Stage G3B
pre-registered A/B/C/D decision gate (hardened, Stage-G3B-IMPL and
G3B-IMPL-2 rounds).

All inputs here are synthetic (no real feature/sensitivity data exists yet
at this stage of the project); these tests prove the gate's COMBINATION
LOGIC -- including the undefined-Spearman (raw_rho=None -> gate_rho=0.0)
convention -- is deterministic and exactly matches the hardened wording in
docs/stage_g3b_pre_registration.md, not that any particular scientific
conclusion holds.
"""
import unittest

from analysis.feature_pilot_gate import (
    GateInputError,
    PRIMARY_STAGE_E_TASKS,
    compute_layer0_diagnostic_rho,
    compute_task_specific_rho,
    evaluate_axis_gate,
    evaluate_criterion_a,
    evaluate_criterion_b,
    evaluate_criterion_c,
    evaluate_criterion_d,
    evaluate_feature_pilot_gate,
    to_gate_rho,
)

ALL_NEGATIVE_STRONG = {"trec": -0.9, "lcc": -0.9, "passage_retrieval_en": -0.9, "2wikimqa": -0.9}


class ToGateRhoTest(unittest.TestCase):
    def test_none_maps_to_zero(self):
        self.assertEqual(to_gate_rho(None), 0.0)

    def test_real_value_passed_through_unchanged(self):
        self.assertEqual(to_gate_rho(-0.73), -0.73)
        self.assertEqual(to_gate_rho(0.0), 0.0)
        self.assertEqual(to_gate_rho(0.5), 0.5)


class KnownTrecValueUndefinedCaseTest(unittest.TestCase):
    """Regression test for the known pre-existing Stage-E fact: trec's
    Value SignedSensitivity is constant across the 8 sampled layers, so
    its task-specific Spearman correlation is mathematically undefined.
    Uses a synthetic representation of exactly that shape (one vector with
    zero variance) -- does not fabricate a finite Spearman coefficient.
    """

    def test_constant_sensitivity_vector_gives_undefined_raw_rho(self):
        relative_l2 = {0: 0.1, 4: 0.3, 9: 0.2, 13: 0.5, 18: 0.4, 22: 0.6, 27: 0.15, 31: 0.25}
        constant_sensitivity = {l: 0.0 for l in relative_l2}  # trec Value: identical delta at every layer
        rho_by_task = compute_task_specific_rho(
            {"trec": relative_l2}, {"trec": constant_sensitivity}, tasks=("trec",)
        )
        self.assertIsNone(rho_by_task["trec"])  # raw_rho = undefined, never a fabricated 0.0

    def test_constant_feature_vector_also_gives_undefined_raw_rho(self):
        # Symmetric case: the FEATURE side (not the sensitivity side) has
        # zero variance -- must be equally undefined, not silently treated
        # differently from a constant sensitivity vector.
        constant_relative_l2 = {0: 0.2, 4: 0.2, 9: 0.2, 13: 0.2}
        sensitivity = {0: -0.5, 4: 0.3, 9: -0.1, 13: 0.9}
        rho_by_task = compute_task_specific_rho(
            {"t": constant_relative_l2}, {"t": sensitivity}, tasks=("t",)
        )
        self.assertIsNone(rho_by_task["t"])

    def test_undefined_maps_to_gate_rho_zero_and_cannot_help_criterion_a(self):
        # trec is undefined (gate_rho=0); the other 3 are strongly
        # negative. Criterion A needs >=3/4 negative -- trec's 0 must NOT
        # count, so only 3 of 4 are negative (the non-trec three), which
        # is exactly the boundary -- still passes on those three alone,
        # but trec itself contributes nothing.
        raw_rho = {"trec": None, "lcc": -0.9, "passage_retrieval_en": -0.9, "2wikimqa": -0.9}
        result = evaluate_criterion_a(raw_rho)
        self.assertEqual(result["gate_rho_by_task"]["trec"], 0.0)
        self.assertEqual(result["negative_count"], 3)  # trec's 0.0 is not negative
        self.assertTrue(result["pass"])

    def test_undefined_trec_cannot_be_the_deciding_vote_for_criterion_a(self):
        # Now only 2 of the other 3 are negative -- trec's gate_rho=0
        # must NOT be able to supply the missing 3rd negative vote.
        raw_rho = {"trec": None, "lcc": -0.9, "passage_retrieval_en": -0.9, "2wikimqa": +0.3}
        result = evaluate_criterion_a(raw_rho)
        self.assertEqual(result["negative_count"], 2)
        self.assertFalse(result["pass"])

    def test_undefined_trec_pulls_criterion_b_median_toward_zero_not_dropped(self):
        # 4 values with trec undefined (-> 0.0) and three real values.
        # Median of [0.0, 0.6, 0.7, 0.8] (sorted abs) = (0.6+0.7)/2 = 0.65
        # if trec were DROPPED (n=3, median=0.7) the test would show a
        # different number -- this proves trec's 0 is actually included,
        # not silently removed from the set.
        raw_rho = {"trec": None, "lcc": -0.6, "passage_retrieval_en": -0.7, "2wikimqa": -0.8}
        result = evaluate_criterion_b(raw_rho)
        self.assertEqual(sorted(result["gate_rho_by_task"].values()), [-0.8, -0.7, -0.6, 0.0])
        self.assertAlmostEqual(result["median_abs_gate_rho"], 0.65, places=6)  # NOT 0.7 (the drop-NaN answer)

    def test_undefined_trec_in_criterion_c_remaining_set(self):
        # After removing lcc, remaining = {trec, passage_retrieval_en, 2wikimqa}.
        # trec is undefined -> gate_rho=0.0, included in both the
        # negative-count and the median, never dropped.
        raw_rho = {"trec": None, "lcc": +0.9, "passage_retrieval_en": -0.6, "2wikimqa": -0.7}
        result = evaluate_criterion_c(raw_rho)
        self.assertEqual(set(result["remaining_tasks"]), {"trec", "passage_retrieval_en", "2wikimqa"})
        self.assertEqual(result["gate_rho_remaining"]["trec"], 0.0)
        self.assertEqual(result["negative_count"], 2)  # trec's 0.0 does not count
        # median of [0.0, -0.6, -0.7] sorted = [-0.7, -0.6, 0.0] -> median = -0.6
        self.assertAlmostEqual(result["median_remaining_gate_rho"], -0.6, places=6)
        self.assertTrue(result["pass"])

    def test_criterion_d_undefined_always_fails(self):
        # Section 5: raw rho undefined (either n=3 vector has zero
        # variance) -> gate_rho=0 -> Criterion D FAILS (0 is not < 0).
        result = evaluate_criterion_d(None)
        self.assertEqual(result["layer0_gate_rho"], 0.0)
        self.assertFalse(result["pass"])

    def test_criterion_d_undefined_from_constant_layer0_sensitivity(self):
        relative_l2 = {"lcc": 0.5, "multifieldqa_en": 0.1, "samsum": 0.3}
        constant_sensitivity = {"lcc": 0.0, "multifieldqa_en": 0.0, "samsum": 0.0}
        raw_rho = compute_layer0_diagnostic_rho(relative_l2, constant_sensitivity)
        self.assertIsNone(raw_rho)
        result = evaluate_criterion_d(raw_rho)
        self.assertFalse(result["pass"])


class CriterionATest(unittest.TestCase):
    def test_four_of_four_negative_passes(self):
        result = evaluate_criterion_a(ALL_NEGATIVE_STRONG)
        self.assertEqual(result["negative_count"], 4)
        self.assertTrue(result["pass"])

    def test_exactly_three_of_four_passes(self):
        rho = {"trec": -0.6, "lcc": -0.6, "passage_retrieval_en": -0.6, "2wikimqa": +0.1}
        self.assertTrue(evaluate_criterion_a(rho)["pass"])

    def test_two_of_four_fails(self):
        rho = {"trec": -0.6, "lcc": -0.6, "passage_retrieval_en": +0.1, "2wikimqa": +0.1}
        self.assertFalse(evaluate_criterion_a(rho)["pass"])

    def test_missing_task_key_raises(self):
        with self.assertRaises(GateInputError):
            evaluate_criterion_a({"trec": -0.5, "lcc": -0.5, "passage_retrieval_en": -0.5})

    def test_extra_task_key_raises(self):
        rho = dict(ALL_NEGATIVE_STRONG)
        rho["samsum"] = -0.5
        with self.assertRaises(GateInputError):
            evaluate_criterion_a(rho)

    def test_raw_rho_preserved_alongside_gate_rho(self):
        raw_rho = {"trec": None, "lcc": -0.9, "passage_retrieval_en": -0.9, "2wikimqa": -0.9}
        result = evaluate_criterion_a(raw_rho)
        self.assertIsNone(result["raw_rho_by_task"]["trec"])  # never overwritten with 0.0
        self.assertEqual(result["gate_rho_by_task"]["trec"], 0.0)


class CriterionBTest(unittest.TestCase):
    def test_exactly_at_threshold_passes(self):
        rho = {"trec": -0.5, "lcc": -0.5, "passage_retrieval_en": -0.5, "2wikimqa": -0.5}
        result = evaluate_criterion_b(rho)
        self.assertEqual(result["median_abs_gate_rho"], 0.5)
        self.assertTrue(result["pass"])

    def test_just_below_threshold_fails(self):
        rho = {"trec": -0.49, "lcc": -0.49, "passage_retrieval_en": -0.49, "2wikimqa": -0.49}
        self.assertFalse(evaluate_criterion_b(rho)["pass"])

    def test_uses_absolute_value_sign_agnostic(self):
        rho = {"trec": +0.6, "lcc": +0.6, "passage_retrieval_en": +0.6, "2wikimqa": +0.6}
        self.assertTrue(evaluate_criterion_b(rho)["pass"])

    def test_undefined_task_contributes_zero_not_a_closed_failure(self):
        # Hardened behavior: no longer an automatic fail-closed on any
        # None -- the undefined task contributes gate_rho=0.0 to the
        # median like any other value.
        rho = {"trec": None, "lcc": -0.9, "passage_retrieval_en": -0.9, "2wikimqa": -0.9}
        result = evaluate_criterion_b(rho)
        self.assertIsNotNone(result["median_abs_gate_rho"])
        self.assertAlmostEqual(result["median_abs_gate_rho"], 0.9, places=6)  # median([0, .9, .9, .9])
        self.assertTrue(result["pass"])

    def test_custom_threshold_respected(self):
        rho = {"trec": -0.3, "lcc": -0.3, "passage_retrieval_en": -0.3, "2wikimqa": -0.3}
        self.assertFalse(evaluate_criterion_b(rho, threshold=0.5)["pass"])
        self.assertTrue(evaluate_criterion_b(rho, threshold=0.2)["pass"])


class CriterionCTest(unittest.TestCase):
    """Hardened wording: after removing lcc, >= 2 of remaining 3 negative
    AND median of those 3 (signed gate_rho) < 0. Explicitly not 3/3."""

    def test_three_of_three_remaining_negative_passes(self):
        rho = {"trec": -0.6, "lcc": +0.9, "passage_retrieval_en": -0.6, "2wikimqa": -0.6}
        result = evaluate_criterion_c(rho)
        self.assertEqual(result["remaining_tasks"], ["trec", "passage_retrieval_en", "2wikimqa"])
        self.assertEqual(result["negative_count"], 3)
        self.assertTrue(result["pass"])

    def test_exactly_two_of_three_remaining_negative_passes(self):
        rho = {"trec": -0.6, "lcc": +0.9, "passage_retrieval_en": -0.6, "2wikimqa": +0.1}
        result = evaluate_criterion_c(rho)
        self.assertEqual(result["negative_count"], 2)
        self.assertTrue(result["pass"])

    def test_one_of_three_remaining_negative_fails(self):
        rho = {"trec": -0.6, "lcc": -0.9, "passage_retrieval_en": +0.6, "2wikimqa": +0.1}
        result = evaluate_criterion_c(rho)
        self.assertEqual(result["negative_count"], 1)
        self.assertFalse(result["pass"])

    def test_two_of_three_negative_implies_negative_median(self):
        # For 3 sorted values, if exactly 2 are negative they must be the
        # two smallest, so the median (middle value) is always one of
        # them -- i.e. negative_count>=2 mathematically guarantees
        # median<0 for n=3. This test documents that fact rather than
        # treating it as a coincidence of the chosen numbers.
        rho = {"trec": -0.6, "lcc": +0.9, "passage_retrieval_en": 0.0, "2wikimqa": -0.6}
        result = evaluate_criterion_c(rho)
        self.assertEqual(result["negative_count"], 2)
        self.assertEqual(result["median_remaining_gate_rho"], -0.6)
        self.assertTrue(result["pass"])

    def test_median_exactly_zero_fails(self):
        rho = {"trec": -0.6, "lcc": +0.9, "passage_retrieval_en": 0.0, "2wikimqa": +0.1}
        result = evaluate_criterion_c(rho)
        self.assertFalse(result["pass"])

    def test_does_not_require_three_of_three(self):
        # Explicit regression guard against the original ambiguous "3/4"
        # wording being misapplied as "3/3 remaining".
        rho = {"trec": -0.6, "lcc": +0.9, "passage_retrieval_en": -0.6, "2wikimqa": +0.4}
        result = evaluate_criterion_c(rho)
        self.assertEqual(result["required_negative_count"], 2)
        self.assertTrue(result["pass"])


class CriterionDTest(unittest.TestCase):
    """Layer-0 n=3 relative_l2-vs-SignedSensitivity correlation only --
    never substituted with a distribution feature."""

    def test_negative_rho_passes(self):
        self.assertTrue(evaluate_criterion_d(-0.6)["pass"])

    def test_positive_rho_fails(self):
        self.assertFalse(evaluate_criterion_d(0.6)["pass"])

    def test_zero_fails(self):
        self.assertFalse(evaluate_criterion_d(0.0)["pass"])

    def test_none_fails_closed(self):
        self.assertFalse(evaluate_criterion_d(None)["pass"])


class ComputeTaskSpecificRhoTest(unittest.TestCase):
    def test_uses_shared_spearman_and_matches_perfect_monotonic_case(self):
        x = {t: {0: 1, 4: 2, 9: 3, 13: 4} for t in PRIMARY_STAGE_E_TASKS}
        y = {t: {0: -1, 4: -2, 9: -3, 13: -4} for t in PRIMARY_STAGE_E_TASKS}
        rho = compute_task_specific_rho(x, y)
        for t in PRIMARY_STAGE_E_TASKS:
            self.assertAlmostEqual(rho[t], -1.0, places=6)

    def test_missing_task_raises(self):
        x = {"trec": {0: 1}, "lcc": {0: 1}, "passage_retrieval_en": {0: 1}}
        y = {"trec": {0: 1}, "lcc": {0: 1}, "passage_retrieval_en": {0: 1}, "2wikimqa": {0: 1}}
        with self.assertRaises(GateInputError):
            compute_task_specific_rho(x, y)


class ComputeLayer0DiagnosticRhoTest(unittest.TestCase):
    def test_perfect_negative_monotonic(self):
        relative_l2 = {"lcc": 0.5, "multifieldqa_en": 0.1, "samsum": 0.3}
        sensitivity = {"lcc": -0.9, "multifieldqa_en": -0.1, "samsum": -0.5}
        rho = compute_layer0_diagnostic_rho(relative_l2, sensitivity)
        self.assertAlmostEqual(rho, -1.0, places=6)

    def test_missing_task_raises(self):
        with self.assertRaises(GateInputError):
            compute_layer0_diagnostic_rho({"lcc": 0.1, "samsum": 0.2}, {"lcc": -0.1, "samsum": -0.2})


class EvaluateAxisGateTest(unittest.TestCase):
    def test_all_criteria_pass_gives_axis_go(self):
        rho = {"trec": -0.6, "lcc": -0.9, "passage_retrieval_en": -0.6, "2wikimqa": -0.6}
        result = evaluate_axis_gate(rho, layer0_raw_rho=-0.6)
        self.assertTrue(result["axis_go"])
        self.assertTrue(all(result[f"criterion_{c}"]["pass"] for c in "abcd"))

    def test_single_failing_criterion_fails_whole_axis(self):
        rho = {"trec": -0.6, "lcc": -0.9, "passage_retrieval_en": -0.6, "2wikimqa": -0.6}
        result = evaluate_axis_gate(rho, layer0_raw_rho=+0.6)  # D fails
        self.assertFalse(result["criterion_d"]["pass"])
        self.assertFalse(result["axis_go"])

    def test_undefined_trec_axis_can_still_go_via_other_three_tasks(self):
        # trec undefined; lcc/passage_retrieval_en/2wikimqa strongly
        # negative; L0 diagnostic negative. A: 3/4 (trec doesn't count,
        # others do) -> pass. B: median([0, .9, .9, .9])=.9 -> pass.
        # C: remaining={trec,passage_retrieval_en,2wikimqa}, trec=0 (not
        # negative), other two negative -> 2/3, median([0,-.9,-.9])=-.9<0 -> pass.
        # D: -0.7 -> pass.
        rho = {"trec": None, "lcc": -0.9, "passage_retrieval_en": -0.9, "2wikimqa": -0.9}
        result = evaluate_axis_gate(rho, layer0_raw_rho=-0.7)
        self.assertTrue(result["axis_go"])

    def test_deterministic(self):
        rho = {"trec": -0.6, "lcc": -0.9, "passage_retrieval_en": -0.6, "2wikimqa": -0.6}
        self.assertEqual(evaluate_axis_gate(rho, -0.6), evaluate_axis_gate(rho, -0.6))


class EvaluateFeaturePilotGateTest(unittest.TestCase):
    def test_go_when_only_key_axis_passes(self):
        key_rho = {"trec": -0.6, "lcc": -0.9, "passage_retrieval_en": -0.6, "2wikimqa": -0.6}
        value_rho = {"trec": 0.1, "lcc": 0.2, "passage_retrieval_en": -0.1, "2wikimqa": 0.3}
        result = evaluate_feature_pilot_gate(key_rho, -0.6, value_rho, +0.4)
        self.assertTrue(result["key"]["axis_go"])
        self.assertFalse(result["value"]["axis_go"])
        self.assertEqual(result["FEATURE_PILOT_STAGE"], "GO")

    def test_no_go_when_neither_axis_passes(self):
        weak_rho = {"trec": 0.1, "lcc": 0.2, "passage_retrieval_en": -0.1, "2wikimqa": 0.05}
        result = evaluate_feature_pilot_gate(weak_rho, 0.3, weak_rho, 0.3)
        self.assertEqual(result["FEATURE_PILOT_STAGE"], "NO_GO")

    def test_no_conditional_go_possible(self):
        # Structural guard: the only two string outcomes are GO/NO_GO.
        key_rho = {"trec": -0.6, "lcc": -0.9, "passage_retrieval_en": -0.6, "2wikimqa": -0.6}
        result = evaluate_feature_pilot_gate(key_rho, -0.6, key_rho, -0.6)
        self.assertIn(result["FEATURE_PILOT_STAGE"], ("GO", "NO_GO"))

    def test_deterministic_across_repeated_calls(self):
        key_rho = {"trec": -0.6, "lcc": -0.9, "passage_retrieval_en": -0.6, "2wikimqa": -0.6}
        value_rho = {"trec": 0.1, "lcc": 0.2, "passage_retrieval_en": -0.1, "2wikimqa": 0.3}
        r1 = evaluate_feature_pilot_gate(key_rho, -0.6, value_rho, 0.4)
        r2 = evaluate_feature_pilot_gate(key_rho, -0.6, value_rho, 0.4)
        self.assertEqual(r1, r2)

    def test_real_trec_undefined_case_does_not_block_a_go(self):
        # End-to-end sanity: the known trec-Value undefined case, embedded
        # in an otherwise-strong axis, must not itself prevent GO.
        key_rho = {"trec": None, "lcc": -0.9, "passage_retrieval_en": -0.7, "2wikimqa": -0.6}
        value_rho = {"trec": None, "lcc": 0.1, "passage_retrieval_en": 0.2, "2wikimqa": -0.1}
        result = evaluate_feature_pilot_gate(key_rho, -0.6, value_rho, 0.3)
        self.assertTrue(result["key"]["axis_go"])
        self.assertEqual(result["FEATURE_PILOT_STAGE"], "GO")


if __name__ == "__main__":
    unittest.main()

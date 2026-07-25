import unittest

from .helper import sum_admissibility, dot_product
from src.conformal_generation import ConformalGeneration
from src.calibration_dataset import IndividualScoreCalibrationDataset
from src.sequence_selector import RunningSumSequenceSelector

TEST_CASES = {
    "Case 0": {
        "x": [],
        "y": [],
        "a": [],
        "gamma": 0,
        "expected_lambda": float("inf"),
        "x_test": [],
        "y_test": [],
        "expected_selection": [],
    },
    "Case 1": {
        "x": [
            [1, 0],
            [1, 1],
        ],
        "y": [
            [[0, 1], [0.5, 0]],
            [[1, 0], [0.5, 1]],
        ],
        "a": [
            [0, 1],
            [0.5, 1],
        ],
        "gamma": 0.5,
        "expected_lambda": 0,
        "x_test": [0.5, 0.5],
        "y_test": [[1, 1.5], [0.25, 0]],
        "expected_selection": [[1, 1.5]],
    },
    # a case where all elements are needed to reach the conformal guarantee
    "Case 2": {
        "x": [
            [0.5, 0.5],
            [0, 2],
            [1, 0],
        ],
        "y": [
            [[1, 0.5], [1.5, 0.5]],
            [[1.2, 2], [0.3, 1.2]],
            [[0, 1.5], [1.3, 1.5], [2, 1.2]],
        ],
        "a": [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.3, 0.3, 0.4],
        ],
        "gamma": 3 / 4,  # correspond to the expected adbmissibility of the sequence being 1
        "expected_lambda": 4,
        "x_test": [2, 1.2],
        "y_test": [[1.5, 0.1], [2, 1.5]],
        "expected_selection": [[1.5, 0.1], [2, 1.5]],
    },
    # a case where the minimal lambda is enough for conformal guarantee
    "Case 3": {
        "x": [
            [1, 0.5],
            [2, 1],
            [1, 1.5],
        ],
        "y": [
            [[0, 0.5], [0.5, 1]],
            [[1, 2], [0, 1.5]],
            [[2, 0.5], [1, 1]],
        ],
        "a": [
            [False, True],
            [True, False],
            [True, True],
        ],
        "gamma": 0.5,  # correspond to 2 out of 3 calibration points being admissible
        "expected_lambda": -float("inf"),
        "x_test": [0.5, 1.5],
        "y_test": [[0.5, 0], [0.25, 1]],
        "expected_selection": [[0.5, 0]],
    },
    # a case where the infimum of lambda is taken from a cumsum instead of individual score
    # this case also verifies scenarios with different sequence lengths for different inputs
    # in the calibration dataset
    "Case 4": {
        "x": [
            [2, 0.5],
            [0.5, 1.5],
            [1, 3],
        ],
        "y": [
            [[1, 1.5], [1, 0.4], [2, 2]],
            [[2, 0.5], [1, 2.5]],
            [[1, 1], [2, 1.5], [0.5, 1.5]],
        ],
        "a": [
            [False, False, True],
            [False, True],
            [False, False, True],
        ],
        "gamma": 0.5,  # correspond to 2 out of 3 calibration points being admissible
        "expected_lambda": 4.95,
        "x_test": [1.5, 2],
        "y_test": [[4, 1.5], [2, 1.5]],
        "expected_selection": [[4, 1.5]],
    },
    "Case 6: iterate inputs": {
        "x": [
            [[0.5, 0.5], [0, 2]],
            [[0, 2], [8, 0]],
            [[2, 0], [1, 0], [0.7, 0.5]],
        ],
        "y": [
            [[1, 0.5], [1.5, 0.5]],
            [[1.2, 2], [0.3, 1.2]],
            [[0, 1.5], [1.3, 1.5], [2, 1.2]],
        ],
        "a": [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.3, 0.3, 0.4],
        ],
        "gamma": 3 / 4,  # correspond to the expected adbmissibility of the sequence being 1
        "expected_lambda": 4,
        "x_test": [2, 1.2],
        "y_test": [[1.5, 0.1], [2, 1.5]],
        "expected_selection": [[1.5, 0.1], [2, 1.5]],
        "iterate_inputs": True,
    },
}


class TestConformalGenerationBaseClass(unittest.TestCase):
    def test_calibrate_and_select(self):
        for case_id, case_dict in TEST_CASES.items():
            with self.subTest(case_id=case_id):
                x = case_dict["x"]
                y = case_dict["y"]
                a = case_dict["a"]
                # create a sequence selector
                seq_selector = RunningSumSequenceSelector(score_fn=dot_product)
                # create a general calibration dataset
                calibration_ds = IndividualScoreCalibrationDataset(
                    sequence_selector=seq_selector,
                    input_dataset=x,
                    raw_generated_dataset=y,
                    admissibility_dataset=a,
                    admissibility_aggregation=sum_admissibility,
                    iterate_inputs=case_dict.get("iterate_inputs", False),
                )
                conformal_generator = ConformalGeneration(
                    sequence_selector=seq_selector,
                    calibration_dataset=calibration_ds,
                    conformal_threshold=None,
                )
                # Test calibration with a gamma value
                conformal_generator.calibrate(case_dict["gamma"], recalibrate=False)
                self.assertEqual(
                    conformal_generator.conformal_threshold, case_dict.get("expected_lambda", None)
                )
                # With calibration done, the lambda should be set within the conformal generator
                x_selected = conformal_generator.select(
                    instance=case_dict["x_test"],
                    raw_generated_sequence=case_dict["y_test"],
                    iterate_inputs=False,
                )
                self.assertEqual(x_selected, case_dict["expected_selection"])

    def test_from_score_function(self):
        for case_id, case_dict in TEST_CASES.items():
            with self.subTest(case_id=case_id):
                x = case_dict["x"]
                y = case_dict["y"]
                a = case_dict["a"]
                conformal_generator = ConformalGeneration.from_score_function(
                    input_dataset=x,
                    raw_generated_dataset=y,
                    score_fn=dot_product,
                    score_method="running_sum",
                    admissibility_dataset=a,
                    admissibility_aggregation=sum_admissibility,
                    iterate_inputs=case_dict.get("iterate_inputs", False),
                )
                conformal_generator.calibrate(case_dict["gamma"], recalibrate=False)
                self.assertEqual(
                    conformal_generator.conformal_threshold, case_dict.get("expected_lambda", None)
                )
                # With calibration done, the lambda should be set within the conformal generator
                x_selected = conformal_generator.select(
                    instance=case_dict["x_test"],
                    raw_generated_sequence=case_dict["y_test"],
                    iterate_inputs=False,
                )
                self.assertEqual(x_selected, case_dict["expected_selection"])


if __name__ == "__main__":
    unittest.main()

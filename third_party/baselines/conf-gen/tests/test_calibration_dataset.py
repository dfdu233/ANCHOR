import functools
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np

from src.calibration_dataset import (
    CacheKey,
    GeneralCalibrationDataset,
    IndividualScoreCalibrationDataset,
    ObservedDenseLambdasCalibrationDataset,
    ObservedSparseLambdasCalibrationDataset,
    _BaseCalibrationDataset,
    known_lambda_calibration,
    _validate_admissibility_dict_list,
    ManualCalibrationDataset,
)
from src.sequence_selector import (
    AboveLambdaSequenceSelector,
    RunningMaxSequenceSelector,
    RunningSumSequenceSelector,
    _BaseSequenceSelector,
    SequenceSelectorFromScore,
)

from .helper import dot_product, max_admissibility, max_admissibility_from_dot_product


class TestCalibrationDataset(unittest.TestCase):
    def test_known_lambda_calibration(self):
        test_cases: Dict[str, Dict[str, Any]] = {
            "Case 0": {
                "admissibility_on_lambda": [{float("-inf"): 0.0}, {float("-inf"): 0.0}],
                "gammas": [5],
                "expected_lambdas": [float("inf")],
                "lower_bound": 0,
            },
            "Case 1: Different values of lambdas.": {
                # possible gamma, 1.5, 5, 5.5, 7 at lambda=10, 20, 30, 40
                "admissibility_on_lambda": [
                    {float("-inf"): 0, 10: 3, 30: 4},
                    {float("-inf"): 0, 20: 7, 40: 10},
                ],
                "gammas": np.asarray([1, 2, 5, 6, 10]) * 2 / 3,  # gamma*(n+1) / n
                "expected_lambdas": [10, 20, 20, 40, float("inf")],
                "lower_bound": 0,
            },
            "Case 2: Repeated values of lambdas.": {
                # possible gamma, 1.5, 5.5, 7, 5.5 at lambda= 10, 20, 40, 50
                "admissibility_on_lambda": [
                    {float("-inf"): 0, 10: 3, 20: 4, 50: 1},
                    {float("-inf"): 0, 20: 7, 40: 10},
                ],
                "gammas": np.asarray([1, 2, 5, 6, 10]) * 2 / 3,  # gamma*(n+1) / n
                "expected_lambdas": [10, 20, 20, 40, float("inf")],
                "lower_bound": 0,
            },
            "Case 2a: Case 2 shifted": {
                # possible gamma, 1.5, 5.5, 7, 5.5 at lambda= 10, 20, 40, 50
                "admissibility_on_lambda": [
                    {float("-inf"): -1, 10: 2, 20: 3, 50: 0},
                    {float("-inf"): -1, 20: 6, 40: 9},
                ],
                "gammas": np.asarray([0, 1, 4, 5, 9]) * 2 / 3,  # gamma*(n+1) / n
                "expected_lambdas": [10, 20, 20, 40, float("inf")],
                "lower_bound": -1,
            },
            "Case 3: Repeated values of lambdas, with nonzero global minimum admissibilities": {
                # possible gamma, 2, 5.5, 7, 5.5 at lambda= 10, 20, 40, 50
                "admissibility_on_lambda": [
                    {float("-inf"): 1, 10: 3, 20: 4, 50: 1},
                    {float("-inf"): 1, 20: 7, 40: 10},
                ],
                "gammas": np.asarray([1, 2, 5, 6, 10]) * 2 / 3,  # gamma*(n+1) / n
                "expected_lambdas": [float("-inf"), 10, 20, 40, float("inf")],
                "lower_bound": 0,
                "hide_logs": True,
            },
            "Case 4: Repeated values of lambdas, with individual minimum admissibilities": {
                # possible gamma, 4/3, 10/3, 14/3, 17/3, 13/3 at lambda=10, 20, 30, 40, 50
                "admissibility_on_lambda": [
                    {float("-inf"): 2, 10: 3, 50: 1},
                    {float("-inf"): 1, 20: 7, 40: 10},
                    {float("-inf"): 0, 30: 4, 50: 2},
                ],
                "gammas": np.asarray([0, 2, 3, 5.5, 8]) * 3 / 4,  # gamma*(n+1) / n
                "expected_lambdas": [float("-inf"), 20, 20, 40, float("inf")],
                "lower_bound": 0,
            },
            "Case 5: Contains positive inf": {
                # possible gamma: 5, 6.5 at lambda=10, 40
                "admissibility_on_lambda": [
                    {float("-inf"): 0, 10: 3, float("inf"): 4},
                    {float("-inf"): 0, 10: 7, 40: 10},
                ],
                "gammas": np.asarray([4, 6, 8]) * 2 / 3,  # gamma*(n+1) / n
                "expected_lambdas": [10, 40, float("inf")],
                "lower_bound": 0,
            },
            "Case 6: Contains nan": {
                # possible gamma: 5, 6.5 at lambda=10, 40
                "admissibility_on_lambda": [
                    {float("-inf"): 0, 10: 3, float("nan"): 4},
                    {float("-inf"): 0, 10: 7, 40: 10},
                ],
                "gammas": np.asarray([4, 6, 8]) * 2 / 3,  # gamma*(n+1) / n
                "expected_lambdas": [10, 40, float("inf")],
                "lower_bound": 0,
            },
        }
        for case_id, case_dict in test_cases.items():
            admissibility_on_lambda = case_dict["admissibility_on_lambda"]
            for gamma, lambda_ in zip(case_dict["gammas"], case_dict["expected_lambdas"]):
                with self.subTest(case_id=case_id, gamma=gamma):
                    log_context = self.assertLogs if "hide_logs" in case_dict else self.assertNoLogs
                    with log_context("src.calibration_dataset"):
                        lambda_hat = known_lambda_calibration(
                            admissibility_on_lambda, gamma, case_dict["lower_bound"]
                        )
                        self.assertEqual(lambda_hat, lambda_)

    def test_validate_admissibility_dict_list(self):
        successful_test_cases = {
            "Case 0: no lower bound": {
                "lower_bound": None,
                "admissibility_dict_list": [
                    {float("-inf"): False, 5.0: True},
                    {float("-inf"): False, 6.0: False},
                ],
                "expected_log": False,
            },
            "Case 1: lower bound = 0": {
                "lower_bound": 0.0,
                "admissibility_dict_list": [
                    {float("-inf"): False, 5.0: True},
                    {float("-inf"): False, 6.0: False},
                ],
                "expected_log": False,
            },
            "Case 2: lower bound = -1": {
                "lower_bound": -1.0,
                "admissibility_dict_list": [
                    {float("-inf"): -0.5, 5.0: 2.0},
                    {float("-inf"): -1.0, 6.0: 3.0},
                ],
                "expected_log": False,
            },
            "Case 3: Non-tight no lower bound": {
                "lower_bound": None,
                "admissibility_dict_list": [
                    {float("-inf"): 0.5, 5.0: 2.0},
                    {float("-inf"): 1.0, 6.0: 3.0},
                ],
                "expected_log": True,
            },
            "Case 4: Non-tight lower bound": {
                "lower_bound": -1.0,
                "admissibility_dict_list": [
                    {float("-inf"): 0.5, 5.0: 2.0},
                    {float("-inf"): 1.0, 6.0: 3.0},
                ],
                "expected_log": True,
            },
        }

        unsuccessful_test_cases = {
            "Case 0: Under lower bound, No lower bound": {
                "lower_bound": None,
                "admissibility_dict_list": [
                    {float("-inf"): -0.5, 5.0: 2.0},
                    {float("-inf"): -1.0, 6.0: 3.0},
                ],
            },
            "Case 1: Under lower bound, lower bound": {
                "lower_bound": 1.0,
                "admissibility_dict_list": [
                    {float("-inf"): -0.5, 5.0: 2.0},
                    {float("-inf"): -1.0, 6.0: 3.0},
                ],
            },
            "Case 2: no negative inf": {
                "lower_bound": None,
                "admissibility_dict_list": [
                    {5.0: 2.0},
                    {float("-inf"): 1.0, 6.0: 3.0},
                ],
            },
        }

        for case_id, case_dict in successful_test_cases.items():
            with self.subTest(case_id=case_id):
                kwargs = (
                    {}
                    if case_dict["lower_bound"] is None
                    else {
                        "admissibility_function_lower_bound": case_dict["lower_bound"],
                    }
                )
                log_context = self.assertLogs if case_dict["expected_log"] else self.assertNoLogs
                with log_context("src.calibration_dataset"):
                    _validate_admissibility_dict_list(
                        admissibility_dict_list=case_dict["admissibility_dict_list"],
                        **kwargs,
                    )

        for case_id, case_dict in unsuccessful_test_cases.items():
            with self.subTest(case_id=case_id):
                kwargs = (
                    {}
                    if case_dict["lower_bound"] is None
                    else {
                        "admissibility_function_lower_bound": case_dict["lower_bound"],
                    }
                )
                with self.assertRaises(ValueError):
                    _validate_admissibility_dict_list(
                        admissibility_dict_list=case_dict["admissibility_dict_list"],
                        **kwargs,
                    )

    def test_validate_list(self):
        successful_test_cases = {
            "No lists": {},
            "one list": {"list_a": [2, 3, 4]},
            "two lists of same length": {
                "list_a": [2, 3],
                "list_b": ["a", "b"],
            },
            "multiple lists of same length": {
                "list_a": [2, 4],
                "list_b": ["a", "b"],
                "list_c": [True, False],
                "list_d": [[0.5, 2], [2.5, 4]],
            },
        }
        unsuccessful_test_cases = {
            "Not a list": {
                "input": {
                    "list_a": [5],
                    "list_b": "5",
                },
                "expected_error": TypeError,
                "error_list_names": ["list_b"],
            },
            "Lengths do not match": {
                "input": {
                    "list_a": [4, 5],
                    "list_b": ["a", "b"],
                    "list_c": [True, False, False],
                },
                "expected_error": ValueError,
                "error_list_names": ["list_a", "list_c"],
            },
        }
        for name, list_kwargs in successful_test_cases.items():
            with self.subTest(name=name):
                _BaseCalibrationDataset._validate_list(**list_kwargs)
        for name, list_dict in unsuccessful_test_cases.items():
            with self.subTest(name=name):
                with self.assertRaises(list_dict["expected_error"]) as cm:
                    _BaseCalibrationDataset._validate_list(**list_dict["input"])
                for error_list_name in list_dict["error_list_names"]:
                    self.assertIn(error_list_name, str(cm.exception))

    def test_validate_list_of_lists(self):
        successful_test_cases = {
            "No lists": {},
            "one list": {"list_a": [[2], [3, 4]]},
            "two lists of same length": {
                "list_a": [[2], [3, 5]],
                "list_b": [["a"], ["b", "c"]],
            },
            "multiple lists of same length": {
                "list_a": [[2], [4, 5]],
                "list_b": [["a"], ["b", "c"]],
                "list_c": [[True], [False, True]],
                "list_d": [[0.5], [2.5, 4]],
            },
        }
        unsuccessful_test_cases = {
            "Not a list": {
                "input": {
                    "list_a": [5],
                    "list_b": "5",
                },
                "expected_error": TypeError,
                "error_list_names": ["list_b"],
            },
            "Elements not a list": {
                "input": {
                    "list_a": [[5], [6, 7]],
                    "list_b": [["a"], ["b", "c"]],
                    "list_c": [True, [False, True]],
                },
                "expected_error": TypeError,
                "error_list_names": ["list_c"],
            },
            "Lengths do not match": {
                "input": {
                    "list_a": [[4], [5]],
                    "list_b": [["a"], ["b"]],
                    "list_c": [[True], [False, False]],
                },
                "expected_error": ValueError,
                "error_list_names": ["list_a", "list_c"],
            },
        }
        for name, list_kwargs in successful_test_cases.items():
            with self.subTest(name=name):
                _BaseCalibrationDataset._validate_list_of_lists(**list_kwargs)
        for name, list_dict in unsuccessful_test_cases.items():
            with self.subTest(name=name):
                with self.assertRaises(list_dict["expected_error"]) as cm:
                    _BaseCalibrationDataset._validate_list_of_lists(**list_dict["input"])
                for error_list_name in list_dict["error_list_names"]:
                    self.assertIn(error_list_name, str(cm.exception))


class ToySequenceSelector(_BaseSequenceSelector):
    def __init__(self):
        super().__init__(sequential=False, returns_subsequence=False)

    def select(self, instance, raw_generated_sequence, lambda_, iterate_inputs=False):
        if iterate_inputs:
            return [
                lambda_ / ((y - ins) ** 2 + 1.0) for ins, y in zip(instance, raw_generated_sequence)
            ]
        return [lambda_ / ((y - instance) ** 2 + 1.0) for y in raw_generated_sequence]


class TestGeneralCalibrationDataset(unittest.TestCase):
    def test_general_calibration(self):
        test_cases: Dict[str, Dict[str, Any]] = {
            "Case 0": {
                "sequence_selector": ToySequenceSelector(),
                "admissibility_function": lambda x, yhat, y: max(y),
                "input_dataset": [2.0, 1.5, 4.0],
                "raw_generated_dataset": [[1.0, 2.0, 3.0], [0.5, 2.5], [-1.0, 6.0]],
                "lower_bound": 0.0,
                # C_lambda(i) = [[lambda/2, lambda, lambda/2], [lambda/2, lambda/2], [lambda/26, lambda/5]]
                # A_lambda = mean(lambda, lambda/2, lambda/5) = 17/30*lambda.
                "gamma": 17 / 30 * 3 / 4,  # gamma*(n+1) / n
                "calibrate_kwargs": {"lambda0": 1.0},
                "expected_lambda": 1.0,
            },
            "Case 1": {
                "sequence_selector": ToySequenceSelector(),
                "admissibility_function": lambda x, yhat, y: max(y),
                "input_dataset": [2.0, 1.5, 4.0],
                "raw_generated_dataset": [[1.0, 2.0, 3.0], [0.5, 2.5], [-1.0, 6.0]],
                "lower_bound": 0.0,
                "gamma": 17 / 30 * 3 / 4,  # gamma*(n+1) / n
                "calibrate_kwargs": {"lambda0": -1.0, "bounds": ((-1.0, 2.0),)},
                "expected_lambda": 1.0,
            },
            "Case 2": {
                "sequence_selector": ToySequenceSelector(),
                "admissibility_function": lambda x, yhat, y: max(y),
                "input_dataset": [2.0, 1.5, 4.0],
                "raw_generated_dataset": [[1.0, 2.0, 3.0], [0.5, 2.5], [-1.0, 6.0]],
                "lower_bound": 0.0,
                "gamma": 17 / 30 * 3 / 4,  # gamma*(n+1) / n
                "lambda0": -1.0,
                "calibrate_kwargs": {"lambda0": -1.0, "bounds": ((-1.0, 0.0),)},
                "expected_lambda": float("inf"),
            },
            "Case 4": {
                "sequence_selector": ToySequenceSelector(),
                "admissibility_function": lambda x, yhat, y: max(y),
                "input_dataset": [2.0, 1.5, 4.0],
                "raw_generated_dataset": [[1.0, 2.0, 3.0], [0.5, 2.5], [-1.0, 6.0]],
                "lower_bound": 0.0,
                "gamma": 17 / 30 * 3 / 4,  # gamma*(n+1) / n
                "calibrate_kwargs": {
                    "lambda0": 0.5,
                    "constraints": {"type": "ineq", "fun": lambda x: x - 0.5},
                },
                "expected_lambda": 1.0,
            },
            "Case 5": {
                "sequence_selector": ToySequenceSelector(),
                "admissibility_function": lambda x, yhat, y: max(y),
                "input_dataset": [2.0, 1.5, 4.0],
                "raw_generated_dataset": [[1.0, 2.0, 3.0], [0.5, 2.5], [-1.0, 6.0]],
                "lower_bound": -1.0,
                "gamma": (17 / 30 - 1 / 3) * 3 / 4,  # gamma*(n+1) / n - lower_bound / n
                "calibrate_kwargs": {
                    "lambda0": 0.5,
                    "constraints": {"type": "ineq", "fun": lambda x: x - 0.5},
                },
                "expected_lambda": 1.0,
            },
            "Case 6: iterate inputs": {
                "sequence_selector": ToySequenceSelector(),
                "admissibility_function": lambda x, yhat, y: max(y),
                "input_dataset": [[2.0, 2.0, 4.0], [1.5, 3.5], [4.0, 8.0]],
                "raw_generated_dataset": [[1.0, 2.0, 3.0], [0.5, 2.5], [-1.0, 6.0]],
                "lower_bound": -1.0,
                "gamma": (17 / 30 - 1 / 3) * 3 / 4,  # gamma*(n+1) / n - lower_bound / n
                "calibrate_kwargs": {
                    "lambda0": 0.5,
                    "constraints": {"type": "ineq", "fun": lambda x: x - 0.5},
                },
                "expected_lambda": 1.0,
                "iterate_inputs": True,
            },
        }
        for case_id, case_dict in test_cases.items():
            general_calibration_dataset = GeneralCalibrationDataset(
                sequence_selector=case_dict["sequence_selector"],
                admissibility_function=case_dict["admissibility_function"],
                input_dataset=case_dict["input_dataset"],
                raw_generated_dataset=case_dict["raw_generated_dataset"],
                admissibility_function_lower_bound=case_dict["lower_bound"],
                iterate_inputs=case_dict.get("iterate_inputs", False),
            )
            lambda_hat = general_calibration_dataset.calibrate(
                case_dict["gamma"], **case_dict["calibrate_kwargs"]
            )
            self.assertAlmostEqual(lambda_hat, case_dict["expected_lambda"])


class FaultyRunningSumSequenceSelector(RunningSumSequenceSelector):
    """
    Faulty Sequence Selector similar to Running Sum except replacing > with >=
    """

    def select(
        self,
        instance,
        raw_generated_sequence,
        lambda_,
        precomputed_scores=None,
        iterate_inputs=False,
    ):
        current_sum = 0
        current_sequence = []
        for i, output in enumerate(raw_generated_sequence):
            current_sequence.append(output)
            score = self._get_score(instance, output, precomputed_scores, i)
            current_sum += score
            if current_sum >= lambda_:
                break
        return current_sequence


class FaultyAboveLambdaSequenceSelector(AboveLambdaSequenceSelector):
    """
    Faulty Sequence Selector similar to above lambda except replacing > with >=
    """

    def select(
        self,
        instance,
        raw_generated_sequence,
        lambda_: float,
        precomputed_scores=None,
        iterate_inputs=False,
    ):
        raw_generated_sequence = list(raw_generated_sequence)
        scores = [
            self._get_score(instance, output, precomputed_scores, i)
            for i, output in enumerate(raw_generated_sequence)
        ]
        indices = [i for i, score in enumerate(scores) if score >= lambda_]
        return [raw_generated_sequence[t] for t in indices]


class TestObservedDenseLambdaCalibrationDataset(unittest.TestCase):
    def test_constructor(self):
        successful_test_cases = {
            "standard": {
                "input_dataset": [2, 6],
                "ground_truths": [4, 1],
                "raw_generated_dataset": [[4, 1], [5, 8, 6]],
                "all_lambdas": [5, 8, 2, 5],
            }
        }
        unsuccessful_test_cases = {
            "datasets are not lists": {
                "input_dataset": [2, 3],
                "ground_truths": [4, 1],
                "raw_generated_dataset": 4,
                "all_lambdas": [2, 4],
                "error_type": TypeError,
            },
            "lengths of datasets does not match": {
                "input_dataset": [2, 4],
                "ground_truths": [4, 1],
                "raw_generated_dataset": [[4], [1], [2]],
                "all_lambdas": [2, 4],
                "error_type": ValueError,
            },
            "lengths of ground truth does not match": {
                "input_dataset": [2, 6],
                "ground_truths": [4, 1, 2],
                "raw_generated_dataset": [[4, 1], [5, 8, 6]],
                "all_lambdas": [5, 8, 2, 5],
                "error_type": ValueError,
            },
        }
        for name, test_dict in successful_test_cases.items():
            with self.subTest(name=name):
                mock_sequence_selector = MagicMock()
                dataset = ObservedDenseLambdasCalibrationDataset(
                    sequence_selector=mock_sequence_selector,
                    admissibility_function=lambda x, yhat, ys: sum(y % x for y in ys),
                    input_dataset=test_dict["input_dataset"],
                    ground_truths=test_dict["ground_truths"],
                    raw_generated_dataset=test_dict["raw_generated_dataset"],
                    all_lambdas=test_dict["all_lambdas"],
                )
                self.assertEqual(dataset._all_lambdas, [float("-inf"), 2, 5, 8])
                self.assertIs(dataset.sequence_selector, mock_sequence_selector)
        for name, test_dict in unsuccessful_test_cases.items():
            with self.subTest(name=name):
                mock_sequence_selector = MagicMock()
                with self.assertRaises(test_dict["error_type"]):
                    # noinspection PyTypeChecker
                    ObservedDenseLambdasCalibrationDataset(
                        sequence_selector=mock_sequence_selector,
                        admissibility_function=lambda x, ys: sum(y % x for y in ys),
                        input_dataset=test_dict["input_dataset"],
                        ground_truths=test_dict["ground_truths"],
                        raw_generated_dataset=test_dict["raw_generated_dataset"],
                        all_lambdas=test_dict["all_lambdas"],
                    )

    def test_get_admissibility_dict_list(self):
        test_cases = {
            "empty dataset": {
                "input_dataset": [],
                "all_lambdas": [2.0, 5.0],
                "selected_sequence": [],
                "expected": [],
            },
            "nonempty dataset": {
                "input_dataset": [5, 7],
                "all_lambdas": [2.0, 5.0],
                "selected_sequence": [
                    {
                        2.0: [5, 5],
                        5.0: [12, 11],
                    },
                    {
                        2.0: [5, 2],
                        5.0: [10, 4],
                    },
                ],
                "expected": [
                    # [5%5+5%5=0, 12%5+11%5=3]
                    {float("-inf"): 0, 2.0: 0, 5.0: 3},
                    # [5%7+2%7=7, 10%7+4%7=7]
                    {float("-inf"): 0, 2.0: 7, 5.0: 7},
                ],
            },
        }

        for name, test_dict in test_cases.items():
            with self.subTest(name=name):
                mock_sequence_selector = MagicMock()
                # raw generated dataset does not matter for this test.
                num_data = len(test_dict["input_dataset"])
                dataset = ObservedDenseLambdasCalibrationDataset(
                    sequence_selector=mock_sequence_selector,
                    admissibility_function=lambda x, yhat, ys: sum(y % x for y in ys),
                    input_dataset=test_dict["input_dataset"],
                    raw_generated_dataset=[[] for _ in range(num_data)],
                    all_lambdas=test_dict["all_lambdas"],
                )

                # noinspection PyUnresolvedReferences
                dataset._cache[CacheKey.SEQUENCE] = {
                    (i, lamb): seq
                    for i, selected in enumerate(test_dict["selected_sequence"])
                    for lamb, seq in selected.items()
                }
                admissibility_dict_list = dataset._get_admissibility_dict_list()
                np.testing.assert_array_equal(admissibility_dict_list, test_dict["expected"])

    def test_get_admissibility_dict_list_with_ground_truth(self):
        successful_test_cases = {
            "case 0": {
                "input_dataset": [5, 7],
                "ground_truths": [1, 2],
                "all_lambdas": [2.0, 5.0],
                "selected_sequence": [
                    {
                        2.0: [6, 6],
                        5.0: [13, 12],
                    },
                    {
                        2.0: [7, 4],
                        5.0: [12, 6],
                    },
                ],
                "expected": [
                    # [(6-1)%5+(6-1)%5=0, (13-1)%5+(12-1)%5=3]
                    {float("-inf"): 0, 2.0: 0, 5.0: 3},
                    # [(7-2)%7+(4-2)%7=7, (12-2)%7+(6-2)%7=7]
                    {float("-inf"): 0, 2.0: 7, 5.0: 7},
                ],
            },
        }
        unsuccessful_test_cases = {
            "No ground truths are inputted": {
                "input_dataset": [5, 7],
                "all_lambdas": [2.0, 5.0],
                "selected_sequence": [
                    {
                        2.0: [6, 6],
                        5.0: [13, 12],
                    },
                    {
                        2.0: [7, 4],
                        5.0: [12, 6],
                    },
                ],
                "error_type": TypeError,
            },
        }

        for name, test_dict in successful_test_cases.items():
            with self.subTest(name=name):
                mock_sequence_selector = MagicMock()
                # raw generated dataset does not matter for this test.
                num_data = len(test_dict["input_dataset"])
                dataset = ObservedDenseLambdasCalibrationDataset(
                    sequence_selector=mock_sequence_selector,
                    admissibility_function=lambda x, yhat, ys: sum((y - yhat) % x for y in ys),
                    input_dataset=test_dict["input_dataset"],
                    ground_truths=test_dict["ground_truths"],
                    raw_generated_dataset=[[] for _ in range(num_data)],
                    all_lambdas=test_dict["all_lambdas"],
                )

                # noinspection PyUnresolvedReferences
                dataset._cache[CacheKey.SEQUENCE] = {
                    (i, lamb): seq
                    for i, selected in enumerate(test_dict["selected_sequence"])
                    for lamb, seq in selected.items()
                }
                admissibility_dict_list = dataset._get_admissibility_dict_list()
                np.testing.assert_array_equal(admissibility_dict_list, test_dict["expected"])

        for name, test_dict in unsuccessful_test_cases.items():
            with self.subTest(name=name):
                # noinspection PyTypeChecker
                dataset = ObservedDenseLambdasCalibrationDataset(
                    sequence_selector=MagicMock(),
                    admissibility_function=lambda x, yhat, ys: sum((y - yhat) % x for y in ys),
                    input_dataset=test_dict["input_dataset"],
                    raw_generated_dataset=[[] for _ in range(num_data)],
                    all_lambdas=test_dict["all_lambdas"],
                )
                dataset._cache[CacheKey.SEQUENCE] = {
                    (i, lamb): seq
                    for i, selected in enumerate(test_dict["selected_sequence"])
                    for lamb, seq in selected.items()
                }
                with self.assertRaises(test_dict["error_type"]):
                    dataset._get_admissibility_dict_list()

    def test_cache(self):
        mock_sequence_selector = MagicMock()
        mock_sequence_selector.select.return_value = MagicMock()
        mock_admissibility_function = MagicMock()
        # noinspection PyTypeChecker
        dataset = ObservedDenseLambdasCalibrationDataset(
            sequence_selector=mock_sequence_selector,
            admissibility_function=mock_admissibility_function,
            input_dataset=[3],
            raw_generated_dataset=[[5, 6]],
            all_lambdas=[9.0],
        )

        with patch.object(mock_sequence_selector, "select") as mock_func:
            dataset._get_admissibility_dict_list()
            # call for lambda = -inf, 9
            self.assertEqual(mock_func.call_count, 2)
            self.assertEqual(mock_admissibility_function.call_count, 2)
            dataset._get_admissibility_dict_list()
            # cumulative call count does not change.
            self.assertEqual(mock_func.call_count, 2)
            self.assertEqual(mock_admissibility_function.call_count, 2)
            dataset.empty_cache()
            dataset._get_admissibility_dict_list()
            # cumulative call count adding 2
            self.assertEqual(mock_func.call_count, 4)
            self.assertEqual(mock_admissibility_function.call_count, 4)
            dataset.disable_cache()
            dataset._get_admissibility_dict_list()
            # cumulative call count adding 2
            self.assertEqual(mock_func.call_count, 6)
            self.assertEqual(mock_admissibility_function.call_count, 6)
            dataset._get_admissibility_dict_list()
            # cumulative call count adding 2 as caching is disabled
            self.assertEqual(mock_func.call_count, 8)
            self.assertEqual(mock_admissibility_function.call_count, 8)
            dataset.enable_cache()
            dataset._get_admissibility_dict_list()
            # cumulative call count adding 2
            self.assertEqual(mock_func.call_count, 10)
            self.assertEqual(mock_admissibility_function.call_count, 10)
            dataset._get_admissibility_dict_list()
            # cumulative call count does not change as caching is enabled again.
            self.assertEqual(mock_func.call_count, 10)
            self.assertEqual(mock_admissibility_function.call_count, 10)


class TestIndividualScoreCalibrationDataset(unittest.TestCase):
    def test_individual_score_calibration(self):
        test_cases: Dict[str, Dict[str, Any]] = {
            "Case 0": {
                "sequence_selector": RunningMaxSequenceSelector(score_fn=dot_product),
                "input_dataset": [],
                "raw_generated_dataset": [],
                "admissibility_dataset": [],
                "admissibility_aggregation": max_admissibility,
                "gamma": 5,
                "expected_dict_list": [],
                "expected_lambda": float("inf"),
            },
            "Case 1": {
                "sequence_selector": RunningMaxSequenceSelector(score_fn=dot_product),
                "input_dataset": [[1, 0], [0, 1]],
                "raw_generated_dataset": [[[0.5, 0], [1.5, 1]], [[0, 0], [0, 0.5], [1, 1]]],
                "admissibility_dataset": [[0.5, 1.5], [0, 0.5, 1]],
                "admissibility_aggregation": max_admissibility,
                "gamma": 0.6,  # 0.9 * 2 / 3,
                "expected_dict_list": [
                    {float("-inf"): 0.5, 0.5: 1.5, 1.5: 1.5},
                    {float("-inf"): 0, 0: 0.5, 0.5: 1, 1: 1},
                ],
                "expected_lambda": 0.5,
            },
            "Case 2": {
                "sequence_selector": RunningMaxSequenceSelector(score_fn=dot_product),
                "input_dataset": [[1, 0], [0, 1]],
                "raw_generated_dataset": [[[0, 0], [0.5, 0]], [[0, 0], [0, 0.5], [1, 1]]],
                "admissibility_dataset": [[0, 75], [0, 0.75, 1]],
                "admissibility_aggregation": max_admissibility,
                "gamma": 0.5,  # 0.75 * 2 / 3,
                "expected_dict_list": [
                    {float("-inf"): 0, 0: 75, 0.5: 75},
                    {float("-inf"): 0, 0: 0.75, 0.5: 1, 1: 1},
                ],
                "expected_lambda": 0.0,
            },
            "Case 3: iterate inputs": {
                "sequence_selector": RunningMaxSequenceSelector(score_fn=dot_product),
                "input_dataset": [[[2, 0], [1, 0]], [[0, 2], [0, 1], [0, 1]]],
                "raw_generated_dataset": [[[0, 0], [0.5, 0]], [[0, 0], [0, 0.5], [1, 1]]],
                "admissibility_dataset": [[0, 75], [0, 0.75, 1]],
                "admissibility_aggregation": max_admissibility,
                "gamma": 0.5,  # 0.75 * 2 / 3,
                "expected_dict_list": [
                    {float("-inf"): 0, 0: 75, 0.5: 75},
                    {float("-inf"): 0, 0: 0.75, 0.5: 1, 1: 1},
                ],
                "expected_lambda": 0.0,
                "iterate_inputs": True,
            },
        }
        for case_id, case_dict in test_cases.items():
            with self.subTest(case_id=case_id):
                individual_score_calibration_dataset = IndividualScoreCalibrationDataset(
                    sequence_selector=case_dict["sequence_selector"],
                    input_dataset=case_dict["input_dataset"],
                    raw_generated_dataset=case_dict["raw_generated_dataset"],
                    admissibility_dataset=case_dict["admissibility_dataset"],
                    admissibility_aggregation=case_dict["admissibility_aggregation"],
                    iterate_inputs=case_dict.get("iterate_inputs", False),
                )
                lambda_hat = individual_score_calibration_dataset.calibrate(case_dict["gamma"])
                self.assertEqual(lambda_hat, case_dict["expected_lambda"])
                admissibility_dict_list = (
                    individual_score_calibration_dataset._get_admissibility_dict_list()
                )
                self.assertEqual(admissibility_dict_list, case_dict["expected_dict_list"])

    def test_validation_for_non_decreasing_admissibility_function(self):
        test_cases = {
            "Case 0: Running Max with max admissibility": {
                "sequence_selector": RunningMaxSequenceSelector(score_fn=dot_product),
                "admissibility_aggregation": max,
                "expected_valid": True,
            },
            "Case 1: Running Max with min admissibility": {
                "sequence_selector": RunningMaxSequenceSelector(score_fn=dot_product),
                "admissibility_aggregation": min,
                "expected_valid": False,
            },
            "Case 2: Above Lambda with max admissibility": {
                "sequence_selector": AboveLambdaSequenceSelector(score_fn=dot_product),
                "admissibility_aggregation": lambda x: max(x) if len(x) > 0 else 0.0,
                "expected_valid": False,
            },
            "Case 3: Above Lambda with min admissibility": {
                "sequence_selector": AboveLambdaSequenceSelector(score_fn=dot_product),
                "admissibility_aggregation": lambda x: min(x) if len(x) > 0 else float("inf"),
                "expected_valid": True,
            },
        }
        for case_id, case_dict in test_cases.items():
            with self.subTest(case_id=case_id):
                calibration_dataset = IndividualScoreCalibrationDataset(
                    sequence_selector=case_dict["sequence_selector"],
                    input_dataset=[[1, 0], [0, 1]],
                    raw_generated_dataset=[[[0.5, 0], [1.5, 1]], [[0, 0], [0, 0.5], [1, 1]]],
                    admissibility_dataset=[[0.5, 1.5], [1, 0.5, 0]],
                    admissibility_aggregation=case_dict["admissibility_aggregation"],
                )
                expected_valid = case_dict["expected_valid"]
                if expected_valid:
                    valid = calibration_dataset.validate_admissibility_function_non_decreasing()
                    self.assertTrue(valid)
                else:
                    with self.assertLogs(IndividualScoreCalibrationDataset.__name__):
                        valid = calibration_dataset.validate_admissibility_function_non_decreasing()
                        self.assertFalse(valid)

    def test_validation_for_non_decreasing_average_admissibility_function(self):
        test_cases = {
            "Case 0: Running Max with max admissibility": {
                "sequence_selector": RunningMaxSequenceSelector(score_fn=dot_product),
                "admissibility_aggregation": max,
                "expected_valid": True,
            },
            "Case 1: Running Max with min admissibility": {
                "sequence_selector": RunningMaxSequenceSelector(score_fn=dot_product),
                "admissibility_aggregation": min,
                "expected_valid": False,
            },
            "Case 2: Above Lambda with max admissibility": {
                "sequence_selector": AboveLambdaSequenceSelector(score_fn=dot_product),
                "admissibility_aggregation": lambda x: max(x) if len(x) > 0 else 0.0,
                "expected_valid": False,
            },
            "Case 3: Above Lambda with min admissibility": {
                "sequence_selector": AboveLambdaSequenceSelector(score_fn=dot_product),
                "admissibility_aggregation": lambda x: min(x) if len(x) > 0 else float("inf"),
                "expected_valid": True,
            },
        }
        for case_id, case_dict in test_cases.items():
            with self.subTest(case_id=case_id):
                dataset = IndividualScoreCalibrationDataset(
                    sequence_selector=case_dict["sequence_selector"],
                    input_dataset=[[1, 0], [0, 1]],
                    raw_generated_dataset=[[[0.5, 0], [1.5, 1]], [[0, 0], [0, 0.5], [1, 1]]],
                    admissibility_dataset=[[0.5, 1.5], [1, 0.5, 0]],
                    admissibility_aggregation=case_dict["admissibility_aggregation"],
                )
                expected_valid = case_dict["expected_valid"]
                log_context_class = self.assertNoLogs if expected_valid else self.assertLogs
                with log_context_class(IndividualScoreCalibrationDataset.__name__):
                    valid = dataset.validate_average_admissibility_function_non_decreasing()
                    self.assertEqual(valid, expected_valid)

    def test_validation_for_right_continuous_admissibility_function(self):
        test_cases = {
            "Case 0: Running Sum sequence selector": {
                "sequence_selector": RunningSumSequenceSelector(score_fn=dot_product),
                "expected_valid": True,
            },
            "Case 1: Faulty Running Sum sequence selector": {
                "sequence_selector": FaultyRunningSumSequenceSelector(score_fn=dot_product),
                "expected_valid": False,
            },
            "Case 2: Above Lambda sequence selector": {
                "sequence_selector": AboveLambdaSequenceSelector(score_fn=dot_product),
                "expected_valid": True,
            },
            "Case 3: Above Lambda sequence selector": {
                "sequence_selector": FaultyAboveLambdaSequenceSelector(score_fn=dot_product),
                "expected_valid": False,
            },
        }
        for case_id, case_dict in test_cases.items():
            with self.subTest(case_id=case_id):
                calibration_dataset = IndividualScoreCalibrationDataset(
                    sequence_selector=case_dict["sequence_selector"],
                    input_dataset=[[1, 0], [0, 1]],
                    raw_generated_dataset=[[[0.5, 0], [1.5, 1]], [[0, 0], [0, 0.5], [1, 1]]],
                    admissibility_dataset=[[0.5, 1.5], [1, 0.5, 0]],
                    admissibility_aggregation=lambda x: max(x) if len(x) > 0 else 0.0,
                )
                expected_valid = case_dict["expected_valid"]
                log_context_class = self.assertNoLogs if expected_valid else self.assertLogs
                with log_context_class(IndividualScoreCalibrationDataset.__name__):
                    valid = calibration_dataset.validate_admissibility_function_right_continuous()
                    self.assertEqual(valid, expected_valid)


class TestObservedSparseLambdasCalibrationDataset(unittest.TestCase):
    def test_observed_sparse_lambdas_calibration(self):
        test_cases: Dict[str, Dict[str, Any]] = {
            "Case 0": {
                "sequence_selector": RunningMaxSequenceSelector(score_fn=dot_product),
                "input_dataset": [],
                "raw_generated_dataset": [],
                "possible_lambdas": [],
                "gamma": 5,
                "expected_dict_list": [],
                "expected_lambda": float("inf"),
            },
            "Case 1": {
                "sequence_selector": RunningMaxSequenceSelector(score_fn=dot_product),
                "input_dataset": [[1, 0], [0, 1]],
                "raw_generated_dataset": [[[0, 0], [1, 0]], [[0, 0], [0, 0.5], [1, 1]]],
                "possible_lambdas": [[0, 1], [0, 0.5, 1]],
                "gamma": 0.5,  # 0.75 * 2 / 3,
                "expected_dict_list": [
                    {float("-inf"): 0, 0: 1, 1: 1},
                    {float("-inf"): 0, 0: 0.5, 0.5: 1, 1: 1},
                ],
                "expected_lambda": 0,
            },
            "Case 2": {
                "sequence_selector": RunningMaxSequenceSelector(score_fn=dot_product),
                "input_dataset": [[1, 0], [0, 1]],
                "raw_generated_dataset": [[[0, 0], [1, 0]], [[0, 0], [0, 0.5], [1, 1]]],
                "possible_lambdas": [[0, 1], [0, 0.5, 1]],
                "gamma": 0.6,  # 0.9 * 2 / 3,
                "expected_dict_list": [
                    {float("-inf"): 0, 0: 1, 1: 1},
                    {float("-inf"): 0, 0: 0.5, 0.5: 1, 1: 1},
                ],
                "expected_lambda": 0.5,
            },
            "Case 3: Iterate inputs": {
                "sequence_selector": RunningMaxSequenceSelector(score_fn=dot_product),
                "input_dataset": [[[2, 0], [1, 0]], [[0, 2], [0.5, 0.5], [0, 1]]],
                "raw_generated_dataset": [[[0, 0], [1, 1]], [[0, 0], [1, 0], [2, 1]]],
                "possible_lambdas": [[0, 1], [0, 0.5, 1]],
                "gamma": 0.6,  # 0.9 * 2 / 3,
                "expected_dict_list": [
                    {float("-inf"): 0, 0: 1, 1: 1},
                    {float("-inf"): 0, 0: 0.5, 0.5: 1, 1: 1},
                ],
                "expected_lambda": 0.5,
                "iterate_inputs": True,
            },
        }
        for case_id, case_dict in test_cases.items():
            with self.subTest(case_id=case_id):
                iterate_inputs = case_dict.get("iterate_inputs", False)
                calibration_dataset = ObservedSparseLambdasCalibrationDataset(
                    sequence_selector=case_dict["sequence_selector"],
                    input_dataset=case_dict["input_dataset"],
                    raw_generated_dataset=case_dict["raw_generated_dataset"],
                    admissibility_function=functools.partial(
                        max_admissibility_from_dot_product, iterate_inputs=iterate_inputs
                    ),
                    possible_lambdas=case_dict["possible_lambdas"],
                    iterate_inputs=iterate_inputs,
                )
                lambda_hat = calibration_dataset.calibrate(case_dict["gamma"])
                self.assertEqual(lambda_hat, case_dict["expected_lambda"])
                admissibility_dict_list = calibration_dataset._get_admissibility_dict_list()
                self.assertEqual(admissibility_dict_list, case_dict["expected_dict_list"])


class TestManualCalibrationDataset(unittest.TestCase):
    def test_expected_dict_dict(self):
        test_cases: Dict[str, Dict[str, Any]] = {
            "Case 0": {
                "sequence_selector": MagicMock(),
                "input_dataset": [[1, 0], [0, 1]],
                "raw_generated_dataset": [[[0, 0]], [[0, 0], [1, 1]]],
                "scores": [8, 7, 6],
                "selected_sequence": [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
                "expected_dict_dict": {
                    "instance_index_0_lambda_index_0": {
                        "lambda": float("-inf"),
                        "instance_index": 0,
                        "instance": [1, 0],
                        "raw_output": [[0, 0]],
                        "selected_sequence": [1, 2],
                    },
                    "instance_index_0_lambda_index_1": {
                        "lambda": 8,
                        "instance_index": 0,
                        "instance": [1, 0],
                        "raw_output": [[0, 0]],
                        "selected_sequence": [3, 4],
                    },
                    "instance_index_1_lambda_index_0": {
                        "lambda": float("-inf"),
                        "instance_index": 1,
                        "instance": [0, 1],
                        "raw_output": [[0, 0], [1, 1]],
                        "selected_sequence": [5, 6],
                    },
                    "instance_index_1_lambda_index_1": {
                        "lambda": 6,
                        "instance_index": 1,
                        "instance": [0, 1],
                        "raw_output": [[0, 0], [1, 1]],
                        "selected_sequence": [7, 8],
                    },
                    "instance_index_1_lambda_index_2": {
                        "lambda": 7,
                        "instance_index": 1,
                        "instance": [0, 1],
                        "raw_output": [[0, 0], [1, 1]],
                        "selected_sequence": [9, 10],
                    },
                },
            },
        }
        for case_id, case_dict in test_cases.items():
            with self.subTest(case_id=case_id):
                input_dataset = case_dict["input_dataset"]
                selected_sequence = case_dict["selected_sequence"]
                sequence_selector = case_dict["sequence_selector"]
                sequence_selector.select.side_effect = selected_sequence
                sequence_selector.score_fn.side_effect = case_dict["scores"]
                sequence_selector.iterate.side_effect = SequenceSelectorFromScore.iterate
                sequence_selector.get_possible_lambdas = lambda x: sorted(list(set(x)))
                calibration_dataset = ManualCalibrationDataset(
                    sequence_selector=sequence_selector,
                    input_dataset=input_dataset,
                    raw_generated_dataset=case_dict["raw_generated_dataset"],
                )
                admissibility = calibration_dataset.get_dict_for_evaluating_admissibilities()
                self.assertEqual(admissibility, case_dict["expected_dict_dict"])

    def test_input_admissibilities(self):
        test_cases: Dict[str, Dict[str, Any]] = {
            "Case 0": {
                "expected_dict_dict": {
                    "instance_index_0_lambda_index_0": {
                        "lambda": float("-inf"),
                        "instance_index": 0,
                        "instance": [1, 0],
                        "raw_output": [[0, 0]],
                        "selected_sequence": [1, 2],
                    },
                    "instance_index_0_lambda_index_1": {
                        "lambda": 8,
                        "instance_index": 0,
                        "instance": [1, 0],
                        "raw_output": [[0, 0]],
                        "selected_sequence": [3, 4],
                    },
                    "instance_index_1_lambda_index_0": {
                        "lambda": float("-inf"),
                        "instance_index": 1,
                        "instance": [0, 1],
                        "raw_output": [[0, 0], [1, 1]],
                        "selected_sequence": [5, 6],
                    },
                    "instance_index_1_lambda_index_1": {
                        "lambda": 6,
                        "instance_index": 1,
                        "instance": [0, 1],
                        "raw_output": [[0, 0], [1, 1]],
                        "selected_sequence": [7, 8],
                    },
                    "instance_index_1_lambda_index_2": {
                        "lambda": 7,
                        "instance_index": 1,
                        "instance": [0, 1],
                        "raw_output": [[0, 0], [1, 1]],
                        "selected_sequence": [9, 10],
                    },
                },
                "manual_admissibilities": {
                    "instance_index_0_lambda_index_0": 2,
                    "instance_index_0_lambda_index_1": 3,
                    "instance_index_1_lambda_index_0": 5,
                    "instance_index_1_lambda_index_1": 0,
                    "instance_index_1_lambda_index_2": 1,
                },
                "expected_lambda_dict": {
                    0: {float("-inf"): 2, 8.0: 3},
                    1: {float("-inf"): 5, 6.0: 0, 7.0: 1},
                },
            },
        }
        for case_id, case_dict in test_cases.items():
            with self.subTest(case_id=case_id):
                calibration_dataset = ManualCalibrationDataset(
                    sequence_selector=MagicMock(),
                    input_dataset=MagicMock(),
                    raw_generated_dataset=MagicMock(),
                )
                calibration_dataset._dict_for_evaluation = case_dict["expected_dict_dict"]
                calibration_dataset.register_admissibilities(
                    case_dict["manual_admissibilities"], reregister=False, update=False
                )
                self.assertEqual(
                    calibration_dataset._admissibility_dict_dict, case_dict["expected_lambda_dict"]
                )

    def test_calibrate(self):
        test_cases: Dict[str, Dict[str, Any]] = {
            "Case 0": {
                "lambda_dict": {
                    "0": {float("-inf"): 2, 8: 3},
                    "1": {float("-inf"): 5, 6: 0, 7: 1},
                },
                "gamma": 5,
                "lower_bound": 3,
            },
        }
        for case_id, case_dict in test_cases.items():
            with self.subTest(case_id=case_id):
                calibration_dataset = ManualCalibrationDataset(
                    sequence_selector=MagicMock(),
                    input_dataset=MagicMock(),
                    raw_generated_dataset=MagicMock(),
                    admissibility_function_lower_bound=case_dict["lower_bound"],
                )
                calibration_dataset._admissibility_dict_dict = case_dict["lambda_dict"]
                gamma = case_dict["gamma"]
                with patch("src.calibration_dataset.known_lambda_calibration") as mock_calibration:
                    calibration_dataset.calibrate(gamma=gamma)
                    mock_calibration.assert_called_with(
                        admissibility_on_lambda=list(case_dict["lambda_dict"].values()),
                        gamma=gamma,
                        admissibility_function_lower_bound=case_dict["lower_bound"],
                    )

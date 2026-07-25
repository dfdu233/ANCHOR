from unittest.mock import MagicMock
import unittest
from src.sequence_selector import (
    RunningMaxSequenceSelector,
    RunningSumSequenceSelector,
    RunningSmallestSubsetSumSequenceSelector,
    RunningMaxSingleSequenceSelector,
    SimpleMovingAverageSequenceSelector,
    ExponentialMovingAverageSequenceSelector,
    RunningCharacteristicSequenceSelector,
    SmallestSubsetSumSequenceSelector,
    AboveLambdaSequenceSelector,
    PostprocessedSequenceSelector,
)
from .helper import dot_product, ema


class ToyIterator:
    def __init__(self, y):
        self.y = y
        self.cur = 0

    def __next__(self):
        if self.cur >= len(self.y):
            raise StopIteration
        item = self.y[self.cur]
        self.cur += 1
        return item

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.y)


TEST_CASES = {
    "Case 0": {
        "x": [],
        "y": [],
        # scores = []
    },
    "Case 1": {
        "x": [2, 3],
        "y": [[0, 1], [1, 0], [1, 1]],
        # scores = [3, 2, 5]
    },
    "Case 2": {
        "x": [3, 4],
        "y": [[1, 0], [0, 1], [1, 1]],
        # scores = [3, 4, 7]
    },
    "Case 3": {
        "x": [3, 3],
        "y": [[1, 0], [0, 1], [1, 1]],
        # scores = [3, 3, 6]
    },
    "Case 4": {
        "x": [1, 0],
        "y": [[0, 0], [0, 1], [1, 1]],
        # scores = [1, 0, 1]
    },
}

TEST_CASES_MULTI_X = {
    "Case 0": {
        "x": [],
        "y": [],
    },
    "Case 1": {
        "x": [[2, 3], [1, 4], [0, 5]],
        "y": [[0, 1], [1, 0], [1, 1]],
    },
    "Case 2": {
        "x": [[3, 4], [2, 5], [6, 1]],
        "y": [[1, 0], [0, 1], [1, 1]],
    },
    "Case 3": {
        "x": [[3, 3], [4, 4], [6, 0]],
        "y": [[1, 0], [0, 1], [1, 1]],
    },
    "Case 4": {
        "x": [[1, 0], [0, 2], [3, 1]],
        "y": [[0, 0], [0, 1], [1, 1]],
    },
}


class TestSequenceSelectorBaseClass(unittest.TestCase):
    def setUp(self):
        self.selector = None

    def _test_sequence_selection_fn(self, test_cases_expected_outputs, multi_x=False):
        for case_id in TEST_CASES.keys():
            with self.subTest(case_id=case_id):
                if not multi_x:
                    x = TEST_CASES[case_id]["x"]
                    y = TEST_CASES[case_id]["y"]
                else:
                    x = TEST_CASES_MULTI_X[case_id]["x"]
                    y = TEST_CASES_MULTI_X[case_id]["y"]
                lambda_val = test_cases_expected_outputs[case_id]["lambda_val"]
                expected = test_cases_expected_outputs[case_id]["expected"]

                subseq_selected = self.selector.select(x, y, lambda_val, iterate_inputs=multi_x)
                self.assertEqual(subseq_selected, expected)
                y = ToyIterator(y)
                subseq_selected_iter = self.selector.select(
                    x, y, lambda_val, iterate_inputs=multi_x
                )
                expected_iter_calls = test_cases_expected_outputs[case_id]["expected_iter_calls"]
                self.assertEqual(subseq_selected_iter, expected)
                self.assertEqual(expected_iter_calls, y.cur)


class TestRunningMaxSequenceSelector(TestSequenceSelectorBaseClass):
    def setUp(self):
        self.selector = RunningMaxSequenceSelector(score_fn=dot_product)

    def test_sequence_selection_fn(self):
        test_cases_expected_outputs = {
            "Case 0": {
                "lambda_val": 0,
                "expected": [],
                "expected_iter_calls": 0,
            },
            "Case 1": {
                "lambda_val": 0,
                "expected": [[0, 1]],
                "expected_iter_calls": 1,
            },
            "Case 2": {
                "lambda_val": 3.5,
                "expected": [[1, 0], [0, 1]],
                "expected_iter_calls": 2,
            },
            "Case 3": {
                "lambda_val": 5,
                "expected": [[1, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
            },
            "Case 4": {
                "lambda_val": 10,
                "expected": [[0, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
            },
        }
        super()._test_sequence_selection_fn(test_cases_expected_outputs)

    def test_get_possible_lambdas(self):
        test_cases = {
            "Base case": {
                "input": [3, 6, 2, 1, 3, 5],
                "expected": [1, 2, 3, 5, 6],
            },
        }
        for name, test_dict in test_cases.items():
            with self.subTest(name=name):
                possible_lambdas = self.selector.get_possible_lambdas(test_dict["input"])
                self.assertEqual(possible_lambdas, test_dict["expected"])


class TestRunningMaxMultiXSequenceSelector(TestSequenceSelectorBaseClass):
    def setUp(self):
        self.selector = RunningMaxSequenceSelector(score_fn=dot_product)

    def test_sequence_selection_fn(self):
        test_cases_expected_outputs = {
            "Case 0": {
                "lambda_val": 0,
                "expected": [],
                "expected_iter_calls": 0,
            },
            "Case 1": {
                "lambda_val": 0,
                "expected": [[0, 1]],
                "expected_iter_calls": 1,
            },
            "Case 2": {
                "lambda_val": 3.5,
                "expected": [[1, 0], [0, 1]],
                "expected_iter_calls": 2,
            },
            "Case 3": {
                "lambda_val": 5,
                "expected": [[1, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
            },
            "Case 4": {
                "lambda_val": 10,
                "expected": [[0, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
            },
        }
        super()._test_sequence_selection_fn(test_cases_expected_outputs, multi_x=True)

    def test_get_possible_lambdas(self):
        test_cases = {
            "Base case": {
                "input": [3, 6, 2, 1, 3, 5],
                "expected": [1, 2, 3, 5, 6],
            },
        }
        for name, test_dict in test_cases.items():
            with self.subTest(name=name):
                possible_lambdas = self.selector.get_possible_lambdas(test_dict["input"])
                self.assertEqual(possible_lambdas, test_dict["expected"])


class TestRunningSumSequenceSelector(TestSequenceSelectorBaseClass):
    def setUp(self):
        self.selector = RunningSumSequenceSelector(score_fn=dot_product)

    def test_get_possible_lambdas(self):
        test_cases = {
            "Base case": {
                "input": [3, 6, 2, 1, 3, 5],
                "expected": [3, 9, 11, 12, 15, 20],
            },
            "Zero scores": {
                "input": [3, 6, 2, 0, 2, 5],
                "expected": [3, 9, 11, 13, 18],
            },
            "Negative scores": {
                "input": [7, -3, -2, 5, 8, -2],
                "expected": [2, 4, 7, 13, 15],
            },
        }
        for name, test_dict in test_cases.items():
            with self.subTest(name=name):
                possible_lambdas = self.selector.get_possible_lambdas(test_dict["input"])
                self.assertEqual(possible_lambdas, test_dict["expected"])

    def test_sequence_selection_fn(self):
        test_cases_expected_outputs = {
            "Case 0": {
                "lambda_val": 0,
                "expected": [],
                "expected_iter_calls": 0,
            },
            "Case 1": {
                "lambda_val": 0,
                "expected": [[0, 1]],
                "expected_iter_calls": 1,
            },
            "Case 2": {
                "lambda_val": 5,
                "expected": [[1, 0], [0, 1]],
                "expected_iter_calls": 2,
            },
            "Case 3": {
                "lambda_val": 7,
                "expected": [[1, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
            },
            "Case 4": {
                "lambda_val": 10,
                "expected": [[0, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
            },
        }
        super()._test_sequence_selection_fn(test_cases_expected_outputs)


class TestRunningSmallestSubsetSumSequenceSelector(TestSequenceSelectorBaseClass):
    def setUp(self):
        self.selector = RunningSmallestSubsetSumSequenceSelector(score_fn=dot_product)

    def test_get_possible_lambdas(self):
        test_cases = {
            "Base case": {
                "input": [3, 6, 2, 1, 3, 5],
                "expected": [3, 9, 11, 12, 15, 20],
            },
            "Zero scores": {
                "input": [3, 6, 2, 0, 2, 5],
                "expected": [3, 9, 11, 13, 18],
            },
            "Negative scores": {
                "input": [7, -3, -2, 5, 8, -2],
                "expected": [2, 4, 7, 13, 15],
            },
        }
        for name, test_dict in test_cases.items():
            with self.subTest(name=name):
                possible_lambdas = self.selector.get_possible_lambdas(test_dict["input"])
                self.assertEqual(possible_lambdas, test_dict["expected"])

    def test_sequence_selection_fn(self):
        test_cases_expected_outputs = {
            "Case 0": {
                "lambda_val": 0,
                "expected": [],
                "expected_iter_calls": 0,
            },
            "Case 1": {
                "lambda_val": 6,
                # here should select [0,1] instead of [1,0] from y, based on sorting
                "expected": [[0, 1], [1, 1]],
                "expected_iter_calls": 3,
            },
            "Case 2": {
                "lambda_val": 11,
                # scores = {3, 4, 7} so 4 + 7 not > 11, i.e. {4, 7} is not enough.
                "expected": [[1, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
            },
            "Case 3": {
                "lambda_val": 7,
                # tie breaker, with [1,0] returned instead of [0,1] due to first
                # index comes first in sorting on tie
                "expected": [[1, 0], [1, 1]],
                "expected_iter_calls": 3,
            },
            "Case 4": {
                "lambda_val": 10,
                "expected": [[0, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
            },
        }
        return super()._test_sequence_selection_fn(test_cases_expected_outputs)


class TestRunningMaxSingleSequenceSelector(TestSequenceSelectorBaseClass):
    def setUp(self):
        self.selector = RunningMaxSingleSequenceSelector(score_fn=dot_product)

    def test_get_possible_lambdas(self):
        test_cases = {
            "Base case": {
                "input": [3, 6, 2, 1, 3, 5],
                "expected": [1, 2, 3, 5, 6],
            },
        }
        for name, test_dict in test_cases.items():
            with self.subTest(name=name):
                possible_lambdas = self.selector.get_possible_lambdas(test_dict["input"])
                self.assertEqual(possible_lambdas, test_dict["expected"])

    def test_sequence_selection_fn(self):
        test_cases_expected_outputs = {
            "Case 0": {
                "lambda_val": 0,
                "expected": [],
                "expected_iter_calls": 0,
            },
            "Case 1": {
                "lambda_val": 0,
                "expected": [[0, 1]],
                "expected_iter_calls": 1,
            },
            "Case 2": {
                "lambda_val": 3.5,
                "expected": [[0, 1]],
                "expected_iter_calls": 2,
            },
            "Case 3": {
                "lambda_val": 5,
                "expected": [[1, 1]],
                "expected_iter_calls": 3,
            },
            "Case 4": {
                "lambda_val": 10,
                "expected": [[1, 1]],
                "expected_iter_calls": 3,
            },
        }
        super()._test_sequence_selection_fn(test_cases_expected_outputs)


class TestSimpleMovingAverageSequenceSelector(TestSequenceSelectorBaseClass):
    def setUp(self):
        self.selector = SimpleMovingAverageSequenceSelector(score_fn=dot_product)

    def test_get_possible_lambdas(self):
        test_cases = {
            "Base case": {
                "input": [1, 2, 3, 4, 5],
                "expected": [1, 1.5, 2, 2.5, 3],
                "window_size": None,
            },
            "Win size 2": {
                "input": [1, 2, 3, 4, 5],
                "expected": [1, 1.5, 2.5, 3.5, 4.5],
                "window_size": 2,
            },
            "Win size 3": {
                "input": [1, 2, 3, 4, 5],
                "expected": [1, 1.5, 2, 3, 4],
                "window_size": 3,
            },
        }
        for name, test_dict in test_cases.items():
            with self.subTest(name=name):
                self.selector._window_size = test_dict["window_size"]
                possible_lambdas = self.selector.get_possible_lambdas(test_dict["input"])
                self.assertEqual(possible_lambdas, test_dict["expected"])

    def test_sequence_selection_fn(self):
        test_cases_expected_outputs = {
            "Case 0": {
                "lambda_val": 0,
                "expected": [],
                "expected_iter_calls": 0,
                # sma = []
            },
            "Case 1": {
                "lambda_val": 0,
                "expected": [[0, 1]],
                "expected_iter_calls": 1,
                # sma = [2, 2.5, 10 / 3]
            },
            "Case 2": {
                "lambda_val": 3,
                "expected": [[1, 0], [0, 1]],
                "expected_iter_calls": 2,
                # sma = [3, 3.5, 14 / 3]
            },
            "Case 3": {
                "lambda_val": 5,
                "expected": [[1, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
                # sma = [3, 3, 4]
            },
            "Case 4": {
                "lambda_val": 10,
                "expected": [[0, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
                # sma = [1, 0.5, 2 / 3]
            },
        }
        super()._test_sequence_selection_fn(test_cases_expected_outputs)


class TestExponentialMovingAverageSequenceSelector(TestSequenceSelectorBaseClass):
    def setUp(self):
        self.selector = ExponentialMovingAverageSequenceSelector(
            score_fn=dot_product,
        )
    
    def test_get_possible_lambdas(self):
        test_cases = {
            "Base case": {
                "input": [1, 1, 1, 1, 1],
                "expected": [1],
            },
            "Increasing scores": {
                "input": [1, 2, 3, 4, 5],
                "expected": sorted([1, 1.5, 2.25, 3.125, 4.0625]),
            },
            "Decreasing scores": {
                "input": [5, 4, 3, 2, 1],
                "expected": sorted([5, 4.5, 3.75, 2.875, 1.9375]),
            },
        }
        for name, test_dict in test_cases.items():
            with self.subTest(name=name):
                possible_lambdas = self.selector.get_possible_lambdas(test_dict["input"])
                self.assertEqual(possible_lambdas, test_dict["expected"])

    def test_sequence_selection_fn(self):
        test_cases_expected_outputs = {
            "Case 0": {
                "lambda_val": 0,
                "expected": [],
                "expected_iter_calls": 0,
                # ema = []
            },
            "Case 1": {
                "lambda_val": 0,
                "expected": [[0, 1]],
                "expected_iter_calls": 1,
                # ema = [3, 2.5, 3.75]
            },
            "Case 2": {
                "lambda_val": 3.25,
                "expected": [[1, 0], [0, 1]],
                "expected_iter_calls": 2,
                # ema = [3, 3.5, 5.25]
            },
            "Case 3": {
                "lambda_val": 4,
                "expected": [[1, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
                # ema = [3, 3, 4.5]
            },
            "Case 4": {
                "lambda_val": 10,
                "expected": [[0, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
                # ema = [1, 0.5, 0.75]
            },
        }
        super()._test_sequence_selection_fn(test_cases_expected_outputs)


class TestRunningCharacteristicSequenceSelector(TestSequenceSelectorBaseClass):
    def setUp(self):
        self.selector = RunningCharacteristicSequenceSelector(
            score_fn=dot_product,
            characteristic_function=ema,
        )

    def test_get_possible_lambdas(self):
        test_cases = {
            "Base case": {
                "input": [1, 1, 1, 1, 1],
                "expected": [1],
            },
            "Increasing scores": {
                "input": [1, 2, 3, 4, 5],
                "expected": sorted([1, 1.5, 2.25, 3.125, 4.0625]),
            },
            "Decreasing scores": {
                "input": [5, 4, 3, 2, 1],
                "expected": sorted([5, 4.5, 3.75, 2.875, 1.9375]),
            },
        }
        for name, test_dict in test_cases.items():
            with self.subTest(name=name):
                possible_lambdas = self.selector.get_possible_lambdas(test_dict["input"])
                self.assertEqual(possible_lambdas, test_dict["expected"])

    def test_sequence_selection_fn(self):
        test_cases_expected_outputs = {
            "Case 0": {
                "lambda_val": 0,
                "expected": [],
                "expected_iter_calls": 0,
                # ema = []
            },
            "Case 1": {
                "lambda_val": 0,
                "expected": [[0, 1]],
                "expected_iter_calls": 1,
                # ema = [3, 2.5, 3.75]
            },
            "Case 2": {
                "lambda_val": 3.25,
                "expected": [[1, 0], [0, 1]],
                "expected_iter_calls": 2,
                # ema = [3, 3.5, 5.25]
            },
            "Case 3": {
                "lambda_val": 4,
                "expected": [[1, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
                # ema = [3, 3, 4.5]
            },
            "Case 4": {
                "lambda_val": 10,
                "expected": [[0, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
                # ema = [1, 0.5, 0.75]
            },
        }
        super()._test_sequence_selection_fn(test_cases_expected_outputs)


class TestSmallestSubsetSumSequenceSelector(TestSequenceSelectorBaseClass):
    def setUp(self):
        self.selector = SmallestSubsetSumSequenceSelector(score_fn=dot_product)

    def test_get_possible_lambdas(self):
        test_cases = {
            "Base case": {
                "input": [3, 6, 2, 1, 3, 5],
                "expected": [6, 11, 14, 17, 19, 20],
            },
            "Zero scores": {
                "input": [3, 6, 2, 0, 2, 5],
                "expected": [6, 11, 14, 16, 18],
            },
            "Negative scores": {
                "input": [7, -3, -2, 5, 8, -2],
                "expected": [8, 13, 15, 16, 18, 20],
            },
        }
        for name, test_dict in test_cases.items():
            with self.subTest(name=name):
                possible_lambdas = self.selector.get_possible_lambdas(test_dict["input"])
                self.assertEqual(possible_lambdas, test_dict["expected"])

    def test_sequence_selection_fn(self):
        test_cases_expected_outputs = {
            "Case 0": {
                "lambda_val": 0,
                "expected": [],
                "expected_iter_calls": 0,
            },
            "Case 1": {
                "lambda_val": 6,
                # tiebreaker, select [0,1] since its score is higher than that for [1,0]
                "expected": [[0, 1], [1, 1]],
                "expected_iter_calls": 3,
            },
            "Case 2": {
                "lambda_val": 11,
                "expected": [[1, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
            },
            "Case 3": {
                "lambda_val": 7,
                # tiebreaker, select [1,0] since it comes first before [0,1]
                "expected": [[1, 0], [1, 1]],
                "expected_iter_calls": 3,
            },
            "Case 4": {
                "lambda_val": 10,
                "expected": [[0, 0], [0, 1], [1, 1]],
                "expected_iter_calls": 3,
            },
        }
        super()._test_sequence_selection_fn(test_cases_expected_outputs)


class TestAboveLambdaSequenceSelector(TestSequenceSelectorBaseClass):
    def setUp(self):
        self.selector = AboveLambdaSequenceSelector(score_fn=dot_product)

    def test_get_possible_lambdas(self):
        test_cases = {
            "Base case": {
                "input": [3, 6, 2, 1, 3, 5],
                "expected": [1, 2, 3, 5, 6],
            },
        }
        for name, test_dict in test_cases.items():
            with self.subTest(name=name):
                possible_lambdas = self.selector.get_possible_lambdas(test_dict["input"])
                self.assertEqual(possible_lambdas, test_dict["expected"])

    def test_sequence_selection_fn(self):
        test_cases_expected_outputs = {
            "Case 0": {
                "lambda_val": 0,
                "expected": [],
                "expected_iter_calls": 0,
            },
            "Case 1": {
                "lambda_val": 2.5,
                "expected": [[0, 1], [1, 1]],
                "expected_iter_calls": 3,
            },
            "Case 2": {
                "lambda_val": 3.5,
                "expected": [[0, 1], [1, 1]],
                "expected_iter_calls": 3,
            },
            "Case 3": {
                "lambda_val": 5,
                "expected": [[1, 1]],
                "expected_iter_calls": 3,
            },
            "Case 4": {
                "lambda_val": 10,
                "expected": [],
                "expected_iter_calls": 3,
            },
        }
        super()._test_sequence_selection_fn(test_cases_expected_outputs)


class TestPostprocessedSequenceSelector(unittest.TestCase):
    def test_sequence_selector_fn(self):
        test_cases = {
            "Base case": {
                "x": 1,
                "y": [1, 2, 3],
                "score_fn": lambda a, b: b - a,
                "selector": RunningMaxSequenceSelector,
                "input": [1, 2, 3, 4, 5, 6],
                "lambda_val": 0,
                "expected": [[2, 3]],
                "process_method": MagicMock(return_value=[[2, 3]]),
            },
        }
        for name, test_dict in test_cases.items():
            with self.subTest(name=name):
                selector = PostprocessedSequenceSelector(
                    base_selector=test_dict["selector"](score_fn=test_dict["score_fn"]),
                    process_method=test_dict["process_method"],
                )
                subseq_selected = selector.select(
                    test_dict["x"],
                    test_dict["y"],
                    test_dict["lambda_val"],
                )
                self.assertEqual(subseq_selected, test_dict["expected"])


if __name__ == "__main__":
    unittest.main()

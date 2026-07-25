import abc
import enum
import logging
import math
from typing import Callable, List, Dict, TypeVar, Generic, Any, Set

import numpy as np
import scipy

from .sequence_selector import (
    SequenceSelectorFromScore,
    _BaseSequenceSelector,
    InputType,
    OutputType,
)

AdmissibilityType = TypeVar("AdmissibilityType", bound=float | bool)
SequenceSelector = TypeVar("SequenceSelector", bound=_BaseSequenceSelector)
GroundTruthType = Any  # Normally GroundTruthType is OutputType

LOGGER = logging.getLogger(__name__)


class CacheKey(enum.IntEnum):
    SEQUENCE = 0
    ADMISSIBLE = 1
    SCORE = 2


def _known_lambda_calibration(
    admissibility_on_lambda: List[Dict[float, AdmissibilityType]],
    gamma: float,
    admissibility_function_lower_bound: AdmissibilityType,
) -> Dict[float, float]:

    # The algorithm scans the unique lambdas over the whole dataset from the smallest to the
    # largest, while stopping when the inequality in the paper is met.
    #
    # We score a sorted list of lambdas for each instance. Note that it is likely that the lambda
    # we want to calculate A_lambda := A(x, C_lambda(x, y)) is not available for this instance x.
    # In this case, if lambda1 < lambda <= lambda2, where lambda1 and lambda2 are available, we
    # have A_lambda = A_lambda1, i.e. lambda is extended to a piecewise constant function such
    # that it is right continuous. If lambda is smaller than all the lambdas available for this
    # instance, A_\lambda is given by minimum_admissibilities.
    #
    # We keep track of the current index of A_lambda to fetch for each instance as we increase
    # lambda over all unique lambdas in the dataset. The index starts at -1 to indicate we
    # have not started using A_\lambda yet. The index will increase for each instance once lambda is
    # equal to an available lambda for that instance.
    num_samples = len(admissibility_on_lambda)
    lambda_per_sample = [
        sorted([lamb for lamb in d.keys() if lamb < float("inf")]) for d in admissibility_on_lambda
    ]

    all_lambdas = sorted(list(set(lambda_ for d in lambda_per_sample for lambda_ in d)))
    cur_index = [-1 for _ in range(len(admissibility_on_lambda))]
    lambda_to_average_admissibility = {}
    for lambda_ in all_lambdas:
        admissibility = []
        for i, (lambda_list, lambda_dict) in enumerate(
            zip(lambda_per_sample, admissibility_on_lambda)
        ):
            ind = cur_index[i]
            # cur_lambda is nan if the lambda < all lambdas for this instance.
            cur_lambda = lambda_list[ind]
            next_lambda = lambda_list[ind + 1] if ind + 1 < len(lambda_list) else float("inf")
            if lambda_ == next_lambda:
                # increase the index as the lambda hits the next one.
                cur_index[i] += 1
                admissibility.append(float(lambda_dict[next_lambda]))
            elif lambda_ < next_lambda:
                admissibility.append(float(lambda_dict[cur_lambda]))
            else:
                # Lambda_ should always <= next entry.
                raise RuntimeError("It should not happen.")
        average_admissibility = float(np.mean(admissibility))
        lambda_to_average_admissibility[lambda_] = average_admissibility
        # compute the inequality in equation (6) in the paper.
        if (
            average_admissibility
            >= ((num_samples + 1) * gamma - admissibility_function_lower_bound) / num_samples
        ):
            # stop computing
            return lambda_to_average_admissibility
    return lambda_to_average_admissibility


def _validate_admissibility_dict_list(
    admissibility_dict_list: List[Dict[float, AdmissibilityType]],
    admissibility_function_lower_bound: AdmissibilityType = 0.0,
) -> None:
    """
    Validate the values of the admissibility function on all observed lambdas. The observed lambdas
    must include negative infinity for all instances.

    Raises an error if some values of the admissibility function are smaller than the provided
    lower bound.

    Raises an error if negative infinity is not among the observed lambdas for some instances.

    Issues a warning if all values of the admissibility function are strictly greater than the
    provided lower bound.

    Args:
        admissibility_dict_list:
            The original values of the admissibility function on all observed lambdas.
        admissibility_function_lower_bound:
            The lower bound for the admissibility function.

    """
    num_smaller = 0
    num_equal = 0
    num_no_neg_inf = 0
    num = 0
    for admissibility_dict in admissibility_dict_list:
        neg_inf = False
        for lambda_, admissibility in admissibility_dict.items():
            if lambda_ == float("-inf"):
                neg_inf = True
            num += 1
            if admissibility < admissibility_function_lower_bound:
                num_smaller += 1
            elif admissibility == admissibility_function_lower_bound:
                num_equal += 1
        if not neg_inf:
            num_no_neg_inf += 1
    if num_smaller > 0:
        raise ValueError(
            "Admissibility function appears to be smaller than the provided lower bound "
            f"for some samples and observed lambdas. Number of violations: {num_smaller}/{num}"
        )
    if num_no_neg_inf > 0:
        raise ValueError(
            f"Lambda=-inf was not included in {num_no_neg_inf}/{len(admissibility_dict_list)} "
            "instances."
        )
    if num_equal == 0 and num > 0:
        LOGGER.warning(
            "Admissibility function appears to always be greater than the provided lower bound."
            " Providing a larger-than-necessary admissibility lower bound this can result"
            " in valid but overly conservative calibration."
        )


def known_lambda_calibration(
    admissibility_on_lambda: List[Dict[float, AdmissibilityType]],
    gamma: float,
    admissibility_function_lower_bound: AdmissibilityType = 0.0,
) -> float:
    """
    Calibrate lambda with the threshold gamma where the value of the admissibility for a set
    of :math:`\\lambda` is provided for each instance. The admissibility function will be
    extended for unseen lambdas so that it is piecewise constant and right continuous.

    Args:
        admissibility_on_lambda:
            A list of length number of instances where each element is a dictionary containing
            the lambda value to its admissibility value. The element at i corresponds to
            :math:`\\lambda \\mapsto A(x^{(i)}, C_\\lambda(x^{(i)}, y_^{(i)}))`. Each instance
            must contain :math:`\\lambda=-\\infty` in the dictionary.
        gamma:
            The threshold :math:`\\gamma` in the paper.
        admissibility_function_lower_bound:
            The lower bound for the admissibility function.

    Returns:
        The conformal threshold :math:`\\hat{\\lambda}`.
    """
    _validate_admissibility_dict_list(admissibility_on_lambda, admissibility_function_lower_bound)
    lambda_to_average_admissibility = _known_lambda_calibration(
        admissibility_on_lambda,
        gamma,
        admissibility_function_lower_bound,
    )
    if len(lambda_to_average_admissibility) == 0:
        # no lambdas were provided.
        return float("inf")

    num_samples = len(admissibility_on_lambda)
    if (
        max(lambda_to_average_admissibility.values())
        < ((num_samples + 1) * gamma - admissibility_function_lower_bound) / num_samples
    ):
        # no lambdas satisfies the equation.
        return float("inf")

    # return the greatest lambda in the dict
    return max(lambda_to_average_admissibility.keys())


class _BaseCalibrationDataset(Generic[SequenceSelector], abc.ABC):
    """
    Abstract base class for calibration dataset. A calibration dataset will contain the input
    dataset and the raw generated dataset, which are used to calibrate the conformal thresholds.
    It also includes the sequence selector object to indicate :math:`C_\\lambda(x, y)`.
    The admissibility function is also needed, but it comes in different formats.
    """

    def __init__(
        self,
        sequence_selector: SequenceSelector,
        input_dataset: List[InputType] | List[List[InputType]],
        raw_generated_dataset: List[List[OutputType]],
        use_cache: bool,
        ground_truths: List[GroundTruthType] | None = None,
        iterate_inputs: bool = False,
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self._sequence_selector = sequence_selector
        self._input_dataset = input_dataset
        if ground_truths is None:
            self._ground_truths = [None] * len(input_dataset)
        else:
            self._ground_truths = ground_truths
        self._raw_generated_dataset = raw_generated_dataset
        self._iterate_inputs = iterate_inputs

        # cache for sequence selection, admissibility function, scores etc.
        self._cache = {}
        self._use_cache = use_cache

    def empty_cache(self) -> None:
        """
        Empty the cache.
        """
        self._cache = {}

    def enable_cache(self):
        """
        Enable the use of cache.
        """
        self._use_cache = True

    def disable_cache(self):
        """
        Disable the use of cache. Also clear the cache.
        """
        self.empty_cache()
        self._use_cache = False

    @property
    def sequence_selector(self) -> SequenceSelector:
        """
        Returns the sequence selector.
        """
        return self._sequence_selector

    def _compute_if_absent(self, cache_key: CacheKey, key, action: Callable[[], Any]) -> Any:
        """
        Compute the action if the key is absent for the particular cache, if caching is enabled.

        Args:
            cache_key:
                The name of the cache.
            key:
                The key to get.
            action:
                The action to perform.

        Returns:
            The output value.

        """
        if not self._use_cache:
            return action()
        cache = self._cache.setdefault(cache_key, {})
        value = cache.get(key)
        if value is None:
            value = action()
            cache[key] = value
        return value

    @abc.abstractmethod
    def calibrate(self, gamma: float, **calibrate_kwargs) -> float:
        """
        Calibrate using the calibration dataset.

        Args:
            gamma:
                The :math:`\\gamma` in equation (5) in the paper.
            calibrate_kwargs:
                Additional keyword arguments for the calibration function.

        Returns:
            The conformal threshold. :math:`\\hat\\lambda` in the paper.

        """

    @staticmethod
    def _validate_list(**kwargs: List[Any]) -> None:
        """
        Validate all provided elements are lists of the same length.
        """
        length = None
        first_name = None
        for name, arg in kwargs.items():
            if not isinstance(arg, list):
                raise TypeError(f"{name} is not a list. Currently it is of type {type(arg)}")
            if length is None:
                length = len(arg)
                first_name = name
                continue
            if len(arg) != length:
                raise ValueError(
                    f"Length of {name} ({len(arg)}) is different from length"
                    f" of {first_name} ({length})"
                )

    @staticmethod
    def _validate_list_of_lists(**kwargs: List[List[Any]]) -> None:
        """
        Validate all provided elements are lists of lists, such that each element in the lists is
        of the same length.
        """
        if len(kwargs) == 0:
            return

        for k, v in kwargs.items():
            if not isinstance(v, list):
                raise TypeError(f"{k} is not a list. Currently it is of type {type(v)}")

        keys, values = list(zip(*kwargs.items()))
        for i, lists in enumerate(zip(*values)):
            length = None
            for key, l in zip(keys, lists):
                if not isinstance(l, list):
                    raise TypeError(
                        f"At instance {i}, {key} is not a list. Currently it is of type {type(l)}"
                    )
                if length is None:
                    length = len(l)
                    continue
                if len(l) != length:
                    raise ValueError(
                        f"At instance {i}, length of {key} ({len(l)}) is different from "
                        f"length of {keys[0]} ({length})"
                    )


class ManualCalibrationDataset(_BaseCalibrationDataset[SequenceSelectorFromScore]):
    """
    A calibration dataset that aims to provide a list of input-output pairs for human to
    manually evaluate the admissibility. The output will be generated by the sequence selector.
    """

    def __init__(
        self,
        sequence_selector: SequenceSelectorFromScore,
        input_dataset: List[InputType] | List[List[InputType]],
        raw_generated_dataset: List[List[OutputType]],
        admissibility_function_lower_bound: float = 0,
        use_cache: bool = True,
        ground_truths: List[GroundTruthType] | None = None,
        iterate_inputs: bool = False,
    ):
        """
        Constructor

        Args:
            sequence_selector:
                The sequence selector indicates :math:`C_\\lambda(x, y)`.
            input_dataset:
                The input dataset.
            raw_generated_dataset:
                The raw generated dataset.
            admissibility_function_lower_bound:
                The lower bound of the admissibility function. Only used when not all lambda=-inf
                are provided manually.
            use_cache:
                Indicates whether we enable the use of cache.
            ground_truths:
                The ground truth to the inputs. Optional.
            iterate_inputs:
                If True, both the instance and the raw_generated_sequence will be iterated.
                If False, only the raw_generated sequence will be iterated.

        """
        super().__init__(
            sequence_selector,
            input_dataset,
            raw_generated_dataset,
            use_cache,
            ground_truths,
            iterate_inputs,
        )
        self._dict_for_evaluation = None
        self._admissibility_dict_dict = None
        self._admissibility_function_lower_bound = admissibility_function_lower_bound

    def register_admissibilities(
        self,
        evaluated_admissibilities: Dict[str, AdmissibilityType],
        reregister: bool = False,
        update: bool = True,
    ) -> None:
        """
        Register the admissibilities for calibration.

        Args:
            evaluated_admissibilities:
                User-provided admissibilities, in the format of ID -> Admissibility.
            reregister:
                If True, the original registered admissibilities will first being wiped.
            update:
                If False, and the admissibilities are already registered and reregister being False,
                this will skip re-registering.
        """
        if reregister:
            self._admissibility_dict_dict = None
        if self._admissibility_dict_dict is not None:
            if not update:
                return
        if self._dict_for_evaluation is None:
            raise ValueError(
                "Instance Lambda IDs are not yet generated! "
                "Evaluated admissibilities should not be inputted before that.\n"
                "Call get_dict_for_evaluating_admissibilities() first then for each "
                "instance lambda ID, manually evaluate the admissibility for each output"
                " sequence."
            )
        admissibility_dict_dict = {}
        for instance_lambda_id, dic in self._dict_for_evaluation.items():
            admissibility = evaluated_admissibilities.get(instance_lambda_id)
            if admissibility is None:
                if (
                    self._admissibility_dict_dict is not None
                    and self._admissibility_dict_dict.get(instance_lambda_id) is None
                ):
                    self._log.warning(
                        f"ID {instance_lambda_id} is not present in the evaluated admissibilities."
                    )
                continue
            instance_index = int(dic["instance_index"])
            lambda_ = float(dic["lambda"])
            admissibility_dict_dict.setdefault(instance_index, {})[lambda_] = admissibility
        self._admissibility_dict_dict = admissibility_dict_dict

    def get_dict_for_evaluating_admissibilities(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the information for evaluating admissibilities.

        Returns a dict of dicts, where it is
        (ID -> dict with keys ("lambda", "instance_index", "instance", "raw_output", "selected_sequence")).
        The user can provide a dict with keys being the ID and values being the admissibilities.

        Call `pandas.DataFrame.from_dict(res, orient="index").to_csv(file)` to output to csv file.
        """
        out_dict = {}
        for i, (instance, raw_generated_sequence) in enumerate(
            zip(self._input_dataset, self._raw_generated_dataset)
        ):
            scores = self._compute_if_absent(
                CacheKey.SCORE,
                i,
                lambda: [
                    self._sequence_selector.score_fn(ins, output)
                    for ins, output in self._sequence_selector.iterate(
                        instance, raw_generated_sequence, self._iterate_inputs
                    )
                ],
            )

            possible_lambdas = [float("-inf")] + self._sequence_selector.get_possible_lambdas(
                scores
            )
            for j, lambda_ in enumerate(possible_lambdas):
                selected_sequence = self._compute_if_absent(
                    CacheKey.SEQUENCE,
                    (i, lambda_),
                    lambda: self._sequence_selector.select(
                        instance,
                        raw_generated_sequence,
                        lambda_,
                        precomputed_scores=scores,
                        iterate_inputs=self._iterate_inputs,
                    ),
                )
                instance_lambda_id = f"instance_index_{i}_lambda_index_{j}"
                dic = {
                    "lambda": lambda_,
                    "instance_index": i,
                    "instance": instance,
                    "raw_output": raw_generated_sequence,
                    "selected_sequence": selected_sequence,
                }
                out_dict[instance_lambda_id] = dic
        self._dict_for_evaluation = out_dict
        return out_dict

    def calibrate(self, gamma: float, **calibrate_kwargs) -> float:
        if self._admissibility_dict_dict is None:
            raise ValueError(
                "Admissibilities are not provided. Call register_admissibility_dict_list first."
            )
        return known_lambda_calibration(
            admissibility_on_lambda=[v for _, v in self._admissibility_dict_dict.items()],
            gamma=gamma,
            admissibility_function_lower_bound=self._admissibility_function_lower_bound,
        )


class GeneralCalibrationDataset(_BaseCalibrationDataset, Generic[SequenceSelector]):
    """
    A general calibration dataset where the admissibility function is provided as a callable.
    """

    def __init__(
        self,
        sequence_selector: SequenceSelector,
        admissibility_function: Callable[
            [InputType, GroundTruthType | None, List[OutputType]], AdmissibilityType
        ],
        input_dataset: List[InputType] | List[List[InputType]],
        raw_generated_dataset: List[List[OutputType]],
        admissibility_function_lower_bound: AdmissibilityType = 0.0,
        use_cache: bool = True,
        ground_truths: List[GroundTruthType] | None = None,
        iterate_inputs: bool = False,
    ):
        """
        Constructor

        Args:
            sequence_selector:
                The sequence selector indicates :math:`C_\\lambda(x, y)`.
            admissibility_function:
                The admissibility function indicates :math:`A(x, y_{GT}, y)`.
            input_dataset:
                The input dataset.
            raw_generated_dataset:
                The raw generated dataset.
            use_cache:
                Indicate whether use cache or not.
            ground_truths:
                The ground truth to the inputs. Optional.
            iterate_inputs:
                If True, both the instance and the raw_generated_sequence will be iterated.
                If False, only the raw_generated sequence will be iterated.
        """
        super().__init__(
            sequence_selector,
            input_dataset,
            raw_generated_dataset,
            use_cache,
            ground_truths,
            iterate_inputs,
        )
        self._admissibility_function = admissibility_function
        self._admissibility_function_lower_bound = admissibility_function_lower_bound
        self._validate_list(
            input_dataset=input_dataset, raw_generated_dataset=raw_generated_dataset
        )
        kwargs = {"input_dataset": input_dataset} if iterate_inputs else {}
        self._validate_list_of_lists(
            raw_generated_dataset=raw_generated_dataset,
            **kwargs,
        )

    def _get_calibration_function(self) -> Callable[[float], float]:
        """
        Returns the calibration function
        :math:`\\lambda \\mapsto \\frac{1}{n}\\sum_{i=1}^n A(x^{(i)}, C_\\lambda(x^{(i)}, y^{(i)}))`
        for optimization.
        """
        # reassign variable so that if the function is pickled, it won't pickle the whole "self".
        input_dataset = self._input_dataset
        ground_truths = self._ground_truths
        raw_generated_dataset = self._raw_generated_dataset
        sequence_selector = self._sequence_selector
        admissibility_function = self._admissibility_function
        compute_if_absent = self._compute_if_absent
        num_sample = len(self._input_dataset)
        iterate_inputs = self._iterate_inputs

        def calibration_function(lambda_: float) -> float:
            all_admissibilities = []
            for i, (instance, ground_truth, raw_generated_sequence) in enumerate(
                zip(input_dataset, ground_truths, raw_generated_dataset)
            ):
                selected_sequence = compute_if_absent(
                    CacheKey.SEQUENCE,
                    (i, lambda_),
                    lambda: sequence_selector.select(
                        instance, raw_generated_sequence, lambda_, iterate_inputs=iterate_inputs
                    ),
                )
                admissibility = compute_if_absent(
                    CacheKey.ADMISSIBLE,
                    (i, lambda_),
                    lambda: admissibility_function(instance, ground_truth, selected_sequence),
                )
                all_admissibilities.append(float(admissibility))
            return sum(all_admissibilities) / num_sample

        return calibration_function

    def calibrate(self, gamma: float, **calibrate_kwargs) -> float:
        """
        Perform calibration by calling the minimizer from scipy.

        Args:
            gamma:
                The :math:`\\gamma` in equation (5) in the paper.
            calibrate_kwargs:
                Additional keyword arguments for the calibration function. Must contain a float "lambda0",
                which specifies the initial guess for the optimizer. calibrate_kwargs is passed to the minimizer.
        Returns:
            The conformal threshold. :math:`\\hat\\lambda` in the paper.
        """
        calibrate_kwargs = calibrate_kwargs.copy()
        if "lambda0" not in calibrate_kwargs:
            raise ValueError("calibrate_kwargs must contain a float 'lambda0' for the optimizer.")
        lambda0 = calibrate_kwargs.pop("lambda0")
        if not isinstance(lambda0, float):
            raise TypeError(f"lambda0 should be a float, but got {type(lambda0)} instead.")
        calibration_func = self._get_calibration_function()
        num_sample = len(self._input_dataset)
        admissibility_function_lower_bound = self._admissibility_function_lower_bound

        def constraint_func(lambda_np):
            target = ((num_sample + 1) * gamma - admissibility_function_lower_bound) / num_sample
            return calibration_func(lambda_np.tolist()[0]) - target

        constraints = [{"type": "ineq", "fun": constraint_func}]
        if "constraints" in calibrate_kwargs:
            additional_constraints = calibrate_kwargs.pop("constraints")
            if not isinstance(additional_constraints, list):
                additional_constraints = [additional_constraints]
            constraints.extend(additional_constraints)

        res = scipy.optimize.minimize(
            fun=lambda x: x, x0=np.array([lambda0]), constraints=constraints, **calibrate_kwargs
        )

        lambda_ = res.x.tolist()[0]
        if not np.isfinite(lambda_):
            self._log.warning("scipy failed to find a float-valued lambda.")
            return float("inf")
        if calibration_func(lambda_) < (num_sample + 1) * gamma / num_sample:
            self._log.warning(
                "scipy failed to find a lambda achieving the desired average admissibility."
            )
            return float("inf")
        elif not res.success:
            self._log.warning(
                "scipy minimization was flagged as unsuccessful, but still produced a lambda achieving"
                " the target average admissibility. It is likely that the obtained lambda is suboptimal."
            )
        return lambda_


class _ObservedLambdaCalibrationDataset(
    _BaseCalibrationDataset, Generic[SequenceSelector], abc.ABC
):
    """
    A helper abstract class for calibration dataset with a known finite set of lambdas (can be
    per sample or global)
    """

    def __init__(
        self,
        sequence_selector: SequenceSelector,
        input_dataset: List[InputType] | List[List[InputType]],
        raw_generated_dataset: List[List[OutputType]],
        admissibility_function_lower_bound: AdmissibilityType = 0.0,
        use_cache: bool = True,
        ground_truths: List[GroundTruthType] | None = None,
        iterate_inputs: bool = False,
    ):
        super().__init__(
            sequence_selector,
            input_dataset,
            raw_generated_dataset,
            use_cache,
            ground_truths,
            iterate_inputs,
        )
        self._admissibility_function_lower_bound = admissibility_function_lower_bound

    def return_all_possible_lambdas(self) -> Set[float]:
        """
        Returns the set of all possible lambdas in the calibration dataset.
        """
        admissibility_dict_list = self._get_admissibility_dict_list()
        return {
            lambda_
            for admissibility_dict in admissibility_dict_list
            for lambda_ in admissibility_dict.keys()
        }

    def validate_admissibility_function_non_decreasing(self) -> bool:
        """
        Validate the admissibility function provided is non-decreasing with respect to the input
        dataset. Note that in the paper we did not impose this restriction. Thus this is a
        stronger assumption. This is particular useful for score-based sequence selector and
        admissibility aggregator, as they may not be compatible to each other.

        Returns:
            True if the admissibility function is non-decreasing with respect to the input dataset.
            False if not, and we will output a warning.

        """
        admissibility_dict_list = self._get_admissibility_dict_list()
        total_count = 0
        fail_count = 0
        fail_sample_count = 0
        for individual_admissibilities in admissibility_dict_list:
            failed = False
            sorted_lambdas = sorted(individual_admissibilities.keys())
            if len(sorted_lambdas) < 2:
                continue
            cur_admissibility = float(individual_admissibilities[sorted_lambdas[0]])
            for lambda_ in sorted_lambdas[1:]:
                total_count += 1
                new_admissibility = float(individual_admissibilities[lambda_])
                if new_admissibility < cur_admissibility:
                    failed = True
                    fail_count += 1
                cur_admissibility = new_admissibility
            if failed:
                fail_sample_count += 1
        if fail_count > 0:
            self._log.warning(
                "Admissibility function is not a non-decreasing function. "
                f"Number of decreasing admissibilities = {fail_count}/{total_count}\n"
                "Number of samples with decreasing admissibilities = "
                f"{fail_sample_count}/{len(admissibility_dict_list)}"
            )
            return False
        return True

    def validate_average_admissibility_function_non_decreasing(self) -> bool:
        """
        Validate the average of the admissibility function provided over the input dataset is
        non-decreasing. Note that in the paper we impose a slightly different assumption.

        Returns:
            True if the average admissibility function over the dataset is non-decreasing.
            False if not, and we will output a warning.

        """
        admissibility_dict_list = self._get_admissibility_dict_list()
        lambda_to_average_admissibility = _known_lambda_calibration(
            admissibility_dict_list,
            gamma=float("inf"),  # output average admissibility for every lambda
            admissibility_function_lower_bound=0.0,  # does not matter to this dict
        )
        sorted_admissibilities = [v for _, v in sorted(lambda_to_average_admissibility.items())]
        if len(sorted_admissibilities) < 2:
            return True

        fail_count = 0
        total_count = len(sorted_admissibilities) - 1
        cur_value = sorted_admissibilities[0]
        for value in sorted_admissibilities[1:]:
            new_value = float(value)
            if new_value < cur_value:
                fail_count += 1
            cur_value = new_value
        if fail_count > 0:
            self._log.warning(
                "Average admissibility function is not a non-decreasing function. "
                f"Number of decreasing average admissibilities = {fail_count}/{total_count}"
            )
            return False
        return True

    @abc.abstractmethod
    def _get_admissibility_dict_list(self) -> List[Dict[float, AdmissibilityType]]:
        """
        Compute the admissibility
        :math:`A(x^{(i)}, y_{GT}^{(i)}, C_\\lambda^{(i)}_j(x^{(i)}, y^{(i)})` for
        each instance i and known lambdas for that instance.
        Returns a list of length number of instances, with each element is a dictionary
        corresponding to :math:`\\lambda \\mapsto A(x^{(i)}, C_\\lambda^{(i)}_j(x^{(i)}, y^{(i)})`
        """


class ObservedSparseLambdasCalibrationDataset(
    _ObservedLambdaCalibrationDataset, Generic[SequenceSelector]
):
    """
    A calibration dataset where the admissibility is provided for a specific (finite) subset of
    lambdas for each instance. The admissibility function will be extended for other lambdas
    such that it is right continuous and piecewise constant.
    """

    def __init__(
        self,
        sequence_selector: SequenceSelector,
        admissibility_function: Callable[
            [InputType, GroundTruthType | None, List[OutputType]], AdmissibilityType
        ],
        input_dataset: List[InputType] | List[List[InputType]],
        raw_generated_dataset: List[List[OutputType]],
        possible_lambdas: List[List[float]],
        admissibility_function_lower_bound: AdmissibilityType = 0.0,
        use_cache: bool = True,
        ground_truths: List[GroundTruthType] | None = None,
        iterate_inputs: bool = False,
    ):
        """
        Constructor

        Args:
            sequence_selector:
                The sequence selector indicates :math:`C_\\lambda(x, y)`.
            admissibility_function:
                The admissibility function indicates :math:`A(x, y_{GT},  y)`.
            input_dataset:
                The input dataset.
            raw_generated_dataset:
                The raw generated dataset.
            possible_lambdas:
                A list of length number of instances such that each element of the list contains
                a list of lambdas used for admissibility computation.
            admissibility_function_lower_bound:
                The lower bound of the admissibility function.
            use_cache:
                Indicates whether we enable the use of cache.
            ground_truths:
                The ground truth to the inputs. Optional.
            iterate_inputs:
                If True, both the instance and the raw_generated_sequence will be iterated.
                If False, only the raw_generated sequence will be iterated.
        """
        super().__init__(
            sequence_selector,
            input_dataset,
            raw_generated_dataset,
            admissibility_function_lower_bound,
            use_cache,
            ground_truths,
            iterate_inputs,
        )
        self._admissibility_function = admissibility_function
        self._possible_lambdas = possible_lambdas
        self._validate_list(
            input_dataset=input_dataset,
            raw_generated_dataset=raw_generated_dataset,
            possible_lambdas=possible_lambdas,
            ground_truths=self._ground_truths,
        )
        kwargs = {"input_dataset": input_dataset} if iterate_inputs else {}
        self._validate_list_of_lists(
            raw_generated_dataset=raw_generated_dataset,
            possible_lambdas=possible_lambdas,
            **kwargs,
        )

    def _get_admissibility_dict_list(self) -> List[Dict[float, AdmissibilityType]]:
        admissibility_dict_list = []
        for i, (instance, ground_truth, raw_generated_sequence, lambdas) in enumerate(
            zip(
                self._input_dataset,
                self._ground_truths,
                self._raw_generated_dataset,
                self._possible_lambdas,
            )
        ):
            instance_dict = {}
            for lambda_ in [float("-inf")] + lambdas:
                selected_sequence = self._compute_if_absent(
                    CacheKey.SEQUENCE,
                    (i, lambda_),
                    lambda: self._sequence_selector.select(
                        instance,
                        raw_generated_sequence,
                        lambda_,
                        iterate_inputs=self._iterate_inputs,
                    ),
                )
                instance_dict[lambda_] = self._compute_if_absent(
                    CacheKey.ADMISSIBLE,
                    (i, lambda_),
                    lambda: self._admissibility_function(instance, ground_truth, selected_sequence),
                )
            admissibility_dict_list.append(instance_dict)
        return admissibility_dict_list

    def calibrate(self, gamma: float, **calibrate_kwargs) -> float:
        self._log.info("Calibrating...")
        admissibility_dict_list = self._get_admissibility_dict_list()
        return known_lambda_calibration(admissibility_dict_list, gamma)


class ObservedDenseLambdasCalibrationDataset(
    _ObservedLambdaCalibrationDataset, Generic[SequenceSelector]
):
    """
    A calibration dataset where the admissibility is provided for a specific (finite) subset of
    lambdas over all instances.
    """

    def __init__(
        self,
        sequence_selector: SequenceSelector,
        admissibility_function: Callable[
            [InputType, GroundTruthType | None, List[OutputType]], AdmissibilityType
        ],
        input_dataset: List[InputType] | List[List[InputType]],
        raw_generated_dataset: List[List[OutputType]],
        all_lambdas: List[float],
        admissibility_function_lower_bound: AdmissibilityType = 0.0,
        use_cache: bool = True,
        ground_truths: List[GroundTruthType] | None = None,
        iterate_inputs: bool = False,
    ):
        """
        Constructor

        Args:
            sequence_selector:
                The sequence selector indicates :math:`C_\\lambda(x, y)`.
            admissibility_function:
                The admissibility function indicates :math:`A(x, y_{GT}, y)`.
            input_dataset:
                The input dataset.
            raw_generated_dataset:
                The raw generated dataset.
            all_lambdas:
                A list of lambdas used for admissibility computation.
            admissibility_function_lower_bound:
                The lower bound of the admissibility function.
            use_cache:
                Indicates whether we enable the use of cache.
            ground_truths:
                The ground truth to the inputs. Optional.
            iterate_inputs:
                If True, both the instance and the raw_generated_sequence will be iterated.
                If False, only the raw_generated sequence will be iterated.

        """
        super().__init__(
            sequence_selector,
            input_dataset,
            raw_generated_dataset,
            admissibility_function_lower_bound,
            use_cache,
            ground_truths,
            iterate_inputs,
        )
        self._admissibility_function = admissibility_function
        self._all_lambdas = sorted(list(set(all_lambdas + [float("-inf")])))
        self._validate_list(
            input_dataset=input_dataset,
            raw_generated_dataset=raw_generated_dataset,
            ground_truths=self._ground_truths,
        )
        kwargs = {"input_dataset": input_dataset} if iterate_inputs else {}
        self._validate_list_of_lists(
            raw_generated_dataset=raw_generated_dataset,
            **kwargs,
        )

    def _get_admissibility_dict_list(self) -> List[Dict[float, AdmissibilityType]]:
        admissibility_dict_list = []
        for i, (instance, ground_truth, raw_generated_sequence) in enumerate(
            zip(self._input_dataset, self._ground_truths, self._raw_generated_dataset)
        ):
            instance_dict = {}
            for lambda_ in self._all_lambdas:
                selected_sequence = self._compute_if_absent(
                    CacheKey.SEQUENCE,
                    (i, lambda_),
                    lambda: self._sequence_selector.select(
                        instance,
                        raw_generated_sequence,
                        lambda_,
                        iterate_inputs=self._iterate_inputs,
                    ),
                )
                instance_dict[lambda_] = self._compute_if_absent(
                    CacheKey.ADMISSIBLE,
                    (i, lambda_),
                    lambda: self._admissibility_function(instance, ground_truth, selected_sequence),
                )
            admissibility_dict_list.append(instance_dict)
        return admissibility_dict_list

    def calibrate(self, gamma: float, **calibrate_kwargs) -> float:
        self._log.info("Calibrating...")
        num_samples = len(self._input_dataset)
        admissibility_dict_list = self._get_admissibility_dict_list()
        _validate_admissibility_dict_list(
            admissibility_dict_list, self._admissibility_function_lower_bound
        )
        admissibility_matrix = np.asarray(
            [
                v
                for instance_dict in admissibility_dict_list
                for k, v in sorted(instance_dict.items())
            ]
        )

        average_admissibility = np.mean(admissibility_matrix, axis=0)
        filtered = (
            average_admissibility
            >= ((num_samples + 1) * gamma - self._admissibility_function_lower_bound) / num_samples
        )
        if np.any(filtered):
            # return smallest index on first true.
            # noinspection PyTypeChecker
            return self._all_lambdas[np.argmax(filtered)]

        # empty set to take infimum
        return float("inf")


class IndividualScoreCalibrationDataset(
    _ObservedLambdaCalibrationDataset[SequenceSelectorFromScore]
):
    """
    A calibration dataset which the sequence selector is given by score function for each
    input-generation pair, as well as the admissibility of each input-generation pair is also provided.
    The admissibility of a generated sequence with respect to the input will be a function of
    the individual admissibility of the input-generation pair.
    """

    def __init__(
        self,
        sequence_selector: SequenceSelectorFromScore,
        input_dataset: List[InputType] | List[List[InputType]],
        raw_generated_dataset: List[List[OutputType]],
        admissibility_dataset: List[List[AdmissibilityType]],
        admissibility_aggregation: Callable[[List[AdmissibilityType]], AdmissibilityType],
        admissibility_function_lower_bound: AdmissibilityType = 0.0,
        use_cache: bool = True,
        ground_truths: List[GroundTruthType] | None = None,
        iterate_inputs: bool = False,
    ):
        """
        Constructor

        Args:
            sequence_selector:
                The sequence selector indicates :math:`C_\\lambda(x, y)`.
            input_dataset:
                The input dataset.
            raw_generated_dataset:
                The raw generated dataset.
            admissibility_dataset:
                The admissibility for each input-generation pair provided. The "shape" must be the same
                as the raw_generated_dataset provided.
            admissibility_aggregation:
                A callable that contain the method to aggregate each admissibility of the
                input-generation pair.
            admissibility_function_lower_bound:
                The lower bound of the admissibility function.
            use_cache:
                Indicates whether we enable the use of cache.
            ground_truths:
                The ground truth to the inputs. Optional.
            iterate_inputs:
                If True, both the instance and the raw_generated_sequence will be iterated.
                If False, only the raw_generated sequence will be iterated.

        """
        super().__init__(
            sequence_selector,
            input_dataset,
            raw_generated_dataset,
            admissibility_function_lower_bound,
            use_cache,
            ground_truths,
            iterate_inputs,
        )
        if not sequence_selector.returns_subsequence:
            raise ValueError(
                "The sequence selector must return subsequence of the raw generated sequence."
            )
        self._admissibility_dataset = admissibility_dataset
        self._admissibility_aggregation = admissibility_aggregation
        self._validate_list(
            input_dataset=input_dataset,
            raw_generated_dataset=raw_generated_dataset,
            admissibility_dataset=admissibility_dataset,
            ground_truths=self._ground_truths,
        )
        kwargs = {"input_dataset": input_dataset} if iterate_inputs else {}
        self._validate_list_of_lists(
            raw_generated_dataset=raw_generated_dataset,
            admissibility_dataset=admissibility_dataset,
            **kwargs,
        )

    def _get_admissibility_dict_list(self) -> List[Dict[float, AdmissibilityType]]:
        """
        Compute the admissibility :math:`A(x^{(i)}, C_\\lambda^{(i)}_j(x^{(i)}, y^{(i)})` for
        each instance i. Here :math:`\\lambda^{(i)}_j` corresponds to the scores
        :math:`s(x^{(i)}, y^{(i)}_j)`.

        Returns a list of length number of instances, where each element is a dictionary
        corresponding to :math:`\\lambda \\mapsto A(x^{(i)}, C_\\lambda^{(i)}_j(x^{(i)}, y^{(i)})`
        """
        admissibility_dict_list = []
        for i, (instance, raw_generated_sequence, admissibility) in enumerate(
            zip(
                self._input_dataset,
                self._raw_generated_dataset,
                self._admissibility_dataset,
            )
        ):
            scores = self._compute_if_absent(
                CacheKey.SCORE,
                i,
                lambda: [
                    self._sequence_selector.score_fn(ins, output)
                    for ins, output in self._sequence_selector.iterate(
                        instance, raw_generated_sequence, self._iterate_inputs
                    )
                ],
            )

            possible_lambdas = [float("-inf")] + self._sequence_selector.get_possible_lambdas(
                scores
            )

            instance_dict = {}
            for lambda_ in possible_lambdas:
                selected_sequence = self._compute_if_absent(
                    CacheKey.SEQUENCE,
                    (i, lambda_),
                    lambda: self._sequence_selector.select(
                        instance,
                        raw_generated_sequence,
                        lambda_,
                        precomputed_scores=scores,
                        iterate_inputs=self._iterate_inputs,
                    ),
                )
                admissibility_selected = [
                    admissibility[raw_generated_sequence.index(seq)] for seq in selected_sequence
                ]
                instance_dict[lambda_] = self._compute_if_absent(
                    CacheKey.ADMISSIBLE,
                    (i, lambda_),
                    lambda: self._admissibility_aggregation(admissibility_selected),
                )
            admissibility_dict_list.append(instance_dict)
        return admissibility_dict_list

    def validate_admissibility_function_right_continuous(self) -> bool:
        """
        Validate the admissibility function provided is right continuous given the input
        dataset. Note that in the paper the restriction is weaker. This is because some selections
        of score-based sequence selector (and admissibility aggregator) may result in
        admissibility function not being right-continuous.

        Returns:
            True if, for a given dataset, the observed admissibility function is right continuous
            with respect to lambda. False if not, and we will output a warning.

        """
        admissibility_dict_list = self._get_admissibility_dict_list()
        total_count = 0
        fail_count = 0
        for i, (instance, raw_generated_sequence, admissibility_dict) in enumerate(
            zip(self._input_dataset, self._raw_generated_dataset, admissibility_dict_list)
        ):
            sorted_lambdas = sorted(list(admissibility_dict.keys()))
            for lambda_ in sorted_lambdas:
                if not math.isfinite(lambda_):
                    continue
                lambda_plus_epsilon = math.nextafter(lambda_, float("inf"))
                selected_sequence = self._compute_if_absent(
                    CacheKey.SEQUENCE,
                    (i, lambda_),
                    lambda: self._sequence_selector.select(
                        instance,
                        raw_generated_sequence,
                        lambda_,
                        iterate_inputs=self._iterate_inputs,
                    ),
                )
                next_selected_sequence = self._compute_if_absent(
                    CacheKey.SEQUENCE,
                    (i, lambda_plus_epsilon),
                    lambda: self._sequence_selector.select(
                        instance,
                        raw_generated_sequence,
                        lambda_plus_epsilon,
                        iterate_inputs=self._iterate_inputs,
                    ),
                )
                total_count += 1
                if selected_sequence != next_selected_sequence:
                    fail_count += 1
        if fail_count > 0:
            self._log.warning(
                f"Admissibility function is not a right continuous function. "
                f"Number of non-admissibilities = {fail_count}/{total_count}"
            )
            return False
        return True

    def calibrate(self, gamma: float, **calibrate_kwargs) -> float:
        self._log.info("Calibrating...")
        admissibility_dict_list = self._get_admissibility_dict_list()
        return known_lambda_calibration(admissibility_dict_list, gamma)

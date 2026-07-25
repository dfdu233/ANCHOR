import logging
import math
from typing import List, TypeVar, Callable, Dict, Type, Iterable
from .sequence_selector import (
    InputType,
    OutputType,
    SequenceSelectorFromScore,
    RunningMaxSequenceSelector,
    RunningSumSequenceSelector,
    SimpleMovingAverageSequenceSelector,
    ExponentialMovingAverageSequenceSelector,
    RunningCharacteristicSequenceSelector,
    RunningSmallestSubsetSumSequenceSelector,
    RunningMaxSingleSequenceSelector,
    SmallestSubsetSumSequenceSelector,
    AboveLambdaSequenceSelector,
)
from .calibration_dataset import (
    _BaseCalibrationDataset,
    SequenceSelector,
    AdmissibilityType,
    IndividualScoreCalibrationDataset,
    GroundTruthType,
)

CalibrationDataset = TypeVar("CalibrationDataset", bound=_BaseCalibrationDataset)
ScoreSequenceSelector = TypeVar("ScoreSequenceSelector", bound=SequenceSelectorFromScore)

_score_mapping_dict: Dict[str, Type[ScoreSequenceSelector]] = {
    "running_max": RunningMaxSequenceSelector,
    "single_running_max": RunningMaxSingleSequenceSelector,
    "running_sum": RunningSumSequenceSelector,
    "running_sum_subset": RunningSmallestSubsetSumSequenceSelector,
    "simple_moving_average": SimpleMovingAverageSequenceSelector,
    "exponential_moving_average": ExponentialMovingAverageSequenceSelector,
    "running_characteristic": RunningCharacteristicSequenceSelector,
    "sum_subset": SmallestSubsetSumSequenceSelector,
    "above_lambda": AboveLambdaSequenceSelector,
}


class ConformalGeneration:
    """
    Main class used for conformal generation.
    """

    def __init__(
        self,
        sequence_selector: SequenceSelector | None = None,
        calibration_dataset: CalibrationDataset | None = None,
        conformal_threshold: float | None = None,
    ) -> None:
        """
        Constructor

        Args:
            sequence_selector:
                The sequence selector used for inference. Must be provided if calibration dataset
                is not provided.
            calibration_dataset:
                The calibration dataset. If not provided, both sequence_selector and conformal
                threshold must be provided. If the sequence selector is provided, then it must match
                the sequence selector in the calibration dataset.
            conformal_threshold:
                The conformal threshold. If provided, we will not use the calibration dataset for
                calibration. If not provided, the calibration dataset must be provided, and the
                user has to call :py:func:`~ConformalGeneration.calibrate`.
        """
        self._log = logging.getLogger(ConformalGeneration.__name__)
        if calibration_dataset is not None:
            if sequence_selector is not None:
                if sequence_selector != calibration_dataset.sequence_selector:
                    raise ValueError(
                        "Sequence selector in calibration dataset does not much the provided one."
                    )
            if conformal_threshold is not None:
                self._log.warning(
                    "Both conformal threshold and conformal dataset are provided. "
                    "They may not be compatible."
                )
            self._sequence_selector = calibration_dataset.sequence_selector
        else:
            if sequence_selector is None or conformal_threshold is None:
                raise ValueError(
                    "Calibration dataset is not provided. Both sequence"
                    " selector and conformal threshold must be provided."
                )
            self._sequence_selector = sequence_selector
        self._calibration_dataset: CalibrationDataset = calibration_dataset
        self._conformal_threshold = conformal_threshold

    def select(
        self,
        instance: InputType | Iterable[InputType],
        raw_generated_sequence: Iterable[OutputType],
        iterate_inputs: bool = False,
    ) -> List[OutputType]:
        """
        Perform conformal generation.

        Args:
            instance:
                The input instance.
            raw_generated_sequence:
                The raw generated sequence for conformal generation.
            iterate_inputs:
                If True, both the instance and the raw_generated_sequence will be iterated.
                If False, only the raw_generated sequence will be iterated.

        Returns:
            The conformal generation :math:`C_\\lambda(x, y)`
        """
        conformal_threshold = self.conformal_threshold
        if conformal_threshold == math.inf:
            self._log.warning(
                "Conformal threshold is infinity. Conformal Generation only vacuously "
                "satisfies the conformal guarantee."
            )
        return self._sequence_selector.select(
            instance, raw_generated_sequence, conformal_threshold, iterate_inputs=iterate_inputs
        )

    @property
    def sequence_selector(self) -> SequenceSelector:
        """
        Returns the sequence selector object.
        """
        return self._sequence_selector

    @property
    def conformal_threshold(self) -> float:
        """
        Returns the conformal threshold.
        """
        if self._conformal_threshold is None:
            raise ValueError("Conformal threshold is not set. Please call calibrate.")

        return self._conformal_threshold

    def calibrate(
        self, gamma: float, recalibrate: bool = False, empty_cache: bool = False, **calibrate_kwargs
    ) -> None:
        """
        Calibrate using the calibration dataset.

        Args:
            gamma:
                The calibration threshold in equation (5) of the paper.
            recalibrate:
                Indicate whether we want to recalibrate using the calibration dataset.
            empty_cache:
                In case of performing recalibration, this indicates whether we first empty caches
                before calibration.
            calibrate_kwargs:
                Any extra kwargs to pass into the calibrate function. Required when the calibrate
                dataset is of type :py:obj:`GeneralCalibrationDataset`.

        """
        if self._conformal_threshold is not None and not recalibrate:
            return
        if self._calibration_dataset is None:
            raise RuntimeError("Calibration dataset is None, which should not happen.")
        if empty_cache:
            self._calibration_dataset.empty_cache()
        self._conformal_threshold = self._calibration_dataset.calibrate(gamma, **calibrate_kwargs)

    @classmethod
    def from_score_function(
        cls,
        input_dataset: List[InputType],
        raw_generated_dataset: List[List[OutputType]],
        score_fn: Callable[[InputType, OutputType], float],
        score_method: str,
        admissibility_dataset: List[List[AdmissibilityType]],
        admissibility_aggregation: Callable[[List[AdmissibilityType]], AdmissibilityType],
        admissibility_function_lower_bound: AdmissibilityType = 0.0,
        use_cache: bool = True,
        ground_truths: List[GroundTruthType] | None = None,
        iterate_inputs: bool = False,
    ):
        """
        A convenient method to construct the conformal generation using score functions.

        Args:
            input_dataset:
                The input dataset.
            raw_generated_dataset:
                The raw generated dataset.
            score_fn:
                The score function that gives an input-generation pair a score.
            score_method:
                The method of selecting sequence based on score. Currently it has to be
                ``running_max``, ``single_running_max``, ``running_sum``, ``running_sum_subset``,
                ``sum_subset`` or ``above_lambda``.
            admissibility_dataset:
                The admissibility for each input-generation pair provided. The "shape" must be the same
                as raw_generated_dataset's.
            admissibility_aggregation:
                A callable that contains the method to aggregate each admissibility of the
                input-generation pairs.
            admissibility_function_lower_bound:
                The lower bound of the admissibility function.
            use_cache:
                Indicates whether we enable the use of cache.
            ground_truths:
                The ground truth to the inputs. Optional.
            iterate_inputs:
                If True, both the instance and the raw_generated_sequence will be iterated.
                If False, only the raw_generated sequence will be iterated.

        Returns:
            A :py:obj:`~ConformalGeneration` object.
        """
        sequence_selector_cls = _score_mapping_dict.get(score_method)
        if sequence_selector_cls is None:
            raise ValueError(f"Score method must be one of {list(_score_mapping_dict.keys())}")
        sequence_selector = sequence_selector_cls(score_fn)
        calibration_dataset = IndividualScoreCalibrationDataset(
            sequence_selector,
            input_dataset,
            raw_generated_dataset,
            admissibility_dataset,
            admissibility_aggregation,
            admissibility_function_lower_bound,
            use_cache,
            ground_truths,
            iterate_inputs,
        )
        return cls(calibration_dataset=calibration_dataset)

import abc
from typing import Callable, List, Any, Iterable, Tuple
from .utils import subset_index_aux

InputType = Any
OutputType = Any


class _BaseSequenceSelector(abc.ABC):
    """
    Abstract Base Class of a sequence selector. A sequence selector contains the way to output
    a sequence of generations given the input instance and a raw generated sequence.
    In the paper, it corresponds to :math:`C_\\lambda(x, y)`
    """

    def __init__(
        self,
        sequential: bool,
        returns_subsequence: bool,
    ) -> None:
        """
        Constructor

        Args:
            sequential:
                Indicates whether the sequence selector's output is sequential. This means that
                the sequence selector produces one output at a time.
            returns_subsequence:
                Indicates whether the output sequence is a subsequence of the input sequence.
        """
        self._sequential = sequential
        self._returns_subsequence = returns_subsequence

    @property
    def returns_subsequence(self) -> bool:
        return self._returns_subsequence

    @property
    def sequential(self) -> bool:
        return self._sequential

    @abc.abstractmethod
    def select(
        self,
        instance: InputType | Iterable[InputType],
        raw_generated_sequence: Iterable[OutputType],
        lambda_: float,
        iterate_inputs: bool,
    ) -> List[OutputType]:
        """
        Output a sequence given the input instance and raw generated sequence for a specific
        value of :math:`\\lambda`. In the paper, this corresponds to :math:`C_\\lambda(x, y)`

        Args:
            instance:
                The input instance.
            raw_generated_sequence:
                The raw generated sequence of the input instance.
            lambda_:
                The value indicating how conservative the output sequence should be.
            iterate_inputs:
                If True, both the instance and the raw_generated_sequence will be iterated.
                If False, only the raw_generated sequence will be iterated.

        Returns:
            Returns a list of outputs corresponding to the selected sequence.
        """


class SequenceSelectorFromScore(_BaseSequenceSelector, abc.ABC):
    """
    A special sequence selector with a score function. For each input instance, the score
    function gives a score to each generation. For each generation in the output sequence,
    the selection will depend on the scores of the raw generations, and the threshold lambda_.
    """

    def __init__(
        self,
        score_fn: Callable[[InputType, OutputType], float],
        sequential: bool,
        returns_subsequence: bool,
    ) -> None:
        """
        Constructor

        Args:
            score_fn:
                The score function that gives an input-generation pair a score.
            sequential:
                Indicates whether the sequence selector output is sequential. This means that
                the sequence selector produces one output at a time.
            returns_subsequence:
                Indicates whether the output is a subsequence of the raw generated sequence.
        """
        super().__init__(
            sequential=sequential,
            returns_subsequence=returns_subsequence,
        )
        self._score_fn = score_fn

    @staticmethod
    def iterate(
        instance: InputType | Iterable[InputType],
        raw_generated_sequence: Iterable[OutputType],
        iterate_inputs: bool,
    ) -> Iterable[Tuple[InputType, OutputType]]:
        if iterate_inputs:
            for i, output in zip(instance, raw_generated_sequence):
                yield i, output
        else:
            for output in raw_generated_sequence:
                yield instance, output

    def score_fn(self, instance: InputType, output: OutputType) -> float:
        """
        Applies the given score function.
        """
        return self._score_fn(instance, output)

    @staticmethod
    @abc.abstractmethod
    def get_possible_lambdas(precomputed_scores: List[float]) -> List[float]:
        """
        Given the possible scores of each input-generation pair, compute the possible lambdas, i.e.
        the points where the output sequences of :py:meth:`~SequenceSelectorFromScore.select`
        are discontinuous. User should not call this method and should only be used
        in calibration phase.

        Args:
            precomputed_scores:
                The precomputed scores for the input-generation pair.

        Returns:
            A sorted list, without duplicates, containing the possible discontinuities for
            the select method.

        """

    @abc.abstractmethod
    def select(
        self,
        instance: InputType | Iterable[InputType],
        raw_generated_sequence: Iterable[OutputType],
        lambda_: float,
        precomputed_scores: List[float] | None = None,
        iterate_inputs: bool = False,
    ) -> List[OutputType]:
        """
        Output a sequence given the input instance and raw generated sequence for a specific value of
        :math:`\\lambda`. In the paper, it corresponds to :math:`C_\\lambda(x, y)`.

        Args:
            instance:
                The input instance.
            raw_generated_sequence:
                The generated sequence for the input instance.
            lambda_:
                The value indicating how conservative the output sequence would be.
            precomputed_scores:
                If provided, we assume that the scores for each generation has been computed, and we
                will not compute the scores again for the sequence selection. No validation is provided.
            iterate_inputs:
                If True, both the instance and the raw_generated_sequence will be iterated.
                If False, only the raw_generated sequence will be iterated.

        Returns:
            Returns a list of outputs corresponding to the selected sequence.
        """

    def _get_score(
        self,
        instance: InputType,
        generation: OutputType,
        precomputed_scores: List[float] | None,
        index: int,
    ) -> float:
        """
        A convenient method that either computes or retrieves the score based on the index of the
        generation in the generated sequence and whether precomputed scores are provided.
        """
        if precomputed_scores is not None and 0 <= index < len(precomputed_scores):
            return precomputed_scores[index]
        return self._score_fn(instance, generation)


class RunningMaxSequenceSelector(SequenceSelectorFromScore):
    """
    The running max sequence selector will output the subsequence of the raw generated sequence up to
    the first element whose score is greater than :math:`\\lambda`. This means
    if y is the raw generated sequence, it will select ``y[:i+1]`` where i is the smallest index
    such that the score of ``y[i]`` is greater than :math:`\\lambda`.

    If all scores are less than or equal to :math:`\\lambda`, it will output the whole output sequence y.
    """

    def __init__(self, score_fn: Callable[[InputType, OutputType], float]) -> None:
        """
        Constructor

        Args:
            score_fn:
                The score function that gives an input-generation pair a score.
        """
        super().__init__(score_fn=score_fn, sequential=True, returns_subsequence=True)

    @staticmethod
    def get_possible_lambdas(precomputed_scores: List[float]) -> List[float]:
        return sorted(list(set(precomputed_scores)))

    def select(
        self,
        instance: InputType | Iterable[InputType],
        raw_generated_sequence: Iterable[OutputType],
        lambda_: float,
        precomputed_scores: List[float] | None = None,
        iterate_inputs: bool = False,
    ) -> List[OutputType]:
        current_sequence = []
        for i, (ins, output) in enumerate(
            self.iterate(instance, raw_generated_sequence, iterate_inputs)
        ):
            current_sequence.append(output)
            score = self._get_score(ins, output, precomputed_scores, i)
            if score > lambda_:
                break
        return current_sequence


class RunningSumSequenceSelector(SequenceSelectorFromScore):
    """
    The running sum sequence selector will output the subsequence of the raw generated sequence up to
    the first element whose cumulative score is greater than :math:`\\lambda`.
    This means if y is the raw generated sequence, it will select ``y[:i+1]`` where i is the smallest
    index such that the sum of the scores of ``y[:i+1]`` is greater than :math:`\\lambda`.

    If the sum all scores are less than or equal to :math:`\\lambda`, it will output the whole
    output sequence y.
    """

    def __init__(self, score_fn: Callable[[InputType, OutputType], float]) -> None:
        """
        Constructor

        Args:
            score_fn:
                The score function that gives an input-generation pair a score.
        """
        super().__init__(score_fn=score_fn, sequential=True, returns_subsequence=True)

    @staticmethod
    def get_possible_lambdas(precomputed_scores: List[float]) -> List[float]:
        possible_lambdas = []
        running_sum = 0
        for score in precomputed_scores:
            running_sum += score
            possible_lambdas.append(running_sum)
        return sorted(list(set(possible_lambdas)))

    def select(
        self,
        instance: InputType | Iterable[InputType],
        raw_generated_sequence: Iterable[OutputType],
        lambda_: float,
        precomputed_scores: List[float] | None = None,
        iterate_inputs: bool = False,
    ) -> List[OutputType]:
        current_sum = 0
        current_sequence = []
        for i, (ins, output) in enumerate(
            self.iterate(instance, raw_generated_sequence, iterate_inputs)
        ):
            current_sequence.append(output)
            score = self._get_score(ins, output, precomputed_scores, i)
            current_sum += score
            if current_sum > lambda_:
                break
        return current_sequence


class RunningSmallestSubsetSumSequenceSelector(SequenceSelectorFromScore):
    """
    The running smallest subset sum sequence selector will output the smallest subsequence of the
    raw generated sequence up to the first element whose cumulative score is greater than
    :math:`\\lambda`. This means if y is the raw generated sequence, it will select the smallest
    subset of ``y[:i+1]`` where i is the smallest index such that the sum of the scores
    of ``y[:i+1]`` is greater than :math:`\\lambda`.

    If the sum all scores are less than or equal to :math:`\\lambda`, it will output the whole
    output sequence y.
    """

    def __init__(self, score_fn: Callable[[InputType, OutputType], float]) -> None:
        """
        Constructor

        Args:
            score_fn:
                The score function that gives an input-generation pair a score.
        """
        super().__init__(score_fn=score_fn, sequential=True, returns_subsequence=True)

    @staticmethod
    def get_possible_lambdas(precomputed_scores: List[float]) -> List[float]:
        possible_lambdas = []
        running_sum = 0
        for score in precomputed_scores:
            running_sum += score
            possible_lambdas.append(running_sum)
        return sorted(list(set(possible_lambdas)))

    def select(
        self,
        instance: InputType | Iterable[InputType],
        raw_generated_sequence: Iterable[OutputType],
        lambda_: float,
        precomputed_scores: List[float] | None = None,
        iterate_inputs: bool = False,
    ) -> List[OutputType]:
        current_sum = 0
        scores = []
        current_sequence = []
        for i, (ins, output) in enumerate(
            self.iterate(instance, raw_generated_sequence, iterate_inputs)
        ):
            current_sequence.append(output)
            score = self._get_score(ins, output, precomputed_scores, i)
            scores.append(score)
            current_sum += score
            if current_sum > lambda_:
                indices = subset_index_aux(scores, lambda_)
                return [current_sequence[t] for t in indices]
        return current_sequence


class RunningMaxSingleSequenceSelector(SequenceSelectorFromScore):
    """
    The running max single sequence selector will output the first element of the
    raw generated sequence with its score greater than :math:`\\lambda`. This means if y is
    the raw generated sequence, it will output ``y[i]`` where i is the smallest index such that the score
    of ``y[i]`` is greater than :math:`\\lambda`.

    If the sum all scores are less than or equal to :math:`\\lambda`, it will output the last element in
    the raw generated sequence.
    """

    def __init__(self, score_fn: Callable[[InputType, OutputType], float]) -> None:
        """
        Constructor

        Args:
            score_fn:
                The score function that gives an input-generation pair a score.
        """
        super().__init__(score_fn=score_fn, sequential=True, returns_subsequence=True)

    @staticmethod
    def get_possible_lambdas(precomputed_scores: List[float]) -> List[float]:
        return sorted(list(set(precomputed_scores)))

    def select(
        self,
        instance: InputType | Iterable[InputType],
        raw_generated_sequence: Iterable[OutputType],
        lambda_: float,
        precomputed_scores: List[float] | None = None,
        iterate_inputs: bool = False,
    ) -> List[OutputType]:
        current_sequence = []
        for i, (ins, output) in enumerate(
            self.iterate(instance, raw_generated_sequence, iterate_inputs)
        ):
            current_sequence.append(output)
            score = self._get_score(ins, output, precomputed_scores, i)
            if score > lambda_:
                return [output]
        return current_sequence[-1:] if len(current_sequence) > 0 else []


class SimpleMovingAverageSequenceSelector(SequenceSelectorFromScore):
    """
    The simple moving average sequence selector will output the subsequence of the raw generated sequence up to
    the first element whose simple moving average (given window size) is greater than :math:`\\lambda`.
    This means if y is the raw generated sequence, it will select ``y[:i+1]`` where i is the smallest
    index such that the simple moving average of the scores of ``y[i-window_size+1:i+1]`` is greater than :math:`\\lambda`.

    If all simple moving averages are less than or equal to :math:`\\lambda`, it will output the whole
    output sequence y.
    """

    def __init__(
        self, score_fn: Callable[[InputType, OutputType], float], window_size: int | None = None
    ) -> None:
        """
        Constructor

        Args:
            score_fn:
                The score function that gives an input-generation pair a score.
            window_size:
                The window size for simple moving average. If None, use all previous scores.
        """
        super().__init__(score_fn=score_fn, sequential=True, returns_subsequence=True)
        self._window_size = window_size

    def sma_step(self, score: float, prev: float = 0, window: List[float] | None = None) -> float:
        if window is None:
            window = []
        total = score + prev * len(window)
        window.append(score)
        if self._window_size is not None and len(window) > self._window_size:
            total -= window.pop(0)
        return total / len(window)

    def get_possible_lambdas(self, precomputed_scores: List[float]) -> List[float]:
        possible_lambdas = []
        window = []
        current_avg = 0
        for n in precomputed_scores:
            current_avg = self.sma_step(n, current_avg, window)
            possible_lambdas.append(current_avg)
        return sorted(list(set(possible_lambdas)))

    def select(
        self,
        instance: InputType,
        raw_generated_sequence: Iterable[OutputType],
        lambda_: float,
        precomputed_scores: List[float] | None = None,
        iterate_inputs: bool = False,
    ) -> List[OutputType]:
        current_sequence = []
        window = []
        current_avg = 0
        for i, (ins, output) in enumerate(
            self.iterate(instance, raw_generated_sequence, iterate_inputs)
        ):
            current_sequence.append(output)
            score = self._get_score(ins, output, precomputed_scores, i)
            current_avg = self.sma_step(score, current_avg, window)
            if current_avg > lambda_:
                break
        return current_sequence


class ExponentialMovingAverageSequenceSelector(SequenceSelectorFromScore):
    """
    The exponential moving average sequence selector will output the subsequence of the raw generated sequence up to
    the first element whose exponential moving average score is greater than :math:`\\lambda`.
    This means if y is the raw generated sequence, it will select ``y[:i+1]`` where i is the smallest
    index such that the exponential moving average of the scores of ``y[:i+1]`` is greater than :math:`\\lambda`.

    If the exponential moving average of all scores is less than or equal to :math:`\\lambda`, it will output the whole
    output sequence y.
    """

    def __init__(
        self, score_fn: Callable[[InputType, OutputType], float], alpha: float = 0.5
    ) -> None:
        """
        Constructor

        Args:
            score_fn:
                The score function that gives an input-generation pair a score.
            alpha:
                The smoothing factor for exponential moving average. Default is 0.5.
        """
        super().__init__(score_fn=score_fn, sequential=True, returns_subsequence=True)
        self._alpha = alpha

    def ema_step(self, score: float, prev: float = 0) -> float:
        if prev is None:
            return score
        return self._alpha * score + (1 - self._alpha) * prev

    def get_possible_lambdas(self, precomputed_scores: List[float]) -> List[float]:
        possible_lambdas = []
        prev = None
        for n in precomputed_scores:
            possible_lambdas.append(self.ema_step(n, prev))
            prev = possible_lambdas[-1]
        return sorted(list(set(possible_lambdas)))

    def select(
        self,
        instance: InputType,
        raw_generated_sequence: Iterable[OutputType],
        lambda_: float,
        precomputed_scores: List[float] | None = None,
        iterate_inputs: bool = False,
    ) -> List[OutputType]:
        current_sequence = []
        prev = None
        for i, (ins, output) in enumerate(
            self.iterate(instance, raw_generated_sequence, iterate_inputs)
        ):
            current_sequence.append(output)
            score = self._get_score(ins, output, precomputed_scores, i)
            avg = self.ema_step(score, prev)
            prev = avg
            if avg > lambda_:
                break
        return current_sequence


class RunningCharacteristicSequenceSelector(SequenceSelectorFromScore):
    """
    The running characteristic sequence selector will output the subsequence of the raw generated sequence up to
    the first element whose characteristic score is greater than :math:`\\lambda`.
    This means if y is the raw generated sequence, it will select ``y[:i+1]`` where i is the smallest
    index such that the characteristic of the scores of ``y[:i+1]`` is greater than :math:`\\lambda`.

    If the characteristic of all scores are less than or equal to :math:`\\lambda`, it will output the whole
    output sequence y.
    """

    def __init__(
        self,
        score_fn: Callable[[InputType, OutputType], float],
        characteristic_function: Callable[[List[float]], float],
    ) -> None:
        """
        Constructor

        Args:
            score_fn:
                The score function that gives an input-generation pair a score.
            characteristic_function:
                The function to compute the characteristic of a list of scores.
        """
        super().__init__(score_fn=score_fn, sequential=True, returns_subsequence=True)
        self._characteristic_function = characteristic_function

    def get_possible_lambdas(self, precomputed_scores: List[float]) -> List[float]:
        possible_lambdas = []
        for i in range(len(precomputed_scores)):
            possible_lambdas.append(self._characteristic_function(precomputed_scores[: i + 1]))
        return sorted(list(set(possible_lambdas)))

    def select(
        self,
        instance: InputType,
        raw_generated_sequence: Iterable[OutputType],
        lambda_: float,
        precomputed_scores: List[float] | None = None,
        iterate_inputs: bool = False,
    ) -> List[OutputType]:
        current_sequence = []
        current_scores = []
        for i, (ins, output) in enumerate(
            self.iterate(instance, raw_generated_sequence, iterate_inputs)
        ):
            current_sequence.append(output)
            score = self._get_score(ins, output, precomputed_scores, i)
            current_scores.append(score)
            char = self._characteristic_function(current_scores)
            if char > lambda_:
                break
        return current_sequence


class SmallestSubsetSumSequenceSelector(SequenceSelectorFromScore):
    """
    The smallest subset sum sequence selector will output the smallest subset such that the
    sum of the scores in this subset is greater than :math:`\\lambda`.

    If the sum all scores is less than or equal to:math:`\\lambda`, it will output the whole sequence.
    """

    def __init__(self, score_fn: Callable[[InputType, OutputType], float]) -> None:
        """
        Constructor

        Args:
            score_fn:
                The score function that gives an input-generation pair a score.
        """
        super().__init__(score_fn=score_fn, sequential=False, returns_subsequence=True)

    @staticmethod
    def get_possible_lambdas(precomputed_scores: List[float]) -> List[float]:
        scores = sorted(precomputed_scores)[::-1]
        running_sum = 0
        possible_lambdas = []
        for score in scores:
            running_sum += score
            possible_lambdas.append(running_sum)
        return sorted(list(set(possible_lambdas)))

    def select(
        self,
        instance: InputType | Iterable[InputType],
        raw_generated_sequence: Iterable[OutputType],
        lambda_: float,
        precomputed_scores: List[float] | None = None,
        iterate_inputs: bool = False,
    ) -> List[OutputType]:
        raw_generated_sequence = list(raw_generated_sequence)
        scores = [
            self._get_score(ins, output, precomputed_scores, i)
            for i, (ins, output) in enumerate(
                self.iterate(instance, raw_generated_sequence, iterate_inputs)
            )
        ]
        indices = subset_index_aux(scores, lambda_)
        return [raw_generated_sequence[t] for t in indices]


class AboveLambdaSequenceSelector(SequenceSelectorFromScore):
    """
    The above lambda sequence selector will output the elements of the raw generated sequence such that
    their scores are greater than :math:`\\lambda`.
    """

    def __init__(self, score_fn: Callable[[InputType, OutputType], float]) -> None:
        """
        Constructor

        Args:
            score_fn:
                The score function that gives an input-generation pair a score.
        """
        super().__init__(score_fn=score_fn, sequential=False, returns_subsequence=True)

    @staticmethod
    def get_possible_lambdas(precomputed_scores: List[float]) -> List[float]:
        return sorted(list(set(precomputed_scores)))

    def select(
        self,
        instance: InputType | Iterable[InputType],
        raw_generated_sequence: Iterable[OutputType],
        lambda_: float,
        precomputed_scores: List[float] | None = None,
        iterate_inputs: bool = False,
    ) -> List[OutputType]:
        raw_generated_sequence = list(raw_generated_sequence)
        scores = [
            self._get_score(ins, output, precomputed_scores, i)
            for i, (ins, output) in enumerate(
                self.iterate(instance, raw_generated_sequence, iterate_inputs)
            )
        ]
        indices = [i for i, score in enumerate(scores) if score > lambda_]
        return [raw_generated_sequence[t] for t in indices]


class PostprocessedSequenceSelector(SequenceSelectorFromScore):
    """
    A sequence selector that ensures all outputs in the selected sequence are chosen based on user provided process method.
    It wraps around another sequence selector and picks a subset of elements.
    """

    def __init__(
        self,
        base_selector: SequenceSelectorFromScore,
        process_method: Callable[[List[OutputType]], List[OutputType]],
    ) -> None:
        """
        Constructor

        Args:
            base_selector:
                The base sequence selector to wrap around.
            process_method:
                A method that takes in a list of outputs and returns a processed list of outputs.
        """
        super().__init__(
            score_fn=base_selector._score_fn,
            sequential=base_selector._sequential,
            returns_subsequence=False,
        )
        self._base_selector = base_selector
        self._process_method = process_method

    def get_possible_lambdas(self, precomputed_scores: List[float]) -> List[float]:
        return self._base_selector.get_possible_lambdas(precomputed_scores)

    def select(
        self,
        instance: InputType | Iterable[InputType],
        raw_generated_sequence: Iterable[OutputType],
        lambda_: float,
        precomputed_scores: List[float] | None = None,
        iterate_inputs: bool = False,
    ) -> List[OutputType]:
        base_selection = self._base_selector.select(
            instance,
            raw_generated_sequence,
            lambda_,
            precomputed_scores,
            iterate_inputs=iterate_inputs,
        )
        processed_outputs = self._process_method(instance, base_selection)
        return processed_outputs

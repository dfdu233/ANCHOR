from typing import List


def dot_product(vector1: List[float], vector2: List[float]) -> float:
    """
    Compute the dot product of two vectors.

    :param vector1: List of coordinates for the first vector.
    :param vector2: List of coordinates for the second vector.
    :returns: The dot product of the two vectors.
    """
    assert len(vector1) == len(
        vector2
    ), f"Vectors must have the same dimension, but receiving vectors {vector1} and {vector2}"
    return sum(a * b for a, b in zip(vector1, vector2))


def ema(scores, alpha=0.5):
    prev = None
    for score in scores:
        if prev is None:
            prev = score
        else:
            prev = alpha * score + (1 - alpha) * prev
    return prev


def max_admissibility(admissibility_list: List[bool | float]) -> bool | float:
    """
    Compute the maximum admissibility for each element across all sequences.

    :param admissibility_list: A list of lists, where each inner list contains boolean or float values
                               indicating the admissibility of a sequence.
    :returns: Max element of admissibility_list.
    """

    return max(admissibility_list)


def sum_admissibility(admissibility_list: List[bool | float]) -> bool | float:
    """
    Compute the sum of admissibility for each element across all sequences.

    :param admissibility_list: A list of lists, where each inner list contains boolean or float values
                               indicating the admissibility of a sequence.
    :returns: Sum of elements in admissibility_list.
    """
    if isinstance(admissibility_list[0], bool):
        # summing boolean values can return a value greater than 1
        return any(admissibility_list)
    return sum(admissibility_list)


def max_admissibility_from_dot_product(x, yhat, seq, iterate_inputs=False):
    assert x is not None, "Input x should not be None"
    assert seq is not None, "Input seq should not be None"
    if iterate_inputs:
        ans = [dot_product(xx, s) for xx, s in zip(x, seq)]
    else:
        ans = [dot_product(x, s) for s in seq]
    return max(ans) if ans else 0

import numpy as np
from PIL import Image

from corrected_sgta.run_huatuo_negation_coverage_probe import (
    CONTROL_BOXES,
    TARGET_BOXES,
    mean_fill,
    raster_mask,
)


def test_target_and_control_masks_have_equal_raster_area() -> None:
    for width, height in ((336, 336), (2048, 2500), (2500, 2048)):
        target = raster_mask(width, height, TARGET_BOXES)
        control = raster_mask(width, height, CONTROL_BOXES)
        assert abs(int(target.sum()) - int(control.sum())) <= 2


def test_mean_fill_changes_only_the_mask() -> None:
    array = np.arange(60 * 80 * 3, dtype=np.uint32).reshape(60, 80, 3) % 251
    image = Image.fromarray(array.astype(np.uint8), mode="RGB")
    result = np.asarray(mean_fill(image, TARGET_BOXES))
    mask = raster_mask(80, 60, TARGET_BOXES)
    assert np.array_equal(result[~mask], np.asarray(image)[~mask])
    assert np.any(result[mask] != np.asarray(image)[mask])

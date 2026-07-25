"""Wave-A structure audit with a defined degenerate-correlation case."""

from __future__ import annotations

import cv2
import numpy as np

from corrected_sgta import structure_audit_v2 as implementation


def structure_proxy(left: np.ndarray, right: np.ndarray) -> dict:
    mask = implementation.central_mask(left.shape)
    left_local = left - cv2.GaussianBlur(left, (0, 0), 3.0)
    right_local = right - cv2.GaussianBlur(right, (0, 0), 3.0)
    a, b = left_local[mask], right_local[mask]
    if a.std() < 1e-8 and b.std() < 1e-8:
        correlation = 1.0 if np.allclose(a, b, atol=1e-8) else 0.0
    elif a.std() < 1e-8 or b.std() < 1e-8:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(a, b)[0, 1])
    left_gx = cv2.Sobel(left, cv2.CV_64F, 1, 0, ksize=3)
    left_gy = cv2.Sobel(left, cv2.CV_64F, 0, 1, ksize=3)
    right_gx = cv2.Sobel(right, cv2.CV_64F, 1, 0, ksize=3)
    right_gy = cv2.Sobel(right, cv2.CV_64F, 0, 1, ksize=3)
    left_gradient = np.hypot(left_gx, left_gy)[mask].mean()
    right_gradient = np.hypot(right_gx, right_gy)[mask].mean()
    if left_gradient < 1e-12 and right_gradient < 1e-12:
        gradient_ratio = 1.0
    else:
        gradient_ratio = float(right_gradient / max(left_gradient, 1e-12))
    return {
        "central_local_contrast_correlation": correlation,
        "central_gradient_magnitude_ratio": gradient_ratio,
        "scope": "deterministic CXR structure proxy; not a validated lesion segmenter",
    }


def main() -> None:
    implementation.structure_proxy = structure_proxy
    implementation.main()


if __name__ == "__main__":
    main()

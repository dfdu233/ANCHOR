"""Auditable MedHEval + SGTA experiment pipeline.

Legacy experiment scripts are intentionally not imported here: several of them
use incompatible parsers, label tokens, or calibration/test splits.  The
``PROTOCOL_VERSION`` is stored in every new cache and result artifact.
"""

from .protocol import PROTOCOL_VERSION

__all__ = ["PROTOCOL_VERSION"]

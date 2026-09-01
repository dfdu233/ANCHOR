#!/usr/bin/env python3
"""Role-aware expanded-prefix spans for independent attention controls.

The builder consumes provenance *after* multimodal expansion.  It never
infers image or role boundaries from a fixed token count or token position.
Padding and generated tokens are explicitly excluded from the frozen prefix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


SCHEMA_VERSION = "cecd-dynamic-expanded-spans-v1"
ALLOWED_ROLES = frozenset({"system", "image", "user_text"})


class DynamicSpanError(ValueError):
    """Raised when expanded-token provenance cannot define valid spans."""


@dataclass(frozen=True)
class ExpandedPrefixSpans:
    prefix_length: int
    system: tuple[int, ...]
    image: tuple[int, ...]
    user_text: tuple[int, ...]
    prefix_before_image: tuple[int, ...]
    suffix_after_image: tuple[int, ...]
    prefix_before_image_is_true_system: bool

    @property
    def image_start(self) -> int:
        return self.image[0]

    @property
    def image_end(self) -> int:
        return self.image[-1] + 1

    def role_partition(self) -> dict[str, tuple[int, ...]]:
        return {
            "system": self.system,
            "image": self.image,
            "user_text": self.user_text,
        }

    def positional_partition(self) -> dict[str, tuple[int, ...]]:
        return {
            "prefix_before_image": self.prefix_before_image,
            "image": self.image,
            "suffix_after_image": self.suffix_after_image,
        }


def build_expanded_prefix_spans(
    role_provenance: Sequence[str],
    *,
    attention_mask: Sequence[bool | int] | None = None,
    frozen_prefix_length: int | None = None,
) -> ExpandedPrefixSpans:
    """Build disjoint spans from one sample's expanded token provenance.

    ``role_provenance`` contains one role for every expanded sequence slot.
    ``attention_mask`` may contain right padding only.  A shorter
    ``frozen_prefix_length`` excludes generated tokens, but cannot exclude an
    active token inside the image run.
    """

    roles = tuple(str(value) for value in role_provenance)
    if not roles:
        raise DynamicSpanError("expanded sequence cannot be empty")
    if set(roles) - ALLOWED_ROLES:
        raise DynamicSpanError("unknown expanded-token role")

    if attention_mask is None:
        active_length = len(roles)
    else:
        mask = tuple(bool(value) for value in attention_mask)
        if len(mask) != len(roles):
            raise DynamicSpanError("attention mask/provenance length mismatch")
        try:
            first_padding = mask.index(False)
        except ValueError:
            first_padding = len(mask)
        if any(mask[first_padding:]):
            raise DynamicSpanError("only right padding is admissible")
        active_length = first_padding

    if frozen_prefix_length is None:
        prefix_length = active_length
    else:
        if isinstance(frozen_prefix_length, bool) or not isinstance(
            frozen_prefix_length, int
        ):
            raise DynamicSpanError("frozen prefix length must be an integer")
        if not 0 < frozen_prefix_length <= active_length:
            raise DynamicSpanError("frozen prefix length is out of active bounds")
        prefix_length = frozen_prefix_length
    if prefix_length < active_length and roles[prefix_length] == "image":
        raise DynamicSpanError("frozen prefix cannot truncate the expanded image run")

    prefix_roles = roles[:prefix_length]
    image = tuple(i for i, role in enumerate(prefix_roles) if role == "image")
    if not image:
        raise DynamicSpanError("expanded image span must be nonempty")
    expected_image = tuple(range(image[0], image[-1] + 1))
    if image != expected_image:
        raise DynamicSpanError("expanded image tokens must be one contiguous run")

    system = tuple(i for i, role in enumerate(prefix_roles) if role == "system")
    user_text = tuple(i for i, role in enumerate(prefix_roles) if role == "user_text")
    union = set(system) | set(image) | set(user_text)
    if union != set(range(prefix_length)):
        raise DynamicSpanError("role partition is not exhaustive")
    if len(system) + len(image) + len(user_text) != prefix_length:
        raise DynamicSpanError("role partition overlaps")

    before = tuple(range(image[0]))
    after = tuple(range(image[-1] + 1, prefix_length))
    return ExpandedPrefixSpans(
        prefix_length=prefix_length,
        system=system,
        image=image,
        user_text=user_text,
        prefix_before_image=before,
        suffix_after_image=after,
        prefix_before_image_is_true_system=bool(before)
        and all(prefix_roles[index] == "system" for index in before),
    )

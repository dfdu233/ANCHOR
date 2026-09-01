"""Fail-closed runtime adapter for the official VISTA intervention.

The official VISTA release implements visual steering by temporarily wrapping
each decoder MLP and self-logit augmentation by mixing selected intermediate
layer logits.  This adapter reuses the hash-audited official VSV code and adds
the minimal Mistral-compatible SLA wrapper without permanently modifying the
canonical LLaVA-Med model class.

`enabled=False` is the T1 method-off path: it performs no import, forward
replacement, hook registration, or tensor operation.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import sys
import types
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
VISTA_ROOT = REPO_ROOT / "third_party/baselines/VISTA"
OFFICIAL_HASHES = {
    "steering_vector.py": "b118dc1f569b0cfa41b295bb93b6e0e19dc792dc4d01b33e3e00c3433102c473",
    "llm_layers.py": "c554af3e71f939159ab45c79f2155e33d3f9072ecee524a663eb78849fdf00f2",
    "myutils.py": "2ed7914987b62d5d0b1cdf6f132f33ec97b932adada83ed216a33005f693eeb4",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_official_modules() -> tuple[Any, Any]:
    for relative, expected in OFFICIAL_HASHES.items():
        path = VISTA_ROOT / relative
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"official VISTA source identity failed: {path}")

    root = str(VISTA_ROOT)
    inserted = root not in sys.path
    if inserted:
        sys.path.insert(0, root)
    try:
        steering = importlib.import_module("steering_vector")
        layers = importlib.import_module("llm_layers")
    finally:
        if inserted and sys.path and sys.path[0] == root:
            sys.path.pop(0)
    for module in (steering, layers):
        module_path = Path(module.__file__).resolve()
        if VISTA_ROOT.resolve() not in module_path.parents:
            raise RuntimeError(f"unexpected VISTA module origin: {module_path}")
    return steering, layers


def _parse_layer_window(spec: str, layer_count: int) -> list[int]:
    fields = spec.split(",")
    if len(fields) != 2:
        raise ValueError("VISTA SLA layer window must be 'start,end'")
    start, end = (int(value) for value in fields)
    if start < 0 or end < start or end >= layer_count:
        raise ValueError(
            f"VISTA SLA layer window {spec!r} is invalid for {layer_count} layers"
        )
    return list(range(start, end + 1))


class VistaRuntimeAdapter:
    """Apply and completely restore VISTA VSV + SLA around one generation."""

    def __init__(
        self,
        *,
        model: Any,
        enabled: bool,
        enable_vsv: bool = True,
        enable_sla: bool = True,
        negative_kwargs: dict[str, Any] | None = None,
        positive_kwargs: dict[str, Any] | None = None,
        vsv_lambda: float = 0.01,
        vsv_layers: str | None = None,
        logits_layers: str = "25,30",
        logits_alpha: float = 0.3,
    ) -> None:
        self.model = model
        self.enabled = bool(enabled)
        self.enable_vsv = bool(enable_vsv)
        self.enable_sla = bool(enable_sla)
        self.negative_kwargs = negative_kwargs
        self.positive_kwargs = positive_kwargs
        self.vsv_lambda = float(vsv_lambda)
        self.vsv_layers = vsv_layers
        self.logits_layers = logits_layers
        self.logits_alpha = float(logits_alpha)
        self._original_forward = None
        self._layers_module = None
        self._vsv_installed = False
        self._damro_attr_existed = False
        self._damro_attr_value = None
        self._damro_attr_touched = False

    def __enter__(self) -> "VistaRuntimeAdapter":
        if not self.enabled:
            return self
        if not self.enable_vsv and not self.enable_sla:
            raise ValueError("enabled VISTA requires VSV, SLA, or both")
        if self.enable_vsv and (
            self.negative_kwargs is None or self.positive_kwargs is None
        ):
            raise ValueError("VISTA VSV requires negative and positive prompts")
        if not 0.0 <= self.logits_alpha <= 1.0:
            raise ValueError("VISTA logits_alpha must lie in [0, 1]")

        steering, layers_module = _load_official_modules()
        self._layers_module = layers_module
        try:
            if self.enable_vsv:
                # Official VSV first calls `forward` directly to fit its
                # image-specific direction. Supply the canonical fork's
                # missing neutral transient and restore it on exit.
                self._damro_attr_existed = hasattr(self.model, "damro_mask_idx")
                if self._damro_attr_existed:
                    self._damro_attr_value = self.model.damro_mask_idx
                else:
                    self.model.damro_mask_idx = None
                self._damro_attr_touched = True
                direction, _ = steering.obtain_vsv(
                    types.SimpleNamespace(),
                    self.model,
                    [[self.negative_kwargs, self.positive_kwargs]],
                )
                stacked = torch.stack([direction], dim=1).to(self.model.device)
                layers_module.add_vsv_layers(
                    self.model,
                    stacked,
                    [self.vsv_lambda],
                    self.vsv_layers,
                )
                self._vsv_installed = True
            if self.enable_sla:
                self._install_sla()
        except Exception:
            self._restore()
            raise
        return self

    def _install_sla(self) -> None:
        original_forward = self.model.forward
        layer_count = len(self.model.get_model().layers)
        selected = _parse_layer_window(self.logits_layers, layer_count)
        alpha = self.logits_alpha

        def vista_forward(this: Any, *args: Any, **kwargs: Any) -> Any:
            if kwargs.get("labels") is not None:
                raise RuntimeError("VISTA SLA adapter is generation-only")
            kwargs["output_hidden_states"] = True
            kwargs["return_dict"] = True
            outputs = original_forward(*args, **kwargs)
            hidden_states = getattr(outputs, "hidden_states", None)
            if hidden_states is None or len(hidden_states) != layer_count + 1:
                raise RuntimeError(
                    "VISTA SLA requires embedding plus every decoder hidden state"
                )
            decoder_states = hidden_states[1:]
            augmented = torch.stack(
                [this.lm_head(decoder_states[index]) for index in selected]
            ).mean(dim=0)
            outputs.logits = alpha * augmented + (1.0 - alpha) * outputs.logits
            return outputs

        # Transformers validates generation kwargs against the reflected
        # forward signature.  A generic MethodType wrapper would hide
        # `attention_mask` and other legitimate arguments.  Prefix the bound
        # original signature with the synthetic `self` consumed by MethodType.
        original_signature = inspect.signature(original_forward)
        vista_forward.__signature__ = original_signature.replace(
            parameters=[
                inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                *original_signature.parameters.values(),
            ]
        )
        self._original_forward = original_forward
        self.model.forward = types.MethodType(vista_forward, self.model)

    def _restore(self) -> None:
        if self._original_forward is not None:
            self.model.forward = self._original_forward
            self._original_forward = None
        if self._vsv_installed and self._layers_module is not None:
            self._layers_module.remove_vsv_layers(self.model)
            self._vsv_installed = False
        if self._damro_attr_touched:
            if self._damro_attr_existed:
                self.model.damro_mask_idx = self._damro_attr_value
            elif hasattr(self.model, "damro_mask_idx"):
                delattr(self.model, "damro_mask_idx")
            self._damro_attr_touched = False

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self._restore()

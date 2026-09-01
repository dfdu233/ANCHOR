"""Compatibility shim for released LLaVA-1.5 evaluation entry points.

Transformers 4.37 registers its own ``llava`` auto classes, while the released
LLaVA-1.5 repositories register their checkpoint-compatible classes under the
same key and assume an older Transformers release.  Making the two explicit
registrations idempotent restores that historical environment without changing
the official model loader or generation code.
"""

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.bloom import modeling_bloom
from transformers.models.opt import modeling_opt


_config_register = AutoConfig.register
_model_register = AutoModelForCausalLM.register


def _register_config(model_type, config, exist_ok=False):
    return _config_register(model_type, config, exist_ok=True)


def _register_model(config_class, model_class, exist_ok=False):
    return _model_register(config_class, model_class, exist_ok=True)


AutoConfig.register = _register_config
AutoModelForCausalLM.register = _register_model

# The released package imports its unused MPT compatibility module at package
# import time. Transformers 4.37 moved these private helpers to the common mask
# utility; provide the historical names so importing the official LLaVA loader
# does not require editing its model or generation implementation.
for _module in (modeling_bloom, modeling_opt):
    if not hasattr(_module, "_expand_mask"):
        _module._expand_mask = AttentionMaskConverter._expand_mask
    if not hasattr(_module, "_make_causal_mask"):
        _module._make_causal_mask = AttentionMaskConverter._make_causal_mask

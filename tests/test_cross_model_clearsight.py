import torch
from anchor.corrected_sgta.cross_model_clearsight import vaf_attention_logits


def test_vaf_attention_equation():
    logits = torch.ones(1, 1, 3)
    text = torch.tensor([[[True]]])
    visual = torch.tensor([[[True, False, False]]])
    system = torch.tensor([[[False, False, True]]])
    out = vaf_attention_logits(logits, text_query=text, visual_key=visual, system_key=system, alpha=.15, beta=.1)
    assert torch.allclose(out, torch.tensor([[[1.15, 1., .9]]]))


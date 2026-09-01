from pathlib import Path


def test_huatuo_adapter_source_freezes_native_prompt_and_generation_controls() -> None:
    source = Path("anchor/corrected_sgta/models_oe.py").read_text()
    block = source[source.index("class HuatuoOEAdapter") : source.index("class HuluOEAdapter")]
    assert "insert_image_placeholder" in block
    assert '"min_new_tokens": 1' in block
    assert '"repetition_penalty": 1.2' in block
    assert '"return_dict_in_generate": True' in block
    assert '"output_scores": True' in block
    assert "_decode_generations" in block

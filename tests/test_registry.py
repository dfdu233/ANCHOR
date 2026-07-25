from pathlib import Path

from anchor.runners import registry


ROOT = Path(__file__).resolve().parents[1]


def test_default_dataset_configs_validate():
    cfg = registry.load_yaml(ROOT / "configs/datasets.yaml")
    for name in cfg["default"]:
        result = registry.validate_dataset(name, cfg["datasets"][name])
        assert result["ok"], result


def test_method_registry_contains_expected_methods():
    cfg = registry.load_yaml(ROOT / "configs/methods.yaml")
    tasks = registry.method_tasks(cfg)
    for method in ["greedy", "source_margin", "source_word_center", "sca_t_tim_kl", "vcd", "opencode"]:
        assert method in tasks


def test_default_runner_skips_unsupported_combinations():
    cfg = registry.load_yaml(ROOT / "configs/methods.yaml")
    tasks = registry.method_tasks(cfg)
    assert "report_generation" not in tasks["source_margin"]
    assert "vqa_binary" not in tasks["source_word_center"]

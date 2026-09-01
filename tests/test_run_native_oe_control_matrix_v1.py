from anchor.medeval.run_native_oe_control_matrix_v1 import frozen_arms


def test_frozen_arms_reuse_seed42_self_consistency_sample() -> None:
    contract = {
        "generation": {"base_seed": 42, "sampling_seed_ledger": [42, 1042, 2042, 3042, 4042]},
        "temperature_length_controls": {
            "arms": [
                {"id": "greedy128", "decode_mode": "greedy", "max_new_tokens": 128},
                {
                    "id": "sample_t07_p09",
                    "decode_mode": "sample",
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_new_tokens": 256,
                    "seed": 42,
                },
            ]
        },
        "self_consistency": {
            "sampling": {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 256}
        },
    }
    arms = frozen_arms(contract)
    names = [arm.name for arm in arms]
    assert names.count("sample_t07_p09_seed42") == 1
    assert set(names) >= {
        "greedy128",
        "sample_t07_p09_seed42",
        "sample_t07_p09_seed4042",
        "replay_t07_p09_seed42",
    }

from anchor.medeval.hashing import source_tree_fingerprint


def test_source_tree_fingerprint_tracks_paths_and_bytes_not_cache(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "a.pyc").write_bytes(b"ignored")
    first = source_tree_fingerprint(tmp_path)
    (cache / "a.pyc").write_bytes(b"changed but ignored")
    assert source_tree_fingerprint(tmp_path)["sha256"] == first["sha256"]
    (tmp_path / "a.py").write_text("x = 2\n")
    assert source_tree_fingerprint(tmp_path)["sha256"] != first["sha256"]

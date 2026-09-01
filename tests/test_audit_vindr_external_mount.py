import json

from anchor.medeval import audit_vindr_external_mount as module


def test_external_mount_requires_exact_source_set_and_valid_selected_subset(tmp_path, monkeypatch):
    source = tmp_path / "labels.csv"
    source.write_text("image_id,rad_id\na,R8\na,R9\nb,R8\n")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"image_id": "a"}) + "\n")
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.dicom").write_bytes(b"a")
    (images / "b.dicom").write_bytes(b"b")
    mounts = tmp_path / "mounts"
    mounts.write_text(f"device {tmp_path} ext4 ro,relatime 0 0\n")
    monkeypatch.setattr(
        module,
        "inspect_one",
        lambda path: {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": path.name,
            "ok": True,
        },
    )

    result = module.audit(manifest, source, images, workers=1, mounts_path=mounts)

    assert result["passed"] is True
    assert result["source_images"] == 2
    assert result["selected_images"] == 1
    assert result["validated_selected_images"] == 1

    (images / "unexpected.dicom").write_bytes(b"x")
    result = module.audit(manifest, source, images, workers=1, mounts_path=mounts)
    assert result["passed"] is False
    assert result["extra_source"] == ["unexpected.dicom"]

from pathlib import Path

from anchor.medeval import audit_vindr_download as module


def _write_test_dicom(path: Path, pixel_data: bytes, transfer_syntax=None) -> None:
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import DigitalXRayImageStorageForPresentation, ExplicitVRLittleEndian

    transfer_syntax = transfer_syntax or ExplicitVRLittleEndian
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = DigitalXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID = path.stem
    meta.TransferSyntaxUID = transfer_syntax
    dataset = FileDataset(path, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.Rows = 2
    dataset.Columns = 3
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    dataset.PixelData = pixel_data
    if transfer_syntax.is_compressed:
        dataset["PixelData"].is_undefined_length = True
    dataset.save_as(path, enforce_file_format=True)


def test_selective_download_requires_exact_parseable_set(tmp_path, monkeypatch):
    urls = tmp_path / "urls.txt"
    urls.write_text("https://example.test/train/a.dicom\nhttps://example.test/train/b.dicom\n")
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.dicom").write_bytes(b"a")

    monkeypatch.setattr(
        module,
        "inspect_one",
        lambda path: {"name": path.name, "size": path.stat().st_size, "sha256": path.name, "ok": True},
    )
    incomplete = module.audit(urls, images, workers=1)
    assert incomplete["passed"] is False
    assert incomplete["missing"] == ["b.dicom"]

    (images / "b.dicom").write_bytes(b"b")
    complete = module.audit(urls, images, workers=1)
    assert complete["passed"] is True
    assert complete["validated"] == 2
    assert complete["ordered_dicom_content_sha256"]


def test_inspect_accepts_anonymized_vindr_without_dataset_sop_uid(tmp_path):
    path = tmp_path / "1.2.3.4.dicom"
    _write_test_dicom(path, bytes(range(12)))

    result = module.inspect_one(path)

    assert result["ok"] is True
    assert result["pixel_data_bytes"] == 12
    assert result["rows"] == 2
    assert result["columns"] == 3


def test_inspect_rejects_truncated_uncompressed_pixel_data(tmp_path):
    path = tmp_path / "1.2.3.5.dicom"
    _write_test_dicom(path, bytes(range(4)))

    result = module.inspect_one(path)

    assert result["ok"] is False
    assert "PixelData length" in result["error"]


def test_inspect_rejects_structured_but_undecodable_compressed_pixels(tmp_path):
    from pydicom.encaps import encapsulate
    from pydicom.uid import JPEG2000Lossless

    path = tmp_path / "1.2.3.6.dicom"
    _write_test_dicom(
        path,
        encapsulate([b"not-a-jpeg2000-codestream"]),
        transfer_syntax=JPEG2000Lossless,
    )

    result = module.inspect_one(path)

    assert result["ok"] is False

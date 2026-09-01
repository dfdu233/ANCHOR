from anchor.corrected_sgta.evaluate_medheval_answers import _clinical_cluster_id
from anchor.medeval.prepare_baseline_matrix_inputs import patient_or_image_cluster


def test_mimic_shard_is_not_patient_identity() -> None:
    image = "p19/p19454978/s57331547/image.jpg"
    assert patient_or_image_cluster(image) == "19454978"
    assert _clinical_cluster_id({"img_name": image, "patient_id": "19"}) == "19454978"


def test_iu_xray_study_remains_cluster_identity() -> None:
    image = "IU-Xray/CXR3030_IM-1405/0.png"
    assert patient_or_image_cluster(image) == "CXR3030_IM-1405"
    assert _clinical_cluster_id({"img_name": image, "patient_id": "CXR3030_IM-1405"}) == "CXR3030_IM-1405"

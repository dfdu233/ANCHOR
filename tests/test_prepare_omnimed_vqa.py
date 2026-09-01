from anchor.medeval.prepare_omnimed_vqa import answer_label, reference_modality


def test_reference_modality_maps_paper_subsets():
    assert reference_modality("MRI(Magnetic Resonance Imaging)") == "MRI"
    assert reference_modality("MR (Mag-netic Resonance Imaging)") == "MRI"
    assert reference_modality("CT(Computed Tomography)") == "CT"
    assert reference_modality("OCT(Optical Coherence Tomography)") == "OCT"
    assert reference_modality("Fundus Photography") == "Fundus"
    assert reference_modality("Endoscopy") is None


def test_answer_label_accepts_only_exact_unique_option():
    options = [("A", "CT scan"), ("B", "X-ray"), ("C", "MRI"), ("D", "Ultrasound")]
    assert answer_label("C", options) == "C"
    assert answer_label("MRI", options) == "C"

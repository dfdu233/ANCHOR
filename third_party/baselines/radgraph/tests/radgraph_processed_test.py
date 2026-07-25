import json
import os
import pytest

from radgraph import RadGraph, get_radgraph_processed_annotations

def test_processed_annotations_match_expected():
    """
    This test verifies that the current implementation of get_radgraph_processed_annotations 
    produces the same results as previously saved.
    """
    # Initialize RadGraph
    model_type = "radgraph-xl"
    radgraph = RadGraph(model_type=model_type)
    
    # Load expected results
    expected_path = os.path.join(os.path.dirname(__file__), 'expected_processed_annotations.json')
    with open(expected_path, 'r') as f:
        expected_results = json.load(f)
    
    # Hardcoded reports
    reports = [
        "The lungs are clear and the costophrenic angles are sharp. There is no pneumonia, pulmonary edema, pleural effusion, or pneumothorax.",
        "Mild dextrocurvature of the midthoracic spine.",
        "The lungs are clear, without evidence of focal consolidation or pleural effusion. Cardiomediastinal silhouette is within normal limits. Visualized osseous and soft tissue structures are unremarkable.",
        "There are scattered areas of fibroglandular density.",
        "The trachea is in the midline. The bony thorax is normal for the patient's age. The cardiomediastinal silhouette is unremarkable and the lung zones are clear. The diaphragm has a normal contour."
    ]
    
    # Test each report
    for expected in expected_results:
        report_id = expected['report_id']
        report_text = expected['report_text']
        
        # Verify report text matches the hardcoded one
        assert report_text == reports[report_id - 1], f"Report text mismatch for report {report_id}"
        
        # Get current annotations
        annotations = radgraph([report_text])
        current_processed = get_radgraph_processed_annotations(annotations)

        # Verify processed annotations match expected
        assert current_processed == expected['processed_annotations'], \
            f"Processed annotations mismatch for report {report_id}"

  
if __name__ == "__main__":
    print("This module is intended to be run with pytest:")
    print("pytest -xvs tests/radgraph_processed_test.py")
    print("\nTo generate expected outputs first, uncomment the @pytest.mark.skip line") 
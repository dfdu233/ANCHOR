<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# Conformal Generation

This project implements conformal generation, as described in [*Conf-Gen: Conformal Uncertainty Quantification for Generative Models*](https://arxiv.org/pdf/2605.28920). The main class, `ConformalGeneration`, provides methods for generating conformal sequences and calibrating parameters based on a calibration dataset.

## Installation

To install the package, you can use pip. First, clone the repository:

```bash
git clone https://github.com/layer6ai-labs/conformal_generation.git
cd conformal_generation
```

### (Optional) Use pyenv

```bash
pyenv virtualenv 3.11.6 conformal_generation
pyenv activate conformal_generation
```

Then, install the package using pip:

```bash
pip install .
```

## Usage

After installation, you can use the `ConformalGeneration` class in your Python scripts:

```python
# import the sequence selector, calibration dataset, and conformal generation classes of your choice
from conformal_generation.sequence_selector import RunningMaxSequenceSelector
from conformal_generation.calibration_dataset import IndividualScoreCalibrationDataset
from conformal_generation.conformal_generation import ConformalGeneration

# Initialize the classes for sequence selector, calibration dataset, and conformal generation
# Please refer to the notebooks in our `example` folder for specific usage details
my_sequence_selector = RunningMaxSequenceSelector(score_fn=lambda x, y: (x - y) ** 2)
my_calibration_dataset = IndividualScoreCalibrationDataset(
    sequence_selector=my_sequence_selector,
    input_dataset=x_train,
    raw_generated_dataset=y_train,
    admissibility_dataset=a_train,
    admissibility_aggregation=lambda a_list: max(a_list) if len(a_list) > 0 else 1,
)
CG = ConformalGeneration(sequence_selector=my_sequence_selector, calibration_dataset=my_calibration_dataset)

# Generate conformal sequences
selected = CG.select(instance=x, raw_generated_sequence=y)
```

See the `example` directory for additional examples reproducing the figures in our paper.

## Testing

To run the tests, navigate to the project directory and execute:

```bash
pytest tests/
```

## Note

There is a known issue where unexpected behaviour happens when the score function takes the same value for various entries within the same sequence. This will be updated in the future, but to avoid this behaviour you can simply add a small amount of Gaussian noise to the score function to avoid ties.

## Citing

If you use any part of this repository in your research, please cite the associated paper with the following bibtex entry:

Authors: Gabriel Loaiza-Ganem, Kevin Zhang, Wei Cui, Marc T. Law, Kin Kwan Leung

```
@inproceedings{loaiza2026conf,
  title={Conf-Gen: Conformal Uncertainty Quantification for Generative Models},
  author={Loaiza-Ganem, Gabriel and Zhang, Kevin and Cui, Wei and Law, Marc T and Leung, Kin Kwan},
  booktitle={International Conference on Machine Learning},
  year={2026}
}
```

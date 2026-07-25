import pathlib

import numpy as np
import openml as openml
from sklearn.preprocessing import LabelEncoder

NAME_TID_MAP = {
    "adult": 7592,
    "Census-Income": 168340,
    "Click_prediction_small": 190408,
    "GesturePhaseSegmentationProcessed": 14969,
    "MiniBooNE": 168335,
}


class OpenMLDataset:
    def __init__(self, name: str, path: pathlib.Path = pathlib.Path(".cache/")):
        openml_task_id = NAME_TID_MAP.get(name)
        if openml_task_id is None:
            raise ValueError(f"Unknown OpenML dataset {name}.")
        dataset_path = path / "openml" / name
        if dataset_path.exists():
            # read from file.
            self.X, self.y, self.train_indices, self.test_indices, self.feature_names = self.read(
                dataset_path
            )
        else:
            dataset_path.mkdir(exist_ok=False, parents=True)
            (
                self.X,
                self.y,
                self.train_indices,
                self.test_indices,
                self.feature_names,
            ) = self.load_and_write(dataset_path, openml_task_id)

    @staticmethod
    def read(path: pathlib.Path):
        X = np.load(str(path / "X.npy"), allow_pickle=True)
        y = np.load(str(path / "y.npy"), allow_pickle=True)
        train_indices = np.load(str(path / "train_indices.npy"), allow_pickle=True)
        test_indices = np.load(str(path / "test_indices.npy"), allow_pickle=True)
        feature_names = np.load(str(path / "feature_names.npy"), allow_pickle=True)
        return X, y, train_indices, test_indices, feature_names

    @staticmethod
    def load_and_write(path: pathlib.Path, openml_task_id: int):
        task = openml.tasks.get_task(task_id=openml_task_id)
        train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0)
        dataset = task.get_dataset()
        X, y, categorical_indicator, feature_names = dataset.get_data(
            dataset_format="dataframe",
            target=task.target_name,
        )
        feature_names = np.asarray(feature_names)
        X = X.to_numpy()
        cat_indices = [idx for idx, indicator in enumerate(categorical_indicator) if indicator]
        y = LabelEncoder().fit_transform(y.to_numpy())
        for cat_index in cat_indices:
            X[:, cat_index] = LabelEncoder().fit_transform(X[:, cat_index])
        X = X.astype(float)
        y = y.astype(int)
        np.save(str(path / "X.npy"), X)
        np.save(str(path / "y.npy"), y)
        np.save(str(path / "train_indices.npy"), train_indices)
        np.save(str(path / "test_indices.npy"), test_indices)
        np.save(str(path / "feature_names.npy"), feature_names)
        return X, y, train_indices, test_indices, feature_names

    def train_instances(self):
        return self.X[self.train_indices], self.y[self.train_indices]

    def test_instances(self):
        return self.X[self.test_indices], self.y[self.test_indices]

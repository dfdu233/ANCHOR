import logging
import pathlib
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import KFold

import tqdm as tqdm
from sklearn.ensemble import RandomForestClassifier

from example.tree.decisiontree import DecisionTree
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from example.tree.openmldataset import OpenMLDataset
from src import (
    IndividualScoreCalibrationDataset,
    RunningSmallestSubsetSumSequenceSelector,
    SmallestSubsetSumSequenceSelector,
)
from src.conformal_generation import ConformalGeneration


def score_function(x, y):
    # y = (preds, weighted_sample)
    return y[1]


def admissibility_func(pred, label):
    return bool(np.argmax(pred) == label)


def aggregate_prediction_function(preds) -> List[float]:
    return np.mean(np.asarray([pred for pred, _ in preds]), axis=0).tolist()


class ConformalDecisionTree:
    def __init__(
        self,
        dataset_name: str,
        calibration_size: int = 100,
        weighted: bool = True,
        running: bool = True,
        cross_validation: bool = False,
        max_num_folds: int = 10,
        cal_split_seed: int = 0,
    ):
        """
        Constructor

        Args:
            dataset_name:
                The name of the tabular dataset
            calibration_size:
                The size of the calibration dataset
            weighted:
                If True, the score function is the weighted number of samples in the leaf
                with the prediction. If False, it is the unweighted number of samples.
            running:
                If True, the RunningSmallestSubsetSumSequenceSelector is used. If False,
                the SmallestSubsetSumSequenceSelector is used.
            cross_validation:
                Whether we use cross validations for conformal generation.
            max_num_folds:
                The maximum number of folds to calculate.
            cal_split_seed:
                The seed on calling KFold on the splitting the original test set into calibration
                set and the new test set.
        """
        dataset = OpenMLDataset(dataset_name)
        train, train_label = dataset.train_instances()
        test, test_label = dataset.test_instances()
        self.train = train
        self.test = test
        self.train_label = train_label
        self.test_label = test_label
        self.dataset_name = dataset_name
        self.weighted = weighted
        if running:
            self.sequence_selector = RunningSmallestSubsetSumSequenceSelector(score_function)
        else:
            self.sequence_selector = SmallestSubsetSumSequenceSelector(score_function)

        self.calibration_size = calibration_size
        kfold_gen = KFold(
            n_splits=len(self.test) // calibration_size,
            shuffle=True,
            random_state=cal_split_seed,
        ).split(self.test)
        if cross_validation:
            self.test_cal_indices_list = list(kfold_gen)[:max_num_folds]
        else:
            self.test_cal_indices_list = [next(kfold_gen)]
        self._admissibility_func = admissibility_func
        self._aggregate_prediction_function = aggregate_prediction_function

        self._all_trees_predictions = None
        self._all_admissibilities = None
        self._num_trees = None
        self._num_classes = None
        self.log = logging.getLogger(ConformalDecisionTree.__name__)
        self.performances_per_fold = self._evaluate_performances()

    def _get_auc_acc(self, test_labels: List[int], test_preds: np.ndarray) -> Tuple[float, float]:
        """
        Compute the AUC and ACC.

        Args:
            test_labels:
                The provided labels
            test_preds:
                The predictions of shape (num_samples, num_classes). If binary, num_classes=2.

        Returns:
            The AUC-ROC (with macro and one-vs-one) and the Accuracy.

        """
        if self.num_classes == 2:
            test_preds = test_preds[:, 1]
            cls_preds = np.round(test_preds)
        else:
            cls_preds = np.argmax(test_preds, axis=1)
        auc = float(roc_auc_score(test_labels, test_preds, multi_class="ovo"))
        acc = float(np.mean(test_labels == cls_preds))
        return auc, acc

    def _evaluate_performances(self) -> List[Dict[str, float]]:
        """
        Compute the AUC and ACC for all folds and output it in the log. Return a list of dicts
        with keys "auc" and "acc".
        """
        performances = []
        for fold_index, (test_indices, cal_indices) in enumerate(self.test_cal_indices_list):
            test_trees_predictions = [self.all_trees_predictions[i] for i in test_indices]
            test_preds = np.asarray(
                [self._aggregate_prediction_function(preds) for preds in test_trees_predictions]
            )
            test_label = [self.test_label[i] for i in test_indices]
            auc, acc = self._get_auc_acc(test_label, test_preds)
            performances.append({"auc": auc, "acc": acc})
            self.log.info(f"Fold {fold_index}. Base AUC = {auc}, Base ACC = {acc}")
        return performances

    def train_model(self, retrain: bool = True) -> None:
        """
        Train a random forest model and save the decision tree classifiers as {DATASET_NAME}.pkl.
        """
        file_path = pathlib.Path(f"{dataset_name}.pkl")
        if file_path.exists() and not retrain:
            return
        self.log.info("Begin training model...")
        clf = RandomForestClassifier(n_estimators=100, random_state=1)
        clf.fit(self.train, self.train_label)

        y_test = clf.predict_proba(self.test)
        with open(file_path, "wb") as f:
            pkl.dump(clf.estimators_, f)

        auc, acc = self._get_auc_acc(self.test_label, y_test)
        self.log.info(f"{auc=}, {acc=}. Saved model to {file_path}")

    def _get_all_predictions(
        self,
        ests: List[DecisionTreeClassifier],
        data: np.ndarray,
    ) -> List[List[Tuple[List[float], float]]]:
        # Output = array of shape (num_data, num_trees), prediction of each tree on each data.
        # sklearn predict decision path is buggy when the value is nan sometimes. So we include
        # this decision tree class.
        trees = [DecisionTree(est.tree_) for est in ests]

        all_leaves_cal = [tree.predict_leaf(data) for tree in trees]

        if self.weighted:
            all_weighted_samples_cal = np.asarray(
                [est.tree_.weighted_n_node_samples[leaf] for est, leaf in zip(ests, all_leaves_cal)]
            )
        else:
            all_weighted_samples_cal = np.asarray(
                [est.tree_.n_node_samples[leaf] for est, leaf in zip(ests, all_leaves_cal)]
            )
        all_preds_cal = np.asarray(
            [est.tree_.value[:, 0, :][leaf] for est, leaf in zip(ests, all_leaves_cal)]
        )
        y_cal = [
            [
                (pred[i].tolist(), float(w[i]))
                for pred, w in zip(all_preds_cal, all_weighted_samples_cal)
            ]
            for i in range(len(data))
        ]
        return y_cal

    @property
    def all_trees_predictions(self) -> List[List[Tuple[List[float], float]]]:
        """
        Compute all the predictions for each sample and each tree.

        Returns:
            A list of lists of tuple. For each sample, for each tree, this returns a tuple of
            2 elements. The first is the prediction. The prediction is a list of float
            corresponding to the probabilities of each class. The second is the weighted samples
            for the predicted leaf.

        """
        if self._all_trees_predictions is None:
            self.train_model(retrain=False)
            with open(f"{self.dataset_name}.pkl", "rb") as f:
                ests = pkl.load(f)
                self._num_trees = len(ests)
                self._num_classes = ests[0].n_classes_
            self._all_trees_predictions = self._get_all_predictions(ests, self.test)
        return self._all_trees_predictions

    @property
    def num_trees(self) -> int:
        if self._num_trees is None:
            with open(f"{self.dataset_name}.pkl", "rb") as f:
                ests = pkl.load(f)
                self._num_trees = len(ests)
        return self._num_trees

    @property
    def num_classes(self) -> int:
        if self._num_classes is None:
            with open(f"{self.dataset_name}.pkl", "rb") as f:
                ests = pkl.load(f)
                self._num_classes = ests[0].n_classes_
        return self._num_classes

    @property
    def all_admissibilities(self) -> List[List[bool]]:
        """
        Returns a list of list (of shape (num_samples, num_trees) of admissibilities (bool)
        """
        if self._all_admissibilities is None:
            all_trees_predictions = self.all_trees_predictions
            self._all_admissibilities = [
                [
                    self._admissibility_func(tree_pred, instance_label)
                    for tree_pred, _ in instance_preds
                ]
                for instance_preds, instance_label in zip(all_trees_predictions, self.test_label)
            ]
        return self._all_admissibilities

    def calibrate_and_generate(self, admissible_k: int) -> pd.DataFrame:
        """
        Perform Conformal Generation by calibrating and evaluating.

        Args:
            admissible_k:
                Admissible meaning the tree is predicting the correct class. This aggregation means
                the selected sequence (trees) is admissible if at least "admissible_k" trees
                are predicting the correct class.

        Returns:
            A pandas dataframe containing the results. Columns are "gamma", "fold", "new AUC",
            "old AUC", "AUC diff", "new ACC", "old ACC", "ACC diff", "Avg Tree Used", "exp. adm.",
            "exp. cal adm." and "satisfy". Expected admissibility is the average admissibility
            over the test set. It is expected to be greater than gamma. If yes, "satisfy" will
            be True. Otherwise, it will be False.

        """

        def admissibility_aggregation(li):
            return sum(li) >= admissible_k

        res = []
        for fold_index, (test_indices, cal_indices) in enumerate(self.test_cal_indices_list):
            self.log.info(f"Beginning calibrating fold {fold_index}...")
            cal_admissibility = [self.all_admissibilities[i] for i in cal_indices]
            cal_predictions = [self.all_trees_predictions[i] for i in cal_indices]
            test_predictions = [self.all_trees_predictions[i] for i in test_indices]
            cal_data = [self.test[i] for i in cal_indices]
            test_data = [self.test[i] for i in test_indices]
            calibration_dataset = IndividualScoreCalibrationDataset(
                sequence_selector=self.sequence_selector,
                input_dataset=cal_data,
                raw_generated_dataset=cal_predictions,
                admissibility_dataset=cal_admissibility,
                admissibility_aggregation=admissibility_aggregation,
            )
            calibration_dataset.validate_admissibility_function_right_continuous()
            calibration_dataset.validate_admissibility_function_non_decreasing()
            calibration_dataset.validate_average_admissibility_function_non_decreasing()

            test_label = [self.test_label[i] for i in test_indices]
            cal_label = [self.test_label[i] for i in cal_indices]
            old_auc = self.performances_per_fold[fold_index]["auc"]
            old_acc = self.performances_per_fold[fold_index]["acc"]

            for gamma in tqdm.tqdm(np.arange(0, 1, 0.05), leave=False):
                conformal_generation = ConformalGeneration(calibration_dataset=calibration_dataset)
                conformal_generation.calibrate(gamma=gamma)
                if conformal_generation.conformal_threshold == float("inf"):
                    # We skip with the conformal threshold is inf
                    dic = {
                        "gamma": gamma,
                        "fold": fold_index,
                    }
                    res.append(dic)
                    continue

                new_preds = [
                    conformal_generation.select(x, y) for x, y in zip(test_data, test_predictions)
                ]
                cal_preds = [
                    conformal_generation.select(x, y) for x, y in zip(cal_data, cal_predictions)
                ]
                new_predictions = np.asarray(
                    [
                        self._aggregate_prediction_function(selected_preds)
                        for selected_preds in new_preds
                    ]
                )
                new_admissibility = [
                    float(
                        admissibility_aggregation(
                            [
                                self._admissibility_func(pred, instance_label)
                                for pred, _ in selected_preds
                            ]
                        )
                    )
                    for selected_preds, instance_label in zip(new_preds, test_label)
                ]
                cal_admissibility = [
                    float(
                        admissibility_aggregation(
                            [
                                self._admissibility_func(pred, instance_label)
                                for pred, _ in selected_preds
                            ]
                        )
                    )
                    for selected_preds, instance_label in zip(cal_preds, cal_label)
                ]
                expected_new_admissibility = np.mean(new_admissibility)
                expected_cal_admissibility = np.mean(cal_admissibility)
                lens = np.mean([len(ps) for ps in new_preds])
                new_auc, new_acc = self._get_auc_acc(test_label, new_predictions)
                dic = {
                    "gamma": gamma,
                    "fold": fold_index,
                    "new AUC": new_auc,
                    "old AUC": old_auc,
                    "AUC diff": new_auc - old_auc,
                    "old ACC": old_acc,
                    "new ACC": new_acc,
                    "ACC diff": new_acc - old_acc,
                    "Avg Tree Used": lens,
                    "exp. adm.": expected_new_admissibility,
                    "exp. cal adm.": expected_cal_admissibility,
                    "satisfy": expected_new_admissibility >= gamma,
                }
                res.append(dic)
        frame = pd.DataFrame.from_records(res)
        return frame


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s,%(msecs)03d] %(name)20s | %(levelname)s | %(message)s",
        level=logging.INFO,
    )
    # Disable logger as number of K and datasets are large enough.
    logging.getLogger("IndividualScoreCalibrationDataset").disabled = True
    log = logging.getLogger("Main")

    dataset_names = [
        "GesturePhaseSegmentationProcessed",  # num classes = 5
        "Click_prediction_small",
        "adult",
        "Census-Income",
        "MiniBooNE",
    ]

    admissible_ks = [1] + list(range(5, 51, 5))
    dfs = []
    for dataset_name in dataset_names:
        conformal = ConformalDecisionTree(
            dataset_name,
            cross_validation=False,
            running=False,
            calibration_size=100,
            cal_split_seed=3786,
        )
        conformal.train_model(retrain=False)

        for admissible_k in admissible_ks:
            log.info(f"{dataset_name=}, {admissible_k=}")
            df = conformal.calibrate_and_generate(
                admissible_k=admissible_k,
            )
            df["admissible_k"] = admissible_k
            df["dataset"] = dataset_name
            dfs.append(df)
    all_df = pd.concat(dfs, axis=0)
    all_df.to_csv(f"all_results.csv")

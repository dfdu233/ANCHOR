"""
This module contains ``DecisionTree``. It provides a method of parsing the decision tree
from a sklearn Decision Tree.

The reason for needing this is that there is currently a possible bug in getting the decision
path if some features are NaN, i.e. missing. Hence we wrote our own "predict leaf"
function - which should be slower but more accurate.
"""

from typing import List

import numpy as np
from sklearn.tree._tree import Tree


__all__ = ["DecisionTree"]

MISSING = -1


class DecisionTree:
    def __init__(
        self,
        tree: Tree,
        feature_names: List[str] = None,
        dtype: np.dtype | type | str = np.float32,
    ):
        self.feature_count = len(feature_names) if feature_names is not None else tree.n_features

        self.yes_nodes: np.ndarray | None = None
        self.no_nodes: np.ndarray | None = None
        self.missing_nodes: np.ndarray | None = None
        self.node_covers: np.ndarray | None = None
        self.decision_values: np.ndarray | None = None
        self.values: np.ndarray | None = None
        self.feature_ids: np.ndarray | None = None
        self.parent_nodes: np.ndarray | None = None
        self.impurity: np.ndarray | None = None
        self.feature_covers: np.ndarray | None = None

        self.num_leaves: int | None = None
        self.num_decision_nodes: int | None = None

        if feature_names is None:
            self.feature_names: List[str] = [f"f{i}" for i in range(self.feature_count)]
        else:
            if len(feature_names) != self.feature_count:
                raise ValueError(
                    f"feature names length ({len(feature_names)}) is different from feature count "
                    f"({self.feature_count})."
                )
            self.feature_names: List[str] = feature_names
        self.dtype: np.dtype = np.dtype(dtype)
        self._parse_tree(tree)

        self.depth: int = self.get_depth(0)
        self._self_cover: np.ndarray | None = None
        self._self_value: np.ndarray | None = None

    def predict_leaf(self, features: np.ndarray) -> np.ndarray:
        res = []
        for feature in features:
            node_id = 0
            while self.is_decision_node(node_id):
                if feature[self.feature_ids[node_id]] <= self.decision_values[node_id]:
                    node_id = self.yes_nodes[node_id]
                elif feature[self.feature_ids[node_id]] > self.decision_values[node_id]:
                    node_id = self.no_nodes[node_id]
                else:
                    node_id = self.missing_nodes[node_id]

            # node_id is leaf node
            res.append(node_id)
        return np.asarray(res, dtype=np.int32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        leaves = self.predict_leaf(features)
        return self.values[leaves]

    def _parse_tree(self, tree: Tree) -> None:
        num_nodes = tree.node_count
        num_features = tree.n_features
        if self.feature_count != num_features:
            raise ValueError(
                f"Feature count provided ({self.feature_count}) does not match the"
                f" num features in the json string ({num_features})."
            )
        self.yes_nodes = np.asarray(tree.children_left, dtype=np.int32)
        self.yes_nodes = np.where(self.yes_nodes == -1, MISSING, self.yes_nodes)
        self.no_nodes = np.asarray(tree.children_right, dtype=np.int32)
        self.no_nodes = np.where(self.no_nodes == -1, MISSING, self.no_nodes)
        decision_nodes = self.yes_nodes != MISSING

        default_left = np.asarray(tree.missing_go_to_left, dtype=np.int32)
        self.missing_nodes = np.full(num_nodes, fill_value=MISSING, dtype=np.int32)
        self.missing_nodes[default_left == 0] = self.no_nodes[default_left == 0]
        self.missing_nodes[default_left == 1] = self.yes_nodes[default_left == 1]

        self.feature_ids = np.asarray(tree.feature, dtype=np.int32)
        leaf_nodes = (~decision_nodes) & (self.feature_ids < num_features)
        self.feature_ids[~decision_nodes] = MISSING

        self.node_covers = np.asarray(tree.n_node_samples, dtype=self.dtype)
        pruned_nodes = ~(decision_nodes | leaf_nodes)
        self.node_covers[pruned_nodes] = np.nan

        split_conditions = np.asarray(tree.threshold, dtype=self.dtype)
        self.decision_values = np.where(decision_nodes, split_conditions, np.nan)
        self.values = np.where(leaf_nodes[:, None], tree.value[:, 0, :], np.nan)

        self.parent_nodes = np.zeros(len(self.yes_nodes), dtype=np.int32)
        self.parent_nodes[self.yes_nodes[~leaf_nodes]] = np.arange(num_nodes)[~leaf_nodes]
        self.parent_nodes[self.no_nodes[~leaf_nodes]] = np.arange(num_nodes)[~leaf_nodes]
        self.parent_nodes[(self.parent_nodes >= num_nodes) | pruned_nodes] = MISSING

        num_leaves = np.sum(leaf_nodes)
        if num_leaves != tree.n_leaves:
            raise ValueError(f"{num_leaves=}, {tree.n_leaves=}")
        self.num_leaves = num_leaves
        self.num_decision_nodes = np.sum(decision_nodes)

        self.impurity = np.asarray(tree.impurity, dtype=self.dtype)

    def is_decision_node(self, node_id: int) -> bool:
        """
        Returns whether a node is a decision node or not.
        """
        if node_id < 0 or node_id >= len(self.yes_nodes):
            raise ValueError(f"Node {node_id} not found.")
        return self.yes_nodes[node_id] != MISSING

    def is_leaf_node(self, node_id: int) -> bool:
        """
        Returns whether a node is a leaf node or not.
        """
        if node_id < 0 or node_id >= len(self.values):
            raise ValueError(f"Node {node_id} not found.")
        return not np.isnan(self.values[node_id][0])

    def get_depth(self, node_id: int = 0) -> int:
        """
        Return the depth of the subtree indexed by the node_id, plus the current depth.

        Args:
            node_id:
              The root node index of the subtree. Default is 0, which represent the tree depth.

        """
        if self.is_leaf_node(node_id):
            return 0

        if self.is_decision_node(node_id):
            left = self.get_depth(self.yes_nodes[node_id])
            right = self.get_depth(self.no_nodes[node_id])
            return 1 + max(left, right)

        raise ValueError(f"node {node_id} not found")

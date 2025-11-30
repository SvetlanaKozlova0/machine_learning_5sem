import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):

    feature_values = np.array(feature_vector)
    target_labels = np.array(target_vector)

    sorted_indices = np.argsort(feature_values)
    sorted_features = feature_values[sorted_indices]
    sorted_targets = target_labels[sorted_indices]

    feature_differences = np.diff(sorted_features)
    unique_change_indices = np.where(feature_differences > 0)[0]

    if len(unique_change_indices) == 0:
        return None, None, None, None

    potential_thresholds = (
        sorted_features[unique_change_indices] + sorted_features[unique_change_indices + 1]) / 2.0

    cumulative_class1 = np.cumsum(sorted_targets)
    total_class1_count = cumulative_class1[-1]

    total_samples = len(target_vector)

    left_subtree_sizes = unique_change_indices + 1
    right_subtree_sizes = total_samples - left_subtree_sizes

    class1_left_counts = cumulative_class1[unique_change_indices]
    class1_right_counts = total_class1_count - class1_left_counts

    class1_ratio_left = class1_left_counts / left_subtree_sizes
    class1_ratio_right = class1_right_counts / right_subtree_sizes

    gini_left = 1 - class1_ratio_left ** 2 - (1 - class1_ratio_left) ** 2
    gini_right = 1 - class1_ratio_right ** 2 - (1 - class1_ratio_right) ** 2

    weighted_gini_scores = -(left_subtree_sizes / total_samples * gini_left +
                             right_subtree_sizes / total_samples * gini_right)

    best_threshold_index = np.argmax(weighted_gini_scores)

    best_threshold = potential_thresholds[best_threshold_index]

    best_gini_score = weighted_gini_scores[best_threshold_index]

    return potential_thresholds, weighted_gini_scores, best_threshold, best_gini_score


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def get_params(self, deep=True):
        return {"feature_types": self._feature_types,
                "max_depth": self._max_depth,
                "min_samples_split": self._min_samples_split,
                "min_samples_leaf": self._min_samples_leaf}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _fit_node(self, sub_X, sub_y, node, depth):

        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and self._max_depth <= depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y,).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y,).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature].astype(float)

            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}

                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0

                    ratio[key] = current_click / current_count

                sorted_categories = list(
                    map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))

                categories_map = dict(
                    zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(
                    list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini is None:
                continue

            if gini_best is None or gini > gini_best:

                split = feature_vector < threshold

                left_length = np.sum(split)
                right_length = len(sub_y) - left_length

                if self._min_samples_leaf is not None and (self._min_samples_leaf > left_length or self._min_samples_leaf > right_length):
                    continue

                feature_best, gini_best = feature, gini

                if feature_type == "real":
                    threshold_best = threshold

                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0], filter(
                        lambda x: x[1] < threshold, categories_map.items())))

                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}

        self._fit_node(sub_X[split], sub_y[split],
                       node["left_child"], depth=depth + 1)

        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(
            split)], node["right_child"], depth=depth + 1)

    def _predict_node(self, x, node):

        if node['type'] == 'terminal':
            return node['class']

        split = node['feature_split']

        if self._feature_types[split] == 'real':
            threshold = node['threshold']

            if x[split] < threshold:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])

        elif self._feature_types[split] == "categorical":

            category = x[split]
            categ_split = node['categories_split']

            if category in categ_split:
                return self._predict_node(x, node['left_child'])
            else:
                return self._predict_node(x, node['right_child'])

        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree, 0)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

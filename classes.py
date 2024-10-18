import numpy as np
import pandas as pd
import random
from joblib import Parallel, delayed

# Custom function for one-hot encoding
def one_hot_encoding(df):
    encoded_df = pd.DataFrame()
    for column in df.columns:
        if df[column].dtype == 'object':
            unique_values = df[column].unique()
            for value in unique_values:
                encoded_df[f"{column}_{value}"] = (df[column] == value).astype(int)
        else:
            encoded_df[column] = df[column]
    return encoded_df

# Custom function for label encoding
def custom_label_encoding(target):
    unique_labels = np.unique(target)
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    encoded_target = target.map(label_to_int)
    return encoded_target, label_to_int

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, is_leaf=False, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.prediction = prediction

    def predict(self, x):
        if self.is_leaf:
            return self.prediction
        if x[self.feature] <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def get_params(self, deep=True):
        return {
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'criterion': self.criterion
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def _gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _entropy(self, y):
        m = len(y)
        return -sum((np.sum(y == c) / m) * np.log2(np.sum(y == c) / m) for c in np.unique(y))

    def _misclassification_error(self, y):
        m = len(y)
        most_common_label_count = np.max(np.bincount(y))
        return 1 - most_common_label_count / m

    def _best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_score = float('inf')
        m, n = X.shape

        for feature in range(n):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                if sum(left_mask) < self.min_samples_split or sum(right_mask) < self.min_samples_split:
                    continue

                if self.criterion == 'gini':
                    score = (sum(left_mask) * self._gini(y[left_mask]) +
                             sum(right_mask) * self._gini(y[right_mask])) / m
                elif self.criterion == 'entropy':
                    score = (sum(left_mask) * self._entropy(y[left_mask]) +
                             sum(right_mask) * self._entropy(y[right_mask])) / m
                elif self.criterion == 'misclassification_error':
                    score = (sum(left_mask) * self._misclassification_error(y[left_mask]) +
                             sum(right_mask) * self._misclassification_error(y[right_mask])) / m

                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        if num_labels == 1 or depth == self.max_depth or num_samples < self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return TreeNode(is_leaf=True, prediction=leaf_value)

        feature, threshold = self._best_split(X, y)

        if feature is None:
            leaf_value = self._most_common_label(y)
            return TreeNode(is_leaf=True, prediction=leaf_value)

        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature, threshold, left_child, right_child)

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self.root.predict(x) for x in X])

    def evaluate(self, X, y):
        predictions = self.predict(X)
        zero_one_loss = np.sum(predictions != y) / len(y)
        return zero_one_loss

def randomized_search_cv(tree, param_grid, X_train, y_train, n_iter=10, cv=5, scoring='accuracy', random_state=None, n_jobs=-1):
    best_score = -1
    best_params = None

    if random_state:
        np.random.seed(random_state)

    def evaluate_fold(params, i):
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = custom_train_test_split(X_train, y_train, test_size=1/cv, random_state=i)
        tree.set_params(**params)
        tree.fit(X_train_fold, y_train_fold)
        y_pred = tree.predict(X_val_fold)
        score = np.mean(y_val_fold == y_pred)
        return score

    for _ in range(n_iter):
        params = {key: random.choice(values) for key, values in param_grid.items()}
        scores = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_fold)(params, i) for i in range(cv)
        )
        mean_score = np.mean(scores)

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    return best_params, best_score

def custom_train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    if isinstance(X, pd.DataFrame):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]

    if isinstance(y, pd.Series):
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
    else:
        y_train = y[train_indices]
        y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp, tn, fp, fn

def precision_score(tp, fp):
    if (tp + fp) == 0:
        return 0.0
    return tp / (tp + fp)

def recall_score(tp, fn):
    if (tp + fn) == 0:
        return 0.0
    return tp / (tp + fn)

def f1_score(precision, recall):
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_model(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    precision = precision_score(tp, fp)
    recall = recall_score(tp, fn)
    f1 = f1_score(precision, recall)
    accuracy = np.mean(y_true == y_pred)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
    }

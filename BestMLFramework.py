import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier


class KNN():
    def __init__(self, nbrs=5):
        self.nbrs = nbrs

    def fit(self, X_t, y_t):
        self.X_t = X_t.to_numpy()
        self.y_t = y_t.to_numpy()

    def dists(self, X_p):
        t = np.dot(X_p, self.X_t.transpose())
        dists = np.sqrt(-2 * t + np.square(self.X_t).sum(1) + np.matrix(np.square(X_p).sum(1)).T)
        return dists

    def predict(self, X_p):
        dists = self.dists(X_p.to_numpy())
        preds = np.zeros(dists.shape[0])
        for i in range(dists.shape[0]):
            labels = self.y_t[np.argsort(dists[i,:])].flatten()
            top_nn_y = labels[:self.nbrs]
            preds[i] = Counter(top_nn_y).most_common(1)[0][0]
        return preds


class NBC():
    def __init__(self):
        pass

    def fit(self, X_t, y_t):
        X_t = X_t.to_numpy()
        y_t = y_t.to_numpy()
        self.num_of_classes = np.max(y_t) + 1
        self.priorities = np.bincount(y_t) / len(y_t)
        self.Ms = np.array([X_t[np.where(y_t == i)].mean(axis=0) for i in range(self.num_of_classes)])
        self.stds = np.array([X_t[np.where(y_t == i)].std(axis=0) for i in range(self.num_of_classes)])
        return self

    def predict(self, X_p):
        X_p = X_p.to_numpy()
        res = []
        for i in range(len(X_p)):
            Ps = []
            for j in range(self.num_of_classes):
                Ps.append((1 / np.sqrt(2 * np.pi * self.stds[j]**2) * np.exp(-0.5*((X_p[i] - self.Ms[j]) / self.stds[j])**2)).prod() * self.priorities[j])
            Ps = np.array(Ps)
            res.append(Ps / Ps.sum())
        return np.array(res).argmax(axis=1)


class LR():
    def __init__(self, lr=0.01, bs=1, steps=5000):
        self.lr = lr
        self.bs = bs
        self.steps = steps

    def fit(self, X_t, y_t):
        X_t = X_t.to_numpy()
        y_t = y_t.to_numpy()
        self.w = np.zeros((X_t.shape[1]))
        self.b = 0
        self.y_t = y_t
        self.X_t = self.norm(X_t)
        self.weights = np.zeros(X_t.shape[1])
        for step in range(self.steps):
            h = self.s(np.dot(self.X_t, self.weights))
            self.weights -= self.lr * np.dot(self.X_t.T, (h - self.y_t)) / self.y_t.size
        return self

    def s(self, z):
        return 1 / (1 + np.exp(-z))

    def norm(self, X):
        for i in range(X.shape[1]):
            X = (X - X.mean(axis=0)/X.std(axis=0))
        return X

    def predict(self, X_p):
        X_p = self.norm(X_p.to_numpy())
        res = self.s(np.dot(X_p, self.w) + self.b)
        return np.array([1 if i > 0.5 else 0 for i in res])


class Node:
    def __init__(self, min_samples_split=20, max_depth=5, depth=0, node_type='root', rule=""):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.depth = depth
        self.node_type = node_type
        self.rule = rule
        self.left = None
        self.right = None
        self.best_feature = None
        self.best_value = None

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.features = list(self.X.columns)
        self.counts = Counter(Y)
        self.gini_impurity = self.get_GINI()
        counts_sorted = list(sorted(self.counts.items(), key=lambda item: item[1]))
        yhat = None
        if len(counts_sorted) > 0:
            yhat = counts_sorted[-1][0]
        self.yhat = yhat
        self.n = len(Y)

    def GINI_impurity(self, y1_count, y2_count):
        if y1_count is None:
            y1_count = 0
        if y2_count is None:
            y2_count = 0
        n = y1_count + y2_count
        if n == 0:
            return 0.0
        p1 = y1_count / n
        p2 = y2_count / n
        gini = 1 - (p1 ** 2 + p2 ** 2)
        return gini

    def ma(self, x, window):
                return np.convolve(x, np.ones(window), 'valid') / window

    def get_GINI(self):
        y1_count, y2_count = self.counts.get(0, 0), self.counts.get(1, 0)
        return self.GINI_impurity(y1_count, y2_count)

    def best_split(self):
        df = self.X.copy()
        df['Y'] = self.Y
        GINI_base = self.get_GINI()
        max_gain = 0
        best_feature = None
        best_value = None
        for feature in self.features:
            Xdf = df.dropna().sort_values(feature)
            xmeans = self.ma(Xdf[feature].unique(), 2)
            for value in xmeans:
                left_counts = Counter(Xdf[Xdf[feature] < value]['Y'])
                right_counts = Counter(Xdf[Xdf[feature] >= value]['Y'])
                y0_left, y1_left, y0_right, y1_right = left_counts.get(0, 0), left_counts.get(1, 0), right_counts.get(0,0), right_counts.get(1, 0)
                gini_left = self.GINI_impurity(y0_left, y1_left)
                gini_right = self.GINI_impurity(y0_right, y1_right)
                n_left = y0_left + y1_left
                n_right = y0_right + y1_right
                w_left = n_left / (n_left + n_right)
                w_right = n_right / (n_left + n_right)
                wGINI = w_left * gini_left + w_right * gini_right
                GINIgain = GINI_base - wGINI
                if GINIgain > max_gain:
                    best_feature = feature
                    best_value = value
                    max_gain = GINIgain
        return (best_feature, best_value)

    def grow_tree(self):
        df = self.X.copy()
        df['Y'] = self.Y
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):
            best_feature, best_value = self.best_split()
            if best_feature is not None:
                self.best_feature = best_feature
                self.best_value = best_value
                left_df, right_df = df[df[best_feature] <= best_value].copy(), df[df[best_feature] > best_value].copy()
                left = Node(
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    node_type='left_node',
                    rule=f"{best_feature} <= {round(best_value, 3)}")
                left.fit(left_df[self.features], left_df['Y'].values.tolist())
                self.left = left
                self.left.grow_tree()

                right = Node(
                    depth=self.depth + 1,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    node_type='right_node',
                    rule=f"{best_feature} > {round(best_value, 3)}")
                right.fit(right_df[self.features], right_df['Y'].values.tolist())
                self.right = right
                self.right.grow_tree()

    def predict(self, X):
        predictions = []
        self.grow_tree()
        for _, x in X.iterrows():
            values = {}
            for feature in self.features:
                values.update({feature: x[feature]})
            predictions.append(self.predict_obs(values))
        return predictions

    def predict_obs(self, values: dict):
        cur_node = self
        while cur_node is not None and cur_node.depth < cur_node.max_depth:
            best_feature = cur_node.best_feature
            best_value = cur_node.best_value
            if cur_node.n < cur_node.min_samples_split:
                break
            if best_feature is not None and values.get(best_feature) is not None and\
                    (values.get(best_feature) < best_value):
                if self.left is not None:
                    cur_node = cur_node.left
            else:
                if self.right is not None:
                    cur_node = cur_node.right
        return cur_node.yhat if cur_node is not None else 0


class RF:
    def __init__(self, num_trees=25, min_samples_split=2, max_depth=5):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.decision_trees = []

    def _sample(self, X, y):
        X = X.to_numpy()
        y = y.to_numpy()
        n_rows, n_cols = X.shape
        samples = np.random.choice(a=n_rows, size=n_rows, replace=True)
        return X[samples], y[samples]

    def fit(self, X, y):
        if len(self.decision_trees) > 0:
            self.decision_trees = []
        num_built = 0
        while num_built < self.num_trees:
            try:
                clf = DecisionTreeClassifier(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
                _X, _y = self._sample(X, y)
                clf.fit(_X, _y)
                self.decision_trees.append(clf)
                num_built += 1
            except Exception as e:
                continue
        return self

    def predict(self, X):
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))
        y = np.swapaxes(a=y, axis1=0, axis2=1)
        predictions = []
        for preds in y:
            counter = Counter(preds)
            predictions.append(counter.most_common(1)[0][0])
        return predictions

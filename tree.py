"""

"""
from copy import copy
import numpy as np
import random

#random.seed(1)

class Node:
    """
    Node Attributes
        ig: Information gain 
        feature_val: (i.e. 'x', 'o', or 'b')
        feature_index: [0-8]
        parent: parent Node
        leaf_val: True or False or None
        x: 'x' child
        o: 'o' child
        b: 'b' child
    """
    def __init__(self, ig=None, parent=None, feature_index=None, feature_val=None, leaf_val=None):
        self.ig = ig 
        self.feature_index = feature_index
        self.feature_val = feature_val
        self.parent = parent
        self.leaf_val = leaf_val
        self.x = None
        self.b = None
        self.o = None

    def get_max_depth(self):
        depths = [0, 0, 0]
        if self.x:
            depths[0] = 1 + self.x.get_max_depth()
        if self.o:
            depths[1] = 1 + self.o.get_max_depth()
        if self.b:
            depths[2] = 1 + self.b.get_max_depth()
        return np.max(depths)
    
    def predict(self, x):
        output = np.ones(x.shape[0])
        if self.feature_index is not None:
            index = self.feature_index
            x_rows = np.where(x[:, index] == 'x')[0]
            o_rows = np.where(x[:, index] == 'o')[0]
            b_rows = np.where(x[:, index] == 'b')[0]
            output[x_rows] *= self.x.predict(x[x_rows])
            output[o_rows] *= self.o.predict(x[o_rows])
            output[b_rows] *= self.b.predict(x[b_rows])
        if self.leaf_val is not None:
            output *= self.leaf_val
        return output

    def print_tree(self, tabs=0):
        if self.parent:
            string = "{}".format(tabs)
            if self.feature_val:
                string += "  " * tabs + "{}".format(self.feature_val)
            if self.feature_index is not None:
                string += " ({}) {}\n".format(self.feature_index, np.round(self.ig, 3))
            elif self.leaf_val is not None:
                string += " ({})\n".format(self.leaf_val)
            else:
                string += "\n"
        else:
            string = "Depth:Node\n{} :Root: ({})\n".format(tabs, self.feature_index)
        if self.x:
            string += self.x.print_tree(tabs + 1)
        if self.o:
            string += self.o.print_tree(tabs + 1)
        if self.b:
            string += self.b.print_tree(tabs + 1)
        return string
    
    def get_node_count(self):
        count = 1
        if self.x:
            count += self.x.get_node_count()
        if self.o:
            count += self.o.get_node_count()
        if self.b:
            count += self.b.get_node_count()
        return count
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

    def get_confusion_matrix(self, X, y):
        pred = np.array(self.predict(X), dtype=np.int32)
        actual = np.array(copy(y), dtype=np.int32)
        conf = np.zeros((2, 2))
        for i in range(len(pred)):
            conf[pred[i], actual[i]] += 1
        return conf

    def __str__(self):
        return self.print_tree()

def get_data():
    X, y = [], []
    with open("tic-tac-toe.data", "r") as f:
        for line in f.readlines():
            split = line.split(",")
            y.append(split[-1].rstrip() == "positive")
            X.append(split[:-1])
    X = np.array(X)
    y = np.array(y)
    return X, y


def get_train_test_split(split = 0.5):
    assert 0.1 <= split <= 0.9
    x, y = get_data()
    combined = list(zip(x, y))
    random.shuffle(combined)
    x, y = zip(*combined)
    x = np.array(x)
    y = np.array(y)
    n = int(split * len(y))
    x_train, x_test = x[:n], x[n:]
    y_train, y_test = y[:n], y[n:]
    return x_train, y_train, x_test, y_test

def remove_nan(x):
    x[np.abs(x) == np.inf] = 0
    val = np.nan_to_num(x, nan=0)
    return val

def get_information_gain(x, y):
    """
    Assume y is a boolean array
    Throws several runtime warnings about zero division and NANs,
    but these are handled in remove_nan
    """
    assert type(x) is np.ndarray
    assert type(y) is np.ndarray
    attrs = ['x', 'o', 'b']
    y_block = np.array([y.tolist()] * x.shape[1]).T
    y_mean = np.mean(y)
    y_entropy = remove_nan(np.array([-(y_mean * np.log2(y_mean) + (1 - y_mean) * np.log2(1 - y_mean))]))
    n = len(y)
    weighted = np.zeros(x.shape[1])
    for attr in attrs:
        attr_total = np.sum(x == attr, axis=0)
        frac = np.sum((x == attr) * y_block, axis=0) / attr_total
        entr_left = remove_nan(-frac * np.log2(frac))
        entr_right = remove_nan(-(1 - frac) * np.log2(1 - frac))
        weighted += attr_total / n * (entr_left + entr_right)
    return y_entropy - weighted
    
def build_tree(x, y, mode: bool, parent=None, feature_val=None):
    y_mean = np.mean(y)
    if y_mean == 1 or y_mean == 0:
        node = Node(leaf_val=bool(y_mean), parent=parent, feature_val=feature_val)
        return node
    if len(y) == 0:
        return Node(leaf_val=mode, parent=parent, feature_val=feature_val)
    ig = get_information_gain(x, y)
    index = np.argmax(ig)
    max_ig = np.max(ig)
    node = Node(ig=max_ig, feature_index=index, parent=parent, feature_val=feature_val)
    x_rows = np.where(x[:, index] == 'x')[0]
    o_rows = np.where(x[:, index] == 'o')[0]
    b_rows = np.where(x[:, index] == 'b')[0]
    node.x = build_tree(x[x_rows], y[x_rows], mode, parent=node, feature_val='x')
    node.o = build_tree(x[o_rows], y[o_rows], mode, parent=node, feature_val='o')
    node.b = build_tree(x[b_rows], y[b_rows], mode, parent=node, feature_val='b')
    return node

def get_stats_str(array):
    mean = np.mean(array)
    std = np.std(array)
    return "Mean: %.3f  SD: %.3f" % (mean, std)

if __name__ == "__main__":
    accuracies = []
    conf_matrices = []
    node_counts = []
    depths = []
    for i in range(30):
        X_train, y_train, X_test, y_test = get_train_test_split()
        mode = np.mean(y_train) >= 0.5
        root = build_tree(X_train, y_train, mode, None)
        pred = root.predict(X_test)
        accuracies.append(root.score(X_test, y_test))
        conf_matrices.append(root.get_confusion_matrix(X_test, y_test))
        node_counts.append(root.get_node_count())
        depths.append(root.get_max_depth())
    print("Accuracy: {}".format(get_stats_str(accuracies)))
    print("Node Count: {}".format(get_stats_str(node_counts)))
    print("Max Depth: {}".format(get_stats_str(depths)))
    avg_conf = sum(conf_matrices) / len(conf_matrices)
    print(avg_conf)
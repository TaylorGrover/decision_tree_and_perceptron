import matplotlib.pyplot as plt
from perceptron import *
import random

def get_train_test_split(attrs, labels, split = 0.5):
    """
    :param split: float between .1 and .9 which determines
    how much training data to use
    """
    assert 0.1 <= split <= 0.9
    combined = list(zip(attrs, labels))
    random.shuffle(combined)
    n = round(split * len(combined))
    attrs, labels = zip(*combined)
    train_attrs, train_labels = np.array(attrs[:n]), np.array(labels[:n])
    test_attrs, test_labels = np.array(attrs[n:]), np.array(labels[n:])
    return train_attrs, train_labels, test_attrs, test_labels

def get_iris(names = ["Iris-versicolor", "Iris-virginica"]):
    """
    Retrieve the iris dataset (assuming 'iris.data' in current directory).
    Assert that there can only be two classes from the iris dataset.
    Encode the classes with either -1 or 1
    """
    assert hasattr(names, "__iter__")
    x = []
    y = []
    if len(names) == 2:
        encoding = dict(zip(list(names), [-1, 1]))
    elif len(names) == 3:
        encoding = dict(zip(names, [0, 1, 2]))
    with open("iris.data", "r") as f:
        for line in f.readlines():
            split = line.split(",")
            name = split[-1].rstrip()
            if name not in names:
                continue
            y.append(encoding[name])
            x.append(np.array(split[:-1], dtype=np.float32))
    return x, y, encoding

def separate_classes(x, y):
    attrs = dict()
    for i in range(len(x)):
        if attrs.get(y[i]) is None:
            attrs[y[i]] = np.array([x[i]])
            continue
        attrs[y[i]] = np.concatenate((attrs[y[i]], [x[i]]))
    return attrs

def plot_predictors(x, y, encoding, name="plot.png"):
    """ 
    Plot the attributes of one class against those of another.
    """
    plt.ion()
    classes = separate_classes(x, y)
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]
    count = 0
    indices = list(encoding.values())
    for i in range(3):
        for j in range(i+1, 4):
            if count > 5:
                continue
            axes[count].scatter(
                classes[indices[0]][:, i],
                classes[indices[0]][:, j],
                color="#66ffff"
            )
            axes[count].scatter(
                classes[indices[1]][:, i],
                classes[indices[1]][:, j],
                color="r"
            )
            axes[count].scatter(
                classes[indices[2]][:, i],
                classes[indices[2]][:, j],
                color = "#000000"
            )
            count += 1
    fig.legend(labels=list(encoding.keys()))
    if not name.endswith(".png"):
        name = name + ".png"
    fig.savefig(name)
    return fig, axes


if __name__ == "__main__":
    x, y, encoding = get_iris(["Iris-virginica", "Iris-setosa"])
    x_train, y_train, x_test, y_test = get_train_test_split(x, y)
    perc = SLPerceptron(x_train.shape[1], max_iter=1000)
    perc.train(x_train, y_train)
    print("Perceptron: {}".format(perc))
    print("Accuracy: {}".format(perc.test(x_test, y_test)))
    x, y, encoding = get_iris(["Iris-virginica", "Iris-setosa", "Iris-versicolor"])
    plot_predictors(x, y, encoding, "_".join(encoding.keys()).replace("Iris-", ""))
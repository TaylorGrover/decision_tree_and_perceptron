from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from PIL import Image
"""
Useful functions shared by MNIST and Iris datasets
"""

def encode_labels(labels):
    le = LabelEncoder()
    new_labels = le.fit_transform(labels)
    return new_labels, le

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
    train_attrs = np.array(attrs[:n])
    train_labels = np.array(labels[:n])
    test_attrs = np.array(attrs[n:])
    test_labels = np.array(labels[n:])
    return train_attrs, train_labels, test_attrs, test_labels

def get_averages(feature_set, labels):
    """
    Determine the average templates for each label
    """
    assert type(labels) is np.ndarray, "labels must be numpy ndarray"
    assert type(feature_set) is np.ndarray, "feature_set must be numpy ndarray"
    if len(labels.shape) == 2:
        labels = np.argmax(labels, axis=1)
    averages = dict()
    for features, label in zip(feature_set, labels):
        if not averages.get(label):
            averages[label] = (features, 1)
            continue
        avg, count = averages[label]
        averages[label] = ((avg * count + features) / (count + 1), count + 1)
    return averages

def get_mode(array):
    """
    Get element which occurs most in an array
    """
    counts = dict()
    for element in array:
        if not counts.get(element):
            counts[element] = 1
            continue
        counts[element] += 1
    return list(counts.keys())[np.argmax(list(counts.values()))]

def euclidean(u, v):
    assert type(u) is np.ndarray, "var u must be an ndarray"
    assert type(v) is np.ndarray, "var v must be an ndarray"
    return np.linalg.norm(u - v, axis=1)

def manhattan(u, v):
    """
    Sum the absolute difference of the vector components of u and v
    """
    assert type(u) is np.ndarray, "var u must be an ndarray"
    assert type(v) is np.ndarray, "var v must be an ndarray"
    return np.sum(np.abs(u - v), axis=1)

def cosine(u, v):
    assert type(u) is np.ndarray, "var u must be an ndarray"
    assert type(v) is np.ndarray, "var v must be an ndarray"
    if u.shape == v.shape and len(u.shape) == 1:
        return np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)
    elif len(u.shape) == 1 and len(v.shape) == 2 and v.shape[1] == u.shape[0]:
        return np.dot(v, u) / np.linalg.norm(v, axis=1) / np.linalg.norm(u)
    elif len(u.shape) == 2 and len(v.shape) == 1 and v.shape[0] == u.shape[1]:
        return np.dot(u, v) / np.linalg.norm(u, axis=1) / np.linalg.norm(v)
    else:
        return u.dot(v)

def predict_by_centroids(centroids, attrs, dist, is_min=True):
    """
    This is essentially the min-distance classification algo.
    :param attrs: feature vectors 
    :param centroids: list of 2-tuples of k-means centroids and their 
        corresponding labels based on voting
    :param dist: distance function
    :param is_min: boolean to determine whether to search for min or max value
        of the given distance metric
    :return: vector of predicted labels for feature vectors
    """
    assert callable(dist), "dist must be a function"
    index = 0 # Index of the minimum distance
    centroid, label = centroids[0]
    distances = []
    centroid_labels = []
    for centroid, label in centroids:
        distances.append(dist(centroid, attrs))
        centroid_labels.append(label)
    distances = np.array(distances)
    centroid_labels = np.array(centroid_labels)
    if is_min:
        return centroid_labels[np.argmin(distances, axis=0)]
    else:
        return centroid_labels[np.argmax(distances, axis=0)]

def avg_to_centroid(averages):
    """
    Take the average value obtained from get averages and convert it into the 
    form needed for predict_by_centroids, i.e. [(average, label), ...]
    """
    assert type(averages) is dict
    centroids = []
    for label in averages:
        centroids.append((averages[label][0], label))
    return centroids

#NOTE: This REQUIRES that both arrays be in the form that all labels are numbers, not strings.
#It also REQUIRES that labels be all numbers starting at 0 and up to n, skipping none on the way. 
def get_confusion_matrix(actual, pred):
    assert len(pred) == len(actual), "passed vectors must be same len"
    lc = len(set(actual)) #make count of labels
    matrix  = np.zeros((lc, lc))
    for i in range(0, len(actual)):
        matrix[actual[i]][pred[i]] += 1
    return matrix
            
def visual_conf_mtrx(data): #visualizes confusion matrix, very basic as of now

    confmtrx = plt.imshow(data , cmap = 'autumn')
    plt.title( "predicted (y) vs actual (x)\n\n" )
    
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            plt.text(x , y , data[y, x]) 
    plt.axis('off')
    plt.show()
   
 
def get_dist_predictions(averages, attrs, labels, is_iris=True):
    centroids = avg_to_centroid(averages)
    predictions = []
    accuracies = []
    confusion_matrices = []
    distances = [euclidean, manhattan, cosine]
    minimize_bools = [True, True, False]
            
    for i, dist in enumerate(distances):
        predictions.append(predict_by_centroids(centroids, attrs, dist, minimize_bools[i]))
        accuracies.append(get_accuracy(predictions[i], labels))
        confusion_matrices.append(get_confusion_matrix(labels, predictions[i]))
    return predictions, accuracies, confusion_matrices

def get_accuracy(pred, actual):
    assert (len(pred)==len(actual)), "err get_accuracy: passed vectors need same length"
    count = 0
    for i in range(0, len(pred)):
        if np.array_equal(pred[i], actual[i]): # More general equality for multiple types, including arrays, floats, str, etc.
            count = count + 1

    return count/len(pred)


def get_MNIST_label(label):
    return np.argmax(label, axis=1)

def get_iris_label(label):
    if(label == 'Iris-setosa'):
        return 0
    if(label == 'Iris-versicolor'):
        return 1
    if(label == 'Iris-virginica'):
        return 2
    return -1

def get_iris_label_string(label):
    if(label == 0):
        return 'Iris-setosa'
    if(label == 1):
        return 'Iris-versicolor'
    if(label == 2):
        return 'Iris-virginica'
    return -1

def get_image(array):
    """
    pass a flat image pixel array and display the image
    """
    assert type(array) is np.ndarray, "array must be a numpy array"
    img = Image.fromarray(np.array((255 * array).reshape(28, 28),
        dtype=np.uint8), mode="L")
    return img

def kmeans_loop(get_data_func, dist_func, ds_name, func_name, is_min=True):
    attrs, labels = get_data_func()
    train_attrs, train_labels, test_attrs, test_labels = get_train_test_split(attrs, labels, 0.5)
    start = time.time()
    train_avgs = get_averages(train_attrs, train_labels) # Problem 6
    print(time.time() - start)
    test_avgs = get_averages(test_attrs, test_labels) # Problem 6
    ks = []
    train_acc = []
    test_acc = []
    for k in range(1, 15 + 1):
        ks.append(k)
        kmeans_train = KMeans(n_clusters=k, max_iter=30)
        start = time.time()
        kmeans_train.fit(train_attrs)
        print("training time: %.f" % (time.time() - start))
        kmeans_test = KMeans(n_clusters=k, max_iter=30)
        kmeans_test.fit(test_attrs)
        centroids_train = []
        centroids_test = []
        for j in range(0, k):
            label_indices = np.where(kmeans_train.labels_ == j)[0]
            cluster_label = get_mode(train_labels[label_indices])
            centroids_train.append((kmeans_train.cluster_centers_[j], cluster_label))
            label_indices = np.where(kmeans_test.labels_ == j)[0]
            cluster_label = get_mode(test_labels[label_indices])
            centroids_test.append((kmeans_test.cluster_centers_[j], cluster_label))
            get_image(centroids_train[j][0]).save(f"images/centroid_{func_name}_train_{k}_{j}.png")
            get_image(centroids_train[j][0]).save(f"images/centroid_{func_name}_test_{k}_{j}.png"  )

        start = time.time()
        train_pred = predict_by_centroids(centroids_train, train_attrs, dist_func, is_min)
        print("train_pred: %.2f" % (time.time() - start))
        test_pred = predict_by_centroids(centroids_test, test_attrs, dist_func, is_min)
        train_acc.append(get_accuracy(train_pred, train_labels))
        test_acc.append(get_accuracy(test_pred, test_labels))

    fig, ax = plt.subplots()
    ax.plot(ks, train_acc, color="r")
    ax.plot(ks, test_acc, color="b")
    fig.savefig("images/elbow_map_kmeans_{}_{}.png".format(ds_name, func_name))

def min_dist(get_data_func, dist_func, is_min=True):
    attrs, labels = get_data_func()
    train_attrs, train_labels, test_attrs, test_labels = get_train_test_split(attrs, labels, 0.5)
    train_avg = get_averages(train_attrs, train_labels)
    test_avg = get_averages(test_attrs, test_labels)
    train_centroids = avg_to_centroid(train_avg)
    test_centroids = avg_to_centroid(test_avg)
    train_pred = predict_by_centroids(train_centroids, train_attrs, dist_func, is_min)
    test_pred = predict_by_centroids(test_centroids, test_attrs, dist_func, is_min)
    train_acc = get_accuracy(train_pred, train_labels)
    test_acc = get_accuracy(test_pred, test_labels)
    print(train_acc)
    print(test_acc)
    return train_pred, train_labels, test_pred, test_labels

def knn(get_data_func):
    attrs, labels = get_data_func()
    train_attrs, train_labels, test_attrs, test_labels = get_train_test_split(attrs, labels)
    distances = ["euclidean", "manhattan", "cosine"]
    le = LabelEncoder()
    le.fit_transform(labels)
    ks = [1, 5, 10]
    for dist in distances:
        accs = []
        print("Distance metric: %s" % dist)
        for i in ks:
            knn = KNeighborsClassifier(n_neighbors=i, metric=dist)
            knn.fit(train_attrs, train_labels)
            pred = knn.predict(test_attrs)
            acc = knn.score(test_attrs, test_labels)
            accs.append(acc)
            print(f"KNN k={i}, score={acc}")
            print("\tConf\n{}".format(get_confusion_matrix(le.transform(pred), le.transform(test_labels))))
    

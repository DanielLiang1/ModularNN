import numpy as np
import h5py


def compute_cost(AL, Y):
    """

    :param AL: Activations from a last layer
    :param Y: labels of data
    :return:
        - cost: The cross-entropy cost function(logistic cost function) result
        - dAL: gradient of AL wrt the cost function
    """
    m = Y.shape[1]
    # Compute loss from aL and y.
    cost = (-1 / m) * np.sum((Y) * np.log(AL) + (1-Y) * np.log(1 - AL))

    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert (cost.shape == ())

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    return cost, dAL


def load_data():
    train_dataset = h5py.File('dataset/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('dataset/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def predict(X, y, Zs, As):
    """

    :param X: Data in shape (features x num_of_examples)
    :param y: labels in shape ( label x num_of_examples)
    :param Zs: All linear layers in form of a list e.g [Z1,Z2,...,Zn]
    :param As: All Activation layers in form of a list e.g [A1,A2,...,An]

    :return: prints accuracy of model given X and Y
        - p: predicted labels
    """
    m = X.shape[1]
    n = len(Zs)  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    Zs[0].linearForward(X)
    As[0].activate(Zs[0].Z)
    for i in range(1, n):
        Zs[i].linearForward(As[i-1].A)
        As[i].activate(Zs[i].Z)
    probas = As[n-1].A

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:  # 0.5 is threshold
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p
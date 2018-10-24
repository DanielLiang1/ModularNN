# RafayAK

import LinearLayer
import ActivationLayer
import matplotlib.pyplot as plt
from utilities import *

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

# Explore your dataset
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# print ("Number of training examples: " + str(m_train))
# print ("Number of testing examples: " + str(m_test))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_x_orig shape: " + str(train_x_orig.shape))
# print ("train_y shape: " + str(train_y.shape))
# print ("test_x_orig shape: " + str(test_x_orig.shape))
# print ("test_y shape: " + str(test_y.shape))


# Reshape the training and test examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

# print ("train_x's shape: " + str(train_x.shape))
# print ("test_x's shape: " + str(test_x.shape))

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# model
learning_rate = .0075
num_of_epochs = 2500
Z1 = LinearLayer.LinearLayer(train_x.shape, 20, ini_type='xavier')
A1 = ActivationLayer.ReluLayer(Z1.Z.shape)

Z2 = LinearLayer.LinearLayer(A1.A.shape, 7, ini_type='xavier')
A2 = ActivationLayer.ReluLayer(Z2.Z.shape)

Z3 = LinearLayer.LinearLayer(A2.A.shape, 5, ini_type='xavier')
A3 = ActivationLayer.ReluLayer(Z3.Z.shape)


Z4 = LinearLayer.LinearLayer(A3.A.shape, 1, ini_type='xavier')
A4 = ActivationLayer.SigmoidLayer(Z4.Z.shape)

costs = []  # list to store costs over training epochs

for epoch in range(num_of_epochs):

    # ------------------------- forward-prop -------------------------
    Z1.linearForward(train_x)
    A1.activate(Z1.Z)

    Z2.linearForward(A1.A)
    A2.activate(Z2.Z)

    Z3.linearForward(A2.A)
    A3.activate(Z3.Z)

    Z4.linearForward(A3.A)
    A4.activate(Z4.Z)

    cost, dA4 = compute_cost(A4.A, train_y)
    if (epoch % 100) == 0:
        print("Cost at epoch#" + str(epoch) + ": " + str(cost))
        costs.append(cost)

    # ------------------------- back-prop -------------------------
    A4.backward(dA4)
    Z4.linearBackward(A4.dZ)

    A3.backward(Z4.dA_prev)
    Z3.linearBackward(A3.dZ)

    A2.backward(Z3.dA_prev)
    Z2.linearBackward(A2.dZ)

    A1.backward(Z2.dA_prev)
    Z1.linearBackward(A1.dZ)

    # ------------------------- update params -------------------------
    Z1.update_params(learning_rate)
    Z2.update_params(learning_rate)
    Z3.update_params(learning_rate)
    Z4.update_params(learning_rate)


# plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()


predictions_train = predict(train_x, train_y, Zs=[Z1, Z2, Z3, Z4], As=[A1, A2, A3, A4])

predictions_test = predict(test_x, test_y, Zs=[Z1, Z2, Z3, Z4], As=[A1, A2, A3, A4])

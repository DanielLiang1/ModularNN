# RafayAK

import LinearLayer
import ActivationLayer
import matplotlib.pyplot as plt
from utilities import *

np.random.seed(1)

# train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
#
# # Explore your dataset
# m_train = train_x_orig.shape[0]
# num_px = train_x_orig.shape[1]
# m_test = test_x_orig.shape[0]

# print ("Number of training examples: " + str(m_train))
# print ("Number of testing examples: " + str(m_test))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("train_x_orig shape: " + str(train_x_orig.shape))
# print ("train_y shape: " + str(train_y.shape))
# print ("test_x_orig shape: " + str(test_x_orig.shape))
# print ("test_y shape: " + str(test_y.shape))


# Reshape the training and test examples
# Standardize data to have feature values between 0 and 1.
train_x = np.array(
		[[0.445, 0.312, 0.242], [0.432, 0.316, 0.252], [0.252, 0.324, 0.424],
		 [0.332, 0.384, 0.284], [0.297, 0.303, 0.400], [0.224, 0.385, 0.391],
		 [0.548, 0.301, 0.152], [0.214, 0.271, 0.515], [0.561, 0.252, 0.187],
		 [0.375, 0.265, 0.361], [0.346, 0.414, 0.241], [0.496, 0.335, 0.170],
		 [0.173, 0.247, 0.580], [0.245, 0.435, 0.320], [0.620, 0.233, 0.147],
		 [0.453, 0.365, 0.183], [0.489, 0.243, 0.268], [0.154, 0.341, 0.505],
		 [0.336, 0.338, 0.326], [0.331, 0.337, 0.332], [0.332, 0.336, 0.331],
		 [0.334, 0.336, 0.331], [0.332, 0.331, 0.337], [0.332, 0.327, 0.341]])

train_y = np.array(
		[[0.555, 0.266, 0.179], [0.526, 0.273, 0.201], [0.167, 0.308, 0.525],
		 [0.337, 0.497, 0.166], [0.252, 0.244, 0.504], [0.117, 0.499, 0.384],
		 [0.759, 0.210, 0.030], [0.107, 0.180, 0.713], [0.735, 0.111, 0.154],
		 [0.368, 0.134, 0.497], [0.393, 0.554, 0.053], [0.674, 0.309, 0.017],
		 [0.061, 0.130, 0.810], [0.131, 0.722, 0.148], [0.839, 0.069, 0.091],
		 [0.594, 0.403, 0.003], [0.568, 0.091, 0.341], [0.000, 0.386, 0.613],
		 [0.339, 0.335, 0.326], [0.330, 0.333, 0.337], [0.330, 0.335, 0.335],
		 [0.329, 0.335, 0.335], [0.322, 0.331, 0.347], [0.320, 0.320, 0.359]])

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

train_x_t = np.transpose(train_x)
train_y_t = np.transpose(train_y)
test_x = train_x_t[:,0].reshape(3,1)
test_y = train_y_t[:, 0].reshape(3,1)

print ("train_x's shape: " + str(train_x_t.shape))
print ("test_x's shape: " + str(test_x.shape))

# model
learning_rate = .03 #.075
num_of_epochs = 500000
Z1 = LinearLayer.LinearLayer(train_x_t.shape, 256, ini_type='xavier')
A1 = ActivationLayer.ReluLayer(Z1.Z.shape)

Z2 = LinearLayer.LinearLayer(A1.A.shape, 256, ini_type='xavier')
A2 = ActivationLayer.SigmoidLayer(Z2.Z.shape)

Z3 = LinearLayer.LinearLayer(A2.A.shape, 256, ini_type='xavier')
A3 = ActivationLayer.SigmoidLayer(Z3.Z.shape)

Z4 = LinearLayer.LinearLayer(A3.A.shape, 256, ini_type='xavier')
A4 = ActivationLayer.ReluLayer(Z4.Z.shape)

Z5 = LinearLayer.LinearLayer(A4.A.shape, 3, ini_type='xavier')
A5 = ActivationLayer.SigmoidLayer(Z5.Z.shape)

costs = []  # list to store costs over training epochs

for epoch in range(num_of_epochs):

	# ------------------------- forward-prop -------------------------
	Z1.linearForward(train_x_t)
	A1.activate(Z1.Z)

	Z2.linearForward(A1.A)
	A2.activate(Z2.Z)

	Z3.linearForward(A2.A)
	A3.activate(Z3.Z)

	Z4.linearForward(A3.A)
	A4.activate(Z4.Z)

	Z5.linearForward(A4.A)
	A5.activate(Z5.Z)

	# print("shape A4.A {}, train_y {}".format(np.shape(A4.A), np.shape(train_y_t)))

	# cost, dA4 = compute_cost(A4.A, train_y_t)
	cost, dA5 = compute_cost_variance(A5.A, train_y_t)
	if (epoch % 1000) == 0:
		print("Cost at epoch#" + str(epoch) + ": " + str(cost))
		costs.append(cost)

	# ------------------------- back-prop -------------------------
	A5.backward(dA5)
	Z5.linearBackward(A5.dZ)

	A4.backward(Z5.dA_prev)
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
	Z5.update_params(learning_rate)


# plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(learning_rate))
plt.show()


predictions_train = predict(train_x_t, train_y_t, Zs=[Z1, Z2, Z3, Z4], As=[A1, A2, A3, A4])

predictions_test = predict(test_x, test_y, Zs=[Z1, Z2, Z3, Z4], As=[A1, A2, A3, A4])

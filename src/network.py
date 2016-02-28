"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import datetime
import copy
# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data=None, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        starttime = datetime.datetime.now()

        if validation_data: n_validation = len(validation_data)
        if test_data: n_test = len(test_data)

        n = len(training_data)

        validation_accuray = 0
        best_validation_accuracy = 0
        test_accuracy = 0
        best_epoch = 0
        best_time = datetime.datetime.now()
        times = 0
        target_time = datetime.datetime.now()
        target_epoch = 0
        flag = 0

        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                if self.update_mini_batch(mini_batch, eta) == 1:
                    break
        
            #modified by Xing
            if validation_data:
                validation_accuray = self.evaluate(validation_data)
                if abs(validation_accuray - best_validation_accuracy) <= 10:
                    times = times + 1
                    if times == 10:
                        break
                else:
                    times = 0

                if validation_accuray > best_validation_accuracy:
                    print "This is the best validation accuracy to date."
                    tmp_time = datetime.datetime.now()
                    print "Time is {0}".format((tmp_time - starttime).seconds)

                    best_validation_accuracy = validation_accuray
                    best_epoch = j
                    best_time = datetime.datetime.now()

                    if best_validation_accuracy >= 9000 and flag == 0:
                        target_epoch = j
                        target_time = datetime.datetime.now()
                        flag = 1
                    
                print "Epoch {0}: {1:.2%}".format(j, validation_accuray/(n_validation-0.0))

            else:
                print "Epoch {0} complete".format(j)

            if test_data:
                test_accuracy =self.evaluate(test_data)
                print "Test"
                print "Epoch {0}: {1:.2%}\n".format(j, test_accuracy/(n_test-0.0))
            else:
                print "Epoch {0} complete".format(j)
        
        endtime = datetime.datetime.now()

        print "Finished training network."
        print "Time used {0}s".format((endtime - starttime).seconds)

        print "Time at accuracy is 90% {0}".format((target_time - starttime).seconds)
        print "Epoch at accuracy is 90% {0}".format(target_epoch)

        print "Best validation accuracy is {0:.2%}".format(best_validation_accuracy/(n_validation-0.0))
        print "Best validation accuracy obtained at epoch {0}".format(best_epoch)
        print "Best validation accuracy obtained at time {0}".format((best_time - starttime).seconds)
        print "Corresponding test accuracy of {0:.2%}\n".format(test_accuracy/(n_test-0.0))


    def cal_diff(self, new_weight, old_weight):
        sqDistances = 0
        for x in range (0, len(old_weight)):
            for y in range(0,len(old_weight[x])):
                for z in range(0,len(old_weight[x][y])):
                    tmp = new_weight[x][y][z] - old_weight[x][y][z]
                    tmp = tmp ** 2
                    sqDistances = sqDistances + tmp

        distance = sqDistances**0.5
    
        return distance


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        #modified by Xing
        new_weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        if self.cal_diff(new_weights, self.weights) - 0.00001 < 0.0000001:
            return 1

        self.weights = copy.deepcopy(new_weights)
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

        return 0


    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

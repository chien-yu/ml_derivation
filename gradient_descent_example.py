#!/usr/python

import numpy as np

np.random.seed(1)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z))

# Input
X = np.array([ [0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1] ])

# Labeled result
Y = np.array([[0,0,1,1]]).T

# First layer weight, something we want to optimize
w = 2*np.random.random((3,1))-1

for iter in range(1000):

    layer0 = X
    z = np.dot(layer0,w)
    # For backpropagation, layer1 is a
    layer1 = sigmoid(z)
    layer1_error = Y - layer1

    # Cost function
    # C = Sigma   (Y_n - layer1_n)^2
    #     (n=1..4)
    C = np.sum(np.square(layer1_error)) 

    # Gradient of cost function (∇C), should be a 3-dimension vector pointed to where the value of
    # cost function grows.
    #
    # GC_w1 is derivative at w1 direction
    # ∂C/∂w1 = ∂C/∂z * ∂z/∂w1
    #        = Sigma     2 * (Y_n - layer1_n) * (-1) * ∂(layer1_n)/∂z_n * ∂z_n/∂w1
    #          (n=1..4)
    #
    # ∂(layer1_n)/∂z_n = ∂(sigmoid(z_n))/∂z_n
    #                  = sigmoid(z_n) * (1-sigmoid(z_n))
    #
    # Note:
    # 1) z_n = (z_n_1 * w1) + (z_n_2 * w2) + (z_n_3 * w3)
    # 2) layer0[:,0] is first column of the matrix. All the data represent weighted by w1.
    #      = [z_1_1, z_2_1, z_3_1, z_4_1]
    # 3) For backpropagation, ∂C/∂z is backward pass
    #                         ∂z/∂w1 is forward pass
    #
    #               Sigma [             4*1               ].T   [   1*4   ]
    GC_w1 = -2 * np.sum(  (layer1_error * sigmoid_deriv(z)).T * layer0[:,0])
    GC_w2 = -2 * np.sum(  (layer1_error * sigmoid_deriv(z)).T * layer0[:,1])
    GC_w3 = -2 * np.sum(  (layer1_error * sigmoid_deriv(z)).T * layer0[:,2])

    # ∇C =           [ ∂C/∂w1,
    #                  ∂C/∂w2,
    #                  ∂C/∂w3 ]
    GC_w = np.array([ [GC_w1],
                      [GC_w2],
                      [GC_w3] ])

    # Quicke way to do calculate gradient:
    #                   [  3*4 ]     [       4*1        ]   [   4*1    ]
    # GC_w_quick = -2 * layer0.T.dot(sigmoid_deriv(z) * layer1_error)
    # if not np.array_equal(GC_w1, GC_w1_quick):
    # 	print(GC_w1, GC_w1_quick)
    # 	assert False

    # Iterative optimization
    # Since gradient point to local maximum, we should re-direct to minimul by minusing it.
    w -= GC_w

print("w\n", w)
print("Output After Training:\n", layer1)

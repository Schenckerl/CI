#!/usr/bin/env python
__author__ = 'bellec,subramoney'

import numpy as np

from toolbox import sig


def cost(theta, x, y):
    """
    Cost of the logistic regression function.

    :param x:
    :param y:
    :return:
    """
    N, n = x.shape

    ##############
    #
    # TODO
    #
    # Write the cost of logistic regression as defined in the lecture
    # - Hint: use the logistic function sig imported from the file toolbox

    costsa = np.zeros(y.shape)
    for i in range(N):
        if not y[i]:
            costsa[i] = -np.log(1-(sig(np.dot(x[i], theta))))
        else:
            costsa[i] = -np.log(sig(np.dot(x[i], theta)))

    print(np.mean(costsa))

    costs = np.zeros(y.shape)
    prob = sig(x.dot(theta))
    for i in range(N):
      costs[i] = y[i]*np.log(prob[i])+(1-y[i])*np.log(1-prob[i])

    c = -np.mean(costs)
    print(c)
    #print(c)
    # END TODO
    ###########

    return c


def grad(theta, x, y):
    """

    Compute the gradient of the cost of logistic regression

    :param theta:
    :param x:
    :param y:
    :return:
    """
    N, n = x.shape

    ##############
    #
    # TODO
    #
    prob = sig(np.dot(x, theta))
    tmp = np.zeros(x.shape)

    for i in range(N):
      tmp[i] = np.abs((prob[i]-y[i]))*x[i]

    g = np.zeros(theta.shape)

    for j in range(N):
      g += tmp[j]

    g /= N

    print(g)
    # END TODO
    ###########

    return g

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

    def subCost(p, y):
        if not y:
            return -np.log(1-p)
        else:
            return -np.log(p)

    cost_per_datapoint = np.zeros(y.shape)
    c = 0

    for i in range(N):
        z = x[i].T.dot(theta)
        cost_per_datapoint[i] = subCost(sig(z), y[i])

    c = np.mean(cost_per_datapoint)

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

    g = np.zeros(theta.shape)

    for i in range(N):
        z = x[i].T.dot(theta)
        g += (sig(z) - y[i]) * x[i]

    g /= N

    # END TODO
    ###########

    return g

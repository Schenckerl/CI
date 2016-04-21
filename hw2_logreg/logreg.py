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

    def subCost(p, act):
        if act == 0:
            return -np.log(1-p)
        else:
            return -np.log(p)

    c = 0
    for i in range(N):
        z = np.dot(np.transpose(x[i]), theta)
        c += subCost(sig(z), y[i])

    c /= N

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
        g += (sig(z) - y[i]) * x[i].T

    for j in range(theta.shape[0]):
         g[j] /= N

    # END TODO
    ###########

    return g

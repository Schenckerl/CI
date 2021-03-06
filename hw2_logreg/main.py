#!/usr/bin/env python
__author__ = 'bellec,subramoney'

"""
Computational Intelligence TU - Graz
Assignment 2: Logistic Regression

This is the main file that loads the data, compute the solution and plot the results.
"""

import json

import matplotlib.pyplot as plt
import numpy as np

import gradient_descent as gd
import logreg as lr
import toolbox


def main():
    # Set parameters
    degree = 1
    eta = 1
    max_iter = 20

    # Load data and expand with polynomial features
    f = open('data.json', 'r')
    data = json.load(f)
    for k, v in data.items(): data[k] = np.array(v)  # Encode list into numpy array

    # Expand with polynomial features
    X_train = toolbox.poly_2D_design_matrix(data['x1_train'], data['x2_train'], degree)
    n = X_train.shape[1]

    # Define the functions of the parameter we want to optimize
    def f(theta): return lr.cost(theta, X_train, data['y_train'])

    def df(theta): return lr.grad(theta, X_train, data['y_train'])

    # Test to verify if the computation of the gradient is correct
    toolbox.check_gradient(f, df, n)

    # Point for initialization of gradient descent
    theta0 = np.zeros(n)

    #### VARIANT 1: Optimize with gradient descent
    theta_opt, E_list = gd.gradient_descent(f, df, theta0, eta, max_iter)

    #### VARIANT 2: Optimize with gradient descent
    # theta_opt, E_list, l_rate_final = gd.adaptative_gradient_descent(f, df, theta0, eta, max_iter)
    # print('Adaptative gradient, final learning rate: {:.3g}'.format(l_rate_final))

    #### VARIANT 3: Optimize with gradient descent
    # res = minimize(f, x0=theta0, jac=df, options={'disp': True})
    # theta_opt = res.x.reshape((n, 1))
    # E_list = []

    toolbox.plot_logreg(data, degree, theta_opt, E_list)
    plt.show()


if __name__ == '__main__':
    main()

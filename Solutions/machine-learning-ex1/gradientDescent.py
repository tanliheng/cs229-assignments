import numpy as np
from computeCost import *


def gradient_descent(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #
        # Hint: X.shape = (97, 2), y.shape = (97, ), theta.shape = (2, )

        error = np.dot(X, theta).flatten() - y #flatten的作用是将多维数组降至一维
        theta -= (alpha/m)*np.sum(X*error[:, np.newaxis], 0) #梯度下降公式theta=theta- alpha/m * sum((fx-y)*x),cost求导一次就变成这个
                                                         #0时,是压缩行,即将每一列的元素相加,将矩阵压缩为一行
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history


def gradient_descent_multi(X, y, theta, alpha, num_iters):
    # Initialize some useful values
    m = y.size
    J_history = np.zeros(num_iters)

    for i in range(0, num_iters):
        # ===================== Your Code Here =====================
        # Instructions : Perform a single gradient step on the parameter vector theta
        #

        error = np.dot(X, theta).flatten() - y
        theta -= (alpha / m) * np.sum(X * error[:, np.newaxis], 0)

        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history

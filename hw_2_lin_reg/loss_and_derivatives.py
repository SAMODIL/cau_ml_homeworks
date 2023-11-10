import numpy as np


class LossAndDerivatives:
    @staticmethod
    def mse(X, Y, w):

        return np.mean((X.dot(w) - Y)**2)

    @staticmethod
    def mae(X, Y, w):
   
        return np.mean(np.abs(X.dot(w) - Y))

    @staticmethod
    def l2_reg(w):

        return np.sum(w**2)

    @staticmethod
    def l1_reg(w):

        return np.sum(np.abs(w))

    @staticmethod
    def no_reg(w):

        return 0.
    
    @staticmethod
    def mse_derivative(X, Y, w):

        return 2* np.dot(X.T, (X.dot(w) - Y)) / Y.size

    @staticmethod
    def mae_derivative(X, Y, w):

        return np.dot(X.T, (np.sign(X.dot(w)-Y))) / Y.size

    @staticmethod
    def l2_reg_derivative(w):

        return 2*w

    @staticmethod
    def l1_reg_derivative(w):

        return np.sign(w)

    @staticmethod
    def no_reg_derivative(w):

        return np.zeros_like(w)

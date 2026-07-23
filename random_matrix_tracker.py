#!/usr/bin/env python
# coding: utf-8

import numpy as np, scipy
from numpy.linalg import inv

from collections import defaultdict
from pprint import pprint


class RandomMatrixTrackerState:
    def __init__(self, tracker, x, X, P, alpha):
        self.tracker = tracker
        self.x = x
        self.X = X
        self.P = P
        self.alpha = alpha


    def update(self, y):
        return self.tracker.update(self, y)
    
    
    def predict(self, t):
        return self.tracker.predict(self, t)
    

    def dumps(self):
        return {k:v.tolist() if type(v) == type(np.array([])) else v for k,v in self.__dict__.items() if k != "tracker"}


class RandomMatrixTracker:
    def __init__(self, H, P, R, F, Q, tau, z, debug=False):
        """
        Sets the parameters for the Random Matrix Tracker

        :param H: measurement matrix
        :param P: prior
        :param R: sensor error cov
        :param F: motion model
        :param Q: motion model error cov
        :param T: time step
        :param tau: object deformation agility over time constant
        """

        self.H = H
        self.P = P
        self.R = R
        self.F = F
        self.Q = Q
        self.tau = tau

        self.z = z
        self.prior_ = None

        self.hist = defaultdict(list)
        self.debug = debug


    def prior(self, x=None, X=None) -> RandomMatrixTrackerState:
        prior = self.prior_
        prior.x = x
        prior.X = X
        return prior


    def setPrior(self, x, X, P, alpha) -> RandomMatrixTrackerState:
        self.prior_ = RandomMatrixTrackerState(self, x, X, P, alpha)
        return self


    def update(self, state:RandomMatrixTrackerState, y):
        return self._update(state.x, state.X, y, self.H, state.P, self.R, state.alpha)


    #def _update(self, y, x, P, X, alpha, H, R)
    def _update(self, x, X, y, H, P, R, alpha):
        z = self.z

        nk = y.shape[1]

        #test
        #d = 2
        #X = X * (alpha - 2 * d - 2) ** (-1)

        Y = z * X + R
        y_mean = np.mean(y, axis=1)[np.newaxis, :].T
        #print(f"{y.shape} - {y_mean.T.shape}")
        Y_centered = y - y_mean
        Y_bar = Y_centered @ Y_centered.T # spread, not covmat
        
        #print(np.cov(y)*(nk-1))
        #print(f"S = {H.shape} @ {P.shape} @ {H.T.shape} + {Y.shape} / nk")
        S = H @ P @ H.T + (Y / nk)
        #print(f"K = {P.shape} @ {H.shape} @ {S.T.shape}")
        K = np.linalg.solve(S.T, H @ P.T).T

        #print(f"x_ = {x.shape} + {K.shape} @ ({y_mean.shape} - {H.shape} @ {x.shape})")
        x_ = x + K @ (y_mean - H @ x)

        P_ = P - K @ S @ K.T

        n_ = (y_mean - (H @ x_))
        #print(n_)
        N = n_ @ n_.T

        alpha_ = alpha + nk
        
        X_half = np.real(scipy.linalg.sqrtm(X)) #
        #X_half = np.linalg.cholesky(X)
        S_half = np.real(scipy.linalg.sqrtm(np.linalg.inv(S))) #
        #S_half = np.linalg.cholesky(np.linalg.inv(S))
        Y_half = np.real(scipy.linalg.sqrtm(np.linalg.inv(Y))) #
        #Y_half = np.linalg.cholesky(np.linalg.inv(Y))

        #print(f"{X_half.shape} @ {S_half.shape} @ {N.shape} @ {S_half.T.shape} @ {Y_half.T.shape}")
        #print(f"{X_half.shape} @ {Y_half.shape} @ {Y_bar.shape} @ {X_half.T.shape} @ {Y_half.T.shape}")
        N_hat = X_half @ S_half @ N @ S_half.T @ X_half.T
        Y_hat = X_half @ Y_half @ Y_bar @ Y_half.T @ X_half.T

        #print(N_hat, Y_hat)

        X_ = (alpha * X + N_hat + Y_hat) / alpha_
        #X_ = X + N_hat + Y_hat
        #print(X + N_hat + Y_hat)

        #print(X_, X)

        if self.debug:
            self.hist["Xs"].append(X)
            self.hist["Ks"].append(K)
            self.hist["Ns"].append(N)
            self.hist["Ss"].append(S)
            self.hist["Ybs"].append(Y_bar)
            self.hist["Ys"].append(Y)
            self.hist["alphasA"].append(alpha_)
            self.hist["alphasB"].append(alpha)
            self.hist["Nhs"].append(N_hat)
            self.hist["Yhs"].append(Y_hat)

        return RandomMatrixTrackerState(self, x_, X_, P_, alpha_)


    def predict(self, state, t):
        return self._predict(state.x, state.X, state.P, self.F, self.Q, state.alpha, t, self.tau)


    def _predict(self, x, X, P, F, Q, alpha, T, tau):
        F = F(T)

        #d = 2
        x_ = F @ x
        P_ = F @ P @ F.T + Q
        X_ = X
        alpha_ = 2 + np.exp(-T / tau) * (alpha - 2)
       

        #alpha_ = np.exp(-T / tau) * (alpha - 2 * d - 2) 
        # #alpha_ += 2 * d + 2
        #X_ = X * (alpha_ - d -1 ) / (alpha -d -1) 

        return RandomMatrixTrackerState(self, x_, X_, P_, alpha_)
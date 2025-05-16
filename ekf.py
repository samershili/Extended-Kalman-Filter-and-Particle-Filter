""" Written by Brian Hou for CSE571: Probabilistic Robotics (Winter 2019)
"""

import numpy as np

from utils import minimized_angle

import matplotlib.pyplot as plt


class ExtendedKalmanFilter:
    def __init__(self, mean, cov, alphas, beta):
        self.alphas = alphas
        self.beta = beta

        self._init_mean = mean
        self._init_cov = cov
        self.reset()

    def reset(self):
        self.mu = self._init_mean
        self.sigma = self._init_cov

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark observation."""
        
        # 1. Predict step
        mu_bar = env.forward(self.mu.ravel(), u).reshape((3, 1))                      # Predicted mean
        G = env.G(self.mu, u)                                                         # Jacobian of motion model wrt state
        V = env.V(self.mu, u)                                                         # Jacobian of motion model wrt control
        M = env.noise_from_motion(u, self.alphas)                                     # Control noise covariance
        sigma_bar = G @ self.sigma @ G.T + V @ M @ V.T                                # Predicted covariance

        # 2. Compute expected observation from predicted state
        z_hat = env.observe(mu_bar.ravel(), marker_id).reshape((1, 1))                # Expected observation
        H = env.H(mu_bar, marker_id)                                                  # Jacobian of observation model wrt state

        # 3. Innovation (measurement residual)
        innovation = z - z_hat
        innovation[0, 0] = minimized_angle(innovation[0, 0])                          # Normalize angle

        # 4. Innovation covariance
        Q = H @ sigma_bar @ H.T + self.beta

        # 5. Kalman gain
        K = sigma_bar @ H.T @ np.linalg.inv(Q)

        # 6. Update state
        self.mu = mu_bar + K @ innovation

        # 7. Update covariance
        self.sigma = (np.eye(3) - K @ H) @ sigma_bar

        return self.mu, self.sigma


    

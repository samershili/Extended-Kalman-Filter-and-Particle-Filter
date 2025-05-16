import numpy as np
from utils import minimized_angle

class ParticleFilter:
    def __init__(self, mean, cov, num_particles, alphas, beta):
        self.alphas = alphas
        self.beta = beta
        self._init_mean = mean
        self._init_cov = cov
        self.num_particles = num_particles
        self.reset()

    def reset(self):
        self.particles = np.zeros((self.num_particles, 3))
        for i in range(self.num_particles):
            self.particles[i, :] = np.random.multivariate_normal(
                self._init_mean.ravel(), self._init_cov)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update(self, env, u, z, marker_id):
        """Update the state estimate after taking an action and receiving a landmark observation."""
        new_particles = np.zeros((self.num_particles, 3))
        eps = 1e-6  # small value to avoid zero variance

        # Prediction Step
        for i in range(self.num_particles):
            x, y, theta = self.particles[i]
            rot1, trans, rot2 = u

            noisy_rot1 = rot1 + np.random.normal(
                0, np.sqrt(self.alphas[0]*rot1**2 + self.alphas[1]*trans**2 + eps))
            noisy_trans = trans + np.random.normal(
                0, np.sqrt(self.alphas[2]*trans**2 + self.alphas[3]*(rot1**2 + rot2**2) + eps))
            noisy_rot2 = rot2 + np.random.normal(
                0, np.sqrt(self.alphas[0]*rot2**2 + self.alphas[1]*trans**2 + eps))

            new_particles[i, 0] = x + noisy_trans * np.cos(theta + noisy_rot1)
            new_particles[i, 1] = y + noisy_trans * np.sin(theta + noisy_rot1)
            new_particles[i, 2] = minimized_angle(theta + noisy_rot1 + noisy_rot2)

        # Measurement Update Step
        weights = np.zeros(self.num_particles)

        # ✅ Correct access to landmark position
        landmark_pos = (env.MARKER_X_POS[marker_id], env.MARKER_Y_POS[marker_id])

        for i in range(self.num_particles):
            x, y, theta = new_particles[i]

            dx = landmark_pos[0] - x
            dy = landmark_pos[1] - y
            expected_bearing = minimized_angle(np.arctan2(dy, dx) - theta)

            error = minimized_angle(z - expected_bearing)
            weights[i] = np.exp(-0.5 * error**2 / self.beta)

        # Normalize weights robustly
        if np.sum(weights) == 0:
            weights = np.ones(self.num_particles) / self.num_particles
        else:
            weights /= np.sum(weights)

        # Resample particles
        self.particles, self.weights = self.resample(new_particles, weights)

        return self.mean_and_variance(self.particles)

    def resample(self, particles, weights):
        """Low-variance systematic resampling"""
        M = len(particles)
        new_particles = np.zeros_like(particles)
        r = np.random.uniform(0, 1 / M)
        c = weights[0]
        i = 0

        for m in range(M):
            U = r + m / M
            while U > c and i < M - 1:
                i += 1
                c += weights[i]
            new_particles[m] = particles[i]

        return new_particles, np.ones(M) / M

    def mean_and_variance(self, particles):
        """Compute mean and covariance of the particles."""
        mean = particles.mean(axis=0)
        mean[2] = np.arctan2(
            np.sin(particles[:, 2]).sum(),
            np.cos(particles[:, 2]).sum()
        )

        zero_mean = particles - mean
        for i in range(zero_mean.shape[0]):
            zero_mean[i, 2] = minimized_angle(zero_mean[i, 2])

        cov = np.dot(zero_mean.T, zero_mean) / self.num_particles

        return mean.reshape((-1, 1)), cov

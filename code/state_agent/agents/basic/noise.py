import numpy as np


# Adds noise to Acceleration, Steering, and Brake actions from ActionNetwork. Noise is not applied to other actions
class ActionNetworkNoise:
    def __init__(self, acceleration_mean=0.5, steering_mean=0., brake_mean=-0.1,
                        acceleration_std=0.1, steering_std=0.3, brake_std=0.05,
                        acceleration_theta=0.6, steering_theta=1.0, brake_theta=1.0):
        
        self.acceleration_noise = OUActionNoise(np.array(acceleration_mean), acceleration_std, acceleration_theta)
        self.steering_noise = OUActionNoise(np.array(steering_mean), steering_std, steering_theta)
        self.brake_noise = OUActionNoise(np.array(brake_mean), brake_std, brake_theta)
    
    def __call__(self):
        acc_noise = self.acceleration_noise()
        steering_noise = self.steering_noise()
        brake_noise = self.brake_noise()
        return acc_noise, steering_noise, brake_noise, 0., 0., 0.

# Implements Ornstein-Uhlenbeck Process to create temporally aware randomness
# for exploration of an action space
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
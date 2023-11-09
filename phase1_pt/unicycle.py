from copy import deepcopy

import numpy as np

class Unicycle:
    def __init__(self, dt=0.25, x0=np.zeros(3)):
        self.dt = dt
        # poses in global frame
        x0 = np.reshape(x0, (3,1))
        self.x0 = x0
        self.x = deepcopy(x0)
        self.history = deepcopy(x0)

    def reset(self):
        self.x = deepcopy(self.x0)
        self.history = deepcopy(self.x0)

    def step(self, u, dt=None):
        dt = self.dt if dt is None else dt
        delta_x, delta_y, delta_theta = Unicycle.step_calc_global(u, self.x[2,0], dt)
        self.x[0,0] += delta_x
        self.x[1,0] += delta_y
        self.x[2,0] += delta_theta
        # make sure theta is in [-pi, pi]
        self.x[2,0] = np.mod(self.x[2,0] + np.pi, 2*np.pi) - np.pi
        self.history = np.concatenate((self.history, self.x), axis=1)


    @staticmethod
    def step_external(x, u, dt):
        delta_x, delta_y, delta_theta = Unicycle.step_calc_global(u, x[2,0], dt)
        x[0,0] += delta_x
        x[1,0] += delta_y
        x[2,0] += delta_theta
        return x

    @staticmethod
    def step_calc_global(u, th, dt):
        # relative to [0,0,0]
        delta_x_local, delta_y_local, delta_theta = Unicycle.step_calc_local(u, dt)
        # convert to global frame
        delta_x, delta_y = Unicycle.local_to_global(delta_x_local, delta_y_local, th)

        return delta_x, delta_y, delta_theta

    @staticmethod
    def local_to_global(delta_x_local, delta_y_local, theta):
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        rot_G_to_L = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
        delta_x, delta_y = np.matmul(rot_G_to_L, np.array([delta_x_local, delta_y_local]))

        # delta_x = delta_x_local * np.cos(theta) - delta_y_local * np.sin(theta)
        # delta_y = delta_x_local * np.sin(theta) + delta_y_local * np.cos(theta)
        return delta_x, delta_y

    @staticmethod
    def step_calc_local(u, dt):
        v, w = u[0], u[1]
        if w == 0:
            delta_x = v * dt
            delta_y = 0
            delta_theta = 0
        else:
            delta_x = (v / w) * np.sin(w * dt)
            delta_y = (v / w) * (1 - np.cos(w * dt))
            delta_theta = w * dt

        return delta_x, delta_y, delta_theta
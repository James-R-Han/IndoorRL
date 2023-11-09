import numpy as np

class Controller:

    def __init__(self, policy_name, max_v=1.25, max_w=1.0):
        self.policy = self.select_policy(policy_name)
        self.max_v = max_v
        self.max_w = max_w

    def select_policy(self, policy_name):
        if policy_name == 'random_policy':
            return self.random_policy
        elif policy_name == 'straight_policy':
            return self.straight_policy
        elif policy_name == 'constant_policy':
            return self.constant_policy
        else:
            raise ValueError('Invalid policy name')

    def compute_step(self, pose):
        u = self.policy(pose)
        u = Controller.bound_u(u, self.max_v, self.max_w)
        return u

    @staticmethod
    def bound_u(u, max_v, max_w):
        v, w = u[0], u[1]
        v = np.clip(v, -max_v, max_v)
        w = np.clip(w, -max_w, max_w)
        u = np.array([v, w])
        return u

    def random_policy(self, pose):
        v = np.random.uniform(0, self.max_v)
        w = np.random.uniform(-self.max_w, self.max_w)
        u = np.array([v, w])
        return u

    def straight_policy(self, pose):
        u = np.array([self.max_v, 0])
        return u

    def constant_policy(self, pose):
        u = np.array([1, 1])
        return u
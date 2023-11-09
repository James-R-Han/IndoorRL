import numpy as np

class Controller:

    def __init__(self, policy_name, dt, max_v=1.25, max_w=1.0):
        self.policy = self.select_policy(policy_name)
        self.dt = dt
        self.max_v = max_v
        self.max_w = max_w

        self.v_for_vtr = 0.2/self.dt #ex. with dt = 0.25 --> 0.2/0.25 = 0.8
        self.w_for_vtr = (10/180*np.pi) /self.dt # ex. with dt = 0.25 --> 10/180*np.pi/0.25 = 0.6981

    def select_policy(self, policy_name):
        if policy_name == 'random_policy':
            return self.random_policy
        elif policy_name == 'straight_policy':
            return self.straight_policy
        elif policy_name == 'constant_policy':
            return self.constant_policy
        elif policy_name == 'vtr_gen_policy':
            return self.vtr_gen_policy
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

    def vtr_gen_policy(self, pose):
        # goal: the robot should move distance 0.2 or have a rotation of 10 degrees
        trigger = np.random.randint(0, 2)
        if trigger == 1:
            v = self.v_for_vtr
            w = np.random.uniform(-self.max_w, self.max_w)
        else:
            trigger2 = np.random.randint(0, 2)
            w = self.w_for_vtr if trigger2 == 1 else -self.w_for_vtr
            v = np.random.uniform(0.01, self.max_v)

        u = np.array([v, w])
        return u




    def straight_policy(self, pose):
        u = np.array([self.max_v, 0])
        return u

    def constant_policy(self, pose):
        u = np.array([1, 1])
        return u
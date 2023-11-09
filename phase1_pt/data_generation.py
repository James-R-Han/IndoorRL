from copy import deepcopy

from unicycle import Unicycle
from controller import Controller

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm

class PathTrackingEnvironment:
    """
    Path Tracking Data Generation for MPC and RL data generation
    """
    def __init__(self, num_paths, len_path, num_samples_per_path, robot, controller, barrier_function_name, action_gen_name, model_name):

        self.num_paths = num_paths
        self.len_path = len_path
        self.num_samples_per_path = num_samples_per_path
        self.robot = robot
        self.controller = controller
        # self.barriers = self.select_barrier(barrier_function_name)
        # self.action_gen = self.select_action_gen(action_gen_name)
        self.model_name = self.verify_model_name(model_name)

    def generate_data(self):
        data = {}

        for path_num in tqdm(range(self.num_paths)):
            path = self.generate_path(self.len_path)
            kd_tree = self.build_kdtree(path)

            data[path_num] = {}
            # generate training points
            path_indices = np.linspace(0, path.shape[1]-1, self.num_samples_per_path, dtype=int)
            poses_on_path = path[:, path_indices]
            poses_not_on_path = PathTrackingEnvironment.generate_points_not_on_path(poses_on_path)

            # localize to path
            for sample_num in range(poses_not_on_path.shape[1]):
                pose = poses_not_on_path[:,sample_num]
                interp_pose, interp_index, closest_indices = PathTrackingEnvironment.localize_on_path(kd_tree, pose[:2], path)
                if interp_pose is None:
                    PathTrackingEnvironment.plot_localization(path, poses_on_path[:, 0], pose, closest_indices[0],
                                                              closest_indices[1], np.array([0,0,0]), 0)
                    continue

                # # example plot
                # PathTrackingEnvironment.plot_localization(path, poses_on_path[:,0], pose, closest_indices[0], closest_indices[1], interp_pose, interp_index)

                state_gen_input = {'path': path, 'pose': pose, 'interp_pose': interp_pose, 'interp_index': interp_index}
                state = PathTrackingEnvironment.state_gen(state_gen_input, self.model_name)


                # action = self.action_gen(state)
                # pose_prime = Unicycle.step_external(pose, action, self.robot.dt)
                # interp_pose_prime, interp_index_prime = PathTrackingEnvironment.localize_on_path(kd_tree, pose_prime, path)
                # state_prime = self.state_gen(pose_prime, interp_index, path)
                # # TODO: import this function
                # # reward = compute_reward(state, action, state_prime)
                # reward = 0
                # data[path_num][sample_num] = {'state': state, 'action': action, 'state_prime': state_prime, 'reward': reward}
            self.robot.reset()
        return data

    @staticmethod
    def generate_points_not_on_path(poses_on_path):
        # TODO: could potentially add different noise generation strategies
        # TODO: modify covariance so, x,y is different than theta
        # add noise

        #TODO:
        # I think we should add lateral noise first. then add a small bubble of gaussian x,y noise, add some noise to theta

        new_xys = poses_on_path + np.random.normal(0, 0.1, size=poses_on_path.shape)

        # TODO: check if point is within bounds

        return new_xys

    def select_barrier(self, barrier_function_name):
        raise NotImplementedError

    def select_action_gen_strategy(self, action_gen_strategy_name):
        raise NotImplementedError

    @staticmethod
    def build_kdtree(path):
        # build only on x,y positions
        path_nxdim = np.transpose(path)
        kdtree = scipy.spatial.KDTree(path_nxdim[:, :2])
        return kdtree

    # TODO: vectorize

    @staticmethod
    def localize_on_path(kdtree, query_xy, path):
        """

        Assume: 2 closest points are next to each other on path. Haven't seen this happen...
        Note: If 2 closest points are both before or after interpolated point it still works

        Find closest interpolated point on path to a xy point.
        Return interpolated pose, and "interpolated index"
        """
        dists, indices = kdtree.query(query_xy, k=2)
        if dists[0] > dists[1]:
            print("WHAT")
        # print(indices)
        if abs(indices[0] - indices[1]) != 1:
            print(indices)
            return None, None, indices
            # raise ValueError('Indices are not next to each other')

        if indices[0] == 0:
            vec1= query_xy - path[:2,0]
            vec2 = path[:2,1] - path[:2,0]
            # if angle obtuse, then point is before path
            if np.dot(vec1, vec2) < 0:
                return path[:,0], 0, indices
        elif indices[0] == path.shape[1] - 1:
            vec1 = query_xy - path[:2,-1]
            vec2 = path[:2,-2] - path[:2,-1]
            # if angle obtuse, then point is after path
            if np.dot(vec1, vec2) < 0:
                return path[:,-1], path.shape[1] - 1, indices

        point1 = path[:, indices[0]]
        point2 = path[:, indices[1]]

        # find interpolated pose
        vec_base = point2[:2] - point1[:2]
        vec_diff = query_xy[:2] - point1[:2]
        proj = np.dot(vec_diff, vec_base) / np.dot(vec_base, vec_base)
        interp_pose = point1 + proj * (point2-point1)
        if indices[0] < indices[1]:
            inter_index = indices[0] + proj*(indices[1]-indices[0])
        else:
            inter_index = indices[1] + (1-proj)*(indices[0]-indices[1])

        return interp_pose, inter_index, indices


    def generate_path(self, num_steps):
        # robot = Unicycle(dt, x0)
        # controller = Controller(policy_name, max_v, max_w)
        for i in range(num_steps):
            u = self.controller.compute_step(self.robot.x)
            self.robot.step(u)
        return robot.history

    @staticmethod
    def plot_localization(path, ref_pos, curr_pos, closest_idx, second_closest_idx, inter_pos, inter_idx):
        th = path[2,:]
        # convert th to dx,dy vector
        dx = np.cos(th)
        dy = np.sin(th)

        fig, ax = plt.subplots()
        plt.quiver(path[0, :-1], path[1, :-1], path[0,1:] - path[0, :-1], path[1,1:] - path[1, :-1], width=0.005, scale_units='xy', angles='xy', scale=1, label='path')
        plt.quiver(path[0, :], path[1, :], dx, dy, label='poses', color='g')
        plt.scatter(ref_pos[0], ref_pos[1], c='orange', label='ref point')
        plt.scatter(path[0, closest_idx], path[1, closest_idx], c='b', label='closest point')
        plt.scatter(path[0, second_closest_idx], path[1, second_closest_idx], c='cyan', label='second closest point')
        plt.scatter(curr_pos[0], curr_pos[1], c='r', label='curr point')
        plt.scatter(inter_pos[0], inter_pos[1], c='r', label='inter point')
        plt.legend()
        plt.title(f"Inter_idx: {inter_idx}, Closest_idx: {closest_idx}, 2nd_closest_idx: {second_closest_idx}")
        ax.set_aspect('equal')
        plt.show()
        print("hi")


    @staticmethod
    def verify_model_name(model_name):
        if model_name == 'MPC_DRL' or model_name == "DRL_MLP":
            return model_name
        else:
            raise ValueError('Invalid model name')

    @staticmethod
    def state_gen(state_gen_input, model_name):
        if model_name == 'MPC':
            return PathTrackingEnvironment.state_gen_MPC(state_gen_input)
        else:
            # DRL Method
            if model_name == 'DRL_MLP':
                state = PathTrackingEnvironment.state_gen_MLP_format(state_gen_input)
            elif model_name == 'DRL_CNN':
                state = PathTrackingEnvironment.state_gen_CNN_format(state_gen_input)
            else:
                raise ValueError('Invalid state gen name')

            # TODO: change appending format
            if "require_MPC" in state_gen_input:
                state += state_gen_input["MPC_action"]
            return state

    @staticmethod
    def state_gen_MLP_format(state_gen_input):
        path = state_gen_input['path']
        curr_pose = state_gen_input['pose']
        interp_index = state_gen_input['interp_index']

        immediate_next_pose_on_path = int(np.ceil(interp_index))
        immediate_next_pose_on_path = 1000
        max_idx = path.shape[1] - 1
        # TODO: 5 is a param that should be in config
        lookahead = 5
        num_unique_poses = min(lookahead , max_idx - immediate_next_pose_on_path + 1)
        state = path[:, immediate_next_pose_on_path:immediate_next_pose_on_path+num_unique_poses]

        if num_unique_poses < lookahead:
            goal_pose = path[:, -1]
            # concat state with last pose repeated 5- num_unique_poses times
            state = np.concatenate((state, np.tile(goal_pose, (5-num_unique_poses, 1)).T), axis=1)

        # make xy of path relative to current pose
        state[:2,:] = state[:2,:] - np.expand_dims(curr_pose[:2], axis=1)
        # make sure theta is between -pi and pi
        state[2,:] = np.mod(state[2,:] + np.pi, 2*np.pi) - np.pi

        return state

    @staticmethod
    def state_gen_CNN_format(state_gen_input):
        raise NotImplementedError

    @staticmethod
    def state_gen_MPC(state_gen_input):
        raise NotImplementedError

if __name__ == '__main__':
    # Seed
    np.random.seed(1)

    # plotting needs
    cm = matplotlib.colormaps['magma']

    robot = Unicycle()
    controller = Controller('random_policy')
    env = PathTrackingEnvironment(500,1000,300,robot,controller, 'none', 'none', 'DRL_MLP')
    env.generate_data()


    # path = PathTrackingEnvironment.generate_path(10, 'random_policy', dt=0.25)
    #
    # kd_tree = PathTrackingEnvironment.build_kdtree(path)
    # sample_point = path[:2, 10] + np.array([-0.01,0.1])
    # ref_point = path[:2, 10]
    #
    # # dists, indices  = kd_tree.query(sample_point, k=2)
    # interpolated_point, interpolated_index = PathTrackingEnvironment.localize_on_path(kd_tree, sample_point, path)
    # print(interpolated_index)
    # th = path[2,:]
    # # convert th to dx,dy vector
    # dx = np.cos(th)
    # dy = np.sin(th)
    #
    # fig, ax = plt.subplots()
    # plt.quiver(path[0, :], path[1, :], dx, dy, label='poses', color='g')
    # plt.quiver(path[0, :-1], path[1, :-1], path[0,1:] - path[0, :-1], path[1,1:] - path[1, :-1], width=0.005, scale_units='xy', angles='xy', scale=1, label='path')
    # plt.scatter(sample_point[0], sample_point[1], c='r', label='sample point')
    # plt.scatter(ref_point[0], ref_point[1], c='b', label='ref point')
    # plt.scatter(interpolated_point[0], interpolated_point[1], c='r', label='interpolated point')
    # ax.set_aspect('equal')
    # plt.legend()
    # plt.show()



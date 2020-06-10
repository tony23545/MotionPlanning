import numpy as np
from RRTTree import RRTTree
import time

class RRTPlannerNonholonomic(object):

    def __init__(self, planning_env, bias=0.05, max_iter=3000, num_control_samples=25):
        self.env = planning_env                 # Car Environment
        self.tree = RRTTree(self.env)
        self.bias = bias                        # Goal Bias
        self.max_iter = max_iter                # Max Iterations
        self.num_control_samples = 25           # Number of controls to sample

    def Plan(self, start_config, goal_config):
        # TODO: YOUR IMPLEMENTATION HERE

        plan_time = time.time()

        # Start with adding the start configuration to the tree.
        self.tree.AddVertex(start_config)

        for it in range(self.max_iter):
            # generate random node
            if it %200 == 0:
                print(it)
            if np.random.random() < self.bias:
                x_rand = goal_config
            else:
                x_rand = self.env.sample()

            # find nearest node 
            vid_near, x_near = self.tree.GetNearestVertex(x_rand)

            # extend
            x_extend, rollout_extend, action_extend = self.extend(x_near, x_rand)
            if x_extend is None:
                continue

            # add extend node 
            cost = self.tree.costs[vid_near]
            vid_extend = self.tree.AddVertex(x_extend, cost + self.env.compute_distance(x_near, x_extend))
            self.tree.AddEdge(vid_near, vid_extend, action_extend, rollout_extend)

            # check goal
            if self.env.goal_criterion(x_extend, goal_config):
                break

        # if fail
        plan = []
        if it == self.max_iter - 1:
            print("fail")
            plan.append(start_config)
            plan.append(goal_config)
            return np.concatenate(plan, axis=1), None, None

        vid = vid_extend
        conf = goal_config
        plan.append(conf)
        rollouts = []
        actions = []
        cost = self.tree.costs[vid]
        while not (conf[0][0] == start_config[0][0] and conf[1][0] == start_config[1][0]):
            (vid, action, rollout) = self.tree.edges[vid]
            actions.append(action.reshape((2, -1)).repeat([len(rollout[0])-1], axis = 1))
            rollouts.append(rollout[:, 1:])
            
            conf = self.tree.vertices[vid]
            plan.append(conf)
            
        plan.reverse()
        actions.reverse()
        rollouts.append(start_config)
        rollouts.reverse()

        plan_time = time.time() - plan_time

        print("Cost: %f" % cost)
        print("Planning Time: %ds" % plan_time)

        return np.concatenate(plan, axis=1), np.concatenate(actions, axis=1), np.concatenate(rollouts, axis=1)

    def extend(self, x_near, x_rand):
        """ Extend method for non-holonomic RRT

            Generate n control samples, with n = self.num_control_samples
            Simulate trajectories with these control samples
            Compute the closest closest trajectory and return the resulting state (and cost)
        """
        # TODO: YOUR IMPLEMENTATION HERE
        x_extend = None
        rollout_extend = None
        action_extend = None
        dist = self.env.compute_distance(x_near, x_rand)
        for i in range(self.num_control_samples):
            linear_vel, steer_angle = self.env.sample_action()
            x_next, rollout = self.env.simulate_car(x_near, x_rand, linear_vel, steer_angle)
            if x_next is None:
                continue
            d = self.env.compute_distance(x_next, x_rand)
            if d < dist:
                dist = d
                x_extend = x_next
                rollout_extend = rollout
                action_extend = np.array([linear_vel, steer_angle])
        return x_extend, rollout_extend, action_extend
            
    def sample(self, goal):
        # Sample random point from map
        if np.random.uniform() < self.bias:
            return goal

        return self.env.sample()
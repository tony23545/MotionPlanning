import numpy as np
from RRTTree import RRTTree
import time

class RRTSTARPlannerNonholonomic(object):

    def __init__(self, planning_env, bias=0.05, max_iter=5000, num_control_samples=25):
        self.env = planning_env                 # Car Environment
        self.tree = RRTTree(self.env)
        self.bias = bias                        # Goal Bias
        self.max_iter = max_iter                # Max Iterations
        self.num_control_samples = 25           # Number of controls to sample

    def Plan(self, start_config, goal_config, rad = 50):
        # TODO: YOUR IMPLEMENTATION HERE

        plan_time = time.time()

        # Start with adding the start configuration to the tree.
        self.tree.AddVertex(start_config)
        fail = True

        for it in range(self.max_iter):
            if it % 200 == 0:
                print(it)

            if it % 2000 == 0 and not fail:
                break
            # generate random node
            if np.random.random() < self.bias:
                x_rand = goal_config
            else:
                x_rand = self.env.sample()

            # find connect node 
            vid_near, x_near = self.tree.GetNearestVertex(x_rand)

            # extend
            x_extend = self.extend(x_near, x_rand)
            if x_extend is None:
                continue

            # find min node to connect x_extend
            x_min = x_near
            vid_min = vid_near
            c_min = self.tree.costs[vid_near] + self.env.compute_distance(x_near, x_extend)
            vids, vertices = self.tree.GetNNInRad(x_extend, rad)
            for i in range(len(vids)):
                c = self.tree.costs[vids[i]] + self.env.compute_distance(vertices[i], x_extend)
                if c < c_min:
                    if self.env.edge_validity_checker(vertices[i], x_extend):
                        c_min = c
                        x_min = vertices[i]
                        vid_min = vids[i]
                        

            vid_extend = self.tree.AddVertex(x_extend, c_min)
            self.tree.AddEdge(vid_min, vid_extend)

            # reconnect
            for i in range(len(vids)):
                c = c_min + self.env.compute_distance(vertices[i], x_extend)
                if c < self.tree.costs[vids[i]]:
                    if self.env.edge_validity_checker(vertices[i], x_extend):
                        self.tree.costs[vids[i]] = c
                        self.tree.AddEdge(vid_extend, vids[i])

            # check goal
            if self.env.goal_criterion(x_extend, goal_config):
               fail = False
               vid_goal = vid_extend
        
        # if fail
        plan = []
        if fail:
            print("fail")
            plan.append(start_config)
            plan.append(goal_config)
            return np.concatenate(plan, axis=1), fail

        vid = vid_goal
        conf = goal_config
        #plan.append(start_config)
        plan.append(conf)
        cost = self.tree.costs[vid]
        while not (conf[0][0] == start_config[0][0] and conf[1][0] == start_config[1][0]):
            vid = self.tree.edges[vid][0]
            conf = self.tree.vertices[vid]
            plan.append(conf)
        plan.reverse()

        #cost = 0
        plan_time = time.time() - plan_time

        print("Cost: %f" % cost)
        print("Planning Time: %ds" % plan_time)

        return np.concatenate(plan, axis=1), fail

    def extend(self, x_near, x_rand):
        """ Extend method for non-holonomic RRT

            Generate n control samples, with n = self.num_control_samples
            Simulate trajectories with these control samples
            Compute the closest closest trajectory and return the resulting state (and cost)
        """
        # TODO: YOUR IMPLEMENTATION HERE
        x_extend = None
        dist = self.env.compute_distance(x_near, x_rand)
        for i in range(self.num_control_samples):
            linear_vel, steer_angle = self.env.sample_action()
            x_next, _ = self.env.simulate_car(x_near, x_rand, linear_vel, steer_angle)
            if x_next is None:
                continue
            d = self.env.compute_distance(x_next, x_rand)
            if d < dist:
                dist = d
                x_extend = x_next
        return x_extend
            
    def sample(self, goal):
        # Sample random point from map
        if np.random.uniform() < self.bias:
            return goal

        return self.env.sample()
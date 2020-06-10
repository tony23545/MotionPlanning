import numpy as np
from matplotlib import pyplot as plt
from map_utils import depth_to_xy, Map

class CarEnvironment(object):
    """ Car Environment. Car is represented as a circular robot.

        Robot state: [x, y, theta]
    """
    
    def __init__(self, mapfile, radius=15,
                 delta_step=10, max_linear_vel=20, max_steer_angle=1.):

        self.radius = radius

        # Obtain the boundary limits.
        # Check if file exists.
        
        self.m = Map('../map/map.yaml', laser_max_range=4, downsample_factor=1)
        self.map = self.m.occupancy_grid / 255.
        self.xlimit = [0, np.shape(self.map)[0]-1]
        self.ylimit = [0, np.shape(self.map)[1]-1]

        self.delta_step = delta_step            # Number of steps in simulation rollout
        self.max_linear_vel = max_linear_vel
        self.max_steer_angle = max_steer_angle

        #self.goal = goal

        # Check if start and goal are within limits and collision free
        # if not self.state_validity_checker(start) or not self.state_validity_checker(goal):
        #     raise ValueError('Start and Goal state must be within the map limits');
        #     exit(0)

    def sample(self):
        # Sample random clear point from map
        clear = np.argwhere(self.map == 0)
        idx = np.random.choice(len(clear))
        xy = clear[idx, :].reshape((2, 1))
        theta = np.random.uniform(0,2*np.pi)
        return np.concatenate([xy, np.array([[theta]])])

    def sample_action(self):
        # Sample random direction of motion and random steer angle
        linear_vel = (0.5 + 0.5*np.random.rand()) * self.max_linear_vel
        if np.random.rand() > 0.5:
            linear_vel = -1 * linear_vel
        steer_angle = (2*np.random.rand() - 1) * self.max_steer_angle # uniformly distributed
        return linear_vel, steer_angle

    def simulate_car(self, x_near, x_rand, linear_vel, steer_angle):
        """ Simulates a given control from the nearest state on the graph to the random sample.

            @param x_near: a [3 x 1] numpy array. Nearest point on the current graph to the random sample
            @param x_rand: a [3 x 1] numpy array. Random sample
            @param linear_vel: a Python float. Linear velocity of the car (control)
            @param steer_angle: a Python float. Steering angle of the car (control)

            @return: x_new: a [3 x 1] numpy array
                     delta_t: a Python float. Time required to execute trajectory
        """

        # Set vehicle constants
        dt = 0.1 # Step by 0.1 seconds
        L = 7.5 # Car length
        
        # Simulate forward from xnear using the controls (linear_vel, steer_angle) to generate the rollout
        x = x_near
        rollout = [x]
        for i in range(self.delta_step):
            x_rollout = np.zeros_like(x)
            x_rollout[0] = x[0] + linear_vel * np.cos(x[2]) * dt
            x_rollout[1] = x[1] + linear_vel * np.sin(x[2]) * dt
            x_rollout[2] = x[2] + (linear_vel/L) * np.tan(steer_angle) * dt
            x_rollout[2] = x_rollout[2] % (2*np.pi)

            x = x_rollout
            rollout.append(x_rollout) # maintain history
        rollout = np.concatenate(rollout, axis=1) # Shape: [3 x delta_step]
        
        # Find the closest point to x_rand on the rollout
        # This is x_new. Discard the rest of the rollout
        min_ind = np.argmin(self.compute_distance(rollout, x_rand))
        x_new = rollout[:, min_ind].reshape(3,1)
        rollout = rollout[:, :min_ind+1] # don't need the rest
        delta_t = rollout.shape[1]
        
        # Check for validity of the path
        if self.state_validity_checker(rollout):
            return x_new, rollout
        else:
            return None, None

    def step(self, x, action):
        linear_vel = action[0]
        steer_angle = action[1]

        dt = 0.1
        L = 7.5
        x_rollout = np.zeros_like(x)
        x_rollout[0] = x[0] + linear_vel * np.cos(x[2]) * dt
        x_rollout[1] = x[1] + linear_vel * np.sin(x[2]) * dt
        x_rollout[2] = x[2] + (linear_vel/L) * np.tan(steer_angle) * dt
        x_rollout[2] = x_rollout[2] % (2*np.pi)

        return x_rollout

    def angular_difference(self, start_config, end_config):
        """ Compute angular difference

            @param start_config: a [3 x n] numpy array of states
            @param end_config: a [3 x 1] numpy array of goal state
        """                
        th1 = start_config[2,:]; th2 = end_config[2,:]
        ang_diff = th1-th2
        ang_diff = ((ang_diff + np.pi) % (2*np.pi)) - np.pi 
        ang_diff = ang_diff*(180/np.pi) # convert to degrees
        return ang_diff

    def compute_distance(self, start_config, end_config):
        """ Distance function: alignment, xy distance, angular difference
                - alignment: measure of how far start_config is from line defined by end_config
                             similar to projection of start_config xy position end_config line

            @param start_config: a [3 x n] numpy array of states
            @param end_config: a [3 x 1] numpy array of goal state
        """        

        ang_diff = np.abs(self.angular_difference(start_config, end_config))/180
        e_to_s = start_config[:2,:] - end_config[:2,:] # Shape: [2 x n]
        euclidean_distance = np.linalg.norm(e_to_s, axis=0) # Shape: [n]
        e_to_s = e_to_s / euclidean_distance[None,:]
        e_vec = np.array([np.cos(end_config[2,0]), np.sin(end_config[2,0])])
        alignment = 1 - np.abs(e_vec.dot(e_to_s)) # Shape: [n]

        # alignment is in [0,1], euclidean_distance can be large, ang_diff is between [0,1]
        return 50*alignment + euclidean_distance + 50*ang_diff


    def goal_criterion(self, config, goal_config, pos_tor = 30, ang_tor = 360, no_angle = False):
        """ Return True if config is close enough to goal

            @param config: a [3 x 1] numpy array of a state
            @param goal_config: a [3 x 1] numpy array of goal state
        """ 
        if no_angle:
            return np.linalg.norm(config[:2,:] - goal_config[:2,:]) < pos_tor
        if np.linalg.norm(config[:2,:] - goal_config[:2,:]) < pos_tor and \
           np.abs(self.angular_difference(config, goal_config)) < ang_tor:
            print(f'Goal reached! State: {config[:,0]}, Goal state: {goal_config[:,0]}')
            print(f'xy_diff: {np.linalg.norm(config[:2,:] - goal_config[:2,:]):.03f}, '\
                  f'ang_diff: {np.abs(self.angular_difference(config, goal_config))[0]:.03f}')
            return True
        else:
            return False

    def out_of_limits_violation(self, config):
        """ Check limit violations

            @param config: a [3 x n] numpy array of n states
        """
        out_of_limits = np.stack([config[0,:] <= self.radius,
                                  config[0,:] >= (self.xlimit[1] - self.radius),
                                  config[1,:] <= self.radius,
                                  config[1,:] >= (self.ylimit[1] - self.radius)])
        return np.any(out_of_limits)

    def collision_violation(self, config):
        """ Check whether car is in collision with obstacle

            @param config: a [3 x n] numpy array of n states
        """

        theta = np.linspace(0, 2*np.pi, 50)
        xs = self.radius * np.cos(theta) + config[0,:,None] # Shape: [n x 50]
        ys = self.radius * np.sin(theta) + config[1,:,None] # Shape: [n x 50]
        xys = np.stack([xs,ys],axis=0) # Shape: [2 x n x 50]

        cfloor = np.round(xys.reshape(2,-1)).astype("int")
        try:
            values = self.map[cfloor[0, :], cfloor[1, :]]
        except IndexError:
            return False

        return np.sum(values) > 0

    def state_validity_checker(self, config):
        """ Check validity of state

            @param config: = a [3 x n] numpy array of n states.
        """

        valid_position = ~self.collision_violation(config)
        valid_limits = ~self.out_of_limits_violation(config)

        return valid_position and valid_limits

    def edge_validity_checker(self, config1, config2):

        assert(config1.shape == (3, 1))
        assert(config2.shape == (3, 1))
        n = max(self.xlimit[1], self.ylimit[1])
        x_vals = np.linspace(config1[0], config2[0], n).reshape(1, n)
        y_vals = np.linspace(config1[1], config2[1], n).reshape(1, n)
        configs = np.vstack((x_vals, y_vals))
        return self.state_validity_checker(configs)

    def plot_car(self, config):
        """ Plot the car

            @param config: a [3 x 1] numpy array
        """
        config = config[:,0]

        # Plot car as a circle
        ax = plt.gca()
        circle1 = plt.Circle(config[:2][::-1], self.radius, fill=True, facecolor='w')
        circle2 = plt.Circle(config[:2][::-1], self.radius, fill=False, color='k')
        car1 = ax.add_artist(circle1)
        car2 = ax.add_artist(circle2)

        # Now plot a line for the direction of the car
        theta = config[2]
        ed = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) @ np.array([[self.radius*1.5, 0]]).T;
        ed = ed[:,0]
        car3 = ax.plot([config[1], config[1]+ed[1]], [config[0], config[0]+ed[0]], 'b-', linewidth=3)
        return car1, car2, car3

    def init_visualizer(self):
        """ Initialize visualizer
        """

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1, 1, 1)

        # Plot img
        visit_map = 1 - np.copy(self.map) # black is obstacle, white is free space
        self.ax1_img = self.ax1.imshow(visit_map, interpolation="nearest", cmap="gray")

    def draw_obstacles(self, xys, *args, **kwargs):
        '''
        xys: a numpy array of size Nx2 representing locations of obstacles in the global coordinate system.
        '''
        ln = self.ax1.plot(xys[:, 0], xys[:, 1], '+', *args, **kwargs)
        return ln

    def draw_particles(self, particles, *args, **kwargs):
        x = particles[:, 0]
        y = particles[:, 1]
        delta = 10
        xd = np.cos(particles[:, 2])*delta 
        yd = np.sin(particles[:, 2])*delta 

        for i in range(x.shape[0]):
            self.ax1.arrow(y[i], x[i], yd[i], xd[i], color = 'r', head_width = 5, head_length = 5)
        self.fig.canvas.draw()
        

    def visualize_plan(self, plan=None, tree=None, visited=None):
        '''
            Visualize the final path
            @param plan: a [3 x n] numpy array of states
        '''
        visit_map = 1 - np.copy(self.map) # black is obstacle, white is free space
        self.ax1.cla()
        self.ax1.imshow(visit_map, interpolation="nearest", cmap="gray")

        if tree is not None:
            for idx in range(len(tree.vertices)):
                if idx == tree.GetRootID():
                    continue
                econfig = tree.vertices[idx]
                sconfig = tree.vertices[tree.edges[idx][0]]
                x = [sconfig[0], econfig[0]]
                y = [sconfig[1], econfig[1]]
                self.ax1.plot(y, x, 'r')

        ln = None
        if plan is not None:
            for i in range(np.shape(plan)[1]):
                self.plot_car(plan[:,i:i+1])
                self.fig.canvas.draw()

                pos = np.array([plan[1, i], plan[0, i]]) / 100
                heading = plan[2, i]
                n_ray = 180
                fov = np.deg2rad(360.0)
                depth = self.m.get_1d_depth(pos, heading, fov, n_ray, resolution=0.01)
                obstacles = depth_to_xy(depth, pos, heading, fov)
                if not ln is None:
                    ln[0].remove()
                ln = self.draw_obstacles(obstacles*100, markeredgewidth=1.5)

                plt.pause(.025) 

        self.fig.canvas.draw()
        plt.pause(1e-10) 

    def get_measurement(self, plan):
        measurement = []
        for i in range(np.shape(plan)[1]):
            pos = np.array([plan[1, i], plan[0, i]]) / 100
            if plan.shape[0] == 2:
                heading = 0.0
            else:
                heading = plan[2, i]
            n_ray = 180
            fov = np.deg2rad(360.0)
            depth = self.m.get_1d_depth(pos, heading, fov, n_ray, resolution=0.01)
            #obstacles = depth_to_xy(depth, pos, heading, fov)
            measurement.append(depth)
        return np.array(measurement)

    def random_walk(self, sim_t = 100):
        start = self.sample()
        state_traj = [start]
        action_traj = []

        x = start
        for t in range(sim_t):
            linear_vel, steer_angle = self.sample_action()
            if t % 10 == 0:
                action = np.array([linear_vel, steer_angle])
            x_next = self.step(x, action)
            if self.edge_validity_checker(x, x_next):
                state_traj.append(x_next)
                action_traj.append(action)
            else:
                break
            x = x_next

        state_traj = np.array(state_traj).squeeze()
        if state_traj.shape[0] < 10:
            return None, None, None

        action_traj = np.array(action_traj)
        measure_traj = self.get_measurement(state_traj[1:].T)

        return state_traj, action_traj, measure_traj

    def visualize_traj(self, plan, measurement):
        visit_map = 1 - np.copy(self.map) # black is obstacle, white is free space
        self.ax1.cla()
        self.ax1.imshow(visit_map, interpolation="nearest", cmap="gray")

        ln = None
        car1, car2, car3 = None, None, None
        if plan is not None:
            for i in range(np.shape(plan)[1]):
                if not ln is None:
                    ln[0].remove()
                    car1.remove()
                    car2.remove()
                    car3[0].remove()
                car1, car2, car3 = self.plot_car(plan[:,i:i+1])
                self.fig.canvas.draw()

                pos = np.array([plan[1, i], plan[0, i]]) / 100
                heading = plan[2, i]
                # n_ray = 180
                fov = np.deg2rad(360.0)
                # depth = self.m.get_1d_depth(pos, heading, fov, n_ray, resolution=0.01)
                obstacles = depth_to_xy(measurement[i], pos, heading, fov)
                
                ln = self.draw_obstacles(obstacles*100, markeredgewidth=1.5)

                plt.pause(.025) 

        self.fig.canvas.draw()
        plt.pause(1e-10) 

    def render(self, state, particles):
        self.init_visualizer()
        self.plot_car(state)
        self.draw_particles(particles.squeeze())
        self.fig.canvas.draw()
        plt.show()

    # MPNet steer
    def steerTo(self, state, delta):
        # import IPython
        # IPython.embed()

        for ip, tp in enumerate(np.linspace(0, np.pi, 13)):
            rot_state_p = state.T + np.dot(np.array([[np.cos(tp), -np.sin(tp)], [np.sin(tp), np.cos(tp)]]), delta.T)
            rot_state_p = np.concatenate([rot_state_p, [[0]]], axis = 0)
            if self.state_validity_checker(rot_state_p):
                break
        for im, tm in enumerate(np.linspace(0, -np.pi, 13)):
            rot_state_m = state.T + np.dot(np.array([[np.cos(tm), -np.sin(tm)], [np.sin(tm), np.cos(tm)]]), delta.T)
            rot_state_m = np.concatenate([rot_state_m, [[0]]], axis = 0)
            if self.state_validity_checker(rot_state_m):
                break

        if ip <= im:
            return rot_state_p.T[:, :2]
        else:
            return rot_state_m.T[:, :2]

    # for reinforcement learning
    def reset(self):
        state = None
        while True:
            state = self.sample()
            if self.state_validity_checker(state):
                break
        return state

    def step(self, state, action):
        dt = 0.1 # Step by 0.1 seconds
        L = 7.5 # Car length
        next_state = np.zeros_like(state)

        linear_vel, steer_angle = action[0], action[1]
        next_state[0] = state[0] + linear_vel * np.cos(state[2]) * dt
        next_state[1] = state[1] + linear_vel * np.sin(state[2]) * dt 
        next_state[2] = state[2] + (linear_vel/L) * np.tan(steer_angle) * dt 
        next_state[2] = next_state[2] % (2*np.pi)

        obs = self.get_measurement(next_state)

        return next_state, obs



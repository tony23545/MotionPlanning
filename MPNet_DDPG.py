import sys
sys.path.append('Env')
sys.path.append('MPNet')
sys.path.append('DRL')
import numpy as np 
import torch
import argparse

from Env.CarEnvironment import CarEnvironment
from MPNet import MPNet
from DRL.DDPG import DDPG

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default='Pendulum-v0')
parser.add_argument('--model', default='SAC')
parser.add_argument('--mode', default='train')
parser.add_argument('--num_envs', default=8)

parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--alpha', default=0.2, type=float)

parser.add_argument('--capacity', default=100000, type=int) # replay buffer size
parser.add_argument('--hidden_dim', default=64, type=int)

parser.add_argument('--max_episode', default=10000, type=int) # num of games
parser.add_argument('--last_episode', default=0, type=int)
parser.add_argument('--max_length_trajectory', default=50, type=int)
parser.add_argument('--print_log', default=100, type=int)
parser.add_argument('--exploration_noise', default=0.1)
parser.add_argument('--policy_delay', default=2)

parser.add_argument('--update_iteration', default=10, type=int)
parser.add_argument('--batch_size', default=128, type=int) # mini batch size

# experiment relater
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--exp_name', default='experiment')
args = parser.parse_args()



def main():
    planning_env = CarEnvironment("map/map.yaml")
    planning_env.init_visualizer()

    mpnet = MPNet()
    mpnet.load()

    agent = DDPG(args)
    agent.load()

    size = np.array([1788, 1240])

    while True:
        start = planning_env.sample()
        if planning_env.state_validity_checker(start):
            break

    while True:
        goal = planning_env.sample()
        dist = planning_env.compute_distance(start, goal)
        if planning_env.state_validity_checker(goal) and dist > 200 and dist < 2000:
            break

    print(start[:2, 0])
    print(goal[:2, 0])
    planning_env.draw_start_goal(start[:2, 0], goal[:2, 0])

    #start = start[:2, :].T / size
    goal_resize = goal[:2, :].T / size
    # import IPython
    # IPython.embed()
    last_angle = 0
    planning_env.setState(start)
    for t in range(500):
        start_tile = np.tile(start, (1, 2))
        start_tile[2, 0] = 0
        obs = planning_env.get_measurement(start_tile)

        # MPnet
        if t % 5 == 0:
            start_goal = np.concatenate([start[:2, :].T / size, goal_resize], axis = 1)
            delta = mpnet.predict(start_goal, obs[:1] / 4.0)
            delta = delta / 20. * size
            next_state = planning_env.steerTo(start[:2, :].T, delta)
            delta = next_state - start[:2, :].T

        # ddpg
        local_goal = np.dot(np.array([[np.cos(-start[2, 0]), -np.sin(-start[2, 0])], [np.sin(-start[2, 0]), np.cos(-start[2, 0])]]), delta.T)
        action = agent.predict(obs[1:] / 4.0, local_goal.T / 20.)

        # execute action
        action[1] = 0.2 * action[1] + last_angle * 0.8
        last_angle = action[1]
        start, _ = planning_env.step_action(action)
        planning_env.render(dt = 0.00001)


if __name__ == "__main__":
    main()
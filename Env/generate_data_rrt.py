import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from CarEnvironment import CarEnvironment
from RRTPlannerNonholonomic import RRTPlannerNonholonomic

parser = argparse.ArgumentParser(description='script for testing planners')

parser.add_argument('-m', '--map', type=str, default='../map/map.txt',
                        help='The environment to plan on')    
parser.add_argument('-eps', '--epsilon', type=float, default=1.0, help='Epsilon for A*')
parser.add_argument('-eta', '--eta', type=float, default=1.0, help='eta for RRT/RRT*')
parser.add_argument('-b', '--bias', type=float, default=0.2, help='Goal bias for RRT/RRT*')

args = parser.parse_args()

def main():
    planning_env = CarEnvironment(args.map)

    it = 0
    max_it = 5
    if not os.path.exists("../data_rrt/"):
        os.mkdir("../data_rrt/")

    while True:
        print("working on %d " % it)
        while True:
            start = planning_env.sample()
            if planning_env.state_validity_checker(start):
                break
        print(start)

        while True:
            goal = planning_env.sample()
            if planning_env.state_validity_checker(goal) and planning_env.compute_distance(start, goal) > 200:
                break
        print(goal)
        planner = RRTPlannerNonholonomic(planning_env, bias=args.bias)
        plan, actions, rollouts = planner.Plan(start, goal)
        measurement = planning_env.get_measurement(rollouts)

        # planning_env.init_visualizer()
        # planning_env.visualize_traj(rollouts, measurement)

        if actions is None:
            continue

        
        np.savetxt("../data_rrt/action_%d.txt" % it, actions)
        np.savetxt("../data_rrt/rollout_%d.txt" % it, rollouts)
        np.savetxt("../data_rrt/measure_%d.txt" % it, measurement)
        

        it += 1
        if it == max_it:
            break


if __name__ == "__main__":
    main()


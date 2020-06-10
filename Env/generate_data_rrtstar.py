import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from CarEnvironment import CarEnvironment
from RRTSTARPlannerNonholonomic import RRTSTARPlannerNonholonomic

parser = argparse.ArgumentParser(description='script for testing planners')

parser.add_argument('-m', '--map', type=str, default='map/map.txt',
                        help='The environment to plan on')    
parser.add_argument('-eps', '--epsilon', type=float, default=1.0, help='Epsilon for A*')
parser.add_argument('-eta', '--eta', type=float, default=1.0, help='eta for RRT/RRT*')
parser.add_argument('-b', '--bias', type=float, default=0.2, help='Goal bias for RRT/RRT*')

args = parser.parse_args()

def main():
    planning_env = CarEnvironment(args.map)

    it = 0
    max_it = 10

    if not os.path.exists("../data_rrtstar/"):
        os.mkdir("../data_rrtstar/")

    while True:
        print("working on %d " % it)
        while True:
            start = planning_env.sample()
            if planning_env.state_validity_checker(start):
                break

        #start = np.array([900 + 10*np.random.random() - 5, 1000 + 10*np.random.random(), np.random.random() * np.pi * 2])[None].T
        while True:
            goal = planning_env.sample()
            dist = planning_env.compute_distance(start, goal)
            if planning_env.state_validity_checker(goal):
                break
        
        planner = RRTSTARPlannerNonholonomic(planning_env, bias=args.bias)
        plan, fail = planner.Plan(start, goal)
        
        if fail:
            continue

        planning_env.init_visualizer()
        tree = planner.tree
        planning_env.visualize_plan(plan, tree, None)

        np.savetxt("../data_rrtstar/plan_%d.txt"%it, plan)

        it += 1
        if it == max_it:
            break


if __name__ == "__main__":
    main()
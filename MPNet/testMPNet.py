import sys
sys.path.append('../Env')
import numpy as np
from MPNet import MPNet
from CarEnvironment import CarEnvironment


if __name__ == "__main__":
    mpnet = MPNet()
    mpnet.load()

    planning_env = CarEnvironment("map/map.png")
    planning_env.init_visualizer()

    size = np.array([1788, 1240])

    while True:
        start = planning_env.sample()
        print(start)
        if planning_env.state_validity_checker(start):
            break

    while True:
        goal = planning_env.sample()
        print(goal)
        dist = planning_env.compute_distance(start, goal)
        if planning_env.state_validity_checker(goal) and dist > 200 and dist < 2000:
            break

    # start = np.array([[900, 1000, 0]]).T
    # goal = np.array([[1300, 600, 0]]).T

    start = start[:2, :].T / size

    print(start)
    goal_ = goal[:2, :].T / size
    plan = []
    plan.append(np.concatenate([start * size, [[0]]], axis = 1))
    delta = np.zeros(2)
    for i in range(100):
        obs = planning_env.get_measurement((start*size).T) / 4.0
        start_goal = np.concatenate([start, goal_], axis = 1)
        delta = mpnet.predict(start_goal, obs)


        start = planning_env.steerTo(start*size, delta / 20. * size)

        #start = start + next_ / 20.
        #print(next_)
        plan.append(np.concatenate([start, [[0]]], axis = 1))
        
        if planning_env.goal_criterion(start.T, goal, 30, no_angle=True):
            print("goal reach!")
            break
        
        start = start / size

    plan.append(goal.T)
    plan = np.array(plan).squeeze()
    planning_env.visualize_plan(plan.T) 

    import IPython
    IPython.embed()


from CarEnvironment import CarEnvironment
import numpy as np 

planning_env = CarEnvironment("../map/map.png")
#plan = np.loadtxt("data_rrtstar/plan_0.txt")
plan = np.loadtxt("../data_rrtstar/data_rrtstar1/plan_0.txt")
print(plan.shape)
planning_env.init_visualizer()
planning_env.visualize_plan(plan, None, None)
import IPython
IPython.embed()
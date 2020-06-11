import sys
sys.path.append('../Env')
import numpy as np
from DPF import DPF
from CarEnvironment import CarEnvironment
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
	
	dpf = DPF()
	dpf.load()

	planning_env = CarEnvironment("../map/map.yaml")
	planning_env.init_visualizer()
	dpf.env = planning_env

	particles = []
	while True:
		p = planning_env.sample()
		if planning_env.state_validity_checker(p):
			particles.append(p)
			if len(particles) >= 200:
				break

	particles = np.array(particles)
	particles = particles.squeeze()[None, ...] / np.array([1788, 1240, 1])
	
	planning_env.reset()
	state = planning_env.state

	# particles = np.array(particles).squeeze()
	obs = planning_env.get_measurement(state)
	encoding = dpf.encoder(torch.FloatTensor(obs/4.0).to(device))
	particles_pro = dpf.propose_particles(encoding, 100).detach()


	#particles = state.repeat(16, axis = 1).T
	dpf.initial_particles(particles)

	for i in range(15):
		action = planning_env.sample_action()
		state, obs = planning_env.step_action(action)
		next_, prob = dpf.update(action, obs)
		pred = (next_ * prob[..., None]).sum(axis = 1)
		planning_env.render(state, dpf.particles.cpu().numpy()*np.array([1788, 1240, 1.0]))
		import IPython
		IPython.embed()


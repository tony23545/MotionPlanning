# MotionPlanning

This is our final project for [Probabilistic Robotics](https://courses.cs.washington.edu/courses/cse571/20sp/). We implemented 3 components for a complete navigation pipeline:

- [Differentiable Particles Filter](https://arxiv.org/pdf/1805.11122.pdf) for localization.
- [Motion Planning Networks](https://arxiv.org/pdf/1806.05767.pdf) for global path planning.
- [Deep Deterministic Policy Gradient](https://arxiv.org/pdf/1509.02971.pdf) for local control.

### run the whole pipeline
```
python main.py
```
### run MPNet and DDPG with ground truth position
```
python MPNet_DDPG.py
```
run the individual module, go to the corresponding folder.

### link to the dataset:  
[rrt* dataset](https://drive.google.com/file/d/1vxX_vBrSBq0mhWsP4usxvzGPjFI33Grp/view?usp=sharing)  
[rrt dataset](https://drive.google.com/file/d/1vHaSx6KfRG5t0hH3ohNb2v90SJkcttvl/view?usp=sharing)
 

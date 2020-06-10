import operator
import numpy as np

class RRTTree(object):
    
    def __init__(self, planning_env):
        
        self.planning_env = planning_env
        self.vertices = []
        self.costs = []
        self.edges = dict()

    def GetRootID(self):
        """ Return the ID of the root in the tree.
        """
        return 0

    def GetNearestVertex(self, config):
        """ Return the nearest state ID in the tree.
            
            @param config: Sampled configuration.
        """
        dists = []
        for v in self.vertices:
            dists.append(self.planning_env.compute_distance(config, v))

        vid, vdist = min(enumerate(dists), key=operator.itemgetter(1))

        return vid, self.vertices[vid]
            
    def GetNNInRad(self, config, rad):
        ''' Return neighbors within ball of radius. Useful for RRT*

            @param config: Sampled configuration.
            @param rad ball radius
        '''
        rad = np.abs(rad)
        vids = []
        vertices = []
        for idx, v in enumerate(self.vertices):
            if self.planning_env.compute_distance(config, v) < rad:
                vids.append(idx)
                vertices.append(v)

        return vids, vertices


    def AddVertex(self, config, cost=0):
        '''
        Add a state to the tree.
        @param config Configuration to add to the tree.
        '''
        vid = len(self.vertices)
        self.vertices.append(config)
        self.costs.append(cost)
        return vid

    def AddEdge(self, sid, eid, action = None, rollout = None):
        '''
        Adds an edge in the tree.
        @param sid start state ID.
        @param eid end state ID.
        '''
        if action is None:
            self.edges[eid] = [sid]
        else:
            self.edges[eid] = (sid, action, rollout)
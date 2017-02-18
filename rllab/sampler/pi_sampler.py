

from rllab.algos.batch_polopt import BatchSampler
import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger

class PISampler(BatchSampler):
    def __init__(self, algo):
        super(PISampler,self).__init__(algo)

    def process_samples(self, itr, paths):

        # add PI stuff here...
        samples_data = super(PISampler,self).process_samples(itr,paths)

        print('processing samples in PISampler----------')

        # construct alternative structures of fixed size rollouts
        # N is number of rollouts
        N = int(self.algo.batch_size/self.algo.max_path_length)
        # T is number of time-steps
        T = self.algo.max_path_length

        # V is NxT matrix of state costs 
        V = np.zeros((N,T))
        # tensor of NxTxu, where u is action dimensions
        U = np.zeros((N,T,self.algo.env.action_dim))
        # tensor of NxTxs, where s is state dimensions
        X = np.zeros((N,T,self.algo.policy.observation_space.flat_dim))
       
        for i in range(0,N) :
            #print(paths[i]["rewards"])
            num_steps = paths[i]["rewards"].size
            U[i,0:num_steps,:] = paths[i]["actions"]
            X[i,0:num_steps,:] = paths[i]["observations"]
            V[i,0:num_steps] = -paths[i]["rewards"]

        samples_data["V"] = V
        samples_data["U"] = U.reshape(N*T,2)
        samples_data["X"] = X.reshape(N*T,2)

        return samples_data

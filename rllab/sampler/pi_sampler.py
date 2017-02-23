

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

        samples_data = super(PISampler,self).process_samples(itr,paths)

        # add PI stuff here...
        print('processing samples in PISampler----------')

        # construct alternative structures of fixed size rollouts
        # N is number of rollouts
        N = len(paths)

        # T is number of time-steps
        T = self.algo.max_path_length

        # V is NxT matrix of state costs 
        V = np.zeros((N,T))
        # tensor of NxTxu, where u is action dimensions
        udim = self.algo.env.action_dim

        U = np.zeros((N,T,udim))
        # tensor of NxTxs, where s is state dimensions
        xdim = self.algo.policy.observation_space.flat_dim
        X = np.zeros((N,T,xdim))

        for i in range(0,N) :
            num_steps = paths[i]["rewards"].size
            if (num_steps != T) :
                logger.log("----------- truncated episode "
                    + str(num_steps) + " < "
                    + str(T)
                )
            U[i,0:num_steps,:] = paths[i]["actions"]
            X[i,0:num_steps,:] = paths[i]["observations"]
            V[i,0:num_steps] = -paths[i]["rewards"]

        samples_data["V"] = V
        samples_data["X"] = X.reshape(N*T,xdim)
        samples_data["U"] = U.reshape(N*T,udim)

        D = dict()
        agent_infos2 = dict()
        for key,val in paths[0]["agent_infos"].items() :
            D[key] = np.zeros((N,T,val.shape[1]))
        for i in range(0,N) :
            for key,val in paths[i]["agent_infos"].items() :
                num_steps = val.shape[0]
                D[key][i,0:num_steps,:] = val
        for key,val in paths[0]["agent_infos"].items() :
            agent_infos2[key] = D[key].reshape(N*T,val.shape[1])
        samples_data["agent_infos2"] = agent_infos2

        return samples_data

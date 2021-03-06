from rllab.algos.batch_polopt import BatchSampler
import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger

class AKLSampler(BatchSampler):
    def __init__(self, algo):
        super(AKLSampler,self).__init__(algo)

    def process_samples(self, itr, paths):

        samples_data = super(AKLSampler,self).process_samples(itr,paths)

        # construct alternative structures of fixed size rollouts
        # N is number of rollouts
        N = int(self.algo.batch_size/self.algo.max_path_length)
        #N = len(paths)

        # T is number of time-steps
        T = self.algo.max_path_length

        # For the moment, we discard those rollouts that not reach the
        # time-horizon, which means they have infinite cost
        Neff = N;
        
        # tensor of NxTxs, where s is state dimensions
        xdim = self.algo.policy.observation_space.flat_dim
        X = np.zeros((Neff,T,xdim))
        # V is NxT matrix of state costs 
        V = np.zeros((Neff,T))
        # tensor of NxTxu, where u is action dimensions
        udim = self.algo.env.action_dim
        U = np.zeros((Neff,T,udim))
        mask = np.zeros((Neff,T))

        for i in range(0,Neff) :
            steps = paths[i]["rewards"].size
            U[i,0:steps,:] = paths[i]["actions"]
            X[i,0:steps,:] = paths[i]["observations"]
            V[i,0:steps] = -paths[i]["rewards"]
            mask[i,0:steps] = 1.

        samples_data["V"] = V
        samples_data["X"] = X.reshape(Neff*T,xdim) #the reshape needs to be done for the neural network...
        samples_data["U"] = U.reshape(Neff*T,udim) #the reshape needs to be done for the neural network...
        
        samples_data["mask"] = mask
        

        D = dict()
        agent_infos2 = dict()
        for key,val in paths[0]["agent_infos"].items() :
            D[key] = np.zeros((Neff,T,val.shape[1]))
        for i in range(0,Neff) :
            for key,val in paths[i]["agent_infos"].items() :
                num_steps = val.shape[0]
                D[key][i,0:num_steps,:] = val
        for key,val in paths[0]["agent_infos"].items() :
            agent_infos2[key] = D[key].reshape(Neff*T,val.shape[1])
        samples_data["agent_infos2"] = agent_infos2

        return samples_data



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

        # matrix of V state costs
        # if rollout does not reach horizon reward is zero-padded
        N = int(self.algo.batch_size/self.algo.max_path_length)
        # N is number of rollouts
        T = self.algo.max_path_length
        # T is number of time-steps
        V = np.zeros((N,T))
        print(V.shape)
        for i in range(0,N) :
            #print(paths[i]["rewards"])
            V[i,0:paths[i]["rewards"].size] = np.exp(-paths[i]["rewards"])

        #print(V)
        print("bye")
        samples_data["V"] = V

        return samples_data

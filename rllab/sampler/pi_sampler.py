import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger

class PISampler(BaseSampler):
    def __init__(self, algo):
        super(BaseSampler,self).__init__(algo)

    def process_samples(self, itr, paths):

        # add PI stuff here...
        samples_date = super(BaseSampler,self).process_samples(itr,paths)

        return samples_data

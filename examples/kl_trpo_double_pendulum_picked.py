from rllab.misc.instrument import run_experiment_lite
from rllab.algos.kl_trpo import KLTRPO
from rllab.algos.npirepsexperiment import NPIREPS
from rllab.sampler.pi_sampler import PISampler
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
#from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.box2d.kl_double_pendulum_env import KLDoublePendulumEnv
#from rllab.envs.box2d.kl_time_double_pendulum_env import KLDoublePendulumEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.KL_gaussian_mlp_policy import KL_GaussianMLPPolicy
import lasagne.nonlinearities as NL
from time import gmtime, strftime


import rllab.misc.logger as logger

import numpy as np
import sys

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

if len(sys.argv) != 6 :
    print('Use as: python fname.py keyword epsilon N seed n_parallel')
else :
    variant = 'kl_trpo'
    keyword = sys.argv[1]
    epsilon = np.float(sys.argv[2])
    N= np.int(sys.argv[3])
    seed = np.int(np.float(sys.argv[4]))
    n_parallel = np.int(sys.argv[5])
              
    kl_trpo = True if variant == 'kl_trpo' else False
    
    plot = False 
    
    def run_task(*_):
        env = normalize(KLDoublePendulumEnv())
    
        print("Action dims = " + str(env.action_dim))
        print("obs dim = " + str(env.observation_space.flat_dim))
        
        
    
        policy = KL_GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32,32),
            learn_std = True,
            hidden_nonlinearity=NL.tanh,
        )
    
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    
        algo = KLTRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            sampler_cls=PISampler,
            kl_trpo=kl_trpo,
            step_size = epsilon,
            plot=plot,
            batch_size=N*100,
        )
    
        logger.log("    variant " + variant)
        logger.log("    eps " + str(epsilon))
        logger.log("    seed " + str(seed))
        logger.log("    keyword " + keyword)
    
        algo.train()
    
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=n_parallel,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        
        seed=seed,
        plot=plot,
        exp_prefix="newsweeps2",
        exp_name='sweep'+strftime("%Y-%m-%d %H:%M:%S", gmtime())+'variant '+variant+'keyword '+keyword+'eps '+str(epsilon)+'seed '+str(seed)+'N '+str(N)
    )
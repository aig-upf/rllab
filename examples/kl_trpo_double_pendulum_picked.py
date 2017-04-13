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
import lasagne.init as LI
from time import gmtime, strftime


import rllab.misc.logger as logger

import numpy as np
import sys

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

if len(sys.argv) != 7 :
    print('Use as: python fname.py keyword epsilon N seed n_parallel')
else :
    variant = 'kl_trpo'
    keyword = sys.argv[1]
    epsilon = np.float(sys.argv[2])
    N= np.int(sys.argv[3])
    seed = np.int(np.float(sys.argv[4]))
    n_parallel = np.int(sys.argv[5])
    which_policy = sys.argv[6]
              
    kl_trpo = True if variant == 'kl_trpo' else False
    
    plot = False 
    
    def run_task(*_):
        env = normalize(KLDoublePendulumEnv())
    
        print("Action dims = " + str(env.action_dim))
        print("obs dim = " + str(env.observation_space.flat_dim))
        
        
    
        policy1 = KL_GaussianMLPPolicy(
            env_spec=env.spec,
            learn_std = True,
            std_hidden_sizes=(32, 32),
            hidden_sizes=(32,32),
            std_hidden_nonlinearity=NL.tanh,
            hidden_nonlinearity=NL.tanh, #NL.rectify
            hidden_W_init_mean=LI.GlorotUniform(),
            hidden_W_init_std=LI.GlorotUniform() #LI.Orthogonal('relu')
        )
        
        policy2 = KL_GaussianMLPPolicy(
            env_spec=env.spec,
            learn_std = True,
            std_hidden_sizes=(50,50,50,50,50),
            hidden_sizes=(50,50,50,50,50),
            std_hidden_nonlinearity=NL.rectify,
            hidden_nonlinearity=NL.rectify, #NL.rectify
            hidden_W_init_mean=LI.Orthogonal('relu'),
            hidden_W_init_std=LI.Orthogonal('relu') #LI.Orthogonal('relu')
        )
        
        if which_policy=='policy1':
            policy = policy1
        else:
            policy = policy2
    
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
        logger.log("    policy " + which_policy)
    
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
        exp_prefix="npirepstests",
        exp_name='sweep'+strftime("%Y-%m-%d %H:%M:%S", gmtime())+'variant '+variant+'keyword '+keyword+'eps '+str(epsilon)+'seed '+str(seed)+'N '+str(N)+which_policy
    )
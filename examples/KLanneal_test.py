
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.kl_trpo import KLTRPO
from rllab.algos.KLannealing import KLANNEAL
#from rllab.algos.npireps import NPIREPS as KLANNEAL
from rllab.sampler.annealkl_sampler import AKLSampler
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

if len(sys.argv) != 3 :
    print('Use as: python fname.py n_parallel n_itr')
else :
    variant = 'KLannealing'
    keyword = 'debug'
    delta = 0.8
    epsilon = 0.1
    N= 200
    seed = 1708
    n_parallel = np.int(sys.argv[1])
    
    
    stduncontrolled = 1.0
    lambd = 0.1
    n_itr = np.int(sys.argv[2])
    
    plot = True 
    
    def run_task(*_):
        env = normalize(KLDoublePendulumEnv())
    
        print("Action dims = " + str(env.action_dim))
        print("obs dim = " + str(env.observation_space.flat_dim))
        
        
    
        policy = KL_GaussianMLPPolicy(
            env_spec=env.spec,
            learn_std = False,
            init_std=stduncontrolled, #set the std of the NN equal to the uncontrolled std
            hidden_sizes=(32,32),
            hidden_nonlinearity=NL.tanh, #NL.rectify
            hidden_W_init_mean=LI.GlorotUniform(),
            hidden_W_init_std=LI.GlorotUniform() #LI.Orthogonal('relu')
        )
        
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    
        algo = KLANNEAL(
            env=env,
            policy=policy,
            baseline=baseline,
            sampler_cls=AKLSampler,
            step_size = epsilon,
            plot=plot,
            max_path_length=100,
            n_itr=n_itr,
            batch_size=N*100,
            delta=delta,
            log_std_uncontrolled= np.log(stduncontrolled),
            lambd = lambd,
            algorithm = 'Algorithm2', #which PoD to take
            PoF = 'KL_CE2', #which PoF to take
            optim = 'firstorder',#'Lbfgs',
        )
    
        logger.log("    variant " + variant)
        logger.log("    eps " + str(epsilon))
        logger.log("    seed " + str(seed))
        logger.log("    keyword " + keyword)
        logger.log("    log_std_uncontrolled " + str(np.log(stduncontrolled)))
        logger.log("    lambda " + str(lambd))
        logger.log("    n_itr " + str(n_itr))
    
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
        exp_prefix=keyword,
        exp_name='debug'
    )
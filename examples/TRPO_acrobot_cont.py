
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.algos.KLannealing import KLANNEAL
from rllab.sampler.annealkl_sampler import AKLSampler
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from rllab.envs.normalized_env import normalize
from rllab.policies.KL_gaussian_mlp_policy import KL_GaussianMLPPolicy
import lasagne.nonlinearities as NL
import lasagne.init as LI
from time import gmtime, strftime
from rllab.envs.gym_env import GymEnv
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy



import rllab.misc.logger as logger

import numpy as np
import sys

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

if len(sys.argv) != 14 :
    print('we need more arguments as input here... sorry I will not list them all please look into the script ;P')
else :
    keyword = sys.argv[1]
    delta = np.float(sys.argv[2])#0.8
    epsilon = np.float(sys.argv[3])#0.1
    N= np.int(sys.argv[4]) #200
    seed = np.int(sys.argv[5]) #1708
    n_parallel = np.int(sys.argv[6])
    algorithm =sys.argv[7]
    PoF =sys.argv[8]
    optim =sys.argv[9]
    stduncontrolled = np.float(sys.argv[10]) #1.0
    lambd = np.float(sys.argv[11]) #0.1
    n_itr = np.int(sys.argv[12])
    
    plot = np.bool(np.int(sys.argv[13]))
    
    def run_task(*_):
        env = normalize(GymEnv("Acrobot-v3", record_video=False, record_log=False))
    
        print("Action dims = " + str(env.action_dim))
        print("obs dim = " + str(env.observation_space.flat_dim))
        
        
    
        policy = KL_GaussianMLPPolicy(
            env_spec=env.spec,
            learn_std = True,
            init_std=stduncontrolled, #set the std of the NN equal to the uncontrolled std
            hidden_sizes=(50,50),
            hidden_nonlinearity=NL.rectify,
            hidden_W_init_mean=LI.Orthogonal('relu'),
            hidden_W_init_std=LI.Orthogonal('relu')
        )
        
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            n_itr=n_itr,
            plot=True,
            max_path_length=env.horizon,
            batch_size=N*env.horizon,
            step_size=delta,
        )

        logger.log("    eps " + str(epsilon))
        logger.log("    seed " + str(seed))
        logger.log("    keyword " + keyword)
        logger.log("    log_std_uncontrolled " + str(np.log(stduncontrolled)))
        logger.log("    lambda " + str(lambd))
        logger.log("    n_itr " + str(n_itr))
        logger.log('    algorithm '+str(algorithm))
        logger.log('    PoF '+str(PoF))
        logger.log('    optim '+str(optim))
    
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
        exp_name='sweep'+strftime("%Y-%m-%d %H:%M:%S", gmtime())+'keyword '+keyword+'eps '+str(epsilon)+'seed '+str(seed)+'delta '+str(delta)+'N '+str(N)+'algorithm '+str(algorithm)+'PoF '+str(PoF)+'optim '+str(optim)
    )
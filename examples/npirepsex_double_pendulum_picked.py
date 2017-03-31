from rllab.misc.instrument import run_experiment_lite
from rllab.algos.npirepsexperiment import NPIREPS
from rllab.sampler.pi_sampler import PISampler
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
#from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.box2d.kl_double_pendulum_env import KLDoublePendulumEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

import rllab.misc.logger as logger

import numpy as np
import sys

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

if len(sys.argv) != 8 :
    print('Use as: python fname.py {kl_trpo|npireps} keywork delta epsilon N seed n_parallel')
else :
    variant = sys.argv[1]
    keyword = sys.argv[2]
    delta = np.float(sys.argv[3])
    epsilon = np.float(sys.argv[4])
    N= np.int(sys.argv[5])
    seed = np.int(sys.argv[6])
    n_parallel = np.int(sys.argv[7])
              
    kl_trpo = True if variant == 'kl_trpo' else False
    
    plot = False 
    
    def run_task(*_):
        env = normalize(KLDoublePendulumEnv())
    
        print("Action dims = " + str(env.action_dim))
        print("obs dim = " + str(env.observation_space.flat_dim))
    
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
        )
    
        baseline = LinearFeatureBaseline(env_spec=env.spec)
    
        algo = NPIREPS(
            env=env,
            policy=policy,
            baseline=baseline,
            sampler_cls=PISampler,
            kl_trpo=kl_trpo,
            step_size = epsilon,
            plot=plot,
            batch_size=N*100,
            delta=delta
        )
    
        logger.log("    variant " + variant)
        logger.log("    eps " + str(epsilon))
        logger.log("    seed " + str(seed))
    
        algo.train()
    
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=n_parallel,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        
        seed=10,
        plot=plot,
        exp_prefix="sweepsex1",
        exp_name='sweep'+variant+keyword+str(epsilon)+str(seed)+str(delta)+str(N)
    )
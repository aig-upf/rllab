from rllab.misc.instrument import run_experiment_lite
from rllab.algos.npireps import NPIREPS
from rllab.sampler.pi_sampler import PISampler
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.misc.instrument import run_experiment_lite

import rllab.misc.logger as logger

import numpy as np
import sys


if len(sys.argv) != 5 :
    print('Use as: python fname.py {kl_trpo|npireps} delta epsilon seed')
else :
    variant = sys.argv[1]
    delta = np.float(sys.argv[2])
    epsilon = np.float(sys.argv[3])
    seed = np.int(sys.argv[4])
    
    kl_trpo = True if variant == 'kl_trpo' else False
    
    plot = True 
    
    def run_task(*_):
        env = normalize(GymEnv("Pendulum-v0"))

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
            delta=delta
        )
    
        logger.log("    variant " + variant)
        logger.log("    eps " + str(epsilon))
        logger.log("    seed " + str(seed))
    
        algo.train()
    
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=10,
        plot=plot,
        exp_prefix="sweeps1",
        exp_name='sweep'+variant+str(epsilon)+str(seed)+str(delta)
    )

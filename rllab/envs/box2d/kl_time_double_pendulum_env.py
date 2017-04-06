import numpy as np
from rllab.envs.box2d.parser import find_body

from rllab.core.serializable import Serializable
from rllab.envs.box2d.kl_box2d_env import KLBox2DEnv
from rllab.misc import autoargs
from rllab.misc.overrides import overrides
from rllab import spaces
from rllab.envs.base import Env, Step

# http://mlg.eng.cam.ac.uk/pilco/
BIG = 1e6
class KLDoublePendulumEnv(KLBox2DEnv, Serializable):

    @autoargs.inherit(KLBox2DEnv.__init__)
    def __init__(self, *args, **kwargs):
        # make sure mdp-level step is 100ms long
        kwargs["frame_skip"] = kwargs.get("frame_skip", 2)
        if kwargs.get("template_args", {}).get("noise", False):
            self.link_len = (np.random.rand()-0.5) + 1
        else:
            self.link_len = 1
        kwargs["template_args"] = kwargs.get("template_args", {})
        kwargs["template_args"]["link_len"] = self.link_len
        super(KLDoublePendulumEnv, self).__init__(
            self.model_path("double_pendulum.xml.mako"),
            *args, **kwargs
        )
        self.link1 = find_body(self.world, "link1")
        self.link2 = find_body(self.world, "link2")
        self.time=0.
        Serializable.__init__(self, *args, **kwargs)

    @overrides
    def reset(self):
        self._set_state(self.initial_state)
        self._invalidate_state_caches()
        stds = np.array([0.1, 0.1, 0.01, 0.01])
        pos1, pos2, v1, v2 = np.random.randn(*stds.shape) * stds*0.
        self.link1.angle = pos1
        self.link2.angle = pos2
        self.link1.angularVelocity = v1
        self.link2.angularVelocity = v2
        self.time=0.
        next_obs = self.get_current_obs()
        next_obs=np.append(next_obs,self.time)
        return next_obs

    def get_tip_pos(self):
        cur_center_pos = self.link2.position
        cur_angle = self.link2.angle
        cur_pos = (
            cur_center_pos[0] - self.link_len*np.sin(cur_angle),
            cur_center_pos[1] - self.link_len*np.cos(cur_angle)
        )
        return cur_pos

    @overrides
    def compute_reward(self, action):
        yield
        tgt_pos = np.asarray([0, self.link_len * 2])
        cur_pos = self.get_tip_pos()
        dist = np.linalg.norm(cur_pos - tgt_pos)
        yield -dist

    def is_current_done(self):
        return False

    @overrides
    def step(self, action):
        """
        Note: override this method with great care, as it post-processes the
        observations, etc.
        """
        reward_computer = self.compute_reward(action)
        # forward the state
        action = self._inject_action_noise(action)
        for _ in range(self.frame_skip):
            self.forward_dynamics(action)
        # notifies that we have stepped the world
        next(reward_computer)
        # actually get the reward
        reward = next(reward_computer)
        self._invalidate_state_caches()
        done = self.is_current_done()
        next_obs = self.get_current_obs()
        self.time+=1.
        next_obs=np.append(next_obs,self.time)
        return Step(observation=next_obs, reward=reward, done=done)
    
    @property
    @overrides
    def observation_space(self):
        if self.position_only:
            d = len(self._get_position_ids())
        else:
            d = len(self.extra_data.states)
        ub = BIG * np.ones(d+1)
        return spaces.Box(ub*-1, ub)
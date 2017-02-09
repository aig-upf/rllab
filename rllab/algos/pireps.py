import theano.tensor as TT
import theano
import scipy.optimize
from rllab.misc import logger
from rllab.misc.overrides import overrides
from rllab.misc import ext
from rllab.algos.batch_polopt import BatchPolopt
from rllab.core.serializable import Serializable
import numpy as np
from rllab.misc import tensor_utils


class PIREPS(BatchPolopt, Serializable):
    """
    Policy Search for Path Integral Control (PI-REPS)

    References
    ----------

    Parameters:
        epsilon         Max KL divergence between new policy and old policy.

    Dual PIREPS function related variables:
        param_eta       dual variable : optimal dual epsilon : 1x1
        param_theta     dual variable : multiplies features : nx1
        qprob           prob of sample trajectory (cumsum policy prob) : Nx1
        rewards         sum of rewards : Nx1
        features        features x_0 : Nxn

    """

    def __init__(
            self,
            epsilon=0.5,
            max_opt_itr=50,
            optimizer=scipy.optimize.fmin_l_bfgs_b,
            **kwargs):
        """

        :param epsilon: Max KL divergence between new policy and old policy.
        :param max_opt_itr: Maximum number of batch optimization iterations.
        :param optimizer: Module path to the optimizer. It must support the same interface as
        scipy.optimize.fmin_l_bfgs_b.
        :return:
        """
        Serializable.quick_init(self, locals())
        super(PIREPS, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.max_opt_itr = max_opt_itr
        self.optimizer = optimizer
        self.opt_info = None

    @overrides
    def init_opt(self):
        is_recurrent = int(self.policy.recurrent)

        # Init dual param values
        self.param_eta = 15.
        # Adjust for linear feature vector.
        self.param_theta = np.random.rand(self.env.observation_space.flat_dim * 2 + 4)

        # Theano vars
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1 + is_recurrent,
        )
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1 + is_recurrent,
        )
        rewards = ext.new_tensor(
            'rewards',
            ndim=1 + is_recurrent,
            dtype=theano.config.floatX,
        )
        # Feature difference variable representing the difference in feature
        # value of the next observation and the current observation \phi(s') -
        # \phi(s).
        feat_diff = ext.new_tensor(
            'feat_diff',
            ndim=2 + is_recurrent,
            dtype=theano.config.floatX
        )
        param_v = TT.vector('param_v')
        param_eta = TT.scalar('eta')

        valid_var = TT.matrix('valid')

        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
        }
        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]

        # Policy-related symbolics
        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
        dist = self.policy.distribution
        # log of the policy dist
        logli = dist.log_likelihood_sym(action_var, dist_info_vars)

        # Symbolic sample Bellman error
        delta_v = rewards + TT.dot(feat_diff, param_v)

        # Policy loss (negative because we minimize)
        loss = - TT.mean(logli * TT.exp(
            delta_v / param_eta - TT.max(delta_v / param_eta)
        ))

        # Add regularization to loss.
        reg_params = self.policy.get_params(regularizable=True)
        loss += self.L2_reg_loss * TT.sum(
            [TT.mean(TT.square(param)) for param in reg_params]
        ) / len(reg_params)

        # Policy loss gradient.
        loss_grad = TT.grad(
            loss, self.policy.get_params(trainable=True))

        input = [rewards, obs_var, feat_diff,
                 action_var] + state_info_vars_list + [param_eta, param_v]

        f_loss = ext.compile_function(
            inputs=input,
            outputs=loss,
        )
        f_loss_grad = ext.compile_function(
            inputs=input,
            outputs=loss_grad,
        )

        # Debug prints
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2 + is_recurrent,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
            }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]

        mean_kl = TT.mean(dist.kl_sym(old_dist_info_vars, dist_info_vars))

        f_kl = ext.compile_function(
            inputs=[obs_var, action_var] + state_info_vars_list +
            old_dist_info_vars_list,
            outputs=mean_kl,
        )

        # Dual-related symbolics
        # Symbolic dual
        R = 'rewards'
        log_prob = 'logprob'
        phi_mat = 'stateFeatures'
        N = 'numsamples'

        V = phi_mat * param_theta
        V_hat = phi_hat * param_theta
        adv = (R - V + log_prob*param_eta)
        max_adv = TT.max(adv)
        dual = (param_eta+1) * TT.log(
                    1/N * TT.sum(
                        TT.exp(adv)
                    )
                )
        dual += param_eta*self.epsilon + V_hat + (param_eta+1)*max_adv

        # Symbolic dual gradient
        dual_grad = TT.grad(cost=dual, wrt=[param_eta, param_v])

        # Eval functions.
        f_dual = ext.compile_function(
            inputs=[rewards, feat_diff] + state_info_vars_list + [param_eta, param_v],
            outputs=dual
        )
        f_dual_grad = ext.compile_function(
            inputs=[rewards, feat_diff] + state_info_vars_list + [param_eta, param_v],
            outputs=dual_grad
        )

        self.opt_info = dict(
            f_loss_grad=f_loss_grad,
            f_loss=f_loss,
            f_dual=f_dual,
            f_dual_grad=f_dual_grad,
            f_kl=f_kl
        )

    def _features(self, path):
        o = np.clip(path["observations"], -10, 10)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

    @overrides
    def optimize_policy(self, itr, samples_data):
        # Init vars
        rewards = samples_data['rewards']
        actions = samples_data['actions']
        observations = samples_data['observations']

        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        # Compute sample Bellman error.
        feat_diff = []
        for path in samples_data['paths']:
            feats = self._features(path)
            feats = np.vstack([feats, np.zeros(feats.shape[1])])
            feat_diff.append(feats[1:] - feats[:-1])

        feat_diff = np.vstack(feat_diff)

        #################
        # Optimize dual #
        #################

        # Here we need to optimize dual through BFGS in order to obtain \eta
        # value. Initialize dual function g(\theta, v). \eta > 0
        # First eval delta_v
        f_dual = self.opt_info['f_dual']
        f_dual_grad = self.opt_info['f_dual_grad']

        # Set BFGS eval function
        def eval_dual(input):
            param_eta = input[0]
            param_v = input[1:]
            val = f_dual(*([rewards, feat_diff] + state_info_list + [param_eta, param_v]))
            return val.astype(np.float64)

        # Set BFGS gradient eval function
        def eval_dual_grad(input):
            param_eta = input[0]
            param_v = input[1:]
            grad = f_dual_grad(*([rewards, feat_diff] + state_info_list + [param_eta, param_v]))
            eta_grad = np.float(grad[0])
            v_grad = grad[1]
            return np.hstack([eta_grad, v_grad])

        # Initial BFGS parameter values.
        x0 = np.hstack([self.param_eta, self.param_v])

        # Set parameter boundaries: \eta>0, v unrestricted.
        bounds = [(-np.inf, np.inf) for _ in x0]
        bounds[0] = (0., np.inf)

        # Optimize through BFGS
        logger.log('optimizing dual')
        eta_before = x0[0]
        dual_before = eval_dual(x0)
        params_ast, _, _ = self.optimizer(
            func=eval_dual, x0=x0,
            fprime=eval_dual_grad,
            bounds=bounds,
            maxiter=self.max_opt_itr,
            disp=0
        )
        dual_after = eval_dual(params_ast)

        # Optimal values have been obtained
        self.param_eta = params_ast[0]
        self.param_v = params_ast[1:]

        ###################
        # Optimize policy #
        ###################
        cur_params = self.policy.get_param_values(trainable=True)
        f_loss = self.opt_info["f_loss"]
        f_loss_grad = self.opt_info['f_loss_grad']
        input = [rewards, observations, feat_diff,
                 actions] + state_info_list + [self.param_eta, self.param_v]

        # Set loss eval function
        def eval_loss(params):
            self.policy.set_param_values(params, trainable=True)
            val = f_loss(*input)
            return val.astype(np.float64)

        # Set loss gradient eval function
        def eval_loss_grad(params):
            self.policy.set_param_values(params, trainable=True)
            grad = f_loss_grad(*input)
            flattened_grad = tensor_utils.flatten_tensors(list(map(np.asarray, grad)))
            return flattened_grad.astype(np.float64)

        loss_before = eval_loss(cur_params)
        logger.log('optimizing policy')
        params_ast, _, _ = self.optimizer(
            func=eval_loss, x0=cur_params,
            fprime=eval_loss_grad,
            disp=0,
            maxiter=self.max_opt_itr
        )
        loss_after = eval_loss(params_ast)

        f_kl = self.opt_info['f_kl']

        mean_kl = f_kl(*([observations, actions] + state_info_list + dist_info_list)).astype(
            np.float64)

        logger.log('eta %f -> %f' % (eta_before, self.param_eta))

        logger.record_tabular("LossBefore", loss_before)
        logger.record_tabular("LossAfter", loss_after)
        logger.record_tabular('DualBefore', dual_before)
        logger.record_tabular('DualAfter', dual_after)
        logger.record_tabular('MeanKL', mean_kl)

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )

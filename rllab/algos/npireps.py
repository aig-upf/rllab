from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
import rllab.misc.logger as logger
import theano
from theano import printing
import theano.tensor as TT
import numpy as np
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer

import matplotlib.pyplot as plt

class NPIREPS(BatchPolopt):
    """
    Natural Policy Optimization.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            truncate_local_is_ratio=None,
            std_uncontrolled=1,
            **kwargs
    ):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.truncate_local_is_ratio = truncate_local_is_ratio
        self.opt_info = None
        self.std_uncontrolled=std_uncontrolled
        self.param_eta = 0.
        super(NPIREPS, self).__init__(**kwargs)


    @overrides
    def init_opt(self):

        # N is number of rollouts
        N = int(self.batch_size/self.max_path_length)
        # T is number of time-steps
        T = self.max_path_length

        # Theano vars
        obs_var = self.env.observation_space.new_tensor_variable(
            'obs',
            extra_dims=1,
        )   # corresponds to state_features
        action_var = self.env.action_space.new_tensor_variable(
            'action',
            extra_dims=1,
        )
        X_var = ext.new_tensor(
            'X',
            ndim=2,
            dtype=theano.config.floatX,
        )   # a N*Txstate_dim array 
        U_var = ext.new_tensor(
            'U',
            ndim=2,
            dtype=theano.config.floatX,
        )   # a N*Txaction_dim array 
        V_var = ext.new_tensor(
            'V',
            ndim=2,
            dtype=theano.config.floatX,
        )   # corresponds to V
        param_eta = TT.scalar('eta')

        dist_info_vars = self.policy.dist_info_sym(X_var)
        dist = self.policy.distribution
        logptheta = dist.log_likelihood_sym(U_var, dist_info_vars)

        logq = TT.log(1/TT.sqrt(2*self.std_uncontrolled*np.pi))
        logptheta_reshaped = logptheta.reshape((N,T))
        S = -(TT.sum(V_var + logptheta_reshaped - logq,1))*(1/(1+param_eta))
        w = TT.exp(S - TT.max(S))
        Z = TT.sum(w)
        w = (w/Z).reshape((w.size,1))
        norm_entropy = -(N/TT.log(N)) * TT.tensordot(w, TT.log(w))
        # line search
        

        #weights = TT.exp(-TT.sum(S,1))

        input = [X_var, U_var, V_var, param_eta]
        #pr_op = printing.Print('obs_var')
        #printed_x = pr_op(obs_var) + pr_op(action_var)
        f_obj = ext.compile_function(
            inputs=input,
            outputs=[norm_entropy,w]
        )
        
        self.opt_info = dict(
            f_obj = f_obj
        )

        ##########################
        # Policy-related symbolics
        ##########################



#        dist = self.policy.distribution
#        old_dist_info_vars = {
#            k: ext.new_tensor(
#                'old_%s' % k,
#                ndim=2,
#                dtype=theano.config.floatX
#            ) for k in dist.dist_info_keys
#            }
#        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]
#
#        state_info_vars = {
#            k: ext.new_tensor(
#                k,
#                ndim=2,
#                dtype=theano.config.floatX
#            ) for k in self.policy.state_info_keys
#        }
#        state_info_vars_list = [state_info_vars[k] for k in self.policy.state_info_keys]
#
#        dist_info_vars = self.policy.dist_info_sym(obs_var, state_info_vars)
#        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
#        lr = dist.likelihood_ratio_sym(action_var, old_dist_info_vars, dist_info_vars)
#        
#        if self.truncate_local_is_ratio is not None:
#            lr = TT.minimum(self.truncate_local_is_ratio, lr)
#        mean_kl = TT.mean(kl)
#        surr_loss = - TT.mean(lr)
#
#        input_list = [
#                         obs_var,
#                     ] + state_info_vars_list + old_dist_info_vars_list
#
#        self.optimizer.update_opt(
#            loss=surr_loss,
#            target=self.policy,
#            leq_constraint=(mean_kl, self.step_size),
#            inputs=input_list,
#            constraint_name="mean_kl"
#        )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):

        # collect data from samples
        all_input_values = [samples_data["X"], samples_data["U"],
                            samples_data["V"]]
        agent_infos = samples_data["agent_infos"]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]

        input_values = all_input_values + [self.param_eta]
        f_dual = self.opt_info['f_obj']
        print('calling dummy function')

        nit = 10
        veta = np.zeros(nit)
        vent = np.zeros(nit)
        rang = np.linspace(0,10,nit)
        for i in np.arange(len(rang)):
            self.param_eta = rang[i]
            input_values = all_input_values + [self.param_eta]
            entropy, weights = f_dual(*input_values)
            veta[i] = self.param_eta
            vent[i] = entropy

        print(vent.T)
#        plt.plot(veta, vent)
#        plt.show()

#        n_iter = 0
#        backtrack_ratio = 0.8
#        max_backtracks = 15
#        for n_iter, ratio in enumerate(backtrack_ratio ** np.arange(max_backtracks)):
#            cur_step = ratio * flat_descent_step
#            cur_param = prev_param - cur_step
#            self._target.set_param_values(cur_param, trainable=True)
#            entropy, weights = f_dual(*all_input_values)
#            if entropy < entropy_before:
#                break
#        if (np.isnan(loss) or np.isnan(constraint_val) or loss >= loss_before or constraint_val >=
# 
        
        
#        loss_before = self.optimizer.loss(all_input_values)
#        mean_kl_before = self.optimizer.constraint_val(all_input_values)
#        
#        # call optimize
#        self.optimizer.optimize(all_input_values)
#        mean_kl = self.optimizer.constraint_val(all_input_values)
#        loss_after = self.optimizer.loss(all_input_values)
#        logger.record_tabular('LossBefore', loss_before)
#        logger.record_tabular('LossAfter', loss_after)
#        logger.record_tabular('MeanKLBefore', mean_kl_before)
#        logger.record_tabular('MeanKL', mean_kl)
#        logger.record_tabular('dLoss', loss_before - loss_after)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )

from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
from rllab.optimizers.KL_conjugate_gradient_optimizer import ConjugateGradientOptimizer
import rllab.misc.logger as logger
import numpy as np
import theano.tensor as TT
import theano
from theano import printing

import matplotlib.pyplot as plt
import rllab.misc.logger as logger

class KLTRPO(BatchPolopt):
    """
    kl_trpo
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            truncate_local_is_ratio=None,
            step_size=0.01,                 # epsilon for 2nd linesearch
            log_std_uncontrolled= 0,   # log_std passive dynamics
            lambd = 1,                      # divides state-cost
            **kwargs
    ):
        if optimizer_args is None:
            optimizer_args = dict()
        optimizer = ConjugateGradientOptimizer(**optimizer_args)
        #optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        self.optimizer = optimizer
        self.step_size = step_size
        self.truncate_local_is_ratio = truncate_local_is_ratio
        self.opt_info = None
        self.log_std_uncontrolled=log_std_uncontrolled
        self.param_eta = 0.
        self.final_rel_entropy = 0.

        self.f_dual = None
        self.f_opt = None
        self.lambd = lambd
        
        super(KLTRPO, self).__init__(optimizer=optimizer, **kwargs)
        
        # N is number of rollouts
        self.N = int(self.batch_size/self.max_path_length)
        # T is number of time-steps
        self.T = self.max_path_length
        logger.log("With " + str(self.N) + " rollouts per iteration")
        logger.log("With time-horizon " + str(self.T))
        
        logger.log('Running KL-TRPO')

    @overrides
    def init_opt(self):

        logger.log(str(self.env))
        ###########################
        # Symbolics for line search 
        ###########################

        # Theano vars
        X_var = ext.new_tensor(
            'X',
            ndim=2,
            dtype=theano.config.floatX,
        )   # a N*T x state_dim array 
        U_var = ext.new_tensor(
            'U',
            ndim=2,
            dtype=theano.config.floatX,
        )   # a N*T x action_dim array 
        V_var = ext.new_tensor(
            'V',
            ndim=2,
            dtype=theano.config.floatX,
        )   # corresponds to V
        param_eta = TT.scalar('eta')

        dist_info_vars = self.policy.dist_info_sym(X_var)
        dist = self.policy.distribution
        logptheta = dist.log_likelihood_sym(U_var, dist_info_vars)
        udim = self.env.action_dim
        
        #this defines the uncontrolled dynamics:
        udist_info_vars = dict(
            mean=np.zeros((1,udim)),
            log_std=np.ones((1,udim))*self.log_std_uncontrolled
        )
        logq = dist.log_likelihood_sym(U_var, udist_info_vars)
#        logq = TT.log(1/TT.sqrt(2*self.std_uncontrolled*np.pi))

        logptheta_reshaped = logptheta.reshape((self.N,self.T))
        logq_reshaped = logq.reshape((self.N,self.T))

        # we run here our TRPO variant 
        S = -(TT.sum(V_var/self.lambd + logptheta_reshaped - logq_reshaped,1))
        #S_min = TT.min(S)
        #S_sum = TT.sum(S - S_min)
        S_min = TT.mean(S)
        S_sum = TT.std(S - S_min)
        w = S - TT.mean(S) #the weights in the kl_trpo case are just the cost itself
        w = TT.reshape(w,(self.N,1))
    
        norm_entropy = -(1/TT.log(self.N)) * TT.tensordot(w, TT.log(w))
        rel_entropy = 1-norm_entropy
        input = [X_var, U_var, V_var, param_eta]

        # This is the first optimization (corresponds to the line search)
        #############
        # line search
        #############
        self.f_dual = ext.compile_function(
            inputs=input,
            outputs=[rel_entropy,w,logq,S,S_min,S_sum]
        )
            #outputs=[norm_entropy,w,logq]

        total_cost = TT.mean(TT.sum(V_var/self.lambd + logptheta_reshaped -
                                    logq_reshaped,1))
        self.f_total_cost = ext.compile_function( 
            inputs=input,         
            outputs=total_cost
        )

        # This is the second optimization (corresponds to the conj. grad)
        ############################
        # PICE gradient optimization 
        ############################
        # reshape kl and lr to be NxT matrices
        weights_var = ext.new_tensor(
            'W',
            ndim=2,
            dtype=theano.config.floatX,
        ) 
        S_min_var = ext.new_tensor(
            'S_min',
            ndim=2,
            dtype=theano.config.floatX,
        )
        S_sum_var = ext.new_tensor(
            'S_sum',
            ndim=2,
            dtype=theano.config.floatX,
        )
        old_dist_info_vars = {
            k: ext.new_tensor(
                'old_%s' % k,
                ndim=2,
                dtype=theano.config.floatX
            ) for k in dist.dist_info_keys
        }
        old_dist_info_vars_list = [old_dist_info_vars[k] for k in dist.dist_info_keys]
        state_info_vars = {
            k: ext.new_tensor(
                k,
                ndim=2,
                dtype=theano.config.floatX
            ) for k in self.policy.state_info_keys
        }
        dist_info_vars = self.policy.dist_info_sym(X_var, state_info_vars)

        #temp1 = dist.log_likelihood_sym(U_var, dist_info_vars)
        #temp2 = dist.log_likelihood_sym(U_var, old_dist_info_vars)

        kl = dist.kl_sym(old_dist_info_vars, dist_info_vars)
        
        lr = dist.likelihood_ratio_sym(U_var, old_dist_info_vars, dist_info_vars)
        lr_reshaped = lr.reshape((self.N,self.T)) 
        S = -(TT.sum(V_var/self.lambd + logptheta_reshaped - logq_reshaped,1))
        S = (S.reshape((self.N,1)) - S_min_var)/S_sum_var
        S_rep = TT.extra_ops.repeat(S,self.T,axis=1)
        
        weights_rep = TT.extra_ops.repeat(weights_var,self.T,axis=1)-TT.mean(weights_var)
        
        surr_loss = - TT.sum(lr_reshaped*S_rep) #this is the surrogate loss for the line search in the cg optimizer
        surr_loss_d = - TT.sum(lr_reshaped*weights_rep)/TT.std(weights_var) #this is the surrogate loss which is used to take the gradient

        if self.truncate_local_is_ratio is not None:
            lr = TT.minimum(self.truncate_local_is_ratio, lr)
        mean_kl = TT.mean(kl)

        input_list = [ X_var, U_var, V_var] + old_dist_info_vars_list + [weights_var] + [S_min_var] + [S_sum_var]
#        self.f_opt = ext.compile_function(
#            inputs = input_list,
#            outputs = [surr_loss, lr_reshaped]
#        )
        # plot lr ration after optimization
        self.optimizer.update_opt(
            loss=surr_loss,
            d_loss=surr_loss_d,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl",
            cg_iters=10,
        )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):

        # collect data from samples
        # get effective number of rollouts
        self.N = len(samples_data["V"])
        all_input_values = [samples_data["X"], samples_data["U"],
                            samples_data["V"]]
        self.param_eta = 0
        input_values = all_input_values + [self.param_eta]

        print("------------ arrived " + str(self.N) + " rollouts")
        # for the variant of trpo we do not need a line search
        rel_entropy, weights, logq, S, S_min, S_sum = self.f_dual(*input_values)

        ws = np.sort(np.squeeze(np.abs(weights)))[::-1]
        logger.log("Three largest weights are " + str(ws[0:3]))
        
        total_cost = self.f_total_cost(*input_values)

        #######################
        # natural gradient
        #######################
        all_input_values = tuple(ext.extract(
            samples_data,
            "X", "U", "V"
        ))
        agent_infos2 = samples_data["agent_infos2"]
        dist_info_list = [agent_infos2[k] for k in self.policy.distribution.dist_info_keys]
        S_min_v = np.ones((self.N,1))*S_min
        S_sum_v = np.ones((self.N,1))*S_sum
        all_input_values += tuple(dist_info_list) + tuple([weights]) + tuple([S_min_v]) + tuple([S_sum_v]) 

        loss_before = self.optimizer.loss(all_input_values)
        mean_kl_before = self.optimizer.constraint_val(all_input_values)

        # call optimize
        self.optimizer.optimize(all_input_values)

        mean_kl = self.optimizer.constraint_val(all_input_values)
        loss_after = self.optimizer.loss(all_input_values)

        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKLBefore', mean_kl_before)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)
        logger.record_tabular('Total cost', total_cost)
        logger.record_tabular('Total cost std', S_sum)
        logger.record_tabular('Total cost mean', S_min)
        logger.record_tabular('largest weight', ws[0])
        logger.record_tabular('smallest weight', ws[-1])
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
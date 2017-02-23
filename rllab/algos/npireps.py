from rllab.misc import ext
from rllab.misc.overrides import overrides
from rllab.algos.batch_polopt import BatchPolopt
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
import rllab.misc.logger as logger
import numpy as np
import theano.tensor as TT
import theano
from theano import printing

import matplotlib.pyplot as plt
import rllab.misc.logger as logger

class NPIREPS(BatchPolopt):
    """
    Natural PIREPS method
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            step_size=0.01,
            truncate_local_is_ratio=None,
            log_std_uncontrolled=0,
            delta = 0.1,
            kl_trpo = False,
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
        self.param_delta = delta
        self.f_dual = None
        self.f_opt = None
        self.kl_trpo = kl_trpo
        if kl_trpo:
            logger.log('Running KL-TRPO')
        else:
            logger.log('Running Natural PIREPS')
        super(NPIREPS, self).__init__(optimizer=optimizer, **kwargs)

    @overrides
    def init_opt(self):

        logger.log(str(sel.env))
        ###########################
        # Symbolics for line search 
        ###########################
        # N is number of rollouts
        N = int(self.batch_size/self.max_path_length)
        # T is number of time-steps
        T = self.max_path_length

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
        udist_info_vars = dict(
            mean=np.zeros((1,udim)),
            log_std=np.ones((1,udim))*self.log_std_uncontrolled
        )
        logq = dist.log_likelihood_sym(U_var, udist_info_vars)
#        logq = TT.log(1/TT.sqrt(2*self.std_uncontrolled*np.pi))

        logptheta_reshaped = logptheta.reshape((N,T))
        logq_reshaped = logq.reshape((N,T))

        if self.kl_trpo :
            # we run here our TRPO variant 
            S = -(TT.sum(V_var + logptheta_reshaped - logq_reshaped,1))
            w = S - TT.mean(S)
            w = TT.reshape(w,(N,1))
        else :
            # we run here natural PIREPS
            S = -(TT.sum(V_var + logptheta_reshaped - logq_reshaped,1))*(1/(1+param_eta))
            w = TT.exp(S - TT.max(S))
            Z = TT.sum(w)
            w = (w/Z).reshape((N,1))

        norm_entropy = -(1/TT.log(N)) * TT.tensordot(w, TT.log(w))
        rel_entropy = 1-norm_entropy
        input = [X_var, U_var, V_var, param_eta]

        self.f_dual = ext.compile_function(
            inputs=input,
            outputs=[norm_entropy,w,logq]
        )
            #outputs=[norm_entropy,w,logq]

        ############################
        # PICE gradient optimization 
        ############################
        # reshape kl and lr to be NxT matrices
        weights_var = ext.new_tensor(
            'W',
            ndim=2,
            dtype=theano.config.floatX,
        )   # a N*T array 

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

        if self.truncate_local_is_ratio is not None:
            lr = TT.minimum(self.truncate_local_is_ratio, lr)
        mean_kl = TT.mean(kl)

        lr_reshaped = lr.reshape((N,T)) 
        weights_rep = TT.extra_ops.repeat(weights_var,T,axis=1)
        surr_loss = - TT.sum(lr_reshaped*weights_rep)

        input_list = [ X_var, U_var] + old_dist_info_vars_list + [weights_var]
        self.f_opt = ext.compile_function(
            inputs = input_list,
            outputs = surr_loss 
        )

        self.optimizer.update_opt(
            loss=surr_loss,
            target=self.policy,
            leq_constraint=(mean_kl, self.step_size),
            inputs=input_list,
            constraint_name="mean_kl"
        )
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data):

        # collect data from samples
        all_input_values = [samples_data["X"], samples_data["U"],
                            samples_data["V"]]
        input_values = all_input_values + [self.param_eta]

        ###############################
        # line search: must be improved
        ###############################
        if not self.kl_trpo :
            outer_it = 5 
            min_log = -20
            max_log = 2 
            it = 0
            rang = np.logspace(min_log,max_log,5)
            nit = 15 
            while (it<outer_it) :
                veta = np.zeros(nit)
                vent = np.zeros(nit)
                i = 0
                while (i<nit) :
                    #print("it = " + str(it) + " i = " + str(i))
                    self.param_eta = rang[i]
                    input_values = all_input_values + [self.param_eta]
                    rel_entropy, weights, logq = self.f_dual(*input_values)
                    veta[i] = self.param_eta
                    vent[i] = rel_entropy
                    if rel_entropy > self.param_delta and i > 0:
                        #print("passed")
                        self.param_eta = rang[i-1]
                        self.final_rel_entropy = vent[i-1]
                        min_eta = rang[i-1]
                        max_eta = rang[i]
                        break
                    i += 1
                #print("it " + str(it) + " i " + str(i) + ": entropy " +
                #      str(rel_entropy))
                it += 1
                rang = np.linspace(min_eta,max_eta,nit)
                #print("new range " + str(min_eta) + "/" + str(max_eta))

            if (self.final_rel_entropy > self.param_delta) :
                logger.log("------------------ Line search for eta failed!!!")

            #logger.log("eta is " + str(self.param_eta))
            rel_entropy = self.final_rel_entropy
            #print(logq)
            #plt.semilogy(veta, vent)
            #plt.show()

        else :

            # for the variant of trpo we do not need a line search
            rel_entropy, weights, logq = self.f_dual(*input_values)

        logger.log("Entropy of weights " + str(rel_entropy))
        ws = np.sort(np.squeeze(weights))[::-1]
        print(ws[0:3])
        #######################
        # natural PICE gradient
        #######################
        all_input_values = tuple(ext.extract(
            samples_data,
            "X", "U"
        ))
        agent_infos2 = samples_data["agent_infos2"]
        dist_info_list = [agent_infos2[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(dist_info_list) + tuple([weights])
        #out1 = self.f_opt(*all_input_values)
        #print("loss before")
        #print(out1)

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
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )
